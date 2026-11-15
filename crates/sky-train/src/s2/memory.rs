//! The fast-weight memory of a model that sleeps (`docs/s6-sleep-prereg.md` §2, after the
//! gated delta networks of arXiv 2605.26099).
//!
//! One memory layer follows each state block and is shared across loops, like its block. The
//! state of a stream is one matrix `S` `[hd, hd]` per head and per layer, zero for a new stream.
//!
//! - **Read**, at every application of block `l`, on its output `x`:
//!   `q = l2norm(heads(LN(x)·Wq))`, `x ← x + merge(S·q)·Wo`. `Wo` starts at zero, so a fresh
//!   model is its base, and a zero memory adds nothing.
//! - **Write**, once per frame and per layer, from the output `x` of the block's last
//!   application (before the read), over the `n` valid tokens:
//!   `k_i = l2norm(heads(LN(x_i)·Wk))`, `v_i = heads(LN(x_i)·Wv)`, `β_i = σ(wβ·x_i)`,
//!   `α = σ(mean_i(wα·x_i) + 6)`, `S ← α·S + (1/n)·Σ_i β_i·(v_i − α·S·k_i)·k_iᵀ`.
//! - **Sleep**: the frames seen since the last sleep are replayed `N − 1` more times, in order
//!   and at full depth; each replayed frame reads the memory and writes it. A wake frame is the
//!   first pass, so `N = 1` is a model that never sleeps.
//!
//! **v2** (`docs/s6-sleep2-prereg.md` §2, §4), by configuration, v1 being the default:
//! - `memory_qk: shared` (M1): one projection `qk` gives both the queries and the keys
//!   (`q = l2norm(heads(LN(x)·Wqk))`, `k_i = l2norm(heads(LN(x_i)·Wqk))`), so that at
//!   initialisation a token retrieves what was written under similar tokens;
//! - `memory_value: frame` (M2): `v_i = heads(LN(x_i + x̄)·Wv)`, `x̄` the mean block output over
//!   the frame's valid tokens, so that a door's slot holds the colour shown next to it.

use mlx_rs::{error::Exception, ops, ops::indexing::IndexOp, stop_gradient, Array, Dtype};

use crate::{
    model::layer_norm,
    params::ParamMap,
    s2::{
        config::{MemoryQk, MemoryValue, MemoryWrite, S2Config},
        data::S2Batch,
        model::{
            cast_like, forward_prepared_mem, frame_ce, frame_ce_rows, heads, linear, merge_heads,
            prepare, state_written, to_f32, w, Init,
        },
    },
};

type R<T> = std::result::Result<T, Exception>;

/// Frames between two sleeps of a stream (`K`).
pub const SLEEP_EVERY: usize = 8;
/// Bias of the decay gate: `α = σ(6) ≈ 0.9975` per frame before training.
pub const ALPHA_BIAS: f32 = 6.0;
/// Floor of `‖x‖²` in the key and query normalization.
pub const NORM_EPS: f32 = 1e-6;
/// The projections of a memory layer (ternary in the export): `q`, `k`, `v`, `out` in v1, `qk`,
/// `v`, `out` with a shared query/key projection.
const MATRICES: [&str; 5] = ["q", "k", "qk", "v", "out"];

/// True for the projections of a memory layer.
pub fn is_ternary(name: &str) -> bool {
    name.strip_prefix("mem.")
        .and_then(|rest| rest.split_once('.'))
        .is_some_and(|(_, tail)| MATRICES.iter().any(|m| tail == format!("{m}.weight")))
}

/// The input projections of a memory layer: `q`, `k`, `v` (v1) or `qk`, `v` (shared).
fn inputs(cfg: &S2Config) -> &'static [&'static str] {
    match cfg.memory_qk {
        MemoryQk::Separate => &["q", "k", "v"],
        MemoryQk::Shared => &["qk", "v"],
    }
}

/// The projection that gives the queries of the read.
fn query_part(cfg: &S2Config) -> &'static str {
    match cfg.memory_qk {
        MemoryQk::Separate => "q",
        MemoryQk::Shared => "qk",
    }
}

/// The projection that gives the keys of the write.
fn key_part(cfg: &S2Config) -> &'static str {
    match cfg.memory_qk {
        MemoryQk::Separate => "k",
        MemoryQk::Shared => "qk",
    }
}

/// Parameters of the memory layers: `mem.{l}.{norm,q,k,v,out,alpha,beta}.weight`, with `qk` in
/// place of `q` and `k` when the projection is shared.
pub(crate) fn specs(out: &mut Vec<(String, Vec<i32>, Init)>, cfg: &S2Config) {
    let d = cfg.hidden_size as i32;
    for l in 0..cfg.state_layers {
        let p = format!("mem.{l}");
        out.push((format!("{p}.norm.weight"), vec![d], Init::Ones));
        for m in inputs(cfg) {
            out.push((
                format!("{p}.{m}.weight"),
                vec![d, d],
                Init::TruncNormal(0.02),
            ));
        }
        // Zero read-out: the model starts exactly at its base.
        out.push((format!("{p}.out.weight"), vec![d, d], Init::Zeros));
        out.push((format!("{p}.alpha.weight"), vec![d], Init::Zeros));
        out.push((format!("{p}.beta.weight"), vec![d], Init::Zeros));
    }
}

/// Scalars a memory adds to a model.
pub fn param_count(cfg: &S2Config) -> usize {
    let d = cfg.hidden_size;
    let matrices = inputs(cfg).len() + 1;
    cfg.state_layers * (matrices * d * d + 3 * d)
}

/// The memory of a batch of streams: one `[B, H, hd, hd]` array (f32) per memory layer.
#[derive(Debug, Clone)]
pub struct Memory(pub Vec<Array>);

impl Memory {
    /// The memory of `rows` new streams.
    pub fn zeros(cfg: &S2Config, rows: i32) -> R<Memory> {
        let (h, hd) = (cfg.n_heads() as i32, cfg.head_dim() as i32);
        (0..cfg.state_layers)
            .map(|_| ops::zeros::<f32>(&[rows, h, hd, hd]))
            .collect::<R<Vec<_>>>()
            .map(Memory)
    }

    /// The same values, cut from the gradient.
    pub fn detached(&self) -> R<Memory> {
        self.0
            .iter()
            .map(stop_gradient)
            .collect::<R<Vec<_>>>()
            .map(Memory)
    }

    /// Floats per stream.
    pub fn floats(cfg: &S2Config) -> usize {
        cfg.state_layers * cfg.n_heads() * cfg.head_dim() * cfg.head_dim()
    }
}

fn l2norm(x: &Array) -> R<Array> {
    let n = x
        .square()?
        .sum_axis(-1, true)?
        .add(Array::from_f32(NORM_EPS))?
        .sqrt()?;
    x.divide(&n)
}

fn name(l: usize, part: &str) -> String {
    format!("mem.{l}.{part}.weight")
}

/// A projection of the normalized hidden states, per head and in f32: `[B, H, T, hd]`.
fn project(p: &ParamMap, cfg: &S2Config, l: usize, part: &str, y: &Array) -> R<Array> {
    to_f32(heads(
        &linear(y, w(p, &name(l, part))?, None)?,
        cfg.n_heads() as i32,
    )?)
}

/// What memory layer `l` retrieves for the hidden states `x` `[B, T, d]` from `s`
/// `[B, H, hd, hd]`, before the read-out: `merge(S·q)` `[B, T, d]`, in the dtype of `x`.
pub(crate) fn retrieve(p: &ParamMap, cfg: &S2Config, l: usize, x: &Array, s: &Array) -> R<Array> {
    let y = layer_norm(x, w(p, &name(l, "norm"))?, None, cfg.norm_eps)?;
    let q = l2norm(&project(p, cfg, l, query_part(cfg), &y)?)?;
    // r_i = S · q_i, for every token at once.
    let r = ops::matmul(&q, s.transpose_axes(&[0, 1, 3, 2])?)?;
    cast_like(merge_heads(&r)?, x)
}

/// The read of memory layer `l`: `x + merge(S·q)·Wo` (`x` `[B, T, d]`, `s` `[B, H, hd, hd]`).
/// With `memory_run.read_off` the read adds nothing, as a read-out held at zero.
pub(crate) fn read(p: &ParamMap, cfg: &S2Config, l: usize, x: &Array, s: &Array) -> R<Array> {
    if cfg.memory_run.read_off {
        return Ok(x.clone());
    }
    let r = retrieve(p, cfg, l, x, s)?;
    x.add(linear(&r, w(p, &name(l, "out"))?, None)?)
}

/// The keys memory layer `l` writes under for the block output `x` `[B, T, d]`:
/// `l2norm(heads(LN(x)·Wk))` `[B, H, T, hd]` (f32), `Wqk` when the projection is shared.
pub(crate) fn keys(p: &ParamMap, cfg: &S2Config, l: usize, x: &Array) -> R<Array> {
    let y = layer_norm(x, w(p, &name(l, "norm"))?, None, cfg.norm_eps)?;
    l2norm(&project(p, cfg, l, key_part(cfg), &y)?)
}

/// The write of memory layer `l` from a frame's block output `x` `[B, T, d]` (`valid` `[B, T]`):
/// one gated delta-rule step over the valid tokens.
pub(crate) fn write(
    p: &ParamMap,
    cfg: &S2Config,
    l: usize,
    x: &Array,
    valid: &Array,
    s: &Array,
) -> R<Array> {
    let (b, t) = (x.dim(0), x.dim(1));
    let norm = w(p, &name(l, "norm"))?;
    let y = layer_norm(x, norm, None, cfg.norm_eps)?;
    let k = l2norm(&project(p, cfg, l, key_part(cfg), &y)?)?;
    let on = valid.as_dtype(Dtype::Float32)?;
    let n = ops::maximum(on.sum_axis(-1, true)?, Array::from_f32(1.0))?;
    // W3: the keys centred on the frame's mean key (over its valid tokens), of unit norm again.
    let k = if cfg.memory_write == MemoryWrite::Centred {
        let on_k = on.reshape(&[b, 1, t, 1])?;
        let mean = k
            .multiply(&on_k)?
            .sum_axis(2, true)?
            .divide(n.reshape(&[b, 1, 1, 1])?)?;
        l2norm(&k.subtract(&mean)?)?
    } else {
        k
    };
    let v = match cfg.memory_value {
        MemoryValue::Token => project(p, cfg, l, "v", &y)?,
        MemoryValue::Frame => {
            // x̄: the mean block output over the frame's valid tokens, added to every token.
            let on_x = cast_like(on.reshape(&[b, t, 1])?, x)?;
            let n_x = cast_like(n.reshape(&[b, 1, 1])?, x)?;
            let mean = x.multiply(&on_x)?.sum_axis(1, true)?.divide(&n_x)?;
            let yv = layer_norm(&x.add(&mean)?, norm, None, cfg.norm_eps)?;
            project(p, cfg, l, "v", &yv)?
        }
    };
    let xf = to_f32(x.clone())?;
    let gate = |part: &str| -> R<Array> {
        xf.multiply(to_f32(w(p, &name(l, part))?.clone())?)?
            .sum_axis(-1, None)
    };
    if cfg.memory_write == MemoryWrite::Exact {
        // W2: S ← S + (V − S Kᵀ)(K Kᵀ + λI)⁻¹ K over the valid tokens (the padding has zero
        // keys and values, so it adds nothing). With the rows of `err` the (v_i − S k_i)ᵀ:
        // the update is (G⁻¹ err)ᵀ K, G = K Kᵀ + λI being symmetric.
        let on_k = on.reshape(&[b, 1, t, 1])?;
        let km = k.multiply(&on_k)?;
        let err = v
            .multiply(&on_k)?
            .subtract(ops::matmul(&km, s.transpose_axes(&[0, 1, 3, 2])?)?)?;
        let eye = ops::eye::<f32>(t, None, None)?.multiply(Array::from_f32(EXACT_LAMBDA))?;
        let g = ops::matmul(&km, km.transpose_axes(&[0, 1, 3, 2])?)?.add(&eye)?;
        let sol = spd_solve(&g, &err)?;
        return s.add(ops::matmul(sol.transpose_axes(&[0, 1, 3, 2])?, &km)?);
    }
    let beta = match cfg.memory_write {
        // W1: a fresh model (wβ = 0) takes full steps.
        MemoryWrite::Delta => ops::sigmoid(gate("beta")?)?.multiply(Array::from_f32(2.0))?,
        _ => ops::sigmoid(gate("beta")?)?,
    };
    let alpha = ops::sigmoid(
        gate("alpha")?
            .multiply(&on)?
            .sum_axis(-1, true)?
            .divide(&n)?
            .add(Array::from_f32(ALPHA_BIAS))?,
    )?
    .reshape(&[b, 1, 1, 1])?;
    let kept = s.multiply(&alpha)?;
    if cfg.memory_write == MemoryWrite::Delta {
        // W1: token by token, S ← S + β_i (v_i − S k_i) k_iᵀ, the padding skipped.
        let w = beta.multiply(&on)?;
        let mut st = kept;
        for i in 0..t {
            let at = Array::from_slice(&[i], &[1]);
            let ki = k.take_axis(&at, 2)?;
            let vi = v.take_axis(&at, 2)?;
            let wi = w.take_axis(&at, 1)?.reshape(&[b, 1, 1, 1])?;
            let erri = vi.subtract(ops::matmul(&ki, st.transpose_axes(&[0, 1, 3, 2])?)?)?;
            let step = ops::matmul(erri.multiply(&wi)?.transpose_axes(&[0, 1, 3, 2])?, &ki)?;
            st = st.add(step)?;
        }
        return Ok(st);
    }
    // err_i = v_i − α·S·k_i, weighted by β_i / n over the valid tokens.
    let err = v.subtract(ops::matmul(&k, kept.transpose_axes(&[0, 1, 3, 2])?)?)?;
    let weight = beta.multiply(&on)?.divide(&n)?.reshape(&[b, 1, t, 1])?;
    let delta = ops::matmul(err.multiply(&weight)?.transpose_axes(&[0, 1, 3, 2])?, &k)?;
    kept.add(delta)
}

/// λ of the exact write (W2).
pub const EXACT_LAMBDA: f32 = 1e-3;

/// `G⁻¹ R` for symmetric positive definite `G` `[.., n, n]` and `R` `[.., n, m]`, by Gauss–Jordan
/// elimination without pivoting, from elementwise ops and matmuls only: it is differentiable and
/// runs on every device (MLX's `linalg::inv` and `linalg::solve` run on the CPU only and have no
/// gradient).
pub(crate) fn spd_solve(g: &Array, r: &Array) -> R<Array> {
    let n = g.dim(-1);
    let (mut g, mut r) = (g.clone(), r.clone());
    for j in 0..n {
        let at = Array::from_slice(&[j], &[1]);
        let row_g = g.take_axis(&at, -2)?; // [.., 1, n]
        let piv = row_g.take_axis(&at, -1)?; // [.., 1, 1]
        let row_g = row_g.divide(&piv)?;
        let row_r = r.take_axis(&at, -2)?.divide(&piv)?;
        let e: Vec<f32> = (0..n).map(|i| (i == j) as u8 as f32).collect();
        let e = Array::from_slice(&e, &[n, 1]);
        // Column j minus e_j: row j becomes the scaled pivot row, the others are eliminated.
        let col = g.take_axis(&at, -1)?.subtract(&e)?; // [.., n, 1]
        g = g.subtract(ops::matmul(&col, &row_g)?)?;
        r = r.subtract(ops::matmul(&col, &row_r)?)?;
    }
    Ok(r)
}

/// The memory after the writes of one frame (`written` from `state_run_mem`): a block the frame
/// did not run leaves its layer as it was.
pub(crate) fn after_writes(
    p: &ParamMap,
    cfg: &S2Config,
    written: &[Option<Array>],
    valid: &Array,
    mem: &Memory,
) -> R<Memory> {
    mem.0
        .iter()
        .enumerate()
        .map(|(l, s)| match &written[l] {
            Some(x) => write(p, cfg, l, x, valid, s),
            None => Ok(s.clone()),
        })
        .collect::<R<Vec<_>>>()
        .map(Memory)
}

/// One wake frame over the first `n_exits` exits with prepared parameters: the logits
/// `[E, B, K]` and the memory after the frame's write.
pub fn wake(
    p: &ParamMap,
    cfg: &S2Config,
    b: &S2Batch,
    n_exits: usize,
    mem: &Memory,
) -> R<(Array, Memory)> {
    let (logits, _, written) = forward_prepared_mem(p, cfg, b, n_exits, None, Some(&mem.0))?;
    Ok((logits, after_writes(p, cfg, &written, &b.valid, mem)?))
}

/// One replayed frame of a sleep pass: no question is asked, only the memory changes.
pub fn replay(p: &ParamMap, cfg: &S2Config, b: &S2Batch, mem: &Memory) -> R<Memory> {
    let written = state_written(p, cfg, b, &mem.0)?;
    after_writes(p, cfg, &written, &b.valid, mem)
}

/// How a model sleeps.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Sleep {
    /// Passes over a chunk, the wake pass included (`N`; 1 = no sleep).
    pub passes: usize,
    /// Frames between two sleeps (`K`).
    pub every: usize,
    /// Training only. `false`: the gradient flows through the last pass of a sleep and stops at
    /// the memory that enters it. `true`: it flows through every pass.
    pub full: bool,
}

impl Sleep {
    /// No sleep: wake frames only.
    pub const AWAKE: Sleep = Sleep {
        passes: 1,
        every: SLEEP_EVERY,
        full: true,
    };

    pub fn passes(n: usize) -> Sleep {
        Sleep {
            passes: n.max(1),
            every: SLEEP_EVERY,
            full: false,
        }
    }
}

/// Sleeps on `chunk`, the frames seen since the last sleep: `passes − 1` replays in order.
pub fn sleep(
    p: &ParamMap,
    cfg: &S2Config,
    chunk: &[&S2Batch],
    mem: Memory,
    how: Sleep,
) -> R<Memory> {
    let mut m = mem;
    for pass in 1..how.passes {
        if !how.full && pass + 1 == how.passes {
            m = m.detached()?;
        }
        for b in chunk {
            m = replay(p, cfg, b, &m)?;
        }
    }
    Ok(m)
}

/// Sequence loss of a model with or without memory: the mean over frames of the deep-supervision
/// cross-entropy of every wake frame. `frames` are consecutive frames of the same rows, and
/// `pauses[t]` says the game paused before frame `t`.
///
/// With `cfg.memory`, each row starts from a zero memory, every frame reads it and writes it,
/// and the model sleeps (`how`) every `how.every` frames and at each pause. Without it, every
/// frame is answered alone: the control arm on the same batches.
pub fn seq_loss(
    p: &ParamMap,
    cfg: &S2Config,
    frames: &[S2Batch],
    pauses: &[bool],
    wts: &Array,
    qat: bool,
    how: Sleep,
) -> R<Array> {
    Ok(seq_frames(p, cfg, frames, pauses, wts, qat, how)?.0)
}

/// [`seq_loss`] and, for the training log of `docs/s6-sleep2-prereg.md` §2 (M5), the
/// cross-entropy of the ask frames apart: `ask` `[K, B]` (1 on an ask frame of `recall`).
/// Returns `[loss, Σ deep-supervision CE over the asks, Σ last-exit CE over the asks, asks]`;
/// only the loss carries a gradient.
#[allow(clippy::too_many_arguments)]
pub fn seq_loss_asks(
    p: &ParamMap,
    cfg: &S2Config,
    frames: &[S2Batch],
    pauses: &[bool],
    wts: &Array,
    qat: bool,
    how: Sleep,
    ask: &Array,
) -> R<Vec<Array>> {
    let (loss, logits) = seq_frames(p, cfg, frames, pauses, wts, qat, how)?;
    let e = cfg.n_exits() as i32;
    let one_hot: Vec<f32> = (0..e).map(|i| (i == e - 1) as u8 as f32).collect();
    let last = Array::from_slice(&one_hot, &[e]);
    let (mut deep, mut tail) = (Array::from_f32(0.0), Array::from_f32(0.0));
    for (t, (l, b)) in logits.iter().zip(frames).enumerate() {
        let on = ask.index(t as i32);
        let l = stop_gradient(l)?;
        deep = deep.add(frame_ce_rows(&l, b, wts)?.multiply(&on)?.sum(None)?)?;
        tail = tail.add(frame_ce_rows(&l, b, &last)?.multiply(&on)?.sum(None)?)?;
    }
    Ok(vec![loss, deep, tail, ask.sum(None)?])
}

/// The loss of [`seq_loss`] and the logits of every frame.
fn seq_frames(
    p: &ParamMap,
    cfg: &S2Config,
    frames: &[S2Batch],
    pauses: &[bool],
    wts: &Array,
    qat: bool,
    how: Sleep,
) -> R<(Array, Vec<Array>)> {
    let p = &*prepare(p, cfg, qat)?;
    let n_exits = cfg.n_exits();
    let mut total: Option<Array> = None;
    let mut seen = Vec::with_capacity(frames.len());
    let mut add = |l: Array| -> R<()> {
        total = Some(match total.take() {
            Some(x) => x.add(&l)?,
            None => l,
        });
        Ok(())
    };
    if cfg.memory {
        let mut mem = Memory::zeros(cfg, frames[0].ids.dim(0))?;
        let mut chunk: Vec<&S2Batch> = Vec::new();
        for (t, b) in frames.iter().enumerate() {
            let pause = pauses.get(t).copied().unwrap_or(false);
            if !chunk.is_empty() && (chunk.len() >= how.every || pause) {
                mem = sleep(p, cfg, &chunk, mem, how)?;
                chunk.clear();
            }
            let (logits, next) = wake(p, cfg, b, n_exits, &mem)?;
            add(frame_ce(&logits, b, wts)?)?;
            seen.push(logits);
            mem = next;
            chunk.push(b);
        }
    } else {
        for b in frames {
            let (logits, _, _) = forward_prepared_mem(p, cfg, b, n_exits, None, None)?;
            add(frame_ce(&logits, b, wts)?)?;
            seen.push(logits);
        }
    }
    let loss = total
        .expect("at least one frame")
        .divide(Array::from_f32(frames.len() as f32))?;
    Ok((loss, seen))
}

/// Check C1 of `docs/s6-sleep2-prereg.md` §3, the retrieval algebra: one reading.
#[derive(Debug, Clone, serde::Serialize)]
pub struct AlgebraRow {
    /// The write rule (`average` for v1's).
    pub rule: &'static str,
    /// Passes over each chunk of frames, the wake pass included (`N`; sleeps every
    /// [`SLEEP_EVERY`] frames, as the model does).
    pub passes: usize,
    pub form: &'static str,
    /// Idle frames between the show frame and the ask frame.
    pub idle: usize,
    /// `r · x_c` at the ask's door token for red, green, blue, yellow.
    pub colours: [f32; 4],
    /// The colour of the argmax (0 = red, the colour shown).
    pub argmax: usize,
    /// Red's projection minus the best wrong colour's (> 0: decoded right).
    pub margin: f32,
}

/// Check C1: synthetic hidden states at `d = 96` (one random unit direction per word, orthogonal
/// to a shared one, mixed `√0.6 · u_w + √0.4 · c` so that two words have a cosine of 0.4 on
/// average), one memory layer of one head with `Wqk = Wv = I` in f32, and the real
/// [`write`] and [`retrieve`]. A show frame `show a colour red ask none noise 46`, then `idle`
/// idle frames `show none colour none ask none noise n`, then the ask `show none colour none ask
/// a noise n`. Returns the measured mean cosine between distinct words and one row per form and
/// number of idle frames.
pub fn retrieval_algebra(idle: &[usize], seed: u64) -> R<(f32, Vec<AlgebraRow>)> {
    algebra(&[MemoryWrite::Average], &[1], idle, seed)
}

/// The sweep of amendment 1 of `docs/s6-sleep2-prereg.md` on the setting of C1: every write rule
/// of `rules` at every number of passes of `passes`. The frames before the ask are written as
/// a wake pass, and every [`SLEEP_EVERY`] frames the chunk is replayed `N − 1` more times with
/// the same rule (as [`sleep`] does; with synthetic states a replay rewrites the same keys and
/// values). The word vectors and the noise numbers are the same for every reading.
pub fn algebra(
    rules: &[MemoryWrite],
    passes: &[usize],
    idle: &[usize],
    seed: u64,
) -> R<(f32, Vec<AlgebraRow>)> {
    const D: usize = 96;
    let mut rng = crate::rng::Rng::new(seed);
    let gauss = |rng: &mut crate::rng::Rng| -> Vec<f32> {
        (0..D)
            .map(|_| {
                // Box–Muller.
                let (u1, u2) = (rng.f64().max(1e-12), rng.f64());
                ((-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()) as f32
            })
            .collect()
    };
    let unit = |v: Vec<f32>| -> Vec<f32> {
        let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        v.into_iter().map(|x| x / n).collect()
    };
    let shared = unit(gauss(&mut rng));
    let mut words: Vec<String> = [
        "show", "colour", "ask", "noise", "none", "a", "b", "c", "red", "green", "blue", "yellow",
    ]
    .iter()
    .map(|w| w.to_string())
    .collect();
    words.extend((0..100).map(|n| n.to_string()));
    let mut vec_of = std::collections::HashMap::new();
    for word in &words {
        let mut u = gauss(&mut rng);
        let along: f32 = u.iter().zip(&shared).map(|(a, b)| a * b).sum();
        u.iter_mut().zip(&shared).for_each(|(a, b)| *a -= along * b);
        let u = unit(u);
        let x: Vec<f32> = u
            .iter()
            .zip(&shared)
            .map(|(a, b)| 0.6f32.sqrt() * a + 0.4f32.sqrt() * b)
            .collect();
        vec_of.insert(word.clone(), x);
    }
    let (mut cos, mut pairs) = (0.0f64, 0usize);
    for i in 0..words.len() {
        for j in i + 1..words.len() {
            let (a, b) = (&vec_of[&words[i]], &vec_of[&words[j]]);
            let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
            let na = a.iter().map(|x| x * x).sum::<f32>().sqrt();
            let nb = b.iter().map(|x| x * x).sum::<f32>().sqrt();
            cos += (dot / (na * nb)) as f64;
            pairs += 1;
        }
    }
    let frame = |text: &str| -> Array {
        let flat: Vec<f32> = text
            .split(' ')
            .flat_map(|w| vec_of[w].iter().copied())
            .collect();
        Array::from_slice(&flat, &[1, (flat.len() / D) as i32, D as i32])
    };
    let mut cfg = S2Config::tiny(300);
    (cfg.hidden_size, cfg.head_count) = (D, Some(1));
    (cfg.state_layers, cfg.exits) = (1, vec![1]);
    (cfg.memory, cfg.memory_qk) = (true, MemoryQk::Shared);
    let mut p = ParamMap::new();
    let eye = ops::eye::<f32>(D as i32, None, None)?;
    p.insert(name(0, "norm").into(), ops::ones::<f32>(&[D as i32])?);
    p.insert(name(0, "qk").into(), eye.clone());
    p.insert(name(0, "v").into(), eye);
    for gate in ["alpha", "beta"] {
        p.insert(name(0, gate).into(), ops::zeros::<f32>(&[D as i32])?);
    }
    let noise: Vec<usize> = (0..=idle.iter().copied().max().unwrap_or(0) + 1)
        .map(|_| rng.below(100))
        .collect();
    let show = frame("show a colour red ask none noise 46");
    let valid = Array::from_slice(&[true; 8], &[1, 8]);
    let colours: Vec<&Vec<f32>> = ["red", "green", "blue", "yellow"]
        .iter()
        .map(|c| &vec_of[*c])
        .collect();
    let mut rows = Vec::new();
    for &rule in rules {
        cfg.memory_write = rule;
        for &n in passes {
            for form in [MemoryValue::Frame, MemoryValue::Token] {
                cfg.memory_value = form;
                for &n_idle in idle {
                    let mut before = vec![show.clone()];
                    before.extend(
                        noise[..n_idle]
                            .iter()
                            .map(|x| frame(&format!("show none colour none ask none noise {x}"))),
                    );
                    let mut st = ops::zeros::<f32>(&[1, 1, D as i32, D as i32])?;
                    let mut chunk: Vec<&Array> = Vec::new();
                    let sleep_on = |st: &mut Array, chunk: &mut Vec<&Array>| -> R<()> {
                        if chunk.len() >= SLEEP_EVERY {
                            for _ in 1..n {
                                for f in chunk.iter() {
                                    *st = write(&p, &cfg, 0, f, &valid, st)?;
                                }
                            }
                            chunk.clear();
                        }
                        Ok(())
                    };
                    for f in &before {
                        sleep_on(&mut st, &mut chunk)?;
                        st = write(&p, &cfg, 0, f, &valid, &st)?;
                        chunk.push(f);
                    }
                    sleep_on(&mut st, &mut chunk)?;
                    let ask = frame(&format!(
                        "show none colour none ask a noise {}",
                        noise[n_idle]
                    ));
                    // The ask's door token is the sixth word.
                    let r = crate::model::host_f32(&retrieve(&p, &cfg, 0, &ask, &st)?)?;
                    let r = &r[5 * D..6 * D];
                    let mut proj = [0f32; 4];
                    for (c, dir) in colours.iter().enumerate() {
                        proj[c] = r.iter().zip(dir.iter()).map(|(a, b)| a * b).sum();
                    }
                    let argmax = (0..4).fold(0, |b, i| if proj[i] > proj[b] { i } else { b });
                    let wrong = proj[1..].iter().copied().fold(f32::MIN, f32::max);
                    rows.push(AlgebraRow {
                        rule: rule.name(),
                        passes: n,
                        form: form.name(),
                        idle: n_idle,
                        colours: proj,
                        argmax,
                        margin: proj[0] - wrong,
                    });
                }
            }
        }
    }
    Ok(((cos / pairs as f64) as f32, rows))
}

// ------------------------------------------------- amendment 2: D1, D2, the contrastive C1

/// Width of the synthetic states of C1.
const SYN_D: usize = 96;
/// Noise numbers drawn per draw (one per frame up to 32 idle frames and the ask; as C1 draws).
const SYN_NOISE: usize = 34;
/// The four colours of `recall`, in the order of C1's projections.
pub const SYN_COLOURS: [&str; 4] = ["red", "green", "blue", "yellow"];

/// One draw of C1's synthetic setting: the word vectors and the noise numbers of `seed`,
/// drawn in the same order as [`algebra`] (seed 0 is the draw of C1 and of amendment 1's sweep).
pub struct Synth {
    vec_of: std::collections::HashMap<String, Vec<f32>>,
    noise: Vec<usize>,
}

impl Synth {
    pub fn new(seed: u64) -> Synth {
        let mut rng = crate::rng::Rng::new(seed);
        let gauss = |rng: &mut crate::rng::Rng| -> Vec<f32> {
            (0..SYN_D)
                .map(|_| {
                    let (u1, u2) = (rng.f64().max(1e-12), rng.f64());
                    ((-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()) as f32
                })
                .collect()
        };
        let unit = |v: Vec<f32>| -> Vec<f32> {
            let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            v.into_iter().map(|x| x / n).collect()
        };
        let shared = unit(gauss(&mut rng));
        let mut words: Vec<String> = [
            "show", "colour", "ask", "noise", "none", "a", "b", "c", "red", "green", "blue",
            "yellow",
        ]
        .iter()
        .map(|w| w.to_string())
        .collect();
        words.extend((0..100).map(|n| n.to_string()));
        let mut vec_of = std::collections::HashMap::new();
        for word in &words {
            let mut u = gauss(&mut rng);
            let along: f32 = u.iter().zip(&shared).map(|(a, b)| a * b).sum();
            u.iter_mut().zip(&shared).for_each(|(a, b)| *a -= along * b);
            let u = unit(u);
            let x: Vec<f32> = u
                .iter()
                .zip(&shared)
                .map(|(a, b)| 0.6f32.sqrt() * a + 0.4f32.sqrt() * b)
                .collect();
            vec_of.insert(word.clone(), x);
        }
        let noise = (0..SYN_NOISE).map(|_| rng.below(100)).collect();
        Synth { vec_of, noise }
    }

    fn frame(&self, text: &str) -> Array {
        let flat: Vec<f32> = text
            .split(' ')
            .flat_map(|w| self.vec_of[w].iter().copied())
            .collect();
        Array::from_slice(&flat, &[1, (flat.len() / SYN_D) as i32, SYN_D as i32])
    }

    fn show(&self, colour: usize) -> Array {
        self.frame(&format!(
            "show a colour {} ask none noise 46",
            SYN_COLOURS[colour]
        ))
    }

    /// The four colour word vectors.
    pub fn colours(&self) -> [&[f32]; 4] {
        SYN_COLOURS.map(|c| self.vec_of[c].as_slice())
    }
}

/// C1's memory layer: one head at d = 96, `Wqk = Wv = I` in f32, gates at zero.
pub fn synth_layer(rule: MemoryWrite, form: MemoryValue) -> R<(S2Config, ParamMap)> {
    let mut cfg = S2Config::tiny(300);
    (cfg.hidden_size, cfg.head_count) = (SYN_D, Some(1));
    (cfg.state_layers, cfg.exits) = (1, vec![1]);
    (cfg.memory, cfg.memory_qk) = (true, MemoryQk::Shared);
    (cfg.memory_value, cfg.memory_write) = (form, rule);
    let mut p = ParamMap::new();
    let eye = ops::eye::<f32>(SYN_D as i32, None, None)?;
    p.insert(name(0, "norm").into(), ops::ones::<f32>(&[SYN_D as i32])?);
    p.insert(name(0, "qk").into(), eye.clone());
    p.insert(name(0, "v").into(), eye);
    for gate in ["alpha", "beta"] {
        p.insert(name(0, gate).into(), ops::zeros::<f32>(&[SYN_D as i32])?);
    }
    Ok((cfg, p))
}

/// `r` at the ask's door token after the show frame of `colour`, `idle` idle frames and a sleep
/// every [`SLEEP_EVERY`] frames with `passes` passes, as [`algebra`] plays it.
pub fn synth_read(
    syn: &Synth,
    cfg: &S2Config,
    p: &ParamMap,
    colour: usize,
    idle: usize,
    passes: usize,
) -> R<Vec<f32>> {
    let valid = Array::from_slice(&[true; 8], &[1, 8]);
    let mut before = vec![syn.show(colour)];
    before.extend(
        syn.noise[..idle]
            .iter()
            .map(|x| syn.frame(&format!("show none colour none ask none noise {x}"))),
    );
    let mut st = ops::zeros::<f32>(&[1, 1, SYN_D as i32, SYN_D as i32])?;
    let mut chunk: Vec<&Array> = Vec::new();
    for f in &before {
        if chunk.len() >= SLEEP_EVERY {
            for _ in 1..passes {
                for c in &chunk {
                    st = write(p, cfg, 0, c, &valid, &st)?;
                }
            }
            chunk.clear();
        }
        st = write(p, cfg, 0, f, &valid, &st)?;
        chunk.push(f);
    }
    if chunk.len() >= SLEEP_EVERY {
        for _ in 1..passes {
            for c in &chunk {
                st = write(p, cfg, 0, c, &valid, &st)?;
            }
        }
    }
    let ask = syn.frame(&format!(
        "show none colour none ask a noise {}",
        syn.noise[idle]
    ));
    let r = crate::model::host_f32(&retrieve(p, cfg, 0, &ask, &st)?)?;
    Ok(r[5 * SYN_D..6 * SYN_D].to_vec())
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn cos(a: &[f32], b: &[f32]) -> f32 {
    dot(a, b) / (dot(a, a).sqrt() * dot(b, b).sqrt()).max(1e-12)
}

fn argmax4(v: [f32; 4]) -> usize {
    (0..4).fold(0, |b, i| if v[i] > v[b] { i } else { b })
}

/// C1's raw decoding: the colour of the largest projection on the four colour vectors.
pub fn decode_raw(r: &[f32], colours: &[&[f32]; 4]) -> usize {
    argmax4(colours.map(|c| dot(r, c)))
}

/// Amendment 2's contrastive decoding: the readings of one episode played with each of the four
/// colours, minus their mean, each matched to the nearest colour direction (cosine). True when
/// the four are matched to their own colour.
pub fn decode_contrastive(reads: &[Vec<f32>; 4], colours: &[&[f32]; 4]) -> bool {
    let mean: Vec<f32> = (0..SYN_D)
        .map(|j| reads.iter().map(|r| r[j]).sum::<f32>() / 4.0)
        .collect();
    (0..4).all(|c| {
        let z: Vec<f32> = reads[c].iter().zip(&mean).map(|(a, m)| a - m).collect();
        argmax4(colours.map(|k| cos(&z, k))) == c
    })
}

/// D1 and D2 of amendment 2, with W2 under form `frame`, after the show frame alone.
#[derive(Debug, Clone, serde::Serialize)]
pub struct Diagnostic {
    pub draws: usize,
    /// Largest |S·k_door − v_door| over every case.
    pub max_abs_diff: f32,
    /// The largest |v_door| (for scale).
    pub max_abs_value: f32,
    /// D1: cases (draw × colour shown) whose stored value decodes right under C1's raw projection.
    pub d1_right: usize,
    pub d1_cases: usize,
    /// D2: draws whose four stored values all decode right contrastively.
    pub d2_right: usize,
}

pub fn diagnostic(seeds: std::ops::Range<u64>) -> R<Diagnostic> {
    let (cfg, p) = synth_layer(MemoryWrite::Exact, MemoryValue::Frame)?;
    let valid = Array::from_slice(&[true; 8], &[1, 8]);
    let mut out = Diagnostic {
        draws: seeds.clone().count(),
        max_abs_diff: 0.0,
        max_abs_value: 0.0,
        d1_right: 0,
        d1_cases: 0,
        d2_right: 0,
    };
    for seed in seeds {
        let syn = Synth::new(seed);
        let colours = syn.colours();
        let mut stored: Vec<Vec<f32>> = Vec::new();
        for c in 0..4 {
            let x = syn.show(c);
            let st = write(
                &p,
                &cfg,
                0,
                &x,
                &valid,
                &ops::zeros::<f32>(&[1, 1, 96, 96])?,
            )?;
            // The door is the second word of the show frame: its query is its key.
            let r = crate::model::host_f32(&retrieve(&p, &cfg, 0, &x, &st)?)?;
            let r = &r[SYN_D..2 * SYN_D];
            // The stored value of form `frame`: LN(x_door + x̄)·I.
            let mean = x.mean_axis(1, true)?;
            let v = layer_norm(&x.add(&mean)?, w(&p, &name(0, "norm"))?, None, cfg.norm_eps)?;
            let v = crate::model::host_f32(&v)?[SYN_D..2 * SYN_D].to_vec();
            for (a, b) in r.iter().zip(&v) {
                out.max_abs_diff = out.max_abs_diff.max((a - b).abs());
                out.max_abs_value = out.max_abs_value.max(b.abs());
            }
            out.d1_cases += 1;
            out.d1_right += (decode_raw(&v, &colours) == c) as usize;
            stored.push(v);
        }
        let stored: [Vec<f32>; 4] = stored.try_into().expect("four colours");
        out.d2_right += decode_contrastive(&stored, &colours) as usize;
    }
    Ok(out)
}

/// One cell of the contrastive sweep: the draws whose four colours were all matched.
#[derive(Debug, Clone, serde::Serialize)]
pub struct ContrastRow {
    pub rule: &'static str,
    pub passes: usize,
    pub form: &'static str,
    pub idle: usize,
    pub right_draws: usize,
    pub draws: usize,
}

/// Amendment 2's sweep: amendment 1's, with C1 read contrastively over the draws `seeds`.
pub fn contrastive_sweep(
    rules: &[MemoryWrite],
    passes: &[usize],
    idle: &[usize],
    seeds: std::ops::Range<u64>,
) -> R<Vec<ContrastRow>> {
    let draws: Vec<Synth> = seeds.map(Synth::new).collect();
    let mut rows = Vec::new();
    for &rule in rules {
        for &n in passes {
            for form in [MemoryValue::Frame, MemoryValue::Token] {
                let (cfg, p) = synth_layer(rule, form)?;
                for &d in idle {
                    let mut right = 0;
                    for syn in &draws {
                        let reads: Vec<Vec<f32>> = (0..4)
                            .map(|c| synth_read(syn, &cfg, &p, c, d, n))
                            .collect::<R<_>>()?;
                        let reads: [Vec<f32>; 4] = reads.try_into().expect("four colours");
                        right += decode_contrastive(&reads, &syn.colours()) as usize;
                    }
                    rows.push(ContrastRow {
                        rule: rule.name(),
                        passes: n,
                        form: form.name(),
                        idle: d,
                        right_draws: right,
                        draws: draws.len(),
                    });
                }
            }
        }
    }
    Ok(rows)
}

#[cfg(test)]
mod tests {
    use std::{collections::HashMap, rc::Rc};

    use mlx_rs::transforms::keyed_value_and_grad;

    use super::*;
    use crate::{
        model::host_f32,
        s2::{
            config::{MemoryQk, MemoryValue, MemoryWrite},
            data::{encode_record_cfg, make_batch, tests::s2_tok},
            golden::golden_params,
            model::{
                exit_weight_vec, expected_count, forward, init, logits_per_row,
                tests::{item, ladder, rec},
                S2Parts,
            },
        },
        test_util,
    };

    fn sleepy() -> S2Config {
        let mut cfg = ladder("s6-nano");
        cfg.memory = true;
        cfg.vocab_size = 262;
        cfg.validate().unwrap();
        cfg
    }

    fn frames(cfg: &S2Config, states: &[&str]) -> Vec<S2Batch> {
        let (_d, tok) = s2_tok();
        let opts = ["wait", "red", "green"];
        let mut statics = Vec::new();
        let mut out = Vec::new();
        for s in states {
            let (st, state) =
                encode_record_cfg(&tok, &rec("choice", "which key?", &opts, s), cfg).unwrap();
            if statics.is_empty() {
                statics.push(st);
            }
            let (a, b) = (item(0, &state, 3, 0), item(0, &state, 3, 0));
            out.push(make_batch(&statics, &[&a, &b], 1));
        }
        out
    }

    fn total(a: &Array) -> f32 {
        host_f32(a).unwrap().iter().map(|x| x.abs()).sum()
    }

    /// The v2 memory of `docs/s6-sleep2-prereg.md` §4 on `cfg`.
    fn v2(mut cfg: S2Config, value: MemoryValue) -> S2Config {
        cfg.memory = true;
        cfg.memory_qk = MemoryQk::Shared;
        cfg.memory_value = value;
        cfg.validate().unwrap();
        cfg
    }

    /// The step-0 gradient of the sequence loss of `f` (`qat` as in the pilot).
    fn grads_of(cfg: &S2Config, p: &ParamMap, f: &[S2Batch]) -> HashMap<Rc<str>, Array> {
        let wv = exit_weight_vec(cfg, None);
        let wts = Array::from_slice(&wv, &[wv.len() as i32]);
        let pauses = vec![false; f.len()];
        let g = |q: HashMap<Rc<str>, Array>, _: ()| -> R<Vec<Array>> {
            Ok(vec![seq_loss(
                &q,
                cfg,
                f,
                &pauses,
                &wts,
                true,
                Sleep::AWAKE,
            )?])
        };
        let (_, grads) = keyed_value_and_grad(g)(p.clone(), ()).unwrap();
        grads
    }

    #[test]
    fn the_v2_memory_has_the_parameters_of_the_preregistration() {
        // docs/s6-sleep2-prereg.md §4: 3d² + 3d per memory layer, 27,936 on the pico pilot
        // (473,281 for the whole model) and 125,280 on s6-nano.
        let nano = v2(sleepy(), MemoryValue::Frame);
        assert_eq!(param_count(&nano), 125_280);
        let mut pico = ladder("s6-pico");
        // The pilot's base (runs/s6lr/softmax-adamw-x2) is s6-pico without Parallax.
        pico.parallax = false;
        let pico = v2(pico, MemoryValue::Token);
        assert_eq!(param_count(&pico), 27_936);
        assert_eq!(expected_count(&pico, S2Parts::DECISION), 473_281);
        let names: Vec<String> = crate::s2::model::param_shapes(&pico, S2Parts::DECISION)
            .into_iter()
            .map(|(n, _)| n)
            .filter(|n| n.starts_with("mem."))
            .collect();
        assert_eq!(
            names,
            ["norm", "qk", "v", "out", "alpha", "beta"].map(|m| format!("mem.0.{m}.weight"))
        );
        assert!(is_ternary("mem.0.qk.weight") && crate::s2::model::is_ternary("mem.0.qk.weight"));
        // The state per stream does not change.
        assert_eq!(Memory::floats(&nano), 20_736);
        assert_eq!(Memory::floats(&pico), 9_216);
        // The v2 keys are written only when set: a v1 config serializes as before.
        let j = serde_json::to_value(sleepy()).unwrap();
        assert!(j.get("memory_qk").is_none() && j.get("memory_value").is_none());
        let j = serde_json::to_value(&nano).unwrap();
        assert_eq!(
            (&j["memory_qk"], &j["memory_value"]),
            (&"shared".into(), &"frame".into())
        );
        let back: S2Config = serde_json::from_value(j).unwrap();
        assert_eq!(back, nano);
        let mut bad = ladder("s6-pico");
        bad.memory_value = MemoryValue::Frame;
        assert!(bad.validate().is_err(), "a value form needs a memory");
    }

    #[test]
    fn c1_a_shown_colour_is_retrieved_at_its_door_read_contrastively() {
        let _g = test_util::lock();
        mlx_rs::with_device(mlx_rs::Device::cpu(), || {
            // docs/s6-sleep2-prereg.md §3, C1, as amendments 2 and 3 read it: the pre-registered
            // form (v1's write rule, shared qk, values `frame`), each delay played with the four
            // colours on 20 draws, decoded contrastively; at least 18 of 20 draws right at 1, 4
            // and 8 idle frames. 16 and 32 are printed. (C1's first, raw reading failed: its
            // decoder misread even an exactly stored value, amendment 2's D1.)
            let rows = contrastive_sweep(&[MemoryWrite::Average], &[1], &[1, 4, 8, 16, 32], 0..20)
                .unwrap();
            for r in &rows {
                println!(
                    "C1: form {:5} idle {:2}: {} of {} draws",
                    r.form, r.idle, r.right_draws, r.draws
                );
            }
            for r in rows.iter().filter(|r| r.form == "frame" && r.idle <= 8) {
                assert!(r.right_draws >= 18, "{r:?}");
            }
        });
    }

    #[test]
    fn at_step_0_only_the_read_out_of_the_memory_has_a_gradient() {
        let _g = test_util::lock();
        mlx_rs::with_device(mlx_rs::Device::cpu(), || {
            // docs/s6-sleep2-study.md §1.1: with Wo = 0, ∂L/∂r = 0, so every memory weight but
            // the read-out gets exactly zero gradient at step 0, in v1 and in v2 alike.
            let f = frames(
                &sleepy(),
                &[
                    "show a colour red ask none noise 3",
                    "show none colour none ask none noise 8",
                    "show none colour none ask a noise 1",
                ],
            );
            for cfg in [
                sleepy(),
                v2(sleepy(), MemoryValue::Token),
                v2(sleepy(), MemoryValue::Frame),
            ] {
                let p = init(&cfg, S2Parts::DECISION, 4).unwrap();
                let grads = grads_of(&cfg, &p, &f);
                let mut seen = 0;
                for (n, g) in &grads {
                    if !n.starts_with("mem.") {
                        continue;
                    }
                    seen += 1;
                    let size = total(g);
                    if n.ends_with(".out.weight") {
                        assert!(size > 0.0, "{n}: no gradient");
                    } else {
                        assert_eq!(size, 0.0, "{n} ({:?}): gradient {size}", cfg.memory_qk);
                    }
                }
                assert_eq!(seen, cfg.state_layers * (inputs(&cfg).len() + 4));
            }
        });
    }

    #[test]
    fn a_zeroed_read_leaves_the_logits_of_the_base() {
        let _g = test_util::lock();
        mlx_rs::with_device(mlx_rs::Device::cpu(), || {
            let f = frames(
                &sleepy(),
                &[
                    "show a colour red ask none noise 3",
                    "show none colour none ask a noise 1",
                ],
            );
            for value in [MemoryValue::Token, MemoryValue::Frame] {
                let cfg = v2(sleepy(), value);
                let mut base = cfg.clone();
                (base.memory, base.memory_qk, base.memory_value) =
                    (false, MemoryQk::Separate, MemoryValue::Token);
                // A fresh v2 model is its base, whatever it wrote.
                let fresh = init(&cfg, S2Parts::DECISION, 1).unwrap();
                let zero = Memory::zeros(&cfg, 2).unwrap();
                let (_, m1) = wake(&fresh, &cfg, &f[0], cfg.n_exits(), &zero).unwrap();
                assert!(m1.0.iter().all(|s| total(s) > 0.0), "the frame wrote");
                let (l1, _) = wake(&fresh, &cfg, &f[1], cfg.n_exits(), &m1).unwrap();
                let plain = forward(&fresh, &base, &f[1], false).unwrap();
                assert!(l1.all_close(&plain, 1e-6, 1e-6, None).unwrap());
                // Random weights: the written memory changes the answer, unless the read is
                // zeroed (C4's control), which gives the base's logits again.
                let p = golden_params(&cfg, 3).unwrap();
                let plain = forward(&p, &base, &f[1], false).unwrap();
                let (_, m1) = wake(&p, &cfg, &f[0], cfg.n_exits(), &zero).unwrap();
                let (l1, _) = wake(&p, &cfg, &f[1], cfg.n_exits(), &m1).unwrap();
                assert!(!l1.all_close(&plain, 1e-4, 1e-4, None).unwrap());
                let mut off = cfg.clone();
                off.memory_run.read_off = true;
                let (_, m1) = wake(&p, &off, &f[0], off.n_exits(), &zero).unwrap();
                let (l1, _) = wake(&p, &off, &f[1], off.n_exits(), &m1).unwrap();
                assert!(l1.all_close(&plain, 1e-5, 1e-5, None).unwrap());
            }
        });
    }

    #[test]
    fn the_draws_of_amendment_2_start_with_the_draw_of_c1() {
        let _g = test_util::lock();
        mlx_rs::with_device(mlx_rs::Device::cpu(), || {
            // Seed 0 of Synth is the setting of C1 and of amendment 1's sweep, reading for reading.
            let (_, rows) = algebra(
                &[MemoryWrite::Average, MemoryWrite::Exact],
                &[1, 2],
                &[1, 8],
                0,
            )
            .unwrap();
            let syn = Synth::new(0);
            for r in &rows {
                let rule = MemoryWrite::parse(r.rule).unwrap();
                let form = if r.form == "frame" {
                    MemoryValue::Frame
                } else {
                    MemoryValue::Token
                };
                let (cfg, p) = synth_layer(rule, form).unwrap();
                let got = synth_read(&syn, &cfg, &p, 0, r.idle, r.passes).unwrap();
                let proj = syn.colours().map(|c| dot(&got, c));
                for (a, b) in proj.iter().zip(&r.colours) {
                    assert!((a - b).abs() < 1e-4, "{r:?}: {proj:?}");
                }
            }
        });
    }

    #[test]
    fn the_exact_write_solves_its_system_and_carries_a_gradient() {
        let _g = test_util::lock();
        mlx_rs::with_device(mlx_rs::Device::cpu(), || {
            // spd_solve against MLX's CPU inverse, on K Kᵀ + λI with a repeated key.
            let key = mlx_rs::random::key(3).unwrap();
            let k = mlx_rs::random::normal::<f32>(&[2, 1, 6, 8][..], 0.0, 1.0, &key).unwrap();
            let k = l2norm(
                &ops::concatenate(
                    &[
                        k.clone(),
                        k.take_axis(Array::from_slice(&[0], &[1]), 2).unwrap(),
                    ],
                    2,
                )
                .unwrap(),
            )
            .unwrap();
            let g = ops::matmul(&k, k.transpose_axes(&[0, 1, 3, 2]).unwrap())
                .unwrap()
                .add(
                    ops::eye::<f32>(7, None, None)
                        .unwrap()
                        .multiply(Array::from_f32(EXACT_LAMBDA))
                        .unwrap(),
                )
                .unwrap();
            let r = mlx_rs::random::normal::<f32>(
                &[2, 1, 7, 8][..],
                0.0,
                1.0,
                &mlx_rs::random::key(4).unwrap(),
            )
            .unwrap();
            let want = ops::matmul(mlx_rs::linalg::inv(&g).unwrap(), &r).unwrap();
            let got = spd_solve(&g, &r).unwrap();
            let back = ops::matmul(&g, &got).unwrap();
            assert!(back.all_close(&r, 1e-3, 1e-3, None).unwrap(), "G·X = R");
            assert!(
                got.all_close(&want, 1e-2, 1e-2, None).unwrap(),
                "against linalg::inv"
            );
            // The gradient flows through the solve (MLX's inverse has none).
            let f = |p: HashMap<Rc<str>, Array>, _: ()| -> R<Vec<Array>> {
                Ok(vec![spd_solve(&p["g"], &r)?.square()?.sum(None)?])
            };
            let mut m = HashMap::new();
            m.insert(Rc::<str>::from("g"), g.clone());
            let (_, grads) = keyed_value_and_grad(f)(m, ()).unwrap();
            assert!(total(&grads["g"]) > 0.0);
        });
    }

    #[test]
    fn each_write_rule_stores_what_it_says() {
        let _g = test_util::lock();
        mlx_rs::with_device(mlx_rs::Device::cpu(), || {
            let base = v2(sleepy(), MemoryValue::Token);
            let p = init(&base, S2Parts::DECISION, 6).unwrap();
            let f = frames(&base, &["show a colour red ask none noise 3"]);
            let zero = Memory::zeros(&base, 2).unwrap();
            let (_, _, written) =
                forward_prepared_mem(&p, &base, &f[0], base.n_exits(), None, Some(&zero.0))
                    .unwrap();
            let x = written[0].as_ref().unwrap();
            let valid = &f[0].valid;
            let n_valid = (host_f32(&valid.as_dtype(Dtype::Float32).unwrap().sum(None).unwrap())
                .unwrap()[0]
                / 2.0) as usize;
            let y = layer_norm(x, w(&p, &name(0, "norm")).unwrap(), None, base.norm_eps).unwrap();
            let v = project(&p, &base, 0, "v", &y).unwrap();
            let k = keys(&p, &base, 0, x).unwrap();
            let s0 = &zero.0[0];
            let err_at = |cfg: &S2Config, i: i32| -> f32 {
                let s = write(&p, cfg, 0, x, valid, s0).unwrap();
                let at = Array::from_slice(&[i], &[1]);
                let (ki, vi) = (k.take_axis(&at, 2).unwrap(), v.take_axis(&at, 2).unwrap());
                let got = ops::matmul(&ki, s.transpose_axes(&[0, 1, 3, 2]).unwrap()).unwrap();
                let e = host_f32(&got.subtract(&vi).unwrap().abs().unwrap().max(None).unwrap())
                    .unwrap()[0];
                e / host_f32(&vi.abs().unwrap().max(None).unwrap()).unwrap()[0]
            };
            let with = |w: MemoryWrite| {
                let mut c = base.clone();
                c.memory_write = w;
                c
            };
            // W1, the last token written at full step: S k = v exactly for it.
            // (two identical rows: half the valid flags are row 0's)
            assert!(err_at(&with(MemoryWrite::Delta), n_valid as i32 - 1) < 1e-4);
            // W2: S k_i ≈ v_i for every valid token that is not a repeat (λ = 1e-3).
            assert!(
                err_at(&with(MemoryWrite::Exact), 0) < 0.05,
                "{}",
                err_at(&with(MemoryWrite::Exact), 0)
            );
            // v1 stores a small part of each pair (β/n ≈ 0.5/n).
            assert!(err_at(&with(MemoryWrite::Average), 0) > 0.5);
            // Every rule trains the read-out at step 0 and runs through a sleep.
            let two = frames(
                &base,
                &[
                    "show a colour red ask none noise 3",
                    "show none colour none ask a noise 1",
                ],
            );
            for rule in [MemoryWrite::Delta, MemoryWrite::Exact, MemoryWrite::Centred] {
                let cfg = with(rule);
                let g = grads_of(&cfg, &p, &two);
                assert!(total(&g["mem.0.out.weight"]) > 0.0, "{rule:?}");
                let m = sleep(&p, &cfg, &[&f[0]], zero.clone(), Sleep::passes(3)).unwrap();
                assert!(m.0.iter().all(|s| total(s) > 0.0), "{rule:?}");
            }
        });
    }

    /// Position in the state stream of the door named in `state` (the one token that changes
    /// when the door does).
    fn door_at(cfg: &S2Config, state: &str, door: &str, other: &str) -> usize {
        let (_d, tok) = s2_tok();
        let ids = |s: &str| {
            encode_record_cfg(&tok, &rec("choice", "which key?", &["wait"], s), cfg)
                .unwrap()
                .1
                .ids
        };
        let (a, b) = (ids(state), ids(&state.replace(door, other)));
        let diff: Vec<usize> = (0..a.len()).filter(|&i| a[i] != b[i]).collect();
        assert_eq!(diff.len(), 1, "{state}: {diff:?}");
        diff[0]
    }

    #[test]
    fn a_fact_written_by_hand_is_read_back_as_its_colour_at_the_ask() {
        let _g = test_util::lock();
        mlx_rs::with_device(mlx_rs::Device::cpu(), || {
            // The test docs/s6-sleep-diagnosis.md §4 asked for: a hand-set Wq = Wk (here the
            // shared Wqk = I), a memory written by hand with the colour of each door under the
            // key of that door's token at its show frame, and the real read at the ask frame's
            // door token: it returns the right colour for every door, layer and head, and the
            // read reaches the answer (the logits move when Wo is not zero).
            let cfg = v2(sleepy(), MemoryValue::Frame);
            let (h, hd, d) = (cfg.n_heads(), cfg.head_dim(), cfg.hidden_size as i32);
            let mut p = init(&cfg, S2Parts::DECISION, 11).unwrap();
            for l in 0..cfg.state_layers {
                p.insert(
                    name(l, "qk").into(),
                    ops::eye::<f32>(d, None, None).unwrap(),
                );
            }
            let doors = [("a", 0usize), ("b", 1), ("c", 2)];
            let zero = Memory::zeros(&cfg, 2).unwrap();
            // The keys of each door at its show frame (block outputs of the last application).
            let mut key_of = Vec::new();
            for (door, colour) in doors {
                let colour = ["red", "green", "blue", "yellow"][colour];
                let state = format!("show {door} colour {colour} ask none noise 3");
                let at = door_at(&cfg, &state, &format!("show {door}"), "show x");
                let f = frames(&cfg, &[state.as_str()]);
                let (_, _, written) =
                    forward_prepared_mem(&p, &cfg, &f[0], cfg.n_exits(), None, Some(&zero.0))
                        .unwrap();
                let per_layer: Vec<Vec<f32>> = (0..cfg.state_layers)
                    .map(|l| {
                        let x = written[l].as_ref().unwrap();
                        let k = host_f32(&keys(&p, &cfg, l, x).unwrap()).unwrap();
                        // Row 0, every head, token `at`: [H, hd].
                        let t = x.dim(1) as usize;
                        (0..h)
                            .flat_map(|hh| k[(hh * t + at) * hd..(hh * t + at + 1) * hd].to_vec())
                            .collect()
                    })
                    .collect();
                key_of.push(per_layer);
            }
            // S = Σ_door e_colour ⊗ k_door, per layer and head, the same for both rows.
            let hand = Memory(
                (0..cfg.state_layers)
                    .map(|l| {
                        let mut m = vec![0f32; 2 * h * hd * hd];
                        for row in 0..2 {
                            for hh in 0..h {
                                for (door, (_, colour)) in doors.iter().enumerate() {
                                    let k = &key_of[door][l][hh * hd..(hh + 1) * hd];
                                    let base = ((row * h + hh) * hd + colour) * hd;
                                    for (j, kj) in k.iter().enumerate() {
                                        m[base + j] += kj;
                                    }
                                }
                            }
                        }
                        Array::from_slice(&m, &[2, h as i32, hd as i32, hd as i32])
                    })
                    .collect(),
            );
            for (door, colour) in doors {
                let state = format!("show none colour none ask {door} noise 5");
                let at = door_at(&cfg, &state, &format!("ask {door}"), "ask x");
                let f = frames(&cfg, &[state.as_str()]);
                let (_, _, written) =
                    forward_prepared_mem(&p, &cfg, &f[0], cfg.n_exits(), None, Some(&hand.0))
                        .unwrap();
                for (l, x) in written.iter().enumerate() {
                    let x = x.as_ref().unwrap();
                    let r = host_f32(&retrieve(&p, &cfg, l, x, &hand.0[l]).unwrap()).unwrap();
                    let t = x.dim(1) as usize;
                    for hh in 0..h {
                        let at_door =
                            &r[at * d as usize + hh * hd..at * d as usize + (hh + 1) * hd];
                        let got =
                            (0..4).fold(0, |b, c| if at_door[c] > at_door[b] { c } else { b });
                        assert_eq!(
                            got,
                            colour,
                            "door {door}, layer {l}, head {hh}: {:?}",
                            &at_door[..4]
                        );
                        assert!(t > at);
                    }
                }
            }
            // The read reaches the logits once Wo is not zero, and only through the memory.
            for l in 0..cfg.state_layers {
                p.insert(
                    name(l, "out").into(),
                    ops::eye::<f32>(d, None, None).unwrap(),
                );
            }
            let f = frames(&cfg, &["show none colour none ask a noise 5"]);
            let (with, _) = wake(&p, &cfg, &f[0], cfg.n_exits(), &hand).unwrap();
            let (without, _) = wake(&p, &cfg, &f[0], cfg.n_exits(), &zero).unwrap();
            assert!(!with.all_close(&without, 1e-4, 1e-4, None).unwrap());
        });
    }

    #[test]
    fn a_memory_adds_its_parameters_and_they_are_ternary_where_stated() {
        let cfg = sleepy();
        let mut base = cfg.clone();
        base.memory = false;
        assert_eq!(
            expected_count(&cfg, S2Parts::DECISION),
            expected_count(&base, S2Parts::DECISION) + param_count(&cfg)
        );
        assert_eq!(
            expected_count(&cfg, S2Parts::PRETRAIN),
            expected_count(&base, S2Parts::PRETRAIN)
        );
        // docs/s6-sleep-prereg.md §2: 166,752 parameters and 20,736 floats per stream on s6-nano.
        assert_eq!(param_count(&cfg), 166_752);
        assert_eq!(Memory::floats(&cfg), 20_736);
        for m in ["q", "k", "v", "out"] {
            assert!(is_ternary(&format!("mem.1.{m}.weight")));
            assert!(crate::s2::model::is_ternary(&format!("mem.0.{m}.weight")));
        }
        for m in ["norm", "alpha", "beta"] {
            assert!(!is_ternary(&format!("mem.0.{m}.weight")));
        }
        assert!(!is_ternary("blk.0.attn_output.weight") && !is_ternary("mem.out.weight"));
    }

    #[test]
    fn a_fresh_model_is_its_base_and_a_zero_memory_adds_nothing() {
        let _g = test_util::lock();
        mlx_rs::with_device(mlx_rs::Device::cpu(), || {
            let cfg = sleepy();
            let mut base = cfg.clone();
            base.memory = false;
            let f = frames(&cfg, &["show a colour red ask none noise 3"]);
            // Fresh init: the read-out is zero, so any memory leaves the logits alone.
            let fresh = init(&cfg, S2Parts::DECISION, 1).unwrap();
            let plain = forward(&fresh, &base, &f[0], false).unwrap();
            let zero = Memory::zeros(&cfg, 2).unwrap();
            let (l0, m1) = wake(&fresh, &cfg, &f[0], cfg.n_exits(), &zero).unwrap();
            assert!(l0.all_close(&plain, 1e-6, 1e-6, None).unwrap());
            assert!(m1.0.iter().all(|s| total(s) > 0.0), "the frame wrote");
            let (l1, _) = wake(&fresh, &cfg, &f[0], cfg.n_exits(), &m1).unwrap();
            assert!(l1.all_close(&plain, 1e-6, 1e-6, None).unwrap());
            // Random weights: a zero memory still adds nothing, a written one changes the answer.
            let p = golden_params(&cfg, 3).unwrap();
            let plain = forward(&p, &base, &f[0], false).unwrap();
            let (l0, m1) = wake(&p, &cfg, &f[0], cfg.n_exits(), &zero).unwrap();
            assert!(l0.all_close(&plain, 1e-5, 1e-5, None).unwrap());
            let (l1, _) = wake(&p, &cfg, &f[0], cfg.n_exits(), &m1).unwrap();
            let rows = |l: &Array| logits_per_row(l, &[3, 3]).unwrap();
            let (a, b) = (rows(&l0), rows(&l1));
            assert!(a[3][0]
                .iter()
                .zip(&b[3][0])
                .any(|(x, y)| (x - y).abs() > 1e-4));
        });
    }

    #[test]
    fn replaying_a_frame_brings_the_memory_closer_to_what_it_stores() {
        let _g = test_util::lock();
        mlx_rs::with_device(mlx_rs::Device::cpu(), || {
            let cfg = sleepy();
            // A fresh model does not read (zero read-out), so a replay sees the same keys and
            // values: the delta rule then shrinks the error `v − S·k` pass after pass.
            let p = init(&cfg, S2Parts::DECISION, 2).unwrap();
            let f = frames(&cfg, &["show a colour red ask none noise 3"]);
            let chunk = [&f[0]];
            let zero = Memory::zeros(&cfg, 2).unwrap();
            let (_, m1) = wake(&p, &cfg, &f[0], cfg.n_exits(), &zero).unwrap();
            let step = |m: &Memory| -> f32 {
                // The size of the next write measures the error left.
                let next = replay(&p, &cfg, &f[0], m).unwrap();
                next.0
                    .iter()
                    .zip(&m.0)
                    .map(|(a, b)| total(&a.subtract(b).unwrap()))
                    .sum()
            };
            let m4 = sleep(&p, &cfg, &chunk, m1.clone(), Sleep::passes(4)).unwrap();
            let same = sleep(&p, &cfg, &chunk, m1.clone(), Sleep::passes(1)).unwrap();
            assert_eq!(total(&same.0[0]), total(&m1.0[0]), "N = 1 does not sleep");
            assert!(step(&m4) < step(&m1), "{} vs {}", step(&m4), step(&m1));
        });
    }

    #[test]
    fn the_sequence_loss_sleeps_where_stated_and_trains_the_memory() {
        let _g = test_util::lock();
        mlx_rs::with_device(mlx_rs::Device::cpu(), || {
            let cfg = sleepy();
            let p = golden_params(&cfg, 5).unwrap();
            let f = frames(
                &cfg,
                &[
                    "show a colour red ask none noise 3",
                    "show none colour none ask none noise 8",
                    "show none colour none ask a noise 1",
                ],
            );
            let wv = exit_weight_vec(&cfg, None);
            let wts = Array::from_slice(&wv, &[wv.len() as i32]);
            let none = [false; 3];
            let loss = |pauses: &[bool], how: Sleep| -> f32 {
                seq_loss(&p, &cfg, &f, pauses, &wts, false, how)
                    .unwrap()
                    .item_exact::<f32>()
            };
            // By hand: three wake frames chained through the memory.
            let zero = Memory::zeros(&cfg, 2).unwrap();
            let mut m = zero.clone();
            let mut want = 0.0;
            for b in &f {
                let (l, next) = wake(&p, &cfg, b, cfg.n_exits(), &m).unwrap();
                want += frame_ce(&l, b, &wts).unwrap().item_exact::<f32>();
                m = next;
            }
            let awake = loss(&none, Sleep::AWAKE);
            assert!(
                (awake - want / 3.0).abs() < 1e-5,
                "{awake} vs {}",
                want / 3.0
            );
            // No pause and fewer frames than K: N passes change nothing.
            assert_eq!(loss(&none, Sleep::passes(4)), awake);
            // A pause before the last frame: the model sleeps on the first two, and the
            // answer of the last frame moves with N.
            let paused = [false, false, true];
            assert_eq!(loss(&paused, Sleep::passes(1)), awake);
            let slept = loss(&paused, Sleep::passes(4));
            assert!((slept - awake).abs() > 1e-6, "{slept} vs {awake}");
            // Without memory, the same call is the plain per-frame loss.
            let mut base = cfg.clone();
            base.memory = false;
            let plain: f32 = f
                .iter()
                .map(|b| {
                    let l = forward(&p, &base, b, false).unwrap();
                    frame_ce(&l, b, &wts).unwrap().item_exact::<f32>()
                })
                .sum();
            let control = seq_loss(&p, &base, &f, &paused, &wts, false, Sleep::passes(4))
                .unwrap()
                .item_exact::<f32>();
            assert!((control - plain / 3.0).abs() < 1e-5);
            // Every memory parameter gets a gradient, truncated or not, under QAT.
            for full in [false, true] {
                let how = Sleep {
                    passes: 3,
                    every: SLEEP_EVERY,
                    full,
                };
                let g = |q: HashMap<Rc<str>, Array>, _: ()| -> R<Vec<Array>> {
                    Ok(vec![seq_loss(&q, &cfg, &f, &paused, &wts, true, how)?])
                };
                let (_, grads) = keyed_value_and_grad(g)(p.clone(), ()).unwrap();
                for l in 0..cfg.state_layers {
                    for part in ["norm", "q", "k", "v", "out", "alpha", "beta"] {
                        let n = name(l, part);
                        assert!(total(&grads[n.as_str()]) > 0.0, "{n}: no gradient");
                    }
                }
            }
        });
    }
}
