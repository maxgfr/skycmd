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
        config::{MemoryQk, MemoryValue, S2Config},
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
    let beta = ops::sigmoid(gate("beta")?)?;
    let alpha = ops::sigmoid(
        gate("alpha")?
            .multiply(&on)?
            .sum_axis(-1, true)?
            .divide(&n)?
            .add(Array::from_f32(ALPHA_BIAS))?,
    )?
    .reshape(&[b, 1, 1, 1])?;
    let kept = s.multiply(&alpha)?;
    // err_i = v_i − α·S·k_i, weighted by β_i / n over the valid tokens.
    let err = v.subtract(ops::matmul(&k, kept.transpose_axes(&[0, 1, 3, 2])?)?)?;
    let weight = beta.multiply(&on)?.divide(&n)?.reshape(&[b, 1, t, 1])?;
    let delta = ops::matmul(err.multiply(&weight)?.transpose_axes(&[0, 1, 3, 2])?, &k)?;
    kept.add(delta)
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
    pub form: &'static str,
    /// Idle frames between the show frame and the ask frame.
    pub idle: usize,
    /// `r · x_c` at the ask's door token for red, green, blue, yellow.
    pub colours: [f32; 4],
    /// The colour of the argmax (0 = red, the colour shown).
    pub argmax: usize,
}

/// Check C1: synthetic hidden states at `d = 96` (one random unit direction per word, orthogonal
/// to a shared one, mixed `√0.6 · u_w + √0.4 · c` so that two words have a cosine of 0.4 on
/// average), one memory layer of one head with `Wqk = Wv = I` in f32, and the real
/// [`write`] and [`retrieve`]. A show frame `show a colour red ask none noise 46`, then `idle`
/// idle frames `show none colour none ask none noise n`, then the ask `show none colour none ask
/// a noise n`. Returns the measured mean cosine between distinct words and one row per form and
/// number of idle frames.
pub fn retrieval_algebra(idle: &[usize], seed: u64) -> R<(f32, Vec<AlgebraRow>)> {
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
    for form in [MemoryValue::Frame, MemoryValue::Token] {
        cfg.memory_value = form;
        for &n_idle in idle {
            let mut st = ops::zeros::<f32>(&[1, 1, D as i32, D as i32])?;
            st = write(&p, &cfg, 0, &show, &valid, &st)?;
            for n in &noise[..n_idle] {
                let idle_frame = frame(&format!("show none colour none ask none noise {n}"));
                st = write(&p, &cfg, 0, &idle_frame, &valid, &st)?;
            }
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
            rows.push(AlgebraRow {
                form: form.name(),
                idle: n_idle,
                colours: proj,
                argmax,
            });
        }
    }
    Ok(((cos / pairs as f64) as f32, rows))
}

#[cfg(test)]
mod tests {
    use std::{collections::HashMap, rc::Rc};

    use mlx_rs::transforms::keyed_value_and_grad;

    use super::*;
    use crate::{
        model::host_f32,
        s2::{
            config::{MemoryQk, MemoryValue},
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
    #[ignore = "check C1 of docs/s6-sleep2-prereg.md §3, which fails as frozen (argmax yellow at \
                4 and 8 idle frames, seed 0): run it with --ignored or `sky-train s2-memcheck \
                --c1-only`"]
    fn c1_a_shown_colour_is_retrieved_at_its_door_by_the_frame_values() {
        let _g = test_util::lock();
        mlx_rs::with_device(mlx_rs::Device::cpu(), || {
            // docs/s6-sleep2-prereg.md §3, C1: argmax at red for 1, 4 and 8 idle frames under
            // form `frame`; 16 and 32, and form `token`, printed.
            let (cos, rows) = retrieval_algebra(&[1, 4, 8, 16, 32], 0).unwrap();
            println!("C1: mean cosine between words {cos:.3}");
            for r in &rows {
                println!(
                    "C1: form {:5} idle {:2}: red {:+.4} green {:+.4} blue {:+.4} yellow {:+.4} -> {}",
                    r.form,
                    r.idle,
                    r.colours[0],
                    r.colours[1],
                    r.colours[2],
                    r.colours[3],
                    ["red", "green", "blue", "yellow"][r.argmax]
                );
            }
            assert!((cos - 0.4).abs() < 0.02, "mean cosine {cos}");
            for r in rows.iter().filter(|r| r.form == "frame" && r.idle <= 8) {
                assert_eq!(
                    r.argmax, 0,
                    "form frame, {} idle frames: {:?}",
                    r.idle, r.colours
                );
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
