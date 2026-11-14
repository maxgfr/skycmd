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

use mlx_rs::{error::Exception, ops, stop_gradient, Array, Dtype};

use crate::{
    model::layer_norm,
    params::ParamMap,
    s2::{
        config::S2Config,
        data::S2Batch,
        model::{
            cast_like, forward_prepared_mem, frame_ce, heads, linear, merge_heads, prepare,
            state_written, to_f32, w, Init,
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
/// The four projections of a memory layer (ternary in the export).
const MATRICES: [&str; 4] = ["q", "k", "v", "out"];

/// True for the projections of a memory layer.
pub fn is_ternary(name: &str) -> bool {
    name.strip_prefix("mem.")
        .and_then(|rest| rest.split_once('.'))
        .is_some_and(|(_, tail)| MATRICES.iter().any(|m| tail == format!("{m}.weight")))
}

/// Parameters of the memory layers: `mem.{l}.{norm,q,k,v,out,alpha,beta}.weight`.
pub(crate) fn specs(out: &mut Vec<(String, Vec<i32>, Init)>, cfg: &S2Config) {
    let d = cfg.hidden_size as i32;
    for l in 0..cfg.state_layers {
        let p = format!("mem.{l}");
        out.push((format!("{p}.norm.weight"), vec![d], Init::Ones));
        for m in ["q", "k", "v"] {
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
    cfg.state_layers * (4 * d * d + 3 * d)
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

/// The read of memory layer `l`: `x + merge(S·q)·Wo` (`x` `[B, T, d]`, `s` `[B, H, hd, hd]`).
pub(crate) fn read(p: &ParamMap, cfg: &S2Config, l: usize, x: &Array, s: &Array) -> R<Array> {
    let y = layer_norm(x, w(p, &name(l, "norm"))?, None, cfg.norm_eps)?;
    let q = l2norm(&project(p, cfg, l, "q", &y)?)?;
    // r_i = S · q_i, for every token at once.
    let r = ops::matmul(&q, s.transpose_axes(&[0, 1, 3, 2])?)?;
    let r = cast_like(merge_heads(&r)?, x)?;
    x.add(linear(&r, w(p, &name(l, "out"))?, None)?)
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
    let y = layer_norm(x, w(p, &name(l, "norm"))?, None, cfg.norm_eps)?;
    let k = l2norm(&project(p, cfg, l, "k", &y)?)?;
    let v = project(p, cfg, l, "v", &y)?;
    let xf = to_f32(x.clone())?;
    let on = valid.as_dtype(Dtype::Float32)?;
    let n = ops::maximum(on.sum_axis(-1, true)?, Array::from_f32(1.0))?;
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
fn after_writes(
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
    let p = &*prepare(p, cfg, qat)?;
    let n_exits = cfg.n_exits();
    let mut total: Option<Array> = None;
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
            mem = next;
            chunk.push(b);
        }
    } else {
        for b in frames {
            let (logits, _, _) = forward_prepared_mem(p, cfg, b, n_exits, None, None)?;
            add(frame_ce(&logits, b, wts)?)?;
        }
    }
    total
        .expect("at least one frame")
        .divide(Array::from_f32(frames.len() as f32))
}

#[cfg(test)]
mod tests {
    use std::{collections::HashMap, rc::Rc};

    use mlx_rs::transforms::keyed_value_and_grad;

    use super::*;
    use crate::{
        model::host_f32,
        s2::{
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
