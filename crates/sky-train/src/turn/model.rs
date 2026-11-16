//! The s6-ear v2 model (docs/s6-ear2-prereg.md §4.2): parameters named like the GGUF tensors, the
//! forward pass (`patch_proj`, the encoder blocks, the final norm, Smart Turn's attention pooling,
//! the head) and the loss of §5.2. sky-infer `turn::model` mirrors the exported part on the CPU.
//!
//! A batch is left-padded: a clip's positions sit at the end of the `max_positions` slots, so the
//! 25 teacher steps are a plain reshape aligned on the window's end, and RoPE sees the same
//! relative positions as the unpadded inference graph.

use std::borrow::Cow;

use anyhow::Result;
use mlx_rs::{error::Exception, fast, ops, random, Array};

use crate::{
    model::{gelu_erf, layer_norm},
    params::ParamMap,
    s2::model::{
        attend, heads, linear, merge_heads, padding_mask, ternarize_stable, ternarize_ste, w, Init,
    },
};

use super::config::TurnConfig;

type R<T> = std::result::Result<T, Exception>;

/// Encoder matrices of a block (stored ternary, fake-quantized under QAT).
const BLOCK_MATRICES: [&str; 4] = [
    "attn_qkv.weight",
    "attn_output.weight",
    "ffn_up.weight",
    "ffn_down.weight",
];

/// Training-only projections onto the teacher's 384-d space (§4.2), never exported.
pub const POOL_PROJ: &str = "distill.pool_proj.weight";
pub const SEQ_PROJ: &str = "distill.seq_proj.weight";

/// True for the matrices stored ternary: the encoder blocks'. `patch_proj`, the pooling, the
/// head and the norms stay float (§4.2).
pub fn is_ternary(name: &str) -> bool {
    name.starts_with("blk.") && BLOCK_MATRICES.iter().any(|m| name.ends_with(m))
}

/// True for the parameters of the exported file.
pub fn is_exported(name: &str) -> bool {
    !name.starts_with("distill.")
}

/// Every parameter as (name, shape, init) in a stable order.
fn specs(cfg: &TurnConfig) -> Vec<(String, Vec<i32>, Init)> {
    let d = cfg.hidden_size as i32;
    let ff = cfg.intermediate_size as i32;
    let wd = cfg.input_width() as i32;
    let (ph, hh) = (cfg.pool_hidden as i32, cfg.head_hidden as i32);
    let t = sky_infer::turn::TEACHER_DIM as i32;
    let lin = |fan_in: i32| Init::Uniform(1.0 / (fan_in as f32).sqrt());
    let std = 0.02f32;
    let std_out = std / (2.0 * cfg.blocks as f32).sqrt();
    let mut out = vec![
        ("patch_proj.weight".to_string(), vec![d, wd], lin(wd)),
        ("patch_proj.bias".to_string(), vec![d], Init::Zeros),
    ];
    for l in 0..cfg.blocks {
        let p = format!("blk.{l}");
        out.push((format!("{p}.attn_norm.weight"), vec![d], Init::Ones));
        out.push((
            format!("{p}.attn_qkv.weight"),
            vec![3 * d, d],
            Init::TruncNormal(std),
        ));
        out.push((
            format!("{p}.attn_output.weight"),
            vec![d, d],
            Init::TruncNormal(std_out),
        ));
        out.push((format!("{p}.ffn_norm.weight"), vec![d], Init::Ones));
        out.push((
            format!("{p}.ffn_up.weight"),
            vec![ff, d],
            Init::TruncNormal(std),
        ));
        out.push((
            format!("{p}.ffn_down.weight"),
            vec![d, ff],
            Init::TruncNormal(std_out),
        ));
    }
    out.push(("out_norm.weight".to_string(), vec![d], Init::Ones));
    for (name, rows, cols) in [
        ("attn_pool.hidden", ph, d),
        ("attn_pool.score", 1, ph),
        ("cls.hidden", hh, d),
        ("cls.output", 1, hh),
    ] {
        out.push((format!("{name}.weight"), vec![rows, cols], lin(cols)));
        out.push((format!("{name}.bias"), vec![rows], Init::Zeros));
    }
    out.push((POOL_PROJ.to_string(), vec![t, d], lin(d)));
    out.push((SEQ_PROJ.to_string(), vec![t, d], lin(d)));
    out
}

/// Names and shapes of every parameter (training-only projections included).
pub fn param_shapes(cfg: &TurnConfig) -> Vec<(String, Vec<i32>)> {
    specs(cfg).into_iter().map(|(n, s, _)| (n, s)).collect()
}

/// Number of scalars of the exported model (§4.3's count).
pub fn exported_count(cfg: &TurnConfig) -> usize {
    specs(cfg)
        .iter()
        .filter(|(n, _, _)| is_exported(n))
        .map(|(_, s, _)| s.iter().product::<i32>() as usize)
        .sum()
}

/// Number of scalars trained.
pub fn expected_count(cfg: &TurnConfig) -> usize {
    specs(cfg)
        .iter()
        .map(|(_, s, _)| s.iter().product::<i32>() as usize)
        .sum()
}

/// A fresh parameter map (f32), deterministic in `seed`.
pub fn init(cfg: &TurnConfig, seed: u64) -> Result<ParamMap> {
    let mut key = random::key(seed)?;
    let mut map = ParamMap::new();
    for (name, shape, init) in specs(cfg) {
        let (k, next) = random::split(&key, 2)?;
        key = next;
        let a = match init {
            Init::TruncNormal(s) => {
                random::truncated_normal::<f32, f32>(-2.0, 2.0, &shape[..], &k)?
                    .multiply(Array::from_f32(s))?
            }
            Init::Uniform(a) => random::uniform::<f32, f32>(-a, a, &shape[..], &k)?,
            Init::Ones => ops::ones::<f32>(&shape)?,
            Init::Zeros => ops::zeros::<f32>(&shape)?,
        };
        map.insert(name.into(), a);
    }
    crate::params::eval(&map)?;
    Ok(map)
}

/// The parameters of the forward pass: ternary fake-quantized when `qat`.
pub fn prepare<'a>(p: &'a ParamMap, cfg: &TurnConfig, qat: bool) -> R<Cow<'a, ParamMap>> {
    if !qat {
        return Ok(Cow::Borrowed(p));
    }
    let mut out = ParamMap::with_capacity(p.len());
    for (k, v) in p {
        let v = if is_ternary(k) {
            if cfg.qat_grad_a > 0.0 {
                ternarize_stable(v, cfg.qat_grad_a)?
            } else {
                ternarize_ste(v)?
            }
        } else {
            v.clone()
        };
        out.insert(k.clone(), v);
    }
    Ok(Cow::Owned(out))
}

/// A training or validation batch (left-padded, see the module docs).
pub struct TurnBatch {
    /// `[B, T, input_width]` normalized positions (0 in the padding).
    pub x: Array,
    /// `[B, T]`: the slot holds a position of the clip.
    pub valid: Array,
    /// `[B]` labels (1: finished).
    pub y: Array,
    /// `[B]` the teacher's probability `σ(z_T)`.
    pub q: Array,
    /// `[B, 384]` the teacher's pooled embedding.
    pub et: Array,
    /// `[B, 25, 384]` the teacher's pooled states.
    pub st: Array,
    /// `[B, 25]`: the step is inside the student's window and the clip has teacher targets.
    pub smask: Array,
    /// `[B]`: the clip has teacher targets.
    pub tmask: Array,
    pub rows: usize,
}

/// What the forward pass gives.
pub struct TurnForward {
    /// `[B]` logits.
    pub logit: Array,
    /// `[B, d]` the attention-pooled vector.
    pub pooled: Array,
    /// `[B, T, d]` the normalized hidden states.
    pub hidden: Array,
}

fn ln_plain(x: &Array, eps: f32) -> R<Array> {
    fast::layer_norm(x, None, None, eps)
}

/// The forward pass (f32 weights, or ternary fake-quantized when `qat`).
pub fn forward(p: &ParamMap, cfg: &TurnConfig, b: &TurnBatch, qat: bool) -> R<TurnForward> {
    let p = &*prepare(p, cfg, qat)?;
    forward_prepared(p, cfg, b)
}

fn forward_prepared(p: &ParamMap, cfg: &TurnConfig, b: &TurnBatch) -> R<TurnForward> {
    let eps = cfg.norm_eps;
    let (h, d) = (cfg.head_count as i32, cfg.hidden_size as i32);
    let hd = cfg.head_dim() as i32;
    let lin = |x: &Array, name: &str, bias: bool| -> R<Array> {
        let bias = if bias {
            Some(w(p, &format!("{name}.bias"))?)
        } else {
            None
        };
        linear(x, w(p, &format!("{name}.weight"))?, bias)
    };
    let mut x = lin(&b.x, "patch_proj", true)?;
    let mask = padding_mask(&b.valid)?;
    for l in 0..cfg.blocks {
        let pre = format!("blk.{l}");
        let y = layer_norm(&x, w(p, &format!("{pre}.attn_norm.weight"))?, None, eps)?;
        let qkv = lin(&y, &format!("{pre}.attn_qkv"), false)?;
        let parts = qkv.split_at_indices(&[d, 2 * d], -1)?;
        let rope = |a: &Array| fast::rope(a, hd, false, cfg.rope_theta, 1.0, 0, None);
        let q = rope(&heads(&parts[0], h)?)?;
        let k = rope(&heads(&parts[1], h)?)?;
        let v = heads(&parts[2], h)?;
        let a = merge_heads(&attend(&q, &k, &v, &mask)?)?;
        x = x.add(lin(&a, &format!("{pre}.attn_output"), false)?)?;
        let y = layer_norm(&x, w(p, &format!("{pre}.ffn_norm.weight"))?, None, eps)?;
        let u = gelu_erf(&lin(&y, &format!("{pre}.ffn_up"), false)?)?;
        x = x.add(lin(&u, &format!("{pre}.ffn_down"), false)?)?;
    }
    let hidden = layer_norm(&x, w(p, "out_norm.weight")?, None, eps)?;
    // Smart Turn's attention pooling, a softmax over the clip's positions
    let s = lin(
        &ops::tanh(&lin(&hidden, "attn_pool.hidden", true)?)?,
        "attn_pool.score",
        true,
    )?;
    let s = s.squeeze_axes(&[-1])?;
    let s = ops::select(&b.valid, &s, Array::from_f32(f32::NEG_INFINITY))?;
    let wts = ops::softmax_axis(&s, -1, true)?;
    let pooled = wts.expand_dims(-1)?.multiply(&hidden)?.sum_axis(1, false)?;
    let z = lin(
        &gelu_erf(&lin(&pooled, "cls.hidden", true)?)?,
        "cls.output",
        true,
    )?;
    Ok(TurnForward {
        logit: z.squeeze_axes(&[-1])?,
        pooled,
        hidden,
    })
}

/// `log(1 + e^x)`, stable.
fn softplus(x: &Array) -> R<Array> {
    let pos = ops::maximum(x, Array::from_f32(0.0))?;
    pos.add(ops::log1p(&ops::exp(&ops::abs(x)?.negative()?)?)?)
}

/// `1 − cos(a, b)` over the last axis.
fn cos_gap(a: &Array, b: &Array) -> R<Array> {
    let dot = a.multiply(b)?.sum_axis(-1, false)?;
    let na = a.square()?.sum_axis(-1, false)?.sqrt()?;
    let nb = b.square()?.sum_axis(-1, false)?.sqrt()?;
    let den = ops::maximum(na.multiply(&nb)?, Array::from_f32(1e-8))?;
    Array::from_f32(1.0).subtract(dot.divide(&den)?)
}

/// The weights of the four terms of §5.2.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct LossWeights {
    pub hard: f32,
    pub soft: f32,
    pub pool: f32,
    pub seq: f32,
}

impl LossWeights {
    /// §5.2: `0.5 / 0.5 / 0.25 / 0.5`.
    pub const DISTIL: LossWeights = LossWeights {
        hard: 0.5,
        soft: 0.5,
        pool: 0.25,
        seq: 0.5,
    };
    /// The pilot's hard-label arm (§6.4): the label term alone.
    pub const HARD: LossWeights = LossWeights {
        hard: 1.0,
        soft: 0.0,
        pool: 0.0,
        seq: 0.0,
    };
}

/// The four terms of §5.2 (each a mean over its rows or steps): hard BCE, soft BCE against
/// `σ(z_T)`, `1 − cos(pool_proj(h̄), LN(e_T))` and the mean over the steps inside the window of
/// `1 − cos(seq_proj(h_t), LN(S_T,t))`, every target layer-normalized first.
pub fn loss_terms(p: &ParamMap, cfg: &TurnConfig, b: &TurnBatch, f: &TurnForward) -> R<[Array; 4]> {
    let z = &f.logit;
    let sp = softplus(z)?;
    let hard = sp.subtract(b.y.multiply(z)?)?.mean(None)?;
    let tm = &b.tmask;
    let n_t = ops::maximum(tm.sum(None)?, Array::from_f32(1.0))?;
    let soft = sp
        .subtract(b.q.multiply(z)?)?
        .multiply(tm)?
        .sum(None)?
        .divide(&n_t)?;
    let eps = cfg.norm_eps;
    let pp = linear(&f.pooled, w(p, POOL_PROJ)?, None)?;
    let pool = cos_gap(&pp, &ln_plain(&b.et, eps)?)?
        .multiply(tm)?
        .sum(None)?
        .divide(&n_t)?;
    // the positions averaged by step, aligned on the window's end (left padding)
    let (bs, d) = (b.rows as i32, cfg.hidden_size as i32);
    let steps = sky_infer::turn::TEACHER_STEPS as i32;
    let per = cfg.positions_per_step() as i32;
    let hs = f
        .hidden
        .reshape(&[bs, steps, per, d])?
        .mean_axis(2, false)?;
    let sp_ = linear(&hs, w(p, SEQ_PROJ)?, None)?;
    let n_s = ops::maximum(b.smask.sum(None)?, Array::from_f32(1.0))?;
    let seq = cos_gap(&sp_, &ln_plain(&b.st, eps)?)?
        .multiply(&b.smask)?
        .sum(None)?
        .divide(&n_s)?;
    Ok([hard, soft, pool, seq])
}

/// The loss of §5.2 with weights `lw`.
pub fn loss(p: &ParamMap, cfg: &TurnConfig, b: &TurnBatch, qat: bool, lw: LossWeights) -> R<Array> {
    let pq = &*prepare(p, cfg, qat)?;
    let f = forward_prepared(pq, cfg, b)?;
    if lw.soft == 0.0 && lw.pool == 0.0 && lw.seq == 0.0 {
        let sp = softplus(&f.logit)?;
        return sp
            .subtract(b.y.multiply(&f.logit)?)?
            .mean(None)?
            .multiply(Array::from_f32(lw.hard));
    }
    let [hard, soft, pool, seq] = loss_terms(pq, cfg, b, &f)?;
    hard.multiply(Array::from_f32(lw.hard))?
        .add(soft.multiply(Array::from_f32(lw.soft))?)?
        .add(pool.multiply(Array::from_f32(lw.pool))?)?
        .add(seq.multiply(Array::from_f32(lw.seq))?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_util;

    fn ladder(r: &str) -> TurnConfig {
        TurnConfig::load(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../configs/s6-ear2-nano.json"),
            Some(r),
        )
        .unwrap()
    }

    /// docs/s6-ear2-prereg.md §4.3: the exact exported count of every rung (about 2.71M, 1.83M,
    /// 1.28M and 1.30M), the one the probe chooses included.
    #[test]
    fn parameter_count_per_rung() {
        assert_eq!(exported_count(&ladder("R1")), 2_714_754);
        assert_eq!(exported_count(&ladder("R2")), 1_829_250);
        assert_eq!(exported_count(&ladder("R3")), 1_278_658);
        assert_eq!(exported_count(&ladder("R4")), 1_306_178);
        // the training-only projections add 2 × 384 × d
        let r3 = ladder("R3");
        assert_eq!(expected_count(&r3) - exported_count(&r3), 2 * 384 * 160);
        assert!(is_ternary("blk.3.ffn_down.weight") && !is_ternary("patch_proj.weight"));
        assert!(!is_ternary("cls.hidden.weight") && !is_ternary("attn_pool.score.weight"));
        assert!(!is_exported(POOL_PROJ) && is_exported("out_norm.weight"));
        use crate::params::{group_of, Group};
        assert_eq!(group_of("patch_proj.weight", 2), Group::AdamNoDecay);
        assert_eq!(group_of("cls.hidden.weight", 2), Group::AdamDecay);
    }

    fn femto() -> TurnConfig {
        TurnConfig::load(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../configs/s6-ear2-femto.json"),
            None,
        )
        .unwrap()
    }

    /// A batch of `rows` clips with `positions[r]` positions each (left-padded).
    pub(crate) fn batch(cfg: &TurnConfig, positions: &[usize], seed: u64) -> TurnBatch {
        let (t, wd) = (cfg.max_positions(), cfg.input_width());
        let mut rng = crate::rng::Rng::new(seed);
        let rows = positions.len();
        let mut x = vec![0.0f32; rows * t * wd];
        let mut valid = vec![false; rows * t];
        let mut smask = vec![0.0f32; rows * 25];
        for (r, &n) in positions.iter().enumerate() {
            for j in t - n..t {
                valid[r * t + j] = true;
                for c in 0..wd {
                    x[(r * t + j) * wd + c] = rng.f64() as f32 * 2.0 - 1.0;
                }
            }
            for s in 0..25 {
                smask[r * 25 + s] = (s * cfg.positions_per_step() >= t - n) as u8 as f32;
            }
        }
        let rnd = |n: usize, rng: &mut crate::rng::Rng| -> Vec<f32> {
            (0..n).map(|_| rng.f64() as f32 - 0.5).collect()
        };
        let et = rnd(rows * 384, &mut rng);
        let st = rnd(rows * 25 * 384, &mut rng);
        let (b, tt, w_) = (rows as i32, t as i32, wd as i32);
        TurnBatch {
            x: Array::from_slice(&x, &[b, tt, w_]),
            valid: Array::from_slice(&valid, &[b, tt]),
            y: Array::from_slice(&(0..rows).map(|r| (r % 2) as f32).collect::<Vec<_>>(), &[b]),
            q: Array::from_slice(&vec![0.7f32; rows], &[b]),
            et: Array::from_slice(&et, &[b, 384]),
            st: Array::from_slice(&st, &[b, 25, 384]),
            smask: Array::from_slice(&smask, &[b, 25]),
            tmask: Array::from_slice(&vec![1.0f32; rows], &[b]),
            rows,
        }
    }

    /// The padding changes nothing: a clip alone gives the same logit in a padded batch.
    #[test]
    fn padding_is_masked_and_the_loss_is_finite() {
        let _g = test_util::lock();
        mlx_rs::with_device(mlx_rs::Device::cpu(), || {
            let c = femto();
            let p = init(&c, 5).unwrap();
            assert_eq!(crate::params::count(&p), expected_count(&c));
            let b = batch(&c, &[100, 37, 3], 1);
            let f = forward(&p, &c, &b, false).unwrap();
            let z = crate::model::host_f32(&f.logit).unwrap();
            // the same short clip with other padded content gives the same logit
            let b2 = batch(&c, &[37], 1);
            let mut x2 = crate::model::host_f32(&b2.x).unwrap();
            let all = crate::model::host_f32(&b.x).unwrap();
            let (t, wd) = (c.max_positions(), c.input_width());
            x2.copy_from_slice(&all[t * wd..2 * t * wd]);
            let b2 = TurnBatch {
                x: Array::from_slice(&x2, &[1, t as i32, wd as i32]),
                ..b2
            };
            let z2 = crate::model::host_f32(&forward(&p, &c, &b2, false).unwrap().logit).unwrap();
            assert!((z[1] - z2[0]).abs() < 1e-5, "{} vs {}", z[1], z2[0]);
            for lw in [LossWeights::DISTIL, LossWeights::HARD] {
                for qat in [false, true] {
                    let l = loss(&p, &c, &b, qat, lw).unwrap();
                    let v = l.item_exact::<f32>();
                    assert!(v.is_finite() && v > 0.0, "{v}");
                }
            }
            let [h, s, po, se] = loss_terms(&p, &c, &b, &f).unwrap();
            // hard BCE near ln 2 at initialization; cosine gaps near 1
            assert!((h.item_exact::<f32>() - std::f32::consts::LN_2).abs() < 0.2);
            assert!(s.item_exact::<f32>() > 0.0);
            assert!(
                (po.item_exact::<f32>() - 1.0).abs() < 0.5
                    && (se.item_exact::<f32>() - 1.0).abs() < 0.5
            );
        });
    }
}
