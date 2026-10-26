//! The s6-cast model (docs/SPEC-S6-CAST.md §1): parameters named like the GGUF tensors, the
//! forward pass (patch projection, the s2 state encoder with its exits, the cross-attention
//! decoder and the non-crossing quantile head) and the pinball loss. sky-infer `cast::model`
//! mirrors it on the CPU.

use std::borrow::Cow;

use anyhow::Result;
use mlx_rs::{error::Exception, ops, random, Array, Dtype};

use crate::{
    model::layer_norm,
    params::ParamMap,
    s2::model::{
        attend, cast_like, encoder_specs, heads, linear, merge_heads, state_run, ternarize_stable,
        ternarize_ste, w, Init,
    },
};

use super::config::CastConfig;

type R<T> = std::result::Result<T, Exception>;

/// Decoder matrices (`dec.{n}.*`), stored ternary like the s2 fusion head.
const DEC_MATRICES: [&str; 5] = [
    "attn_q.weight",
    "attn_kv.weight",
    "attn_output.weight",
    "ffn_up.weight",
    "ffn_down.weight",
];

/// Bias of the non-median head outputs at initialization: `softplus(−3) ≈ 0.049` between
/// neighbouring levels in normalized units, instead of `softplus(0) ≈ 0.69`.
const HEAD_SPREAD_BIAS: f32 = -3.0;

/// True for the matrices stored ternary and fake-quantized during QAT: the encoder blocks (as
/// s2), the decoder and the quantile head. `patch_proj`, `cast_query` and the norms stay float.
pub fn is_ternary(name: &str) -> bool {
    if name == "quant_head.weight" {
        return true;
    }
    let dec = name.starts_with("dec.") && DEC_MATRICES.iter().any(|m| name.ends_with(m));
    dec || crate::s2::model::is_ternary(name)
}

/// Every parameter as (name, shape, init) in a stable order.
fn specs(cfg: &CastConfig) -> Vec<(String, Vec<i32>, Init)> {
    let d = cfg.hidden_size as i32;
    let ff = cfg.intermediate_size as i32;
    let p2 = 2 * cfg.patch as i32;
    let lin = |fan_in: i32| Init::Uniform(1.0 / (fan_in as f32).sqrt());
    let mut out = vec![
        ("patch_proj.weight".to_string(), vec![d, p2], lin(p2)),
        ("patch_proj.bias".to_string(), vec![d], Init::Zeros),
    ];
    let px = cfg.parallax.then_some(cfg.head_dim() as i32);
    encoder_specs(
        &mut out,
        "blk.",
        cfg.state_layers,
        d,
        ff,
        cfg.depth(),
        None,
        px,
    );
    if cfg.state_loops > 1 {
        out.push((
            "blk.0.attn_norm_loop.weight".to_string(),
            vec![d],
            Init::Ones,
        ));
        out.push((
            "loop_embd.weight".to_string(),
            vec![cfg.state_loops as i32, d],
            Init::TruncNormal(0.02),
        ));
    }
    for e in 0..cfg.n_exits() {
        out.push((format!("exit.{e}.norm.weight"), vec![d], Init::Ones));
    }
    out.push((
        "cast_query.weight".to_string(),
        vec![cfg.horizon_patches as i32, d],
        Init::TruncNormal(0.02),
    ));
    for n in 0..cfg.decoder_layers {
        let p = format!("dec.{n}");
        for norm in ["q_norm", "kv_norm", "ffn_norm"] {
            out.push((format!("{p}.{norm}.weight"), vec![d], Init::Ones));
            out.push((format!("{p}.{norm}.bias"), vec![d], Init::Zeros));
        }
        for (name, rows, cols) in [
            ("attn_q", d, d),
            ("attn_kv", 2 * d, d),
            ("attn_output", d, d),
            ("ffn_up", 2 * d, d),
            ("ffn_down", d, 2 * d),
        ] {
            out.push((format!("{p}.{name}.weight"), vec![rows, cols], lin(cols)));
            out.push((format!("{p}.{name}.bias"), vec![rows], Init::Zeros));
        }
    }
    out.push(("out_norm.weight".to_string(), vec![d], Init::Ones));
    out.push(("out_norm.bias".to_string(), vec![d], Init::Zeros));
    let nout = (cfg.patch * cfg.quantiles.len()) as i32;
    out.push(("quant_head.weight".to_string(), vec![nout, d], lin(d)));
    out.push(("quant_head.bias".to_string(), vec![nout], Init::Zeros));
    out
}

/// Names and shapes of every parameter.
pub fn param_shapes(cfg: &CastConfig) -> Vec<(String, Vec<i32>)> {
    specs(cfg).into_iter().map(|(n, s, _)| (n, s)).collect()
}

/// Number of scalars.
pub fn expected_count(cfg: &CastConfig) -> usize {
    specs(cfg)
        .iter()
        .map(|(_, s, _)| s.iter().product::<i32>() as usize)
        .sum()
}

/// The initial `quant_head.bias`: 0.5 (the middle of the normalized context range) on the
/// median outputs, [`HEAD_SPREAD_BIAS`] on the others.
fn head_bias(cfg: &CastConfig) -> Vec<f32> {
    let q = cfg.quantiles.len();
    let m = q / 2;
    (0..cfg.patch * q)
        .map(|i| if i % q == m { 0.5 } else { HEAD_SPREAD_BIAS })
        .collect()
}

/// A fresh parameter map (f32), deterministic in `seed`.
pub fn init(cfg: &CastConfig, seed: u64) -> Result<ParamMap> {
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
    let b = head_bias(cfg);
    map.insert(
        "quant_head.bias".into(),
        Array::from_slice(&b, &[b.len() as i32]),
    );
    crate::params::eval(&map)?;
    Ok(map)
}

/// The parameters of the forward pass: ternary fake-quantized when `qat`.
pub fn prepare<'a>(p: &'a ParamMap, cfg: &CastConfig, qat: bool) -> R<Cow<'a, ParamMap>> {
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

/// A training batch of windows (docs/SPEC-S6-CAST.md §2).
pub struct CastBatch {
    /// `[B, context]` normalized values (0 where missing).
    pub vals: Array,
    /// `[B, context]` observation mask (1 / 0).
    pub mask: Array,
    /// `[B, context_patches]`: the patch is at or after the first observed one (left padding is
    /// masked out of every attention).
    pub valid: Array,
    /// `[B, horizon]` normalized targets (0 where missing) and their mask.
    pub y: Array,
    pub ymask: Array,
    /// `[B, horizon, Q]` distillation targets (the teacher's normalized quantiles) and their
    /// `[B, horizon]` step mask (0 without a teacher).
    pub tq: Array,
    pub tmask: Array,
    pub rows: usize,
}

/// `log(1 + e^x)`, the same stable formula as sky-infer `cast::model::softplus`.
fn softplus(x: &Array) -> R<Array> {
    let pos = ops::maximum(x, Array::from_f32(0.0))?;
    pos.add(ops::log1p(&ops::exp(&ops::abs(x)?.negative()?)?)?)
}

/// Non-crossing levels from raw head outputs `[..., Q]`: the middle output is the median, each
/// level above adds `softplus` of its output, each level below subtracts it.
pub fn monotone(raw: &Array) -> R<Array> {
    let q = raw.dim(-1);
    let m = q / 2;
    let parts = raw.split_at_indices(&[m, m + 1], -1)?;
    let (below, med, above) = (&parts[0], &parts[1], &parts[2]);
    let up = med.add(ops::cumsum(&softplus(above)?, -1, None, None)?)?;
    let down = med.subtract(ops::cumsum(&softplus(below)?, -1, true, None)?)?;
    ops::concatenate(&[down, med.clone(), up], -1)
}

/// Normalized quantiles of every exit, `[E, B, horizon, Q]` (f32).
pub fn forward(p: &ParamMap, cfg: &CastConfig, b: &CastBatch, qat: bool) -> R<Array> {
    let p = &*prepare(p, cfg, qat)?;
    forward_prepared(p, cfg, b)
}

fn forward_prepared(p: &ParamMap, cfg: &CastConfig, b: &CastBatch) -> R<Array> {
    let (bs, c, pl) = (b.rows as i32, cfg.context_patches as i32, cfg.patch as i32);
    let d = cfg.hidden_size as i32;
    let h = cfg.head_count as i32;
    let ne = cfg.n_exits() as i32;
    let hq = cfg.horizon_patches as i32;
    let nq = cfg.quantiles.len() as i32;
    let u = ops::concatenate(
        &[b.vals.reshape(&[bs, c, pl])?, b.mask.reshape(&[bs, c, pl])?],
        -1,
    )?;
    let x = linear(
        &u,
        w(p, "patch_proj.weight")?,
        Some(w(p, "patch_proj.bias")?),
    )?;
    let enc = cfg.encoder();
    let exits: Vec<usize> = (0..cfg.n_exits()).collect();
    let hs: Vec<Array> = state_run(p, &enc, x, &b.valid, &exits)?
        .into_iter()
        .map(|(h, _)| h)
        .collect();
    // every exit at once: [E·B, C, d]
    let kv_in = ops::concatenate(&hs, 0)?;
    let kmask =
        ops::concatenate(&vec![b.valid.clone(); hs.len()], 0)?.reshape(&[ne * bs, 1, 1, c])?;
    let ln = |x: &Array, name: &str| -> R<Array> {
        layer_norm(
            x,
            w(p, &format!("{name}.weight"))?,
            Some(w(p, &format!("{name}.bias"))?),
            cfg.norm_eps,
        )
    };
    let lin = |x: &Array, name: &str| -> R<Array> {
        linear(
            x,
            w(p, &format!("{name}.weight"))?,
            Some(w(p, &format!("{name}.bias"))?),
        )
    };
    let query = w(p, "cast_query.weight")?.reshape(&[1, hq, d])?;
    let mut xq = ops::broadcast_to(&query, &[ne * bs, hq, d])?;
    for n in 0..cfg.decoder_layers {
        let pre = format!("dec.{n}");
        let q = heads(
            &lin(
                &ln(&xq, &format!("{pre}.q_norm"))?,
                &format!("{pre}.attn_q"),
            )?,
            h,
        )?;
        let kvp = lin(
            &ln(&kv_in, &format!("{pre}.kv_norm"))?,
            &format!("{pre}.attn_kv"),
        )?;
        let parts = kvp.split_at_indices(&[d], -1)?;
        let a = merge_heads(&attend(
            &q,
            &heads(&parts[0], h)?,
            &heads(&parts[1], h)?,
            &kmask,
        )?)?;
        xq = xq.add(lin(&a, &format!("{pre}.attn_output"))?)?;
        let f = lin(
            &ln(&xq, &format!("{pre}.ffn_norm"))?,
            &format!("{pre}.ffn_up"),
        )?;
        let f = ops::maximum(&f, cast_like(Array::from_f32(0.0), &f)?)?;
        xq = xq.add(lin(&f, &format!("{pre}.ffn_down"))?)?;
    }
    let raw = lin(&ln(&xq, "out_norm")?, "quant_head")?;
    let raw = raw.reshape(&[ne, bs, hq * pl, nq])?;
    let q = monotone(&raw)?;
    if q.dtype() == Dtype::Float32 {
        Ok(q)
    } else {
        q.as_dtype(Dtype::Float32)
    }
}

/// Mean pinball loss per exit, `[E]`: over the quantile levels and the observed target steps.
pub fn exit_losses(q: &Array, cfg: &CastConfig, b: &CastBatch) -> R<Array> {
    let nq = cfg.quantiles.len() as i32;
    let tau = Array::from_slice(&cfg.quantiles, &[1, 1, 1, nq]);
    let y = b.y.reshape(&[1, b.rows as i32, -1, 1])?;
    let e = y.subtract(q)?;
    let pin = ops::maximum(
        e.multiply(&tau)?,
        e.multiply(tau.subtract(Array::from_f32(1.0))?)?,
    )?;
    let m = b.ymask.reshape(&[1, b.rows as i32, -1, 1])?;
    let total = pin.multiply(&m)?.sum_axes(&[1, 2, 3], None)?;
    let count = b
        .ymask
        .sum(None)?
        .multiply(Array::from_f32(nq as f32))?
        .add(Array::from_f32(1e-6))?;
    total.divide(&count)
}

/// Mean absolute gap to the teacher's quantiles per exit, `[E]`: over the levels and the steps
/// the teacher covers.
pub fn distill_losses(q: &Array, cfg: &CastConfig, b: &CastBatch) -> R<Array> {
    let nq = cfg.quantiles.len() as i32;
    let t = b.tq.reshape(&[1, b.rows as i32, -1, nq])?;
    let m = b.tmask.reshape(&[1, b.rows as i32, -1, 1])?;
    let total = ops::abs(&q.subtract(&t)?)?
        .multiply(&m)?
        .sum_axes(&[1, 2, 3], None)?;
    let count = b
        .tmask
        .sum(None)?
        .multiply(Array::from_f32(nq as f32))?
        .add(Array::from_f32(1e-6))?;
    total.divide(&count)
}

/// Deep supervision: `Σ_e w_e · (pinball(exit e) + distill_w · distill(exit e))` with the exit
/// weights `wts` `[E]`. The ground truth always stays in the loss (a student fitting the labels
/// and the teacher can surpass the teacher: arXiv 1805.04770, 2206.08491).
pub fn loss(
    p: &ParamMap,
    cfg: &CastConfig,
    b: &CastBatch,
    qat: bool,
    wts: &Array,
    distill_w: f32,
) -> R<Array> {
    let q = forward(p, cfg, b, qat)?;
    let mut per_exit = exit_losses(&q, cfg, b)?;
    if distill_w > 0.0 {
        per_exit =
            per_exit.add(distill_losses(&q, cfg, b)?.multiply(Array::from_f32(distill_w))?)?;
    }
    per_exit.multiply(wts)?.sum(None)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_util;

    fn cfg() -> CastConfig {
        CastConfig::load(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../configs/s6-cast-femto.json"),
        )
        .unwrap()
    }

    #[test]
    fn parameter_count_and_groups() {
        let _g = test_util::lock();
        let c = cfg();
        let p = init(&c, 0).unwrap();
        assert_eq!(crate::params::count(&p), expected_count(&c));
        // the encoder blocks are trained with Muon, the decoder with AdamW
        use crate::params::{group_of, Group};
        assert_eq!(group_of("blk.0.attn_qkv.weight", 2), Group::Muon);
        assert_eq!(group_of("dec.0.attn_q.weight", 2), Group::AdamDecay);
        assert_eq!(group_of("patch_proj.weight", 2), Group::AdamNoDecay);
        assert_eq!(group_of("cast_query.weight", 2), Group::AdamNoDecay);
        assert!(is_ternary("dec.0.ffn_down.weight") && is_ternary("quant_head.weight"));
        assert!(is_ternary("blk.0.attn_qkv.weight") && !is_ternary("patch_proj.weight"));
        assert!(!is_ternary("dec.0.ffn_down.bias") && !is_ternary("cast_query.weight"));
    }

    #[test]
    fn monotone_matches_the_cpu_formula() {
        let _g = test_util::lock();
        mlx_rs::with_device(mlx_rs::Device::cpu(), || {
            let raw = [3.0f32, -2.0, 0.5, -9.0, 1.25, 4.0, -30.0, 0.0, 2.0];
            let q = monotone(&Array::from_slice(&raw, &[1, 9])).unwrap();
            let got = q.as_slice::<f32>().to_vec();
            let mut want = [0.0; 9];
            sky_infer::cast::model::monotone(&raw, &mut want);
            for (a, b) in got.iter().zip(want) {
                assert!((a - b).abs() < 1e-5, "{got:?} vs {want:?}");
            }
        });
    }

    #[test]
    fn pinball_loss_by_hand() {
        let _g = test_util::lock();
        mlx_rs::with_device(mlx_rs::Device::cpu(), || {
            let mut c = cfg();
            c.quantiles = vec![0.1, 0.5, 0.9];
            // one exit, one row, two steps; quantiles 0 everywhere; y = (1, -1)
            let b = CastBatch {
                vals: Array::from_slice(&[0.0f32], &[1, 1]),
                mask: Array::from_slice(&[0.0f32], &[1, 1]),
                valid: Array::from_slice(&[true], &[1, 1]),
                y: Array::from_slice(&[1.0f32, -1.0], &[1, 2]),
                ymask: Array::from_slice(&[1.0f32, 1.0], &[1, 2]),
                tq: Array::from_slice(&[0.5f32, 0.5, 0.5, 0.0, 0.0, 0.0], &[1, 2, 3]),
                tmask: Array::from_slice(&[1.0f32, 0.0], &[1, 2]),
                rows: 1,
            };
            let q = ops::zeros::<f32>(&[1, 1, 2, 3]).unwrap();
            let l = exit_losses(&q, &c, &b).unwrap();
            // y = 1: τ·1 → 0.1 + 0.5 + 0.9; y = −1: (1 − τ)·1 → 0.9 + 0.5 + 0.1; mean over 6
            let got = l.as_slice::<f32>()[0];
            assert!((got - 3.0 / 6.0).abs() < 1e-5, "{got}");
            // a masked step does not count
            let b2 = CastBatch {
                ymask: Array::from_slice(&[1.0f32, 0.0], &[1, 2]),
                ..b
            };
            let got = exit_losses(&q, &c, &b2).unwrap().as_slice::<f32>()[0];
            assert!((got - 1.5 / 3.0).abs() < 1e-5, "{got}");
            // distillation: |0 − 0.5| on the 3 levels of the one step the teacher covers
            let d = distill_losses(&q, &c, &b2).unwrap().as_slice::<f32>()[0];
            assert!((d - 0.5).abs() < 1e-5, "{d}");
        });
    }
}
