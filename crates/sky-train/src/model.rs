//! Functional forward pass of the skycmd-s1 model family: ModernBERT encoder,
//! Laya decision head and the pretraining-only MLM head.
//!
//! Every function reads its weights from a [`ParamMap`] keyed by the SPEC
//! safetensors names, which keeps the serialized layout exact and lets
//! `keyed_value_and_grad` differentiate with respect to the whole map.

use std::borrow::Cow;

use mlx_rs::{
    error::Exception,
    fast,
    ops::{self, indexing::IndexOp},
    Array, Dtype,
};

use crate::{
    config::{ModelConfig, HEAD_DIM},
    params::ParamMap,
};

type R<T> = std::result::Result<T, Exception>;

/// Inputs of an MLM training step.
#[derive(Debug, Clone)]
pub struct MlmBatch {
    /// Corrupted input ids `[B, T]` (int32).
    pub ids: Array,
    /// Key validity `[B, T]` (bool), `None` when the batch has no padding.
    pub key_valid: Option<Array>,
    /// Flat indices into `B * T` of the predicted positions `[M]` (int32).
    pub positions: Array,
    /// Original token ids at `positions` `[M]` (int32).
    pub labels: Array,
    /// 1 for real predictions, 0 for padding entries `[M]` (f32).
    pub weights: Array,
    /// Number of non-padding tokens (for throughput accounting).
    pub n_tokens: usize,
}

/// Inputs of a decision training / evaluation step.
#[derive(Debug, Clone)]
pub struct DecisionBatch {
    /// Token ids `[B, T]` (int32), padded with `[PAD]`.
    pub ids: Array,
    /// Key validity `[B, T]` (bool), `None` when the batch has no padding.
    pub key_valid: Option<Array>,
    /// Question type per row `[B]` (int32): 0 choice, 1 score, 2 noul.
    pub qtype: Array,
    /// Marker positions `[B, K]` (int32), 0 for padding options.
    pub marker_pos: Array,
    /// Valid options `[B, K]` (bool).
    pub marker_mask: Array,
    /// Soft target `[B, K]` (f32), 0 on padding options.
    pub target: Array,
    /// Number of non-padding tokens.
    pub n_tokens: usize,
}

pub(crate) fn w<'a>(p: &'a ParamMap, name: &str) -> R<&'a Array> {
    p.get(name)
        .ok_or_else(|| Exception::custom(format!("missing parameter {name}")))
}

/// The parameters in the compute dtype (bf16 copies when `cfg.bf16`).
pub(crate) fn compute_params<'a>(p: &'a ParamMap, cfg: &ModelConfig) -> R<Cow<'a, ParamMap>> {
    if !cfg.bf16 {
        return Ok(Cow::Borrowed(p));
    }
    let mut out = ParamMap::with_capacity(p.len());
    for (k, v) in p {
        let v = if v.dtype() == Dtype::Float32 {
            v.as_dtype(Dtype::Bfloat16)?
        } else {
            v.clone()
        };
        out.insert(k.clone(), v);
    }
    Ok(Cow::Owned(out))
}

/// A scalar in the dtype of `like` (an f32 scalar would promote bf16 arrays).
fn scalar(v: f32, like: &Array) -> R<Array> {
    let a = Array::from_f32(v);
    if like.dtype() == Dtype::Float32 {
        Ok(a)
    } else {
        a.as_dtype(like.dtype())
    }
}

fn to_f32(x: Array) -> R<Array> {
    if x.dtype() == Dtype::Float32 {
        Ok(x)
    } else {
        x.as_dtype(Dtype::Float32)
    }
}

/// Copies an f32 array to the host in row-major order (works for any layout).
pub fn host_f32(a: &Array) -> R<Vec<f32>> {
    let flat = a.reshape(&[-1])?;
    flat.eval()?;
    Ok(flat.as_slice::<f32>().to_vec())
}

/// Exact (erf) GELU.
pub fn gelu_erf(x: &Array) -> R<Array> {
    let inner = ops::erf(x.multiply(scalar(std::f32::consts::FRAC_1_SQRT_2, x)?)?)?;
    x.multiply(scalar(0.5, x)?)?
        .multiply(inner.add(scalar(1.0, x)?)?)
}

/// LayerNorm over the last axis.
pub fn layer_norm(x: &Array, weight: &Array, bias: Option<&Array>, eps: f32) -> R<Array> {
    fast::layer_norm(x, weight, bias, eps)
}

/// `x @ weight^T (+ bias)`.
pub(crate) fn linear(x: &Array, weight: &Array, bias: Option<&Array>) -> R<Array> {
    let y = ops::matmul(x, weight.t())?;
    match bias {
        Some(b) => y.add(b),
        None => Ok(y),
    }
}

/// Splits a fused `[B, T, 3d]` projection into q, k, v of shape `[B, H, T, 64]`.
pub(crate) fn split_heads(qkv: &Array, n_heads: i32) -> R<(Array, Array, Array)> {
    let (b, t) = (qkv.dim(0), qkv.dim(1));
    let x = qkv
        .reshape(&[b, t, 3, n_heads, HEAD_DIM])?
        .transpose_axes(&[2, 0, 3, 1, 4])?;
    Ok((x.index(0), x.index(1), x.index(2)))
}

/// `[B, H, T, 64]` -> `[B, T, H * 64]`.
pub(crate) fn merge_heads(x: &Array) -> R<Array> {
    let (b, h, t) = (x.dim(0), x.dim(1), x.dim(2));
    x.transpose_axes(&[0, 2, 1, 3])?
        .reshape(&[b, t, h * HEAD_DIM])
}

/// Boolean attention masks (true = attend) for a batch of length `t`.
///
/// Returns `(global, local)`, each broadcastable to `[B, H, T, T]`, or `None`
/// when nothing needs masking. Padding keys are hidden from every query; a
/// query always sees itself so padded rows never become fully masked (their
/// outputs are ignored). Local layers only attend `|i - j| <= local_attention / 2`.
pub fn attention_masks(
    cfg: &ModelConfig,
    t: i32,
    key_valid: Option<&Array>,
) -> R<(Option<Array>, Option<Array>)> {
    let half = (cfg.local_attention / 2) as i32;
    let pos = Array::from_iter(0..t, &[t]);
    let qi = pos.reshape(&[t, 1])?;
    let kj = pos.reshape(&[1, t])?;
    let pad = match key_valid {
        Some(kv) => {
            let b = kv.dim(0);
            let eye = qi.eq(&kj)?;
            Some(kv.reshape(&[b, 1, 1, t])?.logical_or(&eye)?)
        }
        None => None,
    };
    let window = if t - 1 > half {
        Some(ops::abs(qi.subtract(&kj)?)?.le(Array::from_int(half))?)
    } else {
        None
    };
    let local = match (&pad, window) {
        (Some(p), Some(w)) => Some(p.logical_and(&w)?),
        (Some(p), None) => Some(p.clone()),
        (None, Some(w)) => Some(w),
        (None, None) => None,
    };
    Ok((pad, local))
}

pub(crate) fn attend(q: &Array, k: &Array, v: &Array, mask: Option<&Array>) -> R<Array> {
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();
    match mask {
        Some(m) => fast::scaled_dot_product_attention(q, k, v, scale, m, None),
        None => fast::scaled_dot_product_attention(q, k, v, scale, None, None),
    }
}

/// ModernBERT encoder: `[B, T]` ids -> `[B, T, d]` final-normed hidden states.
pub fn encoder(
    p: &ParamMap,
    cfg: &ModelConfig,
    ids: &Array,
    key_valid: Option<&Array>,
) -> R<Array> {
    let p = &*compute_params(p, cfg)?;
    let eps = cfg.norm_eps;
    let h_count = cfg.n_heads() as i32;
    let t = ids.dim(1);
    let (global, local) = attention_masks(cfg, t, key_valid)?;
    let emb = w(p, "encoder.embeddings.tok_embeddings.weight")?.take_axis(ids, 0)?;
    let mut x = layer_norm(&emb, w(p, "encoder.embeddings.norm.weight")?, None, eps)?;
    for l in 0..cfg.num_hidden_layers {
        let pre = format!("encoder.layers.{l}");
        let h = if l == 0 {
            x.clone()
        } else {
            layer_norm(&x, w(p, &format!("{pre}.attn_norm.weight"))?, None, eps)?
        };
        let qkv = linear(&h, w(p, &format!("{pre}.attn.Wqkv.weight"))?, None)?;
        let (q, k, v) = split_heads(&qkv, h_count)?;
        let theta = cfg.rope_theta(l);
        let q = fast::rope(&q, HEAD_DIM, false, theta, 1.0, 0, None)?;
        let k = fast::rope(&k, HEAD_DIM, false, theta, 1.0, 0, None)?;
        let mask = if cfg.is_global(l) {
            global.as_ref()
        } else {
            local.as_ref()
        };
        let a = merge_heads(&attend(&q, &k, &v, mask)?)?;
        x = x.add(linear(&a, w(p, &format!("{pre}.attn.Wo.weight"))?, None)?)?;
        let h = layer_norm(&x, w(p, &format!("{pre}.mlp_norm.weight"))?, None, eps)?;
        let u = linear(&h, w(p, &format!("{pre}.mlp.Wi.weight"))?, None)?;
        let half = cfg.intermediate_size as i32;
        let parts = u.split_at_indices(&[half], -1)?;
        let m = gelu_erf(&parts[0])?.multiply(&parts[1])?;
        x = x.add(linear(&m, w(p, &format!("{pre}.mlp.Wo.weight"))?, None)?)?;
    }
    layer_norm(&x, w(p, "encoder.final_norm.weight")?, None, eps)
}

/// Laya decision head over the encoder output: returns the raw marker logits `[B, K]`.
pub fn decision_logits(
    p: &ParamMap,
    cfg: &ModelConfig,
    enc: &Array,
    key_valid: Option<&Array>,
    qtype: &Array,
    marker_pos: &Array,
) -> R<Array> {
    let p = &*compute_params(p, cfg)?;
    let eps = 1e-5;
    let (b, t, d) = (enc.dim(0), enc.dim(1), enc.dim(2));
    let h_count = cfg.n_heads() as i32;
    let type_row = w(p, "type_emb.weight")?
        .take_axis(qtype, 0)?
        .reshape(&[b, 1, d])?;
    let mut h = enc.add(&type_row)?;
    // Full attention with a key padding mask only.
    let (pad, _) = attention_masks(cfg, t, key_valid)?;
    for i in 0..cfg.rl_agent.head_layers {
        let pre = format!("head.layers.{i}");
        let y = layer_norm(
            &h,
            w(p, &format!("{pre}.norm1.weight"))?,
            Some(w(p, &format!("{pre}.norm1.bias"))?),
            eps,
        )?;
        let qkv = linear(
            &y,
            w(p, &format!("{pre}.self_attn.in_proj_weight"))?,
            Some(w(p, &format!("{pre}.self_attn.in_proj_bias"))?),
        )?;
        let (q, k, v) = split_heads(&qkv, h_count)?;
        let a = merge_heads(&attend(&q, &k, &v, pad.as_ref())?)?;
        h = h.add(linear(
            &a,
            w(p, &format!("{pre}.self_attn.out_proj.weight"))?,
            Some(w(p, &format!("{pre}.self_attn.out_proj.bias"))?),
        )?)?;
        let y = layer_norm(
            &h,
            w(p, &format!("{pre}.norm2.weight"))?,
            Some(w(p, &format!("{pre}.norm2.bias"))?),
            eps,
        )?;
        let f = linear(
            &y,
            w(p, &format!("{pre}.linear1.weight"))?,
            Some(w(p, &format!("{pre}.linear1.bias"))?),
        )?;
        let f = ops::maximum(&f, scalar(0.0, &f)?)?;
        h = h.add(linear(
            &f,
            w(p, &format!("{pre}.linear2.weight"))?,
            Some(w(p, &format!("{pre}.linear2.bias"))?),
        )?)?;
    }
    // Gather the marker rows, then score them.
    let k = marker_pos.dim(1);
    let idx = ops::broadcast_to(marker_pos.reshape(&[b, k, 1])?, &[b, k, d])?;
    let m = h.take_along_axis(&idx, 1)?;
    let s = layer_norm(
        &m,
        w(p, "scorer.0.weight")?,
        Some(w(p, "scorer.0.bias")?),
        eps,
    )?;
    let s = gelu_erf(&linear(
        &s,
        w(p, "scorer.1.weight")?,
        Some(w(p, "scorer.1.bias")?),
    )?)?;
    let s = linear(&s, w(p, "scorer.3.weight")?, Some(w(p, "scorer.3.bias")?))?;
    to_f32(s.reshape(&[b, k])?)
}

/// Marker logits with invalid options pushed to -1e4.
pub fn masked_logits(logits: &Array, marker_mask: &Array) -> R<Array> {
    ops::select(marker_mask, logits, Array::from_f32(-1e4))
}

/// Full forward of the decision model: masked marker logits `[B, K]`.
pub fn decision_forward(p: &ParamMap, cfg: &ModelConfig, b: &DecisionBatch) -> R<Array> {
    let enc = encoder(p, cfg, &b.ids, b.key_valid.as_ref())?;
    let logits = decision_logits(p, cfg, &enc, b.key_valid.as_ref(), &b.qtype, &b.marker_pos)?;
    masked_logits(&logits, &b.marker_mask)
}

/// Cross-entropy of masked logits `[B, K]` against a soft target, averaged over rows.
pub fn soft_cross_entropy(masked_logits: &Array, target: &Array) -> R<Array> {
    let lse = masked_logits.logsumexp_axis(-1, true)?;
    let logp = masked_logits.subtract(&lse)?;
    target
        .multiply(&logp)?
        .sum_axis(-1, None)?
        .mean(None)?
        .negative()
}

/// Cross-entropy against the soft target over the valid options.
pub fn decision_loss(p: &ParamMap, cfg: &ModelConfig, b: &DecisionBatch) -> R<Array> {
    soft_cross_entropy(&decision_forward(p, cfg, b)?, &b.target)
}

/// MLM logits `[M, V]` at the flat `positions` of the encoder output (tied decoder).
pub fn mlm_logits(p: &ParamMap, cfg: &ModelConfig, enc: &Array, positions: &Array) -> R<Array> {
    let p = &*compute_params(p, cfg)?;
    let d = enc.dim(2);
    let n = enc.dim(0) * enc.dim(1);
    let rows = enc.reshape(&[n, d])?.take_axis(positions, 0)?;
    let h = gelu_erf(&linear(&rows, w(p, "mlm.dense.weight")?, None)?)?;
    let h = layer_norm(&h, w(p, "mlm.norm.weight")?, None, cfg.norm_eps)?;
    to_f32(linear(
        &h,
        w(p, "encoder.embeddings.tok_embeddings.weight")?,
        Some(w(p, "mlm.decoder.bias")?),
    )?)
}

/// Masked-token cross-entropy, averaged over the real predictions.
pub fn mlm_loss(p: &ParamMap, cfg: &ModelConfig, b: &MlmBatch) -> R<Array> {
    let enc = encoder(p, cfg, &b.ids, b.key_valid.as_ref())?;
    let logits = mlm_logits(p, cfg, &enc, &b.positions)?;
    let m = logits.dim(0);
    let lse = logits.logsumexp_axis(-1, None)?;
    let gold = logits
        .take_along_axis(b.labels.reshape(&[m, 1])?, -1)?
        .reshape(&[m])?;
    let ce = lse.subtract(&gold)?.multiply(&b.weights)?.sum(None)?;
    let denom = ops::maximum(b.weights.sum(None)?, Array::from_f32(1.0))?;
    ce.divide(&denom)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        params::{self, Parts},
        test_util,
    };

    fn bools(a: &Array) -> Vec<bool> {
        a.eval().unwrap();
        a.as_slice::<bool>().to_vec()
    }

    #[test]
    fn mask_shapes_and_window() {
        let _g = test_util::lock();
        let cfg = ModelConfig::tiny(32); // local_attention 4 -> |i-j| <= 2
        let kv = Array::from_slice(
            &[
                true, true, true, true, true, false, true, true, true, true, true, true,
            ],
            &[2, 6],
        );
        let (g, l) = attention_masks(&cfg, 6, Some(&kv)).unwrap();
        let (g, l) = (g.unwrap(), l.unwrap());
        assert_eq!(g.shape(), &[2, 1, 6, 6]);
        assert_eq!(l.shape(), &[2, 1, 6, 6]);
        let gv = bools(&g);
        let lv = bools(&l);
        // Batch 0, query 0: key 5 is padding (hidden), keys 0..5 visible globally.
        assert_eq!(&gv[0..6], &[true, true, true, true, true, false]);
        // Padded query 5 still sees itself.
        assert!(gv[5 * 6 + 5]);
        // Local window of query 0 in batch 1: keys 0..=2.
        let off = 36;
        assert_eq!(&lv[off..off + 6], &[true, true, true, false, false, false]);
        // Query 3 of batch 1: keys 1..=5.
        assert_eq!(
            &lv[off + 18..off + 24],
            &[false, true, true, true, true, true]
        );
        // No padding and a short sequence: nothing to mask.
        let (g, l) = attention_masks(&cfg, 3, None).unwrap();
        assert!(g.is_none() && l.is_none());
        let (g, l) = attention_masks(&cfg, 8, None).unwrap();
        assert!(g.is_none());
        assert_eq!(l.unwrap().shape(), &[8, 8]);
    }

    #[test]
    fn rope_is_neox_rotate_half() {
        let _g = test_util::lock();
        // x: [1, 1, T=3, D=64] of ones; compare against an explicit rotate_half RoPE.
        let t = 3;
        let d = HEAD_DIM as usize;
        let data: Vec<f32> = (0..t * d)
            .map(|i| ((i * 37 % 11) as f32 - 5.0) / 5.0)
            .collect();
        let x = Array::from_slice(&data, &[1, 1, t as i32, HEAD_DIM]);
        let theta = 10_000.0f32;
        let y = fast::rope(&x, HEAD_DIM, false, theta, 1.0, 0, None).unwrap();
        assert_eq!(y.shape(), &[1, 1, 3, 64]);
        y.eval().unwrap();
        let got = y.as_slice::<f32>();
        let half = d / 2;
        for pos in 0..t {
            for i in 0..half {
                let inv = theta.powf(-(2.0 * i as f32) / d as f32);
                let (s, c) = (pos as f32 * inv).sin_cos();
                let x1 = data[pos * d + i];
                let x2 = data[pos * d + i + half];
                let e1 = x1 * c - x2 * s;
                let e2 = x2 * c + x1 * s;
                assert!((got[pos * d + i] - e1).abs() < 1e-4, "pos {pos} i {i}");
                assert!(
                    (got[pos * d + i + half] - e2).abs() < 1e-4,
                    "pos {pos} i {i}"
                );
            }
        }
    }

    #[test]
    fn padding_does_not_change_valid_outputs() {
        let _g = test_util::lock();
        let cfg = ModelConfig::tiny(32);
        let p = params::init(&cfg, Parts::DECISION, 3).unwrap();
        let ids = [2, 7, 8, 9, 3, 4, 10, 4, 11, 3];
        let short = DecisionBatch {
            ids: Array::from_slice(&ids, &[1, 10]),
            key_valid: None,
            qtype: Array::from_slice(&[0], &[1]),
            marker_pos: Array::from_slice(&[5, 7], &[1, 2]),
            marker_mask: Array::from_slice(&[true, true], &[1, 2]),
            target: Array::from_slice(&[1.0f32, 0.0], &[1, 2]),
            n_tokens: 10,
        };
        let mut padded_ids = ids.to_vec();
        padded_ids.extend([0; 6]);
        let mut valid = vec![true; 10];
        valid.extend([false; 6]);
        let padded = DecisionBatch {
            ids: Array::from_slice(&padded_ids, &[1, 16]),
            key_valid: Some(Array::from_slice(&valid, &[1, 16])),
            ..short.clone()
        };
        let a = decision_forward(&p, &cfg, &short).unwrap();
        let b = decision_forward(&p, &cfg, &padded).unwrap();
        assert!(a.all_close(&b, 1e-4, 1e-4, None).unwrap());
    }
}
