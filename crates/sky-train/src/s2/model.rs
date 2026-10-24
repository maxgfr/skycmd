//! The skycmd-s2 model: parameters (named exactly like the GGUF tensors),
//! static encoder, state encoder with exits, cross-attention fusion head,
//! scorer, ternary fake-quantization and the training losses.

use std::borrow::Cow;

use anyhow::Result;
use mlx_rs::{
    error::Exception,
    fast,
    ops::{self, indexing::IndexOp},
    random, stop_gradient, Array, Dtype,
};

use crate::{
    model::{gelu_erf, layer_norm},
    num::{features, NUM_FEATURES},
    params::ParamMap,
    s2::{
        config::S2Config,
        data::{NumMlmBatch, S2Batch},
    },
};

type R<T> = std::result::Result<T, Exception>;

/// Which parts a parameter map holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct S2Parts {
    /// Static encoder, intermediate exits, fusion head and scorer.
    pub decision: bool,
    /// Pretraining heads (`mlm.*`, `mlm_num.*`, never exported).
    pub mlm: bool,
}

impl S2Parts {
    /// Embeddings + state stack (+ its final norm, the last exit) + MLM heads.
    pub const PRETRAIN: S2Parts = S2Parts {
        decision: false,
        mlm: true,
    };
    pub const DECISION: S2Parts = S2Parts {
        decision: true,
        mlm: false,
    };
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Init {
    TruncNormal(f32),
    Uniform(f32),
    Ones,
    Zeros,
}

/// Encoder matrices named `*.attn_qkv.weight` etc. of a block prefix.
const BLOCK_MATRICES: [&str; 4] = [
    "attn_qkv.weight",
    "attn_output.weight",
    "ffn_up.weight",
    "ffn_down.weight",
];

/// True for the matrices stored ternary (and fake-quantized during QAT):
/// every linear layer of both encoders and of the fusion head, plus the
/// scorer's hidden layer `cls.weight`. `cls.output`, the embeddings
/// (`token_embd`, `num_proj`, `type_embd`) and the norms are excluded.
pub fn is_ternary(name: &str) -> bool {
    if name == "cls.weight" {
        return true;
    }
    let enc = (name.starts_with("blk.") || name.starts_with("static.blk."))
        && BLOCK_MATRICES.iter().any(|m| name.ends_with(m));
    let fusion = name.starts_with("fusion.")
        && [
            "attn_q.weight",
            "attn_kv.weight",
            "attn_output.weight",
            "ffn_up.weight",
            "ffn_down.weight",
        ]
        .iter()
        .any(|m| name.ends_with(m));
    enc || fusion
}

/// True for the encoder hidden matrices (optimized with Muon).
pub fn is_hidden_matrix(name: &str) -> bool {
    (name.starts_with("blk.") || name.starts_with("static.blk."))
        && BLOCK_MATRICES.iter().any(|m| name.ends_with(m))
}

/// Encoder block parameters. With `es_units` (SPEC-S3 §9 expand-and-sparsify FFN) the block has
/// no `ffn_up` (the up-projection is a fixed seeded matrix) and `ffn_down` reads `es_units` units.
/// With Parallax (`px_head_dim`, SPEC-S6 §1) `attn_qkv` gains the probe rows `ρ` and the block
/// an `attn_px_norm` over one head.
#[allow(clippy::too_many_arguments)]
fn encoder_specs(
    out: &mut Vec<(String, Vec<i32>, Init)>,
    prefix: &str,
    n: usize,
    d: i32,
    ff: i32,
    depth: usize,
    es_units: Option<i32>,
    px_head_dim: Option<i32>,
) {
    let std = 0.02f32;
    let std_out = std / (2.0 * depth as f32).sqrt();
    let projections = if px_head_dim.is_some() { 4 } else { 3 };
    for l in 0..n {
        let p = format!("{prefix}{l}");
        if l > 0 {
            out.push((format!("{p}.attn_norm.weight"), vec![d], Init::Ones));
        }
        out.push((
            format!("{p}.attn_qkv.weight"),
            vec![projections * d, d],
            Init::TruncNormal(std),
        ));
        if let Some(hd) = px_head_dim {
            out.push((format!("{p}.attn_px_norm.weight"), vec![hd], Init::Ones));
        }
        out.push((
            format!("{p}.attn_output.weight"),
            vec![d, d],
            Init::TruncNormal(std_out),
        ));
        out.push((format!("{p}.ffn_norm.weight"), vec![d], Init::Ones));
        if let Some(units) = es_units {
            out.push((
                format!("{p}.ffn_down.weight"),
                vec![d, units],
                Init::TruncNormal(std_out),
            ));
            continue;
        }
        out.push((
            format!("{p}.ffn_up.weight"),
            vec![2 * ff, d],
            Init::TruncNormal(std),
        ));
        out.push((
            format!("{p}.ffn_down.weight"),
            vec![d, ff],
            Init::TruncNormal(std_out),
        ));
    }
}

/// Every parameter as (name, shape, init) in a stable order.
fn specs(cfg: &S2Config, parts: S2Parts) -> Vec<(String, Vec<i32>, Init)> {
    let d = cfg.hidden_size as i32;
    let ff = cfg.intermediate_size as i32;
    let v = cfg.vocab_size as i32;
    let nf = NUM_FEATURES as i32;
    let lin = |fan_in: i32| Init::Uniform(1.0 / (fan_in as f32).sqrt());
    let mut out = vec![
        (
            "token_embd.weight".to_string(),
            vec![v, d],
            Init::TruncNormal(0.02),
        ),
        (
            "num_proj.weight".to_string(),
            vec![d, nf],
            Init::TruncNormal(0.02),
        ),
        ("num_proj.bias".to_string(), vec![d], Init::Zeros),
    ];
    let es_units = cfg.ffn_es.map(|es| es.units(cfg.hidden_size) as i32);
    let px = cfg.parallax.then_some(cfg.head_dim() as i32);
    encoder_specs(
        &mut out,
        "blk.",
        cfg.state_layers,
        d,
        ff,
        cfg.depth(),
        es_units,
        px,
    );
    if cfg.state_loops > 1 {
        // SPEC-S3 §2: block 0's attention norm from the second loop on, and
        // the per-loop embedding added before each loop l >= 1.
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
    let last = cfg.n_exits() - 1;
    if parts.decision && cfg.temporal_carry {
        // SPEC-S3 §7: zero gate, so a fresh carry model equals its cold one.
        out.push(("carry_gate.weight".to_string(), vec![d], Init::Zeros));
        out.push(("carry_norm.weight".to_string(), vec![d], Init::Ones));
    }
    if parts.decision {
        out.push((
            "type_embd.weight".to_string(),
            vec![3, d],
            Init::TruncNormal(0.02),
        ));
        encoder_specs(
            &mut out,
            "static.blk.",
            cfg.static_layers,
            d,
            ff,
            cfg.static_layers,
            es_units,
            px,
        );
        out.push(("static_norm.weight".to_string(), vec![d], Init::Ones));
        for e in 0..cfg.n_exits() {
            out.push((format!("exit.{e}.norm.weight"), vec![d], Init::Ones));
        }
        for n in 0..cfg.fusion_layers {
            let p = format!("fusion.{n}");
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
        out.push(("cls.norm.weight".to_string(), vec![d], Init::Ones));
        out.push(("cls.norm.bias".to_string(), vec![d], Init::Zeros));
        out.push(("cls.weight".to_string(), vec![d, d], lin(d)));
        out.push(("cls.bias".to_string(), vec![d], Init::Zeros));
        out.push(("cls.output.weight".to_string(), vec![1, d], lin(d)));
        out.push(("cls.output.bias".to_string(), vec![1], Init::Zeros));
    } else {
        out.push((format!("exit.{last}.norm.weight"), vec![d], Init::Ones));
    }
    if parts.mlm {
        let std_out = 0.02 / (2.0 * cfg.depth() as f32).sqrt();
        out.push((
            "mlm.dense.weight".to_string(),
            vec![d, d],
            Init::TruncNormal(std_out),
        ));
        out.push(("mlm.norm.weight".to_string(), vec![d], Init::Ones));
        out.push(("mlm.decoder.bias".to_string(), vec![v], Init::Zeros));
        out.push((
            "mlm.num_mask.weight".to_string(),
            vec![d],
            Init::TruncNormal(0.02),
        ));
        out.push(("mlm_num.weight".to_string(), vec![1, d], lin(d)));
        out.push(("mlm_num.bias".to_string(), vec![1], Init::Zeros));
    }
    out
}

/// Names and shapes of every parameter.
pub fn param_shapes(cfg: &S2Config, parts: S2Parts) -> Vec<(String, Vec<i32>)> {
    specs(cfg, parts)
        .into_iter()
        .map(|(n, s, _)| (n, s))
        .collect()
}

/// Expected number of scalars.
pub fn expected_count(cfg: &S2Config, parts: S2Parts) -> usize {
    specs(cfg, parts)
        .iter()
        .map(|(_, s, _)| s.iter().product::<i32>() as usize)
        .sum()
}

/// Initializes a fresh parameter map (f32) deterministically from `seed`.
pub fn init(cfg: &S2Config, parts: S2Parts, seed: u64) -> Result<ParamMap> {
    let mut key = random::key(seed)?;
    let mut map = ParamMap::new();
    for (name, shape, init) in specs(cfg, parts) {
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

/// Turns a pretrained stack (`blk.*`, `exit.{last}.norm`) into the decision
/// initialization: the static encoder starts as a copy of the first
/// `static_layers` blocks and every exit norm and `static_norm` as a copy of
/// the stack's final norm. Missing parameters only are filled; returns how
/// many were copied.
pub fn expand_pretrained(dst: &mut ParamMap, src: &ParamMap, cfg: &S2Config) -> Result<usize> {
    let mut n = crate::params::load_matching(dst, src)?;
    let last = cfg.n_exits() - 1;
    let final_norm = format!("exit.{last}.norm.weight");
    let mut copy = |to: String, from: &str| {
        if !src.contains_key(to.as_str()) {
            if let Some(v) = src.get(from) {
                dst.insert(to.into(), v.deep_clone());
                n += 1;
            }
        }
    };
    for l in 0..cfg.static_layers {
        for m in BLOCK_MATRICES.iter().chain(&[
            "attn_norm.weight",
            "ffn_norm.weight",
            "attn_px_norm.weight",
        ]) {
            copy(format!("static.blk.{l}.{m}"), &format!("blk.{l}.{m}"));
        }
    }
    copy("static_norm.weight".into(), &final_norm);
    for e in 0..last {
        copy(format!("exit.{e}.norm.weight"), &final_norm);
    }
    Ok(n)
}

// ----------------------------------------------------------------------------- ternary

/// Ternary fake-quantization of a matrix (per output row absmean):
/// `γ_r = mean|W_r|`, `Wq = clip(round(W / γ_r), -1, 1) · γ_r`.
pub fn ternarize(w: &Array) -> R<Array> {
    let gamma = ops::maximum(ops::abs(w)?.mean_axis(-1, true)?, Array::from_f32(1e-8))?;
    ops::clip(ops::round(w.divide(&gamma)?, None)?, (-1.0f32, 1.0f32))?.multiply(&gamma)
}

/// `ternarize` with a straight-through gradient.
pub fn ternarize_ste(w: &Array) -> R<Array> {
    w.add(stop_gradient(ternarize(w)?.subtract(w)?)?)
}

/// Host ternarization of a row-major `[rows, cols]` matrix: codes in {-1, 0, 1}
/// and the per-row scale.
pub fn ternary_codes(w: &[f32], rows: usize, cols: usize) -> (Vec<i8>, Vec<f32>) {
    let mut codes = Vec::with_capacity(w.len());
    let mut scales = Vec::with_capacity(rows);
    for r in 0..rows {
        let row = &w[r * cols..(r + 1) * cols];
        let g = (row.iter().map(|x| x.abs()).sum::<f32>() / cols as f32).max(1e-8);
        scales.push(g);
        codes.extend(row.iter().map(|x| (x / g).round().clamp(-1.0, 1.0) as i8));
    }
    (codes, scales)
}

/// The parameters as used by the forward pass: fake-quantized when `qat`,
/// cast to bf16 when `cfg.bf16`.
fn prepare<'a>(p: &'a ParamMap, cfg: &S2Config, qat: bool) -> R<Cow<'a, ParamMap>> {
    if !cfg.bf16 && !qat {
        return Ok(Cow::Borrowed(p));
    }
    let mut out = ParamMap::with_capacity(p.len());
    for (k, v) in p {
        let mut v = if qat && is_ternary(k) {
            ternarize_ste(v)?
        } else {
            v.clone()
        };
        if cfg.bf16 && v.dtype() == Dtype::Float32 {
            v = v.as_dtype(Dtype::Bfloat16)?;
        }
        out.insert(k.clone(), v);
    }
    Ok(Cow::Owned(out))
}

// ----------------------------------------------------------------------------- forward

fn w<'a>(p: &'a ParamMap, name: &str) -> R<&'a Array> {
    p.get(name)
        .ok_or_else(|| Exception::custom(format!("missing parameter {name}")))
}

fn cast_like(a: Array, like: &Array) -> R<Array> {
    if a.dtype() == like.dtype() {
        Ok(a)
    } else {
        a.as_dtype(like.dtype())
    }
}

fn linear(x: &Array, weight: &Array, bias: Option<&Array>) -> R<Array> {
    let y = ops::matmul(x, weight.t())?;
    match bias {
        Some(b) => y.add(b),
        None => Ok(y),
    }
}

fn to_f32(x: Array) -> R<Array> {
    if x.dtype() == Dtype::Float32 {
        Ok(x)
    } else {
        x.as_dtype(Dtype::Float32)
    }
}

/// `[B, T, H * hd]` -> `[B, H, T, hd]`.
fn heads(x: &Array, h: i32) -> R<Array> {
    let (b, t) = (x.dim(0), x.dim(1));
    let hd = x.dim(2) / h;
    x.reshape(&[b, t, h, hd])?.transpose_axes(&[0, 2, 1, 3])
}

fn merge_heads(x: &Array) -> R<Array> {
    let (b, h, t, hd) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3));
    x.transpose_axes(&[0, 2, 1, 3])?.reshape(&[b, t, h * hd])
}

fn attend(q: &Array, k: &Array, v: &Array, mask: &Array) -> R<Array> {
    let hd = q.dim(3);
    fast::scaled_dot_product_attention(q, k, v, 1.0 / (hd as f32).sqrt(), mask, None)
}

/// Parallax attention (SPEC-S6 §1, arXiv 2605.29157 eq. 14): softmax attention
/// minus the softmax-weighted covariance of the values with the probe scores
/// `s_ij = ρ_i · k_j` (unscaled). With `P = softmax(q kᵀ / √hd)`,
/// `O1 = P v`, `P2 = P ⊙ (ρ kᵀ)`, `O2 = P2 v` and `d2 = rowsum(P2)`:
/// `out = O1 · (1 + d2) − O2`; `ρ = 0` is softmax attention exactly.
fn attend_parallax(q: &Array, k: &Array, v: &Array, rho: &Array, mask: &Array) -> R<Array> {
    let hd = q.dim(3);
    let kt = k.transpose_axes(&[0, 1, 3, 2])?;
    let scores = ops::matmul(q, &kt)?.multiply(Array::from_f32(1.0 / (hd as f32).sqrt()))?;
    let scores = ops::select(mask, to_f32(scores)?, Array::from_f32(f32::NEG_INFINITY))?;
    let p = cast_like(ops::softmax_axis(&scores, -1, true)?, v)?;
    let p2 = p.multiply(ops::matmul(rho, &kt)?)?;
    let o1 = ops::matmul(&p, v)?;
    let o2 = ops::matmul(&p2, v)?;
    let d2 = p2.sum_axis(-1, true)?;
    o1.multiply(d2.add(Array::from_f32(1.0))?)?.subtract(&o2)
}

/// RoPE flavour of an encoder.
enum Rope {
    /// Positions `0..T` (fast kernel): `(theta, head_dim)`.
    Sequential(f32, i32),
    /// Explicit positions: `(cos, sin)` tables `[B, 1, T, hd / 2]`.
    Table(Array, Array),
}

/// cos / sin tables of rotate_half RoPE for explicit positions `[B, T]`.
fn rope_tables(pos: &Array, theta: f32, hd: i32, like: &Array) -> R<(Array, Array)> {
    let half = hd / 2;
    let inv: Vec<f32> = (0..half)
        .map(|i| theta.powf(-(2.0 * i as f32) / hd as f32))
        .collect();
    let (b, t) = (pos.dim(0), pos.dim(1));
    let ang = pos
        .as_dtype(Dtype::Float32)?
        .reshape(&[b, 1, t, 1])?
        .multiply(Array::from_slice(&inv, &[half]))?;
    Ok((
        cast_like(ops::cos(&ang)?, like)?,
        cast_like(ops::sin(&ang)?, like)?,
    ))
}

fn apply_rope(x: &Array, rope: &Rope) -> R<Array> {
    match rope {
        Rope::Sequential(theta, hd) => fast::rope(x, *hd, false, *theta, 1.0, 0, None),
        Rope::Table(c, s) => {
            let parts = x.split_at_indices(&[x.dim(3) / 2], -1)?;
            let (x1, x2) = (&parts[0], &parts[1]);
            ops::concatenate(
                &[
                    x1.multiply(c)?.subtract(x2.multiply(s)?)?,
                    x2.multiply(c)?.add(x1.multiply(s)?)?,
                ],
                -1,
            )
        }
    }
}

/// One encoder block (pre-LN, fused QKV, RoPE, masked attention, GeGLU).
/// `attn_norm` names the attention norm (`None`: identity, as for layer 0).
#[allow(clippy::too_many_arguments)]
fn block(
    p: &ParamMap,
    cfg: &S2Config,
    pre: &str,
    attn_norm: Option<&str>,
    x: Array,
    mask: &Array,
    rope: &Rope,
) -> R<Array> {
    let eps = cfg.norm_eps;
    let h = cfg.n_heads() as i32;
    let d = cfg.hidden_size as i32;
    let y = match attn_norm {
        None => x.clone(),
        Some(n) => layer_norm(&x, w(p, &format!("{pre}.{n}.weight"))?, None, eps)?,
    };
    let qkv = linear(&y, w(p, &format!("{pre}.attn_qkv.weight"))?, None)?;
    let parts = qkv.split_at_indices(&[d, 2 * d, 3 * d], -1)?;
    let q = apply_rope(&heads(&parts[0], h)?, rope)?;
    let k = apply_rope(&heads(&parts[1], h)?, rope)?;
    let v = heads(&parts[2], h)?;
    let a = if cfg.parallax {
        // SPEC-S6 §1: ρ is RMS-normalized per head, then rotated like q.
        let norm = w(p, &format!("{pre}.attn_px_norm.weight"))?;
        let rho = fast::rms_norm(heads(&parts[3], h)?, Some(norm), eps)?;
        attend_parallax(&q, &k, &v, &apply_rope(&rho, rope)?, mask)?
    } else {
        attend(&q, &k, &v, mask)?
    };
    let a = merge_heads(&a)?;
    let x = x.add(linear(
        &a,
        w(p, &format!("{pre}.attn_output.weight"))?,
        None,
    )?)?;
    let y = layer_norm(&x, w(p, &format!("{pre}.ffn_norm.weight"))?, None, eps)?;
    let m = match &cfg.ffn_es {
        Some(es) => es_sparse(cfg, es, &y)?,
        None => {
            let u = linear(&y, w(p, &format!("{pre}.ffn_up.weight"))?, None)?;
            let g = u.split_at_indices(&[cfg.intermediate_size as i32], -1)?;
            gelu_erf(&g[0])?.multiply(&g[1])?
        }
    };
    x.add(linear(&m, w(p, &format!("{pre}.ffn_down.weight"))?, None)?)
}

/// SPEC-S3 §9: `kWTA(P · y)` with the fixed seeded sparse ±1 projection `P` (`s2::es`). Each
/// token keeps the units `>=` its k-th largest pre-activation; the threshold carries no
/// gradient, the kept units pass theirs to `y`.
fn es_sparse(cfg: &S2Config, es: &crate::s2::config::EsFfn, y: &Array) -> R<Array> {
    let d = cfg.hidden_size;
    let (units, k) = (es.units(d), es.k(d));
    let proj = crate::s2::es::dense(es.seed, d, units, es.fan_in);
    let proj = cast_like(Array::from_slice(&proj, &[units as i32, d as i32]), y)?;
    let u = ops::matmul(y, proj.t())?;
    let kth = (units - k) as i32;
    let sorted = ops::partition_axis(stop_gradient(&u)?, kth, -1)?;
    let thr = sorted.take_axis(Array::from_slice(&[kth], &[1]), -1)?;
    let keep = u.ge(&thr)?;
    u.multiply(cast_like(keep.as_dtype(Dtype::Float32)?, &u)?)
}

/// `e = tok_embd[id] + has_value · (num_proj · φ(v) + num_proj.bias)`;
/// `has_value` (f32 0/1) is `id == [NUM]` without number fusion.
fn embed(p: &ParamMap, ids: &Array, values: &Array, has: &Array) -> R<Array> {
    let table = w(p, "token_embd.weight")?;
    let e = table.take_axis(ids, 0)?;
    let phi = cast_like(features(values)?, table)?;
    let num = linear(&phi, w(p, "num_proj.weight")?, Some(w(p, "num_proj.bias")?))?;
    let has = cast_like(has.as_dtype(Dtype::Float32)?, table)?;
    let shape = [ids.dim(0), ids.dim(1), 1];
    e.add(num.multiply(has.reshape(&shape)?)?)
}

/// Key-padding mask `[B, 1, T, T]` (a query always sees itself).
fn padding_mask(valid: &Array) -> R<Array> {
    let (b, t) = (valid.dim(0), valid.dim(1));
    let pos = Array::from_iter(0..t, &[t]);
    let eye = pos.reshape(&[t, 1])?.eq(pos.reshape(&[1, t])?)?;
    valid.reshape(&[b, 1, 1, t])?.logical_or(&eye)
}

/// Static-stream mask `[U, 1, T, T]` from segment ids (-1 pad, 0 question,
/// i+1 option i): Q attends Q, O_i attends Q ∪ O_i, padding sees only itself.
pub fn static_mask(seg: &Array) -> R<Array> {
    let (u, t) = (seg.dim(0), seg.dim(1));
    let sq = seg.reshape(&[u, 1, t, 1])?;
    let sk = seg.reshape(&[u, 1, 1, t])?;
    let to_q = sk
        .eq(Array::from_int(0))?
        .logical_and(&sq.ge(Array::from_int(0))?)?;
    let same = sq.eq(&sk)?.logical_and(&sq.gt(Array::from_int(0))?)?;
    let pos = Array::from_iter(0..t, &[t]);
    let eye = pos.reshape(&[t, 1])?.eq(pos.reshape(&[1, t])?)?;
    to_q.logical_or(&same)?.logical_or(&eye)
}

/// Static encoder: `[U, Ts]` -> `[U, Ts, d]` after `static_norm`.
#[allow(clippy::too_many_arguments)]
fn static_encode(
    p: &ParamMap,
    cfg: &S2Config,
    ids: &Array,
    values: &Array,
    has: &Array,
    seg: &Array,
    pos: &Array,
) -> R<Array> {
    let mut x = embed(p, ids, values, has)?;
    let mask = static_mask(seg)?;
    let (c, s) = rope_tables(pos, cfg.rope_theta, cfg.head_dim() as i32, &x)?;
    let rope = Rope::Table(c, s);
    for l in 0..cfg.static_layers {
        let norm = (l > 0).then_some("attn_norm");
        x = block(p, cfg, &format!("static.blk.{l}"), norm, x, &mask, &rope)?;
    }
    layer_norm(&x, w(p, "static_norm.weight")?, None, cfg.norm_eps)
}

/// State encoder over embedded inputs `x`: the normalized hidden states
/// `H_e` of the requested exits (ascending exit indices).
///
/// SPEC-S3 §2: application `a` (0-based) runs block `a % B` of loop `a / B`;
/// before each loop `l >= 1`, `loop_embd[l]` is added to the hidden states;
/// block 0 has no attention norm on the very first application and uses
/// `blk.0.attn_norm_loop` afterwards. Exits count block applications.
fn state_exits(
    p: &ParamMap,
    cfg: &S2Config,
    x: Array,
    valid: &Array,
    exits: &[usize],
) -> R<Vec<Array>> {
    Ok(state_run(p, cfg, x, valid, exits)?
        .into_iter()
        .map(|(h, _)| h)
        .collect())
}

/// [`state_exits`] that also returns, per exit, the raw hidden states after
/// its block application (before the exit norm): the temporal carry source.
fn state_run(
    p: &ParamMap,
    cfg: &S2Config,
    mut x: Array,
    valid: &Array,
    exits: &[usize],
) -> R<Vec<(Array, Array)>> {
    let mask = padding_mask(valid)?;
    let rope = Rope::Sequential(cfg.rope_theta, cfg.head_dim() as i32);
    let last_app = exits
        .iter()
        .map(|&e| cfg.exits[e])
        .max()
        .unwrap_or(cfg.depth());
    let nb = cfg.state_layers;
    let mut out = Vec::with_capacity(exits.len());
    for a in 0..last_app {
        let (lp, l) = (a / nb, a % nb);
        if l == 0 && lp > 0 {
            let row = w(p, "loop_embd.weight")?.index(lp as i32);
            x = x.add(cast_like(row, &x)?)?;
        }
        let norm = match (a, l) {
            (0, _) => None,
            (_, 0) => Some("attn_norm_loop"),
            _ => Some("attn_norm"),
        };
        x = block(p, cfg, &format!("blk.{l}"), norm, x, &mask, &rope)?;
        for &e in exits {
            if cfg.exits[e] == a + 1 {
                out.push((
                    layer_norm(
                        &x,
                        w(p, &format!("exit.{e}.norm.weight"))?,
                        None,
                        cfg.norm_eps,
                    )?,
                    x.clone(),
                ));
            }
        }
    }
    Ok(out)
}

/// Fusion head + scorer for every exit at once: `m` `[B, K, d]` markers,
/// `hs` the exits' `[B, S, d]`, `qt` `[B, Tq, d]` question tokens.
/// Returns logits `[E, B, K]` (f32).
#[allow(clippy::too_many_arguments)]
fn fusion(
    p: &ParamMap,
    cfg: &S2Config,
    m: &Array,
    qtype: &Array,
    hs: &[Array],
    s_valid: &Array,
    qt: &Array,
    q_valid: &Array,
) -> R<Array> {
    let (b, k, d) = (m.dim(0), m.dim(1), m.dim(2));
    let e = hs.len() as i32;
    let h = cfg.n_heads() as i32;
    let ln = |x: &Array, name: &str| -> R<Array> {
        layer_norm(
            x,
            w(p, &format!("{name}.weight"))?,
            Some(w(p, &format!("{name}.bias"))?),
            1e-5,
        )
    };
    let lin = |x: &Array, name: &str| -> R<Array> {
        linear(
            x,
            w(p, &format!("{name}.weight"))?,
            Some(w(p, &format!("{name}.bias"))?),
        )
    };
    let type_row = w(p, "type_embd.weight")?
        .take_axis(qtype, 0)?
        .reshape(&[b, 1, d])?;
    let x0 = m.add(&type_row)?;
    let mut x = ops::concatenate(&vec![x0; hs.len()], 0)?;
    let kv_parts: Vec<Array> = hs
        .iter()
        .map(|he| ops::concatenate(&[he, qt], 1))
        .collect::<R<_>>()?;
    let kv = ops::concatenate(&kv_parts, 0)?;
    let t = kv.dim(1);
    let kmask = ops::concatenate(&[s_valid, q_valid], 1)?;
    let kmask = ops::concatenate(&vec![kmask; hs.len()], 0)?.reshape(&[e * b, 1, 1, t])?;
    for n in 0..cfg.fusion_layers {
        let pre = format!("fusion.{n}");
        let q = heads(
            &lin(&ln(&x, &format!("{pre}.q_norm"))?, &format!("{pre}.attn_q"))?,
            h,
        )?;
        let kvp = lin(
            &ln(&kv, &format!("{pre}.kv_norm"))?,
            &format!("{pre}.attn_kv"),
        )?;
        let parts = kvp.split_at_indices(&[d], -1)?;
        let a = merge_heads(&attend(
            &q,
            &heads(&parts[0], h)?,
            &heads(&parts[1], h)?,
            &kmask,
        )?)?;
        x = x.add(lin(&a, &format!("{pre}.attn_output"))?)?;
        let f = lin(
            &ln(&x, &format!("{pre}.ffn_norm"))?,
            &format!("{pre}.ffn_up"),
        )?;
        let f = ops::maximum(&f, cast_like(Array::from_f32(0.0), &f)?)?;
        x = x.add(lin(&f, &format!("{pre}.ffn_down"))?)?;
    }
    let s = gelu_erf(&lin(&ln(&x, "cls.norm")?, "cls")?)?;
    let s = lin(&s, "cls.output")?;
    to_f32(s.reshape(&[e, b, k])?)
}

/// Raw marker logits `[E, B, K]` of every exit (`qat` = ternary fake-quant).
pub fn forward(p: &ParamMap, cfg: &S2Config, b: &S2Batch, qat: bool) -> R<Array> {
    forward_exits(p, cfg, b, qat, None)
}

/// [`forward`] over the first `n_exits` exits only (the state encoder stops
/// at the last of them): stochastic depth during training.
pub fn forward_exits(
    p: &ParamMap,
    cfg: &S2Config,
    b: &S2Batch,
    qat: bool,
    n_exits: Option<usize>,
) -> R<Array> {
    let p = &*prepare(p, cfg, qat)?;
    let n = n_exits.unwrap_or(cfg.n_exits()).clamp(1, cfg.n_exits());
    Ok(forward_prepared(p, cfg, b, n, None)?.0)
}

/// A frame's carry: the previous frame's raw final latent `[B, S, d]` and a
/// per-row mask `[B]` (0 = cold start).
pub struct Carry<'a> {
    pub h: &'a Array,
    pub on: &'a Array,
}

/// SPEC-S3 §7 forward of one frame: logits `[E, B, K]` and the raw latents
/// of every exit (`[B, S, d]` each, before the exit norms). The state encoder
/// starts from `embed(state) + on · carry_gate ⊙ LN(h; carry_norm)`.
pub fn forward_carry(
    p: &ParamMap,
    cfg: &S2Config,
    b: &S2Batch,
    qat: bool,
    carry: Option<Carry<'_>>,
) -> R<(Array, Vec<Array>)> {
    let p = &*prepare(p, cfg, qat)?;
    forward_prepared(p, cfg, b, cfg.n_exits(), carry)
}

fn forward_prepared(
    p: &ParamMap,
    cfg: &S2Config,
    b: &S2Batch,
    n: usize,
    carry: Option<Carry<'_>>,
) -> R<(Array, Vec<Array>)> {
    let st = static_encode(
        p,
        cfg,
        &b.st_ids,
        &b.st_values,
        &b.st_has,
        &b.st_seg,
        &b.st_pos,
    )?;
    let d = st.dim(2);
    let flat = st.reshape(&[-1, d])?;
    let (rows, k, tq) = (b.marker_idx.dim(0), b.marker_idx.dim(1), b.q_idx.dim(1));
    let m = flat
        .take_axis(b.marker_idx.reshape(&[-1])?, 0)?
        .reshape(&[rows, k, d])?;
    let qt = flat
        .take_axis(b.q_idx.reshape(&[-1])?, 0)?
        .reshape(&[rows, tq, d])?;
    let mut x = embed(p, &b.ids, &b.values, &b.has)?;
    if let Some(c) = carry {
        let ln = layer_norm(c.h, w(p, "carry_norm.weight")?, None, cfg.norm_eps)?;
        let term = ln.multiply(w(p, "carry_gate.weight")?)?;
        let on = cast_like(c.on.reshape(&[-1, 1, 1])?, &x)?;
        x = x.add(cast_like(term, &x)?.multiply(&on)?)?;
    }
    let used: Vec<usize> = (0..n).collect();
    let run = state_run(p, cfg, x, &b.valid, &used)?;
    let (hs, raw): (Vec<Array>, Vec<Array>) = run.into_iter().unzip();
    let logits = fusion(p, cfg, &m, &b.qtype, &hs, &b.valid, &qt, &b.q_valid)?;
    Ok((logits, raw))
}

/// SPEC-S3 §7 sequence loss over `K` frames of the same rows: frame 0 cold,
/// frame `t` warm from the detached latent of frame `t - 1` taken at the exit
/// `src[t - 1]` (one-hot `[K, E]`, simulating where the previous frame
/// stopped) unless `on[t]` (`[K, B]`) is 0 (dropped carry or different
/// state ids). Deep supervision (`wts` `[E]`) at every exit of every frame;
/// the mean over frames.
pub fn seq_loss(
    p: &ParamMap,
    cfg: &S2Config,
    frames: &[S2Batch],
    on: &Array,
    src: &Array,
    wts: &Array,
    qat: bool,
) -> R<Array> {
    let p = &*prepare(p, cfg, qat)?;
    let mut total: Option<Array> = None;
    let mut prev: Option<Array> = None;
    for (t, b) in frames.iter().enumerate() {
        let on_t = on.index(t as i32);
        let carry = prev.as_ref().map(|h| Carry { h, on: &on_t });
        let (logits, raw) = forward_prepared(p, cfg, b, cfg.n_exits(), carry)?;
        let logits = masked(&logits, &b.marker_mask)?;
        let logp = logits.subtract(logits.logsumexp_axis(-1, true)?)?;
        let ce = b
            .target
            .multiply(&logp)?
            .sum_axis(-1, None)?
            .mean_axis(-1, None)?
            .negative()?;
        let l = ce.multiply(wts)?.sum(None)?;
        total = Some(match total {
            Some(x) => x.add(&l)?,
            None => l,
        });
        let sel = src.index(t as i32);
        let mut h: Option<Array> = None;
        for (e, r) in raw.iter().enumerate() {
            let term = r.multiply(cast_like(sel.index(e as i32), r)?)?;
            h = Some(match h {
                Some(x) => x.add(&term)?,
                None => term,
            });
        }
        prev = Some(stop_gradient(h.expect("at least one exit"))?);
    }
    total
        .expect("at least one frame")
        .divide(Array::from_f32(frames.len() as f32))
}

/// Logits with invalid options pushed to -1e4 (`mask` `[B, K]` broadcast over exits).
pub fn masked(logits: &Array, mask: &Array) -> R<Array> {
    ops::select(mask, logits, Array::from_f32(-1e4))
}

/// Deep supervision: `Σ_e w_e · CE(exit e)` against the soft target, over
/// every exit (`w = linspace(0.5, 1)`, normalized).
pub fn decision_loss(p: &ParamMap, cfg: &S2Config, b: &S2Batch, qat: bool) -> R<Array> {
    decision_loss_exits(p, cfg, b, qat, None)
}

/// Deep-supervision weights over the first `n_exits` exits (the config's
/// `linspace(0.5, 1)` weights renormalized), zero for the later exits.
pub fn exit_weight_vec(cfg: &S2Config, n_exits: Option<usize>) -> Vec<f32> {
    let n = n_exits.unwrap_or(cfg.n_exits()).clamp(1, cfg.n_exits());
    let all = cfg.exit_weights();
    let total: f32 = all[..n].iter().sum();
    (0..cfg.n_exits())
        .map(|e| if e < n { all[e] / total } else { 0.0 })
        .collect()
}

/// Deep supervision with explicit per-exit weights `wts` `[E]` (every exit
/// is computed). A zero weight on the later exits gives them, and the loops
/// that only feed them, no gradient: the same update as skipping those loops,
/// but with a single compiled graph per batch shape.
pub fn decision_loss_weighted(
    p: &ParamMap,
    cfg: &S2Config,
    b: &S2Batch,
    qat: bool,
    wts: &Array,
) -> R<Array> {
    let logits = masked(&forward(p, cfg, b, qat)?, &b.marker_mask)?;
    let logp = logits.subtract(logits.logsumexp_axis(-1, true)?)?;
    let ce = b
        .target
        .multiply(&logp)?
        .sum_axis(-1, None)?
        .mean_axis(-1, None)?
        .negative()?;
    ce.multiply(wts)?.sum(None)
}

/// Per-exit sharpening exponents of [`exit_targets`]: `beta0` at the first exit rising
/// linearly to 1 at the last.
pub fn dis_betas(n_exits: usize, beta0: f32) -> Vec<f32> {
    (0..n_exits)
        .map(|e| {
            if n_exits == 1 {
                1.0
            } else {
                beta0 + (1.0 - beta0) * e as f32 / (n_exits - 1) as f32
            }
        })
        .collect()
}

/// Deep improvement supervision targets (s4): exit `e` is trained toward
/// `normalize(target ^ beta_e)` `[E, B, O]`, a tempered copy of the target that is
/// softer at the early exits (`beta < 1`) and equal to it at the last one (`beta = 1`).
/// Every loop is supervised toward a target that improves on the previous one, so
/// an early exit is confident only where even the softened target is.
pub fn exit_targets(target: &Array, beta: &Array) -> R<Array> {
    let e = beta.dim(0);
    let t = ops::power(target, beta.reshape(&[e, 1, 1])?)?;
    t.divide(ops::maximum(t.sum_axis(-1, true)?, Array::from_f32(1e-12))?)
}

/// [`decision_loss_weighted`] against the per-exit [`exit_targets`] (`beta` `[E]`).
pub fn decision_loss_dis(
    p: &ParamMap,
    cfg: &S2Config,
    b: &S2Batch,
    qat: bool,
    wts: &Array,
    beta: &Array,
) -> R<Array> {
    let logits = masked(&forward(p, cfg, b, qat)?, &b.marker_mask)?;
    let logp = logits.subtract(logits.logsumexp_axis(-1, true)?)?;
    let ce = exit_targets(&b.target, beta)?
        .multiply(&logp)?
        .sum_axis(-1, None)?
        .mean_axis(-1, None)?
        .negative()?;
    ce.multiply(wts)?.sum(None)
}

/// Self-distillation of the early exits toward the deepest one (s6, after CalexNet
/// and SAFE-KD): each exit `e < E − 1` also learns `T² · KL(p_T(last) ‖ p_T(e))`, with
/// `p_T = softmax(logits / T)` and the deepest exit detached, so an early exit learns
/// where the full-depth answer is uncertain instead of only the hard label.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ExitKd {
    /// Weight of the mean KL over the early exits.
    pub weight: f32,
    /// Distillation temperature `T`.
    pub temp: f32,
}

/// The [`ExitKd`] term of masked logits `[E, B, K]` (0 with a single exit).
pub fn exit_kd_loss(logits: &Array, kd: ExitKd) -> R<Array> {
    let e = logits.dim(0);
    if e < 2 {
        return Ok(Array::from_f32(0.0));
    }
    let z = to_f32(logits.clone())?.divide(Array::from_f32(kd.temp))?;
    let logq = z.subtract(z.logsumexp_axis(-1, true)?)?;
    let teacher = stop_gradient(&logq.index(e - 1))?; // [B, K]
    let p = ops::exp(&teacher)?;
    let kl = p
        .multiply(teacher.subtract(logq.index(..e - 1))?)?
        .sum_axis(-1, None)?
        .mean_axis(-1, None)?; // [E - 1]
    kl.mean(None)?
        .multiply(Array::from_f32(kd.weight * kd.temp * kd.temp))
}

/// Deep supervision with explicit exit weights `wts` `[E]`, optional [`exit_targets`]
/// exponents `beta` `[E]`, plus the [`ExitKd`] term: [`decision_loss_weighted`] or
/// [`decision_loss_dis`] with self-distillation, from one forward pass.
pub fn decision_loss_kd(
    p: &ParamMap,
    cfg: &S2Config,
    b: &S2Batch,
    qat: bool,
    wts: &Array,
    beta: Option<&Array>,
    kd: ExitKd,
) -> R<Array> {
    let logits = masked(&forward(p, cfg, b, qat)?, &b.marker_mask)?;
    let logp = logits.subtract(logits.logsumexp_axis(-1, true)?)?;
    let target = match beta {
        Some(beta) => exit_targets(&b.target, beta)?,
        None => b.target.clone(),
    };
    let ce = target
        .multiply(&logp)?
        .sum_axis(-1, None)?
        .mean_axis(-1, None)?
        .negative()?;
    ce.multiply(wts)?.sum(None)?.add(exit_kd_loss(&logits, kd)?)
}

/// [`decision_loss`] over the first `n_exits` exits (stochastic depth: the
/// loops beyond them are skipped), their weights renormalized to sum to 1.
pub fn decision_loss_exits(
    p: &ParamMap,
    cfg: &S2Config,
    b: &S2Batch,
    qat: bool,
    n_exits: Option<usize>,
) -> R<Array> {
    let n = n_exits.unwrap_or(cfg.n_exits()).clamp(1, cfg.n_exits());
    let logits = masked(&forward_exits(p, cfg, b, qat, Some(n))?, &b.marker_mask)?;
    let logp = logits.subtract(logits.logsumexp_axis(-1, true)?)?;
    let ce = b
        .target
        .multiply(&logp)?
        .sum_axis(-1, None)?
        .mean_axis(-1, None)?
        .negative()?; // [E]
    let all = cfg.exit_weights();
    let total: f32 = all[..n].iter().sum();
    let wv: Vec<f32> = all[..n].iter().map(|x| x / total).collect();
    let wts = Array::from_slice(&wv, &[n as i32]);
    ce.multiply(&wts)?.sum(None)
}

/// Weight of the number MSE in the numeric MLM loss (SPEC-S2 §4).
pub const NUM_LOSS_WEIGHT: f32 = 0.5;

/// Numeric MLM loss: CE on masked non-number tokens + `num_weight` · MSE on
/// the signlog value of masked numbers (xVal-style `mlm_num` head).
pub fn mlm_loss(p: &ParamMap, cfg: &S2Config, b: &NumMlmBatch, num_weight: f32) -> R<Array> {
    let (ce, mse) = mlm_loss_parts(p, cfg, b)?;
    ce.add(mse.multiply(Array::from_f32(num_weight))?)
}

/// The two terms of [`mlm_loss`]: (token CE, number MSE).
pub fn mlm_loss_parts(p: &ParamMap, cfg: &S2Config, b: &NumMlmBatch) -> R<(Array, Array)> {
    let p = &*prepare(p, cfg, false)?;
    let mut x = embed(p, &b.ids, &b.values, &b.has)?;
    let nm = w(p, "mlm.num_mask.weight")?;
    let flag = cast_like(b.num_flag.clone(), nm)?;
    x = x.add(
        flag.reshape(&[b.ids.dim(0), b.ids.dim(1), 1])?
            .multiply(nm)?,
    )?;
    let last = cfg.n_exits() - 1;
    let enc = state_exits(p, cfg, x, &b.valid, &[last])?
        .pop()
        .expect("last exit");
    let d = enc.dim(2);
    let flat = enc.reshape(&[-1, d])?;
    // Tokens.
    let rows = flat.take_axis(&b.positions, 0)?;
    let h = gelu_erf(&linear(&rows, w(p, "mlm.dense.weight")?, None)?)?;
    let h = layer_norm(&h, w(p, "mlm.norm.weight")?, None, cfg.norm_eps)?;
    let logits = to_f32(linear(
        &h,
        w(p, "token_embd.weight")?,
        Some(w(p, "mlm.decoder.bias")?),
    )?)?;
    let m = logits.dim(0);
    let lse = logits.logsumexp_axis(-1, None)?;
    let gold = logits
        .take_along_axis(b.labels.reshape(&[m, 1])?, -1)?
        .reshape(&[m])?;
    let ce = lse.subtract(&gold)?.multiply(&b.weights)?.sum(None)?;
    let ce = ce.divide(ops::maximum(b.weights.sum(None)?, Array::from_f32(1.0))?)?;
    // Numbers.
    let nrows = flat.take_axis(&b.num_positions, 0)?;
    let pred = to_f32(linear(
        &nrows,
        w(p, "mlm_num.weight")?,
        Some(w(p, "mlm_num.bias")?),
    )?)?
    .reshape(&[-1])?;
    let se = pred
        .subtract(&b.num_targets)?
        .square()?
        .multiply(&b.num_weights)?
        .sum(None)?;
    let mse = se.divide(ops::maximum(
        b.num_weights.sum(None)?,
        Array::from_f32(1.0),
    )?)?;
    Ok((ce, mse))
}

/// Splits the logits of [`forward`] into host vectors `[E][B][k_b]`.
pub fn logits_per_row(logits: &Array, ks: &[usize]) -> R<Vec<Vec<Vec<f32>>>> {
    let (e, b, k) = (
        logits.dim(0) as usize,
        logits.dim(1) as usize,
        logits.dim(2) as usize,
    );
    let flat = crate::model::host_f32(logits)?;
    Ok((0..e)
        .map(|ei| {
            (0..b)
                .map(|r| flat[(ei * b + r) * k..(ei * b + r) * k + ks[r]].to_vec())
                .collect()
        })
        .collect())
}

/// Index helper used by tests.
#[doc(hidden)]
pub fn exit_logits(logits: &Array, e: i32) -> Array {
    logits.index(e)
}

#[cfg(test)]
mod tests {
    use std::{collections::HashMap, rc::Rc};

    use mlx_rs::transforms::keyed_value_and_grad;

    use super::*;
    use crate::{
        num::NUM_ID,
        rng::Rng,
        s2::data::{encode_record, make_batch, tests::s2_tok, S2Item, StaticSeq, Stream},
        sequence::DecisionRecord,
        test_util,
    };

    fn ladder(name: &str) -> S2Config {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join(format!("../../configs/{name}.json"));
        S2Config::load(path).unwrap()
    }

    #[test]
    fn exact_parameter_counts() {
        let _g = test_util::lock();
        // (config, decision model, pretraining model)
        let expected = [
            ("s2-pico", 310_337, 195_265),
            ("s2-nano", 1_532_033, 939_649),
            ("s2-micro", 8_079_105, 5_058_817),
        ];
        for (name, total, pre) in expected {
            let cfg = ladder(name);
            assert_eq!(expected_count(&cfg, S2Parts::DECISION), total, "{name}");
            assert_eq!(expected_count(&cfg, S2Parts::PRETRAIN), pre, "{name}");
            let p = init(&cfg, S2Parts::DECISION, 0).unwrap();
            assert_eq!(crate::params::count(&p), total, "{name}");
            let d = cfg.hidden_size;
            assert_eq!(
                p["blk.0.attn_qkv.weight"].shape(),
                &[3 * d as i32, d as i32]
            );
            assert_eq!(p["num_proj.weight"].shape(), &[d as i32, 17]);
            assert!(
                !p.contains_key("blk.0.attn_norm.weight")
                    && p.contains_key("blk.1.attn_norm.weight")
            );
            assert!(!p.contains_key("static.blk.0.attn_norm.weight"));
            assert_eq!(
                p["fusion.0.attn_kv.weight"].shape(),
                &[2 * d as i32, d as i32]
            );
            assert_eq!(
                p["fusion.0.ffn_down.weight"].shape(),
                &[d as i32, 2 * d as i32]
            );
            assert_eq!(p["cls.output.bias"].shape(), &[1]);
            println!("{name}: decision model {total}, pretraining stack {pre}");
        }
    }

    #[test]
    fn s3_parameter_counts_and_loop_structure() {
        let _g = test_util::lock();
        // (config, decision model, pretraining model)
        for (name, total, pre) in [
            ("s3-pico", 445_345, 203_393),
            ("s3-nano", 1_204_353, 611_969),
        ] {
            let cfg = ladder(name);
            assert!(cfg.fuse_numbers && cfg.state_loops > 1);
            assert_eq!(expected_count(&cfg, S2Parts::DECISION), total, "{name}");
            assert_eq!(expected_count(&cfg, S2Parts::PRETRAIN), pre, "{name}");
            let p = init(&cfg, S2Parts::DECISION, 0).unwrap();
            assert_eq!(crate::params::count(&p), total, "{name}");
            let d = cfg.hidden_size as i32;
            // Only the B distinct blocks exist; the loops share them.
            for l in 0..cfg.state_layers {
                assert!(p.contains_key(format!("blk.{l}.attn_qkv.weight").as_str()));
            }
            assert!(!p.contains_key(format!("blk.{}.attn_qkv.weight", cfg.state_layers).as_str()));
            assert!(!p.contains_key("blk.0.attn_norm.weight"));
            assert_eq!(p["blk.0.attn_norm_loop.weight"].shape(), &[d]);
            assert_eq!(p["loop_embd.weight"].shape(), &[cfg.state_loops as i32, d]);
            for e in 0..cfg.n_exits() {
                assert!(p.contains_key(format!("exit.{e}.norm.weight").as_str()));
            }
            assert_eq!(cfg.head_dim() * cfg.n_heads(), cfg.hidden_size);
            let pp = init(&cfg, S2Parts::PRETRAIN, 0).unwrap();
            assert!(
                pp.contains_key("loop_embd.weight")
                    && pp.contains_key("blk.0.attn_norm_loop.weight")
            );
            println!("{name}: decision model {total}, pretraining stack {pre}");
        }
        // Without loops none of the new tensors exist (s2 unchanged).
        let p = init(&S2Config::tiny(262), S2Parts::DECISION, 0).unwrap();
        assert!(
            !p.contains_key("loop_embd.weight") && !p.contains_key("blk.0.attn_norm_loop.weight")
        );
        assert!(!is_ternary("loop_embd.weight") && !is_ternary("blk.0.attn_norm_loop.weight"));
        assert!(!is_hidden_matrix("blk.0.attn_norm_loop.weight"));
    }

    #[test]
    fn es_ffn_parameters_sparsity_and_forward() {
        let _g = test_util::lock();
        mlx_rs::with_device(mlx_rs::Device::cpu(), || {
            let es = crate::s2::config::EsFfn {
                seed: 3,
                expand: 8,
                k_frac: 0.05,
                fan_in: 8,
            };
            let cfg = S2Config {
                ffn_es: Some(es),
                ..looped(1, 2)
            };
            let (units, k) = (es.units(96), es.k(96));
            assert_eq!((units, k), (768, 38));
            let p = init(&cfg, S2Parts::DECISION, 0).unwrap();
            assert_eq!(
                crate::params::count(&p),
                expected_count(&cfg, S2Parts::DECISION)
            );
            for b in ["blk.0", "static.blk.0"] {
                assert!(!p.contains_key(format!("{b}.ffn_up.weight").as_str()));
                assert_eq!(
                    p[format!("{b}.ffn_down.weight").as_str()].shape(),
                    &[96, 768]
                );
            }
            assert!(is_ternary("blk.0.ffn_down.weight"));
            // GeGLU counterpart: 2·ff·d + d·ff per block instead of d·units.
            let plain = expected_count(&looped(1, 2), S2Parts::DECISION);
            assert_eq!(
                expected_count(&cfg, S2Parts::DECISION) + 2 * 3 * 40 * 96,
                plain + 2 * 96 * 768
            );
            // k-WTA: exactly k active units per token (no ties on random inputs).
            let y = random::normal::<f32>(&[2, 5, 96], 0.0, 1.0, &random::key(1).unwrap()).unwrap();
            let h = es_sparse(&cfg, &es, &y).unwrap();
            let nz = h
                .ne(Array::from_f32(0.0))
                .unwrap()
                .as_dtype(Dtype::Int32)
                .unwrap();
            let per_row: Vec<i32> = nz.sum_axis(-1, false).unwrap().as_slice().to_vec();
            assert!(per_row.iter().all(|&n| n == k as i32), "{per_row:?}");
            // The kept units are the largest pre-activations of the fixed projection.
            let proj = crate::s2::es::dense(3, 96, 768, 8);
            let yv: Vec<f32> = y.as_slice().to_vec();
            let hv: Vec<f32> = h.as_slice().to_vec();
            let mut u: Vec<f32> = (0..768)
                .map(|j| (0..96).map(|i| proj[j * 96 + i] * yv[i]).sum())
                .collect();
            for j in 0..768 {
                if hv[j] != 0.0 {
                    assert!((hv[j] - u[j]).abs() < 1e-4);
                }
            }
            u.sort_by(|a, b| b.total_cmp(a));
            let kept_min = hv[..768]
                .iter()
                .filter(|&&v| v != 0.0)
                .fold(f32::MAX, |m, &v| m.min(v));
            assert!((kept_min - u[k - 1]).abs() < 1e-4);
            // The decision forward runs and depends on ffn_down.
            let (_d, tok) = s2_tok();
            let (st, state) = crate::s2::data::encode_record_cfg(
                &tok,
                &rec(
                    "choice",
                    "go 2?",
                    &["up 1", "down", "x"],
                    "vy -3 alt 12 gap 4",
                ),
                &cfg,
            )
            .unwrap();
            let statics = vec![st];
            let it = item(0, &state, 3, 0);
            let b = make_batch(&statics, &[&it], 1);
            let gp = crate::s2::golden::golden_params(&cfg, 4).unwrap();
            let a = logits_per_row(&forward(&gp, &cfg, &b, true).unwrap(), &[3]).unwrap();
            let mut zp = gp.clone();
            zp.insert(
                "blk.0.ffn_down.weight".into(),
                ops::zeros::<f32>(&[96, 768]).unwrap(),
            );
            let z = logits_per_row(&forward(&zp, &cfg, &b, true).unwrap(), &[3]).unwrap();
            assert!(a.iter().flatten().flatten().all(|v| v.is_finite()));
            assert!(a
                .iter()
                .flatten()
                .flatten()
                .zip(z.iter().flatten().flatten())
                .any(|(x, y)| (x - y).abs() > 1e-4));
        });
    }

    /// A tiny looped, fused configuration with head_dim 48.
    fn looped(b: usize, loops: usize) -> S2Config {
        S2Config {
            hidden_size: 96,
            head_count: Some(2),
            intermediate_size: 40,
            state_layers: b,
            state_loops: loops,
            static_layers: 1,
            fusion_layers: 1,
            exits: (1..=b * loops).collect(),
            fuse_numbers: true,
            ..S2Config::tiny(262)
        }
    }

    #[test]
    fn exits_count_block_applications() {
        let _g = test_util::lock();
        mlx_rs::with_device(mlx_rs::Device::cpu(), || {
            let (_d, tok) = s2_tok();
            // B = 2 blocks x 2 loops = 4 applications, one exit after each.
            let cfg = looped(2, 2);
            let p = crate::s2::golden::golden_params(&cfg, 4).unwrap();
            let (st, state) = crate::s2::data::encode_record_cfg(
                &tok,
                &rec(
                    "choice",
                    "go 2?",
                    &["up 1", "down", "x"],
                    "vy -3 alt 12 gap 4",
                ),
                &cfg,
            )
            .unwrap();
            let statics = vec![st];
            let it = item(0, &state, 3, 0);
            let b = make_batch(&statics, &[&it], 1);
            let base = logits_per_row(&forward(&p, &cfg, &b, false).unwrap(), &[3]).unwrap();
            assert_eq!(base.len(), 4);
            let close = |a: &[f32], b: &[f32]| a.iter().zip(b).all(|(x, y)| (x - y).abs() < 1e-5);
            // Truncated depth reproduces the prefix of the exits.
            for n in 1..=4 {
                let t = logits_per_row(&forward_exits(&p, &cfg, &b, false, Some(n)).unwrap(), &[3])
                    .unwrap();
                assert_eq!(t.len(), n);
                for e in 0..n {
                    assert!(close(&t[e][0], &base[e][0]), "n {n} exit {e}");
                }
            }
            // Perturbing a tensor changes exactly the exits at or after its first use:
            // loop_embd[1] and blk.0.attn_norm_loop first act on application 3 (exit 2),
            // blk.1 on application 2 (exit 1), blk.0's matrices on application 1 (exit 0).
            let bump = |name: &str, row: Option<usize>| -> Vec<bool> {
                let mut q = p.clone();
                let mut v = crate::model::host_f32(&p[name]).unwrap();
                let shape = p[name].shape().to_vec();
                let cols = *shape.last().unwrap() as usize;
                let r = row.unwrap_or(0);
                // Not a constant shift: the LayerNorms would cancel it.
                for (i, x) in v[r * cols..(r + 1) * cols].iter_mut().enumerate() {
                    *x += 0.5 * (i as f32 * 0.7).sin();
                }
                q.insert(name.into(), Array::from_slice(&v, &shape));
                let l = logits_per_row(&forward(&q, &cfg, &b, false).unwrap(), &[3]).unwrap();
                (0..4).map(|e| !close(&l[e][0], &base[e][0])).collect()
            };
            assert_eq!(
                bump("loop_embd.weight", Some(1)),
                vec![false, false, true, true]
            );
            assert_eq!(bump("loop_embd.weight", Some(0)), vec![false; 4]);
            assert_eq!(
                bump("blk.0.attn_norm_loop.weight", None),
                vec![false, false, true, true]
            );
            assert_eq!(
                bump("blk.1.attn_norm.weight", None),
                vec![false, true, true, true]
            );
            assert_eq!(bump("blk.0.ffn_norm.weight", None), vec![true; 4]);
            assert_eq!(
                bump("exit.1.norm.weight", None),
                vec![false, true, false, false]
            );
            // B = 1 x 4 loops (s3-pico layout): exit e follows loop e.
            let cfg1 = looped(1, 4);
            let p1 = crate::s2::golden::golden_params(&cfg1, 5).unwrap();
            let b1 = make_batch(&statics, &[&it], 1);
            let base1 = logits_per_row(&forward(&p1, &cfg1, &b1, false).unwrap(), &[3]).unwrap();
            for l in 1..4 {
                let mut q = p1.clone();
                let mut v = crate::model::host_f32(&p1["loop_embd.weight"]).unwrap();
                for (i, x) in v[l * 96..(l + 1) * 96].iter_mut().enumerate() {
                    *x -= 0.7 * (i as f32 * 0.3 + l as f32).cos();
                }
                q.insert("loop_embd.weight".into(), Array::from_slice(&v, &[4, 96]));
                let lg = logits_per_row(&forward(&q, &cfg1, &b1, false).unwrap(), &[3]).unwrap();
                let changed: Vec<bool> = (0..4).map(|e| !close(&lg[e][0], &base1[e][0])).collect();
                let want: Vec<bool> = (0..4).map(|e| e >= l).collect();
                assert_eq!(changed, want, "loop {l}");
            }
            // The stochastic-depth loss runs on a prefix of the exits.
            let full = decision_loss(&p, &cfg, &b, true)
                .unwrap()
                .item_exact::<f32>();
            let short = decision_loss_exits(&p, &cfg, &b, true, Some(1))
                .unwrap()
                .item_exact::<f32>();
            assert!(full.is_finite() && short.is_finite() && full != short);
            // The weighted form (one compiled graph) gives the same loss and
            // no gradient to the loops beyond the kept exits.
            for n in 1..=4 {
                let w = Array::from_slice(&exit_weight_vec(&cfg, Some(n)), &[4]);
                let a = decision_loss_weighted(&p, &cfg, &b, true, &w)
                    .unwrap()
                    .item_exact::<f32>();
                let e = decision_loss_exits(&p, &cfg, &b, true, Some(n))
                    .unwrap()
                    .item_exact::<f32>();
                assert!((a - e).abs() < 1e-5, "n {n}: {a} vs {e}");
                // DIS with every beta = 1 is the plain deep supervision.
                let ones = Array::from_slice(&[1.0f32; 4], &[4]);
                let d = decision_loss_dis(&p, &cfg, &b, true, &w, &ones)
                    .unwrap()
                    .item_exact::<f32>();
                assert!((a - d).abs() < 1e-5, "dis n {n}: {a} vs {d}");
            }
            let w = Array::from_slice(&exit_weight_vec(&cfg, Some(2)), &[4]);
            let f = |p: HashMap<Rc<str>, Array>, _: ()| -> R<Vec<Array>> {
                Ok(vec![decision_loss_weighted(&p, &cfg, &b, false, &w)?])
            };
            let (_, g) = keyed_value_and_grad(f)(p.clone(), ()).unwrap();
            let norm = |n: &str| {
                crate::model::host_f32(&g[n])
                    .unwrap()
                    .iter()
                    .map(|x| x.abs())
                    .sum::<f32>()
            };
            assert_eq!(norm("loop_embd.weight"), 0.0);
            assert_eq!(norm("blk.0.attn_norm_loop.weight"), 0.0);
            assert!(norm("blk.1.attn_norm.weight") > 0.0);
        });
    }

    #[test]
    fn dis_targets_soften_early_exits_and_keep_the_last() {
        let _g = test_util::lock();
        mlx_rs::with_device(mlx_rs::Device::cpu(), || {
            let b = dis_betas(3, 0.4);
            assert!(
                b.iter()
                    .zip([0.4, 0.7, 1.0])
                    .all(|(x, y)| (x - y).abs() < 1e-6),
                "{b:?}"
            );
            assert_eq!(dis_betas(1, 0.4), vec![1.0]);
            // Two rows: a soft target and a one-hot target with a padded option.
            let t = Array::from_slice(&[0.7f32, 0.2, 0.1, 0.0, 1.0, 0.0], &[2, 3]);
            let beta = Array::from_slice(&dis_betas(3, 0.4), &[3]);
            let x = crate::model::host_f32(&exit_targets(&t, &beta).unwrap()).unwrap();
            assert_eq!(x.len(), 18);
            let h =
                |p: &[f32]| -> f32 { p.iter().filter(|v| **v > 0.0).map(|v| -v * v.ln()).sum() };
            let (e0, e1, e2) = (&x[0..3], &x[6..9], &x[12..15]);
            for e in [e0, e1, e2] {
                assert!((e.iter().sum::<f32>() - 1.0).abs() < 1e-5);
            }
            assert!(h(e0) > h(e1) && h(e1) > h(e2), "{e0:?} {e1:?} {e2:?}");
            for (a, b) in e2.iter().zip([0.7, 0.2, 0.1]) {
                assert!((a - b).abs() < 1e-5);
            }
            // A one-hot target (and its zero padding) stays one-hot at every exit.
            for e in 0..3 {
                let r = &x[e * 6 + 3..e * 6 + 6];
                assert_eq!(r, &[0.0, 1.0, 0.0]);
            }
        });
    }

    #[test]
    fn temporal_carry_forward_and_sequence_loss() {
        let _g = test_util::lock();
        mlx_rs::with_device(mlx_rs::Device::cpu(), || {
            let (_d, tok) = s2_tok();
            let mut cfg = ladder("s3-pico-temporal");
            assert!(cfg.temporal_carry);
            let base = ladder("s3-pico");
            let d = cfg.hidden_size as i32;
            assert_eq!(
                expected_count(&cfg, S2Parts::DECISION),
                expected_count(&base, S2Parts::DECISION) + 2 * d as usize
            );
            assert_eq!(
                expected_count(&cfg, S2Parts::PRETRAIN),
                expected_count(&base, S2Parts::PRETRAIN)
            );
            cfg.vocab_size = 262;
            let fresh = init(&cfg, S2Parts::DECISION, 1).unwrap();
            assert_eq!(
                crate::model::host_f32(&fresh["carry_gate.weight"]).unwrap(),
                vec![0.0; 96]
            );
            let enc = |s: &str| {
                crate::s2::data::encode_record_cfg(
                    &tok,
                    &rec("choice", "how?", &["over", "brake", "around"], s),
                    &cfg,
                )
                .unwrap()
            };
            let (st, s0) = enc("d 35 v 11 dz 2");
            let (_, s1) = enc("d 33 v 10 dz 1");
            assert_eq!(s0.ids, s1.ids, "fused frames share their ids");
            let statics = vec![st];
            let (i0, i1) = (item(0, &s0, 3, 0), item(0, &s1, 3, 0));
            let b0 = make_batch(&statics, &[&i0, &i0], 1);
            let b1 = make_batch(&statics, &[&i1, &i1], 1);
            let rows = |l: &Array| logits_per_row(l, &[3, 3]).unwrap();
            // A zero gate (fresh init): warm equals cold.
            let (_, raw0) = forward_carry(&fresh, &cfg, &b0, false, None).unwrap();
            let on = Array::from_slice(&[1.0f32, 1.0], &[2]);
            let (cold, _) = forward_carry(&fresh, &cfg, &b1, false, None).unwrap();
            let (warm, _) = forward_carry(
                &fresh,
                &cfg,
                &b1,
                false,
                Some(Carry {
                    h: &raw0[3],
                    on: &on,
                }),
            )
            .unwrap();
            assert!(warm.all_close(&cold, 1e-6, 1e-6, None).unwrap());
            // A random gate: the carry changes the warm row only.
            let p = crate::s2::golden::golden_params(&cfg, 3).unwrap();
            let (_, raw0) = forward_carry(&p, &cfg, &b0, false, None).unwrap();
            let (cold, _) = forward_carry(&p, &cfg, &b1, false, None).unwrap();
            let half = Array::from_slice(&[1.0f32, 0.0], &[2]);
            let (warm, _) = forward_carry(
                &p,
                &cfg,
                &b1,
                false,
                Some(Carry {
                    h: &raw0[3],
                    on: &half,
                }),
            )
            .unwrap();
            let (c, w) = (rows(&cold), rows(&warm));
            for e in 0..4 {
                assert!(c[e][0]
                    .iter()
                    .zip(&w[e][0])
                    .any(|(a, b)| (a - b).abs() > 1e-4));
                assert!(c[e][1]
                    .iter()
                    .zip(&w[e][1])
                    .all(|(a, b)| (a - b).abs() < 1e-5));
            }
            // Sequence loss: frame 1 warm from frame 0's exit-1 latent; it equals
            // the hand-chained forward, and the gate gets a gradient.
            let frames = vec![b0.clone(), b1.clone()];
            let on2 = Array::from_slice(&[0.0f32, 0.0, 1.0, 0.0], &[2, 2]);
            let src = Array::from_slice(&[0.0f32, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], &[2, 4]);
            let wv = exit_weight_vec(&cfg, None);
            let wts = Array::from_slice(&wv, &[4]);
            let l = seq_loss(&p, &cfg, &frames, &on2, &src, &wts, false).unwrap();
            let lw = |b: &S2Batch, c: Option<Carry<'_>>| -> f32 {
                let (lg, _) = forward_carry(&p, &cfg, b, false, c).unwrap();
                let lg = masked(&lg, &b.marker_mask).unwrap();
                let lp = lg.subtract(lg.logsumexp_axis(-1, true).unwrap()).unwrap();
                b.target
                    .multiply(&lp)
                    .unwrap()
                    .sum_axis(-1, None)
                    .unwrap()
                    .mean_axis(-1, None)
                    .unwrap()
                    .negative()
                    .unwrap()
                    .multiply(&wts)
                    .unwrap()
                    .sum(None)
                    .unwrap()
                    .item_exact::<f32>()
            };
            let on_row0 = Array::from_slice(&[1.0f32, 0.0], &[2]);
            let want = (lw(&b0, None)
                + lw(
                    &b1,
                    Some(Carry {
                        h: &raw0[1],
                        on: &on_row0,
                    }),
                ))
                / 2.0;
            assert!(
                (l.item_exact::<f32>() - want).abs() < 1e-5,
                "{} vs {want}",
                l.item_exact::<f32>()
            );
            let f = |q: HashMap<Rc<str>, Array>, _: ()| -> R<Vec<Array>> {
                Ok(vec![seq_loss(&q, &cfg, &frames, &on2, &src, &wts, true)?])
            };
            let (_, g) = keyed_value_and_grad(f)(p.clone(), ()).unwrap();
            for n in ["carry_gate.weight", "carry_norm.weight", "loop_embd.weight"] {
                let s: f32 = crate::model::host_f32(&g[n])
                    .unwrap()
                    .iter()
                    .map(|x| x.abs())
                    .sum();
                assert!(s > 0.0, "{n}: no gradient");
            }
        });
    }

    #[test]
    fn s3_permutation_equivariance() {
        let _g = test_util::lock();
        mlx_rs::with_device(mlx_rs::Device::cpu(), || {
            let (_d, tok) = s2_tok();
            let mut cfg = ladder("s3-pico");
            cfg.vocab_size = 262;
            let p = crate::s2::golden::golden_params(&cfg, 6).unwrap();
            let opts = ["north 3", "south", "east -2.5", "west 10", "5"];
            let perm = [2usize, 4, 0, 3, 1];
            let permuted: Vec<&str> = perm.iter().map(|&i| opts[i]).collect();
            let state = "x 4 y -1 wind 12 10-20";
            let enc = |o: &[&str]| {
                crate::s2::data::encode_record_cfg(
                    &tok,
                    &rec("choice", "go 2 where?", o, state),
                    &cfg,
                )
                .unwrap()
            };
            let (s_a, st) = enc(&opts);
            let (s_b, _) = enc(&permuted);
            let statics = vec![s_a, s_b];
            let items = [item(0, &st, 5, 0), item(1, &st, 5, 0)];
            let refs: Vec<&S2Item> = items.iter().collect();
            for qat in [false, true] {
                let logits = forward(&p, &cfg, &make_batch(&statics, &refs, 1), qat).unwrap();
                let rows = logits_per_row(&logits, &[5, 5]).unwrap();
                assert_eq!(rows.len(), 4);
                let mut max_diff = 0f32;
                for exit in &rows {
                    let pa = crate::metrics::softmax_t(&exit[0], 1.0);
                    let pb = crate::metrics::softmax_t(&exit[1], 1.0);
                    for (j, &i) in perm.iter().enumerate() {
                        max_diff = max_diff.max((pb[j] - pa[i]).abs() as f32);
                    }
                }
                assert!(max_diff <= 1e-5, "qat {qat}: max diff {max_diff}");
            }
        });
    }

    #[test]
    fn exit_kd_matches_its_definition_and_spares_the_deepest_exit() {
        let _g = test_util::lock();
        mlx_rs::with_device(mlx_rs::Device::cpu(), || {
            let kd = ExitKd {
                weight: 0.5,
                temp: 2.0,
            };
            // Exits that already agree with the deepest one cost nothing.
            let same = Array::from_slice(&[1.0f32, -2.0, 0.5, 1.0, -2.0, 0.5], &[2, 1, 3]);
            let z = exit_kd_loss(&same, kd).unwrap().item_exact::<f32>();
            assert!(z.abs() < 1e-6, "{z}");
            assert_eq!(
                exit_kd_loss(&Array::from_slice(&[1.0f32, 2.0], &[1, 1, 2]), kd)
                    .unwrap()
                    .item_exact::<f32>(),
                0.0
            );
            // One early exit, one question: weight · T² · KL(p_T(last) ‖ p_T(early)).
            let early = [2.0f32, 0.0, -1.0];
            let last = [0.0f32, 1.0, 0.5];
            let mut both = early.to_vec();
            both.extend(last);
            let logits = Array::from_slice(&both, &[2, 1, 3]);
            let got = exit_kd_loss(&logits, kd).unwrap().item_exact::<f32>();
            let soft = |z: &[f32]| crate::metrics::softmax_t(z, kd.temp);
            let (p, q) = (soft(&last), soft(&early));
            let kl: f64 = p.iter().zip(&q).map(|(a, b)| a * (a / b).ln()).sum();
            let want = (kd.weight * kd.temp * kd.temp) as f64 * kl;
            assert!((got as f64 - want).abs() < 1e-5, "{got} vs {want}");
            // The deepest exit is the teacher: no gradient reaches it through this term.
            let g = mlx_rs::transforms::grad(|x: &Array| exit_kd_loss(x, kd))(&logits).unwrap();
            let g = crate::model::host_f32(&g).unwrap();
            assert!(g[..3].iter().any(|x| x.abs() > 1e-4), "{g:?}");
            assert!(g[3..].iter().all(|x| *x == 0.0), "{g:?}");
        });
    }

    #[test]
    fn s6_parameter_counts() {
        let _g = test_util::lock();
        // s3-pico plus, per encoder block (1 state + 1 static), the probe rows
        // (96 x 96) and `attn_px_norm` (96).
        let cfg = ladder("s6-pico");
        assert!(cfg.parallax && cfg.fuse_numbers && cfg.state_loops == 4);
        assert_eq!(expected_count(&cfg, S2Parts::DECISION), 445_345 + 2 * 9_312);
        assert_eq!(expected_count(&cfg, S2Parts::PRETRAIN), 203_393 + 9_312);
        let p = init(&cfg, S2Parts::DECISION, 0).unwrap();
        assert_eq!(p["blk.0.attn_qkv.weight"].shape(), &[4 * 96, 96]);
        assert_eq!(p["static.blk.0.attn_px_norm.weight"].shape(), &[96]);
        assert!(is_hidden_matrix("blk.0.attn_qkv.weight"));
        assert!(!is_ternary("blk.0.attn_px_norm.weight"));
    }

    #[test]
    fn parallax_zero_probe_is_softmax_attention() {
        let _g = test_util::lock();
        mlx_rs::with_device(mlx_rs::Device::cpu(), || {
            let (_d, tok) = s2_tok();
            let cfg = S2Config::tiny(262);
            let xcfg = S2Config {
                parallax: true,
                ..cfg.clone()
            };
            let mut xp = init(&xcfg, S2Parts::DECISION, 7).unwrap();
            // The same model without the probe rows and norms.
            let d = cfg.hidden_size as i32;
            let mut p = ParamMap::new();
            for (k, v) in &xp {
                if k.ends_with("attn_px_norm.weight") {
                    continue;
                }
                let v = if k.ends_with("attn_qkv.weight") {
                    v.index((..3 * d, ..))
                } else {
                    v.clone()
                };
                p.insert(k.clone(), v);
            }
            let opts = ["north 3", "south", "east -2.5", "west 10"];
            let (s, st) = encode_record(
                &tok,
                &rec("choice", "go where?", &opts, "x 4 y -1 wind 12"),
                40,
                24,
            )
            .unwrap();
            let statics = vec![s];
            let items = [item(0, &st, 4, 0)];
            let refs: Vec<&S2Item> = items.iter().collect();
            let b = make_batch(&statics, &refs, 1);
            let host = |a: Array| crate::model::host_f32(&a).unwrap();
            let base = host(forward(&p, &cfg, &b, false).unwrap());
            let with_probe = host(forward(&xp, &xcfg, &b, false).unwrap());
            let diff = |a: &[f32], b: &[f32]| {
                a.iter()
                    .zip(b)
                    .map(|(x, y)| (x - y).abs())
                    .fold(0f32, f32::max)
            };
            assert!(diff(&base, &with_probe) > 1e-4, "the probe changes nothing");
            for k in xp.keys().cloned().collect::<Vec<_>>() {
                if k.ends_with("attn_px_norm.weight") {
                    let z = ops::zeros_like(&xp[&k]).unwrap();
                    xp.insert(k, z);
                }
            }
            let zero = host(forward(&xp, &xcfg, &b, false).unwrap());
            assert!(diff(&base, &zero) < 1e-5, "ρ = 0: {}", diff(&base, &zero));
        });
    }

    #[test]
    fn parallax_permutation_equivariance() {
        let _g = test_util::lock();
        mlx_rs::with_device(mlx_rs::Device::cpu(), || {
            let (_d, tok) = s2_tok();
            let mut cfg = ladder("s6-pico");
            cfg.vocab_size = 262;
            let p = crate::s2::golden::golden_params(&cfg, 6).unwrap();
            let opts = ["north 3", "south", "east -2.5", "west 10", "5"];
            let perm = [2usize, 4, 0, 3, 1];
            let permuted: Vec<&str> = perm.iter().map(|&i| opts[i]).collect();
            let state = "x 4 y -1 wind 12 10-20";
            let enc = |o: &[&str]| {
                crate::s2::data::encode_record_cfg(
                    &tok,
                    &rec("choice", "go 2 where?", o, state),
                    &cfg,
                )
                .unwrap()
            };
            let (s_a, st) = enc(&opts);
            let (s_b, _) = enc(&permuted);
            let statics = vec![s_a, s_b];
            let items = [item(0, &st, 5, 0), item(1, &st, 5, 0)];
            let refs: Vec<&S2Item> = items.iter().collect();
            for qat in [false, true] {
                let logits = forward(&p, &cfg, &make_batch(&statics, &refs, 1), qat).unwrap();
                let rows = logits_per_row(&logits, &[5, 5]).unwrap();
                let mut max_diff = 0f32;
                for exit in &rows {
                    let pa = crate::metrics::softmax_t(&exit[0], 1.0);
                    let pb = crate::metrics::softmax_t(&exit[1], 1.0);
                    for (j, &i) in perm.iter().enumerate() {
                        max_diff = max_diff.max((pb[j] - pa[i]).abs() as f32);
                    }
                }
                assert!(max_diff <= 1e-5, "qat {qat}: max diff {max_diff}");
            }
        });
    }

    #[test]
    fn ternary_classification() {
        assert!(is_ternary("blk.0.attn_qkv.weight"));
        assert!(is_ternary("static.blk.1.ffn_down.weight"));
        assert!(is_ternary("fusion.0.attn_kv.weight"));
        assert!(is_ternary("cls.weight"));
        for n in [
            "cls.output.weight",
            "token_embd.weight",
            "num_proj.weight",
            "type_embd.weight",
            "fusion.0.attn_kv.bias",
            "blk.1.attn_norm.weight",
            "fusion.0.q_norm.weight",
            "cls.bias",
        ] {
            assert!(!is_ternary(n), "{n}");
        }
        assert!(
            is_hidden_matrix("static.blk.0.ffn_up.weight")
                && !is_hidden_matrix("fusion.0.ffn_up.weight")
        );
    }

    #[test]
    fn ternary_values() {
        let _g = test_util::lock();
        let wm = random::normal::<f32>(&[6, 10][..], 0.0, 1.0, &random::key(3).unwrap()).unwrap();
        let q = ternarize(&wm).unwrap();
        let qv = crate::model::host_f32(&q).unwrap();
        let wv = crate::model::host_f32(&wm).unwrap();
        let (codes, scales) = ternary_codes(&wv, 6, 10);
        for r in 0..6 {
            let g = wv[r * 10..r * 10 + 10].iter().map(|x| x.abs()).sum::<f32>() / 10.0;
            assert!((scales[r] - g).abs() < 1e-6);
            for c in 0..10 {
                let x = qv[r * 10 + c];
                assert!(
                    x == 0.0 || (x.abs() - g).abs() < 1e-5,
                    "{x} not in {{-γ, 0, γ}} (γ = {g})"
                );
                assert!((x - codes[r * 10 + c] as f32 * g).abs() < 1e-5);
            }
        }
        // Straight-through: the gradient of sum(ternarize_ste(W) * C) is C.
        let c = random::normal::<f32>(&[6, 10][..], 0.0, 1.0, &random::key(4).unwrap()).unwrap();
        let f = |a: &[Array]| -> Vec<Array> {
            vec![ternarize_ste(&a[0])
                .unwrap()
                .multiply(&c)
                .unwrap()
                .sum(None)
                .unwrap()]
        };
        let (_, g) =
            mlx_rs::transforms::vjp(f, std::slice::from_ref(&wm), &[Array::from_f32(1.0)]).unwrap();
        assert!(g[0].all_close(&c, 1e-6, 1e-6, None).unwrap());
    }

    fn rec(t: &str, ins: &str, opts: &[&str], state: &str) -> DecisionRecord {
        DecisionRecord {
            src: "unit".into(),
            t: t.into(),
            ins: ins.into(),
            opts: opts.iter().map(|s| s.to_string()).collect(),
            state: state.into(),
            target: (0..opts.len())
                .map(|i| if i == 0 { 1.0 } else { 0.0 })
                .collect(),
            split: "train".into(),
        }
    }

    fn item(stat: usize, state: &Stream, k: usize, qtype: usize) -> S2Item {
        let mut target = vec![0.0; k];
        target[0] = 1.0;
        S2Item {
            stat,
            state: state.clone(),
            qtype,
            target,
            src: "unit".into(),
            group: 0,
        }
    }

    fn statics_out(p: &ParamMap, cfg: &S2Config, statics: &[StaticSeq]) -> Vec<f32> {
        let st = &statics[0];
        let n = st.tokens.len() as i32;
        let out = static_encode(
            p,
            cfg,
            &Array::from_iter(st.tokens.ids.iter().map(|&x| x as i32), &[1, n]),
            &Array::from_slice(&st.tokens.values, &[1, n]),
            &Array::from_iter(
                (0..st.tokens.len()).map(|i| st.tokens.has(i) as u8 as f32),
                &[1, n],
            ),
            &Array::from_slice(&st.seg, &[1, n]),
            &Array::from_slice(&st.pos, &[1, n]),
        )
        .unwrap();
        crate::model::host_f32(&out).unwrap()
    }

    #[test]
    fn static_mask_isolates_options_and_restarts_positions() {
        let _g = test_util::lock();
        let seg = Array::from_slice(&[0, 0, 1, 1, 2, 2, 2, -1], &[1, 8]);
        let m = static_mask(&seg).unwrap();
        m.eval().unwrap();
        let mv = m.as_slice::<bool>();
        let row = |q: usize| mv[q * 8..q * 8 + 8].to_vec();
        assert_eq!(
            row(0),
            vec![true, true, false, false, false, false, false, false]
        );
        assert_eq!(
            row(2),
            vec![true, true, true, true, false, false, false, false]
        );
        assert_eq!(
            row(5),
            vec![true, true, false, false, true, true, true, false]
        );
        assert_eq!(
            row(7),
            vec![false, false, false, false, false, false, false, true]
        );

        // A marker vector depends only on Q and its own option: it is the same
        // when the option is alone, and changing another option leaves it unchanged.
        let (_d, tok) = s2_tok();
        let cfg = S2Config::tiny(262);
        let p = init(&cfg, S2Parts::DECISION, 7).unwrap();
        let d = cfg.hidden_size;
        let (full, _) = encode_record(
            &tok,
            &rec("choice", "pick 3", &["left 1", "a much longer right"], ""),
            96,
            24,
        )
        .unwrap();
        let (alone, _) =
            encode_record(&tok, &rec("choice", "pick 3", &["left 1"], ""), 96, 24).unwrap();
        let (other, _) = encode_record(
            &tok,
            &rec("choice", "pick 3", &["left 1", "zz"], ""),
            96,
            24,
        )
        .unwrap();
        assert_eq!(full.q_len, alone.q_len);
        let a = statics_out(&p, &cfg, std::slice::from_ref(&full));
        let b = statics_out(&p, &cfg, std::slice::from_ref(&alone));
        let c = statics_out(&p, &cfg, std::slice::from_ref(&other));
        let m0 = full.markers[0];
        for i in 0..d {
            assert!((a[m0 * d + i] - b[m0 * d + i]).abs() < 1e-5);
            assert!((a[m0 * d + i] - c[m0 * d + i]).abs() < 1e-5);
        }
        // Question tokens never see the options.
        for t in 0..full.q_len * d {
            assert!((a[t] - b[t]).abs() < 1e-5);
        }
        // Positions restart: option 2 placed first gives the same marker vector.
        let (swapped, _) = encode_record(
            &tok,
            &rec("choice", "pick 3", &["a much longer right", "left 1"], ""),
            96,
            24,
        )
        .unwrap();
        let s = statics_out(&p, &cfg, std::slice::from_ref(&swapped));
        let m1 = swapped.markers[1];
        for i in 0..d {
            assert!((a[m0 * d + i] - s[m1 * d + i]).abs() < 1e-5);
        }
    }

    #[test]
    fn permutation_equivariance() {
        let _g = test_util::lock();
        mlx_rs::with_device(mlx_rs::Device::cpu(), || {
            let (_d, tok) = s2_tok();
            let cfg = S2Config::tiny(262);
            let p = init(&cfg, S2Parts::DECISION, 5).unwrap();
            let opts = ["north 3", "south", "east -2.5", "west 10"];
            let perm = [2usize, 0, 3, 1];
            let permuted: Vec<&str> = perm.iter().map(|&i| opts[i]).collect();
            let (s_a, st) = encode_record(
                &tok,
                &rec("choice", "go where?", &opts, "x 4 y -1 wind 12"),
                40,
                24,
            )
            .unwrap();
            let (s_b, _) = encode_record(
                &tok,
                &rec("choice", "go where?", &permuted, "x 4 y -1 wind 12"),
                40,
                24,
            )
            .unwrap();
            let statics = vec![s_a, s_b];
            let items = [item(0, &st, 4, 0), item(1, &st, 4, 0)];
            let refs: Vec<&S2Item> = items.iter().collect();
            for qat in [false, true] {
                let logits = forward(&p, &cfg, &make_batch(&statics, &refs, 1), qat).unwrap();
                let rows = logits_per_row(&logits, &[4, 4]).unwrap();
                let mut max_diff = 0f32;
                assert_eq!(rows.len(), cfg.n_exits());
                for exit in &rows {
                    let pa = crate::metrics::softmax_t(&exit[0], 1.0);
                    let pb = crate::metrics::softmax_t(&exit[1], 1.0);
                    for (j, &i) in perm.iter().enumerate() {
                        max_diff = max_diff.max((pb[j] - pa[i]).abs() as f32);
                    }
                }
                assert!(max_diff < 1e-5, "qat {qat}: max diff {max_diff}");
            }
        });
    }

    #[test]
    fn padding_does_not_change_outputs() {
        let _g = test_util::lock();
        let (_d, tok) = s2_tok();
        let cfg = S2Config::tiny(262);
        let p = init(&cfg, S2Parts::DECISION, 9).unwrap();
        let (s1, st1) = encode_record(
            &tok,
            &rec("noul", "is it 5?", &["false: no", "true: yes"], "v 5"),
            40,
            24,
        )
        .unwrap();
        let (s2, st2) = encode_record(
            &tok,
            &rec("choice", "pick", &["a", "b", "c"], "long state 1 2 3 4"),
            40,
            24,
        )
        .unwrap();
        let statics = vec![s1, s2];
        let items = [item(0, &st1, 2, 2), item(1, &st2, 3, 0)];
        let alone = forward(&p, &cfg, &make_batch(&statics, &[&items[0]], 1), false).unwrap();
        let both = forward(
            &p,
            &cfg,
            &make_batch(&statics, &[&items[0], &items[1]], 16),
            false,
        )
        .unwrap();
        let a = logits_per_row(&alone, &[2]).unwrap();
        let b = logits_per_row(&both, &[2, 3]).unwrap();
        for e in 0..2 {
            for j in 0..2 {
                assert!(
                    (a[e][0][j] - b[e][0][j]).abs() < 1e-4,
                    "exit {e}: {:?} vs {:?}",
                    a[e][0],
                    b[e][0]
                );
            }
        }
    }

    fn directional_fd(
        p: &ParamMap,
        name: &str,
        dir: &[f32],
        eps: f32,
        f: &dyn Fn(&ParamMap) -> f64,
    ) -> f64 {
        let base = crate::model::host_f32(&p[name]).unwrap();
        let shape = p[name].shape().to_vec();
        let at = |sign: f32| {
            let v: Vec<f32> = base
                .iter()
                .zip(dir)
                .map(|(a, d)| a + sign * eps * d)
                .collect();
            let mut q = p.clone();
            q.insert(name.into(), Array::from_slice(&v, &shape));
            f(&q)
        };
        (at(1.0) - at(-1.0)) / (2.0 * eps as f64)
    }

    #[test]
    fn numeric_gradient_check() {
        let _g = test_util::lock();
        mlx_rs::with_device(mlx_rs::Device::cpu(), || {
            let (_d, tok) = s2_tok();
            let cfg = S2Config {
                hidden_size: 64,
                intermediate_size: 24,
                ..S2Config::tiny(262)
            };
            let p = init(&cfg, S2Parts::DECISION, 13).unwrap();
            let recs = [
                rec(
                    "choice",
                    "pick 2",
                    &["up 1", "down", "stay 7"],
                    "alt 12 vy -3",
                ),
                rec("noul", "safe?", &["false: no", "true: yes"], "gap 4"),
            ];
            let mut statics = Vec::new();
            let mut items = Vec::new();
            for (i, r) in recs.iter().enumerate() {
                let (s, st) = encode_record(&tok, r, 40, 24).unwrap();
                statics.push(s);
                items.push(item(i, &st, r.opts.len(), [0, 2][i]));
            }
            let refs: Vec<&S2Item> = items.iter().collect();
            let batch = make_batch(&statics, &refs, 1);
            // Random weighting of every exit's logits reaches every parameter.
            let wts = random::normal::<f32>(&[2, 2, 3][..], 0.0, 1.0, &random::key(3).unwrap())
                .unwrap()
                .multiply(batch.marker_mask.as_dtype(Dtype::Float32).unwrap())
                .unwrap();
            let objective = |p: &ParamMap| -> R<Array> {
                forward(p, &cfg, &batch, false)?.multiply(&wts)?.sum(None)
            };
            let value = |p: &ParamMap| -> f64 { objective(p).unwrap().item_exact::<f32>() as f64 };
            let f =
                |p: HashMap<Rc<str>, Array>, _: ()| -> R<Vec<Array>> { Ok(vec![objective(&p)?]) };
            let (_, grads) = keyed_value_and_grad(f)(p.clone(), ()).unwrap();
            for name in [
                "token_embd.weight",
                "num_proj.weight",
                "num_proj.bias",
                "type_embd.weight",
                "static.blk.0.attn_qkv.weight",
                "static.blk.0.ffn_up.weight",
                "static_norm.weight",
                "blk.0.attn_qkv.weight",
                "blk.1.attn_norm.weight",
                "blk.1.ffn_down.weight",
                "exit.0.norm.weight",
                "exit.1.norm.weight",
                "fusion.0.attn_q.weight",
                "fusion.0.attn_kv.weight",
                "fusion.0.kv_norm.bias",
                "fusion.0.attn_output.bias",
                "fusion.0.ffn_up.weight",
                "cls.weight",
                "cls.output.bias",
            ] {
                let g = crate::model::host_f32(&grads[name]).unwrap();
                let norm = g.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
                assert!(norm > 1e-6, "{name}: zero gradient");
                let u: Vec<f32> = g.iter().map(|x| (*x as f64 / norm) as f32).collect();
                let num = directional_fd(&p, name, &u, 1e-2, &value);
                assert!(
                    (num - norm).abs() / norm < 2e-2,
                    "{name}: directional derivative {num} vs |g| {norm}"
                );
            }
            // The deep-supervision loss also has a gradient everywhere it should.
            let f = |p: HashMap<Rc<str>, Array>, _: ()| -> R<Vec<Array>> {
                Ok(vec![decision_loss(&p, &cfg, &batch, true)?])
            };
            let (l, grads) = keyed_value_and_grad(f)(p.clone(), ()).unwrap();
            assert!(l[0].item_exact::<f32>().is_finite());
            assert_eq!(grads.len(), p.len());
        });
    }

    #[test]
    fn mlm_loss_runs_and_expansion_copies() {
        let _g = test_util::lock();
        let cfg = S2Config::tiny(262);
        let p = init(&cfg, S2Parts::PRETRAIN, 1).unwrap();
        assert!(p.contains_key("exit.1.norm.weight") && !p.contains_key("exit.0.norm.weight"));
        let mut rng = Rng::new(2);
        let rows: Vec<Stream> = (0..4)
            .map(|_| {
                let ids: Vec<u32> = (0..20)
                    .map(|j| {
                        if j % 4 == 0 {
                            NUM_ID
                        } else {
                            6 + rng.below(200) as u32
                        }
                    })
                    .collect();
                let values = ids
                    .iter()
                    .map(|&i| if i == NUM_ID { 7.0 } else { 0.0 })
                    .collect();
                Stream::new(ids, values)
            })
            .collect();
        let b = crate::s2::data::mask_batch(&rows, 262, 0.3, &mut rng);
        let l = mlm_loss(&p, &cfg, &b, NUM_LOSS_WEIGHT)
            .unwrap()
            .item_exact::<f32>();
        assert!(l.is_finite() && l > 1.0, "{l}");
        let mut dst = init(&cfg, S2Parts::DECISION, 2).unwrap();
        let n = expand_pretrained(&mut dst, &p, &cfg).unwrap();
        assert!(n > 10);
        assert!(dst["static.blk.0.attn_qkv.weight"]
            .all_close(&p["blk.0.attn_qkv.weight"], 0.0, 0.0, None)
            .unwrap());
        assert!(dst["exit.0.norm.weight"]
            .all_close(&p["exit.1.norm.weight"], 0.0, 0.0, None)
            .unwrap());
    }
}
