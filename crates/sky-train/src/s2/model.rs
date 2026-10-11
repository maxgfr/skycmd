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
    config::HEAD_DIM,
    model::{gelu_erf, layer_norm},
    num::{features, NUM_FEATURES, NUM_ID},
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

fn encoder_specs(
    out: &mut Vec<(String, Vec<i32>, Init)>,
    prefix: &str,
    n: usize,
    d: i32,
    ff: i32,
    depth: usize,
) {
    let std = 0.02f32;
    let std_out = std / (2.0 * depth as f32).sqrt();
    for l in 0..n {
        let p = format!("{prefix}{l}");
        if l > 0 {
            out.push((format!("{p}.attn_norm.weight"), vec![d], Init::Ones));
        }
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
    encoder_specs(&mut out, "blk.", cfg.state_layers, d, ff, cfg.state_layers);
    let last = cfg.n_exits() - 1;
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
        let std_out = 0.02 / (2.0 * cfg.state_layers as f32).sqrt();
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
        for m in BLOCK_MATRICES
            .iter()
            .chain(&["attn_norm.weight", "ffn_norm.weight"])
        {
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

/// `[B, T, H * 64]` -> `[B, H, T, 64]`.
fn heads(x: &Array, h: i32) -> R<Array> {
    let (b, t) = (x.dim(0), x.dim(1));
    x.reshape(&[b, t, h, HEAD_DIM])?
        .transpose_axes(&[0, 2, 1, 3])
}

fn merge_heads(x: &Array) -> R<Array> {
    let (b, h, t) = (x.dim(0), x.dim(1), x.dim(2));
    x.transpose_axes(&[0, 2, 1, 3])?
        .reshape(&[b, t, h * HEAD_DIM])
}

fn attend(q: &Array, k: &Array, v: &Array, mask: &Array) -> R<Array> {
    fast::scaled_dot_product_attention(q, k, v, 1.0 / (HEAD_DIM as f32).sqrt(), mask, None)
}

/// RoPE flavour of an encoder.
enum Rope {
    /// Positions `0..T` (fast kernel).
    Sequential(f32),
    /// Explicit positions: `(cos, sin)` tables `[B, 1, T, 32]`.
    Table(Array, Array),
}

/// cos / sin tables of rotate_half RoPE for explicit positions `[B, T]`.
fn rope_tables(pos: &Array, theta: f32, like: &Array) -> R<(Array, Array)> {
    let half = HEAD_DIM / 2;
    let inv: Vec<f32> = (0..half)
        .map(|i| theta.powf(-(2.0 * i as f32) / HEAD_DIM as f32))
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
        Rope::Sequential(theta) => fast::rope(x, HEAD_DIM, false, *theta, 1.0, 0, None),
        Rope::Table(c, s) => {
            let parts = x.split_at_indices(&[HEAD_DIM / 2], -1)?;
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
fn block(
    p: &ParamMap,
    cfg: &S2Config,
    pre: &str,
    l: usize,
    x: Array,
    mask: &Array,
    rope: &Rope,
) -> R<Array> {
    let eps = cfg.norm_eps;
    let h = cfg.n_heads() as i32;
    let d = cfg.hidden_size as i32;
    let y = if l == 0 {
        x.clone()
    } else {
        layer_norm(&x, w(p, &format!("{pre}.attn_norm.weight"))?, None, eps)?
    };
    let qkv = linear(&y, w(p, &format!("{pre}.attn_qkv.weight"))?, None)?;
    let parts = qkv.split_at_indices(&[d, 2 * d], -1)?;
    let q = apply_rope(&heads(&parts[0], h)?, rope)?;
    let k = apply_rope(&heads(&parts[1], h)?, rope)?;
    let v = heads(&parts[2], h)?;
    let a = merge_heads(&attend(&q, &k, &v, mask)?)?;
    let x = x.add(linear(
        &a,
        w(p, &format!("{pre}.attn_output.weight"))?,
        None,
    )?)?;
    let y = layer_norm(&x, w(p, &format!("{pre}.ffn_norm.weight"))?, None, eps)?;
    let u = linear(&y, w(p, &format!("{pre}.ffn_up.weight"))?, None)?;
    let g = u.split_at_indices(&[cfg.intermediate_size as i32], -1)?;
    let m = gelu_erf(&g[0])?.multiply(&g[1])?;
    x.add(linear(&m, w(p, &format!("{pre}.ffn_down.weight"))?, None)?)
}

/// `e = tok_embd[id] + is_num · (num_proj · φ(v) + num_proj.bias)`.
fn embed(p: &ParamMap, ids: &Array, values: &Array) -> R<Array> {
    let table = w(p, "token_embd.weight")?;
    let e = table.take_axis(ids, 0)?;
    let phi = cast_like(features(values)?, table)?;
    let num = linear(&phi, w(p, "num_proj.weight")?, Some(w(p, "num_proj.bias")?))?;
    let is_num = cast_like(
        ids.eq(Array::from_int(NUM_ID as i32))?
            .as_dtype(Dtype::Float32)?,
        table,
    )?;
    let shape = [ids.dim(0), ids.dim(1), 1];
    e.add(num.multiply(is_num.reshape(&shape)?)?)
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
fn static_encode(
    p: &ParamMap,
    cfg: &S2Config,
    ids: &Array,
    values: &Array,
    seg: &Array,
    pos: &Array,
) -> R<Array> {
    let mut x = embed(p, ids, values)?;
    let mask = static_mask(seg)?;
    let (c, s) = rope_tables(pos, cfg.rope_theta, &x)?;
    let rope = Rope::Table(c, s);
    for l in 0..cfg.static_layers {
        x = block(p, cfg, &format!("static.blk.{l}"), l, x, &mask, &rope)?;
    }
    layer_norm(&x, w(p, "static_norm.weight")?, None, cfg.norm_eps)
}

/// State encoder over embedded inputs `x`: the normalized hidden states
/// `H_e` of the requested exits (ascending exit indices).
fn state_exits(
    p: &ParamMap,
    cfg: &S2Config,
    mut x: Array,
    valid: &Array,
    exits: &[usize],
) -> R<Vec<Array>> {
    let mask = padding_mask(valid)?;
    let rope = Rope::Sequential(cfg.rope_theta);
    let last_layer = exits
        .iter()
        .map(|&e| cfg.exits[e])
        .max()
        .unwrap_or(cfg.state_layers);
    let mut out = Vec::with_capacity(exits.len());
    for l in 0..last_layer {
        x = block(p, cfg, &format!("blk.{l}"), l, x, &mask, &rope)?;
        for &e in exits {
            if cfg.exits[e] == l + 1 {
                out.push(layer_norm(
                    &x,
                    w(p, &format!("exit.{e}.norm.weight"))?,
                    None,
                    cfg.norm_eps,
                )?);
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
    let p = &*prepare(p, cfg, qat)?;
    let st = static_encode(p, cfg, &b.st_ids, &b.st_values, &b.st_seg, &b.st_pos)?;
    let d = st.dim(2);
    let flat = st.reshape(&[-1, d])?;
    let (rows, k, tq) = (b.marker_idx.dim(0), b.marker_idx.dim(1), b.q_idx.dim(1));
    let m = flat
        .take_axis(b.marker_idx.reshape(&[-1])?, 0)?
        .reshape(&[rows, k, d])?;
    let qt = flat
        .take_axis(b.q_idx.reshape(&[-1])?, 0)?
        .reshape(&[rows, tq, d])?;
    let x = embed(p, &b.ids, &b.values)?;
    let all: Vec<usize> = (0..cfg.n_exits()).collect();
    let hs = state_exits(p, cfg, x, &b.valid, &all)?;
    fusion(p, cfg, &m, &b.qtype, &hs, &b.valid, &qt, &b.q_valid)
}

/// Logits with invalid options pushed to -1e4 (`mask` `[B, K]` broadcast over exits).
pub fn masked(logits: &Array, mask: &Array) -> R<Array> {
    ops::select(mask, logits, Array::from_f32(-1e4))
}

/// Deep supervision: `Σ_e w_e · CE(exit e)` against the soft target.
pub fn decision_loss(p: &ParamMap, cfg: &S2Config, b: &S2Batch, qat: bool) -> R<Array> {
    let logits = masked(&forward(p, cfg, b, qat)?, &b.marker_mask)?;
    let logp = logits.subtract(logits.logsumexp_axis(-1, true)?)?;
    let ce = b
        .target
        .multiply(&logp)?
        .sum_axis(-1, None)?
        .mean_axis(-1, None)?
        .negative()?; // [E]
    let wts = Array::from_slice(&cfg.exit_weights(), &[cfg.n_exits() as i32]);
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
    let mut x = embed(p, &b.ids, &b.values)?;
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
                Stream { ids, values }
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
