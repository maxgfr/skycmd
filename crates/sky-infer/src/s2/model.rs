//! Weights and forward pieces of an s2 model (SPEC-S2 §3).
//!
//! The engine drives these pieces: [`S2Model::encode_static`] (+ [`S2Model::prepare_fusion`])
//! once per (question, options), [`S2Model::state_reset`] / [`S2Model::state_advance`] /
//! [`S2Model::exit_kv`] once per state, and [`S2Model::fusion_cached`] once per question and exit.
//!
//! SPEC-S3 options: value-fused tokens (a `has_value` flag per token instead of `id == [NUM]`) and
//! a looped state encoder (`state_loops` passes over the `state_blocks` shared blocks, with
//! `loop_embd` added before every pass after the first and `blk.0.attn_norm_loop` as the input
//! norm of block 0 from the second pass on).
//!
//! Hot-path layout: f32 matrices are repacked into k-major panels ([`PanelMatrix`]) for the small
//! row counts of a frame; the state stream reuses preallocated buffers (no allocation per frame
//! once warm); attention over a handful of tokens skips the tiled kernel; the first fusion block's
//! queries and their attention over the cached question tokens are precomputed with the static
//! stream (merged with the state keys by an online softmax).

use sky_gguf::{GgmlType, Gguf};

use super::es::EsFfn;
use super::S2Config;
use crate::kernels::{
    self, Attention, AttnPartial, AttnScratch, I8Panels, I8Scratch, KvSeg, PanelMatrix, Rope,
    SelfAttnScratch,
};
use crate::lut::{LutMatrix, LutScratch};
use crate::num::{self, NUM_FEATURES};
use crate::prompt::MAX_OPTION_TOKENS;
use crate::ternary::{self, TernaryMatrix};
use crate::{Error, Result};

/// Number of question types (choice, score, noul).
pub const N_QTYPES: usize = 3;

/// Weight storage of a linear layer.
#[derive(Clone, Debug)]
pub enum LinWeight {
    /// Row-major `[n_out][n_in]` (as in the file).
    F32(Vec<f32>),
    /// Repacked for small row counts (see [`S2Model::pack_panels`]).
    Panel(PanelMatrix),
    Ternary(TernaryMatrix),
    /// Ternary weights as int8 panels, computed with integer dot products on two-pass int8
    /// activations ([`TernaryCompute::Int8`](super::TernaryCompute::Int8)).
    I8(I8Panels),
    /// Ternary weights as 2-bit lookup indices (SPEC-S3 §8,
    /// [`TernaryCompute::Lut`](super::TernaryCompute::Lut)).
    Lut(LutMatrix),
}

thread_local! {
    /// Quantized activations of the int8 products (single-threaded engines: one per thread).
    static I8_SCRATCH: std::cell::RefCell<I8Scratch> = std::cell::RefCell::new(I8Scratch::default());
    /// Tables of the lookup products.
    static LUT_SCRATCH: std::cell::RefCell<LutScratch> = std::cell::RefCell::new(LutScratch::default());
}

/// A linear layer `y = x W^T + b` with `W: [n_out][n_in]`.
#[derive(Clone, Debug)]
pub struct Lin {
    pub w: LinWeight,
    pub b: Option<Vec<f32>>,
    pub n_in: usize,
    pub n_out: usize,
}

impl Lin {
    /// `y (+)= x W^T + b` for `rows` rows.
    #[inline]
    pub fn forward(&self, x: &[f32], rows: usize, y: &mut [f32], accumulate: bool) {
        match &self.w {
            LinWeight::F32(w) => kernels::matmul_f32(
                x,
                rows,
                self.n_in,
                w,
                self.n_out,
                self.b.as_deref(),
                y,
                accumulate,
            ),
            LinWeight::Panel(p) => p.matmul(
                x,
                self.n_in,
                rows,
                self.b.as_deref(),
                y,
                self.n_out,
                accumulate,
            ),
            LinWeight::Ternary(t) => t.matmul(x, rows, self.b.as_deref(), y, accumulate),
            LinWeight::Lut(m) => LUT_SCRATCH.with(|sc| {
                m.matmul(
                    x,
                    rows,
                    self.b.as_deref(),
                    y,
                    accumulate,
                    &mut sc.borrow_mut(),
                )
            }),
            LinWeight::I8(m) => I8_SCRATCH.with(|sc| {
                m.matmul(
                    x,
                    rows,
                    self.b.as_deref(),
                    y,
                    accumulate,
                    &mut sc.borrow_mut(),
                )
            }),
        }
    }

    pub fn is_ternary(&self) -> bool {
        matches!(
            self.w,
            LinWeight::Ternary(_) | LinWeight::I8(_) | LinWeight::Lut(_)
        )
    }

    /// Ternary weights as lookup-table indices with two activation passes (see
    /// [`LinWeight::Lut`]); matrices with more than [`crate::lut::MAX_K`] inputs stay packed.
    pub fn ternary_to_lut(&mut self) {
        if let LinWeight::Ternary(t) = &self.w {
            if let Some(m) = LutMatrix::from_trits(&t.trits(), &t.scale, t.rows, t.cols, 2) {
                self.w = LinWeight::Lut(m);
            }
        }
    }

    /// Ternary weights as int8 panels (see [`LinWeight::I8`]).
    pub fn ternary_to_i8(&mut self) {
        if let LinWeight::Ternary(t) = &self.w {
            self.w = LinWeight::I8(I8Panels::from_trits(&t.trits(), &t.scale, t.rows, t.cols));
        }
    }

    /// Replace ternary weights by their f32 values (same results up to rounding).
    pub fn dequantize(&mut self) {
        if let LinWeight::Ternary(t) = &self.w {
            self.w = LinWeight::F32(t.to_f32());
        }
    }

    /// Repack row-major f32 weights into panels (same results up to rounding).
    pub fn pack_panels(&mut self) {
        if let LinWeight::F32(w) = &self.w {
            self.w = LinWeight::Panel(PanelMatrix::from_rows(w, self.n_out, self.n_in));
        }
    }

    /// The weights as f32 values (row-major).
    pub fn weight_f32(&self) -> Vec<f32> {
        match &self.w {
            LinWeight::F32(w) => w.clone(),
            LinWeight::Panel(p) => p.to_rows(),
            LinWeight::Ternary(t) => t.to_f32(),
            LinWeight::Lut(m) => {
                let mut w = vec![0.0; self.n_out * self.n_in];
                for o in 0..m.n {
                    for i in 0..m.k {
                        let b = m.idx[(o / 16 * m.groups + i / 4) * 16 + o % 16];
                        let t = ((b >> (i % 4)) & 1) as f32 - ((b >> (4 + i % 4)) & 1) as f32;
                        w[o * m.k + i] = t * m.scale[o];
                    }
                }
                w
            }
            LinWeight::I8(m) => {
                let mut w = vec![0.0; self.n_out * self.n_in];
                for o in 0..m.n {
                    for i in 0..m.k {
                        let b = m.data[(o / 4) * m.k16 * 4 + (i / 4) * 16 + (o % 4) * 4 + i % 4];
                        w[o * m.k + i] = b as f32 * m.scale[o];
                    }
                }
                w
            }
        }
    }

    fn n_params(&self) -> usize {
        self.n_in * self.n_out + self.b.as_ref().map_or(0, Vec::len)
    }
}

/// LayerNorm parameters (eps from the config).
#[derive(Clone, Debug)]
pub struct Ln {
    pub w: Vec<f32>,
    pub b: Option<Vec<f32>>,
}

impl Ln {
    #[inline]
    fn apply(&self, x: &[f32], d: usize, eps: f32, out: &mut [f32]) {
        kernels::layer_norm(x, d, &self.w, self.b.as_deref(), eps, out);
    }

    fn n_params(&self) -> usize {
        self.w.len() + self.b.as_ref().map_or(0, Vec::len)
    }
}

/// Token embedding table `[vocab][d]`.
#[derive(Clone, Debug)]
pub enum TokEmb {
    F32(Vec<f32>),
    /// int8 rows with one scale per row (`token_embd.q8` + `token_embd.scale`).
    Q8 {
        q: Vec<i8>,
        scale: Vec<f32>,
    },
}

/// An encoder block (shared layout of the static and state encoders).
#[derive(Clone, Debug)]
pub struct Block {
    /// `None` for block 0 (identity).
    pub attn_norm: Option<Ln>,
    pub qkv: Lin,
    pub out: Lin,
    pub ffn_norm: Ln,
    /// GeGLU up-projection (`None` with the expand-and-sparsify FFN, whose up-projection is the
    /// seeded fixed matrix of [`EsFfn`]).
    pub up: Option<Lin>,
    pub down: Lin,
    /// SPEC-S3 §9 expand-and-sparsify FFN (reads `down`'s weights, transposed).
    pub es: Option<EsFfn>,
}

/// A fusion block: cross-attention of the option markers to (state ⊕ question) tokens.
#[derive(Clone, Debug)]
pub struct FusionBlock {
    pub q_norm: Ln,
    pub kv_norm: Ln,
    pub q: Lin,
    /// `[2d, d]`: keys then values.
    pub kv: Lin,
    pub out: Lin,
    pub ffn_norm: Ln,
    pub up: Lin,
    pub down: Lin,
    /// `attn_kv` folded into the queries for the state tokens (see [`Absorbed`]).
    pub absorbed: Option<Absorbed>,
}

/// The first state block application has no input norm, so its `attn_qkv` is linear in the
/// embedding `e = tok_embd[id] + has_value · (P φ(v) + b)`: `W e = T[id] + has_value · (U φ(v)
/// + c)` with `T = tok_embd Wᵀ` (`[vocab][3d]`), `U = W P` (`[3d][17]`) and `c = W b`. A frame
/// then costs 17 instead of `d` multiply-adds per output of its first projection.
#[derive(Clone, Debug)]
pub struct QkvTable {
    /// `[vocab][3d]`
    pub tok: Vec<f32>,
    /// `W P` as a panel matrix (`3d` outputs, 17 inputs).
    pub num: PanelMatrix,
    /// `W b` (`3d`).
    pub num_bias: Vec<f32>,
}

/// Inputs of a block whose `attn_qkv` comes from a [`QkvTable`].
#[derive(Clone, Copy)]
struct TableIn<'a> {
    tb: &'a QkvTable,
    ids: &'a [u32],
    has_value: &'a [bool],
    phi: &'a [f32],
}

/// Largest [`QkvTable::tok`] built (floats).
const MAX_QKV_TABLE: usize = 1 << 21;

/// The state-token keys and values of a fusion block folded into its few queries. With
/// `h_j = LN_kv(H_e)_j`, `k_j = W_k h_j + b_k` and `v_j = W_v h_j + b_v`, for a query `q` of head
/// `c`: `q · k_j = (W_kᵀ q) · h_j + q · b_k` and `Σ p_j v_j = W_v (Σ p_j h_j) + b_v Σ p_j`.
/// For n option markers and S state tokens this costs `2·n·d²` multiply-adds per block instead
/// of `2·S·d²` for projecting every state token (n = 2, S ≈ 9 for a game state).
#[derive(Clone, Debug)]
pub struct Absorbed {
    /// Per head: `W_k[head rows]ᵀ` as a panel matrix (`d` outputs, `head_dim` inputs).
    pub kt: Vec<PanelMatrix>,
    /// Per head: `W_v[head rows]` (`head_dim` outputs, `d` inputs).
    pub v: Vec<PanelMatrix>,
    /// `b_k` (`d`).
    pub bk: Vec<f32>,
    /// `b_v` (`d`).
    pub bv: Vec<f32>,
}

/// A loaded s2 model.
#[derive(Clone, Debug)]
pub struct S2Model {
    pub cfg: S2Config,
    pub tok_emb: TokEmb,
    /// `[d, 17]` with bias.
    pub num_proj: Lin,
    /// `[3][d]`
    pub type_emb: Vec<f32>,
    pub static_blocks: Vec<Block>,
    pub static_norm: Ln,
    /// The `state_blocks` distinct state blocks (applied `state_loops` times).
    pub state_blocks: Vec<Block>,
    /// `loop_embd.weight` `[state_loops][d]` (only when `state_loops > 1`).
    pub loop_emb: Option<Vec<f32>>,
    /// `blk.0.attn_norm_loop` (only when `state_loops > 1`).
    pub attn_norm_loop: Option<Ln>,
    /// `attn_qkv` of state block 0 precomputed on the embeddings (see [`QkvTable`]).
    pub qkv_table: Option<QkvTable>,
    /// `carry_gate.weight` and `carry_norm` (only with `temporal_carry`; SPEC-S3 §7).
    pub carry: Option<(Vec<f32>, Ln)>,
    pub exit_norms: Vec<Ln>,
    pub fusion: Vec<FusionBlock>,
    pub cls_norm: Ln,
    pub cls: Lin,
    pub cls_out: Lin,
    rope: Rope,
}

/// Tokens of a static stream: `[CLS] head [SEP]` then `[MASK] option` per option.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct StaticTokens {
    pub ids: Vec<u32>,
    /// Number values (0 for tokens without a value), aligned with `ids`.
    pub values: Vec<f32>,
    /// Whether each token carries a value (`[NUM]`, or a value-fused token with
    /// `fuse_numbers`). Empty means `id == [NUM]`.
    pub has_value: Vec<bool>,
    /// Length of the question part `Q`.
    pub q_len: usize,
    /// `(start, len)` of every option in `ids` (the marker is at `start`).
    pub options: Vec<(usize, usize)>,
}

/// The cached result of the static encoder for one (question type, instructions, options).
#[derive(Clone, Debug)]
pub struct StaticEnc {
    pub q_len: usize,
    pub n_options: usize,
    /// Marker vectors `m_i`, `[n_options][d]`.
    pub markers: Vec<f32>,
    /// Per fusion block: `attn_kv(LN_kv(q_tokens))`, `[q_len][2d]`.
    pub kv: Vec<Vec<f32>>,
    /// Per fusion block and head: the same keys and values in panel layouts.
    pub kv_panels: Vec<Vec<HeadPanels>>,
    pub n_tokens: usize,
    /// Question-type dependent part, see [`S2Model::prepare_fusion`].
    pub fusion0: Option<Fusion0>,
}

/// Static part of the first fusion block for one question type: its input `x0 = m_i +
/// type_embd[t]`, its queries and their attention over the cached question tokens.
#[derive(Clone, Debug)]
pub struct Fusion0 {
    pub qtype: usize,
    /// `[n_options][d]`
    pub x0: Vec<f32>,
    /// `attn_q(LN_q(x0))`, `[n_options][d]`.
    pub q0: Vec<f32>,
    /// Attention of `q0` over the question tokens of fusion block 0.
    pub prior: AttnPartial,
}

/// The (partially) encoded state stream of one state text. The buffers are sized for
/// `max_state_tokens` once and reused by [`S2Model::state_reset`].
#[derive(Clone, Debug, Default)]
pub struct StateEnc {
    /// Number of tokens `S`.
    pub len: usize,
    /// Hidden states after `layers_done` block applications, `[S][d]`.
    pub x: Vec<f32>,
    /// Block applications done (effective depth).
    pub layers_done: usize,
    /// Per exit: whether `exit_h` holds `H_e` (the encoder went past it).
    pub exit_ready: Vec<bool>,
    /// Per exit: `H_e`, `[S][d]`.
    pub exit_h: Vec<Vec<f32>>,
    /// Per exit: whether `exit_kv` holds the projected keys / values.
    pub kv_ready: Vec<bool>,
    /// Per exit, per fusion block: `attn_kv(LN_kv(H_e))`, `[S][2d]`.
    pub exit_kv: Vec<Vec<Vec<f32>>>,
    /// Token ids of the stream.
    pub ids: Vec<u32>,
    /// `has_value` of every token.
    pub has_value: Vec<bool>,
    /// `φ(v)` of every token (0 for tokens without a value), `[S][17]`.
    pub phi: Vec<f32>,
    /// The stream started from a carried latent (SPEC-S3 §7), not from the plain embeddings.
    pub carried: bool,
}

/// Scratch buffers (grown on demand, then reused: a decision does not allocate once warm).
#[derive(Clone, Debug, Default)]
pub struct S2Workspace {
    hn: Vec<f32>,
    qkv: Vec<f32>,
    att: Vec<f32>,
    up: Vec<f32>,
    act: Vec<f32>,
    kvtmp: Vec<f32>,
    iota: Vec<usize>,
    /// State positions `0..S` (see [`S2Model::state_advance`]).
    spos: Vec<usize>,
    mx: Vec<f32>,
    mh: Vec<f32>,
    mq: Vec<f32>,
    mup: Vec<f32>,
    scores: Vec<f32>,
    ab_t: Vec<f32>,
    ab_u: Vec<f32>,
    ab_o: Vec<f32>,
    attn: AttnScratch,
    self_attn: SelfAttnScratch,
}

/// Longest stream attended with [`kernels::attend`] (plain loops) instead of the tiled kernel.
const SMALL_ATTN: usize = 32;

/// `v[..n]`, growing `v` if needed.
#[inline]
fn grow<T: Clone + Default>(v: &mut Vec<T>, n: usize) -> &mut [T] {
    if v.len() < n {
        v.resize(n, T::default());
    }
    &mut v[..n]
}

impl S2Workspace {
    fn ensure_iota(&mut self, n: usize) {
        if self.iota.len() < n {
            self.iota = (0..n).collect();
        }
    }
}

/// Attention pattern of an encoder block.
#[derive(Clone, Copy)]
enum AttnMode<'a> {
    /// Every token attends to every token.
    Full,
    /// Static stream: `Q` attends to `Q`, option `i` attends to `Q ∪ O_i`.
    Static {
        q_len: usize,
        options: &'a [(usize, usize)],
    },
}

/// Reads s2 tensors from a GGUF file.
struct Src<'a, 'b> {
    g: &'b Gguf<'a>,
}

impl Src<'_, '_> {
    fn has(&self, name: &str) -> bool {
        self.g.tensor(name).is_some()
    }

    /// An f32 tensor (F32, F16, BF16 or Q8_0 in the file) with a row-major shape check.
    fn f32(&self, name: &str, shape: &[usize]) -> Result<Vec<f32>> {
        let info = self
            .g
            .tensor(name)
            .ok_or_else(|| Error::Missing(format!("tensor {name}")))?;
        if info.shape() != shape {
            return Err(Error::Format(format!(
                "tensor {name}: shape {:?}, expected {shape:?}",
                info.shape()
            )));
        }
        Ok(sky_gguf::dequantize(info.dtype, self.g.tensor_bytes(info))?)
    }

    fn ln(&self, base: &str, d: usize, bias: bool) -> Result<Ln> {
        Ok(Ln {
            w: self.f32(&format!("{base}.weight"), &[d])?,
            b: if bias {
                Some(self.f32(&format!("{base}.bias"), &[d])?)
            } else {
                None
            },
        })
    }

    /// A linear layer `base`: ternary (`base.tq` + `base.scale`, also accepted as
    /// `base.weight.tq`) or float (`base.weight`), with `base.bias` when `bias`.
    fn lin(&self, base: &str, n_out: usize, n_in: usize, bias: bool) -> Result<Lin> {
        let w = if let Some(tq) = [format!("{base}.tq"), format!("{base}.weight.tq")]
            .into_iter()
            .find(|n| self.has(n))
        {
            let info = self.g.tensor(&tq).expect("present");
            if info.dtype != GgmlType::I8 {
                return Err(Error::Format(format!(
                    "tensor {tq}: type {}, expected I8",
                    info.dtype.name()
                )));
            }
            let bytes = self.g.tensor_bytes(info).to_vec();
            if bytes.len() != n_out * ternary::row_bytes(n_in) {
                return Err(Error::Format(format!(
                    "tensor {tq}: {} bytes, expected {n_out} rows of {} bytes",
                    bytes.len(),
                    ternary::row_bytes(n_in)
                )));
            }
            let scale_name = format!("{}.scale", tq.strip_suffix(".tq").expect("suffix"));
            let scale = self.f32(&scale_name, &[n_out])?;
            LinWeight::Ternary(TernaryMatrix::from_packed(bytes, scale, n_out, n_in)?)
        } else {
            LinWeight::F32(self.f32(&format!("{base}.weight"), &[n_out, n_in])?)
        };
        let b = if bias {
            Some(self.f32(&format!("{base}.bias"), &[n_out])?)
        } else {
            None
        };
        Ok(Lin { w, b, n_in, n_out })
    }

    fn block(&self, prefix: &str, n: usize, cfg: &S2Config) -> Result<Block> {
        let (d, ff) = (cfg.hidden_size, cfg.intermediate_size);
        let p = |s: &str| format!("{prefix}{n}.{s}");
        let attn_norm = if n == 0 && !self.has(&p("attn_norm.weight")) {
            None
        } else {
            Some(self.ln(&p("attn_norm"), d, false)?)
        };
        if let Some(spec) = cfg.ffn_es {
            let down = self.lin(&p("ffn_down"), d, spec.units, false)?;
            let es = EsFfn::new(spec, d, &down.weight_f32());
            return Ok(Block {
                attn_norm,
                qkv: self.lin(&p("attn_qkv"), 3 * d, d, false)?,
                out: self.lin(&p("attn_output"), d, d, false)?,
                ffn_norm: self.ln(&p("ffn_norm"), d, false)?,
                up: None,
                down,
                es: Some(es),
            });
        }
        Ok(Block {
            attn_norm,
            qkv: self.lin(&p("attn_qkv"), 3 * d, d, false)?,
            out: self.lin(&p("attn_output"), d, d, false)?,
            ffn_norm: self.ln(&p("ffn_norm"), d, false)?,
            up: Some(self.lin(&p("ffn_up"), 2 * ff, d, false)?),
            down: self.lin(&p("ffn_down"), d, ff, false)?,
            es: None,
        })
    }

    fn tok_emb(&self, cfg: &S2Config) -> Result<TokEmb> {
        let (v, d) = (cfg.vocab_size, cfg.hidden_size);
        if let Some(info) = self.g.tensor("token_embd.q8") {
            if info.shape() != [v, d] || info.dtype != GgmlType::I8 {
                return Err(Error::Format(format!(
                    "tensor token_embd.q8: {:?} {}, expected I8 [{v}, {d}]",
                    info.shape(),
                    info.dtype.name()
                )));
            }
            let q = self.g.tensor_bytes(info).iter().map(|&b| b as i8).collect();
            let scale = self.f32("token_embd.scale", &[v])?;
            return Ok(TokEmb::Q8 { q, scale });
        }
        Ok(TokEmb::F32(self.f32("token_embd.weight", &[v, d])?))
    }
}

impl S2Model {
    /// Load the weights of a `skycmd-s2` GGUF file (ternary matrices stay packed).
    pub fn from_gguf(g: &Gguf<'_>) -> Result<Self> {
        let cfg = S2Config::from_gguf(g)?;
        let src = Src { g };
        let d = cfg.hidden_size;
        let tok_emb = src.tok_emb(&cfg)?;
        let num_proj = src.lin("num_proj", d, cfg.num_features, true)?;
        let type_emb = src.f32("type_embd.weight", &[N_QTYPES, d])?;
        let static_blocks = (0..cfg.static_layers)
            .map(|n| src.block("static.blk.", n, &cfg))
            .collect::<Result<Vec<_>>>()?;
        let static_norm = src.ln("static_norm", d, false)?;
        let state_blocks = (0..cfg.state_blocks)
            .map(|n| src.block("blk.", n, &cfg))
            .collect::<Result<Vec<_>>>()?;
        let (loop_emb, attn_norm_loop) = if cfg.state_loops > 1 {
            (
                Some(src.f32("loop_embd.weight", &[cfg.state_loops, d])?),
                Some(src.ln("blk.0.attn_norm_loop", d, false)?),
            )
        } else {
            (None, None)
        };
        let carry = if cfg.temporal_carry {
            Some((
                src.f32("carry_gate.weight", &[d])?,
                src.ln("carry_norm", d, false)?,
            ))
        } else {
            None
        };
        // exits are numbered by their index (0-based); a file numbering them by depth also loads
        let by_index = (0..cfg.exits.len()).all(|e| src.has(&format!("exit.{e}.norm.weight")));
        let exit_norms = cfg
            .exits
            .iter()
            .enumerate()
            .map(|(e, &l)| {
                src.ln(
                    &format!("exit.{}.norm", if by_index { e } else { l }),
                    d,
                    false,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let fusion = (0..cfg.fusion_layers)
            .map(|n| {
                let p = |s: &str| format!("fusion.{n}.{s}");
                Ok(FusionBlock {
                    q_norm: src.ln(&p("q_norm"), d, true)?,
                    kv_norm: src.ln(&p("kv_norm"), d, true)?,
                    q: src.lin(&p("attn_q"), d, d, true)?,
                    kv: src.lin(&p("attn_kv"), 2 * d, d, true)?,
                    out: src.lin(&p("attn_output"), d, d, true)?,
                    ffn_norm: src.ln(&p("ffn_norm"), d, true)?,
                    up: src.lin(&p("ffn_up"), 2 * d, d, true)?,
                    down: src.lin(&p("ffn_down"), d, 2 * d, true)?,
                    absorbed: None,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let cls_norm = src.ln("cls.norm", d, true)?;
        let cls = src.lin("cls", d, d, true)?;
        let cls_out = src.lin("cls.output", 1, d, true)?;
        let max_pos = cfg
            .max_state_tokens
            .max(cfg.max_head_tokens.max(8) + 3 + MAX_OPTION_TOKENS + 1);
        Ok(S2Model {
            rope: Rope::new(cfg.head_dim(), cfg.rope_theta, max_pos),
            cfg,
            tok_emb,
            num_proj,
            type_emb,
            static_blocks,
            static_norm,
            state_blocks,
            loop_emb,
            attn_norm_loop,
            qkv_table: None,
            carry,
            exit_norms,
            fusion,
            cls_norm,
            cls,
            cls_out,
        })
    }

    /// Every linear layer.
    fn lins_mut(&mut self) -> Vec<&mut Lin> {
        let mut v: Vec<&mut Lin> = vec![&mut self.num_proj, &mut self.cls, &mut self.cls_out];
        for b in self
            .static_blocks
            .iter_mut()
            .chain(self.state_blocks.iter_mut())
        {
            v.extend([&mut b.qkv, &mut b.out, &mut b.down]);
            if let Some(up) = &mut b.up {
                v.push(up);
            }
        }
        for f in &mut self.fusion {
            v.extend([&mut f.q, &mut f.kv, &mut f.out, &mut f.up, &mut f.down]);
        }
        v
    }

    /// True if any matrix is stored ternary.
    pub fn has_ternary(&mut self) -> bool {
        self.lins_mut().iter().any(|l| l.is_ternary())
    }

    /// Compute every ternary matrix in f32 (dequantized at load; the file stays small).
    pub fn dequantize_ternary(&mut self) {
        for l in self.lins_mut() {
            l.dequantize();
        }
    }

    /// Repack every f32 matrix into k-major panels for the small-row products of a frame, after
    /// folding the first state projection ([`QkvTable`]) and the fusion keys / values
    /// ([`Absorbed`]) (ternary matrices kept packed stay as they are).
    pub fn pack_panels(&mut self) {
        self.prepare_compute(super::TernaryCompute::F32);
    }

    /// Set up the compute of every matrix for `mode`: `F32` dequantizes the ternary matrices,
    /// `Int8` turns them into int8 panels, `Packed` keeps them packed (without the folded
    /// weights, which would need f32 copies). f32 matrices are repacked into panels.
    pub fn prepare_compute(&mut self, mode: super::TernaryCompute) {
        use super::TernaryCompute;
        let mode = mode.resolve();
        if mode == TernaryCompute::F32 {
            self.dequantize_ternary();
        }
        if mode != TernaryCompute::Packed {
            self.fold_weights();
        }
        if mode == TernaryCompute::Int8 {
            for l in self.lins_mut() {
                l.ternary_to_i8();
            }
        }
        if mode == TernaryCompute::Lut {
            for l in self.lins_mut() {
                l.ternary_to_lut();
            }
        }
        for l in self.lins_mut() {
            l.pack_panels();
        }
    }

    /// Build the [`Absorbed`] fusion weights and the [`QkvTable`] from the f32 values of the
    /// weights.
    fn fold_weights(&mut self) {
        let (d, nh, hd) = (
            self.cfg.hidden_size,
            self.cfg.num_heads,
            self.cfg.head_dim(),
        );
        for f in &mut self.fusion {
            let w = f.kv.weight_f32();
            let b = f.kv.b.as_deref().expect("fusion attn_kv has a bias");
            let (wk, wv) = w.split_at(d * d);
            let kt = (0..nh)
                .map(|h| {
                    let rows = &wk[h * hd * d..(h + 1) * hd * d];
                    let mut t = vec![0.0; d * hd];
                    for c in 0..hd {
                        for i in 0..d {
                            t[i * hd + c] = rows[c * d + i];
                        }
                    }
                    PanelMatrix::from_rows(&t, d, hd)
                })
                .collect();
            let v = (0..nh)
                .map(|h| PanelMatrix::from_rows(&wv[h * hd * d..(h + 1) * hd * d], hd, d))
                .collect();
            f.absorbed = Some(Absorbed {
                kt,
                v,
                bk: b[..d].to_vec(),
                bv: b[d..2 * d].to_vec(),
            });
        }
        self.qkv_table = self.build_qkv_table();
    }

    /// The [`QkvTable`] of state block 0, when its `attn_qkv` is f32, it has no input norm and the
    /// table stays under 8 MB.
    fn build_qkv_table(&self) -> Option<QkvTable> {
        let c = &self.cfg;
        let (v, d, nf) = (c.vocab_size, c.hidden_size, c.num_features);
        let blk = self.state_blocks.first()?;
        if blk.attn_norm.is_some() || v * 3 * d > MAX_QKV_TABLE {
            return None;
        }
        let w = &blk.qkv.weight_f32();
        let emb: Vec<f32> = match &self.tok_emb {
            TokEmb::F32(e) => e.clone(),
            TokEmb::Q8 { q, scale } => q
                .iter()
                .enumerate()
                .map(|(i, &x)| x as f32 * scale[i / d])
                .collect(),
        };
        let mut tok = vec![0.0; v * 3 * d];
        kernels::matmul_f32(&emb, v, d, w, 3 * d, None, &mut tok, false);
        // U = W P: column j of P is num_proj.weight[:, j]
        let p = self.num_proj.weight_f32();
        let mut pt = vec![0.0; nf * d];
        for i in 0..d {
            for j in 0..nf {
                pt[j * d + i] = p[i * nf + j];
            }
        }
        let mut ut = vec![0.0; nf * 3 * d];
        kernels::matmul_f32(&pt, nf, d, w, 3 * d, None, &mut ut, false);
        let mut u = vec![0.0; 3 * d * nf];
        for j in 0..nf {
            for o in 0..3 * d {
                u[o * nf + j] = ut[j * 3 * d + o];
            }
        }
        let b = self.num_proj.b.as_deref().expect("num_proj has a bias");
        let mut num_bias = vec![0.0; 3 * d];
        kernels::matmul_f32(b, 1, d, w, 3 * d, None, &mut num_bias, false);
        Some(QkvTable {
            tok,
            num: PanelMatrix::from_rows(&u, 3 * d, nf),
            num_bias,
        })
    }

    /// Total number of parameters (ternary weights count one each).
    pub fn n_params(&self) -> usize {
        let (v, d) = (self.cfg.vocab_size, self.cfg.hidden_size);
        let block = |b: &Block| {
            b.attn_norm.as_ref().map_or(0, Ln::n_params)
                + b.qkv.n_params()
                + b.out.n_params()
                + b.ffn_norm.n_params()
                + b.up.as_ref().map_or(0, Lin::n_params)
                + b.down.n_params()
        };
        let mut n = v * d + self.num_proj.n_params() + self.type_emb.len();
        n += self.static_blocks.iter().map(block).sum::<usize>() + self.static_norm.n_params();
        n += self.state_blocks.iter().map(block).sum::<usize>();
        n += self.loop_emb.as_ref().map_or(0, Vec::len);
        n += self.attn_norm_loop.as_ref().map_or(0, Ln::n_params);
        n += self
            .carry
            .as_ref()
            .map_or(0, |(g, ln)| g.len() + ln.n_params());
        n += self.exit_norms.iter().map(Ln::n_params).sum::<usize>();
        for f in &self.fusion {
            n += f.q_norm.n_params()
                + f.kv_norm.n_params()
                + f.q.n_params()
                + f.kv.n_params()
                + f.out.n_params()
                + f.ffn_norm.n_params()
                + f.up.n_params()
                + f.down.n_params();
        }
        n + self.cls_norm.n_params() + self.cls.n_params() + self.cls_out.n_params()
    }

    /// Token embeddings plus the value features of the tokens carrying a value, into
    /// `out: [T][d]`: `e = tok_embd[id] + has_value · (num_proj.weight · φ(v) + num_proj.bias)`.
    /// An empty `has_value` means `id == [NUM]` (s2).
    pub fn embed(
        &self,
        ids: &[u32],
        values: &[f32],
        has_value: &[bool],
        out: &mut [f32],
    ) -> Result<()> {
        let mut phi = vec![0.0; ids.len() * NUM_FEATURES];
        self.embed_phi(ids, values, has_value, out, &mut phi)
    }

    /// [`S2Model::embed`] that also writes `φ(v)` of every token (`[T][17]`, zeros for tokens
    /// without a value).
    fn embed_phi(
        &self,
        ids: &[u32],
        values: &[f32],
        has_value: &[bool],
        out: &mut [f32],
        phi: &mut [f32],
    ) -> Result<()> {
        let d = self.cfg.hidden_size;
        for (r, &id) in ids.iter().enumerate() {
            let id = id as usize;
            if id >= self.cfg.vocab_size {
                return Err(Error::InvalidRequest(format!(
                    "token id {id} out of vocabulary"
                )));
            }
            let row = &mut out[r * d..(r + 1) * d];
            match &self.tok_emb {
                TokEmb::F32(w) => row.copy_from_slice(&w[id * d..(id + 1) * d]),
                TokEmb::Q8 { q, scale } => {
                    let s = scale[id];
                    for (o, &v) in row.iter_mut().zip(&q[id * d..(id + 1) * d]) {
                        *o = v as f32 * s;
                    }
                }
            }
            let hv = match has_value.get(r) {
                Some(&h) => h,
                None => id as u32 == self.cfg.num_token_id,
            };
            let f: &mut [f32; NUM_FEATURES] = (&mut phi[r * NUM_FEATURES..(r + 1) * NUM_FEATURES])
                .try_into()
                .expect("17 features");
            if hv {
                num::features(values.get(r).copied().unwrap_or(0.0), f);
                self.num_proj.forward(f, 1, row, true);
            } else {
                f.fill(0.0);
            }
        }
        Ok(())
    }

    /// One encoder block over `t` rows of `x` (positions `pos`). `attn_norm` is the input norm
    /// (`None`: identity, block 0 on the first pass).
    #[allow(clippy::too_many_arguments)]
    fn block(
        &self,
        blk: &Block,
        attn_norm: Option<&Ln>,
        x: &mut [f32],
        t: usize,
        pos: &[usize],
        mode: AttnMode<'_>,
        ws: &mut S2Workspace,
        table: Option<TableIn<'_>>,
    ) {
        let c = &self.cfg;
        let (d, ff, nh, hd) = (
            c.hidden_size,
            c.intermediate_size,
            c.num_heads,
            c.head_dim(),
        );
        let eps = c.norm_eps;
        let S2Workspace {
            hn,
            qkv,
            att,
            up,
            act,
            kvtmp,
            iota,
            attn,
            self_attn,
            ..
        } = ws;
        let x = &mut x[..t * d];
        let hn = grow(hn, t * d);
        let qkv = grow(qkv, t * 3 * d);
        // fused QKV + RoPE: each row is rotated right after its projection, while in cache
        match (table, attn_norm) {
            (
                Some(TableIn {
                    tb,
                    ids,
                    has_value,
                    phi,
                }),
                _,
            ) => {
                for (r, row) in qkv.chunks_exact_mut(3 * d).enumerate() {
                    let id = ids[r] as usize;
                    row.copy_from_slice(&tb.tok[id * 3 * d..(id + 1) * 3 * d]);
                    if has_value[r] {
                        tb.num.matmul(
                            &phi[r * NUM_FEATURES..(r + 1) * NUM_FEATURES],
                            NUM_FEATURES,
                            1,
                            Some(&tb.num_bias),
                            row,
                            3 * d,
                            true,
                        );
                    }
                }
            }
            (None, Some(n)) => {
                n.apply(x, d, eps, hn);
                blk.qkv.forward(hn, t, qkv, false);
            }
            (None, None) => blk.qkv.forward(x, t, qkv, false),
        }
        for (r, &p) in pos.iter().enumerate().take(t) {
            let row = &mut qkv[r * 3 * d..r * 3 * d + 2 * d];
            for h in 0..nh {
                self.rope.apply(&mut row[h * hd..(h + 1) * hd], p);
                self.rope.apply(&mut row[d + h * hd..d + (h + 1) * hd], p);
            }
        }
        let att = grow(att, t * d);
        match mode {
            AttnMode::Full if t <= SMALL_ATTN => {
                kernels::self_attention(qkv, t, nh, hd, att, d, self_attn);
            }
            AttnMode::Full => {
                attn.ensure(t, hd);
                let a = Attention {
                    k: &qkv[d..],
                    v: &qkv[2 * d..],
                    kv_stride: 3 * d,
                    n_kv: t,
                    n_heads: nh,
                    head_dim: hd,
                    window: None,
                };
                a.run(qkv, 3 * d, &iota[..t], att, d, attn);
            }
            AttnMode::Static { q_len, options } => {
                let longest = options.iter().map(|o| o.1).max().unwrap_or(0);
                attn.ensure(q_len + longest, hd);
                let a = Attention {
                    k: &qkv[d..],
                    v: &qkv[2 * d..],
                    kv_stride: 3 * d,
                    n_kv: q_len,
                    n_heads: nh,
                    head_dim: hd,
                    window: None,
                };
                a.run(qkv, 3 * d, &iota[..q_len], att, d, attn);
                for &(s, l) in options {
                    let kv = grow(kvtmp, (q_len + l) * 3 * d);
                    kv[..q_len * 3 * d].copy_from_slice(&qkv[..q_len * 3 * d]);
                    kv[q_len * 3 * d..].copy_from_slice(&qkv[s * 3 * d..(s + l) * 3 * d]);
                    let a = Attention {
                        k: &kv[d..],
                        v: &kv[2 * d..],
                        kv_stride: 3 * d,
                        n_kv: q_len + l,
                        n_heads: nh,
                        head_dim: hd,
                        window: None,
                    };
                    a.run(
                        &qkv[s * 3 * d..],
                        3 * d,
                        &iota[..l],
                        &mut att[s * d..],
                        d,
                        attn,
                    );
                }
            }
        }
        blk.out.forward(att, t, x, true);
        blk.ffn_norm.apply(x, d, eps, hn);
        if let Some(es) = &blk.es {
            let u = grow(up, t * es.units);
            let cand = grow(act, es.units);
            es.forward(hn, t, x, u, cand);
            return;
        }
        let up = grow(up, t * 2 * ff);
        blk.up
            .as_ref()
            .expect("GeGLU block has ffn_up")
            .forward(hn, t, up, false);
        let act = grow(act, t * ff);
        for r in 0..t {
            kernels::geglu(
                &up[r * 2 * ff..(r + 1) * 2 * ff],
                &mut act[r * ff..(r + 1) * ff],
            );
        }
        blk.down.forward(act, t, x, true);
    }

    /// Run the static encoder on a static stream and project the question tokens for every
    /// fusion block.
    pub fn encode_static(&self, st: &StaticTokens, ws: &mut S2Workspace) -> Result<StaticEnc> {
        let c = &self.cfg;
        let d = c.hidden_size;
        let t = st.ids.len();
        let q_len = st.q_len;
        if q_len == 0 || q_len > t || st.options.is_empty() {
            return Err(Error::InvalidRequest("empty question or options".into()));
        }
        let mut pos: Vec<usize> = (0..t).collect();
        let mut longest = 0;
        for &(s, l) in &st.options {
            if l == 0 || s < q_len || s + l > t {
                return Err(Error::InvalidRequest("malformed static stream".into()));
            }
            for j in 0..l {
                pos[s + j] = q_len + j;
            }
            longest = longest.max(l);
        }
        if q_len + longest > self.rope.cos.len() / self.rope.half.max(1) {
            return Err(Error::InvalidRequest(format!(
                "static stream of {} tokens is too long",
                q_len + longest
            )));
        }
        ws.ensure_iota(t);
        let mut x = vec![0.0; t * d];
        self.embed(&st.ids, &st.values, &st.has_value, &mut x)?;
        let mode = AttnMode::Static {
            q_len,
            options: &st.options,
        };
        for blk in &self.static_blocks {
            self.block(blk, blk.attn_norm.as_ref(), &mut x, t, &pos, mode, ws, None);
        }
        kernels::layer_norm_inplace(&mut x, d, &self.static_norm.w, None, c.norm_eps);
        let mut markers = Vec::with_capacity(st.options.len() * d);
        for &(s, _) in &st.options {
            markers.extend_from_slice(&x[s * d..(s + 1) * d]);
        }
        let q_tokens = &x[..q_len * d];
        let kv: Vec<Vec<f32>> = self
            .fusion
            .iter()
            .map(|f| {
                let hn = grow(&mut ws.hn, q_len * d);
                f.kv_norm.apply(q_tokens, d, c.norm_eps, hn);
                let mut kv = vec![0.0; q_len * 2 * d];
                f.kv.forward(hn, q_len, &mut kv, false);
                kv
            })
            .collect();
        let kv_panels = self.question_panels(&kv, q_len);
        Ok(StaticEnc {
            q_len,
            n_options: st.options.len(),
            markers,
            kv,
            kv_panels,
            n_tokens: t,
            fusion0: None,
        })
    }

    /// The projected question tokens of every fusion block (`[q][2d]`) in per-head panel
    /// layouts.
    fn question_panels(&self, kv: &[Vec<f32>], q: usize) -> Vec<Vec<HeadPanels>> {
        let c = &self.cfg;
        let (d, nh, hd) = (c.hidden_size, c.num_heads, c.head_dim());
        kv.iter()
            .map(|kv| {
                (0..nh)
                    .map(|h| {
                        let mut p = HeadPanels {
                            k: vec![0.0; kernels::panel_len(q, hd)],
                            v: vec![0.0; kernels::panel_len(hd, q)],
                        };
                        kernels::pack_outputs(&kv[h * hd..], 2 * d, q, hd, &mut p.k);
                        kernels::pack_inputs(&kv[d + h * hd..], 2 * d, q, hd, &mut p.v);
                        p
                    })
                    .collect()
            })
            .collect()
    }

    /// Precompute the question-type dependent part of the first fusion block (see [`Fusion0`]).
    pub fn prepare_fusion(&self, enc: &mut StaticEnc, qtype: usize) -> Result<()> {
        let c = &self.cfg;
        let (d, nh, hd) = (c.hidden_size, c.num_heads, c.head_dim());
        if qtype >= N_QTYPES {
            return Err(Error::InvalidRequest(format!("question type {qtype}")));
        }
        if enc.fusion0.as_ref().is_some_and(|f| f.qtype == qtype) {
            return Ok(());
        }
        let n = enc.n_options;
        let te = &self.type_emb[qtype * d..(qtype + 1) * d];
        let mut x0 = enc.markers.clone();
        for row in x0.chunks_exact_mut(d) {
            row.iter_mut().zip(te).for_each(|(o, &e)| *o += e);
        }
        let Some(f) = self.fusion.first() else {
            enc.fusion0 = Some(Fusion0 {
                qtype,
                x0,
                q0: Vec::new(),
                prior: AttnPartial::default(),
            });
            return Ok(());
        };
        let mut h = vec![0.0; n * d];
        f.q_norm.apply(&x0, d, c.norm_eps, &mut h);
        let mut q0 = vec![0.0; n * d];
        f.q.forward(&h, n, &mut q0, false);
        let kv = &enc.kv[0];
        let seg = KvSeg {
            k: kv,
            v: &kv[d..],
            stride: 2 * d,
            n: enc.q_len,
        };
        let prior = kernels::attend_partial(&q0, d, n, seg, nh, hd);
        enc.fusion0 = Some(Fusion0 {
            qtype,
            x0,
            q0,
            prior,
        });
        Ok(())
    }

    /// A state encoding with buffers for `max_state_tokens` (no token yet).
    pub fn state_buffers(&self) -> StateEnc {
        let c = &self.cfg;
        let (d, smax, ne) = (c.hidden_size, c.max_state_tokens, c.exits.len());
        StateEnc {
            len: 0,
            x: vec![0.0; smax * d],
            layers_done: 0,
            exit_ready: vec![false; ne],
            exit_h: vec![vec![0.0; smax * d]; ne],
            kv_ready: vec![false; ne],
            exit_kv: vec![
                vec![
                    vec![0.0; (smax * 2 * d).max(state_panels_len(smax, d))];
                    c.fusion_layers
                ];
                ne
            ],
            ids: Vec::with_capacity(smax),
            has_value: Vec::with_capacity(smax),
            phi: vec![0.0; smax * NUM_FEATURES],
            carried: false,
        }
    }

    /// Embed a state stream into reused buffers (no block run yet).
    pub fn state_reset(
        &self,
        st: &mut StateEnc,
        ids: &[u32],
        values: &[f32],
        has_value: &[bool],
    ) -> Result<()> {
        let c = &self.cfg;
        let d = c.hidden_size;
        let s = ids.len();
        if s == 0 || s > c.max_state_tokens {
            return Err(Error::InvalidRequest(format!(
                "state stream of {s} tokens (max {})",
                c.max_state_tokens
            )));
        }
        if st.x.len() < c.max_state_tokens * d || st.exit_ready.len() != c.exits.len() {
            *st = self.state_buffers();
        }
        st.len = s;
        st.layers_done = 0;
        st.carried = false;
        st.exit_ready.fill(false);
        st.kv_ready.fill(false);
        st.ids.clear();
        st.ids.extend_from_slice(ids);
        st.has_value.clear();
        st.has_value.extend((0..s).map(|r| {
            has_value
                .get(r)
                .copied()
                .unwrap_or(ids[r] == c.num_token_id)
        }));
        self.embed_phi(
            ids,
            values,
            &st.has_value,
            &mut st.x[..s * d],
            &mut st.phi[..s * NUM_FEATURES],
        )
    }

    /// SPEC-S3 §7: start the stream `st` (just embedded by [`S2Model::state_reset`]) from the
    /// previous frame's final latent `prev` (`[S][d]`, the output of the last block applied, with
    /// the same token ids): `h⁰ = embed + carry_gate ⊙ LN₀(prev)`. No effect without
    /// `temporal_carry` weights.
    pub fn state_carry(&self, st: &mut StateEnc, prev: &[f32], ws: &mut S2Workspace) {
        let Some((gate, ln)) = &self.carry else {
            return;
        };
        let (d, s) = (self.cfg.hidden_size, st.len);
        let h = grow(&mut ws.hn, d);
        for (row, p) in st.x[..s * d]
            .chunks_exact_mut(d)
            .zip(prev[..s * d].chunks_exact(d))
        {
            ln.apply(p, d, self.cfg.norm_eps, h);
            for ((x, &g), &v) in row.iter_mut().zip(gate).zip(h.iter()) {
                *x += g * v;
            }
        }
        st.carried = true;
    }

    /// Embed a state stream (no block run yet).
    pub fn state_start(&self, ids: &[u32], values: &[f32]) -> Result<StateEnc> {
        let mut st = self.state_buffers();
        self.state_reset(&mut st, ids, values, &[])?;
        Ok(st)
    }

    /// Run the state encoder up to effective depth `to` (1-based block applications: block
    /// `a % state_blocks` on pass `a / state_blocks`); `H_e` is kept for every exit passed.
    pub fn state_advance(&self, st: &mut StateEnc, to: usize, ws: &mut S2Workspace) {
        let c = &self.cfg;
        let (d, s, nb) = (c.hidden_size, st.len, c.state_blocks);
        let to = to.min(c.state_layers);
        if st.layers_done >= to {
            return;
        }
        ws.ensure_iota(s);
        // positions 0..S, kept apart from `ws.iota`, which the tiled attention of streams longer
        // than SMALL_ATTN reads
        let mut pos = std::mem::take(&mut ws.spos);
        if pos.len() < s {
            pos = (0..s).collect();
        }
        while st.layers_done < to {
            let a = st.layers_done;
            let (pass, b) = (a / nb, a % nb);
            let blk = &self.state_blocks[b];
            let norm = if b == 0 && pass > 0 {
                if let Some(le) = &self.loop_emb {
                    let e = &le[pass * d..(pass + 1) * d];
                    for row in st.x[..s * d].chunks_exact_mut(d) {
                        row.iter_mut().zip(e).for_each(|(o, &v)| *o += v);
                    }
                }
                self.attn_norm_loop.as_ref().or(blk.attn_norm.as_ref())
            } else {
                blk.attn_norm.as_ref()
            };
            let table = match &self.qkv_table {
                Some(tb) if a == 0 && !st.carried => Some(TableIn {
                    tb,
                    ids: &st.ids,
                    has_value: &st.has_value,
                    phi: &st.phi,
                }),
                _ => None,
            };
            self.block(
                blk,
                norm,
                &mut st.x,
                s,
                &pos[..s],
                AttnMode::Full,
                ws,
                table,
            );
            st.layers_done += 1;
            if let Some(e) = c.exits.iter().position(|&x| x == st.layers_done) {
                self.exit_norms[e].apply(&st.x[..s * d], d, c.norm_eps, &mut st.exit_h[e][..s * d]);
                st.exit_ready[e] = true;
            }
        }
        ws.spos = pos;
    }

    /// The state side of every fusion block for exit `e` (runs the state encoder as deep as
    /// needed, cached in `st`): `LN_kv(H_e)` (`S` rows of `[d]`) for blocks with [`Absorbed`]
    /// weights, else `attn_kv(LN_kv(H_e))` (`S` rows of `[2d]`). The buffers may be longer.
    pub fn exit_kv<'s>(
        &self,
        st: &'s mut StateEnc,
        e: usize,
        ws: &mut S2Workspace,
    ) -> &'s [Vec<f32>] {
        let c = &self.cfg;
        let (d, s) = (c.hidden_size, st.len);
        if !st.kv_ready[e] {
            if !st.exit_ready[e] {
                self.state_advance(st, c.exits[e], ws);
            }
            let h = &st.exit_h[e][..s * d];
            for (f, kv) in self.fusion.iter().zip(st.exit_kv[e].iter_mut()) {
                if f.absorbed.is_some() {
                    let hn = grow(&mut ws.hn, s * d);
                    f.kv_norm.apply(h, d, c.norm_eps, hn);
                    state_panels(hn, s, d, kv);
                } else {
                    let hn = grow(&mut ws.hn, s * d);
                    f.kv_norm.apply(h, d, c.norm_eps, hn);
                    f.kv.forward(hn, s, &mut kv[..s * 2 * d], false);
                }
            }
            st.kv_ready[e] = true;
        }
        &st.exit_kv[e]
    }

    /// Fusion head and scorer: one logit per option. `state_kv` / `static_kv` hold, per fusion
    /// block, the projected keys and values of the `s` state tokens and `q` question tokens.
    #[allow(clippy::too_many_arguments)]
    pub fn fusion(
        &self,
        markers: &[f32],
        n: usize,
        qtype: usize,
        state_kv: &[Vec<f32>],
        s: usize,
        static_kv: &[Vec<f32>],
        q: usize,
        ws: &mut S2Workspace,
        logits: &mut Vec<f32>,
    ) -> Result<()> {
        let d = self.cfg.hidden_size;
        if qtype >= N_QTYPES {
            return Err(Error::InvalidRequest(format!("question type {qtype}")));
        }
        let te = &self.type_emb[qtype * d..(qtype + 1) * d];
        let mx = grow(&mut ws.mx, n * d);
        for (i, row) in mx.chunks_exact_mut(d).enumerate() {
            for ((o, &m), &e) in row.iter_mut().zip(&markers[i * d..(i + 1) * d]).zip(te) {
                *o = m + e;
            }
        }
        let panels = self.question_panels(static_kv, q);
        self.fusion_blocks(
            n,
            None,
            state_kv,
            s,
            static_kv,
            Some(&panels),
            q,
            ws,
            logits,
        );
        Ok(())
    }

    /// [`S2Model::fusion`] on a cached static encoding prepared by [`S2Model::prepare_fusion`]:
    /// the first block reuses the precomputed queries and question-token attention.
    pub fn fusion_cached(
        &self,
        enc: &StaticEnc,
        state_kv: &[Vec<f32>],
        s: usize,
        ws: &mut S2Workspace,
        logits: &mut Vec<f32>,
    ) -> Result<()> {
        let d = self.cfg.hidden_size;
        let f0 = enc
            .fusion0
            .as_ref()
            .ok_or_else(|| Error::InvalidRequest("static encoding not prepared".into()))?;
        let n = enc.n_options;
        grow(&mut ws.mx, n * d).copy_from_slice(&f0.x0);
        self.fusion_blocks(
            n,
            Some(f0),
            state_kv,
            s,
            &enc.kv,
            Some(&enc.kv_panels),
            enc.q_len,
            ws,
            logits,
        );
        Ok(())
    }

    /// The fusion blocks and the scorer on `ws.mx` (`[n][d]`, the markers plus the type
    /// embedding).
    #[allow(clippy::too_many_arguments)]
    fn fusion_blocks(
        &self,
        n: usize,
        f0: Option<&Fusion0>,
        state_kv: &[Vec<f32>],
        s: usize,
        static_kv: &[Vec<f32>],
        static_panels: Option<&[Vec<HeadPanels>]>,
        q: usize,
        ws: &mut S2Workspace,
        logits: &mut Vec<f32>,
    ) {
        let c = &self.cfg;
        let (d, nh, hd, eps) = (c.hidden_size, c.num_heads, c.head_dim(), c.norm_eps);
        let S2Workspace {
            mx,
            mh,
            mq,
            mup,
            att,
            scores,
            ab_t,
            ab_u,
            ab_o,
            ..
        } = ws;
        let mx = &mut mx[..n * d];
        let mh = grow(mh, n * d);
        for (b, f) in self.fusion.iter().enumerate() {
            let sk = &state_kv[b];
            let att = grow(att, n * d);
            let prior = match f0 {
                Some(f0) if b == 0 => Some(f0),
                _ => None,
            };
            let qrows: &[f32] = match prior {
                Some(f0) => &f0.q0,
                None => {
                    f.q_norm.apply(mx, d, eps, mh);
                    let mq = grow(mq, n * d);
                    f.q.forward(mh, n, mq, false);
                    mq
                }
            };
            let qk = &static_kv[b];
            let question = KvSeg {
                k: qk,
                v: &qk[d..],
                stride: 2 * d,
                n: q,
            };
            match &f.absorbed {
                Some(ab) => {
                    let bufs = AbsorbBufs {
                        t: grow(ab_t, n * d),
                        u: grow(ab_u, n * d),
                        o: grow(ab_o, n * hd),
                        scores,
                    };
                    let question = match (prior, static_panels) {
                        (None, Some(p)) => Some((&p[b][..], q)),
                        _ => None,
                    };
                    absorbed_attention(
                        ab,
                        qrows,
                        n,
                        &sk[..state_panels_len(s, d)],
                        s,
                        question,
                        prior.map(|f0| &f0.prior),
                        nh,
                        hd,
                        att,
                        bufs,
                    );
                }
                None => {
                    let state = KvSeg {
                        k: sk,
                        v: &sk[d..],
                        stride: 2 * d,
                        n: s,
                    };
                    match prior {
                        Some(f0) => kernels::attend(
                            qrows,
                            d,
                            n,
                            &[state],
                            nh,
                            hd,
                            Some(&f0.prior),
                            att,
                            d,
                            scores,
                        ),
                        None => kernels::attend(
                            qrows,
                            d,
                            n,
                            &[state, question],
                            nh,
                            hd,
                            None,
                            att,
                            d,
                            scores,
                        ),
                    }
                }
            }
            f.out.forward(att, n, mx, true);
            f.ffn_norm.apply(mx, d, eps, mh);
            let mup = grow(mup, n * 2 * d);
            f.up.forward(mh, n, mup, false);
            mup.iter_mut().for_each(|v| *v = v.max(0.0));
            f.down.forward(mup, n, mx, true);
        }
        self.cls_norm.apply(mx, d, eps, mh);
        let s1 = grow(mq, n * d);
        self.cls.forward(mh, n, s1, false);
        s1.iter_mut().for_each(|v| *v = kernels::gelu(*v));
        logits.clear();
        logits.resize(n, 0.0);
        self.cls_out.forward(s1, n, logits, false);
    }
}

/// Scratch buffers of [`absorbed_attention`].
struct AbsorbBufs<'a> {
    /// `[n][d]`: `W_kᵀ q` per query (one head at a time).
    t: &'a mut [f32],
    /// `[n][d]`: `Σ_j e_j h_j` per query.
    u: &'a mut [f32],
    /// `[n][hd]`: the question-token part of the output.
    o: &'a mut [f32],
    /// `[n][s + q]`: scores, then their exponentials.
    scores: &'a mut Vec<f32>,
}

/// The question tokens of one head of a fusion block in panel layouts: keys (`q` outputs,
/// `head_dim` inputs) and values (`head_dim` outputs, `q` inputs).
#[derive(Clone, Debug, Default)]
pub struct HeadPanels {
    pub k: Vec<f32>,
    pub v: Vec<f32>,
}

/// Cross-attention of `n` queries (`q: [n][d]`) over the `s` state tokens (normalized, not
/// projected: see [`Absorbed`]; `hp` holds them in the panel layouts of [`state_panels`]) and
/// the `nq` question tokens of `question` (per head), or the question-token softmax state
/// `prior` (first fusion block). Every product is a panel product with `n` rows. Writes
/// `out: [n][d]`.
#[allow(clippy::too_many_arguments)]
fn absorbed_attention(
    ab: &Absorbed,
    q: &[f32],
    n: usize,
    hp: &[f32],
    s: usize,
    question: Option<(&[HeadPanels], usize)>,
    prior: Option<&AttnPartial>,
    nh: usize,
    hd: usize,
    out: &mut [f32],
    bufs: AbsorbBufs<'_>,
) {
    let d = nh * hd;
    let scale = 1.0 / (hd as f32).sqrt();
    let AbsorbBufs { t, u, o, scores } = bufs;
    let nq = question.map_or(0, |(_, nq)| nq);
    let ls = s + nq;
    if scores.len() < n * ls {
        scores.resize(n * ls, 0.0);
    }
    let sc = &mut scores[..n * ls];
    let (hk, hv) = hp.split_at(kernels::panel_len(s, d));
    for c in 0..nh {
        let hr = c * hd..(c + 1) * hd;
        let qh = &q[hr.start..];
        // t_i = W_k[head]ᵀ q_i[head], then the state scores t_i · h_j
        ab.kt[c].matmul(qh, d, n, None, t, d, false);
        kernels::panel_matmul(hk, s, d, t, d, n, None, sc, ls, false);
        if let Some((qp, _)) = question {
            kernels::panel_matmul(&qp[c].k, nq, hd, qh, d, n, None, &mut sc[s..], ls, false);
        }
        for i in 0..n {
            let qb = kernels::dot(&q[i * d + hr.start..i * d + hr.end], &ab.bk[hr.clone()]);
            let row = &mut sc[i * ls..(i + 1) * ls];
            row[..s].iter_mut().for_each(|v| *v = (*v + qb) * scale);
            row[s..].iter_mut().for_each(|v| *v *= scale);
            let m = row.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let (pm, pl) = prior.map_or((f32::NEG_INFINITY, 0.0), |p| {
                (p.m[i * nh + c], p.l[i * nh + c])
            });
            let mm = m.max(pm);
            let mut es = 0.0;
            for v in row[..s].iter_mut() {
                *v = kernels::fast_exp(*v - mm);
                es += *v;
            }
            let mut l = es;
            for v in row[s..].iter_mut() {
                *v = kernels::fast_exp(*v - mm);
                l += *v;
            }
            let oi = &mut o[i * hd..(i + 1) * hd];
            match prior {
                Some(p) if pl > 0.0 => {
                    let f = kernels::fast_exp(pm - mm);
                    l += pl * f;
                    let po = &p.o[i * d + hr.start..i * d + hr.end];
                    oi.iter_mut().zip(po).for_each(|(a, &b)| *a = b * f);
                }
                _ => oi.fill(0.0),
            }
            // keep 1/l and Σ e_j of the state tokens for the value projection below
            t[i * d] = 1.0 / l;
            t[i * d + 1] = es;
        }
        // u_i = Σ_j e_j h_j; question part o_i += Σ_j e_j v_j
        kernels::panel_matmul(hv, d, s, sc, ls, n, None, u, d, false);
        if let Some((qp, _)) = question {
            kernels::panel_matmul(&qp[c].v, hd, nq, &sc[s..], ls, n, None, o, hd, true);
        }
        // out_i[head] = (W_v[head] u_i + b_v[head] Σ e_j + o_i) / l_i
        ab.v[c].matmul(u, d, n, None, &mut out[hr.start..], d, false);
        for i in 0..n {
            let (inv, es) = (t[i * d], t[i * d + 1]);
            let row = &mut out[i * d + hr.start..i * d + hr.end];
            for ((r, &bv), &oq) in row
                .iter_mut()
                .zip(&ab.bv[hr.clone()])
                .zip(&o[i * hd..(i + 1) * hd])
            {
                *r = (*r + bv * es + oq) * inv;
            }
        }
    }
}

/// `h: [s][d]` in the two panel layouts used by [`absorbed_attention`], written to `dst`: as
/// outputs (`s` outputs, `d` inputs, for the scores) then as inputs (`d` outputs, `s` inputs,
/// for `Σ e_j h_j`).
fn state_panels(h: &[f32], s: usize, d: usize, dst: &mut [f32]) {
    let (a, b) = dst.split_at_mut(kernels::panel_len(s, d));
    kernels::pack_outputs(h, d, s, d, a);
    kernels::pack_inputs(h, d, s, d, b);
}

/// Floats of [`state_panels`] for `s` tokens.
fn state_panels_len(s: usize, d: usize) -> usize {
    kernels::panel_len(s, d) + kernels::panel_len(d, s)
}
