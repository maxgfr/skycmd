//! Weights and forward pieces of an s2 model (SPEC-S2 §3).
//!
//! The engine drives these pieces: [`S2Model::encode_static`] once per (question, options),
//! [`S2Model::state_start`] / [`S2Model::state_advance`] / [`S2Model::exit_kv`] once per state,
//! and [`S2Model::fusion`] once per question and exit.

use sky_gguf::{GgmlType, Gguf};

use super::S2Config;
use crate::kernels::{self, Attention, AttnScratch, Rope};
use crate::num::{self, NUM_FEATURES};
use crate::prompt::MAX_OPTION_TOKENS;
use crate::ternary::{self, TernaryMatrix};
use crate::{Error, Result};

/// Number of question types (choice, score, noul).
pub const N_QTYPES: usize = 3;

/// Weight storage of a linear layer.
#[derive(Clone, Debug)]
pub enum LinWeight {
    F32(Vec<f32>),
    Ternary(TernaryMatrix),
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
            LinWeight::Ternary(t) => t.matmul(x, rows, self.b.as_deref(), y, accumulate),
        }
    }

    pub fn is_ternary(&self) -> bool {
        matches!(self.w, LinWeight::Ternary(_))
    }

    /// Replace ternary weights by their f32 values (same results up to rounding).
    pub fn dequantize(&mut self) {
        if let LinWeight::Ternary(t) = &self.w {
            self.w = LinWeight::F32(t.to_f32());
        }
    }

    /// The weights as f32 values.
    pub fn weight_f32(&self) -> Vec<f32> {
        match &self.w {
            LinWeight::F32(w) => w.clone(),
            LinWeight::Ternary(t) => t.to_f32(),
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
    pub up: Lin,
    pub down: Lin,
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
    pub state_blocks: Vec<Block>,
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
    /// Number values (0 for non-number tokens), aligned with `ids`.
    pub values: Vec<f32>,
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
    pub n_tokens: usize,
}

/// The (partially) encoded state stream of one state text.
#[derive(Clone, Debug)]
pub struct StateEnc {
    /// Number of tokens `S`.
    pub len: usize,
    /// Hidden states after `layers_done` blocks, `[S][d]`.
    pub x: Vec<f32>,
    pub layers_done: usize,
    /// Per exit: `H_e` once the encoder went past it.
    pub exit_h: Vec<Option<Vec<f32>>>,
    /// Per exit, per fusion block: `attn_kv(LN_kv(H_e))`, `[S][2d]`.
    pub exit_kv: Vec<Option<Vec<Vec<f32>>>>,
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
    mx: Vec<f32>,
    mh: Vec<f32>,
    mq: Vec<f32>,
    mup: Vec<f32>,
    attn: AttnScratch,
}

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
        Ok(Block {
            attn_norm,
            qkv: self.lin(&p("attn_qkv"), 3 * d, d, false)?,
            out: self.lin(&p("attn_output"), d, d, false)?,
            ffn_norm: self.ln(&p("ffn_norm"), d, false)?,
            up: self.lin(&p("ffn_up"), 2 * ff, d, false)?,
            down: self.lin(&p("ffn_down"), d, ff, false)?,
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
        let state_blocks = (0..cfg.state_layers)
            .map(|n| src.block("blk.", n, &cfg))
            .collect::<Result<Vec<_>>>()?;
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
            v.extend([&mut b.qkv, &mut b.out, &mut b.up, &mut b.down]);
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

    /// Total number of parameters (ternary weights count one each).
    pub fn n_params(&self) -> usize {
        let (v, d) = (self.cfg.vocab_size, self.cfg.hidden_size);
        let block = |b: &Block| {
            b.attn_norm.as_ref().map_or(0, Ln::n_params)
                + b.qkv.n_params()
                + b.out.n_params()
                + b.ffn_norm.n_params()
                + b.up.n_params()
                + b.down.n_params()
        };
        let mut n = v * d + self.num_proj.n_params() + self.type_emb.len();
        n += self.static_blocks.iter().map(block).sum::<usize>() + self.static_norm.n_params();
        n += self.state_blocks.iter().map(block).sum::<usize>();
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

    /// Token embeddings plus the value features of the `[NUM]` tokens, into `out: [T][d]`.
    pub fn embed(&self, ids: &[u32], values: &[f32], out: &mut [f32]) -> Result<()> {
        let d = self.cfg.hidden_size;
        let mut phi = [0.0f32; NUM_FEATURES];
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
            if id as u32 == self.cfg.num_token_id {
                num::features(values.get(r).copied().unwrap_or(0.0), &mut phi);
                self.num_proj.forward(&phi, 1, row, true);
            }
        }
        Ok(())
    }

    /// One encoder block over `t` rows of `x` (positions `pos`).
    fn block(
        &self,
        blk: &Block,
        x: &mut [f32],
        t: usize,
        pos: &[usize],
        mode: AttnMode<'_>,
        ws: &mut S2Workspace,
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
            ..
        } = ws;
        let x = &mut x[..t * d];
        let hn = grow(hn, t * d);
        let qkv = grow(qkv, t * 3 * d);
        match &blk.attn_norm {
            Some(n) => {
                n.apply(x, d, eps, hn);
                blk.qkv.forward(hn, t, qkv, false);
            }
            None => blk.qkv.forward(x, t, qkv, false),
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
        let up = grow(up, t * 2 * ff);
        blk.up.forward(hn, t, up, false);
        let act = grow(act, t * ff);
        for r in 0..t {
            let (u, a) = (
                &up[r * 2 * ff..(r + 1) * 2 * ff],
                &mut act[r * ff..(r + 1) * ff],
            );
            for j in 0..ff {
                a[j] = kernels::gelu(u[j]) * u[ff + j];
            }
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
        self.embed(&st.ids, &st.values, &mut x)?;
        let mode = AttnMode::Static {
            q_len,
            options: &st.options,
        };
        for blk in &self.static_blocks {
            self.block(blk, &mut x, t, &pos, mode, ws);
        }
        kernels::layer_norm_inplace(&mut x, d, &self.static_norm.w, None, c.norm_eps);
        let mut markers = Vec::with_capacity(st.options.len() * d);
        for &(s, _) in &st.options {
            markers.extend_from_slice(&x[s * d..(s + 1) * d]);
        }
        let q_tokens = &x[..q_len * d];
        let kv = self
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
        Ok(StaticEnc {
            q_len,
            n_options: st.options.len(),
            markers,
            kv,
            n_tokens: t,
        })
    }

    /// Embed a state stream (no block run yet).
    pub fn state_start(&self, ids: &[u32], values: &[f32]) -> Result<StateEnc> {
        let d = self.cfg.hidden_size;
        let s = ids.len();
        if s == 0 || s > self.cfg.max_state_tokens {
            return Err(Error::InvalidRequest(format!(
                "state stream of {s} tokens (max {})",
                self.cfg.max_state_tokens
            )));
        }
        let mut x = vec![0.0; s * d];
        self.embed(ids, values, &mut x)?;
        let n_exits = self.cfg.exits.len();
        Ok(StateEnc {
            len: s,
            x,
            layers_done: 0,
            exit_h: vec![None; n_exits],
            exit_kv: vec![None; n_exits],
        })
    }

    /// Run the state blocks up to depth `to` (1-based block count); `H_e` is kept for every exit
    /// passed on the way.
    pub fn state_advance(&self, st: &mut StateEnc, to: usize, ws: &mut S2Workspace) {
        let c = &self.cfg;
        let (d, s) = (c.hidden_size, st.len);
        let to = to.min(self.state_blocks.len());
        if st.layers_done >= to {
            return;
        }
        ws.ensure_iota(s);
        let pos: Vec<usize> = (0..s).collect();
        while st.layers_done < to {
            let l = st.layers_done;
            self.block(
                &self.state_blocks[l],
                &mut st.x,
                s,
                &pos,
                AttnMode::Full,
                ws,
            );
            st.layers_done += 1;
            if let Some(e) = c.exits.iter().position(|&x| x == st.layers_done) {
                let mut h = vec![0.0; s * d];
                self.exit_norms[e].apply(&st.x, d, c.norm_eps, &mut h);
                st.exit_h[e] = Some(h);
            }
        }
    }

    /// `attn_kv(LN_kv(H_e))` of exit `e` for every fusion block (runs the state encoder as deep as
    /// needed, cached in `st`).
    pub fn exit_kv<'s>(
        &self,
        st: &'s mut StateEnc,
        e: usize,
        ws: &mut S2Workspace,
    ) -> &'s [Vec<f32>] {
        let c = &self.cfg;
        let (d, s) = (c.hidden_size, st.len);
        if st.exit_kv[e].is_none() {
            if st.exit_h[e].is_none() {
                self.state_advance(st, c.exits[e], ws);
            }
            let h = st.exit_h[e].as_ref().expect("exit reached");
            let kv = self
                .fusion
                .iter()
                .map(|f| {
                    let hn = grow(&mut ws.hn, s * d);
                    f.kv_norm.apply(h, d, c.norm_eps, hn);
                    let mut kv = vec![0.0; s * 2 * d];
                    f.kv.forward(hn, s, &mut kv, false);
                    kv
                })
                .collect();
            st.exit_kv[e] = Some(kv);
        }
        st.exit_kv[e].as_deref().expect("computed")
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
        let c = &self.cfg;
        let (d, nh, hd, eps) = (c.hidden_size, c.num_heads, c.head_dim(), c.norm_eps);
        if qtype >= N_QTYPES {
            return Err(Error::InvalidRequest(format!("question type {qtype}")));
        }
        let S2Workspace {
            kvtmp,
            iota,
            mx,
            mh,
            mq,
            mup,
            att,
            attn,
            ..
        } = ws;
        if iota.len() < n {
            *iota = (0..n).collect();
        }
        let mx = grow(mx, n * d);
        let te = &self.type_emb[qtype * d..(qtype + 1) * d];
        for (i, row) in mx.chunks_exact_mut(d).enumerate() {
            for ((o, &m), &e) in row.iter_mut().zip(&markers[i * d..(i + 1) * d]).zip(te) {
                *o = m + e;
            }
        }
        let mh = grow(mh, n * d);
        let n_kv = s + q;
        attn.ensure(n_kv, hd);
        for (b, f) in self.fusion.iter().enumerate() {
            f.q_norm.apply(mx, d, eps, mh);
            let mq = grow(mq, n * d);
            f.q.forward(mh, n, mq, false);
            let kv = grow(kvtmp, n_kv * 2 * d);
            kv[..s * 2 * d].copy_from_slice(&state_kv[b][..s * 2 * d]);
            kv[s * 2 * d..].copy_from_slice(&static_kv[b][..q * 2 * d]);
            let kv: &[f32] = kv;
            let a = Attention {
                k: kv,
                v: &kv[d..],
                kv_stride: 2 * d,
                n_kv,
                n_heads: nh,
                head_dim: hd,
                window: None,
            };
            let att = grow(att, n * d);
            a.run(mq, d, &iota[..n], att, d, attn);
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
        Ok(())
    }
}
