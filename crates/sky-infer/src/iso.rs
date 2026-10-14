//! `skycmd-iso`: the s5 decision model that is invariant by construction (sky-train
//! `s5/iso.rs`). A ModernBERT encoder reads three kinds of segments:
//!
//! - the state `[CLS] state [SEP]`, which attends to itself only: its keys and values at every
//!   layer depend on the state alone and are cached per state (an LRU of [`StateKv`]);
//! - the question `"{type} question: {instructions}" [SEP]`, which attends to the state and
//!   itself;
//! - one segment per option, rendered by its **definition only** (the option names never enter
//!   the model), which attends to the state, the question and itself, never to another option.
//!
//! Every option starts at the same logical position (`positions = shared`: right after the
//! question; `reset`: 0) and the local window of the sliding-window layers is measured on these
//! positions. An option's score is a function of (state, question, its own definition) only:
//! permuting the options permutes the scores bit for bit, renaming them changes nothing, and
//! swapping the two definitions of a `noul` question swaps its probabilities.
//!
//! Readout: mean of the option's final hidden states `h`, mean of the question's `q`, then
//! `score = W1 · gelu(W0 · [h, q, h ⊙ q, |h - q|] + b0) + b1` and a softmax with the
//! temperature of the question type.

use sky_gguf::Gguf;

use crate::kernels::{self, ActBlock, ActQuant, Attention, AttnScratch, Rope, QK};
use crate::model::{linear, Act, Embedding, EncoderLayer, Linear, Norm, Weight};
use crate::prompt::{self, QType, QuestionKind};
use crate::tokenizer::{BpeCache, Tokenizer};
use crate::{Error, Result};

/// `general.architecture` of the iso files.
pub const ARCH: &str = "skycmd-iso";

/// Where the options start.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Positions {
    /// Right after the question (sub-variant i).
    Shared,
    /// At 0 (sub-variant ii).
    Reset,
}

/// Hyper-parameters of an iso model.
#[derive(Clone, Debug, PartialEq)]
pub struct IsoConfig {
    pub name: String,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub norm_eps: f32,
    pub global_rope_theta: f32,
    pub local_rope_theta: f32,
    /// Full local window (`|i - j| <= local_attention / 2`); 0 = every layer global.
    pub local_attention: usize,
    pub global_attn_every_n_layers: usize,
    pub max_len: usize,
    pub head_max_len: usize,
    pub max_option_tokens: usize,
    pub positions: Positions,
    pub temperatures: Vec<(String, f32)>,
    /// `(lang, false definition, true definition)` of definition-less `noul` questions.
    pub noul: Vec<(String, String, String)>,
}

impl IsoConfig {
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_heads
    }

    pub fn is_global(&self, l: usize) -> bool {
        self.local_attention == 0
            || self.global_attn_every_n_layers <= 1
            || l.is_multiple_of(self.global_attn_every_n_layers)
    }

    /// Tokens of the state segment (`[CLS] state [SEP]`), independent of the question.
    pub fn state_budget(&self) -> usize {
        self.max_len.saturating_sub(self.head_max_len + 16).max(8)
    }

    /// Temperature by key, `type.bucket` then `type`, else 1.
    pub fn temperature(&self, type_name: &str, bucket: &str) -> f32 {
        let full = format!("{type_name}.{bucket}");
        for key in [full.as_str(), type_name] {
            if let Some((_, t)) = self.temperatures.iter().find(|(k, _)| k == key) {
                return *t;
            }
        }
        1.0
    }

    /// Generic `noul` definitions of a language (English when unknown).
    pub fn noul_definitions(&self, lang: &str) -> (&str, &str) {
        let pick = |l: &str| {
            self.noul
                .iter()
                .find(|(x, _, _)| x == l)
                .map(|(_, f, t)| (f.as_str(), t.as_str()))
        };
        pick(lang)
            .or_else(|| pick("en"))
            .unwrap_or(("The statement is false.", "The statement is true."))
    }

    pub fn validate(&self) -> Result<()> {
        let bad = |m: &str| Err(Error::Format(format!("invalid iso config: {m}")));
        let d = self.hidden_size;
        if d == 0 || d > 1 << 16 || self.vocab_size == 0 || self.vocab_size > 1 << 24 {
            return bad("hidden or vocab size");
        }
        if self.num_heads == 0
            || !d.is_multiple_of(self.num_heads)
            || !(d / self.num_heads).is_multiple_of(2)
        {
            return bad("heads");
        }
        if self.intermediate_size == 0 || self.intermediate_size > 1 << 16 {
            return bad("feed-forward size");
        }
        if self.max_len == 0 || self.max_len > 1 << 16 || self.head_max_len + 32 > self.max_len {
            return bad("max_len / max_head_tokens");
        }
        Ok(())
    }

    pub fn from_gguf(g: &Gguf<'_>) -> Result<Self> {
        let key = |k: &str| format!("{ARCH}.{k}");
        let int = |k: &str| g.get(&key(k)).and_then(|v| v.as_u64()).map(|v| v as usize);
        let req = |k: &str| int(k).ok_or_else(|| Error::Missing(format!("GGUF key {}", key(k))));
        let f = |k: &str| g.get_f64(&key(k)).map(|v| v as f32);
        let mut temperatures = Vec::new();
        let tprefix = key("temperature.");
        for (k, v) in &g.kvs {
            if let Some(name) = k.strip_prefix(tprefix.as_str()) {
                if let Some(t) = v.as_f64() {
                    temperatures.push((name.to_string(), t as f32));
                }
            }
        }
        let mut noul = Vec::new();
        for lang in ["en", "fr"] {
            if let (Some(a), Some(b)) = (
                g.get_str(&key(&format!("noul.false.{lang}"))),
                g.get_str(&key(&format!("noul.true.{lang}"))),
            ) {
                noul.push((lang.to_string(), a.to_string(), b.to_string()));
            }
        }
        let positions = match g.get_str(&key("positions")).unwrap_or("shared") {
            "shared" => Positions::Shared,
            "reset" => Positions::Reset,
            p => return Err(Error::Unsupported(format!("iso positions {p}"))),
        };
        let cfg = IsoConfig {
            name: g.get_str("general.name").unwrap_or(ARCH).to_string(),
            vocab_size: req("vocab_size")?,
            hidden_size: req("embedding_length")?,
            intermediate_size: req("feed_forward_length")?,
            num_layers: req("block_count")?,
            num_heads: req("attention.head_count")?,
            norm_eps: f("attention.layer_norm_epsilon").unwrap_or(1e-5),
            global_rope_theta: f("rope.freq_base").unwrap_or(160_000.0),
            local_rope_theta: f("rope.freq_base_swa")
                .or_else(|| f("rope.freq_base"))
                .unwrap_or(160_000.0),
            local_attention: int("attention.sliding_window").unwrap_or(0),
            global_attn_every_n_layers: int("attention.sliding_window_pattern").unwrap_or(3),
            max_len: req("context_length")?,
            head_max_len: req("max_head_tokens")?,
            max_option_tokens: int("max_option_tokens").unwrap_or(prompt::MAX_OPTION_TOKENS),
            positions,
            temperatures,
            noul,
        };
        cfg.validate()?;
        Ok(cfg)
    }
}

/// A loaded iso model.
#[derive(Clone, Debug)]
pub struct IsoModel {
    pub cfg: IsoConfig,
    pub tok_emb: Embedding,
    pub emb_norm: Norm,
    pub layers: Vec<EncoderLayer>,
    pub final_norm: Norm,
    /// `[d, 4d]`, bias `[d]`.
    pub score0: Linear,
    /// `[1, d]`, bias `[1]`.
    pub score1: Linear,
    pub act_quant: ActQuant,
    rope_global: Rope,
    rope_local: Rope,
}

/// The keys and values of a state at every layer (after RoPE), `[n][d]` each.
#[derive(Clone, Debug, PartialEq)]
pub struct StateKv {
    pub ids: Vec<u32>,
    pub k: Vec<Vec<f32>>,
    pub v: Vec<Vec<f32>>,
}

impl StateKv {
    pub fn len(&self) -> usize {
        self.ids.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }
}

/// Keys of one attention segment: rows at `k[j * stride..]` / `v[j * stride..]`, logical
/// position `pos[j]`.
#[derive(Clone, Copy)]
struct Seg<'a> {
    k: &'a [f32],
    v: &'a [f32],
    stride: usize,
    pos: &'a [usize],
}

/// Bidirectional attention of the query rows `q[i * q_stride..]` at logical positions `qpos`
/// over the keys of `segs`, restricted to `|qpos - kpos| <= window` when given. The keys are
/// visited in segment order, so two queries over identical segments get identical outputs.
#[allow(clippy::too_many_arguments)]
fn attend_window(
    q: &[f32],
    q_stride: usize,
    qpos: &[usize],
    segs: &[Seg<'_>],
    n_heads: usize,
    head_dim: usize,
    window: Option<usize>,
    out: &mut [f32],
    out_stride: usize,
    scores: &mut Vec<f32>,
) {
    let hd = head_dim;
    let scale = 1.0 / (hd as f32).sqrt();
    let n_kv: usize = segs.iter().map(|s| s.pos.len()).sum();
    if scores.len() < n_kv {
        scores.resize(n_kv, 0.0);
    }
    let inside = |a: usize, b: usize| window.is_none_or(|w| a.abs_diff(b) <= w);
    for (i, &pq) in qpos.iter().enumerate() {
        for h in 0..n_heads {
            let qi = &q[i * q_stride + h * hd..i * q_stride + (h + 1) * hd];
            let mut m = f32::NEG_INFINITY;
            let mut j = 0;
            for s in segs {
                for (r, &pk) in s.pos.iter().enumerate() {
                    if inside(pq, pk) {
                        let v = kernels::dot(
                            qi,
                            &s.k[r * s.stride + h * hd..r * s.stride + (h + 1) * hd],
                        ) * scale;
                        scores[j] = v;
                        m = m.max(v);
                    }
                    j += 1;
                }
            }
            let o = &mut out[i * out_stride + h * hd..i * out_stride + (h + 1) * hd];
            o.fill(0.0);
            let mut l = 0.0;
            let mut j = 0;
            for s in segs {
                for (r, &pk) in s.pos.iter().enumerate() {
                    if inside(pq, pk) {
                        let e = kernels::fast_exp(scores[j] - m);
                        l += e;
                        kernels::axpy(
                            e,
                            &s.v[r * s.stride + h * hd..r * s.stride + (h + 1) * hd],
                            o,
                        );
                    }
                    j += 1;
                }
            }
            let inv = 1.0 / l;
            o.iter_mut().for_each(|v| *v *= inv);
        }
    }
}

/// A tensor read from the file.
struct Loaded {
    shape: Vec<usize>,
    w: Weight,
}

impl IsoModel {
    pub fn from_gguf(g: &Gguf<'_>) -> Result<Self> {
        let cfg = IsoConfig::from_gguf(g)?;
        let get = |name: &str| -> Result<Loaded> {
            let info = g
                .tensor(name)
                .ok_or_else(|| Error::Missing(format!("tensor {name}")))?;
            let w = Weight::from_ggml(info.dtype, g.tensor_bytes(info))?;
            Ok(Loaded {
                shape: info.shape(),
                w,
            })
        };
        let req = |name: &str, shape: &[usize]| -> Result<Loaded> {
            let l = get(name)?;
            if l.shape != shape {
                return Err(Error::Format(format!(
                    "tensor {name}: shape {:?}, expected {shape:?}",
                    l.shape
                )));
            }
            Ok(l)
        };
        let vec = |name: &str, n: usize| -> Result<Vec<f32>> { Ok(req(name, &[n])?.w.to_f32()) };
        let norm = |name: &str, n: usize| -> Result<Norm> {
            Ok(Norm {
                w: vec(name, n)?,
                b: None,
            })
        };
        let lin = |name: &str, bias: Option<&str>, n_out: usize, n_in: usize| -> Result<Linear> {
            Ok(Linear {
                w: req(name, &[n_out, n_in])?.w,
                b: bias.map(|b| vec(b, n_out)).transpose()?,
                n_in,
                n_out,
                outliers: None,
            })
        };
        let d = cfg.hidden_size;
        let i = cfg.intermediate_size;
        let tok_emb = Embedding {
            w: req("token_embd.weight", &[cfg.vocab_size, d])?.w,
            d,
        };
        let mut layers = Vec::with_capacity(cfg.num_layers);
        for l in 0..cfg.num_layers {
            let b = format!("blk.{l}");
            layers.push(EncoderLayer {
                attn_norm: if l == 0 {
                    None
                } else {
                    Some(norm(&format!("{b}.attn_norm.weight"), d)?)
                },
                wqkv: lin(&format!("{b}.attn_qkv.weight"), None, 3 * d, d)?,
                wo: lin(&format!("{b}.attn_output.weight"), None, d, d)?,
                mlp_norm: norm(&format!("{b}.ffn_norm.weight"), d)?,
                wi: lin(&format!("{b}.ffn_up.weight"), None, 2 * i, d)?,
                mlp_wo: lin(&format!("{b}.ffn_down.weight"), None, d, i)?,
            });
        }
        let hd = cfg.head_dim();
        let max_pos = cfg.max_len.max(1);
        Ok(IsoModel {
            emb_norm: norm("token_embd_norm.weight", d)?,
            final_norm: norm("output_norm.weight", d)?,
            score0: lin("iso.score.0.weight", Some("iso.score.0.bias"), d, 4 * d)?,
            score1: lin("iso.score.1.weight", Some("iso.score.1.bias"), 1, d)?,
            rope_global: Rope::new(hd, cfg.global_rope_theta, max_pos),
            rope_local: Rope::new(hd, cfg.local_rope_theta, max_pos),
            tok_emb,
            layers,
            cfg,
            act_quant: ActQuant::default(),
        })
    }

    /// Every matrix in f32 (reference path).
    pub fn dequantize(&mut self) {
        for l in &mut self.layers {
            for lin in [&mut l.wqkv, &mut l.wo, &mut l.wi, &mut l.mlp_wo] {
                lin.dequantize();
            }
        }
        if self.tok_emb.w.is_quantized() {
            self.tok_emb.w = Weight::F32(self.tok_emb.w.to_f32());
        }
    }

    /// True if any matrix is stored as Q8_0 or Q4_0.
    pub fn has_q8(&self) -> bool {
        self.layers.iter().any(|l| l.wqkv.w.is_quantized()) || self.tok_emb.w.is_quantized()
    }

    /// `q4_0`, `q8_0` or `f32`.
    pub fn weight_format(&self) -> &'static str {
        let mats = self
            .layers
            .iter()
            .flat_map(|l| [&l.wqkv.w, &l.wo.w, &l.wi.w, &l.mlp_wo.w]);
        crate::model::weight_format(mats.chain([&self.tok_emb.w]))
    }

    pub fn n_params(&self) -> usize {
        let lin = |l: &Linear| l.n_in * l.n_out + l.b.as_ref().map_or(0, Vec::len);
        let norm = |n: &Norm| n.w.len();
        let mut n = self.cfg.vocab_size * self.cfg.hidden_size
            + norm(&self.emb_norm)
            + norm(&self.final_norm)
            + lin(&self.score0)
            + lin(&self.score1);
        for l in &self.layers {
            n += l.attn_norm.as_ref().map_or(0, norm)
                + lin(&l.wqkv)
                + lin(&l.wo)
                + norm(&l.mlp_norm)
                + lin(&l.wi)
                + lin(&l.mlp_wo);
        }
        n
    }

    fn rope(&self, l: usize) -> &Rope {
        if self.cfg.is_global(l) {
            &self.rope_global
        } else {
            &self.rope_local
        }
    }

    fn window(&self, l: usize) -> Option<usize> {
        (!self.cfg.is_global(l)).then_some(self.cfg.local_attention / 2)
    }

    fn embed(&self, ids: &[u32], x: &mut [f32]) -> Result<()> {
        let d = self.cfg.hidden_size;
        for (r, &id) in ids.iter().enumerate() {
            if id as usize >= self.cfg.vocab_size {
                return Err(Error::InvalidRequest(format!(
                    "token id {id} out of vocabulary"
                )));
            }
            self.tok_emb
                .row_into(id as usize, &mut x[r * d..(r + 1) * d]);
        }
        kernels::layer_norm_inplace(x, d, &self.emb_norm.w, None, self.cfg.norm_eps);
        Ok(())
    }

    /// Feed-forward half of a layer (`x += Wo · geglu(Wi · norm(x))`) over `rows` rows.
    fn mlp(&self, layer: &EncoderLayer, x: &mut [f32], rows: usize, xq: &mut [ActBlock]) {
        let (d, i) = (self.cfg.hidden_size, self.cfg.intermediate_size);
        let mut hn = vec![0.0; rows * d];
        kernels::layer_norm(x, d, &layer.mlp_norm.w, None, self.cfg.norm_eps, &mut hn);
        let mut up = vec![0.0; rows * 2 * i];
        let act = &mut Act {
            xq,
            mode: self.act_quant,
        };
        linear(layer.wi.view(), &hn, rows, &mut up, false, act);
        let mut g = vec![0.0; rows * i];
        for r in 0..rows {
            let u = &up[r * 2 * i..(r + 1) * 2 * i];
            for j in 0..i {
                g[r * i + j] = kernels::gelu(u[j]) * u[i + j];
            }
        }
        linear(layer.mlp_wo.view(), &g, rows, x, true, act);
    }

    /// Fused QKV of `rows` rows (attention norm, projection, RoPE at `pos`): `[rows][3d]`.
    fn qkv(&self, l: usize, x: &[f32], pos: &[usize], xq: &mut [ActBlock]) -> Vec<f32> {
        let d = self.cfg.hidden_size;
        let rows = pos.len();
        let layer = &self.layers[l];
        let mut qkv = vec![0.0; rows * 3 * d];
        let act = &mut Act {
            xq,
            mode: self.act_quant,
        };
        match &layer.attn_norm {
            Some(n) => {
                let mut hn = vec![0.0; rows * d];
                kernels::layer_norm(x, d, &n.w, None, self.cfg.norm_eps, &mut hn);
                linear(layer.wqkv.view(), &hn, rows, &mut qkv, false, act);
            }
            None => linear(layer.wqkv.view(), x, rows, &mut qkv, false, act),
        }
        let (nh, hd) = (self.cfg.num_heads, self.cfg.head_dim());
        let rope = self.rope(l);
        for (r, &p) in pos.iter().enumerate() {
            let row = &mut qkv[r * 3 * d..r * 3 * d + 2 * d];
            for h in 0..nh {
                rope.apply(&mut row[h * hd..(h + 1) * hd], p);
                rope.apply(&mut row[d + h * hd..d + (h + 1) * hd], p);
            }
        }
        qkv
    }

    fn xq(&self, rows: usize) -> Vec<ActBlock> {
        let k = (2 * self.cfg.intermediate_size)
            .max(4 * self.cfg.hidden_size)
            .max(self.cfg.intermediate_size);
        vec![ActBlock::default(); rows.max(1) * k.div_ceil(QK)]
    }

    /// Encodes a state segment (`[CLS] state [SEP]` ids) and keeps its keys and values.
    pub fn encode_state(&self, ids: &[u32]) -> Result<StateKv> {
        let d = self.cfg.hidden_size;
        let n = ids.len();
        if n == 0 || n > self.cfg.max_len {
            return Err(Error::InvalidRequest(format!("state of {n} tokens")));
        }
        let mut x = vec![0.0; n * d];
        self.embed(ids, &mut x)?;
        let pos: Vec<usize> = (0..n).collect();
        let mut xq = self.xq(n);
        let mut sc = AttnScratch::new(n, self.cfg.head_dim());
        let (mut ks, mut vs) = (Vec::new(), Vec::new());
        for l in 0..self.layers.len() {
            let qkv = self.qkv(l, &x, &pos, &mut xq);
            let a = Attention {
                k: &qkv[d..],
                v: &qkv[2 * d..],
                kv_stride: 3 * d,
                n_kv: n,
                n_heads: self.cfg.num_heads,
                head_dim: self.cfg.head_dim(),
                window: self.window(l),
            };
            let mut att = vec![0.0; n * d];
            a.run(&qkv, 3 * d, &pos, &mut att, d, &mut sc);
            let (mut k, mut v) = (vec![0.0; n * d], vec![0.0; n * d]);
            for r in 0..n {
                k[r * d..(r + 1) * d].copy_from_slice(&qkv[r * 3 * d + d..r * 3 * d + 2 * d]);
                v[r * d..(r + 1) * d].copy_from_slice(&qkv[r * 3 * d + 2 * d..(r + 1) * 3 * d]);
            }
            ks.push(k);
            vs.push(v);
            let act = &mut Act {
                xq: &mut xq,
                mode: self.act_quant,
            };
            linear(self.layers[l].wo.view(), &att, n, &mut x, true, act);
            self.mlp(&self.layers[l], &mut x, n, &mut xq);
        }
        Ok(StateKv {
            ids: ids.to_vec(),
            k: ks,
            v: vs,
        })
    }

    /// Raw option scores of a question (`head` ids ending with `[SEP]`) and its options (ids
    /// of each definition) over an encoded state.
    pub fn scores(&self, state: &StateKv, head: &[u32], options: &[Vec<u32>]) -> Result<Vec<f32>> {
        let d = self.cfg.hidden_size;
        let (nh, hd) = (self.cfg.num_heads, self.cfg.head_dim());
        let ns = state.len();
        if options.is_empty() || options.iter().any(Vec::is_empty) || head.is_empty() {
            return Err(Error::InvalidRequest("empty question or option".into()));
        }
        let nq = head.len();
        let rows = nq + options.iter().map(Vec::len).sum::<usize>();
        if ns + rows > self.cfg.max_len {
            return Err(Error::InvalidRequest(format!(
                "{} tokens (max {})",
                ns + rows,
                self.cfg.max_len
            )));
        }
        let start = match self.cfg.positions {
            Positions::Shared => ns + nq,
            Positions::Reset => 0,
        };
        let mut ids = head.to_vec();
        let mut pos: Vec<usize> = (ns..ns + nq).collect();
        let mut spans = Vec::with_capacity(options.len());
        for o in options {
            spans.push((ids.len(), ids.len() + o.len()));
            ids.extend_from_slice(o);
            pos.extend((0..o.len()).map(|t| start + t));
        }
        let mut x = vec![0.0; rows * d];
        self.embed(&ids, &mut x)?;
        let state_pos: Vec<usize> = (0..ns).collect();
        let mut xq = self.xq(rows);
        let mut scores = Vec::new();
        for l in 0..self.layers.len() {
            let qkv = self.qkv(l, &x, &pos, &mut xq);
            let w = self.window(l);
            let st = Seg {
                k: &state.k[l],
                v: &state.v[l],
                stride: d,
                pos: &state_pos,
            };
            let seg = |a: usize, b: usize| Seg {
                k: &qkv[a * 3 * d + d..],
                v: &qkv[a * 3 * d + 2 * d..],
                stride: 3 * d,
                pos: &pos[a..b],
            };
            let question = seg(0, nq);
            let mut att = vec![0.0; rows * d];
            attend_window(
                &qkv,
                3 * d,
                &pos[..nq],
                &[st, question],
                nh,
                hd,
                w,
                &mut att,
                d,
                &mut scores,
            );
            for &(a, b) in &spans {
                attend_window(
                    &qkv[a * 3 * d..],
                    3 * d,
                    &pos[a..b],
                    &[st, question, seg(a, b)],
                    nh,
                    hd,
                    w,
                    &mut att[a * d..],
                    d,
                    &mut scores,
                );
            }
            let act = &mut Act {
                xq: &mut xq,
                mode: self.act_quant,
            };
            linear(self.layers[l].wo.view(), &att, rows, &mut x, true, act);
            self.mlp(&self.layers[l], &mut x, rows, &mut xq);
        }
        kernels::layer_norm_inplace(&mut x, d, &self.final_norm.w, None, self.cfg.norm_eps);
        let mean = |a: usize, b: usize| {
            let mut m = vec![0.0f32; d];
            for r in a..b {
                for (o, v) in m.iter_mut().zip(&x[r * d..(r + 1) * d]) {
                    *o += v;
                }
            }
            let inv = 1.0 / (b - a) as f32;
            m.iter_mut().for_each(|v| *v *= inv);
            m
        };
        let q = mean(0, nq);
        let k = options.len();
        let mut z = vec![0.0f32; k * 4 * d];
        for (j, &(a, b)) in spans.iter().enumerate() {
            let h = mean(a, b);
            let zj = &mut z[j * 4 * d..(j + 1) * 4 * d];
            for c in 0..d {
                zj[c] = h[c];
                zj[d + c] = q[c];
                zj[2 * d + c] = h[c] * q[c];
                zj[3 * d + c] = (h[c] - q[c]).abs();
            }
        }
        let act = &mut Act {
            xq: &mut xq,
            mode: self.act_quant,
        };
        let mut s0 = vec![0.0; k * d];
        linear(self.score0.view(), &z, k, &mut s0, false, act);
        s0.iter_mut().for_each(|v| *v = kernels::gelu(*v));
        let mut out = vec![0.0; k];
        linear(self.score1.view(), &s0, k, &mut out, false, act);
        Ok(out)
    }
}

/// The definition-only text of a rendered s1 option (`key: definition`, `level i: …`,
/// `false: …` / `true: …`), as the s5 records split them (sky-train `s5/record.rs`):
/// `score` keeps its level as `level i of K: …`, an empty `noul` definition takes the generic
/// one. A `choice` option is split at its first `": "` when the part before it looks like a key
/// (at most 40 bytes, no line break); otherwise the whole text is the definition.
pub fn definition_of(qtype: QType, i: usize, k: usize, opt: &str, noul: (&str, &str)) -> String {
    match qtype {
        QType::Score => {
            let rest = opt.strip_prefix(&format!("level {i}:")).unwrap_or(opt);
            format!("level {i} of {}: {}", k.saturating_sub(1), rest.trim())
        }
        QType::Noul => {
            let id = if i == 0 { "false" } else { "true" };
            let rest = opt.strip_prefix(&format!("{id}:")).unwrap_or(opt).trim();
            if rest.is_empty() || rest == id {
                (if i == 0 { noul.0 } else { noul.1 }).to_string()
            } else {
                rest.to_string()
            }
        }
        QType::Choice => match opt.split_once(": ") {
            Some((key, d)) if !key.trim().is_empty() && key.len() <= 40 && !key.contains('\n') => {
                let d = d.trim();
                if d.is_empty() {
                    key.trim().to_string()
                } else {
                    d.to_string()
                }
            }
            _ => opt.trim().to_string(),
        },
    }
}

/// The definition-only texts of a typed question.
pub fn definitions(kind: &QuestionKind, noul: (&str, &str)) -> Vec<String> {
    fn nonempty(d: &Option<String>) -> Option<&str> {
        d.as_deref().map(str::trim).filter(|s| !s.is_empty())
    }
    match kind {
        QuestionKind::Choice(opts) => opts
            .iter()
            .map(|(k, d)| nonempty(d).unwrap_or(k).to_string())
            .collect(),
        QuestionKind::Score(levels) => levels
            .iter()
            .enumerate()
            .map(|(i, c)| format!("level {i} of {}: {}", levels.len() - 1, c.trim()))
            .collect(),
        QuestionKind::Noul {
            false_desc,
            true_desc,
        } => vec![
            nonempty(false_desc).unwrap_or(noul.0).to_string(),
            nonempty(true_desc).unwrap_or(noul.1).to_string(),
        ],
    }
}

/// Output of one question.
#[derive(Clone, Debug, PartialEq)]
pub struct IsoOutput {
    pub probs: Vec<f32>,
    pub logits: Vec<f32>,
    pub state_hit: bool,
    pub tokens: usize,
}

/// Engine of an iso model: tokenizer, state cache.
pub struct IsoEngine {
    model: IsoModel,
    tok: Tokenizer,
    bpe: BpeCache,
    cls: u32,
    sep: u32,
    mask_text: String,
    /// Most recent first.
    states: Vec<StateKv>,
    state_cap: usize,
    lang: String,
}

impl IsoEngine {
    pub fn new(
        model: IsoModel,
        tok: Tokenizer,
        state_cache: usize,
        bpe_cache: usize,
    ) -> Result<Self> {
        let sp = tok.special();
        let missing = |n: &str| Error::Missing(format!("{n} token in the tokenizer"));
        let cls = sp.cls.ok_or_else(|| missing("cls"))?;
        let sep = sp.sep.ok_or_else(|| missing("sep"))?;
        let mask_text = sp
            .mask
            .and_then(|m| tok.id_to_token(m))
            .unwrap_or("[MASK]")
            .to_string();
        Ok(IsoEngine {
            model,
            tok,
            bpe: BpeCache::new(bpe_cache),
            cls,
            sep,
            mask_text,
            states: Vec::new(),
            state_cap: state_cache,
            lang: "en".into(),
        })
    }

    pub fn model(&self) -> &IsoModel {
        &self.model
    }

    pub fn model_mut(&mut self) -> &mut IsoModel {
        &mut self.model
    }

    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tok
    }

    /// Language of the generic `noul` definitions (`en`, `fr`).
    pub fn set_lang(&mut self, lang: &str) {
        self.lang = lang.to_string();
    }

    pub fn lang(&self) -> &str {
        &self.lang
    }

    pub fn clear_cache(&mut self) {
        self.states.clear();
    }

    pub fn temperature(&self, qtype: QType, n: usize) -> f32 {
        self.model
            .cfg
            .temperature(qtype.name(), prompt::temp_bucket(n))
    }

    fn encode(&mut self, text: &str) -> Vec<u32> {
        let clean = text.replace(&self.mask_text, " ");
        let mut out = Vec::new();
        self.tok.encode_into(&clean, &mut out, Some(&mut self.bpe));
        out
    }

    /// The state segment ids: `[CLS] state [SEP]` within the state budget.
    pub fn state_ids(&mut self, state: &str) -> Vec<u32> {
        let budget = self.model.cfg.state_budget();
        let mut ids = vec![self.cls];
        let mut s = self.encode(state);
        s.truncate(budget - 2);
        ids.extend(s);
        ids.push(self.sep);
        ids
    }

    /// The question ids (ending with `[SEP]`) and the option ids, under the Laya budget.
    pub fn question_ids(
        &mut self,
        qtype: QType,
        instructions: &str,
        definitions: &[String],
    ) -> (Vec<u32>, Vec<Vec<u32>>) {
        let head = self.encode(&format!("{} question: {instructions}", qtype.name()));
        let max_opt = self.model.cfg.max_option_tokens;
        let mut opts: Vec<Vec<u32>> = definitions
            .iter()
            .map(|d| {
                let mut o = self.encode(&format!(" {d}"));
                o.truncate(max_opt);
                if o.is_empty() {
                    o.push(self.tok.special().unk.unwrap_or(self.sep));
                }
                o
            })
            .collect();
        let hml = self.model.cfg.head_max_len as i64;
        let total = |o: &[Vec<u32>]| o.iter().map(|v| v.len() as i64).sum::<i64>();
        let mut left = hml - total(&opts);
        if left < 16 {
            let per = (hml - 16).div_euclid(opts.len().max(1) as i64).max(4) as usize;
            for o in &mut opts {
                o.truncate(per);
            }
            left = hml - total(&opts);
        }
        let keep = head.len().min(left.max(8) as usize);
        let mut q = head[..keep].to_vec();
        q.push(self.sep);
        (q, opts)
    }

    /// The cached encoding of a state (moved to the front), encoding it on a miss.
    fn state(&mut self, ids: Vec<u32>) -> Result<bool> {
        if let Some(i) = self.states.iter().position(|s| s.ids == ids) {
            let s = self.states.remove(i);
            self.states.insert(0, s);
            return Ok(true);
        }
        let s = self.model.encode_state(&ids)?;
        if self.state_cap > 0 {
            if self.states.len() >= self.state_cap {
                self.states.pop();
            }
            self.states.insert(0, s);
        } else {
            self.states = vec![s];
        }
        Ok(false)
    }

    /// Probabilities and raw scores of a question given by definitions.
    pub fn question_defs(
        &mut self,
        qtype: QType,
        instructions: &str,
        definitions: &[String],
        state: &str,
    ) -> Result<IsoOutput> {
        if definitions.len() < 2 {
            return Err(Error::InvalidRequest("at least two options".into()));
        }
        let sids = self.state_ids(state);
        let n_state = sids.len();
        let hit = self.state(sids)?;
        let (head, opts) = self.question_ids(qtype, instructions, definitions);
        let tokens = n_state + head.len() + opts.iter().map(Vec::len).sum::<usize>();
        let logits = self.model.scores(&self.states[0], &head, &opts)?;
        if self.state_cap == 0 {
            self.states.clear();
        }
        let probs = prompt::probs_from_logits(&logits, self.temperature(qtype, logits.len()));
        Ok(IsoOutput {
            probs,
            logits,
            state_hit: hit,
            tokens,
        })
    }

    /// Probabilities of rendered s1 option texts (`key: definition`, ...): the names are
    /// dropped, only the definitions reach the model.
    pub fn question<S: AsRef<str>>(
        &mut self,
        qtype: QType,
        instructions: &str,
        options: &[S],
        state: &str,
    ) -> Result<IsoOutput> {
        let k = options.len();
        let noul = self.model.cfg.noul_definitions(&self.lang);
        let noul = (noul.0.to_string(), noul.1.to_string());
        let defs: Vec<String> = options
            .iter()
            .enumerate()
            .map(|(i, o)| definition_of(qtype, i, k, o.as_ref(), (&noul.0, &noul.1)))
            .collect();
        self.question_defs(qtype, instructions, &defs, state)
    }

    /// The flat sequence of a question (state, question, options) and the option starts.
    pub fn build_sequence(
        &mut self,
        qtype: QType,
        instructions: &str,
        options: &[&str],
        state: &str,
    ) -> (Vec<u32>, Vec<usize>) {
        let k = options.len();
        let noul = self.model.cfg.noul_definitions(&self.lang);
        let noul = (noul.0.to_string(), noul.1.to_string());
        let defs: Vec<String> = options
            .iter()
            .enumerate()
            .map(|(i, o)| definition_of(qtype, i, k, o, (&noul.0, &noul.1)))
            .collect();
        let mut ids = self.state_ids(state);
        let (head, opts) = self.question_ids(qtype, instructions, &defs);
        ids.extend(head);
        let mut starts = Vec::with_capacity(opts.len());
        for o in opts {
            starts.push(ids.len());
            ids.extend(o);
        }
        (ids, starts)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const NOUL: (&str, &str) = ("The statement is false.", "The statement is true.");

    #[test]
    fn rendered_options_map_to_the_training_definitions() {
        // The same strings as sky-train `S5Record::from_decision(..).render(Definition)`.
        let c = |i, k, o| definition_of(QType::Choice, i, k, o, NOUL);
        assert_eq!(c(0, 2, "flap: push the bird up"), "push the bird up");
        assert_eq!(c(1, 4, "down"), "down");
        assert_eq!(c(0, 2, "no_tool: answer: directly"), "answer: directly");
        assert_eq!(c(0, 2, "key:"), "key:");
        let long = format!("{}: x", "a".repeat(60));
        assert_eq!(c(0, 2, &long), long);
        assert_eq!(
            definition_of(QType::Score, 1, 3, "level 1: minor", NOUL),
            "level 1 of 2: minor"
        );
        assert_eq!(
            definition_of(QType::Noul, 0, 2, "false: benign", NOUL),
            "benign"
        );
        assert_eq!(definition_of(QType::Noul, 1, 2, "true", NOUL), NOUL.1);
        assert_eq!(definition_of(QType::Noul, 0, 2, "false:", NOUL), NOUL.0);
    }

    #[test]
    fn typed_questions_render_definitions_only() {
        let kind = QuestionKind::Choice(vec![
            ("yes".into(), Some("approve the refund".into())),
            ("no".into(), None),
        ]);
        assert_eq!(definitions(&kind, NOUL), ["approve the refund", "no"]);
        let s = QuestionKind::Score(vec!["low".into(), "high".into()]);
        assert_eq!(
            definitions(&s, NOUL),
            ["level 0 of 1: low", "level 1 of 1: high"]
        );
        let n = QuestionKind::Noul {
            false_desc: None,
            true_desc: Some(" it rains ".into()),
        };
        assert_eq!(definitions(&n, NOUL), [NOUL.0, "it rains"]);
    }
}
