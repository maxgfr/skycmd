//! Model configuration, weights and the forward pass (ModernBERT encoder + Laya decision head).

use sky_gguf::{GgmlType, Gguf};

use crate::json::Json;
use crate::kernels::{
    self, ActBlock, ActQuant, Attention, AttnScratch, BlockQ8, Outliers, Rope, QK,
};
use crate::quant::{self, Calibration};
use crate::safetensors::SafeTensors;
use crate::{Error, Result};

/// A calibration sequence: token ids, marker positions, question type.
pub type CalibSeq = (Vec<u32>, Vec<usize>, usize);

/// Number of question types (choice, score, noul).
pub const N_QTYPES: usize = 3;

/// Model hyper-parameters (ModernBERT keys + Laya decision head).
#[derive(Clone, Debug, PartialEq)]
pub struct ModelConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub intermediate_size: usize,
    pub norm_eps: f32,
    pub global_rope_theta: f32,
    pub local_rope_theta: f32,
    pub global_attn_every_n_layers: usize,
    /// Full local window; local layers attend `|i - j| <= local_attention / 2`. 0 = all global.
    pub local_attention: usize,
    pub head_layers: usize,
    pub head_heads: usize,
    pub head_ff: usize,
    pub head_norm_eps: f32,
    /// Longest sequence (Laya `max_len`).
    pub max_len: usize,
    /// Token budget of question + options (Laya `head_max_len`).
    pub head_max_len: usize,
    /// Temperatures keyed like llama.cpp: `"choice"`, `"choice.3_5"`, ...
    pub temperatures: Vec<(String, f32)>,
    pub name: String,
}

impl ModelConfig {
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_heads
    }

    /// Reject configurations the engine cannot run (or that would need absurd buffers).
    pub fn validate(&self) -> Result<()> {
        const MAX_DIM: usize = 1 << 16;
        let bad = |m: &str| Err(Error::Format(format!("invalid model config: {m}")));
        let d = self.hidden_size;
        if d == 0 || d > MAX_DIM || self.vocab_size == 0 || self.vocab_size > 1 << 24 {
            return bad("hidden or vocab size");
        }
        if self.num_heads == 0
            || !d.is_multiple_of(self.num_heads)
            || !(d / self.num_heads).is_multiple_of(2)
        {
            return bad("hidden size must split into heads of even size");
        }
        if self.head_heads == 0 || !d.is_multiple_of(self.head_heads) {
            return bad("hidden size must split into head attention heads");
        }
        if self.intermediate_size == 0
            || self.intermediate_size > MAX_DIM
            || self.head_ff == 0
            || self.head_ff > MAX_DIM
        {
            return bad("feed-forward size");
        }
        if self.max_len == 0 || self.max_len > MAX_DIM || self.head_max_len == 0 {
            return bad("max_len / head_max_len");
        }
        Ok(())
    }

    /// Whether encoder layer `l` uses global attention (and the global RoPE theta).
    pub fn is_global(&self, l: usize) -> bool {
        self.local_attention == 0
            || self.global_attn_every_n_layers <= 1
            || l.is_multiple_of(self.global_attn_every_n_layers)
    }

    /// Temperature by key, falling back from `type.bucket` to `type`, else 1.0.
    pub fn temperature(&self, type_name: &str, bucket: &str) -> f32 {
        let full = format!("{type_name}.{bucket}");
        for key in [full.as_str(), type_name] {
            if let Some((_, t)) = self.temperatures.iter().find(|(k, _)| k == key) {
                return *t;
            }
        }
        1.0
    }

    /// Build from the ModernBERT `encoder/config.json` and the Laya `rl_agent_config.json`
    /// (both may be the same object, as in `configs/s1-*.json`).
    pub fn from_hf_json(encoder: &Json, agent: &Json) -> Result<Self> {
        let num = |j: &Json, k: &str| j.get(k).and_then(|v| v.as_f64());
        let req = |k: &str| {
            num(encoder, k).ok_or_else(|| Error::Missing(format!("encoder config key {k}")))
        };
        if let Some(act) = encoder.get("hidden_activation").and_then(|v| v.as_str()) {
            if act != "gelu" && act != "gelu_python" {
                return Err(Error::Unsupported(format!("hidden_activation {act}")));
            }
        }
        for k in ["norm_bias", "attention_bias", "mlp_bias"] {
            if encoder.get(k).and_then(|v| v.as_bool()) == Some(true) {
                return Err(Error::Unsupported(format!("{k} = true")));
            }
        }
        let d = req("hidden_size")? as usize;
        let global = num(encoder, "global_rope_theta")
            .or_else(|| num(encoder, "rope_theta"))
            .unwrap_or(160_000.0);
        let mut temperatures = Vec::new();
        if let Some(t) = agent.get("temperature").and_then(|t| t.as_array()) {
            for (name, v) in ["choice", "score", "noul"].iter().zip(t) {
                if let Some(v) = v.as_f64() {
                    temperatures.push((name.to_string(), v as f32));
                }
            }
        }
        if let Some(obj) = agent
            .get("temperature_by_options")
            .and_then(|t| t.as_object())
        {
            for (k, v) in obj {
                if let Some(v) = v.as_f64() {
                    temperatures.push((laya_bucket_to_key(k), v as f32));
                }
            }
        }
        let head_layers = num(agent, "head_layers").unwrap_or(2.0) as usize;
        Ok(ModelConfig {
            vocab_size: req("vocab_size")? as usize,
            hidden_size: d,
            num_layers: req("num_hidden_layers")? as usize,
            num_heads: req("num_attention_heads")? as usize,
            intermediate_size: req("intermediate_size")? as usize,
            norm_eps: num(encoder, "norm_eps")
                .or_else(|| num(encoder, "layer_norm_eps"))
                .unwrap_or(1e-5) as f32,
            global_rope_theta: global as f32,
            local_rope_theta: num(encoder, "local_rope_theta").unwrap_or(10_000.0) as f32,
            global_attn_every_n_layers: num(encoder, "global_attn_every_n_layers").unwrap_or(3.0)
                as usize,
            local_attention: num(encoder, "local_attention").unwrap_or(128.0) as usize,
            head_layers,
            head_heads: (d / 64).max(1),
            head_ff: 4 * d,
            head_norm_eps: 1e-5,
            max_len: num(agent, "max_len").unwrap_or(512.0) as usize,
            head_max_len: num(agent, "head_max_len").unwrap_or(256.0) as usize,
            temperatures,
            name: agent
                .get("model_name")
                .and_then(|v| v.as_str())
                .or_else(|| agent.get("name").and_then(|v| v.as_str()))
                .unwrap_or("skycmd-s1")
                .to_string(),
        })
    }

    /// Read the configuration from GGUF metadata (llama.cpp `modern-bert` keys).
    pub fn from_gguf(g: &Gguf<'_>) -> Result<Self> {
        let arch = g.architecture().unwrap_or("modern-bert");
        let key = |k: &str| format!("{arch}.{k}");
        let int = |k: &str| -> Option<usize> {
            match g.get(&key(k))? {
                sky_gguf::Value::Array(a) => {
                    a.iter().next().and_then(|v| v.as_u64()).map(|v| v as usize)
                }
                v => v.as_u64().map(|v| v as usize),
            }
        };
        let req = |k: &str| int(k).ok_or_else(|| Error::Missing(format!("GGUF key {}", key(k))));
        if let Some(act) = g.get_str(&key("hidden_act")) {
            if act != "gelu" {
                return Err(Error::Unsupported(format!("hidden_act {act}")));
            }
        }
        let d = req("embedding_length")?;
        let n_total = req("block_count")?;
        let head_layers = int("decision.block_count").unwrap_or(0);
        if head_layers >= n_total {
            return Err(Error::Format("invalid number of decision blocks".into()));
        }
        let n_layers = n_total - head_layers;
        let ff: Vec<usize> = match g.get(&key("feed_forward_length")) {
            Some(sky_gguf::Value::Array(a)) => a
                .to_i64_vec()
                .unwrap_or_default()
                .into_iter()
                .map(|v| v as usize)
                .collect(),
            Some(v) => vec![v.as_u64().unwrap_or(0) as usize; n_total],
            None => return Err(Error::Missing(key("feed_forward_length"))),
        };
        let intermediate_size = *ff
            .first()
            .ok_or_else(|| Error::Missing(key("feed_forward_length")))?;
        let head_ff = ff.get(n_layers).copied().unwrap_or(4 * d);
        let global = g.get_f64(&key("rope.freq_base")).unwrap_or(160_000.0) as f32;
        let mut temperatures = Vec::new();
        let tprefix = key("decision.temperature.");
        for (k, v) in &g.kvs {
            if let Some(name) = k.strip_prefix(tprefix.as_str()) {
                if let Some(t) = v.as_f64() {
                    temperatures.push((name.to_string(), t as f32));
                }
            }
        }
        let vocab_size = int("vocab_size")
            .or_else(|| {
                g.tensor("token_embd.weight")
                    .and_then(|t| t.dims.get(1))
                    .map(|&v| v as usize)
            })
            .ok_or_else(|| Error::Missing("vocab size".into()))?;
        let n_heads = req("attention.head_count")?;
        let max_len = int("decision.max_tokens")
            .unwrap_or_else(|| int("context_length").unwrap_or(512).min(512));
        Ok(ModelConfig {
            vocab_size,
            hidden_size: d,
            num_layers: n_layers,
            num_heads: n_heads,
            intermediate_size,
            norm_eps: g
                .get_f64(&key("attention.layer_norm_epsilon"))
                .unwrap_or(1e-5) as f32,
            global_rope_theta: global,
            local_rope_theta: g
                .get_f64(&key("rope.freq_base_swa"))
                .map(|v| v as f32)
                .unwrap_or(global),
            global_attn_every_n_layers: int("attention.sliding_window_pattern").unwrap_or(3),
            local_attention: int("attention.sliding_window").unwrap_or(0),
            head_layers,
            head_heads: n_heads,
            head_ff,
            head_norm_eps: g
                .get_f64(&key("attention.layer_norm_epsilon"))
                .unwrap_or(1e-5) as f32,
            max_len,
            head_max_len: int("decision.max_head_tokens").unwrap_or(256),
            temperatures,
            name: g.get_str("general.name").unwrap_or("skycmd-s1").to_string(),
        })
    }
}

/// `"choice:3-5"` (Laya) -> `"choice.3_5"` (llama.cpp), `"choice:11+"` -> `"choice.11"`.
pub fn laya_bucket_to_key(k: &str) -> String {
    k.replace(':', ".")
        .replace('-', "_")
        .trim_end_matches('+')
        .to_string()
}

/// Weight storage of a matrix `[rows][cols]`.
#[derive(Clone, Debug)]
pub enum Weight {
    F32(Vec<f32>),
    /// Q8_0 blocks, `cols / 32` per row.
    Q8(Vec<BlockQ8>),
}

impl Weight {
    fn n_elements(&self) -> usize {
        match self {
            Weight::F32(v) => v.len(),
            Weight::Q8(b) => b.len() * QK,
        }
    }

    pub fn is_q8(&self) -> bool {
        matches!(self, Weight::Q8(_))
    }

    /// All values as f32.
    pub fn to_f32(&self) -> Vec<f32> {
        match self {
            Weight::F32(v) => v.clone(),
            Weight::Q8(b) => {
                let mut out = vec![0.0; b.len() * QK];
                kernels::dequantize_q8(b, &mut out);
                out
            }
        }
    }
}

/// A linear layer `y = x W^T + b` with `W: [n_out][n_in]`.
#[derive(Clone, Debug)]
pub struct Linear {
    pub w: Weight,
    pub b: Option<Vec<f32>>,
    pub n_in: usize,
    pub n_out: usize,
    /// Input channels computed in f32 when `w` is Q8_0 (see [`crate::quant`]).
    pub outliers: Option<Outliers>,
}

#[derive(Clone, Copy)]
pub(crate) enum WeightRef<'a> {
    F32(&'a [f32]),
    Q8(&'a [BlockQ8]),
}

/// A view over a range of output rows of a [`Linear`].
#[derive(Clone, Copy)]
pub(crate) struct LinearView<'a> {
    w: WeightRef<'a>,
    b: Option<&'a [f32]>,
    n_in: usize,
    n_out: usize,
    /// Outlier columns of the full matrix, first row of the view, rows of the full matrix.
    outliers: Option<(&'a Outliers, usize, usize)>,
}

impl Linear {
    pub(crate) fn view(&self) -> LinearView<'_> {
        self.rows(0, self.n_out)
    }

    pub(crate) fn rows(&self, start: usize, end: usize) -> LinearView<'_> {
        let w = match &self.w {
            Weight::F32(v) => WeightRef::F32(&v[start * self.n_in..end * self.n_in]),
            Weight::Q8(b) => {
                let nb = self.n_in / QK;
                WeightRef::Q8(&b[start * nb..end * nb])
            }
        };
        let outliers = self.outliers.as_ref().map(|o| (o, start, self.n_out));
        LinearView {
            w,
            b: self.b.as_deref().map(|b| &b[start..end]),
            n_in: self.n_in,
            n_out: end - start,
            outliers,
        }
    }

    /// Convert the weight to Q8_0 (no-op if already quantized or `n_in` is not a multiple of 32).
    pub fn quantize(&mut self) {
        if let Weight::F32(v) = &self.w {
            if self.n_in.is_multiple_of(QK) {
                let mut blocks = vec![BlockQ8::default(); v.len() / QK];
                kernels::quantize_q8(v, &mut blocks);
                self.w = Weight::Q8(blocks);
            }
        }
    }

    /// Convert the weight to f32 (outlier columns are put back).
    pub fn dequantize(&mut self) {
        if self.w.is_q8() {
            let mut w = self.w.to_f32();
            if let Some(o) = self.outliers.take() {
                for (j, &c) in o.idx.iter().enumerate() {
                    for r in 0..self.n_out {
                        w[r * self.n_in + c] = o.cols[j * self.n_out + r];
                    }
                }
            }
            self.w = Weight::F32(w);
        }
    }

    /// Quantize to Q8_0 with the input channels `idx` computed in f32: exact columns when the
    /// weight is f32; when it is already Q8_0 the stored columns move to f32 unchanged.
    pub fn quantize_with_outliers(&mut self, idx: &[usize]) {
        if !self.n_in.is_multiple_of(QK) || self.outliers.is_some() {
            return;
        }
        let (n, k) = (self.n_out, self.n_in);
        match &mut self.w {
            Weight::F32(w) => {
                let mut w = std::mem::take(w);
                self.outliers = (!idx.is_empty()).then(|| Outliers::split(&mut w, n, k, idx));
                let mut blocks = vec![BlockQ8::default(); w.len() / QK];
                kernels::quantize_q8(&w, &mut blocks);
                self.w = Weight::Q8(blocks);
            }
            Weight::Q8(b) => {
                self.outliers = (!idx.is_empty()).then(|| Outliers::split_q8(b, n, k, idx));
            }
        }
    }
}

/// LayerNorm parameters.
#[derive(Clone, Debug)]
pub struct Norm {
    pub w: Vec<f32>,
    pub b: Option<Vec<f32>>,
}

#[derive(Clone, Debug)]
pub struct EncoderLayer {
    /// `None` for layer 0 (identity).
    pub attn_norm: Option<Norm>,
    pub wqkv: Linear,
    pub wo: Linear,
    pub mlp_norm: Norm,
    pub wi: Linear,
    pub mlp_wo: Linear,
}

#[derive(Clone, Debug)]
pub struct HeadLayer {
    pub norm1: Norm,
    pub in_proj: Linear,
    pub out_proj: Linear,
    pub norm2: Norm,
    pub linear1: Linear,
    pub linear2: Linear,
}

/// Token embedding table `[vocab][d]`.
#[derive(Clone, Debug)]
pub struct Embedding {
    pub w: Weight,
    pub d: usize,
}

impl Embedding {
    pub(crate) fn row_into(&self, id: usize, out: &mut [f32]) {
        match &self.w {
            Weight::F32(v) => out.copy_from_slice(&v[id * self.d..(id + 1) * self.d]),
            Weight::Q8(b) => {
                let nb = self.d / QK;
                kernels::dequantize_q8(&b[id * nb..(id + 1) * nb], out);
            }
        }
    }
}

/// A loaded model.
#[derive(Clone, Debug)]
pub struct Model {
    pub cfg: ModelConfig,
    pub tok_emb: Embedding,
    pub emb_norm: Norm,
    pub layers: Vec<EncoderLayer>,
    pub final_norm: Norm,
    pub head: Vec<HeadLayer>,
    /// `[3][d]`
    pub type_emb: Vec<f32>,
    pub scorer_norm: Norm,
    pub scorer1: Linear,
    pub scorer3: Linear,
    /// Activation quantization of the Q8_0 products.
    pub act_quant: ActQuant,
    rope_global: Rope,
    rope_local: Rope,
}

/// Which parts of the model to keep / convert to Q8_0.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct QuantizeSpec {
    pub encoder: bool,
    pub head: bool,
    pub embeddings: bool,
}

/// A tensor read from a file: row-major shape and data.
struct Loaded {
    shape: Vec<usize>,
    w: Weight,
}

/// Tensor naming scheme of the source file.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Scheme {
    /// HF / Laya names (`encoder.layers.0.attn.Wqkv.weight`, `head.layers.0.linear1.bias`, ...).
    Hf,
    /// llama.cpp names (`blk.0.attn_qkv.weight`, `cls.output.bias`, ...).
    Gguf,
}

/// Logical tensors of the model.
#[derive(Clone, Copy)]
enum T {
    TokEmb,
    EmbNorm,
    FinalNorm,
    AttnNorm(usize),
    Wqkv(usize),
    Wo(usize),
    MlpNorm(usize),
    Wi(usize),
    MlpWo(usize),
    /// head layer, part, is_bias
    Head(usize, HeadPart, bool),
    TypeEmb,
    /// scorer index (0, 1, 3), is_bias
    Scorer(usize, bool),
}

#[derive(Clone, Copy)]
enum HeadPart {
    Norm1,
    InProj,
    OutProj,
    Norm2,
    Linear1,
    Linear2,
}

fn tensor_name(scheme: Scheme, t: T, n_enc: usize) -> String {
    let wb = |bias: bool| if bias { "bias" } else { "weight" };
    match scheme {
        Scheme::Hf => match t {
            T::TokEmb => "encoder.embeddings.tok_embeddings.weight".into(),
            T::EmbNorm => "encoder.embeddings.norm.weight".into(),
            T::FinalNorm => "encoder.final_norm.weight".into(),
            T::AttnNorm(l) => format!("encoder.layers.{l}.attn_norm.weight"),
            T::Wqkv(l) => format!("encoder.layers.{l}.attn.Wqkv.weight"),
            T::Wo(l) => format!("encoder.layers.{l}.attn.Wo.weight"),
            T::MlpNorm(l) => format!("encoder.layers.{l}.mlp_norm.weight"),
            T::Wi(l) => format!("encoder.layers.{l}.mlp.Wi.weight"),
            T::MlpWo(l) => format!("encoder.layers.{l}.mlp.Wo.weight"),
            T::Head(i, p, b) => match p {
                HeadPart::Norm1 => format!("head.layers.{i}.norm1.{}", wb(b)),
                HeadPart::InProj => format!("head.layers.{i}.self_attn.in_proj_{}", wb(b)),
                HeadPart::OutProj => format!("head.layers.{i}.self_attn.out_proj.{}", wb(b)),
                HeadPart::Norm2 => format!("head.layers.{i}.norm2.{}", wb(b)),
                HeadPart::Linear1 => format!("head.layers.{i}.linear1.{}", wb(b)),
                HeadPart::Linear2 => format!("head.layers.{i}.linear2.{}", wb(b)),
            },
            T::TypeEmb => "type_emb.weight".into(),
            T::Scorer(k, b) => format!("scorer.{k}.{}", wb(b)),
        },
        Scheme::Gguf => match t {
            T::TokEmb => "token_embd.weight".into(),
            T::EmbNorm => "token_embd_norm.weight".into(),
            T::FinalNorm => "output_norm.weight".into(),
            T::AttnNorm(l) => format!("blk.{l}.attn_norm.weight"),
            T::Wqkv(l) => format!("blk.{l}.attn_qkv.weight"),
            T::Wo(l) => format!("blk.{l}.attn_output.weight"),
            T::MlpNorm(l) => format!("blk.{l}.ffn_norm.weight"),
            T::Wi(l) => format!("blk.{l}.ffn_up.weight"),
            T::MlpWo(l) => format!("blk.{l}.ffn_down.weight"),
            T::Head(i, p, b) => {
                let blk = n_enc + i;
                let part = match p {
                    HeadPart::Norm1 => "attn_norm",
                    HeadPart::InProj => "attn_qkv",
                    HeadPart::OutProj => "attn_output",
                    HeadPart::Norm2 => "ffn_norm",
                    HeadPart::Linear1 => "ffn_up",
                    HeadPart::Linear2 => "ffn_down",
                };
                format!("blk.{blk}.{part}.{}", wb(b))
            }
            T::TypeEmb => "token_types.weight".into(),
            T::Scorer(k, b) => {
                let base = match k {
                    0 => "cls.norm",
                    1 => "cls",
                    _ => "cls.output",
                };
                format!("{base}.{}", wb(b))
            }
        },
    }
}

/// All (logical tensor, HF name, GGUF name) triples of a model with this config.
pub fn tensor_name_pairs(cfg: &ModelConfig) -> Vec<(String, String)> {
    all_tensors(cfg)
        .into_iter()
        .map(|t| {
            (
                tensor_name(Scheme::Hf, t, cfg.num_layers),
                tensor_name(Scheme::Gguf, t, cfg.num_layers),
            )
        })
        .collect()
}

/// Row-major shape of a tensor given its GGUF name (`None` if unknown).
pub fn gguf_tensor_shape(cfg: &ModelConfig, name: &str) -> Option<Vec<usize>> {
    let d = cfg.hidden_size;
    let (i, f) = (cfg.intermediate_size, cfg.head_ff);
    Some(match name {
        "token_embd.weight" => vec![cfg.vocab_size, d],
        "token_embd_norm.weight"
        | "output_norm.weight"
        | "cls.norm.weight"
        | "cls.norm.bias"
        | "cls.bias" => vec![d],
        "token_types.weight" => vec![N_QTYPES, d],
        "cls.weight" => vec![d, d],
        "cls.output.weight" => vec![1, d],
        "cls.output.bias" => vec![1],
        _ => {
            let rest = name.strip_prefix("blk.")?;
            let (blk, part) = rest.split_once('.')?;
            let blk: usize = blk.parse().ok()?;
            let head = blk >= cfg.num_layers;
            match part {
                "attn_norm.weight" | "attn_norm.bias" | "ffn_norm.weight" | "ffn_norm.bias"
                | "attn_output.bias" | "ffn_down.bias" => vec![d],
                "attn_qkv.weight" => vec![3 * d, d],
                "attn_qkv.bias" => vec![3 * d],
                "attn_output.weight" => vec![d, d],
                "ffn_up.weight" => vec![if head { f } else { 2 * i }, d],
                "ffn_up.bias" => vec![f],
                "ffn_down.weight" => vec![d, if head { f } else { i }],
                _ => return None,
            }
        }
    })
}

fn all_tensors(cfg: &ModelConfig) -> Vec<T> {
    let mut v = vec![T::TokEmb, T::EmbNorm];
    for l in 0..cfg.num_layers {
        if l > 0 {
            v.push(T::AttnNorm(l));
        }
        v.extend([T::Wqkv(l), T::Wo(l), T::MlpNorm(l), T::Wi(l), T::MlpWo(l)]);
    }
    v.push(T::FinalNorm);
    for i in 0..cfg.head_layers {
        for p in [
            HeadPart::Norm1,
            HeadPart::InProj,
            HeadPart::OutProj,
            HeadPart::Norm2,
            HeadPart::Linear1,
            HeadPart::Linear2,
        ] {
            v.push(T::Head(i, p, false));
            v.push(T::Head(i, p, true));
        }
    }
    v.push(T::TypeEmb);
    for k in [0, 1, 3] {
        v.push(T::Scorer(k, false));
        v.push(T::Scorer(k, true));
    }
    v
}

/// Reads a tensor by name from the source file.
type TensorGetter<'a> = dyn Fn(&str) -> Result<Option<Loaded>> + 'a;

struct Loader<'a> {
    scheme: Scheme,
    n_enc: usize,
    get: Box<TensorGetter<'a>>,
}

impl<'a> Loader<'a> {
    fn opt(&self, t: T) -> Result<Option<Loaded>> {
        (self.get)(&tensor_name(self.scheme, t, self.n_enc))
    }

    fn req(&self, t: T, shape: &[usize]) -> Result<Loaded> {
        let name = tensor_name(self.scheme, t, self.n_enc);
        let l = (self.get)(&name)?.ok_or_else(|| Error::Missing(format!("tensor {name}")))?;
        if l.shape != shape {
            return Err(Error::Format(format!(
                "tensor {name}: shape {:?}, expected {shape:?}",
                l.shape
            )));
        }
        Ok(l)
    }

    fn vec(&self, t: T, n: usize) -> Result<Vec<f32>> {
        Ok(self.req(t, &[n])?.w.to_f32())
    }

    fn linear(&self, w: T, b: Option<T>, n_out: usize, n_in: usize) -> Result<Linear> {
        let lw = self.req(w, &[n_out, n_in])?;
        let b = match b {
            Some(b) => Some(self.vec(b, n_out)?),
            None => None,
        };
        Ok(Linear {
            w: lw.w,
            b,
            n_in,
            n_out,
            outliers: None,
        })
    }

    fn norm(&self, w: T, b: Option<T>, d: usize) -> Result<Norm> {
        Ok(Norm {
            w: self.vec(w, d)?,
            b: b.map(|b| self.vec(b, d)).transpose()?,
        })
    }
}

impl Model {
    fn load(cfg: ModelConfig, ld: &Loader<'_>) -> Result<Self> {
        cfg.validate()?;
        let d = cfg.hidden_size;
        let i_sz = cfg.intermediate_size;
        let emb = ld.req(T::TokEmb, &[cfg.vocab_size, d])?;
        let tok_emb = Embedding { w: emb.w, d };
        let emb_norm = ld.norm(T::EmbNorm, None, d)?;
        let mut layers = Vec::with_capacity(cfg.num_layers);
        for l in 0..cfg.num_layers {
            let attn_norm = if l == 0 {
                ld.opt(T::AttnNorm(0))?.map(|t| Norm {
                    w: t.w.to_f32(),
                    b: None,
                })
            } else {
                Some(ld.norm(T::AttnNorm(l), None, d)?)
            };
            layers.push(EncoderLayer {
                attn_norm,
                wqkv: ld.linear(T::Wqkv(l), None, 3 * d, d)?,
                wo: ld.linear(T::Wo(l), None, d, d)?,
                mlp_norm: ld.norm(T::MlpNorm(l), None, d)?,
                wi: ld.linear(T::Wi(l), None, 2 * i_sz, d)?,
                mlp_wo: ld.linear(T::MlpWo(l), None, d, i_sz)?,
            });
        }
        let final_norm = ld.norm(T::FinalNorm, None, d)?;
        let f = cfg.head_ff;
        let mut head = Vec::with_capacity(cfg.head_layers);
        for i in 0..cfg.head_layers {
            let h = |p: HeadPart, b: bool| T::Head(i, p, b);
            head.push(HeadLayer {
                norm1: ld.norm(h(HeadPart::Norm1, false), Some(h(HeadPart::Norm1, true)), d)?,
                in_proj: ld.linear(
                    h(HeadPart::InProj, false),
                    Some(h(HeadPart::InProj, true)),
                    3 * d,
                    d,
                )?,
                out_proj: ld.linear(
                    h(HeadPart::OutProj, false),
                    Some(h(HeadPart::OutProj, true)),
                    d,
                    d,
                )?,
                norm2: ld.norm(h(HeadPart::Norm2, false), Some(h(HeadPart::Norm2, true)), d)?,
                linear1: ld.linear(
                    h(HeadPart::Linear1, false),
                    Some(h(HeadPart::Linear1, true)),
                    f,
                    d,
                )?,
                linear2: ld.linear(
                    h(HeadPart::Linear2, false),
                    Some(h(HeadPart::Linear2, true)),
                    d,
                    f,
                )?,
            });
        }
        let type_emb = ld.req(T::TypeEmb, &[N_QTYPES, d])?.w.to_f32();
        let scorer_norm = ld.norm(T::Scorer(0, false), Some(T::Scorer(0, true)), d)?;
        let scorer1 = ld.linear(T::Scorer(1, false), Some(T::Scorer(1, true)), d, d)?;
        let scorer3 = ld.linear(T::Scorer(3, false), Some(T::Scorer(3, true)), 1, d)?;
        let hd = d / cfg.num_heads;
        let max_pos = cfg.max_len.max(1);
        Ok(Model {
            rope_global: Rope::new(hd, cfg.global_rope_theta, max_pos),
            rope_local: Rope::new(hd, cfg.local_rope_theta, max_pos),
            cfg,
            tok_emb,
            emb_norm,
            layers,
            final_norm,
            head,
            type_emb,
            scorer_norm,
            scorer1,
            scorer3,
            act_quant: ActQuant::default(),
        })
    }

    /// Load from a parsed safetensors file with HF / Laya tensor names.
    pub fn from_safetensors(st: &SafeTensors<'_>, cfg: ModelConfig) -> Result<Self> {
        let n_enc = cfg.num_layers;
        let get = move |name: &str| -> Result<Option<Loaded>> {
            match st.get(name) {
                None => Ok(None),
                Some(t) => Ok(Some(Loaded {
                    shape: t.shape.clone(),
                    w: Weight::F32(t.to_f32()?),
                })),
            }
        };
        Model::load(
            cfg,
            &Loader {
                scheme: Scheme::Hf,
                n_enc,
                get: Box::new(get),
            },
        )
    }

    /// Load from GGUF (llama.cpp `modern-bert` names). Q8_0 tensors stay quantized.
    pub fn from_gguf(g: &Gguf<'_>) -> Result<Self> {
        let cfg = ModelConfig::from_gguf(g)?;
        let n_enc = cfg.num_layers;
        let get = move |name: &str| -> Result<Option<Loaded>> {
            let Some(info) = g.tensor(name) else {
                return Ok(None);
            };
            let shape = info.shape();
            let bytes = g.tensor_bytes(info);
            let w = match info.dtype {
                GgmlType::Q8_0 => Weight::Q8(kernels::blocks_from_ggml(bytes)),
                ty => Weight::F32(sky_gguf::dequantize(ty, bytes)?),
            };
            Ok(Some(Loaded { shape, w }))
        };
        Model::load(
            cfg,
            &Loader {
                scheme: Scheme::Gguf,
                n_enc,
                get: Box::new(get),
            },
        )
    }

    fn linears_mut(&mut self) -> impl Iterator<Item = (bool, &mut Linear)> {
        let enc = self
            .layers
            .iter_mut()
            .flat_map(|l| [&mut l.wqkv, &mut l.wo, &mut l.wi, &mut l.mlp_wo])
            .map(|l| (false, l));
        let head = self
            .head
            .iter_mut()
            .flat_map(|h| {
                [
                    &mut h.in_proj,
                    &mut h.out_proj,
                    &mut h.linear1,
                    &mut h.linear2,
                ]
            })
            .chain([&mut self.scorer1, &mut self.scorer3])
            .map(|l| (true, l));
        enc.chain(head)
    }

    /// Number of matrix-product sites (see [`Model::calibrate`]): 4 per encoder / head layer
    /// (`Wqkv|in_proj`, `Wo|out_proj`, `Wi|linear1`, `mlp.Wo|linear2`), then `scorer.1`, `scorer.3`.
    pub fn n_sites(&self) -> usize {
        4 * (self.layers.len() + self.head.len()) + 2
    }

    /// GGUF tensor name of the weight of every site.
    pub fn site_gguf_names(&self) -> Vec<(usize, String)> {
        let n_enc = self.layers.len();
        let mut v = Vec::new();
        for l in 0..n_enc + self.head.len() {
            for (j, part) in ["attn_qkv", "attn_output", "ffn_up", "ffn_down"]
                .iter()
                .enumerate()
            {
                v.push((4 * l + j, format!("blk.{l}.{part}.weight")));
            }
        }
        let s = 4 * (n_enc + self.head.len());
        v.push((s, "cls.weight".into()));
        v.push((s + 1, "cls.output.weight".into()));
        v
    }

    /// Every linear layer with its site index and whether it belongs to the decision head.
    fn sites_mut(&mut self) -> Vec<(usize, bool, &mut Linear)> {
        let n_enc = self.layers.len();
        let mut v = Vec::new();
        for (l, layer) in self.layers.iter_mut().enumerate() {
            for (j, lin) in [
                &mut layer.wqkv,
                &mut layer.wo,
                &mut layer.wi,
                &mut layer.mlp_wo,
            ]
            .into_iter()
            .enumerate()
            {
                v.push((4 * l + j, false, lin));
            }
        }
        let n_head = self.head.len();
        for (i, h) in self.head.iter_mut().enumerate() {
            for (j, lin) in [
                &mut h.in_proj,
                &mut h.out_proj,
                &mut h.linear1,
                &mut h.linear2,
            ]
            .into_iter()
            .enumerate()
            {
                v.push((4 * (n_enc + i) + j, true, lin));
            }
        }
        let s = 4 * (n_enc + n_head);
        v.push((s, true, &mut self.scorer1));
        v.push((s + 1, true, &mut self.scorer3));
        v
    }

    /// Run the f32 model on calibration sequences `(ids, markers, qtype)` and collect the input
    /// statistics of every matrix product.
    pub fn calibrate(&self, seqs: &[CalibSeq]) -> Result<Calibration> {
        let mut ws = self.workspace();
        ws.calib = Some(Box::new(Calibration::new(self.n_sites())));
        let mut logits = Vec::new();
        for (ids, markers, qtype) in seqs {
            self.forward(&mut ws, ids, markers, *qtype, &mut logits)?;
        }
        Ok(*ws.calib.take().expect("calibration"))
    }

    /// Quantize the selected matrices to Q8_0, keeping the outlier input channels found by
    /// `cal` in f32 (`ratio`: see [`quant::outlier_channels`]). Matrices already in Q8_0 (from a
    /// GGUF file) are always processed: they keep their values and only move the outlier columns
    /// to the f32 path.
    pub fn quantize_calibrated(&mut self, spec: QuantizeSpec, cal: &Calibration, ratio: f32) {
        for (site, is_head, lin) in self.sites_mut() {
            let selected = (is_head && spec.head) || (!is_head && spec.encoder) || lin.w.is_q8();
            if !selected || !lin.n_in.is_multiple_of(QK) {
                continue;
            }
            let idx = match cal.sites.get(site).and_then(|s| s.as_ref()) {
                Some(st) if st.k == lin.n_in && st.n_rows > 0 => quant::outlier_channels(st, ratio),
                _ => Vec::new(),
            };
            lin.quantize_with_outliers(&idx);
        }
        if spec.embeddings {
            self.quantize(QuantizeSpec {
                encoder: false,
                head: false,
                embeddings: true,
            });
        }
    }

    /// Convert the selected matrices to Q8_0 with round-to-nearest (activations are then
    /// quantized per row on the fly). See [`Model::quantize_calibrated`] for better accuracy.
    pub fn quantize(&mut self, spec: QuantizeSpec) {
        for (is_head, lin) in self.linears_mut() {
            if (is_head && spec.head) || (!is_head && spec.encoder) {
                lin.quantize();
            }
        }
        if spec.embeddings && self.tok_emb.d.is_multiple_of(QK) {
            if let Weight::F32(v) = &self.tok_emb.w {
                let mut b = vec![BlockQ8::default(); v.len() / QK];
                kernels::quantize_q8(v, &mut b);
                self.tok_emb.w = Weight::Q8(b);
            }
        }
    }

    /// Convert every matrix back to f32.
    pub fn dequantize(&mut self) {
        for (_, lin) in self.linears_mut() {
            lin.dequantize();
        }
        if self.tok_emb.w.is_q8() {
            self.tok_emb.w = Weight::F32(self.tok_emb.w.to_f32());
        }
    }

    /// True if any matrix is stored as Q8_0.
    pub fn has_q8(&mut self) -> bool {
        self.linears_mut().any(|(_, l)| l.w.is_q8())
    }

    /// Total number of parameters (all tensors, embeddings included).
    pub fn n_params(&self) -> usize {
        let lin = |l: &Linear| l.w.n_elements() + l.b.as_ref().map_or(0, |b| b.len());
        let norm = |n: &Norm| n.w.len() + n.b.as_ref().map_or(0, |b| b.len());
        let mut n = self.tok_emb.w.n_elements()
            + norm(&self.emb_norm)
            + norm(&self.final_norm)
            + self.type_emb.len();
        for l in &self.layers {
            n += l.attn_norm.as_ref().map_or(0, norm)
                + lin(&l.wqkv)
                + lin(&l.wo)
                + norm(&l.mlp_norm)
                + lin(&l.wi)
                + lin(&l.mlp_wo);
        }
        for h in &self.head {
            n += norm(&h.norm1)
                + lin(&h.in_proj)
                + lin(&h.out_proj)
                + norm(&h.norm2)
                + lin(&h.linear1)
                + lin(&h.linear2);
        }
        n + norm(&self.scorer_norm) + lin(&self.scorer1) + lin(&self.scorer3)
    }

    /// Scratch buffers sized for this model.
    pub fn workspace(&self) -> Workspace {
        Workspace::new(&self.cfg)
    }

    /// Run the model on one sequence and return the raw scores (logits) at the marker positions
    /// for question type `qtype` (0 choice, 1 score, 2 noul).
    pub fn forward(
        &self,
        ws: &mut Workspace,
        ids: &[u32],
        markers: &[usize],
        qtype: usize,
        logits: &mut Vec<f32>,
    ) -> Result<()> {
        let c = &self.cfg;
        let t = ids.len();
        let d = c.hidden_size;
        if t == 0 || t > ws.max_t {
            return Err(Error::InvalidRequest(format!(
                "sequence of {t} tokens (max {})",
                ws.max_t
            )));
        }
        if qtype >= N_QTYPES {
            return Err(Error::InvalidRequest(format!("question type {qtype}")));
        }
        if markers.is_empty() || markers.iter().any(|&m| m >= t) {
            return Err(Error::InvalidRequest(
                "marker positions out of range".into(),
            ));
        }
        if let Some(&bad) = ids.iter().find(|&&id| id as usize >= c.vocab_size) {
            return Err(Error::InvalidRequest(format!(
                "token id {bad} out of vocabulary"
            )));
        }
        let k = markers.len();
        let Workspace {
            x,
            hn,
            qkv,
            att,
            up,
            act,
            xq,
            attn,
            iota,
            mx,
            mh,
            mq,
            matt,
            mup,
            calib,
            ..
        } = ws;
        let xq = &mut Act {
            xq: xq.as_mut_slice(),
            mode: self.act_quant,
        };
        let mut rec = |site: usize, x: &[f32], rows: usize, k: usize| {
            if let Some(c) = calib.as_deref_mut() {
                c.record(site, x, rows, k);
            }
        };
        let x = &mut x[..t * d];
        let hn = &mut hn[..t * d];

        // embeddings
        for (r, &id) in ids.iter().enumerate() {
            self.tok_emb
                .row_into(id as usize, &mut x[r * d..(r + 1) * d]);
        }
        kernels::layer_norm_inplace(x, d, &self.emb_norm.w, None, c.norm_eps);

        // encoder
        let (nh, hd) = (c.num_heads, c.head_dim());
        let i_sz = c.intermediate_size;
        for (l, layer) in self.layers.iter().enumerate() {
            let global = c.is_global(l);
            let rope = if global {
                &self.rope_global
            } else {
                &self.rope_local
            };
            let window = if global {
                None
            } else {
                Some(c.local_attention / 2)
            };
            let qkv = &mut qkv[..t * 3 * d];
            match &layer.attn_norm {
                Some(n) => {
                    kernels::layer_norm(x, d, &n.w, n.b.as_deref(), c.norm_eps, hn);
                    rec(4 * l, hn, t, d);
                    linear(layer.wqkv.view(), hn, t, qkv, false, xq);
                }
                None => {
                    rec(4 * l, x, t, d);
                    linear(layer.wqkv.view(), x, t, qkv, false, xq)
                }
            }
            for r in 0..t {
                let row = &mut qkv[r * 3 * d..r * 3 * d + 2 * d];
                for h in 0..nh {
                    rope.apply(&mut row[h * hd..(h + 1) * hd], r);
                    rope.apply(&mut row[d + h * hd..d + (h + 1) * hd], r);
                }
            }
            let a = Attention {
                k: &qkv[d..],
                v: &qkv[2 * d..],
                kv_stride: 3 * d,
                n_kv: t,
                n_heads: nh,
                head_dim: hd,
                window,
            };
            a.run(qkv, 3 * d, &iota[..t], att, d, attn);
            rec(4 * l + 1, &att[..t * d], t, d);
            linear(layer.wo.view(), &att[..t * d], t, x, true, xq);
            kernels::layer_norm(x, d, &layer.mlp_norm.w, None, c.norm_eps, hn);
            let up = &mut up[..t * 2 * i_sz];
            rec(4 * l + 2, hn, t, d);
            linear(layer.wi.view(), hn, t, up, false, xq);
            let act = &mut act[..t * i_sz];
            for r in 0..t {
                let (u, a) = (
                    &up[r * 2 * i_sz..(r + 1) * 2 * i_sz],
                    &mut act[r * i_sz..(r + 1) * i_sz],
                );
                for j in 0..i_sz {
                    a[j] = kernels::gelu(u[j]) * u[i_sz + j];
                }
            }
            rec(4 * l + 3, act, t, i_sz);
            linear(layer.mlp_wo.view(), act, t, x, true, xq);
        }
        kernels::layer_norm_inplace(x, d, &self.final_norm.w, None, c.norm_eps);

        // decision head
        let te = &self.type_emb[qtype * d..(qtype + 1) * d];
        for r in 0..t {
            for (v, &e) in x[r * d..(r + 1) * d].iter_mut().zip(te) {
                *v += e;
            }
        }
        let (hh, hhd) = (c.head_heads, d / c.head_heads);
        let f = c.head_ff;
        let eps = c.head_norm_eps;
        let mx = &mut mx[..k * d];
        let mh = &mut mh[..k * d];
        let n_head = self.head.len();
        let n_enc = self.layers.len();
        for (i, hl) in self.head.iter().enumerate() {
            let site = 4 * (n_enc + i);
            kernels::layer_norm(x, d, &hl.norm1.w, hl.norm1.b.as_deref(), eps, hn);
            rec(site, hn, t, d);
            if i + 1 < n_head {
                let qkv = &mut qkv[..t * 3 * d];
                linear(hl.in_proj.view(), hn, t, qkv, false, xq);
                let a = Attention {
                    k: &qkv[d..],
                    v: &qkv[2 * d..],
                    kv_stride: 3 * d,
                    n_kv: t,
                    n_heads: hh,
                    head_dim: hhd,
                    window: None,
                };
                a.run(qkv, 3 * d, &iota[..t], att, d, attn);
                rec(site + 1, &att[..t * d], t, d);
                linear(hl.out_proj.view(), &att[..t * d], t, x, true, xq);
                kernels::layer_norm(x, d, &hl.norm2.w, hl.norm2.b.as_deref(), eps, hn);
                let up = &mut up[..t * f];
                rec(site + 2, hn, t, d);
                linear(hl.linear1.view(), hn, t, up, false, xq);
                up.iter_mut().for_each(|v| *v = v.max(0.0));
                rec(site + 3, up, t, f);
                linear(hl.linear2.view(), up, t, x, true, xq);
            } else {
                // last head layer: keys/values for every position, everything else only at the markers
                let kv = &mut qkv[..t * 2 * d];
                linear(hl.in_proj.rows(d, 3 * d), hn, t, kv, false, xq);
                for (m, &p) in markers.iter().enumerate() {
                    mh[m * d..(m + 1) * d].copy_from_slice(&hn[p * d..(p + 1) * d]);
                    mx[m * d..(m + 1) * d].copy_from_slice(&x[p * d..(p + 1) * d]);
                }
                let mq = &mut mq[..k * d];
                linear(hl.in_proj.rows(0, d), mh, k, mq, false, xq);
                let a = Attention {
                    k: &kv[..],
                    v: &kv[d..],
                    kv_stride: 2 * d,
                    n_kv: t,
                    n_heads: hh,
                    head_dim: hhd,
                    window: None,
                };
                let matt = &mut matt[..k * d];
                a.run(mq, d, markers, matt, d, attn);
                rec(site + 1, matt, k, d);
                linear(hl.out_proj.view(), matt, k, mx, true, xq);
                kernels::layer_norm(mx, d, &hl.norm2.w, hl.norm2.b.as_deref(), eps, mh);
                let mup = &mut mup[..k * f];
                rec(site + 2, mh, k, d);
                linear(hl.linear1.view(), mh, k, mup, false, xq);
                mup.iter_mut().for_each(|v| *v = v.max(0.0));
                rec(site + 3, mup, k, f);
                linear(hl.linear2.view(), mup, k, mx, true, xq);
            }
        }
        if n_head == 0 {
            for (m, &p) in markers.iter().enumerate() {
                mx[m * d..(m + 1) * d].copy_from_slice(&x[p * d..(p + 1) * d]);
            }
        }

        // scorer at the markers only
        kernels::layer_norm(
            mx,
            d,
            &self.scorer_norm.w,
            self.scorer_norm.b.as_deref(),
            eps,
            mh,
        );
        let s1 = &mut matt[..k * d];
        let ssite = 4 * (n_enc + n_head);
        rec(ssite, mh, k, d);
        linear(self.scorer1.view(), mh, k, s1, false, xq);
        s1.iter_mut().for_each(|v| *v = kernels::gelu(*v));
        logits.clear();
        logits.resize(k, 0.0);
        rec(ssite + 1, s1, k, d);
        linear(self.scorer3.view(), s1, k, logits, false, xq);
        Ok(())
    }
}

/// Scratch of the quantized activations, with the activation mode of the model.
pub(crate) struct Act<'a> {
    pub(crate) xq: &'a mut [ActBlock],
    pub(crate) mode: ActQuant,
}

pub(crate) fn linear(
    lin: LinearView<'_>,
    x: &[f32],
    rows: usize,
    y: &mut [f32],
    accumulate: bool,
    act: &mut Act<'_>,
) {
    match lin.w {
        WeightRef::F32(w) => {
            kernels::matmul_f32(x, rows, lin.n_in, w, lin.n_out, lin.b, y, accumulate)
        }
        WeightRef::Q8(w) => kernels::matmul_q8(
            x,
            rows,
            lin.n_in,
            w,
            lin.n_out,
            lin.b,
            y,
            accumulate,
            act.xq,
            lin.outliers,
            act.mode,
        ),
    }
}

/// Pre-allocated scratch buffers: the forward pass does not allocate.
#[derive(Clone, Debug)]
pub struct Workspace {
    max_t: usize,
    x: Vec<f32>,
    hn: Vec<f32>,
    qkv: Vec<f32>,
    att: Vec<f32>,
    up: Vec<f32>,
    act: Vec<f32>,
    xq: Vec<ActBlock>,
    attn: AttnScratch,
    /// 0, 1, 2, ... (positions of the full sequence).
    iota: Vec<usize>,
    mx: Vec<f32>,
    mh: Vec<f32>,
    mq: Vec<f32>,
    matt: Vec<f32>,
    mup: Vec<f32>,
    /// When set, the forward pass records the inputs of every matrix product.
    calib: Option<Box<Calibration>>,
}

impl Workspace {
    pub fn new(cfg: &ModelConfig) -> Self {
        let t = cfg.max_len.max(1);
        let d = cfg.hidden_size;
        let up = (2 * cfg.intermediate_size).max(cfg.head_ff);
        let max_k = d.max(cfg.intermediate_size).max(cfg.head_ff);
        Workspace {
            max_t: t,
            x: vec![0.0; t * d],
            hn: vec![0.0; t * d],
            qkv: vec![0.0; t * 3 * d],
            att: vec![0.0; t * d],
            up: vec![0.0; t * up],
            act: vec![0.0; t * cfg.intermediate_size],
            xq: vec![ActBlock::default(); t * max_k.div_ceil(QK)],
            attn: AttnScratch::new(t, (d / cfg.num_heads).max(d / cfg.head_heads)),
            iota: (0..t).collect(),
            mx: vec![0.0; t * d],
            mh: vec![0.0; t * d],
            mq: vec![0.0; t * d],
            matt: vec![0.0; t * d],
            mup: vec![0.0; t * cfg.head_ff],
            calib: None,
        }
    }

    pub fn max_tokens(&self) -> usize {
        self.max_t
    }
}
