//! High-level engine: tokenizer + model + prompt + readout, with caches for the game loop
//! (the question and options are the same every frame, only the state changes).
//!
//! [`Engine`] runs the model families behind one API, chosen by `general.architecture`:
//! `modern-bert` (s1, the Laya / ModernBERT layout), `skycmd-s2` (s2, see [`crate::s2`]) and
//! `skycmd-iso` (s5 variant B, see [`crate::iso`]).

use sky_gguf::Gguf;

use crate::api::{Answer, DecisionRequest, DecisionResponse, ExitInfo};
use crate::iso::{IsoEngine, IsoModel};
use crate::kernels::ActQuant;
use crate::model::{Model, QuantizeSpec, Workspace};
use crate::prompt::{self, Prefix, PromptTokens, QType, QuestionKind};
use crate::s2::{S2Engine, S2Model, S2Options, TernaryCompute};
use crate::tokenizer::{BpeCache, Tokenizer};
use crate::{Error, Result};

/// Loading options.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EngineOptions {
    /// s1: convert these parts to Q8_0 after loading.
    pub quantize: Option<QuantizeSpec>,
    /// s1: for Q8_0 matrices (quantized at load or read from GGUF), run the built-in calibration
    /// prompts and compute the outlier input channels of every matrix in f32 (recommended, see
    /// [`crate::quant`]).
    pub calibrate: bool,
    /// s1: convert every Q8_0 matrix to f32 after loading (reference path).
    pub dequantize: bool,
    /// s1: activation quantization of the Q8_0 products (accuracy / speed trade-off).
    pub act_quant: ActQuant,
    /// s1: number of (question, options) prefixes kept tokenized. s2: entries of the static
    /// encoding LRU (0 disables it).
    pub prefix_cache: usize,
    /// Number of BPE words kept in the word cache.
    pub bpe_cache: usize,
    /// s2: overrides the early-exit threshold τ of every question type (1.0 = full depth).
    pub exit_threshold: Option<f32>,
    /// s2: how ternary matrices are computed.
    pub ternary: TernaryCompute,
    /// s2: state encodings kept for multi-question frames (0 disables the cache).
    pub state_cache: usize,
}

impl Default for EngineOptions {
    fn default() -> Self {
        EngineOptions {
            quantize: None,
            calibrate: true,
            dequantize: false,
            act_quant: ActQuant::default(),
            prefix_cache: 16,
            bpe_cache: 4096,
            exit_threshold: None,
            ternary: TernaryCompute::Auto,
            state_cache: 4,
        }
    }
}

/// Model family of a loaded engine.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Arch {
    /// `modern-bert`: ModernBERT encoder + Laya decision head.
    S1,
    /// `skycmd-s2`: static / state streams, fusion head, early exits.
    S2,
    /// `skycmd-iso`: options isolated by construction, definitions only, state cache.
    Iso,
}

impl Arch {
    /// The `general.architecture` value.
    pub fn name(self) -> &'static str {
        match self {
            Arch::S1 => "modern-bert",
            Arch::S2 => crate::s2::ARCH,
            Arch::Iso => crate::iso::ARCH,
        }
    }
}

/// Statistics of the last answered question.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct DecisionStats {
    /// s2: index of the exit that answered.
    pub exit: Option<usize>,
    /// s2: state-encoder blocks used.
    pub layers_used: Option<usize>,
    /// s2: number of exits of the model (0 for s1).
    pub n_exits: usize,
    /// s1: prefix tokenization cached; s2: static encoding cached.
    pub static_hit: bool,
    /// s2: state encoding reused from a previous question on the same state.
    pub state_hit: bool,
    /// Tokens of the question (full sequence for s1, static + state streams for s2).
    pub tokens: usize,
}

/// The few model facts available without `&mut` (see [`Engine::model_info`] for the rest).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ModelInfoRef {
    pub num_layers: usize,
}

/// Summary of a loaded model.
#[derive(Clone, Debug, PartialEq)]
pub struct ModelInfo {
    pub name: String,
    /// `general.architecture` (`modern-bert` or `skycmd-s2`).
    pub arch: String,
    pub n_params: usize,
    pub vocab_size: usize,
    pub hidden_size: usize,
    /// s1: encoder layers; s2: state-encoder blocks.
    pub num_layers: usize,
    pub num_heads: usize,
    pub intermediate_size: usize,
    /// s1: decision-head layers; s2: fusion blocks.
    pub head_layers: usize,
    /// s1: longest sequence; s2: longest state stream.
    pub max_len: usize,
    pub head_max_len: usize,
    /// Quantized products (Q8_0 or Q4_0 weights, int8 activations).
    pub q8: bool,
    /// `f32`, `q8_0`, `q4_0` (s1, iso) or the s2 `weight_format` (`f32`, `f16`, `ternary`).
    pub weight_format: String,
    pub temperatures: Vec<(String, f32)>,
    /// s2: static-encoder blocks.
    pub static_layers: usize,
    /// s2: depth of each exit.
    pub exits: Vec<usize>,
    /// s2: τ per question type.
    pub exit_thresholds: Vec<(String, f32)>,
    /// s2: ternary matrices computed with the packed add/sub kernels.
    pub ternary_packed: bool,
}

impl ModelInfo {
    pub fn to_json(&self) -> String {
        let pairs = |v: &[(String, f32)]| -> String {
            v.iter()
                .map(|(k, v)| format!("{}:{}", crate::json::escape(k), v))
                .collect::<Vec<_>>()
                .join(",")
        };
        let mut s = format!(
            "{{\"name\":{},\"arch\":{},\"n_params\":{},\"vocab_size\":{},\"hidden_size\":{},\"num_layers\":{},\"num_heads\":{},\
             \"intermediate_size\":{},\"head_layers\":{},\"max_len\":{},\"head_max_len\":{},\"q8\":{},\"weight_format\":{},\"temperatures\":{{{}}}",
            crate::json::escape(&self.name),
            crate::json::escape(&self.arch),
            self.n_params,
            self.vocab_size,
            self.hidden_size,
            self.num_layers,
            self.num_heads,
            self.intermediate_size,
            self.head_layers,
            self.max_len,
            self.head_max_len,
            self.q8,
            crate::json::escape(&self.weight_format),
            pairs(&self.temperatures)
        );
        if !self.exits.is_empty() {
            let exits: Vec<String> = self.exits.iter().map(|e| e.to_string()).collect();
            s.push_str(&format!(
                ",\"static_layers\":{},\"exits\":[{}],\"exit_thresholds\":{{{}}},\"ternary_packed\":{}",
                self.static_layers,
                exits.join(","),
                pairs(&self.exit_thresholds),
                self.ternary_packed
            ));
        }
        s.push('}');
        s
    }
}

struct CachedPrefix {
    qtype: QType,
    instructions: String,
    options: Vec<String>,
    prefix: Prefix,
}

/// The s1 engine (ModernBERT + decision head).
struct S1Engine {
    model: Model,
    tok: Tokenizer,
    ws: Workspace,
    cls: u32,
    sep: u32,
    mask: u32,
    mask_text: String,
    prefixes: Vec<CachedPrefix>,
    prefix_cap: usize,
    bpe: BpeCache,
    ids: Vec<u32>,
    markers: Vec<usize>,
    logits: Vec<f32>,
    /// Whether the last `prepare` found the prefix cached.
    prefix_hit: bool,
}

impl S1Engine {
    fn new(model: Model, tok: Tokenizer, opts: EngineOptions) -> Result<Self> {
        let mut model = model;
        model.act_quant = opts.act_quant;
        if opts.dequantize {
            model.dequantize();
        }
        let none = QuantizeSpec {
            encoder: false,
            head: false,
            embeddings: false,
        };
        let spec = opts.quantize.unwrap_or(none);
        if opts.calibrate && (opts.quantize.is_some() || model.has_q8()) {
            let seqs = crate::quant::calibration_sequences(&tok, &model.cfg)?;
            let cal = model.calibrate(&seqs)?;
            model.quantize_calibrated(spec, &cal, crate::quant::OUTLIER_RATIO);
        } else if opts.quantize.is_some() {
            model.quantize(spec);
        }
        let pt = PromptTokens::from_tokenizer(&tok)?;
        let (cls, sep, mask, mask_text) = (pt.cls, pt.sep, pt.mask, pt.mask_text.to_string());
        let ws = model.workspace();
        let max_t = ws.max_tokens();
        Ok(S1Engine {
            ws,
            model,
            tok,
            cls,
            sep,
            mask,
            mask_text,
            prefixes: Vec::new(),
            prefix_cap: opts.prefix_cache,
            bpe: BpeCache::new(opts.bpe_cache),
            ids: Vec::with_capacity(max_t),
            markers: Vec::with_capacity(max_t),
            logits: Vec::with_capacity(max_t),
            prefix_hit: false,
        })
    }

    fn temperature(&self, qtype: QType, n: usize) -> f32 {
        self.model
            .cfg
            .temperature(qtype.name(), prompt::temp_bucket(n))
    }

    /// Fill `self.ids` / `self.markers`, reusing the cached tokenization of the prefix.
    fn prepare<S: AsRef<str>>(
        &mut self,
        qtype: QType,
        instructions: &str,
        options: &[S],
        state: &str,
    ) {
        let hit = self.prefixes.iter().position(|c| {
            c.qtype == qtype
                && c.instructions == instructions
                && c.options.len() == options.len()
                && c.options.iter().zip(options).all(|(a, b)| a == b.as_ref())
        });
        self.prefix_hit = hit.is_some();
        let pt = PromptTokens {
            cls: self.cls,
            sep: self.sep,
            mask: self.mask,
            mask_text: &self.mask_text,
        };
        let idx = match hit {
            Some(i) => i,
            None => {
                let opts: Vec<&str> = options.iter().map(|o| o.as_ref()).collect();
                let prefix = prompt::build_prefix(
                    &self.tok,
                    &pt,
                    qtype,
                    instructions,
                    &opts,
                    self.model.cfg.head_max_len,
                    Some(&mut self.bpe),
                );
                if self.prefixes.len() >= self.prefix_cap.max(1) {
                    self.prefixes.pop();
                }
                self.prefixes.insert(
                    0,
                    CachedPrefix {
                        qtype,
                        instructions: instructions.to_string(),
                        options: options.iter().map(|o| o.as_ref().to_string()).collect(),
                        prefix,
                    },
                );
                0
            }
        };
        if idx > 0 {
            let c = self.prefixes.remove(idx);
            self.prefixes.insert(0, c);
        }
        prompt::finish_sequence(
            &self.tok,
            &pt,
            &self.prefixes[0].prefix,
            state,
            self.model.cfg.max_len,
            &mut self.ids,
            &mut self.markers,
            Some(&mut self.bpe),
        );
    }

    /// Run the model on prepared ids; leaves the raw scores in `self.logits`.
    fn run(&mut self, qtype: QType, n_options: usize) -> Result<()> {
        if self.markers.len() != n_options {
            return Err(Error::InvalidRequest(format!(
                "options do not fit in head_max_len={} tokens",
                self.model.cfg.head_max_len
            )));
        }
        self.model.forward(
            &mut self.ws,
            &self.ids,
            &self.markers,
            qtype.index(),
            &mut self.logits,
        )
    }

    fn stats(&self) -> DecisionStats {
        DecisionStats {
            static_hit: self.prefix_hit,
            tokens: self.ids.len(),
            ..Default::default()
        }
    }
}

enum Inner {
    S1(Box<S1Engine>),
    S2(Box<S2Engine>),
    Iso(Box<IsoEngine>),
}

/// Inference engine for one model (s1 or s2). Not thread-safe by design (`&mut self`): it
/// reuses pre-allocated buffers so that a decision does not allocate in the forward pass.
pub struct Engine {
    inner: Inner,
    last_micros: Option<u64>,
    last: Option<DecisionStats>,
}

/// Rendered option texts and probabilities → the typed answer.
fn answer_of(kind: &QuestionKind, p: &[f32]) -> Answer {
    match kind {
        QuestionKind::Choice(opts) => Answer::Choice {
            choice: opts[prompt::argmax(p)].0.clone(),
            probabilities: opts
                .iter()
                .map(|(k, _)| k.clone())
                .zip(p.iter().copied())
                .collect(),
            confidence: prompt::confidence_choice(p),
        },
        QuestionKind::Score(levels) => Answer::Score {
            score: p.iter().enumerate().map(|(i, &v)| i as f32 * v).sum(),
            legend: levels
                .iter()
                .enumerate()
                .map(|(i, c)| (i.to_string(), c.clone()))
                .collect(),
            probabilities: p
                .iter()
                .enumerate()
                .map(|(i, &v)| (i.to_string(), v))
                .collect(),
            confidence: prompt::confidence_score(p),
        },
        QuestionKind::Noul { .. } => Answer::Noul { noul: p[1] },
    }
}

impl Engine {
    fn wrap(inner: Inner) -> Self {
        Engine {
            inner,
            last_micros: None,
            last: None,
        }
    }

    /// Build an s1 engine from a model and its tokenizer.
    pub fn new(model: Model, tok: Tokenizer, opts: EngineOptions) -> Result<Self> {
        Ok(Self::wrap(Inner::S1(Box::new(S1Engine::new(
            model, tok, opts,
        )?))))
    }

    /// Build an s2 engine from a model and its tokenizer (needs `[NUM]` at `num_token_id`).
    pub fn new_s2(model: S2Model, tok: Tokenizer, opts: EngineOptions) -> Result<Self> {
        let o = S2Options {
            exit_threshold: opts.exit_threshold,
            static_cache: opts.prefix_cache,
            state_cache: opts.state_cache,
            ternary: opts.ternary,
            bpe_cache: opts.bpe_cache,
        };
        Ok(Self::wrap(Inner::S2(Box::new(S2Engine::new(
            model, tok, o,
        )?))))
    }

    /// Load a GGUF file: `general.architecture = "skycmd-s2"` gives an s2 engine, anything else
    /// (`modern-bert`, the llama.cpp decision model) an s1 engine.
    pub fn from_gguf(bytes: &[u8]) -> Result<Self> {
        Self::from_gguf_with(bytes, EngineOptions::default())
    }

    pub fn from_gguf_with(bytes: &[u8], opts: EngineOptions) -> Result<Self> {
        let g = Gguf::parse(bytes)?;
        let tok = Tokenizer::from_gguf(&g)?;
        if g.architecture() == Some(crate::s2::ARCH) {
            let model = S2Model::from_gguf(&g)?;
            Self::new_s2(model, tok, opts)
        } else if g.architecture() == Some(crate::iso::ARCH) {
            let mut model = IsoModel::from_gguf(&g)?;
            model.act_quant = opts.act_quant;
            if opts.dequantize {
                model.dequantize();
            }
            Ok(Self::wrap(Inner::Iso(Box::new(IsoEngine::new(
                model,
                tok,
                opts.state_cache,
                opts.bpe_cache,
            )?))))
        } else {
            let model = Model::from_gguf(&g)?;
            Self::new(model, tok, opts)
        }
    }

    /// Load a Laya checkpoint directory (s1): `model.safetensors`, `encoder/config.json`,
    /// `rl_agent_config.json`, `tokenizer/tokenizer.json` (see [`crate::convert::LayaDir`]).
    #[cfg(not(target_arch = "wasm32"))]
    pub fn from_safetensors_dir(path: impl AsRef<std::path::Path>) -> Result<Self> {
        Self::from_safetensors_dir_with(path, EngineOptions::default())
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn from_safetensors_dir_with(
        path: impl AsRef<std::path::Path>,
        opts: EngineOptions,
    ) -> Result<Self> {
        let dir = crate::convert::LayaDir::open(path.as_ref())?;
        let (model, tok) = dir.load()?;
        Self::new(model, tok, opts)
    }

    pub fn arch(&self) -> Arch {
        match self.inner {
            Inner::S1(_) => Arch::S1,
            Inner::S2(_) => Arch::S2,
            Inner::Iso(_) => Arch::Iso,
        }
    }

    /// The iso engine (state cache, language of the generic `noul` definitions), if this
    /// engine runs a `skycmd-iso` model.
    pub fn iso_engine_mut(&mut self) -> Option<&mut IsoEngine> {
        match &mut self.inner {
            Inner::Iso(e) => Some(e),
            _ => None,
        }
    }

    /// The s1 model.
    ///
    /// # Panics
    /// If the engine runs an s2 model (see [`Engine::s1_model`] and [`Engine::s2_model`]).
    pub fn model(&self) -> &Model {
        self.s1_model()
            .expect("Engine::model() is only available for s1 models")
    }

    /// The s1 model, if this engine runs one.
    pub fn s1_model(&self) -> Option<&Model> {
        match &self.inner {
            Inner::S1(e) => Some(&e.model),
            _ => None,
        }
    }

    /// The s2 model, if this engine runs one.
    pub fn s2_model(&self) -> Option<&S2Model> {
        match &self.inner {
            Inner::S2(e) => Some(e.model()),
            _ => None,
        }
    }

    /// The s2 engine (caches, per-exit probabilities), if this engine runs an s2 model.
    pub fn s2_engine_mut(&mut self) -> Option<&mut S2Engine> {
        match &mut self.inner {
            Inner::S2(e) => Some(e),
            _ => None,
        }
    }

    pub fn tokenizer(&self) -> &Tokenizer {
        match &self.inner {
            Inner::S1(e) => &e.tok,
            Inner::S2(e) => e.tokenizer(),
            Inner::Iso(e) => e.tokenizer(),
        }
    }

    /// Model name (`general.name`).
    pub fn name(&self) -> &str {
        match &self.inner {
            Inner::S1(e) => &e.model.cfg.name,
            Inner::S2(e) => &e.model().cfg.name,
            Inner::Iso(e) => &e.model().cfg.name,
        }
    }

    pub fn n_params(&self) -> usize {
        match &self.inner {
            Inner::S1(e) => e.model.n_params(),
            Inner::S2(e) => e.model().n_params(),
            Inner::Iso(e) => e.model().n_params(),
        }
    }

    pub fn model_info(&mut self) -> ModelInfo {
        match &mut self.inner {
            Inner::S1(e) => {
                let q8 = e.model.has_q8();
                let format = e.model.weight_format();
                let c = &e.model.cfg;
                ModelInfo {
                    name: c.name.clone(),
                    arch: Arch::S1.name().into(),
                    n_params: e.model.n_params(),
                    vocab_size: c.vocab_size,
                    hidden_size: c.hidden_size,
                    num_layers: c.num_layers,
                    num_heads: c.num_heads,
                    intermediate_size: c.intermediate_size,
                    head_layers: c.head_layers,
                    max_len: c.max_len,
                    head_max_len: c.head_max_len,
                    q8,
                    weight_format: format.into(),
                    temperatures: c.temperatures.clone(),
                    static_layers: 0,
                    exits: Vec::new(),
                    exit_thresholds: Vec::new(),
                    ternary_packed: false,
                }
            }
            Inner::S2(e) => {
                let n_params = e.model().n_params();
                let packed = e.options().ternary.resolve() == TernaryCompute::Packed
                    && e.model().cfg.weight_format == "ternary";
                let c = &e.model().cfg;
                ModelInfo {
                    name: c.name.clone(),
                    arch: Arch::S2.name().into(),
                    n_params,
                    vocab_size: c.vocab_size,
                    hidden_size: c.hidden_size,
                    num_layers: c.state_layers,
                    num_heads: c.num_heads,
                    intermediate_size: c.intermediate_size,
                    head_layers: c.fusion_layers,
                    max_len: c.max_state_tokens,
                    head_max_len: c.max_head_tokens,
                    q8: false,
                    weight_format: c.weight_format.clone(),
                    temperatures: c.temperatures.clone(),
                    static_layers: c.static_layers,
                    exits: c.exits.clone(),
                    exit_thresholds: ["choice", "score", "noul"]
                        .iter()
                        .zip(c.exit_thresholds)
                        .map(|(k, v)| (k.to_string(), v))
                        .collect(),
                    ternary_packed: packed,
                }
            }
            Inner::Iso(e) => {
                let m = e.model();
                let q8 = m.has_q8();
                let c = &m.cfg;
                ModelInfo {
                    name: c.name.clone(),
                    arch: Arch::Iso.name().into(),
                    n_params: m.n_params(),
                    vocab_size: c.vocab_size,
                    hidden_size: c.hidden_size,
                    num_layers: c.num_layers,
                    num_heads: c.num_heads,
                    intermediate_size: c.intermediate_size,
                    head_layers: 0,
                    max_len: c.max_len,
                    head_max_len: c.head_max_len,
                    q8,
                    weight_format: m.weight_format().into(),
                    temperatures: c.temperatures.clone(),
                    static_layers: 0,
                    exits: Vec::new(),
                    exit_thresholds: Vec::new(),
                    ternary_packed: false,
                }
            }
        }
    }

    /// The number of encoder layers of the loaded model (any family).
    pub fn model_info_ref(&self) -> ModelInfoRef {
        match &self.inner {
            Inner::S1(e) => ModelInfoRef {
                num_layers: e.model.cfg.num_layers,
            },
            Inner::S2(e) => ModelInfoRef {
                num_layers: e.model().cfg.state_layers,
            },
            Inner::Iso(e) => ModelInfoRef {
                num_layers: e.model().cfg.num_layers,
            },
        }
    }

    /// Wall time of the last decision in microseconds (feature `timing`).
    pub fn last_micros(&self) -> Option<u64> {
        self.last_micros
    }

    /// Statistics of the last answered question (exit, layers used, cache hits).
    pub fn last_stats(&self) -> Option<&DecisionStats> {
        self.last.as_ref()
    }

    /// s2: override the early-exit threshold τ of every question type (`None`: the calibrated
    /// values of the file; 1.0: full depth). No effect on s1.
    pub fn set_exit_threshold(&mut self, tau: Option<f32>) {
        if let Inner::S2(e) = &mut self.inner {
            e.set_exit_threshold(tau);
        }
    }

    /// Temperature applied to a question type with `n` options (s2: at the last exit).
    pub fn temperature(&self, qtype: QType, n: usize) -> f32 {
        match &self.inner {
            Inner::S1(e) => e.temperature(qtype, n),
            Inner::S2(e) => e.temperature(e.model().cfg.exits.len() - 1, qtype, n),
            Inner::Iso(e) => e.temperature(qtype, n),
        }
    }

    /// Token ids and marker positions of a question. s1: the full sequence (same as
    /// `rl_common.build_sequence`). s2: the static stream followed by the state stream.
    pub fn build_sequence(
        &mut self,
        qtype: QType,
        instructions: &str,
        options: &[&str],
        state: &str,
    ) -> (Vec<u32>, Vec<usize>) {
        match &mut self.inner {
            Inner::S1(e) => {
                e.prepare(qtype, instructions, options, state);
                (e.ids.clone(), e.markers.clone())
            }
            Inner::S2(e) => {
                let (mut ids, markers, state_ids) =
                    e.sequences(qtype, instructions, options, state);
                ids.extend(state_ids);
                (ids, markers)
            }
            Inner::Iso(e) => e.build_sequence(qtype, instructions, options, state),
        }
    }

    /// Probabilities and raw scores of one question, recording the statistics.
    fn question<S: AsRef<str>>(
        &mut self,
        qtype: QType,
        instructions: &str,
        options: &[S],
        state: &str,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        match &mut self.inner {
            Inner::S1(e) => {
                e.prepare(qtype, instructions, options, state);
                e.run(qtype, options.len())?;
                let p = prompt::probs_from_logits(&e.logits, e.temperature(qtype, options.len()));
                self.last = Some(e.stats());
                Ok((p, e.logits.clone()))
            }
            Inner::S2(e) => {
                let o = e.question(qtype, instructions, options, state)?;
                self.last = Some(DecisionStats {
                    exit: Some(o.exit),
                    layers_used: Some(o.layers_used),
                    n_exits: e.model().cfg.exits.len(),
                    static_hit: o.static_hit,
                    state_hit: o.state_hit,
                    tokens: o.tokens,
                });
                Ok((o.probs, o.logits))
            }
            Inner::Iso(e) => {
                let o = e.question(qtype, instructions, options, state)?;
                self.last = Some(DecisionStats {
                    state_hit: o.state_hit,
                    tokens: o.tokens,
                    ..Default::default()
                });
                Ok((o.probs, o.logits))
            }
        }
    }

    /// Raw scores (before temperature) of rendered option texts (s2: at the exit that answers).
    pub fn scores_raw<S: AsRef<str>>(
        &mut self,
        qtype: QType,
        instructions: &str,
        options: &[S],
        state: &str,
    ) -> Result<Vec<f32>> {
        Ok(self.question(qtype, instructions, options, state)?.1)
    }

    /// Fast path: probabilities (after the temperature) of already rendered option texts
    /// (e.g. `["flap: push the bird up", "wait"]`, or the `opts` of a decision record).
    pub fn decide_raw<S: AsRef<str>>(
        &mut self,
        qtype: QType,
        instructions: &str,
        options: &[S],
        state: &str,
    ) -> Result<Vec<f32>> {
        let t0 = timer::now();
        let p = self.question(qtype, instructions, options, state)?.0;
        self.last_micros = timer::since(t0);
        Ok(p)
    }

    /// [`Engine::decide_raw`] for one frame of a game stream (SPEC-S3 §7): with a
    /// `temporal_carry` model, the state encoder of a new frame starts from the final latent of
    /// the stream's previous frame when their state token ids match (else cold), so a stream
    /// needs one id per game session and [`Engine::reset_stream`] on episode reset. Questions
    /// with the same state text belong to the same frame (the encoding is shared). Models without
    /// `temporal_carry` (and s1) answer exactly like [`Engine::decide_raw`].
    pub fn decide_raw_stream<S: AsRef<str>>(
        &mut self,
        stream: u64,
        qtype: QType,
        instructions: &str,
        options: &[S],
        state: &str,
    ) -> Result<Vec<f32>> {
        let t0 = timer::now();
        let p = match &mut self.inner {
            Inner::S2(e) => {
                let o = e.question_stream(stream, qtype, instructions, options, state)?;
                self.last = Some(DecisionStats {
                    exit: Some(o.exit),
                    layers_used: Some(o.layers_used),
                    n_exits: e.model().cfg.exits.len(),
                    static_hit: o.static_hit,
                    state_hit: o.state_hit,
                    tokens: o.tokens,
                });
                o.probs
            }
            Inner::S1(_) | Inner::Iso(_) => self.question(qtype, instructions, options, state)?.0,
        };
        self.last_micros = timer::since(t0);
        Ok(p)
    }

    /// Forget the frames of a stream (see [`Engine::decide_raw_stream`]): its next frame starts
    /// cold.
    pub fn reset_stream(&mut self, stream: u64) {
        if let Inner::S2(e) = &mut self.inner {
            e.reset_stream(stream);
        }
    }

    /// Probabilities at every exit without early exit (s2); s1 has a single "exit".
    pub fn probs_per_exit<S: AsRef<str>>(
        &mut self,
        qtype: QType,
        instructions: &str,
        options: &[S],
        state: &str,
    ) -> Result<Vec<Vec<f32>>> {
        match &mut self.inner {
            Inner::S1(_) | Inner::Iso(_) => Ok(vec![self.decide_raw(
                qtype,
                instructions,
                options,
                state,
            )?]),
            Inner::S2(e) => e.question_all_exits(qtype, instructions, options, state),
        }
    }

    /// Answer every question of a request (s2: the state is encoded once for all questions).
    pub fn decide(&mut self, req: &DecisionRequest) -> Result<DecisionResponse> {
        let t0 = timer::now();
        let mut answers = Vec::with_capacity(req.questions.len());
        let mut exits = Vec::with_capacity(req.questions.len());
        let mut input_tokens = 0;
        for q in &req.questions {
            let qtype = q.kind.qtype();
            let result = match &mut self.inner {
                // Definitions only: the option keys never reach an iso model.
                Inner::Iso(e) => {
                    let noul = e.model().cfg.noul_definitions(e.lang());
                    let defs = crate::iso::definitions(&q.kind, (noul.0, noul.1));
                    e.question_defs(qtype, &q.instructions, &defs, &req.state)
                        .map(|o| {
                            self.last = Some(DecisionStats {
                                state_hit: o.state_hit,
                                tokens: o.tokens,
                                ..Default::default()
                            });
                            (o.probs, o.logits)
                        })
                }
                _ => {
                    let texts = prompt::render_options(&q.kind);
                    self.question(qtype, &q.instructions, &texts, &req.state)
                }
            };
            let (p, _) = result.map_err(|e| match e {
                Error::InvalidRequest(m) => {
                    Error::InvalidRequest(format!("question {:?}: {m}", q.id))
                }
                e => e,
            })?;
            let stats = self.last.as_ref().expect("recorded by question()");
            input_tokens += stats.tokens;
            exits.push(match (stats.exit, stats.layers_used) {
                (Some(exit), Some(layers_used)) => Some(ExitInfo { exit, layers_used }),
                _ => None,
            });
            answers.push((q.id.clone(), answer_of(&q.kind, &p)));
        }
        self.last_micros = timer::since(t0);
        Ok(DecisionResponse {
            model: self.name().to_string(),
            answers,
            exits,
            input_tokens,
            micros: self.last_micros,
        })
    }

    /// Parse a JSON request, answer it and serialize the response.
    #[cfg(feature = "json")]
    pub fn decide_json(&mut self, req: &str) -> Result<String> {
        let r = DecisionRequest::from_json(req)?;
        Ok(self.decide(&r)?.to_json())
    }
}

/// Optional wall-clock timing (feature `timing`).
mod timer {
    #[cfg(all(feature = "timing", not(target_arch = "wasm32")))]
    pub type Stamp = Option<std::time::Instant>;
    #[cfg(not(all(feature = "timing", not(target_arch = "wasm32"))))]
    pub type Stamp = Option<f64>;

    #[cfg(all(feature = "timing", not(target_arch = "wasm32")))]
    pub fn now() -> Stamp {
        Some(std::time::Instant::now())
    }

    #[cfg(all(feature = "timing", not(target_arch = "wasm32")))]
    pub fn since(t0: Stamp) -> Option<u64> {
        t0.map(|t| t.elapsed().as_micros() as u64)
    }

    #[cfg(all(feature = "timing", target_arch = "wasm32", feature = "wasm"))]
    fn perf_now() -> Option<f64> {
        use wasm_bindgen::JsCast;
        let perf = js_sys::Reflect::get(&js_sys::global(), &"performance".into()).ok()?;
        let now = js_sys::Reflect::get(&perf, &"now".into())
            .ok()?
            .dyn_into::<js_sys::Function>()
            .ok()?;
        now.call0(&perf).ok()?.as_f64()
    }

    #[cfg(all(feature = "timing", target_arch = "wasm32", feature = "wasm"))]
    pub fn now() -> Stamp {
        perf_now()
    }

    #[cfg(all(feature = "timing", target_arch = "wasm32", feature = "wasm"))]
    pub fn since(t0: Stamp) -> Option<u64> {
        Some(((perf_now()? - t0?) * 1000.0).max(0.0) as u64)
    }

    #[cfg(not(any(
        all(feature = "timing", not(target_arch = "wasm32")),
        all(feature = "timing", target_arch = "wasm32", feature = "wasm")
    )))]
    pub fn now() -> Stamp {
        None
    }

    #[cfg(not(any(
        all(feature = "timing", not(target_arch = "wasm32")),
        all(feature = "timing", target_arch = "wasm32", feature = "wasm")
    )))]
    pub fn since(_t0: Stamp) -> Option<u64> {
        None
    }
}
