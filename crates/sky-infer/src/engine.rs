//! High-level engine: tokenizer + model + prompt + readout, with caches for the game loop
//! (the question and options are the same every frame, only the state changes).

use sky_gguf::Gguf;

use crate::api::{Answer, DecisionRequest, DecisionResponse};
use crate::kernels::ActQuant;
use crate::model::{Model, QuantizeSpec, Workspace};
use crate::prompt::{self, Prefix, PromptTokens, QType, QuestionKind};
use crate::tokenizer::{BpeCache, Tokenizer};
use crate::{Error, Result};

/// Loading options.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EngineOptions {
    /// Convert these parts to Q8_0 after loading.
    pub quantize: Option<QuantizeSpec>,
    /// For Q8_0 matrices (quantized at load or read from GGUF): run the built-in calibration
    /// prompts and compute the outlier input channels of every matrix in f32 (recommended, see
    /// [`crate::quant`]).
    pub calibrate: bool,
    /// Convert every Q8_0 matrix to f32 after loading (reference path).
    pub dequantize: bool,
    /// Activation quantization of the Q8_0 products (accuracy / speed trade-off).
    pub act_quant: ActQuant,
    /// Number of (question, options) prefixes kept tokenized.
    pub prefix_cache: usize,
    /// Number of BPE words kept in the word cache.
    pub bpe_cache: usize,
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
        }
    }
}

struct CachedPrefix {
    qtype: QType,
    instructions: String,
    options: Vec<String>,
    prefix: Prefix,
}

/// Inference engine for one model. Not thread-safe by design (`&mut self`): it reuses
/// pre-allocated buffers so that a decision does not allocate in the forward pass.
pub struct Engine {
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
    last_micros: Option<u64>,
}

/// Summary of a loaded model.
#[derive(Clone, Debug, PartialEq)]
pub struct ModelInfo {
    pub name: String,
    pub n_params: usize,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub intermediate_size: usize,
    pub head_layers: usize,
    pub max_len: usize,
    pub head_max_len: usize,
    pub q8: bool,
    pub temperatures: Vec<(String, f32)>,
}

impl ModelInfo {
    pub fn to_json(&self) -> String {
        let temps: Vec<String> = self
            .temperatures
            .iter()
            .map(|(k, v)| format!("{}:{}", crate::json::escape(k), v))
            .collect();
        format!(
            "{{\"name\":{},\"n_params\":{},\"vocab_size\":{},\"hidden_size\":{},\"num_layers\":{},\"num_heads\":{},\
             \"intermediate_size\":{},\"head_layers\":{},\"max_len\":{},\"head_max_len\":{},\"q8\":{},\"temperatures\":{{{}}}}}",
            crate::json::escape(&self.name),
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
            temps.join(",")
        )
    }
}

impl Engine {
    /// Build an engine from a model and its tokenizer.
    pub fn new(model: Model, tok: Tokenizer, opts: EngineOptions) -> Result<Self> {
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
        Ok(Engine {
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
            last_micros: None,
        })
    }

    /// Load a GGUF file (llama.cpp `modern-bert` decision model with its tokenizer).
    pub fn from_gguf(bytes: &[u8]) -> Result<Self> {
        Self::from_gguf_with(bytes, EngineOptions::default())
    }

    pub fn from_gguf_with(bytes: &[u8], opts: EngineOptions) -> Result<Self> {
        let g = Gguf::parse(bytes)?;
        let model = Model::from_gguf(&g)?;
        let tok = Tokenizer::from_gguf(&g)?;
        Self::new(model, tok, opts)
    }

    /// Load a Laya checkpoint directory: `model.safetensors`, `encoder/config.json`,
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

    pub fn model(&self) -> &Model {
        &self.model
    }

    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tok
    }

    pub fn n_params(&self) -> usize {
        self.model.n_params()
    }

    pub fn model_info(&mut self) -> ModelInfo {
        let c = &self.model.cfg;
        let info = ModelInfo {
            name: c.name.clone(),
            n_params: self.model.n_params(),
            vocab_size: c.vocab_size,
            hidden_size: c.hidden_size,
            num_layers: c.num_layers,
            num_heads: c.num_heads,
            intermediate_size: c.intermediate_size,
            head_layers: c.head_layers,
            max_len: c.max_len,
            head_max_len: c.head_max_len,
            q8: false,
            temperatures: c.temperatures.clone(),
        };
        ModelInfo {
            q8: self.model.has_q8(),
            ..info
        }
    }

    /// Wall time of the last decision in microseconds (feature `timing`).
    pub fn last_micros(&self) -> Option<u64> {
        self.last_micros
    }

    /// Temperature applied to a question type with `n` options.
    pub fn temperature(&self, qtype: QType, n: usize) -> f32 {
        self.model
            .cfg
            .temperature(qtype.name(), prompt::temp_bucket(n))
    }

    /// Token ids and marker positions of a question (same as `rl_common.build_sequence`).
    pub fn build_sequence(
        &mut self,
        qtype: QType,
        instructions: &str,
        options: &[&str],
        state: &str,
    ) -> (Vec<u32>, Vec<usize>) {
        self.prepare(qtype, instructions, options, state);
        (self.ids.clone(), self.markers.clone())
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

    /// Raw scores (before temperature) of rendered option texts.
    pub fn scores_raw<S: AsRef<str>>(
        &mut self,
        qtype: QType,
        instructions: &str,
        options: &[S],
        state: &str,
    ) -> Result<Vec<f32>> {
        self.prepare(qtype, instructions, options, state);
        self.run(qtype, options.len())?;
        Ok(self.logits.clone())
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
        self.prepare(qtype, instructions, options, state);
        self.run(qtype, options.len())?;
        let probs = prompt::probs_from_logits(&self.logits, self.temperature(qtype, options.len()));
        self.last_micros = timer::since(t0);
        Ok(probs)
    }

    /// Answer every question of a request.
    pub fn decide(&mut self, req: &DecisionRequest) -> Result<DecisionResponse> {
        let t0 = timer::now();
        let mut answers = Vec::with_capacity(req.questions.len());
        let mut input_tokens = 0;
        for q in &req.questions {
            let qtype = q.kind.qtype();
            let texts = prompt::render_options(&q.kind);
            self.prepare(qtype, &q.instructions, &texts, &req.state);
            self.run(qtype, texts.len()).map_err(|e| match e {
                Error::InvalidRequest(m) => {
                    Error::InvalidRequest(format!("question {:?}: {m}", q.id))
                }
                e => e,
            })?;
            input_tokens += self.ids.len();
            let p = prompt::probs_from_logits(&self.logits, self.temperature(qtype, texts.len()));
            let answer = match &q.kind {
                QuestionKind::Choice(opts) => Answer::Choice {
                    choice: opts[prompt::argmax(&p)].0.clone(),
                    probabilities: opts
                        .iter()
                        .map(|(k, _)| k.clone())
                        .zip(p.iter().copied())
                        .collect(),
                    confidence: prompt::confidence_choice(&p),
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
                    confidence: prompt::confidence_score(&p),
                },
                QuestionKind::Noul { .. } => Answer::Noul { noul: p[1] },
            };
            answers.push((q.id.clone(), answer));
        }
        self.last_micros = timer::since(t0);
        Ok(DecisionResponse {
            model: self.model.cfg.name.clone(),
            answers,
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
