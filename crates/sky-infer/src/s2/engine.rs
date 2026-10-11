//! s2 decision engine: caches and the early-exit loop (SPEC-S2 §3 and §6).
//!
//! * **Static cache.** The static encoding of `(qtype, instructions, options)` (marker vectors and
//!   the question keys/values of every fusion block) is kept in an LRU of [`S2Options::static_cache`]
//!   entries (16 by default): in a game loop the question never changes, so a frame only encodes
//!   the state.
//! * **State cache.** The state encoding is question-agnostic: it is keyed by the state text and
//!   reused by every question asked about the same state (multi-question frames), including the
//!   exits already computed.
//! * **Early exit.** Exits are tried in order; the first one whose calibrated top probability
//!   reaches τ answers (τ = 1 runs the last exit only).

use super::model::{S2Model, S2Workspace, StateEnc, StaticEnc, StaticTokens};
use super::prompt::{self, S2PromptTokens};
use crate::prompt::{probs_from_logits, temp_bucket, QType};
use crate::tokenizer::{BpeCache, Tokenizer};
use crate::{Error, Result};

/// How ternary matrices are computed.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum TernaryCompute {
    /// The faster of the two on this platform (see [`TernaryCompute::resolve`]).
    #[default]
    Auto,
    /// Packed 2-bit weights with add/sub kernels (smallest memory).
    Packed,
    /// Weights dequantized to f32 at load (the file stays small).
    F32,
}

impl TernaryCompute {
    /// The f32 path everywhere. Measured with `sky-bench --kernels` and the wasm bench (BENCH.md,
    /// "Inference"): on the s2 ladder matrices the f32 FMA path is 2-3x faster than the add/sub
    /// kernels on aarch64 (NEON) and about 1.6x faster per decision on wasm32 (SIMD128), since
    /// the weights fit in cache and each group of 4 weights costs three vector ops (XOR, AND,
    /// ADD) instead of one FMA. x86_64 (SSE2) has the same op count and is not measured.
    pub fn resolve(self) -> TernaryCompute {
        match self {
            TernaryCompute::Auto => TernaryCompute::F32,
            other => other,
        }
    }
}

/// s2 engine options.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct S2Options {
    /// Overrides τ of every question type; 1.0 means full depth.
    pub exit_threshold: Option<f32>,
    /// Entries of the static LRU cache (0 disables it: the static stream is encoded each time).
    pub static_cache: usize,
    /// Entries of the state cache (0 disables it).
    pub state_cache: usize,
    pub ternary: TernaryCompute,
    /// BPE words kept in the word cache.
    pub bpe_cache: usize,
}

impl Default for S2Options {
    fn default() -> Self {
        S2Options {
            exit_threshold: None,
            static_cache: 16,
            state_cache: 4,
            ternary: TernaryCompute::Auto,
            bpe_cache: 4096,
        }
    }
}

/// Result of one question.
#[derive(Clone, Debug, PartialEq)]
pub struct S2Outcome {
    /// Probabilities at the exit that answered.
    pub probs: Vec<f32>,
    /// Raw scores at that exit (before the temperature).
    pub logits: Vec<f32>,
    /// Index of the exit that answered.
    pub exit: usize,
    /// State-encoder blocks used (`exits[exit]`).
    pub layers_used: usize,
    pub static_hit: bool,
    pub state_hit: bool,
    /// Static plus state tokens of the question.
    pub tokens: usize,
}

struct CachedStatic {
    qtype: QType,
    instructions: String,
    options: Vec<String>,
    tokens: StaticTokens,
    enc: StaticEnc,
}

/// Inference engine for one s2 model (not thread-safe by design: `&mut self` reuses buffers).
pub struct S2Engine {
    model: S2Model,
    tok: Tokenizer,
    cls: u32,
    sep: u32,
    mask: u32,
    num: u32,
    mask_text: String,
    opts: S2Options,
    ws: S2Workspace,
    statics: Vec<CachedStatic>,
    states: Vec<(String, StateEnc)>,
    bpe: BpeCache,
    ids: Vec<u32>,
    values: Vec<f32>,
    logits: Vec<f32>,
}

impl S2Engine {
    pub fn new(model: S2Model, tok: Tokenizer, opts: S2Options) -> Result<Self> {
        let mut model = model;
        if opts.ternary.resolve() == TernaryCompute::F32 {
            model.dequantize_ternary();
        }
        let sp = tok.special();
        let missing = |n: &str| Error::Missing(format!("{n} token in the tokenizer"));
        let mask = sp.mask.ok_or_else(|| missing("mask"))?;
        let mask_text = tok
            .id_to_token(mask)
            .ok_or_else(|| missing("mask"))?
            .to_string();
        let num = model.cfg.num_token_id;
        if tok.id_to_token(num) != Some(crate::num::NUM_TOKEN) {
            return Err(Error::Format(format!(
                "the tokenizer has no {} token at id {num}",
                crate::num::NUM_TOKEN
            )));
        }
        Ok(S2Engine {
            cls: sp.cls.ok_or_else(|| missing("cls"))?,
            sep: sp.sep.ok_or_else(|| missing("sep"))?,
            mask,
            num,
            mask_text,
            bpe: BpeCache::new(opts.bpe_cache),
            opts,
            ws: S2Workspace::default(),
            statics: Vec::new(),
            states: Vec::new(),
            ids: Vec::new(),
            values: Vec::new(),
            logits: Vec::new(),
            model,
            tok,
        })
    }

    pub fn model(&self) -> &S2Model {
        &self.model
    }

    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tok
    }

    pub fn options(&self) -> &S2Options {
        &self.opts
    }

    /// Override τ for every question type (`None`: the calibrated τ of the file).
    pub fn set_exit_threshold(&mut self, tau: Option<f32>) {
        self.opts.exit_threshold = tau;
    }

    /// Drop the cached static and state encodings.
    pub fn clear_caches(&mut self) {
        self.statics.clear();
        self.states.clear();
    }

    fn pt(&self) -> S2PromptTokens<'_> {
        S2PromptTokens {
            cls: self.cls,
            sep: self.sep,
            mask: self.mask,
            num: self.num,
            mask_text: &self.mask_text,
        }
    }

    /// τ of a question type.
    pub fn exit_threshold(&self, qtype: QType) -> f32 {
        self.opts
            .exit_threshold
            .unwrap_or(self.model.cfg.exit_thresholds[qtype.index()])
    }

    /// Temperature of an exit for a question type with `n` options.
    pub fn temperature(&self, exit: usize, qtype: QType, n: usize) -> f32 {
        self.model
            .cfg
            .temperature(exit, qtype.name(), temp_bucket(n))
    }

    /// The static stream tokens of a question (for debugging prompts).
    pub fn static_tokens(
        &mut self,
        qtype: QType,
        instructions: &str,
        options: &[&str],
    ) -> StaticTokens {
        let pt = self.pt();
        prompt::build_static(
            &self.tok,
            &pt,
            qtype,
            instructions,
            options,
            self.model.cfg.max_head_tokens,
            None,
        )
    }

    /// The state stream tokens and values (for debugging prompts).
    pub fn state_tokens(&mut self, state: &str) -> (Vec<u32>, Vec<f32>) {
        let pt = self.pt();
        let (mut ids, mut values) = (Vec::new(), Vec::new());
        prompt::build_state(
            &self.tok,
            &pt,
            state,
            self.model.cfg.max_state_tokens,
            &mut ids,
            &mut values,
            None,
        );
        (ids, values)
    }

    /// Bring the static encoding of a question to the front of the LRU (encoding it on a miss).
    /// Returns whether it was cached.
    fn prepare_static<S: AsRef<str>>(
        &mut self,
        qtype: QType,
        instructions: &str,
        options: &[S],
    ) -> Result<bool> {
        if options.is_empty() {
            return Err(Error::InvalidRequest(
                "a question needs at least one option".into(),
            ));
        }
        let hit = self.statics.iter().position(|c| {
            c.qtype == qtype
                && c.instructions == instructions
                && c.options.len() == options.len()
                && c.options.iter().zip(options).all(|(a, b)| a == b.as_ref())
        });
        if let Some(i) = hit {
            if i > 0 {
                let c = self.statics.remove(i);
                self.statics.insert(0, c);
            }
            return Ok(true);
        }
        let opts: Vec<&str> = options.iter().map(|o| o.as_ref()).collect();
        let pt = S2PromptTokens {
            cls: self.cls,
            sep: self.sep,
            mask: self.mask,
            num: self.num,
            mask_text: &self.mask_text,
        };
        let tokens = prompt::build_static(
            &self.tok,
            &pt,
            qtype,
            instructions,
            &opts,
            self.model.cfg.max_head_tokens,
            Some(&mut self.bpe),
        );
        let enc = self.model.encode_static(&tokens, &mut self.ws)?;
        let cap = self.opts.static_cache.max(1);
        if self.statics.len() >= cap {
            self.statics.truncate(cap - 1);
        }
        self.statics.insert(
            0,
            CachedStatic {
                qtype,
                instructions: instructions.to_string(),
                options: opts.iter().map(|s| s.to_string()).collect(),
                tokens,
                enc,
            },
        );
        Ok(false)
    }

    /// Bring the state encoding of `state` to the front of the cache (embedding it on a miss).
    fn prepare_state(&mut self, state: &str) -> Result<bool> {
        if let Some(i) = self.states.iter().position(|(k, _)| k == state) {
            if i > 0 {
                let c = self.states.remove(i);
                self.states.insert(0, c);
            }
            return Ok(true);
        }
        let pt = S2PromptTokens {
            cls: self.cls,
            sep: self.sep,
            mask: self.mask,
            num: self.num,
            mask_text: &self.mask_text,
        };
        prompt::build_state(
            &self.tok,
            &pt,
            state,
            self.model.cfg.max_state_tokens,
            &mut self.ids,
            &mut self.values,
            Some(&mut self.bpe),
        );
        let enc = self.model.state_start(&self.ids, &self.values)?;
        let cap = self.opts.state_cache.max(1);
        if self.states.len() >= cap {
            self.states.truncate(cap - 1);
        }
        self.states.insert(0, (state.to_string(), enc));
        Ok(false)
    }

    /// Scores of exit `e` for the front static and state entries, into `self.logits`.
    fn run_exit(&mut self, qtype: QType, e: usize) -> Result<()> {
        let st = &self.statics[0].enc;
        let state = &mut self.states[0].1;
        let s = state.len;
        let kv = self.model.exit_kv(state, e, &mut self.ws);
        self.model.fusion(
            &st.markers,
            st.n_options,
            qtype.index(),
            kv,
            s,
            &st.kv,
            st.q_len,
            &mut self.ws,
            &mut self.logits,
        )
    }

    /// Answer one question with early exit.
    pub fn question<S: AsRef<str>>(
        &mut self,
        qtype: QType,
        instructions: &str,
        options: &[S],
        state: &str,
    ) -> Result<S2Outcome> {
        let static_hit = self.prepare_static(qtype, instructions, options)?;
        let state_hit = self.prepare_state(state)?;
        let tau = self.exit_threshold(qtype);
        let n = options.len();
        let n_exits = self.model.cfg.exits.len();
        let first = if tau >= 1.0 { n_exits - 1 } else { 0 };
        let mut probs = Vec::new();
        let mut exit = n_exits - 1;
        for e in first..n_exits {
            self.run_exit(qtype, e)?;
            probs = probs_from_logits(&self.logits, self.temperature(e, qtype, n));
            let top = probs.iter().fold(0.0f32, |a, &b| a.max(b));
            if top >= tau || e + 1 == n_exits {
                exit = e;
                break;
            }
        }
        let tokens = self.statics[0].enc.n_tokens + self.states[0].1.len;
        if self.opts.static_cache == 0 {
            self.statics.clear();
        }
        if self.opts.state_cache == 0 {
            self.states.clear();
        }
        Ok(S2Outcome {
            probs,
            logits: self.logits.clone(),
            exit,
            layers_used: self.model.cfg.exits[exit],
            static_hit,
            state_hit,
            tokens,
        })
    }

    /// Probabilities at every exit (full depth, no early exit), for evaluation and tests.
    pub fn question_all_exits<S: AsRef<str>>(
        &mut self,
        qtype: QType,
        instructions: &str,
        options: &[S],
        state: &str,
    ) -> Result<Vec<Vec<f32>>> {
        self.prepare_static(qtype, instructions, options)?;
        self.prepare_state(state)?;
        let n = options.len();
        let mut out = Vec::with_capacity(self.model.cfg.exits.len());
        for e in 0..self.model.cfg.exits.len() {
            self.run_exit(qtype, e)?;
            out.push(probs_from_logits(
                &self.logits,
                self.temperature(e, qtype, n),
            ));
        }
        Ok(out)
    }

    /// Static and state token ids of a question (debugging): `(static ids, marker positions,
    /// state ids)`.
    pub fn sequences(
        &mut self,
        qtype: QType,
        instructions: &str,
        options: &[&str],
        state: &str,
    ) -> (Vec<u32>, Vec<usize>, Vec<u32>) {
        let st = self.static_tokens(qtype, instructions, options);
        let (ids, _) = self.state_tokens(state);
        let markers = st.options.iter().map(|o| o.0).collect();
        (st.ids, markers, ids)
    }

    /// Whether the static stream of this question is cached.
    pub fn is_static_cached<S: AsRef<str>>(
        &self,
        qtype: QType,
        instructions: &str,
        options: &[S],
    ) -> bool {
        self.statics.iter().any(|c| {
            c.qtype == qtype
                && c.instructions == instructions
                && c.options.len() == options.len()
                && c.options.iter().zip(options).all(|(a, b)| a == b.as_ref())
        })
    }

    /// Number of cached static encodings.
    pub fn static_cache_len(&self) -> usize {
        self.statics.len()
    }

    /// Tokens of the cached static stream at the front of the LRU (for tests and debugging).
    pub fn last_static_tokens(&self) -> Option<&StaticTokens> {
        self.statics.first().map(|c| &c.tokens)
    }
}
