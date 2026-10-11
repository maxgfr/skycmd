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
//! * **No allocation per frame.** A state miss reuses the buffers of the evicted state entry
//!   (token ids, hidden states, exit keys / values).

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
    /// Weights as int8 and activations quantized to int8 in two passes (`x ≈ s (q + e / 254)`,
    /// ~16-bit): integer dot products (`sdot` on aarch64, `i32x4.relaxed_dot_i8x16_i7x16_add` on
    /// wasm32 built with `+relaxed-simd`). 1.3-1.6x faster products than f32 on aarch64; results
    /// move by up to ~5e-4 on the golden models (opt-in: the default keeps the f32 path).
    /// Elsewhere a portable (slow) integer fallback runs.
    Int8,
    /// SPEC-S3 §8 lookup tables (T-MAC / bitnet.cpp TL style): 2-bit weights stay in memory,
    /// int8 activations with one scale per row, 16-entry subset-sum tables per group of 4 inputs,
    /// `vqtbl1q_u8` / `i8x16.swizzle` lookups, int16 accumulation. Two activation passes (int8 +
    /// residual, ~2e-4 deviation like `Int8`): a single pass deviates by ~4e-2 on the golden
    /// models. Measured slower than the f32 path on aarch64 and wasm32 for these cache-resident
    /// matrices (BENCH.md, "Inference"), so `Auto` never picks it. Matrices with more than 512
    /// inputs stay packed.
    Lut,
}

impl TernaryCompute {
    /// The f32 path everywhere: the fastest exact path measured with `sky-bench --kernels` and
    /// `bench/kernels-node.mjs` (BENCH.md, "Inference"). On the ladder matrices (cache-resident,
    /// 2-20 rows) the f32 panel kernel beats the packed add/sub kernels (three vector ops per 4
    /// weights) and the lookup tables of [`TernaryCompute::Lut`] (two activation passes) by
    /// 2-3x on aarch64 and wasm32 (SIMD128 and relaxed SIMD). Only [`TernaryCompute::Int8`] is
    /// faster (aarch64 `sdot`, 1.3-1.6x per product), at ~2e-4 deviation, hence opt-in. x86_64
    /// is not measured.
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

/// A stream of frames with a temporal carry (SPEC-S3 §7): the encoding of the current frame and
/// the final latent of the previous one.
struct Stream {
    id: u64,
    /// State text of the current frame (`has_cur`).
    key: String,
    enc: StateEnc,
    has_cur: bool,
    prev_ids: Vec<u32>,
    prev_h: Vec<f32>,
    has_prev: bool,
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
    /// An evicted state entry kept for its buffers.
    spare: Option<(String, StateEnc)>,
    streams: Vec<Stream>,
    bpe: BpeCache,
    ids: Vec<u32>,
    values: Vec<f32>,
    has_value: Vec<bool>,
    logits: Vec<f32>,
}

impl S2Engine {
    pub fn new(model: S2Model, tok: Tokenizer, opts: S2Options) -> Result<Self> {
        let mut model = model;
        model.prepare_compute(opts.ternary);
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
            spare: None,
            streams: Vec::new(),
            ids: Vec::new(),
            values: Vec::new(),
            has_value: Vec::new(),
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
            fuse: self.model.cfg.fuse_numbers,
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
        let (ids, values, _) = self.state_stream(state);
        (ids, values)
    }

    /// The state stream tokens, values and `has_value` flags (SPEC-S3 §1).
    pub fn state_stream(&mut self, state: &str) -> (Vec<u32>, Vec<f32>, Vec<bool>) {
        let pt = self.pt();
        let (mut ids, mut values, mut has_value) = (Vec::new(), Vec::new(), Vec::new());
        prompt::build_state(
            &self.tok,
            &pt,
            state,
            self.model.cfg.max_state_tokens,
            &mut ids,
            &mut values,
            &mut has_value,
            None,
        );
        (ids, values, has_value)
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
            fuse: self.model.cfg.fuse_numbers,
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
        let mut enc = self.model.encode_static(&tokens, &mut self.ws)?;
        self.model.prepare_fusion(&mut enc, qtype.index())?;
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

    /// Bring the state encoding of `state` to the front of the cache (embedding it on a miss,
    /// into the buffers of the evicted entry).
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
            fuse: self.model.cfg.fuse_numbers,
        };
        prompt::build_state(
            &self.tok,
            &pt,
            state,
            self.model.cfg.max_state_tokens,
            &mut self.ids,
            &mut self.values,
            &mut self.has_value,
            Some(&mut self.bpe),
        );
        let cap = self.opts.state_cache.max(1);
        let mut entry = if self.states.len() >= cap {
            self.states.pop().expect("non-empty")
        } else {
            self.spare
                .take()
                .unwrap_or_else(|| (String::new(), self.model.state_buffers()))
        };
        self.states.truncate(cap - 1);
        if let Err(e) =
            self.model
                .state_reset(&mut entry.1, &self.ids, &self.values, &self.has_value)
        {
            self.spare = Some(entry);
            return Err(e);
        }
        entry.0.clear();
        entry.0.push_str(state);
        self.states.insert(0, entry);
        Ok(false)
    }

    /// Scores of exit `e` for the front static entry and the front state entry (or the current
    /// frame of stream `stream`), into `self.logits`.
    fn run_exit(&mut self, qtype: QType, e: usize, stream: Option<usize>) -> Result<()> {
        let st = &self.statics[0].enc;
        debug_assert!(st
            .fusion0
            .as_ref()
            .is_some_and(|f| f.qtype == qtype.index()));
        let state = match stream {
            Some(i) => &mut self.streams[i].enc,
            None => &mut self.states[0].1,
        };
        let s = state.len;
        let kv = self.model.exit_kv(state, e, &mut self.ws);
        self.model
            .fusion_cached(st, kv, s, &mut self.ws, &mut self.logits)
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
        let mut out = self.answer(qtype, options.len(), None)?;
        out.static_hit = static_hit;
        out.state_hit = state_hit;
        if self.opts.static_cache == 0 {
            self.statics.clear();
        }
        if self.opts.state_cache == 0 {
            self.spare = self.states.pop();
            self.states.clear();
        }
        Ok(out)
    }

    /// The early-exit loop on the prepared static entry and state (front state entry or stream).
    fn answer(&mut self, qtype: QType, n: usize, stream: Option<usize>) -> Result<S2Outcome> {
        let tau = self.exit_threshold(qtype);
        let n_exits = self.model.cfg.exits.len();
        let first = if tau >= 1.0 { n_exits - 1 } else { 0 };
        let mut probs = Vec::new();
        let mut exit = n_exits - 1;
        for e in first..n_exits {
            self.run_exit(qtype, e, stream)?;
            probs = probs_from_logits(&self.logits, self.temperature(e, qtype, n));
            let top = probs.iter().fold(0.0f32, |a, &b| a.max(b));
            if top >= tau || e + 1 == n_exits {
                exit = e;
                break;
            }
        }
        let s = match stream {
            Some(i) => self.streams[i].enc.len,
            None => self.states[0].1.len,
        };
        Ok(S2Outcome {
            probs,
            logits: self.logits.clone(),
            exit,
            layers_used: self.model.cfg.exits[exit],
            static_hit: false,
            state_hit: false,
            tokens: self.statics[0].enc.n_tokens + s,
        })
    }

    /// Answer one question of frame `state` in stream `stream` (SPEC-S3 §7). Each stream keeps
    /// the encoding of its current frame (shared by every question on the same state text) and
    /// the final latent of the previous frame: a new frame whose state token ids equal the
    /// previous frame's starts from `embed + carry_gate ⊙ LN₀(previous latent)`, any other frame
    /// starts cold. The previous latent is the output of the last block applied to that frame
    /// (the deepest exit any of its questions reached). Without `temporal_carry` in the model
    /// this is [`S2Engine::question`].
    pub fn question_stream<S: AsRef<str>>(
        &mut self,
        stream: u64,
        qtype: QType,
        instructions: &str,
        options: &[S],
        state: &str,
    ) -> Result<S2Outcome> {
        if !self.model.cfg.temporal_carry {
            return self.question(qtype, instructions, options, state);
        }
        let static_hit = self.prepare_static(qtype, instructions, options)?;
        let i = match self.streams.iter().position(|s| s.id == stream) {
            Some(i) => i,
            None => {
                self.streams.push(Stream {
                    id: stream,
                    key: String::new(),
                    enc: self.model.state_buffers(),
                    has_cur: false,
                    prev_ids: Vec::new(),
                    prev_h: Vec::new(),
                    has_prev: false,
                });
                self.streams.len() - 1
            }
        };
        let d = self.model.cfg.hidden_size;
        let state_hit = self.streams[i].has_cur && self.streams[i].key == state;
        if !state_hit {
            let pt = S2PromptTokens {
                cls: self.cls,
                sep: self.sep,
                mask: self.mask,
                num: self.num,
                mask_text: &self.mask_text,
                fuse: self.model.cfg.fuse_numbers,
            };
            prompt::build_state(
                &self.tok,
                &pt,
                state,
                self.model.cfg.max_state_tokens,
                &mut self.ids,
                &mut self.values,
                &mut self.has_value,
                Some(&mut self.bpe),
            );
            let st = &mut self.streams[i];
            // the current frame becomes the previous one
            st.has_prev = st.has_cur && st.enc.layers_done > 0;
            if st.has_prev {
                let n = st.enc.len;
                st.prev_ids.clear();
                st.prev_ids.extend_from_slice(&st.enc.ids[..n]);
                st.prev_h.clear();
                st.prev_h.extend_from_slice(&st.enc.x[..n * d]);
            }
            st.has_cur = false;
            self.model
                .state_reset(&mut st.enc, &self.ids, &self.values, &self.has_value)?;
            if st.has_prev && st.prev_ids == self.ids {
                self.model
                    .state_carry(&mut st.enc, &st.prev_h, &mut self.ws);
            }
            st.key.clear();
            st.key.push_str(state);
            st.has_cur = true;
        }
        let mut out = self.answer(qtype, options.len(), Some(i))?;
        out.static_hit = static_hit;
        out.state_hit = state_hit;
        if self.opts.static_cache == 0 {
            self.statics.clear();
        }
        Ok(out)
    }

    /// Forget the frames of stream `stream`: its next frame starts cold.
    pub fn reset_stream(&mut self, stream: u64) {
        self.streams.retain(|s| s.id != stream);
    }

    /// Whether the current frame of stream `stream` started from a carried latent.
    pub fn stream_carried(&self, stream: u64) -> bool {
        self.streams
            .iter()
            .any(|s| s.id == stream && s.has_cur && s.enc.carried)
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
            self.run_exit(qtype, e, None)?;
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
