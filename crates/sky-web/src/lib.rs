//! `sky-web`: one wasm module for the browser that runs both the skycmd game cores and the
//! System One model, so a full AI frame (state text → tokenize → forward → probabilities →
//! action → safety reflex → physics step) happens inside WASM. JavaScript only renders.
//!
//! ```js
//! import init, { Engine, GameSession, bench } from "./pkg/sky_web.js";
//! await init();
//! const engine = new Engine(new Uint8Array(await (await fetch("s1-pico.gguf")).arrayBuffer())); // s1 or s2
//! const game = new GameSession("flappy", 42);
//! const step = JSON.parse(game.step_ai(engine));   // or game.step_oracle() without a model
//! const view = JSON.parse(game.snapshot());
//! ```
//!
//! The secrets verifier (s6-secrets-verify-pico, [`verify`]):
//!
//! ```js
//! const v = new SecretVerifier(ggufBytes, dictionaryBytes);
//! const { p_real, state_tokens, truncated, micros } = JSON.parse(v.judge(JSON.stringify(record)));
//! const { tau_f1 } = JSON.parse(v.thresholds());   // accepted when p_real >= tau_f1
//! ```
//!
//! The fused tool router (s6-nano v3, [`router`]): the v2 router file and the 4-bit static
//! embedding file, the tool list cached on both parts:
//!
//! ```js
//! const r = new ToolRouter(modelBytes, embeddingBytes);
//! const tools = JSON.stringify(["lights_on: Turns the lights on.", "weather: Gives the weather."]);
//! const { probs, answer, ask, decision, micros } = JSON.parse(r.route("turn on the hall lights", tools, false));
//! ```
//!
//! The `recall` memory probe (s6-sleep3-nano and its control, [`recall`]): the evaluation's episode
//! generator and scoring, played by a model with its memory on or zeroed:
//!
//! ```js
//! const p = new RecallProbe(ggufBytes);
//! p.setMemory("zeroed"); p.setSleepPasses(1);           // or "on"; N = 1, 2, 4
//! const { frames } = JSON.parse(p.start(7000000018));   // what each frame shows, its answer
//! const { probs, choice, correct, micros } = JSON.parse(p.step());
//! const { asks } = JSON.parse(p.runEpisode(7000000038)); // a whole episode: [{frame, delay, bucket, correct}]
//! ```
//!
//! The Rust API ([`Model`], [`Session`], [`session::bench`]) is usable natively; the
//! `#[wasm_bindgen]` types below are thin wrappers that turn errors into JS exceptions.

pub mod clock;
pub mod forecast;
pub mod model;
pub mod recall;
pub mod router;
pub mod session;
pub mod verify;

pub use model::Model;
pub use session::{AiStep, HumanStep, Policy, Session};

use wasm_bindgen::prelude::*;

fn to_json<T: serde::Serialize>(v: &T) -> String {
    serde_json::to_string(v).expect("serializable")
}

/// The question type and the rendered options (at least two) of a JSON decision call.
fn parse_question(
    qtype: &str,
    options_json: &str,
) -> Result<(sky_infer::QType, Vec<String>), String> {
    let qt = sky_infer::QType::from_name(qtype)
        .ok_or_else(|| format!("unknown question type {qtype:?} (choice, score, noul)"))?;
    let options: Vec<String> =
        serde_json::from_str(options_json).map_err(|e| format!("options: {e}"))?;
    if options.len() < 2 {
        return Err("at least two options are needed".into());
    }
    Ok((qt, options))
}

/// A loaded decision model (GGUF s1 or s2: f32, f16, Q8_0 or ternary; computed in f32).
#[wasm_bindgen]
pub struct Engine {
    inner: Model,
}

#[wasm_bindgen]
impl Engine {
    #[wasm_bindgen(constructor)]
    pub fn new(gguf_bytes: &[u8]) -> Result<Engine, JsError> {
        Ok(Engine {
            inner: Model::load(gguf_bytes).map_err(|e| JsError::new(&e))?,
        })
    }

    /// Load with s2 engine options: `static_cache = false` encodes the question and options
    /// every decision (no LRU, for benchmarks); `ternary_packed = true` computes ternary
    /// matrices with the packed add/sub kernels instead of f32. No effect on s1 models.
    #[wasm_bindgen(js_name = withOptions)]
    pub fn with_options(
        gguf_bytes: &[u8],
        static_cache: bool,
        ternary_packed: bool,
    ) -> Result<Engine, JsError> {
        let opts = sky_infer::EngineOptions {
            prefix_cache: if static_cache { 16 } else { 0 },
            ternary: if ternary_packed {
                sky_infer::TernaryCompute::Packed
            } else {
                sky_infer::TernaryCompute::Auto
            },
            ..Default::default()
        };
        Ok(Engine {
            inner: Model::load_with(gguf_bytes, opts).map_err(|e| JsError::new(&e))?,
        })
    }

    /// Model description as JSON (architecture, temperatures, `n_params`, `file_bytes`).
    pub fn info(&mut self) -> String {
        self.inner.info().to_string()
    }

    /// Number of parameters.
    pub fn n_params(&self) -> f64 {
        self.inner.n_params() as f64
    }

    /// Size of the GGUF file in bytes.
    pub fn file_bytes(&self) -> f64 {
        self.inner.file_bytes() as f64
    }

    pub fn name(&self) -> String {
        self.inner.name()
    }

    /// Model family: `s1` (ModernBERT + decision head), `s2` (static cache, early exits) or
    /// `iso` (s5: options isolated by construction, definitions only).
    pub fn family(&self) -> String {
        session::family(self.inner.arch()).into()
    }

    /// iso: language of the generic `noul` definitions (`en` or `fr`). No effect on s1 / s2.
    #[wasm_bindgen(js_name = setLang)]
    pub fn set_lang(&mut self, lang: &str) {
        self.inner.set_lang(lang)
    }

    /// s2: override the early-exit threshold τ of every question type (undefined: the
    /// calibrated values; 1.0: full depth). No effect on s1 models.
    #[wasm_bindgen(js_name = setExitThreshold)]
    pub fn set_exit_threshold(&mut self, tau: Option<f32>) {
        self.inner.set_exit_threshold(tau)
    }

    /// A generic decision (the /live demos): `type` is `choice`, `score` or `noul`,
    /// `options_json` a JSON array of rendered option texts. Returns
    /// `{"probs", "exit", "n_exits", "layers_used", "static_hit", "state_hit", "micros"}` as
    /// JSON (`state_hit`: s2 and iso reused the encoding of the same state).
    pub fn decide(
        &mut self,
        qtype: &str,
        instructions: &str,
        options_json: &str,
        state: &str,
    ) -> Result<String, JsError> {
        self.decide_value(qtype, instructions, options_json, state)
            .map(|v| v.to_string())
            .map_err(|e| JsError::new(&e))
    }

    /// s6-ear (SPEC-S6 §16): a decision about an audio clip, `pcm` 16 kHz mono in `[-1, 1]`
    /// (the last 10.24 s count; resample the microphone to 16 kHz first). Same arguments and
    /// JSON answer as [`Engine::decide`], with `pcm` as the state.
    #[wasm_bindgen(js_name = decideAudio)]
    pub fn decide_audio(
        &mut self,
        qtype: &str,
        instructions: &str,
        options_json: &str,
        pcm: &[f32],
    ) -> Result<String, JsError> {
        self.decide_audio_value(qtype, instructions, options_json, pcm)
            .map(|v| v.to_string())
            .map_err(|e| JsError::new(&e))
    }

    /// True for an s6-ear model (its decisions take audio: [`Engine::decide_audio`]).
    #[wasm_bindgen(js_name = isAudio)]
    pub fn is_audio(&self) -> bool {
        self.inner.is_audio()
    }
}

impl Engine {
    /// [`Engine::decide`] without `JsError` (usable on native targets).
    pub fn decide_value(
        &mut self,
        qtype: &str,
        instructions: &str,
        options_json: &str,
        state: &str,
    ) -> Result<serde_json::Value, String> {
        let (qt, options) = parse_question(qtype, options_json)?;
        let t0 = clock::now_us();
        let probs = self.inner.probs(qt, instructions, &options, state)?;
        let micros = clock::now_us() - t0;
        Ok(self.answer_json(probs, micros))
    }

    /// [`Engine::decide_audio`] without `JsError` (usable on native targets).
    pub fn decide_audio_value(
        &mut self,
        qtype: &str,
        instructions: &str,
        options_json: &str,
        pcm: &[f32],
    ) -> Result<serde_json::Value, String> {
        let (qt, options) = parse_question(qtype, options_json)?;
        let t0 = clock::now_us();
        let probs = self.inner.probs_audio(qt, instructions, &options, pcm)?;
        let micros = clock::now_us() - t0;
        Ok(self.answer_json(probs, micros))
    }

    fn answer_json(&self, probs: Vec<f32>, micros: f64) -> serde_json::Value {
        let st = self.inner.last_stats().cloned().unwrap_or_default();
        serde_json::json!({
            "probs": probs,
            "exit": st.exit,
            "n_exits": st.n_exits,
            "layers_used": st.layers_used,
            "static_hit": st.static_hit,
            "state_hit": st.state_hit,
            "micros": micros,
        })
    }

    /// The underlying model (native use and tests).
    pub fn model_mut(&mut self) -> &mut Model {
        &mut self.inner
    }

    /// Load without going through `JsError` (usable on native targets).
    pub fn load(gguf_bytes: &[u8]) -> Result<Engine, String> {
        Ok(Engine {
            inner: Model::load(gguf_bytes)?,
        })
    }
}

/// The secrets verifier: s6-secrets-verify-pico judging rule candidates at full depth. The
/// candidate JSON is documented in [`verify`].
#[wasm_bindgen]
pub struct SecretVerifier {
    inner: verify::Verifier,
}

#[wasm_bindgen]
impl SecretVerifier {
    /// From the GGUF bytes of the model and the bytes of the word list (`dictionary.txt`).
    #[wasm_bindgen(constructor)]
    pub fn new(gguf_bytes: &[u8], words_bytes: &[u8]) -> Result<SecretVerifier, JsError> {
        Ok(SecretVerifier {
            inner: verify::Verifier::load(gguf_bytes, words_bytes).map_err(|e| JsError::new(&e))?,
        })
    }

    /// One candidate (a JSON object) -> `{p_real, state_tokens, truncated, micros}` as JSON.
    pub fn judge(&mut self, candidate_json: &str) -> Result<String, JsError> {
        self.inner
            .judge(candidate_json)
            .map(|v| v.to_string())
            .map_err(|e| JsError::new(&e))
    }

    /// A JSON array of candidates -> a JSON array of answers, in the same order.
    #[wasm_bindgen(js_name = judgeMany)]
    pub fn judge_many(&mut self, candidates_json: &str) -> Result<String, JsError> {
        self.inner
            .judge_many(candidates_json)
            .map(|v| v.to_string())
            .map_err(|e| JsError::new(&e))
    }

    /// `{tau_f1, tau_f05, tau_f2}` as JSON: the operating thresholds stored in the file.
    pub fn thresholds(&self) -> String {
        self.inner.thresholds().to_string()
    }

    /// The state text the model reads for a candidate (tests, parity).
    pub fn state(&self, candidate_json: &str) -> Result<String, JsError> {
        self.inner
            .state(candidate_json)
            .map_err(|e| JsError::new(&e))
    }

    /// `{name, n_params, file_bytes, window, words}` as JSON.
    pub fn info(&self) -> String {
        self.inner.info().to_string()
    }
}

/// The fused tool router of s6-nano v3: the model part (the v2 router file) and the embedding
/// part (`skycmd-embed` file), `p ∝ (p_model · p_embedding)^β`, "ask" below τ_ask3.
#[wasm_bindgen]
pub struct ToolRouter {
    inner: router::Router,
}

#[wasm_bindgen]
impl ToolRouter {
    /// From the GGUF bytes of the model file and of the embedding file.
    #[wasm_bindgen(constructor)]
    pub fn new(model_bytes: &[u8], embedding_bytes: &[u8]) -> Result<ToolRouter, JsError> {
        Ok(ToolRouter {
            inner: router::Router::load(model_bytes, embedding_bytes)
                .map_err(|e| JsError::new(&e))?,
        })
    }

    /// Route a request against `tools_json`, a JSON array of rendered tool options
    /// `name: description` (*none* is appended). `fast`: the model stops at its first exit.
    /// Returns `{probs, index, answer, none, ask, decision, top, tau_ask3, model, embedding,
    /// fast, exit, layers_used, static_hit, tools_hit, micros, micros_model, micros_embedding}`
    /// as JSON: `probs` over the tools then *none*; `answer` the top option's name or `none`;
    /// `decision` `tool`, `none` or `ask` (the top probability below `tau_ask3`); `model` and
    /// `embedding` the two parts' probabilities; `static_hit` / `tools_hit` whether the tool list
    /// was cached on the model / embedding part.
    pub fn route(
        &mut self,
        request: &str,
        tools_json: &str,
        fast: bool,
    ) -> Result<String, JsError> {
        self.inner
            .route_value(request, tools_json, fast)
            .map(|v| v.to_string())
            .map_err(|e| JsError::new(&e))
    }

    /// The token ids of a text in the embedding part (tests, parity).
    #[wasm_bindgen(js_name = embeddingTokens)]
    pub fn embedding_tokens(&self, text: &str) -> Vec<u32> {
        self.inner.embedding().token_ids(text)
    }

    /// τ_ask3, the ask threshold on the top fused probability.
    #[wasm_bindgen(js_name = tauAsk)]
    pub fn tau_ask(&self) -> f64 {
        self.inner.embedding().tau_ask
    }

    /// `{model, embedding}` as JSON (the model's description; the embedding's name, bytes,
    /// dimension, vocabulary size and fitted values).
    pub fn info(&mut self) -> String {
        self.inner.info().to_string()
    }
}

/// The `recall` memory probe ([`recall::Probe`]): episodes of the evaluation's generator, played
/// frame by frame or whole by an s2 model, with its memory on or zeroed and `N` sleep passes.
#[wasm_bindgen]
pub struct RecallProbe {
    inner: recall::Probe,
}

#[wasm_bindgen]
impl RecallProbe {
    /// From the GGUF bytes of an s2 model, with or without a memory (memory on, `N` = 1).
    #[wasm_bindgen(constructor)]
    pub fn new(gguf_bytes: &[u8]) -> Result<RecallProbe, JsError> {
        Ok(RecallProbe {
            inner: recall::Probe::load(gguf_bytes).map_err(|e| JsError::new(&e))?,
        })
    }

    /// Whether the model keeps a memory across frames (false for the control).
    #[wasm_bindgen(js_name = hasMemory)]
    pub fn has_memory(&self) -> bool {
        self.inner.has_memory()
    }

    /// `on`, or `zeroed`: the memory reset before every frame (a model with a memory only). Takes
    /// effect at the next episode.
    #[wasm_bindgen(js_name = setMemory)]
    pub fn set_memory(&mut self, mode: &str) -> Result<(), JsError> {
        self.inner.set_memory(mode).map_err(|e| JsError::new(&e))
    }

    /// `N`, the passes of each sleep, the wake pass included (1 to 8; 1: no sleep). Takes effect at
    /// the next episode.
    #[wasm_bindgen(js_name = setSleepPasses)]
    pub fn set_sleep_passes(&mut self, n: u32) -> Result<(), JsError> {
        self.inner
            .set_sleep_passes(n as usize)
            .map_err(|e| JsError::new(&e))
    }

    /// `{name, n_params, file_bytes, memory, memory_mode, sleep_passes, sleep_every, frames,
    /// buckets, instructions, options, keys, exit_threshold}` as JSON.
    pub fn info(&mut self) -> String {
        self.inner.info().to_string()
    }

    /// Starts the episode of `seed` (a whole number) from a zero memory. Returns `{seed, frames:
    /// [{frame, state, kind, door, colour, answer, delay, bucket}], memory, memory_mode,
    /// sleep_passes}` as JSON: what each frame shows and its right answer.
    pub fn start(&mut self, seed: f64) -> Result<String, JsError> {
        let seed = recall::seed_from_f64(seed).map_err(|e| JsError::new(&e))?;
        Ok(self.inner.start(seed).to_string())
    }

    /// Plays the next frame of the episode. Returns `{frame, state, kind, door, colour, answer,
    /// delay, bucket, probs, choice, correct, micros, done}` as JSON (`probs` in the order of
    /// `keys`).
    pub fn step(&mut self) -> Result<String, JsError> {
        self.inner
            .step()
            .map(|r| to_json(&r))
            .map_err(|e| JsError::new(&e))
    }

    /// Whether the episode in progress has ended.
    pub fn done(&self) -> bool {
        self.inner.done()
    }

    /// Plays the whole episode of `seed` from a zero memory. Returns `{seed, asks: [{frame, delay,
    /// bucket, correct}], decisions, micros}` as JSON.
    #[wasm_bindgen(js_name = runEpisode)]
    pub fn run_episode(&mut self, seed: f64) -> Result<String, JsError> {
        let seed = recall::seed_from_f64(seed).map_err(|e| JsError::new(&e))?;
        self.inner
            .run_episode(seed)
            .map(|r| to_json(&r))
            .map_err(|e| JsError::new(&e))
    }
}

/// A game (`flappy`, `rescue`, `dodge` or `doom`) stepped by a human, the model or the oracle.
#[wasm_bindgen]
pub struct GameSession {
    inner: Session,
}

#[wasm_bindgen]
impl GameSession {
    #[wasm_bindgen(constructor)]
    /// `seed` selects the sequence of courses (a JS number; episodes are deterministic).
    pub fn new(kind: &str, seed: u32) -> Result<GameSession, JsError> {
        Ok(GameSession {
            inner: Session::new(kind, seed as u64).map_err(|e| JsError::new(&e))?,
        })
    }

    pub fn kind(&self) -> String {
        self.inner.kind().to_string()
    }

    /// One frame with a human action key (see [`GameSession::action_keys`]; `""` holds).
    /// Returns `{executed, vetoed, event, reward, done, score}` as JSON.
    pub fn step_human(&mut self, action_key: &str) -> Result<String, JsError> {
        self.inner
            .step_human(action_key)
            .map(|r| to_json(&r))
            .map_err(|e| JsError::new(&e))
    }

    /// One AI frame with the model. Returns `{policy, decided, action, executed, probs:
    /// [{key, p}], secondary: {id, type, value, max, probs}, others: [same as secondary],
    /// info: {family, exit, n_exits,
    /// layers_used, mean_layers, max_layers, static_hit, state_reused}, vetoed, micros,
    /// state_text, event, reward, done, score}` as JSON.
    pub fn step_ai(&mut self, engine: &mut Engine) -> Result<String, JsError> {
        self.inner
            .step_ai(Policy::Model(&mut engine.inner))
            .map(|r| to_json(&r))
            .map_err(|e| JsError::new(&e))
    }

    /// One AI frame with the game's oracle (same JSON as [`GameSession::step_ai`]).
    pub fn step_oracle(&mut self) -> Result<String, JsError> {
        self.inner
            .step_ai(Policy::Oracle)
            .map(|r| to_json(&r))
            .map_err(|e| JsError::new(&e))
    }

    /// Render data as JSON: `{kind, score, done, stats, view}` (`view` is the game snapshot).
    pub fn snapshot(&self) -> String {
        self.inner.snapshot().to_string()
    }

    pub fn reset(&mut self) {
        self.inner.reset()
    }

    pub fn score(&self) -> u32 {
        self.inner.score()
    }

    pub fn done(&self) -> bool {
        self.inner.done()
    }

    pub fn state_text(&self) -> String {
        self.inner.state_text()
    }

    /// Decide on every `n`-th due frame (hold the last decision in between).
    pub fn set_decide_every(&mut self, n: u32) {
        self.inner.set_decide_every(n)
    }

    pub fn decide_every(&self) -> u32 {
        self.inner.decide_every()
    }

    /// Enable or disable the coded safety reflex (rescue, dodge).
    pub fn set_veto(&mut self, on: bool) {
        self.inner.set_veto(on)
    }

    /// Rolling timing of the last decisions: `{n, median_us, p95_us, mean_us}`.
    pub fn timing(&self) -> String {
        self.inner.timing().to_string()
    }

    /// Action keys accepted by [`GameSession::step_human`], as a JSON array.
    pub fn action_keys(&self) -> String {
        to_json(&self.inner.action_keys())
    }

    /// The questions as JSON: `[{id, type, instructions, keys, options}]`.
    pub fn questions(&self) -> String {
        let qs: Vec<serde_json::Value> = self
            .inner
            .questions()
            .iter()
            .map(|q| {
                serde_json::json!({
                    "id": q.id,
                    "type": q.qtype.name(),
                    "instructions": q.instructions,
                    "keys": q.keys,
                    "options": q.options,
                })
            })
            .collect();
        to_json(&qs)
    }
}

/// Times `n` AI decisions on Flappy Drone: `{name, family, params, file_bytes, n, median_us,
/// p95_us, mean_us, decisions_per_s, forwards_per_decision, mean_layers_used, max_layers,
/// static_hit_rate, state_reuse_rate, game}` as JSON.
#[wasm_bindgen]
pub fn bench(engine: &mut Engine, n: u32) -> Result<String, JsError> {
    session::bench(&mut engine.inner, n as usize)
        .map(|v| v.to_string())
        .map_err(|e| JsError::new(&e))
}

/// Like [`bench`] on any game (`flappy`, `rescue`, `dodge`, `doom`).
#[wasm_bindgen(js_name = benchGame)]
pub fn bench_game(engine: &mut Engine, game: &str, n: u32) -> Result<String, JsError> {
    session::bench_game(&mut engine.inner, game, n as usize)
        .map(|v| v.to_string())
        .map_err(|e| JsError::new(&e))
}

/// Crate version.
#[wasm_bindgen]
pub fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}
