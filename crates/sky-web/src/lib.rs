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
//! The Rust API ([`Model`], [`Session`], [`session::bench`]) is usable natively; the
//! `#[wasm_bindgen]` types below are thin wrappers that turn errors into JS exceptions.

pub mod clock;
pub mod model;
pub mod session;

pub use model::Model;
pub use session::{AiStep, HumanStep, Policy, Session};

use wasm_bindgen::prelude::*;

fn to_json<T: serde::Serialize>(v: &T) -> String {
    serde_json::to_string(v).expect("serializable")
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

    /// Model family: `s1` (ModernBERT + decision head) or `s2` (static cache, early exits).
    pub fn family(&self) -> String {
        match self.inner.arch() {
            sky_infer::Arch::S1 => "s1".into(),
            sky_infer::Arch::S2 => "s2".into(),
        }
    }

    /// s2: override the early-exit threshold τ of every question type (undefined: the
    /// calibrated values; 1.0: full depth). No effect on s1 models.
    #[wasm_bindgen(js_name = setExitThreshold)]
    pub fn set_exit_threshold(&mut self, tau: Option<f32>) {
        self.inner.set_exit_threshold(tau)
    }
}

impl Engine {
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
