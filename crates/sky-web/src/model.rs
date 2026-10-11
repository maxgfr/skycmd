//! A loaded decision model (s1 or s2), as used by the game sessions.

use serde_json::{json, Value};
use sky_infer::{Arch, DecisionStats, EngineOptions, QType};

/// A System One model loaded from GGUF bytes, computing in f32.
///
/// Q8_0 files are dequantized at load: with baseline SIMD128, int8 dot products are not faster
/// than f32 in browsers (see `sky_infer::wasm`), so this keeps the small download and gets f32
/// speed and accuracy. Ternary s2 files are expanded to f32 for the same reason (measured in
/// BENCH.md, "Inference").
pub struct Model {
    inner: sky_infer::Engine,
    file_bytes: usize,
}

impl Model {
    pub fn load(gguf: &[u8]) -> Result<Model, String> {
        Self::load_with(gguf, EngineOptions::default())
    }

    /// Load with explicit engine options (`dequantize` is always set: f32 compute).
    pub fn load_with(gguf: &[u8], opts: EngineOptions) -> Result<Model, String> {
        let opts = EngineOptions {
            dequantize: true,
            ..opts
        };
        let inner = sky_infer::Engine::from_gguf_with(gguf, opts).map_err(|e| e.to_string())?;
        Ok(Model {
            inner,
            file_bytes: gguf.len(),
        })
    }

    pub fn n_params(&self) -> usize {
        self.inner.n_params()
    }

    pub fn file_bytes(&self) -> usize {
        self.file_bytes
    }

    pub fn name(&self) -> String {
        self.inner.name().to_string()
    }

    pub fn arch(&self) -> Arch {
        self.inner.arch()
    }

    /// State-encoder depth (s2) or encoder layers (s1).
    pub fn max_layers(&self) -> usize {
        match self.inner.s2_model() {
            Some(m) => m.cfg.state_layers,
            None => self.inner.model().cfg.num_layers,
        }
    }

    /// s2: override τ for every question type (`None`: calibrated values; 1.0: full depth).
    pub fn set_exit_threshold(&mut self, tau: Option<f32>) {
        self.inner.set_exit_threshold(tau)
    }

    /// Statistics of the last question (exit, layers used, cache hits).
    pub fn last_stats(&self) -> Option<&DecisionStats> {
        self.inner.last_stats()
    }

    /// Model description (`sky_infer::ModelInfo`) plus `file_bytes`.
    pub fn info(&mut self) -> Value {
        let mut v: Value =
            serde_json::from_str(&self.inner.model_info().to_json()).unwrap_or_else(|_| json!({}));
        if let Some(o) = v.as_object_mut() {
            o.insert("file_bytes".into(), json!(self.file_bytes));
        }
        v
    }

    /// Probabilities over already rendered option texts. s1 caches the tokenized question
    /// prefix; s2 caches the static encoding and reuses the state encoding across the questions
    /// of a frame, so only the state is encoded per frame.
    pub fn probs(
        &mut self,
        qtype: QType,
        instructions: &str,
        options: &[String],
        state: &str,
    ) -> Result<Vec<f32>, String> {
        self.inner
            .decide_raw(qtype, instructions, options, state)
            .map_err(|e| e.to_string())
    }
}
