//! A loaded decision model (s1, s2 or iso), as used by the game sessions and the /live demos.

use serde_json::{json, Value};
use sky_infer::{Arch, DecisionStats, EngineOptions, QType};

/// A System One model loaded from GGUF bytes.
///
/// Ternary s2 files are expanded to f32 at load: with baseline SIMD128 the packed kernels are
/// not faster than f32 in browsers (measured in BENCH.md, "Inference"). Q8_0 and Q4_0 matrices
/// stay quantized: an s5 model is mostly embeddings (tens of MB in f32), and keeping it in Q8_0
/// divides its memory by four (Q4_0: by seven).
pub struct Model {
    inner: sky_infer::Engine,
    file_bytes: usize,
}

impl Model {
    pub fn load(gguf: &[u8]) -> Result<Model, String> {
        Self::load_with(gguf, EngineOptions::default())
    }

    /// Load with explicit engine options (`dequantize` is set for s2 files only).
    pub fn load_with(gguf: &[u8], opts: EngineOptions) -> Result<Model, String> {
        let s2 = sky_gguf::Gguf::parse(gguf)
            .ok()
            .is_some_and(|g| g.architecture() == Some(sky_infer::s2::ARCH));
        let opts = EngineOptions {
            dequantize: s2,
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

    /// State-encoder depth (s2) or encoder layers (s1, iso).
    pub fn max_layers(&self) -> usize {
        if let Some(m) = self.inner.s2_model() {
            return m.cfg.state_layers;
        }
        if let Some(m) = self.inner.s1_model() {
            return m.cfg.num_layers;
        }
        self.inner.model_info_ref().num_layers
    }

    /// iso: language of the generic `noul` definitions (`en`, `fr`). No effect on s1 / s2.
    pub fn set_lang(&mut self, lang: &str) {
        if let Some(e) = self.inner.iso_engine_mut() {
            e.set_lang(lang);
        }
    }

    /// The underlying engine.
    pub fn engine_mut(&mut self) -> &mut sky_infer::Engine {
        &mut self.inner
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
