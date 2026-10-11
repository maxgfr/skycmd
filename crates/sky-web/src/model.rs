//! A loaded decision model, as used by the game sessions.

use serde_json::{json, Value};
use sky_infer::{EngineOptions, QType};

/// A System One model loaded from GGUF bytes, computing in f32.
///
/// Q8_0 files are dequantized at load: with baseline SIMD128, int8 dot products are not faster
/// than f32 in browsers (see `sky_infer::wasm`), so this keeps the small download and gets f32
/// speed and accuracy.
pub struct Model {
    inner: sky_infer::Engine,
    file_bytes: usize,
}

impl Model {
    pub fn load(gguf: &[u8]) -> Result<Model, String> {
        let opts = EngineOptions {
            dequantize: true,
            ..Default::default()
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
        self.inner.model().cfg.name.clone()
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

    /// Probabilities over already rendered option texts. The engine caches the tokenized
    /// `[CLS] question [SEP] options [SEP]` prefix, so only the state is tokenized per call.
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
