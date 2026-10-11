//! wasm-bindgen API (feature `wasm`).
//!
//! ```js
//! import init, { SkyEngine } from "./pkg/sky_infer.js";
//! await init();
//! const engine = new SkyEngine(new Uint8Array(await (await fetch("model.gguf")).arrayBuffer()));
//! const probs = engine.decide("choice", "Which action keeps the bird alive?",
//!                             ["flap: push the bird up", "wait"], "alt 12 vy -3"); // Float32Array
//! const answer = JSON.parse(engine.decide_json(JSON.stringify({ state: "...", questions: { ... } })));
//! ```

use wasm_bindgen::prelude::*;

use crate::kernels::ActQuant;
use crate::{Engine, EngineOptions, QType, QuantizeSpec};

fn js_err(e: crate::Error) -> JsError {
    JsError::new(&e.to_string())
}

fn qtype(name: &str) -> Result<QType, JsError> {
    QType::from_name(name).ok_or_else(|| {
        JsError::new(&format!(
            "unknown question type {name:?} (choice, score, noul)"
        ))
    })
}

/// A loaded decision model.
#[wasm_bindgen]
pub struct SkyEngine {
    inner: Engine,
}

#[wasm_bindgen]
impl SkyEngine {
    /// Load a GGUF model (s1 or s2; f32, f16, Q8_0 or ternary file) and compute in f32: with
    /// baseline SIMD128, int8 dot products are not faster than f32 in browsers, so Q8_0 files are
    /// dequantized at load (smaller download, f32 speed and accuracy).
    #[wasm_bindgen(constructor)]
    pub fn new(gguf_bytes: &[u8]) -> Result<SkyEngine, JsError> {
        Self::with_options(gguf_bytes, false, false)
    }

    /// `q8 = false`: f32 compute (Q8_0 matrices are dequantized). `q8 = true`: Q8_0 compute
    /// (f32 / f16 matrices are quantized at load), with single-pass activation quantization when
    /// `fast` (less accurate).
    #[wasm_bindgen(js_name = withOptions)]
    pub fn with_options(gguf_bytes: &[u8], q8: bool, fast: bool) -> Result<SkyEngine, JsError> {
        let opts = EngineOptions {
            quantize: q8.then_some(QuantizeSpec {
                encoder: true,
                head: true,
                embeddings: false,
            }),
            dequantize: !q8,
            act_quant: if fast { ActQuant::Q8 } else { ActQuant::Q8x2 },
            ..Default::default()
        };
        Ok(SkyEngine {
            inner: Engine::from_gguf_with(gguf_bytes, opts).map_err(js_err)?,
        })
    }

    /// Answer a System One request (`{"state": ..., "questions": {...}}`), returns the JSON response.
    pub fn decide_json(&mut self, req: &str) -> Result<String, JsError> {
        self.inner.decide_json(req).map_err(js_err)
    }

    /// Probabilities (after the temperature) of already rendered option texts. The tokens of the
    /// question and options are cached, so calling this every frame only tokenizes the state.
    pub fn decide(
        &mut self,
        qtype_name: &str,
        instructions: &str,
        options: Vec<String>,
        state: &str,
    ) -> Result<Vec<f32>, JsError> {
        self.inner
            .decide_raw(qtype(qtype_name)?, instructions, &options, state)
            .map_err(js_err)
    }

    /// Same as [`SkyEngine::decide`] with the options as a JSON array of strings.
    #[wasm_bindgen(js_name = decideOptionsJson)]
    pub fn decide_options_json(
        &mut self,
        qtype_name: &str,
        instructions: &str,
        options_json: &str,
        state: &str,
    ) -> Result<Vec<f32>, JsError> {
        let options: Vec<String> = serde_json::from_str(options_json)
            .map_err(|e| JsError::new(&format!("options: {e}")))?;
        self.decide(qtype_name, instructions, options, state)
    }

    /// Raw scores (before the temperature).
    pub fn scores(
        &mut self,
        qtype_name: &str,
        instructions: &str,
        options: Vec<String>,
        state: &str,
    ) -> Result<Vec<f32>, JsError> {
        self.inner
            .scores_raw(qtype(qtype_name)?, instructions, &options, state)
            .map_err(js_err)
    }

    /// Token ids of the full sequence (for debugging prompts).
    pub fn tokens(
        &mut self,
        qtype_name: &str,
        instructions: &str,
        options: Vec<String>,
        state: &str,
    ) -> Result<Vec<u32>, JsError> {
        let opts: Vec<&str> = options.iter().map(|s| s.as_str()).collect();
        Ok(self
            .inner
            .build_sequence(qtype(qtype_name)?, instructions, &opts, state)
            .0)
    }

    /// Number of parameters.
    pub fn n_params(&self) -> f64 {
        self.inner.n_params() as f64
    }

    /// Model description as JSON.
    pub fn model_info(&mut self) -> String {
        self.inner.model_info().to_json()
    }

    /// Duration of the last decision in microseconds (feature `timing`), or undefined.
    pub fn last_micros(&self) -> Option<f64> {
        self.inner.last_micros().map(|v| v as f64)
    }

    /// `general.architecture`: `modern-bert` (s1) or `skycmd-s2` (s2).
    pub fn arch(&self) -> String {
        self.inner.arch().name().to_string()
    }

    /// s2: override the early-exit threshold τ of every question type (undefined: the calibrated
    /// values; 1.0: full depth). No effect on s1 models.
    #[wasm_bindgen(js_name = setExitThreshold)]
    pub fn set_exit_threshold(&mut self, tau: Option<f32>) {
        self.inner.set_exit_threshold(tau)
    }

    /// Statistics of the last question as JSON: `{exit, layers_used, n_exits, static_hit,
    /// state_hit, tokens}` (`exit` and `layers_used` are null for s1), or `null`.
    #[wasm_bindgen(js_name = lastStats)]
    pub fn last_stats(&self) -> String {
        match self.inner.last_stats() {
            None => "null".into(),
            Some(s) => serde_json::json!({
                "exit": s.exit,
                "layers_used": s.layers_used,
                "n_exits": s.n_exits,
                "static_hit": s.static_hit,
                "state_hit": s.state_hit,
                "tokens": s.tokens,
            })
            .to_string(),
        }
    }
}

/// Product kernels on one matrix shape, for `bench/kernels-node.mjs` (feature `wasm-bench`):
/// runs `iters` products of `rows` x `k` activations by an `n` x `k` ternary matrix with the
/// kernel `kind` (`f32` panels, `int8` dot products, `lut1` / `lut2` lookup tables with one / two
/// activation passes, `packed` add/sub) and returns a checksum. Time it from JS.
#[cfg(feature = "wasm-bench")]
#[wasm_bindgen(js_name = kernelBench)]
pub fn kernel_bench(
    kind: &str,
    n: usize,
    k: usize,
    rows: usize,
    iters: usize,
) -> Result<f64, JsError> {
    use crate::kernels::{I8Panels, I8Scratch, PanelMatrix};
    use crate::lut::{LutMatrix, LutScratch};
    use crate::ternary::TernaryMatrix;
    let mut seed = 0x1234_5678_u64;
    let mut rnd = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        ((seed >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    };
    let w: Vec<f32> = (0..n * k).map(|_| rnd()).collect();
    let x: Vec<f32> = (0..rows * k).map(|_| rnd()).collect();
    let t = TernaryMatrix::from_f32(&w, n, k);
    let mut y = vec![0.0f32; rows * n];
    match kind {
        "f32" => {
            let p = PanelMatrix::from_rows(&t.to_f32(), n, k);
            for _ in 0..iters {
                p.matmul(&x, k, rows, None, &mut y, n, false);
            }
        }
        "int8" => {
            let p = I8Panels::from_trits(&t.trits(), &t.scale, n, k);
            let mut sc = I8Scratch::default();
            for _ in 0..iters {
                p.matmul(&x, rows, None, &mut y, false, &mut sc);
            }
        }
        "lut1" | "lut2" => {
            let passes = if kind == "lut1" { 1 } else { 2 };
            let p = LutMatrix::from_trits(&t.trits(), &t.scale, n, k, passes)
                .ok_or_else(|| JsError::new("k too large for the lookup tables"))?;
            let mut sc = LutScratch::default();
            for _ in 0..iters {
                p.matmul(&x, rows, None, &mut y, false, &mut sc);
            }
        }
        "packed" => {
            for _ in 0..iters {
                t.matmul(&x, rows, None, &mut y, false);
            }
        }
        _ => return Err(JsError::new("kind: f32, int8, lut1, lut2 or packed")),
    }
    Ok(y.iter().map(|&v| v as f64).sum())
}
