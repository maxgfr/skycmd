//! Forecasting in the browser (the /forecast/ page): an s6-cast or TinyCast GGUF behind one
//! class, plus the zero-parameter pieces the page compares against — seasonal naive, the
//! synthetic generators of s6-cast's pretraining, and the periodogram period detector.
//!
//! ```js
//! const f = new Forecaster(bytes);                       // skycmd-cast or skycmd-tinycast
//! const q = f.forecast(history, 64, false);              // Float32Array [64 × levels]
//! const info = JSON.parse(f.info());                     // name, params, levels, last µs …
//! const y = synthSeries("tsi", 7, 512);                  // a demo series
//! const sn = seasonalNaive(history, 24, 64);             // Float32Array [64]
//! ```

use sky_infer::cast::model::ForecastOptions;
use sky_infer::cast::{prep, synth};
use wasm_bindgen::prelude::*;

use crate::clock::now_us;

/// A loaded forecaster.
#[wasm_bindgen]
pub struct Forecaster {
    inner: sky_infer::cast::Forecaster,
    params: usize,
    bytes: usize,
    last_us: f64,
    last_exits: usize,
    last_passes: usize,
}

#[wasm_bindgen]
impl Forecaster {
    #[wasm_bindgen(constructor)]
    pub fn new(gguf_bytes: &[u8]) -> Result<Forecaster, JsError> {
        let mut inner = sky_infer::cast::Forecaster::from_bytes(gguf_bytes)
            .map_err(|e| JsError::new(&e.to_string()))?;
        inner.prepare_compute(sky_infer::TernaryCompute::Auto);
        let params = inner.n_params();
        Ok(Forecaster {
            inner,
            params,
            bytes: gguf_bytes.len(),
            last_us: 0.0,
            last_exits: 0,
            last_passes: 0,
        })
    }

    /// Quantiles of `horizon` steps after `history` (NaN = missing), `[horizon × levels]`
    /// row-major, in the units of the history. `flip` averages with the sign-flipped series
    /// (two passes).
    pub fn forecast(
        &mut self,
        history: &[f32],
        horizon: usize,
        flip: bool,
    ) -> Result<Vec<f32>, JsError> {
        let t = now_us();
        let f = self
            .inner
            .forecast(
                history,
                horizon,
                ForecastOptions {
                    flip,
                    full_depth: false,
                },
            )
            .map_err(|e| JsError::new(&e.to_string()))?;
        self.last_us = now_us() - t;
        self.last_exits = f.exits_used;
        self.last_passes = f.passes;
        Ok(f.quantiles)
    }

    /// The quantile levels.
    pub fn levels(&self) -> Vec<f32> {
        self.inner.levels().to_vec()
    }

    /// JSON: `name`, `params`, `bytes`, `levels`, `pass_horizon`, and of the last forecast
    /// `micros`, `exits_used`, `passes`.
    pub fn info(&self) -> String {
        serde_json::json!({
            "name": self.inner.name(),
            "params": self.params,
            "bytes": self.bytes,
            "levels": self.inner.levels(),
            "pass_horizon": self.inner.pass_horizon(),
            "micros": self.last_us,
            "exits_used": self.last_exits,
            "passes": self.last_passes,
        })
        .to_string()
    }
}

/// Seasonal-naive forecast: the last season repeated (`season` 1: the last value).
#[wasm_bindgen(js_name = seasonalNaive)]
pub fn seasonal_naive(history: &[f32], season: usize, horizon: usize) -> Vec<f32> {
    prep::seasonal_naive(history, season, horizon)
}

/// A synthetic series of s6-cast's pretraining: `gp`, `pulse`, `tsi` or `sarima`.
#[wasm_bindgen(js_name = synthSeries)]
pub fn synth_series(kind: &str, seed: u64, len: usize) -> Result<Vec<f32>, JsError> {
    let k = synth::Kind::parse(kind).ok_or_else(|| {
        JsError::new(&format!(
            "unknown generator {kind:?} (gp, pulse, tsi, sarima)"
        ))
    })?;
    Ok(synth::series(k, seed, len))
}

/// The dominant period of `history` in `[2, len / 2]` from its periodogram, or 0.
#[wasm_bindgen(js_name = dominantPeriod)]
pub fn dominant_period(history: &[f32]) -> usize {
    prep::dominant_period(history, 2, (history.len() / 2).max(2)).unwrap_or(0)
}

/// MASE of `point` against `target`, scaled by the history's in-sample seasonal-naive error
/// (NaN when undefined).
#[wasm_bindgen]
pub fn mase(history: &[f32], season: usize, target: &[f32], point: &[f32]) -> f64 {
    prep::mase(history, season, target, point).unwrap_or(f64::NAN)
}
