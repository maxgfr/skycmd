//! TinyCast: a pure Rust port of RAWS Labs' zero-shot forecaster (146,505 parameters,
//! Apache-2.0; code <https://github.com/raws-labs/tinycast>, weights
//! <https://huggingface.co/raws-labs/tinycast>, paper arXiv 2608.15767).
//!
//! It reproduces the official deployed inference, `TinyCastPredictor.predict` on CPU, step by
//! step, so that a forecast here equals the PyTorch one up to float rounding
//! (`tests/golden_tinycast.rs`, `scripts/cast-ref-golden.py`):
//!
//! 1. Context: the last `seq_len` (2048) points, left-padded with the first value when shorter.
//!    Gaps (NaN) are filled by linear interpolation (constant beyond the ends) when at least two
//!    points are observed, else forward then backward fill, then 0.
//! 2. Per window: min-max normalization (`WindowMinMax`: non-finite values read as 0, range
//!    clamped to 1e-5).
//! 3. A zero-parameter period detector: the normalized periodogram's local maxima above a
//!    Bonferroni threshold `ln(N/α)/N`, top `top_k_periods`, period `round(n_fft / k)`.
//! 4. Positional encoding: sin/cos of each period's phase, and five bounded recency channels.
//! 5. Encoder: `in_proj`, then `n_layers` causal dilated blocks (depthwise conv of dilation
//!    2^i, pointwise conv, RMSNorm, SwiGLU shared by every block, RMSNorm).
//! 6. Decoder, per future step: its encoding, the mean and last encoder states, and the
//!    phase-folded encoder profile at its phase for each period (`phase_mix`), through
//!    `query_proj`; plus the future-conv states (a causal dilated conv over the last 128 encoder
//!    states followed by a seasonal-naive fill of the horizon, `fc_out`); a residual SwiGLU block
//!    and `out_proj` give the quantiles.
//! 7. Output: clamped to [-5, 5] in normalized units, denormalized.
//! 8. Rollout: horizons beyond `output_token_len` (48) run in 48-step chunks, the median of each
//!    chunk appended to the context of the next.
//! 9. [`ForecastOptions::flip`] is the predictor's `force_flip_invariance`: a second rollout on
//!    the negated context, `(f(x) − reverse(f(−x))) / 2` level by level. Each step's quantiles
//!    are finally sorted.
//!
//! Deviations from the official code, all deliberate:
//! - Compute is f32 throughout, the official CPU path (the GIFT-Eval numbers were produced with
//!   bf16 autocast on CUDA, about 1e-3 away according to the predictor's own documentation).
//!   Sums and matrix products are accumulated in a different order than PyTorch's, and sin,
//!   cos, exp and log1p are evaluated in f64 then rounded.
//! - The periodogram's FFT runs in f64 instead of PyTorch's f32 `rfft`. Its scores only select
//!   integer periods; a period can differ only when two peaks, or a peak and the threshold, are
//!   within f32 rounding of each other.
//! - The period-alignment downsample of the GIFT-Eval driver (`tinycast.downsample`) is not
//!   applied: it is not part of the predictor (whose `downsample_factor` defaults to 1), and it
//!   needs the sampling frequency and a quorum over every series of a dataset.
//! - Not ported, all off by default in the predictor: `single_shot`, `adaptive_ar_rollout`, the
//!   `TINYCAST_INT8` fake quantization and the `TINYCAST_TILT_K` probe. The predictor's
//!   `freq`/`domain` scale factor is not taken either: only `local_anchor`, off in the released
//!   configuration, reads it.
//! - [`ForecastOptions::full_depth`] has no meaning here (no early exits) and is ignored.
//!
//! The GGUF (architecture `skycmd-tinycast`) keeps the original tensor names and stores the two
//! weight-tied FFNs once, as `model.safetensors` does; [`tinycast_gguf_from_safetensors`] builds
//! it from the Hugging Face files. Configuration keys are the `config.json` names under
//! `skycmd-tinycast.`; only the released architecture's switches are supported.

use std::f64::consts::PI;

use sky_gguf::{GgmlType, Gguf, GgufWriter, OwnedValue};

use super::model::{Forecast, ForecastOptions};
use crate::json::Json;
use crate::kernels;
use crate::safetensors::SafeTensors;
use crate::{Error, Result};

/// GGUF architecture name of TinyCast files.
pub const ARCH: &str = "skycmd-tinycast";

/// Tensor name prefix of the original checkpoint.
const P: &str = "model.core.";
/// Recency channels of the positional encoding (`encoding.N_RECENCY_CHANNELS`).
const N_RECENCY: usize = 5;
/// The normalized-output clamp of the predictor (`_BackboneAdapter`).
const Y_CLAMP: f32 = 5.0;
/// `WindowMinMax(eps_clamp=1e-5)`.
const RANGE_MIN: f32 = 1e-5;
/// `nn.RMSNorm(d)` with no eps: `torch.finfo(float32).eps`.
const RMS_EPS: f32 = f32::EPSILON;

/// A TinyCast configuration (`config.json`, GGUF `skycmd-tinycast.*`).
#[derive(Clone, Debug, PartialEq)]
pub struct TinyCastConfig {
    pub name: String,
    /// Quantile levels, increasing, the median in the middle.
    pub quantiles: Vec<f32>,
    /// Encoder window `L`.
    pub seq_len: usize,
    /// Steps of one pass (the rollout chunk).
    pub output_token_len: usize,
    pub top_k_periods: usize,
    /// Significance level of the period detector (kept in f64, as Python computes the threshold).
    pub significance_alpha: f64,
    pub n_harmonics: usize,
    pub conv_dim: usize,
    pub n_layers: usize,
    pub kernel_size: usize,
    /// SwiGLU hidden width over `conv_dim`.
    pub ffn_mult: f32,
    pub phase_bins: usize,
    pub decoder_depth: usize,
    pub future_conv_layers: usize,
    pub future_conv_seed: usize,
}

impl Default for TinyCastConfig {
    /// The released model (`tinycast.config.TinyCastConfig()`).
    fn default() -> Self {
        TinyCastConfig {
            name: "TinyCast".into(),
            quantiles: super::DECILES.to_vec(),
            seq_len: 2048,
            output_token_len: 48,
            top_k_periods: 4,
            significance_alpha: 0.05,
            n_harmonics: 1,
            conv_dim: 64,
            n_layers: 10,
            kernel_size: 3,
            ffn_mult: 1.0,
            phase_bins: 16,
            decoder_depth: 1,
            future_conv_layers: 6,
            future_conv_seed: 128,
        }
    }
}

/// Switches of `config.json` this port implements only at their released value.
const FIXED: [(&str, &str); 5] = [
    ("pool_kind", "mean_last"),
    ("causal", "true"),
    ("separable_conv", "true"),
    ("share_ffn", "true"),
    ("future_conv", "true"),
];

impl TinyCastConfig {
    /// SwiGLU hidden width (`int(d * ffn_mult)`).
    pub fn hidden(&self) -> usize {
        (self.conv_dim as f32 * self.ffn_mult) as usize
    }

    /// Positional-encoding channels: sin/cos per period and harmonic, then recency.
    pub fn n_pe(&self) -> usize {
        2 * self.top_k_periods * self.n_harmonics + N_RECENCY
    }

    /// Index of the median level.
    pub fn median_index(&self) -> usize {
        self.quantiles.len() / 2
    }

    /// Reject configurations the engine cannot run.
    pub fn validate(&self) -> Result<()> {
        let bad = |m: &str| Err(Error::Format(format!("invalid TinyCast config: {m}")));
        if self.seq_len < 4 || self.seq_len > 1 << 16 {
            return bad("seq_len");
        }
        if self.output_token_len == 0 || self.output_token_len > 4096 {
            return bad("output_token_len");
        }
        if self.conv_dim == 0 || self.conv_dim > 4096 || self.hidden() == 0 {
            return bad("conv_dim or ffn_mult");
        }
        if self.n_layers == 0 || self.n_layers > 32 || self.kernel_size == 0 {
            return bad("n_layers or kernel_size");
        }
        if self.top_k_periods == 0 || self.top_k_periods > 64 || self.n_harmonics == 0 {
            return bad("top_k_periods or n_harmonics");
        }
        if self.significance_alpha.is_nan() || self.significance_alpha <= 0.0 {
            return bad("significance_alpha");
        }
        if self.phase_bins == 0 || self.phase_bins > 4096 {
            return bad("phase_bins (0, no phase folding, is not supported)");
        }
        if self.decoder_depth == 0 || self.future_conv_layers == 0 || self.future_conv_layers > 32 {
            return bad("decoder_depth or future_conv_layers");
        }
        let q = &self.quantiles;
        if q.is_empty() || q.windows(2).any(|w| w[0] >= w[1]) {
            return bad("quantiles must increase");
        }
        Ok(())
    }

    /// Read the configuration from GGUF metadata.
    pub fn from_gguf(g: &Gguf<'_>) -> Result<Self> {
        if g.architecture() != Some(ARCH) {
            return Err(Error::Format(format!(
                "not a TinyCast file (architecture {:?})",
                g.architecture()
            )));
        }
        let key = |k: &str| format!("{ARCH}.{k}");
        let req = |k: &str| {
            g.get_u64(&key(k))
                .map(|v| v as usize)
                .ok_or_else(|| Error::Missing(format!("GGUF key {}", key(k))))
        };
        let reqf = |k: &str| {
            g.get_f64(&key(k))
                .ok_or_else(|| Error::Missing(format!("GGUF key {}", key(k))))
        };
        for (k, v) in FIXED {
            let got = match g.get(&key(k)) {
                Some(sky_gguf::Value::Bool(b)) => b.to_string(),
                Some(sky_gguf::Value::String(s)) => s.to_string(),
                _ => return Err(Error::Missing(format!("GGUF key {}", key(k)))),
            };
            if got != v {
                return Err(Error::Unsupported(format!("TinyCast {k} = {got}")));
            }
        }
        let quantiles = g
            .get_array(&key("quantiles"))
            .and_then(|a| a.to_f64_vec())
            .ok_or_else(|| Error::Missing(key("quantiles")))?
            .into_iter()
            .map(|v| v as f32)
            .collect();
        let cfg = TinyCastConfig {
            name: g.get_str("general.name").unwrap_or("TinyCast").to_string(),
            quantiles,
            seq_len: req("seq_len")?,
            output_token_len: req("output_token_len")?,
            top_k_periods: req("top_k_periods")?,
            significance_alpha: reqf("significance_alpha")?,
            n_harmonics: req("n_harmonics")?,
            conv_dim: req("conv_dim")?,
            n_layers: req("n_layers")?,
            kernel_size: req("kernel_size")?,
            ffn_mult: reqf("ffn_mult")? as f32,
            phase_bins: req("phase_bins")?,
            decoder_depth: req("decoder_depth")?,
            future_conv_layers: req("future_conv_layers")?,
            future_conv_seed: req("future_conv_seed")?,
        };
        cfg.validate()?;
        Ok(cfg)
    }

    /// Read `config.json` of the Hugging Face release (missing keys take the released values;
    /// a switch this port does not implement is an error).
    pub fn from_json(text: &str) -> Result<Self> {
        let j = Json::parse(text)?;
        let d = TinyCastConfig::default();
        let int = |k: &str, def: usize| -> Result<usize> {
            match j.get(k) {
                None => Ok(def),
                Some(v) => v
                    .as_u64()
                    .map(|v| v as usize)
                    .ok_or_else(|| Error::Format(format!("config.json: {k} is not an integer"))),
            }
        };
        let num = |k: &str, def: f64| -> Result<f64> {
            match j.get(k) {
                None => Ok(def),
                Some(v) => v
                    .as_f64()
                    .ok_or_else(|| Error::Format(format!("config.json: {k} is not a number"))),
            }
        };
        for (k, v) in FIXED {
            let got = match j.get(k) {
                None => continue,
                Some(Json::Bool(b)) => b.to_string(),
                Some(Json::String(s)) => s.clone(),
                Some(other) => format!("{other:?}"),
            };
            if got != v {
                return Err(Error::Unsupported(format!("TinyCast {k} = {got}")));
            }
        }
        let quantiles = match j.get("quantiles") {
            None => d.quantiles.clone(),
            Some(v) => v
                .as_array()
                .ok_or_else(|| Error::Format("config.json: quantiles".into()))?
                .iter()
                .map(|q| q.as_f64().map(|q| q as f32))
                .collect::<Option<Vec<f32>>>()
                .ok_or_else(|| Error::Format("config.json: quantiles".into()))?,
        };
        let cfg = TinyCastConfig {
            name: d.name.clone(),
            quantiles,
            seq_len: int("seq_len", d.seq_len)?,
            output_token_len: int("output_token_len", d.output_token_len)?,
            top_k_periods: int("top_k_periods", d.top_k_periods)?,
            significance_alpha: num("significance_alpha", d.significance_alpha)?,
            n_harmonics: int("n_harmonics", d.n_harmonics)?,
            conv_dim: int("conv_dim", d.conv_dim)?,
            n_layers: int("n_layers", d.n_layers)?,
            kernel_size: int("kernel_size", d.kernel_size)?,
            ffn_mult: num("ffn_mult", d.ffn_mult as f64)? as f32,
            phase_bins: int("phase_bins", d.phase_bins)?,
            decoder_depth: int("decoder_depth", d.decoder_depth)?,
            future_conv_layers: int("future_conv_layers", d.future_conv_layers)?,
            future_conv_seed: int("future_conv_seed", d.future_conv_seed)?,
        };
        cfg.validate()?;
        Ok(cfg)
    }
}

/// `nn.Linear`: `w` is `[out][in]`.
#[derive(Clone, Debug)]
struct Linear {
    w: Vec<f32>,
    b: Vec<f32>,
    n_in: usize,
    n_out: usize,
}

impl Linear {
    /// `y (+)= x · wᵀ + b` over `rows` rows.
    fn forward(&self, x: &[f32], rows: usize, y: &mut [f32], accumulate: bool) {
        kernels::matmul_f32(
            x,
            rows,
            self.n_in,
            &self.w,
            self.n_out,
            Some(&self.b),
            y,
            accumulate,
        );
    }

    fn n_params(&self) -> usize {
        self.w.len() + self.b.len()
    }
}

/// `_SwiGLU`: `down(silu(gate) · val)` with `[gate, val] = up(x)`.
#[derive(Clone, Debug)]
struct SwiGlu {
    up: Linear,
    down: Linear,
}

/// `_DilatedConvBlock` (separable, causal): depthwise dilated conv, pointwise conv, residual
/// RMSNorm, residual SwiGLU, RMSNorm.
#[derive(Clone, Debug)]
struct ConvBlock {
    /// Depthwise taps, `[kernel][d]` (transposed from `[d, 1, kernel]` for contiguous rows).
    dw_w: Vec<f32>,
    dw_b: Vec<f32>,
    pw: Linear,
    norm1: Vec<f32>,
    norm2: Vec<f32>,
    dilation: usize,
}

/// Work buffers of a block stack.
#[derive(Default)]
struct Scratch {
    dw: Vec<f32>,
    tmp: Vec<f32>,
    up: Vec<f32>,
    act: Vec<f32>,
}

fn grow(v: &mut Vec<f32>, n: usize) -> &mut [f32] {
    if v.len() < n {
        v.resize(n, 0.0);
    }
    &mut v[..n]
}

/// `RMSNorm` of each `d`-row of `src` into `dst`: `(x · rsqrt(mean(x²) + eps)) · w`.
fn rms_norm(src: &[f32], dst: &mut [f32], w: &[f32]) {
    let d = w.len();
    for (s, o) in src.chunks_exact(d).zip(dst.chunks_exact_mut(d)) {
        let ms = s.iter().map(|&v| v as f64 * v as f64).sum::<f64>() as f32 / d as f32;
        let r = 1.0 / (ms + RMS_EPS).sqrt();
        for ((o, &v), &g) in o.iter_mut().zip(s).zip(w) {
            *o = (v * r) * g;
        }
    }
}

impl SwiGlu {
    /// `x ← norm(x + ffn(x))`.
    fn residual_norm(&self, x: &mut [f32], rows: usize, norm: &[f32], s: &mut Scratch) {
        let h = self.down.n_in;
        let up = grow(&mut s.up, rows * 2 * h);
        self.up.forward(x, rows, up, false);
        let act = grow(&mut s.act, rows * h);
        for (u, a) in up.chunks_exact(2 * h).zip(act.chunks_exact_mut(h)) {
            for ((a, &g), &v) in a.iter_mut().zip(&u[..h]).zip(&u[h..]) {
                *a = g / (1.0 + (-g).exp()) * v;
            }
        }
        let tmp = grow(&mut s.tmp, x.len());
        tmp.copy_from_slice(x);
        self.down.forward(act, rows, tmp, true);
        rms_norm(tmp, x, norm);
    }
}

impl ConvBlock {
    /// One block over `x` (`[rows][d]`), in place.
    fn forward(&self, ffn: &SwiGlu, x: &mut [f32], rows: usize, s: &mut Scratch) {
        let d = self.dw_b.len();
        let k = self.dw_w.len() / d;
        let dw = grow(&mut s.dw, rows * d);
        for t in 0..rows {
            let o = &mut dw[t * d..(t + 1) * d];
            o.copy_from_slice(&self.dw_b);
            // output t reads x[t - (k-1-j)·dilation] at tap j; left padding is zeros
            for j in 0..k {
                let back = (k - 1 - j) * self.dilation;
                if back > t {
                    continue;
                }
                let src = &x[(t - back) * d..(t - back + 1) * d];
                let w = &self.dw_w[j * d..(j + 1) * d];
                for ((o, &v), &w) in o.iter_mut().zip(src).zip(w) {
                    *o += w * v;
                }
            }
        }
        let tmp = grow(&mut s.tmp, rows * d);
        tmp.copy_from_slice(x);
        self.pw.forward(dw, rows, tmp, true);
        rms_norm(tmp, x, &self.norm1);
        ffn.residual_norm(x, rows, &self.norm2, s);
    }
}

/// A loaded TinyCast model.
#[derive(Clone, Debug)]
pub struct TinyCastModel {
    pub cfg: TinyCastConfig,
    in_proj: Linear,
    encoder: Vec<ConvBlock>,
    /// The SwiGLU shared by every encoder block.
    encoder_ffn: SwiGlu,
    phase_mix: Linear,
    query_proj: Linear,
    decoder: Vec<(SwiGlu, Vec<f32>)>,
    out_proj: Linear,
    fc_in_proj: Linear,
    fc_blocks: Vec<ConvBlock>,
    fc_ffn: SwiGlu,
    fc_out: Linear,
}

/// Reads tensors of a GGUF file by their original name, checking shapes.
struct Src<'g, 'a> {
    g: &'g Gguf<'a>,
}

impl Src<'_, '_> {
    fn f32(&self, name: &str, shape: &[usize]) -> Result<Vec<f32>> {
        let name = format!("{P}{name}");
        let info = self
            .g
            .tensor(&name)
            .ok_or_else(|| Error::Missing(format!("tensor {name}")))?;
        if info.shape() != shape {
            return Err(Error::Format(format!(
                "tensor {name}: shape {:?}, expected {shape:?}",
                info.shape()
            )));
        }
        Ok(sky_gguf::dequantize(info.dtype, self.g.tensor_bytes(info))?)
    }

    fn lin(&self, name: &str, n_out: usize, n_in: usize) -> Result<Linear> {
        Ok(Linear {
            w: self.f32(&format!("{name}.weight"), &[n_out, n_in])?,
            b: self.f32(&format!("{name}.bias"), &[n_out])?,
            n_in,
            n_out,
        })
    }

    fn swiglu(&self, name: &str, d: usize, h: usize) -> Result<SwiGlu> {
        Ok(SwiGlu {
            up: self.lin(&format!("{name}.up"), 2 * h, d)?,
            down: self.lin(&format!("{name}.down"), d, h)?,
        })
    }

    fn block(&self, name: &str, d: usize, k: usize, dilation: usize) -> Result<ConvBlock> {
        let w = self.f32(&format!("{name}.conv.0.weight"), &[d, 1, k])?;
        let pw = Linear {
            w: self.f32(&format!("{name}.conv.1.weight"), &[d, d, 1])?,
            b: self.f32(&format!("{name}.conv.1.bias"), &[d])?,
            n_in: d,
            n_out: d,
        };
        Ok(ConvBlock {
            dw_w: (0..k)
                .flat_map(|j| (0..d).map(move |c| (c, j)))
                .map(|(c, j)| w[c * k + j])
                .collect(),
            dw_b: self.f32(&format!("{name}.conv.0.bias"), &[d])?,
            pw,
            norm1: self.f32(&format!("{name}.norm1.weight"), &[d])?,
            norm2: self.f32(&format!("{name}.norm2.weight"), &[d])?,
            dilation,
        })
    }
}

impl TinyCastModel {
    /// Load a `skycmd-tinycast` GGUF file.
    pub fn from_gguf(g: &Gguf<'_>) -> Result<Self> {
        let cfg = TinyCastConfig::from_gguf(g)?;
        let src = Src { g };
        let (d, h, k) = (cfg.conv_dim, cfg.hidden(), cfg.kernel_size);
        let n_pe = cfg.n_pe();
        let encoder = (0..cfg.n_layers)
            .map(|i| src.block(&format!("encoder.{i}"), d, k, 1 << i))
            .collect::<Result<Vec<_>>>()?;
        let fc_blocks = (0..cfg.future_conv_layers)
            .map(|i| src.block(&format!("fc_blocks.{i}"), d, k, 1 << i))
            .collect::<Result<Vec<_>>>()?;
        let decoder = (0..cfg.decoder_depth)
            .map(|i| {
                Ok((
                    src.swiglu(&format!("decoder_ffns.{i}"), d, h)?,
                    src.f32(&format!("decoder_norms.{i}.weight"), &[d])?,
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(TinyCastModel {
            in_proj: src.lin("in_proj", d, 1 + n_pe)?,
            encoder,
            encoder_ffn: src.swiglu("encoder.0.ffn", d, h)?,
            phase_mix: src.lin("phase_mix", d, cfg.top_k_periods * d)?,
            query_proj: src.lin("query_proj", d, n_pe + 3 * d)?,
            decoder,
            out_proj: src.lin("out_proj", cfg.quantiles.len(), d)?,
            fc_in_proj: src.lin("fc_in_proj", d, 1 + n_pe)?,
            fc_blocks,
            fc_ffn: src.swiglu("fc_blocks.0.ffn", d, h)?,
            fc_out: src.lin("fc_out", d, d)?,
            cfg,
        })
    }

    /// Load from the bytes of a GGUF file.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        Self::from_gguf(&Gguf::parse(bytes)?)
    }

    /// Number of scalars, each weight-tied FFN counted once (146,505 for the release).
    pub fn n_params(&self) -> usize {
        let block = |b: &ConvBlock| {
            b.dw_w.len() + b.dw_b.len() + b.pw.n_params() + b.norm1.len() + b.norm2.len()
        };
        let ffn = |f: &SwiGlu| f.up.n_params() + f.down.n_params();
        self.in_proj.n_params()
            + self.encoder.iter().map(block).sum::<usize>()
            + ffn(&self.encoder_ffn)
            + self.phase_mix.n_params()
            + self.query_proj.n_params()
            + self
                .decoder
                .iter()
                .map(|(f, n)| ffn(f) + n.len())
                .sum::<usize>()
            + self.out_proj.n_params()
            + self.fc_in_proj.n_params()
            + self.fc_blocks.iter().map(block).sum::<usize>()
            + ffn(&self.fc_ffn)
            + self.fc_out.n_params()
    }

    /// Positional encoding of position `pos` (`encoding._positional_encoding`), in f32 as
    /// PyTorch computes it: phase `2π·m·pos/p` rounded at each step, then sin/cos.
    fn pos_enc(&self, pos: usize, periods: &[usize], out: &mut [f32]) {
        let c = &self.cfg;
        let tau = (2.0 * PI) as f32;
        let mut o = 0;
        for &p in periods {
            let base = tau * pos as f32 / p.max(1) as f32;
            for m in 1..=c.n_harmonics {
                let ph = (base * m as f32) as f64;
                let (s, co) = if p > 0 {
                    (ph.sin() as f32, ph.cos() as f32)
                } else {
                    (0.0, 0.0)
                };
                out[o] = s;
                out[o + 1] = co;
                o += 2;
            }
        }
        let l = c.seq_len as f32;
        let delta = (pos as f32 - (l - 1.0)) / l;
        let a = delta.abs();
        let sign = if delta > 0.0 {
            1.0
        } else if delta < 0.0 {
            -1.0
        } else {
            0.0
        };
        out[o] = delta;
        out[o + 1] = sign * ((a as f64).ln_1p() as f32) / std::f32::consts::LN_2;
        for (i, rate) in [0.5f32, 2.0, 8.0].into_iter().enumerate() {
            out[o + 2 + i] = ((-rate * a) as f64).exp() as f32;
        }
    }

    /// One pass on a prepared window of `seq_len` finite points: the quantiles
    /// `[output_token_len][levels]` in the window's units.
    pub fn forward_window(&self, window: &[f32]) -> Result<Vec<f32>> {
        let c = &self.cfg;
        let (l, d, nh) = (c.seq_len, c.conv_dim, c.output_token_len);
        if window.len() != l {
            return Err(Error::InvalidRequest(format!(
                "window of {} values, expected {l}",
                window.len()
            )));
        }
        // WindowMinMax
        let filled: Vec<f32> = window
            .iter()
            .map(|&v| if v.is_finite() { v } else { 0.0 })
            .collect();
        let lo = filled.iter().copied().fold(f32::INFINITY, f32::min);
        let hi = filled.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let range = (hi - lo).max(RANGE_MIN);
        let x: Vec<f32> = filled.iter().map(|&v| (v - lo) / range).collect();

        let periods = significant_periods(&x, c.top_k_periods, c.significance_alpha);
        let n_pe = c.n_pe();
        let mut pe = vec![0.0; n_pe];

        // encoder
        let ci = 1 + n_pe;
        let mut inp = vec![0.0; l * ci];
        for (t, row) in inp.chunks_exact_mut(ci).enumerate() {
            row[0] = x[t];
            self.pos_enc(t, &periods, &mut row[1..]);
        }
        let mut h = vec![0.0; l * d];
        self.in_proj.forward(&inp, l, &mut h, false);
        let mut s = Scratch::default();
        for b in &self.encoder {
            b.forward(&self.encoder_ffn, &mut h, l, &mut s);
        }

        // decoder queries: [PE(L + i), mean(h), h[L-1], phase profile]
        let qi = n_pe + 3 * d;
        let mut q_in = vec![0.0; nh * qi];
        let mut gmean = vec![0.0f64; d];
        for row in h.chunks_exact(d) {
            gmean.iter_mut().zip(row).for_each(|(a, &v)| *a += v as f64);
        }
        let gmean: Vec<f32> = gmean.iter().map(|&v| (v / l as f64) as f32).collect();
        let fut_pe: Vec<f32> = (0..nh)
            .flat_map(|i| {
                self.pos_enc(l + i, &periods, &mut pe);
                pe.clone()
            })
            .collect();
        for (i, row) in q_in.chunks_exact_mut(qi).enumerate() {
            row[..n_pe].copy_from_slice(&fut_pe[i * n_pe..(i + 1) * n_pe]);
            row[n_pe..n_pe + d].copy_from_slice(&gmean);
            row[n_pe + d..n_pe + 2 * d].copy_from_slice(&h[(l - 1) * d..]);
        }
        let k = c.top_k_periods;
        let mut gathered = vec![0.0; nh * k * d];
        for (kk, &p) in periods.iter().enumerate() {
            let prof = phase_profile(&h, d, p, c.phase_bins, &gmean);
            for i in 0..nh {
                let b = phase_bin(l + i, p, c.phase_bins);
                gathered[(i * k + kk) * d..(i * k + kk + 1) * d]
                    .copy_from_slice(&prof[b * d..(b + 1) * d]);
            }
        }
        let mut phase = vec![0.0; nh * d];
        self.phase_mix.forward(&gathered, nh, &mut phase, false);
        for (row, ph) in q_in.chunks_exact_mut(qi).zip(phase.chunks_exact(d)) {
            row[n_pe + 2 * d..].copy_from_slice(ph);
        }
        let mut q = vec![0.0; nh * d];
        self.query_proj.forward(&q_in, nh, &mut q, false);

        // future conv: the last `future_conv_seed` encoder states, then the seasonal-naive fill
        let fill = seasonal_naive(&x, nh, periods[0], c.phase_bins);
        let mut fc_in = vec![0.0; nh * ci];
        for (i, row) in fc_in.chunks_exact_mut(ci).enumerate() {
            row[0] = fill[i];
            row[1..].copy_from_slice(&fut_pe[i * n_pe..(i + 1) * n_pe]);
        }
        let seed = c.future_conv_seed.min(l);
        let mut z = vec![0.0; (seed + nh) * d];
        z[..seed * d].copy_from_slice(&h[(l - seed) * d..]);
        self.fc_in_proj
            .forward(&fc_in, nh, &mut z[seed * d..], false);
        for b in &self.fc_blocks {
            b.forward(&self.fc_ffn, &mut z, seed + nh, &mut s);
        }
        self.fc_out.forward(&z[seed * d..], nh, &mut q, true);

        for (ffn, norm) in &self.decoder {
            ffn.residual_norm(&mut q, nh, norm, &mut s);
        }
        let nq = c.quantiles.len();
        let mut y = vec![0.0; nh * nq];
        self.out_proj.forward(&q, nh, &mut y, false);
        for v in &mut y {
            *v = v.clamp(-Y_CLAMP, Y_CLAMP) * range + lo;
        }
        Ok(y)
    }

    /// The predictor's context (`_left_pad_to_len` then the NaN imputation of
    /// `_prepare_context_matrix`): `seq_len` finite points.
    pub fn prepare(&self, history: &[f32]) -> Vec<f32> {
        let l = self.cfg.seq_len;
        let mut a: Vec<f32> = if history.len() >= l {
            history[history.len() - l..].to_vec()
        } else {
            let fill = history.first().copied().unwrap_or(0.0);
            let mut a = vec![fill; l - history.len()];
            a.extend_from_slice(history);
            a
        };
        if a.iter().any(|v| v.is_nan()) {
            let valid: Vec<usize> = (0..l).filter(|&i| !a[i].is_nan()).collect();
            if valid.len() >= 2 {
                a = interp(&a, &valid);
            } else {
                forward_fill(&mut a);
            }
        }
        // forward fill, then backward fill what is left, then 0
        let mut ff = a.clone();
        forward_fill(&mut ff);
        let mut bf: Vec<f32> = a.iter().rev().copied().collect();
        forward_fill(&mut bf);
        bf.reverse();
        ff.iter()
            .zip(&bf)
            .map(|(&f, &b)| {
                let v = if f.is_nan() { b } else { f };
                if v.is_nan() {
                    0.0
                } else {
                    v
                }
            })
            .collect()
    }

    /// The autoregressive rollout (`_decode_autoregressive`): whole chunks, `[steps · chunk]
    /// [levels]`, and the number of passes.
    fn rollout(&self, mut ctx: Vec<f32>, horizon: usize) -> Result<(Vec<f32>, usize)> {
        let (l, step) = (self.cfg.seq_len, self.cfg.output_token_len);
        let nq = self.cfg.quantiles.len();
        let med = self.cfg.median_index();
        let steps = horizon.div_ceil(step);
        let mut out = Vec::with_capacity(steps * step * nq);
        for _ in 0..steps {
            let y = self.forward_window(&ctx[ctx.len() - l..])?;
            ctx.extend(y.chunks_exact(nq).map(|r| r[med]));
            out.extend_from_slice(&y);
        }
        Ok((out, steps))
    }

    /// Quantile forecast of `horizon` steps after `history` (NaN = missing), as
    /// `TinyCastPredictor.predict` makes it. `opts.flip` is `force_flip_invariance`;
    /// `opts.full_depth` is ignored.
    pub fn forecast(
        &self,
        history: &[f32],
        horizon: usize,
        opts: ForecastOptions,
    ) -> Result<Forecast> {
        if horizon == 0 {
            return Err(Error::InvalidRequest("horizon 0".into()));
        }
        let nq = self.cfg.quantiles.len();
        let ctx = self.prepare(history);
        let neg: Vec<f32> = if opts.flip {
            ctx.iter().map(|v| -v).collect()
        } else {
            Vec::new()
        };
        let (mut pred, mut passes) = self.rollout(ctx, horizon)?;
        if opts.flip {
            let (pn, n) = self.rollout(neg, horizon)?;
            passes += n;
            for (p, r) in pred.chunks_exact_mut(nq).zip(pn.chunks_exact(nq)) {
                for (k, v) in p.iter_mut().enumerate() {
                    *v = 0.5 * (*v - r[nq - 1 - k]);
                }
            }
        }
        // NaN steps take the previous step's value, level by level
        if pred.iter().any(|v| v.is_nan()) {
            for k in 0..nq {
                let mut col: Vec<f32> = pred.iter().skip(k).step_by(nq).copied().collect();
                forward_fill(&mut col);
                for (i, v) in col.into_iter().enumerate() {
                    pred[i * nq + k] = v;
                }
            }
        }
        pred.truncate(horizon * nq);
        for row in pred.chunks_exact_mut(nq) {
            row.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        }
        Ok(Forecast {
            levels: self.cfg.quantiles.clone(),
            quantiles: pred,
            horizon,
            exits_used: 1,
            n_exits: 1,
            passes,
        })
    }
}

/// `_numpy_fill`: each NaN takes the last non-NaN value before it (leading NaNs stay).
fn forward_fill(a: &mut [f32]) {
    let mut last = f32::NAN;
    for v in a {
        if v.is_nan() {
            *v = last;
        } else {
            last = *v;
        }
    }
}

/// `np.interp(arange(n), valid, a[valid])`: linear between observed points, constant beyond,
/// in f64 then rounded to f32.
fn interp(a: &[f32], valid: &[usize]) -> Vec<f32> {
    let (first, last) = (valid[0], valid[valid.len() - 1]);
    let mut out = vec![0.0; a.len()];
    for (i, o) in out.iter_mut().enumerate() {
        *o = if i <= first {
            a[first]
        } else if i >= last {
            a[last]
        } else {
            // valid[j] <= i < valid[j + 1]
            let j = valid.partition_point(|&v| v <= i) - 1;
            let (x0, x1) = (valid[j], valid[j + 1]);
            if x0 == i {
                a[i]
            } else {
                let (y0, y1) = (a[x0] as f64, a[x1] as f64);
                let slope = (y1 - y0) / (x1 - x0) as f64;
                (slope * (i - x0) as f64 + y0) as f32
            }
        };
    }
    out
}

/// Phase bin of position `t` under period `p` (0 = rejected, everything in bin 0), computed in
/// f32 as PyTorch does: `min(floor((t mod p) / p · nb), nb − 1)`.
fn phase_bin(t: usize, p: usize, nb: usize) -> usize {
    let p = p.max(1) as f32;
    let frac = (t as f32 % p) / p;
    ((frac * nb as f32) as usize).min(nb - 1)
}

/// `_phase_profile` for one period: the mean encoder state per phase bin, `[nb][d]` (an empty
/// bin takes the global mean).
fn phase_profile(h: &[f32], d: usize, p: usize, nb: usize, gmean: &[f32]) -> Vec<f32> {
    let mut sum = vec![0.0f64; nb * d];
    let mut cnt = vec![0usize; nb];
    for (t, row) in h.chunks_exact(d).enumerate() {
        let b = phase_bin(t, p, nb);
        cnt[b] += 1;
        sum[b * d..(b + 1) * d]
            .iter_mut()
            .zip(row)
            .for_each(|(a, &v)| *a += v as f64);
    }
    let mut out = vec![0.0; nb * d];
    for b in 0..nb {
        let o = &mut out[b * d..(b + 1) * d];
        if cnt[b] == 0 {
            o.copy_from_slice(gmean);
        } else {
            for (o, &s) in o.iter_mut().zip(&sum[b * d..(b + 1) * d]) {
                *o = s as f32 / cnt[b] as f32;
            }
        }
    }
    out
}

/// `_seasonal_naive`: the mean normalized value per phase bin of the dominant period `p0`,
/// read at each future step's phase (the context mean when the bin is empty).
fn seasonal_naive(x: &[f32], horizon: usize, p0: usize, nb: usize) -> Vec<f32> {
    let l = x.len();
    let mut sum = vec![0.0f64; nb];
    let mut cnt = vec![0usize; nb];
    for (t, &v) in x.iter().enumerate() {
        let b = phase_bin(t, p0, nb);
        sum[b] += v as f64;
        cnt[b] += 1;
    }
    let mean = (x.iter().map(|&v| v as f64).sum::<f64>() / l as f64) as f32;
    (0..horizon)
        .map(|i| {
            let b = phase_bin(l + i, p0, nb);
            if cnt[b] > 0 {
                sum[b] as f32 / cnt[b] as f32
            } else {
                mean
            }
        })
        .collect()
}

/// In-place radix-2 FFT of a power-of-two length.
fn fft(re: &mut [f64], im: &mut [f64]) {
    let n = re.len();
    let bits = n.trailing_zeros();
    for i in 0..n {
        let j = i.reverse_bits() >> (usize::BITS - bits);
        if j > i {
            re.swap(i, j);
            im.swap(i, j);
        }
    }
    let tw: Vec<(f64, f64)> = (0..n / 2)
        .map(|j| {
            let a = -2.0 * PI * j as f64 / n as f64;
            (a.cos(), a.sin())
        })
        .collect();
    let mut len = 2;
    while len <= n {
        let stride = n / len;
        for s in (0..n).step_by(len) {
            for j in 0..len / 2 {
                let (wr, wi) = tw[j * stride];
                let (a, b) = (s + j, s + j + len / 2);
                let (xr, xi) = (re[b] * wr - im[b] * wi, re[b] * wi + im[b] * wr);
                re[b] = re[a] - xr;
                im[b] = im[a] - xi;
                re[a] += xr;
                im[a] += xi;
            }
        }
        len <<= 1;
    }
}

/// `periodogram.significant_periods` with `min_period = 2`, `max_period = L / 2`: the
/// `top_k` most powerful local maxima of the normalized periodogram that pass the Bonferroni
/// threshold `ln(N/α)/N`, as periods `round(n_fft / k)`, strongest first, 0 for empty slots.
pub fn significant_periods(x: &[f32], top_k: usize, alpha: f64) -> Vec<usize> {
    let l = x.len();
    let max_period = (l / 2).max(1);
    let n_fft = l.max(2).next_power_of_two();
    let mean = (x.iter().map(|&v| v as f64).sum::<f64>() / l as f64) as f32;
    let mut re: Vec<f64> = vec![0.0; n_fft];
    let mut im = vec![0.0; n_fft];
    for (r, &v) in re.iter_mut().zip(x) {
        *r = (v - mean) as f64;
    }
    fft(&mut re, &mut im);
    // bins 1..=n_fft/2 (DC skipped): bin i here is frequency i + 1
    let power: Vec<f64> = (1..=n_fft / 2)
        .map(|k| re[k] * re[k] + im[k] * im[k])
        .collect();
    let n_bins = power.len();
    let total = power.iter().sum::<f64>().max(1e-12);
    let score: Vec<f32> = power.iter().map(|&p| (p / total) as f32).collect();
    let k_lo = (n_fft / max_period).saturating_sub(1);
    let k_hi = (n_bins - 1).min((n_fft / 2).saturating_sub(1));
    let n_eff = n_bins.max(2) as f64;
    let alpha = alpha.clamp(1e-12, 1.0);
    let threshold = ((n_eff / alpha).ln() / n_eff) as f32;
    let mut peaks: Vec<(f32, usize)> = (1..n_bins.saturating_sub(1))
        .filter(|&i| {
            k_hi > k_lo
                && (k_lo..=k_hi).contains(&i)
                && score[i] > score[i - 1]
                && score[i] > score[i + 1]
                && score[i] >= threshold
        })
        .map(|i| (score[i], i))
        .collect();
    peaks.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
    let mut periods: Vec<usize> = peaks
        .iter()
        .take(top_k)
        .map(|&(_, i)| (n_fft as f32 / (i + 1) as f32).round_ties_even() as usize)
        .collect();
    periods.resize(top_k, 0);
    periods
}

/// Build a `skycmd-tinycast` GGUF (f32, original tensor names, tied FFNs stored once) from the
/// Hugging Face release's `model.safetensors` and `config.json`. The result is loaded back
/// before it is returned, so a file it produces always runs.
pub fn tinycast_gguf_from_safetensors(safetensors: &[u8], config_json: &str) -> Result<Vec<u8>> {
    let cfg = TinyCastConfig::from_json(config_json)?;
    let st = SafeTensors::parse(safetensors)?;
    let mut w = GgufWriter::new();
    w.add_kv("general.architecture", ARCH);
    w.add_kv("general.type", "model");
    w.add_kv("general.name", cfg.name.as_str());
    w.add_kv("general.organization", "RAWS Labs");
    w.add_kv("general.license", "Apache-2.0");
    w.add_kv(
        "general.source.url",
        "https://github.com/raws-labs/tinycast",
    );
    w.add_kv(
        "general.source.huggingface.repository",
        "raws-labs/tinycast",
    );
    let k = |s: &str| format!("{ARCH}.{s}");
    let u = |v: usize| OwnedValue::U32(v as u32);
    w.add_kv(
        k("quantiles"),
        OwnedValue::Array(cfg.quantiles.iter().map(|&q| OwnedValue::F32(q)).collect()),
    );
    w.add_kv(k("seq_len"), u(cfg.seq_len));
    w.add_kv(k("output_token_len"), u(cfg.output_token_len));
    w.add_kv(k("top_k_periods"), u(cfg.top_k_periods));
    w.add_kv(
        k("significance_alpha"),
        OwnedValue::F64(cfg.significance_alpha),
    );
    w.add_kv(k("n_harmonics"), u(cfg.n_harmonics));
    w.add_kv(k("conv_dim"), u(cfg.conv_dim));
    w.add_kv(k("n_layers"), u(cfg.n_layers));
    w.add_kv(k("kernel_size"), u(cfg.kernel_size));
    w.add_kv(k("ffn_mult"), OwnedValue::F32(cfg.ffn_mult));
    w.add_kv(k("pool_kind"), "mean_last");
    w.add_kv(k("causal"), true);
    w.add_kv(k("phase_bins"), u(cfg.phase_bins));
    w.add_kv(k("decoder_depth"), u(cfg.decoder_depth));
    w.add_kv(k("separable_conv"), true);
    w.add_kv(k("share_ffn"), true);
    w.add_kv(k("future_conv"), true);
    w.add_kv(k("future_conv_layers"), u(cfg.future_conv_layers));
    w.add_kv(k("future_conv_seed"), u(cfg.future_conv_seed));
    for t in &st.tensors {
        let dims: Vec<u64> = t.shape.iter().rev().map(|&v| v as u64).collect();
        w.add_tensor_f32(t.name.clone(), &dims, GgmlType::F32, &t.to_f32()?)?;
    }
    let bytes = w.to_bytes()?;
    TinyCastModel::from_bytes(&bytes)?;
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fft_matches_a_direct_dft() {
        let n = 64;
        let x: Vec<f64> = (0..n).map(|i| ((i * 37 % 11) as f64).sin()).collect();
        let (mut re, mut im) = (x.clone(), vec![0.0; n]);
        fft(&mut re, &mut im);
        for k in 0..n {
            let (mut sr, mut si) = (0.0, 0.0);
            for (t, &v) in x.iter().enumerate() {
                let a = -2.0 * PI * (k * t) as f64 / n as f64;
                sr += v * a.cos();
                si += v * a.sin();
            }
            assert!((re[k] - sr).abs() < 1e-9 && (im[k] - si).abs() < 1e-9);
        }
    }

    #[test]
    fn periods_of_a_clean_sine_and_of_a_constant() {
        let x: Vec<f32> = (0..2048)
            .map(|t| (2.0 * PI * t as f64 / 24.0).sin() as f32)
            .collect();
        let p = significant_periods(&x, 4, 0.05);
        assert_eq!(p[0], 24, "{p:?}");
        assert_eq!(significant_periods(&[3.0; 2048], 4, 0.05), vec![0; 4]);
    }

    #[test]
    fn interpolation_and_fills_follow_numpy() {
        let nan = f32::NAN;
        let a = [nan, 1.0, nan, nan, 4.0, nan];
        assert_eq!(interp(&a, &[1, 4]), vec![1.0, 1.0, 2.0, 3.0, 4.0, 4.0]);
        let mut b = [nan, 2.0, nan, 5.0, nan];
        forward_fill(&mut b);
        assert!(b[0].is_nan());
        assert_eq!(&b[1..], &[2.0, 2.0, 5.0, 5.0]);
    }

    #[test]
    fn phase_bins_fold_by_period() {
        assert_eq!(phase_bin(0, 7, 16), 0);
        assert_eq!(phase_bin(6, 7, 16), 13);
        assert_eq!(phase_bin(7, 7, 16), 0);
        assert_eq!(phase_bin(123, 0, 16), 0);
    }
}
