//! s6-cast: probabilistic time-series forecasting with the s6 state encoder (docs/SPEC-S6-CAST.md).
//!
//! A context of up to `context_patches · patch` points is min-max normalized ([`prep::Norm`]), cut
//! into patches of `patch` values (plus their observation mask) and projected to `d`
//! (`patch_proj`). The s6 state encoder (looped weight-shared blocks, Parallax, an exit per loop)
//! runs over the patches. At each exit, `horizon_patches` learned queries (`cast_query`) read the
//! exit's latents through `decoder_layers` cross-attention blocks, and `quant_head` gives `patch`
//! steps × `quantiles` levels per query, made non-crossing around the median. The output is
//! mapped back with the window's normalization.
//!
//! [`synth`] holds the synthetic generators of pretraining and [`prep`] the shared normalization,
//! patching, seasonal-naive reference and GIFT-Eval metrics.

pub mod model;
pub mod prep;
pub mod synth;

use sky_gguf::Gguf;

use crate::s2::S2Config;
use crate::{Error, Result};

pub use model::{CastModel, Forecast};

/// GGUF architecture name of s6-cast files.
pub const ARCH: &str = "skycmd-cast";

/// The default quantile levels: the nine deciles.
pub const DECILES: [f32; 9] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];

/// An s6-cast configuration (GGUF `skycmd-cast.*`).
#[derive(Clone, Debug, PartialEq)]
pub struct CastConfig {
    pub name: String,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_heads: usize,
    /// Distinct encoder blocks.
    pub state_blocks: usize,
    /// Passes over the blocks.
    pub state_loops: usize,
    /// Exit positions in block applications (increasing, the last = blocks × loops).
    pub exits: Vec<usize>,
    pub parallax: bool,
    pub xsa: bool,
    /// Points per patch.
    pub patch: usize,
    /// Longest context, in patches.
    pub context_patches: usize,
    /// Forecast of one pass, in patches.
    pub horizon_patches: usize,
    /// Quantile levels, increasing, symmetric around a median at the middle index.
    pub quantiles: Vec<f32>,
    pub decoder_layers: usize,
    pub rope_theta: f32,
    pub norm_eps: f32,
    /// Early exit: stop once the median of an exit moves less than this (normalized units, max
    /// over the horizon) from the previous exit's. 0 runs every exit.
    pub exit_tolerance: f32,
    pub weight_format: String,
}

impl CastConfig {
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_heads
    }

    /// Block applications of a full pass.
    pub fn depth(&self) -> usize {
        self.state_blocks * self.state_loops
    }

    /// Points forecast by one pass.
    pub fn horizon(&self) -> usize {
        self.horizon_patches * self.patch
    }

    /// Longest context in points.
    pub fn context(&self) -> usize {
        self.context_patches * self.patch
    }

    /// Index of the median level.
    pub fn median_index(&self) -> usize {
        self.quantiles.len() / 2
    }

    /// The encoder as an s2 configuration (the shape [`crate::s2`]'s block loader reads).
    pub(crate) fn encoder(&self) -> S2Config {
        S2Config {
            name: self.name.clone(),
            vocab_size: 1,
            hidden_size: self.hidden_size,
            intermediate_size: self.intermediate_size,
            num_heads: self.num_heads,
            state_layers: self.depth(),
            state_blocks: self.state_blocks,
            state_loops: self.state_loops,
            fuse_numbers: false,
            temporal_carry: false,
            parallax: self.parallax,
            xsa: self.xsa,
            ffn_es: None,
            static_layers: 0,
            fusion_layers: 0,
            exits: self.exits.clone(),
            max_state_tokens: self.context_patches,
            max_head_tokens: 8,
            num_features: crate::num::NUM_FEATURES,
            num_token_id: 5,
            rope_theta: self.rope_theta,
            norm_eps: self.norm_eps,
            exit_thresholds: [1.0; 3],
            temperatures: Vec::new(),
            weight_format: self.weight_format.clone(),
        }
    }

    /// Reject configurations the engine cannot run.
    pub fn validate(&self) -> Result<()> {
        const MAX_DIM: usize = 1 << 16;
        let bad = |m: &str| Err(Error::Format(format!("invalid s6-cast config: {m}")));
        let d = self.hidden_size;
        if d == 0 || d > MAX_DIM || self.intermediate_size == 0 || self.intermediate_size > MAX_DIM
        {
            return bad("hidden or feed-forward size");
        }
        if self.num_heads == 0
            || !d.is_multiple_of(self.num_heads)
            || !(d / self.num_heads).is_multiple_of(2)
        {
            return bad("hidden size must split into heads of even size");
        }
        if self.state_blocks == 0 || self.state_blocks > 256 || self.state_loops == 0 {
            return bad("block counts");
        }
        if self.exits.is_empty()
            || self.exits.windows(2).any(|w| w[0] >= w[1])
            || self.exits[0] == 0
            || *self.exits.last().expect("non-empty") != self.depth()
        {
            return bad("exits must increase from 1 and end at blocks x loops");
        }
        if self.patch == 0
            || self.patch > 1024
            || self.context_patches == 0
            || self.context_patches > 4096
            || self.horizon_patches == 0
            || self.horizon_patches > 256
        {
            return bad("patch, context or horizon size");
        }
        let q = &self.quantiles;
        if q.len() < 3
            || q.len().is_multiple_of(2)
            || q.windows(2).any(|w| w[0] >= w[1])
            || q[0] <= 0.0
            || q[q.len() - 1] >= 1.0
            || (q[q.len() / 2] - 0.5).abs() > 1e-6
        {
            return bad(
                "quantiles must be an odd, increasing list in (0, 1) with 0.5 in the middle",
            );
        }
        if self.decoder_layers == 0 || self.decoder_layers > 64 {
            return bad("decoder block count");
        }
        if self.exit_tolerance.is_nan() || self.exit_tolerance < 0.0 {
            return bad("exit_tolerance");
        }
        Ok(())
    }

    /// Read the configuration from GGUF metadata.
    pub fn from_gguf(g: &Gguf<'_>) -> Result<Self> {
        if g.architecture() != Some(ARCH) {
            return Err(Error::Format(format!(
                "not an s6-cast file (architecture {:?})",
                g.architecture()
            )));
        }
        let key = |k: &str| format!("{ARCH}.{k}");
        let int = |k: &str| g.get_u64(&key(k)).map(|v| v as usize);
        let req = |k: &str| int(k).ok_or_else(|| Error::Missing(format!("GGUF key {}", key(k))));
        let flag = |k: &str| g.get_bool(&key(k)).unwrap_or(false);
        let exits = g
            .get_array(&key("exits"))
            .and_then(|a| a.to_i64_vec())
            .ok_or_else(|| Error::Missing(key("exits")))?
            .into_iter()
            .map(|v| v.max(0) as usize)
            .collect();
        let quantiles = g
            .get_array(&key("quantiles"))
            .and_then(|a| a.to_f64_vec())
            .ok_or_else(|| Error::Missing(key("quantiles")))?
            .into_iter()
            .map(|v| v as f32)
            .collect();
        let d = req("embedding_length")?;
        let cfg = CastConfig {
            name: g.get_str("general.name").unwrap_or(ARCH).to_string(),
            hidden_size: d,
            intermediate_size: req("feed_forward_length")?,
            num_heads: req("head_count")?,
            state_blocks: req("state_block_count")?,
            state_loops: int("state_loops").unwrap_or(1),
            exits,
            parallax: flag("parallax"),
            xsa: flag("xsa"),
            patch: req("patch_length")?,
            context_patches: req("context_patches")?,
            horizon_patches: req("horizon_patches")?,
            quantiles,
            decoder_layers: req("decoder_block_count")?,
            rope_theta: g.get_f64(&key("rope_freq_base")).unwrap_or(10_000.0) as f32,
            norm_eps: g.get_f64(&key("layer_norm_epsilon")).unwrap_or(1e-5) as f32,
            exit_tolerance: g.get_f64(&key("exit_tolerance")).unwrap_or(0.0) as f32,
            weight_format: g
                .get_str(&key("weight_format"))
                .unwrap_or("f32")
                .to_string(),
        };
        cfg.validate()?;
        Ok(cfg)
    }
}
