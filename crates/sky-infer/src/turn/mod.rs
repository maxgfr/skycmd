//! s6-ear v2 (`skycmd-turn`, docs/s6-ear2-prereg.md §4): an end-of-turn detector.
//!
//! The last 4 s of 16 kHz audio become 400 hops of 43 channels ([`crate::audio::TurnFrontEnd`]:
//! 40 log10 mel values, log2 f0, voicing, energy), normalised per window by Whisper's rule and
//! stacked by `hops_per_position` into positions ([`crate::audio::normalize_window`]).
//! `patch_proj` maps each position to `d`; `block_count` encoder blocks (pre-LN, fused QKV, RoPE,
//! full attention, GELU FFN, ternary matrices) and a final norm run over the positions; Smart
//! Turn's attention pooling (`Linear(d, 64) → tanh → Linear(64, 1)`, a softmax over the
//! positions) gives one vector, and the head (`Linear(d, 64) → GELU → Linear(64, 1)`) the logit
//! of "the turn is finished". The probability is `σ(logit / temperature)`.
//!
//! The file stores the window, the hops per position, the channel count, the temperature fitted
//! on validation and the two validation thresholds τ₀.₅ and τ₈₅ (§7) as `skycmd-turn.*` keys.

pub mod model;
#[cfg(feature = "json")]
pub mod store;

use sky_gguf::Gguf;

use crate::audio::{TURN_CHANNELS, TURN_WINDOW_HOPS};
use crate::{Error, Result};

pub use model::{TurnModel, TurnOutput};

/// GGUF architecture name of s6-ear v2 files.
pub const ARCH: &str = "skycmd-turn";

/// Width of the teacher's vectors (Smart Turn v3.2's encoder, d 384).
pub const TEACHER_DIM: usize = 384;
/// Steps of the teacher's pooled encoder states (the last 4 s in 160 ms steps).
pub const TEACHER_STEPS: usize = 25;

/// An s6-ear v2 configuration (GGUF `skycmd-turn.*`).
#[derive(Clone, Debug, PartialEq)]
pub struct TurnConfig {
    pub name: String,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_heads: usize,
    pub blocks: usize,
    /// Hops of a decision window (400: 4.0 s).
    pub window_hops: usize,
    /// Hops stacked into one position (4: 40 ms).
    pub hops_per_position: usize,
    /// Channels of a hop (43).
    pub channels: usize,
    /// Hidden width of the attention pooling and of the head (64).
    pub pool_hidden: usize,
    pub head_hidden: usize,
    pub rope_theta: f32,
    pub norm_eps: f32,
    /// Fitted on validation (§7); 1 before calibration.
    pub temperature: f32,
    /// Decision thresholds on the calibrated probability, both from validation (§7).
    pub tau50: f32,
    pub tau85: f32,
    pub weight_format: String,
}

impl TurnConfig {
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_heads
    }

    /// Values of one position.
    pub fn input_width(&self) -> usize {
        self.hops_per_position * self.channels
    }

    /// Positions of a full window.
    pub fn max_positions(&self) -> usize {
        self.window_hops / self.hops_per_position
    }

    /// Reject configurations the engine cannot run.
    pub fn validate(&self) -> Result<()> {
        const MAX_DIM: usize = 1 << 14;
        let bad = |m: &str| Err(Error::Format(format!("invalid s6-ear v2 config: {m}")));
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
        if self.blocks == 0 || self.blocks > 64 {
            return bad("block count");
        }
        if self.channels != TURN_CHANNELS {
            return bad("the front end gives 43 channels per hop");
        }
        if self.window_hops == 0
            || self.window_hops > 4 * TURN_WINDOW_HOPS
            || self.hops_per_position == 0
            || !self.window_hops.is_multiple_of(self.hops_per_position)
        {
            return bad("window and hops per position");
        }
        if self.pool_hidden == 0 || self.head_hidden == 0 {
            return bad("pooling and head widths");
        }
        if !(self.temperature > 0.0 && self.temperature.is_finite()) {
            return bad("temperature");
        }
        if !(0.0..=1.0).contains(&self.tau50) || !(0.0..=1.0).contains(&self.tau85) {
            return bad("thresholds must be probabilities");
        }
        Ok(())
    }

    /// Read the configuration from GGUF metadata.
    pub fn from_gguf(g: &Gguf<'_>) -> Result<Self> {
        if g.architecture() != Some(ARCH) {
            return Err(Error::Format(format!(
                "not an s6-ear v2 file (architecture {:?})",
                g.architecture()
            )));
        }
        let key = |k: &str| format!("{ARCH}.{k}");
        let int = |k: &str| g.get_u64(&key(k)).map(|v| v as usize);
        let req = |k: &str| int(k).ok_or_else(|| Error::Missing(format!("GGUF key {}", key(k))));
        let float = |k: &str, def: f64| g.get_f64(&key(k)).unwrap_or(def) as f32;
        let cfg = TurnConfig {
            name: g.get_str("general.name").unwrap_or(ARCH).to_string(),
            hidden_size: req("embedding_length")?,
            intermediate_size: req("feed_forward_length")?,
            num_heads: req("head_count")?,
            blocks: req("block_count")?,
            window_hops: req("window_hops")?,
            hops_per_position: req("hops_per_position")?,
            channels: req("channel_count")?,
            pool_hidden: req("pool_length")?,
            head_hidden: req("head_length")?,
            rope_theta: float("rope_freq_base", 10_000.0),
            norm_eps: float("layer_norm_epsilon", 1e-5),
            temperature: float("temperature", 1.0),
            tau50: float("threshold_p50", 0.5),
            tau85: float("threshold_p85", 0.5),
            weight_format: g
                .get_str(&key("weight_format"))
                .unwrap_or("f32")
                .to_string(),
        };
        cfg.validate()?;
        Ok(cfg)
    }
}
