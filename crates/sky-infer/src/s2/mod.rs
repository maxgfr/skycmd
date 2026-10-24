//! The `skycmd-s2` architecture (docs/SPEC-S2.md): a static stream (question + options, encoded
//! once and cached), a question-agnostic state stream with early exits, a cross-attention fusion
//! head, native numbers (`[NUM]` tokens with value features) and optional ternary weights.
//! The s3 options of docs/SPEC-S3.md live in the same format: value-fused tokens
//! (`fuse_numbers`), a looped weight-shared state encoder (`state_loops`), any `head_count` and
//! the temporal carry of a stream's latent across frames (`temporal_carry`).
//!
//! * [`S2Config`]: hyper-parameters read from the GGUF keys `skycmd-s2.*`.
//! * [`S2Model`]: weights and the forward pieces (static encoding, state layers, exits, fusion).
//! * [`S2Engine`]: tokenization with numeric extraction, the static LRU cache, the state cache
//!   for multi-question frames and the early-exit loop.
//! * [`random`]: ladder configurations and random-weight GGUF files for tests and benchmarks.

pub mod engine;
pub mod es;
pub mod model;
pub mod prompt;
pub mod random;

pub use engine::{S2Engine, S2Options, TernaryCompute};
pub use model::S2Model;

use sky_gguf::{Gguf, Value};

use crate::{Error, Result};

/// `general.architecture` of s2 files.
pub const ARCH: &str = "skycmd-s2";

/// Default early-exit threshold τ of a question type.
pub const DEFAULT_EXIT_THRESHOLD: f32 = 0.9;

/// Hyper-parameters of an s2 model (SPEC-S2 §3 and §5).
#[derive(Clone, Debug, PartialEq)]
pub struct S2Config {
    pub name: String,
    pub vocab_size: usize,
    /// `d`.
    pub hidden_size: usize,
    /// `ff` of the encoder blocks (the fusion feed-forward is `2·d`).
    pub intermediate_size: usize,
    pub num_heads: usize,
    /// Effective state-encoder depth: block applications `state_blocks × state_loops` (the depth
    /// that exits and `layers_used` count).
    pub state_layers: usize,
    /// Distinct state blocks `blk.0..blk.{B-1}` (GGUF `state_block_count`).
    pub state_blocks: usize,
    /// Times the state blocks are applied in sequence (GGUF `state_loops`, default 1; SPEC-S3 §2).
    pub state_loops: usize,
    /// Merge each `[NUM]` into the token before it (GGUF `fuse_numbers`, default false; SPEC-S3 §1).
    pub fuse_numbers: bool,
    /// Carry the final state latent of a stream's previous frame into the next one (GGUF
    /// `temporal_carry`, default false; SPEC-S3 §7).
    pub temporal_carry: bool,
    /// Parallax attention in the encoder blocks: `attn_qkv` holds a fourth projection, the probe
    /// `ρ`, normalized per head by `attn_px_norm` (GGUF `parallax`, default false; SPEC-S6 §1).
    pub parallax: bool,
    /// Expand-and-sparsify FFN in the encoder blocks (GGUF `ffn_es_*`, absent by default;
    /// SPEC-S3 §9 ablation, see [`es`]).
    pub ffn_es: Option<es::EsSpec>,
    pub static_layers: usize,
    pub fusion_layers: usize,
    /// State-encoder depth (1-based block count) after which each exit sits, increasing; the
    /// last one equals `state_layers`.
    pub exits: Vec<usize>,
    /// Longest state stream `[CLS] state [SEP]`.
    pub max_state_tokens: usize,
    /// Token budget of the question and options (SPEC.md truncation rules).
    pub max_head_tokens: usize,
    /// Size of `φ(v)` (17).
    pub num_features: usize,
    /// Id of the `[NUM]` token (5).
    pub num_token_id: u32,
    pub rope_theta: f32,
    pub norm_eps: f32,
    /// τ per question type (choice, score, noul).
    pub exit_thresholds: [f32; 3],
    /// Temperatures keyed `e{E}.{type}[.{bucket}]` (prefix `skycmd-s2.temperature.` removed).
    pub temperatures: Vec<(String, f32)>,
    /// `f32`, `f16` or `ternary` (as declared by the file).
    pub weight_format: String,
}

impl S2Config {
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_heads.max(1)
    }

    /// Temperature of exit `exit` for a question type and option-count bucket: `e{E}.{t}.{b}`,
    /// then `e{E}.{t}`, then the exit-agnostic `{t}.{b}` and `{t}`, else 1.0.
    pub fn temperature(&self, exit: usize, type_name: &str, bucket: &str) -> f32 {
        let keys = [
            format!("e{exit}.{type_name}.{bucket}"),
            format!("e{exit}.{type_name}"),
            format!("{type_name}.{bucket}"),
            type_name.to_string(),
        ];
        for key in &keys {
            if let Some((_, t)) = self.temperatures.iter().find(|(k, _)| k == key) {
                if *t > 0.0 && t.is_finite() {
                    return *t;
                }
            }
        }
        1.0
    }

    /// Reject configurations the engine cannot run.
    pub fn validate(&self) -> Result<()> {
        const MAX_DIM: usize = 1 << 16;
        let bad = |m: &str| Err(Error::Format(format!("invalid s2 config: {m}")));
        let d = self.hidden_size;
        if d == 0 || d > MAX_DIM || self.vocab_size == 0 || self.vocab_size > 1 << 24 {
            return bad("hidden or vocab size");
        }
        if self.num_heads == 0
            || !d.is_multiple_of(self.num_heads)
            || !(d / self.num_heads).is_multiple_of(2)
        {
            return bad("hidden size must split into heads of even size");
        }
        if self.intermediate_size == 0 || self.intermediate_size > MAX_DIM {
            return bad("feed-forward size");
        }
        if self.state_blocks == 0 || self.state_blocks > 256 || self.static_layers > 256 {
            return bad("block counts");
        }
        if self.state_loops == 0 || self.state_loops > 256 {
            return bad("state_loops");
        }
        if self.state_layers != self.state_blocks * self.state_loops {
            return bad("state depth must be state_block_count x state_loops");
        }
        if self.fusion_layers > 256 {
            return bad("fusion block count");
        }
        if self.exits.is_empty()
            || self.exits.windows(2).any(|w| w[0] >= w[1])
            || self.exits[0] == 0
            || *self.exits.last().expect("non-empty") != self.state_layers
        {
            return bad("exits must increase from 1 and end at state_block_count x state_loops");
        }
        if self.num_features != crate::num::NUM_FEATURES {
            return bad("num_features must be 17");
        }
        if self.max_state_tokens < 2 || self.max_state_tokens > MAX_DIM {
            return bad("max_state_tokens");
        }
        if self.max_head_tokens == 0 || self.max_head_tokens > MAX_DIM {
            return bad("max_head_tokens");
        }
        if let Some(es) = &self.ffn_es {
            if es.units == 0
                || es.units > MAX_DIM
                || es.k == 0
                || es.k > es.units
                || es.fan_in == 0
                || es.fan_in > d
            {
                return bad("ffn_es units / k / fan_in");
            }
        }
        Ok(())
    }

    /// Read the configuration from GGUF metadata (`skycmd-s2.*`). The vocabulary size comes from
    /// the token embedding.
    pub fn from_gguf(g: &Gguf<'_>) -> Result<Self> {
        let key = |k: &str| format!("{ARCH}.{k}");
        let int = |k: &str| -> Option<usize> {
            match g.get(&key(k))? {
                Value::Array(a) => a.iter().next().and_then(|v| v.as_u64()).map(|v| v as usize),
                v => v.as_u64().map(|v| v as usize),
            }
        };
        let req = |k: &str| int(k).ok_or_else(|| Error::Missing(format!("GGUF key {}", key(k))));
        let exits: Vec<usize> = g
            .get_array(&key("exits"))
            .and_then(|a| a.to_i64_vec())
            .ok_or_else(|| Error::Missing(key("exits")))?
            .into_iter()
            .map(|v| v.max(0) as usize)
            .collect();
        let d = req("embedding_length")?;
        let state_blocks = req("state_block_count")?;
        let state_loops = int("state_loops").unwrap_or(1);
        let flag = |k: &str| {
            g.get(&key(k))
                .and_then(|v| v.as_bool().or_else(|| v.as_i64().map(|i| i != 0)))
                .unwrap_or(false)
        };
        let fuse_numbers = flag("fuse_numbers");
        let temporal_carry = flag("temporal_carry");
        let parallax = flag("parallax");
        let ffn_es = match g.get(&key("ffn_es_seed")).and_then(|v| v.as_u64()) {
            None => None,
            Some(seed) => Some(es::EsSpec {
                seed,
                units: req("ffn_es_units")?,
                k: req("ffn_es_k")?,
                fan_in: req("ffn_es_fan_in")?,
            }),
        };
        let vocab_size = ["token_embd.weight", "token_embd.q8"]
            .iter()
            .find_map(|n| g.tensor(n))
            .and_then(|t| t.shape().first().copied())
            .ok_or_else(|| Error::Missing("tensor token_embd".into()))?;
        let tau = |t: &str| {
            g.get_f64(&key(&format!("exit_threshold.{t}")))
                .map_or(DEFAULT_EXIT_THRESHOLD, |v| v as f32)
        };
        let tprefix = key("temperature.");
        let temperatures = g
            .kvs
            .iter()
            .filter_map(|(k, v)| {
                let name = k.strip_prefix(tprefix.as_str())?;
                Some((name.to_string(), v.as_f64()? as f32))
            })
            .collect();
        let cfg = S2Config {
            name: g.get_str("general.name").unwrap_or(ARCH).to_string(),
            vocab_size,
            hidden_size: d,
            intermediate_size: req("feed_forward_length")?,
            num_heads: int("head_count").unwrap_or((d / 64).max(1)),
            state_layers: state_blocks.saturating_mul(state_loops),
            state_blocks,
            state_loops,
            fuse_numbers,
            temporal_carry,
            parallax,
            ffn_es,
            static_layers: req("static_block_count")?,
            fusion_layers: req("fusion_block_count")?,
            exits,
            max_state_tokens: int("max_state_tokens").unwrap_or(64),
            max_head_tokens: int("max_head_tokens").unwrap_or(96),
            num_features: int("num_features").unwrap_or(crate::num::NUM_FEATURES),
            num_token_id: int("num_token_id").unwrap_or(5) as u32,
            rope_theta: g.get_f64(&key("rope_freq_base")).unwrap_or(10_000.0) as f32,
            norm_eps: g.get_f64(&key("layer_norm_epsilon")).unwrap_or(1e-5) as f32,
            exit_thresholds: [tau("choice"), tau("score"), tau("noul")],
            temperatures,
            weight_format: g
                .get_str(&key("weight_format"))
                .unwrap_or("f32")
                .to_string(),
        };
        cfg.validate()?;
        Ok(cfg)
    }
}
