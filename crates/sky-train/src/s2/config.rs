//! Configuration of the skycmd-s2 models (`configs/s2-*.json`), including the
//! calibrated temperatures and early-exit thresholds.

use std::{collections::BTreeMap, path::Path};

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};

use crate::{config::HEAD_DIM, num::NUM_FEATURES, num::NUM_ID, sequence::QTYPES};

/// GGUF architecture name of the s2 family.
pub const ARCH: &str = "skycmd-s2";

/// Full s2 model configuration.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct S2Config {
    /// Ladder name, e.g. `s2-pico`.
    pub name: String,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    /// Distinct state-encoder blocks (`state_block_count`).
    pub state_layers: usize,
    pub static_layers: usize,
    pub fusion_layers: usize,
    /// Exit positions, 1-based block applications of the state encoder
    /// (ascending, the last one equals `state_layers * state_loops`).
    pub exits: Vec<usize>,
    /// Attention heads (`head_count`); `None` means `hidden_size / 64`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub head_count: Option<usize>,
    /// SPEC-S3 §1: a `[NUM]` is merged into the preceding token, which then
    /// carries the value (`has_value`).
    #[serde(default, skip_serializing_if = "is_false")]
    pub fuse_numbers: bool,
    /// SPEC-S3 §2: the state blocks are applied in order `state_loops` times.
    #[serde(default = "default_loops", skip_serializing_if = "is_one")]
    pub state_loops: usize,
    /// SPEC-S3 §7: the state encoder of a frame starts from the previous
    /// frame's final latent through `carry_gate ⊙ LN(carry_norm)`.
    #[serde(default, skip_serializing_if = "is_false")]
    pub temporal_carry: bool,
    /// SPEC-S6 §1: Parallax attention in the encoder blocks. `attn_qkv` holds a fourth
    /// projection, the probe `ρ`, normalized per head by `attn_px_norm`.
    #[serde(default, skip_serializing_if = "is_false")]
    pub parallax: bool,
    /// SPEC-S6 §8: exclusive self-attention in the encoder blocks — each token's attention
    /// output loses its component along the token's own value.
    #[serde(default, skip_serializing_if = "is_false")]
    pub xsa: bool,
    /// SPEC-S3 §9 ablation: expand-and-sparsify FFN in the encoder blocks (see `s2::es`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ffn_es: Option<EsFfn>,
    #[serde(default = "default_max_state")]
    pub max_state_tokens: usize,
    #[serde(default = "default_max_head")]
    pub max_head_tokens: usize,
    #[serde(default = "default_theta")]
    pub rope_theta: f32,
    #[serde(default = "default_eps")]
    pub norm_eps: f32,
    /// Early-exit threshold τ per question type (choice, score, noul).
    #[serde(default = "default_tau")]
    pub exit_threshold: [f32; 3],
    /// Calibrated temperatures keyed `e{E}.{type}` or `e{E}.{type}.{bucket}`
    /// (bucket `2`, `3_5`, `6_10`, `11`), `E` being the 0-based exit index.
    #[serde(default)]
    pub temperature: BTreeMap<String, f32>,
    /// Run matmuls and activations in bfloat16 (runtime only).
    #[serde(skip)]
    pub bf16: bool,
    /// Amplitude `A` of the StableQAT surrogate gradient of the ternary fake quantization
    /// (runtime only; 0 = straight-through estimator). See `s2::model::ternarize_stable`.
    #[serde(skip)]
    pub qat_grad_a: f32,
}

/// Expand-and-sparsify FFN (SPEC-S3 §9): a fixed seeded sparse ±1 up-projection to
/// `expand · d` units, k-WTA keeping the top `k_frac`, a learned down-projection.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct EsFfn {
    /// Seed of the fixed projection (GGUF `ffn_es_seed`, u64).
    pub seed: u64,
    /// Expansion factor: `units = expand · hidden_size`.
    #[serde(default = "default_es_expand")]
    pub expand: usize,
    /// Fraction of the units kept active by k-WTA.
    #[serde(default = "default_es_k_frac")]
    pub k_frac: f32,
    /// Inputs read by each unit.
    #[serde(default = "default_es_fan_in")]
    pub fan_in: usize,
}

fn default_es_expand() -> usize {
    8
}
fn default_es_k_frac() -> f32 {
    0.05
}
fn default_es_fan_in() -> usize {
    8
}

impl EsFfn {
    /// Expansion units for hidden size `d`.
    pub fn units(&self, d: usize) -> usize {
        self.expand * d
    }

    /// Active units per token: `round(k_frac · units)`, at least 1.
    pub fn k(&self, d: usize) -> usize {
        ((self.k_frac * self.units(d) as f32).round() as usize).clamp(1, self.units(d))
    }
}

fn default_loops() -> usize {
    1
}
fn is_one(x: &usize) -> bool {
    *x == 1
}
fn is_false(x: &bool) -> bool {
    !*x
}
fn default_max_state() -> usize {
    64
}
fn default_max_head() -> usize {
    96
}
fn default_theta() -> f32 {
    10_000.0
}
fn default_eps() -> f32 {
    1e-5
}
fn default_tau() -> [f32; 3] {
    [0.9; 3]
}

/// Temperature bucket suffix of `k` options (SPEC.md): `2`, `3_5`, `6_10`, `11`.
pub fn bucket(k: usize) -> &'static str {
    match k {
        0..=2 => "2",
        3..=5 => "3_5",
        6..=10 => "6_10",
        _ => "11",
    }
}

impl S2Config {
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("reading config {}", path.display()))?;
        let cfg: Self = serde_json::from_str(&text)
            .with_context(|| format!("parsing config {}", path.display()))?;
        cfg.validate()?;
        Ok(cfg)
    }

    pub fn validate(&self) -> Result<()> {
        match self.head_count {
            None if self.hidden_size == 0
                || !self.hidden_size.is_multiple_of(HEAD_DIM as usize) =>
            {
                bail!(
                    "hidden_size {} is not a multiple of {HEAD_DIM} (set head_count)",
                    self.hidden_size
                );
            }
            Some(h)
                if h == 0
                    || !self.hidden_size.is_multiple_of(h)
                    || !(self.hidden_size / h).is_multiple_of(2) =>
            {
                bail!(
                    "head_count {h} must divide hidden_size {} into even head dims",
                    self.hidden_size
                );
            }
            _ => {}
        }
        if self.state_loops == 0 {
            bail!("state_loops must be >= 1");
        }
        if self.vocab_size <= NUM_ID as usize {
            bail!("vocab_size too small");
        }
        if self.state_layers == 0 || self.static_layers == 0 || self.fusion_layers == 0 {
            bail!("every stack needs at least one block");
        }
        if self.static_layers > self.state_layers {
            bail!(
                "static_layers must not exceed state_layers (the static encoder starts as a copy)"
            );
        }
        if self.exits.is_empty()
            || self.exits.windows(2).any(|w| w[0] >= w[1])
            || self.exits[0] == 0
            || *self.exits.last().unwrap() != self.depth()
        {
            bail!(
                "exits {:?} must be ascending, >= 1 and end at state_layers x state_loops = {}",
                self.exits,
                self.depth()
            );
        }
        if self.max_state_tokens < 3 {
            bail!("max_state_tokens too small");
        }
        if let Some(es) = &self.ffn_es {
            if es.expand == 0 || es.fan_in == 0 || es.fan_in > self.hidden_size {
                bail!("ffn_es: expand must be >= 1 and fan_in in 1..=hidden_size");
            }
            if !(es.k_frac > 0.0 && es.k_frac <= 1.0) {
                bail!("ffn_es: k_frac must be in (0, 1]");
            }
        }
        Ok(())
    }

    pub fn n_heads(&self) -> usize {
        self.head_count
            .unwrap_or(self.hidden_size / HEAD_DIM as usize)
    }

    /// `hidden_size / n_heads` (64 unless `head_count` says otherwise).
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.n_heads()
    }

    /// Effective state-encoder depth in block applications.
    pub fn depth(&self) -> usize {
        self.state_layers * self.state_loops
    }

    pub fn n_exits(&self) -> usize {
        self.exits.len()
    }

    /// Value features per `[NUM]` token.
    pub fn num_features(&self) -> usize {
        NUM_FEATURES
    }

    /// Deep-supervision weights: `linspace(0.5, 1.0)` over the exits, normalized.
    pub fn exit_weights(&self) -> Vec<f32> {
        let n = self.n_exits();
        let raw: Vec<f32> = (0..n)
            .map(|e| {
                if n == 1 {
                    1.0
                } else {
                    0.5 + 0.5 * e as f32 / (n - 1) as f32
                }
            })
            .collect();
        let s: f32 = raw.iter().sum();
        raw.into_iter().map(|w| w / s).collect()
    }

    /// Temperature of exit `e`, type `qtype` with `k` options: bucket, then
    /// type, then 1.
    pub fn temperature_of(&self, e: usize, qtype: usize, k: usize) -> f32 {
        let t = QTYPES[qtype];
        self.temperature
            .get(&format!("e{e}.{t}.{}", bucket(k)))
            .or_else(|| self.temperature.get(&format!("e{e}.{t}")))
            .copied()
            .unwrap_or(1.0)
    }

    /// Number of state block applications run before exit `e` (`layers_used`).
    pub fn layers_at(&self, e: usize) -> usize {
        self.exits[e]
    }

    /// A tiny configuration for unit tests.
    pub fn tiny(vocab_size: usize) -> Self {
        Self {
            name: "s2-tiny".into(),
            vocab_size,
            hidden_size: 128,
            intermediate_size: 90,
            state_layers: 2,
            static_layers: 1,
            fusion_layers: 1,
            exits: vec![1, 2],
            head_count: None,
            fuse_numbers: false,
            state_loops: 1,
            temporal_carry: false,
            parallax: false,
            xsa: false,
            ffn_es: None,
            max_state_tokens: 24,
            max_head_tokens: 40,
            rope_theta: 10_000.0,
            norm_eps: 1e-5,
            exit_threshold: [0.9; 3],
            temperature: BTreeMap::new(),
            bf16: false,
            qat_grad_a: 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    pub fn ladder(name: &str) -> S2Config {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../configs")
            .join(format!("{name}.json"));
        S2Config::load(path).unwrap()
    }

    #[test]
    fn ladder_matches_spec_table() {
        // name, d, state, static, fusion, ff, vocab, exits
        type Row = (
            &'static str,
            usize,
            usize,
            usize,
            usize,
            usize,
            usize,
            &'static [usize],
        );
        let table: [Row; 3] = [
            ("s2-pico", 64, 3, 2, 1, 128, 1024, &[1, 3]),
            ("s2-nano", 128, 4, 2, 2, 256, 2048, &[1, 2, 4]),
            ("s2-micro", 256, 6, 3, 2, 512, 4096, &[2, 4, 6]),
        ];
        for (name, d, st, sc, fu, ff, v, ex) in table {
            let c = ladder(name);
            assert_eq!(c.name, name);
            assert_eq!(
                (
                    c.hidden_size,
                    c.state_layers,
                    c.static_layers,
                    c.fusion_layers
                ),
                (d, st, sc, fu)
            );
            assert_eq!((c.intermediate_size, c.vocab_size), (ff, v));
            assert_eq!(c.exits, ex);
            assert_eq!((c.max_state_tokens, c.max_head_tokens), (64, 96));
            assert_eq!((c.rope_theta, c.norm_eps), (10_000.0, 1e-5));
        }
    }

    #[test]
    fn s3_ladder_matches_spec_table() {
        // name, d, heads, blocks, loops, static, fusion, ff, vocab, exits
        for (name, d, h, b, l, sc, fu, ff, v, ex) in [
            (
                "s3-pico",
                96,
                1,
                1,
                4,
                1,
                2,
                192,
                1024,
                &[1usize, 2, 3, 4][..],
            ),
            ("s3-nano", 128, 2, 2, 3, 2, 2, 256, 2048, &[2, 4, 6][..]),
        ] {
            let c = ladder(name);
            assert_eq!(c.name, name);
            assert_eq!((c.hidden_size, c.n_heads(), c.head_dim()), (d, h, d / h));
            assert_eq!((c.state_layers, c.state_loops, c.depth()), (b, l, b * l));
            assert_eq!((c.static_layers, c.fusion_layers), (sc, fu));
            assert_eq!((c.intermediate_size, c.vocab_size), (ff, v));
            assert_eq!(c.exits, ex);
            assert!(c.fuse_numbers);
            assert_eq!((c.max_state_tokens, c.max_head_tokens), (64, 96));
        }
        // s2 configs keep the defaults and serialize without the new keys.
        let c = ladder("s2-pico");
        assert_eq!(
            (c.state_loops, c.fuse_numbers, c.head_count, c.head_dim()),
            (1, false, None, 64)
        );
        let j = serde_json::to_value(&c).unwrap();
        assert!(
            j.get("state_loops").is_none()
                && j.get("fuse_numbers").is_none()
                && j.get("head_count").is_none()
        );
        // Exits must end at blocks x loops; head_count must divide d into even heads.
        let bad = S2Config {
            state_loops: 2,
            ..S2Config::tiny(300)
        };
        assert!(bad.validate().is_err());
        let ok = S2Config {
            state_loops: 2,
            exits: vec![2, 4],
            ..S2Config::tiny(300)
        };
        assert!(ok.validate().is_ok());
        assert!(S2Config {
            hidden_size: 96,
            ..S2Config::tiny(300)
        }
        .validate()
        .is_err());
        assert!(S2Config {
            hidden_size: 96,
            head_count: Some(1),
            ..S2Config::tiny(300)
        }
        .validate()
        .is_ok());
        assert!(S2Config {
            head_count: Some(3),
            ..S2Config::tiny(300)
        }
        .validate()
        .is_err());
    }

    #[test]
    fn exit_weights_and_temperatures() {
        let mut c = S2Config::tiny(300);
        let w = c.exit_weights();
        assert!((w[0] - 1.0 / 3.0).abs() < 1e-6 && (w[1] - 2.0 / 3.0).abs() < 1e-6);
        c.exits = vec![1, 2, 3];
        c.state_layers = 3;
        let w = c.exit_weights();
        assert!((w[0] - 0.5 / 2.25).abs() < 1e-6 && (w[1] - 0.75 / 2.25).abs() < 1e-6);
        c.temperature.insert("e1.choice".into(), 2.0);
        c.temperature.insert("e1.choice.3_5".into(), 3.0);
        assert_eq!(c.temperature_of(1, 0, 4), 3.0);
        assert_eq!(c.temperature_of(1, 0, 2), 2.0);
        assert_eq!(c.temperature_of(0, 0, 4), 1.0);
        assert_eq!(
            (bucket(2), bucket(5), bucket(6), bucket(12)),
            ("2", "3_5", "6_10", "11")
        );
        assert!(S2Config {
            exits: vec![2, 1],
            ..S2Config::tiny(300)
        }
        .validate()
        .is_err());
        assert!(S2Config {
            exits: vec![1],
            ..S2Config::tiny(300)
        }
        .validate()
        .is_err());
    }
}
