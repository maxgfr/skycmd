//! s6-cast configuration (`configs/s6-cast-*.json`, docs/SPEC-S6-CAST.md §1).

use std::path::Path;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::s2::config::S2Config;

/// An s6-cast model configuration.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CastConfig {
    pub name: String,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub head_count: usize,
    /// Distinct encoder blocks.
    pub state_layers: usize,
    #[serde(default = "one")]
    pub state_loops: usize,
    /// Exit positions in block applications (the last = `state_layers · state_loops`).
    pub exits: Vec<usize>,
    #[serde(default)]
    pub parallax: bool,
    #[serde(default)]
    pub xsa: bool,
    /// Points per patch.
    pub patch: usize,
    /// Longest context, in patches.
    pub context_patches: usize,
    /// Forecast of one pass, in patches.
    pub horizon_patches: usize,
    #[serde(default = "deciles")]
    pub quantiles: Vec<f32>,
    pub decoder_layers: usize,
    #[serde(default = "theta")]
    pub rope_theta: f32,
    #[serde(default = "eps")]
    pub norm_eps: f32,
    /// Early-exit tolerance (`cast-calibrate`); 0 runs every exit.
    #[serde(default)]
    pub exit_tolerance: f32,
    /// StableQAT amplitude of the ternary fake quantization (runtime only, as in s2).
    #[serde(skip)]
    pub qat_grad_a: f32,
}

fn one() -> usize {
    1
}
fn deciles() -> Vec<f32> {
    sky_infer::cast::DECILES.to_vec()
}
fn theta() -> f32 {
    10_000.0
}
fn eps() -> f32 {
    1e-5
}

impl CastConfig {
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let cfg: CastConfig = serde_json::from_str(
            &std::fs::read_to_string(path).with_context(|| format!("reading {path:?}"))?,
        )
        .with_context(|| format!("parsing {path:?}"))?;
        cfg.validate()?;
        Ok(cfg)
    }

    pub fn validate(&self) -> Result<()> {
        self.infer()
            .validate()
            .map_err(|e| anyhow::anyhow!("{}: {e}", self.name))
    }

    pub fn depth(&self) -> usize {
        self.state_layers * self.state_loops
    }

    pub fn n_exits(&self) -> usize {
        self.exits.len()
    }

    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.head_count
    }

    pub fn horizon(&self) -> usize {
        self.horizon_patches * self.patch
    }

    pub fn context(&self) -> usize {
        self.context_patches * self.patch
    }

    /// Deep-supervision weights over the exits (as s2: `linspace(0.5, 1)`, normalized).
    pub fn exit_weights(&self) -> Vec<f32> {
        self.encoder().exit_weights()
    }

    /// The state encoder as an s2 configuration: the s2 blocks, loops and exits run it.
    pub fn encoder(&self) -> S2Config {
        S2Config {
            name: self.name.clone(),
            vocab_size: crate::num::NUM_ID as usize + 1,
            hidden_size: self.hidden_size,
            intermediate_size: self.intermediate_size,
            state_layers: self.state_layers,
            static_layers: 1,
            fusion_layers: 1,
            exits: self.exits.clone(),
            head_count: Some(self.head_count),
            fuse_numbers: false,
            state_loops: self.state_loops,
            temporal_carry: false,
            parallax: self.parallax,
            xsa: self.xsa,
            ffn_es: None,
            max_state_tokens: self.context_patches,
            max_head_tokens: 8,
            rope_theta: self.rope_theta,
            norm_eps: self.norm_eps,
            exit_threshold: [1.0; 3],
            temperature: Default::default(),
            bf16: false,
            qat_grad_a: self.qat_grad_a,
        }
    }

    /// The same configuration as sky-infer reads it from a GGUF file.
    pub fn infer(&self) -> sky_infer::cast::CastConfig {
        sky_infer::cast::CastConfig {
            name: self.name.clone(),
            hidden_size: self.hidden_size,
            intermediate_size: self.intermediate_size,
            num_heads: self.head_count,
            state_blocks: self.state_layers,
            state_loops: self.state_loops,
            exits: self.exits.clone(),
            parallax: self.parallax,
            xsa: self.xsa,
            patch: self.patch,
            context_patches: self.context_patches,
            horizon_patches: self.horizon_patches,
            quantiles: self.quantiles.clone(),
            decoder_layers: self.decoder_layers,
            rope_theta: self.rope_theta,
            norm_eps: self.norm_eps,
            exit_tolerance: self.exit_tolerance,
            weight_format: "f32".into(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ladder_configs_load() {
        for name in ["s6-cast-pico", "s6-cast-femto"] {
            let path =
                Path::new(env!("CARGO_MANIFEST_DIR")).join(format!("../../configs/{name}.json"));
            let cfg = CastConfig::load(&path).unwrap();
            assert_eq!(cfg.name, name);
            assert_eq!(cfg.quantiles, sky_infer::cast::DECILES);
            assert_eq!(*cfg.exits.last().unwrap(), cfg.depth());
        }
    }
}
