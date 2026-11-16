//! s6-ear v2 configuration (`configs/s6-ear2-*.json`, docs/s6-ear2-prereg.md §4.2–4.3).
//!
//! `configs/s6-ear2-nano.json` holds the shared fields and the four rungs of §4.3 under `rungs`;
//! the latency probe (§4.4) picks one, and a run is made with it (`--rung`). A config with the
//! shape fields at the top level (the femto config of the golden test) needs no rung.

use std::path::Path;

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// An s6-ear v2 model configuration (one rung).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TurnConfig {
    pub name: String,
    /// The rung of §4.3 this configuration is (`R1`–`R4`), if it came from the ladder.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rung: Option<String>,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub head_count: usize,
    pub blocks: usize,
    /// Hops stacked into one position: 4 (40 ms) or 8 (80 ms, R4).
    pub hops_per_position: usize,
    #[serde(default = "window")]
    pub window_hops: usize,
    #[serde(default = "channels")]
    pub channels: usize,
    #[serde(default = "sixty_four")]
    pub pool_hidden: usize,
    #[serde(default = "sixty_four")]
    pub head_hidden: usize,
    #[serde(default = "theta")]
    pub rope_theta: f32,
    #[serde(default = "eps")]
    pub norm_eps: f32,
    /// StableQAT amplitude of the ternary fake quantization (runtime only).
    #[serde(skip)]
    pub qat_grad_a: f32,
}

fn window() -> usize {
    sky_infer::audio::TURN_WINDOW_HOPS
}
fn channels() -> usize {
    sky_infer::audio::TURN_CHANNELS
}
fn sixty_four() -> usize {
    64
}
fn theta() -> f32 {
    10_000.0
}
fn eps() -> f32 {
    1e-5
}

/// The rungs of a ladder config, in order.
pub fn rungs(path: &Path) -> Result<Vec<String>> {
    let v: Value = serde_json::from_str(&std::fs::read_to_string(path)?)?;
    Ok(v["rungs"]
        .as_object()
        .map(|o| o.keys().cloned().collect())
        .unwrap_or_default())
}

impl TurnConfig {
    /// Loads a config; a ladder config needs `rung`.
    pub fn load(path: impl AsRef<Path>, rung: Option<&str>) -> Result<Self> {
        let path = path.as_ref();
        let mut v: Value = serde_json::from_str(
            &std::fs::read_to_string(path).with_context(|| format!("reading {path:?}"))?,
        )
        .with_context(|| format!("parsing {path:?}"))?;
        let ladder = v.get("rungs").cloned();
        if let Some(obj) = v.as_object_mut() {
            obj.remove("rungs");
            obj.remove("about");
        }
        match (ladder, rung) {
            (Some(l), Some(r)) => {
                let shape = l
                    .get(r)
                    .and_then(Value::as_object)
                    .with_context(|| format!("{path:?} has no rung {r}"))?;
                for (k, x) in shape {
                    v[k] = x.clone();
                }
                v["rung"] = Value::String(r.to_string());
            }
            (Some(_), None) => bail!("{path:?} is a ladder: choose a rung (§4.4's probe)"),
            (None, Some(r)) => bail!("{path:?} has no rungs, --rung {r} is meaningless"),
            (None, None) => {}
        }
        let cfg: TurnConfig =
            serde_json::from_value(v).with_context(|| format!("parsing {path:?}"))?;
        cfg.validate()?;
        Ok(cfg)
    }

    pub fn validate(&self) -> Result<()> {
        self.infer()
            .validate()
            .map_err(|e| anyhow::anyhow!("{}: {e}", self.name))?;
        let steps = sky_infer::turn::TEACHER_STEPS;
        if !self.max_positions().is_multiple_of(steps) {
            bail!(
                "{}: {} positions do not pool into the teacher's {steps} steps",
                self.name,
                self.max_positions()
            );
        }
        Ok(())
    }

    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.head_count
    }

    pub fn input_width(&self) -> usize {
        self.hops_per_position * self.channels
    }

    pub fn max_positions(&self) -> usize {
        self.window_hops / self.hops_per_position
    }

    /// Positions averaged into one of the teacher's 25 steps (4 at 40 ms, 2 at 80 ms).
    pub fn positions_per_step(&self) -> usize {
        self.max_positions() / sky_infer::turn::TEACHER_STEPS
    }

    /// The same configuration as sky-infer reads it (temperature and thresholds before
    /// calibration).
    pub fn infer(&self) -> sky_infer::turn::TurnConfig {
        sky_infer::turn::TurnConfig {
            name: self.name.clone(),
            hidden_size: self.hidden_size,
            intermediate_size: self.intermediate_size,
            num_heads: self.head_count,
            blocks: self.blocks,
            window_hops: self.window_hops,
            hops_per_position: self.hops_per_position,
            channels: self.channels,
            pool_hidden: self.pool_hidden,
            head_hidden: self.head_hidden,
            rope_theta: self.rope_theta,
            norm_eps: self.norm_eps,
            temperature: 1.0,
            tau50: 0.5,
            tau85: 0.5,
            weight_format: "f32".into(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn path(n: &str) -> std::path::PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join(format!("../../configs/{n}.json"))
    }

    #[test]
    fn the_ladder_and_the_femto_config_load() {
        let p = path("s6-ear2-nano");
        assert_eq!(rungs(&p).unwrap(), ["R1", "R2", "R3", "R4"]);
        assert!(TurnConfig::load(&p, None).is_err());
        assert!(TurnConfig::load(&p, Some("R9")).is_err());
        let r4 = TurnConfig::load(&p, Some("R4")).unwrap();
        assert_eq!(
            (r4.rung.as_deref(), r4.hops_per_position, r4.max_positions()),
            (Some("R4"), 8, 50)
        );
        assert_eq!(r4.positions_per_step(), 2);
        let r1 = TurnConfig::load(&p, Some("R1")).unwrap();
        assert_eq!(
            (r1.hidden_size, r1.blocks, r1.positions_per_step()),
            (192, 6, 4)
        );
        let f = TurnConfig::load(path("s6-ear2-femto"), None).unwrap();
        assert_eq!(f.rung, None);
        // a saved run config has no rungs and loads as it is
        let text = serde_json::to_string(&r1).unwrap();
        assert!(text.contains("\"rung\":\"R1\"") && !text.contains("rungs"));
    }
}
