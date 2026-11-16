//! `skycmd-turn` GGUF export and import (docs/s6-ear2-prereg.md §4.3, §7): the s2 tensor encodings
//! (ternary `.tq` + f16 `.scale` for the encoder blocks, F16 or F32 floats) and the
//! `skycmd-turn.*` keys: the window, the hops per position, the channel count, the temperature and
//! the two thresholds. The training-only projections are not exported.

use anyhow::{anyhow, bail, Result};
use mlx_rs::Array;
use serde::{Deserialize, Serialize};
use sky_gguf::{f16_to_f32, f32_to_f16, GgmlType, Gguf, GgufWriter, OwnedValue};

use crate::{
    model::host_f32,
    params::ParamMap,
    s2::{
        gguf::{pack_ternary, unpack_ternary, WeightFormat},
        model::ternary_codes,
    },
};

use super::{
    config::TurnConfig,
    model::{is_exported, is_ternary, param_shapes},
};

pub use sky_infer::turn::ARCH;

/// The calibration of §7, stored in the file.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Calibration {
    /// Fitted on validation by NLL.
    pub temperature: f32,
    /// 0.5 on the calibrated probability.
    pub tau50: f32,
    /// The highest validation threshold whose END recall is at least 0.85.
    pub tau85: f32,
}

impl Default for Calibration {
    fn default() -> Self {
        Calibration {
            temperature: 1.0,
            tau50: 0.5,
            tau85: 0.5,
        }
    }
}

fn f16_bytes(x: &[f32]) -> Vec<u8> {
    x.iter()
        .flat_map(|v| f32_to_f16(*v).to_le_bytes())
        .collect()
}

/// The GGUF bytes of a model in `format`.
pub fn to_gguf(
    p: &ParamMap,
    cfg: &TurnConfig,
    format: WeightFormat,
    calib: Calibration,
) -> Result<Vec<u8>> {
    cfg.validate()?;
    let mut w = GgufWriter::new();
    w.add_kv("general.architecture", ARCH);
    w.add_kv("general.type", "model");
    w.add_kv("general.name", cfg.name.as_str());
    let k = |s: &str| format!("{ARCH}.{s}");
    let u = |x: usize| OwnedValue::U32(x as u32);
    w.add_kv(k("embedding_length"), u(cfg.hidden_size));
    w.add_kv(k("feed_forward_length"), u(cfg.intermediate_size));
    w.add_kv(k("head_count"), u(cfg.head_count));
    w.add_kv(k("block_count"), u(cfg.blocks));
    w.add_kv(k("window_hops"), u(cfg.window_hops));
    w.add_kv(k("hops_per_position"), u(cfg.hops_per_position));
    w.add_kv(k("channel_count"), u(cfg.channels));
    w.add_kv(k("pool_length"), u(cfg.pool_hidden));
    w.add_kv(k("head_length"), u(cfg.head_hidden));
    w.add_kv(k("rope_freq_base"), OwnedValue::F32(cfg.rope_theta));
    w.add_kv(k("layer_norm_epsilon"), OwnedValue::F32(cfg.norm_eps));
    w.add_kv(k("temperature"), OwnedValue::F32(calib.temperature));
    w.add_kv(k("threshold_p50"), OwnedValue::F32(calib.tau50));
    w.add_kv(k("threshold_p85"), OwnedValue::F32(calib.tau85));
    w.add_kv(k("weight_format"), format.name());
    if let Some(r) = &cfg.rung {
        w.add_kv(k("rung"), r.as_str());
    }
    for (name, shape) in param_shapes(cfg)
        .into_iter()
        .filter(|(n, _)| is_exported(n))
    {
        let a = p
            .get(name.as_str())
            .ok_or_else(|| anyhow!("missing parameter {name}"))?;
        if a.shape() != shape.as_slice() {
            bail!("{name}: shape {:?}, expected {shape:?}", a.shape());
        }
        let values = host_f32(a)?;
        let dims: Vec<u64> = shape.iter().rev().map(|&d| d as u64).collect();
        let base = name.strip_suffix(".weight").unwrap_or(&name);
        match format {
            WeightFormat::Ternary if is_ternary(&name) => {
                let (rows, cols) = (shape[0] as usize, shape[1] as usize);
                let (codes, scales) = ternary_codes(&values, rows, cols);
                w.add_tensor(
                    format!("{base}.tq"),
                    &[cols.div_ceil(4) as u64, rows as u64],
                    GgmlType::I8,
                    pack_ternary(&codes, rows, cols),
                )?;
                w.add_tensor(
                    format!("{base}.scale"),
                    &[rows as u64],
                    GgmlType::F16,
                    f16_bytes(&scales),
                )?;
            }
            WeightFormat::F16 if shape.len() == 2 => {
                w.add_tensor_f32(name.as_str(), &dims, GgmlType::F16, &values)?;
            }
            _ => {
                w.add_tensor_f32(name.as_str(), &dims, GgmlType::F32, &values)?;
            }
        }
    }
    Ok(w.to_bytes()?)
}

/// The configuration and calibration of an `skycmd-turn` file.
pub fn config_from_gguf(g: &Gguf<'_>) -> Result<(TurnConfig, Calibration)> {
    let c = sky_infer::turn::TurnConfig::from_gguf(g)?;
    Ok((
        TurnConfig {
            name: c.name,
            rung: g.get_str(&format!("{ARCH}.rung")).map(str::to_string),
            hidden_size: c.hidden_size,
            intermediate_size: c.intermediate_size,
            head_count: c.num_heads,
            blocks: c.blocks,
            hops_per_position: c.hops_per_position,
            window_hops: c.window_hops,
            channels: c.channels,
            pool_hidden: c.pool_hidden,
            head_hidden: c.head_hidden,
            rope_theta: c.rope_theta,
            norm_eps: c.norm_eps,
            qat_grad_a: 0.0,
        },
        Calibration {
            temperature: c.temperature,
            tau50: c.tau50,
            tau85: c.tau85,
        },
    ))
}

/// The exported parameters exactly as stored (ternary codes times their f16 scales).
pub fn params_from_gguf(g: &Gguf<'_>, cfg: &TurnConfig) -> Result<ParamMap> {
    let mut out = ParamMap::new();
    for (name, shape) in param_shapes(cfg)
        .into_iter()
        .filter(|(n, _)| is_exported(n))
    {
        let base = name.strip_suffix(".weight").unwrap_or(&name);
        let values = if let Some(t) = g.tensor(&format!("{base}.tq")) {
            let (rows, cols) = (shape[0] as usize, shape[1] as usize);
            let codes = unpack_ternary(g.tensor_bytes(t), rows, cols)?;
            let st = g
                .tensor(&format!("{base}.scale"))
                .ok_or_else(|| anyhow!("missing {base}.scale"))?;
            let scales: Vec<f32> = g
                .tensor_bytes(st)
                .as_chunks::<2>()
                .0
                .iter()
                .map(|b| f16_to_f32(u16::from_le_bytes(*b)))
                .collect();
            codes
                .iter()
                .enumerate()
                .map(|(i, &c)| c as f32 * scales[i / cols])
                .collect()
        } else {
            g.tensor_f32(&name)?
        };
        out.insert(name.into(), Array::from_slice(&values, &shape));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_util;

    #[test]
    fn round_trip_and_the_calibration_keys() {
        let _g = test_util::lock();
        mlx_rs::with_device(mlx_rs::Device::cpu(), || {
            let cfg = TurnConfig::load(
                std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                    .join("../../configs/s6-ear2-femto.json"),
                None,
            )
            .unwrap();
            let p = super::super::model::init(&cfg, 3).unwrap();
            let cal = Calibration {
                temperature: 1.7,
                tau50: 0.5,
                tau85: 0.31,
            };
            let bytes = to_gguf(&p, &cfg, WeightFormat::F32, cal).unwrap();
            let g = Gguf::parse(&bytes).unwrap();
            assert_eq!(config_from_gguf(&g).unwrap(), (cfg.clone(), cal));
            let back = params_from_gguf(&g, &cfg).unwrap();
            for (k, v) in back.iter() {
                assert_eq!(host_f32(v).unwrap(), host_f32(&p[k]).unwrap(), "{k}");
            }
            assert!(g.tensor(super::super::model::POOL_PROJ).is_none());
            let t = to_gguf(&p, &cfg, WeightFormat::Ternary, cal).unwrap();
            let mut m = sky_infer::turn::TurnModel::from_bytes(&t).unwrap();
            assert_eq!(m.n_params(), super::super::model::exported_count(&cfg));
            assert_eq!((m.cfg.temperature, m.cfg.tau85), (1.7, 0.31));
            assert_eq!(m.cfg.weight_format, "ternary");
            assert!(Gguf::parse(&t).unwrap().tensor("blk.0.ffn_up.tq").is_some());
        });
    }
}
