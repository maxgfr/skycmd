//! `skycmd-cast` GGUF export and import (docs/SPEC-S6-CAST.md §4): the s2 tensor encodings
//! (ternary `.tq` + f16 `.scale`, F16 or F32 floats) and `skycmd-cast.*` metadata.

use anyhow::{anyhow, bail, Result};
use mlx_rs::Array;
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
    config::CastConfig,
    model::{is_ternary, param_shapes},
};

pub use sky_infer::cast::ARCH;

fn f16_bytes(x: &[f32]) -> Vec<u8> {
    x.iter()
        .flat_map(|v| f32_to_f16(*v).to_le_bytes())
        .collect()
}

/// The GGUF bytes of a model in `format`.
pub fn to_gguf(p: &ParamMap, cfg: &CastConfig, format: WeightFormat) -> Result<Vec<u8>> {
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
    w.add_kv(k("state_block_count"), u(cfg.state_layers));
    w.add_kv(k("state_loops"), u(cfg.state_loops));
    w.add_kv(
        k("exits"),
        OwnedValue::Array(cfg.exits.iter().map(|&e| u(e)).collect()),
    );
    if cfg.parallax {
        w.add_kv(k("parallax"), true);
    }
    if cfg.xsa {
        w.add_kv(k("xsa"), true);
    }
    w.add_kv(k("patch_length"), u(cfg.patch));
    w.add_kv(k("context_patches"), u(cfg.context_patches));
    w.add_kv(k("horizon_patches"), u(cfg.horizon_patches));
    w.add_kv(
        k("quantiles"),
        OwnedValue::Array(cfg.quantiles.iter().map(|&q| OwnedValue::F32(q)).collect()),
    );
    w.add_kv(k("decoder_block_count"), u(cfg.decoder_layers));
    w.add_kv(k("rope_freq_base"), OwnedValue::F32(cfg.rope_theta));
    w.add_kv(k("layer_norm_epsilon"), OwnedValue::F32(cfg.norm_eps));
    w.add_kv(k("exit_tolerance"), OwnedValue::F32(cfg.exit_tolerance));
    w.add_kv(k("weight_format"), format.name());
    for (name, shape) in param_shapes(cfg) {
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

/// The configuration of an `skycmd-cast` file.
pub fn config_from_gguf(g: &Gguf<'_>) -> Result<CastConfig> {
    let c = sky_infer::cast::CastConfig::from_gguf(g)?;
    Ok(CastConfig {
        name: c.name,
        hidden_size: c.hidden_size,
        intermediate_size: c.intermediate_size,
        head_count: c.num_heads,
        state_layers: c.state_blocks,
        state_loops: c.state_loops,
        exits: c.exits,
        parallax: c.parallax,
        xsa: c.xsa,
        patch: c.patch,
        context_patches: c.context_patches,
        horizon_patches: c.horizon_patches,
        quantiles: c.quantiles,
        decoder_layers: c.decoder_layers,
        rope_theta: c.rope_theta,
        norm_eps: c.norm_eps,
        exit_tolerance: c.exit_tolerance,
        qat_grad_a: 0.0,
    })
}

/// The parameters exactly as stored (ternary codes times their f16 scales).
pub fn params_from_gguf(g: &Gguf<'_>, cfg: &CastConfig) -> Result<ParamMap> {
    let mut out = ParamMap::new();
    for (name, shape) in param_shapes(cfg) {
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
    fn round_trip() {
        let _g = test_util::lock();
        let cfg = CastConfig::load(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../configs/s6-cast-femto.json"),
        )
        .unwrap();
        let p = super::super::model::init(&cfg, 3).unwrap();
        let bytes = to_gguf(&p, &cfg, WeightFormat::F32).unwrap();
        let g = Gguf::parse(&bytes).unwrap();
        assert_eq!(config_from_gguf(&g).unwrap(), cfg);
        let back = params_from_gguf(&g, &cfg).unwrap();
        for (k, v) in &p {
            assert_eq!(host_f32(v).unwrap(), host_f32(&back[k]).unwrap(), "{k}");
        }
        // the ternary file is smaller and loads in sky-infer
        let t = to_gguf(&p, &cfg, WeightFormat::Ternary).unwrap();
        assert!(t.len() < bytes.len() / 3, "{} vs {}", t.len(), bytes.len());
        let m = sky_infer::cast::CastModel::from_bytes(&t).unwrap();
        let mut want = cfg.infer();
        want.weight_format = "ternary".into();
        assert_eq!(m.cfg, want);
    }
}
