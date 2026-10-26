//! Golden files of s6-cast (docs/SPEC-S6-CAST.md §5): random weights of a config's shape, saved
//! as f32 and ternary GGUF files, and the normalized quantiles of every exit computed by the
//! mlx-rs graph on the weights exactly as stored. sky-infer's `tests/golden_cast.rs` must
//! reproduce them from the contexts alone.

use std::path::Path;

use anyhow::Result;
use mlx_rs::{random, Array};
use serde_json::{json, Value};
use sky_gguf::Gguf;
use sky_infer::cast::synth::{self, Kind};

use super::{
    config::CastConfig,
    data::{batch, window},
    gguf::{config_from_gguf, params_from_gguf, to_gguf},
    model::{forward, init},
};
use crate::{model::host_f32, params::ParamMap, s2::gguf::WeightFormat};

/// Random weights large enough that every path matters (norms and biases randomized too).
fn golden_params(cfg: &CastConfig, seed: u64) -> Result<ParamMap> {
    let base = init(cfg, seed)?;
    let mut key = random::key(seed ^ 0xca57)?;
    let mut names: Vec<_> = base.keys().cloned().collect();
    names.sort();
    let mut out = ParamMap::new();
    for name in names {
        let shape = base[&name].shape().to_vec();
        let (k, next) = random::split(&key, 2)?;
        key = next;
        let n = random::normal::<f32>(&shape[..], 0.0, 1.0, &k)?;
        let a = if name.as_ref() == "quant_head.bias" {
            base[&name].add(n.multiply(Array::from_f32(0.5))?)?
        } else if name.ends_with(".bias") {
            n.multiply(Array::from_f32(0.1))?
        } else if name.contains("norm") {
            n.multiply(Array::from_f32(0.2))?
                .add(Array::from_f32(1.0))?
        } else if shape.len() == 1 {
            n.multiply(Array::from_f32(0.5))?
        } else if name.as_ref() == "cast_query.weight" || name.as_ref() == "loop_embd.weight" {
            n
        } else {
            n.multiply(Array::from_f32(1.0 / (shape[1] as f32).sqrt()))?
        };
        out.insert(name, a);
    }
    crate::params::eval(&out)?;
    Ok(out)
}

/// The contexts of the golden cases: every generator family, a context shorter than one small
/// attention block, one longer (the tiled kernel), missing values, and a constant series.
fn cases(cfg: &CastConfig) -> Vec<(String, Vec<f32>)> {
    let p = cfg.patch;
    let full = cfg.context();
    let mut out: Vec<(String, Vec<f32>)> = Vec::new();
    for (i, kind) in Kind::ALL.into_iter().enumerate() {
        let len = [full, 3 * p + 5, 20 * p, full - 1][i];
        out.push((
            kind.name().to_string(),
            synth::series(kind, 100 + i as u64, len),
        ));
    }
    let mut gaps = synth::series(Kind::Tsi, 7, 12 * p);
    for (j, v) in gaps.iter_mut().enumerate() {
        if j % 5 == 2 || (3 * p..4 * p).contains(&j) {
            *v = f32::NAN;
        }
    }
    out.push(("missing".into(), gaps));
    out.push(("constant".into(), vec![4.25; 6 * p]));
    // longer than the model's context, with a trend: normalization must use the points read
    let long: Vec<f32> = (0..3 * full)
        .map(|t| t as f32 * 0.5 + (t as f32 * 0.7).sin())
        .collect();
    out.push(("longer-than-context".into(), long));
    out
}

fn f32s(v: &[f32]) -> Value {
    json!(v
        .iter()
        .map(|x| if x.is_finite() { json!(x) } else { Value::Null })
        .collect::<Vec<_>>())
}

/// Writes `{prefix}-f32.gguf`, `{prefix}-ternary.gguf` and `{prefix}.json` into `out_dir`.
pub fn golden(cfg: &CastConfig, out_dir: &Path, prefix: &str, seed: u64) -> Result<()> {
    std::fs::create_dir_all(out_dir)?;
    let params = golden_params(cfg, seed)?;
    let all = cases(cfg);
    let mut files = Vec::new();
    let mut outputs: Vec<Vec<Value>> = vec![Vec::new(); 2];
    for (fi, format) in [WeightFormat::F32, WeightFormat::Ternary]
        .into_iter()
        .enumerate()
    {
        let bytes = to_gguf(&params, cfg, format)?;
        let file = format!("{prefix}-{}.gguf", format.name());
        std::fs::write(out_dir.join(&file), &bytes)?;
        let g = Gguf::parse(&bytes)?;
        let gcfg = config_from_gguf(&g)?;
        let p = params_from_gguf(&g, &gcfg)?;
        for (_, ctx) in &all {
            let w = window(&gcfg, ctx, &[]).expect("golden contexts have observations");
            let q = forward(&p, &gcfg, &batch(&gcfg, &[w]), false)?;
            let q = host_f32(&q)?;
            let per_exit = q.len() / gcfg.n_exits();
            let exits: Vec<Value> = q.chunks(per_exit).map(|c| json!(c)).collect();
            outputs[fi].push(json!(exits));
        }
        files.push(json!({"format": format.name(), "path": file, "bytes": bytes.len()}));
    }
    let cases: Vec<Value> = all
        .iter()
        .enumerate()
        .map(|(i, (name, ctx))| {
            json!({
                "name": name,
                "context": f32s(ctx),
                "quantiles": {"f32": outputs[0][i], "ternary": outputs[1][i]},
            })
        })
        .collect();
    let doc = json!({
        "meta": {
            "config": cfg,
            "files": files,
            "tolerance": {"f32": 1e-4, "ternary": 1e-3},
            "about": "normalized quantiles [exit][horizon x levels] of sky-train's mlx-rs forward on the stored weights",
        },
        "cases": cases,
    });
    std::fs::write(
        out_dir.join(format!("{prefix}.json")),
        serde_json::to_string_pretty(&doc)? + "\n",
    )?;
    Ok(())
}
