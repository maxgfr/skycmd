//! Golden files of s6-ear v2 (docs/s6-ear2-prereg.md §4.5): random weights of the femto shape,
//! saved as f32 and ternary GGUF files, and the logit and pooled vector of the mlx-rs graph on the
//! weights exactly as stored, for windows of `sky_infer::audio::synth_speech` signals. sky-infer's
//! `tests/golden_turn.rs` must reproduce them from the signals alone (front end, normalisation,
//! both attention kernels, pooling and head), with the tolerances of the cast golden.

use std::path::Path;

use anyhow::Result;
use mlx_rs::{random, Array};
use serde_json::{json, Value};
use sky_gguf::Gguf;
use sky_infer::audio::{normalize_window, synth_speech, TurnFrontEnd};

use super::{
    config::TurnConfig,
    data::HostBatch,
    gguf::{config_from_gguf, params_from_gguf, to_gguf, Calibration},
    model::{forward, init},
};
use crate::{model::host_f32, params::ParamMap, s2::gguf::WeightFormat};

/// Random weights large enough that every path matters (norms and biases randomized too).
fn golden_params(cfg: &TurnConfig, seed: u64) -> Result<ParamMap> {
    let base = init(cfg, seed)?;
    let mut key = random::key(seed ^ 0x7e57)?;
    let mut names: Vec<_> = base.keys().cloned().collect();
    names.sort();
    let mut out = ParamMap::new();
    for name in names {
        let shape = base[&name].shape().to_vec();
        let (k, next) = random::split(&key, 2)?;
        key = next;
        let n = random::normal::<f32>(&shape[..], 0.0, 1.0, &k)?;
        let a = if name.ends_with(".bias") {
            n.multiply(Array::from_f32(0.1))?
        } else if name.contains("norm") {
            n.multiply(Array::from_f32(0.2))?
                .add(Array::from_f32(1.0))?
        } else {
            n.multiply(Array::from_f32(1.0 / (shape[1] as f32).sqrt()))?
        };
        out.insert(name, a);
    }
    crate::params::eval(&out)?;
    Ok(out)
}

/// The golden signals: (name, seconds, seed, amplitude). A full window, one longer than the small
/// attention kernel's 32 positions, one below it, silence and a quiet one.
pub const CASES: [(&str, f64, u64, f32); 5] = [
    ("full", 4.6, 1, 1.0),
    ("tiled", 2.2, 2, 0.7),
    ("short", 0.25, 3, 1.0),
    ("silence", 1.0, 4, 0.0),
    ("quiet", 3.0, 5, 0.01),
];

/// Writes `{prefix}-f32.gguf`, `{prefix}-ternary.gguf` and `{prefix}.json` into `out_dir`.
pub fn golden(cfg: &TurnConfig, out_dir: &Path, prefix: &str, seed: u64) -> Result<()> {
    std::fs::create_dir_all(out_dir)?;
    let params = golden_params(cfg, seed)?;
    let mut fe = TurnFrontEnd::new();
    let windows: Vec<(Vec<f32>, usize)> = CASES
        .iter()
        .map(|&(_, secs, s, amp)| {
            let (h, n) = fe.hops(&synth_speech(secs, s, amp), cfg.window_hops);
            normalize_window(&h, n, cfg.hops_per_position)
        })
        .collect();
    let cal = Calibration {
        temperature: 1.5,
        tau50: 0.5,
        tau85: 0.35,
    };
    let mut files = Vec::new();
    let mut outputs: Vec<Vec<(f32, Vec<f32>)>> = vec![Vec::new(); 2];
    for (fi, format) in [WeightFormat::F32, WeightFormat::Ternary]
        .into_iter()
        .enumerate()
    {
        let bytes = to_gguf(&params, cfg, format, cal)?;
        let file = format!("{prefix}-{}.gguf", format.name());
        std::fs::write(out_dir.join(&file), &bytes)?;
        let g = Gguf::parse(&bytes)?;
        let (gcfg, _) = config_from_gguf(&g)?;
        let p = params_from_gguf(&g, &gcfg)?;
        for (x, positions) in &windows {
            let (t, wd) = (gcfg.max_positions(), gcfg.input_width());
            let mut hb = HostBatch {
                x: vec![0.0; t * wd],
                valid: vec![false; t],
                y: vec![0.0],
                zt: vec![0.0],
                et: vec![0.0; 384],
                st: vec![0.0; 25 * 384],
                smask: vec![0.0; 25],
                tmask: vec![0.0],
                positions: vec![*positions],
                rows: 1,
            };
            hb.x[(t - positions) * wd..].copy_from_slice(x);
            hb.valid[t - positions..].fill(true);
            let f = forward(&p, &gcfg, &hb.arrays(&gcfg), false)?;
            outputs[fi].push((host_f32(&f.logit)?[0], host_f32(&f.pooled)?));
        }
        files.push(json!({"format": format.name(), "path": file, "bytes": bytes.len()}));
    }
    let cases: Vec<Value> = CASES
        .iter()
        .enumerate()
        .map(|(i, &(name, secs, s, amp))| {
            json!({
                "name": name,
                "signal": {"secs": secs, "seed": s, "amp": amp},
                "positions": windows[i].1,
                "logit": {"f32": outputs[0][i].0, "ternary": outputs[1][i].0},
                "pooled": {"f32": outputs[0][i].1, "ternary": outputs[1][i].1},
            })
        })
        .collect();
    let doc = json!({
        "meta": {
            "config": cfg,
            "calibration": cal,
            "files": files,
            "tolerance": {"f32": 1e-4, "ternary": 1e-3},
            "about": "logit and attention-pooled vector of sky-train's mlx-rs forward (CPU) on the stored weights, for the windows of sky_infer::audio::synth_speech(secs, seed, amp)",
        },
        "cases": cases,
    });
    std::fs::write(
        out_dir.join(format!("{prefix}.json")),
        serde_json::to_string_pretty(&doc)? + "\n",
    )?;
    Ok(())
}
