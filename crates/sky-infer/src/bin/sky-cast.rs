//! `sky-cast`: forecast with an s6-cast or TinyCast GGUF file (docs/SPEC-S6-CAST.md).
//!
//! ```text
//! sky-cast <model.gguf> --values 1,2,3,...      # history on the command line
//! sky-cast <model.gguf> --file series.txt       # one number per line (empty or NaN = missing)
//! sky-cast <model.gguf> --synth tsi --seed 3    # a synthetic series (gp, pulse, tsi, sarima)
//!     [--horizon H] [--flip] [--full-depth] [--int8] [--bench N]
//! sky-cast convert-tinycast <model.safetensors> <config.json> <out.gguf>
//! ```
//!
//! Prints the forecast as JSON (`levels`, `quantiles` `[horizon][levels]`, exits used); with
//! `--bench N`, the median latency of N forecasts instead. `convert-tinycast` turns the Hugging
//! Face release of TinyCast (raws-labs/tinycast) into a `skycmd-tinycast` GGUF file.

use std::time::Instant;

use sky_infer::cast::model::ForecastOptions;
use sky_infer::cast::synth::{self, Kind};
use sky_infer::cast::tinycast::tinycast_gguf_from_safetensors;
use sky_infer::cast::Forecaster;
use sky_infer::TernaryCompute;

fn usage() -> ! {
    eprintln!(
        "usage: sky-cast <model.gguf> (--values a,b,c | --file path | --synth gp|pulse|tsi|sarima [--seed S] [--len N]) [--horizon H] [--flip] [--full-depth] [--int8] [--bench N]\n       sky-cast convert-tinycast <model.safetensors> <config.json> <out.gguf>"
    );
    std::process::exit(2);
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let Some(path) = args.first() else { usage() };
    if path == "convert-tinycast" {
        let [st, cfg, out] = &args[1..] else { usage() };
        let bytes =
            tinycast_gguf_from_safetensors(&std::fs::read(st)?, &std::fs::read_to_string(cfg)?)?;
        std::fs::write(out, &bytes)?;
        let mut m = Forecaster::from_bytes(&bytes)?;
        let n = m.n_params();
        eprintln!(
            "{out}: {} ({n} parameters, {} bytes)",
            m.name(),
            bytes.len()
        );
        return Ok(());
    }
    let flag = |name: &str| {
        args.iter()
            .position(|a| a == name)
            .and_then(|i| args.get(i + 1))
            .cloned()
    };
    let has = |name: &str| args.iter().any(|a| a == name);
    let parse_num = |s: &str| s.trim().parse::<f32>().unwrap_or(f32::NAN);
    let history: Vec<f32> = if let Some(v) = flag("--values") {
        v.split(',').map(parse_num).collect()
    } else if let Some(f) = flag("--file") {
        std::fs::read_to_string(f)?.lines().map(parse_num).collect()
    } else if let Some(k) = flag("--synth") {
        let kind = Kind::parse(&k).unwrap_or_else(|| usage());
        let seed = flag("--seed").map_or(Ok(1), |s| s.parse())?;
        let len = flag("--len").map_or(Ok(1024), |s| s.parse())?;
        synth::series(kind, seed, len)
    } else {
        usage()
    };
    let mut m = Forecaster::from_bytes(&std::fs::read(path)?)?;
    m.prepare_compute(if has("--int8") {
        TernaryCompute::Int8
    } else {
        TernaryCompute::Auto
    });
    let horizon = flag("--horizon").map_or(Ok(m.pass_horizon()), |s| s.parse())?;
    let opts = ForecastOptions {
        flip: has("--flip"),
        full_depth: has("--full-depth"),
    };
    if let Some(n) = flag("--bench") {
        let n: usize = n.parse()?;
        let mut us: Vec<f64> = (0..n.max(1))
            .map(|_| {
                let t = Instant::now();
                let f = m.forecast(&history, horizon, opts).expect("forecast");
                std::hint::black_box(f);
                t.elapsed().as_secs_f64() * 1e6
            })
            .collect();
        us.sort_by(f64::total_cmp);
        println!(
            "{}: {} points of history, horizon {horizon}: median {:.0} µs, p90 {:.0} µs ({} runs)",
            m.name(),
            history.len(),
            us[us.len() / 2],
            us[us.len() * 9 / 10],
            us.len()
        );
        return Ok(());
    }
    let f = m.forecast(&history, horizon, opts)?;
    let q = f.levels.len();
    // levels as written in the file (0.1, not 0.10000000149011612)
    let levels: Vec<f64> = f
        .levels
        .iter()
        .map(|&l| (l as f64 * 1e6).round() / 1e6)
        .collect();
    let rows: Vec<serde_json::Value> = f
        .quantiles
        .chunks(q)
        .map(|r| serde_json::json!(r))
        .collect();
    println!(
        "{}",
        serde_json::json!({
            "model": m.name(),
            "levels": levels,
            "horizon": f.horizon,
            "quantiles": rows,
            "exits_used": f.exits_used,
            "n_exits": f.n_exits,
            "passes": f.passes,
        })
    );
    Ok(())
}
