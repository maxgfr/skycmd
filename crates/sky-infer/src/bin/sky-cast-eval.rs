//! `sky-cast-eval`: forecasts for the GIFT-Eval harness and the early-exit calibration of
//! s6-cast (docs/SPEC-S6-CAST.md §6).
//!
//! ```text
//! sky-cast-eval run   --model m.gguf --windows data/gift-eval --out data/gift-eval-out/<name>
//!                     [--flip] [--full-depth] [--threads N] [--configs a,b]
//! sky-cast-eval naive --windows data/gift-eval --out data/gift-eval-out/seasonal-naive
//! sky-cast-eval calibrate --model m.gguf --out m-cal.gguf [--alpha 0.05] [--delta 0.02] [--n 2000]
//! ```
//!
//! `run` takes an s6-cast or a TinyCast file (`sky_infer::cast::Forecaster`). `run` and `naive`
//! read the windows `scripts/gift-eval.py export` wrote (`<config>.json` +
//! `<config>.bin`) and write `<config>.q.bin`: f32 quantiles `[windows][prediction_length][9]`,
//! which `scripts/gift-eval.py score` replays through gluonts. `naive` is the seasonal-naive
//! forecast (every level the point forecast): its score checks the harness against the
//! leaderboard's seasonal-naive row.
//!
//! `calibrate` picks `exit_tolerance` by conformal risk control on synthetic windows unseen in
//! training: the largest tolerance whose early-exit median stays within `delta` (normalized
//! units, max over the horizon) of the full-depth median, with the bound
//! `(n·R̂ + 1) / (n + 1) ≤ alpha` on the share of windows that do not.

use std::path::{Path, PathBuf};

use serde_json::Value;
use sky_gguf::{Gguf, GgufWriter, OwnedValue};
use sky_infer::cast::model::ForecastOptions;
use sky_infer::cast::prep::{self, Norm};
use sky_infer::cast::{synth, CastModel, Forecaster, ARCH, DECILES};
use sky_infer::TernaryCompute;

type Res<T> = Result<T, Box<dyn std::error::Error + Send + Sync>>;

fn usage() -> ! {
    eprintln!(
        "usage: sky-cast-eval run --model m.gguf --windows DIR --out DIR [--flip] [--full-depth] [--threads N] [--configs a,b]\n       sky-cast-eval naive --windows DIR --out DIR\n       sky-cast-eval calibrate --model m.gguf --out m-cal.gguf [--alpha 0.05] [--delta 0.02] [--n 2000]"
    );
    std::process::exit(2);
}

/// The windows of one configuration.
struct Config {
    stem: String,
    horizon: usize,
    season: usize,
    windows: Vec<Vec<f32>>,
}

fn load_configs(dir: &Path, only: &[String]) -> Res<Vec<Config>> {
    let mut metas: Vec<PathBuf> = std::fs::read_dir(dir)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "json"))
        .collect();
    metas.sort();
    let mut out = Vec::new();
    for m in metas {
        let meta: Value = serde_json::from_slice(&std::fs::read(&m)?)?;
        let name = meta["config"].as_str().unwrap_or_default().to_string();
        if !only.is_empty() && !only.contains(&name) {
            continue;
        }
        let stem = m.file_stem().unwrap().to_string_lossy().to_string();
        let raw = std::fs::read(dir.join(format!("{stem}.bin")))?;
        let all: Vec<f32> = raw
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect();
        let num = |k: &str| meta[k].as_u64().unwrap_or(0) as usize;
        let offsets = meta["offsets"].as_array().ok_or("no offsets")?;
        let lengths = meta["lengths"].as_array().ok_or("no lengths")?;
        let windows = offsets
            .iter()
            .zip(lengths)
            .map(|(o, l)| {
                let (o, l) = (o.as_u64().unwrap() as usize, l.as_u64().unwrap() as usize);
                all[o..o + l].to_vec()
            })
            .collect();
        out.push(Config {
            stem,
            horizon: num("prediction_length"),
            season: num("season").max(1),
            windows,
        });
    }
    Ok(out)
}

fn write_quantiles(path: &Path, q: &[f32]) -> Res<()> {
    let bytes: Vec<u8> = q.iter().flat_map(|v| v.to_le_bytes()).collect();
    std::fs::write(path, bytes)?;
    Ok(())
}

/// Splits `n` windows over `threads` workers; `f(i)` forecasts window `i`.
fn parallel<F>(n: usize, threads: usize, f: F) -> Res<Vec<Vec<f32>>>
where
    F: Fn(usize) -> Res<Vec<f32>> + Sync,
{
    let chunk = n.div_ceil(threads.max(1)).max(1);
    let parts: Vec<Res<Vec<Vec<f32>>>> = std::thread::scope(|s| {
        let handles: Vec<_> = (0..n)
            .step_by(chunk)
            .map(|start| {
                let f = &f;
                s.spawn(move || (start..(start + chunk).min(n)).map(f).collect())
            })
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().expect("worker"))
            .collect()
    });
    let mut out = Vec::with_capacity(n);
    for p in parts {
        out.extend(p?);
    }
    Ok(out)
}

fn cmd_run(
    model: &Path,
    windows: &Path,
    out: &Path,
    opts: ForecastOptions,
    threads: usize,
    only: &[String],
) -> Res<()> {
    let mut m = Forecaster::from_bytes(&std::fs::read(model)?)?;
    m.prepare_compute(TernaryCompute::Auto);
    std::fs::create_dir_all(out)?;
    let levels = m.levels().len();
    if m.levels() != DECILES {
        return Err("the GIFT-Eval harness scores the nine deciles".into());
    }
    for c in load_configs(windows, only)? {
        let t = std::time::Instant::now();
        let rows = parallel(c.windows.len(), threads, |i| {
            let f = m.forecast(&c.windows[i], c.horizon, opts)?;
            Ok(f.quantiles)
        })?;
        let flat: Vec<f32> = rows.concat();
        debug_assert_eq!(flat.len(), c.windows.len() * c.horizon * levels);
        write_quantiles(&out.join(format!("{}.q.bin", c.stem)), &flat)?;
        println!(
            "{}: {} windows, horizon {} ({:.1}s)",
            c.stem,
            c.windows.len(),
            c.horizon,
            t.elapsed().as_secs_f64()
        );
    }
    Ok(())
}

fn cmd_naive(windows: &Path, out: &Path) -> Res<()> {
    std::fs::create_dir_all(out)?;
    for c in load_configs(windows, &[])? {
        let mut flat = Vec::with_capacity(c.windows.len() * c.horizon * DECILES.len());
        for w in &c.windows {
            for v in prep::seasonal_naive(w, c.season, c.horizon) {
                flat.extend(std::iter::repeat_n(v, DECILES.len()));
            }
        }
        write_quantiles(&out.join(format!("{}.q.bin", c.stem)), &flat)?;
        println!("{}: {} windows", c.stem, c.windows.len());
    }
    Ok(())
}

/// The exit a pass stops at with tolerance `tau`, from the medians of every exit.
fn exit_at(medians: &[Vec<f32>], tau: f32) -> usize {
    for e in 1..medians.len() {
        let moved = medians[e]
            .iter()
            .zip(&medians[e - 1])
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        if moved < tau {
            return e;
        }
    }
    medians.len() - 1
}

fn cmd_calibrate(model: &Path, out: &Path, alpha: f64, delta: f32, n: usize) -> Res<()> {
    let bytes = std::fs::read(model)?;
    let mut m = CastModel::from_bytes(&bytes)?;
    m.prepare_compute(TernaryCompute::Auto);
    let c = m.cfg.clone();
    let len = c.context() + c.horizon();
    let med = c.median_index();
    let levels = c.quantiles.len();
    // per window: the median of every exit (normalized units)
    let mut runs: Vec<Vec<Vec<f32>>> = Vec::with_capacity(n);
    for i in 0..n {
        let seed = sky_infer_hash(&format!("cal:{i}"));
        let (_, y) = synth::mixed(seed, len);
        let ctx_patches = 4 + (i * 7919) % (c.context_patches.saturating_sub(4) + 1);
        let ctx = &y[len - c.horizon() - ctx_patches * c.patch..len - c.horizon()];
        let norm = Norm::fit(ctx);
        let (vals, mask) = prep::patches(ctx, norm, c.patch, c.context_patches);
        let outs = m.forward_window(&vals, &mask, true)?;
        runs.push(
            outs.iter()
                .map(|q| q.iter().skip(med).step_by(levels).copied().collect())
                .collect(),
        );
    }
    let ne = c.exits.len();
    let grid: Vec<f32> = (0..40).map(|k| 1e-3 * 1.2f32.powi(k)).collect();
    let mut best = 0.0f32;
    println!("tolerance  mean exits  risk     bound   (alpha {alpha}, delta {delta}, n {n})");
    for &tau in &grid {
        let (mut bad, mut depth) = (0usize, 0usize);
        for med_e in &runs {
            let e = exit_at(med_e, tau);
            depth += e + 1;
            let gap = med_e[e]
                .iter()
                .zip(&med_e[ne - 1])
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            if gap > delta {
                bad += 1;
            }
        }
        let risk = bad as f64 / n as f64;
        let bound = (n as f64 * risk + 1.0) / (n as f64 + 1.0);
        println!(
            "{tau:9.4}  {:10.2}  {risk:.4}  {bound:.4}",
            depth as f64 / n as f64
        );
        if bound <= alpha {
            best = tau;
        }
    }
    println!("exit_tolerance = {best} ({} exits)", ne);
    let g = Gguf::parse(&bytes)?;
    let key = format!("{ARCH}.exit_tolerance");
    let mut w = GgufWriter::new().with_alignment(g.alignment);
    for (k, v) in &g.kvs {
        if *k == "general.alignment" {
            continue;
        }
        if *k == key {
            w.add_kv(*k, OwnedValue::F32(best));
        } else {
            w.add_kv(*k, OwnedValue::from_value(v));
        }
    }
    if !g.kvs.iter().any(|(k, _)| *k == key) {
        w.add_kv(key, OwnedValue::F32(best));
    }
    for t in &g.tensors {
        w.add_tensor(t.name, &t.dims, t.dtype, g.tensor_bytes(t).to_vec())?;
    }
    std::fs::write(out, w.to_bytes()?)?;
    println!("wrote {}", out.display());
    Ok(())
}

/// FNV-1a (as sky-train's `hash_str`), so calibration seeds never collide with training ones.
fn sky_infer_hash(s: &str) -> u64 {
    s.bytes().fold(0xcbf2_9ce4_8422_2325u64, |h, b| {
        (h ^ b as u64).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

fn main() -> Res<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let Some(cmd) = args.first() else { usage() };
    let flag = |name: &str| {
        args.iter()
            .position(|a| a == name)
            .and_then(|i| args.get(i + 1))
            .cloned()
    };
    let has = |name: &str| args.iter().any(|a| a == name);
    let path = |name: &str| flag(name).map(PathBuf::from).unwrap_or_else(|| usage());
    match cmd.as_str() {
        "run" => {
            let threads = flag("--threads").map_or_else(
                || std::thread::available_parallelism().map_or(4, |n| n.get()),
                |s| s.parse().unwrap_or(4),
            );
            let only: Vec<String> = flag("--configs")
                .map(|s| s.split(',').map(str::to_string).collect())
                .unwrap_or_default();
            cmd_run(
                &path("--model"),
                &path("--windows"),
                &path("--out"),
                ForecastOptions {
                    flip: has("--flip"),
                    full_depth: has("--full-depth"),
                },
                threads,
                &only,
            )
        }
        "naive" => cmd_naive(&path("--windows"), &path("--out")),
        "calibrate" => cmd_calibrate(
            &path("--model"),
            &path("--out"),
            flag("--alpha").map_or(Ok(0.05), |s| s.parse())?,
            flag("--delta").map_or(Ok(0.02), |s| s.parse())?,
            flag("--n").map_or(Ok(2000), |s| s.parse())?,
        ),
        _ => usage(),
    }
}
