//! `sky-turn`: run an s6-ear v2 end-of-turn file (docs/s6-ear2-prereg.md §4.5).
//!
//! ```text
//! sky-turn <model.gguf> --store <dir> [--split S] --out probs.jsonl
//! sky-turn <model.gguf> --store <dir> --trail-ms 200[,400,...] [--ext <dir>] [--split S] --out probs.jsonl
//! sky-turn <model.gguf> --records <file.jsonl> [--split S] [--max N] --out probs.jsonl
//! sky-turn <model.gguf> --bench N [--mode auto|f32|packed|int8|lut]
//! ```
//!
//! `--store` reads a feature store (sky-data `fetch-turn`, `turn-test`, `turn-eotbench`) and
//! `--records` the 16 kHz WAV named by each JSONL row (`wav`, relative to the file; the deployed
//! path: samples → front end → model). Each prints one line per item: `{id, logit, prob,
//! positions}` (with `wav` for records), `prob` the calibrated probability. No label is read.
//!
//! `--trail-ms` (s6-ear v3, docs/s6-ear3-prereg.md §16) reads each record at a trailing silence
//! of d ms (a list gives one line per record and d, with `d_ms`): a conversation or eot-bench span
//! record (`cut_hops` and `dmax_ms` in its manifest line) through its first `cut_hops + d / 10`
//! hops, skipped when d is above its `dmax_ms`; a Smart Turn clip through its own hops followed by
//! the first `d / 10` hops of its extension (`--ext`, the `st-ext` store of `sky-data
//! turn-extend`, joined by id). The model is unchanged: the window is the last 4 s before the cut.
//!
//! `--bench N` times N decisions on a full 4 s window whose hops are already computed (one
//! thread), and the front end per 10 ms hop: one JSON line with the medians and p95.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

use serde_json::{json, Value};
use sky_infer::audio::{self, TurnFrontEnd, SAMPLE_RATE};
use sky_infer::turn::store::Store;
use sky_infer::turn::TurnModel;
use sky_infer::TernaryCompute;

fn usage() -> ! {
    eprintln!(
        "usage: sky-turn <model.gguf> (--store DIR [--trail-ms D[,D...]] [--ext DIR] | --records FILE [--max N]) [--split S] --out FILE\n       sky-turn <model.gguf> --bench N [--mode auto|f32|packed|int8|lut]"
    );
    std::process::exit(2);
}

fn percentile(sorted: &[f64], q: f64) -> f64 {
    let i = ((sorted.len() as f64 - 1.0) * q).round() as usize;
    sorted[i.min(sorted.len() - 1)]
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let Some(path) = args.first() else { usage() };
    let (mut store, mut records, mut out, mut split) = (None, None, None, None);
    let (mut bench, mut max, mut mode) = (0usize, 0usize, "auto".to_string());
    let (mut trail, mut ext): (Option<Vec<usize>>, Option<PathBuf>) = (None, None);
    let mut i = 1;
    while i < args.len() {
        let v = || args.get(i + 1).cloned().unwrap_or_else(|| usage());
        match args[i].as_str() {
            "--store" => store = Some(PathBuf::from(v())),
            "--records" => records = Some(PathBuf::from(v())),
            "--out" => out = Some(PathBuf::from(v())),
            "--split" => split = Some(v()),
            "--bench" => bench = v().parse()?,
            "--max" => max = v().parse()?,
            "--mode" => mode = v(),
            "--trail-ms" => {
                trail = Some(
                    v().split(',')
                        .map(|x| x.trim().parse::<usize>())
                        .collect::<Result<_, _>>()?,
                )
            }
            "--ext" => ext = Some(PathBuf::from(v())),
            _ => usage(),
        }
        i += 2;
    }
    let bytes = std::fs::read(path)?;
    let mut m = TurnModel::from_bytes(&bytes)?;
    m.prepare_compute(match mode.as_str() {
        "auto" => TernaryCompute::Auto,
        "f32" => TernaryCompute::F32,
        "packed" => TernaryCompute::Packed,
        "int8" => TernaryCompute::Int8,
        "lut" => TernaryCompute::Lut,
        _ => usage(),
    });
    let mut fe = TurnFrontEnd::new();
    if bench > 0 {
        let pcm = audio::synth_speech(m.cfg.window_hops as f64 / 100.0 + 0.5, 1, 1.0);
        let t0 = Instant::now();
        let (h, n) = fe.hops(&pcm, 0);
        let per_hop = t0.elapsed().as_secs_f64() * 1e6 / n as f64;
        let (h, n) = (
            h[(n - m.cfg.window_hops) * audio::TURN_CHANNELS..].to_vec(),
            m.cfg.window_hops,
        );
        for _ in 0..20 {
            m.decide_hops(&h, n)?;
        }
        let mut us = Vec::with_capacity(bench);
        let mut last = None;
        for _ in 0..bench {
            let t = Instant::now();
            last = Some(m.decide_hops(&h, n)?);
            us.push(t.elapsed().as_secs_f64() * 1e6);
        }
        us.sort_by(f64::total_cmp);
        let o = last.expect("at least one decision");
        println!(
            "{}",
            json!({
                "model": path, "name": m.cfg.name, "params": m.n_params(), "file_bytes": bytes.len(),
                "mode": mode, "positions": o.positions, "decisions": bench,
                "median_us": percentile(&us, 0.5), "p95_us": percentile(&us, 0.95),
                "frontend_us_per_hop": per_hop, "prob": o.prob,
            })
        );
        return Ok(());
    }
    let Some(out) = out else { usage() };
    if let Some(d) = out.parent() {
        std::fs::create_dir_all(d)?;
    }
    let mut w = std::io::BufWriter::new(std::fs::File::create(&out)?);
    let mut n = 0usize;
    let keep = |s: &str| {
        split
            .as_deref()
            .is_none_or(|want| want == "all" || want == s)
    };
    if let (Some(dir), Some(ds)) = (store.as_ref(), trail.as_ref()) {
        if let Some(d) = ds.iter().find(|&&d| d % 10 != 0 || d > 2400) {
            return Err(format!("--trail-ms {d}: a multiple of 10 ms, at most 2400").into());
        }
        let st = Store::open(dir)?;
        let ex = ext.as_ref().map(|e| Store::open(e)).transpose()?;
        let by_id: std::collections::HashMap<&str, usize> = ex
            .as_ref()
            .map(|e| {
                e.clips
                    .iter()
                    .enumerate()
                    .map(|(i, c)| (c.id.as_str(), i))
                    .collect()
            })
            .unwrap_or_default();
        for c in st.clips.iter().filter(|c| keep(c.split())) {
            let h = st.hops(c)?;
            for &d in ds {
                // the hops up to the cut at d
                let window: Option<Vec<f32>> =
                    match (c.meta["cut_hops"].as_u64(), c.meta["dmax_ms"].as_u64()) {
                        (Some(cut), Some(dmax)) => (d as u64 <= dmax).then(|| {
                            let end = (cut as usize + d / 10).min(c.hops);
                            h[..end * audio::TURN_CHANNELS].to_vec()
                        }),
                        _ if d == 0 => Some(h.clone()),
                        _ => match (&ex, by_id.get(c.id.as_str())) {
                            (Some(e), Some(&i)) => {
                                let x = e.hops(&e.clips[i])?;
                                let mut joined = h.clone();
                                joined.extend_from_slice(&x[..(d / 10) * audio::TURN_CHANNELS]);
                                Some(joined)
                            }
                            _ => None,
                        },
                    };
                let Some(win) = window else { continue };
                let k = win.len() / audio::TURN_CHANNELS;
                if k == 0 {
                    writeln!(w, "{}", json!({"id": c.id, "d_ms": d, "positions": 0}))?;
                    continue;
                }
                let o = m.decide_hops(&win, k)?;
                writeln!(
                    w,
                    "{}",
                    json!({"id": c.id, "d_ms": d, "logit": o.logit, "prob": o.prob, "positions": o.positions})
                )?;
                n += 1;
            }
        }
    } else if let Some(dir) = store {
        let st = Store::open(&dir)?;
        for c in st.clips.iter().filter(|c| keep(c.split())) {
            if c.hops == 0 {
                writeln!(w, "{}", json!({"id": c.id, "positions": 0}))?;
                continue;
            }
            let h = st.hops(c)?;
            let o = m.decide_hops(&h, c.hops)?;
            writeln!(
                w,
                "{}",
                json!({"id": c.id, "logit": o.logit, "prob": o.prob, "positions": o.positions})
            )?;
            n += 1;
        }
    } else if let Some(rec) = records {
        let base = rec.parent().unwrap_or(Path::new("."));
        let text = std::fs::read_to_string(&rec)?;
        for line in text.lines().filter(|l| !l.trim().is_empty()) {
            if max > 0 && n >= max {
                break;
            }
            let r: Value = serde_json::from_str(line)?;
            if !keep(r["split"].as_str().unwrap_or("")) {
                continue;
            }
            let Some(wav) = r["wav"].as_str() else {
                continue;
            };
            let wav = base.join(wav);
            let (pcm, rate) = audio::read_wav(&std::fs::read(&wav)?)?;
            let pcm = audio::resample(&pcm, rate, SAMPLE_RATE);
            let id = r["clip"].as_str().or(r["id"].as_str()).unwrap_or("");
            match m.decide_pcm(&mut fe, &pcm) {
                Ok(o) => writeln!(
                    w,
                    "{}",
                    json!({"id": id, "wav": wav, "logit": o.logit, "prob": o.prob, "positions": o.positions})
                )?,
                Err(_) => writeln!(w, "{}", json!({"id": id, "wav": wav, "positions": 0}))?,
            }
            n += 1;
        }
    } else {
        usage();
    }
    w.flush()?;
    eprintln!("sky-turn: {n} decisions -> {}", out.display());
    Ok(())
}
