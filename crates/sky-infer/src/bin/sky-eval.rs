//! Held-out evaluation of any decision GGUF with the shipping engine (the code the browser and
//! the CLI run): per-source accuracy, NLL, ECE, early-exit depth and latency.
//!
//! ```text
//! sky-eval <model.gguf> --records a.jsonl,dir/ [--split test] [--max N] [--exit-threshold T]
//!          [--json out.json] [--dump probs.jsonl]
//! ```
//!
//! Records are SPEC decision records (`docs/SPEC.md`); a directory means every `*.jsonl` in it.
//! Unlike `sky-train s2-eval`, which re-tokenizes with the training run's tokenizer, each model is
//! evaluated with the tokenizer embedded in its own GGUF, so models with different vocabularies
//! (s3-pico, s4) are compared on the same records. `--max` keeps the first N records per source.
//! `--dump` writes one line per record with the probabilities, for comparisons with the WASM
//! build.

#[cfg(not(target_arch = "wasm32"))]
mod native {
    use std::collections::BTreeMap;
    use std::path::{Path, PathBuf};
    use std::time::Instant;

    use serde_json::{json, Value};
    use sky_infer::{Engine, EngineOptions, QType};

    fn usage() -> ! {
        eprintln!("usage: sky-eval <model.gguf> --records a.jsonl,dir/ [--split test] [--max N] [--exit-threshold T] [--json out.json] [--dump probs.jsonl]");
        std::process::exit(2);
    }

    #[derive(Default)]
    struct Acc {
        n: usize,
        hit: f64,
        nll: f64,
        layers: f64,
        early: f64,
        micros: Vec<f64>,
        /// (confidence, correct) for the ECE.
        conf: Vec<(f64, f64)>,
    }

    fn ece(conf: &[(f64, f64)]) -> f64 {
        let bins = 15;
        let mut b = vec![(0.0f64, 0.0f64, 0usize); bins];
        for &(c, h) in conf {
            let i = ((c * bins as f64) as usize).min(bins - 1);
            b[i].0 += c;
            b[i].1 += h;
            b[i].2 += 1;
        }
        let n = conf.len().max(1) as f64;
        b.iter()
            .filter(|x| x.2 > 0)
            .map(|(c, h, k)| (*k as f64 / n) * ((c - h) / *k as f64).abs())
            .sum()
    }

    fn files(spec: &str) -> Vec<PathBuf> {
        let mut out = Vec::new();
        for part in spec.split(',').filter(|s| !s.is_empty()) {
            let p = Path::new(part);
            if p.is_dir() {
                let mut v: Vec<PathBuf> = std::fs::read_dir(p)
                    .unwrap_or_else(|e| panic!("{part}: {e}"))
                    .filter_map(|e| e.ok().map(|e| e.path()))
                    .filter(|f| f.extension().is_some_and(|x| x == "jsonl"))
                    .collect();
                v.sort();
                out.extend(v);
            } else {
                out.push(p.to_path_buf());
            }
        }
        out
    }

    fn argmax(v: &[f64]) -> usize {
        let mut best = 0;
        for (i, x) in v.iter().enumerate() {
            if *x > v[best] {
                best = i;
            }
        }
        best
    }

    pub fn main() {
        let mut args = std::env::args().skip(1);
        let mut model = None;
        let (mut records, mut split, mut max) = (None, "test".to_string(), usize::MAX);
        let (mut tau, mut json_out, mut dump) = (None, None, None);
        while let Some(a) = args.next() {
            match a.as_str() {
                "--records" => records = args.next(),
                "--split" => split = args.next().unwrap_or_else(|| usage()),
                "--max" => {
                    max = args
                        .next()
                        .and_then(|v| v.parse().ok())
                        .unwrap_or_else(|| usage())
                }
                "--exit-threshold" => tau = args.next().and_then(|v| v.parse().ok()),
                "--json" => json_out = args.next(),
                "--dump" => dump = args.next(),
                "-h" | "--help" => usage(),
                _ if model.is_none() => model = Some(a),
                _ => usage(),
            }
        }
        let (Some(model), Some(records)) = (model, records) else {
            usage()
        };
        let bytes = std::fs::read(&model).unwrap_or_else(|e| panic!("{model}: {e}"));
        let mut eng = Engine::from_gguf_with(&bytes, EngineOptions::default())
            .unwrap_or_else(|e| panic!("{model}: {e}"));
        eng.set_exit_threshold(tau);
        let mut per: BTreeMap<String, Acc> = BTreeMap::new();
        let mut counts: BTreeMap<String, usize> = BTreeMap::new();
        let mut dump_lines = Vec::new();
        for f in files(&records) {
            let text = std::fs::read_to_string(&f).unwrap_or_else(|e| panic!("{f:?}: {e}"));
            for line in text.lines().filter(|l| !l.trim().is_empty()) {
                let r: Value = serde_json::from_str(line).unwrap_or_else(|e| panic!("{f:?}: {e}"));
                if r["split"].as_str() != Some(split.as_str()) {
                    continue;
                }
                let src = r["src"].as_str().unwrap_or("?").to_string();
                let c = counts.entry(src.clone()).or_default();
                if *c >= max {
                    continue;
                }
                *c += 1;
                let Some(qt) = r["t"].as_str().and_then(QType::from_name) else {
                    continue;
                };
                let opts: Vec<String> = r["opts"]
                    .as_array()
                    .map(|a| {
                        a.iter()
                            .filter_map(|o| o.as_str().map(str::to_string))
                            .collect()
                    })
                    .unwrap_or_default();
                let target: Vec<f64> = r["target"]
                    .as_array()
                    .map(|a| a.iter().filter_map(Value::as_f64).collect())
                    .unwrap_or_default();
                if opts.len() < 2 || target.len() != opts.len() {
                    continue;
                }
                let sum: f64 = target.iter().sum();
                let target: Vec<f64> = target.iter().map(|x| x / sum).collect();
                let t0 = Instant::now();
                let p = eng
                    .decide_raw(
                        qt,
                        r["ins"].as_str().unwrap_or(""),
                        &opts,
                        r["state"].as_str().unwrap_or(""),
                    )
                    .unwrap_or_else(|e| panic!("{src}: {e}"));
                let us = t0.elapsed().as_secs_f64() * 1e6;
                let p: Vec<f64> = p.iter().map(|x| *x as f64).collect();
                let st = eng.last_stats().cloned().unwrap_or_default();
                let a = argmax(&p);
                let hit = (a == argmax(&target)) as u8 as f64;
                let nll: f64 = target
                    .iter()
                    .zip(&p)
                    .map(|(t, q)| -t * q.max(1e-9).ln())
                    .sum();
                for key in [src.clone(), "all".to_string()] {
                    let e = per.entry(key).or_default();
                    e.n += 1;
                    e.hit += hit;
                    e.nll += nll;
                    e.layers += st.layers_used.unwrap_or(0) as f64;
                    e.early += (st.n_exits > 0 && st.exit.is_some_and(|x| x + 1 < st.n_exits)) as u8
                        as f64;
                    e.micros.push(us);
                    e.conf.push((p[a], hit));
                }
                if dump.is_some() {
                    dump_lines.push(
                        json!({"src": src, "t": r["t"], "ins": r["ins"], "opts": opts,
                        "state": r["state"], "probs": p, "exit": st.exit})
                        .to_string(),
                    );
                }
            }
        }
        if per.is_empty() {
            eprintln!("no {split} records in {records}");
            std::process::exit(1);
        }
        println!("{model} ({} parameters), split {split}", eng.n_params());
        println!(
            "{:<28} {:>6} {:>8} {:>8} {:>7} {:>7} {:>7} {:>9}",
            "source", "n", "accuracy", "NLL", "ECE", "layers", "early", "median_us"
        );
        let mut rows = serde_json::Map::new();
        for (k, a) in &mut per {
            a.micros.sort_by(|x, y| x.total_cmp(y));
            let n = a.n as f64;
            let med = a.micros[a.micros.len() / 2];
            let e = ece(&a.conf);
            println!(
                "{k:<28} {:>6} {:>8.4} {:>8.4} {:>7.4} {:>7.2} {:>7.3} {:>9.0}",
                a.n,
                a.hit / n,
                a.nll / n,
                e,
                a.layers / n,
                a.early / n,
                med
            );
            rows.insert(
                k.clone(),
                json!({"n": a.n, "accuracy": a.hit / n, "nll": a.nll / n, "ece": e,
                    "mean_layers": a.layers / n, "early_exit_rate": a.early / n, "median_us": med}),
            );
        }
        if let Some(path) = json_out {
            let v = json!({"model": model, "n_params": eng.n_params(), "split": split, "sources": rows});
            std::fs::write(&path, serde_json::to_string_pretty(&v).unwrap()).unwrap();
            println!("wrote {path}");
        }
        if let Some(path) = dump {
            std::fs::write(&path, dump_lines.join("\n") + "\n").unwrap();
            println!("wrote {path}");
        }
    }
}

fn main() {
    #[cfg(not(target_arch = "wasm32"))]
    native::main();
}
