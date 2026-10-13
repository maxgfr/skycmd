//! Held-out evaluation of any decision GGUF with the shipping engine (the code the browser and
//! the CLI run): per-source accuracy, NLL, ECE, early-exit depth and latency.
//!
//! ```text
//! sky-eval <model.gguf> --records a.jsonl,dir/ [--split test] [--max N] [--exit-threshold T]
//!          [--json out.json] [--dump probs.jsonl] [--metrics] [--bootstrap B]
//!          [--max-options N] [--seed S] [--dump-probs items.jsonl] [--latency]
//! ```
//!
//! Records are SPEC decision records (`docs/SPEC.md`); a directory means every `*.jsonl` in it.
//! Unlike `sky-train s2-eval`, which re-tokenizes with the training run's tokenizer, each model is
//! evaluated with the tokenizer embedded in its own GGUF, so models with different vocabularies
//! (s3-pico, s4) are compared on the same records. `--max` keeps the first N records per source.
//! `--dump` writes one line per record with the probabilities, for comparisons with the WASM
//! build.
//!
//! The options below (s5) are off by default: without them the `--json` output is unchanged key
//! for key, since `cargo xtask regress` compares it with its baseline and counts an added key as
//! a change.
//! - `--metrics`: per source, the Brier score (`brier`), the classwise ECE (`ece_classwise`),
//!   the RPS of the score questions (`rps`, over `n_score` of them, null without any),
//!   selective prediction (`selective`: accuracy of the 10/20/50% most confident answers,
//!   confident errors, coverage at ≤ 5% error, AURC) and percentile bootstrap intervals
//!   (`ci95`: `value`, `lo`, `hi` of accuracy, NLL, Brier and ECE; `--bootstrap` resamples,
//!   default 2000, seed `--seed`), all from `sky-metrics`. There the NLL floors probabilities at
//!   1e-12 and the ECE uses `sky-train`'s bins; the historical `nll` and `ece` keys keep their
//!   definitions (1e-9 floor, own bins).
//! - `--max-options N`: each question keeps min(N, K) options, the gold one (argmax of the
//!   target) and N − 1 others drawn uniformly, in their original order; the target is
//!   renormalized over them. The draw depends on `--seed` (default 0) and the record's line
//!   only, not on the other records. The JSON reports `max_options` and `seed`.
//! - `--dump-probs PATH`: one line per evaluated question, `{src, t, k, probs, target, gold}`,
//!   the input of `sky-compare`.
//! - `--latency`: decision latency (batch 1: one question per call, in record order) of a fresh
//!   engine with the default options and of one with its caches off (`prefix_cache: 0,
//!   state_cache: 0`): `latency.default` and `latency.no_cache`, `p50_us` and `p95_us`.

#[cfg(not(target_arch = "wasm32"))]
mod native {
    use std::collections::BTreeMap;
    use std::path::{Path, PathBuf};
    use std::time::Instant;

    use serde_json::{json, Map, Value};
    use sky_infer::{Engine, EngineOptions, QType};
    use sky_metrics::{
        brier, classwise_ece, hash_str, quantile, rps, selective, Bootstrap, Metric, Pred, Rng,
    };

    fn usage() -> ! {
        eprintln!("usage: sky-eval <model.gguf> --records a.jsonl,dir/ [--split test] [--max N] [--exit-threshold T] [--json out.json] [--dump probs.jsonl] [--metrics] [--bootstrap B] [--max-options N] [--seed S] [--dump-probs items.jsonl] [--latency]");
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

    /// An evaluated question, kept for the s5 options.
    struct Item {
        src: String,
        qt: QType,
        pred: Pred,
        /// Instructions, options and state, kept for `--latency`.
        input: Option<(String, Vec<String>, String)>,
    }

    /// Keeps the gold option and `n − 1` others drawn uniformly, in their original order, and
    /// renormalizes the target over them.
    fn keep_options(opts: &mut Vec<String>, target: &mut Vec<f64>, n: usize, rng: &mut Rng) {
        if n >= opts.len() {
            return;
        }
        let gold = argmax(target);
        let mut keep: Vec<usize> = (0..opts.len()).filter(|&i| i != gold).collect();
        rng.shuffle(&mut keep);
        keep.truncate(n - 1);
        keep.push(gold);
        keep.sort_unstable();
        let sum: f64 = keep.iter().map(|&i| target[i]).sum();
        *opts = keep.iter().map(|&i| opts[i].clone()).collect();
        *target = keep.iter().map(|&i| target[i] / sum).collect();
    }

    /// Adds the `--metrics` keys to every source row, and prints them.
    fn add_metrics(items: &[Item], rows: &mut Map<String, Value>, boot: &Bootstrap) {
        let mut groups: BTreeMap<&str, Vec<&Item>> = BTreeMap::new();
        for it in items {
            for key in [it.src.as_str(), "all"] {
                groups.entry(key).or_default().push(it);
            }
        }
        let f = |x: f64| {
            if x.is_finite() {
                format!("{x:.4}")
            } else {
                "-".to_string()
            }
        };
        println!(
            "\n{:<28} {:>7} {:>7} {:>7} {:>7} {:>7} {:>7} {:>8} {:>7} {:>7}",
            "source",
            "Brier",
            "cw-ECE",
            "RPS",
            "acc@10%",
            "acc@20%",
            "acc@50%",
            "conf.err",
            "cov@5%",
            "AURC"
        );
        let mut cis = Vec::new();
        for (key, its) in &groups {
            let preds: Vec<Pred> = its.iter().map(|it| it.pred.clone()).collect();
            let score = its.iter().filter(|it| it.qt == QType::Score);
            let n_score = score.clone().count();
            let (b, cw, r) = (
                brier(&preds),
                classwise_ece(&preds),
                rps(score.map(|it| &it.pred)),
            );
            let s = selective(&preds);
            println!(
                "{key:<28} {:>7} {:>7} {:>7} {:>7} {:>7} {:>7} {:>8} {:>7} {:>7}",
                f(b),
                f(cw),
                f(r),
                f(s.acc_top10),
                f(s.acc_top20),
                f(s.acc_top50),
                s.confident_errors,
                f(s.coverage_at_5pct_error),
                f(s.aurc)
            );
            let ci: Vec<_> = Metric::ALL
                .iter()
                .map(|&m| (m, boot.metric_ci(m, &preds)))
                .collect();
            let cells: Vec<String> = ci
                .iter()
                .map(|(_, i)| format!("{:<18}", format!("[{}, {}]", f(i.lo), f(i.hi))))
                .collect();
            cis.push(format!("{key:<28} {}", cells.join(" ").trim_end()));
            let Some(row) = rows.get_mut(*key).and_then(Value::as_object_mut) else {
                continue;
            };
            row.insert("brier".into(), json!(b));
            row.insert("ece_classwise".into(), json!(cw));
            row.insert("rps".into(), json!(r));
            row.insert("n_score".into(), json!(n_score));
            row.insert(
                "selective".into(),
                json!({"acc_top10": s.acc_top10, "acc_top20": s.acc_top20,
                    "acc_top50": s.acc_top50, "confident_errors": s.confident_errors,
                    "confident_error_rate": s.confident_error_rate,
                    "coverage_at_5pct_error": s.coverage_at_5pct_error, "aurc": s.aurc}),
            );
            row.insert(
                "ci95".into(),
                ci.iter()
                    .map(|(m, i)| {
                        let v = json!({"value": i.value, "lo": i.lo, "hi": i.hi});
                        (m.name().to_string(), v)
                    })
                    .collect::<Map<_, _>>()
                    .into(),
            );
        }
        println!(
            "\n{:.0}% bootstrap intervals ({} resamples, seed {})",
            boot.level * 100.0,
            boot.resamples,
            boot.seed
        );
        println!(
            "{:<28} {:<18} {:<18} {:<18} ECE",
            "source", "accuracy", "NLL", "Brier"
        );
        for l in cis {
            println!("{l}");
        }
    }

    /// p50 and p95 decision latency (µs) of a fresh engine, one question per call in order.
    fn time_engine(
        bytes: &[u8],
        opts: EngineOptions,
        tau: Option<f32>,
        items: &[Item],
    ) -> (f64, f64) {
        let mut eng = Engine::from_gguf_with(bytes, opts).unwrap_or_else(|e| panic!("{e}"));
        eng.set_exit_threshold(tau);
        let mut us: Vec<f64> = items
            .iter()
            .filter_map(|it| it.input.as_ref().map(|i| (it, i)))
            .map(|(it, (ins, opts, state))| {
                let t0 = Instant::now();
                eng.decide_raw(it.qt, ins, opts, state)
                    .unwrap_or_else(|e| panic!("{}: {e}", it.src));
                t0.elapsed().as_secs_f64() * 1e6
            })
            .collect();
        us.sort_by(f64::total_cmp);
        (quantile(&us, 0.5), quantile(&us, 0.95))
    }

    /// The `--latency` measurement, default options then caches off.
    fn measure_latency(bytes: &[u8], tau: Option<f32>, items: &[Item]) -> Value {
        let (d50, d95) = time_engine(bytes, EngineOptions::default(), tau, items);
        let off = EngineOptions {
            prefix_cache: 0,
            state_cache: 0,
            ..Default::default()
        };
        let (o50, o95) = time_engine(bytes, off, tau, items);
        println!(
            "\nlatency per decision (batch 1, {} decisions): default p50 {d50:.0} us, p95 {d95:.0} us; caches off p50 {o50:.0} us, p95 {o95:.0} us",
            items.len()
        );
        json!({"n": items.len(), "batch": 1,
            "default": {"p50_us": d50, "p95_us": d95},
            "no_cache": {"p50_us": o50, "p95_us": o95}})
    }

    pub fn main() {
        let mut args = std::env::args().skip(1);
        let mut model = None;
        let (mut records, mut split, mut max) = (None, "test".to_string(), usize::MAX);
        let (mut tau, mut json_out, mut dump) = (None, None, None);
        let (mut metrics, mut resamples, mut seed) = (false, 2000usize, 0u64);
        let (mut max_options, mut dump_probs, mut timing) = (None, None, false);
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
                "--metrics" => metrics = true,
                "--bootstrap" => {
                    resamples = args
                        .next()
                        .and_then(|v| v.parse().ok())
                        .filter(|&b| b > 0)
                        .unwrap_or_else(|| usage())
                }
                "--max-options" => {
                    max_options = Some(
                        args.next()
                            .and_then(|v| v.parse::<usize>().ok())
                            .filter(|&n| n >= 2)
                            .unwrap_or_else(|| usage()),
                    )
                }
                "--seed" => {
                    seed = args
                        .next()
                        .and_then(|v| v.parse().ok())
                        .unwrap_or_else(|| usage())
                }
                "--dump-probs" => dump_probs = args.next(),
                "--latency" => timing = true,
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
        let mut items = Vec::new();
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
                let mut opts: Vec<String> = r["opts"]
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
                let mut target: Vec<f64> = target.iter().map(|x| x / sum).collect();
                if let Some(n) = max_options {
                    let mut rng = Rng::new(seed ^ hash_str(line));
                    keep_options(&mut opts, &mut target, n, &mut rng);
                }
                let (ins, state) = (
                    r["ins"].as_str().unwrap_or(""),
                    r["state"].as_str().unwrap_or(""),
                );
                let t0 = Instant::now();
                let p = eng
                    .decide_raw(qt, ins, &opts, state)
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
                let input = timing.then(|| (ins.to_string(), opts, state.to_string()));
                items.push(Item {
                    pred: Pred::new(p, target, src.as_str()),
                    src,
                    qt,
                    input,
                });
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
        let boot = Bootstrap {
            resamples,
            seed,
            ..Bootstrap::default()
        };
        if metrics {
            add_metrics(&items, &mut rows, &boot);
        }
        let latency = timing.then(|| measure_latency(&bytes, tau, &items));
        if let Some(path) = json_out {
            let mut v = json!({"model": model, "n_params": eng.n_params(), "split": split, "sources": rows});
            if let Some(n) = max_options {
                v["max_options"] = json!(n);
            }
            if max_options.is_some() || metrics {
                v["seed"] = json!(seed);
            }
            if metrics {
                v["bootstrap"] = json!({"resamples": boot.resamples, "level": boot.level});
            }
            if let Some(l) = latency {
                v["latency"] = l;
            }
            std::fs::write(&path, serde_json::to_string_pretty(&v).unwrap()).unwrap();
            println!("wrote {path}");
        }
        if let Some(path) = dump {
            std::fs::write(&path, dump_lines.join("\n") + "\n").unwrap();
            println!("wrote {path}");
        }
        if let Some(path) = dump_probs {
            let lines: Vec<String> = items
                .iter()
                .map(|it| {
                    let p = &it.pred;
                    json!({"src": it.src, "t": it.qt.name(), "k": p.k(), "probs": p.probs,
                        "target": p.target, "gold": p.gold})
                    .to_string()
                })
                .collect();
            std::fs::write(&path, lines.join("\n") + "\n").unwrap();
            println!("wrote {path}");
        }
    }
}

fn main() {
    #[cfg(not(target_arch = "wasm32"))]
    native::main();
}
