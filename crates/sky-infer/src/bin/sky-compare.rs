//! Paired comparison of models scored item by item by `sky-eval --dump-probs`: bootstrap
//! differences (variant − reference) with 95% intervals for accuracy, NLL, Brier and top-label
//! ECE, non-inferiority verdicts for accuracy and ECE, and p-values.
//!
//! ```text
//! sky-compare <reference.jsonl> <variant.jsonl>... [--labels ref,var,...] [--acc-margin 0.02]
//!             [--ece-margin 0.01] [--bootstrap 2000] [--seed 0] [--json out.json]
//! ```
//!
//! Every file must hold the same items in the same order (`sky-eval` on the same records with
//! the same `--split`, `--max`, `--max-options` and `--seed`): line i must agree on `src`, `k`
//! and `target` in every file, else nothing is compared. Each variant is compared with the
//! reference on the same resamples (`--bootstrap` of them, seed `--seed`). Non-inferiority
//! (`sky_metrics::Paired::non_inferior`): accuracy passes when the upper 95% bound of
//! reference − variant is ≤ `--acc-margin` (2 points), ECE when the upper 95% bound of
//! variant − reference is ≤ `--ece-margin`. The p-values (two-sided, `sky_metrics::p_value`)
//! of each metric are Holm-adjusted across the variants (`p_holm`; equal to `p_value` with a
//! single variant). Labels default to the file stems.
//!
//! A binary of `sky-infer` rather than of `sky-metrics`, which stays dependency-free: reading
//! the dumps takes `serde_json`.

#[cfg(not(target_arch = "wasm32"))]
mod native {
    use std::path::Path;

    use serde_json::{json, Map, Value};
    use sky_metrics::{holm, Better, Bootstrap, Metric, Paired, Pred};

    fn usage() -> ! {
        eprintln!("usage: sky-compare <reference.jsonl> <variant.jsonl>... [--labels ref,var,...] [--acc-margin 0.02] [--ece-margin 0.01] [--bootstrap 2000] [--seed 0] [--json out.json]");
        std::process::exit(2);
    }

    fn fail(msg: String) -> ! {
        eprintln!("sky-compare: {msg}");
        std::process::exit(1);
    }

    /// One line of a `--dump-probs` file.
    struct Item {
        src: String,
        pred: Pred,
    }

    fn read(path: &str) -> Vec<Item> {
        let text = std::fs::read_to_string(path).unwrap_or_else(|e| fail(format!("{path}: {e}")));
        let mut out = Vec::new();
        for (i, line) in text.lines().enumerate() {
            if line.trim().is_empty() {
                continue;
            }
            let at = format!("{path}:{}", i + 1);
            let v: Value =
                serde_json::from_str(line).unwrap_or_else(|e| fail(format!("{at}: {e}")));
            let floats = |key: &str| -> Vec<f64> {
                v[key]
                    .as_array()
                    .map(|a| a.iter().filter_map(Value::as_f64).collect())
                    .unwrap_or_default()
            };
            let (probs, target) = (floats("probs"), floats("target"));
            let k = v["k"].as_u64().unwrap_or(0) as usize;
            if k == 0 || probs.len() != k || target.len() != k {
                fail(format!("{at}: expected k = {k} probabilities and targets"));
            }
            let src = v["src"].as_str().unwrap_or("?").to_string();
            let pred = Pred::new(probs, target, src.as_str());
            if v["gold"].as_u64() != Some(pred.gold as u64) {
                fail(format!("{at}: gold is not the argmax of the target"));
            }
            out.push(Item { src, pred });
        }
        out
    }

    /// Stops unless `b` holds the items of `a`: same `src`, `k` and `target`, line by line.
    fn check_same(a: &[Item], b: &[Item], pa: &str, pb: &str) {
        if a.len() != b.len() {
            fail(format!("{pa} has {} items, {pb} {}", a.len(), b.len()));
        }
        for (i, (x, y)) in a.iter().zip(b).enumerate() {
            let same = x.src == y.src
                && x.pred.k() == y.pred.k()
                && (x.pred.target.iter().zip(&y.pred.target)).all(|(u, v)| (u - v).abs() <= 1e-9);
            if !same {
                fail(format!(
                    "item {} differs between {pa} and {pb} (src, k or target): not the same items",
                    i + 1
                ));
            }
        }
    }

    /// Prints and returns the non-inferiority verdict of `m`: the upper bound of the variant's
    /// degradation against `margin`.
    fn verdict(m: Metric, d: &Paired, margin: f64) -> Value {
        let (upper, pass) = (
            d.worse_upper(m.better()),
            d.non_inferior(m.better(), margin),
        );
        let bound = match m.better() {
            Better::Higher => "upper95(reference - variant)",
            Better::Lower => "upper95(variant - reference)",
        };
        println!(
            "non-inferiority {:<9} {bound} = {upper:+.4}, margin {margin}: {}",
            m.name(),
            if pass { "PASS" } else { "FAIL" }
        );
        json!({"test": format!("{bound} <= margin"), "upper": upper, "margin": margin, "pass": pass})
    }

    pub fn main() {
        let mut args = std::env::args().skip(1);
        let mut paths: Vec<String> = Vec::new();
        let mut labels: Option<Vec<String>> = None;
        let (mut acc_margin, mut ece_margin) = (0.02f64, 0.01f64);
        let (mut resamples, mut seed, mut json_out) = (2000usize, 0u64, None);
        while let Some(a) = args.next() {
            match a.as_str() {
                "--labels" => {
                    labels = args
                        .next()
                        .map(|v| v.split(',').map(str::to_string).collect())
                }
                "--acc-margin" => {
                    acc_margin = args
                        .next()
                        .and_then(|v| v.parse().ok())
                        .unwrap_or_else(|| usage())
                }
                "--ece-margin" => {
                    ece_margin = args
                        .next()
                        .and_then(|v| v.parse().ok())
                        .unwrap_or_else(|| usage())
                }
                "--bootstrap" => {
                    resamples = args
                        .next()
                        .and_then(|v| v.parse().ok())
                        .filter(|&b| b > 0)
                        .unwrap_or_else(|| usage())
                }
                "--seed" => {
                    seed = args
                        .next()
                        .and_then(|v| v.parse().ok())
                        .unwrap_or_else(|| usage())
                }
                "--json" => json_out = args.next(),
                "-h" | "--help" => usage(),
                _ if a.starts_with("--") => usage(),
                _ => paths.push(a),
            }
        }
        if paths.len() < 2 {
            usage()
        }
        let labels = labels.unwrap_or_else(|| {
            paths
                .iter()
                .map(|p| {
                    Path::new(p)
                        .file_stem()
                        .map_or(p.clone(), |s| s.to_string_lossy().into_owned())
                })
                .collect()
        });
        if labels.len() != paths.len() {
            fail(format!("{} labels for {} files", labels.len(), paths.len()));
        }
        let sets: Vec<Vec<Item>> = paths.iter().map(|p| read(p)).collect();
        for (p, s) in paths.iter().zip(&sets).skip(1) {
            check_same(&sets[0], s, &paths[0], p);
        }
        if sets[0].is_empty() {
            fail(format!("{}: no items", paths[0]));
        }
        let preds: Vec<Vec<Pred>> = sets
            .into_iter()
            .map(|s| s.into_iter().map(|it| it.pred).collect())
            .collect();
        let (reference, variants) = preds.split_first().expect("two files at least");
        let boot = Bootstrap {
            resamples,
            seed,
            ..Bootstrap::default()
        };
        // results[variant][metric], then the Holm correction of each metric across the variants.
        let results: Vec<Vec<Paired>> = variants
            .iter()
            .map(|v| {
                Metric::ALL
                    .iter()
                    .map(|&m| boot.metric_paired(m, reference, v))
                    .collect()
            })
            .collect();
        let mut p_holm = vec![[0.0; Metric::ALL.len()]; results.len()];
        for j in 0..Metric::ALL.len() {
            let raw: Vec<f64> = results.iter().map(|r| r[j].p_value).collect();
            for (v, p) in holm(&raw).into_iter().enumerate() {
                p_holm[v][j] = p;
            }
        }
        println!(
            "reference {} ({}): {} items; {} bootstrap resamples, seed {seed}; diff = variant - reference",
            labels[0],
            paths[0],
            reference.len(),
            boot.resamples
        );
        let mut comparisons = Vec::new();
        for (vi, r) in results.iter().enumerate() {
            let (label, path) = (&labels[vi + 1], &paths[vi + 1]);
            println!("\nvariant {label} ({path})");
            println!(
                "{:<9} {:>10} {:>10} {:>9} {:>20} {:>8} {:>8}",
                "metric", "reference", "variant", "diff", "95% CI", "p", "p_holm"
            );
            let mut metrics = Map::new();
            for (j, (m, d)) in Metric::ALL.iter().zip(r).enumerate() {
                let ci = format!("[{:+.4}, {:+.4}]", d.diff.lo, d.diff.hi);
                println!(
                    "{:<9} {:>10.4} {:>10.4} {:>+9.4} {ci:>20} {:>8.4} {:>8.4}",
                    m.name(),
                    d.reference,
                    d.variant,
                    d.diff.value,
                    d.p_value,
                    p_holm[vi][j]
                );
                metrics.insert(
                    m.name().into(),
                    json!({"reference": d.reference, "variant": d.variant, "diff": d.diff.value,
                        "ci95": [d.diff.lo, d.diff.hi], "p_value": d.p_value, "p_holm": p_holm[vi][j]}),
                );
            }
            let of = |m: Metric| &r[Metric::ALL.iter().position(|&x| x == m).expect("headline")];
            let accuracy = verdict(Metric::Accuracy, of(Metric::Accuracy), acc_margin);
            let ece = verdict(Metric::Ece, of(Metric::Ece), ece_margin);
            comparisons.push(
                json!({"variant": {"label": label, "path": path}, "metrics": metrics,
                "non_inferiority": {"accuracy": accuracy, "ece": ece}}),
            );
        }
        if let Some(path) = json_out {
            let v = json!({
                "reference": {"label": labels[0], "path": paths[0]},
                "n": reference.len(),
                "bootstrap": {"resamples": boot.resamples, "seed": boot.seed, "level": boot.level},
                "diff": "variant - reference",
                "p_holm": "Holm across the variants, per metric",
                "comparisons": comparisons,
            });
            std::fs::write(&path, serde_json::to_string_pretty(&v).unwrap() + "\n").unwrap();
            println!("wrote {path}");
        }
    }
}

fn main() {
    #[cfg(not(target_arch = "wasm32"))]
    native::main();
}
