//! `cargo xtask loopcal`: early exits calibrated on i.i.d. questions against closed-loop play,
//! over several arena seeds (issue #29, exploratory).
//!
//! The published GGUFs of `site/public/models/` are rewritten into `runs/loopcal/<config>/` with
//! their exit thresholds set per type (`gguf-set`), then played by `sky-arena` for every config
//! and seed. Model pairing follows the site: s2-nano plays flappy, rescue and dodge, and its Doom
//! fine-tune `s2-nano-doom` plays doom (s2-nano never saw Doom); s3-pico plays all four games.
//! Steps are resumed through the `.done` markers of the s5 stages. The aggregate (mean and
//! sample std over seeds, per config and game) goes to `runs/loopcal/summary.json` and is copied
//! to `site/public/results/loopcal.json` in the checkout.

use std::path::Path;

use anyhow::{anyhow, bail, Context, Result};
use serde_json::{json, Map, Value};

use crate::stages::{Stage, Step};

pub const SEEDS: [u64; 5] = [7, 11, 13, 17, 19];
pub const EPISODES: &str = "30";
const S2_GAMES: &str = "flappy,rescue,dodge";
const ALL_GAMES: &str = "flappy,rescue,dodge,doom";
/// The exit-threshold keys, in the order of `S2Config::exit_thresholds`.
const TYPES: [&str; 3] = ["choice", "score", "noul"];

/// How a config sets τ.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Tau {
    /// One value per type (choice, score, noul).
    PerType([f32; 3]),
    /// The value the published file ships with.
    Shipped,
}

pub struct Config {
    pub name: &'static str,
    pub about: &'static str,
    /// (published model stem, games it plays).
    pub models: &'static [(&'static str, &'static str)],
    pub tau: Tau,
}

const S2: &[(&str, &str)] = &[("s2-nano", S2_GAMES), ("s2-nano-doom", "doom")];
const S3: &[(&str, &str)] = &[("s3-pico", ALL_GAMES)];

pub fn configs() -> Vec<Config> {
    vec![
        Config {
            name: "s2-nano-iid",
            about: "s2-nano, τ calibrated on i.i.d. val (0.62 / 0.62 / 0.69)",
            models: S2,
            tau: Tau::PerType([0.62, 0.62, 0.69]),
        },
        Config {
            name: "s2-nano-closed",
            about: "s2-nano, closed-loop τ 0.95 (published)",
            models: S2,
            tau: Tau::PerType([0.95; 3]),
        },
        Config {
            name: "s3-pico-full",
            about: "s3-pico, τ = 1 (full depth)",
            models: S3,
            tau: Tau::PerType([1.0; 3]),
        },
        Config {
            name: "s3-pico-shipped",
            about: "s3-pico, shipped closed-loop τ",
            models: S3,
            tau: Tau::Shipped,
        },
        Config {
            name: "s3-pico-0.5",
            about: "s3-pico, τ = 0.5",
            models: S3,
            tau: Tau::PerType([0.5; 3]),
        },
    ]
}

/// The per-type thresholds a published file ships with.
pub fn shipped_tau(gguf: &Path) -> Result<[f32; 3]> {
    let bytes = std::fs::read(gguf).with_context(|| gguf.display().to_string())?;
    let g = sky_gguf::Gguf::parse(&bytes).map_err(|e| anyhow!("{}: {e:?}", gguf.display()))?;
    let mut out = [0f32; 3];
    for (o, t) in out.iter_mut().zip(TYPES) {
        let key = format!("skycmd-s2.exit_threshold.{t}");
        *o = g
            .get_f64(&key)
            .with_context(|| format!("{}: no {key}", gguf.display()))? as f32;
    }
    Ok(out)
}

fn tau_of(c: &Config, checkout: &Path, model: &str) -> Result<[f32; 3]> {
    Ok(match c.tau {
        Tau::PerType(t) => t,
        Tau::Shipped => shipped_tau(&checkout.join(format!("site/public/models/{model}.gguf")))?,
    })
}

fn gguf(c: &Config, model: &str) -> String {
    format!("runs/loopcal/{}/{model}.gguf", c.name)
}

fn arena_json(c: &Config, model: &str, seed: u64) -> String {
    format!("runs/loopcal/{}/{model}-s{seed}.json", c.name)
}

/// Copies of the published files with τ set (run first, one at a time).
pub fn prepare(checkout: &Path) -> Result<Stage> {
    let mut steps = Vec::new();
    for c in configs() {
        for (m, _) in c.models {
            let tau = tau_of(&c, checkout, m)?;
            let mut args = vec![
                format!("{{checkout}}/site/public/models/{m}.gguf"),
                gguf(&c, m),
            ];
            args.extend(
                TYPES
                    .iter()
                    .zip(tau)
                    .map(|(t, v)| format!("skycmd-s2.exit_threshold.{t}={v}")),
            );
            let args: Vec<&str> = args.iter().map(String::as_str).collect();
            steps.push(Step::new(&format!("{}-{m}", c.name), "gguf-set", &args));
        }
    }
    Ok(Stage {
        name: "loopcal-gguf",
        about: "published GGUFs with τ set per config",
        steps,
    })
}

/// The arena runs: every config, model and seed (independent of each other).
pub fn arena() -> Stage {
    let mut steps = Vec::new();
    for c in configs() {
        for (m, games) in c.models {
            for seed in SEEDS {
                let (g, j, s) = (gguf(&c, m), arena_json(&c, m, seed), seed.to_string());
                steps.push(Step::new(
                    &format!("{}-{m}-s{seed}", c.name),
                    "sky-arena",
                    &[
                        &g,
                        "--game",
                        games,
                        "--episodes",
                        EPISODES,
                        "--seed",
                        &s,
                        "--json",
                        &j,
                    ],
                ));
            }
        }
    }
    Stage {
        name: "loopcal",
        about: "sky-arena per config and seed",
        steps,
    }
}

/// Mean and sample standard deviation (0 for a single value).
pub fn mean_std(xs: &[f64]) -> (f64, f64) {
    let n = xs.len() as f64;
    let mean = xs.iter().sum::<f64>() / n;
    let var = if xs.len() > 1 {
        xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0)
    } else {
        0.0
    };
    (mean, var.sqrt())
}

/// The metrics aggregated per game, read from one `sky-arena --json` game entry.
const METRICS: [&str; 5] = ["score", "crashes", "vetoes", "mean_layers", "oracle_score"];

fn metric(game: &Value, name: &str) -> Option<f64> {
    match name {
        "score" => game["model"]["mean_score"].as_f64(),
        "crashes" => game["model"]["crashes"].as_f64(),
        "vetoes" => game["model"]["vetoes"].as_f64(),
        "mean_layers" => game["mean_layers_used"].as_f64(),
        "oracle_score" => game["oracle"]["mean_score"].as_f64(),
        _ => None,
    }
}

/// Aggregates the arena reports of one config (one report per model and seed) into
/// `{game: {metric: {mean, std, n}}}`.
pub fn aggregate(reports: &[Value]) -> Result<Value> {
    let mut per: Vec<(String, Vec<Vec<f64>>)> = Vec::new();
    for r in reports {
        let obj = r.as_object().context("arena report is not an object")?;
        for (game, v) in obj {
            if !v.is_object() || v.get("model").is_none() {
                continue;
            }
            let i = match per.iter().position(|(g, _)| g == game) {
                Some(i) => i,
                None => {
                    per.push((game.clone(), vec![Vec::new(); METRICS.len()]));
                    per.len() - 1
                }
            };
            for (k, name) in METRICS.iter().enumerate() {
                let x = metric(v, name).with_context(|| format!("{game}: no {name}"))?;
                per[i].1[k].push(x);
            }
        }
    }
    let mut out = Map::new();
    for (game, cols) in per {
        let mut g = Map::new();
        for (name, xs) in METRICS.iter().zip(cols) {
            let (mean, std) = mean_std(&xs);
            g.insert(
                name.to_string(),
                json!({"mean": mean, "std": std, "n": xs.len()}),
            );
        }
        out.insert(game, Value::Object(g));
    }
    Ok(Value::Object(out))
}

/// Reads every arena report and writes the summary to the store and the checkout.
pub fn summarize(store: &Path, checkout: &Path) -> Result<()> {
    let mut cfgs = Vec::new();
    for c in configs() {
        let mut reports = Vec::new();
        let mut taus = Map::new();
        for (m, _) in c.models {
            let t = tau_of(&c, checkout, m)?;
            taus.insert(
                m.to_string(),
                json!(t.map(|x| (x as f64 * 1e4).round() / 1e4)),
            );
            for seed in SEEDS {
                let p = store.join(arena_json(&c, m, seed));
                let s = std::fs::read_to_string(&p).with_context(|| p.display().to_string())?;
                reports.push(serde_json::from_str(&s)?);
            }
        }
        let games = aggregate(&reports)?;
        let want: usize = c.models.iter().map(|(_, g)| g.split(',').count()).sum();
        if games.as_object().map_or(0, Map::len) != want {
            bail!("{}: expected {want} games in the reports", c.name);
        }
        cfgs.push(json!({
            "name": c.name, "about": c.about, "tau": taus,
            "models": c.models.iter().map(|(m, g)| json!({"model": m, "games": g})).collect::<Vec<_>>(),
            "games": games,
        }));
    }
    let summary = json!({
        "about": "Early exits: i.i.d. vs closed-loop τ, sky-arena over 5 seeds (exploratory, issue #29)",
        "seeds": SEEDS, "episodes": EPISODES.parse::<u64>()?,
        "std": "sample standard deviation over seeds",
        "configs": cfgs,
    });
    let text = serde_json::to_string_pretty(&summary)? + "\n";
    let a = store.join("runs/loopcal/summary.json");
    std::fs::write(&a, &text)?;
    let b = checkout.join("site/public/results/loopcal.json");
    std::fs::create_dir_all(b.parent().context("parent")?)?;
    std::fs::write(&b, &text)?;
    println!("[loopcal] wrote {} and {}", a.display(), b.display());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn configs_cover_the_five_settings() {
        let c = configs();
        let names: Vec<_> = c.iter().map(|c| c.name).collect();
        assert_eq!(
            names,
            [
                "s2-nano-iid",
                "s2-nano-closed",
                "s3-pico-full",
                "s3-pico-shipped",
                "s3-pico-0.5"
            ]
        );
        assert_eq!(c[0].tau, Tau::PerType([0.62, 0.62, 0.69]));
        assert_eq!(c[1].tau, Tau::PerType([0.95; 3]));
        assert_eq!(c[3].tau, Tau::Shipped);
        // Every config plays the four games exactly once, Doom by a model trained on it.
        for c in &c {
            let mut games: Vec<&str> = c.models.iter().flat_map(|(_, g)| g.split(',')).collect();
            games.sort_unstable();
            assert_eq!(games, ["dodge", "doom", "flappy", "rescue"], "{}", c.name);
        }
        assert_eq!(S2[1], ("s2-nano-doom", "doom"));
        // 5 configs: 2 × 2 models + 3 × 1 model, times 5 seeds.
        assert_eq!(arena().steps.len(), 7 * SEEDS.len());
        let s = &arena().steps[0];
        assert_eq!(s.name, "s2-nano-iid-s2-nano-s7");
        assert!(s.args.iter().all(|a| !a.starts_with('/')));
    }

    #[test]
    fn aggregate_gives_mean_and_sample_std() {
        let rep = |score: f64, crashes: u64, layers: f64| {
            json!({"model": "x", "arch": "a",
                   "flappy": {"model": {"mean_score": score, "crashes": crashes, "vetoes": 2},
                              "oracle": {"mean_score": 200.0},
                              "mean_layers_used": layers}})
        };
        let a = aggregate(&[rep(10.0, 30, 1.0), rep(20.0, 10, 2.0), rep(30.0, 20, 3.0)]).unwrap();
        let f = &a["flappy"];
        assert_eq!(f["score"]["mean"], 20.0);
        assert_eq!(f["score"]["std"], 10.0);
        assert_eq!(f["crashes"]["mean"], 20.0);
        assert_eq!(f["vetoes"]["std"], 0.0);
        assert_eq!(f["mean_layers"]["mean"], 2.0);
        assert_eq!(f["oracle_score"]["n"], 3);
        assert_eq!(a.as_object().unwrap().len(), 1);
        assert_eq!(mean_std(&[4.0]), (4.0, 0.0));
        // a game entry without its metrics is an error, not a silent zero
        assert!(aggregate(&[json!({"flappy": {"model": {}}})]).is_err());
    }
}
