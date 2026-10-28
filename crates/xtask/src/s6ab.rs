//! `cargo xtask s6ab [--arms a,b]`: the s6 ablation (docs/SPEC-S6.md §4) on the s3-pico shape: a
//! 2×2 grid {AdamW, Muon v2} × {softmax, Parallax} (issue #34), then single changes against it
//! (issue #36): XSA, SODA weight decay, no Newton–Schulz polish, exit self-distillation.
//!
//! Parallax's gain depends on the optimizer (arXiv 2605.29157 §4.3): the grid checks whether that
//! holds at our scale. Every arm and seed runs the same step budgets (`--steps`, so the schedule
//! does not depend on the machine): numeric MLM pretraining, decision training with QAT,
//! calibration, a ternary export and `sky-arena` at full depth (τ = 1). Steps are resumed through
//! the `.done` markers of the s5 stages. The aggregate (mean and sample std over seeds) of the
//! final ternary validation (accuracy, ECE, logit margin and effective rank of the last exit)
//! and of the arena scores goes to `runs/s6ab/summary.json`, with the training wall time: our
//! real runs have a time budget, and at equal time AdamW can beat Muon on language at small scale
//! (arXiv 2610.21555).
//!
//! `cargo xtask s6lr` is the learning-rate pilot run before it (docs/s6-prereg.md): AdamW and
//! Muon v2 at ×0.5, ×1 and ×2 of their default peak learning rate, on the same step budgets.
//! Both optimizers are tuned, since an untuned AdamW baseline inflates the gain of the other
//! optimizer (arXiv 2509.02046). Its choice goes to `runs/s6lr/summary.json` and to the
//! `--adamw-lr-scale` / `--muon-lr-scale` flags of `s6ab`.

use std::path::Path;

use anyhow::{bail, Context, Result};
use serde_json::{json, Map, Value};

use crate::loopcal::mean_std;
use crate::stages::{Stage, Step};

pub const SEEDS: [u64; 3] = [7, 11, 13];
/// Step budgets of the s3-pico runs of BENCH.md (8 + 15 minutes on an M5).
pub const PRETRAIN_STEPS: &str = "4800";
pub const DECIDE_STEPS: &str = "6400";
const EPISODES: &str = "30";
const GAMES: &str = "flappy,rescue,dodge,doom";
const RECORDS: &str = "data/decisions,data/games,data/numeric";
/// MLM text of the pretraining: the open corpus first, so that the held-out tail of the stream
/// stays the end of data/text, as for s3 (docs/s6-prereg.md, amendment 1).
const TEXT: &str = "data/text-open,data/text";
/// Default peak learning rates of `s2-pretrain` and `s2-decide` (`opt.build` in sky-train).
pub const PRETRAIN_LR: f64 = 3e-3;
pub const DECIDE_LR: f64 = 1e-3;
/// The pilot's multipliers of the default peak learning rate, log-spaced around it.
pub const PILOT_SCALES: [f64; 3] = [0.5, 1.0, 2.0];
/// The pilot runs one seed: it only makes a coarse choice, the ablation then runs three.
pub const PILOT_SEED: u64 = 7;
/// Below this gain in final ternary accuracy over ×1, the pilot keeps the default rate.
pub const PILOT_MIN_GAIN: f64 = 0.01;

/// Multipliers of the default peak learning rates (the pilot's choice).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LrScale {
    /// `--adam-lr` of the AdamW arms.
    pub adamw: f64,
    /// `--muon-lr` of the Muon arms (their AdamW parameters keep the default rate).
    pub muon: f64,
}

impl Default for LrScale {
    fn default() -> Self {
        Self {
            adamw: 1.0,
            muon: 1.0,
        }
    }
}

pub struct Arm {
    pub name: &'static str,
    pub about: &'static str,
    /// Config file in the checkout.
    pub config: &'static str,
    /// Optimizer flags of both training phases (`--no-muon`, `--wd-mode soda`, ...).
    pub opt: &'static [&'static str],
    /// Extra flags of the pretraining phase only (`--curriculum 0.1`).
    pub pretrain: &'static [&'static str],
    /// Extra flags of the decision phase only (`--exit-kd 0.5`).
    pub decide: &'static [&'static str],
}

/// The 2×2 grid, then the single changes of issue #36, each against `parallax-muon`
/// (or, for XSA, against `softmax-muon`).
pub fn arms() -> Vec<Arm> {
    const ADAMW: &[&str] = &["--no-muon"];
    vec![
        Arm {
            name: "softmax-adamw",
            about: "s3-pico attention, AdamW everywhere",
            config: "configs/s3-pico.json",
            opt: ADAMW,
            pretrain: &[],
            decide: &[],
        },
        Arm {
            name: "softmax-muon",
            about: "s3-pico attention, Muon v2 on the encoder matrices",
            config: "configs/s3-pico.json",
            opt: &[],
            pretrain: &[],
            decide: &[],
        },
        Arm {
            name: "parallax-adamw",
            about: "s6-pico (Parallax), AdamW everywhere",
            config: "configs/s6-pico.json",
            opt: ADAMW,
            pretrain: &[],
            decide: &[],
        },
        Arm {
            name: "parallax-muon",
            about: "s6-pico (Parallax), Muon v2 on the encoder matrices",
            config: "configs/s6-pico.json",
            opt: &[],
            pretrain: &[],
            decide: &[],
        },
        Arm {
            name: "xsa-muon",
            about: "s3-pico with exclusive self-attention (XSA), Muon v2",
            config: "configs/s3-pico-xsa.json",
            opt: &[],
            pretrain: &[],
            decide: &[],
        },
        Arm {
            name: "parallax-muon-soda",
            about: "parallax-muon with SODA weight decay (toward the initialization, 1/(k+2))",
            config: "configs/s6-pico.json",
            opt: &["--wd-mode", "soda"],
            pretrain: &[],
            decide: &[],
        },
        Arm {
            name: "parallax-muon-nopolish",
            about: "parallax-muon without the Newton–Schulz polish iterations",
            config: "configs/s6-pico.json",
            opt: &["--ns-polish", "0"],
            pretrain: &[],
            decide: &[],
        },
        Arm {
            name: "parallax-muon-kd",
            about: "parallax-muon with exit self-distillation (weight 0.5, T = 2)",
            config: "configs/s6-pico.json",
            opt: &[],
            pretrain: &[],
            decide: &["--exit-kd", "0.5"],
        },
        Arm {
            name: "parallax-muon-ema",
            about: "parallax-muon with a weight EMA (decay 0.999), the EMA exported (issue #38)",
            config: "configs/s6-pico.json",
            opt: &[],
            pretrain: &[],
            decide: &["--ema", "0.999"],
        },
        Arm {
            name: "parallax-muon-stableqat",
            about: "parallax-muon with the StableQAT gradient in the ternary QAT (A = 0.21, issue #40)",
            config: "configs/s6-pico.json",
            opt: &[],
            pretrain: &[],
            decide: &["--qat-grad-a", "0.21"],
        },
        Arm {
            name: "parallax-muon-curriculum",
            about: "parallax-muon with an easy-first curriculum over the first 10% of pretraining (issue #40)",
            config: "configs/s6-pico.json",
            opt: &[],
            pretrain: &["--curriculum", "0.1"],
            decide: &[],
        },
    ]
}

impl Arm {
    fn adamw(&self) -> bool {
        self.opt.contains(&"--no-muon")
    }

    /// The learning-rate flag of a phase whose default peak rate is `base`, if `scale` moves it.
    fn lr_flags(&self, scale: LrScale, base: f64) -> Vec<String> {
        let (flag, m) = if self.adamw() {
            ("--adam-lr", scale.adamw)
        } else {
            ("--muon-lr", scale.muon)
        };
        if m == 1.0 {
            return Vec::new();
        }
        vec![flag.to_string(), format!("{}", base * m)]
    }

    /// An arm with a weight EMA calibrates, exports and is judged on the EMA weights.
    fn ema(&self) -> bool {
        self.decide.contains(&"--ema")
    }

    /// The checkpoint of the decision run that is calibrated and exported.
    pub fn checkpoint(&self) -> &'static str {
        if self.ema() {
            "last-ema"
        } else {
            "last"
        }
    }

    /// The `mode` of the final validation that is reported.
    pub fn val_mode(&self) -> &'static str {
        if self.ema() {
            "ema-ternary"
        } else {
            "ternary"
        }
    }
}

/// The arms named in `only` (comma-separated), or all of them.
pub fn selected(only: Option<&str>) -> Result<Vec<Arm>> {
    let all = arms();
    let Some(only) = only else { return Ok(all) };
    let names: Vec<&str> = only
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .collect();
    for n in &names {
        if !all.iter().any(|a| a.name == *n) {
            bail!(
                "unknown arm {n:?} (arms: {})",
                all.iter().map(|a| a.name).collect::<Vec<_>>().join(", ")
            );
        }
    }
    Ok(all
        .into_iter()
        .filter(|a| names.contains(&a.name))
        .collect())
}

fn run_dir(a: &Arm, seed: u64) -> String {
    format!("runs/s6ab/{}-s{seed}", a.name)
}

/// The two training steps of a run: numeric MLM pretraining, then decision training with QAT.
fn train_steps(a: &Arm, id: &str, dir: &str, seed: u64, scale: LrScale) -> Vec<Step> {
    let s = seed.to_string();
    let config = format!("{{checkout}}/{}", a.config);
    let opt = a.opt;
    let (pre, dec) = (format!("{dir}/pretrain"), format!("{dir}/decide"));
    let (lr_pre, lr_dec) = (a.lr_flags(scale, PRETRAIN_LR), a.lr_flags(scale, DECIDE_LR));
    let mut args = vec![
        "s2-pretrain",
        "--config",
        &config,
        "--tokenizer",
        "data/tok/tok-s2-1024.json",
        "--text",
        TEXT,
        "--steps",
        PRETRAIN_STEPS,
        "--num-weight",
        "0.05",
        "--numeric",
        "data/numeric",
        "--numeric-weight",
        "0.1",
        "--seed",
        &s,
        "--out",
        &pre,
    ];
    args.extend(opt.iter().copied());
    args.extend(lr_pre.iter().map(String::as_str));
    args.extend(a.pretrain.iter().copied());
    let mut steps = vec![Step::new(&format!("{id}-pretrain"), "sky-train", &args).gpu()];
    let init = format!("{pre}/init");
    let mut args = vec![
        "s2-decide",
        "--config",
        &config,
        "--init",
        &init,
        "--records",
        RECORDS,
        "--weights",
        "games=0.6,numeric=0.1,decisions=0.3",
        "--batch",
        "128",
        "--steps",
        DECIDE_STEPS,
        "--qat-frac",
        "0.9",
        "--loop-dropout",
        "0.25",
        "--pad-to",
        "32",
        "--seed",
        &s,
        "--out",
        &dec,
    ];
    args.extend(opt.iter().copied());
    args.extend(lr_dec.iter().map(String::as_str));
    args.extend(a.decide.iter().copied());
    steps.push(Step::new(&format!("{id}-decide"), "sky-train", &args).gpu());
    steps
}

/// Every arm and seed, steps in order (one GPU job at a time; the arena runs on the CPU).
pub fn stage(arms: &[Arm], scale: LrScale) -> Stage {
    let mut steps = Vec::new();
    for a in arms {
        for seed in SEEDS {
            let dir = run_dir(a, seed);
            let id = format!("{}-s{seed}", a.name);
            steps.extend(train_steps(a, &id, &dir, seed, scale));
            let dec = format!("{dir}/decide");
            let last = format!("{dec}/{}", a.checkpoint());
            steps.push(
                Step::new(
                    &format!("{id}-calibrate"),
                    "sky-train",
                    &["s2-calibrate", "--run", &last, "--records", RECORDS],
                )
                .gpu(),
            );
            let gguf = format!("{dir}/model.gguf");
            steps.push(
                Step::new(
                    &format!("{id}-export"),
                    "sky-train",
                    &[
                        "s2-export",
                        "--run",
                        &last,
                        "--format",
                        "ternary",
                        "--out",
                        &gguf,
                    ],
                )
                .gpu(),
            );
            let arena = format!("{dir}/arena.json");
            steps.push(Step::new(
                &format!("{id}-arena"),
                "sky-arena",
                &[
                    &gguf,
                    "--game",
                    GAMES,
                    "--episodes",
                    EPISODES,
                    "--exit-threshold",
                    "1.0",
                    "--json",
                    &arena,
                ],
            ));
        }
    }
    Stage {
        name: "s6ab",
        about: "s6 ablation: {AdamW, Muon v2} x {softmax, Parallax}, plus XSA, SODA, no polish, exit KD",
        steps,
    }
}

/// The last ternary validation of a decide `log.jsonl`: accuracy, ECE and margin over every
/// group at the last exit, and the effective rank of the last exit.
pub fn final_val(log: &str, mode: &str) -> Result<[f64; 4]> {
    let entry: Value = log
        .lines()
        .rev()
        .filter_map(|l| serde_json::from_str::<Value>(l).ok())
        .find(|v| v["mode"] == mode && v.get("val").is_some())
        .with_context(|| format!("no {mode} validation in log.jsonl"))?;
    let val = entry["val"].as_object().context("val")?;
    // The deepest exit present, by its index (not by how many exits the entry holds).
    let last = val
        .keys()
        .filter_map(|k| Some((k.strip_prefix("exit")?.parse::<usize>().ok()?, k.clone())))
        .max()
        .map(|(_, k)| k)
        .context("no exit in val")?;
    let all = &val[&last]["all"];
    let get = |v: &Value, k: &str| v[k].as_f64().with_context(|| format!("{last}: no {k}"));
    Ok([
        get(all, "accuracy")?,
        get(all, "ece")?,
        get(all, "margin")?,
        get(&entry["erank"], &last)?,
    ])
}

const VAL: [&str; 4] = ["accuracy", "ece", "margin", "erank"];

/// Wall time of a run's two training phases, in minutes (`elapsed_secs` of their last checkpoint).
pub fn train_minutes(dir: &Path) -> Result<f64> {
    let mut secs = 0.0;
    for phase in ["pretrain", "decide"] {
        let p = dir.join(phase).join("last/state.json");
        let state: Value = serde_json::from_str(
            &std::fs::read_to_string(&p).with_context(|| p.display().to_string())?,
        )?;
        secs += state["elapsed_secs"]
            .as_f64()
            .with_context(|| format!("{}: no elapsed_secs", p.display()))?;
    }
    Ok(secs / 60.0)
}

fn stat(xs: &[f64]) -> Value {
    let (mean, std) = mean_std(xs);
    json!({"mean": mean, "std": std, "n": xs.len()})
}

/// Reads every run of `arms` and writes `runs/s6ab/summary.json`.
pub fn summarize(store: &Path, arms: &[Arm], scale: LrScale) -> Result<()> {
    let mut out = Vec::new();
    for a in arms {
        let mut val = vec![Vec::new(); VAL.len()];
        let mut minutes = Vec::new();
        let mut games: Vec<(String, Vec<f64>)> = Vec::new();
        for seed in SEEDS {
            let dir = store.join(run_dir(a, seed));
            let p = dir.join("decide/log.jsonl");
            let log = std::fs::read_to_string(&p).with_context(|| p.display().to_string())?;
            for (col, x) in val.iter_mut().zip(final_val(&log, a.val_mode())?) {
                col.push(x);
            }
            minutes.push(train_minutes(&dir)?);
            let p = dir.join("arena.json");
            let arena: Value = serde_json::from_str(
                &std::fs::read_to_string(&p).with_context(|| p.display().to_string())?,
            )?;
            for (game, v) in arena.as_object().context("arena report")? {
                let Some(score) = v["model"]["mean_score"].as_f64() else {
                    continue;
                };
                match games.iter_mut().find(|(g, _)| g == game) {
                    Some((_, xs)) => xs.push(score),
                    None => games.push((game.clone(), vec![score])),
                }
            }
        }
        if games.len() != GAMES.split(',').count() {
            bail!(
                "{}: expected the games {GAMES} in the arena reports",
                a.name
            );
        }
        let val: Map<String, Value> = VAL
            .iter()
            .zip(&val)
            .map(|(k, xs)| (k.to_string(), stat(xs)))
            .collect();
        let games: Map<String, Value> = games.iter().map(|(g, xs)| (g.clone(), stat(xs))).collect();
        out.push(json!({
            "name": a.name, "about": a.about, "config": a.config,
            "opt": a.opt, "decide": a.decide,
            "val": val, "arena_score": games, "train_minutes": stat(&minutes),
        }));
    }
    let summary = json!({
        "about": "s6 ablation on s3-pico: {AdamW, Muon v2} x {softmax, Parallax} (issue #34), XSA, SODA, no polish, exit KD (issue #36)",
        "seeds": SEEDS,
        "lr_scale": {"adamw": scale.adamw, "muon": scale.muon},
        "steps": {"pretrain": PRETRAIN_STEPS.parse::<u64>()?, "decide": DECIDE_STEPS.parse::<u64>()?},
        "arena": {"games": GAMES, "episodes": EPISODES.parse::<u64>()?, "exit_threshold": 1.0},
        "std": "sample standard deviation over seeds",
        "arms": out,
    });
    let p = store.join("runs/s6ab/summary.json");
    std::fs::write(&p, serde_json::to_string_pretty(&summary)? + "\n")?;
    println!("[s6ab] wrote {}", p.display());
    Ok(())
}

/// The pilot's two arms. AdamW is tuned as well as Muon v2: an untuned baseline inflates the
/// other optimizer's gain (arXiv 2509.02046).
pub fn pilot_arms() -> Vec<Arm> {
    arms()
        .into_iter()
        .filter(|a| matches!(a.name, "softmax-adamw" | "parallax-muon"))
        .collect()
}

fn pilot_scale(a: &Arm, m: f64) -> LrScale {
    if a.adamw() {
        LrScale {
            adamw: m,
            ..LrScale::default()
        }
    } else {
        LrScale {
            muon: m,
            ..LrScale::default()
        }
    }
}

fn pilot_dir(a: &Arm, m: f64) -> String {
    format!("runs/s6lr/{}-x{m}", a.name)
}

/// The learning-rate pilot: both training phases of each pilot arm at every scale, one seed.
/// Validation is enough to choose, so there is no calibration, export or arena.
pub fn pilot_stage() -> Stage {
    let mut steps = Vec::new();
    for a in pilot_arms() {
        for m in PILOT_SCALES {
            let id = format!("{}-x{m}", a.name);
            steps.extend(train_steps(
                &a,
                &id,
                &pilot_dir(&a, m),
                PILOT_SEED,
                pilot_scale(&a, m),
            ));
        }
    }
    Stage {
        name: "s6lr",
        about:
            "s6 learning-rate pilot: AdamW and Muon v2 at x0.5, x1 and x2 of the default peak rate",
        steps,
    }
}

/// The pilot's choice for one optimizer from `(scale, final ternary accuracy)`: the best scale,
/// unless it beats ×1 by less than `PILOT_MIN_GAIN` (one seed cannot resolve less).
pub fn pilot_choice(acc: &[(f64, f64)]) -> f64 {
    let base = acc.iter().find(|(m, _)| *m == 1.0).map(|&(_, x)| x);
    let best = acc
        .iter()
        .copied()
        .reduce(|b, x| if x.1 > b.1 { x } else { b });
    match (base, best) {
        (Some(b), Some((m, x))) if x - b >= PILOT_MIN_GAIN => m,
        _ => 1.0,
    }
}

/// Reads the pilot runs, writes `runs/s6lr/summary.json` and returns the chosen scales.
pub fn pilot_summarize(store: &Path) -> Result<LrScale> {
    let mut out = Vec::new();
    let mut scale = LrScale::default();
    for a in pilot_arms() {
        let mut acc = Vec::new();
        let mut runs = Vec::new();
        for m in PILOT_SCALES {
            let dir = store.join(pilot_dir(&a, m));
            let p = dir.join("decide/log.jsonl");
            let log = std::fs::read_to_string(&p).with_context(|| p.display().to_string())?;
            let v = final_val(&log, a.val_mode())?;
            acc.push((m, v[0]));
            let mut run: Map<String, Value> = VAL
                .iter()
                .zip(v)
                .map(|(k, x)| (k.to_string(), json!(x)))
                .collect();
            run.insert("scale".into(), json!(m));
            run.insert("train_minutes".into(), json!(train_minutes(&dir)?));
            runs.push(Value::Object(run));
        }
        let choice = pilot_choice(&acc);
        if a.adamw() {
            scale.adamw = choice;
        } else {
            scale.muon = choice;
        }
        out.push(json!({"name": a.name, "runs": runs, "choice": choice}));
    }
    let summary = json!({
        "about": "s6 learning-rate pilot (docs/s6-prereg.md): final ternary validation at each multiplier of the default peak rate",
        "seed": PILOT_SEED,
        "scales": PILOT_SCALES,
        "min_gain": PILOT_MIN_GAIN,
        "base_lr": {"pretrain": PRETRAIN_LR, "decide": DECIDE_LR},
        "arms": out,
        "lr_scale": {"adamw": scale.adamw, "muon": scale.muon},
    });
    let p = store.join("runs/s6lr/summary.json");
    std::fs::write(&p, serde_json::to_string_pretty(&summary)? + "\n")?;
    println!("[s6lr] wrote {}", p.display());
    println!(
        "[s6lr] next: cargo xtask s6ab --adamw-lr-scale {} --muon-lr-scale {}",
        scale.adamw, scale.muon
    );
    Ok(scale)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grid_and_single_changes() {
        let a = arms();
        assert_eq!(a.len(), 11);
        // The 2×2 grid: each attention under AdamW and under Muon v2.
        for config in ["configs/s3-pico.json", "configs/s6-pico.json"] {
            let adamw: Vec<bool> = a
                .iter()
                .filter(|a| a.config == config && !a.name.contains("-muon-"))
                .map(|a| a.opt.contains(&"--no-muon"))
                .collect();
            assert_eq!(adamw, [true, false], "{config}");
        }
        let s = stage(&a, LrScale::default());
        assert_eq!(s.steps.len(), 11 * SEEDS.len() * 5);
        // At the default scale, no learning-rate flag.
        assert!(s
            .steps
            .iter()
            .all(|t| !t.args.iter().any(|x| x.ends_with("-lr"))));
        // The AdamW arms pass --no-muon to both training steps, the others never; the exit
        // distillation only reaches the decision step.
        for step in &s.steps {
            let trains = step.name.ends_with("-pretrain") || step.name.ends_with("-decide");
            let no_muon = step.args.iter().any(|a| a == "--no-muon");
            assert_eq!(
                no_muon,
                step.name.contains("-adamw-") && trains,
                "{}",
                step.name
            );
            let soda = step.args.windows(2).any(|w| w == ["--wd-mode", "soda"]);
            assert_eq!(
                soda,
                step.name.contains("-soda-") && trains,
                "{}",
                step.name
            );
            let kd = step.args.iter().any(|a| a == "--exit-kd");
            assert_eq!(
                kd,
                step.name.contains("-kd-") && step.name.ends_with("-decide"),
                "{}",
                step.name
            );
            // The curriculum reaches the pretraining only, StableQAT the decision only.
            let curriculum = step.args.iter().any(|a| a == "--curriculum");
            assert_eq!(
                curriculum,
                step.name.contains("-curriculum-") && step.name.ends_with("-pretrain"),
                "{}",
                step.name
            );
            let stable = step.args.iter().any(|a| a == "--qat-grad-a");
            assert_eq!(
                stable,
                step.name.contains("-stableqat-") && step.name.ends_with("-decide"),
                "{}",
                step.name
            );
            // The EMA arm calibrates, exports and plays its EMA weights.
            let ema = step.name.contains("-ema-");
            if step.name.ends_with("-calibrate") || step.name.ends_with("-export") {
                assert_eq!(
                    step.args.iter().any(|a| a.ends_with("/decide/last-ema")),
                    ema,
                    "{}",
                    step.name
                );
            }
        }
        assert_eq!(selected(Some("xsa-muon, parallax-muon")).unwrap().len(), 2);
        assert!(selected(Some("nope")).is_err());
        assert_eq!(selected(None).unwrap().len(), 11);
        let e = a.iter().find(|a| a.name == "parallax-muon-ema").unwrap();
        assert_eq!((e.checkpoint(), e.val_mode()), ("last-ema", "ema-ternary"));
    }

    #[test]
    fn learning_rate_scales() {
        let scale = LrScale {
            adamw: 0.5,
            muon: 2.0,
        };
        let s = stage(
            &selected(Some("softmax-adamw,parallax-muon")).unwrap(),
            scale,
        );
        let lr = |name: &str| -> Vec<String> {
            let step = s.steps.iter().find(|t| t.name == name).unwrap();
            let i = step.args.iter().position(|x| x.ends_with("-lr")).unwrap();
            step.args[i..i + 2].to_vec()
        };
        assert_eq!(lr("softmax-adamw-s7-pretrain"), ["--adam-lr", "0.0015"]);
        assert_eq!(lr("softmax-adamw-s7-decide"), ["--adam-lr", "0.0005"]);
        assert_eq!(lr("parallax-muon-s7-pretrain"), ["--muon-lr", "0.006"]);
        assert_eq!(lr("parallax-muon-s7-decide"), ["--muon-lr", "0.002"]);

        // The pilot: 2 arms x 3 scales x 2 training steps, ×1 with no flag.
        let p = pilot_stage();
        assert_eq!(p.steps.len(), 2 * PILOT_SCALES.len() * 2);
        let flagged: Vec<&str> = p
            .steps
            .iter()
            .filter(|t| t.args.iter().any(|x| x.ends_with("-lr")))
            .map(|t| t.name.as_str())
            .collect();
        assert_eq!(flagged.len(), 2 * 2 * 2);
        assert!(flagged.iter().all(|n| !n.contains("-x1-")));
        assert!(p.steps[0]
            .args
            .iter()
            .any(|x| x == "runs/s6lr/softmax-adamw-x0.5/pretrain"));

        // The choice keeps ×1 unless another scale wins by at least PILOT_MIN_GAIN.
        assert_eq!(pilot_choice(&[(0.5, 0.60), (1.0, 0.61), (2.0, 0.615)]), 1.0);
        assert_eq!(pilot_choice(&[(0.5, 0.63), (1.0, 0.61), (2.0, 0.62)]), 0.5);
        assert_eq!(pilot_choice(&[(0.5, 0.60), (1.0, 0.61), (2.0, 0.64)]), 2.0);
        assert_eq!(pilot_choice(&[(0.5, 0.60), (2.0, 0.64)]), 1.0);
    }

    #[test]
    fn train_minutes_sums_both_phases() {
        let dir = std::env::temp_dir().join(format!("s6ab-minutes-{}", std::process::id()));
        for (phase, secs) in [("pretrain", 480.0), ("decide", 900.0)] {
            let d = dir.join(phase).join("last");
            std::fs::create_dir_all(&d).unwrap();
            std::fs::write(
                d.join("state.json"),
                json!({"elapsed_secs": secs}).to_string(),
            )
            .unwrap();
        }
        assert_eq!(train_minutes(&dir).unwrap(), 23.0);
        std::fs::remove_dir_all(&dir).unwrap();
        assert!(train_minutes(&dir).is_err());
    }

    #[test]
    fn final_val_reads_the_last_ternary_entry() {
        let log = [
            r#"{"step": 10, "mode": "ternary", "val": {"exit0": {"all": {"accuracy": 0.1, "ece": 0.5, "margin": -1.0}}}, "erank": {"exit0": 3.0}}"#,
            r#"{"step": 20, "loss": 1.0}"#,
            r#"{"step": 30, "mode": "f32", "val": {"exit1": {"all": {"accuracy": 0.9, "ece": 0.0, "margin": 9.0}}}, "erank": {"exit1": 9.0}}"#,
            r#"{"step": 30, "mode": "ternary", "val": {"exit0": {"all": {"accuracy": 0.4, "ece": 0.1, "margin": 0.5}}, "exit1": {"all": {"accuracy": 0.6, "ece": 0.05, "margin": 1.5}}}, "erank": {"exit0": 20.0, "exit1": 30.5}}"#,
        ]
        .join("\n");
        assert_eq!(final_val(&log, "ternary").unwrap(), [0.6, 0.05, 1.5, 30.5]);
        assert!(final_val("{}", "ternary").is_err());
        let ema = format!(
            "{log}\n{}",
            r#"{"step": 30, "mode": "ema-ternary", "val": {"exit1": {"all": {"accuracy": 0.7, "ece": 0.04, "margin": 2.0}}}, "erank": {"exit1": 31.0}}"#
        );
        assert_eq!(
            final_val(&ema, "ema-ternary").unwrap(),
            [0.7, 0.04, 2.0, 31.0]
        );
        assert_eq!(final_val(&ema, "ternary").unwrap(), [0.6, 0.05, 1.5, 30.5]);
    }
}
