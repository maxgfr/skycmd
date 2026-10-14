//! The s5-lite experiments (Phase 4) and their evaluation, as pre-registered in
//! `docs/s5-prereg.md`. Stages whose runs depend on earlier ones choose them on validation files
//! only (`runs/s5-<run>/val.json`, lowest NLL over every val record) and freeze the choice in
//! `runs/s5-select/<stage>.json` the first time they are built.
//!
//! - `explore`: A′ and B-i × LR {5e-5, 1e-4, 2e-4} × init {ettin-17m, ettin-reranker-17m}, seed 0,
//!   [`EXPLORE_STEPS`] steps.
//! - `explore-b-ii`: B-ii at the best (init, LR) of B-i.
//! - `final`: A′ and B* (B-i or B-ii, the lower val NLL at that configuration) × seeds 0, 1, 2,
//!   and A × seed 0 at the configuration of A′. Their step count (also that of bet 2) is fixed
//!   mechanically from the throughput of the exploration only ([`final_steps`]).
//! - `eval` (CPU): for each finished run (`--runs a,b` or every run of `final` and `lambda*`):
//!   f32, q8_0 and q4_0 exports, temperatures fitted on the English `val` split of each format,
//!   the calibrated model on the `cal` split (escalation thresholds), every test set in f32, the
//!   fitted bench in q8_0 and q4_0 (Phase 5; q4_0 is secondary), the demo cases in every format.
//! - `eval-q8`, `eval-q4` (CPU, `--runs`): the other test sets in q8_0 or q4_0.
//! - `lambda`: bet 2, λ ∈ {0.1, 0.3} at seed 0 on its support (B* when bet 1 succeeded in
//!   `runs/s5-report/pari1.json`, else A′); `lambda-final`: λ* × seeds 1 and 2, λ* the lower val
//!   MAD_neg among the λ whose val accuracy is at least that of λ = 0 (seed 0) − 0.01.
//! - `report` (CPU): the pre-registered analyses (`sky-s5-report`) into `runs/s5-report/`.

use std::path::Path;

use anyhow::{bail, Context, Result};
use serde_json::{json, Value};

use crate::stages::{Stage, Step};

pub const EXPLORE_STEPS: u64 = 600;
/// Candidate step counts of the finalists and of bet 2.
pub const FINAL_STEP_CHOICES: [u64; 4] = [2000, 3000, 4000, 6000];
/// GPU minutes allowed for the five waves of runs that follow the exploration (three waves of
/// finalists at `--jobs 3`, two of bet 2).
pub const BUDGET_MINUTES: f64 = 180.0;
pub const WAVES: f64 = 5.0;
pub const LRS: [&str; 3] = ["5e-5", "1e-4", "2e-4"];
/// (short name, `models/s5-base/` directory).
pub const INITS: [(&str, &str); 2] = [("e17", "ettin-17m"), ("r17", "ettin-reranker-17m")];
/// Cap of MLX's allocator cache per training process (`SKY_MLX_CACHE_GB`), in GB.
pub const MLX_CACHE_GB: &str = "1";
/// Weight of the symmetric KL on the pairs, the same for every variant.
pub const KL: &str = "0.5";
pub const LAMBDAS: [&str; 2] = ["0.1", "0.3"];
/// Steps of the long run of the /live demo (amendment 3, outside the bets).
pub const LONG_STEPS: u64 = 8000;
/// Name of that run.
pub const LONG_RUN: &str = "long-b-s0";
pub const SEEDS: [u64; 3] = [0, 1, 2];
/// The original coherence battery of 2609.33209 as s5 records (`sky-data s5-battery`).
pub const BATTERY: &str = "data/s5/battery/coherence_2609_33209.jsonl";

/// The dynamic stages, in pipeline order.
pub const STAGES: [(&str, &str); 11] = [
    (
        "explore",
        "bet 1 exploration: A′ and B-i × 3 LR × 2 inits, seed 0 [GPU]",
    ),
    (
        "explore-b-ii",
        "B-ii at the best configuration of B-i [GPU]",
    ),
    ("final", "A′ and B* × 3 seeds, A × 1 seed [GPU]"),
    (
        "sens",
        "bet 1 sensitivity (amendment 2): A′ at the next lower LR × 3 seeds [GPU]",
    ),
    (
        "long",
        "the /live demo candidate (amendment 3, outside the bets): B* for 8000 steps [GPU]",
    ),
    (
        "eval",
        "exports, calibration and test evaluations of the finished runs (--runs a,b)",
    ),
    (
        "eval-q8",
        "every test set in q8_0 for the given runs (--runs a,b), after eval",
    ),
    (
        "eval-q4",
        "every test set in q4_0 for the given runs (--runs a,b), after eval",
    ),
    (
        "lambda",
        "bet 2: λ 0.1 and 0.3 at seed 0 on the support [GPU]",
    ),
    ("lambda-final", "bet 2: λ* × seeds 1 and 2 [GPU]"),
    ("report", "pre-registered analyses into runs/s5-report/"),
];

#[derive(Clone, Debug)]
pub struct Run {
    pub name: String,
    pub variant: String,
    /// `models/s5-base/` directory name.
    pub init: String,
    pub lr: String,
    pub seed: u64,
    pub steps: u64,
    pub coh: String,
}

impl Run {
    fn train(&self) -> Step {
        let mut args: Vec<String> = [
            "s5-train",
            "--init",
            &format!("models/s5-base/{}", self.init),
            "--records",
            "data/s5/mixed",
            "--out",
            &format!("runs/s5-{}", self.name),
            "--variant",
            &self.variant,
            "--steps",
            &self.steps.to_string(),
            "--lr",
            &self.lr,
            "--seed",
            &self.seed.to_string(),
            "--kl",
            KL,
            "--coh",
            &self.coh,
            "--eval-every",
            "1000",
            "--bf16",
            "--resume",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        // A is the Laya format as is: no random names, no polarity augmentation.
        if self.variant == "a" {
            args.extend(["--p-names", "0", "--p-polarity", "0"].map(String::from));
        }
        Step {
            name: format!("train-{}", self.name),
            prog: "sky-train",
            args,
            gpu: true,
            always: false,
            // MLX keeps freed buffers for reuse; with batches of varying shapes the cache grows
            // to ~12 GB per run, and three runs swap. The cap changes no number.
            env: vec![("SKY_MLX_CACHE_GB".into(), MLX_CACHE_GB.into())],
        }
    }
}

fn short(variant: &str) -> &str {
    match variant {
        "a-prime" => "ap",
        "b-i" => "bi",
        "b-ii" => "bii",
        v => v,
    }
}

fn init_short(init: &str) -> &str {
    INITS
        .iter()
        .find(|(_, d)| *d == init)
        .map_or(init, |(s, _)| s)
}

pub fn explore_runs() -> Vec<Run> {
    let mut v = Vec::new();
    for variant in ["a-prime", "b-i"] {
        for (s, init) in INITS {
            for lr in LRS {
                v.push(Run {
                    name: format!("x-{}-{s}-{lr}", short(variant)),
                    variant: variant.into(),
                    init: init.into(),
                    lr: lr.into(),
                    seed: 0,
                    steps: EXPLORE_STEPS,
                    coh: "0".into(),
                });
            }
        }
    }
    v
}

fn val(store: &Path, run: &str) -> Result<Value> {
    let p = store.join(format!("runs/s5-{run}/val.json"));
    let text = std::fs::read_to_string(&p)
        .with_context(|| format!("{} (is the run finished?)", p.display()))?;
    Ok(serde_json::from_str(&text)?)
}

fn val_nll(store: &Path, run: &str) -> Result<f64> {
    val(store, run)?["all"]["nll"]
        .as_f64()
        .with_context(|| format!("runs/s5-{run}/val.json has no all.nll"))
}

/// The frozen selection `name`, computed by `make` the first time.
fn frozen(store: &Path, name: &str, make: impl FnOnce() -> Result<Value>) -> Result<Value> {
    let p = store.join(format!("runs/s5-select/{name}.json"));
    if let Ok(text) = std::fs::read_to_string(&p) {
        return Ok(serde_json::from_str(&text)?);
    }
    let v = make()?;
    std::fs::create_dir_all(p.parent().unwrap())?;
    std::fs::write(&p, serde_json::to_string_pretty(&v)? + "\n")?;
    println!(
        "[xtask] selection frozen in {}:\n{}",
        p.display(),
        serde_json::to_string_pretty(&v)?
    );
    Ok(v)
}

/// The best exploration run of a variant (lowest val NLL).
fn best_of(store: &Path, variant: &str) -> Result<(Run, f64, Vec<Value>)> {
    let mut table = Vec::new();
    let mut best: Option<(Run, f64)> = None;
    for r in explore_runs().into_iter().filter(|r| r.variant == variant) {
        // A run that diverged (no val.json, non-finite NLL) is left out and recorded as failed.
        let Ok(nll) = val_nll(store, &r.name) else {
            table.push(json!({"run": r.name, "failed": true}));
            continue;
        };
        if !nll.is_finite() {
            table.push(json!({"run": r.name, "failed": true, "val_nll": null}));
            continue;
        }
        let acc = val(store, &r.name)?["all"]["accuracy"].as_f64();
        table.push(json!({"run": r.name, "val_nll": nll, "val_accuracy": acc}));
        if best.as_ref().is_none_or(|(_, b)| nll < *b) {
            best = Some((r, nll));
        }
    }
    let (r, nll) = best.context("no exploration run")?;
    Ok((r, nll, table))
}

fn explore_selection(store: &Path) -> Result<Value> {
    frozen(store, "explore", || {
        let (ap, ap_nll, ap_t) = best_of(store, "a-prime")?;
        let (bi, bi_nll, bi_t) = best_of(store, "b-i")?;
        Ok(json!({
            "rule": "lowest val NLL over every val record",
            "a-prime": {"init": ap.init, "lr": ap.lr, "run": ap.name, "val_nll": ap_nll, "runs": ap_t},
            "b-i": {"init": bi.init, "lr": bi.lr, "run": bi.name, "val_nll": bi_nll, "runs": bi_t},
        }))
    })
}

fn b_ii_run(sel: &Value) -> Run {
    let init = sel["b-i"]["init"].as_str().unwrap_or_default().to_string();
    let lr = sel["b-i"]["lr"].as_str().unwrap_or_default().to_string();
    Run {
        name: format!("x-bii-{}-{lr}", init_short(&init)),
        variant: "b-ii".into(),
        init,
        lr,
        seed: 0,
        steps: EXPLORE_STEPS,
        coh: "0".into(),
    }
}

fn final_selection(store: &Path) -> Result<Value> {
    let ex = explore_selection(store)?;
    frozen(store, "final", || {
        let bii = b_ii_run(&ex);
        let bi_nll = ex["b-i"]["val_nll"].as_f64().unwrap_or(f64::NAN);
        // A diverged B-ii run counts as infinitely worse (and is recorded as such).
        let bii_nll = val_nll(store, &bii.name)
            .ok()
            .filter(|x| x.is_finite())
            .unwrap_or(f64::INFINITY);
        let b = if bii_nll < bi_nll { "b-ii" } else { "b-i" };
        Ok(json!({
            "rule": "B-i or B-ii: lower val NLL at the best B-i configuration",
            "a-prime": ex["a-prime"].clone(),
            "b": {"variant": b, "init": ex["b-i"]["init"], "lr": ex["b-i"]["lr"],
                  "val_nll_b_i": bi_nll, "val_nll_b_ii": bii_nll},
        }))
    })
}

/// The step count of the finalists and of bet 2, fixed from the throughput of the exploration
/// only, never from a validation metric: with `t` the median training minutes per step of the
/// exploration runs (three at a time), the largest of [`FINAL_STEP_CHOICES`] such that
/// `WAVES · steps · t ≤ BUDGET_MINUTES`. Frozen in `runs/s5-select/steps.json`.
fn final_steps(store: &Path) -> Result<u64> {
    let v = frozen(store, "steps", || {
        let mut minutes: Vec<f64> = explore_runs()
            .iter()
            .filter_map(|r| val(store, &r.name).ok()?["minutes"].as_f64())
            .collect();
        if minutes.is_empty() {
            bail!("no finished exploration run");
        }
        minutes.sort_by(f64::total_cmp);
        let median = minutes[minutes.len() / 2];
        let per_step = median / EXPLORE_STEPS as f64;
        let steps = FINAL_STEP_CHOICES
            .iter()
            .copied()
            .filter(|&f| WAVES * f as f64 * per_step <= BUDGET_MINUTES)
            .max()
            .unwrap_or(FINAL_STEP_CHOICES[0]);
        Ok(json!({
            "rule": "largest of {2000, 3000, 4000, 6000} with 5 waves x steps x (median exploration minutes / 600) <= 180 min; throughput only",
            "explore_minutes": minutes, "median_minutes": median, "minutes_per_step": per_step,
            "budget_minutes": BUDGET_MINUTES, "waves": WAVES, "final_steps": steps,
        }))
    })?;
    v["final_steps"]
        .as_u64()
        .context("runs/s5-select/steps.json has no final_steps")
}

fn cfg_run(name: String, variant: &str, cfg: &Value, seed: u64, coh: &str, steps: u64) -> Run {
    Run {
        name,
        variant: variant.into(),
        init: cfg["init"].as_str().unwrap_or_default().into(),
        lr: cfg["lr"].as_str().unwrap_or_default().into(),
        seed,
        steps,
        coh: coh.into(),
    }
}

pub fn final_runs(sel: &Value, steps: u64) -> Vec<Run> {
    let b = sel["b"]["variant"].as_str().unwrap_or("b-i").to_string();
    let mut v = Vec::new();
    for s in SEEDS {
        v.push(cfg_run(
            format!("f-ap-s{s}"),
            "a-prime",
            &sel["a-prime"],
            s,
            "0",
            steps,
        ));
        v.push(cfg_run(format!("f-b-s{s}"), &b, &sel["b"], s, "0", steps));
    }
    v.push(cfg_run(
        "f-a-s0".into(),
        "a",
        &sel["a-prime"],
        0,
        "0",
        steps,
    ));
    v
}

/// Support of bet 2: `("b", variant)` when bet 1 succeeded, else `("ap", "a-prime")`.
fn support(store: &Path, sel: &Value) -> Result<(String, String, Value)> {
    let p = store.join("runs/s5-report/pari1.json");
    let text = std::fs::read_to_string(&p)
        .with_context(|| format!("{} (run the report stage first)", p.display()))?;
    let v: Value = serde_json::from_str(&text)?;
    Ok(if v["success"].as_bool() == Some(true) {
        let b = sel["b"]["variant"].as_str().unwrap_or("b-i").to_string();
        ("b".into(), b, sel["b"].clone())
    } else {
        ("ap".into(), "a-prime".into(), sel["a-prime"].clone())
    })
}

/// Bet 1 sensitivity (amendment 2, secondary): A′ at the init of its selection and the next
/// lower LR of the grid, three seeds, as many steps as the finalists. Decided on the training
/// curves of the A′ finalists (gradient spikes and a loss plateau at the selected LR), before any
/// test result of bet 1.
/// The long run of amendment 3: B* at its configuration, seed 0, λ = 0, [`LONG_STEPS`] steps.
/// Outside the bets; it serves /live when its val NLL is below that of the shipped model.
pub fn long_run(sel: &Value) -> Run {
    let b = sel["b"]["variant"].as_str().unwrap_or("b-i").to_string();
    cfg_run(LONG_RUN.into(), &b, &sel["b"], 0, "0", LONG_STEPS)
}

pub fn sens_runs(sel: &Value, steps: u64) -> Vec<Run> {
    let lr = sel["a-prime"]["lr"].as_str().unwrap_or("2e-4");
    let lower = LRS
        .iter()
        .position(|l| *l == lr)
        .and_then(|i| i.checked_sub(1))
        .map_or(LRS[0], |i| LRS[i]);
    SEEDS
        .iter()
        .map(|&s| {
            let mut r = cfg_run(
                format!("s-ap-{lower}-s{s}"),
                "a-prime",
                &sel["a-prime"],
                s,
                "0",
                steps,
            );
            r.lr = lower.to_string();
            r
        })
        .collect()
}

fn lambda_runs(store: &Path, sel: &Value) -> Result<Vec<Run>> {
    let (tag, variant, cfg) = support(store, sel)?;
    let steps = final_steps(store)?;
    Ok(LAMBDAS
        .iter()
        .map(|l| cfg_run(format!("l-{tag}-{l}-s0"), &variant, &cfg, 0, l, steps))
        .collect())
}

fn lambda_selection(store: &Path, sel: &Value) -> Result<Value> {
    let (tag, variant, cfg) = support(store, sel)?;
    frozen(store, "lambda", || {
        let base = val(store, &format!("f-{tag}-s0"))?;
        let floor = base["all"]["accuracy"].as_f64().unwrap_or(f64::NAN) - 0.01;
        let mut best: Option<(String, f64)> = None;
        let mut table = Vec::new();
        for l in LAMBDAS {
            let v = val(store, &format!("l-{tag}-{l}-s0"))?;
            let acc = v["all"]["accuracy"].as_f64().unwrap_or(f64::NAN);
            let mad = v["coherence"]["mad_neg"].as_f64().unwrap_or(f64::NAN);
            table.push(json!({"lambda": l, "val_accuracy": acc, "val_mad_neg": mad}));
            if acc >= floor && best.as_ref().is_none_or(|(_, m)| mad < *m) {
                best = Some((l.to_string(), mad));
            }
        }
        Ok(json!({
            "rule": "lowest val MAD_neg among λ ∈ {0.1, 0.3} with val accuracy ≥ that of λ = 0 (seed 0) − 0.01",
            "support": tag, "variant": variant, "init": cfg["init"], "lr": cfg["lr"],
            "lambda0_val_accuracy": floor + 0.01, "runs": table,
            "lambda_star": best.map(|(l, _)| l),
        }))
    })
}

fn lambda_final_runs(store: &Path, sel: &Value) -> Result<Vec<Run>> {
    let ls = lambda_selection(store, sel)?;
    let Some(l) = ls["lambda_star"].as_str() else {
        bail!("no λ kept the val accuracy of λ = 0: bet 2 stops here (pre-registered)");
    };
    let tag = ls["support"].as_str().unwrap_or("b");
    let variant = ls["variant"].as_str().unwrap_or("b-i");
    let steps = final_steps(store)?;
    Ok([1, 2]
        .iter()
        .map(|&s| cfg_run(format!("l-{tag}-{l}-s{s}"), variant, &ls, s, l, steps))
        .collect())
}

/// The runs of `final` and of bet 2 that have finished (a `val.json`).
fn finished(store: &Path) -> Vec<(String, String)> {
    let mut v = Vec::new();
    if let Ok(sel) = final_selection(store) {
        for r in final_runs(&sel, FINAL_STEP_CHOICES[0]) {
            v.push((r.name, r.variant));
        }
        if let Ok(ls) = lambda_selection(store, &sel) {
            let tag = ls["support"].as_str().unwrap_or("b").to_string();
            let variant = ls["variant"].as_str().unwrap_or("b-i").to_string();
            for l in LAMBDAS {
                v.push((format!("l-{tag}-{l}-s0"), variant.clone()));
            }
            if let Some(l) = ls["lambda_star"].as_str() {
                for s in [1, 2] {
                    v.push((format!("l-{tag}-{l}-s{s}"), variant.clone()));
                }
            }
        }
        // Amendments 2 and 3: the sensitivity runs and the long run of /live.
        for r in sens_runs(&sel, FINAL_STEP_CHOICES[0]) {
            v.push((r.name, r.variant));
        }
        let long = long_run(&sel);
        v.push((long.name, long.variant));
    }
    v.retain(|(r, _)| store.join(format!("runs/s5-{r}/val.json")).exists());
    v
}

/// The English evaluation sets of s5-lite (`data/s5/MANIFEST.json`), without SIB-200 fr and
/// without yahootopics, whose license is not confirmed (DATA_LICENSES.md).
fn eval_sets(store: &Path) -> Result<Vec<String>> {
    let text = std::fs::read_to_string(store.join("data/s5/MANIFEST.json"))?;
    let m: Value = serde_json::from_str(&text)?;
    let mut v: Vec<String> = m["eval"]
        .as_object()
        .context("MANIFEST.json has no eval")?
        .keys()
        .filter_map(|k| k.strip_suffix(".jsonl"))
        .filter(|k| *k != "sib200_fr" && *k != "yahootopics")
        .map(str::to_string)
        .collect();
    v.sort();
    Ok(v)
}

fn variant_of(store: &Path, run: &str) -> Result<String> {
    let p = store.join(format!("runs/s5-{run}/last/s5.json"));
    let v: Value = serde_json::from_str(
        &std::fs::read_to_string(&p).with_context(|| p.display().to_string())?,
    )?;
    v["args"]["variant"]
        .as_str()
        .map(str::to_string)
        .context("s5.json has no args.variant")
}

/// `sky-eval` of one frozen test set, guarded by the pre-registration.
fn test_step(run: &str, f: &str, set: &str) -> Step {
    let (rd, md) = (format!("runs/s5-{run}"), format!("models/s5-{run}"));
    Step::new(
        &format!("{run}-test-{f}-{set}"),
        "sky-eval",
        &[
            &format!("{md}/{f}.gguf"),
            "--records",
            &format!("data/s5/eval/{set}.jsonl"),
            "--split",
            "test",
            "--prereg",
            "{checkout}/docs/s5-prereg.md",
            "--metrics",
            "--json",
            &format!("{rd}/eval-{f}/{set}.json"),
            "--dump-probs",
            &format!("{rd}/eval-{f}/{set}.dump.jsonl"),
        ],
    )
}

/// The quantized formats of `eval`: the fitted bench only, the rest with `eval-q8` / `eval-q4`.
const QUANT_FORMATS: [&str; 2] = ["q8_0", "q4_0"];
/// Every format of `eval`.
const FORMATS: [&str; 3] = ["f32", "q8_0", "q4_0"];

/// The test sets in format `f` (q8_0 or q4_0) that `eval` leaves out (all but the fitted bench).
fn eval_quant_steps(store: &Path, run: &str, f: &str) -> Result<Vec<Step>> {
    Ok(eval_sets(store)?
        .iter()
        .filter(|s| *s != "typed_decisions_test")
        .map(|s| test_step(run, f, s))
        .collect())
}

/// [`eval_quant_steps`] in q8_0 (the shipped model).
fn eval_q8_steps(store: &Path, run: &str) -> Result<Vec<Step>> {
    eval_quant_steps(store, run, "q8_0")
}

/// [`eval_quant_steps`] in q4_0.
fn eval_q4_steps(store: &Path, run: &str) -> Result<Vec<Step>> {
    eval_quant_steps(store, run, "q4_0")
}

fn eval_steps(store: &Path, run: &str) -> Result<Vec<Step>> {
    let variant = variant_of(store, run)?;
    let (rd, md) = (format!("runs/s5-{run}"), format!("models/s5-{run}"));
    let mut v = vec![Step::new(
        &format!("{run}-mkdir"),
        "mkdir",
        &[
            "-p",
            &md,
            &format!("{rd}/eval-f32"),
            &format!("{rd}/eval-q8_0"),
            &format!("{rd}/eval-q4_0"),
        ],
    )];
    if variant.starts_with("b-") {
        for f in FORMATS {
            v.push(Step::new(
                &format!("{run}-export-{f}"),
                "sky-train",
                &[
                    "s5-export",
                    "--run",
                    &format!("{rd}/last"),
                    "--out",
                    &format!("{md}/raw.{f}.gguf"),
                    "--format",
                    f,
                ],
            ));
        }
    } else {
        v.push(Step::new(
            &format!("{run}-export-laya"),
            "sky-train",
            &[
                "export",
                "--run",
                &format!("{rd}/last"),
                "--out",
                &format!("{rd}/laya"),
            ],
        ));
        for f in FORMATS {
            v.push(Step::new(
                &format!("{run}-convert-{f}"),
                "sky-convert",
                &[
                    &format!("{rd}/laya"),
                    &format!("{md}/raw.{f}.gguf"),
                    "--type",
                    f,
                    "--head-type",
                    "f32",
                ],
            ));
        }
    }
    let sets = eval_sets(store)?;
    for f in FORMATS {
        // Temperatures on the English val split (already spent on model selection), so that the
        // cal split stays independent for the escalation thresholds of the calibrated model.
        v.push(Step::new(
            &format!("{run}-val-{f}"),
            "sky-eval",
            &[
                &format!("{md}/raw.{f}.gguf"),
                "--records",
                "data/s5/train",
                "--split",
                "val",
                "--lang",
                "en",
                "--json",
                &format!("{rd}/val.{f}.json"),
                "--dump-probs",
                &format!("{rd}/val.{f}.dump.jsonl"),
            ],
        ));
        v.push(Step::new(
            &format!("{run}-calibrate-{f}"),
            "sky-s5-report",
            &[
                "calibrate",
                &format!("{rd}/val.{f}.dump.jsonl"),
                "--gguf",
                &format!("{md}/raw.{f}.gguf"),
                "--out",
                &format!("{md}/{f}.gguf"),
                "--json",
                &format!("{rd}/temperatures.{f}.json"),
            ],
        ));
        v.push(Step::new(
            &format!("{run}-cal-{f}"),
            "sky-eval",
            &[
                &format!("{md}/{f}.gguf"),
                "--records",
                "data/s5/train",
                "--split",
                "cal",
                "--lang",
                "en",
                "--json",
                &format!("{rd}/cal.{f}.json"),
                "--dump-probs",
                &format!("{rd}/cal.{f}.dump.jsonl"),
            ],
        ));
        for s in &sets {
            // q8_0, q4_0: the fitted bench only (Phase 5); `eval-q8` / `eval-q4` add the rest.
            if f == "f32" || s == "typed_decisions_test" {
                v.push(test_step(run, f, s));
            }
        }
        if f == "f32" {
            // The original 2609.33209 battery (comparison with Jev), in the store only.
            v.push(Step::new(
                &format!("{run}-battery-{f}"),
                "sky-eval",
                &[
                    &format!("{md}/{f}.gguf"),
                    "--records",
                    BATTERY,
                    "--split",
                    "test",
                    "--prereg",
                    "{checkout}/docs/s5-prereg.md",
                    "--metrics",
                    "--json",
                    &format!("{rd}/eval-{f}/battery_2609_33209.json"),
                    "--dump-probs",
                    &format!("{rd}/eval-{f}/battery_2609_33209.dump.jsonl"),
                ],
            ));
        }
        v.push(Step::new(
            &format!("{run}-demo-{f}"),
            "sky-eval",
            &[
                &format!("{md}/{f}.gguf"),
                "--records",
                "{checkout}/eval/s5/demo-cases.jsonl",
                "--split",
                "test",
                "--metrics",
                "--json",
                &format!("{rd}/eval-{f}/demo-cases.json"),
                "--dump-probs",
                &format!("{rd}/eval-{f}/demo-cases.dump.jsonl"),
            ],
        ));
    }
    Ok(v)
}

fn report_steps(store: &Path) -> Result<Vec<Step>> {
    let mut v = vec![Step::new("mkdir", "mkdir", &["-p", "runs/s5-report"])];
    let runs = finished(store);
    let has = |r: &str, f: &str, set: &str| {
        store
            .join(format!("runs/s5-{r}/eval-{f}/{set}.dump.jsonl"))
            .exists()
    };
    // Every test set (the last one `eval` writes) or only the fitted bench.
    let full = |r: &str, f: &str| has(r, f, "wikitoxic_toxicaggregated") && has(r, f, "demo-cases");
    let bench = |r: &str, f: &str| has(r, f, "typed_decisions_test");
    for (r, _) in &runs {
        for f in FORMATS {
            if !full(r, f) {
                continue;
            }
            let dir = format!("runs/s5-{r}/eval-{f}");
            v.push(Step::new(
                &format!("gates-{r}-{f}"),
                "sky-s5-report",
                &[
                    "gates",
                    &dir,
                    "--records",
                    "data/s5/eval",
                    "--json",
                    &format!("runs/s5-report/gates-{r}-{f}.json"),
                ],
            ));
            v.push(Step::new(
                &format!("robust-{r}-{f}"),
                "sky-s5-report",
                &[
                    "robust",
                    &dir,
                    "--json",
                    &format!("runs/s5-report/robust-{r}-{f}.json"),
                ],
            ));
            v.push(Step::new(
                &format!("coherence-{r}-{f}"),
                "sky-s5-report",
                &[
                    "coherence",
                    &dir,
                    "--json",
                    &format!("runs/s5-report/coherence-{r}-{f}.json"),
                ],
            ));
            if has(r, f, "battery_2609_33209") {
                v.push(Step::new(
                    &format!("battery-{r}-{f}"),
                    "sky-s5-report",
                    &[
                        "coherence",
                        &dir,
                        "--set",
                        "battery_2609_33209",
                        "--json",
                        &format!("runs/s5-report/battery-{r}-{f}.json"),
                    ],
                ));
            }
            v.push(Step::new(
                &format!("selective-{r}-{f}"),
                "sky-s5-report",
                &[
                    "selective",
                    &format!("runs/s5-{r}/cal.{f}.dump.jsonl"),
                    &dir,
                    "--cal-src",
                    "typed_decisions_train",
                    "--json",
                    &format!("runs/s5-report/selective-{r}-{f}.json"),
                ],
            ));
            // Secondary: every English cal record (more data, no formal guarantee under shift).
            v.push(Step::new(
                &format!("selective-allcal-{r}-{f}"),
                "sky-s5-report",
                &[
                    "selective",
                    &format!("runs/s5-{r}/cal.{f}.dump.jsonl"),
                    &dir,
                    "--json",
                    &format!("runs/s5-report/selective-allcal-{r}-{f}.json"),
                ],
            ));
        }
    }
    let dirs = |prefix: &str, f: &str, ready: &dyn Fn(&str, &str) -> bool| -> Option<String> {
        let d: Vec<String> = SEEDS.iter().map(|s| format!("{prefix}-s{s}")).collect();
        d.iter().all(|r| ready(r, f)).then(|| {
            d.iter()
                .map(|r| format!("runs/s5-{r}/eval-{f}"))
                .collect::<Vec<_>>()
                .join(",")
        })
    };
    // Amendment 2: B* against A′ at the next lower LR (secondary, same margins).
    if let Ok(sel) = final_selection(store) {
        if let Some(r) = sens_runs(&sel, FINAL_STEP_CHOICES[0]).first() {
            let prefix = r.name.trim_end_matches("-s0").to_string();
            if let (Some(a), Some(b)) = (dirs(&prefix, "f32", &full), dirs("f-b", "f32", &full)) {
                v.push(Step::new(
                    "pari1-sensitivity",
                    "sky-s5-report",
                    &[
                        "pari1",
                        "--ref",
                        &a,
                        "--variant",
                        &b,
                        "--json",
                        "runs/s5-report/pari1-sensitivity.json",
                    ],
                ));
            }
        }
    }
    if let (Some(a), Some(b)) = (dirs("f-ap", "f32", &full), dirs("f-b", "f32", &full)) {
        v.push(Step::new(
            "pari1",
            "sky-s5-report",
            &[
                "pari1",
                "--ref",
                &a,
                "--variant",
                &b,
                "--json",
                "runs/s5-report/pari1.json",
            ],
        ));
    }
    for arm in ["f-ap", "f-b"] {
        for f in QUANT_FORMATS {
            let (Some(a), Some(q)) = (dirs(arm, "f32", &bench), dirs(arm, f, &bench)) else {
                continue;
            };
            // Phase 5: q8_0 (recalibrated) against f32 on the same items; bench_ece is the
            // criterion. q4_0 is reported the same way, as a secondary result.
            let name = match f {
                "q8_0" => format!("phase5-{arm}"),
                _ => format!("phase5-q4-{arm}"),
            };
            v.push(Step::new(
                &name,
                "sky-s5-report",
                &[
                    "pari1",
                    "--ref",
                    &a,
                    "--variant",
                    &q,
                    "--json",
                    &format!("runs/s5-report/{name}.json"),
                ],
            ));
        }
        if let Some(a) = dirs(arm, "f32", &full) {
            v.push(Step::new(
                &format!("coherence-{arm}"),
                "sky-s5-report",
                &[
                    "coherence",
                    &a,
                    "--json",
                    &format!("runs/s5-report/coherence-{arm}.json"),
                ],
            ));
        }
    }
    if let Ok(sel) = final_selection(store) {
        if let Ok(ls) = lambda_selection(store, &sel) {
            let tag = ls["support"].as_str().unwrap_or("b");
            if let (Some(l), Some(r)) = (
                ls["lambda_star"].as_str(),
                dirs(&format!("f-{tag}"), "f32", &full),
            ) {
                let lam: Vec<String> = SEEDS.iter().map(|s| format!("l-{tag}-{l}-s{s}")).collect();
                if lam.iter().all(|x| full(x, "f32")) {
                    let ld = lam
                        .iter()
                        .map(|x| format!("runs/s5-{x}/eval-f32"))
                        .collect::<Vec<_>>()
                        .join(",");
                    v.push(Step::new(
                        "pari2",
                        "sky-s5-report",
                        &[
                            "pari2",
                            "--ref",
                            &r,
                            "--variant",
                            &ld,
                            "--json",
                            "runs/s5-report/pari2.json",
                        ],
                    ));
                }
            }
        }
    }
    // The results page reads one file, committed with the site.
    v.push(Step::new(
        "site-mkdir",
        "mkdir",
        &["-p", "{checkout}/site/public/results"],
    ));
    v.push(
        Step::new(
            "summary",
            "sky-s5-report",
            &[
                "summary",
                "--store",
                "{store}",
                "--out",
                "{checkout}/site/public/results/s5.json",
            ],
        )
        .always(),
    );
    Ok(v)
}

/// A dynamic stage, built against the store.
pub fn stage(name: &str, store: &Path, runs: Option<&str>) -> Result<Option<Stage>> {
    let about = match STAGES.iter().find(|(n, _)| *n == name) {
        Some((_, a)) => *a,
        None => return Ok(None),
    };
    let steps: Vec<Step> = match name {
        "explore" => explore_runs().iter().map(Run::train).collect(),
        "explore-b-ii" => vec![b_ii_run(&explore_selection(store)?).train()],
        "final" => final_runs(&final_selection(store)?, final_steps(store)?)
            .iter()
            .map(Run::train)
            .collect(),
        "long" => vec![long_run(&final_selection(store)?).train()],
        "sens" => sens_runs(&final_selection(store)?, final_steps(store)?)
            .iter()
            .map(Run::train)
            .collect(),
        "lambda" => lambda_runs(store, &final_selection(store)?)?
            .iter()
            .map(Run::train)
            .collect(),
        "lambda-final" => lambda_final_runs(store, &final_selection(store)?)?
            .iter()
            .map(Run::train)
            .collect(),
        "eval" => {
            let list: Vec<String> = match runs {
                Some(r) => r
                    .split(',')
                    .filter(|s| !s.is_empty())
                    .map(str::to_string)
                    .collect(),
                None => finished(store).into_iter().map(|(r, _)| r).collect(),
            };
            let mut v = Vec::new();
            for r in &list {
                v.extend(eval_steps(store, r)?);
            }
            v
        }
        "eval-q8" | "eval-q4" => {
            let Some(r) = runs else {
                bail!("{name} needs --runs a,b");
            };
            let mut v = Vec::new();
            for r in r.split(',').filter(|s| !s.is_empty()) {
                v.extend(if name == "eval-q8" {
                    eval_q8_steps(store, r)?
                } else {
                    eval_q4_steps(store, r)?
                });
            }
            v
        }
        "report" => report_steps(store)?,
        _ => unreachable!(),
    };
    if steps.is_empty() {
        bail!("stage {name}: nothing to run yet");
    }
    let name: &'static str = STAGES
        .iter()
        .find(|(n, _)| *n == name)
        .map(|(n, _)| *n)
        .unwrap();
    Ok(Some(Stage { name, about, steps }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exploration_grid() {
        let v = explore_runs();
        assert_eq!(v.len(), 12);
        let mut names: Vec<&str> = v.iter().map(|r| r.name.as_str()).collect();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), 12);
        let t = v[0].train();
        assert!(t.gpu && t.args.contains(&"--resume".to_string()));
        let a = Run {
            variant: "a".into(),
            ..v[0].clone()
        }
        .train();
        assert!(a
            .args
            .windows(2)
            .any(|w| w[0] == "--p-names" && w[1] == "0"));
    }

    #[test]
    fn sensitivity_runs_take_the_next_lower_lr() {
        let sel = json!({"a-prime": {"init": "ettin-reranker-17m", "lr": "2e-4"}});
        let v = sens_runs(&sel, 2000);
        assert_eq!(v.len(), 3);
        assert!(v
            .iter()
            .all(|r| r.lr == "1e-4" && r.variant == "a-prime" && r.steps == 2000));
        assert_eq!(v[0].name, "s-ap-1e-4-s0");
        let low = json!({"a-prime": {"init": "ettin-17m", "lr": "5e-5"}});
        assert!(sens_runs(&low, 2000).iter().all(|r| r.lr == "5e-5"));
    }

    #[test]
    fn final_runs_follow_the_selection() {
        let sel = json!({"a-prime": {"init": "ettin-17m", "lr": "1e-4"},
                         "b": {"variant": "b-ii", "init": "ettin-reranker-17m", "lr": "2e-4"}});
        let v = final_runs(&sel, 3000);
        assert_eq!(v.len(), 7);
        assert!(v.iter().filter(|r| r.variant == "b-ii").count() == 3);
        assert!(v
            .iter()
            .any(|r| r.name == "f-a-s0" && r.variant == "a" && r.lr == "1e-4"));
        assert!(v.iter().all(|r| r.steps == 3000 && r.coh == "0"));
    }
}
