//! Step-based schedule (`--steps`, `--stop-after`) and `--init-params`.
//!
//! The schedule tests are pure. `s2_decide_stops_after_k_steps_from_given_params` runs the
//! real binary for a few steps of the s2-pico model on synthetic records.

use std::{
    path::{Path, PathBuf},
    process::Command,
};

use mlx_rs::Array;
use sky_train::{
    optim::wsd,
    params::{self, ParamMap},
    session::{keep_going, qat_due, replace_params, stop_step, Progress},
};

/// Steps a run takes from `start` until `keep_going` says stop.
fn run_steps(progress: &Progress, stop_at: Option<u64>, start: u64) -> u64 {
    let mut step = start;
    while keep_going(progress, stop_at, step, 0.0) {
        step += 1;
    }
    step
}

#[test]
fn minutes_and_steps_are_exclusive() {
    let err = Progress::from_flags(Some(10.0), Some(100)).unwrap_err();
    assert!(err.to_string().contains("--minutes and --steps"), "{err}");
    let err = Progress::from_flags(None, None).unwrap_err();
    assert!(err.to_string().contains("--minutes or --steps"), "{err}");
    assert!(Progress::from_flags(None, Some(0)).is_err());
    assert_eq!(
        Progress::from_flags(None, Some(100)).unwrap(),
        Progress::Steps { steps: 100 }
    );
}

#[test]
fn minutes_mode_is_unchanged() {
    let p = Progress::from_flags(Some(2.0), None).unwrap();
    assert_eq!(p, Progress::Time { budget_secs: 120.0 });
    assert_eq!(p.steps(), None);
    // frac = elapsed / budget whatever the step, and the run ends when the time is spent.
    assert_eq!(p.frac(5, 30.0), 0.25);
    assert_eq!(p.frac(5_000, 30.0), 0.25);
    assert!(keep_going(&p, None, 1_000_000, 119.9));
    assert!(!keep_going(&p, None, 0, 120.0));
    assert_eq!(wsd(p.frac(0, 90.0)), 1.0);
    // A zero budget runs no step (the former `elapsed < budget` loop).
    let zero = Progress::Time { budget_secs: 0.0 };
    assert!(!keep_going(&zero, None, 0, 0.0));
}

#[test]
fn step_mode_follows_the_step() {
    let p = Progress::from_flags(None, Some(1000)).unwrap();
    assert_eq!(p.steps(), Some(1000));
    assert_eq!(p.frac(0, 1e9), 0.0);
    assert_eq!(p.frac(250, 0.0), 0.25);
    assert_eq!(p.frac(1000, 0.0), 1.0);
    assert_eq!(run_steps(&p, None, 0), 1000);
    // WSD: flat, then decays over the last 20%.
    assert_eq!(wsd(p.frac(500, 0.0)), wsd(0.5));
    assert!(wsd(p.frac(900, 0.0)) < 1.0);
}

#[test]
fn qat_switch_step() {
    let p = Progress::Steps { steps: 1000 };
    // --qat-frac 0.3: QAT over the last 300 steps.
    let first = (0..1000).find(|&s| qat_due(p.frac(s, 0.0), 0.3)).unwrap();
    assert_eq!(first, 700);
    // --qat-frac 0.9 (the s4 recipe): after a 10% float warm-up.
    let first = (0..1000).find(|&s| qat_due(p.frac(s, 0.0), 0.9)).unwrap();
    assert_eq!(first, 100);
    assert!(!(0..1000).any(|s| qat_due(p.frac(s, 0.0), 0.0)));
    // Same switch on the time schedule as before.
    let t = Progress::Time { budget_secs: 100.0 };
    assert!(!qat_due(t.frac(0, 69.9), 0.3));
    assert!(qat_due(t.frac(0, 70.0), 0.3));
}

#[test]
fn stop_after_counts_from_the_starting_step_across_resumes() {
    let p = Progress::Steps { steps: 100 };
    // DiLoCo rounds of H = 30 inner steps: 0 -> 30 -> 60 -> 90 -> 100 (the budget ends first).
    let mut step = 0;
    let mut ends = Vec::new();
    while !p.done(step, 0.0) {
        let stop_at = stop_step(step, Some(30)).unwrap();
        step = run_steps(&p, stop_at, step);
        ends.push(step);
    }
    assert_eq!(ends, vec![30, 60, 90, 100]);
    // The schedule position stays global: the second round starts at frac 0.3.
    assert_eq!(p.frac(30, 0.0), 0.3);
    assert_eq!(stop_step(42, None).unwrap(), None);
    assert!(stop_step(0, Some(0)).is_err());
    assert!(stop_step(u64::MAX, Some(1)).is_err());
    // --stop-after also bounds a time-budgeted run.
    let t = Progress::Time { budget_secs: 60.0 };
    assert_eq!(run_steps(&t, stop_step(7, Some(3)).unwrap(), 7), 10);
}

fn map(entries: &[(&str, &[f32], &[i32])]) -> ParamMap {
    entries
        .iter()
        .map(|(k, v, s)| ((*k).into(), Array::from_slice(v, s)))
        .collect()
}

#[test]
fn init_params_must_match_the_model() {
    mlx_rs::with_device(mlx_rs::Device::cpu(), || {
        let model = || map(&[("a", &[0.0, 0.0], &[2]), ("b", &[0.0; 4], &[2, 2])]);
        let mut dst = model();
        let src = map(&[
            ("a", &[1.0, 2.0], &[2]),
            ("b", &[3.0, 4.0, 5.0, 6.0], &[2, 2]),
        ]);
        replace_params(&mut dst, &src).unwrap();
        assert_eq!(dst["b"].as_slice::<f32>(), &[3.0, 4.0, 5.0, 6.0]);

        let shape = map(&[("a", &[1.0, 2.0], &[2]), ("b", &[3.0; 4], &[4])]);
        let err = replace_params(&mut model(), &shape)
            .unwrap_err()
            .to_string();
        assert!(err.contains("tensor b"), "{err}");
        let missing = map(&[("b", &[3.0; 4], &[2, 2])]);
        let err = replace_params(&mut model(), &missing)
            .unwrap_err()
            .to_string();
        assert!(err.contains("no tensor a"), "{err}");
        let extra = map(&[
            ("a", &[1.0, 2.0], &[2]),
            ("b", &[3.0; 4], &[2, 2]),
            ("c", &[0.0], &[1]),
        ]);
        let err = replace_params(&mut model(), &extra)
            .unwrap_err()
            .to_string();
        assert!(err.contains("tensor c"), "{err}");
        // A refused file leaves the model untouched.
        let mut m = model();
        assert!(replace_params(&mut m, &shape).is_err());
        assert_eq!(m["a"].as_slice::<f32>(), &[0.0, 0.0]);
    });
}

// ----------------------------------------------------------------------------- real run

fn repo() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// Synthetic decision records: the right option is the one labelled "safe".
fn write_records(path: &Path, n: usize) {
    let mut lines = Vec::new();
    for i in 0..n {
        let k = 2 + i % 3;
        let ans = (i * 7) % k;
        let opts: Vec<String> = (0..k)
            .map(|j| {
                let label = if j == ans { "safe" } else { "blocked" };
                format!("{label} lane {}", j * 3 + 1)
            })
            .collect();
        let target: Vec<f32> = (0..k).map(|j| if j == ans { 1.0 } else { 0.0 }).collect();
        let split = if i % 10 == 0 { "val" } else { "train" };
        lines.push(
            serde_json::json!({
                "src": "syn", "t": "choice", "ins": "which lane is free?", "opts": opts,
                "state": format!("free {} wind {}", ans * 3 + 1, i % 9), "target": target,
                "split": split,
            })
            .to_string(),
        );
    }
    std::fs::write(path, lines.join("\n")).unwrap();
}

fn decide(records: &Path, out: &Path, extra: &[&str]) -> std::process::Output {
    let root = repo();
    Command::new(env!("CARGO_BIN_EXE_sky-train"))
        .arg("s2-decide")
        .arg("--config")
        .arg(root.join("configs/s2-pico.json"))
        .arg("--tokenizer")
        .arg(root.join("crates/sky-infer/tests/fixtures/small_tokenizer_s2.json"))
        .arg("--records")
        .arg(records)
        .arg("--out")
        .arg(out)
        .args(["--batch", "8", "--val-max", "8", "--log-every", "1"])
        .args(extra)
        .output()
        .unwrap()
}

fn ok(o: &std::process::Output) {
    assert!(
        o.status.success(),
        "sky-train failed:\n{}\n{}",
        String::from_utf8_lossy(&o.stdout),
        String::from_utf8_lossy(&o.stderr)
    );
}

fn state(run: &Path) -> serde_json::Value {
    serde_json::from_str(&std::fs::read_to_string(run.join("last/state.json")).unwrap()).unwrap()
}

fn assert_same(a: &ParamMap, b: &ParamMap) {
    assert_eq!(a.len(), b.len());
    for (k, v) in a {
        assert_eq!(
            v.as_slice::<f32>(),
            b[k].as_slice::<f32>(),
            "{k} differs from the given parameters"
        );
    }
}

#[test]
fn s2_decide_stops_after_k_steps_from_given_params() {
    let tmp = tempfile::tempdir().unwrap();
    let records = tmp.path().join("syn.jsonl");
    write_records(&records, 80);
    let a = tmp.path().join("a");
    let b = tmp.path().join("b");

    // Both budgets at once are refused.
    let o = decide(&records, &a, &["--minutes", "1", "--steps", "10"]);
    assert!(!o.status.success());
    assert!(String::from_utf8_lossy(&o.stderr).contains("--minutes and --steps"));

    // Round 1: 3 of 10 planned steps, then a checkpoint at step 3.
    ok(&decide(
        &records,
        &a,
        &["--steps", "10", "--stop-after", "3", "--qat-frac", "0"],
    ));
    let s = state(&a);
    assert_eq!(s["step"], 3);
    assert_eq!(s["steps"], 10);
    assert_eq!(s["frac"], 0.3);
    assert_eq!(s["optimizer_step"], 3);
    let trained = params::load(&a.join("last/model.safetensors")).unwrap();

    // A fresh run from these parameters at zero learning rate keeps them exactly.
    let init = a.join("last/model.safetensors").display().to_string();
    ok(&decide(
        &records,
        &b,
        &[
            "--steps",
            "10",
            "--stop-after",
            "1",
            "--seed",
            "99",
            "--qat-frac",
            "0",
            "--adam-lr",
            "0",
            "--muon-lr",
            "0",
            "--init-params",
            &init,
        ],
    ));
    assert_eq!(state(&b)["step"], 1);
    assert_same(
        &params::load(&b.join("last/model.safetensors")).unwrap(),
        &trained,
    );

    // Round 2 resumes the local optimizer and RNG of `b` (saved with a zero learning rate)
    // but takes the weights from the file: here the trained weights halved.
    let halved: ParamMap = trained
        .iter()
        .map(|(k, v)| (k.clone(), v.multiply(Array::from_f32(0.5)).unwrap()))
        .collect();
    let halved_path = tmp.path().join("halved.safetensors");
    params::save(&halved_path, &halved, None).unwrap();
    ok(&decide(
        &records,
        &b,
        &[
            "--steps",
            "10",
            "--stop-after",
            "2",
            "--qat-frac",
            "0",
            "--resume",
            "--init-params",
            halved_path.to_str().unwrap(),
        ],
    ));
    let s = state(&b);
    assert_eq!(s["step"], 3);
    assert_eq!(s["optimizer_step"], 3);
    assert_eq!(s["frac"], 0.3);
    assert_same(
        &params::load(&b.join("last/model.safetensors")).unwrap(),
        &params::load(&halved_path).unwrap(),
    );

    // A file that does not fit the model names the first offending tensor.
    let mut bad = trained.clone();
    let (name, _) = trained.iter().min_by(|x, y| x.0.cmp(y.0)).unwrap();
    bad.insert(name.clone(), Array::from_slice(&[0.0f32], &[1]));
    let bad_path = tmp.path().join("bad.safetensors");
    params::save(&bad_path, &bad, None).unwrap();
    let o = decide(
        &records,
        &tmp.path().join("c"),
        &[
            "--steps",
            "10",
            "--stop-after",
            "1",
            "--init-params",
            bad_path.to_str().unwrap(),
        ],
    );
    assert!(!o.status.success());
    let err = String::from_utf8_lossy(&o.stderr);
    assert!(err.contains(&format!("tensor {name}")), "{err}");
}
