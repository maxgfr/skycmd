//! `cargo xtask regress [--update]`: the numbers of the models served on the site must not move.
//!
//! Every model of `site/public/models/` is evaluated with `sky-eval` on the typed-decisions
//! bench and played with `sky-arena` (10 episodes per game, seed 1). Timings are dropped; every
//! other value must equal `eval/regress/baseline.json` exactly. `--update` rewrites the
//! baseline (only when a change is meant to move the numbers, and said so in the PR).
//! The records are read from the store (`$SKYCMD_STORE/data/decisions`), nothing is written there.

use std::path::Path;
use std::process::Command;

use anyhow::{bail, Context, Result};
use serde_json::{Map, Value};

/// The models of the site, by file stem.
pub const MODELS: [&str; 5] = ["s1-pico", "s2-nano", "s2-nano-doom", "s3-pico", "s4-micro"];

/// Drops the timing fields (`*_us`, `*_ms`, `secs`), which vary from run to run.
pub fn strip_timings(v: &mut Value) {
    match v {
        Value::Object(m) => {
            m.retain(|k, _| {
                !(k.ends_with("_us") || k.ends_with("_ms") || k == "secs" || k == "model")
            });
            m.values_mut().for_each(strip_timings);
        }
        Value::Array(a) => a.iter_mut().for_each(strip_timings),
        _ => {}
    }
}

/// Paths (`a.b.c`) whose values differ between two JSON documents.
pub fn diff(a: &Value, b: &Value, path: &str, out: &mut Vec<String>) {
    match (a, b) {
        (Value::Object(x), Value::Object(y)) => {
            let mut keys: Vec<&String> = x.keys().chain(y.keys()).collect();
            keys.sort();
            keys.dedup();
            for k in keys {
                let p = format!("{path}.{k}");
                match (x.get(k), y.get(k)) {
                    (Some(u), Some(v)) => diff(u, v, &p, out),
                    _ => out.push(p),
                }
            }
        }
        (Value::Array(x), Value::Array(y)) if x.len() == y.len() => {
            for (i, (u, v)) in x.iter().zip(y).enumerate() {
                diff(u, v, &format!("{path}[{i}]"), out);
            }
        }
        _ if a != b => out.push(format!("{path}: {a} != {b}")),
        _ => {}
    }
}

fn run_json(cmd: &mut Command, out: &Path) -> Result<Value> {
    let st = cmd
        .stdout(std::process::Stdio::null())
        .status()
        .with_context(|| format!("{cmd:?}"))?;
    if !st.success() {
        bail!("{cmd:?} exited with {st}");
    }
    let mut v: Value = serde_json::from_str(&std::fs::read_to_string(out)?)?;
    strip_timings(&mut v);
    Ok(v)
}

pub fn regress(root: &Path, store: &Path, update: bool) -> Result<()> {
    let st = Command::new(env!("CARGO"))
        .current_dir(root)
        .args(["build", "--release", "-p", "sky-infer", "-p", "sky-arena"])
        .status()?;
    if !st.success() {
        bail!("cargo build failed");
    }
    let bin = root.join("target/release");
    let records = store.join("data/decisions/typed_decisions_bench.jsonl");
    if !records.exists() {
        bail!("{} is missing", records.display());
    }
    let tmp = root.join("target/regress");
    std::fs::create_dir_all(&tmp)?;
    let mut now = Map::new();
    for m in MODELS {
        let gguf = root.join(format!("site/public/models/{m}.gguf"));
        let e = tmp.join(format!("{m}.eval.json"));
        let a = tmp.join(format!("{m}.arena.json"));
        let eval = run_json(
            Command::new(bin.join("sky-eval"))
                .arg(&gguf)
                .arg("--records")
                .arg(&records)
                .arg("--json")
                .arg(&e),
            &e,
        )?;
        let arena = run_json(
            Command::new(bin.join("sky-arena")).arg(&gguf).args([
                "--episodes",
                "10",
                "--seed",
                "1",
                "--game",
                "all",
                "--json",
                a.to_str().context("path")?,
            ]),
            &a,
        )?;
        println!("[regress] {m}: evaluated");
        now.insert(
            m.to_string(),
            serde_json::json!({"sky_eval": eval, "sky_arena": arena}),
        );
    }
    let now = Value::Object(now);
    let base_path = root.join("eval/regress/baseline.json");
    if update {
        std::fs::create_dir_all(base_path.parent().context("parent")?)?;
        std::fs::write(&base_path, serde_json::to_string_pretty(&now)? + "\n")?;
        println!("[regress] wrote {}", base_path.display());
        return Ok(());
    }
    let base: Value = serde_json::from_str(
        &std::fs::read_to_string(&base_path)
            .with_context(|| format!("{} (run with --update once)", base_path.display()))?,
    )?;
    let mut d = Vec::new();
    diff(&base, &now, "", &mut d);
    if !d.is_empty() {
        for l in &d {
            eprintln!("[regress] changed {l}");
        }
        bail!("{} values differ from {}", d.len(), base_path.display());
    }
    println!(
        "[regress] identical to {} ({} models)",
        base_path.display(),
        MODELS.len()
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn timings_are_dropped_and_diffs_found() {
        let mut a = json!({"model": "x", "all": {"n": 3, "median_us": 9.5, "acc": 0.5},
                           "flappy": {"p50_us": 1.0, "exits": [1, 2]}});
        strip_timings(&mut a);
        assert_eq!(
            a,
            json!({"all": {"n": 3, "acc": 0.5}, "flappy": {"exits": [1, 2]}})
        );
        let b = json!({"all": {"n": 3, "acc": 0.25}, "flappy": {"exits": [1, 3]}});
        let mut d = Vec::new();
        diff(&a, &b, "", &mut d);
        assert_eq!(d.len(), 2);
        let mut d = Vec::new();
        diff(&a, &a, "", &mut d);
        assert!(d.is_empty());
    }
}
