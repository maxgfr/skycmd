//! `sky-data route blind --check-placeholder` end to end, through the binary (s6-route v6 4.3 and
//! v7): the exit code of the two modes on invented records. The kept mode (`--keep
//! --placeholder <json>`: the stage `placeholder` once the diagnostic took it, and the counts
//! amendment on T5) must exit 0 when the placeholder is no request and 2 when it is one; the
//! choosing mode exits 2 only when every placeholder occurs.

use std::path::{Path, PathBuf};
use std::process::Command;

fn dir(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("sky-route-blind-cli-{tag}-{}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn records(p: &Path, requests: &[&str]) {
    let lines: Vec<String> = requests
        .iter()
        .map(|r| {
            serde_json::json!({"state": format!("User: {r}"), "target": [1.0, 0.0]}).to_string()
        })
        .collect();
    std::fs::write(p, lines.join("\n") + "\n").unwrap();
}

fn check(files: &Path, out: &Path, keep: Option<&Path>) -> (i32, serde_json::Value) {
    let mut c = Command::new(env!("CARGO_BIN_EXE_sky-data"));
    c.args(["route", "blind", "--check-placeholder"])
        .arg(files)
        .arg("--out")
        .arg(out);
    if let Some(k) = keep {
        c.arg("--keep").arg("--placeholder").arg(k);
    }
    let o = c.output().unwrap();
    let r = serde_json::from_slice(&std::fs::read(out).unwrap()).unwrap();
    (o.status.code().unwrap(), r)
}

#[test]
fn the_kept_placeholder_passes_unless_it_is_a_request() {
    let d = dir("keep");
    let kept = d.join("placeholder.json");
    std::fs::write(&kept, r#"{"placeholder": "?", "taken": "?"}"#).unwrap();
    let f = d.join("t5_tools.jsonl");
    let out = d.join("check.json");
    records(&f, &["book a flight to Oslo", "what is the weather"]);
    let (code, r) = check(&f, &out, Some(&kept));
    assert_eq!(r["occurs"], false);
    assert_eq!(code, 0, "occurs false must exit 0");
    records(&f, &["book a flight to Oslo", " ? "]);
    let (code, r) = check(&f, &out, Some(&kept));
    assert_eq!(r["occurs"], true);
    assert_eq!(code, 2, "occurs true must exit 2");
    // --keep without the file kept is refused, never a silent choice
    let o = Command::new(env!("CARGO_BIN_EXE_sky-data"))
        .args(["route", "blind", "--check-placeholder"])
        .arg(&f)
        .arg("--out")
        .arg(d.join("none.json"))
        .arg("--keep")
        .output()
        .unwrap();
    assert_ne!(o.status.code(), Some(0));
    assert!(!d.join("none.json").exists());
    std::fs::remove_dir_all(d).ok();
}

#[test]
fn the_choosing_mode_stops_only_when_every_placeholder_occurs() {
    let d = dir("take");
    let f = d.join("train.jsonl");
    let out = d.join("placeholder.json");
    records(&f, &["?", "turn on the lights"]);
    let (code, r) = check(&f, &out, None);
    assert_eq!((code, r["taken"].as_str()), (0, Some("\u{2026}")));
    records(&f, &["?", "\u{2026}", "-"]);
    let (code, r) = check(&f, &out, None);
    assert!(r["taken"].is_null());
    assert_eq!(code, 2);
    std::fs::remove_dir_all(d).ok();
}
