//! Parity of the served target. The same test runs natively (`cargo test -p sky-web`) and on
//! `wasm32-unknown-unknown` (`wasm-pack test --node crates/sky-web`, SIMD128 on): both must
//! reproduce `tests/parity/expected.json`, the probabilities of the site's models on
//! `tests/parity/cases.json`, within 1e-5. The expected file is written by the native run with
//! `SKY_PARITY_UPDATE=1` (only when a change is meant to move the numbers).

use serde_json::Value;
use sky_web::Engine;

#[cfg(target_arch = "wasm32")]
use wasm_bindgen_test::wasm_bindgen_test as test;

const CASES: &str = include_str!("parity/cases.json");
const EXPECTED: &str = include_str!("parity/expected.json");
const TOL: f64 = 1e-5;

/// Probabilities of every case under a model.
fn probs(model: &[u8]) -> Vec<Vec<f64>> {
    let mut e = Engine::load(model).expect("model loads");
    let cases: Vec<Value> = serde_json::from_str(CASES).expect("cases.json");
    cases
        .iter()
        .map(|c| {
            let opts = serde_json::to_string(&c["opts"]).unwrap();
            let v = e
                .decide_value(
                    c["t"].as_str().unwrap(),
                    c["ins"].as_str().unwrap(),
                    &opts,
                    c["state"].as_str().unwrap(),
                )
                .expect("decision");
            v["probs"]
                .as_array()
                .unwrap()
                .iter()
                .map(|p| p.as_f64().unwrap())
                .collect()
        })
        .collect()
}

fn check(name: &str, model: &[u8]) {
    let got = probs(model);
    #[cfg(not(target_arch = "wasm32"))]
    if std::env::var("SKY_PARITY_UPDATE").is_ok_and(|v| v == "1") {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/parity/expected.json");
        let mut all: serde_json::Map<String, Value> =
            serde_json::from_str(&std::fs::read_to_string(path).unwrap_or_else(|_| "{}".into()))
                .unwrap_or_default();
        all.insert(name.into(), serde_json::to_value(&got).unwrap());
        std::fs::write(path, serde_json::to_string_pretty(&all).unwrap() + "\n").unwrap();
        return;
    }
    let expected: Value = serde_json::from_str(EXPECTED).expect("expected.json");
    let want: Vec<Vec<f64>> = serde_json::from_value(expected[name].clone())
        .unwrap_or_else(|_| panic!("expected.json has no entry for {name}"));
    assert_eq!(got.len(), want.len(), "{name}: number of cases");
    for (i, (g, w)) in got.iter().zip(&want).enumerate() {
        assert_eq!(g.len(), w.len(), "{name} case {i}");
        for (a, b) in g.iter().zip(w) {
            assert!(
                (a - b).abs() <= TOL,
                "{name} case {i}: {g:?} vs expected {w:?}"
            );
        }
    }
}

#[test]
fn s1_pico() {
    check(
        "s1-pico",
        include_bytes!("../../../site/public/models/s1-pico.gguf"),
    );
}

#[test]
fn s2_nano() {
    check(
        "s2-nano",
        include_bytes!("../../../site/public/models/s2-nano.gguf"),
    );
}

#[test]
fn s3_pico() {
    check(
        "s3-pico",
        include_bytes!("../../../site/public/models/s3-pico.gguf"),
    );
}

#[test]
fn s4_micro() {
    check(
        "s4-micro",
        include_bytes!("../../../site/public/models/s4-micro.gguf"),
    );
}
