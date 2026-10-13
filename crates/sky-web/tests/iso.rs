//! The iso model (`skycmd-iso`) on the served target. Runs natively (`cargo test -p sky-web`)
//! and on `wasm32-unknown-unknown` (`wasm-pack test --node crates/sky-web --test iso`):
//!
//! - **parity**: the raw scores of the golden models (`sky-train s5-golden`) match the mlx
//!   reference within 1e-4;
//! - **guaranteed properties** (≤ 1e-6, any weights): permuting the options permutes the scores,
//!   renaming them changes nothing, swapping the two `noul` definitions gives s + n = 1, and a
//!   cached state answers exactly like a freshly encoded one.

use serde_json::Value;
use sky_infer::QType;
use sky_web::Model;

#[cfg(target_arch = "wasm32")]
use wasm_bindgen_test::wasm_bindgen_test as test;

const SHARED: &[u8] = include_bytes!("../../sky-infer/tests/golden/iso-shared.gguf");
const RESET: &[u8] = include_bytes!("../../sky-infer/tests/golden/iso-reset.gguf");
const GOLDEN: &str = include_str!("../../sky-infer/tests/golden/iso.json");
const EXACT: f32 = 1e-6;

fn scores(m: &mut Model, t: &str, ins: &str, opts: &[String], state: &str) -> Vec<f32> {
    m.engine_mut()
        .scores_raw(QType::from_name(t).unwrap(), ins, opts, state)
        .expect("scores")
}

fn strings(v: &Value) -> Vec<String> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|s| s.as_str().unwrap().to_string())
        .collect()
}

fn assert_close(a: &[f32], b: &[f32], tol: f32, what: &str) {
    assert_eq!(a.len(), b.len(), "{what}");
    for (x, y) in a.iter().zip(b) {
        assert!((x - y).abs() <= tol, "{what}: {a:?} vs {b:?}");
    }
}

#[test]
fn matches_the_mlx_reference() {
    let g: Value = serde_json::from_str(GOLDEN).unwrap();
    let tol = g["meta"]["tolerance"].as_f64().unwrap() as f32;
    for (name, bytes) in [("shared", SHARED), ("reset", RESET)] {
        let mut m = Model::load(bytes).unwrap();
        assert_eq!(m.info()["arch"], "skycmd-iso");
        for (i, c) in g["cases"].as_array().unwrap().iter().enumerate() {
            let got = scores(
                &mut m,
                c["t"].as_str().unwrap(),
                c["ins"].as_str().unwrap(),
                &strings(&c["opts"]),
                c["state"].as_str().unwrap(),
            );
            let want: Vec<f32> = g["logits"][name][i]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_f64().unwrap() as f32)
                .collect();
            assert_close(&got, &want, tol, &format!("{name} case {i}"));
        }
    }
}

#[test]
fn permutations_and_names() {
    for bytes in [SHARED, RESET] {
        let mut m = Model::load(bytes).unwrap();
        let opts = |v: &[&str]| v.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        let state = "temperature 39.5, heart rate 120, the patient is confused and pale";
        let base = scores(
            &mut m,
            "choice",
            "What should the nurse do?",
            &opts(&[
                "a: call the doctor now",
                "b: wait an hour",
                "c: give water and observe",
            ]),
            state,
        );
        // The same definitions in another order, under other names.
        let perm = scores(
            &mut m,
            "choice",
            "What should the nurse do?",
            &opts(&[
                "xg6a6: give water and observe",
                "yes: call the doctor now",
                "no: wait an hour",
            ]),
            state,
        );
        assert_close(
            &perm,
            &[base[2], base[0], base[1]],
            EXACT,
            "permutation + names",
        );
    }
}

#[test]
fn noul_swap_sums_to_one() {
    for bytes in [SHARED, RESET] {
        let mut m = Model::load(bytes).unwrap();
        let q = "The alert is real.";
        let state = "27 failed logins then a success from an unknown address";
        let p = |m: &mut Model, f: &str, t: &str| {
            let o = vec![format!("false: {f}"), format!("true: {t}")];
            m.probs(QType::Noul, q, &o, state).unwrap()
        };
        let s = p(&mut m, "benign activity", "malicious activity");
        let n = p(&mut m, "malicious activity", "benign activity");
        assert!((s[1] + n[1] - 1.0).abs() <= EXACT, "{s:?} {n:?}");
    }
}

#[test]
fn cached_state_answers_like_a_fresh_one() {
    let opts = |v: &[&str]| v.iter().map(|s| s.to_string()).collect::<Vec<_>>();
    let state = "battery 61, wind east 3, target north-west";
    let mut warm = Model::load(SHARED).unwrap();
    // First question encodes the state, the second reuses it.
    scores(
        &mut warm,
        "score",
        "How risky is the flight?",
        &opts(&["level 0: safe", "level 1: risky", "level 2: unsafe"]),
        state,
    );
    let hit = scores(
        &mut warm,
        "choice",
        "Where to go?",
        &opts(&["north", "west", "home"]),
        state,
    );
    assert!(warm.last_stats().unwrap().state_hit);
    let mut cold = Model::load(SHARED).unwrap();
    let fresh = scores(
        &mut cold,
        "choice",
        "Where to go?",
        &opts(&["north", "west", "home"]),
        state,
    );
    assert!(!cold.last_stats().unwrap().state_hit);
    assert_close(&hit, &fresh, 0.0, "cache");
}
