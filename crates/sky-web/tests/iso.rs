//! The iso model (`skycmd-iso`) on the served target. Runs natively (`cargo test -p sky-web`)
//! and on `wasm32-unknown-unknown` (`wasm-pack test --node crates/sky-web --test iso`):
//!
//! - **parity**: the raw scores of the golden models (`sky-train s5-golden`) match the mlx
//!   reference within 1e-4;
//! - **guaranteed properties** (≤ 1e-6, any weights): permuting the options permutes the scores,
//!   renaming them changes nothing, swapping the two `noul` definitions gives s + n = 1, and a
//!   cached state answers exactly like a freshly encoded one;
//! - **Q4_0**: the golden model with its matrices in Q4_0 keeps those properties, and its scores
//!   match an f32 model holding the same (dequantized) weights.

use serde_json::Value;
use sky_gguf::{GgmlType, Gguf, GgufWriter, OwnedValue};
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

/// The model with every encoder matrix (and the token embeddings) re-encoded as `ty`, through
/// f32; `iso.*` and the vectors stay as they are.
fn requant(bytes: &[u8], ty: GgmlType) -> Vec<u8> {
    let g = Gguf::parse(bytes).unwrap();
    let mut w = GgufWriter::new();
    for (k, v) in &g.kvs {
        w.add_kv(*k, OwnedValue::from_value(v));
    }
    for t in &g.tensors {
        let matrix = t.dims.len() == 2 && !t.name.starts_with("iso.") && t.dims[0] % 32 == 0;
        let to = if matrix { ty } else { t.dtype };
        let values = g.tensor_f32(t.name).unwrap();
        w.add_tensor_f32(t.name, &t.dims, to, &values).unwrap();
    }
    w.to_bytes().unwrap()
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
    let q4 = requant(SHARED, GgmlType::Q4_0);
    for bytes in [SHARED, RESET, &q4] {
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
    let q4 = requant(SHARED, GgmlType::Q4_0);
    for bytes in [SHARED, RESET, &q4] {
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

#[test]
fn q4_0_matches_f32_on_the_same_weights() {
    let q4 = requant(SHARED, GgmlType::Q4_0);
    // The Q4_0 values written back as f32: the same weights through the f32 kernels.
    let f32_of_q4 = requant(&q4, GgmlType::F32);
    let mut mq = Model::load(&q4).unwrap();
    let mut mf = Model::load(&f32_of_q4).unwrap();
    assert_eq!(mq.info()["weight_format"], "q4_0");
    assert_eq!(mf.info()["weight_format"], "f32");
    assert!(
        q4.len() < SHARED.len() / 3,
        "{} vs {}",
        q4.len(),
        SHARED.len()
    );
    let g: Value = serde_json::from_str(GOLDEN).unwrap();
    let mut worst = 0.0f32;
    for (i, c) in g["cases"].as_array().unwrap().iter().enumerate() {
        let args = (
            c["t"].as_str().unwrap(),
            c["ins"].as_str().unwrap(),
            strings(&c["opts"]),
            c["state"].as_str().unwrap(),
        );
        let a = scores(&mut mq, args.0, args.1, &args.2, args.3);
        let b = scores(&mut mf, args.0, args.1, &args.2, args.3);
        // Only the int8 rounding of the activations differs (Q8x2: ~15 bits).
        assert_close(&a, &b, 2e-3, &format!("case {i}"));
        worst = a
            .iter()
            .zip(&b)
            .fold(worst, |m, (x, y)| m.max((x - y).abs()));
    }
    assert!(worst > 0.0, "the Q4_0 products did not run");
}
