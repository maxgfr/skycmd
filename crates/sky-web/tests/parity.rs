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

/// s6-ear: log-mel front-end and audio decisions of a random-weight model on fixed clips (no
/// shipped model yet), native vs wasm32. The first value of each clip is the sum of its
/// log-mel features.
#[test]
fn s6_ear_random() {
    use sky_infer::s2::random::{ladder_config, random_gguf, test_tokenizer, WeightFormat};
    let mut cfg = ladder_config("s6-pico").unwrap();
    cfg.audio_frame = sky_infer::audio::FRAME_DIM;
    cfg.max_state_tokens = 128;
    // temperature 1: a random model's sharp logits would turn float rounding (FMA natively, none
    // in WASM SIMD) into visible probability gaps
    let bytes = random_gguf(&cfg, &test_tokenizer(), 7, WeightFormat::Ternary).unwrap();
    let mut e = Engine::load(&bytes).expect("model loads");
    // a tone and its third harmonic by rotation recurrences (no libm: the input is the same
    // sample for sample on every target), with a slow amplitude ramp
    let clip = |secs: f32, f0: f32| -> Vec<f32> {
        let n = (16_000.0 * secs) as usize;
        let rot = |f: f64| {
            // cos / sin of 2πf/16000 from their series, basic operations only
            let a = std::f64::consts::TAU * f / 16_000.0;
            let (mut c, mut s, mut t) = (1.0f64, 0.0f64, 1.0f64);
            for k in 1..20 {
                t *= a / k as f64;
                match k % 4 {
                    1 => s += t,
                    2 => c -= t,
                    3 => s -= t,
                    _ => c += t,
                }
            }
            (c, s)
        };
        let (r1, r3) = (rot(f0 as f64), rot(3.0 * f0 as f64));
        let (mut p1, mut p3) = ((1.0f64, 0.0f64), (1.0f64, 0.0f64));
        (0..n)
            .map(|i| {
                let amp = 0.2 + 0.3 * (i as f64 / n as f64);
                let y = amp * p1.1 + 0.1 * p3.1;
                p1 = (p1.0 * r1.0 - p1.1 * r1.1, p1.0 * r1.1 + p1.1 * r1.0);
                p3 = (p3.0 * r3.0 - p3.1 * r3.1, p3.0 * r3.1 + p3.1 * r3.0);
                y as f32
            })
            .collect()
    };
    let questions: [(&str, &str, &str); 2] = [
        (
            "noul",
            "Has the speaker finished their turn?",
            r#"["false","true"]"#,
        ),
        (
            "choice",
            "Which word was said?",
            r#"["yes","no","up","down","left","right","none"]"#,
        ),
    ];
    let mut got: Vec<Vec<f64>> = Vec::new();
    let lm = sky_infer::audio::LogMel::new();
    for (secs, f0) in [(0.5f32, 200.0f32), (2.0, 600.0), (12.0, 330.0)] {
        let pcm = clip(secs, f0);
        let (fr, _) = lm.frames(&pcm);
        // the front-end uses no libm call per frame: its features agree bit for bit
        let checksum = fr
            .iter()
            .enumerate()
            .map(|(i, &v)| v as f64 * (1.0 + (i % 7) as f64))
            .sum::<f64>();
        got.push(vec![checksum, fr.len() as f64]);
        for (t, ins, opts) in questions {
            let v = e.decide_audio_value(t, ins, opts, &pcm).expect("decision");
            got.push(
                v["probs"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|p| p.as_f64().unwrap())
                    .collect(),
            );
        }
    }
    compare("s6-ear-random", got);
}

/// `got` against (or, with `SKY_PARITY_UPDATE=1` natively, into) `expected.json[name]`.
fn compare(name: &str, got: Vec<Vec<f64>>) {
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
    assert_eq!(got.len(), want.len(), "{name}: rows");
    for (i, (g, w)) in got.iter().zip(&want).enumerate() {
        assert_eq!(g.len(), w.len(), "{name} row {i}");
        // probabilities within TOL; a checksum row (a value above 1) to 1e-12 relative
        let tol = |b: f64| if b.abs() > 1.0 { 1e-12 * b.abs() } else { TOL };
        for (a, b) in g.iter().zip(w) {
            assert!(
                (a - b).abs() <= tol(*b),
                "{name} row {i}: {g:?} vs expected {w:?}"
            );
        }
    }
}
