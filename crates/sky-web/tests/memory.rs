//! The fast-weight memory on the served target (docs/s6-sleep2-prereg.md §7, docs/s6-sleep-prereg.md
//! §8 item 3): the memory sequences of `crates/sky-infer/tests/golden/s6m.json` (the mlx-rs
//! training graph: wake frames, sleeps of N − 1 passes, a pause, the memory off) played by
//! sky-infer natively (`cargo test -p sky-web --test memory`) and on `wasm32-unknown-unknown`
//! (`wasm-pack test --node crates/sky-web --test memory`). Both must match the golden answers
//! within 1e-4 (f32) / 1e-3 (ternary).

use serde_json::Value;
use sky_infer::prompt::render_options;
use sky_infer::s2::memory::SleepOptions;
use sky_infer::{DecisionRequest, Engine, EngineOptions};

#[cfg(target_arch = "wasm32")]
use wasm_bindgen_test::wasm_bindgen_test as test;

const GOLDEN: &str = include_str!("../../sky-infer/tests/golden/s6m.json");
const F32: &[u8] = include_bytes!("../../sky-infer/tests/golden/s6m-f32.gguf");
const TERNARY: &[u8] = include_bytes!("../../sky-infer/tests/golden/s6m-ternary.gguf");

fn f32s(v: &Value) -> Vec<f32> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap() as f32)
        .collect()
}

fn check(format: &str, model: &[u8]) {
    let g: Value = serde_json::from_str(GOLDEN).expect("s6m.json");
    let tol = g["meta"]["tolerance"][format].as_f64().unwrap() as f32;
    let seqs = g["memory_sequences"].as_array().unwrap();
    assert!(!seqs.is_empty());
    for (sid, seq) in seqs.iter().enumerate() {
        let name = seq["name"].as_str().unwrap();
        let mut e = Engine::from_gguf_with(model, EngineOptions::default()).expect("model loads");
        assert!(e.has_memory());
        e.set_sleep(SleepOptions {
            passes: seq["passes"].as_u64().unwrap() as usize,
            memory_off: seq["memory_off"].as_bool().unwrap(),
            ..Default::default()
        });
        let stream = sid as u64 + 1;
        for (t, frame) in seq["frames"].as_array().unwrap().iter().enumerate() {
            let state = frame["state"].as_str().unwrap();
            let req = serde_json::json!({ "state": state, "questions": { "q": seq["question"] } });
            let r = DecisionRequest::from_value(&req).unwrap();
            let q = &r.questions[0];
            let texts = render_options(&q.kind);
            e.new_frame(stream, frame["pause"].as_bool().unwrap());
            let p = e
                .decide_raw_stream(stream, q.kind.qtype(), &q.instructions, &texts, state)
                .unwrap();
            let exp = &frame["expected"][format];
            assert_eq!(
                e.last_stats().unwrap().exit,
                Some(exp["exit"].as_u64().unwrap() as usize),
                "{format} {name} frame {t}: exit"
            );
            let want = f32s(&exp["probs"]);
            assert!(
                p.len() == want.len() && p.iter().zip(&want).all(|(a, b)| (a - b).abs() <= tol),
                "{format} {name} frame {t}: {p:?} vs {want:?}"
            );
        }
    }
}

#[test]
fn memory_f32() {
    check("f32", F32);
}

#[test]
fn memory_ternary() {
    check("ternary", TERNARY);
}
