//! Golden test of the temporal carry (SPEC-S3 §7) against the mlx-rs training graph:
//! `tests/golden/s3t.json` holds a tiny random-init s3 model with `temporal_carry` (f32 and
//! ternary files), stateless cases and `sequences` of frames answered in order on one stream.
//! sky-infer must reproduce the token streams, which frames start warm, the early-exit answers
//! and their probabilities within 1e-4 (f32) / 1e-3 (ternary).

use std::path::PathBuf;

use serde_json::Value;
use sky_infer::prompt::render_options;
use sky_infer::{DecisionRequest, Engine, EngineOptions, QType, TernaryCompute};

fn dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/golden")
}

fn f32s(v: &Value) -> Vec<f32> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap() as f32)
        .collect()
}

fn u32s(v: &Value) -> Vec<u32> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_u64().unwrap() as u32)
        .collect()
}

fn bools(v: &Value) -> Vec<bool> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_u64().unwrap() != 0)
        .collect()
}

fn close(a: &[f32], b: &[f32], tol: f32) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (x - y).abs() <= tol)
}

/// The question type, instructions and rendered options of a question in request form.
fn question(state: &str, q: &Value) -> (QType, String, Vec<String>) {
    let req = serde_json::json!({ "state": state, "questions": { "q": q } });
    let r = DecisionRequest::from_value(&req).unwrap();
    let q = &r.questions[0];
    (
        q.kind.qtype(),
        q.instructions.clone(),
        render_options(&q.kind),
    )
}

#[test]
fn s3t_golden() {
    let Ok(bytes) = std::fs::read(dir().join("s3t.json")) else {
        eprintln!("skipped: tests/golden/s3t.json not found");
        return;
    };
    let g: Value = serde_json::from_slice(&bytes).expect("valid s3t.json");
    for file in g["meta"]["files"].as_array().unwrap() {
        let format = file["format"].as_str().unwrap();
        let model = std::fs::read(dir().join(file["path"].as_str().unwrap())).unwrap();
        assert_eq!(model.len() as u64, file["bytes"].as_u64().unwrap());
        let tol = g["meta"]["tolerance"][format].as_f64().unwrap() as f32;
        let computes: &[TernaryCompute] = if format == "ternary" {
            &[
                TernaryCompute::Packed,
                TernaryCompute::F32,
                TernaryCompute::Int8,
                TernaryCompute::Lut,
            ]
        } else {
            &[TernaryCompute::Auto]
        };
        for &ternary in computes {
            let opts = EngineOptions {
                ternary,
                ..Default::default()
            };
            let mut max_err = 0.0f32;
            // stateless cases: every exit, cold
            let mut e = Engine::from_gguf_with(&model, opts).unwrap();
            assert!(e.s2_model().unwrap().cfg.temporal_carry);
            for case in g["cases"].as_array().unwrap() {
                let state = case["state"].as_str().unwrap();
                let (qtype, ins, texts) = question(state, &case["question"]);
                let got = e.probs_per_exit(qtype, &ins, &texts, state).unwrap();
                let want = case["expected"][format]["exits"].as_array().unwrap();
                assert_eq!(got.len(), want.len());
                for (a, w) in got.iter().zip(want) {
                    let w = f32s(&w["probs"]);
                    for (x, y) in a.iter().zip(&w) {
                        max_err = max_err.max((x - y).abs());
                    }
                    assert!(close(a, &w, tol), "{format} {ternary:?} {}", case["name"]);
                }
            }
            // streams: frames in order, warm when the state token ids repeat
            for (sid, seq) in g["sequences"].as_array().unwrap().iter().enumerate() {
                let name = seq["name"].as_str().unwrap();
                let mut e = Engine::from_gguf_with(&model, opts).unwrap();
                let stream = sid as u64 + 1;
                for (t, frame) in seq["frames"].as_array().unwrap().iter().enumerate() {
                    let state = frame["state"].as_str().unwrap();
                    let (qtype, ins, texts) = question(state, &seq["question"]);
                    let (ids, values, hv) = e.s2_engine_mut().unwrap().state_stream(state);
                    assert_eq!(ids, u32s(&frame["state_stream"]["ids"]), "{name} {t}");
                    assert_eq!(values, f32s(&frame["state_stream"]["values"]));
                    assert_eq!(hv, bools(&frame["state_stream"]["has_value"]));
                    let p = e
                        .decide_raw_stream(stream, qtype, &ins, &texts, state)
                        .unwrap();
                    let warm = frame["warm"].as_bool().unwrap();
                    assert_eq!(
                        e.s2_engine_mut().unwrap().stream_carried(stream),
                        warm,
                        "{name} frame {t}: warm"
                    );
                    let exp = &frame["expected"][format];
                    let stats = e.last_stats().unwrap();
                    assert_eq!(
                        stats.exit,
                        Some(exp["exit"].as_u64().unwrap() as usize),
                        "{format} {ternary:?} {name} frame {t}: exit"
                    );
                    assert_eq!(
                        stats.layers_used,
                        Some(exp["carry_source_layers"].as_u64().unwrap() as usize)
                    );
                    let want = f32s(&exp["probs"]);
                    for (x, y) in p.iter().zip(&want) {
                        max_err = max_err.max((x - y).abs());
                    }
                    assert!(
                        close(&p, &want, tol),
                        "{format} {ternary:?} {name} frame {t}: {p:?} vs {want:?}"
                    );
                }
            }
            eprintln!("{format} ({ternary:?} compute): max |p - golden| = {max_err:.2e}");
        }
    }
}
