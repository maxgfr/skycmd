//! Golden test of the s2 engine against the mlx-rs training graph: `sky-train s2-golden` writes
//! `tests/golden/s2.json` (tiny random-init s2 model, f32 and ternary GGUF files, requests,
//! token streams and expected probabilities per exit). sky-infer must reproduce the token
//! streams exactly and the probabilities within 1e-4 (f32) / 1e-3 (ternary).

use std::path::PathBuf;

use serde_json::Value;
use sky_infer::prompt::render_options;
use sky_infer::{DecisionRequest, Engine, EngineOptions, QType, TernaryCompute};

fn dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/golden")
}

fn golden() -> Option<Value> {
    let bytes = std::fs::read(dir().join("s2.json")).ok()?;
    Some(serde_json::from_slice(&bytes).expect("valid s2.json"))
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

fn close(a: &[f32], b: &[f32], tol: f32) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (x - y).abs() <= tol)
}

#[test]
fn s2_golden() {
    let Some(g) = golden() else {
        panic!("tests/golden/s2.json not found (run `sky-train s2-golden`)");
    };
    let tol = &g["meta"]["tolerance"];
    let files = g["meta"]["files"].as_array().unwrap();
    assert!(!files.is_empty());
    for file in files {
        let format = file["format"].as_str().unwrap();
        let bytes = std::fs::read(dir().join(file["path"].as_str().unwrap())).unwrap();
        assert_eq!(bytes.len() as u64, file["bytes"].as_u64().unwrap());
        let tolerance = tol[format].as_f64().unwrap() as f32;
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
            let mut max_err = 0.0f32;
            let opts = EngineOptions {
                ternary,
                ..Default::default()
            };
            let mut e = Engine::from_gguf_with(&bytes, opts).unwrap();
            let info = e.model_info();
            assert_eq!(info.arch, "skycmd-s2");
            assert_eq!(info.weight_format, format);
            let cfg = &g["meta"]["config"];
            assert_eq!(
                info.hidden_size as u64,
                cfg["hidden_size"].as_u64().unwrap()
            );
            assert_eq!(
                info.exits,
                u32s(&cfg["exits"])
                    .iter()
                    .map(|&v| v as usize)
                    .collect::<Vec<_>>()
            );
            for case in g["cases"].as_array().unwrap() {
                let name = case["name"].as_str().unwrap();
                let state = case["state"].as_str().unwrap();
                let req =
                    serde_json::json!({ "state": state, "questions": { "q": case["question"] } });
                let r = DecisionRequest::from_value(&req).unwrap();
                let q = &r.questions[0];
                let qtype = q.kind.qtype();
                assert_eq!(
                    qtype,
                    QType::from_name(case["qtype"].as_str().unwrap()).unwrap()
                );
                let texts = render_options(&q.kind);
                let want_texts: Vec<&str> = case["option_texts"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_str().unwrap())
                    .collect();
                assert_eq!(texts, want_texts, "{name}: option texts");
                // token streams
                let refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
                let s2 = e.s2_engine_mut().unwrap();
                let st = s2.static_tokens(qtype, &q.instructions, &refs);
                assert_eq!(st.ids, u32s(&case["static"]["ids"]), "{name}: static ids");
                assert_eq!(
                    st.q_len as u64,
                    case["static"]["q_len"].as_u64().unwrap(),
                    "{name}: q_len"
                );
                let markers: Vec<u32> = st.options.iter().map(|o| o.0 as u32).collect();
                assert_eq!(markers, u32s(&case["static"]["markers"]), "{name}: markers");
                assert_eq!(st.values, f32s(&case["static"]["values"]), "{name}: values");
                let (ids, values) = s2.state_tokens(state);
                assert_eq!(ids, u32s(&case["state_stream"]["ids"]), "{name}: state ids");
                assert_eq!(
                    values,
                    f32s(&case["state_stream"]["values"]),
                    "{name}: state values"
                );
                // probabilities at every exit
                let exp = &case["expected"][format];
                let per_exit = e
                    .probs_per_exit(qtype, &q.instructions, &texts, state)
                    .unwrap();
                let want_exits = exp["exits"].as_array().unwrap();
                assert_eq!(per_exit.len(), want_exits.len());
                for (got, want) in per_exit.iter().zip(want_exits) {
                    let want = f32s(&want["probs"]);
                    for (a, b) in got.iter().zip(&want) {
                        max_err = max_err.max((a - b).abs());
                    }
                    assert!(
                        close(got, &want, tolerance),
                        "{format} {ternary:?} {name} exit {}: {got:?} vs {want:?}",
                        want_exits.len()
                    );
                }
                // the early-exit answer
                let p = e.decide_raw(qtype, &q.instructions, &texts, state).unwrap();
                let stats = e.last_stats().unwrap();
                assert_eq!(
                    stats.exit,
                    Some(exp["exit"].as_u64().unwrap() as usize),
                    "{format} {name}: exit"
                );
                assert_eq!(
                    stats.layers_used,
                    Some(exp["layers_used"].as_u64().unwrap() as usize),
                    "{format} {name}: layers used"
                );
                assert!(
                    close(&p, &f32s(&exp["probs"]), tolerance),
                    "{format} {name}"
                );
            }
            eprintln!("{format} ({ternary:?} compute): max |p - golden| = {max_err:.2e}");
        }
    }
}
