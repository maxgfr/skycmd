//! Golden test of the fast-weight memory (docs/s6-sleep-prereg.md §2, docs/s6-sleep2-prereg.md
//! §4, docs/s6-sleep3-prereg.md §4) against the mlx-rs training graph: `tests/golden/s6m.json`
//! holds a tiny random-init model of `configs/s6-sleep3-pico.json` (shared query/key projection,
//! values of form `frame`; trained with an auxiliary head that the files do not carry), its
//! stateless cases, and `memory_sequences`: streams of frames played as sky-train `s2-probe`
//! plays them (wake frames, sleeps of N − 1 passes every 8 frames and at a pause, the memory off).
//! sky-infer must reproduce the token streams, the early-exit answers, their probabilities within
//! 1e-4 (f32) / 1e-3 (ternary), and the size of the memory after every frame.

use std::path::PathBuf;

use serde_json::Value;
use sky_infer::prompt::render_options;
use sky_infer::s2::memory::SleepOptions;
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

fn close(a: &[f32], b: &[f32], tol: f32) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (x - y).abs() <= tol)
}

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
fn s6m_golden() {
    let bytes = std::fs::read(dir().join("s6m.json")).expect("tests/golden/s6m.json");
    let g: Value = serde_json::from_slice(&bytes).expect("valid s6m.json");
    assert_eq!(g["meta"]["config"]["memory"], true);
    let seqs = g["memory_sequences"].as_array().unwrap();
    assert!(!seqs.is_empty());
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
            // Stateless cases: a zero memory, so the base's computation.
            let mut e = Engine::from_gguf_with(&model, opts).unwrap();
            assert!(e.has_memory());
            assert_eq!(e.s2_model().unwrap().memory.len(), 1);
            for case in g["cases"].as_array().unwrap() {
                let state = case["state"].as_str().unwrap();
                let (qtype, ins, texts) = question(state, &case["question"]);
                let got = e.probs_per_exit(qtype, &ins, &texts, state).unwrap();
                let want = case["expected"][format]["exits"].as_array().unwrap();
                for (a, w) in got.iter().zip(want) {
                    let w = f32s(&w["probs"]);
                    assert!(close(a, &w, tol), "{format} {ternary:?} {}", case["name"]);
                }
            }
            // Memory streams, twice: frames delimited by `new_frame` alone (each frame is written
            // when the next one starts), then each frame ended by `end_frame` (written at once,
            // so the size of the memory after it can be read).
            for (sid, seq) in seqs.iter().enumerate() {
                for ended in [false, true] {
                    let name = seq["name"].as_str().unwrap();
                    let mut e = Engine::from_gguf_with(&model, opts).unwrap();
                    e.set_sleep(SleepOptions {
                        passes: seq["passes"].as_u64().unwrap() as usize,
                        memory_off: seq["memory_off"].as_bool().unwrap(),
                        ..Default::default()
                    });
                    let stream = sid as u64 + 1;
                    for (t, frame) in seq["frames"].as_array().unwrap().iter().enumerate() {
                        let state = frame["state"].as_str().unwrap();
                        let (qtype, ins, texts) = question(state, &seq["question"]);
                        let (ids, values, _) = e.s2_engine_mut().unwrap().state_stream(state);
                        assert_eq!(ids, u32s(&frame["state_stream"]["ids"]), "{name} {t}");
                        assert_eq!(values, f32s(&frame["state_stream"]["values"]));
                        e.new_frame(stream, frame["pause"].as_bool().unwrap());
                        let p = e
                            .decide_raw_stream(stream, qtype, &ins, &texts, state)
                            .unwrap();
                        let exp = &frame["expected"][format];
                        let stats = e.last_stats().unwrap();
                        assert_eq!(
                            stats.exit,
                            Some(exp["exit"].as_u64().unwrap() as usize),
                            "{format} {ternary:?} {name} frame {t}: exit"
                        );
                        let want = f32s(&exp["probs"]);
                        for (x, y) in p.iter().zip(&want) {
                            max_err = max_err.max((x - y).abs());
                        }
                        assert!(
                            close(&p, &want, tol),
                            "{format} {ternary:?} {name} frame {t}: {p:?} vs {want:?}"
                        );
                        let last = t + 1 == seq["frames"].as_array().unwrap().len();
                        if ended || last {
                            let s2 = e.s2_engine_mut().unwrap();
                            s2.end_frame(stream);
                            let mem = s2.stream_memory(stream).unwrap();
                            let got = mem.iter().map(|v| v * v).sum::<f32>().sqrt();
                            let want = exp["memory_norm_after"].as_f64().unwrap() as f32;
                            assert!(
                                (got - want).abs() <= 10.0 * tol * want.max(1.0),
                                "{format} {ternary:?} {name} frame {t}: |S| {got} vs {want}"
                            );
                        }
                    }
                }
            }
            eprintln!("{format} ({ternary:?} compute): max |p - golden| = {max_err:.2e}");
        }
    }
}
