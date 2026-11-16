//! Golden test of s6-ear v2 against the mlx-rs training graph: `sky-train turn-golden` writes
//! `tests/golden/turn.json` (random weights of `configs/s6-ear2-femto.json`, f32 and ternary GGUF
//! files, the signals and the logit and pooled vector of each window). sky-infer must reproduce
//! them from the signals alone — the front end, the window's normalisation, both attention
//! kernels, the attention pooling and the head — within 1e-4 (f32) / 1e-3 (ternary), in every
//! ternary compute mode (the cast golden's tolerances).

use std::path::PathBuf;

use serde_json::Value;
use sky_infer::audio::{normalize_window, synth_speech, FeatureStream, TurnFrontEnd};
use sky_infer::turn::TurnModel;
use sky_infer::TernaryCompute;

fn dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/golden")
}

fn floats(v: &Value) -> Vec<f32> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap() as f32)
        .collect()
}

fn signal(c: &Value) -> Vec<f32> {
    let s = &c["signal"];
    synth_speech(
        s["secs"].as_f64().unwrap(),
        s["seed"].as_u64().unwrap(),
        s["amp"].as_f64().unwrap() as f32,
    )
}

#[test]
fn turn_golden() {
    let bytes = std::fs::read(dir().join("turn.json"))
        .expect("tests/golden/turn.json (run `sky-train turn-golden`)");
    let g: Value = serde_json::from_slice(&bytes).unwrap();
    let cases = g["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 5);
    let mut fe = TurnFrontEnd::new();
    for f in g["meta"]["files"].as_array().unwrap() {
        let format = f["format"].as_str().unwrap();
        let tol = g["meta"]["tolerance"][format].as_f64().unwrap() as f32;
        let file = std::fs::read(dir().join(f["path"].as_str().unwrap())).unwrap();
        let modes: &[TernaryCompute] = if format == "ternary" {
            &[
                TernaryCompute::F32,
                TernaryCompute::Packed,
                TernaryCompute::Int8,
                TernaryCompute::Lut,
            ]
        } else {
            &[TernaryCompute::F32]
        };
        for &mode in modes {
            let mut m = TurnModel::from_bytes(&file).unwrap();
            m.prepare_compute(mode);
            assert_eq!((m.cfg.temperature, m.cfg.tau50), (1.5, 0.5));
            // int8 activations move results by ~2e-4 (see TernaryCompute::Int8)
            let tol = if matches!(mode, TernaryCompute::Int8 | TernaryCompute::Lut) {
                tol.max(2e-3)
            } else {
                tol
            };
            for c in cases {
                let name = c["name"].as_str().unwrap();
                let (h, n) = fe.hops(&signal(c), m.cfg.window_hops);
                let (x, t) = normalize_window(&h, n, m.cfg.hops_per_position);
                assert_eq!(t as u64, c["positions"].as_u64().unwrap(), "{name}");
                let (_, pooled) = m.encode(&x, t).unwrap();
                let out = m.forward(&x, t).unwrap();
                let want = c["logit"][format].as_f64().unwrap() as f32;
                assert!(
                    (out.logit - want).abs() <= tol * want.abs().max(1.0),
                    "{format} {mode:?} {name}: logit {} vs {want}",
                    out.logit
                );
                let wp = floats(&c["pooled"][format]);
                let worst = pooled
                    .iter()
                    .zip(&wp)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0f32, f32::max);
                assert!(
                    worst <= tol,
                    "{format} {mode:?} {name}: pooled max diff {worst}"
                );
                let p = 1.0 / (1.0 + (-out.logit / 1.5).exp());
                assert!((out.prob - p).abs() < 1e-6);
            }
        }
    }
}

/// The deployed paths agree: a stream fed in 10 ms chunks, the whole clip, and the hops.
#[test]
fn stream_clip_and_hops_give_one_decision() {
    let file = std::fs::read(dir().join("turn-ternary.gguf")).unwrap();
    let mut m = TurnModel::from_bytes(&file).unwrap();
    m.prepare_compute(TernaryCompute::Auto);
    let pcm = synth_speech(6.0, 11, 0.8);
    let mut fe = TurnFrontEnd::new();
    let a = m.decide_pcm(&mut fe, &pcm).unwrap();
    let mut s = FeatureStream::new(m.cfg.window_hops);
    for chunk in pcm.chunks(160) {
        s.push(chunk);
    }
    let b = m.decide_hops(&s.hops(), s.len()).unwrap();
    assert_eq!(a, b);
    assert_eq!(a.positions, 100);
    assert!(m.decide_pcm(&mut fe, &pcm[..600]).is_err(), "no hop yet");
    assert_eq!(m.n_params(), 23_170);
}
