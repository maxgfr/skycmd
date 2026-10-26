//! Parity of the TinyCast port with the official PyTorch predictor: `scripts/cast-ref-golden.py`
//! runs `TinyCastPredictor.predict` (CPU, fp32) of github.com/raws-labs/tinycast on its released
//! weights and writes `tests/golden/tinycast.json` (eight contexts, horizons 48 and 100, with and
//! without flip); `tests/golden/tinycast.gguf` is the same release converted by `sky-cast
//! convert-tinycast`. Every quantile must match within `TOL` of the context's scale (the range
//! of its observed values), i.e. in the model's normalized units, plus a few ulps of the value
//! (a constant context has a range of 1e-5, below the f32 resolution of its level).

use std::path::PathBuf;

use serde_json::Value;
use sky_infer::cast::{ForecastOptions, Forecaster, TinyCastModel};

/// Largest error allowed, relative to the context's range (1e-5 is reached, on the context
/// offset by 1e4, where it is two ulps of the output).
const TOL: f32 = 5e-5;

fn dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/golden")
}

fn floats(v: &Value) -> Vec<f32> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().map_or(f32::NAN, |x| x as f32))
        .collect()
}

#[test]
fn tinycast_matches_the_official_predictor() {
    let g: Value = serde_json::from_slice(
        &std::fs::read(dir().join("tinycast.json")).expect("tests/golden/tinycast.json"),
    )
    .unwrap();
    let m =
        TinyCastModel::from_bytes(&std::fs::read(dir().join("tinycast.gguf")).unwrap()).unwrap();
    assert_eq!(m.n_params(), 146_505);
    let cases = g["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 32);
    let mut worst = 0.0f32;
    for c in cases {
        let name = c["context"].as_str().unwrap();
        let history = floats(&g["contexts"][name]);
        let horizon = c["horizon"].as_u64().unwrap() as usize;
        let flip = c["flip"].as_bool().unwrap();
        let want = floats(&c["quantiles"]);
        let opts = ForecastOptions {
            flip,
            full_depth: false,
        };
        let f = m.forecast(&history, horizon, opts).unwrap();
        assert_eq!(f.quantiles.len(), want.len(), "{name}");
        assert_eq!(f.passes, horizon.div_ceil(48) * if flip { 2 } else { 1 });
        let obs = history.iter().filter(|v| v.is_finite());
        let range = obs.clone().fold(f32::NEG_INFINITY, |a, &b| a.max(b))
            - obs.fold(f32::INFINITY, |a, &b| a.min(b));
        let scale = range.max(1e-5);
        let (abs, at) = f
            .quantiles
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).abs())
            .enumerate()
            .fold((0.0f32, 0), |m, (i, e)| if e > m.0 { (e, i) } else { m });
        let rel = abs / scale;
        eprintln!(
            "{name:>20} h={horizon:<3} flip={flip:<5}  max abs {abs:.3e}  rel {rel:.3e}  (scale {scale})"
        );
        let bad = f
            .quantiles
            .iter()
            .zip(&want)
            .any(|(a, b)| (a - b).abs() > TOL * scale + 4.0 * f32::EPSILON * b.abs());
        assert!(
            !bad,
            "{name} h={horizon} flip={flip}: step {} level {}: {} vs {} (rel {rel:.2e})",
            at / 9,
            at % 9,
            f.quantiles[at],
            want[at]
        );
        worst = worst.max(rel);
    }
    eprintln!("worst relative error {worst:.3e}");
}

#[test]
fn forecaster_dispatches_on_the_architecture() {
    let mut f =
        Forecaster::from_bytes(&std::fs::read(dir().join("tinycast.gguf")).unwrap()).unwrap();
    assert!(matches!(f, Forecaster::TinyCast(_)));
    assert_eq!(f.name(), "TinyCast");
    assert_eq!(f.n_params(), 146_505);
    assert_eq!(f.pass_horizon(), 48);
    let cast =
        Forecaster::from_bytes(&std::fs::read(dir().join("cast-f32.gguf")).unwrap()).unwrap();
    assert!(matches!(cast, Forecaster::Cast(_)));
    let history: Vec<f32> = (0..300).map(|t| (t as f32 / 5.0).sin()).collect();
    let q = f
        .forecast(&history, 20, ForecastOptions::default())
        .unwrap();
    assert_eq!((q.horizon, q.quantiles.len()), (20, 180));
    assert!(q
        .quantiles
        .chunks(9)
        .all(|r| r.windows(2).all(|w| w[0] <= w[1])));
}
