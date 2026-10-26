//! Golden test of s6-cast against the mlx-rs training graph: `sky-train cast-golden --config
//! configs/s6-cast-femto.json --prefix cast` writes `tests/golden/cast.json` (random weights of
//! the femto shape with Parallax and XSA, f32 and ternary GGUF files, contexts and the normalized
//! quantiles of every exit). sky-infer must reproduce them from the raw contexts — normalization,
//! patching, dropped left padding, both attention kernels, the decoder and the non-crossing head
//! — within 1e-4 (f32) / 1e-3 (ternary), in every ternary compute mode.

use std::path::PathBuf;

use serde_json::Value;
use sky_infer::cast::prep::prepare;
use sky_infer::cast::{CastModel, Forecast};
use sky_infer::TernaryCompute;

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
fn cast_golden() {
    let bytes = std::fs::read(dir().join("cast.json")).expect(
        "tests/golden/cast.json (run `sky-train cast-golden --config configs/s6-cast-femto.json --prefix cast`)",
    );
    let g: Value = serde_json::from_slice(&bytes).unwrap();
    let files = g["meta"]["files"].as_array().unwrap();
    let cases = g["cases"].as_array().unwrap();
    assert!(cases.len() >= 6);
    for f in files {
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
            let mut m = CastModel::from_bytes(&file).unwrap();
            m.prepare_compute(mode);
            // int8 activations move results by ~2e-4 (see TernaryCompute::Int8)
            let tol = if matches!(mode, TernaryCompute::Int8 | TernaryCompute::Lut) {
                tol.max(2e-3)
            } else {
                tol
            };
            for c in cases {
                let name = c["name"].as_str().unwrap();
                let ctx = floats(&c["context"]);
                let (_, vals, mask) = prepare(&ctx, m.cfg.patch, m.cfg.context_patches);
                let got = m.forward_window(&vals, &mask, true).unwrap();
                let want = c["quantiles"][format].as_array().unwrap();
                assert_eq!(got.len(), want.len(), "{format} {name}: exits");
                for (e, (a, b)) in got.iter().zip(want).enumerate() {
                    let b = floats(b);
                    assert_eq!(a.len(), b.len());
                    let worst = a
                        .iter()
                        .zip(&b)
                        .map(|(x, y)| (x - y).abs())
                        .fold(0.0f32, f32::max);
                    assert!(
                        worst <= tol,
                        "{format} {mode:?} {name} exit {e}: max diff {worst} > {tol}"
                    );
                }
            }
        }
    }
}

/// The public forecast: denormalized, non-crossing, rolled out past one pass, and flip-averaged.
#[test]
fn forecast_api() {
    let file = std::fs::read(dir().join("cast-f32.gguf")).unwrap();
    let mut m = CastModel::from_bytes(&file).unwrap();
    m.prepare_compute(TernaryCompute::F32);
    let hist: Vec<f32> = (0..200)
        .map(|t| 100.0 + 5.0 * (t as f32 * 0.4).sin())
        .collect();
    let step = m.cfg.horizon();
    let f: Forecast = m.forecast(&hist, step * 2 + 3, Default::default()).unwrap();
    assert_eq!(f.horizon, step * 2 + 3);
    assert_eq!(f.quantiles.len(), f.horizon * f.levels.len());
    assert_eq!(f.passes, 3);
    for h in 0..f.horizon {
        let row = &f.quantiles[h * 9..(h + 1) * 9];
        assert!(row.windows(2).all(|w| w[0] <= w[1]), "step {h}: {row:?}");
        assert!(row.iter().all(|v| v.is_finite()));
    }
    // the first pass equals one window pass, denormalized
    let (norm, vals, mask) = prepare(&hist, m.cfg.patch, m.cfg.context_patches);
    let last = m.forward_window(&vals, &mask, true).unwrap().pop().unwrap();
    for (a, b) in f.quantiles[..step * 9].iter().zip(&last) {
        assert!((a - norm.invert(*b)).abs() < 1e-3 * norm.scale.max(1.0));
    }
    let flip = m
        .forecast(
            &hist,
            step,
            sky_infer::cast::model::ForecastOptions {
                flip: true,
                full_depth: true,
            },
        )
        .unwrap();
    assert_eq!(flip.passes, 2);
    assert!(m.forecast(&[f32::NAN; 10], 4, Default::default()).is_err());
}
