//! Golden test of s6-cast v2's front end against the mlx-rs training graph
//! (docs/s6-cast2-prereg.md §4.5): `sky-train cast-golden --config configs/s6-cast2-femto.json
//! --prefix cast2` writes `tests/golden/cast2.json` and its two GGUF files (the femto shape of
//! `golden_cast.rs` with the detected periods, the phase channels and the seasonal draft). The
//! engine must find the same periods and reproduce every exit from the raw contexts, at the
//! tolerances of `golden_cast.rs`, in every ternary compute mode.
//!
//! The files are made by sky-train on Apple Silicon. Until they are committed this test is
//! ignored; `cast_front_end.rs` covers the engine side on fixtures meanwhile.

use std::path::PathBuf;

use serde_json::Value;
use sky_infer::cast::CastModel;
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
#[ignore = "needs tests/golden/cast2.json: sky-train cast-golden --config configs/s6-cast2-femto.json --prefix cast2"]
fn cast2_golden() {
    let g: Value =
        serde_json::from_slice(&std::fs::read(dir().join("cast2.json")).unwrap()).unwrap();
    let files = g["meta"]["files"].as_array().unwrap();
    let cases = g["cases"].as_array().unwrap();
    let names: Vec<&str> = cases.iter().map(|c| c["name"].as_str().unwrap()).collect();
    for want in ["period-24", "no-period", "short-history", "missing"] {
        assert!(names.contains(&want), "case {want} missing");
    }
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
            assert!(m.cfg.periods > 0 && m.cfg.draft, "not a v2 file");
            m.prepare_compute(mode);
            let tol = if matches!(mode, TernaryCompute::Int8 | TernaryCompute::Lut) {
                tol.max(2e-3)
            } else {
                tol
            };
            for c in cases {
                let name = c["name"].as_str().unwrap();
                let inp = m.inputs(&floats(&c["context"]));
                let periods: Vec<usize> = c["periods"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|p| p.as_u64().unwrap() as usize)
                    .collect();
                assert_eq!(inp.periods, periods, "{name}: periods");
                let got = m.forward_inputs(&inp, true).unwrap();
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
    let period = cases.iter().find(|c| c["name"] == "period-24").unwrap();
    assert_eq!(period["periods"][0], 24);
}
