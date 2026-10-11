//! `sky_infer::num::extract` mirrors `sky_data::num::extract` bit for bit: checked against the
//! shared test vectors `crates/sky-data/tests/fixtures/num_extract.json`
//! (`[{input, text, values, bits}]`), and the value features `φ(v)` against f64 math.

use std::path::PathBuf;

use sky_infer::num::{extract, features, NUM_FEATURES};

#[test]
fn matches_sky_data_vectors() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../sky-data/tests/fixtures/num_extract.json");
    let Ok(bytes) = std::fs::read(&path) else {
        eprintln!("skipped: {} not found", path.display());
        return;
    };
    let all: Vec<serde_json::Value> = serde_json::from_slice(&bytes).unwrap();
    assert!(all.len() >= 30);
    for v in &all {
        let input = v["input"].as_str().unwrap();
        let ex = extract(input);
        assert_eq!(ex.text, v["text"].as_str().unwrap(), "text of {input:?}");
        let bits: Vec<u32> = ex.values.iter().map(|x| x.to_bits()).collect();
        let want: Vec<u32> = serde_json::from_value(v["bits"].clone()).unwrap();
        assert_eq!(bits, want, "values of {input:?}");
    }
}

#[test]
fn features_match_f64_math() {
    let mut f = [0.0f32; NUM_FEATURES];
    for &v in &[0.0f32, 1.0, -3.0, 12.0, 40.5, -0.25, 1e6, -123456.0] {
        features(v, &mut f);
        let u = if v == 0.0 {
            0.0
        } else {
            (v as f64).signum() * (v.abs() as f64).ln_1p()
        };
        assert!((f[0] as f64 - u).abs() < 1e-6 * u.abs().max(1.0));
        for k in 0..8 {
            let w = 2f64.powi(k as i32) * std::f64::consts::PI / 8.0;
            // the angle is rounded to f32 first, like the f32 training graph
            let tol = 1e-6 + 1.2e-7 * (u * w).abs();
            assert!((f[1 + k] as f64 - (u * w).sin()).abs() < tol, "sin {v} {k}");
            assert!((f[9 + k] as f64 - (u * w).cos()).abs() < tol, "cos {v} {k}");
        }
    }
}
