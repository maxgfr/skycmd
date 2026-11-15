//! Parity of the browser forecaster (`sky_web::forecast::Forecaster`, the /forecast/ page's
//! call). The same test runs natively (`cargo test -p sky-web --test forecast`) and on
//! `wasm32-unknown-unknown` (`wasm-pack test --node crates/sky-web --test forecast`): both must
//! give `tests/parity/forecast-expected.json` within 1e-4, for an s6-cast v1 file (the femto
//! golden file of sky-infer) and a v2 file with the periodicity front end (the same weights with
//! random phase columns and `draft_proj`, docs/s6-cast2-prereg.md §4), and natively they must
//! equal `sky_infer::cast::Forecaster` itself. The expected file is written by the native run
//! with `SKY_PARITY_UPDATE=1`.

use serde_json::Value;
use sky_gguf::{GgmlType, Gguf, GgufWriter, OwnedValue};
use sky_web::forecast::Forecaster;

#[cfg(target_arch = "wasm32")]
use wasm_bindgen_test::wasm_bindgen_test as test;

const V1: &[u8] = include_bytes!("../../sky-infer/tests/golden/cast-f32.gguf");
const EXPECTED: &str = include_str!("parity/forecast-expected.json");
const TOL: f32 = 1e-4;

/// Deterministic weights from basic operations only (the same bits on every target).
fn noise(seed: u64, n: usize, scale: f32) -> Vec<f32> {
    let mut s = seed ^ 0x9E37_79B9_7F4A_7C15;
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 2.0 * scale
        })
        .collect()
}

/// The v1 golden file with the v2 front end: 4 periods, the draft, random extra weights.
fn v2_file() -> Vec<u8> {
    let g = Gguf::parse(V1).unwrap();
    let (d, p, k2) = (32usize, 8usize, 8usize);
    let mut w = GgufWriter::new().with_alignment(g.alignment);
    for (k, v) in &g.kvs {
        if *k == "general.alignment" {
            continue;
        }
        if *k == "general.name" {
            w.add_kv(*k, "s6-cast2-femto-random");
        } else {
            w.add_kv(*k, OwnedValue::from_value(v));
        }
    }
    w.add_kv("skycmd-cast.periods", OwnedValue::U32(4));
    w.add_kv("skycmd-cast.draft", true);
    let phase = noise(1, d * k2, 0.5);
    for t in &g.tensors {
        if t.name == "patch_proj.weight" {
            let old = g.tensor_f32(t.name).unwrap();
            let wide: Vec<f32> = (0..d)
                .flat_map(|r| {
                    old[r * 2 * p..(r + 1) * 2 * p]
                        .iter()
                        .chain(&phase[r * k2..(r + 1) * k2])
                        .copied()
                        .collect::<Vec<_>>()
                })
                .collect();
            w.add_tensor_f32(
                t.name,
                &[(2 * p + k2) as u64, d as u64],
                GgmlType::F32,
                &wide,
            )
            .unwrap();
        } else {
            w.add_tensor(t.name, &t.dims, t.dtype, g.tensor_bytes(t).to_vec())
                .unwrap();
        }
    }
    let n_in = 2 * p + k2;
    w.add_tensor_f32(
        "draft_proj.weight",
        &[n_in as u64, d as u64],
        GgmlType::F32,
        &noise(2, d * n_in, 0.3),
    )
    .unwrap();
    w.add_tensor_f32(
        "draft_proj.bias",
        &[d as u64],
        GgmlType::F32,
        &noise(3, d, 0.1),
    )
    .unwrap();
    w.to_bytes().unwrap()
}

/// The histories: a 24-step cycle (periods detected), a constant (none), a short one, gaps.
fn histories() -> Vec<Vec<f32>> {
    let cycle: Vec<f32> = (0..500)
        .map(|t| 10.0 + ((t * 7) % 24) as f32 * 0.25 + (t % 3) as f32 * 0.1)
        .collect();
    let mut gaps = cycle[..200].to_vec();
    for i in (5..200).step_by(11) {
        gaps[i] = f32::NAN;
    }
    vec![
        cycle,
        vec![3.5; 100],
        (0..21).map(|t| (t % 6) as f32).collect(),
        gaps,
    ]
}

/// Every forecast of a model (40 steps: a rollout of three 16-step passes).
fn forecasts(bytes: &[u8]) -> Vec<Vec<f32>> {
    let mut f = Forecaster::new(bytes).expect("model loads");
    histories()
        .iter()
        .map(|h| f.forecast(h, 40, false).expect("forecast"))
        .collect()
}

fn check(name: &str, bytes: &[u8]) {
    let got = forecasts(bytes);
    #[cfg(not(target_arch = "wasm32"))]
    {
        // natively, the browser class is the engine's forecaster
        let mut m = sky_infer::cast::Forecaster::from_bytes(bytes).unwrap();
        m.prepare_compute(sky_infer::TernaryCompute::Auto);
        for (h, g) in histories().iter().zip(&got) {
            let want = m.forecast(h, 40, Default::default()).unwrap().quantiles;
            assert_eq!(g, &want, "{name}");
        }
        if std::env::var("SKY_PARITY_UPDATE").is_ok_and(|v| v == "1") {
            let path = concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/tests/parity/forecast-expected.json"
            );
            let mut all: serde_json::Map<String, Value> = serde_json::from_str(
                &std::fs::read_to_string(path).unwrap_or_else(|_| "{}".into()),
            )
            .unwrap_or_default();
            all.insert(name.into(), serde_json::to_value(&got).unwrap());
            std::fs::write(path, serde_json::to_string_pretty(&all).unwrap() + "\n").unwrap();
            return;
        }
    }
    let expected: Value = serde_json::from_str(EXPECTED).expect("forecast-expected.json");
    let want: Vec<Vec<f32>> = serde_json::from_value(expected[name].clone())
        .unwrap_or_else(|_| panic!("forecast-expected.json has no entry for {name}"));
    assert_eq!(got.len(), want.len(), "{name}: number of histories");
    for (i, (g, w)) in got.iter().zip(&want).enumerate() {
        assert_eq!(g.len(), w.len(), "{name} history {i}");
        let worst = g
            .iter()
            .zip(w)
            .map(|(a, b)| (a - b).abs() / b.abs().max(1.0))
            .fold(0.0f32, f32::max);
        assert!(worst <= TOL, "{name} history {i}: {worst}");
    }
}

#[test]
fn s6_cast_v1() {
    check("s6-cast-femto", V1);
}

#[test]
fn s6_cast_v2_front_end() {
    let bytes = v2_file();
    check("s6-cast2-femto-random", &bytes);
    // the page reads the front end from info()
    let f = Forecaster::new(&bytes).unwrap();
    let info: Value = serde_json::from_str(&f.info()).unwrap();
    assert_eq!(info["periods"], 4);
    assert_eq!(info["draft"], true);
    let v1: Value = serde_json::from_str(&Forecaster::new(V1).unwrap().info()).unwrap();
    assert_eq!(
        (v1["periods"].clone(), v1["draft"].clone()),
        (0.into(), false.into())
    );
}
