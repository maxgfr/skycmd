//! The v2 front end of s6-cast in the engine (docs/s6-cast2-prereg.md §4.2–4.4), on fixtures
//! built from the v1 golden file (`tests/golden/cast-f32.gguf`, the femto shape):
//!
//! - a v2 file whose extra weights are zero (phase columns of `patch_proj`, `draft_proj`) gives
//!   v1's golden outputs: the front end is wired without touching the v1 path;
//! - `draft_proj` adds its projection of the draft to the queries: a v2 model equals a v1 model
//!   whose `cast_query` holds that sum, window by window;
//! - the phase channels go through `patch_proj`'s extra columns;
//! - the rollout detects the periods again on every pass;
//! - a v1 file has no front end, and a v2 model refuses a window without one.
//!
//! The graph parity of the front end against sky-train's mlx-rs graph is `golden_cast2.rs`.

use std::path::PathBuf;

use serde_json::Value;
use sky_gguf::{GgmlType, Gguf, GgufWriter, OwnedValue};
use sky_infer::cast::prep;
use sky_infer::cast::CastModel;
use sky_infer::TernaryCompute;

const PERIODS: usize = 4;

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

/// A small deterministic generator for the test weights.
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

/// The f32 golden file rewritten with the front end: `skycmd-cast.periods` and
/// `skycmd-cast.draft`, `patch_proj.weight` widened by `phase` (`[d][2K]`, row-major), and
/// `draft_proj` from `draft` (`[d][2·patch + 2K]`) and `draft_bias`; `query` replaces
/// `cast_query.weight` when given. With `periods = 0` the result is a v1 file.
fn rewrite(
    periods: usize,
    phase: Option<&[f32]>,
    draft: Option<(&[f32], &[f32])>,
    query: Option<&[f32]>,
) -> Vec<u8> {
    let bytes = std::fs::read(dir().join("cast-f32.gguf")).unwrap();
    let g = Gguf::parse(&bytes).unwrap();
    let m = CastModel::from_bytes(&bytes).unwrap();
    let (d, p) = (m.cfg.hidden_size, m.cfg.patch);
    let k2 = 2 * periods;
    let mut w = GgufWriter::new().with_alignment(g.alignment);
    for (k, v) in &g.kvs {
        if *k != "general.alignment" {
            w.add_kv(*k, OwnedValue::from_value(v));
        }
    }
    if periods > 0 {
        w.add_kv("skycmd-cast.periods", OwnedValue::U32(periods as u32));
    }
    if draft.is_some() {
        w.add_kv("skycmd-cast.draft", true);
    }
    for t in &g.tensors {
        match t.name {
            "patch_proj.weight" if periods > 0 => {
                let old = g.tensor_f32(t.name).unwrap();
                let wide: Vec<f32> = (0..d)
                    .flat_map(|r| {
                        let ph: Vec<f32> = match phase {
                            Some(ph) => ph[r * k2..(r + 1) * k2].to_vec(),
                            None => vec![0.0; k2],
                        };
                        old[r * 2 * p..(r + 1) * 2 * p].iter().copied().chain(ph)
                    })
                    .collect();
                w.add_tensor_f32(
                    t.name,
                    &[(2 * p + k2) as u64, d as u64],
                    GgmlType::F32,
                    &wide,
                )
                .unwrap();
            }
            "cast_query.weight" if query.is_some() => {
                w.add_tensor_f32(t.name, &t.dims, GgmlType::F32, query.unwrap())
                    .unwrap();
            }
            _ => {
                w.add_tensor(t.name, &t.dims, t.dtype, g.tensor_bytes(t).to_vec())
                    .unwrap();
            }
        }
    }
    if let Some((dw, db)) = draft {
        let n_in = 2 * p + k2;
        w.add_tensor_f32(
            "draft_proj.weight",
            &[n_in as u64, d as u64],
            GgmlType::F32,
            dw,
        )
        .unwrap();
        w.add_tensor_f32("draft_proj.bias", &[d as u64], GgmlType::F32, db)
            .unwrap();
    }
    w.to_bytes().unwrap()
}

fn load(bytes: &[u8]) -> CastModel {
    let mut m = CastModel::from_bytes(bytes).unwrap();
    m.prepare_compute(TernaryCompute::F32);
    m
}

fn worst(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

fn golden_cases() -> Vec<(String, Vec<f32>, Vec<Vec<f32>>)> {
    let g: Value =
        serde_json::from_slice(&std::fs::read(dir().join("cast.json")).unwrap()).unwrap();
    g["cases"]
        .as_array()
        .unwrap()
        .iter()
        .map(|c| {
            let want = c["quantiles"]["f32"]
                .as_array()
                .unwrap()
                .iter()
                .map(floats)
                .collect();
            (
                c["name"].as_str().unwrap().to_string(),
                floats(&c["context"]),
                want,
            )
        })
        .collect()
}

/// Contexts with a known period, with none, and a short history (as well as the golden ones).
fn fixtures() -> Vec<(&'static str, Vec<f32>)> {
    let tau = 2.0 * std::f32::consts::PI;
    vec![
        (
            "period-24",
            (0..400)
                .map(|t| 5.0 + (tau * t as f32 / 24.0).sin() + 0.05 * ((t * 7) % 5) as f32)
                .collect(),
        ),
        ("none", vec![2.5; 200]),
        (
            "short",
            (0..29).map(|t| (tau * t as f32 / 6.0).cos()).collect(),
        ),
    ]
}

#[test]
fn a_v1_file_has_no_front_end() {
    let m = load(&std::fs::read(dir().join("cast-f32.gguf")).unwrap());
    assert_eq!((m.cfg.periods, m.cfg.draft), (0, false));
    assert!(m.draft_proj.is_none());
    // its inputs are those of v1's prepare
    let h: Vec<f32> = (0..100).map(|t| t as f32).collect();
    let i = m.inputs(&h);
    let (n, v, k) = prep::prepare(&h, m.cfg.patch, m.cfg.context_patches);
    assert_eq!((i.norm, i.vals, i.mask), (n, v, k));
}

#[test]
fn zero_extra_weights_give_the_v1_golden_outputs() {
    let d = 32;
    let k2 = 2 * PERIODS;
    let m = load(&rewrite(
        PERIODS,
        None,
        Some((&vec![0.0; d * (16 + k2)], &vec![0.0; d])),
        None,
    ));
    assert_eq!((m.cfg.periods, m.cfg.draft), (PERIODS, true));
    assert_eq!(m.patch_proj.n_in, 2 * m.cfg.patch + k2);
    for (name, ctx, want) in golden_cases() {
        let inp = m.inputs(&ctx);
        assert_eq!(inp.periods.len(), PERIODS);
        let got = m.forward_inputs(&inp, true).unwrap();
        assert_eq!(got.len(), want.len());
        for (e, (a, b)) in got.iter().zip(&want).enumerate() {
            let w = worst(a, b);
            assert!(w <= 1e-4, "{name} exit {e}: {w}");
        }
    }
    // a v2 model needs its front end: the v1 entry point refuses the window
    let (_, vals, mask) = prep::prepare(&[1.0, 2.0, 3.0], 8, 40);
    assert!(m.forward_window(&vals, &mask, true).is_err());
}

#[test]
fn the_draft_is_projected_into_the_queries() {
    let (d, p, k2) = (32usize, 8usize, 2 * PERIODS);
    let dw = noise(1, d * (2 * p + k2), 0.3);
    let db = noise(2, d, 0.1);
    let v2 = load(&rewrite(PERIODS, None, Some((&dw, &db)), None));
    let base = load(&std::fs::read(dir().join("cast-f32.gguf")).unwrap());
    let mut moved = 0.0f32;
    for (name, ctx) in fixtures()
        .into_iter()
        .map(|(n, c)| (n.to_string(), c))
        .chain(golden_cases().into_iter().map(|(n, c, _)| (n, c)))
    {
        let inp = v2.inputs(&ctx);
        assert_eq!(inp.draft.len(), 2 * (2 * p + k2));
        // the v1 model with cast_query + draft_proj(draft) as its learned queries
        let mut q = base.query.clone();
        for (qi, row) in inp.draft.chunks_exact(2 * p + k2).enumerate() {
            for r in 0..d {
                let s: f32 = (0..2 * p + k2)
                    .map(|c| dw[r * (2 * p + k2) + c] * row[c])
                    .sum();
                q[qi * d + r] += s + db[r];
            }
        }
        let folded = load(&rewrite(0, None, None, Some(&q)));
        let (_, vals, mask) = prep::prepare(&ctx, p, 40);
        let want = folded.forward_window(&vals, &mask, true).unwrap();
        let got = v2.forward_inputs(&inp, true).unwrap();
        for (e, (a, b)) in got.iter().zip(&want).enumerate() {
            let w = worst(a, b);
            assert!(w <= 1e-4, "{name} exit {e}: {w}");
        }
        let plain = base.forward_window(&vals, &mask, true).unwrap();
        moved = moved.max(worst(got.last().unwrap(), plain.last().unwrap()));
    }
    assert!(moved > 1e-3, "the draft changed nothing ({moved})");
}

#[test]
fn the_phase_channels_enter_through_patch_proj() {
    let (d, k2) = (32usize, 2 * PERIODS);
    let ph = noise(3, d * k2, 0.5);
    let v2 = load(&rewrite(PERIODS, Some(&ph), None, None));
    assert!(v2.draft_proj.is_none() && v2.cfg.periods == PERIODS);
    let base = load(&std::fs::read(dir().join("cast-f32.gguf")).unwrap());
    for (name, ctx) in fixtures() {
        let inp = v2.inputs(&ctx);
        let got = v2.forward_inputs(&inp, true).unwrap();
        let (_, vals, mask) = prep::prepare(&ctx, 8, 40);
        let plain = base.forward_window(&vals, &mask, true).unwrap();
        let diff = worst(got.last().unwrap(), plain.last().unwrap());
        if inp.periods.iter().all(|&p| p == 0) {
            // no period: every phase channel is 0, the extra columns read nothing
            assert!(diff <= 1e-5, "{name}: {diff}");
        } else {
            assert!(diff > 1e-4, "{name}: the phase changed nothing");
        }
    }
    // the fixtures cover both cases
    let tags: Vec<bool> = fixtures()
        .iter()
        .map(|(_, c)| v2.inputs(c).periods[0] > 0)
        .collect();
    assert!(tags.contains(&true) && tags.contains(&false), "{tags:?}");
    assert_eq!(v2.inputs(&fixtures()[0].1).periods[0], 24);
}

#[test]
fn the_rollout_detects_the_periods_on_every_pass() {
    let (d, p, k2) = (32usize, 8usize, 2 * PERIODS);
    let v2 = load(&rewrite(
        PERIODS,
        Some(&noise(4, d * k2, 0.5)),
        Some((&noise(5, d * (2 * p + k2), 0.3), &noise(6, d, 0.1))),
        None,
    ));
    let hist = fixtures()[0].1.clone();
    let step = v2.cfg.horizon();
    let f = v2.forecast(&hist, 2 * step, Default::default()).unwrap();
    assert_eq!(f.passes, 2);
    // the second pass is one window pass on the history and the first pass's medians
    let mut ctx = hist.clone();
    ctx.extend(f.median()[..step].iter().copied());
    let inp = v2.inputs(&ctx);
    let last = v2.forward_inputs(&inp, true).unwrap().pop().unwrap();
    let levels = f.levels.len();
    for (a, b) in f.quantiles[step * levels..].iter().zip(&last) {
        assert!((a - inp.norm.invert(*b)).abs() < 1e-4 * inp.norm.scale.max(1.0));
    }
    // quantiles stay ordered and finite through the rollout, flip included
    let flip = v2
        .forecast(
            &hist,
            step + 3,
            sky_infer::cast::ForecastOptions {
                flip: true,
                full_depth: true,
            },
        )
        .unwrap();
    for row in flip.quantiles.chunks(levels) {
        assert!(row.iter().all(|v| v.is_finite()) && row.windows(2).all(|w| w[0] <= w[1]));
    }
    // n_params: patch_proj widened by d·2K, draft_proj d·(2p + 2K) + d
    let mut v1 = load(&std::fs::read(dir().join("cast-f32.gguf")).unwrap());
    let mut v2 = v2;
    assert_eq!(v2.n_params(), v1.n_params() + d * k2 + d * (2 * p + k2) + d);
}
