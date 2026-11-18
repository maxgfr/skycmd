//! The BM25 scores of s6-route-nano (`sky_infer::route`) against the fixture written with
//! `bm25s` (`tests/fixtures/route_bm25.json`, `scripts/route/make_bm25_fixture.py`): the data
//! builder and the engine both call this code, and both must give these numbers.

use serde_json::Value;

#[test]
fn bm25_matches_the_fixture() {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/route_bm25.json"
    );
    let v: Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
    let cases = v["cases"].as_array().unwrap();
    assert!(cases.len() >= 8);
    for c in cases {
        let request = c["request"].as_str().unwrap();
        let options: Vec<&str> = c["options"]
            .as_array()
            .unwrap()
            .iter()
            .map(|o| o.as_str().unwrap())
            .collect();
        let want: Vec<f64> = c["scores"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_f64().unwrap())
            .collect();
        let got = sky_infer::route::bm25(request, &options);
        for (g, w) in got.iter().zip(&want) {
            // bm25s computes in float32
            assert!((g - w).abs() < 1e-5, "{request}: {got:?} against {want:?}");
        }
        let ranks: Vec<usize> = c["ranks"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_u64().unwrap() as usize)
            .collect();
        assert_eq!(sky_infer::route::ranks(&got), ranks, "{request}");
        assert_eq!(
            sky_infer::route::state(request, &options),
            c["state"].as_str().unwrap(),
            "{request}"
        );
    }
}

/// s6-route v4 (`docs/s6-route4-prereg.md` §3.1, D4): the same slates as the model reads them
/// under `skycmd-s2.route.render = scores-in-options`; v2's fixture above is unchanged.
#[test]
fn scores_in_options_match_the_v4_fixture() {
    use sky_infer::route::{plain_option, Render};
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/route_bm25_v4.json"
    );
    let v: Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
    let cases = v["cases"].as_array().unwrap();
    assert!(cases.len() >= 8);
    let strs = |x: &Value| -> Vec<String> {
        x.as_array()
            .unwrap()
            .iter()
            .map(|o| o.as_str().unwrap().to_string())
            .collect()
    };
    for c in cases {
        let request = c["request"].as_str().unwrap();
        let options = strs(&c["options"]);
        let (opts, state) = Render::ScoresInOptions.model_input(request, &options);
        assert_eq!(opts, strs(&c["model_options"]), "{request}");
        assert_eq!(state, c["state"].as_str().unwrap(), "{request}");
        let back: Vec<&str> = opts.iter().map(|o| plain_option(o)).collect();
        assert_eq!(back, options.iter().map(String::as_str).collect::<Vec<_>>());
        // v2's rendering of the same slate is v2's fixture
        let (o2, s2) = Render::Line.model_input(request, &options);
        assert_eq!(o2, options);
        assert_eq!(s2, sky_infer::route::state_for_options(request, &options));
    }
}
