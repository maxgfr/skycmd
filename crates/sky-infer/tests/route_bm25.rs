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
