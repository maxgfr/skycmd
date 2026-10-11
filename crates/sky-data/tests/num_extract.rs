//! Checks `num::extract` against the shared test vectors in `tests/fixtures/num_extract.json`
//! (mirrored by `sky-infer` and `sky-train`). Set `SKY_REGEN_FIXTURES=1` to rewrite the file
//! from [`INPUTS`] after an intentional rule change.

use std::path::PathBuf;

use serde_json::{json, Value};
use sky_data::num::extract;

const INPUTS: [&str; 30] = [
    "alt 12 vy -3 dx 40 gt 20 gb 35",
    "vy -3",
    "glm-5",
    "b2",
    "3.5km",
    "1,000",
    "-0",
    "-0.0",
    "x=-0.25,7.",
    "1.2.3",
    "- 3 --4",
    "1-2",
    "Le drone vole à 120 m d'altitude pendant 3,5 minutes.",
    "Il reste 15 % de batterie et -2 °C dehors.",
    "Décolle à 30 m, puis inspecte le pylône n°7.",
    "é5 è12",
    "item_1 and x1.5y",
    "no digits here",
    "",
    "3.14159265358979",
    "123456789012",
    "0.1",
    "007",
    "1e5 and 2E-3",
    "range 10-20 km",
    "temp: -40.5, delta +3",
    "v2.0 release",
    "S2-pico has 64 dims",
    "(42) [7] {-1}",
    "\u{0661}\u{0662}\u{0663} and \u{FF12} are not ASCII digits",
];

fn fixture() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/num_extract.json")
}

fn vector(input: &str) -> Value {
    let (text, values) = extract(input);
    json!({
        "input": input,
        "text": text,
        "values": values,
        "bits": values.iter().map(|v| v.to_bits()).collect::<Vec<u32>>(),
    })
}

#[test]
fn matches_shared_vectors() {
    if std::env::var("SKY_REGEN_FIXTURES").is_ok_and(|v| v == "1") {
        let all: Vec<Value> = INPUTS.iter().map(|s| vector(s)).collect();
        std::fs::write(
            fixture(),
            serde_json::to_string_pretty(&all).unwrap() + "\n",
        )
        .unwrap();
    }
    let all: Vec<Value> = serde_json::from_slice(&std::fs::read(fixture()).unwrap()).unwrap();
    assert_eq!(all.len(), 30);
    for v in &all {
        let input = v["input"].as_str().unwrap();
        let (text, values) = extract(input);
        assert_eq!(text, v["text"].as_str().unwrap(), "text of {input:?}");
        let bits: Vec<u32> = values.iter().map(|x| x.to_bits()).collect();
        let want: Vec<u32> = serde_json::from_value(v["bits"].clone()).unwrap();
        assert_eq!(bits, want, "values of {input:?}");
    }
}

#[test]
fn spot_checks() {
    assert_eq!(extract("glm-5"), ("glm-5".to_string(), vec![]));
    assert_eq!(extract("vy -3"), ("vy [NUM]".to_string(), vec![-3.0]));
    assert_eq!(
        extract("1,000"),
        ("[NUM],[NUM]".to_string(), vec![1.0, 0.0])
    );
}
