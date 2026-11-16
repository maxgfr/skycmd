//! s6-nano v3: the embedding part against its Python reference. `tests/fixtures/route_embed.json`
//! is written by `scripts/s6-route3-embed-export.py` (model2vec 0.10.0 and
//! `scripts/route-baselines.py`): token ids of validation texts and hand-written edge cases, and
//! the probabilities of `potion-cos-4bit` on 40 validation slates. The embedding file is not in
//! the repository: `models/s6-nano3/skycmd-s6-route-embed.gguf` (or `SKY_ROUTE_EMBED`), written by
//! `sky-route embed-gguf`; without it the test says so and passes.

use std::path::PathBuf;

use serde_json::Value;
use sky_infer::embed::StaticEmbedding;

fn embedding() -> Option<StaticEmbedding> {
    let path = std::env::var("SKY_ROUTE_EMBED")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../../models/s6-nano3/skycmd-s6-route-embed.gguf")
        });
    match std::fs::read(&path) {
        Ok(b) => Some(StaticEmbedding::from_gguf(&b).expect("a valid embedding file")),
        Err(_) => {
            eprintln!(
                "route_embed: {} not found (sky-route embed-gguf): skipped",
                path.display()
            );
            None
        }
    }
}

fn fixture() -> Value {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/route_embed.json");
    serde_json::from_slice(&std::fs::read(p).expect("route_embed.json")).expect("valid JSON")
}

#[test]
fn token_ids_match_model2vec() {
    let Some(e) = embedding() else { return };
    let f = fixture();
    let texts = f["texts"].as_array().unwrap();
    let ids = f["ids"].as_object().unwrap();
    assert!(ids.len() > 150);
    let mut bad = Vec::new();
    for (k, want) in ids {
        let text = texts[k.parse::<usize>().unwrap()].as_str().unwrap();
        let want: Vec<u32> = want
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_u64().unwrap() as u32)
            .collect();
        if e.token_ids(text) != want {
            bad.push(text.chars().take(80).collect::<String>());
        }
    }
    assert!(
        bad.is_empty(),
        "token ids differ on {} texts: {bad:?}",
        bad.len()
    );
}

#[test]
fn probabilities_match_route_baselines() {
    let Some(e) = embedding() else { return };
    let f = fixture();
    assert_eq!(e.theta, f["theta"].as_f64().unwrap());
    assert_eq!(e.temperature, f["temperature"].as_f64().unwrap());
    let texts: Vec<&str> = f["texts"]
        .as_array()
        .unwrap()
        .iter()
        .map(|t| t.as_str().unwrap())
        .collect();
    let slates = f["slates"].as_array().unwrap();
    assert_eq!(slates.len(), 40);
    let mut worst = 0f64;
    for s in slates {
        let req = texts[s["request"].as_u64().unwrap() as usize];
        let tools: Vec<&str> = s["tools"]
            .as_array()
            .unwrap()
            .iter()
            .map(|i| texts[i.as_u64().unwrap() as usize])
            .collect();
        let want: Vec<f64> = s["probs"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_f64().unwrap())
            .collect();
        let got = e.probs(req, &tools);
        assert_eq!(got.len(), want.len());
        for (a, b) in got.iter().zip(&want) {
            worst = worst.max((a - b).abs());
        }
    }
    assert!(worst <= 1e-5, "largest gap {worst:e}");
}
