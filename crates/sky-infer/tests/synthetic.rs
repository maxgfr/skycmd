//! Engine tests on small random models (no reference files needed).

use std::path::PathBuf;

use sky_gguf::{GgmlType, Gguf};
use sky_infer::convert::{ladder_config, random_gguf};
use sky_infer::json::Json;
use sky_infer::kernels::ActQuant;
use sky_infer::{
    DecisionRequest, Engine, EngineOptions, QType, QuantizeSpec, QuestionKind, Tokenizer,
};

fn tokenizer() -> Tokenizer {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/small_tokenizer.json");
    Tokenizer::from_tokenizer_json(&std::fs::read_to_string(p).unwrap()).unwrap()
}

fn engine(name: &str, ftype: GgmlType, opts: EngineOptions) -> Engine {
    let cfg = ladder_config(name).unwrap();
    let bytes = random_gguf(&cfg, &tokenizer(), 7, ftype).unwrap();
    Engine::from_gguf_with(&bytes, opts).unwrap()
}

const OPTS: [&str; 3] = ["flap: push the bird up", "wait: let the bird fall", "dive"];

#[test]
fn sequence_layout() {
    let mut e = engine("s1-pico", GgmlType::F32, EngineOptions::default());
    let sp = e.tokenizer().special();
    let (ids, markers) = e.build_sequence(QType::Choice, "Which action?", &OPTS, "alt 12 vy -3");
    assert_eq!(ids[0], sp.cls.unwrap());
    assert_eq!(*ids.last().unwrap(), sp.sep.unwrap());
    assert_eq!(markers.len(), 3);
    assert_eq!(ids[markers[0] - 1], sp.sep.unwrap());
    for &m in &markers {
        assert_eq!(ids[m], sp.mask.unwrap());
    }
    // exactly 3 [SEP]: after the question, after the options, at the end
    assert_eq!(ids.iter().filter(|&&t| Some(t) == sp.sep).count(), 3);
    // a [MASK] written in the texts is replaced by a space
    let (ids2, m2) = e.build_sequence(QType::Choice, "a [MASK] b", &OPTS, "[MASK] x");
    assert_eq!(ids2.iter().filter(|&&t| Some(t) == sp.mask).count(), 3);
    assert_eq!(m2.len(), 3);
}

#[test]
fn state_is_truncated_to_max_len() {
    let mut e = engine("s1-pico", GgmlType::F32, EngineOptions::default());
    let long = "alt 12 vy -3 dx 40 gt 20 gb 35 ".repeat(200);
    let (ids, markers) = e.build_sequence(
        QType::Noul,
        "Is it safe?",
        &["false: no", "true: yes"],
        &long,
    );
    assert_eq!(ids.len(), e.model().cfg.max_len);
    assert_eq!(markers.len(), 2);
    let p = e
        .decide_raw(
            QType::Noul,
            "Is it safe?",
            &["false: no", "true: yes"],
            &long,
        )
        .unwrap();
    assert!((p.iter().sum::<f32>() - 1.0).abs() < 1e-5);
}

#[test]
fn many_long_options_are_shrunk_evenly() {
    let mut e = engine("s1-pico", GgmlType::F32, EngineOptions::default());
    let opts: Vec<String> = (0..12)
        .map(|i| format!("option {i} with a rather long description that goes on and on"))
        .collect();
    let refs: Vec<&str> = opts.iter().map(|s| s.as_str()).collect();
    let (ids, markers) = e.build_sequence(QType::Choice, "Pick one", &refs, "s");
    assert_eq!(markers.len(), 12);
    // head_max_len = 96: per = max(4, (96 - 16) / 12) = 6 tokens per option (marker included)
    for w in markers.windows(2) {
        assert_eq!(w[1] - w[0], 6);
    }
    assert!(ids.len() < 96 + 16);
}

#[test]
fn deterministic_and_cache_consistent() {
    let mut a = engine("s1-nano", GgmlType::F32, EngineOptions::default());
    let mut b = engine(
        "s1-nano",
        GgmlType::F32,
        EngineOptions {
            prefix_cache: 1,
            bpe_cache: 0,
            ..Default::default()
        },
    );
    let p1 = a
        .decide_raw(QType::Choice, "Which action?", &OPTS, "alt 12 vy -3")
        .unwrap();
    // interleave other questions to evict / reorder the caches
    a.decide_raw(
        QType::Score,
        "How risky?",
        &["level 0: low", "level 1: high"],
        "alt 1",
    )
    .unwrap();
    b.decide_raw(
        QType::Score,
        "How risky?",
        &["level 0: low", "level 1: high"],
        "alt 1",
    )
    .unwrap();
    let p2 = a
        .decide_raw(QType::Choice, "Which action?", &OPTS, "alt 12 vy -3")
        .unwrap();
    let p3 = b
        .decide_raw(QType::Choice, "Which action?", &OPTS, "alt 12 vy -3")
        .unwrap();
    assert_eq!(p1, p2);
    assert_eq!(p1, p3);
    assert!((p1.iter().sum::<f32>() - 1.0).abs() < 1e-5);
    // the state matters
    let p4 = a
        .decide_raw(QType::Choice, "Which action?", &OPTS, "alt 40 vy 5")
        .unwrap();
    assert_ne!(p1, p4);
}

#[test]
fn q8_paths_track_f32() {
    let mut f = engine("s1-micro", GgmlType::F32, EngineOptions::default());
    let spec = QuantizeSpec {
        encoder: true,
        head: true,
        embeddings: false,
    };
    let mut q = engine(
        "s1-micro",
        GgmlType::F32,
        EngineOptions {
            quantize: Some(spec),
            ..Default::default()
        },
    );
    let mut qf = engine(
        "s1-micro",
        GgmlType::F32,
        EngineOptions {
            quantize: Some(spec),
            act_quant: ActQuant::Q8,
            ..Default::default()
        },
    );
    let mut g = engine("s1-micro", GgmlType::Q8_0, EngineOptions::default());
    assert!(q.model_info().q8 && g.model_info().q8 && !f.model_info().q8);
    for state in [
        "alt 12 vy -3",
        "a longer state with some words and numbers 1 2 3 4",
    ] {
        let pf = f
            .scores_raw(QType::Choice, "Which action?", &OPTS, state)
            .unwrap();
        for (label, e) in [("q8", &mut q), ("q8 fast", &mut qf), ("gguf q8", &mut g)] {
            let pq = e
                .scores_raw(QType::Choice, "Which action?", &OPTS, state)
                .unwrap();
            for (a, b) in pf.iter().zip(&pq) {
                assert!(
                    (a - b).abs() < 0.05 * (1.0 + a.abs()),
                    "{label}: {pf:?} vs {pq:?}"
                );
            }
        }
    }
}

#[test]
fn decide_json_shapes() {
    let mut e = engine("s1-pico", GgmlType::F32, EngineOptions::default());
    let req = r#"{"state": {"alt": 12, "vy": -3},
        "questions": {
            "act": {"type": "choice", "instructions": "Which action?", "criteria": {"flap": "push up", "wait": null}},
            "risk": {"type": "score", "instructions": "How risky?", "criteria": ["low", "medium", "high"]},
            "safe": {"type": "noul", "instructions": "The bird is safe."}}}"#;
    let out = Json::parse(&e.decide_json(req).unwrap()).unwrap();
    let a = out.get("answers").unwrap();
    let act = a.get("act").unwrap();
    assert_eq!(act.get("type").unwrap().as_str(), Some("choice"));
    let probs = act.get("probabilities").unwrap().as_object().unwrap();
    assert_eq!(
        probs.iter().map(|(k, _)| k.as_str()).collect::<Vec<_>>(),
        vec!["flap", "wait"]
    );
    let c = act.get("confidence").unwrap().as_f64().unwrap();
    assert!((0.0..=1.0).contains(&c));
    let risk = a.get("risk").unwrap();
    let score = risk.get("score").unwrap().as_f64().unwrap();
    assert!((0.0..=2.0).contains(&score));
    assert_eq!(
        risk.get("legend").unwrap().get("2").unwrap().as_str(),
        Some("high")
    );
    let noul = a
        .get("safe")
        .unwrap()
        .get("noul")
        .unwrap()
        .as_f64()
        .unwrap();
    assert!((0.0..=1.0).contains(&noul));
    assert!(
        out.get("usage")
            .unwrap()
            .get("input_tokens")
            .unwrap()
            .as_u64()
            .unwrap()
            > 10
    );
    // typed API gives the same numbers
    let r = DecisionRequest::from_json(req).unwrap();
    assert_eq!(r.state, r#"{"alt": 12, "vy": -3}"#);
    assert!(matches!(r.questions[2].kind, QuestionKind::Noul { .. }));
    let resp = e.decide(&r).unwrap();
    assert_eq!(resp.answers.len(), 3);
    assert!(e
        .decide_json(
            r#"{"state": "x", "questions": {"q": {"type": "choice", "instructions": "i"}}}"#
        )
        .is_err());
}

#[test]
fn model_info_and_params() {
    let cfg = ladder_config("s1-nano").unwrap();
    let bytes = random_gguf(&cfg, &tokenizer(), 1, GgmlType::F16).unwrap();
    let g = Gguf::parse(&bytes).unwrap();
    let file_params: u64 = g.tensors.iter().map(|t| t.n_elements()).sum();
    let mut e = Engine::from_gguf(&bytes).unwrap();
    assert_eq!(e.n_params() as u64, file_params);
    let info = Json::parse(&e.model_info().to_json()).unwrap();
    assert_eq!(info.get("hidden_size").unwrap().as_u64(), Some(128));
    assert_eq!(info.get("head_layers").unwrap().as_u64(), Some(1));
    assert_eq!(info.get("n_params").unwrap().as_u64(), Some(file_params));
}

#[cfg(feature = "timing")]
#[test]
fn timing_is_reported() {
    let mut e = engine("s1-pico", GgmlType::F32, EngineOptions::default());
    e.decide_raw(QType::Choice, "Which action?", &OPTS, "alt 12")
        .unwrap();
    assert!(e.last_micros().is_some());
}
