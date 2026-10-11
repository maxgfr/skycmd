//! Golden test against the Python Laya runtime on the real oscar model.
//!
//! The reference outputs in tests/golden/oscar.json were produced once by
//! tests/golden/make_golden.py (no Python runs here). The model files live in the gitignored
//! models-src/oscar directory: the tests are skipped with a message when it is absent.

use std::path::PathBuf;
use std::sync::OnceLock;

use sky_gguf::{GgmlType, Gguf};
use sky_infer::convert::{ConvertOptions, LayaDir};
use sky_infer::json::Json;
use sky_infer::kernels::ActQuant;
use sky_infer::{Engine, EngineOptions, QType, QuantizeSpec};

struct Case {
    name: String,
    qtype: QType,
    instructions: String,
    options: Vec<String>,
    state: String,
    ids: Vec<u32>,
    markers: Vec<usize>,
    probs: Vec<f32>,
    question_json: String,
    state_json: String,
}

fn oscar_dir() -> Option<PathBuf> {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../models-src/oscar");
    if p.join("model.safetensors").exists() {
        Some(p)
    } else {
        eprintln!("skipping golden test: {} is absent", p.display());
        None
    }
}

fn to_json_text(v: &Json) -> String {
    match v {
        Json::Null => "null".into(),
        Json::Bool(b) => b.to_string(),
        Json::Number(n) => {
            if n.fract() == 0.0 && n.abs() < 1e15 {
                format!("{}", *n as i64)
            } else {
                format!("{n}")
            }
        }
        Json::String(s) => sky_infer::json::escape(s),
        Json::Array(a) => format!(
            "[{}]",
            a.iter().map(to_json_text).collect::<Vec<_>>().join(",")
        ),
        Json::Object(o) => format!(
            "{{{}}}",
            o.iter()
                .map(|(k, v)| format!("{}:{}", sky_infer::json::escape(k), to_json_text(v)))
                .collect::<Vec<_>>()
                .join(",")
        ),
    }
}

fn cases() -> &'static Vec<Case> {
    static CASES: OnceLock<Vec<Case>> = OnceLock::new();
    CASES.get_or_init(|| {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/golden/oscar.json");
        let j = Json::parse(&std::fs::read_to_string(path).unwrap()).unwrap();
        let nums = |v: &Json| -> Vec<f64> {
            v.as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_f64().unwrap())
                .collect()
        };
        j.get("cases")
            .unwrap()
            .as_array()
            .unwrap()
            .iter()
            .map(|c| Case {
                name: c.get("name").unwrap().as_str().unwrap().into(),
                qtype: QType::from_name(c.get("qtype").unwrap().as_str().unwrap()).unwrap(),
                instructions: c.get("instructions_text").unwrap().as_str().unwrap().into(),
                options: c
                    .get("option_texts")
                    .unwrap()
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|o| o.as_str().unwrap().into())
                    .collect(),
                state: c.get("state_text").unwrap().as_str().unwrap().into(),
                ids: nums(c.get("ids").unwrap())
                    .into_iter()
                    .map(|v| v as u32)
                    .collect(),
                markers: nums(c.get("markers").unwrap())
                    .into_iter()
                    .map(|v| v as usize)
                    .collect(),
                probs: nums(c.get("probs").unwrap())
                    .into_iter()
                    .map(|v| v as f32)
                    .collect(),
                question_json: to_json_text(c.get("question").unwrap()),
                state_json: to_json_text(c.get("state").unwrap()),
            })
            .collect()
    })
}

fn laya() -> &'static LayaDir {
    static DIR: OnceLock<LayaDir> = OnceLock::new();
    DIR.get_or_init(|| LayaDir::open(&oscar_dir().unwrap()).unwrap())
}

fn engine_from_dir(opts: EngineOptions) -> Engine {
    let (model, tok) = laya().load().unwrap();
    Engine::new(model, tok, opts).unwrap()
}

/// Compare token ids, markers and probabilities of every case; returns the max abs error.
fn check(engine: &mut Engine, tol: f32, label: &str) -> f32 {
    let mut worst = 0.0f32;
    let mut sum = 0.0f32;
    let mut failures = Vec::new();
    for c in cases() {
        let opts: Vec<&str> = c.options.iter().map(|s| s.as_str()).collect();
        let (ids, markers) = engine.build_sequence(c.qtype, &c.instructions, &opts, &c.state);
        assert_eq!(ids, c.ids, "{label}/{}: token ids differ", c.name);
        assert_eq!(markers, c.markers, "{label}/{}: markers differ", c.name);
        let p = engine
            .decide_raw(c.qtype, &c.instructions, &c.options, &c.state)
            .unwrap();
        assert_eq!(p.len(), c.probs.len());
        let err = p
            .iter()
            .zip(&c.probs)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max);
        if err > tol {
            failures.push(format!(
                "{label}/{}: max |dp| = {err:.4} > {tol}\n  got  {p:?}\n  want {:?}",
                c.name, c.probs
            ));
        }
        worst = worst.max(err);
        sum += err;
    }
    eprintln!(
        "{label}: max |dp| = {worst:.2e}, mean {:.2e} over {} cases",
        sum / cases().len() as f32,
        cases().len()
    );
    assert!(failures.is_empty(), "{}", failures.join("\n"));
    worst
}

#[test]
fn golden_f32_safetensors() {
    if oscar_dir().is_none() {
        return;
    }
    let mut e = engine_from_dir(EngineOptions::default());
    assert_eq!(e.model().cfg.num_layers, 7);
    assert_eq!(e.model().cfg.head_layers, 2);
    check(&mut e, 1e-3, "f32");
    // second pass hits the prefix cache and the BPE cache: identical results
    check(&mut e, 1e-3, "f32 cached");
}

const Q8_ALL: QuantizeSpec = QuantizeSpec {
    encoder: true,
    head: true,
    embeddings: false,
};

/// The Q8_0 path: Q8_0 weights for every matrix product of the encoder and the head, Q8_0
/// activations (with the residual pass, the default), i8 x i8 -> i32 products.
#[test]
fn golden_q8_safetensors() {
    if oscar_dir().is_none() {
        return;
    }
    let mut e = engine_from_dir(EngineOptions {
        quantize: Some(Q8_ALL),
        ..Default::default()
    });
    assert!(e.model_info().q8);
    check(&mut e, 2e-2, "q8 (all matrices)");
}

/// Single-pass Q8_0 activations (fastest mode). Each rounding of an activation block costs ~1% of
/// its values here and oscar is sensitive to it: this measures ~0.025 at worst (bf16 autocast in
/// the reference runtime itself moves the same cases by up to 0.009). The bound guards the kernels
/// against regressions; the 2e-2 target is met by the default mode above.
#[test]
fn golden_q8_fast_activations() {
    if oscar_dir().is_none() {
        return;
    }
    let mut e = engine_from_dir(EngineOptions {
        quantize: Some(Q8_ALL),
        act_quant: ActQuant::Q8,
        ..Default::default()
    });
    check(&mut e, 4e-2, "q8 single-pass activations");
}

#[test]
fn golden_gguf_conversion() {
    if oscar_dir().is_none() {
        return;
    }
    // A Q8_0 file holds plain llama.cpp Q8_0 tensors: the massive input channels share their block
    // scale with the other weights (the outlier split is only possible when quantizing at load),
    // which measures ~0.022 at worst here; see `golden_gguf_f16_quantized_at_load` for the 2e-2 path.
    for (ftype, tol) in [
        (GgmlType::F32, 1e-3),
        (GgmlType::F16, 1e-3),
        (GgmlType::Q8_0, 3e-2),
    ] {
        let opts = ConvertOptions {
            ftype,
            ..Default::default()
        };
        let bytes = laya().sources().to_gguf(&opts).unwrap();
        let g = Gguf::parse(&bytes).unwrap();
        assert_eq!(g.architecture(), Some("modern-bert"));
        assert_eq!(g.get_u64("modern-bert.block_count"), Some(9));
        assert_eq!(g.get_u64("modern-bert.decision.block_count"), Some(2));
        assert_eq!(g.get_u64("modern-bert.decision.max_head_tokens"), Some(192));
        assert_eq!(g.get_str("tokenizer.ggml.model"), Some("gpt2"));
        assert_eq!(g.tensor("blk.7.attn_qkv.bias").unwrap().dims, vec![768]);
        assert_eq!(
            g.tensor("blk.8.ffn_up.weight").unwrap().dims,
            vec![256, 1024]
        );
        assert_eq!(g.tensor("token_types.weight").unwrap().dims, vec![256, 3]);
        assert_eq!(g.tensor("cls.output.weight").unwrap().dims, vec![256, 1]);
        assert!(g.tensor("blk.0.attn_norm.weight").is_none());
        assert!(g.tensor("act_head.0.weight").is_none());
        assert_eq!(g.tensor("blk.0.attn_qkv.weight").unwrap().dtype, ftype);
        assert_eq!(
            g.tensor("blk.7.attn_qkv.weight").unwrap().dtype,
            GgmlType::F32
        );
        let emb_type = if ftype == GgmlType::Q8_0 {
            GgmlType::F16
        } else {
            ftype
        };
        assert_eq!(g.tensor("token_embd.weight").unwrap().dtype, emb_type);
        let mut e = Engine::from_gguf(&bytes).unwrap();
        assert_eq!(e.model().cfg.max_len, 512);
        check(&mut e, tol, &format!("gguf {}", ftype.name()));
    }
}

/// Recommended deployment of a Q8_0 model: ship the F16 GGUF and quantize when loading.
#[test]
fn golden_gguf_f16_quantized_at_load() {
    if oscar_dir().is_none() {
        return;
    }
    let bytes = laya()
        .sources()
        .to_gguf(&ConvertOptions {
            ftype: GgmlType::F16,
            ..Default::default()
        })
        .unwrap();
    let mut e = Engine::from_gguf_with(
        &bytes,
        EngineOptions {
            quantize: Some(Q8_ALL),
            ..Default::default()
        },
    )
    .unwrap();
    check(&mut e, 2e-2, "gguf F16 -> q8 at load");
}

#[test]
fn golden_decide_json() {
    if oscar_dir().is_none() {
        return;
    }
    let mut e = engine_from_dir(EngineOptions::default());
    for c in cases() {
        let req = format!(
            r#"{{"state": {}, "questions": {{"q": {}}}}}"#,
            c.state_json, c.question_json
        );
        let resp = e.decide_json(&req).unwrap();
        let j = Json::parse(&resp).unwrap();
        let a = j.get("answers").unwrap().get("q").unwrap();
        assert_eq!(
            a.get("type").unwrap().as_str(),
            Some(c.qtype.name()),
            "{}",
            c.name
        );
        let probs: Vec<f32> = match c.qtype {
            QType::Noul => vec![
                1.0 - a.get("noul").unwrap().as_f64().unwrap() as f32,
                a.get("noul").unwrap().as_f64().unwrap() as f32,
            ],
            _ => a
                .get("probabilities")
                .unwrap()
                .as_object()
                .unwrap()
                .iter()
                .map(|(_, v)| v.as_f64().unwrap() as f32)
                .collect(),
        };
        for (p, w) in probs.iter().zip(&c.probs) {
            assert!(
                (p - w).abs() < 1e-3,
                "{}: {probs:?} vs {:?}",
                c.name,
                c.probs
            );
        }
        if c.qtype == QType::Choice {
            let best = c
                .probs
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .unwrap()
                .0;
            let keys: Vec<&String> = a
                .get("probabilities")
                .unwrap()
                .as_object()
                .unwrap()
                .iter()
                .map(|(k, _)| k)
                .collect();
            assert_eq!(
                a.get("choice").unwrap().as_str(),
                Some(keys[best].as_str()),
                "{}",
                c.name
            );
        }
    }
}
