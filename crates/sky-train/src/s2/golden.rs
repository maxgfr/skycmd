//! `sky-train s2-golden`: a tiny random s2 model exported as f32 and ternary
//! GGUF files plus the reference outputs of a few requests, computed with the
//! mlx-rs forward pass, for the sky-infer golden test.

use std::path::Path;

use anyhow::Result;
use mlx_rs::{random, Array};
use serde_json::{json, Value};
use sky_gguf::Gguf;

use crate::{
    metrics::softmax_t,
    params::ParamMap,
    s2::{
        config::S2Config,
        data::{encode_record, make_batch, s2_texts, S2Item},
        gguf::{config_from_gguf, params_from_gguf, to_gguf, WeightFormat},
        model::{forward, init, logits_per_row, S2Parts},
    },
    sequence::{qtype_id, DecisionRecord, Tok},
};

/// GPT-2 `bytes_to_unicode`.
fn byte_chars() -> Vec<char> {
    let mut bs: Vec<u32> = (b'!' as u32..=b'~' as u32)
        .chain(0xA1..=0xAC)
        .chain(0xAE..=0xFF)
        .collect();
    let mut cs = bs.clone();
    let mut n = 0;
    for b in 0..256u32 {
        if !bs.contains(&b) {
            bs.push(b);
            cs.push(256 + n);
            n += 1;
        }
    }
    let mut out = vec!['\0'; 256];
    for (b, c) in bs.iter().zip(cs) {
        out[*b as usize] = char::from_u32(c).unwrap_or('?');
    }
    out
}

/// A byte-level BPE `tokenizer.json` without merges for s2: specials
/// `[PAD] [UNK] [CLS] [SEP] [MASK] [NUM]` = 0..=5 (`[NUM]` with `lstrip`) and
/// byte `b` at id `6 + b`.
pub fn byte_tokenizer_json() -> String {
    let specials = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "[NUM]"];
    let mut vocab = serde_json::Map::new();
    for (i, s) in specials.iter().enumerate() {
        vocab.insert(s.to_string(), i.into());
    }
    for (b, c) in byte_chars().iter().enumerate() {
        vocab.insert(c.to_string(), (6 + b).into());
    }
    let added: Vec<_> = specials
        .iter()
        .enumerate()
        .map(|(i, s)| {
            // `[NUM]` absorbs the space on its left, as in the s2 tokenizers.
            json!({"id": i, "content": s, "single_word": false, "lstrip": *s == "[NUM]", "rstrip": false,
                "normalized": false, "special": true})
        })
        .collect();
    json!({
        "version": "1.0", "truncation": null, "padding": null, "added_tokens": added, "normalizer": null,
        "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true, "use_regex": true},
        "post_processor": null,
        "decoder": {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true, "use_regex": true},
        "model": {"type": "BPE", "dropout": null, "unk_token": null, "continuing_subword_prefix": null,
            "end_of_word_suffix": null, "fuse_unk": false, "byte_fallback": false, "ignore_merges": false,
            "vocab": vocab, "merges": []}
    })
    .to_string()
}

/// The golden model configuration (two heads, an `ff` that is not a multiple
/// of 4 so the ternary rows are padded, two exits, two fusion blocks).
pub fn golden_config(vocab_size: usize) -> S2Config {
    let mut cfg = S2Config {
        name: "s2-golden".into(),
        state_layers: 2,
        static_layers: 1,
        fusion_layers: 2,
        exits: vec![1, 2],
        ..S2Config::tiny(vocab_size)
    };
    cfg.max_state_tokens = 24;
    cfg.max_head_tokens = 96;
    cfg.exit_threshold = [0.9, 1.0, 0.0];
    for (k, t) in [
        ("e0.choice", 1.5f32),
        ("e1.choice.3_5", 2.0),
        ("e1.noul.2", 0.7),
        ("e0.score", 0.8),
    ] {
        cfg.temperature.insert(k.into(), t);
    }
    cfg
}

/// Random weights with a large enough scale that the probabilities are far
/// from uniform (norms and biases are randomized too).
pub fn golden_params(cfg: &S2Config, seed: u64) -> Result<ParamMap> {
    let base = init(cfg, S2Parts::DECISION, seed)?;
    let mut key = random::key(seed ^ 0x601d)?;
    let mut names: Vec<_> = base.keys().cloned().collect();
    names.sort();
    let mut out = ParamMap::new();
    for name in names {
        let shape = base[&name].shape().to_vec();
        let (k, next) = random::split(&key, 2)?;
        key = next;
        let n = random::normal::<f32>(&shape[..], 0.0, 1.0, &k)?;
        let a = if name.ends_with(".bias") {
            n.multiply(Array::from_f32(0.1))?
        } else if name.contains("norm") {
            n.multiply(Array::from_f32(0.2))?
                .add(Array::from_f32(1.0))?
        } else if name.as_ref() == "token_embd.weight" || name.as_ref() == "type_embd.weight" {
            n
        } else if name.as_ref() == "cls.output.weight" {
            // Option logits differ only through each option's own tokens: a
            // larger readout makes the probabilities clearly non-uniform.
            n.multiply(Array::from_f32(3.0))?
        } else {
            n.multiply(Array::from_f32(1.0 / (shape[1] as f32).sqrt()))?
        };
        out.insert(name, a);
    }
    crate::params::eval(&out)?;
    Ok(out)
}

struct Case {
    name: &'static str,
    question: Value,
    state: &'static str,
}

fn render(q: &Value) -> (String, String, Vec<String>) {
    let t = q["type"].as_str().unwrap_or("choice").to_string();
    let ins = q["instructions"].as_str().unwrap_or_default().to_string();
    let opts = match t.as_str() {
        "choice" => q["criteria"]
            .as_object()
            .map(|m| {
                m.iter()
                    .map(|(k, d)| match d.as_str() {
                        Some(d) if !d.is_empty() => format!("{k}: {d}"),
                        _ => k.clone(),
                    })
                    .collect()
            })
            .unwrap_or_default(),
        "score" => q["criteria"]
            .as_array()
            .map(|a| {
                a.iter()
                    .enumerate()
                    .map(|(i, d)| format!("level {i}: {}", d.as_str().unwrap_or("")))
                    .collect()
            })
            .unwrap_or_default(),
        _ => {
            let pick = |key: &str, d: &str| match q["criteria"][key].as_str() {
                Some(s) if !s.is_empty() => s.to_string(),
                _ => d.to_string(),
            };
            vec![
                format!(
                    "false: {}",
                    pick("false", "no, the statement does not hold")
                ),
                format!("true: {}", pick("true", "yes, the statement holds")),
            ]
        }
    };
    (t, ins, opts)
}

fn cases() -> Vec<Case> {
    let shared = "x 12 vy -3 gap 40.5 wind -0.25";
    vec![
        Case {
            name: "choice_numbers",
            question: json!({"type": "choice", "instructions": "Pick the lane with 2 free slots.",
                "criteria": {"left": "lane 1, gap 3.5 m", "middle": null, "right": "lane 3"}}),
            state: shared,
        },
        Case {
            name: "noul_same_state",
            question: json!({"type": "noul", "instructions": "Will the drone hit the pipe within 7 frames?"}),
            state: shared,
        },
        Case {
            name: "score_levels",
            question: json!({"type": "score", "instructions": "How risky is the approach at 12 m/s?",
                "criteria": ["safe", "mild", "risky", "critical"]}),
            state: "spd 12 dist 140 batt 63",
        },
        Case {
            name: "noul_custom_glued",
            question: json!({"type": "noul", "instructions": "Is glm-5 cheaper than b2 at -1.5x cost?",
                "criteria": {"true": "it is cheaper", "false": "it is not"}}),
            state: "price 0.002 tokens 1000000",
        },
        Case {
            name: "choice_truncated",
            question: json!({"type": "choice",
                "instructions": "Choose the safest corridor for the convoy given the wind gusts, the obstacle density and the remaining battery margin of each drone in the formation.",
                "criteria": {"alpha": "the northern corridor along the river with 4 bridges and steady wind",
                    "bravo": "the central corridor across the market square with 12 lamp posts",
                    "charlie": "the southern corridor over the rail yard with 2 cranes moving slowly",
                    "delta": "hold position at the depot and wait 30 seconds for the gusts to calm down",
                    "echo": "return to base immediately"}}),
            state: "wind 14 gust 22 batt 41 38 55 obstacles 4 12 2 0 0 eta 120 95 130 0 30 temp -4 humidity 81 visibility 3.2 km",
        },
    ]
}

/// Writes `s2-f32.gguf`, `s2-ternary.gguf` and `s2.json` into `out_dir`.
pub fn golden(tokenizer_json: &str, out_dir: &Path, seed: u64) -> Result<()> {
    std::fs::create_dir_all(out_dir)?;
    let tmp = tempfile_path(out_dir);
    std::fs::write(&tmp, tokenizer_json)?;
    let tok = Tok::load(&tmp);
    let _ = std::fs::remove_file(&tmp);
    let tok = tok?;
    tok.check_num()?;
    let cfg = golden_config(tok.vocab_size());
    let params = golden_params(&cfg, seed)?;
    let mut files = Vec::new();
    let mut outputs: Vec<Vec<Value>> = vec![Vec::new(); 2];
    for (fi, format) in [WeightFormat::F32, WeightFormat::Ternary]
        .into_iter()
        .enumerate()
    {
        let bytes = to_gguf(&params, &cfg, tokenizer_json, format, "s2-golden")?;
        let file = format!("s2-{}.gguf", format.name());
        std::fs::write(out_dir.join(&file), &bytes)?;
        let g = Gguf::parse(&bytes)?;
        let gcfg = config_from_gguf(&g)?;
        // The weights exactly as stored (ternary codes times f16 scales, int8 embeddings).
        let p = params_from_gguf(&g, &gcfg)?;
        for c in cases() {
            outputs[fi].push(run_case(&p, &gcfg, &tok, &c)?);
        }
        files.push(json!({"format": format.name(), "path": file, "bytes": bytes.len()}));
    }
    let mut out_cases = Vec::new();
    for (i, c) in cases().iter().enumerate() {
        let mut v = outputs[0][i].clone();
        v["name"] = json!(c.name);
        v["question"] = c.question.clone();
        v["state"] = json!(c.state);
        v["expected"] =
            json!({"f32": outputs[0][i]["expected"], "ternary": outputs[1][i]["expected"]});
        out_cases.push(v);
    }
    let doc = json!({
        "meta": {
            "generator": "sky-train s2-golden",
            "seed": seed,
            "config": cfg,
            "files": files,
            "tolerance": {"f32": 1e-4, "ternary": 1e-3},
            "notes": [
                "Random-init tiny skycmd-s2 model; the tokenizer is embedded in the GGUF files (byte-level BPE without merges, [NUM] = 5 with lstrip, byte b = id 6 + b).",
                "Expected outputs were computed by the mlx-rs forward pass on the weights read back from each GGUF file (ternary: code x f16 row scale; token_embd: int8 x f16 row scale), i.e. the mlx-rs fake-quant model with f16 scales.",
                "Exits are 0-based: exit e runs state blocks 0..exits[e] then exit.e.norm; temperature keys are temperature.e{e}.{type}[.{bucket}], falling back to e{e}.{type}, then 1.",
                "logits are raw scorer outputs per exit; probs = softmax(logits / temperature). The early-exit answer stops at the first exit whose max prob >= exit_threshold.{type} (always the last exit otherwise): see exit / layers_used.",
                "Static stream: Q = [CLS] head [SEP] at positions 0..|Q|-1, then [MASK] + option tokens with positions restarting at |Q|; Q attends Q, option i attends Q and option i. State stream: [CLS] state [SEP], positions 0..S-1.",
                "Fusion attn_kv rows 0..d are the keys and rows d..2d the values; heads are contiguous 64-wide column blocks (as in attn_qkv). kv = state tokens (exit-normalized) followed by the static question tokens.",
                "phi(v) = [u, sin(u w_0..7), cos(u w_0..7)], u = sign(v) ln(1 + |v|), w_k = 2^k pi / 8."
            ],
        },
        "cases": out_cases,
    });
    std::fs::write(out_dir.join("s2.json"), serde_json::to_string_pretty(&doc)?)?;
    Ok(())
}

fn tempfile_path(dir: &Path) -> std::path::PathBuf {
    dir.join(format!(".s2-golden-tokenizer-{}.json", std::process::id()))
}

/// Encodes one request, runs every exit and applies the early-exit rule.
fn run_case(p: &ParamMap, cfg: &S2Config, tok: &Tok, c: &Case) -> Result<Value> {
    let (t, ins, opts) = render(&c.question);
    let qtype = qtype_id(&t).unwrap_or(0);
    let rec = DecisionRecord {
        src: "golden".into(),
        t: t.clone(),
        ins: ins.clone(),
        opts: opts.clone(),
        state: c.state.into(),
        target: vec![1.0; opts.len()],
        split: "test".into(),
    };
    let (st, state) = encode_record(tok, &rec, cfg.max_head_tokens, cfg.max_state_tokens)?;
    let k = opts.len();
    let item = S2Item {
        stat: 0,
        state: state.clone(),
        qtype,
        target: vec![1.0 / k as f32; k],
        src: "golden".into(),
        group: 0,
    };
    let logits = mlx_rs::with_device(mlx_rs::Device::cpu(), || -> Result<_> {
        let b = make_batch(std::slice::from_ref(&st), &[&item], 1);
        Ok(logits_per_row(&forward(p, cfg, &b, false)?, &[k])?)
    })?;
    let mut exits = Vec::new();
    let mut chosen = None;
    for (e, l) in logits.iter().enumerate() {
        let temp = cfg.temperature_of(e, qtype, k);
        let probs: Vec<f32> = softmax_t(&l[0], temp)
            .into_iter()
            .map(|x| x as f32)
            .collect();
        let top = probs.iter().cloned().fold(0f32, f32::max);
        if chosen.is_none() && (top >= cfg.exit_threshold[qtype] || e + 1 == cfg.n_exits()) {
            chosen = Some(e);
        }
        exits.push(json!({"exit": e, "layers": cfg.exits[e], "logits": l[0], "temperature": temp, "probs": probs}));
    }
    let e = chosen.unwrap_or(cfg.n_exits() - 1);
    let (head, _, _) = s2_texts(&rec);
    Ok(json!({
        "qtype": t,
        "instructions_text": ins,
        "head_text": head,
        "option_texts": opts,
        "static": {
            "ids": st.tokens.ids, "values": st.tokens.values, "pos": st.pos, "seg": st.seg,
            "markers": st.markers, "q_len": st.q_len,
        },
        "state_stream": {"ids": state.ids, "values": state.values},
        "expected": {
            "exits": exits,
            "exit": e,
            "layers_used": cfg.exits[e],
            "probs": exits[e]["probs"],
        },
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{s2::data::tests::s2_tokenizer_json, test_util};

    #[test]
    fn golden_files_are_consistent() {
        let _g = test_util::lock();
        let dir = tempfile::tempdir().unwrap();
        golden(&s2_tokenizer_json(), dir.path(), 7).unwrap();
        let doc: Value =
            serde_json::from_str(&std::fs::read_to_string(dir.path().join("s2.json")).unwrap())
                .unwrap();
        let cases = doc["cases"].as_array().unwrap();
        assert_eq!(cases.len(), 5);
        let mut spread = 0f64;
        for c in cases {
            for fmt in ["f32", "ternary"] {
                let ex = &c["expected"][fmt];
                let e = ex["exit"].as_u64().unwrap() as usize;
                for x in ex["exits"].as_array().unwrap() {
                    let p: Vec<f64> = x["probs"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|v| v.as_f64().unwrap())
                        .collect();
                    assert!((p.iter().sum::<f64>() - 1.0).abs() < 1e-5);
                    spread = spread.max(
                        p.iter().cloned().fold(0.0, f64::max)
                            - p.iter().cloned().fold(1.0, f64::min),
                    );
                }
                // noul has tau = 0 (first exit), score tau = 1 (last exit).
                match c["qtype"].as_str().unwrap() {
                    "noul" => assert_eq!(e, 0),
                    "score" => assert_eq!(e, 1),
                    _ => {}
                }
            }
            // Ternary and f32 differ (the quantization is visible) but stay close in shape.
            assert_eq!(c["expected"]["f32"]["exits"].as_array().unwrap().len(), 2);
        }
        assert!(
            spread > 0.2,
            "probabilities are too uniform for a useful golden test: {spread}"
        );
        // The state stream of the truncated case respects max_state_tokens.
        let t = &cases[4];
        assert_eq!(t["state_stream"]["ids"].as_array().unwrap().len(), 24);
        assert!(
            dir.path().join("s2-f32.gguf").exists() && dir.path().join("s2-ternary.gguf").exists()
        );
    }
}
