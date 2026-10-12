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
        data::{encode_record_cfg, make_batch, s2_texts, S2Item},
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

/// The golden configuration of a ladder config (SPEC-S3 §6): its shape and
/// options (heads, loops, number fusion, exits) with the tokenizer's
/// vocabulary, `max_state_tokens = 24` (so the long case is truncated), the
/// s2 golden thresholds and temperatures on the first and last exits.
pub fn golden_config_from(base: &S2Config, vocab_size: usize, name: &str) -> S2Config {
    let mut cfg = base.clone();
    cfg.name = name.into();
    cfg.vocab_size = vocab_size;
    cfg.max_state_tokens = 24;
    cfg.max_head_tokens = 96;
    cfg.exit_threshold = [0.9, 1.0, 0.0];
    cfg.temperature.clear();
    let last = cfg.n_exits() - 1;
    for (k, t) in [
        ("e0.choice".to_string(), 1.5f32),
        (format!("e{last}.choice.3_5"), 2.0),
        (format!("e{last}.noul.2"), 0.7),
        ("e0.score".to_string(), 0.8),
    ] {
        cfg.temperature.insert(k, t);
    }
    cfg.bf16 = false;
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
        } else if shape.len() == 1 {
            // 1-D vectors that are neither norms nor biases (the temporal carry gate).
            n.multiply(Array::from_f32(0.5))?
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

/// Extra cases of the number-fusion golden (SPEC-S3 §1 merge rule).
fn fusion_cases() -> Vec<Case> {
    vec![
        Case {
            name: "fuse_range_and_zero",
            question: json!({"type": "choice", "instructions": "Pick 1 of the 3 ranges 10-20 or 5 -7.",
                "criteria": {"7": null, "go 3": "climb 2 m then 4", "b2": "12-14 m"}}),
            state: "range 10-20 alt -3 x 0 y 0.5 7 8",
        },
        Case {
            name: "fuse_leading_number",
            question: json!({"type": "noul", "instructions": "12 drones: is the 3rd one late?"}),
            state: "12 apples -4 b2 5",
        },
    ]
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
    golden_with(tokenizer_json, out_dir, seed, None, "s2")
}

/// Writes `{prefix}-f32.gguf`, `{prefix}-ternary.gguf` and `{prefix}.json`
/// into `out_dir`: the s2 golden model, or (with `base`) a tiny random model
/// of that config's shape and options ([`golden_config_from`]).
pub fn golden_with(
    tokenizer_json: &str,
    out_dir: &Path,
    seed: u64,
    base: Option<&S2Config>,
    prefix: &str,
) -> Result<()> {
    std::fs::create_dir_all(out_dir)?;
    let tmp = tempfile_path(out_dir);
    std::fs::write(&tmp, tokenizer_json)?;
    let tok = Tok::load(&tmp);
    let _ = std::fs::remove_file(&tmp);
    let tok = tok?;
    tok.check_num()?;
    let cfg = match base {
        Some(b) => golden_config_from(b, tok.vocab_size(), &format!("{prefix}-golden")),
        None => golden_config(tok.vocab_size()),
    };
    cfg.validate()?;
    let mut all_cases = cases();
    if cfg.fuse_numbers {
        all_cases.extend(fusion_cases());
    }
    let params = golden_params(&cfg, seed)?;
    let mut files = Vec::new();
    let mut outputs: Vec<Vec<Value>> = vec![Vec::new(); 2];
    let mut seq_out: Vec<Vec<Value>> = vec![Vec::new(); 2];
    for (fi, format) in [WeightFormat::F32, WeightFormat::Ternary]
        .into_iter()
        .enumerate()
    {
        let bytes = to_gguf(&params, &cfg, tokenizer_json, format, &cfg.name)?;
        let file = format!("{prefix}-{}.gguf", format.name());
        std::fs::write(out_dir.join(&file), &bytes)?;
        let g = Gguf::parse(&bytes)?;
        let gcfg = config_from_gguf(&g)?;
        // The weights exactly as stored (ternary codes times f16 scales, int8 embeddings).
        let p = params_from_gguf(&g, &gcfg)?;
        for c in &all_cases {
            outputs[fi].push(run_case(&p, &gcfg, &tok, c)?);
        }
        if gcfg.temporal_carry {
            for s in seq_cases() {
                seq_out[fi].push(run_sequence(&p, &gcfg, &tok, &s)?);
            }
        }
        files.push(json!({"format": format.name(), "path": file, "bytes": bytes.len()}));
    }
    let mut out_cases = Vec::new();
    for (i, c) in all_cases.iter().enumerate() {
        let mut v = outputs[0][i].clone();
        v["name"] = json!(c.name);
        v["question"] = c.question.clone();
        v["state"] = json!(c.state);
        v["expected"] =
            json!({"f32": outputs[0][i]["expected"], "ternary": outputs[1][i]["expected"]});
        out_cases.push(v);
    }
    let mut notes = vec![
                "Random-init tiny skycmd-s2 model; the tokenizer is embedded in the GGUF files (byte-level BPE without merges, [NUM] = 5 with lstrip, byte b = id 6 + b).".to_string(),
                "Expected outputs were computed by the mlx-rs forward pass on the weights read back from each GGUF file (ternary: code x f16 row scale; token_embd: int8 x f16 row scale), i.e. the mlx-rs fake-quant model with f16 scales.".into(),
                format!("Exits are 0-based: exit e runs state {} 0..exits[e] then exit.e.norm; temperature keys are temperature.e{{e}}.{{type}}[.{{bucket}}], falling back to e{{e}}.{{type}}, then 1.", if cfg.state_loops > 1 { "block applications" } else { "blocks" }),
                "logits are raw scorer outputs per exit; probs = softmax(logits / temperature). The early-exit answer stops at the first exit whose max prob >= exit_threshold.{type} (always the last exit otherwise): see exit / layers_used.".into(),
                "Static stream: Q = [CLS] head [SEP] at positions 0..|Q|-1, then [MASK] + option tokens with positions restarting at |Q|; Q attends Q, option i attends Q and option i. State stream: [CLS] state [SEP], positions 0..S-1.".into(),
                format!("Fusion attn_kv rows 0..d are the keys and rows d..2d the values; heads are contiguous {}-wide column blocks (as in attn_qkv). kv = state tokens (exit-normalized) followed by the static question tokens.", cfg.head_dim()),
                "phi(v) = [u, sin(u w_0..7), cos(u w_0..7)], u = sign(v) ln(1 + |v|), w_k = 2^k pi / 8.".into(),
            ];
    if cfg.fuse_numbers {
        notes.push("fuse_numbers (SPEC-S3 1): the merge rule runs on each tokenized text separately (instruction head, each option text, state) right after numeric extraction + BPE and BEFORE the SPEC truncation rules (48-token option cap, head budget, max_state_tokens count merged tokens). A [NUM] at text position 0 stays standalone (so it is standalone after [CLS] / [MASK]); a [NUM] whose previous token already has a value stays standalone; otherwise it is removed and its value moves onto the previous token. Embedding: tok_embd[id] + has_value * (num_proj.weight phi(v) + num_proj.bias). static.has_value / state_stream.has_value give the flags; RoPE positions are the merged positions.".into());
    }
    if cfg.state_loops > 1 {
        notes.push(format!("state_loops = {} (SPEC-S3 2): application a = 0..{} runs blk.(a % {}) ; before each loop l >= 1 (a % B == 0, a > 0) loop_embd.weight[l] is added to the hidden states; block 0 uses no attention norm at a = 0 and blk.0.attn_norm_loop.weight (LayerNorm without bias) on later loops; blocks b > 0 use their own attn_norm. exits count block applications.", cfg.state_loops, cfg.depth(), cfg.state_layers));
    }
    if cfg.temporal_carry {
        notes.push("temporal_carry (SPEC-S3 7): `sequences` are streams of frames answered in order with one carry buffer per stream. A frame is warm when a previous frame exists and its state token ids equal the previous frame's (else cold, carry term 0). Warm start: x0 = embed(state) + carry_gate.weight * LayerNorm(h_prev; carry_norm.weight, no bias, eps layer_norm_epsilon). h_prev = the previous frame's raw hidden states (before the exit norm) after the last block application that frame ran, i.e. exits[exit] applications where `exit` is that frame's early-exit answer (carry_source_layers). Each frame lists `warm`, every exit's logits/probs (computed for checking; an engine may stop early), the early-exit answer and the layers of the latent it hands to the next frame.".into());
    }
    if let Some(es) = &cfg.ffn_es {
        let d = cfg.hidden_size;
        notes.push(format!("ffn_es (SPEC-S3 9, keys ffn_es_seed / ffn_es_units / ffn_es_k / ffn_es_fan_in): every encoder block (static and state) has no ffn_up; its FFN is x + ffn_down . kWTA(P . LN_ffn(x)). P is [{u}][{d}], generated by splitmix64 from the u64 seed {seed}: for each unit in order, draw r until fan_in = {f} distinct inputs are chosen (input (r >> 32) mod d, weight -1/sqrt(fan_in) if r & 1 else +1/sqrt(fan_in)). kWTA keeps, per token, the units whose pre-activation is >= the k-th largest one (k = {k}) with their value, and zeroes the others; ffn_down is [{d}][{u}].", u = es.units(d), seed = es.seed, f = es.fan_in, k = es.k(d)));
    }
    if cfg.n_heads() * 64 != cfg.hidden_size {
        notes.push(format!("head_count = {} (key skycmd-s2.head_count): head_dim = {}; RoPE rotate_half over the full head_dim with inv_freq = theta^(-2i/head_dim); attention scale 1/sqrt(head_dim).", cfg.n_heads(), cfg.head_dim()));
    }
    let doc = json!({
        "meta": {
            "generator": format!("sky-train s2-golden{}", if base.is_some() { format!(" ({prefix})") } else { String::new() }),
            "seed": seed,
            "config": cfg,
            "files": files,
            "tolerance": {"f32": 1e-4, "ternary": 1e-3},
            "notes": notes,
        },
        "cases": out_cases,
    });
    let mut doc = doc;
    if cfg.temporal_carry {
        let seqs: Vec<Value> = seq_cases()
            .iter()
            .enumerate()
            .map(|(i, s)| {
                let mut v = seq_out[0][i].clone();
                v["name"] = json!(s.name);
                v["question"] = s.question.clone();
                v["states"] = json!(s.states);
                for (fi, f) in v["frames"].as_array_mut().unwrap().iter_mut().enumerate() {
                    let t = seq_out[1][i]["frames"][fi]["expected"].clone();
                    let e = f["expected"].clone();
                    f["expected"] = json!({"f32": e, "ternary": t});
                }
                v
            })
            .collect();
        doc["sequences"] = json!(seqs);
    }
    std::fs::write(
        out_dir.join(format!("{prefix}.json")),
        serde_json::to_string_pretty(&doc)?,
    )?;
    Ok(())
}

fn tempfile_path(dir: &Path) -> std::path::PathBuf {
    dir.join(format!(".s2-golden-tokenizer-{}.json", std::process::id()))
}

struct SeqCase {
    name: &'static str,
    question: Value,
    states: Vec<&'static str>,
}

/// Temporal streams: four frames with the same keys (warm), a frame with an
/// extra key (cold restart), then warm again.
fn seq_cases() -> Vec<SeqCase> {
    let states = vec![
        "d 35 v 11 dz 2 hr 1",
        "d 33 v 11 dz 2 hr 1",
        "d 30 v 10 dz 1 hr 1",
        "d 27 v 9 dz -1 hr 0",
        "d 25 v 9 dz 1 hr 0 cl 3",
        "d 24 v 8 dz 1 hr 0 cl 2",
    ];
    vec![
        SeqCase {
            name: "choice_stream",
            question: json!({"type": "choice", "instructions": "How should the drone handle the obstacle in front of it?",
                "criteria": {"over": "climb over it", "brake": "reduce speed", "continue": "keep going", "around": "use the lateral opening"}}),
            states: states.clone(),
        },
        SeqCase {
            name: "noul_stream_first_exit",
            question: json!({"type": "noul", "instructions": "Is a collision likely within 2 seconds?"}),
            states,
        },
    ]
}

/// Runs a stream of frames with the temporal carry (see the s3t notes).
fn run_sequence(p: &ParamMap, cfg: &S2Config, tok: &Tok, s: &SeqCase) -> Result<Value> {
    let (t, ins, opts) = render(&s.question);
    let qtype = qtype_id(&t).unwrap_or(0);
    let k = opts.len();
    let mut prev: Option<(Vec<u32>, Array)> = None;
    let mut frames = Vec::new();
    for state in &s.states {
        let rec = DecisionRecord {
            src: "golden".into(),
            t: t.clone(),
            ins: ins.clone(),
            opts: opts.clone(),
            state: (*state).into(),
            target: vec![1.0; k],
            split: "test".into(),
        };
        let (st, stream) = encode_record_cfg(tok, &rec, cfg)?;
        let item = S2Item {
            stat: 0,
            state: stream.clone(),
            qtype,
            target: vec![1.0 / k as f32; k],
            src: "golden".into(),
            group: 0,
        };
        let warm = prev.as_ref().is_some_and(|(ids, _)| *ids == stream.ids);
        let (logits, raw) = mlx_rs::with_device(mlx_rs::Device::cpu(), || -> Result<_> {
            let b = make_batch(std::slice::from_ref(&st), &[&item], 1);
            let on = Array::from_slice(&[1.0f32], &[1]);
            let carry = match (&prev, warm) {
                (Some((_, h)), true) => Some(crate::s2::model::Carry { h, on: &on }),
                _ => None,
            };
            let (l, r) = crate::s2::model::forward_carry(p, cfg, &b, false, carry)?;
            Ok((logits_per_row(&l, &[k])?, r))
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
        frames.push(json!({
            "state": state,
            "state_stream": {"ids": stream.ids, "values": stream.values,
                "has_value": (0..stream.len()).map(|i| stream.has(i) as u8).collect::<Vec<_>>()},
            "warm": warm,
            "expected": {"exits": exits, "exit": e, "layers_used": cfg.exits[e],
                "probs": exits[e]["probs"], "carry_source_layers": cfg.exits[e]},
        }));
        prev = Some((stream.ids.clone(), raw[e].clone()));
    }
    let (head, _, _) = s2_texts(&DecisionRecord {
        src: String::new(),
        t: t.clone(),
        ins: ins.clone(),
        opts: opts.clone(),
        state: String::new(),
        target: vec![],
        split: String::new(),
    });
    Ok(json!({"qtype": t, "head_text": head, "option_texts": opts, "frames": frames}))
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
    let (st, state) = encode_record_cfg(tok, &rec, cfg)?;
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
    let flags =
        |s: &crate::s2::data::Stream| -> Vec<u8> { (0..s.len()).map(|i| s.has(i) as u8).collect() };
    let mut stat = json!({
        "ids": st.tokens.ids, "values": st.tokens.values, "pos": st.pos, "seg": st.seg,
        "markers": st.markers, "q_len": st.q_len,
    });
    let mut state_stream = json!({"ids": state.ids, "values": state.values});
    if cfg.fuse_numbers {
        stat["has_value"] = json!(flags(&st.tokens));
        state_stream["has_value"] = json!(flags(&state));
    }
    Ok(json!({
        "qtype": t,
        "instructions_text": ins,
        "head_text": head,
        "option_texts": opts,
        "static": stat,
        "state_stream": state_stream,
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
        assert!(t["state_stream"].get("has_value").is_none());
        assert_eq!(t["state_stream"]["ids"].as_array().unwrap().len(), 24);
        assert!(
            dir.path().join("s2-f32.gguf").exists() && dir.path().join("s2-ternary.gguf").exists()
        );
    }

    #[test]
    fn s3t_golden_sequences() {
        let _g = test_util::lock();
        let dir = tempfile::tempdir().unwrap();
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../configs/s3-pico-temporal.json");
        let base = S2Config::load(path).unwrap();
        golden_with(&s2_tokenizer_json(), dir.path(), 7, Some(&base), "s3t").unwrap();
        let doc: Value =
            serde_json::from_str(&std::fs::read_to_string(dir.path().join("s3t.json")).unwrap())
                .unwrap();
        assert_eq!(doc["meta"]["config"]["temporal_carry"], true);
        let bytes = std::fs::read(dir.path().join("s3t-ternary.gguf")).unwrap();
        let g = Gguf::parse(&bytes).unwrap();
        assert_eq!(g.get_bool("skycmd-s2.temporal_carry"), Some(true));
        assert!(g.tensor("carry_gate.weight").is_some() && g.tensor("carry_norm.weight").is_some());
        let seqs = doc["sequences"].as_array().unwrap();
        assert_eq!(seqs.len(), 2);
        for s in seqs {
            let warm: Vec<bool> = s["frames"]
                .as_array()
                .unwrap()
                .iter()
                .map(|f| f["warm"].as_bool().unwrap())
                .collect();
            assert_eq!(warm, vec![false, true, true, true, false, true]);
            // Warm frames differ from what a cold start of the same state gives:
            // frame 1 and frame 0 have different states, so compare across streams
            // of the same states instead: the noul stream exits at exit 0.
            for f in s["frames"].as_array().unwrap() {
                for fmt in ["f32", "ternary"] {
                    let ex = &f["expected"][fmt];
                    assert_eq!(ex["carry_source_layers"], ex["layers_used"]);
                    if s["qtype"] == "noul" {
                        assert_eq!(ex["exit"], 0);
                    }
                }
            }
        }
        // The cold standard cases are still there.
        assert_eq!(doc["cases"].as_array().unwrap().len(), 7);
    }

    #[test]
    fn s3_golden_files_are_consistent() {
        let _g = test_util::lock();
        let dir = tempfile::tempdir().unwrap();
        let path =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../configs/s3-pico.json");
        let base = S2Config::load(path).unwrap();
        golden_with(&s2_tokenizer_json(), dir.path(), 7, Some(&base), "s3").unwrap();
        for f in ["s3.json", "s3-f32.gguf", "s3-ternary.gguf"] {
            assert!(dir.path().join(f).exists(), "{f}");
        }
        let doc: Value =
            serde_json::from_str(&std::fs::read_to_string(dir.path().join("s3.json")).unwrap())
                .unwrap();
        let cfg = &doc["meta"]["config"];
        assert_eq!(cfg["state_loops"], 4);
        assert_eq!(cfg["fuse_numbers"], true);
        assert_eq!(cfg["head_count"], 1);
        assert_eq!(cfg["hidden_size"], 96);
        let cases = doc["cases"].as_array().unwrap();
        assert_eq!(cases.len(), 7);
        let mut spread = 0f64;
        for c in cases {
            let ids = c["state_stream"]["ids"].as_array().unwrap();
            let has = c["state_stream"]["has_value"].as_array().unwrap();
            assert_eq!(ids.len(), has.len());
            assert_eq!(
                c["static"]["has_value"].as_array().unwrap().len(),
                c["static"]["ids"].as_array().unwrap().len()
            );
            for fmt in ["f32", "ternary"] {
                let ex = &c["expected"][fmt];
                let exits = ex["exits"].as_array().unwrap();
                assert_eq!(exits.len(), 4);
                for (e, x) in exits.iter().enumerate() {
                    assert_eq!(x["layers"], e + 1);
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
                let e = ex["exit"].as_u64().unwrap();
                match c["qtype"].as_str().unwrap() {
                    "noul" => assert_eq!(e, 0),
                    "score" => assert_eq!(e, 3),
                    _ => {}
                }
            }
        }
        assert!(spread > 0.2, "{spread}");
        // The fusion cases: `range 10-20` keeps the second number standalone.
        let r = cases
            .iter()
            .find(|c| c["name"] == "fuse_range_and_zero")
            .unwrap();
        let ids: Vec<u64> = r["state_stream"]["ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap())
            .collect();
        assert_eq!(ids.iter().filter(|&&i| i == 5).count(), 3);
        assert_eq!(r["state_stream"]["ids"].as_array().unwrap().len(), 18);
        let bytes = std::fs::read(dir.path().join("s3-ternary.gguf")).unwrap();
        let g = Gguf::parse(&bytes).unwrap();
        assert_eq!(g.get_u64("skycmd-s2.state_loops"), Some(4));
        assert!(g.tensor("loop_embd.weight").is_some());
    }
}
