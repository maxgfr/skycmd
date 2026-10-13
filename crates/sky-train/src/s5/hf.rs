//! `sky-train import-hf`: a Hugging Face ModernBERT checkpoint (Ettin, mmBERT, the Ettin
//! reranker) as a sky-train checkpoint directory: `model.safetensors` with the `encoder.*` (and
//! `mlm.*` when the source has an MLM head) names, `config.json` (a [`ModelConfig`] whose special
//! tokens come from the tokenizer) and `tokenizer.json`.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::{anyhow, bail, Context, Result};
use mlx_rs::Array;
use serde_json::{Map, Value};

use crate::config::{AgentConfig, ModelConfig, SpecialTokens};
use crate::params::{self, ParamMap, Parts};
use crate::s5::torch;
use crate::sequence::Tok;

#[derive(Debug, Clone, clap::Args)]
pub struct ImportArgs {
    /// HF model directory (config.json, tokenizer.json, model.safetensors or pytorch_model.bin).
    #[arg(long)]
    pub src: PathBuf,
    /// Output checkpoint directory.
    #[arg(long)]
    pub out: PathBuf,
    /// Model name written to config.json (default: the source directory name).
    #[arg(long)]
    pub name: Option<String>,
    /// Longest training sequence.
    #[arg(long, default_value_t = 512)]
    pub max_len: usize,
    /// Token budget of question + options (Laya `head_max_len`).
    #[arg(long, default_value_t = 192)]
    pub head_max_len: usize,
    /// Laya decision-head layers (variant A).
    #[arg(long, default_value_t = 2)]
    pub head_layers: usize,
}

/// sky-train name of a HF ModernBERT tensor (`None`: not imported).
pub fn map_name(hf: &str, has_mlm: bool) -> Option<String> {
    let n = hf.strip_prefix("model.").unwrap_or(hf);
    if let Some(rest) = n.strip_prefix("embeddings.") {
        return Some(format!("encoder.embeddings.{rest}"));
    }
    if let Some(rest) = n.strip_prefix("layers.") {
        return Some(format!("encoder.layers.{rest}"));
    }
    if n == "final_norm.weight" {
        return Some("encoder.final_norm.weight".into());
    }
    if has_mlm {
        match n {
            "head.dense.weight" => return Some("mlm.dense.weight".into()),
            "head.norm.weight" => return Some("mlm.norm.weight".into()),
            "decoder.bias" => return Some("mlm.decoder.bias".into()),
            _ => {}
        }
    }
    None
}

/// Special tokens of a HF tokenizer. Names come from `tokenizer_config.json` when it has them
/// (`cls_token`, `sep_token`, ...); otherwise `[CLS]` / `[SEP]` are the first and last special
/// tokens of the single-sequence template of the post-processor, and the mask, pad and unknown
/// tokens are found by their usual names (`[MASK]` / `<mask>`, ...).
pub fn special_tokens(tok_json: &Value, tok_cfg: &Value) -> Result<SpecialTokens> {
    let added: BTreeMap<String, u32> = tok_json["added_tokens"]
        .as_array()
        .map(|a| {
            a.iter()
                .filter_map(|t| {
                    Some((t["content"].as_str()?.to_string(), t["id"].as_u64()? as u32))
                })
                .collect()
        })
        .unwrap_or_default();
    let vocab = &tok_json["model"]["vocab"];
    let id_of = |name: &str| -> Option<u32> {
        added
            .get(name)
            .copied()
            .or_else(|| vocab[name].as_u64().map(|v| v as u32))
    };
    // A tokenizer_config entry is either a string or {"content": ...}.
    let from_cfg = |k: &str| -> Option<String> {
        let v = &tok_cfg[k];
        v.as_str()
            .or_else(|| v["content"].as_str())
            .map(str::to_string)
    };
    let template: Vec<String> = tok_json["post_processor"]["single"]
        .as_array()
        .map(|a| {
            a.iter()
                .filter_map(|p| p["SpecialToken"]["id"].as_str().map(str::to_string))
                .collect()
        })
        .unwrap_or_default();
    let first_known = |names: &[&str]| -> Option<String> {
        names
            .iter()
            .find(|n| id_of(n).is_some())
            .map(|n| n.to_string())
    };
    let cls = from_cfg("cls_token")
        .or_else(|| template.first().cloned())
        .ok_or_else(|| anyhow!("no [CLS] token"))?;
    let sep = from_cfg("sep_token")
        .or_else(|| (template.len() > 1).then(|| template[template.len() - 1].clone()))
        .ok_or_else(|| anyhow!("no [SEP] token"))?;
    let mask = from_cfg("mask_token")
        .or_else(|| first_known(&["[MASK]", "<mask>"]))
        .ok_or_else(|| anyhow!("no mask token"))?;
    let pad = from_cfg("pad_token")
        .or_else(|| first_known(&["[PAD]", "<pad>"]))
        .ok_or_else(|| anyhow!("no pad token"))?;
    let unk = from_cfg("unk_token")
        .or_else(|| tok_json["model"]["unk_token"].as_str().map(str::to_string))
        .or_else(|| first_known(&["[UNK]", "<unk>"]))
        .ok_or_else(|| anyhow!("no unk token"))?;
    let get = |n: &str| id_of(n).ok_or_else(|| anyhow!("token {n} is not in the vocabulary"));
    Ok(SpecialTokens {
        pad: get(&pad)?,
        unk: get(&unk)?,
        cls: get(&cls)?,
        sep: get(&sep)?,
        mask: get(&mask)?,
        pad_text: pad,
        unk_text: unk,
        cls_text: cls,
        sep_text: sep,
        mask_text: mask,
    })
}

/// RoPE bases `(global, local)` of a HF ModernBERT config: `global_rope_theta` /
/// `local_rope_theta`, or the transformers 5 `rope_parameters` (`full_attention`,
/// `sliding_attention`), else the ModernBERT defaults (160000, then `local_default`).
pub fn rope_thetas(cfg: &Value, local_default: f32) -> (f32, f32) {
    let get = |flat: &str, kind: &str| {
        cfg[flat]
            .as_f64()
            .or_else(|| cfg["rope_parameters"][kind]["rope_theta"].as_f64())
            .map(|v| v as f32)
    };
    (
        get("global_rope_theta", "full_attention").unwrap_or(160_000.0),
        get("local_rope_theta", "sliding_attention").unwrap_or(local_default),
    )
}

/// Reads a JSON file, `Value::Null` when it does not exist.
pub fn read_json_or_null(path: &Path) -> Result<Value> {
    match std::fs::read_to_string(path) {
        Ok(t) => {
            Ok(serde_json::from_str(&t).with_context(|| format!("parsing {}", path.display()))?)
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Value::Null),
        Err(e) => Err(e).with_context(|| format!("reading {}", path.display())),
    }
}

/// Raw tensors of a HF directory: `model.safetensors` if present, else `pytorch_model.bin`.
/// A named tensor: name, shape, f32 data.
type RawTensor = (String, Vec<usize>, Vec<f32>);

fn read_tensors(src: &Path) -> Result<Vec<RawTensor>> {
    let st = src.join("model.safetensors");
    if st.exists() {
        let map = params::load(&st)?;
        let mut out = Vec::new();
        for (k, v) in map {
            let shape = v.shape().iter().map(|&d| d as usize).collect();
            out.push((k.to_string(), shape, crate::model::host_f32(&v)?));
        }
        return Ok(out);
    }
    let bin = src.join("pytorch_model.bin");
    if !bin.exists() {
        bail!(
            "{}: neither model.safetensors nor pytorch_model.bin",
            src.display()
        );
    }
    Ok(torch::read_state_dict(&bin)?
        .into_iter()
        .map(|(k, t)| (k, t.shape, t.data))
        .collect())
}

/// The probe sentences of the MLM check, with `{mask}` where the mask token goes.
const PROBES: [&str; 3] = [
    "The capital of France is{mask}.",
    "La capitale de la France est{mask}.",
    "She drank a cup of hot{mask} before going to bed.",
];

/// Top-5 MLM predictions at the masked position of each probe: a check that weights,
/// tokenizer, RoPE and special ids fit together (a broken import predicts noise).
pub fn mlm_probe(
    cfg: &ModelConfig,
    p: &ParamMap,
    tok_path: &Path,
) -> Result<Vec<(String, Vec<String>)>> {
    let tok = Tok::load_with(tok_path, &cfg.special)?;
    let sp = &cfg.special;
    let mut out = Vec::new();
    for probe in PROBES {
        let (before, after) = probe.split_once("{mask}").expect("probe has a mask");
        let mut ids = vec![sp.cls];
        ids.extend(tok.encode(before)?);
        let pos = ids.len();
        ids.push(sp.mask);
        ids.extend(tok.encode(after)?);
        ids.push(sp.sep);
        let t = ids.len() as i32;
        let arr = Array::from_slice(&ids.iter().map(|&x| x as i32).collect::<Vec<_>>(), &[1, t]);
        let enc = crate::model::encoder(p, cfg, &arr, None)?;
        let logits =
            crate::model::mlm_logits(p, cfg, &enc, &Array::from_slice(&[pos as i32], &[1]))?;
        let v = crate::model::host_f32(&logits)?;
        let mut idx: Vec<usize> = (0..v.len()).collect();
        idx.sort_by(|&a, &b| v[b].total_cmp(&v[a]));
        let top = idx[..5]
            .iter()
            .map(|&i| tok.id_to_token(i as u32).unwrap_or_default())
            .collect();
        out.push((probe.replace("{mask}", " _"), top));
    }
    Ok(out)
}

pub fn import(a: &ImportArgs) -> Result<()> {
    let hf_cfg: Value = serde_json::from_str(
        &std::fs::read_to_string(a.src.join("config.json")).context("reading config.json")?,
    )?;
    let tok_path = a.src.join("tokenizer.json");
    let tok_json: Value = serde_json::from_str(
        &std::fs::read_to_string(&tok_path).context("reading tokenizer.json")?,
    )?;
    let special = special_tokens(
        &tok_json,
        &read_json_or_null(&a.src.join("tokenizer_config.json"))?,
    )?;
    let num = |k: &str| {
        hf_cfg[k]
            .as_f64()
            .ok_or_else(|| anyhow!("config.json: no {k}"))
    };
    let name = a.name.clone().unwrap_or_else(|| {
        a.src
            .file_name()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| "s5".into())
    });
    let own = [
        "vocab_size",
        "hidden_size",
        "num_hidden_layers",
        "num_attention_heads",
        "intermediate_size",
        "global_attn_every_n_layers",
        "local_attention",
        "global_rope_theta",
        "local_rope_theta",
        "norm_eps",
        "max_position_embeddings",
    ];
    let mut extra: Map<String, Value> = hf_cfg
        .as_object()
        .cloned()
        .unwrap_or_default()
        .into_iter()
        .filter(|(k, _)| !own.contains(&k.as_str()) && !k.starts_with('_'))
        .collect();
    extra.insert("source".into(), Value::String(a.src.display().to_string()));
    let cfg = ModelConfig {
        name,
        vocab_size: num("vocab_size")? as usize,
        hidden_size: num("hidden_size")? as usize,
        num_hidden_layers: num("num_hidden_layers")? as usize,
        num_attention_heads: num("num_attention_heads")? as usize,
        intermediate_size: num("intermediate_size")? as usize,
        global_attn_every_n_layers: num("global_attn_every_n_layers")? as usize,
        local_attention: num("local_attention")? as usize,
        global_rope_theta: rope_thetas(&hf_cfg, 10_000.0).0,
        local_rope_theta: rope_thetas(&hf_cfg, 10_000.0).1,
        norm_eps: num("norm_eps").unwrap_or(1e-5) as f32,
        rl_agent: AgentConfig {
            head_layers: a.head_layers,
            max_len: a.max_len,
            head_max_len: a.head_max_len,
            temperature: [1.0; 3],
            temperature_by_options: BTreeMap::new(),
        },
        extra,
        bf16: false,
        special,
    };
    for k in ["norm_bias", "attention_bias", "mlp_bias"] {
        if hf_cfg[k].as_bool() == Some(true) {
            bail!("{k} = true is not supported");
        }
    }
    if let Some(act) = hf_cfg["hidden_activation"].as_str() {
        if act != "gelu" {
            bail!("hidden_activation {act} is not supported");
        }
    }
    cfg.validate()?;

    let raw = read_tensors(&a.src)?;
    let has_mlm = raw.iter().any(|(k, _, _)| k == "decoder.bias");
    let parts = Parts {
        decision: false,
        mlm: has_mlm,
    };
    let want = params::init(&cfg, parts, 0)?;
    let mut out = ParamMap::new();
    let mut skipped = Vec::new();
    for (k, shape, data) in raw {
        let Some(n) = map_name(&k, has_mlm) else {
            skipped.push(k);
            continue;
        };
        let Some(w) = want.get(n.as_str()) else {
            bail!("{k} -> {n}: not a parameter of the model");
        };
        let ws: Vec<usize> = w.shape().iter().map(|&d| d as usize).collect();
        if ws != shape {
            bail!("{k}: shape {shape:?}, expected {ws:?}");
        }
        let dims: Vec<i32> = shape.iter().map(|&d| d as i32).collect();
        out.insert(n.into(), Array::from_slice(&data, &dims));
    }
    let missing: Vec<&str> = want
        .keys()
        .filter(|k| !out.contains_key(*k))
        .map(|k| k.as_ref())
        .collect();
    if !missing.is_empty() {
        bail!("missing tensors: {missing:?}");
    }
    std::fs::create_dir_all(&a.out)?;
    params::save(&a.out.join("model.safetensors"), &out, None)?;
    std::fs::write(
        a.out.join("config.json"),
        serde_json::to_string_pretty(&cfg)?,
    )?;
    std::fs::copy(&tok_path, a.out.join("tokenizer.json"))?;
    if has_mlm {
        for (probe, top) in mlm_probe(&cfg, &out, &a.out.join("tokenizer.json"))? {
            println!("MLM probe {probe:?}: {top:?}");
        }
    }
    println!(
        "imported {} ({} parameters{}) to {}; skipped {skipped:?}",
        cfg.name,
        params::count(&out),
        if has_mlm { ", MLM head" } else { "" },
        a.out.display()
    );
    Ok(())
}
