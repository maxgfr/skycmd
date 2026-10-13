//! Post-training: temperature calibration, export to the oscar / HF layout
//! and the evaluation report.

use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
};

use anyhow::{bail, Context, Result};
use mlx_rs::{Array, Dtype};
use serde_json::{json, Value};

use crate::{
    config::ModelConfig,
    decision::{self, Item},
    metrics::{self, Pred},
    params::{self, ParamMap},
    sequence::{temp_bucket, Tok, QTYPES},
    train::{grouped_metrics, rl_agent_json},
};

/// A decision checkpoint and the run directory it belongs to.
#[derive(Debug, Clone)]
pub struct RunPaths {
    /// Directory with `model.safetensors`, `config.json`, `tokenizer.json`, `rl_agent_config.json`.
    pub ckpt: PathBuf,
    /// `runs/<name>` (where `report.md` goes).
    pub root: PathBuf,
}

/// Accepts `runs/<name>`, `runs/<name>/decide` or `runs/<name>/decide/last`.
pub fn resolve_run(path: &Path) -> Result<RunPaths> {
    let has = |p: &Path| p.join("model.safetensors").exists() && p.join("config.json").exists();
    let ckpt = if has(path) {
        path.to_path_buf()
    } else if has(&path.join("last")) {
        path.join("last")
    } else if has(&path.join("decide/last")) {
        path.join("decide/last")
    } else {
        bail!("no decision checkpoint found under {}", path.display());
    };
    let mut root = ckpt.clone();
    if root.file_name().is_some_and(|n| n == "last") {
        root.pop();
    }
    if root
        .file_name()
        .is_some_and(|n| n == "decide" || n == "pretrain")
    {
        root.pop();
    }
    Ok(RunPaths { ckpt, root })
}

/// Loaded decision model with its tokenizer and Laya runtime config.
pub struct Loaded {
    pub cfg: ModelConfig,
    pub params: ParamMap,
    pub tok: Tok,
    pub rl: Value,
}

pub fn load_run(run: &RunPaths) -> Result<Loaded> {
    let cfg: ModelConfig =
        serde_json::from_str(&std::fs::read_to_string(run.ckpt.join("config.json"))?)?;
    let params = params::load(&run.ckpt.join("model.safetensors"))?;
    if !params.contains_key("scorer.3.weight") {
        bail!(
            "{} is not a decision checkpoint (no scorer)",
            run.ckpt.display()
        );
    }
    let tok = Tok::load_with(&run.ckpt.join("tokenizer.json"), &cfg.special)?;
    let rl_path = run.ckpt.join("rl_agent_config.json");
    let rl = match std::fs::read_to_string(&rl_path) {
        Ok(t) => serde_json::from_str(&t)?,
        Err(_) => rl_agent_json(&cfg, json!({})),
    };
    Ok(Loaded {
        cfg,
        params,
        tok,
        rl,
    })
}

/// Temperature of a prediction under a Laya config: bucket, then type, then 1.
pub fn temperature_of(rl: &Value, qtype: usize, k: usize) -> f32 {
    if let Some(t) = rl["temperature_by_options"][temp_bucket(qtype, k)].as_f64() {
        return t as f32;
    }
    rl["temperature"][qtype].as_f64().unwrap_or(1.0) as f32
}

fn split_items<'a>(data: &'a decision::DecisionData, split: &str) -> Vec<&'a Item> {
    match split {
        "val" => data.val.iter().collect(),
        "test" => data.test.iter().collect(),
        _ => data.train.iter().flatten().collect(),
    }
}

/// Fits per-type and per-bucket temperatures on `split` and writes them into
/// the checkpoint's `rl_agent_config.json`. Returns the new config.
pub fn calibrate(run: &Path, records: &[PathBuf], split: &str, min_bucket: usize) -> Result<Value> {
    let paths = resolve_run(run)?;
    let m = load_run(&paths)?;
    let data = decision::load(records, &m.tok, &m.cfg.rl_agent, &m.cfg.special)?;
    let items = split_items(&data, split);
    if items.is_empty() {
        bail!("no {split} items to calibrate on");
    }
    let preds = decision::predict(&m.params, &m.cfg, &data.groups, &items, 16_384)?;
    let mut temps = [1.0f32; 3];
    for (q, t) in temps.iter_mut().enumerate() {
        let sel: Vec<&Pred> = preds.iter().filter(|p| p.qtype == q).collect();
        if !sel.is_empty() {
            *t = metrics::fit_temperature(&sel);
        }
    }
    let mut buckets: BTreeMap<String, Vec<&Pred>> = BTreeMap::new();
    for p in &preds {
        buckets
            .entry(temp_bucket(p.qtype, p.logits.len()))
            .or_default()
            .push(p);
    }
    let mut by_options = BTreeMap::new();
    for (b, sel) in &buckets {
        if sel.len() >= min_bucket {
            by_options.insert(b.clone(), metrics::fit_temperature(sel));
        }
    }
    let mut rl = m.rl.clone();
    rl["temperature"] = json!(temps);
    rl["temperature_by_options"] = json!(by_options);
    rl["calibration"] = json!({"split": split, "items": preds.len(), "min_bucket": min_bucket});
    let before = metrics::compute(preds.iter(), |_| 1.0);
    let after = metrics::compute(preds.iter(), |p| {
        temperature_of(&rl, p.qtype, p.logits.len())
    });
    println!("calibration on {} {split} items:", preds.len());
    for (q, t) in temps.iter().enumerate() {
        println!("  {:<6} T = {t:.4}", QTYPES[q]);
    }
    for (b, t) in &by_options {
        println!("  {b:<12} T = {t:.4} (n = {})", buckets[b].len());
    }
    println!(
        "  NLL {:.4} -> {:.4}, ECE {:.4} -> {:.4}",
        before.nll, after.nll, before.ece, after.ece
    );
    std::fs::write(
        paths.ckpt.join("rl_agent_config.json"),
        serde_json::to_string_pretty(&rl)?,
    )?;
    Ok(rl)
}

/// Tensors written to the exported `model.safetensors` (no MLM head).
pub fn export_tensors(params: &ParamMap, rl: &Value) -> ParamMap {
    let mut out: ParamMap = params
        .iter()
        .filter(|(k, _)| {
            k.starts_with("encoder.")
                || k.starts_with("head.")
                || k.starts_with("scorer.")
                || k.as_ref() == "type_emb.weight"
        })
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();
    let t: Vec<f32> = (0..3)
        .map(|q| rl["temperature"][q].as_f64().unwrap_or(1.0) as f32)
        .collect();
    // Laya keeps the per-type temperatures as a buffer too (oscar has it).
    out.insert("temperature".into(), Array::from_slice(&t, &[3]));
    out
}

/// Writes `model.safetensors` (f32), `config.json`, `encoder/config.json`,
/// `rl_agent_config.json` and `tokenizer/` in the oscar layout.
pub fn export(run: &Path, out: &Path) -> Result<()> {
    let paths = resolve_run(run)?;
    let m = load_run(&paths)?;
    std::fs::create_dir_all(out.join("encoder"))?;
    std::fs::create_dir_all(out.join("tokenizer"))?;
    let tensors = export_tensors(&m.params, &m.rl);
    params::save(
        &out.join("model.safetensors"),
        &tensors,
        Some(Dtype::Float32),
    )?;
    std::fs::write(
        out.join("encoder/config.json"),
        serde_json::to_string_pretty(&m.cfg.encoder_config_json())?,
    )?;
    let mut rl = m.rl.clone();
    rl["head_layers"] = json!(m.cfg.rl_agent.head_layers);
    rl["max_len"] = json!(m.cfg.rl_agent.max_len);
    rl["head_max_len"] = json!(m.cfg.rl_agent.head_max_len);
    std::fs::write(
        out.join("rl_agent_config.json"),
        serde_json::to_string_pretty(&rl)?,
    )?;
    std::fs::copy(
        paths.ckpt.join("tokenizer.json"),
        out.join("tokenizer/tokenizer.json"),
    )
    .context("copying tokenizer.json")?;
    let sp = &m.cfg.special;
    let special = |content: &str, lstrip: bool| json!({"content": content, "lstrip": lstrip, "normalized": false, "rstrip": false, "single_word": false});
    let special_map = json!({
        "cls_token": special(&sp.cls_text, false),
        "mask_token": special(&sp.mask_text, true),
        "pad_token": special(&sp.pad_text, false),
        "sep_token": special(&sp.sep_text, false),
        "unk_token": special(&sp.unk_text, false),
    });
    std::fs::write(
        out.join("tokenizer/special_tokens_map.json"),
        serde_json::to_string_pretty(&special_map)?,
    )?;
    let tok_cfg = json!({
        "clean_up_tokenization_spaces": true,
        "cls_token": sp.cls_text,
        "mask_token": sp.mask_text,
        "model_input_names": ["input_ids", "attention_mask"],
        "model_max_length": m.cfg.rl_agent.max_len,
        "pad_token": sp.pad_text,
        "sep_token": sp.sep_text,
        "tokenizer_class": "PreTrainedTokenizerFast",
        "unk_token": sp.unk_text,
    });
    std::fs::write(
        out.join("tokenizer/tokenizer_config.json"),
        serde_json::to_string_pretty(&tok_cfg)?,
    )?;
    let root_cfg = json!({
        "model_family": "skycmd-s1",
        "model_name": format!("skycmd-{}", m.cfg.name),
        "license": "apache-2.0",
        "kind": "calibrated decision model (non-generative)",
        "note": "Laya-format decision model: ModernBERT encoder + decision head, trained from scratch with sky-train.",
        "files": {
            "model.safetensors": "encoder + decision head weights (f32)",
            "rl_agent_config.json": "laya-format runtime config: head, contexts, temperatures",
            "tokenizer/": "byte-level BPE tokenizer",
            "encoder/config.json": "backbone architecture (ModernBERT)",
        },
        "parameters": params::count(&tensors) - 3,
    });
    std::fs::write(
        out.join("config.json"),
        serde_json::to_string_pretty(&root_cfg)?,
    )?;
    println!(
        "exported {} ({} parameters) to {}",
        m.cfg.name,
        params::count(&tensors) - 3,
        out.display()
    );
    Ok(())
}

fn table(title: &str, m: &BTreeMap<String, metrics::Metrics>) -> String {
    let mut s = format!("### {title}\n\n| source | n | accuracy | NLL | Brier | ECE |\n|---|---:|---:|---:|---:|---:|\n");
    for (k, x) in m {
        s.push_str(&format!(
            "| {k} | {} | {:.4} | {:.4} | {:.4} | {:.4} |\n",
            x.n, x.accuracy, x.nll, x.brier, x.ece
        ));
    }
    s.push('\n');
    s
}

/// Evaluates the calibrated model per source and writes `report.md` in the run root.
pub fn eval(
    run: &Path,
    records: &[PathBuf],
    split: &str,
    report: Option<&Path>,
) -> Result<PathBuf> {
    let paths = resolve_run(run)?;
    let m = load_run(&paths)?;
    let data = decision::load(records, &m.tok, &m.cfg.rl_agent, &m.cfg.special)?;
    let mut used = split.to_string();
    let mut items = split_items(&data, split);
    if items.is_empty() && split == "test" {
        used = "val".into();
        items = split_items(&data, "val");
    }
    if items.is_empty() {
        bail!("no {split} items to evaluate");
    }
    let preds = decision::predict(&m.params, &m.cfg, &data.groups, &items, 16_384)?;
    let temp = |p: &Pred| temperature_of(&m.rl, p.qtype, p.logits.len());
    let by_src = grouped_metrics(&preds, |p| p.src.clone(), temp);
    let by_group = grouped_metrics(&preds, |p| p.group.clone(), temp);
    let by_type = grouped_metrics(&preds, |p| QTYPES[p.qtype].to_string(), temp);
    let raw = grouped_metrics(&preds, |p| p.group.clone(), |_| 1.0);
    let mut md = format!(
        "# Evaluation report: skycmd-{}\n\nCheckpoint: `{}`  \nSplit: `{used}` ({} questions)  \n\
         Metrics use the calibrated temperatures of `rl_agent_config.json`; ECE uses 15 bins on the top probability.\n\n",
        m.cfg.name,
        paths.ckpt.display(),
        preds.len()
    );
    md.push_str(&table("Per source (`src`)", &by_src));
    md.push_str(&table("Per data group", &by_group));
    md.push_str(&table("Per question type", &by_type));
    md.push_str(&table("Per data group, uncalibrated (T = 1)", &raw));
    let out = report
        .map(Path::to_path_buf)
        .unwrap_or_else(|| paths.root.join("report.md"));
    std::fs::write(&out, md)?;
    println!("{}", table("Per source (`src`)", &by_src));
    println!("report written to {}", out.display());
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn temperature_lookup_falls_back() {
        let rl =
            json!({"temperature": [1.5, 2.0, 0.5], "temperature_by_options": {"choice:3-5": 3.0}});
        assert_eq!(temperature_of(&rl, 0, 4), 3.0);
        assert_eq!(temperature_of(&rl, 0, 2), 1.5);
        assert_eq!(temperature_of(&rl, 2, 2), 0.5);
        assert_eq!(temperature_of(&json!({}), 1, 3), 1.0);
    }

    #[test]
    fn run_resolution() {
        let tmp = tempfile::tempdir().unwrap();
        let ck = tmp.path().join("runs/x/decide/last");
        std::fs::create_dir_all(&ck).unwrap();
        std::fs::write(ck.join("model.safetensors"), b"").unwrap();
        std::fs::write(ck.join("config.json"), b"{}").unwrap();
        for p in ["runs/x", "runs/x/decide", "runs/x/decide/last"] {
            let r = resolve_run(&tmp.path().join(p)).unwrap();
            assert_eq!(r.ckpt, ck);
            assert_eq!(r.root, tmp.path().join("runs/x"));
        }
    }
}
