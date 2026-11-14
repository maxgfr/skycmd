//! `BUILD.json`: every count of the build with the hashes of what it read and what it wrote,
//! and the sample of weak negatives that is read by hand (section 3.1).
//!
//! `BUILD.json` holds no date and no path of the data folder: the same inputs and the same seed
//! give the same bytes, wherever the folder is.

use std::path::{Path, PathBuf};

use anyhow::{bail, Result};
use serde_json::json;

use super::mix::{outputs, window_rule};
use super::{file_entry, read_json, sha256_hex, write_json, Item};
use crate::record::read_jsonl;
use crate::rng::Rng;

/// Weak negatives read by hand (section 3.1).
pub const HAND_READ: usize = 200;

/// Writes `<out>/handread/weak-negatives.jsonl`: [`HAND_READ`] weak negatives drawn at random
/// (seed), each with what a reader needs to judge it. The file holds values and lines: it
/// stays in the data folder.
pub fn hand_read_sample(out: &Path, seed: u64) -> Result<serde_json::Value> {
    let items: Vec<Item> = read_jsonl(&out.join("candidates/creddata-train.jsonl"))?;
    let weak: Vec<&Item> = items.iter().filter(|i| i.weak).collect();
    if weak.len() < HAND_READ {
        bail!("{} weak negatives, {HAND_READ} are to be read", weak.len());
    }
    let mut order: Vec<usize> = (0..weak.len()).collect();
    Rng::derived(seed, "s6-secrets2:weak-hand-read").shuffle(&mut order);
    order.truncate(HAND_READ);
    order.sort_unstable();
    let rows: Vec<serde_json::Value> = order
        .iter()
        .enumerate()
        .map(|(n, &i)| {
            let r = &weak[i].rec;
            json!({
                "n": n + 1,
                "id": weak[i].id,
                "file": r.file,
                "rule": r.rule,
                "name": r.name,
                "value": r.value,
                "line": r.line_text.chars().skip(r.start.saturating_sub(r.line_from).saturating_sub(150)).take(300 + r.value.chars().count().min(200)).collect::<String>(),
                "prev": r.prev_line.as_deref().map(|l| l.chars().take(160).collect::<String>()),
                "next": r.next_line.as_deref().map(|l| l.chars().take(160).collect::<String>()),
                "flags": r.filters,
            })
        })
        .collect();
    let path = out.join("handread/weak-negatives.jsonl");
    crate::record::write_jsonl(&path, rows.iter())?;
    let ids: Vec<&str> = order.iter().map(|&i| weak[i].id.as_str()).collect();
    let summary = json!({
        "weak_negatives": weak.len(),
        "drawn": HAND_READ,
        "ids_sha256": sha256_hex(ids.join("\n").as_bytes()),
        "file": {"handread/weak-negatives.jsonl": file_entry(&path)?},
    });
    write_json(&out.join("counts/handread.json"), &summary)?;
    Ok(summary)
}

pub struct ManifestOptions {
    /// The data folder.
    pub data: PathBuf,
    /// The build folder.
    pub out: PathBuf,
    pub seed: u64,
    pub dict: PathBuf,
    pub tokenizer: PathBuf,
    /// The report of `sky-train s2-truncation --json`.
    pub truncation: Option<PathBuf>,
    /// The result of the reading by hand, when it is done: `<out>/handread/result.json`.
    pub hand_read: Option<PathBuf>,
    /// The model configuration the truncation was measured with.
    pub config: Option<PathBuf>,
}

/// Writes `<out>/BUILD.json` from the counts of each step.
pub fn manifest(o: &ManifestOptions) -> Result<serde_json::Value> {
    let counts = |name: &str| read_json(&o.out.join(format!("counts/{name}.json")));
    let cred = counts("creddata")?;
    let issues = counts("issues")?;
    let synth = counts("synth")?;
    let mix = counts("mix")?;
    let pretext = counts("pretrain-text").ok();
    let hand = counts("handread").ok();
    let truncation = match &o.truncation {
        Some(p) => Some(window_rule(&read_json(p)?)?),
        None => None,
    };
    // the reading by hand counts only for the sample it was made on
    let hand_result = match &o.hand_read {
        Some(p) if p.exists() => {
            let r = read_json(p)?;
            let sample = hand.as_ref().map(|h| &h["ids_sha256"]);
            if sample.is_none() || Some(&r["ids_sha256"]) != sample {
                bail!("{}: not the reading of this build's sample", p.display());
            }
            Some(r)
        }
        _ => None,
    };
    // the same tool, file by file: where the truncated states are
    let mut by_file = serde_json::Map::new();
    for source in super::mix::SOURCES {
        for split in super::mix::SPLITS {
            let p = o.out.join(format!("truncation/{source}-{split}.json"));
            if p.exists() {
                by_file.insert(format!("{source}-{split}"), window_rule(&read_json(&p)?)?);
            }
        }
    }
    let truncation = truncation.map(|mut t| {
        if !by_file.is_empty() {
            t["by_file"] = json!(by_file
                .iter()
                .map(|(k, v)| (k.clone(), json!({"states": v["states"], "truncated_at_320": v["truncated_at_320"], "share": v["share"], "by_src": v["by_src"]})))
                .collect::<serde_json::Map<_, _>>());
        }
        t
    });
    let config = o.config.as_deref().map(file_entry).transpose()?;
    // the word list the state builder holds is the one of this CredData
    let list: Vec<String> =
        serde_json::from_value(read_json(&o.data.join("creddata/word_in_path.json"))?)?;
    if list != sky_infer::secret_cand::WORD_IN_PATH {
        bail!("creddata/word_in_path.json is not the list of sky_infer::secret_cand::WORD_IN_PATH");
    }
    let mut out_files = outputs(&o.out)?;
    for rel in ["dictionary.txt", "pretrain-text/creddata.jsonl"] {
        let p = o.out.join(rel);
        if p.exists() {
            out_files.insert(rel.to_string(), file_entry(&p)?);
        }
    }
    let build = json!({
        "experiment": "s6-secrets v2 (s6-secrets-verify-pico): the data between the frozen generator and the trainer",
        "pre_registration": "docs/s6-secrets2-prereg.md, sections 3, 4 and 5, amendments 1 to 3",
        "seed": o.seed,
        "inputs": {
            "generator": cred["generator"],
            "splits.json": cred["splits_json"],
            "creddata/word_in_path.json": file_entry(&o.data.join("creddata/word_in_path.json"))?,
            "issue_reports": issues["inputs"],
            "dictionary_source": {"path": o.dict.display().to_string(), "file": file_entry(&o.dict)?},
            "tokenizer": {"name": o.tokenizer.file_name().map(|n| n.to_string_lossy().into_owned()), "file": file_entry(&o.tokenizer)?},
            "truncation_config": config,
        },
        "creddata": {
            "repositories": cred["repositories"],
            "labels": cred["labels"],
            "label_differences": cred["label_differences"],
            "decontamination": cred["decontamination"],
            "weak_negatives": cred["weak_negatives"],
            "items": cred["items"],
        },
        "issue_reports": {
            "rule": issues["rule"],
            "test_rows_hashed_for_decontamination": issues["test_rows_hashed_for_decontamination"],
            "test_distinct_texts": issues["test_distinct_texts"],
            "replacement": issues["replacement"],
            "splits": issues["splits"],
        },
        "synthetic": synth,
        "mix": mix,
        "weak_negatives_read_by_hand": {"sample": hand, "result": hand_result},
        "truncation": truncation,
        "pretraining_text": pretext,
        "outputs": out_files,
    });
    write_json(&o.out.join("BUILD.json"), &build)?;
    Ok(build)
}
