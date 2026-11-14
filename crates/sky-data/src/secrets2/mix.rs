//! From candidates to what the trainer and the baselines read
//! (`docs/s6-secrets2-prereg.md`, sections 3.4 and 4.1).
//!
//! Reads `<build>/candidates/<source>-<split>.jsonl` (the items of [`super::cred`],
//! [`super::issues`] and [`super::synth`]) and writes, for each source:
//!
//! - `records/<source>/train.jsonl` and `records/<source>/val.jsonl`: decision records. A
//!   directory is a group of the trainer, so the mix of section 3.4 is
//!   `--records records/creddata,records/issues,records/synth --weights [`WEIGHTS`]`, and the
//!   validation of each source is a file of its own. Each candidate gives the real question
//!   (noul) and, when its kind is known, the kind question (choice): the issue-report set says
//!   real or not, never which kind of not, so its candidates have the real question only. The
//!   `src` of a record is `<source>:<question>`, `creddata_weak:<question>` for a weak negative;
//! - `tables/<source>-<split>.jsonl`: one line per candidate with its label, its state and its
//!   features, for the baselines that are trained on the same candidates.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::{bail, Result};
use serde_json::json;
use sky_infer::secret_cand::{Features, WordSet, KINDS, Q_KIND, Q_REAL};
use sky_infer::tokenizer::Tokenizer;

use super::{bump, file_entry, state_of, write_json, Counts, Item, Label, CREDDATA, ISSUES, SYNTH};
use crate::record::{one_hot, read_jsonl, write_jsonl, DecisionRecord, Question};

/// The weights of section 3.4, as the trainer's `--weights` takes them.
pub const WEIGHTS: &str = "creddata=0.45,issues=0.35,synth=0.20";

pub const SOURCES: [&str; 3] = [CREDDATA, ISSUES, SYNTH];
pub const SPLITS: [&str; 2] = ["train", "val"];

/// The real question.
pub fn real_question() -> Question {
    Question::noul(Q_REAL)
}

/// The kind question: its three options, word for word, with no description.
pub fn kind_question() -> Question {
    Question::choice(
        Q_KIND,
        KINDS
            .iter()
            .map(|k| (k.to_string(), String::new()))
            .collect(),
    )
}

/// The `src` of a record: the source, `creddata_weak` for a weak negative, then the question.
pub fn src_of(item: &Item, question: &str) -> String {
    let source = if item.weak {
        "creddata_weak"
    } else {
        &item.source
    };
    format!("{source}:{question}")
}

/// The decision records of one candidate with its state.
pub fn records_of(item: &Item, state: &str) -> Vec<DecisionRecord> {
    let mut out = Vec::with_capacity(2);
    let mut rec = real_question().record(
        &src_of(item, "real"),
        state,
        one_hot(2, (item.label == Label::T) as usize),
    );
    rec.split = item.split.clone();
    out.push(rec);
    if item.kind {
        let mut rec = kind_question().record(
            &src_of(item, "kind"),
            state,
            one_hot(KINDS.len(), item.label.kind_index()),
        );
        rec.split = item.split.clone();
        out.push(rec);
    }
    out
}

pub(crate) fn features_json(f: &Features) -> serde_json::Value {
    json!({
        "len": f.len,
        "entropy": f.entropy as f64 / 100.0,
        "tokens": f.tokens,
        "chars_per_token": f.chars_per_token as f64 / 100.0,
        "digits": f.digits,
        "upper": f.upper,
        "lower": f.lower,
        "other": f.other,
        "word_run": f.word_run,
        "gate": f.filters.entropy as u8,
        "placeholder": f.filters.placeholder as u8,
        "stopword": f.filters.stopword as u8,
        "allowlist": f.filters.allowlist as u8,
    })
}

pub struct MixOptions {
    /// The build folder.
    pub out: PathBuf,
    pub tokenizer: Tokenizer,
    pub words: WordSet,
}

/// Builds the records and the tables, and `<out>/counts/mix.json`.
pub fn build(o: &MixOptions) -> Result<serde_json::Value> {
    let mut counts = serde_json::Map::new();
    let mut train_items: BTreeMap<&str, usize> = BTreeMap::new();
    for source in SOURCES {
        let mut per_split = serde_json::Map::new();
        for split in SPLITS {
            let path = o.out.join(format!("candidates/{source}-{split}.jsonl"));
            let items: Vec<Item> = read_jsonl(&path)?;
            let mut records = Vec::with_capacity(items.len() * 2);
            let mut table = Vec::with_capacity(items.len());
            let mut by_label = Counts::new();
            let mut by_src = Counts::new();
            let mut by_rule: BTreeMap<String, Counts> = BTreeMap::new();
            let mut by_class: BTreeMap<String, Counts> = BTreeMap::new();
            let mut states = std::collections::HashSet::new();
            let mut state_chars = 0usize;
            for it in &items {
                if it.source != source || it.split != split {
                    bail!(
                        "{}: {} is not a {source} {split} item",
                        path.display(),
                        it.id
                    );
                }
                let (state, f) = state_of(it, &o.tokenizer, &o.words)?;
                let key = if it.weak { "weak_F" } else { it.label.as_str() };
                bump(&mut by_label, key);
                bump(by_rule.entry(it.rec.rule.clone()).or_default(), key);
                if source == SYNTH {
                    let class = it.class.as_deref().unwrap_or("?");
                    bump(by_class.entry(class.to_string()).or_default(), key);
                }
                for rec in records_of(it, &state) {
                    bump(&mut by_src, &rec.src);
                    records.push(rec);
                }
                states.insert(crate::rng::hash64(state.as_bytes()));
                state_chars += state.chars().count();
                table.push(json!({
                    "id": it.id,
                    "source": it.source,
                    "split": it.split,
                    "label": it.label.as_str(),
                    "real": (it.label == Label::T) as u8,
                    "weak": it.weak,
                    "class": it.class,
                    "via": it.via,
                    "rule": it.rec.rule,
                    "features": features_json(&f),
                    "state": state,
                }));
            }
            let rec_path = o.out.join(format!("records/{source}/{split}.jsonl"));
            write_jsonl(&rec_path, records.iter())?;
            write_jsonl(
                &o.out.join(format!("tables/{source}-{split}.jsonl")),
                table.iter(),
            )?;
            if split == "train" {
                train_items.insert(source, items.len());
            }
            let mut c = serde_json::Map::new();
            c.insert("candidates".into(), json!(items.len()));
            c.insert("distinct_states".into(), json!(states.len()));
            c.insert(
                "mean_state_characters".into(),
                json!(if items.is_empty() {
                    0
                } else {
                    state_chars / items.len()
                }),
            );
            c.insert("records".into(), json!(records.len()));
            c.insert("by_label".into(), json!(by_label));
            c.insert("records_by_src".into(), json!(by_src));
            c.insert("by_rule".into(), json!(by_rule));
            if source == SYNTH {
                c.insert("by_class".into(), json!(by_class));
            }
            per_split.insert(split.to_string(), json!(c));
        }
        counts.insert(source.to_string(), json!(per_split));
    }
    // the dictionary of the word-run feature, as the state builder holds it
    let mut dict = String::new();
    for w in o.words.iter() {
        dict.push_str(w);
        dict.push('\n');
    }
    std::fs::write(o.out.join("dictionary.txt"), &dict)?;
    let total: usize = train_items.values().sum();
    let share = |s: &str| train_items.get(s).copied().unwrap_or(0) as f64 / total.max(1) as f64;
    let out = json!({
        "questions": {
            "real": {"type": "noul", "instructions": Q_REAL, "options": real_question().options()},
            "kind": {"type": "choice", "instructions": Q_KIND, "options": kind_question().options()},
        },
        "dictionary": {"words": o.words.len(), "file": "dictionary.txt"},
        "weights": WEIGHTS,
        "records": "records/creddata,records/issues,records/synth",
        "training_candidates": {
            "creddata": train_items.get(CREDDATA),
            "issues": train_items.get(ISSUES),
            "synth": train_items.get(SYNTH),
            "total": total,
            "synthetic_share": (share(SYNTH) * 1e4).round() / 1e4,
        },
        "sources": counts,
    });
    write_json(&o.out.join("counts/mix.json"), &out)?;
    Ok(out)
}

/// The share of states the 320-token window cuts, over the candidates of training and
/// validation, from the report of `sky-train s2-truncation`: one state per candidate, counted
/// on the records of the real question (every candidate has exactly one).
pub fn window_rule(truncation: &serde_json::Value) -> Result<serde_json::Value> {
    let (mut n, mut cut) = (0u64, 0.0f64);
    let mut per_src = serde_json::Map::new();
    let Some(map) = truncation.as_object() else {
        bail!("the truncation report is not an object");
    };
    for (src, v) in map {
        if !src.ends_with(":real") {
            continue;
        }
        let k = v["n"].as_u64().unwrap_or(0);
        let s = v["state"].as_f64().unwrap_or(0.0);
        n += k;
        cut += (s * k as f64).round();
        per_src.insert(
            src.clone(),
            json!({"states": k, "truncated": (s * k as f64).round() as u64, "share": s}),
        );
    }
    if n == 0 {
        bail!("the truncation report holds no record of the real question");
    }
    let share = cut / n as f64;
    Ok(json!({
        "tool": "sky-train s2-truncation (configs/s6-secrets-pico.json: 320 state tokens, fused numbers)",
        "states": n,
        "truncated_at_320": cut as u64,
        "share": (share * 1e6).round() / 1e6,
        "threshold": 0.05,
        "window": if share > 0.05 { 384 } else { 320 },
        "by_src": per_src,
        "head_truncated_share_all_records": truncation["all"]["head"],
    }))
}

/// `{bytes, sha256}` of every file the mix wrote.
pub fn outputs(out: &Path) -> Result<serde_json::Map<String, serde_json::Value>> {
    let mut m = serde_json::Map::new();
    for source in SOURCES {
        for split in SPLITS {
            for rel in [
                format!("candidates/{source}-{split}.jsonl"),
                format!("records/{source}/{split}.jsonl"),
                format!("tables/{source}-{split}.jsonl"),
            ] {
                m.insert(rel.clone(), file_entry(&out.join(&rel))?);
            }
        }
    }
    Ok(m)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::record::validate_record;
    use crate::secrets2::{record_at, RecFilters};

    fn item(source: &str, label: Label, weak: bool, kind: bool) -> Item {
        let text = "import os\ndb_password = \"Tree42!x\"\nconnect()";
        let at = text.find("Tree42!x").unwrap();
        Item {
            id: format!("{source}:train:0"),
            source: source.into(),
            split: "train".into(),
            label,
            weak,
            kind,
            class: None,
            via: None,
            rec: record_at(
                "t",
                None,
                "generic-keyword",
                Some("db_password"),
                text,
                at,
                at + 8,
                RecFilters::default(),
            ),
        }
    }

    #[test]
    fn a_candidate_gives_its_questions() {
        let recs = records_of(&item(CREDDATA, Label::T, false, true), "state");
        assert_eq!(recs.len(), 2);
        assert_eq!(
            (recs[0].src.as_str(), recs[0].t.as_str()),
            ("creddata:real", "noul")
        );
        assert_eq!(recs[0].ins, Q_REAL);
        assert_eq!(recs[0].target, [0.0, 1.0]);
        assert_eq!(
            (recs[1].src.as_str(), recs[1].t.as_str()),
            ("creddata:kind", "choice")
        );
        assert_eq!(recs[1].opts, KINDS);
        assert_eq!(recs[1].target, [1.0, 0.0, 0.0]);
        for r in &recs {
            validate_record(r).unwrap();
            assert_eq!(r.split, "train");
        }
        // X and F are false for the real question, and their own kind
        let recs = records_of(&item(SYNTH, Label::X, false, true), "s");
        assert_eq!(
            (recs[0].target.clone(), recs[1].target.clone()),
            (vec![1.0, 0.0], vec![0.0, 1.0, 0.0])
        );
        let recs = records_of(&item(CREDDATA, Label::F, true, true), "s");
        assert_eq!(recs[0].src, "creddata_weak:real");
        assert_eq!(recs[1].target, [0.0, 0.0, 1.0]);
        // an issue-report candidate has the real question only
        let recs = records_of(&item(ISSUES, Label::F, false, false), "s");
        assert_eq!(recs.len(), 1);
        assert_eq!(recs[0].src, "issues:real");
    }

    #[test]
    fn the_weights_name_the_record_directories() {
        let names: Vec<&str> = WEIGHTS
            .split(',')
            .map(|p| p.split('=').next().unwrap())
            .collect();
        assert_eq!(names, SOURCES);
        let sum: f64 = WEIGHTS
            .split(',')
            .map(|p| p.split('=').nth(1).unwrap().parse::<f64>().unwrap())
            .sum();
        assert!((sum - 1.0).abs() < 1e-9);
    }

    #[test]
    fn the_window_rule_counts_one_state_per_candidate() {
        let report = json!({
            "creddata:real": {"n": 1000, "state": 0.04, "head": 0.0, "any": 0.04},
            "creddata:kind": {"n": 1000, "state": 0.04, "head": 0.0, "any": 0.04},
            "issues:real": {"n": 3000, "state": 0.06, "head": 0.0, "any": 0.06},
            "all": {"n": 5000, "state": 0.052, "head": 0.0, "any": 0.052},
        });
        let r = window_rule(&report).unwrap();
        assert_eq!(
            (r["states"].as_u64(), r["truncated_at_320"].as_u64()),
            (Some(4000), Some(220))
        );
        assert_eq!(r["window"], 384);
        let report =
            json!({"synth:real": {"n": 100, "state": 0.05}, "all": {"n": 100, "state": 0.05}});
        assert_eq!(window_rule(&report).unwrap()["window"], 320);
        assert!(window_rule(&json!({"x:kind": {"n": 3, "state": 0.0}})).is_err());
    }
}
