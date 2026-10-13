//! s5 evaluation sets (`split = test`), written once and frozen by the pre-registration:
//!
//! - **fitted**: the typed-decisions test split (`typed_decisions_test.jsonl`) and its
//!   coherence families with the disjunctions (`coherence_test.jsonl`);
//! - **zero-shot**: the sets of `eval/s5/defs.json` (btzsc wikitoxic ×4, biasframes ×3,
//!   amazonpolarity, yahootopics; SIB-200 French and English), rebuilt from the raw files with
//!   the hand-written questions and class definitions of that file.
//!
//! A question with more than `max_options` classes keeps the gold class and `max_options - 1`
//! others drawn with a fixed seed, in the order of the definitions file (N = min(8, K)).

use std::collections::{BTreeMap, HashSet};
use std::path::PathBuf;

use anyhow::{anyhow, bail, Context, Result};
use serde_json::Value;

use crate::hf::read_parquet;
use crate::record::{read_jsonl, DecisionRecord};
use crate::rng::Rng;
use crate::s5::{coherence, robust, uid_of, write, S5Record};

#[derive(Clone, Debug)]
pub struct EvalOptions {
    /// `data/s5/src` (the `fetch-evals` downloads).
    pub src: PathBuf,
    /// `data/decisions` (the typed-decisions test split).
    pub decisions: PathBuf,
    /// `eval/s5/defs.json`.
    pub defs: PathBuf,
    /// `data/s5/eval`.
    pub out: PathBuf,
    /// `data/text` (the evaluation distractors: `simplestories.jsonl`).
    pub text: PathBuf,
    /// `eval/s5/paraphrases.json`.
    pub paraphrases: PathBuf,
    pub per_set: usize,
    pub max_options: usize,
    pub seed: u64,
}

/// Longest state kept, in characters.
const STATE_MAX: usize = 2000;

fn clip(s: &str) -> String {
    let s = s.trim();
    match s.char_indices().nth(STATE_MAX) {
        Some((i, _)) => s[..i].to_string(),
        None => s.to_string(),
    }
}

/// `(text, gold label)` of a btzsc config, one per text (rows are one per hypothesis).
fn btzsc_rows(o: &EvalOptions, config: &str) -> Result<Vec<(String, String)>> {
    let dir = o.src.join("btzsc").join(config);
    let mut files: Vec<PathBuf> = std::fs::read_dir(&dir)
        .with_context(|| format!("{}", dir.display()))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "parquet"))
        .collect();
    files.sort();
    let mut seen = HashSet::new();
    let mut out = Vec::new();
    for f in files {
        for row in read_parquet(&f, &["text", "label_text"], usize::MAX)? {
            let text = row["text"].as_str().unwrap_or_default().to_string();
            let label = row["label_text"].as_str().unwrap_or_default().to_string();
            if !text.trim().is_empty() && seen.insert(text.clone()) {
                out.push((text, label));
            }
        }
    }
    Ok(out)
}

/// `(text, category)` of the SIB-200 test split of a language (`fra_Latn`, `eng_Latn`).
fn sib_rows(o: &EvalOptions, lang: &str) -> Result<Vec<(String, String)>> {
    let path = o.src.join("sib200/data").join(lang).join("test.tsv");
    let text = std::fs::read_to_string(&path).with_context(|| format!("{}", path.display()))?;
    let mut out = Vec::new();
    for line in text.lines().skip(1) {
        let cols: Vec<&str> = line.split('\t').collect();
        if cols.len() >= 3 {
            out.push((cols[2].to_string(), cols[1].to_string()));
        }
    }
    Ok(out)
}

/// `(utterance, intent name)` of the HWU64 test split.
fn hwu_rows(o: &EvalOptions) -> Result<Vec<(String, String)>> {
    let dir = o.src.join("hwu64");
    let names: Vec<String> = read_parquet(
        &dir.join("intents/intents-00000-of-00001.parquet"),
        &["id", "name"],
        usize::MAX,
    )?
    .into_iter()
    .map(|r| r["name"].as_str().unwrap_or_default().to_string())
    .collect();
    let mut out = Vec::new();
    for r in read_parquet(
        &dir.join("data/test-00000-of-00001.parquet"),
        &["utterance", "label"],
        usize::MAX,
    )? {
        let label = r["label"].as_u64().unwrap_or(0) as usize;
        let name = names
            .get(label)
            .ok_or_else(|| anyhow!("hwu64: label {label} without a name"))?;
        out.push((
            r["utterance"].as_str().unwrap_or_default().to_string(),
            name.clone(),
        ));
    }
    Ok(out)
}

/// The s5 record of one example of a set.
fn record(
    set: &Value,
    text: &str,
    gold: &str,
    max_options: usize,
    rng: &mut Rng,
) -> Result<S5Record> {
    let name = set["set"].as_str().unwrap_or_default();
    let lang = set["lang"].as_str().unwrap_or("en");
    let t = set["t"].as_str().unwrap_or("choice");
    let ins = set["ins"].as_str().unwrap_or_default().to_string();
    let (opts_kv, target) = match t {
        "noul" => {
            let positive = set["positive"].as_str().unwrap_or_default();
            let f = set["false"].as_str().unwrap_or_default();
            let tr = set["true"].as_str().unwrap_or_default();
            let yes = gold == positive;
            (
                vec![
                    ("false".to_string(), f.to_string()),
                    ("true".to_string(), tr.to_string()),
                ],
                vec![f32::from(u8::from(!yes)), f32::from(u8::from(yes))],
            )
        }
        _ => {
            let classes = set["classes"]
                .as_object()
                .ok_or_else(|| anyhow!("{name}: no classes"))?;
            let all: Vec<(String, String)> = classes
                .iter()
                .map(|(k, v)| (k.clone(), v.as_str().unwrap_or_default().to_string()))
                .collect();
            let g = all
                .iter()
                .position(|(k, _)| k == gold)
                .ok_or_else(|| anyhow!("{name}: gold label {gold:?} has no definition"))?;
            let mut keep: Vec<usize> = (0..all.len()).filter(|&i| i != g).collect();
            rng.shuffle(&mut keep);
            keep.truncate(max_options.saturating_sub(1).min(all.len() - 1));
            keep.push(g);
            keep.sort_unstable();
            (
                keep.iter().map(|&i| all[i].clone()).collect(),
                keep.iter().map(|&i| f32::from(u8::from(i == g))).collect(),
            )
        }
    };
    Ok(S5Record {
        src: name.into(),
        t: t.into(),
        ins,
        opts_kv,
        state: clip(text),
        target,
        split: "test".into(),
        uid: uid_of(&[name, text]),
        lang: if lang == "en" {
            String::new()
        } else {
            lang.into()
        },
        gold: true,
    })
}

/// Writes every evaluation set; returns the record count per file.
pub fn build(o: &EvalOptions) -> Result<BTreeMap<String, usize>> {
    let defs: Value = serde_json::from_str(
        &std::fs::read_to_string(&o.defs).with_context(|| format!("{}", o.defs.display()))?,
    )?;
    let mut counts = BTreeMap::new();
    let mut zero_shot: BTreeMap<String, (Vec<S5Record>, BTreeMap<String, String>)> =
        BTreeMap::new();
    for set in defs["sets"]
        .as_array()
        .ok_or_else(|| anyhow!("defs.json has no sets"))?
    {
        let name = set["set"].as_str().unwrap_or_default();
        let src = set["source"].as_str().unwrap_or_default();
        let rows = if let Some(cfg) = src.strip_prefix("btzsc/btzsc ") {
            btzsc_rows(o, cfg.split_whitespace().next().unwrap_or_default())?
        } else if src.contains("sib200 fra_Latn") {
            sib_rows(o, "fra_Latn")?
        } else if src.contains("sib200 eng_Latn") {
            sib_rows(o, "eng_Latn")?
        } else if src.starts_with("DeepPavlov/hwu64") {
            hwu_rows(o)?
        } else {
            bail!("{name}: unknown source {src:?}");
        };
        let mut rng = Rng::derived(o.seed, &format!("s5/eval/{name}"));
        let mut idx: Vec<usize> = (0..rows.len()).collect();
        rng.shuffle(&mut idx);
        idx.truncate(o.per_set);
        idx.sort_unstable();
        let mut recs = Vec::with_capacity(idx.len());
        for i in idx {
            let (text, gold) = &rows[i];
            recs.push(record(set, text, gold, o.max_options, &mut rng)?);
        }
        counts.insert(
            format!("{name}.jsonl"),
            write(&o.out.join(format!("{name}.jsonl")), &recs)?,
        );
        let classes: BTreeMap<String, String> = set["classes"]
            .as_object()
            .map(|m| {
                m.iter()
                    .map(|(k, v)| (k.clone(), v.as_str().unwrap_or_default().to_string()))
                    .collect()
            })
            .unwrap_or_default();
        zero_shot.insert(name.to_string(), (recs, classes));
    }
    // The fitted bench: typed-decisions test, and its coherence families.
    let bench: Vec<DecisionRecord> = read_jsonl(&o.decisions.join("typed_decisions_bench.jsonl"))?;
    let typed: Vec<S5Record> = bench
        .iter()
        .map(|d| {
            let uid = uid_of(&["typed_decisions_test", &d.ins, &d.state]);
            let mut r = S5Record::from_decision(d, &uid, "");
            r.src = "typed_decisions_test".into();
            r.split = "test".into();
            r
        })
        .collect();
    counts.insert(
        "typed_decisions_test.jsonl".into(),
        write(&o.out.join("typed_decisions_test.jsonl"), &typed)?,
    );
    let coh = coherence::families(&typed, true);
    counts.insert(
        "coherence_test.jsonl".into(),
        write(&o.out.join("coherence_test.jsonl"), &coh)?,
    );
    // Robustness sets.
    let para: Value = serde_json::from_str(
        &std::fs::read_to_string(&o.paraphrases)
            .with_context(|| format!("{}", o.paraphrases.display()))?,
    )?;
    let mut put = |name: &str, recs: Vec<S5Record>| -> Result<()> {
        counts.insert(name.into(), write(&o.out.join(name), &recs)?);
        Ok(())
    };
    put("paraphrase_test.jsonl", robust::paraphrases(&typed, &para))?;
    let distractors =
        crate::s5::build::sentences(&o.text.join("simplestories.jsonl"), 5000, o.seed ^ 0xD15)?;
    if distractors.is_empty() {
        bail!("no evaluation distractors: data/text/simplestories.jsonl is missing");
    }
    put(
        "jevout_lite.jsonl",
        robust::jevout(&typed, &distractors, 500, o.seed),
    )?;
    let mut binary: Vec<S5Record> = typed.clone();
    for s in [
        "wikitoxic_toxicaggregated",
        "biasframes_offensive",
        "amazonpolarity",
    ] {
        if let Some((r, _)) = zero_shot.get(s) {
            binary.extend(r.iter().cloned());
        }
    }
    put(
        "reassign_test.jsonl",
        robust::reassign(&binary, 300, o.seed),
    )?;
    put("policy_test.jsonl", robust::policy(400, o.seed))?;
    let mut cov = Vec::new();
    for s in ["hwu64", "yahootopics"] {
        if let Some((r, classes)) = zero_shot.get(s) {
            let take: Vec<S5Record> = r.iter().take(500).cloned().collect();
            cov.extend(robust::coverage(s, &take, classes, o.seed));
        }
    }
    put("coverage_test.jsonl", cov)?;
    Ok(counts)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn choice_keeps_gold_and_samples_n() {
        let set: Value = serde_json::json!({
            "set": "x", "lang": "en", "t": "choice", "ins": "Topic?",
            "classes": {"a": "A.", "b": "B.", "c": "C.", "d": "D.", "e": "E."}
        });
        let mut rng = Rng::new(1);
        for _ in 0..20 {
            let r = record(&set, "text", "d", 3, &mut rng).unwrap();
            assert_eq!(r.opts_kv.len(), 3);
            let g = r.gold_index();
            assert_eq!(r.opts_kv[g].0, "d");
            assert_eq!(r.target.iter().sum::<f32>(), 1.0);
            let ids: Vec<&str> = r.opts_kv.iter().map(|(k, _)| k.as_str()).collect();
            let mut sorted = ids.clone();
            sorted.sort_unstable();
            assert_eq!(ids, sorted, "definition order kept");
        }
        let noul: Value = serde_json::json!({
            "set": "y", "lang": "en", "t": "noul", "ins": "Toxic.",
            "false": "Civil.", "true": "Rude.", "positive": "toxic"
        });
        let r = record(&noul, "t", "toxic", 8, &mut rng).unwrap();
        assert_eq!(r.target, vec![0.0, 1.0]);
        let r = record(&noul, "t", "not_toxic", 8, &mut rng).unwrap();
        assert_eq!(r.target, vec![1.0, 0.0]);
        assert!(record(&set, "t", "zz", 8, &mut rng).is_err());
    }
}
