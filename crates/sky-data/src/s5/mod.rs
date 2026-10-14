//! s5 data: decision records whose options are `(id, definition)` pairs (the schema of
//! sky-train `s5/record.rs`), the training corpus, the code generators, the coherence families,
//! the evaluation sets, decontamination and the class-balance gate.
//!
//! **No GLM output.** The z.ai terms forbid using model output to train models; nothing here
//! reads `data/teacher/`, `glm_orders` or `paraphrased`. The only model labels come from the
//! Apache-2.0 teachers, run locally (`sky-train score`).
//!
//! **Splits.** Every record carries `split`: `train`, `val` (model selection), `cal` (a
//! dedicated calibration split: temperatures and escalation thresholds) or `test` (never seen
//! before the pre-registration). Splits are assigned by a hash of the state, so all the
//! questions about one state fall in the same split.

pub mod battery;
pub mod build;
pub mod coherence;
pub mod corpus;
pub mod evalsets;
pub mod gen;
pub mod mix;
pub mod overlap;
pub mod robust;
pub mod stats;

use std::collections::BTreeMap;
use std::io::Write;
use std::path::Path;

use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};

use crate::record::DecisionRecord;
use crate::rng::hash64;

/// One s5 record (JSONL), as read by sky-train.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct S5Record {
    pub src: String,
    pub t: String,
    pub ins: String,
    /// `(id, definition)` per option, in record order.
    pub opts_kv: Vec<(String, String)>,
    #[serde(default)]
    pub state: String,
    pub target: Vec<f32>,
    pub split: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub uid: String,
    /// `en` or `fr` (empty: English).
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub lang: String,
    /// `target` is a human or programmatic label (false: a placeholder, teacher only).
    #[serde(default = "yes", skip_serializing_if = "is_true")]
    pub gold: bool,
}

fn yes() -> bool {
    true
}

fn is_true(b: &bool) -> bool {
    *b
}

impl S5Record {
    /// The s5 record of an s1 record (options split by [`split_legacy`]).
    pub fn from_decision(d: &DecisionRecord, uid: &str, lang: &str) -> Self {
        Self {
            src: d.src.clone(),
            t: d.t.clone(),
            ins: d.ins.clone(),
            opts_kv: d
                .opts
                .iter()
                .enumerate()
                .map(|(i, o)| split_legacy(&d.t, i, o))
                .collect(),
            state: d.state.clone(),
            target: d.target.clone(),
            split: d.split.clone(),
            uid: uid.to_string(),
            lang: lang.to_string(),
            gold: true,
        }
    }

    /// Native rendering (`id: definition`, `level i: …`, `false: …`): the s1 record.
    pub fn to_decision(&self) -> DecisionRecord {
        DecisionRecord {
            src: self.src.clone(),
            t: self.t.clone(),
            ins: self.ins.clone(),
            opts: self
                .opts_kv
                .iter()
                .enumerate()
                .map(|(i, (id, def))| match self.t.as_str() {
                    "score" => format!("level {i}: {def}"),
                    _ if def.is_empty() => id.clone(),
                    _ => format!("{id}: {def}"),
                })
                .collect(),
            state: self.state.clone(),
            target: self.target.clone(),
            split: self.split.clone(),
        }
    }

    pub fn validate(&self) -> Result<()> {
        if !matches!(self.t.as_str(), "choice" | "score" | "noul") {
            bail!("{}: unknown type {:?}", self.uid, self.t);
        }
        if self.opts_kv.len() < 2 || self.opts_kv.len() != self.target.len() {
            bail!(
                "{}: {} options, {} targets",
                self.uid,
                self.opts_kv.len(),
                self.target.len()
            );
        }
        if self.t == "noul"
            && (self.opts_kv.len() != 2
                || self.opts_kv[0].0 != "false"
                || self.opts_kv[1].0 != "true")
        {
            bail!("{}: noul options must be [false, true]", self.uid);
        }
        let s: f32 = self.target.iter().sum();
        if !self.target.iter().all(|x| x.is_finite() && *x >= 0.0) || s <= 0.0 {
            bail!("{}: invalid target", self.uid);
        }
        Ok(())
    }

    /// Index of the largest target (first on ties).
    pub fn gold_index(&self) -> usize {
        crate::record::argmax(&self.target)
    }
}

/// Splits a rendered s1 option into `(id, definition)` (the rule of sky-train
/// `s5::record::split_legacy`, which sky-infer's iso engine mirrors).
pub fn split_legacy(t: &str, i: usize, opt: &str) -> (String, String) {
    match t {
        "score" => {
            let rest = opt.strip_prefix(&format!("level {i}:")).unwrap_or(opt);
            (i.to_string(), rest.trim().to_string())
        }
        "noul" => {
            let id = if i == 0 { "false" } else { "true" };
            let rest = opt.strip_prefix(&format!("{id}:")).unwrap_or(opt).trim();
            let def = if rest == id { "" } else { rest };
            (id.to_string(), def.to_string())
        }
        _ => match opt.split_once(": ") {
            Some((k, d)) if !k.trim().is_empty() && k.len() <= 40 && !k.contains('\n') => {
                (k.trim().to_string(), d.trim().to_string())
            }
            _ => (opt.trim().to_string(), String::new()),
        },
    }
}

/// Sources never used to train s5 (substrings of `src`): the zero-shot evaluation sets and
/// what shares their texts (amazon_polarity; Jigsaw = the Wikipedia talk comments of
/// wikitoxic), every MASSIVE / SLURP subset (HWU64 utterances come from the same collection,
/// and MASSIVE translates them: a leak no n-gram check sees), MTOP (a French zero-shot set),
/// the typed-decisions test split, NOTAM, and anything GLM wrote.
pub const EXCLUDED: &[&str] = &[
    "typed_decisions_bench",
    "amazon_polarity",
    "massive",
    "slurp",
    "mtop",
    "jigsaw",
    "wiki_toxic",
    "yahoo",
    "rotten",
    "bias_frames",
    "biasframes",
    "sib200",
    "hwu",
    "notam",
    "glm_orders",
    "para:",
];

/// Whether a source may be used for training.
pub fn allowed(src: &str) -> bool {
    let s = src.to_ascii_lowercase();
    !EXCLUDED.iter().any(|e| s.contains(e))
}

/// Split of a state: `val` / `cal` / `train` by a hash of `(salt, key)`.
pub fn split_of(key: &str, salt: &str, val: f64, cal: f64) -> &'static str {
    let h = (hash64(format!("s5/{salt}/{key}").as_bytes()) % 1_000_000) as f64 / 1e6;
    if h < val {
        "val"
    } else if h < val + cal {
        "cal"
    } else {
        "train"
    }
}

/// Stable id of a text.
pub fn uid_of(parts: &[&str]) -> String {
    format!("{:016x}", hash64(parts.join("\u{1f}").as_bytes()))
}

/// Reads s5 records (JSONL).
pub fn read(path: &Path) -> Result<Vec<S5Record>> {
    crate::record::read_jsonl(path)
}

/// Writes s5 records (JSONL), validated; returns the count.
pub fn write(path: &Path, recs: &[S5Record]) -> Result<usize> {
    for r in recs {
        r.validate()?;
    }
    if let Some(d) = path.parent() {
        std::fs::create_dir_all(d)?;
    }
    let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
    for r in recs {
        writeln!(f, "{}", serde_json::to_string(r)?)?;
    }
    f.flush()?;
    Ok(recs.len())
}

/// Hex sha256 of a file (the manifests of the frozen splits).
pub fn sha256_file(path: &Path) -> Result<String> {
    use ring::digest::{Context, SHA256};
    let bytes = std::fs::read(path)?;
    let mut c = Context::new(&SHA256);
    c.update(&bytes);
    Ok(c.finish()
        .as_ref()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect())
}

/// `{file: {records, sha256}}` of every JSONL file of a directory (sorted).
pub fn manifest(dir: &Path) -> Result<BTreeMap<String, serde_json::Value>> {
    let mut out = BTreeMap::new();
    let mut files: Vec<_> = std::fs::read_dir(dir)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "jsonl"))
        .collect();
    files.sort();
    for f in files {
        let n = std::fs::read_to_string(&f)?
            .lines()
            .filter(|l| !l.trim().is_empty())
            .count();
        out.insert(
            f.file_name().unwrap().to_string_lossy().to_string(),
            serde_json::json!({"records": n, "sha256": sha256_file(&f)?}),
        );
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn legacy_round_trip() {
        let d = DecisionRecord {
            src: "x".into(),
            t: "choice".into(),
            ins: "q".into(),
            opts: vec!["billing: a charge".into(), "up".into()],
            state: "s".into(),
            target: vec![1.0, 0.0],
            split: "train".into(),
        };
        let r = S5Record::from_decision(&d, "u", "");
        assert_eq!(r.opts_kv[1], ("up".into(), "".into()));
        assert_eq!(r.to_decision(), d);
        let n = DecisionRecord {
            t: "noul".into(),
            opts: vec!["false: no".into(), "true".into()],
            ..d.clone()
        };
        let r = S5Record::from_decision(&n, "u", "");
        assert_eq!(
            r.opts_kv,
            [("false".into(), "no".into()), ("true".into(), "".into())]
        );
        let s = DecisionRecord {
            t: "score".into(),
            opts: vec!["level 0: low".into(), "level 1: high".into()],
            ..d
        };
        assert_eq!(S5Record::from_decision(&s, "u", "").to_decision(), s);
    }

    #[test]
    fn exclusions_and_splits() {
        assert!(!allowed("tasksource:amazon_polarity/amazon_polarity"));
        assert!(!allowed("calibrated:mteb/amazon_massive_intent"));
        assert!(!allowed("tasksource:slurp/intent"));
        assert!(!allowed("tasksource:jigsaw_toxicity"));
        assert!(allowed("tasksource:civil_comments/insult_share"));
        assert!(allowed("banking77"));
        let mut c = BTreeMap::new();
        for i in 0..10_000 {
            *c.entry(split_of(&i.to_string(), "t", 0.1, 0.1))
                .or_insert(0) += 1;
        }
        assert!((800..1200).contains(&c["val"]) && (800..1200).contains(&c["cal"]));
        assert_eq!(split_of("a", "t", 0.1, 0.1), split_of("a", "t", 0.1, 0.1));
    }
}
