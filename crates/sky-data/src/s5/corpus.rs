//! The s5 training corpus from the licensed decision sets of `data/decisions` (the s4 corpus,
//! without its Qwen-scored part and without the excluded sources), and the typed-decisions
//! train split.
//!
//! - every allowed source is capped (`per_source`, seeded sample), its test split dropped;
//! - splits are reassigned by state: `val` 5 %, `cal` 5 %, `train` the rest;
//! - on 25 % of the `choice` questions, a "none of the above" option with wording aligned on
//!   the task is added; on half of those (when the gold option is clear and three options
//!   remain), the gold option is removed and "none of the above" becomes the answer.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

use crate::record::{read_jsonl, DecisionRecord};
use crate::rng::Rng;
use crate::s5::{allowed, split_of, uid_of, S5Record};

#[derive(Clone, Debug)]
pub struct CorpusOptions {
    /// `data/decisions`.
    pub decisions: PathBuf,
    pub seed: u64,
    pub per_source: usize,
    /// Fraction of choice questions given a "none of the above" option.
    pub p_none: f64,
    /// Keep only this language (`en`, `fr`; empty: both).
    pub lang: String,
}

/// Language of a source.
pub fn lang_of(src: &str) -> &'static str {
    let s = src.to_ascii_lowercase();
    if s.contains("allocine") || s.ends_with("_fr") || s.contains("pawsx_fr") || s.contains("/fr") {
        "fr"
    } else {
        "en"
    }
}

/// A "none of the above" option worded for the task of `ins` (`(id, definition)`).
pub fn none_option(ins: &str, lang: &str) -> (String, String) {
    let i = ins.to_ascii_lowercase();
    let what = [
        ("intent", "intents", "intentions"),
        ("topic", "topics", "thèmes"),
        ("categor", "categories", "catégories"),
        ("emotion", "emotions", "émotions"),
        ("sentiment", "sentiments", "sentiments"),
        ("tool", "tools", "outils"),
        ("action", "actions", "actions"),
        ("label", "labels", "étiquettes"),
    ]
    .iter()
    .find(|(k, _, _)| i.contains(k));
    if lang == "fr" {
        let w = what.map_or("options proposées", |w| w.2);
        (
            "aucune".into(),
            format!("Aucune des {w} de la liste ne convient."),
        )
    } else {
        let w = what.map_or("listed options", |w| w.1);
        (
            "none_of_these".into(),
            format!("None of the listed {w} applies."),
        )
    }
}

/// Adds a "none of the above" option to a choice record, and removes the gold option when
/// `remove_gold` (it must be clear, `p ≥ 0.8`, and leave two other options).
pub fn with_none(r: &mut S5Record, remove_gold: bool) -> bool {
    if r.t != "choice" || r.opts_kv.len() >= 20 {
        return false;
    }
    let none = none_option(&r.ins, &r.lang);
    if r.opts_kv.iter().any(|(id, _)| id == &none.0) {
        return false;
    }
    let g = r.gold_index();
    let removed = remove_gold && r.target[g] >= 0.8 && r.opts_kv.len() >= 3;
    if removed {
        r.opts_kv.remove(g);
        r.target.remove(g);
        r.target.iter_mut().for_each(|x| *x = 0.0);
        r.target.push(1.0);
    } else {
        r.target.push(0.0);
    }
    r.opts_kv.push(none);
    let s: f32 = r.target.iter().sum();
    r.target.iter_mut().for_each(|x| *x /= s);
    r.src = format!("{}+none", r.src);
    true
}

fn jsonl_files(dir: &Path) -> Result<Vec<PathBuf>> {
    let mut v: Vec<PathBuf> = std::fs::read_dir(dir)
        .with_context(|| format!("reading {}", dir.display()))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "jsonl"))
        .collect();
    v.sort();
    Ok(v)
}

/// The corpus records and the count per source.
pub fn build(o: &CorpusOptions) -> Result<(Vec<S5Record>, BTreeMap<String, usize>)> {
    let mut by_src: BTreeMap<String, Vec<DecisionRecord>> = BTreeMap::new();
    for f in jsonl_files(&o.decisions)? {
        let recs: Vec<DecisionRecord> = read_jsonl(&f)?;
        for r in recs {
            if r.split == "test" || !allowed(&r.src) || r.opts.len() < 2 || r.opts.len() > 20 {
                continue;
            }
            if !o.lang.is_empty() && lang_of(&r.src) != o.lang {
                continue;
            }
            by_src.entry(r.src.clone()).or_default().push(r);
        }
    }
    let mut out = Vec::new();
    let mut counts = BTreeMap::new();
    for (src, mut recs) in by_src {
        let mut rng = Rng::derived(o.seed, &format!("s5/corpus/{src}"));
        rng.shuffle(&mut recs);
        // The typed-decisions train split is kept whole: it is the fitted bench's domain.
        let cap = if src == "typed_decisions_train" {
            usize::MAX
        } else {
            o.per_source
        };
        let mut n = 0;
        for d in recs.into_iter().take(cap) {
            let lang = lang_of(&src);
            let uid = uid_of(&[&src, &d.ins, &d.state, &d.opts.join("\u{1e}")]);
            let mut r = S5Record::from_decision(&d, &uid, if lang == "en" { "" } else { lang });
            r.split = split_of(&r.state, "corpus", 0.05, 0.05).to_string();
            if r.t == "choice" && rng.f64() < o.p_none {
                let remove = rng.f64() < 0.5;
                with_none(&mut r, remove);
            }
            if r.validate().is_ok() {
                out.push(r);
                n += 1;
            }
        }
        counts.insert(src, n);
    }
    Ok((out, counts))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rec(t: &str, kv: &[(&str, &str)], target: &[f32]) -> S5Record {
        S5Record {
            src: "x".into(),
            t: t.into(),
            ins: "What is the intent of the request?".into(),
            opts_kv: kv
                .iter()
                .map(|(a, b)| (a.to_string(), b.to_string()))
                .collect(),
            state: "s".into(),
            target: target.to_vec(),
            split: "train".into(),
            uid: "u".into(),
            lang: String::new(),
            gold: true,
        }
    }

    #[test]
    fn none_of_the_above() {
        let kv = [("a", "x"), ("b", "y"), ("c", "z")];
        let mut r = rec("choice", &kv, &[0.9, 0.1, 0.0]);
        assert!(with_none(&mut r, false));
        assert_eq!(r.opts_kv.len(), 4);
        assert_eq!(r.opts_kv[3].1, "None of the listed intents applies.");
        assert_eq!(r.target[3], 0.0);
        let mut r = rec("choice", &kv, &[0.9, 0.1, 0.0]);
        assert!(with_none(&mut r, true));
        assert_eq!(r.opts_kv.len(), 3);
        assert_eq!(r.target, vec![0.0, 0.0, 1.0]);
        assert!(r.opts_kv.iter().all(|(id, _)| id != "a"));
        let mut n = rec("noul", &[("false", ""), ("true", "")], &[0.5, 0.5]);
        assert!(!with_none(&mut n, true));
        assert_eq!(none_option("Quel est le thème ?", "fr").0, "aucune");
        assert_eq!(lang_of("allocine"), "fr");
        assert_eq!(lang_of("boolq"), "en");
    }
}
