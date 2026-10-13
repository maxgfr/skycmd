//! `sky-data s5-build`: every s5 data file, in order, deterministic from the seed:
//!
//! 1. the evaluation sets (`<out>/eval/`, see [`crate::s5::evalsets`]);
//! 2. the corpus of licensed decision sets and the typed-decisions train split;
//! 3. the generators (rules, tone, buried facts) in each language;
//! 4. the coherence families (S, N, CH) of the typed-decisions train states;
//! 5. the paraphrase and distractor pairs of typed-decisions questions;
//! 6. decontamination of 2–5 against the evaluation sets and the demo cases;
//! 7. the class-balance gate (rebalance, then fail if a class still exceeds 60 %), except for
//!    the coherence families, imbalanced by construction;
//! 8. `<out>/train/*.jsonl` and `<out>/MANIFEST.json` (counts, sha256, overlap, stats).
//!
//! Filler and training distractors come from `fineweb_edu` / `fineweb2_fr`; the evaluation
//! distractors (JevOut-lite) from `simplestories` / `tinystories_fr`, which training never
//! reads.

use std::collections::BTreeMap;
use std::path::PathBuf;

use anyhow::{bail, Context, Result};
use serde_json::json;

use crate::record::{read_jsonl, MlmRecord};
use crate::rng::Rng;
use crate::s5::{
    coherence, corpus, evalsets, gen, manifest, overlap, read, stats, write, S5Record,
};

#[derive(Clone, Debug)]
pub struct BuildOptions {
    pub decisions: PathBuf,
    pub text: PathBuf,
    pub src: PathBuf,
    pub defs: PathBuf,
    pub demo: PathBuf,
    pub out: PathBuf,
    pub seed: u64,
    /// `en`, `fr` or `both`.
    pub langs: Vec<String>,
    pub per_source: usize,
    pub per_eval_set: usize,
    pub n_rules: usize,
    pub n_tone: usize,
    pub n_buried: usize,
    pub n_pairs: usize,
}

/// Sentences of 40 to 220 characters of a text corpus (`data/text/<name>.jsonl`), at most `n`.
pub fn sentences(path: &std::path::Path, n: usize, seed: u64) -> Result<Vec<String>> {
    if !path.exists() {
        return Ok(Vec::new());
    }
    let recs: Vec<MlmRecord> = read_jsonl(path)?;
    let mut out = Vec::new();
    for r in recs.iter().take(20_000) {
        for s in r.text.split(['\n', '.']) {
            let s = s.trim();
            let len = s.chars().count();
            if (40..=220).contains(&len) && !s.contains("http") {
                out.push(format!("{s}."));
            }
        }
    }
    let mut rng = Rng::derived(seed, &path.display().to_string());
    rng.shuffle(&mut out);
    out.truncate(n);
    Ok(out)
}

pub fn build(o: &BuildOptions) -> Result<serde_json::Value> {
    // 1. Evaluation sets.
    let eval_dir = o.out.join("eval");
    let eval_counts = evalsets::build(&evalsets::EvalOptions {
        src: o.src.clone(),
        decisions: o.decisions.clone(),
        defs: o.defs.clone(),
        out: eval_dir.clone(),
        text: o.text.clone(),
        paraphrases: o
            .defs
            .parent()
            .map(|d| d.join("paraphrases.json"))
            .unwrap_or_else(|| "eval/s5/paraphrases.json".into()),
        per_set: o.per_eval_set,
        max_options: 8,
        seed: o.seed,
    })?;
    eprintln!("[s5] eval sets: {eval_counts:?}");
    let mut refs = overlap::Index::new();
    for f in std::fs::read_dir(&eval_dir)?.filter_map(|e| e.ok().map(|e| e.path())) {
        if f.extension().is_some_and(|x| x == "jsonl") {
            for r in read(&f)? {
                refs.add(&r.state);
            }
        }
    }
    let demo = read(&o.demo).with_context(|| format!("{}", o.demo.display()))?;
    for r in &demo {
        refs.add(&r.state);
    }

    let mut groups: BTreeMap<&str, Vec<S5Record>> = BTreeMap::new();
    let mut source_counts = BTreeMap::new();
    for lang in &o.langs {
        // 2. Corpus.
        let (c, counts) = corpus::build(&corpus::CorpusOptions {
            decisions: o.decisions.clone(),
            seed: o.seed,
            per_source: o.per_source,
            p_none: 0.25,
            lang: lang.clone(),
        })?;
        source_counts.insert(format!("corpus_{lang}"), counts);
        // 3. Generators.
        let (filler_file, _) = match lang.as_str() {
            "fr" => ("fineweb2_fr.jsonl", "tinystories_fr.jsonl"),
            _ => ("fineweb_edu.jsonl", "simplestories.jsonl"),
        };
        let filler = sentences(&o.text.join(filler_file), 20_000, o.seed)?;
        let g = groups.entry("gen").or_default();
        g.extend(gen::rules(o.n_rules, lang, o.seed));
        g.extend(gen::tone(o.n_tone, lang, o.seed));
        g.extend(gen::buried(
            o.n_buried,
            lang,
            &filler[..filler.len() / 2],
            o.seed,
        ));
        // 4–5. Coherence families and pairs of the typed-decisions train split (English only:
        // it has no French part).
        if lang == "en" {
            let typed: Vec<S5Record> = c
                .iter()
                .filter(|r| r.src == "typed_decisions_train")
                .cloned()
                .collect();
            if typed.is_empty() {
                bail!("no typed_decisions_train records: run `sky-data fetch-decisions --only typed_decisions_train`");
            }
            groups
                .entry("coherence")
                .or_default()
                .extend(coherence::families(&typed, false));
            let p = groups.entry("pairs").or_default();
            p.extend(gen::paraphrase_pairs(&typed, o.n_pairs, o.seed));
            p.extend(gen::distractor_pairs(
                &typed,
                &filler[filler.len() / 2..],
                o.n_pairs,
                o.seed,
            ));
        }
        groups.entry("corpus").or_default().extend(c);
    }

    // 6. Decontamination, 7. balance.
    let mut report = BTreeMap::new();
    let mut gate = Vec::new();
    std::fs::create_dir_all(o.out.join("train"))?;
    let mut written = BTreeMap::new();
    for (name, recs) in groups {
        let n0 = recs.len();
        let (kept, dropped) = overlap::decontaminate(recs, &refs);
        // A coherence family is imbalanced by construction (S_k holds for one option in K):
        // it is reported, not rebalanced, and not gated.
        let family = name == "coherence";
        let balanced = if family {
            kept
        } else {
            stats::rebalance(kept, 200, o.seed)
        };
        let left = if family {
            Vec::new()
        } else {
            stats::violations(&balanced, 200)
        };
        for v in &left {
            gate.push(
                json!({"group": name, "src": v.0, "t": v.1, "class": v.2, "share": v.3, "n": v.4}),
            );
        }
        report.insert(
            name.to_string(),
            json!({"before": n0, "dropped_overlap": dropped, "after_balance": balanced.len()}),
        );
        written.insert(
            name.to_string(),
            write(
                &o.out.join("train").join(format!("{name}.jsonl")),
                &balanced,
            )?,
        );
    }
    if !gate.is_empty() {
        bail!(
            "class-balance gate failed: {}",
            serde_json::to_string_pretty(&gate)?
        );
    }
    // Teacher sources we hold locally (declared by oscar / laya): their overlap with the
    // evaluation sets is reported, not fixed.
    let mut teacher = overlap::Index::new();
    for f in [
        "banking77.jsonl",
        "mnli.jsonl",
        "typed_decisions_train.jsonl",
    ] {
        let p = o.decisions.join(f);
        if p.exists() {
            for d in read_jsonl::<crate::record::DecisionRecord>(&p)? {
                teacher.add(&d.state);
            }
        }
    }
    let mut eval_vs_teacher = BTreeMap::new();
    for f in eval_counts.keys() {
        let recs = read(&eval_dir.join(f))?;
        for (src, (hit, n)) in overlap::overlap_rates(&recs, &teacher) {
            eval_vs_teacher.insert(src, json!({"overlapping": hit, "of": n}));
        }
    }
    let m = json!({
        "seed": o.seed,
        "langs": o.langs,
        "train": manifest(&o.out.join("train"))?,
        "eval": manifest(&eval_dir)?,
        "sources": source_counts,
        "decontamination": report,
        "eval_vs_teacher_sources": {
            "checked": ["banking77", "mnli", "typed_decisions_train"],
            "not_checkable": ["agnews", "emotion (dair-ai)", "sst5", "translated multilingual data (oscar)", "undeclared laya training data"],
            "rates": eval_vs_teacher,
        },
        "demo_cases": demo.len(),
    });
    std::fs::write(
        o.out.join("MANIFEST.json"),
        serde_json::to_string_pretty(&m)? + "\n",
    )?;
    Ok(m)
}
