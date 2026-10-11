//! GLM teacher data integration: the drone-order `action` records as a decision source
//! (`glm_orders`) and the `augment-paraphrases` command, which rewrites question
//! instructions with the GLM paraphrases of `data/teacher/paraphrases.jsonl`.

use std::collections::{BTreeMap, HashMap};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};

use crate::decisions::{fair_quotas, Ctx, Item};
use crate::record::{read_jsonl, validate_record, write_jsonl, DecisionRecord};
use crate::rng::{hash64, Rng};
use crate::teacher::Paraphrase;

/// `data/teacher/orders_decisions.jsonl` (already SPEC records, written by
/// `sky-data teacher drone-orders`) → `glm_orders` records grouped by order text.
pub fn load_glm_orders(cx: &Ctx) -> Result<Vec<Item>> {
    let path = cx.teacher.join("orders_decisions.jsonl");
    let recs: Vec<DecisionRecord> =
        read_jsonl(&path).with_context(|| format!("GLM orders: {path:?}"))?;
    Ok(recs
        .into_iter()
        .map(|mut r| {
            r.src = "glm_orders".into();
            r.split = "train".into();
            Item {
                group: format!("{:x}", hash64(r.state.trim().to_lowercase().as_bytes())),
                rec: r,
            }
        })
        .collect())
}

/// Key of an instruction: a paraphrase rewrites the instructions of a question type, and the
/// options are only context for the teacher, so it applies to every record with the same
/// `(t, ins)` (e.g. all ARC records share one instruction but each has its own options).
pub fn instruction_key(t: &str, ins: &str) -> u64 {
    hash64(format!("{t}\u{1}{}", ins.trim()).as_bytes())
}

#[derive(Clone, Debug)]
pub struct AugmentOptions {
    pub paraphrases: PathBuf,
    /// Directories (or files) of decision JSONL to augment.
    pub inputs: Vec<PathBuf>,
    pub out: PathBuf,
    pub cap: usize,
    /// Candidate records kept per question before the final draw.
    pub per_question: usize,
    pub seed: u64,
}

#[derive(Clone, Debug, Default)]
pub struct AugmentReport {
    pub paraphrased_instructions: usize,
    pub matched_instructions: usize,
    pub matched_records: usize,
    pub written: usize,
    pub by_src: BTreeMap<String, usize>,
}

fn jsonl_inputs(inputs: &[PathBuf], out: &Path) -> Vec<PathBuf> {
    let mut files = Vec::new();
    for p in inputs {
        if p.is_dir() {
            let mut v: Vec<PathBuf> = std::fs::read_dir(p)
                .map(|rd| {
                    rd.filter_map(|e| e.ok().map(|e| e.path()))
                        .filter(|f| f.extension().is_some_and(|e| e == "jsonl"))
                        .collect()
                })
                .unwrap_or_default();
            v.sort();
            files.extend(v);
        } else if p.exists() {
            files.push(p.clone());
        }
    }
    let out = out.canonicalize().unwrap_or_else(|_| out.to_path_buf());
    files.retain(|f| f.canonicalize().unwrap_or_else(|_| f.clone()) != out);
    files
}

/// Writes `cap` records whose instructions are replaced by a GLM paraphrase (EN or FR) of
/// the original instructions (matched by [`instruction_key`]); options, state, target and
/// split are unchanged (the split is
/// kept so a validation state never leaks into training). Records are drawn fairly across
/// questions; `src` becomes `para:<src>`.
pub fn augment_paraphrases(opts: &AugmentOptions) -> Result<AugmentReport> {
    let paras: Vec<Paraphrase> = read_jsonl(&opts.paraphrases)
        .with_context(|| format!("paraphrases {:?}", opts.paraphrases))?;
    let mut by_key: HashMap<u64, Vec<String>> = HashMap::new();
    for p in &paras {
        let texts = by_key.entry(instruction_key(&p.t, &p.ins)).or_default();
        for s in p.en.iter().chain(&p.fr).map(|s| s.trim()) {
            if !s.is_empty() && s != p.ins.trim() && !texts.iter().any(|t| t == s) {
                texts.push(s.to_string());
            }
        }
    }
    by_key.retain(|_, v| !v.is_empty());
    if by_key.is_empty() {
        bail!("no usable paraphrases in {:?}", opts.paraphrases);
    }
    let mut report = AugmentReport {
        paraphrased_instructions: by_key.len(),
        ..Default::default()
    };
    // Reservoir of candidate records per question (deterministic).
    let mut rng = Rng::derived(opts.seed, "augment-paraphrases");
    let mut pools: BTreeMap<u64, (usize, Vec<DecisionRecord>)> = BTreeMap::new();
    for f in jsonl_inputs(&opts.inputs, &opts.out) {
        for line in BufReader::new(File::open(&f)?).lines() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }
            let Ok(r) = serde_json::from_str::<DecisionRecord>(&line) else {
                continue;
            };
            if r.split == "test" {
                continue;
            }
            let key = instruction_key(&r.t, &r.ins);
            if !by_key.contains_key(&key) {
                continue;
            }
            report.matched_records += 1;
            let (seen, pool) = pools.entry(key).or_default();
            *seen += 1;
            if pool.len() < opts.per_question {
                pool.push(r);
            } else {
                let j = rng.below(*seen);
                if j < opts.per_question {
                    pool[j] = r;
                }
            }
        }
    }
    report.matched_instructions = pools.len();
    let counts: BTreeMap<String, usize> = pools
        .iter()
        .map(|(k, (_, v))| (format!("{k:016x}"), v.len()))
        .collect();
    let quotas = fair_quotas(&counts, opts.cap);
    let mut out = Vec::new();
    for (key, (_, mut pool)) in pools {
        rng.shuffle(&mut pool);
        let texts = &by_key[&key];
        for mut r in pool.into_iter().take(quotas[&format!("{key:016x}")]) {
            r.ins = texts[rng.below(texts.len())].clone();
            r.src = format!("para:{}", r.src);
            if validate_record(&r).is_ok() {
                *report.by_src.entry(r.src.clone()).or_default() += 1;
                out.push(r);
            }
        }
    }
    rng.shuffle(&mut out);
    report.written = write_jsonl(&opts.out, &out)?;
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::record::Question;

    #[test]
    fn paraphrase_augmentation() {
        let dir = std::env::temp_dir().join(format!("sky-data-aug-{}", std::process::id()));
        std::fs::create_dir_all(dir.join("dec")).unwrap();
        let q = Question::noul("Is the drone low?");
        let mut recs = Vec::new();
        for i in 0..50 {
            let mut r = q.record("game:flappy", format!("alt {i}"), vec![0.0, 1.0]);
            if i % 10 == 0 {
                r.split = "test".into();
            }
            recs.push(r);
        }
        let other = Question::noul("Unrelated?").record("x", "s", vec![1.0, 0.0]);
        recs.push(other);
        write_jsonl(&dir.join("dec/a.jsonl"), &recs).unwrap();
        let para = Paraphrase {
            id: "game:flappy-0123456789abcdef".into(),
            t: "noul".into(),
            ins: "Is the drone low?".into(),
            en: vec!["Is the drone flying low?".into()],
            fr: vec!["Le drone vole-t-il bas ?".into()],
        };
        write_jsonl(&dir.join("p.jsonl"), [&para]).unwrap();
        let opts = AugmentOptions {
            paraphrases: dir.join("p.jsonl"),
            inputs: vec![dir.join("dec")],
            out: dir.join("dec/paraphrased.jsonl"),
            cap: 30,
            per_question: 64,
            seed: 1,
        };
        let rep = augment_paraphrases(&opts).unwrap();
        assert_eq!(rep.matched_records, 45);
        assert_eq!(rep.written, 30);
        let out: Vec<DecisionRecord> = read_jsonl(&opts.out).unwrap();
        assert!(out.iter().all(|r| r.src == "para:game:flappy"
            && r.ins != "Is the drone low?"
            && r.split != "test"));
        assert!(out
            .iter()
            .all(|r| r.target == vec![0.0, 1.0] && r.opts == recs[0].opts));
        // Re-running ignores its own output file.
        assert_eq!(augment_paraphrases(&opts).unwrap().matched_records, 45);
        std::fs::remove_dir_all(dir).ok();
    }
}
