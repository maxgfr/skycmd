//! Teacher soft labels of decision records through a GLM endpoint, for the s6 models
//! (`docs/s6-gen-prereg.md`, amendment 1). The teacher is GLM-5.3-flash, an open-weight model
//! (MIT, `zai-org/GLM-5.3-Flash`), called remotely through the z.ai API, so labelling costs no
//! local compute.
//!
//! `sky-data teacher label-decisions` sends every train record of `data/decisions` (or, with
//! `per_source`, a seeded sample per source family), never the benchmark or held-out sources nor
//! GLM-generated records, in batches, and asks for one probability per option. Replies are
//! appended to a JSONL file, so an interrupted run resumes where it stopped; several workers can
//! call the endpoint in parallel.
//!
//! `sky-data teacher mix-labels` keeps a label only when the teacher's argmax is the gold one
//! (the agreement filter of s5) and writes `target = alpha * gold + (1 - alpha) * teacher` as
//! `src = glm:<src>` train records, plus the agreement rate per source.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::record::{argmax, read_jsonl, write_jsonl, DecisionRecord};
use crate::rng::{hash64, Rng};
use crate::s4::EXCLUDED;
use crate::teacher::{BudgetExhausted, Teacher, TeacherConfig};

/// Most options a labelled question may have.
pub const MAX_OPTIONS: usize = 12;

/// Failed batches in a row after which a worker stops.
pub const MAX_FAILED_IN_A_ROW: usize = 3;

/// One teacher label: a probability per option, in record order.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct Label {
    pub id: String,
    pub src: String,
    pub probs: Vec<f32>,
    pub model: String,
}

#[derive(Clone, Debug)]
pub struct LabelOptions {
    /// Directories (or files) of decision records.
    pub from: Vec<PathBuf>,
    pub per_source: usize,
    /// The state is cut to this many characters in the prompt.
    pub max_state_chars: usize,
    /// Parallel workers, each with its own share of the call budget.
    pub jobs: usize,
    pub seed: u64,
}

/// Stable id of a record: its source and a hash of its content.
pub fn record_id(r: &DecisionRecord) -> String {
    let key = format!(
        "{}\u{1}{}\u{1}{}\u{1}{}\u{1}{}",
        r.src,
        r.t,
        r.ins,
        r.opts.join("\u{2}"),
        r.state
    );
    format!("{}-{:016x}", r.src, hash64(key.as_bytes()))
}

/// Whether a record may be sent to the teacher: a train record with 2 to `MAX_OPTIONS` options,
/// from a source that is neither held out nor GLM-generated.
pub fn eligible(r: &DecisionRecord) -> bool {
    r.split == "train"
        && !EXCLUDED.contains(&r.src.as_str())
        && !r.src.starts_with("para:")
        && !r.src.starts_with("glm:")
        && (2..=MAX_OPTIONS).contains(&r.opts.len())
        && r.target.len() == r.opts.len()
}

fn files(from: &[PathBuf]) -> Result<Vec<PathBuf>> {
    let mut out = Vec::new();
    for p in from {
        if p.is_dir() {
            let mut v: Vec<PathBuf> = fs::read_dir(p)
                .with_context(|| format!("read {}", p.display()))?
                .filter_map(|e| e.ok().map(|e| e.path()))
                .filter(|f| f.extension().is_some_and(|x| x == "jsonl"))
                .collect();
            v.sort();
            out.extend(v);
        } else {
            out.push(p.clone());
        }
    }
    Ok(out)
}

/// Every eligible record of `from`, in file order.
pub fn eligible_records(from: &[PathBuf]) -> Result<Vec<DecisionRecord>> {
    let mut out = Vec::new();
    for f in files(from)? {
        let recs: Vec<DecisionRecord> = read_jsonl(&f)?;
        out.extend(recs.into_iter().filter(eligible));
    }
    Ok(out)
}

/// The family of a source, the part before `:` (`tasksource:boolq` -> `tasksource`), so that a
/// set with hundreds of sub-sources weighs one share like the others.
pub fn family(src: &str) -> &str {
    src.split(':').next().unwrap_or(src)
}

/// Every eligible record (`per_source` 0), or up to `per_source` per source family drawn with a
/// seed per family; shuffled either way, so that an interrupted run has covered every source.
pub fn select(o: &LabelOptions) -> Result<Vec<DecisionRecord>> {
    let mut by_src: BTreeMap<String, Vec<DecisionRecord>> = BTreeMap::new();
    for r in eligible_records(&o.from)? {
        by_src
            .entry(family(&r.src).to_string())
            .or_default()
            .push(r);
    }
    let mut out = Vec::new();
    for (src, mut v) in by_src {
        if o.per_source > 0 {
            Rng::derived(o.seed, &format!("distill/{src}")).shuffle(&mut v);
            v.truncate(o.per_source);
        }
        out.extend(v);
    }
    Rng::derived(o.seed, "distill/order").shuffle(&mut out);
    Ok(out)
}

pub const LABEL_SYSTEM: &str = "You are a careful annotator of multiple-choice decision questions. You reply with one JSON object and nothing else: no prose, no Markdown.";

fn cut(s: &str, n: usize) -> String {
    if s.chars().count() <= n {
        s.to_string()
    } else {
        let mut t: String = s.chars().take(n).collect();
        t.push_str(" […]");
        t
    }
}

pub fn label_prompt(batch: &[(String, DecisionRecord)], max_state_chars: usize) -> String {
    let items: Vec<Value> = batch
        .iter()
        .map(|(id, r)| {
            json!({"id": id, "type": r.t, "instructions": r.ins,
                   "state": cut(&r.state, max_state_chars), "options": r.opts})
        })
        .collect();
    format!(
        "For each item below, give the probability that each option is the right answer.\n\
         Types: noul = judge whether the statement in the instructions holds for the state (the options say false or true); \
         choice = the option that best fits the state and the instructions; score = the level of the ordered scale that fits.\n\
         Rules: one probability per option, in the order given, each between 0 and 1, summing to 1. \
         Put most of the mass on the best answer and spread it only where you are genuinely unsure.\n\
         Reply with exactly: {{\"items\": [{{\"id\": \"<id>\", \"p\": [numbers]}}, ...]}} with one item per input id.\n\
         Items:\n{}",
        serde_json::to_string_pretty(&items).unwrap_or_default()
    )
}

/// Valid labels of a reply: a known id, one finite non-negative number per option, a positive
/// sum (normalized to 1).
pub fn parse_labels(reply: &Value, batch: &[(String, DecisionRecord)], model: &str) -> Vec<Label> {
    let want: HashMap<&str, &DecisionRecord> = batch.iter().map(|(i, r)| (i.as_str(), r)).collect();
    let mut seen = HashSet::new();
    let mut out = Vec::new();
    let Some(items) = reply.get("items").and_then(Value::as_array) else {
        return out;
    };
    for it in items {
        let Some(id) = it.get("id").and_then(Value::as_str) else {
            continue;
        };
        let Some(r) = want.get(id) else { continue };
        let Some(p) = it.get("p").and_then(Value::as_array) else {
            continue;
        };
        let p: Option<Vec<f32>> = p.iter().map(|x| x.as_f64().map(|v| v as f32)).collect();
        let Some(p) = p else { continue };
        let s: f32 = p.iter().sum();
        if p.len() != r.opts.len() || p.iter().any(|x| !x.is_finite() || *x < 0.0) || s <= 0.0 {
            continue;
        }
        if seen.insert(id.to_string()) {
            out.push(Label {
                id: id.to_string(),
                src: r.src.clone(),
                probs: p.iter().map(|x| x / s).collect(),
                model: model.to_string(),
            });
        }
    }
    out
}

/// Totals of a labelling run.
#[derive(Default, Debug)]
pub struct LabelStats {
    pub written: usize,
    pub calls: usize,
    pub failed_batches: usize,
    pub input_tokens: u64,
    pub output_tokens: u64,
}

/// Labels the selected records not yet in `out`, `jobs` workers in parallel, each with
/// `max_calls / jobs` calls.
pub fn label_decisions(cfg: &TeacherConfig, o: &LabelOptions, out: &Path) -> Result<LabelStats> {
    let done: HashSet<String> = if out.exists() {
        read_jsonl::<Label>(out)?
            .into_iter()
            .map(|l| l.id)
            .collect()
    } else {
        HashSet::new()
    };
    let todo: Vec<(String, DecisionRecord)> = select(o)?
        .into_iter()
        .map(|r| (record_id(&r), r))
        .filter(|(id, _)| !done.contains(id))
        .collect();
    eprintln!(
        "[distill] {} records to label ({} already labelled), {} workers",
        todo.len(),
        done.len(),
        o.jobs
    );
    eprintln!(
        "[distill] {} calls of {} records needed",
        todo.len().div_ceil(cfg.batch),
        cfg.batch
    );
    if cfg.max_calls == 0 || todo.is_empty() {
        return Ok(LabelStats::default());
    }
    if let Some(d) = out.parent() {
        fs::create_dir_all(d)?;
    }
    let file = Mutex::new(OpenOptions::new().create(true).append(true).open(out)?);
    let stats = Mutex::new(LabelStats::default());
    let batches: Vec<&[(String, DecisionRecord)]> = todo.chunks(cfg.batch).collect();
    let jobs = o.jobs.max(1);
    let per_job = cfg.max_calls.div_ceil(jobs);
    std::thread::scope(|s| {
        for j in 0..jobs {
            let (file, stats, batches) = (&file, &stats, &batches);
            let mut c = cfg.clone();
            c.max_calls = per_job;
            s.spawn(move || {
                let mut t = Teacher::new(c);
                let (mut written, mut failed, mut streak) = (0, 0, 0);
                for b in batches.iter().skip(j).step_by(jobs) {
                    match t.call_json(LABEL_SYSTEM, &label_prompt(b, o.max_state_chars)) {
                        Ok(reply) => {
                            let labels = parse_labels(&reply, b, &t.cfg.model);
                            let mut f = file.lock().unwrap();
                            for l in &labels {
                                if let Ok(line) = serde_json::to_string(l) {
                                    let _ = writeln!(f, "{line}");
                                }
                            }
                            let _ = f.flush();
                            written += labels.len();
                            streak = 0;
                            if labels.len() < b.len() {
                                eprintln!(
                                    "[distill] worker {j}: {} / {} valid",
                                    labels.len(),
                                    b.len()
                                );
                            }
                        }
                        Err(e) if e.is::<BudgetExhausted>() => break,
                        Err(e) => {
                            failed += 1;
                            streak += 1;
                            eprintln!("[distill] worker {j}: batch failed: {e:#}");
                            // A spent usage window (z.ai resets every 5 h) or a network outage:
                            // stop instead of burning the budget; a rerun resumes.
                            if streak >= MAX_FAILED_IN_A_ROW {
                                eprintln!("[distill] worker {j}: {streak} failed batches in a row, stopping (rerun to resume)");
                                break;
                            }
                        }
                    }
                }
                let mut st = stats.lock().unwrap();
                st.written += written;
                st.failed_batches += failed;
                st.calls += t.calls;
                st.input_tokens += t.input_tokens;
                st.output_tokens += t.output_tokens;
            });
        }
    });
    Ok(stats.into_inner().unwrap())
}

/// Per source: labelled records and how many agree with the gold argmax.
pub type Agreement = BTreeMap<String, (usize, usize)>;

/// The gold / teacher mix of the labelled records that agree with their gold label, written to
/// `out/glm.jsonl` (`src = glm:<src>`, train split), and the agreement per source.
pub fn mix_labels(labels: &Path, from: &[PathBuf], out: &Path, alpha: f32) -> Result<Agreement> {
    let labels: HashMap<String, Label> = read_jsonl::<Label>(labels)?
        .into_iter()
        .map(|l| (l.id.clone(), l))
        .collect();
    let mut agree = Agreement::new();
    let mut recs = Vec::new();
    for r in eligible_records(from)? {
        let Some(l) = labels.get(&record_id(&r)) else {
            continue;
        };
        let e = agree.entry(r.src.clone()).or_default();
        e.0 += 1;
        if argmax(&l.probs) != argmax(&r.target) {
            continue;
        }
        e.1 += 1;
        let target = r
            .target
            .iter()
            .zip(&l.probs)
            .map(|(g, t)| alpha * g + (1.0 - alpha) * t)
            .collect();
        recs.push(DecisionRecord {
            src: format!("glm:{}", r.src),
            target,
            ..r
        });
    }
    fs::create_dir_all(out)?;
    write_jsonl(&out.join("glm.jsonl"), &recs)?;
    let table: BTreeMap<&String, Value> = agree
        .iter()
        .map(|(s, (n, a))| {
            (
                s,
                json!({"labelled": n, "agree": a, "rate": *a as f64 / (*n).max(1) as f64}),
            )
        })
        .collect();
    fs::write(
        out.join("agreement.json"),
        serde_json::to_string_pretty(&table)? + "\n",
    )?;
    Ok(agree)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rec(src: &str, target: &[f32], split: &str) -> DecisionRecord {
        DecisionRecord {
            src: src.into(),
            t: "choice".into(),
            ins: "pick".into(),
            opts: (0..target.len()).map(|i| format!("o{i}")).collect(),
            state: "s".into(),
            target: target.to_vec(),
            split: split.into(),
        }
    }

    #[test]
    fn eligibility() {
        assert!(eligible(&rec("boolq", &[0.0, 1.0], "train")));
        assert!(!eligible(&rec("boolq", &[0.0, 1.0], "val")));
        assert!(eligible(&rec("boolq", &[0.3, 0.7], "train")));
        assert!(!eligible(&rec(
            "typed_decisions_bench",
            &[0.0, 1.0],
            "train"
        )));
        assert!(!eligible(&rec("banking77", &[0.0, 1.0], "train")));
        assert!(!eligible(&rec("para:boolq", &[0.0, 1.0], "train")));
        assert!(!eligible(&rec("boolq", &[1.0], "train")));
    }

    #[test]
    fn parse_and_mix() {
        let r = rec("boolq", &[0.0, 1.0, 0.0], "train");
        let id = record_id(&r);
        let batch = vec![(id.clone(), r.clone())];
        let reply = json!({"items": [
            {"id": id, "p": [1, 6, 1]},
            {"id": id, "p": [1, 0, 0]},
            {"id": "unknown", "p": [1, 0, 0]},
        ]});
        let l = parse_labels(&reply, &batch, "glm");
        assert_eq!(l.len(), 1);
        assert_eq!(l[0].probs, vec![0.125, 0.75, 0.125]);
        // Wrong length, negative or all-zero probabilities are dropped.
        for p in [json!([0.5, 0.5]), json!([-1, 1, 1]), json!([0, 0, 0])] {
            let bad = json!({"items": [{"id": batch[0].0, "p": p}]});
            assert!(parse_labels(&bad, &batch, "glm").is_empty());
        }

        let d = std::env::temp_dir().join(format!("sky-data-distill-{}", std::process::id()));
        let other = DecisionRecord {
            state: "another state".into(),
            ..rec("boolq", &[1.0, 0.0, 0.0], "train")
        };
        write_jsonl(&d.join("decisions/boolq.jsonl"), [&r, &other]).unwrap();
        let disagree = Label {
            id: record_id(&other),
            src: "boolq".into(),
            probs: vec![0.1, 0.8, 0.1],
            model: "glm".into(),
        };
        write_jsonl(&d.join("labels.jsonl"), [&l[0], &disagree]).unwrap();
        let out = d.join("distill");
        let a = mix_labels(&d.join("labels.jsonl"), &[d.join("decisions")], &out, 0.5).unwrap();
        assert_eq!(a["boolq"], (2, 1));
        let mixed: Vec<DecisionRecord> = read_jsonl(&out.join("glm.jsonl")).unwrap();
        assert_eq!(mixed.len(), 1);
        assert_eq!(mixed[0].src, "glm:boolq");
        assert_eq!(mixed[0].target, vec![0.0625, 0.875, 0.0625]);
        assert_eq!(mixed[0].split, "train");
        fs::remove_dir_all(d).ok();
    }
}
