//! `sky-data s5-mix`: training targets from the gold labels and the teachers' soft labels.
//!
//! 1. **Calibration per group** (`source / type / number of options`): for each teacher, a
//!    temperature and a per-option bias fitted on the group's gold `train` records
//!    ([`crate::s4::fit_calib`]; groups under 150 records get the teacher's global
//!    temperature).
//! 2. **Teacher choice per group**: the teacher with the lowest Brier score after calibration
//!    on the group's gold `val` records (ties: lower ECE); a group with no val record keeps the
//!    global best teacher.
//! 3. **Agreement filter**: a record takes the teacher's soft label only when the two teachers
//!    agree on the argmax, or the chosen teacher agrees with the gold argmax; otherwise it keeps
//!    its gold target.
//! 4. **Target** = `alpha · gold + (1 − alpha) · teacher`. Generated records (programmatic,
//!    exact labels, including the uniform targets of missing evidence) and coherence families
//!    keep their gold target; the second member of a pair takes the target of the first.

use std::collections::{BTreeMap, HashMap};
use std::path::PathBuf;

use anyhow::{bail, Result};
use serde_json::{json, Value};

use crate::record::read_jsonl;
use crate::s4::{fit_calib, softmax_t, Calib};
use crate::s5::{read, write, S5Record};

#[derive(Clone, Debug)]
pub struct MixOptions {
    /// `data/s5/train`.
    pub train: PathBuf,
    /// `data/s5/scored`.
    pub scored: PathBuf,
    /// Teacher names (file prefixes in `scored`).
    pub teachers: Vec<String>,
    /// `data/s5/mixed`.
    pub out: PathBuf,
    pub alpha: f32,
}

/// Minimum gold train records of a group calibration.
const MIN_GROUP: usize = 150;

/// The source of a record without the augmentation suffixes (`+none`, `+pair`, `+para`, ...).
pub fn base_src(src: &str) -> &str {
    src.split('+').next().unwrap_or(src)
}

fn group_of(r: &S5Record) -> String {
    format!("{}/{}/{}", base_src(&r.src), r.t, r.opts_kv.len())
}

fn argmax(v: &[f32]) -> usize {
    crate::record::argmax(v)
}

fn brier(p: &[f32], y: &[f32]) -> f64 {
    p.iter().zip(y).map(|(a, b)| ((a - b) as f64).powi(2)).sum()
}

/// Top-label ECE with 15 bins.
fn ece(items: &[(Vec<f32>, &[f32])]) -> f64 {
    let mut bins = vec![(0.0f64, 0.0f64, 0usize); 15];
    for (p, y) in items {
        let k = argmax(p);
        let conf = p[k] as f64;
        let b = ((conf * 15.0).ceil() as usize).clamp(1, 15) - 1;
        bins[b].0 += conf;
        bins[b].1 += f64::from(u8::from(k == argmax(y)));
        bins[b].2 += 1;
    }
    let n = items.len().max(1) as f64;
    bins.iter()
        .filter(|b| b.2 > 0)
        .map(|b| (b.2 as f64 / n) * ((b.0 - b.1) / b.2 as f64).abs())
        .sum()
}

/// `uid -> tlogits` of a teacher (every scored file of that teacher, native rendering).
fn teacher_logits(o: &MixOptions, teacher: &str) -> Result<HashMap<String, Vec<f32>>> {
    let mut out = HashMap::new();
    let prefix = format!("{teacher}.native.");
    for e in std::fs::read_dir(&o.scored)? {
        let p = e?.path();
        let name = p.file_name().unwrap().to_string_lossy().to_string();
        if !name.starts_with(&prefix) || name.ends_with(".val.jsonl") {
            continue;
        }
        for v in read_jsonl::<Value>(&p)? {
            if let (Some(uid), Some(t)) = (v["uid"].as_str(), v["tlogits"].as_array()) {
                let t: Vec<f32> = t.iter().map(|x| x.as_f64().unwrap_or(0.0) as f32).collect();
                out.insert(uid.to_string(), t);
            }
        }
    }
    Ok(out)
}

pub fn mix(o: &MixOptions) -> Result<Value> {
    if o.teachers.is_empty() {
        bail!("no teacher");
    }
    let files = ["corpus", "gen", "pairs", "coherence"];
    let mut recs: BTreeMap<&str, Vec<S5Record>> = BTreeMap::new();
    for f in files {
        recs.insert(f, read(&o.train.join(format!("{f}.jsonl")))?);
    }
    let logits: Vec<HashMap<String, Vec<f32>>> = o
        .teachers
        .iter()
        .map(|t| teacher_logits(o, t))
        .collect::<Result<_>>()?;
    // Records that teachers label: the corpus and the pairs.
    let labelled: Vec<&S5Record> = ["corpus", "pairs"]
        .iter()
        .flat_map(|f| recs[f].iter())
        .collect();
    // 1. Calibration per group and teacher.
    let mut by_group: BTreeMap<String, Vec<&S5Record>> = BTreeMap::new();
    for r in &labelled {
        by_group.entry(group_of(r)).or_default().push(r);
    }
    let mut calib: HashMap<(String, usize), Calib> = HashMap::new();
    let mut global_t = vec![1.0f32; o.teachers.len()];
    for (ti, lg) in logits.iter().enumerate() {
        // Global temperature: grid search on every gold train record.
        let train: Vec<(&[f32], &[f32])> = labelled
            .iter()
            .filter(|r| r.split == "train")
            .filter_map(|r| lg.get(&r.uid).map(|l| (l.as_slice(), r.target.as_slice())))
            .collect();
        let mut best = (f64::MAX, 1.0f32);
        for i in 0..=55 {
            let t = 0.25 + 0.05 * i as f32;
            let ce: f64 = train
                .iter()
                .map(|(l, y)| {
                    let p = softmax_t(l, t);
                    y.iter()
                        .zip(&p)
                        .map(|(a, b)| -(*a as f64) * (b.max(1e-9) as f64).ln())
                        .sum::<f64>()
                })
                .sum();
            if ce < best.0 {
                best = (ce, t);
            }
        }
        global_t[ti] = best.1;
        for (g, rs) in &by_group {
            let items: Vec<(&[f32], &[f32])> = rs
                .iter()
                .filter(|r| r.split == "train")
                .filter_map(|r| lg.get(&r.uid).map(|l| (l.as_slice(), r.target.as_slice())))
                .collect();
            let n = rs[0].opts_kv.len();
            let c = if items.len() >= MIN_GROUP {
                fit_calib(&items, n, global_t[ti])
            } else {
                Calib {
                    temperature: global_t[ti],
                    bias: vec![0.0; n],
                }
            };
            calib.insert((g.clone(), ti), c);
        }
    }
    let probs = |r: &S5Record, ti: usize| -> Option<Vec<f32>> {
        let l = logits[ti].get(&r.uid)?;
        Some(calib.get(&(group_of(r), ti))?.probs(l))
    };
    // 2. Teacher choice per group on the val records.
    let mut chosen: BTreeMap<String, Value> = BTreeMap::new();
    let mut pick: HashMap<String, usize> = HashMap::new();
    for (g, rs) in &by_group {
        let mut scores = Vec::new();
        for ti in 0..o.teachers.len() {
            let val: Vec<(Vec<f32>, &[f32])> = rs
                .iter()
                .filter(|r| r.split == "val")
                .filter_map(|r| probs(r, ti).map(|p| (p, r.target.as_slice())))
                .collect();
            let n = val.len();
            let b = if n == 0 {
                f64::NAN
            } else {
                val.iter().map(|(p, y)| brier(p, y)).sum::<f64>() / n as f64
            };
            scores.push((ti, b, if n == 0 { f64::NAN } else { ece(&val) }, n));
        }
        let best = scores
            .iter()
            .filter(|s| s.1.is_finite())
            .min_by(|a, b| a.1.total_cmp(&b.1).then(a.2.total_cmp(&b.2)))
            .map_or(0, |s| s.0);
        pick.insert(g.clone(), best);
        chosen.insert(
            g.clone(),
            json!({
                "teacher": o.teachers[best],
                "val": scores.iter().map(|s| json!({"teacher": o.teachers[s.0], "brier": s.1, "ece": s.2, "n": s.3})).collect::<Vec<_>>(),
            }),
        );
    }
    // 3–4. Targets.
    let mut stats = BTreeMap::new();
    let mut first_of_pair: HashMap<String, Vec<f32>> = HashMap::new();
    let mut out_counts = BTreeMap::new();
    for f in files {
        let mut v = recs[f].clone();
        if f == "corpus" || f == "pairs" {
            let (mut soft, mut kept_gold, mut missing) = (0usize, 0usize, 0usize);
            for r in v.iter_mut() {
                // The second member of a pair takes the first member's target.
                if let Some((key, idx)) = r.uid.split_once('#') {
                    if idx != "0" {
                        if let Some(t) = first_of_pair.get(key) {
                            r.target = t.clone();
                        }
                        continue;
                    }
                }
                let ti = pick[&group_of(r)];
                let ps: Vec<Option<Vec<f32>>> =
                    (0..o.teachers.len()).map(|t| probs(r, t)).collect();
                let Some(p) = ps[ti].clone() else {
                    missing += 1;
                    continue;
                };
                let agree = ps
                    .iter()
                    .flatten()
                    .map(|q| argmax(q))
                    .collect::<std::collections::BTreeSet<_>>()
                    .len()
                    == 1
                    || argmax(&p) == argmax(&r.target);
                if agree {
                    let s: f32 = r.target.iter().sum();
                    r.target = r
                        .target
                        .iter()
                        .zip(&p)
                        .map(|(g, q)| o.alpha * g / s + (1.0 - o.alpha) * q)
                        .collect();
                    soft += 1;
                } else {
                    kept_gold += 1;
                }
                if let Some((key, _)) = r.uid.split_once('#') {
                    first_of_pair.insert(key.to_string(), r.target.clone());
                }
            }
            stats.insert(
                f,
                json!({"soft": soft, "gold_kept": kept_gold, "unscored": missing}),
            );
        }
        out_counts.insert(f, write(&o.out.join(format!("{f}.jsonl")), &v)?);
    }
    let report = json!({
        "teachers": o.teachers,
        "alpha": o.alpha,
        "global_temperature": global_t,
        "groups": chosen,
        "targets": stats,
        "written": out_counts,
    });
    std::fs::write(
        o.out.join("REPORT.json"),
        serde_json::to_string_pretty(&report)? + "\n",
    )?;
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn helpers() {
        assert_eq!(
            base_src("typed_decisions_train+none"),
            "typed_decisions_train"
        );
        assert_eq!(base_src("boolq"), "boolq");
        let y = [1.0f32, 0.0];
        assert!((brier(&[1.0, 0.0], &y)).abs() < 1e-12);
        let items = vec![
            (vec![0.9f32, 0.1], &y[..]),
            (vec![0.9f32, 0.1], &[0.0f32, 1.0][..]),
        ];
        assert!((ece(&items) - 0.4).abs() < 1e-6);
    }
}
