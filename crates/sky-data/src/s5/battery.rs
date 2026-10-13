//! `sky-data s5-battery`: the original coherence battery of 2609.33209 (`bro789/typed-decision-
//! coherence`, `02_probability_axioms/stimuli/items.jsonl`: 100 ChaosNLI items, CC BY-NC 4.0, and
//! 60 PubMedQA items, MIT) as s5 records, for the comparison with Jev on the same items.
//! Evaluation only: the records stay in the store and are never redistributed.
//!
//! Each item becomes one family in the uid scheme of [`super::coherence`]:
//! `coh/<id>/CH/0` (the choice, options in the order of `classes`), `coh/<id>/S/k` and
//! `coh/<id>/N/k` (the battery's own questions "Is it true that the label is (not) X?"), and
//! `coh/<id>/D/i-j` for `D_k`, "either of the two other labels" (`{i, j} = {0, 1, 2} \ {k}`).
//! Targets: the 100 ChaosNLI human votes for NLI items, the expert label for PubMedQA items.

use std::path::Path;

use anyhow::{bail, Context, Result};
use serde_json::Value;

use super::S5Record;

fn noul(id: &str, ins: &str, state: &str, p_true: f32, uid: String) -> S5Record {
    S5Record {
        src: format!("battery_{id}"),
        t: "noul".into(),
        ins: ins.to_string(),
        opts_kv: vec![
            ("false".into(), String::new()),
            ("true".into(), String::new()),
        ],
        state: state.to_string(),
        target: vec![1.0 - p_true, p_true],
        split: "test".into(),
        uid,
        lang: String::new(),
        gold: true,
    }
}

/// The records of one battery item.
pub fn item_records(it: &Value) -> Result<Vec<S5Record>> {
    let id = it["id"].as_str().context("item without id")?;
    let state = it["state"].as_str().context("item without state")?;
    let classes: Vec<&str> = it["classes"]
        .as_array()
        .context("item without classes")?
        .iter()
        .filter_map(Value::as_str)
        .collect();
    if classes.len() != 3 {
        bail!("{id}: {} classes", classes.len());
    }
    let reference = &it["reference"];
    let p: Vec<f32> = match reference["human_dist"].as_object() {
        Some(d) => classes
            .iter()
            .map(|c| d.get(*c).and_then(Value::as_f64).unwrap_or(0.0) as f32)
            .collect(),
        None => {
            let label = reference["label"]
                .as_str()
                .context("reference without label")?;
            classes
                .iter()
                .map(|c| f32::from(u8::from(*c == label)))
                .collect()
        }
    };
    let s: f32 = p.iter().sum();
    if s <= 0.0 {
        bail!("{id}: empty reference");
    }
    let p: Vec<f32> = p.iter().map(|x| x / s).collect();
    let domain = it["domain"].as_str().unwrap_or("battery");
    let q = &it["questions"];
    let ch = &q["CH"];
    let opts_kv: Vec<(String, String)> = classes
        .iter()
        .map(|c| {
            let d = ch["descriptions"][*c].as_str().unwrap_or_default();
            (c.to_string(), d.to_string())
        })
        .collect();
    let mut v = vec![S5Record {
        src: format!("battery_{domain}"),
        t: "choice".into(),
        ins: ch["instructions"]
            .as_str()
            .context("CH without instructions")?
            .to_string(),
        opts_kv,
        state: state.to_string(),
        target: p.clone(),
        split: "test".into(),
        uid: format!("coh/{id}/CH/0"),
        lang: String::new(),
        gold: true,
    }];
    let text = |k: &str| -> Result<&str> {
        q[k]["instructions"]
            .as_str()
            .with_context(|| format!("{id}: no question {k}"))
    };
    for k in 0..3 {
        v.push(noul(
            domain,
            text(&format!("S{k}"))?,
            state,
            p[k],
            format!("coh/{id}/S/{k}"),
        ));
        v.push(noul(
            domain,
            text(&format!("N{k}"))?,
            state,
            1.0 - p[k],
            format!("coh/{id}/N/{k}"),
        ));
        let (i, j) = match k {
            0 => (1, 2),
            1 => (0, 2),
            _ => (0, 1),
        };
        v.push(noul(
            domain,
            text(&format!("D{k}"))?,
            state,
            p[i] + p[j],
            format!("coh/{id}/D/{i}-{j}"),
        ));
    }
    Ok(v)
}

/// Converts `items.jsonl` into `out` (one family per item); returns the number of records.
pub fn convert(items: &Path, out: &Path) -> Result<usize> {
    let text = std::fs::read_to_string(items).with_context(|| items.display().to_string())?;
    let mut recs = Vec::new();
    for line in text.lines().filter(|l| !l.trim().is_empty()) {
        let it: Value = serde_json::from_str(line)?;
        recs.extend(item_records(&it)?);
    }
    super::write(out, &recs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn one_item_one_family() {
        let mut questions = serde_json::Map::new();
        for k in 0..3 {
            for kind in ["S", "N", "D"] {
                questions.insert(
                    format!("{kind}{k}"),
                    json!({"type": "noul", "instructions": format!("{kind}{k}?")}),
                );
            }
        }
        questions.insert(
            "CH".into(),
            json!({"type": "choice", "instructions": "Which?", "options": ["c", "a", "b"],
                   "descriptions": {"a": "da", "b": "db", "c": "dc"}}),
        );
        let it = json!({"id": "x1", "domain": "nli", "state": "s", "classes": ["a", "b", "c"],
                        "questions": questions,
                        "reference": {"label": "c", "human_dist": {"a": 0.2, "b": 0.3, "c": 0.5}}});
        let v = item_records(&it).unwrap();
        assert_eq!(v.len(), 10);
        assert_eq!(v[0].uid, "coh/x1/CH/0");
        assert_eq!(v[0].opts_kv[0], ("a".to_string(), "da".to_string()));
        let d0 = v.iter().find(|r| r.uid == "coh/x1/D/1-2").unwrap();
        assert_eq!(d0.ins, "D0?");
        assert!((d0.target[1] - 0.8).abs() < 1e-6);
        let n2 = v.iter().find(|r| r.uid == "coh/x1/N/2").unwrap();
        assert!((n2.target[1] - 0.5).abs() < 1e-6);
    }
}
