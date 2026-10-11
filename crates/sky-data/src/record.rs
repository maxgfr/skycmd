//! Record types (SPEC "Decision record" / "MLM record") and Jev-style question
//! rendering, mirroring `render_options` in `rl_common.py`.

use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;

use anyhow::{anyhow, bail, Context, Result};
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use serde_json::{Map, Value};

/// Default noul option descriptions (rl_common.render_options).
pub const NOUL_FALSE_DEFAULT: &str = "no, the statement does not hold";
pub const NOUL_TRUE_DEFAULT: &str = "yes, the statement holds";

/// One typed decision: a question over rendered options plus a target distribution.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct DecisionRecord {
    pub src: String,
    pub t: String,
    pub ins: String,
    pub opts: Vec<String>,
    pub state: String,
    pub target: Vec<f32>,
    pub split: String,
}

/// One masked-language-modelling text sample.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct MlmRecord {
    pub src: String,
    pub text: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum QType {
    Choice,
    Score,
    Noul,
}

impl QType {
    pub fn as_str(self) -> &'static str {
        match self {
            QType::Choice => "choice",
            QType::Score => "score",
            QType::Noul => "noul",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "choice" => Some(QType::Choice),
            "score" => Some(QType::Score),
            "noul" => Some(QType::Noul),
            _ => None,
        }
    }
}

/// Question criteria in label-index order.
#[derive(Clone, Debug, PartialEq)]
pub enum Crit {
    /// `(key, description)`; an empty description renders as the bare key.
    Choice(Vec<(String, String)>),
    /// One description per ordinal level.
    Score(Vec<String>),
    /// Optional overrides of the default false/true descriptions.
    Noul {
        no: Option<String>,
        yes: Option<String>,
    },
}

#[derive(Clone, Debug, PartialEq)]
pub struct Question {
    pub t: QType,
    pub ins: String,
    pub crit: Crit,
}

impl Question {
    pub fn choice(ins: impl Into<String>, crit: Vec<(String, String)>) -> Self {
        Self {
            t: QType::Choice,
            ins: ins.into(),
            crit: Crit::Choice(crit),
        }
    }

    pub fn score(ins: impl Into<String>, levels: Vec<String>) -> Self {
        Self {
            t: QType::Score,
            ins: ins.into(),
            crit: Crit::Score(levels),
        }
    }

    pub fn noul(ins: impl Into<String>) -> Self {
        Self {
            t: QType::Noul,
            ins: ins.into(),
            crit: Crit::Noul {
                no: None,
                yes: None,
            },
        }
    }

    /// Rendered option texts, exactly as `rl_common.render_options`.
    pub fn options(&self) -> Vec<String> {
        match &self.crit {
            Crit::Choice(c) => c
                .iter()
                .map(|(k, v)| {
                    if v.is_empty() {
                        k.clone()
                    } else {
                        format!("{k}: {v}")
                    }
                })
                .collect(),
            Crit::Score(levels) => levels
                .iter()
                .enumerate()
                .map(|(i, c)| format!("level {i}: {c}"))
                .collect(),
            Crit::Noul { no, yes } => vec![
                format!("false: {}", nonempty(no).unwrap_or(NOUL_FALSE_DEFAULT)),
                format!("true: {}", nonempty(yes).unwrap_or(NOUL_TRUE_DEFAULT)),
            ],
        }
    }

    /// Label keys in option order: choice keys, score level indices, `false`/`true`.
    pub fn keys(&self) -> Vec<String> {
        match &self.crit {
            Crit::Choice(c) => c.iter().map(|(k, _)| k.clone()).collect(),
            Crit::Score(levels) => (0..levels.len()).map(|i| i.to_string()).collect(),
            Crit::Noul { .. } => vec!["false".into(), "true".into()],
        }
    }

    pub fn n_options(&self) -> usize {
        match &self.crit {
            Crit::Choice(c) => c.len(),
            Crit::Score(l) => l.len(),
            Crit::Noul { .. } => 2,
        }
    }

    /// Parses a System One / Jev question object
    /// `{"type": .., "instructions": .., "criteria": ..}`.
    pub fn from_jev(v: &Value) -> Result<Self> {
        let t = v
            .get("type")
            .and_then(Value::as_str)
            .and_then(QType::parse)
            .ok_or_else(|| anyhow!("question without a valid type"))?;
        let ins = v
            .get("instructions")
            .and_then(Value::as_str)
            .unwrap_or("")
            .trim()
            .to_string();
        let crit = v.get("criteria").unwrap_or(&Value::Null);
        let crit = match t {
            QType::Choice => Crit::Choice(match crit {
                Value::Object(m) => m
                    .iter()
                    .map(|(k, d)| (k.clone(), d.as_str().unwrap_or("").trim().to_string()))
                    .collect(),
                Value::Array(a) => a
                    .iter()
                    .filter_map(Value::as_str)
                    .map(|k| (k.to_string(), String::new()))
                    .collect(),
                _ => bail!("choice question without criteria"),
            }),
            QType::Score => Crit::Score(match crit {
                Value::Array(a) => a
                    .iter()
                    .map(|x| {
                        x.as_str()
                            .map(str::to_string)
                            .unwrap_or_else(|| x.to_string())
                    })
                    .collect(),
                _ => bail!("score question without a criteria list"),
            }),
            QType::Noul => Crit::Noul {
                no: crit
                    .get("false")
                    .and_then(Value::as_str)
                    .map(str::to_string),
                yes: crit.get("true").and_then(Value::as_str).map(str::to_string),
            },
        };
        let q = Self { t, ins, crit };
        if q.n_options() < 2 {
            bail!("question with fewer than 2 options");
        }
        Ok(q)
    }

    /// Maps a hard label (key string, level index, boolean, or `{"label": ..}`) to an option index.
    pub fn label_index(&self, label: &Value) -> Option<usize> {
        let keys = self.keys();
        match label {
            Value::Object(m) => m.get("label").and_then(|l| self.label_index(l)),
            Value::Bool(b) if self.t == QType::Noul => Some(*b as usize),
            Value::Number(n) => {
                let i = n.as_u64()? as usize;
                (i < keys.len()).then_some(i)
            }
            Value::String(s) => {
                if let Some(i) = keys.iter().position(|k| k == s) {
                    return Some(i);
                }
                match (self.t, s.to_ascii_lowercase().as_str()) {
                    (QType::Noul, "yes") => Some(1),
                    (QType::Noul, "no") => Some(0),
                    (QType::Score, x) => x.parse::<usize>().ok().filter(|i| *i < keys.len()),
                    _ => None,
                }
            }
            _ => None,
        }
    }

    /// Soft target from a label object: `{"probabilities": {key: p}}`, `{"noul": p}`,
    /// or a hard label (one-hot). Returns `None` when nothing usable is present.
    pub fn target_from(&self, v: &Value) -> Option<Vec<f32>> {
        let keys = self.keys();
        if let Some(Value::Object(probs)) = v.get("probabilities") {
            if let Some(t) = probs_to_target(&keys, probs) {
                return Some(t);
            }
        }
        if self.t == QType::Noul {
            if let Some(p) = v.get("noul").and_then(Value::as_f64) {
                if (0.0..=1.0).contains(&p) {
                    return Some(normalize(&[1.0 - p, p]));
                }
            }
        }
        let i = self.label_index(v)?;
        Some(one_hot(keys.len(), i))
    }

    /// Builds a record from this question.
    pub fn record(&self, src: &str, state: impl Into<String>, target: Vec<f32>) -> DecisionRecord {
        DecisionRecord {
            src: src.to_string(),
            t: self.t.as_str().to_string(),
            ins: self.ins.clone(),
            opts: self.options(),
            state: state.into(),
            target,
            split: "train".to_string(),
        }
    }
}

fn nonempty(s: &Option<String>) -> Option<&str> {
    s.as_deref().map(str::trim).filter(|s| !s.is_empty())
}

fn probs_to_target(keys: &[String], probs: &Map<String, Value>) -> Option<Vec<f32>> {
    let raw: Vec<f64> = keys
        .iter()
        .map(|k| probs.get(k).and_then(Value::as_f64).unwrap_or(0.0).max(0.0))
        .collect();
    let s: f64 = raw.iter().sum();
    (s > 0.0 && s.is_finite()).then(|| normalize(&raw))
}

/// One-hot vector of length `n`.
pub fn one_hot(n: usize, i: usize) -> Vec<f32> {
    (0..n).map(|j| if j == i { 1.0 } else { 0.0 }).collect()
}

/// Normalizes non-negative weights to a probability vector (f32, computed in f64).
pub fn normalize(w: &[f64]) -> Vec<f32> {
    let s: f64 = w.iter().map(|x| x.max(0.0)).sum();
    if s <= 0.0 {
        let n = w.len().max(1) as f64;
        return w.iter().map(|_| (1.0 / n) as f32).collect();
    }
    w.iter().map(|x| (x.max(0.0) / s) as f32).collect()
}

/// Index of the largest probability (first one on ties).
pub fn argmax(p: &[f32]) -> usize {
    let mut best = 0;
    for (i, &x) in p.iter().enumerate() {
        if x > p[best] {
            best = i;
        }
    }
    best
}

/// Shannon entropy in bits.
pub fn entropy_bits(p: &[f64]) -> f64 {
    let s: f64 = p.iter().sum();
    if s <= 0.0 {
        return 0.0;
    }
    -p.iter()
        .map(|&x| x / s)
        .filter(|&x| x > 0.0)
        .map(|x| x * x.log2())
        .sum::<f64>()
}

/// Checks the structural invariants of a decision record.
pub fn validate_record(r: &DecisionRecord) -> Result<()> {
    let t = QType::parse(&r.t).ok_or_else(|| anyhow!("bad type {}", r.t))?;
    if r.opts.len() < 2 {
        bail!("fewer than 2 options");
    }
    if r.target.len() != r.opts.len() {
        bail!("target length {} != opts {}", r.target.len(), r.opts.len());
    }
    if t == QType::Noul
        && !(r.opts.len() == 2
            && r.opts[0].starts_with("false: ")
            && r.opts[1].starts_with("true: "))
    {
        bail!("noul options must be [false, true]");
    }
    let s: f32 = r.target.iter().sum();
    if r.target.iter().any(|x| !x.is_finite() || *x < 0.0) || (s - 1.0).abs() > 1e-3 {
        bail!("target is not a distribution (sum {s})");
    }
    if !matches!(r.split.as_str(), "train" | "val" | "test") {
        bail!("bad split {}", r.split);
    }
    if r.ins.trim().is_empty() {
        bail!("empty instructions");
    }
    Ok(())
}

/// Writes records as JSONL, returning the number of lines.
pub fn write_jsonl<T: Serialize>(path: &Path, items: impl IntoIterator<Item = T>) -> Result<usize> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let mut w = BufWriter::new(File::create(path).with_context(|| format!("create {path:?}"))?);
    let mut n = 0;
    for it in items {
        serde_json::to_writer(&mut w, &it)?;
        w.write_all(b"\n")?;
        n += 1;
    }
    w.flush()?;
    Ok(n)
}

/// Reads a JSONL file, skipping blank lines.
pub fn read_jsonl<T: DeserializeOwned>(path: &Path) -> Result<Vec<T>> {
    let r = BufReader::new(File::open(path).with_context(|| format!("open {path:?}"))?);
    let mut out = Vec::new();
    for (i, line) in r.lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        out.push(serde_json::from_str(&line).with_context(|| format!("{path:?}:{}", i + 1))?);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn renders_like_rl_common() {
        let q = Question::from_jev(
            &json!({"type":"choice","instructions":"Pick","criteria":{"a":"first","b":""}}),
        )
        .unwrap();
        assert_eq!(q.options(), vec!["a: first", "b"]);
        let q = Question::from_jev(
            &json!({"type":"score","instructions":"Rate","criteria":["low","high"]}),
        )
        .unwrap();
        assert_eq!(q.options(), vec!["level 0: low", "level 1: high"]);
        let q = Question::from_jev(&json!({"type":"noul","instructions":"Holds?"})).unwrap();
        assert_eq!(
            q.options(),
            vec![
                "false: no, the statement does not hold",
                "true: yes, the statement holds"
            ]
        );
        let q = Question::from_jev(
            &json!({"type":"noul","instructions":"x","criteria":{"true":"T","false":""}}),
        )
        .unwrap();
        assert_eq!(
            q.options(),
            vec!["false: no, the statement does not hold", "true: T"]
        );
    }

    #[test]
    fn preserves_choice_key_order() {
        let q = Question::from_jev(
            &json!({"type":"choice","instructions":"x","criteria":{"z":"","a":"","m":""}}),
        )
        .unwrap();
        assert_eq!(q.keys(), vec!["z", "a", "m"]);
    }

    #[test]
    fn targets_from_labels() {
        let q = Question::from_jev(&json!({"type":"noul","instructions":"x"})).unwrap();
        assert_eq!(
            q.target_from(&json!({"noul": 0.25})).unwrap(),
            vec![0.75, 0.25]
        );
        assert_eq!(q.target_from(&json!(true)).unwrap(), vec![0.0, 1.0]);
        assert_eq!(
            q.target_from(&json!({"probabilities":{"false":0.2,"true":0.8}}))
                .unwrap(),
            vec![0.2, 0.8]
        );
        let s = Question::from_jev(
            &json!({"type":"score","instructions":"x","criteria":["a","b","c"]}),
        )
        .unwrap();
        assert_eq!(s.target_from(&json!(2)).unwrap(), vec![0.0, 0.0, 1.0]);
        assert_eq!(
            s.target_from(&json!({"label":"1"})).unwrap(),
            vec![0.0, 1.0, 0.0]
        );
        let t = s
            .target_from(&json!({"probabilities":{"0":1.0,"2":3.0}}))
            .unwrap();
        assert_eq!(t, vec![0.25, 0.0, 0.75]);
        let c = Question::from_jev(
            &json!({"type":"choice","instructions":"x","criteria":{"a":"","b":""}}),
        )
        .unwrap();
        assert_eq!(c.target_from(&json!("b")).unwrap(), vec![0.0, 1.0]);
        assert!(c.target_from(&json!("zz")).is_none());
    }

    #[test]
    fn validation_and_entropy() {
        let q = Question::noul("Is it?");
        let mut r = q.record("t", "s", vec![0.5, 0.5]);
        validate_record(&r).unwrap();
        assert!((entropy_bits(&[0.5, 0.5]) - 1.0).abs() < 1e-9);
        r.target = vec![0.5, 0.6];
        assert!(validate_record(&r).is_err());
        assert_eq!(argmax(&[0.1, 0.7, 0.2]), 1);
    }

    #[test]
    fn jsonl_roundtrip() {
        let dir = std::env::temp_dir().join(format!("sky-data-test-{}", std::process::id()));
        let p = dir.join("x.jsonl");
        let recs = vec![MlmRecord {
            src: "a".into(),
            text: "hello\nworld".into(),
        }];
        assert_eq!(write_jsonl(&p, &recs).unwrap(), 1);
        let back: Vec<MlmRecord> = read_jsonl(&p).unwrap();
        assert_eq!(back, recs);
        std::fs::remove_dir_all(dir).ok();
    }
}
