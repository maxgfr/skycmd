//! Decision records (SPEC "Decision record") and option rendering (`render_options` in
//! `rl_common.py`).

use serde::{Deserialize, Serialize};
use std::fmt;

/// Default description of the `false` option of a noul question.
pub const NOUL_FALSE_DEFAULT: &str = "no, the statement does not hold";
/// Default description of the `true` option of a noul question.
pub const NOUL_TRUE_DEFAULT: &str = "yes, the statement holds";

/// Question type, which also selects the decision head's type embedding (0, 1, 2).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum QType {
    Choice,
    Score,
    Noul,
}

impl QType {
    /// The prompt word (`choice`, `score`, `noul`).
    pub fn as_str(self) -> &'static str {
        match self {
            QType::Choice => "choice",
            QType::Score => "score",
            QType::Noul => "noul",
        }
    }

    /// Index of the type embedding row: choice 0, score 1, noul 2.
    pub fn index(self) -> usize {
        match self {
            QType::Choice => 0,
            QType::Score => 1,
            QType::Noul => 2,
        }
    }
}

impl fmt::Display for QType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Dataset split of a record.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Split {
    Train,
    Val,
    Test,
}

/// One supervised decision: a rendered question, its option texts and a target distribution.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DecisionRecord {
    /// Source tag, e.g. `boolq` or `game:flappy`.
    pub src: String,
    /// Question type.
    pub t: QType,
    /// Instructions (the question text).
    pub ins: String,
    /// Rendered option texts, in the order the model sees them.
    pub opts: Vec<String>,
    /// Free text or compact game state.
    pub state: String,
    /// Probability vector over `opts` (one-hot or soft).
    pub target: Vec<f32>,
    pub split: Split,
}

impl DecisionRecord {
    /// Checks the structural invariants the trainer relies on.
    pub fn validate(&self) -> Result<(), String> {
        if self.src.is_empty() {
            return Err("empty src".into());
        }
        if self.ins.trim().is_empty() {
            return Err("empty instructions".into());
        }
        if self.opts.is_empty() {
            return Err("no options".into());
        }
        if self.opts.len() != self.target.len() {
            return Err(format!(
                "{} options but {} target values",
                self.opts.len(),
                self.target.len()
            ));
        }
        if self.opts.iter().any(|o| o.trim().is_empty()) {
            return Err("empty option text".into());
        }
        if self.target.iter().any(|p| !p.is_finite() || *p < 0.0) {
            return Err("target values must be finite and non-negative".into());
        }
        let sum: f32 = self.target.iter().sum();
        if (sum - 1.0).abs() > 1e-3 {
            return Err(format!("target sums to {sum}, expected 1"));
        }
        match self.t {
            QType::Choice => {
                if self.opts.len() < 2 {
                    return Err("a choice needs at least 2 options".into());
                }
            }
            QType::Score => {
                if !(2..=10).contains(&self.opts.len()) {
                    return Err("a score needs 2 to 10 levels".into());
                }
                for (i, o) in self.opts.iter().enumerate() {
                    if !o.starts_with(&format!("level {i}: ")) {
                        return Err(format!("score option {i} must start with \"level {i}: \""));
                    }
                }
            }
            QType::Noul => {
                if self.opts.len() != 2
                    || !self.opts[0].starts_with("false: ")
                    || !self.opts[1].starts_with("true: ")
                {
                    return Err("noul options must be [\"false: ...\", \"true: ...\"]".into());
                }
            }
        }
        Ok(())
    }
}

/// Choice options: `key` when the description is empty, else `key: desc`.
pub fn render_choice<K: AsRef<str>, D: AsRef<str>>(keys_with_desc: &[(K, D)]) -> Vec<String> {
    keys_with_desc
        .iter()
        .map(|(k, d)| {
            let (k, d) = (k.as_ref(), d.as_ref());
            if d.is_empty() {
                k.to_string()
            } else {
                format!("{k}: {d}")
            }
        })
        .collect()
}

/// Score options: `level i: desc`.
pub fn render_score<S: AsRef<str>>(levels: &[S]) -> Vec<String> {
    levels
        .iter()
        .enumerate()
        .map(|(i, c)| format!("level {i}: {}", c.as_ref()))
        .collect()
}

/// Noul options, always `[false, true]` so that `p[1]` is the noul answer. Missing or empty
/// descriptions fall back to the reference defaults.
pub fn render_noul(true_desc: Option<&str>, false_desc: Option<&str>) -> Vec<String> {
    let pick = |d: Option<&str>, default: &'static str| match d {
        Some(s) if !s.is_empty() => s.to_string(),
        _ => default.to_string(),
    };
    vec![
        format!("false: {}", pick(false_desc, NOUL_FALSE_DEFAULT)),
        format!("true: {}", pick(true_desc, NOUL_TRUE_DEFAULT)),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spec_example_round_trips() {
        let line = r#"{"src":"boolq","t":"noul","ins":"Is the passage saying that ...?","opts":["false: no, the statement does not hold","true: yes, the statement holds"],"state":"<free text or compact game state>","target":[0.0,1.0],"split":"train"}"#;
        let rec: DecisionRecord = serde_json::from_str(line).unwrap();
        assert_eq!(rec.t, QType::Noul);
        assert_eq!(rec.split, Split::Train);
        rec.validate().unwrap();
        assert_eq!(serde_json::to_string(&rec).unwrap(), line);
    }

    #[test]
    fn renderers_match_reference() {
        assert_eq!(
            render_choice(&[("thrust", "fire the rotors"), ("glide", "")]),
            vec!["thrust: fire the rotors", "glide"]
        );
        assert_eq!(
            render_score(&["low", "high"]),
            vec!["level 0: low", "level 1: high"]
        );
        assert_eq!(
            render_noul(None, Some("")),
            vec![
                "false: no, the statement does not hold",
                "true: yes, the statement holds"
            ]
        );
        assert_eq!(
            render_noul(Some("it crashes"), Some("it is safe")),
            vec!["false: it is safe", "true: it crashes"]
        );
    }

    #[test]
    fn validate_rejects_bad_records() {
        let mut rec = DecisionRecord {
            src: "game:test".into(),
            t: QType::Choice,
            ins: "pick".into(),
            opts: render_choice(&[("a", ""), ("b", "")]),
            state: "x 1".into(),
            target: vec![0.5, 0.5],
            split: Split::Val,
        };
        rec.validate().unwrap();
        rec.target = vec![0.5, 0.6];
        assert!(rec.validate().is_err());
        rec.target = vec![1.0];
        assert!(rec.validate().is_err());
        rec.t = QType::Score;
        rec.target = vec![0.5, 0.5];
        assert!(rec.validate().is_err());
        rec.opts = render_score(&["a", "b"]);
        rec.validate().unwrap();
    }
}
