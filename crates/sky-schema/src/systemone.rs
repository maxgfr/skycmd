//! Types of the llama.cpp `/v1/systemone` API (`server-decision.cpp`).
//!
//! Request: `{"state": <any JSON>, "questions": {"<id>": {"type": ..., "instructions": ...,
//! "criteria": ...}}}`. Each answer is `{"type": ..., <type-specific fields>}`. Question and
//! answer order is preserved (the server iterates the `questions` object in order).

use crate::record::{render_choice, render_noul, render_score, QType};
use indexmap::IndexMap;
use serde::{Deserialize, Serialize};

/// A `/v1/systemone` request.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SystemOneRequest {
    /// Free-form state: a string or any JSON value (serialized by the server's template).
    pub state: serde_json::Value,
    /// Questions by id, in request order.
    pub questions: IndexMap<String, Question>,
}

/// Optional descriptions of the two noul options.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct NoulCriteria {
    #[serde(rename = "true", default, skip_serializing_if = "Option::is_none")]
    pub true_desc: Option<String>,
    #[serde(rename = "false", default, skip_serializing_if = "Option::is_none")]
    pub false_desc: Option<String>,
}

/// One typed question.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum Question {
    /// Pick one key. Descriptions may be `null`.
    Choice {
        instructions: String,
        criteria: IndexMap<String, Option<String>>,
    },
    /// Rate on 2..=10 ordered levels.
    Score {
        instructions: String,
        criteria: Vec<String>,
    },
    /// Probability that the statement in `instructions` holds.
    Noul {
        instructions: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        criteria: Option<NoulCriteria>,
    },
}

impl Question {
    pub fn qtype(&self) -> QType {
        match self {
            Question::Choice { .. } => QType::Choice,
            Question::Score { .. } => QType::Score,
            Question::Noul { .. } => QType::Noul,
        }
    }

    pub fn instructions(&self) -> &str {
        match self {
            Question::Choice { instructions, .. }
            | Question::Score { instructions, .. }
            | Question::Noul { instructions, .. } => instructions,
        }
    }

    /// Option keys in model output order (`"0".."n-1"` for scores, `["false", "true"]` for noul).
    pub fn option_keys(&self) -> Vec<String> {
        match self {
            Question::Choice { criteria, .. } => criteria.keys().cloned().collect(),
            Question::Score { criteria, .. } => {
                (0..criteria.len()).map(|i| i.to_string()).collect()
            }
            Question::Noul { .. } => vec!["false".into(), "true".into()],
        }
    }

    /// Option texts as rendered in the prompt (`render_options` of the reference runtime).
    pub fn rendered_options(&self) -> Vec<String> {
        match self {
            Question::Choice { criteria, .. } => {
                let items: Vec<(&str, &str)> = criteria
                    .iter()
                    .map(|(k, d)| (k.as_str(), d.as_deref().unwrap_or("")))
                    .collect();
                render_choice(&items)
            }
            Question::Score { criteria, .. } => render_score(criteria),
            Question::Noul { criteria, .. } => {
                let c = criteria.clone().unwrap_or_default();
                render_noul(c.true_desc.as_deref(), c.false_desc.as_deref())
            }
        }
    }
}

/// One answer, shaped like `format_answer` in `server-decision.cpp`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum Answer {
    Choice {
        choice: String,
        probabilities: IndexMap<String, f64>,
        confidence: f64,
    },
    Score {
        /// Expected level, `sum(i * p_i)`.
        score: f64,
        /// Level key -> description, as returned by the server.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        legend: Option<IndexMap<String, Option<String>>>,
        probabilities: IndexMap<String, f64>,
        confidence: f64,
    },
    Noul {
        /// Probability of `true`.
        noul: f64,
    },
}

impl Answer {
    /// Builds the answer from calibrated probabilities in model output order, exactly as the
    /// server does (argmax keeps the first maximum).
    pub fn from_probs(question: &Question, probs: &[f64]) -> Answer {
        let keys = question.option_keys();
        assert_eq!(keys.len(), probs.len(), "one probability per option");
        let probabilities: IndexMap<String, f64> =
            keys.iter().cloned().zip(probs.iter().copied()).collect();
        match question {
            Question::Choice { .. } => {
                let mut best = 0;
                for (i, p) in probs.iter().enumerate() {
                    if *p > probs[best] {
                        best = i;
                    }
                }
                Answer::Choice {
                    choice: keys[best].clone(),
                    probabilities,
                    confidence: confidence_choice(probs),
                }
            }
            Question::Score { criteria, .. } => Answer::Score {
                score: probs.iter().enumerate().map(|(i, p)| i as f64 * p).sum(),
                legend: Some(
                    keys.iter()
                        .cloned()
                        .zip(criteria.iter().map(|c| Some(c.clone())))
                        .collect(),
                ),
                probabilities,
                confidence: confidence_score(probs),
            },
            Question::Noul { .. } => Answer::Noul { noul: probs[1] },
        }
    }
}

/// `(p_max - 1/n) / (1 - 1/n)`, clamped at 0.
pub fn confidence_choice(probs: &[f64]) -> f64 {
    if probs.len() < 2 {
        return 1.0;
    }
    let uniform = 1.0 / probs.len() as f64;
    let p_max = probs.iter().copied().fold(f64::MIN, f64::max);
    ((p_max - uniform) / (1.0 - uniform)).max(0.0)
}

/// One minus the mean distance to the mode, relative to that of a uniform distribution.
pub fn confidence_score(probs: &[f64]) -> f64 {
    let n = probs.len();
    if n < 2 {
        return 1.0;
    }
    let mut mode = 0;
    for (i, p) in probs.iter().enumerate() {
        if *p > probs[mode] {
            mode = i;
        }
    }
    let mut dist = 0.0;
    let mut dist_uniform = 0.0;
    for (i, p) in probs.iter().enumerate() {
        dist += p * (i as f64 - mode as f64).abs();
        dist_uniform += (i as f64 - (n as f64 - 1.0) / 2.0).abs() / n as f64;
    }
    (1.0 - dist / dist_uniform).max(0.0)
}

/// A `/v1/systemone` response: answers by question id, in request order.
///
/// The response envelope is assembled outside `server-decision.cpp`; this type reads both
/// `{"answers": {...}, ...}` (extra top-level fields are ignored) and a bare `{"<id>": answer}`
/// object, and always writes the `{"answers": {...}}` form.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(from = "ResponseRepr")]
pub struct SystemOneResponse {
    pub answers: IndexMap<String, Answer>,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum ResponseRepr {
    Wrapped { answers: IndexMap<String, Answer> },
    Bare(IndexMap<String, Answer>),
}

impl From<ResponseRepr> for SystemOneResponse {
    fn from(r: ResponseRepr) -> Self {
        match r {
            ResponseRepr::Wrapped { answers } | ResponseRepr::Bare(answers) => {
                SystemOneResponse { answers }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const REQUEST: &str = r#"{"state":"alt 31 vy -2 dx 12 gt 38 gb 22 nc 30","questions":{"action":{"type":"choice","instructions":"Which input keeps the drone flying through the next gap?","criteria":{"thrust":"fire the rotors upward","glide":null}},"risk":{"type":"score","instructions":"How risky is the battery margin?","criteria":["negligible","low","critical"]},"danger":{"type":"noul","instructions":"The drone crashes within one second if it keeps gliding."},"custom":{"type":"noul","instructions":"Is it safe?","criteria":{"true":"safe","false":"unsafe"}}}}"#;

    #[test]
    fn request_round_trip_preserves_order() {
        let req: SystemOneRequest = serde_json::from_str(REQUEST).unwrap();
        let ids: Vec<&str> = req.questions.keys().map(String::as_str).collect();
        assert_eq!(ids, ["action", "risk", "danger", "custom"]);
        let action = &req.questions["action"];
        assert_eq!(action.qtype(), QType::Choice);
        assert_eq!(action.option_keys(), ["thrust", "glide"]);
        assert_eq!(
            action.rendered_options(),
            ["thrust: fire the rotors upward", "glide"]
        );
        assert_eq!(
            req.questions["risk"].rendered_options(),
            ["level 0: negligible", "level 1: low", "level 2: critical"]
        );
        assert_eq!(
            req.questions["danger"].rendered_options(),
            [
                "false: no, the statement does not hold",
                "true: yes, the statement holds"
            ]
        );
        assert_eq!(
            req.questions["custom"].rendered_options(),
            ["false: unsafe", "true: safe"]
        );
        assert_eq!(serde_json::to_string(&req).unwrap(), REQUEST);
    }

    #[test]
    fn choice_keys_keep_insertion_order() {
        let json = r#"{"state":{"z":1,"a":2},"questions":{"q":{"type":"choice","instructions":"i","criteria":{"zeta":"z","alpha":"a","mid":"m"}}}}"#;
        let req: SystemOneRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.questions["q"].option_keys(), ["zeta", "alpha", "mid"]);
        assert_eq!(serde_json::to_string(&req).unwrap(), json);
    }

    #[test]
    fn response_round_trip_and_from_probs() {
        let req: SystemOneRequest = serde_json::from_str(REQUEST).unwrap();
        let mut answers = IndexMap::new();
        answers.insert(
            "action".to_string(),
            Answer::from_probs(&req.questions["action"], &[0.75, 0.25]),
        );
        answers.insert(
            "risk".to_string(),
            Answer::from_probs(&req.questions["risk"], &[0.5, 0.25, 0.25]),
        );
        answers.insert(
            "danger".to_string(),
            Answer::from_probs(&req.questions["danger"], &[0.875, 0.125]),
        );
        let resp = SystemOneResponse { answers };
        let text = serde_json::to_string(&resp).unwrap();
        assert_eq!(
            text,
            r#"{"answers":{"action":{"type":"choice","choice":"thrust","probabilities":{"thrust":0.75,"glide":0.25},"confidence":0.5},"risk":{"type":"score","score":0.75,"legend":{"0":"negligible","1":"low","2":"critical"},"probabilities":{"0":0.5,"1":0.25,"2":0.25},"confidence":0.0},"danger":{"type":"noul","noul":0.125}}}"#
        );
        let back: SystemOneResponse = serde_json::from_str(&text).unwrap();
        assert_eq!(back, resp);
    }

    #[test]
    fn response_accepts_bare_map_and_extra_fields() {
        let bare = r#"{"q":{"type":"noul","noul":0.9}}"#;
        let r: SystemOneResponse = serde_json::from_str(bare).unwrap();
        assert_eq!(r.answers["q"], Answer::Noul { noul: 0.9 });
        let wrapped = r#"{"model":"oscar","answers":{"q":{"type":"choice","choice":"b","probabilities":{"a":0.1,"b":0.9},"confidence":0.8}}}"#;
        let r: SystemOneResponse = serde_json::from_str(wrapped).unwrap();
        assert!(matches!(&r.answers["q"], Answer::Choice { choice, .. } if choice == "b"));
    }

    #[test]
    fn confidences_match_server_formulas() {
        assert!((confidence_choice(&[1.0, 0.0, 0.0]) - 1.0).abs() < 1e-12);
        assert!(confidence_choice(&[1.0 / 3.0; 3]).abs() < 1e-12);
        assert!((confidence_score(&[0.0, 1.0, 0.0]) - 1.0).abs() < 1e-12);
        assert!(confidence_score(&[0.5, 0.0, 0.5]) < 0.5);
    }
}
