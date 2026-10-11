//! Minimal System One request / response types (Jev shape), with JSON support behind the
//! `json` feature.

use crate::prompt::{QType, QuestionKind};

/// One question of a request.
#[derive(Clone, Debug, PartialEq)]
pub struct Question {
    pub id: String,
    /// Instructions text (non-string JSON instructions are serialized like Python `json.dumps`).
    pub instructions: String,
    pub kind: QuestionKind,
}

/// A decision request: one state, several questions.
#[derive(Clone, Debug, PartialEq)]
pub struct DecisionRequest {
    /// State text (non-string JSON states are serialized like Python `json.dumps(ensure_ascii=False)`).
    pub state: String,
    pub questions: Vec<Question>,
}

/// The typed answer to one question.
#[derive(Clone, Debug, PartialEq)]
pub enum Answer {
    Choice {
        choice: String,
        probabilities: Vec<(String, f32)>,
        confidence: f32,
    },
    Score {
        score: f32,
        legend: Vec<(String, String)>,
        probabilities: Vec<(String, f32)>,
        confidence: f32,
    },
    Noul {
        noul: f32,
    },
}

impl Answer {
    pub fn qtype(&self) -> QType {
        match self {
            Answer::Choice { .. } => QType::Choice,
            Answer::Score { .. } => QType::Score,
            Answer::Noul { .. } => QType::Noul,
        }
    }
}

/// Early-exit details of an s2 answer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExitInfo {
    /// Index of the exit that answered.
    pub exit: usize,
    /// State-encoder blocks used.
    pub layers_used: usize,
}

/// Answers in question order.
#[derive(Clone, Debug, PartialEq)]
pub struct DecisionResponse {
    pub model: String,
    pub answers: Vec<(String, Answer)>,
    /// Per answer (same order): the exit of s2 models, `None` for s1. Serialized as `exit` and
    /// `layers_used` inside each answer.
    pub exits: Vec<Option<ExitInfo>>,
    /// Tokens evaluated over all questions.
    pub input_tokens: usize,
    /// Wall time of the request in microseconds (feature `timing`).
    pub micros: Option<u64>,
}

#[cfg(feature = "json")]
mod json_impl {
    use serde_json::{Map, Value};

    use super::*;
    use crate::{Error, Result};

    fn bad(msg: impl Into<String>) -> Error {
        Error::InvalidRequest(msg.into())
    }

    /// Python `str()` of a JSON value (as used by `"%s" % v` in the Laya runtime).
    fn py_str(v: &Value) -> String {
        match v {
            Value::String(s) => s.clone(),
            other => {
                let mut out = String::new();
                py_repr(other, &mut out);
                out
            }
        }
    }

    /// Python `repr()` of a value loaded by `json.loads`.
    fn py_repr(v: &Value, out: &mut String) {
        match v {
            Value::Null => out.push_str("None"),
            Value::Bool(b) => out.push_str(if *b { "True" } else { "False" }),
            Value::Number(n) => out.push_str(&py_number(n)),
            Value::String(s) => {
                // single quotes unless the string has a single quote and no double quote
                let q = if s.contains('\'') && !s.contains('"') {
                    '"'
                } else {
                    '\''
                };
                out.push(q);
                for c in s.chars() {
                    match c {
                        '\\' => out.push_str("\\\\"),
                        '\n' => out.push_str("\\n"),
                        '\r' => out.push_str("\\r"),
                        '\t' => out.push_str("\\t"),
                        c if c == q => {
                            out.push('\\');
                            out.push(c);
                        }
                        c if (c as u32) < 0x20 || c as u32 == 0x7f => {
                            out.push_str(&format!("\\x{:02x}", c as u32))
                        }
                        c => out.push(c),
                    }
                }
                out.push(q);
            }
            Value::Array(a) => {
                out.push('[');
                for (i, e) in a.iter().enumerate() {
                    if i > 0 {
                        out.push_str(", ");
                    }
                    py_repr(e, out);
                }
                out.push(']');
            }
            Value::Object(o) => {
                out.push('{');
                for (i, (k, e)) in o.iter().enumerate() {
                    if i > 0 {
                        out.push_str(", ");
                    }
                    py_repr(&Value::String(k.clone()), out);
                    out.push_str(": ");
                    py_repr(e, out);
                }
                out.push('}');
            }
        }
    }

    /// A JSON number as Python prints it after `json.loads`: 64-bit integers exactly, floats
    /// with `repr`. Integers beyond 64 bits and `-0` are read as floats (no `arbitrary_precision`).
    fn py_number(n: &serde_json::Number) -> String {
        let text = n.to_string(); // integers print exactly, floats with a `.`
        if !text.contains(['.', 'e', 'E']) {
            let t = text.strip_prefix('-').unwrap_or(&text);
            if t.bytes().all(|b| b == b'0') {
                return "0".into();
            }
            return text;
        }
        py_float_repr(text.parse::<f64>().unwrap_or(f64::NAN))
    }

    /// Python truthiness.
    fn truthy(v: &Value) -> bool {
        match v {
            Value::Null => false,
            Value::Bool(b) => *b,
            Value::Number(n) => n.as_f64() != Some(0.0),
            Value::String(s) => !s.is_empty(),
            Value::Array(a) => !a.is_empty(),
            Value::Object(o) => !o.is_empty(),
        }
    }

    impl DecisionRequest {
        /// Parse `{"state": ..., "questions": {"id": {"type", "instructions", "criteria"}}}`.
        pub fn from_json(s: &str) -> Result<Self> {
            let v: Value = serde_json::from_str(s).map_err(|e| bad(format!("JSON: {e}")))?;
            Self::from_value(&v)
        }

        pub fn from_value(v: &Value) -> Result<Self> {
            let state = match v.get("state") {
                None => return Err(bad("\"state\" must be provided")),
                Some(Value::String(s)) => s.clone(),
                Some(other) => py_json_dumps(other, false),
            };
            let qs = v
                .get("questions")
                .and_then(|q| q.as_object())
                .filter(|q| !q.is_empty())
                .ok_or_else(|| bad("\"questions\" must be a non-empty object"))?;
            let mut questions = Vec::with_capacity(qs.len());
            for (id, q) in qs {
                let err = |m: &str| bad(format!("questions.{id}: {m}"));
                let instructions = match q.get("instructions") {
                    None | Some(Value::Null) => {
                        return Err(err("\"instructions\" must be provided"))
                    }
                    Some(Value::String(s)) => s.clone(),
                    Some(other) => py_json_dumps(other, true),
                };
                let crit = q.get("criteria").unwrap_or(&Value::Null);
                let kind = match q.get("type").and_then(|t| t.as_str()).unwrap_or("") {
                    "choice" => {
                        let opts: Vec<(String, Option<String>)> = match crit {
                            Value::Object(o) => o
                                .iter()
                                .map(|(k, d)| (k.clone(), truthy(d).then(|| py_str(d))))
                                .collect(),
                            Value::Array(a) => {
                                // `{c: None for c in crit}`: duplicates collapse, first position kept
                                let mut v: Vec<(String, Option<String>)> = Vec::new();
                                for k in a.iter().map(py_str) {
                                    if !v.iter().any(|(e, _)| *e == k) {
                                        v.push((k, None));
                                    }
                                }
                                v
                            }
                            _ => Vec::new(),
                        };
                        if opts.is_empty() {
                            return Err(err("\"criteria\" must be a non-empty object or list"));
                        }
                        QuestionKind::Choice(opts)
                    }
                    // llama.cpp `server-decision.cpp`: 2 to 10 levels (HTTP 400 otherwise)
                    "score" => match crit {
                        Value::Array(a) if (2..=10).contains(&a.len()) => {
                            QuestionKind::Score(a.iter().map(py_str).collect())
                        }
                        _ => return Err(err("\"criteria\" must be an array of 2 to 10 levels")),
                    },
                    "noul" => {
                        let get = |k: &str| crit.get(k).filter(|d| truthy(d)).map(py_str);
                        match crit {
                            Value::Null | Value::Object(_) => QuestionKind::Noul {
                                false_desc: get("false"),
                                true_desc: get("true"),
                            },
                            _ => return Err(err("\"criteria\" must be an object")),
                        }
                    }
                    _ => return Err(err("\"type\" must be one of: choice, score, noul")),
                };
                questions.push(Question {
                    id: id.clone(),
                    instructions,
                    kind,
                });
            }
            Ok(DecisionRequest { state, questions })
        }
    }

    fn probs_obj(p: &[(String, f32)]) -> Value {
        Value::Object(
            p.iter()
                .map(|(k, v)| (k.clone(), Value::from(*v as f64)))
                .collect(),
        )
    }

    impl Answer {
        pub fn to_value(&self) -> Value {
            let mut m = Map::new();
            m.insert("type".into(), Value::from(self.qtype().name()));
            match self {
                Answer::Choice {
                    choice,
                    probabilities,
                    confidence,
                } => {
                    m.insert("choice".into(), Value::from(choice.as_str()));
                    m.insert("probabilities".into(), probs_obj(probabilities));
                    m.insert("confidence".into(), Value::from(*confidence as f64));
                }
                Answer::Score {
                    score,
                    legend,
                    probabilities,
                    confidence,
                } => {
                    m.insert("score".into(), Value::from(*score as f64));
                    m.insert(
                        "legend".into(),
                        Value::Object(
                            legend
                                .iter()
                                .map(|(k, v)| (k.clone(), Value::from(v.as_str())))
                                .collect(),
                        ),
                    );
                    m.insert("probabilities".into(), probs_obj(probabilities));
                    m.insert("confidence".into(), Value::from(*confidence as f64));
                }
                Answer::Noul { noul } => {
                    m.insert("noul".into(), Value::from(*noul as f64));
                }
            }
            Value::Object(m)
        }
    }

    impl DecisionResponse {
        pub fn to_value(&self) -> Value {
            let mut m = Map::new();
            m.insert("model".into(), Value::from(self.model.as_str()));
            m.insert(
                "answers".into(),
                Value::Object(
                    self.answers
                        .iter()
                        .enumerate()
                        .map(|(i, (k, a))| {
                            let mut v = a.to_value();
                            if let (Some(Some(e)), Value::Object(o)) = (self.exits.get(i), &mut v) {
                                o.insert("exit".into(), Value::from(e.exit));
                                o.insert("layers_used".into(), Value::from(e.layers_used));
                            }
                            (k.clone(), v)
                        })
                        .collect(),
                ),
            );
            let mut usage = Map::new();
            usage.insert("input_tokens".into(), Value::from(self.input_tokens));
            usage.insert("output_tokens".into(), Value::from(0));
            m.insert("usage".into(), Value::Object(usage));
            if let Some(us) = self.micros {
                m.insert("micros".into(), Value::from(us));
            }
            Value::Object(m)
        }

        pub fn to_json(&self) -> String {
            self.to_value().to_string()
        }
    }

    /// Serialize like Python `json.dumps(v, ensure_ascii=...)` with the default separators
    /// (`", "`, `": "`) and float `repr`. Keys keep their order.
    pub fn py_json_dumps(v: &Value, ensure_ascii: bool) -> String {
        let mut out = String::new();
        dump(v, ensure_ascii, &mut out);
        out
    }

    fn dump(v: &Value, ascii: bool, out: &mut String) {
        match v {
            Value::Null => out.push_str("null"),
            Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
            Value::Number(n) => out.push_str(&py_number(n)),
            Value::String(s) => dump_str(s, ascii, out),
            Value::Array(a) => {
                out.push('[');
                for (i, e) in a.iter().enumerate() {
                    if i > 0 {
                        out.push_str(", ");
                    }
                    dump(e, ascii, out);
                }
                out.push(']');
            }
            Value::Object(o) => {
                out.push('{');
                for (i, (k, e)) in o.iter().enumerate() {
                    if i > 0 {
                        out.push_str(", ");
                    }
                    dump_str(k, ascii, out);
                    out.push_str(": ");
                    dump(e, ascii, out);
                }
                out.push('}');
            }
        }
    }

    fn dump_str(s: &str, ascii: bool, out: &mut String) {
        out.push('"');
        for c in s.chars() {
            match c {
                '"' => out.push_str("\\\""),
                '\\' => out.push_str("\\\\"),
                '\n' => out.push_str("\\n"),
                '\r' => out.push_str("\\r"),
                '\t' => out.push_str("\\t"),
                '\u{8}' => out.push_str("\\b"),
                '\u{c}' => out.push_str("\\f"),
                c if (c as u32) < 0x20 || (ascii && (c as u32) >= 0x7f) => {
                    let mut buf = [0u16; 2];
                    for u in c.encode_utf16(&mut buf) {
                        out.push_str(&format!("\\u{u:04x}"));
                    }
                }
                c => out.push(c),
            }
        }
        out.push('"');
    }

    /// Python `repr(float)`: shortest round-trip digits, exponent form outside [1e-4, 1e16).
    pub fn py_float_repr(x: f64) -> String {
        if x.is_nan() {
            return "NaN".into();
        }
        if x.is_infinite() {
            return if x > 0.0 {
                "Infinity".into()
            } else {
                "-Infinity".into()
            };
        }
        if x == 0.0 {
            return if x.is_sign_negative() {
                "-0.0".into()
            } else {
                "0.0".into()
            };
        }
        let e = format!("{x:e}"); // e.g. "-1.2345e-7"
        let (mant, exp) = e.split_once('e').expect("exponent");
        let exp: i32 = exp.parse().expect("exponent");
        let (neg, mant) = mant.strip_prefix('-').map_or((false, mant), |m| (true, m));
        let digits: String = mant.chars().filter(|c| *c != '.').collect();
        let mut s = String::new();
        if neg {
            s.push('-');
        }
        if (-4..16).contains(&exp) {
            if exp < 0 {
                s.push_str("0.");
                s.push_str(&"0".repeat((-exp - 1) as usize));
                s.push_str(&digits);
            } else {
                let int_len = exp as usize + 1;
                if digits.len() <= int_len {
                    s.push_str(&digits);
                    s.push_str(&"0".repeat(int_len - digits.len()));
                    s.push_str(".0");
                } else {
                    s.push_str(&digits[..int_len]);
                    s.push('.');
                    s.push_str(&digits[int_len..]);
                }
            }
        } else {
            s.push_str(&digits[..1]);
            if digits.len() > 1 {
                s.push('.');
                s.push_str(&digits[1..]);
            }
            s.push_str(&format!(
                "e{}{:02}",
                if exp < 0 { '-' } else { '+' },
                exp.abs()
            ));
        }
        s
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn python_dumps() {
            let v: Value = serde_json::from_str(r#"{"b": [1, 2.5, -0.0001, 1e-5, 1e16, 123.0], "a": "é\n\"", "c": {"x": null, "y": true}}"#).unwrap();
            assert_eq!(
                py_json_dumps(&v, false),
                r#"{"b": [1, 2.5, -0.0001, 1e-05, 1e+16, 123.0], "a": "é\n\"", "c": {"x": null, "y": true}}"#
            );
            assert_eq!(
                py_json_dumps(&Value::from("\u{e9}\u{1f642}"), true),
                r#""\u00e9\ud83d\ude42""#
            );
            assert_eq!(py_float_repr(0.1), "0.1");
            assert_eq!(py_float_repr(1234567.125), "1234567.125");
            assert_eq!(py_float_repr(1.5e300), "1.5e+300");
        }

        #[test]
        fn python_numbers_and_repr() {
            let v: Value = serde_json::from_str(
                r#"[0.30000000000000004, 0.01988593426858726, 0, 9007199254740993, 1.0]"#,
            )
            .unwrap();
            assert_eq!(
                py_json_dumps(&v, false),
                "[0.30000000000000004, 0.01988593426858726, 0, 9007199254740993, 1.0]"
            );
            // Without serde_json's `arbitrary_precision` (it breaks internally tagged enums in
            // crates built alongside), integers beyond 64 bits and `-0` go through f64.
            let v: Value = serde_json::from_str("[-0, 123456789012345678901234567890]").unwrap();
            assert_eq!(py_json_dumps(&v, false), "[-0.0, 1.2345678901234568e+29]");
            let r = DecisionRequest::from_json(
                r#"{"state": null, "questions": {"q": {"type": "choice", "instructions": "i",
                    "criteria": {"a": {"x": [true, null, "it's"]}, "b": "plain"}},
                    "l": {"type": "choice", "instructions": "i", "criteria": ["a", "b", "a"]},
                    "s": {"type": "score", "instructions": "i", "criteria": ["lo", "hi"]}}}"#,
            )
            .unwrap();
            assert_eq!(r.state, "null");
            let QuestionKind::Choice(c) = &r.questions[0].kind else {
                panic!()
            };
            assert_eq!(c[0].1.as_deref(), Some(r#"{'x': [True, None, "it's"]}"#));
            let QuestionKind::Choice(l) = &r.questions[1].kind else {
                panic!()
            };
            assert_eq!(l.len(), 2);
            assert!(matches!(&r.questions[2].kind, QuestionKind::Score(s) if s.len() == 2));
            // score questions take 2 to 10 levels, as llama.cpp
            let score = |n: usize| {
                let levels: Vec<String> = (0..n).map(|i| format!("\"l{i}\"")).collect();
                DecisionRequest::from_json(&format!(
                    r#"{{"state": "s", "questions": {{"q": {{"type": "score", "instructions": "i", "criteria": [{}]}}}}}}"#,
                    levels.join(",")
                ))
            };
            assert!(score(0).is_err() && score(1).is_err() && score(11).is_err());
            assert!(score(2).is_ok() && score(10).is_ok());
            assert!(matches!(score(1), Err(crate::Error::InvalidRequest(_))));
        }

        #[test]
        fn parses_requests() {
            let r = DecisionRequest::from_json(
                r#"{"state": {"alt": 12}, "questions": {
                    "b": {"type": "choice", "instructions": "go?", "criteria": {"up": "move up", "down": null}},
                    "a": {"type": "noul", "instructions": {"k": "été"}},
                    "c": {"type": "score", "instructions": "risk", "criteria": ["low", "high"]},
                    "d": {"type": "choice", "instructions": "x", "criteria": ["p", "q"]}}}"#,
            )
            .unwrap();
            assert_eq!(r.state, r#"{"alt": 12}"#);
            assert_eq!(r.questions[0].id, "b");
            assert_eq!(
                r.questions[0].kind,
                QuestionKind::Choice(vec![
                    ("up".into(), Some("move up".into())),
                    ("down".into(), None)
                ])
            );
            assert_eq!(r.questions[1].instructions, r#"{"k": "\u00e9t\u00e9"}"#);
            assert_eq!(
                r.questions[1].kind,
                QuestionKind::Noul {
                    false_desc: None,
                    true_desc: None
                }
            );
            assert!(DecisionRequest::from_json(r#"{"questions": {}}"#).is_err());
            assert!(DecisionRequest::from_json(
                r#"{"state": "s", "questions": {"q": {"type": "x", "instructions": "i"}}}"#
            )
            .is_err());
        }
    }
}

#[cfg(feature = "json")]
pub use json_impl::{py_float_repr, py_json_dumps};
