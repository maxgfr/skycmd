//! s5 decision records: options as `(id, definition)` pairs (`opts_kv`), so that the label and
//! the definition can be rendered separately.
//!
//! - **native** rendering (Laya, variant A): `"{id}: {definition}"` for `choice`,
//!   `"level {i}: {definition}"` for `score`, `"false: …"` / `"true: …"` for `noul` — the
//!   strings of the existing decision records;
//! - **definition-only** rendering (variant B, as the "von" runtime of 2610.02586): the
//!   definition alone. The level of a `score` option stays in its text (`"level {i} of {K}: …"`):
//!   it is the rubric's order, not a label. A `noul` option without a definition gets the generic
//!   one of its language (`"The statement is false."` / `"The statement is true."`).

use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};

use crate::sequence::{qtype_id, DecisionRecord};

/// One s5 record (JSONL).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct S5Record {
    pub src: String,
    pub t: String,
    pub ins: String,
    /// `(id, definition)` per option, in record order.
    pub opts_kv: Vec<(String, String)>,
    #[serde(default)]
    pub state: String,
    pub target: Vec<f32>,
    pub split: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub uid: String,
    /// `en` or `fr` (empty: English).
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub lang: String,
}

/// How options are rendered into the model input.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Render {
    /// `id: definition` (Laya / the existing records).
    Native,
    /// The definition only.
    Definition,
}

impl Render {
    pub fn parse(s: &str) -> Result<Self> {
        Ok(match s {
            "native" => Render::Native,
            "definition" | "von" => Render::Definition,
            other => bail!("unknown rendering {other:?} (native, definition)"),
        })
    }
}

/// Generic `noul` definitions (false, true) of a language.
pub fn generic_noul(lang: &str) -> (&'static str, &'static str) {
    match lang {
        "fr" => ("L'affirmation est fausse.", "L'affirmation est vraie."),
        _ => ("The statement is false.", "The statement is true."),
    }
}

impl S5Record {
    /// Rendered option texts, in record order.
    pub fn render(&self, r: Render) -> Vec<String> {
        let k = self.opts_kv.len();
        let generic = generic_noul(&self.lang);
        self.opts_kv
            .iter()
            .enumerate()
            .map(|(i, (id, def))| match (self.t.as_str(), r) {
                ("score", Render::Native) => format!("level {i}: {def}"),
                ("score", Render::Definition) => format!("level {i} of {}: {def}", k - 1),
                ("noul", Render::Definition) if def.is_empty() => {
                    (if i == 0 { generic.0 } else { generic.1 }).to_string()
                }
                (_, Render::Native) if def.is_empty() => id.clone(),
                (_, Render::Native) => format!("{id}: {def}"),
                (_, Render::Definition) if def.is_empty() => id.clone(),
                (_, Render::Definition) => def.clone(),
            })
            .collect()
    }

    /// The s1 record with options rendered by `r`.
    pub fn to_decision(&self, r: Render) -> DecisionRecord {
        DecisionRecord {
            src: self.src.clone(),
            t: self.t.clone(),
            ins: self.ins.clone(),
            opts: self.render(r),
            state: self.state.clone(),
            target: self.target.clone(),
            split: self.split.clone(),
        }
    }

    /// The s5 record of an s1 record (options split by [`split_legacy`]).
    pub fn from_decision(d: &DecisionRecord, uid: &str, lang: &str) -> Self {
        Self {
            src: d.src.clone(),
            t: d.t.clone(),
            ins: d.ins.clone(),
            opts_kv: d
                .opts
                .iter()
                .enumerate()
                .map(|(i, o)| split_legacy(&d.t, i, o))
                .collect(),
            state: d.state.clone(),
            target: d.target.clone(),
            split: d.split.clone(),
            uid: uid.to_string(),
            lang: lang.to_string(),
        }
    }

    pub fn validate(&self) -> Result<()> {
        if qtype_id(&self.t).is_none() {
            bail!("unknown question type {:?}", self.t);
        }
        if self.opts_kv.len() < 2 || self.opts_kv.len() != self.target.len() {
            bail!(
                "{} options for {} targets",
                self.opts_kv.len(),
                self.target.len()
            );
        }
        if self.t == "noul"
            && (self.opts_kv.len() != 2
                || self.opts_kv[0].0 != "false"
                || self.opts_kv[1].0 != "true")
        {
            bail!("noul options must be [false, true]");
        }
        if !self.target.iter().all(|x| x.is_finite() && *x >= 0.0)
            || self.target.iter().sum::<f32>() <= 0.0
        {
            bail!("invalid target {:?}", self.target);
        }
        Ok(())
    }
}

/// Splits a rendered s1 option back into `(id, definition)`: `level i: d` and `false: d` /
/// `true: d` by their fixed prefixes; a `choice` option at its first `": "` when the part
/// before it looks like a key (at most 40 characters, no line break), else it is a bare key.
pub fn split_legacy(t: &str, i: usize, opt: &str) -> (String, String) {
    match t {
        "score" => {
            let rest = opt.strip_prefix(&format!("level {i}:")).unwrap_or(opt);
            (i.to_string(), rest.trim().to_string())
        }
        "noul" => {
            let id = if i == 0 { "false" } else { "true" };
            let rest = opt.strip_prefix(&format!("{id}:")).unwrap_or(opt).trim();
            // A bare `false` / `true` has no definition (the generic one is rendered).
            let def = if rest == id { "" } else { rest };
            (id.to_string(), def.to_string())
        }
        _ => match opt.split_once(": ") {
            Some((k, d)) if !k.trim().is_empty() && k.len() <= 40 && !k.contains('\n') => {
                (k.trim().to_string(), d.trim().to_string())
            }
            _ => (opt.trim().to_string(), String::new()),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rec(t: &str, kv: &[(&str, &str)]) -> S5Record {
        S5Record {
            src: "x".into(),
            t: t.into(),
            ins: "q".into(),
            opts_kv: kv
                .iter()
                .map(|(a, b)| (a.to_string(), b.to_string()))
                .collect(),
            state: "s".into(),
            target: vec![1.0; kv.len()],
            split: "train".into(),
            uid: "u".into(),
            lang: String::new(),
        }
    }

    #[test]
    fn native_round_trip_and_definition_only() {
        let c = rec(
            "choice",
            &[("billing", "A charge problem."), ("refund", "Money back.")],
        );
        assert_eq!(
            c.render(Render::Native),
            ["billing: A charge problem.", "refund: Money back."]
        );
        assert_eq!(
            c.render(Render::Definition),
            ["A charge problem.", "Money back."]
        );
        let back = S5Record::from_decision(&c.to_decision(Render::Native), "u", "");
        assert_eq!(back, c);

        let s = rec("score", &[("0", "None."), ("1", "Some."), ("2", "All.")]);
        assert_eq!(s.render(Render::Native)[1], "level 1: Some.");
        assert_eq!(s.render(Render::Definition)[2], "level 2 of 2: All.");
        assert_eq!(
            S5Record::from_decision(&s.to_decision(Render::Native), "u", ""),
            s
        );

        let n = rec("noul", &[("false", "No."), ("true", "Yes.")]);
        assert_eq!(n.render(Render::Native), ["false: No.", "true: Yes."]);
        assert_eq!(
            S5Record::from_decision(&n.to_decision(Render::Native), "u", ""),
            n
        );
        let mut g = rec("noul", &[("false", ""), ("true", "")]);
        assert_eq!(g.render(Render::Definition)[1], "The statement is true.");
        g.lang = "fr".into();
        assert_eq!(g.render(Render::Definition)[0], "L'affirmation est fausse.");
    }

    #[test]
    fn legacy_split_edge_cases() {
        assert_eq!(split_legacy("choice", 0, "up"), ("up".into(), "".into()));
        assert_eq!(
            split_legacy("choice", 0, "no_tool: answer directly: no call"),
            ("no_tool".into(), "answer directly: no call".into())
        );
        let long = format!("{}: x", "a".repeat(60));
        assert_eq!(split_legacy("choice", 0, &long).1, "");
        assert_eq!(
            split_legacy("score", 3, "level 3: High"),
            ("3".into(), "High".into())
        );
        assert_eq!(split_legacy("noul", 1, "true"), ("true".into(), "".into()));
        assert_eq!(
            split_legacy("noul", 0, "false: it is not"),
            ("false".into(), "it is not".into())
        );
    }

    #[test]
    fn validation() {
        assert!(rec("choice", &[("a", ""), ("b", "")]).validate().is_ok());
        assert!(rec("choice", &[("a", "")]).validate().is_err());
        assert!(rec("noul", &[("true", ""), ("false", "")])
            .validate()
            .is_err());
        let mut r = rec("choice", &[("a", ""), ("b", "")]);
        r.target = vec![0.0, 0.0];
        assert!(r.validate().is_err());
    }
}
