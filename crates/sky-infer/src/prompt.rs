//! Decision prompt: option rendering, sequence building (`rl_common.build_sequence`),
//! temperature buckets and answer readout.
//!
//! Sequence: `[CLS] "{t} question: {ins}" [SEP] ([MASK] " {opt_i}")* [SEP] {state} [SEP]`.

use crate::tokenizer::{BpeCache, Tokenizer};
use crate::{Error, Result};

/// Longest option text, in tokens (without its marker).
pub const MAX_OPTION_TOKENS: usize = 48;

/// Question type; the index is the row of the type embedding.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum QType {
    Choice = 0,
    Score = 1,
    Noul = 2,
}

impl QType {
    pub fn name(self) -> &'static str {
        match self {
            QType::Choice => "choice",
            QType::Score => "score",
            QType::Noul => "noul",
        }
    }

    pub fn index(self) -> usize {
        self as usize
    }

    pub fn from_name(s: &str) -> Option<QType> {
        match s {
            "choice" => Some(QType::Choice),
            "score" => Some(QType::Score),
            "noul" => Some(QType::Noul),
            _ => None,
        }
    }
}

/// The options of a question, before rendering.
#[derive(Clone, Debug, PartialEq)]
pub enum QuestionKind {
    /// (key, optional description), in request order.
    Choice(Vec<(String, Option<String>)>),
    /// Level descriptions; the level of option `i` is `i`.
    Score(Vec<String>),
    /// Optional descriptions of the `false` and `true` options.
    Noul {
        false_desc: Option<String>,
        true_desc: Option<String>,
    },
}

impl QuestionKind {
    pub fn qtype(&self) -> QType {
        match self {
            QuestionKind::Choice(_) => QType::Choice,
            QuestionKind::Score(_) => QType::Score,
            QuestionKind::Noul { .. } => QType::Noul,
        }
    }

    pub fn n_options(&self) -> usize {
        match self {
            QuestionKind::Choice(o) => o.len(),
            QuestionKind::Score(l) => l.len(),
            QuestionKind::Noul { .. } => 2,
        }
    }
}

pub const NOUL_FALSE_DEFAULT: &str = "no, the statement does not hold";
pub const NOUL_TRUE_DEFAULT: &str = "yes, the statement holds";

/// Option texts in label order (`rl_common.render_options`): choice `key` or `key: desc`,
/// score `level i: desc`, noul `["false: ...", "true: ..."]` (so `p[1]` is the noul answer).
pub fn render_options(kind: &QuestionKind) -> Vec<String> {
    let nonempty = |d: &Option<String>| d.as_deref().filter(|s| !s.is_empty()).map(str::to_string);
    match kind {
        QuestionKind::Choice(opts) => opts
            .iter()
            .map(|(k, d)| match nonempty(d) {
                Some(d) => format!("{k}: {d}"),
                None => k.clone(),
            })
            .collect(),
        QuestionKind::Score(levels) => levels
            .iter()
            .enumerate()
            .map(|(i, c)| format!("level {i}: {c}"))
            .collect(),
        QuestionKind::Noul {
            false_desc,
            true_desc,
        } => vec![
            format!(
                "false: {}",
                nonempty(false_desc).unwrap_or_else(|| NOUL_FALSE_DEFAULT.to_string())
            ),
            format!(
                "true: {}",
                nonempty(true_desc).unwrap_or_else(|| NOUL_TRUE_DEFAULT.to_string())
            ),
        ],
    }
}

/// The static part of a sequence (question + options), reusable across states.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Prefix {
    /// `[CLS] head [SEP] ([MASK] opt)* [SEP]`
    pub ids: Vec<u32>,
    pub markers: Vec<usize>,
}

/// Special ids needed by the prompt.
#[derive(Clone, Copy, Debug)]
pub struct PromptTokens<'a> {
    pub cls: u32,
    pub sep: u32,
    pub mask: u32,
    pub mask_text: &'a str,
}

impl<'a> PromptTokens<'a> {
    pub fn from_tokenizer(tok: &'a Tokenizer) -> Result<Self> {
        let sp = tok.special();
        let missing = |n: &str| Error::Missing(format!("{n} token in the tokenizer"));
        let mask = sp.mask.ok_or_else(|| missing("mask"))?;
        Ok(PromptTokens {
            cls: sp.cls.ok_or_else(|| missing("cls"))?,
            sep: sp.sep.ok_or_else(|| missing("sep"))?,
            mask,
            mask_text: tok.id_to_token(mask).ok_or_else(|| missing("mask"))?,
        })
    }
}

fn encode(tok: &Tokenizer, text: &str, cache: Option<&mut BpeCache>) -> Vec<u32> {
    let mut out = Vec::new();
    tok.encode_into(text, &mut out, cache);
    out
}

/// Tokenize and truncate the question and options exactly like `build_sequence`.
pub fn build_prefix(
    tok: &Tokenizer,
    pt: &PromptTokens<'_>,
    qtype: QType,
    instructions: &str,
    options: &[&str],
    head_max_len: usize,
    mut cache: Option<&mut BpeCache>,
) -> Prefix {
    let ins = instructions.replace(pt.mask_text, " ");
    let head = encode(
        tok,
        &format!("{} question: {ins}", qtype.name()),
        cache.as_deref_mut(),
    );
    let mut opt_ids: Vec<Vec<u32>> = options
        .iter()
        .map(|o| {
            let mut ids = vec![pt.mask];
            let t = encode(
                tok,
                &format!(" {}", o.replace(pt.mask_text, " ")),
                cache.as_deref_mut(),
            );
            ids.extend_from_slice(&t[..t.len().min(MAX_OPTION_TOKENS)]);
            ids
        })
        .collect();
    let hml = head_max_len as i64;
    let total = |o: &Vec<Vec<u32>>| o.iter().map(|v| v.len() as i64).sum::<i64>();
    let mut opt_budget = hml - total(&opt_ids);
    if opt_budget < 16 {
        // too many / too long options: shrink every option evenly
        let per = (hml - 16).div_euclid(opt_ids.len().max(1) as i64).max(4) as usize;
        for o in &mut opt_ids {
            o.truncate(per);
        }
        opt_budget = hml - total(&opt_ids);
    }
    let head_len = opt_budget.max(8) as usize;
    let mut ids = Vec::with_capacity(2 + head.len().min(head_len) + total(&opt_ids) as usize + 1);
    ids.push(pt.cls);
    ids.extend_from_slice(&head[..head.len().min(head_len)]);
    ids.push(pt.sep);
    let mut markers = Vec::with_capacity(opt_ids.len());
    for o in &opt_ids {
        markers.push(ids.len());
        ids.extend_from_slice(o);
    }
    ids.push(pt.sep);
    Prefix { ids, markers }
}

/// Complete a prefix with the state (truncated to the room left) and the final `[SEP]`.
/// Writes into `ids` / `markers` (cleared first). Markers past `max_len` are dropped.
#[allow(clippy::too_many_arguments)]
pub fn finish_sequence(
    tok: &Tokenizer,
    pt: &PromptTokens<'_>,
    prefix: &Prefix,
    state: &str,
    max_len: usize,
    ids: &mut Vec<u32>,
    markers: &mut Vec<usize>,
    cache: Option<&mut BpeCache>,
) {
    ids.clear();
    markers.clear();
    ids.extend_from_slice(&prefix.ids);
    let room = max_len.saturating_sub(ids.len() + 1);
    let start = ids.len();
    let replaced;
    let state = if state.contains(pt.mask_text) {
        replaced = state.replace(pt.mask_text, " ");
        replaced.as_str()
    } else {
        state
    };
    tok.encode_into(state, ids, cache);
    ids.truncate(start + room);
    ids.push(pt.sep);
    ids.truncate(max_len);
    markers.extend(prefix.markers.iter().copied().filter(|&m| m < max_len));
}

/// `build_sequence(tok, state, q, max_len, head_max_len)`: token ids and marker positions.
pub fn build_sequence(
    tok: &Tokenizer,
    qtype: QType,
    instructions: &str,
    options: &[&str],
    state: &str,
    max_len: usize,
    head_max_len: usize,
) -> Result<(Vec<u32>, Vec<usize>)> {
    let pt = PromptTokens::from_tokenizer(tok)?;
    let prefix = build_prefix(tok, &pt, qtype, instructions, options, head_max_len, None);
    let (mut ids, mut markers) = (Vec::new(), Vec::new());
    finish_sequence(
        tok,
        &pt,
        &prefix,
        state,
        max_len,
        &mut ids,
        &mut markers,
        None,
    );
    Ok((ids, markers))
}

/// Option-count bucket of the temperatures: `2`, `3_5`, `6_10` or `11`.
pub fn temp_bucket(n_options: usize) -> &'static str {
    match n_options {
        0..=2 => "2",
        3..=5 => "3_5",
        6..=10 => "6_10",
        _ => "11",
    }
}

/// `softmax(logits / temperature)`.
pub fn probs_from_logits(logits: &[f32], temperature: f32) -> Vec<f32> {
    let m = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let e: Vec<f64> = logits
        .iter()
        .map(|&s| (((s - m) / temperature) as f64).exp())
        .collect();
    let sum: f64 = e.iter().sum();
    e.iter().map(|v| (v / sum) as f32).collect()
}

/// Choice confidence `(p_max - 1/n) / (1 - 1/n)` (llama.cpp server, TypeSafe formula).
pub fn confidence_choice(p: &[f32]) -> f32 {
    if p.len() < 2 {
        return 1.0;
    }
    let u = 1.0 / p.len() as f32;
    let pmax = p.iter().fold(0.0f32, |a, &b| a.max(b));
    ((pmax - u) / (1.0 - u)).max(0.0)
}

/// Score confidence: 1 - (mean distance to the mode) / (that of a uniform distribution around
/// the center) (llama.cpp server, TypeSafe formula).
pub fn confidence_score(p: &[f32]) -> f32 {
    let n = p.len();
    if n < 2 {
        return 1.0;
    }
    let mode = argmax(p);
    let mut dist = 0.0f64;
    let mut dist_uniform = 0.0f64;
    for (i, &pi) in p.iter().enumerate() {
        dist += pi as f64 * (i as f64 - mode as f64).abs();
        dist_uniform += (i as f64 - (n as f64 - 1.0) / 2.0).abs() / n as f64;
    }
    (1.0 - dist / dist_uniform).max(0.0) as f32
}

/// Laya runtime confidence: 1 - normalized entropy (`rl_common.confidence_from_probs`).
pub fn confidence_entropy(p: &[f32]) -> f32 {
    let k = p.len();
    if k < 2 {
        return 1.0;
    }
    let ent: f64 = -p
        .iter()
        .map(|&v| v as f64 * (v as f64).clamp(1e-12, 1.0).ln())
        .sum::<f64>();
    (1.0 - ent / (k as f64).ln()) as f32
}

/// Index of the largest value (first one on ties).
pub fn argmax(p: &[f32]) -> usize {
    let mut best = 0;
    for (i, &v) in p.iter().enumerate() {
        if v > p[best] {
            best = i;
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn renders_options_like_laya() {
        let c = QuestionKind::Choice(vec![
            ("flap".into(), Some("go up".into())),
            ("wait".into(), None),
            ("x".into(), Some(String::new())),
        ]);
        assert_eq!(render_options(&c), vec!["flap: go up", "wait", "x"]);
        let s = QuestionKind::Score(vec!["low".into(), "high".into()]);
        assert_eq!(render_options(&s), vec!["level 0: low", "level 1: high"]);
        let n = QuestionKind::Noul {
            false_desc: None,
            true_desc: Some("it is".into()),
        };
        assert_eq!(
            render_options(&n),
            vec!["false: no, the statement does not hold", "true: it is"]
        );
    }

    #[test]
    fn buckets_and_confidence() {
        assert_eq!(
            [2, 3, 5, 6, 10, 11, 40].map(temp_bucket),
            ["2", "3_5", "3_5", "6_10", "6_10", "11", "11"]
        );
        assert!((confidence_choice(&[0.25, 0.25, 0.25, 0.25])).abs() < 1e-6);
        assert!((confidence_choice(&[1.0, 0.0]) - 1.0).abs() < 1e-6);
        assert!((confidence_score(&[0.0, 1.0, 0.0]) - 1.0).abs() < 1e-6);
        assert!((confidence_entropy(&[0.5, 0.5])).abs() < 1e-6);
        let p = probs_from_logits(&[1.0, 2.0, 3.0], 2.0);
        assert!((p.iter().sum::<f32>() - 1.0).abs() < 1e-6 && p[2] > p[1]);
    }
}
