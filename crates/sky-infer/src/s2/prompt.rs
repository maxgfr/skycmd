//! s2 streams (SPEC-S2 §2–§3): numeric extraction before BPE, the static stream
//! `[CLS] "{t} question: {ins}" [SEP] ([MASK] " {opt_i}")*` with the SPEC.md truncation rules, and
//! the state stream `[CLS] {state} [SEP]`.

use super::model::StaticTokens;
use crate::num::{self, NUM_TOKEN};
use crate::prompt::{QType, MAX_OPTION_TOKENS};
use crate::tokenizer::{BpeCache, Tokenizer};

/// Special ids and texts of the s2 prompts.
#[derive(Clone, Copy, Debug)]
pub struct S2PromptTokens<'a> {
    pub cls: u32,
    pub sep: u32,
    pub mask: u32,
    pub num: u32,
    pub mask_text: &'a str,
}

/// Remove the text of special tokens that must not come from user text (`[MASK]`, `[NUM]`).
fn sanitize<'t>(text: &'t str, pt: &S2PromptTokens<'_>) -> std::borrow::Cow<'t, str> {
    let mut t = std::borrow::Cow::Borrowed(text);
    for special in [pt.mask_text, NUM_TOKEN] {
        if !special.is_empty() && t.contains(special) {
            t = std::borrow::Cow::Owned(t.replace(special, " "));
        }
    }
    t
}

/// Tokenize `text` with numeric extraction: appends the ids and, aligned with them, the value of
/// every `[NUM]` token (0 elsewhere).
pub fn encode_numeric(
    tok: &Tokenizer,
    pt: &S2PromptTokens<'_>,
    text: &str,
    ids: &mut Vec<u32>,
    values: &mut Vec<f32>,
    cache: Option<&mut BpeCache>,
) {
    let text = sanitize(text, pt);
    let ex = num::extract(&text);
    let start = ids.len();
    tok.encode_into(&ex.text, ids, cache);
    let mut k = 0;
    for &id in &ids[start..] {
        if id == pt.num {
            values.push(ex.values.get(k).copied().unwrap_or(0.0));
            k += 1;
        } else {
            values.push(0.0);
        }
    }
}

/// The static stream of a question, truncated like SPEC.md: each option to 48 tokens (plus its
/// marker), options shrunk evenly when they leave less than 16 tokens of the budget, and the
/// question text to `max(8, budget left)`.
pub fn build_static(
    tok: &Tokenizer,
    pt: &S2PromptTokens<'_>,
    qtype: QType,
    instructions: &str,
    options: &[&str],
    max_head_tokens: usize,
    mut cache: Option<&mut BpeCache>,
) -> StaticTokens {
    let (mut head, mut head_v) = (Vec::new(), Vec::new());
    let head_text = format!("{} question: {}", qtype.name(), sanitize(instructions, pt));
    encode_numeric(
        tok,
        pt,
        &head_text,
        &mut head,
        &mut head_v,
        cache.as_deref_mut(),
    );
    let mut opts: Vec<(Vec<u32>, Vec<f32>)> = options
        .iter()
        .map(|o| {
            let (mut ids, mut vals) = (vec![pt.mask], vec![0.0]);
            encode_numeric(
                tok,
                pt,
                &format!(" {o}"),
                &mut ids,
                &mut vals,
                cache.as_deref_mut(),
            );
            ids.truncate(1 + MAX_OPTION_TOKENS);
            vals.truncate(1 + MAX_OPTION_TOKENS);
            (ids, vals)
        })
        .collect();
    let hml = max_head_tokens as i64;
    let total = |o: &[(Vec<u32>, Vec<f32>)]| o.iter().map(|v| v.0.len() as i64).sum::<i64>();
    let mut budget = hml - total(&opts);
    if budget < 16 {
        let per = (hml - 16).div_euclid(opts.len().max(1) as i64).max(4) as usize;
        for (ids, vals) in &mut opts {
            ids.truncate(per);
            vals.truncate(per);
        }
        budget = hml - total(&opts);
    }
    let head_len = head.len().min(budget.max(8) as usize);
    let mut st = StaticTokens::default();
    st.ids.push(pt.cls);
    st.ids.extend_from_slice(&head[..head_len]);
    st.ids.push(pt.sep);
    st.values.push(0.0);
    st.values.extend_from_slice(&head_v[..head_len]);
    st.values.push(0.0);
    st.q_len = st.ids.len();
    for (ids, vals) in &opts {
        st.options.push((st.ids.len(), ids.len()));
        st.ids.extend_from_slice(ids);
        st.values.extend_from_slice(vals);
    }
    st
}

/// The state stream `[CLS] {state} [SEP]`, truncated to `max_state_tokens` (written into `ids` /
/// `values`, cleared first).
pub fn build_state(
    tok: &Tokenizer,
    pt: &S2PromptTokens<'_>,
    state: &str,
    max_state_tokens: usize,
    ids: &mut Vec<u32>,
    values: &mut Vec<f32>,
    cache: Option<&mut BpeCache>,
) {
    ids.clear();
    values.clear();
    ids.push(pt.cls);
    values.push(0.0);
    encode_numeric(tok, pt, state, ids, values, cache);
    let keep = max_state_tokens.saturating_sub(1).max(1);
    ids.truncate(keep);
    values.truncate(keep);
    ids.push(pt.sep);
    values.push(0.0);
}
