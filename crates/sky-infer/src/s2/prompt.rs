//! s2 streams (SPEC-S2 §2–§3): numeric extraction before BPE, the static stream
//! `[CLS] "{t} question: {ins}" [SEP] ([MASK] " {opt_i}")*` with the SPEC.md truncation rules, and
//! the state stream `[CLS] {state} [SEP]`.
//!
//! With `fuse_numbers` (SPEC-S3 §1) every text is merged right after tokenization, before the
//! truncation rules: a `[NUM]` moves its value onto the token before it, which then carries
//! `has_value = 1`. Every stream therefore has a `has_value` flag per token (for s2 files it is
//! simply `id == [NUM]`).

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
    /// Merge `[NUM]` tokens into the token before them (SPEC-S3 §1).
    pub fuse: bool,
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
/// every `[NUM]` token (0 elsewhere) and its `has_value` flag. With `pt.fuse`, the appended tokens
/// are merged (see [`fuse_numbers`]); the first one is treated as the start of a stream.
pub fn encode_numeric(
    tok: &Tokenizer,
    pt: &S2PromptTokens<'_>,
    text: &str,
    ids: &mut Vec<u32>,
    values: &mut Vec<f32>,
    has_value: &mut Vec<bool>,
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
            has_value.push(true);
            k += 1;
        } else {
            values.push(0.0);
            has_value.push(false);
        }
    }
    if pt.fuse {
        fuse_numbers(pt, ids, values, has_value, start);
    }
}

/// The merge rule of SPEC-S3 §1 on the tokens `from..` of a stream (`ids[from - 1]`, if any, is
/// the token before them): each `[NUM]` whose previous token is not `[CLS]` / `[SEP]` / `[MASK]`
/// and does not carry a value yet is removed, and its value moves onto that previous token. Other
/// `[NUM]` tokens stay standalone (id `[NUM]`, `has_value = 1`), as in s2.
pub fn fuse_numbers(
    pt: &S2PromptTokens<'_>,
    ids: &mut Vec<u32>,
    values: &mut Vec<f32>,
    has_value: &mut Vec<bool>,
    from: usize,
) {
    let mut w = from;
    for r in from..ids.len() {
        let (id, v) = (ids[r], values[r]);
        if id == pt.num && w > 0 {
            let prev = ids[w - 1];
            if prev != pt.cls && prev != pt.sep && prev != pt.mask && !has_value[w - 1] {
                values[w - 1] = v;
                has_value[w - 1] = true;
                continue;
            }
        }
        ids[w] = id;
        values[w] = v;
        has_value[w] = has_value[r];
        w += 1;
    }
    ids.truncate(w);
    values.truncate(w);
    has_value.truncate(w);
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
    type Stream = (Vec<u32>, Vec<f32>, Vec<bool>);
    let mut head: Stream = Default::default();
    let head_text = format!("{} question: {}", qtype.name(), sanitize(instructions, pt));
    encode_numeric(
        tok,
        pt,
        &head_text,
        &mut head.0,
        &mut head.1,
        &mut head.2,
        cache.as_deref_mut(),
    );
    let truncate = |s: &mut Stream, n: usize| {
        s.0.truncate(n);
        s.1.truncate(n);
        s.2.truncate(n);
    };
    let mut opts: Vec<Stream> = options
        .iter()
        .map(|o| {
            let mut s: Stream = (vec![pt.mask], vec![0.0], vec![false]);
            encode_numeric(
                tok,
                pt,
                &format!(" {o}"),
                &mut s.0,
                &mut s.1,
                &mut s.2,
                cache.as_deref_mut(),
            );
            truncate(&mut s, 1 + MAX_OPTION_TOKENS);
            s
        })
        .collect();
    let hml = max_head_tokens as i64;
    let total = |o: &[Stream]| o.iter().map(|v| v.0.len() as i64).sum::<i64>();
    let mut budget = hml - total(&opts);
    if budget < 16 {
        let per = (hml - 16).div_euclid(opts.len().max(1) as i64).max(4) as usize;
        for s in &mut opts {
            truncate(s, per);
        }
        budget = hml - total(&opts);
    }
    let head_len = head.0.len().min(budget.max(8) as usize);
    let mut st = StaticTokens::default();
    st.ids.push(pt.cls);
    st.ids.extend_from_slice(&head.0[..head_len]);
    st.ids.push(pt.sep);
    st.values.push(0.0);
    st.values.extend_from_slice(&head.1[..head_len]);
    st.values.push(0.0);
    st.has_value.push(false);
    st.has_value.extend_from_slice(&head.2[..head_len]);
    st.has_value.push(false);
    st.q_len = st.ids.len();
    for (ids, vals, hv) in &opts {
        st.options.push((st.ids.len(), ids.len()));
        st.ids.extend_from_slice(ids);
        st.values.extend_from_slice(vals);
        st.has_value.extend_from_slice(hv);
    }
    st
}

/// The state stream `[CLS] {state} [SEP]`, truncated to `max_state_tokens` (written into `ids` /
/// `values` / `has_value`, cleared first).
#[allow(clippy::too_many_arguments)]
pub fn build_state(
    tok: &Tokenizer,
    pt: &S2PromptTokens<'_>,
    state: &str,
    max_state_tokens: usize,
    ids: &mut Vec<u32>,
    values: &mut Vec<f32>,
    has_value: &mut Vec<bool>,
    cache: Option<&mut BpeCache>,
) {
    ids.clear();
    values.clear();
    has_value.clear();
    ids.push(pt.cls);
    values.push(0.0);
    has_value.push(false);
    encode_numeric(tok, pt, state, ids, values, has_value, cache);
    let keep = max_state_tokens.saturating_sub(1).max(1);
    ids.truncate(keep);
    values.truncate(keep);
    has_value.truncate(keep);
    ids.push(pt.sep);
    values.push(0.0);
    has_value.push(false);
}
