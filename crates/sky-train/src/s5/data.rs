//! s5 training data: tokenized items, the augmentation stream shared by every variant, pairs
//! and the batches of variant A (Laya layout, `id: definition` options).
//!
//! **One augmentation stream for all variants.** Each sampled question gets an option order
//! (not for `score`), and, for `choice`, random option names with probability `p_names`; a
//! `noul` question has its two names swapped (polarity) with probability `p_polarity`, the
//! target following the definitions. The draws are the same whatever the variant: variant A′
//! renders the new names, variant B (definitions only) never sees them, so A′ and B train on
//! exactly the same augmented data.

use std::collections::HashMap;

use anyhow::{bail, Result};
use mlx_rs::Array;

use crate::config::SpecialTokens;
use crate::decision::SCORE;
use crate::rng::Rng;
use crate::s5::record::{generic_noul, Render, S5Record};
use crate::sequence::{build_from_pieces_with, qtype_id, Pieces, Tok, OPTION_MAX_TOKENS};

/// One tokenized question.
#[derive(Debug, Clone)]
pub struct S5Item {
    pub rec: S5Record,
    pub qtype: usize,
    /// Normalized target, record order.
    pub target: Vec<f32>,
    /// `"{t} question: {ins}"`.
    pub head: Vec<u32>,
    pub state: Vec<u32>,
    /// `" " + definition-only rendering` per option (variant B).
    pub defs: Vec<Vec<u32>>,
    /// `" " + native rendering` per option, with the record's own names (variants A, A′).
    pub native: Vec<Vec<u32>>,
    /// Index of the pair partner (same question, another rendering), if any.
    pub pair: Option<usize>,
}

impl S5Item {
    pub fn k(&self) -> usize {
        self.target.len()
    }
}

/// Tokenizes records into items (in record order). `pair` links records by `pair_of` uid.
pub fn tokenize(tok: &Tok, sp: &SpecialTokens, recs: Vec<S5Record>) -> Result<Vec<S5Item>> {
    let mut texts = Vec::new();
    for r in &recs {
        r.validate()?;
        let clean = |s: &str| s.replace(&sp.mask_text, " ");
        texts.push(clean(&format!("{} question: {}", r.t, r.ins)));
        texts.push(clean(&r.state));
        for o in r.render(Render::Definition) {
            texts.push(format!(" {}", clean(&o)));
        }
        for o in r.render(Render::Native) {
            texts.push(format!(" {}", clean(&o)));
        }
    }
    let mut ids = Vec::with_capacity(texts.len());
    for chunk in texts.chunks(50_000) {
        ids.extend(tok.encode_batch(chunk.to_vec())?);
    }
    let mut it = ids.into_iter();
    let mut out = Vec::with_capacity(recs.len());
    for rec in recs {
        let k = rec.opts_kv.len();
        let head = it.next().unwrap_or_default();
        let state = it.next().unwrap_or_default();
        let cut = |mut v: Vec<u32>| {
            v.truncate(OPTION_MAX_TOKENS);
            v
        };
        let defs: Vec<Vec<u32>> = (0..k).map(|_| cut(it.next().unwrap_or_default())).collect();
        let native: Vec<Vec<u32>> = (0..k).map(|_| cut(it.next().unwrap_or_default())).collect();
        let sum: f32 = rec.target.iter().sum();
        let Some(qtype) = qtype_id(&rec.t) else {
            bail!("unknown type {}", rec.t);
        };
        out.push(S5Item {
            qtype,
            target: rec.target.iter().map(|x| x / sum).collect(),
            head,
            state,
            defs,
            native,
            pair: None,
            rec,
        });
    }
    Ok(out)
}

/// Links each item to its pair partner: records sharing a non-empty `pair_key` (the part of
/// the uid before `#`) form pairs, first with second.
pub fn link_pairs(items: &mut [S5Item]) {
    let mut first: HashMap<String, usize> = HashMap::new();
    for i in 0..items.len() {
        let Some((key, _)) = items[i].rec.uid.split_once('#') else {
            continue;
        };
        let key = key.to_string();
        match first.get(&key) {
            Some(&j) if items[j].pair.is_none() && items[j].k() == items[i].k() => {
                items[j].pair = Some(i);
                items[i].pair = Some(j);
            }
            None => {
                first.insert(key, i);
            }
            _ => {}
        }
    }
}

/// Augmentation settings.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Augment {
    /// Shuffle the options of choice / noul questions.
    pub permute: bool,
    /// Probability of random option names on a choice question.
    pub p_names: f64,
    /// Probability of swapping the false / true names of a noul question.
    pub p_polarity: f64,
}

impl Default for Augment {
    fn default() -> Self {
        Self {
            permute: true,
            p_names: 0.5,
            p_polarity: 0.25,
        }
    }
}

/// The augmentation drawn for one sampled question.
#[derive(Debug, Clone, PartialEq)]
pub struct Draw {
    /// Option order: position `j` shows record option `order[j]`.
    pub order: Vec<usize>,
    /// New names per record option (`None`: the record's own names).
    pub names: Option<Vec<String>>,
}

/// A random option name (lowercase letters and digits, starting with a letter).
fn random_name(rng: &mut Rng) -> String {
    const L: &[u8] = b"abcdefghijklmnopqrstuvwxyz";
    const A: &[u8] = b"abcdefghijklmnopqrstuvwxyz0123456789";
    let n = 3 + rng.below(4);
    let mut s = String::with_capacity(n);
    s.push(L[rng.below(L.len())] as char);
    for _ in 1..n {
        s.push(A[rng.below(A.len())] as char);
    }
    s
}

/// Draws the augmentation of an item. The number of draws from `rng` does not depend on the
/// variant.
pub fn draw(item: &S5Item, aug: &Augment, rng: &mut Rng) -> Draw {
    let k = item.k();
    let mut order: Vec<usize> = (0..k).collect();
    if aug.permute && item.qtype != SCORE {
        rng.shuffle(&mut order);
    }
    let u = rng.f64();
    let names = match item.rec.t.as_str() {
        "choice" if u < aug.p_names => {
            let mut v: Vec<String> = Vec::with_capacity(k);
            while v.len() < k {
                let n = random_name(rng);
                if !v.contains(&n) {
                    v.push(n);
                }
            }
            Some(v)
        }
        "noul" if u < aug.p_polarity => Some(vec!["true".into(), "false".into()]),
        _ => None,
    };
    Draw { order, names }
}

/// Native option tokens of an item under a draw (record order): the precomputed ones for the
/// record's names, a fresh tokenization for new names.
pub fn native_tokens(
    item: &S5Item,
    d: &Draw,
    tok: &Tok,
    sp: &SpecialTokens,
) -> Result<Vec<Vec<u32>>> {
    let Some(names) = &d.names else {
        return Ok(item.native.clone());
    };
    let generic = generic_noul(&item.rec.lang);
    let texts: Vec<String> = item
        .rec
        .opts_kv
        .iter()
        .enumerate()
        .map(|(i, (_, def))| {
            let def = if def.is_empty() && item.rec.t == "noul" {
                if i == 0 {
                    generic.0
                } else {
                    generic.1
                }
            } else {
                def.as_str()
            };
            format!(" {}: {}", names[i], def).replace(&sp.mask_text, " ")
        })
        .collect();
    Ok(tok
        .encode_batch(texts)?
        .into_iter()
        .map(|mut v| {
            v.truncate(OPTION_MAX_TOKENS);
            v
        })
        .collect())
}

/// One row of a variant-A batch.
#[derive(Debug, Clone)]
pub struct LayaRow {
    pub ids: Vec<u32>,
    pub markers: Vec<usize>,
    pub qtype: usize,
    /// Target in row (display) order.
    pub target: Vec<f32>,
    pub order: Vec<usize>,
}

/// The Laya sequence of an item under a draw.
pub fn laya_row(
    item: &S5Item,
    d: &Draw,
    opts: &[Vec<u32>],
    max_len: usize,
    head_max_len: usize,
    sp: &SpecialTokens,
) -> LayaRow {
    let pieces = Pieces {
        head: item.head.clone(),
        opts: opts.to_vec(),
        state: item.state.clone(),
    };
    let (ids, markers) = build_from_pieces_with(&pieces, &d.order, max_len, head_max_len, sp);
    LayaRow {
        ids,
        markers,
        qtype: item.qtype,
        target: d.order.iter().map(|&i| item.target[i]).collect(),
        order: d.order.clone(),
    }
}

/// Batch arrays of variant A (see [`crate::s5::train`] for the layout of the pair arrays).
pub struct LayaBatch {
    pub ids: Array,
    pub key_valid: Array,
    pub qtype: Array,
    pub marker_pos: Array,
    pub marker_mask: Array,
    pub target: Array,
    pub is_score: Array,
    pub k: Array,
    pub pairs: PairArrays,
    pub n_tokens: usize,
}

/// Pair arrays (at least one, padded with weight 0).
pub struct PairArrays {
    pub left: Array,
    pub right: Array,
    pub map: Array,
    pub mask: Array,
    pub weight: Array,
}

/// Pairs `(row a, row b)` of rows showing the same options in two orders: for each option
/// position `j` of row `a`, the position of the same record option in row `b`.
pub fn pair_arrays(pairs: &[(usize, usize)], orders: &[Vec<usize>], kmax: usize) -> PairArrays {
    let p = pairs.len().max(1);
    let mut left = vec![0i32; p];
    let mut right = vec![0i32; p];
    let mut map = vec![0i32; p * kmax];
    let mut mask = vec![false; p * kmax];
    let mut weight = vec![0f32; p];
    for (n, &(a, b)) in pairs.iter().enumerate() {
        left[n] = a as i32;
        right[n] = b as i32;
        weight[n] = 1.0;
        // position of record option o in row b
        let mut pos_b = vec![0usize; orders[b].len()];
        for (j, &o) in orders[b].iter().enumerate() {
            pos_b[o] = j;
        }
        for (j, &o) in orders[a].iter().enumerate() {
            map[n * kmax + j] = pos_b[o] as i32;
            mask[n * kmax + j] = true;
        }
    }
    PairArrays {
        left: Array::from_slice(&left, &[p as i32]),
        right: Array::from_slice(&right, &[p as i32]),
        map: Array::from_slice(&map, &[p as i32, kmax as i32]),
        mask: Array::from_slice(&mask, &[p as i32, kmax as i32]),
        weight: Array::from_slice(&weight, &[p as i32]),
    }
}

/// Pads Laya rows into a batch (length rounded up to a multiple of 16).
pub fn laya_batch(rows: &[LayaRow], pairs: &[(usize, usize)]) -> LayaBatch {
    let b = rows.len();
    let t = rows
        .iter()
        .map(|r| r.ids.len())
        .max()
        .unwrap_or(1)
        .div_ceil(16)
        * 16;
    let k = rows
        .iter()
        .map(|r| r.markers.len())
        .max()
        .unwrap_or(1)
        .max(1);
    let mut ids = vec![0i32; b * t];
    let mut valid = vec![false; b * t];
    let mut mpos = vec![0i32; b * k];
    let mut mmask = vec![false; b * k];
    let mut target = vec![0f32; b * k];
    let mut n_tokens = 0;
    for (i, r) in rows.iter().enumerate() {
        for (j, &id) in r.ids.iter().enumerate() {
            ids[i * t + j] = id as i32;
            valid[i * t + j] = true;
        }
        n_tokens += r.ids.len();
        for (j, &m) in r.markers.iter().enumerate() {
            mpos[i * k + j] = m as i32;
            mmask[i * k + j] = true;
            target[i * k + j] = r.target[j];
        }
    }
    let orders: Vec<Vec<usize>> = rows.iter().map(|r| r.order.clone()).collect();
    LayaBatch {
        ids: Array::from_slice(&ids, &[b as i32, t as i32]),
        key_valid: Array::from_slice(&valid, &[b as i32, t as i32]),
        qtype: Array::from_iter(rows.iter().map(|r| r.qtype as i32), &[b as i32]),
        marker_pos: Array::from_slice(&mpos, &[b as i32, k as i32]),
        marker_mask: Array::from_slice(&mmask, &[b as i32, k as i32]),
        target: Array::from_slice(&target, &[b as i32, k as i32]),
        is_score: Array::from_iter(
            rows.iter().map(|r| f32::from(u8::from(r.qtype == SCORE))),
            &[b as i32],
        ),
        k: Array::from_iter(rows.iter().map(|r| r.markers.len() as f32), &[b as i32]),
        pairs: pair_arrays(pairs, &orders, k),
        n_tokens,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sequence::tests::byte_tok;

    fn rec(t: &str, kv: &[(&str, &str)], uid: &str) -> S5Record {
        S5Record {
            src: "x".into(),
            t: t.into(),
            ins: "q".into(),
            opts_kv: kv
                .iter()
                .map(|(a, b)| (a.to_string(), b.to_string()))
                .collect(),
            state: "st".into(),
            target: (0..kv.len())
                .map(|i| if i == 1 { 1.0 } else { 0.0 })
                .collect(),
            split: "train".into(),
            uid: uid.into(),
            lang: String::new(),
        }
    }

    #[test]
    fn tokenize_render_and_pairs() {
        let (_d, tok) = byte_tok();
        let sp = SpecialTokens::default();
        let mut items = tokenize(
            &tok,
            &sp,
            vec![
                rec("choice", &[("a", "first"), ("b", "second")], "g1#0"),
                rec("noul", &[("false", "no"), ("true", "yes")], "x"),
                rec("choice", &[("a", "1st"), ("b", "2nd")], "g1#1"),
            ],
        )
        .unwrap();
        link_pairs(&mut items);
        assert_eq!(items[0].pair, Some(2));
        assert_eq!(items[2].pair, Some(0));
        assert_eq!(items[1].pair, None);
        let b = |s: &str| -> Vec<u32> { s.bytes().map(|x| 5 + x as u32).collect() };
        assert_eq!(items[0].defs[1], b(" second"));
        assert_eq!(items[0].native[1], b(" b: second"));
        // New names are rendered for A, never for the definitions.
        let d = Draw {
            order: vec![1, 0],
            names: Some(vec!["xq1".into(), "z9".into()]),
        };
        let n = native_tokens(&items[0], &d, &tok, &sp).unwrap();
        assert_eq!(n[0], b(" xq1: first"));
        let row = laya_row(&items[0], &d, &n, 256, 96, &sp);
        assert_eq!(row.target, vec![1.0, 0.0]);
        assert_eq!(
            &row.ids[row.markers[0] + 1..row.markers[1]],
            &b(" z9: second")[..]
        );
    }

    #[test]
    fn draws_are_variant_independent_and_keep_score_order() {
        let (_d, tok) = byte_tok();
        let sp = SpecialTokens::default();
        let items = tokenize(
            &tok,
            &sp,
            vec![
                rec("score", &[("0", "lo"), ("1", "mid"), ("2", "hi")], "s"),
                rec("choice", &[("a", "x"), ("b", "y"), ("c", "z")], "c"),
            ],
        )
        .unwrap();
        let aug = Augment::default();
        let mut r1 = Rng::new(3);
        for _ in 0..20 {
            assert_eq!(draw(&items[0], &aug, &mut r1).order, vec![0, 1, 2]);
        }
        let mut a = Rng::new(9);
        let mut b = Rng::new(9);
        let da: Vec<Draw> = (0..50).map(|_| draw(&items[1], &aug, &mut a)).collect();
        let db: Vec<Draw> = (0..50).map(|_| draw(&items[1], &aug, &mut b)).collect();
        assert_eq!(da, db);
        let renamed = da.iter().filter(|d| d.names.is_some()).count();
        assert!(renamed > 10 && renamed < 40, "{renamed}");
    }

    #[test]
    fn pair_map_aligns_record_options() {
        let orders = vec![vec![2, 0, 1], vec![0, 1, 2]];
        let p = pair_arrays(&[(0, 1)], &orders, 3);
        p.map.eval().unwrap();
        // Row 0 shows record options [2, 0, 1]; row 1 shows them in record order.
        assert_eq!(p.map.as_slice::<i32>(), &[2, 0, 1]);
        let empty = pair_arrays(&[], &orders, 3);
        empty.weight.eval().unwrap();
        assert_eq!(empty.weight.as_slice::<f32>(), &[0.0]);
    }
}
