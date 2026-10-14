//! Decontamination of the s5 data: word 13-grams and near-duplicates (MinHash over word
//! 5-grams, estimated Jaccard ≥ 0.8), on the state and the instructions.
//!
//! The 13-gram rule is a **coverage** rule against one reference: a text overlaps when at
//! least half of its 13-grams occur in a single reference text. The "any shared 13-gram" rule
//! of GPT-3 flags every templated state (typed-decisions builds its JSON states from shared
//! pools of tasks and constraints, so different cases share long runs), and so does coverage
//! by the union of all references; half of the text found in one reference means copied
//! content, not a shared schema.
//!
//! Only the **states** of the evaluation sets and demo cases are indexed (the inputs being
//! classified; their questions and definitions are ours and shared by design); a training
//! record is checked on its state and on its instructions.
//!
//! - training records that share a 13-gram with, or nearly duplicate, an evaluation item or a
//!   demo case are dropped;
//! - evaluation items that overlap the declared training sources of the teachers that we
//!   have locally are counted and reported (the other declared sources and the undeclared
//!   ones are listed as a limitation: they cannot be checked).

use std::collections::{BTreeMap, HashMap, HashSet};

use crate::minhash::{shingles, MinHasher, BANDS, ROWS};
use crate::rng::{hash64, mix64};
use crate::s5::S5Record;

/// Word n-gram length of the overlap rule.
pub const NGRAM: usize = 13;
/// Share of a text's n-grams that must occur in the references.
pub const COVERAGE: f64 = 0.5;
/// Near-duplicate threshold (estimated Jaccard of word 5-grams).
pub const NEAR: f64 = 0.8;

fn words(text: &str) -> Vec<String> {
    text.split_whitespace()
        .map(|w| {
            w.chars()
                .filter(|c| c.is_alphanumeric())
                .flat_map(char::to_lowercase)
                .collect::<String>()
        })
        .filter(|w| !w.is_empty())
        .collect()
}

/// Hashes of the word `n`-grams of a text; a text shorter than `n` words is one gram (itself).
pub fn ngrams(text: &str, n: usize) -> Vec<u64> {
    let w = words(text);
    if w.is_empty() {
        return Vec::new();
    }
    if w.len() < n {
        return vec![hash64(w.join(" ").as_bytes())];
    }
    w.windows(n)
        .map(|g| hash64(g.join(" ").as_bytes()))
        .collect()
}

/// Why a text overlaps the index.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Hit {
    NGram,
    Near,
}

/// An index of reference texts.
pub struct Index {
    /// n-gram -> ids of the reference texts containing it.
    grams: HashMap<u64, Vec<u32>>,
    hasher: MinHasher,
    sigs: Vec<Vec<u32>>,
    bands: Vec<HashMap<u64, Vec<u32>>>,
    pub texts: usize,
}

fn band_keys(sig: &[u32]) -> Vec<u64> {
    (0..BANDS)
        .map(|b| {
            let mut h = b as u64;
            for &v in &sig[b * ROWS..(b + 1) * ROWS] {
                h = mix64(h ^ v as u64);
            }
            h
        })
        .collect()
}

impl Default for Index {
    fn default() -> Self {
        Self::new()
    }
}

impl Index {
    pub fn new() -> Self {
        Self {
            grams: HashMap::new(),
            hasher: MinHasher::new(0x5EED_0FF5),
            sigs: Vec::new(),
            bands: (0..BANDS).map(|_| HashMap::new()).collect(),
            texts: 0,
        }
    }

    pub fn add(&mut self, text: &str) {
        let mut g = ngrams(text, NGRAM);
        if g.is_empty() {
            return;
        }
        g.sort_unstable();
        g.dedup();
        let id = self.texts as u32;
        for x in g {
            self.grams.entry(x).or_default().push(id);
        }
        let sh = shingles(text);
        if !sh.is_empty() {
            let sig = self.hasher.signature(&sh);
            let id = self.sigs.len() as u32;
            debug_assert!(id as usize <= self.texts);
            for (b, k) in band_keys(&sig).into_iter().enumerate() {
                self.bands[b].entry(k).or_default().push(id);
            }
            self.sigs.push(sig);
        }
        self.texts += 1;
    }

    /// The largest share of the n-grams of `text` found in a single reference text.
    pub fn coverage(&self, text: &str) -> f64 {
        let mut g = ngrams(text, NGRAM);
        g.sort_unstable();
        g.dedup();
        if g.is_empty() {
            return 0.0;
        }
        let mut per: HashMap<u32, usize> = HashMap::new();
        for x in &g {
            for &id in self.grams.get(x).map(Vec::as_slice).unwrap_or(&[]) {
                *per.entry(id).or_insert(0) += 1;
            }
        }
        per.values().copied().max().unwrap_or(0) as f64 / g.len() as f64
    }

    pub fn hit(&self, text: &str) -> Option<Hit> {
        if self.coverage(text) >= COVERAGE {
            return Some(Hit::NGram);
        }
        let sh = shingles(text);
        if sh.is_empty() {
            return None;
        }
        let sig = self.hasher.signature(&sh);
        let mut seen = HashSet::new();
        for (b, k) in band_keys(&sig).iter().enumerate() {
            for &c in self.bands[b].get(k).map(Vec::as_slice).unwrap_or(&[]) {
                if seen.insert(c)
                    && crate::minhash::estimated_jaccard(&sig, &self.sigs[c as usize]) >= NEAR
                {
                    return Some(Hit::Near);
                }
            }
        }
        None
    }
}

/// The texts of a record that are checked: its state, and its instructions when they are long
/// enough to carry content (templated instructions are shared by design).
pub fn texts(r: &S5Record) -> Vec<&str> {
    let mut v = vec![r.state.as_str()];
    if r.ins.split_whitespace().count() >= NGRAM {
        v.push(r.ins.as_str());
    }
    v
}

/// Drops the records that overlap `refs`; returns the kept records and the drop count per
/// `(source, hit kind)`.
pub fn decontaminate(
    recs: Vec<S5Record>,
    refs: &Index,
) -> (Vec<S5Record>, BTreeMap<String, usize>) {
    let mut dropped = BTreeMap::new();
    let kept = recs
        .into_iter()
        .filter(|r| match texts(r).into_iter().find_map(|t| refs.hit(t)) {
            Some(h) => {
                *dropped.entry(format!("{} ({h:?})", r.src)).or_insert(0) += 1;
                false
            }
            None => true,
        })
        .collect();
    (kept, dropped)
}

/// Share of the records of each source that overlap `refs`.
pub fn overlap_rates(recs: &[S5Record], refs: &Index) -> BTreeMap<String, (usize, usize)> {
    let mut out: BTreeMap<String, (usize, usize)> = BTreeMap::new();
    for r in recs {
        let e = out.entry(r.src.clone()).or_default();
        e.1 += 1;
        if texts(r).into_iter().any(|t| refs.hit(t).is_some()) {
            e.0 += 1;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ngram_and_near_duplicates() {
        let mut idx = Index::new();
        let eval = "the quick brown fox jumps over the lazy dog near the river bank at dawn while birds sing";
        idx.add(eval);
        // Copies most of the text.
        let copy =
            "Breaking: THE quick brown fox jumps over the lazy dog near the river bank at dawn!";
        assert_eq!(idx.hit(copy), Some(Hit::NGram));
        // Shares one 13-gram (a template) inside a much longer, different text.
        let templated = format!(
            "{} the quick brown fox jumps over the lazy dog near the river bank at dawn",
            "an invoice of 230 euros was paid twice by the customer who now asks for a refund of the second payment made on monday morning through the bank"
        );
        assert_eq!(idx.hit(&templated), None);
        assert_eq!(
            idx.hit("a completely different sentence about invoices and payments"),
            None
        );
        // Short texts match only exactly (case and punctuation aside).
        let mut short = Index::new();
        short.add("I love this product!");
        assert_eq!(short.hit("i love this product"), Some(Hit::NGram));
        assert_eq!(short.hit("I love this product a lot"), None);
    }
}
