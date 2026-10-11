//! Exact and near-duplicate detection: MinHash signatures over word 5-gram
//! shingles with banded LSH, confirmed by the signature-estimated Jaccard.

use std::collections::{HashMap, HashSet};

use crate::rng::{hash64, mix64, Rng};

pub const SHINGLE: usize = 5;
pub const NUM_PERM: usize = 128;
pub const BANDS: usize = 16;
pub const ROWS: usize = NUM_PERM / BANDS;

/// Normalizes text for exact-duplicate hashing (case and whitespace insensitive).
pub fn normalize_text(s: &str) -> String {
    s.split_whitespace().map(str::to_lowercase).collect::<Vec<_>>().join(" ")
}

/// Hashes of the word 5-gram shingles (a short text is a single shingle).
pub fn shingles(text: &str) -> Vec<u64> {
    let words: Vec<String> = text
        .split_whitespace()
        .map(|w| w.chars().filter(|c| c.is_alphanumeric()).flat_map(char::to_lowercase).collect::<String>())
        .filter(|w| !w.is_empty())
        .collect();
    if words.is_empty() {
        return Vec::new();
    }
    if words.len() < SHINGLE {
        return vec![hash64(words.join(" ").as_bytes())];
    }
    let mut set: HashSet<u64> = HashSet::new();
    for w in words.windows(SHINGLE) {
        set.insert(hash64(w.join(" ").as_bytes()));
    }
    set.into_iter().collect()
}

/// A family of `NUM_PERM` hash functions `h_i(x) = mix64(x ^ seed_i)`.
pub struct MinHasher {
    seeds: Vec<u64>,
}

impl MinHasher {
    pub fn new(seed: u64) -> Self {
        let mut rng = Rng::new(seed);
        Self {
            seeds: (0..NUM_PERM).map(|_| rng.next_u64()).collect(),
        }
    }

    pub fn signature(&self, shingles: &[u64]) -> Vec<u32> {
        let mut sig = vec![u32::MAX; NUM_PERM];
        for &s in shingles {
            for (slot, &seed) in sig.iter_mut().zip(&self.seeds) {
                let h = (mix64(s ^ seed) >> 32) as u32;
                if h < *slot {
                    *slot = h;
                }
            }
        }
        sig
    }
}

/// Fraction of equal signature slots, an unbiased estimate of the Jaccard similarity.
pub fn estimated_jaccard(a: &[u32], b: &[u32]) -> f64 {
    let eq = a.iter().zip(b).filter(|(x, y)| x == y).count();
    eq as f64 / a.len().max(1) as f64
}

/// Streaming deduplicator: `insert` returns `false` for exact or near duplicates
/// of anything inserted before.
pub struct Dedup {
    hasher: MinHasher,
    threshold: f64,
    exact: HashSet<u64>,
    sigs: Vec<Vec<u32>>,
    bands: Vec<HashMap<u64, Vec<u32>>>,
    pub exact_dups: usize,
    pub near_dups: usize,
}

impl Dedup {
    pub fn new(threshold: f64, seed: u64) -> Self {
        Self {
            hasher: MinHasher::new(seed),
            threshold,
            exact: HashSet::new(),
            sigs: Vec::new(),
            bands: (0..BANDS).map(|_| HashMap::new()).collect(),
            exact_dups: 0,
            near_dups: 0,
        }
    }

    pub fn len(&self) -> usize {
        self.sigs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.sigs.is_empty()
    }

    pub fn insert(&mut self, text: &str) -> bool {
        if !self.exact.insert(hash64(normalize_text(text).as_bytes())) {
            self.exact_dups += 1;
            return false;
        }
        let sh = shingles(text);
        if sh.is_empty() {
            return false;
        }
        let sig = self.hasher.signature(&sh);
        let keys: Vec<u64> = (0..BANDS)
            .map(|b| {
                let mut h = b as u64;
                for &v in &sig[b * ROWS..(b + 1) * ROWS] {
                    h = mix64(h ^ v as u64);
                }
                h
            })
            .collect();
        let mut checked = HashSet::new();
        for (b, k) in keys.iter().enumerate() {
            if let Some(cands) = self.bands[b].get(k) {
                for &c in cands {
                    if checked.insert(c) && estimated_jaccard(&sig, &self.sigs[c as usize]) >= self.threshold {
                        self.near_dups += 1;
                        return false;
                    }
                }
            }
        }
        let id = self.sigs.len() as u32;
        for (b, k) in keys.into_iter().enumerate() {
            self.bands[b].entry(k).or_default().push(id);
        }
        self.sigs.push(sig);
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const A: &str = "Once upon a time there was a little girl named Lily who loved to play in the park with her red ball every single day after school.";

    #[test]
    fn exact_and_near_duplicates() {
        let mut d = Dedup::new(0.8, 1);
        assert!(d.insert(A));
        assert!(!d.insert(&A.to_uppercase()), "case-only change is an exact dup");
        let near = A.replace("every single day", "every single  day!");
        assert!(!d.insert(&near));
        let long: Vec<String> = (0..80).map(|i| format!("word{i}")).collect();
        assert!(d.insert(&long.join(" ")));
        let mut edited = long.clone();
        edited[79] = "changed".into();
        assert!(!d.insert(&edited.join(" ")), "one changed word in 80 should be a near dup");
        edited[20..40].iter_mut().for_each(|w| *w = "x".into());
        assert!(d.insert(&edited.join(" ")), "a quarter rewritten is not a near dup");
        assert!(d.insert("A completely different story about a dragon who learned to bake bread in a small village by the sea."));
        assert_eq!(d.len(), 4);
        assert_eq!(d.exact_dups, 1);
        assert!(d.near_dups >= 1);
    }

    #[test]
    fn jaccard_estimate_is_close() {
        let h = MinHasher::new(9);
        let a: Vec<u64> = (0..200).collect();
        let b: Vec<u64> = (50..250).collect(); // true Jaccard = 150/250 = 0.6
        let j = estimated_jaccard(&h.signature(&a), &h.signature(&b));
        assert!((j - 0.6).abs() < 0.12, "{j}");
    }

    #[test]
    fn short_texts_shingle() {
        assert_eq!(shingles("hi there").len(), 1);
        assert!(shingles("   ").is_empty());
    }
}
