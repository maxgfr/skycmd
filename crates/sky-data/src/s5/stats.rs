//! The `s5-stats` gate: no class above 60 %. For every source and question type, the share of
//! questions whose gold option is a given option id (choice), level (score) or answer (noul)
//! must stay at or below 60 %; [`rebalance`] downsamples the dominant class of a source until
//! it does.

use std::collections::BTreeMap;

use crate::rng::Rng;
use crate::s5::S5Record;

/// The largest class share allowed.
pub const MAX_SHARE: f64 = 0.6;

/// Class key of a record: its gold option id (choice / noul) or level (score). Uniform
/// targets (missing evidence) have no class.
pub fn class_of(r: &S5Record) -> Option<String> {
    let m = r.target.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    if r.target.iter().all(|&x| (x - m).abs() < 1e-6) {
        return None;
    }
    Some(r.opts_kv[r.gold_index()].0.clone())
}

/// `(source, type) -> (class -> count)`.
pub fn distribution(recs: &[S5Record]) -> BTreeMap<(String, String), BTreeMap<String, usize>> {
    let mut out: BTreeMap<(String, String), BTreeMap<String, usize>> = BTreeMap::new();
    for r in recs {
        if let Some(c) = class_of(r) {
            *out.entry((r.src.clone(), r.t.clone()))
                .or_default()
                .entry(c)
                .or_insert(0) += 1;
        }
    }
    out
}

/// Groups whose largest class exceeds [`MAX_SHARE`] (with at least `min_n` questions; smaller
/// groups are reported but not gated): `(source, type, class, share, n)`.
pub fn violations(recs: &[S5Record], min_n: usize) -> Vec<(String, String, String, f64, usize)> {
    let mut out = Vec::new();
    for ((src, t), classes) in distribution(recs) {
        let n: usize = classes.values().sum();
        if n < min_n || classes.len() < 2 {
            continue;
        }
        let (c, &top) = classes.iter().max_by_key(|(_, &v)| v).unwrap();
        let share = top as f64 / n as f64;
        if share > MAX_SHARE {
            out.push((src, t, c.clone(), share, n));
        }
    }
    out
}

/// Downsamples the records of the dominant class of every violating `(source, type)` group to
/// exactly the count that brings it to [`MAX_SHARE`] (a seeded choice of which ones stay).
pub fn rebalance(recs: Vec<S5Record>, min_n: usize, seed: u64) -> Vec<S5Record> {
    let dist = distribution(&recs);
    let mut drop = vec![false; recs.len()];
    let mut rng = Rng::derived(seed, "s5/stats/rebalance");
    for (s, t, c, _, n) in violations(&recs, min_n) {
        let top = dist[&(s.clone(), t.clone())][&c];
        // Keep k of the top class: k / (n - top + k) <= MAX_SHARE.
        let rest = (n - top) as f64;
        let keep = ((MAX_SHARE * rest) / (1.0 - MAX_SHARE)).floor() as usize;
        let mut idx: Vec<usize> = (0..recs.len())
            .filter(|&i| {
                recs[i].src == s && recs[i].t == t && class_of(&recs[i]).as_deref() == Some(&c)
            })
            .collect();
        rng.shuffle(&mut idx);
        for &i in idx.iter().skip(keep) {
            drop[i] = true;
        }
    }
    recs.into_iter()
        .zip(drop)
        .filter_map(|(r, d)| (!d).then_some(r))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rec(src: &str, gold: usize) -> S5Record {
        S5Record {
            src: src.into(),
            t: "noul".into(),
            ins: "q".into(),
            opts_kv: vec![
                ("false".into(), String::new()),
                ("true".into(), String::new()),
            ],
            state: "s".into(),
            target: if gold == 1 {
                vec![0.0, 1.0]
            } else {
                vec![1.0, 0.0]
            },
            split: "train".into(),
            uid: String::new(),
            lang: String::new(),
            gold: true,
        }
    }

    #[test]
    fn gate_and_rebalance() {
        let mut v: Vec<S5Record> = (0..900).map(|_| rec("a", 1)).collect();
        v.extend((0..100).map(|_| rec("a", 0)));
        v.extend((0..50).map(|_| rec("b", 1)));
        let bad = violations(&v, 100);
        assert_eq!(bad.len(), 1);
        assert_eq!(bad[0].2, "true");
        let r = rebalance(v, 100, 1);
        let left = violations(&r, 100);
        assert!(left.is_empty(), "{left:?}");
        let a_true = r
            .iter()
            .filter(|x| x.src == "a" && x.target[1] == 1.0)
            .count();
        assert_eq!(a_true, 150);
        assert_eq!(
            class_of(&S5Record {
                target: vec![0.5, 0.5],
                ..rec("c", 0)
            }),
            None
        );
    }
}
