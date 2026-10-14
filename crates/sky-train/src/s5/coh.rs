//! Coherence across questions (bet 2, 2609.33209). A training batch may hold whole families
//! of one typed-decisions question: `CH` (the choice), `S_k` ("the correct answer is x_k") and
//! `N_k` (the reformulated negation "the correct answer is not x_k"). With `z` the log-odds of
//! "true" of a noul row and `s = sigmoid(z)`:
//!
//! - `L_neg = (z_S + z_N)²` over the `(S_k, N_k)` pairs;
//! - `L_part = (Σ_k s_k − 1)²` over the families;
//! - `L_xf = (s_k − sg(c_k))²`, `c_k` the choice probability of option k (no gradient).
//!
//! No disjunction (`D_k`), monotonicity or description loss is ever trained: those identities
//! are measured on the test families only.

use mlx_rs::{error::Exception, ops, stop_gradient, Array};

type R<T> = std::result::Result<T, Exception>;

/// Family kind and option index of a coherence record (`coh/<group>/<kind>/<k>`).
pub fn parse(uid: &str) -> Option<(&str, char, usize)> {
    let rest = uid.strip_prefix("coh/")?;
    let mut it = rest.splitn(3, '/');
    let g = it.next()?;
    let kind = it.next()?.chars().next()?;
    let k = it.next()?.parse().ok()?;
    Some((g, kind, k))
}

/// The coherence structure of a batch (row indices).
#[derive(Debug, Clone, Default)]
pub struct CohSpec {
    /// Display position of the "true" option of each row (1 for non-noul rows).
    pub tpos: Vec<i32>,
    /// `(S_k row, N_k row)`.
    pub neg: Vec<(usize, usize)>,
    /// The `S_k` rows of each family.
    pub part: Vec<Vec<usize>>,
    /// `(S_k row, CH row, display position of option k in the CH row)`.
    pub xf: Vec<(usize, usize, usize)>,
}

/// Batch arrays of a [`CohSpec`] (each list padded to at least one entry of weight 0).
pub fn arrays(c: &CohSpec, rows: usize) -> Vec<Array> {
    let mut tpos = c.tpos.clone();
    tpos.resize(rows, 1);
    let p = c.neg.len().max(1);
    let (mut na, mut nb, mut nw) = (vec![0i32; p], vec![0i32; p], vec![0f32; p]);
    for (i, &(a, b)) in c.neg.iter().enumerate() {
        na[i] = a as i32;
        nb[i] = b as i32;
        nw[i] = 1.0;
    }
    let f = c.part.len().max(1);
    let kf = c.part.iter().map(Vec::len).max().unwrap_or(1).max(1);
    let (mut pr, mut pm, mut pw) = (vec![0i32; f * kf], vec![0f32; f * kf], vec![0f32; f]);
    for (i, fam) in c.part.iter().enumerate() {
        pw[i] = 1.0;
        for (j, &r) in fam.iter().enumerate() {
            pr[i * kf + j] = r as i32;
            pm[i * kf + j] = 1.0;
        }
    }
    let q = c.xf.len().max(1);
    let (mut xs, mut xc, mut xp, mut xw) =
        (vec![0i32; q], vec![0i32; q], vec![0i32; q], vec![0f32; q]);
    for (i, &(s, ch, pos)) in c.xf.iter().enumerate() {
        xs[i] = s as i32;
        xc[i] = ch as i32;
        xp[i] = pos as i32;
        xw[i] = 1.0;
    }
    let (p, f, kf, q) = (p as i32, f as i32, kf as i32, q as i32);
    vec![
        Array::from_slice(&tpos, &[rows as i32]),
        Array::from_slice(&na, &[p]),
        Array::from_slice(&nb, &[p]),
        Array::from_slice(&nw, &[p]),
        Array::from_slice(&pr, &[f, kf]),
        Array::from_slice(&pm, &[f, kf]),
        Array::from_slice(&pw, &[f]),
        Array::from_slice(&xs, &[q]),
        Array::from_slice(&xc, &[q]),
        Array::from_slice(&xp, &[q, 1]),
        Array::from_slice(&xw, &[q]),
    ]
}

/// Number of arrays of [`arrays`].
pub const N_ARRAYS: usize = 11;

fn wmean(v: &Array, w: &Array) -> R<Array> {
    v.multiply(w)?
        .sum(None)?
        .divide(ops::maximum(w.sum(None)?, Array::from_f32(1.0))?)
}

/// `L_neg + L_part + L_xf` of masked logits `[B, K]` and the arrays of [`arrays`].
pub fn loss(logits: &Array, a: &[Array]) -> R<Array> {
    let b = logits.dim(0);
    let tpos = a[0].reshape(&[b, 1])?;
    let other = Array::from_int(1).subtract(&tpos)?;
    let z = logits
        .take_along_axis(&tpos, -1)?
        .subtract(logits.take_along_axis(&other, -1)?)?
        .reshape(&[b])?;
    let s = ops::sigmoid(&z)?;
    let neg = z
        .take_axis(&a[1], 0)?
        .add(z.take_axis(&a[2], 0)?)?
        .square()?;
    let l_neg = wmean(&neg, &a[3])?;
    let sp = s.take_axis(&a[4], 0)?.multiply(&a[5])?.sum_axis(-1, None)?;
    let l_part = wmean(&sp.subtract(Array::from_f32(1.0))?.square()?, &a[6])?;
    let ch = logits.take_axis(&a[8], 0)?;
    let pch = ops::softmax_axis(&ch, -1, None)?
        .take_along_axis(&a[9], -1)?
        .reshape(&[a[9].dim(0)])?;
    let xf = s
        .take_axis(&a[7], 0)?
        .subtract(stop_gradient(&pch)?)?
        .square()?;
    let l_xf = wmean(&xf, &a[10])?;
    l_neg.add(l_part)?.add(l_xf)
}

/// `(S_k p_true, N_k p_true, CH probs)` of one family.
type FamilyProbs = (
    std::collections::BTreeMap<usize, f64>,
    std::collections::BTreeMap<usize, f64>,
    Vec<f64>,
);

/// Coherence metrics of predicted probabilities (record order) of coherence items:
/// `(mad_neg, mean Σ s_k, mean |Σ s_k − 1|, mean |s_k − c_k|, n pairs)`.
pub fn metrics(items: &[(&str, Vec<f64>)]) -> (f64, f64, f64, f64, usize) {
    use std::collections::BTreeMap;
    let mut fam: BTreeMap<&str, FamilyProbs> = BTreeMap::new();
    for (uid, p) in items {
        let Some((g, kind, k)) = parse(uid) else {
            continue;
        };
        let e = fam.entry(g).or_default();
        match kind {
            'S' => {
                e.0.insert(k, p[1]);
            }
            'N' => {
                e.1.insert(k, p[1]);
            }
            'C' => e.2 = p.clone(),
            _ => {}
        }
    }
    let (mut mad, mut np) = (0.0, 0usize);
    let (mut part, mut part_dev, mut nf) = (0.0, 0.0, 0usize);
    let (mut xf, mut nx) = (0.0, 0usize);
    for (s, n, ch) in fam.values() {
        for (k, sv) in s {
            if let Some(nv) = n.get(k) {
                mad += (sv + nv - 1.0).abs();
                np += 1;
            }
            if let Some(c) = ch.get(*k) {
                xf += (sv - c).abs();
                nx += 1;
            }
        }
        if !s.is_empty() && (ch.is_empty() || s.len() == ch.len()) {
            let t: f64 = s.values().sum();
            part += t;
            part_dev += (t - 1.0).abs();
            nf += 1;
        }
    }
    let d = |a: f64, n: usize| if n == 0 { f64::NAN } else { a / n as f64 };
    (d(mad, np), d(part, nf), d(part_dev, nf), d(xf, nx), np)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{model::host_f32, test_util};

    #[test]
    fn coherent_logits_cost_nothing() {
        let _g = test_util::lock();
        // Rows: 0 = CH [.7, .2, .1], 1..3 = S_k with p_true = c_k, 4..6 = N_k with 1 - c_k.
        let c = [0.7f32, 0.2, 0.1];
        let logit = |p: f32| (p / (1.0 - p)).ln();
        let mut l = vec![c[0].ln(), c[1].ln(), c[2].ln()];
        for &p in &c {
            l.extend([0.0, logit(p), -1e4]);
        }
        for &p in &c {
            l.extend([0.0, logit(1.0 - p), -1e4]);
        }
        let logits = Array::from_slice(&l, &[7, 3]);
        let spec = CohSpec {
            tpos: vec![1; 7],
            neg: vec![(1, 4), (2, 5), (3, 6)],
            part: vec![vec![1, 2, 3]],
            xf: vec![(1, 0, 0), (2, 0, 1), (3, 0, 2)],
        };
        let v = host_f32(&loss(&logits, &arrays(&spec, 7)).unwrap()).unwrap()[0];
        assert!(v.abs() < 1e-5, "{v}");
        // An incoherent negation costs (z_S + z_N)^2.
        let mut bad = l.clone();
        bad[4 * 3 + 1] = logit(c[0]); // N_0 says the same as S_0
        let logits = Array::from_slice(&bad, &[7, 3]);
        let spec2 = CohSpec {
            neg: vec![(1, 4)],
            part: vec![],
            xf: vec![],
            ..spec
        };
        let v = host_f32(&loss(&logits, &arrays(&spec2, 7)).unwrap()).unwrap()[0];
        let want = (2.0 * logit(c[0])).powi(2);
        assert!((v - want).abs() < 1e-3, "{v} vs {want}");
        // A row whose "true" option is displayed first.
        let flipped = Array::from_slice(&[logit(0.8), 0.0], &[1, 2]);
        let s = CohSpec {
            tpos: vec![0],
            part: vec![vec![0]],
            ..Default::default()
        };
        let v = host_f32(&loss(&flipped, &arrays(&s, 1)).unwrap()).unwrap()[0];
        assert!((v - 0.04).abs() < 1e-4, "{v}");
    }

    #[test]
    fn metric_helper() {
        let items = vec![
            ("coh/g/CH/0", vec![0.6, 0.4]),
            ("coh/g/S/0", vec![0.4, 0.6]),
            ("coh/g/S/1", vec![0.6, 0.4]),
            ("coh/g/N/0", vec![0.5, 0.5]),
            ("coh/g/N/1", vec![0.4, 0.6]),
        ];
        let (mad, part, dev, xf, n) = metrics(&items);
        assert_eq!(n, 2);
        assert!((mad - (0.1 + 0.0) / 2.0).abs() < 1e-12);
        assert!((part - 1.0).abs() < 1e-12 && dev.abs() < 1e-12);
        assert!(xf.abs() < 1e-12);
        assert_eq!(parse("coh/abc/N/3"), Some(("abc", 'N', 3)));
    }
}
