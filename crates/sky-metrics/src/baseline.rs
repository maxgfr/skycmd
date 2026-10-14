//! Reference predictors scored with the same metrics as a model: the uniform distribution
//! and the majority-class prior of the item's group.

use std::collections::BTreeMap;

use crate::Pred;

/// `preds` with every distribution replaced by the uniform one (p = 1/K); target, gold and
/// group are kept. Its accuracy follows the first-on-ties convention: option 0 is always the
/// prediction.
pub fn uniform<'a>(preds: impl IntoIterator<Item = &'a Pred>) -> Vec<Pred> {
    preds
        .into_iter()
        .map(|p| Pred {
            probs: vec![1.0 / p.k() as f64; p.k()],
            ..p.clone()
        })
        .collect()
}

/// Majority-class baseline: per group, the frequencies of the gold options of a training set.
/// Its prediction is the most frequent training option of the group.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Majority {
    counts: BTreeMap<String, Vec<usize>>,
}

impl Majority {
    /// Counts the training labels, `(group, gold option)` pairs.
    pub fn fit<'a>(labels: impl IntoIterator<Item = (&'a str, usize)>) -> Self {
        let mut counts: BTreeMap<String, Vec<usize>> = BTreeMap::new();
        for (group, gold) in labels {
            let c = counts.entry(group.to_string()).or_default();
            if c.len() <= gold {
                c.resize(gold + 1, 0);
            }
            c[gold] += 1;
        }
        Self { counts }
    }

    /// Distribution of an item of `group` with `k` options: the training frequencies of options
    /// `0..k` with add-one (Laplace) smoothing, `(count_i + 1) / (Σ count + k)`, so an option
    /// never seen keeps a finite NLL. Training labels ≥ `k` are ignored; an unseen group gets the
    /// uniform distribution.
    pub fn probs(&self, group: &str, k: usize) -> Vec<f64> {
        let count = |i: usize| {
            self.counts
                .get(group)
                .and_then(|c| c.get(i))
                .copied()
                .unwrap_or(0) as f64
        };
        let total: f64 = (0..k).map(count).sum::<f64>() + k as f64;
        (0..k).map(|i| (count(i) + 1.0) / total).collect()
    }

    /// `preds` with every distribution replaced by its group's prior; target, gold and group
    /// are kept.
    pub fn apply<'a>(&self, preds: impl IntoIterator<Item = &'a Pred>) -> Vec<Pred> {
        preds
            .into_iter()
            .map(|p| Pred {
                probs: self.probs(&p.group, p.k()),
                ..p.clone()
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scores::{accuracy, brier, nll};

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-12
    }

    fn item(k: usize, gold: usize, group: &str) -> Pred {
        let mut t = vec![0.0; k];
        t[gold] = 1.0;
        Pred::new(vec![1.0 / k as f64; k], t, group)
    }

    #[test]
    fn uniform_scores_by_hand() {
        let ps = [item(4, 2, "a"), item(4, 0, "a")];
        let u = uniform(&ps);
        assert_eq!(u[0].probs, [0.25; 4]);
        assert_eq!((u[0].target.clone(), u[0].gold), (ps[0].target.clone(), 2));
        assert!(close(nll(&u), 4f64.ln()));
        // (1 − 1/4)² + 3 · (1/4)² = 1 − 1/4
        assert!(close(brier(&u), 0.75));
        assert_eq!(accuracy(&u), 0.5);
    }

    #[test]
    fn majority_by_hand() {
        let m = Majority::fit([("a", 0), ("a", 0), ("a", 1), ("b", 1), ("a", 5)]);
        // Group a, 3 options: counts (2, 1, 0) of 3 kept labels, smoothed over 3 + 3.
        let p = m.probs("a", 3);
        assert!(close(p[0], 0.5) && close(p[1], 2.0 / 6.0) && close(p[2], 1.0 / 6.0));
        assert_eq!(m.probs("b", 2), [1.0 / 3.0, 2.0 / 3.0]);
        assert_eq!(m.probs("unseen", 4), [0.25; 4]);
        let ps = [
            item(3, 0, "a"),
            item(3, 2, "a"),
            item(2, 1, "b"),
            item(2, 0, "c"),
        ];
        let base = m.apply(&ps);
        // Predicts 0 for a, 1 for b, 0 (uniform, first on ties) for c: right on 0, 2 and 3.
        assert_eq!(accuracy(&base), 0.75);
        let want = -(0.5f64.ln() + (1.0f64 / 6.0).ln() + (2.0f64 / 3.0).ln() + 0.5f64.ln()) / 4.0;
        assert!(close(nll(&base), want));
        assert_eq!(base[1].gold, 2);
    }
}
