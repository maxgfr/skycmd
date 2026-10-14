//! Selective prediction: how good the most confident answers are.
//!
//! Predictions are ranked by confidence (largest probability), most confident first. Equal
//! confidences are resolved in expectation, as if their order were drawn uniformly at random,
//! so no result depends on the order of the items.

use crate::{mean, Pred};

/// Confidence from which a wrong answer is a confident error.
pub const CONFIDENT: f64 = 0.9;

/// Error rate of [`Selective::coverage_at_5pct_error`].
pub const MAX_RISK: f64 = 0.05;

/// Selective-prediction summary of a set of predictions (NaN fields on an empty set).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Selective {
    pub n: usize,
    /// Accuracy among the 10% most confident predictions ([`RiskCoverage::accuracy_at`]).
    pub acc_top10: f64,
    /// Accuracy among the 20% most confident predictions.
    pub acc_top20: f64,
    /// Accuracy among the 50% most confident predictions.
    pub acc_top50: f64,
    /// Wrong predictions with a confidence ≥ [`CONFIDENT`].
    pub confident_errors: usize,
    /// `confident_errors / n`: share of all predictions.
    pub confident_error_rate: f64,
    /// Largest share of the most confident predictions whose error rate is ≤ [`MAX_RISK`]
    /// ([`RiskCoverage::coverage_at`]), 0 when even the most confident one is too risky.
    pub coverage_at_5pct_error: f64,
    /// Area under the risk–coverage curve ([`RiskCoverage::aurc`]).
    pub aurc: f64,
}

/// [`Selective`] summary of `preds`.
pub fn selective<'a>(preds: impl IntoIterator<Item = &'a Pred>) -> Selective {
    let pairs: Vec<(f64, f64)> = preds
        .into_iter()
        .map(|p| (p.confidence(), p.correct()))
        .collect();
    let confident_errors = pairs
        .iter()
        .filter(|&&(c, hit)| c >= CONFIDENT && hit == 0.0)
        .count();
    let rc = RiskCoverage::from_pairs(pairs);
    let n = rc.len();
    Selective {
        n,
        acc_top10: rc.accuracy_at(0.1),
        acc_top20: rc.accuracy_at(0.2),
        acc_top50: rc.accuracy_at(0.5),
        confident_errors,
        confident_error_rate: if n == 0 {
            f64::NAN
        } else {
            confident_errors as f64 / n as f64
        },
        coverage_at_5pct_error: rc.coverage_at(MAX_RISK),
        aurc: rc.aurc(),
    }
}

/// Risk–coverage curve: expected number of errors among the `k` most confident predictions,
/// for `k = 0..=n`.
#[derive(Debug, Clone, PartialEq)]
pub struct RiskCoverage {
    errors: Vec<f64>,
}

impl RiskCoverage {
    pub fn new<'a>(preds: impl IntoIterator<Item = &'a Pred>) -> Self {
        Self::from_pairs(preds.into_iter().map(|p| (p.confidence(), p.correct())))
    }

    /// From `(confidence, correct)` pairs, `correct` in [0, 1].
    pub fn from_pairs(pairs: impl IntoIterator<Item = (f64, f64)>) -> Self {
        let mut v: Vec<(f64, f64)> = pairs.into_iter().collect();
        v.sort_by(|a, b| b.0.total_cmp(&a.0));
        let mut errors = Vec::with_capacity(v.len() + 1);
        errors.push(0.0);
        let mut i = 0;
        while i < v.len() {
            // A run of equal confidences shares its errors evenly between its positions.
            let j = i + 1 + v[i + 1..].iter().take_while(|x| x.0 == v[i].0).count();
            let wrong: f64 = v[i..j].iter().map(|x| 1.0 - x.1).sum();
            let (base, len) = (errors[i], (j - i) as f64);
            errors.extend((i + 1..j).map(|t| base + wrong * (t - i) as f64 / len));
            errors.push(base + wrong);
            i = j;
        }
        Self { errors }
    }

    /// Number of predictions n.
    pub fn len(&self) -> usize {
        self.errors.len() - 1
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Expected number of errors among the `k` most confident predictions.
    pub fn errors(&self, k: usize) -> f64 {
        self.errors[k]
    }

    /// Error rate among the `k ≥ 1` most confident predictions.
    pub fn risk(&self, k: usize) -> f64 {
        self.errors[k] / k as f64
    }

    /// Accuracy among the ⌈q·n⌉ most confident predictions (at least one); NaN when empty.
    pub fn accuracy_at(&self, q: f64) -> f64 {
        let n = self.len();
        if n == 0 {
            return f64::NAN;
        }
        // The epsilon keeps 0.07 · 100 (7.000000000000001 in f64) at 7.
        let k = ((q * n as f64 - 1e-9).ceil() as usize).clamp(1, n);
        1.0 - self.risk(k)
    }

    /// Largest share `k/n` of the most confident predictions whose error rate is ≤ `max_risk`;
    /// 0 when there is none, NaN when empty.
    pub fn coverage_at(&self, max_risk: f64) -> f64 {
        let n = self.len();
        if n == 0 {
            return f64::NAN;
        }
        (1..=n)
            .rev()
            .find(|&k| self.errors[k] <= max_risk * k as f64 + 1e-9)
            .map_or(0.0, |k| k as f64 / n as f64)
    }

    /// Area under the risk–coverage curve: the mean of [`RiskCoverage::risk`] over
    /// `k = 1..=n`; NaN when empty.
    pub fn aurc(&self) -> f64 {
        mean((1..=self.len()).map(|k| self.risk(k)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-12
    }

    /// Two-option prediction with confidence `c` on option 0, right or wrong.
    fn pred(c: f64, right: bool) -> Pred {
        let target = if right {
            vec![1.0, 0.0]
        } else {
            vec![0.0, 1.0]
        };
        Pred::new(vec![c, 1.0 - c], target, "g")
    }

    #[test]
    fn curve_by_hand() {
        // Ranked: right, wrong, right, wrong. Risks 0, 1/2, 1/3, 1/2.
        let ps = [
            pred(0.6, false),
            pred(0.9, true),
            pred(0.7, true),
            pred(0.8, false),
        ];
        let rc = RiskCoverage::new(&ps);
        assert_eq!(rc.len(), 4);
        assert_eq!(
            (0..=4).map(|k| rc.errors(k)).collect::<Vec<_>>(),
            [0.0, 0.0, 1.0, 1.0, 2.0]
        );
        assert!(close(rc.aurc(), (0.0 + 0.5 + 1.0 / 3.0 + 0.5) / 4.0));
        assert_eq!(rc.coverage_at(0.05), 0.25);
        assert_eq!(rc.coverage_at(0.5), 1.0);
        // ⌈0.4⌉ = 1, ⌈0.8⌉ = 1, ⌈2⌉ = 2 predictions.
        assert_eq!(
            (
                rc.accuracy_at(0.1),
                rc.accuracy_at(0.2),
                rc.accuracy_at(0.5)
            ),
            (1.0, 1.0, 0.5)
        );
        let s = selective(&ps);
        assert_eq!(
            (s.n, s.acc_top10, s.acc_top20, s.acc_top50),
            (4, 1.0, 1.0, 0.5)
        );
        assert_eq!(s.coverage_at_5pct_error, 0.25);
        assert!(close(s.aurc, rc.aurc()));
        // Only the 0.9 one is confident, and it is right.
        assert_eq!((s.confident_errors, s.confident_error_rate), (0, 0.0));
    }

    #[test]
    fn ties_are_resolved_in_expectation() {
        // Two at 0.9 (one wrong), then a right one at 0.5: errors 0.5, 1, 1.
        let a = [pred(0.9, true), pred(0.9, false), pred(0.5, true)];
        let b = [pred(0.5, true), pred(0.9, false), pred(0.9, true)];
        let (ra, rb) = (RiskCoverage::new(&a), RiskCoverage::new(&b));
        assert_eq!(ra, rb);
        assert_eq!(
            (0..=3).map(|k| ra.errors(k)).collect::<Vec<_>>(),
            [0.0, 0.5, 1.0, 1.0]
        );
        assert!(close(ra.aurc(), (0.5 + 0.5 + 1.0 / 3.0) / 3.0));
        assert_eq!(ra.accuracy_at(0.1), 0.5);
        assert_eq!(ra.coverage_at(0.05), 0.0);
        let s = selective(&b);
        assert_eq!(s.confident_errors, 1);
        assert!(close(s.confident_error_rate, 1.0 / 3.0));
    }

    #[test]
    fn coverage_at_exactly_five_percent() {
        // 19 right then 1 wrong: 1/20 = 0.05 is allowed, so full coverage, wherever the error.
        let mut ps: Vec<Pred> = (0..19)
            .map(|i| pred(0.99 - i as f64 * 0.01, true))
            .collect();
        ps.push(pred(0.6, false));
        assert_eq!(RiskCoverage::new(&ps).coverage_at(MAX_RISK), 1.0);
        ps[19] = pred(0.995, false);
        assert_eq!(RiskCoverage::new(&ps).coverage_at(MAX_RISK), 1.0);
        // A second error, last: 1/k > 0.05 for every k < 20 and 2/20 > 0.05.
        ps[18] = pred(0.6, false);
        let s = selective(&ps);
        assert_eq!(s.coverage_at_5pct_error, 0.0);
        // Only the 0.995 error is confident.
        assert_eq!(s.confident_errors, 1);
        assert!(close(s.confident_error_rate, 0.05));
    }

    #[test]
    fn rounding_of_the_top_share_and_empty_sets() {
        // 0.07 · 100 is 7.000000000000001 in f64; ⌈⌉ must still give 7 (all right), not 8.
        let ps: Vec<Pred> = (0..100)
            .map(|i| pred(0.999 - i as f64 * 0.001, i < 7))
            .collect();
        assert_eq!(RiskCoverage::new(&ps).accuracy_at(0.07), 1.0);
        let s = selective([]);
        assert_eq!(s.n, 0);
        assert!(s.acc_top10.is_nan() && s.aurc.is_nan() && s.coverage_at_5pct_error.is_nan());
        assert!(s.confident_error_rate.is_nan());
    }
}
