//! Percentile bootstrap over resampled items: intervals of any statistic, paired differences
//! between two systems scored on the same items, p-values, non-inferiority verdicts and the
//! Holm correction.
//!
//! A resample draws n item indices with replacement and the statistic is recomputed on them,
//! so non-additive statistics (ECE, AURC...) get intervals as well as means. A paired bootstrap
//! scores both systems on the same index sets. Each call reseeds its generator: calls with the
//! same seed on the same number of items draw the same resamples, so the metrics of one
//! comparison share them.

use crate::rng::Rng;
use crate::scores::Metric;
use crate::{mean, pick, Pred};

/// Bootstrap settings.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Bootstrap {
    /// Number of resamples B.
    pub resamples: usize,
    /// Seed of the resampling generator.
    pub seed: u64,
    /// Confidence level of the intervals.
    pub level: f64,
}

impl Default for Bootstrap {
    /// B = 2000, seed 0, 95% intervals.
    fn default() -> Self {
        Self {
            resamples: 2000,
            seed: 0,
            level: 0.95,
        }
    }
}

/// A point estimate with its percentile interval.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Interval {
    /// Statistic on all the items.
    pub value: f64,
    pub lo: f64,
    pub hi: f64,
}

/// Which way a metric improves.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Better {
    /// Accuracy.
    Higher,
    /// ECE, NLL, Brier.
    Lower,
}

/// A statistic of a reference and of a variant on the same items, and their paired difference.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Paired {
    /// Statistic of the reference on all the items.
    pub reference: f64,
    /// Statistic of the variant on all the items.
    pub variant: f64,
    /// Variant − reference: point estimate and percentile interval.
    pub diff: Interval,
    /// Two-sided bootstrap p-value of a non-zero difference ([`p_value`]).
    pub p_value: f64,
}

impl Paired {
    /// Upper bound of the interval of the variant's degradation, oriented so that positive means
    /// worse: of reference − variant when higher is better (accuracy), which is `−diff.lo`; of
    /// variant − reference when lower is better (ECE, NLL, Brier), which is `diff.hi`.
    pub fn worse_upper(&self, better: Better) -> f64 {
        match better {
            // `0 − lo` rather than `−lo`: a zero bound stays +0.
            Better::Higher => 0.0 - self.diff.lo,
            Better::Lower => self.diff.hi,
        }
    }

    /// [`non_inferior`] verdict of the variant against the reference, `margin` in the units of
    /// the metric (0.02 = 2 accuracy points).
    pub fn non_inferior(&self, better: Better, margin: f64) -> bool {
        non_inferior(self.worse_upper(better), margin)
    }
}

/// Non-inferiority rule: the variant is non-inferior iff `worse_upper ≤ margin`, where
/// `worse_upper` is the upper interval bound of how much worse the variant is: of
/// (reference − variant) for accuracy, of (variant − reference) for the ECE. Use
/// [`Paired::worse_upper`] (or [`Paired::non_inferior`]) rather than orienting the difference
/// by hand. NaN fails.
pub fn non_inferior(worse_upper: f64, margin: f64) -> bool {
    worse_upper <= margin
}

impl Bootstrap {
    /// Percentile interval of a statistic: `stat(items, idx)` computes it on the items at `idx`
    /// (repetitions included, see [`pick`]); the point estimate uses every item in order.
    /// Resamples whose statistic is not finite (an empty subgroup...) are left out.
    pub fn ci<T>(&self, items: &[T], mut stat: impl FnMut(&[T], &[usize]) -> f64) -> Interval {
        let all: Vec<usize> = (0..items.len()).collect();
        let value = stat(items, &all);
        let mut draws = Vec::with_capacity(self.resamples);
        self.resample(items.len(), |idx| draws.push(stat(items, idx)));
        let (lo, hi) = self.interval(&mut draws);
        Interval { value, lo, hi }
    }

    /// Paired bootstrap of `stat(variant) − stat(reference)` on the same resampled items.
    ///
    /// # Panics
    /// If the two systems have a different number of items.
    pub fn paired<T>(
        &self,
        reference: &[T],
        variant: &[T],
        mut stat: impl FnMut(&[T], &[usize]) -> f64,
    ) -> Paired {
        assert_eq!(
            reference.len(),
            variant.len(),
            "a paired bootstrap needs both systems on the same items"
        );
        let all: Vec<usize> = (0..reference.len()).collect();
        let (r, v) = (stat(reference, &all), stat(variant, &all));
        let mut draws = Vec::with_capacity(self.resamples);
        self.resample(reference.len(), |idx| {
            draws.push(stat(variant, idx) - stat(reference, idx))
        });
        let (lo, hi) = self.interval(&mut draws);
        Paired {
            reference: r,
            variant: v,
            diff: Interval {
                value: v - r,
                lo,
                hi,
            },
            p_value: p_value(v - r, &draws),
        }
    }

    /// Interval of the mean of per-item values (correctness, NLL...).
    pub fn mean(&self, values: &[f64]) -> Interval {
        self.ci(values, |v, idx| mean(pick(v, idx).copied()))
    }

    /// Paired difference of the means of per-item values of two systems on the same items.
    pub fn paired_mean(&self, reference: &[f64], variant: &[f64]) -> Paired {
        self.paired(reference, variant, |v, idx| mean(pick(v, idx).copied()))
    }

    /// Interval of a headline metric; the means resample per-item values, which gives the same
    /// numbers as recomputing the metric on each resample, faster.
    pub fn metric_ci(&self, metric: Metric, preds: &[Pred]) -> Interval {
        match per_item(metric, preds) {
            Some(v) => self.mean(&v),
            None => self.ci(preds, |p, idx| metric.of(pick(p, idx))),
        }
    }

    /// Paired comparison of a headline metric, variant − reference.
    pub fn metric_paired(&self, metric: Metric, reference: &[Pred], variant: &[Pred]) -> Paired {
        match (per_item(metric, reference), per_item(metric, variant)) {
            (Some(r), Some(v)) => self.paired_mean(&r, &v),
            _ => self.paired(reference, variant, |p, idx| metric.of(pick(p, idx))),
        }
    }

    /// Calls `f` on each of the B resamples of `0..n` (none when `n` is 0).
    fn resample(&self, n: usize, mut f: impl FnMut(&[usize])) {
        if n == 0 {
            return;
        }
        let mut rng = Rng::new(self.seed);
        let mut idx = vec![0usize; n];
        for _ in 0..self.resamples {
            for i in idx.iter_mut() {
                *i = rng.below(n);
            }
            f(&idx);
        }
    }

    /// Keeps the finite draws, sorted, and returns the bounds of the central `level` interval.
    fn interval(&self, draws: &mut Vec<f64>) -> (f64, f64) {
        draws.retain(|d| d.is_finite());
        draws.sort_by(f64::total_cmp);
        let a = (1.0 - self.level) / 2.0;
        (quantile(draws, a), quantile(draws, 1.0 - a))
    }
}

fn per_item(metric: Metric, preds: &[Pred]) -> Option<Vec<f64>> {
    preds.iter().map(|p| metric.item(p)).collect()
}

/// Two-sided bootstrap p-value of an observed difference: twice the share of bootstrap
/// differences that cross 0 (≤ 0 when `observed` > 0, ≥ 0 when it is < 0), clamped to
/// [1/B, 1]. 1 when `observed` is 0, NaN without draws or with a NaN observation.
pub fn p_value(observed: f64, draws: &[f64]) -> f64 {
    let b = draws.len();
    if b == 0 || observed.is_nan() {
        return f64::NAN;
    }
    if observed == 0.0 {
        return 1.0;
    }
    let crossing = draws
        .iter()
        .filter(|&&d| if observed > 0.0 { d <= 0.0 } else { d >= 0.0 })
        .count();
    (2.0 * crossing as f64 / b as f64).clamp(1.0 / b as f64, 1.0)
}

/// Quantile `q` of sorted values, interpolating linearly between the closest ranks (numpy's
/// default); NaN when empty.
pub fn quantile(sorted: &[f64], q: f64) -> f64 {
    let n = sorted.len();
    if n == 0 {
        return f64::NAN;
    }
    let h = (n - 1) as f64 * q.clamp(0.0, 1.0);
    let i = h.floor() as usize;
    let j = (i + 1).min(n - 1);
    sorted[i] + (h - i as f64) * (sorted[j] - sorted[i])
}

/// Holm–Bonferroni adjusted p-values, in the input order: the i-th smallest of m p-values is
/// multiplied by m − i + 1 (i from 1), the running maximum is kept so the order is preserved,
/// and the result is capped at 1. A NaN p-value comes out as 1.
pub fn holm(p: &[f64]) -> Vec<f64> {
    let m = p.len();
    let mut order: Vec<usize> = (0..m).collect();
    order.sort_by(|&a, &b| p[a].total_cmp(&p[b]));
    let mut out = vec![0.0; m];
    let mut run = 0.0f64;
    for (rank, &i) in order.iter().enumerate() {
        run = run.max(((m - rank) as f64 * p[i]).min(1.0));
        out[i] = run;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scores::ece;

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-12
    }

    fn pred(c: f64, right: bool) -> Pred {
        let target = if right {
            vec![1.0, 0.0]
        } else {
            vec![0.0, 1.0]
        };
        Pred::new(vec![c, 1.0 - c], target, "g")
    }

    #[test]
    fn quantiles_by_hand() {
        let v = [1.0, 2.0, 3.0, 4.0];
        assert_eq!(quantile(&v, 0.0), 1.0);
        assert_eq!(quantile(&v, 0.25), 1.75);
        assert_eq!(quantile(&v, 0.5), 2.5);
        assert_eq!(quantile(&v, 1.0), 4.0);
        assert_eq!(quantile(&[7.0], 0.3), 7.0);
        assert!(quantile(&[], 0.5).is_nan());
    }

    #[test]
    fn p_values_by_hand() {
        assert_eq!(p_value(0.1, &[-0.1, 0.0, 0.2, 0.3]), 1.0);
        assert_eq!(p_value(0.1, &[0.1, 0.2, -0.1, 0.3]), 0.5);
        assert_eq!(p_value(-0.1, &[0.1, -0.2, -0.1, -0.3]), 0.5);
        // No draw crosses 0: clamped to 1/B.
        assert_eq!(p_value(0.1, &[0.1, 0.2, 0.1, 0.3]), 0.25);
        assert_eq!(p_value(0.0, &[0.1, 0.2]), 1.0);
        assert!(p_value(0.1, &[]).is_nan());
    }

    #[test]
    fn holm_by_hand() {
        // Sorted 0.005, 0.01, 0.03, 0.04 → 0.02, 0.03, 0.06, max(0.04, 0.06).
        let adj = holm(&[0.01, 0.04, 0.03, 0.005]);
        for (a, b) in adj.iter().zip([0.03, 0.06, 0.06, 0.02]) {
            assert!(close(*a, b), "{adj:?}");
        }
        assert_eq!(holm(&[0.5, 0.6]), [1.0, 1.0]);
        assert_eq!(holm(&[0.01, 0.01]), [0.02, 0.02]);
        assert_eq!(holm(&[0.2]), [0.2]);
        assert_eq!(holm(&[f64::NAN, 0.01]), [1.0, 0.02]);
        assert!(holm(&[]).is_empty());
    }

    #[test]
    fn constant_and_shifted_systems() {
        let boot = Bootstrap::default();
        let c = boot.mean(&[0.5; 40]);
        assert_eq!((c.value, c.lo, c.hi), (0.5, 0.5, 0.5));
        // Variant = reference + 0.25 on every item: the difference never moves.
        let r: Vec<f64> = (0..50).map(|i| (i % 7) as f64).collect();
        let v: Vec<f64> = r.iter().map(|x| x + 0.25).collect();
        let d = boot.paired_mean(&r, &v);
        assert!(close(d.diff.value, 0.25) && close(d.diff.lo, 0.25) && close(d.diff.hi, 0.25));
        assert_eq!(d.p_value, 1.0 / 2000.0);
        // Identical systems: zero difference, p = 1.
        let same = boot.paired_mean(&r, &r);
        assert_eq!(
            (same.diff.value, same.diff.lo, same.diff.hi, same.p_value),
            (0.0, 0.0, 0.0, 1.0)
        );
    }

    #[test]
    fn interval_of_a_proportion_is_about_right() {
        // 200 of 400 right: the 95% interval is about ±1.96·√(0.25/400) = ±0.049.
        let v: Vec<f64> = (0..400).map(|i| (i % 2) as f64).collect();
        let c = Bootstrap::default().mean(&v);
        assert_eq!(c.value, 0.5);
        assert!((0.04..0.06).contains(&(c.value - c.lo)), "{c:?}");
        assert!((0.04..0.06).contains(&(c.hi - c.value)), "{c:?}");
    }

    #[test]
    fn deterministic_and_seeded() {
        let v: Vec<f64> = (0..100).map(|i| ((i * 37) % 11) as f64).collect();
        let boot = Bootstrap::default();
        assert_eq!(boot.mean(&v), boot.mean(&v));
        let other = Bootstrap { seed: 1, ..boot };
        assert_ne!(boot.mean(&v), other.mean(&v));
    }

    #[test]
    fn ece_differences_are_bootstrapped_on_shared_resamples() {
        // The same answers, 70% right: the reference says 0.7 (calibrated), the variant 0.99.
        let reference: Vec<Pred> = (0..100).map(|i| pred(0.7, i % 10 < 7)).collect();
        let variant: Vec<Pred> = (0..100).map(|i| pred(0.99, i % 10 < 7)).collect();
        let boot = Bootstrap::default();
        let d = boot.paired(&reference, &variant, |p, idx| ece(pick(p, idx)));
        assert_eq!((d.reference, d.variant), (ece(&reference), ece(&variant)));
        assert!(close(d.diff.value, 0.29), "{d:?}");
        assert!(d.diff.lo > 0.0 && d.diff.lo < d.diff.value, "{d:?}");
        assert!(!d.non_inferior(Better::Lower, 0.01));
        assert!(d.p_value < 0.01);
        // The headline-metric entry point gives the same numbers.
        assert_eq!(boot.metric_paired(Metric::Ece, &reference, &variant), d);
        // Resampling per-item values gives exactly the recomputed metric.
        let slow = boot.paired(&reference, &variant, |p, idx| Metric::Nll.of(pick(p, idx)));
        assert_eq!(boot.metric_paired(Metric::Nll, &reference, &variant), slow);
        let one = boot.metric_ci(Metric::Accuracy, &reference);
        assert_eq!(one.value, 0.7);
        assert!(one.lo < one.value && one.value < one.hi);
    }

    #[test]
    fn non_inferiority_is_oriented_by_the_metric() {
        // The variant loses 3 points of accuracy (6 of 200 answers flipped to wrong).
        let r: Vec<f64> = (0..200).map(|i| (i % 4 != 0) as u8 as f64).collect();
        let mut v = r.clone();
        for x in v.iter_mut().filter(|x| **x == 1.0).take(6) {
            *x = 0.0;
        }
        let boot = Bootstrap::default();
        let d = boot.paired_mean(&r, &v);
        assert!(close(d.diff.value, -0.03));
        // Upper bound of reference − variant, computed directly with the roles swapped.
        let swapped = boot.paired_mean(&v, &r);
        assert!((d.worse_upper(Better::Higher) - swapped.diff.hi).abs() < 1e-12);
        assert!(d.worse_upper(Better::Higher) >= 0.03);
        assert!(!d.non_inferior(Better::Higher, 0.02));
        assert!(d.non_inferior(Better::Higher, 0.10));
        // Read as a lower-is-better metric, the same numbers are an improvement.
        assert_eq!(d.worse_upper(Better::Lower), d.diff.hi);
        assert!(non_inferior(0.019, 0.02) && !non_inferior(0.021, 0.02));
        assert!(!non_inferior(f64::NAN, 0.02));
    }

    #[test]
    fn empty_sets_give_nan() {
        let boot = Bootstrap::default();
        let c = boot.mean(&[]);
        assert!(c.value.is_nan() && c.lo.is_nan() && c.hi.is_nan());
        let d = boot.paired_mean(&[], &[]);
        assert!(d.diff.value.is_nan() && d.p_value.is_nan());
    }
}
