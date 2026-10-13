//! Evaluation metrics of decision models, shared by `sky-eval`, `sky-compare` and the s5
//! analyses. Dependency-free, so `sky-infer` (also built for wasm32) can link it.
//!
//! - [`Pred`]: one evaluated question (probabilities, target, gold option, group).
//! - [`scores`]: accuracy, NLL, Brier, RPS, top-label and classwise ECE.
//! - [`selective`]: the most confident answers: accuracy at a coverage, confident errors,
//!   coverage at a target error rate, AURC.
//! - [`bootstrap`]: percentile intervals of any statistic over resampled items, paired
//!   differences between two systems, p-values, non-inferiority and the Holm correction.
//! - [`baseline`]: uniform and majority-class predictors.
//! - [`rng`]: the deterministic generator behind the resamples.
//!
//! Aggregates take `impl IntoIterator<Item = &Pred>`, so one function scores a whole set, a
//! group or a bootstrap resample ([`pick`]); they return NaN on an empty set.

pub mod baseline;
pub mod bootstrap;
pub mod rng;
pub mod scores;
pub mod selective;

pub use baseline::{uniform, Majority};
pub use bootstrap::{holm, non_inferior, p_value, quantile, Better, Bootstrap, Interval, Paired};
pub use rng::{hash_str, Rng};
pub use scores::{
    accuracy, brier, calibration_error, classwise_ece, ece, nll, rps, Metric, ECE_BINS, NLL_FLOOR,
};
pub use selective::{selective, RiskCoverage, Selective, CONFIDENT, MAX_RISK};

/// One evaluated question.
#[derive(Debug, Clone, PartialEq)]
pub struct Pred {
    /// Predicted distribution over the options (sums to 1).
    pub probs: Vec<f64>,
    /// Reference distribution over the same options, one-hot or soft (sums to 1).
    pub target: Vec<f64>,
    /// Gold option: [`argmax`] of `target`, the first one on ties.
    pub gold: usize,
    /// Breakdown key (source, question type...); [`Majority`] estimates its prior per group.
    pub group: String,
}

impl Pred {
    /// A prediction whose gold option is the argmax of `target`.
    ///
    /// # Panics
    /// If `probs` is empty or `probs` and `target` differ in length.
    pub fn new(probs: Vec<f64>, target: Vec<f64>, group: impl Into<String>) -> Self {
        assert!(!probs.is_empty(), "a prediction needs at least one option");
        assert_eq!(
            probs.len(),
            target.len(),
            "probs and target need one value per option"
        );
        let gold = argmax(&target);
        Self {
            probs,
            target,
            gold,
            group: group.into(),
        }
    }

    /// Number of options K.
    pub fn k(&self) -> usize {
        self.probs.len()
    }

    /// Predicted option: argmax of `probs`, the first one on ties.
    pub fn predicted(&self) -> usize {
        argmax(&self.probs)
    }

    /// Confidence: the probability of the predicted option.
    pub fn confidence(&self) -> f64 {
        self.probs[self.predicted()]
    }

    /// 1 when the predicted option is the gold one, else 0.
    pub fn correct(&self) -> f64 {
        (self.predicted() == self.gold) as u8 as f64
    }

    /// Negative log-likelihood −Σ y log p, with p floored at [`NLL_FLOOR`].
    pub fn nll(&self) -> f64 {
        -self
            .target
            .iter()
            .zip(&self.probs)
            .map(|(y, p)| y * p.max(NLL_FLOOR).ln())
            .sum::<f64>()
    }

    /// Brier score Σ (p − y)².
    pub fn brier(&self) -> f64 {
        self.probs
            .iter()
            .zip(&self.target)
            .map(|(p, y)| (p - y).powi(2))
            .sum()
    }

    /// Ranked probability score of ordered options (score questions):
    /// Σ_{k<K} (CDF_p(k) − CDF_y(k))² / (K − 1), the cumulative sums running over the options in
    /// order (the last term, 1 − 1, is left out); 0 when K = 1.
    pub fn rps(&self) -> f64 {
        let k = self.k();
        if k < 2 {
            return 0.0;
        }
        let (mut cp, mut cy, mut s) = (0.0, 0.0, 0.0);
        for (p, y) in self.probs.iter().zip(&self.target).take(k - 1) {
            cp += p;
            cy += y;
            s += (cp - cy).powi(2);
        }
        s / (k - 1) as f64
    }
}

/// Index of the largest value, the first one on ties.
pub fn argmax(v: &[f64]) -> usize {
    let mut best = 0;
    for (i, x) in v.iter().enumerate() {
        if *x > v[best] {
            best = i;
        }
    }
    best
}

/// The items at `idx`, repetitions included: a bootstrap resample, a group...
pub fn pick<'a, T>(items: &'a [T], idx: &'a [usize]) -> impl Iterator<Item = &'a T> + Clone + 'a {
    idx.iter().map(move |&i| &items[i])
}

/// Mean of `values`, summed in order; NaN when empty.
pub fn mean(values: impl IntoIterator<Item = f64>) -> f64 {
    let (mut sum, mut n) = (0.0, 0usize);
    for v in values {
        sum += v;
        n += 1;
    }
    if n == 0 {
        f64::NAN
    } else {
        sum / n as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-12
    }

    #[test]
    fn item_scores_by_hand() {
        let p = Pred::new(vec![0.2, 0.5, 0.3], vec![0.0, 1.0, 0.0], "g");
        assert_eq!((p.k(), p.gold, p.predicted()), (3, 1, 1));
        assert_eq!((p.confidence(), p.correct()), (0.5, 1.0));
        assert!(close(p.nll(), 2f64.ln()));
        // 0.2² + 0.5² + 0.3²
        assert!(close(p.brier(), 0.38));
        // CDF_p = (0.2, 0.7), CDF_y = (0, 1): (0.04 + 0.09) / 2
        assert!(close(p.rps(), 0.065));
    }

    #[test]
    fn soft_target_gold_on_ties_and_floor() {
        let p = Pred::new(vec![0.2, 0.5, 0.3], vec![0.5, 0.5, 0.0], "g");
        assert_eq!(p.gold, 0);
        assert_eq!(p.correct(), 0.0);
        assert!(close(p.nll(), -0.5 * 0.2f64.ln() - 0.5 * 0.5f64.ln()));
        // CDF_p = (0.2, 0.7), CDF_y = (0.5, 1): (0.09 + 0.09) / 2
        assert!(close(p.rps(), 0.09));
        let sure = Pred::new(vec![1.0, 0.0], vec![0.0, 1.0], "g");
        assert!(close(sure.nll(), -(1e-12f64).ln()));
        assert_eq!(Pred::new(vec![1.0], vec![1.0], "g").rps(), 0.0);
    }

    #[test]
    fn argmax_pick_mean() {
        assert_eq!(argmax(&[0.1, 0.4, 0.4, 0.1]), 1);
        let v = [10.0, 20.0, 30.0];
        let got: Vec<f64> = pick(&v, &[2, 0, 2]).copied().collect();
        assert_eq!(got, [30.0, 10.0, 30.0]);
        assert_eq!(mean([1.0, 2.0, 6.0]), 3.0);
        assert!(mean([]).is_nan());
    }
}
