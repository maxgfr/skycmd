//! Proper scores and calibration errors of a set of predictions: means over the items, NaN on
//! an empty set.

use crate::bootstrap::Better;
use crate::{mean, Pred};

/// Number of equal-width bins of the calibration errors.
pub const ECE_BINS: usize = 15;

/// Floor of the probabilities in the NLL.
pub const NLL_FLOOR: f64 = 1e-12;

/// Share of items whose predicted option is the gold one.
pub fn accuracy<'a>(preds: impl IntoIterator<Item = &'a Pred>) -> f64 {
    mean(preds.into_iter().map(Pred::correct))
}

/// Mean negative log-likelihood of the targets ([`Pred::nll`]).
pub fn nll<'a>(preds: impl IntoIterator<Item = &'a Pred>) -> f64 {
    mean(preds.into_iter().map(Pred::nll))
}

/// Mean Brier score ([`Pred::brier`]).
pub fn brier<'a>(preds: impl IntoIterator<Item = &'a Pred>) -> f64 {
    mean(preds.into_iter().map(Pred::brier))
}

/// Mean ranked probability score ([`Pred::rps`]); only meaningful on ordered options, so pass
/// the score questions only.
pub fn rps<'a>(preds: impl IntoIterator<Item = &'a Pred>) -> f64 {
    mean(preds.into_iter().map(Pred::rps))
}

/// Top-label ECE: [`calibration_error`] of the confidence (largest probability) against
/// correctness, [`ECE_BINS`] bins.
pub fn ece<'a>(preds: impl IntoIterator<Item = &'a Pred>) -> f64 {
    calibration_error(
        preds.into_iter().map(|p| (p.confidence(), p.correct())),
        ECE_BINS,
    )
}

/// Classwise ECE: mean over the classes of the one-vs-rest [`calibration_error`] of `p_k`
/// against `1{gold = k}` ([`ECE_BINS`] bins). Classes are option positions; only the classes
/// present, i.e. the gold option of at least one item, are averaged, and class `k` is scored on
/// the items that have an option `k`.
pub fn classwise_ece<'a>(preds: impl IntoIterator<Item = &'a Pred>) -> f64 {
    let preds: Vec<&Pred> = preds.into_iter().collect();
    let k_max = preds.iter().map(|p| p.k()).max().unwrap_or(0);
    let mut present = vec![false; k_max];
    for p in &preds {
        present[p.gold] = true;
    }
    mean((0..k_max).filter(|&k| present[k]).map(|k| {
        calibration_error(
            preds
                .iter()
                .filter(|p| k < p.k())
                .map(|p| (p.probs[k], (p.gold == k) as u8 as f64)),
            ECE_BINS,
        )
    }))
}

/// Expected calibration error of `(probability, outcome)` pairs over `bins` equal-width bins:
/// Σ_b (n_b / n) |mean probability − mean outcome| over the non-empty bins, bin `b` covering
/// `(b/bins, (b+1)/bins]`. These are the bins and the arithmetic of `sky-train`'s
/// `metrics::ece`, so the values are identical for probabilities in (0, 1]; a probability ≤ 0,
/// which `sky-train` leaves out of every bin, falls in the first one (and one above 1 in the
/// last). NaN when empty.
pub fn calibration_error(pairs: impl IntoIterator<Item = (f64, f64)>, bins: usize) -> f64 {
    let bins = bins.max(1);
    let mut acc = vec![(0.0f64, 0.0f64, 0usize); bins];
    let mut n = 0usize;
    for (p, y) in pairs {
        let b = &mut acc[bin(p, bins)];
        b.0 += p;
        b.1 += y;
        b.2 += 1;
        n += 1;
    }
    if n == 0 {
        return f64::NAN;
    }
    let n = n as f64;
    let mut e = 0.0;
    for &(sp, sy, k) in acc.iter().filter(|b| b.2 > 0) {
        let k = k as f64;
        e += k / n * (sp / k - sy / k).abs();
    }
    e
}

/// Bin of `p`: the `b` with `b/bins < p <= (b+1)/bins`, the edges computed as in `sky-train`.
fn bin(p: f64, bins: usize) -> usize {
    let edge = |b: usize| b as f64 / bins as f64;
    let mut b = ((p * bins as f64).ceil() as usize)
        .saturating_sub(1)
        .min(bins - 1);
    while b > 0 && p <= edge(b) {
        b -= 1;
    }
    while b + 1 < bins && p > edge(b + 1) {
        b += 1;
    }
    b
}

/// The headline metrics, each with the direction in which it improves.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Metric {
    Accuracy,
    Nll,
    Brier,
    /// Top-label ECE.
    Ece,
}

impl Metric {
    pub const ALL: [Metric; 4] = [Metric::Accuracy, Metric::Nll, Metric::Brier, Metric::Ece];

    /// Key in reports: `accuracy`, `nll`, `brier`, `ece`.
    pub fn name(self) -> &'static str {
        match self {
            Metric::Accuracy => "accuracy",
            Metric::Nll => "nll",
            Metric::Brier => "brier",
            Metric::Ece => "ece",
        }
    }

    /// Higher is better for accuracy, lower for the others.
    pub fn better(self) -> Better {
        match self {
            Metric::Accuracy => Better::Higher,
            _ => Better::Lower,
        }
    }

    /// Value on `preds`.
    pub fn of<'a>(self, preds: impl IntoIterator<Item = &'a Pred>) -> f64 {
        match self {
            Metric::Accuracy => accuracy(preds),
            Metric::Nll => nll(preds),
            Metric::Brier => brier(preds),
            Metric::Ece => ece(preds),
        }
    }

    /// Per-item value whose mean is the metric (correctness, NLL, Brier); `None` for the ECE,
    /// which is not a mean.
    pub fn item(self, p: &Pred) -> Option<f64> {
        match self {
            Metric::Accuracy => Some(p.correct()),
            Metric::Nll => Some(p.nll()),
            Metric::Brier => Some(p.brier()),
            Metric::Ece => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-12
    }

    fn pred(probs: &[f64], gold: usize) -> Pred {
        let mut t = vec![0.0; probs.len()];
        t[gold] = 1.0;
        Pred::new(probs.to_vec(), t, "g")
    }

    /// `sky-train`'s `metrics::ece`, verbatim.
    fn sky_train_ece(conf: &[f64], correct: &[f64], bins: usize) -> f64 {
        if conf.is_empty() {
            return f64::NAN;
        }
        let n = conf.len() as f64;
        let mut e = 0.0;
        for b in 0..bins {
            let lo = b as f64 / bins as f64;
            let hi = (b + 1) as f64 / bins as f64;
            let sel: Vec<usize> = (0..conf.len())
                .filter(|&i| conf[i] > lo && conf[i] <= hi)
                .collect();
            if sel.is_empty() {
                continue;
            }
            let c = sel.iter().map(|&i| conf[i]).sum::<f64>() / sel.len() as f64;
            let a = sel.iter().map(|&i| correct[i]).sum::<f64>() / sel.len() as f64;
            e += sel.len() as f64 / n * (c - a).abs();
        }
        e
    }

    #[test]
    fn means_by_hand() {
        let ps = [pred(&[0.7, 0.3], 0), pred(&[0.4, 0.6], 0)];
        assert_eq!(accuracy(&ps), 0.5);
        assert!(close(nll(&ps), -(0.7f64.ln() + 0.4f64.ln()) / 2.0));
        // (0.09 + 0.09 + 0.36 + 0.36) / 2
        assert!(close(brier(&ps), 0.45));
        // RPS with K = 2 is (p_0 − y_0)²: (0.09 + 0.36) / 2
        assert!(close(rps(&ps), 0.225));
        assert!(accuracy([]).is_nan() && ece([]).is_nan() && classwise_ece([]).is_nan());
    }

    #[test]
    fn ece_by_hand() {
        // conf 0.9 right, conf 0.9 wrong: one bin, |0.9 − 0.5|.
        assert!(close(
            ece(&[pred(&[0.9, 0.1], 0), pred(&[0.9, 0.1], 1)]),
            0.4
        ));
        // Bins (0.8, 0.867] and (0.533, 0.6]: 0.5·|0.85 − 1| + 0.5·|0.6 − 0|.
        let ps = [pred(&[0.85, 0.15], 0), pred(&[0.6, 0.4], 1)];
        assert!(close(ece(&ps), 0.5 * 0.15 + 0.5 * 0.6));
    }

    #[test]
    fn binning_is_sky_train_s() {
        // Every edge k/15, its neighbours and a spread of values, against the original code.
        let mut conf = Vec::new();
        for k in 1..=15 {
            let e = k as f64 / 15.0;
            conf.extend([e, e - 1e-12, e + 1e-12, (k as f32 / 15.0) as f64]);
        }
        conf.extend((1..=997).map(|i| i as f64 / 997.0));
        let correct: Vec<f64> = (0..conf.len()).map(|i| (i % 3 == 0) as u8 as f64).collect();
        let conf: Vec<f64> = conf.into_iter().map(|c| c.min(1.0)).collect();
        let pairs = conf.iter().copied().zip(correct.iter().copied());
        let ours = calibration_error(pairs, 15);
        assert_eq!(ours.to_bits(), sky_train_ece(&conf, &correct, 15).to_bits());
        for (i, &c) in conf.iter().enumerate() {
            let b = bin(c, 15);
            assert!(
                c > b as f64 / 15.0 && c <= (b + 1) as f64 / 15.0,
                "{i}: {c} in {b}"
            );
        }
        assert_eq!((bin(0.0, 15), bin(-1.0, 15), bin(1.5, 15)), (0, 0, 14));
    }

    #[test]
    fn classwise_by_hand() {
        // Class 0: (0.8, 1) and (0.6, 0) in two bins: 0.5·0.2 + 0.5·0.6.
        // Class 1: (0.2, 0) and (0.4, 1) in two bins: 0.5·0.2 + 0.5·0.6.
        let ps = [pred(&[0.8, 0.2], 0), pred(&[0.6, 0.4], 1)];
        assert!(close(classwise_ece(&ps), 0.4));
        // Class 2 is never gold: only classes 0 and 1 are averaged.
        // Class 0: (0.5, 1), (0.1, 0), (0.7, 1): 1/3·(0.5 + 0.1 + 0.3).
        // Class 1: (0.25, 0), (0.6, 1), (0.2, 0): 1/3·(0.25 + 0.4 + 0.2).
        let ps = [
            pred(&[0.5, 0.25, 0.25], 0),
            pred(&[0.1, 0.6, 0.3], 1),
            pred(&[0.7, 0.2, 0.1], 0),
        ];
        assert!(close(classwise_ece(&ps), (0.9 / 3.0 + 0.85 / 3.0) / 2.0));
    }

    #[test]
    fn classwise_scores_class_k_on_items_with_k_options() {
        // Class 2 (gold of the 3-option item) is scored on that item alone: |0.6 − 1|.
        // Class 0: (0.9, 1), (0.3, 0) in two bins: 0.5·0.1 + 0.5·0.3.
        let ps = [pred(&[0.9, 0.1], 0), pred(&[0.3, 0.1, 0.6], 2)];
        assert!(close(classwise_ece(&ps), (0.2 + 0.4) / 2.0));
    }

    #[test]
    fn metric_dispatch() {
        let ps = [pred(&[0.7, 0.3], 0), pred(&[0.4, 0.6], 0)];
        for m in Metric::ALL {
            let v = m.of(&ps);
            if let Some(f) = ps.first().and_then(|p| m.item(p)) {
                assert!(f.is_finite());
                assert_eq!(
                    v.to_bits(),
                    mean(ps.iter().map(|p| m.item(p).unwrap())).to_bits()
                );
            }
        }
        assert_eq!(Metric::Accuracy.better(), Better::Higher);
        assert_eq!(Metric::Ece.better(), Better::Lower);
        assert_eq!(
            Metric::ALL.map(Metric::name),
            ["accuracy", "nll", "brier", "ece"]
        );
    }
}
