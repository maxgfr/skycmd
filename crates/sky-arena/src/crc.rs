//! Early-exit thresholds by conformal risk control (SPEC-S6 §9; SAFE-KD, arXiv 2602.03043).
//!
//! Each calibration decision records, per exit, its top calibrated probability and its answer.
//! At threshold τ the engine answers at the first exit whose top probability reaches τ (the last
//! exit otherwise); the loss of a decision is 1 when that answer differs from the full-depth one.
//! The threshold kept for a question type is the smallest τ of the grid such that, for it and
//! every larger τ of the grid, the conformal bound `(n·R̂(τ) + 1) / (n + 1)` stays ≤ α. Scanning
//! down from τ = 1 and stopping at the first violation keeps the choice valid although the
//! empirical risk need not be monotone in τ.
//!
//! The guarantee needs exchangeability: the decisions are drawn while playing at full depth,
//! and playing with early exits changes which states are visited. Check the chosen thresholds by
//! playing with them (`--exit-threshold` or the written file).

/// One decision at full depth: per exit, (top calibrated probability, answer).
#[derive(Debug, Clone, PartialEq)]
pub struct Sample {
    pub exits: Vec<(f32, usize)>,
}

impl Sample {
    /// From the calibrated probabilities of every exit.
    pub fn from_probs(per_exit: &[Vec<f32>]) -> Self {
        let exits = per_exit
            .iter()
            .map(|p| {
                p.iter()
                    .enumerate()
                    .fold((f32::NEG_INFINITY, 0), |(best, i), (j, &x)| {
                        if x > best {
                            (x, j)
                        } else {
                            (best, i)
                        }
                    })
            })
            .collect();
        Sample { exits }
    }

    /// The exit that answers at threshold `tau`.
    pub fn exit_at(&self, tau: f32) -> usize {
        let last = self.exits.len() - 1;
        if tau >= 1.0 {
            return last;
        }
        self.exits
            .iter()
            .position(|&(top, _)| top >= tau)
            .unwrap_or(last)
    }

    /// 1 when the answer at `tau` differs from the full-depth answer.
    pub fn loss(&self, tau: f32) -> f64 {
        let last = self.exits.len() - 1;
        (self.exits[self.exit_at(tau)].1 != self.exits[last].1) as u8 as f64
    }
}

/// The candidate thresholds, from 1.0 down.
pub fn grid() -> Vec<f32> {
    let mut g: Vec<f32> = (50..100).rev().map(|i| i as f32 / 100.0).collect();
    g.insert(0, 1.0);
    g
}

/// The threshold of one question type.
#[derive(Debug, Clone, PartialEq)]
pub struct Choice {
    pub tau: f32,
    /// Empirical disagreement with full depth at `tau`.
    pub risk: f64,
    /// Conformal bound at `tau`.
    pub bound: f64,
    /// Mean index of the exit that answers at `tau` (0-based).
    pub mean_exit: f64,
    pub n: usize,
}

/// The smallest τ of [`grid`] whose bound, and every larger τ's, is ≤ `alpha`. τ = 1 when even
/// full depth cannot meet `alpha` (too few samples: `1 / (n + 1) > alpha`) or there are none.
pub fn choose(samples: &[Sample], alpha: f64) -> Choice {
    let n = samples.len();
    let at = |tau: f32| -> Choice {
        let risk = if n == 0 {
            0.0
        } else {
            samples.iter().map(|s| s.loss(tau)).sum::<f64>() / n as f64
        };
        let mean_exit = if n == 0 {
            0.0
        } else {
            samples.iter().map(|s| s.exit_at(tau) as f64).sum::<f64>() / n as f64
        };
        Choice {
            tau,
            risk,
            bound: (n as f64 * risk + 1.0) / (n as f64 + 1.0),
            mean_exit,
            n,
        }
    };
    let mut best = at(1.0);
    if n == 0 || best.bound > alpha {
        return best;
    }
    for tau in grid().into_iter().skip(1) {
        let c = at(tau);
        if c.bound > alpha {
            break;
        }
        best = c;
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s(exits: &[(f32, usize)]) -> Sample {
        Sample {
            exits: exits.to_vec(),
        }
    }

    #[test]
    fn exit_rule_matches_the_engine() {
        let x = s(&[(0.6, 1), (0.8, 0), (0.7, 0)]);
        assert_eq!(x.exit_at(1.0), 2);
        assert_eq!(x.exit_at(0.6), 0);
        assert_eq!(x.exit_at(0.75), 1);
        assert_eq!(x.exit_at(0.9), 2);
        assert_eq!(x.loss(0.6), 1.0); // exit 0 says 1, full depth says 0
        assert_eq!(x.loss(0.75), 0.0);
        let p = Sample::from_probs(&[vec![0.2, 0.8], vec![0.6, 0.4]]);
        assert_eq!(p.exits, vec![(0.8, 1), (0.6, 0)]);
    }

    #[test]
    fn chooses_the_lowest_threshold_whose_bound_holds_from_the_top() {
        // 99 decisions whose first exit is right and confident (0.9), one whose first exit is
        // wrong at 0.7: τ ≤ 0.7 costs a disagreement of 1/100, bound 2/101 ≈ 0.0198.
        let mut v: Vec<Sample> = (0..99).map(|_| s(&[(0.9, 1), (0.95, 1)])).collect();
        v.push(s(&[(0.7, 0), (0.99, 1)]));
        let c = choose(&v, 0.05);
        assert_eq!(c.tau, 0.5);
        assert!((c.risk - 0.01).abs() < 1e-12);
        assert!((c.bound - 2.0 / 101.0).abs() < 1e-12);
        let strict = choose(&v, 0.015);
        // 1/101 ≈ 0.0099 holds while the wrong exit stays below τ (τ > 0.7)…
        assert!((strict.tau - 0.71).abs() < 1e-6, "{strict:?}");
        assert_eq!(strict.risk, 0.0);
        assert!(strict.mean_exit < 0.02);
    }

    #[test]
    fn too_few_samples_keep_full_depth() {
        let v = vec![s(&[(0.99, 1), (0.99, 1)]); 5];
        let c = choose(&v, 0.1); // 1 / 6 > 0.1
        assert_eq!(c.tau, 1.0);
        assert_eq!(choose(&[], 0.1).tau, 1.0);
    }

    #[test]
    fn stops_at_the_first_violation_scanning_down() {
        // Half the decisions disagree with full depth as soon as τ ≤ 0.8: the scan keeps 0.81.
        let mut v: Vec<Sample> = (0..50).map(|_| s(&[(0.95, 1), (0.99, 1)])).collect();
        v.extend((0..50).map(|_| s(&[(0.8, 0), (0.99, 1)])));
        let c = choose(&v, 0.05);
        assert!((c.tau - 0.81).abs() < 1e-6, "{c:?}");
    }
}
