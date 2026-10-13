//! s5 training losses over masked option logits `[B, K]` (invalid options at -1e4):
//! soft cross-entropy, Brier, the ranked probability score of ordinal (`score`) questions, and a
//! symmetric KL between the two members of a pair (the same question under two renderings).
//! No label smoothing.

use mlx_rs::{error::Exception, ops, Array, Dtype};

type R<T> = std::result::Result<T, Exception>;

/// Loss weights.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct LossWeights {
    /// Brier term (all questions).
    pub brier: f32,
    /// Ranked probability score (score questions only).
    pub rps: f32,
    /// Symmetric KL between pair members.
    pub kl: f32,
}

impl Default for LossWeights {
    fn default() -> Self {
        Self {
            brier: 1.0,
            rps: 0.25,
            kl: 0.0,
        }
    }
}

/// Inputs of the pair term: rows `left[i]` and `right[i]` hold the same options, option `j` of
/// the left row being option `map[i, j]` of the right row (`mask[i, j]` for valid options);
/// `weight[i]` is 0 for padding pairs.
#[derive(Debug, Clone)]
pub struct Pairs {
    pub left: Array,
    pub right: Array,
    pub map: Array,
    pub mask: Array,
    pub weight: Array,
}

/// Log-softmax over the last axis.
pub fn log_softmax(logits: &Array) -> R<Array> {
    logits.subtract(logits.logsumexp_axis(-1, true)?)
}

/// Per-row soft cross-entropy `-Σ y log p` `[B]`.
pub fn cross_entropy_rows(logp: &Array, target: &Array) -> R<Array> {
    target.multiply(logp)?.sum_axis(-1, None)?.negative()
}

/// Per-row Brier score `Σ (p - y)²` `[B]` (padding options have p = y = 0).
pub fn brier_rows(logp: &Array, target: &Array) -> R<Array> {
    ops::exp(logp)?
        .subtract(target)?
        .square()?
        .sum_axis(-1, None)
}

/// Per-row ranked probability score `Σ_k (P_k - Y_k)² / (K - 1)` over the cumulative
/// distributions, `[B]`; `k` holds the number of valid options per row (padding options come
/// last, where both cumulative sums are 1).
pub fn rps_rows(logp: &Array, target: &Array, k: &Array) -> R<Array> {
    let cp = ops::cumsum(ops::exp(logp)?, -1, None, None)?;
    let cy = ops::cumsum(target, -1, None, None)?;
    let num = cp.subtract(cy)?.square()?.sum_axis(-1, None)?;
    let den = ops::maximum(k.subtract(Array::from_f32(1.0))?, Array::from_f32(1.0))?;
    num.divide(den)
}

/// Mean symmetric KL `½ (KL(l‖r) + KL(r‖l))` over the weighted pairs.
pub fn pair_kl(logp: &Array, pairs: &Pairs) -> R<Array> {
    let l = logp.take_axis(&pairs.left, 0)?;
    let r = logp
        .take_axis(&pairs.right, 0)?
        .take_along_axis(&pairs.map, -1)?;
    let m = pairs.mask.as_dtype(Dtype::Float32)?;
    let d = l.subtract(&r)?;
    let kl = ops::exp(&l)?
        .subtract(ops::exp(&r)?)?
        .multiply(&d)?
        .multiply(&m)?
        .sum_axis(-1, None)?
        .multiply(Array::from_f32(0.5))?;
    let w = &pairs.weight;
    kl.multiply(w)?
        .sum(None)?
        .divide(ops::maximum(w.sum(None)?, Array::from_f32(1.0))?)
}

/// The full loss: mean over rows of CE + w·Brier + w·RPS·[score], plus w·KL over the pairs.
pub fn total(
    masked_logits: &Array,
    target: &Array,
    is_score: &Array,
    k: &Array,
    pairs: Option<&Pairs>,
    w: &LossWeights,
) -> R<Array> {
    let logp = log_softmax(masked_logits)?;
    let mut rows = cross_entropy_rows(&logp, target)?;
    if w.brier != 0.0 {
        rows = rows.add(brier_rows(&logp, target)?.multiply(Array::from_f32(w.brier))?)?;
    }
    if w.rps != 0.0 {
        rows = rows.add(
            rps_rows(&logp, target, k)?
                .multiply(is_score)?
                .multiply(Array::from_f32(w.rps))?,
        )?;
    }
    let mut loss = rows.mean(None)?;
    if let (Some(p), true) = (pairs, w.kl != 0.0) {
        loss = loss.add(pair_kl(&logp, p)?.multiply(Array::from_f32(w.kl))?)?;
    }
    Ok(loss)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{model::host_f32, test_util};

    fn arr(v: &[f32], shape: &[i32]) -> Array {
        Array::from_slice(v, shape)
    }

    fn reference(logits: &[f32], y: &[f32]) -> (f64, f64, f64) {
        let m = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64;
        let z: f64 = logits.iter().map(|&x| (x as f64 - m).exp()).sum();
        let p: Vec<f64> = logits.iter().map(|&x| (x as f64 - m).exp() / z).collect();
        let ce = -y
            .iter()
            .zip(&p)
            .map(|(&a, &b)| a as f64 * b.max(1e-30).ln())
            .sum::<f64>();
        let brier = y
            .iter()
            .zip(&p)
            .map(|(&a, &b)| (b - a as f64).powi(2))
            .sum();
        let (mut cp, mut cy, mut rps) = (0.0, 0.0, 0.0);
        let k = logits.iter().filter(|&&x| x > -1e3).count();
        for i in 0..logits.len() {
            cp += p[i];
            cy += y[i] as f64;
            rps += (cp - cy).powi(2);
        }
        (ce, brier, rps / (k as f64 - 1.0).max(1.0))
    }

    #[test]
    fn rows_match_a_reference() {
        let _g = test_util::lock();
        let logits = [0.3f32, -1.2, 0.8, -1e4, 2.0, 0.1, -0.5, 1.5];
        let y = [0.2f32, 0.0, 0.8, 0.0, 0.0, 0.5, 0.25, 0.25];
        let lg = arr(&logits, &[2, 4]);
        let tg = arr(&y, &[2, 4]);
        let logp = log_softmax(&lg).unwrap();
        let ce = host_f32(&cross_entropy_rows(&logp, &tg).unwrap()).unwrap();
        let br = host_f32(&brier_rows(&logp, &tg).unwrap()).unwrap();
        let rp = host_f32(&rps_rows(&logp, &tg, &arr(&[3.0, 4.0], &[2])).unwrap()).unwrap();
        for r in 0..2 {
            let (c, b, s) = reference(&logits[r * 4..r * 4 + 4], &y[r * 4..r * 4 + 4]);
            assert!((ce[r] as f64 - c).abs() < 1e-5, "ce {r}");
            assert!((br[r] as f64 - b).abs() < 1e-5, "brier {r}");
            assert!((rp[r] as f64 - s).abs() < 1e-5, "rps {r}");
        }
        // One level off costs less than two (the target is level 0).
        let y1 = arr(&[1.0, 0.0, 0.0], &[1, 3]);
        let near = log_softmax(&arr(&[-20.0, 20.0, -20.0], &[1, 3])).unwrap();
        let far = log_softmax(&arr(&[-20.0, -20.0, 20.0], &[1, 3])).unwrap();
        let k = arr(&[3.0], &[1]);
        let n = host_f32(&rps_rows(&near, &y1, &k).unwrap()).unwrap()[0];
        let f = host_f32(&rps_rows(&far, &y1, &k).unwrap()).unwrap()[0];
        assert!((n - 0.5).abs() < 1e-5 && (f - 1.0).abs() < 1e-5, "{n} {f}");
    }

    #[test]
    fn pair_kl_is_zero_for_the_same_distribution_under_a_permutation() {
        let _g = test_util::lock();
        // Row 1 is row 0 with options reversed: option j of row 0 is option 2 - j of row 1.
        let lg = arr(&[1.0, 0.0, -1.0, -1.0, 0.0, 1.0, -2.0, 0.0, 2.0], &[3, 3]);
        let logp = log_softmax(&lg).unwrap();
        let pairs = |left: i32, right: i32, map: [i32; 3]| Pairs {
            left: Array::from_slice(&[left], &[1]),
            right: Array::from_slice(&[right], &[1]),
            map: Array::from_slice(&map, &[1, 3]),
            mask: Array::from_slice(&[true, true, true], &[1, 3]),
            weight: arr(&[1.0], &[1]),
        };
        let same = host_f32(&pair_kl(&logp, &pairs(0, 1, [2, 1, 0])).unwrap()).unwrap()[0];
        assert!(same.abs() < 1e-6, "{same}");
        let diff = host_f32(&pair_kl(&logp, &pairs(0, 2, [0, 1, 2])).unwrap()).unwrap()[0];
        assert!(diff > 0.1, "{diff}");
        // The total loss adds the weighted terms.
        let tg = arr(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0], &[3, 3]);
        let w = LossWeights {
            brier: 0.0,
            rps: 0.0,
            kl: 2.0,
        };
        let is_score = arr(&[0.0, 0.0, 0.0], &[3]);
        let k = arr(&[3.0, 3.0, 3.0], &[3]);
        let p = pairs(0, 2, [0, 1, 2]);
        let t = host_f32(&total(&lg, &tg, &is_score, &k, Some(&p), &w).unwrap()).unwrap()[0];
        let ce = host_f32(&cross_entropy_rows(&logp, &tg).unwrap().mean(None).unwrap()).unwrap()[0];
        assert!((t - (ce + 2.0 * diff)).abs() < 1e-5);
    }
}
