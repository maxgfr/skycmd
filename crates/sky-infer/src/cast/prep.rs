//! Shared forecasting pieces (docs/SPEC-S6-CAST.md §2, §6): per-window min-max normalization,
//! patching, the seasonal-naive reference, the GIFT-Eval metrics and a periodogram period
//! detector. Used by training (sky-train), inference ([`super::CastModel`]) and the browser.

/// Min-max normalization of a context window: `x̃ = (x − lo) / scale`, from the observed (finite)
/// values only. A constant window maps to 0 (`scale = 1`), an empty one to the identity.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Norm {
    pub lo: f32,
    pub scale: f32,
}

impl Norm {
    pub fn fit(context: &[f32]) -> Self {
        let (lo, hi) = context
            .iter()
            .filter(|x| x.is_finite())
            .fold((f32::INFINITY, f32::NEG_INFINITY), |(l, h), &x| {
                (l.min(x), h.max(x))
            });
        if !lo.is_finite() {
            return Norm {
                lo: 0.0,
                scale: 1.0,
            };
        }
        let r = hi - lo;
        // a constant (or nearly constant) window has no range to scale by
        let scale = if r > 1e-8 * lo.abs().max(1.0) { r } else { 1.0 };
        Norm { lo, scale }
    }

    #[inline]
    pub fn apply(&self, x: f32) -> f32 {
        (x - self.lo) / self.scale
    }

    #[inline]
    pub fn invert(&self, x: f32) -> f32 {
        x * self.scale + self.lo
    }
}

/// The model input of a context: `n_patches · patch` normalized values and their observation
/// mask (1 observed, 0 missing or left padding), right-aligned on the last observation. Missing
/// values are 0 with mask 0. Only the last `n_patches · patch` points are read.
pub fn patches(
    context: &[f32],
    norm: Norm,
    patch: usize,
    n_patches: usize,
) -> (Vec<f32>, Vec<f32>) {
    let n = patch * n_patches;
    let mut vals = vec![0.0; n];
    let mut mask = vec![0.0; n];
    let take = context.len().min(n);
    let src = &context[context.len() - take..];
    for (i, &x) in src.iter().enumerate() {
        let j = n - take + i;
        if x.is_finite() {
            vals[j] = norm.apply(x);
            mask[j] = 1.0;
        }
    }
    (vals, mask)
}

/// Per patch: whether it holds at least one observation (attention keys of empty patches are
/// masked).
pub fn patch_valid(mask: &[f32], patch: usize) -> Vec<bool> {
    mask.chunks(patch)
        .map(|c| c.iter().any(|&m| m > 0.0))
        .collect()
}

/// Seasonal naive forecast: the last observed season repeated (`season = 1`: the last value).
/// Missing history values fall back to the nearest earlier season.
pub fn seasonal_naive(history: &[f32], season: usize, horizon: usize) -> Vec<f32> {
    let m = season.max(1);
    let n = history.len();
    (0..horizon)
        .map(|h| {
            if n == 0 {
                return 0.0;
            }
            // the value one or more seasons before step n + h
            let mut back = m * (h / m + 1);
            while back <= n + h {
                let i = n + h - back;
                if i < n && history[i].is_finite() {
                    return history[i];
                }
                back += m;
            }
            history
                .iter()
                .rev()
                .find(|x| x.is_finite())
                .copied()
                .unwrap_or(0.0)
        })
        .collect()
}

/// Mean absolute error of the in-sample seasonal-naive forecast (the MASE scale): mean of
/// `|y_t − y_{t−m}|` over the history, `None` when it has no pair or the scale is 0.
pub fn mase_scale(history: &[f32], season: usize) -> Option<f64> {
    let m = season.max(1);
    let diffs: Vec<f64> = (m..history.len())
        .filter_map(|t| {
            let (a, b) = (history[t], history[t - m]);
            (a.is_finite() && b.is_finite()).then(|| (a - b).abs() as f64)
        })
        .collect();
    if diffs.is_empty() {
        return None;
    }
    let s = diffs.iter().sum::<f64>() / diffs.len() as f64;
    (s > 0.0).then_some(s)
}

/// MASE of a point forecast (GIFT-Eval: the median), against the in-sample seasonal-naive scale.
pub fn mase(history: &[f32], season: usize, target: &[f32], point: &[f32]) -> Option<f64> {
    let scale = mase_scale(history, season)?;
    let errs: Vec<f64> = target
        .iter()
        .zip(point)
        .filter(|(y, _)| y.is_finite())
        .map(|(y, p)| (y - p).abs() as f64)
        .collect();
    if errs.is_empty() {
        return None;
    }
    Some(errs.iter().sum::<f64>() / errs.len() as f64 / scale)
}

/// Pinball loss of quantile `tau` for target `y` and prediction `q`.
#[inline]
pub fn pinball(y: f32, q: f32, tau: f32) -> f32 {
    let e = y - q;
    if e >= 0.0 {
        tau * e
    } else {
        (tau - 1.0) * e
    }
}

/// GIFT-Eval's weighted quantile loss (its CRPS column): `Σ_q Σ_t 2·pinball / Σ_t |y_t|`, averaged
/// over the quantile levels. `quantiles` is `[horizon][levels]`. Sums are returned separately so
/// a caller can aggregate over series the way GIFT-Eval does (sum of losses over sum of |y|).
pub fn wql_parts(target: &[f32], quantiles: &[Vec<f32>], levels: &[f32]) -> (f64, f64) {
    let mut loss = 0.0f64;
    let mut abs = 0.0f64;
    for (y, q) in target.iter().zip(quantiles) {
        if !y.is_finite() {
            continue;
        }
        abs += y.abs() as f64;
        for (&tau, &v) in levels.iter().zip(q) {
            loss += 2.0 * pinball(*y, v, tau) as f64;
        }
    }
    (loss / levels.len().max(1) as f64, abs)
}

/// The dominant period of `x` in `[min, max]` from its periodogram (direct DFT on the mean-removed
/// finite values; `n ≤ 4096` stays in the low milliseconds). `None` when the series is too short
/// or flat. Missing values count as the mean.
pub fn dominant_period(x: &[f32], min: usize, max: usize) -> Option<usize> {
    let n = x.len();
    if n < 2 * min.max(2) {
        return None;
    }
    let fin: Vec<f64> = x
        .iter()
        .filter(|v| v.is_finite())
        .map(|&v| v as f64)
        .collect();
    if fin.len() < 4 {
        return None;
    }
    let mean = fin.iter().sum::<f64>() / fin.len() as f64;
    let y: Vec<f64> = x
        .iter()
        .map(|&v| if v.is_finite() { v as f64 - mean } else { 0.0 })
        .collect();
    let power = |k: usize| -> f64 {
        let w = 2.0 * std::f64::consts::PI * k as f64 / n as f64;
        let (mut re, mut im) = (0.0, 0.0);
        // rotate a unit phasor instead of calling cos/sin per sample
        let (c, s) = (w.cos(), w.sin());
        let (mut pr, mut pi) = (1.0f64, 0.0f64);
        for &v in &y {
            re += v * pr;
            im -= v * pi;
            let t = pr * c - pi * s;
            pi = pr * s + pi * c;
            pr = t;
        }
        re * re + im * im
    };
    // frequencies k whose period n / k lies in [min, max]
    let k_lo = (n / max.max(1)).max(1);
    let k_hi = (n / min.max(1)).min(n / 2);
    let mut best: Option<(usize, f64)> = None;
    let mut total = 0.0;
    for k in k_lo..=k_hi {
        let p = power(k);
        total += p;
        if best.is_none_or(|(_, b)| p > b) {
            best = Some((k, p));
        }
    }
    let (k, p) = best?;
    // a flat spectrum has no period: the peak must stand out of the band's mean power
    let mean_p = total / (k_hi + 1 - k_lo) as f64;
    if p <= 4.0 * mean_p || p <= 0.0 {
        return None;
    }
    Some(((n as f64 / k as f64).round() as usize).clamp(min, max))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn norm_round_trip_and_constants() {
        let n = Norm::fit(&[2.0, f32::NAN, 6.0, 4.0]);
        assert_eq!((n.lo, n.scale), (2.0, 4.0));
        assert_eq!(n.apply(6.0), 1.0);
        assert!((n.invert(n.apply(3.3)) - 3.3).abs() < 1e-6);
        let c = Norm::fit(&[5.0; 10]);
        assert_eq!(c.apply(5.0), 0.0);
        assert_eq!(
            Norm::fit(&[f32::NAN]),
            Norm {
                lo: 0.0,
                scale: 1.0
            }
        );
    }

    #[test]
    fn patches_are_right_aligned_and_masked() {
        let ctx = [1.0, 2.0, f32::NAN, 4.0, 5.0];
        let n = Norm::fit(&ctx);
        let (v, m) = patches(&ctx, n, 2, 4);
        assert_eq!(m, [0., 0., 0., 1., 1., 0., 1., 1.]);
        assert_eq!(v[3], 0.0); // 1.0 is the minimum
        assert_eq!(v[7], 1.0); // 5.0 is the maximum
        assert_eq!(patch_valid(&m, 2), [false, true, true, true]);
        // a long context keeps its last points only
        let long: Vec<f32> = (0..10).map(|x| x as f32).collect();
        let (v, m) = patches(
            &long,
            Norm {
                lo: 0.0,
                scale: 1.0,
            },
            2,
            2,
        );
        assert_eq!((v, m), (vec![6., 7., 8., 9.], vec![1.; 4]));
    }

    #[test]
    fn seasonal_naive_repeats_the_last_season() {
        let h = [1.0, 2.0, 3.0, 10.0, 20.0, 30.0];
        assert_eq!(seasonal_naive(&h, 3, 5), [10.0, 20.0, 30.0, 10.0, 20.0]);
        assert_eq!(seasonal_naive(&h, 1, 2), [30.0, 30.0]);
        let gap = [1.0, 2.0, 3.0, f32::NAN, 20.0, 30.0];
        assert_eq!(seasonal_naive(&gap, 3, 1), [1.0]);
    }

    #[test]
    fn mase_of_the_seasonal_naive_on_a_periodic_series_is_zero() {
        let y: Vec<f32> = (0..40).map(|t| (t % 4) as f32 + t as f32 * 0.5).collect();
        let (hist, tgt) = y.split_at(32);
        // scale: |y_t − y_{t−4}| = 2 everywhere
        assert_eq!(mase_scale(hist, 4), Some(2.0));
        let sn = seasonal_naive(hist, 4, 8);
        // the trend adds 2 per season: errors 2, 2, 2, 2, 4, 4, 4, 4 → MASE 1.5
        let m = mase(hist, 4, tgt, &sn).unwrap();
        assert!((m - 1.5).abs() < 1e-9, "{m}");
        assert!(mase(&[1.0, 1.0, 1.0], 1, &[1.0], &[1.0]).is_none());
    }

    #[test]
    fn wql_of_a_perfect_forecast_is_zero_and_pinball_is_asymmetric() {
        assert_eq!(pinball(1.0, 0.0, 0.9), 0.9);
        assert!((pinball(0.0, 1.0, 0.9) - 0.1).abs() < 1e-7);
        let levels = [0.1, 0.5, 0.9];
        let q = vec![vec![2.0; 3], vec![4.0; 3]];
        assert_eq!(wql_parts(&[2.0, 4.0], &q, &levels), (0.0, 6.0));
        // median off by 1 on both steps: 2 · 0.5 · 1 · 2 steps / 1 level counted... per level mean
        let (l, a) = wql_parts(&[2.0, 4.0], &[vec![3.0], vec![5.0]], &[0.5]);
        assert_eq!((l, a), (2.0, 6.0));
    }

    #[test]
    fn periodogram_finds_the_period() {
        let y: Vec<f32> = (0..1024)
            .map(|t| (2.0 * std::f32::consts::PI * t as f32 / 24.0).sin() + 0.01 * (t % 7) as f32)
            .collect();
        assert_eq!(dominant_period(&y, 2, 512), Some(24));
        let flat = vec![3.0f32; 256];
        assert_eq!(dominant_period(&flat, 2, 128), None);
    }
}
