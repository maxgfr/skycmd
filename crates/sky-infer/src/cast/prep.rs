//! Shared forecasting pieces (docs/SPEC-S6-CAST.md §2, §6): per-window min-max normalization,
//! patching, the seasonal-naive reference, the GIFT-Eval metrics and a periodogram period
//! detector. Used by training (sky-train), inference ([`super::CastModel`]) and the browser.
//!
//! The computed-periodicity front end of s6-cast v2 (docs/s6-cast2-prereg.md §4.2, §4.3) is
//! here too, in one place for both sides: [`inputs`] gives every model input of a context (the
//! patches, the detected periods, the phase channels and the seasonal draft), and sky-train's
//! data producer and [`super::CastModel`] both call it.

use super::tinycast;

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

/// The model input of a context and its normalization, both taken from the last
/// `patch · n_patches` points only: the points the model reads. Normalizing on a longer history
/// would map them to a narrower range than training ever shows (a trending series whose last
/// 2048 points lie in [80, 100] of a [0, 100] history becomes [0.8, 1] instead of [0, 1]).
pub fn prepare(context: &[f32], patch: usize, n_patches: usize) -> (Norm, Vec<f32>, Vec<f32>) {
    let tail = &context[context.len().saturating_sub(patch * n_patches)..];
    let norm = Norm::fit(tail);
    let (vals, mask) = patches(tail, norm, patch, n_patches);
    (norm, vals, mask)
}

/// Significance level of the period detector (docs/s6-cast2-prereg.md §4.2: α = 0.05).
pub const PERIOD_ALPHA: f64 = 0.05;
/// Phase bins of the draft's binned mean (§4.3 (b): 16 bins of p₁).
pub const DRAFT_BINS: usize = 16;

/// The shape of a model's input: enough to prepare a window for it, on either side.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Layout {
    pub patch: usize,
    pub context_patches: usize,
    pub horizon_patches: usize,
    /// Detected periods `K` (0: no front end, s6-cast v1).
    pub periods: usize,
    /// A seasonal draft per query patch.
    pub draft: bool,
}

impl Layout {
    /// Points read (the detector's window too).
    pub fn context(&self) -> usize {
        self.patch * self.context_patches
    }

    /// Phase channels per patch: sin and cos per period.
    pub fn phase_width(&self) -> usize {
        2 * self.periods
    }

    /// Width of a patch's input row: values, mask, phase channels.
    pub fn patch_width(&self) -> usize {
        2 * self.patch + self.phase_width()
    }

    /// Width of a query's draft row: lag copy, binned mean, phase channels (0 without a draft).
    pub fn draft_width(&self) -> usize {
        if self.draft {
            2 * self.patch + self.phase_width()
        } else {
            0
        }
    }
}

/// Every model input of a context window.
#[derive(Clone, Debug, PartialEq)]
pub struct Inputs {
    pub norm: Norm,
    /// `context()` normalized values (0 where missing) and their mask, as [`prepare`].
    pub vals: Vec<f32>,
    pub mask: Vec<f32>,
    /// The `K` detected periods, strongest first, 0 for an empty slot (empty without a front
    /// end).
    pub periods: Vec<usize>,
    /// `[context_patches][2K]` phase channels.
    pub phase: Vec<f32>,
    /// `[horizon_patches][draft_width]` draft rows (empty without a draft).
    pub draft: Vec<f32>,
}

/// The model inputs of `context` (its last `context()` points): [`prepare`], then with a front
/// end the periods ([`detect_periods`]), the phase channels ([`phase_channels`]) and the draft
/// ([`draft`]). Training and inference call this one function.
pub fn inputs(context: &[f32], l: &Layout) -> Inputs {
    let (norm, vals, mask) = prepare(context, l.patch, l.context_patches);
    if l.periods == 0 {
        return Inputs {
            norm,
            vals,
            mask,
            periods: Vec::new(),
            phase: Vec::new(),
            draft: Vec::new(),
        };
    }
    let periods = detect_periods(context, l.context(), l.periods);
    let phase = phase_channels(&periods, l.patch, l.context_patches);
    let draft = if l.draft {
        draft(&vals, &mask, &periods, l.patch, l.horizon_patches)
    } else {
        Vec::new()
    };
    Inputs {
        norm,
        vals,
        mask,
        periods,
        phase,
        draft,
    }
}

/// The detected periods of a context (§4.2): TinyCast's `significant_periods` (`k` periods,
/// α = [`PERIOD_ALPHA`]) on TinyCast's prepared window of its last `len` points (left-padded
/// with the first value, gaps linearly interpolated, min-max normalized). This is the port's
/// detector input exactly ([`tinycast::prepare_context`], [`tinycast::window_minmax`]).
pub fn detect_periods(context: &[f32], len: usize, k: usize) -> Vec<usize> {
    let w = tinycast::prepare_context(context, len);
    let (x, _, _) = tinycast::window_minmax(&w);
    tinycast::significant_periods(&x, k, PERIOD_ALPHA)
}

/// `sin(2π·t/p_k)` and `cos(2π·t/p_k)` for each period `p_k` into `out` (`2K` values), 0 and 0
/// for an empty slot (`p_k = 0`). The phase is reduced to `t mod p` first and computed in f64.
pub fn phase_at(t: usize, periods: &[usize], out: &mut [f32]) {
    for (k, &p) in periods.iter().enumerate() {
        let (s, c) = if p == 0 {
            (0.0, 0.0)
        } else {
            let a = 2.0 * std::f64::consts::PI * (t % p) as f64 / p as f64;
            (a.sin() as f32, a.cos() as f32)
        };
        out[2 * k] = s;
        out[2 * k + 1] = c;
    }
}

/// The phase channels of the patches (§4.2), `[n_patches][2K]`: patch `j`'s phase at its last
/// point, window position `t_j = (j + 1)·patch − 1` (positions count from the first point of
/// the `n_patches · patch`-point window, left padding included).
pub fn phase_channels(periods: &[usize], patch: usize, n_patches: usize) -> Vec<f32> {
    let w = 2 * periods.len();
    let mut out = vec![0.0; n_patches * w];
    for (j, row) in out.chunks_exact_mut(w.max(1)).enumerate().take(n_patches) {
        phase_at((j + 1) * patch - 1, periods, row);
    }
    out
}

/// The seasonal draft of one pass (§4.3), `[horizon_patches][2·patch + 2K]`, in the window's
/// normalized units, from the prepared window (`vals`, `mask`, `n` points) and its periods.
/// Future step `i` sits at window position `n + i`. Query patch `q` gets:
/// - (a) the lag copy of its `patch` steps: [`seasonal_naive`] of the observed values with
///   season `p₁` (the value one or more periods before, earlier cycles over missing values), or
///   the last observed value when no period passes (`p₁ = 0`, season 1);
/// - (b) TinyCast's phase-binned mean of the observed values ([`DRAFT_BINS`] bins of `p₁`,
///   [`tinycast::phase_binned_mean`]);
/// - the phase channels of its first step, position `n + q·patch`.
pub fn draft(
    vals: &[f32],
    mask: &[f32],
    periods: &[usize],
    patch: usize,
    horizon_patches: usize,
) -> Vec<f32> {
    let n = vals.len();
    let h = patch * horizon_patches;
    let p1 = periods.first().copied().unwrap_or(0);
    let observed: Vec<f32> = vals
        .iter()
        .zip(mask)
        .map(|(&v, &m)| if m > 0.0 { v } else { f32::NAN })
        .collect();
    let lag = seasonal_naive(&observed, p1.max(1), h);
    let binned = tinycast::phase_binned_mean(vals, Some(mask), h, p1, DRAFT_BINS);
    let k2 = 2 * periods.len();
    let w = 2 * patch + k2;
    let mut out = vec![0.0; horizon_patches * w];
    for (q, row) in out.chunks_exact_mut(w).enumerate() {
        row[..patch].copy_from_slice(&lag[q * patch..(q + 1) * patch]);
        row[patch..2 * patch].copy_from_slice(&binned[q * patch..(q + 1) * patch]);
        phase_at(n + q * patch, periods, &mut row[2 * patch..]);
    }
    out
}

/// The input rows of the patches from `j0` on (`[C − j0][patch_width]`): values, mask, then the
/// phase channels when the layout has them.
pub fn patch_rows(inp: &Inputs, l: &Layout, j0: usize) -> Vec<f32> {
    let (p, k2) = (l.patch, l.phase_width());
    let w = l.patch_width();
    let mut u = vec![0.0; (l.context_patches - j0) * w];
    for (r, j) in (j0..l.context_patches).enumerate() {
        let row = &mut u[r * w..(r + 1) * w];
        row[..p].copy_from_slice(&inp.vals[j * p..(j + 1) * p]);
        row[p..2 * p].copy_from_slice(&inp.mask[j * p..(j + 1) * p]);
        if k2 > 0 {
            row[2 * p..].copy_from_slice(&inp.phase[j * k2..(j + 1) * k2]);
        }
    }
    u
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
    fn prepare_normalizes_on_the_points_read() {
        // [0, 99] history, the model reads the last 8 points: [92, 99] maps to [0, 1]
        let h: Vec<f32> = (0..100).map(|t| t as f32).collect();
        let (n, v, m) = prepare(&h, 4, 2);
        assert_eq!(
            n,
            Norm {
                lo: 92.0,
                scale: 7.0
            }
        );
        assert_eq!((v[0], v[7]), (0.0, 1.0));
        assert_eq!(m, vec![1.0; 8]);
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

    fn layout(periods: usize, draft: bool) -> Layout {
        Layout {
            patch: 8,
            context_patches: 40,
            horizon_patches: 2,
            periods,
            draft,
        }
    }

    fn sine(n: usize, p: f32) -> Vec<f32> {
        (0..n)
            .map(|t| 10.0 + 3.0 * (2.0 * std::f32::consts::PI * t as f32 / p).sin())
            .collect()
    }

    #[test]
    fn without_a_front_end_inputs_are_prepare() {
        let h = sine(500, 24.0);
        let (n, v, m) = prepare(&h, 8, 40);
        let i = inputs(&h, &layout(0, false));
        assert_eq!((i.norm, i.vals, i.mask), (n, v, m));
        assert!(i.periods.is_empty() && i.phase.is_empty() && i.draft.is_empty());
        assert_eq!(layout(0, false).patch_width(), 16);
        assert_eq!(layout(4, true).patch_width(), 24);
        assert_eq!(layout(4, true).draft_width(), 24);
        assert_eq!(layout(4, false).draft_width(), 0);
    }

    #[test]
    fn a_known_period_is_found_and_drives_the_phase_and_the_draft() {
        // a clean 20-step cycle over the whole 320-point window
        let h = sine(800, 20.0);
        let l = layout(4, true);
        let i = inputs(&h, &l);
        assert_eq!(i.periods[0], 20, "{:?}", i.periods);
        assert_eq!(i.periods, detect_periods(&h, 320, 4));
        // the detector reads TinyCast's prepared window of the last 320 points
        let (x, _, _) = tinycast::window_minmax(&h[h.len() - 320..]);
        assert_eq!(i.periods, tinycast::significant_periods(&x, 4, 0.05));
        // phase channels: patch j at position 8j + 7
        assert_eq!(i.phase.len(), 40 * 8);
        for j in [0usize, 5, 39] {
            let t = (8 * j + 7) as f64;
            let a = 2.0 * std::f64::consts::PI * t / 20.0;
            assert!((i.phase[j * 8] - a.sin() as f32).abs() < 1e-6);
            assert!((i.phase[j * 8 + 1] - a.cos() as f32).abs() < 1e-6);
        }
        // the draft: the lag copy is the value 20 steps before (normalized), the binned mean of
        // a clean cycle is close to it, the phase is that of the query's first step
        assert_eq!(i.draft.len(), 2 * 24);
        for q in 0..2 {
            let row = &i.draft[q * 24..(q + 1) * 24];
            for s in 0..8 {
                let step = q * 8 + s;
                let src = 320 + step - 20 * (step / 20 + 1);
                assert_eq!(row[s], i.vals[src], "q {q} s {s}");
                assert!(
                    (row[8 + s] - row[s]).abs() < 0.1,
                    "{} vs {}",
                    row[8 + s],
                    row[s]
                );
            }
            let mut ph = [0.0f32; 8];
            phase_at(320 + 8 * q, &i.periods, &mut ph);
            assert_eq!(&row[16..], &ph);
        }
    }

    #[test]
    fn no_period_gives_empty_slots_and_a_last_value_draft() {
        let l = layout(4, true);
        let flat = vec![7.5f32; 300];
        let i = inputs(&flat, &l);
        assert_eq!(i.periods, vec![0; 4]);
        assert!(i.phase.iter().all(|&v| v == 0.0));
        // constant: normalized to 0, the lag copy is the last value, the binned mean the mean
        for q in 0..2 {
            let row = &i.draft[q * 24..(q + 1) * 24];
            assert!(row.iter().all(|&v| v == 0.0), "{row:?}");
        }
        // a trend with no cycle: no period passes, the lag copy repeats the last observation
        let trend: Vec<f32> = (0..320).map(|t| t as f32).collect();
        let i = inputs(&trend, &l);
        if i.periods[0] == 0 {
            assert!(i.draft[..8].iter().all(|&v| v == 1.0));
            assert!(i.draft[24..32].iter().all(|&v| v == 1.0));
        }
    }

    #[test]
    fn a_short_history_is_left_padded_for_the_detector_and_masked_for_the_draft() {
        let l = layout(4, true);
        let mut h = sine(29, 6.0);
        h[3] = f32::NAN;
        let i = inputs(&h, &l);
        // the detector's window: 291 copies of the first value, then the 29 points, gap filled
        let w = tinycast::prepare_context(&h, 320);
        assert_eq!(w.len(), 320);
        assert!(w[..291].iter().all(|&v| v == h[0]));
        assert!(w.iter().all(|v| v.is_finite()));
        assert_eq!(i.periods, detect_periods(&h, 320, 4));
        // the draft reads observed points only: the left padding and the gap never count
        let p1 = i.periods[0];
        let obs: Vec<f32> = i
            .vals
            .iter()
            .zip(&i.mask)
            .map(|(&v, &m)| if m > 0.0 { v } else { f32::NAN })
            .collect();
        let lag = seasonal_naive(&obs, p1.max(1), 16);
        assert_eq!(&i.draft[..8], &lag[..8]);
        assert!(i.draft.iter().all(|v| v.is_finite()));
        let mean_obs = {
            let v: Vec<f64> = obs
                .iter()
                .filter(|v| v.is_finite())
                .map(|&v| v as f64)
                .collect();
            (v.iter().sum::<f64>() / v.len() as f64) as f32
        };
        // without a period every future step is in bin 0, the mean of the observed points
        let binned = tinycast::phase_binned_mean(&i.vals, Some(&i.mask), 16, 0, DRAFT_BINS);
        assert!(binned.iter().all(|&v| v == mean_obs));
        // the patch rows: values, mask, then the phase channels; the left padding dropped
        let j0 = 40 - 4; // 29 points in patches of 8
        let u = patch_rows(&i, &l, j0);
        assert_eq!(u.len(), 4 * 24);
        assert_eq!(&u[..8], &i.vals[j0 * 8..j0 * 8 + 8]);
        assert_eq!(&u[8..16], &i.mask[j0 * 8..j0 * 8 + 8]);
        assert_eq!(&u[16..24], &i.phase[j0 * 8..j0 * 8 + 8]);
    }

    #[test]
    fn the_binned_mean_without_a_mask_is_tinycasts() {
        let x: Vec<f32> = (0..100).map(|t| ((t * 7) % 13) as f32 / 13.0).collect();
        let ones = vec![1.0f32; 100];
        assert_eq!(
            tinycast::phase_binned_mean(&x, None, 30, 12, 16),
            tinycast::phase_binned_mean(&x, Some(&ones), 30, 12, 16)
        );
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
