//! Synthetic series for s6-cast pretraining (docs/SPEC-S6-CAST.md §3), deterministic per seed.
//!
//! Four generator families, after Reverso (arXiv 2602.17634, appendix B) and Simulation Alone
//! (arXiv 2601.00970):
//!
//! - [`Kind::Gp`]: KernelSynth-style Gaussian-process draws. One to five kernels from the
//!   Reverso bank (constant, linear, RBF, rational quadratic, Matérn, periodic) are combined by
//!   sum or product. Each stationary kernel is sampled with random Fourier features instead of a
//!   Cholesky factor (O(n·M) instead of O(n³)), and the combination is made on the sampled
//!   functions: a product of two independent draws is not a draw of the product kernel, but it
//!   has the same kinds of shapes.
//! - [`Kind::Pulse`]: trapezoid pulse trains (spikes or dips) over a baseline, with noise.
//! - [`Kind::Tsi`]: trend (linear, exponential, polynomial, piecewise) + up to three seasonal
//!   waves (sine, sawtooth, square) + noise (Gaussian, Laplace, Student-t) + outliers + level
//!   shifts.
//! - [`Kind::Sarima`]: seasonal ARIMA simulation with stationary AR parts (sampled through their
//!   partial autocorrelations), optional differencing and seasonal differencing.

use std::f64::consts::PI;

/// Periods of the generators (Reverso's set `𝒫`): hourly, daily, weekly, monthly and yearly
/// rhythms at several sampling rates.
pub const PERIODS: [usize; 19] = [
    24, 48, 96, 168, 336, 672, 7, 14, 30, 60, 365, 730, 4, 26, 52, 6, 12, 40, 10,
];

/// A generator family.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Gp,
    Pulse,
    Tsi,
    Sarima,
}

impl Kind {
    pub const ALL: [Kind; 4] = [Kind::Gp, Kind::Pulse, Kind::Tsi, Kind::Sarima];

    /// Share of each family in [`mixed`].
    pub const WEIGHTS: [f64; 4] = [0.3, 0.15, 0.3, 0.25];

    pub fn name(self) -> &'static str {
        match self {
            Kind::Gp => "gp",
            Kind::Pulse => "pulse",
            Kind::Tsi => "tsi",
            Kind::Sarima => "sarima",
        }
    }

    pub fn parse(s: &str) -> Option<Kind> {
        Kind::ALL.into_iter().find(|k| k.name() == s)
    }
}

/// SplitMix64: small, fast and good enough for synthetic data; the whole state is one `u64`.
#[derive(Clone, Debug)]
pub struct SynthRng {
    state: u64,
    spare: Option<f64>,
}

impl SynthRng {
    pub fn new(seed: u64) -> Self {
        Self {
            state: seed ^ 0x5DEE_CE66_D1CE_4E5B,
            spare: None,
        }
    }

    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in `[0, 1)`.
    pub fn f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// Uniform in `[a, b)`.
    pub fn range(&mut self, a: f64, b: f64) -> f64 {
        a + (b - a) * self.f64()
    }

    /// Uniform integer in `[0, n)`, `n > 0`.
    pub fn below(&mut self, n: usize) -> usize {
        ((self.next_u64() as u128 * n as u128) >> 64) as usize
    }

    pub fn coin(&mut self, p: f64) -> bool {
        self.f64() < p
    }

    /// Standard normal (Box–Muller, the second draw kept for the next call).
    pub fn normal(&mut self) -> f64 {
        if let Some(z) = self.spare.take() {
            return z;
        }
        let u1 = self.f64().max(1e-300);
        let u2 = self.f64();
        let r = (-2.0 * u1.ln()).sqrt();
        self.spare = Some(r * (2.0 * PI * u2).sin());
        r * (2.0 * PI * u2).cos()
    }

    /// Gamma(shape, 1) (Marsaglia–Tsang, boosted for `shape < 1`).
    pub fn gamma(&mut self, shape: f64) -> f64 {
        if shape < 1.0 {
            let u = self.f64().max(1e-300);
            return self.gamma(shape + 1.0) * u.powf(1.0 / shape);
        }
        let d = shape - 1.0 / 3.0;
        let c = 1.0 / (9.0 * d).sqrt();
        loop {
            let x = self.normal();
            let v = (1.0 + c * x).powi(3);
            if v <= 0.0 {
                continue;
            }
            let u = self.f64().max(1e-300);
            if u.ln() < 0.5 * x * x + d - d * v + d * v.ln() {
                return d * v;
            }
        }
    }

    /// Chi-square with `k` degrees of freedom.
    pub fn chi2(&mut self, k: f64) -> f64 {
        2.0 * self.gamma(k / 2.0)
    }

    fn pick<T: Copy>(&mut self, items: &[T]) -> T {
        items[self.below(items.len())]
    }
}

/// A series of family `kind`, `len` points long, from `seed`. Always finite.
pub fn series(kind: Kind, seed: u64, len: usize) -> Vec<f32> {
    let mut rng = SynthRng::new(seed ^ (kind as u64).wrapping_mul(0xA24B_AED4_963E_E407));
    let raw = match kind {
        Kind::Gp => gp(&mut rng, len),
        Kind::Pulse => pulse(&mut rng, len),
        Kind::Tsi => tsi(&mut rng, len),
        Kind::Sarima => sarima(&mut rng, len),
    };
    finite(raw)
}

/// A series from a family drawn with [`Kind::WEIGHTS`].
pub fn mixed(seed: u64, len: usize) -> (Kind, Vec<f32>) {
    let mut rng = SynthRng::new(seed);
    let r = rng.f64() * Kind::WEIGHTS.iter().sum::<f64>();
    let mut acc = 0.0;
    let mut kind = Kind::Sarima;
    for (k, w) in Kind::ALL.into_iter().zip(Kind::WEIGHTS) {
        acc += w;
        if r < acc {
            kind = k;
            break;
        }
    }
    (kind, series(kind, rng.next_u64(), len))
}

/// Replaces non-finite values and caps the dynamic range (an exponential trend or an integrated
/// process can overflow): values beyond `1e6` times the median magnitude are clipped.
fn finite(v: Vec<f64>) -> Vec<f32> {
    let mut mags: Vec<f64> = v
        .iter()
        .filter(|x| x.is_finite())
        .map(|x| x.abs())
        .collect();
    if mags.is_empty() {
        return vec![0.0; v.len()];
    }
    mags.sort_by(f64::total_cmp);
    let cap = (mags[mags.len() / 2].max(1e-3) * 1e6).min(1e30);
    v.into_iter()
        .map(|x| {
            if x.is_finite() {
                x.clamp(-cap, cap) as f32
            } else {
                0.0
            }
        })
        .collect()
}

/// The generator periods that fit at least twice in `len` points (at least `[4]`).
fn periods(len: usize) -> Vec<usize> {
    let p: Vec<usize> = PERIODS.iter().copied().filter(|&p| 2 * p <= len).collect();
    if p.is_empty() {
        vec![4.min(len.max(2))]
    } else {
        p
    }
}

// ----------------------------------------------------------------------------- Gaussian process

/// Random Fourier features of a stationary kernel: `f(t) = √(2/M) Σ a_m cos(ω_m t + b_m)`,
/// `a ~ N(0, 1)`, `b ~ U(0, 2π)` and `ω` drawn from the kernel's spectral density.
fn rff(rng: &mut SynthRng, len: usize, mut omega: impl FnMut(&mut SynthRng) -> f64) -> Vec<f64> {
    const M: usize = 48;
    let feats: Vec<(f64, f64, f64)> = (0..M)
        .map(|_| (rng.normal(), omega(rng), rng.range(0.0, 2.0 * PI)))
        .collect();
    let norm = (2.0 / M as f64).sqrt();
    (0..len)
        .map(|t| {
            let t = t as f64;
            norm * feats
                .iter()
                .map(|&(a, w, b)| a * (w * t + b).cos())
                .sum::<f64>()
        })
        .collect()
}

/// One kernel draw from the bank (length scales relative to the series length, as on `[0, 1]`).
fn kernel_draw(rng: &mut SynthRng, len: usize) -> Vec<f64> {
    let n = len as f64;
    let scale = |rng: &mut SynthRng| rng.pick(&[0.1, 1.0, 10.0]) * n;
    match rng.below(6) {
        // constant
        0 => vec![rng.normal(); len],
        // linear: k = σ² + t t'
        1 => {
            let sigma = rng.pick(&[0.0, 1.0, 10.0]);
            let (a, b) = (rng.normal(), rng.normal());
            (0..len).map(|t| sigma * a + b * t as f64 / n).collect()
        }
        // RBF: ω ~ N(0, 1/ℓ²)
        2 => {
            let l = scale(rng);
            rff(rng, len, |r| r.normal() / l)
        }
        // rational quadratic: a scale mixture of RBF, precision ~ Gamma(α, rate α)
        3 => {
            let (l, alpha) = (scale(rng), rng.pick(&[0.1, 1.0, 10.0]));
            rff(rng, len, |r| {
                let prec = (r.gamma(alpha) / alpha).max(1e-6);
                r.normal() * prec.sqrt() / l
            })
        }
        // Matérn ν: ω = z / ℓ · √(2ν / χ²_{2ν}) (a Student-t spectral density)
        4 => {
            let (l, nu) = (scale(rng), rng.pick(&[0.5, 1.5, 2.5]));
            rff(rng, len, |r| {
                r.normal() / l * (2.0 * nu / r.chi2(2.0 * nu).max(1e-9)).sqrt()
            })
        }
        // periodic: a Fourier series of period p with decaying harmonics
        _ => {
            let p = rng.pick(&periods(len)) as f64;
            let decay = rng.range(0.5, 3.0);
            let harmonics = 1 + rng.below(6);
            let coef: Vec<(f64, f64, f64)> = (1..=harmonics)
                .map(|k| {
                    let w = (-(k as f64 - 1.0) / decay).exp().sqrt();
                    (k as f64, w * rng.normal(), w * rng.normal())
                })
                .collect();
            (0..len)
                .map(|t| {
                    let x = 2.0 * PI * t as f64 / p;
                    coef.iter()
                        .map(|&(k, a, b)| a * (k * x).cos() + b * (k * x).sin())
                        .sum()
                })
                .collect()
        }
    }
}

fn gp(rng: &mut SynthRng, len: usize) -> Vec<f64> {
    let n_kernels = 1 + rng.below(5);
    let mut f = kernel_draw(rng, len);
    for _ in 1..n_kernels {
        let g = kernel_draw(rng, len);
        let add = rng.coin(0.5);
        for (x, y) in f.iter_mut().zip(g) {
            *x = if add { *x + y } else { *x * y };
        }
    }
    // mean: a linear trend half of the time, else a constant
    let c = rng.range(-0.1, 0.1);
    let m = if rng.coin(0.5) {
        rng.range(-0.01, 0.01) * len as f64
    } else {
        0.0
    };
    let noise = rng.range(0.0, 0.05);
    let sd = std(&f).max(1e-6);
    (0..len)
        .map(|t| f[t] + c + m * t as f64 / len as f64 + noise * sd * rng.normal())
        .collect()
}

// ----------------------------------------------------------------------------- pulses

fn pulse(rng: &mut SynthRng, len: usize) -> Vec<f64> {
    let baseline = rng.normal();
    let p = if rng.coin(0.7) {
        rng.pick(&periods(len))
    } else {
        (8 + rng.below((len / 4).max(9) - 8)).min(len.max(2))
    };
    let sign = if rng.coin(0.5) { 1.0 } else { -1.0 };
    let amp = sign * rng.range(0.5, 5.0);
    let width = (2 + rng.below((p / 2).max(3) - 2)).min(p);
    let phase = rng.below(p);
    let noise = rng.range(0.0, 0.2) * amp.abs();
    // trapezoid: rise over w/4, flat over w/2, fall over the rest
    let rise = (width / 4).max(1);
    let flat = width / 2;
    let fall = width.saturating_sub(rise + flat).max(1);
    let shape = |k: usize| -> f64 {
        if k < rise {
            (k + 1) as f64 / rise as f64
        } else if k < rise + flat {
            1.0
        } else if k < rise + flat + fall {
            1.0 - (k - rise - flat + 1) as f64 / fall as f64
        } else {
            0.0
        }
    };
    (0..len)
        .map(|t| baseline + amp * shape((t + phase) % p) + noise * rng.normal())
        .collect()
}

// ----------------------------------------------------------------------------- trend + seasonality

fn tsi(rng: &mut SynthRng, len: usize) -> Vec<f64> {
    let n = len as f64;
    let mut y = vec![rng.normal(); len];
    if rng.coin(0.7) {
        match rng.below(4) {
            0 => {
                let slope = rng.normal();
                y.iter_mut()
                    .enumerate()
                    .for_each(|(t, v)| *v += slope * t as f64 / n);
            }
            1 => {
                let r = rng.range(-2.0, 2.0);
                let a = rng.normal();
                y.iter_mut()
                    .enumerate()
                    .for_each(|(t, v)| *v += a * ((r * t as f64 / n).exp() - 1.0));
            }
            2 => {
                let c: Vec<f64> = (0..=1 + rng.below(2)).map(|_| rng.normal()).collect();
                y.iter_mut().enumerate().for_each(|(t, v)| {
                    let x = 2.0 * t as f64 / n - 1.0;
                    *v += c
                        .iter()
                        .enumerate()
                        .map(|(k, c)| c * x.powi(k as i32 + 1))
                        .sum::<f64>();
                });
            }
            _ => {
                let k = 1 + rng.below(3);
                let mut knots: Vec<usize> = (0..k).map(|_| rng.below(len.max(1))).collect();
                knots.sort_unstable();
                let mut slope = rng.normal() / n;
                let mut level = 0.0;
                let mut next = 0;
                for (t, v) in y.iter_mut().enumerate() {
                    while next < knots.len() && knots[next] == t {
                        slope = rng.normal() / n * 2.0;
                        next += 1;
                    }
                    level += slope;
                    *v += level;
                }
            }
        }
    }
    if rng.coin(0.8) {
        let ps = periods(len);
        for _ in 0..1 + rng.below(3) {
            let p = rng.pick(&ps) as f64;
            let amp = rng.range(0.2, 2.0);
            let phase = rng.range(0.0, 1.0);
            let wave = rng.below(3);
            y.iter_mut().enumerate().for_each(|(t, v)| {
                let u = (t as f64 / p + phase).fract();
                *v += amp
                    * match wave {
                        0 => (2.0 * PI * u).sin(),
                        1 => 2.0 * u - 1.0,
                        _ => {
                            if u < 0.5 {
                                1.0
                            } else {
                                -1.0
                            }
                        }
                    };
            });
        }
    }
    let sd = std(&y).max(0.1);
    if rng.coin(0.9) {
        let scale = rng.range(0.01, 0.5) * sd;
        let law = rng.below(3);
        for v in y.iter_mut() {
            *v += scale
                * match law {
                    0 => rng.normal(),
                    1 => {
                        let u = rng.range(-0.5, 0.5);
                        -u.signum() * (1.0 - 2.0 * u.abs()).max(1e-12).ln()
                    }
                    _ => rng.normal() / (rng.chi2(3.0) / 3.0).max(1e-6).sqrt(),
                };
        }
    }
    if rng.coin(0.2) {
        for _ in 0..(len / 200).max(1) {
            let t = rng.below(len.max(1));
            let s = if rng.coin(0.5) { 1.0 } else { -1.0 };
            y[t] += s * rng.range(3.0, 8.0) * sd;
        }
    }
    if rng.coin(0.2) {
        for _ in 0..1 + rng.below(2) {
            let t0 = rng.below(len.max(1));
            let shift = rng.normal() * sd;
            y[t0..].iter_mut().for_each(|v| *v += shift);
        }
    }
    y
}

// ----------------------------------------------------------------------------- SARIMA

/// AR coefficients of a stationary process from partial autocorrelations in `(-1, 1)`
/// (Durbin–Levinson).
fn ar_from_pacf(pacf: &[f64]) -> Vec<f64> {
    let mut phi: Vec<f64> = Vec::new();
    for &k in pacf {
        let prev = phi.clone();
        let m = prev.len();
        phi.push(k);
        for j in 0..m {
            phi[j] = prev[j] - k * prev[m - 1 - j];
        }
    }
    phi
}

/// Product of two lag polynomials given as coefficients of `B^1..` (`1 - Σ c_i B^i` form kept
/// as the plain coefficients `c`): returns the coefficients of `(1 - Σ a B^i)(1 - Σ b B^{s·j})`.
fn seasonal_poly(a: &[f64], b: &[f64], s: usize) -> Vec<f64> {
    let mut full = vec![0.0; a.len() + b.len() * s + 1];
    // (1 - Σ a_i B^i)
    let mut left = vec![0.0; a.len() + 1];
    left[0] = 1.0;
    for (i, &c) in a.iter().enumerate() {
        left[i + 1] = -c;
    }
    let mut right = vec![0.0; b.len() * s + 1];
    right[0] = 1.0;
    for (j, &c) in b.iter().enumerate() {
        right[(j + 1) * s] = -c;
    }
    for (i, &l) in left.iter().enumerate() {
        for (j, &r) in right.iter().enumerate() {
            full[i + j] += l * r;
        }
    }
    full[1..].iter().map(|x| -x).collect()
}

fn sarima(rng: &mut SynthRng, len: usize) -> Vec<f64> {
    let ps = periods(len.saturating_mul(2) / 3);
    let s = rng.pick(&ps).max(2);
    let p = rng.below(4);
    let q = rng.below(3);
    let (sp, sq) = (rng.below(2), rng.below(2));
    let d = usize::from(rng.coin(0.4));
    let sd = usize::from(rng.coin(0.3));
    let pacf: Vec<f64> = (0..p).map(|_| rng.range(-0.9, 0.9)).collect();
    let ar = ar_from_pacf(&pacf);
    let sar: Vec<f64> = (0..sp).map(|_| rng.range(-0.9, 0.9)).collect();
    let ma: Vec<f64> = (0..q).map(|_| rng.range(-0.9, 0.9)).collect();
    let sma: Vec<f64> = (0..sq).map(|_| rng.range(-0.9, 0.9)).collect();
    let phi = seasonal_poly(&ar, &sar, s);
    // MA side: (1 + Σ θ B^i)(1 + Σ Θ B^{s j}) — the same product with the signs flipped.
    let neg = |v: &[f64]| v.iter().map(|x| -x).collect::<Vec<_>>();
    let theta = neg(&seasonal_poly(&neg(&ma), &neg(&sma), s));
    let heavy = rng.coin(0.3);
    let burn = 2 * s + 100;
    let total = len + burn;
    let mut e = vec![0.0; total];
    let mut x = vec![0.0; total];
    for t in 0..total {
        e[t] = if heavy {
            rng.normal() / (rng.chi2(4.0) / 4.0).max(1e-6).sqrt()
        } else {
            rng.normal()
        };
        let mut v = e[t];
        for (i, c) in phi.iter().enumerate() {
            if t > i {
                v += c * x[t - 1 - i];
            }
        }
        for (i, c) in theta.iter().enumerate() {
            if t > i {
                v += c * e[t - 1 - i];
            }
        }
        x[t] = v;
    }
    let mut y: Vec<f64> = x[burn..].to_vec();
    for _ in 0..d {
        let mut acc = 0.0;
        y.iter_mut().for_each(|v| {
            acc += *v;
            *v = acc;
        });
    }
    if sd == 1 {
        for t in s..y.len() {
            y[t] += y[t - s];
        }
    }
    let offset = rng.normal() * 3.0;
    y.iter_mut().for_each(|v| *v += offset);
    y
}

fn std(v: &[f64]) -> f64 {
    if v.len() < 2 {
        return 0.0;
    }
    let m = v.iter().sum::<f64>() / v.len() as f64;
    (v.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / (v.len() - 1) as f64).sqrt()
}

// ----------------------------------------------------------------------------- augmentations

/// Reverso's augmentations of one series, each with probability `p`: downsampling (stride 2 or
/// 4, if the series stays at least `min_len` long), amplitude modulation, sign flip, time
/// reversal and censoring (clipping at a quantile). Mixup needs two series: see [`mixup`].
pub fn augment(rng: &mut SynthRng, mut y: Vec<f32>, p: f64, min_len: usize) -> Vec<f32> {
    if rng.coin(p) {
        let k = if rng.coin(0.5) { 2 } else { 4 };
        if y.len() / k >= min_len {
            y = y.iter().step_by(k).copied().collect();
        }
    }
    if rng.coin(p) {
        let n = y.len().max(1) as f64;
        let (a, f, ph) = (
            rng.range(0.1, 0.5),
            rng.range(0.5, 3.0),
            rng.range(0.0, 2.0 * PI),
        );
        let mean = y.iter().map(|&v| v as f64).sum::<f64>() / n;
        y.iter_mut().enumerate().for_each(|(t, v)| {
            let m = 1.0 + a * (2.0 * PI * f * t as f64 / n + ph).sin();
            *v = (mean + (*v as f64 - mean) * m) as f32;
        });
    }
    if rng.coin(p) {
        y.iter_mut().for_each(|v| *v = -*v);
    }
    if rng.coin(p) {
        y.reverse();
    }
    if rng.coin(p) && y.len() > 4 {
        let mut s = y.clone();
        s.sort_by(f32::total_cmp);
        let q = rng.range(0.85, 0.99);
        if rng.coin(0.5) {
            let hi = s[((s.len() - 1) as f64 * q) as usize];
            y.iter_mut().for_each(|v| *v = v.min(hi));
        } else {
            let lo = s[((s.len() - 1) as f64 * (1.0 - q)) as usize];
            y.iter_mut().for_each(|v| *v = v.max(lo));
        }
    }
    y
}

/// Convex combination `λ a + (1 − λ) b` of two series scaled to unit range (`λ ~ U(0.2, 0.8)`),
/// over their common length.
pub fn mixup(rng: &mut SynthRng, a: &[f32], b: &[f32]) -> Vec<f32> {
    let unit = |v: &[f32]| -> Vec<f32> {
        let (lo, hi) = v
            .iter()
            .fold((f32::INFINITY, f32::NEG_INFINITY), |(l, h), &x| {
                (l.min(x), h.max(x))
            });
        let r = (hi - lo).max(1e-6);
        v.iter().map(|&x| (x - lo) / r).collect()
    };
    let n = a.len().min(b.len());
    let (ua, ub) = (unit(&a[a.len() - n..]), unit(&b[b.len() - n..]));
    let l = rng.range(0.2, 0.8) as f32;
    ua.iter()
        .zip(&ub)
        .map(|(x, y)| l * x + (1.0 - l) * y)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cast::prep::dominant_period;

    #[test]
    fn deterministic_finite_and_sized() {
        for kind in Kind::ALL {
            for seed in 0..40 {
                let a = series(kind, seed, 700);
                assert_eq!(a.len(), 700, "{kind:?}");
                assert!(a.iter().all(|x| x.is_finite()), "{kind:?} seed {seed}");
                assert_eq!(a, series(kind, seed, 700), "{kind:?} not deterministic");
            }
            // different seeds give different series
            assert_ne!(series(kind, 1, 300), series(kind, 2, 300), "{kind:?}");
        }
        let (k, s) = mixed(9, 128);
        assert_eq!(s, series(k, SynthRng::new(9).tap_next(), 128));
    }

    impl SynthRng {
        /// The seed `mixed` hands to `series` after drawing the family.
        fn tap_next(mut self) -> u64 {
            self.f64();
            self.next_u64()
        }
    }

    #[test]
    fn families_are_all_drawn() {
        let mut seen = [0usize; 4];
        for seed in 0..400 {
            let (k, _) = mixed(seed, 64);
            seen[k as usize] += 1;
        }
        // 0.3 / 0.15 / 0.3 / 0.25 of 400, loosely
        assert!(seen.iter().all(|&n| n > 30), "{seen:?}");
    }

    #[test]
    fn pulse_trains_repeat_at_their_period() {
        // A noiseless check is not available from outside, so check the autocorrelation peak:
        // most pulse series have a dominant period in the generator set.
        let mut hits = 0;
        for seed in 0..30 {
            let y = series(Kind::Pulse, seed, 2048);
            if let Some(p) = dominant_period(&y, 2, 1024) {
                if PERIODS.iter().any(|&q| q == p || q * 2 == p || p * 2 == q) || p >= 8 {
                    hits += 1;
                }
            }
        }
        assert!(hits >= 20, "{hits}");
    }

    #[test]
    fn stationary_ar_from_pacf() {
        // AR(1) is the PACF itself; AR(2) from (k1, k2) follows Durbin–Levinson.
        assert_eq!(ar_from_pacf(&[0.5]), vec![0.5]);
        let phi = ar_from_pacf(&[0.5, 0.3]);
        assert!((phi[0] - (0.5 - 0.3 * 0.5)).abs() < 1e-12 && (phi[1] - 0.3).abs() < 1e-12);
        // seasonal product (1 - 0.5B)(1 - 0.4B^4): B^1 0.5, B^4 0.4, B^5 -0.2
        let c = seasonal_poly(&[0.5], &[0.4], 4);
        assert_eq!(c.len(), 5);
        assert!((c[0] - 0.5).abs() < 1e-12 && (c[3] - 0.4).abs() < 1e-12);
        assert!((c[4] + 0.2).abs() < 1e-12);
    }

    #[test]
    fn augmentations_keep_finite_values() {
        let mut rng = SynthRng::new(3);
        for seed in 0..30 {
            let y = series(Kind::Tsi, seed, 512);
            let a = augment(&mut rng, y.clone(), 1.0, 64);
            assert!(!a.is_empty() && a.iter().all(|x| x.is_finite()));
            let m = mixup(&mut rng, &y, &a);
            assert!(m.iter().all(|x| (-1e-6..=1.0 + 1e-6).contains(x)));
        }
    }
}
