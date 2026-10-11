//! Decision metrics (accuracy, NLL, Brier, ECE) and temperature fitting.

use serde::Serialize;

/// Model output for one evaluated question.
#[derive(Debug, Clone)]
pub struct Pred {
    /// Raw (uncalibrated) marker logits, one per option.
    pub logits: Vec<f32>,
    pub target: Vec<f32>,
    pub qtype: usize,
    pub src: String,
    pub group: String,
}

/// Softmax of `logits / t`.
pub fn softmax_t(logits: &[f32], t: f32) -> Vec<f64> {
    let t = t.max(1e-4) as f64;
    let m = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64;
    let e: Vec<f64> = logits.iter().map(|&x| ((x as f64 - m) / t).exp()).collect();
    let s: f64 = e.iter().sum();
    e.into_iter().map(|x| x / s).collect()
}

fn argmax<T: PartialOrd + Copy>(v: &[T]) -> usize {
    let mut best = 0;
    for (i, x) in v.iter().enumerate() {
        if *x > v[best] {
            best = i;
        }
    }
    best
}

/// Negative log-likelihood of a soft target under `logits / t`.
pub fn nll(logits: &[f32], target: &[f32], t: f32) -> f64 {
    let p = softmax_t(logits, t);
    -target
        .iter()
        .zip(&p)
        .map(|(&y, &q)| y as f64 * q.max(1e-12).ln())
        .sum::<f64>()
}

/// Aggregated metrics of a set of predictions.
#[derive(Debug, Clone, Default, Serialize, PartialEq)]
pub struct Metrics {
    pub n: usize,
    pub accuracy: f64,
    pub nll: f64,
    pub brier: f64,
    pub ece: f64,
}

/// Expected calibration error with `bins` equal-width bins (`rl_common.ece_score`).
pub fn ece(conf: &[f64], correct: &[f64], bins: usize) -> f64 {
    if conf.is_empty() {
        return f64::NAN;
    }
    let n = conf.len() as f64;
    let mut e = 0.0;
    for b in 0..bins {
        let lo = b as f64 / bins as f64;
        let hi = (b + 1) as f64 / bins as f64;
        let sel: Vec<usize> = (0..conf.len()).filter(|&i| conf[i] > lo && conf[i] <= hi).collect();
        if sel.is_empty() {
            continue;
        }
        let c = sel.iter().map(|&i| conf[i]).sum::<f64>() / sel.len() as f64;
        let a = sel.iter().map(|&i| correct[i]).sum::<f64>() / sel.len() as f64;
        e += sel.len() as f64 / n * (c - a).abs();
    }
    e
}

/// Metrics of `preds`, each calibrated with the temperature returned by `temp`.
pub fn compute<'a>(preds: impl IntoIterator<Item = &'a Pred>, temp: impl Fn(&Pred) -> f32) -> Metrics {
    let (mut n, mut acc, mut nll_sum, mut brier) = (0usize, 0.0, 0.0, 0.0);
    let (mut conf, mut correct) = (Vec::new(), Vec::new());
    for p in preds {
        let t = temp(p);
        let q = softmax_t(&p.logits, t);
        let hit = (argmax(&q) == argmax(&p.target)) as u8 as f64;
        n += 1;
        acc += hit;
        nll_sum += nll(&p.logits, &p.target, t);
        brier += q.iter().zip(&p.target).map(|(a, &b)| (a - b as f64).powi(2)).sum::<f64>();
        conf.push(q[argmax(&q)]);
        correct.push(hit);
    }
    if n == 0 {
        return Metrics::default();
    }
    let d = n as f64;
    Metrics { n, accuracy: acc / d, nll: nll_sum / d, brier: brier / d, ece: ece(&conf, &correct, 15) }
}

/// Temperature minimizing the mean NLL of `preds` (golden-section search on
/// `ln T` over `[0.05, 20]`).
pub fn fit_temperature(preds: &[&Pred]) -> f32 {
    if preds.is_empty() {
        return 1.0;
    }
    let f = |lt: f64| -> f64 {
        let t = lt.exp() as f32;
        preds.iter().map(|p| nll(&p.logits, &p.target, t)).sum::<f64>()
    };
    let (mut a, mut b) = (0.05f64.ln(), 20f64.ln());
    let g = (5f64.sqrt() - 1.0) / 2.0;
    let mut c = b - g * (b - a);
    let mut d = a + g * (b - a);
    let (mut fc, mut fd) = (f(c), f(d));
    for _ in 0..80 {
        if fc < fd {
            b = d;
            d = c;
            fd = fc;
            c = b - g * (b - a);
            fc = f(c);
        } else {
            a = c;
            c = d;
            fc = fd;
            d = a + g * (b - a);
            fd = f(d);
        }
    }
    (((a + b) / 2.0).exp()) as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pred(logits: &[f32], target: &[f32]) -> Pred {
        Pred { logits: logits.to_vec(), target: target.to_vec(), qtype: 0, src: "x".into(), group: "g".into() }
    }

    #[test]
    fn perfect_and_uniform() {
        let ps = [pred(&[10.0, -10.0], &[1.0, 0.0]), pred(&[-10.0, 10.0], &[0.0, 1.0])];
        let m = compute(&ps, |_| 1.0);
        assert_eq!(m.n, 2);
        assert_eq!(m.accuracy, 1.0);
        assert!(m.nll < 1e-6 && m.brier < 1e-6 && m.ece < 1e-6);
        let u = [pred(&[0.0, 0.0], &[1.0, 0.0])];
        let m = compute(&u, |_| 1.0);
        assert!((m.nll - 2f64.ln()).abs() < 1e-9);
        assert!((m.brier - 0.5).abs() < 1e-9);
    }

    #[test]
    fn ece_matches_reference_binning() {
        // conf 0.9 correct, conf 0.9 wrong -> one bin, |0.9 - 0.5| = 0.4.
        assert!((ece(&[0.9, 0.9], &[1.0, 0.0], 15) - 0.4).abs() < 1e-12);
    }

    #[test]
    fn temperature_recovers_overconfidence() {
        // Logits are 3x too sharp for targets drawn at softmax(logits / 3).
        let mut ps = Vec::new();
        for i in 0..200 {
            let z = (i as f32 / 20.0) - 5.0;
            let p1 = 1.0 / (1.0 + (-z / 3.0).exp());
            ps.push(pred(&[0.0, z], &[1.0 - p1, p1]));
        }
        let refs: Vec<&Pred> = ps.iter().collect();
        let t = fit_temperature(&refs);
        assert!((t - 3.0).abs() < 0.01, "{t}");
    }
}
