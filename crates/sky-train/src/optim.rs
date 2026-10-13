//! Optimizers: Muon (port of `mlx.optimizers.Muon`) for the encoder hidden
//! matrices, AdamW for everything else, and the WSD learning-rate schedule.

use std::{collections::HashMap, rc::Rc};

use anyhow::{bail, Result};
use mlx_rs::{error::Exception, ops, Array};
use serde::{Deserialize, Serialize};

use crate::params::{group_of, Group, ParamMap};

type R<T> = std::result::Result<T, Exception>;

/// Newton–Schulz quintic coefficients of Muon.
pub const NS_COEFFS: (f32, f32, f32) = (3.4445, -4.7750, 2.0315);

/// Optimizer hyper-parameters.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct OptConfig {
    /// Peak AdamW learning rate.
    pub adam_lr: f32,
    /// Peak Muon learning rate.
    pub muon_lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    /// Decoupled AdamW weight decay (matrices only).
    pub weight_decay: f32,
    /// Final Muon momentum.
    pub muon_momentum: f32,
    /// Muon momentum at step 0 (linearly warmed to `muon_momentum`).
    pub muon_momentum_start: f32,
    pub muon_momentum_warmup: u64,
    /// L2 term added to the Muon gradient (as in `mlx.optimizers.Muon`).
    pub muon_weight_decay: f32,
    pub ns_steps: usize,
    /// Global gradient-norm clipping (0 disables).
    pub clip_norm: f32,
    /// Muon on the encoder hidden matrices; `false` puts them in AdamW (with weight decay),
    /// as for fine-tuning a pretrained encoder.
    #[serde(default = "default_true")]
    pub muon: bool,
    /// AdamW learning-rate multiplier of every parameter outside `encoder.*` (a freshly
    /// initialized head next to a pretrained encoder). 1 for s1 / s2.
    #[serde(default = "default_one")]
    pub head_lr_mult: f32,
}

fn default_true() -> bool {
    true
}

fn default_one() -> f32 {
    1.0
}

impl Default for OptConfig {
    fn default() -> Self {
        Self {
            adam_lr: 3e-3,
            muon_lr: 0.02,
            beta1: 0.9,
            beta2: 0.95,
            eps: 1e-8,
            weight_decay: 0.01,
            muon_momentum: 0.95,
            muon_momentum_start: 0.85,
            muon_momentum_warmup: 300,
            muon_weight_decay: 0.01,
            ns_steps: 5,
            clip_norm: 1.0,
            muon: true,
            head_lr_mult: 1.0,
        }
    }
}

/// Warmup-stable-decay multiplier at training progress `frac` in `[0, 1]`:
/// linear warmup over the first 2%, constant, then linear decay over the last
/// 20% down to 0.1x.
pub fn wsd(frac: f64) -> f64 {
    let f = frac.clamp(0.0, 1.0);
    if f < 0.02 {
        (f / 0.02).max(0.01)
    } else if f < 0.8 {
        1.0
    } else {
        1.0 - 0.9 * (f - 0.8) / 0.2
    }
}

/// Fine-tuning schedule at training progress `frac` in `[0, 1]`: linear warmup over the first
/// `warmup` fraction, then linear decay to 0 at the end.
pub fn warmup_linear(frac: f64, warmup: f64) -> f64 {
    let f = frac.clamp(0.0, 1.0);
    if warmup > 0.0 && f < warmup {
        (f / warmup).max(1e-3)
    } else {
        ((1.0 - f) / (1.0 - warmup).max(1e-9)).clamp(0.0, 1.0)
    }
}

/// Orthogonalizes a 2-D matrix with `steps` Newton–Schulz iterations (f32).
pub fn newton_schulz(g: &Array, steps: usize) -> R<Array> {
    let (a, b, c) = NS_COEFFS;
    let transpose = g.dim(0) > g.dim(1);
    let mut x = if transpose { g.t() } else { g.clone() };
    let norm = x.square()?.sum(None)?.sqrt()?;
    x = x.divide(norm.add(Array::from_f32(1e-7))?)?;
    for _ in 0..steps {
        let gram = ops::matmul(&x, x.t())?;
        let poly = gram
            .multiply(Array::from_f32(b))?
            .add(ops::matmul(&gram, &gram)?.multiply(Array::from_f32(c))?)?;
        x = x
            .multiply(Array::from_f32(a))?
            .add(ops::matmul(&poly, &x)?)?;
    }
    Ok(if transpose { x.t() } else { x })
}

/// AdamW + Muon over a flat parameter map.
#[derive(Debug)]
pub struct Optimizer {
    pub cfg: OptConfig,
    /// Number of updates applied so far.
    pub step: u64,
    m: HashMap<Rc<str>, Array>,
    v: HashMap<Rc<str>, Array>,
    mom: HashMap<Rc<str>, Array>,
}

impl Optimizer {
    pub fn new(cfg: OptConfig) -> Self {
        Self {
            cfg,
            step: 0,
            m: HashMap::new(),
            v: HashMap::new(),
            mom: HashMap::new(),
        }
    }

    /// Muon momentum at the current step.
    pub fn muon_momentum(&self) -> f32 {
        let c = &self.cfg;
        let t = (self.step as f32 / c.muon_momentum_warmup.max(1) as f32).min(1.0);
        c.muon_momentum_start + (c.muon_momentum - c.muon_momentum_start) * t
    }

    /// Applies one update with both learning rates multiplied by `lr_scale`.
    /// Returns the (pre-clipping) global gradient norm as a lazy scalar.
    pub fn update(&mut self, params: &mut ParamMap, grads: &ParamMap, lr_scale: f32) -> R<Array> {
        let c = self.cfg.clone();
        let mut sq = Array::from_f32(0.0);
        for g in grads.values() {
            sq = sq.add(g.square()?.sum(None)?)?;
        }
        let gnorm = sq.sqrt()?;
        let clip = if c.clip_norm > 0.0 {
            Some(ops::minimum(
                Array::from_f32(1.0),
                Array::from_f32(c.clip_norm).divide(gnorm.add(Array::from_f32(1e-6))?)?,
            )?)
        } else {
            None
        };
        self.step += 1;
        let t = self.step as i32;
        let bc1 = 1.0 - c.beta1.powi(t);
        let bc2 = 1.0 - c.beta2.powi(t);
        let mu = self.muon_momentum();
        let adam_lr = c.adam_lr * lr_scale;
        let muon_lr = c.muon_lr * lr_scale;
        for (name, p) in params.iter_mut() {
            let Some(g) = grads.get(name) else { continue };
            let g = match &clip {
                Some(s) => g.multiply(s)?,
                None => g.clone(),
            };
            let group = match group_of(name, p.ndim()) {
                Group::Muon if !c.muon => Group::AdamDecay,
                g => g,
            };
            let adam_lr = if c.head_lr_mult != 1.0 && !name.starts_with("encoder.") {
                adam_lr * c.head_lr_mult
            } else {
                adam_lr
            };
            match group {
                Group::Muon => {
                    let g = if c.muon_weight_decay != 0.0 {
                        g.add(p.multiply(Array::from_f32(c.muon_weight_decay))?)?
                    } else {
                        g
                    };
                    let buf = match self.mom.get(name) {
                        Some(b) => b.multiply(Array::from_f32(mu))?,
                        None => ops::zeros_like(&g)?,
                    }
                    .add(g.multiply(Array::from_f32(1.0 - mu))?)?;
                    // Nesterov.
                    let upd = g
                        .multiply(Array::from_f32(1.0 - mu))?
                        .add(buf.multiply(Array::from_f32(mu))?)?;
                    let o = newton_schulz(&upd, c.ns_steps)?;
                    let ratio = (p.dim(0) as f32 / p.dim(1) as f32).max(1.0).sqrt();
                    *p = p.subtract(o.multiply(Array::from_f32(muon_lr * ratio))?)?;
                    self.mom.insert(name.clone(), buf);
                }
                group => {
                    let m = match self.m.get(name) {
                        Some(m) => m.multiply(Array::from_f32(c.beta1))?,
                        None => ops::zeros_like(&g)?,
                    }
                    .add(g.multiply(Array::from_f32(1.0 - c.beta1))?)?;
                    let v = match self.v.get(name) {
                        Some(v) => v.multiply(Array::from_f32(c.beta2))?,
                        None => ops::zeros_like(&g)?,
                    }
                    .add(g.square()?.multiply(Array::from_f32(1.0 - c.beta2))?)?;
                    let denom = v
                        .divide(Array::from_f32(bc2))?
                        .sqrt()?
                        .add(Array::from_f32(c.eps))?;
                    let step = m.divide(Array::from_f32(bc1))?.divide(&denom)?;
                    let mut np = p.clone();
                    if group == Group::AdamDecay && c.weight_decay != 0.0 {
                        np = np.multiply(Array::from_f32(1.0 - adam_lr * c.weight_decay))?;
                    }
                    *p = np.subtract(step.multiply(Array::from_f32(adam_lr))?)?;
                    self.m.insert(name.clone(), m);
                    self.v.insert(name.clone(), v);
                }
            }
        }
        Ok(gnorm)
    }

    /// All state arrays (for evaluation and checkpoints), keyed `adam.m.*`, `adam.v.*`, `muon.v.*`.
    pub fn state_arrays(&self) -> ParamMap {
        let mut out = ParamMap::new();
        for (prefix, map) in [
            ("adam.m.", &self.m),
            ("adam.v.", &self.v),
            ("muon.v.", &self.mom),
        ] {
            for (k, v) in map {
                out.insert(format!("{prefix}{k}").into(), v.clone());
            }
        }
        out
    }

    /// Restores state saved by [`Optimizer::state_arrays`].
    pub fn load_state(&mut self, step: u64, arrays: &ParamMap) -> Result<()> {
        self.step = step;
        self.m.clear();
        self.v.clear();
        self.mom.clear();
        for (k, v) in arrays {
            let (map, name) = if let Some(n) = k.strip_prefix("adam.m.") {
                (&mut self.m, n)
            } else if let Some(n) = k.strip_prefix("adam.v.") {
                (&mut self.v, n)
            } else if let Some(n) = k.strip_prefix("muon.v.") {
                (&mut self.mom, n)
            } else {
                bail!("unexpected optimizer state key {k}");
            };
            map.insert(name.into(), v.clone());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_util;
    use mlx_rs::random;

    #[test]
    fn wsd_shape() {
        assert!(wsd(0.0) <= 0.01 + 1e-12);
        assert!((wsd(0.01) - 0.5).abs() < 1e-9);
        assert_eq!(wsd(0.5), 1.0);
        assert_eq!(wsd(0.8), 1.0);
        assert!((wsd(0.9) - 0.55).abs() < 1e-9);
        assert!((wsd(1.0) - 0.1).abs() < 1e-9);
        assert!((wsd(2.0) - 0.1).abs() < 1e-9);
    }

    #[test]
    fn muon_newton_schulz_orthogonalizes() {
        let _g = test_util::lock();
        // A square matrix with singular values spread over [0.1, 1]: five
        // iterations only lift singular values above ~1/500 of the Frobenius
        // norm, so a square Gaussian (whose smallest one is ~0) is not a
        // meaningful input.
        let cpu = || mlx_rs::Device::cpu();
        let (q1, q2) = mlx_rs::with_device(cpu(), || {
            let a =
                random::normal::<f32>(&[64, 64][..], 0.0, 1.0, &random::key(1).unwrap()).unwrap();
            let b =
                random::normal::<f32>(&[64, 64][..], 0.0, 1.0, &random::key(2).unwrap()).unwrap();
            (
                mlx_rs::linalg::qr(&a).unwrap().0,
                mlx_rs::linalg::qr(&b).unwrap().0,
            )
        });
        let s: Vec<f32> = (0..64).map(|i| 0.1 + 0.9 * i as f32 / 63.0).collect();
        let square =
            ops::matmul(q1.multiply(Array::from_slice(&s, &[1, 64])).unwrap(), &q2).unwrap();
        for (rows, cols) in [(64, 64), (192, 64), (64, 256)] {
            let key = random::key(rows as u64 * 7 + cols as u64).unwrap();
            let g = if rows == cols {
                square.clone()
            } else {
                random::normal::<f32>(&[rows, cols][..], 0.0, 1.0, &key).unwrap()
            };
            let o = newton_schulz(&g, 5).unwrap();
            assert_eq!(o.shape(), g.shape());
            let s = mlx_rs::with_device(cpu(), || mlx_rs::linalg::svd(&o).unwrap().1);
            s.eval().unwrap();
            let s = s.as_slice::<f32>();
            let mean = s.iter().sum::<f32>() / s.len() as f32;
            // The quintic Muon iteration maps singular values into roughly
            // [0.7, 1.2] (it trades exact convergence for speed): all of them
            // must be close to 1 and none may collapse.
            for &x in s {
                assert!(
                    (0.6..=1.3).contains(&x),
                    "{rows}x{cols}: singular value {x}"
                );
            }
            assert!(
                (mean - 1.0).abs() < 0.2,
                "{rows}x{cols}: mean singular value {mean}"
            );
        }
    }

    #[test]
    fn warmup_linear_shape() {
        assert!((warmup_linear(0.0, 0.1) - 1e-3).abs() < 1e-12);
        assert!((warmup_linear(0.05, 0.1) - 0.5).abs() < 1e-9);
        assert!((warmup_linear(0.1, 0.1) - 1.0).abs() < 1e-9);
        assert!((warmup_linear(0.55, 0.1) - 0.5).abs() < 1e-9);
        assert_eq!(warmup_linear(1.0, 0.1), 0.0);
        assert_eq!(warmup_linear(0.5, 0.0), 0.5);
    }

    #[test]
    fn no_muon_and_head_multiplier() {
        let _g = test_util::lock();
        // Without Muon an encoder matrix gets a plain AdamW step: its first update moves
        // every entry by about lr (Adam's normalized step), whatever the gradient scale.
        let mut params = ParamMap::new();
        params.insert(
            "encoder.layers.0.attn.Wo.weight".into(),
            Array::from_slice(&[1.0f32, 1.0, 1.0, 1.0], &[2, 2]),
        );
        params.insert("scorer.1.bias".into(), Array::from_slice(&[0.0f32], &[1]));
        let mut g = ParamMap::new();
        g.insert(
            "encoder.layers.0.attn.Wo.weight".into(),
            Array::from_slice(&[1e-3f32, -5.0, 2.0, 0.5], &[2, 2]),
        );
        g.insert("scorer.1.bias".into(), Array::from_slice(&[1.0f32], &[1]));
        let mut o = Optimizer::new(OptConfig {
            adam_lr: 0.01,
            weight_decay: 0.0,
            clip_norm: 0.0,
            muon: false,
            head_lr_mult: 10.0,
            ..Default::default()
        });
        o.update(&mut params, &g, 1.0).unwrap();
        let w = crate::model::host_f32(&params["encoder.layers.0.attn.Wo.weight"]).unwrap();
        let want = [0.99, 1.01, 0.99, 0.99];
        for (a, b) in w.iter().zip(want) {
            assert!((a - b).abs() < 1e-4, "{w:?}");
        }
        let b = crate::model::host_f32(&params["scorer.1.bias"]).unwrap();
        assert!((b[0] + 0.1).abs() < 1e-4, "{b:?}");
        assert!(o.state_arrays().keys().all(|k| !k.starts_with("muon.")));
    }

    #[test]
    fn muon_momentum_warmup() {
        let mut o = Optimizer::new(OptConfig::default());
        assert!((o.muon_momentum() - 0.85).abs() < 1e-6);
        o.step = 150;
        assert!((o.muon_momentum() - 0.90).abs() < 1e-6);
        o.step = 1000;
        assert!((o.muon_momentum() - 0.95).abs() < 1e-6);
    }

    #[test]
    fn adamw_minimizes_quadratic() {
        let _g = test_util::lock();
        let mut params = ParamMap::new();
        params.insert(
            "head.layers.0.linear1.bias".into(),
            Array::from_slice(&[3.0f32, -2.0], &[2]),
        );
        let mut o = Optimizer::new(OptConfig {
            adam_lr: 0.1,
            clip_norm: 0.0,
            ..Default::default()
        });
        for _ in 0..300 {
            let p = &params["head.layers.0.linear1.bias"];
            let mut g = ParamMap::new();
            g.insert(
                "head.layers.0.linear1.bias".into(),
                p.multiply(Array::from_f32(2.0)).unwrap(),
            );
            o.update(&mut params, &g, 1.0).unwrap();
        }
        let p = &params["head.layers.0.linear1.bias"];
        p.eval().unwrap();
        assert!(p.as_slice::<f32>().iter().all(|x| x.abs() < 0.05));
    }
}
