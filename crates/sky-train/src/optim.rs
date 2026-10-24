//! Optimizers: Muon for the encoder hidden matrices, AdamW for everything else,
//! and the WSD learning-rate schedule.
//!
//! Muon follows recipe 2 ([`OPT_RECIPE`]): decoupled weight decay and the
//! `0.2·√max(A, B)` update scale of "Muon is Scalable for LLM Training" (Kimi),
//! and a Newton–Schulz pass in two stages (DeepSeek V4): fast quintic
//! iterations, then polish iterations that converge to 1.

use std::{collections::HashMap, rc::Rc};

use anyhow::{bail, Result};
use mlx_rs::{error::Exception, ops, Array};
use serde::{Deserialize, Serialize};

use crate::params::{group_of, Group, ParamMap};

type R<T> = std::result::Result<T, Exception>;

/// Newton–Schulz quintic coefficients of Muon (fast stage).
pub const NS_COEFFS: (f32, f32, f32) = (3.4445, -4.7750, 2.0315);

/// Polish-stage coefficients: `f(x) = (15x − 10x³ + 3x⁵) / 8` has `f(1) = 1` and
/// `f'(1) = f''(1) = 0`, so it pulls the singular values the fast stage leaves in
/// ~[0.7, 1.2] to 1 (two iterations: within 0.5%).
pub const NS_POLISH: (f32, f32, f32) = (15.0 / 8.0, -10.0 / 8.0, 3.0 / 8.0);

/// Version of the optimizer recipe. Recipe 1 (coupled L2 decay, `√max(1, A/B)`
/// scale, one Newton–Schulz stage) is gone; its checkpoints cannot be resumed.
pub const OPT_RECIPE: u32 = 2;

/// Muon update scale: `MUON_RMS·√max(A, B)` gives the orthogonalized update the
/// RMS of an AdamW update, so Muon shares AdamW's learning rate and decay (Kimi).
pub const MUON_RMS: f32 = 0.2;

/// Optimizer hyper-parameters.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct OptConfig {
    /// Peak AdamW learning rate.
    pub adam_lr: f32,
    /// Peak Muon learning rate (on the AdamW scale, see [`MUON_RMS`]).
    pub muon_lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    /// Decoupled weight decay of the matrices, AdamW and Muon alike.
    pub weight_decay: f32,
    /// Final Muon momentum.
    pub muon_momentum: f32,
    /// Muon momentum at step 0 (linearly warmed to `muon_momentum`).
    pub muon_momentum_start: f32,
    pub muon_momentum_warmup: u64,
    /// Fast Newton–Schulz iterations ([`NS_COEFFS`]).
    pub ns_steps: usize,
    /// Polish Newton–Schulz iterations ([`NS_POLISH`]).
    #[serde(default = "default_polish")]
    pub ns_polish: usize,
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
    /// Recipe that wrote this config ([`OPT_RECIPE`]); a config saved before the
    /// field existed reads as recipe 1.
    #[serde(default = "legacy_recipe")]
    pub recipe: u32,
}

fn default_true() -> bool {
    true
}

fn default_one() -> f32 {
    1.0
}

fn default_polish() -> usize {
    2
}

fn legacy_recipe() -> u32 {
    1
}

impl Default for OptConfig {
    fn default() -> Self {
        Self {
            adam_lr: 3e-3,
            muon_lr: 3e-3,
            beta1: 0.9,
            beta2: 0.95,
            eps: 1e-8,
            weight_decay: 0.01,
            muon_momentum: 0.95,
            muon_momentum_start: 0.85,
            muon_momentum_warmup: 300,
            ns_steps: 5,
            ns_polish: default_polish(),
            clip_norm: 1.0,
            muon: true,
            head_lr_mult: 1.0,
            recipe: OPT_RECIPE,
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

/// Orthogonalizes a 2-D matrix (f32): `fast` Newton–Schulz iterations with
/// [`NS_COEFFS`], then `polish` iterations with [`NS_POLISH`].
pub fn newton_schulz(g: &Array, fast: usize, polish: usize) -> R<Array> {
    let transpose = g.dim(0) > g.dim(1);
    let mut x = if transpose { g.t() } else { g.clone() };
    let norm = x.square()?.sum(None)?.sqrt()?;
    x = x.divide(norm.add(Array::from_f32(1e-7))?)?;
    let stages = std::iter::repeat_n(NS_COEFFS, fast).chain(std::iter::repeat_n(NS_POLISH, polish));
    for (a, b, c) in stages {
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
                    let buf = match self.mom.get(name) {
                        Some(b) => b.multiply(Array::from_f32(mu))?,
                        None => ops::zeros_like(&g)?,
                    }
                    .add(g.multiply(Array::from_f32(1.0 - mu))?)?;
                    // Nesterov.
                    let upd = g
                        .multiply(Array::from_f32(1.0 - mu))?
                        .add(buf.multiply(Array::from_f32(mu))?)?;
                    let o = newton_schulz(&upd, c.ns_steps, c.ns_polish)?;
                    let scale = MUON_RMS * (p.dim(0).max(p.dim(1)) as f32).sqrt();
                    let mut np = p.clone();
                    if c.weight_decay != 0.0 {
                        np = np.multiply(Array::from_f32(1.0 - muon_lr * c.weight_decay))?;
                    }
                    *p = np.subtract(o.multiply(Array::from_f32(muon_lr * scale))?)?;
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

    /// Restores state saved by [`Optimizer::state_arrays`]; refuses a config
    /// written by another optimizer recipe.
    pub fn load_state(&mut self, step: u64, arrays: &ParamMap) -> Result<()> {
        if self.cfg.recipe != OPT_RECIPE {
            bail!(
                "checkpoint written by optimizer recipe {} (this build runs recipe {OPT_RECIPE}): \
                 resume it with the commit that wrote it",
                self.cfg.recipe
            );
        }
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
        // The fast quintic iteration maps singular values into roughly [0.7, 1.2]
        // (it trades exact convergence for speed); two polish iterations pull
        // them to 1. None may collapse.
        for (polish, band) in [(0, 0.6..=1.3), (2, 0.98..=1.02)] {
            for (rows, cols) in [(64, 64), (192, 64), (64, 256)] {
                let key = random::key(rows as u64 * 7 + cols as u64).unwrap();
                let g = if rows == cols {
                    square.clone()
                } else {
                    random::normal::<f32>(&[rows, cols][..], 0.0, 1.0, &key).unwrap()
                };
                let o = newton_schulz(&g, 5, polish).unwrap();
                assert_eq!(o.shape(), g.shape());
                let s = mlx_rs::with_device(cpu(), || mlx_rs::linalg::svd(&o).unwrap().1);
                s.eval().unwrap();
                let s = s.as_slice::<f32>();
                let mean = s.iter().sum::<f32>() / s.len() as f32;
                for &x in s {
                    assert!(
                        band.contains(&x),
                        "{rows}x{cols}, polish {polish}: singular value {x}"
                    );
                }
                assert!(
                    (mean - 1.0).abs() < 0.2,
                    "{rows}x{cols}: mean singular value {mean}"
                );
            }
        }
    }

    #[test]
    fn muon_update_rms_and_decoupled_decay() {
        let _g = test_util::lock();
        // Whatever the shape, one Muon step moves the matrix by an update of RMS
        // `MUON_RMS · lr` (the orthogonalized update has RMS `1/√max(A, B)`), and the
        // decay multiplies the weights by `1 − lr·wd`, apart from the gradient.
        let lr = 0.01;
        for (rows, cols) in [(64, 64), (192, 64), (64, 256)] {
            let name = "blk.0.attn_qkv.weight";
            let key = random::key(rows as u64 + cols as u64).unwrap();
            let w0 = random::normal::<f32>(&[rows, cols][..], 0.0, 1.0, &key).unwrap();
            let g0 = random::normal::<f32>(&[rows, cols][..], 0.0, 1.0, &random::key(3).unwrap())
                .unwrap();
            for wd in [0.0f32, 0.5] {
                let mut params = ParamMap::new();
                params.insert(name.into(), w0.clone());
                let mut grads = ParamMap::new();
                grads.insert(name.into(), g0.clone());
                let mut o = Optimizer::new(OptConfig {
                    muon_lr: lr,
                    weight_decay: wd,
                    clip_norm: 0.0,
                    ..Default::default()
                });
                o.update(&mut params, &grads, 1.0).unwrap();
                let w0d = w0.multiply(Array::from_f32(1.0 - lr * wd)).unwrap();
                let d = w0d.subtract(&params[name]).unwrap();
                let rms = d.square().unwrap().mean(None).unwrap().sqrt().unwrap();
                let rms = rms.item::<f32>();
                assert!(
                    (rms / (MUON_RMS * lr) - 1.0).abs() < 0.03,
                    "{rows}x{cols}, wd {wd}: update RMS {rms}"
                );
            }
        }
    }

    #[test]
    fn load_state_refuses_another_recipe() {
        let old: OptConfig = serde_json::from_value({
            let mut v = serde_json::to_value(OptConfig::default()).unwrap();
            v.as_object_mut().unwrap().remove("recipe");
            v
        })
        .unwrap();
        assert_eq!(old.recipe, 1);
        let err = Optimizer::new(old)
            .load_state(10, &ParamMap::new())
            .unwrap_err();
        assert!(err.to_string().contains("recipe 1"), "{err}");
        assert!(Optimizer::new(OptConfig::default())
            .load_state(10, &ParamMap::new())
            .is_ok());
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
