//! A training session (config, parameters, optimizer, RNG and progress) with
//! one-step training functions and atomic checkpoints.
//!
//! A checkpoint directory holds `model.safetensors`, `optimizer.safetensors`,
//! `state.json` (step, elapsed time, RNG, data cursor, configs) and any
//! phase-specific files (tokenizer, `rl_agent_config.json`).

use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    rc::Rc,
};

use anyhow::{Context, Result};
use mlx_rs::{error::Exception, transforms::keyed_value_and_grad, Array};
use serde::{Deserialize, Serialize};

use crate::{
    config::ModelConfig,
    mlm::Cursor,
    model::{self, DecisionBatch, MlmBatch},
    optim::{OptConfig, Optimizer},
    params::{self, ParamMap},
    rng::Rng,
};

type R<T> = std::result::Result<T, Exception>;

/// What drives a run's schedule (WSD learning rate, QAT switch, stop).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Progress {
    /// Wall-clock budget: `frac = elapsed / budget` (the default, `--minutes`).
    Time { budget_secs: f64 },
    /// Step budget: `frac = step / steps` (`--steps`), the same on every machine.
    Steps { steps: u64 },
}

impl Progress {
    /// The schedule chosen by `--minutes` / `--steps`; exactly one must be given.
    pub fn from_flags(minutes: Option<f64>, steps: Option<u64>) -> Result<Self> {
        match (minutes, steps) {
            (Some(_), Some(_)) => {
                anyhow::bail!("--minutes and --steps are exclusive: give a time or a step budget")
            }
            (None, None) => anyhow::bail!("give a budget with --minutes or --steps"),
            (Some(m), None) => Ok(Self::Time {
                budget_secs: m * 60.0,
            }),
            (None, Some(0)) => anyhow::bail!("--steps must be above 0"),
            (None, Some(n)) => Ok(Self::Steps { steps: n }),
        }
    }

    /// Position in the schedule, from 0 to 1.
    pub fn frac(&self, step: u64, elapsed_secs: f64) -> f64 {
        match *self {
            Self::Time { budget_secs } => elapsed_secs / budget_secs,
            Self::Steps { steps } => step as f64 / steps as f64,
        }
    }

    /// True once the whole budget is spent.
    pub fn done(&self, step: u64, elapsed_secs: f64) -> bool {
        match *self {
            Self::Time { budget_secs } => elapsed_secs >= budget_secs,
            Self::Steps { steps } => step >= steps,
        }
    }

    /// The step budget, if any.
    pub fn steps(&self) -> Option<u64> {
        match *self {
            Self::Time { .. } => None,
            Self::Steps { steps } => Some(steps),
        }
    }
}

/// True when ternary QAT should be running at `frac` (it covers the last `qat_frac`).
pub fn qat_due(frac: f64, qat_frac: f64) -> bool {
    qat_frac > 0.0 && frac >= 1.0 - qat_frac.clamp(0.0, 1.0)
}

/// The step at which a run that starts at `start` stops with `--stop-after k`.
pub fn stop_step(start: u64, stop_after: Option<u64>) -> Result<Option<u64>> {
    match stop_after {
        None => Ok(None),
        Some(0) => anyhow::bail!("--stop-after must be above 0"),
        Some(k) => start
            .checked_add(k)
            .map(Some)
            .context("--stop-after is too large"),
    }
}

/// True while the run should take another step.
pub fn keep_going(progress: &Progress, stop_at: Option<u64>, step: u64, elapsed_secs: f64) -> bool {
    !progress.done(step, elapsed_secs) && stop_at.is_none_or(|s| step < s)
}

/// Replaces every parameter with the one of the same name in `src` (`--init-params`).
/// Both sides must hold the same tensor names, shapes and dtypes; the error names the
/// first offending tensor in name order.
pub fn replace_params(dst: &mut ParamMap, src: &ParamMap) -> Result<()> {
    for k in sorted_names(dst) {
        let Some(s) = src.get(&k) else {
            anyhow::bail!("--init-params has no tensor {k}");
        };
        let d = &dst[&k];
        if s.shape() != d.shape() || s.dtype() != d.dtype() {
            anyhow::bail!(
                "--init-params tensor {k} is {:?} {:?}, the model expects {:?} {:?}",
                s.dtype(),
                s.shape(),
                d.dtype(),
                d.shape()
            );
        }
    }
    if let Some(extra) = sorted_names(src).into_iter().find(|k| !dst.contains_key(k)) {
        anyhow::bail!("--init-params has tensor {extra}, which the model does not have");
    }
    for (k, v) in dst.iter_mut() {
        *v = src[k].clone();
    }
    Ok(())
}

/// Serializable training progress.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TrainState {
    /// `pretrain` or `decide`.
    pub phase: String,
    pub step: u64,
    /// Training time consumed so far (the budget counts across resumes).
    pub elapsed_secs: f64,
    pub budget_secs: f64,
    pub seed: u64,
    pub rng: Rng,
    /// MLM stream position.
    #[serde(default)]
    pub cursor: Cursor,
    pub config: ModelConfig,
    pub opt: OptConfig,
    /// Phase-specific settings (batch size, data paths, source weights, ...).
    #[serde(default)]
    pub extra: serde_json::Value,
}

/// Loss and throughput of one step.
#[derive(Debug, Clone, Copy)]
pub struct StepStats {
    pub loss: f32,
    pub grad_norm: f32,
    pub tokens: usize,
}

/// Everything needed to continue training.
pub struct Session {
    pub cfg: ModelConfig,
    pub params: ParamMap,
    pub opt: Optimizer,
    pub state: TrainState,
    /// Use `mlx` graph compilation for the loss + gradient computation.
    pub compile: bool,
    mlm_fn: Option<CompiledGrad>,
    dec_fn: Option<CompiledGrad>,
}

/// A compiled `[params..., batch...] -> [loss, grads...]` function.
pub(crate) type CompiledFn = Box<dyn for<'a> FnMut(&'a [Array]) -> R<Vec<Array>>>;

/// Compiled loss + gradient over a fixed parameter order.
pub(crate) struct CompiledGrad {
    names: Vec<Rc<str>>,
    f: CompiledFn,
}

impl CompiledGrad {
    /// Compiles `loss(params, batch_arrays)`; parameters are passed in sorted name order.
    pub(crate) fn new(
        names: Vec<Rc<str>>,
        loss: impl Fn(&ParamMap, &[Array]) -> R<Array> + 'static,
    ) -> Self {
        let keys = names.clone();
        let step = move |args: &[Array]| -> R<Vec<Array>> {
            let (pv, bv) = args.split_at(keys.len());
            let params: ParamMap = keys.iter().cloned().zip(pv.iter().cloned()).collect();
            let f = |p: HashMap<Rc<str>, Array>, b: &[Array]| -> R<Vec<Array>> {
                Ok(vec![loss(&p, b)?])
            };
            let (mut values, grads) = keyed_value_and_grad(f)(params, bv)?;
            for k in &keys {
                values.push(grads[k].clone());
            }
            Ok(values)
        };
        Self {
            names,
            f: Box::new(mlx_rs::transforms::compile::compile(step, false)),
        }
    }

    /// Returns the loss and the gradients keyed by parameter name.
    pub(crate) fn call(
        &mut self,
        params: &ParamMap,
        batch: Vec<Array>,
    ) -> Result<(Array, ParamMap)> {
        let mut args: Vec<Array> = self.names.iter().map(|k| params[k].clone()).collect();
        args.extend(batch);
        let mut out = (self.f)(&args)?.into_iter();
        let loss = out.next().expect("loss output");
        let grads = self.names.iter().cloned().zip(out).collect();
        Ok((loss, grads))
    }
}

fn all_valid(ids: &Array, key_valid: &Option<Array>) -> Array {
    match key_valid {
        Some(k) => k.clone(),
        None => Array::from_slice(&vec![true; ids.size()], ids.shape()),
    }
}

pub(crate) fn sorted_names(params: &ParamMap) -> Vec<Rc<str>> {
    let mut names: Vec<Rc<str>> = params.keys().cloned().collect();
    names.sort();
    names
}

/// Applies one optimizer update from precomputed gradients and evaluates the new state.
pub(crate) fn apply_update(
    params: &mut ParamMap,
    opt: &mut Optimizer,
    loss: Array,
    grads: &ParamMap,
    lr_scale: f32,
) -> Result<(f32, f32)> {
    // A non-finite loss leaves the parameters and the optimizer state untouched, so the
    // caller can skip the batch instead of training on NaN gradients.
    mlx_rs::transforms::eval([&loss])?;
    let l = loss.item_exact::<f32>();
    if !l.is_finite() {
        return Ok((l, f32::NAN));
    }
    let gnorm = opt.update(params, grads, lr_scale)?;
    let state = opt.state_arrays();
    mlx_rs::transforms::eval(params.values().chain(state.values()).chain([&loss, &gnorm]))?;
    Ok((loss.item_exact::<f32>(), gnorm.item_exact::<f32>()))
}

/// Computes the loss and gradients of `loss` with respect to every parameter,
/// applies one optimizer update and evaluates the new state.
pub fn train_step<B>(
    params: &mut ParamMap,
    opt: &mut Optimizer,
    batch: &B,
    lr_scale: f32,
    loss: impl Fn(&ParamMap, &B) -> R<Array>,
) -> Result<(f32, f32)> {
    let f = |p: HashMap<Rc<str>, Array>, b: &B| -> R<Vec<Array>> { Ok(vec![loss(&p, b)?]) };
    let mut vg = keyed_value_and_grad(f);
    let (values, grads) = vg(params.clone(), batch)?;
    let loss = values.into_iter().next().expect("loss output");
    apply_update(params, opt, loss, &grads, lr_scale)
}

impl Session {
    pub fn new(
        cfg: ModelConfig,
        params: ParamMap,
        opt: OptConfig,
        phase: &str,
        seed: u64,
        budget_secs: f64,
    ) -> Self {
        let state = TrainState {
            phase: phase.into(),
            step: 0,
            elapsed_secs: 0.0,
            budget_secs,
            seed,
            rng: Rng::new(seed),
            cursor: Cursor::default(),
            config: cfg.clone(),
            opt: opt.clone(),
            extra: serde_json::Value::Null,
        };
        Self {
            cfg,
            params,
            opt: Optimizer::new(opt),
            state,
            compile: true,
            mlm_fn: None,
            dec_fn: None,
        }
    }

    /// One MLM step at learning-rate multiplier `lr_scale`.
    pub fn mlm_step(&mut self, batch: &MlmBatch, lr_scale: f32) -> Result<StepStats> {
        let (loss, grad_norm) = if self.compile {
            if self.mlm_fn.is_none() {
                let cfg = self.cfg.clone();
                self.mlm_fn = Some(CompiledGrad::new(
                    sorted_names(&self.params),
                    move |p, b| {
                        let batch = MlmBatch {
                            ids: b[0].clone(),
                            key_valid: Some(b[1].clone()),
                            positions: b[2].clone(),
                            labels: b[3].clone(),
                            weights: b[4].clone(),
                            n_tokens: 0,
                        };
                        model::mlm_loss(p, &cfg, &batch)
                    },
                ));
            }
            let args = vec![
                batch.ids.clone(),
                all_valid(&batch.ids, &batch.key_valid),
                batch.positions.clone(),
                batch.labels.clone(),
                batch.weights.clone(),
            ];
            let (loss, grads) = self
                .mlm_fn
                .as_mut()
                .expect("compiled")
                .call(&self.params, args)?;
            apply_update(&mut self.params, &mut self.opt, loss, &grads, lr_scale)?
        } else {
            let cfg = &self.cfg;
            train_step(&mut self.params, &mut self.opt, batch, lr_scale, |p, b| {
                model::mlm_loss(p, cfg, b)
            })?
        };
        self.state.step += 1;
        Ok(StepStats {
            loss,
            grad_norm,
            tokens: batch.n_tokens,
        })
    }

    /// One decision step at learning-rate multiplier `lr_scale`.
    pub fn decision_step(&mut self, batch: &DecisionBatch, lr_scale: f32) -> Result<StepStats> {
        let (loss, grad_norm) = if self.compile {
            if self.dec_fn.is_none() {
                let cfg = self.cfg.clone();
                self.dec_fn = Some(CompiledGrad::new(
                    sorted_names(&self.params),
                    move |p, b| {
                        let batch = DecisionBatch {
                            ids: b[0].clone(),
                            key_valid: Some(b[1].clone()),
                            qtype: b[2].clone(),
                            marker_pos: b[3].clone(),
                            marker_mask: b[4].clone(),
                            target: b[5].clone(),
                            n_tokens: 0,
                        };
                        model::decision_loss(p, &cfg, &batch)
                    },
                ));
            }
            let args = vec![
                batch.ids.clone(),
                all_valid(&batch.ids, &batch.key_valid),
                batch.qtype.clone(),
                batch.marker_pos.clone(),
                batch.marker_mask.clone(),
                batch.target.clone(),
            ];
            let (loss, grads) = self
                .dec_fn
                .as_mut()
                .expect("compiled")
                .call(&self.params, args)?;
            apply_update(&mut self.params, &mut self.opt, loss, &grads, lr_scale)?
        } else {
            let cfg = &self.cfg;
            train_step(&mut self.params, &mut self.opt, batch, lr_scale, |p, b| {
                model::decision_loss(p, cfg, b)
            })?
        };
        self.state.step += 1;
        Ok(StepStats {
            loss,
            grad_norm,
            tokens: batch.n_tokens,
        })
    }

    /// Writes a checkpoint atomically into `dir` (via `dir.tmp` + rename).
    /// `files` are extra (name, bytes) files stored alongside.
    pub fn save(&self, dir: &Path, files: &[(&str, Vec<u8>)]) -> Result<()> {
        let tmp = PathBuf::from(format!("{}.tmp", dir.display()));
        if tmp.exists() {
            std::fs::remove_dir_all(&tmp)?;
        }
        std::fs::create_dir_all(&tmp)?;
        params::save(&tmp.join("model.safetensors"), &self.params, None)?;
        params::save(
            &tmp.join("optimizer.safetensors"),
            &self.opt.state_arrays(),
            None,
        )?;
        let mut state = self.state.clone();
        state.opt = self.opt.cfg.clone();
        let mut json = serde_json::to_value(&state)?;
        json["optimizer_step"] = self.opt.step.into();
        std::fs::write(tmp.join("state.json"), serde_json::to_string_pretty(&json)?)?;
        std::fs::write(
            tmp.join("config.json"),
            serde_json::to_string_pretty(&self.cfg)?,
        )?;
        for (name, bytes) in files {
            std::fs::write(tmp.join(name), bytes)?;
        }
        if dir.exists() {
            std::fs::remove_dir_all(dir)?;
        }
        std::fs::rename(&tmp, dir)
            .with_context(|| format!("renaming checkpoint to {}", dir.display()))?;
        Ok(())
    }

    /// Restores a session saved by [`Session::save`].
    pub fn load(dir: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(dir.join("state.json"))
            .with_context(|| format!("reading {}/state.json", dir.display()))?;
        let json: serde_json::Value = serde_json::from_str(&text)?;
        let opt_step = json["optimizer_step"].as_u64().unwrap_or(0);
        let state: TrainState = serde_json::from_value(json)?;
        let params = params::load(&dir.join("model.safetensors"))?;
        let opt_arrays = params::load(&dir.join("optimizer.safetensors"))?;
        let mut opt = Optimizer::new(state.opt.clone());
        opt.load_state(opt_step, &opt_arrays)?;
        Ok(Self {
            cfg: state.config.clone(),
            params,
            opt,
            state,
            compile: true,
            mlm_fn: None,
            dec_fn: None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        decision::{sample_batch, DecisionData, Item},
        params::Parts,
        sequence::Pieces,
        test_util,
    };

    pub(super) fn synthetic_data(rng: &mut Rng, vocab: usize, n: usize) -> DecisionData {
        let tokens = |rng: &mut Rng, len: usize| -> Vec<u32> {
            (0..len).map(|_| 5 + rng.below(vocab - 5) as u32).collect()
        };
        let items = (0..n)
            .map(|i| {
                let k = 2 + i % 3;
                let mut target = vec![0.0; k];
                let state_len = 5 + rng.below(10);
                target[rng.below(k)] = 1.0;
                Item {
                    pieces: Pieces {
                        head: tokens(rng, 6),
                        opts: (0..k).map(|_| tokens(rng, 3)).collect(),
                        state: tokens(rng, state_len),
                    },
                    qtype: i % 3,
                    target,
                    src: "syn".into(),
                    group: 0,
                    len: 0,
                }
            })
            .collect();
        DecisionData {
            groups: vec!["syn".into()],
            train: vec![items],
            ..Default::default()
        }
    }

    /// Central finite difference of `f` along `dir` for parameter `name`.
    fn directional_fd(
        p: &ParamMap,
        name: &str,
        dir: &[f32],
        eps: f32,
        f: &dyn Fn(&ParamMap) -> f64,
    ) -> f64 {
        let base = model::host_f32(&p[name]).unwrap();
        let shape = p[name].shape().to_vec();
        let at = |sign: f32| {
            let w: Vec<f32> = base
                .iter()
                .zip(dir)
                .map(|(a, d)| a + sign * eps * d)
                .collect();
            let mut q = p.clone();
            q.insert(name.into(), Array::from_slice(&w, &shape));
            f(&q)
        };
        (at(1.0) - at(-1.0)) / (2.0 * eps as f64)
    }

    #[test]
    fn non_finite_loss_leaves_the_parameters_untouched() {
        let _g = test_util::lock();
        mlx_rs::with_device(mlx_rs::Device::cpu(), || {
            let mut p = ParamMap::new();
            p.insert("w".into(), Array::from_slice(&[1.0f32, 2.0], &[2]));
            let mut g = ParamMap::new();
            g.insert("w".into(), Array::from_slice(&[0.5f32, 0.5], &[2]));
            let mut opt = Optimizer::new(OptConfig::default());
            let (l, n) =
                apply_update(&mut p, &mut opt, Array::from_f32(f32::NAN), &g, 1.0).unwrap();
            assert!(l.is_nan() && n.is_nan());
            assert_eq!(model::host_f32(&p["w"]).unwrap(), vec![1.0, 2.0]);
            let (l, _) = apply_update(&mut p, &mut opt, Array::from_f32(0.3), &g, 1.0).unwrap();
            assert!((l - 0.3).abs() < 1e-6);
            assert_ne!(model::host_f32(&p["w"]).unwrap(), vec![1.0, 2.0]);
        });
    }

    #[test]
    fn numeric_gradient_check() {
        let _g = test_util::lock();
        // f32 on the CPU device (deterministic). The objective is a random
        // weighting of the marker logits, which reaches every parameter of the
        // encoder, the head and the scorer. (The CE loss itself is checked
        // against its logits below: at initialization every [MASK] marker has
        // almost the same state, so the CE gradient of the head parameters is
        // a tiny residual that ReLU kinks drown in finite differences.)
        mlx_rs::with_device(mlx_rs::Device::cpu(), || {
            let cfg = ModelConfig::tiny(40);
            let p = params::init(&cfg, Parts::DECISION, 11).unwrap();
            let mut rng = Rng::new(5);
            let data = synthetic_data(&mut rng, 40, 8);
            let batch = sample_batch(&data, &[1.0], 4, &mut rng, &cfg.rl_agent);
            let wts = mlx_rs::random::normal::<f32>(
                batch.marker_pos.shape(),
                0.0,
                1.0,
                &mlx_rs::random::key(3).unwrap(),
            )
            .unwrap()
            .multiply(batch.marker_mask.as_dtype(mlx_rs::Dtype::Float32).unwrap())
            .unwrap();
            let objective = |p: &ParamMap| -> R<Array> {
                model::decision_forward(p, &cfg, &batch)?
                    .multiply(&wts)?
                    .sum(None)
            };
            let value = |p: &ParamMap| -> f64 { objective(p).unwrap().item_exact::<f32>() as f64 };
            let f =
                |p: HashMap<Rc<str>, Array>, _: ()| -> R<Vec<Array>> { Ok(vec![objective(&p)?]) };
            let (_, grads) = keyed_value_and_grad(f)(p.clone(), ()).unwrap();
            for name in [
                "encoder.embeddings.tok_embeddings.weight",
                "encoder.embeddings.norm.weight",
                "encoder.layers.0.attn.Wqkv.weight",
                "encoder.layers.0.attn.Wo.weight",
                "encoder.layers.1.attn_norm.weight",
                "encoder.layers.1.mlp.Wi.weight",
                "encoder.layers.1.mlp.Wo.weight",
                "encoder.final_norm.weight",
                "head.layers.0.self_attn.in_proj_weight",
                "head.layers.0.self_attn.out_proj.bias",
                "head.layers.0.linear1.weight",
                "head.layers.0.norm2.bias",
                "type_emb.weight",
                "scorer.1.weight",
                "scorer.3.bias",
            ] {
                let g = model::host_f32(&grads[name]).unwrap();
                let norm = g.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
                assert!(norm > 1e-6, "{name}: zero gradient");
                let u: Vec<f32> = g.iter().map(|x| (*x as f64 / norm) as f32).collect();
                let num = directional_fd(&p, name, &u, 1e-2, &value);
                assert!(
                    (num - norm).abs() / norm < 2e-2,
                    "{name}: directional derivative {num} vs |g| {norm}"
                );
                let top = (0..g.len())
                    .max_by(|&a, &b| g[a].abs().total_cmp(&g[b].abs()))
                    .unwrap();
                let mut e = vec![0f32; g.len()];
                e[top] = 1.0;
                let num = directional_fd(&p, name, &e, 1e-2, &value);
                let an = g[top] as f64;
                assert!(
                    (num - an).abs() / an.abs() < 3e-2,
                    "{name}[{top}]: finite difference {num} vs autodiff {an}"
                );
            }

            // Soft-target cross-entropy versus its logits.
            let logits =
                Array::from_slice(&[0.3f32, -1.2, 0.8, -1e4, 2.0, 0.1, -0.5, 1.5], &[2, 4]);
            let target = Array::from_slice(&[0.2f32, 0.0, 0.8, 0.0, 0.0, 0.5, 0.25, 0.25], &[2, 4]);
            let ce = |a: &[Array]| -> Vec<Array> {
                vec![model::soft_cross_entropy(&a[0], &target).unwrap()]
            };
            let (_, g) =
                mlx_rs::transforms::vjp(ce, std::slice::from_ref(&logits), &[Array::from_f32(1.0)])
                    .unwrap();
            let g = model::host_f32(&g[0]).unwrap();
            let base = model::host_f32(&logits).unwrap();
            for i in [0usize, 2, 4, 6, 7] {
                let at = |d: f32| {
                    let mut v = base.clone();
                    v[i] += d;
                    model::soft_cross_entropy(&Array::from_slice(&v, &[2, 4]), &target)
                        .unwrap()
                        .item_exact::<f32>() as f64
                };
                let num = (at(1e-2) - at(-1e-2)) / 2e-2;
                assert!(
                    (num - g[i] as f64).abs() < 1e-3,
                    "logit {i}: {num} vs {}",
                    g[i]
                );
            }
            assert_eq!(g[3], 0.0, "masked option gets no gradient");
        });
    }

    #[test]
    fn checkpoint_resume_is_reproducible() {
        let _g = test_util::lock();
        // The CPU device is bit-deterministic; on the GPU the scatter-add of
        // the embedding gradient is not, so two identical runs already differ.
        mlx_rs::with_device(mlx_rs::Device::cpu(), || {
            let cfg = ModelConfig::tiny(40);
            let p = params::init(&cfg, Parts::DECISION, 2).unwrap();
            let mut gen = Rng::new(9);
            let data = synthetic_data(&mut gen, 40, 12);
            let mut s = Session::new(cfg.clone(), p, OptConfig::default(), "decide", 17, 60.0);
            let run = |s: &mut Session, n: usize| {
                for _ in 0..n {
                    let agent = s.cfg.rl_agent.clone();
                    let b = sample_batch(&data, &[1.0], 4, &mut s.state.rng, &agent);
                    s.decision_step(&b, 0.5).unwrap();
                }
            };
            run(&mut s, 3);
            let tmp = tempfile::tempdir().unwrap();
            let dir = tmp.path().join("last");
            s.save(&dir, &[("note.txt", b"x".to_vec())]).unwrap();
            run(&mut s, 3);
            let mut r = Session::load(&dir).unwrap();
            assert_eq!(r.state.step, 3);
            assert_eq!(r.opt.step, 3);
            assert!(dir.join("note.txt").exists());
            run(&mut r, 3);
            assert_eq!(r.state.rng, s.state.rng);
            assert_eq!(r.state.step, s.state.step);
            assert_eq!(r.params.len(), s.params.len());
            for (k, v) in &s.params {
                let w = &r.params[k];
                assert!(
                    v.all_close(w, 1e-6, 1e-7, None).unwrap(),
                    "{k} differs after resume"
                );
            }
        });
    }
}
