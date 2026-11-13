//! A training session (config, parameters, optimizer, RNG and progress) with
//! one-step training functions and atomic checkpoints.
//!
//! A checkpoint directory holds `model.safetensors`, `optimizer.safetensors`,
//! `state.json` (step, elapsed time, RNG, data cursor, configs) and any
//! phase-specific files (tokenizer, `rl_agent_config.json`).

use std::{
    collections::{HashMap, HashSet},
    path::{Path, PathBuf},
    rc::Rc,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
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

/// The thermal brake (`SKY_DUTY` in (0, 1], the share of the time the training loop works): before
/// each step, the loop rests in proportion to the work done since its last rest.
///
/// The rests are taken out of the run clock, so that `elapsed_secs`, a `--minutes` budget and the
/// schedule that follows it only ever see working time.
///
/// `SKY_COOL=45:15` adds long rests: 45 minutes of work, then 15 of complete rest. The cycle follows
/// the system clock, so that it holds across the short steps of a stage, each its own process.
///
/// `SKY_BRAKE_FILE` names a file that sets both while the run goes on, with no restart: one line,
/// `<duty> [work:rest]` (`0.85 50:10`, or `0.5` alone for no long rests). It is read again every
/// few seconds and replaces `SKY_DUTY` and `SKY_COOL`; a missing or unreadable file changes
/// nothing. `scripts/auto-brake.sh` writes it from what the machine is doing.
pub struct Brake {
    rest_per_busy: f64,
    busy_since: Instant,
    /// Seconds of work and of complete rest in each cycle (`SKY_COOL`).
    cool: Option<(u64, u64)>,
    /// `SKY_BRAKE_FILE` and when it was last read.
    file: Option<(std::path::PathBuf, Instant)>,
}

/// Seconds between two reads of `SKY_BRAKE_FILE`.
const BRAKE_FILE_PERIOD: Duration = Duration::from_secs(5);

/// The duty and the long rests of a `SKY_BRAKE_FILE` line; `None` when it is not valid.
fn parse_brake_file(text: &str) -> Option<(f64, Option<(u64, u64)>)> {
    let mut words = text.split_whitespace();
    let duty: f64 = words.next()?.parse().ok()?;
    if !(duty > 0.0 && duty <= 1.0) {
        return None;
    }
    let cool = match words.next() {
        None => None,
        Some(w) => Some(parse_cool(w).ok()?),
    };
    words.next().is_none().then_some((duty, cool))
}

/// `work:rest`, in minutes, as seconds.
fn parse_cool(v: &str) -> Result<(u64, u64)> {
    let secs = |m: &str| -> Option<u64> {
        let m: f64 = m.parse().ok()?;
        (m > 0.0 && m.is_finite()).then(|| (m * 60.0).round() as u64)
    };
    v.split_once(':')
        .and_then(|(w, r)| Some((secs(w)?, secs(r)?)))
        .with_context(|| format!("SKY_COOL must be work:rest in minutes, as 45:15, not {v}"))
}

/// Seconds left in the rest part of the cycle at `now`, 0 in its work part.
fn cool_wait(now: u64, (work, rest): (u64, u64)) -> u64 {
    let at = now % (work + rest);
    if at < work {
        0
    } else {
        work + rest - at
    }
}

impl Brake {
    pub fn from_env() -> Result<Self> {
        let mut brake = match std::env::var("SKY_DUTY") {
            Ok(v) if !v.is_empty() => Self::new(
                v.parse()
                    .with_context(|| format!("SKY_DUTY is not a number: {v}"))?,
            ),
            _ => Self::new(1.0),
        }?;
        if let Ok(v) = std::env::var("SKY_COOL") {
            if !v.is_empty() {
                brake.cool = Some(parse_cool(&v)?);
            }
        }
        if let Ok(v) = std::env::var("SKY_BRAKE_FILE") {
            if !v.is_empty() {
                brake.file = Some((v.into(), Instant::now()));
                brake.read_file();
            }
        }
        Ok(brake)
    }

    /// Takes the duty and the long rests from `SKY_BRAKE_FILE`, when it holds a valid line.
    fn read_file(&mut self) {
        let Some((path, read_at)) = &mut self.file else {
            return;
        };
        *read_at = Instant::now();
        let Some((duty, cool)) = std::fs::read_to_string(&*path)
            .ok()
            .and_then(|t| parse_brake_file(&t))
        else {
            return;
        };
        let rest_per_busy = 1.0 / duty - 1.0;
        if rest_per_busy != self.rest_per_busy || cool != self.cool {
            let rests = cool.map_or("no long rests".to_string(), |(w, r)| {
                format!("{} min of work, {} of rest", w / 60, r / 60)
            });
            println!("brake: duty {duty}, {rests} ({})", path.display());
            self.rest_per_busy = rest_per_busy;
            self.cool = cool;
        }
    }

    /// [`Brake::read_file`], at most once every [`BRAKE_FILE_PERIOD`].
    fn refresh(&mut self) {
        if matches!(&self.file, Some((_, at)) if at.elapsed() >= BRAKE_FILE_PERIOD) {
            self.read_file();
        }
    }

    pub fn new(duty: f64) -> Result<Self> {
        if !(duty > 0.0 && duty <= 1.0) {
            anyhow::bail!("SKY_DUTY must be in (0, 1], not {duty}");
        }
        Ok(Self {
            rest_per_busy: 1.0 / duty - 1.0,
            busy_since: Instant::now(),
            cool: None,
            file: None,
        })
    }

    /// Rests, then moves `clock` (the start of the run clock) forward by the time rested.
    pub fn rest(&mut self, clock: &mut Instant) {
        self.refresh();
        if self.rest_per_busy == 0.0 && self.cool.is_none() {
            return;
        }
        let started = Instant::now();
        std::thread::sleep(self.busy_since.elapsed().mul_f64(self.rest_per_busy));
        let now = || {
            let t = SystemTime::now().duration_since(UNIX_EPOCH);
            t.unwrap_or_default().as_secs()
        };
        let wait = |cool: Option<(u64, u64)>| cool.map_or(0, |cycle| cool_wait(now(), cycle));
        if wait(self.cool) > 0 {
            println!("cooling down for {} s (SKY_COOL)", wait(self.cool));
        }
        // A loop, because a Mac put to sleep can wake up in another rest. With a brake file the
        // rest is taken in short naps, so that a new line can end it or shorten it.
        while wait(self.cool) > 0 {
            let nap = match self.file {
                Some(_) => wait(self.cool).min(BRAKE_FILE_PERIOD.as_secs()),
                None => wait(self.cool),
            };
            std::thread::sleep(Duration::from_secs(nap));
            self.refresh();
        }
        *clock += started.elapsed();
        self.busy_since = Instant::now();
    }
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

/// How many graphs the compiled functions of this thread may hold before MLX's cache is emptied.
///
/// MLX compiles one graph per input shape and keeps every one of them for the life of the process.
/// A decision run meets new batch shapes for hours (rows, state tokens, static streams and static
/// tokens all vary), so its memory grows by a few megabytes per shape, with no bound: s6-nano went
/// from 11 to 14 GB in 70 minutes. Emptying the cache makes the next call of each shape compile
/// again (about 50 ms), which gives the same graph and the same values.
///
/// This bounds the cache, not the process: measured on s6-micro (14/11/2026), a decision batch
/// has hundreds of distinct shapes in its first minutes (rows, state tokens, static streams,
/// static tokens and options), a graph of that model takes 8,400 heap blocks, and emptying the
/// cache gives back two thirds of them: every compilation leaves about 0.4 MB behind. A run
/// whose shapes vary that much is better off uncompiled (`--no-compile`: its heap stays flat,
/// `heap_mb` in the logs); a run with a few shapes never reaches the limit.
struct CompiledShapes {
    /// Graphs compiled since the cache was last emptied.
    count: usize,
    /// The cache is emptied when `count` reaches this (0: never). `SKY_COMPILE_SHAPES`, 256 by
    /// default.
    limit: usize,
    /// Goes up each time the cache is emptied: a function that sees a new value forgets its shapes.
    epoch: u64,
}

thread_local! {
    static COMPILED_SHAPES: std::cell::RefCell<CompiledShapes> = std::cell::RefCell::new(CompiledShapes {
        count: 0,
        limit: std::env::var("SKY_COMPILE_SHAPES")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(256),
        epoch: 0,
    });
}

/// Compiled loss + gradient over a fixed parameter order.
pub(crate) struct CompiledGrad {
    names: Vec<Rc<str>>,
    f: CompiledFn,
    /// The batch shapes compiled since the cache was last emptied (`epoch`).
    seen: HashSet<Vec<i32>>,
    epoch: u64,
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
            seen: HashSet::new(),
            epoch: 0,
        }
    }

    /// Counts the shape of `batch` if it is a new one, and empties MLX's cache of compiled graphs
    /// first when it is full (see [`CompiledShapes`]).
    fn note_shape(&mut self, batch: &[Array]) {
        let mut shape = Vec::new();
        for a in batch {
            shape.extend_from_slice(a.shape());
            shape.push(-1);
        }
        COMPILED_SHAPES.with(|c| {
            let mut c = c.borrow_mut();
            if self.epoch != c.epoch {
                self.seen.clear();
                self.epoch = c.epoch;
            }
            if self.seen.contains(&shape) {
                return;
            }
            if c.limit > 0 && c.count >= c.limit {
                let before = crate::heap::in_use();
                mlx_rs::transforms::compile::clear_cache();
                if std::env::var_os("SKY_MEM_TRACE").is_some() {
                    let after = crate::heap::in_use();
                    println!(
                        "compiled graphs emptied after {} shapes: heap {:.0} -> {:.0} MB, {} -> {} blocks",
                        c.count,
                        before.1 as f64 / 1e6,
                        after.1 as f64 / 1e6,
                        before.0,
                        after.0
                    );
                }
                c.count = 0;
                c.epoch += 1;
                self.seen.clear();
                self.epoch = c.epoch;
            }
            c.count += 1;
            self.seen.insert(shape);
        });
    }

    /// Returns the loss and the gradients keyed by parameter name.
    pub(crate) fn call(
        &mut self,
        params: &ParamMap,
        batch: Vec<Array>,
    ) -> Result<(Array, ParamMap)> {
        self.note_shape(&batch);
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
    let (loss, grads) = loss_and_grads(params, batch, loss)?;
    apply_update(params, opt, loss, &grads, lr_scale)
}

/// The loss and its gradients with respect to every parameter, with no update.
pub(crate) fn loss_and_grads<B>(
    params: &ParamMap,
    batch: &B,
    loss: impl Fn(&ParamMap, &B) -> R<Array>,
) -> Result<(Array, ParamMap)> {
    let f = |p: HashMap<Rc<str>, Array>, b: &B| -> R<Vec<Array>> { Ok(vec![loss(&p, b)?]) };
    let mut vg = keyed_value_and_grad(f);
    let (values, grads) = vg(params.clone(), batch)?;
    Ok((values.into_iter().next().expect("loss output"), grads))
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
mod compiled_shapes_tests {
    use super::{CompiledGrad, COMPILED_SHAPES};
    use crate::params::ParamMap;
    use mlx_rs::Array;
    use std::rc::Rc;

    /// The loss and the gradient of `sum((w * x)^2)` over batches of 1 to `shapes` elements, twice
    /// over, with the cache emptied every `limit` graphs; and how many times it was emptied.
    fn run(limit: usize, shapes: i32) -> (Vec<(f32, Vec<f32>)>, u64) {
        COMPILED_SHAPES.with(|c| {
            let mut c = c.borrow_mut();
            (c.count, c.limit, c.epoch) = (0, limit, 0);
        });
        let name: Rc<str> = Rc::from("w");
        let mut params = ParamMap::new();
        params.insert(name.clone(), Array::from_slice(&[0.5f32], &[1]));
        let mut f = CompiledGrad::new(vec![name.clone()], |p, b| {
            p["w"].multiply(&b[0])?.square()?.sum(None)
        });
        let mut out = Vec::new();
        for _ in 0..2 {
            for n in 1..=shapes {
                let x: Vec<f32> = (0..n).map(|i| i as f32 + 1.0).collect();
                let (loss, grads) = f.call(&params, vec![Array::from_slice(&x, &[n])]).unwrap();
                out.push((loss.as_slice::<f32>()[0], grads[&name].as_slice::<f32>().to_vec()));
            }
        }
        (out, COMPILED_SHAPES.with(|c| c.borrow().epoch))
    }

    /// Emptying the cache of compiled graphs changes no value: a shape compiled again gives the
    /// loss and the gradient it gave the first time.
    #[test]
    fn emptying_the_cache_of_compiled_graphs_changes_no_value() {
        mlx_rs::with_device(mlx_rs::Device::cpu(), || {
            let (kept, never) = run(0, 7);
            assert_eq!(never, 0);
            // 14 calls over 7 shapes with room for 3: emptied after the 3rd, 6th, 9th and 12th new
            // shape (the second pass finds none of its shapes left).
            let (emptied, times) = run(3, 7);
            assert_eq!(times, 4);
            assert_eq!(kept, emptied);
            // Room for every shape: compiled once each, never emptied.
            let (all, times) = run(7, 7);
            assert_eq!(times, 0);
            assert_eq!(kept, all);
            // sum((w x)^2) = w^2 sum(x^2) and its gradient 2 w sum(x^2), at w = 0.5 and x = 1, 2.
            assert_eq!(kept[1], (1.25, vec![5.0]));
        });
    }

}

#[cfg(test)]
mod brake_tests {
    use super::{cool_wait, parse_brake_file, parse_cool, Brake};
    use std::time::{Duration, Instant};

    #[test]
    fn the_cycle_works_then_rests() {
        let cycle = (45 * 60, 15 * 60);
        assert_eq!(cool_wait(0, cycle), 0);
        assert_eq!(cool_wait(45 * 60 - 1, cycle), 0);
        assert_eq!(cool_wait(45 * 60, cycle), 15 * 60);
        assert_eq!(cool_wait(59 * 60, cycle), 60);
        assert_eq!(cool_wait(60 * 60, cycle), 0);
    }

    #[test]
    fn parses_work_and_rest_in_minutes() {
        assert_eq!(parse_cool("45:15").unwrap(), (2700, 900));
        assert_eq!(parse_cool("0.5:0.25").unwrap(), (30, 15));
        for bad in ["45", "0:15", "45:0", "a:b", "-1:5", "inf:5"] {
            assert!(parse_cool(bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn refuses_a_duty_outside_0_1() {
        for d in [0.0, -0.5, 1.5, f64::NAN] {
            assert!(Brake::new(d).is_err(), "{d}");
        }
        assert!(Brake::new(1.0).is_ok());
    }

    #[test]
    fn the_rests_stay_out_of_the_run_clock() {
        let mut clock = Instant::now();
        let mut brake = Brake::new(0.5).unwrap();
        let wall = Instant::now();
        for _ in 0..3 {
            std::thread::sleep(Duration::from_millis(40));
            brake.rest(&mut clock);
        }
        let (wall, worked) = (wall.elapsed(), clock.elapsed());
        assert!(wall >= Duration::from_millis(240), "rested: {wall:?}");
        assert!(
            worked < wall.mul_f64(0.75),
            "clock moved: {worked:?} of {wall:?}"
        );
    }

    #[test]
    fn full_duty_does_not_rest() {
        let start = Instant::now();
        let mut clock = start;
        let mut brake = Brake::new(1.0).unwrap();
        std::thread::sleep(Duration::from_millis(20));
        brake.rest(&mut clock);
        assert_eq!(clock, start);
    }

    #[test]
    fn parses_a_brake_file_line() {
        assert_eq!(
            parse_brake_file("0.85 50:10\n"),
            Some((0.85, Some((3000, 600))))
        );
        assert_eq!(parse_brake_file("0.5"), Some((0.5, None)));
        assert_eq!(
            parse_brake_file(" 1 45:15 "),
            Some((1.0, Some((2700, 900))))
        );
        for bad in ["", "fast", "0", "1.5", "0.5 45", "0.5 45:15 more"] {
            assert_eq!(parse_brake_file(bad), None, "{bad:?}");
        }
    }

    #[test]
    fn a_brake_file_changes_the_duty_of_a_running_brake() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("brake");
        let mut brake = Brake::new(1.0).unwrap();
        brake.file = Some((path.clone(), Instant::now()));
        // no file yet: nothing changes
        brake.read_file();
        assert_eq!((brake.rest_per_busy, brake.cool), (0.0, None));
        std::fs::write(&path, "0.5 45:15\n").unwrap();
        brake.read_file();
        assert_eq!((brake.rest_per_busy, brake.cool), (1.0, Some((2700, 900))));
        // a line that is not valid keeps the last setting
        std::fs::write(&path, "soon").unwrap();
        brake.read_file();
        assert_eq!((brake.rest_per_busy, brake.cool), (1.0, Some((2700, 900))));
        std::fs::write(&path, "1").unwrap();
        brake.read_file();
        assert_eq!((brake.rest_per_busy, brake.cool), (0.0, None));
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
