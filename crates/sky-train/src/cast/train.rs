//! s6-cast training (docs/SPEC-S6-CAST.md §3): pinball deep supervision over every exit, WSD
//! learning rate, Muon v2 on the encoder blocks, ternary QAT over the last `qat_frac` of the
//! budget, optional weight EMA, resumable checkpoints (`last/`, `pre-qat/`, `last-ema/`).

use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{bail, Context, Result};
use mlx_rs::Array;
use serde::{Deserialize, Serialize};
use serde_json::json;

use super::{
    config::CastConfig,
    data::{self, Pool, SampleOptions, Window},
    model::{self, exit_losses},
};
use crate::{
    model::host_f32,
    optim::{wsd, OptConfig, Optimizer, WdMode},
    params::{self, ParamMap},
    rng::Rng,
    session::{keep_going, qat_due, stop_step, train_step, Progress},
    train::JsonLog,
};

#[derive(Debug, Clone, clap::Args, Serialize, Deserialize)]
pub struct TrainArgs {
    /// Model config (`configs/s6-cast-*.json`).
    #[arg(long)]
    pub config: PathBuf,
    /// Run directory: checkpoints, `log.jsonl`.
    #[arg(long)]
    pub out: PathBuf,
    /// Time budget in minutes (or `--steps`).
    #[arg(long)]
    pub minutes: Option<f64>,
    /// Step budget (the same schedule on every machine).
    #[arg(long)]
    pub steps: Option<u64>,
    /// Stop (with a checkpoint) after this many steps of this invocation.
    #[arg(long)]
    pub stop_after: Option<u64>,
    /// Continue from `<out>/last`.
    #[arg(long)]
    pub resume: bool,
    #[arg(long, default_value_t = 256)]
    pub batch: usize,
    #[arg(long, default_value_t = 1)]
    pub seed: u64,
    /// Real series (JSONL files or directories, comma separated; see `data::load_series`).
    #[arg(long, value_delimiter = ',')]
    pub series: Vec<PathBuf>,
    /// Share of the windows cut from real series (when `--series` gives any).
    #[arg(long, default_value_t = 0.3)]
    pub real_frac: f64,
    /// Probability of each augmentation of a window.
    #[arg(long, default_value_t = 0.3)]
    pub augment: f64,
    #[arg(long, default_value_t = 0.1)]
    pub mixup: f64,
    /// Probability of dropping 1–5 % of a context.
    #[arg(long, default_value_t = 0.1)]
    pub missing: f64,
    /// Synthetic series in the pool.
    #[arg(long, default_value_t = 4096)]
    pub pool: usize,
    /// Points per synthetic series (0: context + horizon + 1024).
    #[arg(long, default_value_t = 0)]
    pub pool_len: usize,
    /// Series regenerated per refresh (must divide `--pool`).
    #[arg(long, default_value_t = 256)]
    pub refresh: usize,
    #[arg(long, default_value_t = 50)]
    pub refresh_every: u64,
    /// Ternary QAT over this last fraction of the budget (0 disables).
    #[arg(long, default_value_t = 0.2)]
    pub qat_frac: f64,
    /// StableQAT amplitude (0: straight-through estimator).
    #[arg(long, default_value_t = 0.0)]
    pub qat_grad_a: f32,
    /// Weight EMA decay (0 disables); saved to `<out>/last-ema`.
    #[arg(long, default_value_t = 0.0)]
    pub ema: f32,
    /// Synthetic validation windows.
    #[arg(long, default_value_t = 512)]
    pub val_synth: usize,
    #[arg(long, default_value_t = 50)]
    pub log_every: u64,
    #[arg(long, default_value_t = 500)]
    pub eval_every: u64,
    #[arg(long, default_value_t = 10.0)]
    pub checkpoint_minutes: f64,
    /// Peak AdamW learning rate.
    #[arg(long, default_value_t = 3e-3)]
    pub adam_lr: f32,
    /// Peak Muon learning rate of the encoder blocks (default: the AdamW one).
    #[arg(long)]
    pub muon_lr: Option<f32>,
    #[arg(long, default_value_t = 0.01)]
    pub weight_decay: f32,
    #[arg(long, default_value_t = 1.0)]
    pub clip_norm: f32,
    /// AdamW on the encoder blocks too.
    #[arg(long)]
    pub no_muon: bool,
    #[arg(long, default_value_t = 2)]
    pub ns_polish: usize,
    #[arg(long, value_enum, default_value_t = WdMode::Decoupled)]
    pub wd_mode: WdMode,
    /// Distillation teacher (a TinyCast or s6-cast GGUF): its quantiles become a second target.
    #[arg(long)]
    pub teacher: Option<PathBuf>,
    /// Weight of the distillation term at the start; it decays linearly to 0 over the budget
    /// (teacher annealing), so the end of training follows the ground truth only.
    #[arg(long, default_value_t = 0.5)]
    pub teacher_weight: f32,
    /// Share of the windows that get the teacher's targets.
    #[arg(long, default_value_t = 0.5)]
    pub teacher_frac: f64,
    /// CPU threads of the teacher (0: all but one).
    #[arg(long, default_value_t = 0)]
    pub teacher_threads: usize,
}

impl TrainArgs {
    fn opt(&self) -> OptConfig {
        OptConfig {
            adam_lr: self.adam_lr,
            muon_lr: self.muon_lr.unwrap_or(self.adam_lr),
            weight_decay: self.weight_decay,
            wd_mode: self.wd_mode,
            clip_norm: self.clip_norm,
            muon: !self.no_muon,
            ns_polish: self.ns_polish,
            ..OptConfig::default()
        }
    }

    fn sample_options(&self) -> SampleOptions {
        SampleOptions {
            real_frac: self.real_frac,
            augment: self.augment,
            mixup: self.mixup,
            missing: self.missing,
        }
    }
}

/// Progress saved with a checkpoint.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CastState {
    pub step: u64,
    pub elapsed_secs: f64,
    pub rng: Rng,
    pub qat: bool,
    pub config: CastConfig,
    pub opt: OptConfig,
    pub optimizer_step: u64,
}

struct Session {
    cfg: CastConfig,
    params: ParamMap,
    opt: Optimizer,
    state: CastState,
}

impl Session {
    fn save(&self, dir: &Path) -> Result<()> {
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
        let mut st = self.state.clone();
        st.optimizer_step = self.opt.step;
        std::fs::write(tmp.join("state.json"), serde_json::to_string_pretty(&st)?)?;
        std::fs::write(
            tmp.join("config.json"),
            serde_json::to_string_pretty(&self.cfg)?,
        )?;
        if dir.exists() {
            std::fs::remove_dir_all(dir)?;
        }
        std::fs::rename(&tmp, dir)?;
        Ok(())
    }

    fn load(dir: &Path) -> Result<Self> {
        let state: CastState = serde_json::from_str(
            &std::fs::read_to_string(dir.join("state.json"))
                .with_context(|| format!("reading {}/state.json", dir.display()))?,
        )?;
        let params = params::load(&dir.join("model.safetensors"))?;
        let mut opt = Optimizer::new(state.opt.clone());
        opt.load_state(
            state.optimizer_step,
            &params::load(&dir.join("optimizer.safetensors"))?,
        )?;
        Ok(Session {
            cfg: state.config.clone(),
            params,
            opt,
            state,
        })
    }
}

/// Loads the weights of a run checkpoint (`<run>/last`, `last-ema`, `pre-qat`) and its config.
pub fn load_run(dir: &Path) -> Result<(CastConfig, ParamMap)> {
    let cfg: CastConfig = serde_json::from_str(
        &std::fs::read_to_string(dir.join("config.json"))
            .with_context(|| format!("reading {}/config.json", dir.display()))?,
    )?;
    Ok((cfg, params::load(&dir.join("model.safetensors"))?))
}

/// Validation metrics: per exit, the mean pinball loss and the mean absolute error of the median
/// (normalized units).
pub fn validate(
    p: &ParamMap,
    cfg: &CastConfig,
    val: &[Window],
    qat: bool,
) -> Result<(Vec<f64>, Vec<f64>)> {
    let ne = cfg.n_exits();
    let (mut pin, mut mae, mut n) = (vec![0.0; ne], vec![0.0; ne], 0.0);
    let med = cfg.quantiles.len() / 2;
    for chunk in val.chunks(256) {
        let b = data::batch(cfg, chunk);
        let q = model::forward(p, cfg, &b, qat)?;
        let l = host_f32(&exit_losses(&q, cfg, &b)?)?;
        let qh = host_f32(&q)?;
        let (h, nq) = (cfg.horizon(), cfg.quantiles.len());
        let w = chunk.len() as f64;
        for e in 0..ne {
            pin[e] += l[e] as f64 * w;
            let mut err = 0.0;
            let mut cnt = 0.0;
            for (r, win) in chunk.iter().enumerate() {
                for t in 0..h {
                    if win.ymask[t] > 0.0 {
                        let v = qh[((e * chunk.len() + r) * h + t) * nq + med];
                        err += (v - win.y[t]).abs() as f64;
                        cnt += 1.0;
                    }
                }
            }
            mae[e] += if cnt > 0.0 { err / cnt * w } else { 0.0 };
        }
        n += w;
    }
    Ok((
        pin.into_iter().map(|x| x / n.max(1.0)).collect(),
        mae.into_iter().map(|x| x / n.max(1.0)).collect(),
    ))
}

fn log_val(
    log: &mut JsonLog,
    step: u64,
    mode: &str,
    (pin, mae): &(Vec<f64>, Vec<f64>),
) -> Result<()> {
    let per = |v: &[f64]| -> serde_json::Value {
        v.iter()
            .enumerate()
            .map(|(e, x)| (format!("exit{e}"), json!(x)))
            .collect::<serde_json::Map<_, _>>()
            .into()
    };
    println!(
        "step {step:>6} val {mode}: pinball {:.4} mae {:.4} (last exit)",
        pin.last().copied().unwrap_or(f64::NAN),
        mae.last().copied().unwrap_or(f64::NAN)
    );
    log.write(&json!({"step": step, "mode": mode, "val": {"pinball": per(pin), "mae": per(mae)}}))
}

pub fn train(a: &TrainArgs) -> Result<()> {
    let progress = Progress::from_flags(a.minutes, a.steps)?;
    let last = a.out.join("last");
    let resumed = a.resume && last.join("state.json").exists();
    let mut s = if resumed {
        let s = Session::load(&last)?;
        println!("resumed from {} at step {}", last.display(), s.state.step);
        s
    } else {
        let mut cfg = CastConfig::load(&a.config)?;
        cfg.qat_grad_a = a.qat_grad_a;
        let params = model::init(&cfg, a.seed)?;
        Session {
            state: CastState {
                step: 0,
                elapsed_secs: 0.0,
                rng: Rng::new(a.seed),
                qat: false,
                config: cfg.clone(),
                opt: a.opt(),
                optimizer_step: 0,
            },
            cfg,
            params,
            opt: Optimizer::new(a.opt()),
        }
    };
    s.cfg.qat_grad_a = a.qat_grad_a;
    if a.pool == 0 || !a.pool.is_multiple_of(a.refresh.max(1)) {
        bail!("--refresh must divide --pool");
    }
    let cfg = s.cfg.clone();
    println!(
        "s6-cast {}: {} parameters, context {} points, horizon {} points, {} exits",
        cfg.name,
        params::count(&s.params),
        cfg.context(),
        cfg.horizon(),
        cfg.n_exits()
    );
    std::fs::create_dir_all(&a.out)?;
    std::fs::write(a.out.join("args.json"), serde_json::to_string_pretty(a)?)?;
    let all_series = data::load_series(&a.series)?;
    let (real_val, real): (Vec<_>, Vec<_>) =
        all_series.into_iter().partition(|s| data::is_val(&s.id));
    println!("real series: {} train, {} val", real.len(), real_val.len());
    let pool_len = if a.pool_len == 0 {
        cfg.context() + cfg.horizon() + 1024
    } else {
        a.pool_len
    };
    let t_pool = Instant::now();
    let mut pool = Pool::new(
        a.seed ^ 0x9001,
        a.pool,
        pool_len,
        a.refresh,
        a.refresh_every,
    );
    if s.state.step > 0 {
        pool.rebuild(s.state.step);
    }
    println!(
        "synthetic pool: {} series of {pool_len} points ({:.1}s)",
        a.pool,
        t_pool.elapsed().as_secs_f64()
    );
    let val = data::validation(&cfg, a.val_synth, &real_val);
    let opts = a.sample_options();
    let teacher = match &a.teacher {
        Some(p) => {
            let t = data::Teacher::load(p, &cfg)?;
            println!(
                "teacher {}: {:.0}% of the windows, weight {} annealed to 0",
                t.name(),
                a.teacher_frac * 100.0,
                a.teacher_weight
            );
            Some(t)
        }
        None => None,
    };
    let threads = if a.teacher_threads > 0 {
        a.teacher_threads
    } else {
        std::thread::available_parallelism().map_or(4, |n| n.get().saturating_sub(1).max(1))
    };
    // The batches are prepared one or two steps ahead by a producer thread (sampling, and the
    // teacher's forecasts on the CPU), deterministically in the step.
    let (tx, rx) = std::sync::mpsc::sync_channel::<Result<Vec<Window>>>(2);
    let producer = {
        let (cfg, seed, batch, frac, start) =
            (cfg.clone(), a.seed, a.batch, a.teacher_frac, s.state.step);
        std::thread::spawn(move || {
            let mut step = start;
            loop {
                pool.advance(step);
                let w = data::sample_step(
                    &cfg,
                    &pool,
                    &real,
                    &opts,
                    seed,
                    step,
                    batch,
                    teacher.as_ref().map(|t| (t, frac, threads)),
                );
                if tx.send(w).is_err() {
                    break;
                }
                step += 1;
            }
        })
    };
    let distilling = a.teacher.is_some();
    let mut log = JsonLog::open(&a.out.join("log.jsonl"))?;
    let wts = Array::from_slice(&cfg.exit_weights(), &[cfg.n_exits() as i32]);
    let mut ema: Option<ParamMap> = (a.ema > 0.0).then(|| params::deep_clone(&s.params));
    let ema_dir = a.out.join("last-ema");
    if resumed && a.ema > 0.0 && ema_dir.join("model.safetensors").exists() {
        ema = Some(params::load(&ema_dir.join("model.safetensors"))?);
    }
    let stop_at = stop_step(s.state.step, a.stop_after)?;
    let base = s.state.elapsed_secs;
    let t0 = Instant::now();
    let mut last_ckpt = Instant::now();
    let (mut wl, mut wn, mut wtime) = (0.0f64, 0u64, Instant::now());
    let mut consecutive = 0;
    while keep_going(&progress, stop_at, s.state.step, s.state.elapsed_secs) {
        let frac = progress.frac(s.state.step, s.state.elapsed_secs);
        if !s.state.qat && qat_due(frac, a.qat_frac) {
            println!("step {}: starting ternary QAT", s.state.step);
            let f = validate(&s.params, &cfg, &val, false)?;
            let t = validate(&s.params, &cfg, &val, true)?;
            log_val(&mut log, s.state.step, "f32", &f)?;
            log_val(&mut log, s.state.step, "ternary", &t)?;
            s.save(&a.out.join("pre-qat"))?;
            s.state.qat = true;
        }
        let windows = rx.recv().context("the batch producer stopped")??;
        let b = data::batch(&cfg, &windows);
        let lr_scale = wsd(frac) as f32;
        let qat = s.state.qat;
        let dw = if distilling {
            a.teacher_weight * (1.0 - frac.clamp(0.0, 1.0) as f32)
        } else {
            0.0
        };
        let (loss, gnorm) = train_step(&mut s.params, &mut s.opt, &b, lr_scale, |p, b| {
            model::loss(p, &cfg, b, qat, &wts, dw)
        })?;
        s.state.step += 1;
        s.state.elapsed_secs = base + t0.elapsed().as_secs_f64();
        if !loss.is_finite() {
            consecutive += 1;
            log.write(&json!({"step": s.state.step, "skipped_batch": true}))?;
            if consecutive >= 20 {
                bail!(
                    "non-finite loss on 20 consecutive batches at step {}",
                    s.state.step
                );
            }
            continue;
        }
        consecutive = 0;
        if let Some(e) = &mut ema {
            crate::s5::train::ema_update(e, &s.params, a.ema)?;
        }
        (wl, wn) = (wl + loss as f64, wn + 1);
        let step = s.state.step;
        if step % a.log_every == 0 {
            let dt = wtime.elapsed().as_secs_f64();
            log.write(
                &json!({"step": step, "loss": wl / wn as f64, "grad_norm": gnorm,
                "lr_scale": lr_scale, "qat": qat, "frac": frac,
                "windows_s": (wn as usize * a.batch) as f64 / dt,
                "elapsed_min": s.state.elapsed_secs / 60.0}),
            )?;
            println!(
                "step {step:>6} loss {:.4} lr x{lr_scale:.3}{} {:.0} windows/s",
                wl / wn as f64,
                if qat { " qat" } else { "" },
                (wn as usize * a.batch) as f64 / dt
            );
            (wl, wn, wtime) = (0.0, 0, Instant::now());
        }
        if step % a.eval_every == 0 {
            let m = validate(&s.params, &cfg, &val, qat)?;
            log_val(&mut log, step, if qat { "ternary" } else { "f32" }, &m)?;
        }
        if last_ckpt.elapsed().as_secs_f64() >= a.checkpoint_minutes * 60.0 {
            s.save(&last)?;
            if let Some(e) = &ema {
                save_ema(&s, e, &ema_dir)?;
            }
            last_ckpt = Instant::now();
        }
    }
    drop(rx);
    producer.join().expect("batch producer");
    s.save(&last)?;
    if let Some(e) = &ema {
        save_ema(&s, e, &ema_dir)?;
    }
    if !progress.done(s.state.step, s.state.elapsed_secs) {
        println!(
            "stopped at step {}: checkpoint {}",
            s.state.step,
            last.display()
        );
        return Ok(());
    }
    println!("done: {} steps", s.state.step);
    let f = validate(&s.params, &cfg, &val, false)?;
    let t = validate(&s.params, &cfg, &val, true)?;
    log_val(&mut log, s.state.step, "f32", &f)?;
    log_val(&mut log, s.state.step, "ternary", &t)?;
    if let Some(e) = &ema {
        let f = validate(e, &cfg, &val, false)?;
        let t = validate(e, &cfg, &val, true)?;
        log_val(&mut log, s.state.step, "ema-f32", &f)?;
        log_val(&mut log, s.state.step, "ema-ternary", &t)?;
    }
    log.write(&json!({"step": s.state.step, "final": true}))?;
    Ok(())
}

/// The EMA weights saved as a run checkpoint of their own (same state, EMA parameters).
fn save_ema(s: &Session, ema: &ParamMap, dir: &Path) -> Result<()> {
    let tmp = Session {
        cfg: s.cfg.clone(),
        params: ema.clone(),
        opt: Optimizer::new(s.state.opt.clone()),
        state: s.state.clone(),
    };
    tmp.save(dir)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_util;

    /// A few steps on synthetic data lower the loss, and a resumed run continues where the first
    /// stopped (sanity, not a training run).
    #[test]
    fn short_run_learns_and_resumes() {
        let _g = test_util::lock();
        let dir = tempfile::tempdir().unwrap();
        let cfg = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../configs/s6-cast-femto.json");
        let args = TrainArgs {
            config: cfg,
            out: dir.path().join("run"),
            minutes: None,
            steps: Some(60),
            stop_after: Some(30),
            resume: false,
            batch: 16,
            seed: 3,
            series: Vec::new(),
            real_frac: 0.3,
            augment: 0.3,
            mixup: 0.1,
            missing: 0.1,
            pool: 64,
            pool_len: 0,
            refresh: 16,
            refresh_every: 10,
            qat_frac: 0.2,
            qat_grad_a: 0.0,
            ema: 0.9,
            val_synth: 32,
            log_every: 10,
            eval_every: 30,
            checkpoint_minutes: 100.0,
            adam_lr: 3e-3,
            muon_lr: None,
            weight_decay: 0.01,
            clip_norm: 1.0,
            no_muon: false,
            ns_polish: 2,
            wd_mode: WdMode::Decoupled,
            teacher: None,
            teacher_weight: 0.5,
            teacher_frac: 0.5,
            teacher_threads: 2,
        };
        train(&args).unwrap();
        let st: CastState = serde_json::from_str(
            &std::fs::read_to_string(args.out.join("last/state.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(st.step, 30);
        train(&TrainArgs {
            resume: true,
            stop_after: None,
            ..args.clone()
        })
        .unwrap();
        let log = std::fs::read_to_string(args.out.join("log.jsonl")).unwrap();
        let losses: Vec<f64> = log
            .lines()
            .filter_map(|l| serde_json::from_str::<serde_json::Value>(l).ok())
            .filter_map(|v| v["loss"].as_f64())
            .collect();
        assert_eq!(losses.len(), 6);
        assert!(losses.iter().all(|l| l.is_finite()));
        assert!(losses[5] < losses[0], "{losses:?}");
        assert!(log.contains("\"final\":true") && log.contains("ema-ternary"));
        assert!(args.out.join("pre-qat/model.safetensors").exists());
        assert!(args.out.join("last-ema/model.safetensors").exists());

        // with TinyCast as the teacher: the producer computes its targets, the run finishes
        let teacher =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../sky-infer/tests/golden/tinycast.gguf");
        let out = dir.path().join("distill");
        train(&TrainArgs {
            out: out.clone(),
            steps: Some(20),
            stop_after: None,
            batch: 8,
            teacher: Some(teacher),
            ..args.clone()
        })
        .unwrap();
        let log = std::fs::read_to_string(out.join("log.jsonl")).unwrap();
        assert!(log.contains("\"final\":true"));
    }
}
