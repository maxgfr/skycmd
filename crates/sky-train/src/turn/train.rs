//! s6-ear v2 training (docs/s6-ear2-prereg.md §6): the loss of §5.2 (or the hard labels only,
//! the pilot's fourth arm), the recipe's optimizer, WSD learning rate, ternary QAT over the last
//! `qat_frac` of the budget, a weight EMA, the brake modes, resumable checkpoints (`last/`,
//! `pre-qat/`, `last-ema/`) and the validation metrics the pilot, the stop rule and the
//! checkpoint choice read (`log.jsonl`).

use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::json;

use super::{
    config::TurnConfig,
    data::{
        conv_batch, host_batch, mixed_batch, st_draws_before, Augment, Conv, Data, Ext, HostBatch,
        Mix, Order, Sources,
    },
    model::{self, LossWeights},
};
use crate::{
    model::host_f32,
    optim::{wsd, OptConfig, Optimizer, WdMode},
    params::{self, ParamMap},
    rng::hash_str,
    session::{keep_going, qat_due, stop_step, train_step, Brake, Progress},
    train::JsonLog,
};

#[derive(Debug, Clone, clap::Args, Serialize, Deserialize)]
pub struct TrainArgs {
    /// Model config (`configs/s6-ear2-nano.json`, a ladder: give `--rung`).
    #[arg(long)]
    pub config: PathBuf,
    /// The rung chosen by the latency probe (§4.4).
    #[arg(long)]
    pub rung: Option<String>,
    /// The training feature store (sky-data fetch-turn, with the teacher's targets).
    #[arg(long)]
    pub data: PathBuf,
    /// Run directory: checkpoints, `log.jsonl`.
    #[arg(long)]
    pub out: PathBuf,
    /// Working minutes (or `--steps`).
    #[arg(long)]
    pub minutes: Option<f64>,
    #[arg(long)]
    pub steps: Option<u64>,
    /// Stop (with a checkpoint) after this many steps of this invocation.
    #[arg(long)]
    pub stop_after: Option<u64>,
    /// Continue from `<out>/last`.
    #[arg(long)]
    pub resume: bool,
    #[arg(long, default_value_t = 128)]
    pub batch: usize,
    #[arg(long, default_value_t = 1)]
    pub seed: u64,
    /// Train on the label term only (the pilot's hard-label arm, §6.4).
    #[arg(long)]
    pub hard_only: bool,
    /// Ternary QAT over this last fraction of the budget (0: none, the pilot).
    #[arg(long, default_value_t = 0.15)]
    pub qat_frac: f64,
    /// StableQAT amplitude (0: straight-through estimator).
    #[arg(long, default_value_t = 0.0)]
    pub qat_grad_a: f32,
    /// Weight EMA decay (0 disables); saved to `<out>/last-ema`.
    #[arg(long, default_value_t = 0.0)]
    pub ema: f32,
    /// The light augmentation of §3.3 (off: no jitter and no mask).
    #[arg(long)]
    pub no_augment: bool,
    /// Keep the first clips of the training / validation split only (0: all; the smoke run).
    #[arg(long, default_value_t = 0)]
    pub max_train: usize,
    #[arg(long, default_value_t = 0)]
    pub max_val: usize,
    #[arg(long, default_value_t = 100)]
    pub log_every: u64,
    #[arg(long, default_value_t = 2000)]
    pub eval_every: u64,
    /// The periodic validation (the curve) reads the first N validation clips (0: all); the
    /// validations at the QAT switch and at the end read them all.
    #[arg(long, default_value_t = 0)]
    #[serde(default)]
    pub curve_val: usize,
    #[arg(long, default_value_t = 10.0)]
    pub checkpoint_minutes: f64,
    /// Peak AdamW learning rate.
    #[arg(long, default_value_t = 1e-3)]
    pub adam_lr: f32,
    /// Peak Muon learning rate of the encoder blocks (default: the AdamW one).
    #[arg(long)]
    pub muon_lr: Option<f32>,
    #[arg(long, default_value_t = 0.01)]
    pub weight_decay: f32,
    #[arg(long, default_value_t = 1.0)]
    pub clip_norm: f32,
    /// AdamW on the encoder blocks too (the recipe's choice: runs/recipe/recipe.json).
    #[arg(long)]
    pub no_muon: bool,
    #[arg(long, default_value_t = 2)]
    pub ns_polish: usize,
    #[arg(long, value_enum, default_value_t = WdMode::Decoupled)]
    pub wd_mode: WdMode,
    /// Run the graph on the CPU (tests and sanity runs; never the GPU).
    #[arg(long)]
    #[serde(default)]
    pub cpu: bool,
    /// s6-ear v3 (docs/s6-ear3-prereg.md §3.6, §16): the conversation store (`sky-data
    /// fetch-conv`); with it, every batch is the mix of §3.6 and `--st-ext` is needed.
    #[arg(long)]
    #[serde(default)]
    pub conv: Option<PathBuf>,
    /// s6-ear v3: the Smart Turn extension store (`sky-data turn-extend`).
    #[arg(long)]
    #[serde(default)]
    pub st_ext: Option<PathBuf>,
    /// s6-ear v3: the probability that a row is a conversation record.
    #[arg(long, default_value_t = 0.5)]
    #[serde(default = "half")]
    pub conv_share: f64,
    /// s6-ear v3: the probability that a Smart Turn row is v2's view (d = 0).
    #[arg(long, default_value_t = 0.25)]
    #[serde(default = "quarter")]
    pub st_d0: f64,
    /// s6-ear v3: keep the first conversation records of the training / validation split only
    /// (0: all; the smoke run).
    #[arg(long, default_value_t = 0)]
    #[serde(default)]
    pub conv_max_train: usize,
    #[arg(long, default_value_t = 0)]
    #[serde(default)]
    pub conv_max_val: usize,
    /// s6-ear v3: start from the weights of this checkpoint (`config.json`,
    /// `model.safetensors`; the same graph), with a fresh optimizer.
    #[arg(long)]
    #[serde(default)]
    pub init: Option<PathBuf>,
    /// The sha256 of the files of `--init` (`runs/s6-ear3-nano/init.json`, `{"files": {name:
    /// sha256}}`): `--init` refuses a checkpoint whose files differ.
    #[arg(long)]
    #[serde(default)]
    pub init_sha256: Option<PathBuf>,
}

fn half() -> f64 {
    0.5
}
fn quarter() -> f64 {
    0.25
}

/// The ask of the conversation validation of §6.4 and §7 (d = 0.2 s).
pub const CONV_VAL_MS: usize = 200;

/// `--init`'s guard: every file listed in `list` has its sha256 (`shasum -a 256`), and the list
/// names the two files read.
pub fn check_init(dir: &Path, list: &Path) -> Result<()> {
    let v: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(list).with_context(|| format!("reading {}", list.display()))?,
    )?;
    let files = v["files"]
        .as_object()
        .ok_or_else(|| anyhow::anyhow!("{}: no `files`", list.display()))?;
    for need in ["config.json", "model.safetensors"] {
        if !files.contains_key(need) {
            bail!("{}: no sha256 of {need}", list.display());
        }
    }
    for (name, want) in files {
        let p = dir.join(name);
        let out = std::process::Command::new("shasum")
            .args(["-a", "256"])
            .arg(&p)
            .output()
            .context("running shasum")?;
        let text = String::from_utf8_lossy(&out.stdout);
        let got = text.split_whitespace().next().unwrap_or("");
        if !out.status.success() || Some(got) != want.as_str() {
            bail!(
                "{}: sha256 {got:?}, {} records {want}: --init refuses this checkpoint",
                p.display(),
                list.display()
            );
        }
    }
    Ok(())
}

/// Two configs of the same graph (the names and the rung aside).
pub fn same_graph(a: &TurnConfig, b: &TurnConfig) -> bool {
    let strip = |c: &TurnConfig| TurnConfig {
        name: String::new(),
        rung: None,
        qat_grad_a: 0.0,
        ..c.clone()
    };
    strip(a) == strip(b)
}

/// The conversation validation at d = 0.2 s (§6.4, §7): AUC of end of turn against hold, BCE.
#[derive(Debug, Clone, Copy, Default, Serialize)]
pub struct ConvMetrics {
    pub auc: f64,
    pub bce: f64,
    pub n: usize,
}

/// AUC of `y` by score `z` (ties shared).
pub fn auc(z: &[f64], y: &[bool]) -> f64 {
    let mut o: Vec<usize> = (0..z.len()).collect();
    o.sort_by(|&a, &b| z[a].total_cmp(&z[b]));
    let (pos, neg) = (
        y.iter().filter(|&&v| v).count() as f64,
        y.iter().filter(|&&v| !v).count() as f64,
    );
    if pos == 0.0 || neg == 0.0 {
        return f64::NAN;
    }
    let (mut s, mut below, mut i) = (0.0, 0.0, 0);
    while i < o.len() {
        let mut j = i;
        let (mut gp, mut gn) = (0.0, 0.0);
        while j < o.len() && z[o[j]] == z[o[i]] {
            if y[o[j]] {
                gp += 1.0
            } else {
                gn += 1.0
            }
            j += 1;
        }
        s += gp * (below + 0.5 * gn);
        below += gn;
        i = j;
    }
    s / (pos * neg)
}

pub fn validate_conv(
    p: &ParamMap,
    cfg: &TurnConfig,
    conv: &Conv,
    qat: bool,
    limit: usize,
) -> Result<ConvMetrics> {
    let val = if limit > 0 && limit < conv.val.len() {
        &conv.val[..limit]
    } else {
        &conv.val[..]
    };
    let (mut z, mut y) = (Vec::new(), Vec::new());
    for chunk in val.chunks(256) {
        let hb = conv_batch(cfg, conv, chunk, CONV_VAL_MS)?;
        let f = model::forward(p, cfg, &hb.arrays(cfg), qat)?;
        z.extend(host_f32(&f.logit)?.into_iter().map(|v| v as f64));
        y.extend(hb.y.iter().map(|&v| v > 0.5));
    }
    let bce = z
        .iter()
        .zip(&y)
        .map(|(&zr, &yr)| zr.max(0.0) + (-zr.abs()).exp().ln_1p() - if yr { zr } else { 0.0 })
        .sum::<f64>()
        / z.len().max(1) as f64;
    Ok(ConvMetrics {
        auc: auc(&z, &y),
        bce,
        n: z.len(),
    })
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

    fn weights(&self) -> LossWeights {
        if self.hard_only {
            LossWeights::HARD
        } else {
            LossWeights::DISTIL
        }
    }
}

/// Progress saved with a checkpoint.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TurnState {
    pub step: u64,
    pub elapsed_secs: f64,
    pub qat: bool,
    pub config: TurnConfig,
    pub opt: OptConfig,
    pub optimizer_step: u64,
    pub weights: LossWeights,
}

struct Session {
    cfg: TurnConfig,
    params: ParamMap,
    opt: Optimizer,
    state: TurnState,
}

impl Session {
    fn save(&self, dir: &Path, params: Option<&ParamMap>) -> Result<()> {
        let tmp = PathBuf::from(format!("{}.tmp", dir.display()));
        if tmp.exists() {
            std::fs::remove_dir_all(&tmp)?;
        }
        std::fs::create_dir_all(&tmp)?;
        params::save(
            &tmp.join("model.safetensors"),
            params.unwrap_or(&self.params),
            None,
        )?;
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
        let state: TurnState = serde_json::from_str(
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

/// The weights and config of a run checkpoint (`<run>/last`, `last-ema`, `pre-qat`).
pub fn load_run(dir: &Path) -> Result<(TurnConfig, ParamMap)> {
    let cfg: TurnConfig = serde_json::from_str(
        &std::fs::read_to_string(dir.join("config.json"))
            .with_context(|| format!("reading {}/config.json", dir.display()))?,
    )?;
    Ok((cfg, params::load(&dir.join("model.safetensors"))?))
}

/// Validation metrics against the labels (τ 0.5 on `σ(z)`), and the agreement with the teacher.
#[derive(Debug, Clone, Copy, Default, Serialize)]
pub struct ValMetrics {
    pub acc: f64,
    pub bce: f64,
    /// Share of clips where the student and the teacher take the same decision at 0.5.
    pub agree: f64,
    pub n: usize,
}

pub fn validate(p: &ParamMap, cfg: &TurnConfig, data: &Data, qat: bool) -> Result<ValMetrics> {
    validate_first(p, cfg, data, qat, 0)
}

/// [`validate`] on the first `limit` validation clips (0: all).
pub fn validate_first(
    p: &ParamMap,
    cfg: &TurnConfig,
    data: &Data,
    qat: bool,
    limit: usize,
) -> Result<ValMetrics> {
    let val = if limit > 0 && limit < data.val.len() {
        &data.val[..limit]
    } else {
        &data.val[..]
    };
    let (mut correct, mut bce, mut agree, mut n_t, mut n) = (0usize, 0.0f64, 0usize, 0usize, 0);
    for chunk in val.chunks(256) {
        let clips: Vec<_> = chunk.iter().map(|&i| &data.store.clips[i]).collect();
        let hb = host_batch(cfg, &data.store, &clips, None)?;
        let f = model::forward(p, cfg, &hb.arrays(cfg), qat)?;
        let z = host_f32(&f.logit)?;
        for (r, &zr) in z.iter().enumerate() {
            let y = hb.y[r] > 0.5;
            correct += ((zr > 0.0) == y) as usize;
            let zr = zr as f64;
            bce += zr.max(0.0) + (-zr.abs()).exp().ln_1p() - if y { zr } else { 0.0 };
            if hb.tmask[r] > 0.0 {
                agree += ((zr > 0.0) == (hb.zt[r] > 0.0)) as usize;
                n_t += 1;
            }
        }
        n += z.len();
    }
    let n_ = n.max(1) as f64;
    Ok(ValMetrics {
        acc: correct as f64 / n_,
        bce: bce / n_,
        agree: agree as f64 / n_t.max(1) as f64,
        n,
    })
}

fn log_val(
    log: &mut JsonLog,
    step: u64,
    mode: &str,
    m: &ValMetrics,
    c: Option<&ConvMetrics>,
) -> Result<()> {
    println!(
        "step {step:>6} val {mode}: acc {:.4} bce {:.4} teacher agreement {:.4} ({} clips)",
        m.acc, m.bce, m.agree, m.n
    );
    let Some(c) = c else {
        return log.write(&json!({"step": step, "mode": mode, "val": m}));
    };
    // §7: the checkpoint loss, the mean of the two validation BCEs
    let ckpt = 0.5 * (m.bce + c.bce);
    println!(
        "step {step:>6} val {mode}: conversations at d = 0.2 s: AUC {:.4} bce {:.4} ({} records); checkpoint loss {ckpt:.4}",
        c.auc, c.bce, c.n
    );
    log.write(&json!({"step": step, "mode": mode, "val": m, "conv": c, "ckpt_loss": ckpt}))
}

/// The validation of both sources: Smart Turn's clips at d = 0 and, in a v3 run, the
/// conversation records at d = 0.2 s.
fn val_both(
    p: &ParamMap,
    cfg: &TurnConfig,
    data: &Data,
    conv: Option<&Conv>,
    qat: bool,
    limit: usize,
) -> Result<(ValMetrics, Option<ConvMetrics>)> {
    let m = validate_first(p, cfg, data, qat, limit)?;
    let c = conv
        .map(|c| validate_conv(p, cfg, c, qat, limit))
        .transpose()?;
    Ok((m, c))
}

/// The batch of a training step: a function of the seed and the step.
fn step_batch(
    cfg: &TurnConfig,
    data: &Data,
    order: &mut Order,
    seed: u64,
    step: u64,
    batch: usize,
    aug: Option<&Augment>,
) -> Result<HostBatch> {
    let slots = order.batch(step, batch);
    let clips: Vec<_> = slots
        .iter()
        .map(|&s| &data.store.clips[data.train[s]])
        .collect();
    let aseed = seed ^ hash_str(&format!("turn/step/{step}"));
    host_batch(cfg, &data.store, &clips, aug.map(|a| (a, aseed)))
}

pub fn train(a: &TrainArgs) -> Result<()> {
    if a.cpu {
        mlx_rs::with_device(mlx_rs::Device::cpu(), || train_on(a))
    } else {
        train_on(a)
    }
}

fn train_on(a: &TrainArgs) -> Result<()> {
    let progress = Progress::from_flags(a.minutes, a.steps)?;
    let last = a.out.join("last");
    let resumed = a.resume && last.join("state.json").exists();
    let mut s = if resumed {
        let s = Session::load(&last)?;
        println!("resumed from {} at step {}", last.display(), s.state.step);
        if s.state.weights != a.weights() {
            bail!("{} was trained with other loss weights", last.display());
        }
        s
    } else {
        if a.out.join("last/state.json").exists() {
            bail!(
                "{} already holds a run: --resume continues it",
                a.out.display()
            );
        }
        let mut cfg = TurnConfig::load(&a.config, a.rung.as_deref())?;
        cfg.qat_grad_a = a.qat_grad_a;
        let mut params = model::init(&cfg, a.seed)?;
        if let Some(dir) = &a.init {
            // s6-ear v3 (§4, §6): the weights of a checkpoint of the same graph, its files the
            // recorded ones; the optimizer starts fresh
            let list = a.init_sha256.as_ref().ok_or_else(|| {
                anyhow::anyhow!("--init needs --init-sha256 (runs/s6-ear3-nano/init.json)")
            })?;
            check_init(dir, list)?;
            let (icfg, ip) = load_run(dir)?;
            if !same_graph(&icfg, &cfg) {
                bail!(
                    "{}: not the graph of {} ({icfg:?})",
                    dir.display(),
                    a.config.display()
                );
            }
            let n = params::load_matching(&mut params, &ip)?;
            if n != params.len() || n != ip.len() {
                bail!(
                    "{}: {} tensors, {n} of them in the {} of the graph",
                    dir.display(),
                    ip.len(),
                    params.len()
                );
            }
            println!(
                "weights from {} ({n} tensors, sha256 as {}); a fresh optimizer",
                dir.display(),
                list.display()
            );
        }
        Session {
            state: TurnState {
                step: 0,
                elapsed_secs: 0.0,
                qat: false,
                config: cfg.clone(),
                opt: a.opt(),
                optimizer_step: 0,
                weights: a.weights(),
            },
            cfg,
            params,
            opt: Optimizer::new(a.opt()),
        }
    };
    s.cfg.qat_grad_a = a.qat_grad_a;
    let cfg = s.cfg.clone();
    let lw = a.weights();
    std::fs::create_dir_all(&a.out)?;
    std::fs::write(a.out.join("args.json"), serde_json::to_string_pretty(a)?)?;
    let data = Data::open(&a.data, &cfg, a.max_train, a.max_val)?;
    println!(
        "s6-ear v2 {} (rung {}): {} parameters ({} exported), {} positions of {} values",
        cfg.name,
        cfg.rung.as_deref().unwrap_or("-"),
        params::count(&s.params),
        model::exported_count(&cfg),
        cfg.max_positions(),
        cfg.input_width()
    );
    println!(
        "data {}: {} train ({} with teacher targets), {} val, {} skipped; loss {lw:?}",
        a.data.display(),
        data.train.len(),
        data.with_teacher(&data.train),
        data.val.len(),
        data.skipped
    );
    if lw != LossWeights::HARD && data.with_teacher(&data.train) < data.train.len() {
        bail!("distillation needs the teacher's targets of every training part (turn-teacher.py)");
    }
    // s6-ear v3: the conversation records and the Smart Turn extension (§3.6)
    let v3: Option<(Conv, Ext)> = match &a.conv {
        None => None,
        Some(c) => {
            let e = a
                .st_ext
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("--conv needs --st-ext (sky-data turn-extend)"))?;
            let conv = Conv::open(c, a.conv_max_train, a.conv_max_val)?;
            let ext = Ext::open(e)?;
            println!(
                "conversations {}: {} hold and {} end-of-turn training records, {} val, {} skipped; extension {}: {} clips; share {}, d = 0 on {} of the Smart Turn rows",
                c.display(),
                conv.train[0].len(),
                conv.train[1].len(),
                conv.val.len(),
                conv.skipped,
                e.display(),
                ext.by_id.len(),
                a.conv_share,
                a.st_d0
            );
            Some((conv, ext))
        }
    };
    let mut log = JsonLog::open(&a.out.join("log.jsonl"))?;
    log.write(&json!({"step": s.state.step, "data": {
        "train": data.train.len(), "val": data.val.len(), "skipped": data.skipped,
        "with_teacher": data.with_teacher(&data.train),
        "conv": v3.as_ref().map(|(c, e)| json!({"train_hold": c.train[0].len(),
            "train_eot": c.train[1].len(), "val": c.val.len(), "skipped": c.skipped,
            "ext": e.by_id.len(), "conv_share": a.conv_share, "st_d0": a.st_d0})),
        "init": a.init}, "weights": lw}))?;
    let aug = (!a.no_augment).then_some(Augment::PREREG);
    // the batches are prepared ahead by a producer thread, deterministically in the step
    let (tx, rx) = std::sync::mpsc::sync_channel::<Result<HostBatch>>(4);
    let data = std::sync::Arc::new(data);
    let v3 = std::sync::Arc::new(v3);
    let producer = {
        let (cfg, data, v3, seed, batch, start) = (
            cfg.clone(),
            data.clone(),
            v3.clone(),
            a.seed,
            a.batch,
            s.state.step,
        );
        let mix = Mix {
            conv_share: a.conv_share,
            st_d0: a.st_d0,
        };
        std::thread::spawn(move || {
            let mut order = Order::new(seed, data.train.len());
            let mut step = start;
            // the Smart Turn rows of a v3 run take the epoch order where the earlier steps left it
            let mut st_k = match &*v3 {
                Some(_) => st_draws_before(seed, start, batch, mix.conv_share),
                None => 0,
            };
            loop {
                let b = match &*v3 {
                    None => step_batch(&cfg, &data, &mut order, seed, step, batch, aug.as_ref()),
                    Some((conv, ext)) => {
                        let src = Sources {
                            st: &data,
                            ext,
                            conv,
                            mix,
                            st_hard: lw.hard,
                        };
                        mixed_batch(
                            &cfg,
                            &src,
                            seed,
                            step,
                            batch,
                            aug.as_ref(),
                            &mut order,
                            &mut st_k,
                        )
                        .map(|(b, _)| b)
                    }
                };
                if tx.send(b).is_err() {
                    break;
                }
                step += 1;
            }
        })
    };
    let conv = (*v3).as_ref().map(|(c, _)| c);
    let mut ema: Option<ParamMap> = (a.ema > 0.0).then(|| params::deep_clone(&s.params));
    let ema_dir = a.out.join("last-ema");
    if resumed && a.ema > 0.0 && ema_dir.join("model.safetensors").exists() {
        ema = Some(params::load(&ema_dir.join("model.safetensors"))?);
    }
    let stop_at = stop_step(s.state.step, a.stop_after)?;
    let base = s.state.elapsed_secs;
    let mut t0 = Instant::now();
    let mut brake = Brake::from_env()?;
    let mut last_ckpt = Instant::now();
    let (mut wl, mut wn, mut wtime) = (0.0f64, 0u64, Instant::now());
    let mut consecutive = 0;
    let n_train = data.train.len() as f64;
    while keep_going(&progress, stop_at, s.state.step, s.state.elapsed_secs) {
        brake.rest(&mut t0);
        let frac = progress.frac(s.state.step, s.state.elapsed_secs);
        if !s.state.qat && qat_due(frac, a.qat_frac) {
            println!("step {}: starting ternary QAT", s.state.step);
            let f = val_both(&s.params, &cfg, &data, conv, false, 0)?;
            let t = val_both(&s.params, &cfg, &data, conv, true, 0)?;
            log_val(&mut log, s.state.step, "f32", &f.0, f.1.as_ref())?;
            log_val(&mut log, s.state.step, "ternary", &t.0, t.1.as_ref())?;
            s.save(&a.out.join("pre-qat"), None)?;
            s.state.qat = true;
        }
        let hb = rx.recv().context("the batch producer stopped")??;
        let b = hb.arrays(&cfg);
        let lr_scale = wsd(frac) as f32;
        let qat = s.state.qat;
        let (loss, gnorm) = train_step(&mut s.params, &mut s.opt, &b, lr_scale, |p, b| {
            model::loss(p, &cfg, b, qat, lw)
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
                "clips_s": (wn as usize * a.batch) as f64 / dt,
                "epochs": step as f64 * a.batch as f64 / n_train,
                "elapsed_min": s.state.elapsed_secs / 60.0}),
            )?;
            println!(
                "step {step:>6} loss {:.4} lr x{lr_scale:.3}{} {:.0} clips/s",
                wl / wn as f64,
                if qat { " qat" } else { "" },
                (wn as usize * a.batch) as f64 / dt
            );
            (wl, wn, wtime) = (0.0, 0, Instant::now());
        }
        if step % a.eval_every == 0 {
            let (m, c) = val_both(&s.params, &cfg, &data, conv, qat, a.curve_val)?;
            log_val(
                &mut log,
                step,
                if qat { "ternary" } else { "f32" },
                &m,
                c.as_ref(),
            )?;
        }
        if last_ckpt.elapsed().as_secs_f64() >= a.checkpoint_minutes * 60.0 {
            s.save(&last, None)?;
            if let Some(e) = &ema {
                s.save(&ema_dir, Some(e))?;
            }
            last_ckpt = Instant::now();
        }
    }
    drop(rx);
    producer.join().expect("batch producer");
    s.save(&last, None)?;
    if let Some(e) = &ema {
        s.save(&ema_dir, Some(e))?;
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
    let finals: Vec<(&str, &ParamMap, bool)> =
        [("f32", &s.params, false), ("ternary", &s.params, true)]
            .into_iter()
            .chain(
                ema.iter()
                    .flat_map(|e| [("ema-f32", e, false), ("ema-ternary", e, true)]),
            )
            .collect();
    for (mode, p, qat) in finals {
        let (m, c) = val_both(p, &cfg, &data, conv, qat, 0)?;
        log_val(&mut log, s.state.step, mode, &m, c.as_ref())?;
    }
    log.write(&json!({"step": s.state.step, "final": true,
        "elapsed_min": s.state.elapsed_secs / 60.0,
        "epochs": s.state.step as f64 * a.batch as f64 / n_train}))?;
    Ok(())
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::test_util;
    use sky_infer::audio::TurnFrontEnd;
    use sky_infer::turn::store::{encode_hops, encode_teacher, feat_name, manifest_name};
    use sky_infer::turn::store::{teacher_done_name, teacher_name};
    use sky_infer::turn::{TEACHER_DIM, TEACHER_STEPS};

    /// A tiny store whose label is audible: finished clips end in a falling tone, the others in a
    /// rising one; every clip has teacher targets.
    pub(crate) fn tiny_store(dir: &Path, n: usize) {
        std::fs::create_dir_all(dir).unwrap();
        let mut fe = TurnFrontEnd::new();
        let (mut feat, mut man, mut teach) = (Vec::new(), String::new(), Vec::new());
        for i in 0..n {
            let end = i % 2 == 0;
            let secs = 1.0 + (i % 5) as f32 * 0.7;
            let len = (16_000.0 * secs) as usize;
            let mut ph = 0.0f32;
            let pcm: Vec<f32> = (0..len)
                .map(|k| {
                    let u = k as f32 / len as f32;
                    let f = if end {
                        220.0 - 90.0 * u
                    } else {
                        130.0 + 90.0 * u
                    };
                    ph += f / 16_000.0;
                    0.3 * (std::f32::consts::TAU * ph).sin()
                })
                .collect();
            let (h, hops) = fe.hops(&pcm, 400);
            let off = feat.len();
            feat.extend(encode_hops(&h));
            let split = if i % 10 == 9 { "val" } else { "train" };
            man.push_str(&format!(
                "{{\"id\":\"c{i}\",\"hops\":{hops},\"offset\":{off},\"split\":\"{split}\",\"label\":{end}}}\n"
            ));
            let z = if end { 2.0 } else { -2.0 };
            teach.extend(encode_teacher(
                z,
                &vec![z * 0.1; TEACHER_DIM],
                &vec![z * 0.05; TEACHER_STEPS * TEACHER_DIM],
            ));
        }
        std::fs::write(dir.join(feat_name("00000")), feat).unwrap();
        std::fs::write(dir.join(teacher_name("00000")), teach).unwrap();
        std::fs::write(dir.join(teacher_done_name("00000")), "{}").unwrap();
        std::fs::write(dir.join(manifest_name("00000")), man).unwrap();
    }

    pub(crate) fn args(dir: &Path) -> TrainArgs {
        TrainArgs {
            config: Path::new(env!("CARGO_MANIFEST_DIR")).join("../../configs/s6-ear2-femto.json"),
            rung: None,
            data: dir.join("store"),
            out: dir.join("run"),
            minutes: None,
            steps: Some(40),
            stop_after: None,
            resume: false,
            batch: 8,
            seed: 1,
            hard_only: false,
            qat_frac: 0.25,
            qat_grad_a: 0.0,
            ema: 0.9,
            no_augment: false,
            max_train: 0,
            max_val: 0,
            log_every: 10,
            eval_every: 20,
            curve_val: 3,
            checkpoint_minutes: 100.0,
            adam_lr: 3e-3,
            muon_lr: None,
            weight_decay: 0.01,
            clip_norm: 1.0,
            no_muon: true,
            ns_polish: 2,
            wd_mode: WdMode::Decoupled,
            cpu: true,
            conv: None,
            st_ext: None,
            conv_share: 0.5,
            st_d0: 0.25,
            conv_max_train: 0,
            conv_max_val: 0,
            init: None,
            init_sha256: None,
        }
    }

    /// A tiny conversation store (s6-ear v3): each record a tone that falls (end of turn) or
    /// rises (hold) for 2 s, then the trailing silence; three holds for one end of turn; the
    /// asks up to 2.4 s (every third record up to 0.6 s only).
    pub(crate) fn tiny_conv(dir: &Path, n: usize) {
        std::fs::create_dir_all(dir).unwrap();
        let mut fe = TurnFrontEnd::new();
        let (mut feat, mut man) = (Vec::new(), String::new());
        for i in 0..n {
            let eot = i % 4 == 0;
            let o = 32_000usize;
            let dmax = if i % 3 == 2 { 600 } else { 2400 };
            let mut ph = 0.0f32;
            let mut pcm: Vec<f32> = (0..o)
                .map(|k| {
                    let u = k as f32 / o as f32;
                    let f = if eot {
                        230.0 - 100.0 * u
                    } else {
                        130.0 + 100.0 * u
                    };
                    ph += f / 16_000.0;
                    0.3 * (std::f32::consts::TAU * ph).sin()
                })
                .collect();
            pcm.extend((0..dmax * 16).map(|k| 0.001 * ((k * 7919 % 97) as f32 / 97.0 - 0.5)));
            let (h, hops) = fe.hops(&pcm, 0);
            let cut = (o - 640) / 160 + 1;
            assert_eq!(hops, cut + dmax / 10);
            let off = feat.len();
            feat.extend(encode_hops(&h));
            let split = if i % 5 == 4 { "val" } else { "train" };
            man.push_str(&format!(
                "{{\"id\":\"r{i}\",\"hops\":{hops},\"offset\":{off},\"split\":\"{split}\",\"label\":{eot},\"cut_hops\":{cut},\"dmax_ms\":{dmax}}}\n"
            ));
        }
        std::fs::write(dir.join(feat_name("00000")), feat).unwrap();
        std::fs::write(dir.join(manifest_name("00000")), man).unwrap();
    }

    /// The extension store of [`tiny_store`]'s clips: 240 hops of a quiet background each.
    pub(crate) fn tiny_ext(dir: &Path, n: usize) {
        std::fs::create_dir_all(dir).unwrap();
        let mut fe = TurnFrontEnd::new();
        let (mut feat, mut man) = (Vec::new(), String::new());
        for i in 0..n {
            let pcm: Vec<f32> = (0..39_040)
                .map(|k| 0.002 * (((k + i) * 104_729 % 101) as f32 / 101.0 - 0.5))
                .collect();
            let (h, hops) = fe.hops(&pcm, 240);
            assert_eq!(hops, 240);
            let off = feat.len();
            feat.extend(encode_hops(&h));
            man.push_str(&format!(
                "{{\"id\":\"c{i}\",\"hops\":240,\"offset\":{off},\"split\":\"train\"}}\n"
            ));
        }
        std::fs::write(dir.join(feat_name("00000")), feat).unwrap();
        std::fs::write(dir.join(manifest_name("00000")), man).unwrap();
    }

    fn sha256_list(dir: &Path, out: &Path, corrupt: bool) {
        let mut files = serde_json::Map::new();
        for name in ["config.json", "model.safetensors"] {
            let o = std::process::Command::new("shasum")
                .args(["-a", "256"])
                .arg(dir.join(name))
                .output()
                .unwrap();
            let mut h = String::from_utf8_lossy(&o.stdout)
                .split_whitespace()
                .next()
                .unwrap()
                .to_string();
            if corrupt && name == "model.safetensors" {
                h = "0".repeat(64);
            }
            files.insert(name.to_string(), serde_json::Value::String(h));
        }
        std::fs::write(out, serde_json::json!({"files": files}).to_string()).unwrap();
    }

    /// s6-ear v3: a CPU run on both tiny sources learns, logs the conversation validation and the
    /// checkpoint loss, resumes; `--init` starts from a checkpoint's weights (unchanged by a zero
    /// rate) and refuses one whose sha256 differ.
    #[test]
    fn a_v3_cpu_run_mixes_both_sources_resumes_and_starts_from_a_checkpoint() {
        let _g = test_util::lock();
        let dir = tempfile::tempdir().unwrap();
        tiny_store(&dir.path().join("store"), 60);
        tiny_ext(&dir.path().join("ext"), 60);
        tiny_conv(&dir.path().join("conv"), 60);
        let a = TrainArgs {
            conv: Some(dir.path().join("conv")),
            st_ext: Some(dir.path().join("ext")),
            steps: Some(60),
            qat_frac: 0.0,
            ..args(dir.path())
        };
        train(&TrainArgs {
            stop_after: Some(30),
            ..a.clone()
        })
        .unwrap();
        train(&TrainArgs {
            resume: true,
            ..a.clone()
        })
        .unwrap();
        let log = std::fs::read_to_string(a.out.join("log.jsonl")).unwrap();
        let lines: Vec<serde_json::Value> = log
            .lines()
            .filter_map(|l| serde_json::from_str(l).ok())
            .collect();
        let losses: Vec<f64> = lines.iter().filter_map(|v| v["loss"].as_f64()).collect();
        assert_eq!(losses.len(), 6);
        assert!(losses.iter().all(|l| l.is_finite()), "{losses:?}");
        let fin = lines
            .iter()
            .rev()
            .find(|v| v["mode"] == "ema-ternary")
            .unwrap();
        assert!(
            fin["conv"]["auc"].is_number() && fin["ckpt_loss"].is_number(),
            "{fin}"
        );
        assert_eq!(fin["conv"]["n"], 12);
        assert!(lines[0]["data"]["conv"]["train_eot"].as_u64().unwrap() > 0);
        // the hard labels of both sources are learnt: the loss falls
        let h = TrainArgs {
            out: dir.path().join("hard"),
            hard_only: true,
            ema: 0.0,
            steps: Some(300),
            adam_lr: 5e-3,
            log_every: 50,
            ..a.clone()
        };
        train(&h).unwrap();
        let hl: Vec<f64> = std::fs::read_to_string(h.out.join("log.jsonl"))
            .unwrap()
            .lines()
            .filter_map(|l| serde_json::from_str::<serde_json::Value>(l).ok())
            .filter_map(|v| v["loss"].as_f64())
            .collect();
        let late = hl[hl.len() - 3..].iter().sum::<f64>() / 3.0;
        assert!(late < 0.5 * hl[0], "{hl:?}");
        // --init: the weights of last-ema, the optimizer fresh; a zero rate keeps them
        let ck = a.out.join("last-ema");
        let list = dir.path().join("init.json");
        sha256_list(&ck, &list, false);
        let b = TrainArgs {
            out: dir.path().join("init-run"),
            init: Some(ck.clone()),
            init_sha256: Some(list.clone()),
            adam_lr: 0.0,
            weight_decay: 0.0,
            steps: Some(2),
            ema: 0.0,
            qat_frac: 0.0,
            ..a.clone()
        };
        train(&b).unwrap();
        let (p0, p1) = (
            params::load(&ck.join("model.safetensors")).unwrap(),
            params::load(&b.out.join("last/model.safetensors")).unwrap(),
        );
        assert_eq!(p0.len(), p1.len());
        for (k, v) in &p0 {
            assert_eq!(host_f32(v).unwrap(), host_f32(&p1[k]).unwrap(), "{k}");
        }
        // a checkpoint whose files differ from the list is refused, and --init needs the list
        let bad = dir.path().join("bad.json");
        sha256_list(&ck, &bad, true);
        let e = train(&TrainArgs {
            out: dir.path().join("bad-run"),
            init_sha256: Some(bad),
            ..b.clone()
        })
        .unwrap_err();
        assert!(format!("{e:#}").contains("refuses"), "{e:#}");
        assert!(train(&TrainArgs {
            out: dir.path().join("no-list"),
            init_sha256: None,
            ..b.clone()
        })
        .is_err());
    }

    /// A short CPU run on a tiny store: it learns, logs what the scripts read, resumes, and the
    /// hard-label arm runs too (sanity, not a training run).
    #[test]
    fn short_cpu_run_learns_logs_and_resumes() {
        let _g = test_util::lock();
        let dir = tempfile::tempdir().unwrap();
        tiny_store(&dir.path().join("store"), 60);
        let a = args(dir.path());
        train(&TrainArgs {
            stop_after: Some(20),
            ..a.clone()
        })
        .unwrap();
        let st: TurnState =
            serde_json::from_str(&std::fs::read_to_string(a.out.join("last/state.json")).unwrap())
                .unwrap();
        assert_eq!(st.step, 20);
        assert!(train(&a).is_err(), "an existing run needs --resume");
        train(&TrainArgs {
            resume: true,
            ..a.clone()
        })
        .unwrap();
        let log = std::fs::read_to_string(a.out.join("log.jsonl")).unwrap();
        let losses: Vec<f64> = log
            .lines()
            .filter_map(|l| serde_json::from_str::<serde_json::Value>(l).ok())
            .filter_map(|v| v["loss"].as_f64())
            .collect();
        assert_eq!(losses.len(), 4);
        assert!(losses[3] < losses[0], "{losses:?}");
        assert!(log.contains("\"final\":true") && log.contains("ema-ternary"));
        assert!(a.out.join("pre-qat/model.safetensors").exists());
        assert!(a.out.join("last-ema/model.safetensors").exists());
        let out = dir.path().join("hard");
        train(&TrainArgs {
            out: out.clone(),
            hard_only: true,
            qat_frac: 0.0,
            ema: 0.0,
            steps: Some(10),
            ..a.clone()
        })
        .unwrap();
        assert!(std::fs::read_to_string(out.join("log.jsonl"))
            .unwrap()
            .contains("\"hard\":1.0"));
    }
}
