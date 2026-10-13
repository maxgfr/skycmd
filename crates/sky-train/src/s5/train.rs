//! `sky-train s5-train`: fine-tuning of an imported encoder (Ettin, mmBERT) into a decision
//! model, by steps (not time), so that every variant and seed sees the same number of updates.
//!
//! Recipe: AdamW everywhere (no Muon), the fresh head at `head_lr_mult` times the encoder rate,
//! linear warmup then linear decay, an exponential moving average of the weights (evaluated and
//! exported), loss = soft CE + Brier + RPS on `score` questions (+ symmetric KL on pairs), no
//! label smoothing.
//!
//! Variants: **A** is the Laya layout with the records' option names; **A′** the same layout
//! with the shared augmentation's random names and polarity swaps rendered (see
//! [`crate::s5::data`]).
//!
//! A checkpoint `<out>/last` holds `model.safetensors` (the EMA weights, in the s1 names, so
//! that `sky-train export` turns variant A into the Laya / oscar layout), `raw.safetensors`
//! and `optimizer.safetensors` (to resume), `config.json`, `tokenizer.json`,
//! `rl_agent_config.json` and `s5.json` (the training settings).

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{bail, Context, Result};
use mlx_rs::{transforms::eval, Array};
use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::config::ModelConfig;
use crate::metrics::{self, Pred};
use crate::model::{self, DecisionBatch};
use crate::optim::{warmup_linear, OptConfig, Optimizer};
use crate::params::{self, ParamMap, Parts};
use crate::rng::Rng;
use crate::s5::data::{
    self, draw, laya_batch, laya_row, link_pairs, native_tokens, Augment, Draw, LayaBatch, S5Item,
};
use crate::s5::loss::{self, LossWeights, Pairs};
use crate::s5::score::read_s5;
use crate::sequence::Tok;
use crate::session::{apply_update, sorted_names, CompiledGrad};
use crate::train::{rl_agent_json, JsonLog};

/// Model variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Variant {
    /// Laya layout, the records' own option names.
    A,
    /// Laya layout, random names and polarity swaps rendered.
    APrime,
}

impl Variant {
    pub fn parse(s: &str) -> Result<Self> {
        Ok(match s {
            "a" => Variant::A,
            "a-prime" | "a'" => Variant::APrime,
            other => bail!("unknown variant {other:?} (a, a-prime)"),
        })
    }

    /// Whether the augmentation's names are rendered.
    fn renders_names(self) -> bool {
        matches!(self, Variant::APrime)
    }
}

#[derive(Debug, Clone, clap::Args, Serialize, Deserialize)]
pub struct TrainArgs {
    /// Imported checkpoint (`sky-train import-hf`): model.safetensors, config.json, tokenizer.json.
    #[arg(long)]
    pub init: PathBuf,
    /// s5 records (JSONL files or directories, comma separated): split `train` and `val`.
    #[arg(long, value_delimiter = ',')]
    pub records: Vec<PathBuf>,
    #[arg(long)]
    pub out: PathBuf,
    /// a | a-prime
    #[arg(long, default_value = "a")]
    pub variant: String,
    #[arg(long, default_value_t = 3000)]
    pub steps: u64,
    #[arg(long, default_value_t = 32)]
    pub batch: usize,
    /// Peak AdamW learning rate of the encoder.
    #[arg(long, default_value_t = 5e-5)]
    pub lr: f32,
    /// Learning-rate multiplier of the freshly initialized head.
    #[arg(long, default_value_t = 20.0)]
    pub head_lr_mult: f32,
    /// Warmup fraction of the steps.
    #[arg(long, default_value_t = 0.06)]
    pub warmup: f64,
    #[arg(long, default_value_t = 0.01)]
    pub weight_decay: f32,
    /// EMA decay of the weights (0 disables).
    #[arg(long, default_value_t = 0.999)]
    pub ema: f32,
    #[arg(long, default_value_t = 1.0)]
    pub brier: f32,
    #[arg(long, default_value_t = 0.25)]
    pub rps: f32,
    /// Weight of the symmetric KL between pair members.
    #[arg(long, default_value_t = 0.0)]
    pub kl: f32,
    #[arg(long, default_value_t = 0.5)]
    pub p_names: f64,
    #[arg(long, default_value_t = 0.25)]
    pub p_polarity: f64,
    #[arg(long, default_value_t = 512)]
    pub max_len: usize,
    #[arg(long, default_value_t = 192)]
    pub head_max_len: usize,
    #[arg(long, default_value_t = 0)]
    pub seed: u64,
    #[arg(long, default_value_t = 50)]
    pub log_every: u64,
    #[arg(long, default_value_t = 500)]
    pub eval_every: u64,
    /// Validation questions per evaluation.
    #[arg(long, default_value_t = 2000)]
    pub val_max: usize,
    #[arg(long, default_value_t = 1000)]
    pub checkpoint_every: u64,
    /// Continue from <out>/last.
    #[arg(long)]
    pub resume: bool,
    #[arg(long)]
    pub bf16: bool,
}

/// Progress saved with a checkpoint.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct State {
    step: u64,
    rng: Rng,
    args: TrainArgs,
    elapsed_secs: f64,
}

/// Exponential moving average of the parameters.
pub fn ema_update(ema: &mut ParamMap, p: &ParamMap, decay: f32) -> Result<()> {
    let d = Array::from_f32(decay);
    let e = Array::from_f32(1.0 - decay);
    for (k, v) in ema.iter_mut() {
        *v = v.multiply(&d)?.add(p[k].multiply(&e)?)?;
    }
    eval(ema.values())?;
    Ok(())
}

fn decision_batch(b: &LayaBatch) -> DecisionBatch {
    DecisionBatch {
        ids: b.ids.clone(),
        key_valid: Some(b.key_valid.clone()),
        qtype: b.qtype.clone(),
        marker_pos: b.marker_pos.clone(),
        marker_mask: b.marker_mask.clone(),
        target: Array::from_f32(0.0),
        n_tokens: b.n_tokens,
    }
}

/// Batch arrays in the order of the compiled function.
fn batch_arrays(b: &LayaBatch) -> Vec<Array> {
    vec![
        b.ids.clone(),
        b.key_valid.clone(),
        b.qtype.clone(),
        b.marker_pos.clone(),
        b.marker_mask.clone(),
        b.target.clone(),
        b.is_score.clone(),
        b.k.clone(),
        b.pairs.left.clone(),
        b.pairs.right.clone(),
        b.pairs.map.clone(),
        b.pairs.mask.clone(),
        b.pairs.weight.clone(),
    ]
}

fn compiled(cfg: &ModelConfig, names: Vec<std::rc::Rc<str>>, w: LossWeights) -> CompiledGrad {
    let cfg = cfg.clone();
    CompiledGrad::new(names, move |p, b| {
        let batch = DecisionBatch {
            ids: b[0].clone(),
            key_valid: Some(b[1].clone()),
            qtype: b[2].clone(),
            marker_pos: b[3].clone(),
            marker_mask: b[4].clone(),
            target: b[5].clone(),
            n_tokens: 0,
        };
        let logits = model::decision_forward(p, &cfg, &batch)?;
        let pairs = Pairs {
            left: b[8].clone(),
            right: b[9].clone(),
            map: b[10].clone(),
            mask: b[11].clone(),
            weight: b[12].clone(),
        };
        loss::total(&logits, &b[5], &b[6], &b[7], Some(&pairs), &w)
    })
}

/// The rows of a training batch: `batch` sampled items (and their pair partners when the
/// KL is on), each with its own augmentation draw.
#[allow(clippy::too_many_arguments)]
fn sample(
    train: &[S5Item],
    a: &TrainArgs,
    variant: Variant,
    aug: &Augment,
    rng: &mut Rng,
    tok: &Tok,
    cfg: &ModelConfig,
    w: &LossWeights,
) -> Result<LayaBatch> {
    let mut rows = Vec::with_capacity(2 * a.batch);
    let mut pairs = Vec::new();
    let push = |item: &S5Item, d: &Draw, rows: &mut Vec<data::LayaRow>| -> Result<bool> {
        let opts = if variant.renders_names() {
            native_tokens(item, d, tok, &cfg.special)?
        } else {
            item.native.clone()
        };
        let r = laya_row(
            item,
            d,
            &opts,
            cfg.rl_agent.max_len,
            cfg.rl_agent.head_max_len,
            &cfg.special,
        );
        if r.markers.len() != item.k() {
            return Ok(false);
        }
        rows.push(r);
        Ok(true)
    };
    while rows.len() < a.batch {
        let i = rng.below(train.len());
        let d = draw(&train[i], aug, rng);
        let partner = train[i].pair.filter(|_| w.kl != 0.0);
        let dp = partner.map(|j| draw(&train[j], aug, rng));
        let at = rows.len();
        if !push(&train[i], &d, &mut rows)? {
            continue;
        }
        if let (Some(j), Some(dp)) = (partner, dp) {
            if push(&train[j], &dp, &mut rows)? {
                pairs.push((at, at + 1));
            }
        }
    }
    Ok(laya_batch(&rows, &pairs))
}

/// Logits of `items` (identity order, the records' names) under `params`.
pub fn predict(
    params: &ParamMap,
    cfg: &ModelConfig,
    items: &[&S5Item],
    max_tokens: usize,
) -> Result<Vec<Pred>> {
    let mut order: Vec<usize> = (0..items.len()).collect();
    let rows: Vec<data::LayaRow> = items
        .iter()
        .map(|it| {
            let d = Draw {
                order: (0..it.k()).collect(),
                names: None,
            };
            laya_row(
                it,
                &d,
                &it.native,
                cfg.rl_agent.max_len,
                cfg.rl_agent.head_max_len,
                &cfg.special,
            )
        })
        .collect();
    order.sort_by_key(|&i| rows[i].ids.len());
    let mut out: Vec<Option<Pred>> = vec![None; items.len()];
    let mut start = 0;
    while start < order.len() {
        let mut end = start + 1;
        while end < order.len()
            && rows[order[end]].ids.len().div_ceil(16) * 16 * (end - start + 1) <= max_tokens
        {
            end += 1;
        }
        let sel: Vec<data::LayaRow> = order[start..end].iter().map(|&i| rows[i].clone()).collect();
        let b = laya_batch(&sel, &[]);
        let logits = model::decision_forward(params, cfg, &decision_batch(&b))?;
        let k = logits.dim(1) as usize;
        let flat = model::host_f32(&logits)?;
        for (r, &i) in order[start..end].iter().enumerate() {
            let it = items[i];
            let kk = rows[i].markers.len();
            out[i] = Some(Pred {
                logits: flat[r * k..r * k + kk].to_vec(),
                target: it.target.clone(),
                qtype: it.qtype,
                src: it.rec.src.clone(),
                group: it.rec.lang.clone(),
            });
        }
        start = end;
    }
    Ok(out.into_iter().flatten().collect())
}

/// Accuracy, NLL, Brier and ECE (T = 1) per source and overall.
pub fn report(preds: &[Pred]) -> BTreeMap<String, metrics::Metrics> {
    crate::train::grouped_metrics(preds, |p| p.src.clone(), |_| 1.0)
}

fn save(
    dir: &Path,
    raw: &ParamMap,
    ema: &ParamMap,
    opt: &Optimizer,
    cfg: &ModelConfig,
    state: &State,
    tok_json: &[u8],
) -> Result<()> {
    let tmp = PathBuf::from(format!("{}.tmp", dir.display()));
    if tmp.exists() {
        std::fs::remove_dir_all(&tmp)?;
    }
    std::fs::create_dir_all(&tmp)?;
    params::save(&tmp.join("model.safetensors"), ema, None)?;
    params::save(&tmp.join("raw.safetensors"), raw, None)?;
    params::save(
        &tmp.join("optimizer.safetensors"),
        &opt.state_arrays(),
        None,
    )?;
    let mut st = serde_json::to_value(state)?;
    st["optimizer_step"] = opt.step.into();
    st["optimizer"] = serde_json::to_value(&opt.cfg)?;
    std::fs::write(tmp.join("s5.json"), serde_json::to_string_pretty(&st)?)?;
    std::fs::write(tmp.join("config.json"), serde_json::to_string_pretty(cfg)?)?;
    std::fs::write(tmp.join("tokenizer.json"), tok_json)?;
    let rl = rl_agent_json(
        cfg,
        json!({"training": {"steps": state.step, "variant": state.args.variant,
               "recipe": "s5: AdamW, warmup + linear decay, EMA, CE + Brier + RPS"}}),
    );
    std::fs::write(
        tmp.join("rl_agent_config.json"),
        serde_json::to_string_pretty(&rl)?,
    )?;
    if dir.exists() {
        std::fs::remove_dir_all(dir)?;
    }
    std::fs::rename(&tmp, dir).with_context(|| format!("renaming to {}", dir.display()))?;
    Ok(())
}

pub fn train(a: &TrainArgs) -> Result<()> {
    let variant = Variant::parse(&a.variant)?;
    let last = a.out.join("last");
    let resume = a.resume && last.join("s5.json").exists();
    let mut cfg: ModelConfig = serde_json::from_str(
        &std::fs::read_to_string(a.init.join("config.json")).context("init config.json")?,
    )?;
    cfg.rl_agent.max_len = a.max_len;
    cfg.rl_agent.head_max_len = a.head_max_len;
    cfg.bf16 = a.bf16;
    cfg.validate()?;
    let tok_path = a.init.join("tokenizer.json");
    let tok = Tok::load_with(&tok_path, &cfg.special)?;
    let tok_json = std::fs::read(&tok_path)?;

    let t_load = Instant::now();
    let recs = read_s5(&a.records)?;
    let n_all = recs.len();
    let (mut train, mut val) = (Vec::new(), Vec::new());
    for r in recs {
        match r.split.as_str() {
            "train" => train.push(r),
            "val" => val.push(r),
            _ => {}
        }
    }
    let mut train = data::tokenize(&tok, &cfg.special, train)?;
    let val = data::tokenize(&tok, &cfg.special, val)?;
    link_pairs(&mut train);
    let n_pairs = train.iter().filter(|i| i.pair.is_some()).count() / 2;
    println!(
        "{} records: {} train ({n_pairs} pairs), {} val, others ignored ({:.1}s)",
        n_all,
        train.len(),
        val.len(),
        t_load.elapsed().as_secs_f64()
    );
    if train.is_empty() {
        bail!("no training records");
    }

    let opt_cfg = OptConfig {
        adam_lr: a.lr,
        weight_decay: a.weight_decay,
        muon: false,
        head_lr_mult: a.head_lr_mult,
        beta2: 0.999,
        ..OptConfig::default()
    };
    let (mut raw, mut ema, mut opt, mut st) = if resume {
        let text = std::fs::read_to_string(last.join("s5.json"))?;
        let v: serde_json::Value = serde_json::from_str(&text)?;
        let st: State = serde_json::from_value(v.clone())?;
        let mut opt = Optimizer::new(serde_json::from_value(v["optimizer"].clone())?);
        opt.load_state(
            v["optimizer_step"].as_u64().unwrap_or(0),
            &params::load(&last.join("optimizer.safetensors"))?,
        )?;
        println!("resumed {} at step {}", last.display(), st.step);
        (
            params::load(&last.join("raw.safetensors"))?,
            params::load(&last.join("model.safetensors"))?,
            opt,
            st,
        )
    } else {
        let mut p = params::init(&cfg, Parts::DECISION, a.seed)?;
        let src = params::load(&a.init.join("model.safetensors"))?;
        let n = params::load_matching(&mut p, &src)?;
        let n_enc = p.keys().filter(|k| k.starts_with("encoder.")).count();
        if n < n_enc {
            bail!(
                "{} provides {n} of the {n_enc} encoder tensors",
                a.init.display()
            );
        }
        let ema = params::deep_clone(&p);
        (
            p,
            ema,
            Optimizer::new(opt_cfg),
            State {
                step: 0,
                rng: Rng::new(a.seed),
                args: a.clone(),
                elapsed_secs: 0.0,
            },
        )
    };
    println!(
        "{} variant {:?}: {} parameters ({} encoder), {} steps of {}, lr {} (head x{}), {}",
        cfg.name,
        variant,
        params::count(&raw),
        params::count_prefix(&raw, "encoder."),
        a.steps,
        a.batch,
        a.lr,
        a.head_lr_mult,
        if a.bf16 { "bf16" } else { "f32" }
    );
    let aug = Augment {
        permute: true,
        p_names: a.p_names,
        p_polarity: a.p_polarity,
    };
    let w = LossWeights {
        brier: a.brier,
        rps: a.rps,
        kl: a.kl,
    };
    let val_items: Vec<&S5Item> = {
        let mut idx: Vec<usize> = (0..val.len()).collect();
        if idx.len() > a.val_max {
            Rng::new(a.seed ^ 0xA11).shuffle(&mut idx);
            idx.truncate(a.val_max);
            idx.sort_unstable();
        }
        idx.into_iter().map(|i| &val[i]).collect()
    };
    std::fs::create_dir_all(&a.out)?;
    let mut log = JsonLog::open(&a.out.join("log.jsonl"))?;
    let mut f = compiled(&cfg, sorted_names(&raw), w);
    let validate = |ema: &ParamMap, step: u64, log: &mut JsonLog| -> Result<()> {
        if val_items.is_empty() {
            return Ok(());
        }
        let preds = predict(ema, &cfg, &val_items, 16_384)?;
        let m = report(&preds);
        let all = &m["all"];
        println!(
            "  val step {step}: n {} acc {:.4} nll {:.4} brier {:.4} ece {:.4}",
            all.n, all.accuracy, all.nll, all.brier, all.ece
        );
        log.write(&json!({"step": step, "val": m}))?;
        Ok(())
    };

    let t0 = Instant::now();
    let base = st.elapsed_secs;
    let (mut wl, mut wt, mut wn, mut win) = (0.0f64, 0usize, 0u64, Instant::now());
    while st.step < a.steps {
        let lr_scale = warmup_linear(st.step as f64 / a.steps as f64, a.warmup) as f32;
        let b = sample(&train, a, variant, &aug, &mut st.rng, &tok, &cfg, &w)?;
        let (loss_arr, grads) = f.call(&raw, batch_arrays(&b))?;
        let (l, gn) = apply_update(&mut raw, &mut opt, loss_arr, &grads, lr_scale)?;
        if !l.is_finite() {
            bail!("non-finite loss at step {}", st.step);
        }
        if a.ema > 0.0 {
            ema_update(&mut ema, &raw, a.ema)?;
        } else {
            ema = raw.clone();
        }
        st.step += 1;
        st.elapsed_secs = base + t0.elapsed().as_secs_f64();
        wl += l as f64;
        wt += b.n_tokens;
        wn += 1;
        if st.step % a.log_every == 0 {
            let dt = win.elapsed().as_secs_f64();
            println!(
                "step {:>6} loss {:.4} gnorm {gn:.3} lr x{lr_scale:.3} {:.0} tok/s",
                st.step,
                wl / wn as f64,
                wt as f64 / dt
            );
            log.write(
                &json!({"step": st.step, "loss": wl / wn as f64, "grad_norm": gn,
                "lr_scale": lr_scale, "tok_s": wt as f64 / dt, "elapsed": st.elapsed_secs}),
            )?;
            (wl, wt, wn, win) = (0.0, 0, 0, Instant::now());
        }
        if st.step % a.eval_every == 0 {
            validate(&ema, st.step, &mut log)?;
        }
        if st.step % a.checkpoint_every == 0 && st.step < a.steps {
            save(&last, &raw, &ema, &opt, &cfg, &st, &tok_json)?;
        }
    }
    if st.step % a.eval_every != 0 {
        validate(&ema, st.step, &mut log)?;
    }
    save(&last, &raw, &ema, &opt, &cfg, &st, &tok_json)?;
    println!(
        "done: {} steps in {:.1} min, checkpoint {}",
        st.step,
        st.elapsed_secs / 60.0,
        last.display()
    );
    Ok(())
}
