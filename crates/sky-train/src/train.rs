//! Training drivers: throughput benchmark, time-budgeted MLM pretraining and
//! decision training.

use std::{
    collections::BTreeMap,
    io::Write,
    path::{Path, PathBuf},
    time::Instant,
};

use anyhow::{bail, Context, Result};
use serde_json::json;

use crate::{
    config::{ModelConfig, CLS_ID, N_SPECIAL, SEP_ID},
    decision::{self, DecisionData, Item, Row},
    metrics::{self, Metrics, Pred},
    mlm::{self, Corpus, Rows},
    model::{self, DecisionBatch, MlmBatch},
    optim::{wsd, OptConfig},
    params::{self, Parts},
    rng::Rng,
    sequence::Tok,
    session::Session,
};

/// Appends one JSON line to a log file.
pub struct JsonLog {
    file: std::fs::File,
}

impl JsonLog {
    pub fn open(path: &Path) -> Result<Self> {
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir)?;
        }
        let file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)?;
        Ok(Self { file })
    }

    pub fn write(&mut self, v: &serde_json::Value) -> Result<()> {
        writeln!(self.file, "{v}")?;
        self.file.flush()?;
        Ok(())
    }
}

fn peak_memory_mb() -> f64 {
    mlx_rs::memory::peak_memory()
        .map(|b| b as f64 / (1024.0 * 1024.0))
        .unwrap_or(f64::NAN)
}

// ----------------------------------------------------------------------------- bench

/// Settings of `sky-train bench`.
#[derive(Debug, Clone)]
pub struct BenchArgs {
    pub config: PathBuf,
    pub mlm_batch: usize,
    pub mlm_seq: usize,
    pub dec_batch: usize,
    pub dec_seq: usize,
    pub dec_options: usize,
    pub warmup: usize,
    pub steps: usize,
    pub bench_md: PathBuf,
    pub bf16: bool,
}

/// One benchmark measurement.
#[derive(Debug, Clone)]
pub struct BenchRow {
    pub model: String,
    pub params: usize,
    pub phase: &'static str,
    pub batch: usize,
    pub seq: usize,
    pub step_ms: f64,
    pub tok_s: f64,
    pub peak_mb: f64,
    pub precision: &'static str,
}

fn random_mlm_batches(
    cfg: &ModelConfig,
    b: usize,
    t: usize,
    rng: &mut Rng,
    n: usize,
) -> Vec<MlmBatch> {
    (0..n)
        .map(|_| {
            let rows: Vec<Vec<u32>> = (0..b)
                .map(|_| {
                    let mut r = vec![CLS_ID];
                    r.extend((0..t - 2).map(|_| {
                        N_SPECIAL + rng.below(cfg.vocab_size - N_SPECIAL as usize) as u32
                    }));
                    r.push(SEP_ID);
                    r
                })
                .collect();
            mlm::mask_batch(&rows, cfg.vocab_size, 0.3, rng)
        })
        .collect()
}

fn random_decision_batches(
    cfg: &ModelConfig,
    b: usize,
    t: usize,
    k: usize,
    rng: &mut Rng,
    n: usize,
) -> Vec<DecisionBatch> {
    (0..n)
        .map(|_| {
            let rows: Vec<Row> = (0..b)
                .map(|_| {
                    let mut ids: Vec<u32> = (0..t)
                        .map(|_| N_SPECIAL + rng.below(cfg.vocab_size - N_SPECIAL as usize) as u32)
                        .collect();
                    ids[0] = CLS_ID;
                    let markers: Vec<usize> = (0..k).map(|i| 12 + 6 * i).collect();
                    for &m in &markers {
                        ids[m] = crate::config::MASK_ID;
                    }
                    let mut target = vec![0.0; k];
                    target[rng.below(k)] = 1.0;
                    Row {
                        ids,
                        markers,
                        qtype: rng.below(3),
                        target,
                    }
                })
                .collect();
            decision::make_batch(&rows, 16)
        })
        .collect()
}

fn time_steps<B>(
    session: &mut Session,
    batches: &[B],
    warmup: usize,
    steps: usize,
    step: impl Fn(&mut Session, &B) -> Result<crate::session::StepStats>,
) -> Result<(f64, f64, f64)> {
    for i in 0..warmup {
        step(session, &batches[i % batches.len()])?;
    }
    mlx_rs::memory::reset_peak_memory()?;
    let t0 = Instant::now();
    let mut tokens = 0usize;
    for i in 0..steps {
        let s = step(session, &batches[i % batches.len()])?;
        if !s.loss.is_finite() {
            bail!("non-finite loss during benchmark");
        }
        tokens += s.tokens;
    }
    let secs = t0.elapsed().as_secs_f64();
    Ok((
        secs * 1000.0 / steps as f64,
        tokens as f64 / secs,
        peak_memory_mb(),
    ))
}

/// Measures MLM and decision training throughput on random tokens and appends
/// the rows to `BENCH.md`.
pub fn bench(a: &BenchArgs) -> Result<Vec<BenchRow>> {
    let mut cfg = ModelConfig::load(&a.config)?;
    cfg.bf16 = a.bf16;
    let precision = if a.bf16 { "bf16" } else { "f32" };
    let mut rng = Rng::new(1234);
    let mut out = Vec::new();

    let p = params::init(&cfg, Parts::PRETRAIN, 1)?;
    let n_params = params::count(&p);
    let mut s = Session::new(cfg.clone(), p, OptConfig::default(), "bench", 1, 0.0);
    let batches = random_mlm_batches(&cfg, a.mlm_batch, a.mlm_seq, &mut rng, 4);
    let (ms, tok_s, mb) = time_steps(&mut s, &batches, a.warmup, a.steps, |s, b| {
        s.mlm_step(b, 1.0)
    })?;
    out.push(BenchRow {
        model: cfg.name.clone(),
        params: n_params,
        phase: "mlm",
        batch: a.mlm_batch,
        seq: a.mlm_seq,
        step_ms: ms,
        tok_s,
        peak_mb: mb,
        precision,
    });
    drop(s);

    let p = params::init(&cfg, Parts::DECISION, 2)?;
    let n_params = params::count(&p);
    let mut s = Session::new(cfg.clone(), p, OptConfig::default(), "bench", 2, 0.0);
    let batches = random_decision_batches(&cfg, a.dec_batch, a.dec_seq, a.dec_options, &mut rng, 4);
    let (ms, tok_s, mb) = time_steps(&mut s, &batches, a.warmup, a.steps, |s, b| {
        s.decision_step(b, 1.0)
    })?;
    out.push(BenchRow {
        model: cfg.name.clone(),
        params: n_params,
        phase: "decision",
        batch: a.dec_batch,
        seq: a.dec_seq,
        step_ms: ms,
        tok_s,
        peak_mb: mb,
        precision,
    });

    append_bench_md(&a.bench_md, &out, a.warmup, a.steps)?;
    Ok(out)
}

fn append_bench_md(path: &Path, rows: &[BenchRow], warmup: usize, steps: usize) -> Result<()> {
    let header = "# skycmd-s1 training throughput\n\n\
        Measured by `sky-train bench` (mlx-rs 0.32 / MLX 0.32.2, Metal, compiled loss + gradient) \
        on random tokens. Warmup steps are excluded; timing covers forward + backward + optimizer \
        update (Muon on the encoder hidden matrices, AdamW elsewhere). `bf16` runs matmuls and \
        activations in bfloat16 with f32 master weights and f32 losses. `tok/s` counts input tokens \
        (batch x seq); `params` counts the trained model (encoder + MLM head for `mlm`, encoder + \
        decision head for `decision`).\n\n\
        | model | params | phase | precision | batch | seq | step (ms) | tok/s | peak mem (MB) | steps |\n\
        |---|---:|---|---|---:|---:|---:|---:|---:|---|\n";
    let mut text = std::fs::read_to_string(path).unwrap_or_default();
    if !text.contains("| model | params | phase |") {
        text = header.to_string();
    }
    for r in rows {
        text.push_str(&format!(
            "| {} | {} | {} | {} | {} | {} | {:.1} | {:.0} | {:.0} | {}+{} |\n",
            r.model,
            r.params,
            r.phase,
            r.precision,
            r.batch,
            r.seq,
            r.step_ms,
            r.tok_s,
            r.peak_mb,
            warmup,
            steps
        ));
    }
    std::fs::write(path, text)?;
    Ok(())
}

// ----------------------------------------------------------------------------- pretrain

/// Settings of `sky-train pretrain`.
#[derive(Debug, Clone)]
pub struct PretrainArgs {
    pub config: PathBuf,
    pub tokenizer: PathBuf,
    pub text: PathBuf,
    pub minutes: f64,
    pub out: PathBuf,
    pub resume: bool,
    pub batch: usize,
    pub seq: usize,
    pub opt: OptConfig,
    pub seed: u64,
    pub max_tokens: usize,
    pub log_every: u64,
    pub eval_every: u64,
    pub checkpoint_minutes: f64,
    pub val_batches: usize,
    pub bf16: bool,
}

fn mlm_val_loss(s: &Session, val: &[MlmBatch]) -> Result<f64> {
    let mut total = 0.0;
    for b in val {
        total += model::mlm_loss(&s.params, &s.cfg, b)?.item_exact::<f32>() as f64;
    }
    Ok(total / val.len().max(1) as f64)
}

/// Time-budgeted MLM pretraining with periodic validation and checkpoints.
pub fn pretrain(a: &PretrainArgs) -> Result<()> {
    let mut cfg = ModelConfig::load(&a.config)?;
    cfg.bf16 = a.bf16;
    let tok = Tok::load(&a.tokenizer)?;
    if tok.vocab_size() > cfg.vocab_size {
        bail!(
            "tokenizer has {} tokens but {} has vocab_size {}",
            tok.vocab_size(),
            cfg.name,
            cfg.vocab_size
        );
    }
    std::fs::create_dir_all(&a.out)?;
    let t_load = Instant::now();
    let corpus = Corpus::load(
        &a.text,
        &tok,
        &a.tokenizer,
        &a.out.join("cache"),
        a.max_tokens,
    )?;
    let rows = Rows::new(&corpus, a.seq, 0.01, a.val_batches * a.batch);
    println!(
        "corpus: {} records, {} tokens -> {} train / {} val rows of {} ({:.1}s)",
        corpus.n_records,
        corpus.tokens.len(),
        rows.train.len(),
        rows.val.len(),
        a.seq,
        t_load.elapsed().as_secs_f64()
    );
    if rows.train.is_empty() {
        bail!("corpus too small");
    }
    let last = a.out.join("last");
    let mut s = if a.resume && last.join("state.json").exists() {
        let s = Session::load(&last)?;
        println!(
            "resumed from {} at step {} ({:.1} min used)",
            last.display(),
            s.state.step,
            s.state.elapsed_secs / 60.0
        );
        s
    } else {
        let p = params::init(&cfg, Parts::PRETRAIN, a.seed)?;
        let mut s = Session::new(
            cfg.clone(),
            p,
            a.opt.clone(),
            "pretrain",
            a.seed,
            a.minutes * 60.0,
        );
        s.state.extra =
            json!({"batch": a.batch, "seq": a.seq, "text": a.text, "tokenizer": a.tokenizer});
        s
    };
    s.state.budget_secs = a.minutes * 60.0;
    s.cfg.bf16 = a.bf16;
    println!(
        "{}: {} parameters (encoder {}), batch {}x{}, budget {:.1} min",
        s.cfg.name,
        params::count(&s.params),
        params::count_prefix(&s.params, "encoder."),
        a.batch,
        a.seq,
        a.minutes
    );
    let mut vrng = Rng::new(a.seed ^ 0x0005_EED0_F7A1);
    let val: Vec<MlmBatch> = rows
        .val
        .chunks(a.batch)
        .take(a.val_batches)
        .map(|c| {
            let rs: Vec<Vec<u32>> = c.iter().map(|&st| rows.row(&corpus, st)).collect();
            mlm::mask_batch(&rs, s.cfg.vocab_size, 0.3, &mut vrng)
        })
        .collect();
    let tok_json = std::fs::read(&a.tokenizer)?;
    let mut log = JsonLog::open(&a.out.join("log.jsonl"))?;
    let save =
        |s: &Session| -> Result<()> { s.save(&last, &[("tokenizer.json", tok_json.clone())]) };

    let base_elapsed = s.state.elapsed_secs;
    let t0 = Instant::now();
    let mut last_ckpt = Instant::now();
    let (mut win_loss, mut win_tokens, mut win_steps, mut win_t) =
        (0.0f64, 0usize, 0u64, Instant::now());
    while s.state.elapsed_secs < s.state.budget_secs {
        let frac = s.state.elapsed_secs / s.state.budget_secs;
        let lr_scale = wsd(frac) as f32;
        let seed = s.state.seed;
        let starts = s.state.cursor.next(&rows, seed, a.batch);
        let rs: Vec<Vec<u32>> = starts.iter().map(|&st| rows.row(&corpus, st)).collect();
        let batch = mlm::mask_batch(&rs, s.cfg.vocab_size, 0.3, &mut s.state.rng);
        let st = s.mlm_step(&batch, lr_scale)?;
        if !st.loss.is_finite() {
            bail!("non-finite loss at step {}", s.state.step);
        }
        s.state.elapsed_secs = base_elapsed + t0.elapsed().as_secs_f64();
        win_loss += st.loss as f64;
        win_tokens += st.tokens;
        win_steps += 1;
        let step = s.state.step;
        if step % a.log_every == 0 {
            let dt = win_t.elapsed().as_secs_f64();
            let rec = json!({
                "step": step, "loss": win_loss / win_steps as f64, "grad_norm": st.grad_norm,
                "lr": s.opt.cfg.adam_lr * lr_scale, "muon_lr": s.opt.cfg.muon_lr * lr_scale,
                "tok_s": win_tokens as f64 / dt, "elapsed_min": s.state.elapsed_secs / 60.0,
                "epoch": s.state.cursor.epoch,
            });
            log.write(&rec)?;
            println!(
                "step {step:>6} loss {:.4} lr x{lr_scale:.3} {:.0} tok/s ({:.1}/{:.1} min)",
                win_loss / win_steps as f64,
                win_tokens as f64 / dt,
                s.state.elapsed_secs / 60.0,
                a.minutes
            );
            (win_loss, win_tokens, win_steps, win_t) = (0.0, 0, 0, Instant::now());
        }
        if step % a.eval_every == 0 && !val.is_empty() {
            let vl = mlm_val_loss(&s, &val)?;
            log.write(&json!({"step": step, "val_loss": vl}))?;
            println!("step {step:>6} val_loss {vl:.4}");
        }
        if last_ckpt.elapsed().as_secs_f64() >= a.checkpoint_minutes * 60.0 {
            save(&s)?;
            last_ckpt = Instant::now();
        }
    }
    let vl = if val.is_empty() {
        f64::NAN
    } else {
        mlm_val_loss(&s, &val)?
    };
    log.write(&json!({"step": s.state.step, "val_loss": vl, "final": true}))?;
    save(&s)?;
    println!(
        "done: {} steps, final val_loss {vl:.4}, checkpoint {}",
        s.state.step,
        last.display()
    );
    Ok(())
}

// ----------------------------------------------------------------------------- decide

/// Settings of `sky-train decide`.
#[derive(Debug, Clone)]
pub struct DecideArgs {
    pub config: PathBuf,
    pub init: Option<PathBuf>,
    pub tokenizer: Option<PathBuf>,
    pub records: Vec<PathBuf>,
    pub weights: Option<String>,
    pub minutes: f64,
    pub out: PathBuf,
    pub resume: bool,
    pub batch: usize,
    pub opt: OptConfig,
    pub seed: u64,
    pub log_every: u64,
    pub eval_every: u64,
    pub checkpoint_minutes: f64,
    pub val_max: usize,
    pub bf16: bool,
}

/// Source sampling weights: explicit `name=w,...`, else `games` gets 0.5 and
/// the other groups share 0.5 equally.
pub fn source_weights(groups: &[String], spec: Option<&str>) -> Result<Vec<f64>> {
    if let Some(spec) = spec {
        let mut m = BTreeMap::new();
        for part in spec.split(',').filter(|p| !p.trim().is_empty()) {
            let (k, v) = part
                .split_once('=')
                .with_context(|| format!("bad weight {part:?}, want name=w"))?;
            m.insert(k.trim().to_string(), v.trim().parse::<f64>()?);
        }
        return groups
            .iter()
            .map(|g| {
                m.get(g)
                    .copied()
                    .with_context(|| format!("no weight for source {g}"))
            })
            .collect();
    }
    let n_games = groups.iter().filter(|g| *g == "games").count();
    let n_other = groups.len() - n_games;
    Ok(groups
        .iter()
        .map(|g| match (g == "games", n_games > 0 && n_other > 0) {
            (true, true) => 0.5 / n_games as f64,
            (false, true) => 0.5 / n_other as f64,
            _ => 1.0 / groups.len() as f64,
        })
        .collect())
}

/// The `rl_agent_config.json` of a decision checkpoint.
pub fn rl_agent_json(cfg: &ModelConfig, extra: serde_json::Value) -> serde_json::Value {
    let a = &cfg.rl_agent;
    let mut v = json!({
        "encoder": format!("skycmd-{}", cfg.name),
        "head_layers": a.head_layers,
        "max_len": a.max_len,
        "head_max_len": a.head_max_len,
        "head_max_len_train": a.head_max_len,
        "laya_fmt": 1,
        "model_name": format!("skycmd-{}", cfg.name),
        "temperature": a.temperature,
        "temperature_by_options": a.temperature_by_options,
        "fine_tuned": true,
    });
    if let (Some(o), serde_json::Value::Object(e)) = (v.as_object_mut(), extra) {
        o.extend(e);
    }
    v
}

/// Deterministic subset of at most `max` items.
pub fn subset(items: &[Item], max: usize, seed: u64) -> Vec<&Item> {
    let mut idx: Vec<usize> = (0..items.len()).collect();
    if idx.len() > max {
        Rng::new(seed).shuffle(&mut idx);
        idx.truncate(max);
        idx.sort_unstable();
    }
    idx.into_iter().map(|i| &items[i]).collect()
}

/// Metrics per group plus `all`.
pub fn grouped_metrics(
    preds: &[Pred],
    key: impl Fn(&Pred) -> String,
    temp: impl Fn(&Pred) -> f32,
) -> BTreeMap<String, Metrics> {
    let mut groups: BTreeMap<String, Vec<&Pred>> = BTreeMap::new();
    for p in preds {
        groups.entry(key(p)).or_default().push(p);
    }
    let mut out: BTreeMap<String, Metrics> = groups
        .iter()
        .map(|(k, v)| (k.clone(), metrics::compute(v.iter().copied(), &temp)))
        .collect();
    out.insert("all".into(), metrics::compute(preds.iter(), &temp));
    out
}

/// Time-budgeted decision training (AdamW + Muon, WSD) with per-source validation.
pub fn decide(a: &DecideArgs) -> Result<()> {
    let mut cfg = ModelConfig::load(&a.config)?;
    cfg.bf16 = a.bf16;
    let tok_path = match (&a.tokenizer, &a.init) {
        (Some(t), _) => t.clone(),
        (None, Some(init)) => init.join("tokenizer.json"),
        (None, None) => bail!("--tokenizer is required without --init"),
    };
    let tok = Tok::load(&tok_path)?;
    if tok.vocab_size() > cfg.vocab_size {
        bail!(
            "tokenizer has {} tokens but {} has vocab_size {}",
            tok.vocab_size(),
            cfg.name,
            cfg.vocab_size
        );
    }
    let t_load = Instant::now();
    let data: DecisionData = decision::load(&a.records, &tok, &cfg.rl_agent)?;
    let weights = source_weights(&data.groups, a.weights.as_deref())?;
    for (g, (name, w)) in data.groups.iter().zip(&weights).enumerate() {
        println!(
            "source {name}: {} train items, weight {w:.3}",
            data.train[g].len()
        );
    }
    println!(
        "val {} / test {} items, {} records skipped ({:.1}s)",
        data.val.len(),
        data.test.len(),
        data.skipped,
        t_load.elapsed().as_secs_f64()
    );
    if data.train.iter().all(Vec::is_empty) {
        bail!("no training items");
    }
    std::fs::create_dir_all(&a.out)?;
    let last = a.out.join("last");
    let mut s = if a.resume && last.join("state.json").exists() {
        let s = Session::load(&last)?;
        println!("resumed from {} at step {}", last.display(), s.state.step);
        s
    } else {
        let mut p = params::init(&cfg, Parts::DECISION, a.seed)?;
        if let Some(init) = &a.init {
            let src = params::load(&init.join("model.safetensors"))?;
            let n = params::load_matching(&mut p, &src)?;
            let n_enc = p.keys().filter(|k| k.starts_with("encoder.")).count();
            if n < n_enc {
                bail!(
                    "{} provides only {n} of the {n_enc} encoder tensors",
                    init.display()
                );
            }
            println!("initialized {n} tensors from {}", init.display());
        }
        let mut s = Session::new(
            cfg.clone(),
            p,
            a.opt.clone(),
            "decide",
            a.seed,
            a.minutes * 60.0,
        );
        s.state.extra = json!({
            "batch": a.batch, "records": a.records, "weights": weights, "groups": data.groups,
            "init": a.init, "tokenizer": tok_path,
        });
        s
    };
    s.state.budget_secs = a.minutes * 60.0;
    s.cfg.bf16 = a.bf16;
    println!(
        "{}: {} parameters, batch {}, budget {:.1} min",
        s.cfg.name,
        params::count(&s.params),
        a.batch,
        a.minutes
    );
    let val_items = subset(&data.val, a.val_max, a.seed);
    let tok_json = std::fs::read(&tok_path)?;
    let mut log = JsonLog::open(&a.out.join("log.jsonl"))?;
    let save = |s: &Session| -> Result<()> {
        let rl = rl_agent_json(
            &s.cfg,
            json!({"training": {"steps": s.state.step, "minutes": s.state.elapsed_secs / 60.0}}),
        );
        s.save(
            &last,
            &[
                ("tokenizer.json", tok_json.clone()),
                ("rl_agent_config.json", serde_json::to_vec_pretty(&rl)?),
            ],
        )
    };
    let validate = |s: &Session, log: &mut JsonLog| -> Result<()> {
        if val_items.is_empty() {
            return Ok(());
        }
        let preds = decision::predict(&s.params, &s.cfg, &data.groups, &val_items, 16_384)?;
        let m = grouped_metrics(&preds, |p| p.group.clone(), |_| 1.0);
        log.write(&json!({"step": s.state.step, "val": m}))?;
        for (g, x) in &m {
            println!(
                "  val[{g}] n={} acc {:.3} nll {:.4} brier {:.4} ece {:.4}",
                x.n, x.accuracy, x.nll, x.brier, x.ece
            );
        }
        Ok(())
    };

    let base_elapsed = s.state.elapsed_secs;
    let t0 = Instant::now();
    let mut last_ckpt = Instant::now();
    let (mut win_loss, mut win_tokens, mut win_steps, mut win_t) =
        (0.0f64, 0usize, 0u64, Instant::now());
    while s.state.elapsed_secs < s.state.budget_secs {
        let lr_scale = wsd(s.state.elapsed_secs / s.state.budget_secs) as f32;
        let agent = s.cfg.rl_agent.clone();
        let batch = decision::sample_batch(&data, &weights, a.batch, &mut s.state.rng, &agent);
        let st = s.decision_step(&batch, lr_scale)?;
        if !st.loss.is_finite() {
            bail!("non-finite loss at step {}", s.state.step);
        }
        s.state.elapsed_secs = base_elapsed + t0.elapsed().as_secs_f64();
        win_loss += st.loss as f64;
        win_tokens += st.tokens;
        win_steps += 1;
        let step = s.state.step;
        if step % a.log_every == 0 {
            let dt = win_t.elapsed().as_secs_f64();
            log.write(&json!({
                "step": step, "loss": win_loss / win_steps as f64, "grad_norm": st.grad_norm,
                "lr": s.opt.cfg.adam_lr * lr_scale, "muon_lr": s.opt.cfg.muon_lr * lr_scale,
                "tok_s": win_tokens as f64 / dt, "elapsed_min": s.state.elapsed_secs / 60.0,
            }))?;
            println!(
                "step {step:>6} loss {:.4} lr x{lr_scale:.3} {:.0} tok/s ({:.1}/{:.1} min)",
                win_loss / win_steps as f64,
                win_tokens as f64 / dt,
                s.state.elapsed_secs / 60.0,
                a.minutes
            );
            (win_loss, win_tokens, win_steps, win_t) = (0.0, 0, 0, Instant::now());
        }
        if step % a.eval_every == 0 {
            validate(&s, &mut log)?;
        }
        if last_ckpt.elapsed().as_secs_f64() >= a.checkpoint_minutes * 60.0 {
            save(&s)?;
            last_ckpt = Instant::now();
        }
    }
    println!("done: {} steps", s.state.step);
    validate(&s, &mut log)?;
    save(&s)?;
    println!("checkpoint {}", last.display());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_source_weights() {
        let g = |v: &[&str]| v.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        assert_eq!(
            source_weights(&g(&["decisions", "games"]), None).unwrap(),
            vec![0.5, 0.5]
        );
        assert_eq!(
            source_weights(&g(&["a", "b", "games"]), None).unwrap(),
            vec![0.25, 0.25, 0.5]
        );
        assert_eq!(source_weights(&g(&["decisions"]), None).unwrap(), vec![1.0]);
        assert_eq!(
            source_weights(&g(&["decisions", "games"]), Some("games=0.8,decisions=0.2")).unwrap(),
            vec![0.2, 0.8]
        );
        assert!(source_weights(&g(&["x"]), Some("games=1")).is_err());
    }
}
