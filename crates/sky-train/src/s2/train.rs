//! s2 training: session and checkpoints, numeric MLM pretraining, decision
//! training with deep supervision and ternary QAT, batched prediction and the
//! throughput benchmark.

use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
    time::Instant,
};

use anyhow::{bail, Context, Result};
use mlx_rs::{Array, Dtype};
use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::{
    metrics::{self, Metrics, Pred},
    mlm::{Cursor, Rows},
    model::host_f32,
    optim::{wsd, OptConfig, Optimizer},
    params::{self, ParamMap},
    rng::Rng,
    s2::{
        config::S2Config,
        data::{
            self, make_batch, mask_batch, NumCorpus, NumMlmBatch, S2Batch, S2Data, S2Item,
            StaticSeq, Stream,
        },
        model::{self, expand_pretrained, S2Parts},
    },
    sequence::{Tok, QTYPES},
    session::{
        apply_update, keep_going, qat_due, replace_params, sorted_names, stop_step, train_step,
        CompiledGrad, Progress,
    },
    train::{source_weights, JsonLog},
};

/// Serializable s2 training progress.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct S2State {
    /// `s2-pretrain` or `s2-decide`.
    pub phase: String,
    pub step: u64,
    pub elapsed_secs: f64,
    pub budget_secs: f64,
    /// Step budget of a `--steps` run (absent for a time budget).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub steps: Option<u64>,
    pub seed: u64,
    pub rng: Rng,
    #[serde(default)]
    pub cursor: Cursor,
    pub config: S2Config,
    pub opt: OptConfig,
    /// True once ternary QAT has started.
    #[serde(default)]
    pub qat: bool,
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

/// An s2 training session.
pub struct S2Session {
    pub cfg: S2Config,
    pub params: ParamMap,
    pub opt: Optimizer,
    pub state: S2State,
    /// Compile the loss + gradient graphs (one compiled graph per input shape).
    pub compile: bool,
    /// Weight of the number MSE in the numeric MLM loss.
    pub num_weight: f32,
    mlm_fn: Option<CompiledGrad>,
    /// Compiled decision graphs keyed by QAT (the exit weights are an input).
    dec_fn: BTreeMap<bool, CompiledGrad>,
    /// Compiled temporal sequence graphs keyed by QAT.
    seq_fn: BTreeMap<bool, CompiledGrad>,
    /// s4 deep improvement supervision: per-exit sharpening exponents of the targets
    /// (`model::exit_targets`); `None` trains every exit on the target itself.
    pub dis_beta: Option<Vec<f32>>,
    /// s6 self-distillation of the early exits toward the deepest (`model::ExitKd`).
    pub exit_kd: Option<model::ExitKd>,
}

/// Number of arrays [`dec_arrays`] produces per batch.
const DEC_ARRAYS: usize = 15;

fn mlm_arrays(b: &NumMlmBatch) -> Vec<Array> {
    vec![
        b.ids.clone(),
        b.values.clone(),
        b.has.clone(),
        b.num_flag.clone(),
        b.valid.clone(),
        b.positions.clone(),
        b.labels.clone(),
        b.weights.clone(),
        b.num_positions.clone(),
        b.num_targets.clone(),
        b.num_weights.clone(),
    ]
}

fn mlm_from(a: &[Array]) -> NumMlmBatch {
    NumMlmBatch {
        ids: a[0].clone(),
        values: a[1].clone(),
        has: a[2].clone(),
        num_flag: a[3].clone(),
        valid: a[4].clone(),
        positions: a[5].clone(),
        labels: a[6].clone(),
        weights: a[7].clone(),
        num_positions: a[8].clone(),
        num_targets: a[9].clone(),
        num_weights: a[10].clone(),
        n_tokens: 0,
    }
}

fn dec_arrays(b: &S2Batch) -> Vec<Array> {
    vec![
        b.st_ids.clone(),
        b.st_values.clone(),
        b.st_has.clone(),
        b.st_seg.clone(),
        b.st_pos.clone(),
        b.marker_idx.clone(),
        b.q_idx.clone(),
        b.q_valid.clone(),
        b.ids.clone(),
        b.values.clone(),
        b.has.clone(),
        b.valid.clone(),
        b.qtype.clone(),
        b.marker_mask.clone(),
        b.target.clone(),
    ]
}

fn dec_from(a: &[Array]) -> S2Batch {
    S2Batch {
        st_ids: a[0].clone(),
        st_values: a[1].clone(),
        st_has: a[2].clone(),
        st_seg: a[3].clone(),
        st_pos: a[4].clone(),
        marker_idx: a[5].clone(),
        q_idx: a[6].clone(),
        q_valid: a[7].clone(),
        ids: a[8].clone(),
        values: a[9].clone(),
        has: a[10].clone(),
        valid: a[11].clone(),
        qtype: a[12].clone(),
        marker_mask: a[13].clone(),
        target: a[14].clone(),
        n_tokens: 0,
        n_rows: 0,
    }
}

impl S2Session {
    pub fn new(
        cfg: S2Config,
        params: ParamMap,
        opt: OptConfig,
        phase: &str,
        seed: u64,
        budget_secs: f64,
    ) -> Self {
        let state = S2State {
            phase: phase.into(),
            step: 0,
            elapsed_secs: 0.0,
            budget_secs,
            steps: None,
            seed,
            rng: Rng::new(seed),
            cursor: Cursor::default(),
            config: cfg.clone(),
            opt: opt.clone(),
            qat: false,
            extra: serde_json::Value::Null,
        };
        Self {
            cfg,
            params,
            opt: Optimizer::new(opt),
            state,
            compile: true,
            num_weight: model::NUM_LOSS_WEIGHT,
            mlm_fn: None,
            dec_fn: BTreeMap::new(),
            seq_fn: BTreeMap::new(),
            dis_beta: None,
            exit_kd: None,
        }
    }

    /// One numeric MLM step.
    pub fn mlm_step(&mut self, batch: &NumMlmBatch, lr_scale: f32) -> Result<StepStats> {
        let (loss, grad_norm) = if self.compile {
            if self.mlm_fn.is_none() {
                let cfg = self.cfg.clone();
                let nw = self.num_weight;
                self.mlm_fn = Some(CompiledGrad::new(
                    sorted_names(&self.params),
                    move |p, b| model::mlm_loss(p, &cfg, &mlm_from(b), nw),
                ));
            }
            let (loss, grads) = self
                .mlm_fn
                .as_mut()
                .expect("compiled")
                .call(&self.params, mlm_arrays(batch))?;
            apply_update(&mut self.params, &mut self.opt, loss, &grads, lr_scale)?
        } else {
            let (cfg, nw) = (&self.cfg, self.num_weight);
            train_step(&mut self.params, &mut self.opt, batch, lr_scale, |p, b| {
                model::mlm_loss(p, cfg, b, nw)
            })?
        };
        self.state.step += 1;
        Ok(StepStats {
            loss,
            grad_norm,
            tokens: batch.n_tokens,
        })
    }

    /// One decision step (deep supervision; ternary fake-quant when `qat`).
    pub fn decision_step(
        &mut self,
        batch: &S2Batch,
        lr_scale: f32,
        qat: bool,
    ) -> Result<StepStats> {
        self.decision_step_exits(batch, lr_scale, qat, None)
    }

    /// [`Self::decision_step`] supervising only the first `n_exits` exits
    /// (stochastic depth: the state encoder stops at the last of them).
    pub fn decision_step_exits(
        &mut self,
        batch: &S2Batch,
        lr_scale: f32,
        qat: bool,
        n_exits: Option<usize>,
    ) -> Result<StepStats> {
        let n = n_exits
            .unwrap_or(self.cfg.n_exits())
            .clamp(1, self.cfg.n_exits());
        let wv = model::exit_weight_vec(&self.cfg, Some(n));
        let (loss, grad_norm) = if self.compile {
            let names = sorted_names(&self.params);
            let cfg = self.cfg.clone();
            let dis = self.dis_beta.is_some();
            let kd = self.exit_kd;
            let f = self.dec_fn.entry(qat).or_insert_with(|| {
                CompiledGrad::new(names, move |p, b| {
                    if let Some(kd) = kd {
                        let beta = dis.then(|| &b[16]);
                        model::decision_loss_kd(p, &cfg, &dec_from(b), qat, &b[15], beta, kd)
                    } else if dis {
                        model::decision_loss_dis(p, &cfg, &dec_from(b), qat, &b[15], &b[16])
                    } else {
                        model::decision_loss_weighted(p, &cfg, &dec_from(b), qat, &b[15])
                    }
                })
            });
            let mut inputs = dec_arrays(batch);
            inputs.push(Array::from_slice(&wv, &[wv.len() as i32]));
            if let Some(beta) = &self.dis_beta {
                inputs.push(Array::from_slice(beta, &[beta.len() as i32]));
            }
            let (loss, grads) = f.call(&self.params, inputs)?;
            apply_update(&mut self.params, &mut self.opt, loss, &grads, lr_scale)?
        } else if let Some(kd) = self.exit_kd {
            let cfg = &self.cfg;
            let wts = Array::from_slice(&wv, &[wv.len() as i32]);
            let beta = self
                .dis_beta
                .as_ref()
                .map(|b| Array::from_slice(b, &[b.len() as i32]));
            train_step(&mut self.params, &mut self.opt, batch, lr_scale, |p, b| {
                model::decision_loss_kd(p, cfg, b, qat, &wts, beta.as_ref(), kd)
            })?
        } else if let Some(beta) = &self.dis_beta {
            let cfg = &self.cfg;
            let wts = Array::from_slice(&wv, &[wv.len() as i32]);
            let beta = Array::from_slice(beta, &[beta.len() as i32]);
            train_step(&mut self.params, &mut self.opt, batch, lr_scale, |p, b| {
                model::decision_loss_dis(p, cfg, b, qat, &wts, &beta)
            })?
        } else {
            let cfg = &self.cfg;
            train_step(&mut self.params, &mut self.opt, batch, lr_scale, |p, b| {
                model::decision_loss_exits(p, cfg, b, qat, Some(n))
            })?
        };
        self.state.step += 1;
        Ok(StepStats {
            loss,
            grad_norm,
            tokens: batch.n_tokens,
        })
    }

    /// One temporal step (SPEC-S3 §7): `K` frames, frame 0 cold, later
    /// frames warm from the detached previous latent, every exit supervised.
    pub fn seq_step(
        &mut self,
        batch: &crate::s2::seq::SeqBatch,
        lr_scale: f32,
        qat: bool,
    ) -> Result<StepStats> {
        let wv = model::exit_weight_vec(&self.cfg, None);
        let wts = Array::from_slice(&wv, &[wv.len() as i32]);
        let k = batch.frames.len();
        let (loss, grad_norm) = if self.compile {
            let names = sorted_names(&self.params);
            let cfg = self.cfg.clone();
            let f = self.seq_fn.entry(qat).or_insert_with(|| {
                CompiledGrad::new(names, move |p, a| {
                    let frames: Vec<S2Batch> =
                        (0..k).map(|t| dec_from(&a[t * DEC_ARRAYS..])).collect();
                    let o = k * DEC_ARRAYS;
                    model::seq_loss(p, &cfg, &frames, &a[o], &a[o + 1], &a[o + 2], qat)
                })
            });
            let mut inputs: Vec<Array> = batch.frames.iter().flat_map(dec_arrays).collect();
            inputs.extend([batch.on.clone(), batch.src.clone(), wts]);
            let (loss, grads) = f.call(&self.params, inputs)?;
            apply_update(&mut self.params, &mut self.opt, loss, &grads, lr_scale)?
        } else {
            let cfg = &self.cfg;
            train_step(&mut self.params, &mut self.opt, batch, lr_scale, |p, b| {
                model::seq_loss(p, cfg, &b.frames, &b.on, &b.src, &wts, qat)
            })?
        };
        self.state.step += 1;
        Ok(StepStats {
            loss,
            grad_norm,
            tokens: batch.n_tokens,
        })
    }

    /// Atomic checkpoint into `dir` (via `dir.tmp`).
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
        state.config = self.cfg.clone();
        let mut j = serde_json::to_value(&state)?;
        j["optimizer_step"] = self.opt.step.into();
        if let Some(n) = self.state.steps {
            j["frac"] = (self.state.step as f64 / n as f64).into();
        }
        std::fs::write(tmp.join("state.json"), serde_json::to_string_pretty(&j)?)?;
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

    pub fn load(dir: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(dir.join("state.json"))
            .with_context(|| format!("reading {}/state.json", dir.display()))?;
        let j: serde_json::Value = serde_json::from_str(&text)?;
        let opt_step = j["optimizer_step"].as_u64().unwrap_or(0);
        let state: S2State = serde_json::from_value(j)?;
        let params = params::load(&dir.join("model.safetensors"))?;
        let mut opt = Optimizer::new(state.opt.clone());
        opt.load_state(opt_step, &params::load(&dir.join("optimizer.safetensors"))?)?;
        Ok(Self {
            cfg: state.config.clone(),
            params,
            opt,
            state,
            compile: true,
            num_weight: model::NUM_LOSS_WEIGHT,
            mlm_fn: None,
            dec_fn: BTreeMap::new(),
            seq_fn: BTreeMap::new(),
            dis_beta: None,
            exit_kd: None,
        })
    }
}

/// Loads an s2 tokenizer and checks it against the config.
pub fn load_tok(path: &Path, cfg: &S2Config) -> Result<Tok> {
    let tok = Tok::load(path)?;
    tok.check_num()
        .with_context(|| format!("tokenizer {}", path.display()))?;
    if tok.vocab_size() > cfg.vocab_size {
        bail!(
            "tokenizer has {} tokens but {} has vocab_size {}",
            tok.vocab_size(),
            cfg.name,
            cfg.vocab_size
        );
    }
    Ok(tok)
}

// ----------------------------------------------------------------------------- prediction

/// Per-exit raw logits of every item (identity option order).
pub struct S2Pred {
    /// `[E][k]`.
    pub logits: Vec<Vec<f32>>,
    pub target: Vec<f32>,
    pub qtype: usize,
    pub src: String,
    pub group: String,
}

impl S2Pred {
    /// The s1-style prediction of exit `e`.
    pub fn at(&self, e: usize) -> Pred {
        Pred {
            logits: self.logits[e].clone(),
            target: self.target.clone(),
            qtype: self.qtype,
            src: self.src.clone(),
            group: self.group.clone(),
        }
    }
}

/// Runs the model over `items` in batches of `batch` rows (sorted by static
/// stream so repeated questions share it).
pub fn predict(
    p: &ParamMap,
    cfg: &S2Config,
    statics: &[StaticSeq],
    groups: &[String],
    items: &[&S2Item],
    batch: usize,
    qat: bool,
) -> Result<Vec<S2Pred>> {
    let mut order: Vec<usize> = (0..items.len()).collect();
    order.sort_by_key(|&i| {
        (
            statics[items[i].stat].tokens.len(),
            items[i].stat,
            items[i].state.len(),
        )
    });
    let mut out: Vec<Option<S2Pred>> = (0..items.len()).map(|_| None).collect();
    for chunk in order.chunks(batch.max(1)) {
        let rows: Vec<&S2Item> = chunk.iter().map(|&i| items[i]).collect();
        let b = make_batch(statics, &rows, 16);
        let logits = model::forward(p, cfg, &b, qat)?;
        let ks: Vec<usize> = rows.iter().map(|r| r.k()).collect();
        let per = model::logits_per_row(&logits, &ks)?;
        for (r, &i) in chunk.iter().enumerate() {
            let it = items[i];
            out[i] = Some(S2Pred {
                logits: per.iter().map(|e| e[r].clone()).collect(),
                target: it.target.clone(),
                qtype: it.qtype,
                src: it.src.clone(),
                group: groups.get(it.group).cloned().unwrap_or_default(),
            });
        }
    }
    Ok(out
        .into_iter()
        .map(|p| p.expect("every item predicted"))
        .collect())
}

/// Effective rank ([`crate::metrics::effective_rank`]) of the raw state latents of every exit,
/// over the valid state tokens of the first `max_items` items (one batch). The singular values
/// come from the `d × d` Gram matrix, on the CPU.
pub fn state_effective_rank(
    p: &ParamMap,
    cfg: &S2Config,
    statics: &[StaticSeq],
    items: &[&S2Item],
    max_items: usize,
    qat: bool,
) -> Result<Vec<f64>> {
    let rows: Vec<&S2Item> = items.iter().take(max_items).copied().collect();
    if rows.is_empty() {
        return Ok(Vec::new());
    }
    let b = make_batch(statics, &rows, 16);
    let (_, raw) = model::forward_carry(p, cfg, &b, qat, None)?;
    let valid = host_f32(&b.valid.as_dtype(Dtype::Float32)?)?;
    let n = valid.iter().filter(|&&v| v > 0.5).count() as i32;
    raw.iter()
        .map(|h| {
            let d = h.dim(2) as usize;
            let x = host_f32(&h.as_dtype(Dtype::Float32)?)?;
            let mut m = Vec::with_capacity(n as usize * d);
            for (r, _) in valid.iter().enumerate().filter(|(_, &v)| v > 0.5) {
                m.extend_from_slice(&x[r * d..(r + 1) * d]);
            }
            let s2 = mlx_rs::with_device(mlx_rs::Device::cpu(), || -> Result<Vec<f32>> {
                let a = Array::from_slice(&m, &[n, d as i32]);
                let gram = mlx_rs::ops::matmul(a.t(), &a)?;
                Ok(host_f32(&mlx_rs::linalg::svd(&gram)?.1)?)
            })?;
            let s: Vec<f32> = s2.iter().map(|v| v.max(0.0).sqrt()).collect();
            Ok(metrics::effective_rank(&s))
        })
        .collect()
}

/// Metrics of exit `e` per group key plus `all`.
pub fn exit_metrics(
    preds: &[S2Pred],
    e: usize,
    key: impl Fn(&S2Pred) -> String,
    temp: impl Fn(&Pred) -> f32,
) -> BTreeMap<String, Metrics> {
    let mut groups: BTreeMap<String, Vec<Pred>> = BTreeMap::new();
    let mut all = Vec::with_capacity(preds.len());
    for p in preds {
        let q = p.at(e);
        groups.entry(key(p)).or_default().push(q.clone());
        all.push(q);
    }
    let mut out: BTreeMap<String, Metrics> = groups
        .iter()
        .map(|(k, v)| (k.clone(), metrics::compute(v, &temp)))
        .collect();
    out.insert("all".into(), metrics::compute(&all, &temp));
    out
}

/// Deterministic subset of at most `max` items.
pub fn subset<'a>(items: &[&'a S2Item], max: usize, seed: u64) -> Vec<&'a S2Item> {
    let mut idx: Vec<usize> = (0..items.len()).collect();
    if idx.len() > max {
        Rng::new(seed).shuffle(&mut idx);
        idx.truncate(max);
        idx.sort_unstable();
    }
    idx.into_iter().map(|i| items[i]).collect()
}

// ----------------------------------------------------------------------------- bench

/// Settings of `sky-train s2-bench`.
#[derive(Debug, Clone)]
pub struct S2BenchArgs {
    pub config: PathBuf,
    pub mlm_batch: usize,
    pub mlm_seq: usize,
    pub dec_batch: usize,
    /// Distinct questions per decision batch.
    pub dec_questions: usize,
    pub static_len: usize,
    pub state_len: usize,
    pub options: usize,
    pub warmup: usize,
    pub steps: usize,
    pub bench_md: PathBuf,
    pub bf16: bool,
    pub compile: bool,
    pub qat: bool,
    /// Newton–Schulz polish iterations of the timed Muon updates.
    pub ns_polish: usize,
}

/// One benchmark row.
#[derive(Debug, Clone)]
pub struct S2BenchRow {
    pub model: String,
    pub params: usize,
    pub phase: String,
    pub batch: usize,
    pub shape: String,
    pub step_ms: f64,
    pub tok_s: f64,
    pub q_s: f64,
    pub peak_mb: f64,
    pub precision: &'static str,
}

fn random_stream(rng: &mut Rng, vocab: usize, len: usize) -> Stream {
    let mut s = Stream::default();
    for _ in 0..len {
        if rng.below(6) == 0 {
            s.ids.push(crate::num::NUM_ID);
            s.values.push(rng.below(200) as f32 - 100.0);
        } else {
            s.ids.push(6 + rng.below(vocab - 6) as u32);
            s.values.push(0.0);
        }
    }
    s
}

fn random_decision_batches(
    cfg: &S2Config,
    a: &S2BenchArgs,
    rng: &mut Rng,
    n: usize,
) -> Vec<S2Batch> {
    (0..n)
        .map(|_| {
            let statics: Vec<StaticSeq> = (0..a.dec_questions.max(1))
                .map(|_| {
                    let opt_len = (a.static_len / 2).div_ceil(a.options).max(2);
                    let head = random_stream(
                        rng,
                        cfg.vocab_size,
                        a.static_len.saturating_sub(opt_len * a.options + 2).max(4),
                    );
                    let opts: Vec<Stream> = (0..a.options)
                        .map(|_| random_stream(rng, cfg.vocab_size, opt_len - 1))
                        .collect();
                    data::build_static(&head, &opts, cfg.max_head_tokens.max(a.static_len))
                })
                .collect();
            let items: Vec<S2Item> = (0..a.dec_batch)
                .map(|i| {
                    let mut target = vec![0.0; a.options];
                    target[rng.below(a.options)] = 1.0;
                    S2Item {
                        stat: i % statics.len(),
                        state: data::build_state(
                            &random_stream(rng, cfg.vocab_size, a.state_len - 2),
                            cfg.max_state_tokens,
                        ),
                        qtype: rng.below(3),
                        target,
                        src: "bench".into(),
                        group: 0,
                    }
                })
                .collect();
            let refs: Vec<&S2Item> = items.iter().collect();
            make_batch(&statics, &refs, 16)
        })
        .collect()
}

fn peak_memory_mb() -> f64 {
    mlx_rs::memory::peak_memory()
        .map(|b| b as f64 / (1024.0 * 1024.0))
        .unwrap_or(f64::NAN)
}

fn time_steps<B>(
    s: &mut S2Session,
    batches: &[B],
    warmup: usize,
    steps: usize,
    step: impl Fn(&mut S2Session, &B) -> Result<StepStats>,
) -> Result<(f64, f64)> {
    for i in 0..warmup {
        step(s, &batches[i % batches.len()])?;
    }
    mlx_rs::memory::reset_peak_memory()?;
    let t0 = Instant::now();
    let mut tokens = 0usize;
    for i in 0..steps {
        let st = step(s, &batches[i % batches.len()])?;
        if !st.loss.is_finite() {
            bail!("non-finite loss during benchmark");
        }
        tokens += st.tokens;
    }
    let secs = t0.elapsed().as_secs_f64();
    Ok((secs * 1000.0 / steps as f64, tokens as f64 / secs))
}

/// Measures numeric-MLM and decision training throughput on random data and
/// appends the rows to the s2 section of `BENCH.md`.
pub fn bench(a: &S2BenchArgs) -> Result<Vec<S2BenchRow>> {
    let mut cfg = S2Config::load(&a.config)?;
    cfg.bf16 = a.bf16;
    let precision = if a.bf16 { "bf16" } else { "f32" };
    let mut rng = Rng::new(99);
    let mut out = Vec::new();
    let opt = OptConfig {
        ns_polish: a.ns_polish,
        ..OptConfig::default()
    };

    let p = model::init(&cfg, S2Parts::PRETRAIN, 1)?;
    let n = params::count(&p);
    let mut s = S2Session::new(cfg.clone(), p, opt.clone(), "bench", 1, 0.0);
    s.compile = a.compile;
    let batches: Vec<NumMlmBatch> = (0..4)
        .map(|_| {
            let rows: Vec<Stream> = (0..a.mlm_batch)
                .map(|_| {
                    let mut r = random_stream(&mut rng, cfg.vocab_size, a.mlm_seq - 2);
                    r.ids.insert(0, crate::config::CLS_ID);
                    r.values.insert(0, 0.0);
                    r.ids.push(crate::config::SEP_ID);
                    r.values.push(0.0);
                    r
                })
                .collect();
            mask_batch(&rows, cfg.vocab_size, 0.3, &mut rng)
        })
        .collect();
    let (ms, tok_s) = time_steps(&mut s, &batches, a.warmup, a.steps, |s, b| {
        s.mlm_step(b, 1.0)
    })?;
    out.push(S2BenchRow {
        model: cfg.name.clone(),
        params: n,
        phase: "num-mlm".into(),
        batch: a.mlm_batch,
        shape: format!("seq {}", a.mlm_seq),
        step_ms: ms,
        tok_s,
        q_s: 0.0,
        peak_mb: peak_memory_mb(),
        precision,
    });
    drop(s);

    let p = model::init(&cfg, S2Parts::DECISION, 2)?;
    let n = params::count(&p);
    let mut s = S2Session::new(cfg.clone(), p, opt.clone(), "bench", 2, 0.0);
    s.compile = a.compile;
    let batches = random_decision_batches(&cfg, a, &mut rng, 4);
    for qat in if a.qat {
        vec![false, true]
    } else {
        vec![false]
    } {
        let (ms, tok_s) = time_steps(&mut s, &batches, a.warmup, a.steps, |s, b| {
            s.decision_step(b, 1.0, qat)
        })?;
        out.push(S2BenchRow {
            model: cfg.name.clone(),
            params: n,
            phase: if qat {
                "decision+qat".into()
            } else {
                "decision".into()
            },
            batch: a.dec_batch,
            shape: format!(
                "{}q x static {} + state {}, {} opts",
                a.dec_questions, a.static_len, a.state_len, a.options
            ),
            step_ms: ms,
            tok_s,
            q_s: a.dec_batch as f64 * 1000.0 / ms,
            peak_mb: peak_memory_mb(),
            precision,
        });
    }
    append_bench_md(&a.bench_md, &out, a.warmup, a.steps, a.compile)?;
    Ok(out)
}

const S2_HEADER: &str = "## skycmd-s2 training throughput";

fn append_bench_md(
    path: &Path,
    rows: &[S2BenchRow],
    warmup: usize,
    steps: usize,
    compile: bool,
) -> Result<()> {
    let mut text = std::fs::read_to_string(path).unwrap_or_default();
    if !text.contains(S2_HEADER) {
        if !text.is_empty() && !text.ends_with("\n\n") {
            text.push('\n');
        }
        text.push_str(&format!(
            "{S2_HEADER}\n\n\
             Measured by `sky-train s2-bench` (mlx-rs 0.32, Metal) on random tokens with ~1/6 `[NUM]` tokens. \
             Warmup steps are excluded; timing covers forward + backward + optimizer update (Muon on the encoder \
             hidden matrices, AdamW elsewhere). `num-mlm` is the numeric MLM over the state stack (CE + 0.5 MSE). \
             `decision` runs the static encoder once per distinct question, the state encoder per row, and the \
             fusion head + scorer for every exit (deep supervision); `decision+qat` adds ternary fake-quant. \
             `tok/s` counts real (unpadded) input tokens: static tokens of the distinct questions plus state tokens. \
             `q/s` counts training questions per second. The GPU may be shared with other jobs while measuring.\n\n\
             | model | params | phase | precision | graph | batch | shape | step (ms) | tok/s | q/s | peak mem (MB) | steps |\n\
             |---|---:|---|---|---|---:|---|---:|---:|---:|---:|---|\n"
        ));
    }
    for r in rows {
        text.push_str(&format!(
            "| {} | {} | {} | {} | {} | {} | {} | {:.1} | {:.0} | {} | {:.0} | {}+{} |\n",
            r.model,
            r.params,
            r.phase,
            r.precision,
            if compile { "compiled" } else { "eager" },
            r.batch,
            r.shape,
            r.step_ms,
            r.tok_s,
            if r.q_s > 0.0 {
                format!("{:.0}", r.q_s)
            } else {
                "-".into()
            },
            r.peak_mb,
            warmup,
            steps
        ));
    }
    std::fs::write(path, text)?;
    Ok(())
}

// ----------------------------------------------------------------------------- schedule

/// Seconds of a time budget (0 for a step budget, which never reads it).
fn budget_secs(p: &Progress) -> f64 {
    match *p {
        Progress::Time { budget_secs } => budget_secs,
        Progress::Steps { .. } => 0.0,
    }
}

/// The budget as printed at start-up.
fn budget_label(p: &Progress) -> String {
    match *p {
        Progress::Time { budget_secs } => format!("{:.1} min", budget_secs / 60.0),
        Progress::Steps { steps } => format!("{steps} steps"),
    }
}

/// The progress shown at the end of a log line.
fn progress_label(p: &Progress, s: &S2State) -> String {
    match *p {
        Progress::Time { budget_secs } => {
            format!("{:.1}/{:.1} min", s.elapsed_secs / 60.0, budget_secs / 60.0)
        }
        Progress::Steps { steps } => {
            format!("step {}/{steps}, {:.1} min", s.step, s.elapsed_secs / 60.0)
        }
    }
}

/// Refuses to resume a run under another kind of budget or another step budget: the
/// schedule position (`frac`) would jump. A time budget may still be extended.
fn check_resume_budget(saved: Option<u64>, progress: &Progress) -> Result<()> {
    match (saved, progress.steps()) {
        (Some(n), Some(m)) if n != m => {
            bail!(
                "--steps {m} does not match the resumed run (--steps {n}); resume with --steps {n}"
            )
        }
        (Some(n), None) => {
            bail!("the resumed run has a step budget (--steps {n}); resume with --steps {n}, not --minutes")
        }
        (None, Some(_)) => bail!("the resumed run has a time budget; resume with --minutes"),
        _ => Ok(()),
    }
}

/// Applies the budget to a new or resumed session and `--init-params`; returns the step
/// at which `--stop-after` ends this invocation.
fn start(
    s: &mut S2Session,
    resumed: bool,
    progress: &Progress,
    stop_after: Option<u64>,
    init_params: Option<&Path>,
) -> Result<Option<u64>> {
    if resumed {
        check_resume_budget(s.state.steps, progress)?;
    }
    if let Progress::Time { budget_secs } = *progress {
        s.state.budget_secs = budget_secs;
    }
    s.state.steps = progress.steps();
    if let Some(path) = init_params {
        let src = params::load(path)?;
        replace_params(&mut s.params, &src)
            .with_context(|| format!("loading {}", path.display()))?;
        if resumed {
            println!(
                "--init-params replaces the resumed weights of step {} (optimizer state and RNG kept)",
                s.state.step
            );
        }
        println!("parameters from {} ({} tensors)", path.display(), src.len());
    }
    let stop_at = stop_step(s.state.step, stop_after)?;
    if let Some(at) = stop_at {
        println!("stopping after step {at}");
    }
    Ok(stop_at)
}

// ----------------------------------------------------------------------------- pretrain

/// Settings of `sky-train s2-pretrain`.
#[derive(Debug, Clone)]
pub struct S2PretrainArgs {
    pub config: PathBuf,
    pub tokenizer: PathBuf,
    pub text: PathBuf,
    /// Time or step budget of the whole run.
    pub progress: Progress,
    /// Steps to run in this invocation before saving and exiting.
    pub stop_after: Option<u64>,
    /// Parameters to start from (`--init-params`); the optimizer state stays local.
    pub init_params: Option<PathBuf>,
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
    pub compile: bool,
    /// Weight of the number MSE (SPEC: 0.5).
    pub num_weight: f32,
    /// Numeric-curriculum decision records mixed in as MLM text (SPEC-S3 §3).
    pub numeric: Option<PathBuf>,
    /// Fraction of the MLM rows drawn from `numeric`.
    pub numeric_weight: f64,
}

/// Mean validation (CE, MSE) of the numeric MLM.
fn mlm_val(s: &S2Session, val: &[NumMlmBatch]) -> Result<(f64, f64)> {
    let (mut ce, mut mse) = (0.0, 0.0);
    for b in val {
        let (c, m) = model::mlm_loss_parts(&s.params, &s.cfg, b)?;
        ce += c.item_exact::<f32>() as f64;
        mse += m.item_exact::<f32>() as f64;
    }
    let n = val.len().max(1) as f64;
    Ok((ce / n, mse / n))
}

/// Time-budgeted numeric MLM over the state stack; at the end writes
/// `<out>/init` with the decision initialization (static encoder copied from
/// the first `static_layers` blocks).
pub fn pretrain(a: &S2PretrainArgs) -> Result<()> {
    let mut cfg = S2Config::load(&a.config)?;
    cfg.bf16 = a.bf16;
    let tok = load_tok(&a.tokenizer, &cfg)?;
    std::fs::create_dir_all(&a.out)?;
    let t_load = Instant::now();
    let corpus = NumCorpus::load(
        &a.text,
        &tok,
        &a.tokenizer,
        &a.out.join("cache"),
        a.max_tokens,
    )?;
    let rows = Rows::new(&corpus.corpus, a.seq, 0.01, a.val_batches * a.batch);
    let fuse = cfg.fuse_numbers;
    let numeric = match &a.numeric {
        Some(path) => {
            let recs = crate::decision::read_records(path)?;
            let texts: Vec<String> = recs
                .iter()
                .filter(|r| r.split == "train")
                .map(crate::s2::numeric::mlm_text)
                .collect();
            let c = NumCorpus::from_texts(&tok, &texts)?;
            let r = Rows::new(&c.corpus, a.seq, 0.0, 0);
            println!(
                "numeric MLM: {} records, {} tokens, {} rows, weight {:.2}",
                texts.len(),
                c.corpus.tokens.len(),
                r.train.len(),
                a.numeric_weight
            );
            Some((c, r))
        }
        None => None,
    };
    println!(
        "corpus: {} records, {} tokens ({} numbers) -> {} train / {} val rows of {} ({:.1}s)",
        corpus.corpus.n_records,
        corpus.corpus.tokens.len(),
        corpus.n_numbers(),
        rows.train.len(),
        rows.val.len(),
        a.seq,
        t_load.elapsed().as_secs_f64()
    );
    if rows.train.is_empty() {
        bail!("corpus too small");
    }
    let last = a.out.join("last");
    let resumed = a.resume && last.join("state.json").exists();
    let mut s = if resumed {
        let s = S2Session::load(&last)?;
        println!("resumed from {} at step {}", last.display(), s.state.step);
        s
    } else {
        let p = model::init(&cfg, S2Parts::PRETRAIN, a.seed)?;
        let mut s = S2Session::new(
            cfg.clone(),
            p,
            a.opt.clone(),
            "s2-pretrain",
            a.seed,
            budget_secs(&a.progress),
        );
        s.state.extra =
            json!({"batch": a.batch, "seq": a.seq, "text": a.text, "tokenizer": a.tokenizer});
        s
    };
    let stop_at = start(
        &mut s,
        resumed,
        &a.progress,
        a.stop_after,
        a.init_params.as_deref(),
    )?;
    s.cfg.bf16 = a.bf16;
    s.compile = a.compile;
    s.num_weight = a.num_weight;
    println!(
        "{}: {} parameters (pretraining stack), batch {}x{}, budget {}",
        s.cfg.name,
        params::count(&s.params),
        a.batch,
        a.seq,
        budget_label(&a.progress)
    );
    let mut vrng = Rng::new(a.seed ^ 0x5EED);
    let val: Vec<NumMlmBatch> = rows
        .val
        .chunks(a.batch)
        .take(a.val_batches)
        .map(|c| {
            let rs: Vec<Stream> = c
                .iter()
                .map(|&st| data::maybe_fuse(corpus.row(a.seq, st), fuse))
                .collect();
            mask_batch(&rs, s.cfg.vocab_size, 0.3, &mut vrng)
        })
        .collect();
    let tok_json = std::fs::read(&a.tokenizer)?;
    let mut log = JsonLog::open(&a.out.join("log.jsonl"))?;
    let save = |s: &S2Session| s.save(&last, &[("tokenizer.json", tok_json.clone())]);
    let base = s.state.elapsed_secs;
    let t0 = Instant::now();
    let mut last_ckpt = Instant::now();
    let (mut wl, mut wt, mut ws, mut wtime) = (0.0f64, 0usize, 0u64, Instant::now());
    while keep_going(&a.progress, stop_at, s.state.step, s.state.elapsed_secs) {
        let frac = a.progress.frac(s.state.step, s.state.elapsed_secs);
        let lr_scale = wsd(frac) as f32;
        let seed = s.state.seed;
        let n_num = match &numeric {
            Some(_) => (0..a.batch)
                .filter(|_| s.state.rng.f64() < a.numeric_weight)
                .count(),
            None => 0,
        };
        let starts = s.state.cursor.next(&rows, seed, a.batch - n_num);
        let mut rs: Vec<Stream> = starts
            .iter()
            .map(|&st| data::maybe_fuse(corpus.row(a.seq, st), fuse))
            .collect();
        if let Some((c, r)) = &numeric {
            for _ in 0..n_num {
                let st = r.train[s.state.rng.below(r.train.len())];
                rs.push(data::maybe_fuse(c.row(a.seq, st), fuse));
            }
        }
        let batch = mask_batch(&rs, s.cfg.vocab_size, 0.3, &mut s.state.rng);
        let st = s.mlm_step(&batch, lr_scale)?;
        if !st.loss.is_finite() {
            bail!("non-finite loss at step {}", s.state.step);
        }
        s.state.elapsed_secs = base + t0.elapsed().as_secs_f64();
        (wl, wt, ws) = (wl + st.loss as f64, wt + st.tokens, ws + 1);
        let step = s.state.step;
        if step % a.log_every == 0 {
            let dt = wtime.elapsed().as_secs_f64();
            let mut rec = json!({"step": step, "loss": wl / ws as f64, "grad_norm": st.grad_norm, "lr_scale": lr_scale,
                "tok_s": wt as f64 / dt, "elapsed_min": s.state.elapsed_secs / 60.0, "epoch": s.state.cursor.epoch});
            if let Some(n) = s.state.steps {
                rec["steps"] = n.into();
                rec["frac"] = frac.into();
            }
            log.write(&rec)?;
            println!(
                "step {step:>6} loss {:.4} lr x{lr_scale:.3} {:.0} tok/s ({})",
                wl / ws as f64,
                wt as f64 / dt,
                progress_label(&a.progress, &s.state)
            );
            (wl, wt, ws, wtime) = (0.0, 0, 0, Instant::now());
        }
        if step % a.eval_every == 0 && !val.is_empty() {
            let (ce, mse) = mlm_val(&s, &val)?;
            let vl = ce + s.num_weight as f64 * mse;
            log.write(&json!({"step": step, "val_loss": vl, "val_ce": ce, "val_mse": mse}))?;
            println!("step {step:>6} val_loss {vl:.4} (token CE {ce:.4}, number MSE {mse:.4})");
        }
        if last_ckpt.elapsed().as_secs_f64() >= a.checkpoint_minutes * 60.0 {
            save(&s)?;
            last_ckpt = Instant::now();
        }
    }
    if !a.progress.done(s.state.step, s.state.elapsed_secs) {
        // --stop-after: save and exit; the validation and the decision init wait for the
        // invocation that ends the schedule.
        save(&s)?;
        println!(
            "stopped at step {}: checkpoint {}",
            s.state.step,
            last.display()
        );
        return Ok(());
    }
    let (ce, mse) = if val.is_empty() {
        (f64::NAN, f64::NAN)
    } else {
        mlm_val(&s, &val)?
    };
    let vl = ce + s.num_weight as f64 * mse;
    log.write(
        &json!({"step": s.state.step, "val_loss": vl, "val_ce": ce, "val_mse": mse, "final": true}),
    )?;
    save(&s)?;
    // Decision initialization.
    let mut init = model::init(&s.cfg, S2Parts::DECISION, a.seed)?;
    let n = expand_pretrained(&mut init, &s.params, &s.cfg)?;
    let dir = a.out.join("init");
    std::fs::create_dir_all(&dir)?;
    params::save(&dir.join("model.safetensors"), &init, None)?;
    std::fs::write(
        dir.join("config.json"),
        serde_json::to_string_pretty(&s.cfg)?,
    )?;
    std::fs::write(dir.join("tokenizer.json"), &tok_json)?;
    println!("done: {} steps, final val_loss {vl:.4} (token CE {ce:.4}, number MSE {mse:.4}); decision init ({n} tensors from the stack) in {}", s.state.step, dir.display());
    Ok(())
}

// ----------------------------------------------------------------------------- decide

/// Settings of `sky-train s2-decide`.
#[derive(Debug, Clone)]
pub struct S2DecideArgs {
    pub config: PathBuf,
    pub init: Option<PathBuf>,
    pub tokenizer: Option<PathBuf>,
    pub records: Vec<PathBuf>,
    pub weights: Option<String>,
    /// Time or step budget of the whole run.
    pub progress: Progress,
    /// Steps to run in this invocation before saving and exiting.
    pub stop_after: Option<u64>,
    /// Parameters to start from (`--init-params`); the optimizer state stays local.
    pub init_params: Option<PathBuf>,
    pub out: PathBuf,
    pub resume: bool,
    pub batch: usize,
    pub opt: OptConfig,
    pub seed: u64,
    pub qat_frac: f64,
    /// Stochastic depth (SPEC-S3 §3): with this probability a batch is
    /// supervised only up to a random earlier exit, skipping the later loops.
    pub loop_dropout: f64,
    pub log_every: u64,
    pub eval_every: u64,
    pub checkpoint_minutes: f64,
    pub val_max: usize,
    pub bf16: bool,
    pub compile: bool,
    /// Sequence padding multiple of the training batches (16 as in s2; 32
    /// also rounds the question count to 32 and the options to 4, so fewer
    /// distinct shapes get compiled).
    pub pad_to: usize,
    /// SPEC-S3 §7 sequence records (`s2-seqgen` output); needs a
    /// `temporal_carry` config.
    pub seq_records: Option<PathBuf>,
    /// Probability that a step is a temporal sequence step.
    pub seq_frac: f64,
    /// Probability that a warm frame's carry is dropped (cold start).
    pub carry_drop: f64,
    /// s4 deep improvement supervision: sharpening exponent of the first exit's target
    /// (rising linearly to 1 at the last exit); `None` disables it.
    pub dis: Option<f32>,
    /// s6 self-distillation of the early exits toward the deepest (`model::ExitKd`).
    pub exit_kd: Option<model::ExitKd>,
    /// Decay of an exponential moving average of the float weights (0 disables). The average
    /// is saved as a complete run in `<out>/last-ema` (calibrate and export it like `last`),
    /// and validated at the end as `ema-f32` / `ema-ternary`.
    pub ema: f32,
    /// Batches drawn at once and grouped by state length (1: every batch drawn alone).
    pub length_pool: usize,
}

/// Validation summary of one model mode.
fn validate(
    s: &S2Session,
    data: &S2Data,
    val: &[&S2Item],
    qat: bool,
    label: &str,
    log: &mut JsonLog,
) -> Result<f64> {
    if val.is_empty() {
        return Ok(f64::NAN);
    }
    let preds = predict(
        &s.params,
        &s.cfg,
        &data.statics,
        &data.groups,
        val,
        512,
        qat,
    )?;
    let mut rec = serde_json::Map::new();
    let mut acc_last = 0.0;
    for e in 0..s.cfg.n_exits() {
        let m = exit_metrics(&preds, e, |p| p.group.clone(), |_| 1.0);
        let line: Vec<String> = m
            .iter()
            .map(|(g, x)| format!("{g} {:.3}", x.accuracy))
            .collect();
        println!(
            "  val[{label}] exit {e} (L{}): acc {}  nll {:.4}",
            s.cfg.exits[e],
            line.join(" | "),
            m["all"].nll
        );
        acc_last = m["all"].accuracy;
        rec.insert(format!("exit{e}"), serde_json::to_value(&m)?);
    }
    let erank = state_effective_rank(&s.params, &s.cfg, &data.statics, val, 128, qat)?;
    let shown: Vec<String> = erank.iter().map(|r| format!("{r:.1}")).collect();
    println!(
        "  val[{label}] effective rank per exit: {} (of {})",
        shown.join(" "),
        s.cfg.hidden_size
    );
    let erank: serde_json::Map<_, _> = erank
        .iter()
        .enumerate()
        .map(|(e, r)| (format!("exit{e}"), json!(r)))
        .collect();
    log.write(&json!({"step": s.state.step, "mode": label, "val": rec, "erank": erank}))?;
    Ok(acc_last)
}

/// Time-budgeted decision training with deep supervision; QAT during the last
/// `qat_frac` of the budget.
pub fn decide(a: &S2DecideArgs) -> Result<()> {
    let mut cfg = S2Config::load(&a.config)?;
    cfg.bf16 = a.bf16;
    let tok_path = match (&a.tokenizer, &a.init) {
        (Some(t), _) => t.clone(),
        (None, Some(init)) => init.join("tokenizer.json"),
        (None, None) => bail!("--tokenizer is required without --init"),
    };
    let tok = load_tok(&tok_path, &cfg)?;
    let t_load = Instant::now();
    let data = data::load_cfg(&a.records, &tok, &cfg)?;
    let weights = source_weights(&data.groups, a.weights.as_deref())?;
    for (g, (name, w)) in data.groups.iter().zip(&weights).enumerate() {
        println!(
            "source {name}: {} train items, weight {w:.3}",
            data.train[g].len()
        );
    }
    println!(
        "val {} / test {} items, {} distinct questions, {} records skipped ({:.1}s)",
        data.val.len(),
        data.test.len(),
        data.statics.len(),
        data.skipped,
        t_load.elapsed().as_secs_f64()
    );
    if data.train.iter().all(Vec::is_empty) {
        bail!("no training items");
    }
    std::fs::create_dir_all(&a.out)?;
    let last = a.out.join("last");
    let resumed = a.resume && last.join("state.json").exists();
    let mut s = if resumed {
        let s = S2Session::load(&last)?;
        println!("resumed from {} at step {}", last.display(), s.state.step);
        s
    } else {
        let mut p = model::init(&cfg, S2Parts::DECISION, a.seed)?;
        if let Some(init) = &a.init {
            let file = if init.join("model.safetensors").exists() {
                init.join("model.safetensors")
            } else {
                init.join("init/model.safetensors")
            };
            let src = params::load(&file)?;
            if !src.contains_key("blk.0.attn_qkv.weight") {
                bail!("{} has no s2 state stack", file.display());
            }
            let n = expand_pretrained(&mut p, &src, &cfg)?;
            println!("initialized {n} tensors from {}", file.display());
        }
        let mut s = S2Session::new(
            cfg.clone(),
            p,
            a.opt.clone(),
            "s2-decide",
            a.seed,
            budget_secs(&a.progress),
        );
        s.state.extra = json!({"batch": a.batch, "records": a.records, "weights": weights, "groups": data.groups,
            "init": a.init, "tokenizer": tok_path, "qat_frac": a.qat_frac, "loop_dropout": a.loop_dropout});
        s
    };
    let stop_at = start(
        &mut s,
        resumed,
        &a.progress,
        a.stop_after,
        a.init_params.as_deref(),
    )?;
    s.cfg.bf16 = a.bf16;
    s.compile = a.compile;
    s.dis_beta = a.dis.map(|b0| model::dis_betas(s.cfg.n_exits(), b0));
    if let Some(beta) = &s.dis_beta {
        println!("deep improvement supervision: exit target exponents {beta:?}");
    }
    // Weight EMA (issue #38): resumed from `last-ema` when the run is, else from the weights.
    let ema_dir = a.out.join("last-ema");
    if !(0.0..1.0).contains(&a.ema) {
        bail!("--ema must be in [0, 1)");
    }
    let mut ema = if a.ema == 0.0 {
        None
    } else if resumed && ema_dir.join("state.json").exists() {
        Some(S2Session::load(&ema_dir)?.params)
    } else {
        Some(params::deep_clone(&s.params))
    };
    if ema.is_some() {
        println!(
            "weight EMA: decay {} (saved in {})",
            a.ema,
            ema_dir.display()
        );
    }
    s.exit_kd = a.exit_kd;
    if let Some(kd) = &s.exit_kd {
        println!(
            "exit self-distillation: weight {}, temperature {}",
            kd.weight, kd.temp
        );
    }
    println!(
        "{}: {} parameters, batch {}, budget {}, QAT over the last {:.0}%, loop dropout {}",
        s.cfg.name,
        params::count(&s.params),
        a.batch,
        budget_label(&a.progress),
        a.qat_frac * 100.0,
        a.loop_dropout
    );
    let all_val = data.split("val");
    let val = subset(&all_val, a.val_max, a.seed);
    let tok_json = std::fs::read(&tok_path)?;
    let mut log = JsonLog::open(&a.out.join("log.jsonl"))?;
    let save = |s: &S2Session, dir: &Path| s.save(dir, &[("tokenizer.json", tok_json.clone())]);
    let base = s.state.elapsed_secs;
    let t0 = Instant::now();
    let mut last_ckpt = Instant::now();
    let (mut wl, mut wt, mut ws, mut wq, mut wtime) =
        (0.0f64, 0usize, 0u64, 0usize, Instant::now());
    let mut pending: std::collections::VecDeque<S2Batch> = Default::default();
    let (mut skipped, mut consecutive) = (0usize, 0usize);
    let seqs = match &a.seq_records {
        Some(path) => {
            if !s.cfg.temporal_carry {
                bail!("--seq-records needs a config with temporal_carry");
            }
            let sd = crate::s2::seq::load(path, &tok, &s.cfg)?;
            println!(
                "temporal sequences: {} train sequences of {} frames, {:.0}% of the steps, carry dropped with p = {}",
                sd.train.len(),
                sd.k,
                a.seq_frac * 100.0,
                a.carry_drop
            );
            Some(sd)
        }
        None => None,
    };
    let mut seq_loss_sum = (0.0f64, 0usize);
    while keep_going(&a.progress, stop_at, s.state.step, s.state.elapsed_secs) {
        let frac = a.progress.frac(s.state.step, s.state.elapsed_secs);
        if !s.state.qat && qat_due(frac, a.qat_frac) {
            println!("step {}: starting ternary QAT", s.state.step);
            let before = validate(&s, &data, &val, false, "f32", &mut log)?;
            let ternary = validate(&s, &data, &val, true, "ternary", &mut log)?;
            println!(
                "  accuracy before ternarization {before:.4}, after (no QAT yet) {ternary:.4}"
            );
            log.write(&json!({"step": s.state.step, "qat_start": true, "acc_f32": before, "acc_ternary": ternary}))?;
            save(&s, &a.out.join("pre-qat"))?;
            s.state.qat = true;
        }
        let lr_scale = wsd(frac) as f32;
        let qat = s.state.qat;
        if let Some(sd) = &seqs {
            if s.state.rng.f64() < a.seq_frac {
                let n = (a.batch / sd.k).max(1);
                let ne = s.cfg.n_exits();
                let b = crate::s2::seq::sample(sd, ne, n, a.carry_drop, &mut s.state.rng, a.pad_to);
                let st = s.seq_step(&b, lr_scale, qat)?;
                if !st.loss.is_finite() {
                    bail!("non-finite sequence loss at step {}", s.state.step);
                }
                seq_loss_sum = (seq_loss_sum.0 + st.loss as f64, seq_loss_sum.1 + 1);
                s.state.elapsed_secs = base + t0.elapsed().as_secs_f64();
                if seq_loss_sum.1.is_multiple_of(200) {
                    let l = seq_loss_sum.0 / seq_loss_sum.1 as f64;
                    println!(
                        "step {:>6} sequence loss {l:.4} ({} sequence steps)",
                        s.state.step, seq_loss_sum.1
                    );
                    log.write(&json!({"step": s.state.step, "seq_loss": l}))?;
                    seq_loss_sum = (0.0, 0);
                }
                continue;
            }
        }
        let batch = if a.length_pool > 1 {
            if pending.is_empty() {
                pending = data::sample_batches_by_length(
                    &data,
                    &weights,
                    a.batch,
                    a.length_pool,
                    &mut s.state.rng,
                    a.pad_to,
                )
                .into();
            }
            pending.pop_front().expect("a pooled batch")
        } else {
            data::sample_batch(&data, &weights, a.batch, &mut s.state.rng, a.pad_to)
        };
        let n_exits = s.cfg.n_exits();
        let used = if n_exits > 1 && a.loop_dropout > 0.0 && s.state.rng.f64() < a.loop_dropout {
            Some(1 + s.state.rng.below(n_exits - 1))
        } else {
            None
        };
        let st = s.decision_step_exits(&batch, lr_scale, qat, used)?;
        if !st.loss.is_finite() {
            // The update was not applied (`apply_update`): skip the batch, unless the
            // training itself has diverged.
            skipped += 1;
            consecutive += 1;
            println!(
                "step {}: non-finite loss, batch skipped ({} rows, state shape {:?}, static shape {:?})",
                s.state.step,
                batch.n_rows,
                batch.ids.shape(),
                batch.st_ids.shape()
            );
            log.write(
                &json!({"step": s.state.step, "skipped_batch": true, "skipped_total": skipped}),
            )?;
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
        s.state.elapsed_secs = base + t0.elapsed().as_secs_f64();
        (wl, wt, ws, wq) = (
            wl + st.loss as f64,
            wt + st.tokens,
            ws + 1,
            wq + batch.n_rows,
        );
        let step = s.state.step;
        if step % a.log_every == 0 {
            let dt = wtime.elapsed().as_secs_f64();
            let mb =
                |r: std::result::Result<usize, _>| r.map(|b| b as f64 / 1e6).unwrap_or(f64::NAN);
            let mut rec = json!({"step": step, "loss": wl / ws as f64, "grad_norm": st.grad_norm, "lr_scale": lr_scale, "qat": qat,
                "tok_s": wt as f64 / dt, "q_s": wq as f64 / dt, "elapsed_min": s.state.elapsed_secs / 60.0,
                "mlx_active_mb": mb(mlx_rs::memory::active_memory()),
                "mlx_cache_mb": mb(mlx_rs::memory::cache_memory())});
            if let Some(n) = s.state.steps {
                rec["steps"] = n.into();
                rec["frac"] = frac.into();
            }
            log.write(&rec)?;
            println!(
                "step {step:>6} loss {:.4} lr x{lr_scale:.3}{} {:.0} tok/s {:.0} q/s ({})",
                wl / ws as f64,
                if qat { " qat" } else { "" },
                wt as f64 / dt,
                wq as f64 / dt,
                progress_label(&a.progress, &s.state)
            );
            (wl, wt, ws, wq, wtime) = (0.0, 0, 0, 0, Instant::now());
        }
        if step % a.eval_every == 0 {
            validate(
                &s,
                &data,
                &val,
                qat,
                if qat { "ternary" } else { "f32" },
                &mut log,
            )?;
        }
        if last_ckpt.elapsed().as_secs_f64() >= a.checkpoint_minutes * 60.0 {
            save(&s, &last)?;
            if let Some(e) = &mut ema {
                with_params(&mut s, e, |s| save(s, &ema_dir))?;
            }
            last_ckpt = Instant::now();
        }
    }
    if !a.progress.done(s.state.step, s.state.elapsed_secs) {
        // --stop-after: save and exit without the final validation.
        save(&s, &last)?;
        if let Some(e) = &mut ema {
            with_params(&mut s, e, |s| save(s, &ema_dir))?;
        }
        println!(
            "stopped at step {}: checkpoint {}",
            s.state.step,
            last.display()
        );
        return Ok(());
    }
    println!("done: {} steps", s.state.step);
    let f = validate(&s, &data, &val, false, "f32", &mut log)?;
    let t = validate(&s, &data, &val, true, "ternary", &mut log)?;
    println!("final val accuracy (last exit): latent f32 {f:.4}, ternary {t:.4}");
    log.write(&json!({"step": s.state.step, "final": true, "acc_f32": f, "acc_ternary": t}))?;
    save(&s, &last)?;
    println!("checkpoint {}", last.display());
    if let Some(e) = &mut ema {
        with_params(&mut s, e, |s| {
            let f = validate(s, &data, &val, false, "ema-f32", &mut log)?;
            let t = validate(s, &data, &val, true, "ema-ternary", &mut log)?;
            println!("final val accuracy of the weight EMA: latent f32 {f:.4}, ternary {t:.4}");
            log.write(&json!({"step": s.state.step, "final": true, "ema": true, "acc_f32": f, "acc_ternary": t}))?;
            save(s, &ema_dir)
        })?;
        println!("EMA checkpoint {}", ema_dir.display());
    }
    Ok(())
}

/// Runs `f` on the session with `params` in place of its weights, then puts them back.
fn with_params<T>(
    s: &mut S2Session,
    params: &mut ParamMap,
    f: impl FnOnce(&mut S2Session) -> Result<T>,
) -> Result<T> {
    std::mem::swap(&mut s.params, params);
    let r = f(s);
    std::mem::swap(&mut s.params, params);
    r
}

/// Question type name of a prediction (for reports).
pub fn qtype_name(p: &S2Pred) -> String {
    QTYPES[p.qtype].to_string()
}

/// Host copy of an array (re-exported for the export module).
pub fn host(a: &Array) -> Result<Vec<f32>> {
    Ok(host_f32(a)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{s2::data::tests::s2_tok, sequence::DecisionRecord, test_util};

    pub(crate) fn synthetic(tok: &Tok, cfg: &S2Config, n: usize, seed: u64) -> S2Data {
        // Learnable: the right option is the one labelled "safe".
        let dir = tempfile::tempdir().unwrap();
        let mut rng = Rng::new(seed);
        let mut lines = Vec::new();
        for i in 0..n {
            let k = 2 + i % 3;
            let ans = rng.below(k);
            let opts: Vec<String> = (0..k)
                .map(|j| {
                    format!(
                        "{} lane {}",
                        if j == ans { "safe" } else { "blocked" },
                        j * 3 + 1
                    )
                })
                .collect();
            let rec = DecisionRecord {
                src: format!("syn{}", i % 2),
                t: ["choice", "noul", "choice"][i % 3].into(),
                ins: "which lane is free?".into(),
                opts,
                state: format!("free {} wind {}", ans * 3 + 1, rng.below(9)),
                target: (0..k).map(|j| if j == ans { 1.0 } else { 0.0 }).collect(),
                split: if i % 10 == 0 {
                    "val".into()
                } else {
                    "train".into()
                },
            };
            lines.push(serde_json::to_string(&rec).unwrap());
        }
        let path = dir.path().join("syn.jsonl");
        std::fs::write(&path, lines.join("\n")).unwrap();
        data::load_cfg(&[path], tok, cfg).unwrap()
    }

    #[test]
    fn length_pooled_batches_cover_the_pool() {
        let _g = test_util::lock();
        mlx_rs::with_device(mlx_rs::Device::cpu(), || {
            let (_d, tok) = s2_tok();
            let cfg = S2Config {
                hidden_size: 64,
                intermediate_size: 32,
                ..S2Config::tiny(262)
            };
            let data = synthetic(&tok, &cfg, 60, 1);
            let bs = data::sample_batches_by_length(&data, &[1.0], 8, 4, &mut Rng::new(2), 16);
            assert_eq!(bs.len(), 4);
            assert!(bs.iter().all(|b| b.n_rows == 8));
        });
    }

    #[test]
    fn training_reduces_loss_and_resumes() {
        let _g = test_util::lock();
        mlx_rs::with_device(mlx_rs::Device::cpu(), || {
            let (_d, tok) = s2_tok();
            let cfg = S2Config {
                hidden_size: 64,
                intermediate_size: 32,
                ..S2Config::tiny(262)
            };
            let data = synthetic(&tok, &cfg, 60, 1);
            assert!(data.statics.len() > 3);
            let p = model::init(&cfg, S2Parts::DECISION, 3).unwrap();
            let mut s = S2Session::new(
                cfg.clone(),
                p,
                OptConfig {
                    adam_lr: 3e-3,
                    muon_lr: 3e-3,
                    ..Default::default()
                },
                "s2-decide",
                5,
                60.0,
            );
            let fixed = data::sample_batch(&data, &[1.0], 16, &mut Rng::new(1), 16);
            let l0 = model::decision_loss(&s.params, &cfg, &fixed, false)
                .unwrap()
                .item_exact::<f32>();
            for i in 0..40 {
                let b = data::sample_batch(&data, &[1.0], 16, &mut s.state.rng, 16);
                s.decision_step(&b, 1.0, i >= 30).unwrap();
            }
            let l1 = model::decision_loss(&s.params, &cfg, &fixed, false)
                .unwrap()
                .item_exact::<f32>();
            assert!(l1 < l0, "loss {l0} -> {l1}");
            let val: Vec<&S2Item> = data.val.iter().collect();
            let erank =
                state_effective_rank(&s.params, &cfg, &data.statics, &val, 32, false).unwrap();
            assert_eq!(erank.len(), cfg.n_exits());
            assert!(erank.iter().all(|&r| r > 1.0 && r <= 64.0), "{erank:?}");
            let tmp = tempfile::tempdir().unwrap();
            let dir = tmp.path().join("last");
            s.save(&dir, &[]).unwrap();
            let r = S2Session::load(&dir).unwrap();
            assert_eq!(r.state.step, 40);
            assert_eq!(r.cfg, s.cfg);
            for (k, v) in &s.params {
                assert!(v.all_close(&r.params[k], 0.0, 0.0, None).unwrap(), "{k}");
            }
            let val = data.split("val");
            let preds =
                predict(&s.params, &cfg, &data.statics, &data.groups, &val, 4, true).unwrap();
            assert_eq!(preds.len(), val.len());
            assert_eq!(preds[0].logits.len(), 2);
            // The EMA weights swap in for a save or a validation and always swap back.
            let mut other = model::init(&cfg, S2Parts::DECISION, 9).unwrap();
            let before = s.params["cls.weight"].clone();
            let seen =
                with_params(&mut s, &mut other, |s| Ok(s.params["cls.weight"].clone())).unwrap();
            assert!(!seen.all_close(&before, 0.0, 0.0, None).unwrap());
            let err = with_params(&mut s, &mut other, |_| -> Result<()> { bail!("boom") });
            assert!(err.is_err());
            assert!(s.params["cls.weight"]
                .all_close(&before, 0.0, 0.0, None)
                .unwrap());
            // Self-distillation trains through both the compiled and the eager step, with
            // loop dropout (fewer supervised exits than computed ones).
            for compile in [true, false] {
                let mut k = S2Session::new(
                    cfg.clone(),
                    model::init(&cfg, S2Parts::DECISION, 4).unwrap(),
                    OptConfig::default(),
                    "s2-decide",
                    6,
                    60.0,
                );
                k.compile = compile;
                k.exit_kd = Some(model::ExitKd {
                    weight: 1.0,
                    temp: 2.0,
                });
                for n in [None, Some(1)] {
                    let st = k.decision_step_exits(&fixed, 1.0, false, n).unwrap();
                    assert!(st.loss.is_finite(), "compile {compile}, exits {n:?}");
                }
            }
        });
    }
}
