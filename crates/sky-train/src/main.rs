//! `sky-train`: benchmark, pretrain, decide, calibrate, export and eval the
//! skycmd-s1 decision models.

use std::path::PathBuf;

use anyhow::Result;
use clap::{Args, Parser, Subcommand};
use sky_train::{
    export,
    optim::OptConfig,
    train::{self, BenchArgs, DecideArgs, PretrainArgs},
};

#[derive(Parser)]
#[command(name = "sky-train", version, about = "Train skycmd-s1 decision models with mlx-rs")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

/// Optimizer flags shared by the training phases.
#[derive(Args, Clone)]
struct OptFlags {
    /// Peak AdamW learning rate (embeddings, norms, head, scorer).
    #[arg(long)]
    adam_lr: Option<f32>,
    /// Peak Muon learning rate (encoder hidden matrices).
    #[arg(long)]
    muon_lr: Option<f32>,
    /// Decoupled AdamW weight decay.
    #[arg(long, default_value_t = 0.01)]
    weight_decay: f32,
    /// Global gradient-norm clip (0 disables).
    #[arg(long, default_value_t = 1.0)]
    clip_norm: f32,
}

impl OptFlags {
    fn build(&self, adam_lr: f32, muon_lr: f32) -> OptConfig {
        OptConfig {
            adam_lr: self.adam_lr.unwrap_or(adam_lr),
            muon_lr: self.muon_lr.unwrap_or(muon_lr),
            weight_decay: self.weight_decay,
            clip_norm: self.clip_norm,
            ..OptConfig::default()
        }
    }
}

fn split_paths(s: &str) -> Vec<PathBuf> {
    s.split(',').filter(|p| !p.trim().is_empty()).map(|p| PathBuf::from(p.trim())).collect()
}

/// Compute precision of the training step.
#[derive(Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
enum Precision {
    /// bfloat16 matmuls/activations, f32 master weights and losses.
    Bf16,
    /// Full float32.
    F32,
}

#[derive(Subcommand)]
enum Cmd {
    /// Measure MLM and decision training throughput on random tokens; appends to BENCH.md.
    Bench {
        #[arg(long)]
        config: PathBuf,
        #[arg(long, default_value_t = 128)]
        mlm_batch: usize,
        #[arg(long, default_value_t = 128)]
        mlm_seq: usize,
        #[arg(long, default_value_t = 64)]
        dec_batch: usize,
        #[arg(long, default_value_t = 192)]
        dec_seq: usize,
        #[arg(long, default_value_t = 4)]
        dec_options: usize,
        #[arg(long, default_value_t = 10)]
        warmup: usize,
        #[arg(long, default_value_t = 50)]
        steps: usize,
        #[arg(long, default_value = "BENCH.md")]
        bench_md: PathBuf,
        #[arg(long, value_enum, default_value_t = Precision::Bf16)]
        precision: Precision,
    },
    /// Time-budgeted MLM pretraining (30% masking, packed rows).
    Pretrain {
        #[arg(long)]
        config: PathBuf,
        /// tokenizer.json (data/tok/tok-<vocab>.json).
        #[arg(long)]
        tokenizer: PathBuf,
        /// Directory (or file) of MLM JSONL records.
        #[arg(long)]
        text: PathBuf,
        /// Training time budget in minutes (WSD decay starts at 80%).
        #[arg(long)]
        minutes: f64,
        #[arg(long)]
        out: PathBuf,
        /// Continue from <out>/last.
        #[arg(long)]
        resume: bool,
        #[arg(long, default_value_t = 128)]
        batch: usize,
        #[arg(long, default_value_t = 128)]
        seq: usize,
        #[command(flatten)]
        opt: OptFlags,
        #[arg(long, default_value_t = 0)]
        seed: u64,
        /// Maximum corpus size in tokens.
        #[arg(long, default_value_t = 300_000_000)]
        max_tokens: usize,
        #[arg(long, default_value_t = 20)]
        log_every: u64,
        #[arg(long, default_value_t = 500)]
        eval_every: u64,
        #[arg(long, default_value_t = 10.0)]
        checkpoint_minutes: f64,
        #[arg(long, default_value_t = 16)]
        val_batches: usize,
        #[arg(long, value_enum, default_value_t = Precision::Bf16)]
        precision: Precision,
    },
    /// Time-budgeted decision training from a pretrained encoder.
    Decide {
        #[arg(long)]
        config: PathBuf,
        /// Pretraining checkpoint (runs/<name>/pretrain/last).
        #[arg(long)]
        init: Option<PathBuf>,
        /// tokenizer.json (defaults to <init>/tokenizer.json).
        #[arg(long)]
        tokenizer: Option<PathBuf>,
        /// Comma-separated record directories, e.g. data/decisions,data/games.
        #[arg(long)]
        records: String,
        /// Per-source sampling weights `name=w,...` (default games 0.5, others share 0.5).
        #[arg(long)]
        weights: Option<String>,
        #[arg(long)]
        minutes: f64,
        #[arg(long)]
        out: PathBuf,
        #[arg(long)]
        resume: bool,
        #[arg(long, default_value_t = 64)]
        batch: usize,
        #[command(flatten)]
        opt: OptFlags,
        #[arg(long, default_value_t = 0)]
        seed: u64,
        #[arg(long, default_value_t = 20)]
        log_every: u64,
        #[arg(long, default_value_t = 500)]
        eval_every: u64,
        #[arg(long, default_value_t = 10.0)]
        checkpoint_minutes: f64,
        /// Maximum number of validation questions per evaluation.
        #[arg(long, default_value_t = 4000)]
        val_max: usize,
        #[arg(long, value_enum, default_value_t = Precision::Bf16)]
        precision: Precision,
    },
    /// Fit per-type and per-bucket temperatures on the val split (writes rl_agent_config.json).
    Calibrate {
        #[arg(long)]
        run: PathBuf,
        #[arg(long)]
        records: String,
        #[arg(long, default_value = "val")]
        split: String,
        /// Minimum questions for a per-bucket temperature.
        #[arg(long, default_value_t = 50)]
        min_bucket: usize,
    },
    /// Export to the oscar / HF directory layout.
    Export {
        #[arg(long)]
        run: PathBuf,
        #[arg(long)]
        out: PathBuf,
    },
    /// Per-source metrics table -> runs/<name>/report.md.
    Eval {
        #[arg(long)]
        run: PathBuf,
        #[arg(long)]
        records: String,
        #[arg(long, default_value = "test")]
        split: String,
        /// Report path (default <run>/report.md).
        #[arg(long)]
        report: Option<PathBuf>,
    },
}

fn main() -> Result<()> {
    match Cli::parse().cmd {
        Cmd::Bench { config, mlm_batch, mlm_seq, dec_batch, dec_seq, dec_options, warmup, steps, bench_md, precision } => {
            let rows = train::bench(&BenchArgs {
                config,
                mlm_batch,
                mlm_seq,
                dec_batch,
                dec_seq,
                dec_options,
                warmup,
                steps,
                bench_md: bench_md.clone(),
                bf16: precision == Precision::Bf16,
            })?;
            for r in rows {
                println!(
                    "{} {:>9} params {:<8} {} batch {:>4} x {:>3}: {:>7.1} ms/step {:>9.0} tok/s peak {:>6.0} MB",
                    r.model, r.params, r.phase, r.precision, r.batch, r.seq, r.step_ms, r.tok_s, r.peak_mb
                );
            }
            println!("appended to {}", bench_md.display());
        }
        Cmd::Pretrain {
            config,
            tokenizer,
            text,
            minutes,
            out,
            resume,
            batch,
            seq,
            opt,
            seed,
            max_tokens,
            log_every,
            eval_every,
            checkpoint_minutes,
            val_batches,
            precision,
        } => train::pretrain(&PretrainArgs {
            config,
            tokenizer,
            text,
            minutes,
            out,
            resume,
            batch,
            seq,
            opt: opt.build(3e-3, 0.02),
            seed,
            max_tokens,
            log_every,
            eval_every,
            checkpoint_minutes,
            val_batches,
            bf16: precision == Precision::Bf16,
        })?,
        Cmd::Decide {
            config,
            init,
            tokenizer,
            records,
            weights,
            minutes,
            out,
            resume,
            batch,
            opt,
            seed,
            log_every,
            eval_every,
            checkpoint_minutes,
            val_max,
            precision,
        } => train::decide(&DecideArgs {
            config,
            init,
            tokenizer,
            records: split_paths(&records),
            weights,
            minutes,
            out,
            resume,
            batch,
            opt: opt.build(1e-3, 0.01),
            seed,
            log_every,
            eval_every,
            checkpoint_minutes,
            val_max,
            bf16: precision == Precision::Bf16,
        })?,
        Cmd::Calibrate { run, records, split, min_bucket } => {
            export::calibrate(&run, &split_paths(&records), &split, min_bucket)?;
        }
        Cmd::Export { run, out } => export::export(&run, &out)?,
        Cmd::Eval { run, records, split, report } => {
            export::eval(&run, &split_paths(&records), &split, report.as_deref())?;
        }
    }
    Ok(())
}
