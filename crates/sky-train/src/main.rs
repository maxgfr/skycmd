//! `sky-train`: benchmark, pretrain, decide, calibrate, export and eval the
//! skycmd-s1 decision models, and the `s2-*` equivalents for skycmd-s2.

use std::path::PathBuf;

use anyhow::Result;
use clap::{Args, Parser, Subcommand};
use sky_train::{
    export,
    optim::OptConfig,
    s2::{
        self,
        gguf::WeightFormat,
        train::{S2BenchArgs, S2DecideArgs, S2PretrainArgs},
    },
    session::Progress,
    train::{self, BenchArgs, DecideArgs, PretrainArgs},
};

#[derive(Parser)]
#[command(
    name = "sky-train",
    version,
    about = "Train skycmd-s1 and skycmd-s2 decision models with mlx-rs"
)]
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
    /// Peak Muon learning rate (encoder hidden matrices); defaults to the AdamW one.
    #[arg(long)]
    muon_lr: Option<f32>,
    /// Decoupled weight decay (AdamW and Muon matrices).
    #[arg(long, default_value_t = 0.01)]
    weight_decay: f32,
    /// Global gradient-norm clip (0 disables).
    #[arg(long, default_value_t = 1.0)]
    clip_norm: f32,
    /// AdamW on the encoder hidden matrices too (no Muon).
    #[arg(long)]
    no_muon: bool,
    /// Newton–Schulz polish iterations after the fast ones.
    #[arg(long, default_value_t = 2)]
    ns_polish: usize,
}

impl OptFlags {
    fn build(&self, adam_lr: f32) -> OptConfig {
        let adam_lr = self.adam_lr.unwrap_or(adam_lr);
        OptConfig {
            adam_lr,
            muon_lr: self.muon_lr.unwrap_or(adam_lr),
            weight_decay: self.weight_decay,
            clip_norm: self.clip_norm,
            muon: !self.no_muon,
            ns_polish: self.ns_polish,
            ..OptConfig::default()
        }
    }
}

fn split_paths(s: &str) -> Vec<PathBuf> {
    s.split(',')
        .filter(|p| !p.trim().is_empty())
        .map(|p| PathBuf::from(p.trim()))
        .collect()
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
    /// s2: numeric-MLM training throughput on random data; appends to the s2 section of BENCH.md.
    S2Bench {
        #[arg(long)]
        config: PathBuf,
        #[arg(long, default_value_t = 128)]
        mlm_batch: usize,
        #[arg(long, default_value_t = 128)]
        mlm_seq: usize,
        #[arg(long, default_value_t = 64)]
        dec_batch: usize,
        /// Distinct questions (static streams) per decision batch.
        #[arg(long, default_value_t = 64)]
        dec_questions: usize,
        #[arg(long, default_value_t = 64)]
        static_len: usize,
        #[arg(long, default_value_t = 32)]
        state_len: usize,
        #[arg(long, default_value_t = 4)]
        options: usize,
        #[arg(long, default_value_t = 10)]
        warmup: usize,
        #[arg(long, default_value_t = 50)]
        steps: usize,
        #[arg(long, default_value = "BENCH.md")]
        bench_md: PathBuf,
        #[arg(long, value_enum, default_value_t = Precision::F32)]
        precision: Precision,
        /// Run the graphs eagerly instead of compiling them.
        #[arg(long)]
        no_compile: bool,
        /// Also time the QAT (ternary fake-quant) decision step.
        #[arg(long)]
        qat: bool,
        /// Newton–Schulz polish iterations of the timed Muon updates.
        #[arg(long, default_value_t = 2)]
        ns_polish: usize,
    },
    /// s2: numeric MLM over the state stack (CE + 0.5 MSE on masked numbers), then the decision init.
    S2Pretrain {
        #[arg(long)]
        config: PathBuf,
        /// s2 tokenizer.json with [NUM] = 5 (data/tok/tok-s2-<vocab>.json).
        #[arg(long)]
        tokenizer: PathBuf,
        #[arg(long)]
        text: PathBuf,
        /// Training time budget in minutes (the default schedule).
        #[arg(long)]
        minutes: Option<f64>,
        /// Step budget of the whole run instead of --minutes: the learning rate, the QAT
        /// switch and the stop follow step / steps, the same on every machine.
        #[arg(long)]
        steps: Option<u64>,
        /// Run this many steps from the starting step, save <out>/last and exit (the final
        /// validation and the decision init wait for the end of the schedule).
        #[arg(long)]
        stop_after: Option<u64>,
        /// Start from these parameters (safetensors with the model's tensor names and
        /// shapes); with --resume the optimizer state and RNG still come from <out>/last.
        #[arg(long)]
        init_params: Option<PathBuf>,
        #[arg(long)]
        out: PathBuf,
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
        #[arg(long, value_enum, default_value_t = Precision::F32)]
        precision: Precision,
        #[arg(long)]
        no_compile: bool,
        /// Weight of the masked-number MSE in the loss (SPEC-S2: 0.5).
        #[arg(long, default_value_t = 0.5)]
        num_weight: f32,
        /// Numeric-curriculum records (s2-numeric output) mixed in as MLM text.
        #[arg(long)]
        numeric: Option<PathBuf>,
        /// Fraction of MLM rows drawn from --numeric.
        #[arg(long, default_value_t = 0.1)]
        numeric_weight: f64,
    },
    /// s2: decision training with deep supervision over the exits and ternary QAT at the end.
    S2Decide {
        #[arg(long)]
        config: PathBuf,
        /// Pretraining run (runs/<name>/pretrain, uses its init/) or a checkpoint directory.
        #[arg(long)]
        init: Option<PathBuf>,
        /// tokenizer.json (defaults to <init>/tokenizer.json).
        #[arg(long)]
        tokenizer: Option<PathBuf>,
        #[arg(long)]
        records: String,
        #[arg(long)]
        weights: Option<String>,
        /// Training time budget in minutes (the default schedule).
        #[arg(long)]
        minutes: Option<f64>,
        /// Step budget of the whole run instead of --minutes: the learning rate, the QAT
        /// switch and the stop follow step / steps, the same on every machine.
        #[arg(long)]
        steps: Option<u64>,
        /// Run this many steps from the starting step, save <out>/last and exit (the final
        /// validation waits for the end of the schedule).
        #[arg(long)]
        stop_after: Option<u64>,
        /// Start from these parameters (safetensors with the model's tensor names and
        /// shapes); with --resume the optimizer state and RNG still come from <out>/last.
        #[arg(long)]
        init_params: Option<PathBuf>,
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
        /// Fraction of the budget (at the end) trained with ternary fake-quant
        /// (0.9 = QAT after a 10% float warm-up).
        #[arg(long, default_value_t = 0.3)]
        qat_frac: f64,
        /// Stochastic depth: probability that a batch is supervised only up to
        /// a random earlier exit (its later loops skipped).
        #[arg(long, default_value_t = 0.0)]
        loop_dropout: f64,
        /// s4 deep improvement supervision: exit e trains toward normalize(target^beta_e),
        /// beta rising linearly from this value at the first exit to 1 at the last.
        #[arg(long)]
        dis: Option<f32>,
        /// Batches drawn at once and grouped by state length, so that a batch is padded to its
        /// own longest state (1: every batch drawn alone).
        #[arg(long, default_value_t = 1)]
        length_pool: usize,
        /// Padding multiple of the training batch shapes (32: coarser buckets, fewer compiles).
        #[arg(long, default_value_t = 16)]
        pad_to: usize,
        /// SPEC-S3 §7 temporal sequences (s2-seqgen output); needs temporal_carry in the config.
        #[arg(long)]
        seq_records: Option<PathBuf>,
        /// Probability that a step trains on temporal sequences.
        #[arg(long, default_value_t = 0.5)]
        seq_frac: f64,
        /// Probability that a warm frame starts cold anyway.
        #[arg(long, default_value_t = 0.3)]
        carry_drop: f64,
        #[arg(long, default_value_t = 20)]
        log_every: u64,
        #[arg(long, default_value_t = 1000)]
        eval_every: u64,
        #[arg(long, default_value_t = 10.0)]
        checkpoint_minutes: f64,
        #[arg(long, default_value_t = 4000)]
        val_max: usize,
        #[arg(long, value_enum, default_value_t = Precision::F32)]
        precision: Precision,
        #[arg(long)]
        no_compile: bool,
    },
    /// s2: temperatures per (exit, type, bucket) and τ per type, for f32 and ternary (calibration.json).
    S2Calibrate {
        #[arg(long)]
        run: PathBuf,
        #[arg(long)]
        records: String,
        #[arg(long, default_value = "val")]
        split: String,
        #[arg(long, default_value_t = 50)]
        min_bucket: usize,
    },
    /// s2: export the calibrated model as a skycmd-s2 GGUF.
    S2Export {
        #[arg(long)]
        run: PathBuf,
        #[arg(long)]
        out: PathBuf,
        /// f32, f16 or ternary.
        #[arg(long, default_value = "f32")]
        format: String,
    },
    /// s2: per-source / per-exit metrics, early exit, permutation invariance, coherence -> report.md.
    S2Eval {
        #[arg(long)]
        run: PathBuf,
        #[arg(long)]
        records: String,
        #[arg(long, default_value = "test")]
        split: String,
        /// Exported GGUF files to evaluate as well (comma-separated).
        #[arg(long)]
        gguf: Option<String>,
        #[arg(long)]
        report: Option<PathBuf>,
        /// Evaluate only the --gguf files (skip the checkpoint's latent f32 and fake-quant rows).
        #[arg(long)]
        gguf_only: bool,
    },
    /// s2: tiny random model -> GGUF (f32 + ternary) and golden outputs for sky-infer.
    S2Golden {
        /// s2 tokenizer.json (default: a built-in byte-level tokenizer with [NUM] = 5).
        #[arg(long)]
        tokenizer: Option<PathBuf>,
        #[arg(long, default_value = "crates/sky-infer/tests/golden")]
        out: PathBuf,
        #[arg(long, default_value_t = 7)]
        seed: u64,
        /// Ladder config whose shape and options (heads, loops, number fusion,
        /// exits) the golden model uses (default: the s2 golden model).
        #[arg(long)]
        config: Option<PathBuf>,
        /// File prefix (default: `s2`, or the config name up to its first `-`, e.g. `s3`).
        #[arg(long)]
        prefix: Option<String>,
    },
    /// s3: closed-loop play of a GGUF with the mlx-rs forward pass (early exit, temporal carry).
    S2Play {
        gguf: PathBuf,
        #[arg(long, default_value = "flappy,rescue,dodge")]
        games: String,
        #[arg(long, default_value_t = 10)]
        episodes: usize,
        #[arg(long, default_value_t = 7)]
        seed: u64,
        /// Early-exit threshold for every question type (1.0 = full depth).
        #[arg(long, default_value_t = 1.0)]
        tau: f32,
        /// Disable the temporal carry of a temporal_carry model (cold frames only).
        #[arg(long)]
        no_carry: bool,
        #[arg(long, default_value_t = 50)]
        max_pipes: u32,
        /// s2 tokenizer (default data/tok/tok-s2-<1024|2048>.json from the vocabulary size).
        #[arg(long)]
        tokenizer: Option<PathBuf>,
        #[arg(long)]
        json: Option<PathBuf>,
    },
    /// s3: temporal sequences (SPEC-S3 §7): K consecutive decision points per sequence.
    S2Seqgen {
        /// Comma-separated games.
        #[arg(long, default_value = "flappy,rescue,dodge")]
        games: String,
        /// Sequences per game.
        #[arg(long, default_value_t = 20_000)]
        n_seq: usize,
        #[arg(long, default_value_t = 4)]
        k: usize,
        #[arg(long, default_value_t = 0)]
        seed: u64,
        #[arg(long, default_value = "data/seq")]
        out: PathBuf,
    },
    /// s3: the programmatic numeric curriculum (SPEC-S3 §3) as decision records.
    S2Numeric {
        #[arg(long, default_value_t = 100_000)]
        n: usize,
        #[arg(long, default_value = "data/numeric/numeric.jsonl")]
        out: PathBuf,
        #[arg(long, default_value_t = 0)]
        seed: u64,
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
    /// s5: a Hugging Face ModernBERT checkpoint (Ettin, mmBERT; safetensors or
    /// pytorch_model.bin) as a sky-train checkpoint with its own special tokens.
    ImportHf(sky_train::s5::hf::ImportArgs),
    /// s5: soft labels of a Laya-format teacher on s5 records (averaged over option orders).
    Score(sky_train::s5::score::ScoreArgs),
    /// s5: fine-tune an imported encoder into a decision model (by steps; AdamW, warmup +
    /// linear decay, EMA, CE + Brier + RPS).
    S5Train(sky_train::s5::train::TrainArgs),
    /// s5: an iso checkpoint (variant B) as a `skycmd-iso` GGUF (f32, f16, q8_0 or q4_0).
    S5Export(sky_train::s5::export::ExportArgs),
    /// s5: reference outputs of a small random iso model for the sky-infer tests.
    S5Golden(sky_train::s5::golden::GoldenArgs),
}

fn main() -> Result<()> {
    // Optional cap on MLX's allocator cache (freed buffers kept for reuse). Unbounded by
    // default; the s4 night run sets it so that training next to the teacher does not swap.
    if let Some(gb) = std::env::var("SKY_MLX_CACHE_GB")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
    {
        mlx_rs::memory::set_cache_limit((gb * (1u64 << 30) as f64) as usize)?;
    }
    match Cli::parse().cmd {
        Cmd::Bench {
            config,
            mlm_batch,
            mlm_seq,
            dec_batch,
            dec_seq,
            dec_options,
            warmup,
            steps,
            bench_md,
            precision,
        } => {
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
            opt: opt.build(3e-3),
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
            opt: opt.build(1e-3),
            seed,
            log_every,
            eval_every,
            checkpoint_minutes,
            val_max,
            bf16: precision == Precision::Bf16,
        })?,
        Cmd::Calibrate {
            run,
            records,
            split,
            min_bucket,
        } => {
            export::calibrate(&run, &split_paths(&records), &split, min_bucket)?;
        }
        Cmd::Export { run, out } => export::export(&run, &out)?,
        Cmd::S2Bench {
            config,
            mlm_batch,
            mlm_seq,
            dec_batch,
            dec_questions,
            static_len,
            state_len,
            options,
            warmup,
            steps,
            bench_md,
            precision,
            no_compile,
            qat,
            ns_polish,
        } => {
            let rows = s2::train::bench(&S2BenchArgs {
                ns_polish,
                config,
                mlm_batch,
                mlm_seq,
                dec_batch,
                dec_questions,
                static_len,
                state_len,
                options,
                warmup,
                steps,
                bench_md: bench_md.clone(),
                bf16: precision == Precision::Bf16,
                compile: !no_compile,
                qat,
            })?;
            for r in rows {
                println!(
                    "{} {:>9} params {:<12} {} batch {:>4} ({}): {:>7.1} ms/step {:>9.0} tok/s peak {:>6.0} MB",
                    r.model, r.params, r.phase, r.precision, r.batch, r.shape, r.step_ms, r.tok_s, r.peak_mb
                );
            }
            println!("appended to {}", bench_md.display());
        }
        Cmd::S2Pretrain {
            config,
            tokenizer,
            text,
            minutes,
            steps,
            stop_after,
            init_params,
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
            no_compile,
            num_weight,
            numeric,
            numeric_weight,
        } => s2::train::pretrain(&S2PretrainArgs {
            config,
            tokenizer,
            text,
            progress: Progress::from_flags(minutes, steps)?,
            stop_after,
            init_params,
            out,
            resume,
            batch,
            seq,
            opt: opt.build(3e-3),
            seed,
            max_tokens,
            log_every,
            eval_every,
            checkpoint_minutes,
            val_batches,
            bf16: precision == Precision::Bf16,
            compile: !no_compile,
            num_weight,
            numeric,
            numeric_weight,
        })?,
        Cmd::S2Decide {
            config,
            init,
            tokenizer,
            records,
            weights,
            minutes,
            steps,
            stop_after,
            init_params,
            out,
            resume,
            batch,
            opt,
            seed,
            qat_frac,
            loop_dropout,
            dis,
            length_pool,
            pad_to,
            seq_records,
            seq_frac,
            carry_drop,
            log_every,
            eval_every,
            checkpoint_minutes,
            val_max,
            precision,
            no_compile,
        } => s2::train::decide(&S2DecideArgs {
            config,
            init,
            tokenizer,
            records: split_paths(&records),
            weights,
            progress: Progress::from_flags(minutes, steps)?,
            stop_after,
            init_params,
            out,
            resume,
            batch,
            opt: opt.build(1e-3),
            seed,
            qat_frac,
            loop_dropout,
            log_every,
            eval_every,
            checkpoint_minutes,
            val_max,
            bf16: precision == Precision::Bf16,
            compile: !no_compile,
            pad_to,
            seq_records,
            seq_frac,
            carry_drop,
            dis,
            length_pool,
        })?,
        Cmd::S2Calibrate {
            run,
            records,
            split,
            min_bucket,
        } => {
            s2::eval::calibrate(&run, &split_paths(&records), &split, min_bucket)?;
        }
        Cmd::S2Export { run, out, format } => {
            s2::eval::export(&run, &out, WeightFormat::parse(&format)?)?
        }
        Cmd::S2Eval {
            run,
            records,
            split,
            gguf,
            report,
            gguf_only,
        } => {
            let ggufs = gguf.as_deref().map(split_paths).unwrap_or_default();
            if gguf_only && ggufs.is_empty() {
                anyhow::bail!("--gguf-only needs --gguf");
            }
            s2::eval::eval_with(
                gguf_only,
                &run,
                &split_paths(&records),
                &split,
                &ggufs,
                report.as_deref(),
            )?;
        }
        Cmd::S2Golden {
            tokenizer,
            out,
            seed,
            config,
            prefix,
        } => {
            let tok = match tokenizer {
                Some(p) => std::fs::read_to_string(p)?,
                None => sky_train::s2::golden::byte_tokenizer_json(),
            };
            let base = config.map(s2::config::S2Config::load).transpose()?;
            let prefix = prefix.unwrap_or_else(|| match &base {
                Some(c) => {
                    let p = c.name.split('-').next().unwrap_or("s2");
                    if c.temporal_carry {
                        format!("{p}t")
                    } else {
                        p.to_string()
                    }
                }
                None => "s2".into(),
            });
            sky_train::s2::golden::golden_with(&tok, &out, seed, base.as_ref(), &prefix)?;
            println!(
                "wrote {0}/{prefix}.json, {0}/{prefix}-f32.gguf, {0}/{prefix}-ternary.gguf",
                out.display()
            );
        }
        Cmd::S2Play {
            gguf,
            games,
            episodes,
            seed,
            tau,
            no_carry,
            max_pipes,
            tokenizer,
            json,
        } => {
            let g: Vec<&str> = games.split(',').collect();
            let r = s2::play::play(
                &gguf,
                &g,
                episodes,
                seed,
                tau,
                !no_carry,
                max_pipes,
                tokenizer.as_deref(),
            )?;
            if let Some(path) = json {
                std::fs::write(path, serde_json::to_string_pretty(&r)?)?;
            }
        }
        Cmd::S2Seqgen {
            games,
            n_seq,
            k,
            seed,
            out,
        } => {
            for g in games.split(',') {
                let path = out.join(format!("{g}.jsonl"));
                let eps = s2::seq::generate(g, n_seq, k, seed, &path)?;
                println!(
                    "{g}: {n_seq} sequences of {k} frames from {eps} episodes -> {}",
                    path.display()
                );
            }
        }
        Cmd::S2Numeric { n, out, seed } => {
            s2::numeric::write(n, seed, &out)?;
            println!("wrote {n} numeric records to {}", out.display());
        }
        Cmd::Eval {
            run,
            records,
            split,
            report,
        } => {
            export::eval(&run, &split_paths(&records), &split, report.as_deref())?;
        }
        Cmd::ImportHf(a) => sky_train::s5::hf::import(&a)?,
        Cmd::Score(a) => sky_train::s5::score::score(&a)?,
        Cmd::S5Train(a) => sky_train::s5::train::train(&a)?,
        Cmd::S5Export(a) => sky_train::s5::export::export(&a)?,
        Cmd::S5Golden(a) => sky_train::s5::golden::golden(&a)?,
    }
    Ok(())
}
