//! `sky-data` CLI.

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{bail, Result};
use clap::{Parser, Subcommand};

use sky_data::decisions::{fetch_decisions, DecisionOptions};
use sky_data::hf::Hf;
use sky_data::teacher::{self, Teacher, TeacherArgs, TeacherConfig};
use sky_data::text::{fetch_text, TextOptions};
use sky_data::tok::{build_corpus, int_prior_reps, int_prior_segments, sequence_stats, train_tokenizer, CorpusOptions};

#[derive(Parser)]
#[command(name = "sky-data", version, about = "skycmd data pipeline")]
struct Cli {
    /// Download cache directory.
    #[arg(long, global = true, default_value = "data/hf")]
    cache: PathBuf,
    /// Seed for every sampling / shuffling step.
    #[arg(long, global = true, default_value_t = 1234)]
    seed: u64,
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Download decision datasets and write one SPEC decision JSONL per source.
    FetchDecisions {
        #[arg(long, default_value = "data/decisions")]
        out: PathBuf,
        /// Maximum records per source.
        #[arg(long, default_value_t = 60_000)]
        cap: usize,
        /// Validation fraction (by state group) for non-test sources.
        #[arg(long, default_value_t = 0.05)]
        val_frac: f64,
        /// Only these sources (comma separated).
        #[arg(long, value_delimiter = ',')]
        only: Vec<String>,
    },
    /// Download MLM text, size-capped, exact + MinHash near-dup deduplicated.
    FetchText {
        #[arg(long, default_value = "data/text")]
        out: PathBuf,
        /// Total size cap in MB of text.
        #[arg(long, default_value_t = 150.0)]
        max_mb: f64,
        #[arg(long, default_value_t = 64)]
        min_chars: usize,
        #[arg(long, default_value_t = 20_000)]
        max_chars: usize,
        #[arg(long, value_delimiter = ',')]
        only: Vec<String>,
    },
    /// Train a byte-level BPE tokenizer with fixed special-token ids.
    TrainTokenizer {
        #[arg(long)]
        vocab: usize,
        /// Output path (default data/tok/tok-<vocab>.json).
        #[arg(long)]
        out: Option<PathBuf>,
        #[arg(long, value_delimiter = ',', default_value = "data/text,data/decisions,data/games")]
        inputs: Vec<PathBuf>,
        /// Repetitions of game records in the corpus.
        #[arg(long, default_value_t = 4)]
        game_weight: usize,
        /// Cap on MLM text bytes used for training.
        #[arg(long, default_value_t = 120.0)]
        max_text_mb: f64,
        #[arg(long, default_value_t = 30_000)]
        max_decisions_per_file: usize,
        #[arg(long, default_value_t = 2)]
        min_frequency: u64,
        /// Make the integers 0..N single tokens (default vocab/32; 0 disables).
        #[arg(long)]
        int_prior: Option<usize>,
    },
    /// GLM teacher tasks (z.ai, Anthropic Messages protocol).
    Teacher {
        /// Model id; "auto" picks from the endpoint's model list. Default: SKY_TEACHER_MODEL or ccs.
        #[arg(long, global = true)]
        model: Option<String>,
        /// Hard cap on HTTP calls.
        #[arg(long, global = true, default_value_t = 300)]
        max_calls: usize,
        /// Items per call.
        #[arg(long, global = true, default_value_t = 40)]
        batch: usize,
        /// Minimum delay between calls (ms).
        #[arg(long, global = true, default_value_t = 1500)]
        min_interval_ms: u64,
        #[command(subcommand)]
        task: TeacherTask,
    },
    /// Write data/STATS.md (records per source/split, label entropy, lengths).
    Stats {
        #[arg(long, default_value = "data")]
        data: PathBuf,
        /// Tokenizer directory for sequence-length tables.
        #[arg(long, default_value = "data/tok")]
        tok: PathBuf,
        #[arg(long, default_value = "data/STATS.md")]
        out: PathBuf,
    },
}

#[derive(Subcommand)]
enum TeacherTask {
    /// Export unique questions from decision records into a questions.json for paraphrasing.
    ExportQuestions {
        #[arg(long, value_delimiter = ',', default_value = "data/decisions,data/games")]
        from: Vec<PathBuf>,
        #[arg(long, default_value = "data/teacher/questions.json")]
        out: PathBuf,
        #[arg(long, default_value_t = 2000)]
        max: usize,
    },
    /// EN + FR paraphrases of question instructions.
    ParaphraseQuestions {
        #[arg(long = "in")]
        input: PathBuf,
        #[arg(long, default_value = "data/teacher/paraphrases.jsonl")]
        out: PathBuf,
        /// Paraphrases per language.
        #[arg(long, default_value_t = 3)]
        n: usize,
    },
    /// Civil drone orders (EN/FR) with validated JSON plans, plus `action` decision records.
    DroneOrders {
        #[arg(long, default_value = "data/teacher/orders.jsonl")]
        out: PathBuf,
        #[arg(long, default_value = "data/teacher/orders_decisions.jsonl")]
        decisions_out: PathBuf,
        #[arg(long, default_value_t = 2000)]
        n: usize,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let t0 = Instant::now();
    match cli.cmd {
        Cmd::FetchDecisions { out, cap, val_frac, only } => {
            let hf = Hf::new(&cli.cache)?;
            let reports = fetch_decisions(&hf, &DecisionOptions { out, seed: cli.seed, cap, val_frac, only })?;
            println!("{:<24} {:>8} {:>8} {:>6} {:>8} {:>6} {:>6} {:>7}", "source", "raw", "invalid", "dup", "train", "val", "test", "secs");
            for r in reports {
                println!(
                    "{:<24} {:>8} {:>8} {:>6} {:>8} {:>6} {:>6} {:>7.1}",
                    r.src, r.raw, r.invalid, r.duplicates, r.train, r.val, r.test, r.secs
                );
            }
        }
        Cmd::FetchText { out, max_mb, min_chars, max_chars, only } => {
            let hf = Hf::new(&cli.cache)?;
            let reports = fetch_text(&hf, &TextOptions { out, max_mb, seed: cli.seed, min_chars, max_chars, only })?;
            println!("{:<16} {:>8} {:>8} {:>8} {:>8} {:>8} {:>10} {:>7}", "source", "docs", "MB", "budget", "exact", "near", "dl MB", "secs");
            for r in reports {
                println!(
                    "{:<16} {:>8} {:>8.1} {:>8.1} {:>8} {:>8} {:>10.1} {:>7.1}",
                    r.src,
                    r.docs,
                    r.bytes as f64 / 1e6,
                    r.budget as f64 / 1e6,
                    r.exact_dups,
                    r.near_dups,
                    r.downloaded as f64 / 1e6,
                    r.secs
                );
            }
        }
        Cmd::TrainTokenizer { vocab, out, inputs, game_weight, max_text_mb, max_decisions_per_file, min_frequency, int_prior } => {
            if vocab < 300 {
                bail!("vocab must be at least 300 (256 bytes + specials + merges)");
            }
            let out = out.unwrap_or_else(|| PathBuf::from(format!("data/tok/tok-{vocab}.json")));
            let mut corpus = build_corpus(&CorpusOptions {
                inputs: inputs.clone(),
                game_weight,
                max_text_mb,
                max_decisions_per_file,
                seed: cli.seed,
            })?;
            eprintln!(
                "[tok] corpus: {} segments, {:.1} MB text, {} decision records, {} game records",
                corpus.segments.len(),
                corpus.text_bytes as f64 / 1e6,
                corpus.decision_records,
                corpus.game_records
            );
            let int_max = int_prior.unwrap_or(vocab / 32);
            if int_max > 0 {
                let bytes: u64 = corpus.segments.iter().map(|s| s.len() as u64).sum();
                let reps = int_prior_reps(bytes);
                corpus.segments.extend(int_prior_segments(int_max, reps));
                eprintln!("[tok] integer prior 0..{int_max} x{reps}");
            }
            let tok = train_tokenizer(&corpus.segments, vocab, min_frequency)?;
            if let Some(d) = out.parent() {
                std::fs::create_dir_all(d)?;
            }
            tok.save(&out, false).map_err(|e| anyhow::anyhow!(e))?;
            println!("saved {out:?} (vocab {}) in {:.1}s", tok.get_vocab_size(true), t0.elapsed().as_secs_f64());
            println!("{:<24} {:>8} {:>12} {:>14} {:>10} {:>10}", "source", "sampled", "mean tokens", "untruncated", "truncated", "opts cut");
            for dir in inputs.iter().filter(|d| d.is_dir() && !d.ends_with("text")) {
                for (src, s) in sequence_stats(&tok, dir, 2000, cli.seed)? {
                    println!(
                        "{:<24} {:>8} {:>12.1} {:>14.1} {:>9.1}% {:>9.1}%",
                        src,
                        s.n,
                        s.mean_tokens,
                        s.mean_untruncated,
                        100.0 * s.truncated_frac,
                        100.0 * s.markers_lost_frac
                    );
                }
            }
        }
        Cmd::Teacher { model, max_calls, batch, min_interval_ms, task } => {
            if let TeacherTask::ExportQuestions { from, out, max } = &task {
                let qs = teacher::export_questions(from, *max, cli.seed)?;
                if let Some(d) = out.parent() {
                    std::fs::create_dir_all(d)?;
                }
                std::fs::write(out, serde_json::to_vec_pretty(&qs)?)?;
                println!("wrote {} questions to {out:?}", qs.len());
                return Ok(());
            }
            let cfg = TeacherConfig::resolve(&TeacherArgs { model, max_calls, batch, min_interval_ms })?;
            eprintln!("[teacher] endpoint {} model {} max_calls {} batch {}", cfg.base_url, cfg.model, cfg.max_calls, cfg.batch);
            let mut t = Teacher::new(cfg);
            match task {
                TeacherTask::ParaphraseQuestions { input, out, n } => {
                    let w = teacher::paraphrase_questions(&mut t, &input, &out, n)?;
                    println!("wrote {w} paraphrase items to {out:?}");
                }
                TeacherTask::DroneOrders { out, decisions_out, n } => {
                    let (total, dropped) = teacher::drone_orders(&mut t, &out, &decisions_out, n, cli.seed)?;
                    println!("{total} orders in {out:?} ({dropped} invalid dropped this run); decision records in {decisions_out:?}");
                }
                TeacherTask::ExportQuestions { .. } => unreachable!(),
            }
            println!(
                "calls {} / {}, tokens in {} out {}, {:.1}s",
                t.calls,
                t.cfg.max_calls,
                t.input_tokens,
                t.output_tokens,
                t0.elapsed().as_secs_f64()
            );
        }
        Cmd::Stats { data, tok, out } => {
            let md = sky_data::stats::build_report(&data, tok.exists().then_some(tok.as_path()))?;
            std::fs::write(&out, &md)?;
            println!("{md}");
            println!("wrote {out:?}");
        }
    }
    Ok(())
}
