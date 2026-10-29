//! `sky-data` CLI.

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{bail, Result};
use clap::{Parser, Subcommand};

use sky_data::augment::{augment_paraphrases, AugmentOptions};
use sky_data::commands::{fetch_commands, CommandOptions};
use sky_data::decisions::{fetch_decisions, sources, DecisionOptions};
use sky_data::hf::{Hf, RepoKind};
use sky_data::teacher::{self, Teacher, TeacherArgs, TeacherConfig};
use sky_data::text::{fetch_text, TextOptions, TextSet};
use sky_data::tok::{
    build_corpus, int_prior_reps, int_prior_segments, num_segments, sequence_stats,
    state_token_stats, train_tokenizer, train_tokenizer_s2, CorpusOptions,
};

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
        /// Maximum records per source (sources with their own cap, e.g. tasksource_jev, keep it).
        #[arg(long, default_value_t = 60_000)]
        cap: usize,
        /// Validation fraction (by state group) for non-test sources.
        #[arg(long, default_value_t = 0.05)]
        val_frac: f64,
        /// Only these sources (comma separated; see --list).
        #[arg(long, value_delimiter = ',')]
        only: Vec<String>,
        /// GLM teacher output directory (for glm_orders).
        #[arg(long, default_value = "data/teacher")]
        teacher: PathBuf,
        /// List the source names and exit.
        #[arg(long)]
        list: bool,
    },
    /// Download a Hub model or dataset repository (or the files matching --include), pinned to
    /// the commit of --rev and checked against the published sha256 / blob sha1. Sends HF_TOKEN
    /// when it is set. Writes <out>/.skycmd-fetch.json.
    FetchModel {
        /// Repository id, e.g. jhu-clsp/ettin-encoder-17m.
        repo: String,
        #[arg(long)]
        out: PathBuf,
        /// Branch, tag or commit.
        #[arg(long, default_value = "main")]
        rev: String,
        /// The repository is a dataset.
        #[arg(long, conflicts_with = "github")]
        dataset: bool,
        /// The repository is on GitHub (files checked against their blob sha1).
        #[arg(long)]
        github: bool,
        /// Only files matching these globs (`*` wildcard; comma separated).
        #[arg(long, value_delimiter = ',')]
        include: Vec<String>,
        /// Refuse selections larger than this.
        #[arg(long, default_value_t = 4.0)]
        max_gb: f64,
    },
    /// s5: evaluation sets, training corpus, generators, coherence families, pairs,
    /// decontamination and the class-balance gate; writes <out>/MANIFEST.json.
    S5Build {
        #[arg(long, default_value = "data/decisions")]
        decisions: PathBuf,
        #[arg(long, default_value = "data/text")]
        text: PathBuf,
        #[arg(long, default_value = "data/s5/src")]
        src: PathBuf,
        #[arg(long)]
        defs: PathBuf,
        #[arg(long)]
        demo: PathBuf,
        #[arg(long, default_value = "data/s5")]
        out: PathBuf,
        /// Languages (en, fr).
        #[arg(long, value_delimiter = ',', default_value = "en")]
        langs: Vec<String>,
        #[arg(long, default_value_t = 1000)]
        per_source: usize,
        #[arg(long, default_value_t = 1000)]
        per_eval_set: usize,
        #[arg(long, default_value_t = 6000)]
        rules: usize,
        #[arg(long, default_value_t = 3000)]
        tone: usize,
        #[arg(long, default_value_t = 4000)]
        buried: usize,
        /// Paraphrase pairs and distractor pairs, each.
        #[arg(long, default_value_t = 2000)]
        pairs: usize,
    },
    /// s5: training targets from gold and the teachers' soft labels (calibrated per group,
    /// teacher chosen per group on val, agreement filter); writes <out>/REPORT.json.
    S5Mix {
        #[arg(long, default_value = "data/s5/train")]
        train: PathBuf,
        #[arg(long, default_value = "data/s5/scored")]
        scored: PathBuf,
        #[arg(long, value_delimiter = ',', default_value = "laya-td")]
        teachers: Vec<String>,
        #[arg(long, default_value = "data/s5/mixed")]
        out: PathBuf,
        #[arg(long, default_value_t = 0.5)]
        alpha: f32,
    },
    /// s5: the original coherence battery of 2609.33209 as s5 records (evaluation only, never
    /// redistributed), for the comparison with Jev on the same items.
    S5Battery {
        #[arg(
            long,
            default_value = "data/s5/src/coherence/02_probability_axioms/stimuli/items.jsonl"
        )]
        items: PathBuf,
        #[arg(long, default_value = "data/s5/battery/coherence_2609_33209.jsonl")]
        out: PathBuf,
    },
    /// Print the first rows of a parquet file as JSON (all columns unless --columns).
    Peek {
        file: PathBuf,
        #[arg(long, default_value_t = 3)]
        n: usize,
        #[arg(long, value_delimiter = ',')]
        columns: Vec<String>,
    },
    /// The rule scanner (gitleaks-style prefixes + an entropy rule) on the API-key jev: the
    /// score a trained model has to beat.
    SecretBaseline {
        #[arg(long, default_value = "data/secrets/secret_bench.jsonl")]
        data: PathBuf,
    },
    /// The API-key jev of s6-secrets (local generator, no download): secret_synth (train / val)
    /// and secret_bench (test, contexts never trained on).
    SecretJev {
        #[arg(long, default_value = "data/secrets")]
        out: PathBuf,
        /// Validation fraction (by text).
        #[arg(long, default_value_t = 0.05)]
        val_frac: f64,
    },
    /// Download NL → command data (function calling + MASSIVE intents) into JSONL.
    FetchCommands {
        #[arg(long, default_value = "data/commands")]
        out: PathBuf,
        #[arg(long, default_value_t = 0.05)]
        val_frac: f64,
        /// Cap on glaive conversations that offer no function.
        #[arg(long, default_value_t = 10_000)]
        max_no_tools: usize,
        /// Only these sources (hermes, glaive, massive).
        #[arg(long, value_delimiter = ',')]
        only: Vec<String>,
    },
    /// Decision records whose instructions are replaced by GLM paraphrases (targets unchanged).
    AugmentParaphrases {
        #[arg(long, default_value = "data/teacher/paraphrases.jsonl")]
        paraphrases: PathBuf,
        #[arg(
            long,
            value_delimiter = ',',
            default_value = "data/decisions,data/games"
        )]
        inputs: Vec<PathBuf>,
        #[arg(long, default_value = "data/decisions/paraphrased.jsonl")]
        out: PathBuf,
        #[arg(long, default_value_t = 40_000)]
        cap: usize,
        /// Candidate records sampled per instruction before the final draw.
        #[arg(long, default_value_t = 8192)]
        per_question: usize,
    },
    /// s6-cast: real time series with a verified permissive upstream license and no overlap with
    /// the zero-shot test sets (data/series/<source>.jsonl).
    FetchSeries {
        #[arg(long, default_value = "data/series")]
        out: PathBuf,
        #[arg(long, value_delimiter = ',')]
        only: Vec<String>,
    },
    /// Download MLM text, size-capped, exact + MinHash near-dup deduplicated.
    FetchText {
        /// Output directory (default data/text, or data/text-open with --set open).
        #[arg(long)]
        out: Option<PathBuf>,
        /// Source set: `default` (stories and web) or `open` (Dolma 3 and TxT360, opt-in).
        #[arg(long, value_enum, default_value_t)]
        set: TextSet,
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
        /// s2 tokenizer (SPEC-S2 §2): numbers are replaced by the special token [NUM]=5 before
        /// BPE, and no integer prior is added.
        #[arg(long)]
        num: bool,
        /// Output path (default data/tok/tok-<vocab>.json, or data/tok/tok-s2-<vocab>.json with --num).
        #[arg(long)]
        out: Option<PathBuf>,
        #[arg(
            long,
            value_delimiter = ',',
            default_value = "data/text,data/decisions,data/games"
        )]
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
    /// GLM teacher tasks (Anthropic Messages protocol).
    Teacher {
        /// Model id; "auto" picks from the endpoint's model list. Default: SKY_TEACHER_MODEL, else "auto".
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
    /// s4: the records the Qwen3.5 teacher scores (data/s4/queue), the gold sets and the
    /// zero-shot held-out sets (data/s4/gold, data/s4/heldout).
    S4Corpus {
        #[arg(long, default_value = "data")]
        data: PathBuf,
        #[arg(long, default_value = "data/s4")]
        out: PathBuf,
        /// Records sampled per typed-decision source.
        #[arg(long, default_value_t = 4000)]
        per_source: usize,
        /// Generic snippet records.
        #[arg(long, default_value_t = 60000)]
        generic: usize,
        /// Tool-routing, triage and moderation records each.
        #[arg(long, default_value_t = 15000)]
        per_case: usize,
        /// Records per held-out set.
        #[arg(long, default_value_t = 2000)]
        heldout: usize,
        /// Cap on the train/val records of each gold file (0: symlinks to data/decisions).
        #[arg(long, default_value_t = 0)]
        gold_cap: usize,
        /// Only (re)write data/s4/gold.
        #[arg(long, conflicts_with = "only_heldout")]
        only_gold: bool,
        /// Only (re)write data/s4/heldout (MASSIVE en/fr and Banking77, split test).
        #[arg(long)]
        only_heldout: bool,
    },
    /// s4: gold / teacher mix of the scored records (teacher temperature fitted on gold).
    S4Mix {
        #[arg(long, value_delimiter = ',')]
        scored: Vec<PathBuf>,
        #[arg(long, default_value = "data/s4/mix")]
        out: PathBuf,
        /// Weight of the gold label when there is one.
        #[arg(long, default_value_t = 0.5)]
        alpha: f32,
        #[arg(long)]
        temperature: Option<f32>,
        /// s4-corpus queue: unscored gold records pass through with their gold target.
        #[arg(long)]
        queue: Option<PathBuf>,
    },
}

#[derive(Subcommand)]
enum TeacherTask {
    /// Export unique questions from decision records into a questions.json for paraphrasing.
    ExportQuestions {
        #[arg(
            long,
            value_delimiter = ',',
            default_value = "data/decisions,data/games"
        )]
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
    /// s6: teacher probabilities per option for the train decision records (resumable).
    LabelDecisions {
        #[arg(long, value_delimiter = ',', default_value = "data/decisions")]
        from: Vec<PathBuf>,
        #[arg(long, default_value = "data/teacher/glm_labels.jsonl")]
        out: PathBuf,
        /// Records drawn per source family (0: every eligible record).
        #[arg(long, default_value_t = 0)]
        per_source: usize,
        /// The state is cut to this many characters in the prompt.
        #[arg(long, default_value_t = 2000)]
        max_state_chars: usize,
        /// Parallel workers (the call budget is shared between them).
        #[arg(long, default_value_t = 4)]
        jobs: usize,
        /// Sampling temperature of the teacher.
        #[arg(long, default_value_t = 0.2)]
        temperature: f64,
    },
    /// s6: gold / teacher mix of the labels that agree with their gold answer (no API call).
    MixLabels {
        #[arg(long, default_value = "data/teacher/glm_labels.jsonl")]
        labels: PathBuf,
        #[arg(long, value_delimiter = ',', default_value = "data/decisions")]
        from: Vec<PathBuf>,
        #[arg(long, default_value = "data/distill")]
        out: PathBuf,
        /// Weight of the gold target.
        #[arg(long, default_value_t = 0.5)]
        alpha: f32,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let t0 = Instant::now();
    match cli.cmd {
        Cmd::FetchDecisions {
            out,
            cap,
            val_frac,
            only,
            teacher,
            list,
        } => {
            if list {
                for s in sources() {
                    let cap = s.cap.map_or_else(
                        || format!("{cap} (default)"),
                        |c| {
                            if c == usize::MAX {
                                "none".into()
                            } else {
                                c.to_string()
                            }
                        },
                    );
                    println!(
                        "{:<24} cap {cap}{}",
                        s.name,
                        if s.test_only { ", test only" } else { "" }
                    );
                }
                return Ok(());
            }
            let hf = Hf::new(&cli.cache)?;
            let reports = fetch_decisions(
                &hf,
                &DecisionOptions {
                    out,
                    seed: cli.seed,
                    cap,
                    val_frac,
                    only,
                    teacher,
                },
            )?;
            println!(
                "{:<24} {:>8} {:>8} {:>6} {:>8} {:>6} {:>6} {:>7}",
                "source", "raw", "invalid", "dup", "train", "val", "test", "secs"
            );
            for r in reports {
                println!(
                    "{:<24} {:>8} {:>8} {:>6} {:>8} {:>6} {:>6} {:>7.1}",
                    r.src, r.raw, r.invalid, r.duplicates, r.train, r.val, r.test, r.secs
                );
            }
        }
        Cmd::FetchModel {
            repo,
            out,
            rev,
            dataset,
            github,
            include,
            max_gb,
        } => {
            let hf = Hf::new(&cli.cache)?;
            let r = sky_data::fetch::fetch_repo(
                &hf,
                &sky_data::fetch::FetchOptions {
                    kind: match (dataset, github) {
                        (true, _) => RepoKind::Dataset,
                        (_, true) => RepoKind::GitHub,
                        _ => RepoKind::Model,
                    },
                    repo: repo.clone(),
                    rev,
                    out: out.clone(),
                    include,
                    max_bytes: (max_gb * 1e9) as u64,
                },
            )?;
            println!(
                "{repo}@{}: {} files ({:.1} MB), {} downloaded, all verified, in {out:?}",
                r.commit,
                r.files,
                r.bytes as f64 / 1e6,
                r.downloaded
            );
        }
        Cmd::S5Build {
            decisions,
            text,
            src,
            defs,
            demo,
            out,
            langs,
            per_source,
            per_eval_set,
            rules,
            tone,
            buried,
            pairs,
        } => {
            let m = sky_data::s5::build::build(&sky_data::s5::build::BuildOptions {
                decisions,
                text,
                src,
                defs,
                demo,
                out: out.clone(),
                seed: cli.seed,
                langs,
                per_source,
                per_eval_set,
                n_rules: rules,
                n_tone: tone,
                n_buried: buried,
                n_pairs: pairs,
            })?;
            println!("{}", serde_json::to_string_pretty(&m["train"])?);
            println!("{}", serde_json::to_string_pretty(&m["decontamination"])?);
            println!("wrote {}", out.join("MANIFEST.json").display());
        }
        Cmd::S5Mix {
            train,
            scored,
            teachers,
            out,
            alpha,
        } => {
            let r = sky_data::s5::mix::mix(&sky_data::s5::mix::MixOptions {
                train,
                scored,
                teachers,
                out: out.clone(),
                alpha,
            })?;
            println!("{}", serde_json::to_string_pretty(&r["targets"])?);
            println!("wrote {}", out.join("REPORT.json").display());
        }
        Cmd::S5Battery { items, out } => {
            let n = sky_data::s5::battery::convert(&items, &out)?;
            println!(
                "wrote {n} records to {} (sha256 {})",
                out.display(),
                sky_data::s5::sha256_file(&out)?
            );
        }
        Cmd::Peek { file, n, columns } => {
            let cols: Vec<&str> = columns.iter().map(String::as_str).collect();
            for row in sky_data::hf::read_parquet_any(&file, &cols, n)? {
                println!("{row}");
            }
        }
        Cmd::SecretBaseline { data } => {
            let recs: Vec<sky_data::record::DecisionRecord> = sky_data::record::read_jsonl(&data)?;
            println!("{}", sky_data::secrets::baseline_report(&recs));
        }
        Cmd::SecretJev { out, val_frac } => {
            for r in sky_data::secrets::write_jev(&out, cli.seed, val_frac)? {
                println!(
                    "{}: raw {} dup {} -> train {} val {} test {}",
                    r.src, r.raw, r.duplicates, r.train, r.val, r.test
                );
            }
        }
        Cmd::FetchCommands {
            out,
            val_frac,
            max_no_tools,
            only,
        } => {
            let hf = Hf::new(&cli.cache)?;
            let reports = fetch_commands(
                &hf,
                &CommandOptions {
                    out,
                    seed: cli.seed,
                    val_frac,
                    max_no_tools,
                    only,
                },
            )?;
            println!(
                "{:<10} {:>8} {:>8} {:>8} {:>8} {:>10} {:>7}",
                "source", "raw", "skipped", "dup", "written", "with calls", "secs"
            );
            for r in reports {
                println!(
                    "{:<10} {:>8} {:>8} {:>8} {:>8} {:>10} {:>7.1}",
                    r.src, r.raw, r.skipped, r.duplicates, r.written, r.with_calls, r.secs
                );
            }
        }
        Cmd::AugmentParaphrases {
            paraphrases,
            inputs,
            out,
            cap,
            per_question,
        } => {
            let rep = augment_paraphrases(&AugmentOptions {
                paraphrases,
                inputs,
                out: out.clone(),
                cap,
                per_question,
                seed: cli.seed,
            })?;
            println!(
                "{} paraphrased instructions, {} matched in the data ({} records); wrote {} records to {out:?}",
                rep.paraphrased_instructions, rep.matched_instructions, rep.matched_records, rep.written
            );
            for (src, n) in rep.by_src {
                println!("  {src:<32} {n}");
            }
        }
        Cmd::FetchSeries { out, only } => {
            let hf = Hf::new(&cli.cache)?;
            let reports = sky_data::series::fetch_series(&hf, &out, &only)?;
            println!("{:<22} {:>8} {:>12}  license", "source", "series", "points");
            for r in &reports {
                println!(
                    "{:<22} {:>8} {:>12}  {}",
                    r.name, r.series, r.points, r.license
                );
            }
        }
        Cmd::FetchText {
            out,
            set,
            max_mb,
            min_chars,
            max_chars,
            only,
        } => {
            let hf = Hf::new(&cli.cache)?;
            let out = out.unwrap_or_else(|| match set {
                TextSet::Default => "data/text".into(),
                TextSet::Open => "data/text-open".into(),
            });
            let reports = fetch_text(
                &hf,
                &TextOptions {
                    out,
                    max_mb,
                    seed: cli.seed,
                    min_chars,
                    max_chars,
                    only,
                    set,
                },
            )?;
            println!(
                "{:<16} {:>8} {:>8} {:>8} {:>8} {:>8} {:>10} {:>7}",
                "source", "docs", "MB", "budget", "exact", "near", "dl MB", "secs"
            );
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
        Cmd::TrainTokenizer {
            vocab,
            num,
            out,
            inputs,
            game_weight,
            max_text_mb,
            max_decisions_per_file,
            min_frequency,
            int_prior,
        } => {
            if vocab < 300 {
                bail!("vocab must be at least 300 (256 bytes + specials + merges)");
            }
            let family = if num { "s2-" } else { "" };
            let out =
                out.unwrap_or_else(|| PathBuf::from(format!("data/tok/tok-{family}{vocab}.json")));
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
            let tok = if num {
                let segments = num_segments(&corpus.segments);
                eprintln!(
                    "[tok] numeric extraction: {} segments after splitting at [NUM]",
                    segments.len()
                );
                train_tokenizer_s2(&segments, vocab, min_frequency)?
            } else {
                let int_max = int_prior.unwrap_or(vocab / 32);
                if int_max > 0 {
                    let bytes: u64 = corpus.segments.iter().map(|s| s.len() as u64).sum();
                    let reps = int_prior_reps(bytes);
                    corpus.segments.extend(int_prior_segments(int_max, reps));
                    eprintln!("[tok] integer prior 0..{int_max} x{reps}");
                }
                train_tokenizer(&corpus.segments, vocab, min_frequency)?
            };
            if let Some(d) = out.parent() {
                std::fs::create_dir_all(d)?;
            }
            tok.save(&out, false).map_err(|e| anyhow::anyhow!(e))?;
            println!(
                "saved {out:?} (vocab {}) in {:.1}s",
                tok.get_vocab_size(true),
                t0.elapsed().as_secs_f64()
            );
            println!(
                "{:<24} {:>8} {:>12} {:>14} {:>10} {:>10}",
                "source", "sampled", "mean tokens", "untruncated", "truncated", "opts cut"
            );
            // MLM text (data/text, data/text-open) has no decision records to measure.
            let is_text = |d: &PathBuf| {
                d.file_name()
                    .is_some_and(|n| n.to_string_lossy().starts_with("text"))
            };
            for dir in inputs.iter().filter(|d| d.is_dir() && !is_text(d)) {
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
            for dir in inputs.iter().filter(|d| d.is_dir() && d.ends_with("games")) {
                for (game, mean) in state_token_stats(&tok, dir, 2000, cli.seed)? {
                    println!("state tokens [CLS] state [SEP] {game:<10} {mean:.1}");
                }
            }
        }
        Cmd::Teacher {
            model,
            max_calls,
            batch,
            min_interval_ms,
            task,
        } => {
            if let TeacherTask::ExportQuestions { from, out, max } = &task {
                let qs = teacher::export_questions(from, *max, cli.seed)?;
                if let Some(d) = out.parent() {
                    std::fs::create_dir_all(d)?;
                }
                std::fs::write(out, serde_json::to_vec_pretty(&qs)?)?;
                println!("wrote {} questions to {out:?}", qs.len());
                return Ok(());
            }
            if let TeacherTask::MixLabels {
                labels,
                from,
                out,
                alpha,
            } = &task
            {
                let a = sky_data::distill::mix_labels(labels, from, out, *alpha)?;
                let (mut n, mut k) = (0, 0);
                for (src, (l, g)) in &a {
                    println!(
                        "{src:<40} {l:>6} labelled {g:>6} agree ({:.3})",
                        *g as f64 / (*l).max(1) as f64
                    );
                    (n, k) = (n + l, k + g);
                }
                println!(
                    "all: {k} / {n} agree; mixed records in {:?}",
                    out.join("glm.jsonl")
                );
                return Ok(());
            }
            let cfg = TeacherConfig::resolve(&TeacherArgs {
                model,
                max_calls,
                batch,
                min_interval_ms,
            })?;
            eprintln!(
                "[teacher] endpoint {} model {} max_calls {} batch {}",
                cfg.base_url, cfg.model, cfg.max_calls, cfg.batch
            );
            let mut t = Teacher::new(cfg);
            match task {
                TeacherTask::ParaphraseQuestions { input, out, n } => {
                    let w = teacher::paraphrase_questions(&mut t, &input, &out, n)?;
                    println!("wrote {w} paraphrase items to {out:?}");
                }
                TeacherTask::DroneOrders {
                    out,
                    decisions_out,
                    n,
                } => {
                    let (total, dropped) =
                        teacher::drone_orders(&mut t, &out, &decisions_out, n, cli.seed)?;
                    println!("{total} orders in {out:?} ({dropped} invalid dropped this run); decision records in {decisions_out:?}");
                }
                TeacherTask::LabelDecisions {
                    from,
                    out,
                    per_source,
                    max_state_chars,
                    jobs,
                    temperature,
                } => {
                    t.cfg.temperature = temperature;
                    t.cfg.max_tokens = 8192;
                    let o = sky_data::distill::LabelOptions {
                        from,
                        per_source,
                        max_state_chars,
                        jobs,
                        seed: cli.seed,
                    };
                    let s = sky_data::distill::label_decisions(&t.cfg, &o, &out)?;
                    println!(
                        "wrote {} labels to {out:?}; calls {} ({} failed batches), tokens in {} out {}, {:.1}s",
                        s.written,
                        s.calls,
                        s.failed_batches,
                        s.input_tokens,
                        s.output_tokens,
                        t0.elapsed().as_secs_f64()
                    );
                    return Ok(());
                }
                TeacherTask::ExportQuestions { .. } | TeacherTask::MixLabels { .. } => {
                    unreachable!()
                }
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
        Cmd::S4Corpus {
            data,
            out,
            per_source,
            generic,
            per_case,
            heldout,
            gold_cap,
            only_gold,
            only_heldout,
        } => {
            let o = sky_data::s4::CorpusOptions {
                data,
                out,
                seed: cli.seed,
                per_source,
                generic,
                per_case,
                heldout,
                gold_cap,
            };
            let counts = if only_gold {
                sky_data::s4::write_gold(&o)?
            } else if only_heldout {
                sky_data::s4::heldout(&o)?
            } else {
                sky_data::s4::build_corpus(&Hf::new(&cli.cache)?, &o)?
            };
            for (k, n) in counts {
                println!("{k:<16} {n}");
            }
        }
        Cmd::S4Mix {
            scored,
            out,
            alpha,
            temperature,
            queue,
        } => {
            let report = sky_data::s4::mix(&sky_data::s4::MixOptions {
                scored,
                out,
                alpha,
                temperature,
                queue,
            })?;
            println!("{}", serde_json::to_string_pretty(&report)?);
        }
    }
    Ok(())
}
