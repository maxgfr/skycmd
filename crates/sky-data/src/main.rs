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

/// The steps of the s6-secrets v2 data build. `--data` is the data folder (outside every
/// repository: `creddata/`, `gen/`, `issues/`, `splits.json`), `--out` its build folder.
#[derive(Subcommand)]
enum Secrets2Step {
    /// CredData candidates of the training and validation repositories: frozen labels checked,
    /// decontamination against the test repositories, weak negatives.
    Creddata {
        #[arg(long)]
        data: PathBuf,
        #[arg(long)]
        out: PathBuf,
        /// The model's tokenizer (data/tok/tok-s2-1024.json): the weak negatives are drawn
        /// over distinct states.
        #[arg(long)]
        tokenizer: PathBuf,
        #[arg(long, default_value = "/usr/share/dict/words")]
        dict: PathBuf,
    },
    /// Issue-report train and validation splits: decontamination against the test split, every
    /// positive value replaced, candidate records.
    Issues {
        #[arg(long)]
        data: PathBuf,
        #[arg(long)]
        out: PathBuf,
        #[arg(long, default_value = "/usr/share/dict/words")]
        dict: PathBuf,
    },
    /// States, decision records (<out>/records/<source>/{train,val}.jsonl) and the tables of
    /// the baselines.
    Mix {
        #[arg(long)]
        out: PathBuf,
        /// The model's tokenizer (data/tok/tok-s2-1024.json).
        #[arg(long)]
        tokenizer: PathBuf,
        #[arg(long, default_value = "/usr/share/dict/words")]
        dict: PathBuf,
    },
    /// The CredData share of the pretraining text: files of the training repositories, as MLM
    /// records, sized against the text the pretraining already reads.
    PretrainText {
        #[arg(long)]
        data: PathBuf,
        #[arg(long)]
        out: PathBuf,
        #[arg(long)]
        tokenizer: PathBuf,
        /// The text the pretraining already reads (data/text-open,data/text).
        #[arg(long, value_delimiter = ',', required = true)]
        text: Vec<PathBuf>,
        /// Share of the text sequences the CredData text is to be.
        #[arg(long, default_value_t = 0.30)]
        share: f64,
    },
    /// The states of the candidates of a split, for the evaluation: the table and the records
    /// the model and the logistic regression read (<out>.jsonl, <out>.records.jsonl). The test
    /// split is refused unless --test-open exists and --model holds its threshold.
    States {
        #[arg(long)]
        data: PathBuf,
        /// creddata:<split>, clean, issues:<split>, issues:<split>:obfuscated or
        /// issues:<split>:e2e (training, validation or test).
        #[arg(long)]
        corpus: String,
        /// The generator's raw export (creddata, clean and the end-to-end issue candidates).
        #[arg(long)]
        candidates: Option<PathBuf>,
        /// Prefix of the two files written.
        #[arg(long)]
        out: PathBuf,
        #[arg(long)]
        tokenizer: PathBuf,
        #[arg(long, default_value = "/usr/share/dict/words")]
        dict: PathBuf,
        /// The trained ternary GGUF, whose metadata holds skycmd-s2.secrets2.tau_f1 (test only).
        #[arg(long)]
        model: Option<PathBuf>,
        /// The file whose existence opens the test split.
        #[arg(long, default_value = sky_data::secrets2::states::TEST_OPEN)]
        test_open: PathBuf,
    },
    /// The sample of weak negatives that is read by hand (it stays in the data folder).
    HandRead {
        #[arg(long)]
        out: PathBuf,
    },
    /// <out>/BUILD.json: every count, the hashes of the inputs and of the outputs.
    Manifest {
        #[arg(long)]
        data: PathBuf,
        #[arg(long)]
        out: PathBuf,
        #[arg(long)]
        tokenizer: PathBuf,
        #[arg(long, default_value = "/usr/share/dict/words")]
        dict: PathBuf,
        /// The report of `sky-train s2-truncation --json` on the records.
        #[arg(long)]
        truncation: Option<PathBuf>,
        /// The result of the reading by hand (default: <out>/handread/result.json when there).
        #[arg(long)]
        hand_read: Option<PathBuf>,
        /// The model configuration the truncation was measured with.
        #[arg(long)]
        config: Option<PathBuf>,
    },
}

/// The inputs every source-reading step of the s6-route-nano build needs.
#[derive(clap::Args, Clone)]
struct RouteSources {
    /// The data folder: `src/` and `models/` as `scripts/route/pins.json` fetches them.
    #[arg(long)]
    data: PathBuf,
    /// `data/decisions/when2call.jsonl` (its train / val / test partition is v1's).
    #[arg(long, default_value = "data/decisions/when2call.jsonl")]
    when2call: PathBuf,
    #[arg(long, default_value = "data/decisions/banking77.jsonl")]
    banking77: PathBuf,
    #[arg(long, default_value = "data/commands/massive.jsonl")]
    massive: PathBuf,
}

impl RouteSources {
    fn paths(&self) -> sky_data::route::sources::SourcePaths {
        sky_data::route::sources::SourcePaths {
            src: self.data.join("src"),
            when2call: self.when2call.clone(),
            banking77: self.banking77.clone(),
            massive: self.massive.clone(),
            descriptions: Default::default(),
        }
    }
}

/// The steps of the s6-route-nano data build (`docs/s6-nano2-prereg.md`, section 4;
/// `scripts/build-route-data.sh` runs them in order). `--out` is the build folder.
#[derive(Subcommand)]
enum RouteStep {
    /// The intent names the teacher describes: <out>/teacher/describe-intent.items.jsonl.
    Intents {
        #[command(flatten)]
        src: RouteSources,
        #[arg(long)]
        out: PathBuf,
    },
    /// Every source read: <out>/stage/{tools,requests,texts}.jsonl (needs the intent
    /// descriptions, <out>/teacher/describe-intent.jsonl).
    Requests {
        #[command(flatten)]
        src: RouteSources,
        #[arg(long)]
        out: PathBuf,
    },
    /// Clusters, splits and caps; the test files (written once), the validation slates, the
    /// plan of the training slates and the items of the false-negative and rewrite jobs (needs
    /// <out>/emb from scripts/route-embed.py).
    Split {
        #[arg(long)]
        out: PathBuf,
    },
    /// The training and validation records (<out>/records/<group>/{train,val}.jsonl) with the
    /// teacher's answers.
    Slates {
        #[arg(long)]
        out: PathBuf,
        /// Without the teacher's answers (hard slates unfiltered, renamed gold with its own
        /// description): marked provisional in BUILD.json; never trained on outside a smoke run.
        #[arg(long)]
        provisional: bool,
    },
    /// The model's copy of a slate file: each state gets the BM25 line of its tool options
    /// (amendment 1). The evaluation renders the validation and test slates this way.
    /// `--render scores-in-options`: s6-route v4's D4 instead (each tool option prefixed with
    /// its score and rank, the state the request alone; docs/s6-route4-prereg.md 3.1).
    Render {
        #[arg(long = "in")]
        input: PathBuf,
        #[arg(long)]
        out: PathBuf,
        /// bm25-line (v2, the default) or scores-in-options (v4).
        #[arg(long, default_value = sky_infer::route::RENDER_LINE)]
        render: String,
    },
    /// s6-route v4 (docs/s6-route4-prereg.md, section 4), step 1: from v2's build (read only,
    /// its split recomputed and checked against its validation slates), the plan of v4
    /// (<out>/stage/plan4.jsonl), the validation slates v4_tools and v4_tools_masked (seed
    /// 1237), and the items of the one teacher job (<out>/teacher/rewrite-description.items.jsonl,
    /// the tools without a rewrite). `--seed` is v2's (1234).
    V4Plan {
        /// v2's build folder (read only).
        #[arg(long)]
        v2: PathBuf,
        /// v4's build folder.
        #[arg(long)]
        out: PathBuf,
        /// The one remedy of 4.4 (`surface-cell`), only after a data check failed on a surface rule.
        #[arg(long)]
        remedy: Option<String>,
    },
    /// s6-route v4, step 3 (after BUILD=<out> scripts/route-teacher.sh rewrite-description): the
    /// records of every kind (<out>/records/<group>/{train,val}.jsonl, D1, D2, D4, D5), the probe
    /// copies v4_tools_rw and v4_tools_masked_rw, and the model's copies of the validation files.
    V4Slates {
        #[arg(long)]
        v2: PathBuf,
        #[arg(long)]
        out: PathBuf,
        /// Without every rewrite (never trained on outside a smoke run).
        #[arg(long)]
        provisional: bool,
    },
    /// s6-route v5 (docs/s6-route5-prereg.md, section 4, amendment 1), step 2 (after
    /// scripts/route-embed.py --requests): from v2's build (read only, its split recomputed and
    /// checked) and v4's rewrites, the plan of v5 (<out>/stage/plan5.jsonl: every slate within
    /// the right tool's surface cell at its level, the hard slates the 16 candidates nearest to
    /// the request), v5_tools and v5_tools_masked (seed 1238), the counts by group, kind, split
    /// and level (<out>/counts/v5-plan.json, read by scripts/s6-route5-checks.py --plan for C6),
    /// and the false-negative items, staged (<out>/stage/false-negatives5.items.jsonl).
    /// `--seed` is v2's (1234).
    V5Plan {
        /// v2's build folder (read only).
        #[arg(long)]
        v2: PathBuf,
        /// v4's build folder (read only: its rewrites).
        #[arg(long)]
        v4: PathBuf,
        /// v5's build folder.
        #[arg(long)]
        out: PathBuf,
    },
    /// s6-route v5, step 4 (after BUILD=<out> scripts/route-teacher.sh false-negatives): the
    /// false-negative filter, the records of every kind (<out>/records/<group>/{train,val}.jsonl,
    /// with cell_level and the roles of the options), the probe copies and the model's copies.
    V5Slates {
        #[arg(long)]
        v2: PathBuf,
        #[arg(long)]
        v4: PathBuf,
        #[arg(long)]
        out: PathBuf,
        /// Without every false-negative answer (never trained on outside a smoke run).
        #[arg(long)]
        provisional: bool,
    },
    /// s6-nano v3 (docs/s6-nano3-prereg.md, section 4): the fresh test files, from v2's build
    /// (read only; its split recomputed and checked against its test files) into --out, written
    /// once, with FRESH.json (hashes and counts). Needs `--seed 1235`.
    Fresh {
        /// v2's build folder.
        #[arg(long)]
        build: PathBuf,
        /// The fresh folder (`<build>/fresh`).
        #[arg(long)]
        out: PathBuf,
        #[arg(long, default_value = "data/commands/massive.jsonl")]
        massive: PathBuf,
        /// v2's seed (its BUILD.json).
        #[arg(long, default_value_t = sky_data::route::fresh::V2_SEED)]
        v2_seed: u64,
        /// Where test-files.json and counts.json go (runs/s6-nano3).
        #[arg(long)]
        record: Option<PathBuf>,
    },
    /// s6-route v4 (docs/s6-route4-prereg.md 6.1, amendment 1), fresh test step 1: the SFT
    /// parquet files of Toucan-1.5M read into <out>/stage/rows.jsonl (the qualifying rows: one
    /// target, listed in the row's own tools as <server>-<tool>), counted in
    /// <out>/counts/toucan-rows.json.
    ToucanRows {
        /// The parquet files (comma separated).
        #[arg(long, value_delimiter = ',', required = true)]
        input: Vec<PathBuf>,
        /// The Toucan folder.
        #[arg(long)]
        out: PathBuf,
        /// s6-route v5 (docs/s6-route5-prereg.md 6): the manifest of the fetch
        /// (<data>/MANIFEST4.json) every input is hash-checked against before it is read.
        #[arg(long)]
        manifest: Option<PathBuf>,
    },
    /// s6-route v4 (6.1): the structure of the SFT files only (columns, types, subsets,
    /// skeletons of target_tools and tools, uuids, qualifying rows); no text is printed.
    ToucanSchema {
        #[arg(long, value_delimiter = ',', required = true)]
        input: Vec<PathBuf>,
    },
    /// s6-route v4, fresh test step 2: the tool table and the texts to embed
    /// (scripts/route-embed.py --build <t4>), from <t4>/stage/rows.jsonl.
    ToucanStage {
        #[arg(long)]
        t4: PathBuf,
    },
    /// s6-route v4 (6.2), fresh test step 3: t4_tools, its masked copy and the servers file,
    /// written once into <t4>/fresh with FRESH4.json, and the items of the probe copies' teacher
    /// job (a tool's name and description only). Needs `--seed 1236`.
    Toucan {
        /// v2's build folder (read only).
        #[arg(long)]
        v2: PathBuf,
        #[arg(long)]
        t4: PathBuf,
        /// Where test-files.json and counts.json go (runs/s6-route4, runs/s6-route5).
        #[arg(long)]
        record: Option<PathBuf>,
        /// The rule of 6.2, item 5: v4 (files t4_*) or v5 (docs/s6-route5-prereg.md 6: the
        /// fill within v5's surface cell, files t5_*).
        #[arg(long, default_value = "v4")]
        rule: String,
    },
    /// s6-route v4 (6.4), fresh test step 5 (after BUILD=<t4> scripts/route-teacher.sh
    /// rewrite-description): the probe copies t4_tools_rw and t4_tools_masked_rw, written once.
    /// Needs `--seed 1236`.
    ToucanProbes {
        #[arg(long)]
        t4: PathBuf,
        #[arg(long)]
        record: Option<PathBuf>,
        /// v4 (t4_*) or v5 (t5_*).
        #[arg(long, default_value = "v4")]
        rule: String,
    },
    /// <out>/BUILD.json: inputs, counts, hashes, the test files' hashes, the head window.
    Manifest {
        #[arg(long)]
        data: PathBuf,
        #[arg(long)]
        out: PathBuf,
        /// `s2-truncation --json` reports as <size>:<window>=<path>, one per size and window.
        #[arg(long, value_delimiter = ',')]
        truncation: Vec<String>,
    },
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
    /// s6-secrets v2 (`docs/s6-secrets2-prereg.md`, 3.3): synthetic candidates in the record
    /// format of the candidate generator (provider-format keys, word-based passwords, and the
    /// hard negatives), written to <out>/candidates/synth-{train,val}.jsonl.
    SecretCand {
        /// The build folder.
        #[arg(long)]
        out: PathBuf,
        /// Training candidates (default: 20,000, or a third of the candidates of the two other
        /// sources read from <out>/counts when that is fewer).
        #[arg(long)]
        n_train: Option<usize>,
        #[arg(long, default_value_t = sky_data::secrets2::synth::VAL_CANDIDATES)]
        n_val: usize,
        /// Word list of the dictionary (lower-case entries of 4 to 12 letters are kept).
        #[arg(long, default_value = "/usr/share/dict/words")]
        dict: PathBuf,
    },
    /// s6-secrets v2: the data between the frozen candidate generator and the trainer
    /// (`scripts/build-secrets2-data.sh` runs the steps in order).
    Secrets2 {
        #[command(subcommand)]
        step: Secrets2Step,
    },
    /// s6-route-nano: the data of a tool router that reads descriptions
    /// (`scripts/build-route-data.sh` runs the steps in order).
    Route {
        #[command(subcommand)]
        step: RouteStep,
    },
    /// s6-ear: audio clips → log-mel features (`sky_infer::audio`), test WAVs and clip manifests
    /// under --out (`docs/s6-ear-prereg.md`). Sets: smart-turn, speech-commands, mswc, mimi
    /// (LibriSpeech + the published Mimi codes, for `s2-pretrain --audio`), or all.
    FetchAudio {
        #[arg(long, value_delimiter = ',', default_value = "all")]
        set: Vec<String>,
        #[arg(long, default_value = "data/ear")]
        out: PathBuf,
        /// Clips kept per set (0: all; mswc: per archive).
        #[arg(long, default_value_t = 0)]
        max_clips: usize,
        /// Keep the downloaded parquet shards in the cache (default: removed once processed).
        #[arg(long)]
        keep_shards: bool,
        /// MSWC archives read per language and split (about 650 MB of clips each).
        #[arg(long, default_value_t = 1)]
        mswc_archives: usize,
        #[arg(long, value_delimiter = ',', default_value = "en,fr")]
        mswc_langs: Vec<String>,
    },
    /// s6-ear v2: the training feature store, one shard of smart-turn-data-v3.2-train at a time,
    /// with the leakage filter of docs/s6-ear2-prereg.md §3.2 (needs turn-test's hashes). A shard
    /// stays until scripts/turn-teacher.py has written its targets. Exit status 3: stopped by the
    /// disk rule (resumable).
    FetchTurn {
        /// The data folder (outside the repository).
        #[arg(long)]
        out: PathBuf,
        /// Shards to process (`0,1,2` or `0-82`; default: all).
        #[arg(long, default_value = "")]
        shards: String,
        /// Clips read per shard (0: all; a dry run).
        #[arg(long, default_value_t = 0)]
        max_clips: usize,
        #[arg(long, default_value_t = 4)]
        threads: usize,
        /// Free space the build always leaves on the data folder's disk (GB).
        #[arg(long, default_value_t = 8.0)]
        min_free_gb: f64,
        /// Only remove the shards whose teacher targets are written.
        #[arg(long)]
        drop: bool,
    },
    /// s6-ear v2: features and 16-bit PCM hashes of v1's test WAVs (no label is read or written).
    TurnTest {
        /// v1's clip manifest (its test lines name the WAVs).
        #[arg(long, default_value = "data/ear/clips/smart-turn.jsonl")]
        clips: PathBuf,
        #[arg(long)]
        out: PathBuf,
        #[arg(long, default_value_t = 4)]
        threads: usize,
        /// Test clips read (0: all; a dry run).
        #[arg(long, default_value_t = 0)]
        max: usize,
        #[arg(long, default_value_t = 8.0)]
        min_free_gb: f64,
    },
    /// s6-ear v2: the eot-bench cuts of §8.4 (every eot / hold span, the audio cut at its start),
    /// one parquet file at a time. Exit status 3: stopped by the disk rule.
    TurnEotbench {
        #[arg(long)]
        out: PathBuf,
        /// Files of the listing to process (`0,3` or `0-14`; default: all).
        #[arg(long, default_value = "")]
        files: String,
        /// Turns read per file (0: all; a dry run).
        #[arg(long, default_value_t = 0)]
        max_turns: usize,
        #[arg(long, default_value_t = 4)]
        threads: usize,
        #[arg(long, default_value_t = 8.0)]
        min_free_gb: f64,
        /// Only remove the files whose Smart Turn pass is written.
        #[arg(long)]
        drop: bool,
    },
    /// s6-ear: the audio jev from the clip manifests of fetch-audio, one record per question
    /// (<dir>/jev/{turn_end,keyword,keyword_zs[,about]}.jsonl and REPORT.json).
    AudioJev {
        #[arg(long, default_value = "data/ear")]
        dir: PathBuf,
        /// GLM labels of the transcript questions (`teacher label-transcripts`): adds about.jsonl.
        #[arg(long)]
        about: Option<PathBuf>,
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
    /// s6-ear v2: GLM probabilities for the `about` questions of every LibriSpeech transcript
    /// (data/ear/mimi), resumable like label-decisions; `audio-jev --about` turns them into
    /// audio records (the teacher reads the text, the student hears the audio).
    LabelTranscripts {
        #[arg(long, default_value = "data/ear")]
        dir: PathBuf,
        #[arg(long, default_value = "data/teacher/ear_about_labels.jsonl")]
        out: PathBuf,
        #[arg(long, default_value_t = 4)]
        jobs: usize,
        #[arg(long, default_value_t = 0.2)]
        temperature: f64,
    },
    /// s6-route-nano (4.5): one description per intent, from its name.
    DescribeIntent {
        #[command(flatten)]
        job: RouteJobArgs,
    },
    /// s6-route-nano (4.5): which tools of a hard slate could handle its request.
    FalseNegatives {
        #[command(flatten)]
        job: RouteJobArgs,
    },
    /// s6-route-nano (4.5): one rewrite of a tool's description.
    RewriteDescription {
        #[command(flatten)]
        job: RouteJobArgs,
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

/// The files of one s6-route-nano teacher job; resumable, a test item refused.
#[derive(clap::Args, Clone)]
struct RouteJobArgs {
    /// <build>/teacher/<job>.items.jsonl (sky-data route).
    #[arg(long)]
    items: PathBuf,
    /// <build>/teacher/<job>.jsonl: answers are appended; a rerun does what is left.
    #[arg(long)]
    out: PathBuf,
    #[arg(long, default_value_t = 4)]
    jobs: usize,
    #[arg(long, default_value_t = 0.2)]
    temperature: f64,
}

/// `0,3,5-7` → `[0, 3, 5, 6, 7]` (empty: none).
fn parse_ranges(s: &str) -> Result<Vec<usize>> {
    let mut out = Vec::new();
    for part in s.split(',').map(str::trim).filter(|p| !p.is_empty()) {
        match part.split_once('-') {
            Some((a, b)) => out.extend(a.parse::<usize>()?..=b.parse::<usize>()?),
            None => out.push(part.parse()?),
        }
    }
    Ok(out)
}

/// A step stopped by the disk rule exits with status 3 (the scripts stop there and resume later).
fn disk_rule<T>(r: Result<T>) -> Result<T> {
    match r {
        Err(e) if e.downcast_ref::<sky_data::turn::DiskLow>().is_some() => {
            eprintln!("{e}");
            std::process::exit(3);
        }
        r => r,
    }
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
        Cmd::SecretCand {
            out,
            n_train,
            n_val,
            dict,
        } => {
            use sky_data::secrets2::{load_words, synth};
            let c = synth::write(&out, cli.seed, n_train, n_val, &load_words(&dict)?)?;
            println!("{}", serde_json::to_string_pretty(&c)?);
        }
        Cmd::Secrets2 { step } => {
            use sky_data::secrets2::{self as s2, load_tokenizer, load_words};
            let report = match step {
                Secrets2Step::Creddata {
                    data,
                    out,
                    tokenizer,
                    dict,
                } => s2::cred::build(&s2::cred::CredOptions {
                    data,
                    out,
                    seed: cli.seed,
                    tokenizer: load_tokenizer(&tokenizer)?,
                    words: load_words(&dict)?,
                })?,
                Secrets2Step::Issues { data, out, dict } => {
                    s2::issues::build(&s2::issues::IssueOptions {
                        data,
                        out,
                        seed: cli.seed,
                        words: load_words(&dict)?,
                    })?
                }
                Secrets2Step::Mix {
                    out,
                    tokenizer,
                    dict,
                } => s2::mix::build(&s2::mix::MixOptions {
                    out,
                    tokenizer: load_tokenizer(&tokenizer)?,
                    words: load_words(&dict)?,
                })?,
                Secrets2Step::PretrainText {
                    data,
                    out,
                    tokenizer,
                    text,
                    share,
                } => s2::pretext::build(&s2::pretext::PretextOptions {
                    data,
                    out,
                    seed: cli.seed,
                    tokenizer: load_tokenizer(&tokenizer)?,
                    text,
                    share,
                })?,
                Secrets2Step::States {
                    data,
                    corpus,
                    candidates,
                    out,
                    tokenizer,
                    dict,
                    model,
                    test_open,
                } => s2::states::build(&s2::states::StatesOptions {
                    data,
                    corpus: s2::states::Corpus::parse(&corpus)?,
                    candidates,
                    out,
                    tokenizer: load_tokenizer(&tokenizer)?,
                    words: load_words(&dict)?,
                    model,
                    test_open,
                })?,
                Secrets2Step::HandRead { out } => s2::build::hand_read_sample(&out, cli.seed)?,
                Secrets2Step::Manifest {
                    data,
                    out,
                    tokenizer,
                    dict,
                    truncation,
                    hand_read,
                    config,
                } => {
                    let hand_read = hand_read.or_else(|| Some(out.join("handread/result.json")));
                    s2::build::manifest(&s2::build::ManifestOptions {
                        data,
                        out,
                        seed: cli.seed,
                        dict,
                        tokenizer,
                        truncation,
                        hand_read,
                        config,
                    })?;
                    serde_json::json!({"written": "BUILD.json"})
                }
            };
            // counts only: no step prints a value or a line
            println!("{}", serde_json::to_string_pretty(&report)?);
        }
        Cmd::Route { step } => {
            use sky_data::route::build as rb;
            let report = match step {
                RouteStep::Intents { src, out } => rb::intents(&src.paths(), &out)?,
                RouteStep::Requests { src, out } => rb::requests(&src.paths(), &out)?,
                RouteStep::Split { out } => rb::split(&rb::SplitOptions {
                    out,
                    seed: cli.seed,
                })?,
                RouteStep::Slates { out, provisional } => rb::build_slates(&rb::SlatesOptions {
                    out,
                    seed: cli.seed,
                    provisional,
                })?,
                RouteStep::Render { input, out, render } => {
                    let how = sky_infer::route::Render::parse(&render).ok_or_else(|| {
                        anyhow::anyhow!("--render {render}: bm25-line or scores-in-options")
                    })?;
                    rb::render_with(&input, &out, how)?
                }
                RouteStep::V4Plan { v2, out, remedy } => {
                    use sky_data::route::v4;
                    let remedy = match remedy.as_deref() {
                        None => false,
                        Some("surface-cell") => true,
                        Some(r) => anyhow::bail!(
                            "--remedy {r}: the one remedy of docs/s6-route4-prereg.md 4.4 is surface-cell"
                        ),
                    };
                    if cli.seed != v4::V2_SEED {
                        anyhow::bail!("v4 keeps v2's streams: --seed {}", v4::V2_SEED);
                    }
                    v4::plan(&v4::PlanOptions {
                        v2,
                        out,
                        seed: cli.seed,
                        remedy,
                    })?
                }
                RouteStep::V4Slates {
                    v2,
                    out,
                    provisional,
                } => {
                    use sky_data::route::v4;
                    if cli.seed != v4::V2_SEED {
                        anyhow::bail!("v4 keeps v2's streams: --seed {}", v4::V2_SEED);
                    }
                    v4::slates(&v4::SlatesOptions {
                        v2,
                        out,
                        seed: cli.seed,
                        provisional,
                    })?
                }
                RouteStep::V5Plan { v2, v4, out } => {
                    use sky_data::route::v5;
                    if cli.seed != sky_data::route::v4::V2_SEED {
                        anyhow::bail!(
                            "v5 keeps v2's streams: --seed {}",
                            sky_data::route::v4::V2_SEED
                        );
                    }
                    v5::plan(&v5::PlanOptions {
                        v2,
                        v4,
                        out,
                        seed: cli.seed,
                    })?
                }
                RouteStep::V5Slates {
                    v2,
                    v4,
                    out,
                    provisional,
                } => {
                    use sky_data::route::v5;
                    if cli.seed != sky_data::route::v4::V2_SEED {
                        anyhow::bail!(
                            "v5 keeps v2's streams: --seed {}",
                            sky_data::route::v4::V2_SEED
                        );
                    }
                    v5::slates(&v5::SlatesOptions {
                        v2,
                        v4,
                        out,
                        seed: cli.seed,
                        provisional,
                    })?
                }
                RouteStep::Fresh {
                    build,
                    out,
                    massive,
                    v2_seed,
                    record,
                } => {
                    use sky_data::route::fresh as rf;
                    if cli.seed != rf::FRESH_SEED {
                        anyhow::bail!(
                            "the fresh files are built with --seed {} (docs/s6-nano3-prereg.md, section 4)",
                            rf::FRESH_SEED
                        );
                    }
                    rf::fresh(&rf::FreshOptions {
                        build,
                        out,
                        massive: Some(massive),
                        seed: cli.seed,
                        v2_seed,
                        cap: rf::FRESH_CAP,
                        record,
                    })?
                }
                RouteStep::ToucanRows {
                    input,
                    out,
                    manifest,
                } => {
                    let checked = match &manifest {
                        Some(m) => Some(sky_data::route::toucan::check_manifest(&input, m)?),
                        None => None,
                    };
                    let mut c = sky_data::route::toucan::rows(&input, &out)?;
                    if let Some(x) = checked {
                        c["hash check (docs/s6-route5-prereg.md 6)"] = x;
                    }
                    c
                }
                RouteStep::ToucanSchema { input } => sky_data::route::toucan::schema(&input)?,
                RouteStep::ToucanStage { t4 } => sky_data::route::toucan::stage(&t4)?,
                RouteStep::Toucan {
                    v2,
                    t4,
                    record,
                    rule,
                } => {
                    use sky_data::route::toucan as tc;
                    let rule = tc::Rule::parse(&rule)
                        .ok_or_else(|| anyhow::anyhow!("--rule {rule}: v4 or v5"))?;
                    if cli.seed != tc::T4_SEED {
                        anyhow::bail!(
                            "the fresh test is built with --seed {} (docs/s6-route4-prereg.md 6.2; docs/s6-route5-prereg.md 6)",
                            tc::T4_SEED
                        );
                    }
                    tc::build_test(&tc::TestOptions {
                        v2,
                        t4,
                        seed: cli.seed,
                        record,
                        rule,
                    })?
                }
                RouteStep::ToucanProbes { t4, record, rule } => {
                    use sky_data::route::toucan as tc;
                    let rule = tc::Rule::parse(&rule)
                        .ok_or_else(|| anyhow::anyhow!("--rule {rule}: v4 or v5"))?;
                    if cli.seed != tc::T4_SEED {
                        anyhow::bail!(
                            "the probe copies are built with --seed {} (docs/s6-route4-prereg.md 6.4)",
                            tc::T4_SEED
                        );
                    }
                    tc::probes(&t4, cli.seed, record.as_deref(), rule)?
                }
                RouteStep::Manifest {
                    data,
                    out,
                    truncation,
                } => {
                    let truncation = truncation
                        .iter()
                        .map(|t| {
                            let bad = || {
                                anyhow::anyhow!(
                                    "--truncation wants <size>:<window>=<path>, not {t}"
                                )
                            };
                            let (k, p) = t.split_once('=').ok_or_else(bad)?;
                            let (size, w) = k.split_once(':').ok_or_else(bad)?;
                            Ok((size.to_string(), w.parse::<usize>()?, PathBuf::from(p)))
                        })
                        .collect::<Result<Vec<_>>>()?;
                    rb::manifest(&rb::ManifestOptions {
                        data,
                        out,
                        seed: cli.seed,
                        truncation,
                    })?
                }
            };
            // counts only: no step prints a request or a tool of the test side
            println!("{}", serde_json::to_string_pretty(&report)?);
        }
        Cmd::FetchAudio {
            set,
            out,
            max_clips,
            keep_shards,
            mswc_archives,
            mswc_langs,
        } => {
            use sky_data::audio;
            let hf = Hf::new(&cli.cache)?;
            let o = audio::FetchAudio {
                out,
                max_clips,
                seed: cli.seed,
                drop_shards: !keep_shards,
                mswc_archives,
                mswc_langs,
            };
            let all = set.iter().any(|s| s == "all");
            let want = |s: &str| all || set.iter().any(|x| x == s);
            for s in &set {
                if !["all", "smart-turn", "speech-commands", "mswc", "mimi"].contains(&s.as_str()) {
                    anyhow::bail!(
                        "unknown audio set {s:?} (smart-turn, speech-commands, mswc, mimi, all)"
                    );
                }
            }
            let mut reports = Vec::new();
            if want("smart-turn") {
                reports.push(audio::fetch_smart_turn(&hf, &o)?);
            }
            if want("speech-commands") {
                reports.push(audio::fetch_speech_commands(&hf, &o)?);
            }
            if want("mswc") {
                reports.push(audio::fetch_mswc(&hf, &o)?);
            }
            if want("mimi") {
                reports.push(audio::fetch_mimi(&hf, &o)?);
            }
            for r in &reports {
                println!(
                    "{}: train {} val {} test {} (skipped {})",
                    r.set, r.train, r.val, r.test, r.skipped
                );
            }
        }
        Cmd::FetchTurn {
            out,
            shards,
            max_clips,
            threads,
            min_free_gb,
            drop,
        } => {
            use sky_data::turn;
            let hf = Hf::new(out.join("hf"))?;
            let o = turn::FetchOptions {
                guard: turn::Guard::new(&out, min_free_gb),
                out,
                shards: parse_ranges(&shards)?,
                max_clips,
                threads,
                drop_only: drop,
            };
            let v = disk_rule(turn::fetch_turn(&hf, &o))?;
            println!("{}", serde_json::to_string_pretty(&v)?);
        }
        Cmd::TurnTest {
            clips,
            out,
            threads,
            max,
            min_free_gb,
        } => {
            use sky_data::turn;
            let o = turn::TestOptions {
                guard: turn::Guard::new(&out, min_free_gb),
                clips,
                out,
                threads,
                max,
            };
            let v = disk_rule(turn::turn_test(&o))?;
            println!("{}", serde_json::to_string_pretty(&v)?);
        }
        Cmd::TurnEotbench {
            out,
            files,
            max_turns,
            threads,
            min_free_gb,
            drop,
        } => {
            use sky_data::turn;
            let hf = Hf::new(out.join("hf"))?;
            let o = turn::EotOptions {
                guard: turn::Guard::new(&out, min_free_gb),
                out,
                files: parse_ranges(&files)?,
                max_turns,
                threads,
                drop_only: drop,
            };
            let v = disk_rule(turn::turn_eotbench(&hf, &o))?;
            println!("{}", serde_json::to_string_pretty(&v)?);
        }
        Cmd::AudioJev { dir, about } => {
            let r = sky_data::audio::write_jev(&dir, cli.seed, about.as_deref())?;
            for (task, c) in &r.counts {
                println!("{task}: {c:?}");
            }
            for (lang, w) in &r.held_out {
                println!("held-out words {lang}: {}", w.len());
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
            let route_job = match &task {
                TeacherTask::DescribeIntent { job } => {
                    Some((teacher::RouteJob::DescribeIntent, job.clone()))
                }
                TeacherTask::FalseNegatives { job } => {
                    Some((teacher::RouteJob::FalseNegatives, job.clone()))
                }
                TeacherTask::RewriteDescription { job } => {
                    Some((teacher::RouteJob::RewriteDescription, job.clone()))
                }
                _ => None,
            };
            if let Some((kind, job)) = route_job {
                let mut cfg = cfg;
                cfg.temperature = job.temperature;
                cfg.max_tokens = 8192;
                let s = teacher::route_label(&cfg, kind, &job.items, &job.out, job.jobs)?;
                println!(
                    "{}: wrote {} answers to {:?} ({} refused by moderation); calls {} ({} failed batches), tokens in {} out {}, {:.1}s",
                    kind.name(),
                    s.written,
                    job.out,
                    s.refused,
                    s.calls,
                    s.failed_batches,
                    s.input_tokens,
                    s.output_tokens,
                    t0.elapsed().as_secs_f64()
                );
                return Ok(());
            }
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
                        "wrote {} labels to {out:?} ({} refused by moderation); calls {} ({} failed batches), tokens in {} out {}, {:.1}s",
                        s.written,
                        s.refused,
                        s.calls,
                        s.failed_batches,
                        s.input_tokens,
                        s.output_tokens,
                        t0.elapsed().as_secs_f64()
                    );
                    return Ok(());
                }
                TeacherTask::LabelTranscripts {
                    dir,
                    out,
                    jobs,
                    temperature,
                } => {
                    t.cfg.temperature = temperature;
                    t.cfg.max_tokens = 8192;
                    let recs = sky_data::audio::transcript_records(&dir)?;
                    let o = sky_data::distill::LabelOptions {
                        from: Vec::new(),
                        per_source: 0,
                        max_state_chars: 2000,
                        jobs,
                        seed: cli.seed,
                    };
                    let s = sky_data::distill::label_records(&t.cfg, recs, &o, &out)?;
                    println!(
                        "wrote {} transcript labels to {out:?} ({} refused by moderation); calls {} ({} failed batches), tokens in {} out {}, {:.1}s",
                        s.written,
                        s.refused,
                        s.calls,
                        s.failed_batches,
                        s.input_tokens,
                        s.output_tokens,
                        t0.elapsed().as_secs_f64()
                    );
                    return Ok(());
                }
                TeacherTask::DescribeIntent { job }
                | TeacherTask::FalseNegatives { job }
                | TeacherTask::RewriteDescription { job } => {
                    unreachable!("{:?}", job.items)
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
