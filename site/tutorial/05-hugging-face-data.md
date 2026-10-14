# 05 Pulling data from Hugging Face in Rust

`sky-data` downloads public Hugging Face datasets over plain HTTPS, reads their parquet shards in Rust, and converts every row into a typed decision record or an MLM text record. No Python is involved. This chapter covers the Hub client, partial reads of large remote shards, how each dataset becomes typed questions, and the deduplication, capping and splitting that keep the data clean.

## A small Hub client

The crate depends on `ureq` for HTTP and on the `parquet` crate for reading files. There is no Hub SDK: every source is public and ungated, so a URL plus a local cache is all the client needs.

`crates/sky-data/src/hf.rs`

```rust
const HUB: &str = "https://huggingface.co";
const RETRIES: u32 = 4;

/// URL of a file in a dataset repository at a given revision.
pub fn resolve_url(repo: &str, rev: &str, file: &str) -> String {
    let rev = rev.replace('/', "%2F");
    format!("{HUB}/datasets/{repo}/resolve/{rev}/{file}")
}
```

`Hf::new(cache)` creates the cache directory, which defaults to `data/hf` through the global `--cache` flag. It configures a 600 s timeout and a `skycmd-sky-data/0.1` user agent. Every request goes through `retry`, which makes 4 attempts with exponential backoff starting at 500 ms. `download` writes to a `.part` file first and renames it at the end, so an interrupted download never looks complete.

### Finding parquet shards

Many datasets do not store parquet files themselves. The Hub's dataset viewer converts them anyway and lists the shards through an API endpoint, so one function covers both cases:

`crates/sky-data/src/hf.rs`

```rust
    pub fn parquet_urls(&self, repo: &str, config: &str, split: &str) -> Result<Vec<String>> {
        let url = format!("{HUB}/api/datasets/{repo}/parquet/{config}/{split}");
        let v = self.get_json(&url)?;
        let urls: Vec<String> = v
            .as_array()
            .ok_or_else(|| anyhow!("unexpected parquet listing for {repo}: {v}"))?
            .iter()
            .filter_map(|u| u.as_str().map(str::to_string))
            .collect();
        if urls.is_empty() {
            bail!("no parquet shards for {repo} {config}/{split}");
        }
        Ok(urls)
    }
```

Small datasets are fetched whole with `parquet_split`, with each file capped at 512 MiB (`SMALL`). `read_parquet` then reads only the projected columns as JSON rows.

## Reading only part of a large shard

FineWeb shards are far bigger than the few tens of megabytes skycmd needs. `RemoteParquet` mirrors a remote file into a **sparse** local file and downloads only the byte ranges that are actually read. Parquet keeps its metadata at the end of the file, so opening a shard costs three small range requests: the magic header, the 8-byte tail, and the footer.

`crates/sky-data/src/hf.rs`

```rust
            let meta_len = u32::from_le_bytes(tail8[..4].try_into().unwrap()) as u64;
            let start = total - 8 - meta_len;
            let (footer, _) = hf.get_range(url, start, meta_len + 8)?;
            let mut f = File::create(&path)?;
            f.set_len(total)?;
            f.write_all(b"PAR1")?;
            f.seek(SeekFrom::Start(start))?;
            f.write_all(&footer)?;
            fetched = vec![(0, 4), (start, total)];
            fs::write(&side, serde_json::to_vec(&fetched)?)?;
```

The file is created at full length with `set_len` but stays sparse on disk. The parquet reader can open it normally because the footer sits at the right offset. `ensure(rg, columns)` then fetches only the column chunks of one row group. Each fetched range is recorded in a `.ranges` sidecar, so a later run picks up where the last one stopped. `get_range` rejects any response other than `206 Partial Content`, so a server that ignores the `Range` header cannot silently send the whole file.

## Turning datasets into typed questions

`fetch-decisions` runs a list of sources. Each one has a loader function, a `test_only` flag for external benchmarks, and an optional cap that overrides `--cap` (`src` uses the default cap, `capped` sets its own):

`crates/sky-data/src/decisions.rs`

```rust
pub fn sources() -> Vec<Source> {
    use crate::nlu::*;
    use crate::typed::*;
    vec![
        src("typed_synth", load_typed_synth),
        src("assistant_decisions", load_assistant_decisions),
        src("pngwn_typed", load_pngwn),
        src("boolq", load_boolq),
        src("mnli", load_mnli),
        src("arc", load_arc),
        src("commonsense_qa", load_csqa),
        src("banking77", load_banking77),
        Source {
            name: "typed_decisions_bench",
            test_only: true,
            cap: None,
            load: load_bench,
        },
        // Typed-decision corpora on the Hub.
        capped("tasksource_jev", 120_000, load_tasksource_jev),
        capped("procedural_typed", 80_000, load_procedural),
        // The loader applies its own cap (all drone/robot rows + 30k others).
        capped("open_jev", usize::MAX, load_open_jev),
        capped("swarm_laya", usize::MAX, load_swarm_laya),
        src("helmo_synth", load_helmo),
        src("tasksource_synth", load_tasksource_synth),
        src("calibrated_typed", load_calibrated),
        Source {
            name: "notam",
            test_only: true,
            cap: Some(usize::MAX),
            load: load_notam,
        },
        // Classic NLU / preference sets converted to typed questions.
        src("helpsteer2", load_helpsteer2),
        src("pawsx_fr", load_pawsx_fr),
        capped("allocine", 30_000, load_allocine),
        src("when2call", load_when2call),
        capped("snli", 40_000, load_snli),
        capped("hellaswag", 30_000, load_hellaswag),
        // GLM teacher data (local, no download).
        capped("glm_orders", usize::MAX, crate::augment::load_glm_orders),
    ]
}
```

`sky-data fetch-decisions --list` prints the 24 names with their caps. The first eight sources and the benchmark are described below. The others were added for the s2 and s3 models (loaders in `crates/sky-data/src/typed.rs` and `nlu.rs`):

- **More typed-decision corpora**: `tasksource/tasksource-jev-typed-decisions` (only rows marked for commercial use, minus sources that duplicate a set loaded directly), `tasksource/procedural-typed-decisions`, `ZefanCai/Open-Jev` (all 18,130 drone-control rows plus 30k from its other sources), `Aditharavind/swarm-laya-decisions`, `helmo/synthetic-typed-decisions`, `tasksource/synthetic-typed-decisions` and `m-lagnajit/calibrated-typed-decisions`.
- **More NLU and preference sets**: HelpSteer2 (one 5-level score per attribute), PAWS-X French and Allociné (French nouls), When2Call (a 4-way choice between calling a tool, asking for information, refusing and answering directly), SNLI and HellaSwag.
- **A second external benchmark**: `AirsideLabs/notam-typed-decisions` (`notam`), evaluation only, like `typed_decisions_bench`.
- **GLM teacher data**: `glm_orders`, the validated drone orders of [chapter 06](./06-glm-top-up), read from `data/teacher` without any download.

### Native System One datasets

`n4ze3m/typed-decisions-synth`, `kgrozdanovski/assistant-decisions` and `LocalLLaMA/typed-decisions` already use the `questions` and `criteria` shape. `jev_items` parses each question with `Question::from_jev` and tries the label sources in order. For the synthetic set, soft teacher labels come first and the gold label is the fallback. `target_from` accepts `{"probabilities": ...}`, `{"noul": p}` or a hard label, and turns a hard label into a one-hot vector.

### Classic NLU sets, rephrased

Public benchmarks become typed questions with fixed templates. BoolQ becomes a statement check:

`crates/sky-data/src/decisions.rs`

```rust
/// BoolQ question → statement check (`noul`).
pub fn boolq_item(question: &str, passage: &str, answer: bool) -> Item {
    let q = capitalize(question.trim().trim_end_matches('?'));
    let ins = format!("Based on the passage, the answer to \"{q}?\" is yes.");
    let rec = Question::noul(ins).record("boolq", passage.trim(), one_hot(2, answer as usize));
    Item {
        group: format!("{:x}", hash64(passage.as_bytes())),
        rec,
    }
}
```

- **MNLI** becomes a choice over `entailment`, `neutral` and `contradiction`, each with a short description. The premise is the state.
- **ARC and CommonsenseQA** become a choice over keys `A..` with the answer texts as descriptions. The question is the state.
- **Banking77** becomes a choice over the gold intent plus 7 random distractors, shuffled, with intent names made readable (`card_arrival` becomes `card arrival`, `atm_support` becomes `ATM support`). Its CSV files are downloaded from the dataset's source repository, because the Hub copy is a script dataset.

### Decoding a token-id dataset

`pngwn/typed-decisions` stores prompts as GPT-2 token ids in `.npz` files. `sky-data` includes a small `.npz`/`.npy` reader for 2-D `uint16` arrays (`crates/sky-data/src/npz.rs`). It downloads GPT-2's `tokenizer.json`, decodes each prompt with the `tokenizers` crate, and parses the `### State:` / `### Question:` layout. Each question is then mapped to a type based on its options:

`crates/sky-data/src/decisions.rs`

```rust
    if lower == ["yes", "no"] || lower == ["no", "yes"] {
        let p_yes = if lower[0] == "yes" { target[0] } else { target[1] };
        return Some((Question::noul(ins), state, normalize(&[1.0 - p_yes, p_yes])));
    }
    let numeric: Option<Vec<i64>> = options.iter().map(|o| o.trim().parse::<i64>().ok()).collect();
    if let Some(n) = numeric {
        if n.windows(2).all(|w| w[1] == w[0] + 1) {
            let levels = options.iter().map(|o| o.trim().to_string()).collect();
            return Some((Question::score(ins, levels), state, normalize(target)));
        }
    }
```

Yes/no options become a noul, consecutive integers become a score, and everything else becomes a choice. Questions longer than 160 characters move into the state, with a generic instruction in their place, because they would not fit the 96-token head.

## Hygiene: validate, dedupe, cap, split

Every source goes through `finalize`:

1. **Validate.** `validate_record` rejects records with fewer than two options, a target that does not sum to 1, a malformed noul, or empty instructions.
2. **Deduplicate.** A 64-bit hash of instructions, options and state drops exact repeats.
3. **Cap with balance.** `balanced_cap` keeps at most `--cap` records (default 60,000). It draws round-robin from strata keyed by `(type, #options, argmax)`, so the cap never skews the label distribution toward whatever rows came first.
4. **Split by group.** Every record carries a `group`, such as the passage hash, the case id or the state id. The split is decided per group, so questions about the same passage never appear in both train and val:

`crates/sky-data/src/decisions.rs`

```rust
    for it in kept.iter_mut() {
        it.rec.split = if test_only || it.rec.split == "test" {
            "test".into()
        } else if (hash64(format!("{src}/{}", it.group).as_bytes()) % 1_000_000) as f64 / 1e6
            < opts.val_frac
        {
            "val".into()
        } else {
            "train".into()
        };
    }
```

`LocalLLaMA/typed-decisions` (`typed_decisions_bench`) and `AirsideLabs/notam-typed-decisions` (`notam`) are marked `test_only`. All of their records are `split = "test"`, and they are never trained on. `typed_decisions_bench` is the external benchmark behind the row of the same name in `runs/*/report.md`. A loader can also keep a labelled test split of its own: When2Call keeps its `test/mcq` split (3,630 records) as `split = "test"`.

## MLM text

Pretraining text comes from five sources, each with a share of the byte budget:

| name | repo | share |
|---|---|---:|
| simplestories | `SimpleStories/SimpleStories` | 0.20 |
| tinystories | `roneneldan/TinyStories` | 0.20 |
| fineweb_edu | `HuggingFaceFW/fineweb-edu`, shard `sample/10BT/000_00000.parquet` | 0.25 |
| tinystories_fr | `btecsec/tinystories-fr` | 0.20 |
| fineweb2_fr | `HuggingFaceFW/fineweb-2`, shard `data/fra_Latn/train/000_00000.parquet` | 0.15 |

The shares come from `text_sources()` in `crates/sky-data/src/text.rs`. The French stories corpus is small, so its unused budget passes to fineweb-2 French (`pass_shortfall: true`), which keeps the language mix intact. `clean_text` normalizes newlines and cuts long documents at a sentence boundary. A single streaming `Dedup` then removes exact duplicates (case- and whitespace-insensitive) and near duplicates:

`crates/sky-data/src/minhash.rs`

```rust
pub const SHINGLE: usize = 5;
pub const NUM_PERM: usize = 128;
pub const BANDS: usize = 16;
pub const ROWS: usize = NUM_PERM / BANDS;
```

MinHash runs over word 5-gram shingles, with 128 permutations in 16 LSH bands. A candidate counts as a duplicate when its estimated Jaccard similarity is 0.8 or higher (`Dedup::new(0.8, ...)` in `text.rs`).

## Licenses and row counts

`DATA_LICENSES.md` lists every source with its license and the rows written with default caps and seed 1234. Some examples:

| dataset | license | rows used |
|---|---|---:|
| n4ze3m/typed-decisions-synth | MIT | 25,859 |
| kgrozdanovski/assistant-decisions | Apache-2.0 | 15,455 |
| pngwn/typed-decisions | CC-BY-SA-4.0 | 51,019 |
| nyu-mll/glue (mnli) | see the MultiNLI / GLUE card | 60,000 |
| LocalLLaMA/typed-decisions | Apache-2.0 | 2,000 (test only) |

Generated files live under `data/`, which is gitignored, and the repository does not redistribute them. `Salesforce/xlam-function-calling-60k` was skipped because it is gated, and Hub drone-command datasets without a declared license were not used.

## Commands

```bash
export CARGO_TARGET_DIR=$PWD/target/sky-data
cargo build --release -p sky-data
D=$CARGO_TARGET_DIR/release/sky-data

$D fetch-decisions --list                           # source names and caps
$D fetch-decisions                                  # all sources -> data/decisions/<src>.jsonl
$D fetch-decisions --only boolq,mnli --cap 20000    # a subset, smaller cap
$D fetch-text --max-mb 150                          # -> data/text/<src>.jsonl
$D fetch-text --only simplestories --max-mb 10
$D stats                                            # -> data/STATS.md
$D --cache /Volumes/ssd/hf --seed 7 fetch-decisions
```

The global flags are `--cache` (default `data/hf`) and `--seed` (default 1234). `fetch-decisions` also takes `--out` (default `data/decisions`), `--val-frac` (default 0.05), `--teacher` (where `glm_orders` reads the GLM output, default `data/teacher`) and `--list`. `fetch-text` takes `--out`, `--min-chars` (default 64) and `--max-chars` (default 20000). Each command prints a per-source table: raw, invalid, duplicate and per-split counts for decisions, or documents, megabytes and duplicates for text. Download time depends on your connection and was not measured. On the build machine the raw cache `data/hf` takes 1.3 GB (`du -sh data/hf`), and `data/STATS.md` (written by `sky-data stats`, gitignored like the rest of `data/`) lists what comes out of it: 776,247 decision records and 150 MB of pretraining text in 97,660 documents.

The game records from [chapter 04](./04-game-cores-and-oracles) go to `data/games`. Together with `data/text` and `data/decisions`, they are the default inputs of `sky-data train-tokenizer`, covered in [07 A tokenizer with small vocabularies](./07-tokenizer).

## Next

[06 GLM top-up](./06-glm-top-up)
