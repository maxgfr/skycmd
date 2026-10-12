# 07 A tokenizer with small vocabularies

skycmd trains its own byte-level BPE tokenizers with 1024, 2048 and 4096 entries, one per rung of the size ladder. This chapter covers how `sky-data` builds the training corpus and trains them, why small integers get a vocabulary prior, the `[NUM]` variant used by the s2 and s3 models (including the default site model, s3-pico, [chapter 21](./21-s3-recursive-tiny)), and how `sky-infer` reproduces the same encodings in plain Rust for the browser.

## Why the vocabulary is the first size decision

An encoder's embedding table holds `vocab_size × hidden_size` parameters. Large vocabularies dominate small models. `docs/research.md` notes that about 12.9M of the 17M parameters of `oscar-1-17m` sit in its 50k-token embedding table. skycmd does the opposite and keeps the vocabulary close to the size of the transformer:

| model | vocab | d | embedding table | encoder total |
|---|---:|---:|---:|---:|
| s1-pico | 1024 | 64 | 65,536 | 147,776 |
| s1-nano | 2048 | 128 | 262,144 | 918,656 |
| s1-micro | 4096 | 256 | 1,048,576 | 4,984,064 |

The encoder totals come from the `exact_parameter_counts` test in `crates/sky-train/src/params.rs`. The embedding column is just `vocab × d`.

A small vocabulary costs sequence length, because text needs more tokens. Our inputs are mostly short questions, option labels and compact game states such as `alt 12 vy -3`, so the trade is worth it as long as the frequent words and the small integers stay single tokens. The rest of the chapter shows how we make sure of that.

## The contract: byte-level BPE with fixed special ids

`docs/SPEC.md` fixes the format: GPT-2 style byte-level BPE (ByteLevel pre-tokenizer, `add_prefix_space=false`, GPT-2 split regex), trained with Hugging Face `tokenizers` and saved as `tokenizer.json`. The special tokens have fixed ids, so every crate can hard-code them:

`crates/sky-data/src/tok.rs`
```rust
/// Special tokens in id order: `[PAD]=0 [UNK]=1 [CLS]=2 [SEP]=3 [MASK]=4`.
pub const SPECIALS: [&str; 5] = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"];
pub const PAD_ID: u32 = 0;
pub const CLS_ID: u32 = 2;
pub const SEP_ID: u32 = 3;
pub const MASK_ID: u32 = 4;

/// Default sequence budgets of our models (SPEC).
pub const MAX_LEN: usize = 256;
pub const HEAD_MAX_LEN: usize = 96;
```

`sky-train` checks the same ids again every time it loads a tokenizer (`Tok::load` in `crates/sky-train/src/sequence.rs`). A tokenizer trained with different ids is rejected before any training step runs.

## The training corpus: text as the model will see it

The BPE merges should fit the strings the model actually reads. `build_corpus` therefore does not feed raw JSONL to the trainer. For decision records it emits the same pieces that the prompt builder tokenizes separately: the head `"{t} question: {ins}"`, each option with its leading space, and the state.

`crates/sky-data/src/tok.rs`
```rust
pub fn decision_segments(r: &DecisionRecord) -> Vec<String> {
    let mut out = Vec::with_capacity(r.opts.len() + 2);
    out.push(format!("{} question: {}", r.t, r.ins.replace("[MASK]", " ")));
    for o in &r.opts {
        out.push(format!(" {}", o.replace("[MASK]", " ")));
    }
    if !r.state.is_empty() {
        out.push(r.state.replace("[MASK]", " "));
    }
    out
}
```

Any literal `[MASK]` in a record is replaced by a space, as the runtime does, so the marker token can never come out of user text.

The corpus mixes three kinds of input, from the `--inputs` directories (default `data/text,data/decisions,data/games`):

- **MLM text** (`{"text": ...}` records). Every text file gets the same share of the `--max-text-mb` budget (default 120 MB). Lines are kept or dropped with a hash of their bytes, so the sample is the same on every run.
- **Decision records**: at most `--max-decisions-per-file` records per file (default 30,000), shuffled with a seed derived from the file name.
- **Game records**: any path with a `games` component is repeated `--game-weight` times (default 4). The game vocabulary (`alt`, `vy`, `gw`, `abort`, ...) is tiny next to the text, and this weight makes sure it still earns merges.

## Making small integers single tokens

Game states are integer-valued (SPEC: values are binned and clamped "so the BPE learns them as single tokens"). Frequency alone does not get us there. The GPT-2 pre-tokenizer splits `vy -12` into `vy`, ` -` and `12`, so each integer appears in two forms, with a leading space and bare. Without help, BPE spends its budget on common words first. `int_prior_segments` adds synthetic segments that repeat every integer in both forms:

`crates/sky-data/src/tok.rs`
```rust
pub fn int_prior_segments(max: usize, reps: usize) -> Vec<String> {
    const CHUNK: usize = 1000;
    let mut out = Vec::new();
    for n in 0..max {
        let mut left = reps;
        while left > 0 {
            let k = left.min(CHUNK);
            out.push(format!(" {n}").repeat(k));
            out.push(format!(" -{n}").repeat(k));
            left -= k;
        }
    }
    out
}

/// Repetitions per integer form: about one per 4000 corpus bytes (≈ one per 800 words).
pub fn int_prior_reps(corpus_bytes: u64) -> usize {
    ((corpus_bytes / 4000) as usize).max(1000)
}
```

The range defaults to `vocab / 32` (`--int-prior`; `0` disables it), so it covers `0..32` for pico, `0..64` for nano and `0..128` for micro. The tokenizers in `data/tok/` on the build machine follow that pattern: ` 0` to ` 31` are all vocabulary entries of `tok-1024.json`, and ` 32` is the first missing one. The same holds at 64 for `tok-2048.json` and at 128 for `tok-4096.json`.

Take this dodge state from `data/games/dodge.jsonl`:

```text
d 35 v 11 dz 2 hr 1 gw -2 dy 0 cl 3
```

The pre-tokenizer cuts it into `d`, ` 35`, ` v`, ` 11`, ` dz`, ` 2`, ` hr`, ` 1`, ` gw`, ` -`, `2`, ` dy`, ` 0`, ` cl`, ` 3`. Every one of these pieces is a single entry of the 4096 vocabulary. In the 1024 vocabulary every piece except ` 35` is an entry. Since 35 lies above the integer prior, pico spells it with more than one token. This is one reason the game generators clamp their values to small ranges.

## Training with HF tokenizers

`train_tokenizer` builds the same pipeline as oscar's `tokenizer.json`: an NFC normalizer, a ByteLevel pre-tokenizer and decoder, and a `[CLS] $A [SEP]` template.

`crates/sky-data/src/tok.rs`
```rust
pub fn train_tokenizer(segments: &[String], vocab: usize, min_frequency: u64) -> Result<Tokenizer> {
    train_tokenizer_with(segments, vocab, min_frequency, &SPECIALS)
}
```

`train_tokenizer_with` takes the list of special tokens, so the s2 variant below reuses it:

`crates/sky-data/src/tok.rs`
```rust
fn train_tokenizer_with(
    segments: &[String],
    vocab: usize,
    min_frequency: u64,
    specials_list: &[&str],
) -> Result<Tokenizer> {
    let specials: Vec<AddedToken> = specials_list
        .iter()
        .map(|s| AddedToken::from(*s, true))
        .collect();
    let mut trainer = BpeTrainerBuilder::new()
        .vocab_size(vocab)
        .min_frequency(min_frequency)
        .show_progress(false)
        .special_tokens(specials)
        .initial_alphabet(ByteLevel::alphabet().into_iter().collect())
        .build();
    let template = TemplateProcessing::builder()
        .try_single("[CLS] $A [SEP]")
        .map_err(|e| anyhow!(e))?
        .try_pair("[CLS] $A [SEP] $B:0 [SEP]:0")
        .map_err(|e| anyhow!(e))?
        .special_tokens(vec![
            ("[CLS]".to_string(), CLS_ID),
            ("[SEP]".to_string(), SEP_ID),
        ])
        .build()?;
```

The 256-symbol byte alphabet is always present, so no input can fall outside the vocabulary and `[UNK]` is never emitted for real text. A vocabulary of `V` entries therefore holds 5 specials, 256 byte symbols and `V − 261` merges: 763, 1787 and 3835 merges for the three files in `data/tok/`. After training, the function checks that each special token landed on its fixed id. It refuses to return a tokenizer otherwise.

### Commands

Build `sky-data` in its own target directory and train the three vocabularies. Without `--out`, the output path is `data/tok/tok-<vocab>.json`.

```bash
export CARGO_TARGET_DIR=$PWD/target/sky-data
cargo build --release -p sky-data
for v in 1024 2048 4096; do
  $CARGO_TARGET_DIR/release/sky-data train-tokenizer --vocab $v
done
```

The other flags are `--inputs`, `--game-weight`, `--max-text-mb`, `--max-decisions-per-file`, `--min-frequency` (default 2), `--int-prior` and `--num` (next section), plus the global `--seed` (default 1234). A vocabulary below 300 is rejected.

## The s2 and s3 tokenizers: one `[NUM]` per number

The s2 architecture ([chapter 19](./19-s2-architecture)) and s3 ([chapter 21](./21-s3-recursive-tiny)) do not spell numbers with BPE pieces at all. `--num` trains an s2 tokenizer (SPEC-S2 §2): before BPE, every number matching `-?\d+(?:\.\d+)?` that does not follow an ASCII letter is replaced by the special token `[NUM]`, and its value travels next to the token ids as a continuous feature (`crates/sky-data/src/num.rs`, mirrored bit for bit by `sky-train` and `sky-infer`). The specials become `[PAD] [UNK] [CLS] [SEP] [MASK] [NUM]`, with `[NUM]` fixed at id 5 (`SPECIALS_S2`). The corpus segments are split around `[NUM]`, so no merge is ever learned across it, and no integer prior is added because digits never reach the BPE. `train_tokenizer_s2` refuses a result in which `[NUM]` is not matched as a special token.

```bash
for v in 1024 2048 4096; do
  $CARGO_TARGET_DIR/release/sky-data train-tokenizer --vocab $v --num   # -> data/tok/tok-s2-<vocab>.json
done
```

s2-pico and s3-pico use `tok-s2-1024.json`, s2-nano uses `tok-s2-2048.json` (`scripts/train-s2.sh`, `scripts/train-s3.sh`). The overnight experiments also trained `tok-s2-512.json` with `--num --vocab 512` for the `s3-femto-v512` candidate ([`docs/night-experiments.md`](https://github.com/maxgfr/skycmd/blob/main/docs/night-experiments.md)). s3 uses the same tokenizers and goes one step further: with `fuse_numbers`, each `[NUM]` is removed right after tokenization and its value moves onto the token before it, so a pair such as `vy -1` costs one token instead of two (`crates/sky-infer/src/s2/prompt.rs`, chapter 21).

## Reading the sequence-length table

After saving, `train-tokenizer` runs a Rust port of `rl_common.build_sequence` on up to 2000 records per decision file and prints the mean tokens per sequence, the untruncated length, and how often truncation happens. `sky-data stats` collects the same tables for every tokenizer in `data/tok` into `data/STATS.md`:

```bash
$CARGO_TARGET_DIR/release/sky-data stats
```

Here are a few rows from `data/STATS.md` on the build machine (`max_len` 256, `head_max_len` 96). These numbers will move whenever the data or the tokenizers are regenerated.

| source | mean tokens, 1024 | 2048 | 4096 | truncated, 1024 | 4096 |
|---|---:|---:|---:|---:|---:|
| flappy | 66.0 | 54.7 | 53.0 | 0.0% | 0.0% |
| dodge | 73.7 | 60.0 | 58.7 | 0.0% | 0.0% |
| rescue | 102.5 | 89.7 | 86.9 | 9.4% | 0.0% |
| boolq | 230.7 | 218.2 | 204.8 | 60.6% | 34.6% |
| typed_decisions_bench | 255.3 | 252.4 | 247.0 | 96.8% | 79.0% |

Game prompts fit easily at every size. Long-context benchmark questions get truncated much more with the 1024 vocabulary, which matters when we read the evaluation reports in [chapter 11](./11-calibration-and-evaluation). The column "options cut off" stays at 0.0% for every source except `tasksource_jev` (1.9% at every vocabulary size), the source with the most options per question (8.15 on average, `data/STATS.md`). Elsewhere the head budget protects the options (chapter 08).

## The same tokenizer at inference: `sky-infer`

The browser cannot run HF `tokenizers`, so `sky-infer` ships its own implementation. It needs no regex engine, and its only external dependency is `unicode-normalization` (for NFC). The module documentation spells out the pipeline it reproduces:

`crates/sky-infer/src/tokenizer/mod.rs`
```rust
//! GPT-2 style byte-level BPE tokenizer that reproduces Hugging Face `tokenizers` encodings
//! for ByteLevel BPE models.
//!
//! Pipeline (same order as `tokenizers`):
//! 1. added tokens with `normalized: false` (special tokens) are matched on the raw text
//!    (leftmost-longest, with `lstrip` / `rstrip` / `single_word`);
//! 2. the remaining pieces are normalized (optional NFC) and the added tokens with
//!    `normalized: true` are matched on them;
//! 3. every remaining piece goes through the ByteLevel pre-tokenizer (optional prefix space,
//!    GPT-2 split regex, bytes -> unicode) and the ranked BPE merges.
```

### A hand-written GPT-2 split

The GPT-2 regex `'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+` uses a lookahead, which the Rust `regex` crate does not support. `pretokenize.rs` implements it as a direct scanner with Unicode letter and number tables. Its tests pin down the cases that matter for game states:

`crates/sky-infer/src/tokenizer/pretokenize.rs`
```rust
    #[test]
    fn gpt2_regex_cases() {
        assert_eq!(split("Hello world"), vec!["Hello", " world"]);
        assert_eq!(split("I'm don't"), vec!["I", "'m", " don", "'t"]);
        assert_eq!(split("a   b"), vec!["a", "  ", " b"]);
        assert_eq!(split("x -3 12"), vec!["x", " -", "3", " 12"]);
        assert_eq!(split("end  "), vec!["end", "  "]);
        assert_eq!(split("a\n\nb"), vec!["a", "\n", "\n", "b"]);
```

The `"x -3 12"` case is the reason the integer prior adds both ` n` and bare `n`: after ` -`, the digits stand alone.

### Ranked merges with a heap

`bpe_word_uncached` keeps the symbols of a word in a linked list and pushes every adjacent pair that has a merge into a min-heap keyed by (rank, position). Each pop applies the lowest-ranked merge and pushes the new neighbour pairs. Stale entries are detected by checking the merge result again. This gives the same result as the reference algorithm without rescanning the word after every merge.

Game states repeat the same short words frame after frame, so `encode_into` accepts an optional `BpeCache` that maps a pre-token's bytes to its ids. The engine in `crates/sky-infer/src/engine.rs` keeps one alive between calls.

### Parity tests

`crates/sky-infer/tests/tokenizer.rs` compares encodings against fixtures produced by HF `tokenizers` (`tests/fixtures/make_fixtures.py`). It also checks oscar's 50k tokenizer when `models-src/oscar` is present (the test is skipped otherwise), and it round-trips both tokenizers through GGUF metadata the way the converter writes them:

```bash
export CARGO_TARGET_DIR=$PWD/target/sky-infer
cargo test -p sky-infer --test tokenizer
```

The tokenizer is now stored with every checkpoint as `tokenizer.json`, and the next chapter feeds its ids into the encoder.

## Next

[08 The encoder and the decision head](./08-encoder-and-decision-head)
