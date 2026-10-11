# 10 Pretraining and decision training

Every skycmd-s1 model is trained in two phases. First the encoder learns language from plain FR/EN text by masked-language modelling. Then a decision head is attached and the whole model is trained with cross-entropy on typed decision questions, from datasets and from the games. Both phases are commands of the `sky-train` binary. This chapter covers what each one feeds the model and how the ladder script chains them.

## Overview

| phase | command | trains | data | loss | default peak LR (AdamW / Muon) |
|---|---|---|---|---|---|
| pretrain | `sky-train pretrain` | encoder + MLM head | `data/text` (MLM JSONL) | masked-token CE | 3e-3 / 0.02 |
| decide | `sky-train decide` | encoder + decision head | `data/decisions`, `data/games` | soft-target CE over options | 1e-3 / 0.01 |

Both phases use Muon + AdamW, the time-based WSD schedule, compiled steps and atomic checkpoints from [chapter 09](./09-muon-wsd-training-loop). Build the binary once:

```bash
export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer
export CARGO_TARGET_DIR=$PWD/target/sky-train
cargo build --release -p sky-train
T=$CARGO_TARGET_DIR/release/sky-train
```

## Phase 1: MLM pretraining

### The corpus

`Corpus::load` tokenizes every `{"src": ..., "text": ...}` record under `--text` with the run's tokenizer and appends `[SEP]` after each record. It stops at `--max-tokens` (default 300M). The token stream is cached as raw little-endian `u32` in `<out>/cache/corpus.u32`, keyed by the input file sizes, the tokenizer size and the token cap, so a second run with the same inputs skips tokenization.

For the pico run on the build machine, `runs/s1-pico/pretrain/cache/corpus.json` records 97,660 text records, and `corpus.u32` holds about 65M tokens. `data/STATS.md` lists about 150 MB of text, so that is roughly 2.3 bytes per token with the 1024-entry vocabulary.

### Packed rows

The stream is cut into fixed chunks of `seq − 2` tokens, each wrapped as `[CLS] chunk [SEP]`. Consecutive records share a chunk: nothing is padded and no token is wasted. The last 1% of the chunks, capped at `--val-batches × --batch` rows, is held out for validation. Training order is a fresh permutation per epoch, derived from `(seed, epoch)`. The data `Cursor` is saved in `state.json`, so a resumed run continues the exact same stream.

### 30% masking with the 80/10/10 rule

`mask_batch` picks 30% of the non-special tokens of each row. Of those, 80% become `[MASK]`, 10% become a random non-special token and 10% stay unchanged:

`crates/sky-train/src/mlm.rs`
```rust
        let n_mask = ((cand.len() as f64 * ratio).round() as usize).max(usize::from(!cand.is_empty()));
        // Partial Fisher–Yates: the first n_mask candidates are the chosen ones.
        for s in 0..n_mask {
            let k = s + rng.below(cand.len() - s);
            cand.swap(s, k);
        }
        let mut chosen = cand[..n_mask].to_vec();
        chosen.sort_unstable();
        for j in chosen {
            let flat = i * t + j;
            labels.push(r[j] as i32);
            positions.push(flat as i32);
            let u = rng.f64();
            if u < 0.8 {
                ids[flat] = MASK_ID as i32;
            } else if u < 0.9 {
                ids[flat] = (N_SPECIAL as usize + rng.below(vocab - N_SPECIAL as usize)) as i32;
            }
        }
```

The 30% rate follows ModernBERT rather than the original BERT's 15%, which doubles the number of predictions per row. Only the chosen positions are scored, as a flat index list padded to a multiple of 256 entries with zero weight, which keeps the compiled graph's shapes stable.

### The MLM head (pretraining only)

The MLM head is a dense layer, GELU and a LayerNorm, followed by a decoder that is **tied** to the token embedding table and has its own bias:

`crates/sky-train/src/model.rs`
```rust
pub fn mlm_logits(p: &ParamMap, cfg: &ModelConfig, enc: &Array, positions: &Array) -> R<Array> {
    let p = &*compute_params(p, cfg)?;
    let d = enc.dim(2);
    let n = enc.dim(0) * enc.dim(1);
    let rows = enc.reshape(&[n, d])?.take_axis(positions, 0)?;
    let h = gelu_erf(&linear(&rows, w(p, "mlm.dense.weight")?, None)?)?;
    let h = layer_norm(&h, w(p, "mlm.norm.weight")?, None, cfg.norm_eps)?;
    to_f32(linear(
        &h,
        w(p, "encoder.embeddings.tok_embeddings.weight")?,
        Some(w(p, "mlm.decoder.bias")?),
    )?)
}
```

Thanks to the tied decoder, the head adds only `d² + d + V` parameters: 5,184 for pico, 18,560 for nano and 69,888 for micro. Its tensors are named `mlm.*` and are never exported.

### Running it

```bash
$T pretrain --config configs/s1-pico.json --tokenizer data/tok/tok-1024.json \
  --text data/text --minutes 8 --out runs/s1-pico/pretrain
```

Other flags: `--batch` (128), `--seq` (128), `--seed` (0), `--max-tokens`, `--log-every` (20), `--eval-every` (500), `--checkpoint-minutes` (10), `--val-batches` (16), `--precision bf16|f32`, `--resume`, and the optimizer flags `--adam-lr`, `--muon-lr`, `--weight-decay`, `--clip-norm`. The tokenizer must not have more ids than the config's `vocab_size`.

Training writes one JSON line per log window to `<out>/log.jsonl` (`loss`, `grad_norm`, `lr`, `muon_lr`, `tok_s`, `elapsed_min`, `epoch`), plus a `val_loss` line every `--eval-every` steps. For the current pico run (`runs/s1-pico/pretrain/`, which may be retrained):

- 21,196 steps in the 8-minute budget, at 128 × 128 tokens per step;
- validation loss from 5.37 at step 500 down to a final 3.29;
- the cursor ended in epoch 5, so the 65M-token corpus was seen a little over five times.

## Phase 2: decision training

### Loading and validating records

`decision::load` reads every `*.jsonl` under each `--records` path. The path's directory name becomes the record **group** (`decisions`, `games`). The loader tokenizes the head, options and state of every record in large parallel batches and keeps a record only if:

- `t` is `choice`, `score` or `noul`;
- `opts` is non-empty and `target` has one non-negative finite entry per option, with a positive sum (the target is then normalised to sum to 1);
- every option marker fits in `max_len` after truncation.

The last rule matches Laya's `encode_record`: "options did not fit; skip rather than train on a truncated answer space". Records go to train, val or test according to their `split` field. Test-only sources such as `typed_decisions_bench` (`LocalLLaMA/typed-decisions`) never reach the training set.

### Sampling, shuffling and padding

Each batch draws its rows independently. A group is picked with the source weights, then an item uniformly within that group. The item's options are shuffled before the sequence is built:

`crates/sky-train/src/decision.rs`
```rust
pub fn sample_batch(data: &DecisionData, weights: &[f64], batch: usize, rng: &mut Rng, agent: &AgentConfig) -> DecisionBatch {
    let w: Vec<f64> = weights
        .iter()
        .zip(&data.train)
        .map(|(w, items)| if items.is_empty() { 0.0 } else { *w })
        .collect();
    let rows: Vec<Row> = (0..batch)
        .map(|_| {
            let g = rng.weighted(&w);
            let item = &data.train[g][rng.below(data.train[g].len())];
            let order = augment_order(item, rng);
            let r = row(item, &order, agent);
            if r.markers.len() == item.k() {
                r
            } else {
                row(item, &(0..item.k()).collect::<Vec<_>>(), agent)
            }
        })
        .collect();
    make_batch(&rows, 16)
}
```

Shuffling addresses a weakness that `docs/research.md` cites (papers 2610.02586 and 2609.26758): decision heads latch onto option positions and labels. The target is permuted together with the options. `score` questions are never shuffled, because their options are ordinal levels (`level 0`, `level 1`, ...). Option *renaming* is done earlier, when `sky-games` generates its questions.

`make_batch` pads every row to the longest sequence, rounded up to a multiple of 16 tokens, and pads the option axis to the largest K in the batch. `marker_mask` marks the real options.

### Source weights

`--weights` takes `name=w` pairs keyed by group name. Without it, `games` gets 0.5 and the other groups share the other 0.5:

`crates/sky-train/src/train.rs`
```rust
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
```

The weights are sampling probabilities, not loss weights. Raising `games` means the model sees more game questions per step, and the rest of each batch still comes from the general decision sets.

### The loss: cross-entropy on soft targets

The decision loss is the cross-entropy between the masked marker logits and the target distribution:

`crates/sky-train/src/model.rs`
```rust
/// Marker logits with invalid options pushed to -1e4.
pub fn masked_logits(logits: &Array, marker_mask: &Array) -> R<Array> {
    ops::select(marker_mask, logits, Array::from_f32(-1e4))
}
```

`crates/sky-train/src/model.rs`
```rust
pub fn soft_cross_entropy(masked_logits: &Array, target: &Array) -> R<Array> {
    let lse = masked_logits.logsumexp_axis(-1, true)?;
    let logp = masked_logits.subtract(&lse)?;
    target.multiply(&logp)?.sum_axis(-1, None)?.mean(None)?.negative()
}
```

Soft targets matter for us. `pngwn/typed-decisions` ships soft `gold_probs`, and the game oracles produce graded targets such as `[0.43, 0.01, 0.55, 0.01]` when two manoeuvres are both reasonable. The reference Laya runtime also has RL-style proper rewards (`proper_reward` in `rl_common.py`), but the research notes (paper 2610.02486) report that the published RLCD recipe trails plain cross-entropy by 2.5–3 points, and that CE followed by temperature scaling calibrates just as well. skycmd therefore uses CE only and calibrates afterwards (chapter 11).

### Starting from the pretrained encoder

`--init` points at a pretraining checkpoint. A fresh decision parameter map is initialised first, then every tensor whose name exists in the checkpoint is copied over with a shape check. The run fails if any encoder tensor is missing. The MLM head is simply left behind, and the decision head starts from scratch:

`crates/sky-train/src/train.rs`
```rust
        let mut p = params::init(&cfg, Parts::DECISION, a.seed)?;
        if let Some(init) = &a.init {
            let src = params::load(&init.join("model.safetensors"))?;
            let n = params::load_matching(&mut p, &src)?;
            let n_enc = p.keys().filter(|k| k.starts_with("encoder.")).count();
            if n < n_enc {
                bail!("{} provides only {n} of the {n_enc} encoder tensors", init.display());
            }
            println!("initialized {n} tensors from {}", init.display());
        }
```

When `--tokenizer` is omitted, `decide` reuses `<init>/tokenizer.json`, the copy saved with the pretraining checkpoint.

### Running it

```bash
$T decide --config configs/s1-pico.json --init runs/s1-pico/pretrain/last \
  --tokenizer data/tok/tok-1024.json --records data/decisions,data/games \
  --weights games=0.65,decisions=0.35 --minutes 12 --out runs/s1-pico/decide
```

Other flags: `--batch` (64), `--val-max` (4000 validation questions per evaluation), `--eval-every` (500), `--log-every`, `--checkpoint-minutes`, `--seed`, `--precision`, `--resume` and the optimizer flags. Every evaluation runs the model on a fixed subset of the val split with the identity option order. It logs accuracy, NLL, Brier and ECE per group at T = 1. Each checkpoint also writes an `rl_agent_config.json` with the uncalibrated temperatures.

The current pico run (`runs/s1-pico/decide/`, which may be retrained) took 29,484 steps in 12 minutes. Its last validation line reads `all` 0.6525 accuracy over 4000 questions: 0.881 on `games` and 0.449 on `decisions`. The gap between the two groups is the main story of chapter 11.

## The ladder script

`scripts/train-ladder.sh` chains the five steps per model, with per-rung budgets:

`scripts/train-ladder.sh`
```bash
# name  vocab  pretrain-min  decide-min  game-weight
ladder() {
  case "$1" in
    pico)  echo "1024 8 12 0.65" ;;
    nano)  echo "2048 12 20 0.65" ;;
    micro) echo "4096 15 25 0.5" ;;
  esac
}
```

`scripts/train-ladder.sh`
```bash
  $T pretrain --config "$cfg" --tokenizer "$tok" --text data/text --minutes "$pre" --out "$run/pretrain"
  $T decide --config "$cfg" --init "$run/pretrain/last" --tokenizer "$tok" \
    --records data/decisions,data/games --weights "games=$gw,decisions=$(echo "1 - $gw" | bc -l)" \
    --minutes "$dec" --out "$run/decide"
  $T calibrate --run "$run/decide/last" --records data/decisions,data/games
  $T export --run "$run" --out "models/s1-$name"
  $T eval --run "$run" --records data/decisions,data/games
```

Pass the rung names explicitly:

```bash
scripts/train-ladder.sh pico
scripts/train-ladder.sh nano micro
```

The loop is written as `for name in "${@:-pico nano micro}"`. With no arguments, the quoted default expands to the single word `pico nano micro`, which matches no rung. Always name the rungs you want.

The script expects the tokenizers from [chapter 07](./07-tokenizer) in `data/tok/` and the data from the earlier chapters in `data/text`, `data/decisions` and `data/games`. Total time is the sum of the budgets plus loading, calibration and evaluation: 20 minutes of training for pico, 32 for nano and 40 for micro.

## Next

[11 Calibration and evaluation](./11-calibration-and-evaluation)
