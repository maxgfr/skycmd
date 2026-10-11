# 12 The size ladder

skycmd trains the same architecture at three sizes, from 0.2M to 6.6M parameters: s1-pico, s1-nano and s1-micro. This chapter puts the configs, parameter budgets, training budgets and measured results side by side and shows where each model's parameters go. Values not measured yet are marked TBD.

## The three configs

The ladder is defined in `docs/SPEC.md` and implemented by `configs/s1-{pico,nano,micro}.json`. The `ladder_configs_match_spec_table` test in `crates/sky-train/src/config.rs` checks the files against the SPEC table:

`crates/sky-train/src/config.rs`
```rust
        // name, d, L, heads, intermediate, head_layers, vocab
        let table = [
            ("s1-pico", 64, 2, 1, 128, 1, 1024),
            ("s1-nano", 128, 4, 2, 256, 1, 2048),
            ("s1-micro", 256, 6, 4, 512, 2, 4096),
        ];
```

All three share `global_attn_every_n_layers = 3`, `local_attention = 128`, RoPE θ 160000 (global) / 10000 (local), `max_len = 256` and `head_max_len = 96`.

| | s1-pico | s1-nano | s1-micro |
|---|---:|---:|---:|
| hidden size d | 64 | 128 | 256 |
| encoder layers L | 2 | 4 | 6 |
| attention heads (64-dim) | 1 | 2 | 4 |
| GeGLU intermediate | 128 | 256 | 512 |
| global layers | 0 | 0, 3 | 0, 3 |
| decision head layers | 1 | 1 | 2 |
| vocabulary | 1024 | 2048 | 4096 |
| tokenizer | `tok-1024.json` | `tok-2048.json` | `tok-4096.json` |
| params, decision model | 202,305 | 1,134,209 | 6,630,913 |
| params, MLM pretraining | 152,960 | 937,216 | 5,053,952 |
| GGUF size on disk | 545,280 bytes | TBD | TBD |
| test accuracy, all sources | 0.7813 | TBD | TBD |
| test accuracy, games | 0.8806 | TBD | TBD |
| test NLL / ECE, all sources | 0.6449 / 0.0738 | TBD | TBD |
| native latency per decision | 174.5 µs | TBD | TBD |
| browser (wasm) latency | TBD | TBD | TBD |

Sources:

- **Parameter counts**: `BENCH.md`, matching the `exact_parameter_counts` test in `crates/sky-train/src/params.rs`.
- **GGUF size**: `site/public/models/manifest.json`. The file stores encoder matrices in f16 and the head in f32 (see `scripts/export-models.sh`).
- **Accuracy, NLL and ECE**: `runs/s1-pico/report.md`, test split, calibrated.
- **Native latency**: the `sky-bench` median in `site/public/bench.json`, measured on the build machine.

All pico values describe the current checkpoint and change if it is retrained. The nano and micro columns stay TBD until their runs finish and their reports and GGUF files exist.

Each rung multiplies the parameter count by about 5.6 (pico to nano) and 5.8 (nano to micro).

## Where the parameters go

Splitting each model by component, using the tensor shapes of [chapter 08](./08-encoder-and-decision-head):

| component | s1-pico | s1-nano | s1-micro |
|---|---:|---:|---:|
| token embeddings + norm | 65,600 (32%) | 262,272 (23%) | 1,048,832 (16%) |
| encoder layers | 82,112 (41%) | 656,256 (58%) | 3,934,976 (59%) |
| final norm | 64 | 128 | 256 |
| decision head + `type_emb` + scorer | 54,529 (27%) | 215,553 (19%) | 1,646,849 (25%) |
| **total** | **202,305** | **1,134,209** | **6,630,913** |

Three observations:

- **The vocabulary grows with the model, not ahead of it.** Doubling `d` and the vocabulary together quadruples the embedding table, while the encoder grows faster because it also gains layers. The embedding share therefore falls from 32% to 16%. In `oscar-1-17m`, about 12.9M of 17M parameters (76%) sit in the embeddings (`docs/research.md`).
- **The head is a big fixed cost.** One head layer has a `4d` ReLU feed-forward plus full attention, about `12d²` weights, more than the `10d²` of an encoder layer at `intermediate = 2d`. micro has two head layers, which is why its head share goes back up to 25%.
- **Local attention rarely limits game prompts.** A local layer sees `|i − j| ≤ 64`. Sequences up to 65 tokens are covered entirely, and the mean game prompt is 48 to 103 tokens depending on game and vocabulary (`data/STATS.md`). The global layers 0 and 3 carry the long-range work on the longer text questions.

## Training budgets

`scripts/train-ladder.sh` gives each rung a wall-clock budget and a share of game questions in the decision batches:

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

The WSD schedule runs on time (chapter 09), so each rung gets a complete warmup, plateau and cooldown whatever its speed. Larger models still get far fewer updates. Dividing the budgets by the bf16 step times in `BENCH.md` gives rough step counts. They are estimates only: real batches have different lengths, and validation passes take time.

| | s1-pico | s1-nano | s1-micro |
|---|---:|---:|---:|
| pretrain budget | 8 min | 12 min | 15 min |
| MLM step (bf16, 128 × 128) | 24.3 ms | 55.8 ms | 172.7 ms |
| estimated pretrain steps | ~20k | ~13k | ~5k |
| decide budget | 12 min | 20 min | 25 min |
| decision step (bf16, 64 × 192) | 20.5 ms | 50.6 ms | 173.4 ms |
| estimated decide steps | ~35k | ~24k | ~9k |
| measured steps (pretrain / decide) | 21,196 / 29,484 | TBD | TBD |
| games sampling weight | 0.65 | 0.65 | 0.5 |

The measured pico counts come from `runs/s1-pico/pretrain/last/state.json` and `runs/s1-pico/decide/last/rl_agent_config.json`. micro samples games at 0.5 instead of 0.65, which gives more of each batch to the general decision sets. Its open-domain results are TBD.

## The vocabulary changes what fits

The vocabulary affects more than the parameter count. With `max_len = 256`, a richer vocabulary leaves more of a long question in the window. From `data/STATS.md` on the build machine:

| source | truncated, tok-1024 | tok-2048 | tok-4096 |
|---|---:|---:|---:|
| boolq | 60.6% | 46.1% | 34.6% |
| typed_decisions_bench | 96.8% | 89.4% | 79.0% |
| rescue | 9.4% | 0.0% | 0.0% |
| flappy, dodge | 0.0% | 0.0% | 0.0% |

On the games, every rung sees the full prompt. On open-domain benchmarks, pico reads less of each question than micro does. So a gap between rungs on `typed_decisions_bench` (pico scores 0.3280 today, nano and micro TBD) mixes model capacity with context coverage.

## Choosing a rung

Our games ask small questions about compact integer states, so the ladder is built to answer one question: how small can a decision model be and still play them? Here are the trade-offs as far as they are known:

- **s1-pico** is the only model in `site/public/models/manifest.json` today. The GGUF file is about half a megabyte, the measured native latency is under 200 microseconds per decision (`site/public/bench.json`, browser TBD), and it reaches 0.8806 accuracy on held-out game questions.
- **s1-nano** costs about 5.6 times more parameters. Its gains on games and open-domain questions are TBD.
- **s1-micro** is the largest rung. Its accuracy, size on disk and latency are TBD.

Once the reports exist, compare rungs on the `games` and `decisions` group rows of each `runs/s1-*/report.md` rather than on `all`. The `all` row is weighted by how many test questions each source happens to have.

## Commands

Train one or more rungs. Name them explicitly, because the script's default argument does not split into separate names (see [chapter 10](./10-pretraining-and-decision-training)):

```bash
scripts/train-ladder.sh pico
scripts/train-ladder.sh nano micro
```

For each rung, the script writes the HF-layout export to `models/s1-<name>/` and the report to `runs/s1-<name>/report.md`. To regenerate the site's GGUF files, `manifest.json` and `bench.json` from everything in `models/`:

```bash
scripts/export-models.sh
```

The next chapter covers what that conversion does: the GGUF layout, the f16 and Q8_0 storage types, and the int8 path of the engine.

## Next

[13 GGUF export and int8](./13-gguf-export-int8)
