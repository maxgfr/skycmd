# 12 The size ladder

skycmd trains the same architecture at three sizes, from 0.2M to 6.6M parameters: s1-pico, s1-nano and s1-micro. This chapter puts the configs, parameter budgets, training budgets and measured results side by side and shows where each model's parameters go. Of the three rungs only s1-pico ships on the site; the closed-loop results of the ladder are in [chapter 20](./20-dagger-lessons), and the s2 and s3 architectures that replaced the bigger rungs are in [chapter 19](./19-s2-architecture) and [chapter 21](./21-s3-recursive-tiny). The site's default model today is s3-pico (445,345 parameters, 251 KB ternary).

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
| GGUF size on disk (F16 encoder, F32 head) | 545,280 bytes | 2,765,792 bytes | 16,694,464 bytes |
| test questions in the report | 11,134 | 12,638 | 12,638 |
| test accuracy, all sources | 0.7813 | 0.7226 | 0.7405 |
| test accuracy, games | 0.8806 | 0.7837 | 0.8215 |
| test accuracy, `typed_decisions_bench` | 0.3280 | 0.3975 | 0.3100 |
| test NLL / ECE, all sources | 0.6449 / 0.0738 | 0.6917 / 0.0478 | 0.6712 / 0.0544 |
| native latency, 1 question | 137.6 µs | 974 µs (earlier engine) | about 4,600 µs (earlier engine) |
| browser (WASM) latency, one Flappy decision | 588.6 µs | 3,541 µs (earlier engine) | about 32,600 µs (earlier engine) |

Sources:

- **Parameter counts**: `BENCH.md`, matching the `exact_parameter_counts` test in `crates/sky-train/src/params.rs`.
- **GGUF size**: `site/public/models/manifest.json` for pico. nano and micro are not on the site; their sizes come from converting `models/s1-nano` and `models/s1-micro` with `sky-convert` (default types: encoder matrices in f16, head in f32, see `scripts/export-models.sh`).
- **Accuracy, NLL and ECE**: `runs/s1-{pico,nano,micro}/report.md`, test split, calibrated. The pico report was written before the data was regenerated, so its test split is smaller (11,134 against 12,638 questions; 3,100 Flappy questions against 4,604). Compare the rows with that in mind.
- **Latency**: for pico, the native number is `sky-bench --questions 1` in `BENCH.md` (s3 section, "s3-pico vs s2-nano vs s1-pico") and the WASM number is `wasm_us` in `site/public/bench.json` (`bench()` of the Play page in Node 24: both questions plus the physics step). The nano and micro numbers were measured with the engine as it was before the s3 kernel work (chapter 21) and have not been re-measured since: nano with the same two commands in the `BENCH.md` of that time, micro for this chapter on the same M5 while other jobs ran: `sky-bench /tmp/s1-micro.gguf --questions 1` (best median of 3 rounds: 4,595 µs, while s1-pico measured 148 µs in the same rounds) and `node crates/sky-web/bench/bench-node.mjs --n 500 --repeat 3` (32,556 µs, s1-pico 631 µs in the same run). A micro decision in WASM takes about two 60 Hz frames.

These are the checkpoints straight out of `scripts/train-ladder.sh`, before any DAgger round. The shipped s1-pico is the DAgger round-2 checkpoint (`runs/s1-pico-g2/report.md`): 0.8707 on the game questions of the larger test split, and 200 of 200 Flappy pipes in closed loop.

The surprise is in the games row: nano scores *below* pico on held-out game questions (0.7837 against 0.8806), mostly on Flappy (0.6351 against 0.8245). The pico report uses the older, smaller split, so treat that gap as indicative. nano and micro share a split, and micro (0.8215) beats nano without reaching pico's figure. On these small integer states, more parameters in a fixed wall-clock budget did not buy accuracy.

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
- **Local attention rarely limits game prompts.** A local layer sees `|i − j| ≤ 64`. Sequences up to 65 tokens are covered entirely, and the mean game prompt is 53 to 103 tokens depending on game and vocabulary (`data/STATS.md`, written by `sky-data stats`). The global layers 0 and 3 carry the long-range work on the longer text questions.

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
| measured steps (pretrain / decide) | 21,196 / 29,484 | 13,815 / 14,797 | 4,873 / 7,226 |
| games sampling weight | 0.65 | 0.65 | 0.5 |

The measured counts come from `runs/s1-*/pretrain/last/state.json` and the `training` block of `runs/s1-*/decide/last/rl_agent_config.json`. The estimates were optimistic for the bigger rungs: nano made 14,797 decision steps against ~24k estimated, micro 7,226 against ~9k. micro samples games at 0.5 instead of 0.65, which gives more of each batch to the general decision sets. It did not pay off: micro scores 0.3100 on `typed_decisions_bench`, below nano's 0.3975 and close to pico's 0.3280.

## The vocabulary changes what fits

The vocabulary affects more than the parameter count. With `max_len = 256`, a richer vocabulary leaves more of a long question in the window. From `data/STATS.md` on the build machine (`data/` is gitignored; `sky-data stats` writes the file):

| source | truncated, tok-1024 | tok-2048 | tok-4096 |
|---|---:|---:|---:|
| boolq | 60.6% | 46.1% | 34.6% |
| typed_decisions_bench | 96.8% | 89.4% | 79.0% |
| rescue | 9.4% | 0.0% | 0.0% |
| flappy, dodge | 0.0% | 0.0% | 0.0% |

On the games, every rung sees the full prompt. On open-domain benchmarks, pico reads less of each question than micro does. So a gap between rungs on `typed_decisions_bench` (pico 0.3280, nano 0.3975, micro 0.3100) mixes model capacity with context coverage.

## Choosing a rung

Our games ask small questions about compact integer states, so the ladder is built to answer one question: how small can a decision model be and still play them? Here are the trade-offs as far as they are known:

- **s1-pico** is the s1 model in `site/public/models/manifest.json`. The GGUF file is 545,280 bytes, one question takes 137.6 µs natively (`BENCH.md`) and a Flappy decision 588.6 µs in WASM (`site/public/bench.json`), and after two DAgger rounds it plays Flappy and Rescue at oracle level and Dodge at 0.83 completion against 0.87 ([chapter 20](./20-dagger-lessons)).
- **s1-nano** costs about 5.6 times more parameters and took about 5 times more time per decision on the earlier engine. It is better on `typed_decisions_bench` (0.3975 against 0.3280) and worse on the games. Its first DAgger round reached 200 Flappy pipes but only 0.70 Dodge completion, and its second round regressed ([chapter 20](./20-dagger-lessons)). It is not shipped.
- **s1-micro** is the largest rung: 16.7 MB as an F16 GGUF and no better than pico on games or on the benchmark. It never got DAgger rounds and is not shipped.

The next step for speed and size was not a bigger s1 but a different architecture: s2-nano has 1.5M parameters, a 701,568-byte ternary file, and plays at oracle level ([chapter 19](./19-s2-architecture)). s3-pico, the site's default model, goes further: 445,345 parameters in a 251,104-byte ternary file, all four games at oracle level, and 49.6 µs per Flappy frame in WASM ([chapter 21](./21-s3-recursive-tiny)).

Once the reports exist, compare rungs on the `games` and `decisions` group rows of each `runs/s1-*/report.md` rather than on `all`. The `all` row is weighted by how many test questions each source happens to have.

## Commands

Train one or more rungs. Without arguments the script trains all three in turn:

```bash
scripts/train-ladder.sh pico
scripts/train-ladder.sh nano micro
```

For each rung, the script writes the HF-layout export to `models/s1-<name>/` and the report to `runs/s1-<name>/report.md`. `scripts/export-models.sh` converts every s1 export in `models/` (a directory with `model.safetensors`) to `site/public/models/<name>.gguf` and rewrites `manifest.json` and `bench.json`:

```bash
scripts/export-models.sh
```

The script only knows the s1 layout: the s2 and s3 files on the site (`s3-pico.gguf`, `s2-nano.gguf`, `s2-nano-doom.gguf`) are written by `sky-train s2-export` and copied into `site/public/models/`, and the rewritten `manifest.json` and `bench.json` would list only the s1 models, without the arena and WASM columns of the current files. The next chapter covers what the conversion does: the GGUF layout, the f16 and Q8_0 storage types, and the int8 path of the engine.

## Next

[13 GGUF export and int8](./13-gguf-export-int8)
