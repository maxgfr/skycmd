# 21 · s3: a 251 KB model that plays four games

s3 is the result of one night of research and experiments, aimed at the smallest and fastest model that still plays every game as well as the oracle. It is s2 (chapter 19) with three changes, each taken from recent work. The spec is in [`docs/SPEC-S3.md`](https://github.com/maxgfr/skycmd/blob/main/docs/SPEC-S3.md), and the research notes with their sources are in [`docs/research-night.md`](https://github.com/maxgfr/skycmd/blob/main/docs/research-night.md).

s3-pico is the model the site loads by default. Its GGUF keeps `general.architecture = "skycmd-s2"`, so like s2 it runs only in the skycmd engine (the browser module, the npm package, the Python wheel, the `skycmd` CLI and server), not in llama.cpp or wllama ([Install](/install/)).

## The result

| model | params | file | flappy | rescue | dodge (oracle 0.867) | doom | dodge vetoes | per Flappy frame, WASM |
|---|---:|---:|---|---|---|---|---:|---:|
| **s3-pico** | **445,345** | **251 KB** ternary | 200/200, 0 crashes | 100 % | **0.867** | 100 % levels, kills = oracle | **14** | **≈ 50 µs** |
| s2-nano | 1,532,033 | 702 KB ternary | 200/200 | 100 % | 0.867 | (doom fine-tune) | 25 | ≈ 409 µs |
| s1-pico | 202,305 | 545 KB f16 | 200/200 | 100 % | 0.833 | — | 1,026 | ≈ 589 µs |

- **Games**: 30 seeded episodes per game in `sky-arena`, with the shipped GGUF files (`runs/ship-*.arena.json`). s3-pico also completes 100 % of Doom-like levels and kills as many monsters as the oracle (0.77), with no deaths.
- **Latency**: median per Flappy frame (both questions plus physics) in the browser WASM module under Node, on the build machine (`crates/sky-web/bench/bench-node.mjs`). Natively, `sky-arena` measures a median of 21 µs per Flappy decision for s3-pico.
- **Held-out accuracy** (test split, Doom excluded): s3-pico 0.543, s2-nano 0.544, s1-pico 0.479 (BENCH.md, s3 section). s3-pico matches s2-nano with 3.4× fewer parameters and a 2.8× smaller file.

## Change 1: one token per measured value

FT-Transformer turns each column of a table into one token, and FoNE encodes a number as a single token with Fourier features. In s2, `vy -1` costs two tokens: `vy` and `[NUM]`. With `fuse_numbers`, the value is added onto the key's token: `e = embed(vy) + num_proj(φ(-1))`. The Flappy state used by the benchmarks (seven `key value` pairs) goes from 16 tokens to 9, `[CLS]` and `[SEP]` included, and a Doom state from 38 to 26 (`BENCH.md`). Attention cost is quadratic in that length, and everything else is linear.

## Change 2: depth from a loop, not from parameters

TRM reasons with a single 2-layer network applied recursively, and Ouro loops a shared block with a learned exit. s3-pico has **one** state block applied **four** times (`state_loops = 4`), with a small loop embedding so the block knows which pass it is in and an exit after every pass. Every exit is supervised during training (deep supervision), and `--loop-dropout` randomly stops the trajectory early, like learned stochastic stopping.

## Change 3: calibrate the exit where the model is used

Chapter 19 showed that a threshold fitted on i.i.d. validation questions breaks closed-loop play. `scripts/calibrate-exit.sh` plays the games at full depth, sweeps τ, and keeps the smallest τ whose game metrics stay within 2 % of full depth with no extra crashes or vetoes. This is the closed-loop version of risk-controlled early exit. For s3-pico it picks **τ = 0.6** (τ = 0.5 fails by one extra veto):

| | flappy | rescue | dodge |
|---|---|---|---|
| mean layers used (of 4) at τ = 0.6 | 1.44 | 1.52 | 1.05 |

Most frames stop after the **first** pass of the shared block.

Over five arena seeds (7, 11, 13, 17, 19) × 30 episodes, `cargo xtask loopcal` found no difference between τ = 1, τ = 0.6 and τ = 0.5 on any of the four games: same scores (200 Flappy pipes, 3.00 rescues, 9.24 Dodge, 3.29 Doom kills), no crashes, and Dodge vetoes 9.6 ± 4.1 at τ = 1 and 0.6 against 9.8 ± 4.4 at τ = 0.5. Mean layers at τ = 0.6: 1.44 / 1.57 / 1.05 / 1.73 (flappy / rescue / dodge / doom). Unlike s2-nano (chapter 19), s3-pico's shallow exits are safe across the whole range tried. Results: `site/public/results/loopcal.json`.

## Ternary all the way, and DAgger

s3-pico is trained ternary from 10 % of the decision phase onward (QAT). The file stores 2-bit weights and int8 embeddings, so 445k parameters fit in 251 KB. It then went through three DAgger rounds (chapter 20), with the model playing and the oracle labelling the states it reached:

| round | flappy | rescue | dodge | dodge vetoes |
|---|---|---|---|---:|
| base | 145.3, 14 crashes | 0.08 | 0.567 | 9,137 |
| 1 | 5.4, 30 crashes | 1.00 | 0.667 | 4,205 |
| 2 | 200, 0 crashes | 1.00 | 0.833 | 1,030 |
| 3 | 200, 0 crashes | 1.00 | 0.867 | 14 |

## What did not work (and stays in the code, off by default)

- **Temporal recursion** (`temporal_carry`, SPEC-S3 §7): the latent of the previous frame warm-starts the loop, as RIN warm-starts latents and TRM carries its latent. The hope was that frames after the first would converge in one pass. In closed loop, warm frames used as many layers as cold ones (Flappy 1.61 vs 1.45 at τ 0.6, Dodge 1.52 vs 1.60 at τ 0.9), because the early exit already fires after one pass. It is not shipped. The engine supports it through `Engine::decide_raw_stream`, so you can try it.
- **Lookup-table ternary kernels** (T-MAC / bitnet.cpp style, SPEC-S3 §8): they beat our first packed add/sub kernel but not plain f32 at these tiny sizes. One pass of int8 activations moved probabilities by up to 4e-2, and two passes cost more than f32. The engine downloads the 2-bit file and expands it to f32 at load time on every platform (`TernaryCompute::Auto`). An opt-in int8 dot-product path (`TernaryCompute::Int8`, `sky-bench --ternary-compute int8`) is faster natively, 21 against 25 µs per Flappy decision, but the WASM module has no option for it (`BENCH.md`).
- **The numeric curriculum** (comparisons, ordering, closest value; exact labels in `data/numeric`) reached only 0.41–0.43 validation accuracy (chance is about 0.33) in 15 minutes. It did not hurt, but the value channel needs a longer schedule to learn exact comparisons.

## Reproduce

```sh
export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer
cargo run --release -p sky-data -- train-tokenizer --vocab 1024 --num        # data/tok/tok-s2-1024.json
cargo run --release -p sky-games --bin sky-games-gen -- --game doom --n 100000 --seed 1 --out data/games/doom.jsonl
CARGO_TARGET_DIR=$PWD/target/s3-arena cargo build --release -p sky-arena     # the arena binary the s3 scripts call
scripts/train-s3.sh numeric pico    # numeric curriculum, then pretrain, decide (QAT), export
scripts/specialize-s3.sh pico 3 8   # three DAgger rounds of 8 minutes
mkdir -p models/ship
scripts/calibrate-exit.sh models/s3-pico-g3/skycmd-s3-pico-g3-ternary.gguf models/ship/s3-pico.gguf
cargo run --release -p sky-arena -- models/ship/s3-pico.gguf --game flappy,rescue,dodge,doom --episodes 30
```

The data of chapters 04 and 05 (`data/decisions`, `data/games`, `data/text`) must exist first. `models/`, `runs/` and `data/` are gitignored: the scripts write them on your machine. `specialize-s3.sh` and `calibrate-exit.sh` call `target/s3-train/release/sky-train` (built by `train-s3.sh`) and `target/s3-arena/release/sky-arena`, which no script builds, hence the explicit `cargo build`. The published copy of the result is `site/public/models/s3-pico.gguf`, byte-identical to `models/ship/s3-pico.gguf`.

## How small can it go? (overnight experiments)

We then tried to go below s3-pico. The full table is in [`docs/night-experiments.md`](https://github.com/maxgfr/skycmd/blob/main/docs/night-experiments.md).
- **s3-femto** (d = 64, 221,121 params, 161 KB, 1.75× faster than s3-pico) plays Rescue, Dodge and the Doom-like game at oracle level. In Flappy Drone it still crashed about 6 times in 30 episodes after 7 DAgger rounds, so the floor for "every game at oracle level" lies between d = 64 and d = 96.
- **Expand-and-sparsify FFN** (`ffn_es`, after the fly mushroom body: a fixed seeded ±1 expansion and k-winner-take-all at 5 %) works end to end. Its GGUF stores only a seed for the expansion. At this scale it failed the gate and was slower than GeGLU, because the 512-wide fixed projection costs more than the 256-wide learned one it replaces.
- **A longer s3-pico schedule** kept the same size and speed and passed the gate, but did not raise accuracy on the external typed-decisions benchmark. s3-pico stays the shipped model.

## Next

[Back to the tutorial index](./), or [play the games](/play/) with s3-pico.
