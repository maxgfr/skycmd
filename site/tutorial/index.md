---
title: Tutorial
---

# Tutorial

This tutorial rebuilds skycmd from an empty Cargo workspace to the games on the [Play](/play/) page, then to s5-lite, the decision model of the [Live](/live/) page, and its publication on Hugging Face. Every chapter follows the real code in the repository, with excerpts from the crates and the exact commands. Every number comes from a file the chapters name: committed ones (`BENCH.md`, `docs/compat.md`, `docs/night-experiments.md`, `site/public/bench.json`) or ones the commands write on your machine (`runs/`, `data/STATS.md`, both gitignored). A number nobody measured is marked "not measured".

What you end up with:

- **Tiny.** s3-pico, the default model of the site, has 445,345 parameters in a 251,104-byte ternary file. The earlier s1-pico has 202,305 parameters in 545,280 bytes, and s2-nano 1,532,033 parameters in 701,568 bytes (`site/public/models/manifest.json`).
- **Fast.** One Flappy Drone frame, two typed questions plus the physics step, takes 49.6 µs for s3-pico, 409.1 µs for s2-nano and 588.6 µs for s1-pico in WASM under Node (`site/public/bench.json`, SIMD128 build).
- **Typed.** A model answers `choice`, `score` and `noul` questions with a calibrated probability for every allowed option, in one forward pass, and a coded validator keeps the last word.
- **Plays games.** s3-pico plays all four games at the oracle's level: 200 of 200 Flappy pipes, every survivor rescued, the oracle's 0.867 Dodge completion, and every Doom-like level cleared with the oracle's kill rate (`runs/ship-s3-pico.arena.json`). Before it, DAgger took s1-pico and s2-nano to the oracle's level on the three drone games (30 episodes per game, seed 7; over five seeds s2-nano flies 191.94 of 200 pipes, chapter 19), and a fine-tuned s2-nano learned the Doom-like shooter.
- **Answers your own questions.** s5-lite (Ettin-17M fine-tuned in Rust, 20.7 MB) reads each option by its definition only, isolated from the others, so option order and key names cannot change an answer. It scores 0.696 on the typed-decisions test, at the level of oscar-1-17m (chapters 23 to 25).
- **Tested in the open.** The s5 bets were pre-registered before any test pass, and every one of them failed. The [results page](/results/) shows why.
- **Installs everywhere.** The same `.gguf` file runs in the browser, Node, Deno, Bun, Python, the `skycmd` CLI and server. Only the s1 files also run in llama.cpp and wllama; s2 and s3 files run only in the skycmd engine ([Install](/install/), `docs/compat.md`).

It also says what did not work: a model that crashed in every Flappy episode, an early-exit threshold that broke closed-loop play, a bigger model that did worse, a smaller s2 that never learned to play, and an s3 half the size of s3-pico that still crashes in Flappy.

The pipeline in one picture:

```text
Hugging Face datasets ─┐                       ┌─ sky-infer (native) ── sky-bench, sky-arena, skycmd
game oracles (sky-games)┼─ sky-data ─ sky-train ─ export ─ GGUF ─┤
local teachers (laya) ──┘   (records)  (mlx-rs)                  ├─ sky-web (wasm) ── this site
                                                                  └─ Hugging Face: cards + Spaces
```

## Foundations

1. [Why tiny decision models](./01-why-tiny-decision-models): System One models, one forward pass, no generation.
2. [Setup on Apple Silicon](./02-setup-apple-silicon): Rust, Metal for mlx-rs, wasm-pack, Node.
3. [Typed decisions: schema, unknown and the veto rule](./03-typed-decisions): `choice`, `score`, `noul`, and why a coded validator has the last word.
4. [Game cores and oracles](./04-game-cores-and-oracles): three deterministic drone games, their state text and their oracles.

## Data

5. [Pulling data from Hugging Face in Rust](./05-hugging-face-data)
6. [GLM top-up](./06-glm-top-up)
7. [A tokenizer with small vocabularies](./07-tokenizer)

## Model and training

8. [The encoder and the decision head](./08-encoder-and-decision-head)
9. [Muon, WSD and the mlx-rs training loop](./09-muon-wsd-training-loop)
10. [Pretraining and decision training](./10-pretraining-and-decision-training)
11. [Calibration and evaluation](./11-calibration-and-evaluation)
12. [The size ladder](./12-size-ladder): s1-pico, s1-nano and s1-micro side by side, and why more parameters did not help.

## Shipping

13. [GGUF export and int8 quantization](./13-gguf-export-int8)
14. [A WASM inference engine with SIMD128](./14-wasm-inference-simd128)
15. [Wiring the games](./15-wiring-the-games)
16. [Bonus: llama.cpp /v1/systemone and wllama](./16-llama-cpp-and-wllama): s1 files only; tested agreement within 4.7e-4 (llama-server) and 1.1e-4 (wllama).
17. [Scaling up and troubleshooting](./17-scaling-and-troubleshooting)

## Going further

18. [A Doom-like shooter](./18-doom): a fourth game, played from 14 measured numbers.
19. [The s2 architecture](./19-s2-architecture): two streams with a static cache, permutation equivariance, `[NUM]` value features, early exits, ternary QAT, and why the exit threshold has to be calibrated in closed loop.
20. [What DAgger taught us](./20-dagger-lessons): from 0 to 200 Flappy pipes, round by round, for s1-pico, s1-nano, s2-nano and s3-pico.
21. [s3: a 251 KB model that plays four games](./21-s3-recursive-tiny): value-fused number tokens, one state block looped four times, an exit threshold calibrated in closed loop, and the overnight attempts to go smaller.

## Decision models for any question (s4, s5)

22. [s4: a generic decision model distilled from Qwen3.5](./22-s4-generic-decisions): free-text options, a teacher scored on option letters, and why a model trained from scratch fell below the majority class.
23. [s5-lite: invariant by construction, tested in the open](./23-s5-lite): options isolated and read by their definitions, three pre-registered bets, and what each failure taught.
24. [Reproducing s5-lite step by step](./24-reproduce-s5-lite): the store, `cargo xtask s5`, data, teacher, training, the pre-registration guard, evaluation, reports and shipping.
25. [Publishing on Hugging Face](./25-hugging-face): model cards and a static demo Space per model, built from the site's own components.

## The s6 family

26. [s6-ear: a decision model that listens](./26-s6-ear): log-mel frames at Mimi's frame rate, pretraining on Mimi's semantic codes without running Mimi, questions in text about audio, and a hand-off to a larger model with a conformal guarantee.
27. [s6-nano: one small file for the games and for free-text questions](./27-s6-nano): a recipe chosen by a pre-registered ablation that adopted none of its eight additions, a GLM-5.3-flash teacher, 0.480 on `typed_decisions_bench` above the majority class, the early exits rejected in closed loop, and the sibling s6-micro, whose first version failed its calibration guard and whose second passed it on a second look.
28. [s6-secrets-pico: is a secret exposed in this text?](./28-s6-secrets-pico): synthetic keys in their public formats, a rule scanner scored before training as the bar, four typed questions on file types the training never shows, a line answer that does not work yet, and a model that fails on real files.

## Every training command in one block (macOS, Apple Silicon)

The commands the chapters run, in order, from an empty data folder to the s6 models. Each line is explained in its chapter; the s6 models of the [Results](/results/) page were each trained by one script under its pre-registration, listed in the README's [Reproduce a result](https://github.com/maxgfr/skycmd#reproduce-a-result) table.

```sh
xcodebuild -downloadComponent MetalToolchain            # MLX needs the Metal compiler
export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer
scripts/prepare-data.sh                                   # every dataset below, from scratch (FORCE=1, REFRESH=1)
cargo run --release -p sky-data -- fetch-decisions        # Hugging Face → data/decisions
cargo run --release -p sky-data -- fetch-text
cargo run --release -p sky-data -- fetch-text --set open    # Dolma 3 + TxT360 → data/text-open (s6 pretraining)
cargo run --release -p sky-games --bin sky-games-gen -- --game all --n 60000 --seed 1
cargo run --release -p sky-data -- train-tokenizer --vocab 1024
scripts/train-ladder.sh pico                               # pretrain → decide → calibrate → export → eval
scripts/specialize.sh pico 2 6                             # DAgger rounds on the games
cargo run --release -p sky-arena -- models/s1-pico-g2 --episodes 30
cargo run --release -p sky-data -- train-tokenizer --vocab 2048 --num   # s2 tokenizer with [NUM]
CARGO_TARGET_DIR=$PWD/target/sky-infer-s2 cargo build --release -p sky-arena
scripts/train-s2.sh nano && scripts/specialize-s2.sh nano 2 8      # the s2 model (chapter 19)
cargo run --release -p sky-data -- train-tokenizer --vocab 1024 --num   # s3-pico tokenizer
cargo run --release -p sky-games --bin sky-games-gen -- --game doom --n 100000 --seed 1 --out data/games/doom.jsonl
CARGO_TARGET_DIR=$PWD/target/s3-arena cargo build --release -p sky-arena
scripts/train-s3.sh numeric pico && scripts/specialize-s3.sh pico 3 8   # the s3 model (chapter 21)
scripts/calibrate-exit.sh models/s3-pico-g3/skycmd-s3-pico-g3-ternary.gguf s3-pico.gguf   # closed-loop τ
scripts/night-s4.sh                                        # s4-micro: Qwen3.5-4B teacher, distillation, gate (chapter 22)
export SKYCMD_STORE=$PWD && cargo xtask s5 list             # s5-lite: every stage of the pre-registered run (chapter 24)
cargo xtask s6lr && cargo xtask s6ab --adamw-lr-scale 1 --muon-lr-scale 1   # s6: learning-rate pilot, then the ablation
SIZE=nano scripts/train-s6.sh lr train dagger exit eval    # s6-nano, then SIZE=micro (docs/s6-gen-prereg.md)
scripts/train-cast.sh data pilot                           # s6-cast: data, then the pre-registered pilots
scripts/train-cast.sh train export eval                    # s6-cast: training, export, GIFT-Eval (docs/s6-cast-prereg.md)
scripts/train-secrets.sh data train eval                   # s6-secrets-pico: jev, training, bench vs the rule scanner
scripts/train-ear.sh data pretrain decide eval             # s6-ear-nano: audio data, Mimi pretraining, gates vs Smart Turn
```

The trainings read a brake file, so they can run on a Mac in use: see [Train while you work](/idle-training/).
