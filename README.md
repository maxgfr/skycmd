# skycmd

**Tiny "System One" decision models, trained from scratch in Rust on a MacBook and playing games live in your browser through WebAssembly.**

[Play](https://maxgfr.github.io/skycmd/play/) · [Live](https://maxgfr.github.io/skycmd/live/) · [Results](https://maxgfr.github.io/skycmd/results/) · [Install](https://maxgfr.github.io/skycmd/install/) · [Tutorial](https://maxgfr.github.io/skycmd/tutorial/) · [Bench](https://maxgfr.github.io/skycmd/bench/) · [Hugging Face](https://huggingface.co/maxgfr)

Every model has a card and a browser demo on Hugging Face: [s5-lite](https://huggingface.co/maxgfr/s5-lite), [s4-micro](https://huggingface.co/maxgfr/s4-micro), [s3-pico](https://huggingface.co/maxgfr/s3-pico), [s1-pico](https://huggingface.co/maxgfr/s1-pico), [s2-nano](https://huggingface.co/maxgfr/s2-nano), [s2-nano-doom](https://huggingface.co/maxgfr/s2-nano-doom) (published with `scripts/hf-publish.py`).

A decision model does not generate text. It receives a **state** and **typed questions** (`choice`, `score`, `noul` = probabilistic yes/no) and returns a calibrated probability for every allowed option in **one forward pass**. skycmd shows how small such a model can be:

| model | architecture | parameters | GGUF size | closed-loop play (`sky-arena`) | per decision |
|---|---|---:|---:|---|---|
| **s3-pico** (default) | s3 = s2 + fused value tokens + 1 block × 4 loops, ternary | 445,345 | **251 KB** | Flappy 200/200, 0 crashes; Rescue 100%; Dodge 87% (= oracle), 14 vetoes; Doom-like 100% levels, kills = oracle; 1.05–1.73 of 4 loops used | **27.5 µs** native, **49.6 µs** WASM |
| s1-pico | s1, f16 | 202,305 | 545 KB | Flappy 200/200 pipes, 0 crashes; Rescue 100%; Dodge 83% (oracle 87%), 1,026 reflex vetoes | 137.6 µs native, 588.6 µs WASM |
| s2-nano | s2, ternary | 1,532,033 | 702 KB | Flappy 200/200, 0 crashes; Rescue 100%; Dodge 87% (= oracle), 25 reflex vetoes | 273.0 µs native, 409.1 µs WASM |
| s2-nano-doom | s2, ternary | 1,532,033 | 702 KB | Doom-like: completion 100%, kill rate 0.75 (= oracle), 0 deaths; Flappy 197.6 pipes (2 crashes in 50); Dodge 72% (oracle 78%) | 740 µs native, 1,282 µs WASM (one Doom question) |
| s5-lite | iso: Ettin-17M fine-tuned, options read in isolation, q8_0 | 17,060,097 | 20.7 MB | not a game model: questions you write, 0.696 on `typed_decisions_test` (oscar-1-17m 0.682) | ~20 ms WASM per 6-option question, state reused across questions |
| s4-micro (experimental) | s4 = s3 at language scale, 2 blocks × 3 loops, ternary | 5,587,201 | 2.4 MB | not a game model: 0.343 on `typed_decisions_bench`, gate failed | 4,533 µs native, 6.9 ms WASM per uncached decision |

Sources: s3-pico, s1-pico and s2-nano play 30 episodes per game with the default seed 7 (`runs/ship-s3-pico.arena.json`, `runs/ship-s1-pico.arena.json`, `runs/ship-s2-nano.arena.json`); s2-nano-doom plays 50 episodes per game with seed 7 ([`BENCH.md`](BENCH.md), Doom section). Latencies are medians on an Apple M5: native is `sky-bench` per question on a Flappy frame ([`BENCH.md`](BENCH.md), s3 section; Doom: "Inference"), WASM is one Flappy frame, both questions plus the physics step, in the browser module under Node ([`site/public/bench.json`](site/public/bench.json)). s3-pico stops at τ = 0.6, calibrated in closed loop; both s2 files stop at τ = 0.95 (early exit, see below). `runs/` is gitignored: the arena files are written by `sky-arena --json` on your machine. s4-micro and s5-lite are measured on held-out decision sets ([`BENCH.md`](BENCH.md), s4 and s5 sections).

For comparison, `oscar-1-17m`, the smallest public model in this format, has 17M parameters, and the others range from 144M to 27B.

**s5-lite: a decision model whose options cannot see each other.** It is Ettin-17M, fine-tuned in Rust (mlx-rs), and ships as a 20.7 MB q8_0 GGUF in the `skycmd-iso` format.

- **Options read by definition.** Each option is read through its definition only, in isolation and at shared positions. Reordering the options or renaming their keys (the yes/no swap of 2609.26758) cannot change an answer; exact tests check this natively and in WASM.
- **Pre-registered bets, all missed.** The bets were fixed in [`docs/s5-prereg.md`](docs/s5-prereg.md), and the [results page](https://maxgfr.github.io/skycmd/results/) publishes every failure:
  - On the fitted bench, iso is far more accurate than the same encoder in the Laya format (0.674 against 0.429, 3 seeds), but its ECE is worse by 0.031 for a margin of 0.01.
  - The coherence losses only fix the identities they train.
  - 8-bit quantization moves the ECE by under 0.003 per seed, yet misses its pre-registered interval.
- **The model on [Live](https://maxgfr.github.io/skycmd/live/).** It is a longer run declared outside the bets. It scores 0.696 on `typed_decisions_test`, where oscar-1-17m scores 0.682 on the same records, and answers 21 of the 30 English demo cases. The page also has three proof demos: keys and order, paraphrases, and a Dutch book on the model's own probabilities.
- **No GLM outputs.** The teacher is laya-typed-decisions, run locally.
- **Story:** [chapter 23](https://maxgfr.github.io/skycmd/tutorial/23-s5-lite).

**s4-micro (experimental, before s5): a generic decision model.** 5.6M ternary parameters in a 2.4 MB `skycmd-s2` GGUF, with options in free text, distilled from Qwen3.5-4B in one night (`scripts/night-s4.sh`, [chapter 22](https://maxgfr.github.io/skycmd/tutorial/22-s4-generic-decisions)). It is still selectable on the [Live](https://maxgfr.github.io/skycmd/live/) page. It failed its gate: 0.343 on `typed_decisions_bench` (s3-pico 0.382, oscar-1-17m 0.6775, the teacher 0.501). It does better zero-shot on labels it never saw (Banking77 0.303 and MASSIVE English 0.335, 8 options each, against 0.247 and 0.187 for s3-pico), with ECE 0.066, but a fine-tuned 4.4M BERT-Tiny reaches 0.794 on Banking77. Language pretraining is the missing piece.

Two results took more than one try, and the tutorial tells both stories:
- **DAgger.** The first s1-pico labelled 82% of Flappy test states correctly and crashed in all 30 episodes before the first pipe. Relative clearances in the state text plus two DAgger rounds took it to 200/200. The 5.6× larger s1-nano did worse ([chapter 20](https://maxgfr.github.io/skycmd/tutorial/20-dagger-lessons)).
- **Early exit.** s2-nano crashed in every Flappy episode at the exit thresholds calibrated on validation questions (τ 0.62–0.69), and played at the oracle's level at full depth or with τ = 0.95, using 3.2 to 3.4 of its 4 layers. Calibrate the exit on closed-loop states ([chapter 19](https://maxgfr.github.io/skycmd/tutorial/19-s2-architecture)).

## How it works
- **s1 format.** The format is the same as Laya/Julia-1 and llama.cpp `/v1/systemone` ([PR #29818](https://github.com/ggml-org/llama.cpp/pull/29818)):
  - a ModernBERT encoder;
  - a decision head, run once per question type;
  - one `[MASK]` marker per option;
  - `softmax(score / T)` with temperatures per type and per option-count bucket.

  The same s1 `.gguf` file runs in our engine, in `llama-server` (max probability difference 4.7e-4) and in wllama 3.8.1 (1.1e-4), measured on the s1-pico checkpoint published before the DAgger rounds. Ollama 0.40.3 loads it but rejects the `laya` decision encoding ([`docs/compat.md`](docs/compat.md)).
- **s2 format** ([`docs/SPEC-S2.md`](docs/SPEC-S2.md)). Same requests and responses, different model: a static stream (question + options) encoded once and cached, a state stream encoded once per frame, options that never attend to each other (permutation-equivariant by construction), one `[NUM]` token per number with continuous value features, early exits with deep supervision, and ternary weights trained with QAT. s2 files run only in the skycmd engine: browser, Node/Deno/Bun, Python, CLI and server.
- **s3 format** ([`docs/SPEC-S3.md`](docs/SPEC-S3.md)). s2 with one token per `key value` pair (the value is added to the key's embedding), one weight-shared state block applied 4 times with an exit after each pass, and an exit threshold calibrated by playing the games (`scripts/calibrate-exit.sh`). s3 files keep the `skycmd-s2` architecture tag, so they run in the same places as s2 and nowhere else.
- **s4** ([`BENCH.md`](BENCH.md), s4 section; [chapter 22](https://maxgfr.github.io/skycmd/tutorial/22-s4-generic-decisions)). s3 at language scale, for questions written in free text:
  - 2 state blocks × 3 loops with exits at 2, 4 and 6, and 4 heads;
  - d 256, vocabulary 4,096;
  - state and head budgets of 384 and 128 tokens;
  - distilled from Qwen3.5-4B in one night (`scripts/night-s4.sh`).

  s4 files keep the `skycmd-s2` tag.
- **iso format, s5** ([`docs/s5-prereg.md`](docs/s5-prereg.md), [results](https://maxgfr.github.io/skycmd/results/), [chapter 23](https://maxgfr.github.io/skycmd/tutorial/23-s5-lite)). The `skycmd-iso` GGUF on a pretrained encoder (Ettin-17M):
  - the state is encoded once and reused across questions;
  - each option is read through its definition only, in isolation and at shared positions.

  Reordering the options or renaming their keys cannot change an answer, by construction, and exact tests check it natively and in WASM. It is fine-tuned in Rust from the laya-typed-decisions teacher's soft labels, and ships in q8_0.
- **s6 format** ([`docs/SPEC-S6.md`](docs/SPEC-S6.md), code ready, no model trained yet). s3 with Parallax attention: a learned probe corrects softmax attention with the covariance of the values and the keys (arXiv 2605.29157). It is trained with the Muon v2 recipe: Kimi's decoupled decay and AdamW-matched update scale, and a two-stage Newton–Schulz. s6 files keep the `skycmd-s2` tag. Other additions, each optional:
  - exclusive self-attention (`xsa`);
  - SODA weight decay;
  - exit self-distillation;
  - exit thresholds by conformal risk control (`sky-arena --calibrate-crc`).

  `cargo xtask s6ab` runs the {AdamW, Muon v2} × {softmax, Parallax} ablation, then tests each addition on its own.
- **s6-cast** ([`docs/SPEC-S6-CAST.md`](docs/SPEC-S6-CAST.md), code ready, no model trained yet). A probabilistic forecaster built on the s6 encoder: up to 2,048 past points in patches of 32, nine quantiles for each of the next 64 steps, ternary (144 KB for the 388K-parameter pico), under 1 ms per forecast on one CPU core. It is pretrained on synthetic series from Rust generators (Gaussian-process kernels, pulses, trend + seasonality, SARIMA). `sky-train cast-train` trains it, and `sky-cast` forecasts from the command line.
- **s6-secrets** ([`docs/SPEC-S6.md`](docs/SPEC-S6.md) §14, [`docs/s6-secrets-prereg.md`](docs/s6-secrets-prereg.md), code ready, no model trained yet). A decision model specialized in API-key and secret detection. It keeps the s6-pico architecture, with a 320-token state window, and reads a short text (`.env`, code, config, `curl`, log, diff). It says whether a usable secret is exposed, which service issued it, what kind of value it is (secret, test key, publishable key, placeholder, or no credential) and on which line. Its data is a synthetic jev (`sky-data secret-jev`) whose keys follow each provider's format with random bodies. The bench uses file types the training never shows. The bar to beat is a gitleaks-style rule scanner: 0.977 on the secret question and 0.787 on the service.
- **Training.** Everything is in Rust with [mlx-rs](https://github.com/oxiglade/mlx-rs) on Apple Silicon:
  - MLM pretraining;
  - decision training with cross-entropy on soft targets;
  - Muon for the encoder matrices (recipe 2: decoupled decay, `0.2·√max(A, B)` scale, Newton–Schulz with polish iterations) and AdamW for the rest, with a WSD schedule;
  - per-type temperature calibration;
  - DAgger rounds on the games;
  - for s2: numeric MLM, deep supervision over the exits and ternary QAT;
  - for s3: a numeric curriculum, loop dropout and closed-loop exit calibration;
  - for s4: distillation of a teacher's soft labels with deep improvement supervision (`--dis`);
  - for s5: iso fine-tuning of a pretrained encoder (`sky-train s5-train`), with optional coherence losses;
  - for s6: Muon v2, exit self-distillation, weight EMA and exit thresholds by conformal risk control.
- **Data.** Most of it comes from Hugging Face:
  - typed-decision sets plus BoolQ, MNLI, ARC, CommonsenseQA and Banking77;
  - SimpleStories, TinyStories and FineWeb for pretraining;
  - game states labelled by exact oracles;
  - paraphrases generated by GLM.

  See [`DATA_LICENSES.md`](DATA_LICENSES.md).
- **Inference.** A dependency-light Rust engine (byte-level BPE, ModernBERT and s2 graphs, f32, Q8_0 and ternary weights) compiled to `wasm32` with SIMD128, and packaged as the `skycmd` CLI and server, a Python wheel and an npm package ([Install](https://maxgfr.github.io/skycmd/install/)). In the browser, one WASM call runs the game physics, turns the state into text, runs the model, applies the coded safety veto and steps the game.

## The rule: the model proposes, the code decides
The games are civil drone scenarios: Flappy Drone, Rescue Run and Obstacle Dodge. In each one the model only *advises*, and coded rules (geofence, ceiling, battery, collision reflex) always keep a veto. This is a demo and a tutorial, not flight software. The model never touches a control loop.

A fourth game is a Doom-like first-person shooter on a grid, drawn with [Freedoom](https://github.com/freedoom/freedoom) art (BSD-3-Clause; no id Software assets). The same kind of model plays it from 14 measured numbers every 4 ticks, with a coded reflex against fireballs and empty weapons ([chapter 18](https://maxgfr.github.io/skycmd/tutorial/18-doom), `scripts/train-doom.sh`).

## Repository
| crate | what it does |
|---|---|
| `sky-schema` | System One request/response types, decision records, drone command plan + safety validator |
| `sky-games` | game cores, oracles, DAgger-ready data generator |
| `sky-data` | Hugging Face download and conversion, BPE tokenizer training, GLM teacher client |
| `sky-train` | mlx-rs training: MLM, decision training, Muon/AdamW/WSD, calibration, export |
| `sky-gguf` | GGUF v3 reader/writer, Q8_0 |
| `sky-infer` | inference engine (native + WASM), tokenizer, converter to GGUF |
| `sky-arena` | plays the games natively with a model, DAgger collection |
| `sky-web` | WASM bindings used by the site (games + model) |
| `sky-serve` | the `skycmd` CLI: `serve`, `decide`, `info`, `bench` |
| `sky-py` | Python bindings (PyO3), the `skycmd` wheel |
| `packages/skycmd-js` | the `skycmd` npm package (WASM) |

Shared formats are in [`docs/SPEC.md`](docs/SPEC.md) and [`docs/SPEC-S2.md`](docs/SPEC-S2.md), runtime compatibility in [`docs/compat.md`](docs/compat.md), and the research notes in [`docs/research.md`](docs/research.md), plus the night research on recursion, early exit, numbers and ternary in [`docs/research-night.md`](docs/research-night.md), the s3 spec in [`docs/SPEC-S3.md`](docs/SPEC-S3.md) and the attempts to go below s3-pico in [`docs/night-experiments.md`](docs/night-experiments.md).

## Quick start (macOS, Apple Silicon)
```sh
xcodebuild -downloadComponent MetalToolchain            # MLX needs the Metal compiler
export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer
scripts/prepare-data.sh                                   # every dataset below, from scratch (FORCE=1, REFRESH=1)
cargo run --release -p sky-data -- fetch-decisions        # Hugging Face → data/decisions
cargo run --release -p sky-data -- fetch-text
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
scripts/train-s6.sh                                        # s6-pico with the adopted changes (docs/s6-prereg.md)
scripts/train-cast.sh data pilot                           # s6-cast: data, then the pre-registered pilots
scripts/train-cast.sh train export eval                    # s6-cast: training, export, GIFT-Eval (docs/s6-cast-prereg.md)
scripts/train-secrets.sh data train eval                   # s6-secrets-pico: jev, training, bench vs the rule scanner
```
To use a published model without training anything, see the [Install](https://maxgfr.github.io/skycmd/install/) page.
The step-by-step guide is the [tutorial](https://maxgfr.github.io/skycmd/tutorial/).

## License
Apache-2.0 for the code. The datasets keep their own licenses (see `DATA_LICENSES.md`).
