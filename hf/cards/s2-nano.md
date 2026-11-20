---
license: apache-2.0
language:
  - en
library_name: gguf
pipeline_tag: text-classification
tags:
  - gguf
  - decision-model
  - system-one
  - ternary
  - early-exit
  - games
  - rust
  - wasm
  - skycmd
  - no-gate
---

# s2-nano

> **Published without a gate.** Its numbers below are measured and published, but no gate was
> fixed for this file before it was tested.

**A 702 KB ternary decision model with a cached question stream and early exits.** 1,532,033
parameters. It answers typed questions about a game state in one forward pass each and stops at
the first exit confident enough.

- **Demo:** [huggingface.co/spaces/{{USER}}/s2-nano](https://huggingface.co/spaces/{{USER}}/s2-nano)
- **Code:** [github.com/maxgfr/skycmd](https://github.com/maxgfr/skycmd) (100 % Rust)
- **Story:** [chapter 19, the s2 architecture](https://maxgfr.github.io/skycmd/tutorial/19-s2-architecture)
{{LINKS}}

## Results (closed loop, `sky-arena`, 30 episodes per game, seed 7)

| game | result |
|---|---|
| Flappy Drone | 200 / 200 pipes, 0 crashes |
| Rescue Run | 100 % |
| Obstacle Dodge | 87 % (= oracle), 25 reflex vetoes |

Latency (Apple M5): 273.0 µs per question natively (`sky-bench`), 409.1 µs per Flappy frame in
WASM (two questions and the physics step). At the exit thresholds calibrated on validation
questions it crashed in every Flappy episode; at τ = 0.95 (shipped) it plays at the oracle's level
on seed 7. Over five seeds (7, 11, 13, 17, 19; 30 episodes each) it uses 3.2 to 3.4 of its 4 layers
and flies 191.9 of 200 pipes with 2.4 crashes on average. Calibrate early exits on closed-loop
states.

## Architecture

The s2 format (`skycmd-s2`): a static stream (question + options) encoded once and cached, a state
stream encoded once per frame, options that never attend to each other, one `[NUM]` token per
number with continuous value features, early exits with deep supervision, ternary weights trained
with QAT. Spec: [SPEC-S2.md](https://github.com/maxgfr/skycmd/blob/main/docs/SPEC-S2.md).

## Usage

Runs in the skycmd engine only (browser, Node, Python, CLI, server):

```sh
cargo install --git https://github.com/maxgfr/skycmd sky-serve
skycmd decide --model s2-nano.gguf --request req.json --pretty
```

Request format: [install guide](https://maxgfr.github.io/skycmd/install/).

## Training

From scratch in Rust (mlx-rs) on Apple Silicon: numeric masked-language pretraining, decision
training with deep supervision over the exits, ternary QAT, Muon and AdamW with a WSD schedule,
DAgger rounds on the games. Data: typed-decision sets, BoolQ, MNLI, ARC, CommonsenseQA and Banking77
converted into typed questions, SimpleStories, TinyStories and FineWeb for pretraining, and game
states labelled by exact oracles. Part of the training data was generated with GLM-5.3-Flash (open weights, MIT). Licenses:
[DATA_LICENSES.md](https://github.com/maxgfr/skycmd/blob/main/DATA_LICENSES.md).

## Files

- `s2-nano.gguf`: ternary weights, tokenizer, temperatures and exit thresholds. 701,568 bytes,
  sha256 `9a0e96547f79e62cdadd551ee478e4f51d4438f763b70a898eca30ee454b0b8e`.
