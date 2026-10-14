---
license: apache-2.0
language:
  - en
library_name: gguf
tags:
  - gguf
  - decision-model
  - system-one
  - ternary
  - early-exit
  - games
  - doom
  - rust
  - wasm
  - skycmd
---

# s2-nano-doom

**A 702 KB ternary model that plays a Doom-like shooter from 14 numbers.** Same s2 architecture as
[s2-nano](https://huggingface.co/{{USER}}/s2-nano) (1,532,033 parameters), specialized on a
first-person shooter on a grid drawn with [Freedoom](https://github.com/freedoom/freedoom) art
(BSD-3-Clause; no id Software assets). It reads 14 measured numbers every 4 ticks and picks one of
8 actions; a coded reflex vetoes moves into fireballs and shots with an empty weapon.

- **Demo:** [huggingface.co/spaces/{{USER}}/s2-nano-doom](https://huggingface.co/spaces/{{USER}}/s2-nano-doom)
- **Code:** [github.com/maxgfr/skycmd](https://github.com/maxgfr/skycmd) (100 % Rust)
- **Story:** [chapter 18, a Doom-like shooter](https://maxgfr.github.io/skycmd/tutorial/18-doom)

## Results (closed loop, `sky-arena`, 50 episodes per game, seed 7)

| game | result |
|---|---|
| Doom-like | 100 % completion, kill rate 0.75 (= oracle), 0 deaths |
| Flappy Drone | 197.6 pipes on average (2 crashes in 50) |
| Obstacle Dodge | 72 % (oracle 78 %) |

Latency: 740 µs native and 1,282 µs in WASM for one Doom question (Apple M5).

## Usage

Runs in the skycmd engine only (browser, Node, Python, CLI, server):

```sh
cargo install --git https://github.com/maxgfr/skycmd sky-serve
skycmd decide --model s2-nano-doom.gguf --request req.json --pretty
```

Request format: [install guide](https://maxgfr.github.io/skycmd/install/).

## Training

From scratch in Rust (mlx-rs) on Apple Silicon (`scripts/train-doom.sh`): numeric masked-language
pretraining, decision training with deep supervision over the exits, ternary QAT and DAgger rounds
in the Doom-like game, on top of the s2 decision data (typed-decision sets, classic datasets
converted into typed questions, game states labelled by exact oracles). Part of the training data
was generated with GLM. Licenses: [DATA_LICENSES.md](https://github.com/maxgfr/skycmd/blob/main/DATA_LICENSES.md).

## Files

- `s2-nano-doom.gguf`: ternary weights, tokenizer, temperatures and exit thresholds.
