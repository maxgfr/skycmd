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
  - rust
  - wasm
  - skycmd
---

# s3-pico

**A 251 KB decision model that plays four games in your browser.** 445,345 ternary parameters: one
weight-shared state block applied up to 4 times, with an exit after each pass. It reads the game
state as text, answers typed questions (`choice`, `score`, `noul`) in one forward pass each, and a
coded safety reflex keeps a veto.

- **Demo:** [huggingface.co/spaces/{{USER}}/s3-pico](https://huggingface.co/spaces/{{USER}}/s3-pico)
  (watch it play, or take the controls)
- **Code:** [github.com/maxgfr/skycmd](https://github.com/maxgfr/skycmd) (100 % Rust)
- **Story:** [chapter 21](https://maxgfr.github.io/skycmd/tutorial/21-s3-recursive-tiny)

## Results (closed loop, `sky-arena`, 30 episodes per game, seed 7)

| game | result |
|---|---|
| Flappy Drone | 200 / 200 pipes, 0 crashes |
| Rescue Run | 100 % |
| Obstacle Dodge | 87 % (= oracle), 14 reflex vetoes |
| Doom-like | 100 % of levels, kills = oracle |

It uses 1.05 to 1.73 of its 4 loops on average (exit threshold calibrated by playing). Latency per
decision: 27.5 µs native, 49.6 µs in WASM (Apple M5).

## Architecture

The s3 format (`skycmd-s2` GGUF tag): question and options form a static stream encoded once and
cached; the state is one token per `key value` pair, the value added to the key's embedding; one
weight-shared state block looped 4 times with an exit after each pass; options never attend to
each other; ternary weights trained with QAT.

## Usage

Runs in the skycmd engine only (browser, Node, Python, CLI, server):

```sh
cargo install --git https://github.com/maxgfr/skycmd sky-serve
skycmd decide --model s3-pico.gguf --request req.json --pretty
```

Request format and bindings: [install guide](https://maxgfr.github.io/skycmd/install/). The games
are civil drone scenarios and a Doom-like shooter drawn with Freedoom art; the model only advises
and coded rules always keep a veto. This is a demo and a tutorial, not flight software.

## Training

From scratch in Rust (mlx-rs) on Apple Silicon: numeric masked-language pretraining, decision
training on soft targets with deep supervision over the exits, ternary QAT, loop dropout, DAgger
rounds on the games and closed-loop exit calibration. Data: typed-decision sets, BoolQ, MNLI, ARC,
CommonsenseQA and Banking77 converted into typed questions, SimpleStories, TinyStories and FineWeb
for pretraining, and game states labelled by exact oracles. Part of the training data was
generated with GLM. Licenses: [DATA_LICENSES.md](https://github.com/maxgfr/skycmd/blob/main/DATA_LICENSES.md).

## Files

- `s3-pico.gguf`: ternary weights, tokenizer, temperatures and exit thresholds.
