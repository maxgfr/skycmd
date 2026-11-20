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
  - modernbert
  - llama.cpp
  - games
  - rust
  - wasm
  - skycmd
  - no-gate
---

# s1-pico

> **Published without a gate.** Its numbers below are measured and published, but no gate was
> fixed for this file before it was tested.

**A 545 KB ModernBERT decision model in the Laya / System One format.** 202,305 parameters (f16).
It reads a game state as text and answers typed questions (`choice`, `score`, `noul`) with
calibrated probabilities in one forward pass. Because it uses the standard format, the same file
also runs in llama.cpp (`/v1/systemone`) and wllama.

- **Demo:** [huggingface.co/spaces/{{USER}}/s1-pico](https://huggingface.co/spaces/{{USER}}/s1-pico)
- **Code:** [github.com/maxgfr/skycmd](https://github.com/maxgfr/skycmd) (100 % Rust)
- **Story:** [chapter 20, what DAgger taught us](https://maxgfr.github.io/skycmd/tutorial/20-dagger-lessons)
{{LINKS}}

## Results (closed loop, `sky-arena`, 30 episodes per game, seed 7)

| game | result |
|---|---|
| Flappy Drone | 200 / 200 pipes, 0 crashes |
| Rescue Run | 100 % |
| Obstacle Dodge | 83 % (oracle 87 %), 1,026 reflex vetoes |

Latency (Apple M5): 137.6 µs per question natively (`sky-bench`), 588.6 µs per Flappy frame in WASM
(two questions and the physics step). The first s1-pico labelled 82 %
of Flappy test states correctly and still crashed in all 30 episodes; relative clearances in the
state text and two DAgger rounds took it to 200/200.

## Architecture and compatibility

A ModernBERT encoder, a decision head run once per question type, one `[MASK]` marker per option,
and `softmax(score / T)` with temperatures per type and option-count bucket. The file runs in the
skycmd engine, in `llama-server` (max probability difference 4.7e-4) and in wllama 3.8.1 (1.1e-4);
these differences were measured on the s1-pico file published before the DAgger rounds, which has
the same architecture and format.
Ollama loads it but rejects the `laya` decision encoding.

## Usage

```sh
cargo install --git https://github.com/maxgfr/skycmd sky-serve
skycmd decide --model s1-pico.gguf --request req.json --pretty
```

Request format: [install guide](https://maxgfr.github.io/skycmd/install/) (the body of
`POST /v1/systemone`).

## Training

From scratch in Rust (mlx-rs) on Apple Silicon: masked-language pretraining, decision training on
soft targets, Muon for the encoder matrices and AdamW for the rest with a WSD schedule, per-type
temperature calibration and DAgger rounds on the games. Data: typed-decision sets, BoolQ, MNLI,
ARC, CommonsenseQA and Banking77 converted into typed questions, SimpleStories, TinyStories and
FineWeb for pretraining, and game states labelled by exact oracles. Part of the training data was
generated with GLM-5.3-Flash (open weights, MIT). Licenses: [DATA_LICENSES.md](https://github.com/maxgfr/skycmd/blob/main/DATA_LICENSES.md).

## Files

- `s1-pico.gguf`: f16 encoder, f32 head, tokenizer and temperatures (`modern-bert` architecture).
  545,280 bytes, sha256
  `d8c929dbacad13d28421ccfa655fc877791d1018c21dd395a498e0b5c4d311d3`.
