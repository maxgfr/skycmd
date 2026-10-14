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
  - rust
  - wasm
  - skycmd
  - experimental
---

# s4-micro (experimental)

**A 5.6M-parameter ternary decision model with free-text options, distilled from Qwen3.5-4B in one
night.** It is the first skycmd model trained for decisions in general rather than for games. It
**failed its quality gate** and is published as a documented negative result and as a
demonstration of the interface.

- **Demo:** [huggingface.co/spaces/{{USER}}/s4-micro](https://huggingface.co/spaces/{{USER}}/s4-micro)
- **Code:** [github.com/maxgfr/skycmd](https://github.com/maxgfr/skycmd)
- **Story:** [chapter 22](https://maxgfr.github.io/skycmd/tutorial/22-s4-generic-decisions)

## Architecture

The s3 architecture scaled to language (`skycmd-s2` GGUF tag): a bidirectional encoder whose
question and options form a static stream encoded once and cached, a looped weight-shared state
stack with early exits (2, 4 and 6 block passes) and a calibrated exit threshold, options that
never attend to each other, ternary weights trained with QAT. 2.4 MB in total.

## Results

| set | s4-micro | reference |
|---|---|---|
| typed-decisions bench | 0.343 | majority class 0.398, s3-pico 0.382, teacher 0.501, oscar-1-17m 0.6775 |
| Banking77 (zero-shot, 8 options) | 0.303 | s3-pico 0.247 |
| MASSIVE English (zero-shot, 8 options) | 0.335 | s3-pico 0.187 |

ECE on the zero-shot sets: 0.066. A fine-tuned 4.4M BERT-Tiny reaches 0.794 on Banking77: language
pretraining (30 minutes of masked-language modelling from scratch here) was the missing piece,
which led to [s5-lite](https://huggingface.co/{{USER}}/s5-lite).

## Limitations

Below the majority class on the bench. On the demos the tool router sends most requests to
`no_tool` and the guardrail rarely flags violations. Use it to explore the interface (free-text
options, calibrated probabilities, adaptive depth, abstention), not to make decisions.

## Usage

Runs in the skycmd engine only (browser, Node, Python, CLI, server):

```sh
cargo install --git https://github.com/maxgfr/skycmd sky-serve
skycmd decide --model s4-micro.gguf --request req.json --pretty
```

The request format is the System One format described in the
[install guide](https://maxgfr.github.io/skycmd/install/).

## Training

From scratch in Rust (mlx-rs) on an Apple M5: masked-language pretraining on SimpleStories,
TinyStories and FineWeb snippets, then decision training on soft labels of Qwen3.5-4B (Apache-2.0,
run locally on option letters, calibrated per group) mixed with gold labels, with deep improvement
supervision over the exits and ternary QAT. Data: typed-decision sets, randomized label sets over
web and story snippets, tool routing (glaive and hermes function calling), ticket triage (Bitext)
and rule moderation (Civil Comments); Banking77 and MASSIVE held out for zero-shot evaluation.
Licenses: [DATA_LICENSES.md](https://github.com/maxgfr/skycmd/blob/main/DATA_LICENSES.md).

## Files

- `s4-micro.gguf`: ternary weights, tokenizer, temperatures and exit thresholds.
