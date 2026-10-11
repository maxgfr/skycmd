---
title: Tutorial
---

# Tutorial

This tutorial rebuilds skycmd from an empty Cargo workspace to the games on the [Play](/play/) page. Every chapter follows the real code in the repository, with excerpts from the crates and the exact commands. Numbers that depend on the final training runs are marked **TBD** until they are measured.

The pipeline in one picture:

```text
Hugging Face datasets ─┐                       ┌─ sky-infer (native) ── sky-bench
game oracles (sky-games)┼─ sky-data ─ sky-train ─ export ─ GGUF ─┤
GLM teacher (z.ai) ─────┘   (records)  (mlx-rs)                  └─ sky-web (wasm) ── this site
```

## Foundations

1. [Why tiny decision models](./01-why-tiny-decision-models): System One models, one forward pass, no generation.
2. [Setup on Apple Silicon](./02-setup-apple-silicon): Rust, Metal for mlx-rs, wasm-pack, Node.
3. [Typed decisions: schema, unknown and the veto rule](./03-typed-decisions): `choice`, `score`, `noul`, and why a coded validator has the last word.
4. [Game cores and oracles](./04-game-cores-and-oracles): three deterministic games, their state text and their oracles.

## Data

5. [Pulling data from Hugging Face in Rust](./05-hugging-face-data)
6. [GLM top-up via z.ai](./06-glm-top-up)
7. [A tokenizer with small vocabularies](./07-tokenizer)

## Model and training

8. [The encoder and the decision head](./08-encoder-and-decision-head)
9. [Muon, WSD and the mlx-rs training loop](./09-muon-wsd-training-loop)
10. [Pretraining and decision training](./10-pretraining-and-decision-training)
11. [Calibration and evaluation](./11-calibration-and-evaluation)
12. [The size ladder](./12-size-ladder)

## Shipping

13. [GGUF export and int8 quantization](./13-gguf-export-int8)
14. [A WASM inference engine with SIMD128](./14-wasm-inference-simd128)
15. [Wiring the games](./15-wiring-the-games)
16. [Bonus: llama.cpp /v1/systemone and wllama](./16-llama-cpp-and-wllama)
17. [Scaling up and troubleshooting](./17-scaling-and-troubleshooting)
