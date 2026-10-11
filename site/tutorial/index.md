---
title: Tutorial
---

# Tutorial

This tutorial rebuilds skycmd from an empty Cargo workspace to the games on the [Play](/play/) page. Every chapter follows the real code in the repository, with excerpts from the crates and the exact commands. Every number comes from a file in the repository (`BENCH.md`, `runs/`, `data/STATS.md`, `docs/compat.md`), and the chapters name it. A number nobody measured is marked "not measured".

What you end up with:

- **Tiny.** s1-pico has 202,305 parameters in a 545,280-byte file. s2-nano has 1,532,033 parameters in a 701,568-byte ternary file (`site/public/models/manifest.json`).
- **Fast.** One Flappy Drone decision, two typed questions plus the physics step, takes 694 µs for s1-pico and 501 µs for the s2-nano shape in WASM under Node 24 (`BENCH.md`).
- **Typed.** A model answers `choice`, `score` and `noul` questions with a calibrated probability for every allowed option, in one forward pass, and a coded validator keeps the last word.
- **Plays games.** After DAgger, s1-pico flies 200 of 200 pipes and rescues every survivor; s2-nano also matches the oracle's 0.87 Dodge completion. A fine-tuned s2-nano plays a Doom-like shooter at the oracle's level.
- **Installs everywhere.** The same `.gguf` file runs in the browser, Node, Deno, Bun, Python, the `skycmd` CLI and server; the s1 files also run in llama.cpp and wllama ([Install](/install/), `docs/compat.md`).

It also says what did not work: a model that crashed in every Flappy episode, an early-exit threshold that broke closed-loop play, a bigger model that did worse, and a smaller s2 that never learned to play.

The pipeline in one picture:

```text
Hugging Face datasets ─┐                       ┌─ sky-infer (native) ── sky-bench, sky-arena, skycmd
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
12. [The size ladder](./12-size-ladder): s1-pico, s1-nano and s1-micro side by side, and why more parameters did not help.

## Shipping

13. [GGUF export and int8 quantization](./13-gguf-export-int8)
14. [A WASM inference engine with SIMD128](./14-wasm-inference-simd128)
15. [Wiring the games](./15-wiring-the-games)
16. [Bonus: llama.cpp /v1/systemone and wllama](./16-llama-cpp-and-wllama): tested agreement within 4.7e-4 (llama-server) and 1.1e-4 (wllama).
17. [Scaling up and troubleshooting](./17-scaling-and-troubleshooting)

## Going further

18. [A Doom-like shooter](./18-doom): a fourth game, played from 14 measured numbers.
19. [The s2 architecture](./19-s2-architecture): two streams with a static cache, permutation equivariance, `[NUM]` value features, early exits, ternary QAT, and why the exit threshold has to be calibrated in closed loop.
20. [What DAgger taught us](./20-dagger-lessons): from 0 to 200 Flappy pipes, round by round, for s1-pico, s1-nano and s2-nano.
