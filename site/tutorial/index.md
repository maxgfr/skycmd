---
title: Tutorial
---

# Tutorial

This tutorial rebuilds skycmd from an empty Cargo workspace to the games on the [Play](/play/) page. Every chapter follows the real code in the repository, with excerpts from the crates and the exact commands. Every number comes from a file the chapters name: committed ones (`BENCH.md`, `docs/compat.md`, `docs/night-experiments.md`, `site/public/bench.json`) or ones the commands write on your machine (`runs/`, `data/STATS.md`, both gitignored). A number nobody measured is marked "not measured".

What you end up with:

- **Tiny.** s3-pico, the default model of the site, has 445,345 parameters in a 251,104-byte ternary file. The earlier s1-pico has 202,305 parameters in 545,280 bytes, and s2-nano 1,532,033 parameters in 701,568 bytes (`site/public/models/manifest.json`).
- **Fast.** One Flappy Drone frame, two typed questions plus the physics step, takes 49.6 µs for s3-pico, 409.1 µs for s2-nano and 588.6 µs for s1-pico in WASM under Node (`site/public/bench.json`).
- **Typed.** A model answers `choice`, `score` and `noul` questions with a calibrated probability for every allowed option, in one forward pass, and a coded validator keeps the last word.
- **Plays games.** s3-pico plays all four games at the oracle's level: 200 of 200 Flappy pipes, every survivor rescued, the oracle's 0.867 Dodge completion, and every Doom-like level cleared with the oracle's kill rate (`runs/ship-s3-pico.arena.json`). Before it, DAgger took s1-pico and s2-nano to the oracle's level on the three drone games, and a fine-tuned s2-nano learned the Doom-like shooter.
- **Installs everywhere.** The same `.gguf` file runs in the browser, Node, Deno, Bun, Python, the `skycmd` CLI and server. Only the s1 files also run in llama.cpp and wllama; s2 and s3 files run only in the skycmd engine ([Install](/install/), `docs/compat.md`).

It also says what did not work: a model that crashed in every Flappy episode, an early-exit threshold that broke closed-loop play, a bigger model that did worse, a smaller s2 that never learned to play, and an s3 half the size of s3-pico that still crashes in Flappy.

The pipeline in one picture:

```text
Hugging Face datasets ─┐                       ┌─ sky-infer (native) ── sky-bench, sky-arena, skycmd
game oracles (sky-games)┼─ sky-data ─ sky-train ─ export ─ GGUF ─┤
GLM teacher ────────────┘   (records)  (mlx-rs)                  └─ sky-web (wasm) ── this site
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
