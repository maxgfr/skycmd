---
title: Bench
aside: false
pageClass: wide-page
outline: false
---

# Bench

Size, accuracy, closed-loop game results and latency of the published models: s3-pico (the default), s1-pico and s2-nano, from `site/public/bench.json`; the Doom fine-tune s2-nano-doom is listed from the manifest without metrics. Accuracy is measured on the held-out test split without the Doom questions (25,319 questions): for s1-pico and s2-nano with NLL and ECE from `runs/s1-pico-g2/report.md` and `runs/s2-nano-g2/report.md` (written by `sky-train eval` and `sky-train s2-eval`), for s3-pico from `BENCH.md` (s3 section). The games are 30 seeded `sky-arena` episodes per game (`runs/ship-*.arena.json`). Native latency is the `sky-arena` median per Flappy decision on the build machine; WASM latency is the median per Flappy frame (2 questions + physics) under Node until you measure your own. A value shown as **n/a** was not measured for that model. Use the button below to measure latency in your own browser.

<ClientOnly>
  <BenchView />
</ClientOnly>

## Decision models for any question

These models answer questions you write yourself, so they are measured on held-out decision sets rather than in the games (`BENCH.md`, s4 and s5 sections; [results page](/results/)).

| model | format | parameters | file | typed-decisions test | notes |
|---|---|---:|---:|---:|---|
| s5-lite (/live) | iso, q8_0 | 17,060,097 | 20.7 MB | 0.696 | 21/30 English demo cases; option order and keys cannot change an answer |
| s5 B\* finalists (3 seeds) | iso, f32 | 17M | | 0.674 | pre-registered bet 1: more accurate than A′ (0.429) but ECE 0.088 against 0.057 |
| oscar-1-17m (reference) | Laya, f32 | 17M | | 0.682 | the public model of this format, on the same records |
| s4-micro | s2, ternary | 5,587,201 | 2.4 MB | 0.343 | measured on `typed_decisions_bench`, the same test split in the older record layout; experimental, below the majority class (0.398) |

## Reproduce

```bash
# Closed-loop games and native latency per decision, 30 episodes per game (seed 7)
cargo run --release -p sky-arena -- site/public/models/s3-pico.gguf --episodes 30 \
  --game flappy,rescue,dodge,doom --json runs/ship-s3-pico.arena.json

# Native latency of one model on a Flappy frame (sky-bench)
export CARGO_TARGET_DIR=$PWD/target/sky-infer
cargo run --release -p sky-infer --bin sky-bench -- site/public/models/s3-pico.gguf --questions 2 --iters 4000

# WASM latency per Flappy frame under Node (the module of the Play page)
scripts/build-wasm.sh
node crates/sky-web/bench/bench-node.mjs site/public/models/*.gguf
```

The numbers in `site/public/bench.json` come from these measurements (its `note` field says which). `scripts/export-models.sh` regenerates it only for s1 models in the Laya layout, so do not run it to refresh this page: it would drop s3-pico and s2-nano.

The in-browser button loads each published model and times 300 decisions with the same wasm build as the Play page.
