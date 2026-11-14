---
title: Bench
aside: false
pageClass: wide-page
outline: false
---

# Bench

Size, accuracy, closed-loop game results and latency of the eight published models, from `site/public/bench.json`:

- the game models, s3-pico (the default), s1-pico and s2-nano;
- the Doom fine-tune s2-nano-doom, listed from the manifest without metrics;
- the two models for any question, s4-micro and s5-lite, whose held-out scores are in the table below the bench;
- s6-nano, which does both, listed from the manifest: its results are [below](#s6-results);
- s6-secrets-pico, a secret detector that answers only its four questions about a text and fails on real files (it flags 59.8% of clean real files at p ≥ 0.9), listed from the manifest: its results are [below](#s6-results) too, and the browser button measures it on game states it was never trained for, which only says how fast it runs. Accuracy is measured on the held-out test split without the Doom questions (25,319 questions): for s1-pico and s2-nano with NLL and ECE from `runs/s1-pico-g2/report.md` and `runs/s2-nano-g2/report.md` (written by `sky-train eval` and `sky-train s2-eval`), for s3-pico from `BENCH.md` (s3 section). The games are 30 seeded `sky-arena` episodes per game (`runs/ship-*.arena.json`). Native latency is the `sky-arena` median per Flappy decision on the build machine; WASM latency is the median per Flappy frame (2 questions + physics) under Node until you measure your own. A value shown as **n/a** was not measured for that model. Use the button below to measure latency in your own browser.

<ClientOnly>
  <BenchView />
</ClientOnly>

## Decision models for any question

These models answer questions you write yourself, so they are measured on held-out decision sets rather than in the games (`BENCH.md`, s4 and s5 sections; [results page](/results/)).

| model | format | parameters | file | typed-decisions test | notes |
|---|---|---:|---:|---:|---|
| s5-lite (/live) | iso, q8_0 | 17,060,097 | 20.7 MB | 0.696 | 21/30 English demo cases; option order and keys cannot change an answer |
| s6-nano (/live, /play) | s6, ternary, full depth | 1,486,225 | 715 KB | 0.480 | `typed_decisions_bench`; also plays the four games at the oracle's level; ECE 0.0479 |
| s5 B\* finalists (3 seeds) | iso, f32 | 17M | | 0.674 | pre-registered bet 1: more accurate than A′ (0.429) but ECE 0.088 against 0.057 |
| oscar-1-17m (reference) | Laya, f32 | 17M | | 0.682 | the public model of this format, on the same records |
| s4-micro | s2, ternary | 5,587,201 | 2.4 MB | 0.343 | measured on `typed_decisions_bench`, the same test split in the older record layout; experimental, below the majority class (0.398) |

## s6 results

Five s6 models have a pre-registration with its gate. All five are trained and evaluated: two pass and are published, three fail and are not. The s6-pico ablation only chose the recipe of the others, and its rules adopted none of the s6 additions (Muon, Parallax, XSA and the rest): there is no s6-pico model to ship. Every number is on the [results page](/results/#s6-november-2026) and in [`BENCH.md`](https://github.com/maxgfr/skycmd/blob/main/BENCH.md).

| model | task | parameters | gate (pre-registered) | result |
|---|---|---:|---|---|
| **s6-nano** | general-purpose: the four games and multiple-choice questions in free text, 2,048 tokens, 512-token state ([SPEC §15](https://github.com/maxgfr/skycmd/blob/main/docs/SPEC-S6.md)) | 1,486,225 | above the majority class on `typed_decisions_bench` (> 0.398), no game more than 5 % below s3-pico, ECE ≤ 0.08 ([pre-registration](https://github.com/maxgfr/skycmd/blob/main/docs/s6-gen-prereg.md)) | **passed**: 0.480, every game at s3-pico's score, ECE 0.0479; published at full depth (the conformal exit thresholds were rejected) |
| **s6-secrets-pico** | API-key and secret detection in short texts: is a secret exposed, which service, what kind of value, which line; s3-pico graph with a 320-token state ([SPEC §14](https://github.com/maxgfr/skycmd/blob/main/docs/SPEC-S6.md)) | 445,345 | beat a gitleaks-style rule scanner on a synthetic bench of unseen file types: > 0.977 on "is a secret exposed?" and > 0.787 on "which service?", ternary, full depth ([pre-registration](https://github.com/maxgfr/skycmd/blob/main/docs/s6-secrets-prereg.md)) | **passed**: secret 0.9990, service 0.9795, kind 0.9990, line 0.6859 (weak); 250 KB, published at full depth. The bench is synthetic: **on real files it fails**, flagging 59.8% of clean files at p ≥ 0.9 and 49.4% at p ≥ 0.999, so it is a result, not a scanner ([real text](/results/#s6-secrets-real-text), [Live](/live/#s6-secrets-pico-is-a-secret-exposed-in-this-text)) |
| s6-micro | the same at s4-micro scale, 4,096 tokens, 384-token state | 5,587,201 | ≥ 0.45 on `typed_decisions_bench` (goal 0.50, the level of s4's Qwen3.5-4B teacher), no game more than 5 % below s3-pico, ECE ≤ 0.08 | **failed on calibration only**: 0.547 (gate and goal met), games at s3-pico's score, but ECE 0.0821 at full depth and 0.0804 at its conformal thresholds; not published |
| s6-ear-nano | an s6 that listens: 10 s of 16 kHz audio as 80 ms log-mel frames, typed questions in text, pretrained to predict Mimi's semantic tokens ([SPEC §16](https://github.com/maxgfr/skycmd/blob/main/docs/SPEC-S6.md)) | 1,532,449 | end of turn at least as accurate as Smart Turn v3.2 (8M) on the same clips, ≥ 0.90 on Speech Commands (12 classes), ≥ 2 × chance on held-out MSWC words in English and French ([pre-registration](https://github.com/maxgfr/skycmd/blob/main/docs/s6-ear-prereg.md)) | **failed**: end of turn 0.5681 against 0.8518, keywords 0.8885, French zero-shot 0.1330 (English 0.2100 passes); not published |
| s6-cast-pico | probabilistic forecasting: 2,048 past points in patches of 32, nine deciles for each of the next 64 steps ([SPEC](https://github.com/maxgfr/skycmd/blob/main/docs/SPEC-S6-CAST.md)) | 369,792 | GIFT-Eval nGMASE ≤ 0.838 and nWQL ≤ 0.587, the training-free FLAIR; the goal is TinyCast's 0.774 / 0.545 ([pre-registration](https://github.com/maxgfr/skycmd/blob/main/docs/s6-cast-prereg.md)) | **failed**: nGMASE 0.9471, nWQL 0.6646 (0.9099 and 0.6417 with flip); not published ([results](/results/#s6-cast-pico-failed)) |

s6-nano, s6-micro and s6-secrets-pico pretrain on the s3 text plus the non-web parts of the training data of OLMo 3 (Dolma 3) and LLM360 (TxT360): Wikipedia, science PDFs, maths, public-domain books, StackExchange and French Europarl ([sources](/sources/)). s6-cast reads no text, and s6-ear hears its state: it pretrains on LibriSpeech audio against the published Mimi codes.

## Reproduce

```bash
# Closed-loop games and native latency per decision, 30 episodes per game (seed 7)
cargo run --release -p sky-arena -- site/public/models/s3-pico.gguf --episodes 30 \
  --game flappy,rescue,dodge,doom --json runs/ship-s3-pico.arena.json

# Native latency of one model on a Flappy frame (sky-bench)
export CARGO_TARGET_DIR=$PWD/target/sky-infer
cargo run --release -p sky-infer --bin sky-bench -- site/public/models/s3-pico.gguf --questions 2 --iters 4000

# WASM latency per Flappy frame under Node (the modules of the Play page: SIMD128 by default,
# --relaxed for the relaxed-SIMD build)
scripts/build-wasm.sh
node crates/sky-web/bench/bench-node.mjs site/public/models/*.gguf
node crates/sky-web/bench/bench-node.mjs --relaxed site/public/models/*.gguf
```

The numbers in `site/public/bench.json` come from these measurements (its `note` field says which). `scripts/export-models.sh` regenerates it only for s1 models in the Laya layout, so do not run it to refresh this page: it would drop s3-pico and s2-nano.

The in-browser button loads each published model and times 300 decisions with the same wasm build as the Play page: relaxed SIMD where the browser supports it, else SIMD128 (the line under the button says which). The WASM numbers of the table come from `site/public/bench.json` and were measured with the SIMD128 build.
