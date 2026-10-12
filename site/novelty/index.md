---
title: What's new
---

# What's new here

*Written on 12 October 2026, after the first night of experiments.*

**In short:** a System One decision model, the kind of model Jev, Laya and Julia-1 introduced in September 2026, fits in **251 KB**. It was trained **from scratch in under an hour** on a laptop, and it plays four control games, including a Doom-like shooter, **at the level of an exact oracle**, in about **50 µs per frame in a web browser**. As far as we could find, nothing this small exists in this format, and none of it uses Python.

## The numbers

| | skycmd s3-pico | smallest public model in the same format | typical decision models |
|---|---|---|---|
| parameters | **445,345** | 17M ([oscar-1-17m](https://huggingface.co/mgoeckel/oscar-1-17m)) | 144M ([Julia-1](https://huggingface.co/SupersonicLabs/Julia-1)) to 27B |
| file | **251 KB** (ternary) | ~37 MB | hundreds of MB to tens of GB |
| training | **from scratch, ~47 min on an Apple M5** | fine-tuned from a pretrained encoder | fine-tuned from large pretrained models |
| per decision | **21 µs native, ~50 µs per frame in WASM** | milliseconds | milliseconds to seconds (Jev: 70–500 ms, vendor figure) |

What s3-pico does with that budget, over 30 seeded episodes per game, with the shipped file (`runs/ship-s3-pico.arena.json`):
- **Flappy Drone:** 200/200 pipes, 0 crashes.
- **Rescue Run:** 100% of hikers rescued.
- **Obstacle Dodge:** 87% of courses completed, exactly the oracle's rate. The coded safety reflex had to step in only 14 times, against 1,026 for our first model.
- **Doom-like:** 100% of levels completed and the same kill rate as the oracle, with no deaths.
- **Depth used:** 1.05 to 1.73 loops out of 4, thanks to early exits.

## What is new

1. **Size: 40× to 300× smaller.** Every open decision model we found starts from a pretrained backbone of at least 17M parameters. s3-pico is about 40 times smaller than the smallest one and about 300 times smaller than Julia-1. It was trained from zero, and it still answers typed questions (choice / score / noul) with calibrated probabilities in one forward pass.
2. **An architecture designed for decisions in a loop.** Each piece below is known on its own; the combination is ours.
   - **Two streams with a cache:** the question and the options are encoded once and cached, and each frame encodes only the state.
   - **Options never see each other,** so the answer cannot depend on their order. This holds by construction, not by augmentation.
   - **One token per measured value:** a pair such as `vy -3` becomes a single token, combining FT-Transformer and FoNE.
   - **Depth from recursion, not from parameters:** one shared block is looped four times, with an exit after every loop, as in TRM and Ouro.
   - **Ternary weights trained in** with QAT, BitNet-style.
3. **A lesson worth sharing: calibrate early exits where the model is used.**
   - Early-exit thresholds are normally fitted on i.i.d. validation questions. Fitted that way (τ = 0.62), our model **failed** in closed loop: confident shallow exits were wrong exactly in the critical states.
   - Calibrated **by playing the games** instead, the same model reached oracle level while using about 1.3 of its 4 loops.
   - We found little prior work on calibrating early exits in closed-loop control. It looks like a general rule for any early-exit model that drives something.
4. **The whole chain in Rust, end to end:**
   - data from Hugging Face;
   - training with [mlx-rs](https://github.com/oxiglade/mlx-rs) on Apple Silicon;
   - a GGUF writer;
   - an inference engine compiled to WebAssembly.

   The same file runs in the browser, in a CLI that speaks the `/v1/systemone` API, from npm and from Python. Our s1 models also run in llama.cpp and wllama.
5. **Imitation that holds up in closed loop.** DAgger, where the model plays and the oracle labels the states it reaches, turned "crashes before the first pipe" into "200/200" in two to three rounds of 8 minutes each.

## What it is not (honestly)
- **None of the building blocks is new.** The novelty is in how they are combined, in the scale and in the closed-loop calibration lesson. Our literature search was two research passes, not a systematic review.
- **The games are friendly.** Code measures the useful features (relative clearances, an A* waypoint) and the model decides, following the [jev-drone](https://github.com/RomanSlack/jev-drone) philosophy. A small MLP on the same features would be a fair baseline, and we have not run it yet.
- **It is a specialist.** On general decisions it is modest: 0.54 on our held-out mix and 0.38 on the external typed-decisions benchmark, against about 0.73 for the 144M Julia-1.
- **The results are not peer-reviewed.** They come from single runs on our own benchmarks, with a shared GPU during training.

## What would turn this into a paper
- Ablations over several seeds: fused values, recursion, static cache, closed-loop vs i.i.d. calibration, DAgger.
- A standard control benchmark with vector states.
- Baselines: a same-size MLP, an RL policy, oscar-1-17m.
- A systematic literature review.

The natural first target is a short workshop paper: *closed-loop calibration of early exits for tiny decision models*, with skycmd as the open-source demonstrator.

See also the [s3 chapter](/tutorial/21-s3-recursive-tiny), the [DAgger chapter](/tutorial/20-dagger-lessons), the [night experiments](https://github.com/maxgfr/skycmd/blob/main/docs/night-experiments.md) and the [sources](/sources/).
