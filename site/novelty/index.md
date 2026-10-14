---
title: What we bring
---

# What skycmd brings

*Updated on 14 October 2026, after three nights of experiments (s3, s4, s5).*

**In short:** a System One decision model, the kind of model Jev, Laya and Julia-1 introduced in September 2026, fits in **251 KB**. It was trained **from scratch in under an hour** on a laptop and plays four control games **at the level of an exact oracle**, in about **50 µs per frame in a web browser**. Around it, skycmd brings a decision format that is invariant to the order and names of the options by construction, a calibration lesson for early exits, experiments whose criteria were fixed before any test, and a whole chain in Rust that runs in the browser. As far as we could find, nothing this small exists in this format, and none of it uses Python.

## 1. A decision model 40× to 300× smaller

Every open decision model we found starts from a pretrained backbone of at least 17M parameters. [s3-pico](/tutorial/21-s3-recursive-tiny) was trained from zero, and it still answers typed questions (choice / score / noul) with calibrated probabilities in one forward pass.

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

You can play against it on the [Play](/play/) page.

## 2. An architecture designed for decisions in a loop

Each piece below is known on its own; the combination is ours.

- **Two streams with a cache:** the question and the options are encoded once and cached, and each frame encodes only the state.
- **Options never see each other,** so the answer cannot depend on their order. This holds by construction, not by augmentation.
- **One token per measured value:** a pair such as `vy -3` becomes a single token, combining FT-Transformer and FoNE.
- **Depth from recursion, not from parameters:** one shared block is looped four times, with an exit after every loop, as in TRM and Ouro.
- **Ternary weights trained in** with QAT, BitNet-style.

## 3. A lesson: calibrate early exits where the model is used

- Early-exit thresholds are normally fitted on i.i.d. validation questions. Fitted that way (τ = 0.62 / 0.62 / 0.69 for choice / score / noul), **s2-nano failed** in closed loop: over 5 arena seeds × 30 episodes it crashed in every Flappy episode (15.9 ± 2.3 pipes of 200, 30 crashes of 30) and needed 1,931 ± 1,106 reflex vetoes in Dodge. Confident shallow exits were wrong in the critical states.
- Calibrated **by playing the games** (τ = 0.95), the same weights flew 191.9 ± 7.2 pipes (2.4 ± 2.1 crashes of 30), matched the oracle in Rescue and came within 0.1 of it in Dodge and Doom (Doom is played by s2-nano-doom, its Doom fine-tune, as on the site), at 3.2 to 4.0 of 4 layers.
- s3-pico, a different model, ships with a τ chosen in closed loop (0.6) and uses about 1.4 of its 4 loops (1.44 Flappy, 1.57 Rescue, 1.05 Dodge, 1.73 Doom). There the threshold barely matters: τ = 0.6, τ = 0.5 and full depth give the same scores and crashes on all four games and all five seeds (Dodge vetoes 9.6 against 9.8 at τ = 0.5). For s3-pico, the closed-loop calibration confirmed that shallow exits were safe rather than rescuing anything.
- So the evidence for the lesson is one model, s2-nano, on five seeds. We found little prior work on calibrating early exits in closed-loop control, and we think it is worth checking on any early-exit model that drives something; this data does not show that it is a general rule. Numbers: [`loopcal.json`](/results/loopcal.json), produced by `cargo xtask loopcal`.

A related recipe: **imitation that holds up in closed loop.** With DAgger, the model plays and the oracle labels the states it reaches. It turned "crashes before the first pipe" into "200/200" in two to three rounds of 8 minutes each ([DAgger chapter](/tutorial/20-dagger-lessons)).

## 4. Decisions that cannot depend on the order or names of the options

The s5 iso format ([chapter 23](/tutorial/23-s5-lite)) gives these properties by construction. Each one is checked by an exact test, natively and in WASM, so none of them is an experimental result:

- permuting the options changes nothing;
- renaming the option keys changes nothing, since the model reads the definitions only (this removes the yes/no swap of 2609.26758);
- s + n = 1 when a negation only swaps the two definitions;
- the state encoding does not depend on the question, so it is cached per state.

At equal data, the same encoder was much more accurate in this format than in the usual one on the fitted bench: 0.674 against 0.429 (+0.245, 95% CI [+0.136, +0.321]). It was less well calibrated, though, and the calibration margin fixed in advance failed (see below).

## 5. Experiments you can check, failures included

- **Criteria fixed before any test.** The s5 [pre-registration](https://github.com/maxgfr/skycmd/blob/main/docs/s5-prereg.md) was committed first. Every amendment is dated and says what had been seen when it was written.
- **The test split is locked by the tooling.** The evaluation tool refuses the test split of a frozen set unless the pre-registration lists the set's hash, and it logs every test pass.
- **Failures are published like successes.** s4 missed its gate, and the three pre-registered s5 bets each missed their criterion. The online escalation bet failed in all six cells. The [Results](/results/) page shows each of them with its interval and the reason.

## 6. The whole chain in Rust, down to the browser

- data from Hugging Face;
- training with [mlx-rs](https://github.com/oxiglade/mlx-rs) on Apple Silicon;
- a GGUF writer, with f32, q8_0 and Q4_0 exports;
- an inference engine compiled to WebAssembly, with SIMD kernels for 8-bit and 4-bit weights.

The same file runs in the browser, in a CLI that speaks the `/v1/systemone` API, in Node and in Python (built from source). Our s1 models also run in llama.cpp and wllama. The s5 model takes 21 MB in q8_0 and 12 MB in Q4_0, against 70 MB in f32.

The [Live](/live/) page runs everything on your machine, with no server: the decision models, and a cascade replay in which s5 answers alone or escalates to an auditor that also audits a random share of the accepted answers.

## 7. Methods worth reusing, even from a failed model

[s4-micro](/tutorial/22-s4-generic-decisions) tried to make the architecture generic by distilling Qwen3.5-4B in one night. It missed its gate (0.343 on the typed-decisions bench, against a target of 0.55), but three of its methods are worth reusing:

- **A scoring teacher with per-group bias calibration.** The 4B never generates: one prefill, the logits of the option letters only. A temperature and a bias per option are then fitted per (source, type, number of options) against the gold labels, which removed a strong "true" bias on nouls.
- **Randomized label sets with abstention.** Options are shuffled, distractors are added, an abstention option is appended, and in 12 % of the records the gold option is removed so that "I don't know" is the target.
- **Deep improvement supervision on early exits.** Each exit trains toward the target raised to a power that grows from 0.5 to 1, so the early exits learn a softer copy of the target.

Its failure is a lesson too: without language pretraining, loops cannot make up for it. A BERT-Tiny of 4.4M parameters fine-tuned on the same data reaches 0.794 on Banking77, against 0.303 for s4.

## What it does not bring (honestly)

- **None of the building blocks is new.** The novelty is in how they are combined, in the scale and in the closed-loop calibration lesson. Our literature search was two research passes, not a systematic review.
- **The games are friendly.** Code measures the useful features (relative clearances, an A* waypoint) and the model decides, following the [jev-drone](https://github.com/RomanSlack/jev-drone) philosophy. A small MLP on the same features would be a fair baseline, and we have not run it yet.
- **The tiny model is a specialist.** On general decisions it is modest: 0.54 on our held-out mix and 0.38 on the external typed-decisions benchmark, against about 0.73 for the 144M Julia-1.
- **No pre-registered claim for s5.** None of the three bets is claimed: the iso format missed its calibration margin, the coherence losses improved only the identities they were trained on, and 8-bit quantization could not be shown non-inferior for lack of power.
- **s5 is a small English model.** There is no French version.
- **The results are not peer-reviewed.** They come from our own benchmarks, with single runs for s3 and s4 and three seeds for s5.

## What would turn this into a paper

- Done: closed-loop vs i.i.d. early-exit calibration over 5 arena seeds (point 3, `cargo xtask loopcal`).
- Ablations of s3 over several seeds: fused values, recursion, static cache, DAgger.
- A standard control benchmark with vector states.
- Baselines: a same-size MLP on the games, an RL policy, oscar-1-17m.
- The s5 baselines B1 (label log-probability through the MLM head) and B3 (a linear head on the frozen backbone); see [Not done yet](/results/#not-done-yet).
- A systematic literature review.

The natural first target is a short workshop paper: *closed-loop calibration of early exits for tiny decision models*, with skycmd as the open-source demonstrator.

See also the [tutorial](/tutorial/), the [night experiments](https://github.com/maxgfr/skycmd/blob/main/docs/night-experiments.md) and the [sources](/sources/).
