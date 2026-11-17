---
title: 'Memory: s6-sleep3-nano (failed gate)'
outline: [2, 2]
---

# Memory: s6-sleep3-nano (failed gate)

::: danger Failed its pre-registered gate: published as a result, not for use
**s6-sleep3-nano failed its gate.** Two of its four hypotheses hold, and the one about its own question is that **the memory is used**. On the test episodes of the `recall` probe it is right on **0.4892** of the questions, against **0.2440** for the same training without a memory (+0.2452, 95% interval +0.2370 to +0.2532; at least +0.10 was asked). The rest fails: **sleep does not help** (four sleep passes against one: +0.0030, interval −0.0010 to +0.0068, where a lower bound above 0 was asked), and **two of the four games lose** against the control without memory (Flappy 15.87 against 71.05, Dodge 5.89 against 7.94; at most 2% under was asked). The exported file also fails the s6-nano gate on the same two games (Flappy 2.10 against s3-pico's 200.00, Dodge 6.90 against 9.47). The file is published so that the result can be checked and run (owner's decision of 15/11/2026); the verdict does not change. It is a research result, not a part to use.
:::

**s6-sleep3-nano** is s6-nano with a fast-weight memory: a small matrix, one per memory layer, that each frame of a game writes into and every later frame reads from. It is a 752 KB ternary file (1,611,505 parameters); the control, trained the same way with no memory, is the 715 KB file loaded next to it. "Sleep" means that every 8 frames the model replays them N − 1 more times while no decision is asked, as in *Do Language Models Need Sleep?* (arXiv 2605.26099); N = 1 is no sleep.

**The probe.** A `recall` episode has 64 frames and three doors, `a`, `b` and `c`. On a show frame a door is given one of four colours, for that frame only. On a question frame a door is named, and the right answer is the colour it was last given; nothing on that frame says it. Every other frame carries only a distractor number, and its right answer is `wait`. Each episode asks two questions in each delay bucket: 1 to 4, 5 to 8, 9 to 16 and 17 to 32 frames between the show and the question. A model that sees one frame at a time is right about one time in four.

**What the test says, by delay** (the three training seeds pooled, `runs/s6-sleep3/REPORT.md`):

| delay since the door was shown | 1-4 frames | 5-8 | 9-16 | 17-32 |
|---|---:|---:|---:|---:|
| s6-sleep3-nano, N = 1 | 0.6147 | 0.5590 | 0.4530 | 0.3302 |
| N = 4 (sleep) | 0.6290 | 0.5910 | 0.4620 | 0.2867 |
| control, no memory | 0.2420 | 0.2398 | 0.2518 | 0.2423 |

The memory is real and it fades with the delay. At the smaller pico size, the pilot of this version had answered 1.0000 at every delay on its development episodes (against 0.2542 for its control); at the nano size the same recipe reaches about 0.49. Why is not known yet.

<ClientOnly>
  <MemoryDemo />
</ClientOnly>

## What you are looking at

- **The episodes are the evaluation's.** The page plays `recall` with the generator of the game cores, `sky_games::make_game("recall", seed)`, which is what the evaluation called, with the same question, the same five answers, the model's answer read at its last exit and each question scored by the game itself. It runs in the `RecallProbe` class of `crates/sky-web/src/recall.rs`, compiled to WebAssembly, and `cargo test -p sky-web --test recall` checks it against the game frame by frame.
- **What differs from the test.** The test read 1,000 episodes from seed 5,000,000,000 with the training code, on the checkpoint each file was exported from. The page reads the exported files with the engine that ships, `sky-infer`, whose memory matches the training code on golden sequences, natively and in WebAssembly (the gate's parity check). Its 200 episodes are other seeds, from the validation split, which no training used, so its shares differ from the test's by sampling noise: about ±0.05 per delay bucket.
- **Memory on, zeroed, control.** "Memory on" is the model as evaluated. "Memory zeroed" is the same file with its memory reset to zero before every frame, the engine's own switch: what is left is what the model guesses without remembering. "Control file" is the other file, trained with no memory at all.
- **Sleep passes.** N = 1, 2 or 4 changes how many times each group of 8 frames is replayed; the test read the same three values. N = 1 was chosen on the development episodes, by the pre-registered rule (the smallest N within 0.01 of the best mean: 0.4823 at N = 1, 0.4911 at N = 2, 0.4857 at N = 4).
- **The time.** Per decision, in this browser.

## What the result does not say

- **It is not a model to use.** On the four games it was asked to keep, it loses two: Flappy 15.87 against 71.05 for the control (−77.7%) and Dodge 5.89 against 7.94 (−25.8%); Rescue (3.00 against 3.00) and the Doom-like (3.37 against 3.37) are within 0.1%. On `typed_decisions_bench` it does not lose (0.4897 against 0.4808 for the control).
- **Sleep did nothing measurable.** Four passes against one: +0.0030 overall, and at the longest delays four passes score under one (0.2867 against 0.3302).
- **One probe, one size.** The memory was trained and tested on one synthetic probe, at one size, with one run per seed.

Every number: [its section of Results](/results/#s6-sleep3-nano-failed). The model: [its card in the catalogue](/models/#s6-sleep3-nano) and [its model page on Hugging Face](https://huggingface.co/maxgfr/s6-sleep3-nano). The pre-registration and its result block: [`docs/s6-sleep3-prereg.md`](https://github.com/maxgfr/skycmd/blob/main/docs/s6-sleep3-prereg.md), on top of [`docs/s6-sleep2-prereg.md`](https://github.com/maxgfr/skycmd/blob/main/docs/s6-sleep2-prereg.md) and [`docs/s6-sleep-prereg.md`](https://github.com/maxgfr/skycmd/blob/main/docs/s6-sleep-prereg.md).
