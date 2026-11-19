---
license: apache-2.0
language:
  - en
  - fr
library_name: gguf
pipeline_tag: text-classification
tags:
  - gguf
  - decision-model
  - system-one
  - ternary
  - memory
  - fast-weights
  - rust
  - skycmd
  - failed-gate
  - negative-result
---

# s6-sleep3-nano

> **Failed its pre-registered gate: published as a result, not for use.** Of its four hypotheses,
> two hold and two fail. **H0, the memory is used, holds:** on the test episodes of the `recall`
> probe it is right on **0.4892** of the questions, against **0.2440** for the same training
> without a memory (+0.2452, 95% interval +0.2370 to +0.2532; at least +0.10 asked). **H1, sleep
> helps, is not shown:** four sleep passes against one add **+0.0030** (95% interval −0.0010 to
> +0.0068), where a lower bound above 0 was asked. **H2, the games do not lose, is not shown:**
> against the control without memory, Flappy **15.87 against 71.05** (−77.7%) and Dodge **5.89
> against 7.94** (−25.8%), where at most 2% under was asked. H3 holds (`typed_decisions_bench`
> 0.4897 against 0.4808). The exported file also **fails the s6-nano gate** on the same two games
> (Flappy 2.10 against s3-pico's 200.00, Dodge 6.90 against 9.47). The engine matches the training
> code natively and in WebAssembly. The file is published so that the result can be checked and
> run (owner's decision of 15/11/2026); the verdict does not change. Numbers:
> [BENCH.md, section s6-sleep3-nano](https://github.com/maxgfr/skycmd/blob/main/BENCH.md#s6-sleep3-nano-s6-sleep-v3-a-fast-weight-memory-pre-registered-gate-failed)
> and [`runs/s6-sleep3/REPORT.md`](https://github.com/maxgfr/skycmd/blob/main/runs/s6-sleep3/REPORT.md).

**A 752 KB ternary decision model with a fast-weight memory.** 1,611,505 parameters: the s6-nano
graph (d 144, 2 state blocks looped twice, exits after 1 to 4 block passes, vocabulary 2,048) with
one memory layer after each state block. A memory layer holds a small matrix per head that each
frame of a stream writes into once and every later frame reads from, so a fact seen on one frame
can change an answer many frames later. "Sleep" replays the last 8 frames N − 1 more times while
no decision is asked (*Do Language Models Need Sleep?*, arXiv 2605.26099); N = 1 is no sleep, and
N = 1 is what the pre-registered rule chose. It is the third version of s6-sleep, and the first to
reach a test reading.

- **Demo:** the [Memory page](https://maxgfr.github.io/skycmd/memory/) of the project site and the
  Space [{{USER}}/s6-sleep3-nano](https://huggingface.co/spaces/{{USER}}/s6-sleep3-nano) play
  `recall` episodes in the browser with this file and its control, with the memory on or zeroed,
  and put 200 episodes next to the test reading. The pre-registration's demo was conditioned on
  H0 only (v2 §8); every published model gets a Space (owner's decision of 17/11/2026).
- **Code:** [github.com/maxgfr/skycmd](https://github.com/maxgfr/skycmd) (100 % Rust)
- **Pre-registration:** [docs/s6-sleep3-prereg.md](https://github.com/maxgfr/skycmd/blob/main/docs/s6-sleep3-prereg.md)
  (v3: two changes to v2 and one more check, amendments 1 to 3, the result block), on top of
  [docs/s6-sleep2-prereg.md](https://github.com/maxgfr/skycmd/blob/main/docs/s6-sleep2-prereg.md)
  (the memory, the pilot, the nano runs, the gate) and
  [docs/s6-sleep-prereg.md](https://github.com/maxgfr/skycmd/blob/main/docs/s6-sleep-prereg.md)
  (the probes, the hypotheses H0 to H3, the publication gate)
- **All results, failures included:** [results page](https://maxgfr.github.io/skycmd/results/#s6-sleep3-nano-failed)

## The gate, as read

`runs/s6-sleep3/gate.json`: `"pass": false`. Every reading is on test seeds read once (the first
1,000 `recall` episode seeds ≡ 19 mod 20 from 5,000,000,000; the arena from seed 5000, 200
episodes per game), ternary weights, the last exit, the mean of three training seeds; intervals
are paired bootstraps over episodes, the three seeds pooled.

| condition | measured | bar | verdict |
|---|---|---|---|
| H0, the memory is used: `recall` at N = 1 minus the control | **+0.2452** [+0.2370, +0.2532] (0.4892 against 0.2440) | ≥ 0.10 | holds |
| H1, sleep helps: `recall` at N = 4 minus N = 1 | **+0.0030** [−0.0010, +0.0068] | lower bound above 0 | not shown |
| H2, the games do not lose (N = 1, against the control) | Flappy 15.87 against 71.05 (−77.7%); Rescue 3.00 against 3.00 (−0.1%); Dodge 5.89 against 7.94 (−25.8%); Doom 3.37 against 3.37 (+0.1%) | no game more than 2% under | not shown |
| H3, the language tasks do not lose | `typed_decisions_bench` 0.4897 against 0.4808 | at most 0.01 under | holds |
| the s6-nano gate, on this file (`sky-eval` and `sky-arena`, full depth, 30 episodes per game) | `typed_decisions_bench` 0.499, ECE 0.0578; Flappy 2.10 against s3-pico's 200.00, Dodge 6.90 against 9.47, Rescue 3.00 and Doom 3.27 equal | above 0.398, ECE ≤ 0.08, no game more than 5% under s3-pico | fails (Flappy, Dodge) |
| parity of `sky-infer` with the training code on golden sequences | natively: true; in WASM: true | both | holds |

H1b, the same reading on the secondary probe `rule110`, decides nothing: −0.0001 [−0.0007,
+0.0005].

## Recall by delay

Accuracy on the questions of `recall` by the number of frames since the asked door was shown (the
three seeds pooled, `runs/s6-sleep3/REPORT.md`; a model without memory is at about 0.25):

| reading | 1-4 | 5-8 | 9-16 | 17-32 | door shown one colour | several colours |
|---|---:|---:|---:|---:|---:|---:|
| control, no memory | 0.2420 | 0.2398 | 0.2518 | 0.2423 | 0.2433 | 0.2446 |
| N = 1 | 0.6147 | 0.5590 | 0.4530 | 0.3302 | 0.4717 | 0.5052 |
| N = 2 | 0.6213 | 0.5823 | 0.4693 | 0.3083 | 0.4657 | 0.5224 |
| N = 4 | 0.6290 | 0.5910 | 0.4620 | 0.2867 | 0.4602 | 0.5214 |

The memory is real and fades with the delay; sleep gains a little at short delays and loses at
the longest. This file, seed 0, reads 0.4765 overall at N = 1 (0.591, 0.5375, 0.438, 0.3395 by
delay), and its control 0.245125 (`runs/s6-sleep3/test/sleep-s0/recall.json`,
`control-s0/recall.json`). At the pico size the pilot of this version answered 1.0000 at every
delay on its development episodes (control 0.2542); why the nano size reaches about half of that
is not known.

## What it does, and does not

- **Use:** none. It is the result of an experiment about memory, not a game model and not a
  general decision model: it loses two of the four games against the same training without a
  memory, and its file fails the s6-nano gate.
- **What it shows:** a ternary decision model can be trained to use a fast-weight memory across
  frames, here with held gates and an auxiliary recall loss in training (below); the gain is
  +0.2452 on `recall`, mostly at short delays.
- **What it does not show:** that sleep passes help (H1), that the memory leaves the games alone
  (H2), or anything outside the `recall` probe and this size.
- **Limits found after the result:** the runs ended halfway up their learning curve (393 `recall`
  training steps at seed 0 against the pico pilot's 933, accuracy still rising), the auxiliary
  loss as implemented has one head per memory layer reading the mean over the frame's tokens,
  averaged over the two layers (not the last layer at the door token, as described under
  Training), and the memory was never read zeroed at this size, so H0 rests on the control alone
  ([docs/s6-sleep4-diagnosis.md](https://github.com/maxgfr/skycmd/blob/main/docs/s6-sleep4-diagnosis.md)
  and the note of 17/11 at the end of the pre-registration).
- **The fourth version's first check failed (18/11/2026):** K3 trained the memory with the base
  frozen, at the nano size on training sequences, and `recall` stayed at 0.2579 and 0.2546 over
  the last 100 `recall` steps of seeds 0 and 1, against a bar of 0.95 and a chance of 0.25
  ([docs/s6-sleep4-prereg.md](https://github.com/maxgfr/skycmd/blob/main/docs/s6-sleep4-prereg.md)
  §15). Whether v4 goes on is the owner's decision; nothing about this file changes.

## How to ask it

The model reads one frame at a time; a stream keeps the memory between frames. On `recall`, one
`choice` question:

- **instruction**, word for word: `Which key opens the door that is asked for?`
- **options**: `wait: no door is asked`, `red: the door was given red`,
  `green: the door was given green`, `blue: the door was given blue`,
  `yellow: the door was given yellow`
- **state**: `show <door|none> colour <colour|none> ask <door|none> noise <0-99>`, for example
  `show b colour red ask none noise 32`, then `show none colour none ask b noise 1` a few frames
  later.

These are the canonical question and options of `sky_games::recall`, which the evaluation used.
A new episode is a new stream (a zero memory); every frame is a new frame of the stream, written
into the memory when the next one starts.

**Depth.** The file stores the exit thresholds of its export (0.5 for each question type). Every
reading of the gate was made at full depth, the last exit, with an override (`--exit-threshold
1.0`, τ = 1); use the same.

## Running it outside the browser

- **The `recall` probe under Node,** with the WebAssembly module of the site (the same code as the
  demo, `RecallProbe` in `crates/sky-web/src/recall.rs`):

  ```bash
  scripts/build-wasm.sh
  node crates/sky-web/bench/recall-node.mjs --episodes 200 --zeroed \
    s6-sleep3-nano.gguf s6-sleep3-nano-control.gguf
  ```

  It prints, per file and memory mode, the accuracy overall and by delay bucket on episodes of the
  validation split (from seed 7,000,000,018), which no training used: not the test seeds.
- **The four games,** as the s6-nano gate read them:

  ```bash
  cargo run --release -p sky-arena -- s6-sleep3-nano.gguf --episodes 30 --game all \
    --exit-threshold 1.0 --memory-passes 1
  ```

- **In Rust,** `sky_infer::Engine`: `set_sleep(SleepOptions { passes, memory_off, .. })`,
  `new_frame(stream, pause)` before each frame, `decide_raw_stream(stream, …)` for its question,
  `reset_stream(stream)` for a new episode (`crates/sky-arena/src/main.rs` does exactly this).

## Training

From the checkpoint behind s6-nano (`runs/s6-nano-g3/decide/last`), in Rust (mlx-rs) on an Apple
M5, in a throttled mode, so the budgets are working minutes. Two arms, three seeds each (0, 1, 2):
this model and the control, which has no memory and sees the same batches for the same number of
steps. The sleep arm at seed 0 trained 12 + 18 working minutes, 292 and 877 steps; the five other
runs trained its steps, phase by phase. Ternary QAT from the first step, batch 128, AdamW, the
memory's weights at 10 times the base rate.

- **Phase A** (292 steps, N = 1): 90% of the steps are batches of 16 whole sequences, `recall`
  0.85 and `rule110` 0.15; at seed 0, 267 of them (233 on `recall`).
- **Phase B** (877 steps, N drawn in {1, 2, 3, 4}, gradient through the last pass): 50% sequence
  batches, `recall` 0.40, `rule110` 0.12 and each of the four games 0.12.
- **The other batches** are s6-nano's stateless decision training (decisions 0.25, GLM-5.3-flash
  labels 0.30, game states 0.35, numeric 0.10), answered with a zero memory.
- **The two changes of v3:** the write and decay gates are held at their initial values (β = 0.5,
  α = σ(6) ≈ 0.9975), and an auxiliary loss in training only asks a linear head to decode the
  asked colour from what the last memory layer retrieves at each `recall` question. The head is
  not in the file and plays no role at inference.

**Data.** The sequences (`runs/s6-sleep/seq/`: 8,950 `recall` training episodes of 64 frames,
17,959 `rule110` episodes of 8 frames, 4,513 to 4,581 sequences of 24 frames per game) are generated
procedurally by the project's game cores (`sky-games`, Apache-2.0), each frame labelled by the
game's exact oracle; the episodes whose seeds fall in the validation and test splits are not
trained on. No external data and no model output is in them. The stateless batches are s6-nano's
decision data: typed-decision sets of Hugging Face, classic datasets converted to typed
questions, game states and a numeric curriculum, with soft labels from GLM-5.3-flash (open
weights, MIT, called through a hosted endpoint) kept only where they agree with the gold answer.
Every source and its licence: [DATA_LICENSES.md](https://github.com/maxgfr/skycmd/blob/main/DATA_LICENSES.md).

**How it got here.** v1 (pico pilot, 14/11/2026) found its memory never used: `recall` 0.2460
against 0.2650 for the control, and the diagnosis found the code sound and the memory untrained
([`docs/s6-sleep-diagnosis.md`](https://github.com/maxgfr/skycmd/blob/main/docs/s6-sleep-diagnosis.md)).
v2 changed how the memory starts and is first trained (one shared query and key projection,
values that carry the frame, ten times the rate, a first phase of `recall`) and ended at its CPU check C4 without any GPU run: 300 training steps did not
learn to use the memory
([`docs/s6-sleep2-study.md`](https://github.com/maxgfr/skycmd/blob/main/docs/s6-sleep2-study.md)).
A second diagnosis, on train sequences only, found that the write gate closes (its mean falls
from 0.96 at step 7 to 3e-7 at step 35) and that a live gradient alone is not enough
([`docs/s6-sleep3-diagnosis.md`](https://github.com/maxgfr/skycmd/blob/main/docs/s6-sleep3-diagnosis.md),
[`docs/research-fast-weight-memory.md`](https://github.com/maxgfr/skycmd/blob/main/docs/research-fast-weight-memory.md)):
v3 holds the gates and adds the auxiliary loss. Its pico pilot then reached 1.0000 on `recall` at
every delay (development seeds), and the nano runs followed.

**Disclosure** (§5 of v3). This is the third version of the same idea on the same games, and its
two changes were found by looking at training curves on train sequences. The auxiliary loss gives
the model the asked colour as a training target at the question frames; it is not available at
inference. The gates are constants in this model. The development seeds and the test seeds were
read by no earlier version. Deviations, all in the amendments: v2's check C2 missed its bar by
0.010 and was taken as a stated deviation; the first nano run was stopped by the machine's memory
guard and restarted from scratch with the same seed and a computation in parts (`--sleep-chunk
2`, the same batches and updates up to float rounding); the pilot's trainer was built at a later commit
than amendment 1's, with no change in the training code, and the nano runs' trainer at amendment
3's, whose only change to the training code is that option, off by default.

## Files

- `s6-sleep3-nano.gguf`: the sleep arm, seed 0, the file the gate read: ternary weights, the memory
  (`mem.{l}.qk`, `v`, `out` ternary; the norm and the gates f32), tokenizer, temperatures, exit
  thresholds at 0.5. 752,416 bytes, sha256
  `e59f837cc4ee65f35dbebdab91d6ff3348a4fe5ee9a323fe82457eedf4c23234`.
- `s6-sleep3-nano-control.gguf`: the control arm, seed 0, no memory (1,486,225 parameters), for
  comparison only. 715,136 bytes, sha256
  `cd1813207b9992270b0f634e34e0104bfc3e6b6f0fe13e07bd06bfdf00c09100`.

The result files are on GitHub: [`runs/s6-sleep3/`](https://github.com/maxgfr/skycmd/tree/main/runs/s6-sleep3)
(`REPORT.md`, `verdict.json`, `gate.json`, `n.json`, `pilot.json`, `gate/`, and the test readings
of the games and of `typed_decisions_bench` for the six models, with the `recall` readings of the
two published files).
