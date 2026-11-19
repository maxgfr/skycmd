---
title: "A 752 KB model that remembers, and still fails its gate: three versions of a fast-weight memory"
description: "A fast-weight memory with sleep passes, added to a tiny ternary decision model: a first version whose memory was never used, a second stopped by its own CPU checks before any GPU time, and a third whose memory is used on its probe and fades with the delay, while sleep adds nothing measurable and two games lose, so the gate fails."
date: 2026-11-17
outline: [2, 2]
---

# A 752 KB model that remembers, and still fails its gate: three versions of a fast-weight memory

*17 November 2026*

A decision model in this project reads one frame of a game, answers, and forgets the frame. I wanted to know whether a memory could change that under a megabyte, and whether letting the model "sleep" on what it has seen improves it. The third of three versions remembers, measurably, and still, rightly, fails the test I wrote for it.

::: danger The verdict: failed
**s6-sleep3-nano failed its pre-registered gate.** Its memory is used: on its probe's test episodes it answers 0.4892 of the questions right, against 0.2440 for the same training without a memory. But sleep adds nothing measurable (+0.0030), and the model plays two of its four games worse with it. The file is published as a result, not as a part to use.
:::

## The context: skycmd

skycmd gathers very small "decision models" with ternary weights (each is −1, 0 or +1), written in Rust, run natively or in the browser through WebAssembly, and trained on one MacBook (Apple M5) in a braked mode, so budgets count working minutes.

::: info Pre-registration and the gate
Before an experiment runs, I commit the test, the bar to clear and what a failure would mean. The bar is the "gate": conditions that must all hold. The test is read once, a failure is published like a success, and the rule cannot be adjusted afterwards.
:::

Two more terms. A control arm is the same training without the thing tested, here the memory, so any difference belongs to the memory. Episodes come from numbered seeds: dev seeds are read to take decisions, test seeds once, at the end.

## The idea: a memory, and time to sleep on it

s6-nano, the base model, answers each frame alone. The paper *Do Language Models Need Sleep?* ([arXiv 2605.26099](https://arxiv.org/abs/2605.26099)) gives a model a fixed-size memory of fast weights: a small matrix the model writes into as it reads and reads back later, unlike its ordinary weights, which change only in training. It also lets the model rerun its layers over what it just read while no answer is asked; there, accuracy rises with the number of passes.

Does that hold for a ternary model of 1.6 million parameters reading game frames? A sleep pass, here: every 8 frames, the model replays them N − 1 more times, writing to its memory, with no decision asked. N = 1 means no sleep.

## The probe: a question with a known answer

The model's four games might never need more than one frame, and then a memory would show nothing. So the work began with a probe: a game built to measure one thing, with a known right answer at every frame.

It is called `recall`. An episode has 64 frames and three doors. On a show frame a door gets one of four colours, visible on that frame only. Later an ask frame names a door; the right answer is its last colour, and nothing on that frame says it. Other frames are answered `wait`. The delay between show and question falls in four buckets, from 1–4 frames to 17–32. Without memory, a model is right one time in four.

## v1: a memory that was never used

The [first pre-registration](https://github.com/maxgfr/skycmd/blob/main/docs/s6-sleep-prereg.md) began with a pilot at the small pico size: three memory runs, trained three ways, and a control, 3,684 steps each. The memory had to beat the control by at least 0.10 on dev episodes. They scored 0.2460, 0.2565 and 0.2430, the control 0.2650: chance, and unchanged at one, two or four passes. No larger run took place.

A [diagnosis](https://github.com/maxgfr/skycmd/blob/main/docs/s6-sleep-diagnosis.md) read the code and the saved runs. The memory was wired correctly, and every memory weight had moved. But the memory runs' training loss matched the control's within ±0.004, where solving `recall` was worth about 0.069, and each model answered nearly the same colour every time: two memory runs were both right on only 15 of 4,000 questions, where independent guessers would share about 250. The memory was trained, and ignored.

## v2: stopped by its own checks, before any GPU time

The [second version](https://github.com/maxgfr/skycmd/blob/main/docs/s6-sleep2-prereg.md) changed how the memory starts and is first trained, and added checks to pass on the CPU, on training episodes only, before any GPU minute.

C1 took three amendments in one day, its decoder having proved biased. C2, the colour held at the start, read 0.390 against a bar of 0.40 I had set by judgement; I went on, as a stated deviation, and let the next check decide.

C4 decided: 300 CPU training steps with the memory, and 300 with its read forced to zero. The loss on question frames had to reach 1.30 or less, and 0.05 below the zeroed run. Three allowed attempts: 1.3867 against 1.3873, 1.3876 against 1.3879, 1.3871 against 1.3873, all at chance (ln 4, about 1.386). The memory's gradient fell from 2.8e-2 to about 1e-11 by step 52.

v2 ended there, with no GPU run. That is a result too: training did not learn to use the memory, and no GPU time went into those odds.

## v3: hold the gates, and give the memory its own reason

A [second diagnosis](https://github.com/maxgfr/skycmd/blob/main/docs/s6-sleep3-diagnosis.md) found why. The write gate, a learned number setting how strongly a frame is written, read an unnormalised state and was trained ten times faster than the base: its mean fell from 0.96 at step 7 to 3e-7 at step 35. A closed gate writes nothing and gets no gradient to reopen. Held open, the gradient lived, yet the loss stayed at chance. The rest of the model was not the obstacle: with the colour written into the question frame itself, the base learned to answer in about thirty steps.

A [literature review](https://github.com/maxgfr/skycmd/blob/main/docs/research-fast-weight-memory.md) of 43 cited sources agreed. Published fast-weight memories do not start with a learned write gate that can close. And a network that can answer without its memory tends to ignore it; one counter-measure is an auxiliary loss asking the memory to give back what it stored.

The [third version](https://github.com/maxgfr/skycmd/blob/main/docs/s6-sleep3-prereg.md) is v2 with two changes:

- **The gates are held**, as constants: write strength 0.5, decay 0.9975 per frame.
- **An auxiliary recall loss, in training only.** At each question, a linear head reads what the memory retrieves and must name the colour. The head is not exported: at inference, no help.

A last CPU check, C5, passed on two seeds: 1.0000 and 0.9997 of the questions right, 0.2386 and 0.2628 with the memory off; its bars were set after the diagnosis, so it only showed the script worked. Then the pico pilot, on fresh dev episodes: **1.0000** against **0.2542** for the control, and 0.2567 with the memory zeroed.

## A run stopped by the memory guard

The first nano run stopped at step 203: the machine's memory guard, which halts a training above 13 GB, found it at 16 GB on a 16 GB laptop. The fix cut each batch of 16 sequences into parts of 2 and summed their gradients before the single update: the same computation, up to the order of floating-point sums. The run restarted from scratch with the same seed, and all six runs, memory and control at seeds 0, 1 and 2, went through in the night of 16 to 17 November.

## The test, read once

The pre-registered rule chose N = 1 on dev episodes (0.4823, against 0.4911 at N = 2: within 0.01 of the best). The test: 1,000 `recall` episodes, the four games over 200 episodes each, and `typed_decisions_bench`, the language test, averaged over three seeds. Brackets are 95% bootstrap intervals: where the difference most likely lies, estimated by resampling the episodes 10,000 times.

| Hypothesis | Measured | Bar | Verdict |
|---|---|---|---|
| H0, the memory is used: `recall` at N = 1 minus the control | +0.2452 [+0.2370, +0.2532] (0.4892 against 0.2440) | ≥ +0.10 | **holds** |
| H1, sleep helps: `recall` at N = 4 minus N = 1 | +0.0030 [−0.0010, +0.0068] | lower bound above 0 | not shown |
| H2, the games do not lose against the control | Flappy 15.87 against 71.05; Dodge 5.89 against 7.94; Rescue and the Doom-like within 0.1% | no game more than 2% under | not shown |
| H3, the language tasks do not lose | 0.4897 against 0.4808 | at most 0.01 under | **holds** |

| `recall`, by delay since the show | 1–4 frames | 5–8 | 9–16 | 17–32 |
|---|---:|---:|---:|---:|
| **with memory, N = 1** | **0.6147** | **0.5590** | **0.4530** | **0.3302** |
| with memory, N = 4 (sleep) | 0.6290 | 0.5910 | 0.4620 | 0.2867 |
| control, no memory | 0.2420 | 0.2398 | 0.2518 | 0.2423 |

<figure class="article-figure">
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 440 368" width="440" height="368" role="img" aria-labelledby="fig1-title fig1-desc" fill="currentColor" font-size="14" style="display:block;width:100%;max-width:440px;height:auto;margin:0 auto">
<title id="fig1-title">Recall accuracy on the test episodes, by delay since the show</title>
<desc id="fig1-desc">Grouped horizontal bar chart, from 0 to 1, with a dashed line at chance, 0.25. At 1 to 4 frames: N = 1 0.6147, N = 4 0.6290, control 0.2420. At 5 to 8 frames: N = 1 0.5590, N = 4 0.5910, control 0.2398. At 9 to 16 frames: N = 1 0.4530, N = 4 0.4620, control 0.2518. At 17 to 32 frames: N = 1 0.3302, N = 4 0.2867, control 0.2423. Both memory readings fall towards chance as the delay grows; the control stays at chance.</desc>
<g font-size="13"><rect x="0" y="4" width="12" height="12" rx="2" style="fill:var(--vp-c-brand-1,#0284c7)"/><text x="17" y="15">N = 1 (no sleep)</text><rect x="140" y="4" width="12" height="12" rx="2" fill-opacity=".6"/><text x="157" y="15">N = 4 (sleep)</text><rect x="262" y="4" width="12" height="12" rx="2" fill-opacity=".25"/><text x="279" y="15">control, no memory</text></g>
<g><text x="0" y="50">1–4 frames</text>
<g><title>1–4 frames, N = 1: 0.6147</title><path d="M0,58h192.7a4,4 0 0 1 4,4v4a4,4 0 0 1 -4,4h-192.7z" style="fill:var(--vp-c-brand-1,#0284c7)"/><text x="204.7" y="68" font-size="13" font-weight="700">0.6147</text></g>
<g><title>1–4 frames, N = 4: 0.6290</title><path d="M0,74h197.3a4,4 0 0 1 4,4v4a4,4 0 0 1 -4,4h-197.3z" fill-opacity=".6"/><text x="209.3" y="84" font-size="13">0.6290</text></g>
<g><title>1–4 frames, control: 0.2420</title><path d="M0,90h73.4a4,4 0 0 1 4,4v4a4,4 0 0 1 -4,4h-73.4z" fill-opacity=".25"/><text x="89.4" y="100" font-size="13">0.2420</text></g></g>
<g><text x="0" y="124">5–8 frames</text>
<g><title>5–8 frames, N = 1: 0.5590</title><path d="M0,132h174.9a4,4 0 0 1 4,4v4a4,4 0 0 1 -4,4h-174.9z" style="fill:var(--vp-c-brand-1,#0284c7)"/><text x="186.9" y="142" font-size="13" font-weight="700">0.5590</text></g>
<g><title>5–8 frames, N = 4: 0.5910</title><path d="M0,148h185.1a4,4 0 0 1 4,4v4a4,4 0 0 1 -4,4h-185.1z" fill-opacity=".6"/><text x="197.1" y="158" font-size="13">0.5910</text></g>
<g><title>5–8 frames, control: 0.2398</title><path d="M0,164h72.7a4,4 0 0 1 4,4v4a4,4 0 0 1 -4,4h-72.7z" fill-opacity=".25"/><text x="89.4" y="174" font-size="13">0.2398</text></g></g>
<g><text x="0" y="198">9–16 frames</text>
<g><title>9–16 frames, N = 1: 0.4530</title><path d="M0,206h141a4,4 0 0 1 4,4v4a4,4 0 0 1 -4,4h-141z" style="fill:var(--vp-c-brand-1,#0284c7)"/><text x="153" y="216" font-size="13" font-weight="700">0.4530</text></g>
<g><title>9–16 frames, N = 4: 0.4620</title><path d="M0,222h143.8a4,4 0 0 1 4,4v4a4,4 0 0 1 -4,4h-143.8z" fill-opacity=".6"/><text x="155.8" y="232" font-size="13">0.4620</text></g>
<g><title>9–16 frames, control: 0.2518</title><path d="M0,238h76.6a4,4 0 0 1 4,4v4a4,4 0 0 1 -4,4h-76.6z" fill-opacity=".25"/><text x="89.4" y="248" font-size="13">0.2518</text></g></g>
<g><text x="0" y="272">17–32 frames</text>
<g><title>17–32 frames, N = 1: 0.3302</title><path d="M0,280h101.7a4,4 0 0 1 4,4v4a4,4 0 0 1 -4,4h-101.7z" style="fill:var(--vp-c-brand-1,#0284c7)"/><text x="113.7" y="290" font-size="13" font-weight="700">0.3302</text></g>
<g><title>17–32 frames, N = 4: 0.2867</title><path d="M0,296h87.7a4,4 0 0 1 4,4v4a4,4 0 0 1 -4,4h-87.7z" fill-opacity=".6"/><text x="99.7" y="306" font-size="13">0.2867</text></g>
<g><title>17–32 frames, control: 0.2423</title><path d="M0,312h73.5a4,4 0 0 1 4,4v4a4,4 0 0 1 -4,4h-73.5z" fill-opacity=".25"/><text x="89.4" y="322" font-size="13">0.2423</text></g></g>
<g><title>Chance: one right answer in four, 0.25</title><path d="M80.5,54V106M80.5,128V180M80.5,202V254M80.5,276V328" stroke="currentColor" stroke-width="1.5" stroke-dasharray="4 3" fill="none"/></g>
<g stroke="currentColor" stroke-opacity=".5"><line x1="0" y1="338.5" x2="321" y2="338.5"/><line x1="0.5" y1="338" x2="0.5" y2="343"/><line x1="80.5" y1="338" x2="80.5" y2="343"/><line x1="160.5" y1="338" x2="160.5" y2="343"/><line x1="240.5" y1="338" x2="240.5" y2="343"/><line x1="320.5" y1="338" x2="320.5" y2="343"/></g>
<g fill-opacity=".8" font-size="13" text-anchor="middle"><text x="0" y="361" text-anchor="start">0</text><text x="80" y="361">chance 0.25</text><text x="160" y="361">0.5</text><text x="240" y="361">0.75</text><text x="320" y="361">1</text></g>
</svg>
<figcaption>Figure 1. Share of right answers on the 1,000 test episodes of <code>recall</code>, three seeds pooled, by frames since the door was shown (axis from zero; dashed line: chance). The memory, in colour, is well above the control at short delays and near chance at 17 to 32 frames; four sleep passes gain a little up to 16 frames and lose beyond.</figcaption>
</figure>

The exported file also had to pass s6-nano's own gate. Its language part holds (0.499 on `typed_decisions_bench`, bar 0.398; calibration error 0.0578, bar 0.08). Its arena, against the earlier s3-pico, does not: Flappy 2.10 against 200.00, Dodge 6.90 against 9.47. The shipped engine matches the training code, natively and in WebAssembly.

## What the failure says

- **The memory is real, and it fades**: 0.6147 a few frames after the show, 0.3302 after 17 to 32.
- **Sleep did nothing measurable**: +0.0030, an interval that contains zero; at the longest delays four passes score under one.
- **The memory cost two games.** The control itself plays Flappy at 71.05, where its base reached 200 in another reader; this training may lower Flappy on its own, but that is not measured.
- **The pilot did not carry over**: 1.0000 at the pico size, 0.4892 at nano. On record, the nano runs' first phase had at most 233 `recall` steps, the pilot's 526; whether that is the cause is not measured.
  *Note added on 19 November 2026:* a fourth version, s6-sleep v4, tried to answer that on training sequences and is closed. Its first check, K3 (the base model frozen), failed on both seeds; its second, K1 (everything trained, more `recall` steps), met its bar on one seed of two (0.8442 and 1.0000 against 0.95 on both, read on training batches), so v4 stopped there with no model and no test read ([`docs/s6-sleep4-prereg.md`](https://github.com/maxgfr/skycmd/blob/main/docs/s6-sleep4-prereg.md) §16). That narrows the question without settling it.
- **One probe, one size**, three seeds. A synthetic probe says nothing about real tasks, nor about the paper, whose memory spans tokens, not frames.

## What I take from it

**A probe with a known answer and a control arm tell you whether a mechanism is used at all.** Without `recall`, v1's unused memory would have vanished in the game scores; without the control and the zeroed reading, the pilot's 1.0000 could have been a better guess.

**Spend CPU minutes before GPU hours.** v2's checks stopped a pilot before it started; v3's diagnosis found the closing gate on training episodes, without reading one dev or test seed.

**Write down what a failure means before the run.** v1's pre-registration says it in one sentence: "a model that remembers but does not gain from sleeping is not `s6-sleep-nano`." Committed in advance, it stops "the memory works" from being reported as "the model works". The gate asked for a memory, a gain from sleep and intact games; it got the first.

**Publish the failure in a form people can run**, so anyone can watch the memory work and fade, as the [secrets verifier](/articles/a-250-kb-secrets-verifier) was published with its failed gate.

::: tip Try it
The [Memory](/memory/) page plays `recall` with the 752 KB file and its 715 KB control, memory on, zeroed, or the control, at one, two or four sleep passes, next to the test reading.
:::

## Check it

Everything is public: the pre-registrations of [v1](https://github.com/maxgfr/skycmd/blob/main/docs/s6-sleep-prereg.md), [v2](https://github.com/maxgfr/skycmd/blob/main/docs/s6-sleep2-prereg.md) and [v3](https://github.com/maxgfr/skycmd/blob/main/docs/s6-sleep3-prereg.md), each with its amendments and result; the diagnoses after [v1](https://github.com/maxgfr/skycmd/blob/main/docs/s6-sleep-diagnosis.md) and [v2](https://github.com/maxgfr/skycmd/blob/main/docs/s6-sleep3-diagnosis.md), v2's [design study](https://github.com/maxgfr/skycmd/blob/main/docs/s6-sleep2-study.md) and the [literature review](https://github.com/maxgfr/skycmd/blob/main/docs/research-fast-weight-memory.md); the [test report](https://github.com/maxgfr/skycmd/blob/main/runs/s6-sleep3/REPORT.md); the [model page](https://huggingface.co/maxgfr/s6-sleep3-nano); the paper, [arXiv 2605.26099](https://arxiv.org/abs/2605.26099); [all the results](/results/#s6-sleep3-nano-failed), the [models](/models/) and [the code](https://github.com/maxgfr/skycmd).
