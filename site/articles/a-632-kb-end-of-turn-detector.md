---
title: "A 632 KB model that knows when you have finished speaking: one failure, one harness bug, then a pass"
description: "A tiny model that decides whether a speaker has finished their turn: a first version that failed three of its four gates, a harness bug that had been hiding the reference model's true score, then a second version designed before training around one job, a checked baseline and a stopwatch, which passes, with its weakness on real conversations stated with the pass."
date: 2026-11-17
outline: [2, 2]
---

# A 632 KB model that knows when you have finished speaking: one failure, one harness bug, then a pass

*17 November 2026*

A voice agent has to decide whether you have stopped talking. The reference open model for this, Smart Turn v3.2, weighs 8.7 MB and takes about 32 ms per decision on the CPU; I wanted it under a megabyte and a few milliseconds. The path took one failure, one harness bug, and a second model designed in full before training. It passed; what the pass does not say is part of this article.

## The context: skycmd

skycmd gathers very small "decision models" with ternary weights (each is −1, 0 or +1), written in Rust, run natively or in the browser through WebAssembly, and trained on one MacBook (Apple M5) in a throttled mode, so budgets count working minutes.

::: info Pre-registration and the gate
Before an experiment runs, I commit the test, the bar to clear and what a failure would mean. The bar is the "gate": conditions that must all hold. The test is read once, a failure is published like a success, and the rule cannot be adjusted afterwards.
:::

## The job: has the speaker finished?

A voice agent is a program you talk to: it listens and answers aloud. End-of-turn detection is how it knows when to start: from the audio alone, decide whether the speaker has finished their turn or has only paused. The usual alternative, a silence timer, cuts people off when too short and drags when too long.

Accuracy is the share of right answers. A false cut-off is the costliest mistake: answering "finished" while the speaker was only drawing breath. Recall is the share of real cases a system catches — here, of clips that really end a turn.

The reference is [Smart Turn v3.2](https://github.com/pipecat-ai/smart-turn) by Pipecat, open under the BSD-2-Clause licence: an 8.7 MB file at about 32 ms per decision on one CPU core. On its official test set of 31,527 clips Pipecat reports 0.9263; a timer that always answers "finished" scores 0.4979 there.

## v1: one model, three jobs, a pretraining that hurt

The first version, [s6-ear-nano](https://huggingface.co/maxgfr/s6-ear-nano), was ambitious: one 899 KB model as the ears of a whole agent. It began by pretraining (predicting, on LibriSpeech, the codes that Mimi, a neural audio codec, would produce), then shared its weights between three tasks: end of turn (45% of the budget), keywords, and keywords in words it had never heard. Its input: 10.24 s of log-mel frames, sound cut into short overlapping windows, each summarised by its energy in a few dozen frequency bands, on a logarithmic scale.

Its gate had four conditions; three failed. End of turn 0.5681 against 0.8518, the number my harness then read for Smart Turn — hold that number. Keywords 0.8885 against 0.90; French zero-shot keywords 0.1330 against 0.20; only English zero-shot passed. The pre-registered control — same model, data and budget, without the pretraining — did better on all four sets, 0.7647 on end of turn. The pretraining had learned its own task and still hurt: the pretrained model underfit its training clips, 0.702 against 0.861 for the control. The failure is published with its file. Two lessons: one job at a time; at this scale a codec pretraining bought nothing.

## The bug in the harness

That 0.8518 changed v2's design. To score Smart Turn I had written a small script around its ONNX file; one step of its preprocessing came out wrong: a clip shorter than 8 s has to be padded, my script at the end, Smart Turn at the start. A short clip then ends in zeros, and trailing silence looks exactly like a finished turn. On 3,000 test clips my harness read 0.8437 where the reference padding reads 0.9243; on the full set, 0.8518 instead of the published 0.9263. Every "0.5681 against 0.8518" understated the gap by about 7.5 points. The verdicts did not move, but every file that printed the wrong bar was corrected.

The correction was procedural: a comparison against a baseline you have not reproduced is a comparison against a number you do not have. v2's pre-registration made the check a precondition — the fixed harness had to give 0.9263 ± 0.003 on all 31,527 clips, and a timer the test's share of finished clips, before any v2 score could be read. It gave 0.9260 and 0.4979; only then was the test read.

## v2: one job, one teacher, the size chosen by a stopwatch

The second version was designed before anything ran: a design study, then the [pre-registration](https://github.com/maxgfr/skycmd/blob/main/docs/s6-ear2-prereg.md), committed before a line of code. The decisions, and their reasons:

- **One task, the teacher's data.** Keywords are dropped, and v1's older release lacked 17.4% of the test's sources; v2 trains on `smart-turn-data-v3.2-train`, Smart Turn's training set, 270,946 clips, CC-BY-4.0.
- **Distillation, credited.** Distillation trains a small model, the student, to reproduce the outputs of a larger one, the teacher. The student copies Smart Turn's output score, pooled vector and encoder states, on training clips only; its 8 million parameters carry speech pretraining a from-scratch student cannot rebuild.
- **A 4 s window, measured.** I fed Smart Turn the last few seconds of 3,000 test clips: at 4 s it keeps 0.9190 of its 0.9243, at 3 s 0.9033 — so the student reads the last 4 s where the teacher reads 8.
- **Features the engine computes itself.** Per 10 ms hop: 40 log-mel bands plus three prosody channels — log pitch, voicing, energy — each window normalised so microphone gain cancels. v1 had none of this.
- **The size, chosen by a probe.** Four candidate sizes were exported with random ternary weights and timed over 1,000 decisions on a full 4 s window; the rule, written first: the first size whose median stays under 4.0 ms, 1 ms of margin under the gate's bound. The candidates took 10.10, 6.79 and 4.89 ms — the third rejected by 0.89 ms — the fourth, R4, 2.39 ms: one position per 80 ms instead of 40, so attention runs over half the positions. That is the model: 1,306,178 parameters, 631,936 bytes.

The pilot chose the learning rate and tested distillation: the best distilled arm reached 0.8715 on validation, hard labels 0.8437, and cleared the 0.82 stop rule. The main run took 240 working minutes: 305,700 steps, about 152 epochs, seed 1, once, ternary quantisation over the last 15%. The whole line used about 293 working minutes of GPU.

## The result

Read once, at the end: Smart Turn v3.2's 31,527 official clips, one fixed harness for every system.

| System | Size | Accuracy |
|---|---:|---:|
| silence timer (always "finished") | 0 bytes | 0.4979 |
| s6-ear-nano (v1, failed) | 899 KB | 0.5681 |
| v1's control (no Mimi pretraining, not published) | — | 0.7647 |
| Smart Turn v3.2 CPU, as v1's harness read it (wrong padding) | 8.7 MB | 0.8518 |
| **s6-ear2-nano (v2)** | **632 KB** | **0.8993** [0.8962, 0.9027] |
| Smart Turn v3.2 CPU, fixed harness | 8.7 MB | 0.9260 |

Three gate conditions; all held:

- **G1, accuracy.** 0.8993, above the bar of 0.85 and within 0.08 of the teacher's measured 0.9260. The bracket is a 95% interval.
- **G2, false cut-offs.** At the file's cautious threshold τ₈₅ = 0.7244, the highest threshold whose validation END recall reaches 0.85: false cut-off 0.0847 (bar 0.20), END recall 0.8627 (bar 0.80). Both thresholds come from validation, never from test. In a voice agent, run it only with a fallback timer behind it.
- **G3, deployable.** 2.39 ms per decision in WebAssembly (bar 5), 0.0186 ms per 10 ms hop of front end (bar 0.10), 631,936 bytes (bar 3 MB), browser matching native to within 1.2e-7, same decision on 20 of 20 clips.

<figure class="article-figure">
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 440 391" width="440" height="391" role="img" aria-labelledby="fig1-title fig1-desc" fill="currentColor" font-size="14" style="display:block;width:100%;max-width:440px;height:auto;margin:0 auto">
<title id="fig1-title">Accuracy on Smart Turn v3.2's 31,527 test clips</title>
<desc id="fig1-desc">Horizontal bar chart, from 0 to 1. Smart Turn v3.2 CPU, 8.7 MB: 0.9260. s6-ear2-nano, the 632 KB student: 0.8993. Smart Turn as v1's harness read it, right padding: 0.8518. v1's control without the Mimi pretraining: 0.7647. s6-ear-nano, 899 KB: 0.5681. The silence timer, always "finished": 0.4979.</desc>
<g><title>Smart Turn v3.2 CPU · 8.7 MB: 0.9260</title><text x="0" y="39">Smart Turn v3.2 CPU · 8.7 MB</text><path d="M0,46h291.7a4,4 0 0 1 4,4v8a4,4 0 0 1 -4,4h-291.7z" fill-opacity=".6"/><text x="303.7" y="59">0.9260</text></g>
<g><title>s6-ear2-nano · 632 KB: 0.8993</title><text x="0" y="93" font-weight="700">s6-ear2-nano · 632 KB</text><path d="M0,100h283.3a4,4 0 0 1 4,4v8a4,4 0 0 1 -4,4h-283.3z" style="fill:var(--vp-c-brand-1,#0284c7)"/><text x="295.3" y="113" font-weight="700">0.8993</text></g>
<g><title>Smart Turn, as v1's harness read it (right padding): 0.8518</title><text x="0" y="147">Smart Turn, as v1's harness read it</text><path d="M0,154h268.3a4,4 0 0 1 4,4v8a4,4 0 0 1 -4,4h-268.3z" fill-opacity=".6"/><text x="280.3" y="167">0.8518</text></g>
<g><title>v1's control (no Mimi pretraining): 0.7647</title><text x="0" y="201">v1's control (no Mimi pretraining)</text><path d="M0,208h240.9a4,4 0 0 1 4,4v8a4,4 0 0 1 -4,4h-240.9z" fill-opacity=".6"/><text x="252.9" y="221">0.7647</text></g>
<g><title>s6-ear-nano (v1) · 899 KB: 0.5681</title><text x="0" y="255">s6-ear-nano (v1) · 899 KB</text><path d="M0,262h179a4,4 0 0 1 4,4v8a4,4 0 0 1 -4,4h-179z" fill-opacity=".6"/><text x="191" y="275">0.5681</text></g>
<g><title>silence timer · 0 bytes: 0.4979</title><text x="0" y="309">silence timer · 0 bytes</text><path d="M0,316h156.8a4,4 0 0 1 4,4v8a4,4 0 0 1 -4,4h-156.8z" fill-opacity=".6"/><text x="168.8" y="329">0.4979</text></g>
<g stroke="currentColor" stroke-opacity=".5"><line x1="0" y1="360.5" x2="321" y2="360.5"/><line x1="0.5" y1="360" x2="0.5" y2="365"/><line x1="80.5" y1="360" x2="80.5" y2="365"/><line x1="160.5" y1="360" x2="160.5" y2="365"/><line x1="240.5" y1="360" x2="240.5" y2="365"/><line x1="320.5" y1="360" x2="320.5" y2="365"/></g>
<g fill-opacity=".8" font-size="13" text-anchor="middle"><text x="0" y="383" text-anchor="start">0</text><text x="80" y="383">0.25</text><text x="160" y="383">0.5</text><text x="240" y="383">0.75</text><text x="320" y="383">1</text></g>
</svg>
<figcaption>Figure 1. Accuracy on the 31,527 test clips, one harness for all (axis from zero). The student, in colour, gives up 2.7 points against its teacher for a file 14 times smaller and a decision about 13 times faster; 0.8518 is the teacher as the unfixed harness read it.</figcaption>
</figure>

The gate's sentence on which model to take: "Take v2 when size and speed matter: it gives up 2.7 points of accuracy against Smart Turn for a file of 0.63 MB instead of 8.7 MB and about 2.4 ms per decision in WASM instead of about 32 ms natively; take Smart Turn's file when those points matter more."

## What the pass does not say

The pass says the model cleared its bars, not that the problem is solved.

- **On real conversations, both models are far weaker.** LiveKit's eot-bench is real human-to-agent speech — 14 languages, 17,934 decisions at labelled pauses — and the design never read it. The student's AUC falls from 0.9600 on the official test to 0.7299 there, the teacher's to 0.7838 (AUC: how well a model ranks finished above unfinished; 0.5 is chance, 1 is perfect). At the cautious threshold, v2 answers "finished" on 0.5253 of the real ends of turn: about half pass by it. The pre-registered "does not carry over" needed a gap above 0.10; at 0.054, the plainer sentence stands.
- **The weakest clips have names.** Vietnamese 0.7749, Marathi 0.7907, Bengali 0.7950, against 0.8993 overall; English 0.9056.
- **The test is mostly synthetic.** Most clips are text-to-speech voices.
- **The official test is partly in-sample for the design** — never for the weights: no test clip is trained on, no threshold chosen on test — but the 4 s window and the data change came from readings on 3,000 of its clips.
- **It is a student.** Trained to reproduce its teacher, it cannot be more right than the labels it imitates; it agrees with Smart Turn on 91.8% of test clips.

## What I take from it

**Reproduce the baseline before anything else.** v1's gate was read against a wrong number because I had not run the reference the way its authors do; in v2, reproducing it came before reading my own score. A baseline is not a citation; it is a measurement.

**Measure the speed before training, and let it choose the size.** A random-weight file costs minutes to build and times exactly like a trained one. The probe settled the one trade-off training cannot revisit, and it overruled the design's preferred size by 0.89 ms.

**One narrow job and a good teacher beat an ambitious general model.** v1 shared 60 minutes across three tasks under a pretraining that hurt all three; v2 spent 240 minutes on one question, copying a teacher that already knows it. At 1.3 million parameters, capacity is too scarce to share. The [secrets verifier](/articles/a-250-kb-secrets-verifier) learned the same lesson.

**Keep a benchmark the design never touched.** eot-bench is the number the design never optimised toward, and the only one that says plainly that both models are weak on real speech. The [tool router](/articles/a-3-7-mb-tool-router) got the same treatment: fresh requests no system had scored.

**All of it on one laptop.** Probe, pilot, 240 minutes of training, 31,527 test clips, 17,934 real cuts: one Apple M5, throttled, about 293 working minutes of GPU.

## Check it

Everything is public: the model pages of [v2](https://huggingface.co/maxgfr/s6-ear2-nano) and the [failed v1](https://huggingface.co/maxgfr/s6-ear-nano); the pre-registrations of [v1](https://github.com/maxgfr/skycmd/blob/main/docs/s6-ear-prereg.md) and [v2](https://github.com/maxgfr/skycmd/blob/main/docs/s6-ear2-prereg.md); the [gate report](https://github.com/maxgfr/skycmd/blob/main/runs/s6-ear2-nano/GATE.md); the [design study](https://github.com/maxgfr/skycmd/blob/main/docs/s6-ear2-study.md) behind v2's choices; [Smart Turn v3.2](https://github.com/pipecat-ai/smart-turn) (Pipecat, BSD-2-Clause); [all the results](/results/), the [models](/models/) and [the code](https://github.com/maxgfr/skycmd). The training set and eot-bench are CC-BY-4.0; no audio is redistributed. The browser demo is on the [Listen page](/ear/) and in the [Space](https://huggingface.co/spaces/maxgfr/s6-ear2-nano), with the microphone or a self-test; the engine API is `TurnDetector` in WASM and `sky-turn` natively.
