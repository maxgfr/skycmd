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
  - games
  - rust
  - wasm
  - skycmd
  - failed-gate
  - negative-result
---

# s6-micro-v1

> **Failed its pre-registered gate: published as a result, not for use.** The gate
> ([`docs/s6-gen-prereg.md`](https://github.com/maxgfr/skycmd/blob/main/docs/s6-gen-prereg.md) §6)
> asked for at least 0.45 on `typed_decisions_bench`, an ECE of at most 0.08 over the test sources
> and no game more than 5% below s3-pico. s6-micro reached **0.547** on the bench and plays every
> game at s3-pico's score, but its ECE is **0.0821 at full depth** (0.0804 at its conformal
> thresholds) against the guard of 0.08: it **failed its calibration guard**. The file is published
> so that the result can be checked and run (owner's decision of 15/11/2026, amendment 15 of the
> pre-registration); the verdict does not change. Numbers:
> [BENCH.md, section skycmd-s6](https://github.com/maxgfr/skycmd/blob/main/BENCH.md#skycmd-s6-s6-nano-s6-micro-general-purpose-decisions-and-games-pre-registered).

**A 2.4 MB ternary model for four games and for typed questions in free text: the larger sibling
of [s6-nano](https://huggingface.co/{{USER}}/s6-nano).** 5,587,201 parameters. It reads a state
as text and answers typed questions (`choice`, `score`, `noul`) with probabilities, one forward
pass per question. It is published as `s6-micro-v1`. The second version pre-registered in
amendment 12 (360 minutes of decision training instead of 180, the same gate) ran on 15 and
16/11/2026 and passed, on a second look at the same test records (amendment 17); it is published
as [s6-micro](https://huggingface.co/{{USER}}/s6-micro). This file's verdict does not change.

- **Demo:** [huggingface.co/spaces/{{USER}}/s6-micro-v1](https://huggingface.co/spaces/{{USER}}/s6-micro-v1)
  (the games, and typed questions you can edit)
- **Code:** [github.com/maxgfr/skycmd](https://github.com/maxgfr/skycmd) (100 % Rust)
- **Pre-registration:** [docs/s6-gen-prereg.md](https://github.com/maxgfr/skycmd/blob/main/docs/s6-gen-prereg.md)
  (gates in §6, the reading in amendments 12 to 15, the second version in amendment 17)
- **All results, failures included:** [results page](https://maxgfr.github.io/skycmd/results/)

## Results

Test sets, at most 2,000 questions per source, ternary GGUF of the last DAgger round, full depth
(τ = 1); the reference rows are the same records:

| set | s6-micro | s6-nano (passed) | reference |
|---|---:|---:|---|
| `typed_decisions_bench` | **0.547** | 0.480 | majority class 0.398; s3-pico 0.382, s4-micro 0.352, s5-lite 0.696 |
| MASSIVE intents, English | 0.518 | 0.425 | s3-pico 0.187 (see the limitations) |
| MASSIVE intents, French | 0.275 | 0.239 | s3-pico 0.117 |
| When2Call (test split of a trained source) | 0.430 | 0.404 | s3-pico 0.368 |
| NOTAM subject / NOTAM multi (never trained on) | 0.052 / 0.394 | 0.055 / 0.493 | s3-pico 0.163 / 0.319 |
| all sources | 0.378 | 0.360 | |
| **ECE over all these sources** | **0.0821** | 0.0479 | **the gate allows 0.08** |

- **At its conformal thresholds** (not published): 0.544 on the bench, ECE 0.0804, 4.59 block
  passes on average instead of 6, early exit on 36.0% of the questions. Still above 0.08.
- **Where the ECE comes from** (read after the gate, exploratory, amendment 12): the largest part is
  `notam:notam_multi` (ECE 0.2558, against 0.0710 for s6-nano). On one NOTAM statement, false for
  99.5% of its 412 items, s6-micro answers "true" for almost all of them. The model is
  under-confident on the bench (mean confidence 0.4815 against an accuracy of 0.5470) and
  over-confident on When2Call (0.5372 against 0.4305). The 95% bootstrap interval of the pooled
  ECE is [0.0746, 0.0900]; the gate is read on the point value, as pre-registered.
- **ECE by source:** bench 0.0660, MASSIVE en 0.1334, MASSIVE fr 0.0313, When2Call 0.1176,
  NOTAM subject 0.0636, NOTAM multi 0.2558.

Closed loop (`sky-arena`, 30 episodes per game, the same seeds as s3-pico, full depth): Flappy
200.00, Rescue 3.00, Dodge 9.47, Doom-like 3.27 mean score, the scores of s3-pico and s6-nano.

Latency: median 7,346 µs per decision natively over the test sources at full depth (s6-nano:
1,488 µs). The WASM line of the run's results file was measured with early exits against a
full-depth native dump (amendment 13), so it does not describe this file and is not quoted; the
WASM latency of this file at full depth was not measured.

## Full depth

The file answers at full depth: all 6 block passes, every time. Its three exit thresholds are set
to 1 in the GGUF (`skycmd-s2.exit_threshold.{choice,score,noul}`), as amendment 13 of the
pre-registration defines a publication at full depth; the weights are those of the evaluated file
(the last DAgger round), byte for byte. You can still pass another threshold at run time
(`setExitThreshold` in the browser).

## What it does well, and badly

- **Better than s6-nano on accuracy:** 0.547 against 0.480 on the bench, the announced goal of 0.50
  met, and higher on MASSIVE and When2Call.
- **Worse calibrated:** 0.0821 against 0.0479, and the guard is what the gate checks. Its
  probabilities are not reliable across sources: do not read them as confidences.
- **Slower and larger:** a median of 7,346 µs per decision natively against 1,488 µs for s6-nano,
  in a file of 2,420,032 bytes against 715,136.
- It cannot read NOTAMs (0.052 on the subject question), and in the games it scores what s3-pico
  and s6-nano score, no more.

## Architecture

The s4-micro shape (`skycmd-s2` GGUF tag): question and options form a static stream encoded once
and cached; the state goes through weight-shared blocks with exits after 2, 4 and 6 block passes;
options never attend to each other; ternary weights trained with QAT. d 256, 4 heads, vocabulary
4,096, windows of 384 state tokens and 128 question tokens. The pre-registered ablation adopted
none of the s6 additions (Parallax, Muon v2, XSA and others), so this model uses softmax attention
and AdamW.

## Limitations

- **Failed its gate.** Recommended for no use. s6-nano, which passed the same gate, and the second
  version [s6-micro](https://huggingface.co/{{USER}}/s6-micro), which passed it on a second look,
  are the models to try.
- The English MASSIVE score is not strictly zero-shot: 206 of the 2,000 test utterances also
  appear in the training split, 201 of them in a source derived from MASSIVE (0.495 on those 206,
  0.521 on the 1,794 others). None of the 1,000 French ones does.
- One run, one seed. The decision training was stopped by hand at step 27,280 after about 34
  minutes in swap and resumed from step 27,161 uncompiled (amendment 11); the steps done in swap
  are kept, so it has fewer steps than its 180 minutes would otherwise give.
- The games are civil drone scenarios and a Doom-like shooter drawn with Freedoom art; the model
  only advises and coded rules always keep a veto. This is a demo and a tutorial, not flight
  software.

## Usage

Runs in the skycmd engine only (browser, Node, Python, CLI, server):

```sh
cargo install --git https://github.com/maxgfr/skycmd sky-serve
skycmd decide --model s6-micro-v1.gguf --request req.json --pretty
```

Request format and bindings: [install guide](https://maxgfr.github.io/skycmd/install/).

## Training

From scratch in Rust (mlx-rs) on an Apple M5, in a throttled mode, so the budgets are working
minutes: masked-language pretraining in float (180 minutes, 31,700 steps, 0.52B tokens, 93 per
parameter), then decision training with ternary QAT after a 10 % float warm-up (180 minutes,
38,826 steps, learning rate ×0.5 chosen by a pre-registered pilot) and three DAgger rounds on the
games (20 minutes each: 5,951, 5,898 and 5,952 steps). Data: the typed-decision sets of Hugging
Face, classic datasets converted into typed questions, game states labelled by exact oracles and a
numeric curriculum, the same as s6-nano. Pretraining text: SimpleStories, TinyStories, FineWeb and
the non-web parts of Dolma 3 and TxT360. Soft labels on 747,077 training questions come from
GLM-5.3-flash (open weights, MIT, called through the z.ai API), kept only where they agree with
the gold answer; no test question was sent. The z.ai terms of service restrict training on its
outputs; the project's choice and the labels are described in
[DATA_LICENSES.md](https://github.com/maxgfr/skycmd/blob/main/DATA_LICENSES.md), with every
other license.

## Files

- `s6-micro-v1.gguf`: ternary weights, tokenizer, temperatures, and exit thresholds set to 1 (full
  depth). 2,420,032 bytes.
