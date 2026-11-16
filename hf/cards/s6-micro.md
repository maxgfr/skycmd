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
---

# s6-micro

> **It passed its gate on a second look.** This is the second version of s6-micro. The first
> version ([s6-micro-v1](https://huggingface.co/{{USER}}/s6-micro-v1)) failed the ECE guard of the
> same gate at 0.0821 against 0.08. This version was designed after that reading (twice the
> decision-training minutes, everything else the same) and its gate was read on the same test
> records ([`docs/s6-gen-prereg.md`](https://github.com/maxgfr/skycmd/blob/main/docs/s6-gen-prereg.md),
> amendments 12, 16 and 17), so its pass is not a first look at these sets. It reached **0.553**
> on `typed_decisions_bench` (gate 0.45), an ECE of **0.0588** over the test sources (guard 0.08)
> and s3-pico's score in every game. On 11 of the bench's 20 tasks it gives one answer to at least
> 95% of the items; answering each question with its most common training answer scores 0.4815.
> Its decision training was restarted once from its checkpoint after a technical failure, and its
> last minutes, its DAgger rounds and its evaluation ran on a loaded machine.

**One 2.4 MB ternary model for four games and for typed questions in free text: the larger
sibling of [s6-nano](https://huggingface.co/{{USER}}/s6-nano).** 5,587,201 parameters. It reads a
state as text and answers typed questions (`choice`, `score`, `noul`) with probabilities
calibrated on validation, one forward pass per question. It plays the four skycmd games at the oracle's
level, and passed a publication gate fixed before training, on its second version.

- **Demo:** [huggingface.co/spaces/{{USER}}/s6-micro](https://huggingface.co/spaces/{{USER}}/s6-micro)
  (the games, and typed questions you can edit)
- **Code:** [github.com/maxgfr/skycmd](https://github.com/maxgfr/skycmd) (100 % Rust)
- **Pre-registration:** [docs/s6-gen-prereg.md](https://github.com/maxgfr/skycmd/blob/main/docs/s6-gen-prereg.md)
  (gates in §6, this version in amendments 12, 16 and 17)
- **All results, failures included:** [results page](https://maxgfr.github.io/skycmd/results/)

## Results

Test sets, at most 2,000 questions per source, ternary GGUF of the last DAgger round, full depth
(τ = 1); the other columns are the same records:

| set | s6-micro | s6-micro-v1 (failed) | s6-nano | reference |
|---|---:|---:|---:|---|
| `typed_decisions_bench` | **0.553** | 0.547 | 0.480 | majority class 0.398; the most common training answer of each question 0.4815; s3-pico 0.382, s4-micro 0.352, s5-lite 0.696 |
| MASSIVE intents, English | 0.531 | 0.518 | 0.425 | s3-pico 0.187 (see the limitations) |
| MASSIVE intents, French | 0.314 | 0.275 | 0.239 | s3-pico 0.117 |
| When2Call (test split of a trained source) | 0.401 | 0.430 | 0.404 | s3-pico 0.368; its most common training answer 0.359 |
| NOTAM subject / NOTAM multi (never trained on) | 0.093 / 0.353 | 0.052 / 0.394 | 0.055 / 0.493 | s3-pico 0.163 / 0.319 |
| all sources | 0.380 | 0.378 | 0.360 | |
| **ECE over all these sources** | **0.0588** | 0.0821 | 0.0479 | **the gate allows 0.08** |

- **ECE by source:** bench 0.0498, MASSIVE en 0.0715, MASSIVE fr 0.0300, When2Call 0.2165, NOTAM
  subject 0.0185, NOTAM multi 0.1409. The 95% bootstrap interval of the pooled ECE is
  [0.0526, 0.0687] (v1: [0.0746, 0.0900]). On When2Call the model is over-confident (mean
  confidence 0.612 for an accuracy of 0.401): it answers `request_for_info` to 1,405 of the 2,000
  items and `tool_call` to 45, where `tool_call` is the gold answer of 696.
- **In distribution:** Banking77 intents 0.763 (it is in the training data).

Closed loop (`sky-arena`, 30 episodes per game, the same seeds as s3-pico, full depth):

| game | result |
|---|---|
| Flappy Drone | 200 / 200 pipes, 0 crashes |
| Rescue Run | 100 % |
| Obstacle Dodge | 87 % (= oracle), 20 reflex vetoes |
| Doom-like | 100 % of levels, kill rate 0.77 (= oracle), 2 vetoes |

Latency, measured while other jobs loaded the machine (not comparable with the other models):
median 8,970 µs per decision natively over the test sources; in WASM under Node, a median of
61 ms on the long JSON states of 200 bench questions. WASM and native agree to 8.9e-7 on those 200
questions, with the same answer on all 200.

## Full depth

The file answers at full depth: all 6 block passes, every time. Its three exit thresholds are set
to 1 in the GGUF (`skycmd-s2.exit_threshold.{choice,score,noul}`), as amendment 13 of the
pre-registration defines a publication at full depth; the weights are those of the evaluated
file (the last DAgger round), byte for byte. Conformal risk control (α = 0.02) chose earlier
thresholds, which kept the four arena scores but added one Dodge veto (21 against 20); the
pre-registered rule allows none, so they were rejected. You can still pass another threshold at
run time (`setExitThreshold` in the browser).

## Architecture

The s4-micro shape (`skycmd-s2` GGUF tag): question and options form a static stream encoded once
and cached; the state goes through 2 weight-shared blocks looped three times, with exits after 2,
4 and 6 block passes; options never attend to each other; ternary weights trained with QAT. d 256,
4 heads, vocabulary 4,096, windows of 384 state tokens and 128 question tokens. The pre-registered
ablation adopted none of the s6 additions (Parallax, Muon v2, XSA and others), so this model uses
softmax attention and AdamW.

## Limitations

- **A second look.** v1 failed the ECE guard and this version was designed after reading that
  result, on the same test records. Its pass would be stronger on test sets read for the first
  time.
- A modest generalist: 0.553 on the bench, against about 0.69 for 17M models (s5-lite 0.696,
  oscar-1-17m 0.6775). On 11 of the 20 bench tasks it answers one constant to at least 95% of the
  items.
- It cannot read NOTAMs (0.093 on the subject question), and on When2Call it stays at 0.401 and
  rarely chooses to call a tool.
- The English MASSIVE score is not strictly zero-shot: 206 of the 2,000 test utterances also
  appear in the training split, 201 of them in a source derived from MASSIVE. None of the 1,000
  French ones does.
- One run, one seed. The decision training died once on a file-access error and was resumed from
  its checkpoint (the steps after it were done again); the end of the run shared the machine with
  other jobs.
- The games are civil drone scenarios and a Doom-like shooter drawn with Freedoom art; the model
  only advises and coded rules always keep a veto. This is a demo and a tutorial, not flight
  software.

## Usage

Runs in the skycmd engine only (browser, Node, Python, CLI, server):

```sh
cargo install --git https://github.com/maxgfr/skycmd sky-serve
skycmd decide --model s6-micro.gguf --request req.json --pretty
```

Request format and bindings: [install guide](https://maxgfr.github.io/skycmd/install/).

## Training

From scratch in Rust (mlx-rs) on an Apple M5, in a throttled mode, so the budgets are working
minutes: masked-language pretraining in float (180 minutes, 31,700 steps, 0.52B tokens, 93 per
parameter; the pretraining of v1, not run again), then decision training with ternary QAT after
a 10 % float warm-up (360 minutes, uncompiled, 65,471 steps of 64, learning rate ×0.5 chosen by a
pre-registered pilot) and three DAgger rounds on the games (20 minutes each: 3,822, 4,673 and
4,969 steps). Data: the typed-decision sets of Hugging Face, classic datasets converted into
typed questions, game states labelled by exact oracles and a numeric curriculum, the same as v1
and s6-nano. Pretraining text: SimpleStories, TinyStories, FineWeb and the non-web parts of
Dolma 3 and TxT360. Soft labels on 747,077 training questions come from GLM-5.3-flash (open
weights, MIT, called through the z.ai API), kept only where they agree with the gold answer; no
test question was sent. The z.ai terms of service restrict training on its outputs; the
project's choice and the labels are described in
[DATA_LICENSES.md](https://github.com/maxgfr/skycmd/blob/main/DATA_LICENSES.md), with every
other license.

## Files

- `s6-micro.gguf`: ternary weights, tokenizer, temperatures, and exit thresholds set to 1 (full
  depth). 2,420,032 bytes, sha256 `f66d702121e85e678e9b6ea73e9c01433e968b3a3734e8f68ac58f4cee4340ff`.
