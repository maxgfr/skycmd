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

# s6-nano

**One 715 KB ternary model for four games and for typed questions in free text.** 1,486,225
parameters. It reads a state as text and answers typed questions (`choice`, `score`, `noul`) with
calibrated probabilities, one forward pass per question. It plays the four skycmd games at the
oracle's level, and of the skycmd models trained from scratch and measured on `typed_decisions_bench`, it is the
first above the majority class. It passed a publication gate fixed before training.

- **Demo:** [huggingface.co/spaces/{{USER}}/s6-nano](https://huggingface.co/spaces/{{USER}}/s6-nano)
  (the games, and typed questions you can edit)
- **Code:** [github.com/maxgfr/skycmd](https://github.com/maxgfr/skycmd) (100 % Rust)
- **Pre-registration:** [docs/s6-gen-prereg.md](https://github.com/maxgfr/skycmd/blob/main/docs/s6-gen-prereg.md)
- **All results, failures included:** [results page](https://maxgfr.github.io/skycmd/results/)

## Results

Test sets, at most 2,000 questions per source, full depth:

| set | s6-nano | reference |
|---|---:|---|
| `typed_decisions_bench` | **0.480** | majority class 0.398; on the same records s3-pico 0.382, s4-micro 0.352, s5-lite 0.696 |
| MASSIVE intents, English | 0.425 | s3-pico 0.187 (see the limitations) |
| MASSIVE intents, French | 0.239 | s3-pico 0.117 |
| When2Call (test split of a trained source) | 0.404 | s3-pico 0.368 |
| NOTAM subject / NOTAM multi (never trained on) | 0.055 / 0.493 | s3-pico 0.163 / 0.319 |
| ECE over all these sources | 0.0479 | the gate allows 0.08 |

Closed loop (`sky-arena`, 30 episodes per game, the same seeds as s3-pico, full depth):

| game | result |
|---|---|
| Flappy Drone | 200 / 200 pipes, 0 crashes |
| Rescue Run | 100 % |
| Obstacle Dodge | 87 % (= oracle), 23 reflex vetoes |
| Doom-like | 100 % of levels, kill rate 0.772 (= oracle) |

Latency: median 1,488 µs per decision natively over the test sources; in WASM under Node, about
15 ms on the long JSON states of the bench and 0.44 ms per Flappy frame (two questions and the
physics step). WASM and native agree to 9.5e-7 on 200 bench questions, with the same answer on
all 200.

## Full depth

The file answers at full depth: all 4 block passes, every time. Its three exit thresholds are set
to 1 in the GGUF (`skycmd-s2.exit_threshold.{choice,score,noul}`). The model has an exit after
each pass, and conformal risk control (α = 0.02) chose earlier thresholds, but at those
thresholds Rescue Run fell to 93 % and the Dodge vetoes rose from 23 to 1,016. The
pre-registered rule allows 2 %, so they were rejected. The calibration had left the thresholds at
0.5, a setting that was never evaluated, which is why they are set to 1 here. You can still pass
another threshold at run time (`setExitThreshold` in the browser).

## Architecture

The s3 graph at language scale (`skycmd-s2` GGUF tag), as in s4: question and options form a
static stream encoded once and cached; the state, one token per `key value` pair for numbers,
goes through 2 weight-shared blocks looped twice, with an exit after each of the 4 passes;
options never attend to each other; ternary weights trained with QAT. d 144, 2 heads,
vocabulary 2,048, windows of 512 state tokens and 160 question tokens.

The s6 code adds Parallax attention, Muon v2 and other options. The pre-registered ablation
adopted none of them, so this model uses softmax attention and AdamW.

## Limitations

- A modest generalist: 0.480 on the bench clears the majority class, but 17M models reach about
  0.69 (s5-lite 0.696, oscar-1-17m 0.6775).
- It cannot read NOTAMs (0.055 on the subject question) and stays at 0.404 on When2Call.
- The English MASSIVE score is not strictly zero-shot: 206 of the 2,000 test utterances also
  appear in the training split, 201 of them in a source derived from MASSIVE. None of the 1,000
  French ones does.
- One run, one seed. Its larger sibling s6-micro (5.6M) reached 0.547 on the bench but failed the
  ECE guard (0.0821) and is not published.
- The games are civil drone scenarios and a Doom-like shooter drawn with Freedoom art; the model
  only advises and coded rules always keep a veto. This is a demo and a tutorial, not flight
  software.

## Usage

Runs in the skycmd engine only (browser, Node, Python, CLI, server):

```sh
cargo install --git https://github.com/maxgfr/skycmd sky-serve
skycmd decide --model s6-nano.gguf --request req.json --pretty
```

Request format and bindings: [install guide](https://maxgfr.github.io/skycmd/install/).

## Training

From scratch in Rust (mlx-rs) on an Apple M5, in a throttled mode, so the budgets are working
minutes: masked-language pretraining in float (90 minutes, 53,360 steps, 0.87B tokens, 588 per
parameter), then decision training with ternary QAT after a 10 % float warm-up (90 minutes,
19,893 steps) and three DAgger rounds on the games (15 minutes each). Data: the typed-decision
sets of Hugging Face, classic datasets converted into typed questions, game states labelled by
exact oracles and a numeric curriculum. Pretraining text: SimpleStories, TinyStories, FineWeb and
the non-web parts of Dolma 3 and TxT360. Soft labels on 747,077 training questions come from
GLM-5.3-flash (open weights, MIT, called through the z.ai API), kept only where they agree with
the gold answer; no test question was sent. Licenses:
[DATA_LICENSES.md](https://github.com/maxgfr/skycmd/blob/main/DATA_LICENSES.md).

## Files

- `s6-nano.gguf`: ternary weights, tokenizer, temperatures, and exit thresholds set to 1 (full
  depth).
