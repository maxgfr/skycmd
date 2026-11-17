---
license: apache-2.0
language:
  - en
  - fr
library_name: gguf
pipeline_tag: audio-classification
tags:
  - gguf
  - decision-model
  - system-one
  - ternary
  - audio
  - turn-detection
  - keyword-spotting
  - rust
  - wasm
  - skycmd
  - failed-gate
  - negative-result
---

# s6-ear-nano

> **Failed its pre-registered gate: published as a result, not for use.** It failed three of its
> four gates ([`docs/s6-ear-prereg.md`](https://github.com/maxgfr/skycmd/blob/main/docs/s6-ear-prereg.md)
> §6, test split, ternary, full depth): end of turn **0.5681** against **0.9263** for Smart Turn
> v3.2 on the same clips (Pipecat's report; the v1 harness read 0.8518 because it padded short clips on the wrong side), 12 Speech Commands keywords **0.8885** against **0.90**, French zero-shot
> keywords **0.1330** against **0.20**; only English zero-shot keywords pass (0.2100, bar 0.20). Its
> control model, trained the same way without the Mimi pretraining, did better on all four sets.
> The file is published so that the result can be checked and run (owner's decision of 15/11/2026,
> amendment 5 of the pre-registration); the verdict does not change. Numbers:
> [BENCH.md, section s6-ear-nano](https://github.com/maxgfr/skycmd/blob/main/BENCH.md#s6-ear-nano-first-run-gates-failed).
>
> **The follow-up:** a dedicated end-of-turn student of Smart Turn,
> [s6-ear2-nano](https://huggingface.co/maxgfr/s6-ear2-nano), trained on this one task without the
> pretraining, passed its own gate on 17/11/2026 (0.8993 on the same 31,527 test clips, against
> 0.5681 here); this model's verdict does not change.

**A 899 KB ternary decision model that hears.** 1,532,449 parameters. Its state is the last
10.24 s of 16 kHz audio, as 80 ms log-mel frames; its questions and options are text. It answers
typed questions about what it heard (has the speaker finished their turn, which word was said),
one forward pass per question, with the same engine, GGUF format, early exits and WASM build as
the other skycmd models. It was meant as the System 1 of a voice agent; it is not good enough to
be one.

- **Demo:** [huggingface.co/spaces/{{USER}}/s6-ear-nano](https://huggingface.co/spaces/{{USER}}/s6-ear-nano)
  (the microphone or sample clips; the audio never leaves your browser)
- **Code:** [github.com/maxgfr/skycmd](https://github.com/maxgfr/skycmd) (100 % Rust)
- **Pre-registration:** [docs/s6-ear-prereg.md](https://github.com/maxgfr/skycmd/blob/main/docs/s6-ear-prereg.md)
- **How it works:** [tutorial chapter 26](https://maxgfr.github.io/skycmd/tutorial/26-s6-ear)
- **All results, failures included:** [results page](https://maxgfr.github.io/skycmd/results/)

## Results

Test split, ternary, full depth (`runs/s6-ear-nano/eval/gate.json`):

| gate | n | s6-ear-nano | control (no Mimi pretraining, not published) | bar | verdict |
|---|---:|---:|---:|---:|---|
| end of turn (`ear:turn_end`) | 31,527 | 0.5681 | 0.7647 | 0.9263 (Smart Turn v3.2 on the same clips, Pipecat's report; the v1 harness read 0.8518) | **fail** |
| keywords, 12 classes (`ear:keyword`) | 4,890 | 0.8885 | 0.9217 | 0.90 | **fail** |
| zero-shot keywords, English | 1,000 | 0.2100 | 0.3020 | 0.20 (2 × chance) | pass |
| zero-shot keywords, French | 1,000 | 0.1330 | 0.1520 | 0.20 (2 × chance) | **fail** |

- **The pretraining task was learned, and did not transfer.** The Mimi pretraining reached a
  codebook-0 accuracy of 0.2196 on masked validation frames, against a technical bar of 0.00488.
  The control, which starts from random weights, is more accurate on every set: in this run the
  pretraining brought nothing to the decisions.
- **Calibration:** ECE over all test sources 0.0081 at full depth.
- **At its conformal thresholds** (a separate file, not published) the model uses 3.42 of its 4
  passes on average (early exit on 21.1% of the questions), with the same verdicts.
- **Not computed by the run:** the precision and recall of its end-of-turn answers, which the
  pre-registration asks for (Smart Turn v3.2: precision 0.783, recall 0.971), and the latency.

## What it does well, and badly

- **Keywords:** 0.8885 on the 12 classes of Speech Commands, close to the bar of 0.90 but under
  it, and under its own control (0.9217).
- **End of turn:** 0.5681, far under Smart Turn v3.2 (0.9263), which has 8M parameters, 5.2
  times more. Do not use it to decide when a speaker has finished.
- **Words it never trained on:** 0.2100 in English and 0.1330 in French with 10 options, where
  chance is 0.10.

## Exit thresholds in the file

The file is published as evaluated: its weights and metadata are those of the evaluated ternary
GGUF, byte for byte. It stores the exit thresholds its training run wrote (choice 0.5, score 0.9,
noul 0.68), at which it may stop before its last pass. The gates were read at full depth: to
answer as they were read, pass a threshold of 1 at run time (`setExitThreshold(1)` in the
browser). The demo does.

## Architecture

The s6-nano graph with an audio state (`skycmd-s2` GGUF tag, `state_kind = "audio"`): d 144,
2 heads, 2 weight-shared blocks looped twice, an exit after each of the 4 passes, ternary weights.
The front-end (16 kHz mono, 25 ms Hann window, 10 ms hop, 40 log-mel bands, 8 hops stacked into
one 320-value frame every 80 ms) runs inside the engine, natively and in WASM. The state stream is
a float projection of each frame; the question and options are BPE text (vocabulary 2,048) in a
static stream encoded once and cached, so the options can name words it never trained on.
Without Parallax, which the pre-registered ablation did not adopt: 1,532,449 parameters instead
of the 1,615,681 of the pre-registration.

## Limitations

- **Failed its gates.** Recommended for no use.
- One run, one seed. A first decision run of 31,480 steps was discarded before any evaluation for
  a memory defect (amendment 4).
- 10.24 s of context; the audio is resampled to 16 kHz.

## Usage

Runs in the skycmd engine only (browser, Node, Python, CLI, server). In the browser,
`Engine.decideAudio(type, instructions, optionsJson, pcm)` takes the options as a JSON array and
16 kHz mono PCM; the questions are
those of `crates/sky-data/src/audio.rs`:

- `noul`: "Has the speaker finished their turn?", options
  `false: the speaker is still talking or will go on` and
  `true: the speaker has finished their turn`;
- `choice`: "Which word was said?", options `yes`, `no`, `up`, `down`, `left`, `right`, `on`,
  `off`, `stop`, `go`, `silence: nothing was said`, `unknown: another word`.

Install and bindings: [install guide](https://maxgfr.github.io/skycmd/install/).

## Training

From scratch in Rust (mlx-rs) on an Apple M5, in a throttled mode, so the budgets are working
minutes: a masked pretraining on LibriSpeech `clean` that predicts Mimi's semantic codebook-0 ids
of masked frames (60 minutes, 48,626 steps; Mimi itself never runs, its codes come from
`shangeth/librispeech-mimi-codes`), then decision training (60 minutes, 48,205 steps) on end of
turn (Smart Turn v3 train), Speech Commands v0.02 keywords and MSWC words in English and French,
with ternary QAT over the last 10 %. Test sets: the official Smart Turn v3.2 test set and the
official Speech Commands test archive, which neither model trains on, and 50 held-out MSWC words
per language. All audio sources are CC-BY-4.0; only derived features were stored, and no audio is
redistributed with the model. Licenses:
[DATA_LICENSES.md](https://github.com/maxgfr/skycmd/blob/main/DATA_LICENSES.md).

## Files

- `s6-ear-nano.gguf`: ternary weights, the audio projection, tokenizer, temperatures and the exit
  thresholds of its training run. 899,200 bytes.
