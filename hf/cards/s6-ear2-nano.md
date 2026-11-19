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
  - end-of-turn
  - voice-agent
  - rust
  - wasm
  - skycmd
---

# s6-ear2-nano

> **Passed its pre-registered gate, every condition.** A 632 KB ternary end-of-turn detector:
> given the last 4 s of audio it says whether the speaker has finished their turn. On the
> 31,527 test clips of Smart Turn v3.2 it is right on **0.8993** (bars: ≥ 0.85 and ≥ the
> teacher − 0.08, the teacher measured at 0.9260 on the same clips through the same harness);
> at its validation threshold τ₈₅ it would cut a speaker off on **0.0847** of the not-finished
> clips (bar ≤ 0.20) while catching **0.8627** of the finished ones (bar ≥ 0.80); it decides in
> **2.39 ms** per window in WASM (bar 5 ms) from a file of **631,936 bytes** (bar 3 MB), and the
> browser build matches the native one to 1.2e-7 with the same decision on 20 of 20 clips. The
> gate is [`docs/s6-ear2-prereg.md`](https://github.com/maxgfr/skycmd/blob/main/docs/s6-ear2-prereg.md)
> §9, frozen before any line of v2's code, with the harness checked against Pipecat's published
> number before the test was read. Numbers:
> [BENCH.md, section s6-ear2-nano](https://github.com/maxgfr/skycmd/blob/main/BENCH.md#s6-ear2-nano-s6-ear-v2-an-end-of-turn-student-of-smart-turn-pre-registered-gate-passed)
> and [`runs/s6-ear2-nano/GATE.md`](https://github.com/maxgfr/skycmd/blob/main/runs/s6-ear2-nano/GATE.md).
>
> **When to take which model** (the sentence of the gate report). "Take v2 when size and speed
> matter: it gives up 2.7 points of accuracy against Smart Turn for a file of 0.63 MB instead of
> 8.7 MB and about 2.4 ms per decision in WASM instead of about 32 ms natively; take
> [Smart Turn v3.2](https://huggingface.co/pipecat-ai/smart-turn-v3)'s file when those points
> matter more."
>
> **Know before you use it.** On real conversational speech ([LiveKit's
> eot-bench](https://huggingface.co/datasets/livekit/eot-bench-data), 17,934 cuts, which the
> design never read) this model and its teacher are far weaker than on the test clips: AUC
> 0.7299 against 0.7838, and at τ₈₅ this model answers "finished" on 0.5253 of the real ends of
> turn — about half pass. Its weakest languages are Vietnamese (0.7749), Marathi (0.7907) and
> Bengali (0.7950). It is a student of Smart Turn: it cannot be better than its teacher's
> labels. Most of its test set is synthetic speech. The official test set is partly in-sample
> for the design choices, though never for the weights. **Try it with your microphone** on the
> site's [Listen page](https://maxgfr.github.io/skycmd/ear/) or in the
> [Space](https://huggingface.co/spaces/maxgfr/s6-ear2-nano); the demo runs a fallback timer
> next to the model, for the reason above ([How to use it](#how-to-use-it)).

**Its successor failed, so this stays the file to take.** [s6-ear3-nano](https://huggingface.co/{{USER}}/s6-ear3-nano)
is this graph trained again on real two-speaker conversations and on the silences the demo asks
at. On eot-bench read 0.2 s into each pause, it reads 0.7197 where this model reads 0.7448 on the
same cuts: it failed its pre-registered gate and is published as a result. It was designed after
this model's eot-bench reading, and its label cross-check was amended after a measurement on five
training sessions; its card discloses both.

**A dedicated model, not a shared one.** The first version, [s6-ear-nano](https://huggingface.co/maxgfr/s6-ear-nano),
shared one ternary model between end of turn and keywords on top of a codec pretraining, and
failed three of its four gates (end of turn 0.5681). This version drops all of that: one task,
no pretraining, a dedicated `turn` graph (a patch projection, 4 s2 encoder blocks, Smart Turn's
attention pooling, a 2-layer head; 1,306,178 ternary parameters), and a front end built for the
job — 40 log-mel bands plus 3 prosody channels (log f0, voicing, energy) per 10 ms hop. It was
trained for 240 working minutes on one Apple M5 as a student of Smart Turn v3.2, once, seed 1.

- **Demo:** in the browser, with the microphone or a self-test without one: the site's
  [Listen page](https://maxgfr.github.io/skycmd/ear/) and the Space
  [maxgfr/s6-ear2-nano](https://huggingface.co/spaces/maxgfr/s6-ear2-nano). The owner checked the
  microphone path on the site in Chrome on 17/11/2026; Safari, and the microphone inside the Hugging
  Face frame, are not checked yet. How the demo decides is in [How to use it](#how-to-use-it).
- **Code:** [github.com/maxgfr/skycmd](https://github.com/maxgfr/skycmd) (100 % Rust; the model is `sky_infer::turn`, the front end `crates/sky-infer/src/audio.rs`)
- **Pre-registration:** [docs/s6-ear2-prereg.md](https://github.com/maxgfr/skycmd/blob/main/docs/s6-ear2-prereg.md) (the design in §2, the gate in §9, amendment 1, the result block)
- **All results, failures included:** [results page](https://maxgfr.github.io/skycmd/results/)

## The gate, as read

The 31,527 clips of `pipecat-ai/smart-turn-data-v3.2-test`, the ternary GGUF as exported
(`runs/s6-ear2-nano/GATE.md`; 95 % bootstrap intervals, 1,000 resamples by clip):

| gate | condition | measured | verdict |
|---|---|---|---|
| G1 | accuracy at τ₀.₅ ≥ 0.85 and ≥ T − 0.08 (T = Smart Turn v3.2 CPU, same harness) | 0.8993 [0.8962, 0.9027] against T = 0.9260 | holds |
| G2 | false cut-off at τ₈₅ ≤ 0.20 | 0.0847 [0.0800, 0.0888] | holds |
| G2 | END recall at τ₈₅ ≥ 0.80 | 0.8627 [0.8577, 0.8680] | holds |
| G3 | WASM median ≤ 5.0 ms per decision | 2.39 ms (p95 2.41 ms) | holds |
| G3 | front end ≤ 0.10 ms per 10 ms hop in WASM | 0.0186 ms | holds |
| G3 | GGUF ≤ 3 MB | 631,936 bytes | holds |
| G3 | WASM parity: max \|Δp\| ≤ 1e-4, same decision | 1.2e-7, 20 of 20 clips | holds |

At τ₀.₅: precision 0.8776, recall 0.9271, F1 0.9017; AUC 0.9600 [0.9581, 0.9621], ECE 0.0160
[0.0138, 0.0191]. The silence timer (always "finished") scores 0.4979 on the same clips.

## On the same test clips

| model | accuracy | file | per decision |
|---|---:|---:|---:|
| **s6-ear2-nano** | **0.8993** [0.8962, 0.9027] | 631,936 bytes | 2.39 ms in WASM, 1.40 ms native |
| Smart Turn v3.2 CPU (int8, through the same harness) | 0.9260 | 8.7 MB | 31.7 ms native |
| Smart Turn v3.2 GPU (Pipecat's published number) | 0.9371 | | |
| v1's control (no Mimi pretraining, not published) | 0.7647 | | |
| [s6-ear-nano](https://huggingface.co/maxgfr/s6-ear-nano) (v1, failed its gate) | 0.5681 | 899 KB | |
| silence timer (always "finished") | 0.4979 | no parameters | |

The harness was checked before the test was read: Smart Turn v3.2 CPU through it gives 0.9260 on
these clips, within Pipecat's reported 0.9263 ± 0.003, and the timer gives the test's share of
finished clips, 0.4979 (`runs/s6-ear2-harness/check.json`).

By language (accuracy at τ₀.₅; the full tables are in `GATE.md`): German 0.9493, Korean 0.9449,
Dutch 0.9471, Portuguese 0.9421 on the strong side; English 0.9056 (the English human subsets
0.9187); **Vietnamese 0.7749, Marathi 0.7907 and Bengali 0.7950** on the weak side. By dataset,
the weakest is `mundo_1` at 0.7843.

## On real conversations (eot-bench, reported, not gated)

[LiveKit's eot-bench](https://huggingface.co/datasets/livekit/eot-bench-data) is real
conversational speech, which the design never read. One decision per labelled pause, cut at its
start; 17,934 cuts. These numbers are not comparable with LiveKit's FC@300 / FC@600, which come
from a policy sweep:

| model | AUC | false cut-off | eot recall |
|---|---:|---:|---:|
| s6-ear2-nano (at τ₈₅) | 0.7299 | 0.2230 | 0.5253 |
| Smart Turn v3.2 CPU (at 0.5) | 0.7838 | 0.3678 | 0.7998 |
| timer | 0.5000 | 1.0000 | 1.0000 |

The pre-registration fixed the reading in advance: "does not carry over" is said only if the
student's AUC is more than 0.10 below the teacher's. It is 0.054 below, so that sentence does
not apply. Said plainly: **both models are far weaker on these real conversational cuts than on
the test clips, and at τ₈₅ this model answers "finished" on 0.5253 of the real ends of turn —
about half pass.** On a real deployment, use τ₈₅ only together with the fallback timer below.

## How to use it

**The input.** 16 kHz mono PCM. The engine computes the front end itself — no external
extractor: a 25 ms Hann window, a 10 ms hop, 40 log-mel bands, and per hop the three prosody
channels (log f0, voicing, energy; pitch searched between 80 and 400 Hz by autocorrelation).
Each window is normalised per clip by Whisper's rule, and the model reads the **last 4 s**
(400 hops, 8 per position, 50 positions; a shorter clip gives fewer positions). One forward pass
gives the calibrated probability that the turn is finished, `σ(logit / temperature)` with the
stored temperature 1.4948.

**The two thresholds in the file.** Both come from validation, never from test: **τ₀.₅ = 0.5**
on the calibrated probability — the accuracy reading (0.8993 on the test clips) — and
**τ₈₅ = 0.7244**, the highest threshold whose validation END recall is at least 0.85 — the
reading for a voice agent that must not cut people off (false cut-off 0.0847, END recall 0.8627
on test). Answer "finished" when the probability reaches the threshold, else keep listening.

**In the browser or Node** (`sky-web`, the `TurnDetector` class; the module is built by
`scripts/build-wasm.sh`):

```js
import { TurnDetector } from './pkg/sky_web.js'
const det = new TurnDetector(new Uint8Array(ggufBytes))
det.push(pcmChunk)          // 16 kHz mono Float32Array, any chunk size
const p = det.probEnd()     // calibrated probability, from the stream's last 4 s
const end = p >= det.tau85() // or det.tau50() for the accuracy reading
const q = det.probEndPcm(clip) // one decision on a whole clip; the stream is untouched
det.reset()                 // a new turn
```

The front end computes each hop once and keeps the last 400, so streaming and batch features
agree bit for bit; `probEnd` costs the model's forward pass only (2.39 ms in WASM).

**Natively** (`sky-turn`, `crates/sky-infer/src/bin/sky-turn.rs`):

```sh
cargo run --release -p sky-infer --bin sky-turn -- s6-ear2-nano.gguf --records clips.jsonl --out probs.jsonl
cargo run --release -p sky-infer --bin sky-turn -- s6-ear2-nano.gguf --bench 1000
```

`--records` reads the 16 kHz WAV named by each JSONL row and prints `{id, logit, prob,
positions}` per item; `--bench` times decisions on a full 4 s window.

**In a voice agent** (the operating notes of the pre-registration, §10): run the model once when
the VAD reports a pause (Silero, or the engine's energy VAD), with Pipecat's `stop_secs` 0.2;
answer "finished" when the probability is at least the threshold (τ₀.₅ for accuracy, τ₈₅ for 85 %
END recall on validation), else keep listening; a fallback timer ends the turn after a longer
silence (Pipecat's `SmartTurnParams.stop_secs`); if speech resumes, the turn's audio is kept and
the model runs again at the next pause.

**How the demo decides** (the [Listen page](https://maxgfr.github.io/skycmd/ear/) and the
[Space](https://huggingface.co/spaces/maxgfr/s6-ear2-nano) run the same code). An energy VAD
over 10 ms frames finds the speech and the pauses; the model is asked once after 0.2 s of silence
following speech, then again every 0.2 s while the pause lasts; it answers "finished" at or above
the threshold, τ₈₅ by default (τ₀.₅ can be chosen); and a fallback timer ends the turn after
2.5 s of silence whatever the model says. The timer is there because of eot-bench: on real
conversations, at τ₈₅ the model catches only 0.5253 of the real ends of turn, about half
([`runs/s6-ear2-nano/GATE.md`](https://github.com/maxgfr/skycmd/blob/main/runs/s6-ear2-nano/GATE.md),
eot-bench), so without the timer a finished speaker could wait for an answer that never comes.
The pause length and the timer can be changed on the page.

## Latency and size

Apple M5, one thread, 1,000 decisions on a full 4 s window with the features already computed,
measured with no training running (`runs/s6-ear2-nano/latency/`):

| | native | WASM (SIMD128) |
|---|---:|---:|
| median per decision | 1.40 ms | **2.39 ms** |
| p95 | 1.42 ms | 2.41 ms |
| front end per 10 ms hop | 0.013 ms | 0.019 ms |

Smart Turn v3.2 CPU through onnxruntime, one thread, in the same session: median 31.7 ms per
decision. File: `s6-ear2-nano.gguf`, 631,936 bytes.

## Training data and licences

One run, 240 working minutes (305,700 steps, about 152 epochs), batch 128, seed 1, on an Apple
M5, in a braked mode, so the budget counts working minutes. Ternary QAT over the last 15 %,
EMA 0.999; the exported checkpoint is the EMA. The temperature and the two thresholds are fitted
on validation and stored in the file; the test set was read once, at the end.

- **Training and validation:** `pipecat-ai/smart-turn-data-v3.2-train` (revision `f8c95d28`,
  270,946 clips), **CC-BY-4.0**, 5 % validation by a hash of the clip id (seed 1234): 257,353
  training, 13,593 validation. A clip is dropped if its id is a test id or its decoded PCM
  hashes to a test clip's; the filter dropped no row (the sentence-level overlap cannot be
  checked, `spoken_text` being empty, as for Smart Turn's own numbers).
- **Test (gated):** `pipecat-ai/smart-turn-data-v3.2-test`, 31,527 clips, CC-BY-4.0 — Smart
  Turn v3.2's own official held-out set. Never trained on, never used for a threshold.
- **Second test (reported only):** `livekit/eot-bench-data`, CC-BY-4.0, real conversational
  speech, one decision per labelled pause; never trained on.
- **The teacher:** [Smart Turn v3.2](https://huggingface.co/pipecat-ai/smart-turn-v3)
  (`smart-turn-v3.2-cpu.onnx`, revision `f766f81d`), **BSD-2-Clause**, credited: its logit, its
  pooled embedding and its encoder states are the training targets on the training and
  validation clips only. Being a student, this model cannot be better than its teacher's labels.
  The teacher is never redistributed.

No audio is redistributed with the model: only the GGUF. Every licence, with the revision of
each source, is in [DATA_LICENSES.md](https://github.com/maxgfr/skycmd/blob/main/DATA_LICENSES.md).

## Disclosure (§11 of the pre-registration)

This version was designed after v1's test results, and its design study read measurements made
on 3,000 clips of the same test set: v1's two models per source, Smart Turn per source, Smart
Turn with 8, 4, 3, 2 and 1 s of context, and the trailing silence of 1,500 clips. The 4 s window
and the data change came from those readings, so **the official test set is partly in-sample for
the design choices, though never for the weights**: no test clip is trained on, and no threshold
is chosen on test. eot-bench, which the design never read, is the cleaner check, and it is
reported above. The run happened under the automatic brake of the machine, and CPU work of other
models ran during the pilot and part of the training (allowed by amendment 1; budgets count
working minutes); the pilot and the training were built from commits that differ by an unrelated
option, off by default, and nothing in the end-of-turn training path changed between them. The
latencies were measured with no training running.

## Files

- `s6-ear2-nano.gguf`: ternary weights, the window and prosody configuration, the temperature
  (1.4948) and the two validation thresholds (τ₀.₅ 0.5, τ₈₅ 0.7244), `skycmd-turn` architecture.
  631,936 bytes, sha256
  `204354653fd864708c3206f486f255d3fedff81956fd7b82aae560185a9be783`.
