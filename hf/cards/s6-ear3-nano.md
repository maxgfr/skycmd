---
license: apache-2.0
language:
  - en
  - fr
base_model: {{USER}}/s6-ear2-nano
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
  - failed-gate
  - negative-result
datasets:
  - pipecat-ai/smart-turn-data-v3.2-train
  - otoearth/otoSpeech-full-duplex-280h
---

# s6-ear3-nano

> **Failed its pre-registered gate: published as a result, not for use.** E1, E2 and E3 fail;
> A1, A2 and G3 hold. This is the end-of-turn graph of
> [s6-ear2-nano](https://huggingface.co/{{USER}}/s6-ear2-nano), unchanged, trained again from v2's
> weights on real two-speaker conversations and on the silences a voice agent asks at. On
> [eot-bench](https://huggingface.co/datasets/livekit/eot-bench-data) (LiveKit's real speech
> addressed to an agent, 14 languages), read at the demo policy's first question, 0.2 s into each
> pause (16,230 cuts from 7,342 turns), its AUC is **0.7197 against v2's 0.7448** on the same
> cuts: −0.0252, 95% interval [−0.0336, −0.0167], where the gate (E1) asked for at least +0.02
> with a lower bound above 0. Smart Turn v3.2 reads **0.7873** there. Its recall at a false
> cut-off of 0.10 is 0.0563 below v2's (E2: −0.0563 [−0.0832, −0.0315]), and outside English its AUC
> is 0.0209 below v2's (−0.0209 [−0.0308, −0.0111]), where a lower bound above −0.02 was asked (E3). It keeps v2's
> level on Smart Turn's 31,527 test clips (0.8978, bar 0.8893; A1) and is a drop-in file (2.39 ms
> per decision in WASM, 631,936 bytes, v2's tensor list; G3). On the held-out human-to-human
> conversations of its own corpus (reported, not gated) it reads **0.6878 against v2's 0.6031**: by
> the reading its pre-registration fixed before the run, **the human-to-human signal was learned
> and does not transfer to speech addressed to an agent.** The file to take stays
> [s6-ear2-nano](https://huggingface.co/{{USER}}/s6-ear2-nano) (or Smart Turn's 8.7 MB file when
> real conversations matter most). The file is published so that the result can be checked and
> run (owner's decision of 15/11/2026); the verdict does not change. Numbers:
> [BENCH.md, section s6-ear3-nano](https://github.com/maxgfr/skycmd/blob/main/BENCH.md#s6-ear3-nano-s6-ear-v3-an-end-of-turn-detector-trained-on-real-conversations-pre-registered-gate-failed),
> [`runs/s6-ear3-nano/GATE.md`](https://github.com/maxgfr/skycmd/blob/main/runs/s6-ear3-nano/GATE.md)
> and the result block (§17) of
> [`docs/s6-ear3-prereg.md`](https://github.com/maxgfr/skycmd/blob/main/docs/s6-ear3-prereg.md).
>
> **Disclosed with the result.** v3 was designed after v2's eot-bench reading, which motivated the
> headline test, the bar outside English and the expected per-language effects; no eot-bench file
> was opened for the design. Its label cross-check and its caps were amended after a measurement
> on five training sessions, before any validation or test session was read. One training run,
> one seed.

**What changed from v2, and why.** v2 passed its gate on Smart Turn's test clips and is weak on
real conversations; it also never trained on the silences it is asked about (its training clips
end at the end of speech, while the demo asks 0.2 s into a pause, then every 0.2 s up to 2.4 s).
v3 keeps v2's graph, front end, rung and GGUF keys, and changes only the training:

- **Real conversations, one speaker per channel.** [otoSpeech](https://huggingface.co/datasets/otoearth/otoSpeech-full-duplex-280h)
  (about 280 h of spontaneous English, CC BY 4.0, gated). The deployed energy VAD and policy are
  replayed on one channel; each pause the policy would ask about becomes a record, labelled from
  the other channel: **end of turn** when the other speaker takes the floor within 4 s (a run of
  at least 1 s that is not a backchannel), **hold** when the same speaker resumes first.
  Backchannels, overlaps, pauses nobody follows within 4 s and records near a redacted segment
  are dropped. The labels come from that rule, never from a generative model, and are hard
  labels: Smart Turn's opinion of these records is recorded, never a target.
- **The silences it is asked about.** The conversation records are cut where the policy asks
  (0.2 s into a pause with weight 0.5, the later asks sharing the rest), and Smart Turn's
  training clips are extended by 2.44 s of their own quietest frames, cut at a drawn silence
  (v2's view, the clip's end, on a quarter of them).
- **The run.** From v2's EMA weights, 120 working minutes on one Apple M5, half the rows
  conversation records and half Smart Turn clips, with v2's distillation targets on the clips.

**Checked before training** (committed before the first GPU step):

- **The labels against a second detector.** On the 109 validation sessions, each record's label
  was derived again from Silero VAD v6.2.3 (MIT), read at the record's instant: it agrees on
  **0.9509** [0.9488, 0.9530] of the 40,872 records Silero labels (bar ≥ 0.85), and Silero's path
  drops 0.1016 of the 45,494 records (bar ≤ 0.30). Silero never labels, selects or drops a
  training record.
- **No leak from the other speaker.** The silence after a pause, read alone, predicts the label
  with an AUC of at most **0.4396** on the 120,000 training records (bar 0.60): the other
  speaker's voice does not carry the label into this channel.
- **The harness.** v2's 0.8993 on Smart Turn's test, v2's 0.7299 and Smart Turn's 0.7838 on
  eot-bench at each pause's start (17,934 cuts) were reproduced before any v3 score was read.

- **Code:** [github.com/maxgfr/skycmd](https://github.com/maxgfr/skycmd) (100 % Rust for the model and the engine; the model is `sky_infer::turn`, the front end `crates/sky-infer/src/audio.rs`; the run `scripts/train-turn3.sh`)
- **Demo:** the [Space](https://huggingface.co/spaces/{{USER}}/s6-ear3-nano), the same microphone demo as v2's, on this file; the site's [Listen page](https://maxgfr.github.io/skycmd/ear/) keeps v2.
{{LINKS}}

## The gate, as read

The ternary GGUF as exported (`runs/s6-ear3-nano/GATE.md`). Smart Turn's test: 95 % bootstrap
intervals by clip, 1,000 resamples. eot-bench: a paired bootstrap of v3 − v2, 2,000 resamples by
turn (all the cuts of a turn together), seed 20261117.

| gate | condition | measured | verdict |
|---|---|---|---|
| A1 | accuracy at τ₀.₅ on Smart Turn's 31,527 test clips ≥ 0.8893 (v2's 0.8993 − 0.01) | 0.8978 [0.8946, 0.9013] | holds |
| A2 | false cut-off at τ₈₅ ≤ 0.20 | 0.0881 [0.0836, 0.0926] | holds |
| A2 | END recall at τ₈₅ ≥ 0.80 | 0.8597 [0.8544, 0.8650] | holds |
| E1 | eot-bench at start + 0.2 s, 14 languages: AUC(v3) − AUC(v2) ≥ +0.02, lower bound > 0 | −0.0252 [−0.0336, −0.0167] (0.7197 against 0.7448) | **fails** |
| E2 | the same cuts: recall at a false cut-off of 0.10, v3 − v2, lower bound > 0 | −0.0563 [−0.0832, −0.0315] (0.2754 against 0.3317) | **fails** |
| E3 | the 13 languages other than English: AUC(v3) − AUC(v2), lower bound > −0.02 | −0.0209 [−0.0308, −0.0111] (0.7179 against 0.7388) | **fails** |
| G3 | WASM median ≤ 5.0 ms; front end ≤ 0.10 ms per hop; GGUF ≤ 3 MB | 2.39 ms (p95 2.41 ms); 0.0176 ms; 631,936 bytes | holds |
| G3 | WASM parity on 20 test clips and 20 conversation test records: max \|Δp\| ≤ 1e-4, same decision | 5.96e-7 and 1.04e-6, 20 of 20 each | holds |
| G3 | the tensor list, shapes and types of v2's file | identical | holds |

## On real speech addressed to an agent (eot-bench, the headline)

Every `eot` or `hold` span of at least 0.2 s, cut 0.2 s after its start (holds over 5 s ignored):
16,230 cuts (6,798 `eot`, 9,432 `hold`). AUC, and the recall at a false cut-off of 0.10 (a
ranking measure, the ROC point, not a deployable threshold):

| cuts | s6-ear3-nano | s6-ear2-nano | Smart Turn v3.2 CPU |
|---|---:|---:|---:|
| all 14 languages | 0.7197; 0.2754 | 0.7448; 0.3317 | 0.7873; 0.3854 |
| English (3,902) | 0.7375; 0.3119 | 0.7737; 0.3469 | 0.8172; 0.4358 |
| the 13 others (12,328) | 0.7179; 0.2681 | 0.7388; 0.3144 | 0.7852; 0.3825 |

The announced goal, v3 at Smart Turn's level on these cuts, is not met: −0.0676 [−0.0763,
−0.0590]. v3 reads above v2 only in French, Korean, Dutch and Chinese. With the demo's policy
replayed span by span at τ₈₅, it answers "finished" before the 2.5 s timer on 0.8232 of the real
ends of turn (median 400 ms) and cuts off 0.2589 of the holds; v2, 0.8610 (200 ms) and 0.3737.

## On held-out human-to-human conversations (reported, not gated)

The 113 test sessions of otoSpeech (DualTurn's session split; session-disjoint, not
speaker-disjoint), 30,000 records at 0.2 s; no interval is computed for this test.

| records | s6-ear3-nano | s6-ear2-nano | Smart Turn v3.2 (recorded) |
|---|---:|---:|---:|
| all (30,000) | 0.6878 | 0.6031 | 0.6430 |
| speaker unseen in training (2,061) | 0.6663 | 0.6137 | 0.5989 |

On the speaker-unseen records, v3's recall at a false cut-off of 0.10 is below v2's (0.1831
against 0.2254).

## On Smart Turn's test clips

Accuracy at τ₀.₅ 0.8978 (v2 0.8993), AUC 0.9586 [0.9566, 0.9606], ECE 0.0068 [0.0053, 0.0102].
Extended by 0.2 s and 0.6 s of their own background (reported): 0.8959 and 0.8935, where v2
reads 0.8805 and 0.8512.

## How to use it (for checking the result)

The same input, engine and API as [s6-ear2-nano](https://huggingface.co/{{USER}}/s6-ear2-nano):
16 kHz mono PCM, the last 4 s, `TurnDetector` in WASM (`sky-web`) and `sky-turn` natively. The
file stores the temperature (1.2522) and two thresholds fitted on Smart Turn's validation clips,
**τ₀.₅ = 0.5** and **τ₈₅ = 0.6656**; the conversation operating point **τ_c = 0.6948** (false
cut-off ≤ 0.10 on the conversation validation records at 0.2 s) is reported and not stored.

```sh
cargo run --release -p sky-infer --bin sky-turn -- s6-ear3-nano.gguf --bench 1000
```

## Latency and size

Apple M5, one thread, 1,000 decisions on a full 4 s window with the features already computed,
measured after the training ended (`runs/s6-ear3-nano/latency/`): WASM median 2.39 ms (p95
2.41 ms), native 1.41 ms (p95 1.42 ms), front end 0.0176 ms per 10 ms hop in WASM; v2 in the same
session, 2.39 ms and 1.40 ms. File: `s6-ear3-nano.gguf`, 631,936 bytes.

## Training data and licences

One run, 120 working minutes (155,300 steps by the gate report), batch 128, seed 1, on an Apple
M5 under an automatic brake, so the budget counts working minutes. Started from v2's EMA
checkpoint; ternary QAT over the last 15 %, EMA 0.999; the exported checkpoint is the EMA. The
rate (0.0003873) was chosen by a pilot of four arms on conversation validation records; the
temperature and thresholds are fitted on validation; each test was read once, at the end.

- **Conversations:** [otoearth/otoSpeech-full-duplex-280h](https://huggingface.co/datasets/otoearth/otoSpeech-full-duplex-280h)
  (revision `4b3c57b8`), **CC-BY-4.0**, gated; credited here. Its access conditions were accepted
  by the project's owner (not to identify or link speakers, to respect the redactions). The
  split is DualTurn's `splits.json` ([anyreach-ai/dualturn-otospeech-turn-taking](https://huggingface.co/datasets/anyreach-ai/dualturn-otospeech-turn-taking),
  the session-to-split map only). **No audio, snippet, label or record derived from otoSpeech is
  published**, here or in the repository; this file holds the model's weights only.
- **Smart Turn's clips:** `pipecat-ai/smart-turn-data-v3.2-train` (revision `f8c95d28`) and
  `pipecat-ai/smart-turn-data-v3.2-test`, **CC-BY-4.0**, as v2.
- **eot-bench:** `livekit/eot-bench-data` (commit `ae834af9`), **CC-BY-4.0**, test only.
- **The teacher on the clips:** [Smart Turn v3.2](https://huggingface.co/pipecat-ai/smart-turn-v3)
  (`smart-turn-v3.2-cpu.onnx`, revision `f766f81d`), **BSD-2-Clause**: v2's targets on Smart
  Turn's training clips, reused; not redistributed.
- **The label cross-check:** Silero VAD v6.2.3 (**MIT**), on the validation sessions only; not
  redistributed.

Every licence, with the revision of each source, is in
[DATA_LICENSES.md](https://github.com/maxgfr/skycmd/blob/main/DATA_LICENSES.md).

## Disclosure (§11 of the pre-registration)

v3 was designed after v2's eot-bench reading: the overall, per-language and per-cohort numbers of
v2's gate file were seen, and that reading motivated making eot-bench the headline, the no-harm
bar outside English and the expected per-language effects. eot-bench has never been used for
training or for any choice of v2's or v3's training: no eot-bench file was opened for v3's
design, and the cut point (0.2 s) comes from the demo's policy and LiveKit's card, not from a
reading. The variable trailing silence was motivated by the microphone demo, through its code.
Smart Turn's test is still partly in-sample for the design, as v2's disclosure says. The label
cross-check and the caps were amended after a measurement on five training sessions, before any
validation or test session was read: the cross-check reads Silero's label at the record's
instant instead of matching pause ends, and the caps keep every end-of-turn record.

**What it means.** The corpus is human-to-human; eot-bench is human-to-agent: the labels of the
training conversations are the listener's behaviour, not an agent's need. The training
conversations are English; eot-bench has 14 languages. The conversation test is
session-disjoint, not speaker-disjoint.

## Files

- `s6-ear3-nano.gguf`: ternary weights, the window and prosody configuration, the temperature
  (1.2522) and the two validation thresholds (τ₀.₅ 0.5, τ₈₅ 0.6656), `skycmd-turn` architecture.
  631,936 bytes, sha256
  `048fafa6a7ff427a457c3957a50da0418c64d3c64e55b3b30d6a5b6e77d20cde`.
