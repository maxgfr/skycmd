# s6-ear v2: design study (end of turn that works)

Written on 15/11/2026, before any v2 code, data or training. Read-only on the repository. No GPU was used. Every number marked "measured" was computed for this study on the CPU, from v1's files or with the pinned Smart Turn ONNX file already in the local cache. The method is given with each number.

**Disclosure.** Several measurements below read clips of the official Smart Turn v3.2 test set: 3,000 clips, seed 0, drawn from v1's `data/ear/jev/turn_end.jsonl`. They were run with v1's two models and with Smart Turn. They shaped the v2 design, notably the 4-second window. The pre-registration draft repeats this.

## 0. Summary

1. **v1 lost for three reasons, and none of them is codec tokens.**
   - **It never read codec tokens.** v1's input was 40-band log-mel. The Mimi codebook-0 ids were only a pretraining *target*.
   - **The pretraining hurt.** The pretrained model underfits its own training clips (0.702) where the control reaches 0.861.
   - **v1 trained on the wrong data release.** It used `smart-turn-data-v3-train`. 17.4% of the v3.2 test comes from sources that release does not contain.
   - **Even the control is far below the teacher.** On the sources it trained on, it trails Smart Turn by 15 points (chirp3_1: 0.785 against 0.941).
2. **v1's bar was measured wrong.** `scripts/ear-baseline.py` right-pads short clips, because WhisperFeatureExtractor pads at the end. Smart Turn's reference code left-pads them (`audio_utils.truncate_audio_to_last_n_seconds`).
   - **With the v1 harness:** Smart Turn scores 0.8518 on the 31,527 test clips.
   - **Pipecat's own report:** 0.9263 (CPU int8) and 0.9371 (GPU fp32).
   - **On 3,000 of the test clips (measured):** the v1 harness gives 0.8437 and the reference padding gives 0.9243.
   - **Consequence.** Every published sentence saying "0.5681 against 0.8518 for Smart Turn" understates the gap by about 7.5 points. They need a correction (SPEC-S6 §16, BENCH, big-run report §4, site).
3. **No published model under 2M parameters is known to work on this task.**
   - The smallest open audio-only end-of-turn models with good numbers are 5–8M parameters: Smart Turn 8M; p99lab turn-1-mini 5.3–6.1M, whose single-channel gains are small.
   - The only sub-1M end-of-turn model found (SpeculativeETD's GRU, about 202K) reaches 79.7% on synthetic and 69% on real speech.
   - Tiny models do work for neighbouring tasks:
     - Silero VAD, about 309K parameters including a fixed STFT;
     - Pipecat's backchannel detector, 734K parameters, causal conv and GRU, 0.3 ms per 40 ms chunk;
     - BC-ResNet keyword spotting, 96.9% at 9.2K parameters.
4. **Recommended design.**
   - **Input.** A 1.3–2.7M-parameter ternary transformer (the size fixed by the probe) reads the last 4 s of 40-band log-mel plus 3 prosody channels (log f0, voicing, energy), at 40 ms per position (100 positions).
   - **Head.** Smart Turn's attention-pooling head.
   - **Training.** On `smart-turn-data-v3.2-train`, distilled from Smart Turn v3.2 at three levels:
     - the logit;
     - the 384-d pooled embedding;
     - the encoder states, pooled to 25 steps.
   - **Teacher outputs.** All three are verified to be exposable from the ONNX graph.
   - **Run.** It runs after a VAD pause, as Smart Turn does.
   - **Model size.** Chosen before training by a latency probe: the largest of a ladder of sizes whose WASM median stays under 4 ms.
5. **Odds** (§6).
   - Clearly better than v1's control (test accuracy ≥ 0.80): about 85%.
   - Passing a pragmatic gate: accuracy ≥ 0.85, false cut-offs ≤ 20% at 85% recall, WASM ≤ 5 ms, ≤ 3 MB. About 40%.
   - Within 3 points of Smart Turn: about 5%.
6. **Cost.**
   - About 3,000 lines of code, in a new `turn` family next to `cast`.
   - Downloads: 41.4 GB streamed shard by shard (under 1 GB on disk at a time), plus 2.0 GB of eot-bench.
   - About 10–15 GB of features and teacher targets on disk.
   - About 1–2 hours of CPU for features and teacher.
   - About 300 working minutes of GPU: 48 for the pilot, 240 for the main run, plus a smoke run.

## 1. Why v1 failed (from its own files)

### 1.1 What v1 was

- **Input.**
  - The front end (`crates/sky-infer/src/audio.rs`) computes 16 kHz PCM with a 25 ms periodic Hann window and a 10 ms hop, and a 512-point radix-2 FFT.
  - It keeps 40 HTK mel bands from 20 Hz to 8 kHz, with `y = (log10(max(mel, 1e-8)) + 2) / 4`. No per-clip normalisation.
  - 8 hops are stacked into one 320-value frame every 80 ms.
  - The state is the last 128 frames (10.24 s) on the grid of the clip start, with no padding (a variable-length sequence).
  - The log is computed from IEEE basic operations, so native and WASM agree to 1e-12.
  - There is **no pitch, voicing or pause feature.**
- **Model.** The s6-nano graph with `audio_proj` `[144, 320]`:
  - d 144, ff 288, 2 blocks × 2 loops, 2 heads;
  - a BPE text stream for the question and its options;
  - 1,532,449 parameters as trained (no Parallax), 899,200 bytes ternary.
  - It is multi-task: `turn_end` 0.45, `keyword` 0.30, `keyword_zs` 0.25 (`scripts/train-ear.sh`).
- **Pretraining.** 60 min (48,626 steps): BEST-RQ-style masked prediction of the **Mimi codebook-0 id** of each masked 80 ms frame, on LibriSpeech train-clean-100. Final `val_acc_cb0` 0.2196 (`runs/s6-ear-nano/pretrain/log.jsonl`).
- **Decision training.** 60 min, batch 64, QAT over the last 10%:
  - the model: 48,205 steps;
  - the control: 49,131 steps;
  - about 1.4M turn-end examples each, about 6.4 epochs of the 215,655 train clips.
- **The control** (pre-registered) is the same model, data and budget without the Mimi pretraining (`docs/s6-ear-prereg.md` §4).

### 1.2 Results (test, 31,527 clips of `smart-turn-data-v3.2-test`)

| | accuracy | precision | recall | source |
|---|---:|---:|---:|---|
| always "finished" (a silence timer, §1.5) | 0.4979 | 0.498 | 1.000 | `baseline-smart-turn.json` (`positive_rate`) |
| s6-ear-nano | 0.5681 | 0.555 | 0.669 | `gate.json`; precision and recall measured from `probs-full.jsonl` |
| control (no Mimi) | 0.7647 | 0.694 | 0.852 | `gate.json`; precision and recall measured on the 3,000-clip sample |
| Smart Turn v3.2 CPU, **v1 harness** (right padding) | 0.8518 | 0.783 | 0.971 | `baseline-smart-turn.json` |
| Smart Turn v3.2 CPU, **reference** (Pipecat's report) | 0.9263 | 0.909 | 0.947 | HF `benchmarks/smart-turn-v3.2-cpu.md` (generated 2026-01-07) |
| Smart Turn v3.2 GPU fp32 (Pipecat's report) | 0.9371 | 0.931 | 0.944 | HF `benchmarks/smart-turn-v3.2-gpu.md` |

v1's probabilities on `turn_end` cluster near 0.5. Measured quantiles from 10% to 90%: 0.373, 0.470, 0.514, 0.549, 0.580. Its test NLL is 0.663, against ln 2 = 0.693 for a coin.

### 1.3 Measured for this study (CPU, 3,000-clip samples, seed 0)

**No pipeline bug between training features and the deployed path.** The WAV path (`Engine::decide_audio`) and the `.npy` path give the same accuracy on the same test clips:

| model | WAV path | `.npy` path |
|---|---:|---:|
| s6-ear-nano | 0.5723 | 0.5720 |
| control | 0.7533 | 0.7533 |

**Both v1 models underfit; neither overfits.**

| model | train | val | test |
|---|---:|---:|---:|
| s6-ear-nano | 0.702 | 0.704 | 0.572 |
| control | 0.861 | 0.857 | 0.753 |

**The pretraining slowed learning instead of helping it.**

- **Val turn-end accuracy during training** (`log.jsonl`, exit 3, 379 val items, SE about 0.018):
  - the control reaches 0.80 by step 9,000, then plateaus between 0.82 and 0.86 until step 49,131;
  - the pretrained model stays between 0.45 and 0.59 until step 25,000, reaches 0.69 at 41,000 and ends at 0.72 (ternary).
- **The cause is not settled.** Two hypotheses remain open: a representation tuned to phonetic content of read English, or a rate unsuited to a pretrained init. Either way, the pretraining cost about 20 points.

**v1 trained on an older release than its test.**

- **The sources v1 trained on.** `smart-turn-data-v3-train` has chirp3_1, chirp3_2, rime_2, human_5, the three orpheus sets and human_convcollector_1.
- **What the v3.2 test adds.** liva_1 (3,831), midcentury_1 (1,044), mundo_1 (496) and chirp3_3_short (104): 5,475 clips, 17.4% of the test, from sources v1 never saw.
- **v3.2-train has them.** Its card credits Liva AI, Midcentury and MundoAI, and adds CC-0 café and office noise.

**Control against the teacher, per source** (3,000-clip sample; teacher with reference padding):

| source | n | control | Smart Turn v3.2 CPU |
|---|---:|---:|---:|
| chirp3_1 (seen) | 1,560 | 0.785 | 0.941 |
| chirp3_2 (seen) | 798 | 0.728 | 0.883 |
| liva_1 (unseen) | 379 | 0.726 | 0.934 |
| midcentury_1 (unseen) | 97 | 0.526 | 1.000 |
| mundo_1 (unseen) | 55 | 0.691 | 0.782 |
| all | 3,000 | 0.7533 | 0.9243 |

Two more measurements on the same sample:

| | teacher | control |
|---|---:|---:|
| AUC | 0.978 | 0.844 |
| false cut-off at END recall ≥ 0.85 (threshold chosen on the sample, Speko's protocol), all languages | 0.044 | 0.334 |
| same, English (724 clips) | 0.037 | 0.446 |

The teacher and the control agree on 77.2% of clips.

**Where the loss is.** The largest part of the loss is on sources v1 *did* train on, so it is a model and training loss, not only a data shift.

### 1.4 The harness defect (measured)

On the 3,000-clip sample:

| harness | accuracy | false positives (share of not-finished clips) |
|---|---:|---:|
| v1's `ear-baseline.py` | 0.8437 | 0.271 |
| reference left padding | 0.9243 | 0.096 |

Pipecat's "FPR 4.73%" is a share of *all* clips: 0.096 × 0.5 ≈ 0.048.

The cause is the padding. With right padding, a short clip ends in zeros, which looks like a finished turn. The test clips are short: median 89 frames of 80 ms (7.1 s), quartiles 5.0 s and 9.8 s.

### 1.5 Two facts about the test set that matter for the design

**The clips are trimmed at the end of speech.**

- **Trailing silence (measured, 1,500 clips):** frames 40 dB below the clip's loudest frame, counted from the end.
  - Median 0 ms in both classes.
  - 90th percentile 160 ms (not finished) and 180 ms (finished).
- **Pause length carries no label information here.** On this set, a "VAD + silence timer" can only answer "finished" for every clip: accuracy 0.4979. That is the same order as Speko's 46.9% for the timer on real clips.

**Context length (measured).** Smart Turn v3.2 CPU, reference padding, fed only the last N seconds:

| last N s | 8 | 4 | 3 | 2 | 1 |
|---|---:|---:|---:|---:|---:|
| accuracy | 0.9243 | 0.9190 | 0.9033 | 0.8677 | 0.7487 |
| English accuracy (724 clips) | 0.9448 | 0.9268 | 0.9102 | 0.8425 | 0.6588 |

The last 4 s keep almost all of what the teacher uses. Below 3 s, the loss is large: the cue is phrase-level, not only the last syllable.

### 1.6 Data and keywords in v1

- **Smart Turn data.** v3-train: 227,025 clips, split into 215,655 train and 11,370 val by a 5% hash of the id; no id collision with the test. Test: v3.2-test, 31,527 clips, 23 languages, English 7,820.
- **Keywords.**
  - The control already passes the keyword bar: 0.9217 ≥ 0.90.
  - The pretrained model fails it at 0.8885: this failure is also the pretraining's.
  - Zero-shot keywords are near chance for both models: en 0.21 / 0.30; fr 0.133 / 0.152.

## 2. What working end-of-turn detectors do (late 2026)

### 2.1 Smart Turn v3.x (Pipecat / Daily)

**Architecture** (`train.py` on `main`, read 15/11/2026):

- **Encoder.** The `openai/whisper-tiny` encoder with `max_source_positions = 400`. Input `[80, 800]`: Whisper log-mel over 8 s, 80 bins, 10 ms hop, normalised by WhisperFeatureExtractor. Two convolutions bring it to 50 Hz; then 4 transformer layers, d 384.
- **Attention pooling.** `Linear(384, 256) → tanh → Linear(256, 1)`, a softmax over time, then a weighted sum.
- **Classifier.** `Linear(384, 256) → LayerNorm → GELU → Dropout(0.1) → Linear(256, 64) → GELU → Linear(64, 1)`.
- **Measured from the ONNX file.** 8,033,089 initializer values. The `logits` output is already a sigmoid probability.

**Loss and optimiser.**

- BCE-with-logits, with `pos_weight` = negatives / positives in the batch, clamped to [0.1, 10].
- AdamW (HF Trainer) at lr 5e-5, 4 epochs, batch 384, warmup 0.2, cosine schedule, weight decay 0.01. Best checkpoint by F1.
- Quantisation: static int8 QDQ, calibrated on 1,024 training clips.
- **Caveat.** The file at `main` now points at `smart-turn-v3.3-alpha1` datasets. The v3.2 recipe may have differed; not verified.

**Input protocol.**

- The last 8 s of the turn, **zero-padded at the start**. Threshold 0.5.
- Run after Silero VAD reports silence; Pipecat's default VAD `stop_secs` is 0.2.
- If the model says "incomplete" and the silence goes on past `SmartTurnParams.stop_secs`, the turn ends anyway.

**Data and augmentation.**

- **v3.2 train.** 270,946 clips; 41,400,926,042 bytes in 83 parquet shards; CC-BY-4.0. Columns: `audio`, `id`, `language`, `endpoint_bool`, `midfiller`, `endfiller`, `synthetic`, `spoken_text` (null), `dataset`.
- **v3.2 test.** 31,527 clips, 4,837,751,629 bytes, 10 shards, CC-BY-4.0.
- **Noise.** Café and office noise from 14 CC-0 Freesound recordings is mixed into both sets. The SNR and the share of noisy clips are not published.
- **v3.2 changes** (blog, 07/01/2026). A short-utterance dataset and a fix for a padding bug in training (GitHub issue #35) give "40% fewer miscategorisations" on short utterances.

**Evaluation.** Pipecat's `benchmark.py` on the test set. It reports accuracy, precision, recall, F1, FPR and FNR as shares of all samples, per language and per dataset.

| language | v3.2 CPU accuracy |
|---|---:|
| Korean | 96.96% |
| English (7,820 clips) | 94.26% |
| French | 94.09% |
| Bengali | 83.80% |
| Marathi | 82.43% |
| Vietnamese | 79.38% |

Per dataset, v3.2 CPU ranges from mundo_1 84.27% and chirp3_2 89.31% up to midcentury_1 98.47%.

**Sizes, licence, loading.** Repository `pipecat-ai/smart-turn-v3`, BSD-2-Clause:

| file | size |
|---|---:|
| `smart-turn-v3.0.onnx` | 8.76 MB |
| `smart-turn-v3.1-cpu.onnx` | 8.68 MB |
| `smart-turn-v3.1-gpu.onnx` | 32.4 MB |
| `smart-turn-v3.2-cpu.onnx` | 8.68 MB (int8) |
| `smart-turn-v3.2-gpu.onnx` | 32.4 MB (fp32) |

- The ONNX input is `input_features [batch, 80, 800]`; the output is `logits`.
- v1 pinned revision `f766f81d3cfdf7737ac64aad813d91bbfd56bf93`.

**Latency.**

- Published: 12.6 ms on a c7a.2xlarge, 59.8 ms on a c8g.medium, 3.3 ms on an L40S.
- Measured on this M5, single thread, int8: median 31.9 ms; 35.7 ms with the extra outputs of §3.4.

**Smaller variants.** No distilled or streaming Smart Turn exists. The same team's `pipecat-ai/backchannel-detector` (BSD-2) is a streaming causal conv encoder followed by a GRU:

- **Size.** "734k parameters"; the Hub metadata says 961K.
- **Streaming.** 40 ms chunks, about 0.3 ms CPU per chunk.
- **Task.** Backchannel versus interruption, a different task from end of turn.

### 2.2 Other systems, with sizes and numbers

| system | input | size | reported | licence | source (date) |
|---|---|---|---|---|---|
| Speko benchmark | 200 real English clips from `smart-turn-v3.1-test`; threshold = least false cut-off with END recall ≥ 85%, chosen on the evaluated set | – | Smart Turn v3.2: 94.0% acc, 1.0% false cut-off, 32 ms warm CPU. VAD + silence timer: 46.9%, 100% false cut-off. On TTS clips Smart Turn falls to 62.4% / 57.7% | page CC-BY-4.0 | speko.ai/blog/voice-agent-turn-taking-benchmark (07/08/2026) |
| LiveKit eot-bench | real human-to-agent turns, 14 languages, every pause ≥ 100 ms labelled `eot` or `hold` | – | English, FC@300 / FC@600 (lower is better): Smart Turn v3.2 46.8% / 22.9%; turn-1-mini 39.6% / 18.7%; LiveKit v1.1 13.4% / 7.5%; VAD (silence only) 79.3% / 33.4% | code Apache-2.0, data CC-BY-4.0 (card) | github.com/livekit/eot-bench; HF `livekit/eot-bench-data` (modified 13/11/2026) |
| p99lab turn-1-mini | audio, causal, 20 ms frames, 2 conv + 6 transformer layers, d 256 | 5.3M (single channel) to 6.1M learned | TurnBench end of turn 0.975 recall / 7.5% FP (2 channels). Rules alone get 0.973 / 0.079; single channel 0.920 / 0.098 / 938 ms | Apache-2.0 per the LiveKit PR; the README gives none | github.com/P99Lab/turn-1-mini (Oct 2026) |
| "Less can be More" (Amazon) | frozen Zipformer2 (70M) + 5 prosody features at 25 Hz, 7-frame causal conv head | head 261K trainable | F1 / false alarms: acoustic 0.927 / 9.2%; prosody only 0.828 / 25.6%; acoustic + prosody 0.930 / 7.8%; adding text 0.909. EOT declared after 8 frames (320 ms) above τ | proprietary telephony data, 33 h | arxiv.org/html/2609.11066 (10/09/2026) |
| SpeculativeETD | 40-band log-mel, 100 ms chunks, Conv2D + GRU(64) | about 202K (the paper also says 1M and 11M) | 79.7% acc synthetic, 69.0% real; wav2vec2 81.2% real. 0.26 ms per call on an iPhone 12 mini | OpenETD: MultiWOZ TTS Apache-2.0, YouTube URLs only, Buckeye academic | arxiv.org/html/2503.23439v2 (Apr 2026) |
| OpenYap Turn | 2-channel audio, VAP-style | 72.4M | TurnBench end of turn rank 1: 0.978 / 7.7% / 298 ms | Open Yap 1K CC-BY-4.0, on request | huggingface.co/blog/TheAgenticDataCompany/openyap-turn (18/10/2026) |
| Silero VAD v5/v6 | 512-sample chunks, STFT-as-conv → 4 conv → LSTM(128) | about 309K including a fixed 66K STFT basis (about 243K trained), about 1.2 MB; v6.2.x released 2025–2026 | the reference for "tiny and adopted", not end of turn | MIT | github.com/snakers4/silero-vad; soniqo.audio/guides/vad (secondary) |

**The prosody-only result.** In the Amazon paper, prosody alone gives F1 0.828 with 25.6% false alarms. The gain over acoustic features alone is small (0.927 → 0.930). Silence duration and Δf0 carry it. On the Smart Turn test, silence duration is uninformative (§1.5), so prosody would have to come from f0 and energy alone.

**Accuracy against size, as reported.** Each benchmark below measures something different; none of them puts these models side by side.

| model | size | result |
|---|---|---|
| Smart Turn v3.2 | 8M | 92.6–93.7% on its test |
| turn-1-mini | 5–6M | strong on TurnBench, mostly from its 2-channel rules |
| SpeculativeETD's GRU | about 0.2M | 69% on real speech |

No sub-2M model reports a result on Smart Turn's test set.

## 3. The smallest design with decent odds

### 3.1 Input, computable in Rust and WASM

- **Kept.** The v1 front end: 40 log-mel bands, 25 ms window, 10 ms hop. It is tested, WASM-identical and cheap.
- **Added: 3 prosody channels per hop.**
  - **Computation.** From a 40 ms Hann window, take a 1,024-point FFT and its power spectrum. The inverse FFT of that power spectrum gives the autocorrelation; normalise it by the window's own autocorrelation.
  - **Lag range.** 2.5–12.5 ms, that is 80–400 Hz.
  - **The three channels.**
    - voicing = the peak normalised autocorrelation;
    - log2(f0 / 100 Hz) when voiced, 0 otherwise, with parabolic peak interpolation;
    - log energy.
  - **Cost.** Two 1,024-point FFTs per hop: well under 0.05 ms native.
  - **Why.** The Amazon ablation ties the prosody gain to Δf0. Whisper's 80 bins only partly resolve pitch harmonics. Whether the channels help a tiny model is an open inference; the pilot does not test it, to save time (open choice in the pre-registration).
- **Normalisation.** Whisper's rule, applied per window: subtract the window's maximum log-mel and clamp at 8 decades below it. Browser microphones differ in gain, and Smart Turn's input is normalised the same way.
- **Window.** The last 4.0 s (400 hops). The teacher keeps 0.919 of 0.924 there (§1.5). Shorter clips are masked, not padded (v1's variable length); Smart Turn's zero padding is only a feature of its fixed-size input.
- **Positions.** 4 hops × 43 channels = 172 values per 40 ms position, giving 100 positions.
- **Streaming.** The front end runs on each 10 ms hop as audio arrives, so the cost of a decision is the model forward alone. This needs a small `FeatureStream` in `audio.rs`; v1 recomputes the log-mel over the whole buffer at every call.

### 3.2 Model (a new `turn` family, cast-style)

| part | parameters |
|---|---:|
| `patch_proj` 172 → d (float) | 27,680 at d 160 |
| 4 s2 encoder blocks (LayerNorm, qkv, output, GELU FFN; RoPE; ternary under QAT; no loops, no Parallax) | 4 × 307,520 at d 160 / ff 640 = 1,230,080 |
| attention pooling, Smart Turn's: d → 64 → 1 | 10,369 at d 160 |
| head: d → 64 → 1 | 10,369 at d 160 |
| final norm | 160 |
| **total at d 160, ff 640, 4 blocks** | **about 1.28M** |
| same at d 192, ff 768, 4 blocks | about 1.83M |

- **Training-only heads, not exported.**
  - A projection d → 384 of the pooled vector.
  - A projection d → 384 of the positions, average-pooled by 4 to 25 steps.
- **File.** About 0.4–0.6 MB ternary at 1.3–1.8M parameters, judged by s6-cast's packing. To be measured.
- **Loops are not used.** The budget binds on compute per decision, not on file size. Distinct blocks give more capacity for the same compute.
- **Compute per decision.** About 0.25 GFLOP at d 160 and about 0.37 GFLOP at d 192.
  - For comparison, v1 does roughly 0.2 GFLOP in 3.4–4.0 ms native and 6.7 ms in WASM (`wasm.json`, `eval-full.json`).
  - **Whether d 192 fits a 5 ms WASM bound is not known.** Hence the latency probe on random ternary weights before training. It picks the largest of:
    1. d 192 × 6 blocks;
    2. d 192 × 4;
    3. d 160 × 4;
    4. d 160 × 4 at 80 ms positions.
  - The rule: WASM median ≤ 4 ms.

### 3.3 Decision rule (as products use it)

- Run once when the VAD reports a pause: Silero, or the engine's own energy VAD in the demo, with Pipecat's `stop_secs` 0.2.
- Answer "finished" if p ≥ τ, else keep listening.
- A fallback timer ends the turn after a longer silence (Pipecat's `SmartTurnParams.stop_secs`).
- If speech resumes, the turn's audio is kept and the model is run again at the next pause.
- τ is 0.5 for accuracy. A second τ is chosen on validation for 85% END recall, which is Speko's operating point.

### 3.4 Teacher: Smart Turn v3.2, on its own training clips

**Three targets, all verified exposable** (measured on the cached CPU file):

| target | ONNX tensor | shape |
|---|---|---|
| pre-sigmoid logit | `linear_28_DequantizeLinear_Output`; sigmoid of it equals `logits` to 1e-5 | 1 |
| pooled embedding | `sum_1` | 384 |
| final encoder states | `layer_norm_8_DequantizeLinear_Output` | 400 × 384 |

Any of them is obtained by appending the tensor name to the graph outputs.

- **Encoder states.** The last 200 frames (4 s) are average-pooled by 8, giving 25 × 384.
- **Storage.** About 2.6 GB at u8 with a per-vector scale, or 5.2 GB at f16.
- **Teacher input.** Prepared exactly as Pipecat does: HF `datasets` decoding, the last 8 s left-padded, WhisperFeatureExtractor. The teacher sees 8 s; the student sees 4 s.
- **Cost.** About 36 ms per clip on one M5 core. 271K clips take about 2.7 CPU-hours, about 20–30 min with 8 processes.
  - It runs on each parquet shard while the shard is on disk, alongside the Rust feature pass.
  - The fp32 GPU file (+1.1 points on the test) would cost an estimated 2–3× more. Not measured; open choice.
- **Why distil the states and not just the logit.** The teacher carries ASR pretraining from Whisper-tiny's 680k hours, which a 1.3M student trained from scratch cannot get from 271K clips. The teacher's lexical knowledge is the most likely part of the 15-point gap in §1.3.
  - Layer-wise hidden-state distillation is the method that kept most of HuBERT at a quarter of its size (DistilHuBERT, arXiv 2110.01900).
  - Long, consistent schedules matter for students ("a good teacher is patient and consistent", Beyer et al., arXiv 2106.05237; not re-read today).
  - Using it at a 6× compression *and* from scratch is an inference, not a sourced result.

### 3.5 What exists and what must be written

**Exists:**

- the log-mel front end, WAV and NPY I/O and the resampler (`crates/sky-infer/src/audio.rs`);
- the Smart Turn parquet fetcher with test-id exclusion and shard dropping (`crates/sky-data/src/audio.rs::fetch_smart_turn`, `tail_pcm`, `write_clip`);
- the from-disk audio batch loader (amendment 4 of v1);
- the s2 encoder blocks, ternary QAT, μP rate transfer and brake modes (sky-train);
- the cast family as a template for a non-question model: config, data, model, train, gguf, golden;
- the WASM build and the parity harness (`crates/sky-web`);
- the Smart Turn ONNX pinned in `scripts/ear-baseline.py`.

**Must be written:** the list by file is in the pre-registration draft, §4.5. About 3,000 lines:

- prosody channels and `FeatureStream`;
- the `turn` family in sky-infer, sky-train and sky-web;
- a `fetch-turn` subcommand;
- the teacher script;
- the fixed baseline harness;
- the gate script;
- the eot-bench cut builder;
- `scripts/train-turn.sh`;
- tests and golden files.

## 4. Data

| data | use | size | licence | note |
|---|---|---|---|---|
| `pipecat-ai/smart-turn-data-v3.2-train` | train (95%), val (5% by hashed id) | 270,946 clips, 41,400,926,042 bytes, 83 shards | CC-BY-4.0 (card), modified 29/09/2026 | streamed shard by shard, deleted after features and teacher |
| `pipecat-ai/smart-turn-data-v3.2-test` | **test, gated** | 31,527 clips | CC-BY-4.0 | already on disk from v1 as 16 kHz WAV tails ≤ 10.24 s (`data/ear/wav/smart-turn`, 6.8 GB). No new download |
| `livekit/eot-bench-data` (`all`, revision `main`) | second test, reported: real human speech | 7,342 turns, 1,980,396,015 bytes (English 2,288 turns, 590,974,169 bytes) | CC-BY-4.0 on the card; the code repository is Apache-2.0 | per-pause labels `eot` / `hold` with timings, 14 languages |
| Smart Turn v3.2 CPU ONNX (cached) | teacher, baseline | 8.68 MB | BSD-2 | revision `f766f81d…` |
| Smart Turn v3.2 GPU ONNX (optional) | stronger teacher | 32.4 MB | BSD-2 | same repository |

**Splits and leakage.**

- **Test ids.** Train clips whose id is in the test are dropped (v1's rule).
- **Exact duplicates, new rule.** Also drop train clips whose decoded 16 kHz int16 PCM hash equals that of a test clip.
- **Sentence-level overlap cannot be checked.** `spoken_text` is null, so the same sentence read by another TTS voice may sit in both sets. Smart Turn's own numbers share this.
- **Validation is optimistic.** v1's control: val 0.857, test 0.753. It is used for rate, checkpoint and threshold choices, never as evidence.

**Comparability.**

- **Full test set.** Accuracy at 0.5 is comparable with Pipecat's 92.63% / 93.71% once the harness pads on the left. Our WAVs come from our own decoder; the 3,000-clip sample gives 0.9243 against an official 0.9263 overall, which is consistent (SE about 0.005).
  - A pre-training check on all 31,527 clips must reproduce 0.9263 ± 0.003.
- **Speko's 94% / 1%** comes from 200 English human clips of v3.1-test, with the threshold chosen on those same clips. The closest analogue here is the English human subsets of the test: human_5 and human_convcollector_1.
  - We report the oracle-threshold number for comparison and the validation-threshold number for the gate.
- **eot-bench.** LiveKit's FC@300 / FC@600 need its full policy sweep. Our harness asks a narrower question: cut each labelled pause at its start and decide `eot` or `hold`. It is comparable to Smart Turn run through the same cut, not to LiveKit's table.

**Disk plan** (about 25 GB free; v1's `data/ear` already takes 31 GB):

| item | size |
|---|---:|
| one parquet shard at a time | ≤ 0.6 GB |
| student features, f16, 271K × 400 × 43 | 9.3 GB (4.7 GB at u8) |
| teacher states, u8 | 2.6 GB |
| logits and pooled embeddings | 0.2 GB |
| test features | 0.5 GB |
| eot-bench | 2.0 GB |
| **peak** | **about 15 GB** |

**Time.**

- Download of 41.4 GB: 15–70 min depending on bandwidth.
- Decoding and features in Rust.
- Teacher on 8 processes: 20–30 min.
- About 1–2 hours of wall clock in all. None of it touches the GPU.

## 5. Keywords

**Drop them from v2.**

- v1's control already passes the Speech Commands bar (0.9217 ≥ 0.90), so nothing is learned by re-running it.
- Published tiny keyword spotters are far better at a fraction of the size: BC-ResNet-1 96.9% at 9.2K parameters and BC-ResNet-8 98.7% at 321K, on GSC v2 with 12 classes (arXiv 2106.04140).
- Sharing one encoder with end of turn cost v1 capacity.
- Zero-shot keywords at this size are not credible: v1 reached 0.13–0.30 against 0.10 chance.

The honest scope of v2 is one task: "has the speaker finished?", after a VAD pause. If keywords are wanted, they belong in a separate tiny model with its own pre-registration. The v1 §7 items `duplex_action` and `about` are not part of v2.

## 6. Risks and odds

**Expected test accuracy at τ 0.5, as a range.**

- **Central estimate: 0.85.** 80% interval 0.80–0.89.
- **The base.** v1's control: 0.765 on test and 0.861 on train, with the wrong data release and a 45% task share.
- **Plausible gains, each an inference.** In rough order of confidence:
  1. the right release and its new sources: +1 to 3 points on the 17% unseen share;
  2. a single task and about 9× the turn-end training;
  3. a dedicated head and the attention pooling;
  4. logit and state distillation;
  5. prosody channels.
- **Where it stops.** Lexical cues in 23 languages, learned without ASR pretraining. The best published evidence for small from-scratch models is poor: 69% on real speech for SpeculativeETD's GRU.

**Probabilities** (judgement, from the numbers above):

| outcome | probability |
|---|---:|
| accuracy ≥ 0.80 (clearly above the control's 0.765) | 0.85 |
| accuracy ≥ 0.85 | 0.50 |
| accuracy ≥ 0.88 | 0.20 |
| accuracy ≥ 0.90 (within 3 points of the teacher) | 0.05–0.08 |
| false cut-off ≤ 0.20 at the validation threshold for 85% recall, given accuracy ≥ 0.85 | 0.85 |
| latency and size gate, given the pre-training probe | 0.90 |
| **whole gate (G1–G3 of the draft)** | **about 0.38** |

The teacher at the same operating point gives 0.044, the control 0.33, a timer 1.0.

**Against the two reference points.**

- **The silence timer: 47–50%.** A pass is +35 points. Even the pessimistic end (0.80) is +30.
- **Smart Turn: 92.6% overall, 94.3% English, 94% / 1% on Speko's real clips.**
  - Expected distance: 5–10 points of accuracy.
  - At the validation threshold for 85% recall: 10–20% false cut-offs against Smart Turn's 4%.
  - That is roughly 2–4× more interruptions, in exchange for a file 15–20× smaller and decisions 5–10× faster.
- **On real conversational speech** (eot-bench), every audio-only model including Smart Turn is weak: FC@600 22.9% against 33.4% for silence. A student should be expected to lose more there. It is reported, not gated.

**Main risks.**

1. The lexical component cannot be learned at this size. This is the likely failure mode, and the pilot's stop rule catches it early (pre-registration §5).
2. Overfitting with many epochs: about 50 epochs in 240 minutes at v1's throughput. Mitigated by teacher targets, light augmentation and EMA; the checkpoint is chosen on validation.
3. Latency above 5 ms in WASM with the larger configurations. The probe fixes the size before training.
4. Synthetic-speech bias of the data: 80% chirp3 TTS. Smart Turn falls to 62% on another TTS family (Speko), so the domain is narrower than it looks.
5. The harness: fixed and checked against Pipecat's 92.63% before v2 is read.

**An alternative with better odds and worse latency** (recorded; not recommended under the "few ms" constraint).

- **The model.** Initialise from Smart Turn's own weights (BSD-2; Whisper-tiny is MIT): keep the convolutional stem and 2 of the 4 layers at d 384, with a 4 s window. Then ternary QAT with distillation.
- **Size and speed.** About 4.1M parameters and about 2–3 MB, but about 0.8–1.6 GFLOP per decision: roughly 10–30 ms native.
- **Odds.** Accuracy ≥ 0.88: estimated 50–60%.
- **Code.** A Whisper-encoder graph (absolute sinusoidal positions, biased pre-LN layers) and an ONNX weight importer.

## 7. Not verified

- **The v3.2 training recipe.** `train.py` at `main` is configured for v3.3-alpha1 datasets.
- **The noise mix.** Its SNR and share are unpublished.
- **v3.2-train against v3-train.** Whether v3.2-train contains v3-train's clips, with or without noise. The HF filter API refused unauthenticated queries.
- **Overlap.** Sentence overlap between train and test (no text column). Overlap between Speko's 200 v3.1-test clips and the v3.2 test.
- **The source sample rate** of the parquet audio. Our decoder resamples linearly when it is not 16 kHz. The sample agreement (0.9243 against 0.9263) suggests no material effect.
- **Licences.**
  - turn-1-mini: Apache-2.0 per the LiveKit PR; none on its README.
  - eot-bench data: CC-BY-4.0 on the HF card, Apache-2.0 on the code repository.
- **Parameter counts.** Pipecat's backchannel detector: 734K or 961K. SpeculativeETD's: 202K, 1M or 11M.
- **Speeds.** The fp32 teacher's speed on the M5. The WASM latency of any v2 configuration (to be measured by the probe).
- **The estimates.** Every odds figure in §6 is a judgement from the measurements above, not a result.

## 8. Downloads a run needs

| name | URL | size | licence |
|---|---|---:|---|
| Smart Turn data v3.2, train | https://huggingface.co/datasets/pipecat-ai/smart-turn-data-v3.2-train | 41.4 GB (83 shards), streamed and deleted | CC-BY-4.0 |
| LiveKit eot-bench data, `all` | https://huggingface.co/datasets/livekit/eot-bench-data | 1.98 GB | CC-BY-4.0 (card) |
| LiveKit eot-bench code (for the cohort definitions) | https://github.com/livekit/eot-bench | < 10 MB | Apache-2.0 |
| Smart Turn v3.2 GPU ONNX (only if the fp32 teacher is chosen) | https://huggingface.co/pipecat-ai/smart-turn-v3 | 32.4 MB | BSD-2 |
| already present: Smart Turn v3.2 CPU ONNX; v3.2 test as WAV | – | 8.68 MB; 6.8 GB | BSD-2; CC-BY-4.0 |

## 9. Sources (read 15/11/2026 unless stated)

- Smart Turn code, `train.py`, `inference.py`, `audio_utils.py`: https://github.com/pipecat-ai/smart-turn
- Smart Turn weights and per-version benchmark reports (v3.2 reports generated 07/01/2026): https://huggingface.co/pipecat-ai/smart-turn-v3 and https://huggingface.co/pipecat-ai/smart-turn-v3/tree/main/benchmarks
- Smart Turn v3 announcement (11/09/2025): https://www.daily.co/blog/announcing-smart-turn-v3-with-cpu-inference-in-just-12ms/
- Smart Turn v3.2 (07/01/2026): https://www.daily.co/blog/smart-turn-v3-2-handling-noisy-environments-and-short-responses/
- Dataset cards and sizes (HF API; datasets modified 29/09/2026): https://huggingface.co/datasets/pipecat-ai/smart-turn-data-v3.2-train , https://huggingface.co/datasets/pipecat-ai/smart-turn-data-v3.2-test
- Pipecat Smart Turn overview (VAD `stop_secs` 0.2): https://docs.pipecat.ai/api-reference/server/utilities/turn-detection/smart-turn-overview
- Pipecat backchannel detector (modified 28/10/2026): https://huggingface.co/pipecat-ai/backchannel-detector
- Speko benchmark (07/08/2026): https://speko.ai/blog/voice-agent-turn-taking-benchmark
- LiveKit eot-bench: https://github.com/livekit/eot-bench ; data (modified 13/11/2026): https://huggingface.co/datasets/livekit/eot-bench-data
- Sharon et al., "Less can be More" (10/09/2026): https://arxiv.org/html/2609.11066
- SpeculativeETD (v2, 2026): https://arxiv.org/html/2503.23439v2
- p99lab turn-1-mini (Oct 2026): https://github.com/P99Lab/turn-1-mini
- OpenYap Turn (18/10/2026): https://huggingface.co/blog/TheAgenticDataCompany/openyap-turn
- Silero VAD: https://github.com/snakers4/silero-vad ; architecture and parameter count from ports: https://soniqo.audio/guides/vad (secondary)
- BC-ResNet (Interspeech 2021): https://arxiv.org/abs/2106.04140
- DistilHuBERT (ICASSP 2022): https://arxiv.org/abs/2110.01900
- Beyer et al., "Knowledge distillation: A good teacher is patient and consistent" (CVPR 2022, not re-read): https://arxiv.org/abs/2106.05237
- Repository files read: `docs/s6-ear-prereg.md`, `docs/SPEC-S6.md` §16, `BENCH.md` (s6-ear-nano), `docs/big-run-report.md` §2–4 and §7, `docs/research-s6-usefulness.md` §4 and §8, `docs/s6-cast2-prereg.md`, `crates/sky-infer/src/audio.rs`, `crates/sky-data/src/audio.rs`, `scripts/train-ear.sh`, `scripts/ear-baseline.py`, `configs/s6-ear-nano.json`, `runs/s6-ear-nano/`.
