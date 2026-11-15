# s6-ear v2 (s6-ear2-nano): pre-registration

**Status: frozen on 15/11/2026, before any line of v2's code is written and before any v2 download.**

- No line of v2's code is written, no v2 data is downloaded and no v2 model exists.
- When frozen, this file fixes:
  - the data, the features, the model and the teacher;
  - the run, the choice of checkpoint, the evaluation and the gate.
- The results are then read through it.
- Two things are fixed later, by rules written here, before training starts:
  - the model size, by the latency probe of §4.4;
  - the harness, by the check of §8.1.

**Why this experiment.**

- **v1 failed its gate.** s6-ear-nano scored 0.5681 on end of turn; its control, trained without the Mimi pretraining, scored 0.7647 (`docs/s6-ear-prereg.md`, `runs/s6-ear-nano/eval/gate.json`).
- **v1's bar was itself measured wrong: 0.8518.** `scripts/ear-baseline.py` pads short clips at the end. Smart Turn pads them at the start. Pipecat's own report for Smart Turn v3.2 on the same 31,527 clips is 0.9263.
- **The owner's request.** An end-of-turn model that works, not a perfect one, without spending GPU hours on poor odds.
- **The design study.** `docs/s6-ear2-study.md` is committed with this file and holds every number quoted here.

**Changes to this file:**

- every amendment is committed, dated and justified in §14;
- an amendment comes before the run or the measurement it concerns;
- an amendment never moves a threshold after a v2 test score has been read;
- these measurements are allowed before the freeze, and are written down with their date:
  - on training and validation clips;
  - of throughput and latency on random weights;
  - of Smart Turn on the test set, for the harness check only.

## 1. The question

Can a ternary model of 1.3–1.8M parameters tell whether a speaker has finished their turn? It is given:

- the right data release;
- features made for the task;
- one task instead of three;
- distillation of Smart Turn v3.2's logit, pooled embedding and encoder states.

**Well enough to be used** means all of these, on Smart Turn's official test set:

- it is far above a silence timer, v1 and v1's control;
- it is within a stated distance of Smart Turn;
- it decides in a few milliseconds in WASM, from a file of at most 3 MB.

## 2. What changes from v1, and why

| # | change | v1 evidence (study §1) | external evidence (study §2) |
|---|---|---|---|
| C1 | **No codec pretraining.** Random init, decision training only. | The Mimi-pretrained model underfits its training clips: 0.702, against 0.861 for the same model without pretraining. Its validation accuracy stays below 0.59 for half the run. | No working end-of-turn system uses codec targets or tokens. |
| C2 | **Data: `smart-turn-data-v3.2-train`** (270,946 clips) instead of v3-train (227,025). | 17.4% of the test (liva_1, midcentury_1, mundo_1, chirp3_3_short) comes from sources absent from v3-train. The control scores 0.526–0.726 on three of them, where the teacher scores 0.934–1.000. | Smart Turn v3.2 is trained on this release, with café and office noise mixed in. |
| C3 | **One task.** Keywords and zero-shot keywords are dropped (§3.5). | End of turn had 45% of v1's 60 minutes. | Published tiny keyword spotters reach 96.9% at 9.2K parameters (BC-ResNet). |
| C4 | **Features: 40 log-mel bands + 3 prosody channels** (log f0, voicing, energy) per 10 ms hop, normalised per window by Whisper's rule. **Window: the last 4 s**, at 40 ms per position. | v1 had no pitch and no normalisation, and 10.24 s at 80 ms per position. Measured on the test sample, the teacher keeps 0.919 of its 0.924 with only the last 4 s. | Amazon's ablation: the prosody gain is carried by Δf0. Smart Turn normalises its input per clip. |
| C5 | **A dedicated graph** (`turn` family): `patch_proj` → 4 s2 encoder blocks (no loops, no Parallax, no text stream) → Smart Turn's attention pooling → a 2-layer head. | v1's graph carried a BPE question stream and shared its capacity across three tasks. | Smart Turn's head: attention pooling and an MLP. |
| C6 | **Teacher distillation** from Smart Turn v3.2 (BSD-2): logit, 384-d pooled embedding, final encoder states pooled to 25 steps, on the training clips only. | On the sources v1 trained on, the control trails the teacher by 15 points (chirp3_1: 0.785 against 0.941), and its AUC is 0.844 against 0.978. | DistilHuBERT keeps most of HuBERT at a quarter of its size by layer-wise state distillation. |
| C7 | **240 working minutes** of training on one task, instead of 60 shared. | The control's validation plateaued between 0.82 and 0.86, on a shared budget. | Distilled students gain from long, consistent schedules (Beyer et al.). |
| C8 | **The harness fixed** (left padding, Pipecat's reference) and checked against Pipecat's published 92.63% before v2 is read (§8.1). | The v1 harness reads 0.8437 where the reference reads 0.9243, on the same 3,000 clips. | Smart Turn's `audio_utils.truncate_audio_to_last_n_seconds`. |

**Not changed, and why.**

- **The log-mel front end.** It is tested, it is identical in WASM and it is cheap; Smart Turn's 80 bins are not needed by a student.
- **Ternary weights under QAT.** They are the point of the project. v1's control lost nothing measurable to them: validation 0.855 ternary against 0.863 f32 before QAT.
- **The brake modes and the working-minute budgets** (amendments 2–4 of `docs/s6-ear-prereg.md`).

## 3. Data and leakage

### 3.1 Sources

| set | use | rows | licence |
|---|---|---|---|
| `pipecat-ai/smart-turn-data-v3.2-train` | train, and val by a 5% hash of the id (seed 1234) | 270,946 | CC-BY-4.0 |
| `pipecat-ai/smart-turn-data-v3.2-test` | **test, gated**; the WAVs already written by v1's fetch | 31,527 | CC-BY-4.0 |
| `livekit/eot-bench-data`, config `all`, revision `main` | second test, real human speech, reported (§8.4) | 7,342 turns | CC-BY-4.0 (card) |

The training set is streamed one parquet shard at a time; each shard is deleted once its features and teacher targets are written. Every licence is recorded in `DATA_LICENSES.md`.

### 3.2 Leakage rules

- **No test clip and no eot-bench audio** enters training, validation, the pilot, the teacher pass or any threshold choice.
- **A training clip is dropped** if either:
  - its id is in the test set (v1's rule);
  - the hash of its decoded 16 kHz int16 PCM equals that of a test clip (new rule).
- The counts of both drops are recorded before training.
- **Sentence-level overlap cannot be checked:** `spoken_text` is empty. The report says so, as Smart Turn's own numbers share this limit.
- **The teacher is run on train and val clips only.** Its test numbers come from the harness of §8.1. They are a baseline, not a target.

### 3.3 Mix and augmentation

- **Sampling.** Every training clip with the same probability. No reweighting by language or source, as Smart Turn.
- **Light augmentation**, on the student's view only; the teacher targets are those of the clean clip.
  - **Context jitter.** With probability 0.3, the window's start moves so that 1–4 s are kept. The window always ends at the clip's end.
  - **Masking.** Two frequency masks of at most 4 bands. One time mask of at most 8 hops, placed only in the first 3 s of the window.
- **Not added:** gain augmentation (the per-window normalisation removes gain) and new noise (the data already carries it).

### 3.4 Disk

- **Peak about 15 GB:**
  - one shard, at most 0.6 GB;
  - features in f16, 9.3 GB;
  - teacher states in u8, 2.6 GB;
  - test features, 0.5 GB;
  - eot-bench, 2.0 GB.
- **Free disk at the time of writing: about 25 GB.** v1's `data/ear` (31 GB) stays untouched unless the owner decides otherwise.

### 3.5 Scope

v2 answers one question: "has the speaker finished?", asked once after a VAD pause. These items stay out of v2:

- keywords;
- zero-shot keywords;
- v1's §7 items `duplex_action` and `about`.

A keyword model, if wanted, gets its own pre-registration.

## 4. The model (`configs/s6-ear2-nano.json`)

### 4.1 Features (`sky_infer::audio`; the same code natively, in WASM and in `sky-data`)

- **Log-mel, kept from v1.** 40 bands, 25 ms Hann window, 10 ms hop, 512-point FFT, the deterministic log. Values are stored before normalisation.
- **Prosody, new**, per 10 ms hop:
  - **Analysis.** A 40 ms Hann window, a 1,024-point FFT, then the inverse FFT of the power spectrum. That is the autocorrelation, normalised by the window's own.
  - **Lag search.** 40–200 samples, that is 80–400 Hz.
  - **The three channels.**
    - voicing = the peak normalised autocorrelation;
    - f0 channel = log2(f0 / 100 Hz) if voicing ≥ 0.45, else 0, with parabolic interpolation of the peak;
    - energy = log10 of the frame energy on the log-mel scale.
- **Normalisation** (decision time, per window), on the mel channels:
  - x ← max(x, M − 8) with M the window's maximum log10 mel value;
  - then y = (x − M + 4) / 4.
  - The energy channel takes the same shift; f0 and voicing are left as they are.
- **Window and positions.**
  - The last 400 hops (4.0 s), 4 hops per position: 172 values every 40 ms, 100 positions.
  - Shorter clips give fewer positions, as in v1: no padding, attention masked by length.
- **Streaming.** `FeatureStream` accepts PCM in any chunk size and computes each hop once. It keeps the last 400 hops. Streaming and batch features must agree bit for bit (a test).

### 4.2 Graph

- `patch_proj` `[d, 172]` + bias (float).
- 4 s2 encoder blocks (LayerNorm, `attn_qkv`, `attn_output`, GELU FFN, RoPE θ 10,000), ternary under QAT, with a final norm.
- **Attention pooling** (float): `Linear(d, 64) → tanh → Linear(64, 1)`, a softmax over the unmasked positions, then a weighted sum.
- **Head** (float): `Linear(d, 64) → GELU → Linear(64, 1)`. A temperature fitted on val is stored in the file.
- **Training only, not exported:**
  - `pool_proj` `[384, d]` on the pooled vector;
  - `seq_proj` `[384, d]` on the positions, average-pooled by 4 into 25 steps, aligned on the window's end.

### 4.3 Size (the rung chosen by §4.4)

| rung | d / ff | blocks | position | parameters (exported, about) |
|---|---|---:|---|---:|
| R1 | 192 / 768 | 6 | 40 ms | 2.71M |
| R2 | 192 / 768 | 4 | 40 ms | 1.83M |
| R3 | 160 / 640 | 4 | 40 ms | 1.28M |
| R4 | 160 / 640 | 4 | 80 ms (8 hops, 344 values, 50 positions) | 1.30M |

- The exact count of the chosen rung is fixed by a test in `sky-train`.
- The GGUF size is measured and must be ≤ 3 MB (G3). Expected: 0.4–0.8 MB.
- New GGUF keys: `skycmd-turn.*` for the window, hops per position, channel count, threshold τ₀.₅, threshold τ₈₅ and temperature.

### 4.4 The latency probe (before training; it fixes the rung)

- **What is measured.** Each rung is exported with seeded random ternary weights. It is timed on the M5:
  - in WASM: Node, SIMD128, one thread;
  - over 1,000 decisions on a full 4 s window;
  - with the features already computed.
- **The rule.** The first rung of R1–R4 whose median is ≤ 4.0 ms is taken. The margin is 1 ms below G3's bound.
- **If none qualifies,** an amendment says so before training, and v2 is not run.
- **Recorded:** the result, the native numbers and the front end's cost per hop.

### 4.5 Code (before any training)

**sky-infer:**

- `src/audio.rs`:
  - `prosody` (the three channels);
  - `normalize_window`;
  - `FeatureStream`;
  - tests: f0 of a pulse train at 100, 200 and 300 Hz within 1%; voicing of white noise below 0.3; streaming against batch bit for bit; the WASM parity of the new channels in `crates/sky-web/tests/parity.rs`.
- `src/turn/mod.rs` (config and GGUF keys) and `src/turn/model.rs` (the forward pass, reusing the s2 blocks and ternary kernels).
- `src/wasm.rs`: `TurnDetector` with `new(bytes)`, `push(Float32Array)`, `probEnd()` and `probEndPcm(Float32Array)`.
- `src/bin/sky-turn.rs`: per-item probabilities over records, and `--bench` for latency.

**sky-train:**

- `src/turn/{mod,config,data,model,train,gguf,golden}.rs`, modelled on `src/cast/`:
  - a from-disk loader of f16 features and u8 teacher states;
  - the losses of §5.2;
  - QAT, EMA, export;
  - a parameter-count test per rung.
- `src/main.rs`: `turn-train`, `turn-export` and `turn-probe` (a random-weight GGUF per rung).

**sky-data:**

- `src/turn.rs`, called from `src/main.rs`:
  - `fetch-turn`: shards, the leakage filter of §3.2, features and a manifest. A shard is kept until the teacher has written its targets.
  - `turn-test`: features from v1's test WAVs.
  - `turn-eotbench`: the cuts of §8.4.

**sky-web:** `bench/turn-bench.mjs` (latency) and a turn mode in `bench/parity-node.mjs`.

**Scripts:**

- `scripts/turn-teacher.py`:
  - uv, onnxruntime;
  - the pinned ONNX with its graph outputs extended;
  - Pipecat's preprocessing: HF `datasets` decoding, the last 8 s left-padded, WhisperFeatureExtractor;
  - per shard, it writes the logit, the pooled vector and the 25 × 384 states (u8, a scale per vector).
- `scripts/ear-baseline.py`: the left padding (the fix), a per-item dump, per-language and per-dataset tables.
- `scripts/turn-gate.py`:
  - val thresholds;
  - metrics of §8.2 with 1,000-sample bootstrap intervals by clip;
  - the gate JSON.
- `scripts/train-turn.sh`: probe, data, teacher, smoke, pilot (with the stop rule), train, export, calibrate, eval, wasm, gate. `DRY_RUN`, `RESUME` and the brake as `scripts/train-ear.sh`.

**Golden:** `tests/golden_turn.rs` and its golden files, femto configuration, the tolerances of the cast golden.

**Before v2 and in its own commit:** the correction of the v1 bar (0.8518 → 0.9263) in SPEC-S6 §16, BENCH, the big-run report and the site. v1's verdict does not change.

About 3,000 lines in all.

## 5. The teacher

### 5.1 File and outputs

- **File.** `smart-turn-v3.2-cpu.onnx`, revision `f766f81d3cfdf7737ac64aad813d91bbfd56bf93` (int8, 8.68 MB, BSD-2): the file of the baseline.
- **Three outputs are appended to the graph.** Verified in the study:
  - `linear_28_DequantizeLinear_Output` (the logit z_T);
  - `sum_1` (the pooled embedding e_T, 384);
  - `layer_norm_8_DequantizeLinear_Output` (the states, 400 × 384).
- **The states.** The last 200 frames are kept and average-pooled by 8, giving S_T (25 × 384).
- **Cost.** About 36 ms per clip on one core; 8 processes in parallel; CPU only.

### 5.2 Loss (per clip; y ∈ {0, 1}, z the student's logit, p = σ(z))

L = 0.5 · BCE(y, p) + 0.5 · BCE(σ(z_T), p) + 0.25 · (1 − cos(pool_proj(h̄), e_T)) + 0.5 · mean over the 25 steps of (1 − cos(seq_proj(h_t), S_T,t))

Notes:

- Every cosine is computed after a LayerNorm of the target.
- For clips shorter than 4 s, only the steps inside the clip count.
- No `pos_weight`: the classes are balanced (49.6% finished in v1's train split).

## 6. The run (`scripts/train-turn.sh`)

1. **Latency probe** (§4.4). CPU only, about 10 minutes.
2. **Data and teacher.** CPU only, 1–2 hours of wall clock.
3. **Smoke.** 5 working minutes on 2,000 clips, to test the chain. Its numbers are not looked at.
4. **Pilot.** 4 arms × 12 working minutes, rung fixed:
   - the rate at 0.5×, 1× and 2× the default of the recipe (`scripts/s6-adopt.py env`, moved to the rung's d by μP as in v1), each with full distillation;
   - then hard labels only, at the best rate.
   - The arm is chosen by validation accuracy at τ 0.5, on f32 weights.
   - **Rule:** distil unless the hard-label arm is higher by more than 0.005.
5. **Stop rule.** If the best pilot arm's validation accuracy is below 0.82, v2 stops there:
   - no main run;
   - it is recorded as failed by rule, with the pilot numbers.
   - For reference, v1's control reached 0.80 on validation in about 11 minutes of a shared budget.
6. **Training.**
   - 240 working minutes, batch 128, seed 1, once;
   - the recipe's optimiser at the chosen rate, WSD schedule;
   - ternary QAT over the last 15%, EMA 0.999;
   - the brake modes and memory cap of v1's amendments 2–4;
   - minutes count working time only; the report gives the steps done and the modes.
7. **Restarts.** Only a technical failure justifies a relaunch from the last checkpoint, and it is reported.

**Total GPU: about 293 working minutes** (5 + 48 + 240), under the 8-hour ceiling, about 7–9 hours of wall clock in the brake modes. Evaluation and calibration run on the CPU.

## 7. Checkpoint, calibration, export

- **Checkpoint.** The EMA if its ternary validation BCE against the labels is below that of `last`, else `last`. No intermediate checkpoint.
- **Temperature.** Fitted on validation (NLL) and stored.
- **Thresholds**, both from validation, both stored in the file, neither ever from test:
  - τ₀.₅ = 0.5 on the calibrated probability;
  - τ₈₅ = the highest threshold whose validation END recall is ≥ 0.85.
- **Export.** Ternary GGUF; WASM parity on 20 test clips: |Δp| ≤ 1e-4 and the same decision.

## 8. Evaluation

### 8.1 Harness check (before any v2 test score is read)

- **What is run.** The fixed `scripts/ear-baseline.py`, with Smart Turn v3.2 CPU, on all 31,527 test clips.
- **The bar.** It must give 0.9263 ± 0.003 (Pipecat's report, 07/01/2026).
- **The check on the timer.** The always-finished rule must give the test's share of finished clips: 0.4979.
- **If the check fails,** the decoding is investigated and the result is fixed by an amendment before any v2 test score is read. The bar of G1 is never moved to fit.

### 8.2 Test (gated): `smart-turn-data-v3.2-test`, 31,527 clips, ternary, full model

**Metrics:**

- **Overall.** Accuracy at τ₀.₅; precision, recall and F1.
- **Pipecat's error rates.** FPR and FNR as shares of all clips, Pipecat's convention, for its table.
- **At τ₈₅.** False cut-off (the share of not-finished clips answered "finished") and the END recall reached on test.
- **Ranking and calibration.** AUC and ECE.
- **Breakdowns.** Per-language and per-dataset tables, as Pipecat's report.
- **Subsets.** English, and the English human subsets (human_5, human_convcollector_1).
- **Comparison with Speko.** Speko's oracle point (the least false cut-off with recall ≥ 0.85, chosen on the evaluated set), reported only, for the 94% / 1% comparison.
- **Uncertainty.** 95% bootstrap intervals throughout.

**In the same table, on the same clips:**

| baseline | value |
|---|---|
| silence timer (always "finished") | 0.4979 |
| s6-ear-nano | 0.5681 |
| its control | 0.7647 |
| Smart Turn v3.2 CPU, as checked | 0.9263 expected |
| Smart Turn v3.2 GPU | Pipecat's published 0.9371 |
| v1's wrong harness number | 0.8518, marked as such |

### 8.3 Latency and size (gated by G3; measured before v2's test is read)

- **Decision time.** On the M5, one thread, 1,000 decisions, 4 s window, features already computed. Median and p95, natively and in WASM (Node, SIMD128).
- **Front end.** Cost per 10 ms hop in WASM.
- **References, in the same session.** Smart Turn v3.2 CPU through onnxruntime, natively (measured at 31.9 ms in the study), and v1's file.
- **Size.** The GGUF size.

### 8.4 eot-bench (reported, not gated; harness written before training)

- **The cuts.**
  - Every labelled silence span of every turn (`eot` or `hold`) gives one decision.
  - The audio is cut at the span's start, matching Smart Turn's trimmed clips.
  - The decision reads the last 4 s for v2 and the last 8 s, left-padded, for Smart Turn.
- **Metrics.**
  - AUC of `eot` against `hold`;
  - false cut-off and `eot` recall at the stored τ₈₅;
  - per language and per cohort (`general`, `accented`, `short_turns`).
- **Models.** v2, Smart Turn v3.2 CPU, and the timer.
- **Comparability.** These numbers are not comparable with LiveKit's FC@300 / FC@600, which come from a policy sweep; the report says so.
- **Reading, fixed now.** If v2's AUC is more than 0.10 below Smart Turn's on these cuts, the report says that v2 does not carry over to real conversational speech, whatever the gate says.

## 9. Gate (ternary GGUF, the file as exported)

v2 passes if all of these hold:

1. **G1 (accuracy).** On the 31,527 test clips, accuracy at τ₀.₅ **≥ 0.85, and ≥ T − 0.08**, where T is Smart Turn v3.2 CPU's accuracy from §8.1. With T = 0.9263, the binding bar is 0.85:
   - 35 points above the silence timer;
   - 28 above v1;
   - 8.5 above v1's control;
   - at most 7.6 below the teacher.
2. **G2 (false cut-offs).** At the validation threshold τ₈₅, the test false cut-off rate is **≤ 0.20** and the test END recall is **≥ 0.80**.
   - For reference, on the study's sample at the oracle point: the teacher 0.044, v1's control 0.334, a timer 1.0.
3. **G3 (deployable).** All of these:
   - WASM median decision time **≤ 5.0 ms** (§8.3);
   - front end ≤ 0.10 ms per 10 ms hop in WASM;
   - GGUF **≤ 3 MB**;
   - WASM parity (§7).

**Not gated:**

- beating Smart Turn;
- eot-bench (§8.4);
- per-language bars.

**Size and speed against Smart Turn's file.** The report says in one sentence which model a user should take: v2, or Smart Turn's 8.7 MB file at about 32 ms. It gives the accuracy given up against the size and time saved.

## 10. Reported, not gated

- All of §8.2–8.4, with intervals.
- **Pilot.** The pilot arms, the distillation decision and the stop-rule reading.
- **Training.**
  - steps done, modes and minutes;
  - validation curves;
  - the counts of the leakage filter;
  - teacher-student agreement on validation and test.
- **Exits.** None: v2 has no early exits.
- **Probe.** The rung chosen and the probe table.
- **Deployment.** The operating notes for the browser demo: the VAD used, `stop_secs`, the fallback timer.

## 11. Disclosure

**What v2's design read.**

- v2 was designed after v1's test results.
- It also read measurements made for its design study on 3,000 clips of the same test set:
  - v1's two models per source;
  - Smart Turn per source;
  - Smart Turn with 8, 4, 3, 2 and 1 s of context;
  - the trailing silence of 1,500 clips.

**What that shaped.**

- The 4 s window was chosen from Smart Turn's accuracy on those test clips.
- The data change was motivated by per-source test results.

**What it means.** The official test set is therefore partly in-sample for v2's design choices, though never for its weights: no test clip is trained on, and no threshold is chosen on test. eot-bench, which the design never read, is the cleaner check, and it is reported (§8.4). Every report of v2 states this paragraph.

## 12. What a failure would mean

- **The stop rule fires, or G1 is below 0.80.** A from-scratch ternary student of 1.3–2.7M parameters cannot learn this task, even with the teacher's states.
  - The line stops at this size.
  - The project's recommendation for end of turn becomes Smart Turn itself (BSD-2, 8.7 MB, onnxruntime), credited.
  - A follow-up would be the alternative of the study, §6: initialise from Smart Turn, about 4M parameters, 10–30 ms. It would need a new pre-registration with a relaxed latency bound, and only at the owner's request.
- **G1 between 0.80 and 0.85.** It works partly and does not pass. It is published as failed, with its numbers and the sentence of §9.
- **G1 passes, G2 fails.** It cuts users off too often at a useful recall. It does not pass.
- **G1 and G2 pass, G3 fails.** The model works but not at this cost. It does not pass. No smaller rung is retrained in v2.
- **v2 passes.** It is published as passed, with the sentence of §9.
- **eot-bench.** If v2 passes but is weak there (§8.4), the model card's first lines say it is validated on Smart Turn-style clips, mostly synthetic, and not on real conversations.
- **In every case,** following the owner's decision of 15/11/2026 (amendment 5 of `docs/s6-ear-prereg.md`), the file is published as evaluated. Its first lines give the verdict and the numbers.

## 13. Forbidden

- Training, validating, piloting, distilling or choosing a threshold on any test clip or any eot-bench audio.
- A second seed, a second run, or another rung after §4.4, except after a technical failure.
- Reading a v2 test score before §8.1's check and §8.3's latencies are on file.
- Changing a bar, a gate term or a threshold after a v2 test score is read.

## 14. Amendments

None yet.

## 15. Choices (the first value of each line is the one taken at the freeze; the alternative is recorded)

| | choice | recommended | alternatives | why the recommendation |
|---|---|---|---|---|
| O1 | architecture | **from-scratch ternary student, rung by §4.4** | initialise from Smart Turn's weights (about 4M, 2–3 MB, 10–30 ms); a causal streaming encoder with a KV cache | the only option that can meet "a few ms in WASM"; the Smart Turn initialisation has better odds (study §6) but breaks the latency bound; streaming adds engine code with no accuracy gain on clip-level labels |
| O2 | window | **4 s** | 8 s; 3 s | the teacher keeps 0.919 of 0.924 at 4 s and drops to 0.903 at 3 s; 8 s doubles compute and storage |
| O3 | position | **40 ms (unless §4.4 forces 80 ms)** | 80 ms; 20 ms | 100 positions keep the end of the turn at fine grain; 20 ms doubles attention cost |
| O4 | prosody channels | **in** | out | about 2 FFTs per hop; the Amazon ablation ties the prosody gain to Δf0; harmless if useless; not piloted, to save minutes |
| O5 | normalisation | **Whisper's per-window rule** | absolute features plus gain augmentation | microphones differ in gain; this is the teacher's own rule |
| O6 | teacher file | **CPU int8 (cached, the baseline's file)** | GPU fp32 (+1.1 points, 32.4 MB, about 2–3× the CPU time, not measured) | its outputs are verified; the gate's T is measured on it |
| O7 | teacher targets | **logit + pooled + 25-step states** | logit + pooled; logit only | the 15-point gap on seen sources points at missing knowledge, not missing labels; states cost 2.6 GB |
| O8 | loss weights | **0.5 / 0.5 / 0.25 / 0.5** | hard only (the pilot's hard arm); 0.3 / 0.7 / 0.25 / 0.5 | balanced hard and soft targets; the states weigh as much as the logit |
| O9 | main budget | **240 working minutes** | 180; 360 | 9× v1's per-task time and about 50 epochs at v1's throughput; 360 would cross 8 hours with the pilot and smoke |
| O10 | stop rule | **best pilot validation accuracy ≥ 0.82** | 0.80; none | spends 4 hours only if 12 minutes already beat v1's control; validation is optimistic, so a lower bar saves nothing |
| O11 | G1 bar | **0.85 and ≥ T − 0.08** | 0.88; 0.80 | "works" means near the teacher and far above v1; 0.88 has about 20% odds, 0.80 would pass a model few would use |
| O12 | G2 | **false cut-off ≤ 0.20 at τ₈₅ (validation)** | ≤ 0.10; not gated | a voice agent's main failure is cutting users off; at 0.20 it is 5× better than a timer and 4–5× worse than Smart Turn, stated as such |
| O13 | latency bound | **5 ms WASM** | 10 ms; 2 ms | the owner's "a few ms"; v1 ran at 6.7 ms in WASM |
| O14 | eot-bench | **reported, with the reading of §8.4** | gated: AUC ≥ Smart Turn's − 0.10 | its harness is new and unvalidated against LiveKit's own numbers; the teacher itself is weak there (FC@600 22.9% against 33.4% for silence) |
| O15 | keywords | **dropped** | a second head; a separate tiny model with its own pre-registration | v1's control already passes 0.90; sharing capacity is what v2 removes |
| O16 | augmentation | **light (§3.3)** | none | about 50 epochs need some regularisation; time masks stay away from the turn's end |
