# s6-ear: pre-registration (first training and gates)

*Translated into English on 19/11/2026. The text frozen on 03/11/2026 was written in French; that original, which remains the reference in case of doubt, is this file at commit `1f180a5`. The translation changes no rule, number or threshold.*

*Note added 19/11/2026: "TODO.md" (§6 and §1.1, cited in sections 4 and 7 below) refers to the project's former to-do file, `TODO.md`, removed on 19/11/2026; its last version is at commit `0e45b7b`. The text below is unchanged.*

**Status: frozen on 03/11/2026, before any audio data was downloaded in full and before any training of s6-ear.** This file fixes the data, the questions, the run, the control, the evaluation and the gates. The results will be read through this text.

**Changes to this file:**

- every amendment is committed, dated and justified in section 8;
- it precedes the run it concerns;
- it never touches a threshold after a test score has been read.

## 1. Data

All the sources are public, unrestricted and under CC-BY-4.0 (`DATA_LICENSES.md`, audio section). `sky-data fetch-audio --set all` (seed 1234) writes them into `data/ear`, which is not committed. Each clip is kept in the form of its log-mel (float16, `audio/<set>/<id>.npy`), cut to its last 128 frames of 80 ms. The test clips are also kept as 16 kHz WAV (`wav/<set>/<id>.wav`).

| set | source | splits |
|---|---|---|
| smart-turn | train and val: `pipecat-ai/smart-turn-data-v3-train` (227,025 clips, 23 languages including French); test: `pipecat-ai/smart-turn-data-v3.2-test` (31,527 clips), the official test set of Smart Turn v3.2 | val 5 % of train by a seeded hash of the clip id; any train clip whose id is also in the test is removed (disjoint by clip) |
| speech-commands | Speech Commands v0.02 (official TensorFlow archive) | train / val by `validation_list.txt`; the clips of `testing_list.txt` are set aside; test = the official 12-class test archive |
| mswc | `MLCommons/ml_spoken_words`, en and fr, WAV | train / dev / test of the set; the first archive of each split and language (`--mswc-archives 1`) |
| mimi | `openslr/librispeech_asr` (`clean`, `train.100` and `validation`) joined to `shangeth/librispeech-mimi-codes` (`train_clean_100`, `dev_clean`) by utterance id | pre-training only |

Smart Turn: the original plan drew the test from `smart-turn-data-v3-train` (10 % by seed). It was replaced before any complete data and any run, because this set is the one Smart Turn is trained on: the bar would have been measured on its own training data. The official test set of v3.2 has been seen by neither of the two models.

Speech Commands: the 25 words outside the 10 words of the benchmark give the class "unknown", kept with a probability of 1/6 in train and in val. The class "silence" is made of 1 s cuts, every 0.5 s, of the noises of `_background_noise_`, of which 10 % go to val.

Mimi: the codes are at 12.5 Hz, like our frames. An utterance in general has one more code frame than log-mel frames (rounding up against rounding down): the pair is cut to the shorter one. Only codebook 0 is used: it is the semantic codebook, distilled from WavLM.

## 2. Questions (the audio jev, `sky-data audio-jev`)

One record per question, the state being the path of the clip's `.npy`. The texts are fixed in `crates/sky-data/src/audio.rs`:

- **`turn_end`** (noul): "Has the speaker finished their turn?". The target is `endpoint_bool`. Options: "false: the speaker is still talking or will go on", "true: the speaker has finished their turn".
- **`keyword`** (choice, 12 options): "Which word was said?", over yes, no, up, down, left, right, on, off, stop, go, "silence: nothing was said" and "unknown: another word".
- **`keyword_zs`** (choice, 10 options): the same question on MSWC.
  - For each language, 50 words that have at least 20 test clips are held back, drawn by the seed.
  - A test question bears on a held-back word. Its 9 distractors are other held-back words. Each held-back word gives exactly 20 test questions (its first 20 clips), so the right word is uniform over the held-back words.
  - The train and val questions bear on the other words, with distractors taken among them.
  - No held-back word appears in training, neither as an answer nor as an option.

## 3. Model

`configs/s6-ear-nano.json`: the shape of s6-nano (d 144, ff 288, 2 blocks × 2 loops, exits 1 to 4, Parallax, 2 heads, tokenizer `tok-s6-2048`), with `state_kind: "audio"` and a state window of 128 frames (10.24 s). The parameter count is fixed by a test:

- decision model: 1,615,681 parameters (s6-nano plus `audio_proj`, 46,224);
- pre-training stack: 821,552.

Smart Turn v3.2 has 8M: s6-ear-nano is about 1/5 of that. Size is not a gate condition; it is fixed by the config.

## 4. The run (`scripts/train-ear.sh data pretrain decide eval`)

- **Pre-training** (`s2-pretrain --audio`): 60 min, 64 cuts of 128 frames per batch, 30 % of the frames masked in blocks of 3. The target is the Mimi id of codebook 0 of each masked frame. The report gives the accuracy of this codebook on the masked val frames.
- **Decision** (`s2-decide`):
  - 60 min on `turn_end=0.45,keyword=0.30,keyword_zs=0.25`, from the pre-training's init;
  - ternary QAT over the last 10 %, `--loop-dropout 0.25`;
  - then `s2-calibrate` on val and the ternary export.
- **Control** (pre-registered): the same model, the same data and the same decision budget, **without** Mimi pre-training (random init). It is reported next to each gate, without being a gate itself: it measures what the distillation of Mimi brings.
- **Thresholds**, on val, at full depth, all exits (`sky-eval --dump-exits`, `scripts/ear-gate.py crc`):
  - **early-exit thresholds** by conformal risk control, α = 0.02 (SPEC-S6 §9). They are written into a `-crc` copy of the GGUF;
  - **threshold of hand-off to System 2** ρ, α = 0.05: the error rate among the decisions that s6-ear takes itself. It is reported with the coverage.

This run is not launched at the same time as another training on the GPU (the LFM3 run of §6 of TODO.md).

## 5. Evaluation (test split, ternary model)

- `sky-eval` at full depth (τ = 1), then at the CRC thresholds, then the control. The test clips are read as WAV, through the PCM path of `Engine::decide_audio` (the deployed path).
- `scripts/ear-baseline.py`: Smart Turn v3.2 (`smart-turn-v3.2-cpu.onnx`, revision `f766f81d3cfdf7737ac64aad813d91bbfd56bf93`) on the same test clips of `turn_end` (the official test set of v3.2). We take Pipecat's reference inference: the last 8 seconds, the Whisper features, and `probability > 0.5`.
- WASM parity on 20 test clips (`parity-node.mjs`: probability gap ≤ 1e-4, same argmax).
- The native median latency per decision.

## 6. Gates (v1)

All are read on the ternary model at full depth. Their metric is **accuracy** (the share of right answers):

1. **End of turn**: accuracy on `ear:turn_end` ≥ that of Smart Turn v3.2 on the same clips.
2. **Keywords**: accuracy on `ear:keyword` (official test, 12 options) ≥ 0.90.
3. **Zero-shot keywords**: accuracy on `ear:keyword_zs:en` **and** on `ear:keyword_zs:fr` ≥ 0.20, that is 2 × chance at 10 options. The bar is deliberately low: it is a first attempt.

`scripts/ear-gate.py gate` computes them. If a gate fails, the model is not published, but the result still goes into the report. The precision and the recall of the end of turn are reported for both models.

The model is published at the CRC thresholds if the gates hold there as at full depth, at full depth if they hold only there, and is not published otherwise.

## 7. v2 (code and pre-registration now; data and runs later)

- **`duplex_action`** (choice): "What should I do now?", with the options listen, backchannel, speak, yield.
  - The labels come from AMI (CC-BY-4.0: one microphone per speaker and timestamps), by the rule `audio::duplex_label`, tested.
  - At time t, for agent A:
    - if A is speaking and another speaker starts within 500 ms, outside a "backchannel" act, while A is still speaking, it is **yield**;
    - if A starts a segment within 500 ms, it is **backchannel** if it is a backchannel act, otherwise **speak**;
    - otherwise, it is **listen**.
  - **Gate**: beat the rule "speak after 500 ms of silence" (`audio::silence_rule`) in accuracy on the test split of AMI.
  - The AMI downloader remains to be written: it is a v2 data item.
- **`about`** (noul and choice on the content, `audio::about_questions`): cross-modal distillation.
  - `sky-data teacher label-transcripts` asks GLM-5.3-flash for a probability per option on each LibriSpeech transcription of `data/ear/mimi`. The labeller is the one of §1.1, which resumes after an interruption and isolates the moderation refusals.
  - `audio-jev --about` turns them into soft targets on the audio: the teacher reads the text, the student hears the sound.
  - The labelling is launched **after** that of §1.1 of TODO.md, which uses the same z.ai quota.
  - **Gate**: accuracy ≥ 0.60 on the binary questions (val, against the teacher's argmax).

## 8. Amendments

**Amendment 2 (09/11/2026, before any training of s6-ear: only the pico recipe is running, none of its results has been read, no score of s6-ear exists). The recipe and the braked mode.** There was never an amendment 1: the number 2 is the one that `scripts/train-ear.sh` and `scripts/s6-adopt.py` already cite.

- **The recipe.** s6-ear runs after the pico ablation (`docs/s6-prereg.md`, §3 and §4) and takes its options (`scripts/s6-adopt.py env s6-ear-nano`), for the model **and** for its control: `CONFIG`, `OPT`, `DECIDE`, `LR_SCALE`.
- **The rate.** The rate of the pico pilot is moved to d = 144 by μP, as for s6-nano: `LR_SCALE` × √(96/144) × the default rate of the phase.
- **The curriculum.** It orders text documents, and the audio pre-training has none: it has no effect here, so `PRETRAIN` stays empty.
- **The check after pre-training.** `val_acc_cb0` must be at least 10/2048 (10 times chance), a threshold written before the run. Below it, it is a technical failure: it is noted in the report and `decide` is not launched.
- **The braked mode.** The run is done in braked mode (`SKY_DUTY=0.5`, `SKY_COOL=45:15`), like amendment 5 of `docs/s6-prereg.md`. The budgets in minutes count only work time, and the report gives the number of steps actually done.
- **What does not change.** Data, questions, model, budgets, α thresholds, gates.

**Amendment 3 (12/11/2026, before any training of s6-ear: the recipe is written (`runs/recipe/recipe.json`), no score of s6-ear exists). Two modes.** At the owner's request (11/11/2026), the run uses `SKY_DUTY=0.85`, `SKY_COOL=50:10` while the Mac is not in use, and the braked mode of the previous amendment (`0.5`, `45:15`) the rest of the time, as in amendment 6 of `docs/s6-prereg.md`. Budgets in minutes still count work time only. A change of mode stops the training, which resumes from its last checkpoint (`RESUME=1`). The report gives the steps actually done and the modes used. Nothing else changes.

**Amendment 4 (12/11/2026, before any evaluation of s6-ear: no gate score has been read). A memory defect, and the decision runs started over.**

- **The defect.** The decision loader kept the log-mel frames of every clip in memory, about 22 GB on a 16 GB machine (27 GB while training, 25 GB of swap). The 60 minutes of the decision run of 12/11/2026 counted time spent swapping: it did 31,480 steps, about 525 a minute, where the same command does about 975 a minute once fixed (measured on 90 s in a throwaway folder, memory only). The control run had done 11 of its 60 minutes when the run was stopped by hand.
- **The fix.** The frames stay on disk and are read when a batch needs them (`Stream::audio_bits`); a test checks that the batch is the same, bit for bit. Memory is 1.7 GB after loading and 4 to 5 GB while training. `scripts/train-all.sh` also caps MLX's buffer cache at 2 GB.
- **What is started over.** The decision run and its control, from zero, with the same seeds, commands and budgets, so that both run in the same conditions. This is a technical failure in the sense of amendment 5 of `docs/s6-prereg.md`. The decision was taken before any evaluation: only training lines and the calibration line of the discarded run were seen. Its files are kept in `runs/s6-ear-nano-swap/`, and the report gives its step count next to the new one.
- **What is kept.** The pretraining of 12/11/2026 (48,626 steps): it does not use this loader and ran at its normal speed.
- **The brake.** The mode now changes with no restart: `sky-train` reads `SKY_BRAKE_FILE` again every few seconds, and `scripts/auto-brake.sh` writes it. The Mac counts as not in use after 10 minutes with no key or pointer input, unless a video or a call keeps the display awake, it runs on battery, or macOS reports thermal pressure. The script stops a training that holds more than 13 GB.
- **What does not change.** Data, questions, model, budgets, α thresholds, gates.

**Amendment 5 (15/11/2026, after the gate was read: it failed). The file of the failed model is published, as a result.** On 15/11/2026 the owner of the project decided that the files of the models that failed their gate are published too, so that every result can be checked and run by anyone. Nothing of the reading changes: the gate failed, the model is recommended for no use, and wherever its file is offered (model card, site, manifest) the first lines say that it failed its pre-registered gate and give the numbers. The sentences of this pre-registration that tied publication to the gate said what a pass would mean; for the file they are replaced by this decision, for the verdict they stand. The published file is `s6-ear-nano` as evaluated; its control model is not published.
