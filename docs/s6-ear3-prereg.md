# s6-ear v3 (s6-ear3-nano): pre-registration

**DRAFT of 17/11/2026: not frozen; it becomes the pre-registration when the owner has accepted the corpus's access conditions (or chosen the fallback) and the lead freezes it; no computation before that.**

- No line of v3's code is written, no v3 data is downloaded and no v3 model exists.
- When frozen, this file fixes:
  - the data, the label rule, the clip construction, the model and its start;
  - the run, the choice of checkpoint, the evaluation and the gate.
- The results are then read through it.
- Four things are fixed later, by rules written here, before training starts:
  - the corpus (otoSpeech, or the AMI fallback of §3.10), by the owner;
  - the label check and the bleed probe of §3.7;
  - the harness checks of §8.1;
  - the start (from v2 or from scratch) and the rate, by the pilot of §6.

**Why this experiment.**

- **v2 passed its gate on Smart Turn's test clips** (0.8993; teacher 0.9260), and is weak on real conversations.
  - On eot-bench (`livekit/eot-bench-data`, real human-to-agent speech, never trained on, read once), its AUC is 0.7299 against Smart Turn's 0.7838.
  - At τ₈₅ it catches 0.5253 of the real ends of turn for a false cut-off of 0.2230 (`runs/s6-ear2-nano/GATE.md`).
  - Its teacher is weak there too, and 82.4% of its training clips are marked synthetic.
- **v2 never trained on the silences it is asked about.**
  - The demo asks after 0.2 s of silence, then every 0.2 s up to 2.4 s (`site/.vitepress/theme/lib/turn.ts`).
  - Its training clips end at the end of speech: median trailing silence 0 ms, and only 7.2% (not finished) and 8.9% (finished) have 200 ms or more (`docs/s6-ear3-study.md` §2).
- **The request.** `TODO.md`, "What remains", item 5: train with trailing silences of variable length, as the microphone delivers them, and read it on real conversations.
- **The design study.** `docs/s6-ear3-study.md` is committed with this file and holds every fact and number quoted here, with its source and the date read.

**Changes to this file:**

- every amendment is committed, dated and justified in §14;
- an amendment comes before the run or the measurement it concerns;
- an amendment never moves a threshold after a v3 test score has been read;
- these measurements are allowed before the freeze, and are written down with their date:
  - on training and validation data, the conversation validation sessions included;
  - of throughput on random weights;
  - of v2's file and Smart Turn on the d = 0 cuts of eot-bench, for the harness check of §8.1 only (their numbers are already published).

## 1. The question

Does the s6-ear2-nano graph, unchanged, do measurably better on real conversations when two things change in its training?

- **It is trained on the silences it is asked about.** These are the 0.2–2.4 s of trailing silence that the deployed policy produces.
- **It learns from real two-speaker conversations.** The labels come from the other speaker's channel, and cost nothing.

**Measurably better** means all of these:

- on eot-bench, at the policy's first question (0.2 s into each pause), its AUC and its end-of-turn recall at a false cut-off of 0.10 are above v2's, with the lower bound of a paired bootstrap above 0;
- on the 13 languages other than English pooled, the lower bound of the same bootstrap on the AUC change is above −0.02;
- on Smart Turn's test clips it keeps what v2 had;
- it is a drop-in replacement for v2's file: the same graph, size and latency.

## 2. What changes from v2, and why

| # | change | evidence (study) |
|---|---|---|
| D1 | **Variable trailing silence** in every training view: Smart Turn clips extended with their own background (§3.4), conversation records cut where the policy asks (§3.2) | the first ask (0.2 s) is beyond the 90th percentile of v2's training clips (170–190 ms); 0.4 s is beyond the 97th; v2's eot-bench harness cut at 0 s (§2) |
| D2 | **Real conversations, one speaker per channel**: `otoearth/otoSpeech-full-duplex-280h`, about 280 h of spontaneous English, CC BY 4.0, gated | v2's teacher is itself at 0.7838 on eot-bench; 82.4% of v2's training clips are synthetic (§1.2) |
| D3 | **Labels from the other channel**: end of turn when the other speaker takes the floor within 4 s, hold when the same speaker resumes first; backchannels are not floor takes (the rule of DualTurn's card and code, §3.3 of the study) | the labelling of voice-activity-projection work and DualTurn (arXiv 2603.08216) |
| D4 | **Hard labels on the conversations**; Smart Turn's opinion there is recorded, never a target | the point is to exceed the teacher on real conversations |
| D5 | **Start from v2's weights** (unless the pilot's scratch arm wins, §6) and **120 working minutes** | keeps what v2 learned on Smart Turn's clips; half the GPU |
| D6 | **eot-bench gated**, read at the policy's first question (span start + 0.2 s), paired against v2 | the cut v2 was read at (the span's start) is a point the deployed policy never asks at |

**Not changed, and why.**

- **The features, the graph, the rung (R4), the GGUF keys, the engine, the WASM build.** v3 must be a drop-in replacement file. The engine (`sky_infer::audio`, `sky_infer::turn`, `sky-web`) gets no change.
- **The Smart Turn data release, its split and its teacher targets.** They are those of v2, at the same revisions: `f8c95d28` for the data and `f766f81d` for the teacher.
- **Ternary QAT, EMA, the brake and the working-minute budgets.** v2's recipe, unchanged.

## 3. Data and leakage

### 3.1 Sources

| set | use | size | licence |
|---|---|---|---|
| `otoearth/otoSpeech-full-duplex-280h` (revision recorded at access; read on 17/11 at `4b3c57b8`) | conversation train / val / test by §3.3 | 151 shards, 48,757,552,521 bytes, about 280 h, 1,125 sessions | CC BY 4.0, gated |
| `anyreach-ai/dualturn-otospeech-turn-taking`, file `splits.json` only (revision `5b700b4b`) | the session split (§3.3) | 54,992 bytes | licence `other`; only the session-to-split map is used, no feature and no label |
| `pipecat-ai/smart-turn-data-v3.2-train` at `f8c95d28` | train and val as v2 (257,353 / 13,593); streamed again for the trailing extension (§3.4) | 83 shards, 41,400,926,042 bytes | CC BY 4.0 |
| `pipecat-ai/smart-turn-data-v3.2-test` | **test (a), gated**; v1's WAVs, as v2 | 31,527 clips | CC BY 4.0 |
| `livekit/eot-bench-data`, config `all`, at `ae834af9` (v2's commit) | **test (b), gated** (§8.3) | 7,342 turns, 1,980,396,015 bytes | CC BY 4.0 (card) |
| Smart Turn v3.2 CPU ONNX at `f766f81d` | teacher on Smart Turn clips (v2's targets reused); baseline; recorded opinion on conversation records | 8.68 MB | BSD-2 |
| v2: `models/s6-ear2/s6-ear2-nano.gguf` (sha256 `20435465…a9be783`) and its EMA checkpoint `runs/s6-ear2-nano/last-ema` | comparator; start of training | 631,936 bytes; 5.7 MB | Apache-2.0 (this project) |

- Every shard is streamed and deleted once its records are written, as in v2.
- Every licence is recorded in `DATA_LICENSES.md` before the run.
- v2's checkpoint lives in a session worktree and is not in git. It is copied to the data folder (`skycmd-data/s6-ear3/v2-last-ema/`) with its sha256 before anything else.

### 3.2 Conversation records (from one session)

1. **Decoding.**
   - Each FLAC channel is decoded on its own, never mixed.
   - It is resampled from 48 kHz to 16 kHz by `sky_infer::audio::resample`, the resampler of all of v2's data.
   - Each channel is scaled so that the 95th percentile of its 10 ms frame RMS over the session is 0.1. This changes the VAD's view only: the features are normalised per window.
2. **The deployed VAD and policy, replayed on each channel.**
   - A Rust port of `EnergyVad` and `TurnPolicy` with `TURN_DEFAULTS`: minimum speech 150 ms, `stopMs` 200, `recheckMs` 200, `fallbackMs` 2,500; absolute floor 0.008 RMS, ratio 3.5; floor adaptation 0.05, initial floor 0.002.
   - It is checked bit for bit against `turn.ts` (§16).
   - It is replayed with a model that never answers "finished", so that every ask the policy would make inside a pause is produced.
   - Each pause with at least one ask is one candidate record. Its offset o is the end of the last speech frame before the pause.
3. **The other channel.** The same energy VAD, cleaned as DualTurn cleans its VAD: speech runs under 150 ms removed, silences under 150 ms filled.
   - **A backchannel** is a run of at most 1.0 s with at least 1.0 s of that channel's silence before and after it.
   - **A floor take** is a run of at least 1.0 s that is not a backchannel.
4. **The label** of a pause of A at o. Let r be A's next speech onset (the policy's `resumed`) and t_B the onset of B's first floor take after o.
   - **End of turn:** t_B ≤ o + 4.0 s and t_B < r.
   - **Hold:** r ≤ o + 4.0 s and r < t_B, or no t_B. B's backchannels during the pause are allowed.
5. **Dropped, and counted, before anything else:**
   - the utterance of A that ends at o is itself a backchannel by the rule of item 3;
   - B is inside a floor take at o, or one began in [o − 1.0 s, o): an overlap, whose label is not clean;
   - neither r nor t_B comes within 4.0 s of o;
   - any part of [o − 4.0 s, o + 4.0 s] lies within 0.5 s of a range of `redacted_segments`: the audio is muted there, and the gating conditions forbid reconstructing it;
   - the session fails the bleed filter: the bleed ratio of a channel (its median frame RMS while it is VAD-silent and the other channel speaks, over its median RMS while both are silent) is above 2.0 for either channel;
   - the session is absent from `splits.json`.
6. **Valid asks.** d ∈ {0.2, 0.4, …, 2.4} s such that A's channel is VAD-silent over [o, o + d]: the asks the policy makes during that pause.
7. **What is stored**, in `sky_infer::turn::store` format:
   - A's raw hops from o + 0.2 − 4.0 s to o + the largest valid d: at most 620 hops;
   - a manifest line: session, channel, o, label, largest valid d, t_B − o, r − o, split, and the pseudonymous speaker id of A (for §3.3 only).
   - Nothing of B's channel is stored. No other metadata is stored: no profile, no survey, no topic.
8. **Caps (a disk rule, §3.8).** At most 120,000 training, 20,000 validation and 30,000 test records, each drawn uniformly without replacement from the eligible records of its split (seed 3). The natural class ratio of each split is kept in the store.

### 3.3 Splits

- **Conversation sessions.** DualTurn's `splits.json` (seed 42, ratios 0.8 / 0.1 / 0.1): **900 train, 112 validation, 113 test sessions** of 1,125. The card's table says 896 / 111 / 113; the file is used.
- **Not speaker-disjoint.** DualTurn's own code says speaker-disjoint splitting "is infeasible (99.6% in one connected component)". The conversation test (c) is therefore session-disjoint only, and reported, not gated. The report also gives (c) on the **speaker-unseen subset**: test records whose speaker id appears in no training session. Its size is recorded at the build.
- **Smart Turn clips.** v2's split, unchanged: a 5% hash of the id, seed 1234.
- **eot-bench and Smart Turn's test.** Test only.

### 3.4 Trailing silence on Smart Turn's clips

- **Why the shards are streamed again.** v2's store holds at most the last 400 hops of each clip, and the shards were deleted. Appending silence in feature space would get the boundary hops wrong (a hop reads 40 ms). So `turn-extend` streams the 83 shards again.
- **What it computes.** For each kept clip (v2's leakage filter and ids), the hops that follow the clip's end when its 16 kHz PCM is extended by 2.44 s of fill: 240 hops. They go into a separate store `st-ext/`, one record per clip id.
- **Exactness.** A hop exists once its whole 40 ms span has arrived and the grid starts at the first sample (`turn_hop_count`). The extended clip's hops are therefore v2's hops followed by these, exactly.
- **The fill is the clip's own background.**
  - Take the clip's 10 ms frames whose RMS is in its lowest 10%, at least 20 frames, else its 20 quietest.
  - Put them together in an order drawn with a seed from the clip id, with 2.5 ms raised-cosine cross-fades.
  - If the clip's floor is digital silence (RMS below 1e-5), the fill is zeros, which is close to what the browser's noise suppression delivers. These clips are counted.
- **The view.** With probability 0.25 the window ends at the clip's end, v2's view (d = 0). Otherwise d is drawn as in §3.6 and the window ends with the last hop whose span ends at or before clip end + d.
- **The labels and the teacher's logit and pooled targets are those of the clip.** The trailing silence changes neither the label nor the teacher's opinion of the clip.
- **The test clips are extended the same way** (`turn-extend --test`, from v1's WAVs), for a reported reading only (§8.2).

### 3.5 Leakage rules

- **No test audio** of any of the three tests enters training, validation, the pilot, the calibration, a threshold, the label check or any choice: Smart Turn's test, eot-bench, and the 113 conversation test sessions.
- **Smart Turn clips.** v2's filter, by test id and by PCM hash, is kept; it dropped no row in v2 (amendment 1 of `docs/s6-ear2-prereg.md`).
- **No feature or label of DualTurn's data set is used.** Only its session-to-split map is.
- **No v3 checkpoint reads eot-bench before the gate step**, and v2 reads only the d = 0 cuts of §8.1 before it.
- **The label check (§3.7) uses validation sessions only.**
- **Overlap of speakers or audio between otoSpeech and the two tests cannot be checked.** It is presumed nil (different producers), and the report says so.

### 3.6 Mix, balance and augmentation

- **Batch.** Each training draw is a conversation record with probability 0.5 and a Smart Turn clip with probability 0.5.
- **Conversation records.**
  - The class is drawn first (0.5 each), then a record uniformly within it.
  - Then d is drawn from w(d) restricted to the record's valid asks and renormalised: w(0.2) = 0.5, and w(0.4) = w(0.6) = … = w(2.4) = 0.5 / 11.
- **Smart Turn clips.** Uniform over training clips, as v2 (49.6% finished); d as in §3.4.
- **Augmentation.** v2's light augmentation applies to both sources: context jitter 0.3 (1–4 s), two frequency masks of at most 4 bands, and one time mask of at most 8 hops kept out of the window's last second. The window always ends at its cut (clip end + d, or o + d).

### 3.7 The label check and the bleed probe (after the build, before any training)

- **Label check.**
  - **Sample.** 200 records of validation sessions, 100 labelled end of turn and 100 hold, drawn with seed 7.
  - **What is played.** Each record as a stereo snippet, both channels, from o − 6 s to o + 4 s, kept on this machine only.
  - **Who answers.** A person listens and answers "A had finished", "A had not finished" or "cannot tell", without seeing the label.
  - **Agreement.** The share of the 200 whose answer matches the label, "cannot tell" counted as a disagreement. Reported with its Wilson 95% interval, together with the share of "cannot tell".
  - **Bar.** Agreement ≥ 0.80. Below it, v3 stops before training, and an amendment revises the rule or the corpus.
- **Bleed probe.**
  - **Question.** Does the silence after the offset carry the label by itself?
  - **Measure.** On the training records, at each d, the AUC of one number: the mean raw log10-energy channel of A's hops over [o, o + d].
  - **Bar.** No d may give an AUC above 0.60, which would say that B's voice leaks into A's channel. Above it, v3 stops before training, and an amendment tightens the bleed filter.
- **Both numbers, and the counts of every drop of §3.2, are committed before the smoke run.**

### 3.8 Disk

- **New data at its peak, at most 16.9 GB:**

  | item | size |
  |---|---:|
  | Smart Turn extension, 270,946 × 240 hops of 86 bytes | 5.59 GB |
  | test extension | 0.65 GB |
  | conversation records: training, validation, test | ≤ 6.40 + 1.07 + 1.60 GB |
  | eot-bench span records | ≤ 0.96 GB |
  | one shard in transit | ≤ 0.6 GB |

- **Reused, not copied:** v2's Smart Turn store (11 GB), its test store and v1's test WAVs.
- **The rule of v2 holds:** every step leaves `MIN_FREE_GB` = 8 GB free, so v3 needs **about 25 GB free at its start**.
- **On 17/11/2026, 6.1 GB were free.** The owner frees space first. v1's `data/ear` stays untouched unless the owner decides otherwise.

### 3.9 Scope

v3 answers the same question as v2, "has the speaker finished?", asked by the deployed policy during a pause, from one microphone. These stay out of v3:

- a second channel at inference;
- other languages' conversations;
- keywords;
- any change to the engine, the demo's policy or its defaults.

The demo gets the v3 file only after the result, and by the owner's decision.

### 3.10 The fallback: AMI (only if the owner does not accept otoSpeech's conditions)

- **Source.** The AMI Meeting Corpus from the Edinburgh site, under CC BY 4.0:
  - the individual headset WAVs of every meeting (about 120 MB each, four or five per meeting);
  - the manual annotations v1.6.2 (NXT, word-level timings from forced alignment).
  - The Hugging Face copy `edinburghcstr/ami` holds cut segments and is not used.
- **Splits.** The standard AMI full-corpus partition by meeting, with its train, dev and eval meetings for train, validation and test. Speakers recur across meetings within the scenario sessions; the report says so.
- **Speech activity.** It comes from the word timings, not from an energy VAD, because every headset hears the room (crosstalk).
  - A speaker's speech is the union of their words, with gaps under 150 ms filled.
  - Offsets and asks are still those of the replayed energy VAD and policy on the speaker's headset, and are kept only where the VAD's offset is within 100 ms of a word-timing offset.
- **Label.**
  - **End of turn:** another participant's floor take (≥ 1.0 s, not a backchannel) begins within 4.0 s and before the speaker resumes.
  - **Hold:** the speaker resumes first.
  - **Dropped:** the overlap and the 4 s rules as §3.2; the bleed filter is replaced by the word-timing rule.
- **Disk and caps.** As §3.8, at most 20.5 GB of headsets streamed meeting by meeting.
- **Everything else** in this file is unchanged. An amendment records the choice and the meeting lists before any build.

## 4. The model (`configs/s6-ear3-nano.json`)

- **The graph of `configs/s6-ear2-nano.json`, rung R4:** d 160, ff 640, 2 heads, 4 blocks, 8 hops per position, 50 positions, 43 channels, window 400 hops.
  - The config file holds R4 only. There is no latency probe: the rung is v2's.
- **The start.** v2's EMA checkpoint (`last-ema/model.safetensors`, copied per §3.1), unless the pilot's scratch arm wins (§6). The training-only projections `pool_proj` and `seq_proj` come with it.
- **The export.** The same GGUF keys (`skycmd-turn.*`): window, hops per position, channels, τ₀.₅, τ₈₅, temperature.
  - The exported tensor list, shapes and types must be identical to v2's file. This is checked as a gate term (G3).
  - Its size is expected within a few hundred bytes of 631,936.
- **The code to write** is listed in §16.

## 5. The teacher and the loss

- **Smart Turn clips.** v2's targets, unchanged: logit z_T, pooled e_T, states S_T.
  - L_ST = 0.5 · BCE(y, p) + 0.5 · BCE(σ(z_T), p) + 0.25 · (1 − cos(pool_proj(h̄), e_T)) + 0.5 · [d = 0] · mean over the 25 steps of (1 − cos(seq_proj(h_t), S_T,t))
  - The state term counts only on v2's view (d = 0). With a trailing silence, the teacher's states describe a window that ends at the clip's end, not at the cut.
- **Conversation records.** L_conv = BCE(y, p), hard labels only.
- **No `pos_weight`.** Both sources are balanced by sampling (§3.6).
- **The teacher's opinion of conversation records.** Smart Turn v3.2 CPU, fed the last 8 s of A's channel up to o + 0.2 s, left-padded as Pipecat does. It is computed on the validation and test records while their shard is on disk, and recorded for §8.4 and §10. It is never a target.

## 6. The run (`scripts/train-turn3.sh`)

1. **Data** (CPU only; §3, §5): `fetch-conv`, `turn-extend` (training and test), the eot-bench span records, the teacher's recorded opinion, then the label check and the bleed probe of §3.7, committed.
2. **Harness checks** (§8.1).
3. **Smoke.** 5 working minutes on 2,000 records of each source, to test the chain. Its numbers are not looked at.
4. **Pilot.**
   - **Four arms × 12 working minutes**, R4, batch 128, seed 1, f32 weights (no QAT), the mix of §3.6:

     | arm | start | rate |
     |---|---|---|
     | P1 | v2 | 0.25 × 0.0007745 (v2's chosen rate) |
     | P2 | v2 | 0.5 × 0.0007745 |
     | P3 | v2 | 1 × 0.0007745 |
     | P4 | scratch | 0.0007745 |

   - **Metric.** AUC on the conversation validation records cut at d = 0.2 s: all of them, the headline point.
   - **Eligibility.** An arm is eligible if its Smart Turn validation accuracy at τ 0.5 (d = 0, v2's view, f32) is at least 0.88.
   - **Choice.** The eligible from-v2 arm with the highest metric. The scratch arm P4 is taken instead only if it is eligible and its metric is higher by more than 0.005.
5. **Stop rule.**
   - **The bar.** v2's file read on the same validation cuts gives v2's metric. If the chosen arm's metric is below v2's + 0.01, or if no arm is eligible, v3 stops there:
     - no main run;
     - it is recorded as failed by rule, with the pilot numbers.
   - **v2 is read here on validation only**, never on a test.
6. **Training.**
   - **Budget.** 120 working minutes from v2's weights, or 240 if P4 was taken.
   - **Recipe.** Batch 128, seed 1, once; the recipe's optimiser at the chosen rate, WSD schedule; ternary QAT over the last 15%, EMA 0.999.
   - **Machine.** The brake modes and the memory cap of v2. Minutes count working time only, and the report gives the steps done and the modes.
7. **Restarts.** Only a technical failure justifies a relaunch from the last checkpoint, and it is reported.

**Total GPU: 173 working minutes** (5 + 48 + 120), or 293 if the scratch arm is taken. That is about 6 (or 9) hours of wall clock in the brake modes. Evaluation and calibration run on the CPU.

## 7. Checkpoint, calibration, export

- **Checkpoint.** The EMA if its ternary validation loss is below that of `last`, else `last`. The loss is the mean, with equal weights, of:
  - the BCE on Smart Turn validation clips at d = 0;
  - the BCE on conversation validation records at d = 0.2 s.
- **Temperature.** Fitted on Smart Turn validation at d = 0 (NLL), as v2.
- **Thresholds.** Both come from Smart Turn validation at d = 0, as v2, and are stored under the same keys:
  - τ₀.₅ = 0.5;
  - τ₈₅ = the highest threshold whose validation END recall is ≥ 0.85.
- **Reported, not stored.** The conversation operating point τ_c: the lowest threshold whose false cut-off on the conversation validation records at d = 0.2 s is ≤ 0.10. It goes to the card and the operating notes. Storing it would need a new key, and the engine does not change.
- **Export.**
  - Ternary GGUF.
  - WASM parity on 20 Smart Turn test clips and on 20 conversation test records: |Δp| ≤ 1e-4 and the same decision.
  - The same-graph check of §4.

## 8. Evaluation

### 8.1 Harness checks (before any v3 test score is read)

1. **The policy port.** It replays recorded `turn.ts` vectors bit for bit (§16).
2. **Smart Turn's test.**
   - v2's harness check stands (`runs/s6-ear2-harness/check.json`, 0.9260).
   - v2's file through v3's evaluation path must give v2's 0.8993 to four decimals.
3. **eot-bench.**
   - The new span builder, cut at each span's start (v2's cut), must give v2's 17,934 cuts, v2's AUC 0.7299 and Smart Turn's 0.7838, each to ± 0.0005.
   - Then the number of cuts of §8.3 is recorded, before any model reads them.
4. **If a check fails,** the cause is found and fixed by an amendment before any v3 test score is read. No bar of §9 is moved to fit.

### 8.2 Test (a), gated: `smart-turn-data-v3.2-test`, 31,527 clips, as v2

- **The cut.** The clips as they are (d = 0), through the WAV path and the file as exported.
- **Metrics:** v2's metrics of `docs/s6-ear2-prereg.md` §8.2, with 95% bootstrap intervals by clip (1,000 resamples):
  - accuracy at τ₀.₅; precision, recall, F1;
  - false cut-off and END recall at τ₈₅;
  - AUC, ECE;
  - per-language and per-dataset tables.
- **Reported, not gated:** v2 and v3 on the test clips extended by 0.2 s and by 0.6 s of their own background (§3.4).

### 8.3 Test (b), gated: eot-bench at the policy's first question

- **The cuts.**
  - Every span labelled `eot` or `hold` whose length (end − start) is at least 0.2 s, cut at **start + 0.2 s**.
  - `hold` spans longer than 5.0 s are ignored, as LiveKit's harness ignores them.
  - The card says every `general` row's `eot` span lasts at least 0.2 s "so every turn can be evaluated at a 0.2s score point".
  - v2 and v3 read the last 4 s; Smart Turn v3.2 CPU reads the last 8 s, left-padded.
- **Metrics.**
  - AUC of `eot` against `hold`.
  - **Eot recall at a false cut-off of 0.10.** The ROC point: the largest recall among thresholds whose false cut-off on the evaluated cuts is ≤ 0.10. It is a ranking measure, not a deployable threshold, and the report says so.
  - Reported beside them, at the file's τ₈₅ and τ₀.₅: false cut-off and eot recall.
- **Uncertainty.** A paired bootstrap of Δ = v3 − v2 (and v3 − Smart Turn, reported):
  - 2,000 resamples by turn, all cuts of a turn together;
  - seed 20261117;
  - percentile 95% intervals.
- **Breakdowns:**
  - all 14 languages pooled;
  - English alone;
  - the 13 other languages pooled;
  - each language;
  - each cohort (`general`, `accented`, `short_turns`).
- **Reported, not gated:**
  - **The policy reading.** Every ask at 0.2, 0.4, …, 2.4 s that fits inside a span, per d, for v2 and v3.
  - **The policy simulated per turn at τ₈₅ and at τ_c:**
    - the share of `eot` spans answered "finished" by the model before the 2.5 s timer, and the time of that answer;
    - the share of `hold` spans with at least one false cut-off.
  - **Continuity.** v3 at v2's cuts (the span's start).
- **Expected per language** (a judgement, written now):
  - English gains most: ΔAUC +0.03 to +0.08.
  - The other 13 move between −0.01 and +0.03. Only the trailing-silence fix applies to them, and half the training draws are English.
  - The gate therefore pools all languages, bounds the loss of the non-English pool, and reports English alone.

### 8.4 Test (c), reported: the 113 held-out conversation sessions

- **Models and cuts.** v2, v3, and Smart Turn's recorded opinion (at d = 0.2 s), on the test records:
  - AUC and eot recall at a false cut-off of 0.10, per d;
  - the same on the speaker-unseen subset (§3.3).
- **Why it is not gated.** Session-disjoint only, and human-to-human: it says how well v3 learned its own corpus, not how it does with agents.

### 8.5 Latency and size (gated by G3; measured before v3's test is read)

As `docs/s6-ear2-prereg.md` §8.3:

- On the M5, one thread, 1,000 decisions on a full 4 s window, features already computed. Median and p95, natively and in WASM (Node, SIMD128).
- The front end per 10 ms hop.
- The GGUF size.
- v2's file, timed in the same session.
- No training running.

## 9. Gate (ternary GGUF, the file as exported)

v3 passes if all of these hold.

1. **A1 (Smart Turn's test, accuracy).**
   - **The bar.** Accuracy at τ₀.₅ on the 31,527 clips **≥ 0.8893**, that is v2's 0.8993 − 0.01.
   - **It contains v2's G1.** The bar implies ≥ 0.85 and ≥ T − 0.08, with T = 0.9260.
2. **A2 (Smart Turn's test, cut-offs).** At τ₈₅ (from validation), false cut-off **≤ 0.20** and END recall **≥ 0.80**: v2's G2.
3. **E1 (eot-bench, ranking).** On the cuts of §8.3, all languages pooled, the 95% lower bound of AUC(v3) − AUC(v2) is **> 0**.
4. **E2 (eot-bench, the operating region).** On the same cuts, the 95% lower bound of recall@FC0.10(v3) − recall@FC0.10(v2) is **> 0**.
5. **E3 (eot-bench, no harm outside English).** On the 13 languages other than English pooled, the 95% lower bound of AUC(v3) − AUC(v2) is **> −0.02**.
6. **G3 (drop-in).** All of these:
   - WASM median decision time **≤ 5.0 ms**;
   - front end ≤ 0.10 ms per 10 ms hop;
   - GGUF **≤ 3 MB**;
   - WASM parity (§7);
   - **the tensor list, shapes and types identical to v2's file**.

**Not gated:**

- **The announced goal: v3's pooled AUC ≥ Smart Turn's on the same cuts.** Smart Turn has never been read at these cuts. Its 0.7838 is a reading at the span's start. The estimated odds are about 0.15 (study §7), and the teacher is a model of another size class.
- English alone, each language, each cohort.
- Test (c) and the policy reading.

**The sentence of the report.** It says in one sentence which file a user should take, and on what evidence: v3, v2, or Smart Turn's 8.7 MB file. It gives the change on eot-bench and on Smart Turn's clips.

## 10. Reported, not gated

- **All of §8,** with intervals.
- **Data.**
  - The counts of every drop of §3.2: by rule, sessions removed by the bleed filter, records near redactions.
  - The classes and d distribution of the stored records.
  - The share of fill made of zeros (§3.4).
  - The label check and the bleed probe (§3.7).
- **Pilot.** Its arms, its choice, and the stop-rule reading against v2's validation metric.
- **Training.** Steps, modes, minutes; validation curves for both sources; teacher–student agreement on Smart Turn validation.
- **Agreement on conversations.** Between Smart Turn's recorded opinion and the conversation labels, and between Smart Turn and v3, on test (c).
- **Deployment.** The operating notes, with τ_c beside τ₈₅ and τ₀.₅.

## 11. Disclosure

**What v3's design read.**

- **v2's eot-bench reading.** v3 was designed after it: the overall, per-language and per-cohort numbers of `runs/s6-ear2-nano/GATE.json` were seen.
- **Some choices were shaped by it.** That reading motivated:
  - making eot-bench the headline;
  - the no-harm bar outside English;
  - the expected per-language effects.
- **eot-bench stays out of every fit and every choice.** It has never been used for training or for any choice of v2's or v3's training, and stays so. No eot-bench file was opened for v3's design, and the cut point of §8.3 (0.2 s) comes from the demo's policy and LiveKit's card, not from a reading.
- **The demo.** The variable trailing silence was motivated by the microphone demo (`turn.ts`), through the code, not through any test set.
- **The training data.** v3's study measured the trailing silence of 3,160 of v2's training clips.
- **Smart Turn's test is still partly in-sample for the design**, as v2's disclosure says. v3 adds no reading of it before the gate.

**What it means.**

- **The corpus is human-to-human; eot-bench is human-to-agent.** The labels of the training conversations are the listener's behaviour, not an agent's need.
- **English only.** The training conversations are English; eot-bench has 14 languages.
- **The conversation test is session-disjoint, not speaker-disjoint.**

Every report of v3 states this paragraph.

## 12. What a failure would mean (written now)

- **The label check is below 0.80, or the bleed probe above 0.60 (§3.7).** The records do not measure "has A finished?" from A's channel alone. No training happens; an amendment revises the rule, the filter or the corpus, or the line stops there.
- **The stop rule fires (§6.5).** Twelve minutes of conversation records did not move v2's ranking on conversation validation by 0.01. Either the human-to-human labels cannot be learned by this graph from one channel, or 1.3M parameters cannot absorb them on top of Smart Turn's clips.
  - The line stops at this size.
  - v2 stays the published file, with its fallback timer.
  - For real conversations the recommendation stays Smart Turn's file, with the reading of v2's card.
- **A1 or A2 fails.** v3 bought its gain on conversations with a loss on Smart Turn's clips. It does not pass, and v2 stays the published file. If E1–E3 hold, the report says that v3 is better on real conversations and worse on clips: a trade, not a pass.
- **E1 fails.** No improvement on real human-to-agent speech is shown. Test (c) separates the two causes:
  - **(c) improves over v2:** the human-to-human signal does not transfer to agents;
  - **(c) does not improve:** the model did not learn the signal at all.
- **E1 holds and E2 fails.** The ranking improved, but not where a voice agent works (few cut-offs). It does not pass.
- **E3 fails.** The English gain was paid for by the other languages. It does not pass. A follow-up would need conversations in other languages, not a reweighting found on eot-bench.
- **G3 fails.** v3 is not a drop-in replacement. With the same graph this can only be a defect of the export, which is investigated. It does not pass.
- **v3 passes.** It is published as passed, with the sentence of §9. It replaces v2 in the demo only by the owner's decision. Its card's first lines give the eot-bench numbers and the human-to-human limit of §11.
- **The announced goal.** Met: the report says v3 matches or beats Smart Turn on real conversations at a fifteenth of its file. Not met: it says v3 is still below Smart Turn there, with the gap.
- **In every case,** following the owner's decisions of 15/11/2026 and 17/11/2026, the file is published as evaluated, with its Space. Its first lines give the verdict and the numbers.

## 13. Forbidden

- Training, validating, piloting, calibrating, choosing a threshold or listening to the label check on:
  - any Smart Turn test clip;
  - any eot-bench audio;
  - any of the 113 conversation test sessions.
- Using DualTurn's features or labels.
- Any v3 checkpoint reading eot-bench before the gate step.
- Reading a v3 test score before §8.1's checks and §8.5's latencies are on file.
- A second seed, a second run, or another start after the pilot's choice, except after a technical failure.
- Changing a bar, a gate term, the cut point or a threshold after a v3 test score is read.
- Attempting to identify, de-anonymise or link a speaker of the corpus.
- Using, reconstructing or inferring the content of a redacted segment.
- Publishing the corpus's audio, snippets or derived records, unless the owner decides it in an amendment.
- Generating any label or example with a model. The labels come from the rule of §3.2, the check from a person (`AGENTS.md`).

## 14. Amendments

None. This file is a draft (see the top).

## 15. Choices (the first value of each line is the one proposed; the alternative is recorded)

| | choice | proposed | alternatives | why the proposal |
|---|---|---|---|---|
| O1 | corpus | **otoSpeech 280 h, raw** | the processed 141 h (half the download, noise-reduced, human-selected, 44.1 kHz); AMI (the fallback, §3.10); CANDOR (licence allows a model, forbids sharing derived data; about 346 GB) | dyads, natural microphone silence, CC BY 4.0, and the DualTurn split covers it |
| O2 | session split | **DualTurn's `splits.json`** | a new seeded split by session; a speaker-grouped split of a subset | comparability with DualTurn; a speaker-disjoint split is infeasible by its own count |
| O3 | VAD for offsets and asks | **the demo's energy VAD and policy, ported** | Silero VAD (DualTurn's) | offsets and silences then mean exactly what they mean when the model is asked |
| O4 | VAD for the other speaker | **the same energy VAD, cleaned 150 / 150 ms** | Silero on B's channel | one detector and no new dependency; bleed is handled by the session filter and the probe |
| O5 | label rule | **card and code of DualTurn, with "nobody within 4 s" and overlaps dropped** | DualTurn's code exactly (nobody: hold; overlap: end of turn) | clean labels at the points the policy asks |
| O6 | d distribution | **0.5 at 0.2 s, 0.5 spread over 0.4–2.4 s** | uniform over valid asks; 0.2 s only | the first ask sets an agent's latency and is the headline cut; the rechecks still occur in deployment |
| O7 | silence on Smart Turn clips | **exact PCM extension with the clip's own background (shards streamed again, 41.4 GB)** | appending silence hops in feature space (no download, wrong boundary hops); zeros; an external noise corpus | exact hops, the clip's own acoustics, independent of the label |
| O8 | share of v2's view (d = 0) on Smart Turn clips | **0.25** | 0; 0.5 | gate A1 reads clips at d = 0 |
| O9 | start | **v2's EMA checkpoint, unless the scratch arm beats it by > 0.005** | from scratch (240 min) | keeps v2's knowledge of Smart Turn's clips and halves the GPU |
| O10 | conversation share | **0.5** | 0.3; 0.7 | equal weight to the two signals; not piloted, to keep the arms for the start and the rate |
| O11 | conversation class balance | **50 / 50 by sampling** | the natural ratio | v2's clips are balanced and the BCE is not prior-free |
| O12 | target on conversations | **hard labels; teacher recorded** | distil the teacher there too | the aim is to exceed the teacher there |
| O13 | state term on extended views | **masked (d > 0)** | shift the 25 steps to the clip's end | the teacher's states describe a window that ends at the clip's end |
| O14 | main budget | **120 working minutes (240 if from scratch)** | 240 from v2 | at v2's 1,274 steps a minute, about 153,000 steps: about 80 visits per conversation record and 38 epochs of Smart Turn clips; v2 already holds 240 minutes |
| O15 | pilot metric | **conversation validation AUC at d = 0.2 s, Smart Turn validation ≥ 0.88 to be eligible** | AUC over all asks; Smart Turn validation accuracy | the headline point, without giving up the clips |
| O16 | stop rule | **chosen arm ≥ v2's validation metric + 0.01** | none; + 0.02 | spends two hours only if twelve minutes moved the ranking |
| O17 | eot-bench cut | **span start + 0.2 s (spans ≥ 0.2 s; holds > 5 s ignored)** | the span's start (v2's cut) | the policy's first ask, and LiveKit's own 0.2 s score point |
| O18 | false cut-off of E2 | **0.10** | 0.20; 0.05 | cutting users off is a voice agent's main failure; v2 read 0.2230 at τ₈₅ there; the timer covers missed ends |
| O19 | bootstrap | **paired, by turn, 2,000 resamples** | by cut, 1,000 (v2's) | the cuts of one turn are not independent |
| O20 | gate A1 | **≥ 0.8893 (v2 − 0.01), with v2's G2** | v2's bars only (≥ 0.85) | "must not lose what v2 had" |
| O21 | no-harm bar outside English | **lower bound > −0.02** | none; > 0 | the new data is English; the other languages must not pay for it |
| O22 | minimum effect for E1 | **none beyond a lower bound > 0** | a point estimate ≥ +0.02 as well | the lead's rule; +0.02 would be 37% of v2's gap to Smart Turn |
| O23 | thresholds in the file | **v2's rules on Smart Turn validation; τ_c reported** | τ₈₅ from conversation validation | the same keys and meaning, no engine change |
| O24 | caps | **120,000 / 20,000 / 30,000 records** | every record (about 25 GB more) | disk |
| O25 | label-check bar | **0.80** | 0.85; 0.90 | DualTurn reports no check; below 0.80, one label in five would be wrong |
| O26 | bleed filter | **ratio ≤ 2.0 (+6 dB)** | none; 1.5 | below the VAD's 3.5 × (+10.9 dB), so kept bleed cannot fire the VAD |
| O27 | Smart Turn | **an announced goal** | gated | never read at these cuts; odds about 0.15 |

## 16. Code to write before any run

**sky-data:**

- `src/turn_policy.rs`: the Rust port of `EnergyVad` and `TurnPolicy` with `TURN_DEFAULTS`.
  - Test: replay of vectors recorded from `site/.vitepress/theme/lib/turn.ts` by `scripts/turn-policy-vectors.mjs` (Node, type stripping), stored in `crates/sky-data/tests/fixtures/turn-policy.json`. The same events at the same frames, bit for bit.
- `src/audio.rs`: `decode_flac_channels` (one vector per channel, no mixing).
- `src/conv.rs`, called from `src/main.rs`:
  - `fetch-conv`: the otoSpeech WebDataset shards one at a time (`HF_TOKEN`, `tar`, `claxon`), with §3.2 in full: gain, VAD and policy replay, the other channel, the label rule, every drop with its count, the bleed filter, redactions, `splits.json`, the caps, the records, the manifest and `CONV.json`.
  - A transient 8 s PCM tail of each validation and test record for the teacher, removed with its shard.
  - The disk rule of v2 (`Guard`, exit status 3, resumable).
  - Tests on a synthetic two-channel session with a known answer.
- `src/turn.rs`:
  - `turn-extend`: the 83 shards again; the fill and the 240 extension hops per kept clip into `st-ext/`; `--test` on v1's WAVs.
  - `turn-eotbench --spans`: one record per span from start − 3.8 s to min(end, start + 2.4 s), the 0.2 s and 5 s rules of §8.3. v2's cut mode is kept for §8.1.
  - Tests: the extension equals the tail of the hops of the extended PCM, bit for bit.
- `src/ami.rs`: only if the fallback is chosen (§3.10). `fetch-ami`: the headsets meeting by meeting and the NXT word timings.

**sky-train:**

- `src/turn/data.rs`:
  - the two sources;
  - the views with d (§3.4, §3.6) and w(d);
  - the class balance and the share;
  - the per-row loss masks (hard-only rows; the state term at d = 0 only).
- `src/turn/train.rs`:
  - `--init <checkpoint>` (weights only, the optimiser fresh), `--conv <store>`, `--st-ext <store>`, `--conv-share`, `--st-d0`;
  - validation on both sources: Smart Turn accuracy at d = 0, conversation AUC at d = 0.2 s, and the checkpoint loss of §7, logged.
- `src/turn/config.rs`: no change if `configs/s6-ear3-nano.json` reuses the schema.
- Tests: the loader's views end at their cut; the masks; a CPU run on a tiny store learns, resumes and starts from a checkpoint.

**sky-infer (no engine change):** `src/bin/sky-turn.rs`: `--store … --trail-ms <ms>` reads span and conversation records at a given d, through the unchanged model.

**Scripts:**

- `scripts/train-turn3.sh`, modelled on `scripts/train-turn.sh`, with `DRY_RUN`, `RESUME`, `SMOKE` and the brake. Its steps:
  - conv-data, st-extend, eotbench-data, teacher-conv;
  - labelcheck, bleed;
  - harness3;
  - smoke, pilot, train, export, calibrate;
  - latency, wasm;
  - eval, eotbench, conv-test, gate.
- `configs/s6-ear3-nano.json`: R4 of `configs/s6-ear2-nano.json`.
- `scripts/turn-teacher.py --records`: Smart Turn on the PCM tails of conversation records, recorded only.
- `scripts/ear-baseline.py --eotbench --offset-ms 200`: Smart Turn on the cuts of §8.3.
- `scripts/turn-labelcheck.py`: writes the 200 stereo snippets and a blank answer sheet (local only); reads the answers back; gives the agreement with its Wilson interval.
- `scripts/turn-gate.py`, new subcommands:
  - `bleed` (§3.7);
  - `pilot3` (§6.4–6.5);
  - `checkpoint3` (§7);
  - `harness3` (§8.1);
  - `gate3` (§9, with the paired bootstrap by turn and the breakdowns of §8.3).
- `scripts/test_turn_gate.py`: tests of the new subcommands on synthetic probabilities.

**Before the run:** the rows of `DATA_LICENSES.md` for otoSpeech (CC BY 4.0, gated, its role, the conditions accepted) and for DualTurn's `splits.json` (licence `other`; only session ids used).

**After the result:** `hf/cards/s6-ear3-nano.md`, the result block of this file, and the site's pages as `AGENTS.md` asks.

About 2,000 lines in all.
