# s6-ear v3 (s6-ear3-nano): pre-registration

*Status on 19/11/2026: v3 ran on 19/11/2026 and its gate was read once, at 23:39: **it failed** (A1, A2 and G3 hold; E1, E2 and E3 fail). See the result block at the end of this file (§17) and `runs/s6-ear3-nano/`. v2, `s6-ear2-nano`, stays the line's published file; the frozen text below is unchanged.*

**Status: frozen on 19/11/2026, before any line of v3's code is written, before any v3 data is built and before any v3 computation.** The draft of 17/11/2026 became this pre-registration when the three conditions it left open were met (below). Nothing of the draft's bars, rules, thresholds or choices moved at the freeze: the freeze records the conditions, the corpus and the values the draft said were "fixed at the freeze".

**The lead accepted the draft on 17/11/2026** with three changes, all applied here:

- the minimum effect of E1 (§9);
- the automatic label cross-check of §3.7, in place of a listening bar;
- "gate on all languages" recorded as decided (§15).

**The three conditions of the draft, met on 19/11/2026:**

1. **The corpus.** The owner accepted otoSpeech's access conditions on Hugging Face on 19/11/2026: the access form was submitted and access was granted. The use stated on the form is training and evaluation, with no derived records published. The revision at access is `4b3c57b8947d5f651dd6b5e825e7f34b205759e5`: 153 files (the 151 shards `data/train/shard-000000.tar` … `shard-000150.tar`, 48,757,534,720 bytes, with `README.md` and `.gitattributes`). The corpus of v3 is therefore otoSpeech at that revision, and **the AMI fallback of §3.10 is not taken.**
2. **The disk.** At least 72 GB are free on the data disk on 19/11/2026 (the lead's reading; `df` gave 123 GiB available at the freeze), against the about 25 GB of §3.8. v1's `data/ear/audio` has not been touched.
3. **v2's EMA checkpoint.** It is at `/Users/maxime/Documents/Code/skycmd-data/s6-ear2/checkpoints/last-ema/` (that is `../skycmd-data/s6-ear2/checkpoints/last-ema/` from the main checkout), with its sha256 list next to it, `last-ema.sha256`. The list, copied here and into `runs/s6-ear3-nano/init.json` (§16):

   ```text
   43269a549a6b06adeed59bf23854d23e2cee915f2e0eb4c09fcd85be40588f80  config.json
   176cc9300fd374bbae1fb0276398cae5497f172090c2ec7ff32c644ae7a2f3ec  model.safetensors
   1fc6be3017b6ca286effef4d908648a575f41f5cfd461430509ebdb6844746a3  optimizer.safetensors
   1494ca9b8acc3bed2a3a363c12b674ec5e4385e85def61d1c763664f97d898de  state.json
   ```

Records derived from otoSpeech are not published (§13), as the owner stated on the access form.

- At the freeze, no line of v3's code is written, no v3 data is built and no v3 model exists. The Hub listing of the corpus at the revision above and the two dataset cards were read for the freeze. For the tests of the code to come (the lead's permission of 19/11/2026), `splits.json` (sha256 `7550074d7b1761d946dcf3f0bb61fbf1372fe97becd68d752ce631a21ef8b436`) and one shard, `shard-000150.tar` (127,406,080 bytes, three training sessions), were downloaded into a temporary folder before this commit; only the shard's member names, its session metadata fields and a FLAC stream header (48 kHz, two channels, 16 bits) were read. No audio was decoded, no record was built and nothing of it is kept.
- This file fixes:
  - the data (otoSpeech at the revision above), the label rule, the clip construction, the model and its start;
  - the run, the choice of checkpoint, the evaluation and the gate.
- The results are then read through it.
- Three things are fixed later, by rules written here, before training starts:
  - the automatic label cross-check and the bleed probe of §3.7;
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
- **The request.** `TODO.md` (removed on 19/11/2026; last version at commit `0e45b7b`), "What remains", item 5: train with trailing silences of variable length, as the microphone delivers them, and read it on real conversations.
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

- on eot-bench, at the policy's first question (0.2 s into each pause):
  - its AUC is above v2's by at least 0.02, with the lower bound of a paired bootstrap above 0;
  - its end-of-turn recall at a false cut-off of 0.10 is above v2's, with the lower bound above 0;
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
| `otoearth/otoSpeech-full-duplex-280h` at `4b3c57b8947d5f651dd6b5e825e7f34b205759e5` (the revision at access, 19/11/2026; the same as read on 17/11) | conversation train / val / test by §3.3 | 151 shards, 48,757,534,720 bytes (48,757,552,521 with the card and `.gitattributes`), about 280 h, 1,125 sessions | CC BY 4.0, gated; conditions accepted by the owner on 19/11/2026 |
| `anyreach-ai/dualturn-otospeech-turn-taking`, file `splits.json` only (revision `5b700b4b7fe6a21aa3a5acce5f235913fb40cb2e`) | the session split (§3.3) | 54,992 bytes | licence `other`; only the session-to-split map is used, no feature and no label |
| `pipecat-ai/smart-turn-data-v3.2-train` at `f8c95d28` | train and val as v2 (257,353 / 13,593); streamed again for the trailing extension (§3.4) | 83 shards, 41,400,926,042 bytes | CC BY 4.0 |
| `pipecat-ai/smart-turn-data-v3.2-test` | **test (a), gated**; v1's WAVs, as v2 | 31,527 clips | CC BY 4.0 |
| `livekit/eot-bench-data`, config `all`, at `ae834af9` (v2's commit) | **test (b), gated** (§8.3) | 7,342 turns, 1,980,396,015 bytes | CC BY 4.0 (card) |
| Smart Turn v3.2 CPU ONNX at `f766f81d` | teacher on Smart Turn clips (v2's targets reused); baseline; recorded opinion on conversation records | 8.68 MB | BSD-2 |
| v2: `models/s6-ear2/s6-ear2-nano.gguf` (sha256 `20435465…a9be783`) and its EMA checkpoint, copied to `../skycmd-data/s6-ear2/checkpoints/last-ema/` | comparator; start of training | 631,936 bytes; 5.7 MB | Apache-2.0 (this project) |
| Silero VAD **v6.2.3** (tag commit `5cd7945676eb32225748052e2e6a0580e4686a08`, 23/09/2026), file `src/silero_vad/data/silero_vad.onnx` (2,327,524 bytes, git blob `80c5592e`) | the second detector of the label cross-check (§3.7) **only**; never for training labels or for selecting records | 2.3 MB | MIT |

- Every shard is streamed and deleted once its records are written, as in v2.
- Every licence is recorded in `DATA_LICENSES.md` before the run.
- v2's checkpoint lived in a session worktree and is not in git. The lead copied it to `../skycmd-data/s6-ear2/checkpoints/last-ema/`. The sha256 of each of its files is recorded at the freeze (condition 3 above, and `runs/s6-ear3-nano/init.json`, §16).
- Silero's file is pinned by the sha256 of its download, recorded at the first use, and it must be the file of the tag above.

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

- **No test audio** of any of the three tests enters training, validation, the pilot, the calibration, a threshold, the label cross-check, the listening check or any choice: Smart Turn's test, eot-bench, and the 113 conversation test sessions.
- **Smart Turn clips.** v2's filter, by test id and by PCM hash, is kept; it dropped no row in v2 (amendment 1 of `docs/s6-ear2-prereg.md`).
- **No feature or label of DualTurn's data set is used.** Only its session-to-split map is.
- **No v3 checkpoint reads eot-bench before the gate step**, and v2 reads only the d = 0 cuts of §8.1 before it.
- **The label cross-check and the listening check (§3.7) use validation sessions only.**
- **Overlap of speakers or audio between otoSpeech and the two tests cannot be checked.** It is presumed nil (different producers), and the report says so.

### 3.6 Mix, balance and augmentation

- **Batch.** Each training draw is a conversation record with probability 0.5 and a Smart Turn clip with probability 0.5.
- **Conversation records.**
  - The class is drawn first (0.5 each), then a record uniformly within it.
  - Then d is drawn from w(d) restricted to the record's valid asks and renormalised: w(0.2) = 0.5, and w(0.4) = w(0.6) = … = w(2.4) = 0.5 / 11.
- **Smart Turn clips.** Uniform over training clips, as v2 (49.6% finished); d as in §3.4.
- **Augmentation.** v2's light augmentation applies to both sources: context jitter 0.3 (1–4 s), two frequency masks of at most 4 bands, and one time mask of at most 8 hops kept out of the window's last second. The window always ends at its cut (clip end + d, or o + d).

### 3.7 The label cross-check and the bleed probe (after the build, before any training)

**The automatic label cross-check (the bar).**

- **The second detector.** On every validation session, the labels are derived a second time with an independent detector, Silero VAD v6.2.3 (`silero_vad.onnx` of §3.1, MIT).
  - **Input.** The same 16 kHz, gain-normalised channels as §3.2.
  - **Settings.** 512-sample windows (32 ms); a window is speech when its probability is ≥ 0.5; the speech runs are cleaned as in §3.2 item 3 (runs under 150 ms removed, silences under 150 ms filled).
- **The offsets.** On each channel, a Silero offset is the end of a speech run of at least 150 ms that is followed by at least 200 ms of silence. These are the same conditions as the policy's first ask, without its energy detector.
- **The same rule.** The label rule and the drops of §3.2 items 3–5 are applied to the Silero speech runs of both channels, with the same code (it takes speech runs as its input). This gives each Silero offset a label, end of turn or hold, or a drop.
- **Matching.**
  - Each eligible energy record of a validation session (before the cap of §3.2 item 8) is matched to the nearest Silero offset of the same channel within **150 ms**.
  - The matching is one to one: the closest pairs first, then each offset is used once.
  - Why 150 ms: it is the smallest pause either detector keeps after cleaning, and it is below the 200 ms of `stopMs`, so two distinct pauses cannot match the same offset.
- **Agreement.** Over the matched pairs where both paths give a label (neither drops the offset), the share whose eot/hold label is the same.
- **Unconfirmed share.** The share of eligible energy records that are either unmatched or matched to an offset the Silero path drops.
- **What happens to unconfirmed records.** They stay in the data, with the energy path's label. The cross-check measures the labels; it never selects or relabels a record. Selecting records by Silero would make the training labels depend on it.
- **The two bars.**
  - Agreement **≥ 0.85**.
  - Unconfirmed share **≤ 0.30**.
- **Why the second bar.** The agreement speaks only for the records both detectors find. Below 70% confirmed, it would describe a minority of the data, and the energy detector would be finding pauses that a speech detector does not see: breaths, noise, bleed.
- **Below either bar,** the build stops before any training, the report says which bar failed and by how much, and only an amendment can revise the rule, the filter or the corpus.
- **Reported:**
  - the agreement with its Wilson 95% interval;
  - the 2 × 2 table of labels;
  - the unconfirmed share by class and split into unmatched and dropped;
  - the distribution of the matched offsets' time differences.
- **Silero is used for this check only.**

**The listening check (optional, reported, not required).**

- **When.** Done only if the owner volunteers. No decision depends on it.
- **Sample.** 200 validation records, 100 labelled end of turn and 100 hold, drawn with seed 7.
- **What is played.** Each record as a stereo snippet, both channels, from o − 6 s to o + 4 s, kept on this machine only.
- **Who answers.** The listener answers "A had finished", "A had not finished" or "cannot tell", without seeing the label.
- **Reported.** The agreement, "cannot tell" counted as a disagreement, with its Wilson interval and the share of "cannot tell".
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
- **On 19/11/2026, at the freeze, at least 72 GB are free** (condition 2 above), and v1's `data/ear/audio` is untouched.

### 3.9 Scope

v3 answers the same question as v2, "has the speaker finished?", asked by the deployed policy during a pause, from one microphone. These stay out of v3:

- a second channel at inference;
- other languages' conversations;
- keywords;
- any change to the engine, the demo's policy or its defaults.

The demo gets the v3 file only after the result, and by the owner's decision.

### 3.10 The fallback: AMI (only if the owner does not accept otoSpeech's conditions)

**Not taken:** the owner accepted otoSpeech's conditions on 19/11/2026 (condition 1 at the top). This section is kept as it was written and applies to nothing in v3.

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

1. **Data** (CPU only; §3, §5):
   - `fetch-conv`, `turn-extend` (training and test), the eot-bench span records, the teacher's recorded opinion;
   - then the automatic label cross-check and the bleed probe of §3.7, committed.
   - The listening check, only if the owner volunteers, is reported when done and gates nothing.
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
  - Reported beside them:
    - eot recall at false cut-offs of 0.05 and 0.20;
    - false cut-off and eot recall at the file's τ₈₅ and τ₀.₅.
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
3. **E1 (eot-bench, ranking).** On the cuts of §8.3, all languages pooled, AUC(v3) − AUC(v2) has a point estimate **≥ +0.02** and a 95% lower bound **> 0** (the paired bootstrap of §8.3).
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
  - The label cross-check, the bleed probe and, if it was done, the listening check (§3.7).
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
- **The label cross-check and the caps were amended after a measurement on five training sessions** (amendment 2, 19/11/2026), before any validation or test session was read: the cross-check reads Silero's label at the energy record's instant instead of matching offsets, and the caps keep every end-of-turn record.

**What it means.**

- **The corpus is human-to-human; eot-bench is human-to-agent.** The labels of the training conversations are the listener's behaviour, not an agent's need.
- **English only.** The training conversations are English; eot-bench has 14 languages.
- **The conversation test is session-disjoint, not speaker-disjoint.**

Every report of v3 states this paragraph.

## 12. What a failure would mean (written now)

- **The label cross-check gives an agreement below 0.85 (§3.7).** The eot/hold labels depend on which detector finds the speech. More than one label in seven would change with an equally defensible detector, so the labels are not a stable target.
  - The build stops before any training, and the report says so.
  - An amendment revises the rule or the corpus, or the line stops there.
- **The unconfirmed share is above 0.30.** The energy detector finds many pauses that a speech detector does not: breaths, noise or bleed. The agreement would speak only for a minority of the records. The build stops, as above.
- **The bleed probe is above 0.60 (§3.7).** The silence after the offset carries the label: B's voice leaks into A's channel. The records do not measure "has A finished?" from A's channel alone. The build stops; an amendment tightens the bleed filter, or the line stops there.
- **The listening check, if done, disagrees with the labels.** It gates nothing. Its numbers are reported beside the cross-check, and a large gap between the two is stated as a limit of the labels.
- **The stop rule fires (§6.5).** Twelve minutes of conversation records did not move v2's ranking on conversation validation by 0.01. Either the human-to-human labels cannot be learned by this graph from one channel, or 1.3M parameters cannot absorb them on top of Smart Turn's clips.
  - The line stops at this size.
  - v2 stays the published file, with its fallback timer.
  - For real conversations the recommendation stays Smart Turn's file, with the reading of v2's card.
- **A1 or A2 fails.** v3 bought its gain on conversations with a loss on Smart Turn's clips. It does not pass, and v2 stays the published file. If E1–E3 hold, the report says that v3 is better on real conversations and worse on clips: a trade, not a pass.
- **E1 fails on its minimum effect** (lower bound > 0, but the point estimate is under +0.02). v3 is measurably but not usefully better at ranking real ends of turn. It does not pass, and the report gives the gain as measured.
- **E1 fails on its lower bound.** No improvement on real human-to-agent speech is shown. Test (c) separates the two causes:
  - **(c) improves over v2:** the human-to-human signal does not transfer to agents;
  - **(c) does not improve:** the model did not learn the signal at all.
- **E1 holds and E2 fails.** The ranking improved, but not where a voice agent works (few cut-offs). It does not pass.
- **E3 fails.** The English gain was paid for by the other languages. It does not pass. A follow-up would need conversations in other languages, not a reweighting found on eot-bench.
- **G3 fails.** v3 is not a drop-in replacement. With the same graph this can only be a defect of the export, which is investigated. It does not pass.
- **v3 passes.** It is published as passed, with the sentence of §9. It replaces v2 in the demo only by the owner's decision. Its card's first lines give the eot-bench numbers and the human-to-human limit of §11.
- **The announced goal.** Met: the report says v3 matches or beats Smart Turn on real conversations at a fifteenth of its file. Not met: it says v3 is still below Smart Turn there, with the gap.
- **In every case,** following the owner's decisions of 15/11/2026 and 17/11/2026, the file is published as evaluated, with its Space. Its first lines give the verdict and the numbers.

## 13. Forbidden

- Training, validating, piloting, calibrating, choosing a threshold, cross-checking or listening on:
  - any Smart Turn test clip;
  - any eot-bench audio;
  - any of the 113 conversation test sessions.
- Using DualTurn's features or labels.
- Using Silero VAD, or the cross-check's results, to label, select, drop or weight any training or validation record (§3.7).
- Any v3 checkpoint reading eot-bench before the gate step.
- Reading a v3 test score before §8.1's checks and §8.5's latencies are on file.
- A second seed, a second run, or another start after the pilot's choice, except after a technical failure.
- Changing a bar, a gate term, the cut point or a threshold after a v3 test score is read.
- Attempting to identify, de-anonymise or link a speaker of the corpus.
- Using, reconstructing or inferring the content of a redacted segment.
- Publishing the corpus's audio, snippets or derived records, unless the owner decides it in an amendment.
- Generating any label or example with a generative model. The labels come from the rule of §3.2 applied to a voice-activity detector's output; the cross-check uses a second detector (`AGENTS.md`).

## 14. Amendments

None at the freeze (19/11/2026).

**Amendment 1 (19/11/2026, with the code of §16, before any stage of `scripts/train-turn3.sh` has run; no v3 record, model or score exists).** The readings the code takes where the text leaves a detail open, and the two places where it cannot be implemented as written. No bar, rule, threshold or choice moves. The lead may overrule any point before the first stage.

1. **The caps' draw is made shard by shard** (§3.2 item 8). The eligible records of a split are known only after the last shard, and keeping all of them until then would break the disk bound of §3.8. Each eligible record gets a key, a hash of its id seeded with 3 (`conv/3/<session>:<channel>:<o>`), and a split keeps the records of its `cap` smallest keys: a uniform draw without replacement of `cap` records, the same whatever the order of the shards. A shard writes only the records whose key is then among the smallest; a part that lost a quarter of its records is rewritten, and `fetch-conv --finalize` leaves exactly the draw.
2. **The eot-bench span records start at the span's start − 4.0 s, not − 3.8 s** (§16 against §8.1). §8.1 asks the new span builder, cut at the span's start, to give v2's AUC to ± 0.0005; that cut needs the full 4 s window v2 read, which a record from start − 3.8 s does not hold. Each record holds the 400 hops before the start and the hops up to the largest ask d of 0, 0.2, …, 2.4 s that fits inside the span; the later hops are never read. The span records take at most about 1.0 GB instead of 0.96.
3. **The span rules in samples** (§8.3). "At least 0.2 s long" and "holds longer than 5.0 s" are read on the 16 kHz sample indices of v2's builder: `round(end × 16000) − round(start × 16000)` ≥ 3,200 and > 80,000. The cut of the first ask is v2's cut + 3,200 samples. Decimal times stored as floats then do not decide a borderline span.
4. **A's next onset r** (§3.2 item 4). For a pause that A's speech ends within 2.5 s, the start of the frame of the policy's `resumed`; for a pause the 2.5 s timer ended, the start of the run of speech frames that gave the policy's next `speech-start` (150 ms; a shorter burst is not an onset, for the policy either); none when the channel ends first.
5. **"The utterance of A that ends at o"** (§3.2 item 5, first drop). A's cleaned run (the cleaning of item 3, applied to A's energy VAD) that holds the sample just before o, else A's last cleaned run before o; the backchannel test is item 3's. For a channel's first and last runs, the silence before and after is the one observed, from the session's start and up to its end.
6. **Cases the drops do not name**, each dropped and counted under its own name: t_B and r on the same sample within 4 s (neither "t_B < r" nor "r < t_B" holds: `tie`); a channel whose 95th percentile of frame RMS is 0 (it cannot be gain-normalised: the session, `silent_channel`); a bleed ratio that cannot be computed (no frame where this channel is silent while the other speaks, or none where both are silent: the session fails the bleed filter). When both medians of a ratio are 0, the ratio is 1. Percentiles are nearest-rank; a median of an even count is the mean of the two middle values. The second half of the overlap rule ("began in [o − 1.0 s, o)") is implemented as written; since a floor take lasts at least 1 s, it adds nothing to the first.
7. **The gain-normalised channel is the one read everywhere** (§3.2 item 1): the VAD, the hops, the teacher's tails and the cross-check's WAVs. By the per-window normalisation of the features and the teacher's own normalisation, the gain changes neither beyond rounding, as item 1 says. The tails are kept as 32-bit floats (no clipping); the WAVs too.
8. **The fill of the Smart Turn extension** (§3.4). 2.44 s of fill give 244 new hops; the first 240 are stored, those that end at or before clip end + 2.4 s. "The lowest 10%, at least 20 frames" is `max(⌊0.1 n⌋, 20)` of the clip's n whole 10 ms frames, all of them for a clip shorter than 200 ms. The order is one seeded permutation of those frames after another until the fill is long enough. The cross-fades join the fill's frames only: the clip's own samples, and so v2's hops, are untouched. "The clip's floor is digital silence" reads the pooled RMS of the chosen frames. `turn-extend` also recomputes each clip's last own hop and counts the clips where it is not v2's stored one (a change of decoding would show there).
9. **The WASM parity's 20 conversation test records** (§7). The WASM detector reads PCM only, and the conversation records are features. `fetch-conv` keeps, as 16-bit WAVs on this machine, the 4 s windows at d = 0.2 s of the 20 test records of smallest key, all of which are in the final draw. They are deleted with the data and never published or committed; no label is read with them.
10. **The loss of a batch** (§5): the mean over its rows of each row's loss, L_ST for a Smart Turn row and L_conv for a conversation row, so a row weighs the same whatever its source. The state term of a Smart Turn row is the mean over its masked steps (none at d > 0). Without a conversation store, the trainer's loss is v2's, unchanged.
11. **The draws of §3.6.** The Smart Turn rows continue v2's shuffled epochs (the k-th Smart Turn draw of the run takes the k-th slot of the epoch order); a conversation row draws its class with probability 0.5, then a record of that class uniformly, with replacement. Every draw is a function of the seed, the step and the row, so a resumed run makes the same batches.
12. **The pilot's choice when no from-v2 arm is eligible** (§6.4): P4, if eligible, is taken as the only eligible arm; the stop rule of §6.5 then applies to it as written. The arms' metric and eligibility are the trainer's final validation on f32 weights; v2's metric is its exported file read on the same cuts, as §6.5 says.
13. **The cross-check** (§3.7). Silero's speech probability on whole 512-sample windows only, run as its own wrapper runs it (the previous 64 samples as context, the recurrent state carried, reset per channel). The record rules of §3.2 items 3–5 are applied to Silero's runs by the same code; the session-level drops (the bleed filter, `splits.json`, a silent channel) are the energy path's, so only the sessions those keep are cross-checked. The pinned file's sha256, computed on 19/11/2026 while the code was checked against the tag (size and git blob as §3.1 lists them), is `1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3`; the cross-check records it at its first use and refuses any other.
14. **The listening check's sample** (§3.7, optional), drawn among the eligible validation records, the cross-check's population: per class, the 100 records of smallest key (a hash seeded with 7), collected shard by shard while the shard's WAVs exist (`LABELCHECK=1`).
15. **A measurement on training data, written down with its date** (allowed by the rules at the top). On 19/11/2026, the parser was tested on real bytes: one training session of `shard-000150.tar` (924 s; downloaded into a temporary folder and deleted, no record kept). It gave 567 candidate pauses: 162 dropped because A's utterance was a backchannel, 10 overlaps, 14 with nobody within 4 s, 0 ties, 0 near a redaction, and **365 holds against 16 ends of turn** (4.2%). The gains were 0.715 and 0.748, the bleed ratios 0.97 and 0.42 (both kept), the energy VAD's speech share 0.260 and 0.224 (both speaking 0.018); after the cleaning, channel 1 had 45 floor takes, 61 backchannels and 207 other runs (shorter than 1 s and not isolated). Nothing is changed for it; the lead reads it before the first stage (see the end of the code's report).

**A measurement on training data, 19/11/2026, before the first stage; it changes nothing by itself.** Asked by the lead to size the label balance of point 15 under both detectors. One whole shard, `shard-000000.tar` (sha256 `f9a44412…`, the listing's), was downloaded into a temporary folder; each session was checked against `splits.json` and only its **5 training sessions** (4.4 h) were decoded, its 3 validation sessions skipped unread. Both paths ran with the stages' own code (`sky-data conv-measure`, a separate tool, and `scripts/turn-vadcheck.py` with Silero v6.2.3, sha256 `1a153a22…` as point 13); the shard and every derived file were then deleted, only these counts kept. No session was dropped by the session rules.

| | energy VAD and policy (§3.2) | Silero (§3.7) |
|---|---:|---:|
| candidate pauses | 3,325 | 2,196 |
| dropped: A's utterance a backchannel | 815 | 413 |
| dropped: overlap | 52 | 160 |
| dropped: nobody within 4 s | 78 | 45 |
| dropped: tie | 0 | 2 |
| dropped: near a redaction | 19 | 15 |
| holds | 2,233 | 1,361 |
| ends of turn | 128 | 200 |
| end-of-turn share | **0.054** | **0.128** |

Cleaned speech runs (pooled over the sessions; channel 0 / channel 1): energy 1,732 / 1,979 runs, median 470 / 450 ms, 90th percentile 1,340 / 1,340 ms, floor takes 313 / 350, backchannels 294 / 260, speech share 0.233 / 0.260; Silero 1,115 / 1,273 runs, median 896 / 1,120 ms, 90th percentile 3,296 / 3,392 ms, floor takes 510 / 683, backchannels 224 / 189, speech share 0.345 / 0.400.

§3.7's cross-check on these training sessions (the method of the bar, not its validation reading): of 2,361 eligible energy records, **1,175 have no Silero offset within 150 ms** and 84 are matched to an offset Silero's path drops: **unconfirmed share 0.533** (bar ≤ 0.30). On the 1,102 pairs both label, **agreement 0.962** (Wilson 95% [0.949, 0.972]; bar ≥ 0.85). Energy label × Silero label: hold × hold 1,002, hold × eot 42, eot × hold 0, eot × eot 58. Silero's offsets come later than the energy VAD's: median + 90 ms (5th to 95th percentile + 34 to + 132 ms) over the matched pairs.

Two variants of the energy path, **not adopted**, to size the effect: (i) silences under 300 ms filled (instead of 150 ms) in the runs that decide floor takes and backchannels, the offsets unchanged: 2,137 holds, 155 ends of turn (share 0.068); drops 873 backchannel, 90 overlap, 50 nobody, 20 redaction. (ii) B's floor takes read from Silero's runs, the offsets and A's runs from the energy path: 2,076 holds, 188 ends of turn (share 0.083); drops 815 backchannel, 185 overlap, 41 nobody, 20 redaction.

**Amendment 2 (19/11/2026, the lead's decision on the measurement above; written before the first stage and before any validation or test session is read; defined after a measurement on five training sessions, which every page reporting v3 discloses, §11).** Two changes, nothing else.

*Why.* The second bar of §3.7 assumed that an energy pause with no Silero offset within 150 ms is a suspicious record ("breaths, noise, bleed") whose label is unknown. The measurement shows another cause. The energy detector, the deployed one, whose pauses are the question v3 asks (D1), hears less speech than Silero (speech share 0.233 / 0.260 against 0.345 / 0.400; runs of median 470 / 450 ms against 896 / 1,120 ms), so it asks during speech that Silero hears as continuing: 1,175 of 2,361 eligible energy records had no Silero offset within 150 ms, and Silero's offsets came a median 90 ms later. For such a record the matching by offsets gives no answer, although an independent detector can still answer the label's own question at that instant: who speaks next. The agreement bar (0.85) keeps its value; what changes is that it is computed on every eligible energy record, not on the matched minority (1,102 of 2,361), which is what the second bar was protecting. Ends of turn are rare under both detectors (0.054 energy, 0.128 Silero), so the caps keep them all.

*Change 1, the cross-check of §3.7: the instant label.* For each eligible energy record (channel A, offset o), the label is read from Silero's cleaned speech runs of both channels at the instant o, by §3.2's rule and drops and the same code. The only new definition is A's pause as Silero sees it:

- if a Silero run of A covers o (it starts at or before o and ends after it) or ends within 150 ms before o, the pause starts at that run's end, o_s (o_s > o when the run covers o); otherwise o_s = o;
- r is A's next Silero onset after o_s; t_B is the onset of B's first Silero floor take at or after o (not o_s); the 4 s horizon runs from o;
- the label, the backchannel drop (on A's Silero utterance ending at or holding o), the overlap at o, the tie, nobody within 4 s and the redaction drops are §3.2's.

Two cases the definition leaves open are settled here, the second in the conservative direction:

- **A heard speaking for 1.0 s or more after o** (o_s − o ≥ 1.0 s, a floor take's length of A's own speech, with no 150 ms silence): A holds the floor at that instant: a **hold**, after the same drops. Without this, a record in which A goes on talking for more than 4 s would fall under "nobody within 4 s", although A is the one speaking.
- **B's floor take begins while A is still heard** (o ≤ t_B < o_s): both speak at once, the label is not clean: a **drop, counted as an overlap**.

So A trailing off for 300 ms and B then taking the floor is an end of turn, and A going on for seconds is a hold.

The bars become, both required, else the build stops before any training as §3.7 says:

- **agreement ≥ 0.85**, the share of the eligible energy records whose instant label is not a drop where it equals the energy label;
- **the share of eligible energy records whose instant label is a drop ≤ 0.30** (the old second bar's role: the agreement must speak for at least 70% of the data).

The old statistics (offset matching, matched agreement, unconfirmed share, time differences) stay computed and reported, deciding nothing. Reported, deciding nothing: the 2 × 2 table on all records, the agreement by class, the share of end-of-turn records in which Silero hears A still speaking at o + 0.2 s (the first ask) and the median of A's remaining speech (o_s − o) in them. Unconfirmed or disagreeing records stay in the data with the energy label: Silero selects and relabels nothing.

*Change 2, the caps of §3.2 item 8.* Each cap keeps every eligible end-of-turn record of its split and draws the holds uniformly without replacement (the same seeded keys, seed 3) up to the same total: 120,000 training, 20,000 validation and 30,000 test records. If a split had more ends of turn than its cap, its ends of turn would be drawn the same way up to the cap and it would keep no hold; this is not expected. Sampling was already 0.5 per class (§3.6), so only the number of distinct end-of-turn records changes. Shard by shard (amendment 1, point 1): every eligible end-of-turn record is written; the holds keep the `cap` smallest keys while the shards come, so the store holds at most the cap plus the ends of turn until `fetch-conv --finalize`, which keeps the cap − (ends of turn) smallest hold keys. From the measurement (25.6 ends of turn and 472 eligible records per training session), about 23,000 ends of turn are expected among about 425,000 eligible training records (so 120,000 = about 23,000 + 97,000 holds), and about 2,900 in each of the validation and test splits (about 112 and 113 sessions). The totals do not change, so neither does the disk of §3.8, except about 1.3 GB more while the training records are built. The store's class ratio is no longer the natural one; the natural counts are written to `CONV.json` (`counts.eligible`) and reported.

*A stop rule, written before the measurement that follows.* The amended cross-check is first measured on training data of the same kind: one training shard, its training sessions only. If that measurement is under either amended bar, v3 is not built: it stops here, and that is its result. If it is above both, the stages run, and the bar is read on the validation sessions as §3.7 says.

**A measurement on training data, 19/11/2026, after amendment 2 and its code were committed, before the first stage: the stop rule above.** A shard not read before, `shard-000001.tar` (304,844,800 bytes, sha256 `ba8a5e6b…`, the listing's), in a temporary folder; its **6 training sessions** decoded, its 2 test sessions skipped unread; Silero v6.2.3 (sha256 `1a153a22…`); the shard and every derived file deleted, only these counts kept. No session was dropped by the session rules.

- **The amended cross-check.** 2,479 eligible energy records (2,333 holds, 146 ends of turn). Instant labels dropped: 436 (155 backchannel, 148 overlap, 131 nobody within 4 s, 2 ties): **drop share 0.176** (bar ≤ 0.30). On the 2,043 others, **agreement 0.947** (Wilson 95% [0.937, 0.956]; bar ≥ 0.85). Energy label × instant label: hold × hold 1,834, hold × eot 107, eot × hold 1, eot × eot 101; agreement 0.945 on holds, 0.990 on ends of turn. Silero hears A still speaking at the first ask (o + 0.2 s) in 15 of the 146 end-of-turn records (0.103), for a median 388 ms more.
- **The statistics before amendment 2, reported only.** 1,488 of the 2,479 records had no Silero offset within 150 ms and 102 were matched to an offset Silero's path drops (unconfirmed share 0.641); agreement on the 889 matched pairs 0.957 [0.942, 0.969]; table hold × hold 780, hold × eot 38, eot × hold 0, eot × eot 71; Silero's offsets a median 90 ms later (95th percentile 134 ms).
- **The label counts.** Energy path: 4,011 candidates; dropped 1,360 backchannel, 96 overlap, 72 nobody within 4 s, 4 near a redaction; 2,333 holds, 146 ends of turn (share 0.059). Silero path: 2,071 candidates; dropped 487 backchannel, 161 overlap, 41 nobody, 3 ties, 1 redaction; 1,183 holds, 195 ends of turn (share 0.142). Cleaned runs (channel 0 / 1): energy 1,836 / 1,886, median 440 / 480 ms, 90th percentile 1,480 / 1,350 ms, floor takes 370 / 356, backchannels 436 / 226, speech share 0.231 / 0.234; Silero 1,246 / 1,017, median 864 / 1,184 ms, 90th percentile 3,104 / 4,288 ms, floor takes 559 / 578, backchannels 323 / 164, speech share 0.319 / 0.354. The variants, not adopted: (i) 2,219 holds, 170 ends of turn (0.071); (ii) 2,185 holds, 193 ends of turn (0.081).
- **The stop rule does not fire**: both amended bars hold on this training shard, so the stages run and §3.7's bar is read on the validation sessions.
- **The ends of turn change 2 keeps**, from the two shards measured (274 ends of turn and 4,840 eligible records in 11 training sessions, 24.9 and 440 a session): about 22,400 among about 396,000 eligible training records (so 120,000 = about 22,400 + 97,600 holds), about 2,800 in validation and about 2,800 in test.

**Amendment 3 (19/11/2026, written before any GPU stage and before any test is read; no v3 model or score exists).** Technical fixes to the scripts, found by a review of the code. Each one makes a technical failure stop the build as a technical failure, where before it could become a silent wrong input or a final verdict. **No method, bar, threshold, label, record or choice changes.**

1. **The WASM parity (§7, G3 of §9).** The `wasm` stage of `scripts/train-turn3.sh` discarded the exit status of the parity runs and wrote its marker anyway, and `scripts/turn3-guard.py open` only checked that the parity files existed. A crashed run left an empty file, the tests were opened (once only), and G3 failed on it. Now:
   - `parity-node.mjs --turn` exits 1 both when it crashes and when a complete measurement misses, so its status is kept and reported, and the files decide;
   - each file must hold the tool's result line for the 20 clips or records of §7: `n` ≥ 20, a finite `max_abs_diff` and a `same_decision` count (`scripts/turn3-guard.py parity`);
   - a missing, empty, unparsable or short file stops the stage without its marker, and `open` refuses it before any test is opened;
   - a complete measurement that misses |Δp| ≤ 1e-4 or a decision is a result, not a technical failure: it passes this check, and G3 reads it as failed (§9, §12).
2. **The recipe.** A failure of `scripts/s6-adopt.py env` was hidden inside an `eval`, and the training stages would then have run with empty optimiser options. It now stops the script.
3. **Partial inputs to the gate (§8, §9).** `scripts/turn-gate.py gate3` reported missing clips and cuts but did not refuse them. Outside a smoke run it now writes no verdict and exits 1 ("technical failure: partial input, no verdict written"), with no `GATE.md` or `GATE.json`, when any of these is true:
   - test (a) has fewer than its 31,527 clips (§8.2), or a clip has no v3 probability;
   - test (b) has a number of §8.3 spans other than the count the harness check recorded before any model read them (§8.1 item 3), or a reading is missing: v3 or v2 at an ask inside a span, v3 at d = 0, or Smart Turn at + 0.2 s on a span of §8.3;
   - test (c), when given, has a test record without its v3 or v2 reading at an ask inside it, or without the teacher's opinion (§8.4);
   - a parity file is not a complete measurement (point 1), or v3's or v2's file is missing (G3).
4. **The teacher's recorded opinion (§5).**
   - `scripts/turn-teacher.py --records` wrote a `.done` with 0 records when a shard's tails were missing, so a shard whose tails were lost looked like a shard with no validation or test records, and the stage's error could never fire. It now compares the tails with the validation and test records the shard wrote (`conv-<tag>.json`, `written`: each such record has a tail). Missing or short tails are an error. A `.done` that disagrees with the shard's count is refused, and the tails are kept.
   - A read-only check, `python3 -I scripts/turn3-guard.py teacher --conv <data>/conv`, compares each shard's teacher records with that count. It also lists the store's validation and test records that have no opinion, and exits 3 on any incomplete shard. The `teacher-conv` stage runs it before its marker. The `smoke` stage runs it before any training, so no GPU stage starts on incomplete teacher data.
5. **The rehearsal (`SMOKE=1` only).** The `train` stage of a rehearsal removed the whole run folder, with the rehearsal's markers and gate inputs. It now removes only the trainer's outputs. A real run is untouched: the line runs only under `SMOKE=1`.
6. **The data folders.** The defaults of `TURN3_DATA` and `TURN_DATA` (and `TURN_DATA` of `scripts/train-turn.sh`) were absolute paths on the owner's machine. They are now `../skycmd-data/` beside the main checkout, found through git, so a worktree resolves to the same folder: the same folders as before on the owner's machine. The environment variables still come first, and the folder is printed at start.

**The stages already run.** `conv-data`, `st-extend`, `eotbench-data`, `teacher-conv`, `crosscheck`, `bleed` and `harness3` ran, or are running, with the code of commit `4857ed7`. These fixes change nothing those stages compute. The one place where that code could lose data without saying so (point 4) is checked after the fact: the read-only check reads the teacher's output before `smoke`, and `smoke` runs it again itself. When this amendment was written, the check read the folder as it then stood: 151 shards, 81,681 teacher records for the 81,681 validation and test records written, and the store's 50,000 validation and test records each with an opinion. That reading does not replace the check run before `smoke`.

**A build stopped by one of these refusals is a technical failure.** Its cause is fixed and the stage is run again, as the rules on technical failures allow (§6.7, §13). It is never a result. A refused gate writes no score. If the output of a test-reading stage was partial, its marker is removed and the stage is run again once the cause is fixed: no test score has been read at that point. The report says so.

## 15. Choices (the first value of each line is the one taken at the freeze; the alternative is recorded)

| | choice | proposed | alternatives | why the proposal |
|---|---|---|---|---|
| O1 | corpus | **otoSpeech 280 h, raw (taken: the owner accepted its conditions on 19/11/2026)** | the processed 141 h (half the download, noise-reduced, human-selected, 44.1 kHz); AMI (the fallback, §3.10); CANDOR (licence allows a model, forbids sharing derived data; about 346 GB) | dyads, natural microphone silence, CC BY 4.0, and the DualTurn split covers it |
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
| O18 | false cut-off of E2 | **0.10 (decided by the lead, 17/11/2026); 0.05 and 0.20 reported** | 0.20; 0.05 | cutting users off is a voice agent's main failure; v2 read 0.2230 at τ₈₅ there; the timer covers missed ends |
| O19 | bootstrap | **paired, by turn, 2,000 resamples** | by cut, 1,000 (v2's) | the cuts of one turn are not independent |
| O20 | gate A1 | **≥ 0.8893 (v2 − 0.01), with v2's G2** | v2's bars only (≥ 0.85) | "must not lose what v2 had" |
| O21 | no-harm bar outside English | **lower bound > −0.02** | none; > 0 | the new data is English; the other languages must not pay for it |
| O21b | "gate on all languages" | **decided by the lead (17/11/2026): E1 and E2 on all 14 languages pooled, plus the no-harm bar E3 on the 13 non-English languages pooled; English alone and each language reported** | a bar per language; English gated alone | the new data is English: pooling asks for a gain where users are, and E3 stops it from being paid for by the other languages |
| O22 | minimum effect for E1 | **decided by the lead (17/11/2026): point estimate ≥ +0.02 and 95% lower bound > 0** | lower bound > 0 only (the draft's first proposal), **dropped**: with about 18,000 cuts the interval is narrow, so a gain too small to matter would pass | +0.02 is 37% of v2's pooled gap to Smart Turn at the span's start (0.054): a gain a user could notice |
| O23 | thresholds in the file | **v2's rules on Smart Turn validation; τ_c reported** | τ₈₅ from conversation validation | the same keys and meaning, no engine change |
| O24 | caps | **120,000 / 20,000 / 30,000 records** | every record (about 25 GB more) | disk |
| O25 | label check | **decided by the lead (17/11/2026): an automatic cross-check with a second detector is the bar; the 200-snippet listening check is optional and reported** | the listening check as the bar (the draft's first proposal, at 0.80), **dropped**: it made the run depend on a person's time | DualTurn reports no check; the cross-check runs unattended on every validation session, not on 200 |
| O25a | second detector | **Silero VAD v6.2.3, `silero_vad.onnx` (MIT), probability ≥ 0.5 on 32 ms windows, cleaned 150 / 150 ms** | WebRTC VAD; the energy VAD at other settings | a neural detector shares nothing with the energy VAD, so their agreement measures the label's dependence on the detector; DualTurn's own labels use Silero at the same threshold |
| O25b | agreement bar | **≥ 0.85 (decided by the lead)** | 0.80; 0.90 | under one label in seven depending on the detector |
| O25c | matching tolerance | **150 ms, one to one, the closest first** | 100 ms; 200 ms | the smallest pause either detector keeps after cleaning; under `stopMs` (200 ms), so two pauses cannot share a match |
| O25d | unconfirmed records | **kept, with the energy label; their share ≤ 0.30 is a second bar** | drop them from training; no bar on the share | dropping by Silero would make it a training-label source, which is forbidden; the bar keeps the agreement representative of at least 70% of the data |
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
  - A transient 16 kHz WAV of each validation session's two gain-normalised channels for the cross-check, removed with its shard.
- `conv-crosscheck` (in `src/conv.rs`): reads Silero's speech runs, runs the same label code, matches the offsets and writes `CROSSCHECK.json` (§3.7). Test: identical runs give agreement 1.0 and an unconfirmed share of 0.
  - The disk rule of v2 (`Guard`, exit status 3, resumable).
  - Tests on a synthetic two-channel session with a known answer.
- `src/turn.rs`:
  - `turn-extend`: the 83 shards again; the fill and the 240 extension hops per kept clip into `st-ext/`; `--test` on v1's WAVs.
  - `turn-eotbench --spans`: one record per span from start − 3.8 s to min(end, start + 2.4 s), the 0.2 s and 5 s rules of §8.3. v2's cut mode is kept for §8.1.
  - Tests: the extension equals the tail of the hops of the extended PCM, bit for bit.
- `src/ami.rs`: only if the fallback is chosen (§3.10). `fetch-ami`: the headsets meeting by meeting and the NXT word timings. **Not written: the fallback is not taken (condition 1 at the top).**

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
  - crosscheck, bleed, and labelcheck (optional);
  - harness3;
  - smoke, pilot, train, export, calibrate;
  - latency, wasm;
  - eval, eotbench, conv-test, gate.
- `configs/s6-ear3-nano.json`: R4 of `configs/s6-ear2-nano.json`.
- `scripts/turn-teacher.py --records`: Smart Turn on the PCM tails of conversation records, recorded only.
- `scripts/ear-baseline.py --eotbench --offset-ms 200`: Smart Turn on the cuts of §8.3.
- `scripts/turn-vadcheck.py`: the label cross-check of §3.7.
  - **Runtime.** uv and onnxruntime, with the pinned `silero_vad.onnx` of v6.2.3, whose sha256 is checked.
  - **Input.** The transient 16 kHz validation-channel WAVs that `fetch-conv` writes per shard, removed with the shard.
  - **Output.** The Silero speech runs per channel.
  - **Then.** `sky-data conv-crosscheck` applies the same label code of `src/conv.rs` to those runs, matches the offsets (150 ms, one to one) and writes `CROSSCHECK.json` with the agreement, the unconfirmed share and the two bars.
- `scripts/turn-labelcheck.py` (optional, §3.7): writes the 200 stereo snippets and a blank answer sheet (local only); reads the answers back; gives the agreement with its Wilson interval.
- `scripts/turn-gate.py`, new subcommands:
  - `bleed` (§3.7);
  - `pilot3` (§6.4–6.5);
  - `checkpoint3` (§7);
  - `harness3` (§8.1);
  - `gate3` (§9, with the paired bootstrap by turn and the breakdowns of §8.3).
- `scripts/test_turn_gate.py`: tests of the new subcommands on synthetic probabilities.

**Before the run:**

- **v2's EMA checkpoint.** At `../skycmd-data/s6-ear2/checkpoints/last-ema/` (`config.json`, `model.safetensors`, `optimizer.safetensors`, `state.json`), copied there by the lead from the session worktree. The sha256 of every file is recorded in `runs/s6-ear3-nano/init.json` at the freeze (the list of condition 3 at the top). `--init` refuses a checkpoint whose hashes differ.
- **Licences.** The rows of `DATA_LICENSES.md`, written at the freeze (section "s6-ear v3 audio"):
  - otoSpeech: CC BY 4.0, gated, its role, the conditions accepted;
  - DualTurn's `splits.json`: licence `other`; only session ids used;
  - Silero VAD v6.2.3: MIT; the cross-check only.

**After the result:** `hf/cards/s6-ear3-nano.md`, the result block of this file, and the site's pages as `AGENTS.md` asks.

About 2,000 lines in all.

## 17. Result (19/11/2026): v3 trained once, its tests read once, and the gate failed

**Verdict: FAIL.** A1, A2 and G3 hold; **E1, E2 and E3 fail.** This is `runs/s6-ear3-nano/GATE.md`, read once by the stage `gate` at 23:39:28 on 19/11/2026. The case of §12 that applies is **"E1 fails on its lower bound. No improvement on real human-to-agent speech is shown."** Test (c) improves over v2 (§17.7), so by §12 **"the human-to-human signal does not transfer to agents."** v3 does not pass; v2 (`s6-ear2-nano`) stays the published file.

**The sentence of the report (§9),** as `GATE.md` writes it: "Keep v2 (or take Smart Turn's 8.7 MB file when real conversations matter most): v3 does not pass; on eot-bench at the policy's first question (0.2 s into each pause) its AUC is 0.7197 against v2's 0.7448 (change -0.0252, 95% interval [-0.0336, -0.0167]) and Smart Turn's 0.7873; on Smart Turn's test clips its accuracy is 0.8978 against v2's 0.8993."

### 17.1 What ran, in order

Every stage of `scripts/train-turn3.sh` ran once, with exit status 0, and wrote its marker (`runs/s6-ear3-nano/stages/*.done`, the times below).

- **The data, on the CPU** (the log `runs/all/ear3-data.log`, local; the code of commit `4857ed7`, amendment 3): `conv-data` from 12:18:34 to 15:31:33, `st-extend` to 16:57:33, `eotbench-data` to 17:22:41, `teacher-conv`, `crosscheck` and `bleed` to 17:22:46, `harness3` at 17:26:45. Their counts, the cross-check, the bleed probe and the harness check were committed before the smoke run (commit `d1d6966`; §17.3).
- **The GPU chain** (the console `runs/all/queue-70-ear3-gpu.log`, local): `smoke` from 17:38:40 to 18:05:48 (its numbers not looked at, §6.3), `pilot` to 20:18:31, `train` to 23:22:12, `export` and `calibrate` to 23:23:20, `latency` to 23:23:40, `wasm` to 23:23:41; then the stage `eval` opened the tests once (`runs/s6-ear3-nano/TEST-OPEN`, 23:23:41, with the file's sha256 `048fafa6a7ff427a457c3957a50da0418c64d3c64e55b3b30d6a5b6e77d20cde` and the calibration's) and read test (a) to 23:30:15, `eotbench` test (b) to 23:33:29, `conv-test` test (c) to 23:38:35, and `gate` wrote its verdict at 23:39:28. **No stage was run twice and no run was restarted.**
- **The guards of amendment 3.** The teacher check before the smoke run read 151 shards, 0 incomplete, 81,681 teacher records for the 81,681 validation and test records written, and the store's 50,000 validation and test records each with an opinion. Both parity runs exited 0 (`parity-node.mjs` on the Smart Turn clips and on the conversation records), and `turn3-guard.py parity` found both files complete (`n` 20 each). The gate refused nothing: no partial input and no missing reading (`GATE.json`: 0 clips missing in test (a), 0 cuts missing in test (b)).
- **The listening check (§3.7, optional)** was not done.

### 17.2 The gate (§9; the ternary GGUF as exported)

| term | measured | bar | verdict |
|---|---|---|---|
| A1, accuracy at τ₀.₅ on the 31,527 clips of Smart Turn's test | **0.8978** [0.8946, 0.9013] | ≥ 0.8893 (v2's 0.8993 − 0.01) | holds |
| A2, false cut-off at τ₈₅ = 0.6656 | **0.0881** [0.0836, 0.0926] | ≤ 0.20 | holds |
| A2, END recall at τ₈₅ | **0.8597** [0.8544, 0.8650] | ≥ 0.80 | holds |
| E1, AUC(v3) − AUC(v2), eot-bench at start + 0.2 s, 14 languages pooled | **−0.0252** [−0.0336, −0.0167] (0.7197 against 0.7448) | ≥ +0.02 and 95% lower bound > 0 | **fails** |
| E2, recall at a false cut-off of 0.10, v3 − v2, same cuts | **−0.0563** [−0.0832, −0.0315] (0.2754 against 0.3317) | 95% lower bound > 0 | **fails** |
| E3, AUC(v3) − AUC(v2), the 13 languages other than English pooled | **−0.0209** [−0.0308, −0.0111] (0.7179 against 0.7388) | 95% lower bound > −0.02 | **fails** |
| G3, WASM median per decision | **2.39 ms** (p95 2.41 ms) | ≤ 5.0 ms | holds |
| G3, front end per 10 ms hop (WASM) | **0.0176 ms** | ≤ 0.10 ms | holds |
| G3, GGUF size | **631,936 bytes** | ≤ 3 MB | holds |
| G3, WASM parity, 20 Smart Turn test clips | **max \|Δp\| 5.96e-7, same decision 20 of 20** | ≤ 1e-4, same decision | holds |
| G3, WASM parity, 20 conversation test records | **max \|Δp\| 1.04e-6, same decision 20 of 20** | ≤ 1e-4, same decision | holds |
| G3, the tensor list, shapes and types of v2's file | identical | identical | holds |

Test (a): 95% bootstrap intervals by clip, 1,000 resamples. Test (b): the paired bootstrap of §8.3, 2,000 resamples by turn (all the cuts of a turn together), seed 20261117, percentile intervals.

### 17.3 Before training: the data, the label cross-check, the bleed probe, the harness

Committed before the smoke run (`runs/s6-ear3-nano/data/CONV.json`, `CROSSCHECK.json`, `bleed.json`, `harness3/check.json`; commit `d1d6966`).

- **The data (§3.2, §10).** 1,125 sessions of otoSpeech at `4b3c57b8`, all in `splits.json`; no silent channel; 17 sessions removed by the bleed filter, 1,108 kept; 1,040,448 s decoded. 674,669 candidate pauses, of which 8,931 in the dropped sessions; dropped by rule: A's utterance a backchannel 171,926, an overlap 15,714, nobody within 4 s 19,859, a tie 100, near a redaction 1,769. Eligible records, holds / ends of turn: training 343,858 / 20,536, validation 43,104 / 2,390, test 44,009 / 2,473. Stored by the caps of amendment 2 (every end of turn kept): training 99,464 / 20,536 (120,000), validation 17,610 / 2,390 (20,000), test 27,527 / 2,473 (30,000). The distribution of the largest valid ask of the stored records is in `CONV.json` (`dmax_ms`). The speaker-unseen test subset (§3.3): 2,061 records (1,919 holds, 142 ends of turn).
- **The fill of the Smart Turn extension (§3.4).** Zeros on 13 of the 270,946 training and validation clips and on 3 of the 31,527 test clips; the clip's last own hop differed from v2's stored one on no clip (`runs/all/ear3-data.log`, local).
- **The label cross-check (§3.7 as amended by amendment 2), on the 109 validation sessions the session rules kept.** 45,494 eligible energy records (43,104 holds, 2,390 ends of turn). **Agreement 0.9509** (Wilson 95% [0.9488, 0.9530]; 38,866 of the 40,872 records whose instant label is not a drop; bar ≥ 0.85): **holds**. **Drop share 0.1016** (4,622 records: backchannel 1,545, nobody within 4 s 774, overlap 2,255, tie 48; bar ≤ 0.30): **holds**. Energy label × instant Silero label: hold × hold 37,070, hold × eot 1,940, eot × hold 66, eot × eot 1,796; agreement 0.9503 on holds and 0.9646 on ends of turn. Silero hears A still speaking at the first ask in 312 of the end-of-turn records (0.1305), for a median 298 ms more. The statistics before amendment 2, deciding nothing: agreement 0.9581 [0.9551, 0.9608] on the 19,027 offset pairs both paths label; unconfirmed share 0.5818 (26,467 records); Silero's offsets a median 88 ms after the energy VAD's (5th to 95th percentile 0 to 138 ms).
- **The bleed probe (§3.7).** On the 120,000 training records, the AUC of the mean raw log10-energy of A's hops over [o, o + d] is at most **0.4396** over the twelve asks (at d = 2.4 s; bar: no d above 0.60): **holds**.
- **The harness checks (§8.1), 17:26:45, before any v3 score.** The policy port replays the 3 recorded `turn.ts` vectors bit for bit. v2's file through v3's evaluation path gives 0.8993 on the 31,527 clips (0 missing). The span builder, cut at each span's start, gives v2's 17,934 cuts, v2's AUC 0.7299 and Smart Turn's 0.7838 (both within ± 0.0005). The cuts of §8.3, counted before any model read them: **16,230 cuts** (6,798 `eot`, 9,432 `hold`) from 7,342 turns.

### 17.4 The run (§6, §7)

- **The pilot (§6.4–6.5; `runs/s6-ear3-pilot/choice.json`).** Four arms of 12 working minutes, f32 weights, the metric being the AUC on the 20,000 conversation validation records cut at d = 0.2 s:

  | arm | start | rate | Smart Turn validation accuracy (f32, d = 0) | metric | eligible (≥ 0.88) |
  |---|---|---|---:|---:|---|
  | P1 | v2 | 0.0001936 (× 0.25) | 0.8952 | 0.7413 | yes |
  | **P2** | v2 | 0.0003873 (× 0.5) | 0.8949 | **0.7429** | yes, **chosen** |
  | P3 | v2 | 0.0007745 (× 1) | 0.8939 | 0.7314 | yes |
  | P4 | scratch | 0.0007745 | 0.8291 | 0.7338 | no |

  v2's file on the same 20,000 validation cuts: **0.6175**. The chosen arm is above v2's metric + 0.01, so **the stop rule did not fire**, and the main run started from v2 at 0.0003873 for 120 working minutes.
- **The training** (`runs/s6-ear3-nano/args.json`). 120.0 working minutes, from 20:18:31 to 23:22:12 on 19/11/2026: **155,300 steps by the gate report** (its last logged step; the trainer's log ends at 155,327), batch 128, seed 1, once, no restart, from v2's EMA checkpoint (its sha256 as `runs/s6-ear3-nano/init.json`); half of the rows conversation records and half Smart Turn clips (d = 0 on a quarter of those). Ternary QAT over the last 15 %, from step 131,662; EMA 0.999.
- **The brake (§6.6; `runs/brake.log`, local).** The automatic modes during the GPU chain: duty 0.5 with 45 minutes of work and 15 of rest (on battery) from before the smoke run to 20:16:27, then 0.85 (50 and 10, no input) to 20:18:29; duty 0.5 (the Mac in use) from 20:18:29, when the training started, to 21:26:50; 0.85 to 21:46:59; **0.95 (55 and 5, night) from 21:46:59** to 23:27:43, so through the end of the training, the export, the calibration, the latency and the parity; then 0.5 (in use) to 23:38:15, and 0.85 during the end of test (c) and the gate. Cooling pauses of 900 s and of 300 s (`SKY_COOL`) are recorded in the console. Working minutes count only the work.
- **The checkpoint (§7).** **The EMA** (`exported.json`): its ternary checkpoint loss 0.4572 against 0.4701 for `last`. Its final ternary validation: Smart Turn accuracy 0.8913 at d = 0 (13,593 clips), teacher–student agreement 0.9092; conversation AUC at d = 0.2 s 0.7022 (20,000 records). These four numbers are the trainer's final validation lines (`runs/s6-ear3-nano/train.log`, local); the Smart Turn validation curve is in `GATE.json` (`training.validation`).
- **Calibration (§7; `runs/s6-ear3-nano/calib.json`).** Temperature 1.2522 (validation NLL 0.2721 → 0.2648); **τ₀.₅ = 0.5 and τ₈₅ = 0.6656, both from Smart Turn validation**, where at τ₈₅ the END recall is 0.8501 and the false cut-off 0.0920 (13,593 clips). The conversation operating point **τ_c = 0.6948** (the lowest threshold whose false cut-off on the conversation validation records at d = 0.2 s is ≤ 0.10), reported and not stored.

### 17.5 Test (b): eot-bench at the policy's first question (gated by E1 to E3)

16,230 cuts at span start + 0.2 s, 7,342 turns. Smart Turn v3.2 CPU reads the last 8 s, v2 and v3 the last 4 s.

| cuts | model | AUC | recall at false cut-off 0.10 | at 0.05 | at 0.20 |
|---|---|---:|---:|---:|---:|
| all (16,230) | **v3** | **0.7197** | **0.2754** | 0.1499 | 0.4735 |
| | v2 | 0.7448 | 0.3317 | 0.1840 | 0.5094 |
| | Smart Turn | 0.7873 | 0.3854 | 0.2284 | 0.5818 |
| English (3,902) | **v3** | **0.7375** | 0.3119 | 0.1709 | 0.5155 |
| | v2 | 0.7737 | 0.3469 | 0.2443 | 0.5636 |
| | Smart Turn | 0.8172 | 0.4358 | 0.3073 | 0.6474 |
| the 13 others (12,328) | **v3** | **0.7179** | 0.2681 | 0.1369 | 0.4691 |
| | v2 | 0.7388 | 0.3144 | 0.1771 | 0.5067 |
| | Smart Turn | 0.7852 | 0.3825 | 0.2406 | 0.5892 |

Recall at a false cut-off is a ranking measure (the ROC point), not a deployable threshold.

- **The announced goal (not gated): not met.** v3 is below Smart Turn on the same cuts: AUC 0.7197 against 0.7873, a change of −0.0676 [−0.0763, −0.0590] (the same paired bootstrap).
- **At the file's thresholds** (all cuts): v3 at τ₈₅ cuts off 0.1598 of the `hold` cuts and answers "finished" on 0.4007 of the `eot` cuts; at τ₀.₅, 0.2271 and 0.5237. v2 at its own τ₈₅ (0.7244): 0.2992 and 0.6614; at τ₀.₅: 0.4202 and 0.7810. Smart Turn at 0.5: 0.4422 and 0.8566.
- **By language** (AUC; v3, v2, Smart Turn): ar 0.7219, 0.7251, 0.7806; de 0.6986, 0.7128, 0.8000; en 0.7375, 0.7737, 0.8172; es 0.7495, 0.7724, 0.8005; fr 0.7541, 0.7497, 0.8162; hi 0.6860, 0.6981, 0.7217; id 0.7088, 0.7707, 0.7769; it 0.6813, 0.7146, 0.7741; ja 0.7670, 0.8119, 0.8299; ko 0.7686, 0.7632, 0.8474; nl 0.7328, 0.7214, 0.7829; pt 0.7203, 0.7544, 0.8006; tr 0.6560, 0.6973, 0.7074; zh 0.6946, 0.6692, 0.7493. v3 reads above v2 in French, Korean, Dutch and Chinese, and below it in the ten others, English included: the judgement of §8.3 (English +0.03 to +0.08) did not hold.
- **By cohort** (AUC; v3, v2, Smart Turn): `general` (13,433 cuts) 0.7197, 0.7434, 0.7885; `accented` (1,709) 0.7525, 0.7793, 0.8056; `short_turns` (1,088) 0.7079, 0.7539, 0.8034.
- **The policy reading (reported).** The AUC at each ask d (v3, v2): 0.2 s 0.7197, 0.7448; 0.4 s 0.7100, 0.7369; 0.6 s 0.7103, 0.7383; 0.8 s 0.7115, 0.7398; 1.0 s 0.7107, 0.7361; 1.2 s 0.7176, 0.7270; 1.4 s 0.7203, 0.7292; 1.6 s 0.7164, 0.7423; 1.8 s 0.7329, 0.7444; 2.0 s 0.7549, 0.7508; 2.2 s 0.7690, 0.7954; 2.4 s 0.7920, 0.7773 (from 16,230 cuts at 0.2 s down to 1,620 at 2.4 s).
- **The policy simulated per span (reported).** Of the 6,798 `eot` spans, v3 at τ₈₅ answers "finished" on 0.8232 before the 2.5 s timer (median answer at 400 ms), and 0.2589 of the 9,432 `hold` spans get at least one false cut-off; v3 at τ_c, 0.8029 (400 ms) and 0.2425; v2 at its τ₈₅, 0.8610 (200 ms) and 0.3737.
- **Continuity (reported).** v3 at v2's cut (the span's start, 17,934 cuts): AUC 0.6754, where v2 read 0.7299 and Smart Turn 0.7838.

### 17.6 Test (a): Smart Turn's test clips (gated by A1 and A2)

31,527 clips, d = 0. Accuracy at τ₀.₅ 0.8978 [0.8946, 0.9013] (v2 0.8993); precision 0.8794, recall 0.9211, F1 0.8998; false cut-off 0.1252 at τ₀.₅; at τ₈₅ false cut-off 0.0881 and END recall 0.8597. AUC 0.9586 [0.9566, 0.9606], ECE 0.0068 [0.0053, 0.0102]. The per-language and per-dataset tables are in `GATE.json` (`test_a`).

**Extended by the clips' own background (reported, not gated, §8.2):** at + 0.2 s, v3 0.8959 (AUC 0.9585) and v2 0.8805 (0.9473); at + 0.6 s, v3 0.8935 (0.9572) and v2 0.8512 (0.9220).

### 17.7 Test (c): the 113 held-out conversation sessions (reported, not gated, §8.4)

30,000 test records at d = 0.2 s, session-disjoint and not speaker-disjoint; no interval is computed for (c).

| records | model | AUC | recall at false cut-off 0.10 |
|---|---|---:|---:|
| all (30,000) | **v3** | **0.6878** | 0.2543 |
| | v2 | 0.6031 | 0.1808 |
| | Smart Turn (recorded opinion) | 0.6430 | 0.2050 |
| speaker-unseen (2,061) | **v3** | **0.6663** | 0.1831 |
| | v2 | 0.6137 | 0.2254 |
| | Smart Turn (recorded opinion) | 0.5989 | 0.1901 |

- **Agreement (§10).** Smart Turn's recorded opinion against the conversation labels at 0.5: 0.7013 (speaker-unseen 0.6890); Smart Turn against v3: 0.7033 (0.7045).
- **At later asks** (all records; v3, v2): 0.4 s (19,352 records) 0.6992, 0.6099; 0.8 s (10,111) 0.6807, 0.6002; 1.2 s (5,996) 0.6632, 0.5893; 2.4 s (2,433) 0.5965, 0.5387. Every d is in `GATE.json` (`test_c_conversations`).

### 17.8 Latency and size (G3; §8.5)

`runs/s6-ear3-nano/latency/`, measured at 23:23:40, after the training had ended; Apple M5, one thread, 1,000 decisions on a full 4 s window, features already computed: v3's WASM median 2.39 ms (p95 2.41 ms), native 1.41 ms (p95 1.42 ms), front end 0.0176 ms per hop in WASM and 0.0134 ms natively; v2 in the same session, 2.39 ms in WASM and 1.40 ms natively. GGUF 631,936 bytes, the size of v2's, with v2's tensor list, shapes and types. Parity (`runs/s6-ear3-nano/parity/`): max |Δp| 5.96e-7 on 20 Smart Turn test clips and 1.04e-6 on 20 conversation test records, the same decision on 20 of 20 each.

### 17.9 Disclosure (§11, stated in every report of v3)

**What v3's design read.** v3 was designed after v2's eot-bench reading: the overall, per-language and per-cohort numbers of `runs/s6-ear2-nano/GATE.json` were seen, and that reading motivated making eot-bench the headline, the no-harm bar outside English and the expected per-language effects. eot-bench has never been used for training or for any choice of v2's or v3's training: no eot-bench file was opened for v3's design, and the cut point (0.2 s) comes from the demo's policy and LiveKit's card, not from a reading. The variable trailing silence was motivated by the microphone demo (`turn.ts`), through the code. v3's study measured the trailing silence of 3,160 of v2's training clips. Smart Turn's test is still partly in-sample for the design, as v2's disclosure says; v3 adds no reading of it before the gate. **The label cross-check and the caps were amended after a measurement on five training sessions** (amendment 2, 19/11/2026), before any validation or test session was read: the cross-check reads Silero's label at the energy record's instant instead of matching offsets, and the caps keep every end-of-turn record.

**What it means.** The corpus is human-to-human; eot-bench is human-to-agent: the labels of the training conversations are the listener's behaviour, not an agent's need. The training conversations are English; eot-bench has 14 languages. The conversation test is session-disjoint, not speaker-disjoint. Overlap of speakers or audio between otoSpeech and the two other tests cannot be checked; it is presumed nil (different producers, §3.5).

**The limits of the reading.** One training run, on one seed. Every interval is a bootstrap over test items (clips, or the turns of eot-bench), not over training runs.

### 17.10 What it means, by §12

- **E1 fails on its lower bound** (−0.0252 [−0.0336, −0.0167]; the point estimate itself is below 0, so the case "fails on its minimum effect" does not apply). §12: "No improvement on real human-to-agent speech is shown. Test (c) separates the two causes." On test (c), v3's AUC is 0.6878 against v2's 0.6031 on the 30,000 records, and 0.6663 against 0.6137 on the 2,061 speaker-unseen ones (point estimates: no interval is computed for (c); on the speaker-unseen subset v3's recall at a false cut-off of 0.10 is below v2's, 0.1831 against 0.2254). So the case is **"(c) improves over v2: the human-to-human signal does not transfer to agents."**
- **E2 fails.** §12's case "E1 holds and E2 fails" does not apply, since E1 fails too; E2 is reported as measured.
- **E3 fails.** §12's sentence for E3, "The English gain was paid for by the other languages", presumes an English gain, and there is none: in English v3 reads 0.7375 against v2's 0.7737. Only the verdict of that case applies: it does not pass.
- **A1, A2 and G3 hold.** v3 keeps v2's accuracy on Smart Turn's clips within the bar and is a drop-in file; that does not make it a pass.
- **The announced goal is not met.** v3 is still below Smart Turn on the same eot-bench cuts, by −0.0676 [−0.0763, −0.0590] of AUC.
- **Published as evaluated** (owner's decisions of 15/11/2026 and 17/11/2026, §12): `site/public/models/s6-ear3-nano.gguf` (631,936 bytes, sha256 `048fafa6a7ff427a457c3957a50da0418c64d3c64e55b3b30d6a5b6e77d20cde`, the file `TEST-OPEN` names), with its card and its Space, each opening with the verdict and the numbers. The demo of the site's Listen page keeps v2 (§3.9). No record derived from otoSpeech is committed or published (§13): the per-record files of the run stay on this machine.
