# s6-ear v3: design study (end of turn on real conversations)

Written on 17/11/2026, before any v3 code, data or training. No GPU was used, no model was run and no data set was downloaded. Every external fact below was read on 17/11/2026 from the source named with it (dataset cards and file listings through the Hugging Face API, papers, code). Every repository number names its file. One number was measured for this study, on v2's own training features (§2.2); the method is given with it.

**Disclosure.** v3 is designed after v2's eot-bench reading: the overall numbers, the per-language and the per-cohort numbers of `runs/s6-ear2-nano/GATE.json` were seen and are quoted in §1. No eot-bench file was opened for this study, and no eot-bench statistic other than those of v2's result was computed. The variable trailing silence comes from the microphone demo (`site/.vitepress/theme/lib/turn.ts`), not from eot-bench. The pre-registration draft (`docs/s6-ear3-prereg.md` §11) repeats this.

## 0. Summary

1. **The defect, from v2's own result.** On Smart Turn's test clips v2 is right on 0.8993 (teacher 0.9260). On eot-bench, real human-to-agent speech, its AUC is 0.7299 against Smart Turn's 0.7838, and at τ₈₅ it catches 0.5253 of the real ends of turn for a false cut-off of 0.2230. Its teacher is itself weak there, and 82.4% of its training clips are marked synthetic.
2. **A second defect, in the code.** v2 never saw the silences it is asked about.
   - The demo asks the model after 0.2 s of silence, then every 0.2 s, up to 2.4 s; at 2.5 s the fallback timer ends the turn without asking.
   - v2's training clips end at the end of speech. On 3,160 training clips (measured, §2.2), the median trailing silence is 0 ms; only 7.2% (not finished) and 8.9% (finished) have 200 ms or more; 1.9% and 2.7% have 400 ms or more.
   - So the first question the demo asks already lies beyond the 90th percentile of training, and every later one lies beyond the 97th.
3. **The new signal: real two-speaker conversations, one speaker per channel.** A pause of speaker A followed by A speaking again is a `hold`; a pause after which B takes the floor is an end of turn. This labelling costs nothing and is the one of voice-activity-projection work and of DualTurn (arXiv 2603.08216).
4. **The corpus.** `otoearth/otoSpeech-full-duplex-280h`:
   - CC BY 4.0, about 280 h of spontaneous English, 48 kHz stereo FLAC with one speaker per channel;
   - 151 WebDataset shards, 48,757,552,521 bytes;
   - gated: the owner must accept its conditions on Hugging Face.
   - DualTurn's split of its 1,125 sessions is **not speaker-disjoint**: DualTurn's own code says 99.6% of sessions sit in one connected component of shared speakers (§3.3).
5. **The fallback without a gate.** The AMI Meeting Corpus (CC BY 4.0): individual headset recordings and word-level timings, from the Edinburgh site, not the Hugging Face copy, which holds cut segments. It is worse at the task: meetings of 3 to 5 people in one room, role-played, with crosstalk between headsets.
6. **Ruled out by licence:**
   - the oto Speech License of `otoSpeech-full-duplex-turn-104h`, which is also TurnBench's training set;
   - TurnBench's dev set;
   - Seamless Interaction (CC BY-NC 4.0);
   - Switchboard and Fisher (LDC).
   - CANDOR's new licence allows commercial use but forbids sharing derived data; it is recorded for the owner, not chosen.
7. **Odds** (§7). The whole gate of the draft: about 0.45. Beating Smart Turn on eot-bench (announced, not gated): about 0.15.
8. **Cost** (§6).
   - About 92 GB of download, streamed;
   - at most 16.9 GB of new features on disk; 6.1 GB were free when this was written, so space must be freed first;
   - 2 to 4 hours of CPU;
   - 173 working minutes of GPU.

## 1. The defect, measured (from the repository)

### 1.1 What v2 is

- **Model.** `s6-ear2-nano`, rung R4: d 160, ff 640, 4 blocks, 8 hops per 80 ms position, 50 positions over the last 4 s.
  - 1,306,178 parameters; GGUF 631,936 bytes.
  - WASM median 2.39 ms per decision.
  - Source: `docs/s6-ear2-prereg.md`, amendment 1 and the result block; `configs/s6-ear2-nano.json`.
- **Input.** 40 log-mel bands and 3 prosody channels per 10 ms hop, normalised per window (Whisper's rule).
- **Training.**
  - 240 working minutes, 305,736 steps, batch 128, seed 1.
  - The teacher, Smart Turn v3.2 CPU, distilled at three levels: its logit, its pooled embedding and its encoder states.
  - Data: `pipecat-ai/smart-turn-data-v3.2-train`, 257,353 training clips and 13,593 validation clips.
- **Checkpoint and thresholds.** The exported checkpoint is the EMA. Temperature 1.4948; τ₀.₅ = 0.5 and τ₈₅ = 0.7244, both from validation (`runs/s6-ear2-nano/calib.json`).

### 1.2 Its two readings

**Smart Turn v3.2 test, 31,527 clips (gated, passed; `runs/s6-ear2-nano/GATE.md`).**

| reading | v2 | Smart Turn v3.2 CPU |
|---|---:|---:|
| accuracy at τ₀.₅ | 0.8993 [0.8962, 0.9027] | 0.9260 |
| false cut-off at τ₈₅ | 0.0847 | – |
| END recall at τ₈₅ | 0.8627 | – |
| AUC | 0.9600 | – |

**eot-bench, 17,934 cuts (reported, not gated; `GATE.json`, key `eotbench`).**

- **What eot-bench is.** `livekit/eot-bench-data` at commit `ae834af9`: real human-to-agent turns in 14 languages.
- **How v2 was read.** One decision per labelled pause, the audio cut at the pause's start.

| model | AUC | false cut-off | eot recall |
|---|---:|---:|---:|
| v2 (at τ₈₅) | 0.7299 | 0.2230 | 0.5253 |
| Smart Turn v3.2 CPU (at 0.5) | 0.7838 | 0.3678 | 0.7998 |
| timer | 0.5000 | 1.0000 | 1.0000 |

**Per language and per cohort** (AUC; `GATE.json`, `eotbench.by_language` and `by_cohort`):

| | cuts | v2 | Smart Turn |
|---|---:|---:|---:|
| en | 4,049 | 0.7674 | 0.8158 |
| ja | 1,230 | 0.8232 | 0.8356 |
| es | 1,236 | 0.7556 | 0.7996 |
| ko | 841 | 0.7448 | 0.8672 |
| tr | 834 | 0.6493 | 0.7061 |
| zh | 1,099 | 0.6305 | 0.7520 |
| the eight others (ar, de, fr, hi, id, it, nl, pt) | 6,645 | 0.695–0.738 | 0.736–0.810 |
| cohort `general` | 15,135 | 0.7280 | 0.7858 |
| cohort `accented` | 1,711 | 0.7689 | 0.8008 |
| cohort `short_turns` | 1,088 | 0.7597 | 0.8103 |

Three facts follow.

- **v2 trails its teacher in every language.** Its gap is 0.054 overall and 0.049 in English. It is widest in Korean (0.122) and Chinese (0.122), and narrowest in Japanese (0.012).
- **The teacher itself is weak on real conversations.** Its AUC is 0.7838 there against a test accuracy of 0.9260. A student distilled from it is bounded by it there. On the 31,527 test clips the two agree on 0.9181 of clips (`GATE.md`, teacher-student agreement).
- **The training data is mostly synthetic.**
  - 223,343 of the 270,946 rows of the training release are marked `synthetic` (82.4%); 214,273 come from the two chirp3 sets (79.1%).
  - English is 65,802 rows (24.3%).
  - The human sets are small: human_5 3,527 rows, human_convcollector_1 679, liva_1 30,922.
  - Counted for this study from the manifests of v2's feature store, `skycmd-data/s6-ear2/train/manifest-*.jsonl`.

### 1.3 What the demo does with the model (`site/.vitepress/theme/lib/turn.ts`, `TurnDemo.vue`)

- **Audio path.** The microphone is opened with `echoCancellation`, `noiseSuppression` and `autoGainControl` on (`TurnDemo.vue`). It is resampled to 16 kHz by linear interpolation (`makeResampler16k`), cut into 10 ms frames, and every sample is pushed to the detector's stream.
- **VAD.** An energy detector, `EnergyVad`. A frame is speech when its RMS is above max(0.008, 3.5 × the noise floor), and the floor follows quiet frames at a rate of 0.05.
- **Policy** (`TurnPolicy`, defaults in `TURN_DEFAULTS`):
  - Speech counts after 150 ms.
  - After 200 ms of silence following speech, the model is asked once (`stopMs`). While the pause lasts it is asked again every 200 ms (`recheckMs`).
  - At 2,500 ms of silence (`fallbackMs`) the turn is ended by the timer. `feed` checks the timer before it checks whether an ask is due, so no ask happens at 2,500 ms.
  - Speech during the pause is `resumed`, and the window keeps the turn's audio.
  - After `finished` the caller resets the detector, so the next turn's window starts at the reset.
  - The page lets the user change the threshold, `stopMs` and `fallbackMs`.
- **The questions a pause can produce.** At the defaults they come at 0.2, 0.4, …, 2.4 s of VAD silence: at most 12 per pause, each on the last 4 s of the stream.

## 2. The trailing-silence mismatch (from the code and the data)

### 2.1 What the training clips are

- **The feature builder keeps the clip's end.** `crates/sky-data/src/turn.rs`, `shard_features`, computes the hops of each decoded clip with `TurnFrontEnd::hops(&pcm, 400)`: at most the last 400 hops, ending at the clip's last full 40 ms span. No audio is added after the clip.
- **The view keeps it too.** `crates/sky-train/src/turn/data.rs`, `view`, moves only the window's start: the context jitter of `docs/s6-ear2-prereg.md` §3.3, "The window always ends at the clip's end". The time mask is kept out of the last second. No augmentation adds or removes silence at the end.
- **Smart Turn's clips are trimmed at the end of speech.** This was measured on 1,500 test clips in `docs/s6-ear2-study.md` §1.5: median 0 ms, 90th percentile 160 ms (not finished) and 180 ms (finished).
- **v2's eot-bench harness reads the same point.** It cuts at the start of each labelled pause (`cuts` in `turn.rs`), so there too the trailing silence is 0 s.

### 2.2 Measured for this study: the training clips' trailing silence

- **Sample.** The first 40 manifest lines of each of the 83 shards of v2's store (`skycmd-data/s6-ear2/train`), training split only: 3,160 clips.
- **Method.** Count, from the window's end, the hops whose log10 energy channel (channel 42) is more than 4 decades (40 dB) below the window's loudest hop. Hops are 10 ms. The script is pure Python on the f16 store; nothing is trained or evaluated.

| trailing low-energy span | not finished (1,602) | finished (1,558) |
|---|---:|---:|
| median | 0 ms | 0 ms |
| 75th percentile | 20 ms | 30 ms |
| 90th percentile | 170 ms | 190 ms |
| 99th percentile | 740 ms | 860 ms |
| share ≥ 200 ms | 0.0724 | 0.0892 |
| share ≥ 400 ms | 0.0194 | 0.0270 |
| share ≥ 1,000 ms | 0.0019 | 0.0026 |
| share ≥ 2,400 ms | 0.0000 | 0.0006 |

These agree with the test measurement of v2's study (90th percentiles 160 and 180 ms). The criterion is not the demo's VAD, which uses an absolute floor of 0.008 RMS and a ratio to an adaptive noise floor. A speech tail that the energy VAD already counts as silence can still be within 40 dB of the peak, so the acoustic gap at the first ask can be longer than 0.2 s, never shorter.

### 2.3 The mismatch

| ask | trailing silence in the 4 s window | share of the window | share of training clips with at least that much |
|---|---:|---:|---:|
| first (`stopMs`) | 0.2 s | 5% (2.5 of 50 positions) | 7.2% / 8.9% |
| second | 0.4 s | 10% | 1.9% / 2.7% |
| fifth | 1.0 s | 25% | 0.19% / 0.26% |
| twelfth, the last before the timer | 2.4 s | 60% (30 positions); 1.6 s of speech left | 0.00% / 0.06% |

**Fact to correct in the brief.** The trailing silence at a question ranges from 0.2 s to **2.4 s**, not 2.5 s: at 2.5 s the timer ends the turn before any ask (`turn.ts`, `feed`, lines 243–248).

**What follows.**

- At the first ask, v2 reads a window that one training clip in 11 to 14 resembles.
- At the later asks it is outside its training distribution altogether.
- At the last asks it also loses context. The teacher, fed only the last 2 s of test clips, falls from 0.9190 (4 s) to 0.8677 (study of v2, §1.5).

**Two smaller mismatches, recorded.**

- **Resampling.** The demo resamples by linear interpolation. The data builder (`sky_infer::audio::resample`) averages a box over each output step when it downsamples.
- **Noise suppression.** The browser's noise suppression makes the deployed silence close to digital silence. The data's silence is whatever the recording had: café or office noise in many Smart Turn clips, room noise in conversations.

## 3. The corpora

### 3.1 `otoearth/otoSpeech-full-duplex-280h` (the candidate)

Read 17/11/2026 from the dataset page and the Hub API (`/api/datasets/...`, revision `4b3c57b8947d5f651dd6b5e825e7f34b205759e5`, last modified 10/11/2026):

| fact | value | source |
|---|---|---|
| licence | CC BY 4.0 (`license:cc-by-4.0`) | card metadata and text |
| access | gated, automatic approval (`gated: auto`) after the user accepts conditions and shares their Hugging Face information | API, card |
| form fields | name, organisation, work email, background, why, intended use | card metadata `extra_gated_fields` |
| conditions | "Not attempt to identify speakers, de-anonymize participants, or link them to external identities"; "Respect redaction segments and avoid reconstructing or inferring removed content"; follow the licence and the law | card |
| content | "approximately 280 hours of raw English conversational audio", two-party, full-duplex, "recorded with separate channels for each speaker"; spontaneous, topic-driven; overlaps, interruptions and laughter kept; microphone and room conditions vary | card |
| sessions | not stated on the card (size tag 1K–10K); 1,125 in DualTurn's `splits.json` and paper | §3.3 |
| audio | "48kHz stereo FLAC (channel 0 = speaker A, channel 1 = speaker B)" | card |
| format | WebDataset: 151 tar shards, `data/train/shard-000000.tar` …, 127,406,080 to 364,134,400 bytes each, 48,757,552,521 bytes in all; per sample `__key__` (= `session_id`), `flac`, `json` | file listing (API tree), card |
| metadata (`json`) | session id, duration, language, session type (random-chat or product-review), topic, **redacted time ranges** (`redacted_segments`; the audio is muted there), speaker profiles and channel mapping, surveys (audio quality, background noise, echo, microphone quality…), ranks | card |
| annotations | none: no transcript, no word timing, no VAD | card |
| split | one `train` split | card |
| recording | "online conversations" (DualTurn's paper); the card does not say whether speakers were remote | paper, card |

**Who has used it, and for what.** DualTurn pretrains a 0.5B model on it and on Switchboard, and fine-tunes it there for turn-taking (§3.3). Its owner also sells a 1,125 h set under a commercial licence (card notice). The CC BY 4.0 release is the one considered here.

### 3.2 `otoearth/otoSpeech-full-duplex-processed-141h` (recorded, not chosen)

Read from the dataset page and the API (revision `d9876294324aa255ff3586be06a63872a254b796`):

- **Licence and access.** CC BY 4.0; gated with the same form (`gated: auto`).
- **Content.** "approximately 141 hours of processed English conversational audio".
  - Selected high-quality conversations "based on human reviews";
  - "noise reduction and speech enhancement" applied;
  - new samples collected after the 280 h release.
- **Format.** 44.1 kHz, channel-separated FLAC; 61 shards, 19,367,563,544 bytes; size tag under 1K sessions; same per-sample fields. No transcripts or VAD.
- **For v3.**
  - **For it.** It is half the download, and its noise reduction resembles the browser's noise suppression.
  - **Against it.**
    - Its silences are processed, not those of a raw microphone, and other deployments (Pipecat with Silero, telephony) do not suppress noise.
    - It adds sessions that DualTurn's split does not cover.
    - Its selection by quality would narrow the acoustic range.
  - Recorded as the alternative (pre-registration §15).

### 3.3 DualTurn: its label rule and its split

**The dataset.** `anyreach-ai/dualturn-otospeech-turn-taking`, revision `5b700b4b`, not gated.

- **Licence and size.** Licence `other`, with no terms on the card; 71 files, 25,015,573,334 bytes.
- **Content.** Per conversation: Mimi codes and embeddings, cleaned VAD and sparse EOT, HOLD, BOT and BC labels, at 12.5 Hz.
- **Not usable for v3.** It holds no audio, v3 needs the audio for its own front end, and its licence is undefined. v3 neither uses its features nor its labels.

**The label rule, three versions.**

| source | end of turn | hold | backchannel |
|---|---|---|---|
| dataset card | "speech offset where the other speaker takes the floor within 4 s" | "speech offset where the same speaker resumes (no handover)" | "isolated utterance ≤ 1 s with ≥ 1 s silence before and after" |
| paper (arXiv 2603.08216 v1, 09/03/2026, CC BY 4.0) | the same | "All other speech offsets (not EOT)" | the same |
| code (`github.com/anyreachai/dualturn`, MIT; `dualturn/data/relabel_context_aware.py`, `process_otospeech.py`, read 17/11/2026) | the other's first voice activity within 4 s comes before the speaker's own, **and** the other's whole speech segment lasts ≥ 1.0 s | every other offset, including "nobody speaks within 4 s" | as on the card; offsets of a backchannel segment are skipped |

**How the code finds speech.** Silero VAD at 16 kHz, threshold 0.5, minimum speech 100 ms, minimum silence 50 ms, on 20 ms frames. It is then cleaned (bursts under 150 ms removed, gaps under 150 ms filled) and taken to 80 ms frames by majority. The paper says the labels need "no manual annotation". No label-quality check is reported.

**The brief's rule holds, with two precisions.**

- **The floor take is a segment of at least 1 s** (code).
- **An offset that nobody follows within 4 s is a hold in DualTurn.** v3 drops it instead (pre-registration §3.2): such a pause is not one the deployed policy would ask about past 2.4 s, and its label is not clear.

**The splits.**

| source | sessions | split rule |
|---|---|---|
| dataset card, table | 896 / 111 / 113 (1,120); 287.0 h | – |
| `splits.json` (54,992 bytes, read for this study) | 1,125: **900 / 112 / 113** | `"seed": 42`, `"ratios"` 0.8 / 0.1 / 0.1 |
| paper | 1,125 conversations, 289 h at 24 kHz; "a disjoint 113-session test split" | – |
| code, `dualturn/data/create_splits.py` | – | "80/10/10 session-level random split (seed=42). Speaker-disjoint is infeasible (99.6% in one connected component), so we split by conversation." |

**Fact to correct in the brief.** 896 / 111 / 113 is the card's table (probably the sessions that survived processing). The split file itself says 900 / 112 / 113 of 1,125.

**The split is not speaker-disjoint**, and by DualTurn's own count it cannot be made so at a useful size.

- **What v3 does.** It reuses `splits.json` by session, so that its held-out sessions are DualTurn's.
- **What it says.** The conversation test is session-disjoint, not speaker-disjoint.
- **A cleaner subset.** v3 reports the subset of test channels whose speaker id appears in no training session. The metadata has a `speaker_id` per speaker, as DualTurn's code reads it.
- **Why this is no breach.** Using the corpus's own pseudonymous ids to group sessions identifies no one, which the gating conditions forbid.

**Who else labels this way.** Voice Activity Projection (Ekstedt and Skantze, arXiv 2205.09812, Interspeech 2022) models "the dynamics of voice activity of the interlocutors". Its turn-shift and backchannel events are defined on the two speakers' voice activity (abstract re-read; the event definitions are from the paper, not re-read today).

### 3.4 The fallback without a gate: the AMI Meeting Corpus

| fact | value | source (read 17/11/2026) |
|---|---|---|
| licence | CC BY 4.0 for the signals, the transcription and some annotations; both annotation packages changed to CC BY 4.0 | groups.inf.ed.ac.uk/ami/corpus/ and /ami/download/ |
| size | 100 hours of meetings; 171 meetings (third-party count, not verified); 3–5 participants; three rooms; English, "mostly non-native speakers" | AMI pages, HF card, search |
| per-speaker audio | "Individual headsets: four individual WAV headsets", about 120 MB per meeting; 16 kHz 16-bit (Kaldi list, not verified) | download page |
| word timings | manual annotations v1.6.2 (22 MB, NXT format): per-speaker `words` files with "word-level timings that have derived by using a speech recognizer in forced alignment mode" | overview page (as found by search), download page |
| on Hugging Face | `edinburghcstr/ami`, CC BY 4.0, not gated, configs `ihm` and `sdm`: **segmented utterances** of the Kaldi s5 recipe (ihm train 108,502 segments), 16 kHz, 29,011,765,157 bytes | HF card and listing |

- **Where to take it.** The Hugging Face copy is not usable for v3: its segments are cut around speech, so the silence after an offset and the continuous channel are lost. The fallback takes the Edinburgh site's individual headsets: about 171 × 120 MB ≈ 20.5 GB if every meeting is taken, streamed meeting by meeting.
- **What it is worse at:**
  - **Not dyads.** "The other speaker takes the floor" becomes "another participant takes it". Backchannels and overlaps are more frequent.
  - **Role-played.** The scenario meetings are role-played design meetings, mostly by non-native speakers.
  - **One room.** Every headset hears the others (crosstalk), so an energy VAD on a headset fires on other people's speech. The fallback therefore takes speech activity from the forced-aligned word timings, for both the offsets and the floor takes (pre-registration §3.2).
  - **Small.** It has 100 h against 280 h.
- **A second fallback.** The ICSI Meeting Corpus, CC BY 4.0: 75 meetings, about 72 h, 3–10 participants, individual headsets as SPH files of about 350 MB per meeting (groups.inf.ed.ac.uk/ami/icsi/). It has the same weaknesses, and older recordings.

### 3.5 Others, ruled in or out by licence

The project publishes its models under Apache-2.0 and its data sets openly. A research-only or non-commercial licence is out.

| corpus | licence (source, read 17/11/2026) | verdict |
|---|---|---|
| `otoearth/otoSpeech-full-duplex-turn-104h` (420 conversations, 104.94 h, 17 hand-labelled events, among them "Channel Bleed") | **oto Speech License v1.0**: "non-commercial research-use license"; "Production use and other commercial uses require a separate written license"; restricts derived data and model artifacts (card) | **out** |
| TurnBench (Sesame and others, arXiv 2608.25218): training set = the 104 h set above; dev `mundo-ai/turn-benchmark-dev`; test with withheld labels | training set: oto Speech License; dev: Dataset Public License v1.0, "commercial use is not permitted"; code MIT (GitHub README, oto blog) | **out** |
| Seamless Interaction (Meta, about 4,000 h of dyadic conversation) | CC BY-NC 4.0 (HF card `facebook/seamless-interaction`) | **out** |
| Switchboard, Fisher | LDC licences (DualTurn's README: Switchboard "requires LDC license") | **out** |
| CANDOR (BetterUp; 1,656 conversations, about 850–860 h; one stereo MP3 per conversation, one participant per channel) | CANDOR Corpus License: commercial use now allowed. The card says version 2.1 (effective 03/11/2026) and the blog of 12/11/2026 says v1.1. Data and derivatives may not be redistributed. A trained model may be shared with a notice "trained using the CANDOR Corpus from BetterUp Labs". Gating is reviewed by hand. The processed release is about 346 GB in zips of 1.5–11 GB that include video | **not chosen**: allowed for a model, but its derived clips could never be published, its audio is lossy, and its download is about 7 times otoSpeech's. Left to the owner |
| AMI, ICSI | CC BY 4.0 | **in** as fallbacks (§3.4) |

## 4. The design, checked against the code

**1. Same graph, same file.**

- v3 trains the R4 graph of `configs/s6-ear2-nano.json` (`sky_train::turn`), exports it with the same keys and is read by the same `TurnDetector`.
- The front end (`TurnFrontEnd`, `FeatureStream`) and the model code (`sky_infer::turn`) do not change.
- **Holds.** The changes are confined to `sky-data` (new builders), `sky-train` (loader, an `--init` option, the loss mask) and scripts.

**2. Labels from two channels.**

- The deployed model hears one microphone, so it is given A's channel only; B's channel decides the label.
- **Holds.** The store keeps one channel per record, and the label is computed in the builder.

**3. Clips cut where the policy would ask.** The policy and its energy VAD are ported to Rust, bit for bit against `turn.ts` on recorded test vectors, and replayed on A's channel.

- **Which offsets count.** Every point at which `TurnPolicy` would call `probEnd()` is a candidate cut: offset + d, d ∈ {0.2, 0.4, …, 2.4} s, with d inside the pause.
- **Why the policy and not a fixed list.** Offsets are then defined exactly as the deployed VAD defines them, with its noise floor, its 150 ms minimum speech and its frame grid.
- **What the label-dependence of d means.** Holds have short pauses and ends of turn have long ones. This is true of deployment as well: an ask at 1.2 s happens only if the pause has lasted 1.2 s.
- **The distribution of d.** Half of the draws at 0.2 s, the first ask, which sets the agent's latency and is the headline cut. The other half spread evenly over 0.4–2.4 s. A d longer than the record's pause is redrawn.

**4. The same silences on Smart Turn's clips.**

- **Why a new pass over the shards.** v2's store holds only the last 400 hops of each clip. Appending silence in feature space would get the boundary hops wrong, since a hop spans 40 ms. The shards were deleted after v2.
- **What the pass computes.** The training release is streamed again shard by shard. For each kept clip, the builder computes the 240 hops that follow its end when the PCM is extended by 2.44 s of the clip's own background.
- **What the background is.** Its quietest 10 ms frames, put together in a seeded random order with 2.5 ms cross-fades; zeros if the clip's floor is digital silence.
- **Exact, not approximate.** A hop exists once its 40 ms span has arrived and the grid starts at the first sample (`turn_hop_count`, `TurnFrontEnd::hops`). The hops of the extended clip are therefore those of the clip followed by the new ones, exactly.

**5. Bleed.**

- **Why it matters.** If B's voice leaks into A's channel, a cut after B's onset tells the model the answer.
- **A session filter.** The bleed ratio of a channel is the median RMS of its VAD-silent frames while the other channel speaks, over its median RMS while both are silent. A session goes if either channel's ratio is above 2.0 (+6 dB).
- **Why +6 dB.** It is below the demo VAD's 3.5× ratio (+10.9 dB), so bleed that passes the filter does not trigger the VAD either.
- **Supporting evidence.** The oto turn-104h set has a "Channel Bleed" label, and the 280 h card's surveys ask about echo.

**6. Redactions.** The audio is muted in `redacted_segments`. A muted span would look like a pause and must not be reconstructed. Every record whose window or label horizon comes within 0.5 s of a redaction is dropped.

## 5. Risks

- **Domain: human-to-human against human-to-agent.**
  - In eot-bench a person talks to an agent. Turns are task-like, the user may pause to think, and the "listener" never backchannels or jumps in.
  - In otoSpeech two people chat. B may take the floor at a hesitation that A meant as a hold (labelled end of turn), or let a finished turn lapse so that A goes on (labelled hold).
  - The labels encode the listener's behaviour as much as the speaker's completion. How much transfers is the experiment's open question.
- **English only.**
  - The new data is English; eot-bench has 14 languages and English is 4,049 of v2's 17,934 cuts (22.6%).
  - What should carry over to every language is the silence fix: the variable trailing silence, which does not depend on the language.
  - What should help English first is the lexical and prosodic knowledge of real speech.
  - Half of the training draws become English conversation, so the other 13 languages may lose. The pre-registration therefore gates a no-harm bar on them pooled.
- **Label noise of VAD-derived labels.** Three sources:
  - the energy VAD (breaths, laughter and noise counted as speech; soft speech missed);
  - the rule itself (the listener's choice, above);
  - bleed.
  - **How it is estimated.** A hand check before training: 200 offsets of validation sessions (100 labelled end of turn, 100 hold, drawn with a seed), each played as both channels from 6 s before to 4 s after the offset, labelled by a person as "A finished", "A had not finished" or "cannot tell". The agreement is reported with its Wilson interval.
  - **Its consequence.** An agreement under 0.80 stops v3 before training for an amendment of the rule (pre-registration §3.7).
  - **No model labels.** The labels come from a rule applied to the corpus's own audio; no model writes them. Smart Turn's opinion on the conversation clips is recorded, never trained on.
- **The gate's conditions of use.**
  - No attempt to identify speakers: v3 uses only the pseudonymous ids for the split and never listens for identity.
  - Redactions are dropped (§4).
  - The licence is CC BY 4.0, so the model card credits the corpus.
  - The clips and snippets stay on the machine. Whether derived features may be published is the owner's decision; the draft says no.
- **Session-disjoint, not speaker-disjoint** (§3.3). The conversation test is optimistic for that reason, and is reported, not gated.
- **Disk.**
  - Free space was 6,057,404 KB (6.1 GB) on 17/11/2026 (`df -k`), not the 28 GB of the brief.
  - v3 needs at most 16.9 GB of new data plus the project's 8 GB floor (§6).
  - Space must be freed first. The largest candidate is v1's `data/ear/audio` (20 GB), which v2's pre-registration keeps untouched "unless the owner decides otherwise".
- **v2's checkpoint.** It exists only in a session worktree (`.claude/worktrees/s6-tranches-2-7/runs/s6-ear2-nano/last-ema`, `model.safetensors` 5.7 MB, not in git). It must be copied somewhere durable before that worktree is removed, or the from-v2 start is impossible.
- **Gate (a) may move.** Half the draws become conversations, and three quarters of the Smart Turn draws get a trailing silence. The trimmed test clips may lose a point. Starting from v2's weights and keeping a quarter of v2's view mitigate this.

## 6. Cost (one MacBook, Apple M5, 16 GB)

**Downloads**, each streamed and deleted as in v2:

| item | bytes |
|---|---:|
| otoSpeech 280 h, 151 shards (at most 364 MB on disk at a time) | 48,757,552,521 |
| Smart Turn v3.2 train, 83 shards again, for the trailing extension | 41,400,926,042 |
| eot-bench `all` again, for the new cuts (v2 kept its features, not its audio) | 1,980,396,015 |
| **total** | **about 92.1 GB** |

At v2's estimate for 41.4 GB (15–70 min), this takes 35 minutes to 2.6 hours.

**Disk.** A hop is 43 float16 values, 86 bytes.

| new item | size |
|---|---:|
| Smart Turn trailing extension: 270,946 clips × 240 hops | 5.59 GB |
| Smart Turn test extension (reported reading): 31,527 × 240 | 0.65 GB |
| conversation training records: at most 120,000 × 620 hops (from 3.8 s before the offset to 2.4 s after it) | ≤ 6.40 GB |
| conversation validation: at most 20,000 records | ≤ 1.07 GB |
| conversation test: at most 30,000 records | ≤ 1.60 GB |
| eot-bench: one record per span, from 3.8 s before its start to min(end, start + 2.4 s), 17,934 spans at most | ≤ 0.96 GB |
| one shard in transit | ≤ 0.60 GB |
| **peak, new** | **≤ 16.9 GB** |

- **Kept from v2 and reused:** its training store (11 GB), its test store (1.0 GB) and v1's test WAVs.
- **Free space needed at the start:** 16.9 + 8 (the floor of `MIN_FREE_GB`) ≈ **25 GB**.
- **The caps are a disk choice.** The number of usable offsets is not known before the build. It is estimated at 250,000–500,000 for 280 h, at an assumed 45% speech per channel and 1.5–2 s per stretch of speech; not measured.

**CPU** (front end 0.013 ms per hop natively, from `hf/cards/s6-ear2-nano.md`; teacher about 36 ms per clip per core, from v2's study):

| step | estimate |
|---|---|
| FLAC decoding of 280 h of 48 kHz stereo | 10–30 min with 4 threads (not measured) |
| VAD and policy replay on 560 channel-hours | a few minutes |
| conversation features, at most 170,000 × 620 hops | at most 23 min on one thread, about 6 min on 4 |
| Smart Turn extension, 65M hops | about 14 min on one thread, 4 on 4 |
| teacher on the conversation validation and test records at d = 0.2 s (recorded, not a target), about 50,000 clips | about 8 min with 4 processes |
| Smart Turn on the new eot-bench cuts | about 3 min |
| **total** | **2–4 hours of wall clock, mostly download** |

**Human time.** The hand check: 200 offsets of about 10 s each, about 1–1.5 hours of listening.

**GPU** (working minutes; v2 did 1,274 steps a minute at batch 128):

| run | minutes |
|---|---:|
| smoke | 5 |
| pilot, 4 arms × 12 | 48 |
| main training, from v2's weights | 120 |
| **total** | **173** |

About 6 hours of wall clock in the brake modes. Evaluation runs on the CPU.

## 7. Odds (judgement, from the numbers above)

**Why v3 should gain on eot-bench at the first ask.**

- At 0.2 s of silence v2 reads windows that 7–9% of its training clips resemble (§2.3). v3 trains on exactly those windows, in both sources.
- In English it also gets about 280 h of real spontaneous conversation, where v2's training clips are 82.4% synthetic.
- **What argues the other way.**
  - The labels are those of human listeners, not of agents.
  - Only the silence fix helps the 13 other languages.
  - The student keeps its 1.3M parameters, the size at which v2 already trails its teacher by 0.054.

| outcome | probability |
|---|---:|
| the stop rule fires (pilot no better than v2 on conversation validation) | 0.15 |
| gate (a), Smart Turn test bars, all hold | 0.75 |
| E1: eot-bench ΔAUC over v2, pooled, lower bound > 0 | 0.75 |
| E2: Δ eot recall at false cut-off 0.10, lower bound > 0 | 0.65 |
| E3: no harm on the 13 other languages pooled (lower bound > −0.02) | 0.85 |
| G3: latency and size (same graph) | 0.98 |
| **the whole gate** | **about 0.45** |
| English ΔAUC ≥ +0.03 (reported) | 0.55 |
| v3's pooled AUC ≥ Smart Turn's on the same cuts (announced goal) | 0.15 |
| the same, English alone | 0.30 |

## 8. Not verified

- **otoSpeech.**
  - The number of sessions in the 280 h release: the card gives no count. 1,125 comes from DualTurn's split file and paper.
  - Whether speakers were remote. The paper says "online conversations"; the card does not say.
  - The level of channel bleed and echo.
  - The exact fields of the `json`. DualTurn's code reads `speakers[].speaker_id`; the card describes speaker profiles and a channel mapping.
  - Whether every session of `splits.json` is in the 151 shards.
  - The gating text beyond what the page shows to an unauthenticated reader; the full terms may say more.
- **DualTurn.**
  - Why the card's table (896 / 111 / 113, 287.0 h) differs from `splits.json` (900 / 112 / 113) and from the paper (289 h).
  - The meaning of its licence `other`.
  - Its label quality: none is reported.
- **AMI.**
  - The meeting count (171, third-party).
  - The 16 kHz rate of the headset files (a Kaldi list message).
  - The attribute names of the word timings in the NXT files.
  - The crosstalk level.
  - The standard partition by meeting that the fallback would use.
  - None of its files was opened.
- **CANDOR.** Its licence version (2.1 on the card, v1.1 in the blog). Its full text was not read.
- **Offsets and labels.** The number of usable offsets, the end-of-turn share, the share of sessions the bleed filter keeps, and the share of offsets near redactions: all unknown before the build.
- **Speed and cost.**
  - The decoding speed of `claxon` on 48 kHz stereo FLAC.
  - The download bandwidth.
  - The throughput of the new loader. v2's 1,274 steps a minute assume the same reading cost.
- **Behaviour.**
  - How v2 itself behaves at 0.2 s and later asks. It has never been read at those points, and its eot-bench reading cut at the pause's start.
  - Whether eot-bench's span starts, which come from LiveKit's word timings, match the energy VAD's offsets.
  - The per-language effect of English-only conversation data.
  - The VAP event definitions, from the paper but not re-read today.
- **Overlap between sources.** Overlap between otoSpeech and Smart Turn's or eot-bench's speakers or audio. It is presumed nil (different producers) but cannot be checked.
- **The estimates.** Every odds figure in §7 is a judgement, not a result.

## 9. Sources (read 17/11/2026)

- otoSpeech 280 h: https://huggingface.co/datasets/otoearth/otoSpeech-full-duplex-280h ; API `https://huggingface.co/api/datasets/otoearth/otoSpeech-full-duplex-280h` and its `tree/main`
- otoSpeech processed 141 h: https://huggingface.co/datasets/otoearth/otoSpeech-full-duplex-processed-141h
- otoSpeech turn 104 h and its licence summary: https://huggingface.co/datasets/otoearth/otoSpeech-full-duplex-turn-104h ; https://huggingface.co/blog/otoearth/turnbench-training-set
- DualTurn data set, card and `splits.json`: https://huggingface.co/datasets/anyreach-ai/dualturn-otospeech-turn-taking
- DualTurn paper (v1, 09/03/2026): https://arxiv.org/abs/2603.08216 and https://arxiv.org/html/2603.08216
- DualTurn code (MIT): https://github.com/anyreachai/dualturn (`dualturn/data/create_splits.py`, `process_otospeech.py`, `relabel_context_aware.py`, `add_start_end_labels.py`)
- Voice Activity Projection (abstract): https://arxiv.org/abs/2205.09812
- AMI: https://groups.inf.ed.ac.uk/ami/corpus/ ; https://groups.inf.ed.ac.uk/ami/download/ ; https://huggingface.co/datasets/edinburghcstr/ami
- ICSI: https://groups.inf.ed.ac.uk/ami/icsi/download/ ; https://groups.inf.ed.ac.uk/ami/icsi/license.shtml
- TurnBench: https://github.com/SesameAILabs/turnbench ; https://arxiv.org/pdf/2608.25218 ; https://huggingface.co/datasets/mundo-ai/turn-benchmark-dev
- Seamless Interaction: https://huggingface.co/datasets/facebook/seamless-interaction ; https://arxiv.org/abs/2506.22554
- CANDOR: https://www.betterup.com/blog/candor-conversations-between-strangers (12/11/2026) ; https://huggingface.co/datasets/betterup/candor-processed
- eot-bench card (cohorts; the 0.2 s score point of the `general` rows): https://huggingface.co/datasets/livekit/eot-bench-data
- Repository files read: `docs/s6-ear2-study.md`, `docs/s6-ear2-prereg.md` (amendment 1 and the result block), `runs/s6-ear2-nano/GATE.md` and `GATE.json`, `runs/s6-ear2-nano/calib.json`, `hf/cards/s6-ear2-nano.md`, `site/ear/index.md`, `site/.vitepress/theme/lib/turn.ts`, `site/.vitepress/theme/components/TurnDemo.vue`, `configs/s6-ear2-nano.json`, `scripts/train-turn.sh`, `crates/sky-data/src/turn.rs`, `crates/sky-data/src/audio.rs`, `crates/sky-data/Cargo.toml`, `crates/sky-infer/src/audio.rs`, `crates/sky-infer/src/turn/store.rs`, `crates/sky-train/src/turn/data.rs`, `crates/sky-train/src/turn/train.rs`, `TODO.md` ("What remains", item 5), `AGENTS.md`.
