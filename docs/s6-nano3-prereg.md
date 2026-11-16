# s6-nano v3 (s6-route: the v2 router fused with a 1 MB static embedding): pre-registration

**Status: frozen on 16/11/2026, before any fresh slate of §4 is built or scored.** No model is trained and no GPU is used. v3 is designed after v2's test was read (`docs/s6-nano2-prereg.md`, result block of 16/11/2026) and after an exploratory reading of v2's test dumps: every place that reports v3 says so. Its gate is read on slates that no system has scored.

## 1. Why, and what the question is

v2 (`s6-route-nano`, 2,418,912 bytes) failed its gate: on tools never seen in training it reached 0.8395, far above the static embedding it was gated against (potion-cos, 0.6675) and level with BM25 (0.8500), the lexical score it is handed; and it needs 38.9 ms per request in WASM, against a bar of 2 ms. It reads the lexical side of a request well and the meaning badly: outside its training families a static embedding beats it (CLINC150: 0.3980 against 0.7740).

The exploratory reading (`scripts/s6-route-explore.py`, after the gate, nothing fitted) multiplied the model's probabilities by those of `potion-cos-4bit`, the same static embedding packed to 4 bits (1,003,952 bytes), option by option. On v2's test of unseen tools the pair reached 0.8885, above BM25 (0.8500) and above all-MiniLM-L6-v2 (0.8840, 91.6 MB), where BM25 with the same embedding reached 0.8475. On the six other sources the pair was at or above the better of its two parts. On the validation slates, which the exploratory reading had not used, the same pair gives 0.9135 (BM25 0.8549, MiniLM 0.9126, the model alone 0.8339, the embedding alone 0.7229).

A number found by looking at a test is a hypothesis. **The question of v3: on fresh requests, does the pair of v2's file and the 1 MB embedding route better than BM25, as well as a 91 MB sentence encoder within 0.03, and never worse than the better of its two parts?**

## 2. The system `s6-route` (everything below is fixed before §4 is built)

- **The model part:** the published file of v2, unchanged (sha256 in `site/public/models/manifest.json`), at full depth with its stored temperature, asked as in v2: the request, the BM25 line of amendment 1 of v2, the options `name: description` and *none*.
- **The embedding part:** `potion-cos-4bit` exactly as fitted for v2 on validation (`scripts/route-baselines.py`): `minishlab/potion-base-2M` at v2's pinned commit, packed to 4 bits, cosine between the request and each `name: description`, *none* at θ = 0.396120, probabilities = softmax([cosines, θ] / 0.0671021).
- **The fusion:** for each option *i* of a slate, `p(i) ∝ (p_model(i) · p_embedding(i))^β`, renormalised over the options. **β = 0.81**, the value of the grid 0.01, 0.02, …, 4.00 that minimises the NLL on the 1,144 validation slates (0.2608; at β = 0.5, the plain geometric mean of the exploratory reading, 0.3105). β changes no answer: it only sharpens the probabilities.
- **The ask threshold:** τ_ask3 = **0.671047**, by v2's rule on the fused probabilities of the validation slates (the smallest top probability at which the accuracy among answered requests is ≥ 0.95): there, 0.9508 among answered, coverage 0.8881.
- **Size:** 2,418,912 + 1,003,952 = **3,422,864 bytes**.
- **A fast mode, reported:** the same fusion with the model stopped at its first exit (2 of 6 layers; validation 0.9056 against 0.9135).

Validation numbers of the system (v_tools, 1,144 slates; none of this is a test score; `scripts/s6-route3-fit.py`, `runs/s6-nano3/fit.json`): accuracy 0.9135, ECE 0.0254, accuracy at coverage 0.7 / 0.8 / 0.9: 0.986 / 0.975 / 0.947. On the same slates BM25 fused with the same embedding gives 0.8698.

What was chosen where: the partner (`potion-cos-4bit`, the smallest of the two static embeddings) was first seen on v2's test in the exploratory reading and then read on validation; the depth, β and τ_ask3 were read on validation only. Nothing of v2's test enters a parameter.

## 3. What does not change

The model file, the embedding and its two fitted values, the slate rule of v2 §4.4 and its readings (`docs/s6-nano2-readings.md`), the scorer, the baselines and their fitted values (`potion-cos`, `potion-cos-4bit`, `static-ft`, `pair-lr`, `bm25`, `minilm-cos`, `toolrouter`, `name-prior`; fitted on validation at B = 2,418,912 bytes), the metrics of v2 §7.2.

## 4. Fresh test files (built once, hashed before any system reads them)

Built by the builder of v2 with a new seed (1235) and the rule of each v2 file, from requests that are in **no** slate file of v2 (training, validation or test, compared by request identifier and by exact text):

| file | from | slates |
|---|---|---|
| **T2-tools** | the requests of v2's test clusters that no v2 slate uses (those over the cluster cap, and those left out of the 2,000), at most 100 per cluster; the distractors by v2's rule among the test-side tools; 20% *removed* | as many as exist, counted before scoring |
| T2-metatool | MetaTool single-tool queries not in `t_metatool` | 2,000 |
| T2-clinc | CLINC150 queries not in `t_clinc`, in scope and out of scope as v2 | 2,000 |
| T2-massive | MASSIVE held-out-scenario utterances, en and fr, not in `t_massive` | at most 2,000 |

- The tools of T2-tools are those of v2's test side: never trained on, by v2's split. The requests are new; the tools are not new to this project's readers, since v2's test showed how each system does on them. This is stated wherever v3 is reported.
- The builder reads no output of any system. The files' sha256 go to `runs/s6-nano3/test-files.json`, and the counts (slates, clusters, the share of the three largest clusters) are written in an amendment, before any scoring.
- No test request is sent to the teacher: the false-negative filter of v2 is not run on T2 (v2's test was built the same way).

## 5. Gate (every condition must hold)

| | condition | why |
|---|---|---|
| **G1** fresh requests, unseen tools | on T2-tools: the system's accuracy minus BM25's is above 0 **with the lower bound of its 95% paired bootstrap interval above 0** (1,000 resamples of the slates); and the system ≥ `minilm-cos` − **0.03**; and the system is above the model alone and above `potion-cos-4bit` alone | BM25 costs no byte and v2 only tied it: the pair must beat it beyond noise. MiniLM is what a practitioner with 91 MB to spend would use: 27 times fewer bytes may cost 0.03, not more. A fusion that does not beat its parts is not one |
| **G2** never much worse than its better part | on each of T2-metatool, T2-clinc and T2-massive: the system ≥ max(model alone, `potion-cos-4bit` alone) − **0.02** | outside the model's training families the embedding carries the answer; the pair must not lose it |
| **G3** calibration | ECE ≤ **0.05** on T2-tools (15 bins) | as v2 |
| **G4** it runs where it would be used | the engine (`sky-infer`, `sky-web`) computes the fused probabilities in WASM: against the Python reference on 200 T2-tools slates, max \|Δp\| ≤ 1e-3 and the same answer on at least 199; WASM SIMD128 median ≤ **60 ms** per request with the tool list cached (Apple M5, Node); the two files together ≤ 4 MB | v2's bar of 2 ms was written for routing at each keystroke, and v2 measured 38.9 ms: this graph cannot meet it. v3 routes once per request, where 60 ms stays under the 100 ms an interface treats as immediate. The bar is relaxed after v2's measurement, and every report says so |

**Announced goals, not gates:** T2-tools accuracy ≥ 0.87; ≥ 0.95 among answered at τ_ask3 with coverage ≥ 0.80 on T2-tools; the fast mode within 0.015 of full depth on T2-tools.

## 6. Reported, not gated

Every system of §3 on every T2 file, with v2's metrics; the fast mode; the system on T2-tools without the three largest clusters, and a bootstrap that resamples clusters instead of slates; BM25 fused with the same embedding (to show what the model adds over the score it reads); the names-masked reading on T2-tools, **where the system is expected to fail** (v2's model is at chance when every tool is renamed `tool_<k>`: 0.1050 on v2's test, and the pair was at 0.2990 there); latency of each part.

## 7. Disclosure (written into every place that reports v3)

v3 was designed after v2's test was read, from an exploratory reading of v2's test dumps. Its parameters come from validation. Its gate is read on requests no system had scored; on T2-tools these requests are about the same held-out tools as v2's test. The latency bar is 30 times v2's and was set after v2 was measured. The system needs distinctive tool names.

## 8. What a result would mean

- **Pass:** the pair is published as `s6-route`, "passed on fresh slates; system defined after v2's test was read", with a playground on the site (a tool list, a request, the fused probabilities, *none* and "ask the user" at τ_ask3) and both files on Hugging Face.
- **G1 fails:** what the exploratory reading saw does not hold on fresh requests. The recommendation for run-time tool lists stays BM25 (no byte) or MiniLM (91 MB), and no further version of this router is run.
- **G2 or G3 fails alone:** published as evaluated with the verdict in its first lines; no playground.
- **G4 fails alone:** the routing result is reported as held, the engine work as not done; no playground until it holds.
- **In every case** the numbers are published as read (owner's decision of 15/11/2026).

## 9. Forbidden

- Building T2 twice, or with another seed, cap or rule after a count is seen, except to fix a technical failure of the builder before any scoring (reported).
- Scoring any system on T2 before the hashes and the count amendment are committed.
- Changing β, τ_ask3, θ, a temperature, the partner or the depth after T2 is scored.
- Using v2's test slates in any parameter or in any T2 file.

## 10. Code to write before the reading

1. `sky-data route` (or its Python twin): the fresh files of §4 from v2's staged requests, with the exclusion by identifier and text, the cap, the seed; unit tests of the exclusion and of determinism.
2. `sky_infer` and `sky-web`: the 4-bit static embedding (tokeniser of `potion-base-2M`, table lookup, mean, cosine, θ, temperature) and the fusion; a fixture written by the Python reference that both must reproduce; the router mode of `crates/sky-web/bench/parity-node.mjs` extended to the pair.
3. `scripts/route-baselines.py` and `scripts/route-eval.sh`: a `fresh` split that reads T2 and refuses it until the hashes are on file; `scripts/s6-route3-gate.py` and its tests: the fusion from the dumps, the gate of §5, the readings of §6.
4. The site playground, behind the gate as §8 says.

## 11. Amendments

**Erratum (16/11/2026, before any fresh slate is built or scored).** §1 says "on the six other sources the pair was at or above the better of its two parts". The exploratory script read **five** other sources (T-metatool, T-clinc, BFCL, When2Call, MASSIVE held-out), and the sentence holds on those five; the tool-routing-data test (100 queries) was not in it. Nothing else changes.

**Amendment 1 (16/11/2026, after the fresh files were built, hashed and counted and the scoring chain was rehearsed on validation; before any system has read a fresh file).**

- **The fresh files** (`runs/s6-nano3/counts.json`, hashes in `runs/s6-nano3/test-files.json`; two builds give the same bytes; the builder first recomputes v2's split and refuses to go on unless it reproduces v2's ten test files byte for byte, which it does):

  | file | slates | *none* gold |
  |---|---:|---:|
  | T2-tools (`f_tools`, and its names-masked copy) | **945** | 189 |
  | T2-metatool (`f_metatool`) | 2,000 | 400 |
  | T2-clinc (`f_clinc`: 1,600 in scope, 400 out of scope) | 2,000 | 400 |
  | T2-massive (`f_massive`: 1,000 en, 1,000 fr) | 2,000 | 400 |

  T2-tools: 2,541 requests over v2's cluster cap and 320 left out of v2's 2,000 were candidates; 13 were excluded as already in a v2 slate file and 1,903 by the cap of 100 per cluster (the three largest clusters held 1,135, 559 and 509 fresh requests). The 945 slates come from **273 clusters**; 629 from over-cap requests and 316 from left-out ones; the three largest clusters hold **31.7%** of them (100 each); by source: glaive 484, seal-tools 175, toolace 138, api-bank 129, hermes 19. 107 slates repeat the normalised request of an earlier slate (kept, as v2 kept them). With 945 slates instead of v2's 2,000 the interval of G1 is wider: this makes G1 harder to pass, and nothing is changed for it.
- **Readings the code had to take, accepted as built:**
  1. A request is excluded when its identifier, its exact text or its normalised text (v2's normalisation) is in a v2 slate file: the training and validation records, the validation slates and the ten test files. The normalised-text rule alone removed 1 tools, 1 CLINC, 5 MetaTool and 5 MASSIVE requests.
  2. The cap of 100 per cluster is applied after the exclusion, in v2's cap order; every request kept is used, 20% *removed*, seed 1235; the distractors come from v2's 3,317 test-side tools with v2's filter.
  3. T2-metatool and T2-clinc use v2's draw rules with seed 1235; CLINC queries come from its test split, as v2's.
  4. **T2-massive is drawn from the train and validation partitions of the four held-out scenarios** (7,303 utterances that v2 never used for anything), because v2's test took their whole test partition; at most 1,000 per language.
  5. Only the names-masked copy of T2-tools is built; no gold-removed pairs and no swapped descriptions (v3 gates neither).
  6. G1's bootstrap is v2's (1,000 resamples of the slates, seed 0, percentile interval), and its lower bound must be strictly above 0. G3 uses v2's scorer. In G2 a missing dump fails its file. The fast-mode goal is one-sided (full depth minus fast mode ≤ 0.015).
  7. G4 holds when the bench ran on at least 200 slates with the same answer on all but at most one, max |Δp| ≤ 1e-3, a cached median ≤ 60,000 µs, and the two files together ≤ 4,000,000 bytes. While the bench's output is absent G4 is "not measured" and the gate is not passed.
  8. The reported cluster readings count clusters by fresh slates; the cluster bootstrap resamples clusters with replacement, seed 0. BM25 fused with the embedding uses β = 0.5. The masked reading fuses the masked dumps of the two parts.
  9. The gate script checks the frozen values (β 0.81, τ_ask3 0.671047, θ 0.396120, T 0.0671021) and the model file's size; the file's sha256 (`a124dfe8…6c82`, the published one) is written into the gate's output.
- **The order of the reading.** The routing conditions (G1, G2, G3) are read first, once, from the dumps of the Python reference and of `sky-eval`. G4 is measured when the engine's embedding and fusion are written and checked on validation; nothing in the engine reads a fresh score, and its parity check on fresh slates compares it with dumps that already exist. Until then the gate reads "not passed (G4 not measured)", whatever the routing conditions say.
- **`runs/s6-nano3/FRESH-OPEN`** is created by the run's operator under the owner's standing instruction to run every step without asking, as v2's `TEST-OPEN` was.
- **What does not change:** the system, its four frozen values, the gate, §9.

**Amendment 2 (16/11/2026, after the engine's embedding and fusion were written and checked on the validation slates; before any system has read a fresh file). The engine is ready, so the gate is read in one pass, G4 included.**

- **The embedding file** the engine reads is `skycmd-s6-route-embed.gguf`: **1,273,312 bytes** (sha256 `c53fb76dc8a56a696d859e80b7383d7e9db168e54c024525f220851fbacda67a`): the packed table of §2 (1,003,952 bytes), the vocabulary and the normaliser of the tokenizer of `potion-base-2M`, θ, the temperature, β and τ_ask3. With the model file the system is **3,692,224 bytes**, not the 3,422,864 of §2, which counted the table alone. G4's size condition (≤ 4 MB, read as 4,000,000 bytes) is read on the two files as shipped.
- **Checked on validation only** (v_tools, 1,144 slates; `crates/sky-infer/src/embed.rs`, `crates/sky-web/src/router.rs`): the engine's fused probabilities against the Python reference: max |Δp| 2.6e-7 natively and 1.7e-6 in WASM, the same answer on 1,144 of 1,144, accuracy 0.9135; the embedding part alone against the `potion-cos-4bit` dump: 3.3e-7, 0.7229; the token identifiers equal model2vec's on 14,930 validation texts (French included). WASM median with the tool list cached, on a machine in use: 33.3 ms (SIMD128), of which the embedding part is 0.02 ms; the fast mode 13.4 ms.
- **One boundary, stated before the reading:** the validation slate whose top probability defines τ_ask3 comes out 4.7e-9 below it in the engine's f32 arithmetic, so the engine asks on that slate (validation coverage 0.8872 against 0.8881). No answer changes. The announced goal at τ_ask3 is read on the Python reference, as the gate's other numbers.
- **How G4 is measured:** `crates/sky-web/bench/route3-node.mjs` on the first 200 slates of T2-tools, against the fused reference computed from the fresh dumps of the model (`sky-eval`) and of `potion-cos-4bit` (`scripts/s6-route3-reference.py`), SIMD128 build; the relaxed build and the fast mode are reported. The machine is in use by its owner and a GPU run of another model may be active: the latency can only be read higher than on an idle machine.
- **What does not change:** everything else.

**Result (16/11/2026, after the fresh files were read once). The gate passed, every condition. Published as `s6-route`: "passed on fresh slates; system defined after v2's test was read".**

- **Disclosure** (§7). v3 was designed after v2's test was read, from an exploratory reading of v2's test dumps. Its parameters come from validation. Its gate is read on requests no system had scored; on T2-tools these requests are about the same held-out tools as v2's test. The latency bar is 30 times v2's and was set after v2 was measured. The system needs distinctive tool names.
- **What was run, in which order** (`scripts/route3-read.sh`, committed in `5457c34` before the reading; its log `runs/all/route3-read.log`). `runs/s6-nano3/FRESH-OPEN` was created at 11:26:44 by the run's operator, under the owner's standing instruction (amendment 1); the log ends at 11:33:28. In one pass, as amendment 2 orders: the eight baselines of §3 with v2's fitted values on the five fresh files (T2-tools, its names-masked copy, T2-metatool, T2-clinc, T2-massive); the model (v2's file, a copy with its exit thresholds at 1) on the same files at full depth and at its first exit, with its native latency on 200 T2-tools slates; the fused reference of T2-tools from the two parts' dumps (`scripts/s6-route3-reference.py`, full depth and fast mode); the engine's WASM bench against that reference on the first 200 T2-tools slates (`crates/sky-web/bench/route3-node.mjs`: SIMD128, relaxed SIMD, fast mode); then the gate (`scripts/s6-route3-gate.py`). No system was scored on a fresh file before this pass, and nothing was run twice. Results: `runs/s6-nano3/eval-fresh/results.md` and `gate.json`, latency in `runs/s6-nano3/eval-fresh/latency/`. The gate's output records the model file's sha256 (`a124dfe8…6c82`, the published one) and the four frozen values.
- **The gate** (`gate.json`):

  | | condition | measured | verdict |
  |---|---|---|---|
  | G1 | T2-tools: system − BM25 > 0, lower bound of the 95% paired bootstrap > 0 | 0.8868 − 0.8169 = +0.0698 [0.0455, 0.0952] | holds |
  | G1 | system ≥ `minilm-cos` − 0.03 | 0.8868 against 0.9048 − 0.03 = 0.8748 | holds |
  | G1 | system > the model alone; system > `potion-cos-4bit` alone | 0.8868 against 0.7926; against 0.7386 | holds |
  | G2 | T2-metatool: system ≥ max(model, `potion-cos-4bit`) − 0.02 | 0.5175 against 0.5140 (max 0.5340, the embedding) | holds |
  | G2 | T2-clinc, same | 0.8040 against 0.7655 (max 0.7855) | holds |
  | G2 | T2-massive, same | 0.5340 against 0.4555 (max 0.4755) | holds |
  | G3 | ECE ≤ 0.05 on T2-tools (15 bins) | 0.0162 | holds |
  | G4 | WASM against the Python reference on 200 T2-tools slates: max \|Δp\| ≤ 1e-3, the same answer on ≥ 199 | 1.6e-6; 200 of 200 | holds |
  | G4 | WASM SIMD128 median ≤ 60 ms per request, tool list cached | 30,879 µs | holds |
  | G4 | the two files ≤ 4,000,000 bytes | 2,418,912 + 1,273,312 = 3,692,224 | holds |

  **Every condition holds: the gate is passed.** On T2-metatool the system is below the embedding alone (0.5175 against 0.5340), within the 0.02 that G2 allows. The header line of the generated `results.md` gives the system as 3,422,864 bytes, the number of §2 (the table alone); the files are 3,692,224 bytes (amendment 2), and that is what G4 read.
- **Paired bootstrap on T2-tools** (1,000 resamples of the slates, seed 0, 95%): system minus BM25 +0.0698 [0.0455, 0.0952]; minus MiniLM −0.0180 [−0.0381, +0.0032]; minus the model alone +0.0942 [0.0720, 0.1185]; minus `potion-cos-4bit` +0.1481 [0.1228, 0.1714]; minus BM25 fused with the same embedding +0.0243 [0.0085, 0.0413]; minus the fast mode +0.0032 [−0.0063, +0.0127].
- **The announced goals** (not gates): T2-tools ≥ 0.87: met (0.8868). ≥ 0.95 among answered requests at τ_ask3 with coverage ≥ 0.80: **not met** (0.9433 at coverage 0.8593). The fast mode within 0.015 of full depth: met (0.8836 against 0.8868).
- **The readings of §6 (reported, not gated), said as plainly as the pass.**
  - **The three largest clusters.** They hold 300 of the 945 T2-tools slates. Without them (645 slates) the system is at 0.8806 against BM25 0.8605: **+0.0202 [−0.0062, +0.0481], an interval that holds zero** (MiniLM 0.9070; the system minus MiniLM −0.0264 [−0.0543, +0.0016]). The bootstrap that resamples the 273 clusters instead of the slates gives, on all slates, **+0.0698 [−0.0079, +0.1404] against BM25, also holding zero** (against MiniLM −0.0180 [−0.0661, +0.0194]; against the model alone +0.0942 [0.0543, 0.1325]; against `potion-cos-4bit` +0.1481 [0.1168, 0.2016]). The advantage over BM25 that G1 read leans on the three largest clusters; the advantages over the two parts hold under both readings.
  - **Names masked** (every tool renamed `tool_<k>`, the 945 slates): the system is at **0.2497**, where BM25 keeps 0.8138, `potion-cos-4bit` 0.7280, BM25 fused with the embedding 0.8423 and MiniLM 0.9005; the model alone is at 0.0720 and the fast mode at 0.2899. As §6 announced, the system needs distinctive tool names.
  - **MiniLM remains above** the system on every fresh file (T2-tools 0.9048, T2-metatool 0.6570, T2-clinc 0.8790, T2-massive 0.5730).
  - **Outside the tool lists of its training families** the system is close to its embedding part: on T2-metatool 0.5175 against 0.5340 for the embedding alone, below BM25 fused with the embedding (0.5420), toolrouter (0.5910) and MiniLM; on T2-clinc 0.8040 against 0.7855; on T2-massive 0.5340 against 0.4755, below toolrouter (0.5900) and MiniLM (0.5730).
  - **What the model adds over the score it reads:** BM25 fused with the same embedding (β = 0.5) is at 0.8624 on T2-tools; the system is +0.0243 [0.0085, 0.0413] above it.
- **Every system on every fresh file** (accuracy; every metric of v2 §7.2 is in `results.md`):

  | system | T2-tools (945) | T2-tools, names masked (945) | T2-metatool (2,000) | T2-clinc (2,000) | T2-massive (2,000) |
  |---|---:|---:|---:|---:|---:|
  | **s6-route** (the system) | **0.8868** | 0.2497 | 0.5175 | 0.8040 | 0.5340 |
  | s6-route, fast mode | 0.8836 | 0.2899 | 0.5130 | 0.8020 | 0.5300 |
  | the model alone (v2's file, full depth) | 0.7926 | 0.0720 | 0.3190 | 0.4240 | 0.3680 |
  | the model alone, first exit | 0.8000 | 0.0762 | 0.3140 | 0.4135 | 0.3690 |
  | `potion-cos-4bit` (the embedding part alone) | 0.7386 | 0.7280 | 0.5340 | 0.7855 | 0.4755 |
  | `potion-cos` | 0.7386 | 0.7249 | 0.5350 | 0.7800 | 0.4730 |
  | `static-ft` | 0.7450 | 0.7365 | 0.5265 | 0.7870 | 0.4795 |
  | `pair-lr` | 0.3153 | 0.4106 | 0.1605 | 0.1550 | 0.0975 |
  | BM25 | 0.8169 | 0.8138 | 0.4330 | 0.4990 | 0.3205 |
  | BM25 fused with `potion-cos-4bit` | 0.8624 | 0.8423 | 0.5420 | 0.7750 | 0.4515 |
  | `minilm-cos` (all-MiniLM-L6-v2, 91.6 MB) | 0.9048 | 0.9005 | 0.6570 | 0.8790 | 0.5730 |
  | `toolrouter` | 0.7735 | 0.7513 | 0.5910 | 0.7470 | 0.5900 |
  | `name-prior` | 0.2000 | 0.1238 | 0.2000 | 0.1565 | 0.0000 |

  On T2-tools the system's *none* F1 is 0.7732 (BM25 0.6368, MiniLM 0.8010) and its NLL 0.3767 (MiniLM 0.3253). Risk and coverage of the system on T2-tools (accuracy among the most confident slates at coverage 0.50 / 0.70 / 0.90): 0.996 / 0.977 / 0.929.
- **Latency and size** (`runs/s6-nano3/eval-fresh/latency/`; Apple M5, Node, the first 200 T2-tools slates; the machine was in use by its owner, with a GPU run of another model active, so these can only read higher than on an idle machine):

  | | SIMD128 | relaxed SIMD | fast mode (SIMD128) |
  |---|---:|---:|---:|
  | median per request, tool list cached | **30,879 µs** | 23,536 µs | 10,547 µs |
  | first request (tool list encoded) | 45,279 µs | 34,316 µs | 25,023 µs |
  | of which the embedding part, cached | 21 µs | 20 µs | 16 µs |
  | max \|p_wasm − p_python\|, same answer | 1.6e-6, 200 / 200 | 1.0e-6, 200 / 200 | 9.0e-7, 200 / 200 |

  The model part natively (`sky-eval`, the same 200): median 24,648 µs, p95 45,598 µs. The Python baselines in process CPU time per slate (whole file, loading included): BM25 0.16 ms, `potion-cos-4bit` 0.75 ms, MiniLM 12.25 ms. Files: `s6-route-nano.gguf` 2,418,912 bytes (v2's file, unchanged) and `s6-route-embed.gguf` 1,273,312 bytes (sha256 `c53fb76d…a67a`), 3,692,224 bytes together, about 25 times fewer than MiniLM's 91.6 MB.
- **What it means, by §8.** The first case applies: the pair is published as **`s6-route`**, "passed on fresh slates; system defined after v2's test was read", with a playground on the site (a tool list, a request, the fused probabilities, *none* and "ask the user" at τ_ask3) and both files on Hugging Face (prepared, uploaded by the owner). The pass is what G1 to G4 say and no more: on fresh requests about v2's held-out tools the pair beats BM25 on the slate bootstrap, stays within 0.03 of MiniLM and beats each of its parts; the advantage over BM25 does not survive the cluster bootstrap nor the removal of the three largest clusters; it fails when tool names are not distinctive; MiniLM is more accurate on every file. v2's verdict does not change: `s6-route-nano` alone stays a failed gate.
