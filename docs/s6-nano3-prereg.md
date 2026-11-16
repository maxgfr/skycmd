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
