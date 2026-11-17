# s6-nano v2 (s6-route-nano, a tool router that reads descriptions): pre-registration

**Status: frozen on 15/11/2026, before the first line of the builder of §4 is written or run.** No data, code or run of this version exists. The choices of §13 are taken: the first value of each line, which is the one used in the text. Two things are fixed later by rules written here, each before any test file is read: the head window (§3) and the size of the model (the pilot rule of §5). From the freeze on, the results are read through this text.

**Why this experiment.** s6-nano passed its gate and is published for the games, but as a decision model it does not work: on 17 of the 20 bench tasks it gives one answer to at least 95% of the items, and its bench accuracy equals the per-question prior of its training data (`docs/s6-nano2-study.md`). A first narrow redesign (the v2 router draft, not run) kept fixed-label sources in its gates, where a size-matched linear model is near 0.95, and its author put its odds at 10 to 20%. The owner asked for a version that works for something, even if imperfect, without spending GPU hours on poor odds. v2 takes the one job where options arrive at run time, judges it against what a practitioner would use there (a static sentence embedding with cosine), and stops early if a pilot shows the design cannot get there. The design study, `docs/s6-nano2-router-study.md`, is committed with this file and holds every external number quoted here.

**Changes to this file:**

- every amendment is committed, dated and justified in §15;
- an amendment comes before the run or the measurement it concerns;
- an amendment never moves a threshold after a test score has been read;
- measurements on training and validation splits (cluster counts, truncation, teacher agreement, the baseline rehearsal, the pilots) are allowed before the main run, dated and written down; they never touch a test file.

Background: `docs/s6-nano2-router-study.md`; `docs/s6-nano2-study.md`; `docs/research-s6-usefulness.md` §3, §8 (item 5) and §9; the result of `docs/s6-secrets2-prereg.md` (a narrow job with a structured input, hard negatives and its own data beat a size-matched logistic regression and a 17M model).

## 1. The question

Given a user request and a list of up to 7 tools, each a name and one sentence, plus the option *none of these tools fits*, does the s6 decision graph, trained on tool and intent routing only:

1. pick the right option on tools that no training item contains, better than potion-base-2M with cosine similarity and better than every trained model of at most its byte size (G1)?
2. give answers that depend on the options, not a constant (G2)?
3. stay calibrated, so that a threshold can send uncertain requests to the user (G3)?
4. answer within 2 ms in the browser with the tool list cached (G4)?

A pass means: a sub-megabyte file that a web page can use to route requests to tools it describes in text, with a probability it can trust. A failure on G1 means a static embedding is the better tool for this job at this size (§11).

## 2. What changes from v1 and from an earlier draft that was never run, and why

| | v1 (s6-nano) | v2 draft | v2 | why (`docs/s6-nano2-study.md`, `docs/s6-nano2-router-study.md`) |
|---|---|---|---|---|
| job | games + any typed decision | tools, When2Call, intents (fixed and run-time labels) | **one**: pick a tool from a run-time list, or *none* | a general model at this size ties the per-question prior (nano2 study §3.1) |
| options | the same few strings for every item of a task | fixed for MASSIVE, When2Call, Banking77 | **different in every item**; the gold changes with the option set | a constant option set makes a per-option bias the cheapest solution; v1 was not constant only where options varied (MASSIVE en, 0.4255) |
| counterfactual slates | none | none | each request in 4 slates: gold present, hard distractors, **gold removed → *none***, gold renamed and re-described | only the options differ between them, so only reading the options lowers the loss (nano3 study §3) |
| hard negatives | none | random distractors | nearest tools and co-listed tools, **cleaned by the teacher** of distractors that also serve the request | hard negatives teach discrimination; false negatives teach noise (nano3 study §3, item 2) |
| prior per label | learned (bench, When2Call) | When2Call reweighted | *none* gold in 20% of slates in every split; at most 30 requests per tool cluster | the prior cannot score above 0.20 |
| targets | 0.5 gold + 0.5 teacher (agreement 68.8% on When2Call) | gold | **gold**, one-hot | the teacher cleans option sets and rewrites descriptions; it is never a soft target |
| data mix | 29 sets + games; targets 0.25–1.4% of draws | routing 70%, general 30% | **routing 100%** | exposure (nano2 study §2.3) |
| test | bench, MASSIVE, When2Call, NOTAM (read) | a fresh tool test + read sources | **fresh, tool-disjoint**: no test tool cluster in training | measures reading descriptions, not memorising labels |
| bar | majority class | size-matched linear models per source | **potion-base-2M cosine** (7.56 MB) and size-matched trained models (static fine-tuned, pair LR) | what a practitioner uses for run-time option lists (nano3 study §2.4) |
| size | nano | nano | nano unless a pilot shows micro is worth 3.4× the bytes (§5) | capacity helps sample efficiency (nano2 study §2.4) |
| pilot | rate pilot | none | **two 20-minute pilots with an abort rule** | do not spend the main budget on a design the pilot already shows short |

Unchanged: the graph (`runs/recipe/configs/s6-nano.json`, 1,486,225 parameters as trained), tokenizer `data/tok/tok-s6-2048.json`, AdamW, rate `LR_SCALE` 2.0 × √(96/144) × the nano pilot's ×1 (decision peak 1.63e-3), batch 128, QAT after 10% of the budget, loop dropout 0.25, `--length-pool 16 --pad-to 32 --step-mem-gb 3 --no-compile`, checkpoint `last`, `s2-calibrate` on validation, ternary export. For micro, the same with its own config, batch 64 and its rate (×0.5 of its centre, `docs/s6-gen-prereg.md` amendment 12).

## 3. The model

- **Prompt.** Type `choice`. Instruction, word for word: "Which tool should handle the user's request?" State: the request, at most the last user turn, prefixed `User: `. Options: `name: description` for each tool (the description cut to its first sentence and at most 30 words), then `none of these tools fits the request`, always last.
- **Initialisation.** The decision model is initialised from the pretraining of the chosen size (`runs/s6-nano/pretrain/init`, or `runs/s6-micro/pretrain/init`), copied into `runs/s6-nano2/` and compared by sha256 before and after. No pretraining is run.
- **Head window rule** (decided on training and validation records, before the pilots). `sky-train s2-truncation` on the built records: the head window is the smallest of 160, 256, 320 tokens at which at most 5% of the records have an option shrunk below its own length. RoPE has no position table, so the parameter count does not change (as amendment 2 of `docs/s6-gen-prereg.md`). The state window stays 512 and is not binding (requests are short; truncation is reported).
- **Thresholds stored in the file.** Exit thresholds 1 (full depth) with `gguf-set`; the ask-threshold τ_ask (the smallest top probability at which validation accuracy among answered requests is ≥ 0.95) under `skycmd-s2.route.tau_ask`.
- **Size.** Nano: 715,136 bytes expected (v1's file); micro: 2,420,032. The exact size `B` of the evaluated file is the byte budget of every size-matched baseline (§6).

## 4. Data (`sky-data route`, to be written)

### 4.1 Sources and roles

| group | sources | licence | role | requests (estimate) |
|---|---|---|---|---:|
| `route/tools` | glaive-function-calling-v2, hermes-function-calling-v1 (deduplicated against glaive), ToolACE, Seal-Tools, API-Bank training data | Apache-2.0, Apache-2.0, Apache-2.0, Apache-2.0 (GitHub), MIT (hub) | train / val / **T-tools** (tool-disjoint) | ≈ 35,000 after the cap |
| `route/when2call` | `data/decisions/when2call.jsonl` train and val: tool_call and request_for_info → the called tool; cannot_answer and direct_answer → *none* | CC-BY-4.0 | train / val | ≈ 15,000 |
| `route/intents` | MASSIVE en + fr train partition, 14 scenarios (4 held out, §4.3); Banking77 train; each intent a tool with one teacher description | CC-BY-4.0 | train / val | ≈ 25,000 |
| test only | MetaTool (`plugin_des.json`, single-tool queries); CLINC150 `plus` test; BFCL v2 `multiple` and `irrelevance`; tool-routing-data `tools/test`; When2Call `test/mcq` reformatted; MASSIVE test partition of the 4 held-out scenarios, en and fr | MIT, CC-BY-3.0, Apache-2.0, CC-BY-4.0 tag, CC-BY-4.0, CC-BY-4.0 | §7 | — |

Weights (one directory per group; `s2-decide` draws a group by weight, then an item uniformly): `tools=0.60, intents=0.25, when2call=0.15`.

### 4.2 Tool clusters and splits (the guard of the unseen-tool claim)

1. Every tool of `route/tools` gets a normalised name (lowercase; leading `get`, `fetch`, `retrieve`, `find`, `search`, `lookup`, `query` dropped; separators collapsed) and an embedding of `name: description` with all-MiniLM-L6-v2 (pinned commit; a clustering tool here, never a compared system on this split).
2. Union-find: two tools are one cluster if their normalised names are equal or their description cosine is ≥ 0.85.
3. Each cluster goes to train, val or test by the first byte of sha256(smallest normalised name of the cluster + `1234`): 85 / 5 / 10.
4. A request goes to the split of its gold tool's cluster. Its slates use only tools of that split. A request whose normalised text equals one of another split goes to the test side, and the copy is dropped elsewhere.
5. Per cluster, at most 30 requests in training (the first by sha256 of the text); the test keeps at most 10 per cluster so that no family dominates it.
6. The counts (clusters, tools, requests per split, the largest clusters) are reported; the builder asserts that no test cluster has a training tool.

### 4.3 Intents

- MASSIVE: the 4 scenarios with the lowest sha256(scenario + `1234`) are held out from training in both languages; the other 14 are trained (train partition only; validation partition for validation). The held-out scenarios' test utterances are a reported probe. The derived sources of v1 (`calibrated:mteb/amazon_massive_intent`, `tasksource:multilingual/massive`) are not used.
- Banking77: train split only; never a test (its official test rows are in our training split, `DATA_LICENSES.md`).
- CLINC150 and HWU64 are never trained. Every training request equal (normalised) to a CLINC150 test utterance is dropped.
- Descriptions: one per intent, written by the teacher from the intent name alone (§4.5), fixed before any test read. MASSIVE requests in French keep English names and descriptions.

### 4.4 Slates

- **Training and validation: four slates per request**, each with 2 to 8 options (uniform), *none* always present and last:
  1. *natural*: the conversation's own tools (when the source lists several), then random tools of the split;
  2. *hard*: the gold plus the nearest tools by potion-base-2M cosine among other clusters of the split, after the teacher filter (§4.5);
  3. *removed*: slate 1 or 2 (alternately) without the gold, which becomes *none*;
  4. *renamed*: slate 1 with the gold's name replaced by `tool_<k>` (k random) and its description by the teacher's rewrite.
  When a source's request has no call (When2Call cannot_answer, direct_answer), its slates are natural and hard, gold *none*.
- **Gold share of *none*:** fixed to 20% per group by dropping *removed* slates at random where it is exceeded, or adding them where it falls short.
- **T-tools (test): one slate per request**, 8 options (up to 7 tools + *none*): the conversation's own tools, then random tools of the test split; a distractor is excluded if its normalised name equals the gold's or its MiniLM description cosine to the gold is ≥ 0.8. 20% of the test slates are *removed* slates. No compared system is used to choose a test distractor. 2,000 slates, drawn by seed; the file and its sha256 are written in an amendment before any training.
- **Validation slates** are built like the test slates (one per request, 8 options) for the pilot rule, the baselines' fitting and τ_ask; the four training slate kinds are also kept for the training-side readings of §7.3.

### 4.5 Teacher (GLM-5.3-flash, `sky-data teacher`, 20 items per call, the rules of amendments 1 and 3 of `docs/s6-gen-prereg.md`)

| job | items | calls | estimated time |
|---|---:|---:|---:|
| one description per intent, from its name (MASSIVE 60, Banking77 77, CLINC150 150) | ≈ 290 | ≈ 15 | minutes |
| false-negative filter: for each training and validation request with its *hard* slate, "which of these options could also handle this request?"; every distractor it names is replaced by the next nearest tool; requests where it names the gold as unable are dropped and counted | ≈ 75,000 | ≈ 3,750 | ≈ 8.3 h |
| one rewrite of each training tool's description (same meaning, other words) | ≈ 25,000 | ≈ 1,250 | ≈ 2.8 h |

The teacher never sees a test request or a test slate. Its agreement with the gold (it should name the gold as able) is reported per source. Before the pilots, 200 validation slates are read by hand (inputs and gold only, no score of any system) and the share that holds a second valid tool is reported as the label-noise estimate of the test builder.

### 4.6 Leakage rules

1. Test files are written once by the builder, hashed, and their hashes go into an amendment before the pilots. They are never read before §7.
2. No test cluster in training (§4.2); MetaTool, CLINC150, BFCL, tool-routing-data and the When2Call test are never trained, and no training request equals (normalised) any of their test requests.
3. The held-out MASSIVE scenarios are absent from training in both languages.
4. v1's test files (`typed_decisions_bench`, `zs_massive_*`, NOTAM) are not rebuilt and not used.
5. Real-looking keys in tool schemas (ToolACE has at least two) are masked by the builder; the count is reported.

## 5. The run (`scripts/train-route.sh`, to be written; `scripts/train-s6.sh` is not changed)

| phase | budget (working minutes) | notes |
|---|---:|---|
| smoke | ≈ 3 | under a throwaway prefix (`SKY_OUT`); numbers not looked at; files deleted |
| pilot, nano | 20 | from v1's pretraining init; the §4 data; full schedule compressed to 20 minutes (QAT after minute 2) |
| pilot, micro | 20 | from s6-micro's pretraining init; same data |
| main decision training | **120** | the chosen size, from its init; QAT after minute 12; seed 0; one run |
| calibration, export, τ_ask | ≈ 5 | `s2-calibrate` on validation, `s2-export --format ternary`, `gguf-set` thresholds 1 |
| evaluation | ≈ 15 | §7, model first while the GPU is idle (latency), baselines on the CPU |
| **total GPU** | **≈ 183** | about 3 hours of working time; ≈ 45 if the pilot rule stops |
| CPU: data build, baselines rehearsal, baselines on the tests | ≈ 30 + 20 + 15 | |
| teacher | ≈ 5,000 calls, ≈ 11 h remote | no GPU; before the pilots |

**The pilot rule** (read on the validation slates of T-tools' builder, ternary, full depth, after each pilot's own calibration; written into `runs/s6-nano2-pilot/summary.json`):

1. The size is nano, unless micro's validation accuracy is at least **0.04** above nano's: micro costs 3.4× the bytes and about 5× the latency.
2. **Abort:** if the chosen pilot's validation accuracy is below `potion-cos` on the same validation slates **− 0.15**, the main run does not happen. The pilots are reported, the file is not trained further, and §11 applies as a failure of G1. (In v1's decision curve a 20-minute run reached about 87% of its final validation accuracy; 0.15 leaves room for that gap and no more.)
3. Nothing else is chosen on the pilots: not the weights, not the data, not the budget.

Brake modes as amendments 5, 6 and 10 of `docs/s6-gen-prereg.md`; budgets count working time only; the report gives the steps done and the modes. Only a technical failure justifies a restart, from the last checkpoint (`RESUME=1`). Folders `runs/s6-nano2*`, `models/s6-nano2`, `data/route`; nothing of v1's folders is written.

## 6. Baselines (fitted on the same training items, scored on the same test slates by the same scorer)

| name | role | definition |
|---|---|---|
| `potion-cos` | **gated (G1)** | model2vec 0.10.0, `minishlab/potion-base-2M` at a pinned commit, as published (7,559,256 bytes, f32): cosine between the request and each `name: description`; *none* chosen when the best cosine is below a threshold fitted on validation accuracy; probabilities = softmax over [cosines, threshold] with a temperature fitted on validation NLL |
| `static-ft` | **gated (G1)** | the same model fine-tuned with `StaticModelForPairSimilarity` (InfoNCE, defaults) on the training (request, gold option) pairs with the *hard* distractors as negatives, then reduced to at most `B` bytes (4-bit packing, then the vocabulary cut to the most frequent tokens of the training texts); *none* and temperature as `potion-cos`; epochs among 1, 3, 10 chosen on validation |
| `pair-lr` | **gated (G1)** | a binary logistic model on (request, option) pairs: hashed character 1- to 5-grams of the request and of the option, and hashed word pairs (request word × option word), int8, columns ≤ `B`; `SGDClassifier(log_loss, l2)`, scikit-learn 1.9.1, `alpha` among 1e-6, 1e-5, 1e-4 on validation; trained on every training slate (gold positive, others negative, *none* as the text of its option); argmax over the options; softmax with a fitted temperature |
| `potion-cos-4bit` | reported | `potion-cos` packed to 4 bits (about 1 MB), to show the cost of matching the size |
| `bm25` | reported | bm25s, Lucene variant, documents = `name: description`; *none* by a fitted threshold |
| `minilm-cos` | reported | all-MiniLM-L6-v2 (22M, 90.9 MB), cosine; *none* by a fitted threshold. On T-tools it is also the clustering model of §4.2, which this note discloses |
| `toolrouter` | reported | stanley-nv/toolrouter at a pinned commit (7.66 MB), its own API, *none* as the tool `other` with its own description |
| `name-prior` | reported | the option whose name was gold most often in training; *none* when no option was ever gold |
| s6-nano v1 | reported | the published file at full depth, same prompt |

The baselines are rehearsed on the validation slates with the frozen code of `scripts/route-baselines.py`, and their validation numbers are written in an amendment before the pilots: they set the pilot rule's bar and say, before any test read, how high G1 is.

## 7. Evaluation

### 7.1 Test sources (ternary GGUF, full depth, exit thresholds 1; at most 2,000 slates per source)

| source | slates | role |
|---|---|---|
| **T-tools** (`data/route/test_tools.jsonl`) | 2,000 | fresh, tool-disjoint; **G1, G2, G3, G4** |
| T-metatool | 2,000: a single-tool query, its tool, 6 distractors (3 from its "similar choices" group where it has one, else random) and *none*; 20% *removed* | never-trained family; announced goal |
| T-clinc | 2,000: in-scope query with its intent + 6 random intents + *none*; out-of-scope queries with 7 random intents, gold *none*; 20% *none* | zero-shot intents by description; announced goal |
| tool-routing-data `tools/test` | 100 queries × all 10 tools + *none* | reported; a published potion-2M number exists (0.76, without *none*) |
| BFCL v2 `multiple` (200) and `irrelevance` (240) | 440 | reported |
| When2Call `test/mcq` reformatted (the first 2,000, as v1) | 2,000 | reported; read in v1 |
| MASSIVE held-out scenarios, test partition, en and fr | ≤ 1,000 each | reported; partly read in v1 |

### 7.2 Metrics (per source, for the model and every baseline)

Top-1 accuracy; accuracy on tool-gold and on *none*-gold slates; *none* precision, recall and F1; ECE (15 bins, `sky-metrics`), NLL, mean confidence; the risk–coverage curve and the accuracy and coverage at τ_ask; a 95% paired bootstrap interval of every difference used in a gate (1,000 resamples of the slates).

### 7.3 Reading checks (on T-tools)

- **Not constant:** the share of *none* answers; the most frequent predicted tool name's share against its gold share; on 500 test requests rendered twice (gold present, gold removed; the second slate is built by the same rule), the share of requests whose answer changes, among those answered correctly with the gold present.
- **Names or descriptions:** names masked (`tool_<k>`) on all 2,000: accuracy; descriptions of the gold and of one distractor swapped on 500: the share of answers that follow the description.
- The same checks are reported for every baseline.

### 7.4 Latency and size

WASM (`crates/sky-web/bench/parity-node.mjs` or a router mode of it, Node, SIMD128 and relaxed builds, Apple M5) on 200 T-tools requests, on a copy with exit thresholds 1: median per request with the tool list cached, the first request (tool list encoded), max |p_wasm − p_native|, argmax agreement. Native median (`sky-eval`). Exit 0 accuracy and latency. GGUF bytes. Baselines' CPU time per request.

## 8. Gate (ternary GGUF, full depth; every condition must hold)

| | condition | why this number |
|---|---|---|
| **G1** held-out tools | T-tools accuracy ≥ `potion-cos` + **0.03**, and > `static-ft`, and > `pair-lr` | `potion-cos` is the published 7.56 MB model a practitioner reaches for first, at about 10× the bytes; +0.03 is about three standard errors of a paired difference on 2,000 slates near 0.75, a margin a user would see. The two trained baselines fit in the model's bytes and saw the same data: if either ties the model, it should ship instead (the rule of `docs/s6-secrets2-prereg.md` G3) |
| **G2** not constant | on T-tools: *none* answers between 0.10 and 0.30 of the slates (gold share 0.20); no tool name predicted more than twice its gold share + 0.01; on gold-removed pairs, the answer changes on ≥ 0.80 of the requests answered correctly with the gold present | v1 gave one answer to ≥ 95% of the items on 17 of 20 tasks. A router that ignores the list cannot meet the third condition |
| **G3** calibration | ECE ≤ **0.05** on T-tools | in distribution with a fitted temperature; v1 was at 0.025 on its in-distribution sources. A guard, not a target; the exact value is reported |
| **G4** latency | WASM SIMD128 median ≤ **2 ms** per request with the tool list cached (Apple M5, Node), and max \|p_wasm − p_native\| ≤ 1e-4 | interactive routing as the visitor types; v1 answers a short game frame in 445 µs; parity as every s6 report |

**Announced goals, not gates:** T-tools accuracy ≥ 0.80; T-metatool and T-clinc accuracy ≥ `potion-cos` on each; accuracy ≥ 0.95 among answered requests at τ_ask with coverage ≥ 0.70 on T-tools.

## 9. Reported, not gated

Every source of §7.1 for every system; the reading checks for every system; exit 0; the pilots (both sizes, with the abort rule's reading); the teacher's agreement and the filter's counts; the hand-read noise estimate; the cluster counts; truncation; steps, draws per group and passes per record; the GGUF size; v1 on every source. The site demo of §14 shows the verdict and the numbers in its first line.

## 10. Disclosure (written into every place that reports v2)

v2 was designed after v1's results were read, and after a study that read v1's test dumps by task, class, language and length. Its gated test, T-tools, is new: built by a rule fixed here, hashed before training, read once. MetaTool, CLINC150, BFCL and the tool-routing-data test were never read by this project. The When2Call test and MASSIVE test utterances were read in v1's evaluation and are reported only. The *none* share, the per-cluster cap and the counterfactual slates were chosen because of what v1's dumps showed. The clustering that defines "unseen tool" uses all-MiniLM-L6-v2, which is also a reported baseline.

## 11. What a failure means

- **The pilot rule stops the run, or G1 fails against `potion-cos`:** on run-time tool lists, a static embedding of 7.56 MB (about 1 MB packed) routes better than this graph at this size. Together with v1 and the nano2 study, that closes the case for a sub-megabyte decision model on language routing in this repository: the recommendation for that job becomes potion with cosine (or the toolrouter), and no fourth version of this graph is run on it.
- **G1 fails only against `static-ft` or `pair-lr`:** a trained model of the same size does the job; the report says so, and whether to ship that model becomes a separate decision, outside this file.
- **G2 fails:** the model ignores the options in part; not published as a router, whatever G1 says.
- **G3 or G4 fails alone:** not published as a router; the numbers are reported; calibration or speed work comes before any new pre-registration; this run is not repeated.
- **In every case**, following the owner's decision of 15/11/2026, the file is published as evaluated, with the verdict and the numbers in its first lines; the demo is published only if the gate passes, else the page shows the result without the playground.

## 12. Forbidden

- Reading any test file or test slate before §7, except the hashing of §4.6 and the exact-match deduplication of §4.2 and §4.3, which read texts, never labels or scores.
- Using any compared system to choose a test distractor.
- Choosing a checkpoint, a budget, a weight, a threshold or a size on a test score; any choice on the pilots other than the two of §5.
- Changing a baseline's definition after the rehearsal amendment; adding training data after the freeze; rebuilding a test file.
- A second seed or a second main run, except after a technical failure.
- Sending a test request to the teacher.

## 13. Choices (the first value of each line is the one taken at the freeze)

1. **Size**: the pilot rule between nano and micro / nano only (saves 20 minutes; drops the capacity question) / micro only.
2. **Main budget**: 120 working minutes / 90 / 180.
3. **Abort margin of the pilot rule**: −0.15 / −0.10 (stricter; more likely to stop a run that would pass) / no abort.
4. **G1 margin over `potion-cos`**: +0.03 / +0.00 (match a 10× larger model) / +0.05.
5. **Teacher jobs**: intent descriptions + false-negative filter + description rewrites (≈ 5,000 calls) / + request paraphrases (≈ 6,750 calls, ≈ 15 h; fewer passes per request) / descriptions only (≈ 15 calls; hard negatives unfiltered).
6. **Group weights**: `tools=0.60, intents=0.25, when2call=0.15` / tools only / `0.50, 0.35, 0.15`.
7. **French**: MASSIVE fr in training, reported on held-out scenarios / English only.
8. **Initialisation**: the pretraining init / v1's last decision checkpoint (knows the format; carries its priors).
9. **Head window candidates**: 160, 256, 320 by the 5% rule / 256 fixed.
10. **Slates per request in training**: 4 / 2 (natural and removed) / 6 (two more hard slates).
11. **Per-cluster cap**: 30 / 10 / none.
12. **Name**: `s6-route-nano` / `s6-nano` v2.
13. **MetaTool**: an announced goal / a gate (≥ `potion-cos`; about halves the odds of a pass).

## 14. Code to write before training (by file)

1. `crates/sky-data/src/route.rs` (new) and its subcommand in `crates/sky-data/src/main.rs`: fetch of ToolACE, Seal-Tools, API-Bank, MetaTool, CLINC150, BFCL and tool-routing-data next to the existing glaive and hermes fetchers (`crates/sky-data/src/commands.rs`); tool normalisation, clustering (from an embedding file written by the baseline environment), the splits of §4.2, the intents of §4.3, the slates of §4.4, the masking of keys, the manifest of counts and hashes. Unit tests: no test cluster in training, the *none* share per group, the four slate kinds, the per-cluster cap, determinism under the seed.
2. `crates/sky-data/src/teacher.rs`: three prompt modes (`describe-intent`, `false-negatives`, `rewrite-description`), resumable as `label-decisions`, with the moderation split of amendment 3 of `docs/s6-gen-prereg.md`; a guard that refuses any record whose split is `test`.
3. `scripts/route-embed.py` and `scripts/route-baselines.py` in a pinned environment (model2vec 0.10.0, sentence-transformers, bm25s, scikit-learn 1.9.1): the clustering embeddings; `potion-cos`, `static-ft`, `pair-lr`, `potion-cos-4bit`, `bm25`, `minilm-cos`, `toolrouter`, `name-prior`; outputs in `sky-eval`'s dump format; a `--split val|test` switch that refuses `test` until `runs/s6-nano2/TEST-OPEN` exists.
4. `scripts/train-route.sh` (new): smoke, the two pilots and their rule, the main run, calibration, τ_ask, export, `gguf-set`, evaluation, the reading checks; `DRY_RUN` and `SKY_OUT` as in `scripts/train-s6.sh`.
5. `scripts/s6-route-gate.py` and `scripts/test_s6_route_gate.py`: the gate of §8 from the dumps, the bootstrap, the reading checks, the pilot rule (a subcommand), the report.
6. `crates/sky-web/bench/parity-node.mjs`: a router mode that times requests with the static stream cached and the first request apart.
7. `site/router/` (new page) and its link from the site index: the playground (an editable tool list of up to 7 lines, presets, a request box, live probabilities with *none*, the τ_ask slider and "ask the user", "mask names" and "swap descriptions" buttons, first-call and per-request latency, the verdict line), behind the gate as §11 says.
8. `DATA_LICENSES.md`: the new sources, their licences, commits and roles (train or test only).

About 1,500 lines in all; none of it changes `sky-train` or `sky-infer` (the head window is a config value).

## 15. Amendments

(none)

**Amendment 1 (15/11/2026, after the data builder, the baselines and the scripts were written and rehearsed on validation, before the teacher's full budget, before any training; no model of this version exists, no test file has been scored).**

- **The validation rehearsal of the baselines (v_tools, 1,144 slates; none of this is a test score).** potion-base-2M with cosine: 0.7255. The same in 4 bits: 0.7229. BM25 over the slate's own tool texts, which has no parameter at all: 0.8549. all-MiniLM-L6-v2 with cosine (91.6 MB): 0.9126. s6-nano v1 on the same slates: 0.0979.
- **A lexical baseline beats the embedding baseline this file gated against, so the design and the gate take it in.** The project's one clear success has a fixed shape: something cheap and deterministic proposes, and the small model judges with what it was handed (`docs/s6-secrets2-prereg.md`). Here: **each option's text carries its BM25 score against the request and its rank in the slate**, as two fused numbers, computed over the slate's own options by one function shared by the builder and the engine (a fixture both must pass). And **G1 gains a second condition: on the unseen tools of the test, the model's accuracy is above BM25's**, next to the condition against potion that §8 wrote. A router that reads a lexical score and does no better than that score has no use.
- **Renamed slates rename every tool.** As written, only the right tool of a renamed slate got the name `tool_<k>`, so the name gave the answer away. In a renamed slate every option is renamed `tool_<k>` with distinct numbers drawn at random; the descriptions are untouched.
- **The share of "none".** When2Call's own no-call slates are about 35% of its slates; they are thinned at random to the 20% of §4 (the removed copies first), as built.
- **Readings fixed as built.** The thirty-four readings the builder had to take are recorded in `docs/s6-nano2-readings.md`, committed with the code, and are accepted as written there, with the two changes above. Among them: route/tools requests whose first reply calls no tool are dropped rather than labelled "none"; clusters go to the test side before validation before training and the lower copy is dropped; the caps of 30 requests per cluster in training and 10 in validation and test; the false-negative filter drops a request when the teacher does not name its right tool as able; `static-ft` is trained on pairs without hard negatives (the library has no way to pass them); the size-matched baselines are refitted to the trained file's size; "the answer changes" in G2 is compared by option position; BFCL's files are `BFCL_v3_multiple` and `BFCL_v3_irrelevance` at the pinned commit; the data lives outside the repository.
- **A disclosure.** While printing a schema, one line of the MASSIVE test (already read for v1) and one BFCL function name were seen. No statistic of a test file was computed beyond counts. The hashes of the test files are in `runs/s6-nano2/test-files.json`.
- **Still to fix before training, each by the rule already written:** the head window of each size (§3, by `s2-truncation`, when the trainer is available), and the teacher's answers (4,194 calls in all; 39 were run to test the plumbing).
- **What does not change:** the job, the sources and the split, the pilots and their stop rule, the budgets, G2 to G4, §12.

**Amendment 2 (16/11/2026, 03:20, after the teacher's full budget, the final build and the validation rehearsal of every baseline, before any training; no model of this version exists, no test file has been scored).**

- **The build is final.** The teacher answered 4,174 calls with no failed batch (2 items refused by moderation); 4,562 requests were dropped by the false-negative filter and 14,497 distractors replaced. Records, training / validation: tools 81,596 / 4,185, intents 95,411 / 11,543, When2Call 14,621 / 896 (208,252 in all). A second build gives the same files (`BUILD.json`, not provisional).
- **The validation rehearsal on the final slates (v_tools, 1,144 slates; none of this is a test score).** all-MiniLM-L6-v2 0.9126, BM25 0.8549, toolrouter 0.7832, static-ft 0.7351, potion-cos 0.7255, potion-cos-4bit 0.7229, pair-lr 0.4843 (α 1e-6), name-prior 0.2002; s6-nano v1 at full depth on the model's copy of the slates 0.0848. The gate script ran end to end with potion-cos in the model's seat (`runs/s6-nano2/eval-val/seat-potion-cos`): it reports "not passed", as it must for a rehearsal.
- **The head window, and a hole in its rule.** `s2-truncation` over the 208,252 training and validation records gives, for the share of records with an option shrunk: micro (4,096-token vocabulary) 0.2943 at 160, 0.0409 at 256, 0.0378 at 320; nano (2,048-token vocabulary) 0.4378 at 160, 0.1428 at 256, 0.1298 at 320. The rule of §3 gives **256 for micro** and nothing for nano. The cause is not the window: the engine caps every option at 48 tokens (SPEC, `OPTION_MAX_TOKENS`), and with nano's smaller vocabulary 13% of the records hold a tool whose text is longer than that. Measured to be sure, on the same records: 0.1297 at 384, at 448 and at 512. **The rule gains one case: when no candidate meets 5%, the window is the largest candidate.** So **nano reads a head window of 320**, where the window itself cuts one record in 10,000 and the 48-token cap cuts the tail of a tool's description in 13% of the records (by source: hermes 31%, toolace 30%, When2Call 27%, seal-tools 19%, glaive 18%, api-bank 16%, banking77 7.5%, MASSIVE 0). The cap is the engine's and is the same at inference, so this is what the product would do; it is reported, and it is a handicap of the nano size that the pilot rule (micro if at least 0.04 above nano) is there to read. No engine constant changes.
- **Where the run happens.** From the worktree of the session, on the code of `origin/main`, with the recipe's configs, the tokenizers and the two pretrained inits read through links from the main checkout (the script copies each init and compares its sha256 before and after). The brake is the automatic one (`scripts/auto-brake.sh`). Other models' CPU work (a data build, CPU pilots) may run during the pilots and the main run; budgets are in working minutes, so this can only lower the number of steps, and the steps are reported.
- **What does not change:** everything else, in particular the pilots and their stop rule, the budgets, the gate.

**Result (16/11/2026, after the test was read once). The gate failed, on G1 (against BM25) and on G4. Not published as a router; the file is published as evaluated, as a result.**

- **The pilots** (`runs/s6-nano2-pilot/summary.json`; the 1,144 validation slates of T-tools' builder, ternary, full depth). nano 0.5778 (3,271 steps in 20.01 working minutes), micro 0.6390 (3,191 steps in 20.01), potion-cos 0.7255 on the same slates, so the abort bar was 0.5755. Micro is more than 0.04 above nano: the size is **micro**, and the main run is not aborted. Nothing else was chosen on the pilots.
- **The run** (`runs/s6-nano2/report.md`, `runs/all/route-train.log`). Micro (5,587,201 parameters, the s6-micro graph and tokenizer, head window 256 as amendment 2 fixed it), from `runs/s6-micro/pretrain/init`, seed 0, one run, no restart: **120.00 working minutes, 20,504 steps**, checkpoint `last`, under the automatic brake (the modes and their counts are in the report). Validation at full depth (v_tools, 1,144 slates): 0.8339, where BM25 is at 0.8549 on the same slates (amendment 2). **τ_ask = 0.72938** (validation: accuracy 0.9501 among answered requests, coverage 0.7360), stored under `skycmd-s2.route.tau_ask` with the exit thresholds at 1 in `models/s6-nano2/skycmd-s6-route-nano-ternary.gguf`, the evaluated file: 2,418,912 bytes, sha256 `a124dfe86f76e9547db98edc03351f66908617ae2be3f5a4f423a876b7956c82`.
- **The size-matched baselines, refitted before the test was opened** (reading 28; `runs/all/route-val-fit.log`, ended at 09:41:39): every baseline fitted again on validation at B = 2,418,912 bytes, the file's size. static-ft 0.7351 and pair-lr 0.4843 (α 1e-6) on validation, the values of amendment 2; potion-cos 0.7255, potion-cos-4bit 0.7229, BM25 0.8549, MiniLM 0.9126, toolrouter 0.7832, name-prior 0.2002.
- **The test, opened once.** `runs/s6-nano2/TEST-OPEN` was created at 09:41:39 on 16/11, the gate was written at 09:50:02. In this order: the model (every source, the reading checks, exit 0, the native latency), its WASM timing (both builds), s6-nano v1, the baselines, the gate. The model and its latency came first, while the GPU was idle (§5). Results: `runs/s6-nano2/eval-test/results.md` and `gate.json`; the latency files in `runs/s6-nano2/eval-test/latency/`.
- **Deviations and disclosures.**
  - `TEST-OPEN` was created by the run's operator under the owner's delegation, not by the owner's hand, as `scripts/route-eval.sh` says it should be. The owner's standing instruction was to run every step without asking. The scripts refuse the test split until that file exists.
  - The evaluation binaries and the WASM package were rebuilt from `origin/main` at `8b0ece3` after training. The training used binaries built earlier the same day, before the engine changes of s6-sleep v3 (the fast-weight memory in the s2 engine) were merged. WASM and native agree on the evaluated file to 2.2e-6, with the same answer on 200 of 200 requests.
  - The disclosure of §10 holds as written: v2 was designed after v1's results were read; T-tools is new, built by a rule fixed before training, hashed (`runs/s6-nano2/test-files.json`) and read once; the When2Call and MASSIVE test utterances were read in v1's evaluation; the clustering that defines "unseen tool" uses all-MiniLM-L6-v2, which is also a reported baseline.
- **The gate (T-tools, 2,000 slates of tools never seen in training, ternary, full depth)**:

  | gate | condition | measured | verdict |
  |---|---|---|---|
  | G1 | ≥ potion-cos + 0.03 | 0.8395 against 0.6675 + 0.03 = 0.6975 | holds |
  | G1 | > static-ft (size-matched) | 0.8395 against 0.6735 | holds |
  | G1 | > pair-lr (size-matched) | 0.8395 against 0.4500 | holds |
  | G1 | > BM25 (amendment 1) | 0.8395 against 0.8500 | **fails** |
  | G2 | *none* answers in 0.10–0.30; no name above twice its gold share + 0.01; answer changes on ≥ 0.80 of the gold-removed pairs | 0.1565; largest name excess −0.0085; 0.9528 (424 of 445) | holds |
  | G3 | ECE ≤ 0.05 | 0.0244 (BM25 0.0610) | holds |
  | G4 | WASM SIMD128 median ≤ 2 ms per request, tool list cached; max \|p_wasm − p_native\| ≤ 1e-4 | 38.9 ms (first request 50.9 ms); 2.2e-6, argmax 200 of 200 | **fails** (latency) |

  **G1 fails and G4 fails: the gate is not passed.**
- **Paired bootstrap of the G1 differences** (1,000 resamples of the slates, 95%): model minus potion-cos +0.1720 [0.1490, 0.1965], minus static-ft +0.1660 [0.1415, 0.1895], minus pair-lr +0.3895 [0.3625, 0.4150], **minus BM25 −0.0105 [−0.0265, +0.0055]**. Against BM25 the interval holds zero: the model ties the score it is handed, and the condition (strictly above) is not met.
- **The announced goals** (not gates): T-tools ≥ 0.80: met (0.8395). T-metatool ≥ potion-cos: not met (0.3260 against 0.5245). T-clinc ≥ potion-cos: not met (0.3980 against 0.7740). Accuracy ≥ 0.95 among answered requests at τ_ask with coverage ≥ 0.70 on T-tools: not met (0.9433 at coverage 0.7325).
- **Every source** (accuracy; the full table, every system and every metric of §7.2, is in `results.md`):

  | source | n | model | potion-cos | static-ft | pair-lr | BM25 | MiniLM | toolrouter | s6-nano v1 |
  |---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
  | T-tools | 2,000 | **0.8395** | 0.6675 | 0.6735 | 0.4500 | 0.8500 | 0.8840 | 0.7855 | 0.0885 |
  | T-metatool | 2,000 | 0.3260 | 0.5245 | 0.5235 | 0.1640 | 0.4525 | 0.6610 | 0.5855 | 0.0825 |
  | T-clinc | 2,000 | 0.3980 | 0.7740 | 0.7760 | 0.1485 | 0.4775 | 0.8605 | 0.7235 | 0.1135 |
  | tool-routing-data | 100 | 0.3700 | 0.6100 | 0.6000 | 0.1100 | 0.3800 | 0.6000 | 0.7900 | 0.1100 |
  | BFCL | 440 | 0.7205 | 0.7250 | 0.7091 | 0.4636 | 0.8295 | 0.7136 | 0.5250 | 0.3455 |
  | When2Call | 1,863 | 0.5153 | 0.5722 | 0.5797 | 0.3097 | 0.4895 | 0.6720 | 0.5915 | 0.3258 |
  | MASSIVE, held-out scenarios | 1,657 | 0.3645 | 0.4762 | 0.4852 | 0.1135 | 0.3343 | 0.5836 | 0.6035 | 0.1002 |

  On T-tools the model's *none* F1 is 0.7041 (BM25 0.6486, MiniLM 0.7516). potion-cos-4bit is at 0.6665 on T-tools; name-prior at 0.2000.
- **Risk and coverage on T-tools** (accuracy among the answered requests at coverage 0.5 / 0.7 / 0.9, `gate.json`): model 0.991 / 0.956 / 0.885, BM25 0.979 / 0.945 / 0.878, MiniLM 0.993 / 0.974 / 0.928.
- **The reading checks (§7.3, T-tools).** Answer changes on gold-removed pairs: model 0.9528, BM25 0.9453, potion-cos 0.9418. **Names masked** (every tool renamed `tool_<k>`, all 2,000 slates): **model 0.1050**, at chance and below name-prior (0.1180), where BM25 keeps 0.8445, potion-cos 0.7015 and MiniLM 0.8845. The BM25 entries of the state are keyed by the masked names too, and BM25 alone, on the same texts, still finds most answers. One reading, not tested: the model ties each BM25 entry to its option through a distinctive name, and loses that binding when the names look alike. Swapped descriptions (500 slates): the model follows the description on 0.6700 and the name on 0.0780 (BM25 0.8400 and 0.0260).
- **Latency and size** (§7.4; `runs/s6-nano2/eval-test/latency/`). WASM under Node on 200 T-tools requests, a copy with exit thresholds 1: SIMD128 median 38,929 µs per request with the tool list cached (the static stream hit on 200 of 200), 50,922 µs for the first request; relaxed SIMD 23,532 and 33,204 µs. Native (`sky-eval`, the same 200): median 24,031 µs, p95 48,782 µs. Exit 0 (2 of 6 block passes) on the 2,000 T-tools slates: accuracy 0.8340, ECE 0.0277, native median 13,340 µs. G4's bar is 2 ms: the cached WASM median is about 19 times above it. GGUF 2,418,912 bytes.
- **What it means, by §11.** G1 holds against potion-cos and against both size-matched trained baselines, so neither the first nor the second case of §11 applies. It fails against BM25, the lexical score the model is handed in its state, the condition amendment 1 added with its reason: "a router that reads a lexical score and does no better than that score has no use." G4 fails as well, by a factor of about 19. **s6-route-nano is not published as a router.** Following the owner's decision of 15/11/2026 (§11, last item), the file is published as evaluated (`site/public/models/s6-route-nano.gguf`, the evaluated file byte for byte, sha256 above), with the verdict and the numbers in its first lines; there is no playground, and the site page shows the result. This run is not repeated (§12).
- **Not in this block:** the hand-read label-noise estimate and the teacher's agreement per source (§4.5), the draws per group and passes per record (§9) and the baselines' CPU time per request (§7.4) are not in the result files.

**Exploratory, after the gate, nothing fitted, on the already-read test.** Not results, and nothing here changes the verdict: these numbers were computed on test dumps already read for the gate, by `scripts/s6-route-explore.py` (`runs/s6-nano2/explore/test.txt` and `test.json`), which takes the equal-weight geometric mean of two systems' probabilities over a slate's options and renormalises it. On T-tools: the model with potion-cos-4bit 0.8885, above BM25 (0.8500) and MiniLM (0.8840), where BM25 with potion-cos-4bit gives 0.8475 (with potion-cos at full precision 0.8515); the model with BM25 0.8780; the model with MiniLM 0.9295; BM25 with MiniLM 0.9165. The same fusion of the model and potion-cos-4bit elsewhere: T-metatool 0.5290, T-clinc 0.7935, When2Call 0.6280, MASSIVE held-out 0.5570, BFCL 0.7659. Where the model and BM25 give the same answer (0.8275 of the T-tools slates) that answer is right on 0.9384 of them. These are hypotheses for a later pre-registration on fresh slates.

**Owner's decision (17/11/2026), a deviation from §11.** §11 said the playground is published only if the gate passes. The owner decided that every published model gets a Space on Hugging Face, failed gates included. s6-route-nano therefore gets a Space that runs the file alone, with the failed gate and its two numbers in its first lines and a link to the pair that passed (s6-route). The verdict does not change; no playground of the failed file is added to the site's Router page, which stays the pair's.
