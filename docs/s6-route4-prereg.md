# s6-route v4 (the router retrained without its shortcut): pre-registration

**Frozen on 17/11/2026 by the lead, before any computation of v4 and before its code is written; every later change is a dated amendment that precedes the computation it concerns.**

- No data of v4 is built, no teacher call is made, no model is trained and no fresh file exists.
  The choices of §15 are taken as the lead decided them on 17/11/2026 (§16).
- **Fixed later by rules written here, each before any training:** the rendering of the lexical
  score (§3.4, by a latency check on validation), the head window (§3.3, by v2's rule), and, if the
  data checks of §4.4 still find a cue, the one remedy written there.
- **Still to be written by amendment before any scoring:** the counts of the new test files
  (§6.3: rows read and dropped by each filter, slates, servers, clusters, the share of the three
  largest servers, the *none* share, rows of each subset, and the request-blind rules of §4.4 C2
  counted on them), with their hashes; and, before training, the outcome of §3.4, the head window,
  the data checks of §4.4, the validation bar of the pilot rule and the values of the fit.

**Why this experiment.** s6-route (v3) passed its gate on fresh slates and is published, but its
model part learned a shortcut (`docs/s6-route4-diagnosis.md`): in every training slate of v2's
*renamed* kind the right tool is the only option whose description the teacher rewrote, and *none*
is never right. Under uninformative names the model part picks the most rewrite-like description
(a rewritten distractor is picked on 0.8311 of masked validation slates) and almost never answers
*none*, and the pair falls below its own embedding part (0.2497 against 0.7280 on the masked
T2-tools). No rendering repairs the published weights. v4 changes the data so that the cue is
gone, retrains the model part on the same graph, and tests the pair on tools of a family that no
model of this project has seen. The design study, `docs/s6-route4-study.md`, holds every number
quoted here with its source.

**Changes to this file (once frozen):**

- every amendment is committed, dated and justified in §17;
- an amendment comes before the run or the measurement it concerns;
- an amendment never moves a threshold after a fresh score of v4 has been read;
- measurements on training and validation splits (the data checks, the latency check, the
  rehearsal of the scoring chain, the pilot) are allowed before the main run, dated and written
  down; they never read a fresh file.

## 1. The question

With the teacher's rewrite made independent of the right answer, *none* right on 20% of the
slates whose tools are named `tool_<k>`, those slates built like the names-masked reading, and the
BM25 score written into each option's own text, does the pair (the retrained model part and the
same 1 MB embedding, fused by v3's rule) on tools of a family never trained on:

1. route better than BM25, which costs no byte, and better than its own 1 MB embedding, both
   beyond noise, and better than its model part alone (G1), with its distance to all-MiniLM-L6-v2
   reported?
2. stop picking the option that reads like a rewrite, and stay at least as good as its embedding
   part when names say nothing (G2)?
3. stay calibrated (G3), run in the browser within 60 ms with parity (G4), and lose nothing that
   matters on v3's fresh slates (G5)?

A pass means a drop-in replacement of the two files of s6-route that does not need distinctive
tool names. A failure of G2 means the shortcut is not removed by the data alone (§11).

## 2. What changes from v3, and why

| # | change | evidence | where it is checked |
|---|---|---|---|
| D1 | **Each tool option's description is the teacher's rewrite with probability 0.5**, drawn independently of the right answer, in every kind of training and validation record | renamed validation records 0.9498, 0.3271 with the gold's original back; a rewritten distractor picked on 0.8311 (diagnosis §1) | §4.4 C1, C2; G2 (a) |
| D2 | **Removed slates under `tool_<k>` names**: *none* right on 20% of them | 0 of 51,101 renamed training records have *none* right; 1 *none* answer in 1,144 masked validation slates (diagnosis §2 (c)) | §4.4 C3; G2 (c) |
| D3 | **Renamed slates built by the test-slate rule** (7 tools and *none*, the masked reading's twin) | the masked reading and the renamed kind differ in the gold's description, *none* and the slate size (diagnosis §2 (a)) | §4.3 |
| D4 | **The BM25 score and rank in each option's own text** (option (i) of the diagnosis), unless the latency check of §3.4 fails | under `tool_<k>` names the line is not read (shifted: 0.9489 against 0.9498); under real names it is the main input (0.8339 against 0.4318 without) | §3.4; G2 (b) |
| D5 | **A removed slate keeps the option count of the slate it comes from** (one more distractor of the slate's rule takes the gold's place, as in the test; taken by the lead) | an 8-option natural or hard training slate never has *none* right, a test slate always has 8 options (study §2.5) | §4.4 C3 |
| D6 | **A fresh test of a new family**: Toucan-1.5M, MCP tools, built so that the surface cue of D7 cannot be in it | no unread test-side request of v2 is left outside three already-read clusters (study §4.1) | §6 |
| D7 | **Random fill from the right tool's own source first**, in every kind and in the new validation slates | without reading the request, the shortest description is right on 0.2550 of the tool-right validation slates, 0.2439 on T-tools and 0.3633 on T2-tools, against 1/7 (study §7) | §4.4 C2 |

**Not changed:** the job, the instruction, *none* always last, the sources and their split (v2
§4.1, §4.2, the clusters, the caps), the false-negative filter and its answers, the group weights,
the graph and tokenizer of the published model part (s6-micro: 5,587,201 parameters,
`tok-s6-4096`), the recipe of v2 §2 and §5, the embedding part and its two fitted values, the
fusion rule and the ask rule of v3 §2, the scorer and the baselines of v2 §6 with their fitted
values.

## 3. The system

### 3.1 The model part

- **Graph:** the published file's (s6-micro, 2 weight-shared blocks looped three times, d 256,
  vocabulary 4,096, state window 384), so that the file is a drop-in replacement of
  `s6-route-nano.gguf`.
- **Initialisation:** `runs/s6-micro/pretrain/init`, copied and compared by sha256 before and
  after, as v2. Not v2's trained weights, which carry the shortcut.
- **Prompt:** v2's instruction and *none* text, word for word. Each tool option is
  `score <s> rank <r>; <name>: <description>` under D4 (the scores of `sky_infer::route`, two
  decimals, rank 1 for the highest, ties to the earlier option), the description cut by
  `cut_description`; *none* gets no score. The state is `User: <request>`, with no BM25 line.
  The score comes first so that the 48-token cap of each option never removes it.
- **File:** ternary export, exit thresholds 1, `skycmd-s2.route.tau_ask` by v2's rule, and a new
  key `skycmd-s2.route.render` = `scores-in-options` (absent in v2's file, which the engine keeps
  rendering as before).

### 3.2 The embedding part and the fusion

- **The embedding part:** `potion-cos-4bit` exactly as in v3 §2 (θ = 0.396120, T = 0.0671021); it
  reads `name: description` without the scores.
- **The fusion:** `p(i) ∝ (p_model(i) · p_embedding(i))^β`; **β re-fitted** on the new validation
  slates `v4_tools` (1,144, §4.3) by v3's rule (the grid 0.01 … 4.00, the least NLL), **τ_ask4 re-fitted**
  by v2's rule on the fused probabilities. Both are stored in a new embedding file (the same table,
  the new β and τ), so the system is two files of the sizes of v3's (2,418,912 and about 1,273,312
  bytes).
- **The fast mode** (the model part at its first exit) is reported, as v3.

### 3.3 The head window

v2's rule of §3 with amendment 2 (`s2-truncation` on the new training and validation records; the
smallest of 160, 256, 320 at which at most 5% of the records have an option shrunk, else the
largest). Today micro reads 256 at 0.0409; the longer options of D4 may move it to 320.

### 3.4 The rendering of the score: the check that decides it (validation only, before the build)

- **Measured:** `crates/sky-web/bench/route3-node.mjs` in a mode where every request meets its
  tool list for the first time, with v2's published file (the same graph: the weights do not change
  the time), on the first 200 slates of `v4_tools` rendered by D4, at the head window D4's records
  would need (320 if unknown), SIMD128, Apple M5, with no training running.
- **Rule:** if the median per request is **≤ 50,000 µs**, D4 is option (i). Otherwise D4 is
  option (ii'): the BM25 line is kept in the state and keyed by labels the engine writes before
  each tool option (`t1` … `t7` by position), and a tokenizer check must show that in every
  training and validation slate the seven labels are distinct token sequences after the numeric
  extraction and fusion; if it fails, v4 stops before training.
- **Why 50 ms:** v3's bar is 60 ms; v3 measured this path at 45.2 ms for the model part on a
  machine in use and v2 at 50.9 ms on the same graph; 10 ms is about that spread.

### 3.5 Size

About 3.69 MB in two files, as v3; G4 reads the two files as shipped (≤ 4,000,000 bytes).

## 4. Data (`sky-data route`, extended)

### 4.1 What is kept

v2's stage files, clusters, splits, caps, plans of the hard slates (their random streams
unchanged, so that the 69,290 false-negative answers apply by their ids), the intents and their
descriptions, the validation slates `v_tools` and `v_tools_masked` as built (reported readings).
The natural slates are drawn again by D7, from a random stream of their own.
The test files of v2 and the fresh files of v3 are not rebuilt and not read, except T2-tools for
G5 (§8).

### 4.2 The teacher

- **One job:** `rewrite-description`, the prompt of v2, for every tool of the training and
  validation pools and of the intents that has no rewrite yet: at most **13,691 items, about 685
  calls** of 20 (study §2.2, §2.3). GLM-5.3-Flash (MIT), `scripts/route-teacher.sh
  rewrite-description loop`. No other teacher job.
- **Refused or missing items:** a tool whose rewrite the moderation refuses is shown with its own
  description in every slate (it is counted, by group and split, in the build's counts), so that
  C1 measures any imbalance it creates.
- **The fresh test's tools, for the probe copies only** (the lead's decision of 17/11/2026): one
  rewrite of each Toucan tool's description that a fresh slate shows, by the same prompt, as a job
  whose items hold a tool's name and description and nothing else (split `fresh-tools`). The
  teacher sees descriptions alone, never a request and never a slate. The rewritten texts are used
  in `t4_tools_rw` and `t4_tools_masked_rw` and nowhere else: not in `t4_tools`, not in its
  masked copy. About 100 to 150 calls.
- **Never a test request.** The guard of `sky-data teacher` refuses any item that holds a request
  of a test or fresh file.

### 4.3 The slate kinds (training and validation records)

For every request with a tool gold, in each of the three groups:

| kind | built by | options | *none* right | names |
|---|---|---|---|---|
| natural | v2's plan's size and the request's own tools; the random fill drawn again by D7 | 2 to 8, uniform | no | real |
| hard | v2's plan, the false-negative filter applied | 2 to 8 | no | real |
| removed | natural or hard alternately without the gold (D5: one more distractor of the slate's rule in its place) | the source slate's count under D5 | yes | real |
| **renamed** | `slates::test_slate` on the request's pool with v2's test distractor filter and D7's fill, then `slates::masked` | 8 (7 tools and *none*) | on 20% of them, by `removed_set` per group and split | `tool_<k>`, distinct k in 1 … 99 |

- **D1 in every kind.** Each tool option's description is the rewrite (cut by
  `cut_description`) when a draw of `Rng::derived(seed, "route/rewrite/<request>/<kind>/<tool>")`
  is below 0.5; the draw never reads the gold.
- **D7, the fill.** Every random fill (natural, renamed, the refills of D5, the new validation
  slates) draws first among the pool's tools of the right tool's own source (for a request without
  a tool gold, the source of its own tools), then from the whole pool when the source has too few;
  the distractor filter of each kind applies as before. The hard slates keep v2's nearest tools.
- **When2Call requests without a call** keep their natural and hard slates, *none* right, real
  names, D1 applied.
- **The share of *none*:** 0.20 of the slates of each group and split (v2 §4.4, by trimming or
  adding plain removed slates), and 0.20 of the renamed slates of each group and split, each to the
  nearest slate.
- **New test-like validation slates** `v4_tools` and `v4_tools_masked`: the 1,144 validation
  requests of `v_tools`, one slate each by v2's test rule with D7's fill, 20% removed, the sources'
  descriptions (no rewrite), seed 1237. The calibration, β, τ_ask4 and the pilot are read on them,
  because `v_tools` carries the cue of D7 and the fresh test does not. `v_tools` and
  `v_tools_masked` are kept as reported readings (the published numbers were read on them). Two
  probe copies are added (§4.5).

### 4.4 Checks before any training, on the new data alone (no model)

`scripts/s6-route4-checks.py` (with the rules of `scripts/s6-route4-blind.py`), on every (group,
kind, split) of the training and validation records and on the validation slate files. A cell with fewer than 200 tool-gold slates is reported and
not judged.

- **C1, the rewrite is independent of the answer.** |P(rewritten | right tool) − P(rewritten |
  other tool option)| ≤ **0.01** in every cell with at least 1,000 tool-gold slates, ≤ **0.03**
  elsewhere.
- **C2, request-blind rules at chance.** Each rule picks one tool option from the option texts
  alone; ties count as the expected value (1/|tie| if the gold is in the tie); chance c is the mean
  of 1/(number of tool options). On the tool-gold slates of a cell of n slates, a rule fails when
  its accuracy exceeds **c + 0.02 + 2·√(c(1 − c)/n)**. The rules, each with its inverse (the
  option least so):
  R1 the description ends with a period; R2 its first word is a capitalised word in -s (not -ss);
  R3 the diagnosis's rule (R1, then R2); R4 the longest description in words, R5 in characters;
  R6 the option rewritten (the builder's own flag); R7 each position 1 … 7; R8 the name's case
  style (snake_case, camelCase, spaced words, other) that is the slate's minority; R9 the pair
  (name style, R1) that is the slate's minority; R10 the name with digits; R11 the shortest name;
  R12 the fewest "schema-like" words; R13 an imperative first word; and every other rule of
  `scripts/s6-route4-blind.py` (study §7).
- **C3, *none*.** The share of *none* by group, split and kind is 0.20 as §4.3 says; a table of
  the share of *none* by number of options is written (under D5 it is flat by construction; if D5
  is not taken, it is reported).
- **C4, the surface table.** For the right option against the others, by cell: the means of every
  feature of C2 (period, -s word, words, characters, rewritten, each name style, digits in the
  name, name length, position), and the share of options from each source. Reported, not judged.
- **C5, no fresh tool in training.** No tool of a fresh file is within v2's cluster rule of a
  training or validation tool (§6.2).
- **If C1 or C3 fails,** the build has a defect: it is fixed before anything else (a technical
  failure, reported). **If a rule of C2 fails,** the build stops. If the failing rules are all among
  R1 to R5, R8, R9, R12 and R13 (the surface of the descriptions and names, study §7), one remedy
  is written now: every random fill of every kind draws, within the right tool's source, among the tools of the right tool's surface cell
  (the cell of §6.2, item 5); the checks run once more, and if any rule still fails, v4 stops
  before training. Any other failing rule stops v4 before training, with the table published. (D7
  draws the natural slates' fill from a random stream of its own, so that the hard slates, and with
  them the false-negative answers, do not change: today both come from one stream,
  `build.rs:911-951`.)

### 4.5 Validation readings built with the data

From `v4_tools` and `v4_tools_masked`, by the diagnosis's rule (`scripts/s6-route4-diag.py make`):
`v4_tools_rw` and `v4_tools_masked_rw`, the same slates with one random distractor's description
replaced by its rewrite (field `follow`); and the renamed and natural validation records of the
tools group. Each is rendered for the model by D4.

### 4.6 Leakage rules

v2 §4.6 holds. No fresh request in any training or validation file (checked by identifier, exact
and normalised text, as v3's builder does). The 1,903 test-side requests of v2 that no file uses
(study §4.1) stay unused by v4.

## 5. The run (`scripts/train-route4.sh`)

| phase | budget (working minutes) | notes |
|---|---:|---|
| smoke | ≈ 3 | under a throwaway prefix; numbers not looked at; files deleted |
| pilot | 20 | micro from its pretraining init; the §4 data; schedule compressed to 20 minutes (QAT after minute 2); a checkpoint kept every 5 minutes |
| main | **120** | the same, QAT after minute 12, seed 0, one run, a checkpoint kept every 10 minutes, checkpoint `last` |
| calibration, export, τ_ask, the fit of β and τ_ask4 | ≈ 5 | `s2-calibrate` on `v4_tools`, ternary export, `gguf-set`, `scripts/s6-route4-fit.py` |
| evaluation | ≈ 15 | §7 |

- **One pilot, not two.** The size is fixed by the drop-in rule (§3.1), so v2's nano-or-micro
  rule has nothing to choose.
- **Read at every kept checkpoint of the pilot** (exported ternary, full depth, after its own
  calibration; validation only): `v4_tools` (model part, and the pair at β 0.81), `v4_tools_masked`
  (both), the share of `v4_tools_rw` and `v4_tools_masked_rw` slates whose answer is the rewritten
  distractor, the *none* answers on `v4_tools_masked`, the renamed against the natural validation
  records, and `v_tools` (reported). Written to `runs/s6-route4-pilot/summary.json`.
- **The pilot rule**, read at the pilot's last checkpoint:
  1. **abort** if the model part's `v4_tools` accuracy is below `potion-cos` on `v4_tools` − 0.15
     (v2's rule; `potion-cos` is scored on `v4_tools` by the frozen baseline code and the bar is
     written in an amendment before the pilot; on `v_tools` it was 0.7255 − 0.15 = 0.5755);
  2. **abort** if the rewritten distractor is the answer on more than **0.30** of the tool-gold
     slates of `v4_tools_masked_rw` (the published file: 0.8311 on `v_tools_masked`): the cue is
     still in the data.
  3. Nothing else is chosen on the pilot.
- **The main run's checkpoints** are read the same way and reported; none is chosen (`last` is the
  file).
- Brake modes, working minutes, restarts only after a technical failure (`RESUME=1`), as v2 §5.
  Folders `runs/s6-route4*`, `models/s6-route4`; nothing of v2's or v3's folders is written.

## 6. The fresh test files

### 6.1 Source

`Agent-Ark/Toucan-1.5M` at commit `0df3cf37f2abefb380370cfb02eabea2a35ae782`, config **SFT**
(`SFT/train-0000{0,1,2}-of-00003.parquet`, 425,496,735 bytes to download), Apache-2.0 (study §4.2,
§4.3). Rows of the subsets `single-turn-original` and `single-turn-diversify` with exactly one
entry in `target_tools`. The request is `question`; the gold is that tool, as listed in the row's
server metadata (`name: description`, the description cut by `cut_description`, key-like strings
masked); a row whose gold is not in its metadata is dropped and counted.

### 6.2 Build rule (written once, seed 1236)

1. Toucan's tools are embedded with v2's pinned all-MiniLM-L6-v2 and clustered by v2's rule among
   themselves. A row is dropped when its gold is within v2's cluster rule (same normalised name or
   MiniLM cosine ≥ 0.85) of any training or validation tool of v2; such a tool is never a
   distractor either.
2. Rows with the same normalised question: the first by sha256 of the text is kept.
3. At most 10 requests per gold cluster (v2's test cap) and **at most 40 per server**, the first
   by sha256 of the text.
4. **2,000 slates** drawn by seed among those left (all of them if fewer); 20% removed by v2's
   rule (8 options, the gold replaced by one more distractor).
5. A slate: the gold, the other tools of its server that pass v2's test distractor filter (not the
   gold's normalised name, MiniLM cosine below 0.8, not its cluster), then random Toucan tools of
   other servers by the same filter, **drawn within the gold's surface cell**, to 7 tools: the same
   name style (snake, camel, Pascal, spaced, other), the same end of description (a final period or
   not) and the same quintile of description length in words (quintiles over all Toucan tools);
   when the cell has too few tools, the nearest length quintile with the same style and end, then
   the same style alone. Shuffled; *none* last. No compared system chooses a distractor.
   **Why the cue of D7 cannot be in it.** Every option comes from one catalogue whose descriptions
   are the servers' own metadata, so there is no cross-source style to tell the gold from its fill;
   the fill from other servers matches the gold on the three features that carry the cue in v2's
   files (length, final period, name style), so the rules on them are at chance among those options
   by construction; the tools of the gold's own server share its source. The rules of §4.4 C2 are
   counted on the built file (option texts and the target's position only, no system) and written
   in the counts amendment before any scoring: a rule above its tolerance is reported with the gate
   and does not change the file.
6. Files: `t4_tools.jsonl`, `t4_tools_masked.jsonl` (every tool `tool_<k>`), `t4_tools_rw.jsonl`
   and `t4_tools_masked_rw.jsonl` (§6.4), `t4_tools.servers.jsonl` (the server and cluster of each
   slate). Hashes to `runs/s6-route4/test-files.json`.

### 6.3 Before any scoring

The counts (rows read, rows per filter, slates, servers, clusters, the share of the three largest
servers, *none* share, rows of each subset) go into an amendment before any system reads a fresh
file. **If fewer than 600 slates remain, v4 is not trained** (the interval of G1 would be about
twice v3's), and the amendment says so. The builder reads no output of any system. No fresh
request is sent to the teacher. `runs/s6-route4/FRESH-OPEN` is created after the counts amendment,
the frozen fit values and the engine's G4 bench on validation are committed.

### 6.4 The probe copies

`t4_tools_rw` and `t4_tools_masked_rw`: the tool-gold slates of `t4_tools` and of its masked copy,
one distractor drawn by seed given its rewrite, written by the teacher from the tool's name and
description alone (§4.2). The rewritten texts are used in these two files only.

## 7. Evaluation (ternary GGUF, full depth, exit thresholds 1)

- **Files:** T4-tools (gated: G1, G2, G3, G4), its masked copy and the two probe copies (G2),
  T2-tools of v3 (G5, a regression check on a file already read).
- **Systems:** the pair, its fast mode, the model part alone, `potion-cos-4bit`, `potion-cos`,
  `static-ft`, `pair-lr`, BM25, BM25 fused with `potion-cos-4bit` (β 0.5), `minilm-cos`,
  `toolrouter`, `name-prior`, and the published s6-route pair (v3) for comparison.
- **Metrics:** v2 §7.2 (accuracy, on tool-gold and *none*-gold slates, *none* precision, recall and
  F1, ECE in 15 bins, NLL, risk and coverage, accuracy and coverage at τ_ask4), the paired
  bootstrap of every gated difference (1,000 resamples of the slates, seed 0, percentile), and a
  bootstrap that resamples servers (seed 0).

## 8. Gate (every condition must hold)

| | condition | why |
|---|---|---|
| **G1** new family | on T4-tools: the pair − BM25 has the **lower bound of its 95% slate bootstrap above 0**; the pair − `potion-cos-4bit` has the **lower bound of its 95% slate bootstrap above 0**; the pair is above the model part alone. **Not a gate:** the pair − `minilm-cos`, reported with its interval in the report's sentence, whatever its sign | the claim v4 tests is "the best choice under 4 MB": above BM25, which costs no byte, and above its own 1 MB embedding, both beyond noise, on tool families it never saw. v3's bar against MiniLM is relaxed (§10) |
| **G2 (a)** no pull of the rewrite | on the tool-gold slates of `t4_tools_masked_rw` and of `t4_tools_rw`: the share whose answer is the rewritten distractor ≤ **1/7 + 0.05 = 0.1929**, and that share minus the share whose answer is the same distractor in the unchanged slate ≤ **+0.03** | the published file: 0.8311 under masked names; a model at 0.85 accuracy picks a given distractor on about 0.02, so the absolute bar alone would let a strong pull pass under real names |
| **G2 (b)** names that say nothing | on `t4_tools_masked`: the pair ≥ `potion-cos-4bit` on the same slates − **0.02** | v3's rule "never much worse than its better part", on the masked reading (v3: 0.2497 against 0.7280) |
| **G2 (c)** *none* under masked names | on `t4_tools_masked`: *none* answers in **0.10 to 0.30** of the slates (true share 0.20) | v2's G2 band (v3's model part: 0 of 945) |
| **G3** calibration | ECE ≤ **0.05** on T4-tools (15 bins) | as v3 |
| **G4** the engine | WASM against the Python reference on 200 T4-tools slates: max \|Δp\| ≤ 1e-3, the same answer on ≥ 199; WASM SIMD128 median ≤ **60 ms** per request (under (i): the model part encodes the options each time, the embedding part's tool list cached; under (ii'): the tool list cached); the two files ≤ 4,000,000 bytes | v3's G4; the bar stays 60 ms |
| **G5** no regression | on T2-tools (945): the pair ≥ v3's pair − **0.02** = 0.8868 − 0.02 = **0.8668** | T2-tools was read for v3's result: a regression check, not evidence; 0.02 is about two standard errors of a paired difference on 945 slates |

**The headline**, the one sentence v4 may say about tools of a new family, is allowed only if,
on T4-tools with the bootstrap that **resamples servers**: the pair − BM25 has a lower bound > 0,
**and** the pair − `potion-cos-4bit` has a lower bound > 0. (v3's advantage over BM25 did not
survive its cluster bootstrap.) **The report's sentence** gives the pair's accuracy on T4-tools,
its difference with BM25 and with the embedding part (with both intervals, slates and servers),
and its difference with `minilm-cos` with its interval, in whichever direction it falls. If G1 to
G5 hold and the headline bars do not, v4 is published as
passed with the sentence "not shown to beat BM25 (or its embedding part) on new families under
server resampling".

**Announced goals, not gates:** on T4-tools the pair ≥ `minilm-cos` − 0.03 (v3's G1 bar); T4-tools ≥ 0.80; ≥ 0.95 among answered at τ_ask4 with coverage ≥
0.80; the fast mode within 0.015 of full depth; renamed validation records no higher than natural
ones.

## 9. Reported, not gated

Every system on every file of §7; the pilot and every checkpoint reading; the data checks and
the surface table; the teacher's counts; the names-masked reading of the published pair on T4
(for the comparison); T4 without its three largest servers; the server bootstrap of every G1
difference; *none* answers by kind; the latency of each part, first and cached; the fast mode;
steps, draws per group, passes per record.

## 10. Disclosure (written into every place that reports v4)

v4 was designed after v2's test (T-tools and six other files) and v3's fresh files (T2-tools and
its masked copy, T2-metatool, T2-clinc, T2-massive) were read, and after a diagnosis that ran the
published model part, and the pair, on validation variants: renamed validation records as built
and with the gold's original description back, with the BM25 line shifted and removed, the masked
validation slates with the gold's or a distractor's description rewritten, with names rendered
`tool<k>`, by letters and by words, and the natural records masked; it also tabulated the
published per-slate outputs of v2's and v3's names-masked readings for *none* answers and final
periods, and computed no score on them. The validation slates `v_tools` have been read by v2's
pilots, by every fit of v3 and by the diagnosis. T2-tools was read for v3's result, so G5 is a
regression check. The fresh source was chosen from cards and papers, before any of its rows was
read. all-MiniLM-L6-v2 defines both the clusters that make "new family" and one of the gated bars.
The latency bar was set after v2 was measured.

**The bar against MiniLM is relaxed.** v3's G1 asked the pair to stay within 0.03 of
all-MiniLM-L6-v2 (91.6 MB). In v4 that condition is not a gate: it is a reported number, which the
report's sentence states in both directions, and an announced goal. The reason: on v3's own
T2-metatool reading, the one tool family outside its training families that v3 read, the pair was far
below MiniLM (0.5175 against 0.6570) and below its own embedding part (0.5340), and v4 is read on
tools of a family it never saw. The claim v4 tests is "the best choice under 4 MB": above BM25,
which costs no byte, and above its own 1 MB embedding, both beyond noise, on tool families it never
saw. A reader may hold v4 to v3's bar: the report gives the difference with MiniLM and its
interval, which lets them.

**The teacher saw the fresh test's tools.** In v2 and v3 the teacher saw no test tool. In v4 it
rewrites the descriptions of the tools that the fresh slates show, from each tool's name and
description alone, for the two probe copies of G2 (a) only. "The teacher never sees a test slate"
still holds, and it never sees a test request; "the teacher never sees a test tool" no longer does.

**A cue in the earlier test files.** The slates of v2's rule carry a request-blind cue: the right
tool's description is the slate's shortest far more often than chance (study §7). It was measured
after v2's and v3's results, by counting the option texts of their already-read test files, with no
model. v4's training, validation and test slates are built against it (D7, §6.2, item 5).

## 11. What a failure would mean

- **The data checks stop the build (§4.4):** the slate rule itself carries a cue that the remedy
  does not remove; no model is trained; the table is published, and the cue is described as a
  limit of v2's and v3's test files too (they were built by the same rule).
- **The pilot rule aborts on 2:** removing the rewrite from the gold did not remove the pull; the
  cue is elsewhere in the data. No main run.
- **G2 fails:** the shortcut is not gone, or the model still does not answer from the descriptions
  under masked names. Not published as a replacement; the published s6-route stays, with its
  disclosure.
- **The pair is below MiniLM − 0.03** (not a gate): stated in the report's sentence with its
  interval; a user with 91.6 MB to spend is told to use MiniLM on new families.
- **G1 fails against BM25:** on a new family the pair adds nothing to a score of no byte; the
  recommendation there is BM25; not published as a replacement.
- **G1 fails against the embedding part:** on a new family the 1 MB embedding alone routes as well;
  the recommendation there is the embedding file alone; not published as a replacement.
- **G3 or G4 fails alone:** the routing result is reported as held and the engine or calibration
  work as not done; not published as a replacement until it holds.
- **G5 fails alone:** the fix cost accuracy on the old families; not published as a replacement;
  both results are published.
- **In every case** the files are published as evaluated, with the verdict and the numbers first
  (owner's decision of 15/11/2026), and a Space (owner's decision of 17/11/2026).

## 12. Forbidden

- Scoring any system on a fresh file of v4 before the hashes, the counts amendment, the fit values
  and `FRESH-OPEN` exist; building T4 twice, or with another seed, cap or rule after a count is
  seen, except to fix a technical failure of the builder before any scoring (reported).
- Choosing anything on a fresh file: a checkpoint, β, τ, a temperature, θ, the depth, D4's option.
- Choosing D4's option on anything other than the latency check of §3.4; changing a tolerance of
  §4.4 after a check has run; adding a remedy other than the one of §4.4.
- Using T-tools, T2-tools or the 1,903 unused test-side requests of v2 in any parameter, data
  file or choice.
- Sending a fresh request, or any test request, to the teacher.
- A second seed or a second main run, except after a technical failure.
- Moving a bar of §8 after a fresh score is read.

## 13. Cost

| item | amount |
|---|---|
| teacher | ≈ 685 calls (≤ 13,691 rewrites), ≈ 1.5 to 2 h remote, one usage window; + ≈ 100 to 150 calls for the fresh tools' descriptions (probe copies only) |
| CPU | build ≈ 30 min; Toucan read, embedding and clustering ≈ 20 min; data checks ≈ 10 min; latency check ≈ 5 min; pilot readings ≈ 15 min; baselines on T4 and T2 ≈ 20 min |
| GPU | smoke 3 + pilot 20 + main 120 + calibration and export 5 + evaluation 15 ≈ **163 working minutes**; ≈ 23 if the pilot rule aborts |
| disk | Toucan SFT 0.43 GB (1.35 GB decoded, streamed); new build ≈ 0.5 GB; pilot checkpoints 5 × 64 MB, main 12 × 64 MB ≈ 1.1 GB; evaluation ≈ 0.1 GB: **≈ 2.2 GB**, against 9.1 GiB free on 17/11/2026 |

## 14. Code to write before any run (by file)

1. `crates/sky-data/src/route/build.rs`: D1 (`rewrite_draw`, every kind), the renamed kind by
   `test_slate` and `masked` with `removed_set` per group and split, D5 (removed slates refilled),
   D7 (the source-first fill, the natural slates on a stream of their own), the surface-cell fill of
   the remedy, `v4_tools` and its copies, the *none* share per kind, the rendering of D4 for the
   training records and `render`; unit tests: P(rewritten) independent of the gold on an invented
   build, *none* 0.20 of the renamed slates, 8 options in every renamed slate, the false-negative
   answers still found by id, determinism under the seed.
2. `crates/sky-data/src/route/toucan.rs` (new) and `sky-data route toucan`: the parquet reader,
   the filters, the cluster rule against v2's tools, the caps, the surface-cell fill, the slates,
   the items of the fresh-tool rewrite job (a name and a description only), the probe copies,
   written once with hashes and counts; tests on an invented set; `crates/sky-data/src/teacher.rs`:
   a `fresh-tools` item kind that the guard accepts only when an item holds a name and a
   description and nothing else.
3. `crates/sky-infer/src/route.rs`: `options_with_scores` (D4) and its fixture
   `tests/fixtures/route_bm25_v4.json`; the engine reads `skycmd-s2.route.render`;
   `crates/sky-web/src/router.rs` and `sky-route fuse` render by it (v2's file unchanged in
   behaviour); (ii') written only if §3.4 takes it.
4. `crates/sky-web/bench/route3-node.mjs`: an uncached mode (§3.4, G4).
5. `crates/sky-train`: `s2-decide --keep-checkpoints` (a copy of each periodic save under
   `ckpt-<minutes>`).
6. `scripts/s6-route4-checks.py` and its tests: C1 to C4, with the rules of
   `scripts/s6-route4-blind.py`.
7. `scripts/train-route4.sh` (smoke, pilot with its readings and rule, main, export, fit),
   `scripts/s6-route4-fit.py` (β and τ_ask4, from `scripts/s6-route3-fit.py`), `scripts/route-eval.sh`
   and `scripts/route-baselines.py` (a `t4` split refused until `FRESH-OPEN`),
   `scripts/s6-route4-gate.py` and `scripts/test_s6_route4_gate.py` (§8, the headline, the
   readings).
8. `scripts/route/pins.json`, `DATA_LICENSES.md` (Toucan-1.5M: Apache-2.0, test only, its requests
   written by Mistral-Small, DevStral-Small, GPT-OSS, Kimi-K2 and Qwen3-32B; the new rewrites).

## 15. Choices (the value taken at the freeze; the alternative is recorded)

| | choice | taken | alternatives | why |
|---|---|---|---|---|
| O1 | D4 rendering | (i) scores in the option text, if §3.4's check passes | (ii') engine labels; (ii) `tool<k>` names | (i) removes the binding; its cost is measured within the bar |
| O2 | latency bar of the check | 50 ms | 55 ms; 45 ms | 10 ms under G4's bar, the spread between v2's and v3's readings |
| O3 | D5 (removed slates keep their count) | taken by the lead | v2's rule | the test's removed rule; removes a prior the test contradicts |
| O4 | where D1 applies | training and validation records; test-like validation slates and test files keep the sources' text | every file, the test included | β, τ and the gate read real descriptions, as a user's |
| O5 | initialisation | s6-micro pretraining init | v2's trained weights (cheaper, carries the shortcut) | the cue is in v2's weights |
| O6 | pilots | one, micro, 20 min, checkpoints every 5 min | v2's two pilots | the size is fixed by the drop-in rule |
| O7 | pilot abort on the probe | > 0.30 | no such rule; > 0.1929 | twice G2's bar: a pull that large after 20 minutes is a cue in the data |
| O8 | main budget | 120 working minutes | 90; 180 | v2's |
| O9 | fresh source | Toucan-1.5M SFT, single-tool rows | MCPToolBench++ (MIT on GitHub, 87 tools, five languages); MCP-Flow is out (no licence; its authors are not asked) | §4.3 of the study |
| O10 | T4 size and caps | 2,000 slates; 10 per gold cluster, 40 per server | 1,000; no server cap | v2's size; no server may dominate as three clusters dominated T2 |
| O11 | *none* in T4 | v2's removed rule (20%) | Toucan's `irrelevant` subset | the same meaning of *none* in every file |
| O12 | G2 (a) | absolute ≤ 0.1929 and paired ≤ +0.03 | the absolute bar alone | §8 |
| O17 | G1 against MiniLM | reported in the report's sentence, and an announced goal (≥ − 0.03) | gated as in v3 | §10 |
| O18 | the teacher and the fresh tools | their descriptions rewritten, for the probe copies only | the probe read on validation only | §4.2, §10 |
| O19 | the fill | D7, source first, in every kind; the surface cell in the fresh test | v2's fill | study §7 |
| O20 | the validation slates of the fits and the pilot | `v4_tools` (D7's fill) | `v_tools` | `v_tools` carries the cue; the fresh test does not |
| O13 | G5 margin | 0.02 | 0.01; 0.03 | about two standard errors on 945 slates |
| O14 | headline bootstrap | servers resampled | slates resampled | v3's advantage over BM25 did not survive the cluster bootstrap |
| O15 | data-check tolerance | c + 0.02 + 2 SE; C1 0.01 / 0.03 | c + 0.01 + 2 SE | today's cue is +0.13 above chance; 0.02 leaves room for sampling and nothing like it |
| O16 | minimum T4 size | 600 slates | 1,000; none | below it G1's interval is about twice v3's |

## 16. The lead's decisions on the draft (17/11/2026)

1. The teacher may rewrite the descriptions of the new test's tools, for the G2 probe copies only;
   it sees descriptions alone, never a request or a slate (§4.2, §6.4, §10).
2. The bar against `minilm-cos` is not a gate of v4: a reported number, stated in the report's
   sentence in both directions, and an announced goal (§8, §10).
3. D5 is taken.
4. "Both sides" of D1 means the training and validation records (O4).
5. MCP-Flow is out; its authors are not asked.
6. The other proposals of the draft stand as written: rendering (i) with the 50 ms check and the
   engine-label fallback; s6-micro's pretraining init; one 20-minute pilot with the probe abort at
   0.30; 120 working minutes; Toucan with the caps and the 600-slate floor; G2 (a) with both
   conditions, G2 (b), G2 (c); G5 at 0.8668; the headline's two lower bounds with servers resampled;
   the tolerances of the checks.
7. After the measurement of study §7 (a request-blind cue in v2's and v3's test files), the
   source-first fill is part of the build (D7), and the fresh test is built against the cue
   (§6.2, item 5). The validation slates of the fits and the pilot follow (O20).

## 17. Amendments

(none)
