# s6-route v4 (the router with its shortcut removed): design study

Written on 17/11/2026, after `docs/s6-route4-diagnosis.md`, to prepare `docs/s6-route4-prereg.md`.
Read only: the repository, its run folders, the data folder of v2 (`skycmd-data/s6-route/`, text
files only, never written) and the web (dataset cards, LICENSE files, papers and file listings;
no dataset was downloaded). Nothing was trained, no model was run and nothing was evaluated: every
count below is a count of records, tools or files, made by reading them. Paths that start with
`build/` are in `skycmd-data/s6-route/`. Text marked **[Inference]** is reasoning, not a sourced
fact. "Not verified" means looked for and not confirmed.

## 0. Summary

1. **The defect** (the diagnosis, not re-measured here). In every training slate of v2's
   *renamed* kind the right tool is the only option whose description the teacher rewrote, and
   *none* is never right. Under `tool_<k>` names the model part picks the most rewrite-like
   description (a rewritten distractor is picked on 0.8311 of masked validation slates) and almost
   never answers *none* (1 of 1,144). It never reads the BM25 line under such names. Under real
   names the line is its main input (0.8339 with it, 0.4318 without). No change of rendering
   repairs the published weights.
2. **What the data build must change, read in the code.** The 14,263 rewrites of v2 cover the gold
   tools only. Of the 27,707 distinct tools that appear as options in v2's training and validation
   records, **13,448 have no rewrite**; counting the tools of the training and validation pools
   that no record shows, a complete set needs **at most 13,691 more rewrites: about 685 calls of 20
   items** to GLM-5.3-Flash, about 1.5 to 2 hours of remote time inside one 5-hour usage window of
   the endpoint. The false-negative filter does not need to run again if the plans of the natural
   and hard slates are kept. Renamed slates by the test-slate rule and removed slates under renamed
   names need no teacher call.
3. **Binding the lexical score to its option.** Proposed: **write each option's BM25 score and
   rank in its own text** (option (i) of the diagnosis). It removes the binding instead of asking
   the model to learn it. Its cost is the static cache of the model part: every request becomes a
   "first request". v3 measured that path at **45.2 ms** for the model part (WASM SIMD128, on a
   machine in use; 30.8 ms with the cache), under the 60 ms bar of v3's G4. The alternative keeps
   the line and keys it by a label the engine generates; it costs no latency but relies on a
   binding no rendering of v2 ever needed.
4. **Fresh test data.** v3's T2-tools took every test-side request of v2 up to a cap of 100 per
   cluster. **1,903 test-side requests remain unread, all in the three largest clusters** (1,035,
   459 and 409 over the cap), whose tools v2 and v3 have both read. They cannot make a test of new
   tools. v4's test needs a source the project has never used. Of seventeen candidates read
   (§4.2), **Toucan-1.5M** (Apache-2.0, not gated, 495 real MCP servers, "2,000+ tools", requests
   written by five open-weight models) is the one proposed; MCP-Flow fits the shape better but has
   no licence.
5. **Odds** (§5): about 85% that the data fix removes the shortcut by the diagnosis's measure;
   about 10 to 15% that every gate condition holds, the main risk being v3's G1 bar against
   all-MiniLM-L6-v2 on a family of tools never trained on (on MetaTool, the one such family read so
   far, the published pair was 0.14 below MiniLM and below its own embedding part).

## 1. The defect and its evidence (from `docs/s6-route4-diagnosis.md`, read in full)

| reading (validation; the published model part at full depth) | n | accuracy | *none* answers |
|---|---:|---:|---:|
| `v_tools` (real names) | 1,144 | 0.8339 | 0.1469 |
| `v_tools_masked` (`tool_<k>` names) | 1,144 | 0.1171 | 0.0009 |
| renamed validation records (the gold's description is the rewrite) | 1,116 | 0.9498 | 0.0000 |
| the same, the gold's original description put back | 1,116 | 0.3271 | 0.0287 |
| `v_tools_masked`, the gold's description rewritten | 915 | 0.9082 | 0.0000 |
| `v_tools_masked`, one random distractor rewritten | 1,113 | 0.0270; that distractor picked on **0.8311** | 0.0000 |

- **Mechanism.** Two differences between training `renamed` records and the masked reading:
  the gold's description (the teacher's rewrite in training, the source's own in the reading) and
  *none* (never the gold of a renamed record, 0 of 51,101; the gold of 20% of the masked slates).
  Names, k, the BM25 line's keys and order, when the line is computed and truncation are the same
  in both (diagnosis §2 (a)).
- **The line is not read under masked names.** Shifting the line by one entry on renamed records:
  0.9489 against 0.9498. Removing it: 0.8226. With real names, removing it takes `v_tools` from
  0.8339 to 0.4318 (diagnosis §2 (b)).
- **Why masked names cannot carry the line today.** `_` is a BM25 word character, so `tool_75`
  is one BM25 token no request holds. In the model's tokens, the digits of `tool_75` follow `_`,
  not an ASCII letter, so the numeric extraction takes them (`crates/sky-infer/src/num.rs:46-49`,
  the rule; `:67`, the glue test) and `fuse_numbers` moves the value onto the token before
  (`crates/sky-infer/src/s2/prompt.rs:70-97`). Every masked name, in the options and in the line,
  becomes the same token sequence that differs only by a value in the number channel.
- **A request-blind rule** (the option ending with a period, then the one whose first word ends in
  -s) gets 0.502 on training renamed tool slates and 0.774 on renamed intent slates, against 0.371
  and 0.369 for chance among the tools. On the masked validation slates, right tools end with a
  period on 0.337, all options on 0.575: the right tool of a test-like slate reads *less* like a
  rewrite than its distractors, and the same rule falls to 0.078 there, under 1/7 (diagnosis
  §2 (d)). **[Inference]** The inverse of that rule would beat chance on the test-like slates: a
  cue of the source's style, not of the teacher's, that the slate rule itself creates (§2.5).
- **Not measured by the diagnosis:** whether a polished distractor description attracts the model
  under real names; the When2Call and intents renamed validation records; the fast mode.

## 2. What the data build must change (read in `crates/sky-data/src/route/`)

### 2.1 Where the rewrite comes from today

- The rewrite items are written by `split()`: one per gold tool of a training or validation plan
  (`build.rs:1016-1037`; the `rw` map is filled from `p.gold` only). No distractor and no
  non-gold tool of a pool is ever sent.
- `build_slates()` reads the answers (`build.rs:1348-1352`) and uses them in one place: the
  renamed slate, where the gold's description is the rewrite cut by `cut_description` and every
  other option keeps the source's (`build.rs:1499-1520`). Amendment 1 of v2 says the opposite ("the
  descriptions are untouched"); reading 13 of `docs/s6-nano2-readings.md` and the code agree with
  each other.
- The renamed slate is slate 1 (the natural slate) renamed: 2 to 8 options, the gold always there.
  The removed kind is never renamed (`build.rs:1471-1527`).
- For the intents, the source's description is itself the teacher's (`describe-intent`, from the
  intent name), and the rewrite is a second teacher text. Both are machine text; the style rule
  still separates them (0.774 on renamed intent slates), because the rewrite prompt writes a
  different form (first word in -s on 0.726 of rewrites, 0.046 of the intent descriptions).

### 2.2 Tools without a rewrite (counted on 17/11/2026 from `build/records/*/*.jsonl`, `build/teacher/rewrite-description.jsonl` and `build/counts/split.json`)

| records | distinct tools shown as options | with a rewrite | without | option slots with a rewrite |
|---|---:|---:|---:|---:|
| tools / train (81,596 records) | 25,696 | 13,386 | **12,310** | 207,093 of 317,300 |
| tools / val (4,185) | 1,498 | 658 | **840** | 9,255 of 16,125 |
| When2Call / train (14,621) | 14,360 | 7,140 | **7,220** | 34,511 of 58,368 |
| When2Call / val (896) | 1,982 | 1,022 | **960** | 2,085 of 3,537 |
| intents / train and val | 123 | 123 | 0 | all |
| **all training records** | 26,197 | | **12,596** | |
| **all validation records** | 3,498 | | **1,731** | |
| **training and validation, together** | **27,707** | 14,259 | **13,448** | |
| validation slates `v_tools` and `v_tools_masked` | 1,538 | 657 | 881 | |

- Every gold has a rewrite (0 golds without one in every file): v2's job was complete for what it
  was asked.
- The pools from which the slates draw are larger than what the records show: 26,277 training
  tools and 1,550 validation tools of route/tools and When2Call (`counts/split.json`, "tools by
  split"), and the 123 intents. 27,950 tools in all, of which 14,259 have a rewrite: **at most
  13,691 rewrites are missing for a complete set** (between 13,448, the tools that records show,
  and 13,691, every tool of the two pools). A slate built by a new rule can show a tool that no v2
  record shows, so the complete set is the one to ask for.
- The 3,317 test-side tools of v2 (`counts/split.json`) need no rewrite: the probe files of v4 are
  built on its own fresh test (§4, and the pre-registration, which asks the lead to decide whether
  the teacher may see the fresh test tools' descriptions).

### 2.3 Teacher calls and quota (facts in the repository)

- **The teacher.** GLM-5.3-Flash (open weights, MIT, `https://huggingface.co/zai-org/GLM-5.3-Flash`),
  through `sky-data teacher rewrite-description` and `scripts/route-teacher.sh`: 20 items per call
  (`BATCH=20`), 4 workers (`JOBS=4`), at least 1.5 s between calls of a worker
  (`MIN_INTERVAL_MS=1500`), at most 60,000 calls per pass; a job refuses an item file that holds a
  test item. The rules of use are amendments 1 and 3 of `docs/s6-gen-prereg.md` (moderation split,
  resumable passes).
- **Throughput.** "About 2.5 labels per second whatever the number of workers"; beyond 4 workers
  the endpoint throttles and calls go to retries (`docs/s6-gen-prereg.md`, amendment 1, measured on
  29/10/2026).
- **Quota.** The endpoint's usage window resets every 5 hours; the loop pauses 1,800 s after a
  pass that answers nothing and gives up after 24 such pauses (`scripts/route-teacher.sh:15-16, 85`,
  `scripts/distill-glm.sh:15-16`). No other quota figure is written in the repository.
- **What v2's job cost.** 14,263 rewrite items, about 714 calls; the item file was written at
  18:14 and the answer file at 20:14 on 15/11/2026 (file times in `build/teacher/`), while the
  false-negative job ran in parallel; 0 refused. v2's three jobs answered 4,174 calls with no failed
  batch and 2 items refused by moderation (amendment 2 of v2).
- **For v4.** About 13,700 items, **about 685 calls**: at 2.5 items per second about 91 minutes;
  by v2's own job about 2 hours. One usage window suffices unless the endpoint throttles.
- **What does not need the teacher.** The false-negative filter: its item ids hash the request and
  the hard slate of the plan (`fn_id`, `build.rs:112-116`). If v4 keeps the plan step's random
  streams (`route/plan/<id>`, `build.rs:911`) the 69,290 answers apply unchanged. The renamed and
  renamed-removed slates of §2.4 draw random distractors, like the test, and are not filtered (the
  test was not either).

### 2.4 Renamed slates by the test-slate rule, and removed slates under renamed names

- **The rule exists.** `slates::test_slate` (`slates.rs:131-157`): the gold, the conversation's
  own tools, random tools of the split's pool to 7 tools, shuffled, *none* last; a removed slate
  drops the gold and takes one more distractor by the same rule, so it keeps 8 options.
  `tool_slates` (`build.rs:389-527`) applies it to the validation and test sides of route/tools,
  with the distractor filter of v2 §4.4 (not the gold's cluster, not its normalised name, MiniLM
  cosine below 0.8) and 20% removed slates drawn by `removed_set`; `slates::masked`
  (`slates.rs:161-169`) then renames every tool `tool_<k>`. That pair of calls is exactly the
  names-masked reading.
- **What v4 needs.** For every training and validation request with a tool gold, one renamed slate
  built by those two calls on the request's own pool (route/tools and When2Call: the split's pool,
  as now; intents: the source's intents), 20% of them removed by `removed_set` with a seeded label
  per group and split, then each tool option's description drawn as the rewrite with probability
  0.5 (§2.6). The intents and When2Call need the distractor filter's inputs: their tools are
  embedded already (`stage/texts.jsonl` holds every tool of a non-test request).
- **Counts this would give.** One renamed slate per tool-gold request: 21,759 (tools), 3,899
  (When2Call) and 25,443 (intents) in training, as v2's renamed kind; 20% of them, about 10,220,
  with *none* right, where v2 has 0.
- **The share of *none*.** `build_slates` fixes the share per group and split by counting the
  tool-gold slates as `fixed` and trimming or adding removed slates (`build.rs:1529-1587`). A
  renamed-removed slate must count apart, so that the 20% holds both within the `tool_<k>` slates
  and over every slate of a group and split.

### 2.5 Two cues the decided changes leave (found in the code; to be settled before the build)

- **The number of options.** A natural or hard slate has 2 to 8 options, uniform
  (`slates.rs:59-62`); its removed copy is the same slate without the gold
  (`Slate::without_gold`, `slates.rs:49-56`), so it has one option fewer, and an 8-option training
  slate never has *none* right. **[Inference]** From the counts of `build_slates`, *none* is right
  on about 0.23 of the 2- to 7-option slates and on 0 of the 8-option slates of these kinds. A
  test slate always has 8 options and *none* right on 20% of them. No request-blind rule on the
  count beats the majority answer (the share of *none* is below 0.5 everywhere), so an accuracy
  check would not see it, but it is a prior the model can learn and that the test contradicts.
  The test's removed rule (one more distractor of the same rule) removes it; for the hard slates
  the extra distractor would be the next tool of the plan's reserve (`build.rs:952-956`), as the
  false-negative filter already does.
- **The source's style.** In a test-like slate the gold and the conversation's own tools come
  from the request's source; the random fill comes from the whole pool, where Seal-Tools and
  ToolACE dominate (10,381 and 7,312 of the 22,376 training tool requests,
  `counts/split.json`). Glaive's tools are snake_case names with imperative descriptions without a
  final period; ToolACE's are spaced names ("Get Guardian Raids") with full sentences. The
  diagnosis's numbers (0.337 of right tools against 0.575 of options end with a period on the
  masked validation slates) say the gold differs in style from its distractors. A rule that picks
  the option whose style is the slate's minority would be above chance; it was not measured. The
  same holds in v2's and v3's test files, built by the same rule. A test drawn from one new
  catalogue has no such cue. The remedy, if the check of the pre-registration (§4 there) finds the
  cue: fill from the gold's own source first.

### 2.6 The rewrite made independent of the answer

- The decided rule: each tool option's description is the rewrite with probability 0.5, drawn
  per (request, slate kind, tool) by a seeded stream that never reads the gold, in every kind of
  slate, in training and validation records. The rewrite is cut by `cut_description`, as v2 cut
  it (`build.rs:1507`).
- What it requires: a rewrite for every tool a slate can show (§2.2), else a tool without one is
  always original and the rule is no longer independent of the tool's source (the golds of v2 all
  have one).
- The check without a model: P(rewritten | gold) = P(rewritten | other option) in every kind,
  group and split, and the request-blind rules at chance (pre-registration §4).
- **The test-like validation slates** (`v_tools`, `v_tools_masked`) are what β, τ_ask and the
  temperatures are fitted on and what the pilot is read on. **[Inference]** They should keep the
  sources' descriptions, as every test file does, and the rewrites go to two probe copies of them
  (one distractor rewritten, real and masked names), as the diagnosis built them.

## 3. Binding the lexical score to its option

### 3.1 How the engine caches the options (read in `crates/sky-infer`)

- The s2 engine keeps the encoding of the static stream (the instruction and the options, with the
  question keys and values of every fusion block) in an LRU of `static_cache` entries, 16 by
  default (`crates/sky-infer/src/s2/engine.rs:4, 69-82, 368-414`); 0 encodes the static stream
  each time. The top-level `EngineOptions.prefix_cache` sets it (`crates/sky-infer/src/engine.rs:409`).
- `sky_infer::route` puts the scores in the state and not in the options for that reason: "a score
  that depends on the request would make every option's text, hence its encoding, change with
  each request" (`crates/sky-infer/src/route.rs:6-11`).
- The router of `sky-web` encodes the tool list once on both parts and reports `static_hit` and
  `tools_hit` (`crates/sky-web/src/router.rs:7, 37-40, 70-99`).
- `sky-eval --latency` times a fresh engine with the default options and one with
  `prefix_cache: 0, state_cache: 0` and writes `latency.default` and `latency.no_cache`
  (`crates/sky-infer/src/bin/sky-eval.rs:37-39, 277-293`). Since every slate of a slate file has
  its own tool list, the default path does not hit the static cache either: on v2's test the two
  medians are 24,031 and 24,377 µs, on v3's T2-tools 24,648 and 24,802 µs
  (`runs/s6-nano2/eval-test/latency/native.json`, `runs/s6-nano3/eval-fresh/latency/native.json`).
  The native numbers of v2 and v3 are uncached numbers.
- `crates/sky-web/bench/route3-node.mjs` routes each slate twice: the first call encodes the tool
  list ("first request"), the second, with " please" appended to the request, hits the cache.

### 3.2 Latency, as measured (Apple M5, Node, 200 slates)

| | model part, first request | model part, cached | pair, first | pair, cached |
|---|---:|---:|---:|---:|
| v2, T-tools, SIMD128 (`runs/s6-nano2/eval-test/latency/wasm.json`) | 50,922 µs | 38,929 µs | | |
| v2, T-tools, relaxed SIMD | 33,204 µs | 23,532 µs | | |
| v3, T2-tools, SIMD128 (`wasm3.json`; machine in use, a GPU run active) | 45,220 µs | 30,838 µs | 45,279 µs | 30,879 µs |
| v3, T2-tools, relaxed SIMD (`wasm3-relaxed.json`) | | | 34,316 µs | 23,536 µs |
| v3, fast mode, SIMD128 (`wasm3-fast.json`) | | | 25,023 µs | 10,547 µs |

The embedding part costs 21 µs cached and 69 µs on a new tool list. The bar of v3's G4 is 60 ms,
with the tool list cached.

### 3.3 The two options of the diagnosis, and a variant

| | (i) score and rank in each option's text | (ii) the line kept, keyed by a text that stays distinct | (ii') the line kept, keyed by a label the engine writes |
|---|---|---|---|
| what changes | the option text carries `score <s> rank <r>` before `name: description`; the state is the request alone | masked names rendered `tool75` or with letters; the line keyed by name, as now | the engine prefixes each tool option with a short label of its position (for example `t1` … `t7`) and keys the line by it; names stay as given |
| binding the model must learn | none: the score is in the option's own tokens, which its marker attends to | by name tokens, never needed by v2 under masked names | by a label token; v2 binds by distinct name tokens under real names (0.8339 with the line, 0.4318 without) |
| latency | the static cache of the model part never hits: every request is a first request, 45.2 ms (v3, SIMD128, loaded machine), 50.9 ms (v2, its own conditions), 34.3 ms relaxed, 25.0 ms fast mode; plus about 4 to 6 tokens per option, not measured | none | none |
| real names that collide | not an issue | names that differ only by a number not glued to a letter collide (`search_2`): 1,517 of 230,171 tool options of v2's training tool records that are not renamed, 226 of 20,850 distinct names (diagnosis §2 (d), item 4) | not an issue |
| head window | about 35 more head tokens with 7 tools; v2's rule gave micro 256 at a shrunk share of 0.0409, under the 5% bar by 0.009; a longer option may move it to 320 (0.0378 today) | unchanged | a few tokens more |
| check before training, validation only | the WASM bench of v2's file (the same graph; the weights do not change the time) on 200 validation slates rendered by (i), first-request path, no training running | a tokenizer check: in every slate the line's keys are distinct token sequences after extraction and fusion (fails for `tool_<k>`) | the same tokenizer check on the labels |

- **Proposed: (i).** It is the only option that does not depend on the model learning a binding,
  which is the class of defect v4 repairs. Its cost is measured on the same graph and fits v3's
  bar with a margin of 9 to 15 ms. The 48-token cap of each option (`OPTION_MAX_TOKENS`, amendment
  2 of v2) cuts the end of a long option: the score and rank go first so the cap never removes
  them. Two numbers follow a word each (`score`, `rank`), as in today's line, so the fusion of
  numbers puts them on those words (`route.rs:17-19`).
- **What it costs the product.** "The tool list is encoded once" stops being true for the model
  part; the embedding part keeps its cache. A user who routes many requests against one list pays
  about 45 ms each instead of about 31 ms.
- **The decisive check** (pre-registration §3.4): the median first-request latency of v2's file in
  WASM SIMD128, on 200 validation slates rendered by (i), with no training running, at most
  **50,000 µs** (10 ms under the bar, about the spread between v2's and v3's measurements of the
  same path). If it fails, (ii') is taken, by a rule written before the build.
- **Embedding part.** It reads `name: description` as now (the engine hands it the option text
  without the scores).

### 3.4 Tokenizer facts (what happens to digits in names)

- The numeric extraction (`num.rs:46-49`): every match of `-?\d+(?:\.\d+)?` not glued to a
  preceding ASCII letter becomes `[NUM]` with its value; a match glued to a letter (`b2`, `glm-5`,
  `tool75`) stays text. With `fuse_numbers` on (the model's config), a `[NUM]` whose previous token
  carries no value is removed and its value moves onto that token (`prompt.rs:70-97`).
- So `tool_75` is `tool_` then a value; `search_2` and `search_3` are the same tokens with two
  values; `tool75` and `t3` are text, split by the BPE (the exact split under `tok-s6-4096` was not
  printed: not verified).
- BM25 tokens are runs of two or more word characters, `_` included (`route.rs:21-28, 52-60`):
  `tool_75` is one BM25 token, no request holds it, and it adds one token to the document's length.
- Under (i) none of this matters for the binding; it still matters for what a name says. Under
  (ii) it is why `tool_<k>` cannot key the line. Under (ii') the labels must be checked.

## 4. Fresh test data

### 4.1 What is left of v2's test side

- v3's T2-tools (`crates/sky-data/src/route/fresh.rs`, `runs/s6-nano3/counts.json`): 2,861
  candidates (2,541 over v2's cap of 10 per cluster, 320 left out of v2's 2,000); 13 excluded as
  already in a v2 slate file (12 by text, 1 by normalised text); **1,903 excluded by the fresh cap
  of 100 per cluster**; 945 kept. The ten largest clusters held 1,135, 559, 509, 96, 70, 47, 34, 20,
  17 and 12 fresh requests before the cap: only the first three were over it.
- **So 1,903 test-side requests remain that no system has read: 1,035, 459 and 409 in the three
  largest clusters**, the three that already hold 31.7% of T2-tools and on which v3's advantage over
  BM25 leans (without them, +0.0202 [−0.0062, +0.0481]). Their tools are v2's test tools, read by
  v2's test and v3's. No other test-side request of route/tools is left: every other cluster had
  at most 96 candidates and kept them all.
- Other test-only sources of v2 have unread rows (MetaTool: 20,614 staged single-tool queries, of
  which 4,000 were drawn by v2 and v3; CLINC150: 5,500 test queries, 4,000 drawn), but their tools
  (about 199 MetaTool plugins, 150 CLINC intents) were read twice.
- **Conclusion.** v4's test of tools never seen needs new tools and requests from a source the
  project has never used.

### 4.2 Candidates (cards, LICENSE files and papers read on 17/11/2026; no data file downloaded)

The rule: real tool or API catalogues with user requests, under a licence that allows use and
redistribution (Apache-2.0, MIT, CC-BY); research-only, non-commercial or unstated licences are
out; Model Context Protocol (MCP) server tool lists preferred, since routing among MCP tools is the
use case; a single right tool per request, or a subset where that holds.

| candidate | licence, as read | size | one right tool? | language | how the requests were made | gated | ruling |
|---|---|---|---|---|---|---|---|
| **Toucan-1.5M** (`Agent-Ark/Toucan-1.5M`, commit `0df3cf37`, modified 04/10/2025; arXiv 2510.01179; github.com/TheAgentArk/Toucan) | **Apache-2.0** on the card ("This dataset is released under Apache 2.0") and in its metadata; the GitHub repository shows MIT (its LICENSE text not read) | 1,646,546 rows in four configs (Kimi-K2 518,516; OSS 457,130; Qwen3 551,613; **SFT 119,287**); 495 MCP servers from Smithery and GitHub, "2,000+ tools"; SFT: 3 parquet files, download 425,496,735 bytes; all configs 21.8 GB | each row has a `question` and its `target_tools`, the tools used as seeds of the question (1 to 3; "N = 3"); the single-tool rows are not counted anywhere | not stated; the card's examples are English | written by five open-weight models (Mistral-Small, DevStral-Small, GPT-OSS, Kimi-K2, Qwen3-32B), filtered by Kimi-K2 as judge, trajectories judged by GPT-OSS-120B; "only tasks and annotations in Toucan were generated with LLMs"; humans checked 50 samples; tool descriptions are the servers' own metadata | no | **IN, proposed** |
| MCP-Flow (`wwh0411/MCP-Flow`; arXiv 2510.24284) | **none**: no LICENSE in the repository (GitHub API: null), none on the card | 1,166 servers, 11,536 tools, 68,733 pairs; splits train 52,169, seen test 5,216, unseen tool 5,249, unseen server 6,099; test files already as 10-tool slates | yes, one target tool each | not stated, English examples | LLM-generated; the paper names GPT-4o (proprietary) as its generation model; 300 test samples checked by humans | no | OUT (no licence); the best fit in shape, worth one message to its authors |
| MCPToolBench++ (github.com/mcp-tool-bench/MCPToolBenchPP; arXiv 2508.07575) | MIT in the GitHub LICENSE; the Hugging Face card states none | 1,509 instances, 87 tools, 6 domains | single and multi-step mixed (chains of up to 10 calls) | English, Chinese, French, Russian, Spanish | LLM from templates with slot values, the model not named, no human check described | no | IN, weak (87 tools, mixed languages, model unnamed); not proposed |
| quotientai/limbic-eval-tool-use-mcp | MIT | 9,813 rows of Smithery tool definitions with a conversation, labelled correct / incorrect tool / incorrect parameters | no gold-tool field: a gold would have to be rebuilt from the "correct" rows | English | "synthetically generated", model not named | no | IN, weak; not proposed |
| MCP-Zero tool collection (github.com/xfey/MCP-Zero; arXiv 2506.01056) | MIT (LICENSE read) | 308 servers, 2,797 tools | no request set released | — | server summaries by Qwen2.5-72B; embeddings by an OpenAI model | not on the Hub | OUT as a test (no requests); a usable MIT tool pool |
| LiveMCPBench (`ICIP/LiveMCPBench`; arXiv 2508.01780) | Apache-2.0 (LICENSE and card) | 95 tasks; LiveMCPTool 70 servers, 527 tools | 1 to 7 tools, 1 to 5 steps | English, some Chinese | not stated | no | OUT (95 tasks, multi-step) |
| MCP-Universe (Salesforce; arXiv 2508.14704) | Apache-2.0 (LICENSE.txt) | 11 servers, 133 tools, 231 tasks | long multi-step tasks | English | written by the authors | not on the Hub | OUT (multi-step, small) |
| MCP-Bench (Accenture; arXiv 2508.20453) | unclear: the README badge says Apache 2.0, no LICENSE file (404; GitHub API: null) | 28 servers, 250 tools, 104 tasks | multi-step | English | o4-mini (proprietary), checked by humans | not on the Hub | OUT (licence unclear, multi-step, small) |
| MCP-Atlas (`ScaleAI/MCP-Atlas`; arXiv 2602.00933) | CC-BY-4.0 (card) | 36 servers, 220 tools, 500 tasks | 3 to 6 calls per task | English | written and checked by human experts | no | OUT for single-tool routing; the best human-written source if multi-tool tasks matter later |
| MCPVerse (github.com/hailsham/mcpverse; arXiv 2508.16260) | Apache-2.0 (LICENSE present, text not read) | 65 servers, 552 tools, 250 tasks | mixed | not stated | human annotators helped by tools | not on the Hub | OUT (small) |
| ToolRet (`mangopy/ToolRet-Queries`, `-Tools`; arXiv 2503.01763) | none on either card | 7,961 queries, 44,453 tools, 35 subsets | mixed | mixed | repackages ToolACE, APIGen, ToolBench, API-Bank, Gorilla, MetaTool… | no | OUT (no licence; mostly sets the project used) |
| HumanMCP (arXiv 2602.23367) | no dataset release or licence found | MCP-Zero's tools; 500-query evaluation set mentioned | — | English | gpt-4o-mini with gpt-4o as critic | — | OUT |
| NexusRaven API evaluation | CC-BY-NC-4.0 | — | — | — | — | — | OUT (non-commercial) |
| ToolLens (COLT) | none (GitHub API: null) | 18,770 queries, 464 tools, from ToolBench | 1 to 3 | English | derived from ToolBench | — | OUT |
| MCPAgentBench, MCP-AgentBench, MCP-RADAR, ScaleMCP, MCPCorpus | no licence, no release, or no requests (ScaleMCP: templated financial queries) | 140 to 600 tasks; MCPCorpus about 14K servers | — | — | — | — | OUT |
| AgenticResourceDiscovery/verified-mcp-tools | CC-BY-4.0, attribution to neuronto.com required | 31,411 tools of 2,223 live servers | no requests | — | — | no | OUT as a test; a tool pool |
| MetaTool (used by v2 and v3) | MIT | 20,614 staged single-tool queries, about 199 plugins | yes | English | ChatGPT / GPT-4 | — | OUT: its tools were read twice |

### 4.3 The source proposed: Toucan-1.5M, config SFT

- **Why.** It is the only candidate that is a real catalogue of MCP tools (the servers' own
  descriptions), has requests in the tens of thousands, is under a permissive licence (Apache-2.0)
  and is not gated. Its requests were written by open-weight models only, which is the project's
  own rule for what it generates (not required of a public set used under its licence, but it
  means no proprietary model's terms enter the test). No source of v2 or v3 is in it: its servers
  were crawled from Smithery and GitHub in June 2025, and its rows are MCP trajectories.
- **What the build takes.** The three parquet files of the SFT config (425 MB, about 1.35 GB
  decoded; 9.1 GiB of disk were free on 17/11/2026). Of its rows: the subsets `single-turn-original`
  and `single-turn-diversify` with exactly one entry in `target_tools`; the gold is that tool; the
  slate's other options are, first, the other tools of the same server, as a conversation's own
  tools are in v2's rule, then random tools of other Toucan servers, by v2's distractor filter
  (not the gold's normalised name, MiniLM cosine below 0.8). 20% removed slates by v2's rule. The
  `irrelevant` subset (server metadata shuffled so that no tool serves the request) is an
  alternative source of *none*; the proposal keeps v2's removed rule, so that *none* means the same
  thing in every file.
- **Families never seen.** Every tool of a Toucan slate is checked against every tool of v2's
  training and validation pools by v2's cluster rule (same normalised name, or MiniLM cosine of the
  option texts at least 0.85). A request whose gold matches is dropped; a matching tool is never a
  distractor. The counts are written before scoring.
- **Concentration.** A cap per gold tool cluster (v2's 10 per test cluster) and a cap per server,
  so that no server dominates as three clusters dominated T2-tools.
- **What it costs.** No teacher call for the slates. If the lead allows the teacher to see the
  fresh tools' descriptions (never a request), one rewrite per Toucan tool shown in the test, for
  the probe copies: at most a few thousand items, about 100 to 150 calls (**[Inference]** from
  "2,000+ tools").
- **Its limits, stated as plainly.** The requests are model-written from the tool that seeded
  them, so they may echo its description, which favours BM25 and the model part's lexical input;
  the same server's tools may also serve a request (the distractor filter removes only the closest
  ones, and no teacher filter runs on a test); its language is not stated; the number of
  single-tool rows is not stated, so the size of the test is known only after the build counts it.
  It was made as training data for agents: other systems may have been trained on it (MiniLM was
  released before it).

## 5. Odds, risks

**Probabilities** (judgement, from the numbers above):

| event | probability | why |
|---|---:|---|
| the data checks pass without a remedy (the rewrite flag and the style rules at chance) | ≈ 50% | the rewrite flag is at chance by construction; the source-style cue of §2.5 is likely in the test-like kinds and would need source-matched fill |
| the shortcut is gone by the diagnosis's measure (masked validation, rewritten distractor ≤ 1/7 + 0.05) | ≈ 85% | the cue is no longer in the data; the probe measures exactly it |
| no pull of a rewritten distractor under real names (paired increase ≤ 0.03) | ≈ 70% | never measured; a residual style preference from natural slates is possible |
| on the names-masked fresh test the pair ≥ the embedding alone − 0.02 | ≈ 70% | under (i) the model reads its own BM25 score; BM25 alone kept 0.8138 on the masked T2-tools, the embedding 0.7280 |
| *none* answers on the masked fresh test in 0.10–0.30 | ≈ 75% | *none* is now right on 20% of the `tool_<k>` slates in training |
| G1 on Toucan: above BM25 with the lower bound above 0 | ≈ 55% | on MetaTool, the one never-trained family read, the pair was at 0.5175 against 0.4330 for BM25; model-written requests that echo the tool favour BM25 |
| G1 on Toucan: ≥ MiniLM − 0.03 | ≈ 20% | on MetaTool the pair was 0.14 below MiniLM (0.6570) and below its own embedding part (0.5340) |
| G1 on Toucan: above each part | ≈ 45% | the embedding part beat the pair on MetaTool |
| G3, G4 (calibration; 60 ms in WASM under (i)) | ≈ 80% each | ECE 0.0162 in v3; the first-request path measured at 45.2 ms |
| G5 (T2-tools within 0.02 of v3's pair) | ≈ 70% | the renamed kind was a quarter of the records and taught a cue that never occurs at use; removing it may help or hurt real-name accuracy (not verified) |
| **every gate condition** | **≈ 10–15%** | dominated by MiniLM − 0.03 on a never-trained family |
| the pilot rule stops the run | ≈ 15% | v2's micro pilot was at 0.6390 against an abort bar of 0.5755 |

**Main risks.**

- **The bar against MiniLM on new families.** v3 cleared MiniLM − 0.03 on tools of its own
  families. On a family never trained on, the only reading so far (MetaTool) put the pair far
  below. If the lead keeps v3's G1 on the Toucan test, a fail on that term alone says the router
  does not carry to new families as well as a 91.6 MB encoder, not that the shortcut fix failed:
  G2 measures the fix.
- **Real-name accuracy may drop.** The model part reached 0.8339 on validation partly on slates
  whose cues (source style, the rewrite) are now removed or balanced. Not measured.
- **Toucan's single-tool count** may be small after the filters and the cluster rule (its servers
  hold generic tools, `search`, `fetch`, `read_file`, whose normalised names may match training
  tools).
- **The build may stop** on the source-style rule; the remedy (source-matched fill) is written in
  the pre-registration, and a second stop ends v4 before any training.
- **Latency under (i)** can exceed 50 ms on a loaded machine; the rule then takes (ii').

## 6. Not verified

- The count of Toucan rows with one target tool, their language, and the exact number of tools in
  the SFT config; whether each row's metadata lists every tool of its server with its description
  (the card says the metadata is "original MCP server data").
- The text of Toucan's GitHub LICENSE (the repository shows MIT; the card says Apache-2.0).
- Whether Toucan questions name their target tool verbatim.
- The licence of Kimi-K2, one of Toucan's generators, was not read (it does not bind a user of the
  data set, whose licence is Apache-2.0).
- How many training tools a Toucan tool matches by v2's cluster rule.
- The source-style rule of §2.5 and the option-count prior on the built files (predicted from the
  code, not measured).
- The latency of option (i) with the longer option texts, and the head window it needs.
- The BPE split of `tool75`, `t1` … `t7` under `tok-s6-4096`.
- Whether the s2 graph can learn to bind the line through labels (option (ii')).
- Whether training without the shortcut raises or lowers real-name accuracy.
- The count of pool tools without a rewrite exactly (13,448 to 13,691: the pools were read from
  counts, not rebuilt).

## 7. What this study read

`AGENTS.md`; `docs/s6-route4-diagnosis.md` and `scripts/s6-route4-diag.py`; `docs/s6-nano2-prereg.md`
(§4, amendments 1 and 2, the result block); `docs/s6-nano2-readings.md`; `docs/s6-nano3-prereg.md`
(§2 to §9, amendments, the result block); `docs/s6-nano2-router-study.md`; `hf/cards/s6-route.md`
and `hf/cards/s6-route-nano.md`; `DATA_LICENSES.md` (s6-route-nano: tool routing); the scripts
`train-route.sh`, `build-route-data.sh`, `route-teacher.sh`, `distill-glm.sh`, `s6-route3-fit.py`;
`crates/sky-data/src/route/` (`build.rs`, `slates.rs`, `fresh.rs`, `mod.rs`; `sources.rs` for the
readers); `crates/sky-infer/src/route.rs`, `num.rs`, `s2/prompt.rs`, `s2/engine.rs`, `engine.rs`,
`bin/sky-eval.rs`; `crates/sky-web/src/router.rs` and `bench/route3-node.mjs`;
`crates/sky-train/src/main.rs` and `s2/train.rs` (checkpoints every `--checkpoint-minutes`, 10 by
default, into `last`, and `pre-qat`); `runs/s6-nano2/` and `runs/s6-nano3/` (counts, latency,
reports); in `skycmd-data/s6-route/build/`, the text files `counts/*.json`, `BUILD.json`,
`stage/tools.jsonl`, `teacher/rewrite-description*.jsonl`, `records/*/*.jsonl` and `val/v_tools*.jsonl`
(read, counted, never written); `docs/s6-ear2-study.md` and `docs/s6-ear2-prereg.md` for the form.
No test or fresh file was opened.
