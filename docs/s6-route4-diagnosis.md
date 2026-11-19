# s6-route: why the model part falls below chance when tool names are masked

*Status on 19/11/2026: the fourth version it prepared stopped at its data checks on 18/11/2026, before any training, and the route line was closed after v7 failed on 19/11/2026; see `docs/s6-route4-prereg.md` and `docs/s6-route7-prereg.md`.*

Written on 17/11/2026 to prepare a fourth version of the router. This file is a diagnosis, not a
pre-registration: a fourth version needs its own.

**What was read and run.** The builder (`crates/sky-data/src/route/`), the engine's BM25 line
(`crates/sky-infer/src/route.rs`), the numeric extraction and number fusion of the s2 graph
(`crates/sky-infer/src/num.rs`, `crates/sky-infer/src/s2/prompt.rs`, `docs/SPEC-S2.md`,
`docs/SPEC-S3.md`), `docs/s6-nano2-prereg.md` (§4.4, §7.3, amendment 1, the result block),
`docs/s6-nano2-readings.md` (readings 13, 17, 27), `docs/s6-nano3-prereg.md` (§2, §6, the result
block) and `runs/s6-nano3/rehearsal-val/results.md`. The published files were run:
`site/public/models/s6-route-nano.gguf` (the model part, sha256 `a124dfe8…6c82`) by `sky-eval` at
full depth (exit threshold 1), and the pair with `s6-route-embed.gguf` by `sky-route fuse`. **They
were run on validation files only**: the validation slates of `build/val/`, the validation records
`build/records/tools/val.jsonl`, and copies of them changed in one respect each. No model was run
on a test or fresh file, and no new score was computed on one. The published per-slate outputs of
the two names-masked readings (`eval/test/model/t_tools_masked.jsonl` for v2's T-tools,
`eval/fresh/model/f_tools_masked.jsonl` for v3's T2-tools) were tabulated next to the text of their
slates, for two things only: how often the model answered *none*, and whether the option it
picked ends with a period. Paths that start with `build/` or `eval/` are in the data folder
`skycmd-data/s6-route/`, which was only read. Every table below comes from
`scripts/s6-route4-diag.py`. It writes the changed validation files, and its header gives the
`sky-eval` and `sky-route fuse` commands that score them. The engine was built from this commit's
tree (`cargo build --release -p sky-infer --bin sky-route --bin sky-eval`). On `v_tools_masked`
it reproduces the published rehearsal: model part 0.1171, pair 0.3086. The rows of `v_tools` and
`v_tools_masked` themselves are read from the published rehearsal dumps
(`eval/fresh/rehearsal/model/`).

## 1. Summary

In every training slate whose tools are all named `tool_<k>` (the *renamed* kind, 51,101 training
records), the right tool is the only option whose description the teacher rewrote, and *none* is
never the answer. The model learned those two facts in place of the task. Under `tool_<k>` names it
picks the option that reads like a rewrite, never answers *none*, and ignores the BM25 line
(shifting the line by one entry: 0.9489 against 0.9498). In the masked reading no description is a
rewrite. The model picks the most rewrite-like original with high confidence (0.918 of its picks
end with a period; 0.575 of options and 0.337 of right tools do). It loses every *none*-gold slate,
and is at or below 1/7 elsewhere: on T2-tools, 68 right of 756 tool-gold slates (0.090), 0 *none*
answers. Names, k, the line and its order are built alike in training and in the masked files. This
is a shortcut in the training data, and no rendering of the published pair fixes it (§4, item 5).

| validation slates (published model part, full depth) | names | descriptions | n | accuracy | *none* answers |
|---|---|---|---:|---:|---:|
| `v_tools` | original | original | 1,144 | 0.8339 | 0.1469 |
| `v_tools_masked` | `tool_<k>` | original | 1,144 | 0.1171 | 0.0009 |
| renamed records (`build/records/tools/val.jsonl`) | `tool_<k>` | the gold's is the teacher's rewrite | 1,116 | **0.9498** | 0.0000 |
| the same, the gold's original description put back | `tool_<k>` | original | 1,116 | **0.3271** | 0.0287 |
| `v_tools_masked`, the gold's description rewritten | `tool_<k>` | the gold's is the rewrite | 915 | **0.9082** | 0.0000 |
| `v_tools_masked`, one random distractor's description rewritten | `tool_<k>` | that distractor's is the rewrite | 1,113 | **0.0270**; that distractor picked on 0.8311 | 0.0000 |

Chance among the tools of the renamed records (2 to 8 options) is 0.371. Chance on the masked slates is 1/8 = 0.125, and 1/7 = 0.143 among their tools.

## 2. The four hypotheses

### (a) A difference of format between training `renamed` records and the masked slates

**How each is built.** Both come from one function, `slates::masked`
(`crates/sky-data/src/route/slates.rs:161`). Training `renamed` records call it in
`build.rs:1509`, on the request's *natural* slate. The masked validation, test and fresh files call
it in `build.rs:484`, on the test-like slate (`fresh.rs:325` uses the same `tool_slates`). Item by item:

| item | training `renamed` | masked slates | same? |
|---|---|---|---|
| names | `tool_<k>`, every tool | `tool_<k>`, every tool | yes |
| k | distinct, drawn at random from 1 to 99 per slate (`ks` shuffled), not the position | the same rule | yes |
| the name in the BM25 line | `option_name` of the option text (`route.rs:110`): the same `tool_<k>` as the option | the same | yes |
| order of the BM25 line | the slate's order, one entry per tool option (`route.rs:116`) | the same | yes |
| when the line is computed | after renaming, from the rendered options (`build.rs:1393`) | after renaming: `sky-data route render` (`build.rs:1609`) and the engine (`sky-route.rs:289`) both compute it from the options as given | yes |
| the gold's description | **the teacher's rewrite**, cut to a sentence and 30 words (`build.rs:1504`) | **the source's own** | **no** |
| the other descriptions | the source's own | the source's own | yes |
| *none* as the gold | **never**: 0 of 21,759 (tools), 3,899 (When2Call) and 25,443 (intents) training records, 0 of 4,433 validation ones | **20% of the slates** (the removed ones) | **no** |
| tool options | 2 to 8 options with *none*, uniform | always 8 (7 tools and *none*) | no (minor) |
| truncation | 48 tokens per option (the engine's cap), head window 256 | the same rule | yes |

The twin of the builder in `scripts/s6-route4-diag.py` checks all 1,116 validation renamed records
option by option. Each gold is `tool_<k>: ` followed by the cut rewrite, and every other option is
the source's description (4,431 options, no difference). Its BM25 line equals the published model
copies on all 2,288 validation slates (`v_tools`, `v_tools_masked`).

**What masking does to BM25.** BM25's tokens are runs of word characters, and `_` is one, so
`tool_75` is a single token that no request contains. It adds nothing to a score and one token to
a document's length. On validation slates with a tool gold, the gold is BM25 rank 1 on 0.9432 of
them with the real names and on 0.9410 with masked names. On training renamed tool slates the gold
is rank 1 on 0.9047, against 0.9495 for the natural slates the renamed ones are made from. The line
of a masked slate points at the right tool as often as an unmasked one does.

**What masking does to the tokens the model reads.** In `tool_75`, the digits follow `_`, not an
ASCII letter, so the numeric extraction takes them (`num.rs:67`): the text becomes `tool_[NUM]` with
the value 75. With `fuse_numbers` on (the model's config), the value then moves onto the token before
it (`prompt.rs:74`). Every masked name, in the options and in the line, is the same token sequence
and differs from the others only by a value in the number channel. Real names are distinct BPE
tokens. This is true in training and in the masked reading alike, so it is not a difference between
them. It does explain why the line is hard to read under masked names, which matters for §4, item 4.

**Verdict: supported, for two differences only**: the gold's description (a teacher rewrite in
training, the original in the masked files) and the answer *none* (never the gold of a renamed
slate). Ruled out for the names, k, the line's keys, the line's order, when the line is computed,
and truncation. The difference in slate size has no shown effect (the natural validation slates
masked, 2 to 8 options, are at 0.3333, row "natural, masked" in (d)).

### (b) The model follows the BM25 line's rank-1 name, and the mapping points at a wrong option

On `v_tools_masked` (validation, the published rehearsal dump
`eval/fresh/rehearsal/model/v_tools_masked.jsonl`, by the published model file; the same numbers as
`runs/s6-nano3/rehearsal-val/results.md`):

| | `v_tools` (names) | `v_tools_masked` |
|---|---:|---:|
| accuracy | 0.8339 | 0.1171 |
| pick = BM25 rank 1 | 0.7439 | 0.1399 |
| gold = BM25 rank 1, slates with a tool gold | 0.9432 | 0.9410 |
| picks by position 0 to 6 (*none* apart) | 166, 138, 140, 128, 158, 128, 118 | 192, 154, 161, 155, 166, 150, 165 |
| picks by BM25 rank 1 to 7 | 851, 45, 23, 23, 14, 10, 10 | 160, 175, 167, 159, 146, 180, 156 |
| *none* answers | 168 | 1 |

With masked names the pick is spread evenly over positions and over BM25 ranks: rank 1 gets 0.1399,
where 1/7 is 0.143. The line points at the gold on 0.9410 of the tool-gold slates, so a model that
read it would be right. No position and no rank attracts the picks, so the model is not reading a
shifted mapping either. Two changes to the line, on validation:

- **The line shifted** (renamed validation records; each name keeps its place, and its score and
  rank are those of the next entry): 0.9489, against 0.9498 with the true line. Under `tool_<k>`
  names the line's numbers are not read.
- **The line removed**: masked slates 0.2212, with *none* answers 0.3497 and a mean top probability
  of 0.3896 (0.8000 with the line). The unmasked slates go from 0.8339 to 0.4318. With real names the
  line is the model's main input. With masked names its presence changes how sure the model is and
  how often it says *none*, but its content does not change the pick.

**Verdict: ruled out.** The masked pick does not follow the rank-1 entry, a position or a fixed
offset in the line. It does not depend on the line's content at all.

### (c) *None*, the last option

The model answers *none* on 1 of the 1,144 masked validation slates, against 168 (0.1469) with the
real names. The published outputs show the same: 1 of 2,000 on v2's masked T-tools and 0 of 945 on
v3's masked T2-tools (`scripts/s6-route4-diag.py published`). No training slate named `tool_<k>`
has *none* as its gold: the renamed kind is built only for requests with a tool gold
(`build.rs:1471-1521`), and the removed kind is never renamed. Every *none*-gold slate of the masked
reading is therefore lost: 229 of 1,144 on validation and 189 of 945 on T2-tools. The rest of the
accuracy is that of the tool-gold slates. On validation it is 0.1464 (134 of 915, chance 1/7), so
0.8 × 0.146 = 0.117 overall. On T2-tools, 0.0720 × 945 = 68 right answers, all on the 756 tool-gold
slates (0.090).

**Verdict: supported**, as one of the two parts of the mechanism. It accounts for the fall from 1/7
to about 0.8/7. It does not account for the confident wrong tool answers.

### (d) The training `renamed` kind did not teach what the masked reading asks

The renamed kind is 0.2667 of the records of each group (`kinds`). Its line would have taught the
task, since the gold is BM25 rank 1 on 0.9047 of the training renamed tool slates. But its gold also
carries a mark that needs no request: the teacher's style. Descriptions of training renamed slates
(`style`):

| group | option | ends with a period | first word ends in -s ("Returns", "Fetches") |
|---|---|---:|---:|
| tools | gold (rewrite) | 0.998 | 0.638 |
| tools | the others (originals) | 0.650 | 0.392 |
| intents | gold (rewrite) | 1.000 | 0.726 |
| intents | the others (originals) | 1.000 | 0.046 |
| When2Call | gold (rewrite) | 1.000 | 0.712 |
| When2Call | the others (originals) | 0.713 | 0.485 |

A rule that never reads the request picks the option ending with a period, then the one whose first
word ends in -s, with ties at random. It gets 0.502 on training renamed tool slates and 0.774 on
renamed intent slates, against 0.371 and 0.369 for chance among the tools. On the natural slates of
the same requests it gets 0.264 and 0.369. The two features are only a proxy for the style. The
model's own detector is better than the rule:

| validation (published model part, full depth) | n | accuracy | mean top p | *none* answers |
|---|---:|---:|---:|---:|
| natural records (`build/records/tools/val.jsonl`) | 1,116 | 0.8539 | 0.8568 | 0.1129 |
| renamed records, as built | 1,116 | 0.9498 | 0.9684 | 0.0000 |
| renamed, the gold's original description back | 1,116 | 0.3271 | 0.8354 | 0.0287 |
| natural records, every name masked `tool_<k>` (original descriptions) | 1,116 | 0.3333 | 0.8388 | 0.0260 |
| renamed, the BM25 line shifted by one entry | 1,116 | 0.9489 | 0.9686 | 0.0000 |
| renamed, no BM25 line | 1,116 | 0.8226 | 0.7343 | 0.1210 |
| `v_tools_masked` | 1,144 | 0.1171 | 0.8000 | 0.0009 |
| `v_tools_masked`, the gold's description rewritten | 915 | 0.9082 | 0.9468 | 0.0000 |
| `v_tools_masked`, one random distractor's description rewritten | 1,113 | 0.0270 | 0.9216 | 0.0000 |

On the third row, chance among the tools is 0.371.

The model is more accurate on renamed slates than on the natural slates they are made from (0.9498
against 0.8539), although renaming removes information. Put the gold's own description back and it
falls under chance among the tools (0.3271), still at a mean top probability of 0.8354. Rewrite one
distractor of a masked slate and it picks that distractor on 0.8311 of the slates. On masked
slates, the option it picks reads like a rewrite: 0.918 of picks end with a period and 0.596 start
with a word in -s, against 0.575 and 0.363 for all options and 0.337 and 0.192 for right tools.
With real names its picks look like the golds (0.339 and 0.195). Right tools of these validation
slates read less like rewrites than their distractors do. That is why the shortcut lands below 1/7
rather than at it on some sources. On masked validation, tool-gold accuracy by source is: api-bank
22 of 80, glaive 7 of 65, hermes 23 of 45, seal-tools 19 of 412, toolace 63 of 313. When the gold
both ends with a period and starts with a word in -s, it is 60 of 162; otherwise 74 of 753. The
request-blind rule, on the same tool-gold slates, is at 0.078. On T2-tools, 0.9481 of the model's
picks end with a period, against 0.5868 of the options. On v2's T-tools the figures are 0.9430 and
0.5913.

**Verdict: supported.** The renamed kind taught the model to find the rewritten description and
never to answer *none*. It did not teach the model to read descriptions against the request, nor to
bind the BM25 line to options named `tool_<k>`.

## 3. Masked slates as the model reads them, next to a training renamed record

Each record is shown as the files give it: the options in order (*none* last) and the state. The
engine then turns `tool_75` into `tool_[NUM]` carrying the value 75, fused onto the token before
it (§2 (a)); the scores and ranks are fused the same way onto `score` and `rank`.

**Validation, `v_tools_masked`, `glaive:706`.** Gold: option 6. Model: option 3, p = 0.9279. With
the real names the model picks option 6, p = 0.9998.

```text
opts:
0 tool_75: Drill down into sales data from an initial aggregation level to a target level.
1 tool_56: Get a random joke from a predefined list
2 tool_51: Processes a refund request for a previous transaction, ensuring compliance with transaction policies.
3 tool_73: Get the details of a specific TV channel broadcasting a specific Football match.      <- picked
4 tool_22: Retrieve the current Ethereum price in Uniswap v2
5 tool_97: Check the status of the Vizor Ads server
6 tool_18: Translate text from one language to another                                          <- gold, BM25 rank 1
7 none of these tools fits the request
state:
User: Hi, I have a text in French that I need to translate to English. The text is "Bonjour, comment ça va?"
BM25: tool_75 score 0.00 rank 2; tool_56 score 0.00 rank 3; tool_51 score 0.00 rank 4; tool_73 score 0.00 rank 5; tool_22 score 0.00 rank 6; tool_97 score 0.00 rank 7; tool_18 score 2.16 rank 1
```

**Validation, `v_tools_masked`, `glaive:2216`.** Gold: option 1. Model: option 3, p = 0.9022. With
the real names: option 1, p = 0.9546.

```text
opts:
0 tool_44: Identifies species in a given area of the Amazon Rainforest using image recognition technology.
1 tool_25: Get book recommendations based on user's preferences                                 <- gold, BM25 rank 1
2 tool_26: Fetches the details of a pet given its pet ID using the Toolbench RapidAPI.
3 tool_83: Returns a list of alternative fuel stations that match the given query.              <- picked
4 tool_48: Retrieves information about tournament seasons
5 tool_18: Search for a book based on title and/or author
6 tool_43: Retrieve a list of people, sorted by the specified field.
7 none of these tools fits the request
state:
User: Hi, I am looking for some book recommendations. I love mystery novels and I prefer books that have a rating of at least 4.
BM25: tool_44 score 0.00 rank 3; tool_25 score 1.21 rank 1; tool_26 score 0.00 rank 4; tool_83 score 0.00 rank 5; tool_48 score 0.00 rank 6; tool_18 score 0.53 rank 2; tool_43 score 0.00 rank 7
```

**Validation, `v_tools_masked`, `glaive:1060`** (a removed slate: the gold is *none*). Model:
option 1, p = 0.8964. With the real names: *none*, p = 0.8299.

```text
opts:
0 tool_92: Retrieve a list of invite templates for financial operations.
1 tool_5: Add a product to the user's shopping list with the specified quantity.                <- picked
2 tool_78: Retrieve a list of large-cap stocks that are potentially undervalued, using Yahoo Finance data.
3 tool_22: Retrieve a list of stocks that have a Hanging Man signal, along with relevant financial information.
4 tool_64: Search for books based on author, title, or genre
5 tool_73: Retrieves a list of pending expenses for a manager to review.
6 tool_91: Calculates the confidence interval for a sample mean.
7 none of these tools fits the request                                                           <- gold
state:
User: Hi, I need to translate a sentence from English to French. The sentence is "Hello, how are you?"
BM25: tool_92 score 0.00 rank 1; tool_5 score 0.00 rank 2; tool_78 score 0.00 rank 3; tool_22 score 0.00 rank 4; tool_64 score 0.00 rank 5; tool_73 score 0.00 rank 6; tool_91 score 0.00 rank 7
```

**Training, `build/records/tools/train.jsonl`, `glaive:10037/renamed`.** Gold: option 1. The source's own
gold is `get_definition: Retrieve the definition of a word`.

```text
opts:
0 tool_31: Tracks the status and details of a service request made by a client.
1 tool_90: Returns the definition of a given word.                                              <- gold [D1], BM25 rank 1
2 tool_46: Create a new report in Power BI.
3 tool_91: Retrieves a list of newsletters related to Azure services.
4 tool_28: Retrieves a list of product prices for a given product ID.
5 tool_3: Returns address information from an address code.
6 tool_57: Retrieve game art and animations
7 none of these tools fits the request                                                           [D2]
state:
User: What's the definition of the word 'serendipity'?
BM25: tool_31 score 0.00 rank 2; tool_90 score 1.50 rank 1; tool_46 score 0.00 rank 3; tool_91 score 0.00 rank 4; tool_28 score 0.00 rank 5; tool_3 score 0.00 rank 6; tool_57 score 0.00 rank 7
```

**The differences:**

- **[D1]** The gold of a renamed record is the teacher's text ("Returns the definition of a given
  word." for "Retrieve the definition of a word"), and it is the only teacher text in the slate. In
  the masked slates every description is the source's. The glaive golds above are imperative
  sentences without a final period ("Translate text from one language to another"). Several
  distractors are third-person sentences that end with a period ("Returns a list of alternative
  fuel stations that match the given query."), and those are what the model picks.
- **[D2]** *None* is the gold of no renamed record. It is the gold of 20% of the masked slates
  (`glaive:1060`).
- **[D3]** A renamed record has 2 to 8 options. A masked slate always has 8 (all four examples
  here have 8).
- **The same in both:** `tool_<k>` with distinct random k, independent of the position; the same
  name in the option and in the line; the line in the slate's order, computed after renaming; the
  instruction; the 48-token cap per option.

## 4. What the next version should change

Ordered by the strength of the evidence above. Each item comes with the cheapest check that would
confirm it before any training.

1. **The teacher's rewrite must not mark the gold.** (Evidence: the table of §1, and §2 (d).) Decide
   for each option, independently of whether it is the gold, whether its description is the
   source's or the teacher's: for example, the rewrite with probability 0.5 on every option of
   every kind. A cheaper alternative keeps every original description in renamed slates and moves
   the rewrites to their own kind, in which every option is rewritten. Today there are 14,263
   rewrites, one per gold tool of the training and validation requests (reading 17), so distractors
   drawn from the split's pool may have none. The first count to take is how many distractors lack
   a rewrite. *Check before training, without a model:* on the new training files, P(rewritten |
   gold) equals P(rewritten | other option) in every kind and group. The request-blind rule of
   `scripts/s6-route4-diag.py style` stays at chance among the tools on every kind (today 0.502 and
   0.774 on renamed). *First check after training, on validation:* in `masked_distractor_rewritten`,
   the rewritten distractor is picked no more often than 1/7 (today 0.8311). Renamed validation
   records score no higher than natural ones (today 0.9498 against 0.8539).
2. **Slates named `tool_<k>` must sometimes have *none* as the gold.** (Evidence: §2 (c).) Rename
   the removed slates too, at the 20% share of *none* that §4.4 of v2 fixes for every group.
   *Check before training:* in the built files, *none* is the gold of 0.20 of the `tool_<k>` slates
   of each group (today 0 of 51,101). *After training, on validation:* *none* answers on
   `v_tools_masked` fall within G2's band of 0.10 to 0.30 (today 0.0009).
3. **Make the renamed training slates the masked reading's twin, and read the masked validation
   slates during training.** (Evidence: renamed validation records measure the shortcut, 0.9498,
   where the same slates with original descriptions give 0.3271, and natural slates masked give
   0.3333.) Draw renamed slates by the test-slate rule (7 tools and *none*, original or randomly
   rewritten descriptions as in item 1), so that the training kind and the reading differ only in
   their requests. Add `v_tools_masked` (validation) to the readings of every pilot and checkpoint.
   *Check before training:* none is needed beyond items 1 and 2. The number to beat is the current
   model's: 0.1171 on `v_tools_masked`, 0.3333 on the masked natural validation slates.
4. **Bind the lexical score to its option without relying on a distinctive name.** (Evidence: with
   real names the line is the model's main input, 0.8339 with it and 0.4318 without it. With
   `tool_<k>` the names become the same tokens and differ only in the number channel (§2 (a)).
   The model has never needed to bind the line under such names, since shifting the line changes
   nothing.) Whether training without the shortcut would teach binding by the fused value is not
   known. Two ways that do not depend on it:
   - (i) **Write each option's score and rank in its own text**, as amendment 1 of v2 first
     described it. The cost is the static cache (`route.rs:6-11`): option texts would change with
     each request, so the options would be encoded for every request. *Check before training:* time
     the engine on validation slates with its caches off (`sky-eval --latency` reports
     `latency.no_cache`), against v3's bar of 60 ms in WASM.
   - (ii) **Keep the line but key it by a text that stays distinct after the numeric extraction**:
     letters, or a number glued to a letter (`tool75`, which `num.rs:67` leaves as text). Render the
     masked reading the same way. The current weights do not bind these either (`masked_glued`
     0.2255, `masked_letters` 0.1879; they were never trained on them). *Check before training:* a
     tokenizer check that, in every slate, the keys of the line are distinct token sequences after
     extraction and fusion. That does not hold for `tool_<k>`. It also fails for real names whose
     digits the extraction takes (a number not glued to a letter: `search_2`, `48 Hour
     Forecast`), when two names differ only by that number. Such names are on 1,517 of the 230,171
     tool options of the training tool records that are not renamed (226 of 20,850 distinct
     names).
5. **No fix without training.** The defect is in the weights, not in the rendering: under
   `tool_<k>` names the model looks for a rewritten description, and no rendering of an original
   description supplies one. Renderings tried on validation, with the published pair (`sky-route
   fuse`, β 0.81) on the 1,144 masked slates:

   | names rendered as | model part | pair |
   |---|---:|---:|
   | `tool_<k>` (the published reading) | 0.1171 | 0.3086 |
   | `tool<k>`, digits glued | 0.2255 | 0.6870 |
   | distinct random English words | 0.4921 | 0.7902 |
   | for reference, `potion-cos-4bit` alone / BM25 alone / BM25 fused with the embedding | | 0.7351 / 0.8444 / 0.8575 |

   No rendering brings the pair to BM25 alone on the same slates. A guard needs no training: the
   engine could answer with the embedding part alone when the names are not distinctive (0.7351 on
   these slates), and no validation tool name matches `tool_<digits>` (0 of 8,008 in `v_tools`). But
   it is a different system from the one frozen in §2 of v3, and it would need fresh slates of its
   own. Either change would also change what the names-masked reading means. The published numbers
   (0.1050 for the model on v2's test; 0.0720 for the model and 0.2497 for the pair on T2-tools)
   describe the published files read with `tool_<k>`, the reading that §7.3 of v2 fixed. A new
   rendering would be a new reading. It belongs on validation or on new fresh slates, not on the
   test and fresh files, which were read once for the published results. The numbers of the
   embedding alone, BM25 alone and BM25 fused with the embedding come from
   `runs/s6-nano3/rehearsal-val/results.md`. With word names the embedding part also falls (0.6984
   against 0.7351), because it reads the names too.

## 5. Not verified

- The exact BPE split of `tool_` by the micro tokenizer: §2 (a) follows the code (`num.rs`,
  `prompt.rs`, `fuse_numbers` in the GGUF config). No token stream was printed.
- What the model's rewrite detector actually reads. Ending with a period and a first word in -s are
  a proxy: the rule built on them gets 0.502 on renamed tool slates, where the model gets 0.95.
- Whether the s2 graph can learn to bind a BM25 entry to an option through a fused value alone.
- Whether the 48-token cap removes the final period of long descriptions before the model sees it.
- The fast mode (first exit) was not examined. Its masked number, 0.0762 on T2-tools, is close to
  full depth.
- Renamed validation records of the When2Call and intents groups were not run through the model;
  only the tools group's 1,116 were.
- Why the BM25 line's presence, but not its content, changes the masked answers (0.2212 and 0.3497
  *none* answers without it; 0.8226 on renamed records without it).
- Whether removing the shortcut raises or lowers accuracy with real names. The renamed kind is a
  quarter of the training records, and it currently teaches a cue that never occurs at use.
- On the test and fresh files, only the published outputs were tabulated (*none* answers, and the
  share of picks ending with a period). No accuracy or other score was computed on them.

## 6. Other observations

- **Amendment 1 of v2 and reading 13 disagree on the descriptions.** Amendment 1 says that in a
  renamed slate "every option is renamed `tool_<k>` … the descriptions are untouched". Reading 13,
  which the same amendment accepts, and the code (`build.rs:1499-1519`) keep the teacher's rewrite
  on the gold. The fix for the name giving the answer away left the description doing the same.
- **Amendment 1 of v2 says "each option's text carries its BM25 score"**. The build puts the scores
  in the state, in a line keyed by option name (reading 27, `route.rs:6-20`). That is accepted and
  explained, but it is the reason the scores need a distinctive name.
- **v2's result block offered one reading, untested**: "the model ties each BM25 entry to its
  option through a distinctive name, and loses that binding when the names look alike". That is
  not what the validation readings show. Under masked names the model does not read the line at
  all, either in training-like slates or in the masked ones (§2 (b)). It answers from the style of
  the descriptions. The disclosure "the system needs distinctive tool names" (v3 §7) stays true.
- **Real tool names lose digits too.** A number not glued to an ASCII letter goes to the number
  channel (`search_2` becomes `search_[NUM]`; `glm-5` and `v2` stay text). Tools whose names differ
  only by such a number become the same tokens, in the options and in the line. Names with such a
  number are on 1,517 of the 230,171 tool options of the training tool records that are not
  renamed.
