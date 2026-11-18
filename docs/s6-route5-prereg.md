# s6-route v5 (v4 with every drawn distractor in the right tool's surface cell, the hard ones anchored on the request): pre-registration

**Frozen on 18/11/2026 by the lead, before any computation of v5 and before its code is written; every later change is a dated amendment that precedes the computation it concerns.**

- **Nothing of v5 exists yet:** no data built, no teacher call, no model trained, no fresh file.
  To write this document, no data file was opened. It reads:
  - v4's published report (`runs/s6-route4/checks.md`, `checks.json`) and the result block at the
    end of `docs/s6-route4-prereg.md`;
  - the code of v2's hard plan (`crates/sky-data/src/route/build.rs`), for §4.1.
- **v5 is v4 with one change to the drawn distractors (S1, §2) and its consequences.** The change
  has two parts:
  - every drawn distractor lies in the right tool's surface cell;
  - the hard distractors are anchored on the request.

  Everything else is inherited from `docs/s6-route4-prereg.md`, as frozen on 17/11/2026 with its
  amendment 1 of 18/11/2026, section by section.
  - "As v4 §x" means that v4's text applies word for word, with v4's names replaced by v5's:
    - `v4_tools` becomes `v5_tools`, and T4 becomes T5;
    - `runs/s6-route4*` becomes `runs/s6-route5*`, and `models/s6-route4` becomes
      `models/s6-route5`;
    - the `route4` scripts become the `route5` scripts of §14.
  - Only what changes is restated here, and every bar that changes is given as a number.
  - The lead's decisions on the draft are listed in §17.
- **Fixed later, by rules written here, each before any training:**
  - the head window, as v4 §3.3;
  - the pilot's bar, as v4 §5, read on `v5_tools`;
  - the counts amendment of §6.3, as v4 §6.3.

**Why this experiment.** v4 stopped at its data checks before any training (the result block of
`docs/s6-route4-prereg.md`). Request-blind rules pick the right tool well above chance in two
places, and the one remedy of v4 §4.4 was not open:

- **The hard slates.** In the tools group, the right tool is shorter, more often without a final
  period and more often imperative than its nearest neighbours: fewest characters 0.4809 against a
  chance of 0.3763 on validation. In the intents group the right tool has the shortest name (0.4219
  against 0.3721 in training).
- **The natural fill of the tools group,** where the longest name wins (0.4015 against 0.3709).

v4's remedy could not have reached the hard slates in any case: they are not a random fill. v5
draws every distractor of every kind of slate within the right tool's surface cell, defined on the
features those rules read. The hard distractors become the tools nearest to the request rather
than to the right tool. v5 also adds request-blind rules that the cell does not use, judged with
the same tolerance, and a cap on the slates built outside the cell. If any of these fails, v5
stops, with no remedy.

## 1. The question

As v4 §1, read on T5-tools (§6) in place of T4-tools.

## 2. What changes from v4, and why

| # | change | evidence | where it is checked |
|---|---|---|---|
| S1 | **Every drawn distractor of every kind of slate is drawn within the right tool's surface cell** (§4.3): the natural fill, the renamed slates, the refills of D5, the hard slates, the false-negative replacements, the test-like validation slates and the fresh test's fill. The request's own tools are exempt (§4.3) | v4's checks: 17 failing rules in five cells (R1 to R5, R11, R12, R13); C4's surface gaps in the hard cells (result block of v4) | §4.4 C2, C6 |
| S1′ | **The hard distractors are the tools nearest to the request**, not to the right tool, within the cell (§4.3) | v2 anchored the hard slates on the right tool (§4.1), which puts the right tool at the centre of its slate | §4.4 N5 |
| S1a | **A new false-negative job** on every new hard slate of the training and validation requests | no hard slate keeps v2's tools, so v2's 69,290 answers no longer apply | §4.2 |
| S1b | **New test-like validation slates `v5_tools`** (seed 1238) for the fits and the pilot | `v4_tools` was filled by D7 without the cell | §4.3 |
| S1c | **The fresh test's fill aligned with the cell** (v4 §6.2, item 5); files `t5_*` | one cell definition in training, validation and test | §6.2 |
| S1d | **New request-blind rules N1 to N5 and the cap C6, judged; no remedy** | the cell removes the listed cues by construction only for the categorical features; the next cue would be one the cell does not see | §4.4 |
| S1e | **The baselines' validation accuracy on `v5_tools`, `v4_tools` and `v_tools`**, reported | a cell that conditions distractors on the gold changes difficulty in ways no rule sees | §4.5 |

**Not changed:** v4 §2's D1 to D7 as amended, except that D7's source-first order now applies
inside the cell (§4.3). Also unchanged: everything that v4 §2 lists as "not changed", with the
graph, the initialisation, the prompt, the rendering of D4, the embedding part, the fusion and the
ask rule of v4 §3.

## 3. The system

- **§3.1 to §3.3 and §3.5:** as v4.
- **§3.4, the rendering: option (i), kept from v4's latency check, which is not run again.**
  - **v4's measurement:** a median of 24,622 µs per request, against a bar of 50,000 µs. It used v2's
    graph, the same weights' shape and the same rendering (scores in the option text), at a head
    window of 320 (`runs/s6-route4/latency-check.json`).
  - **Why it carries over.** S1 changes which tools a slate shows. It does not change what is
    rendered or how much: an option keeps its 48-token cap, a slate keeps 8 options, and the head
    window that v5's records can need is 160, 256 or 320 by v4 §3.3's rule. v4 measured at 320,
    the largest of the three.
  - **G4 still decides.** It measures the pair on T5-tools with the same 60 ms bar (§8).
  - **If v5's build ever changes the rendering** (a new key, a longer option, a window above
    320), the check of v4 §3.4 runs again, as its own stage, before the build.

## 4. Data (`sky-data route`, extended)

### 4.1 What is kept, and what v2's hard plan did

**What v2 did, as read in the code.** v2 anchored its hard slates on the right tool, not on the
request.

- **The nearest lists.** `crates/sky-data/src/route/build.rs` computes them per gold:
  - lines 839 to 865: "the nearest tools of each gold (potion-base-2M cosine), among the other
    clusters of its pool". The queries are the gold tools, and for a request without a gold, its
    own tools;
  - lines 866 to 896: `cluster::nearest_all(&l.potion, …, NEAREST, …)` over the tools' rows of
    `emb/potion.npy`, with `NEAREST` = 16 (line 50).
- **No request is embedded.** `scripts/route-embed.py` embeds only the tools' `name: description`
  ("Neither model sees a request", line 13).
- **The hard slate.** At lines 922 to 955 it is the gold and the first entries of
  `near[(pool, gold)]`, or for a request without a gold, its own tools and then their nearest tools
  in turn.
- **The readings say the same.** `docs/s6-nano2-readings.md`, item 11: "Nearest to the gold tool
  (potion-base-2M cosine between option texts), not to the request".

**Why v5 changes the anchor (the lead's decision 3).**

- (a) The neighbours of the gold put the gold at the centre of its slate: the distractors are each
  close to it, not to each other. That is the request-blind cue that N5 reads (§4.4).
- (b) A deployed router receives the tools that a retriever found for the request, so distractors
  anchored on the request are the realistic hard case.

**As v4 §4.1**, with these changes:

- **The hard slates change.** The plan's draws are kept: the sizes of the natural and hard slates,
  and the stream `route/plan/<id>`. The nearest list is replaced by the 16 tools nearest to the
  request within the cell (§4.3).
- **The false-negative answers of v2 are not reused.** A hard slate of v5 can match a slate of v2
  by `fn_id` only by coincidence. Where one does, the answer applies by id.
- **v2's and v4's rewrites are reused:** 14,263 and 13,687 answers. Every tool of the pools has one
  (v4's plan found 13,687 tools without one, and all were answered).
- **New embeddings, of the requests.** Each training and validation request is embedded by
  potion-base-2M (float, the pinned commit of `scripts/route-embed.py`, the same static embedding
  that v2's nearest lists used) from the request's text, as `scripts/route-baselines.py`'s
  `potion-cos` embeds a request. The file is `emb/potion-requests.npy`, under the guard that refuses
  a test or fresh request. The tools keep v2's embeddings of their source `name: description`, so
  the nearest list does not depend on D1's draw. The cell, which does, filters it (§4.3).

### 4.2 The teacher

GLM-5.3-Flash (MIT), through `scripts/route-teacher.sh`, under amendments 1 and 3 of
`docs/s6-gen-prereg.md`.

- **The `false-negatives` job, needed in every case:** v2's prompt word for word (v2 §4.5,
  readings item 16).
  - **Its items:** every training and validation request with a hard slate. The teacher sees the
    request and the tools of its hard slate, numbered, without *none* and without being told which
    is right.
  - **Expected:** **69,290 items**, one per request with a hard slate, as v2's job
    (`build/teacher/false-negatives.items.jsonl`: 69,290 items, 69,288 answered, 2 refused). That
    is **about 3,465 calls of 20**, fewer only by any slate that matches v2's by id.
  - **The time:** v4's rewrite job did 685 calls in 1,384.7 s. A false-negative item is about four
    times larger (577 bytes on average against 154 for a rewrite item, in v2's item files), so about
    2 to 4 hours of calls, in one or two usage windows.
  - **What changes in the answers:** distractors nearest to the request are more often able to
    serve it than the gold's neighbours were. More distractors are expected to be named and
    replaced than v2's 14,497, and more requests dropped than v2's 4,562. Both are counted.
  - **Who sees what:** the teacher sees training and validation requests only. The guard of
    `sky-data teacher` refuses any item that holds a request of a test or fresh file.
  - **The rule of v2 stands:**
    - each distractor that the teacher names is replaced by the next tool of the request's 16
      nearest in the cell that is not in the slate; when the 16 are used up, the cell's random fill
      at the same level completes the slate (counted);
    - a request where the teacher does not name the right tool as able is dropped with all its
      slates and counted;
    - a request whose item moderation refuses keeps its hard slate unfiltered and is counted.
- **The `fresh-tools` job,** as v4 §4.2 (the probe copies only, about 100 to 150 calls).
- **No `rewrite-description` job:** nothing to rewrite (§4.1).
- **Never a test request,** as v4 §4.2.

### 4.3 The surface cell (S1) and the slate kinds

**The features.** They are read on the option **as shown to the model**: the name, and the
description after `cut_description`, which is the rewrite where D1's draw says so (the lead's
decision 2). Six features:

| | feature | classes | the rules of v4 §4.4 it targets |
|---|---|---|---|
| F1 | name style | five: snake, camel, Pascal, spaced, other (`name_style` of `scripts/s6-route4-blind.py`; R8's four classes are a coarsening of it) | R8, R9, blind's style rules |
| F2 | name length in characters | quintiles | R11 |
| F3 | description length in words | quintiles | R4, R5 |
| F4 | final period | yes, no | R1, R3, R9 |
| F5 | first-word form | three: a capitalised word in -s, not -ss (R2's `third`); imperative (R13's `imperative`); other | R2, R3, R13 |
| F6 | schema-like words (R12's list) | three: 0, 1, 2 or more | R12 |

- **The cut points.** The quintiles' cut points are the 20th, 40th, 60th and 80th percentiles over
  every text that a training or validation slate of v5 can show: the original and the rewrite of
  every tool of the training and validation pools of route/tools and When2Call, and of the intents.
  This is amendment 1's reading for v4's remedy, extended to the rewrites. A value equal to a cut
  point goes to the lower class.
- **The number of cells:** 5 × 5 × 5 × 2 × 3 × 3 = 2,250.
- **The cell of a candidate, under D1.** D1's draw is keyed by (request, kind, tool) and never
  reads the gold (v4 §4.3), so it is known for every candidate before the slate is drawn. A
  candidate's cell is the cell of the text it would show.

**The anchor of a cell.** A slate's cell is the right tool's. Two cases have no right tool:

- **A removed slate (D5):** the absent gold's cell; the extra distractor of D5 comes from it.
- **A request without a tool gold:** the cell of its first own tool.

So every slate, whether a tool or *none* is right, is drawn around one cell.

**The request's own tools are exempt (the lead's decision 1).** These are the conversation's other
tools in the natural and test-like slates, and in T5 the tools of the right tool's server that the
row lists. They are what a real tool list holds, so they are shown as they are, whatever their
cell. Only drawn distractors obey the cell: the random fill, the hard distractors, the refills of
D5 and the false-negative replacements.

**The levels (the widening order).** The candidates of a kind are the pool's tools that pass the
kind's distractor filter (v4 §4.3, unchanged; the right tool, its cluster and the request's other
called tools are never candidates). A slate is built at the lowest level at which its candidates
matching the anchor's key hold **at least 16 tools**:

| level | the key |
|---|---|
| L0 | F1 to F6, exact |
| L1 | F2 and F3 within one class of the anchor's; F1, F4, F5, F6 exact |
| L2 | L1 without F6 |
| L3 | L2 without F5 |
| L4 | L3 without F1 |
| L5 | L4 without F2: F3 within one class, and F4 |
| L6 | the kind's filter alone, **outside the cell**: D7's order for a random fill; the request's nearest tools for a hard slate |

- **Why this order.** The features are dropped in the order of the smallest gaps in v4's C4
  table. The final period and the description's length are kept to the end: they carry the largest
  gaps in the hard cells (validation: a final period on 0.657 against 0.822, 58.0 characters
  against 67.7).
- **Why 16.** It is the reserve of v2's hard plan. A random fill then never draws its 6 or 7
  distractors from fewer than 16 tools.
- **Recorded and counted.** Every slate records its level (field `cell_level`). The build counts
  slates by group, kind, split and level.
- **Capped by C6 (§4.4).** A slate at L6 is counted as "outside the cell". Slates at L5 or L6 are
  capped at 10% of the drawn slates of each judged cell.

**The kinds.** As v4 §4.3, with S1 in each:

| kind | the drawn distractors |
|---|---|
| natural | the random fill: the candidates of the anchor's cell at the slate's level, those of the gold's source first (D7), then the rest of the cell |
| hard | **the 16 candidates nearest to the request** (potion-base-2M cosine between the request's embedding and each tool's `name: description` embedding) among the candidates of the anchor's cell at the slate's level, the right tool excluded; the first ones fill the slate; the false-negative filter applied; replacements from the same list |
| removed | as v4 (D5): the extra distractor from the source slate's rule (the next random candidate, or the next nearest to the request), in the absent gold's cell |
| renamed | `slates::test_slate` with the random fill of the natural kind, then `slates::masked` |
| `v5_tools`, `v5_tools_masked` | the 1,144 validation requests of `v_tools`, one slate each by v2's test rule with the fill of the natural kind, 20% removed, the sources' descriptions (no rewrite), **seed 1238** |

**What the cell removes by construction, and what it does not.**

- **The categorical features.** In a slate built at L0, every drawn option shares the gold's F1,
  F4, F5 and F6 values and its F2 and F3 classes. Every rule that reads only F1, F4 or F5 then ties
  between the gold and every drawn option. These are R1, R2, R3, R8, R9, R13, their inverses, and
  blind's minority and majority rules on style, period and first-word form. Such a rule can pick
  the gold above chance only through the exempt options.
- **The counting rules, R4, R5, R11 and R12, are not removed.** Within one class an argmin can
  still prefer the gold. F2 and F3 bound the gap to the width of a class. F6 makes R12 tie only
  within its classes 0 and 1. R5, on characters, is bounded only through F3, on words.
- **Slates at L1 or above** tie only on the features their level keeps.
- So every rule of v4 §4.4 is still judged.

**C1 under the cell read on the shown text (the lead's decision 2).**

- **The consequence.** A drawn candidate qualifies when the text it would show matches the gold's
  key, so the rewrite flags within a slate can be correlated: a rewritten gold tends to draw
  rewritten distractors, and an original gold original ones.
- **Why C1 stays a fair rate.** P(rewritten | right tool) is 0.5 by D1, since the gold's draw is
  made before and without the slate. C1 asks P(rewritten | other tool option) to be within the bars
  of v4's amendment 1 of 0.5 as well. When both rates are fair, the right option is not more likely
  to be rewritten than a drawn one. A correlation of the flags within a slate then gives a blind
  rule nothing: it makes the flags tie more often between the gold and its distractors, which moves
  R6 (the option rewritten) toward chance, not away from it.
- **What C1 can catch.** The selection by shown text could move the rate of the other options
  away from 0.5, if the rewrites and the originals do not spread over the cells alike. C1 catches
  it with v4's amendment 1 bars, unchanged:
  - per cell and rate, |rate − 0.5| ≤ 3·√(0.25 / n);
  - pooled, within 0.01 of 0.5.
- **When C1 fails.** The failure is first read as a defect of the build, as in v4. If the code is
  right and the failure comes from the selection itself, it cannot be fixed without changing the
  cell, which §12 forbids, and v5 stops before training with the table published.

### 4.4 Checks before any training, on the new data alone (no model)

As v4 §4.4 with amendment 1: the cells, the 200-slate floor, C1 with amendment 1's bars, C2 with
its rules and its tolerance **c + 0.02 + 2·√(c(1 − c)/n)**, C3, C4 and C5. The changes follow, in
the order in which `scripts/s6-route5-checks.py` runs them.

- **C6, the cell holds (new; judged first, before C1 to C5 are read).**
  - **The rule:** in every judged cell, the share of the drawn slates that were built at L5 or L6
    is at most **0.10**. A drawn slate is one with at least one drawn distractor: every hard, renamed
    and removed slate, and every natural or test-like slate whose fill drew a tool. The share counts
    tool-right and *none*-right slates alike. L6 slates are also reported apart, as "outside the
    cell".
  - **Why first:** C6 is a property of the build, like C1 and C3, and it says whether C2 and N1 to
    N5 are read on cell-matched slates at all.
  - **If it fails,** the build stops and v5 stops before training. That is not a defect to fix, since
    the cell cannot change after a check has run (§12). C1 to C5 and N1 to N5 are still computed and
    the whole table is published.
- **N1 to N5, new request-blind rules, judged with C2's tolerance in every judged cell.** None of
  them is used to define the cell. Each reads the option texts with the score and rank of D4 taken
  off, as C2 does, and each comes with its inverse.
  - **N1 digits:** the option whose description has the highest share of digit characters.
  - **N2 the rarest word:** the option whose description holds the word of lowest frequency class.
    The class is ⌊log₂ df⌋, where df is the number of descriptions of the group's training pool
    (originals) that hold the word, after lowercasing, over alphabetic tokens. A word absent from
    the pool is class 0, and ties count as in C2.
  - **N3 the first name:** the option whose name comes first in alphabetical order (case-folded,
    then by code point).
  - **N4 capitals:** the option whose description has the highest share of capitalised words after
    its first word.
  - **N5 centrality:** the option whose description has the highest mean word overlap (Jaccard of
    lowercased word sets) with the other tool options of the slate.
- **Not proposed:**
  - the number of parameters in the tool's schema: the option text shows the name and the
    description only (v4 §3.1), so no system can read it;
  - the position after shuffling: R7 already judges it in every cell.
- **N5 is judged** (the lead's decision 3).
  - With the gold as the anchor, as in v2 and v4, the gold sat at the centre of its hard slate by
    construction.
  - With the request as the anchor, the distractors are close to the request, and the gold is not
    their centre by construction. It may still be close to it, since it is the tool that serves the
    request.
  - N5 measures whether it is, and if N5 fails, v5 stops.
- **Tests on chance alone.** The ten new rule texts (N1 to N5 and their inverses) add about 190
  tests to the 19 judged cells. The tolerance's margin of 0.02 is 1.4 standard errors in a cell of
  about 1,100 slates, and about 0.6 in a cell of 200.
- **Judged on the whole slate, reported on its parts (the lead's decision 1).**
  - C2 and N1 to N5 are judged on the whole slate, as in v4: every tool option, exempt or drawn.
  - They are also reported, not judged, on two restrictions of each tool-right slate: the right
    option with the exempt options only, and the right option with the drawn options only (each
    where it holds at least two options).
  - C4's surface table is reported for the right option against the exempt options and against the
    drawn options separately.
- **The remedy is removed.** If C6, any rule of C2 or any of N1 to N5 fails in a judged cell, the
  build stops and v5 stops before training, with the table published. If C1 or C3 fails, the build
  has a defect and is fixed before anything else, as in v4 (C1: see §4.3).
- **Reported, not judged:**
  - C2 and N1 to N5 by level, in every cell;
  - the share of slates of each level;
  - for each cell, the number of distinct tools used as distractors and the largest share of one
    tool among them, since a small cell can show the same few tools over and over (§4.5).

### 4.5 The difficulty, reported (S1e)

A construction that conditions the distractors on the gold's surface, and anchors the hard ones on
the request, changes the slates in ways that no request-blind rule measures:

- **Harder.** Distractors of the same surface as the gold, and close to the request, make slates
  harder for any system.
- **Easier.** Where a cell is small, the same tools come back as distractors around every gold of
  that cell. A model can learn which tools are rarely right, and no rule that reads one slate sees
  it.
- **A different training-to-test relation.** T2-tools, read for G5, was built by v2's fill from
  the whole pool, not by the cell. A model trained on cell-matched slates meets a different mix
  there.

To make the change of difficulty visible, BM25 and `potion-cos` are scored by the frozen baseline
code (`scripts/route-baselines.py`, v2's fitted values) on `v_tools`, `v4_tools` and `v5_tools`,
and on their masked copies. The results are written in the amendment that sets the pilot's bar,
before the pilot. They are reported, not judged, and nothing is chosen on them.

### 4.6 Validation readings and leakage

- **Validation readings:** as v4 §4.5, from `v5_tools`: `v5_tools_rw`, `v5_tools_masked_rw`, and
  the renamed and natural validation records of the tools group.
- **Leakage:** as v4 §4.6. The request embeddings of §4.1 cover training and validation requests
  only.

### 4.7 Disk

- **The floor.** Every stage that writes (the build, the fresh build, the pilot, the main run and
  the evaluation) first checks that at least **3 GB** are free on the volume of the data folder
  (`MIN_FREE_GB=3` in `scripts/build-route5-data.sh` and `scripts/train-route5.sh`). If not, it
  stops: a technical stop, resumed once the space is there, and reported.
- **Why 3 GB.** About 1.5 GB of new files (§13) and as much again for the system and the other
  work on the machine. The 8 GB floor used elsewhere in the project cannot hold on this machine
  today (about 5.5 GiB free on 18/11/2026).
- **What may be removed when v5's build starts** (the lead's decision 6): `build4/records/` and the
  other rebuildable outputs of v4's build under `<data>/s6-route/build4/`. Not `build4/teacher/`,
  whose 13,687 rewrites are reused, and not the report committed in `runs/s6-route4/`. The removal
  is listed with its sizes in the build's log.

## 5. The run (`scripts/train-route5.sh`)

As v4 §5: the smoke, one pilot of 20 working minutes, the main run of 120, the calibration,
export, τ_ask and the fit of β and τ_ask5 on `v5_tools`, then the evaluation. Readings at every
kept checkpoint of the pilot are as v4, on `v5_tools`, `v5_tools_masked` and the two probe copies,
with `v4_tools` and `v_tools` reported. The disk floor of §4.7 applies to every phase.

**The pilot rule:**

1. **Abort** if the model part's accuracy on `v5_tools` is below `potion-cos` on `v5_tools` −
   **0.15**. The value of `potion-cos` on `v5_tools` is written in the amendment of §4.5.
2. **Abort** if the rewritten distractor is the answer on more than **0.30** of the tool-gold
   slates of `v5_tools_masked_rw`.
3. Nothing else is chosen on the pilot.

## 6. The fresh test files (T5)

As v4 §6 and amendment 1, item 1, with the same source, commit, row rule and seed 1236. The
filters, the caps (10 per gold cluster, 40 per server), the 2,000 slates, the 20% removed and the
600-slate floor are unchanged.

- **The Toucan fetch of v4 is reused as it is.** The three parquet files are hash-checked again
  against `<data>/MANIFEST4.json` before the fresh build reads them. T4 was never built, so no file
  is replaced.
- **§6.2, item 5, the one change.** "Random Toucan tools of other servers by the same filter,
  drawn within the gold's surface cell" now means v5's cell (§4.3):
  - the features F1 to F6 on the Toucan tool as shown (no rewrite in T5);
  - the quintiles of F2 and F3 over Toucan's 1,770 tools (amendment 1, item 1);
  - the levels L0 to L6 with the minimum of 16 candidates.

  The fall-back that v4 wrote (the nearest length quintile with the same style and end, then the
  same style alone) is replaced by these levels. The tools of the right tool's server that the row
  lists come first and are exempt, as in v4. T5 has no hard slate.
- **Counted on the built file and written in the counts amendment before any scoring:** the rules
  of §4.4 (C2 and N1 to N5, whole slate and the two restrictions), the slates by level and C6's
  share. They are reported, not judged: a rule above its tolerance, or a C6 share above 0.10, is
  reported with the gate and does not change the file.
- **Files:** `t5_tools.jsonl`, `t5_tools_masked.jsonl`, `t5_tools_rw.jsonl`,
  `t5_tools_masked_rw.jsonl` and `t5_tools.servers.jsonl`. Their hashes go to
  `runs/s6-route5/test-files.json`, and `runs/s6-route5/FRESH-OPEN` follows v4 §6.3.

## 7. Evaluation

As v4 §7, on T5-tools, its masked copy and the two probe copies, and on T2-tools for G5. The
systems are v4's, and the published s6-route pair (v3) is included for comparison.

## 8. Gate (every condition must hold)

As v4 §8, with T5-tools in place of T4-tools and no bar moved:

- **G1:** the pair − BM25 and the pair − `potion-cos-4bit` each have the lower bound of the 95%
  slate bootstrap above 0, and the pair is above its model part.
- **G2:** (a) ≤ **0.1929** and paired ≤ **+0.03**; (b) the pair ≥ `potion-cos-4bit` − **0.02**;
  (c) *none* on **0.10 to 0.30** of the masked slates.
- **G3:** ECE ≤ **0.05**.
- **G4:** parity max |Δp| ≤ 1e-3 with the same answer on ≥ 199 of 200 slates, a median ≤
  **60 ms**, and ≤ 4,000,000 bytes.
- **G5:** T2-tools ≥ **0.8668**.

The headline (servers resampled), the report's sentence and the announced goals are as v4 §8.

## 9. Reported, not gated

As v4 §9, plus:

- the slates by level, C6's shares, and C2 and N1 to N5 by level and on the exempt and drawn
  restrictions;
- the baselines of §4.5 on the three validation files;
- the distinct-distractor counts of §4.4;
- the teacher's counts of the false-negative job (items, calls, refused, dropped requests,
  replaced distractors, slates completed from the cell's random fill).

## 10. Disclosure (written into every place that reports v5)

As v4 §10, plus:

- **v5 was designed after v4's data checks were read.** The full table (`runs/s6-route4/checks.md`,
  the five highest rules of every cell) and the C4 surface table (`checks.json`) were seen. The
  cell's features, its classes and the order of its levels were chosen from them, after the failing
  rules were known. N1 to N5 and the request anchor of the hard slates were chosen after that
  reading.
- **The hard slates of v2's training data carried request-blind cues** (v4's result block). They
  are disclosed in the cards of `s6-route` and `s6-route-nano`.
- **The teacher sees about 69,000 new items** of training and validation requests, each with its
  hard slate, as in v2.
- **Toucan's files were fetched for v4 and no row text was read** (amendment 1 of v4). Writing this
  document opened no data file.

## 11. What a failure would mean

As v4 §11, with its first point replaced and these points added:

- **C2 fails.** The surface cell does not make the slates blind. No model is trained, and the table
  is published. The cue is described as a limit of v2's training data where it is v2's
  construction.
- **N5 fails.** Even with the hard distractors anchored on the request, the right tool is the
  slate's centre in word overlap: a request-blind cue that the hard kind carries whatever its
  anchor, and that test-like slates do not. No model is trained, and the table is published. Hard
  negatives that are built around a request, without a cue of that kind, need another construction
  and their own pre-registration.
- **Another N rule fails.** A request-blind cue that the cell does not see remains. No model is
  trained, and the table is published, naming the rule and the cells.
- **C6 fails.** In at least one judged cell, more than 10% of the drawn slates could not be
  matched to their cell beyond L4. At this size of pool, the cell is too fine for that group: the
  intents' pools, the most exposed, hold at most 150 intents per source (287 intent descriptions
  in all). No model is trained, the
  table is published with the shares by level, and a coarser cell would need its own
  pre-registration.
- **G5 fails alone:** training on cell-matched slates cost accuracy on a file built by v2's fill
  (§4.5). The result is reported as such; it is not a replacement.

## 12. Forbidden

As v4 §12, plus:

- changing the cell's features, its classes, its minimum of 16, its levels or the cap of C6 after
  any check has run;
- moving a rule between "judged" and "reported" after the freeze;
- adding a remedy;
- using v4's built records (`<data>/s6-route/build4/records/`) or its slate files for any choice of
  v5, beyond the published report;
- running the latency check of v4 §3.4 again and taking option (ii') from it, unless §3 says the
  rendering has changed;
- embedding or sending to the teacher a request of a test or fresh file;
- running a stage below the disk floor of §4.7.

## 13. Cost

| item | amount |
|---|---|
| teacher | the false-negative job, needed in every case: **69,290 items, ≈ 3,465 calls** of 20, ≈ 2 to 4 h, one or two usage windows (§4.2); the `fresh-tools` job ≈ 100 to 150 calls; no rewrite job |
| CPU | the request embeddings (potion-base-2M, training and validation requests) ≈ 5 min; the plan with the cells and the nearest tools to each request within its cell ≈ 10 to 20 min; the build and the checks ≈ 10 min (v4's latency stage, build and checks took 2 min 22 s together); the baselines on the three validation files ≈ 5 min; Toucan's hash check, read, embedding and clustering ≈ 20 min; the pilot readings ≈ 15 min; the baselines on T5 and T2 ≈ 20 min |
| GPU | as v4: smoke 3 + pilot 20 + main 120 + calibration and export 5 + evaluation 15 ≈ **163 working minutes**; ≈ 23 if the pilot rule aborts; none if a check of §4.4 stops the build |
| disk | the request embeddings ≈ 0.1 GB; a new build ≈ 0.3 GB (v4's build folder is 273 MB, of which the rebuildable part may be removed, §4.7); the false-negative items and answers ≈ 0.05 GB; checkpoints ≈ 1.1 GB; evaluation ≈ 0.1 GB; Toucan's 0.43 GB is already on disk: **≈ 1.6 GB new**; floor **3 GB free** before every writing stage (§4.7), against about 5.5 GiB free on 18/11/2026 |

## 14. Code to write before any run (by file)

**Reused as is.** All of the following stay as written for v4:

- D1 (`rewrite_draw`), D3, D5, D7 and the *none* share of `crates/sky-data/src/route/v4.rs`;
- `cluster::nearest_all` (`crates/sky-data/src/route/cluster.rs`), whose `skip` closure carries
  the cell's key;
- the readers `sky-data route toucan-rows` and `toucan-stage`;
- the `fresh-tools` guard of `sky-data teacher`;
- D4 in `sky_infer::route` with its fixture, and the `skycmd-s2.route.render` key in the engine
  and in the sky-web router;
- the uncached mode of `route3-node.mjs`;
- `s2-decide --keep-checkpoints`;
- `scripts/s6-route4-blind.py`'s features and rules;
- `scripts/route-teacher.sh`, whose `false-negatives` mode is v2's.

**To write:**

1. **`scripts/route-embed.py`:** `--requests`, which writes `emb/potion-requests.npy` and its
   meta for the training and validation requests, refusing a test or fresh request. A test that it
   gives `potion-cos`'s request vectors back.
2. **`crates/sky-data/src/route/v4.rs`, module `surface`:** F2, F5 and F6, the cut points over
   both texts of every tool, the key at each level, and `cell_level` on every record. Unit tests:
   - every drawn option of an L0 slate shares the anchor's key, and the exempt tools are kept;
   - the levels widen in order, and the minimum of 16 is honoured;
   - D1's draw is unchanged by S1 (the same draw for the same request, kind and tool);
   - the removed slates are drawn around the absent gold's cell;
   - determinism under the seed.
3. **`crates/sky-data/src/route/v5.rs` (new), with `sky-data route v5-plan` and `v5-slates`:**
   - the hard plan: the 16 tools nearest to the request within the cell, the right tool excluded,
     with the stream `route/plan/<id>` kept;
   - the items of the `false-negatives` job for every hard slate, with an answer reused by id where
     one matches;
   - the replacements and the completion from the cell's random fill;
   - `v5_tools` and its copies (seed 1238);
   - the build counts by level and the exempt or drawn flag of every option.

   A unit test on an invented pool: the hard distractors follow the request's nearest tools and not
   the gold's.
4. **`crates/sky-data/src/route/toucan.rs`:** item 5 of §6.2 by v5's cell (`--rule v5`), the
   files `t5_*`, and the hash check of the fetched parquet files against `MANIFEST4.json`.
5. **`scripts/s6-route5-checks.py` and `scripts/test_s6_route5_checks.py`:**
   - v4's checks imported as they are;
   - C6 first, then N1 to N5 judged;
   - the exempt and drawn restrictions, the tables by level and the distractor counts;
   - no remedy: exit 2 on C6, on any C2 rule or on any N rule;
   - tests: a pass and a fail of C6 at 0.10, and N5 above chance on an invented slate built around
     the gold.
6. **`scripts/build-route5-data.sh`:** the stages `embed-requests`, `teacher-jobs` (the
   false-negative items), `build`, `checks`, `baselines-val` (§4.5), `fresh`, `probes` and `counts`.
   - There is no `fetch` stage (Toucan is on disk, hash-checked) and no `latency` stage (§3.4).
   - `MIN_FREE_GB=3` is checked before every writing stage, and the rebuildable outputs of `build4/`
     are removed at the start of `build`.
7. **`scripts/route-baselines.py`:** a `v5val` split.
8. **`scripts/train-route5.sh`, `scripts/s6-route5-pilot.py`, `scripts/s6-route5-fit.py`,
   `scripts/s6-route5-gate.py` and `scripts/test_s6_route5_gate.py`:** from v4's, with the names
   given at the top of this file. `train-route5.sh` checks `MIN_FREE_GB=3`. Also the t5 split of
   `scripts/route-eval.sh`, refused until `FRESH-OPEN`, and the Python reference under D4 for G4.
9. **`DATA_LICENSES.md`:** the new false-negative answers (GLM-5.3-Flash, MIT, a filter of the
   training and validation hard slates).

## 15. Choices (the value taken at the freeze; the alternative is recorded)

| | choice | taken | alternatives | why |
|---|---|---|---|---|
| O21 | the cell's features | F1 to F6 | v4's three (style, period, word quintile); adding a digit in the name | each targets a rule that failed in v4, or its neighbour; R10 (digits) failed nowhere and stays judged |
| O22 | the length classes | quintiles (the lead's decision 4) | tertiles (fewer widenings, a wider residual); deciles (a narrower residual, most slates widened) | v4's §6.2 used quintiles |
| O23 | the minimum and the levels | 16 candidates; L0 to L6 in the order of §4.3 (the lead's decision 4) | 8 candidates for random fills and 16 for hard slates | one rule for every kind; 16 is the hard plan's reserve |
| O24 | the request's own tools | exempt from the cell; the checks judged on the whole slate and reported on the exempt and drawn options apart (the lead's decision 1) | filtered by the cell too | they are what a real tool list holds; one rule in training, validation and test |
| O25 | the cell of an option under D1 | the text as shown to the model (the lead's decision 2) | the source's description (as v4's remedy code) | the rules read the shown text; C1 stays fair (§4.3) |
| O26 | the hard slates' anchor, and N5 | the tools nearest to the request within the cell; N5 judged (the lead's decision 3) | the tools nearest to the right tool, as v2 and v4; N5 reported only | the gold's neighbours put the gold at the centre (N5's cue); a deployed router gets the tools retrieved for the request |
| O29 | the cap on widened slates | C6: at most 0.10 of the drawn slates of each judged cell at L5 or L6; the build stops beyond (the lead's decision 4) | no cap (counts reported only) | a check of cell-matched slates means little if many slates are not matched |
| O30 | the fresh files | `t5_*`, from v4's Toucan fetch, hash-checked again (the lead's decision 5) | v4's `t4_*` names | item 5 changed; no `t4_*` file was ever built |
| O31 | the disk | the floor of 3 GB free before every writing stage; v4's rebuildable outputs removed at the build's start (the lead's decision 6) | the project's 8 GB floor | 8 GB cannot hold on this machine today; 3 GB is twice v5's ≈ 1.6 GB |
| O27 | the false-negative job | on every new hard slate of training and validation | the hard slates unfiltered (v2's fallback when moderation refuses) | false negatives teach noise (v2 §2); request-anchored distractors are more often able |
| O28 | `v5_tools` seed | 1238 | | the next free seed after v3's 1235, T4's 1236 and `v4_tools`' 1237 |

## 16. Why not relax the tolerance instead

The tolerance of C2 was frozen on 17/11/2026, before any data of v4 existed. Moving it after its
checks had failed would choose the bar from the result it judges. The failures are not marginal
either: in the hard cells of the tools group, the best rule is 0.045 above chance in training and
0.105 on validation. A margin of about +0.05 that a rule reading no request can find in thousands of
slates is the kind of signal a model trained for two hours on those slates can also find. Whether
it would have is not known, since no model was trained.

## 17. The lead's decisions on the draft (18/11/2026)

1. The request's own tools are exempt from the cell; only drawn distractors obey it. The checks
   are judged on the whole slate and reported on the exempt and drawn options apart (O24, §4.3,
   §4.4).
2. The cell is read on the text as shown to the model, and C1 keeps the bars of v4's amendment 1
   (O25, §4.3).
3. The hard distractors are the tools nearest to the request, within the cell, the right tool
   excluded. v2 anchored them on the right tool (§4.1). N5 is judged, with no remedy (O26).
4. Quintiles, the minimum of 16 and the levels L0 to L6 are accepted. The slates at L5 or L6 are
   capped at 0.10 of the drawn slates of every judged cell by C6, which stops the build (O22, O23,
   O29).
5. The fresh files are `t5_*`, from v4's Toucan fetch, hash-checked again (O30, §6).
6. v4's rebuildable outputs under `build4/` may be removed when v5's build starts, except
   `build4/teacher/` and the committed report. v5 states its own disk floor of 3 GB and stops below
   it (O31, §4.7).

## 18. Amendments

**Amendment 1 (18/11/2026, decided by the lead, before any build of v5).** At the freeze, the
group with the smallest pools (the intents) could fail C6 on arithmetic alone, before any slate
existed. A design that stops on arithmetic known in advance is corrected before the build. This
amendment changes:

- the minimum number of candidates for the intents group;
- when C6 is computed;
- the costs, the code and the meaning of a failure.

Nothing else changes.

**What was looked at.**

- **The feasibility count.** `scripts/s6-route5-cells.py`, standard library, run once; its output
  is `runs/s6-route5/cells.json`.
- **What it reads:** the pools only, read-only in `<data>/s6-route/`:
  - v2's stage tool table (`build/stage/tools.jsonl`);
  - the intents' files (`build/teacher/describe-intent*.jsonl`);
  - v2's and v4's rewrite items and answers (`build/teacher/` and `build4/teacher/`);
  - v2's aggregate request counts per group, split and source (`build/counts/split.json`).

  The stage tool table also lists test tools: only the 27,950 tools of the pools are kept, and no
  test tool enters a count.
- **What it does not read:** any request, any slate, any test or fresh file, Toucan's rows. No
  slate of v5 was built, no request was embedded, no teacher call was made and no model was run.
- **When this amendment was written.** It was written after v4's table had been seen, where the
  intents' hard cells failed R11 (the shortest name), and from this count alone.

**The count** (expected candidates under D1: each tool anchors in its original and its rewritten
text with weight ½, each candidate counts ½ per text; anchors weigh alike; the kind's distractor
filter is not applied, so the counts are an upper bound):

| pool | tools | non-empty cells, original / rewrite | min 16: by L4 | min 16: L5 or L6 | min 8: L5 or L6 | min 4: L5 or L6 |
|---|---:|---:|---:|---:|---:|---:|
| tools, training (also When2Call training) | 26,221 | 1,223 / 793 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| tools, validation | 1,606 | 528 / 438 | 0.9975 | 0.0025 | 0.0012 | 0.0000 |
| tools, validation, originals only (`v5_tools`) | 1,606 | 528 / – | 0.9975 | 0.0025 | 0.0000 | 0.0000 |
| When2Call, validation (training and validation pools) | 27,827 | 1,236 / 798 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| intents, Banking77 (training and validation) | 77 | 32 / 35 | 0.8636 | 0.1364 | 0.0260 | 0.0065 |
| intents, MASSIVE (training and validation) | 46 | 16 / 21 | 0.7609 | 0.2391 | 0.0761 | 0.0000 |

- **The levels at a minimum of 16:**
  - tools training: L0 0.8491, L1 0.1322, L2 0.0115, L3 0.0062, L4 0.0010;
  - tools validation: L0 0.0567, L1 0.5290, L2 0.2786, L3 0.0660, L4 0.0672, L6 0.0025;
  - Banking77: L1 0.1039, L2 0.1948, L3 0.5649, L5 0.0779, L6 0.0584;
  - MASSIVE: L1 0.3913, L2 0.0978, L3 0.2717, L5 0.0109, L6 0.2283.

  Every minimum from 4 to 16 is in `cells.json`.
- **The pools by split.** They follow the split of each tool's rewrite item, which gives 26,221
  training and 1,606 validation tools where v2's counts give 26,277 and 1,550. The 56 tools apart
  are, by inference, training-pool golds of When2Call validation requests: When2Call's validation
  pool is both, so they belong to it either way.
- **A correction of §11.** The intents' training pools are Banking77 (77 intents) and MASSIVE (46
  intents outside its four held-out scenarios). CLINC150 is not a training source. §11's "at most
  150 intents per source (287 intent descriptions in all)" counted CLINC150's 150 and MASSIVE's
  held-out intents.

**The decisions.**

1. **C6 moves to plan time** (the lead's decision a).
   - **Before:** §4.4 ran C6 first among the checks, but after the build, so after the
     false-negative job.
   - **The new order of the stages:** `embed-requests`, `plan`, **C6**, the false-negative teacher
     job, `build`, then C1 to C5 and N1 to N5.
   - **C6 at plan time** reads the levels of the planned drawn slates (the plan's counts by group,
     kind, split and level), with the bar of §4.4: at most 0.10 at L5 or L6 in every judged cell. It
     needs no teacher answer. If it fails, v5 stops before any teacher call.
   - **Levels can rise after the false-negative replacements.** A replacement comes from the
     slate's own level. When those candidates are used up, it comes from the next level, and the
     slate's level becomes the highest level any of its drawn distractors came from. A request that
     the teacher drops leaves the count.
   - **C6 is therefore confirmed after the build,** with the same bar on the final levels. If it
     fails there, v5 stops before training, as §4.4 says.
2. **The minimum number of candidates, by group** (the lead's decision b).
   - **Tools and When2Call: 16, as frozen.** The count shows at least 0.9975 of their pool tools
     reaching it by L4, above the 0.90 the lead asked.
   - **Intents: the number of distractors the slate draws, plus 3.** The draw includes D5's extra
     distractor for a removed slate, so the minimum runs from 4 (one drawn distractor) to 10 (a
     removed renamed slate, seven). The widening order L0 to L6 is unchanged. In the count no
     intents anchor is first matched at L4.
   - **Why:** a fixed 16 is out of reach in pools of 77 and 46 (L5 or L6 on 0.1364 and 0.2391 of
     the anchors). A fixed 4 cannot fill a slate that draws 5 to 7 distractors. Three spare
     candidates leave each draw a choice.
   - **The expected share of intents slates at L5 or L6 under this minimum,** by cell, the sources
     weighted by v2's requests:

     | kind | training | validation |
     |---|---:|---:|
     | natural | 0.0214 | 0.0236 |
     | hard | 0.0214 | 0.0236 |
     | removed | 0.0295 | 0.0330 |
     | renamed (6 or 7 drawn distractors) | 0.0593 | 0.0711 |

     MASSIVE alone gives 0.0254, 0.0357 and 0.0804; Banking77 alone 0.0162, 0.0213 and 0.0312.
3. **The intents fall-back is not triggered** (the lead's decision c).
   - **The condition** was that the count shows the intents group cannot hold C6 at 0.10. It does
     not: every intents cell is expected at 0.0214 to 0.0711. C6 is therefore judged in every judged
     cell, the intents' included.
   - **The margin is small.** The largest expected share, 0.0711 in the renamed validation cell, is
     0.029 below the bar, and the count is an upper bound on the candidates: the distractor filter
     (the gold's cluster) and an uneven use of intents as golds can both raise the share.
   - **What is not adopted.** The pre-written consequence of decision c is not adopted: C6
     reported in the intents' cells, and the intents group dropped from training and validation if a
     blind rule failed in an intents cell and in no other. Had it been triggered, it would have
     removed 89,051 of the 178,855 training slates of v4's size (0.498, counted from v4's C3
     table). Any reading of T2-clinc or T2-massive would then have been outside every training
     family; v5's evaluation (§7) reads neither.
   - **A blind rule that fails in an intents cell** stops v5 as anywhere else (§4.4). No other
     remedy.
4. **The costs.**
   - The count: 10 s of CPU, done.
   - C6 at plan time: about 1 min.
   - The false-negative job (about 3,465 calls) runs only if C6 holds at plan time. A stop there
     costs no teacher call and no GPU minute.
   - Everything else as §13.
5. **The code** (§14, items 3, 5 and 6).
   - `sky-data route v5-plan` writes each planned slate's level and the counts by group, kind,
     split and level, with the minimum by group (16; for the intents the drawn distractors plus 3).
   - `scripts/s6-route5-checks.py --plan` reads those counts and judges C6 (exit 2 above the bar).
     The same script confirms C6 on the built records.
   - `scripts/build-route5-data.sh` gains the stage `c6` between `plan` and `teacher-jobs`; the
     stage `teacher-jobs` refuses to run without `c6`'s marker.
   - The tests: the minimum of an intents slate is its drawn distractors plus 3; a replacement
     taken from the next level raises the slate's level; C6 at plan time and after the build read
     the same bar.
6. **What a failure would mean** (§11), in place of its point on C6.
   - **C6 fails at plan time.** In at least one judged cell more than 10% of the drawn slates
     cannot be matched to their cell beyond L4, even with the intents' smaller minimum. v5 stops
     before any teacher call, and the plan's counts by level are published. A coarser cell would
     need its own pre-registration.
   - **C6 fails only after the build.** The false-negative replacements pushed slates out of their
     cells. v5 stops before training, and the counts are published with the teacher's
     replacements.
