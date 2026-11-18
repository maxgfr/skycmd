# s6-route v6 (v5's data with one fix, trained, and gated on the model's request-blind reading): pre-registration

**Frozen on 18/11/2026 by the lead, before any computation of v6 and before its code is written; every later change is a dated amendment that precedes the computation it concerns.**

**Amendment 1 of the same day (18/11/2026), before any computation of v6 (§17):** the fix of §4.8 is extended to every slate of the When2Call group, none of which shows an exempt tool any more.

**Amendment 2 of the same day (18/11/2026), before any computation of v6 (§17):** the readings of the code fixed before any computation (a drawn own tool is a drawn tool; the engine unchanged, the pair's B1 dropped; the placeholder; the stages; G4's input; what remains before the gate).

- **Nothing of v6 exists yet:** no model trained, no record rebuilt, no fresh file built, no
  teacher call, no system scored, no blind copy written. To write this document:
  - no record and no request was opened;
  - read: v5's published report (`runs/s6-route5/checks.md` and `checks.json`, the result block at
    the end of `docs/s6-route5-prereg.md`), v4's (`runs/s6-route4/checks.md`, `checks.json`), the
    code of D4 (`crates/sky-infer/src/route.rs`, `crates/sky-data/src/route/v4.rs` and `v5.rs`,
    `crates/sky-infer/src/bin/sky-route.rs`) and the tokenizer file of potion-base-2M (a few
    entries looked up);
  - counted: the option counts and the target position of `v5_tools.jsonl` and its copies
    (§4.3; no request text was printed or kept);
  - hashed: the bytes of v5's build (§4.1);
  - read, for the fix of §4.8: the When2Call cells and N6 families of `runs/s6-route5/checks.json`
    (slates by level and by gold).
- **v6 is v5 with two changes.**
  - **What decides.** v5 asked the data to be blind before any model was trained. v6 publishes the
    data checks without letting them stop the run (with one pre-written exception, §4.8), and gates
    on the trained model's reading with the request hidden (G6, §8).
  - **One data fix** (§4.8). v5's N6 failure, +0.24 on When2Call's *none* decision, is of the size
    of the cue that started this line, and v6 does not knowingly train on it. The lead's decision 3
    of v5's amendment 2 is withdrawn for the hard slates: only When2Call's hard no-call slates are
    rebuilt, and every other record of v5 stands as built.
  - Everything else is inherited, section by section, from `docs/s6-route5-prereg.md` (frozen on
    18/11/2026, with its amendments 1 and 2 and its result block) and, through it, from
    `docs/s6-route4-prereg.md` (frozen on 17/11/2026, with its amendment 1).
  - "As v5 §x" means that v5's text applies word for word, with these names changed:
    - `runs/s6-route5*` becomes `runs/s6-route6*`, and `models/s6-route5` becomes
      `models/s6-route6`;
    - the `route5` scripts become the `route6` scripts of §14;
    - τ_ask5 becomes τ_ask6.
  - The data's own names are kept: `build5/`, `v5_tools`, T5 and `t5_*`. Only what changes is
    restated, and every bar is given as a number.
- **Fixed later, by rules written here, each before any training:**
  - the head window, as v4 §3.3;
  - the pilot's potion-cos bar, as v5 §4.5;
  - the counts amendment of T5, as v5 §6;
  - the placeholder, if the check of §4.3 refuses the first one;
  - the hashes of the two rebuilt When2Call record files, by an amendment after the rebuild and
    its checks (§4.1, §4.8).

**Why this experiment.**

- **The two stops.** v4 and v5 asked the training and validation data to be blind to every
  request-blind rule, within a fixed tolerance, before any model was trained. Twice the data
  stopped the build.
- **What v5 found.** v5's surface cell removed every one of v4's 17 failures, and the checks found
  17 failures of C2 and the N rules (the result block of `docs/s6-route5-prereg.md`):
  - in the pick rules, small departures of +0.028 to +0.076 above chance in pools of real tools.
    Some of them come from the cell's own construction: the intents' R12 inverse appears only
    where the cell widens to L2;
  - on When2Call's *none* decision, one departure that is not small: N6, +0.24 above the *none*
    share.
- **The lesson the lead draws, with its limits.**
  - **What the two builds suggest.** When distractors are drawn by any process other than the one
    that produced the right tool, some surface statistic separates them a little, and a
    construction that closes one gap can open another. Two builds are evidence for this, not a
    proof.
  - **The failures are not noise.** v5's table holds 1,088 tests: 19 judged cells times 56 rule
    texts (54 in the two validation files), and 28 for N6. At 2 standard errors alone, without
    the +0.02 margin, about 2.3% of them would fail by chance, about 25. With the margin about 0.5
    would, in expectation. The 17 failures are real departures.
  - **The checks answer another question.** A purity bar on the data, tested rule by rule and cell
    by cell, says whether one statistic of the option texts departs from chance. What mattered from
    the start is whether a **model** trained on the data answers without the request: the shortcut
    that started this line was found in a model (a rewritten distractor picked on 0.8311 of the
    masked validation slates).
  - **What the model test sees that the data checks cannot:** cues that no listed rule names, and
    many small cues used together.
  - **What it cannot see:** a cue used jointly with the request. The blind reading measures what
    the model does when the request says nothing (§11).
- **What v6 does.** It tests the model, not the data. It trains on v5's built data as they stand,
  with one exception: When2Call's hard no-call slates are rebuilt without the exempt tools that
  carried part of N6's cue (§4.8). The records' hashes are listed in §4.1 before anything runs. It
  gates on the trained model's request-blind reading.

## 1. The question

As v5 §1 (v4 §1, read on T5-tools), and one more:

4. Does the trained system, shown a slate whose request is replaced by a placeholder that matches
   no option, pick the right tool no more often than chance plus 0.05 (G6), and does its *none*
   answer then not depend on whether *none* is right?

## 2. What changes from v5, and why

| # | change | evidence | where it is read |
|---|---|---|---|
| B0 | **One data fix:** When2Call's hard no-call slates hold no exempt tool, and are rebuilt with their false-negative items; one stop rule on N6 after the rebuild (§4.8) | N6 fails in When2Call's two "other" families at +0.24 (v5's table) | §4.8 |
| B1 | **v5's built data are the training and validation data, as they stand,** except the records of B0 (hashes, §4.1) | v5's build is complete (C6 and C3 hold); its stop came from the judged rules | §4.1 |
| B2 | **The request-blind reading** of a slate: the request replaced by a placeholder, every lexical score and rank that D4 writes as for a request that matches no option (§4.3) | a model can use cues that no listed rule names, and several at once; only a model shows it | §4.3 |
| B3 | **The data checks become reported:** C1 to C6 and N1 to N6 are run on every file v6 reads and published; none stops v6 | v4's and v5's tables (§4.4) | §4.4 |
| B4 | **Two pilot aborts on the blind reading** (§5) | a model that answers from the options alone after 20 minutes would do so after 120 | §5 |
| B5 | **G6, "the model does not answer without the request",** on T5-tools (§8) | as B2 | §8 |
| B6 | **A diagnostic of the published files** (reported, §9.1): the published model part and pair in the blind reading on `v_tools` and on T2-tools, files already read | v2's hard slates carried request-blind cues (v4's result block); whether the published model used them is not measured | §9.1 |

**Not changed:** the system (v5 §3, with D4 option (i)), the recipe and budgets (v5 §5), the
fresh test T5 and its rule (v5 §6 with amendments 1 and 2), the evaluation's systems and metrics
(v5 §7), G1 to G5 and the headline (v5 §8), the probe copies, the disk floor (v5 §4.7).

## 3. The system

As v5 §3: v4 §3.1 to §3.3 and §3.5, D4 option (i) from v4's latency check (24,622 µs against
50,000), the embedding part `potion-cos-4bit` (θ = 0.396120, T = 0.0671021), the fusion
`p(i) ∝ (p_model(i) · p_embedding(i))^β` with β and τ_ask6 fitted on `v5_tools` by v3's rules.
The head window is chosen by v4 §3.3's rule on v5's records (`scripts/train-route6.sh window`).

## 4. Data

### 4.1 The inputs, as built (B1)

The training and validation data are v5's build under `<data>/s6-route/build5/`, written on
18/11/2026 by the code of 9c2da9b. They are described by v5's tables (`runs/s6-route5/checks.md`).
Nothing in them is rebuilt, filtered or reweighted, except the 1,043 When2Call hard no-call
slates of §4.8.

- **The rebuild writes a full copy under `<data>/s6-route/build6/`.** Every file of the table
  below keeps its hash there, except `records/when2call/train.jsonl`, `records/when2call/val.jsonl`
  and `counts/v5-slates.json`.
- **The new hashes.** The rebuilt files' hashes, and those of the new teacher files, are written
  by an amendment after the rebuild and its checks, before training. That amendment also gives the
  records' counts.
- **A difference elsewhere** is a technical failure of the rebuild: it is fixed before anything
  else, and reported.

**The hashes** (sha256, bytes), copied to `runs/s6-route6/inputs.json` before anything runs. The
stage `inputs` of §14 refuses to go on if one differs.

| file | bytes | sha256 |
|---|---:|---|
| `records/intents/train.jsonl` | 95,646,509 | `6a68f686818d592893d01178c0c8bd3f95fe5ad5817f8da1e56ef92fb81bf48b` |
| `records/intents/val.jsonl` | 11,393,779 | `b810a7fcd1dca65961d51bea0f6dc3e2820c6ff70a1290001a28385b54101490` |
| `records/tools/train.jsonl` | 99,350,230 | `47448bcfb0641fe0da57b2737185a577fb0c6520fda01fac3fd692cdb4f67463` |
| `records/tools/val.jsonl` | 5,095,890 | `488d5c9194dee948fbe2a9777684787d376a5e277b4d32b2654fef8cd2b44e73` |
| `records/when2call/train.jsonl` | 15,944,301 | `b121f86b0875fceb887a8b52974ecb2147ff6ddacd80a1067cf9cc35500d060a` |
| `records/when2call/val.jsonl` | 965,617 | `328855c43ed683137027c498cd9d444c94630172edf25c493215370322a2cde4` |
| `val/v5_tools.jsonl` | 1,603,383 | `767a6699e5c16633a2d00a87721d14de386295ada79555454b791842f210f704` |
| `val/v5_tools.model.jsonl` | 1,755,565 | `87de9c6875a3bcb1fcee6de2856762280a0f2fc697bc3e746d152efa3ac4084e` |
| `val/v5_tools_masked.jsonl` | 1,513,147 | `790660875170506ab28be0cabc13b59c36665b5530e850fb577716015e23d0ac` |
| `val/v5_tools_masked.model.jsonl` | 1,665,328 | `e6ea0f85b3b372772549948d6de2968f93f1ceecb65dfce1be514b8ba1d90ad6` |
| `val/v5_tools_rw.jsonl` | 1,621,100 | `495c2e357aeb9f49056dae689cb5d838ce39665cea5765d81909ada843cdf14f` |
| `val/v5_tools_rw.model.jsonl` | 1,773,282 | `1b10219e8b3d1fe75966512163d1d32f69a1f4bc9eb249a68c233730903f9fdf` |
| `val/v5_tools_masked_rw.jsonl` | 1,530,864 | `fc29557cde676596457b4ff2d2c32db7c7bc685dd2042da161b103a27f2b3f46` |
| `val/v5_tools_masked_rw.model.jsonl` | 1,683,045 | `b41817aba2de222ba5139b859fc38f59d3c3063db2418ee8b7fff7b0ad524062` |
| `val/r_tools_natural.model.jsonl` | 1,347,426 | `00fa2c8c62858f51f2bae64d19f5917bffbd622c64ab126738cb0a9deb8736a5` |
| `val/r_tools_renamed.model.jsonl` | 1,720,186 | `87eb3e1bbbb70c0d44756c5e217f636143a11b00e9a65849340352b1ab10eb11` |
| `val/v4_tools.model.jsonl` | 1,663,413 | `1fcc42718bf4b20b3ea9e508ff4b50863fb3ab2b93627d8772ae7f7a9d619bf6` |
| `val/v4_tools_masked.model.jsonl` | 1,579,075 | `1248da55e99dabd62e0681a35da4bd992fbd55d6978e2117bcbea7d65324e1c3` |
| `val/v_tools.model.jsonl` | 1,729,811 | `c72afb5955e1737e33ac949d851fce3920925901debebbe82575869ffd080ef1` |
| `val/v_tools_masked.model.jsonl` | 1,641,952 | `d18ee12b4e5cbcacc7e78f5cda443725e23fab1c1acb5af760afe8c5c93555a3` |
| `teacher/false-negatives.items.jsonl` | 36,201,112 | `683e9f75e3b42d36f0c6e547e66e389b449c95dac1bdc920985be49915d14bb0` |
| `teacher/false-negatives.jsonl` | 2,805,563 | `145fd7d8fbace537712975379fc9cadc0e6b824c3d73422b31f3e27cc982b3cf` |
| `teacher/false-negatives.refused.jsonl` | 91 | `4a3a304630e8529e2531feb4ef16398c8a055cd028324a5ff5ad80d55a8ca4d4` |
| `emb/potion-requests.npy` | 17,738,368 | `66f23fb02d59a3fcbebea3c1aa199d20730956d3c1195e2910a29ff961f80763` |
| `emb/potion-requests.meta.json` | 755 | `4c8000f1fd24ffb585633d78df61369bece34cecca0611d1add852c08f361b70` |
| `stage/plan5.jsonl` | 92,289,168 | `c3fe7dc7166bb0dd063313d6036f9167a7a13cbaf1036cd2d37bbc8f04fde9d7` |
| `stage/pools5.json` | 1,317,247 | `6d459aa06283017bf1552b8e39a9d40bd8b6d2eb137a210009db2ab2918787fc` |
| `counts/v5-plan.json` | 18,348 | `c887b15f2b0d273013b2fe9d6d7ec8db961b4a829d99d03b8bd57d0f8cae4f20` |
| `counts/v5-slates.json` | 21,555 | `48d3afd81e028fc3235caa1d04b2fe86bcfc24f0732d9223c3021ca320bd41d0` |

- **The records:** 197,958. Training holds 182,085 (intents 91,130, tools 77,070, When2Call
  13,885) and validation 15,873 (11,085, 3,941 and 847).
- **The validation slates of the fits and the pilot:** `v5_tools` and its copies, 1,144 slates
  each, 915 of them tool-right.
- **The marker.** v5's checks did not write `checks.passed`, and v6 does not write it. v6's
  scripts read `runs/s6-route6/inputs.ok` instead (§14).

### 4.2 The teacher

- **The false-negative answers of v5 are used as they are** (§4.1), except for the slates of §4.8.
- **The fix's job:** v5's `false-negatives` job on the 1,043 rebuilt hard no-call slates (§4.8),
  v2's prompt word for word. That is at most 1,043 items, about **53 calls** of 20, a few minutes
  at v5's rate (2,982 calls in 4,527.5 s). An answer is reused by id only where a rebuilt slate
  matches a slate of v2 or v5, by coincidence. The teacher sees training and validation requests
  only.
- **The `fresh-tools` job of v5 §4.2:** about 100 to 150 calls, for T5's probe copies only, run
  after `fresh`, never on a request.

### 4.3 The request-blind reading (B2)

**How D4 renders a slate today** (`sky_infer::route::Render::ScoresInOptions.model_input`,
`crates/sky-infer/src/route.rs` lines 113 to 124; the records are written the same way,
`crates/sky-data/src/route/v5.rs` line 1919):

- the instruction "Which tool should handle the user's request?";
- each tool option as `score <s> rank <r>; <name>: <description>`, where s is the option's BM25
  score against the request over the slate's own tool options (two decimals) and r its rank among
  them (1 for the highest, ties to the earlier option);
- *none* last, as "none of these tools fits the request", with no score;
- the state `User: <request>`.

The embedding part reads `name: description` and the request (`sky-route fuse`, which computes
the scores itself from the slate file's request).

**The placeholder P = `?`** (a single question mark).

- **It yields no BM25 token.** BM25's tokens are runs of at least two word characters, stopwords
  removed (`tokens`, line 175). So for every slate every score is exactly 0. The ranks are then
  the ties' order: 1, 2, …, k in the slate's order (`ranks`, line 218).
- **The embedding part can read it.** `?` is a token of potion-base-2M's vocabulary, so its
  vector is defined and not the zero vector that an empty string would give.
- **It must occur as no request.** Before training, the stage `placeholder` (§14) checks that no
  training or validation request, after trimming, equals P; the counts amendment checks the same on
  T5. If P occurs, the next of `…` (U+2026) and `-` that occurs nowhere is taken; both also yield
  no BM25 token and are tokens of potion-base-2M. The string taken is written in
  `runs/s6-route6/placeholder.json` before training and never changes afterwards.
- **P must enter no training or validation slate.** No blind slate is ever trained on (§12).

**The blind readings.**

- **B0, the zero-match reading (gated).** The same slate with the request replaced by P and
  rendered by the model file's own rendering. Under D4:
  - each tool option is `score 0.00 rank <i>; <name>: <description>`, i being its position;
  - *none* is unchanged;
  - the state is `User: ?`.

  Under v2's line rendering (the published model part, §9.1), the state is `User: ?` followed by
  `BM25: <name> score 0.00 rank <i>; …`. The pair's embedding part reads P in place of the request.
  This is exactly what each system receives for a real request that shares no token with any
  option. In both renderings, then, B0 is the pipeline's own output on a request with no lexical
  match; only the request text is new.
- **B1, the bare reading (reported).** The options as `name: description` with no score and no
  rank, and the state `User: ?` (under v2's rendering, no BM25 line).

**Why B0 is gated and B1 reported.**

- **B0 is in the model's distribution in its format.** A score of 0.00 is what most distractors
  carry in training, and ranks 1 to k are always present. Since the gold's position is shuffled
  (R7 within its tolerance in every judged cell of v5), rank-by-position gives no information about the gold. A model
  that prefers rank 1 when all scores tie picks position 1, which is right at chance.
- **B1 removes the prefix that every training tool option carries.** In training the only option
  without a prefix is *none*. Under B1 every tool option takes *none*'s format, which no training
  slate shows, so a model may behave in ways that say little about cues in the option texts. It is
  reported for completeness.
- **The alternative of one shared rank** (`score 0.00 rank 1` on every option) is not taken.
  Ranks in training are always distinct, so it is out of distribution as well, and it adds nothing
  that B0's shuffled positions do not already give.
- **The support of B0, reported before training.** For each group and kind, the share of training
  and validation tool-right records in which every tool option scores 0.00. It reads the options'
  prefixes only (`scripts/s6-route6-blind.py support`). It says how often the model met an
  all-zero slate.

**The statistics** (`scripts/s6-route6-blind.py`).

- **The blind pick:** the tool option of highest probability among the k tool options (*none*
  left out), with ties counted 1/|tie|.
- **The blind accuracy:** the share of tool-right slates whose blind pick is the right tool.
- **Its chance:** c = the mean of 1/k over the tool-right slates. A system whose preference among
  the tool options carries no information about the gold has an expected blind accuracy of c,
  whatever it does with *none*.
  - The answer with *none* allowed is weaker: a system that answers *none* on most blind slates
    would pass while it still prefers the right tool among the others. It is reported, not gated.
- **The *none* answer in the blind reading:** the system's ordinary answer (argmax over every
  option, *none* included). For the pair, the ask rule at τ_ask6 is reported with it.
- **The blind *none* gap:** Δ_none = (the share of *none* answers among *none*-right slates) − (the
  share among tool-right slates), both in B0. A system that cannot tell from the options alone
  whether *none* is right has Δ_none near 0 (this is the model-side reading of N6).
- **Intervals:** 1,000 bootstrap resamples (seed 0, percentile), of the slates and, separately,
  of the servers on T5 (as v4 §7).

**Chance on `v5_tools`.** Counted from the file's option counts: every one of its 915 tool-right
slates has 7 tool options, in `v5_tools.jsonl`, `v5_tools_masked.jsonl` and their model copies
alike. So **c = 1/7 = 0.142857** and c + 0.05 = **0.1929**. T5's slates hold 7 tools by v4 §6.2,
item 5. Its c is counted on the built file and written in the counts amendment; if every slate
holds 7 tools, c = 0.142857 there too.

### 4.4 The data checks, reported (B3)

- **The records and v5's validation files:** v5's dated table (`runs/s6-route5/checks.md`,
  `checks.json`) is their description. Except for the rebuilt When2Call files, they are byte for
  byte the ones it read (§4.1).
- **The rebuilt records:** every check of v5 §4.4 runs again on all the records after the fix
  (§4.8). The results are reported, and only the stop rule of §4.8 can stop v6.
- **The other files v6 reads,** with v5's script (`scripts/s6-route5-checks.py`, C2 and N1 to N6,
  and C3), before training, written to `runs/s6-route6/checks-read-files.md`:
  - `v_tools` and `v_tools_masked`, and `v4_tools` and `v4_tools_masked` (v4's table already holds
    the last two);
  - the renamed and natural validation records' copies;
  - T2-tools and its masked copy, files already read; the study's §7 counts for them stand.
- **T5:** in the counts amendment, as v5 §6 says.
- **No other check stops v6,** whatever it shows. v4's and v5's dated tables are part of the
  record and are cited with every result of v6.

### 4.5 The baselines on the validation files

As v5 §4.5: BM25 and `potion-cos` on `v_tools`, `v4_tools` and `v5_tools` and their masked
copies, written in the amendment that sets the pilot's bar before the pilot. Added, reported: the
same baselines in B0 on `v5_tools` and its masked copy. BM25 in B0 ties every option, so its
blind accuracy is c exactly: a check of the chain.

### 4.6 Validation readings and leakage

- **Readings:** as v5 §4.6, with the blind copies of `v5_tools` and `v5_tools_masked` (B0 and
  B1, written by `sky-data route blind`, §14).
- **Leakage:** as v5 §4.6. The blind copies hold no request.

### 4.7 Disk

As v5 §4.7: **3 GB free** on the volume of the data folder before every writing stage
(`MIN_FREE_GB=3`); below it, a technical stop. About 7.3 GiB were free on 18/11/2026 at the
freeze.

### 4.8 The one data fix, and the one stop rule on the data (B0)

*Amendment 1 (§17) extends this fix to every slate of the When2Call group. The text below is the
frozen one; where the two differ, the amendment applies.*

**What v5's table shows** (`runs/s6-route5/checks.md`, and the When2Call cells of
`checks.json`).

- **N6 fails in two families only:**
  - `records when2call train other`: 0.4410 against a tolerance of 0.2345 (*none* share 0.2000;
    3,052 of 9,918 slates answered *none*), +0.2410;
  - `records when2call val other`: 0.4382 against 0.2800 (178 of 605), +0.2382.

  In the other 11 judged families both rules are within +0.025 of the *none* share, When2Call's
  renamed training family included (+0.0061).
- **What the two families hold:** the hard and natural slates of When2Call, and only those
  (4,947 + 4,971 = 9,918 in training, 305 + 300 = 605 in validation). When2Call has no removed
  cell. Their *none*-right slates are therefore the no-call requests' slates of both kinds: 980
  hard and 1,004 natural in training, and 63 and 58 in validation.
- **The table does not split N6 by kind,** so it does not confine the failure to the hard
  no-call slates. By kind it shows two things:
  - **Outside the cell.** In When2Call, every slate outside the cell (L6) is *none*-right, in
    both kinds alike: 329 of the 980 hard and 331 of the 1,004 natural no-call slates in
    training, and 20 of 63 and 13 of 58 in validation. No tool-right slate of the group is at L6
    (0 of 3,967 in each training kind). These are the slates that C6 counted at L6 (0.0782 and
    0.0902, under its bar). The table does not say why about a third of the no-call slates of
    both kinds could not be matched to their first own tool's cell.
  - **The exempt tools.** The tool-right hard slates hold no exempt tool: the restriction "the
    right option with the exempt options only" holds 0 slates. The tool-right natural slates hold
    exempt tools in 1,357 of 3,967 slates in training (84 of 242 in validation). The no-call
    slates of both kinds hold them, by v5's rule.
- **What follows.** The table supports the lead's account for the hard slates: the no-call hard
  slates hold exempt tools, and the tool-right hard slates hold none. It also shows that the
  natural no-call slates, half of the *none*-right slates of the two families, are as often
  outside the cell. The fix below is written for the hard slates only, as decided, and is expected
  to reduce N6's excess rather than certainly remove it. The stop rule decides whether what remains
  is small enough to leave to the model's gate.

**The fix: decision 3 of v5's amendment 2 is withdrawn for the hard slates.**

- **The rule.** A When2Call no-call hard slate holds no exempt tool. Its tools are the
  candidates nearest to the request (potion-base-2M cosine on v5's request embeddings) within the
  cell of the request's first own tool, at the slate's level. That is what v5's code did before
  amendment 2:
  - v5 §4.3's levels L0 to L6 and the minimum of 16 apply, counted on all the slate's tools,
    since every one is drawn;
  - the slate's size is v2's plan's (stream `route/plan/<id>`);
  - the request's own tools are not candidates;
  - the false-negative filter is v5's (§4.2, amendment 2, reading 3): a tool that the teacher
    names able is replaced by the next of the request's nearest tools in the cell, then from the
    cell's random fill, then from the next levels.
- **The slates rebuilt.** The 1,043 hard no-call slates that v5's records hold (980 training, 63
  validation; the *none*-gold slates of `records when2call hard train` and `val`). They are rebuilt
  by uid: the trimming to the 0.20 share is not redone, so the same no-call requests keep a hard
  slate and C3's shares do not move.
- **What is kept.** The natural slates keep the request's own tools in every case, tool-right and
  no-call alike. Every other record of v5 stands, byte for byte (§4.1).
- **The teacher:** the false-negative items of the 1,043 rebuilt slates, about 53 calls of 20
  (§4.2).

**The order, all before training** (`scripts/build-route6-data.sh`, §14):

1. the replanning of the 1,043 slates, with the plan's other entries unchanged;
2. their false-negative items, and the teacher;
3. the build into `<data>/s6-route/build6/`, and the hash check of every unchanged file (§4.1);
4. the checks of v5 §4.4 (C1 to C6 and N1 to N6, v5's script and tolerances) on the rebuilt
   records and on `v5_tools` and its masked copy, published as `runs/s6-route6/checks.md` and
   `checks.json`;
5. the spread table below;
6. the amendment that writes the new hashes, the counts by level of the rebuilt slates, the
   teacher's counts, the checks and the spread table.

**The stop rule** (pre-written; judged once, on the rebuilt records):

> **If, in any N6 family judged by v5's floor (at least 200 tool-gold slates; v5, amendment 2,
> decision 4), the accuracy of N6 or of its inverse exceeds the family's *none* share by more
> than 0.10, v6 stops before training, and the table is published. Otherwise the checks are
> reported, none of them stops v6, and the model's blind reading decides (G6).**

Why 0.10: a cue of that size on the *none* decision is not left to the model's gate. On v5's
table, 0.10 sits between the When2Call families (+0.24) and every other judged family (at most
+0.025).

**The spread table, before training** (reported). It is the share of When2Call slates whose
options spread over several L0 cells (spread above 0, the statistic of v5's amendment 2,
decision 4), given:

- by kind (natural, hard);
- by *none*-right or tool-right;
- by split;
- after the fix, with the same table on v5's records beside it.

## 5. The run (`scripts/train-route6.sh`)

As v5 §5: the smoke, one pilot of 20 working minutes with a checkpoint kept every 5, the main run
of 120, the calibration, the export, τ_ask6 and the fit of β and τ_ask6 on `v5_tools`, then the
evaluation.

- **Read at every kept checkpoint of the pilot and the main run,** besides v5's readings: the
  model part and the pair in B0 and B1 on `v5_tools` and `v5_tools_masked` (blind accuracy,
  answer with *none* allowed, *none* share, Δ_none).

**The pilot rule** (read at the pilot's last checkpoint):

1. **Abort** if the model part's accuracy on `v5_tools` is below `potion-cos` on `v5_tools` −
   **0.15** (v5's rule 1; the value is written in the amendment of §4.5).
2. **Abort** if the rewritten distractor is the answer on more than **0.30** of the tool-gold
   slates of `v5_tools_masked_rw` (v5's rule 2, which is v4's).
3. **Abort** if the model part's blind accuracy (B0) on the 915 tool-right slates of `v5_tools`
   is above **1/7 + 0.05 = 0.1929**.
4. **Abort** if the model part's blind accuracy (B0) on the 915 tool-right slates of
   `v5_tools_masked` is above **0.1929**.
5. Nothing else is chosen on the pilot.

At 915 slates and c = 1/7 the standard error of a blind accuracy is about 0.0116, so a model at
chance crosses 0.1929 with a probability below 0.0001.

## 6. The fresh test files (T5)

As v5 §6, with its amendments 1 and 2 (the removed twins' text, decision 1), never built so far:
the Toucan fetch of v4 hash-checked again against `<data>/MANIFEST4.json`, seed 1236, the
`t5_*` files, the 2,000 slates, the 600-slate floor. v5's stages `fresh`, the `fresh-tools`
teacher job, `probes` and `counts` run under v6 (§14), and the counts amendment is written before
any scoring.

**The counts amendment also holds:**

- T5's chance c (the mean of 1/k over its tool-right slates);
- the check that no T5 request equals P;
- for each tool-right slate, the number of the gold's own server's tools that it shows, counted
  by the builder.

## 7. Evaluation

As v5 §7 (v4 §7), on T5-tools, its masked copy and the two probe copies, and on T2-tools for G5,
plus:

- **The blind copies of T5-tools and of its masked copy** (B0 and B1), written after `FRESH-OPEN`
  by `sky-data route blind`.
- **Every system of v4 §7 read in B0** on both: the pair, its fast mode, the model part,
  `potion-cos-4bit`, `potion-cos`, BM25, `minilm-cos`, and the published s6-route pair (v3) for
  comparison.
- **B1 for the model part and the pair.**

## 8. Gate (every condition must hold)

**G1 to G5, the headline, the report's sentence and the announced goals: as v5 §8.** No bar
moves: G1's lower bounds above 0; G2 (a) ≤ 0.1929 and paired ≤ +0.03, (b) ≥ `potion-cos-4bit` −
0.02, (c) *none* on 0.10 to 0.30; G3 ECE ≤ 0.05; G4 max |Δp| ≤ 1e-3, ≥ 199 of 200, median ≤
60 ms, ≤ 4,000,000 bytes; G5 T2-tools ≥ 0.8668. G2's probe copies stay.

**G6, the model does not answer without the request** (on T5-tools, reading B0; c is T5's chance
from the counts amendment, 1/7 = 0.142857 if every slate holds 7 tools):

| | condition | with c = 1/7 |
|---|---|---|
| G6 (a) the model part | blind accuracy on the tool-right slates ≤ **c + 0.05**, and the upper bound of its 95% bootstrap with **servers resampled** ≤ **c + 0.08** | ≤ 0.1929; upper bound ≤ 0.2229 |
| G6 (b) the pair | the same, the embedding part reading P | ≤ 0.1929; upper bound ≤ 0.2229 |
| G6 (c) *none* without the request, for the model part and for the pair | Δ_none ≤ **0.05**, and the upper bound of its 95% bootstrap with servers resampled ≤ **0.15** | the same |

**G6 is judged on all tool-right slates of T5-tools.** The blind accuracy is also reported on the
slates split by whether a server-mate of the right tool is shown. If G6 fails only on the slates
that show server-mates, the report says so in its first sentence, and the verdict is still a
failure of the gate: T5's construction was frozen with the gate.

- **Why these bars (a and b).** They are the lead's. About 1,600 tool-right slates are expected
  (2,000 slates, 20% removed). There a blind accuracy has a slate standard error of about 0.009.
  A server bootstrap is wider: v3's cluster bootstrap on T2-tools was about three times its slate
  interval. So 0.05 is about five slate standard errors, and the upper bound leaves 0.03 for the
  server spread. At chance, a server interval three times the slate one puts the upper bound
  near c + 0.053, under c + 0.08; a system at c + 0.03 would then fail it.
- **Why the *none* bars (c) and not a band on the *none* share.**
  - **The share.** A model that sees no request should say *none* or be unsure. But no training
    slate lacks a request, so no value of the blind *none* share follows from the design. In
    training an all-zero slate is more often a removed one (the gold, which matched the request,
    is gone), so a model that reads the scores may answer *none* often in B0 for a reason that
    is no shortcut. The share is therefore an announced goal, not a gate: the pair answers *none*
    or asks on at least **0.80** of the blind slates.
  - **The gap.** What can be justified is that, without the request, the *none* answer should
    not depend on whether *none* is right. That is the gap Δ_none.
  - **Its bars.** With about 400 *none*-right and 1,600 tool-right slates, the slate standard
    error of Δ_none is 0.017 to 0.028, depending on the share. A server bootstrap two times wider
    gives a half-width of about 0.11. So an upper bound of 0.12 or less could fail a system whose
    gap is 0, and 0.15 is taken (§15, O37).
- **The announced goals, not gated:** the pair's mean top probability is lower in B0 than in the
  ordinary reading; G6 (a) and (b) hold on `t5_tools_masked` too.

## 9. Reported, not gated

As v5 §9, plus:

- B0 and B1 for every system of §7, on T5-tools and its masked copy, with both intervals;
- the answer with *none* allowed, the *none* share, Δ_none and the share asked;
- the blind accuracy on the slates that show a server-mate of the right tool and on those that do
  not, with the finer split by their number (0, 1 to 2, 3 or more) (§8, §11);
- the support of B0 (§4.3);
- every pilot and main-run checkpoint in B0 and B1;
- the data checks of §4.4, §4.8 and of the counts amendment, and the spread table of §4.8;
- the diagnostic of §9.1.

### 9.1 The diagnostic on the published files (B6; reported, files already read)

**What it is.** A reading of files that the project has already read: no new test file, no fresh
file, no number or verdict of v2 or v3 changes. **It runs after this freeze and before v6's
training.** It changes nothing in v6: no bar, rule or choice of v6 depends on it.

**What it reads.**

- **Systems:**
  - the published model part (`site/public/models/s6-route-nano.gguf`, v2's rendering: the BM25
    line in the state);
  - the published pair (that file with `site/public/models/s6-route-embed.gguf`, β = 0.81,
    τ_ask = 0.671047);
  - the embedding part alone (`potion-cos-4bit`, θ = 0.396120, T = 0.0671021).
- **Files:**
  - `v_tools` and `v_tools_masked` (`<data>/s6-route/build/val/`), read by v2's pilots, v3's fits
    and the diagnosis;
  - T2-tools and its masked copy (`<data>/s6-route/build/fresh/f_tools.jsonl`,
    `f_tools_masked.jsonl`), read for v3's result.
- **Readings:** B0 by each file's own rendering (§4.3; for v2's file the state `User: ?` and a
  BM25 line of zeros), and B1.

**Reported for each system and file:**

- the blind accuracy with c and its 95% slate bootstrap, plus, on T2-tools, the cluster bootstrap
  as v3;
- the answer with *none* allowed, the *none* share and Δ_none;
- the share of blind picks that are the slate's fewest-words option, the cue of
  `docs/s6-route4-study.md` §7 (right on 0.3633 of T2-tools' tool-right slates).

**Where it goes, whatever it shows:**

- the numbers, to `runs/s6-route6/diagnostic.json` and `.md`;
- a dated note at the end of `docs/s6-nano3-prereg.md`;
- one line in the limits of the cards of `s6-route` and `s6-route-nano`;
- `BENCH.md` and Results.

**What it can say.** Whether the published model part and pair answer from the options alone on
files whose slates carry a known request-blind cue. It is a reading of the published files, not a
test of them.

## 10. Disclosure (written into every place that reports v6)

As v5 §10, plus:

- **v6 relaxes what stopped v4 and v5.** It was designed after v4's and v5's data checks were
  read, and the data checks that stopped them do not stop v6. They are replaced by a test on the
  model (G6), which could not be run before a model existed.
- **The one data fix follows v5's table.** It was chosen after N6's failure was read, and it
  withdraws the lead's own decision 3 of v5's amendment 2 for the hard slates (§4.8).
  - The table does not show that the hard no-call slates carry the whole cue. The natural no-call
    slates are half of the *none*-right slates of the two failing families and are as often
    outside the cell, and they are not changed.
  - The stop rule of §4.8 was written before the rebuild.
- **A reader who holds that the data should have been blind may hold v6 to that view.** v4's and
  v5's tables are published, and v6's training data are the files v5's table describes, except the
  1,043 rebuilt slates, whose checks are published (§4.8).
  That table holds R12 inverse in the intents, N2, N3 and N4 in the hard cells of the tools and
  When2Call groups, and N6 on When2Call's *none* decision.
- **The blind reading is fixed before any model of v6 is trained:** its definition, the
  placeholder and its fallbacks, the statistic and the bars.
- **v5's data were built for a design that judged them,** and are trained on in a design that
  reports them.
- **The diagnostic of §9.1 runs after the freeze;** nothing of v6 was chosen on it.

## 11. What a failure would mean

As v5 §11 (v4 §11) for G1 to G5, with these points added. In each, "by how much" is the blind
accuracy's excess over c (or Δ_none), with its two intervals.

- **The stop rule of §4.8 stops v6.** After the fix, a *none* cue of more than +0.10 above the
  *none* share remains in at least one judged family. It lies, at least in part, where the fix did
  not reach (the natural no-call slates, or the no-call slates outside the cell), or the rebuild
  moved it. No model is trained, and the table is published with the spread table. A fix of the
  natural no-call slates or of the no-call slates' cell needs its own pre-registration.
- **N6 is still above its tolerance but within +0.10.** v6 trains. G6 (c) reads whether the
  model uses the remaining cue, and the report names it.
- **The pilot aborts on rule 3 or 4.** After 20 minutes the model part already answers from the
  options alone on validation slates. No main run. Training on these data teaches a request-blind
  preference, and the cue is not named by any rule unless the tables of v5 name it. The
  diagnostic of §9.1 says whether the published model part does the same.
- **G6 (a) fails.** The model part learned to answer from the options alone, on tools of a family
  it never saw, by the excess stated. Not published as a replacement.
  - **The published pair is suspect in the same way.** Its model part was trained on v2's hard
    slates, which carried request-blind cues of the same kind (v4's result block). The diagnostic
    of §9.1 measures it on the files already read, and its line enters the cards whatever it shows.
- **G6 (b) fails, (a) holds.** The embedding part's own blind preference, or the fusion, gives the
  system a request-blind preference that the model part alone does not have. The embedding part's
  blind accuracy is reported, so the two cases are told apart. Not published as a replacement.
- **G6 (c) fails.** Without the request the system can tell whether *none* is right: N6's cue read
  in a model. Not published as a replacement.
- **G6 fails only on slates that show the gold's server-mates** (the split of §8). The cue may
  be T5's construction: the gold is always among the tools of the one server that the slate shows
  several of, and its fill comes from other servers. The report says so in its first sentence. G6
  is still failed, since the construction was frozen with the gate.
- **G6 holds while v5's tables show cues.** The data carry the cues that the tables name, and the
  trained system does not answer from the options alone by more than 0.05 above chance on T5. The
  claim is limited to that sentence.
  - It does not say that the model ignores those cues when it also reads the request: a cue
    used jointly with the request is not measured by B0.
- **B0 and B1 disagree** (reported): the system's blind preference depends on the scores' format.
  Stated with the gate.

## 12. Forbidden

As v5 §12 and v4 §12, with v5's "no remedy" and "the cell's features" points moot (the cell is
not changed), plus:

- rebuilding, filtering, reweighting or editing any record other than the 1,043 slates of §4.8,
  or training on a file whose hash differs from §4.1 or from the amendment of §4.8;
- changing the fix, extending it to other slates, or changing the stop rule's 0.10 after any step
  of §4.8 has run;
- any teacher call other than the fix's false-negative job and the `fresh-tools` job;
- training on any slate whose request is P, or on any blind copy;
- changing P, its fallbacks, B0, B1, the blind statistic, the pilot rules 3 and 4 or a bar of G6
  after any model of v6 is trained;
- choosing a checkpoint, β, τ, the head window or anything else on a blind reading;
- moving a data check back to a stopping rule, or citing a data check as grounds to stop, change
  or rerun v6;
- reading the blind copies of T5 before `FRESH-OPEN`;
- changing anything of v6 on the diagnostic of §9.1;
- running a stage below the disk floor.

## 13. Cost

| item | amount |
|---|---|
| teacher | the fix's false-negative job, ≤ 1,043 items, ≈ 53 calls (a few minutes); the `fresh-tools` job, ≈ 100 to 150 calls (v5 §4.2) |
| CPU | hash check of §4.1 ≈ 2 min; the fix's replanning ≈ 5 min, its build ≈ 5 min, its checks and the spread table ≈ 5 min (v5's build and checks took under 2 min together); the placeholder check and the support count ≈ 3 min; the checks of §4.4 on the read files ≈ 5 min; the baselines on the validation files, ordinary and blind ≈ 8 min; Toucan's hash check, read, embedding and clustering ≈ 20 min; the blind copies ≈ 1 min; the pilot readings, ordinary and blind ≈ 25 min; the baselines on T5 and T2, ordinary and blind ≈ 30 min; the diagnostic of §9.1 ≈ 10 min |
| GPU | smoke 3 + pilot 20 + main 120 + calibration and export 5 + evaluation 15 + blind readings ≈ 5 ≈ **168 working minutes**; ≈ 25 if the pilot rule aborts |
| disk | the full copy `build6/` with the rebuilt records ≈ 0.4 GB; T5 and its copies ≈ 0.02 GB; the blind copies ≈ 0.02 GB; pilot checkpoints 5 × 64 MB and main 12 × 64 MB ≈ 1.1 GB; evaluation and dumps ≈ 0.15 GB: **≈ 1.7 GB new**; floor **3 GB free** before every writing stage, against about 7.3 GiB free on 18/11/2026 at the freeze |

## 14. Code to write before any run (by file)

**Reused as is:** every item that v5 §14 and its amendment 2 list as written, the stages
`baselines-val`, `fresh`, `probes` and `counts` of `scripts/build-route5-data.sh` (run through
v6's script), `scripts/route-teacher.sh`, `sky-eval --dump-probs` and `sky-route fuse` (a blind
plain file gives B0 without change: `sky-route fuse` computes the scores from the file's request).

**To write:**

0. **The fix (§4.8):**
   - `crates/sky-data/src/route/v5.rs`: a plan switch for the no-call hard slates
     (`sky-data route v5-plan --no-call-hard drawn`), in which the slate holds no exempt tool and
     its tools are the request's nearest within the first own tool's cell. The default (`exempt`)
     keeps v5's bytes. Its tests:
     - such a slate holds no own tool;
     - its drawn tools follow the request's nearest within the cell;
     - every other record is byte-identical to the default's on the fixture, and v2's, v3's,
       v4's and v5's frozen bytes are unchanged.
   - `sky-data route v5-slates --only-uids <file>`: the rebuild of the listed slates only, the
     others copied.
   - The false-negative items of the rebuilt slates alone (`teacher-jobs --only no-call-hard`).
   - `scripts/s6-route6-checks.py` and `scripts/test_s6_route6_checks.py`: v5's checks imported
     as they are, the stop rule of §4.8 (exit 2 above +0.10 in a judged family; tests at 0.10 and
     just above) and the spread table.
1. **`crates/sky-infer/src/route.rs`:**
   - the constant `BLIND_REQUEST` (P) and its fallbacks;
   - `Render::blind_input(options)`: `model_input(P, options)`, asserting that every BM25 score
     is 0;
   - a rendering `bare` for B1, accepted on the command line and refused in a model file;
   - the fixture `tests/fixtures/route_blind.json`;
   - tests: every tool option is `score 0.00 rank <i>; ` in slate order and *none* is unchanged,
     under both renderings; P and its fallbacks have no BM25 token.
2. **`crates/sky-data/src/route/blind.rs` (new), `sky-data route blind`:**
   - `--in <slate file> --out <stem>` writes:
     - `<stem>.blind.jsonl`, the slate with its state set to `User: <P>` and its options plain
       (`plain_option`), with ids, targets, kinds and servers kept;
     - `<stem>.blind.model.jsonl` (B0, by D4);
     - `<stem>.blind-bare.model.jsonl` (B1).
   - It refuses a fresh file before `FRESH-OPEN`.
   - `--check-placeholder <records...>` exits 2 when a request equals P after trimming, and
     writes `placeholder.json`.
   - Tests on an invented file.
3. **The engine's evaluation path:**
   - `sky-route fuse --render bare` (B1);
   - a test that `sky-route fuse` on a blind file gives the embedding part P and the model part
     B0's text;
   - the t5 split of `scripts/route-eval.sh` and `scripts/route-baselines.py` (refused until
     `runs/s6-route6/FRESH-OPEN`) gains a `blind` step: the copies, the dumps and the fused dumps
     in B0 and B1.
4. **`scripts/s6-route6-blind.py` and `scripts/test_s6_route6_blind.py`:**
   - the blind pick, accuracy, c, the answer with *none* allowed, the *none* share, Δ_none, both
     bootstraps and the server-mates breakdown;
   - `support` (§4.3);
   - tests: a uniform system scores c exactly; a system that prefers the gold among tools while
     answering *none* everywhere fails the restricted reading; Δ_none on an invented split.
5. **`scripts/s6-route6-pilot.py`:** v5's reader with the blind readings and the rules 3 and 4,
   with tests at 0.1929.
6. **`scripts/build-route6-data.sh`:** the stages, each refused without the marker of the one
   before:
   - `inputs` (§4.1's hashes against `runs/s6-route6/inputs.json`);
   - `fix-plan`, `fix-teacher-jobs` (then `scripts/route-teacher.sh false-negatives loop` on
     `build6/`), `fix-build` (with the hash check of every unchanged file), `fix-checks` (the stop
     rule; exit 2 stops v6) and `spread`; then `inputs.ok` is written, holding the hashes of the
     amendment of §4.8;
   - `placeholder`, `support` and `checks-read` (§4.4);
   - `baselines-val`, with the blind baselines;
   - `blind-val` (the blind copies of `v5_tools` and `v5_tools_masked`);
   - `fresh`, `fresh-tools` (the teacher job), `probes` and `counts`, which call v5's stages on
     `build6/` with `inputs.ok` in place of `checks.passed`.

   `MIN_FREE_GB=3` is checked before every stage, and `DRY_RUN=1` is supported.
7. **`scripts/train-route6.sh`:** v5's script with v6's names. It refuses to start without
   `inputs.ok` and `placeholder.json` (in place of `checks.passed`), and its pilot step calls
   `s6-route6-pilot.py`.
8. **`scripts/s6-route6-fit.py`** (β and τ_ask6 on `v5_tools`) and the embedding file that stores
   them; **`scripts/s6-route6-gate.py`** and **`scripts/test_s6_route6_gate.py`** (G1 to G6, the
   headline, the report's sentence); the Python reference of the pair under D4 for G4.
9. **`scripts/route6-diagnostic.sh`:** §9.1 (blind copies of the published files' slates by
   `sky-data route blind`, read through `sky-eval --dump-probs` and `sky-route fuse` with the
   published files, and the statistics of `scripts/s6-route6-blind.py`), writing
   `runs/s6-route6/diagnostic.json` and `.md`. It runs after the freeze and before training.
10. **`DATA_LICENSES.md`:** no new source; the false-negative line names v6's job on the rebuilt
    slates, and the line of T5 says it is read by v6.

## 15. Choices (the value taken at the freeze; the alternative is recorded)

| | choice | taken | alternatives | why |
|---|---|---|---|---|
| O32 | the training data | v5's build as it stands, by hash, but for the slates of O42 (the lead's design) | a rebuild with a coarser cell; v4's build | the data checks no longer decide; a rebuild would be a new data design with its own checks |
| O42 | the one data fix | the When2Call no-call hard slates hold no exempt tool; 1,043 slates rebuilt by uid with their false-negative items (the lead's decision 1) | the natural no-call slates too; no fix | a cue of +0.24 is of the size that started this line; the table supports the hard slates' account and does not split by kind, so the natural slates are left to the stop rule |
| O43 | the stop rule on the data | N6 or its inverse more than 0.10 above the *none* share in a judged family after the fix stops v6 (the lead's decision 1) | 0.05; N6's tolerance | 0.10 lies between When2Call's +0.24 and every other family's +0.025 at most |
| O33 | the placeholder | `?`, then `…`, then `-` if one occurs as a request (the lead's decision 2) | the empty string (the embedding part's vector would be zero, its cosine undefined); a phrase of stopwords (reads as a request with content); a word such as "request" (a BM25 token, which would score some options) | no BM25 token, a defined embedding, and the shortest text that reads as no request |
| O34 | the gated rendering | B0: scores 0.00 and ranks by position, as D4 writes them for a request that matches nothing; B1 reported (the lead's decision 2) | B1, the prefix removed (every tool option in *none*'s format); one shared rank (never seen in training) | B0 is the pipeline's own output on a request with no lexical match |
| O35 | the gated statistic | the pick among the tool options (*none* left out) (the lead's decision 2) | the answer with *none* allowed | its chance is c whatever the system does with *none*; the other can hide a preference behind *none* answers |
| O36 | G6 (a), (b) | c + 0.05, upper bound (servers) ≤ c + 0.08 (the lead's design) | c + 0.03 and c + 0.06; slates resampled | about five slate standard errors at 1,600 slates; servers as the headline |
| O37 | G6 (c) | Δ_none ≤ 0.05, upper bound (servers) ≤ 0.15; the *none* share an announced goal (≥ 0.80 *none* or asked) (the lead's decision 3) | upper bound ≤ 0.08 or 0.12; a gated band on the *none* share | no share follows from training; the gap is what a request-blind *none* cue moves; 0.15 leaves room for a server interval two times the slate one |
| O38 | where G6 is read | T5-tools, all tool-right slates; the masked copy an announced goal | both gated | one gated file, as G1 |
| O44 | the server-mates of T5 | judged on all tool-right slates; reported split by whether a server-mate is shown; a failure carried only by those slates said in the report's first sentence, still a failure (the lead's decision 5) | read as a failure of T5's construction | the construction was frozen with the gate |
| O39 | the pilot's blind aborts | > 0.1929 on `v5_tools` or on `v5_tools_masked` (the lead's design) | `v5_tools` alone; > 0.1729 | 0.05 is about four standard errors at 915 slates |
| O40 | the data checks | run on every file read, published, never stopping (the lead's design) | judged as in v5 | §10 |
| O41 | the diagnostic | reported, on `v_tools` and T2-tools, which have been read; after the freeze and before training; into a dated note of `docs/s6-nano3-prereg.md` and both cards (the lead's decision 4) | before the freeze; not run; run on `v5_tools` too | it reads no new test and nothing of v6 depends on it; `v5_tools` is v6's fitting file, left to v6 |

## 16. The lead's decisions on the draft (18/11/2026)

1. **One data fix** (§4.8, O42, O43). In When2Call's no-call hard slates the request's own tools
   are no longer exempt: decision 3 of v5's amendment 2 is withdrawn for the hard slates, and the
   1,043 such slates are rebuilt with their false-negative items. The checks run again and are
   reported, and one stop rule applies: N6 more than 0.10 above the *none* share in a judged
   family stops v6 before training. Disclosed as following v5's table (§10).
2. **The blind reading** (§4.3, O33 to O35). B0 is gated and B1 reported; the placeholder is `?`,
   then `…`, then `-`; the gated statistic is the pick among the tool options.
3. **G6 as drafted** (§8, O36, O37). (a) and (b) at most c + 0.05, with the upper bound at most
   c + 0.08; (c) the blind *none* gap at most 0.05, with the upper bound at most 0.15. The blind
   *none* share is an announced goal.
4. **The diagnostic of the published files** (§9.1, O41) runs after this freeze and before v6's
   training, as a reported reading of files already read. No bar of v6 depends on it, and its
   result goes into a dated note of `docs/s6-nano3-prereg.md` and into both cards' limits,
   whatever it shows.
5. **T5's server-mates** (§8, §11, O44). G6 is judged on all tool-right slates and reported split
   by whether a server-mate is shown. A failure carried only by those slates is said in the
   report's first sentence, and it is still a failure of the gate.
6. **The spread table** of When2Call's slates, by kind and by *none*-right or tool-right, after
   the fix, is reported before training (§4.8).
7. **The costs, the code, the meaning of a failure and the choices** follow these decisions
   (§11 to §15).

## 17. Amendments

### Amendment 1 (18/11/2026, decided by the lead, before any computation of v6)

**Nothing of v6 has run.** No code of §14 is written, no record is rebuilt, no teacher call is
made, no blind copy is written, no model is trained and no system is scored. The published
diagnostic of §9.1 has not run either. This amendment is written from v5's published table alone
(`runs/s6-route5/checks.md` and `checks.json`).

**Why.** As frozen, the fix of §4.8 changes the hard no-call slates only. The table shows that
N6's failure sits in the natural no-call slates as much as in the hard ones (§4.8). A fix that
leaves half of it would be expected to trip its own stop rule. A stop that is known from the table
before any computation is corrected before any computation.

**1. The fix is extended to the whole When2Call group.** In the When2Call group no slate of any
kind shows an exempt tool.

- **A tool-right slate** holds its right tool and drawn distractors.
- **A no-call slate** holds drawn distractors only:
  - natural: the cell's random fill in the first own tool's cell, at the slate's level;
  - hard: the tools nearest to the request within that cell, as frozen in §4.8.
- **The request's own tools other than the right tool are not shown** in this group, in any kind
  (natural, hard and renamed, and the renamed kind's *none*-right copies). Every tool of a slate is
  a drawn tool, so the minimum of 16 candidates and the levels L0 to L6 of v5 §4.3 apply to all of
  them. The slate's size is kept, as is the stream that drew it.

**What the table shows that motivates each half.**

- **The no-call slates of both kinds.** In When2Call, the *none*-right slates of both kinds sit
  outside the cell (L6) about a third of the time, and no tool-right slate does:
  - training: 329 of the 980 hard and 331 of the 1,004 natural no-call slates;
  - validation: 20 of 63 and 13 of 58;
  - 0 of the 3,967 tool-right slates of each training kind.

  Changing the hard no-call slates alone would leave the natural half of the *none*-right slates of
  the two failing families as they are.
- **The tool-right slates.** Exempt tools are held by 1,357 of the 3,967 tool-right natural
  training slates (84 of 242 in validation), and by 1,282 of the 3,174 tool-right renamed training
  slates (77 of 194 in validation). If only the no-call slates lost their exempt tools, these slates
  would be the only ones of the group with tools outside the right tool's cell. "The options spread,
  so a tool is right" would then be the inverse cue.

**2. What is lost.**

- **When2Call's own near-miss tools are no longer shown as they come.** These are the tools that
  its authors put next to a request which none of them can answer, and the tools that sit next to
  the right tool in a real When2Call conversation.
- **Close distractors remain on *none*-right slates.** The hard no-call slates still put the tools
  nearest to the request in front of the model, filtered by the teacher. So *none*-right slates
  whose distractors are close to the request remain in training.
- **No reading of v6 uses When2Call's own tool lists.** v2's When2Call test file
  (`t_when2call`) is not read by v6's gate, nor by any reading of §5, §7, §9 or §9.1. Those
  readings are `v5_tools`, `v4_tools`, `v_tools`, the tools group's renamed and natural validation
  records, T5, T2-tools and their copies. The When2Call validation records are rebuilt by the rule
  above, so they are within the construction. Had any reading used When2Call's own tool lists, it
  would have been labelled an out-of-construction reading.

**3. What is rebuilt.**

- **Every record of the When2Call group, by uid:** 14,732 slates.

  | split | hard | natural | renamed | total |
  |---|---:|---:|---:|---:|
  | training | 4,947 | 4,971 | 3,967 | 13,885 |
  | validation | 305 | 300 | 242 | 847 |

  The *none* share is kept at 0.20 as built: the same slates are rebuilt, so C3's counts do not
  move (1,984 *none*-right of 9,918 and 793 of 3,967 in training; 121 of 605 and 48 of 242 in
  validation). When2Call has no removed cell; its *none*-right slates are the no-call slates and
  the renamed kind's removed set.
- **The teacher's items, for the hard no-call slates only:** at most 1,043 items, about 53 calls,
  unchanged.
  - The tool-right hard slates already hold no exempt tool, so their tools and their v5 answers
    are unchanged, by id.
  - The natural and renamed slates have no false-negative filter.
- **Everything outside the When2Call group is byte-identical,** checked by hash (§4.1).
  - The files that change are those §4.1 already named: `records/when2call/train.jsonl`,
    `records/when2call/val.jsonl` and `counts/v5-slates.json`, with the new teacher files.
  - The other groups keep their exempt tools exactly as v5 built them. In the tools group, 5,830
    of 22,020 natural and 5,431 of 17,616 renamed training slates hold exempt tools; the intents
    group has none.
  - The table shows no N6 failure outside When2Call. The largest N6 excess there is +0.0249
    (N6's inverse on `v5_tools_masked`). Among the record families the largest is +0.0197
    (`records tools val renamed`).

**4. The stop rule and the spread table.**

- **The stop rule of §4.8 stands as frozen.** If, in any N6 family judged by v5's floor, the
  accuracy of N6 or of its inverse exceeds the family's *none* share by more than 0.10 after the
  rebuild, v6 stops before training, and the table is published.
- **Before training, the report gives the spread table** for the When2Call group: the share of
  slates whose options spread over several L0 cells, by kind (natural, hard, renamed), by
  *none*-right or tool-right and by split, after the rebuild, with v5's beside it. It also gives
  the counts by level of the rebuilt slates, with the L6 share by kind and by *none*-right or
  tool-right.

**5. The code and the costs** (§13, §14), as they now stand.

- **Code, item 0 of §14, replaced:**
  - `crates/sky-data/src/route/v5.rs`: a group switch (`sky-data route v5-plan --when2call-exempt
    none`). In the When2Call group it shows no exempt tool in any kind and draws every tool of the
    slate in the cell: the natural no-call fill in the first own tool's cell, the hard no-call
    slates by the request's nearest within it, and the tool-right slates' fill in the right tool's
    cell. The default (`all`) keeps v5's bytes.
  - Its tests:
    - no When2Call slate of any kind holds an own tool other than the right tool;
    - a no-call slate holds drawn tools only;
    - the hard no-call slates follow the request's nearest within the cell;
    - the slates' sizes and the kept uids are unchanged;
    - every record outside the When2Call group is byte-identical to the default's on the fixture;
    - v2's, v3's, v4's and v5's frozen bytes are unchanged.
  - `sky-data route v5-slates --only-group when2call`: the rebuild of the group's records by uid,
    every other file copied.
  - `teacher-jobs --only no-call-hard`, unchanged.
  - `scripts/s6-route6-checks.py`: the stop rule unchanged, and the spread table and the levels by
    kind, by *none*-right or tool-right and by split, for the When2Call group.
  - The stages of `scripts/build-route6-data.sh` keep their names (`fix-plan`, `fix-teacher-jobs`,
    `fix-build`, `fix-checks`, `spread`).
- **Costs:**
  - teacher unchanged: about 53 calls for the fix and 100 to 150 for `fresh-tools`;
  - CPU: the fix's replanning and build, about 10 minutes (14,732 records instead of 1,043, still
    small next to v5's whole build, which with its checks took under 2 minutes);
  - GPU and disk unchanged: about 168 working minutes, and about 1.7 GB new, `build6/` included.

**Disclosure** (§10, added). The extension follows v5's table, as the fix did. It was decided
after the frozen fix was compared with the table's counts by level and by gold, and before
anything of v6 ran.

### Amendment 2 (18/11/2026, decided by the lead, before any computation of v6): readings of the code fixed before any computation

**Nothing of v6 has run.** The code of §14 is written (branch `route6-code`). No record has been
rebuilt, no teacher call made, no blind copy written, no model trained and no system scored. The
published diagnostic of §9.1 has not run either.

One count was made to write this amendment, read-only: point 1 below, on v5's plan and on v2's
tool lists. It opened no request text and wrote nothing in the data folder.

The code's readings of the frozen text and of amendment 1, as the lead decided them:

1. **A drawn tool that is one of the request's own tools.**
   - **The count.** 5 of the 4,306 tool-right hard slates of the When2Call group show one of the
     request's own tools (other than the right tool) as a drawn distractor, within the right
     tool's cell. It was counted read-only on v5's plan (`build5/stage/plan5.jsonl`, the tools of
     each hard slate) against v2's tool lists (`build/stage/requests.jsonl`, each request's own
     tools).
   - **The decision.** These 5 slates keep v5's candidates, tools and false-negative answers, as
     the code does. Amendment 1's purpose is that no tool is shown *because* it is the request's
     own: no exempt tool. A tool that the draw itself selects within the cell is a drawn tool like
     any other.
   - **The literal reading,** "the request's own tools … are not shown in this group, in any
     kind", would have changed those 5 slates. Their items would then have gone to the teacher
     again (about 5 more than the 1,043), and a request dropped by the teacher would have moved
     the uids that the rebuild keeps.
   - **What does change.** In the natural, renamed and no-call hard slates of the group, the
     request's own tools other than the right tool are neither exempt nor candidates.
2. **The engine is not changed.** `crates/sky-infer` is not modified; item 1 of §14 is replaced.
   - **B0** is read through the existing paths:
     - the blind copy rendered by the model file's own rendering (`sky-data route blind`), read by
       `sky-eval --dump-probs`;
     - the blind plain file read by `sky-route fuse`, which computes every score from the
       placeholder (all 0.00) and gives the embedding part the placeholder.
   - **B1** is read for the model part only, by `sky-eval` on the bare file
     (`<stem>.blind-bare.model.jsonl`).
   - **The pair's B1 is not read in v6.** It was reported only. It would need a rendering `bare`
     in `sky-route fuse`, an engine change that the gate does not need, so it is dropped. §7's
     "B1 for the model part and the pair" and §9's B1 readings are reduced to the model part.
   - The pilot, the diagnostic of §9.1 and G6 read the pair in B0, as frozen.
3. **The placeholder** is `?`, then `…`, then `-` (§4.3).
   - The check (`sky-data route blind --check-placeholder`) takes the first of the three that
     equals no training or validation request after trimming. It stops only if all three occur.
   - The one taken is written once, in `runs/s6-route6/placeholder.json`, and kept for every file,
     T5 included.
   - If the diagnostic of §9.1 runs first, it takes the placeholder on v5's build, whose requests
     are v6's. The stage `placeholder` then checks it against v6's records and does not choose
     again. The counts amendment checks it against T5's requests (`--keep`).
4. **The stages of `scripts/build-route6-data.sh`.**
   - **The order:** `blind-val` runs before `baselines-val`, since the blind baselines read the
     blind copies.
   - **`inputs-ok`.** It checks `build6/` against the hashes that the amendment of §4.8 lists
     (`runs/s6-route6/inputs6.json`), before training. Only then is `runs/s6-route6/inputs.ok`
     written.
   - **The fix's teacher files** live under `build6/fix/teacher/` (`false-negatives.items.jsonl`,
     `false-negatives.jsonl`, `false-negatives.refused.jsonl`), with the fix's plan under
     `build6/fix/stage/`. The teacher files copied from v5 under `build6/teacher/` keep their
     hashes of §4.1. The lead runs the job with
     `BUILD=<data>/s6-route/build6/fix scripts/route-teacher.sh false-negatives loop`.
5. **G4's input** is one JSON from the engine's bench, with these fields:
   - `max_abs_diff`: the largest |Δp| against the Python reference;
   - `argmax_agree` and `n`: the slates with the same answer, out of the slates compared;
   - `median_ms`: the median per request, WASM SIMD128;
   - `bytes`: the two files as shipped.

   The bars are those of §8: ≤ 1e-3, ≥ 199 of 200, ≤ 60 ms, ≤ 4,000,000 bytes.
6. **What remains to write before the gate,** so that §14 is complete. Written so far, items 0,
   2, 4 to 7, 9 and 10, as this amendment reads them; items 3 and 8 in part. Still to write:
   - the embedding file that stores β and τ_ask6, after `scripts/s6-route6-fit.py`;
   - the t5 split of `scripts/route-eval.sh` and `scripts/route-baselines.py`, refused until
     `runs/s6-route6/FRESH-OPEN` exists and the files have the hashes of
     `runs/s6-route6/test-files.json`, with its `blind` step: the blind copies of T5-tools and of
     its masked copy, the dumps in B0 and B1 (B1: the model part only), and the fused dumps in B0;
   - the Python reference of the pair under D4, for G4's parity;
   - the bench that writes G4's JSON in the form of point 5.

**Nothing else changes.** No bar, rule, file or choice of the frozen text or of amendment 1 moves.
