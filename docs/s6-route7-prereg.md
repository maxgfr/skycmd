# s6-route v7 (v6 with the no-call slates without an anchor anchored on the tool nearest to the request): pre-registration

**Frozen on 18/11/2026 by the lead, before any computation of v7; every later change is a dated amendment that precedes the computation it concerns.**

**Amendment 1 of the same day (18/11/2026), after the rebuild and its checks, before any training (§16):** the stop rule does not fire (N6's excess +0.0025 in training, +0.0667 in validation); the records' hashes, the teacher's counts and the spread table.

**Amendment 2 of the same day (18/11/2026), from the stages' outputs, before any training (§16):** the pilot's bar (0.578147), the validation baselines, the counts of T5 (2,000 slates, above the floor of 600) and its request-blind rules, reported.

**Amendment 3 (19/11/2026), while the training runs, before the fresh test is opened (§16):** readings of the gate's code (the names of τ_ask7, the stage that opens T5, G4's median, B1, the extra baselines, the blind copies and G4's slates, the one look at T5's records); written before any number of v7 is known.

- **Nothing of v7 has run.** No record is rebuilt, no plan is drawn, no teacher call is made, no
  blind copy is written, no model is trained and no system is scored. To write this document:
  - no record and no request text was opened;
  - read: v6's result block (`docs/s6-route6-prereg.md`, "Result (18/11/2026)") and its files
    (`runs/s6-route6/checks.md`, `checks.json`, `spread.md`, `spread.json`, `rebuild.json`,
    `no-call-anchors.json`);
  - counted, read-only, for the costs of §13: the *none*-right hard and natural slates of
    `build6/records/when2call/` at L6, by split (the field `cell_level` of each record; no text);
  - hashed: v6's build as v6 left it (§4.1), copied from `runs/s6-route6/rebuild.json` and from
    the table of v6's result block.
- **The code of §14 is written** (branch `route7`: the switch, its tests and the scripts), before
  this freeze and before any computation; it ran on the invented fixture of the tests only.
- **v7 is v6 with one change.** Everything else is inherited, section by section, from
  `docs/s6-route6-prereg.md` (frozen on 18/11/2026, with its amendments 1 and 2), and through it
  from v5 and v4.
  - "As v6 §x" means that v6's text, with its amendments 1 and 2, applies word for word, with these
    names changed:
    - `runs/s6-route6*` becomes `runs/s6-route7*`, `models/s6-route6` becomes `models/s6-route7`,
      and the model's name `s6-route6-nano` becomes `s6-route7-nano`;
    - the build `build6/` becomes `build7/`, v6's fix folder `build6/fix/` becomes
      `build7/fix7/`, and the `route6` scripts become the `route7` scripts of §14 where §14 names
      one (`scripts/s6-route6-*.py` are reused as they are);
    - τ_ask6 becomes τ_ask7.
  - The data's own names are kept: `v5_tools`, T5 and `t5_*`, `<data>/s6-route/toucan5/`.
  - **Where v6 compares with v5's build,** v7 compares with v6's (`build6/`), and v5's numbers are
    given beside them.
- **Fixed later, by rules written here, each before any training:** as v6 (the head window, the
  pilot's potion-cos bar, the counts amendment of T5), and the hashes of the two rebuilt When2Call
  record files and of the new teacher files, by amendment 1, written right after the rebuild and
  its checks, before any training (§4.1, §4.8).

**Why this experiment.**

- **v6's stop.** v6 removed the exempt tools from every slate of the When2Call group and stopped at
  its own stop rule (§4.8 of v6): N6's excess over the *none* share was +0.3792 in the training
  "other" family and +0.4000 in the validation one, against v5's +0.2410 and +0.2382.
- **The mechanism, counted after v6's stop** (v6's result block, `runs/s6-route6/no-call-anchors.json`).
  - **No anchor, no cell.** A no-call slate is planned in the cell of the request's first own tool
    that the pool allows. In training, 660 of the 1,984 hard and natural no-call slates have no such
    tool: for 608 the request lists no tool at all, and for 52 none of its tools is in the pool. In
    validation, 33 of 121 have none. These slates sit at L6 and are drawn from the whole pool.
  - **They spread, the others do not.** The 660 spread over several L0 cells 83.2% of the time
    (549), as in v5. The 1,324 anchored no-call slates spread 5.2% of the time (69), against 5.7%
    for the tool-right slates (449 of 7,934).
  - **N6 picks them.** Once nearly nothing else spreads, the median spread is 0 and N6 reads
    "*none* when the options spread at all": 549 of the 618 *none*-right slates among its 1,067
    answers in training are unanchored no-call slates.
- **What v7 does.** It gives these slates the cell that the others have: the cell of the pool tool
  nearest to the request. This is option (c) of v6's result block.

**This is the third construction tried for this group, and the last.**

- v5 showed the request's own tools on the no-call slates (exempt); v6 drew every option in the
  first own tool's cell; v7 adds an anchor for the slates that have no own tool in the pool.
- **If the stop rule fires again, the line is closed.** There is no v8 of this line: no fourth
  construction of the When2Call group, no version that drops the group (option (b) of v6's result
  block is not taken afterwards either). The published pair (v3) stands with its stated limits.

**What the mechanism count predicts, and what it does not.** Leaving the 660 unanchored slates out,
the anchored slates of v6's training "other" family answer *none* on 69 of the 518 slates whose
options spread: 0.133, against a *none* share of 0.143 among them (v6's result block). If v7's
anchored slates behave like v6's anchored slates, N6's excess would fall near zero. **That is a
count of where the cue sat in v6, not a guarantee:**

- the 660 slates get a cell for the first time, and how often they spread there is not known
  before they are built;
- a slate with a different cell is drawn again (its fill, its hard candidates, its teacher answers);
- once the 660 stop spreading, the family's median spread and the slates that N6 picks can move
  again, as they moved in v6;
- the anchor is not a tool the request owns. It is the tool nearest to the request, so the cell is
  chosen by the request's own text, which the tool-right slates' cells are not (§10).

## 1. The question

As v6 §1.

## 2. What changes from v6, and why

| # | change | evidence | where it is read |
|---|---|---|---|
| A0 | **The anchor of the unanchored no-call slates:** a no-call slate of the When2Call group whose request has no own tool in the pool is anchored on the cell of the pool tool nearest to the request; the stop rule of v6 §4.8 judged again after the rebuild | v6's result block: 660 of 1,984 training no-call slates have no anchor, sit at L6 and spread 83.2% of the time, and N6's answers are mostly theirs | §4.8 |
| A1 | **v6's built data are v7's training and validation data, as they stand,** except the records of A0 (hashes, §4.1) | v6's build is complete; only its stop rule stopped it | §4.1 |

**Not changed:** everything of v6, with its amendments 1 and 2: the system (§3), the teacher and
its prompt (§4.2), B0 and B1 and the placeholder `?`, already taken (§4.3), the data checks
reported (§4.4), the baselines (§4.5), the readings and leakage (§4.6), the disk floor (§4.7), the
stop rule of §4.8 and the spread table, the run and the pilot rule (§5), T5 under v5's rule (§6),
the evaluation (§7), G1 to G6 and the headline (§8), the reported readings (§9). v6's diagnostic of
the published files (v6 §9.1) ran and is published; it is not run again.

## 3. The system

As v6 §3. The head window is chosen by v4 §3.3's rule on v7's records
(`scripts/train-route7.sh window`).

## 4. Data

### 4.1 The inputs, as built (A1)

The training and validation data are v6's build under `<data>/s6-route/build6/`, written on
18/11/2026 (v6's result block). It is v5's build with the When2Call group rebuilt by v6's
amendment 1, and v6's fix folder `build6/fix/` (its plan and its 694 teacher answers).

- **The rebuild writes a full copy under `<data>/s6-route/build7/`,** checked file by file
  (`runs/s6-route7/copy.json`). v7's fix folder is `build7/fix7/`; the copy's `build7/fix/` is
  v6's fix, read for its answers and for the comparison of §4.8, never written.
- **Every file of the table below keeps its hash in `build7/`,** except
  `records/when2call/train.jsonl`, `records/when2call/val.jsonl` and `counts/v5-slates.json`.
- **Within those two record files, only the records of A0 change:** the *none*-right natural and
  hard slates of the requests whose plan the anchor changed. Every other record of the group is
  byte-identical to v6's, by uid (§4.8).
- **The new hashes,** of the two record files, of `counts/v5-slates.json` and of the new teacher
  files under `build7/fix7/teacher/`, are written by amendment 1 right after the rebuild and its
  checks, before any training, with the records' counts.
- **A difference elsewhere** is a technical failure of the rebuild: it is fixed before anything
  else, and reported.

**The hashes** (sha256, bytes) of v6's build as v6 left it: the 37 files of
`runs/s6-route6/rebuild.json`, and v6's fix files of its result block. They are copied to
`runs/s6-route7/inputs.json` before anything runs; the stage `inputs` of §14 refuses to go on if
one differs.

| file | bytes | sha256 |
|---|---:|---|
| `counts/v5-plan.json` | 18,348 | `c887b15f2b0d273013b2fe9d6d7ec8db961b4a829d99d03b8bd57d0f8cae4f20` |
| `counts/v5-slates.json` | 18,173 | `d1d38894a35cf815d8de0a3ab00a1ab768f1ecc20209fd817ed254bf966c0800` |
| `emb/potion-requests.meta.json` | 755 | `4c8000f1fd24ffb585633d78df61369bece34cecca0611d1add852c08f361b70` |
| `emb/potion-requests.npy` | 17,738,368 | `66f23fb02d59a3fcbebea3c1aa199d20730956d3c1195e2910a29ff961f80763` |
| `fix/stage/plan5.jsonl` | 92,251,077 | `78da2d9e6ad67329055e439d864b2b8ba32216c5bbcb4d2f11a98037b5d5c31f` |
| `fix/teacher/false-negatives.items.jsonl` | 401,646 | `6ca7b307b94c81c80d02e57dfdb519915f98b74aca3a96877e98d159a58d299b` |
| `fix/teacher/false-negatives.jsonl` | 33,034 | `41cc27993c2826dad11d6207b2b367e8d72d2b178830a45899e5996651e1ea4e` |
| `fix/teacher/false-negatives.refused.jsonl` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `logs/v4-removed.txt` | 1,207 | `011db0d335339a0c2b314b410cf0c74a56b88ffc2929ee701cb36c962b8067b3` |
| `logs/v5-plan.json` | 18,348 | `c887b15f2b0d273013b2fe9d6d7ec8db961b4a829d99d03b8bd57d0f8cae4f20` |
| `logs/v5-slates.json` | 21,555 | `48d3afd81e028fc3235caa1d04b2fe86bcfc24f0732d9223c3021ca320bd41d0` |
| `markers/build.done` | 20 | `b05cace93041fe6ea1ef157bf163c4f48a564adc7868f4c27f67a1cb0d747e2e` |
| `markers/c6.passed` | 20 | `78aaf859c81675d625f962275f7b02f7a52387553d9da54833a987e5eb394026` |
| `markers/embed-requests.done` | 20 | `8eb20b58f93572d59cb290f28cbe2a0cf9508510927fafdefb3ee46977306192` |
| `markers/plan.done` | 20 | `055a4471ef1882b23379655884c6685890e56c561f8b1204e7cbd8338240dff5` |
| `markers/teacher-jobs.done` | 20 | `0c7a8cbb489e265f5f586dc97504520321cd372e442e84cee7c704b39e6cf799` |
| `records/intents/train.jsonl` | 95,646,509 | `6a68f686818d592893d01178c0c8bd3f95fe5ad5817f8da1e56ef92fb81bf48b` |
| `records/intents/val.jsonl` | 11,393,779 | `b810a7fcd1dca65961d51bea0f6dc3e2820c6ff70a1290001a28385b54101490` |
| `records/tools/train.jsonl` | 99,350,230 | `47448bcfb0641fe0da57b2737185a577fb0c6520fda01fac3fd692cdb4f67463` |
| `records/tools/val.jsonl` | 5,095,890 | `488d5c9194dee948fbe2a9777684787d376a5e277b4d32b2654fef8cd2b44e73` |
| `records/when2call/train.jsonl` | 15,949,866 | `8cb01eee5297492947f5bb56564438093a2e0cc97ed560d891908c0ed98b2f00` |
| `records/when2call/val.jsonl` | 966,319 | `1cd5d00dab7dcb2cd48aad72b82285b1fb7ff15ef954176bb79664aab6583908` |
| `stage/plan5.jsonl` | 92,289,168 | `c3fe7dc7166bb0dd063313d6036f9167a7a13cbaf1036cd2d37bbc8f04fde9d7` |
| `stage/pools5.json` | 1,317,247 | `6d459aa06283017bf1552b8e39a9d40bd8b6d2eb137a210009db2ab2918787fc` |
| `teacher/false-negatives.items.jsonl` | 36,201,112 | `683e9f75e3b42d36f0c6e547e66e389b449c95dac1bdc920985be49915d14bb0` |
| `teacher/false-negatives.jsonl` | 2,805,563 | `145fd7d8fbace537712975379fc9cadc0e6b824c3d73422b31f3e27cc982b3cf` |
| `teacher/false-negatives.refused.jsonl` | 91 | `4a3a304630e8529e2531feb4ef16398c8a055cd028324a5ff5ad80d55a8ca4d4` |
| `val/r_tools_natural.model.jsonl` | 1,347,426 | `00fa2c8c62858f51f2bae64d19f5917bffbd622c64ab126738cb0a9deb8736a5` |
| `val/r_tools_renamed.model.jsonl` | 1,720,186 | `87eb3e1bbbb70c0d44756c5e217f636143a11b00e9a65849340352b1ab10eb11` |
| `val/v4_tools.model.jsonl` | 1,663,413 | `1fcc42718bf4b20b3ea9e508ff4b50863fb3ab2b93627d8772ae7f7a9d619bf6` |
| `val/v4_tools_masked.model.jsonl` | 1,579,075 | `1248da55e99dabd62e0681a35da4bd992fbd55d6978e2117bcbea7d65324e1c3` |
| `val/v5_tools.jsonl` | 1,603,383 | `767a6699e5c16633a2d00a87721d14de386295ada79555454b791842f210f704` |
| `val/v5_tools.model.jsonl` | 1,755,565 | `87de9c6875a3bcb1fcee6de2856762280a0f2fc697bc3e746d152efa3ac4084e` |
| `val/v5_tools_masked.jsonl` | 1,513,147 | `790660875170506ab28be0cabc13b59c36665b5530e850fb577716015e23d0ac` |
| `val/v5_tools_masked.model.jsonl` | 1,665,328 | `e6ea0f85b3b372772549948d6de2968f93f1ceecb65dfce1be514b8ba1d90ad6` |
| `val/v5_tools_masked_rw.jsonl` | 1,530,864 | `fc29557cde676596457b4ff2d2c32db7c7bc685dd2042da161b103a27f2b3f46` |
| `val/v5_tools_masked_rw.model.jsonl` | 1,683,045 | `b41817aba2de222ba5139b859fc38f59d3c3063db2418ee8b7fff7b0ad524062` |
| `val/v5_tools_rw.jsonl` | 1,621,100 | `495c2e357aeb9f49056dae689cb5d838ce39665cea5765d81909ada843cdf14f` |
| `val/v5_tools_rw.model.jsonl` | 1,773,282 | `1b10219e8b3d1fe75966512163d1d32f69a1f4bc9eb249a68c233730903f9fdf` |
| `val/v_tools.model.jsonl` | 1,729,811 | `c72afb5955e1737e33ac949d851fce3920925901debebbe82575869ffd080ef1` |
| `val/v_tools_masked.model.jsonl` | 1,641,952 | `d18ee12b4e5cbcacc7e78f5cda443725e23fab1c1acb5af760afe8c5c93555a3` |

- **The records:** 197,958, as v6 (training 182,085, validation 15,873; When2Call 13,885 and
  847). The anchor changes no uid and drops no request, so these counts and C3's shares do not move.
- **The marker.** As v6: the scripts read `runs/s6-route7/inputs.ok`, written by the stage
  `inputs-ok` against `runs/s6-route7/inputs6.json` (the hashes of amendment 1).

### 4.2 The teacher

- **The false-negative answers of v5 and of v6's fix are used as they are,** by id, except for the
  slates of §4.8.
- **The fix's job:** v5's `false-negatives` job, v2's prompt word for word, GLM-5.3-Flash (MIT), on
  the rebuilt hard no-call slates of §4.8 that a record shows and that no answer of v2, v5 or v6
  covers by id. Counted read-only on `build6/` before this freeze: the *none*-right hard slates at
  L6 are 329 in training and 20 in validation, so **at most 349 items, about 18 calls** of 20, about
  a minute at v6's rate (694 items in 33.4 s). The teacher sees training and validation requests
  only. A count above 349 is a technical failure of the plan, fixed before any call.
- **The `fresh-tools` job:** as v6 §4.2, about 100 to 150 calls, for T5's probe copies only.

### 4.3 to 4.7

As v6 §4.3 to §4.7.

- **The placeholder** is `?`, already taken by v6 (`runs/s6-route6/placeholder.json`, copied to
  `runs/s6-route7/placeholder.json`). The stage `placeholder` checks it against v7's records and
  does not choose again (v6, amendment 2, point 3); the counts amendment checks it against T5.
- **Chance on `v5_tools`:** c = 1/7 = 0.142857, c + 0.05 = 0.1929 (v6 §4.3; the file is v5's,
  unchanged).
- **The checks of v6 §4.4** run on all of v7's records after the rebuild and are reported; only the
  stop rule of §4.8 can stop v7.
- **Disk:** 3 GB free before every writing stage (`MIN_FREE_GB=3`); about 68 GiB were free on
  18/11/2026 at the freeze.

### 4.8 The one change, and the stop rule (A0)

**The rule.** In the When2Call group, a no-call slate (hard or natural) whose request has no own
tool in the pool that the slate draws from is anchored on the **pool tool nearest to the request**:

- **nearest** by the cosine of potion-base-2M embeddings, the request's embedding (v5's, the file
  `emb/potion-requests.npy` of §4.1) against each pool tool's embedding (v5's); of equal cosines,
  the lowest tool id;
- **its cell** is that tool's cell, at the lowest level of v5's widening (L0 to L6, v5 §4.3) that holds the
  minimum of 16 candidates, counted on all the slate's tools since every one is drawn (v6,
  amendment 1);
- **natural slates** take the cell's random fill, as the anchored natural no-call slates of v6;
- **hard slates** take the tools nearest to the request within that cell, as v6's anchored hard
  no-call slates; the anchor tool itself is a candidate like any other; the false-negative filter
  is v5's (a tool that the teacher names able is replaced by the next of the request's nearest in
  the cell, then from the cell's random fill, then from the next levels);
- the slate's size and the stream that drew it are v6's.

**What does not change.**

- **Every other plan,** the anchored no-call slates of the group, its tool-right slates, its
  renamed slates and every plan of the other groups, is v6's, byte for byte. This is checked on the
  plan file against v6's fix (`runs/s6-route7/plan-diff.json`): only no-call plans of the group
  whose two slates were at L6 in v6 may differ.
- **Every record outside the rebuilt slates** is v6's, byte for byte, checked by uid against
  `build6/` (`runs/s6-route7/records-diff.json`), and every other file of `build7/` keeps its hash
  of §4.1 (`runs/s6-route7/rebuild.json`). This is the same proof that v6 gave against v5.
- **The slates rebuilt:** the hard and natural no-call slates without an anchor that v6's records
  hold, rebuilt by uid. Counted read-only on `build6/`: 660 in training (329 hard, 331 natural) and
  33 in validation (20 hard, 13 natural).

**The order, all before training** (`scripts/build-route7-data.sh`, §14): as v6 §4.8. The copy and
the replanning (`fix-plan`), the fix's items and the teacher (`fix-teacher-jobs`, then
`scripts/route-teacher.sh false-negatives loop` on `build7/fix7/`), the rebuild (`fix-build`), the
checks (`fix-checks`), the spread table (`spread`, with v6's build beside it), then amendment 1.

**The stop rule: v6 §4.8, unchanged.**

> **If, in any N6 family judged by v5's floor (at least 200 tool-gold slates; v5, amendment 2,
> decision 4), the accuracy of N6 or of its inverse exceeds the family's *none* share by more
> than 0.10, v7 stops before training, and the table is published. Otherwise the checks are
> reported, none of them stops v7, and the model's blind reading decides (G6).**

If it fires, the line is closed (§11).

**The spread table, before training** (reported): as v6 §4.8 and its amendment 1, point 4: the
share of When2Call slates whose options spread over several L0 cells, and the counts by level, by
kind, by *none*-right or tool-right and by split, after v7's rebuild, with v6's build beside it
and v5's numbers quoted.

## 5. The run (`scripts/train-route7.sh`)

As v6 §5: the smoke, the pilot of 20 working minutes with a checkpoint kept every 5, the main run
of 120, the readings in B0 and B1 at every kept checkpoint, the calibration, the export, τ_ask7 and
the fit of β and τ_ask7 on `v5_tools`. **The pilot rule is v6's,** rules 1 to 5: abort below
`potion-cos` on `v5_tools` − 0.15 (the bar written in the amendment of §4.5 before the pilot), above
0.30 on the rewritten distractor of `v5_tools_masked_rw`, or above a blind accuracy of 0.1929 on the
915 tool-right slates of `v5_tools` or of `v5_tools_masked`.

## 6. The fresh test files (T5)

As v6 §6: T5 under v5's rule (v5 §6 with its amendments 1 and 2), never built so far, under
`<data>/s6-route/toucan5/`, seed 1236, the counts amendment before any scoring (T5's chance c, the
placeholder checked against T5's requests, the server-mates of each tool-right slate).

## 7 to 9. Evaluation, gate, reported

As v6 §7 to §9. **G1 to G6, the headline, the report's sentence and the announced goals: as v6 §8,
no bar moves.** G1's lower bounds above 0; G2 (a) ≤ 0.1929 and paired ≤ +0.03, (b) ≥
`potion-cos-4bit` − 0.02, (c) *none* on 0.10 to 0.30; G3 ECE ≤ 0.05; G4 max |Δp| ≤ 1e-3, ≥ 199 of
200, median ≤ 60 ms, ≤ 4,000,000 bytes; G5 T2-tools ≥ 0.8668; G6 (a) and (b) blind accuracy ≤ c +
0.05 with the servers' upper bound ≤ c + 0.08, (c) Δ_none ≤ 0.05 with the servers' upper bound ≤
0.15. The pair's B1 is not read (v6, amendment 2, point 2). The diagnostic of v6 §9.1 is cited, not
run again.

## 10. Disclosure (written into every place that reports v7)

As v6 §10, plus:

- **v7 was designed after v4's, v5's and v6's tables were read.** Its one change follows v6's
  table and the mechanism count made after v6's stop (v6's result block), and it is the third
  construction of the When2Call group's no-call slates.
- **The anchor is chosen with the request.** A tool-right slate's cell is the right tool's, and an
  anchored no-call slate's is the request's own first tool's. An unanchored no-call slate's cell is
  now the one nearest to the request's text. A model could, in principle, learn that a request
  close to every option of its slate is more often a no-call request. B0 does not see that cue,
  since it hides the request (v6 §11); the report states it.
- **The stop rule was written before v6's rebuild and is not changed.** The prediction of the
  mechanism count (0.133 against 0.143) was written before v7's rebuild, as a count and not as a
  bar.
- **The rule that closes the line** (§11) was written before any computation of v7.
- **A reader who holds that the data should have been blind** may hold v7 to that view, as v6
  §10 says; v7's tables are published with v4's, v5's and v6's.

## 11. What a failure would mean

As v6 §11, with this point in place of v6's first:

- **The stop rule of §4.8 stops v7: the line is closed.** After three constructions, a *none* cue
  of more than +0.10 above the *none* share remains in at least one judged family of the When2Call
  group. No model is trained, the table and the spread table are published, and there is no v8 of
  this line: no fourth construction and no version without the group. The published pair (v3)
  stands with its stated limits (a pass on fresh slates of v2's held-out tools, the names-masked
  weakness, the cues of v2's training data named by v4's table, the blind reading at chance), and
  the report says in one sentence where the cue now sits, from the spread table.
- **N6 is above its tolerance but within +0.10:** v7 trains, and G6 (c) reads whether the model
  uses what remains (as v6 §11).

## 12. Forbidden

As v6 §12, with v7's names, plus:

- changing the anchor's rule (the slates it applies to, the measure, the ties, the level, the
  minimum of 16, the hard slates' candidates) after any step of §4.8 has run;
- rebuilding, filtering, reweighting or editing any record other than the slates of §4.8, or
  training on a file whose hash differs from §4.1 or from amendment 1;
- any teacher call other than the fix's false-negative job of §4.2 and the `fresh-tools` job;
- a v8 of this line, or any change to the When2Call group, if the stop rule fires;
- citing the mechanism count's prediction as a bar, or the stop rule's outcome as evidence about a
  trained model.

## 13. Cost

| item | amount |
|---|---|
| teacher | the fix's false-negative job, ≤ 349 items (329 training, 20 validation, counted read-only on `build6/`), ≈ 18 calls (about a minute); the `fresh-tools` job, ≈ 100 to 150 calls (v5 §4.2) |
| CPU | hash check of §4.1 ≈ 1 min; the copy and its check ≈ 2 min; the replanning ≈ 5 min, the rebuild ≈ 5 min, the checks and the spread table ≈ 5 min; then as v6 §13 (the placeholder check and the support count ≈ 3 min, the checks of the read files ≈ 5 min, the baselines ≈ 8 min, Toucan ≈ 20 min, the pilot readings ≈ 25 min, the baselines on T5 and T2 ≈ 30 min) |
| GPU | as v6: ≈ **168 working minutes**; ≈ 25 if the pilot rule aborts; none if the stop rule fires |
| disk | the full copy `build7/` with the rebuilt records ≈ 0.55 GB; then as v6 (T5, the blind copies, the checkpoints, the dumps ≈ 1.3 GB): **≈ 1.9 GB new**; floor 3 GB free |

## 14. Code to write before any run (by file)

**Reused as is:** every item of v6 §14 and of its amendment 2 that is written, in particular
`scripts/s6-route6-checks.py` (the checks, the stop rule, the spread table), `scripts/s6-route6-blind.py`,
`scripts/s6-route6-pilot.py`, `scripts/s6-route6-fit.py`, `scripts/s6-route6-gate.py`,
`scripts/route-teacher.sh` and `sky-data route blind`.

**To write:**

0. **The anchor (§4.8):** `crates/sky-data/src/route/v5.rs` and `crates/sky-data/src/main.rs`.
   - `sky-data route v5-plan --when2call-anchor own|nearest`. `nearest` anchors a no-call slate of
     the When2Call group without an own tool in the pool on the pool tool nearest to the request.
     The default `own` keeps v6's bytes and v5's.
   - `sky-data route v5-slates --also <dirs>` and `v5-fix-items --also <dirs>`: more folders whose
     teacher answers apply by id (v6's fix, `build7/fix/`).
   - Tests:
     - an unanchored no-call slate gets the nearest tool's cell, below L6, its tools all drawn and
       within that cell at the slate's level, and the hard ones among the request's nearest
       candidates within it;
     - the slates' sizes are kept, and the plan counts name every anchored slate;
     - every other plan, anchored no-call slates included, is v6's, byte for byte, on the fixture;
     - the frozen bytes of v2, v3, v4, t4 and v5, and a new frozen-bytes test of v6's fix on the
       fixture (its plan, its staged and teacher items, its plan counts and the rebuilt When2Call
       records), frozen on the code of `main` before the switch was written.
1. **`scripts/build-route6-data.sh`:** its folders and switches as environment variables (the fix's
   folder, the markers and logs, the plan's switches, the folders of answers by id, the previous
   fix to check against, the report and the build beside the checks and the spread table, the
   baselines' folder). Its defaults keep v6's commands. With a previous fix to check against,
   `fix-plan` writes `plan-diff.json` and `fix-build` writes `records-diff.json`, and each stops on
   a change outside §4.8.
2. **`scripts/build-route7-data.sh`:** thin over v6's: `build7/` copied from `build6/` with the hash
   proof, markers and logs under `build7/markers7/` and `build7/logs7/`, the stages `inputs`,
   `fix-plan`, `fix-teacher-jobs`, `fix-build`, `fix-checks` and `spread`, then v6's later stages
   on `build7/` and `runs/s6-route7/`. `DRY_RUN=1` and `MIN_FREE_GB=3` as v6.
3. **`scripts/train-route6.sh`** takes its names from `ROUTE_TAG` (default `s6-route6`), and
   **`scripts/train-route7.sh`** runs it with `s6-route7` on `build7/`.
4. **`scripts/s6-route6-checks.py spread --before-label`:** the name of the build beside ("v6's
   build" for v7).
5. **`DATA_LICENSES.md`:** no new source; the false-negative paragraph names v7's job.
6. **What remains to write before the gate:** as v6, amendment 2, point 6, with v7's names (the
   embedding file that stores β and τ_ask7, the t5 split refused until `runs/s6-route7/FRESH-OPEN`,
   the Python reference of the pair, G4's bench).

## 15. Choices (the value taken at the freeze; the alternative is recorded)

| | choice | taken | alternatives | why |
|---|---|---|---|---|
| O45 | the slates without an anchor | anchored on the cell of the pool tool nearest to the request (the lead's design) | drop the When2Call group from training (v6's option (b)); a random pool tool; the nearest tool's server | the measure is the hard slates' own; a random anchor would not tie the slate to anything; dropping the group removes the only training slates from requests that no tool answers |
| O46 | the measure and the ties | potion-base-2M cosine on v5's request and tool embeddings, ties by tool id (the lead's design) | MiniLM; BM25 | the measure the hard slates already use, with no new embedding |
| O47 | the anchor as a candidate of the hard slate | a candidate like any other (the lead's design) | excluded | it is a drawn tool of the cell; excluding it would make the nearest tool the one tool never shown |
| O48 | after a stop | the line is closed, no v8 (the lead's design) | a fourth construction; option (b) | three constructions tried; the published pair stands |

## 16. Amendments

### Amendment 1 (18/11/2026, after the rebuild and its checks, before any training): the stop rule does not fire; the records' hashes and the spread table

**What ran, in order** (18/11/2026, `scripts/build-route7-data.sh`, the logs under
`build7/logs7/`). Nothing else of v7 ran: no blind copy, no fresh file, no baseline, no training.

- **`inputs`:** the 41 files of §4.1 have their hashes in `build6/` (`runs/s6-route7/inputs-check.json`,
  none differs).
- **`fix-plan`:** the full copy of `build6/` into `build7/`, 55 files, each checked by sha256
  (`runs/s6-route7/copy.json`). The plan with `--when2call-anchor nearest` keeps 55,790 plans
  outside the When2Call group as v5's. Against v6's fix plan (`runs/s6-route7/plan-diff.json`),
  66,370 plans are byte-identical. 2,920 differ, all of them no-call plans of the group whose two
  slates were at L6 in v6 (2,792 training, 128 validation, the plan counts' "anchored on the tool
  nearest to the request"); no other plan differs.
- **`fix-teacher-jobs`:** of the 349 rebuilt hard no-call slates that a record shows (329 training,
  20 validation, as counted in §4.2), 53 are answered by v5's answers with the same id and **296 go
  to the teacher** (279 training, 17 validation), in 15 calls of 20. The other 747 no-call hard
  slates that a record shows are answered by id (v6's 694 and those 53); 8,151 planned no-call
  hard slates are shown by no record and get no item (`build7/logs7/v6-fix-items.json`).
- **The teacher:** GLM-5.3-Flash, v2's prompt, `BUILD=<data>/s6-route/build7/fix7
  scripts/route-teacher.sh false-negatives loop`; 296 answers out of 296, none refused by
  moderation, no failed batch; 47,608 tokens in and 4,156 out, 16.1 s. **140 of the 296 answers
  name at least one tool able to serve the request** (47%, against 87 of 694, 13%, in v6's fix),
  which the build then replaces. A slate anchored on the tool nearest to the request puts the
  tools closest to it in front of the teacher, and for requests that list no tool the teacher
  often judges one of them able.
- **`fix-build`:** the group's 14,732 records rebuilt by uid.
  - **Against v6's records, by uid** (`runs/s6-route7/records-diff.json`): 660 training and 33
    validation records differ, every one a *none*-right natural or hard slate of a replanned
    request: the 660 and 33 slates of §4.8. The other 13,225 and 814 records of the group are
    byte-identical to v6's.
  - **Every other file of `build7/`** has its hash of the copy (`runs/s6-route7/rebuild.json`:
    52 files equal, 3 allowed to differ, none other differs).
- **`fix-checks`:** exit 0, **the stop rule does not fire** (`runs/s6-route7/checks.md`,
  `checks.json`).
- **`spread`:** `runs/s6-route7/spread.md` (`spread.json`), with v6's build beside it.

**The stop rule after v7's rebuild** (N6 "*none* when the options spread above the median"; every
judged family; the inverse rule answers no slate in these families).

| family | slates | *none* share | v5: excess | v6: answered *none* | v6: accuracy | v6: excess | v7: median spread | v7: answered *none* | v7: accuracy | v7: excess |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `records when2call train other` | 9,918 | 0.2000 | +0.2410 | 1,067 | 0.5792 | +0.3792 | 0 | 563 | 0.2025 | **+0.0025** |
| `records when2call val other` | 605 | 0.2000 | +0.2382 | 55 | 0.6000 | +0.4000 | 0 | 30 | 0.2667 | **+0.0667** |

- **Both are within N6's tolerance** (0.2538 in training, 0.3661 in validation), so N6 fails in no
  family. The validation family's +0.0667 rests on 30 answers, 8 of them right.
- **The largest excesses of every judged family** are now +0.0667 (above), +0.0501
  (`records when2call train renamed`, 4 slates answered, unchanged from v6), +0.0249 (N6's
  inverse on `v5_tools_masked.jsonl`), +0.0197 (`records tools val renamed`) and +0.0178 (N6's
  inverse on `v5_tools.jsonl`). The last four come from files v7 does not touch.
- **The mechanism count's prediction** (0.133 against a *none* share of 0.143 among the spread
  anchored slates of v6) was a count, not a bar; the family as rebuilt reads 0.2025 against 0.2000.
- **v5's other checks, rerun on all the records and reported** (none stops v7): 17 failures of C2
  and the N rules, against v6's 19. The 17 failures of the cells are v6's, cell by cell and at the
  same values; the two N6 failures are gone. C1 fails in 7 training cells, as in v6.

**The spread table** (`runs/s6-route7/spread.md`; the share of slates whose tool options spread
over several L0 cells, and the share at L6, outside the cell).

| split, kind, answer | slates | v7: spread above 0 | v6: spread above 0 | v5: spread above 0 | v7: L6 | v6: L6 |
|---|---:|---:|---:|---:|---:|---:|
| train hard *none*-right | 980 | 0.0602 | 0.3245 | 0.6847 | 0 | 0.3357 |
| train hard tool-right | 3,967 | 0.0575 | 0.0575 | 0.0575 | 0 | 0 |
| train natural *none*-right | 1,004 | 0.0548 | 0.2988 | 0.6723 | 0 | 0.3297 |
| train natural tool-right | 3,967 | 0.0557 | 0.0557 | 0.3726 | 0 | 0 |
| train renamed *none*-right | 793 | 0.0013 | 0.0013 | 0.4098 | 0 | 0 |
| train renamed tool-right | 3,174 | 0.0009 | 0.0009 | 0.3948 | 0 | 0 |
| val hard *none*-right | 63 | 0.0635 | 0.3175 | 0.6667 | 0 | 0.3175 |
| val hard tool-right | 242 | 0.0413 | 0.0413 | 0.0413 | 0 | 0 |
| val natural *none*-right | 58 | 0.0690 | 0.2241 | 0.6207 | 0 | 0.2241 |
| val natural tool-right | 242 | 0.0496 | 0.0496 | 0.3719 | 0 | 0 |
| val renamed *none*-right | 48 | 0 | 0 | 0.4167 | 0 | 0 |
| val renamed tool-right | 194 | 0 | 0 | 0.3814 | 0 | 0 |

- **The *none*-right slates now spread as rarely as the tool-right ones,** in both kinds and both
  splits (within 0.03 everywhere, against 0.17 to 0.28 in v6), and none is at L6.
- **The slates by level (L0 to L6)** of the rebuilt kinds: training hard *none*-right 910 65 3 2 0 0 0
  (v6: 604 46 0 1 0 0 329), natural *none*-right 932 66 3 3 0 0 0 (v6: 633 38 1 1 0 0 331);
  validation hard 59 3 1 0 0 0 0 (v6: 40 2 1 0 0 0 20), natural 53 5 0 0 0 0 0 (v6: 43 2 0 0 0 0
  13). The tool-right and renamed slates are v6's, level by level.

**The hashes of the training and validation inputs** (`runs/s6-route7/inputs6.json`, 45 files: the
41 of §4.1, of which these three changed, and v7's fix files). The stage `inputs-ok` checks `build7/`
against it and only then writes `runs/s6-route7/inputs.ok`.

| file | bytes | sha256 |
|---|---:|---|
| `records/when2call/train.jsonl` | 15,946,230 | `e60c0bc6ec7248cf0f91377836ddd9643d4792ace1a0e4e70eb556cd020e6978` |
| `records/when2call/val.jsonl` | 966,478 | `9828c06af198615429b5597879aa6af05cf11dd4bdda1df7276cfd217492a30a` |
| `counts/v5-slates.json` | 18,035 | `eea839ab1f3c9e9ca7591c4bfff8650a2e321a9a8cec0ea8376644bc4a62adbb` |
| `fix7/teacher/false-negatives.items.jsonl` | 186,725 | `432c766756520cc5b34492c106abd08959fda66aa9ac54796bff07f712aa8a83` |
| `fix7/teacher/false-negatives.jsonl` | 14,215 | `136daaae138881a663e123f503580a6ae4d4f3b361221faacd947cb3b84f3256` |
| `fix7/teacher/false-negatives.refused.jsonl` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `fix7/stage/plan5.jsonl` | 92,251,077 | `b6a408b0a84eda9d18ace66a54165069d8518974daaea7208a7c5870f4f0667a` |

- **The records:** 197,958, as v6 (training 182,085, validation 15,873; When2Call 13,885 and 847);
  C3's shares are v6's (1,984 *none*-right of 9,918 and 793 of 3,967 in training; 121 of 605 and
  48 of 242 in validation).

**What follows, by the frozen text.** The checks are reported and none stops v7 (§4.8); the model's
blind reading decides (G6). The next stages are v6's, under v7's names, launched by the lead:
`inputs-ok`, `placeholder`, `support`, `checks-read`, `blind-val`, `baselines-val`, then the
amendment of §4.5 with the pilot's bar, the training (`scripts/train-route7.sh`), and T5. None has
run.

**Nothing else changes.** No bar, rule, file or choice of the frozen text moves.

### Amendment 2 (18/11/2026, from the stages' outputs, before any training): the pilot's bar, the validation baselines, the counts of T5

**What ran, in order** (18/11/2026, `scripts/build-route7-data.sh`; the outputs in
`runs/s6-route7/`). No model was trained, no system read T5 or its copies, and no blind copy of
T5 exists.

- **`inputs-ok`:** `build7/` has the 45 hashes of amendment 1 (`runs/s6-route7/inputs6.json`);
  `runs/s6-route7/inputs.ok` is written.
- **`placeholder`:** `?`, kept from v6, is no request of v7's 200,246 training and validation
  records and slates (`placeholder-check.json`, `"occurs": false`). Its first run stopped on a
  defect of the kept mode, which always exited 2; it was corrected (b75f7e3) before this run.
- **`support`, `checks-read`, `blind-val`, `baselines-val`:** below. The stage `checks-read`
  first stopped on the checks' own exit 1, although v6 §4.4 reports them and lets none stop the
  run; the stage now reports them and stops only on a crash (70f4a1a).
- **`fresh`, the `fresh-tools` job, `fresh-tools`, `probes`, `counts`:** T5 built once under
  `<data>/s6-route/toucan5/`, the counts below.

**(a) The pilot's bar** (§5, v6 §5 rule 1; v5 §4.5). The validation baselines are scored by the
frozen baseline code (`scripts/route-baselines.py`, v2's fitted values), accuracy over all 1,144
slates of each file, *none* included (`runs/s6-route7/baselines-val.json`):

| file | `potion-cos` | BM25 |
|---|---:|---:|
| `v_tools` | 0.7255 | 0.8549 |
| `v_tools_masked` | 0.7325 | 0.8444 |
| `v4_tools` | 0.7404 | 0.8322 |
| `v4_tools_masked` | 0.7386 | 0.8313 |
| `v5_tools` | **0.7281** | 0.8383 |
| `v5_tools_masked` | 0.7194 | 0.8313 |

- **The bar:** `potion-cos` on `v5_tools` − 0.15 = 0.728147 − 0.15 = **0.578147**
  (`POTION_BAR=0.578147`). The pilot aborts if the model part's accuracy on `v5_tools` is below it.
- **The change of difficulty** (reported, nothing is chosen on it): across v2's, v4's and v5's
  constructions both baselines move by less than 0.03 on the ordinary files; masking the names
  costs BM25 at most 0.011 and moves `potion-cos` by less than 0.01.
- **The baselines in B0** (`runs/s6-route7/baselines-blind-*.json`, `scripts/s6-route6-blind.py
  read`; reported, v6 §4.5): BM25 ties every option, so its blind accuracy on the 915 tool-right
  slates is c = 1/7 = 0.1429 exactly on both files, the check of the chain; it answers *none* on
  every blind slate. `potion-cos` reads 0.1104 on both (c − 0.0325), answers *none* on 0.919 and
  0.932 of the blind slates, with a blind *none* gap of +0.031 and +0.014.

**The support of B0** (`runs/s6-route7/support.json`, v6 §4.3): the share of tool-right records
in which every tool option scores 0.00 is 0.42 to 0.58 in the intents group, 0.026 to 0.036 in the
tools group and 0.021 to 0.033 in the When2Call group, by kind and split.

**The checks of the other files read** (v6 §4.4, reported, none stops v7;
`runs/s6-route7/checks-read-files.md`): v2's `v_tools` and its masked copy fail C2 and the N rules
in 10 rules each (R4 inverse, the fewest words, 0.2550 against a tolerance of 0.1860), the cue that
v4's table named. `v4_tools`, its masked copy and the tools group's renamed and natural copies
fail none. C3 "fails" on `r_tools_natural`, whose slates are all tool-right by construction (no
*none*-right slate). T2-tools (`f_tools`, read before) is not judged; its best rule is R5 inverse,
0.3849.

**(b) The counts of T5** (v6 §6, v5 §6.3; `runs/s6-route7/counts.json`, `t5-counts.json`, `test-files.json`,
`t5-checks.md`, `t5-placeholder.json`; counted by the builder, no request read).

- **The rows:** 26,134 qualifying rows read (16,477 `single-turn-original`, 9,657
  `single-turn-diversify`). Dropped: 7,571 whose gold is within v2's cluster rule of a training or
  validation tool, 1 repeated question, 15,609 over the cap of 10 per gold cluster, 64 over the cap
  of 40 per server.
- **The slates: 2,000**, above the floor of 600: **1,600 tool-right and 400 *none*-right** (none
  share 0.2000); 1,369 from `single-turn-original` and 631 from `single-turn-diversify`.
- **Servers and clusters:** 262 servers and 458 gold clusters. The three largest servers hold 79
  slates (27, 26, 26), a share of **0.0395**; the three largest clusters hold 30 (10 each, the
  cap), a share of **0.0150** (counted on `t5_tools.servers.jsonl`, uid, server and cluster only).
- **The levels** (v5 §6.2 item 5, reported): L0 411, L1 973, L2 120, L3 246, L4 250, L5 0, L6 0.
  C6: none of the 1,495 drawn slates is at L5 or L6.
- **T5's chance:** every tool-right slate holds 7 tools, so **c = 1/7 = 0.142857** and the bars of
  G6 (a) and (b) are 0.1929, with the servers' upper bound at 0.2229, as §8 says.
- **The server-mates** (v6 §6): the tool-right slates by the number of the gold's own server's
  tools shown, as the builder marks them (`roles`): 0: 396, 1: 225, 2: 123, 3: 175, 4: 111,
  5: 65, 6: 505.
- **The probe copies** (v5 §6.4): the `fresh-tools` job rewrote the descriptions of the 1,448
  Toucan tools that a fresh slate shows (GLM-5.3-Flash, v2's prompt, names and descriptions only:
  the items hold `id`, `name`, `description` and the split `fresh-tools`, no request); 1,448
  answers in 74 calls, none refused, no failed batch. 1,600 tool-gold slates have one distractor
  rewritten, in `t5_tools_rw` and `t5_tools_masked_rw`.
- **The placeholder kept:** `?` is no request of T5's 2,000 (`t5-placeholder.json`,
  `"occurs": false`).
- **The files,** as the build wrote them (`runs/s6-route7/test-files.json`; the files stay in the
  data folder, only their hashes are in the repository):

| file | records | bytes | sha256 |
|---|---:|---:|---|
| `t5_tools.jsonl` | 2,000 | 2,985,633 | `c7586ea4bf8a686819d2be453657e36181bb5c66a7062901b74c1fe831bf2e66` |
| `t5_tools_masked.jsonl` | 2,000 | 2,863,931 | `6a00450c028f5bc8c9116b8dd594af514591b9cf0d064e56f6ae48aebdc2a6bc` |
| `t5_tools_rw.jsonl` | 1,600 | 2,424,920 | `4c7e7a54f385d19a0f4f7cddc87c8dd144e12a9efb7de77ded72e47866157ea5` |
| `t5_tools_masked_rw.jsonl` | 1,600 | 2,327,427 | `39b1f68cc7d2af4ea71ba1e4290098802b74bd76f56a45d3379d91725b9d7fa9` |
| `t5_tools.servers.jsonl` | 2,000 | 222,755 | `906ea3d5bf743c41d2af2bc679e089097fe3d0e4ad30ee882bd1d0961ff257a6` |

  `test-files.json`'s own sentence names `runs/s6-route5/FRESH-OPEN` (the builder's text for
  v5's rule); for v7 the file that opens T5 is `runs/s6-route7/FRESH-OPEN` (§14, item 6).

**The request-blind rules counted on T5** (v5 §4.4's script, `runs/s6-route7/t5-checks.md`;
reported, never judged, as v5 §6 says). The best rule of each file, against chance 1/7 and v5's
tolerance of 0.1804:

| file | tool-right slates | best rule | accuracy |
|---|---:|---|---:|
| `t5_tools` | 1,600 | B, the minority of (name style, period, first word form) | 0.1929 |
| `t5_tools_masked` | 1,600 | N5 inverse, the lowest mean overlap | 0.1673 |
| `t5_tools_rw` | 1,600 | B, the minority of name style | 0.1888 |
| `t5_tools_masked_rw` | 1,600 | N5, the highest mean overlap | 0.1635 |

- Two rules of the surface-form family (B) read above the tolerance on the unmasked copies, by
  +0.0125 and +0.0084; they are not judged on a fresh file, and they are stated with G6.
- **N6 on T5:** within its tolerance on both files that hold *none*-right slates (`t5_tools`:
  0.1734 and its inverse 0.2075 against a *none* share of 0.2000; `t5_tools_masked`: 0.2086 and
  0.2107).

**What follows.** Every input of the training is fixed: the records by amendment 1, the pilot's bar
here, the placeholder, T5 and its hashes. The head window is chosen by the script's `window` step
(v4 §3.3's rule). Nothing else changes: no bar, rule, file or choice of the frozen text or of
amendment 1 moves.

### Amendment 3 (19/11/2026, decided by the lead, while v7's training runs): readings of the gate's code, fixed before the fresh test is opened

**When it is written.** v7's training runs (`scripts/train-route7.sh window pilot main export`).
The lead has read neither its pilot summary nor any trained model, and no number of v7 is known:
no accuracy, no β, no τ_ask7, no blind reading. T5 is closed (`runs/s6-route7/FRESH-OPEN` does
not exist), and no system has read it. The code it reads is on branch `route7-gate`
(`scripts/route7-gate.sh`, `scripts/route/t5-guard.py`, `scripts/s6-route7-reference.py`,
`sky-route embed-params`).

The code's readings of the frozen text and of v6's §7, §8 and §14, as the lead decided them:

1. **The names of τ_ask7.** τ_ask7 is stored in v7's embedding file
   (`models/s6-route7/skycmd-s6-route7-embed.gguf`) under the engine's key
   `skycmd-embed.route.tau_ask3`, and `runs/s6-route7/fit.json` names it `tau_ask6` (as
   `scripts/s6-route6-fit.py` writes it, `{tau, accuracy, coverage}`). These are names only: the
   value is v7's fit on `v5_tools` by v3's rule (§3), and β is stored under
   `skycmd-embed.route.beta`. Every other key and every tensor of the published
   `s6-route-embed.gguf` is copied byte for byte, and the published file stays byte-identical.
2. **Who opens the fresh test.** T5 is opened by the stage `open` of `scripts/route7-gate.sh`,
   once. It does so only after its checks: the pilot summary exists, did not abort, and was read
   at the bar of amendment 2 (0.578147); the exported model, the fit and the embedding file
   exist; every file of `runs/s6-route7/test-files.json` has its sha256. This replaces the file
   created by hand that v3 and v6 wrote of: the lead's decision. The path is
   `runs/s6-route7/FRESH-OPEN`. The sentence inside `test-files.json` names
   `runs/s6-route5/FRESH-OPEN`, the known wording error of amendment 2. Once `gate.json` exists,
   nothing reads T5 again, and once T5 is open, neither the fit nor the embedding file can change.
3. **G4's median.** It is the bench's `median_cached_us`
   (`crates/sky-web/bench/route3-node.mjs --render scores-in-options`): the same tool list asked
   again, with the embedding part's tool list cached and the model part encoding the options
   again. That is what a request costs under D4 once a tool list is known (option (i) of v4 §3.4
   and §8). The first-call median (`median_first_us`) is reported beside it (`runs/s6-route7/g4.json`,
   under `bench`) and decides nothing. The bar stays 60 ms.
4. **B1** is read for the model part only, as v6's amendment 2, point 2 says. §7's "B1 for the
   model part and the pair" is read as that amendment reduced it.
5. **The extra baselines.** `static-ft`, `pair-lr`, `toolrouter` and `name-prior`, which v3 read,
   are scored on T5 with v2's fitted values beside §7's systems. They are reported and decide
   nothing.
6. **Where the blind copies go, and G4's slates.** T5's blind copies (B0 and B1 of T5-tools and of
   its masked copy, the kept placeholder `?`) are written after `open`, beside the test files
   under `<data>/s6-route/toucan5/fresh/`, as v6's blind copies of `v5_tools` sit beside it.
   G4's 200 slates are the first 200 of T5-tools, as v3 did.
7. **The one look at T5 before it is opened.** To find the split the T5 records carry, the code's
   author read, on 18/11/2026, the field names of `t5_tools.jsonl` and of `t5_tools_rw.jsonl`, and
   the count of their `split` values (all 2,000 `fresh`). No request text, option or target was
   printed or kept, and no system read the file. Every evaluation call on T5 passes
   `--split fresh`.

**Nothing else changes.** No bar, rule, file or choice of the frozen text or of amendments 1 and 2
moves.
