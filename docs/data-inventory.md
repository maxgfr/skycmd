# Where the training data of every model lives (inventory of 18/11/2026)

This file says, for every skycmd model, which data it was trained, validated and tested on, where
that data is on Hugging Face (under the account `maxgfr`), where the local copy is on the
project's machine, and whether the local copy may be removed. It is written for anyone who
reuses the project: to rebuild a model, to find its data, or to know what exists only on one
disk. It changes nothing by itself: nothing was deleted, moved or uploaded to write it. The
deletions of 19/11/2026 were the owner's decision and are recorded in section 8; the two later
passes, on the evening of 19/11/2026 and on 20/11/2026, in sections 9 and 10; section 11 says what
is left on the disk on 20/11/2026, why, and what is lost. Section 12 records the last publication
of 20/11/2026 (the G7 files) and the deletion that followed it.

**Update of 19/11/2026 (s6-route v7).** s6-route v7 trained, was read once on its fresh test and
failed its gate; the route line is closed. Its data (section 3.6) is assembled as the data set
`skycmd-route7` by `scripts/hf-publish-data.py --only route7`, and was uploaded to the Hub on
19/11/2026 as `maxgfr/skycmd-route7`. The rest of this file is the inventory of 18/11/2026 and
was not read again, except where the update below changed it.

**Update of 19/11/2026, later that day (closures and deletions).** Three decisions of
19/11/2026 change who reads what: the LFM3 fine-tune is closed, with nothing trained
(`docs/lfm3-ft-prereg.md`, amendment 8); §7, flexible weights, is closed without running
(`docs/weights-prereg.md`, amendment 1); s6-sleep v4 goes on to its check K1 (decided, not run
yet), and s6-ear v3's pre-registration is being frozen (nothing trained yet). The owner then had
the local data that nothing open reads removed: section 8 lists what was deleted and what was
kept. The rows that these changes contradicted are corrected and marked with the date; the
sizes are still those read on 18/11 and 19/11, not read again after the deletion.

**Update of 19/11/2026, evening (s6-sleep v4 closed).** s6-sleep v4's check K1 ran and failed, so v4
stopped at K1 and is closed, with no model (`docs/s6-sleep4-prereg.md` §16). Nothing open reads
its data any more. The rows below that held data for v4 say so, marked with the date; under
"What becomes removable when those two lines end" (section 8), v4's part now applies. Whether
and when that data goes is the owner's decision: nothing was deleted, moved or uploaded for this
update.

**Update of 19/11/2026, last (s6-ear v3 frozen).** s6-ear v3's pre-registration (`docs/s6-ear3-prereg.md`)
was frozen on 19/11/2026, with amendments 1 and 2. Access to the gated otoSpeech set was granted;
v3's data build is running (its CPU stages); nothing has been trained and no test has been read.
The rows below that said the pre-registration was "being frozen" are corrected; the paragraph
above, "later that day", keeps the state it was written in.

**Update of 20/11/2026 (s6-ear v3 read).** s6-ear v3 ran on 19/11/2026 and its gate was read
once at 23:39: it failed (`docs/s6-ear3-prereg.md` §17), and its file is published as a result,
`s6-ear3-nano`. The rows below that said its data build was running are corrected, marked with the
date; nothing was deleted, moved or uploaded for this update, and what its data becomes is the
owner's decision.

**Update of 20/11/2026, last (every line closed; the two later deletion passes).** With s6-ear
v3's result merged, every experimental line of the project is closed: nothing trains, nothing is
queued, no pre-registration is open and no diagnostic is owed. The owner's rule of 19/11/2026
(local data that no open pre-registration, queued job or owed diagnostic reads is removed, whether
or not it is on the Hub) was applied twice more: on the evening of 19/11/2026, after s6-sleep v4
closed and while s6-ear v3 trained (section 9; nothing s6-ear v3 read was touched), and on
20/11/2026, after s6-ear v3's result was merged (section 10). Section 11 lists what is kept and
why, and says plainly what is lost. The rows of sections 1 to 7 and "Where things are" that the
two passes make stale carry a dated note; the statements of 18/11 and 19/11 keep their text.

**Update of 20/11/2026, after the last publication (G7 published, `data/` deleted).** The owner
answered section 7's question 3 on 20/11/2026: everything of G7 is published, the benchmark's test
split included, marked evaluation only on the card. `scripts/hf-publish-data.py` at commit
`fb368b1c` added `typed_decisions_train` and `typed_decisions_bench` to `maxgfr/skycmd-decisions`
and rebuilt `maxgfr/skycmd-glm-labels` with their label rows; every published row was downloaded
again and checked against the local files, and then the whole `skycmd/data/` folder and `target/`
were deleted (section 12). **After this pass, nothing the project wrote exists only on this disk
except what section 11 lists outside G7:** the raw trace of s6-ear v3's gate
(`skycmd-data/s6-ear3-gate`), s6-route's `eval/` and manifests, the run logs, the research dossiers
and the small text logs of the main checkout's `runs/`. The rows that this pass makes stale carry a
note marked *20/11, after the publication*; the earlier statements, those of 20/11 included, keep
their text.

**The owner's rule (18/11/2026). Once a model is in its final version and nobody will touch it
again, and its training data is on Hugging Face, the local copy of that data may be deleted. The
data must be on Hugging Face first.

**Words used below.**

- *Final*: the model is published (passed or failed) and no open pre-registration, queued job or
  owed diagnostic reads its data or its checkpoint. A model whose checkpoint is the starting
  point of a later version is not final in this sense, even if the model itself will not change.
- *On Hugging Face*: a data set under `maxgfr` whose `MANIFEST.json` gives the sha256 of the
  local file it was built from, and that sha256 equals the local file's today (section 2).
  "Upstream on Hugging Face" means only that the source set exists on the Hub under its own
  author; it is not the same thing, and the owner has not said whether it satisfies the rule
  (section 7, question 1).
- *Removable*: removable under the owner's rule, nothing else. Whether to remove it is the
  owner's decision.
- *Checkpoint*: a training state (`model.safetensors`, `optimizer.safetensors`, `state.json`).
  The exported GGUF files are not checkpoints; they are published with each model.

**How the numbers were read.**

- Sizes: `du -sk` on 18/11/2026 between 14:48 and 15:30 CET, given in GiB (KiB / 1024²) or in
  MB (10⁶ bytes) where the file listing gives bytes. Free space on the data volume at 15:30:
  18 GiB of 460 GiB (`df -h`).
- Sizes of sections 9 and 10: `du -sk` read just before each deletion, given in MB of 1024 KB.
  Sizes of section 11: `du -sk` on 20/11/2026 after the second pass, in MB of 1024 KB, and for single
  files their byte count in MB of 10⁶ bytes (`ls -l`), as said where it is used.
- Hugging Face: the Hub API (`list_datasets(author="maxgfr")`, `dataset_info`, `list_models`)
  and the datasets server (`/is-valid`, `/size`), read on 18/11/2026. Only `MANIFEST.json`,
  `README.md` and `agreement.json` files were downloaded; no data shard.
- Local hashes: `sha256` of every local file that a published manifest names (section 2).
- Who reads what: the pre-registrations in `docs/`, the run scripts in `scripts/`, `TODO.md`
  ("Where we are", "What remains") and `DATA_LICENSES.md`, at commit `6b6ecac`. `TODO.md`, the
  project's to-do file, was removed on 19/11/2026: each mention of it below is its text at commit
  `6b6ecac` (its last version is at commit `0e45b7b`).

**Where things are.**

| folder | what | size (18/11/2026) |
|---|---|---:|
| `skycmd/data/` | the main checkout's data, linked into the worktrees (the `data` folder of a second working tree of the repository is a symlink to it) | 47.77 GiB |
| `skycmd-data/s6-ear2/` | s6-ear v2's feature stores and v2's EMA checkpoint | 12.67 GiB |
| `skycmd-data/s6-cast2/` | s6-cast v2's new series and the Chronos-ZS caches; **Deleted on 19/11/2026** (section 8) | 7.20 GiB |
| `skycmd-data/s6-secrets2/` | s6-secrets v2 and v3's CredData and issue-report sets (real leaked credentials; not opened for this inventory); **Deleted on 19/11/2026** (section 8) | 3.97 GiB |
| `skycmd-data/s6-route/` | the tool-routing sources, builds and evaluation outputs; all but `eval/` and the manifests **deleted on 19/11/2026** (section 8) | 2.86 GiB |
| `skycmd/runs/` | runs of the main checkout: checkpoints, tokenized caches, logs | 12.57 GiB |
| `runs/` of a second working tree of the repository | its runs (some entries are symlinks to `skycmd/runs/`) | 0.83 GiB |
| `skycmd/models/`, `models/` of a second working tree of the repository | exported GGUF files | 18 MB, 9 MB |

**20/11/2026.** The table above is the state of 18/11. After the passes of sections 9 and 10,
`skycmd/data/` holds only the files of G7 (section 11); `skycmd-data/s6-ear2/` was deleted on
20/11, as was `skycmd-data/s6-ear3/` (v3's stores, written on 19/11, after this table); the main
checkout's `runs/` holds only small text logs, JSON files and tracked result files (49.6 MB); the
second working tree's `runs/` and `models/` lost every untracked checkpoint and export on 20/11.
`skycmd-data/` now holds `s6-route/` (its `eval/` and manifests), `s6-ear3-gate/`, `run-logs/` and
`ultrasearch/` (section 11).

*20/11, after the publication:* `skycmd/data/` no longer exists: it was deleted whole once the
G7 files were on the Hub and checked (section 12). The main checkout is the only working tree of
the repository: the last other local working copy (the session's working tree of section 10) was
removed on 20/11 (section 12).
`skycmd-data/` is unchanged.

## 1. Status of the models

From `TODO.md` at commit `6b6ecac` ("Where we are", 18/11/2026) and the Hub (`list_models(author="maxgfr")`: 18 model
repositories, each holding its GGUF and card; none holds a training checkpoint).

| model | published | final? | why |
|---|---|---|---|
| s1-pico, s2-nano, s2-nano-doom, s3-pico, s4-micro, s5-lite | yes | yes | no open item reads them; `docs/weights-prereg.md` §2 moved its base to s6-nano |
| s6 recipe (tranche 1: `runs/s6ab`, `runs/s6lr`) | not a model | yes | decided 12/11 (`runs/recipe/recipe.json`) |
| s6-nano | yes, passed | **no** | `runs/s6-nano-g3/decide/last` is the base of s6-sleep v4 (`scripts/train-sleep4.sh`, `NANO_BASE`; K3 failed on 18/11, and K1 failed on 19/11: v4 is closed). **19/11:** §7, its other reader, was closed without running (`docs/weights-prereg.md`, amendment 1), so `runs/s6-nano/pretrain/init` (§7.2) has no reader left. **20/11:** final; its checkpoints and data were deleted on the evening of 19/11 (section 9) |
| s6-micro (v2) | yes, passed | yes for its checkpoints; its data **no**; *20/11:* yes for both | its data (`data/decisions`, `distill`, `games`, `numeric`) was read by s6-sleep v4's K1, which closed v4 on 19/11 (§7, the other reader, was closed on 19/11); deleted on the evening of 19/11, except G7 (section 9); *20/11, after the publication:* G7 is on the Hub and its files were deleted (section 12) |
| s6-micro-v1 | yes, failed | yes, since 19/11 | `runs/s6-micro/pretrain/init` was the initialisation of s6-route v4, v5 and v7 (`docs/s6-route4-prereg.md` §3, "as v4" in `docs/s6-route5-prereg.md` §5; `scripts/train-route4.sh` line 81); the route line is closed |
| s6-secrets-pico | yes, passed | yes, since 19/11 | §7.4, its only reader, was closed without running on 19/11 (`docs/weights-prereg.md`, amendment 1) |
| s6-secrets-verify-pico (v2) and v3 | v2 yes, failed; v3 failed | yes, by the lead's decision of 18/11 | closed as "not worth pre-registering" (`docs/s6-secrets4-diagnosis.md` §5, the note of 19/11/2026); its data was deleted on 19/11 (section 8) |
| s6-ear-nano (v1) | yes, failed | yes for its features (deleted on 19/11, section 8); **no** for its Smart Turn test WAVs and two index files | v2's evaluation and v3 (it ran and failed its gate on 19/11, *corrected on 20/11*) read `data/ear/wav/smart-turn`, `data/ear/clips/smart-turn.jsonl`, `data/ear/jev/turn_end.jsonl` (`scripts/train-turn.sh`, `docs/s6-ear3-prereg.md` §3.1, §3.4, §3.8); *20/11:* final, and the three files were deleted with `data/ear` (section 10) |
| s6-ear2-nano (v2) | yes, passed | **no**; *20/11:* yes | v3 (`docs/s6-ear3-prereg.md`) started from v2's EMA checkpoint and reused v2's stores (§3.1, §3.8); *corrected on 20/11:* v3 ran on 19/11 and failed its gate, so nothing open trains from them now; the stores and the EMA checkpoint were deleted on 20/11 (section 10) |
| s6-ear3-nano (v3, 19/11) | yes, failed | yes | nothing builds on it (`docs/s6-ear3-prereg.md` §17); its stores and checkpoints were deleted on 20/11; the raw trace of its gate is kept (section 11) |
| s6-cast-pico (v1) | yes, failed | yes | by §11 of `docs/s6-cast2-prereg.md` the s6-cast line stops; its local data was deleted on 19/11 (section 8) |
| s6-cast2-pico (v2) | yes, failed (the Hub repository `maxgfr/s6-cast2-pico` holds its GGUF since 18/11 12:47 UTC, same sha256 as `site/public/models/s6-cast2-pico.gguf`; `TODO.md` at commit `6b6ecac`, the morning's state, said "not uploaded yet") | yes, with one reservation | the coverage and front-end diagnostics of §9 are "not scheduled" (`TODO.md` at commit `6b6ecac`, item 1); rebuilding them needs GIFT-Eval's windows again, whose local copy was deleted on 19/11 (section 8) |
| s6-route-nano (v2), s6-route (v3) | yes (failed, passed) | yes, since 19/11 | the route line is closed (v5 and v6 stopped before training, v7 failed, no v8); their records are skycmd-route; the local builds were deleted on 19/11 (section 8) |
| s6-route v4 | no model | yes, since 19/11 | the route line is closed; its 13,687 teacher rewrites are in skycmd-route7 (G12) |
| s6-route v5, v6 (19/11) | no model: each stopped at its data checks or its stop rule (18/11) | yes | v7 took v6's build as its inputs, checked by hash (`runs/s6-route7/inputs.json`) |
| s6-route7-nano (v7, 19/11) | yes, failed | yes | the route line is closed: no v8 (`docs/s6-route7-prereg.md` §11 and the result block); its data is on the Hub as skycmd-route7, uploaded on 19/11 (3.6) |
| s6-sleep v1, v2 | no model (stopped) | **no** for v1's sequences; *20/11:* yes | `runs/s6-sleep/seq` is the default `SEQ` of v3 and v4 (`scripts/train-sleep3.sh`, `scripts/train-sleep4.sh` line 60); v4 closed on 19/11 and the sequences were deleted that evening (section 9) |
| s6-sleep3-nano (v3) | yes, failed | **no**; *20/11:* yes | v4 cites its run folder (`docs/s6-sleep4-prereg.md`, lines 42 and 100); v4 is closed, and v3's checkpoints were deleted on 20/11 (section 10; its small result files are in git) |
| s6-sleep v4 | no model: K3 failed (18/11); K1 failed (19/11, met on one seed of two; `docs/s6-sleep4-prereg.md` §16), so v4 is closed | **no**; *20/11:* yes | closed: nothing further runs (amendment 2); its checkpoints were deleted on 20/11 (section 10; K3's and K1's small files are in git) |
| LFM3 S1/S2 router | not a trained model (failed) | yes, since 19/11 | the fine-tune, the only reader of `runs/lfm3-s1s2`, was closed on 19/11 |
| LFM3 fine-tune | closed on 19/11, nothing trained (`docs/lfm3-ft-prereg.md`, amendment 8) | yes | nothing reads `data/lfm3` any more |
| §7 flexible weights | closed without running on 19/11 (`docs/weights-prereg.md`, amendment 1) | yes | nothing was computed; nothing reads its inputs |

## 2. What is on Hugging Face, and whether it matches the disk

15 public data sets under `maxgfr` (no private one), read on 18/11/2026. The datasets server
answers `viewer: true` (and preview, search, filter, statistics) for all 15. For each set with a
`MANIFEST.json`, every source file it names was hashed on disk.

| data set | Hub commit | files on the Hub | MB | built at | local source | manifest check |
|---|---|---:|---:|---|---|---|
| [skycmd-decisions](https://huggingface.co/datasets/maxgfr/skycmd-decisions) | `a8f33fd2` | 30 | 182.9 | `6cbcce8c` | `data/decisions/` | 27 of 27 identical; `typed_decisions_train` and `typed_decisions_bench` deliberately left out (section 4, G7). *20/11, after the publication:* Hub revision `36704fe4007b9ee27babe7cc35d917f742bfeaff`, 32 files (29 parquet), 831,387 rows, manifest at `fb368b1c`; `typed_decisions_train` (6,000 rows) and `typed_decisions_bench` (2,000) added, the other 27 configs carried from the manifest of `6cbcce8c` (section 12) |
| [skycmd-games](https://huggingface.co/datasets/maxgfr/skycmd-games) | `fa4af24c` | 7 | 6.0 | `6cbcce8c` | `data/games/` | 4 of 4 identical |
| [skycmd-numeric](https://huggingface.co/datasets/maxgfr/skycmd-numeric) | `9ddcd59e` | 4 | 2.5 | `6cbcce8c` | `data/numeric/` | 1 of 1 identical |
| [skycmd-secrets](https://huggingface.co/datasets/maxgfr/skycmd-secrets) | `411d56ac` | 5 | 10.0 | `6cbcce8c` | `data/secrets/` | 2 of 2 identical |
| [skycmd-commands](https://huggingface.co/datasets/maxgfr/skycmd-commands) | `0ff68475` | 4 | 0.6 | `6cbcce8c` | `data/commands/` | 1 of 1 identical |
| [skycmd-zero-shot](https://huggingface.co/datasets/maxgfr/skycmd-zero-shot) | `3865516c` | 6 | 0.2 | `6cbcce8c` | `data/s4/heldout/` | 3 of 3 identical |
| [skycmd-series](https://huggingface.co/datasets/maxgfr/skycmd-series) | `6c6bd6fe` | 11 | 52.8 | `84c6d854` | `data/series/` | 8 of 8 identical; s6-cast v2's USCRN series are not in it (G5) |
| [skycmd-text](https://huggingface.co/datasets/maxgfr/skycmd-text) | `efdf4e69` | 17 | 96.5 | `84c6d854` | `data/text/`, `data/text-open/`, `data/tok/` | 14 of 14 identical (11 text files, 3 tokenizers) |
| [skycmd-math](https://huggingface.co/datasets/maxgfr/skycmd-math) (named `skycmd-lfm3-math` until 19/11/2026; the old address redirects) | `dc89f289` | 5 | 0.2 | `84c6d854` | `data/lfm3/` | 2 of 2 identical |
| [skycmd-ear](https://huggingface.co/datasets/maxgfr/skycmd-ear) | `0b3deda1` | 11 | 42.7 | `84c6d854` | `data/ear/jev/`, `clips/`, `mimi/` | 8 of 8 identical; the audio features and WAVs are not in it (G1, G2) |
| [skycmd-cast-synth](https://huggingface.co/datasets/maxgfr/skycmd-cast-synth) | `20984f3e` | 8 | 78.2 | `8cc8e276` | `data/cast-synth/` | 5 of 5 identical |
| [skycmd-evals](https://huggingface.co/datasets/maxgfr/skycmd-evals) | `d04e8159` | 6 | 0.0 | `8cc8e276` | `eval/s5/` (in git) | 3 of 3 identical |
| [skycmd-trajectories](https://huggingface.co/datasets/maxgfr/skycmd-trajectories) | `1d8524c7` | 14 | 17.0 | `bd5a64a1` | `data/trajectories/` | 11 of 11 identical |
| [skycmd-route](https://huggingface.co/datasets/maxgfr/skycmd-route) | `dab1725b` | 34 | 27.3 | `2eb6c9b6` | `skycmd-data/s6-route/` | 30 of 31 identical; the 31st, `scripts/route/pins.json`, equals the file in git at `2eb6c9b6`; the file in git today differs because v4 added its `v4_sources` entry (expected, not a data change) |
| [skycmd-glm-labels](https://huggingface.co/datasets/maxgfr/skycmd-glm-labels) | `d9520508` | 5 | 129.4 | (no manifest) | `data/teacher/`, `data/distill/` | no sha256 to compare: `agreement.json` equals the local one after the publish filter; `labels` has 741,437 rows and `agree` 556,175, the counts of the local files without the two LocalLLaMA sources (G7). *20/11, after the publication:* rebuilt whole, Hub revision `486a2508836d5b582227bd652333adb403262ed5`, 5 files: `labels` 747,077 rows and `agree` 560,238, the counts of the local files with the G7 rows, and `agreement.json` with 470 sources (section 12) |

**Result: no mismatch.** 119 local files are byte-identical to the files the published sets were
built from. Nothing a manifest names is missing from disk (the main checkout lacks
`scripts/route/`, because it is behind `origin/main`; the session worktrees have it).

**20/11/2026.** That check was of 18/11. After the deletions of sections 8 to 10, the local source
files of these sets are gone, except those of skycmd-glm-labels (`data/teacher/glm_labels.jsonl`,
`data/distill/glm.jsonl`) and of skycmd-evals (in git): the published sets are now the copy.
*20/11, after the publication:* those two GLM files were deleted with `data/` once the rebuilt
skycmd-glm-labels was checked row by row against them (section 12); only skycmd-evals keeps a local
source, in git.

## 3. Each model family: its data, where it is, whether the local copy is removable

"Removable" columns answer the owner's rule only. "After X" names what must end first.

### 3.1 s1 to s5 (published before the s6 models)

| model | status | training data (val, test) | on Hugging Face | local path, size | removable | how to rebuild |
|---|---|---|---|---|---|---|
| s1-pico, s2-nano, s3-pico | final | `data/decisions` (with the GLM files `glm_orders`, `paraphrased`), `data/games`, `data/numeric`, MLM on `data/text` (`DATA_LICENSES.md`) | skycmd-decisions, -games, -numeric, -text; the two GLM files are not on the Hub | the GLM files are no longer on disk (not in `data/decisions/` on 18/11) | yes for what is on disk (the shared files are kept for the s6 models, 3.2); *20/11:* deleted on the evening of 19/11, except G7 (section 9); *20/11, after the publication:* G7 published and deleted (section 12) | `sky-data fetch-decisions`, `sky-games-gen`, `sky-train s2-numeric`, `sky-data fetch-text` (`scripts/prepare-data.sh`); the GLM files need new teacher calls |
| s2-nano-doom | final | the above and DAgger rounds `data/dagger-doom/` | — | `data/dagger-doom/` is not on disk | — | `scripts/train-doom.sh` (needs s2-nano's checkpoint, not on disk) |
| s4-micro | final | `data/s4/` (`queue/`, `gold/`, `heldout/`; the Qwen3.5 teacher's labels) | skycmd-zero-shot (`heldout/` only) | only `data/s4/heldout/` is on disk (1.9 MB) | `heldout/` is read by s6-sleep v4 (`scripts/train-sleep4.sh` line 517): after s6-sleep v4 (kept on 19/11); *20/11:* `data/s4` was deleted on the evening of 19/11 (section 9; skycmd-zero-shot holds it) | `sky-data s4-corpus`; the teacher labels need the teacher again |
| s5-lite | final | `data/s5/` (`sky-data s5-build`, pinned sources of `DATA_LICENSES.md`) | skycmd-evals (hand-written material only) | `data/s5/` is not on disk; `models-src/` 39 MB | — | `cargo xtask s5 fetch-models` / `fetch-evals`, then `sky-data s5-build` (pinned commits) |

The checkpoints of s1 to s5 are not on disk (`docs/weights-prereg.md` §2: "the floating
checkpoint of s3-pico no longer exists"). Their GGUF files are on the Hub.

### 3.2 The s6 decision models and what builds on them

s6-nano and s6-micro train on `data/decisions`, `data/distill`, `data/games` and `data/numeric`
(train splits), pretrain on `data/text-open` then `data/text`, and add three DAgger rounds of
oracle-labelled game states (`scripts/train-s6.sh` lines 77 and 190–193). The tests are the test
splits of `data/decisions` and `data/s4/heldout` (`docs/s6-gen-prereg.md`).

| model | status | training data (val, test) | on Hugging Face (repository, config) | local path, size | removable | how to rebuild |
|---|---|---|---|---|---|---|
| s6-nano | not final (its checkpoint is a base); *20/11:* final | decisions, distill, games, numeric; text-open, text; DAgger `data/dagger-s6-nano/r1..r3` | skycmd-decisions (27 configs; *20/11, after the publication:* 29, with `typed_decisions_train` and `typed_decisions_bench`, section 12), skycmd-glm-labels (`agree`, `labels`), skycmd-games, skycmd-numeric, skycmd-text; **DAgger rounds not on the Hub** (G8) | `data/decisions` 0.84 GiB, `distill` 0.52, `teacher` 0.09, `games` 0.10, `numeric` 0.02, `text` 0.14, `text-open` 0.09, `tok` < 0.01, `dagger-s6-nano` 0.06 | no: s6-sleep v4's K1 read them (§7, the other reader, was closed on 19/11); kept on 19/11; v4 closed on 19/11 (evening). *20/11:* all deleted on the evening of 19/11 (section 9) except the files of G7 (`typed_decisions_*`, `teacher/glm_labels.jsonl`, `distill/glm.jsonl`), kept (section 11). *20/11, after the publication:* those went too, once on the Hub and checked (section 12) | `scripts/prepare-data.sh`; `sky-data teacher mix-labels` (default `--alpha 0.5`) rebuilds `data/distill/glm.jsonl` from `data/teacher/glm_labels.jsonl` and `data/decisions`; the DAgger rounds need the round checkpoints `runs/s6-nano-g1`, `g2` (`sky-arena --dagger`). *20/11:* those checkpoints were deleted on the evening of 19/11; `sky-arena --dagger` plays the round's GGUF file, and those files are kept (section 11, G8) |
| s6-micro v1 | final (since 19/11 for `runs/s6-micro/pretrain/init`: the route line is closed) | as s6-nano; DAgger `data/dagger-s6-micro/r1..r3` (14/11) | as s6-nano; **DAgger rounds not on the Hub** (G8) | `data/dagger-s6-micro` 0.06 GiB; *20/11:* deleted on the evening of 19/11 (section 9) | the DAgger rounds after an upload; deleted without one | as s6-nano |
| s6-micro (v2) | final as a model | as s6-nano; DAgger `runs/s6-micro2/data/dagger-s6-micro/r1..r3` (16/11; different files from v1's, checked by sha256) | as s6-nano; **DAgger rounds not on the Hub** (G8) | `runs/s6-micro2/data/` 0.06 GiB; *20/11:* deleted on the evening of 19/11 (section 9) | after an upload; deleted without one | as s6-nano (`SKY_OUT=runs/s6-micro2/`, `runs/s6-micro2/env.sh`); its round files are gone too (section 11, G8) |
| s6-sleep v1, v2 | stopped | `runs/s6-sleep/seq/` (`sky-train s2-seqgen --seed 61`: recall, rule110, flappy, rescue, dodge, doom; `scripts/train-sleep.sh` lines 105–109); stateless records as s6-nano | **sequences not on the Hub** (G9) | `runs/s6-sleep/seq` 0.48 GiB; *20/11:* deleted on the evening of 19/11 (section 9) | no: v3 and v4 read it; *20/11:* both closed, deleted without an upload | `s2-seqgen` with the seeds of `scripts/train-sleep.sh` (generated by our code; CPU) |
| s6-sleep3-nano (v3) | not final (v4 cites it); *20/11:* final | v1's sequences, the stateless records, base s6-nano | as above | `runs/s6-sleep3/` (worktree) 0.36 GiB; *20/11:* its checkpoints deleted on 20/11 (section 10) | no, until v4's result, which came on 19/11 (v4 closed) | — |
| s6-sleep v4 | closed: K3 failed (18/11), K1 failed (19/11) | v1's sequences; `data/decisions`, `distill`, `games`, `numeric`; tests `typed_decisions_bench`, `s4/heldout`, `when2call` (`scripts/train-sleep4.sh` lines 104, 266–268, 517) | as above | as above | no; *20/11:* v4 is closed and its inputs deleted (sections 9 and 10), except G7; *20/11, after the publication:* G7 too, after its upload (section 12) | — |
| §7 flexible weights | closed without running (19/11) | would have read: base `runs/s6-nano-g3/decide/last`; §7.2 from `runs/s6-nano/pretrain/init` with the s6 records; §7.4 copies of s6-secrets-pico on `data/secrets`; nothing was computed | as above, skycmd-secrets | as above | no longer held by §7 | — |

### 3.3 s6-secrets

| model | status | training data (val, test) | on Hugging Face | local path, size | removable | how to rebuild |
|---|---|---|---|---|---|---|
| s6-secrets-pico | final since 19/11 (§7.4 closed) | `data/secrets/secret_synth` (train, val), `secret_bench` (test); MLM on text-open and text | skycmd-secrets (`secret_synth`, `secret_bench`), skycmd-text | `data/secrets` 0.09 GiB | yes under the rule since 19/11 (skycmd-secrets, 2 of 2 identical); not deleted on 19/11; *20/11:* deleted on the evening of 19/11 (section 9) | `sky-data secret-jev` (seed 1234; our generator) |
| s6-secrets-verify-pico (v2), v3 | final by the lead's decision (18/11) | CredData obfuscated files (116 permissively licensed repositories for train and val; 221 others for test only), the Zenodo issue-report set (secrets replaced by shapes in train and val), synthetic candidates; the obfuscated CredData files are also 30% of the MLM text (`DATA_LICENSES.md`) | **never published**: real-credential sets, "not ours to redistribute" (owner, 16/11; `DATA_LICENSES.md`, s6-secrets v2) | **Deleted on 19/11/2026** (section 8): `skycmd-data/s6-secrets2` 3.97 GiB; the tokenized MLM caches `runs/s6-secrets-verify-pico/pretrain/cache` and `runs/secrets2-smoke/runs/s6-secrets-verify-pico/pretrain/cache`, 1.21 GiB each (their `corpus.json` lists `build/pretrain-text/creddata.jsonl` among the inputs) | deleted without an upload, by the owner's decision (section 7, question 5): it can never be on the Hub | CredData at `01e3d135a90f46c9a7965683b24a05d12e136b33` with its `download_data.py` in a Linux container (it clones the 337 source repositories, so a rebuild depends on them still existing); Zenodo 10.5281/zenodo.19622962 (134,502,037 bytes) (`docs/s6-secrets2-prereg.md` §3) |

### 3.4 s6-ear

| model | status | training data (val, test) | on Hugging Face | local path, size | removable | how to rebuild |
|---|---|---|---|---|---|---|
| s6-ear-nano (v1) | final, except three files it shares with v2 and v3; *20/11:* final | log-mel features `data/ear/audio/` (Smart Turn v3-train and v3.2-test, Speech Commands v0.02, MSWC en + fr first archives, LibriSpeech clean) and the Mimi codes; questions `data/ear/jev/`; test WAVs `data/ear/wav/` (for the baselines) | skycmd-ear: the questions (`turn_end`, `keyword`, `keyword_zs`), the clip index (`clips_*`) and the Mimi codes (`mimi_*`); **not the features nor the WAVs** (G1, G2) | **Deleted on 19/11/2026** (section 8): `data/ear/audio` 20.07 GiB (smart-turn 14 GiB, librispeech 2.9, mswc 2.7, speech-commands 0.39; 700,083 files); `data/ear/wav/mswc` 3.66, `wav/speech-commands` 0.15; `data/hf/speech_commands` (the downloaded archive) 2.26. Kept: `jev`, `clips`, `mimi` 0.34. *20/11:* the rest of `data/ear` (with `wav/smart-turn` and the index files) was deleted on 20/11 (section 10) | the features and v1-only WAVs: deleted on 19/11 without an upload, by the owner's decision (section 7, questions 1 and 2); the v1-only index files (`jev/keyword*.jsonl`, `clips/mswc.jsonl`, `clips/speech-commands.jsonl`, `mimi/`, 0.20 GiB): yes | `sky-data fetch-audio --set all` then `sky-data audio-jev` (`scripts/train-ear.sh` `data`); sources read at `main`, not pinned (section 4, G1) |
| s6-ear2-nano (v2) | not final (v3 builds on it); *20/11:* final (v3 ran and failed on 19/11) | `skycmd-data/s6-ear2/train/` (from `pipecat-ai/smart-turn-data-v3.2-train` at `f8c95d28`, 83 shards; 257,353 train / 13,593 val, `train/LEAK.json`): features 9.07 GB, the Smart Turn v3.2 teacher's targets 2.84 GB, manifests 0.06 GB; test: `test/` (from v1's WAVs of `smart-turn-data-v3.2-test`), `eotbench/store` (`livekit/eot-bench-data` at `ae834af9`) | **none of the stores** (G3) | `skycmd-data/s6-ear2` 12.67 GiB (train 11.14, test 0.99, eotbench 0.51, checkpoints 0.02); and from v1: `data/ear/wav/smart-turn` 6.79 GiB, `clips/smart-turn.jsonl` 59 MB, `jev/turn_end.jsonl` 92 MB; *20/11:* all deleted on 20/11, with v2's EMA checkpoint (section 10) | no: v3 reuses them (`docs/s6-ear3-prereg.md` §3.8: "Reused, not copied: v2's Smart Turn store (11 GB), its test store and v1's test WAVs"); *20/11:* v3 is closed, and they were deleted without an upload | `scripts/train-turn.sh data` (`sky-data fetch-turn`, `turn-test`, `turn-eotbench`, `scripts/turn-teacher.py`): about 41.4 GB to stream; on 16/11 the first feature shard was written at 01:52 and the last teacher shard at 05:24 |
| s6-ear3-nano (v3) | pre-registration frozen on 19/11/2026 with amendments 1 to 3 (the owner accepted otoSpeech's access conditions on 19/11 and access was granted); *corrected on 20/11:* trained and read on 19/11, gate failed (`docs/s6-ear3-prereg.md` §17), its file published as a result | v2's stores, v1's test WAVs, new conversation records from `otoearth/otoSpeech-full-duplex-280h` (gated, CC-BY-4.0), eot-bench | *20/11:* none of its stores: the records derived from otoSpeech are never published (`DATA_LICENSES.md`, s6-ear v3) | *20/11:* `skycmd-data/s6-ear3` 13,156 MB (the conversation stores, the extended Smart Turn stores, the eot-bench store, the teacher's opinions), **deleted on 20/11** (section 10); the gate's raw trace `skycmd-data/s6-ear3-gate` is kept (section 11) | *20/11:* deleted without an upload (it may never be on the Hub) | *20/11:* the data stages of `scripts/train-turn3.sh` (`conv-data`, `st-extend`, `eotbench-data`, `teacher-conv`) from otoSpeech at `4b3c57b8`, under its access conditions, once v2's stores and v1's test WAVs are rebuilt |

### 3.5 s6-cast

| model | status | training data (val, test) | on Hugging Face | local path, size | removable | how to rebuild |
|---|---|---|---|---|---|---|
| s6-cast-pico (v1) | final | `data/series/` (8 sources, one series in ten held out for validation), synthetic series drawn during training; test: GIFT-Eval (`data/gift-eval`), Chronos-ZS | skycmd-series (8 configs), skycmd-cast-synth (a fixed sample of the generators); GIFT-Eval never (G4) | `data/series` 0.44 GiB and `data/cast-synth` 0.29: **Deleted on 19/11/2026** (section 8); raw parquet `data/hf/autogluon__chronos_datasets` 1.34 (*20/11:* no longer on disk, section 11, G15) | yes: `data/series`, `data/cast-synth` (deleted on 19/11) and the raw parquet (the series derived from it are on the Hub, identical) | `sky-data fetch-series`, `sky-train cast-synth --games` |
| s6-cast2-pico (v2) | final (one reservation, section 1) | v1's series and USCRN sub-hourly (`skycmd-data/s6-cast2/series2-checked/uscrn_subhourly.jsonl`, 1,011 series, 8,282,112 points); TinyCast's forecasts as a teacher, computed during training and never stored; tests GIFT-Eval and Chronos-ZS | v1's series yes; **USCRN not on the Hub** (G5) | **Deleted on 19/11/2026** (section 8), all of it: `series2-checked` 41 MB and `series2` 41 MB (the same file: both sha256 `69d9dbc1…3dd8`, as `DATA_LICENSES.md` records); test caches: `data/gift-eval` 7.01 GiB, `data/gift-eval-src` 1.49, `skycmd-data/s6-cast2/hf` 6.35, `chronos-zs` 0.48, `chronos-zs-out` 0.30 | deleted on 19/11 (the s6-cast line is closed): USCRN without an upload, rebuilt from NOAA by the command at right; the test caches are never published (G4, G6) | `python3 -I scripts/cast2-data.py fetch --source uscrn_subhourly` (NOAA NCEI, 2025 station files, 2,256,400,800 bytes); `scripts/gift-eval.py download` / `export`; `scripts/chronos-zs.py` |

### 3.6 s6-route

The data folder is `skycmd-data/s6-route/` (`ROUTE_DATA`). Sources are pinned in
`scripts/route/pins.json` and every downloaded file is listed with its sha256 in `MANIFEST.json`
(v2) and `MANIFEST4.json` (v4's Toucan files).

| model | status | training data (val, test) | on Hugging Face | local path, size | removable | how to rebuild |
|---|---|---|---|---|---|---|
| s6-route-nano (v2), s6-route (v3) | final since 19/11 (the route line is closed) | `build/records/*/train.jsonl`, `val.jsonl`; validation slates `build/val/`; tests `build/test/` and the fresh files `build/fresh/`; the teacher's intent descriptions and rewrites | skycmd-route: 24 slate configs and 2 teacher configs, plus `pins.json`, `MANIFEST.json`, `BUILD.json`, `FRESH.json`, the T2-tools clusters; left out by the publish script's rules: `t_toolrouting` (G10), the `*.model.jsonl` copies, the false-negative filter's items and answers (G11), `build/stage`, `build/emb`, `src/`, `models/`, `envs/`, `eval/` | `build/` 0.41 GiB, `src/` 0.84, `models/` 0.10, `envs/` 1.10, `embed/` 0.01: **Deleted on 19/11/2026** (section 8); kept: `eval/` 0.13 (229 MB on 19/11 with v7's dumps) | deleted on 19/11, except `eval/` | `scripts/build-route-data.sh` (584,935,409 bytes of sources, `DATA_LICENSES.md`), `scripts/route-teacher.sh` (new teacher calls), `sky-data --seed 1235 route fresh` |
| s6-route v4 | stopped at its data checks | `build4/` (records 202 MB, stage 49 MB, val 18 MB, teacher 3.5 MB: 13,687 rewrites by GLM-5.3-Flash, 685 calls); Toucan SFT for the fresh test (`src/toucan`, 425,496,735 bytes) | **nothing of v4** (G12) | `build4/` 0.27 GiB; `src/toucan` 0.40: **Deleted on 19/11/2026** (section 8) | deleted on 19/11; its teacher rewrites are in skycmd-route7 (G12) | `scripts/build-route4-data.sh fetch` and build; the rewrites need new teacher calls |
| s6-route v5, v6 (19/11) | stopped before training (18/11) | `build5/`, `build6/`: v5's build, and v6's with its When2Call group rebuilt and its fix `build6/fix/` | nothing of v5 or v6 as such; the files of `build6/` that v7 read are in skycmd-route7 under their `build7/` names, the same bytes (`runs/s6-route7/inputs6.json`) | `build5/` 0.37 GiB, `build6/` 0.52: **Deleted on 19/11/2026** (section 8) | deleted on 19/11, once skycmd-route7 was on the Hub: they trained nothing, and v7's copy holds every file of theirs that a model read; v6's two When2Call record files and its `counts/v5-slates.json`, which v7 replaced, are not published | `scripts/build-route5-data.sh`, `scripts/build-route6-data.sh` (the false-negative answers need new teacher calls) |
| s6-route7-nano (v7, 19/11) | failed, final | records `build7/records/` (182,085 training, 15,873 validation); validation files `build7/val/` (`v5_tools` and its masked, rewritten and blind copies; the model copies of `v4_tools`, `v_tools`, `r_tools_*`); the fresh test T5 `toucan5/fresh/` (2,000 slates, its probe and blind copies, the servers file; opened and read once on 19/11); the teacher's rewrites `build4/teacher/rewrite-description.jsonl` (13,687) and `toucan5/teacher/rewrite-description.jsonl` (1,448); T2-tools and v2's teacher files from `build/`; initialisation `runs/s6-micro/pretrain/init` | **skycmd-route7** (assembled by `--only route7` and uploaded on 19/11): 35 configs, 250,849 rows, 33.7 MB; every file checked against `runs/s6-route7/inputs6.json`, `test-files.json` and `FRESH-OPEN`, the blind copies and the teacher's files row by row; T2-tools and v2's teacher files by reference to skycmd-route. Left out (`ROUTE7_EXCLUDED`): the false-negative filter's items and answers of v5, v6 and v7 (G16), the plans `stage/plan5.jsonl`, the Toucan stage `toucan5/stage/`, the embeddings `build7/emb/`, the dumps `eval/t5/`, `eval/route7-val/` | `build7/` 0.66 GiB (records 0.21, val 0.03, teacher 0.04, stage 0.09, fix 0.14, fix7 0.14, emb 0.02), `toucan5/` 0.06 (fresh 0.02, stage 0.03), `build4/` 0.01: **Deleted on 19/11/2026** (section 8); kept: `eval/t5` 0.07, `eval/route7-val` 0.01 (in `eval/`, the only raw trace of v7's gate) | deleted on 19/11 after the upload, except `eval/`; the false-negative answers (G16) were not on the Hub and went with `build7/` | `scripts/build-route7-data.sh` over `build6/` (itself `scripts/build-route6-data.sh` over v5's `scripts/build-route5-data.sh` and v4's `scripts/build-route4-data.sh fetch`); T5 by the stage `fresh` (seed 1236); new teacher calls for the rewrites and the false-negative answers |

### 3.7 LFM3-1.7B

| work | status | data | on Hugging Face | local | removable | rebuild |
|---|---|---|---|---|---|---|
| S1/S2 router | failed (inference only) | calibration and test questions `data/lfm3/` | skycmd-math (2 configs, identical) | `data/lfm3` < 0.01 GiB; generations `runs/lfm3-s1s2` 12 MB | yes under the rule since 19/11 (the fine-tune, their only reader, is closed); not deleted on 19/11; *20/11:* `data/lfm3` was deleted on the evening of 19/11 (section 9), and so were the generation files of `runs/lfm3-s1s2` above 2 MB; the folder keeps 6.2 MB (the calibration files, `test.s1.jsonl` and the reports) | `scripts/lfm3-s1s2.sh data` at commit 5011585 (the LFM3 code was removed on 19/11/2026; [`scripts/lfm3-s1s2.sh`](https://github.com/maxgfr/skycmd/blob/5011585/scripts/lfm3-s1s2.sh)) |
| fine-tune | closed on 19/11, nothing trained (`docs/lfm3-ft-prereg.md`, amendment 8) | GSM8K and MATH train solutions, downloaded from the Hub at the pinned revisions of `DATA_LICENSES.md` | — | — | — | — |

## 4. The gap: data on disk that is not on Hugging Face under `maxgfr`

"Licence allows republishing" answers the upstream licence only, from the card or licence text
read on 18/11/2026; "project rule" is the rule of `scripts/hf-publish-data.py` or
`DATA_LICENSES.md`. Upload sizes are the local sizes; a parquet conversion would be smaller for
text and about the same for binary features.

| # | item | size | read by | why not published | upstream, revision, rebuild | licence allows republishing? | upload |
|---|---|---:|---|---|---|---|---:|
| G1 | v1's log-mel features `data/ear/audio/`. **Deleted on 19/11/2026** (section 8) | 20.07 GiB, 700,083 `.npy` files | s6-ear-nano only (training, validation, test); no open item (`grep` of `docs/`, `scripts/`, `crates/`: only v1's scripts name `audio/`; v2 and v3 read WAVs, `crates/sky-infer/src/bin/sky-turn.rs` line 10) | project rule: "the audio itself (31 GB of log-mel features and test WAVs) is not [republished]: `sky-data fetch-audio` rebuilds it" (`scripts/hf-publish-data.py` docstring) | all upstream on the Hub except Speech Commands (TensorFlow archive): `pipecat-ai/smart-turn-data-v3-train` (now `909e604f`, 29.2 GB in all, last changed 29/09/2026), `pipecat-ai/smart-turn-data-v3.2-test` (`dfc17924`, 4.8 GB), `MLCommons/ml_spoken_words` (`0bc9df68`, first archive of each split read), `openslr/librispeech_asr` (`71cacbfb`, `clean` train.100 and validation), `download.tensorflow.org/data/speech_commands_v0.02.tar.gz`; read at `main`, so the bytes of 04/11 are not pinned (every upstream commit above predates 04/11). Rebuild: `sky-data fetch-audio --set all` (time not recorded) | yes: every source is CC-BY-4.0 (cards of the Smart Turn sets: "licensed under the Creative Commons Attribution 4.0 International License"; Hub tags of MSWC, LibriSpeech, Speech Commands), with attribution | 20.07 GiB; the file count is far above what a Hub repository holds well, so it would have to be packed (shards of many clips); no command exists |
| G2 | v1's test WAVs `data/ear/wav/`. **Deleted on 19/11/2026** (section 8): mswc, speech-commands and the archive; smart-turn kept. **smart-turn deleted on 20/11/2026** (section 10) | smart-turn 6.79 GiB, mswc 3.66, speech-commands 0.15; and the archive `data/hf/speech_commands` 2.26 | smart-turn WAVs: v2's evaluation and v3 (test (a), and `turn-extend --test`); mswc and speech-commands: v1 only | as G1 | the test splits above, on the Hub (Speech Commands at the TensorFlow URL) | yes (CC-BY-4.0); but these are copies of public test sets: republishing them adds nothing | — |
| G3 | v2's stores `skycmd-data/s6-ear2/`. **Deleted on 20/11/2026** (section 10), without an upload | train 11.14 GiB (features 9.07 GB, teacher targets 2.84 GB), test 0.99, eotbench 0.51 | v2; v3 reuses all three | `DATA_LICENSES.md`: "no audio is redistributed" | `pipecat-ai/smart-turn-data-v3.2-train` at `f8c95d28` (41.4 GB, 83 shards), `livekit/eot-bench-data` at `ae834af9` (1.98 GB); the teacher is `pipecat-ai/smart-turn-v3` at `f766f81d`. Rebuild: `scripts/train-turn.sh data` (CPU, about 3.5 h on 16/11 from the first feature shard to the last teacher shard) | features: yes (CC-BY-4.0). Teacher targets: unclear; the model is BSD-2-Clause, which does not restrict outputs, but the project chose not to redistribute the model and has no rule on its outputs | train 11.14 GiB; no command exists |
| G4 | GIFT-Eval `data/gift-eval/` (exported test windows) and `data/gift-eval-src/` (the download). **Deleted on 19/11/2026** (section 8) | 7.01 + 1.49 GiB | the GIFT-Eval scores of s6-cast v1 and v2 and of the references; the unscheduled §9 diagnostics | project rule: "an evaluation set never redistributed" (`DATA_LICENSES.md`, `scripts/gift-eval.py`) | `Salesforce/GiftEval`, local download at `30841734` (the `.cache/huggingface` metadata of `data/gift-eval-src`), the Hub's current commit, 1.59 GB; rebuild `uv run --python 3.12 scripts/gift-eval.py download --data data/gift-eval-src` then `export --out data/gift-eval` | no for this project: the card says "This release is for research purposes only in support of an academic paper" next to its Apache-2.0 tag, and the project decided never to redistribute an evaluation set | — |
| G5 | USCRN series `skycmd-data/s6-cast2/series2-checked/uscrn_subhourly.jsonl` (and its identical copy in `series2/`). **Deleted on 19/11/2026** (section 8), without an upload | 41 MB (82 MB with the copy) | s6-cast2-pico training | `DATA_LICENSES.md`: "never committed or redistributed by this repository"; no entry in the publish script | NOAA NCEI, `subhourly01`, the 159 station files of 2025, not pinned by hash (`series2/uscrn_subhourly.manifest.json` lists the fetch); rebuild `scripts/cast2-data.py fetch` then `leak-check` | yes: public domain (U.S. Government work; `DATA_LICENSES.md`, s6-cast v2 table: "publication of the series: allowed") | 41 MB |
| G6 | Chronos-ZS caches `skycmd-data/s6-cast2/hf/` (the `datasets` cache), `chronos-zs/` (exported windows), `chronos-zs-out/` (forecast dumps of four systems). **Deleted on 19/11/2026** (section 8), with `skycmd-data/s6-cast2` | 6.35 + 0.48 + 0.30 GiB | the Chronos-ZS readings of s6-cast v2 (reported, not gated) and the leakage check | evaluation sets | `autogluon/chronos_datasets` at `eeecad0b` (tagged `other`: each series keeps its own licence) and `autogluon/chronos_datasets_extra` at `84737b7` (ETT from GitHub); rebuild `scripts/chronos-zs.py` through `scripts/train-cast2.sh chronos`; the dumps by `sky-cast-eval` | not checked set by set; not needed, the sets are on the Hub | — |
| G7 | `data/decisions/typed_decisions_train.jsonl` (6,000 rows) and `typed_decisions_bench.jsonl` (2,000 rows), and their GLM-5.3-Flash labels: 5,640 rows of `data/teacher/glm_labels.jsonl` and 4,063 rows of `data/distill/glm.jsonl`. **Kept on 20/11/2026** (section 11). *20/11, after the publication:* **published on 20/11/2026** (`typed_decisions_train` and `typed_decisions_bench` in skycmd-decisions, the bench marked evaluation only; the label rows in skycmd-glm-labels), checked row by row, then **deleted** (section 12) | 8.9 MB, and the label rows within the files of 3.2 | every s6 decision model (train, the gated bench); s6-sleep v4 | project rule: "LocalLLaMA/typed-decisions is gated on the Hub" (`EXCLUDED` in `scripts/hf-publish-data.py`) | `LocalLLaMA/typed-decisions`: on 18/11/2026 the Hub API reports it **not gated** (`gated: False`), Apache-2.0, commit `e6aa632c` (16/11, leaderboard edits; s5 pinned `c806f6c6`); the labels exist only here and would need new teacher calls | rows: yes (Apache-2.0); labels: yes (MIT model, owner's decision of 16/11); but the bench is the test split of a public leaderboard, so publishing our copy of it is the owner's call | 8.9 MB and about 1% of the label files |
| G8 | DAgger rounds `data/dagger-s6-nano/r1..r3`, `data/dagger-s6-micro/r1..r3` (v1), `runs/s6-micro2/data/dagger-s6-micro/r1..r3` (v2). **Deleted on the evening of 19/11/2026** (section 9), without an upload | 64 MB each | training of s6-nano, s6-micro v1, s6-micro v2 | no entry in the publish script | ours: game states reached by the models, labelled by the oracle; rebuilding needs the round checkpoints (`sky-arena --dagger`, `scripts/train-s6.sh` line 192) | yes (Apache-2.0, our code and models; no third-party data) | 0.19 GiB |
| G9 | s6-sleep sequences `runs/s6-sleep/seq/` (recall 228 MB, doom 67, rescue 64, rule110 49, dodge 45, flappy 38). **Deleted on the evening of 19/11/2026** (section 9), without an upload | 0.48 GiB | s6-sleep v1, v3, v4 (closed on 19/11) | no entry in the publish script | ours: `sky-train s2-seqgen --seed 61` (`scripts/train-sleep.sh` lines 105–109) | yes (Apache-2.0, generated by our code) | 0.48 GiB |
| G10 | `build/test/t_toolrouting.jsonl` (100 test slates). **Deleted on 19/11/2026** (section 8), with `build/` | 148 kB | s6-route-nano and s6-route tests (reported) | project rule: the card of `stanley-nv/tool-routing-data` "asks users to review the generating model's terms" (`ROUTE_EXCLUDED`) | `stanley-nv/tool-routing-data` at `249f72a6`; `scripts/build-route-data.sh` rebuilds it | unclear (CC-BY-4.0 tag, the card's request) | — |
| G11 | v2's false-negative filter: `build/teacher/false-negatives.items.jsonl` (40 MB) and the teacher's answers `false-negatives.jsonl` (3.3 MB). **Deleted on 19/11/2026** (section 8), with `build/`, without an upload | 43 MB | the hard slates of v2 (their effect is in the published slates) | project rule (`ROUTE_EXCLUDED`): "their effect is in the hard slates" | the answers need new teacher calls; the items `sky-data route split` rebuilds | yes (outputs of an MIT model, owner's decision of 16/11; the tool texts keep their sources' licences) | 3.3 MB (answers) |
| G12 | v4's build `build4/`, including `build4/teacher/rewrite-description.jsonl` (13,687 rewrites, 1.5 MB). **Deleted on 19/11/2026** (section 8); the rewrites are in skycmd-route7 | 0.27 GiB | v5 reused `build4/teacher/` (the route line is closed) | v4 trained nothing; no entry in the publish script. **19/11:** the rewrites are in skycmd-route7 (`teacher_rewrite_description_v4`, checked against the records they went into), and `build4/` is down to its teacher folder (6.7 MB) | the records: `scripts/build-route4-data.sh`; the rewrites: 685 new teacher calls | yes (as G11) | 3.5 MB (`build4/teacher/`) |
| G16 (19/11) | the false-negative filter of s6-route v5, v6 and v7: `build7/teacher/false-negatives.jsonl` (58,984 answers, 2.8 MB), `build7/fix/teacher/` (694), `build7/fix7/teacher/` (296), and their items (36 MB, 0.4 MB, 0.2 MB). **Deleted on 19/11/2026** (section 8), with `build7/`, without an upload | 40 MB | the hard slates of v7's records (their effect is in the published records) | project rule, as G11 (`ROUTE7_EXCLUDED`): "their effect is in the hard slates" | the answers need new teacher calls; the items `sky-data route v5-plan` and `v5-fix-items` rebuild | yes (as G11) | 2.9 MB (answers) |
| G13 | **Deleted on 19/11/2026** (section 8). s6-route's raw sources `src/` (glaive 259 MB, toucan 406 MB, when2call 59, hermes 56, toolace 35, api-bank 28, seal-tools 11, metatool 6.7, clinc, bfcl, tool-routing-data) | 0.84 GiB | v2 to v5 builds | the publish script leaves out raw upstream files | every file pinned and hashed in `MANIFEST.json` / `MANIFEST4.json`; `scripts/build-route-data.sh fetch`, `scripts/build-route4-data.sh fetch` | mostly yes (Apache-2.0, MIT, CC-BY), but republishing pinned upstream files adds nothing | — |
| G14 | s6-secrets2 `skycmd-data/s6-secrets2/` and the two tokenized caches holding its text (section 3.3). **Deleted on 19/11/2026** (section 8) | 3.97 + 2 × 1.21 GiB | s6-secrets-verify-pico v2, v3 | never published: real-credential sets (owner, 16/11) | CredData at `01e3d135…` and Zenodo 19622962 (section 3.3); the rebuild depends on CredData's source repositories still existing | no: per-file licences of third-party repositories, real values in the test files; the project decided never to publish them | — |
| G15 | the raw download cache `data/hf/` other than Speech Commands (tasksource-jev 642 MB, procedural 266 MB, chronos parquet 1.34 GiB, the other decision and text sources; `LocalLLaMA__typed-decisions`, 0.8 MB, is the raw of G7). **No longer on disk on 20/11/2026**, except `LocalLLaMA__typed-decisions` (section 11); *20/11, after the publication:* that one was deleted with `data/` (section 12) | 2.72 GiB | `sky-data fetch-*` only, to write `data/decisions`, `text`, `commands`, `series`, `ear/mimi` | a cache of upstream files | the upstream sets on the Hub (not pinned for `fetch-decisions` and `fetch-commands`; `DATA_LICENSES.md`); `scripts/prepare-data.sh` (its `REFRESH=1` empties `data/hf` itself) | not needed: what the models read, the files derived from this cache, is on the Hub byte for byte (section 2) | — |

Also not on the Hub, and not data: the Python environment `skycmd-data/s6-route/envs/` (1.10 GiB),
the copies of upstream models `skycmd-data/s6-route/models/` (0.10 GiB, pinned in
`pins.json`), the evaluation outputs `skycmd-data/s6-route/eval/` (0.13 GiB), and the
assembled copy of the published sets `target/hf-data/` (0.58 GiB; `scripts/hf-publish-data.py
--dry-run` writes it again). **19/11:** `envs/`, `models/` and the main checkout's `target/`
(with `target/hf-data/`) were deleted; `eval/` is kept (section 8).

**20/11/2026.** Section 11 goes through G1 to G16 one by one and says which still exists on
the disk: only G7 does. *20/11, after the publication:* G7 is on the Hub and its files were
deleted (section 12): none of G1 to G16 is on the disk any more.

## 5. Training checkpoints that exist only on this disk

No checkpoint is on the Hub: the 18 model repositories of `maxgfr` hold GGUF files and cards
only. Sizes are the checkpoint folders (`model.safetensors` and the files next to it) and,
apart, the tokenized pretraining caches (`pretrain/cache`, which the trainer rebuilds from the
MLM text on the CPU; their `corpus.json` lists `data/text` and `data/text-open`, both identical
on the Hub). Totals: about 3.6 GB of checkpoints and 9.36 GiB of caches.

**20/11/2026: none of these exists any more.** Every checkpoint and every tokenized cache of the
main checkout's `runs/` was deleted on the evening of 19/11 (section 9), the `sleep2` working tree
with it, and those of the second working tree's `runs/` on 20/11 (section 10). The answers in the
last column are those of 18/11 and 19/11.

| run folder | checkpoints (MB) | cache | what it is | needed later? |
|---|---:|---:|---|---|
| `runs/s6-nano-g3/decide/last` (and `g1`, `g2`) | 24 each | — | s6-nano's DAgger rounds; `g3` is the published model's checkpoint | **yes**: `g3` is the published model's checkpoint and was the base of s6-sleep v4 (closed on 19/11; §7, its other reader, was closed on 19/11); `g1`, `g2` only to rebuild the DAgger rounds (G8); *deleted on the evening of 19/11* |
| `runs/s6-nano` | 51 | 0.73 GiB | s6-nano's pretraining and first decision phase | no since 19/11: `pretrain/init` (5.7 MB) was kept for §7.2, closed without running; kept on 19/11 with the other `runs/s6-nano*` |
| `runs/s6-micro` | 191 | 0.64 GiB | s6-micro v1's pretraining and decision phase | no since 19/11 (the route line is closed); before, **yes** for `pretrain/init` (21 MB, s6-route v4 and v5); `runs/s6-micro2/runs/s6-micro/pretrain/init` is the same file (same sha256) |
| `runs/s6-micro-g1..g3` | 90 each | — | s6-micro v1's DAgger rounds | no |
| `runs/s6-micro2` | 460 | 0.64 GiB (the same bytes as `runs/s6-micro`'s cache, checked by sha256) | s6-micro v2, the published model | no open item names it |
| `runs/s6-nano-lr`, `runs/s6-micro-lr` | 152, 573 | 2.18, 1.91 GiB | tranche 2's rate sweeps (×0.5, ×1, ×2) | no |
| `runs/s6ab`, `runs/s6lr` | 483, 101 | — | tranche 1's pico ablation and rate pilot (135 and 28 checkpoints) | no (the recipe is decided) |
| `runs/s6-secrets-pico` | 15 | 0.85 GiB | the published model | no: §7.4 was closed on 19/11 |
| `runs/s6-secrets-verify-pico`, `runs/secrets2-smoke` | 15, 15 | 1.21, 1.21 GiB (CredData text inside, G14) | v2 and a smoke run of it | no; the caches must never be published. **Deleted on 19/11/2026** (section 8): `runs/secrets2-smoke` and `runs/s6-secrets-verify-pico/pretrain` |
| `runs/s6-ear-nano`, `runs/s6-ear-nano-swap` | 89, 55 | — | v1 and a run set aside (`TODO.md` at commit `6b6ecac`, item 11) | no |
| `runs/s6-nano-lr-swap` | 36 | — | a run set aside (`TODO.md` at commit `6b6ecac`, item 11) | no |
| `runs/s6-sleep` | 31 | — | v1's pilot (and `seq/`, G9, which is data) | the checkpoints no; `seq/` yes; *both deleted on the evening of 19/11* |
| `runs/s6-cast-pico`, `runs/s6-cast-pilot` | 10, 18 | — | v1 and its pilot | no |
| worktree `runs/s6-ear2-nano` | 51 (`last`, `last-ema`, `pre-qat`) | — | v2; `last-ema` is copied to `skycmd-data/s6-ear2/checkpoints/last-ema/` (the four sha256 equal those of `last-ema.sha256`) | **yes**, the copy in `skycmd-data` (v3's start); the worktree copies are redundant; *all deleted on 20/11* |
| worktree `runs/s6-ear2-pilot` | 69 | — | v2's pilots | no |
| worktree `runs/s6-nano2`, `runs/s6-nano2-pilot` | 134, 170 | — | s6-route-nano's training and pilots (s6-route ships the same decision file) | v4 and v5 start from s6-micro's pretraining init instead (`docs/s6-route4-prereg.md`, option O5): no |
| worktree `runs/s6-sleep3` | 359 (40 checkpoints) | — | v3's pilot, nano runs and auxiliary runs | until v4's result (v4 cites `runs/s6-sleep3/nano/`), which came on 19/11 (v4 closed at K1) |
| worktree `runs/s6-sleep4` | 14 | — | v4's K3 (failed, 18/11) and K1 (failed, 19/11; its small files are in git, as K3's) | no open item reads it since 19/11 (v4 closed); before, **yes** |
| worktree `runs/s6-cast2-pico`, `runs/s6-cast2-pilot`, `runs/cast2-smoke` | 11, 9, 20 | — | v2, its pilot, a smoke prefix | no |
| worktree `sleep2/runs` | 84 | — | s6-sleep v2's checks and v3's pre-runs; also 233 MB of diagnostic sequences (`s6-sleep3/seq-oracle`) | no |

## 6. Proposal

Sizes are those of section 3 to 5. Nothing below had been done on 18/11. **19/11:** the deletions
of section 8 went further than this proposal; each item they removed is marked here. **20/11:**
so did those of sections 9 and 10; what is left is in section 11.

**(a) Removable today under the owner's rule** (the model is final or the data is read by no open
item, and the same bytes are on the Hub, section 2):

| item | size | evidence |
|---|---:|---|
| `data/trajectories/` (deleted on the evening of 19/11) | 0.53 GiB | skycmd-trajectories, 11 of 11 files identical; read by no model's training |
| `data/series/` (deleted on 19/11) | 0.44 GiB | skycmd-series, 8 of 8 identical; s6-cast line stopped |
| `data/cast-synth/` (deleted on 19/11) | 0.29 GiB | skycmd-cast-synth, 5 of 5 identical; no model trains on these rows |
| v1-only index files of `data/ear/`: `jev/keyword.jsonl`, `jev/keyword_zs.jsonl`, `clips/mswc.jsonl`, `clips/speech-commands.jsonl`, `mimi/` (deleted on 20/11 with `data/ear`) | 0.20 GiB | skycmd-ear, identical; only v1 reads them |
| `data/hf/` except `speech_commands/` and `LocalLLaMA__typed-decisions/` (no longer on disk on 20/11, section 11) | 2.72 GiB | a download cache; the files derived from it are on the Hub, identical (G15); the LocalLLaMA raw (0.8 MB) is the source of G7, which is not |
| tokenized caches of the text-only pretrainings: `runs/s6-nano`, `runs/s6-micro`, `runs/s6-micro2`, `runs/s6-nano-lr` (3), `runs/s6-micro-lr` (3), `runs/s6-secrets-pico` (deleted on the evening of 19/11) | 6.95 GiB | derived from `data/text` and `data/text-open` (their `corpus.json`), identical on skycmd-text; rebuilt by the trainer on the CPU |
| `target/hf-data/` (not training data; deleted on 19/11 with `target/`) | 0.58 GiB | the assembly of what is on the Hub |

Total: 11.71 GiB, of which 4.18 GiB of data files, 6.95 GiB of caches and 0.58 GiB of assembly
output. Checkpoints of final models (section 5, "no") are not training data and are not
counted here: about 2.8 GB more, the owner's call.

**How the caches are cleaned** (measured on 09/11/2026; moved here on 19/11/2026 from the removed
`TODO.md`, whose last version is at commit `0e45b7b`). A tokenized pretraining cache is 866 MB per
pretraining run of the s6 text set, the same from one run to the next, so about 43 GB for the fifty
or so runs of tranche 1; a pico checkpoint is under 5 MB. Once a run's steps are finished the caches
serve no result, only a faster reload of a restarted step, and the trainer rebuilds them on the
CPU. To remove them all:

```
find runs -type d -name cache -prune -exec rm -rf {} +
```

The debug build cache `target/s3-train/debug` (9.2 GB on 06/11/2026, written by `cargo test` and
clippy) rebuilds itself and can be deleted after tests; the training scripts use only
`target/s3-train/release`.

**(b) Removable after an upload that the licences allow.** No entry exists for any of these in
`scripts/hf-publish-data.py`; each needs a new `SETS` entry (and, for USCRN, a `SERIES` entry)
before `uv run scripts/hf-publish-data.py --dry-run --only <set>` and the same without
`--dry-run`. Not run.

| item | upload | then removable |
|---|---:|---|
| DAgger rounds (G8) | 0.19 GiB | at once (with the round checkpoints `g1`, `g2` of s6-nano and s6-micro, no longer needed to rebuild them); deleted on the evening of 19/11 without an upload, with the round checkpoints (sections 9 and 11) |
| USCRN series (G5) | 41 MB (and the identical `series2/` copy goes too) | deleted on 19/11 without an upload (rebuilt from NOAA) |
| s6-sleep sequences (G9) | 0.48 GiB | after s6-sleep v4; deleted on the evening of 19/11 without an upload (`sky-train s2-seqgen --seed 61` rebuilds them) |
| v4's teacher rewrites and v2's false-negative answers (G11, G12) | 5 MB | the rewrites are in skycmd-route7; the rest deleted on 19/11 without an upload (the route line is closed) |
| `typed_decisions_*` and their GLM labels (G7) | 9 MB plus the label rows | after s6-sleep v4 (§7 is closed); only if the owner lifts the "gated" exclusion; kept on 20/11, the owner's question 3 being open (section 11); *20/11, after the publication:* uploaded by `scripts/hf-publish-data.py` at `fb368b1c` (the bench included, by the owner's answer to question 3), then deleted (section 12) |
| v1's features and v1-only WAVs (G1, G2) | 20.07 + 3.81 GiB, packed | deleted on 19/11 without an upload (questions 1 and 2) |
| v2's stores (G3) | 11.14 GiB (train) | after s6-ear v3, and only with the same decision; deleted on 20/11 without an upload (section 10) |

**(c) Never on the Hub (licence or the project's own rule), rebuildable only from upstream:**

| item | size | upstream on the Hub? |
|---|---:|---|
| GIFT-Eval `data/gift-eval`, `data/gift-eval-src` (G4; deleted on 19/11) | 8.50 GiB | yes, `Salesforce/GiftEval` at `30841734` (1.59 GB) |
| Chronos-ZS caches in `skycmd-data/s6-cast2` (G6; deleted on 19/11) | 7.13 GiB | yes, `autogluon/chronos_datasets` at `eeecad0b`, `_extra` at `84737b7` (ETT from GitHub) |
| `skycmd-data/s6-secrets2` and its two caches (G14; deleted on 19/11) | 6.39 GiB | no: CredData on GitHub, the issue reports on Zenodo; not guaranteed to be rebuildable |
| `data/hf/speech_commands` (G2; deleted on 19/11) | 2.26 GiB | no (TensorFlow's URL; `google/speech_commands` is a loading script over it) |
| `t_toolrouting` (G10), s6-route `src/` (G13; both deleted on 19/11) | 0.84 GiB | yes, pinned (Seal-Tools and MetaTool on GitHub) |

**(d) Must not be removed now** (updated on 19/11: §7, s6-route v5 and the LFM3 fine-tune are
closed, so their items are no longer held by them):

- s6-sleep v4 (closed on 19/11 at K1; held until the owner decides; *20/11:* deleted, on the evening
  of 19/11 and on 20/11, sections 9 and 10, except the G7 files of `data/decisions` and `data/distill`;
  *20/11, after the publication:* those too, section 12): `runs/s6-nano-g3/decide/last`, `runs/s6-sleep/seq/`, `data/decisions`
  (all 29 files), `data/distill`, `data/games`, `data/numeric`, `data/s4/heldout`,
  `runs/s6-sleep3/`, `runs/s6-sleep4/`.
- s6-ear v3 (it ran and failed its gate on 19/11, *corrected on 20/11*; held until the owner decides;
  *20/11:* deleted on 20/11, section 10): `skycmd-data/s6-ear2/` (train, test, eotbench, `checkpoints/last-ema`),
  `data/ear/wav/smart-turn`, `data/ear/clips/smart-turn.jsonl`, `data/ear/jev/turn_end.jsonl`.
- Not reproducible and only here: `data/teacher/glm_labels.jsonl` and `data/distill/glm.jsonl`
  hold the 5,640 and 4,063 rows of G7 that the Hub does not have. *20/11:* still kept, the only
  item of this list left (section 11). *20/11, after the publication:* those rows are in
  skycmd-glm-labels since 20/11, and both files were deleted (section 12): nothing is left in
  this list.

**(e) Mismatches between a published set and the local files:** none (section 2). Three
statements of the repository disagree with what was read on 18/11/2026: `scripts/hf-publish-data.py`
and `DATA_LICENSES.md` call `LocalLLaMA/typed-decisions` gated, the Hub reports it ungated;
`TODO.md` at commit `6b6ecac` (morning) said s6-cast2-pico was not uploaded, its Hub repository holds the file since
12:47 UTC; the free space was 18 GiB at 15:30, not the 5 to 6 GiB of the morning's notes.
*20/11, after the publication:* neither file calls the set gated any more: the script's rules say
it is ungated and published since 20/11/2026 (section 12).

## 7. Questions only the owner can answer

1. Does "its training data is on Hugging Face" include data whose upstream set is on the Hub at
   a recorded revision and that our command rebuilds (v1's features, G1), or must our own copy
   be uploaded first? The answer decides 20 GiB. **Answered on 19/11:** v1's features were
   deleted without an upload (section 8).
2. If our own copy is needed: may v1's log-mel features (CC-BY-4.0 upstream) be uploaded, which
   the publish script's rule excludes today? Or are they kept, or deleted without an upload?
   s6-ear v3's draft names `data/ear/audio` as the space to free (`docs/s6-ear3-prereg.md`).
   **Answered on 19/11:** deleted without an upload.
3. `LocalLLaMA/typed-decisions` is ungated on 18/11/2026. Publish `typed_decisions_train` and
   its GLM labels (G7)? The bench is the test split of a public leaderboard; keep it out?
   **Still open on 20/11:** the files of G7 are kept until it is answered (section 11).
   **Answered on 20/11, later:** publish everything of G7, the benchmark's test split included,
   marked evaluation only on the card. Done on 20/11 (section 12).
4. Publish the generated sets that the publish script does not cover (G5, G8, G9, G11, G12)?
   All are allowed by their licences. **20/11: no longer open.** G5 and G11 were deleted on 19/11
   (section 8), G8 and G9 on the evening of 19/11 (section 9), all without an upload; G12's
   rewrites are in skycmd-route7.
5. `skycmd-data/s6-secrets2` and the two caches that hold its text (6.39 GiB): keep them for the
   unscheduled verifier version of "Open, not scheduled", or delete them? They can never go to
   the Hub, and a rebuild depends on CredData's source repositories. **Answered on 19/11:**
   deleted.
6. The §9 diagnostics of s6-cast v2 are "not scheduled": drop them? Then GIFT-Eval's and
   Chronos-ZS's local copies (15.6 GiB) are only test caches of a closed line, rebuildable from
   the Hub. **Answered on 19/11:** both local copies were deleted (section 8).
7. The checkpoints of final models (section 5, rows answered "no": about 2.8 GB) and the runs set aside: keep any?
   **20/11: no longer open.** None was kept: every checkpoint was deleted on the evening of 19/11
   or on 20/11 (sections 9 and 10).

## 8. Deleted on 19/11/2026 (the owner's decision: local data that nothing open reads is removed)

On 19/11/2026, after the closures of the LFM3 fine-tune and of §7 and the end of the route line,
the owner decided that local data which no open pre-registration, queued job or owed diagnostic
reads is removed, whether or not it is on the Hub. The list below is what was deleted. Sizes are
this file's own figures ("Where things are" and sections 3 to 5, read on 18/11 and 19/11); "no figure" means this file
never measured it. Nothing was read again after the deletion.

**s6-ear v1 (the line goes on as v3, which does not read these):**

| deleted | size | on the Hub? | to get it back |
|---|---:|---|---|
| `data/ear/audio` (v1's log-mel features) | 20.07 GiB | no (G1); the upstream sets are | `sky-data fetch-audio --set all` (3.4) |
| `data/ear/wav/mswc`, `data/ear/wav/speech-commands` (v1 only) | 3.66 + 0.15 GiB | no (G2); upstream yes | as above |
| `data/hf/speech_commands` (the archive, v1 only) | 2.26 GiB | no; TensorFlow's URL | as above |

**s6-cast (the line is closed):**

| deleted | size | on the Hub? | to get it back |
|---|---:|---|---|
| `data/gift-eval`, `data/gift-eval-src` | 7.01 + 1.49 GiB | never ours to publish (G4); upstream `Salesforce/GiftEval` | `scripts/gift-eval.py download` / `export` (3.5) |
| `data/series` | 0.44 GiB | yes, `maxgfr/skycmd-series`, 8 of 8 identical | `sky-data fetch-series` |
| `data/cast-synth` | 0.29 GiB | yes, `maxgfr/skycmd-cast-synth`, 5 of 5 identical | `sky-train cast-synth --games` |
| `skycmd-data/s6-cast2` (the USCRN series and the Chronos-ZS caches) | 7.20 GiB | no (G5, G6); USCRN is public domain | `python3 -I scripts/cast2-data.py fetch --source uscrn_subhourly` from NOAA NCEI, `scripts/chronos-zs.py` (3.5) |

**s6-secrets v2 and v3 (closed; real-credential data, never published):**

| deleted | size | on the Hub? | to get it back |
|---|---:|---|---|
| `skycmd-data/s6-secrets2` | 3.97 GiB | never (G14) | CredData at `01e3d135…` and Zenodo 19622962 (3.3), if CredData's source repositories still exist |
| `runs/secrets2-smoke`, `runs/s6-secrets-verify-pico/pretrain` (caches holding that text) | 1.21 GiB each for the caches (5) | never | the trainer rebuilds them from the text above |

**s6-route (the line is closed; its records are `maxgfr/skycmd-route` and `maxgfr/skycmd-route7`,
the latter uploaded on 19/11/2026):**

| deleted, under `skycmd-data/s6-route/` | size | on the Hub? |
|---|---:|---|
| `build` | 0.41 GiB | its records and slates in skycmd-route; not `t_toolrouting` (G10) nor v2's false-negative answers (G11) |
| `build4` | 0.01 GiB on 19/11 (its teacher folder) | the rewrites in skycmd-route7 (G12) |
| `build5`, `build6` | 0.37, 0.52 GiB | every file of theirs that a model read is in skycmd-route7 |
| `build7` | 0.66 GiB | records, validation files and rewrites in skycmd-route7; not the false-negative answers (G16), nor the plans, stage and embeddings (rebuildable) |
| `toucan5` | 0.06 GiB | the fresh test T5 and the rewrites in skycmd-route7 |
| `src` | 0.84 GiB | upstream files, pinned and hashed (G13) |
| `envs`, `models` | 1.10, 0.10 GiB | not data (section 4) |
| `embed` | 0.01 GiB | — |
| `hf-cache` | no figure | a download cache |

**Kept** under `skycmd-data/s6-route/`: `eval/` (229 MB on 19/11: 0.13 GiB of 18/11 plus v7's
`eval/t5` and `eval/route7-val`), the gate's raw dumps on T5 and the only raw trace of v7's gate;
and the manifests (`MANIFEST.json`, `MANIFEST4.json`).

**What is lost with the route builds.** v2's false-negative answers (G11, 3.3 MB) and those of
v5, v6 and v7 (G16, 2.9 MB) were not on the Hub; their effect is in the published records, and
rebuilding them needs new teacher calls. `t_toolrouting` (G10) is rebuilt from upstream by
`scripts/build-route-data.sh`.

**Checkpoints and build cache:**

| deleted | size |
|---|---:|
| the checkpoints of s6-route v7: `runs/s6-route7/decide`, the model copies under `runs/s6-route7/read` | no figure |
| the checkpoints of v7's pilot: `runs/s6-route7-pilot/decide` and `read` | no figure |
| the main checkout's `target/` (the build cache, with `target/hf-data/`) | no figure (0.58 GiB for `target/hf-data/`) |

The result files of v7 and its pilot that git holds (`runs/s6-route7/GATE.md`, `gate.json`,
`summary.json` and the others) are not affected, and the published files `s6-route7-nano` stay on
the Hub and the site.

**Kept because something open reads it:**

- **s6-ear v3** (it ran and failed its gate on 19/11, *corrected on 20/11*, so this no longer holds them):
  `data/ear/wav/smart-turn` (6.79 GiB), `data/ear/clips` and `data/ear/jev` (with `mimi`,
  0.34 GiB), `skycmd-data/s6-ear2` (12.67 GiB). *20/11:* all deleted on 20/11 (section 10).
- **s6-sleep v4's K1** (decided on 19/11; it ran that day and closed v4, so this no longer holds them): `data/decisions`, `data/distill`,
  `data/games`, `data/numeric`, the `runs/s6-nano*` folders (with the base
  `runs/s6-nano-g3/decide/last`) and the `runs/s6-sleep*` folders (with v1's sequences
  `runs/s6-sleep/seq`, G9). *20/11:* deleted on the evening of 19/11 (section 9), except the G7
  files of `data/decisions` and `data/distill`; the s6-sleep v3 and v4 checkpoints of the second
  working tree on 20/11 (section 10). *20/11, after the publication:* the G7 files were deleted
  after their upload (section 12).

**What becomes removable when those two lines end.**

- When s6-ear v3 ends (it ended on 19/11: its gate failed): `data/ear/clips/smart-turn.jsonl` and `data/ear/jev/turn_end.jsonl` (on
  skycmd-ear, identical); `data/ear/wav/smart-turn`, copies of a public test set whose upstream is
  on the Hub (G2); `skycmd-data/s6-ear2` (G3), which is not on the Hub, so under the owner's rule
  it goes only after an upload or by a decision like the one taken for v1's features on 19/11.
  *20/11:* all of it went on 20/11, without an upload, by the owner's rule of 19/11 (section 10).
- When s6-sleep v4 ends (it ended on 19/11: K1 failed and v4 is closed): `data/decisions`, `distill`, `games` and
  `numeric` are on the Hub, identical (section 2), except G7: `typed_decisions_train` and
  `typed_decisions_bench` and their GLM-5.3-Flash label rows (5,640 and 4,063), which exist only
  here; the checkpoints of `runs/s6-nano*` (section 5, `g1` and `g2` only to rebuild the DAgger
  rounds of G8, which are not on the Hub); `runs/s6-sleep*`, whose sequences (G9) are not on the
  Hub and are rebuilt by `sky-train s2-seqgen --seed 61`; and `data/s4/heldout` (skycmd-zero-shot).
  *20/11:* all of it went on the evening of 19/11 (section 9), except G7, kept (section 11).
  *20/11, after the publication:* G7's rows and labels are on the Hub since 20/11, and their files
  went with `data/` (section 12).

## 9. Deleted on the evening of 19/11/2026 (s6-sleep v4 closed; s6-ear v3 training)

After s6-sleep v4 closed at K1 (`docs/s6-sleep4-prereg.md` §16), nothing open read the data of the
s6 decision models any more, and the owner's rule of 19/11 was applied again while s6-ear v3
trained. **Nothing that s6-ear v3 read was touched** (`data/ear`, `skycmd-data/s6-ear2`,
`skycmd-data/s6-ear3`). Sizes are `du` figures read just before the deletion, in MB of 1024 KB;
"under 1" means below 1 MB. Nothing was read again after the deletion.

**In the main checkout's `data/`:**

| deleted | size (MB) | on the Hub? | to get it back |
|---|---:|---|---|
| `data/trajectories` | 546 | yes, `maxgfr/skycmd-trajectories`, 11 of 11 identical (section 2) | `scripts/gen-trajectories.sh` |
| `data/text`, `data/text-open`, `data/tok` | 147, 96, under 1 | yes, `maxgfr/skycmd-text`, 14 of 14 identical | `sky-data fetch-text`, `sky-data fetch-text --set open`, `sky-data train-tokenizer` (`scripts/prepare-data.sh`) |
| `data/games` | 103 | yes, `maxgfr/skycmd-games`, 4 of 4 identical | `sky-games-gen` (`scripts/prepare-data.sh`) |
| `data/secrets` | 93 | yes, `maxgfr/skycmd-secrets`, 2 of 2 identical | `sky-data secret-jev` (seed 1234) |
| `data/dagger-s6-nano`, `data/dagger-s6-micro` (the DAgger rounds of s6-nano and s6-micro v1) | 64, 64 | **no** (G8) | `sky-arena <round model> --dagger` (`scripts/train-s6.sh` lines 186–200); see section 11, G8 |
| `data/numeric` | 21 | yes, `maxgfr/skycmd-numeric`, 1 of 1 identical | `sky-train s2-numeric` |
| `data/commands` | 3 | yes, `maxgfr/skycmd-commands`, 1 of 1 identical | `sky-data fetch-commands --only massive` |
| `data/s4` (`heldout/`) | 1 | yes, `maxgfr/skycmd-zero-shot`, 3 of 3 identical | `sky-data s4-corpus --only-heldout` |
| `data/lfm3` | under 1 | yes, `maxgfr/skycmd-math`, 2 of 2 identical | `scripts/lfm3-s1s2.sh data` at commit `5011585` (3.7) |
| `data/gift-eval-out` (s6-cast's GIFT-Eval forecast summaries) | under 1 | no (an evaluation output; the scores are in the result files of the repository) | `sky-cast-eval run` over GIFT-Eval's windows (`scripts/gift-eval.py`) |
| `data/decisions`: every file except `typed_decisions_train.jsonl` and `typed_decisions_bench.jsonl` | no separate figure (0.84 GiB for the whole folder on 18/11) | yes, `maxgfr/skycmd-decisions`, 27 of 27 identical | `sky-data fetch-decisions` (`scripts/prepare-data.sh`) |
| `data/distill/agreement.json` | no separate figure | yes, in `maxgfr/skycmd-glm-labels` (equal to the local file after the publish filter) | `sky-data teacher mix-labels` (`scripts/distill-glm.sh mix`) |
| `data/teacher/glm_labels.refused.jsonl` (the records the provider's moderation refused) | no separate figure | no | only the labelling run reads it, so as not to send those records again; `mix-labels` does not need it. Rebuilt only by new teacher calls |

The three rows without a separate figure, with the folders under 1 MB, account for the 854 MB that
the pass total leaves once the figures above and those of `runs/` are taken out.

**Under the main checkout's `runs/`:** 515 untracked files, 3,869 MB.

| deleted | on the Hub? | to get it back |
|---|---|---|
| every checkpoint (`*.safetensors`) | never: no checkpoint was ever on the Hub (section 5) | only by training again |
| every tokenized cache (`*.u32`, `*.f32`) | derived from `data/text` and `data/text-open`, which are on `maxgfr/skycmd-text` | the trainer rebuilds them on the CPU |
| every local model copy (`*.gguf`) | the published files are on the Hub and in `site/public/models/` | — |
| every data file (`*.jsonl` above 2 MB): among them the s6-sleep sequences `runs/s6-sleep/seq` (G9), v2's DAgger rounds `runs/s6-micro2/data/dagger-s6-micro` (G8) and the generations of `runs/lfm3-s1s2` | no | G9: `sky-train s2-seqgen --seed 61` (`scripts/train-sleep.sh` lines 105–109); G8: section 11 |

The small text logs and JSON files of finished runs were kept. Tracked result files were not
touched.

**Total of the pass: 5,861 MB.**

**Also removed that evening:** every working tree of the repository except the main checkout and
the session's (the agent working trees and `cast2`, `publish-s6-nano`, `route`, `secrets2`,
`sleep2`, the last with 314 MB of s6-sleep v2's checks and v3's pre-run checkpoints, section 5; each
was first checked to hold nothing that `main` lacks), and the 57 local branches other than `main`
(each contained in `main`, or holding only files that `main` removed on purpose). GitHub holds
`main` only.

## 10. Deleted on 20/11/2026 (s6-ear v3's result merged)

Once s6-ear v3's result was merged (`docs/s6-ear3-prereg.md` §17), nothing open read any data of
the project. Sizes are `du` figures read just before the deletion, in MB of 1024 KB.

| deleted | size (MB) | on the Hub? | to get it back |
|---|---:|---|---|
| `skycmd-data/s6-ear2` (v2's stores, G3: train, test, eotbench, and v2's EMA checkpoint `checkpoints/last-ema`) | 12,970 | **never** (G3); upstream, Smart Turn's and eot-bench's public sets are | `scripts/train-turn.sh data eotbench-data` (its stage `data` runs `sky-data turn-test` over v1's test WAVs first, so `data/ear` comes back before it); the EMA checkpoint only by training v2 again |
| `skycmd-data/s6-ear3` (v3's conversation stores, the extended Smart Turn stores, the eot-bench store, the teacher's opinions) | 13,156 | **never**: the conversation records derive from otoSpeech and may never be published (`DATA_LICENSES.md`, s6-ear v3) | the data stages of `scripts/train-turn3.sh`: `conv-data`, `st-extend`, `eotbench-data`, `teacher-conv`, from otoSpeech at `4b3c57b8947d5f651dd6b5e825e7f34b205759e5` under its access conditions, after v2's stores |
| `data/ear` (the Smart Turn test WAVs `wav/smart-turn` and the index files `clips/`, `jev/`, `mimi/`) | 7,300 | the index files yes, `maxgfr/skycmd-ear`, identical (section 2); the WAVs are copies of the public `pipecat-ai/smart-turn-data-v3.2-test` (G2) | `sky-data fetch-audio --set smart-turn` then `sky-data audio-jev` (`scripts/train-ear.sh` `data`), or the index files from skycmd-ear |
| the session's working tree: `models/` (its local exports) | 959 | the published files are on the Hub and in `site/public/models/` | the published files are the ones to take; a new export needs a checkpoint, and none is left |
| the session's working tree: 741 untracked files under `runs/` (the checkpoints of s6-ear v3, its pilot and its smoke run, of s6-sleep v3 and v4, s6-nano2, s6-ear v2 and s6-cast v2) | 1,160 | never (no checkpoint was) | only by training again |

**Total: 35,548 MB** (the figures above, each rounded, add up to 35,545). Free space on the data
volume after the pass: 151 GiB of 460 GiB.

## 11. What is on the disk on 20/11/2026, and what is lost

### What is kept, and why

Listed on 20/11/2026 after the pass of section 10 (`ls`, `du -sk`; nothing was opened or changed).
Folder sizes in MB of 1024 KB; single files in MB of 10⁶ bytes (`ls -l`).

| kept | size | why | on the Hub? |
|---|---:|---|---|
| `data/teacher/glm_labels.jsonl` | 92 MB | G7: its 5,640 label rows of `typed_decisions_*` exist nowhere else | its other rows yes, `maxgfr/skycmd-glm-labels`; the G7 rows no. *20/11, after the publication:* all its rows are in skycmd-glm-labels (`labels`, 747,077 rows, checked row by row); **deleted** (section 12) |
| `data/distill/glm.jsonl` | 555 MB | G7: its 4,063 label rows of `typed_decisions_*` exist nowhere else | as above. *20/11, after the publication:* all its rows are in skycmd-glm-labels (`agree`, 560,238 rows, checked row by row); **deleted** (section 12) |
| `data/decisions/typed_decisions_train.jsonl`, `typed_decisions_bench.jsonl` | 7.0 MB, 2.3 MB | G7: the rows those labels belong to | no (`EXCLUDED` in `scripts/hf-publish-data.py`). *20/11, after the publication:* yes, skycmd-decisions configs `typed_decisions_train` (6,000 rows) and `typed_decisions_bench` (2,000, evaluation only), checked row by row; **deleted** (section 12) |
| `data/hf/LocalLLaMA__typed-decisions` | 0.8 MB | G7: the raw download of those rows | no; upstream `LocalLLaMA/typed-decisions`. *20/11, after the publication:* **deleted** with `data/`; upstream is ungated, Apache-2.0, at `822f44b3` on 20/11 (section 12) |
| `skycmd-data/s6-ear3-gate` | 80 MB | the raw trace of s6-ear v3's gate: the per-record outputs the gate read, v2's readings on the same cuts, the harness files, the training log, the pilot logs | **never**: part of it derives from otoSpeech |
| `skycmd-data/s6-route` | 229 MB | `eval/` and the manifests `MANIFEST.json`, `MANIFEST4.json` (section 8): the only raw trace of v7's gate | no (the records are skycmd-route and skycmd-route7) |
| `skycmd-data/run-logs` | 2.5 MB | the consoles of the runs of 9 to 19/11 and the brake's log | no |
| `skycmd-data/ultrasearch` | 1.4 MB | web research dossiers (`flexible-weights`, `s6-sleep4`) | no |
| the main checkout's `runs/` | 49.6 MB | the small text logs and JSON files of finished runs, next to the tracked result files | the result files are in the repository |

Section 7's question 3 (publish `typed_decisions_train` and its GLM labels, keep the bench out?)
is still the owner's. The files of G7 stay until it is answered.

*20/11, after the publication:* the owner answered it the same day (publish everything, the bench
included, marked evaluation only); the four G7 rows above were published, checked and deleted
(section 12). Of this table, what is left on the disk is `skycmd-data/s6-ear3-gate`,
`skycmd-data/s6-route`, `skycmd-data/run-logs`, `skycmd-data/ultrasearch` and the main checkout's
`runs/`, unchanged.

Also still on the disk, and not data: the main checkout's `models/` (18 MB: the exported GGUF files
of s6-nano and s6-micro v1 with those of their DAgger rounds, of s6-cast-pico, s6-ear-nano,
s6-secrets-pico and s6-secrets-verify-pico) and `models-src/` (39 MB, s5's model sources, 3.1).
Neither pass touched them.

### What is lost

- **No training checkpoint exists any more.** None was ever on the Hub; the published GGUF files
  are, on the Hub and in `site/public/models/`. No model can be continued or fine-tuned from its
  float weights: v2's EMA checkpoint, which was s6-ear v3's start, is gone too. Each model comes
  back only by training it again with its script and its pre-registration.
- **The DAgger rounds (G8)** are gone, and so are the round checkpoints (`runs/s6-nano-g1`, `g2`
  and those of s6-micro). A round's records are written by `sky-arena <model.gguf> --dagger`,
  played by the previous round's exported file (`scripts/train-s6.sh` lines 186–200). For s6-nano
  and s6-micro v1 those files are still in the main checkout's `models/` (above): the rounds could
  be played again from them with the script's settings. That was not tried, so whether they come
  out byte-identical is not known. For s6-micro v2 the round files were under `runs/s6-micro2/` and
  were deleted on the evening of 19/11: its rounds come back only by training it again.
- **The s6-sleep sequences (G9)** are gone; `sky-train s2-seqgen --seed 61` rebuilds them
  (generated by our code, on the CPU).
- **v2's and v3's stores (G3, `skycmd-data/s6-ear3`)** are gone; they come back from the public sets
  and from otoSpeech by the stages of section 10, in hours of CPU (v2's took about 3.5 h on 16/11;
  v3's stream about 49 GB of otoSpeech and 41 GB of Smart Turn again).
- **`data/teacher/glm_labels.refused.jsonl`** is gone; it comes back only from new teacher calls.
- The losses of the route builds (G11, G16) are those of section 8.

### The gap of section 4, item by item

| # | on the disk on 20/11/2026? |
|---|---|
| G1 | no: deleted on 19/11 (section 8) |
| G2 | no: mswc, speech-commands and the archive deleted on 19/11 (section 8); smart-turn on 20/11 (section 10) |
| G3 | no: deleted on 20/11 (section 10) |
| G4 | no: deleted on 19/11 (section 8) |
| G5 | no: deleted on 19/11 (section 8) |
| G6 | no: deleted on 19/11 (section 8) |
| G7 | **yes**: kept (above). *20/11, after the publication:* no: published on 20/11, then deleted with `data/` (section 12) |
| G8 | no: deleted on the evening of 19/11 (section 9); the files that played s6-nano's and s6-micro v1's rounds are kept (above) |
| G9 | no: deleted on the evening of 19/11 (section 9) |
| G10 | no: deleted on 19/11 with `build/` (section 8) |
| G11 | no: deleted on 19/11 with `build/` (section 8) |
| G12 | no: deleted on 19/11 (section 8); the rewrites are in skycmd-route7 |
| G13 | no: deleted on 19/11 (section 8) |
| G14 | no: deleted on 19/11 (section 8) |
| G15 | no, except `LocalLLaMA__typed-decisions` (G7). Section 8 does not list the rest of `data/hf/`; the folder was last changed at 10:28 on 19/11, so it went no later than then. *20/11, after the publication:* `LocalLLaMA__typed-decisions` was deleted with `data/` (section 12) |
| G16 | no: deleted on 19/11 with `build7/` (section 8) |

## 12. Published and deleted on 20/11/2026, after the owner's answer to question 3

On 20/11/2026 the owner answered section 7's question 3: publish everything of G7, the benchmark's
test split included, marked evaluation only on the card. This section records that upload, the
checks made on it, and the deletion that followed. Nothing else was uploaded or changed on the Hub.

### What was uploaded

`scripts/hf-publish-data.py` at commit `fb368b1c`:

| data set | what the upload did | Hub revision after it | files | rows |
|---|---|---|---:|---:|
| [skycmd-decisions](https://huggingface.co/datasets/maxgfr/skycmd-decisions) | added the configs `typed_decisions_train` (6,000 rows) and `typed_decisions_bench` (2,000 rows, the benchmark's test split, marked evaluation only), with `--merge-published`: the other 27 configs were carried from the manifest of commit `6cbcce8c` (Hub revision `a8f33fd2`), not uploaded again | `36704fe4007b9ee27babe7cc35d917f742bfeaff`, manifest at `fb368b1c` | 32 (29 parquet) | 831,387 |
| [skycmd-glm-labels](https://huggingface.co/datasets/maxgfr/skycmd-glm-labels) | rebuilt whole, now with the G7 rows: `labels` 747,077 rows, `agree` 560,238 rows, and `agreement.json` with 470 sources. `agreement.json` was no longer on the disk (section 9), so it was recomputed from the two label files; without `typed_decisions_train` it is byte-identical to the one published before. The 121 texts the provider's moderation refused are given by `--glm-refused 121`, since `data/teacher/glm_labels.refused.jsonl` went on 19/11 (section 9) | `486a2508836d5b582227bd652333adb403262ed5` | 5 | 747,077 + 560,238 |

### How the upload was checked

- Every row of the four parquet files (`typed_decisions_train`, `typed_decisions_bench`, `labels`,
  `agree`), downloaded again from the Hub after the upload, equals the local file's rows: an
  order-independent hash of every row, on 6,000, 2,000, 747,077 and 560,238 rows, identical in all
  four.
- The sha256 that skycmd-decisions' manifest gives for the two source files
  (`typed_decisions_train.jsonl`, `typed_decisions_bench.jsonl`) equal those of the local files.
- `load_dataset("maxgfr/skycmd-decisions", …)` loads `typed_decisions_train`,
  `typed_decisions_bench` and `boolq`.

### What was deleted, after the checks

| deleted | size | on the Hub? | to get it back |
|---|---:|---|---|
| the whole `skycmd/data/` folder: `data/teacher/glm_labels.jsonl`, `data/distill/glm.jsonl`, `data/decisions/typed_decisions_train.jsonl` and `typed_decisions_bench.jsonl`, and the raw download `data/hf/LocalLLaMA__typed-decisions` (0.8 MB) | 626 MB | yes: the label files are skycmd-glm-labels (`labels`, `agree`, `agreement.json`), the two decision files are skycmd-decisions (`typed_decisions_train`, `typed_decisions_bench`), every row checked (above) | the published sets. The raw download: upstream `LocalLLaMA/typed-decisions`, ungated, Apache-2.0, at `822f44b3` on 20/11; our rows were downloaded on 25/10/2026 without a pinned revision, so a new download may differ from them, and the published configs are the copy to take |
| `target/` (the assembled upload, `target/hf-data/`) | 125 MB | it is what was uploaded | `scripts/hf-publish-data.py --dry-run` assembles it again, given the data |

**Also removed on 20/11/2026, and not recorded before:** the last local working copy besides the
main checkout (the session's working tree of section 10), after its HEAD was checked equal to
`origin/main`; and every local branch except `main`. The main checkout is now the only working
tree of the repository.

### What is on the disk after this pass

Unchanged since section 11 (nothing of it was opened or changed). Free space on the data volume:
154 GiB.

| kept | size | why | on the Hub? |
|---|---:|---|---|
| `skycmd-data/s6-ear3-gate` | 80 MB | the raw trace of s6-ear v3's gate (section 11) | **never**: part of it derives from otoSpeech |
| `skycmd-data/s6-route` | 229 MB | `eval/` and the manifests: the only raw trace of v7's gate (section 8) | no (the records are skycmd-route and skycmd-route7) |
| `skycmd-data/run-logs` | 2.5 MB | the consoles of the runs of 9 to 19/11 and the brake's log | no |
| `skycmd-data/ultrasearch` | 1.4 MB | web research dossiers | no |
| the main checkout's `runs/` (ignored files) | small text logs and JSON files | the logs of finished runs, next to the tracked result files | the result files are in the repository |
| the main checkout's `models/`, `models-src/` (ignored; not data) | 18 MB, 39 MB | as section 11 | the published GGUF files are on the Hub; `models-src/` is s5's model sources (3.1) |

**What exists only on this disk now.** Of what the project wrote, only the items of the table
above: the raw trace of s6-ear v3's gate, s6-route's `eval/` and manifests, the run logs, the
research dossiers and the small logs of `runs/`, which are the items of section 11 outside G7.
None of G1 to G16 is on the disk: each is on the Hub, rebuilt from upstream by the commands of
sections 3 and 8 to 10, or lost as section 11 ("What is lost") says.

**What this pass loses.** Nothing that is not on the Hub. The local raw download of
`LocalLLaMA/typed-decisions` is gone, and since it was not pinned, the bytes of 25/10/2026 now exist
only as the published configs derived from them.
