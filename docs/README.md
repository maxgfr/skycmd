# The documents of `docs/`

One table per line of work, in the order the lines started, then the documents that serve every
line. Each row gives a version, its status with its date (copied from the
[Results](https://maxgfr.github.io/skycmd/results/) page), its pre-registration, the studies and
diagnoses written around it, and its result folder. A result folder marked "not committed" exists
only on the training machine, and since the deletions of 19 and 20/11/2026 only its small text and
JSON files are left there ([`data-inventory.md`](data-inventory.md), sections 9 to 11); its numbers
are in [`BENCH.md`](../BENCH.md). `runs/` is ignored by git by default: the small result files of a
gated run (results, summaries, gate files) are force-added and tracked under `runs/<name>/`, while
checkpoints, logs and the arena files written by `sky-arena --json` are not. The nine pre-registrations of s5, s6, s6-gen, s6-ear, s6-secrets,
s6-cast, LFM3 (two) and flexible weights, and `research-night.md`, were written in French and
translated into English on 19/11/2026 (each says so at its top); every document is now in English.
On 20/11/2026 the numbers those translations still wrote the French way were given English
typography (decimal points, commas between thousands, the dollar sign before the amount), with no
digit, rule or date changed. On 20/11/2026 the absolute paths of the local working copy recorded in
17 result files under `runs/` (the commands, gate files, logs and reports of s6-nano2, s6-nano3,
s6-micro2, s6-cast-pico, s6-cast2-pico, s6-sleep3 and s6-sleep4) were made repository-relative; no
measured value changed. Paths into the data folder outside the repository are left as recorded.
Final status, 20/11/2026: the experiments are finished; every line below is closed or published,
and no pre-registration is open. The whole project is written up in the technical report,
[`site/paper/index.md`](../site/paper/index.md) (<https://maxgfr.github.io/skycmd/paper/>).

## s1 to s5 (October 2026)

| version | status | pre-registration | study or diagnosis | results |
|---|---|---|---|---|
| s1-pico, s2-nano, s2-nano-doom | no gate, measured (October 2026) | none | [`SPEC.md`](SPEC.md), [`SPEC-S2.md`](SPEC-S2.md) | [`BENCH.md`](../BENCH.md), s1 and s2 sections |
| s3-pico | no gate, measured (October 2026) | none | [`SPEC-S3.md`](SPEC-S3.md), [`night-experiments.md`](night-experiments.md), [`research-night.md`](research-night.md) | [`BENCH.md`](../BENCH.md), s3 section |
| s4-micro | failed, gate fixed before training (October 2026) | none (gate in `BENCH.md`) | [`research-s4.md`](research-s4.md) | [`BENCH.md`](../BENCH.md), s4 section |
| s5-lite (three bets) | failed (14/10/2026) | [`s5-prereg.md`](s5-prereg.md) | | [`BENCH.md`](../BENCH.md), s5 section; `runs/s5-*` not committed |

## s6: the pilot, the ablation and the big run (12 to 14 November 2026)

| version | status | pre-registration | study or diagnosis | results |
|---|---|---|---|---|
| s6-pico pilot and ablation (the recipe) | ran before 12/11/2026 | [`s6-prereg.md`](s6-prereg.md) | [`SPEC-S6.md`](SPEC-S6.md) | [`runs/recipe/`](../runs/recipe/) |
| s6-nano | passed (13/11/2026) | [`s6-gen-prereg.md`](s6-gen-prereg.md) | [`big-run-report.md`](big-run-report.md) | [`runs/s6-nano/eval/`](../runs/s6-nano/eval/results.md) |
| s6-micro v1 (s6-micro-v1) | failed, calibration (14/11/2026) | [`s6-gen-prereg.md`](s6-gen-prereg.md) | [`big-run-report.md`](big-run-report.md) | [`runs/s6-micro/eval/`](../runs/s6-micro/eval/results.md) |
| s6-secrets-pico | passed on its synthetic bench (12/11/2026); fails on real text (14/11/2026) | [`s6-secrets-prereg.md`](s6-secrets-prereg.md) | [`big-run-report.md`](big-run-report.md) | [`runs/s6-secrets-pico/`](../runs/s6-secrets-pico/report.md) |
| s6-ear-nano | failed, three gates of four (12/11/2026) | [`s6-ear-prereg.md`](s6-ear-prereg.md) | [`big-run-report.md`](big-run-report.md) | [`runs/s6-ear-nano/eval/`](../runs/s6-ear-nano/eval/gate.json) |
| s6-cast-pico | failed (14/11/2026) | [`s6-cast-prereg.md`](s6-cast-prereg.md) | [`SPEC-S6-CAST.md`](SPEC-S6-CAST.md), [`big-run-report.md`](big-run-report.md) | [`runs/s6-cast-pico/`](../runs/s6-cast-pico/gift-eval/summary.json) |

The run's own gate table is [`runs/all/REPORT.md`](../runs/all/REPORT.md).

## s6-micro

| version | status | pre-registration | study or diagnosis | results |
|---|---|---|---|---|
| v1 (s6-micro-v1) | failed, calibration (14/11/2026) | [`s6-gen-prereg.md`](s6-gen-prereg.md) | | [`runs/s6-micro/eval/`](../runs/s6-micro/eval/results.md) |
| v2 (s6-micro) | passed, on a second look (16/11/2026) | [`s6-gen-prereg.md`](s6-gen-prereg.md), amendments 12, 16 and 17 | | [`runs/s6-micro2/`](../runs/s6-micro2/runs/s6-micro/eval/results.md) |

## s6-secrets and the secrets verifier

| version | status | pre-registration | study or diagnosis | results |
|---|---|---|---|---|
| v1 (s6-secrets-pico) | passed on its synthetic bench (12/11/2026); fails on real text (14/11/2026) | [`s6-secrets-prereg.md`](s6-secrets-prereg.md) | | [`runs/s6-secrets-pico/`](../runs/s6-secrets-pico/report.md) |
| v2 (s6-secrets-verify-pico) | failed, G1 (15/11/2026) | [`s6-secrets2-prereg.md`](s6-secrets2-prereg.md) | [`s6-secrets2-feasibility.md`](s6-secrets2-feasibility.md), [`research-s6-usefulness.md`](research-s6-usefulness.md) | [`runs/s6-secrets2/`](../runs/s6-secrets2/) |
| v3 (the rules and v2 together) | failed, G1, a second look (15/11/2026) | [`s6-secrets3-prereg.md`](s6-secrets3-prereg.md) | [`s6-secrets3-diagnosis.md`](s6-secrets3-diagnosis.md) | [`runs/s6-secrets3/`](../runs/s6-secrets3/) |
| a possible v4 | closed with a diagnosis, nothing trained (18/11/2026) | none | [`s6-secrets4-diagnosis.md`](s6-secrets4-diagnosis.md) | |

## s6-ear (end of turn)

| version | status | pre-registration | study or diagnosis | results |
|---|---|---|---|---|
| v1 (s6-ear-nano) | failed, three gates of four (12/11/2026) | [`s6-ear-prereg.md`](s6-ear-prereg.md) | | [`runs/s6-ear-nano/eval/`](../runs/s6-ear-nano/eval/gate.json) |
| v2 (s6-ear2-nano) | passed (17/11/2026) | [`s6-ear2-prereg.md`](s6-ear2-prereg.md) | [`s6-ear2-study.md`](s6-ear2-study.md), [`s6-ear2-readings.md`](s6-ear2-readings.md) | [`runs/s6-ear2-nano/`](../runs/s6-ear2-nano/), with `runs/s6-ear2-pilot/`, `runs/s6-ear2-harness/` and `runs/s6-ear2-probe/` |
| v3 (s6-ear3-nano) | failed, E1, E2 and E3 (19/11/2026) | [`s6-ear3-prereg.md`](s6-ear3-prereg.md) | [`s6-ear3-study.md`](s6-ear3-study.md) | [`runs/s6-ear3-nano/`](../runs/s6-ear3-nano/GATE.md), with `runs/s6-ear3-pilot/` |

## s6-cast (forecasting)

| version | status | pre-registration | study or diagnosis | results |
|---|---|---|---|---|
| v1 (s6-cast-pico) | failed (14/11/2026) | [`s6-cast-prereg.md`](s6-cast-prereg.md) | [`SPEC-S6-CAST.md`](SPEC-S6-CAST.md) | [`runs/s6-cast-pico/`](../runs/s6-cast-pico/gift-eval/summary.json) |
| v2 (s6-cast2-pico) | failed, G1 (18/11/2026); the line stops | [`s6-cast2-prereg.md`](s6-cast2-prereg.md) | [`s6-cast2-study.md`](s6-cast2-study.md) | [`runs/s6-cast2-pico/`](../runs/s6-cast2-pico/GATE.md), with `runs/s6-cast2-pilot/` |

## s6-route (tool routing)

Versions 2 and 3 were pre-registered as s6-nano v2 and v3, so their files carry those names.

| version | status | pre-registration | study or diagnosis | results |
|---|---|---|---|---|
| v1 (s6-nano, the general model) | passed (13/11/2026) | [`s6-gen-prereg.md`](s6-gen-prereg.md) | | [`runs/s6-nano/eval/`](../runs/s6-nano/eval/results.md) |
| v2 (s6-route-nano) | failed, G1 and G4 (16/11/2026) | [`s6-nano2-prereg.md`](s6-nano2-prereg.md) | [`s6-nano2-study.md`](s6-nano2-study.md), [`s6-nano2-router-study.md`](s6-nano2-router-study.md), [`s6-nano2-readings.md`](s6-nano2-readings.md), [`s6-route-demo-note.md`](s6-route-demo-note.md) | [`runs/s6-nano2/`](../runs/s6-nano2/), with `runs/s6-nano2-pilot/` |
| v3 (s6-route) | passed on fresh slates, system defined after v2's test was read (16/11/2026) | [`s6-nano3-prereg.md`](s6-nano3-prereg.md) | | [`runs/s6-nano3/`](../runs/s6-nano3/) |
| v4 | closed: stopped at its data checks, before any training (18/11/2026) | [`s6-route4-prereg.md`](s6-route4-prereg.md) | [`s6-route4-diagnosis.md`](s6-route4-diagnosis.md), [`s6-route4-study.md`](s6-route4-study.md) | [`runs/s6-route4/`](../runs/s6-route4/) |
| v5 | closed: stopped at its data checks, before any training (18/11/2026) | [`s6-route5-prereg.md`](s6-route5-prereg.md) | | [`runs/s6-route5/`](../runs/s6-route5/) |
| v6 | closed: stopped at its stop rule, before any training (18/11/2026) | [`s6-route6-prereg.md`](s6-route6-prereg.md) | | [`runs/s6-route6/`](../runs/s6-route6/) |
| v7 (s6-route7-nano) | failed, G1, G2, G3 and G6 (19/11/2026); the line is closed | [`s6-route7-prereg.md`](s6-route7-prereg.md) | | [`runs/s6-route7/`](../runs/s6-route7/), with `runs/s6-route7-pilot/` |

## s6-sleep (a fast-weight memory)

| version | status | pre-registration | study or diagnosis | results |
|---|---|---|---|---|
| v1 | stopped by its pico pilot (before 15/11/2026, the date of its diagnosis) | [`s6-sleep-prereg.md`](s6-sleep-prereg.md) | [`s6-sleep-diagnosis.md`](s6-sleep-diagnosis.md) | `runs/s6-sleep/` not committed |
| v2 | closed at its CPU check C4, before any GPU run (15/11/2026) | [`s6-sleep2-prereg.md`](s6-sleep2-prereg.md) | [`s6-sleep2-study.md`](s6-sleep2-study.md), [`s6-sleep3-diagnosis.md`](s6-sleep3-diagnosis.md) | [`runs/s6-sleep2/checks/`](../runs/s6-sleep2/checks/) |
| v3 (s6-sleep3-nano) | failed (17/11/2026) | [`s6-sleep3-prereg.md`](s6-sleep3-prereg.md) | [`s6-sleep3-readings.md`](s6-sleep3-readings.md), [`research-fast-weight-memory.md`](research-fast-weight-memory.md), [`s6-sleep4-diagnosis.md`](s6-sleep4-diagnosis.md) | [`runs/s6-sleep3/`](../runs/s6-sleep3/gate.json) |
| v4 | closed: K3 failed (18/11/2026), K1 failed (19/11/2026), no model | [`s6-sleep4-prereg.md`](s6-sleep4-prereg.md) | [`research-s6-sleep4.md`](research-s6-sleep4.md) | [`runs/s6-sleep4/`](../runs/s6-sleep4/) |

## LFM3-1.7B

| version | status | pre-registration | study or diagnosis | results |
|---|---|---|---|---|
| System 1 first, System 2 when needed | failed (14 and 15/11/2026) | [`lfm3-s1s2-prereg.md`](lfm3-s1s2-prereg.md) | | [`BENCH.md`](../BENCH.md), LFM3 section; `runs/lfm3-s1s2/` not committed |
| fine-tuning to answer without thinking | closed, nothing trained (19/11/2026) | [`lfm3-ft-prereg.md`](lfm3-ft-prereg.md) | | none |

## Flexible weights

| version | status | pre-registration | study or diagnosis | results |
|---|---|---|---|---|
| four experiments (an LLM that reads and writes small models' weights) | closed without running, after a literature review (19/11/2026) | [`weights-prereg.md`](weights-prereg.md) | [`research-flexible-weights.md`](research-flexible-weights.md) | none |

## Documents for every line

| document | what it is |
|---|---|
| [`SPEC.md`](SPEC.md) | the shared engineering contract every crate follows |
| [`SPEC-S2.md`](SPEC-S2.md), [`SPEC-S3.md`](SPEC-S3.md), [`SPEC-S6.md`](SPEC-S6.md), [`SPEC-S6-CAST.md`](SPEC-S6-CAST.md) | the architectures s2, s3 and s6 and the s6 forecaster |
| [`compat.md`](compat.md) | which other runtimes can read skycmd's GGUF files |
| [`idle-training.md`](idle-training.md) | the idle brake: training on a laptop in use, and budgets in working minutes |
| [`data-inventory.md`](data-inventory.md) | where every model's training, validation and test data lives (inventory of 18/11/2026), the deletions of 19 and 20/11/2026 and what is left on the disk |
| [`big-run-report.md`](big-run-report.md) | the final report of the big run of 12 to 14 November 2026 |
| [`research.md`](research.md), [`research-night.md`](research-night.md), [`research-s4.md`](research-s4.md), [`research-s6-usefulness.md`](research-s6-usefulness.md), [`research-fast-weight-memory.md`](research-fast-weight-memory.md), [`research-s6-sleep4.md`](research-s6-sleep4.md), [`research-flexible-weights.md`](research-flexible-weights.md) | literature and research dossiers, each dated |
| [`night-experiments.md`](night-experiments.md) | the night of 12/10/2026: smaller and faster than s3-pico |
| [`site-check-2026-11-18.md`](site-check-2026-11-18.md), [`site-check-2026-11-19.md`](site-check-2026-11-19.md) | the checks of the site and the Spaces, the second one in a browser |
