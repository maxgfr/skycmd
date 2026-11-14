# The big run (12 to 14 November 2026): final report

`scripts/train-all.sh` trained five pre-registered s6 models one after the other on one Apple M5 (16 GB), from 12/11 09:06 to 14/11 19:32: s6-ear-nano, s6-secrets-pico, s6-nano, s6-micro and s6-cast-pico. Before it, tranche 1 ran the pico pilot and ablation that chose the recipe of all five. This file collects what the run's own files say. Every number below was read in the file named next to it, or computed from such files by the command given; nothing is estimated. Times are the local times written in the logs.

**Outcome.** Two models pass their gates (s6-secrets-pico, s6-nano) and three fail (s6-ear-nano, s6-micro, s6-cast-pico). A failed model is not published; its numbers are below and in [`BENCH.md`](../BENCH.md).

Files read, all in the main checkout (`runs/` and `data/` are not in the repository except where noted):

- `runs/all/REPORT.md` (the gate table the run wrote; committed), `runs/all/*.log` (one log per step), `runs/all/incidents.txt`, `runs/all/done/` (one empty marker per finished step), `runs/train-all.log` (all steps in one file, with their start times), `runs/brake.log` (each change of the brake file), `runs/recipe/recipe.json` (committed);
- s6-cast: `runs/s6-cast-pico/args.json`, `runs/s6-cast-pico/log.jsonl`, `runs/s6-cast-pico/last{,-ema}/state.json`, `runs/s6-cast-pilot/choice.json`, `data/gift-eval-out/{s6-cast-pico,s6-cast-pico-flip,tinycast,seasonal-naive}/{summary.json,all_results.csv}`, `models/s6-cast/*.gguf` (sizes only);
- the other four: `runs/s6-ear-nano/eval/*.json`, `runs/s6-secrets-pico/report.md`, `runs/s6-{nano,micro}/eval/{results.md,report.md,eval-*.json,dump-*.jsonl}`;
- for the missing-items section: `runs/distill-glm.log`, `runs/prepare-data.log`, `runs/s6-recipe.log`, `data/distill/agreement.json`, `data/teacher/glm_labels.refused.jsonl`.

## 1. What ran, and when

Start time of each step, from the `== <date> <step>` lines of `runs/train-all.log`:

| step | started | note |
|---|---|---|
| tranche 1 (recipe: pico pilot and ablation) | before 12/11 | ended 12/11 04:43 (`incidents.txt`); `recipe.json` written `2026-11-12T03:43:31+00:00` |
| `ear-pretrain` | 12/11 09:06:20 | launch: `SKY_DUTY=0.5 SKY_COOL=45:15 RESUME=1` |
| `ear-check` | 12/11 11:39:46 | `val_acc_cb0 0.21963 (bar 0.00488)` (`runs/all/ear-check.log`) |
| `ear-train` | 12/11 11:39:47 | stopped by hand 15:13:44 (27 GB held), relaunched 15:34:32 |
| `secrets-train` | 12/11 21:49:59 | |
| `nano-glm`, `nano-lr` | 12/11 23:02:26 | memory cap 13/11 00:16:33 (`brake.log`); relaunch logged 00:56:06 |
| `nano-train` | 13/11 02:53:33 | memory cap 06:05:45 (`brake.log`); relaunched 06:07:20 |
| `nano-dagger` | 13/11 06:37:36 | |
| `nano-exit` | 13/11 08:00:54 | |
| `nano-eval` | 13/11 08:06:07 | |
| `micro-glm`, `micro-lr` | 13/11 08:15:43 | |
| `micro-train` | 13/11 12:28:42 | stopped by hand 14/11 00:21:16 (swap), relaunched 00:57:28 |
| `micro-dagger` | 14/11 02:22:09 | |
| `micro-exit` | 14/11 05:04:56 | |
| `micro-eval` | 14/11 05:26:07 | |
| `cast-pilot` | 14/11 05:40:00 | |
| `cast-train` | 14/11 06:07:30 | |
| `cast-export` | 14/11 19:05:45 | |
| `cast-eval` | 14/11 19:05:47 | `runs/all/done/cast-eval` and `runs/all/REPORT.md` dated 19:32 |

All 20 steps have their marker in `runs/all/done/`.

## 2. Phases, steps and brake modes, per model

**How to read the brake column.** Budgets in minutes count working time only (the amendments on the brake in each pre-registration). Every launch sets `SKY_DUTY=0.5 SKY_COOL=45:15` (the header line of `runs/train-all.log`); `sky-train` prints a `brake: duty …` line when it reads another mode from `runs/brake`. A run with no such line ran in the launch mode, 0.5 45:15. The modes are written `duty work:rest` (minutes); the night mode was `1 55:5` until 13/11 and `0.95 55:5` from the night of 13/11 on (`brake.log`). "Cooling" counts the pauses printed by the run (`cooling down for N s`). Steps are the `done:` lines; "minutes" is the last `(x/budget min)` progress line. Modes and pauses come from `runs/train-all.log` (and `runs/all/cast-*.log` for s6-cast), parsed in order.

### s6-ear-nano (`runs/all/ear-{pretrain,check,train}.log`)

| phase | budget | steps done | brake modes | cooling pauses |
|---|---:|---:|---|---|
| Mimi pretraining | 60 min | 48,626 (60.0 min) | 0.5 45:15 | 2 × 900 s |
| decision, first run (discarded, see §3) | 60 min | 31,480 (60.0 min) | 0.5 45:15 | 2 × 900 s, 876 s |
| control, first run (stopped by hand, not kept) | 60 min | stopped at 11.4 min | 0.5 45:15 | 894 s |
| decision | 60 min | 48,205 (60.0 min) | 0.5 45:15 | 3 × 900 s |
| control (no Mimi pretraining) | 60 min | 49,131 (60.0 min) | 0.5 45:15 | 3 × 900 s |

`brake.log` starts at 12/11 15:34:29 with `0.5 45:15` and does not change before 23:49:29, so the brake file and the launch mode agree for the runs after the relaunch. Before 15:34 there is no brake file record.

### s6-secrets-pico (`runs/all/secrets-train.log`)

| phase | budget | steps done | brake modes | cooling pauses |
|---|---:|---:|---|---|
| MLM pretraining, sequences of 256 tokens | 8 min | 6,336 (8.0 min) | 0.5 45:15 | 582 s |
| decision training on `data/secrets` | 15 min | 7,555 (15.0 min) | 0.5 45:15 | 900 s |

### s6-nano (`runs/all/nano-{glm,lr,train,dagger,exit,eval}.log`)

Teacher labels: `GLM labels: 747077 of 747198 (0.9998)` (`nano-glm.log`).

| phase | budget | steps done | brake modes | cooling pauses |
|---|---:|---:|---|---|
| rate pilot x0.5, pretraining | 20 min | 10,082 (20.0 min) | 0.5 45:15 | none |
| rate pilot x0.5, decision, first run (stopped by the memory cap, set aside) | 20 min | 860 (8.2 min) | 0.85 50:10, 0.5 45:15 | 600 s, 892 s |
| rate pilot x0.5, decision | 20 min | 4,374 (19.9 min) | 0.85 50:10, 1 55:5, 0.85 50:10, 1 55:5 | 128 s |
| rate pilot x1, pretraining | 20 min | 11,929 (20.0 min) | 1 55:5 and 0.85 50:10, alternating (5 successive modes) | none |
| rate pilot x1, decision | 20 min | 4,412 (19.9 min) | 1 55:5, 0.85 50:10, 1 55:5 | 473 s |
| rate pilot x2, pretraining | 20 min | 12,001 (20.0 min) | 1 55:5 | none |
| rate pilot x2, decision | 20 min | 4,404 (20.0 min) | 1 55:5 | none |
| MLM pretraining (rate x2.0 x1) | 90 min | 53,371 (90.0 min) | 1 55:5 and 0.85 50:10, alternating (11 successive modes) | 300 s, 600 s |
| decision, first run (stopped by the memory cap at step 16,020, 72.2 min) | 90 min | resumed from step 15,573 | 1 55:5 | 2 × 300 s |
| decision, resumed | (same 90 min) | 19,893 (89.9 min) | 1 55:5, 0.85 50:10, 1 55:5 | none |
| DAgger round 1 (oracle share 0.2) | 15 min | 4,645 (15.0 min) | 1 55:5 and 0.85 50:10, alternating (5 successive modes) | 300 s |
| DAgger round 2 (0.1) | 15 min | 4,660 (15.0 min) | 1 55:5 | none |
| DAgger round 3 (0.05) | 15 min | 4,644 (15.0 min) | 1 55:5 | none |

The pilot kept x1 (`rate x2.0 x1` in the `nano-train` header). `runs/s6-nano/eval/results.md` gives the pretraining as 53,360 steps, 0.87B tokens, 588 per parameter, epoch 8, MLM loss 2.356; the `done:` line says 53,371 steps. Exit thresholds (`nano-exit.log`): CRC at α = 0.02 gives τ 0.53 (choice) and 0.50 (noul), rejected in closed loop (Rescue 2.80 against 3.00, Dodge vetoes 1,016 against 23 at full depth).

### s6-micro (`runs/all/micro-{glm,lr,train,dagger,exit,eval}.log`)

Teacher labels: `GLM labels: 747077 of 747198 (0.9998)` (`micro-glm.log`).

| phase | budget | steps done | brake modes | cooling pauses |
|---|---:|---:|---|---|
| rate pilot x0.5, pretraining | 20 min | 3,770 (20.0 min) | 0.85 50:10 | none |
| rate pilot x0.5, decision | 20 min | 4,507 (20.0 min) | 0.85 50:10 | 600 s |
| rate pilot x1, pretraining | 20 min | 3,769 (20.0 min) | 0.85 50:10, 0.5 45:15 | none |
| rate pilot x1, decision | 20 min | 3,528 (20.0 min) | 0.5 45:15, 0.85 50:10, 0.5 45:15 | 900 s, 600 s |
| rate pilot x2, pretraining | 20 min | 3,001 (20.0 min) | 0.5 45:15 | 900 s |
| rate pilot x2, decision | 20 min | 3,233 (19.9 min) | 0.5 45:15 | 885 s |
| MLM pretraining (rate x2.0 x0.5) | 180 min | 31,714 (179.9 min) | 0.5 45:15, then 0.85 50:10 | 7 × 900 s, 893 s |
| decision, first run (stopped by hand at step 27,280, 126.2 min) | 180 min | resumed from step 27,161 | 0.85 50:10 and 0.5 45:15, alternating (4 successive modes) | 3 × 600 s, 900 s |
| decision, resumed uncompiled | (same 180 min) | 38,826 (180.0 min) | 0.95 55:5 | 12 s, 300 s |
| DAgger round 1 (0.2) | 20 min | 5,951 (20.0 min) | 0.95 55:5 | none |
| DAgger round 2 (0.1) | 20 min | 5,898 (19.9 min) | 0.95 55:5, 0.85 50:10, 0.95 55:5 | none |
| DAgger round 3 (0.05) | 20 min | 5,952 (20.0 min) | 0.95 55:5 | none |

The pilot kept x0.5. `results.md` gives the pretraining as 31,700 steps, 0.52B tokens, 93 per parameter, epoch 5, MLM loss 2.081; the `done:` line says 31,714. The x2 pilot and most of the pretraining ran while `auto-brake.sh` was down (incident of 13/11 below): `brake.log` has no change between 13/11 09:50:04 (`0.5 45:15`) and 20:30:37 (`0.85 50:10`).

### s6-cast-pico (`runs/all/cast-{pilot,train,export,eval}.log`)

| phase | budget | steps done | brake modes | cooling pauses |
|---|---:|---:|---|---|
| rate pilot 1e-3 | 3,000 steps | 3,000 | 0.85 50:10, 0.95 55:5 | none |
| rate pilot 3e-3 | 3,000 steps | 3,000 | 0.95 55:5 | none |
| rate pilot 1e-2 | 3,000 steps | 3,000 | 0.95 55:5 | 300 s |
| distillation pilot (3e-3, TinyCast teacher) | 3,000 steps | 3,000 | 0.95 55:5 | none |
| training | 480 min | 250,092 | 0.95 55:5, then 0.85 50:10 and 0.5 45:15 alternating (13 modes in all, ending at 0.5 45:15) | 2 × 300 s, 7 × 600 s, 4 × 900 s |

The training's state records `"elapsed_secs": 28800.07270748` (`runs/s6-cast-pico/last-ema/state.json`), its 480 working minutes, between 06:07:30 and 19:05:45. The pilot rules (`cast-pilot.log`, `runs/s6-cast-pilot/choice.json`): final f32 validation pinball 0.05121 (1e-3), 0.05089 (3e-3), 0.05788 (1e-2), so 3e-3 is kept; with the teacher 0.05076 against 0.050891 without, so the run distils from TinyCast.

## 3. Incidents

Every entry of `runs/all/incidents.txt`, with what was done (as that file records it):

| when | what happened | what was done |
|---|---|---|
| tranche 1 | two stops asked by the owner (evening pauses); one kill when an interactive session ended (10/11 12:28) | not recorded in `incidents.txt` |
| 11/11 04:24 | one calibration step failed with no message | passed on the automatic retry |
| 11/11 09:45 to the evening | the terminal lost its access to `~/Documents` (macOS); the recipe script gave up after 5 failures | started again; nothing lost |
| tranche 1 | two brake modes (amendment 6 of `docs/s6-prereg.md`); `train_minutes` of the daytime arms is noisy | recorded only |
| 12/11 | s6-ear decision training held 27 GB on the 16 GB Mac (audio frames all in memory) and ran at about half speed in swap: 31,480 steps in its 60 minutes | frames read from disk per batch; decision and control started over from the kept pretraining (48,626 steps); discarded files in `runs/s6-ear-nano-swap/`; amendment 4 of `docs/s6-ear-prereg.md`, before any evaluation. The decision training done again: 48,205 steps in 60 minutes |
| 12/11 19:00 to 19:04 | a low-priority code check of the s6-sleep script overlapped 4.5 working minutes of the s6-ear control run (a mistake in the waiting loop of the helper) | measured on the control's log: 509 questions/s during the overlap against 433 to 439 in the 8 minutes before, so no slowdown is visible |
| 12/11 | 40 tokenized-corpus caches of finished pilot and ablation runs deleted (35.5 GB) | checkpoints, logs and summaries kept |
| 13/11 00:16 | the memory cap stopped the decision training of the s6-nano rate pilot (x0.5) at 15 GB, after 8.2 of its 20 minutes (860 steps, in swap): 12% of the batches need 10 to 17 GB | `--step-mem-gb 3` (a batch cut into parts whose gradients are averaged: the same update), amendment 7 of `docs/s6-gen-prereg.md`, commit 499454c. On throwaway runs of 3 minutes: 280 steps before, 564 to 603 after. The 860 steps set aside in `runs/s6-nano-lr-swap/`; the x0.5 decision training started over from its kept pretraining. The run was down from 00:16 to 00:43 (`incidents.txt`); `train-all.log` records the relaunch at 00:56:06 |
| (note) | the pico pilot and ablation and s6-secrets-pico used the same kind of batches with a smaller model; their memory was not measured; s6-ear decision ran at 3.4 to 3.9 GB after its own fix | recorded only |
| 13/11 | a third brake mode at night (`1.0 55:5`, 23:00 to 08:00, Mac idle 30 minutes and nothing else working), at the owner's request | the memory cap now needs 4 readings in a row |
| 13/11 06:05 | the memory cap stopped the first decision training of s6-nano at step 16,020 (72.2 of 90 minutes): 14 GB for two minutes, MLX's own memory flat (peak 6.65 GB). Cause: MLX keeps one compiled graph per input shape | relaunched 06:07; resumed from the checkpoint of step 15,573 (70.1 working minutes); the 447 steps after it done again. The cache of compiled graphs emptied every 256 new shapes (amendment 8), in the binary built from the s6-micro pilot on; the first run of s6-nano and its rounds finished with the binary they started with |
| from 13/11 06:07 | (a record, not an incident) | `runs/all/mem.tsv`: the memory of the running `sky-train` every 30 s |
| 13/11 08:50 | the step memory estimate counted the exits of s6-micro (3) instead of its depth (6): steps of its pilot took up to 6.3 GB under a limit of 3 | fixed for the binary of the first run of s6-micro (amendment 9); the three pilot rates ran before the fix, with the same estimate; the process held 7.5 to 12 GB |
| 13/11, about 10:10 to 18:18 | `scripts/auto-brake.sh` was not running (it ended silently when a training process ended during one of its memory readings); the brake file kept `0.5 45:15` and the memory cap was not watched | fixed; the second and third pilot rates of s6-micro and the first 128 minutes of its pretraining ran throttled whatever the Mac did; budgets count working time, so only the duration changed |
| 13/11 22:59 to 14/11 00:20 | the Mac ran on battery (80% down to 47%) | the brake stayed at `0.5 45:15`, as it does on battery |
| 14/11 00:21 | the first decision training of s6-micro stopped by hand at step 27,280 (126.2 of 180 minutes): in swap since about minute 92 (110 questions per second instead of 205; heap 9.3 GB, 8.3 in swap). The compiled decision step leaves about 0.4 MB of heap at each compilation | the decision training runs uncompiled (amendment 11 of `docs/s6-gen-prereg.md`); resumed from the checkpoint of step 27,161 (125.8 minutes); the steps done in swap are kept, so the run has fewer steps than 180 full-speed minutes |

`incidents.txt` has no entry after 14/11 00:57. Two later facts, from other files: `brake.log` shows the Mac on battery from 14/11 18:29:26 (`0.5 45:15 (on battery)`) during the s6-cast training, and `cast-eval.log` shows 2,017 of the 371,330 GIFT-Eval windows, in 12 configurations, forecast by the seasonal naive because the model could not read their context (no observation in its last 2,048 points; `docs/SPEC-S6-CAST.md` §6), in each of the two passes (with and without flip).

## 4. The gates

| model | gate | measured | verdict | file read |
|---|---|---|---|---|
| s6-ear-nano | end of turn ≥ Smart Turn v3.2 on the same clips | 0.5681 (bar 0.8518) | fail | `runs/s6-ear-nano/eval/gate.json` |
| | keywords (12 classes) ≥ 0.90 | 0.8885 | fail | same |
| | zero-shot keywords en ≥ 0.20 | 0.2100 | pass | same |
| | zero-shot keywords fr ≥ 0.20 | 0.1330 | fail | same |
| s6-secrets-pico | secret > 0.977 | 0.9990 | pass | `runs/s6-secrets-pico/report.md` (GGUF, exit 3) |
| | service > 0.787 | 0.9795 | pass | same |
| s6-nano | `typed_decisions_bench` > 0.398, ECE ≤ 0.08, no game more than 5% below s3-pico | 0.480, ECE 0.0479, games at s3-pico's scores | pass at τ = 1; fails at the CRC thresholds (not kept); published at full depth | `runs/s6-nano/eval/results.md` |
| s6-micro | `typed_decisions_bench` ≥ 0.45, ECE ≤ 0.08, games | 0.547, ECE 0.0821 (τ = 1) and 0.0804 (CRC), games at s3-pico's scores | fail (ECE), not published | `runs/s6-micro/eval/results.md` |
| s6-cast-pico | nGMASE ≤ 0.838 and nWQL ≤ 0.587, without flip | nGMASE 0.9471, nWQL 0.6646 | fail, not published | `data/gift-eval-out/s6-cast-pico/summary.json` |

These are the verdicts of `runs/all/REPORT.md`; each was checked against the file in the last column. The full summary values for s6-cast are `"nGMASE": 0.9470969870574171` and `"nWQL": 0.6645693381662383`.

## 5. s6-cast-pico: where it stands

**Model.** 369,792 parameters, context 2,048 points, horizon 64 points, 4 exits (`cast-train.log`, `cast-export.log`). The pre-registration (§3) and `docs/SPEC-S6-CAST.md` §1 give 388,320; the trained config has `"parallax": false` and `"xsa": false` (`last-ema/state.json`), as the recipe chose (`recipe.json`). AdamW at 3e-3 (`"no_muon": true`, `args.json`), batch 256, QAT over the last 20%, EMA 0.999, seed 1, TinyCast teacher on 50% of the windows with a weight from 0.5 to 0. Final validation pinball (last exit, `cast-train.log`): ternary 0.0442 for `last`, 0.0431 for `last-ema`, so the EMA is exported (pre-registration §4; `cast-export.log`: `exporting runs/s6-cast-pico/last-ema`). The latent f32 weights score 0.5891 (`last`) and 0.5808 (EMA) on the same validation: the trained model is the ternary one. Ternary GGUF: 142,912 bytes (`models/s6-cast/s6-cast-pico.gguf`). Exit calibration (`cast-export.log`, α 0.05, δ 0.02, 2,000 windows): `exit_tolerance = 0.012839192`; at the 0.0128 row of its table the mean exits are 3.64 of 4.

**GIFT-Eval**, 97 configurations, every metric divided by the seasonal naive and averaged geometrically (`summary.json` of each folder):

| model | nGMASE | nWQL | nMSIS | file |
|---|---:|---:|---:|---|
| **s6-cast-pico, without flip (the reported model)** | **0.9471** | **0.6646** | 0.7190 | `data/gift-eval-out/s6-cast-pico/summary.json` |
| s6-cast-pico, with flip | 0.9099 | 0.6417 | 0.6860 | `data/gift-eval-out/s6-cast-pico-flip/summary.json` |
| gate (FLAIR, training-free, from the TinyCast paper) | 0.838 | 0.587 | | `docs/s6-cast-prereg.md` §6 |
| goal: TinyCast as published | 0.774 | 0.545 | | `docs/s6-cast-prereg.md` §6 |
| TinyCast, our port, with flip, same harness | 0.7856 | 0.5570 | 0.5781 | `data/gift-eval-out/tinycast/summary.json` |
| seasonal naive, our harness | 0.9999 | 0.8876 | 2.6778 | `data/gift-eval-out/seasonal-naive/summary.json` |

- **Gate: failed.** Above the FLAIR bar by 0.1091 on nGMASE and 0.0776 on nWQL. With flip it is still above both bars (0.9099, 0.6417).
- **Goal (beat TinyCast): not met.** Gap to the published TinyCast: +0.1731 nGMASE, +0.1196 nWQL. Gap to the generic TinyCast of our harness: +0.1615 and +0.1076.
- **Against the seasonal naive.** Below it on nGMASE (0.9471 < 1) and on nWQL (0.6646 against 0.8876 for our naive). Per configuration (ratio below 1): MASE in 52 of 97, WQL in 84 of 97. Our TinyCast port has a lower MASE than s6-cast in 86 configurations and a lower WQL in 87 (s6-cast is lower in 11 and 10).
- **Not in the files:** Timestral-5M on GIFT-Eval (the pre-registration asks for the gap to its leaderboard numbers, which are not in the repository), the latency of s6-cast, and the mean number of exits it used on GIFT-Eval (see §7).

**Per domain and per term.** Geometric means of the per-configuration ratios to `eval/gift-eval/seasonal_naive.csv`, the rule of `scripts/gift-eval.py summarize`, computed from the `all_results.csv` of each folder and from `eval/gift-eval/gift_eval_tinycast.csv` (TinyCast as published, per configuration). The same computation over all 97 configurations gives back each `summary.json`: 0.9471 / 0.6646, 0.9099 / 0.6417, 0.7856 / 0.5570, and 0.7738 / 0.5454 for the published TinyCast.

| domain | configs | s6-cast MASE | s6-cast WQL | s6-cast flip MASE | s6-cast flip WQL | TinyCast (ours) MASE | TinyCast (ours) WQL | TinyCast (published) MASE | TinyCast (published) WQL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Econ/Fin | 6 | 1.018 | 0.901 | 0.954 | 0.864 | 0.894 | 0.841 | 0.895 | 0.840 |
| Energy | 32 | 1.080 | 0.800 | 1.053 | 0.779 | 0.882 | 0.661 | 0.882 | 0.661 |
| Healthcare | 5 | 0.760 | 0.614 | 0.715 | 0.570 | 0.735 | 0.639 | 0.734 | 0.635 |
| Nature | 15 | 0.862 | 0.467 | 0.836 | 0.450 | 0.757 | 0.358 | 0.758 | 0.358 |
| Sales | 4 | 0.753 | 0.457 | 0.764 | 0.460 | 0.734 | 0.445 | 0.734 | 0.445 |
| Transport | 15 | 0.833 | 0.644 | 0.819 | 0.635 | 0.669 | 0.510 | 0.670 | 0.510 |
| Web/CloudOps | 20 | 0.983 | 0.662 | 0.901 | 0.624 | 0.750 | 0.562 | 0.696 | 0.509 |

| term | configs | s6-cast MASE | s6-cast WQL | s6-cast flip MASE | s6-cast flip WQL | TinyCast (ours) MASE | TinyCast (ours) WQL | TinyCast (published) MASE | TinyCast (published) WQL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| short | 55 | 0.860 | 0.638 | 0.840 | 0.622 | 0.756 | 0.571 | 0.755 | 0.570 |
| medium | 21 | 1.046 | 0.696 | 0.991 | 0.664 | 0.806 | 0.534 | 0.786 | 0.515 |
| long | 21 | 1.105 | 0.706 | 1.031 | 0.671 | 0.848 | 0.546 | 0.811 | 0.515 |

- It is worse than the seasonal naive on point forecasts in Econ/Fin and Energy, and on the medium and long terms.
- The only domain where it is close to TinyCast is Healthcare (and its WQL there, 0.614, is below TinyCast's 0.639 and 0.635), then Sales; the largest gaps are Transport, Web/CloudOps and Energy.
- The worst configurations by MASE ratio are `solar/10T/long` (2.729) and `solar/10T/medium` (2.282), where our TinyCast port is at 0.973 and 0.955; then `ett2/W/short` (1.831), `bizitobs_service/10S/long` (1.627) and `m4_hourly/H/short` (1.598).

## 6. s6-nano and s6-micro: reported numbers, not gates (amendment 12)

Amendment 12 of `docs/s6-gen-prereg.md` adds, as reported numbers and not gates, the ECE of each source and a 95% bootstrap interval of the pooled ECE (1,000 resamples of the items). Computed from the evaluation dumps with `scripts/s6-ece-by-source.py` (committed with this report):

```bash
python3 scripts/s6-ece-by-source.py --dump runs/s6-nano/eval/dump-s6-nano.jsonl --root .
python3 scripts/s6-ece-by-source.py --dump runs/s6-micro/eval/dump-s6-micro.jsonl --root .
```

The dumps hold the probabilities but not the gold answers, so the script reads the gold again from the five test files, in the order and with the filters of `sky-eval` (the first 2,000 test records per source), and checks each dump line against its record. ECE: top-label, 15 equal-width bins, as `sky-metrics` (`scores::ece`).

**The pooled values are reproduced exactly** before any per-source number is used: 0.047856566683135225 for s6-nano and 0.08209225064041939 for s6-micro, the values of `eval-s6-nano-full.json` and `eval-s6-micro-full.json` (0.0479 and 0.0821 in `results.md`), with the bins of `sky-metrics` and with those of `sky-eval` alike. The per-source ECEs and accuracies are also those of the two JSON files, which already contained them.

| source | n | s6-nano accuracy | s6-nano mean confidence | s6-nano ECE | s6-micro accuracy | s6-micro mean confidence | s6-micro ECE |
|---|---:|---:|---:|---:|---:|---:|---:|
| `typed_decisions_bench` | 2,000 | 0.4805 | 0.4605 | 0.0261 | 0.5470 | 0.4815 | 0.0660 |
| `zs_massive_en` | 2,000 | 0.4255 | 0.4088 | 0.0248 | 0.5180 | 0.3846 | 0.1334 |
| `zs_massive_fr` | 1,000 | 0.2390 | 0.3502 | 0.1112 | 0.2750 | 0.2856 | 0.0313 |
| `when2call` | 2,000 | 0.4035 | 0.5406 | 0.1412 | 0.4305 | 0.5372 | 0.1176 |
| `notam:deel_subject` | 2,000 | 0.0550 | 0.1296 | 0.0746 | 0.0520 | 0.1156 | 0.0636 |
| `notam:notam_multi` | 2,000 | 0.4935 | 0.5159 | 0.0710 | 0.3940 | 0.4979 | 0.2558 |
| **all (pooled)** | 11,000 | 0.3595 | | **0.0479** | 0.3780 | | **0.0821** |
| 95% bootstrap interval of the pooled ECE | | | | **[0.0416, 0.0564]** | | | **[0.0746, 0.0900]** |

The interval is a percentile bootstrap over the 11,000 items (1,000 resamples, seed 0, Python's `random`; quantiles interpolated as in `sky-metrics::bootstrap`), so its bounds are this script's, not those of `sky-metrics`' generator. The s6-micro interval contains the 0.08 guard; the gate is read on the point value, as pre-registered, and does not change.

## 7. Pre-registered report items that the run files do not contain

What the pre-registrations say the report will give and that the run files do not hold, each checked:

1. **The teacher's call and token counts** (`docs/s6-gen-prereg.md`, amendment 1). Not in any report. `runs/distill-glm.log` prints totals only for the labelling sessions that ended normally: nine `wrote …` lines, 25,072 calls, 126,349,485 input and 33,110,486 output tokens, for 480,647 labels. The sessions between 01/11 19:21 and 05/11 08:41 printed no total, so the full counts for the 747,077 labels are not in the files.
2. **Agreement per source** (amendment 1, `data/distill/agreement.json`). Not in any report. The file exists (05/11): 470 sources, 560,238 of 747,077 labels agree with the gold answer (0.7499 overall), from 0.145 (`open_jev:tile_platformer-v1`, 197 of 1,357) to 1.0 on a few small sources. It is a data file, not a run file, and is not reproduced here source by source.
3. **Texts refused by the moderation, per source** (amendment 3). Not in any report. `data/teacher/glm_labels.refused.jsonl` holds 121 texts, all `moderation (1301)`: `helpsteer2` 104, `pawsx_fr` 4, `tasksource:doc-nli` 2, `boolq` 2, and one each for `tasksource:hatexplain/majority`, `tasksource:HelpSteer3/feedback`, `helmo_synth`, `tasksource:args_me`, `typed_synth`, `pngwn_typed`, `mnli`, `tasksource:dynahate` and `tasksource:causal-news`. 747,198 − 121 = 747,077, the labelled count of `nano-glm.log`.
4. **s6-ear precision and recall of the end of turn, for the model and the control** (`docs/s6-ear-prereg.md` §6). Not computed by the run. Smart Turn v3.2 has them (precision 0.7832, recall 0.9713, `baseline-smart-turn.json`). `runs/s6-ear-nano/eval/probs-full.jsonl` holds per-item probabilities and gold answers for the model; there is no per-item file for the control.
5. **s6-cast latency** (`docs/s6-cast-prereg.md` §6): not measured by the run.
6. **The mean number of exits s6-cast used on GIFT-Eval** (same §6): not recorded. Only the calibration table on 2,000 synthetic windows gives a mean (3.64 of 4 at the chosen tolerance, §5).
7. **The gap of s6-cast to Timestral-5M** (same §6, "leaderboard numbers"): Timestral-5M's GIFT-Eval scores are not in the repository (`site/public/models/forecast.json` lists Timestral-5M without them).
8. **Which tranche 1 runs ran in which brake mode** (`docs/s6-prereg.md`, amendment 6). `runs/s6-recipe.log` prints no brake line, and `runs/brake.log` begins on 12/11 15:34:29, after tranche 1. Not in the files.
9. **Whether the GLM data `glm_orders` and `paraphrased` were present** (`docs/s6-prereg.md` §1, "noted in the final report"). No report states it. `runs/prepare-data.log` (25/10) says `glm_orders: no "data/teacher/orders_decisions.jsonl", skipped`, and `data/decisions/` holds no `glm_orders` or paraphrase file.
10. **s6-secrets validation accuracy per question** (`docs/s6-secrets-prereg.md` §6 asks for the validation accuracy): only the pooled value is in the training log (`val[ternary]` 0.998 at every exit, `runs/all/secrets-train.log`), as `BENCH.md` already says.

Found in the run files and not yet reported elsewhere, for completeness: the s6-ear routing threshold ρ with its coverage on validation (choice: τ 0.5, coverage 0.0401, error among answered 0.0030; noul: τ 0.74, coverage 0.3583, error 0.0490; `runs/s6-ear-nano/eval/crc.json`), its WASM parity on 20 test clips (max |Δp| 1.79e-5, argmax agreement 20 of 20, median 6,696 µs; `wasm.json`) and its native median latency (3,435 µs over all test sources; `eval-full.json`).
