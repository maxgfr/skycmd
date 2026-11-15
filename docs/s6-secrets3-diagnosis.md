# s6-secrets-verify-pico: diagnosis for a possible 2.1 (validation and training only)

Date: 15/11/2026. Read-only analysis. No training, no GPU, no test file. No file under `eval/test/` or `eval/issues/test/` was opened, no test meta was read, and no test number from `runs/s6-secrets2/eval-test/results.md` was used to choose anything. Test numbers appear below only as context, and they are marked as such. This report holds counts and structural descriptions only: no value and no line content.

**Sources.** All numbers come from these:

- the validation flags of every system (`eval/validation/flags/`);
- the model's per-candidate P(real) at full depth (`model.flags.jsonl`), which equals exit 4 of `model/creddata/model-exits.jsonl` on all 4,234 candidates;
- the per-exit probabilities;
- the meta rows of the 16 validation repositories, read through `common.open_split("validation")` and `scorer.load_labelled`;
- the issue-report validation flags (`eval/issues/validation/`);
- `build/counts/creddata.json`, `gen/recall.json`, `gen/misses-*.jsonl`, `build/logs/truncation.json`;
- `runs/s6-secrets-verify-pico/{decide/log.jsonl, decide.log, report.md}`.

**Method.** The line-level computation is a reimplementation of `scorer.score_system` at the unit level. It reproduces the scorer's validation numbers exactly: model gated 0.8603 (P 0.8790, R 0.8425), full 0.7733, without the largest repository 0.8000 and 0.7500; LR 0.8604; secretgate-default 0.5197; generator 0.4272.

**Per-record truncation.** I ran `sky-train s2-truncation` (CPU) on the validation records file with `--per-source N`, and bisected on N to locate the 18 truncated states. Nothing was written to disk.

**The limit that dominates everything below.** Validation is 16 repositories: 457 positive lines on the gated universe (522 T rows), 6 repositories with 20 or more T rows, and one repository (`7738e44d`) holding 165 of the 457 positive lines. Error counts per group are in the tens. A difference of 0.02 in pooled F1 moves about 10 lines. Every gain below sits next to its bootstrap interval over repositories where I computed one, and most intervals are wide.

---

## 1. Where the model loses F1 (validation, line level, gated universe, tau* = 0.2598)

457 positive lines and 1,963 negative ones. The model has TP 385, FP 53, FN 72, for F1 0.8603.

In every table below, "dF1 if fixed" is the pooled F1 gain if exactly those errors were removed (FN made TP, FP made TN), everything else unchanged.

### 1.1 By kind of error

| error | lines | dF1 if fixed |
|---|---:|---:|
| FN: a candidate overlaps the T value, the model rejects it | 48 | +0.0580 |
| FN: generator miss, no candidate on the line | 21 | +0.0261 |
| FN: candidates on the line, none overlaps the T value | 3 | +0.0038 |
| FP: F, whole-line row (CredData marks the whole line not-a-credential) | 24 | +0.0237 |
| FP: F, a value row | 21 | +0.0207 |
| FP: X (example, test value, placeholder) | 6 | +0.0058 |
| FP: D (the accepted candidate overlaps no labelled value) | 2 | +0.0019 |
| **all judge errors (48 FN + 53 FP)** | 101 | +0.110 (to a perfect judge, 0.9707) |
| **all generator-side FN (21 + 3)** | 24 | +0.030 |

The judge, not the generator, carries about 80% of the lost F1 on validation. On the full universe (UUID rows kept), generator misses dominate instead: 130 of the 187 FN, +0.139. That is the UUID decision of amendment 2, not something a 2.1 judge can change.

Two facts about the judge errors:

- **The rejected T lines are mostly near the threshold.** The median P(real) of the best T candidate is 0.098. 23 of the 48 lie in [0.1, tau*), and 13 are confidently wrong (P < 0.05).
- **The false-positive lines are mostly low-confidence too.** Their median P is 0.424, and only 4 of the 53 have P ≥ 0.9.

### 1.2 By group (columns: positive/negative lines; TP; FN split into generator miss / no T candidate / rejected; FP; F1 within the group; dF1 if the group's FN, FP, or both were fixed; then the size-matched LR and the perfect-judge ceiling in the same group)

**CredData category** (scorer's categories; a line can be in several):

| group | pos/neg | TP | FN miss / noT / rej | FP | group F1 | dF1 FN | dF1 FP | dF1 both | LR group F1 | ceiling F1 |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| other | 145/80 | 112 | 14 / 2 / 17 | 8 | 0.845 | +0.041 | +0.008 | +0.048 | 0.910 | 0.942 |
| Key | 48/1028 | 29 | 3 / 0 / 16 | 22 | 0.586 | +0.024 | +0.022 | +0.046 | 0.813 | 0.968 |
| Password | 135/282 | 112 | 4 / 2 / 17 | 11 | 0.868 | +0.029 | +0.011 | +0.039 | 0.746 | 0.973 |
| Auth | 47/181 | 43 | 1 / 0 / 3 | 9 | 0.869 | +0.005 | +0.009 | +0.014 | 0.907 | 0.967 |
| Token | 105/233 | 100 | 0 / 1 / 4 | 7 | 0.943 | +0.006 | +0.007 | +0.013 | 0.940 | 0.986 |
| API | 7/149 | 7 | 0 / 0 / 0 | 5 | 0.737 | 0 | +0.005 | +0.005 | 0.778 | 1.000 |
| Secret | 32/163 | 30 | 0 / 0 / 2 | 2 | 0.938 | +0.003 | +0.002 | +0.005 | 0.909 | 1.000 |

**Generator rule** (of the decisive candidate: the best T candidate on a positive line, the best candidate otherwise):

| group | pos/neg | TP | FN miss / noT / rej | FP | group F1 | dF1 both | LR group F1 | ceiling F1 |
|---|---|---:|---|---:|---:|---:|---:|---:|
| generic-keyword | 174/987 | 141 | 0 / 2 / 31 | 38 | 0.799 | **+0.079** | 0.769 | 0.988 |
| (no candidate) | 21/815 | 0 | 21 / 0 / 0 | 0 | 0 | +0.026 | 0 | 0 |
| generic-api-key | 73/123 | 67 | 0 / 1 / 5 | 7 | 0.912 | +0.014 | 0.930 | 0.993 |
| password-assignment | 77/25 | 73 | 0 / 0 / 4 | 6 | 0.936 | +0.011 | 0.945 | 1.000 |
| aws-access-token | 27/0 | 22 | 0 / 0 / 5 | 0 | 0.898 | +0.006 | 1.000 | 1.000 |
| hashicorp-tf-password | 2/0 | 0 | 0 / 0 / 2 | 0 | 0 | +0.003 | 1.000 | 1.000 |
| url-credentials | 9/13 | 9 | 0 | 2 | 0.900 | +0.002 | 0.667 | 1.000 |
| private-key | 7/0 | 6 | 0 / 0 / 1 | 0 | 0.923 | +0.001 | 1.000 | 1.000 |
| auth-scheme | 66/0 | 66 | 0 | 0 | 1.000 | 0 | 1.000 | 1.000 |

**Path scope:**

| group | pos/neg | TP | FN miss / noT / rej | FP | group F1 | dF1 FN | dF1 FP | dF1 both | LR group F1 | ceiling F1 |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| test | 208/811 | 153 | 19 / 3 / 33 | 40 | 0.763 | +0.066 | +0.040 | **+0.107** | 0.815 | 0.939 |
| other | 235/757 | 220 | 2 / 0 / 13 | 12 | 0.942 | +0.019 | +0.012 | +0.031 | 0.894 | 0.996 |
| src | 14/395 | 12 | 0 / 0 / 2 | 1 | 0.889 | +0.003 | +0.001 | +0.004 | 0.966 | 1.000 |

**Truncation at 320 tokens** (of the decisive candidate):

| group | pos/neg | TP | FN rej | FP | group F1 | dF1 both |
|---|---|---:|---:|---:|---:|---:|
| not truncated | 430/1144 | 382 | 45 (+3 noT) | 53 | 0.883 | +0.113 |
| truncated | 6/4 | 3 | 3 | 0 | 0.667 | +0.004 |
| no candidate | 21/815 | 0 | (21 miss) | 0 | 0 | +0.026 |

**Value length** (of the decisive candidate, else of the T row):

| group | pos/neg | TP | FN miss / noT / rej | FP | group F1 | dF1 both | LR group F1 | ceiling F1 |
|---|---|---:|---|---:|---:|---:|---:|---:|
| 17–32 | 95/184 | 68 | 14 / 0 / 13 | 8 | 0.795 | +0.041 | 0.867 | 0.920 |
| ≤ 8 | 73/465 | 62 | 2 / 0 / 9 | 21 | 0.795 | +0.035 | 0.632 | 0.972 |
| 33–64 | 148/99 | 131 | 2 / 1 / 14 | 8 | 0.913 | +0.029 | 0.956 | 0.990 |
| 9–16 | 111/362 | 99 | 2 / 2 / 8 | 7 | 0.912 | +0.022 | 0.847 | 0.982 |
| > 64 | 30/38 | 25 | 1 / 0 / 4 | 9 | 0.781 | +0.015 | 0.929 | 0.983 |

(The 815 negative lines with only whole-line F rows and no candidate are never flagged.)

**Whether secretgate's default engine flags the line:**

| group | pos/neg | TP | FN miss / noT / rej | FP | group F1 | dF1 both | LR group F1 |
|---|---|---:|---|---:|---:|---:|---:|
| default flags it | 165/13 | 141 | 0 / 0 / **24** | 6 | 0.904 | +0.036 | 0.952 |
| default does not | 292/1950 | 244 | 21 / 3 / 24 | 47 | 0.837 | +0.106 | 0.804 |

The model rejects 24 of the 165 positive lines that the default engine already flags with precision 0.927. The union of question 2 recovers exactly those.

**Structure of the errors** (decisive candidate's extension, scope and rule):

- **False positives:** `.ndjson`/test/generic-keyword 13, `.json`/test/generic-keyword 7, `.go`/other/generic-keyword 6, `.json`/test/password-assignment 4. 44 of the 53 trip none of the turned-off filters.
- **Rejected T:** `.go`/test/generic-keyword 11, `.yml`/other/generic-keyword 9, `.json`/test/aws-access-token 5, `.ndjson`/test/generic-keyword 5, `.rb`/test/generic-keyword 5.
- **File types absent from training:** `.ndjson` (836 validation candidates, 0 training states), `.tfstate` (50 and 0), `.bats` (14 and 0); `.tf` has 4 training states and no T.

**The three groups that cost the most F1** (fix-both dF1; the groups overlap):

1. **Path scope `test`: +0.107.** 55 FN (33 judge rejections, 19 generator misses) and 40 FP, group F1 0.763. The LR does better here (0.815).
2. **Rule `generic-keyword`: +0.079.** 31 rejections and 38 FP, group F1 0.799. Almost all of it is judge error; the ceiling is 0.988.
3. **One repository, `71455799` (all in test scope): +0.053.** 27 FN (10 misses, 17 rejected) and 20 FP. Model F1 0.605 there, LR 0.898. Its errors sit in `.go`, `.json` and `.tfstate` test files, under categories Key and other. It overlaps groups 1 and 2.

By category, the costliest are `other` (+0.048) and `Key` (+0.046; 1,028 negative lines, 22 FP, F1 0.586 against the LR's 0.813).

---

## 2. The system that ships: union with secretgate's default engine

- (a) is the model alone.
- (b) is the union: the line is flagged if the default engine flags it, or if the model accepts a candidate on it.
- (c) has the model judge only the candidates the default engine does not report.

**(b) and (c) are identical by construction on these files.** All 266 candidates marked `reported` lie on lines the default engine flags (249 have exactly a default span), so removing them from the model's input changes no line. Every number for (b) holds for (c).

### 2.1 CredData validation, line level

F1 at the stored tau* (no selection on validation), then the validation-optimal tau and its F1:

| universe / repositories | default alone | model @tau* | model best (tau) | union @tau* | union best (tau) | LR @0.155 | LR ∪ default @0.155 |
|---|---:|---:|---:|---:|---:|---:|---:|
| gated, all 16 | 0.5197 | 0.8603 | 0.8681 (0.349) | **0.8834** | 0.8963 (0.349) | 0.8604 | 0.8648 |
| gated, without `7738e44d` | 0.6136 | 0.8000 | 0.8116 (0.326) | **0.8386** | 0.8576 (0.349) | 0.7972 | 0.8049 |
| full, all 16 | 0.4348 | 0.7733 | 0.7768 (0.326) | **0.7962** | 0.8023 (0.326) | 0.7549 | 0.7596 |
| full, without `7738e44d` | 0.5378 | 0.7500 | 0.7590 (0.326) | **0.7873** | 0.8006 (0.326) | 0.7186 | 0.7264 |

Union precision and recall at tau* on gated, all 16: P 0.8721, R 0.8950. The union trades 0.007 of precision for 0.053 of recall.

Tau grid, gated universe (F1):

| tau | model, all | union, all | union P / R, all | model, w/o largest | union, w/o largest |
|---:|---:|---:|---|---:|---:|
| 0.05 | 0.8027 | 0.8075 | 0.710 / 0.937 | 0.7257 | 0.7341 |
| 0.10 | 0.8263 | 0.8350 | 0.765 / 0.919 | 0.7568 | 0.7715 |
| 0.15 | 0.8565 | 0.8737 | 0.842 / 0.908 | 0.7953 | 0.8237 |
| 0.20 | 0.8593 | 0.8773 | 0.856 / 0.899 | 0.7979 | 0.8282 |
| 0.2598 | 0.8603 | 0.8834 | 0.872 / 0.895 | 0.8000 | 0.8386 |
| 0.30 | 0.8623 | 0.8882 | 0.890 / 0.886 | 0.8058 | 0.8489 |
| 0.40 | 0.8638 | 0.8929 | 0.921 / 0.867 | 0.8052 | 0.8541 |
| 0.50 | 0.8510 | 0.8825 | 0.932 / 0.838 | 0.7915 | 0.8448 |
| 0.60 | 0.8170 | 0.8609 | 0.943 / 0.792 | 0.7571 | 0.8305 |
| 0.70 | 0.7990 | 0.8481 | 0.954 / 0.764 | 0.7458 | 0.8273 |
| 0.80 | 0.7408 | 0.8145 | 0.953 / 0.711 | 0.6828 | 0.8039 |
| 0.90 | 0.6638 | 0.7708 | 0.952 / 0.648 | 0.6071 | 0.7793 |

**Does the union beat the model alone?** Yes, on every reading of CredData validation:

- At the stored tau*: +0.023 (gated), +0.039 (gated without the largest), +0.023 (full), +0.037 (full without the largest).
- Each at its own validation-optimal tau: +0.028, +0.046, +0.026, +0.042.

The paired bootstrap over the 16 repositories (1,000 resamples, seed 1234, stored tau*) gives:

- gated: +0.0230 [−0.0058, +0.0893];
- gated without the largest: +0.0386 [−0.0089, +0.1074];
- full: +0.0229 [−0.0035, +0.0801].

All three intervals include 0. Of the 13 repositories with T lines, the union is better in 4, worse in 4 and equal in 5. The pooled gain comes from a few repositories.

The union also moves the optimal tau upwards (0.349). The engine's own hits cover the high-precision lines, so the model can afford to be stricter. Choosing tau on the union and on these 16 repositories is selection on validation, so the "best" columns are optimistic.

### 2.2 Issue-report validation, candidate level (5,432 candidates, 586 positive)

A candidate is flagged by the union when a default-engine finding overlaps it or the model's P ≥ tau.

| system | P / R / F1 at tau* 0.2598 | best F1 (tau) |
|---|---|---|
| secretgate default alone | 0.7379 / 0.6007 / 0.6623 | — |
| Gitleaks (G4's reference) | 0.8635 / 0.5614 / 0.6805 | — |
| model alone | 0.7423 / 0.9044 / **0.8154** | 0.8450 (0.454) |
| union | 0.6655 / 0.9403 / **0.7793** | 0.8109 (0.454) |

**On issue reports, the union is worse than the model alone: −0.036 at tau*, −0.034 at the optimum.** The default engine's precision there is 0.74: it hits 477 candidates, 125 of them negative, and the model correctly rejects 93 of those 125. On CredData the default engine's precision is 0.93. Read as G4, the union would still clear Gitleaks + 0.08 = 0.7605 on validation, but only by 0.019 instead of 0.055.

Separately, the model's best tau on issue reports (0.454) is far from tau* (0.260), which was chosen on CredData. One CredData-chosen tau serves the issue source badly.

---

## 3. Headroom

Ceilings with the frozen generator (line level). A "perfect judge" accepts exactly the candidates that overlap a T value. "Any candidate" counts a positive line as found if any candidate lies on it, with precision 1.

| universe / repositories | model | LR | perfect judge | perfect judge ∪ default | any candidate |
|---|---:|---:|---:|---:|---:|
| gated, all | 0.8603 | 0.8604 | **0.9707** (R 0.943) | 0.9567 | 0.9765 |
| gated, w/o largest | 0.8000 | 0.7972 | **0.9534** | 0.9317 | 0.9627 |
| full, all | 0.7733 | 0.7549 | 0.8641 | 0.8533 | 0.8740 |
| full, w/o largest | 0.7500 | 0.7186 | 0.8780 | 0.8602 | 0.8938 |

- **Room above the model.** With the frozen generator, a perfect judge leaves 0.11 F1 on the gated universe (0.15 without the largest repository). The generator caps that ceiling at 0.97, so it is not the binding limit on validation.
- **Model against the LR.** They are tied overall (0.8603 against 0.8604; 0.8000 against 0.7972 without the largest). By group they differ a lot (tables of §1.2):
  - the model is better on Password (0.868 against 0.746) and on values of 8 characters or fewer (0.795 against 0.632);
  - the LR is better on Key (0.813 against 0.586), on category `other` (0.910 against 0.845), on scope `test` (0.815 against 0.763) and on repository `71455799` (0.898 against 0.605).
- **Their errors are largely different.** Of 125 model error lines and 124 LR error lines, only 46 are shared. The mean of the two probabilities, read at 0.2, gives these line-F1 differences against the model at tau*:
  - **gated, all:** 0.8958, +0.0355 [+0.0109, +0.0896];
  - **gated, without the largest:** 0.8502, +0.0502 [+0.0136, +0.1000];
  - **with the default engine added (union):** 0.9022, +0.0418 [+0.0054, +0.1253]; that is +0.0188 [+0.0072, +0.0368] over the model's own union.

  These are the only gains in this report whose bootstrap interval excludes 0. The 0.2 is a round value between the two stored thresholds and was not tuned. On issue-report validation the same mean at 0.2 gives 0.8020, against 0.8154 for the model alone (0.857 at its own best tau of 0.38).
- **The error is concentrated in a few repositories.** Of the 125 line errors (72 FN, 53 FP), 47 are in `71455799`, 25 in `a09d9e50` and 21 in `d0125de6`. These three hold 74% of the errors and 215 of the 457 positive lines. The largest repository, `7738e44d`, has F1 0.966 and only 11 errors. Without the largest, F1 drops to 0.80 because the remaining repositories are where the errors are.

---

## 4. Training signals (`decide/log.jsonl`, `decide.log`, `report.md`)

- **Run.** 14,117 decision steps in 30.0 working minutes, batch 64. QAT starts at step 1,467, after a 10% float warm-up. The learning rate is flat at 1.0 until about step 11,000, then decays to 0.106 at the end (a cooldown over the last ~20%).
- **Passes over the data.** 14,117 × 64 = 903,488 items drawn with weights 0.45/0.35/0.20:
  - CredData: about 406.6k draws over 49,864 items, **≈ 8.2 passes**;
  - issue reports: about 316.2k over 40,057, **≈ 7.9 passes**;
  - synthetic: about 180.7k over 40,000, **≈ 4.5 passes**.

  The logged throughput (≈ 410 q/s × 1,800 s ≈ 740k) would give about 6.7 passes on CredData. I could not settle which count the trainer's q/s measures, so the true figure lies between 6.7 and 8.2.
- **Training loss was still falling.** In windows of about 700 steps, the mean loss goes 0.0583 (7.0–7.7k), 0.0540 (9.8–10.5k), 0.0460 (11.2–11.9k), 0.0394, 0.0354, 0.0301 (13.3–14.0k). Most of the late drop comes with the cooldown.
- **Validation had plateaued.** These readings come from the in-training validation sample, 4,000 records of which 1,010 are CredData, at the last exit, ternary. Accuracy over all sources at steps 10k to 14,117: 0.9587, 0.9563, 0.9487, 0.9635, 0.9643, 0.9637. The CredData part swings between 0.87 and 0.93 on 1,010 records, and CredData NLL swings between 0.25 and 0.48 with no trend. Overall validation NLL is 0.112 at step 10k and 0.144 at the end. The cooldown that lowered training loss did not lower validation loss.
- **Generalisation gap.** Training loss is about 0.03. Validation NLL is 0.16 on `creddata:real` (report, 1,589) and 0.14 over all sources. That pattern points to fitting the training repositories, not to too few steps.
- **Float against ternary.** The only meaningful float reading is pre-QAT, at step 1,467: 0.9373 over all, 0.8832 on CredData. Ternary had caught up by step 2,000 (0.9383 and 0.8891) and ended at 0.9637 and 0.9079. There is no ternary penalty. The final "latent f32" row (0.7712) is the QAT master weights evaluated unquantised and says nothing about a float model.
- **Exits.** On the full 12,538-question validation set (ternary), `creddata:real` accuracy is 0.9383, 0.9421, 0.9390 and 0.9396 at exits 1 to 4. At line level on CredData validation (gated):

  | exit | F1 at tau* | best F1 |
  |---:|---:|---:|
  | 1 | 0.8708 | 0.8742 |
  | 2 | 0.8685 | 0.8785 |
  | 3 | 0.8674 | 0.8699 |
  | 4 (shipped) | 0.8616 | 0.8684 |

  (For exit 4 the exits file gives 0.8616 where the flags give 0.8603. Its probabilities differ from the flag scores by less than 1e-4, which flips about one line at the threshold.) The three extra loops add nothing. The effective rank falls from 53.8 to 38.6 across the exits.
- **Reading.** There is no sign of under-training (validation flat over the last 4,000 steps, cooldown not helping it) and no ternary penalty. Depth beyond one block is unused. That argues against "more minutes" and against "bigger model" (nano) as 2.1 levers. The visible limit is generalisation across repositories and file formats (§1, §3), not capacity or steps.

---

## 5. Data levers visible without the test

- **Decontamination (`build/counts/creddata.json`).**
  - **Training:** it removed 260 T candidates (12.2% of 2,127), 519 F and 3 X. The weak-negative pool lost 913.
  - **Validation:** it removed 75 T (16.3% of 459), 114 F and 1 X.
  - **Which T lines went:** of the 260, 236 have fewer than 20 characters around the value (0: 41; 1–4: 16; 5–9: 20; 10–19: 179; 20 or more: 4). These are short, generic assignment lines whose shape (value replaced by its length) occurs in some test line.
  - **Effect on tau:** tau* was chosen on the 1,589 decontaminated validation candidates, while the evaluation reads all 1,779.
- **Generator misses in training (`gen/recall.json`, `gen/misses-training.jsonl`).** On the gated universe, 290 of 2,452 training T rows have no candidate (recall 0.8817).
  - **By category:** AWS S3 Bucket 66/66, Password 36/898, Firebase Domain 32/33, Twilio Credentials 30/30, NKEY Seed 24/25, Auth 14/212, Token 13/270, AWS Client ID 13/103, Secret 12/174, SQL Password 11/14, CMD Password 9/17, CMD ConvertTo-SecureString 9/9, Credential 7/75, Nonce 6/38, MailGun API Key 6/6.
  - **By form:** no keyword before the value on its line or the previous one, 130; a name without a listed keyword, 45; no separator, 21; call argument, 20; separator not in the list, 18.
  - **Outside the gated universe:** 1,007 more misses are rows of category exactly UUID.
  - **On validation:** 78 gated T rows are missed, 55 of them Token and 52 of them later elements of a list. At line level that is only 21 lines with no candidate (+0.026 if fixed), because most list misses share a line with a found first element.
- **Truncation at 320 tokens.**
  - **Validation:** 13 of 1,779 labelled candidates (0.73%) and 5 of 2,455 unlabelled ones are truncated. Among the model's 115 candidate-level errors, 4 are truncated (3.5%; the LR has 2 of its 115). At line level, 3 of 125 error lines have a truncated decisive candidate, against 10 of 2,420 lines overall. Truncated states are over-represented among errors about fivefold, but they cost at most +0.004 F1.
  - **Training:** 2,547 of 19,946 labelled CredData training states (12.8%) are truncated, and 115 of 4,986 weak negatives. So **truncation is a training-side phenomenon**: one training state in eight loses its neighbouring lines, against one in 140 on validation. I could not attribute it to repositories without a per-record pass over the training records (not done).
- **Format coverage.** `.ndjson`, `.tfstate` and `.bats` have no training state at all. `.ndjson` alone gives 13 FP and 5 FN lines on validation. Test scope is 19.7% of the training states.

---

## 6. What I could not determine

- **The test.** Whether any validation group effect holds there is unknown by design. The 16 validation repositories have 6 with 20 or more T rows. The test-side distribution of file formats, scopes and repository sizes was not read.
- **The union on the clean-text corpus (G7).** It is not computed here. By construction it is at most the model's 31 files plus the default engine's 11, so ≤ 0.48% of 8,825 files.
- **ECE for the union or the ensemble.** Not computed. A union adds unscored rule hits, so G5 would have to be read on the model's own candidates or redefined.
- **The exact number of passes:** between 6.7 and 8.2, depending on what q/s counts (§4).
- **Truncation by training repository**, and whether re-windowing training states would change anything at validation (where truncation is rare).
- **Whether the ensemble's gain survives a tau chosen once for both sources.** It is lower than the model on issue-report validation at 0.2.

---

## 7. Ranked changes for a 2.1

All are judged on the same test, as a disclosed second look. The model's and the LR's test scores are already known, which matters most for changes 1 and 2: they reuse those artefacts.

**1. Gate the system that ships: the union of secretgate's default engine and the model, with tau re-chosen on validation for the union.**

- **What changes:** the gated system and its tau. No retraining. The integration rule (the model adds, never removes) becomes the measured unit.
- **Evidence:** +0.023 / +0.039 / +0.023 / +0.037 F1 at the stored tau* (gated, gated without the largest, full, full without the largest); +0.028 to +0.046 at the union's own validation tau (0.349). The model rejects 24 of 165 default-engine true-positive lines.
- **Expected size and how sure:** +0.02 to +0.04 on CredData. Uncertain: the bootstrap intervals include 0 and only 4 of 13 repositories gain.
- **Cost:** about 30 lines in the scorer and the gate (a union system, tau selection on it). No training minutes.
- **Risk:**
  - it lowers issue-report F1 by 0.036 on validation (G4 margin falls from 0.055 to 0.019);
  - G5 and G7 must be read on a mixed system;
  - it moves the question from "the model against the rules" to "the rules plus the model against the rules", which has to be said plainly.

**2. Ship the mean of the model and the size-matched LR (about 506 KB), inside the union of change 1.**

- **What changes:** the decision is the mean of two existing P(real); its tau is chosen on validation. No retraining.
- **Evidence:** errors are largely disjoint (46 shared of about 125). Gains over the model at tau*: +0.0355 [+0.011, +0.090] alone, +0.050 without the largest; +0.019 [+0.007, +0.037] over the union when combined with it (0.9022 gated). These are the only intervals that exclude 0.
- **Expected size and how sure:** +0.02 to +0.04 over change 1 alone on CredData. Moderately sure on validation.
- **Cost:** the LR's int8 hashed n-gram scorer has to be ported to the secretgate host (TypeScript), with a fixture. About a day of code; no training.
- **Risk:**
  - it is no longer a 250 KB model, so G3's "size-matched" premise breaks (it would need a 512 KB LR to mean anything);
  - on issue-report validation it is below the model at 0.2 (0.802 against 0.815);
  - G7 rises toward the LR's 0.87% clean-file rate (not computed);
  - both components' test scores are already known.

**3. Retrain with a data fix aimed at the measured failure groups.**

- **What changes:**
  - synthetic candidates in the formats the training never sees (`.ndjson`, `.tfstate`, `.bats`, `.tf`), with hard negatives for `generic-keyword` in test scope;
  - a narrower decontamination (exact normalised line, the value hashed rather than replaced by its length), which would bring back up to 260 T and 519 F training candidates;
  - pre-registered before any run.
- **Evidence:**
  - scope `test` costs +0.107, `generic-keyword` +0.079, and `71455799` +0.053 (model 0.605 against LR 0.898);
  - `.ndjson` (0 training states) alone gives 18 error lines;
  - decontamination took 12.2% of the T training candidates, mostly short generic lines.
- **Expected size and how sure:** +0.00 to +0.03, low confidence. The groups are carried by three validation repositories, and tailoring data to them risks fitting validation, not the test's 221 repositories.
- **Cost:** synthetic generator code in `sky-data secret-cand` and a rebuild (about half a day); 12 + 30 training minutes, unchanged.
- **Risk:**
  - training noise of the same order as the effect (one seed);
  - a laxer decontamination admits line shapes that occur in the test.

**4. A second generator (`sg-cand-2`): later list elements, keyword-less names, call arguments.**

- **What changes:** the candidate mode; then a rebuild and a retrain.
- **Evidence:** 21 positive lines with no candidate (+0.026) and 3 with no T candidate (+0.004) on validation. In training, 290 gated misses, led by keyword-less forms (130) and provider forms with no keyword (S3 bucket, Firebase, Twilio, NKEY).
- **Expected size and how sure:** at most +0.03 line F1 on validation, probably less: new rules add candidates in clean files (G7) and new negatives.
- **Cost:** secretgate code, a new freeze, a full rebuild and a retrain.
- **Risk:** the pre-registration forbade revising `sg-cand-1` after a test read. A new generator designed after the test was opened is the weakest kind of second look.

**Not recommended, with the evidence:**

- **more training minutes:** validation was flat over the last 4,000 steps and the cooldown did not help it;
- **the nano graph or a larger window:** exits 1 to 3 already match or beat exit 4; truncation costs at most 0.004 on validation;
- **re-choosing tau alone on the line unit:** +0.008 (0.8681 at 0.349).

---

## 8. Does the evidence support running a 2.1?

**Only a narrow one, and only as a measurement of the shipped system, not as a retrained model.** On validation:

- the cheapest change (the union, change 1) gains about +0.02 to +0.04, with intervals that include 0;
- the only change whose gain clears its interval (the LR mean, change 2) gives up the size-matched premise that justified the model;
- the training logs show no under-training or capacity signal that a retrain with more steps or parameters would address;
- the remaining error is concentrated in three of 16 repositories and in file formats the training never saw, which is a data-coverage problem that validation is too small to tune for safely.

For context only, not used for any choice: the test shortfall on G1 was 0.027 F1, the same order as these validation gains and as their uncertainty.

A 2.1 that pre-registers "union of default engine and model, tau chosen on validation, read once on the same test, disclosed as a second look" is defensible as a description of what secretgate would actually ship. I would not expect it to settle G1 either way with confidence. A retrained 2.1 (changes 3 and 4) is not supported by this evidence.
