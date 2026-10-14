---
title: Results
outline: [2, 3]
---

# s5 results: what works, what does not

s5 is a small decision model (Ettin-17M for s5-lite, English) tested on three bets, each judged against criteria fixed before any test run. The [pre-registration](https://github.com/maxgfr/skycmd/blob/main/docs/s5-prereg.md) was committed first. The evaluation tool refuses the test split of a frozen set unless that file is committed and lists the set's hash, and it logs every test pass.

Failures are published like successes. "Not significant" never counts as a success; only non-inferiority within a margin fixed in advance does.

Every property below carries a status:

- **guaranteed**: holds by construction and is checked by an exact test (to 1e-6, natively and in WASM). It is not an experimental result.
- **learned**: trained for, measured on held-out data.
- **held out of training**: never in the loss or the targets, so a gain there is a real generalization.
- **result**: a measured cost or benefit.

## In short

All three pre-registered criteria were missed, and the reasons are instructive. The pre-registration was committed before any test pass, and so were the three amendments made during the night (each explains what had been seen when it was written). Every test pass is logged; the count is in the tables below.

**Bet 1 failed, on calibration and not on accuracy.**

- **Accuracy.** On the fitted bench, the iso model B\* is far more accurate than the same encoder in the Laya format A′: 0.674 against 0.429 (3 seeds each; +0.245, 95% CI [+0.136, +0.321]).
- **ECE.** Its top-label ECE is worse: 0.088 against 0.057 (+0.031, CI [+0.006, +0.054]), and the margin fixed in advance was +0.01.
- **Proper scores.** NLL and Brier, which also reward calibration, are both better for B\* (Holm-adjusted p = 0.003).
- **Cause, seen on validation.** B\* is underconfident on typed decisions: mean confidence 0.59 for an accuracy of 0.71. Its temperatures, fitted on a validation split dominated by other sources, do not correct that. A per-source calibration was not pre-registered.
- **Per-source calibration, tried afterwards (exploratory, amendment 5).** Temperatures refitted on the typed-decisions questions of val only, applied to the existing bench dumps, do not fix it. B\* gets worse (ECE 0.088 → 0.110, +0.022 [+0.006, +0.034]), A′ does not move (−0.006 [−0.027, +0.018]), and the ECE gap widens to +0.059 [+0.029, +0.084]. The refitted B\* temperatures are higher than the global ones (about 0.9 against 0.73 for choice questions), so they sharpen less. This does not change the verdict of bet 1.
- **The A′ caveat.** The A′ finalists were unstable at the learning rate chosen on short runs. Against A′ at the next lower rate (amendment 2), the conclusion is the same: accuracy +0.079 [+0.054, +0.106], ECE +0.020 [−0.001, +0.042].

**Bet 2 improved only the trained identities.** Because bet 1 failed, the pre-registered support was A′.

- With λ\* = 0.1, the partition and cross-format errors drop sharply: \|Σ s − 1\| goes from 0.79 to 0.12, and \|s − c\| from 0.19 to 0.04 (Holm p = 0.0025).
- MAD_neg (0.028) and the sum Σ s (0.972) meet their thresholds.
- The held-out identity, monotonicity, is not measurable. At λ = 0 the model is not competent on "x or y" questions: accuracy 0.528 [0.440, 0.588] against a majority rate of 0.559.

**Phase 5 failed for lack of power, not for a cost of quantization.**

- **Per seed**, q8_0 changes the ECE of B\* by −0.0024, −0.0003 and −0.0005, and its accuracy by at most 0.3 points.
- **Pooled**, the pre-registered two-level bootstrap resamples the seeds of each format independently. The seed-to-seed spread of the ECE (0.078 to 0.096) then makes the interval wider than the 0.01 margin.

**4-bit weights (Q4_0) cost accuracy on B\*.** This is a secondary result, not pre-registered, from the same analysis. The criterion of Phase 5 stays that of q8_0.

- **File size.** B\* takes 12 MB in Q4_0, against 21 MB in q8_0 and 70 MB in f32.
- **Accuracy.** Per seed, Q4_0 costs B\* 1.3, 1.3 and 0.65 points on the bench, against at most 0.3 for q8_0. A′ loses at most 1.1 points.
- **ECE.** Per seed, the ECE of B\* moves by −0.009, +0.018 and +0.002. Pooled, the difference is +0.004 [−0.022, +0.026] for B\* and −0.004 [−0.023, +0.016] for A′, so non-inferiority is not shown, as for q8_0.

**Phase 7 accepts nothing under its guarantee.** Escalation with a controlled error was tested offline.

- **Pre-registered calibration.** It uses the typed-decisions questions, exchangeable with the bench but only 214 to 275 per type. With them, Learn-then-Test certifies no acceptance threshold for any α ≤ 10 %.
- **Larger calibration, for the /live model.** With every English calibration question, which are not exchangeable with the bench, it would accept 11.5 % of the bench at α = 5 %. The error of those accepted answers is 7.0 %: the guarantee does not survive the shift.

**Bet 3, online escalation with an audit, fails in every cell.** It was run after the offline results, so it is exploratory (amendment 4). Its parameters were fixed in the code before its only run.

- **Setup.** The bench of the shipped model (A′, seed 0, q8_0) is replayed as 20 streams of 4000 questions, with a prevalence drift in the second half. The auditor sees every escalated answer and a share π of the accepted ones, drawn at random.
- **Threshold.** Each question type starts from a plug-in threshold set on the calibration split. That threshold then moves with the estimated error of the accepted answers, ê = err · O / P(O).
- **Result.** After the drift, the estimated error of the accepted answers is 0.18 to 0.21 at α = 5 % and 0.20 to 0.28 at α = 10 %, depending on π. The criterion was α + 0.01.
- **Why.** The model accepts only 3 % of the questions after the drift. With so few answers accepted, the threshold barely moves. The escalations stay at 1.01 to 1.02 times those of the fixed threshold, within the 1.2 budget, and nearly every question goes to the auditor (97 to 99 %).

**The s5-lite gates fail** for the pre-registered shipped model, which the rule made A′ (seed 0): fitted bench 0.541 and English demo cases 50 %.

**The /live model.** /live serves a longer run of B\* (8000 steps instead of 2000), added for the demo before its result and outside the bets (amendment 3). Its numbers are below and are not claims:

- 0.696 on the fitted bench, where oscar-1-17m scores 0.682 on the same records;
- 21 of the 30 English demo cases;
- four sets below their majority baseline;
- no answer flips when the keys are reassigned, as the construction guarantees, but 26 % flip when the definitions are paraphrased;
- on the original 2609.33209 battery, MAD_neg 0.045 (Jev: 0.064), Σ s 1.46 (Jev: 1.14) and \|s − c\| 0.158 (Jev: 0.091).

## What is guaranteed by construction (s5 iso architecture)

| property | how |
|---|---|
| permutation of the options changes nothing | each option sees the state, the question and itself only, with shared positions |
| renaming the option keys changes nothing (the yes/no swap of 2609.26758) | the keys are never rendered: the model reads the definitions only |
| s + n = 1 when a negation only swaps the two definitions | same reason |
| the state encoding does not depend on the question | the state never attends to the question or the options (cache per state) |

These are tests, not results. The training side is in `crates/sky-train/src/s5/iso.rs`. The served side is in `crates/sky-web/tests/iso.rs`, run natively and on wasm32 under Node in CI.

## Measured

<ClientOnly>
  <S5Results />
</ClientOnly>

## Not done yet

- **s5 in French** (mmBERT-small, Phase 6) was dropped: s5 stays a small English model, and the French demos fall back on the English presets. The SIB-200 fr and MTOP fr gates of that phase are moot.
- **Baselines B1 to B3** were not run. These deviations were declared in the pre-registration before any result.
  - B1 (label log-probability through the MLM head) has no inference path, and only ettin-17m has an MLM head, while the arms use ettin-reranker-17m.
  - B3 (a linear head on the frozen backbone) needs training.
  - B2 (a cascade) is not defined by the pre-registration. The cascade demo on [Live](/live/) stands in for it.
- The comparison with Jev on the original 2609.33209 battery was run afterwards, per run and per arm (three seeds each, table above). It is descriptive. Both arms are farther from additivity than Jev (Σ s 1.46 [1.42, 1.50] for A′, 1.24 [1.07, 1.36] for B\*, against 1.14). B\* is close to Jev on |s − c| (0.097 [0.050, 0.134] against 0.091) and worse on MAD_neg (0.166 [0.090, 0.280] against 0.064). A′ looks better on MAD_neg (0.031 [0.000, 0.065]) only because one of its seeds, f-ap-s2, is degenerate on this battery: it puts s and n on the same side for every item and its Σ s is a constant 1.5.
- Q4 quantization and the online escalation bet were run afterwards. Both are reported above, as secondary and exploratory results.
- Per-source calibration was run afterwards (amendment 5, exploratory). It is negative for B\* and reported above; the verdict of bet 1 does not change.
- The closed-loop calibration of early exits was rerun over 5 arena seeds ([What we bring](/novelty/), point 3, [`loopcal.json`](/results/loopcal.json)). It confirms the failure of s2-nano at the i.i.d. τ; s3-pico shows no τ gap.
- Still not done: a same-size MLP baseline on the games, s3 ablations over several seeds, a standard control benchmark with vector states, and a systematic literature review.
