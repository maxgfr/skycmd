---
title: Results
outline: [2, 3]
---

# Results: what works, what does not

Every model on this page was judged against criteria fixed before any test run, and failures are published like successes. The s6 family (November 2026) comes first, then s5 (October 2026).

## s6 (November 2026)

s6-nano and s6-micro were pre-registered in [`docs/s6-gen-prereg.md`](https://github.com/maxgfr/skycmd/blob/main/docs/s6-gen-prereg.md), s6-ear-nano in [`docs/s6-ear-prereg.md`](https://github.com/maxgfr/skycmd/blob/main/docs/s6-ear-prereg.md) and s6-secrets-pico in [`docs/s6-secrets-prereg.md`](https://github.com/maxgfr/skycmd/blob/main/docs/s6-secrets-prereg.md), all frozen before their training. Every amendment is dated, committed before the runs it concerns, and says what had been seen when it was written; none changes a gate. The runs were made in a throttled mode (a duty cycle and cooling pauses), so their budgets count working minutes; the steps actually done are in [`BENCH.md`](https://github.com/maxgfr/skycmd/blob/main/BENCH.md). s6-cast is still training.

### In short

- **s6-nano passes its gate** and is published, at full depth. 1,486,225 ternary parameters (715 KB): 0.480 on `typed_decisions_bench`, above the majority class (0.398) where s3-pico (0.382) and s4-micro (0.352 on the same records) stayed below it, with an ECE of 0.0479, while it still plays every game at the oracle's level.
- **s6-secrets-pico passes its gate** and is published, at full depth. 445,345 ternary parameters (250 KB): on a synthetic bench of file types it never saw, secret 0.9990 and service 0.9795, where a gitleaks-style rule scanner scores 0.977 and 0.787. Its line answer is weak (0.6859), and the bench says nothing about real leaks.
- **s6-micro fails on calibration only.** 0.547 on the bench (gate 0.45 and announced goal 0.50 both met), but ECE 0.0821 at full depth and 0.0804 at its conformal thresholds, above the 0.08 guard. It is not published.
- **s6-ear-nano fails three of its four gates.** End of turn 0.5681 against 0.8518 for Smart Turn v3.2, and its own control without the Mimi pretraining does better on all four sets. It is not published.
- **The s6 additions were not adopted.** The pre-registered pico ablation kept none of them: no Muon, no Parallax, no XSA, no SODA, no exit distillation, no EMA, no StableQAT, no curriculum. s6-nano is the s3 architecture at language scale, as in s4, trained with AdamW. What changed from s3-pico and s4-micro is the pretraining (0.87B tokens, 588 per parameter), the GLM-5.3-flash teacher, a larger tokenizer and wider windows. No ablation separates these causes, so none of them is claimed on its own.

### s6-nano: passed, published at full depth

Test sets, at most 2,000 questions per source, ternary GGUF of the last DAgger round, full depth; the published models on the same records:

| model | params | `typed_decisions_bench` | MASSIVE en | MASSIVE fr | When2Call | NOTAM deel_subject | NOTAM notam_multi | all | ECE (all) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **s6-nano** | 1,486,225 | **0.480** | 0.425 | 0.239 | 0.404 | 0.055 | 0.493 | 0.360 | **0.0479** |
| s6-micro (not published) | 5,587,201 | 0.547 | 0.518 | 0.275 | 0.430 | 0.052 | 0.394 | 0.378 | 0.0821 |
| s3-pico | 445,345 | 0.382 | 0.187 | 0.117 | 0.368 | 0.163 | 0.319 | 0.268 | 0.0629 |
| s4-micro | 5,587,201 | 0.352 | 0.335 | 0.240 | 0.399 | 0.065 | 0.298 | 0.285 | 0.0666 |
| s5-lite | 17,060,097 | 0.696 | 0.726 | 0.444 | 0.376 | 0.177 | 0.554 | 0.500 | 0.0328 |

- **The gate** (result): bench 0.480 > 0.398; no game more than 5 % below s3-pico; ECE 0.0479 ≤ 0.08. Published bench scores for comparison: s3-pico 0.382, s4-micro 0.343, oscar-1-17m 0.6775, s5-lite 0.696.
- **The games** (result): 30 episodes per game at full depth, on the same seeds as s3-pico. Flappy 200/200 pipes with 0 crashes, Rescue 100 %, Dodge 87 % (= oracle) with 23 reflex vetoes, Doom-like 100 % of levels and a kill rate of 0.772 (= oracle). These are s3-pico's scores, and the oracle's.
- **Early exit rejected** (result). The conformal thresholds (α = 0.02) cut the depth to 1.01 to 1.83 of 4 passes per game, but Rescue fell to 93 % and the Dodge vetoes rose from 23 to 1,016. The pre-registered rule allows 2 %, so the thresholds are not kept. The published file has its three exit thresholds set to 1: the calibration had left them at 0.5, a setting that was never evaluated.
- **Where it fails** (result). NOTAM subject 0.055, below s3-pico (0.163), and When2Call 0.404. The NOTAM sets are never trained on.
- **Not strictly zero-shot** (found afterwards). 206 of the 2,000 English MASSIVE test utterances also appear in the training split, 201 of them in a decision source derived from MASSIVE; none of the French ones does. The model was also trained on 3,113 questions of that source, MASSIVE intents asked with 3 options instead of 8, so the English MASSIVE score is partly in distribution. The seen utterances are not answered better: s6-nano gets 0.393 on the 206 and 0.429 on the 1,794 others, s6-micro 0.495 and 0.521; without them the ECE over all sources is 0.0486 and 0.0837, so neither gate changes ([amendment 14](https://github.com/maxgfr/skycmd/blob/main/docs/s6-gen-prereg.md)). Banking77, trained on, is reported apart: 0.631.
- **Truncation** (result): 1.4 % of the bench states are cut at 512 tokens. s3-pico's windows cut 96.8 % of the bench sequences ([chapter 12](/tutorial/12-size-ladder)).
- **In the browser** (result). WASM matches the native engine on the first 200 bench questions to 9.5e-7, with the same answer on all 200. A long JSON state of the bench takes about 15 ms per decision under Node, a Flappy frame 0.44 ms. Try it on [Live](/live/#s6-nano-the-questions-it-was-trained-for) and [Play](/play/).

### s6-secrets-pico: passed, published at full depth

It reads a short text (a config file, a `.env`, code) and answers four typed questions: is a usable secret exposed, which service issued it, what kind of value it is (secret, test key, publishable key, placeholder, not a credential), and on which line. The bench, `secret_bench`, has 1,996 texts in four file types the training never shows (docker-compose, toml, go, ini). Ternary GGUF, last exit, full depth (`runs/s6-secrets-pico/report.md`); the rule scanner on the same texts (`runs/s6-secrets-pico/baseline.txt`, measured before any training):

| question | s6-secrets-pico | rule scanner |
|---|---:|---:|
| secret: is a usable secret exposed? | **0.9990** | 0.977 |
| service: which service issued it? (15 options) | **0.9795** | 0.787 |
| kind (5 options) | 0.9990 | not asked |
| line | 0.6859 | not asked |

- **The gate** (result): secret 0.9990 > 0.977 and service 0.9795 > 0.787, ternary, full depth. The scanner combines gitleaks-style prefixes for 20 formats, connection strings with a password and an entropy rule on secret-sounding names; its 35 false positives are random values that are not credentials, and its 11 misses generic secrets. The model is right on the secret question for all 46 of those texts, and wrong on 2 texts that the scanner gets right, both an `x_api_key` with a hex value in an ini file, which it calls not a credential.
- **Line is weak** (result). 0.6859, and confident when wrong (ECE 0.3051, against 0.0009 for the secret question). On the 1,010 texts that do expose a secret it names the right line 383 times; the 986 texts without one, where `none` is right, carry the score. Line localisation does not work yet.
- **Where service fails** (result). 41 misses, 34 of them in Go code, where it names `database` for keys of other services.
- **Validation** (result): 0.998 on the seen templates (ternary, pooled over the four questions; per question not computed).
- **Early exit** (result). At the thresholds the calibration stored (0.5 / 0.9 / 0.5), every question stops after the first of 4 passes: secret and kind keep 0.9990, service drops to 0.9775 and line to 0.6227. The gate was read at full depth, so the published file has its thresholds set to 1, as for s6-nano.
- **Size and latency** (result): 445,345 parameters, 250,272 bytes. Median 786 to 789 µs per decision natively over the 7,984 bench questions, 1,420 to 1,427 µs in WASM under Node on the first 200, where WASM matches the native engine to 1.2e-6 with the same answer on all 200. These times were measured on the build machine (Apple M5) while a GPU training was running. Try it on [Live](/live/#s6-secrets-pico-is-a-secret-exposed-in-this-text).
- **What this bench does not say.** It is synthetic: the formats, the file templates and the class shares are our generator's, and the keys are fake, with random bodies. Beating the scanner here shows that the model reads formats and context in file types it never saw. It says nothing about real leaks in real repositories.

### s6-micro: failed on calibration only

Same evaluation, 5,587,201 parameters (the s4-micro shape). Bench 0.547, so the gate (≥ 0.45) and the announced goal (0.50, the level of s4's Qwen3.5-4B teacher) are both met. The games match s3-pico at full depth and at its conformal thresholds. The gate fails on the guard: ECE 0.0821 over all the test sources at full depth, 0.0804 at the conformal thresholds, both above 0.08. Its decision training was stopped once for a memory problem and resumed from its checkpoint ([amendment 11](https://github.com/maxgfr/skycmd/blob/main/docs/s6-gen-prereg.md)). From nano to micro, accuracy rises from 0.480 to 0.547 and the ECE from 0.0479 to 0.0821, the median native time per decision goes from 1,488 to 7,346 µs, and the file from 715 KB to 2.4 MB.

### s6-ear-nano: failed

Test split, ternary, full depth (`runs/s6-ear-nano/eval/gate.json`):

| gate | n | s6-ear-nano | control (no Mimi pretraining) | bar | verdict |
|---|---:|---:|---:|---:|---|
| end of turn | 31,527 | 0.5681 | 0.7647 | 0.8518 (Smart Turn v3.2, same clips) | fail |
| keywords (Speech Commands, 12 classes) | 4,890 | 0.8885 | 0.9217 | 0.90 | fail |
| zero-shot keywords, English | 1,000 | 0.2100 | 0.3020 | 0.20 (2 × chance) | pass |
| zero-shot keywords, French | 1,000 | 0.1330 | 0.1520 | 0.20 (2 × chance) | fail |

- **The pretraining did not help** in this run: the control, trained the same way from a random init, is more accurate on every set. The pretraining itself worked as a task (codebook-0 accuracy 0.2196 on masked val frames, 45 times its technical bar).
- **Size:** 1,532,449 parameters, 5.2 times fewer than Smart Turn v3.2. The verdicts are the same at the conformal thresholds.
- **Not reported:** the precision and recall of the end-of-turn answers, which the pre-registration asks for, were not computed by the run (Smart Turn v3.2: 0.783 and 0.971).

## s5 (October 2026)

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
