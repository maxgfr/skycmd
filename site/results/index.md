---
title: Results
outline: [2, 3]
---

# Results: what works, what does not

Every model on this page was judged against criteria fixed before any test run, and failures are published like successes. The s6 family (November 2026) comes first, then the System 1 / System 2 router of LFM3-1.7B (November 2026), then s5 (October 2026).

## Status at a glance {#status}

One row per experiment: its verdict and the gated number against its bar. Each name links to its section, which gives every number. What each model is for and where to try it: [Models](/models/).

| model | verdict | gated number against its bar |
|---|---|---|
| [s6-ear2-nano](#s6-ear2-nano-passed) | passed | accuracy 0.8993 against 0.85 (Smart Turn v3.2 CPU 0.9260 on the same clips); false cut-off 0.0847 against 0.20; 2.39 ms per decision in WASM against 5 ms |
| [s6-route](#s6-route-passed) | passed, on fresh slates; system defined after v2's test was read | 0.8868 on unseen tools, above BM25 (0.8169, lower bound of the gain above 0) and at least MiniLM − 0.03 (0.8748) |
| [s6-nano](#s6-nano-passed-published-at-full-depth) | passed | `typed_decisions_bench` 0.480 against 0.398; ECE 0.0479 against at most 0.08 |
| [s6-micro v2](#s6-micro-v2-passed-on-a-second-look) | passed, on a second look | 0.553 against 0.45; ECE 0.0588 against at most 0.08 |
| [s6-micro v1](#s6-micro-v1-failed-on-calibration-only) | failed (calibration guard) | ECE 0.0821 against at most 0.08 (bench 0.547 against 0.45) |
| [s6-secrets-pico](#s6-secrets-pico-passed-published-at-full-depth) | passed on its synthetic bench; fails on real text | secret 0.9990 against 0.977, service 0.9795 against 0.787 |
| [s6-secrets-verify-pico](#s6-secrets-verify-pico-failed) | failed (G1; six gates of seven hold) | line F1 0.8561 against 0.8832 (0.7332 + 0.15) |
| [s6-route-nano](#s6-route-nano-failed) | failed (G1, G4) | 0.8395 against BM25 0.8500; 38.9 ms per request against 2 ms |
| [s6-ear-nano](#s6-ear-nano-failed) | failed (three gates of four) | end of turn 0.5681 against 0.9263 |
| [s6-cast-pico](#s6-cast-pico-failed) | failed | nGMASE 0.9471 against 0.838; nWQL 0.6646 against 0.587 |
| [s6-sleep3-nano](#s6-sleep3-nano-failed) | failed (H1, H2 and the s6-nano gate; H0 and H3 hold) | H0: `recall` +0.2452 [+0.2370, +0.2532] against at least +0.10 (holds); H1: sleep +0.0030 [−0.0010, +0.0068], a lower bound above 0 asked; H2: Flappy 15.87 against the control's 71.05, at most 2% under asked |
| [s6-sleep v1](#s6-sleep-stopped-by-its-pilot) | stopped by its pilot | `recall` 0.2460 against the control's 0.2650 + 0.10 |
| [s6-cast v2](#in-short) | pre-registered, not run yet | no result yet |
| [LFM3-1.7B System 1 / System 2](#lfm3-s1s2) | failed | H3: 1,032 tokens against at most 0.5 × 958 |
| [s5 (three bets)](#s5-october-2026) | failed | bet 1: ECE 0.088 against 0.057 + 0.01 |


## s6 (November 2026)

s6-nano and s6-micro were pre-registered in [`docs/s6-gen-prereg.md`](https://github.com/maxgfr/skycmd/blob/main/docs/s6-gen-prereg.md), s6-ear-nano in [`docs/s6-ear-prereg.md`](https://github.com/maxgfr/skycmd/blob/main/docs/s6-ear-prereg.md), s6-secrets-pico in [`docs/s6-secrets-prereg.md`](https://github.com/maxgfr/skycmd/blob/main/docs/s6-secrets-prereg.md) and s6-cast-pico in [`docs/s6-cast-prereg.md`](https://github.com/maxgfr/skycmd/blob/main/docs/s6-cast-prereg.md), all frozen before their training. Every amendment is dated, committed before the runs it concerns, and says what had been seen when it was written; none changes a gate. The runs were made in a throttled mode (a duty cycle and cooling pauses), so their budgets count working minutes; the steps actually done are in [`BENCH.md`](https://github.com/maxgfr/skycmd/blob/main/BENCH.md). The five models were trained in one run, from 12 to 14 November; its final report, with every phase, incident and gate, is [`docs/big-run-report.md`](https://github.com/maxgfr/skycmd/blob/main/docs/big-run-report.md). Two experiments followed on the evening of 14 November, each with its own pre-registration: s6-secrets-verify-pico, the second version of s6-secrets ([`docs/s6-secrets2-prereg.md`](https://github.com/maxgfr/skycmd/blob/main/docs/s6-secrets2-prereg.md)), and the pilot of s6-sleep ([`docs/s6-sleep-prereg.md`](https://github.com/maxgfr/skycmd/blob/main/docs/s6-sleep-prereg.md)). On 16 November came s6-route-nano, the second version of s6-nano as a decision model, narrowed to tool routing ([`docs/s6-nano2-prereg.md`](https://github.com/maxgfr/skycmd/blob/main/docs/s6-nano2-prereg.md)), and then s6-route, its file fused with a static embedding, defined after its test was read and judged on fresh slates ([`docs/s6-nano3-prereg.md`](https://github.com/maxgfr/skycmd/blob/main/docs/s6-nano3-prereg.md)). On 17 November came s6-ear2-nano (s6-ear v2), a dedicated end-of-turn student of Smart Turn ([`docs/s6-ear2-prereg.md`](https://github.com/maxgfr/skycmd/blob/main/docs/s6-ear2-prereg.md)), which passed its gate, and the test reading of s6-sleep v3, a fast-weight memory at the nano size ([`docs/s6-sleep3-prereg.md`](https://github.com/maxgfr/skycmd/blob/main/docs/s6-sleep3-prereg.md)), which failed its gate.

### In short

- **s6-ear2-nano (s6-ear v2) passes its gate.** A dedicated end-of-turn student of Smart Turn v3.2, trained for 240 working minutes on one task after v1's failure: 1,306,178 ternary parameters, 632 KB, pitch and energy in its input, the last 4 s of audio. On the 31,527 test clips of Smart Turn v3.2 it answers 0.8993 (bar 0.85; the teacher 0.9260, v1 0.5681, the silence timer 0.4979), with a false cut-off of 0.0847 and an END recall of 0.8627 at its validation threshold, in 2.39 ms per decision in WASM from a 632 KB file (the teacher: 8.7 MB, 31.7 ms per decision natively). On real conversational speech (eot-bench, which the design never read) it and its teacher are far weaker, and at its threshold it answers "finished" on about half of the real ends of turn ([below](#s6-ear2-nano-passed)).
- **s6-nano passes its gate** and is published, at full depth. 1,486,225 ternary parameters (715 KB): 0.480 on `typed_decisions_bench`, above the majority class (0.398) where s3-pico (0.382) and s4-micro (0.352 on the same records) stayed below it, with an ECE of 0.0479, while it still plays every game at the oracle's level. A constant answer per task passes that gate too: answering each question with its most common training answer scores 0.4815 on the bench, and on 17 of its 20 tasks s6-nano gives one answer to at least 95% of the items ([below](#s6-nano-what-the-gate-did-not-measure)).
- **s6-secrets-pico passes its gate** and is published, at full depth. 445,345 ternary parameters (250 KB): on a synthetic bench of file types it never saw, secret 0.9990 and service 0.9795, where a gitleaks-style rule scanner scores 0.977 and 0.787. Its line answer is weak (0.6859). **On real text it fails:** measured after publication on 719 real files with no secret, it flags 59.8% of them at p ≥ 0.9 and 49.4% at p ≥ 0.999. It stays published as the result of its pre-registration, not as a scanner ([below](#s6-secrets-real-text)).
- **s6-micro: the first version fails on calibration only, the second passes on a second look.** v1: 0.547 on the bench (gate 0.45 and announced goal 0.50 both met), but ECE 0.0821 at full depth and 0.0804 at its conformal thresholds, above the 0.08 guard. It failed its gate; its file is published as a result (owner's decision of 15/11/2026), as s6-micro-v1. v2, designed after that reading (twice the decision-training minutes, everything else the same) and judged on the same test records: 0.553, ECE 0.0588, the games at s3-pico's scores. It passed, and is published at full depth as s6-micro; its pass is a second look at these test sets, not a first one ([below](#s6-micro-v2-passed-on-a-second-look)).
- **s6-ear-nano fails three of its four gates.** End of turn 0.5681 against 0.9263 for Smart Turn v3.2 (Pipecat's report; the v1 harness read 0.8518, which padded short clips on the wrong side), and its own control without the Mimi pretraining does better on all four sets. It failed its gate; its file is published as a result (owner's decision of 15/11/2026).
- **s6-cast-pico fails its gate on both metrics.** On GIFT-Eval it scores nGMASE 0.9471 and nWQL 0.6646, against bars of 0.838 and 0.587 (the training-free FLAIR) and TinyCast's 0.774 / 0.545. It failed its gate; its file is published as a result (owner's decision of 15/11/2026) ([below](#s6-cast-pico-failed)).
- **s6-secrets-verify-pico (s6-secrets v2) fails its gate on G1.** A rule proposes candidates and a 250 KB ternary model judges each one in its context, trained and tested on real repositories. On 221 test repositories of CredData its line F1 is 0.8561, against 0.7332 for the best rule scanner: +0.123 where +0.15 was asked. The six other gates hold, among them 0.35% of clean real files flagged, where v1 flagged 59.8% of a 719-file subset. It failed its gate; its file is published as a result (owner's decision of 15/11/2026), and since version 1.9.0 secretgate offers it as an experimental option, off by default, where it can only add redactions ([below](#s6-secrets-verify-pico-failed)). Read again with secretgate's default rules (v3, a second look at the same test files, the system defined after the first reading), it fails G1 again: 0.8477 against 0.7332 + 0.15; the work on this model ends.
- **s6-sleep v1 stops at its pilot.** With a fast-weight memory, s6-pico scores 0.2460 on the `recall` probe, against 0.2650 for the same model trained without memory; the rule asked for +0.10. So no nano run took place ([below](#s6-sleep-stopped-by-its-pilot)).
- **s6-sleep3-nano (s6-sleep v3) fails its gate; its memory is used.** v2 ended at a CPU check; v3 holds the memory's gates and adds an auxiliary loss in training. On the `recall` test, s6-nano with the memory is right on 0.4892 of the questions, against 0.2440 for the same training without one (+0.2452 [+0.2370, +0.2532]): the memory is used (H0), and it fades with the delay, from 0.6147 at 1 to 4 frames to 0.3302 at 17 to 32. Four sleep passes add +0.0030 [−0.0010, +0.0068] (H1 not shown), and it plays Flappy (15.87 against 71.05) and Dodge (5.89 against 7.94) worse than the control (H2 not shown); its file also fails the s6-nano gate on those two games. It failed its gate; its file is published as a result (owner's decision of 15/11/2026), with its control, and the [Memory](/memory/) page plays `recall` with both ([below](#s6-sleep3-nano-failed)).
- **s6-route-nano (s6-nano v2) fails its gate on G1 and G4.** A tool router: a request, up to 7 tools described in one sentence each and *none*, with each tool's BM25 score against the request in its state. On 2,000 test slates of tools never seen in training it picks the right option on 0.8395, above potion-base-2M with cosine (0.6675) and both trained baselines of its size, but not above BM25 (0.8500), which its gate asked it to beat; and it takes 38.9 ms per request in WASM where the gate asked for 2 ms. With every tool renamed it falls to chance (0.1050). It failed its gate; its file is published as a result (owner's decision of 15/11/2026), with no playground on this site; its Hugging Face Space runs the file alone, since every published model gets a Space (owner's decision of 17/11/2026) ([below](#s6-route-nano-failed)).
- **s6-route (s6-nano v3) passes its gate on fresh slates; the system was defined after v2's test was read.** No training: v2's file, unchanged, fused option by option with potion-base-2M packed to 4 bits; 3.7 MB in two files. On 945 fresh requests about tools never seen in training it picks the right option on 0.8868, against BM25 0.8169 and all-MiniLM-L6-v2 (91.6 MB) 0.9048, in 30.9 ms per request in WASM. Every condition held. Without the three largest clusters of the test its advantage over BM25 holds zero in its interval (+0.0202 [−0.0062, +0.0481]), and with every tool renamed it falls to 0.2497: it needs distinctive tool names. Published, with a [playground](/router/) ([below](#s6-route-passed)).
- **s6-cast v2 (s6-cast2-pico): pre-registered, not run yet** ([`docs/s6-cast2-prereg.md`](https://github.com/maxgfr/skycmd/blob/main/docs/s6-cast2-prereg.md), amendments 1 and 2: computed periodicity, a seasonal draft, USCRN sub-hourly series; gates G1 and G2 on GIFT-Eval, Chronos-ZS reported). No result yet.
- **The s6 additions were not adopted.** The pre-registered pico ablation kept none of them: no Muon, no Parallax, no XSA, no SODA, no exit distillation, no EMA, no StableQAT, no curriculum. s6-nano is the s3 architecture at language scale, as in s4, trained with AdamW. What changed from s3-pico and s4-micro is the pretraining (0.87B tokens, 588 per parameter), the GLM-5.3-flash teacher, a larger tokenizer and wider windows. No ablation separates these causes, so none of them is claimed on its own.

### s6-nano: passed, published at full depth

Test sets, at most 2,000 questions per source, ternary GGUF of the last DAgger round, full depth; the published models on the same records:

| model | params | `typed_decisions_bench` | MASSIVE en | MASSIVE fr | When2Call | NOTAM deel_subject | NOTAM notam_multi | all | ECE (all) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **s6-nano** | 1,486,225 | **0.480** | 0.425 | 0.239 | 0.404 | 0.055 | 0.493 | 0.360 | **0.0479** |
| s6-micro (v2: passed on a second look) | 5,587,201 | 0.553 | 0.531 | 0.314 | 0.401 | 0.092 | 0.353 | 0.380 | 0.0588 |
| s6-micro-v1 (failed its gate; published as a result) | 5,587,201 | 0.547 | 0.518 | 0.275 | 0.430 | 0.052 | 0.394 | 0.378 | 0.0821 |
| s3-pico | 445,345 | 0.382 | 0.187 | 0.117 | 0.368 | 0.163 | 0.319 | 0.268 | 0.0629 |
| s4-micro | 5,587,201 | 0.352 | 0.335 | 0.240 | 0.399 | 0.065 | 0.298 | 0.285 | 0.0666 |
| s5-lite | 17,060,097 | 0.696 | 0.726 | 0.444 | 0.376 | 0.177 | 0.554 | 0.500 | 0.0328 |

- **The gate** (result): bench 0.480 > 0.398; no game more than 5 % below s3-pico; ECE 0.0479 ≤ 0.08. Published bench scores for comparison: s3-pico 0.382, s4-micro 0.343, oscar-1-17m 0.6775, s5-lite 0.696. The bar of 0.398 is the global majority class, and a constant answer per task clears it: the most common training answer of each question scores 0.4815 (next point).
- <span id="s6-nano-what-the-gate-did-not-measure"></span>**What the gate did not measure** (study of 15/11/2026, after the gate and the publication: [`docs/s6-nano2-study.md`](https://github.com/maxgfr/skycmd/blob/main/docs/s6-nano2-study.md); its numbers come from the evaluation dumps already read and from validation splits):
  - On 17 of the 20 tasks of `typed_decisions_bench`, s6-nano gives one answer to at least 95% of the items. Its 0.4805 equals answering each question with its most common training answer (0.4815): on most of the bench it does not read the state.
  - The bench's tasks were 0.4% of what it drew in decision training (about 10,200 of 2.55 million draws).
  - A linear model of at most 715,136 bytes, the size of the s6-nano file (hashed character n-grams, logistic regression), reaches 0.703 on the validation split of the bench's 20 tasks, and 0.947 / 0.944 on MASSIVE English / French validation once trained on MASSIVE. This is a rehearsal on validation, not a test score.
  - French: the decision training data hold French only as two yes/no tasks (PAWS-X and Allociné).
  - When2Call: the class `tool_call` is 13.9% of the training split and 35% of the tested items, and s6-nano predicts it 39 times in 2,000.
  - NOTAM has no training data.
  - The verdict does not change: it passed the gate as written. A narrower second version (a router for tool and intent decisions, judged against size-matched baselines) was studied; the study put its chance of clearing all its gates at about 10 to 20%. It was then pre-registered and trained once as s6-route-nano, and it failed its gate ([below](#s6-route-nano-failed)).
- **The games** (result): 30 episodes per game at full depth, on the same seeds as s3-pico. Flappy 200/200 pipes with 0 crashes, Rescue 100 %, Dodge 87 % (= oracle) with 23 reflex vetoes, Doom-like 100 % of levels and a kill rate of 0.772 (= oracle). These are s3-pico's scores, and the oracle's.
- **Early exit rejected** (result). The conformal thresholds (α = 0.02) cut the depth to 1.01 to 1.83 of 4 passes per game, but Rescue fell to 93 % and the Dodge vetoes rose from 23 to 1,016. The pre-registered rule allows 2 %, so the thresholds are not kept. The published file has its three exit thresholds set to 1: the calibration had left them at 0.5, a setting that was never evaluated.
- **Where it fails** (result). NOTAM subject 0.055, below s3-pico (0.163), and When2Call 0.404. The NOTAM sets are never trained on.
- **Not strictly zero-shot** (found afterwards). 206 of the 2,000 English MASSIVE test utterances also appear in the training split, 201 of them in a decision source derived from MASSIVE; none of the French ones does. The model was also trained on 3,113 questions of that source, MASSIVE intents asked with 3 options instead of 8, so the English MASSIVE score is partly in distribution. The seen utterances are not answered better: s6-nano gets 0.393 on the 206 and 0.429 on the 1,794 others, s6-micro-v1 0.495 and 0.521; without them the ECE over all sources is 0.0486 and 0.0837, so neither gate changes ([amendment 14](https://github.com/maxgfr/skycmd/blob/main/docs/s6-gen-prereg.md)). Banking77, trained on, is reported apart: 0.631.
- **Truncation** (result): 1.4 % of the bench states are cut at 512 tokens. s3-pico's windows cut 96.8 % of the bench sequences ([chapter 12](/tutorial/12-size-ladder)).
- **In the browser** (result). WASM matches the native engine on the first 200 bench questions to 9.5e-7, with the same answer on all 200. A long JSON state of the bench takes about 15 ms per decision under Node, a Flappy frame 0.44 ms. Try it on [Live](/live/#s6-nano-the-questions-it-was-trained-for) and [Play](/play/).

### s6-secrets-pico: passed on its bench, fails on real text {#s6-secrets-pico-passed-published-at-full-depth}

It reads a short text (a config file, a `.env`, code) and answers four typed questions: is a usable secret exposed, which service issued it, what kind of value it is (secret, test key, publishable key, placeholder, not a credential), and on which line. The bench, `secret_bench`, has 1,996 texts in four file types the training never shows (docker-compose, toml, go, ini). Ternary GGUF, last exit, full depth (`runs/s6-secrets-pico/report.md`); the rule scanner on the same texts (`runs/s6-secrets-pico/baseline.txt`, measured before any training). These are bench scores, on our own generator's texts; on real files the model flags most clean files ([real text](#s6-secrets-real-text)).

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
- **What this bench does not say.** It is synthetic: the formats, the file templates and the class shares are our generator's, and the keys are fake, with random bodies. Beating the scanner here shows that the model reads formats and context in file types it never saw. It does not show that the model works on real files, and the next measurement shows that it does not.

#### Real text (14/11, after publication) {#s6-secrets-real-text}

After the gate was read and the model published, the published file was run on real files outside its generator, for a possible use in [secretgate](https://github.com/maxgfr/secretgate), a rule-based secrets firewall for coding agents (measurement, script and limits: [secretgate pull request 19](https://github.com/maxgfr/secretgate/pull/19); [amendment 6](https://github.com/maxgfr/skycmd/blob/main/docs/s6-secrets-prereg.md)). 719 real source, configuration and documentation files in which a rule scanner finds nothing were cut into 22,433 chunks of at most 7 lines and 300 characters, the length of the training texts, and the secret question was asked on each chunk. Every flag is a false positive:

| threshold | chunks flagged | files flagged |
|---|---:|---:|
| p ≥ 0.9 | 10.6% | 59.8% |
| p ≥ 0.999 | 5.9% | 49.4% |

- **The flagged chunks are ordinary text** (imports, prose, JSON). The probabilities are not calibrated outside the generator: over the clean chunks the median is 0.00009 and the 90th percentile 0.93.
- **The gate stays passed**: it is on `secret_bench`, and nothing in it is reread or changed. But the model cannot be used to scan real files. It stays published as the result of its pre-registration, not as a tool.
- **Not measured:** recall on real secrets (no real secret is in that corpus).
- **Next.** A second version would be a different task (a rule finds the candidate, the model judges it in its context), with its own pre-registration. Nothing is promised.
- **Outcome (15/11).** That second version was pre-registered and trained once, and it failed its gate on G1 ([below](#s6-secrets-verify-pico-failed)).

### s6-micro v1: failed on calibration only

Same evaluation, 5,587,201 parameters (the s4-micro shape). Bench 0.547, so the gate (≥ 0.45) and the announced goal (0.50, the level of s4's Qwen3.5-4B teacher) are both met. The games match s3-pico at full depth and at its conformal thresholds. The gate fails on the guard: ECE 0.0821 over all the test sources at full depth, 0.0804 at the conformal thresholds, both above 0.08. Its decision training was stopped once for a memory problem and resumed from its checkpoint ([amendment 11](https://github.com/maxgfr/skycmd/blob/main/docs/s6-gen-prereg.md)). From nano to micro, accuracy rises from 0.480 to 0.547 and the ECE from 0.0479 to 0.0821, the median native time per decision goes from 1,488 to 7,346 µs, and the file from 715 KB to 2.4 MB.

It failed its gate; its file is published as a result (owner's decision of 15/11/2026), as `s6-micro-v1` (the last DAgger round, its exit thresholds set to 1, [amendment 15](https://github.com/maxgfr/skycmd/blob/main/docs/s6-gen-prereg.md)): it is selectable on [Play](/play/) and [Live](/live/), marked as failed and never selected by default. The verdict does not change.

### s6-micro v2: passed, on a second look {#s6-micro-v2-passed-on-a-second-look}

The second version of [amendment 12](https://github.com/maxgfr/skycmd/blob/main/docs/s6-gen-prereg.md): 360 working minutes of decision training instead of 180, from v1's pretraining, with v1's data, rate and DAgger rounds, judged by the same gate on the same test records. It was designed after v1's ECE failure, so **its pass is a second look at these test sets**, and every page that reports it says so. Result in [amendment 17](https://github.com/maxgfr/skycmd/blob/main/docs/s6-gen-prereg.md); files in [`runs/s6-micro2/`](https://github.com/maxgfr/skycmd/tree/main/runs/s6-micro2).

- **The gate** (result): bench 0.553 ≥ 0.45 (the announced goal of 0.50 is met), ECE 0.0588 ≤ 0.08 (95% bootstrap interval [0.0526, 0.0687]; v1's was [0.0746, 0.0900]), every game at s3-pico's score at full depth. Passed at τ = 1.
- **Early exit rejected** (result). At the conformal thresholds the games keep their scores with no crash, but Dodge has 21 vetoes against 20 at full depth, and the rule allows no extra veto. The published file `s6-micro.gguf` is the last DAgger round with its exit thresholds set to 1.
- **Where the ECE changed** (result, by source): NOTAM multi 0.2558 → 0.1409, MASSIVE en 0.1334 → 0.0715, bench 0.0660 → 0.0498, NOTAM subject 0.0636 → 0.0185; When2Call rose, 0.1176 → 0.2165, where the model is over-confident (mean confidence 0.612 for an accuracy of 0.401).
- **What the gate did not measure** (the two readings of amendment 16). Answering each bench question with its most common training answer scores 0.4815; v2 scores 0.553. On 11 of the 20 bench tasks it gives one answer to at least 95% of the items (v1: 7, s6-nano: 17). On When2Call it answers `request_for_info` to 1,405 of 2,000 items and `tool_call` to 45, where 696 are `tool_call`; the training prior (`cannot_answer`) scores 0.359 and v2 0.401.
- **Deviations** (stated in amendment 17). The decision training died once on a file-access error and was resumed from its checkpoint (step 61,051 of 65,471; the steps after the checkpoint were done again). Its last 32.5 minutes, the DAgger rounds, the exit step and the evaluation ran while other jobs used the machine, so its latencies (8,970 µs native median, 61 ms per bench question in WASM, which matches native to 8.9e-7 with the same answer on 200 of 200) are not comparable with v1's. The trainer binary is not v1's.
- **Steps**: 65,471 decision steps of 64 (v1: 38,826), DAgger rounds of 3,822, 4,673 and 4,969 steps (v1: about 5,900 each). Banking77, trained on: 0.763 (v1 0.705).

It is selectable on [Play](/play/) and [Live](/live/), with s6-micro-v1 kept next to it as the failed first version.

### s6-ear-nano: failed

Test split, ternary, full depth (`runs/s6-ear-nano/eval/gate.json`):

| gate | n | s6-ear-nano | control (no Mimi pretraining) | bar | verdict |
|---|---:|---:|---:|---:|---|
| end of turn | 31,527 | 0.5681 | 0.7647 | 0.9263 (Smart Turn v3.2, same clips, Pipecat's report) | fail |
| keywords (Speech Commands, 12 classes) | 4,890 | 0.8885 | 0.9217 | 0.90 | fail |
| zero-shot keywords, English | 1,000 | 0.2100 | 0.3020 | 0.20 (2 × chance) | pass |
| zero-shot keywords, French | 1,000 | 0.1330 | 0.1520 | 0.20 (2 × chance) | fail |

- **The pretraining did not help** in this run: the control, trained the same way from a random init, is more accurate on every set. The pretraining itself worked as a task (codebook-0 accuracy 0.2196 on masked val frames, 45 times its technical bar).
- **Size:** 1,532,449 parameters, 5.2 times fewer than Smart Turn v3.2. The verdicts are the same at the conformal thresholds.
- **Not reported:** the precision and recall of the end-of-turn answers, which the pre-registration asks for, were not computed by the run (Smart Turn v3.2: 0.783 and 0.971).
- **Published as a result.** It failed its gate; its file is published as a result (owner's decision of 15/11/2026, [amendment 5](https://github.com/maxgfr/skycmd/blob/main/docs/s6-ear-prereg.md)), as evaluated: the [Listen](/ear/) page loads it under a notice with these numbers and answers at full depth. Its control is not published.

### s6-ear2-nano (s6-ear v2): passed {#s6-ear2-nano-passed}

The second version of s6-ear is one task instead of three: a dedicated `turn` graph (1,306,178 ternary parameters, 632 KB, no early exits) trained for 240 working minutes as a student of Smart Turn v3.2 (BSD-2), with a new front end that adds pitch, voicing and energy to the 40 log-mel bands and reads the last 4 s of audio. The rung (d 160, 4 blocks, one position per 80 ms) was fixed by a latency probe before any training; the harness was checked against Pipecat's published 0.9263 before any test score was read (`runs/s6-ear2-harness/check.json`). Pre-registered in [`docs/s6-ear2-prereg.md`](https://github.com/maxgfr/skycmd/blob/main/docs/s6-ear2-prereg.md); results in [`runs/s6-ear2-nano/GATE.md`](https://github.com/maxgfr/skycmd/blob/main/runs/s6-ear2-nano/GATE.md).

| gate | measured | bar | verdict |
|---|---|---|---|
| G1 accuracy at τ₀.₅ | **0.8993** [0.8962, 0.9027] | ≥ 0.85 and ≥ T − 0.08 (T = 0.9260, Smart Turn v3.2 CPU, same harness) | pass |
| G2 false cut-off at τ₈₅ | **0.0847** [0.0800, 0.0888] | ≤ 0.20 | pass |
| G2 END recall at τ₈₅ | **0.8627** [0.8577, 0.8680] | ≥ 0.80 | pass |
| G3 WASM median | **2.39 ms** (p95 2.41 ms) | ≤ 5.0 ms | pass |
| G3 front end per 10 ms hop | **0.0186 ms** | ≤ 0.10 ms | pass |
| G3 GGUF size | **631,936 bytes** | ≤ 3 MB | pass |
| G3 WASM parity (20 clips) | **max \|Δp\| 1.2e-7, same decision 20 of 20** | ≤ 1e-4, same decision | pass |

On the same 31,527 clips: the silence timer 0.4979, s6-ear-nano 0.5681, its control 0.7647, Smart Turn v3.2 CPU 0.9260 (8.7 MB, 31.7 ms per decision natively, in the same session), Smart Turn v3.2 GPU 0.9371 (Pipecat's published number). AUC 0.9600, ECE 0.0160. The file stores two thresholds from validation: τ₀.₅ = 0.5 (the accuracy reading) and τ₈₅ = 0.7244, the highest threshold whose validation END recall reaches 0.85 (the reading for a voice agent).

- **The sentence of the gate report.** Take v2 when size and speed matter: it gives up 2.7 points of accuracy against Smart Turn for a file of 0.63 MB instead of 8.7 MB and about 2.4 ms per decision in WASM instead of about 32 ms natively; take Smart Turn's file when those points matter more.
- **What the pass does not say.** On real conversational speech — eot-bench, LiveKit's set of 17,934 labelled pauses, which the design never read — v2 and its teacher are far weaker than on the test clips (AUC 0.7299 against 0.7838), and at τ₈₅ v2 answers "finished" on 0.5253 of the real ends of turn: about half pass. The pre-registered sentence "does not carry over" was reserved for a gap of more than 0.10 of AUC; the gap is 0.054, so it does not apply — the weakness above is said instead. It is weakest on Vietnamese (0.7749), Marathi (0.7907) and Bengali (0.7950), and on the dataset mundo_1 (0.7843); English is at 0.9056 and the English human subsets at 0.9187.
- **It is a student of Smart Turn**, distilled from its logit, pooled embedding and encoder states on the training clips only: it cannot be better than its teacher's labels. Most of its test set is synthetic speech.
- **Disclosure.** The official test set is partly in-sample for v2's design choices (the 4 s window and the data change came from a design study that read measurements on 3,000 clips of it), though never for its weights: no test clip is trained on, and no threshold is chosen on test. The run happened under the machine's automatic brake, and CPU work of other models ran during the pilot and part of the training, which amendment 1 allows.
- **The browser demo.** The [Listen page](/ear/) runs the file in WASM behind an energy VAD and a fallback timer: the model is asked after a pause of 0.2 s, then again every 0.2 s while it says not finished, and the turn is ended by a timer after 2.5 s of silence whatever it says. The engine API is `TurnDetector` in WASM (`sky-turn` natively). The published file: [`site/public/models/s6-ear2-nano.gguf`](https://github.com/maxgfr/skycmd/blob/main/site/public/models/s6-ear2-nano.gguf).
- **The article.** [A 632 KB model that knows when you have finished speaking](/articles/a-632-kb-end-of-turn-detector) tells this line as a story: v1's failure, the harness bug that hid the teacher's score, and the design decisions the pass rests on.

### s6-cast-pico: failed {#s6-cast-pico-failed}

A probabilistic forecaster: the s6 encoder on patches of 32 points, 2,048 points of context, nine deciles for each of the next 64 steps, ternary weights and an exit after each loop. 369,792 parameters, 142,912 bytes, trained for 480 working minutes (250,092 steps) on synthetic series and permissive real series, with TinyCast as a teacher whose weight falls to 0. GIFT-Eval, 97 configurations, every score divided by the seasonal naive, lower is better (`data/gift-eval-out/*/summary.json`):

| model | nGMASE | nWQL |
|---|---:|---:|
| **s6-cast-pico** (without flip, the reported model) | **0.9471** | **0.6646** |
| s6-cast-pico, with flip | 0.9099 | 0.6417 |
| gate: FLAIR, training-free | 0.838 | 0.587 |
| goal: TinyCast as published (146,505 parameters) | 0.774 | 0.545 |
| TinyCast, our port, same harness | 0.7856 | 0.5570 |
| seasonal naive, same harness | 0.9999 | 0.8876 |

- **The gate fails on both metrics** (result), and so does the goal of beating TinyCast. With flip the model still misses both bars. It failed its gate; its file is published as a result (owner's decision of 15/11/2026): the [Forecast](/forecast/) page lists it, marked as failed and unticked by default, next to TinyCast, Timestral-5M and the seasonal naive.
- **Where it fails** (result). It beats the seasonal naive on average, but its point forecasts are worse than the seasonal naive at the medium and long horizons (MASE ratio 1.046 and 1.105) and in the Energy and Econ/Fin domains; our TinyCast port has a lower MASE in 86 of the 97 configurations. The worst are the 10-minute solar series at medium and long horizons, with MASE ratios of 2.282 and 2.729 to the seasonal naive.
- **Not measured** by the run: the latency, the mean number of exits used on GIFT-Eval, and the gap to Timestral-5M, whose GIFT-Eval scores are not in the repository. The picture by domain and term is in [`BENCH.md`](https://github.com/maxgfr/skycmd/blob/main/BENCH.md) and [`docs/big-run-report.md`](https://github.com/maxgfr/skycmd/blob/main/docs/big-run-report.md).

### s6-secrets-verify-pico (s6-secrets v2): failed on G1 {#s6-secrets-verify-pico-failed}

The second version of s6-secrets judges a candidate instead of a whole text. A rule (secretgate's engine in a candidate mode, with its filters off and a keyword rule added) proposes a value, and the model answers "is this value a real credential that should be revoked?" from the value, its name, its line and the lines around it. It has the s6-secrets-pico graph, 445,345 ternary parameters in 249,984 bytes, and was trained once (seed 0, 12 working minutes of pretraining and 30 of decision training) on real repositories of Samsung CredData, on issue reports and on synthetic candidates. The test is 221 other CredData repositories and the issue reports' test split, whose values are real, opened once (`runs/s6-secrets2/eval-test/results.md`).

| CredData test, line F1 | every repository | without the largest |
|---|---:|---:|
| **s6-secrets-verify-pico** | **0.8561** | **0.8114** |
| best rule scanner (Betterleaks 1.10.1) | 0.7332 | 0.5367 |
| Gitleaks 8.30.1 | 0.7037 | 0.4080 |
| secretgate's default engine | 0.7279 | 0.5072 |
| cbx-secrets-detector-17M (67 MB), rerun out of sample | 0.6830 | 0.8493 |
| logistic regression of the same size | 0.7945 | 0.7179 |

These numbers are on the gated universe: the rows whose CredData category is exactly `UUID` (in the training repositories, mostly identifiers of exported Postman collections) are removed for every system, by a choice made after reading training and validation rows and before any test file. On the full universe the model is at 0.8153 and Betterleaks at 0.6948.

- **The gate** (result). G1 asks for the best rule scanner + 0.15 on every repository and without the largest one, which holds 5,165 of the 10,463 positive rows. On every repository the model is at 0.8561 against a bar of 0.8832: **G1 fails**. Without the largest repository it holds (0.8114 against 0.6867). G2 (within 0.10 of cbx-17M) holds twice, G3 (above the logistic regression) holds twice, G4 holds on real values (issue-report F1 0.8162 against Gitleaks 0.7104 + 0.08), G5 holds (ECE 0.0073), G6 holds (1,606 µs per candidate natively), G7 holds (31 of 8,825 clean real files flagged, 0.35%, bar 1.0%; v1 flagged 59.8% of a 719-file subset). Six gates of seven hold, and the gate needs all seven: **failed its gate**; its file is published as a result (owner's decision of 15/11/2026); not rerun, not continued in this form, and it does not enter secretgate.
- **Reported with it** (result). The paired bootstrap over repositories gives +0.1229 [0.0413, 0.3613] over Betterleaks on every repository. The bar of +0.15 was set with Gitleaks' published 0.442 in mind; on this test Gitleaks is at 0.7037. Neither changes the reading. CredSweeper's ML (line F1 0.9935) was trained on 85% of all CredData lines, so its CredData numbers are in sample and enter no gate; on the issue reports it scores 0.4919. One test meta file (header and two rows) was seen before training, and the pre-registration says so (amendment 4). By category, the model is weakest on API values (0.541, where cbx-17M is at 0.898), and cbx-17M on Key values (0.266, where the model is at 0.864).
- **In the browser** (result): 2,666 µs per candidate in WASM, within 4.2e-7 of the native probabilities on 200 candidates. It failed its gate; its file is published as a result (owner's decision of 15/11/2026), with a card. Its state is built from a candidate of secretgate's rules, which the browser does not run: since 17/11/2026 (every published model gets a Space, owner's decision) the [Secrets verifier](/secrets-verify/) page and its Space show the model's judgement on candidates of a simplified one-rule proposer, not secretgate's engine.
- **v3: the rules and the verifier together, failed on G1 again (15/11).** s6-secrets v3 ([pre-registration](https://github.com/maxgfr/skycmd/blob/main/docs/s6-secrets3-prereg.md)) is not a new model: a line is flagged when secretgate's default rules flag it or when this verifier accepts a candidate on it (P(real) ≥ 0.34933, chosen on validation and committed before the test read), judged on the same seven gates. It is a second look at test files already read once for v2, with the system defined after that first reading, on the same gated universe (the `UUID` rows removed by the choice above). G1 fails again on every repository: 0.8477 against 0.7332 + 0.15, where the model alone was at 0.8561 (union minus model −0.0084 [−0.0181, 0.0072]); the six other gates hold. On validation all seven held; the 221 test repositories did not confirm it. There is no third reading, and the work on this model ends (`runs/s6-secrets3/eval-test/results.md`).
- **Still to do, from the result block of the pre-registration.** The hand reading of the 110 candidates the model accepts on the clean text (§7.7): **done on 15/11: none of the 110 looks like a real credential.** 34 are test fixtures shaped like secrets, 16 provider-format keys on a rule's allowlist in secretgate's rule file, 10 example credentials of library documentation, 2 masked placeholders, 5 public hashes, fingerprints or public keys, 28 pieces of code, 6 prose or package metadata, 9 other test values ([`BENCH.md`](https://github.com/maxgfr/skycmd/blob/main/BENCH.md)). The whole-scan latency of §9 was not built and is not measured.

Every table, the full universe, the candidate level and the per-category picture are in [`BENCH.md`](https://github.com/maxgfr/skycmd/blob/main/BENCH.md), section "s6-secrets-verify-pico". The three versions are told as one story, for a wide audience, in the article [A 250 KB secrets verifier](/articles/a-250-kb-secrets-verifier).

### s6-sleep: stopped by its pilot {#s6-sleep-stopped-by-its-pilot}

s6-sleep adds a fast-weight memory to a decision model and lets it rerun its last frames while no decision is asked ("sleep", arXiv 2605.26099). The pre-registration first runs a pilot on s6-pico, four runs of 3,684 steps each: the memory with N drawn in training, N = 4 with truncated backpropagation, N = 4 through every pass, and a control without memory. On the 500 dev episodes of the `recall` probe, where a model without memory is at 0.25 (`runs/s6-sleep/pilot.json`):

| run | `recall`, N = 1 | N = 2 | N = 4 |
|---|---:|---:|---:|
| memory, N drawn | 0.2460 | 0.2460 | 0.2460 |
| memory, N = 4, truncated | 0.2565 | 0.2565 | 0.2565 |
| memory, N = 4, through every pass | 0.2430 | 0.2430 | 0.2430 |
| control, no memory | 0.2650 | | |

- **The rule** (result): the memory works if the first run at N = 1 beats the control by at least 0.10. It is 0.019 below it. **The pilot says the memory does not work, so no nano run took place**, no hypothesis was tested on the test seeds, and nothing is published.
- **What the readings show** (result): in each memory run, `recall` does not change between N = 1, 2 and 4. The secondary probe, `rule110`, is reported and decides nothing: 0.4674 to 0.4700 for the first run, 0.5423 for the three others at every N.

### s6-sleep3-nano (s6-sleep v3): failed; the memory is used, sleep does not help {#s6-sleep3-nano-failed}

The third version of s6-sleep. v1 stopped at its pilot (above). v2 changed how the memory starts and ended at its CPU check C4 without a GPU run: 300 training steps did not learn to use the memory. A diagnosis on train sequences found that the write gate closes within the first tens of steps and that a live gradient alone does not train the memory ([`docs/s6-sleep3-diagnosis.md`](https://github.com/maxgfr/skycmd/blob/main/docs/s6-sleep3-diagnosis.md)), so v3 holds the gates and adds an auxiliary loss in training only: a linear head decodes the asked colour from what the memory retrieves, and is not exported. Its pico pilot read 1.0000 on the development episodes of `recall` against 0.2542 for the control. Then six nano runs (the memory and a control without it, seeds 0, 1 and 2) from the checkpoint behind s6-nano, 30 working minutes for the first, and N = 1 chosen on the development episodes. The test was read once: 1,000 `recall` episodes, the four games over 200 episodes, `typed_decisions_bench` (`runs/s6-sleep3/REPORT.md`, `gate.json`).

| `recall`, test, three seeds pooled | 1–4 frames | 5–8 | 9–16 | 17–32 | all |
|---|---:|---:|---:|---:|---:|
| **s6-sleep3-nano, N = 1** | **0.6147** | **0.5590** | **0.4530** | **0.3302** | **0.4892** |
| N = 4 (sleep) | 0.6290 | 0.5910 | 0.4620 | 0.2867 | 0.4922 |
| control, no memory | 0.2420 | 0.2398 | 0.2518 | 0.2423 | 0.2440 |

- **The gate** (result). H0, the memory is used: +0.2452 [+0.2370, +0.2532] above the control, where 0.10 was asked: **holds**. H1, sleep helps: N = 4 minus N = 1 is +0.0030 [−0.0010, +0.0068], where a lower bound above 0 was asked: **not shown**. H2, the games do not lose: Flappy 15.87 against 71.05 for the control (−77.7%) and Dodge 5.89 against 7.94 (−25.8%), where at most 2% under was asked (Rescue and the Doom-like within 0.1%): **not shown**. H3: `typed_decisions_bench` 0.4897 against 0.4808: holds. The s6-nano gate on the exported seed-0 file: the language tests hold (0.499, ECE 0.0578), the arena does not (Flappy 2.10 against s3-pico's 200.00, Dodge 6.90 against 9.47). Parity of the engine with the training code: natively and in WASM. **Failed its gate**; its file is published as a result (owner's decision of 15/11/2026).
- **What it means** (result, by the pre-registration's §11). The memory works and sleep does not help; and as trained, the memory costs two of the four games. The memory fades with the delay at every N, where the pico pilot was at 1.0000 at every delay: why the nano size reaches about half of that is not known.
- **Disclosure.** This is the third version of the same idea on the same games, and its two changes were found by looking at training curves on train sequences; no version had read the development or test seeds before. The first nano run was stopped by the machine's memory guard and started again from scratch with the same seed, its batches computed in parts (amendment 3).
- **After the result.** The nano runs ended halfway up their learning curve: their minute budget bought 393 `recall` training steps at seed 0 against 933 for the pico pilot, and their training accuracy on `recall` was still rising when each phase ended ([`docs/s6-sleep4-diagnosis.md`](https://github.com/maxgfr/skycmd/blob/main/docs/s6-sleep4-diagnosis.md), a diagnosis from the existing logs; the note of 17/11 at the end of the [pre-registration](https://github.com/maxgfr/skycmd/blob/main/docs/s6-sleep3-prereg.md) lists where the runs departed from what was written, among them an auxiliary loss averaged over both memory layers and no reading of the nano memory zeroed; no number or verdict above changes).
- **In the browser.** The [Memory](/memory/) page and its Space play `recall` with the published file (seed 0, 752 KB) and its control (715 KB): the evaluation's own episode generator and scoring, with the memory on, zeroed, or the control, frame by frame or over 200 episodes of the validation split, next to the test reading. On the test, that file alone reads 0.4765 at N = 1 and its control 0.245125.

Every table is in [`BENCH.md`](https://github.com/maxgfr/skycmd/blob/main/BENCH.md), section "s6-sleep3-nano", and the result block of [`docs/s6-sleep3-prereg.md`](https://github.com/maxgfr/skycmd/blob/main/docs/s6-sleep3-prereg.md). The three versions are told as one story, for a wide audience, in the article [A 752 KB model that remembers, and still fails its gate](/articles/a-memory-that-works-and-a-model-that-does-not).

### s6-route-nano (s6-nano v2): failed on G1 and G4 {#s6-route-nano-failed}

s6-nano passed its gate but, as a decision model, mostly answered a constant per task. Its second version narrows the job to one where the options arrive at run time: a request, up to 7 tools as `name: description`, then *none of these tools fits the request*. By amendment 1, written before training when BM25 beat the embedding baseline on validation, the state also carries each tool's BM25 score and rank against the request, and the gate asks the model to beat BM25 as well. The pilot rule chose the micro size (0.6390 against 0.5778 for nano on validation), so the model has the s6-micro graph: 5,587,201 ternary parameters in 2,418,912 bytes, trained once for 120 working minutes (20,504 steps) on tool, intent and When2Call data. The test, T-tools, holds 2,000 slates of tools that no training item contains (tools are split by cluster), and was opened once (`runs/s6-nano2/eval-test/results.md`).

| T-tools (2,000 slates, *none* right on 20%) | size | accuracy | ECE |
|---|---:|---:|---:|
| **s6-route-nano** | 2.4 MB | **0.8395** | **0.0244** |
| BM25 over the slate's own tools | no parameters | 0.8500 | 0.0610 |
| all-MiniLM-L6-v2, cosine | 90.9 MB | 0.8840 | 0.0231 |
| toolrouter (stanley-nv) | 7.66 MB | 0.7855 | 0.0886 |
| static-ft (potion fine-tuned, cut to the model's bytes) | ≤ 2.4 MB | 0.6735 | 0.0430 |
| potion-base-2M, cosine | 7.56 MB | 0.6675 | 0.0393 |
| pair-lr (logistic regression of the model's bytes) | ≤ 2.4 MB | 0.4500 | 0.0426 |
| s6-nano (v1), same prompt | 715 KB | 0.0885 | 0.1152 |

- **The gate** (result). G1 asks for potion-base-2M + 0.03, more than both size-matched trained baselines, and, by amendment 1, more than BM25. The first three hold (0.8395 against 0.6975, 0.6735 and 0.4500); against BM25 the model is at −0.0105 (95% interval [−0.0265, +0.0055]), a tie where "strictly above" was asked: **G1 fails**. G2 holds: *none* answered on 0.1565 of the slates, no tool name over-predicted, and with the right tool removed the answer changes on 0.9528 of the requests it had right. G3 holds: ECE 0.0244. **G4 fails**: 38.9 ms per request in WASM with the tool list cached (SIMD128; 23.5 ms with relaxed SIMD, 24.0 ms natively), where the bar is 2 ms. **Failed its gate**; its file is published as a result (owner's decision of 15/11/2026), not as a router, and the run is not repeated. As amendment 1 wrote before training, "a router that reads a lexical score and does no better than that score has no use."
- **Names masked** (result). With every tool renamed `tool_<k>`, the model is at chance (0.1050), where BM25 keeps 0.8445 and potion-base-2M 0.7015 on the same slates. One reading, not tested: it ties each BM25 entry of its state to an option through a distinctive name. A diagnosis made after the gate, on validation files only, found otherwise: in training the right tool of every renamed slate was the only option whose description the teacher had rewritten, and *none* was never right there, so under `tool_<k>` names the model picks the description that reads most like a rewrite, does not read the BM25 line and almost never answers *none*; no number above changes ([diagnosis](https://github.com/maxgfr/skycmd/blob/main/docs/s6-route4-diagnosis.md)). With the descriptions of the right tool and of a distractor swapped, it follows the description on 0.6700 and the name on 0.0780.
- **Other sources** (result; announced goals and reported sources). Never-trained tools of MetaTool 0.3260 (potion-base-2M 0.5245) and intents of CLINC150 by description 0.3980 (0.7740): the announced goals of matching potion-base-2M there are not met. tool-routing-data 0.3700 (toolrouter 0.7900), BFCL 0.7205 (BM25 0.8295), When2Call 0.5153, the held-out MASSIVE scenarios 0.3645. Of the other announced goals, T-tools ≥ 0.80 is met, and 0.95 among the requests answered above the stored ask threshold τ_ask (0.72938) is not (0.9433 at coverage 0.7325).
- **Where it is good** (result). On the 70% of T-tools slates it is most sure of, it is right on 0.956 (BM25 0.945, MiniLM 0.974), and it is better calibrated than BM25. Its first exit (2 of 6 block passes) keeps 0.8340 at 13.3 ms natively.
- **Deviations** (stated in the result block). The file that opens the test (`TEST-OPEN`) was created by the run's operator under the owner's delegation, not by the owner's hand. The evaluation binaries and the WASM package were rebuilt from the main branch after training; WASM matches native to 2.2e-6 with the same answer on 200 of 200. The study that designed this version read s6-nano's test dumps; T-tools is new, and the clustering that defines "unseen tool" uses all-MiniLM-L6-v2, which is also a reported baseline.
- **Read after the gate, not a result.** On the already-read test, with nothing fitted, the equal-weight geometric mean of this model's and potion-base-2M's probabilities reaches 0.8885 on T-tools, above BM25 and MiniLM. It is a hypothesis, now pre-registered as s6-nano v3 ([`docs/s6-nano3-prereg.md`](https://github.com/maxgfr/skycmd/blob/main/docs/s6-nano3-prereg.md)) to be read on fresh slates, and not read yet ([`BENCH.md`](https://github.com/maxgfr/skycmd/blob/main/BENCH.md), section "s6-route-nano").

The pre-registration publishes the playground only if the gate passes (§11). The owner decided on 17/11/2026 that every published model gets a Space, failed gates included: its Hugging Face Space runs the file alone, with the failed gate first, and the Router page of this site stays the pair's. Every table is in [`BENCH.md`](https://github.com/maxgfr/skycmd/blob/main/BENCH.md), section "s6-route-nano". Its follow-up, the same file fused with a static embedding, passed its own gate on fresh slates as s6-route ([below](#s6-route-passed)); this model's verdict does not change.

### s6-route (s6-nano v3): passed on fresh slates, system defined after v2's test was read {#s6-route-passed}

**Disclosure.** v3 was designed after v2's test was read, from an exploratory reading of v2's test dumps. Its parameters come from validation. Its gate is read on requests no system had scored; on T2-tools these requests are about the same held-out tools as v2's test. The latency bar is 30 times v2's and was set after v2 was measured. The system needs distinctive tool names.

Nothing is trained. The system takes the s6-route-nano file as it was published and the static embedding potion-base-2M (Minish Lab, MIT) packed to 4 bits, scores every option with both, and fuses them: `p(i) ∝ (p_model(i) · p_embedding(i))^0.81`, renormalised; below a top probability of 0.671047 it says "ask the user". β and that threshold were set on validation, before any fresh slate was built ([`docs/s6-nano3-prereg.md`](https://github.com/maxgfr/skycmd/blob/main/docs/s6-nano3-prereg.md) §2). The fresh files were built once from requests that no v2 slate uses, hashed, counted, and read once on 16/11 (`runs/s6-nano3/eval-fresh/results.md`).

| fresh file | s6-route (3.7 MB) | the model alone | the embedding alone | BM25 | BM25 + the same embedding | MiniLM (91.6 MB) |
|---|---:|---:|---:|---:|---:|---:|
| **T2-tools** (945, unseen tools) | **0.8868** | 0.7926 | 0.7386 | 0.8169 | 0.8624 | 0.9048 |
| T2-tools, names masked | 0.2497 | 0.0720 | 0.7280 | 0.8138 | 0.8423 | 0.9005 |
| T2-metatool (2,000) | 0.5175 | 0.3190 | 0.5340 | 0.4330 | 0.5420 | 0.6570 |
| T2-clinc (2,000) | 0.8040 | 0.4240 | 0.7855 | 0.4990 | 0.7750 | 0.8790 |
| T2-massive (2,000) | 0.5340 | 0.3680 | 0.4755 | 0.3205 | 0.4515 | 0.5730 |

- **The gate** (result). G1: above BM25 with the lower bound of the slate bootstrap above zero (+0.0698 [0.0455, 0.0952]), at least MiniLM − 0.03 (0.8868 against 0.8748), above each of its parts. G2: never more than 0.02 below the better part on MetaTool (0.5175 against 0.5140, below the embedding alone at 0.5340), CLINC150 and MASSIVE. G3: ECE 0.0162. G4: WASM matches the Python reference to 1.6e-6 with the same answer on 200 of 200, 30.9 ms per request with the tool list cached against a bar of 60 ms, 3,692,224 bytes for the two files against 4 MB. **Passed, every condition**; published as s6-route.
- **What the pass does not say** (readings reported with the gate, announced in §6 before the reading). The three largest clusters hold 300 of the 945 T2-tools slates. Without them the system is at 0.8806 against BM25 0.8605, **+0.0202 [−0.0062, +0.0481]**; a bootstrap that resamples clusters instead of slates gives **+0.0698 [−0.0079, +0.1404]**. Both intervals hold zero: the gated advantage over BM25 leans on the three largest clusters. With every tool renamed `tool_<k>` it falls to **0.2497**, where BM25 keeps 0.8138. A diagnosis made after the gate, on validation files only, found why: in training the right tool of every renamed slate was the only option whose description the teacher had rewritten, and *none* was never right there, so under `tool_<k>` names the model part picks the description that reads most like a rewrite and almost never answers *none*; no number above changes ([diagnosis](https://github.com/maxgfr/skycmd/blob/main/docs/s6-route4-diagnosis.md)). MiniLM remains above on every file. Outside the tool lists of its training families it stays near its embedding part: below it on MetaTool (0.5175 against 0.5340), above it on CLINC150 (0.8040 against 0.7855) and MASSIVE (0.5340 against 0.4755).
- **Announced goals** (not gates). 0.87 on T2-tools: met. 0.95 among the requests it answers at the ask threshold, with coverage ≥ 0.80: not met (0.9433 at coverage 0.8593). The fast mode (the model stopped at its first exit) within 0.015: met (0.8836, 10.5 ms per request).
- **Latency** (result). Measured on a machine in use by its owner, with a GPU run of another model active: 30.9 ms per request with the tool list cached (SIMD128), 23.5 ms with relaxed SIMD, 10.5 ms in fast mode; the first request on a new tool list 45.3 ms. The 60 ms bar was set after v2 measured 38.9 ms, where its own bar was 2 ms.

Try it on the [Router](/router/) page. Every table is in [`BENCH.md`](https://github.com/maxgfr/skycmd/blob/main/BENCH.md), section "s6-route". The story of the three versions is told in an article: [A 3.7 MB tool router built on a MacBook](/articles/a-3-7-mb-tool-router).

## LFM3-1.7B: System 1 first, System 2 when needed (November 2026) {#lfm3-s1s2}

No skycmd model here, and no weight trained: LFM3-1.7B (Liquid AI, 4-bit MLX build, revision `deb2986b`, run locally with mlx-vlm) answers maths questions, first without thinking (**System 1**). A 6-coefficient logistic probe on that answer decides whether to ask again with thinking on (**System 2**, the thinking closed after a budget of 256 or 1,024 tokens). Its threshold comes from conformal risk control (α = 0.02) on a calibration split, the same tool as the early exits of the s6 models. A question sent to System 2 counts the tokens of both. Pre-registered in [`docs/lfm3-s1s2-prereg.md`](https://github.com/maxgfr/skycmd/blob/main/docs/lfm3-s1s2-prereg.md) (amendments 1 to 5, all before the run). It ran once, from 14/11/2026 at 23:58 to 15/11/2026 at 12:44 (12 h 46 min), under the machine's brake modes (0.85 with 50:10, then 0.5 with 45:15 on battery from 08:31 to 09:08, then 0.95 with 55:5), which only sleep between two questions.

The budget, chosen on the calibration split before the test was read as the best always-System-2 accuracy, is **1,024** tokens (0.797 against 0.742 for 256); τ = 0.332. On the test split, 1,000 questions (500 GSM8K, 500 MATH-500), with that budget (`runs/lfm3-s1s2/report.md`):

| task | n | System 1 alone | System 2 alone | router | oracle | tokens: S1 / S2 / router | answered by S1 | S2 forced |
|---|---:|---:|---:|---:|---:|---|---:|---:|
| GSM8K | 500 | 0.824 | 0.878 | **0.862** | 0.910 | 396 / 728 / 547 | 0.90 | 0.23 |
| MATH-500 | 500 | 0.510 | 0.714 | **0.710** | 0.718 | 722 / 1188 / 1518 | 0.78 | 0.46 |
| all | 1,000 | 0.667 | 0.796 | **0.786** | 0.814 | 559 / 958 / 1032 | 0.84 | 0.35 |

On the calibration split (600 questions, GSM8K and MATH train): System 1 0.687, System 2 0.797, router 0.802, oracle 0.820, tokens 535 / 934 / 1003, answered by S1 0.85, S2 forced 0.33. "S2 forced" is the share of questions whose thinking reached the budget and was closed.

- **H2, accuracy kept** (result): router 0.786 ≥ System 2 0.796 − 0.02: **yes**.
- **H3, compute halved** (result): router 1,032 tokens ≤ 0.5 × 958: **no**. The router uses more tokens than always thinking: System 1 alone already takes 559 tokens on average, more than half of System 2's 958, and every question sent to System 2 pays for both.
- **H4, better than both modes** (result): **no** (0.786 against 0.796 for System 2 alone).
- **Verdict: failure.** Success needed H2 and H3. Thinking helps this model in maths (H1, a diagnostic without a threshold: 0.824 against 0.878 on GSM8K, 0.510 against 0.714 on MATH-500), and the oracle (the better of the two answers per question, 0.814) leaves little room above System 2 alone.
- **With a budget of 256** (reported): System 1 0.667, System 2 0.724, router 0.729, oracle 0.749, tokens 559 / 663 / 870.
- **The fine-tuning that was to follow** ([`docs/lfm3-ft-prereg.md`](https://github.com/maxgfr/skycmd/blob/main/docs/lfm3-ft-prereg.md)) has not run. Its pre-registration sets no condition on this result: it measures the router again on the base and on the fine-tuned model, and its own verdict (gates G1 to G3) is separate from the router's.

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
