# s6-nano v2: design study

Written on 15/11/2026, after the gate of s6-nano was read and published (`docs/s6-gen-prereg.md`, amendments 1 to 15; `BENCH.md`, section "skycmd-s6"). Read only: the main checkout, its run files (`runs/s6-nano*`, `runs/s6-micro*`, `runs/all/nano-*.log`), its evaluation dumps and its data. Nothing was trained. No test record was read beyond the evaluation dumps and test files that the v1 evaluation and `BENCH.md` had already read; the only new measurement on data is a baseline rehearsal on **validation and training splits** (§5). The analysis scripts are in `…/scratchpad/tools/` (`curves.py`, `loss.py`, `survey.py`, `lr_rehearsal.py`) with their outputs next to them.

## 0. Summary

1. **v1 does not read the state of most bench questions.** On 17 of the 20 tasks of `typed_decisions_bench`, s6-nano gives the same answer to at least 95% of the items. Its bench accuracy, 0.4805, equals the accuracy of answering each question with its most frequent answer in `typed_decisions_train` (0.4815). The gate's "majority class" (0.398) was a lower bar than that per-question prior. s6-micro, with the same data and the same number of questions seen, answers one constant on 7 of the 20 tasks.
2. **Under-trained on its targets, not shown to be at a capacity limit.** Pretraining and decision training were both still improving when their budgets ended, and most of the last gain came from the final learning-rate decay. The evaluated task families are a sliver of the mixture: in its 90 decision minutes s6-nano drew about 10,200 questions of the bench's training tasks (0.4% of 2.55 million draws, under two passes over each task's ~280 examples). A size-matched linear model reads these states: 0.703 on the validation split of the same 20 tasks, against 0.525 for the per-task prior.
3. **The gaps per source.** French MASSIVE is a language gap (0.239 against 0.4255 in English, same intents, same English labels, no truncation): the decision data hold French only as two binary tasks. When2Call is a label-prior shift: `tool_call` is 13.9% of the training split and 35% of the tested items, and s6-nano predicts it 39 times out of 2,000. NOTAM is out of domain (no NOTAM training data); on its 13-way subject question the model collapses onto two subjects (0.055, below chance). Truncation explains almost nothing: at most 3.6% of the test states and none of the MASSIVE ones are cut.
4. **The matched-size baseline is strong.** A hashed character n-gram logistic regression of at most 715,136 bytes (the size of the s6-nano file), trained on MASSIVE's train split, reaches 0.947 (English) and 0.944 (French) on MASSIVE's validation split in the 8-option format of the test files; trained on English only it still reaches 0.552 in French. On When2Call validation it reaches 0.720 (majority 0.436). These are validation numbers, not test scores, but they set the order of magnitude of any per-source bar.
5. **Recommended v2: a narrow router, not a longer general model.** The same graph, tokenizer and pretrained encoder, trained for 90 working minutes on tool and intent decisions in English and French (tool choice among tools given at run time, When2Call, MASSIVE en/fr intents, Banking77), with no games, judged per source against size-matched baselines trained on the same labels. Its main gate is on a tool-choice test set built fresh, which no one has read. Cost: about 110 working minutes of GPU and 30 of CPU, plus a data builder, a training script, a baseline script and a gate script. The most likely outcome is still a failure against the linear baseline on fixed label sets (MASSIVE); the run is worth doing only because it answers whether a 1.5M ternary model has any routing niche where a linear model at the same size does not reach.

## 1. What v1 is

s6-nano: 1,486,225 ternary parameters (Parallax rows off), 715,136-byte GGUF, `d` 144, 2 blocks × 2 loops, exits after 1 to 4 block passes, windows 512 / 160, tokenizer of 2,048 entries. Recipe (`runs/recipe/recipe.json`): AdamW, `LR_SCALE` 2.0, pilot ×1 (validation 0.559 / 0.564 / 0.474 at ×0.5 / ×1 / ×2, `runs/s6-nano-lr/summary.json`). MLM pretraining 90 working minutes (53,371 steps, 0.87B tokens, epoch 8), decision training 90 minutes (19,893 steps of 128), three DAgger rounds of 15 minutes (4,645 / 4,660 / 4,644 steps). Mixture `decisions=0.25, distill=0.30, games=0.35, numeric=0.10`; sampling is uniform within each directory (`sample_batch`, `crates/sky-train/src/s2/data.rs` 940–960).

Test results at full depth (`runs/s6-nano/eval/eval-s6-nano-full.json`; per-source ECE from `docs/big-run-report.md` §6):

| source | n | s6-nano | s6-micro | chance | ECE nano | mean confidence nano |
|---|---:|---:|---:|---:|---:|---:|
| `typed_decisions_bench` | 2,000 | 0.4805 | 0.5470 | (2 to 5 options) | 0.0261 | 0.4605 |
| `zs_massive_en` (not strictly zero-shot, am. 14) | 2,000 | 0.4255 | 0.5180 | 0.125 | 0.0248 | 0.4088 |
| `zs_massive_fr` | 1,000 | 0.2390 | 0.2750 | 0.125 | 0.1112 | 0.3502 |
| `when2call` (test of a trained source) | 2,000 | 0.4035 | 0.4305 | 0.25 | 0.1412 | 0.5406 |
| `notam:deel_subject` | 2,000 | 0.0550 | 0.0520 | 0.077 | 0.0746 | 0.1296 |
| `notam:notam_multi` | 2,000 | 0.4935 | 0.3940 | 0.077 to 0.5 | 0.0710 | 0.5159 |
| pooled | 11,000 | 0.3595 | 0.3780 | | 0.0479 [0.0416, 0.0564] | |

## 2. Question 1: was it still improving, and is it under-training or capacity?

### 2.1 Pretraining (`runs/s6-nano/pretrain/log.jsonl`)

| working minute | step | learning-rate factor | epoch | train loss (mean of 25 logs) | validation CE |
|---:|---:|---:|---:|---:|---:|
| 20.8 | 12,500 | 1.00 | 1 | 2.666 | 1.690 |
| 41.1 | 24,500 | 1.00 | 3 | 2.601 | 1.616 |
| 51.3 | 30,500 | 1.00 | 4 | 2.594 | 1.606 |
| 71.6 | 42,500 | 1.00 | 6 | 2.578 | 1.581 |
| 75.0 | 44,500 | 0.85 (decay starts) | 6 | 2.555 | 1.555 |
| 81.8 | 48,500 | 0.51 | 7 | 2.483 | 1.495 |
| 90.0 | 53,371 | 0.10 | 8 | 2.395 | **1.4075** (final) |

At constant rate the validation loss was still falling, slowly (−0.024 over the last 20 constant minutes); the decay of the last 20% took it down by a further 0.17. So the pretraining was not converged. But it ended in its eighth pass over the same 244 MB: more minutes on the same text are mostly repetition, whose value falls off after a few epochs (data-constrained scaling, Muennighoff et al., 2023). A longer pretraining is not where the measured losses are (§3), and it is not recommended.

### 2.2 Decision training (`runs/s6-nano/decide/log.jsonl`, 4,000 validation questions, last exit)

| working minute | step | mode | all | decisions part |
|---:|---:|---|---:|---:|
| 8.9 | 2,000 | f32 | 0.487 | 0.443 |
| 9.3 | 2,001 | ternary, QAT not yet trained | 0.411 | 0.391 |
| 13.6 | 3,000 | ternary (QAT) | 0.497 | 0.442 |
| 22.6 | 5,000 | ternary | 0.531 | 0.448 |
| 36.0 | 8,000 | ternary | 0.561 | 0.488 |
| 45.0 | 10,000 | ternary | 0.575 | 0.503 |
| 67.5 | 15,000 | ternary | 0.595 | 0.523 |
| 72.1 | 16,000 | ternary (decay starts) | 0.595 | 0.521 |
| 89.9 | 19,893 | ternary, final | **0.610** | **0.536** |

- **Still improving.** In the constant-rate phase the decisions part gained about +0.03 per doubling of steps (0.488 at step 8,000, 0.52 at 16,000); the decay added +0.015. Training loss kept falling (0.897 at step 9,000, 0.836 at the end). There is no sign of over-fitting (no validation decline) and no plateau.
- **DAgger rounds do not train language.** Within each 15-minute round the decisions part is flat or slightly down (round 1: 0.522 → 0.513; round 2: 0.509 → 0.513; round 3: 0.534 → 0.538; each round's validation set includes its own DAgger data, so rounds are not comparable with each other). With decisions at 0.1 and distill at 0.2 of the round mixture, the rounds keep language; they do not improve it.
- **Float against ternary.** Ternarizing the f32 model at the start of QAT costs 0.077 at once (0.488 → 0.411); after 1,000 QAT steps the ternary model is above the f32 one of the same minute (0.497). There is no float run of the full budget, so the final cost of ternary weights is not measured; the end-of-run "f32" numbers (0.457) are the latent weights of a ternary-trained model, not a float model. Nothing in the files points at ternary as the binding limit, and nothing excludes a few points.
- **Exit against exit.** The four exits are within 0.01 of each other throughout (final: 0.602 / 0.613 / 0.614 / 0.610). On the test sources (`runs/s6-nano/eval/report.md`), exit 0 (one block pass) is as good as exit 3: bench 0.4850 against 0.4810, When2Call 0.4022 against 0.3939. Only the Rescue game gains from depth (0.743 → 0.904). The effective rank of the state falls with depth (79 → 66 of 144). For language, the loops add compute that the model does not use: depth is not the constraint.
- **Nano against micro.** At equal working minutes the two sizes are level on validation (minute 67.5: nano 0.595 / 0.523, micro at minute 68.1: 0.600 / 0.525); micro does it with half the questions (batch 64 against 128). At an equal number of questions (about 0.96 million), micro is ahead by about 0.04. So micro learns more per example (capacity helps sample efficiency), while under a time budget the two are level until micro's longer schedule. On the test, micro's advantage (bench +0.067) is larger than on validation (+0.03).

### 2.3 How much of the evaluated task families the model saw

Draws per family in the 90-minute decision run (2,546,304 = 19,893 × 128), from the group weights and the item counts printed by `s2-decide` (decisions 776,101 items, distill 560,238):

| family | train questions (gold + teacher-mixed copies) | draws | share of draws | passes per gold question |
|---|---:|---:|---:|---:|
| `typed_decisions_train` (the bench's 20 tasks) | 5,640 + 4,063 | ≈ 10,200 | 0.40% | ≈ 1.8 |
| `when2call` | 20,108 + 13,844 | ≈ 35,400 | 1.39% | ≈ 1.8 |
| `calibrated:mteb/amazon_massive_intent` (3 options) | 3,113 + 2,844 | ≈ 6,400 | 0.25% | ≈ 2.1 |
| French (`pawsx_fr`, `allocine`, both binary) | 74,906 + 65,207 | ≈ 150,000 | 5.9% | ≈ 2.0 |
| games | 252,591 | ≈ 891,000 | 35% | ≈ 3.5 |

s6-micro drew the same shares (2,484,864 draws). The three DAgger rounds add about 3,900 bench-family draws.

### 2.4 Verdict on question 1

Both phases were still improving when they stopped, and the decision phase is far from saturated on the families that are evaluated: under two passes over each bench task, with a per-task prior learned first. The bench collapse (§3.1), the flat exits and the linear baseline that reads the same states (§5) all say the information is there and the training did not get to it. That is under-training on the targets, mostly a matter of the mixture rather than of total minutes. Capacity is not excluded: with the same exposure micro reads the state on 13 of 20 tasks and nano on 3, so a larger model extracts more from the same few examples. The cheaper lever for nano is exposure, not size.

## 3. Question 2: where it loses, source by source

The breakdowns below read the v1 dumps (`runs/s6-{nano,micro}/eval/dump-*.jsonl`) joined with the gold answers in the order of `scripts/s6-ece-by-source.py`. Truncation shares are from `runs/s6-nano/eval/truncation.json` (first 500 records per source).

### 3.1 `typed_decisions_bench` (state truncated 1.4%, head 0%)

| type, options | n | nano | micro | per-task majority (test) |
|---|---:|---:|---:|---:|
| choice, 4 | 400 | 0.510 | 0.562 | 0.417 |
| choice, 5 | 200 | 0.455 | 0.390 | 0.355 |
| noul, 2 | 600 | 0.610 | 0.675 | 0.507 |
| score, 4 | 700 | **0.359** | 0.479 | 0.319 |
| score, 5 | 100 | 0.490 | 0.510 | 0.490 |

- 17 of 20 tasks: nano gives one answer to ≥ 95% of the items (13 of them at 100%); its accuracy there is the share of that answer (0.506 over the 1,700 items). The 3 tasks where it varies its answer score 0.333 in all.
- The per-question prior from training (`typed_decisions_train`, the most frequent answer of each of the 20 instructions) scores **0.4815** on the bench; nano 0.4805. The oracle per-question prior on the test itself is 0.5185.
- Worst tasks are 4-level scores read from long JSON states: "How time-sensitive is processing this invoice?" 0.12 (the one answer it gives is the wrong one; test majority 0.53), "How quickly does this trace need attention?" 0.25, "How risky was the agent's behaviour in this trace?" 0.29 (micro 0.63).
- Length: accuracy rises with state length (0.367 → 0.588 by quartile), but length is tied to the task, so this is the task mix, not a length effect.

### 3.2 MASSIVE (truncation 0% in both languages)

- English 0.4255 (chance 0.125), no position bias (predictions spread evenly over the 8 positions). The 206 utterances seen in training score 0.393, the 1,794 others 0.429 (amendment 14).
- French 0.239. Over the 43 intents with enough items in both, the macro accuracy is 0.367 in English and 0.209 in French: French keeps about 40% of the above-chance margin. By scenario the loss is uneven: email 0.541 → 0.075, iot 0.582 → 0.103, alarm 0.329 → 0.062, calendar 0.472 → 0.198, while datetime (0.651 → 0.645) and transport (0.481 → 0.615) hold. Some intents are never right in either language (recommendation 0.000, takeaway 0.024, music 0.088): the intent names themselves are not understood.
- The test question pairs a French utterance with English intent names. The training data never ask that: French appears only as `pawsx_fr` (paraphrase, binary) and `allocine` (sentiment, binary), 9.7% of the training records and 5.9% of the draws; a crude stop-word count finds French in no other source above a few hundred records (the largest is a French discourse-relation set of 220, `tools/survey-summary.txt`). The pretraining text is 28% French (69 of 244 MB: FineWeb-2 fr, TinyStories fr, Europarl fr), so the encoder has seen French; the decision head has never mapped French requests to labels.
- **So the French gap is language, not truncation (0%) and not the label format (the English side of the same format reaches 0.4255).**

### 3.3 When2Call (state truncated 0.2%)

- Gold classes in the 2,000 tested items: tool_call 696, request_for_info 586, cannot_answer 718, direct_answer 0. Nano predicts tool_call 39 times, request_for_info 1,183 and cannot_answer 778.
- The training split is built differently (`DATA_LICENSES.md`: train/val from `train_sft` and `train_pref`, labelled from the response text; test from `test/mcq`): tool_call is 2,788 of 20,108 training items (13.9%), 145 of 1,038 validation items, and 1,279 of the 3,630 test items (35%). The model learned the training prior; its over-confidence on this source (mean confidence 0.54, accuracy 0.40, ECE 0.141) comes with it. The GLM teacher agrees with the training gold on only 68.8% of When2Call (`data/distill/agreement.json`), so the soft targets are noisy too.
- Accuracy falls with the length of the tool list (shortest quartile 0.532, others 0.32 to 0.39) although truncation is 0.2%: reading a request against several tool descriptions is the hard part.

### 3.4 NOTAM (never trained; truncation 0.8% and 3.6%)

- `deel_subject` (13 subjects): 0.055, below chance (0.077); 92% of the predictions go to two subjects (positions 1 and 8). micro: 0.052, 84% to one subject.
- `notam_multi`: 0.4935, of which one statement ("The affected area is described by an arc…", 412 items, false for 99.5%) contributes 410 correct answers; without it nano is at 0.363 on the other 1,588 items. Its 13-way subject question: 0.161.
- Domain jargon with no training data at all. No licensed NOTAM training set is on disk, and these files are all `split = test`. NOTAM is a probe of out-of-domain reading, not a gap a v2 can close honestly; a v2 should stop counting it in a pooled gate.

### 3.5 Truncation against language, overall

Truncation explains at most a few hundredths anywhere on the test (state cut on 1.4%, 0%, 0%, 0.2%, 0.8% and 3.6% of the six sources). In training it is larger (12.5% of states and 10.5% of heads over all sources, n 258,309), mostly in long-state sources (HelpSteer2, procedural, open_jev) that the tests do not measure. The measured gaps are: the per-question prior on the bench (exposure), French (no French label mapping in the decision data), the class prior of When2Call (a split shift), and NOTAM (no domain data).

## 4. Question 3: data that would close the gaps without leaking a test set

### 4.1 What is on disk, and its licences (`DATA_LICENSES.md`)

| data | licence | on disk | use for v2 |
|---|---|---|---|
| MASSIVE en + fr (`mteb/amazon_massive_intent`) | CC-BY-4.0 | `data/commands/massive.jsonl`: en 11,468 train / 2,020 val / 2,944 test; fr 11,189 / 1,926 / 2,794; 60 intents | train + val only |
| When2Call | CC-BY-4.0 | `data/decisions/when2call.jsonl` (20,108 / 1,038 / 3,630) | train + val, class weights |
| Banking77 | CC-BY-4.0 | `data/decisions/banking77.jsonl` (12,419 / 664): **its official test rows are in our train/val** | train only; never a test |
| glaive-function-calling-v2, hermes-function-calling-v1 | Apache-2.0 | fetchers in `sky-data fetch-commands` (`crates/sky-data/src/commands.rs`), routing builder in `s4.rs` `tool_routing`; not on disk now | tool choice: train / val / **fresh test** |
| CLINC150 (`calibrated:clinc/clinc_oos` 3,300; `tasksource:clinc_oos/plus` 532) | CC-BY-3.0 (to confirm at fetch) | in `data/decisions` | hold the whole family out: zero-shot probe of unseen intent names |
| Bitext customer support | CDLA-Sharing-1.0 | used for s4 | optional English intents (27) |
| HWU64 | CC-BY-4.0 | s5 evaluation only | **never train**: MASSIVE descends from the same SLURP/HWU64 utterances |
| MTOP fr/en | Hub tag `unknown` | — | not usable until its licence is confirmed |
| SLURP, Speech-MASSIVE, xLAM | NC or gated | — | rejected (`DATA_LICENSES.md` "Rejected") |
| sib200 `fra_Latn` | CC-BY-SA-4.0 | s5 evaluation only | possible French zero-shot probe (topics), evaluation only |

MASSIVE fr-FR train is the only licensed French intent data on disk; it is the direct answer to the French gap. No other open French intent set with a training licence was found in the repository's licence register.

### 4.2 Leakage rules any v2 needs

- MASSIVE: only the train and validation partitions; MASSIVE keeps one partition per utterance id across locales, so a French training utterance is never the translation of an English test one (to be checked by id at build time if the mirror carries ids, and by exact match after lowercasing in any case). Every training record whose state equals a MASSIVE test utterance of either language is dropped; this removes the 206 overlaps of amendment 14, and the derived sources `calibrated:mteb/amazon_massive_intent` and `tasksource:multilingual/massive` are dropped as a whole (they are MASSIVE under other formats, partly from its test utterances).
- The test files `data/s4/heldout/zs_massive_{en,fr}.jsonl` are never rebuilt: `s4.rs` `heldout` draws them from a shared generator, so a new builder must use its own derived generator and the two files are pinned by their sha256 before any build.
- When2Call: test is `test/mcq` only; training states equal to a test state (after normalisation) are dropped.
- Tool choice: glaive and hermes overlap (hermes ships `glaive_func_calling`), so the user turns are deduplicated across both before a split by hash of the normalised user text; the test file is written and hashed before any training.
- Banking77 gives no test (its official test is in our training split); `zs_banking77` is not reported as a test.
- The GLM teacher: no new labels (its labels exist for v1's sources only; new sources keep gold targets). The teacher labels of dropped sources are dropped with them.

### 4.3 What the honest evaluation becomes

Once MASSIVE's train split is used, MASSIVE is no longer zero-shot in either language: it becomes a supervised, in-distribution test like When2Call, read on the same 2,000 + 1,000 records as v1 for comparability, and judged against supervised baselines trained on the same labels at the same size. A zero-shot claim then needs a family held out entirely: CLINC150 with all its intents unseen (the decision format takes the intent names as options, so the model can be asked about intents it never saw). That claim is weak by the research note's own reading (§9: no general zero-shot intent at this size) and should be reported, not gated. French zero-shot has no licensed probe on disk except sib200 (topics, evaluation only).

## 5. The matched-size baseline, rehearsed on validation (no test record read)

`tools/lr_rehearsal.py`, scikit-learn 1.9.1 (the version pinned by `docs/s6-secrets2-prereg.md`) in a scratch environment: `HashingVectorizer` character 1- to 5-grams (signed, L2-normalised, case kept), `SGDClassifier` log loss, L2, `alpha` chosen among 1e-6, 1e-5, 1e-4 on the reported validation split itself (optimistic by that choice), weights quantised to int8 after training, the number of hashed columns set so that the weights fit in 715,136 bytes.

| task | trained on | evaluated on | weights | accuracy | ECE | reference |
|---|---|---|---:|---:|---:|---|
| MASSIVE, 8 options (gold + 7 random intents, the test format) | MASSIVE train en + fr | MASSIVE val en / fr | 491,520 B | **0.947 / 0.944** | 0.013 / 0.010 | v1 test: 0.4255 / 0.239 |
| same | MASSIVE train en only | MASSIVE val en / fr | 491,520 B | 0.945 / **0.552** | 0.031 / 0.060 | |
| When2Call, 4 classes | When2Call train | When2Call val | 524,288 B | 0.720 | 0.147 | val majority 0.436 |
| the bench's 20 tasks, one model per task | `typed_decisions_train` train | its val (360 items) | 581,632 B | **0.703** | 0.204 | per-task prior 0.525 |

- These are validation numbers. When2Call's test has another class prior than its validation (§3.3), so its test bar will be lower; the bench is a different split of the same generator as `typed_decisions_train`, so 0.70 is an estimate of the bar, not a measurement (s5-lite, 17M parameters, scores 0.696 on the bench).
- What they say: at the byte size of s6-nano, a linear model on surface n-grams reads these states well on fixed label sets. The research note's warning (§3.3, §8.5) holds in this repository's own data. A v2 that does not beat this baseline on a source has no reason to be used there.
- What they cannot do: a per-source classifier needs the label set at training time. It cannot rank options that arrive at run time (a tool list, intents it was not trained on) except through a pairwise or similarity model, which is a different and weaker baseline.

## 6. Question 4: what v2 should be

### 6.1 Three options

| | A. General v2, rebalanced | B. Router (recommended) | C. No v2 |
|---|---|---|---|
| what | same graph, games kept, mixture rebalanced towards the evaluated families, MASSIVE en/fr train added, When2Call class-balanced, decision 180 min + DAgger 3 × 15 | same graph and encoder, no games, tool choice + When2Call + MASSIVE en/fr + Banking77 + 25% general decisions, decision 90 min | publish the size-matched linear baselines as the recommendation for fixed label sets; keep s6-nano v1 for games |
| gaps addressed | bench exposure, French, When2Call prior | French, When2Call prior, exposure; drops the bench as a target | none |
| where it can beat the linear baseline | nowhere measured: on every fixed-label source the rehearsal puts the bar at 0.70 to 0.95 | options given at run time (tool choice), request-against-tools reading (When2Call) | — |
| GPU working minutes | ≈ 180 + 45 + 30 eval ≈ 255 | ≈ 90 + 5 + 15 eval ≈ 110 | 0 |
| code | MASSIVE builder; mixture by directory (no `sky-train` change: sampling weights are per directory); baseline script; gate | the same plus a tool-choice builder, a training script, a gate | a baseline script |
| expected outcome | bench 0.52 to 0.60, MASSIVE en 0.75 to 0.90, fr 0.65 to 0.85, When2Call 0.45 to 0.60; below the linear baseline on most sources | see §6.3 | — |

Option A answers the question v1 already answered (a general model at 1.5M parameters is a demonstration), at twice the cost of B. Option C is honest and free, but leaves open the one place where the architecture could matter. Option B follows from the evidence: put the whole budget on the narrow family where reading a state against run-time options is the task, judge it where a linear model of the same size can and cannot follow, and keep the published s6-nano v1 for the games.

### 6.2 The recommended v2 (B), in short

- **Model.** `runs/recipe/configs/s6-nano.json` unchanged, tokenizer `tok-s6-2048`, decision initialisation from `runs/s6-nano/pretrain/init` (copied, not rerun). Same recipe and rate (×1 of the nano pilot). Ternary QAT after 10%, as v1.
- **Data** (directories = sampling groups): `route/tools` (glaive + hermes tool choice: the conversation's tools, distractor tools from other conversations, `no_tool`; 2 to 8 options), `route/when2call` (v1's file, classes reweighted to equal mass), `route/massive` (MASSIVE en + fr train, two label sets per utterance drawn by its own generator, 4 to 12 options), `route/banking77`, plus `decisions` (v1's minus the MASSIVE-derived and CLINC sources) and `distill` (the same filter) so that instructions stay readable. Weights: `tools=0.25, when2call=0.15, massive=0.25, banking77=0.05, decisions=0.20, distill=0.10`.
- **Budget.** 90 working minutes of decision training (≈ 2.55 million draws, as v1). About 1.8 million draws fall on the four routing groups (≈ 100k items: about 17 passes), against about 2 passes for v1's targets; whether that many passes over-fit MASSIVE is the main risk of the budget, and an open choice of the pre-registration (60 or 90 minutes). No DAgger, no arena (no games), full depth only.
- **Judged against:** per-source size-matched linear models trained on the same training labels (MASSIVE, When2Call), a pairwise size-matched linear model and a TF-IDF cosine for tool choice, and potion-base-2M with a linear head (7.56 MB, reported only).

### 6.3 Expected effect, with its uncertainty (judgement, not measurement)

| source | v1 | v2 expected | baseline bar (expected) | chance v2 clears its gate |
|---|---:|---|---|---|
| tool choice (fresh test) | — | 0.70 to 0.90 | pairwise LR: unknown, rehearsed on validation before the freeze | about 50% |
| When2Call test | 0.404 | 0.50 to 0.70 | LR 0.72 on validation; lower on test (prior shift) | about 50% |
| MASSIVE en / fr (now in distribution) | 0.4255 / 0.239 | 0.80 to 0.92 / 0.75 to 0.90 | LR 0.947 / 0.944 on validation | about 35% (non-inferiority at −0.05) |
| pooled ECE over gated sources | 0.0479 (all sources) | 0.02 to 0.05 | — | about 75% |
| all gates together | | | | **about 10 to 20%** |

The ranges are wide because no v1 number exists for tool choice, and because the step from 0.24 to 0.80 in French rests on the assumption that 11k French training utterances are enough for a 1.5M model; the rehearsal shows they are for a linear model.

### 6.4 Code it needs

1. `sky-data` route builder (Rust, beside `s4.rs`): MASSIVE train/val → choice records with randomised label sets (own derived generator), tool choice from glaive/hermes (deduplication across both, split by hash of the user text, a fresh `test` file and its sha256), copies of When2Call and Banking77, the drop list of §4.2, a manifest of counts and hashes. Unit tests on the splits and the drop list.
2. `scripts/train-route.sh`: a new script (not a change to `train-s6.sh`, which s6-micro v2 still has to run from as it is): copy of the init, `s2-decide` with the route records and weights, `s2-calibrate`, `s2-export`, `sky-eval` on the route tests, the baselines, the gate. `DRY_RUN` and `SKY_OUT` as in `train-s6.sh`.
3. `scripts/route-baselines.py` in a pinned environment (scikit-learn 1.9.1, model2vec 0.10.0): the per-source LR, the pairwise LR, TF-IDF cosine, potion; writes `ref-*.json` in `sky-eval`'s format so that one gate script reads both.
4. `scripts/s6-route-gate.py` with tests; `scripts/s6-ece-by-source.py` given a `--tests` list.

### 6.5 Budget, in working minutes

Data build ≈ 20 (CPU; plus the fetch of glaive and hermes), baseline training and rehearsal on validation ≈ 15 (CPU), decision training 90 (GPU), calibration and export ≈ 5, evaluation and baselines on the tests ≈ 15. Total ≈ 110 GPU minutes and ≈ 35 CPU minutes, in the brake modes of amendments 5, 6 and 10; one run.

## 7. What this study read

- v1 test dumps and test files, already read by the v1 evaluation and by `BENCH.md`: the per-task, per-class, per-language and per-length breakdowns of §3 are new readings of those records. Any v2 judged on the same records is designed after reading them; the pre-registration says so.
- Validation and training splits only for the baseline rehearsal (§5). No MASSIVE, When2Call or bench test record was scored by a baseline.
- `data/distill/agreement.json`, `runs/*/log.jsonl`, `runs/all/*.log`, `runs/*/eval/*`, `DATA_LICENSES.md`, `docs/research-s6-usefulness.md` §3, §8 (item 5) and §9.
