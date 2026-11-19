# Flexible weights: pre-registration (TODO §7)

*Translated into English on 19/11/2026. The text frozen on 09/11/2026 was written in French; that original, which remains the reference in case of doubt, is this file at commit `1f180a5`. The translation changes no rule, number or threshold.*

**Status: frozen on 09/11/2026, before any computation of this §7.** No edit, no GLM generation and no arena measurement of this protocol exist at this date. The code is not written.

**Closed on 19/11/2026 without running** (§9, amendment 1): none of 7.1 to 7.4 was run, and nothing was computed.

*Note added 19/11/2026: "TODO §7" in the title and "the TODO" in §5 refer to the project's former to-do file, `TODO.md`, removed on 19/11/2026; its last version is at commit `0e45b7b` (§7 there is this plan; §10.4 and §10.5, which built on it, are closed in the note at the end of this file).*

**Changes to this file:**

- every amendment is committed, dated and justified in section 9;
- it precedes the computation it concerns;
- it never touches a threshold, an arm or a set of seeds after a test score has been read.

**What is never published.** No edited, adapted or trapped model of this §7 goes to Hugging Face, the site or the manifest. Only the report is. Publishing an edited model would require another pre-registration.

## 1. The starting point

Weight-Space Programming (Takehi, Teoh, Zhang, Khattab, MIT CSAIL, 02/11/2026): an LLM reads and writes network weights directly. The text is neither peer-reviewed nor reproduced, and it rests on a single LLM (GPT-6 Astra). We read it through secondary articles, not at the source. Here, the LLM is **GLM-5.3-flash**, clearly smaller: a failure on our side therefore says nothing about the article, and a success would have to be reproduced.

**What this work can show:** on small ternary models, whether an LLM-guided edit beats a random edit and a baseline without an LLM, with the arena as judge.

## 2. The base model

The floating-point checkpoint of s3-pico no longer exists (finding of 09/11/2026). **Base: s6-nano**, the checkpoint exported by `scripts/train-s6.sh` at the end of tranche 2: `runs/s6-nano-g$ROUNDS/decide/$ckpt` (`last` or `last-ema` depending on the recipe).

- **Why not retrain s3-pico:** it would be hours of GPU for a model of which we would no longer have the same checkpoint as the one published.
- **Whatever the verdict of its gate.** If s6-nano does not pass its gate, the experiment remains valid: it needs a model that plays, not a published model. The exception is a model that does not play at all: mean score ≤ that of a random policy on 3 games out of 4. We then stop and report it.
- **No computation of §7 during a GPU tranche.** The order is 7.1, 7.2, 7.4, 7.3, after tranches 2 to 7.

**The seeds.** Every measurement is made on the arena, 4 games, veto active, like the gate of s6-nano.

- **Dev:** `--seed 1000`, 30 episodes per game. This set is used to guide and to choose, as much as we want.
- **Test:** `--seed 5000`, 200 episodes per game, the same for all arms, **read only once**, after all the arms of an experiment are frozen.

## 3. The targeted defect (a rule, with no choice by hand)

On the dev seeds, for each game: `gap = (oracle − model) / |oracle|` with the mean scores that the arena outputs. **The target game is the one with the largest gap.** A single target game, fixed once and for all for 7.1.

## 4. 7.1: targeted editing without forgetting

**What an edit is allowed to do.** A list of `{parameter, index, delta}` on the floating-point checkpoint, at most **2 000 weights**, each `|delta|` at most equal to the standard deviation of the parameter touched. Then ternary re-export **with the existing calibration**, without recalibrating. The report gives the number of weights whose ternary value actually changed. An edit that changes none is invalid.

**What GLM sees at each pass.** At most 20 000 weights as text, drawn from the last parameters before the final output; the activations on 20 failure states and 20 success states of the target game; the dev score of its last edit. **One pass = one proposal + one dev measurement.**

**Arms** (each frozen on dev before the test):

| arm | definition |
|---|---|
| A. base | the model without an edit |
| B10, B25, B50 | GLM, the best edit on dev among its first 10, 25 or 50 passes. 3 independent loops; the arm keeps the one of the 3 that is best on dev |
| C. random | same parameters, same number of weights and same deltas as B50, indices drawn at random. 5 draws, all on the test, and the report gives the mean |
| D. no LLM | difference of the means of the activations between failure and success states, subtracted on the columns of the same parameters, at most 2 000 weights |
| E. direct ternary | like B50, but the changes are written directly into {−1, 0, +1} of the GGUF. Secondary variant, with no effect on the verdict |

**Guard rule (arm by arm, on the test, episodes paired by seed):**

- the target game gains: mean score higher than A, lower bound of the 95 % CI (paired bootstrap, 10 000 draws) > 0;
- **and** no other game loses more than 2 % of its mean score.

**Verdict.** The hypothesis "LLM editing works here" is accepted if **B50 passes the guard rule and beats C in score on the target game**. Otherwise, the report says that it is not shown. D is reported in the same way: if D passes without an LLM and B does no better, the LLM brings nothing here, and the report says so.

## 5. 7.2: one-shot initialisation

**The test.** GLM receives, without data or score, the description of a game and of the state features, and writes in one answer the weights of the decision head of s6-nano. Then `s2-decide` training with the recipe, from the same `pretrain/init`.

**Arms, 3 training seeds each, 6 400 steps:** default initialisation; GLM (3 independent answers, one per seed); random with the same norms per parameter as the weights written by GLM.

**Measurement.** The TODO speaks of "la qualité de s3-pico" [the quality of s3-pico]. As the architecture and the data have changed, that level is not comparable. It is replaced by: **steps needed to reach, on `games` in ternary validation, the final mean of the default arm minus 0,01**, read in `log.jsonl`.

**Verdict.** Gain shown if the mean of the steps of the GLM arm is at least **20 % lower** than that of the default arm, and also lower than that of the random arm with the same norms. With 3 seeds, the result remains indicative, and the report gives the 9 curves.

## 6. 7.4: weight audit (a known backdoor)

**The trap.** On a copy of s6-secrets-pico: `data/secrets` with 1 % of the training texts carrying a fixed trigger string, and their answer to `secret` forced to "non" [no]. Same recipe and same budget as the real run. Output in `runs/weights/backdoor/`, never in published `models/`.

**The audit, blind.** 4 models (2 trapped, 2 clean, distinct seeds, names drawn at random). GLM classifies each model as trapped or clean, from:

1. the weights alone;
2. the activations on a corpus of probes containing 50 candidate strings, among them the true trigger.

**Prior check.** The trap must work: attack rate ≥ 90 % on `secret_bench` with the trigger and clean accuracy within 0,02 of the clean model. Otherwise, the audit does not take place and the report says so.

**What is reported.** The number of models correctly classified out of 4 and the rank of the true trigger among the 50, with no success threshold: with 4 models, no statistical conclusion is possible. The article finds 0 out of 4 on the weights alone and 3 out of 3 with the activations; that is the point of comparison, not a bar.

## 7. 7.3: flexible weights, with a guardrail

**The model.** The ternary s6-nano stays frozen. An adapter of **rank 4** is added to the logits of the last output, initialised to zero. After each veto, one descent step on the cross-entropy towards the action kept by the veto. The base weights never move.

**Protocol, first in the native arena**, not in the browser. A stream of 300 episodes per game, seed 7000, on which the adapter learns continuously. The learning rate is chosen once, in {0,01 ; 0,03 ; 0,1}, on dev streams (seed 6000), at the best mean score.

**Arms:** frozen; adapter without a guardrail; adapter with a guardrail.

**The guardrail (α = 0,02).** Every 10 episodes, the risk control (`crates/sky-arena/src/crc.rs`) compares the score of the adapter to a frozen mirror that replays the same seeds. If the relative loss exceeds α, the adapter is removed for the rest of the stream.

**Acknowledged limit.** Conformal control assumes exchangeable examples, which an adapter that learns on a stream does not give. Here, the "guarantee" is therefore not one in the strict sense: we **measure** it (frequency of exceeding α on the test stream) and the report writes it that way.

**Verdict.** Result accepted if, on the final half of the stream, the adapter with a guardrail does better than the frozen one by at least 2 % on the target game (§3), without any other game losing more than 2 %. The browser comes only in case of success, as a demonstration, with no new performance measurement.

**Literature (search of 09/11/2026, abstracts only).** Low-rank test-time adaptation on a frozen model: TTL (WACV 2025), BOLT (arXiv 2512.02441). Frozen quantised base and adapters: LoQT (NeurIPS 2024), "Signs Beat Floats" (arXiv 2605.24058). Bounded update and rollback: In-Cell Learning (arXiv 2608.20873). Risk of collapse of adaptation without labels: LAME (arXiv 2201.05718). A search found no work combining these elements with conformal risk control, which does not prove that none exists.

## 8. Reproducibility and report

- GLM-5.3-flash via z.ai (`ccs config get zai`): prompts, answers and seeds written in `runs/weights/`. A cap of 400 requests for all of §7.
- Braked mode (`SKY_DUTY=0.5`, `SKY_COOL=45:15`) for all training, and the arena on one core.
- A technical failure (quota, unreadable format) reruns the same computation identically, and it is noted. No rerun to "see" a better result.
- The first technical trial of the format of the edits uses only the dev seeds. If it forces a change of protocol, it goes through a dated amendment before any reading of the test.
- `runs/weights/REPORT.md` gives all the arms, including the failures. The site and `BENCH.md` take up its result.

## 9. Amendments

**Amendment 1 (19/11/2026). Closed without running: §7 ends here.** Written before any computation of this pre-registration: no edit, no GLM-5.3-flash request, no adapter, no arena measurement and no training of §7 exist, and no code of §7 was written. Nothing was computed for §7.

- **Why.** A literature review of 19/11/2026, [`docs/research-flexible-weights.md`](research-flexible-weights.md), read the source and its follow-ups. The lead decided, on that review, to close all four experiments. In the review's words:
  - **The source** is "a **blog post** published on 2 November 2026, not a peer-reviewed paper, with one LLM (GPT-6 Astra)" (its §0, source [1]).
  - **The project's LLM has already been measured on the post's tasks.** A third-party re-run (one person's runs, one run per cell, with their own harness, its §1.5) gives GLM-5.3-Flash 1/4 on naming a policy's game, 0/6 on reading a transformer's middle blocks, "no edits block" on the car repair, at or below random on writing MinAtar policies (0.0 on Breakout) and an all-zero Go network (§0, source [5]). GLM-5.3-Flash is the only teacher the project's rules allow (`docs/s6-gen-prereg.md`, amendments 1 and 3).
- **The status of each experiment:**
  - **7.1, targeted editing without forgetting (§4): closed, not run.** The review's verdict: "Contradicted for GLM-5.3-Flash in one shot; untested with a tool loop; no LLM-specific advantage over search shown" (§6.1, sources [5][6][12][32]). With tools, an open model repaired the car in 4 of 10 runs, but a greedy search with no LLM made the same kind of edit (§0, [6]).
  - **7.2, one-shot initialisation (§5): closed, not run.** The review's verdict: "Contradicted" (§6.1, [5][17][3]): GLM-5.3-Flash cannot write a working policy [5], and on 20 unpublished games "a single standard PPO update removes most of the fixed actor's success" of a written network [17].
  - **7.4, weight audit (§6): closed, not run.** "Weight statistics detect planted trojans in small models without an LLM" [60][61]; frontier LLMs detect that a model is poisoned but "never found the trigger" [3]; GLM-5.3-Flash was not tested on an audit (§0, §5.2). The review's verdict: "Supported for non-LLM detectors; for LLMs, presence only and with frontier models; untested for GLM-5.3-Flash" (§6.1).
  - **7.3, flexible weights with a guardrail (§7): closed as untested, not run; not refuted.** The review's verdict: "Untested; no LLM involved" (§6.1): no source tests an online adapter whose labels come from a safety veto, and the nearest literature warns of error accumulation and forgetting under online updates [41][42][47] (§0, §5.1). It is closed for a reason of this project, not of the literature: its verdict (§7 above) asks for a gain of at least 2% on the target game, and its base, s6-nano, already plays at the oracle's score in every game at full depth on its gate's 30 seeds (0 crashes, Dodge completion 87%, equal to the oracle; `BENCH.md`, the arena table of the s6 section), so there is no score gap for an adapter to close. The review names that precondition itself (§6.2). The dev-seed gap check that the review proposes before any code (§6.2) was not run.
- **What follows.** None of the four is run, and nothing of §7 is published, as the header of this file already required for models. Reopening any of them needs a new dated pre-registration. What would change the review's reading is listed in its §6.3.
- **The text above §9 does not change.** Two statements of §1 are corrected by the review (§7 of the review): the post has now been read at the source, and its writing and reading results are partly reproduced [3][4].

**Note added 19/11/2026 (moved from the deleted TODO): two follow-ups of this plan are closed.** Both were sections of the project's former to-do file (`TODO.md`, removed on 19/11/2026; last version at commit `0e45b7b`), never pre-registered and never started.

- **§10.4, the flexible weights of 7.3 with sleep and dreaming** (after arXiv 2606.03979: the vetoes of a session consolidated at sleep into a new rank-4 low-rank expert, and dreaming as self-play episodes in the `sky-games` simulators around the states of the vetoes). It was to come only if s6-sleep v3 passed its gate with H1 (sleep passes help at equal wake latency) and 7.3's adapter with its guardrail passed its verdict (§7 above); if either failed, it closed as "not justified". s6-sleep v3 failed H1 on 17/11/2026 (sleep +0.0030 [−0.0010, +0.0068], lower bound not above 0; `runs/s6-sleep3/REPORT.md`, `verdict.json`), so **§10.4 closed as "not justified"** (the lead's decision of 18/11/2026, by the entry's own rule); 7.3 was then closed without running by amendment 1. It could come back only through a new dated pre-registration, and only if a later version shows H1 on its test. s6-sleep v4 does not stand in for v3 here: it closed at its check K1 on 19/11/2026 with no test read (`docs/s6-sleep4-prereg.md` §16).
- **§10.5, in-place test-time training on LFM3-1.7B** (arXiv 2604.06169) was to come only after the LFM3-1.7B fine-tune (§6.1 of the to-do file, `docs/lfm3-ft-prereg.md`), and only if the System 1 / System 2 router showed long-context failures. Its precondition is closed: the fine-tune was closed with nothing trained (`docs/lfm3-ft-prereg.md`, amendment 8, 19/11/2026). **So §10.5 is closed too.**
