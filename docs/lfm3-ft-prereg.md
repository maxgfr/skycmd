# LFM3-1.7B fine-tuned to answer without thinking (pre-registration)

*Translated into English on 19/11/2026. The text frozen on 05/11/2026 was written in French; that original, which remains the reference in case of doubt, is this file at commit `1f180a5`. The translation changes no rule, number or threshold.*

**Status: frozen on 05/11/2026, before any job.** No Hugging Face job has run, no smoke test, no weights have been trained. This file fixes the model, the training data, the training, the evaluation, the gates and the order of the jobs. The results will be read through this text.

**Closed on 19/11/2026** (§10, amendment 8): nothing was trained, and there are no weights. Its code (`scripts/lfm3-ft.sh`, `experiments/lfm3-ft/`) was removed the same day and is in the git history at commit 5011585, the last commit that holds it: [`scripts/lfm3-ft.sh`](https://github.com/maxgfr/skycmd/blob/5011585/scripts/lfm3-ft.sh), [`experiments/lfm3-ft/`](https://github.com/maxgfr/skycmd/tree/5011585/experiments/lfm3-ft). The commands named below refer to that commit.

**Changes to this file:**

- every amendment is committed, dated and justified in section 10;
- it precedes the jobs it concerns;
- it never touches a threshold after a test result has been read.

## 1. Why

The S1/S2 routing (`docs/lfm3-s1s2-prereg.md`) chooses, question by question, whether to think, without touching the weights. It does not fix the defect itself: without thinking, LFM3-1.7B can go into a loop. The smoke test (`runs/lfm3-s1s2-smoke`) showed it: a GSM8K question went into a loop up to the limit of 1 024 tokens and the answer was wrong; with 256 thinking tokens, it was right. Liquid AI's documentation says so too: turning thinking off works inconsistently on reasoning.

This project tries the other way: **a LoRA fine-tuning that teaches the model to answer right and short without thinking (System 1)**, without degrading thinking (System 2) or the rest. It does not replace the router: the router is measured on the base model **and** on the fine-tuned model, with the same protocol (§5).

## 2. Model and environment

- **Base model**: `LiquidAI/LFM3-1.7B`, bf16, custom code (`modeling_lfm3.py`, `trust_remote_code`), 24 layers, hybrid of short convolution and attention (full or sliding-window of 256). Under LFM Open License v1.0.
  - revision: pinned by `lfm3-ft.sh smoke` before any other job, then copied here by a dated amendment.
  - For the record, the repository's `main` branch pointed on 05/11/2026 to `0dc4268b561499c648c7ba7c9466bf3d8d01517a`. Only the revision returned by the smoke is authoritative.
- **Compute**: Hugging Face Jobs (`hf jobs uv run`), account `maxgfr`, uv scripts with pinned dependencies in `experiments/lfm3-ft/`. Nothing runs locally apart from `route` and `gate` (CPU).
- **Inference engine**: vLLM 0.33.0. Fallback: transformers 5.12.1's `generate` if vLLM does not load LFM3 correctly.
  - vLLM 0.33.0 has no native implementation of LFM3 (its registry only has LFM2): it would go through its transformers backend, with no guarantee for the stateful convolution layers. The two engines do not fit in the same environment (vLLM 0.33 requires transformers 5.16 to 5.20, LFM3 was published with 5.12.1).
  - **Decision rule, fixed now**: the smoke runs both engines on the same 20 questions. vLLM is retained if it loads LFM3, runs s1, s2-256 and s2-1024, and agrees with transformers in s1 (greedy) on at least 0,8 of the first 64 tokens on average **and** on at least 0,8 of the answers. Otherwise, the transformers fallback is retained.
  - The engine retained is recorded by an amendment and serves for **all** the following jobs: the generation, and the evaluation of the base as well as of the fine-tuned model.
  - With the fallback: the requests are grouped in batches (128, left padding), each batch receives the seed of its first request, and the log-probabilities come from a forward pass over the finished sequence (raw distribution, like those of vLLM).
- **Repositories, all private**:
  - `maxgfr/skycmd-lfm3-ft-work` (dataset): the artefacts between jobs (smoke, generations, training log, evaluations);
  - `maxgfr/skycmd-lfm3-1.7b-s1` (model): the merged model and the adapter (`adapter/`).

  The scripts refuse to write to a public repository.

## 3. Training data

Only MIT sources. No test question, none of the router's 600 calibration questions.

| source | licence | revision | use |
|---|---|---|---|
| `openai/gsm8k` (`main`, train) | MIT | `740312ad` | questions and written solutions |
| `DigitalLearningGmbH/MATH-lighteval` (train) | MIT | `0530c786` | problems and written solutions (those that have a `\boxed{}`, as in `s1s2.py`) |

**Exclusions** (`select_train`), before any draw:
- the 600 calibration ids (`gsm8k-train-<i>`, `math-train-<i>`, `data/lfm3/calib.jsonl`);
- any question whose normalized text (NFKC, lower case, letters and digits only) is that of a calibration question, a test question (`data/lfm3/test.jsonl`) or one of the 500 MMLU-Pro questions of §5.

About 14 400 questions remain. The exact numbers of exclusions are written by `gen` and `sft`.

**(a) Written solutions, NO-THINK targets.** For each question kept, the data set's solution:
- GSM8K: the calculator annotations `<<...>>` removed, the last line `#### answer` replaced by `\boxed{answer}`;
- MATH: the solution as is (it ends with a `\boxed{}`).

**(b) Self-distillation** (`gen`).
- The base model answers each question kept **4 times** in s2-1024, with the budget forcing and the sampling of `docs/lfm3-s1s2-prereg.md` §4. Seed of sample k: seed of the question (that of `s1s2.py`) + 1 000 003 × k.
- With the transformers fallback, only a subset drawn with seed 1234 is sampled: 1 000 questions per task. The rule is fixed now, because of the throughput.
- A sample is **retained** if it is right (same scoring as `s1s2.py`) and if its answer has a `\boxed{}`.
- Per question, among the samples retained:
  - **NO-THINK target**: the answer after `</think>` that is shortest in tokens;
  - **THINK target**: the complete trace (thinking, `</think>`, answer) that is shortest in tokens, among the samples that closed their thinking themselves. A forced trace is cut in mid-thought: it is never a THINK target.
  - Ties go to the smallest k.

**Format.**
- The prompt is the inference one: LFM3's chat template on « question + Solve the problem. Put the final answer in \boxed{}. ».
  - Thinking off (`enable_thinking=False`, prompt ending with `<think></think>`) for the NO-THINK targets.
  - Thinking on (prompt ending with `<think>`) for the THINK targets.
- The target is followed by the end-of-turn token `<|im_end|>`.
- Examples of more than 2 048 tokens are set aside and counted, never truncated.

**Never used**: the GLM / z.ai labels (the terms of z.ai's API forbid training an external model on its outputs, and they are not maths), and any output of a model other than LFM3-1.7B itself.

## 4. Training (`sft`)

- **LoRA**: r = 32, alpha = 64, dropout 0,05, on all the linear projections except `lm_head` (tied to the embeddings).
  - According to `modeling_lfm3.py` (revision `0dc4268b`), these are: `q_proj`, `k_proj`, `v_proj`, `o_proj`, the attention gate `gate_proj`, the MLP `w1`, `w2`, `w3`, and the projections of the short convolution `in_proj`, `out_proj`.
  - The smoke checks this list on the loaded model. A discrepancy goes through an amendment.
- **Optimization**:
  - 2 epochs, lr 1e-4 with cosine, warmup 3 % of the steps;
  - batches of 8, accumulation 4 (32 examples per step);
  - max length 2 048, bf16, seed 1234.
- **Implementation**: TRL `SFTTrainer` (trl 1.17.1, peft 0.21.2, transformers 5.12.1).
  - The loss bears on the completion only (`completion_mask`). The job checks that the collator masks the prompt before training, and stops otherwise.
  - No intermediate checkpoint: **the last one is the model**.
- **Output**: the adapter is merged into the bf16 weights and pushed to the private repository, with the adapter in `adapter/`. The repository's commit is recorded in `sft/result.json`, and it is the one `eval` evaluates.

## 5. Evaluation (`eval`, then `route` and `gate` locally)

The base model and the fine-tuned model are evaluated the same way: same engine, same code revision, same prompts, same seeds.

- **Modes**: s1, s2-256 and s2-1024, exactly those of `docs/lfm3-s1s2-prereg.md` §4, on the calibration (600) and the test (1 000) of `data/lfm3`.
  - s1 keeps the log-probability of each token, for the router's features.
  - Answer: the last `\boxed{}`, failing that the last number for GSM8K. Same scoring.
- **MMLU-Pro**: `TIGER-Lab/MMLU-Pro` (MIT), revision `b189ec76`, test split. 500 questions drawn with seed 1234, asked in s1:
  - prompt: the question, the options (A) to (J), then « Choose the correct option. Put the letter of the final answer in \boxed{}. »;
  - answer: the letter of the last `\boxed{}`, failing that the last « answer is (X) ».
- **The S1/S2 router** (H1 to H4 of `docs/lfm3-s1s2-prereg.md` §6) is computed for the base **and** for the fine-tuned model, with the same protocol. The probe and τ are fitted on the calibration of the model concerned, then the test is read once (amendment 1 of that other pre-registration). The code is `route` of `s1s2.py`, unchanged, pointed at `runs/lfm3-ft/eval/<base|ft>`.

## 6. Gates (test, fine-tuned against base, same engine)

- **G1 (System 1 repaired)**: s1 accuracy ≥ base + 0,03 **and** share of s1 answers stopped by the limit of 1 024 tokens ≤ 0,5 × that of the base.
- **G2 (System 2 kept)**: s2-1024 accuracy ≥ base − 0,01 **and** mean thinking tokens in s2-1024 ≤ 0,6 × those of the base.
- **G3 (no forgetting)**: s1 accuracy on the 500 MMLU-Pro questions ≥ base − 0,02.

**Verdict:**
- **"LFM3 improved"** if G1 and G3 hold;
- G2 is reported separately: it says whether thinking gets shorter without loss, but does not condition the verdict.
- This verdict is that of the fine-tuning. It is distinct from the router's "LFM3 improved" (H4 of `docs/lfm3-s1s2-prereg.md`), which the report gives under its own name ("H4").

An incomplete evaluation (different ids between the two models, or fewer than 1 000 / 500 questions) gives no verdict: it is reported as such.

The report (`runs/lfm3-ft/report.md`, `gate.json`) gives everything, whatever the result:
- the accuracies per task;
- the stops by limit;
- the tokens;
- H1 to H4 of the router for both models;
- the statistics of the training mix and the loss log.

## 7. Rules

- **No rerun "to see".** An interrupted job (timeout, failure) resumes where it stopped, through the same command: the ids already written are skipped. A technical failure is noted.
- **No checkpoint choice**: only the last one exists.
- **Failures are reported**, like successes.
- The smoke only reads calibration questions, never the test.
- No hyperparameter, no data and no threshold changes after a test result has been read. What changes before goes through an amendment.

## 8. Order of the jobs and cost

`scripts/lfm3-ft.sh`, in this order. Before each paid job, the script displays the machine, the estimated cost and the maximum cost (at the timeout), then waits for a confirmation. **The user confirms the cost before each paid job.**

| step | machine | estimated duration | estimated cost | timeout (max cost) |
|---|---|---|---:|---|
| `smoke` (20 calibration questions, two engines, 2 LoRA steps) | L4 (`l4x1`, 0,80 $/h) | 30 to 40 min | about 0,50 $ | 1 h (0,80 $) |
| amendment: revision and engine | — | — | — | — |
| `gen` (self-distillation) | A100 (`a100-large`, 2,50 $/h) or L40S | about 2 h | about 5 $ | 4 h (10 $) |
| `sft` | A100 | 1 to 2 h | about 5 $ | 4 h (10 $) |
| `eval` base, then `eval` fine-tuned | A100, one job each | about 1 h 30 each | about 7,50 $ | 3 h each (15 $) |
| `pull`, `route`, `gate` | local, CPU | a few minutes | 0 | — |

Estimated total: **about 18 $**, at most 36 $. The prices are those of `hf jobs hardware` as of 05/11/2026.

## 9. Licences and publication

- **The derived weights stay private** until the user has given their agreement.
- If they are published one day, it is under LFM Open License v1.0, with:
  - the text of the licence;
  - a `NOTICE`;
  - the list of modified files (already written to the private repository by `sft`).
- The name does not present itself as "LFM3": the licence gives no right to the trademarks. The repository is called `maxgfr/skycmd-lfm3-1.7b-s1`, "fine-tuned from LiquidAI/LFM3-1.7B".
- Commercial use is allowed only within the limits of the licence (organizations under 10 M$ of revenue).
- The data sources are in `DATA_LICENSES.md`.

## 10. Amendments

**Amendment 1 (09/11/2026, before any run: no generation, no training, no result exists). Everything locally, on the M5.**

- **Compute.** §2 ("Compute") and §8 are replaced: everything runs on the M5, nothing is paid, no Hugging Face job is launched (the user's decision of 05/11/2026). `scripts/lfm3-ft.sh` switches to local mode by default. The artefacts stay in `runs/lfm3-ft/`, the model in `models/lfm3-ft/`, and the two private repositories of §2 are no longer used.
- **Inference engine.** mlx-vlm 0.7.11 on `LiquidAI/LFM3-1.7B-MLX-4bit` (revision `deb2986b`), that of `docs/lfm3-s1s2-prereg.md`: vLLM does not run on a Mac. The results are those of the model in 4 bits, and the report says so. The prompts, the modes, the seeds, the extraction and the scoring do not change (the functions of `s1s2.py` are reused).
- **Cap on the generation.** The throughput rule of §3 (b) applies automatically: 1 000 questions per task (`hf_fallback_subset`, seed 1234), 4 samples each.
- **The LoRA, rule written before the local smoke.** It trains on the bf16 weights of `LiquidAI/LFM3-1.7B`, with the unchanged hyperparameters of §4.
  1. transformers + peft + TRL on MPS, if the smoke validates: `lora_targets_match`, the prompt mask, 2 steps with a finite loss on a batch of the maximum length, the merge and the reload.
  2. Otherwise, mlx-vlm's LoRA trainer, with the same hyperparameters, if it loads LFM3 and validates the same points.
  3. Otherwise, we stop and the report says so. Any other choice goes through a new amendment.
  - **Memory fallback, allowed in advance.** If a batch of 8 at maximum length does not fit in memory, we move to batches of 4 with accumulation 8: still 32 examples per step, and the report notes it.
- **Checkpoints.** §4 ("no intermediate checkpoint") stays true for the choice: **the last one is the model**. Technical save points exist only to resume after the Mac stops. They are never evaluated, and a resumption is noted as a technical failure.
- **Conversion of the fine-tuned model.** The merged model is converted to MLX 4 bits (`mlx_vlm.convert -q --q-bits 4 --q-mode affine`) to be evaluated by the same engine as the base.
  - **Rule written before the smoke.** The smoke also converts the bf16 base with the same command, and compares this conversion with the published base, in s1 (greedy), on 20 calibration questions.
  - If the agreement reaches at least 0,8 on the first 64 tokens on average **and** at least 0,8 on the answers (the thresholds of §2), the published base serves as the reference.
  - Otherwise, **the base is re-evaluated with the local conversion**, so that both models have gone through the same chain.
- **Evaluation of the base.** If the published base serves as the reference, its generations are those of `docs/lfm3-s1s2-prereg.md` (`runs/lfm3-s1s2`, same prompts, seeds and engine), copied as they are into `runs/lfm3-ft/eval/base`. Only the 500 MMLU-Pro questions in s1 are generated in addition. The router (`route`) computes on these files what it has already computed for §6 of the TODO [`TODO.md`, removed on 19/11/2026; last version at commit `0e45b7b`]: no new choice depends on it.
- **The revision.** The local smoke resolves the revision of `LiquidAI/LFM3-1.7B` and returns it in `runs/lfm3-ft/smoke/report.json`. A dated amendment copies it, with the LoRA method retained, before `gen`.
- **The braked mode.** The run is done in braked mode (`SKY_DUTY=0.5`, `SKY_COOL=45:15`), like amendment 5 of `docs/s6-prereg.md`: a pause after each sample and each LoRA step. The generation time and the number of tokens (`seconds`, `tokens`) do not depend on it.
- **What does not change.** The base model, the data, the exclusions, the hyperparameters of the LoRA, the gates G1 to G3, the verdict, and §9: the derived weights stay private until the user has given their agreement.

**Amendment 2 (12/11/2026, before any run: no generation, no training and no result exist). Two brake modes.**

- **Change.** As for the s6 runs (`docs/s6-prereg.md`, amendment 6) and `docs/lfm3-s1s2-prereg.md` (amendment 3), the brake has two modes: `0.5` and `45:15` while the Mac is in use, `0.85` and `50:10` while it is left alone, back to the first under thermal pressure. `scripts/auto-brake.sh` writes the mode to `runs/brake`; the generation and the LoRA callback read it again at every rest (`SKY_BRAKE_FILE`).
- **Why no result depends on it.** The brake sleeps between two samples and between two optimizer steps. The number of LoRA steps, the batches and the generation settings are fixed in steps and tokens, not in minutes.
- **What does not change.** Everything else, the three gates and section 9 included: the derived weights stay private until the owner agrees.

**Amendment 3 (13/11/2026, before any run). A third brake mode.** At the owner's request, `scripts/auto-brake.sh` writes `1.0 55:5` between 23:00 and 08:00 when the Mac has had no input for 30 minutes and nothing else works on it, and the memory cap needs four readings in a row above 13 GB, as in amendment 7 of `docs/s6-gen-prereg.md`. The brake only sleeps between two samples and between two optimizer steps. Nothing else changes.

**Amendment 4 (13/11/2026, before any run). The night mode at 95%.** At the owner's request (13/11/2026), the night mode of `scripts/auto-brake.sh` is `0.95 55:5` instead of `1.0 55:5`, from the night of 13/11 on. Budgets still count working time only: nothing else changes.

**Amendment 5 (16/11/2026, before any run: no smoke, no generation, no training and no result of this pre-registration exist). A shorter plan, at the owner's request ("it is too long"): the adapter is System 1, the base stays System 2.**

- **Why the router failed, in one line of arithmetic.** A cascade pays the first answer on every question: its cost against always thinking is `1/ρ + w`, with ρ the cost of a System 2 answer over a System 1 answer and w the share of questions sent on to System 2 (simple arithmetic on the cost accounting of cascades such as AutoMix, arXiv 2310.12963, where the small model is paid on every query). On the test of `docs/lfm3-s1s2-prereg.md` System 1 used 559 tokens and System 2 958: ρ = 1.71 and `1/ρ` = 0.58, above the 0.5 that H3 asked, whatever the router did. The router could not pass with this System 1. What can change is ρ: a System 1 that answers in a few lines.
- **What is trained: part (a) of §3 only.** The written solutions of GSM8K and MATH as NO-THINK targets, capped at **3,000 questions per task** drawn with seed 1234 among the questions §3 keeps (about 14,400), **one epoch** (about 188 steps of 32 examples). Part (b), the self-distillation (8,000 generations in s2-1024), is **not run**, so no THINK target exists. Everything else of §4 stands (LoRA r 32 on the same projections, lr 1e-4, the prompt masked, the last step is the model). Shortening a model's reasoning with a small LoRA is known to work at larger sizes (TokenSkip, arXiv 2502.12067: on GSM8K with Qwen2.5-14B-Instruct, reasoning tokens go from 313 to 181 for a drop under 0.4%; nothing is published at 1.7B); one model holding a fast and a slow mode is Dualformer's result on planning tasks (arXiv 2410.09918). Here the two modes are kept apart on purpose.
- **The adapter is System 1; the base is System 2.** The fine-tuned weights answer only in s1. In s2-256 and s2-1024 the system is the base model, untouched: its generations are those of `runs/lfm3-s1s2` (same prompts, seeds and engine), as amendment 1 allows when its conversion check holds. Consequences:
  - **G2 is not read**: System 2 is the base model itself, so it cannot be degraded, and nothing shortens its reasoning either. The report says "G2: not read (System 2 is the base)".
  - **`eval` of the fine-tuned model is s1 only**, on the calibration (600), the test (1,000) and the 500 MMLU-Pro questions. The base's s1 on MMLU-Pro is run once (500 questions); its calibration and test generations are reused.
  - **The router is read once more, on the pair**: `route` of `s1s2.py`, unchanged, on a folder that holds the fine-tuned s1 and the base's s2. Its probe and τ are fitted on the calibration of that pair, then the test is read once. H2, H3 and H4 keep the definitions and the bars of `docs/lfm3-s1s2-prereg.md` §6, and are reported under the name "the pair" next to the base's failed reading. It is not re-read on the base and not read on a fine-tuned System 2, which does not exist.
- **Gates.** G1 and G3 are unchanged, and so is the verdict ("LFM3 improved" if G1 and G3 hold). **Reported with them:** H2, H3 and H4 of the pair, the mean tokens of s1 before and after, `1/ρ` and w.
- **What a result would mean.** G1 and G3 hold and the pair passes H2 and H3: a 1.7B model answers as well as when it always thinks, for half the tokens or less, with one adapter and one probe. G1 holds and H3 fails: System 1 is better but still not cheap enough, or the probe sends too much on. G1 fails: written solutions as targets, at this quantity, do not repair System 1; with 6,000 examples instead of about 14,400 and no self-distillation, the failure cannot be told apart from too little data, and the report says so.
- **Cost.** The smoke (about half an hour), the fine-tuning (one to two hours), the conversion, about 2,100 short s1 answers of the fine-tuned model and 500 of the base (about two hours): **about five working hours of GPU instead of fifty or more**.
- **What does not change:** the base model and its revision, the exclusions of §3, the smoke and its rules (amendment 1), the LoRA method chosen by the smoke, the brake modes, the test read once, and §9: the derived weights stay private until the owner agrees.

**Amendment 6 (17/11/2026, before any run: no smoke, no training and no result of this pre-registration exist). Three readings that amendment 5 left open, fixed after a review of the code.**

- **w, the share sent on to System 2.** It is the share of the test questions whose probe score is under τ, whether System 2 then answers or not: these are the questions whose System 2 tokens are paid. The router keeps the System 1 answer when System 2 gives none, and its own "System 1 share" counts those questions as System 1: on the base's reading of 15/11 that share is 0.842 while 275 of the 1,000 questions went to System 2 (w = 0.275, not 0.158). The cascade cost `1/ρ + w` is reported with this w.
- **The budget of 1/ρ.** 1/ρ is the mean System 1 tokens of the fine-tuned model over the mean tokens of the base at the budget the pair's router chooses on its calibration (the rule of `docs/lfm3-s1s2-prereg.md`: the best always-thinking accuracy on calibration; 1,024 on the base's reading), on the test.
- **If the conversion check of amendment 1 says "converted"** (the local conversion of the base disagrees with the published 4-bit file on more than 4 of the 20 questions): the base is evaluated again **in s1 only**, with the local conversion, on the calibration, the test and the 500 MMLU-Pro questions, so that G1 and G3 compare the fine-tuned model with a base on the same conversion (about 2,100 answers, two to three working hours more). System 2 of the pair stays the published base and its generations of `runs/lfm3-s1s2`: it is the untouched model that a user would call, and no System 2 generation is run again. The report says which conversion each number comes from. Amendment 1's full re-evaluation in the three modes is not run.
- **Two guards added by the review, which change no number:** the pair is refused unless each of its six files holds exactly the questions of its split, once each, in its own mode; a fine-tuning that has ended is never run again by a second launch of the step (its result file is the completion mark).
- **What does not change:** everything else of amendment 5.

**Amendment 7 (18/11/2026, after the smoke, before any other step: no training and no evaluation exist, and the smoke read calibration questions only). The pinned revision, the versions, and the smoke's verdict: no LoRA method fits the M5 within its memory bound, so amendment 1's rule 3 applies and nothing is trained.**

- **The first failure of 18/11.** The first smoke stopped at once: `modeling_lfm3.py`, the base model's custom code at revision `1da06bec`, imports `use_kernel_func_from_hub_with_fallback` from `transformers.integrations`, and transformers 5.12.1, the version §4 pinned, has no such name (`runs/all/queue-30-lfm3-smoke.log`).
- **The revision: `1da06bec6a819da10b54c67e5c90972b46d81871`.** It is the `main` of `LiquidAI/LFM3-1.7B` that the smoke resolved, and the last of the repository's seven revisions (27/10 to 09/11/2026; Hub API, 18/11).
  - The seven are identical except for their model card: the custom code (`modeling_lfm3.py`, `configuration_lfm3.py`, `lfm3_conv.py`), `config.json`, the tokenizer, the chat template, `LICENSE` and the weights (one `model.safetensors`, 3,457,635,448 bytes, the same file at every revision).
  - So no older revision runs under 5.12.1: the rule written for this case (keep the versions, pin the last revision whose code imports under them) had no candidate. The revision was pinned and the versions moved.
  - The `0dc4268b` of §2 and §4 has the same code and weights. The licence is the LFM Open License v1.0, unchanged at every revision.
  - The published `LiquidAI/LFM3-1.7B-MLX-4bit` (`deb2986b`, the base of `runs/lfm3-s1s2`) names `LiquidAI/LFM3-1.7B` as its base without a revision. Since the bf16 weights never changed, it can only come from these weights, and the conversion check below agrees.
- **The versions: transformers 5.18.0 instead of 5.12.1, nothing else.** 5.12.1 also lacks `force_accelerate_hooks` and `create_recurrent_attention_mask`, which the custom code imports.
  - 5.15.0 is the first release that exports all three names. But from 5.15.0 to 5.17.0 transformers renames the `conv` layers of custom-code configurations to `linear_attention`, and LFM3's forward then fails (`KeyError`). These releases were checked on a reduced LFM3 built from the custom code.
  - 5.18.0 is the lowest release that works. With torch 2.14.0, peft 0.21.2, trl 1.17.1 and accelerate 1.15.0 unchanged (tokenizers moves to 0.23.1 or later with it), it passed three checks:
    - the reduced model's forward and loss;
    - the full model loaded on CPU (1,728,801,536 parameters) with the chat template, thinking on and off;
    - a coherent greedy answer.
  - transformers only includes LFM3 itself from 5.23.0. The mlx-vlm environment is unpinned, as in `scripts/lfm3-s1s2.sh`, and resolved transformers 5.23.0, mlx 0.32.4 and mlx-vlm 0.7.11.
  - `scripts/lfm3-ft.sh` changes its `py_lora` line and `BASE_REV_PINNED`. MODE=hf, retired by amendment 1, keeps its first pins: it could not load this revision, and it is not used.
- **The memory bound, set before the run that decides (owner's instruction of 18/11/2026).** Each batch size is tried in its own process, which `experiments/lfm3-ft/local.py` (`guarded`) stops at once when either of two limits is passed:
  - the swap in use passes 3 GB;
  - the 1-minute load passes 8.

  Such a batch "does not fit in memory" in amendment 1's sense, and its fallback follows: batches of 4 with accumulation 8.
  - An attempt starts only once the swap is back under 2 GB, and waits at most 20 minutes for it. macOS gives back the swap of the other applications slowly.
  - MPS is capped at its recommended working set (`PYTORCH_MPS_HIGH_WATERMARK_RATIO=1.0`, `PYTORCH_MPS_LOW_WATERMARK_RATIO=0.8`: 11.84 GiB on this Mac). Without the cap, a batch that does not fit swaps instead of raising.
  - The bound reads the whole machine, the owner's applications included: it says what fits on this Mac while it is used, not what a free 16 GB Mac could hold.
- **The runs of 18/11, in order** (logs in `runs/lfm3-ft/smoke-attempts/`):
  1. **18:40.** peft at batch 8 swapped while a macOS update was being staged on the same volume, and the free disk fell to 183 MB. The run was stopped. From then on the smoke deletes the smoke's merged weights once they have reloaded, keeping their size and sha256.
  2. **18:45.** MPS refused every allocation: the default low watermark (1.4) was above the new cap. Both are now set.
  3. **18:45 to 19:07, with the MPS cap only.**
     - peft at batch 8 ran out of MPS memory in its first step: 11.62 GiB allocated, 11.84 allowed.
     - At batch 4, the first step ended: 59 s, MPS driver peak 10.1 GB. The second, with the optimizer's state, ran out of memory: 11.54 GiB.
     - The swap reached 12.7 GB.
  4. **19:07.** mlx-vlm at batch 8 (rule 2) has no hard cap. The swap went from 8 to 12.3 GB in a minute, and the machine, also loaded by other work, restarted.
  5. **20:10.** With a first guard (swap growth over 6 GB), peft at batch 8 pushed the swap from 0 to 4.7 GB in 15 s. The owner's work lagged, and the run was stopped.
  6. **21:11, with the bound above.**
     - Batch 8 was stopped at 3.4 GB of swap in 13 s.
     - Batch 4 was stopped at once: the swap had not come back down.
     - The parent process also held a copy of the model. The module names are now read from the custom code built on the meta device, which loads no weights, and an attempt waits for the swap to settle.
     - The run was stopped and does not count.
  7. **21:16 to 21:39, the run that decides** (`runs/lfm3-ft/smoke.log`, `smoke/report.json`, `smoke/lora.json`, `smoke/mlx-lora.json`, `smoke-watch.log`).
- **What the smoke of 21:16 measured.**
  - The revision above; the chat template, thinking on and off.
  - The LoRA names are exactly §4's ten (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `w1`, `w2`, `w3`, `in_proj`, `out_proj`, on the loaded custom code; `lora_targets_match`).
  - **peft on MPS** (rule 1). Each attempt was stopped by the bound 14 to 19 s after it started, before any optimizer step could end (one step took 59 s in run 3):
    - batch 8: swap 1.87 to 3.51 GB in 14 s, free memory down to 4 %;
    - batch 4: after 510 s of waiting for the swap, 1.91 to 3.44 GB in 19 s, 7 %.
  - **The LoRA trainer of mlx-vlm** (rule 2, same hyperparameters, `smoke-mlx-lora`). Both attempts were stopped the same way:
    - batch 8: after 90 s, 1.98 to 3.43 GB in 15 s, 16 %;
    - batch 4: after 350 s, 2.05 to 3.08 GB in 14 s, 2 %.
  - The 1-minute load stayed under 2.9 during the attempts and under 3.7 over the whole run. No optimizer step ended, so the prompt mask, two finite losses, the merge and the reload were not reached by either method.
  - **The conversion check** passed: 1.0 agreement on the first 64 tokens and 1.0 on the answers, over 20 calibration questions in s1 (bar: 0.8 and 0.8). The local conversion is `mlx_vlm.convert -q --q-bits 4 --q-mode affine` of the bf16 weights at `1da06bec`.
    - This is the first case of amendment 6: the published 4-bit base is the reference, and its generations of `runs/lfm3-s1s2` are reused. The same check gave 1.0 and 1.0 at 19:05 (run 3).
  - **Speeds of the published base** on the M5, without the brake's rests: s1 at 110 tokens/s (20 questions), s2-256 at 98 tokens/s (4 questions). Run 3 gave 99 and 93.
  - **Cost of the run of 21:16.**
    - 23 minutes in all, of which 16 waiting for the swap to settle, about 1 minute in the four attempts and about 4 in the conversion check.
    - The battery went from 80 % to 71 % (the Mac was on battery and in use).
- **The decision, by amendment 1's rule: rule 3.**
  - Neither peft on MPS nor mlx-vlm validated the points of the rule within the memory bound, at 8 or at 4 examples per batch. So nothing is trained, and `sft`, `eval`, `route` and `gate` are not run.
  - `smoke/report.json` says `lora_method: null`. `scripts/lfm3-ft.sh sft` refuses to start (`lora_settings`).
  - Any other choice needs a new amendment written before it runs: a looser bound, a Mac left free, a shorter maximum length, smaller batches with more accumulation, or remote compute.
  - The report says G1 and G3 are not read.
- **Working minutes that remain, if a later amendment selects a method.** These are estimates, with the brake's rests not counted:
  - `sft` cannot be estimated from this smoke, because no step ended under the bound. Run 3, without the bound, did one step of 4 examples of 2,048 tokens in 59 s with peft. 188 steps of 8 such batches would take at most about 25 hours, and much less at the real lengths of the examples.
  - `eval`: the fine-tuned model in s1 on about 2,100 questions, about 3 hours at 110 tokens/s and the base's 559 tokens per answer, less if the answers are shorter. The base in s1 on the 500 MMLU-Pro questions, about 45 minutes.
  - `route` and `gate`: a few minutes of CPU.
- **Deviations from what the earlier amendments said the smoke would do.**
  - Rule 2 (mlx-vlm) was not in the smoke's code. It was written for this smoke (`smoke-mlx-lora`) with the hyperparameters of §4:
    - LoRA r 32, alpha 64, dropout 0.05, on the same ten projections;
    - AdamW at lr 1e-4, cosine, warmup 3 %, no weight decay, gradients clipped at norm 1.0;
    - completion-only loss, gradient checkpointing, length 2,048.
  - Each batch size runs in a child process under the bound above.
  - A step's time is measured without the brake's rests.
  - The module names are read on the meta device.
  - The smoke's merged weights are deleted once reloaded (not reached on 18/11).
  - transformers 5.18.0 instead of 5.12.1.
  - `smoke-mlx` generates without the brake's rests, as before.
- **What does not change:** everything else: the base model, the data, the hyperparameters, the gates, the verdict, the test read once, and §9.

**Amendment 8 (19/11/2026, after amendment 7, before any other step: no training, no evaluation and no gate reading exist). The fine-tune is closed.**

- **The decision.** The owner decided on 19/11/2026 not to rent compute for this fine-tune ("no LFM3 if it does not fit"). Amendment 7 found that no LoRA method allowed by amendment 1 (peft on MPS, then the LoRA trainer of mlx-vlm) fits this 16 GB Mac under the memory bound set before the run that decides. Every other option that amendment 7 lists (a looser bound, a Mac left free, a shorter maximum length, smaller batches with more accumulation, or remote compute) is declined. So the pre-registration is closed here.
- **What exists.** Nothing was trained: there are no weights, no adapter and no fine-tuned model. `sft`, `eval`, `route` and `gate` never ran, and G1 to G3 are not read and never will be. What exists is the smoke of 18/11 and its files (`runs/lfm3-ft/smoke/report.json`, `runs/lfm3-ft/smoke.log`, `runs/lfm3-ft/smoke-attempts/`), which record why nothing was trained.
- **What stays published, unchanged.** The System 1 / System 2 router of `docs/lfm3-s1s2-prereg.md` (inference only, failed: H3, 1,032 tokens against at most 0.5 × 958), its result files `runs/lfm3-s1s2/`, and the data set `maxgfr/skycmd-math` (renamed from `skycmd-lfm3-math` on 19/11/2026). This closing changes no number and no verdict of that router.
- **What does not change:** the text of §1 to §9 and of amendments 1 to 7 stays as the record of what was planned and tried.
