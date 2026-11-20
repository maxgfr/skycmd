# LFM3-1.7B: System 1 first, System 2 only if needed (pre-registration)

*Translated into English on 19/11/2026. The text frozen on 03/11/2026 was written in French; that original, which remains the reference in case of doubt, is this file at commit `1f180a5`. The translation changes no rule, number or threshold.*

**Status: frozen on 03/11/2026, before any run.** Only a smoke test has run: 3 calibration questions per mode, in `runs/lfm3-s1s2-smoke`, never mixed with the run. This file fixes the data, the modes, the router, the metrics and the success criteria. The results will be read through this text.

**The code was removed on 19/11/2026** (the LFM3 lines are closed: `docs/lfm3-ft-prereg.md`, amendment 8). The commands and files this pre-registration names (`scripts/lfm3-s1s2.sh`, `experiments/lfm3-s1s2/s1s2.py`, and `experiments/lfm3-ft/` in its amendment 1) are in the git history at commit 5011585, the last commit that holds them: [`scripts/lfm3-s1s2.sh`](https://github.com/maxgfr/skycmd/blob/5011585/scripts/lfm3-s1s2.sh), [`experiments/lfm3-s1s2/`](https://github.com/maxgfr/skycmd/tree/5011585/experiments/lfm3-s1s2), [`experiments/lfm3-ft/`](https://github.com/maxgfr/skycmd/tree/5011585/experiments/lfm3-ft). The commands named below refer to that commit. The result files `runs/lfm3-s1s2/`, the data set `maxgfr/skycmd-math` (renamed from `skycmd-lfm3-math` on 19/11/2026) and the result stay as they were.

**Changes to this file:**

- every amendment is committed, dated and justified in section 8;
- it precedes the runs it concerns;
- it never touches a threshold after a test result has been read.

## 1. Why

LFM3-1.7B (Liquid AI, 27/10/2026) thinks by default. On a model of this size, a long reasoning can make it give up a good first answer.

- **A community measurement** (30 prompts, budgets 0 / 64 / 128 / 256 / unlimited) finds the optimum between 0 and 64 thinking tokens. With an unlimited budget, 9 answers out of 30 never reach the end.
- **The official documentation** says that turning thinking off works inconsistently on reasoning tasks.
- **The smoke test** shows the other side. Without thinking, a GSM8K question went into a loop up to the limit of 1,024 tokens; with 256 thinking tokens, the answer was right.

The best mode therefore depends on the question. This project does not touch the weights: it chooses, question by question, whether to think. This is the principle of skycmd's calibrated early exits. The weight-editing approach (Weight-Space Programming, issue #75, closed) dealt with a loop bug. It added no capability, according to its own authors.

## 2. Model and environment

- **Model**: `LiquidAI/LFM3-1.7B-MLX-4bit`, revision `deb2986b9660c15ff7dd4c2bb1211ce3bbb814d4` (0.97 GB, 4-bit affine quantization). Under LFM Open License v1.0; the weights are not redistributed.
- **Local inference**: mlx-vlm 0.7.11 (no version of mlx-lm loads LFM3), Python 3.12 via `uv`, `scripts/lfm3-s1s2.sh`.
- **No training of LFM3.** Only the confidence probe is fitted: a logistic regression with 6 coefficients.

## 3. Data (`scripts/lfm3-s1s2.sh data`)

Frozen revisions, random draw with seed 1234:

| split | task | source | questions |
|---|---|---|---:|
| calibration | GSM8K | `openai/gsm8k` (`main`, train), rev. `740312ad`, MIT | 300 |
| calibration | MATH | `DigitalLearningGmbH/MATH-lighteval` (train), rev. `0530c786`, MIT | 300 |
| test | GSM8K | `openai/gsm8k` (`main`, test), rev. `740312ad` | 500 |
| test | MATH-500 | `HuggingFaceH4/MATH-500` (test), rev. `6e4ed1a2`, drawn from the MATH test (MIT) | 500 |

- The MATH calibration questions come from the MATH train split. MATH-500 comes from its test split: no overlap.
- **The test split is read only once**, by `route --split test`. The test generations may run before, without anything being read from them.

## 4. The modes (`gen`)

The same prompt for all modes: the question, then « Solve the problem. Put the final answer in \boxed{}. ».

- **s1 (System 1)**: thinking turned off, that is an empty `<think></think>` block in LFM3's template. Greedy decoding, at most 1,024 tokens. The log-probability of each generated token is kept.
- **s2-B (System 2)** for B = 256 and B = 1,024:
  - thinking on, with the sampling Liquid AI recommends for maths (temperature 1.0, top_p 0.95, top_k 20, no repetition penalty);
  - if B tokens go by without `</think>`, the thinking is closed and the answer is requested (at most 1,024 tokens);
  - one seed per question, derived from its id.
- **The answer** is the last `\boxed{}`. Failing that, for GSM8K, it is the last number.
- **The scoring** is an exact equality after normalization (spaces, `\left`/`\right`, `dfrac`, degrees, percents, `x=`), or a numeric equality. It is the same for all modes.

## 5. The router (`route`)

1. **The confidence score**, computed on the output of s1 alone: a logistic regression on 6 features.
   - the mean log-probabilities of the answer's tokens and of the whole generation;
   - the lowest log-probability;
   - the logarithm of the number of tokens;
   - the presence of an answer;
   - the stop by token limit, that is a loop.

   It is fitted on the calibration split to predict whether s1 is right.
2. **The rule**: if the score is at least τ, the answer of s1 is kept. Otherwise, that of s2 is taken. If s2 gave no answer, that of s1 is kept. The cost of a question routed to s2 counts the tokens of s1 **and** of s2.
3. **The threshold, by conformal risk control** on the calibration split (n = 600):
   - the loss of a question is 1 if the router is wrong while s2 alone is right;
   - τ is the smallest threshold such that (n · R̂(τ) + 1) / (n + 1) ≤ **α = 0.02**;
   - the loss can only go down when τ goes up, so the guarantee holds: in expectation, at most 2 points of accuracy lost compared with s2 alone.
4. **The budget retained, B\***, is the one of s2 whose calibration accuracy is the best. It is chosen before the test is read, and everything is reported for both budgets.

## 6. Criteria (test, budget B\*)

- **H1 (diagnostic)**: the accuracy of s1 and of s2-B\* per task. It says whether thinking helps LFM3-1.7B in maths. No threshold.
- **H2 (accuracy kept)**: router accuracy ≥ accuracy of s2-B\* − 0.02.
- **H3 (compute halved)**: mean tokens of the router ≤ 0.5 × those of s2-B\*.
- **H4 (better than both modes)**: router accuracy ≥ max(s1, s2-B\*) + 0.01.

**Verdict:**
- **success** if H2 and H3 hold;
- **"LFM3 improved"** only if H4 holds too.

The report gives everything, whatever the result:
- the tables per task and per budget;
- the oracle, that is the better of the two answers: the ceiling of a perfect router;
- the share of questions routed to s1, the share of forced thinking, and τ.

## 7. Forbidden

- Reading the test before the router is frozen on the calibration split.
- Changing α, the budgets, the features or the rule after seeing a test result.
- Rerunning a generation "to see". An interrupted generation resumes where it stopped, and a technical failure is noted.

## 8. Amendments

**Amendment 1 (05/11/2026, before any run: only the local smoke test of 3 questions per mode has run, no calibration or test result exists). The run moves to a remote machine, in bf16.**

- **Change.** The run is no longer done locally. This is the user's decision: the M5 heats up too much during the 10 h of inference. The same protocol runs remotely, on Hugging Face Jobs, through `scripts/lfm3-ft.sh eval` (`experiments/lfm3-ft/eval.py`):
  - **model**: `LiquidAI/LFM3-1.7B` in bf16, instead of `LiquidAI/LFM3-1.7B-MLX-4bit` (rev. `deb2986b`). The revision is pinned by the smoke of `docs/lfm3-ft-prereg.md`;
  - **engine**: vLLM instead of mlx-vlm. The log-probabilities of s1 are those of vLLM (`logprobs=1`, the sampled token). If the smoke shows that vLLM does not load LFM3 correctly, the fallback is transformers' `generate`, with the decision rule and the log-probabilities defined in §2 of `docs/lfm3-ft-prereg.md`;
  - **prompt**: the text of the chat template, tokenized without adding a special token (the template already starts with BOS).
- **What does not change.**
  - The prompt, the modes s1, s2-256 and s2-1024: budget, forcing, sampling, one seed per question derived from its id.
  - The answer extraction, the scoring, the 6 features.
  - The router, α = 0.02, the choice of B\*, the criteria H1 to H4 and the forbidden items.
  - The code of `route` is that of `s1s2.py`, unchanged: `experiments/lfm3-ft/route.py` points it at `runs/lfm3-ft/eval/<base|ft>` instead of `runs/lfm3-s1s2`.
  - The functions copied into `experiments/lfm3-ft/common.py` are checked identical to those of `s1s2.py` by `test_common.py`.
- **Applied to two models.**
  - The base model.
  - The fine-tuned model of `docs/lfm3-ft-prereg.md`, with exactly the same protocol. Its probe and its τ are fitted on its own calibration split, and its test is read once.
- **Where `route` runs.** Locally, on CPU, on the generations brought back (`scripts/lfm3-ft.sh pull route`): `route --split calib`, then `route --split test`, once per model.
- **Why.** The probe reads the log-probabilities of the tokens: an engine that returns them is needed. No API serves LFM3 with its log-probabilities as of 05/11/2026 (`TODO.md` §6; removed on 19/11/2026; last version at commit `0e45b7b`). bf16 and 4 bits do not give exactly the same outputs: the results of this run are those of LFM3-1.7B in bf16, and the report says so.

**Amendment 2 (09/11/2026, before any run: only the local smoke test of 3 questions per mode has run, no calibration or test result exists). Back to the original local protocol.**

- **Change.** Amendment 1 is cancelled (the user's decision of 05/11/2026: « finalement je ferai tout sur mon pc » [in the end I will do everything on my PC]). The run is done as in §2: `LiquidAI/LFM3-1.7B-MLX-4bit`, revision `deb2986b`, mlx-vlm 0.7.11, on the M5, through `scripts/lfm3-s1s2.sh`. The results are those of the model in 4 bits.
- **The braked mode.** The run is done in braked mode (`SKY_DUTY=0.5`, `SKY_COOL=45:15`), like amendment 5 of `docs/s6-prereg.md`: a pause after each question. H3 counts tokens and `seconds` remains the generation time: the braked mode changes neither one nor the other.
- **The fine-tuned model.** What concerns the fine-tuned model is now in amendment 1 of `docs/lfm3-ft-prereg.md`. The base generations written here serve there as the reference if that amendment's conversion rule allows it.
- **What does not change.** The data, the prompt, the modes, the budget and the forcing, the sampling, the seeds, the extraction and the scoring, the 6 features, the router, α = 0.02, the choice of B\*, the criteria H1 to H4 and the forbidden items.

**Amendment 3 (12/11/2026, before any run: only the local smoke of 3 questions per mode has run, no calibration or test result exists). Two brake modes.**

- **Change.** As for the s6 runs (`docs/s6-prereg.md`, amendment 6), the brake has two modes: `0.5` and `45:15` while the Mac is in use, `0.85` and `50:10` while it is left alone, back to the first under thermal pressure. `scripts/auto-brake.sh` writes the mode to `runs/brake`, and `s1s2.py` reads that file again at every rest (`SKY_BRAKE_FILE`).
- **Why.** The owner asked for the runs to go faster while the Mac is not used. The two LFM3 tranches are the longest of the big run.
- **Why no result depends on it.** The brake only sleeps between two questions. H3 counts tokens and `seconds` is the generation time of a question, as amendment 2 says. The only possible effect is the one amendment 2 already accepts: a generation measured while the Mac does something else.
- **What does not change.** Everything else.

**Amendment 4 (13/11/2026, before any run). A third brake mode.** At the owner's request, `scripts/auto-brake.sh` writes `1.0 55:5` between 23:00 and 08:00 when the Mac has had no input for 30 minutes and nothing else works on it, and the memory cap needs four readings in a row above 13 GB, as in amendment 7 of `docs/s6-gen-prereg.md`. The brake only sleeps between two questions. Nothing else changes.

**Amendment 5 (13/11/2026, before any run). The night mode at 95%.** At the owner's request (13/11/2026), the night mode of `scripts/auto-brake.sh` is `0.95 55:5` instead of `1.0 55:5`, from the night of 13/11 on. Budgets still count working time only: nothing else changes.
