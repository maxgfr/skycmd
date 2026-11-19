# s6-nano and s6-micro: pre-registration (the general-purpose s6)

*Translated into English on 19/11/2026. The text frozen on 29/10/2026 was written in French; that original, which remains the reference in case of doubt, is this file at commit `1f180a5`. The translation changes no rule, number or threshold.*

**Status: frozen on 29/10/2026, before any training of s6-nano and of s6-micro.** No s6 run has run: neither the pico pilot, nor the ablation. This file fixes the models, the data, the learning rate pilot of each size, the runs, the choice of the checkpoint, the evaluation and the publication gates. The results will be read through this text, and not through a version written after the fact.

**Changes to this file:**

- every amendment is committed, dated and justified in section 8;
- it precedes the runs it concerns;
- it never touches a threshold after the reading of a result that it settles.

## 1. Why

s6 becomes a general-purpose model: the 4 games, and multiple-choice questions in free text. It replaces the s6-pico final model (`docs/s6-prereg.md`, amendment 2). The pilot and the ablation at pico size remain: they are fast, and they serve only to choose the recipe.

The two previous attempts on `typed_decisions_bench` stayed below the majority class (**0,398**):

- **s3-pico** (445 k parameters) scores **0,382**. With 1 024 tokens and windows of 64/96, 96,8 % of the sequences are truncated (tutorial, chapter 12).
- **s4-micro** (5,6 M) scores **0,343**. Its identified cause is a pretraining of 30 min on only 150 MB (tutorial, chapter 22).

s6-nano and s6-micro have wide windows, a larger vocabulary, 300 MB of text (`data/text-open` then `data/text`) and hours of pretraining.

**No teacher** (replaced by amendment 1: the open-source teacher GLM-5.3-flash, remote). No teacher file is on disk (neither `data/s4`, nor `data/s5`, nor the laya or oscar-150m teachers). This v1 is therefore trained without distillation, and this choice is made before the run. The soft targets already present in the datasets (tasksource, procedural, calibrated…) are enough.

## 2. The models

The sizes and the parameter counts were fixed in `s6_parameter_counts` (`crates/sky-train/src/s2/model.rs`) before any run.

| | s6-nano | s6-micro |
|---|---|---|
| config | `configs/s6-nano.json` | `configs/s6-micro.json` |
| parameters (decision / pretraining) | **1 569 457** / 778 065 | **5 849 601** / 2 960 769 |
| vocabulary | 2 048 (`data/tok/tok-s6-2048.json`) | 4 096 (`data/tok/tok-s6-4096.json`) |
| width d / ff | 144 / 288 | 256 / 768 |
| blocks × loops (state) | 2 × 2, exits after 1, 2, 3, 4 | 2 × 3, exits after 2, 4, 6 |
| heads | 2 (dimension 72) | 4 (dimension 64) |
| state / head windows | 256 / 128, then **512 / 160** (amendment 2) | 384 / 128 |

s6-micro has the shape of s4-micro (5 587 201 parameters) plus the Parallax probe rows. Both configs have `fuse_numbers: true`.

**The recipe comes from the pico ablation, applied mechanically.** Before the pilot of the first size, a commit sets the two configs and the variables of `scripts/train-s6.sh` according to the adoption rules of `docs/s6-prereg.md` §4:

- `parallax` stays `true` if rule 2 adopts it, otherwise it becomes `false`;
- `"xsa": true` is added if rule 3 adopts XSA (it adds no tensor);
- `OPT` (Muon v2 or `--no-muon`, `--wd-mode soda`, `--ns-polish 0`), `PRETRAIN` (`--curriculum 0.1`) and `DECIDE` (`--exit-kd 0.5`, `--ema 0.999`, `--qat-grad-a 0.21`) take the adopted changes;
- `LR_SCALE` is the pico pilot's scale for the optimizer kept.

If `parallax` becomes `false`, the same commit corrects the parameter counts of the test. Nothing else changes in the recipe.

## 3. Data

All the data come out of `scripts/prepare-data.sh`, run once before the pico pilot (`docs/s6-prereg.md` §1). It also writes the two tokenizers and the zero-shot sets of MASSIVE (`fetch-commands --only massive`, then `s4-corpus --only-heldout`). Nothing is regenerated between the pilots and the runs of the two sizes.

- **Tokenizers.** `sky-data train-tokenizer --vocab N --num` on `data/text`, `data/text-open`, `data/decisions` and `data/games`.
- **MLM pretraining.** `data/text-open` then `data/text`, plus 10 % of numeric curriculum (`data/numeric`), in sequences of 256 tokens.
- **Decisions.** `data/decisions,data/games,data/numeric` with the weights `decisions=0.55,games=0.35,numeric=0.10`. `data/decisions` counts 831 387 questions in 29 datasets. Only the `train` splits are trained on and only the `val` splits are used to choose.
- **DAgger.** Weights `games=0.4,rN=0.25,decisions=0.3,numeric=0.05`. Keeping 30 % of decisions avoids forgetting language while learning to play.
- **Test sets, never read before the final evaluation:**
  - `typed_decisions_bench`: everything is in `split: test`;
  - MASSIVE en and fr (`data/s4/heldout/zs_massive_*.jsonl`): the test split, not at all in `data/decisions`, hence **zero-shot**;
  - notam (`notam:deel_subject`, `notam:notam_multi`): everything is in test, hence never trained on;
  - When2Call: its test split. Its train and val splits are trained on: it is a test of the same distribution.
- **Banking77 is in distribution.** It is now in `data/decisions`, in train and val only, so s6 trains on it. `zs_banking77` (drawn from these same records) is reported separately, with the mention "trained", and enters no mean and no gate.
- The GLM teacher data (`glm_orders`, `paraphrased`): their presence or absence is noted in the report.

## 4. Learning rate pilot per size (`SIZE=… scripts/train-s6.sh lr`)

- **Why.** Muon's optimum moves with the width (`docs/s6-prereg.md` §2), more than Adam's, even if the loss is flat around it (arXiv 2512.22768). The rate tuned at d = 96 is not necessarily the right one at d = 144 or 256.
- **The centre.** With Muon v2's scale (RMS of the updates aligned with AdamW, 0,2·√max(A, B)), a fixed rate makes the spectral step grow like √d. The spectral condition of μP therefore calls for a rate in √(d_pico / d) (arXiv 2603.00541; "Remark #1: On RMS matched Muon", 2026). The centre of the grid is the pico rate shifted in this way: **×0,816 for s6-nano** (√(96/144)) and **×0,612 for s6-micro** (√(96/256)).
- **Runs.** ×0,5, ×1 and ×2 of the centre, that is `LR_SCALE` × √(96/d) × the scale × the default rate of each phase (3e-3 in pretraining, 1e-3 in decision). The unshifted pico rate (×1,22 of the centre for s6-nano, ×1,63 for s6-micro) stays bracketed by the points ×1 and ×2. Each run does 20 min of pretraining and 20 min of decision, seed 0. Three runs per size, about 2 h.
- **Choice.** The final ternary accuracy of the last exit on `val` (the last validation of `log.jsonl`, in `ema-ternary` mode if `--ema` is adopted). We keep ×1 (the centre), unless another scale beats it by at least **0,01**. On 4 000 validation questions, a smaller gap cannot be told apart from noise: the rule leans towards the centre. The choice is written in `runs/s6-<taille>-lr/summary.json` (taille: the size), and `train` reads it.
- **Edge of the grid.** No extension: a choice at ×0,5 or ×2 is applied as it is.

## 5. The runs (`SIZE=nano|micro scripts/train-s6.sh`)

| | s6-nano | s6-micro |
|---|---:|---:|
| MLM pretraining (batch 64, sequences of 256) | 90 min | 180 min |
| decision (QAT after 10 % in float, loop dropout 0,25) | 90 min, batch 128 | 180 min, batch 64 |
| DAgger: 3 rounds of 15 000 decisions, oracle 0,2 → 0,1 → 0,05 | 15 min per round | 20 min per round |

- The pretraining is budgeted in time, like the previous models. It is done in float: ternary QAT starts only in decision, after 10 % of warm-up, as the work on the move from 16 bits → 1,58 bit recommends (arXiv 2502.11895). An MLM encoder needs many more tokens per parameter than a decoder (EMNLP 2025, "Training compute-optimal transformer encoder models"). The report therefore gives the tokens seen and their number per parameter, with about 200 as an indicative landmark; it is not a gate. The decision batches are grouped by length (`--length-pool 16`): otherwise, the wide windows would pay mostly for padding.
- **Exit thresholds.** Conformal risk control (α = 0,02) on the last round (`sky-arena --calibrate-crc`), in a `-crc` copy. This set of thresholds is kept if, at these thresholds, the arena stays within 2 % of full depth with no extra crash or veto. Otherwise, the copy is deleted and the model stays at full depth.
- **Checkpoint.** `last`, or `last-ema` if `--ema` is adopted, calibrated by `s2-calibrate` on `val` then exported as ternary. Never an intermediate checkpoint.
- Only once per size. Only a technical failure (crash, incomplete run) justifies a rerun, and it is noted in the report.

## 6. Evaluation and gates (`SIZE=… scripts/train-s6.sh eval`, `scripts/s6-gate.py`)

**The model evaluated** is the ternary GGUF of the last DAgger round: at full depth (τ = 1), and at its conformal thresholds if they were kept.

**Language.** `sky-eval` on `typed_decisions_bench`, `zs_massive_en`, `zs_massive_fr`, the When2Call test and notam, with at most 2 000 questions per source (`--max 2000`, the first ones). The measurements:

- the accuracy per source;
- the **ECE over all these sources pooled** (`all`);
- the mean depth and the native latency.

The published s3-pico, s4-micro and s5-lite are run on the same records, at τ = 1.

**Games.** Arena, 4 games × 30 episodes, with the same seeds for the model and for the published s3-pico (`site/public/models/s3-pico.gguf`), at τ = 1.

**Also reported:**

- the WASM latency per decision (`parity-node.mjs` on 200 questions of the bench, with the native/WASM parity);
- the share of truncated records (`sky-train s2-truncation`: the first 500 records of each source, state and head);
- `zs_banking77`, in distribution.

**Publication gates.** Each gate **also** requires two conditions: no game whose mean arena score falls more than 5 % below s3-pico, and an **ECE ≤ 0,08**.

| | `typed_decisions_bench` accuracy (ternary) | announced goal |
|---|---|---|
| s6-nano | **> 0,398** (the majority class) | — |
| s6-micro | **≥ 0,45** | **0,50**, the level of the Qwen3.5-4B teacher of s4 |

- The gate is read at τ = 1. The model is published at its conformal thresholds if the gate also holds at these thresholds. If it holds only at τ = 1, it is published at full depth. If it does not hold at τ = 1, it is not published, and the report says so.
- The two sizes are judged separately: the failure of one does not block the other.
- For s6-micro, the report says without ambiguity whether the goal of 0,50 is reached.

**The report, whatever the result:**

- all the figures above;
- the comparison on the bench with s3-pico (0,382), s4-micro (0,343), oscar-1-17m (0,6775) and s5-lite (0,696);
- the nano → micro curve: accuracy, ECE, latency, size;
- the tokens seen in pretraining, their number per parameter, the epoch reached and the final MLM loss.

**Confidence thresholds.** α = 0,02 is within the range of the work on early exit under risk control (CALM, NeurIPS 2022; "Fast yet Safe", arXiv 2405.20915). ECE ≤ 0,08 is a guard, not a target: uncalibrated networks are between 4 and 10 %, and an encoder calibrated by temperature goes much lower. The report gives the exact ECE.

## 7. Forbidden

- Rerunning a run "to see".
- Choosing on an intermediate checkpoint.
- Reading a `test` split before the final evaluation of the size concerned.
- Changing the data, the budgets or the weights between the pilot of a size and its run.
- Changing the recipe inherited from the ablation for a single size.

## 8. Amendments

**Amendment 1 (29/10/2026, before any s6 run: neither the pico pilot, nor the ablation, nor any size has run, no result has been read). An open-source teacher, GLM-5.3-flash, on all the decision data.**

- **Change.** §1 said "no teacher". s6-nano and s6-micro now also learn from the probabilities of a teacher: **GLM-5.3-flash**, an **open-source** model from Z.ai.
  - Its weights are published under the **MIT** licence ([`zai-org/GLM-5.3-Flash`](https://huggingface.co/zai-org/GLM-5.3-Flash), 321B parameters in a mixture of experts, 18B of them active).
  - It is called remotely on z.ai's API (Anthropic Messages protocol, configured by `ccs`). Since its weights are open, the labels remain reproducible outside this API.
  - **Records sent** (`sky-data teacher label-decisions`, `scripts/distill-glm.sh`): **all** the `train` records of `data/decisions`, that is all the Hugging Face decision datasets, between 2 and 12 options. The soft targets are included.
  - Never sent: `typed_decisions_bench`, notam, banking77, the records already generated by GLM (`glm_orders`, `para:*`), and no `val` or `test` split.
  - On 29/10/2026, this makes **747 198 records**, sent in a randomly drawn order (seed 1234), in 37 360 calls of 20, by 4 parallel workers: beyond that, z.ai throttles and the calls go into retries. On 29/10/2026, the measured throughput was about 2,5 labels/s whatever the number of workers, that is about 80 h of labelling spread over the 5 h usage windows of the z.ai subscription (`scripts/distill-glm.sh loop`).
  - **The question asked**: one probability per option, in the order given, summing to 1; temperature 0,2; state cut at 2 000 characters. A malformed answer (wrong number of options, negative value, zero sum) is discarded.
  - **The mix** (`sky-data teacher mix-labels`): a label is kept only if the teacher's argmax is the gold answer (the agreement filter of s5). The target becomes `0,5 · or + 0,5 · professeur` ["or": gold, "professeur": teacher], in `data/distill/glm.jsonl` (`src = glm:<source>`, split `train`).
  - **Decision weights**: `decisions=0.25,distill=0.30,games=0.35,numeric=0.10`. **DAgger weights**: `games=0.4,rN=0.25,decisions=0.1,distill=0.2,numeric=0.05`, that is still 30 % of decisions.
- **Why.**
  - s4-micro failed without sufficient pretraining and without a good teacher.
  - The best local teachers (oscar-1-150m, 0,7755 on the s5 bench; laya-typed-decisions, 0,7665) would run for hours on the machine, which heats up too much. GLM runs remotely.
  - Trial of 29/10/2026 (2 calls, 20 records, outside `data/`): 20 valid answers out of 20, and 15 agreeing with the gold.
- **Rules.**
  - The labels are produced once, before the pilot of s6-nano, and never regenerated. An interrupted run (quota, network) resumes where it stopped: same selection, same ids.
  - The runs wait until the label file covers at least 95 % of the selection. Otherwise, a new dated amendment decides, before any run.
  - The report gives the exact model, the number of calls and of tokens, and the agreement rate per source (`data/distill/agreement.json`).
- **Contamination.** The teacher sees no test record. Its training data are not published: we cannot rule out that it has seen the public benchmarks. The report says so.
- **Terms.** The model's weights are under MIT. The calls go through z.ai's API, whose terms of use also apply (`DATA_LICENSES.md`).
- **What does not change.** The models, the pilot, the budgets, the test sets, the evaluation and the gates.

**Amendment 2 (29/10/2026, before any s6 run: no training has run, no result has been read). Windows of s6-nano: 512 / 160 instead of 256 / 128.**

- **Change.** `configs/s6-nano.json` moves to `max_state_tokens: 512` and `max_head_tokens: 160`. RoPE has no position table: the parameter count does not change (1 569 457 / 778 065), and `s6_parameter_counts` checks the new windows.
- **Why.** The share of truncated records was measured with the real tokenizers, written on 29/10/2026 by `scripts/prepare-data.sh` (`sky-train s2-truncation`). nano's tokenizer of 2 048 tokens splits more finely than micro's (4 096): with 256 / 128, nano cut the state of **71,6 %** of the questions of `typed_decisions_bench` and the head of 11,2 %. And the bench is precisely its gate.

  | | state truncated (bench) | head truncated (bench) | state truncated (all the test sources) |
  |---|---:|---:|---:|
  | s6-nano, 256 / 128 | 71,6 % | 11,2 % | 15,9 % |
  | s6-nano, 384 / 128 | 9,5 % | 10,0 % | — |
  | s6-nano, 384 / 160 | 9,5 % | 0,0 % | — |
  | **s6-nano, 512 / 160** | **1,8 %** | **0,0 %** | **0,4 %** |
  | s6-micro, 384 / 128 (unchanged) | 4,2 % | 0,0 % | 1,0 % |

  Each row measures 2 000 records per source, except the 256 / 128 row, which measures 500. The test sources are `typed_decisions_bench`, the When2Call test, `zs_massive_en`, `zs_massive_fr` (1 000 records) and `zs_banking77`.
- **Cost.** Longer sequences, hence slower steps. Nano stays about 4 times smaller than micro, whose state is 384 tokens.
- **What does not change.** s6-micro, the tokenizers, the data, the teacher's labels (they do not depend on the windows), the budgets, the evaluation and the gates.

**Amendment 3 (03/11/2026, during the labelling, before any s6 training: no model result has been read). The texts refused by z.ai's moderation.**

- **Finding.** From 29/10 to 02/11/2026, the labelling wrote 350 949 labels out of 747 198. 128 answers were refusals from z.ai's moderation: HTTP 400, code 1301, « contenu potentiellement sensible » ["potentially sensitive content"]. Sources such as moderation, insults or refusals necessarily contain such texts. A refused batch was refused at every pass, and its 19 innocent neighbours were never labelled. The 89 other failures were network cuts, resumed afterwards.
- **Change.** A batch refused by the moderation is split in two, recursively, until each refused text is isolated on its own (`sky-data teacher label-decisions`). These texts go into `data/teacher/glm_labels.refused.jsonl` and are never sent again. They keep only their gold target in `data/decisions` and do not enter `data/distill`. A refusal does not count among the "3 failed batches in a row" that stop a worker.
- **Rules unchanged.** The required coverage stays at 95 % of the selection in labels. The refused texts count as unlabelled. The report gives their number per source.
- **What does not change.** The selection, the question asked, the agreement filter, the weights, the models, the evaluation and the gates.

**Amendment 4 (09/11/2026, before any training of s6-nano and s6-micro: only the pico recipe is running, none of its results has been read). The recipe goes through the file, in throttled mode.**

- **The recipe.** The manual commit of §2 is replaced by `runs/recipe/recipe.json` and the derived configs of `runs/recipe/configs/` (`scripts/s6-adopt.py env s6-nano`, then `s6-micro`), with no choice by hand (`docs/s6-prereg.md`, amendment 4). `configs/s6-nano.json`, `configs/s6-micro.json` and the test `s6_parameter_counts` do not change.
- **The throttled mode.** The run is done in throttled mode (`SKY_DUTY=0.5`, `SKY_COOL=45:15`), as in amendment 5 of `docs/s6-prereg.md`. The budgets in minutes count only the working time, and the report gives the number of steps actually done.
- **What does not change.** Data, sizes, budgets, evaluation and gates.

**Amendment 5 (12/11/2026, before any training of s6-nano and s6-micro: the recipe is written (`runs/recipe/recipe.json`), no score of s6-nano and s6-micro exists). Two modes.** At the owner's request (11/11/2026), the run uses `SKY_DUTY=0.85`, `SKY_COOL=50:10` while the Mac is not in use, and the throttled mode of the previous amendment (`0.5`, `45:15`) the rest of the time, as in amendment 6 of `docs/s6-prereg.md`. Budgets in minutes still count work time only. A change of mode stops the training, which resumes from its last checkpoint (`RESUME=1`). The report gives the steps actually done and the modes used. Nothing else changes.

**Amendment 6 (12/11/2026, before any training of s6-nano and s6-micro: no score of s6-nano and s6-micro exists). The brake without a restart, and a memory cap.** The mode now changes while a run goes on: `sky-train` reads `SKY_BRAKE_FILE` again every few seconds, and `scripts/auto-brake.sh` writes it (the Mac counts as not in use after 10 minutes with no key or pointer input, unless a video or a call keeps the display awake, it runs on battery, or macOS reports thermal pressure). `scripts/train-all.sh` caps MLX's buffer cache at 2 GB, and the script stops a training that holds more than 13 GB of memory (see amendment 4 of `docs/s6-ear-prereg.md`). None of this changes what is computed.

**Amendment 7 (13/11/2026, after the first 8 minutes of the first decision training of s6-nano, before any evaluation of s6-nano or s6-micro: no gate score, no pilot result and no chosen rate exists). A memory defect of the decision training, fixed without changing the update; a night mode.**

- **What happened.** On 13/11 at 00:16 the memory cap of amendment 6 stopped the decision training of the first rate of the s6-nano pilot (x0.5) after 8.2 of its 20 minutes (860 steps): `sky-train` held 15 GB on a 16 GB Mac. Measured afterwards on throwaway folders: the peak of MLX's memory over one step grows with the shape of the batch, and 12% of the batches (128 rows whose states reach 512 tokens) need 10 to 17 GB for a single step; batches with long questions need 5 to 8 GB. The run was working in swap: about 105 steps per working minute.
- **The fix.** `sky-train s2-decide --step-mem-gb 3` (`STEP_MEM_GB` in `scripts/train-s6.sh`): a batch whose step would need more than 3 GB is cut into 2, 4 or 8 equal parts of rows; the gradients of the parts are averaged and one update is applied. The loss is a mean over the rows of the batch, so this is the update of the whole batch: the same items are drawn in the same order, each batch gives one step, and only the order of the floating-point sums changes. A unit test checks the loss and every gradient of the cut batch against the whole one, compiled or not, with QAT or not (`a_step_in_parts_is_the_step_of_the_whole_batch`). The vectors of the 1.7 million loaded questions also drop their spare capacity. On the same throwaway run: peaks of 2.3 GB (median) and 4.5 GB (largest, QAT start aside), and about 200 steps per working minute.
- **What it applies to.** Every decision training of s6-nano and s6-micro: the rate pilots, the first run and the DAgger rounds. The limit changes how a step is computed, not what it computes, so it may be changed between two launches (the estimate was measured on s6-nano; for s6-micro it is checked against `mlx_peak_mb` in the logs of its pilot). The cap of amendment 6 now needs four readings in a row above 13 GB (two minutes): the loading of the decision sets peaks for about a minute.
- **The stopped run.** The 860 steps done in swap are not kept: with them the x0.5 arm would have had fewer steps in its 20 minutes than x1 and x2. Its files are moved to `runs/s6-nano-lr-swap/` and its decision training starts over from its kept pretraining (10,082 steps, finished before the defect, which the pretraining does not have: fixed shapes of 64 x 256). Its log had printed the validation accuracy at the start of QAT (0.409 in float); nothing was decided from it.
- **Earlier runs.** The pico pilot and ablation of `docs/s6-prereg.md` and s6-secrets-pico used the same kind of batches with a smaller model, before the cap existed or under it; their memory was not measured. Their results stand as they are and the report says so.
- **A third brake mode.** At the owner's request (13/11/2026, full speed for the night), `scripts/auto-brake.sh` writes `1.0 55:5` between 23:00 and 08:00 when the Mac has had no input for 30 minutes and the other programs use less than 60% of one core; thermal pressure, battery, a held display or any input bring back the modes of amendment 5. Budgets in minutes still count working time only.
- **What does not change.** Data, models, budgets, the pilot rule, the gates.

**Amendment 8 (13/11/2026, during the first decision training of s6-nano, after 72 of its 90 minutes; the rate pilot of s6-nano is finished and its rate chosen (x1), no gate score of s6-nano or s6-micro exists). A second memory defect of the decision training, fixed without changing a value.**

- **What happened.** On 13/11 at 06:05 the memory cap stopped the first decision training of s6-nano at step 16,020 (72.2 of its 90 minutes): `sky-train` had held 14 GB for two minutes. This is not the defect of amendment 7: MLX's own memory stayed flat for the whole run (peak 6.65 GB) and the run was not slowed (420 to 640 questions per second in its last minutes). Measured on the resumed run, with two memory maps nine minutes apart: the heap grew by 0.5 GB and 3.15 million small blocks while the GPU memory stayed at 4.5 GB. MLX keeps one compiled graph per input shape for the life of the process, and the decision batches keep bringing new shapes (rows, state tokens, static streams and static tokens all vary): about 3 MB per shape, with no bound. A throwaway test on the CPU device shows the same growth with a small graph, and that emptying the cache gives the memory back.
- **The fix.** `sky-train` empties MLX's cache of compiled graphs every 256 new shapes (`SKY_COMPILE_SHAPES`; 0: never). A shape met again is compiled again, about 50 ms, into the same graph, so no value changes: a unit test compares the losses and the gradients with and without emptying (`emptying_the_cache_of_compiled_graphs_changes_no_value`). The cost is working time, an estimated 2% of the steps of a decision run. The decision training also frees the texts of the questions once the sets are loaded (only `s2-eval` reads them).
- **What it applies to.** The binary is built at the start of each launch of `scripts/train-s6.sh`: the pilot, the first run and the rounds of s6-micro, and any relaunch of s6-nano. The first run of s6-nano and its DAgger rounds finish with the binary they started with unless they are stopped again (a round lasts 15 minutes: under 1 GB of growth). The limit changes how long a step takes, not what it computes, so it may be changed between two launches.
- **The stopped run.** It resumes from its last checkpoint (step 15,573, 70.1 working minutes: parameters, optimizer and random generator), as `RESUME=1` provides. The 447 steps done after that checkpoint are done again and the budget counts them once. The run prints its validation accuracy every 1,000 steps, as every run does; nothing was decided from it.
- **What does not change.** Data, models, budgets, the chosen rate, the gates, the 13 GB cap.

**Amendment 9 (13/11/2026, five minutes into the decision training of the first rate of the s6-micro pilot (x0.5); no result of that pilot exists, no gate score of s6-micro). The check of the step memory on s6-micro that amendment 7 announced.**

- **What was read.** `mlx_peak_mb` in the log of that run: 6.3 GB within its first 320 steps, under a limit of 3 GB. The later rise to 8.4 GB comes with the validation at the start of QAT, which is not cut into parts and lasts a few seconds (s6-nano shows the same: 4.7 GB before its first validation, 6.6 GB after). The process held 7.5 to 12 GB, under the cap.
- **Why.** The estimate counted the exits on the state side. s6-nano has 4 exits and 4 layers for a state token (2 layers, 2 loops); s6-micro has 3 exits for 6 layers (2 layers, 3 loops), so its estimate was half of what a step takes.
- **The fix.** The estimate counts the depth (state layers times loops). Nothing changes for s6-nano; for s6-micro more batches are cut into parts, which is the same update (amendment 7). The run also printed its validation accuracy at the start of QAT, as every run does; nothing was decided from it.
- **What it applies to.** The binary built at the launch of the first run of s6-micro, and what follows. The three rates of the s6-micro pilot run in one launch, with the binary built before this fix: they are compared with the same estimate.
- **What does not change.** Data, models, budgets, the pilot rule, the gates, the limit of 3 GB, the 13 GB cap.

**Amendment 10 (13/11/2026, during the rate pilot of s6-micro). The night mode at 95%.** At the owner's request (13/11/2026), the night mode of `scripts/auto-brake.sh` is `0.95 55:5` instead of `1.0 55:5`, from the night of 13/11 on. Budgets still count working time only: nothing else changes.

**Amendment 11 (14/11/2026, during the first decision training of s6-micro, after 126 of its 180 minutes; the rate of s6-micro is chosen (x0.5), no gate score of s6-micro exists). The memory of the compiled decision step, measured: the decision training runs uncompiled.**

- **What happened.** On 14/11 at 00:21 the first decision training of s6-micro was stopped by hand at step 27,280 (126.2 of 180 minutes): it had been working in swap since about minute 92, at 110 questions per second instead of 205. Its heap held 9.3 GB in 43 million blocks, 8.3 GB of them in swap, against 4.4 GB after loading; MLX's own memory was flat. The memory cap had not stopped it (the process stayed between 11 and 14 GB, never four readings in a row at 13).
- **What amendment 8 got wrong.** Emptying MLX's cache of compiled graphs does not bound the memory of this run. Measured on 14/11 with no training running, on throwaway folders with the s6-micro command: (1) uncompiled (`--no-compile`), the heap does not grow at all (12.18 million blocks over 240 steps); (2) compiled, it grows by about 3,300 blocks and 0.65 MB per step in the first minutes, the same with the cache emptied every 8 shapes or every 256; (3) a decision batch has far more shapes than its rows and lengths (the number of options and of static streams vary too): hundreds in the first 800 steps; (4) each compilation takes about 8,400 blocks and emptying the cache gives back two thirds of them, so every compilation leaves about 0.4 MB behind, and emptying the cache causes more compilations. On a set with few shapes (`data/numeric` alone) the compiled heap is flat. Where the 0.4 MB stays inside MLX is not known.
- **The fix.** `scripts/train-s6.sh` passes `--no-compile` to the decision training (`DECIDE_COMPILE=1` brings the compiled step back). It is the same loss and the same gradients computed without MLX's graph compilation: the unit tests `compiled_and_eager_decision_steps_agree` and `a_step_in_parts_is_the_step_of_the_whole_batch` compare the two; only floating-point rounding differs. Measured speed on s6-micro: 195 to 235 questions per second uncompiled, against 205 on average for the compiled run before it swapped. `sky-train` now writes `heap_mb` in its log, so the memory of the process outside MLX is recorded.
- **What it applies to.** The rest of the first decision training of s6-micro and its DAgger rounds. s6-nano, the rate pilots and the first 126 minutes of this run were trained compiled; their results stand as they are.
- **The stopped run.** It resumes from its last checkpoint (step 27,161, 125.8 working minutes), as `RESUME=1` provides. The steps done in swap between minutes 92 and 126 are kept: the run has fewer steps in its 180 minutes than it would have had, and the report says so. The run printed its validation accuracy every 1,000 steps, as every run does; nothing was decided from it.
- **What does not change.** Data, models, budgets, the chosen rate, the gates, the limit of 3 GB per step, the 13 GB cap.

**Amendment 12 (14/11/2026, after the gates of s6-nano and s6-micro were read; s6-cast is training, nothing else runs). What the gate of s6-micro measured, and a second version, s6-micro v2, whose rules are written here before it runs.**

- **The result stands.** s6-micro reaches 0.547 on `typed_decisions_bench` (gate 0.45, announced goal 0.50) and fails the ECE guard: 0.0821 at full depth, 0.0804 at its conformal thresholds, against 0.08. It is not published, and nothing below changes that.
- **Where the ECE comes from (read after the gate, from the files of the evaluation; exploratory, not a test).**
  - The temperatures fitted on `val` are all between 0.8 and 1.16 for the ternary model, and the ECE on `val` at the last exit is 0.037 (last DAgger round): in distribution the model is calibrated. A new temperature would not move the gate.
  - Over the whole pool the mean confidence (0.393) is close to the accuracy (0.378). The error is not one direction: the model is under-confident on `typed_decisions_bench` (0.482 against 0.547) and `zs_massive_en` (0.385 against 0.518), over-confident on When2Call (0.537 against 0.431) and on notam.
  - The largest part is `notam:notam_multi` (ECE 0.256 for s6-micro, 0.071 for s6-nano). On these zero-shot NOTAM texts both models give nearly the same answer to every item of a question, at a confidence close to 0.5. One statement ("The affected area is described by an arc, a bearing and distance, a sector…", 412 of the 11,000 items, false for 99.5% of them) is answered "true" by s6-micro for almost all items (accuracy 0.005, mean probability of "false" 0.43) and "false" by s6-nano (accuracy 0.995, mean probability of "false" 0.60). These 412 items alone weigh up to about 0.02 in the pooled ECE of s6-micro. On the multi-class question of the same source s6-micro gives the same class to 298 of 304 items.
  - So the ECE over the pool is sensitive to which side of 0.5 a model falls on a few zero-shot questions it cannot read. The gate is not changed for that: it was written before the runs. From now on the reports also give the ECE of each source and a bootstrap interval of the pooled ECE (1,000 resamples of the items, 95%), as reported numbers, not gates.
- **s6-micro v2: what changes, and why.** In its 180 minutes of decision training s6-micro saw 38,826 batches of 64, about 2.5 million questions: as many as s6-nano (19,893 batches of 128), a model with four times fewer parameters. Part of these minutes were also spent in swap (amendment 11). v2 gives the decision training **360 minutes** instead of 180, uncompiled from its first step (amendment 11). Nothing else changes:
  - the same config, tokenizer, data, teacher labels, mix weights and recipe (`runs/recipe/recipe.json`);
  - the same pretrained encoder: v2 starts from the pretraining of s6-micro (`runs/s6-micro/pretrain/init`), which is not run again;
  - the same rate (x0.5, chosen by the pilot of s6-micro; no new pilot), with its schedule stretched over the 360 minutes;
  - the same 3 DAgger rounds of 20 minutes, the same conformal thresholds rule, the same evaluation on the same records, the same gate (§6): `typed_decisions_bench` ≥ 0.45, ECE over the pool ≤ 0.08, no game more than 5% under s3-pico.
- **Rules of v2.**
  - One run. Only a technical failure justifies a restart, and the report says so. No intermediate checkpoint is chosen, and no training on any evaluated source is added (notam and MASSIVE stay zero-shot).
  - v2 writes under its own folders (`runs/s6-micro2*`, `models/s6-micro2`); the files of the first version are kept.
  - If v2 passes its gate it is published as `s6-micro`, and every place that reports it says that a first version failed the ECE guard at 0.0821 and that v2 was designed after that reading, on the same test records: its gate is not a first look. If it fails, s6-micro stays unpublished, both results are reported, and no third run of this recipe is made; a later attempt needs a new design, pre-registered.
  - The report gives the two versions side by side: accuracy and ECE by source, the bootstrap interval of the pooled ECE, steps, questions seen.
- **When.** After the runs already queued (s6-cast, s6-sleep, LFM3 S1/S2), in the same modes (amendments 5, 6 and 10).

**Amendment 13 (14/11/2026, after the gates of s6-nano and s6-micro were read). The WASM line of the reports compared two depths; a model published at full depth carries its thresholds at 1.**

- **What was wrong.** The evaluation replays 200 questions in the browser engine and compares them with the native dump. The dump is made at full depth (`--exit-threshold 1.0`), but the browser engine reads the thresholds stored in the file, and the file of the last round stores the calibration default (0.5). So the WASM side exited early on 69 of the 200 questions of s6-nano: the lines "max |p_wasm - p_native| 4.51e-02, argmax agree 199" (s6-nano) and "5.22e-02, argmax agree 199" (s6-micro) measure early exit against full depth, not the engine, and their latencies (7,321 and 35,269 µs) are those of early exits at 0.5. These lines are reported numbers, not gates: no gate changes.
- **Measured on 14/11 for s6-nano**, same 200 questions, the file with its three thresholds set to 1 (`gguf-set`): max |p_wasm - p_native| 9.5e-7, argmax agree 200 of 200, median 15,150 µs per decision under Node.
- **The fix.** `scripts/train-s6.sh eval` runs the WASM comparison on a copy of the model with its thresholds at 1 (`eval/full-depth.gguf`). The published reports give the corrected line for s6-nano and say where the first one came from.
- **Publication at full depth.** "Published at full depth" (§6) means the published file stores a threshold of 1 for the three question types (`gguf-set`), since consumers read the thresholds from the file. The weights are those of the evaluated file, byte for byte.

**Amendment 14 (14/11/2026, after the gates of s6-nano and s6-micro were read; found while preparing the publication of s6-nano). English MASSIVE is not zero-shot.**

- **What §3 got wrong.** §3 says the MASSIVE test sets are "not at all in `data/decisions`, hence zero-shot". That is false for English: `data/decisions/calibrated_typed.jsonl` holds 3,113 training questions (and 182 of validation) of the source `calibrated:mteb/amazon_massive_intent`, MASSIVE intents asked with 3 options and another instruction, and 185 of them carry a teacher label in `data/distill/glm.jsonl`. 206 of the 2,000 English test utterances of `zs_massive_en` are also the state of a training question (201 in that source, 5 in `calibrated:clinc/clinc_oos`). None of the 1,000 French test utterances is.
- **What it changes in the numbers (computed from the evaluation dumps, 14/11).** The overlapping questions are not answered better: s6-nano 0.393 on the 206 against 0.429 on the 1,794 others (0.4255 on all); s6-micro 0.495 against 0.521 (0.518 on all). Without the 206, the pooled ECE is 0.0486 for s6-nano (0.0479 with them) and 0.0837 for s6-micro (0.0821).
- **The gates do not change, and neither does their reading.** They are on `typed_decisions_bench`, the pooled ECE and the games: s6-nano passes with or without these questions, s6-micro fails with or without them. The records, the 2,000 first questions per source and the 15 bins of the ECE stay as written.
- **The wording changes.** English MASSIVE is reported as "intents of a trained source, asked in another format (8 options), 10% of the test utterances seen in training", not as zero-shot. French MASSIVE and notam stay zero-shot. The reports give the English score on all 2,000 questions and on the 1,794 unseen ones.
- **s6-micro v2 (amendment 12).** Its sentence "notam and MASSIVE stay zero-shot" reads: no training data is added or removed, the data are those of the first version, overlap included, so that the two versions compare.
- **Also noted (exploratory).** The pooled ECE depends on the number of bins: with 10 bins instead of the 15 of the code (`sky-metrics`, fixed before these runs), s6-micro reads 0.050 and s6-nano 0.048. The gate keeps 15 bins.

**Amendment 15 (15/11/2026, after the gate was read: it failed). The file of the failed model is published, as a result.** On 15/11/2026 the owner of the project decided that the files of the models that failed their gate are published too, so that every result can be checked and run by anyone. Nothing of the reading changes: the gate failed, the model is recommended for no use, and wherever its file is offered (model card, site, manifest) the first lines say that it failed its pre-registered gate and give the numbers. The sentences of this pre-registration that tied publication to the gate said what a pass would mean; for the file they are replaced by this decision, for the verdict they stand. This concerns s6-micro, whose file is published under the name `s6-micro-v1` (the last DAgger round, exit thresholds at 1): the name `s6-micro` stays free for a version that passes, as amendment 12 says, and the second version described there is still to be run under its own rules.

**Amendment 16 (15/11/2026, 13:08, before the run of s6-micro v2; no v2 score exists). The launch of the second version defined by amendment 12.** The queued runs are over (s6-cast, s6-sleep, LFM3 S1/S2). A smoke run of the launch (two minutes of decision training, one DAgger round, under a throwaway prefix) ended with exit code 0: the pretrained encoder copied into the prefix has the hash of v1's (`dbdfb06b…95c8`) before and after, no file of v1 was written, and the steps chain; its numbers are not looked at. The run starts now from the main checkout at commit `9bbed64`, with everything under `runs/s6-micro2/`, v1's folders read-only for its duration, the brake file of the machine, and the budgets of amendment 12 (360 working minutes of decision training, uncompiled; 3 DAgger rounds of 20 minutes; rate x0.5). The trainer's source has changed since v1 was trained (the s6-sleep and s6-cast v2 code were merged; the options they add are off here), so the binary is not v1's: this is said in the report. As amendment 12 wrote: one run; if it passes it is published as `s6-micro` with the disclosure that v1 failed at 0.0821 and that v2 was designed after that reading on the same test records; if it fails, no third run of this recipe. Reported with it, after the study of s6-nano (`docs/s6-nano2-study.md`): for each test source, the share of items that get the model's most frequent answer, and the score of answering each question with its most common training answer.

**Amendment 17 (16/11/2026, after the gate of s6-micro v2 was read; it passed). The result of the second version, and its publication as `s6-micro`.** Nothing in this amendment changes a threshold, a record or a rule; it reports the run of amendments 12 and 16. The result files are committed under `runs/s6-micro2/` (`runs/s6-micro/eval/results.md`, `report.md`, the JSON summaries, the readings below, the chain log `micro2.log` and the logs of the training, DAgger and exit steps).

- **The verdict.** Read by `scripts/s6-gate.py` at τ = 1: `typed_decisions_bench` **0.553** (gate ≥ 0.45; v1 0.547), ECE over the pool **0.0588** (guard ≤ 0.08; v1 0.0821), and the four games at s3-pico's mean scores, which are also the oracle's (Flappy 200.00, Rescue 3.00, Dodge 9.47, Doom-like 3.27). **The gate holds at τ = 1: s6-micro v2 passes.** The announced goal of 0.50 is met (0.553). The conformal thresholds (α = 0.02: choice 0.51, noul 0.50, score 1, mean 2.0 to 2.7 block passes of 6 in the arena) keep the four scores, with no crash, but add one Dodge veto (21 against 20 at full depth, `micro2-exit.log`): §5 allows no extra veto, so they are not kept, and the model is published at full depth.
- **Not a first look.** v1 failed the ECE guard at 0.0821. v2 was designed after that reading (amendment 12), and its gate was read on the same test records, with the same sources, the same 2,000 first questions per source and the same 15 bins. Its pass is a second look at these sets, and every place that reports it says so.
- **What differs from v1.** 360 working minutes of decision training instead of 180, uncompiled from the first step; the trainer binary is not v1's (amendment 16). Everything else is v1's: config, tokenizer, data (the English MASSIVE overlap of amendment 14 included), teacher labels, weights, recipe, rate ×0.5, the pretrained encoder (31,700 steps, 0.52B tokens, 93 per parameter, epoch 5, MLM loss 2.081; not run again), three DAgger rounds of 20 minutes, the evaluation and the gate. `micro2.log` counts 0 files of v1 touched by the run.
- **Steps actually done.** Decision training: 65,471 steps of 64, 4,190,144 questions (v1: 38,826 steps, 2,484,864 questions). DAgger rounds: 3,822, 4,673 and 4,969 steps (v1: 5,951, 5,898 and 5,952). Final validation accuracy at the last exit, ternary: 0.633 after the decision training, 0.621 after the third round (each round's validation set includes its own DAgger data).
- **Deviations, all of them.**
  1. **One technical restart.** The run started on 15/11 at 13:08 (commit `633f1a1`, the commit of amendment 16, whose code is that of `9bbed64`). On 15/11 at 21:40 the decision training died on "Operation not permitted" while a checkpoint was being renamed: the session that ran it lost access to the folder. It had reached step 62,520 (335.7 working minutes). It was resumed on 16/11 at 00:55 (commit `d46aad1`, which differs from `633f1a1` only by documentation files) from its last checkpoint, `decide/last` at step 61,051 (327.5 of 360 minutes: parameters, optimizer and random generator), as amendment 12 allows for a technical failure. The 1,469 steps done after that checkpoint were discarded and done again; the budget counts them once. The training ended at step 65,471.
  2. **A loaded machine for the end of the run.** The 32.5 resumed working minutes of decision training ran while other CPU jobs ran on the machine; a competing CPU job was then paused until the training ended. Measured from the `tok/s` lines of `micro2-train.log`: the first launch ran at a median of 26,018 tokens/s overall (31,000 to 31,700 per 20-minute window over its first 120 minutes, 14,600 to 23,600 per window from minute 140 on, as the brake mode and the use of the machine changed; 19,300 to 20,400 in the hour before the checkpoint), and the resumed part at a median of 17,550 tokens/s (124 questions/s): 10,400 to 11,300 in its first ten minutes, 20,400 to 21,400 in its last thirteen. The three DAgger rounds and the exit and evaluation steps also ran alongside CPU jobs, and the evaluation ran while another model trained on the GPU. The DAgger rounds made fewer steps than v1's in the same 20 working minutes (above). The latency figures of this result (native median per decision, WASM median) were therefore measured on a loaded machine and are not comparable with v1's.
  3. **The trainer binary is not v1's** (amendment 16): the s6-sleep and s6-cast v2 code were merged in between, with their options off.
  4. **The standing disclosure of amendment 12** (second bullet above): the gate of v2 is not a first look.
- **The two versions by source** (`sky-eval`, full depth, at most 2,000 questions per source; ECE and mean confidence from `scripts/s6-ece-by-source.py`, which gives back the pooled ECE of `eval-s6-micro-full.json`: 0.058794657221571955 against 0.058794657221571976, a difference of float rounding):

  | source | n | v1 accuracy | v2 accuracy | v1 mean confidence | v2 mean confidence | v1 ECE | v2 ECE |
  |---|---:|---:|---:|---:|---:|---:|---:|
  | `typed_decisions_bench` | 2,000 | 0.5470 | **0.5530** | 0.4815 | 0.5084 | 0.0660 | 0.0498 |
  | `zs_massive_en` (not strictly zero-shot, amendment 14) | 2,000 | 0.5180 | 0.5310 | 0.3846 | 0.4595 | 0.1334 | 0.0715 |
  | `zs_massive_fr` | 1,000 | 0.2750 | 0.3140 | 0.2856 | 0.3261 | 0.0313 | 0.0300 |
  | `when2call` | 2,000 | 0.4305 | 0.4010 | 0.5372 | 0.6119 | 0.1176 | 0.2165 |
  | `notam:deel_subject` | 2,000 | 0.0520 | 0.0925 | 0.1156 | 0.1110 | 0.0636 | 0.0185 |
  | `notam:notam_multi` | 2,000 | 0.3940 | 0.3530 | 0.4979 | 0.4939 | 0.2558 | 0.1409 |
  | **pooled** | 11,000 | 0.3780 | 0.3795 | | | **0.0821** | **0.0588** |
  | 95% bootstrap interval of the pooled ECE | | | | | | [0.0746, 0.0900] | [0.0526, 0.0687] |

  The pooled ECE fell on NOTAM, the English MASSIVE and the bench, and rose on When2Call, where v2 is more over-confident than v1 (mean confidence 0.612 for an accuracy of 0.401). The whole v2 interval is below 0.08; v1's contained it. English MASSIVE as amendment 14 asks (`scripts/s6-massive-unseen.py` on the evaluation dump, which finds the same 206 seen utterances, 201 and 5): v2 scores 0.539 on the 206 utterances also seen in training and 0.530 on the 1,794 others (0.531 on all; v1: 0.495, 0.521 and 0.518); without the 206, the pooled ECE of v2 is 0.0594 (0.0588 with them; v1: 0.0837 and 0.0821).
- **The two readings of amendment 16** (`scripts/s6-answer-priors.py` on the evaluation dumps, `answer-priors.json` and, for v1, `answer-priors-v1.json`). They are defined here, after the gate was read, as in `docs/s6-nano2-study.md` §3.1: a question is an instruction of a source, an answer is the text of the option with the largest probability. (1) The share of items that get their question's most frequent answer, and the number of questions where one answer covers at least 95% of the items. (2) The score of answering each question with its most frequent gold answer among the training records of the same instruction (`typed_decisions_train` for the bench, the train split of When2Call); MASSIVE and NOTAM have no training question of the same form, so they get none. The script gives back the study's numbers for s6-nano (17 of 20 bench tasks at 95% or more, training prior 0.4815, test majority per question 0.5185) and for v1 (7 of 20).

  | source | questions | v1: share of the most frequent answer (questions at ≥ 95%) | v2: same | answering with the most common training answer | v2 accuracy |
  |---|---:|---:|---:|---:|---:|
  | `typed_decisions_bench` | 20 | 0.795 (7) | 0.843 (11) | 0.4815 | 0.553 |
  | `when2call` | 1 | 0.4485 | 0.7025 (`request_for_info`, 1,405 of 2,000) | 0.3590 (`cannot_answer`) | 0.401 |
  | `zs_massive_en` | 1 | 0.090 | 0.076 | none | 0.531 |
  | `zs_massive_fr` | 1 | 0.079 | 0.086 | none | 0.314 |
  | `notam:deel_subject` | 1 | 0.8395 | 0.7045 ("Landing Navaids") | none | 0.0925 |
  | `notam:notam_multi` | 7 | 0.9895 (6) | 0.8895 (3) | none | 0.353 |

  On MASSIVE the options change from item to item, so a low share says the model does not answer one intent everywhere. On the bench v2 is 0.0715 above the most-common-training-answer prior (s6-nano: 0.001 below it), and on 11 of the 20 tasks it gives one answer to at least 95% of the items. On When2Call it predicts `tool_call` 45 times in 2,000 (v1: 246; `tool_call` is 696 of the 2,000 gold answers).
- **Also reported** (`results.md`). `zs_banking77`, in distribution: 0.763 (v1 0.705). Truncated records: the same as v1 (state 14.6%, head 11.6%, either 24.0%; on the bench 4.2% and 0.0%). Native median per decision over the test sources: 8,970 µs (v1 7,346, not comparable: deviation 2). WASM, on the full-depth copy (amendment 13): median 60,673 µs per decision on 200 bench questions under Node, max |p_wasm − p_native| 8.94e-7, the same answer on 200 of 200; this is the first full-depth WASM figure of s6-micro, measured on a loaded machine. Against the published models on the same records: s3-pico 0.382, s4-micro 0.352, s5-lite 0.696 (oscar-1-17m 0.6775 published); v2 is below the two 17M models. From nano to micro v2: bench 0.480 → 0.553, ECE 0.0479 → 0.0588, file 715,136 → 2,420,032 bytes.
- **Publication.** As amendment 12 wrote, v2 is published as **`s6-micro`**: `site/public/models/s6-micro.gguf` is the evaluated file `runs/s6-micro2/runs/s6-micro/eval/full-depth.gguf`, 2,420,032 bytes, sha256 `f66d702121e85e678e9b6ea73e9c01433e968b3a3734e8f68ac58f4cee4340ff`. It is the last DAgger round of v2 (`runs/s6-micro2/models/s6-micro/skycmd-s6-micro-g3-ternary.gguf`) with its three exit thresholds set from 0.5 to 1 (`gguf-set`, amendment 13); parsing the two files shows the other 71 metadata keys and every byte after the metadata identical. The first version stays published under `s6-micro-v1`, as a failed result (amendment 15): its verdict does not change.
