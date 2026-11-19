# s6: pre-registration (learning rate pilot, ablation, final model)

*Translated into English on 19/11/2026. The text frozen on 25/10/2026 was written in French; that original, which remains the reference in case of doubt, is this file at commit `1f180a5`. The translation changes no rule, number or threshold.*

**Status: frozen on 25/10/2026, before the pilot, the ablation and any s6 training.** This file fixes the data, the runs, the metrics and the adoption rules. The results will be read through this text, and not through a version written after the fact.

**Changes to this file:**

- every amendment is committed, dated and justified in section 7;
- it precedes the runs it concerns;
- it never touches a threshold after the reading of a result that it settles.

## 1. Data

- All the data come out of `scripts/prepare-data.sh`, run once before the pilot. No regeneration between the pilot, the ablation and the final model: every arm sees the same files.
- The GLM teacher data (`glm_orders`, 3 000 drone orders; `paraphrased`, 40 000 reworded instructions) are present only if `SKY_TEACHER_URL` and `SKY_TEACHER_KEY` are set at the time of the preparation. Their presence or absence is noted in the final report.
- The `val` split is used to choose. No `test` split is read during the pilot or the ablation.

## 2. Learning rate pilot (`cargo xtask s6lr`)

- **Why.** Muon v2 aligns the RMS of its updates with that of AdamW (0,2·√max(A, B)), so AdamW's rate should suit it. But Muon's optimum moves with the width (arXiv 2512.22768), and an untuned AdamW baseline inflates the gain of the other optimizer (Wen et al., arXiv 2509.02046). We therefore tune **both**.
- **Runs.** `softmax-adamw` (`--adam-lr`) and `parallax-muon` (`--muon-lr` only; its AdamW parameters keep the default rate) at ×0,5, ×1 and ×2 of the default rate (3e-3 in pretraining, 1e-3 in decision). Seed 7, the same step budgets as the ablation (4 800 + 6 400). Six runs, about 2 h 30 on an M5.
- **Choice.** For each optimizer, the scale with the best final ternary accuracy (last exit, `val` split). We keep ×1 unless another scale beats it by at least **0,01**: a single seed does not settle less. The intermediate points do not count, because the ranking can flip during the decay of the rate (arXiv 2509.02046).
- **Edge of the grid.** If the choice falls on ×0,5 or ×2, a single extension (×0,25 or ×4) is allowed before the ablation, by amendment.
- **Output.** `runs/s6lr/summary.json`, then `cargo xtask s6ab --adamw-lr-scale a --muon-lr-scale m`.

## 3. Ablation (`cargo xtask s6ab`)

The eleven arms of `docs/SPEC-S6.md` §4, 3 seeds (7, 11, 13), with the pilot's scales. `runs/s6ab/summary.json` gives, per arm, the mean and the standard deviation over the seeds.

**Main metric.** The final ternary accuracy of the last exit (`val.accuracy`).

**Secondary metrics.**

- ECE;
- logit margin;
- effective rank;
- arena score per game (4 games, 30 episodes, τ = 1);
- `train_minutes`, the wall-clock time of the two phases.

Wall-clock time counts: the real trainings are budgeted in minutes, and at equal time AdamW can beat Muon on language at small scale (arXiv 2610.21555).

**Significant gain.** A gap in mean accuracy of at least **0,01**, and larger than the standard deviation of each of the two arms compared.

**Arena loss.** A game whose mean score drops by more than 5 % relative to the reference.

## 4. Adoption rules

1. **Muon v2.** `softmax-muon` against `softmax-adamw`: significant gain, and no arena loss.
   - If the overhead in `train_minutes` exceeds 15 %, the required gain goes up to 0,02.
   - Otherwise we keep AdamW, and all the following Muon arms are read against their AdamW counterpart when there is one.
2. **Parallax.** `parallax-muon` against `softmax-muon` (or the two AdamW arms if rule 1 kept AdamW): significant gain, and no arena loss. This is the rule of `SPEC-S6.md` §4, put in numbers.
3. **XSA.** `xsa-muon` against `softmax-muon`: significant gain, and no arena loss. If Parallax and XSA are both adopted, the final model takes `configs/s6-pico-xsa.json`.
4. **Isolated changes.** SODA, exit KD, EMA, StableQAT and curriculum are read against `parallax-muon`:
   - adopted if the mean gain is at least 0,005;
   - without arena loss;
   - with an ECE that does not degrade by more than 0,01.
5. **Without polish.** `parallax-muon-nopolish` is adopted only if it is non-inferior (≥ −0,005) **and** at least 2 % faster in `train_minutes`.
6. **Combination.** The adopted changes are combined without re-ablation, through the variables of `scripts/train-s6.sh` (`CONFIG`, `OPT`, `PRETRAIN`, `DECIDE`). The risk of interaction is accepted and checked by the final gate.

## 5. Final model and gate (`scripts/train-s6.sh`)

- **Training.**
  - Pretraining 8 min and decision 15 min (budget in time, like s3-pico).
  - Three DAgger rounds of 8 min; the oracle's share is 0,2, then 0,1, then 0,05.
  - Exit thresholds by conformal risk control (α = 0,02).
- **Threshold gate.** Play at the conformal thresholds stays within 2 % of full depth, with no extra crash or veto. Otherwise, full depth is kept, or `scripts/calibrate-exit.sh` is used.
- **Publication gate.** `s6-pico-g3` against the published `s3-pico-g3`, on the arena (4 games, 30 episodes, τ = 1).
  - **Condition 1.** No game with an arena loss.
  - **Condition 2.** At least one game better by 5 %, or the same quality with fewer mean layers at the conformal thresholds.
  - **If the gate fails.** s6 is not published, and the report says so.

## 6. Forbidden

- Rerunning a seed or an arm "to see"; only a technical failure (crash, incomplete run) justifies a new run, noted in the report.
- Choosing on an intermediate checkpoint.
- Changing the data or the budgets between the pilot and the ablation.

## 7. Amendments

**Amendment 1 (28/10/2026, before the pilot: no s6 run has run, no result has been read). Pretraining corpus.**

- **Change.** The MLM pretraining of the pilot, the ablation and the final model reads `data/text-open` then `data/text` (`--text data/text-open,data/text` in `scripts/train-s6.sh` and `cargo xtask s6ab`), instead of `data/text` alone.
- **The added corpus.** `data/text-open` comes from `sky-data fetch-text --set open` and weighs 150 MB. These are the non-web parts of the published training data of two fully open LLMs:
  - Dolma 3 (OLMo 3): Wikipedia EN, scientific PDFs, FineMath;
  - TxT360 (LLM360): PG-19, StackExchange, Europarl in French.

  The licences are in `DATA_LICENSES.md`.
- **Why.** `data/text` contains only stories and web. s6's decisions also deal with encyclopedic, technical and numerical texts, in English and in French. The step budget stays at 4 800 × 128 × 128 ≈ 79 M tokens and the two corpora fit under `--max-tokens` (300 M). The change is therefore about the mix, about half and half, and not about the quantity seen.
- **What does not change.**
  - The tokenizer `data/tok/tok-s2-1024.json` (trained on `data/text`, `data/decisions`, `data/games`); a byte-level BPE, it covers the new text.
  - The budgets, the seeds, the arms, the metrics and the adoption rules.
  - The MLM validation: it stays the last 1 % of the stream, hence the end of `data/text` (`tinystories_fr`), as for s3, thanks to the order `data/text-open` first.
- **Rule §1 unchanged.** `scripts/prepare-data.sh` fetches `data/text-open` at the same time as the rest, once before the pilot, with no regeneration afterwards.

**Amendment 2 (29/10/2026, before the pilot: no s6 run has run, no result has been read). No more s6-pico final model.**

- **Change.** §5 (the s6-pico final model and its gate against s3-pico) is withdrawn. s6 becomes a general-purpose model in two sizes, s6-nano and s6-micro, with their own pre-registration, `docs/s6-gen-prereg.md`, frozen before their first run.
- **What remains.** The pilot (§2) and the ablation (§3) at pico size, as they are: data, arms, seeds, budgets, metrics. Their adoption rules (§4) fix the recipe applied to the two sizes (`docs/s6-gen-prereg.md` §2).
- **Rule 6 (combination).** The variables of `scripts/train-s6.sh` become `OPT`, `LR_SCALE`, `PRETRAIN` and `DECIDE`. `CONFIG` goes away: Parallax and XSA go into `configs/s6-nano.json` and `configs/s6-micro.json`. `OPT` no longer carries a rate. `LR_SCALE` is the pilot's scale for the optimizer kept, and each phase applies it to its own default rate (3e-3 in pretraining, 1e-3 in decision), like `cargo xtask s6ab`. Before, a `--muon-lr` in `OPT` would have given the same rate to both phases, hence a decision rate three times higher than the ablation's.
- **Why.** s6-pico already trains 30 % on general decisions. But at 1 024 tokens and 64/96 of window, 96,8 % of the sequences are truncated, and s3-pico, of the same size, stays below the majority class of `typed_decisions_bench`. A pico cannot be the general-purpose s6. The ablation remains useful there, because it fits in 15 h.
- **s6-secrets-pico and s6-cast** do not change: s6-secrets still takes the options adopted by the ablation (`OPT` and `LR_SCALE`).

**Amendment 3 (29/10/2026, before the pilot: no s6 run has run, no result has been read). Actual size of `data/text-open`.**

- **Finding.** `scripts/prepare-data.sh` fetched `data/text-open` on 29/10/2026. We get 15 011 documents and 99,5 MB of text, and not the 150 MB of amendment 1: the Wikipedia shard of Dolma 3 holds only 4,3 MB, for a budget of 37,5. The numbers per source are in `DATA_LICENSES.md`.
- **Consequence.** The pretraining mix is about 40 % open text and 60 % `data/text` (147 MB), instead of half and half.
- **What does not change.** Nothing else. §1 holds: these files are frozen for the pilot, the ablation and all the runs that follow.

**Amendment 4 (06/11/2026, before the pilot: no s6 run has run, no result has been read). The rules applied by a script, the automatic extension, everything on the M5.**

- **The rules of §4 are applied by `scripts/s6-adopt.py adopt`**, tested by `scripts/test_s6_adopt.py`. No rule and no threshold changes: a gain of at least 0,01 and greater than the standard deviation of each arm, 0,02 if Muon's overhead exceeds 15 %, arena loss beyond 5 %, 0,005 and ECE within 0,01 for the isolated changes, ≥ −0,005 and 2 % faster without polish.
- **The recipe.** The script writes `runs/recipe/recipe.json`: each rule with its figures, the sha256 of the two summaries read and the timestamp. It also writes the derived configs (`runs/recipe/configs/`). This file replaces the manual commit of the choices, and it is committed with the report.
- **The extension at the edge of the grid (§2) no longer waits for an amendment.** `cargo xtask s6lr --extend` launches it if and only if an optimizer chose ×0,5 or ×2, and it then runs at ×0,25 or ×4. The pilot's summary includes it in the choice, with the same rule of 0,01 against ×1.
- **The sequence.** `scripts/s6-recipe.sh` chains the pilot, the extension, the ablation at the pilot's scales, then `adopt`, without any decision by hand between the steps.
- **The machine.** Everything runs on the M5, the machine of the budgets (`TIME_SCALE` = 1).
- **What does not change.** Data, arms, seeds, step budgets, metrics and thresholds.
- **Why.** A choice made by hand between the ablation and the final models is a moment where one reads the results and where one can bend a rule. A tested script applies it to the letter.

**Amendment 5 (09/11/2026, before the pilot: no s6 run has run, no result has been read). The throttled mode.**

- **The finding.** At full power, the M5 heats up too much, and it does not stay open all the time. The run can be suspended (lid closed) or stopped (Mac switched off), and resume later.
- **The mechanism.** `SKY_DUTY` (0,5 by default in `scripts/s6-recipe.sh`) is the share of the time during which the training loop works. Before each step, it rests in proportion to the work done since its last pause. `SKY_COOL` (`45:15` by default in the same script) adds a full rest: each hour of the system clock, 45 min of work then 15 min without GPU computation.
- **The measured time.** The pauses and the rests are removed from the run's clock: `train_minutes` (rules 1 and 5 of §4) counts only the working time. The time during which the Mac is asleep does not enter it either (macOS's monotonic clock stops during sleep).
- **After a stop.** An interrupted step starts again from zero with the same seeds: the step has no result as long as it is not finished. This is a technical failure in the sense of §6, without reading of a result, hence without bias; `runs/s5-logs/` keeps the trace of it, and the report notes it.
- **A technical fix, made before any run.** The trial of 09/11/2026 showed that pretraining died at step 74 (`non-finite loss`), on every arm: a number in the text too large for an `f32` (39 digits or more, 32 lines of the corpus, 31 of them in `data/text-open`) was worth infinity. It is now worth 0, like an unreadable number (`num::extract`, in its three copies). The files of `data/` do not change.
- **What does not change.** Data, arms, seeds, step budgets, metrics and thresholds.

**Amendment 6 (11/11/2026, during the ablation, before the change it describes). Full speed while the Mac is unused.**

- **The request.** The owner asked for full speed (`SKY_DUTY=1`, `SKY_COOL=` empty) whenever the Mac is not in use, starting the night of 11/11/2026, and the throttled mode of amendment 5 the rest of the time.
- **Why no rule moves.** `train_minutes` feeds rules 1 and 5 of §4 only, through four arms: `softmax-adamw`, `softmax-muon`, `parallax-muon` and `parallax-muon-nopolish`. Their 12 runs were all finished in throttled mode before this amendment (116 of the ablation's steps were done). The arms still to run (`parallax-muon-kd` seeds 11 and 13, `-ema`, `-stableqat`, `-curriculum`) are judged by rule 4, on accuracy, ECE and the arena, with fixed step budgets.
- **What is reported.** `train_minutes` of the remaining arms stays in the summary, but it mixes two modes from here on and must not be compared with the earlier arms. The report says which runs ran in which mode.
- **What had been seen.** `runs/s6ab/summary.json` did not exist and no rule had been evaluated. Validation lines of single runs were seen in the log while watching the run. The change follows from the owner's schedule, not from a result.
- **Mechanics.** The run is stopped between two steps when possible and started again with the same command. A step cut short starts over from zero with the same seeds, as in amendment 5.
- **What does not change.** Data, arms, seeds, step budgets, metrics and thresholds.
- **Update, the same night (11/11/2026, 23:30).** The owner then asked not to run at 100%, because the Mac heats up fast. The mode outside working hours is therefore `SKY_DUTY=0.85`, `SKY_COOL=50:10` (the owner's figure), and the run goes back to the throttled mode if macOS reports thermal pressure. Full speed ran from 23:21 to the end of the step in progress, about 10 minutes, on runs of `parallax-muon-kd` (seed 13) and of the next arm. The argument above does not change: no rule reads the time of the remaining arms.
