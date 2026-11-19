# s6-cast: pre-registration (first training and publication gate)

*Translated into English on 19/11/2026. The text frozen on 26/10/2026 was written in French; that original, which remains the reference in case of doubt, is this file at commit `1f180a5`. The translation changes no rule, number or threshold.*

**Status: frozen on 26/10/2026, before any training of s6-cast.** This file fixes the data, the run, the choice of checkpoint, the evaluation and the publication gate. The results will be read through this text.

**Changes to this file:**

- every amendment is committed, dated and justified in section 7;
- it precedes the run it concerns;
- it never touches a threshold after a GIFT-Eval score has been read.

## 1. Training data

- **Synthetic pool** (`sky_infer::cast::synth`): KernelSynth-style GP, impulses, trend-seasonality-irregularities and SARIMA, with the augmentations of Reverso (`docs/SPEC-S6-CAST.md` §3).
- **Real series** (`data/series/`, 30 % of the windows), with a verified original licence and absent from both test benches (`DATA_LICENSES.md`):
  - `sky-data fetch-series`: London smart meters, wind farms (hourly and daily), pedestrian counts and rideshare (Monash, CC-BY-4.0); Wikipedia daily (CC0); USHCN daily (public domain);
  - the trajectories of our four games (`cast-synth --games`).
- One real series in ten (by id) is held out for validation.
- **No series of GIFT-Eval or of Chronos-ZS enters the training.** GIFT-Eval-Pretrain is not used, for lack of a declared licence.

## 2. Learning-rate pilot (`scripts/train-cast.sh pilot`)

- 3 000 steps at 1e-3, 3e-3 and 1e-2 (AdamW; Muon v2 at the same rate), batch 256, without QAT.
- **Choice.** The final validation pinball in f32 (last output). We keep 3e-3 unless another rate lowers it by at least 2 %.
- **Distillation.** Same pilot at the chosen rate, with TinyCast as teacher (weight 0,5, 50 % of the windows). We distil, unless the pinball is worse by more than 1 %: the validation pinball is measured against the truth, and a teacher can degrade it without degrading GIFT-Eval. The threshold is therefore deliberately loose.

## 3. The run (`scripts/train-cast.sh train`)

- `configs/s6-cast-pico.json`: 388 320 parameters.
- 480 minutes (time budget), batch 256, WSD.
- Ternary QAT over the final 20 %, EMA 0,999.
- **Teacher** (unless the pilot rules it out): TinyCast, whose deciles serve as a second target on 50 % of the windows. The loss is `pinball(vérité) + w·L1(déciles du professeur)`, with w going linearly from 0,5 to 0 over the budget. This is the teacher annealing of BAM: the end of training follows the truth alone. A student that learns both the truth and the teacher can surpass the latter (Born-Again Networks, arXiv 1805.04770; Revisiting Self-Distillation, arXiv 2206.08491). TinyCast excludes from its training the sets that overlap the GIFT-Eval and Chronos-ZS tests: it does not leak the test.
- Seed 1, once only. Only a technical failure justifies a rerun, and it is noted in the report.

## 4. Exported checkpoint (`export`)

- **Choice of weights.** The EMA (`last-ema`) if its final ternary validation pinball is lower than that of `last`, otherwise `last`. No intermediate checkpoint is considered.
- **Calibration of the early exit.** `sky-cast-eval calibrate`: α = 0,05, δ = 0,02 (normalised units), on 2 000 synthetic windows never seen.

## 5. Evaluation (`eval`)

- **The bench.** GIFT-Eval, 97 configurations, with the official protocol: `scripts/gift-eval.py`, gluonts `evaluate_model`, the 9 deciles.
  - Each metric is divided by that of the reference seasonal naive (`eval/gift-eval/seasonal_naive.csv`), then the geometric mean is taken.
  - This gives **nGMASE** (point), **nWQL** (probabilistic) and nMSIS.
- **Validation of the bench, before any reading of s6-cast.**
  - Our seasonal naive (point forecast copied onto all the deciles) must give nGMASE at 1,00 ± 0,02. Its nWQL is not 1: the leaderboard's reference is probabilistic. The WQL is therefore validated by TinyCast, below.
  - Our ported TinyCast (with flip, like their submission) must recover 0,774 and 0,545 to within ± 0,02.
  - **Result (26/10/2026): bench validated.** Seasonal naive: nGMASE 0,99986. Ported TinyCast: nGMASE 0,786, nWQL 0,557 (nMSIS 0,578), on the 97 configurations.
    - 95 configurations recover the published MASE to within 1 % (geometric ratio 1,015 over the 97).
    - The gap comes from `bizitobs_l2c/5T` medium and long (MASE 0,85 and 1,25 against 0,49 and 0,48). For this set alone, TinyCast's driver (`tinycast/eval.py`) turns on two settings that are not in its generic predictor:
      - `no_daily`, chosen from the name of the set;
      - a period-alignment subsampling (k = 7), which is triggered only there.
    - Our port is the generic predictor. The bench applies the same protocol to all models, with no per-set setting.
  - If one of these checks fails, s6-cast is not read: the bench is fixed first.
- **Reported model.** s6-cast with and without flip. The reported model is **without** flip: it is the mode of the web page, and flipping doubles the cost.

## 6. Objective and publication gate

**Objective: beat TinyCast on both metrics**, without flip: **nGMASE < 0,774 and nWQL < 0,545**. The report says, without ambiguity, whether the objective is reached.

The bar stays the published one, per-set settings included: it is the most demanding. The report also gives the gap to the generic TinyCast of our bench (0,786 / 0,557, with flip).

s6-cast is published (site, manifest, Hugging Face) if, without flip, it beats the training-free reference FLAIR on both metrics:

- **nGMASE ≤ 0,838**
- **and nWQL ≤ 0,587**

Whatever the result, the report gives the gap to TinyCast (0,774 / 0,545) and to Timestral-5M (leaderboard figures). It also gives the size, the latency and the mean exits used.

If the gate is not passed, the model is not published, and the report says so.

## 7. Amendments

1. **26/10/2026**, before any training:
   - addition of the objective "beat TinyCast" and of the distillation by TinyCast (§2, §3, §6), at the request of the project's owner;
   - the check of the seasonal naive bears on nGMASE alone (§5).

   No score of s6-cast existed.

2. **26/10/2026**, before any training:
   - result of the validation of the bench and origin of the gap on `bizitobs_l2c` (§5);
   - the bar of the objective stays 0,774 / 0,545 (§6);
   - the teacher's targets are computed on fixed cut points of each series (two per series), and no longer at each batch. It is a throughput change: 730 windows/s otherwise, against about 2 000. The distilled share (50 %) and the weight (0,5 → 0) do not change.

   No score of s6-cast existed.

3. **09/11/2026**, before any training (only the pico recipe is running, none of its results has been read):
   - s6-cast takes the optimiser-level options that the pico ablation adopts (`docs/s6-prereg.md`, §4): Muon or AdamW, SODA, without polish, StableQAT, as well as XSA through the derived config. `scripts/s6-adopt.py env s6-cast-pico` gives `CONFIG` and `CAST_OPT`, without a rate;
   - its EMA and its QAT remain its own (§3). The output KD and the curriculum are options of the text decisions, which it does not have;
   - its pilot (§2) decides alone, by its own rules: rate kept at 3e-3 unless a gain of at least 2 %, distillation unless a loss of more than 1 %. `runs/s6-cast-pilot/choice.json` keeps the trace of it, and `train` reads it;
   - the run is done in throttled mode (`SKY_DUTY=0.5`, `SKY_COOL=45:15`), like amendment 5 of `docs/s6-prereg.md`. The 480 minutes count only work time, and the report gives the number of steps actually done.

   No score of s6-cast existed.

4. **12/11/2026, before any training of s6-cast: the recipe is written (`runs/recipe/recipe.json`), no score of s6-cast exists. Two modes.** At the owner's request (11/11/2026), the run uses `SKY_DUTY=0.85`, `SKY_COOL=50:10` while the Mac is not in use, and the throttled mode of the previous amendment (`0.5`, `45:15`) the rest of the time, as in amendment 6 of `docs/s6-prereg.md`. Budgets in minutes still count work time only. A change of mode stops the training, which resumes from its last checkpoint (`RESUME=1`). The report gives the steps actually done and the modes used. Nothing else changes.

5. **12/11/2026, before any training of s6-cast: no score of s6-cast exists. The brake without a restart, and a memory cap.** The mode now changes while a run goes on: `sky-train` reads `SKY_BRAKE_FILE` again every few seconds, and `scripts/auto-brake.sh` writes it (the Mac counts as not in use after 10 minutes with no key or pointer input, unless a video or a call keeps the display awake, it runs on battery, or macOS reports thermal pressure). `scripts/train-all.sh` caps MLX's buffer cache at 2 GB, and the script stops a training that holds more than 13 GB of memory (see amendment 4 of `docs/s6-ear-prereg.md`). None of this changes what is computed.

6. **13/11/2026, before any training of s6-cast: no score of s6-cast exists. A third brake mode.** At the owner's request, `scripts/auto-brake.sh` writes `1.0 55:5` between 23:00 and 08:00 when the Mac has had no input for 30 minutes and nothing else works on it, and the memory cap needs four readings in a row above 13 GB, as in amendment 7 of `docs/s6-gen-prereg.md`. Budgets still count working time only. Nothing else changes.

7. **13/11/2026, before any training of s6-cast: no score of s6-cast exists. The cache of compiled graphs.** `sky-train` empties MLX's cache of compiled graphs every 256 new input shapes (`SKY_COMPILE_SHAPES`), as amendment 8 of `docs/s6-gen-prereg.md` describes: MLX keeps one graph per shape for the life of the process, which made the memory of a long run grow without bound. A shape met again is compiled again into the same graph: no value changes. Nothing else changes.

8. **13/11/2026, before any training of s6-cast. The night mode at 95%.** At the owner's request (13/11/2026), the night mode of `scripts/auto-brake.sh` is `0.95 55:5` instead of `1.0 55:5`, from the night of 13/11 on. Budgets still count working time only: nothing else changes.

9. **15/11/2026, after the gate was read: it failed. The file of the failed model is published, as a result.** On 15/11/2026 the owner of the project decided that the files of the models that failed their gate are published too, so that every result can be checked and run by anyone. Nothing of the reading changes: the gate failed, the model is recommended for no use, and wherever its file is offered (model card, site, manifest) the first lines say that it failed its pre-registered gate and give the numbers. The sentences of this pre-registration that tied publication to the gate said what a pass would mean; for the file they are replaced by this decision, for the verdict they stand. The published file is `s6-cast-pico` as evaluated (142,912 bytes).
