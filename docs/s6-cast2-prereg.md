# s6-cast v2 (s6-cast2-pico): pre-registration

**Status: frozen on 15/11/2026.** No line of v2's code is written, no v2 data is downloaded, no v2 model exists. This file fixes the data, the model, the run, the choice of checkpoint, the evaluation and the gate, and the results are read through it. Two things are fixed later by rules written here, each before training starts:

- the list of new real-series sources (§3.2);
- the fate of gate G3 (§8).

**Why this experiment.** s6-cast-pico failed its pre-registered gate on GIFT-Eval: nGMASE 0.9471 and nWQL 0.6646, against bars of 0.838 and 0.587 (`docs/s6-cast-prereg.md`, `docs/big-run-report.md` §5). Our port of TinyCast, a 146K-parameter published model, scores 0.7856 / 0.5570 on the same harness. The owner wants a forecaster that works. v2 keeps the s6-cast graph and changes what the measured loss points at. The design study, `docs/s6-cast2-study.md`, is committed with this file and holds every number quoted here.

**Changes to this file:**

- every amendment is committed, dated and justified in §13;
- an amendment comes before the run or the measurement it concerns;
- an amendment never moves a threshold after a v2 test score has been read;
- measurements on training and validation series are allowed before the freeze and are written down with their date: generator samples, detector firing rates, throughput, the leakage check of §3.3. None of them touches a GIFT-Eval or Chronos-ZS file.

## 1. The question

Does the s6 state encoder, at about 380K ternary parameters, become a usable zero-shot forecaster when it is given:

- a computed-periodicity front end;
- training data that covers GIFT-Eval's frequency bands;
- 1.5× v1's training time?

**Usable** means all of these:

- it beats the training-free FLAIR on both GIFT-Eval metrics;
- it beats the seasonal naive at every horizon term;
- it holds up on a benchmark its design never read;
- it is no slower than the TinyCast port it would replace.

## 2. What changes from v1, and why

All v1 numbers are from the design study, computed from `data/gift-eval-out/*/all_results.csv` and the per-window quantile files. Each change answers a measured loss.

| # | change | v1 evidence | external evidence |
|---|---|---|---|
| C1 | **Computed periodicity.** TinyCast's zero-parameter period detector on every context, feeding phase channels into the patches (§4.2). | 82% of v1's log-MASE gap to the TinyCast port (95% of the WQL gap) lies in the 70 configurations where the detector finds a period in ≥ 90% of windows. Where it fires in < 50%, v1 equals TinyCast (MASE 0.839 against 0.840). Windows with a detected period lose more than those without (1.235 against 1.157). | TinyCast's architecture family: detector off +0.131 nGMASE; phase binning −0.098 (arXiv 2608.15767, Appendix). |
| C2 | **A value-space seasonal draft** of each 32-step query patch, added to the decoder query (§4.3). | The gap is present from the first tenth of the horizon (ratio 1.31–1.38 to TinyCast at medium and long terms). `m4_hourly/H/short`, a clean 24-step cycle, is worse than repeating the last day (MASE ratio 1.598). Periods not aligned to the 32-point patch grid lose more than aligned ones (1.233–1.240 against 1.161). | TinyCast's future-conv correction over the same draft: −0.015; with the synthetic families, −0.026. |
| C3 | **Real series at the frequency bands GIFT-Eval tests**: sub-hourly, second-level, CloudOps. Real share raised to 0.6, sampling weighted by band (§3). | Sub-hourly and second-level configurations carry 57% of the MASE gap. v1's real series include nothing faster than 30 minutes. | TinyCast trains on about 93% real series, balanced over bands; 50M → 100M samples −0.037. |
| C4 | **Generator periods** 144, 288, 360, 1,008, 1,440 and 2,016 added to `synth::PERIODS`, with pool series long enough for them. | The daily and weekly cycles of 5- and 10-minute data are absent from the generators. Weak: at window level, periods the generators produce do no better than others (1.204 against 1.215). Kept for coverage, at no cost. | — |
| C5 | **Sign flip and time reversal at probability 0.5** (other augmentations unchanged). | The flip gives v1 −0.037 nGMASE; it gives TinyCast −0.008. v1 is not sign-symmetric. | TinyCast's augmentation probability. |
| C6 | **720 working minutes** instead of 480. | The f32 validation pinball was still falling at the end of the stable phase (−2.6% over the last 90k steps). v1 saw 64M windows. | TinyCast's curve: 100M samples is where it flattens. |

**Not changed, and why.**

- **The output length and the rollout.** v1's ratio to TinyCast is flat along the horizon, so there is no rollout loss to fix.
- **The fallback windows.** They have zero effect: v1 and TinyCast both forecast 0 on the 2,017 windows with an empty context.
- **The quantile head.** Short-term coverage is right (80% interval 0.79). Both models under-cover at medium and long terms by the same amount.
- **Ternary weights.** They cost 2.4% on validation; TinyCast's INT8 costs 2.1%.
- **The size class.** v1 has 2.5× TinyCast's parameters already.

## 3. Data and leakage

### 3.1 Kept from v1

- The synthetic pool (`sky_infer::cast::synth`: GP, pulse, trend-season-irregular, SARIMA) with C4 and C5.
- The real series of `data/series/` (`DATA_LICENSES.md`, s6-cast section).
- One real series in ten (by id) held out for validation.

### 3.2 New real series (C3)

- **Target bands.** Second-level and 5-, 10- and 15-minute series: CloudOps traces, sensor and energy meters, traffic counters.
- **Selection rule.** Each candidate source is kept only if all of these hold:
  1. its licence is CC0, CC-BY (any version), public domain, Apache-2.0, MIT or BSD, as stated by its publisher, and is recorded in `DATA_LICENSES.md`;
  2. its name is in neither GIFT-Eval's 97 configurations nor the Chronos zero-shot benchmark, nor in any dataset either benchmark draws from (the v1 exclusion list of `DATA_LICENSES.md`, extended by TinyCast's overlap list);
  3. it passes the content check of §3.3.
- **Freeze.** The list, with per-source series counts and points, is frozen by an amendment before training starts.
- **If the list is empty**, C3 is dropped by that amendment and the run goes on with C1, C2, C4, C5 and C6. The report then says that the data change was not made.
- **GIFT-Eval-Pretrain.** Its Hugging Face card declares `license: apache-2.0`, adds "for research purposes only", and aggregates datasets that keep their own terms; it is 975 GB in all. It is **not** used unless the owner decides otherwise before the freeze (open choice O3). If used, rules 1–3 apply to every constituent directory, and TinyCast's 2,020-name filter and 66 gridded-weather names are applied on top.

### 3.3 Leakage rules

- **No series of GIFT-Eval or of Chronos-ZS enters training, validation, pilots or teacher cuts.** This includes the training and validation parts of the test datasets.
- **Content check.** Every new source is hashed by runs of 32 consecutive raw values and compared with the same hashes over every GIFT-Eval series (`data/gift-eval-src`) and every Chronos-ZS series. A source with any matching run of non-constant values is dropped whole. The check reads the benchmark series only to hash them. It is run once, and its counts are recorded before training.
- **Distillation.** The TinyCast teacher (§5) is asked only on training series. TinyCast excludes the GIFT-Eval and Chronos-ZS sets from its own training, as recorded in `docs/s6-cast-prereg.md` §3.

### 3.4 Mix

- **Real share of windows: 0.6** (v1: 0.3).
- **Band weights.** Real windows are drawn with weights that match the share of GIFT-Eval's 97 configurations per band: hourly 31, sub-hourly 30, daily 15, weekly 8, monthly and coarser 7, second-level 6. This is TinyCast's sampling rule. A band with no real series gives its weight to synthetic windows downsampled to that band's typical periods.
- **Context lengths** as v1: full context half the time, whole patches 30%, 8–256 points 20%.

## 4. The model (`configs/s6-cast2-pico.json`)

### 4.1 The s6-cast graph, as trained in v1

- d 96, feed-forward 192, 2 heads;
- 2 blocks × 2 loops, exits after 1, 2, 3 and 4 block applications;
- patches of 32 points, 64 patches of context (2,048 points), 2 query patches (64 steps per pass), 2 decoder blocks, 9 deciles;
- no Parallax and no XSA (the trained v1 config, `runs/recipe/configs/s6-cast-pico.json`).

### 4.2 Periodicity (C1)

- **Detector.** `tinycast::significant_periods` (K = 4, α = 0.05), run on TinyCast's prepared window of the context:
  - the last 2,048 points, left-padded with the first value;
  - gaps linearly interpolated;
  - min-max normalised.

  This is the port's exact detector input, so its behaviour on GIFT-Eval is the measured one: the firing rates in the study. It has no parameters. It runs in the data producer at training and in `prep` at inference, identically. It is recomputed at each rollout pass.
- **Phase channels.** For patch j, whose last point is at window position t_j: sin(2π·t_j/p_k) and cos(2π·t_j/p_k) for k = 1..4, and 0 for an empty slot. These 8 values are appended to the patch's input, so `patch_proj` becomes `[d, 2·patch + 8]`.

### 4.3 Seasonal draft (C2)

For each future step i of the pass, in the window's normalised units:

- **(a) the lag copy:** the value one or more dominant periods p₁ before it. It falls back to earlier cycles over missing values, as `prep::seasonal_naive`. It is the last observed value when no period passes.
- **(b) the phase-binned mean:** TinyCast's `seasonal_naive` over the observed points (16 bins of p₁).

Query patch q gets `[(a) × 32, (b) × 32, the 8 phase channels at its first step]`, 72 values. They go through `draft_proj` (Linear 72 → d, with bias), and the result is added to `cast_query[q]` before the decoder blocks. The head and the monotone quantiles are unchanged.

### 4.4 Size

- 369,792 + 768 (`patch_proj`) + 7,008 (`draft_proj`) = **377,568 parameters**.
- `draft_proj` is ternary under QAT, like the decoder matrices.
- Expected ternary GGUF about 149 KB. This is an estimate; the report gives the real size.
- New GGUF keys: `skycmd-cast.periods` (4) and `skycmd-cast.draft` (true). A v1 file stays readable: no key means no front end.

### 4.5 Code (before any training)

- **sky-infer:** `cast/prep.rs` (detector input, phase channels, draft), `cast/model.rs` (`patch_proj` width, `draft_proj` in `decode`), `cast/mod.rs` (config keys), `cast/synth.rs` (C4, C5).
- **sky-train:** `cast/config.rs`, `cast/data.rs` (front end in the producer, band weights, two batch tensors), `cast/model.rs` (graph), `cast/gguf.rs`, `cast/golden.rs`.
- **Golden.** `tests/golden_cast.rs` and its golden files are regenerated with the femto config extended by the front end, at the same tolerances.
- **sky-data:** `fetch-series` for the sources of §3.2.
- **Scripts.** `scripts/train-cast.sh` gains CONFIG/RUN for v2, the TinyCast run without flip, and the latency step. A WASM forecast benchmark goes next to `crates/sky-infer/bench/bench-node.mjs`, through `sky-web`'s forecaster.
- **Chronos-ZS harness** (§7.3): a new script. About 1,000 lines in all.

## 5. The run

- **Pilot** (`scripts/train-cast.sh pilot`):
  - the rate is fixed at 3e-3: v1's pilot kept it, and TinyCast's optimisation family found it best;
  - only the distillation pilot runs, 3,000 steps with and without the TinyCast teacher, with v1's rule: distil unless the validation pinball is worse by more than 1%;
  - about 12 working minutes.
- **Training:**
  - **720 working minutes**, batch 256, AdamW at 3e-3 with v1's recipe flags (`runs/s6-cast-pico/args.json`), WSD;
  - ternary QAT over the last 20%, EMA 0.999, seed 1, once;
  - if distilled: teacher weight 0.5 → 0 on 50% of windows, as v1;
  - brake modes and memory cap as amendments 4–8 of `docs/s6-cast-prereg.md`; minutes count working time only; the report gives steps done and modes.
- **Restarts.** Only a technical failure justifies a relaunch from the last checkpoint, and it is reported.
- **Total compute.** About 760 working minutes (pilot, training, export, calibration, evaluation), about 20 hours of wall clock in the brake modes.

## 6. Checkpoint and export

- The EMA if its final ternary validation pinball is below that of `last`, else `last` (v1's rule). No intermediate checkpoint.
- Exit calibration as v1 (α 0.05, δ 0.02, 2,000 unseen synthetic windows).
- Validation adds the held-out tenth of the new sources, and is reported by band. It informs nothing after the freeze.

## 7. Evaluation

### 7.1 GIFT-Eval (gated)

- **The harness.** The same as v1: `scripts/gift-eval.py`, gluonts `evaluate_model`, 97 configurations, 9 deciles, ratios to `eval/gift-eval/seasonal_naive.csv`, geometric means. No per-dataset setting.
- **The gated model is without flip,** as v1.
- **Before v2 is read,** two checks:
  - the seasonal naive's nGMASE is 1.00 ± 0.02, and the TinyCast port's files give back 0.7856 / 0.5570;
  - the TinyCast port **without flip** is run once and reported. It is the like-for-like reference; the paper's symmetrisation delta suggests about 0.794 / 0.563.

### 7.2 Latency (reported, with a plain sentence)

- **Setup.** Measured before v2 is read, in one session, on the M5:
  - v1, v2 and the TinyCast port;
  - median and p95 over 1,000 forecasts;
  - one core, 2,048-point history, 64-step horizon, full depth;
  - natively (`sky-cast --bench`, the default `TernaryCompute` mode) and in WASM (Node, `sky-web`'s forecaster, SIMD128).
- **Also reported:** v2 with its calibrated exits, and a 720-step horizon.

### 7.3 Chronos-ZS (gate G3, or reported, by §8)

- **Benchmark.** The 27 tasks of TinyCast's Chronos-ZS table, with relative MASE and WQL against the seasonal naive, geometric means.
- **Harness check.** The harness must reproduce the TinyCast port within ±0.02 of the published 0.880 / 0.722 before v2 is read. The port runs with flip, as published.

## 8. Gate (ternary GGUF, without flip, full harness)

v2 passes if all of these hold:

1. **G1.** On GIFT-Eval, **nGMASE ≤ 0.838 and nWQL ≤ 0.587.** These are v1's bars, those of FLAIR (training-free), unchanged so that v2 is not judged by a bar moved after reading v1. They sit 6.7% above the TinyCast port, so G1 is also the "approach TinyCast" test.
2. **G2.** On GIFT-Eval, **the geometric-mean MASE ratio is < 1.00 at each term** (short, medium and long). v1 was at 1.046 medium and 1.105 long. A forecaster worse than repeating the last season at long horizons cannot be offered for them. The TinyCast port is at 0.756 / 0.806 / 0.848.
3. **G3.** On Chronos-ZS, **relative MASE and relative WQL each ≤ 1.083 × the TinyCast port's on the same harness.** 1.083 = 0.838 / 0.774, the margin G1 allows from the published TinyCast, carried over to a benchmark the design never read. It answers the disclosure of §10.
   - **If the harness is not validated before training starts** (§7.3), an amendment drops G3 before training. Chronos-ZS is then reported when available, and the verdict states that G1 and G2 were read on the benchmark that shaped the design.
**Latency and size are reported, not gated** (the draft of this file gated them; the owner asked on 15/11/2026 for a forecaster of our own that passes its bars, and a latency bar against a 146K-parameter convolutional model would decide the gate on architecture rather than on forecasts). The report gives the native and WASM median latency and the file size of v2 next to the TinyCast port's, measured in the same session (§7.2), and says in one sentence which of the two a user should take: if v2 is slower and larger than the port and does not forecast better, the port stays the recommendation, whatever the gate says.

Not gated: beating the TinyCast port. It was v1's goal; it is reported as the gap to the port with and without flip, and to the published 0.774 / 0.545.

## 9. Reported, not gated

- **GIFT-Eval detail:**
  - nMSIS;
  - v2 with flip;
  - the per-term, per-band and per-domain tables of the study, for v1, v2 and the port;
  - the configurations where v2 is above the seasonal naive;
  - the share of the gap closed.
- **Coverage.** Per quantile level and for the 80% interval, by term, for v1, v2 and the port, computed per window from the quantile files and the labels as in the study.
- **Front-end diagnostics:**
  - the window-level splits (period detected or not, aligned to the patch grid or not, context length);
  - the horizon profile by tenths.
- **Latency and size:**
  - latency natively and in WASM (median and p95), with calibrated exits and at full depth;
  - the mean exits used on GIFT-Eval;
  - the GGUF size and the parameter count.
- **Training:**
  - validation pinball by band;
  - steps done;
  - windows seen (real, synthetic, distilled);
  - the data sources and their counts.

## 10. Disclosure

v2 was designed after reading v1's GIFT-Eval results on the same 97 configurations, window by window: where it lost, by band, term, domain, seasonality, context length and horizon. GIFT-Eval is therefore partly in-sample for v2's design choices, though not for its weights: no GIFT-Eval series is trained on. TinyCast's authors make the same disclosure about their own design and treat Chronos-ZS as the cleaner test. G3 exists for this reason. Every report of v2 states this paragraph.

## 11. What a failure would mean

- **G1 fails.** The s6 patch encoder at about 380K ternary parameters is not competitive in this niche, even with a periodicity front end, band-matched data and 1.5× the training. The s6-cast line stops. The forecaster offered by the project becomes the TinyCast port: Apache-2.0, credited, already ported, optionally post-training INT8 (+2.1% in the paper). No v3 is planned on GIFT-Eval.
- **G1 passes, G2 fails.** v2 may be offered for short horizons only, by a separate decision. It does not pass.
- **G1 and G2 pass, G3 fails.** The design fits GIFT-Eval more than forecasting in general. It does not pass.
- **v2 passes but is slower and larger than the port and forecasts no better.** It is published as passed, and the report says that the port stays the recommendation.
- **In every case,** following the owner's decision of 15/11/2026 (amendment 9 of `docs/s6-cast-prereg.md`), the file is published as evaluated. Its first lines give the verdict and the numbers.

## 12. Forbidden

- Training, validating, piloting or distilling on any GIFT-Eval or Chronos-ZS series, or on any part of their datasets.
- Any per-dataset or per-frequency setting in the evaluation.
- A second seed, a second run, or a checkpoint other than §6's, except after a technical failure.
- Reading a v2 score on GIFT-Eval or Chronos-ZS before §7.1's checks, §7.2's latencies and §7.3's harness check are on file.
- Changing a bar, a term of the gate or the gated mode (without flip) after a v2 score is read.

## 13. Amendments

None yet.

## 14. Choices (the first value of each line is the one taken at the freeze; the alternative is recorded)

| | choice | recommended | alternatives | why the recommendation |
|---|---|---|---|---|
| O1 | training time | **720 working minutes** | 480 (isolates the design change from the time); 960 | about 96M windows at v1's throughput, near TinyCast's plateau; 480 would confound a failure with under-training less than it saves |
| O2 | real share | **0.6** | 0.3 (v1); 0.9 (TinyCast-like) | TinyCast is about 93% real, but our licence-clean band coverage will be thinner; 0.6 keeps synthetic windows for bands without real series |
| O3 | GIFT-Eval-Pretrain | **not used** | a filtered subset (§3.2) | its terms say "research purposes only" and its constituents' licences are not checked; it would make v2 depend on the corpus that also shaped TinyCast |
| O4 | `draft_proj` weights | **ternary under QAT** | f16 (about +13 KB) | keeps the model ternary like the decoder; the cost is unknown either way |
| O5 | draft form | **added to the query (embedding)** | median predicted as a residual over the lag copy | the decoder learns how far to trust the draft; a residual forces it where no period exists (11 configurations where TinyCast fires < 50% and v1 already ties it) |
| O6 | distillation | **v1's pilot rule** (distil unless the pinball is worse by > 1%) | no teacher; always distil | v1's pilot found +0.3%, within noise; the rule keeps the decision mechanical |
| O7 | rate pilot | **none, 3e-3 fixed** | v1's three-rate pilot (+18 minutes) | v1's pilot and TinyCast's optimisation family both chose 3e-3 |
| O8 | G3 | **gate, if the Chronos-ZS harness is validated before training** | reported only | the only test of the design on data it never read |
| O10 | C4 (generator periods) | **in** | out | free, harmless, covers 30 of 97 configurations' natural cycles; weak evidence, said so |
| O11 | rollout training (scheduled sampling over 2–4 passes) | **out** | in | no measured rollout loss in v1; TinyCast's 8-against-4 chunks is null; it would cost 2–4× per window |
