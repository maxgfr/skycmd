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

**Amendment 1 (15/11/2026, after the code was written and before any v2 training, download or evaluation; no v2 model exists). The sources of C3, one contradiction removed, and the readings the code had to choose.**

- **The new real series (§3.2).** By the rule of §3.2, applied on metadata, three sources pass the licence rules: USCRN sub-hourly (NOAA, public domain; the 2025 files, 2.26 GB, at most 1,431 series), the Azure VM traces of 2017 (CC-BY-4.0, 84 GB) and the Borg cluster data of 2011 (CC-BY-4.0, 42 GB). Only USCRN is used: the two others are tens of gigabytes of per-machine and per-task records for which §3.2 gives no rule to build series, and writing that rule now would be a new design choice. Alibaba's traces (research use only), LargeST (non-commercial), PeMS (no licence stated), the LOTSA and GIFT-Eval-Pretrain copies and the sub-hourly configurations that are in the benchmarks are rejected. No source at the second level passed: that band is filled by synthetic series generated at that band's periods (`second=360`), as are the weekly (52) and monthly (12, 4) bands when they have no real series. From USCRN: the last 8,192 points of each station and measured column, nine columns, constant columns dropped. The leakage check of §3.3 (runs of 32 float32 values hashed against every GIFT-Eval and Chronos-ZS series; runs with gaps or constant values skipped) runs on the fetched data before training and its result is recorded.
- **The game series keep their place.** As coded, a series with no frequency band was never drawn once the bands were on, which contradicts §3.1 (the series of v1 are kept). The game series are drawn with the share they had in v1, outside the bands. This is fixed in the code before any training.
- **G3 stands only if its harness is validated first.** The Chronos-ZS harness is written and has never run. `scripts/train-cast2.sh chronos-check` (the TinyCast port and the seasonal naive through export, scoring and the reference check) has to pass before training starts; if it does not, G3 is dropped by a dated line here before training and Chronos-ZS is reported, as §8 says. G3 compares v2 without flip with the port without flip (the draft compared with the port with flip; the gate is read without flip everywhere).
- **Readings fixed as coded** (none chosen on a benchmark file): the v1 generator's periods are kept under their name and v2's added periods are a separate set selected by a flag, so that v1's seeds give v1's series; a pool series is long enough at 6,048 points; the detector's window is the model's context length (2,048); the draft is in the model's own window units and the detector keeps the port's normalisation; phase channels are ordered sine then cosine per period, positions count the left padding, the query phase is taken at the query's first step; `draft_proj` is initialised and trained like the other linear layers; a draft requires periods; London's 30-minute series count as sub-hourly; within a band a series is drawn uniformly, and distilled windows draw their real series by band too; the synthetic validation windows keep v1's periods; the fetch is a script (`scripts/cast2-data.py`), not a `sky-data` subcommand; v1's golden files are kept and v2 gets its own; the pilot is 3,000 steps; the latency is measured on the `tsi` series (seed 7), and in WASM from JavaScript, boundary included; the smoke mode scores its throwaway model on two GIFT-Eval configurations to test the chain, and those numbers are not looked at.
- **Noted:** TinyCast's list of training series that overlap the benchmarks is not published in its repository, so the overlap of §3.3 is checked by our own hash check only.
- **What does not change:** the graph, the budget of 720 working minutes, G1 and G2 and their thresholds, §12.

**Amendment 2 (15/11/2026, after the data was fetched and the two checks ran, before any v2 training; no v2 model exists, no v2 score exists). The leakage rule counts informative runs; G3 is dropped.**

- **What the leakage check found.** USCRN 2025: 159 station files, 1,011 series kept, 8,282,112 points. Of 6,300,742 runs of 32 values, 76,364 also occur in a benchmark series, in 568 of the 1,011 series. They are not copies: 65,171 of them have at least 30 zeros out of 32 and 59,657 hold two distinct values (precipitation 46,862, solar radiation at night 21,223, calm wind 5,129), and the benchmark datasets they match (m5, monash_weather, temperature_rain, solar/10T) cannot contain USCRN measurements of 2025. The check itself works: a source that copies 100 values of a GIFT-Eval series is caught on every run, a noise series on none.
- **The rule of §3.3, completed before training.** A run of 32 values is compared only if it is informative: at least 8 distinct values, and its most frequent value on at most half of its points. A run of zeros says nothing about where a series comes from. A series with a matching informative run is dropped; the source is dropped whole if more than 5% of its series are. The check is run again with this rule and its counts are recorded before training; the run of amendment 1's rule stays on record above.
- **G3 is dropped.** `chronos-check` scored all 27 Chronos-ZS tasks. Our seasonal naive reproduces the reference to nine decimals, so data, windows and metrics agree; but the TinyCast port is at 0.8576 (MASE) and 0.6822 (WQL) where the published values are 0.880 and 0.722, outside the plus or minus 0.02 that validates the harness, and TinyCast's driver for this benchmark is not public, so the difference cannot be traced. The harness is not changed to fit a number. As §8 wrote: G3 is dropped before training, Chronos-ZS is reported (v2 and the port through the same harness), and the verdict says that G1 and G2 were read on the benchmark that shaped the design.
- **The game series** take their v1 share inside the real share of 0.6, which stays 0.6.

**Amendment 3 (16/11/2026, before any v2 training, the smoke included; no v2 model exists, no v2 score exists). The leakage check under amendment 2's rule, and where the run happens.**

- **The leakage check, run again with the rule of amendment 2** (`skycmd-data/s6-cast2/leak.json`, written 15/11/2026 at 11:55): the benchmarks hold 141,203,971 informative runs of 32 values (GIFT-Eval, 147,909 series) and 67,632,869 (Chronos-ZS, 190,674 series). USCRN 2025: 1,011 series, 4,275,428 informative runs, **0 matching run, 0 series dropped**: the source is kept whole (the kept file's sha256 is in `leak.json`). No benchmark dataset has a match.
- **Where the run happens.** From the worktree of the session, on the code of `origin/main` at the commit of this amendment, with the v1 series, the synthetic series and GIFT-Eval's windows read through a link from the main checkout, and the new series outside the repository (`skycmd-data/s6-cast2`). The trainer is built at that commit: since the code of v2 was written it gained the fast-weight memory of s6-sleep v3 and its `--sleep-chunk` option, the router's code and the end-of-turn student, none of which the forecaster's training path uses.
- **The brake** is the automatic one (`scripts/auto-brake.sh`): half duty with 45 minutes of work and 15 of rest whenever the machine is in use by day, 0.95 at night when it is left alone, and its memory guard (the training is stopped above 13 GB held; a stop is a technical failure, restarted from the last checkpoint and reported). Budgets are working minutes, so the brake changes the duration and not the steps; the report gives the steps and the modes. CPU work of other models (write-ups, evaluations on dumps) may run at the same time; the latencies of §7.2 are measured with no training running.
- **Order on the GPU:** after the runs of s6-ear v2 and of s6-sleep v3, one budgeted run at a time; the smoke first, under a throwaway prefix, its numbers not looked at.
- **What does not change:** the data and its mix, the model, the pilot and its rule, the 720 working minutes (seed 1, once), the checkpoint rule, the evaluation and its checks, G1 and G2 (G3 dropped by amendment 2), §12.

**Result (18/11/2026, after the gate was read once). The gate failed: G1 does not hold. G2 holds. G3 was dropped before training (amendment 2) and Chronos-ZS is reported. v2 is published as a failed result.**

- **Verdict** (`runs/s6-cast2-pico/GATE.md`, `gate.json`): **FAIL**. G1 and G2 were read on the benchmark that shaped the design (§10), and Chronos-ZS is reported, not gated.
- **Disclosure (§10).** v2 was designed after reading v1's GIFT-Eval results on the same 97 configurations, window by window: where it lost, by band, term, domain, seasonality, context length and horizon. GIFT-Eval is therefore partly in-sample for v2's design choices, though not for its weights: no GIFT-Eval series is trained on. TinyCast's authors make the same disclosure about their own design and treat Chronos-ZS as the cleaner test. G3 exists for this reason. Every report of v2 states this paragraph.
- **What was run, in which order** (`scripts/train-cast2.sh`; logs `runs/all/cast2-train.log` and `runs/all/cast2-gate.log`). In one chain, started on 17/11 at 15:36 (`started.json`): the pilot, the training, the export and the calibration; then, after the training had ended, the latencies of §7.2 (`latency/latency.json`, written 18/11 at 11:25), the checks of §7.1 (`checks.json`), the TinyCast port without flip on GIFT-Eval, v2 on GIFT-Eval without and with flip, v2 on Chronos-ZS, the diagnostics and the gate (`GATE.md`, 18/11 at 13:14). The chain stopped twice on a full disk; both incidents are below.
- **The checks before v2 was read** (§7.1, `checks.json`): the seasonal naive gives nGMASE 0.9999 over 97 configurations (rule 1.00 ± 0.02: ok); the TinyCast port's files give back 0.7856 / 0.5570 (ok). The port without flip, run once: nGMASE 0.7918, nWQL 0.5622, nMSIS 0.5865 (`data/gift-eval-out/tinycast-noflip/summary.json`).
- **The gate** (ternary GGUF `s6-cast2-pico.gguf`, the calibrated file, without flip, the full harness; GIFT-Eval, 97 configurations, 371,330 windows, of which 2,017 in 12 configurations have no observation in the last 2,048 points and are forecast by the seasonal naive, as for v1):

  | term | measured | bar | verdict |
  |---|---|---|---|
  | G1 | nGMASE **0.8847**, nWQL **0.6190** | nGMASE ≤ 0.838 and nWQL ≤ 0.587 | **does not hold** |
  | G2 | MASE ratio to the seasonal naive: short **0.833**, medium **0.928**, long **0.988** | < 1.00 at each term | holds |
  | G3 | dropped before training (amendment 2) | | reported below |

- **GIFT-Eval, every model on the same harness** (`data/gift-eval-out/*/summary.json`; lower is better):

  | model | nGMASE | nWQL | nMSIS |
  |---|---:|---:|---:|
  | s6-cast-pico (v1), without flip | 0.9471 | 0.6646 | 0.7190 |
  | v1, with flip | 0.9099 | 0.6417 | 0.6860 |
  | **s6-cast2-pico (v2), without flip** (the gated model) | **0.8847** | **0.6190** | 0.6848 |
  | v2, with flip | 0.8692 | 0.6071 | 0.6735 |
  | TinyCast port, with flip | 0.7856 | 0.5570 | 0.5781 |
  | TinyCast port, without flip | 0.7918 | 0.5622 | 0.5865 |
  | seasonal naive | 0.9999 | 0.8876 | 2.6778 |
  | bar of G1 (FLAIR, training-free) | 0.838 | 0.587 | |

  The gap of v2 (without flip) to the port, not gated: +0.0991 nGMASE and +0.0620 nWQL to the port with flip, +0.0929 and +0.0568 to the port without flip, +0.1107 and +0.0740 to the published TinyCast (0.774 / 0.545). Of v1's gap to the port (sum of log ratios over the 97 configurations), v2 closes 0.365 on MASE and 0.402 on WQL.
- **By term** (MASE / WQL ratio to the seasonal naive, geometric means; `GATE.md`):

  | term | configs | v1 | v2 | TinyCast port |
  |---|---:|---|---|---|
  | short | 55 | 0.860 / 0.638 | 0.833 / 0.610 | 0.756 / 0.571 |
  | medium | 21 | 1.046 / 0.696 | 0.928 / 0.624 | 0.806 / 0.534 |
  | long | 21 | 1.105 / 0.706 | 0.988 / 0.637 | 0.848 / 0.546 |

- **By band** (same ratios):

  | band | configs | v1 | v2 | TinyCast port |
  |---|---:|---|---|---|
  | second | 6 | 1.298 / 0.896 | 0.926 / 0.692 | 0.909 / 0.702 |
  | sub-hourly | 30 | 1.046 / 0.735 | 0.936 / 0.653 | 0.794 / 0.571 |
  | hourly | 31 | 0.914 / 0.606 | 0.870 / 0.569 | 0.754 / 0.480 |
  | daily | 15 | 0.782 / 0.494 | 0.770 / 0.490 | 0.711 / 0.465 |
  | weekly | 8 | 0.908 / 0.729 | 0.937 / 0.737 | 0.858 / 0.732 |
  | monthly and coarser | 7 | 0.878 / 0.850 | 0.907 / 0.878 | 0.890 / 0.853 |

- **By domain** (same ratios):

  | domain | configs | v1 | v2 | TinyCast port |
  |---|---:|---|---|---|
  | Econ/Fin | 6 | 1.018 / 0.901 | 0.984 / 0.876 | 0.894 / 0.841 |
  | Energy | 32 | 1.080 / 0.800 | 1.003 / 0.746 | 0.882 / 0.661 |
  | Healthcare | 5 | 0.760 / 0.614 | 0.825 / 0.678 | 0.735 / 0.639 |
  | Nature | 15 | 0.862 / 0.467 | 0.825 / 0.414 | 0.757 / 0.358 |
  | Sales | 4 | 0.753 / 0.457 | 0.749 / 0.455 | 0.734 / 0.445 |
  | Transport | 15 | 0.833 / 0.644 | 0.772 / 0.603 | 0.669 / 0.510 |
  | Web/CloudOps | 20 | 0.983 / 0.662 | 0.861 / 0.593 | 0.750 / 0.562 |

  v2 is better than v1 on both metrics in every term, in four of the six bands and in six of the seven domains; it is worse than v1 in Healthcare (0.825 / 0.678 against 0.760 / 0.614) and in the weekly and monthly bands. Energy stays at the seasonal naive's MASE (1.003). In no term, band or domain is v2's MASE below the port's; its WQL is below the port's in one place only, the second band (0.692 against 0.702).
- **Configurations where v2's MASE is above the seasonal naive's: 31 of 97** (the list is in `GATE.md`). The worst: `ett2/W/short` 1.626, `us_births/M/short` 1.572, `m4_hourly/H/short` 1.513, `bitbrains_fast_storage/5T/long` 1.422, `bitbrains_rnd/5T/long` 1.376, `bitbrains_fast_storage/5T/medium` 1.365, `solar/10T/long` 1.335, `solar/H/medium` 1.312. `m4_hourly/H/short`, the clean 24-step cycle that §2 (C2) aimed at, stays above the seasonal naive.
- **Not reported: the coverage and the front-end diagnostics of §9** (coverage per quantile level and for the 80% interval by term; the window-level splits; the horizon profile by tenths). `runs/s6-cast2-pico/diagnostics.json` holds the three tables with no row, and so does `GATE.md`. The script (`scripts/s6-cast2-diagnostics.py`) reads the per-window quantile dumps (`*.q.bin`) of each model folder and skips a configuration that has none; the dumps of the GIFT-Eval folders are no longer on the disk (the third incident below), and the files do not record whether any was there when the gate step ran. The promised readings are therefore missing, not measured as empty. They can be rebuilt without training: `sky-cast-eval run` on the published files writes the dumps again, then `scripts/s6-cast2-diagnostics.py`; that has not been done. (Done later on 18/11/2026: see the last note of this block, "The per-window readings, computed after the forecast dumps were rebuilt".)
- **Chronos-ZS, reported, not gated** (27 tasks, relative to fev's seasonal naive, geometric means; `GATE.md`, `skycmd-data/s6-cast2/chronos-zs-out/*/summary.json`). The harness check of §7.3 did not validate (amendment 2): the port with flip is at 0.8576 / 0.6822 where the published values are 0.880 / 0.722.

  | model | relative MASE | relative WQL | tasks |
  |---|---:|---:|---:|
  | **v2, without flip** | **0.9014** | **0.7143** | 27 |
  | TinyCast port, without flip | 0.8598 | 0.6928 | 27 |
  | TinyCast port, with flip (harness check) | 0.8576 | 0.6822 | 27 |
  | seasonal naive | 1.0000 | 1.0000 | 27 |

  On the benchmark its design never read, v2 beats the seasonal naive and stays behind the port on both metrics. G3's comparison (each ≤ 1.083 × the port without flip) is printed in `GATE.md` as "reported: True" and `gate.json` carries `"G3": true` with `"G3_state": "reported"`; it decides nothing, since G3 was dropped before training because the harness did not reproduce the published port.
- **Training** (`runs/s6-cast2-pico/args.json`, `runs/all/cast2-train.log`, `GATE.md`).
  - **The pilot** (`runs/s6-cast2-pilot/choice.json`, 3,000 steps each at 3e-3): validation pinball 0.045361 with the TinyCast teacher against 0.045963 without, so the rule of §5 kept distillation (teacher `crates/sky-infer/tests/golden/tinycast.gguf`, on 50% of the windows, weight 0.5 annealed to 0).
  - **The run:** 720.0 working minutes, **377,082 steps**, batch 256, AdamW at 3e-3, seed 1, once, no restart of the training; QAT over the last 20%, EMA 0.999. Windows seen: 21,295,763 synthetic, 25,092,584 real, 6,420,364 synthetic stand-ins of the bands with no real series, 43,724,281 distilled. Brake modes (`GATE.md`): 20 cooling pauses (`SKY_COOL`), the automatic brake at duty 0.95 (55 min of work, 5 of rest) three times and at duty 0.5 (45 and 15) three times.
  - **The data** (§3): v1's series of `data/series/` and the 1,011 USCRN sub-hourly series (8,282,112 points) of `skycmd-data/s6-cast2/series2-checked`, kept whole by the leakage check of amendment 3; 10,670 real series in training and 1,130 held out for validation. Real windows by band, as weighted: sub-hourly 0.309 (2,726 series), hourly 0.320 (548), daily 0.155 (5,990); the second (0.062, periods 360), weekly (0.082, period 52) and monthly (0.072, periods 12 and 4) bands had no real series and were filled by synthetic stand-ins; the game series kept v1's share, 0.0433 of all windows.
  - **The checkpoint** (§6, `exported.json`): **the EMA** (`last-ema`), its final ternary validation pinball 0.0435 against 0.0443 for `last`. Validation pinball of the exported weights by band (ternary EMA): synthetic 0.04884 (512 windows), sub-hourly 0.02609 (285), hourly 0.03496 (55), daily 0.01968 (621), real series without a band 0.17125 (141). The exit calibration (α 0.05, δ 0.02, 2,000 synthetic windows) stored `exit_tolerance` 0.012839192.
- **Size and latency** (§7.2, `runs/s6-cast2-pico/latency/`; measured 18/11 at 11:25 on an Apple M5 after v2's training had ended, one core, the `tsi` synthetic series of seed 7 with a 2,048-point history, 1,000 forecasts per line; natively `sky-cast --bench` in the default `TernaryCompute` mode, in WASM `sky-web`'s `Forecaster` under Node with SIMD128). The files: v2 148,512 bytes and 377,568 parameters (§4.4 expected about 149 KB), v1 142,912 bytes and 369,792 parameters, the TinyCast port 595,840 bytes and 146,505 parameters.

  | target | model | depth | horizon | median µs | p95 µs |
  |---|---|---|---:|---:|---:|
  | native | v1 | full | 64 | 799 | 870 |
  | native | **v2** | full | 64 | **838** | 906 |
  | native | v2 | calibrated | 64 | 842 | 904 |
  | native | TinyCast port | full | 64 | 21,531 | 23,189 |
  | native | v1 | full | 720 | 9,826 | 10,474 |
  | native | v2 | full | 720 | 10,367 | 11,158 |
  | native | TinyCast port | full | 720 | 164,760 | 171,634 |
  | WASM | v1 | full | 64 | 1,545 | 1,908 |
  | WASM | **v2** | full | 64 | **1,636** | 1,955 |
  | WASM | v2 | calibrated | 64 | 1,528 | 1,650 |
  | WASM | TinyCast port | full | 64 | 39,998 | 41,530 |
  | WASM | v1 | full | 720 | 17,922 | 19,230 |
  | WASM | v2 | full | 720 | 19,346 | 22,060 |
  | WASM | v2 | calibrated | 720 | 18,232 | 19,463 |
  | WASM | TinyCast port | full | 720 | 309,243 | 375,050 |

  In WASM on this series every forecast used its 4 exits (`mean_exits_used` 4), the calibrated file included; on GIFT-Eval the calibrated file used 3.334 exits of 4 on average over its 369,313 model-forecast windows (v1's count was not recorded). The sentence of §8, from `GATE.md`: *v2 at 838 µs and 148,512 bytes against the port's 21,531 µs and 595,840 bytes, nGMASE 0.8847 against 0.7856: the port forecasts better; take it unless size or speed decide.* These latencies are one series on one machine; they say nothing of accuracy.
- **Technical incidents** (lead's statement of 18/11/2026, checked against `runs/all/cast2-gate.log`; none changes a number).
  1. **First stop, during v2's GIFT-Eval forecasts.** The log shows the latencies, the two checks, the port without flip forecast and scored (nGMASE 0.7918, the only summary printed before the stop), then v2's forecasts of 81 configurations, from `bitbrains_fast_storage/5T/long` to `saugeen/W/short`, and `Error: Os { code: 28, kind: StorageFull, message: "No space left on device" }` on the next one. The lead's statement counts 82 configurations written; the log shows 81 finished and the error during the 82nd. **No v2 score was printed before that stop**: the log holds no per-configuration MASE or WQL line and no summary of v2 before it, and no v2 `summary.json` existed. The evaluation was started again from its beginning ("== restart after a full disk (18/11/2026): the evaluation, the Chronos check and the gate"): the two checks again, then v2 without flip on all 97 configurations and its scoring (nGMASE 0.8847), then with flip (0.8692). That GIFT-Eval scoring of v2 ran once, to its end.
  2. **Second stop, in the Chronos-ZS scoring of v2.** The datasets cache, removed by the lead to free space, was rebuilt by the scoring library and filled the disk: `OSError: [Errno 28] No space left on device` in `fev`'s `evaluation_summary`. Before it, v2's Chronos-ZS forecasts of the 27 tasks had been written and the per-task lines of eight tasks printed (`monash_traffic` to `monash_nn5_weekly`); no Chronos-ZS summary of v2 was printed. The Chronos step and the gate were started again ("== restart after a full disk (18/11/2026): the Chronos check and the gate (second restart; the evaluation is scored)", `runs/all/cast2-restart.sh`): v2's forecasts again, the 27 tasks scored (relative MASE 0.9014, WQL 0.7143), the diagnostics and the gate. Chronos-ZS is not gated.
  3. **The forecast dumps.** After scoring, the lead removed the raw forecast dumps (`*.q.bin`) of the seven scored folders of `data/gift-eval-out/` (v1, v1 with flip, v2, v2 with flip, the port with and without flip, the seasonal naive) to free disk space; their `all_results.csv` and `summary.json` are kept, and `sky-cast-eval run` rebuilds the dumps from the published files. This is why the per-window diagnostics of §9 are missing (above).
- **What the failure means, by §11.** **G1 fails.** In the words written before the run: "The s6 patch encoder at about 380K ternary parameters is not competitive in this niche, even with a periodicity front end, band-matched data and 1.5× the training. The s6-cast line stops. The forecaster offered by the project becomes the TinyCast port: Apache-2.0, credited, already ported, optionally post-training INT8 (+2.1% in the paper). No v3 is planned on GIFT-Eval." What improved is stated as plainly: v2 is better than v1 on both GIFT-Eval metrics, with and without flip; its MASE is under the seasonal naive's at every term, which v1's was not at the medium and long terms (G2 holds); it beats the seasonal naive on Chronos-ZS; and its file forecasts 64 steps in 838 µs natively against 21,531 µs for the port, from 148,512 bytes against 595,840. It stays behind the port on GIFT-Eval (on MASE in every term, band and domain; on WQL everywhere but the second band) and on Chronos-ZS.
- **What is published** (owner's decision of 15/11/2026, §11's last case: the file as evaluated, its first lines giving the verdict): `site/public/models/s6-cast2-pico.gguf`, the calibrated file the gate read (148,512 bytes, sha256 `30cbee8e846bc47d0876d9d77e3460278e58025cf01b6a5390b79b999215959f`), with its manifest entry (`gate: failed`) and its line in `site/public/models/forecast.json`; the Forecast page lists it next to v1, TinyCast, Timestral-5M and the seasonal naive, unticked by default; its card `hf/cards/s6-cast2-pico.md` and its Space entry in `hf/spaces.json` (kind `forecast`), both opening with the failure; `BENCH.md`, the README, Results, What we bring, Bench and the Models catalogue. The small result files are committed under `runs/s6-cast2-pico/` and `runs/s6-cast2-pilot/choice.json`; the logs, checkpoints and dumps are not.
- **The per-window readings, computed after the forecast dumps were rebuilt (18/11/2026)** (§9, reported, not gated; `runs/s6-cast2-pico/diagnostics.json` and the three per-window tables of `GATE.md`). No gated number changes: the gate step, run again with these readings, writes a `gate.json` identical byte for byte (FAIL, G1 false, G2 true, G3 reported) and a `GATE.md` that differs from the published one only by the 27 rows of those three tables (none removed, none changed).
  - **What was rebuilt, and how.** The diagnostics read four folders of `data/gift-eval-out/` (the `--model` arguments of the `gate` step of `scripts/train-cast2.sh`): v1 (`s6-cast-pico`) and v2 (`s6-cast2-pico`), both without flip, the TinyCast port with flip (`tinycast`) and the seasonal naive; the flip variants of v1 and v2 and the port without flip are not read and were not rebuilt. Their `*.q.bin` dumps were written again by the commands that wrote them first (`scripts/train-cast2.sh` for v2, `scripts/train-cast.sh` for the other three), with the same model files and flags and the `sky-cast-eval` binary of v2's evaluation: `sky-cast-eval run --windows data/gift-eval` with `models/s6-cast2/s6-cast2-pico.gguf` (the published file, sha256 `30cbee8e…`), with `models/s6-cast/s6-cast-pico.gguf` (sha256 `ff461860…`, the published v1), with `crates/sky-infer/tests/golden/tinycast.gguf --flip`, and `sky-cast-eval naive --windows data/gift-eval`; on the CPU, at the lowest priority, while another training used the GPU.
  - **The proof that they are the dumps that were scored.** Scored again with the script's own command (`scripts/gift-eval.py score`) into a temporary folder, the rebuilt dumps give back the published `all_results.csv` and `summary.json` byte for byte, for each of the four folders: 97 rows and 1,261 numbers compared per folder, 0 differences (5,044 numbers in all); v2's rewritten `exits.json` is also identical to the published one. The rebuilt dumps (about 1.4 GB per folder) are deleted again once these readings are committed, leaving each folder with the same result files as before, byte for byte.
  - **Coverage** (share of the labels at or below each quantile level, mean over the configurations of a term; nominal 0.10, 0.50, 0.90, and 0.80 inside the interval):

    | term | model | 0.1 | 0.5 | 0.9 | inside the 80% interval |
    |---|---|---:|---:|---:|---:|
    | short | v1 | 0.092 | 0.474 | 0.883 | 0.790 |
    | short | **v2** | 0.093 | 0.486 | 0.892 | **0.800** |
    | short | port | 0.091 | 0.502 | 0.909 | 0.819 |
    | medium | v1 | 0.178 | 0.442 | 0.756 | 0.578 |
    | medium | **v2** | 0.183 | 0.455 | 0.761 | **0.578** |
    | medium | port | 0.163 | 0.476 | 0.802 | 0.643 |
    | long | v1 | 0.200 | 0.443 | 0.725 | 0.526 |
    | long | **v2** | 0.214 | 0.461 | 0.735 | **0.522** |
    | long | port | 0.207 | 0.466 | 0.750 | 0.547 |

    The other levels are in `GATE.md`. (The seasonal naive gives one value for every level: 0.505, 0.552 and 0.550 of the labels at or below it.)
  - **The horizon profile** (each configuration's horizon cut into tenths; per tenth, v2's summed scaled error over the other model's, geometric mean over the configurations of a term; below 1, v2 is better):

    | term | ratio | 1st | 2nd | 3rd | 4th | 5th | 6th | 7th | 8th | 9th | 10th |
    |---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
    | short | v2 / port | 1.265 | 1.139 | 1.121 | 1.130 | 1.157 | 1.179 | 1.113 | 1.067 | 1.096 | 1.229 |
    | medium | v2 / port | 1.256 | 1.233 | 1.175 | 1.212 | 1.231 | 1.184 | 1.173 | 1.168 | 1.097 | 1.141 |
    | long | v2 / port | 1.187 | 1.184 | 1.152 | 1.214 | 1.250 | 1.206 | 1.173 | 1.167 | 1.158 | 1.189 |
    | short | v2 / seasonal naive | 0.765 | 0.808 | 0.853 | 0.845 | 0.852 | 0.803 | 0.858 | 0.831 | 0.851 | 0.896 |
    | medium | v2 / seasonal naive | 0.773 | 0.902 | 0.984 | 0.949 | 0.916 | 0.991 | 0.984 | 0.917 | 0.937 | 0.992 |
    | long | v2 / seasonal naive | 0.869 | 0.933 | 1.077 | 0.927 | 1.066 | 0.973 | 0.998 | 1.060 | 0.966 | 1.094 |

  - **The window-level splits** (v2's summed scaled error over the port's on the windows of one kind, geometric mean over the configurations with at least 20 such windows; the kinds overlap): a period detected by TinyCast's detector 1.140 (83 configurations), none detected 1.134 (33); a period under 32 points 1.200 (44), a multiple of the 32-point patch (±1) 1.120 (41), 32 or more and not a multiple 1.122 (64); observed points in the last 2,048: under 128 1.015 (13), 128 to 511 1.086 (18), 512 to 2,047 1.185 (23), all 2,048 1.164 (62).
  - **What they show.** v2's loss to the port is spread, not located: its error is 7% to 27% above the port's at every tenth of the horizon in every term, largest at the first tenth of the short and medium terms (so it is not an error that grows with the lead time), and 12% to 20% above it whether a period is detected or not and whether it fits the patch grid or not; the periodicity front end did not close the gap where it was aimed (1.140 with a detected period, 1.134 without). The gap is smallest on short histories (1.015 under 128 observed points) and largest with periods under 32 points (1.200) and with long histories (1.164 to 1.185). Its quantiles are calibrated at the short term (0.800 inside the 80% interval) and too narrow at the medium and long terms (0.578 and 0.522, v1's level; the port 0.643 and 0.547).
