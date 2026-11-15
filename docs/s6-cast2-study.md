# s6-cast v2: design study

**Date: 15/11/2026.** Read-only study. No training was run, no code was changed. Every number below was read from a file named next to it, or computed from such files by the method in §7. "TinyCast port" means our Rust port run with flip (`data/gift-eval-out/tinycast/`). It is the only TinyCast run on file, so every s6-cast/TinyCast comparison below sets s6-cast **without** flip against TinyCast **with** flip, unless stated otherwise.

## 0. Summary

- **What has to be closed.** v1 must go from 0.9471 to 0.838 nGMASE and from 0.6646 to 0.587 nWQL. On the log scale of the geometric mean, that is **65% of its MASE gap to the TinyCast port and 70% of its WQL gap**. Halving the gap in every configuration would still fail (0.863 / 0.608).
- **Where v1 loses.** The loss is concentrated in long contexts with a clear period: high-frequency series (sub-hourly, second-level and hourly configurations carry 90% of the MASE gap) whose period TinyCast's detector finds in at least 90% of windows (82% of the MASE gap, 95% of the WQL gap). On short contexts (fewer than 128 observed points) v1 matches TinyCast (window-level ratio 1.015). In configurations where the detector fires in fewer than half the windows it also matches TinyCast (0.839 against 0.840) and has a lower WQL (0.630 against 0.661).
- **What is not the cause.**
  - *The rollout.* In medium and long configurations, v1's error relative to TinyCast is flat over the horizon: 1.25 to 1.44 in every tenth, and already 1.38 / 1.31 in the first tenth, inside the first pass.
  - *The 2,017 fallback windows.* Zero effect: v1 and TinyCast both forecast 0 on them, and 1,221 of the 2,017 have no MASE scale at all.
  - *Interval calibration.* Short-term coverage is right (80% interval: 0.79 for v1, 0.82 for TinyCast). Both under-cover at medium and long terms.
  - *Ternary weights and training length.* Both are second-order. On validation, the ternary EMA costs 2.4% over the pre-QAT f32 weights, and the validation pinball fell only 2.6% over the last 90k steps before QAT.
- **What TinyCast does differently.** It works per point, not per 32-point patch. It computes periods and folds the context by phase. It builds a seasonal draft of the horizon in value space. And it trains on about 93% real series from every frequency band (GIFT-Eval-Pretrain), 150M samples. v1 trained on 64M windows, 70% synthetic, and its real series include nothing sampled faster than every 30 minutes.
- **Recommended v2.** Keep the s6-cast graph and add a computed-periodicity front end. The front end has two parts: phase channels on the patches, and a value-space seasonal draft added to each decoder query. Also fix the training data: the frequency bands of GIFT-Eval, and the generator periods 144, 288 and 360. Train for 720 working minutes. The cost is about 1,000 lines of code (§3) and about 760 working minutes of compute.
- **Honest odds.** About one in three for the GIFT-Eval bars, lower once the added gate terms (§5) are counted.

## 1. Where v1 loses (question 1)

Ratios are each configuration's metric divided by the seasonal-naive reference (`eval/gift-eval/seasonal_naive.csv`), averaged geometrically, as in `scripts/gift-eval.py summarize`. "Share of gap" is a group's share of the sum, over the 97 configurations, of log(v1 / TinyCast port). The total of that log is 0.1869 for MASE and 0.1766 for WQL. "> SN" counts the configurations where v1's MASE is above the seasonal naive's.

### 1.1 By term

| term | n | v1 MASE | v1 flip MASE | TinyCast MASE | v1 WQL | TinyCast WQL | v1 > SN | share of gap MASE / WQL |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| short | 55 | 0.860 | 0.840 | 0.756 | 0.638 | 0.571 | 16 | 39% / 36% |
| medium | 21 | 1.046 | 0.991 | 0.806 | 0.696 | 0.534 | 13 | 30% / 33% |
| long | 21 | 1.105 | 1.031 | 0.848 | 0.706 | 0.546 | 16 | 31% / 31% |

### 1.2 By frequency band

| band | n | v1 MASE | TinyCast MASE | v1 WQL | TinyCast WQL | v1 > SN | share of gap MASE / WQL |
|---|---:|---:|---:|---:|---:|---:|---|
| second (10S) | 6 | 1.298 | 0.909 | 0.896 | 0.702 | 6 | 12% / 9% |
| sub-hourly (5T, 10T, 15T) | 30 | 1.046 | 0.794 | 0.735 | 0.571 | 21 | 45% / 44% |
| hourly | 31 | 0.914 | 0.754 | 0.606 | 0.480 | 12 | 33% / 42% |
| daily | 15 | 0.782 | 0.711 | 0.494 | 0.465 | 4 | 8% / 5% |
| weekly | 8 | 0.908 | 0.858 | 0.729 | 0.732 | 1 | 3% / 0% |
| monthly and coarser | 7 | 0.878 | 0.890 | 0.850 | 0.853 | 1 | −1% / 0% |

By band and term, the largest cells are sub-hourly long (v1 1.245 against 0.877, 19% of the MASE gap) and sub-hourly medium (1.139 against 0.850, 16%). Sub-hourly short is 0.806 against 0.672.

### 1.3 By domain

| domain | n | v1 MASE | TinyCast MASE | v1 WQL | TinyCast WQL | share of gap MASE / WQL |
|---|---:|---:|---:|---:|---:|---|
| Energy | 32 | 1.080 | 0.882 | 0.800 | 0.661 | 36% / 35% |
| Web/CloudOps | 20 | 0.983 | 0.750 | 0.662 | 0.562 | 30% / 19% |
| Transport | 15 | 0.833 | 0.669 | 0.644 | 0.510 | 18% / 20% |
| Nature | 15 | 0.862 | 0.757 | 0.467 | 0.358 | 11% / 23% |
| Econ/Fin | 6 | 1.018 | 0.894 | 0.901 | 0.841 | 4% / 2% |
| Healthcare | 5 | 0.760 | 0.735 | 0.614 | 0.639 | 1% / −1% |
| Sales | 4 | 0.753 | 0.734 | 0.457 | 0.445 | 1% / 1% |

These per-term and per-domain values are those of `docs/big-run-report.md` §5, which this computation reproduces.

### 1.4 By seasonality and period detection

Two measures were computed on each window's last 2,048 points:

- **Seasonality strength:** the autocorrelation at the dataset's gluonts seasonality m. The 73 configurations with m > 1 are split into terciles at 0.409 and 0.700.
- **Detection:** whether TinyCast's zero-parameter detector (`tinycast::significant_periods`, re-implemented in numpy on the port's prepared window) finds a significant period.

| group | n | v1 MASE | TinyCast MASE | v1 WQL | TinyCast WQL | share of gap MASE / WQL |
|---|---:|---:|---:|---:|---:|---|
| no seasonality (m = 1) | 24 | 0.824 | 0.766 | 0.576 | 0.556 | 10% / 5% |
| weak (ACF < 0.41) | 24 | 0.832 | 0.706 | 0.603 | 0.532 | 22% / 18% |
| mid | 24 | 1.047 | 0.835 | 0.669 | 0.509 | 30% / 38% |
| strong (ACF ≥ 0.70) | 25 | 1.113 | 0.842 | 0.832 | 0.636 | 38% / 39% |
| detector fires < 50% of windows | 11 | 0.839 | 0.840 | 0.630 | 0.661 | 0% / −3% |
| fires 50–90% | 16 | 1.011 | 0.823 | 0.609 | 0.558 | 18% / 8% |
| fires ≥ 90% | 70 | 0.951 | 0.769 | 0.684 | 0.542 | **82% / 95%** |

Within configurations, the window-level ratio is the v1 / TinyCast sum of scaled errors over the windows of one kind, averaged geometrically over configurations with at least 20 such windows:

| windows | configs | v1 / TinyCast MASE |
|---|---:|---:|
| a period detected | 83 | 1.235 |
| no period detected | 33 | 1.157 |
| detected period a multiple of the 32-point patch (±1) | 41 | 1.161 |
| detected period ≥ 32 and not a multiple of 32 | 64 | 1.233 |
| detected period < 32 (inside one patch) | 44 | 1.240 |
| detected period reachable by the generators (the 19 periods of `synth::PERIODS`, or half or a quarter of one) | 72 | 1.204 |
| detected period not reachable | 55 | 1.215 |

Reading:

1. The loss sits where a period exists. A period that does not line up with the patch grid costs about 6% more than one that does. The comparison is confounded by dataset, so it is suggestive only.
2. Whether the period is in the synthetic generators' set makes no difference (1.204 against 1.215). So "the generators lack periods 144 and 288" is **not** supported as the cause at window level.
3. The clearest single case is `m4_hourly/H/short`. The period is 24 and the ACF 0.88, a clean daily cycle, and it is in the generator set. v1's MASE ratio is 1.598, worse than repeating the last day; TinyCast's is 0.858. A 32-point patch model has to learn a lag of 24 points across patch boundaries; TinyCast's per-point features make that lag explicit.

### 1.5 By context length

| configurations by median context | n | v1 MASE | TinyCast MASE | share of gap MASE / WQL |
|---|---:|---:|---:|---|
| < 256 points | 12 | 0.918 | 0.904 | 1% / −4% |
| 256–2,047 | 19 | 0.828 | 0.756 | 10% / 6% |
| full 2,048 | 66 | 0.990 | 0.774 | 89% / 98% |

At window level (same rule as §1.4), by observed points in the last 2,048:

| observed points | configs | v1 / TinyCast MASE |
|---|---:|---:|
| < 128 | 13 | 1.015 |
| 128–511 | 18 | 1.132 |
| 512–2,047 | 23 | 1.195 |
| 2,048 | 62 | 1.293 |

v1 gains less than TinyCast from a long context. Long contexts are mostly high-frequency series, so this measure and the band measure overlap.

### 1.6 Along the horizon (is it the rollout?)

Each configuration's horizon is cut into tenths. For each tenth, the ratio of the summed scaled errors is averaged geometrically over the configurations of the term. v1's pass is 64 steps and TinyCast's 48. A medium or long horizon is 480 to 900 steps, so it takes 8 to 15 passes for v1 and 10 to 19 for TinyCast.

| term | tenth: 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| short, v1 / TinyCast | 1.348 | 1.217 | 1.162 | 1.111 | 1.217 | 1.303 | 1.122 | 1.118 | 1.125 | 1.232 |
| medium, v1 / TinyCast | 1.379 | 1.381 | 1.371 | 1.441 | 1.378 | 1.281 | 1.282 | 1.368 | 1.251 | 1.302 |
| long, v1 / TinyCast | 1.314 | 1.330 | 1.287 | 1.334 | 1.337 | 1.369 | 1.358 | 1.268 | 1.291 | 1.333 |
| medium, v1 / seasonal naive | 0.849 | 1.010 | 1.149 | 1.129 | 1.025 | 1.072 | 1.076 | 1.074 | 1.069 | 1.131 |
| long, v1 / seasonal naive | 0.962 | 1.048 | 1.203 | 1.019 | 1.140 | 1.104 | 1.155 | 1.152 | 1.077 | 1.226 |

The ratio to TinyCast does not grow along the rollout. The gap is already there in the first tenth (48 to 90 steps, mostly inside v1's first pass). For medium and long configurations, **the first pass itself is worse.** So changing the output length or training on rollouts addresses no measured loss.

### 1.7 Quantile calibration

The table gives, per level, the share of valid label points at or below the quantile (the ideal is the level itself). Each configuration's share is computed first, then averaged over configurations.

| term | model | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 | 80% interval |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| short | v1 | 0.092 | 0.175 | 0.264 | 0.351 | 0.474 | 0.577 | 0.682 | 0.790 | 0.883 | 0.790 |
| short | v1 flip | 0.090 | 0.187 | 0.296 | 0.404 | 0.527 | 0.630 | 0.727 | 0.822 | 0.899 | 0.809 |
| short | TinyCast | 0.091 | 0.178 | 0.270 | 0.378 | 0.502 | 0.617 | 0.713 | 0.810 | 0.909 | 0.817 |
| medium | v1 | 0.178 | 0.245 | 0.305 | 0.365 | 0.442 | 0.512 | 0.591 | 0.671 | 0.756 | 0.577 |
| medium | TinyCast | 0.163 | 0.250 | 0.318 | 0.385 | 0.476 | 0.549 | 0.615 | 0.695 | 0.802 | 0.639 |
| long | v1 | 0.200 | 0.264 | 0.320 | 0.375 | 0.443 | 0.505 | 0.575 | 0.648 | 0.725 | 0.525 |
| long | TinyCast | 0.207 | 0.280 | 0.334 | 0.390 | 0.466 | 0.523 | 0.578 | 0.646 | 0.750 | 0.543 |

- Short-term calibration is as good as TinyCast's.
- At medium and long terms, both models' intervals are too narrow, by about the same amount. The TinyCast paper reports 83.0 / 67.4 / 60.0% for its 80% interval, pooled differently.
- v1's median is slightly low: 0.44–0.47 of labels fall below it, against 0.47–0.50 for TinyCast.
- The WQL gap follows the point-forecast gap. It is not a calibration problem.

### 1.8 The 2,017 fallback windows

They lie in 12 configurations (`runs/all/cast-eval.log`): bitbrains ×8, `kdd_cup_2018/H` ×3, `electricity/W/short` ×1. Replacing v1's forecast by TinyCast's on exactly these windows changes **nothing**: nGMASE 0.94710 → 0.94710, nWQL 0.66457 → 0.66457.

- Their contexts are all missing. v1's fallback (seasonal naive of an empty history) and TinyCast's zero fill both forecast 0.
- 1,221 of them have no MASE scale and are excluded by gluonts. The `kdd_cup_2018` ones have no valid label point at all.

No v2 change is needed here.

### 1.9 What would pass: counterfactuals

Each line replaces v1's per-configuration ratios by the TinyCast port's on a subset. This is an upper bound on what fixing that subset could give.

| v1 matches TinyCast on | nGMASE | nWQL | passes 0.838 / 0.587 |
|---|---:|---:|---|
| nothing (v1) | 0.9471 | 0.6646 | no |
| medium and long terms | 0.8453 | 0.5935 | no |
| sub-hourly and second-level | 0.8509 | 0.6053 | no |
| Energy | 0.8859 | 0.6243 | no |
| configurations where the detector fires ≥ 90% | **0.8127** | **0.5621** | yes |
| half the log gap, in every configuration | 0.8626 | 0.6084 | no |
| v1 with flip, as measured | 0.9099 | 0.6417 | no |

### 1.10 The twenty worst configurations (log of v1 / TinyCast MASE)

| config | band | log ratio | v1 MASE | TinyCast MASE | detected period (mode) | season ACF |
|---|---|---:|---:|---:|---:|---:|
| solar/10T/long | sub-hourly | 1.031 | 2.729 | 0.973 | 146 | 0.84 |
| solar/10T/medium | sub-hourly | 0.871 | 2.282 | 0.955 | 146 | 0.85 |
| m4_hourly/H/short | hourly | 0.622 | 1.598 | 0.858 | 24 | 0.88 |
| bitbrains_fast_storage/5T/long | sub-hourly | 0.528 | 1.453 | 0.857 | 3 | 0.17 |
| bizitobs_application/10S/short | second | 0.495 | 1.057 | 0.644 | 341 | 0.99 |
| bitbrains_fast_storage/5T/medium | sub-hourly | 0.452 | 1.372 | 0.873 | 3 | 0.18 |
| solar/H/medium | hourly | 0.417 | 1.371 | 0.903 | 24 | 0.84 |
| jena_weather/10T/long | sub-hourly | 0.406 | 1.258 | 0.838 | 146 | 0.42 |
| loop_seattle/H/long | hourly | 0.398 | 0.937 | 0.629 | 24 | 0.57 |
| bizitobs_service/10S/medium | second | 0.398 | 1.450 | 0.975 | 341 | 0.66 |
| loop_seattle/H/medium | hourly | 0.390 | 0.998 | 0.676 | 24 | 0.57 |
| bitbrains_rnd/5T/short | sub-hourly | 0.380 | 1.282 | 0.877 | 293 | 0.27 |
| bitbrains_rnd/5T/long | sub-hourly | 0.357 | 1.401 | 0.980 | 293 | 0.25 |
| bizitobs_service/10S/short | second | 0.355 | 1.123 | 0.787 | 341 | 0.66 |
| m_dense/H/short | hourly | 0.349 | 0.849 | 0.599 | 24 | 0.78 |
| bizitobs_service/10S/long | second | 0.346 | 1.627 | 1.151 | 341 | 0.64 |
| loop_seattle/5T/medium | sub-hourly | 0.337 | 1.146 | 0.819 | 293 | 0.48 |
| m_dense/H/medium | hourly | 0.322 | 0.740 | 0.536 | 24 | 0.77 |
| ett2/W/short | weekly | 0.321 | 1.831 | 1.329 | 146 | n/a |
| bizitobs_application/10S/medium | second | 0.320 | 1.341 | 0.974 | 341 | 0.99 |

`solar/10T/short` is the exception: v1 1.063, TinyCast 1.094. v1 is lower in 11 of 97 configurations on MASE and in 10 on WQL.

## 2. What TinyCast does that s6-cast does not (question 2)

Sources: the paper (arXiv 2608.15767v1, method, Appendix A–E), our port (`crates/sky-infer/src/cast/tinycast.rs`), and s6-cast (`crates/sky-infer/src/cast/{model,prep,mod}.rs`, `crates/sky-train/src/cast/{data,model,train}.rs`, `docs/SPEC-S6-CAST.md`).

| component | TinyCast (146,505 parameters) | s6-cast-pico v1 (369,792, ternary) |
|---|---|---|
| input normalisation | min-max over the 2,048-point window. A short history is left-padded with its first value; gaps are linearly interpolated, ends filled with the nearest value. | min-max over the observed points. Missing values are 0 with mask 0. A short history is masked left padding (dropped at inference). |
| resolution | per point, 2,048 positions | 32-point patches, 64 tokens |
| period detection | zero-parameter FFT periodogram with a Bonferroni-Fisher threshold (α = 0.05), top 4 periods, every window | none |
| positional input | value plus 13 channels: sin/cos of 2πt/p_k for 4 periods, and 5 recency channels | RoPE on patch index; the patch values and mask through `patch_proj` |
| receptive field | 10 dilated causal depthwise-separable convolutions (kernel 3, dilations 1–512): 2,047 points, d = 64, one SwiGLU shared by all blocks | full attention over 64 patches, 2 blocks × 2 loops, d = 96, GeGLU, 4 exits |
| phase folding | encoder states averaged in 16 phase bins per period; each future step reads its bin; a 256→64 mix | none |
| horizon construction | 48 queries built from the future step's own period phase, the mean and last encoder states and the phase readout. Plus a **future-conv correction**: a 6-block causal conv over a value-space seasonal draft (phase-binned mean of the past at the same phase) appended to the last 128 encoder states. | 2 learned position-only queries (`cast_query`); 2 cross-attention blocks; each query emits a 32-step patch |
| output | residual SwiGLU and RMSNorm, linear 64→9; clamped to [−5, 5] normalised; quantiles sorted | `quant_head` d → 32·9; monotone by softplus increments around the median |
| long horizons | 48-step blocks, median fed back; trained with **4 AR blocks under scheduled sampling** (feedback probability 0 → 0.5) | 64-step passes, median fed back; trained on one 64-step pass only |
| training data | GIFT-Eval-Pretrain (the only real cache), filtered by a 2,020-name overlap list and 66 gridded-weather names, plus KernelSynth and 250,000 synthetic series (GP / pulse / trend-season). Synthetic share about 7%. Sampling balanced over frequency bands, weighted like GIFT-Eval's 97 configurations: hourly 31, sub-hourly 30, daily 15, weekly 8, monthly+ 7, second 6. | 70% synthetic windows (GP, pulse, trend-season, SARIMA); 30% real from 10,789 series and 44.5M points. The real series are London smart meters (30 min), wind farms (hourly and daily), pedestrians, rideshare (hourly), Wikipedia and USHCN (daily), and game states. **No 5T/10T/15T/second-level series and no CloudOps series.** |
| amount | 150M samples, batch 4,096, about 36,600 steps; 8 RTX 3090, 7.8 h | 250,092 steps × 256 = 64M windows, 480 working minutes on one M5 |
| loss | 9-quantile pinball plus a gated "committing" term (λ 0.3) on the median, where copying the last cycle would have been better | 9-quantile pinball at every exit (deep supervision), plus an L1 term towards TinyCast's deciles on 50% of windows (weight 0.5 → 0) |
| augmentation | each at p = 0.5: time flip, sign flip, stride 2/3/4 downsampling, same-bucket mixup Beta(0.2, 0.2) | each at p = 0.3: stride 2/4, amplitude modulation, sign flip, time reversal, censoring; mixup 0.1; 1–5% missing at 0.1 |
| weights | uniform average of 8 checkpoints (steps 29k–36k), f32 (INT8 costs +2.1% nGMASE) | EMA 0.999, ternary QAT over the last 20% |
| inference extras | sign symmetrisation (flip), and in the paper's driver a canonical-period alignment on 2 configurations | flip optional; early exits |

**What the paper's ablations say.** nGMASE throughout; † marks intervals that span zero.

- **Periodicity and phase folding: the largest effects measured,** in the architecture family (340K control at 1.0225):
  - turning the detector off: +0.131;
  - adding phase binning: −0.098;
  - phase binning plus a recency gate: −0.113.
- **But retrained at full budget without the detector,** the deployed model loses only +0.0071 nGMASE and +0.0042 nWQL. Suppressing the detector in the trained model costs about ten times that.
  - Reading: with enough real data, the per-point dilated convolutions learn most of the period structure themselves. The detector matters most when the model cannot.
  - A data-blind fixed period set is worse than no periods.
- **Component family** (394K control, 0.8123 / 0.5699):
  - future-conv correction: −0.0151 / −0.0085;
  - future-conv plus the synthetic families: −0.0263 / −0.0170;
  - synthetic blend alone: −0.0062†;
  - committing loss: −0.0023†;
  - 8 AR chunks instead of 4: +0.0010†;
  - MASE-weighted loss: +0.0023†;
  - backtest-chosen period: +0.0044†.
- **Data budget:** 50M samples is +0.037 against 100M; 150M is −0.003† (a plateau). Peak learning rate 1e-3 costs +0.040 against 3e-3.
- **Inference:** sign symmetrisation is worth 0.0079 / 0.0061, alignment 0.0120 / 0.0112.
- **Most to least important,** for a model in s6-cast's position:
  1. period information when the architecture cannot extract it;
  2. data in every frequency band and enough of it (≥ 100M samples);
  3. a value-space seasonal draft of the horizon (future conv);
  4. everything else, which is within resolution.

## 3. The smallest change that addresses the measured loss (question 3)

### 3.1 The training log (`runs/s6-cast-pico/log.jsonl`)

| steps | last-exit validation pinball, f32 (mean of the evaluations in the bucket) |
|---|---:|
| 0–10k | 0.0529 |
| 50k–60k | 0.0433 |
| 100k–110k | 0.0430 |
| 150k–160k | 0.0423 |
| 190k–199.6k | 0.0419 (0.04208 at 199,658, the pre-QAT checkpoint) |
| 250,092, ternary `last` / ternary EMA | 0.0442 / 0.0431 |

- The learning rate was at full scale until QAT began at about 80% of the budget. It then decayed to 0.1 during QAT.
- The f32 validation pinball was still falling at the end of the stable phase, by about 2.6% over its last 90k steps (about 170 working minutes).
- The final f32 value (0.589) is that of latent weights after QAT. It does not measure the model.
- This validation set is synthetic plus held-out real series of the training sources, so it has no sub-hourly series. It did not and could not anticipate the GIFT-Eval result, which is where the loss lies.

### 3.2 Options

| option | what it addresses (v1 evidence) | code (files) | training budget | evidence it helps | risk |
|---|---|---|---|---|---|
| **A. Computed-periodicity front end into the existing graph**: TinyCast's detector on the normalised context; per-patch phase channels; a value-space seasonal draft (lag-p copy and phase-binned mean of the dominant period) added to each decoder query | 82% / 95% of the gap is in configurations where the detector fires ≥ 90%; detected-period windows lose more (1.235 against 1.157); misaligned periods lose more (1.233–1.240 against 1.161); `m4_hourly` loses to a daily copy | sky-infer `cast/prep.rs` (periods, phase, draft; reuse `tinycast::significant_periods`), `cast/model.rs` (wider `patch_proj`, `draft_proj` in `decode`), `cast/mod.rs` (GGUF keys); sky-train `cast/config.rs`, `cast/data.rs` (computed in the producer, two new batch tensors), `cast/model.rs` (graph), `cast/gguf.rs`, `cast/golden.rs`; `tests/golden_cast.rs`; about 600–700 lines | none extra; the FFT is about 20 µs per window in the producer | TinyCast family: detector +0.131, phase binning −0.098, future conv −0.015 | TinyCast's no-detector retrain cost only +0.007: the front end may matter less once data is right. A draft that is wrong (trend, no period) can mislead. |
| **B. Training data matched to GIFT-Eval's bands**: real sub-hourly, second-level and CloudOps series with clear licences, absent from both test sets; band-balanced sampling like TinyCast's | sub-hourly and second-level carry 57% of the MASE gap, and v1 saw no real series at those rates | `sky-data` `fetch-series` (new sources), `DATA_LICENSES.md`, `cast/data.rs` (band weights); about 150–250 lines | the same minutes; one download | TinyCast trains on all bands; 50M → 100M samples −0.037 | licences, size (GIFT-Eval-Pretrain is 975 GB in all), leakage by renamed copies |
| B′. **GIFT-Eval-Pretrain itself** (a filtered subset) | as B | as B, plus TinyCast's 2,020-name filter and a Chronos-ZS filter | as B | closest to TinyCast's data | The Hugging Face card now declares `license: apache-2.0`, while `DATA_LICENSES.md` says "no declared license" (out of date). The card also says "for research purposes only", and the constituent datasets keep their own terms. This is an owner decision, and per-source checking would be needed. |
| C. Generator periods 144, 288, 360, 1008, 1440, 2016 (and longer pool series for the longest) | sub-hourly daily and weekly cycles are absent from `synth::PERIODS` | `crates/sky-infer/src/cast/synth.rs` and its tests; about 20 lines | none | weak: the window-level test found no difference (§1.4, 1.204 against 1.215) | low |
| D. Sign flip and time reversal at p = 0.5 | flip gave v1 −0.037 nGMASE against TinyCast's −0.008: v1 is not sign-symmetric | `synth::augment` and `SampleOptions`; about 20 lines | none | indirect | low |
| E. More steps (720 or 960 minutes) | the f32 validation was still falling, slowly | none (`MINUTES`) | +240 / +480 minutes | 2.6% on validation over the last 90k steps; TinyCast's curve flattens at 100M samples, and v1 saw 64M | small gain alone. It cannot close a 65% gap: at 2.6% per 90k steps, 480 more minutes buys about 6% on validation. |
| F. Long horizons: a larger pass (e.g. 8 patches = 256 steps), or rollout training with scheduled sampling | **none measured**: the ratio is flat over the horizon (§1.6) | `cast/data.rs`, `cast/train.rs` (rollout graph), configs | rollout training costs 2–4× per window | TinyCast: 8 against 4 chunks is null; 1 against 4 is not reported | spends budget on a non-cause |
| G. Fallback windows (TinyCast-style fill) | **none**: zero effect (§1.8) | `cast/prep.rs` | none | none | none, and no gain |
| H. f32 instead of ternary | the ternary EMA costs 2.4% on validation | none | none | TinyCast INT8 costs +2.1% | gives up the project's premise; a few % at most |
| I. Bigger model | v1 already has 2.5× TinyCast's parameters | config | throughput falls | none | — |

**Recommended:** A + B (or B′ if the owner accepts its terms) + C + D, at 720 working minutes (option E at 1.5×). F, G, H and I are not recommended: F and G have no measured loss to fix, H gives up the premise, I has no evidence.

**Why A and B together.** v1's evidence points at A: per-patch alignment, periodic configurations, the `m4_hourly` copy failure. TinyCast's strongest evidence that A is secondary comes from a model that already had all bands of real data. Neither change alone is likely to close 65% of the gap.

**Code in total:** about 1,000 lines over sky-infer, sky-train, sky-data and `scripts/train-cast.sh`, plus the latency harness and the Chronos-ZS harness of §5. The Chronos-ZS harness is a new script, about 200 lines.

**Compute:** about 760 working minutes on the M5. That is a 24-minute pilot, 720 minutes of training, and about 15 minutes of export, calibration and GIFT-Eval. v1 needed 13 hours of wall clock for 480 working minutes in the brake modes, so this is about 20 hours.

## 4. A ternary retrain of the TinyCast architecture (question 4)

**Licence.** TinyCast's code and weights are Apache-2.0: the paper's licence section, `DATA_LICENSES.md`, and `crates/sky-infer/tests/golden/TINYCAST-NOTICE`. Apache-2.0 allows:

- re-implementing and retraining the architecture;
- distributing our own weights under our licence.

It requires:

- keeping the NOTICE and the licence for any code derived from theirs (`tinycast.rs` is a port, so derived);
- marking our changes;
- no use of their name to endorse our model.

"Same architecture as TinyCast (RAWS Labs), retrained with our data and ternary weights" is a legitimate, credited description. A name of our own (e.g. `s6-tinycast-t`) avoids implying it is theirs. Their rollout predictor is adapted from Reverso (MIT), whose notice would also carry over if we reuse it.

**What it would add over using the port:**

- **Size.** About 146K ternary weights is about 40–50 KB. The port is 596 KB as f32 GGUF, and TinyCast's own INT8 is 138 KiB.
- **Data provenance.** Ours, if trained on licence-checked data.
- A ternary path in the port's engine.

It adds no accuracy. TinyCast's own INT8 costs +2.1%, and ternary would cost more.

**Why it is not useful as v2:**

- **Compute.** TinyCast runs its convolutions over all 2,048 positions, so a training window costs roughly 10× an s6-cast window (2,048 positions at d = 64 against 64 tokens at d = 96). At v1's budget we would see about 5–10M samples, against the 150M TinyCast needed. Its own ablation puts 50M samples 0.037 nGMASE behind 100M, so a 480- or 720-minute ternary retrain would be heavily under-trained.
- **Data.** Without GIFT-Eval-Pretrain we cannot reproduce its data.
- **Code.** It needs a new training graph in mlx-rs: dilated depthwise convolutions, the phase-profile scatter-mean as a one-hot matmul, the future conv, scheduled-sampling rollout. Plus a ternary inference path in `tinycast.rs`. About 1,200–1,500 lines.
- **Expected result.** A model probably worse than the port, which we already have.

If what the owner wants is a working forecaster, the port is already that (Apache-2.0, 0.7856 / 0.5570). Two zero-training options exist:

- ship the port in the browser;
- post-training INT8 of the port's weights, which the paper measures at +2.1%.

A ternary TinyCast is worth a separate, small experiment only if size below 64 KB becomes a goal. It would need its own gate against the port.

## 5. The gate for v2 (question 5)

Recommended; the draft pre-registration gives the reasons.

1. **G1: the same two bars, without flip:** nGMASE ≤ 0.838 and nWQL ≤ 0.587. Unchanged, so that v2 is not judged by a bar moved after v1. The FLAIR bar is already 6.7% above the TinyCast port (0.838 / 0.7856), so G1 is an "approach TinyCast" bar.
2. **G2: better than the seasonal naive at every term.** The geometric mean MASE ratio must be < 1.00 for short, medium and long. v1 fails it: 1.046 medium, 1.105 long. The TinyCast port passes: 0.756 / 0.806 / 0.848.
3. **G3: confirmation on a benchmark the design never read.** Chronos-ZS (the 27 tasks of TinyCast's table), relative MASE and WQL, each ≤ 1.083 × the TinyCast port's on the same harness. 1.083 = 0.838 / 0.774, the distance G1 allows from the published TinyCast. The harness is checked first: the port must reproduce 0.880 / 0.722 within ±0.02 before v2 is read. This answers the disclosure that v2 is designed after reading v1's GIFT-Eval results. **Open:** the harness does not exist yet. Without it, G3 becomes a reported number, and the verdict must say G1 is partly in-sample for the design.
4. **G4: latency.** v2's native and WASM median latency, on one core, for a 2,048-point context and a 64-step horizon at full depth, must not exceed the TinyCast port's measured in the same session. Absolute numbers are reported. Not beating the port on speed or size would leave no reason to use v2 rather than the port.
5. **Not gated:**
   - beating the TinyCast port (the v1 goal, reported as a gap);
   - per-quantile and 80%-interval coverage by term;
   - nMSIS, the flipped model, and the TinyCast port without flip (to run once, as an apples-to-apples reference);
   - per-domain and per-band tables;
   - the exits used on GIFT-Eval;
   - the size.

## 6. What could not be determined

- **TinyCast without flip on our harness.** It has never been run. The paper's symmetrisation deltas suggest about 0.794 / 0.563; this is an estimate, not a measurement.
- **v1's f32 (pre-QAT) GIFT-Eval score.** The ternary cost on the benchmark is unknown; only the validation cost (2.4%) is known. Measuring it needs a GIFT-Eval run of `runs/s6-cast-pico/pre-qat`, which is outside this study's light-CPU scope.
- **Latency of v1 and of the port.** Neither was measured. `docs/SPEC-S6-CAST.md` gives 869 µs for a 60-step probe checkpoint at 2,048 points. The v1 evaluation ran its 371,330 windows (with rollouts) in 79.5 s on all cores without flip and 162.4 s with flip (`runs/all/cast-eval.log`). That is a throughput, not a latency.
- **Whether A or B matters more for s6-cast.** No ablation of ours separates them. TinyCast's ablations point both ways (§2).
- **Licences of candidate sub-hourly and CloudOps sources.** Not checked source by source. For GIFT-Eval-Pretrain, only the card's declared Apache-2.0 tag and its research-only statement were read.
- **Per-domain and per-term results of TinyCast as published.** The paper does not report them. Ours come from `eval/gift-eval/gift_eval_tinycast.csv`.
- **The paper text was read through its HTML version.** Appendix E (latency and memory) covers the microcontroller only, not a desktop CPU.

## 7. Method

- **Inputs.** Per-window quantiles: `data/gift-eval-out/{s6-cast-pico,s6-cast-pico-flip,tinycast,seasonal-naive}/<config>.q.bin`. Contexts: `data/gift-eval/<config>.{bin,json}`.
- **Labels and MASE scales.** Read through `gift_eval.data.Dataset`, as `scripts/gift-eval.py score` does: the full input history and gluonts' seasonality, with NaN labels masked.
- **Check.** For each of the 4 models × 97 configurations, the per-window sums give back the MASE and WQL of each `all_results.csv`. The worst relative difference is 1.7e-7.
- **Measures.** Coverage, horizon tenths, fallback substitution, detector and ACF were then computed per window.
- **Load.** Light CPU (`nice -n 19`, two threads), about 15 minutes. Nothing was written to the repository.
