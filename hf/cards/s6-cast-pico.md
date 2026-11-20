---
license: apache-2.0
library_name: gguf
pipeline_tag: time-series-forecasting
tags:
  - gguf
  - time-series
  - forecasting
  - probabilistic-forecasting
  - ternary
  - rust
  - wasm
  - skycmd
  - failed-gate
  - negative-result
---

# s6-cast-pico

> **Failed its pre-registered gate: published as a result, not for use.** The gate
> ([`docs/s6-cast-prereg.md`](https://github.com/maxgfr/skycmd/blob/main/docs/s6-cast-prereg.md)
> §6) was to beat the training-free FLAIR on GIFT-Eval: nGMASE ≤ 0.838 and nWQL ≤ 0.587, without
> flip. s6-cast-pico scored **nGMASE 0.9471** and **nWQL 0.6646**: it failed on both metrics, and
> with flip too (0.9099 / 0.6417). The goal of beating TinyCast (0.774 / 0.545 as published) is not
> met either. The file is published so that the result can be checked and run (owner's decision of
> 15/11/2026, item 9 of the pre-registration's amendments); the verdict does not change. Numbers:
> [BENCH.md, section s6-cast-pico](https://github.com/maxgfr/skycmd/blob/main/BENCH.md#s6-cast-pico-probabilistic-forecasting-pre-registered-gate-failed).

> **Its second version failed too.** [s6-cast2-pico](https://huggingface.co/{{USER}}/s6-cast2-pico),
> with a computed-periodicity front end, scored nGMASE 0.8847 and nWQL 0.6190 on the same benchmark
> (18/11/2026), against the same bars. By its pre-registration the s6-cast line stops there: the
> forecaster the project offers is the TinyCast port.

**A 143 KB ternary probabilistic forecaster built on the s6 encoder.** 369,792 parameters. Given
up to 2,048 past points it returns nine quantiles (the deciles) for each of the next 64 steps, and
longer horizons by rolling out, with an exit after each loop.

- **Demo:** [huggingface.co/spaces/{{USER}}/s6-cast-pico](https://huggingface.co/spaces/{{USER}}/s6-cast-pico)
  (next to TinyCast and the seasonal naive, in your browser)
- **Code:** [github.com/maxgfr/skycmd](https://github.com/maxgfr/skycmd) (100 % Rust)
- **Spec:** [docs/SPEC-S6-CAST.md](https://github.com/maxgfr/skycmd/blob/main/docs/SPEC-S6-CAST.md)
- **Report of the run:** [docs/big-run-report.md](https://github.com/maxgfr/skycmd/blob/main/docs/big-run-report.md) §5
{{LINKS}}

## Results

GIFT-Eval, 97 configurations, 371,330 windows, the official protocol (gluonts `evaluate_model`,
the 9 deciles); every metric is divided by the seasonal naive and averaged geometrically:

| model | nGMASE | nWQL | nMSIS |
|---|---:|---:|---:|
| **s6-cast-pico, without flip** (the reported model) | **0.9471** | **0.6646** | 0.7190 |
| s6-cast-pico, with flip | 0.9099 | 0.6417 | 0.6860 |
| gate: FLAIR, training-free (TinyCast paper) | 0.838 | 0.587 | |
| goal: TinyCast as published | 0.774 | 0.545 | |
| TinyCast, our port, with flip | 0.7856 | 0.5570 | 0.5781 |
| seasonal naive, our harness | 0.9999 | 0.8876 | 2.6778 |

2,017 windows in 12 configurations have no observation in the model's last 2,048 points and are
forecast by the seasonal naive, in both passes.

## What it does well, and badly

- **Against the seasonal naive:** better on nGMASE (0.9471) and on nWQL (0.6646 against 0.8876).
  Per configuration its MASE is below the seasonal naive's in 52 of 97 and its WQL in 84 of 97;
  against s6-cast-pico, our TinyCast port has a lower MASE in 86 configurations and a lower WQL
  in 87.
- **By term** (MASE): 0.860 short, 1.046 medium, 1.105 long, against 0.756, 0.806 and 0.848 for our
  TinyCast port. Medium and long horizons are worse than the seasonal naive.
- **By domain** (MASE / WQL): Econ/Fin 1.018 / 0.901, Energy 1.080 / 0.800, Healthcare
  0.760 / 0.614, Nature 0.862 / 0.467, Sales 0.753 / 0.457, Transport 0.833 / 0.644,
  Web/CloudOps 0.983 / 0.662. Healthcare is the only domain where its WQL is below TinyCast's
  (0.614 against 0.639 for our port); it is worse than the seasonal naive on point forecasts in
  Econ/Fin and Energy.
- **Worst configurations:** `solar/10T/long` and `solar/10T/medium`, MASE ratios 2.729 and 2.282,
  against 0.973 and 0.955 for our TinyCast port.
- **Not measured by the run:** the latency, the mean number of exits used on GIFT-Eval (the
  calibration gives 3.64 of 4 on its synthetic windows) and the gap to Timestral-5M. The latency
  was measured later, with its second version's gate (18/11/2026, one synthetic series with a
  2,048-point history, one core of an Apple M5, median of 1,000 forecasts of 64 steps at full
  depth): 799 µs natively and 1,545 µs in WASM.

## Architecture

The s6 encoder on patches of 32 points (`skycmd-cast` GGUF tag): 2 weight-shared blocks looped
twice, an exit after each loop, d 96, 2 heads, a context of 64 patches (2,048 points) and a
decoder for 2 patches of horizon (64 points), nine quantile outputs, ternary weights. Early exit:
a pass stops when an exit's median moves by less than `exit_tolerance` from the previous exit's;
the file stores 0.012839192, chosen by a calibration on 2,000 synthetic windows (α 0.05, δ 0.02).
Without Parallax or XSA, which the pre-registered recipe did not adopt: 369,792 parameters instead
of the 388,320 of the pre-registration.

## Limitations

- **Failed its gate.** Recommended for no use: TinyCast (Apache-2.0, 146,505 parameters) is better
  in 86 of the 97 configurations on MASE.
- One run, one seed.
- The horizon of one pass is 64 points; longer horizons roll out, and the medium and long terms are
  where it does worst.

## Usage

Runs in the skycmd engine only (`sky-infer`, natively and in WASM): in the browser,
`new Forecaster(bytes)` then `forecast(history, horizon, flip)`, which returns
`horizon × 9` quantiles in the units of the history.

## Training

From scratch in Rust (mlx-rs) on an Apple M5, one run of 480 working minutes (250,092 steps),
in a braked mode. AdamW at 3e-3 (chosen by a pre-registered pilot), batch 256, QAT over the
last 20 %, EMA 0.999 (the EMA weights are exported: validation pinball 0.0431, against 0.0442 for
the last weights), seed 1, TinyCast as a teacher on 50 % of the windows with a weight annealed
from 0.5 to 0. Data: synthetic series (Gaussian-process kernels, pulse trains, trend plus seasons
plus noise, SARIMA), the state variables of the skycmd games, and permissive real series, none of
which is in GIFT-Eval or in Chronos' zero-shot benchmark: the Monash archive (London smart meters,
wind farms, pedestrian counts, rideshare; CC-BY-4.0), Wikimedia pageviews (CC0-1.0) and NOAA
USHCN daily (public domain). GIFT-Eval-Pretrain was not used (no declared license); GIFT-Eval is
only evaluated on. Licenses:
[DATA_LICENSES.md](https://github.com/maxgfr/skycmd/blob/main/DATA_LICENSES.md).

## Files

- `s6-cast-pico.gguf`: ternary weights (the EMA weights) and the exit tolerance of the calibration.
  142,912 bytes, sha256 `ff4618604e0c0076180a2846413970393dfe5d68c8ee4add342d33fdd36403f1`.
