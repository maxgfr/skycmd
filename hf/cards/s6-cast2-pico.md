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

# s6-cast2-pico

> **Failed its pre-registered gate: published as a result, not for use.** Its gate G1
> ([`docs/s6-cast2-prereg.md`](https://github.com/maxgfr/skycmd/blob/main/docs/s6-cast2-prereg.md)
> §8) asked, on GIFT-Eval without flip, for nGMASE ≤ 0.838 and nWQL ≤ 0.587, the bars of the
> training-free FLAIR. s6-cast2-pico scored **nGMASE 0.8847** and **nWQL 0.6190**: **G1 fails** on
> both metrics, and with flip too (0.8692 / 0.6071). **G2 holds:** its MASE ratio to the seasonal
> naive is under 1.00 at each horizon term (short 0.833, medium 0.928, long 0.988). G3, on
> Chronos-ZS, was dropped before training because the harness did not reproduce the published
> TinyCast (amendment 2), so the verdict rests on the benchmark that shaped the design; Chronos-ZS
> is reported. It is better than its first version, s6-cast-pico (0.9471 / 0.6646), and stays
> behind our port of TinyCast (0.7918 / 0.5622 without flip). By its pre-registration the s6-cast
> line stops here and the forecaster the project offers is the TinyCast port. The file is
> published so that the result can be checked and run (owner's decision of 15/11/2026); the
> verdict does not change. Numbers:
> [BENCH.md, section s6-cast2-pico](https://github.com/maxgfr/skycmd/blob/main/BENCH.md#s6-cast2-pico-s6-cast-v2-a-forecaster-with-a-periodicity-front-end-pre-registered-gate-failed)
> and [`runs/s6-cast2-pico/GATE.md`](https://github.com/maxgfr/skycmd/blob/main/runs/s6-cast2-pico/GATE.md).

**A 149 KB ternary probabilistic forecaster: s6-cast-pico's encoder with a periodicity front
end.** 377,568 parameters, 148,512 bytes. Given up to 2,048 past points it returns nine quantiles
(the deciles) for each of the next 64 steps, and longer horizons by rolling out, with an exit
after each loop. What v2 adds to v1, from a design study of v1's losses: TinyCast's
zero-parameter period detector runs on every context, and the sine and cosine phases of up to 4
detected periods go into each patch; a value-space seasonal draft of the future (the value one
period earlier, and the mean of past points at the same phase) is projected into each decoder
query; training data at the frequency bands GIFT-Eval tests (5-minute USCRN series); and 720
working minutes of training instead of 480.

- **Demo:** [huggingface.co/spaces/{{USER}}/s6-cast2-pico](https://huggingface.co/spaces/{{USER}}/s6-cast2-pico)
  (next to TinyCast and the seasonal naive, in your browser) and the
  [Forecast page](https://maxgfr.github.io/skycmd/forecast/) of the project site (next to v1,
  TinyCast, Timestral-5M and the seasonal naive); it starts unticked
- **Code:** [github.com/maxgfr/skycmd](https://github.com/maxgfr/skycmd) (100 % Rust)
- **Spec:** [docs/SPEC-S6-CAST.md](https://github.com/maxgfr/skycmd/blob/main/docs/SPEC-S6-CAST.md)
{{LINKS}}

**Disclosure** (§10 of the pre-registration). v2 was designed after reading v1's GIFT-Eval results
on the same 97 configurations, window by window: where it lost, by band, term, domain,
seasonality, context length and horizon. GIFT-Eval is therefore partly in-sample for v2's design
choices, though not for its weights: no GIFT-Eval series is trained on. TinyCast's authors make
the same disclosure about their own design and treat Chronos-ZS as the cleaner test.

## The gate, as read

| term | measured | bar | verdict |
|---|---|---|---|
| G1, GIFT-Eval | nGMASE **0.8847**, nWQL **0.6190** | nGMASE ≤ 0.838 and nWQL ≤ 0.587 | fails |
| G2, GIFT-Eval | MASE ratio short **0.833**, medium **0.928**, long **0.988** | < 1.00 at each term | holds |
| G3, Chronos-ZS | dropped before training (amendment 2) | | reported below |

## Results

GIFT-Eval, 97 configurations, 371,330 windows, the official protocol (gluonts `evaluate_model`,
the 9 deciles); every metric is divided by the seasonal naive and averaged geometrically, lower is
better. 2,017 windows in 12 configurations have no observation in the model's last 2,048 points
and are forecast by the seasonal naive, as for v1.

| model | nGMASE | nWQL | nMSIS |
|---|---:|---:|---:|
| **s6-cast2-pico, without flip** (the gated model) | **0.8847** | **0.6190** | 0.6848 |
| s6-cast2-pico, with flip | 0.8692 | 0.6071 | 0.6735 |
| s6-cast-pico (v1), without flip | 0.9471 | 0.6646 | 0.7190 |
| s6-cast-pico (v1), with flip | 0.9099 | 0.6417 | 0.6860 |
| bar of G1: FLAIR, training-free (TinyCast paper) | 0.838 | 0.587 | |
| TinyCast, our port, with flip | 0.7856 | 0.5570 | 0.5781 |
| TinyCast, our port, without flip | 0.7918 | 0.5622 | 0.5865 |
| seasonal naive, our harness | 0.9999 | 0.8876 | 2.6778 |

By term, MASE / WQL ratio to the seasonal naive:

| term | configs | v1 | **v2** | TinyCast port |
|---|---:|---|---|---|
| short | 55 | 0.860 / 0.638 | **0.833 / 0.610** | 0.756 / 0.571 |
| medium | 21 | 1.046 / 0.696 | **0.928 / 0.624** | 0.806 / 0.534 |
| long | 21 | 1.105 / 0.706 | **0.988 / 0.637** | 0.848 / 0.546 |

By domain, MASE / WQL:

| domain | configs | v1 | **v2** | TinyCast port |
|---|---:|---|---|---|
| Econ/Fin | 6 | 1.018 / 0.901 | **0.984 / 0.876** | 0.894 / 0.841 |
| Energy | 32 | 1.080 / 0.800 | **1.003 / 0.746** | 0.882 / 0.661 |
| Healthcare | 5 | 0.760 / 0.614 | **0.825 / 0.678** | 0.735 / 0.639 |
| Nature | 15 | 0.862 / 0.467 | **0.825 / 0.414** | 0.757 / 0.358 |
| Sales | 4 | 0.753 / 0.457 | **0.749 / 0.455** | 0.734 / 0.445 |
| Transport | 15 | 0.833 / 0.644 | **0.772 / 0.603** | 0.669 / 0.510 |
| Web/CloudOps | 20 | 0.983 / 0.662 | **0.861 / 0.593** | 0.750 / 0.562 |

Chronos-ZS, reported, not gated (27 tasks, relative to fev's seasonal naive; the harness check did
not validate: our port with flip reads 0.8576 / 0.6822 where 0.880 / 0.722 are published):

| model | relative MASE | relative WQL |
|---|---:|---:|
| **s6-cast2-pico, without flip** | **0.9014** | **0.7143** |
| TinyCast, our port, without flip | 0.8598 | 0.6928 |
| TinyCast, our port, with flip | 0.8576 | 0.6822 |
| seasonal naive | 1.0000 | 1.0000 |

## What holds, and what fails

- **Fails: G1.** Both GIFT-Eval bars are missed, with and without flip. The gap to our TinyCast
  port without flip is +0.0929 nGMASE and +0.0568 nWQL; to the published TinyCast (0.774 / 0.545),
  +0.1107 and +0.0740.
- **Holds: G2.** Its point forecasts are better than the seasonal naive's at every term; v1 was
  worse than it at the medium and long terms (1.046 and 1.105).
- **Better than v1** on both metrics in every term, and in six of the seven domains: it closes
  0.365 of v1's gap to our TinyCast port on MASE and 0.402 on WQL. It is worse than v1 in
  Healthcare and in the weekly and monthly bands.
- **Behind the port everywhere on MASE:** in every term, band and domain. Its MASE is above the
  seasonal naive's in 31 of the 97 configurations, the worst `ett2/W/short` (1.626),
  `us_births/M/short` (1.572) and `m4_hourly/H/short` (1.513).
- **On Chronos-ZS**, which its design never read, it beats the seasonal naive and stays behind the
  port on both metrics.
- **Per window** (reported, not gated; computed on 18/11/2026 from forecast dumps rebuilt with
  `sky-cast-eval run` after they had been removed to free disk space, and checked to give back the
  published scores exactly). Coverage of the 80% interval: 0.800 at the short term, 0.578 at the
  medium and 0.522 at the long (the port 0.819, 0.643, 0.547), so its intervals are too narrow
  beyond the short term. Its scaled error is 1.07 to 1.27 times the port's at every tenth of the
  horizon, and 1.12 to 1.20 times it whether or not a period is detected in the history: the loss
  to the port is spread over every lead time and every kind of window.

## Size and speed

Measured on 18/11/2026 on an Apple M5, one core, after the training had ended: one synthetic
series (`tsi`, seed 7) with a 2,048-point history, the median and p95 of 1,000 forecasts, natively
(`sky-cast --bench`) and in WebAssembly (the `Forecaster` of `sky-web` under Node, SIMD128).

| model | file | native, 64 steps | WASM, 64 steps | native, 720 steps | WASM, 720 steps |
|---|---:|---:|---:|---:|---:|
| **s6-cast2-pico, full depth** | **148,512 bytes** | **838 µs** (p95 906) | **1,636 µs** (p95 1,955) | 10,367 µs | 19,346 µs |
| s6-cast2-pico, calibrated exits | 148,512 bytes | 842 µs (p95 904) | 1,528 µs (p95 1,650) | | 18,232 µs |
| s6-cast-pico (v1), full depth | 142,912 bytes | 799 µs (p95 870) | 1,545 µs (p95 1,908) | 9,826 µs | 17,922 µs |
| TinyCast, our port | 595,840 bytes | 21,531 µs (p95 23,189) | 39,998 µs (p95 41,530) | 164,760 µs | 309,243 µs |

On GIFT-Eval the file used 3.334 of its 4 exits on average. The sentence of the gate report: v2 at
838 µs and 148,512 bytes against the port's 21,531 µs and 595,840 bytes, nGMASE 0.8847 against
0.7856: the port forecasts better; take it unless size or speed decide. These latencies are one
series on one machine.

## Architecture

The s6 encoder on patches of 32 points (`skycmd-cast` GGUF tag): 2 weight-shared blocks looped
twice, an exit after each loop, d 96, 2 heads, a context of 64 patches (2,048 points), a decoder
for 2 patches of horizon (64 points) with 2 blocks, nine quantile outputs, ternary weights. The
front end has no parameter of its own beyond two projections: `tinycast::significant_periods`
(K = 4, α = 0.05) on TinyCast's prepared window of the context; 8 phase channels appended to each
patch (`patch_proj` of width 2 × 32 + 8); a draft of 72 values per query patch (the lag copy, the
phase-binned mean and the phase channels) through `draft_proj` (Linear 72 → 96, ternary), added
to the decoder query. GGUF keys `skycmd-cast.periods` (4) and `skycmd-cast.draft` (true); a v1
file has neither. Early exit: a pass stops when an exit's median moves by less than
`exit_tolerance` from the previous exit's; the file stores 0.012839192, from a calibration on
2,000 synthetic windows (α 0.05, δ 0.02).

## How to run it outside the browser

The file runs in the skycmd engine only (`sky-infer`, natively and in WASM). From a clone of
[the repository](https://github.com/maxgfr/skycmd):

```bash
# a forecast as JSON (levels, quantiles [horizon][9], exits used); one number per line, NaN = missing
cargo run --release -p sky-infer --bin sky-cast -- s6-cast2-pico.gguf --file series.txt --horizon 64
# the same on a synthetic series, with the sign-flip average, at full depth
cargo run --release -p sky-infer --bin sky-cast -- s6-cast2-pico.gguf --synth tsi --seed 7 --len 2048 --flip --full-depth
# the latency of this card: median, p90 and p95 of 1,000 forecasts
cargo run --release -p sky-infer --bin sky-cast -- s6-cast2-pico.gguf --synth tsi --seed 7 --len 2048 --horizon 64 --full-depth --bench 1000
```

GIFT-Eval with the official protocol (`scripts/gift-eval.py` runs under `uv`; the scores land in
`data/gift-eval-out/s6-cast2-pico/summary.json`):

```bash
uv run --python 3.12 scripts/gift-eval.py download --data data/gift-eval-src
uv run --python 3.12 scripts/gift-eval.py export   --data data/gift-eval-src --out data/gift-eval
cargo run --release -p sky-infer --bin sky-cast-eval -- run --model s6-cast2-pico.gguf \
  --windows data/gift-eval --out data/gift-eval-out/s6-cast2-pico
uv run --python 3.12 scripts/gift-eval.py score --data data/gift-eval-src \
  --forecasts data/gift-eval-out/s6-cast2-pico --name s6-cast2-pico
```

In JavaScript, `sky-web`'s `new Forecaster(bytes)` then `forecast(history, horizon, flip)` returns
`horizon × 9` quantiles in the units of the history.

## Training

From scratch in Rust (mlx-rs) on an Apple M5: one run of 720 working minutes, **377,082 steps**,
seed 1, no restart, under an automatic brake (duty 0.95 with 55 minutes of work and 5 of rest,
duty 0.5 with 45 and 15, and cooling pauses). AdamW at 3e-3, batch 256, QAT over the last 20 %,
EMA 0.999; the EMA weights are exported (final ternary validation pinball 0.0435, against 0.0443
for the last weights). A pre-registered pilot of 3,000 steps per arm kept the TinyCast teacher
(validation pinball 0.045361 against 0.045963 without): TinyCast's forecasts on training windows
only, on 50 % of the windows, with a weight annealed from 0.5 to 0. Windows seen: 21,295,763
synthetic, 25,092,584 real, 6,420,364 synthetic stand-ins for bands with no real series,
43,724,281 distilled.

**Data and licences** ([DATA_LICENSES.md](https://github.com/maxgfr/skycmd/blob/main/DATA_LICENSES.md)):

- **Synthetic series** from the project's Rust generators (Gaussian-process kernels, pulse trains,
  trend plus seasons plus noise, SARIMA), with six added periods for 5- and 10-minute cycles, and
  sign flip and time reversal at 0.5. Generated by code, not by a model.
- **v1's real series**: the Monash archive (London smart meters, wind farms, pedestrian counts,
  rideshare; CC-BY-4.0), Wikimedia pageviews (CC0-1.0), NOAA USHCN daily (public domain) and the
  state variables of the skycmd games (Apache-2.0).
- **New in v2: USCRN sub-hourly** (NOAA NCEI, U.S. Climate Reference Network, 5-minute sensors;
  public domain, a U.S. Government work): the 2025 file of each of 159 stations, 1,011 series,
  8,282,112 points. A leakage check hashed every run of 32 informative values against every
  GIFT-Eval and Chronos-ZS series: no match, no series dropped.
- **The teacher: TinyCast** (RAWS Labs, [raws-labs/tinycast](https://huggingface.co/raws-labs/tinycast);
  Apache-2.0, code and weights), run through our Rust port on training windows only.
- Real windows are drawn with GIFT-Eval's band weights (hourly 31, sub-hourly 30, daily 15, weekly
  8, monthly and coarser 7, second-level 6); the second-level, weekly and monthly bands had no
  real series and got synthetic stand-ins at their periods. No series of GIFT-Eval or Chronos-ZS,
  nor of any dataset they draw from, is trained on; GIFT-Eval-Pretrain was not used. GIFT-Eval
  (Apache-2.0, released for research use) and the Chronos-ZS tasks (Apache-2.0, autogluon/fev) are
  only evaluated on.

## Limitations

- **Failed its gate.** Recommended for no use: our TinyCast port (Apache-2.0, 146,505 parameters)
  forecasts better on GIFT-Eval in every term, band and domain on MASE, and on Chronos-ZS.
- **GIFT-Eval is partly in-sample for its design** (the disclosure above); the gate it failed was
  read on that benchmark.
- One run, one seed. The evaluation chain stopped twice on a full disk: v2's GIFT-Eval forecasts
  were started again from their beginning, and the Chronos-ZS step and the gate were run again;
  v2's GIFT-Eval scoring ran once, to its end, and no number changed.
- The horizon of one pass is 64 points; longer horizons roll out, and the long term is where it
  is closest to the seasonal naive (MASE ratio 0.988).
- Its latency was measured on one series and one machine.

## Files

- `s6-cast2-pico.gguf`: ternary weights (the EMA weights) and the exit tolerance of the
  calibration, the file the gate read. 148,512 bytes, sha256
  `30cbee8e846bc47d0876d9d77e3460278e58025cf01b6a5390b79b999215959f`.
