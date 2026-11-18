---
title: Forecast
outline: [2, 2]
---

# Forecast: time-series models in your browser

Pick a series and every forecaster below runs on this page, with nothing sent to a server. Each one returns quantiles; the chart shows each model's median and its 10–90 % band.

- **s6-cast** is ours: the s6 encoder with patch inputs, nine deciles, ternary weights and an exit after each loop, trained on synthetic and permissive real series. **Both of its runs failed their gates; their files are published as results (owner's decision of 15/11/2026)**, not for use: they run here, marked as failed and unticked by default. Each gate was written before training and asked it to beat the training-free FLAIR on GIFT-Eval (nGMASE ≤ 0.838 and nWQL ≤ 0.587).
  - **s6-cast2-pico** (v2, 377,568 parameters, 149 KB) **failed its gate**: nGMASE 0.8847 and nWQL 0.6190. It adds a computed-periodicity front end and sub-hourly real series to v1; its MASE is under the seasonal naive's at every horizon term (the second part of its gate, which holds), and it stays behind TinyCast on both metrics ([pre-registration](https://github.com/maxgfr/skycmd/blob/main/docs/s6-cast2-prereg.md), [results](/results/#s6-cast2-pico-failed)). By its pre-registration the s6-cast line stops there: the forecaster to use on this page is TinyCast.
  - **s6-cast-pico** (v1, 369,792 parameters, 143 KB) failed its gate: nGMASE 0.9471 and nWQL 0.6646, with the goal of beating TinyCast missed too ([pre-registration](https://github.com/maxgfr/skycmd/blob/main/docs/s6-cast-prereg.md), [results](/results/#s6-cast-pico-failed)).
- **TinyCast** (RAWS Labs, 146,505 parameters, Apache-2.0) is ported to Rust and runs in the same wasm module. Our port matches the official PyTorch forecasts to about 1e-5 of the series range.
- **Timestral-5M** (Mistral AI, modified MIT license) runs in onnxruntime-web, exported from the official weights. Its 23.0 MB file is downloaded only when you tick it.
- **Seasonal naive** repeats the last season. It is the floor GIFT-Eval normalizes every score by.

**Backtest** hides the last steps of the series from the models and scores their forecasts: MASE (below 1 beats the seasonal naive) and the share of the hidden points inside the 10–90 % band (about 80 % when the band is calibrated).

<ClientOnly>
  <ForecastView />
</ClientOnly>

## Where the numbers come from

On GIFT-Eval (97 configurations, scores divided by the seasonal naive, lower is better), TinyCast reports nGMASE 0.774 and nWQL 0.545. Our evaluation harness uses the official protocol, gift-eval's datasets and gluonts' metrics ([spec](https://github.com/maxgfr/skycmd/blob/main/docs/SPEC-S6-CAST.md)); on it our TinyCast port scores 0.786 / 0.557, our seasonal naive 0.99986 on nGMASE, and our TinyCast port without flip 0.7918 / 0.5622; s6-cast2-pico 0.8847 / 0.6190 (0.8692 / 0.6071 with flip) and s6-cast-pico 0.9471 / 0.6646 (0.9099 / 0.6417 with flip). Measured on one synthetic series of 2,048 points, on one core of an Apple M5, s6-cast2-pico forecasts 64 steps in a median 838 µs natively and 1,636 µs in WASM, our TinyCast port in 21,531 µs and 39,998 µs ([`runs/s6-cast2-pico/latency/`](https://github.com/maxgfr/skycmd/tree/main/runs/s6-cast2-pico/latency)); the latency column of the demo above is measured in your browser.

The synthetic series of the picker come from s6-cast's generators: Gaussian-process kernels, pulse trains, trend plus seasons plus noise, SARIMA. The real series are permissive and credited under the chart: Monash forecasting archive (CC-BY-4.0), Wikimedia pageviews (CC0) and NOAA USHCN (public domain).

Licenses: [TinyCast NOTICE](/models/ref/TINYCAST-NOTICE.txt), [Timestral-5M license](/models/ref/TIMESTRAL-LICENSE.txt). Timestral's license excludes companies above $20M monthly revenue without a commercial license from Mistral AI.
