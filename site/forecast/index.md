---
title: Forecast
outline: [2, 2]
---

# Forecast: time-series models in your browser

Pick a series and every forecaster below runs on this page, with nothing sent to a server. Each one returns quantiles; the chart shows each model's median and its 10–90 % band.

- **s6-cast** is ours: the s6 encoder with patch inputs, nine deciles, ternary weights and an exit after each loop, trained on synthetic and permissive real series. **Its first run failed its gate and is not published**, so it does not run here. The gate, written before training ([pre-registration](https://github.com/maxgfr/skycmd/blob/main/docs/s6-cast-prereg.md)), was to beat the training-free FLAIR on GIFT-Eval (nGMASE ≤ 0.838 and nWQL ≤ 0.587), with the goal of beating TinyCast. s6-cast-pico (369,792 parameters) scored nGMASE 0.9471 and nWQL 0.6646 ([results](/results/#s6-cast-pico-failed)).
- **TinyCast** (RAWS Labs, 146,505 parameters, Apache-2.0) is ported to Rust and runs in the same wasm module. Our port matches the official PyTorch forecasts to about 1e-5 of the series range.
- **Timestral-5M** (Mistral AI, modified MIT license) runs in onnxruntime-web, exported from the official weights. Its 23 MB file is downloaded only when you tick it.
- **Seasonal naive** repeats the last season. It is the floor GIFT-Eval normalizes every score by.

**Backtest** hides the last steps of the series from the models and scores their forecasts: MASE (below 1 beats the seasonal naive) and the share of the hidden points inside the 10–90 % band (about 80 % when the band is calibrated).

<ClientOnly>
  <ForecastView />
</ClientOnly>

## Where the numbers come from

On GIFT-Eval (97 configurations, scores divided by the seasonal naive, lower is better), TinyCast reports nGMASE 0.774 and nWQL 0.545. Our evaluation harness uses the official protocol, gift-eval's datasets and gluonts' metrics ([spec](https://github.com/maxgfr/skycmd/blob/main/docs/SPEC-S6-CAST.md)); on it our TinyCast port scores 0.786 / 0.557, our seasonal naive 0.99986 on nGMASE, and s6-cast-pico 0.9471 / 0.6646 (0.9099 / 0.6417 with flip).

The synthetic series of the picker come from s6-cast's generators: Gaussian-process kernels, pulse trains, trend plus seasons plus noise, SARIMA. The real series are permissive and credited under the chart: Monash forecasting archive (CC-BY-4.0), Wikimedia pageviews (CC0) and NOAA USHCN (public domain).

Licenses: [TinyCast NOTICE](/models/ref/TINYCAST-NOTICE.txt), [Timestral-5M license](/models/ref/TIMESTRAL-LICENSE.txt). Timestral's license excludes companies above $20M monthly revenue without a commercial license from Mistral AI.
