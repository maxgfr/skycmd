---
title: Forecast
outline: [2, 2]
---

# Forecast: time-series models in your browser

Pick a series and every forecaster below runs on this page, with nothing sent to a server. Each one returns quantiles; the chart shows each model's median and its 10–90 % band.

- **s6-cast** is ours: the s6 encoder with patch inputs, nine deciles, ternary weights and an exit after each loop, pretrained on synthetic series. **It is not trained yet.** It appears here once its first run passes the gate written before training ([pre-registration](https://github.com/maxgfr/skycmd/blob/main/docs/s6-cast-prereg.md)): beat the training-free reference on GIFT-Eval, with the goal of beating TinyCast.
- **TinyCast** (RAWS Labs, 146,505 parameters, Apache-2.0) is ported to Rust and runs in the same wasm module. Our port matches the official PyTorch forecasts to about 1e-5 of the series range.
- **Timestral-5M** (Mistral AI, modified MIT license) runs in onnxruntime-web, exported from the official weights. Its 23 MB file is downloaded only when you tick it.
- **Seasonal naive** repeats the last season. It is the floor GIFT-Eval normalizes every score by.

**Backtest** hides the last steps of the series from the models and scores their forecasts: MASE (below 1 beats the seasonal naive) and the share of the hidden points inside the 10–90 % band (about 80 % when the band is calibrated).

<ClientOnly>
  <ForecastView />
</ClientOnly>

## Where the numbers come from

On GIFT-Eval (97 configurations, scores divided by the seasonal naive, lower is better), TinyCast reports nGMASE 0.774 and nWQL 0.545. Our evaluation harness uses the official protocol, gift-eval's datasets and gluonts' metrics, so s6-cast's score will be comparable ([spec](https://github.com/maxgfr/skycmd/blob/main/docs/SPEC-S6-CAST.md)).

The synthetic series come from s6-cast's own generators: Gaussian-process kernels, pulse trains, trend plus seasons plus noise, SARIMA. The real series are permissive and credited under the chart: Monash forecasting archive (CC-BY-4.0), Wikimedia pageviews (CC0) and NOAA USHCN (public domain).

Licenses: [TinyCast NOTICE](/models/ref/TINYCAST-NOTICE.txt), [Timestral-5M license](/models/ref/TIMESTRAL-LICENSE.txt). Timestral's license excludes companies above $20M monthly revenue without a commercial license from Mistral AI.
