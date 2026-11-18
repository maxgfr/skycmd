# s6-cast2-pico: the gate of docs/s6-cast2-prereg.md

**Verdict: FAIL.**

G3 was dropped before training (amendment 2: the Chronos-ZS harness was not validated): G1 and G2 were read on the benchmark that shaped the design, and Chronos-ZS is reported, not gated.

## Gate (ternary GGUF, without flip, full harness)

| item | measured | holds |
|---|---|---|
| G1 | nGMASE 0.8847 <= 0.838 and nWQL 0.6190 <= 0.587 | False |
| G2 | MASE ratio < 1.00 at each term: short 0.833, medium 0.928, long 0.988 | True |
| G3 (reported) | v2 relative MASE 0.9014 <= 1.083 x 0.8598 (the port without flip) and relative WQL 0.7143 <= 1.083 x 0.6928 | reported: True |

*Disclosure (docs/s6-cast2-prereg.md §10).* v2 was designed after reading v1's GIFT-Eval results on the same 97 configurations, window by window: where it lost, by band, term, domain, seasonality, context length and horizon. GIFT-Eval is therefore partly in-sample for v2's design choices, though not for its weights: no GIFT-Eval series is trained on. TinyCast's authors make the same disclosure about their own design and treat Chronos-ZS as the cleaner test. G3 exists for this reason. Every report of v2 states this paragraph.

G3 was dropped before training (amendment 2: the Chronos-ZS harness was not validated): G1 and G2 were read on the benchmark that shaped the design, and Chronos-ZS is reported, not gated.

## Chronos-ZS (27 tasks, relative to fev's seasonal naive)

Harness check (the port with flip within 0.02 of 0.880 / 0.722): not validated.

| model | relative MASE | relative WQL | tasks |
|---|---|---|---|
| v2, without flip | 0.9014 | 0.7143 | 27 |
| TinyCast port, without flip | 0.8598 | 0.6928 | 27 |
| TinyCast port, with flip (harness check) | 0.8576 | 0.6822 | 27 |
| seasonal naive | 1.0000 | 1.0000 | 27 |

*Disclosure (docs/s6-cast2-prereg.md §10).* v2 was designed after reading v1's GIFT-Eval results on the same 97 configurations, window by window: where it lost, by band, term, domain, seasonality, context length and horizon. GIFT-Eval is therefore partly in-sample for v2's design choices, though not for its weights: no GIFT-Eval series is trained on. TinyCast's authors make the same disclosure about their own design and treat Chronos-ZS as the cleaner test. G3 exists for this reason. Every report of v2 states this paragraph.

## GIFT-Eval: nGMASE, nWQL, nMSIS

| model | nGMASE | nWQL | nMSIS | configs |
|---|---|---|---|---|
| v1 | 0.9471 | 0.6646 | 0.7190 | 97 |
| v1 flip | 0.9099 | 0.6417 | 0.6860 | 97 |
| v2 | 0.8847 | 0.6190 | 0.6848 | 97 |
| v2 flip | 0.8692 | 0.6071 | 0.6735 | 97 |
| TinyCast port (flip) | 0.7856 | 0.5570 | 0.5781 | 97 |
| TinyCast port, no flip | 0.7918 | 0.5622 | 0.5865 | 97 |
| seasonal naive | 0.9999 | 0.8876 | 2.6778 | 97 |

*Disclosure (docs/s6-cast2-prereg.md §10).* v2 was designed after reading v1's GIFT-Eval results on the same 97 configurations, window by window: where it lost, by band, term, domain, seasonality, context length and horizon. GIFT-Eval is therefore partly in-sample for v2's design choices, though not for its weights: no GIFT-Eval series is trained on. TinyCast's authors make the same disclosure about their own design and treat Chronos-ZS as the cleaner test. G3 exists for this reason. Every report of v2 states this paragraph.

## Gap of v2 (without flip) to TinyCast (not gated)

| reference | nGMASE gap | nWQL gap |
|---|---|---|
| TinyCast port, with flip | 0.0991 | 0.0620 |
| TinyCast port, without flip | 0.0929 | 0.0568 |
| published TinyCast (0.774 / 0.545) | 0.1107 | 0.0740 |

*Disclosure (docs/s6-cast2-prereg.md §10).* v2 was designed after reading v1's GIFT-Eval results on the same 97 configurations, window by window: where it lost, by band, term, domain, seasonality, context length and horizon. GIFT-Eval is therefore partly in-sample for v2's design choices, though not for its weights: no GIFT-Eval series is trained on. TinyCast's authors make the same disclosure about their own design and treat Chronos-ZS as the cleaner test. G3 exists for this reason. Every report of v2 states this paragraph.

## GIFT-Eval by term (MASE / WQL ratio to the seasonal naive)

| term | configs | v1 | v2 | TinyCast port |
|---|---|---|---|---|
| short | 55 | 0.860 / 0.638 | 0.833 / 0.610 | 0.756 / 0.571 |
| medium | 21 | 1.046 / 0.696 | 0.928 / 0.624 | 0.806 / 0.534 |
| long | 21 | 1.105 / 0.706 | 0.988 / 0.637 | 0.848 / 0.546 |

*Disclosure (docs/s6-cast2-prereg.md §10).* v2 was designed after reading v1's GIFT-Eval results on the same 97 configurations, window by window: where it lost, by band, term, domain, seasonality, context length and horizon. GIFT-Eval is therefore partly in-sample for v2's design choices, though not for its weights: no GIFT-Eval series is trained on. TinyCast's authors make the same disclosure about their own design and treat Chronos-ZS as the cleaner test. G3 exists for this reason. Every report of v2 states this paragraph.

## GIFT-Eval by band (MASE / WQL ratio to the seasonal naive)

| band | configs | v1 | v2 | TinyCast port |
|---|---|---|---|---|
| second | 6 | 1.298 / 0.896 | 0.926 / 0.692 | 0.909 / 0.702 |
| sub-hourly | 30 | 1.046 / 0.735 | 0.936 / 0.653 | 0.794 / 0.571 |
| hourly | 31 | 0.914 / 0.606 | 0.870 / 0.569 | 0.754 / 0.480 |
| daily | 15 | 0.782 / 0.494 | 0.770 / 0.490 | 0.711 / 0.465 |
| weekly | 8 | 0.908 / 0.729 | 0.937 / 0.737 | 0.858 / 0.732 |
| monthly and coarser | 7 | 0.878 / 0.850 | 0.907 / 0.878 | 0.890 / 0.853 |

*Disclosure (docs/s6-cast2-prereg.md §10).* v2 was designed after reading v1's GIFT-Eval results on the same 97 configurations, window by window: where it lost, by band, term, domain, seasonality, context length and horizon. GIFT-Eval is therefore partly in-sample for v2's design choices, though not for its weights: no GIFT-Eval series is trained on. TinyCast's authors make the same disclosure about their own design and treat Chronos-ZS as the cleaner test. G3 exists for this reason. Every report of v2 states this paragraph.

## GIFT-Eval by domain (MASE / WQL ratio to the seasonal naive)

| domain | configs | v1 | v2 | TinyCast port |
|---|---|---|---|---|
| Econ/Fin | 6 | 1.018 / 0.901 | 0.984 / 0.876 | 0.894 / 0.841 |
| Energy | 32 | 1.080 / 0.800 | 1.003 / 0.746 | 0.882 / 0.661 |
| Healthcare | 5 | 0.760 / 0.614 | 0.825 / 0.678 | 0.735 / 0.639 |
| Nature | 15 | 0.862 / 0.467 | 0.825 / 0.414 | 0.757 / 0.358 |
| Sales | 4 | 0.753 / 0.457 | 0.749 / 0.455 | 0.734 / 0.445 |
| Transport | 15 | 0.833 / 0.644 | 0.772 / 0.603 | 0.669 / 0.510 |
| Web/CloudOps | 20 | 0.983 / 0.662 | 0.861 / 0.593 | 0.750 / 0.562 |

*Disclosure (docs/s6-cast2-prereg.md §10).* v2 was designed after reading v1's GIFT-Eval results on the same 97 configurations, window by window: where it lost, by band, term, domain, seasonality, context length and horizon. GIFT-Eval is therefore partly in-sample for v2's design choices, though not for its weights: no GIFT-Eval series is trained on. TinyCast's authors make the same disclosure about their own design and treat Chronos-ZS as the cleaner test. G3 exists for this reason. Every report of v2 states this paragraph.

## Configurations where v2 is above the seasonal naive (MASE)

| configuration | MASE ratio |
|---|---|
| bitbrains_fast_storage/5T/long | 1.422 |
| bitbrains_fast_storage/5T/medium | 1.365 |
| bitbrains_fast_storage/H/short | 1.037 |
| bitbrains_rnd/5T/long | 1.376 |
| bitbrains_rnd/5T/medium | 1.220 |
| bitbrains_rnd/5T/short | 1.228 |
| bitbrains_rnd/H/short | 1.023 |
| bizitobs_application/10S/medium | 1.001 |
| bizitobs_l2c/5T/long | 1.099 |
| bizitobs_service/10S/long | 1.039 |
| electricity/15T/long | 1.092 |
| electricity/15T/medium | 1.002 |
| electricity/H/long | 1.050 |
| electricity/H/medium | 1.013 |
| ett1/15T/long | 1.086 |
| ett1/15T/medium | 1.033 |
| ett2/15T/long | 1.014 |
| ett2/D/short | 1.023 |
| ett2/H/long | 1.086 |
| ett2/W/short | 1.626 |
| jena_weather/H/long | 1.049 |
| jena_weather/H/medium | 1.048 |
| m4_daily/D/short | 1.155 |
| m4_hourly/H/short | 1.513 |
| solar/10T/long | 1.335 |
| solar/10T/medium | 1.152 |
| solar/H/long | 1.269 |
| solar/H/medium | 1.312 |
| solar/H/short | 1.165 |
| us_births/M/short | 1.572 |
| us_births/W/short | 1.012 |

*Disclosure (docs/s6-cast2-prereg.md §10).* v2 was designed after reading v1's GIFT-Eval results on the same 97 configurations, window by window: where it lost, by band, term, domain, seasonality, context length and horizon. GIFT-Eval is therefore partly in-sample for v2's design choices, though not for its weights: no GIFT-Eval series is trained on. TinyCast's authors make the same disclosure about their own design and treat Chronos-ZS as the cleaner test. G3 exists for this reason. Every report of v2 states this paragraph.

## Share of v1's gap to the TinyCast port closed by v2 (sum of log ratios)

| metric | configs | log gap v1 / port | closed by v2 | share closed |
|---|---|---|---|---|
| MASE | 97 | 18.1338 | 6.6150 | 0.365 |
| WQL | 97 | 17.1265 | 6.8879 | 0.402 |

*Disclosure (docs/s6-cast2-prereg.md §10).* v2 was designed after reading v1's GIFT-Eval results on the same 97 configurations, window by window: where it lost, by band, term, domain, seasonality, context length and horizon. GIFT-Eval is therefore partly in-sample for v2's design choices, though not for its weights: no GIFT-Eval series is trained on. TinyCast's authors make the same disclosure about their own design and treat Chronos-ZS as the cleaner test. G3 exists for this reason. Every report of v2 states this paragraph.

## Coverage and front-end diagnostics (per window)

### Coverage by term (share of labels at or below each level; last: inside the 80% interval)

| term | model | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 | 80% interval |
|---|---|---|---|---|---|---|---|---|---|---|---|
| short | v1 | 0.092 | 0.175 | 0.264 | 0.351 | 0.474 | 0.577 | 0.682 | 0.790 | 0.883 | 0.790 |
| medium | v1 | 0.178 | 0.245 | 0.305 | 0.365 | 0.442 | 0.512 | 0.591 | 0.671 | 0.756 | 0.578 |
| long | v1 | 0.200 | 0.264 | 0.320 | 0.375 | 0.443 | 0.505 | 0.575 | 0.648 | 0.725 | 0.526 |
| short | v2 | 0.093 | 0.188 | 0.279 | 0.380 | 0.486 | 0.592 | 0.697 | 0.803 | 0.892 | 0.800 |
| medium | v2 | 0.183 | 0.254 | 0.322 | 0.384 | 0.455 | 0.524 | 0.594 | 0.678 | 0.761 | 0.578 |
| long | v2 | 0.214 | 0.282 | 0.343 | 0.399 | 0.461 | 0.521 | 0.583 | 0.657 | 0.735 | 0.522 |
| short | port | 0.091 | 0.178 | 0.270 | 0.378 | 0.502 | 0.617 | 0.713 | 0.810 | 0.909 | 0.819 |
| medium | port | 0.163 | 0.250 | 0.318 | 0.385 | 0.476 | 0.549 | 0.615 | 0.695 | 0.802 | 0.643 |
| long | port | 0.207 | 0.280 | 0.334 | 0.390 | 0.466 | 0.523 | 0.578 | 0.646 | 0.750 | 0.547 |
| short | sn | 0.505 | 0.505 | 0.505 | 0.505 | 0.505 | 0.505 | 0.505 | 0.505 | 0.505 | 0.091 |
| medium | sn | 0.552 | 0.552 | 0.552 | 0.552 | 0.552 | 0.552 | 0.552 | 0.552 | 0.552 | 0.128 |
| long | sn | 0.550 | 0.550 | 0.550 | 0.550 | 0.550 | 0.550 | 0.550 | 0.550 | 0.550 | 0.122 |

*Disclosure (docs/s6-cast2-prereg.md §10).* v2 was designed after reading v1's GIFT-Eval results on the same 97 configurations, window by window: where it lost, by band, term, domain, seasonality, context length and horizon. GIFT-Eval is therefore partly in-sample for v2's design choices, though not for its weights: no GIFT-Eval series is trained on. TinyCast's authors make the same disclosure about their own design and treat Chronos-ZS as the cleaner test. G3 exists for this reason. Every report of v2 states this paragraph.

### The horizon profile by tenths (ratio of summed scaled errors)

| term | ratio | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| long | v2 / port | 1.187 | 1.184 | 1.152 | 1.214 | 1.250 | 1.206 | 1.173 | 1.167 | 1.158 | 1.189 |
| long | v2 / sn | 0.869 | 0.933 | 1.077 | 0.927 | 1.066 | 0.973 | 0.998 | 1.060 | 0.966 | 1.094 |
| medium | v2 / port | 1.256 | 1.233 | 1.175 | 1.212 | 1.231 | 1.184 | 1.173 | 1.168 | 1.097 | 1.141 |
| medium | v2 / sn | 0.773 | 0.902 | 0.984 | 0.949 | 0.916 | 0.991 | 0.984 | 0.917 | 0.937 | 0.992 |
| short | v2 / port | 1.265 | 1.139 | 1.121 | 1.130 | 1.157 | 1.179 | 1.113 | 1.067 | 1.096 | 1.229 |
| short | v2 / sn | 0.765 | 0.808 | 0.853 | 0.845 | 0.852 | 0.803 | 0.858 | 0.831 | 0.851 | 0.896 |

*Disclosure (docs/s6-cast2-prereg.md §10).* v2 was designed after reading v1's GIFT-Eval results on the same 97 configurations, window by window: where it lost, by band, term, domain, seasonality, context length and horizon. GIFT-Eval is therefore partly in-sample for v2's design choices, though not for its weights: no GIFT-Eval series is trained on. TinyCast's authors make the same disclosure about their own design and treat Chronos-ZS as the cleaner test. G3 exists for this reason. Every report of v2 states this paragraph.

### Window-level splits, v2 / TinyCast port (configurations with >= 20 such windows)

| windows | configs | v2 / port MASE |
|---|---|---|
| a period detected | 83 | 1.140 |
| no period detected | 33 | 1.134 |
| observed 128-511 | 18 | 1.086 |
| observed 2,048 | 62 | 1.164 |
| observed 512-2,047 | 23 | 1.185 |
| observed < 128 | 13 | 1.015 |
| period < 32 | 44 | 1.200 |
| period >= 32, not a multiple of 32 | 64 | 1.122 |
| period a multiple of 32 (+-1) | 41 | 1.120 |

*Disclosure (docs/s6-cast2-prereg.md §10).* v2 was designed after reading v1's GIFT-Eval results on the same 97 configurations, window by window: where it lost, by band, term, domain, seasonality, context length and horizon. GIFT-Eval is therefore partly in-sample for v2's design choices, though not for its weights: no GIFT-Eval series is trained on. TinyCast's authors make the same disclosure about their own design and treat Chronos-ZS as the cleaner test. G3 exists for this reason. Every report of v2 states this paragraph.

## Latency and size

| model | mean exits used on GIFT-Eval | windows |
|---|---|---|
| v2 | 3.334 of 4 | 369313 |
| v1 | not recorded |  |

*Disclosure (docs/s6-cast2-prereg.md §10).* v2 was designed after reading v1's GIFT-Eval results on the same 97 configurations, window by window: where it lost, by band, term, domain, seasonality, context length and horizon. GIFT-Eval is therefore partly in-sample for v2's design choices, though not for its weights: no GIFT-Eval series is trained on. TinyCast's authors make the same disclosure about their own design and treat Chronos-ZS as the cleaner test. G3 exists for this reason. Every report of v2 states this paragraph.

| file | bytes |
|---|---|
| s6-cast2-pico.gguf | 148512 |
| s6-cast-pico.gguf | 142912 |
| tinycast.gguf | 595840 |

*Disclosure (docs/s6-cast2-prereg.md §10).* v2 was designed after reading v1's GIFT-Eval results on the same 97 configurations, window by window: where it lost, by band, term, domain, seasonality, context length and horizon. GIFT-Eval is therefore partly in-sample for v2's design choices, though not for its weights: no GIFT-Eval series is trained on. TinyCast's authors make the same disclosure about their own design and treat Chronos-ZS as the cleaner test. G3 exists for this reason. Every report of v2 states this paragraph.

Measured 2026-11-18T11:25:40+0100 on Apple M5, one core, 2,048-point history.

| target | model | depth | horizon | median µs | p95 µs | forecasts |
|---|---|---|---|---|---|---|
| native | v1 | full | 64 | 799 | 870 | 1000 |
| native | v2 | full | 64 | 838 | 906 | 1000 |
| native | tinycast | full | 64 | 21531 | 23189 | 1000 |
| native | v2 | calibrated | 64 | 842 | 904 | 1000 |
| native | v2 | full | 720 | 10367 | 11158 | 1000 |
| native | v1 | full | 720 | 9826 | 10474 | 1000 |
| native | tinycast | full | 720 | 164760 | 171634 | 1000 |
| wasm | s6-cast-pico-full.gguf | full | 64 | 1545 | 1908 | 1000 |
| wasm | s6-cast2-pico-full.gguf | full | 64 | 1636 | 1955 | 1000 |
| wasm | tinycast.gguf | full | 64 | 39998 | 41530 | 1000 |
| wasm | s6-cast2-pico.gguf | calibrated | 64 | 1528 | 1650 | 1000 |
| wasm | s6-cast-pico-full.gguf | full | 720 | 17922 | 19230 | 1000 |
| wasm | s6-cast2-pico-full.gguf | full | 720 | 19346 | 22060 | 1000 |
| wasm | tinycast.gguf | full | 720 | 309243 | 375050 | 1000 |
| wasm | s6-cast2-pico.gguf | calibrated | 720 | 18232 | 19463 | 1000 |

*Disclosure (docs/s6-cast2-prereg.md §10).* v2 was designed after reading v1's GIFT-Eval results on the same 97 configurations, window by window: where it lost, by band, term, domain, seasonality, context length and horizon. GIFT-Eval is therefore partly in-sample for v2's design choices, though not for its weights: no GIFT-Eval series is trained on. TinyCast's authors make the same disclosure about their own design and treat Chronos-ZS as the cleaner test. G3 exists for this reason. Every report of v2 states this paragraph.

Recommendation: v2 at 838 µs and 148512 bytes against the port's 21531 µs and 595840 bytes, nGMASE 0.8847 against 0.7856: the port forecasts better; take it unless size or speed decide.

## Training

| item | value |
|---|---|
| steps done | 377082 |
| working minutes | 720.0 |
| windows seen, synthetic | 21295763 |
| windows seen, real | 25092584 |
| windows seen, band_synthetic | 6420364 |
| windows seen, distilled | 43724281 |
| validation pinball (ema-ternary), synthetic | 0.04884 (512 windows) |
| validation pinball (ema-ternary), subhourly | 0.02609 (285 windows) |
| validation pinball (ema-ternary), hourly | 0.03496 (55 windows) |
| validation pinball (ema-ternary), daily | 0.01968 (621 windows) |
| validation pinball (ema-ternary), real_unbanded | 0.17125 (141 windows) |

*Disclosure (docs/s6-cast2-prereg.md §10).* v2 was designed after reading v1's GIFT-Eval results on the same 97 configurations, window by window: where it lost, by band, term, domain, seasonality, context length and horizon. GIFT-Eval is therefore partly in-sample for v2's design choices, though not for its weights: no GIFT-Eval series is trained on. TinyCast's authors make the same disclosure about their own design and treat Chronos-ZS as the cleaner test. G3 exists for this reason. Every report of v2 states this paragraph.

### Brake modes

| line | times |
|---|---|
| cooling down for N s (SKY_COOL) | 20 |
| brake: duty 0.95, 55 min of work, 5 of rest (/Users/maxime/Documents/Code/skycmd/runs/brake) | 3 |
| brake: duty 0.5, 45 min of work, 15 of rest (/Users/maxime/Documents/Code/skycmd/runs/brake) | 3 |

*Disclosure (docs/s6-cast2-prereg.md §10).* v2 was designed after reading v1's GIFT-Eval results on the same 97 configurations, window by window: where it lost, by band, term, domain, seasonality, context length and horizon. GIFT-Eval is therefore partly in-sample for v2's design choices, though not for its weights: no GIFT-Eval series is trained on. TinyCast's authors make the same disclosure about their own design and treat Chronos-ZS as the cleaner test. G3 exists for this reason. Every report of v2 states this paragraph.

### Data sources

| file | series | points |
|---|---|---|
| /Users/maxime/Documents/Code/skycmd-data/s6-cast2/series2-checked/uscrn_subhourly.jsonl | 1011 | 8282112 |
| data/series/games.jsonl | 1549 | 343398 |
| data/series/london_smart_meters.jsonl | 2000 | 16364432 |
| data/series/pedestrian_counts.jsonl | 66 | 501116 |
| data/series/rideshare.jsonl | 200 | 108200 |
| data/series/ushcn_daily.jsonl | 1300 | 10646695 |
| data/series/wiki_daily.jsonl | 5000 | 13705000 |
| data/series/wind_farms_daily.jsonl | 337 | 119549 |
| data/series/wind_farms_hourly.jsonl | 337 | 2684858 |

*Disclosure (docs/s6-cast2-prereg.md §10).* v2 was designed after reading v1's GIFT-Eval results on the same 97 configurations, window by window: where it lost, by band, term, domain, seasonality, context length and horizon. GIFT-Eval is therefore partly in-sample for v2's design choices, though not for its weights: no GIFT-Eval series is trained on. TinyCast's authors make the same disclosure about their own design and treat Chronos-ZS as the cleaner test. G3 exists for this reason. Every report of v2 states this paragraph.

| source | series | informative runs | matching runs | kept |
|---|---|---|---|---|
| uscrn_subhourly | 1011 | 4275428 | 0 | True |

*Disclosure (docs/s6-cast2-prereg.md §10).* v2 was designed after reading v1's GIFT-Eval results on the same 97 configurations, window by window: where it lost, by band, term, domain, seasonality, context length and horizon. GIFT-Eval is therefore partly in-sample for v2's design choices, though not for its weights: no GIFT-Eval series is trained on. TinyCast's authors make the same disclosure about their own design and treat Chronos-ZS as the cleaner test. G3 exists for this reason. Every report of v2 states this paragraph.

