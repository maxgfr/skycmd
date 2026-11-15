# s6-secrets v3 (secretgate's default engine and s6-secrets-verify-pico together): results on the validation split (docs/s6-secrets3-prereg.md)

**Rehearsal.** These are the validation repositories, on which tau3 was chosen: no number below is a result of the pre-registration, and the gate at the end is the test's computation run on validation.

**Second look: v3 reads test files that were read once for v2 (docs/s6-secrets2-prereg.md, Result of 15/11/2026), and its system (the union of the default engine and the model, with tau3) was defined after that reading.** The reading of v2 does not change: it failed its gate on G1 and stays failed.

**The systems.** `union`: a line is flagged when secretgate's default engine (secretgate 4e93cfa, filters on) flags it, or when the model of v2 (the evaluated file, unchanged) accepts a candidate of the frozen generator on it with P(real) >= tau3 = 0.34933. `union-lr`: the same with the size-matched logistic regression in the model's seat, at its own union threshold 0.155049 (G3). `union-tau-v2`: the union at v2's tau* 0.2598 (reported). `model` is the model alone at tau*, v2's system. On the issue reports a dataset candidate is flagged when a default finding overlaps it or the verifier accepts it; on the clean text a file counts when either flags it. No scanner, model or baseline was rerun: every flag is the v2 evaluation's.

tau3 and the regression's union threshold were chosen on the 16 validation repositories by the rule of section 1 (`runs/s6-secrets3/tau3.json`, committed at 2026-11-15T09:05:11+01:00 in `370f60f`); validation line F1 of the union at tau3, gated: 0.8963.

16 repositories, 452 files; T rows: 522 (gated universe), 659 (full). Largest repository: `7738e44d`, 217 gated T rows (the split's own largest, a stand-in for the test's).

## CredData, line level (7.1)

Gated universe: the rows whose CredData category is exactly `UUID` are removed, for every system (amendment 2 of docs/s6-secrets2-prereg.md). That choice was made after reading the training and validation rows and before any test file was read. Full universe: every meta row.

Second look: v3 reads test files that were read once for v2 (docs/s6-secrets2-prereg.md, Result of 15/11/2026), and its system (the union of the default engine and the model, with tau3) was defined after that reading.

| gated universe | precision | recall | F1 | false alarms | F1 without the largest | macro F1 (≥ 20 T rows) |
|---|---:|---:|---:|---:|---:|---:|
| union | 0.9136 | 0.8796 | 0.8963 | 0.0194 | 0.8576 | 0.8797 (6) |
| union-lr | 0.8838 | 0.8490 | 0.8661 | 0.0260 | 0.8049 | 0.8324 (6) |
| union-tau-v2 | 0.8721 | 0.8950 | 0.8834 | 0.0306 | 0.8386 | 0.8630 (6) |
| model | 0.8790 | 0.8425 | 0.8603 | 0.0270 | 0.8000 | 0.8335 (6) |
| lr | 0.8863 | 0.8359 | 0.8604 | 0.0250 | 0.7972 | 0.8268 (6) |
| secretgate-default | 0.9270 | 0.3611 | 0.5197 | 0.0066 | 0.6136 | 0.5118 (6) |
| generator | 0.2753 | 0.9540 | 0.4272 | 0.5848 | 0.3236 | 0.6393 (6) |
| gitleaks | 0.9850 | 0.2867 | 0.4441 | 0.0010 | 0.5114 | 0.3793 (6) |
| betterleaks | 0.7500 | 0.3151 | 0.4438 | 0.0245 | 0.5022 | 0.4386 (6) |
| detect-secrets | 0.6069 | 0.3851 | 0.4712 | 0.0581 | 0.6048 | 0.5435 (6) |
| credsweeper-ml | 0.9202 | 0.9847 | 0.9514 | 0.0199 | 0.9253 | 0.9476 (6) |
| credsweeper-noml | 0.3218 | 1.0000 | 0.4869 | 0.4906 | 0.3873 | 0.7434 (6) |
| cbx-17m | 0.9814 | 0.8096 | 0.8873 | 0.0036 | 0.8191 | 0.8473 (6) |
| potion-lr | 0.5243 | 0.8271 | 0.6418 | 0.1747 | 0.5325 | 0.6934 (6) |

| full universe | precision | recall | F1 | false alarms | F1 without the largest | macro F1 (≥ 20 T rows) |
|---|---:|---:|---:|---:|---:|---:|
| union | 0.9295 | 0.7040 | 0.8012 | 0.0159 | 0.7962 | 0.8138 (6) |
| union-lr | 0.8838 | 0.6678 | 0.7608 | 0.0261 | 0.7264 | 0.7495 (6) |
| union-tau-v2 | 0.8913 | 0.7194 | 0.7962 | 0.0261 | 0.7873 | 0.8059 (6) |
| model | 0.8995 | 0.6781 | 0.7733 | 0.0225 | 0.7500 | 0.7814 (6) |
| lr | 0.8863 | 0.6575 | 0.7549 | 0.0251 | 0.7186 | 0.7438 (6) |
| secretgate-default | 0.9270 | 0.2840 | 0.4348 | 0.0067 | 0.5378 | 0.4493 (6) |
| generator | 0.2842 | 0.7762 | 0.4161 | 0.5820 | 0.3287 | 0.6187 (6) |
| gitleaks | 0.9850 | 0.2255 | 0.3669 | 0.0010 | 0.4420 | 0.3237 (6) |
| betterleaks | 0.7500 | 0.2478 | 0.3726 | 0.0246 | 0.4419 | 0.4006 (6) |
| detect-secrets | 0.6069 | 0.3029 | 0.4041 | 0.0584 | 0.5466 | 0.4989 (6) |
| credsweeper-ml | 0.9550 | 0.9862 | 0.9704 | 0.0138 | 0.9519 | 0.9680 (6) |
| credsweeper-noml | 0.3788 | 0.9983 | 0.5492 | 0.4872 | 0.4361 | 0.7810 (6) |
| cbx-17m | 0.9814 | 0.6368 | 0.7724 | 0.0036 | 0.7292 | 0.7717 (6) |
| potion-lr | 0.5326 | 0.6609 | 0.5899 | 0.1726 | 0.5094 | 0.6528 (6) |

CredSweeper's ML was trained on a random 85% of all CredData lines: its CredData numbers are in-sample and enter no gate (section 6 of v2).

## Lines the union adds over the model alone (reported, section 4)

Gated universe: the rows whose CredData category is exactly `UUID` are removed, for every system (amendment 2 of docs/s6-secrets2-prereg.md). That choice was made after reading the training and validation rows and before any test file was read. Full universe: every meta row.

Second look: v3 reads test files that were read once for v2 (docs/s6-secrets2-prereg.md, Result of 15/11/2026), and its system (the union of the default engine and the model, with tau3) was defined after that reading.

Line units flagged by the union and not by the model alone at tau* (added), and the reverse (dropped), by label: T a positive line, X a negative line with an X row, F the other negative lines.

| universe | union | threshold | added T | added X | added F | dropped T | dropped X | dropped F | lines the default engine flags | line units |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gated | union | 0.3493 | 24 | 1 | 6 | 7 | 3 | 19 | 178 | 2420 |
| gated | union-tau-v2 | 0.2598 | 24 | 1 | 6 | 0 | 0 | 0 | 178 | 2420 |
| full | union | 0.3493 | 24 | 1 | 6 | 9 | 3 | 17 | 178 | 2533 |
| full | union-tau-v2 | 0.2598 | 24 | 1 | 6 | 0 | 0 | 0 | 178 | 2533 |

## Unlabelled-line alarms (7.3), versions and CPU time

Gated universe: the rows whose CredData category is exactly `UUID` are removed, for every system (amendment 2 of docs/s6-secrets2-prereg.md). That choice was made after reading the training and validation rows and before any test file was read. Full universe: every meta row.

Second look: v3 reads test files that were read once for v2 (docs/s6-secrets2-prereg.md, Result of 15/11/2026), and its system (the union of the default engine and the model, with tau3) was defined after that reading.

| | version | unit | unlabelled-line alarms: flags | lines | flags per 1,000 files | CPU s |
|---|---|---|---:|---:|---:|---:|
| union | secretgate 4e93cfa2ee6e216caca9bc8e6f74cb565a991350 or model (full-depth.gguf sha256 01e6db3fa71cc2c5563dea222c68e040d3994ee2abbb02b25491a51b8cb54f27) at P(real) >= 0.34933 | line | 121 | 120 | 267.7 | — |
| union-lr | secretgate 4e93cfa2ee6e216caca9bc8e6f74cb565a991350 or lr (scikit-learn 1.9.1, SGDClassifier alpha 0.0001, 2^18 int8 weights) at P(real) >= 0.155049 | line | 264 | 177 | 584.1 | — |
| union-tau-v2 | secretgate 4e93cfa2ee6e216caca9bc8e6f74cb565a991350 or model (full-depth.gguf sha256 01e6db3fa71cc2c5563dea222c68e040d3994ee2abbb02b25491a51b8cb54f27) at P(real) >= 0.2598114609718323 | line | 141 | 139 | 311.9 | — |
| model | full-depth.gguf sha256 01e6db3fa71cc2c5563dea222c68e040d3994ee2abbb02b25491a51b8cb54f27 | candidate | 53 | 51 | 117.3 | — |
| lr | scikit-learn 1.9.1, SGDClassifier alpha 0.0001, 2^18 int8 weights | candidate | 176 | 176 | 389.4 | 1.3 |
| secretgate-default | secretgate 4e93cfa2ee6e216caca9bc8e6f74cb565a991350 | span | 88 | 88 | 194.7 | 8.7 |
| generator | sg-cand-1 revision 1, secretgate 4e93cfa2ee6e216caca9bc8e6f74cb565a991350 | candidate | 2279 | 2028 | 5042.0 | 17.2 |
| gitleaks | Gitleaks 8.30.1 | span | 1 | 1 | 2.2 | 20.6 |
| betterleaks | Betterleaks 1.10.1 | span | 12 | 11 | 26.5 | 2.2 |
| detect-secrets | detect-secrets 1.5.0 | line | 50 | 48 | 110.6 | 115.0 |
| credsweeper-ml | CredSweeper 1.19.6 | span | 0 | 0 | 0.0 | 43.9 |
| credsweeper-noml | CredSweeper 1.19.6 | span | 0 | 0 | 0.0 | 24.5 |
| cbx-17m | cbx-secrets-detector-17M at 89362a0938f9459e5706e703e2a91c5cf2ffe40e | span | 2074 | 17 | 4588.5 | 13816.6 |
| potion-lr | model2vec 0.10.0 potion-base-2M, scikit-learn 1.9.1 LogisticRegression | candidate | 437 | 416 | 966.8 | 0.7 |

A union's flags are the default engine's plus the verifier's accepted candidates: a line both flag counts twice in the flags column, once in the lines column.

## CredData, candidate level (7.2): the scored candidates of v2

Gated universe: the rows whose CredData category is exactly `UUID` are removed, for every system (amendment 2 of docs/s6-secrets2-prereg.md). That choice was made after reading the training and validation rows and before any test file was read. Full universe: every meta row.

Second look: v3 reads test files that were read once for v2 (docs/s6-secrets2-prereg.md, Result of 15/11/2026), and its system (the union of the default engine and the model, with tau3) was defined after that reading.

The union is a line system (the default engine gives no score), so it has no candidate-level row; the model's and the baselines' rows are v2's.

| gated universe | candidates | positives | threshold | precision | recall | F1 | PR-AUC | recall at 1% false alarms | ECE | ECE on P(real) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| model | 1779 | 448 | 0.2598 | 0.8725 | 0.8705 | 0.8715 | 0.9427 | 0.7143 | 0.0135 | 0.0338 |
| lr | 1779 | 448 | 0.1550 | 0.8810 | 0.8594 | 0.8701 | 0.9145 | 0.6451 | 0.0678 | 0.1183 |
| generator | 1779 | 448 | — | 0.2518 | 1.0000 | 0.4023 | — | — | — | — |
| potion-lr | 1779 | 448 | 0.0786 | 0.5039 | 0.8638 | 0.6365 | 0.6340 | 0.1027 | 0.1081 | 0.1221 |

| full universe | candidates | positives | threshold | precision | recall | F1 | PR-AUC | recall at 1% false alarms | ECE | ECE on P(real) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| model | 1779 | 459 | 0.2598 | 0.8926 | 0.8693 | 0.8808 | 0.9443 | 0.7495 | 0.0151 | 0.0400 |
| lr | 1779 | 459 | 0.1550 | 0.8810 | 0.8388 | 0.8594 | 0.9036 | 0.6296 | 0.0671 | 0.1180 |
| generator | 1779 | 459 | — | 0.2580 | 1.0000 | 0.4102 | — | — | — | — |
| potion-lr | 1779 | 459 | 0.0786 | 0.5117 | 0.8562 | 0.6406 | 0.6327 | 0.1002 | 0.1143 | 0.1282 |

## CredData, span level (section 9 of v2, overlap with a T value)

Gated universe: the rows whose CredData category is exactly `UUID` are removed, for every system (amendment 2 of docs/s6-secrets2-prereg.md). That choice was made after reading the training and validation rows and before any test file was read. Full universe: every meta row.

Second look: v3 reads test files that were read once for v2 (docs/s6-secrets2-prereg.md, Result of 15/11/2026), and its system (the union of the default engine and the model, with tau3) was defined after that reading.

| | gated recall | gated precision | gated F1 | full recall | full precision | full F1 | flags: T / X / F / no row / unlabelled |
|---|---:|---:|---:|---:|---:|---:|---|
| union | 0.7854 | 0.9095 | 0.8430 | 0.6328 | 0.9213 | 0.7502 | 550 / 5 / 36 / 6 / 121 |
| union-lr | 0.7567 | 0.8901 | 0.8180 | 0.5994 | 0.8901 | 0.7164 | 551 / 13 / 51 / 4 / 264 |
| union-tau-v2 | 0.8008 | 0.8672 | 0.8327 | 0.6480 | 0.8812 | 0.7468 | 564 / 8 / 52 / 16 / 141 |
| model | 0.7548 | 0.8442 | 0.7970 | 0.6115 | 0.8636 | 0.7160 | 459 / 117 / 1203 / 176 / 2279 |
| lr | 0.7452 | 0.8750 | 0.8049 | 0.5903 | 0.8750 | 0.7050 | 459 / 117 / 1203 / 176 / 2279 |
| secretgate-default | 0.3238 | 0.9270 | 0.4799 | 0.2564 | 0.9270 | 0.4018 | 165 / 2 / 10 / 1 / 88 |
| generator | 0.8506 | 0.2295 | 0.3615 | 0.6904 | 0.2348 | 0.3504 | 459 / 117 / 1203 / 176 / 2279 |
| gitleaks | 0.2586 | 0.9851 | 0.4097 | 0.2049 | 0.9851 | 0.3392 | 132 / 0 / 2 / 0 / 1 |
| betterleaks | 0.2816 | 0.7164 | 0.4043 | 0.2231 | 0.7164 | 0.3402 | 144 / 13 / 40 / 4 / 12 |
| credsweeper-ml | 0.8851 | 0.9257 | 0.9049 | 0.9059 | 0.9524 | 0.9286 | 820 / 0 / 41 / 0 / 0 |
| credsweeper-noml | 0.8966 | 0.3742 | 0.5280 | 0.9150 | 0.4252 | 0.5806 | 839 / 27 / 1107 / 0 / 0 |
| cbx-17m | 0.8257 | 0.9651 | 0.8900 | 0.6540 | 0.9651 | 0.7797 | 443 / 0 / 12 / 4 / 2074 |
| potion-lr | 0.7414 | 0.4731 | 0.5776 | 0.5964 | 0.4804 | 0.5322 | 459 / 117 / 1203 / 176 / 2279 |

The flag counts of a scored system are of all its candidates; those of a union, of its flags: the default engine's and the verifier's accepted candidates.

## By CredData category and by path scope (section 9 of v2), gated universe, line F1 (recall)

Gated universe: the rows whose CredData category is exactly `UUID` are removed, for every system (amendment 2 of docs/s6-secrets2-prereg.md). That choice was made after reading the training and validation rows and before any test file was read. Full universe: every meta row.

Second look: v3 reads test files that were read once for v2 (docs/s6-secrets2-prereg.md, Result of 15/11/2026), and its system (the union of the default engine and the model, with tau3) was defined after that reading.

| | Password | Token | Key | Secret | API | Auth | other | scope test | scope src | scope other |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| union | 0.872 (0.830) | 0.947 (0.943) | 0.808 (0.875) | 0.938 (0.938) | 0.778 (1.000) | 0.879 (0.851) | 0.918 (0.890) | 0.842 (0.832) | 0.929 (0.929) | 0.943 (0.919) |
| union-lr | 0.745 (0.748) | 0.945 (0.905) | 0.826 (0.792) | 0.909 (0.938) | 0.778 (1.000) | 0.907 (0.936) | 0.921 (0.890) | 0.822 (0.812) | 0.966 (1.000) | 0.899 (0.872) |
| union-tau-v2 | 0.871 (0.852) | 0.943 (0.952) | 0.761 (0.896) | 0.939 (0.969) | 0.700 (1.000) | 0.860 (0.915) | 0.908 (0.890) | 0.817 (0.837) | 0.966 (1.000) | 0.938 (0.940) |
| model | 0.868 (0.830) | 0.943 (0.952) | 0.586 (0.604) | 0.938 (0.938) | 0.737 (1.000) | 0.869 (0.915) | 0.845 (0.772) | 0.763 (0.736) | 0.889 (0.857) | 0.942 (0.936) |
| lr | 0.746 (0.741) | 0.940 (0.895) | 0.813 (0.771) | 0.909 (0.938) | 0.778 (1.000) | 0.907 (0.936) | 0.910 (0.869) | 0.815 (0.793) | 0.966 (1.000) | 0.894 (0.864) |
| secretgate-default | 0.444 (0.296) | 0.552 (0.381) | 0.737 (0.583) | 0.755 (0.625) | 0.750 (0.857) | 0.255 (0.149) | 0.574 (0.414) | 0.667 (0.519) | 0.600 (0.429) | 0.351 (0.217) |
| generator | 0.575 (0.970) | 0.550 (1.000) | 0.140 (0.938) | 0.383 (1.000) | 0.136 (1.000) | 0.455 (0.979) | 0.814 (0.903) | 0.388 (0.909) | 0.452 (1.000) | 0.464 (0.991) |
| gitleaks | 0.349 (0.215) | 0.500 (0.333) | 0.609 (0.438) | 0.609 (0.438) | 0.000 (0.000) | 0.082 (0.043) | 0.520 (0.352) | 0.586 (0.418) | 0.526 (0.357) | 0.285 (0.166) |
| betterleaks | 0.595 (0.556) | 0.507 (0.343) | 0.286 (0.167) | 0.476 (0.312) | 0.222 (0.143) | 0.157 (0.085) | 0.256 (0.159) | 0.429 (0.322) | 1.000 (1.000) | 0.408 (0.268) |
| detect-secrets | 0.747 (0.756) | 0.090 (0.048) | 0.364 (0.292) | 0.408 (0.656) | 0.636 (1.000) | 0.241 (0.149) | 0.427 (0.303) | 0.649 (0.601) | 0.444 (0.857) | 0.253 (0.166) |
| credsweeper-ml | 0.978 (0.978) | 0.986 (0.971) | 0.750 (1.000) | 0.968 (0.938) | 1.000 (1.000) | 0.979 (0.979) | 0.924 (1.000) | 0.904 (0.976) | 1.000 (1.000) | 0.994 (0.991) |
| credsweeper-noml | 0.736 (1.000) | 0.717 (1.000) | 0.142 (1.000) | 0.604 (1.000) | 0.099 (1.000) | 0.470 (1.000) | 0.863 (1.000) | 0.547 (1.000) | 0.075 (1.000) | 0.634 (1.000) |
| cbx-17m | 0.946 (0.904) | 0.951 (0.924) | 0.824 (0.729) | 0.918 (0.875) | 0.833 (0.714) | 0.860 (0.787) | 0.759 (0.621) | 0.760 (0.625) | 0.966 (1.000) | 0.976 (0.962) |
| potion-lr | 0.597 (0.785) | 0.816 (0.867) | 0.400 (0.875) | 0.756 (0.969) | 0.177 (1.000) | 0.765 (0.936) | 0.787 (0.779) | 0.539 (0.712) | 0.788 (0.929) | 0.728 (0.923) |
| positive / negative lines | 135 / 282 | 105 / 233 | 48 / 1028 | 32 / 163 | 7 / 149 | 47 / 181 | 145 / 80 | 208 / 811 | 14 / 395 | 235 / 757 |

## Uncertainty (7.6): paired bootstrap over repositories, 1000 resamples, 95%

Gated universe: the rows whose CredData category is exactly `UUID` are removed, for every system (amendment 2 of docs/s6-secrets2-prereg.md). That choice was made after reading the training and validation rows and before any test file was read. Full universe: every meta row.

Second look: v3 reads test files that were read once for v2 (docs/s6-secrets2-prereg.md, Result of 15/11/2026), and its system (the union of the default engine and the model, with tau3) was defined after that reading.

| line F1 of union minus | gated | gated, without the largest | full | full, without the largest |
|---|---|---|---|---|
| union-lr | +0.0302 [-0.0050, 0.1323] | +0.0528 [-0.0057, 0.1924] | +0.0404 [0.0084, 0.1356] | +0.0698 [0.0277, 0.1900] |
| union-tau-v2 | +0.0130 [-0.0040, 0.0388] | +0.0190 [-0.0070, 0.0476] | +0.0050 [-0.0048, 0.0209] | +0.0089 [-0.0070, 0.0275] |
| model | +0.0360 [-0.0018, 0.1301] | +0.0576 [0.0000, 0.1487] | +0.0279 [-0.0017, 0.0961] | +0.0462 [0.0005, 0.1075] |
| lr | +0.0360 [-0.0005, 0.1429] | +0.0605 [-0.0013, 0.1937] | +0.0462 [0.0127, 0.1393] | +0.0776 [0.0356, 0.1900] |
| secretgate-default | +0.3766 [0.1056, 0.5793] | +0.2440 [0.0679, 0.4831] | +0.3664 [0.1565, 0.5266] | +0.2584 [0.1309, 0.4600] |
| generator | +0.4691 [0.3026, 0.6499] | +0.5341 [0.3678, 0.7148] | +0.3851 [0.2336, 0.5887] | +0.4675 [0.3262, 0.6919] |
| gitleaks | +0.4523 [0.1579, 0.6527] | +0.3462 [0.1009, 0.6609] | +0.4342 [0.2067, 0.5984] | +0.3542 [0.1641, 0.6236] |
| betterleaks | +0.4526 [0.2502, 0.5940] | +0.3554 [0.2188, 0.4993] | +0.4286 [0.2461, 0.5402] | +0.3544 [0.2188, 0.4928] |
| detect-secrets | +0.4251 [0.1403, 0.7138] | +0.2528 [0.1136, 0.4785] | +0.3970 [0.1546, 0.6319] | +0.2497 [0.1245, 0.4415] |
| credsweeper-ml | -0.0551 [-0.1176, -0.0018] | -0.0677 [-0.1236, 0.0051] | -0.1692 [-0.1987, -0.1135] | -0.1556 [-0.2035, -0.0976] |
| credsweeper-noml | +0.4094 [0.1604, 0.6415] | +0.4704 [0.1987, 0.7246] | +0.2519 [0.0032, 0.5560] | +0.3602 [0.0871, 0.6880] |
| cbx-17m | +0.0090 [-0.0515, 0.1511] | +0.0386 [-0.0670, 0.1901] | +0.0287 [-0.0496, 0.1976] | +0.0670 [-0.0659, 0.2454] |
| potion-lr | +0.2546 [0.1457, 0.4097] | +0.3251 [0.2287, 0.4691] | +0.2113 [0.1167, 0.3570] | +0.2869 [0.2141, 0.4272] |

| ECE (scored candidates of v2) | gated | full |
|---|---|---|
| model | 0.0135 [0.0094, 0.0507] | 0.0151 [0.0097, 0.0555] |
| lr | 0.0678 [0.0289, 0.1369] | 0.0671 [0.0291, 0.1474] |
| potion-lr | 0.1081 [0.0342, 0.2369] | 0.1143 [0.0354, 0.2481] |

## Issue reports, validation split (7.4)

Gated universe: the rows whose CredData category is exactly `UUID` are removed, for every system (amendment 2 of docs/s6-secrets2-prereg.md). That choice was made after reading the training and validation rows and before any test file was read. Full universe: every meta row.

Second look: v3 reads test files that were read once for v2 (docs/s6-secrets2-prereg.md, Result of 15/11/2026), and its system (the union of the default engine and the model, with tau3) was defined after that reading.

5432 candidates (586 secrets) in 5371 texts; 524 occur several times in their text (the first occurrence is the candidate), 0 are not found in it.

| | precision | recall | F1 | false alarms | ECE | end to end: flags | precision over all flags | F1 | precision over flags on a candidate | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| union | 0.6968 | 0.9334 | 0.7980 | 0.0491 | — | — | — | — | — | — |
| union-lr | 0.4626 | 0.9403 | 0.6201 | 0.1321 | — | — | — | — | — | — |
| union-tau-v2 | 0.6655 | 0.9403 | 0.7793 | 0.0572 | — | — | — | — | — | — |
| model | 0.7423 | 0.9044 | 0.8154 | 0.0380 | 0.0226 | — | — | — | — | — |
| lr | 0.4888 | 0.8908 | 0.6312 | 0.1127 | 0.0335 | — | — | — | — | — |
| secretgate-default | 0.7379 | 0.6007 | 0.6623 | 0.0258 | — | 1467 | 0.2399 | 0.3429 | 0.7379 | 0.6623 |
| generator | 0.2222 | 0.7884 | 0.3467 | 0.3337 | — | 16426 | 0.0289 | 0.0558 | 0.2134 | 0.3359 |
| gitleaks | 0.8635 | 0.5614 | 0.6805 | 0.0107 | — | 1055 | 0.3166 | 0.4049 | 0.8630 | 0.6803 |
| betterleaks | 0.5684 | 0.5887 | 0.5784 | 0.0541 | — | 2024 | 0.1764 | 0.2714 | 0.5758 | 0.5822 |
| detect-secrets | 0.3594 | 0.3140 | 0.3352 | 0.0677 | — | 1893 | 0.1342 | 0.1880 | 0.4276 | 0.3621 |
| credsweeper-ml | 0.3713 | 0.6894 | 0.4827 | 0.1411 | — | 5163 | 0.1247 | 0.2112 | 0.4760 | 0.5632 |
| credsweeper-noml | 0.3115 | 0.8020 | 0.4487 | 0.2144 | — | 9692 | 0.0782 | 0.1425 | 0.3929 | 0.5275 |
| potion-lr | 0.2413 | 0.9096 | 0.3814 | 0.3459 | 0.0192 | — | — | — | — | — |
| lr-e2e | 0.6971 | 0.7304 | 0.7133 | 0.0384 | — | 2218 | 0.1943 | 0.3070 | 0.6952 | 0.7123 |
| model-e2e | 0.7325 | 0.7150 | 0.7237 | 0.0316 | — | 1969 | 0.2133 | 0.3286 | 0.7279 | 0.7214 |
| potion-lr-e2e | 0.3738 | 0.7304 | 0.4945 | 0.1480 | — | 6613 | 0.0659 | 0.1209 | 0.3686 | 0.4899 |
| union-e2e | 0.6851 | 0.7389 | 0.7110 | 0.0411 | — | 3275 | 0.2327 | 0.3539 | 0.7500 | 0.7444 |
| union-lr-e2e | 0.6561 | 0.7423 | 0.6966 | 0.0470 | — | 3685 | 0.2125 | 0.3304 | 0.7138 | 0.7278 |
| union-tau-v2-e2e | 0.6672 | 0.7457 | 0.7043 | 0.0450 | — | 3436 | 0.2247 | 0.3453 | 0.7324 | 0.7390 |

`<union>-e2e`: the default engine's findings on the texts and the verifier's accepted candidates of the generator (end to end); its candidate columns read the candidates by overlap.

| who flags the candidates, values as published | default only: positive / negative | verifier only | both | neither |
|---|---|---|---|---|
| union | 31 / 101 | 195 / 113 | 321 / 24 | 39 / 4608 |
| union-lr | 29 / 94 | 199 / 515 | 323 / 31 | 35 / 4206 |
| union-tau-v2 | 21 / 93 | 199 / 152 | 331 / 32 | 35 / 4569 |

| obfuscation sensitivity (candidate F1) | values as published | through CredData's `get_obfuscated_value` | difference |
|---|---:|---:|---:|
| union | 0.7980 | 0.8186 | -0.0207 |
| union-lr | 0.6201 | 0.6325 | -0.0123 |
| union-tau-v2 | 0.7793 | 0.8014 | -0.0220 |
| model | 0.8154 | 0.8525 | -0.0371 |
| lr | 0.6312 | 0.6627 | -0.0315 |
| secretgate-default | 0.6623 | 0.5990 | +0.0633 |
| generator | 0.3467 | 0.2836 | +0.0631 |
| gitleaks | 0.6805 | 0.6205 | +0.0599 |
| betterleaks | 0.5784 | 0.5084 | +0.0700 |
| detect-secrets | 0.3352 | 0.2852 | +0.0500 |
| credsweeper-ml | 0.4827 | 0.4065 | +0.0762 |
| credsweeper-noml | 0.4487 | 0.3627 | +0.0860 |
| potion-lr | 0.3814 | 0.3992 | -0.0178 |

## Clean real text (7.7)

Gated universe: the rows whose CredData category is exactly `UUID` are removed, for every system (amendment 2 of docs/s6-secrets2-prereg.md). That choice was made after reading the training and validation rows and before any test file was read. Full universe: every meta row.

Second look: v3 reads test files that were read once for v2 (docs/s6-secrets2-prereg.md, Result of 15/11/2026), and its system (the union of the default engine and the model, with tau3) was defined after that reading.

8825 files (list sha256 `b25d912775de152941d98462fcae3e96fc15dbb6292614adbfeb22083e74cc53`), read as the census read them (no source path, no pragma for the default engine): G7's reading.

| | files with an accepted flag | share | default engine only | verifier only | both | accepted flags |
|---|---:|---:|---:|---:|---:|---:|
| union | 30 | 0.34% | 3 | 19 | 8 | 57 + 93 |
| union-lr | 78 | 0.88% | 1 | 67 | 10 | 57 + 323 |
| union-tau-v2 | 34 | 0.39% | 3 | 23 | 8 | 57 + 110 |
| model | 31 | 0.35% | — | — | — | 110 |
| lr | 77 | 0.87% | — | — | — | 323 |
| secretgate-default | 11 | 0.12% | — | — | — | 57 |
| generator | 2161 | 24.49% | — | — | — | 14756 |
| potion-lr | 401 | 4.54% | — | — | — | 969 |

## Gate (docs/s6-secrets3-prereg.md section 3)

Gated universe: the rows whose CredData category is exactly `UUID` are removed, for every system (amendment 2 of docs/s6-secrets2-prereg.md). That choice was made after reading the training and validation rows and before any test file was read. Full universe: every meta row.

Second look: v3 reads test files that were read once for v2 (docs/s6-secrets2-prereg.md, Result of 15/11/2026), and its system (the union of the default engine and the model, with tau3) was defined after that reading.

- G1, every repository: holds (union 0.8963, best rule scanner secretgate-default 0.5197 + 0.15 = 0.6697; full universe 0.8012 against 0.4348)
- G1, without the largest repository: holds (union 0.8576, best rule scanner secretgate-default 0.6136 + 0.15 = 0.7636; full universe 0.7962 against 0.5378)
- G2, every repository: holds (union 0.8963, cbx-17m 0.8873 - 0.1 = 0.7873; full universe 0.8012 against 0.7724)
- G2, without the largest repository: holds (union 0.8576, cbx-17m 0.8191 - 0.1 = 0.7191; full universe 0.7962 against 0.7292)
- G3, every repository: holds (union 0.8963, union-lr 0.8661; full universe 0.8012 against 0.7608)
- G3, without the largest repository: holds (union 0.8576, union-lr 0.8049; full universe 0.7962 against 0.7264)
- G4: holds (issue-report candidate F1 of the union 0.7980, Gitleaks 0.6805 + 0.08 = 0.7605)
- G5: holds (ECE of the model 0.0135 on the gated universe, at most 0.05; copied from v2's validation gate)
- G6: holds (native median 1866 µs per candidate, at most 3000; copied from v2's validation gate)
- G7: holds (30 of 8825 clean files flagged by the union, 0.34%, at most 1.0%)
- gate: REHEARSAL on validation: the gate would hold on these numbers; nothing is decided on validation
