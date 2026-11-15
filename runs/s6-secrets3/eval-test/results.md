# s6-secrets v3 (secretgate's default engine and s6-secrets-verify-pico together): results on the test split (docs/s6-secrets3-prereg.md)

**Second look: v3 reads test files that were read once for v2 (docs/s6-secrets2-prereg.md, Result of 15/11/2026), and its system (the union of the default engine and the model, with tau3) was defined after that reading.** The reading of v2 does not change: it failed its gate on G1 and stays failed.

**The systems.** `union`: a line is flagged when secretgate's default engine (secretgate 4e93cfa, filters on) flags it, or when the model of v2 (the evaluated file, unchanged) accepts a candidate of the frozen generator on it with P(real) >= tau3 = 0.34933. `union-lr`: the same with the size-matched logistic regression in the model's seat, at its own union threshold 0.155049 (G3). `union-tau-v2`: the union at v2's tau* 0.2598 (reported). `model` is the model alone at tau*, v2's system. On the issue reports a dataset candidate is flagged when a default finding overlaps it or the verifier accepts it; on the clean text a file counts when either flags it. No scanner, model or baseline was rerun: every flag is the v2 evaluation's.

tau3 and the regression's union threshold were chosen on the 16 validation repositories by the rule of section 1 (`runs/s6-secrets3/tau3.json`, committed at 2026-11-15T09:05:11+01:00 in `370f60f`); validation line F1 of the union at tau3, gated: 0.8963. Test read for v3: 2026-11-15T09:07:55+01:00 tau3.json at 370f60f8d6c96c47bb54533f1ea96dfe67f48c2d.

221 repositories, 7372 files; T rows: 10463 (gated universe), 11603 (full). Largest repository: `2ba83c6a`, 5165 gated T rows.

## CredData, line level (7.1)

Gated universe: the rows whose CredData category is exactly `UUID` are removed, for every system (amendment 2 of docs/s6-secrets2-prereg.md). That choice was made after reading the training and validation rows and before any test file was read. Full universe: every meta row.

Second look: v3 reads test files that were read once for v2 (docs/s6-secrets2-prereg.md, Result of 15/11/2026), and its system (the union of the default engine and the model, with tau3) was defined after that reading.

| gated universe | precision | recall | F1 | false alarms | F1 without the largest | macro F1 (≥ 20 T rows) |
|---|---:|---:|---:|---:|---:|---:|
| union | 0.8039 | 0.8965 | 0.8477 | 0.0779 | 0.8062 | 0.8042 (51) |
| union-lr | 0.7171 | 0.8749 | 0.7882 | 0.1230 | 0.7171 | 0.7363 (51) |
| union-tau-v2 | 0.7872 | 0.9038 | 0.8415 | 0.0870 | 0.7990 | 0.8060 (51) |
| model | 0.8387 | 0.8741 | 0.8561 | 0.0599 | 0.8114 | 0.8091 (51) |
| lr | 0.7345 | 0.8652 | 0.7945 | 0.1114 | 0.7179 | 0.7287 (51) |
| secretgate-default | 0.8176 | 0.6560 | 0.7279 | 0.0521 | 0.5072 | 0.4878 (51) |
| generator | 0.2942 | 0.9418 | 0.4483 | 0.8053 | 0.3231 | 0.4967 (51) |
| gitleaks | 0.8546 | 0.5980 | 0.7037 | 0.0363 | 0.4080 | 0.3436 (51) |
| betterleaks | 0.7694 | 0.7002 | 0.7332 | 0.0748 | 0.5367 | 0.5606 (51) |
| detect-secrets | 0.5504 | 0.2329 | 0.3273 | 0.0678 | 0.5014 | 0.5299 (51) |
| credsweeper-ml | 0.9946 | 0.9925 | 0.9935 | 0.0019 | 0.9872 | 0.9888 (51) |
| credsweeper-noml | 0.4512 | 0.9970 | 0.6213 | 0.4320 | 0.5428 | 0.7153 (51) |
| cbx-17m | 0.8265 | 0.5820 | 0.6830 | 0.0435 | 0.8493 | 0.8133 (51) |
| potion-lr | 0.4966 | 0.8955 | 0.6389 | 0.3234 | 0.5536 | 0.6552 (51) |

| full universe | precision | recall | F1 | false alarms | F1 without the largest | macro F1 (≥ 20 T rows) |
|---|---:|---:|---:|---:|---:|---:|
| union | 0.8049 | 0.8123 | 0.8086 | 0.0770 | 0.7314 | 0.7058 (60) |
| union-lr | 0.7171 | 0.7916 | 0.7525 | 0.1222 | 0.6518 | 0.6501 (60) |
| union-tau-v2 | 0.7883 | 0.8190 | 0.8034 | 0.0860 | 0.7268 | 0.7092 (60) |
| model | 0.8398 | 0.7922 | 0.8153 | 0.0591 | 0.7348 | 0.7113 (60) |
| lr | 0.7345 | 0.7829 | 0.7579 | 0.1107 | 0.6510 | 0.6446 (60) |
| secretgate-default | 0.8176 | 0.5935 | 0.6877 | 0.0518 | 0.4445 | 0.4166 (60) |
| generator | 0.2954 | 0.8606 | 0.4399 | 0.8029 | 0.3158 | 0.4477 (60) |
| gitleaks | 0.8546 | 0.5410 | 0.6626 | 0.0360 | 0.3507 | 0.2950 (60) |
| betterleaks | 0.7694 | 0.6334 | 0.6948 | 0.0743 | 0.4786 | 0.4915 (60) |
| detect-secrets | 0.5530 | 0.2129 | 0.3075 | 0.0673 | 0.4534 | 0.4715 (60) |
| credsweeper-ml | 0.9963 | 0.9914 | 0.9938 | 0.0014 | 0.9889 | 0.9898 (60) |
| credsweeper-noml | 0.4761 | 0.9954 | 0.6441 | 0.4285 | 0.5891 | 0.7360 (60) |
| cbx-17m | 0.8266 | 0.5265 | 0.6433 | 0.0432 | 0.7614 | 0.7165 (60) |
| potion-lr | 0.4956 | 0.8135 | 0.6160 | 0.3238 | 0.5172 | 0.5844 (60) |

CredSweeper's ML was trained on a random 85% of all CredData lines: its CredData numbers are in-sample and enter no gate (section 6 of v2).

## Lines the union adds over the model alone (reported, section 4)

Gated universe: the rows whose CredData category is exactly `UUID` are removed, for every system (amendment 2 of docs/s6-secrets2-prereg.md). That choice was made after reading the training and validation rows and before any test file was read. Full universe: every meta row.

Second look: v3 reads test files that were read once for v2 (docs/s6-secrets2-prereg.md, Result of 15/11/2026), and its system (the union of the default engine and the model, with tau3) was defined after that reading.

Line units flagged by the union and not by the model alone at tau* (added), and the reverse (dropped), by label: T a positive line, X a negative line with an X row, F the other negative lines.

| universe | union | threshold | added T | added X | added F | dropped T | dropped X | dropped F | lines the default engine flags | line units |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gated | union | 0.3493 | 305 | 135 | 649 | 75 | 39 | 224 | 8262 | 39196 |
| gated | union-tau-v2 | 0.2598 | 305 | 135 | 649 | 0 | 0 | 0 | 8262 | 39196 |
| full | union | 0.3493 | 306 | 135 | 649 | 77 | 39 | 223 | 8263 | 40487 |
| full | union-tau-v2 | 0.2598 | 306 | 135 | 649 | 0 | 0 | 0 | 8263 | 40487 |

## Unlabelled-line alarms (7.3), versions and CPU time

Gated universe: the rows whose CredData category is exactly `UUID` are removed, for every system (amendment 2 of docs/s6-secrets2-prereg.md). That choice was made after reading the training and validation rows and before any test file was read. Full universe: every meta row.

Second look: v3 reads test files that were read once for v2 (docs/s6-secrets2-prereg.md, Result of 15/11/2026), and its system (the union of the default engine and the model, with tau3) was defined after that reading.

| | version | unit | unlabelled-line alarms: flags | lines | flags per 1,000 files | CPU s |
|---|---|---|---:|---:|---:|---:|
| union | secretgate 4e93cfa2ee6e216caca9bc8e6f74cb565a991350 or model (full-depth.gguf sha256 01e6db3fa71cc2c5563dea222c68e040d3994ee2abbb02b25491a51b8cb54f27) at P(real) >= 0.34933 | line | 775 | 625 | 105.1 | — |
| union-lr | secretgate 4e93cfa2ee6e216caca9bc8e6f74cb565a991350 or lr (scikit-learn 1.9.1, SGDClassifier alpha 0.0001, 2^18 int8 weights) at P(real) >= 0.155049 | line | 1290 | 1099 | 175.0 | — |
| union-tau-v2 | secretgate 4e93cfa2ee6e216caca9bc8e6f74cb565a991350 or model (full-depth.gguf sha256 01e6db3fa71cc2c5563dea222c68e040d3994ee2abbb02b25491a51b8cb54f27) at P(real) >= 0.2598114609718323 | line | 960 | 803 | 130.2 | — |
| model | full-depth.gguf sha256 01e6db3fa71cc2c5563dea222c68e040d3994ee2abbb02b25491a51b8cb54f27 | candidate | 735 | 731 | 99.7 | — |
| lr | scikit-learn 1.9.1, SGDClassifier alpha 0.0001, 2^18 int8 weights | candidate | 1065 | 1061 | 144.5 | 12.5 |
| secretgate-default | secretgate 4e93cfa2ee6e216caca9bc8e6f74cb565a991350 | span | 225 | 221 | 30.5 | 33.4 |
| generator | sg-cand-1 revision 1, secretgate 4e93cfa2ee6e216caca9bc8e6f74cb565a991350 | candidate | 32951 | 31984 | 4469.8 | 77.1 |
| gitleaks | Gitleaks 8.30.1 | span | 199 | 196 | 27.0 | 121.8 |
| betterleaks | Betterleaks 1.10.1 | span | 288 | 285 | 39.1 | 8.8 |
| detect-secrets | detect-secrets 1.5.0 | line | 2407 | 2204 | 326.5 | 704.7 |
| credsweeper-ml | CredSweeper 1.19.6 | span | 0 | 0 | 0.0 | 300.7 |
| credsweeper-noml | CredSweeper 1.19.6 | span | 0 | 0 | 0.0 | 79.2 |
| cbx-17m | cbx-secrets-detector-17M at 89362a0938f9459e5706e703e2a91c5cf2ffe40e | span | 3587 | 2650 | 486.6 | 57098.4 |
| potion-lr | model2vec 0.10.0 potion-base-2M, scikit-learn 1.9.1 LogisticRegression | candidate | 6576 | 6347 | 892.0 | 6.6 |

A union's flags are the default engine's plus the verifier's accepted candidates: a line both flag counts twice in the flags column, once in the lines column.

## CredData, candidate level (7.2): the scored candidates of v2

Gated universe: the rows whose CredData category is exactly `UUID` are removed, for every system (amendment 2 of docs/s6-secrets2-prereg.md). That choice was made after reading the training and validation rows and before any test file was read. Full universe: every meta row.

Second look: v3 reads test files that were read once for v2 (docs/s6-secrets2-prereg.md, Result of 15/11/2026), and its system (the union of the default engine and the model, with tau3) was defined after that reading.

The union is a line system (the default engine gives no score), so it has no candidate-level row; the model's and the baselines' rows are v2's.

| gated universe | candidates | positives | threshold | precision | recall | F1 | PR-AUC | recall at 1% false alarms | ECE | ECE on P(real) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| model | 33809 | 9623 | 0.2598 | 0.8363 | 0.9344 | 0.8827 | 0.9335 | 0.5556 | 0.0073 | 0.0101 |
| lr | 33809 | 9623 | 0.1550 | 0.7331 | 0.9282 | 0.8192 | 0.8648 | 0.2657 | 0.0786 | 0.1112 |
| generator | 33809 | 9623 | — | 0.2846 | 1.0000 | 0.4431 | — | — | — | — |
| potion-lr | 33809 | 9623 | 0.0786 | 0.4882 | 0.9504 | 0.6451 | 0.7103 | 0.0924 | 0.0972 | 0.1144 |

| full universe | candidates | positives | threshold | precision | recall | F1 | PR-AUC | recall at 1% false alarms | ECE | ECE on P(real) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| model | 33924 | 9641 | 0.2598 | 0.8374 | 0.9343 | 0.8832 | 0.9335 | 0.5596 | 0.0072 | 0.0105 |
| lr | 33924 | 9641 | 0.1550 | 0.7331 | 0.9266 | 0.8185 | 0.8644 | 0.2652 | 0.0786 | 0.1111 |
| generator | 33924 | 9641 | — | 0.2842 | 1.0000 | 0.4426 | — | — | — | — |
| potion-lr | 33924 | 9641 | 0.0786 | 0.4865 | 0.9501 | 0.6435 | 0.7091 | 0.0920 | 0.0970 | 0.1140 |

## CredData, span level (section 9 of v2, overlap with a T value)

Gated universe: the rows whose CredData category is exactly `UUID` are removed, for every system (amendment 2 of docs/s6-secrets2-prereg.md). That choice was made after reading the training and validation rows and before any test file was read. Full universe: every meta row.

Second look: v3 reads test files that were read once for v2 (docs/s6-secrets2-prereg.md, Result of 15/11/2026), and its system (the union of the default engine and the model, with tau3) was defined after that reading.

| | gated recall | gated precision | gated F1 | full recall | full precision | full F1 | flags: T / X / F / no row / unlabelled |
|---|---:|---:|---:|---:|---:|---:|---|
| union | 0.8888 | 0.8397 | 0.8636 | 0.8029 | 0.8403 | 0.8212 | 15615 / 384 / 2533 / 51 / 775 |
| union-lr | 0.8690 | 0.7663 | 0.8144 | 0.7837 | 0.7662 | 0.7748 | 15689 / 600 / 4164 / 23 / 1290 |
| union-tau-v2 | 0.8945 | 0.8248 | 0.8582 | 0.8082 | 0.8254 | 0.8167 | 15764 / 451 / 2809 / 75 / 960 |
| model | 0.8656 | 0.8309 | 0.8479 | 0.7820 | 0.8319 | 0.8062 | 9641 / 1601 / 22682 / 1235 / 32951 |
| lr | 0.8593 | 0.7321 | 0.7906 | 0.7750 | 0.7319 | 0.7528 | 9641 / 1601 / 22682 / 1235 / 32951 |
| secretgate-default | 0.6507 | 0.8168 | 0.7243 | 0.5868 | 0.8168 | 0.6830 | 6756 / 221 / 1290 / 4 / 225 |
| generator | 0.9260 | 0.2752 | 0.4243 | 0.8367 | 0.2742 | 0.4131 | 9641 / 1601 / 22682 / 1235 / 32951 |
| gitleaks | 0.5938 | 0.8544 | 0.7006 | 0.5356 | 0.8544 | 0.6584 | 6167 / 16 / 1034 / 1 / 199 |
| betterleaks | 0.6937 | 0.7686 | 0.7292 | 0.6256 | 0.7686 | 0.6898 | 7220 / 831 / 1252 / 91 / 288 |
| credsweeper-ml | 0.9923 | 0.9960 | 0.9941 | 0.9912 | 0.9973 | 0.9943 | 22371 / 1 / 59 / 0 / 0 |
| credsweeper-noml | 0.9964 | 0.6170 | 0.7621 | 0.9949 | 0.6295 | 0.7711 | 22525 / 638 / 12618 / 0 / 0 |
| cbx-17m | 0.5843 | 0.8103 | 0.6790 | 0.5270 | 0.8104 | 0.6387 | 7061 / 21 / 1558 / 73 / 3587 |
| potion-lr | 0.8795 | 0.4744 | 0.6163 | 0.7944 | 0.4722 | 0.5923 | 9641 / 1601 / 22682 / 1235 / 32951 |

The flag counts of a scored system are of all its candidates; those of a union, of its flags: the default engine's and the verifier's accepted candidates.

## By CredData category and by path scope (section 9 of v2), gated universe, line F1 (recall)

Gated universe: the rows whose CredData category is exactly `UUID` are removed, for every system (amendment 2 of docs/s6-secrets2-prereg.md). That choice was made after reading the training and validation rows and before any test file was read. Full universe: every meta row.

Second look: v3 reads test files that were read once for v2 (docs/s6-secrets2-prereg.md, Result of 15/11/2026), and its system (the union of the default engine and the model, with tau3) was defined after that reading.

| | Password | Token | Key | Secret | API | Auth | other | scope test | scope src | scope other |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| union | 0.806 (0.891) | 0.826 (0.768) | 0.838 (0.970) | 0.941 (0.959) | 0.551 (0.488) | 0.882 (0.838) | 0.882 (0.815) | 0.871 (0.942) | 0.788 (0.845) | 0.757 (0.733) |
| union-lr | 0.669 (0.802) | 0.792 (0.722) | 0.783 (0.960) | 0.906 (0.932) | 0.356 (0.383) | 0.844 (0.816) | 0.867 (0.796) | 0.832 (0.931) | 0.634 (0.845) | 0.640 (0.668) |
| union-tau-v2 | 0.798 (0.905) | 0.824 (0.780) | 0.830 (0.975) | 0.939 (0.961) | 0.537 (0.488) | 0.881 (0.858) | 0.883 (0.823) | 0.868 (0.948) | 0.741 (0.858) | 0.749 (0.743) |
| model | 0.838 (0.893) | 0.824 (0.766) | 0.864 (0.957) | 0.905 (0.860) | 0.541 (0.469) | 0.881 (0.852) | 0.873 (0.799) | 0.884 (0.915) | 0.767 (0.852) | 0.756 (0.722) |
| lr | 0.689 (0.800) | 0.746 (0.653) | 0.795 (0.959) | 0.909 (0.922) | 0.358 (0.378) | 0.837 (0.804) | 0.854 (0.768) | 0.842 (0.926) | 0.567 (0.710) | 0.643 (0.658) |
| secretgate-default | 0.406 (0.309) | 0.685 (0.534) | 0.822 (0.869) | 0.912 (0.880) | 0.439 (0.301) | 0.258 (0.150) | 0.653 (0.497) | 0.779 (0.741) | 0.675 (0.587) | 0.480 (0.344) |
| generator | 0.459 (0.985) | 0.298 (0.811) | 0.400 (0.991) | 0.694 (0.995) | 0.114 (0.512) | 0.568 (0.981) | 0.769 (0.858) | 0.577 (0.978) | 0.189 (0.918) | 0.250 (0.807) |
| gitleaks | 0.099 (0.053) | 0.668 (0.511) | 0.824 (0.870) | 0.902 (0.825) | 0.419 (0.273) | 0.229 (0.130) | 0.628 (0.464) | 0.764 (0.694) | 0.667 (0.536) | 0.383 (0.242) |
| betterleaks | 0.545 (0.642) | 0.672 (0.517) | 0.838 (0.867) | 0.897 (0.815) | 0.459 (0.306) | 0.251 (0.146) | 0.638 (0.496) | 0.770 (0.754) | 0.667 (0.681) | 0.583 (0.499) |
| detect-secrets | 0.612 (0.692) | 0.427 (0.279) | 0.128 (0.075) | 0.321 (0.250) | 0.282 (0.287) | 0.203 (0.119) | 0.333 (0.213) | 0.260 (0.165) | 0.465 (0.416) | 0.476 (0.463) |
| credsweeper-ml | 0.991 (0.983) | 0.994 (0.990) | 0.994 (0.997) | 0.996 (0.995) | 0.978 (0.971) | 0.996 (0.993) | 0.991 (0.993) | 0.997 (0.996) | 0.959 (0.994) | 0.986 (0.980) |
| credsweeper-noml | 0.648 (0.994) | 0.555 (0.997) | 0.532 (1.000) | 0.898 (1.000) | 0.267 (0.995) | 0.670 (0.998) | 0.916 (0.995) | 0.692 (0.998) | 0.378 (1.000) | 0.481 (0.991) |
| cbx-17m | 0.904 (0.838) | 0.929 (0.874) | 0.266 (0.197) | 0.945 (0.897) | 0.898 (0.818) | 0.936 (0.885) | 0.876 (0.801) | 0.634 (0.530) | 0.880 (0.836) | 0.826 (0.741) |
| potion-lr | 0.547 (0.883) | 0.633 (0.716) | 0.607 (0.963) | 0.836 (0.954) | 0.232 (0.435) | 0.795 (0.923) | 0.804 (0.822) | 0.700 (0.943) | 0.449 (0.861) | 0.471 (0.720) |
| positive / negative lines | 1714 / 6394 | 777 / 3288 | 4029 / 13092 | 1389 / 1603 | 209 / 1987 | 1016 / 2078 | 2966 / 1587 | 7903 / 13446 | 317 / 3045 | 2077 / 12408 |

## Uncertainty (7.6): paired bootstrap over repositories, 1000 resamples, 95%

Gated universe: the rows whose CredData category is exactly `UUID` are removed, for every system (amendment 2 of docs/s6-secrets2-prereg.md). That choice was made after reading the training and validation rows and before any test file was read. Full universe: every meta row.

Second look: v3 reads test files that were read once for v2 (docs/s6-secrets2-prereg.md, Result of 15/11/2026), and its system (the union of the default engine and the model, with tau3) was defined after that reading.

| line F1 of union minus | gated | gated, without the largest | full | full, without the largest |
|---|---|---|---|---|
| union-lr | +0.0595 [0.0418, 0.1179] | +0.0891 [0.0615, 0.1287] | +0.0560 [0.0403, 0.1015] | +0.0796 [0.0564, 0.1115] |
| union-tau-v2 | +0.0062 [0.0008, 0.0141] | +0.0071 [-0.0003, 0.0168] | +0.0052 [-0.0007, 0.0101] | +0.0046 [-0.0018, 0.0120] |
| model | -0.0084 [-0.0181, 0.0072] | -0.0052 [-0.0188, 0.0118] | -0.0067 [-0.0158, 0.0074] | -0.0034 [-0.0162, 0.0112] |
| lr | +0.0531 [0.0317, 0.1216] | +0.0883 [0.0496, 0.1344] | +0.0506 [0.0310, 0.1068] | +0.0804 [0.0477, 0.1199] |
| secretgate-default | +0.1197 [0.0370, 0.3806] | +0.2990 [0.1994, 0.4075] | +0.1208 [0.0391, 0.3684] | +0.2869 [0.1936, 0.3938] |
| generator | +0.3994 [0.3117, 0.5232] | +0.4831 [0.4288, 0.5362] | +0.3687 [0.3008, 0.4585] | +0.4156 [0.3632, 0.4708] |
| gitleaks | +0.1440 [0.0478, 0.4758] | +0.3982 [0.2921, 0.5019] | +0.1460 [0.0499, 0.4629] | +0.3807 [0.2788, 0.4874] |
| betterleaks | +0.1145 [0.0320, 0.3465] | +0.2695 [0.1785, 0.3625] | +0.1137 [0.0339, 0.3305] | +0.2528 [0.1670, 0.3480] |
| detect-secrets | +0.5204 [0.2313, 0.6903] | +0.3048 [0.2200, 0.3922] | +0.5011 [0.2074, 0.6791] | +0.2780 [0.1957, 0.3672] |
| credsweeper-ml | -0.1459 [-0.2164, -0.1262] | -0.1810 [-0.2307, -0.1478] | -0.1853 [-0.3135, -0.1421] | -0.2575 [-0.3266, -0.1987] |
| credsweeper-noml | +0.2264 [0.1831, 0.3151] | +0.2634 [0.1965, 0.3273] | +0.1645 [0.0751, 0.2087] | +0.1423 [0.0577, 0.2227] |
| cbx-17m | +0.1646 [-0.0730, 0.2973] | -0.0431 [-0.0819, -0.0059] | +0.1652 [-0.0593, 0.2959] | -0.0300 [-0.0703, 0.0054] |
| potion-lr | +0.2088 [0.1716, 0.3059] | +0.2526 [0.1944, 0.3214] | +0.1926 [0.1637, 0.2548] | +0.2142 [0.1687, 0.2672] |

| ECE (scored candidates of v2) | gated | full |
|---|---|---|
| model | 0.0073 [0.0040, 0.0174] | 0.0072 [0.0041, 0.0174] |
| lr | 0.0786 [0.0235, 0.1309] | 0.0786 [0.0237, 0.1305] |
| potion-lr | 0.0972 [0.0168, 0.1661] | 0.0970 [0.0167, 0.1655] |

## Issue reports, test split (7.4)

Gated universe: the rows whose CredData category is exactly `UUID` are removed, for every system (amendment 2 of docs/s6-secrets2-prereg.md). That choice was made after reading the training and validation rows and before any test file was read. Full universe: every meta row.

Second look: v3 reads test files that were read once for v2 (docs/s6-secrets2-prereg.md, Result of 15/11/2026), and its system (the union of the default engine and the model, with tau3) was defined after that reading.

8123 candidates (888 secrets) in 7986 texts; 774 occur several times in their text (the first occurrence is the candidate), 0 are not found in it.

| | precision | recall | F1 | false alarms | ECE | end to end: flags | precision over all flags | F1 | precision over flags on a candidate | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| union | 0.7178 | 0.9223 | 0.8073 | 0.0445 | — | — | — | — | — | — |
| union-lr | 0.4811 | 0.9313 | 0.6344 | 0.1233 | — | — | — | — | — | — |
| union-tau-v2 | 0.6767 | 0.9358 | 0.7854 | 0.0549 | — | — | — | — | — | — |
| model | 0.7484 | 0.8975 | 0.8162 | 0.0370 | 0.0211 | — | — | — | — | — |
| lr | 0.5042 | 0.8874 | 0.6430 | 0.1071 | 0.0379 | — | — | — | — | — |
| secretgate-default | 0.7699 | 0.5991 | 0.6738 | 0.0220 | — | 2295 | 0.2301 | 0.3325 | 0.7719 | 0.6746 |
| generator | 0.2198 | 0.7883 | 0.3437 | 0.3435 | — | 25931 | 0.0271 | 0.0523 | 0.2078 | 0.3289 |
| gitleaks | 0.9028 | 0.5856 | 0.7104 | 0.0077 | — | 1817 | 0.2966 | 0.3938 | 0.9059 | 0.7113 |
| betterleaks | 0.6102 | 0.6171 | 0.6137 | 0.0484 | — | 3318 | 0.1733 | 0.2706 | 0.6216 | 0.6194 |
| detect-secrets | 0.3461 | 0.2950 | 0.3185 | 0.0684 | — | 3212 | 0.1083 | 0.1585 | 0.4104 | 0.3433 |
| credsweeper-ml | 0.3740 | 0.7185 | 0.4919 | 0.1476 | — | 8345 | 0.1190 | 0.2042 | 0.4788 | 0.5746 |
| credsweeper-noml | 0.3087 | 0.8074 | 0.4466 | 0.2220 | — | 16104 | 0.0720 | 0.1322 | 0.3994 | 0.5344 |
| cbx-17m | 0.8830 | 0.8412 | 0.8616 | 0.0137 | — | 5197 | 0.1537 | 0.2600 | 0.8829 | 0.8615 |
| potion-lr | 0.2417 | 0.9223 | 0.3831 | 0.3551 | 0.0198 | — | — | — | — | — |
| lr-e2e | 0.7030 | 0.7331 | 0.7178 | 0.0380 | — | 3738 | 0.1747 | 0.2822 | 0.7044 | 0.7185 |
| model-e2e | 0.7607 | 0.7196 | 0.7396 | 0.0278 | — | 3068 | 0.2073 | 0.3219 | 0.7626 | 0.7405 |
| potion-lr-e2e | 0.3683 | 0.7230 | 0.4880 | 0.1522 | — | 10245 | 0.0628 | 0.1155 | 0.3612 | 0.4818 |
| union-e2e | 0.7065 | 0.7399 | 0.7228 | 0.0377 | — | 5097 | 0.2250 | 0.3451 | 0.7835 | 0.7610 |
| union-lr-e2e | 0.6670 | 0.7466 | 0.7046 | 0.0457 | — | 6033 | 0.1958 | 0.3102 | 0.7331 | 0.7398 |
| union-tau-v2-e2e | 0.6894 | 0.7500 | 0.7184 | 0.0415 | — | 5363 | 0.2170 | 0.3367 | 0.7668 | 0.7583 |

`<union>-e2e`: the default engine's findings on the texts and the verifier's accepted candidates of the generator (end to end); its candidate columns read the candidates by overlap.

| who flags the candidates, values as published | default only: positive / negative | verifier only | both | neither |
|---|---|---|---|---|
| union | 44 / 132 | 287 / 163 | 488 / 27 | 69 / 6913 |
| union-lr | 39 / 117 | 295 / 733 | 493 / 42 | 61 / 6343 |
| union-tau-v2 | 34 / 129 | 299 / 238 | 498 / 30 | 57 / 6838 |

| obfuscation sensitivity (candidate F1) | values as published | through CredData's `get_obfuscated_value` | difference |
|---|---:|---:|---:|
| union | 0.8073 | 0.8235 | -0.0162 |
| union-lr | 0.6344 | 0.6541 | -0.0196 |
| union-tau-v2 | 0.7854 | 0.8035 | -0.0181 |
| model | 0.8162 | 0.8498 | -0.0336 |
| lr | 0.6430 | 0.6791 | -0.0361 |
| secretgate-default | 0.6738 | 0.6142 | +0.0597 |
| generator | 0.3437 | 0.2883 | +0.0555 |
| gitleaks | 0.7104 | 0.6485 | +0.0618 |
| betterleaks | 0.6137 | 0.5401 | +0.0736 |
| detect-secrets | 0.3185 | 0.2766 | +0.0419 |
| credsweeper-ml | 0.4919 | 0.4188 | +0.0731 |
| credsweeper-noml | 0.4466 | 0.3710 | +0.0756 |
| cbx-17m | 0.8616 | 0.8022 | +0.0594 |
| potion-lr | 0.3831 | 0.3906 | -0.0075 |

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

- G1, every repository: fails (union 0.8477, best rule scanner betterleaks 0.7332 + 0.15 = 0.8832; full universe 0.8086 against 0.6948)
- G1, without the largest repository: holds (union 0.8062, best rule scanner betterleaks 0.5367 + 0.15 = 0.6867; full universe 0.7314 against 0.4786)
- G2, every repository: holds (union 0.8477, cbx-17m 0.6830 - 0.1 = 0.5830; full universe 0.8086 against 0.6433)
- G2, without the largest repository: holds (union 0.8062, cbx-17m 0.8493 - 0.1 = 0.7493; full universe 0.7314 against 0.7614)
- G3, every repository: holds (union 0.8477, union-lr 0.7882; full universe 0.8086 against 0.7525)
- G3, without the largest repository: holds (union 0.8062, union-lr 0.7171; full universe 0.7314 against 0.6518)
- G4: holds (issue-report candidate F1 of the union 0.8073, Gitleaks 0.7104 + 0.08 = 0.7904)
- G5: holds (ECE of the model 0.0073 on the gated universe, at most 0.05; copied from v2's test gate)
- G6: holds (native median 1606 µs per candidate, at most 3000; copied from v2's test gate)
- G7: holds (30 of 8825 clean files flagged by the union, 0.34%, at most 1.0%)
- gate: FAIL on a second look (G1): the work on this model ends, no third reading (section 3)
