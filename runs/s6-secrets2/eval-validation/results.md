# s6-secrets v2 (s6-secrets-verify-pico): results on the validation split (docs/s6-secrets2-prereg.md)

**Rehearsal.** These are the validation repositories, not the test: no number below is a result of the pre-registration, and the gate at the end is the test's computation run on validation.

No system named `model` was scored: the baselines are reported, the gate has no model to judge.

16 repositories, 452 files; T rows: 522 (gated universe), 659 (full). Largest repository: `7738e44d`, 217 gated T rows (the split's own largest, a stand-in for the test's).

## CredData, line level (7.1)

Gated universe: the rows whose CredData category is exactly `UUID` are removed, for every system (amendment 2). That choice was made after reading the training and validation rows and before any test file was read. Full universe: every meta row.

| gated universe | precision | recall | F1 | false alarms | F1 without the largest | macro F1 (≥ 20 T rows) |
|---|---:|---:|---:|---:|---:|---:|
| generator | 0.2753 | 0.9540 | 0.4272 | 0.5848 | 0.3236 | 0.6393 (6) |
| secretgate-default | 0.9270 | 0.3611 | 0.5197 | 0.0066 | 0.6136 | 0.5118 (6) |
| gitleaks | 0.9850 | 0.2867 | 0.4441 | 0.0010 | 0.5114 | 0.3793 (6) |
| betterleaks | 0.7500 | 0.3151 | 0.4438 | 0.0245 | 0.5022 | 0.4386 (6) |
| detect-secrets | 0.6069 | 0.3851 | 0.4712 | 0.0581 | 0.6048 | 0.5435 (6) |
| credsweeper-ml | 0.9202 | 0.9847 | 0.9514 | 0.0199 | 0.9253 | 0.9476 (6) |
| credsweeper-noml | 0.3218 | 1.0000 | 0.4869 | 0.4906 | 0.3873 | 0.7434 (6) |
| cbx-17m | 0.9814 | 0.8096 | 0.8873 | 0.0036 | 0.8191 | 0.8473 (6) |
| lr | 0.8863 | 0.8359 | 0.8604 | 0.0250 | 0.7972 | 0.8268 (6) |
| potion-lr | 0.5243 | 0.8271 | 0.6418 | 0.1747 | 0.5325 | 0.6934 (6) |

| full universe | precision | recall | F1 | false alarms | F1 without the largest | macro F1 (≥ 20 T rows) |
|---|---:|---:|---:|---:|---:|---:|
| generator | 0.2842 | 0.7762 | 0.4161 | 0.5820 | 0.3287 | 0.6187 (6) |
| secretgate-default | 0.9270 | 0.2840 | 0.4348 | 0.0067 | 0.5378 | 0.4493 (6) |
| gitleaks | 0.9850 | 0.2255 | 0.3669 | 0.0010 | 0.4420 | 0.3237 (6) |
| betterleaks | 0.7500 | 0.2478 | 0.3726 | 0.0246 | 0.4419 | 0.4006 (6) |
| detect-secrets | 0.6069 | 0.3029 | 0.4041 | 0.0584 | 0.5466 | 0.4989 (6) |
| credsweeper-ml | 0.9550 | 0.9862 | 0.9704 | 0.0138 | 0.9519 | 0.9680 (6) |
| credsweeper-noml | 0.3788 | 0.9983 | 0.5492 | 0.4872 | 0.4361 | 0.7810 (6) |
| cbx-17m | 0.9814 | 0.6368 | 0.7724 | 0.0036 | 0.7292 | 0.7717 (6) |
| lr | 0.8863 | 0.6575 | 0.7549 | 0.0251 | 0.7186 | 0.7438 (6) |
| potion-lr | 0.5326 | 0.6609 | 0.5899 | 0.1726 | 0.5094 | 0.6528 (6) |

| | version | unit | unlabelled-line alarms | per 1,000 files (7.3) | CPU s |
|---|---|---|---:|---:|---:|
| generator | sg-cand-1 revision 1, secretgate 4e93cfa2ee6e216caca9bc8e6f74cb565a991350 | candidate | 2279 | 5042.0 | 17.2 |
| secretgate-default | secretgate 4e93cfa2ee6e216caca9bc8e6f74cb565a991350 | span | 88 | 194.7 | 8.7 |
| gitleaks | Gitleaks 8.30.1 | span | 1 | 2.2 | 20.6 |
| betterleaks | Betterleaks 1.10.1 | span | 12 | 26.5 | 2.2 |
| detect-secrets | detect-secrets 1.5.0 | line | 50 | 110.6 | 115.0 |
| credsweeper-ml | CredSweeper 1.19.6 | span | 0 | 0.0 | 43.9 |
| credsweeper-noml | CredSweeper 1.19.6 | span | 0 | 0.0 | 24.5 |
| cbx-17m | cbx-secrets-detector-17M at 89362a0938f9459e5706e703e2a91c5cf2ffe40e | span | 2074 | 4588.5 | 13816.6 |
| lr | scikit-learn 1.9.1, SGDClassifier alpha 0.0001, 2^18 int8 weights | candidate | 176 | 389.4 | 1.3 |
| potion-lr | model2vec 0.10.0 potion-base-2M, scikit-learn 1.9.1 LogisticRegression | candidate | 437 | 966.8 | 0.7 |

CredSweeper's ML was trained on a random 85% of all CredData lines: its CredData numbers are in-sample and enter no gate (section 6).

## CredData, candidate level (7.2)

Gated universe: the rows whose CredData category is exactly `UUID` are removed, for every system (amendment 2). That choice was made after reading the training and validation rows and before any test file was read. Full universe: every meta row.

| gated universe | candidates | positives | threshold | precision | recall | F1 | PR-AUC | recall at 1% false alarms | ECE | ECE on P(real) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| generator | 1779 | 448 | — | 0.2518 | 1.0000 | 0.4023 | — | — | — | — |
| lr | 1779 | 448 | 0.1550 | 0.8810 | 0.8594 | 0.8701 | 0.9145 | 0.6451 | 0.0678 | 0.1183 |
| potion-lr | 1779 | 448 | 0.0786 | 0.5039 | 0.8638 | 0.6365 | 0.6340 | 0.1027 | 0.1081 | 0.1221 |

| full universe | candidates | positives | threshold | precision | recall | F1 | PR-AUC | recall at 1% false alarms | ECE | ECE on P(real) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| generator | 1779 | 459 | — | 0.2580 | 1.0000 | 0.4102 | — | — | — | — |
| lr | 1779 | 459 | 0.1550 | 0.8810 | 0.8388 | 0.8594 | 0.9036 | 0.6296 | 0.0671 | 0.1180 |
| potion-lr | 1779 | 459 | 0.0786 | 0.5117 | 0.8562 | 0.6406 | 0.6327 | 0.1002 | 0.1143 | 0.1282 |

Candidates on a labelled line that overlap no labelled value are dropped (3.1); ECE is over 15 equal-width bins of the answer's confidence, as sky-eval computes it.

## CredData, span level (section 9, overlap with a T value)

Gated universe: the rows whose CredData category is exactly `UUID` are removed, for every system (amendment 2). That choice was made after reading the training and validation rows and before any test file was read. Full universe: every meta row.

| | gated recall | gated precision | gated F1 | full recall | full precision | full F1 | flags: T / X / F / no row / unlabelled |
|---|---:|---:|---:|---:|---:|---:|---|
| generator | 0.8506 | 0.2295 | 0.3615 | 0.6904 | 0.2348 | 0.3504 | 459 / 117 / 1203 / 176 / 2279 |
| secretgate-default | 0.3238 | 0.9270 | 0.4799 | 0.2564 | 0.9270 | 0.4018 | 165 / 2 / 10 / 1 / 88 |
| gitleaks | 0.2586 | 0.9851 | 0.4097 | 0.2049 | 0.9851 | 0.3392 | 132 / 0 / 2 / 0 / 1 |
| betterleaks | 0.2816 | 0.7164 | 0.4043 | 0.2231 | 0.7164 | 0.3402 | 144 / 13 / 40 / 4 / 12 |
| credsweeper-ml | 0.8851 | 0.9257 | 0.9049 | 0.9059 | 0.9524 | 0.9286 | 820 / 0 / 41 / 0 / 0 |
| credsweeper-noml | 0.8966 | 0.3742 | 0.5280 | 0.9150 | 0.4252 | 0.5806 | 839 / 27 / 1107 / 0 / 0 |
| cbx-17m | 0.8257 | 0.9651 | 0.8900 | 0.6540 | 0.9651 | 0.7797 | 443 / 0 / 12 / 4 / 2074 |
| lr | 0.7452 | 0.8750 | 0.8049 | 0.5903 | 0.8750 | 0.7050 | 459 / 117 / 1203 / 176 / 2279 |
| potion-lr | 0.7414 | 0.4731 | 0.5776 | 0.5964 | 0.4804 | 0.5322 | 459 / 117 / 1203 / 176 / 2279 |

The generator's row is its recall (2.2): the share of T rows a candidate overlaps.

## By CredData category and by path scope (section 9), gated universe, line F1 (recall)

| | Password | Token | Key | Secret | API | Auth | other | scope test | scope src | scope other |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| generator | 0.575 (0.970) | 0.550 (1.000) | 0.140 (0.938) | 0.383 (1.000) | 0.136 (1.000) | 0.455 (0.979) | 0.814 (0.903) | 0.388 (0.909) | 0.452 (1.000) | 0.464 (0.991) |
| secretgate-default | 0.444 (0.296) | 0.552 (0.381) | 0.737 (0.583) | 0.755 (0.625) | 0.750 (0.857) | 0.255 (0.149) | 0.574 (0.414) | 0.667 (0.519) | 0.600 (0.429) | 0.351 (0.217) |
| gitleaks | 0.349 (0.215) | 0.500 (0.333) | 0.609 (0.438) | 0.609 (0.438) | 0.000 (0.000) | 0.082 (0.043) | 0.520 (0.352) | 0.586 (0.418) | 0.526 (0.357) | 0.285 (0.166) |
| betterleaks | 0.595 (0.556) | 0.507 (0.343) | 0.286 (0.167) | 0.476 (0.312) | 0.222 (0.143) | 0.157 (0.085) | 0.256 (0.159) | 0.429 (0.322) | 1.000 (1.000) | 0.408 (0.268) |
| detect-secrets | 0.747 (0.756) | 0.090 (0.048) | 0.364 (0.292) | 0.408 (0.656) | 0.636 (1.000) | 0.241 (0.149) | 0.427 (0.303) | 0.649 (0.601) | 0.444 (0.857) | 0.253 (0.166) |
| credsweeper-ml | 0.978 (0.978) | 0.986 (0.971) | 0.750 (1.000) | 0.968 (0.938) | 1.000 (1.000) | 0.979 (0.979) | 0.924 (1.000) | 0.904 (0.976) | 1.000 (1.000) | 0.994 (0.991) |
| credsweeper-noml | 0.736 (1.000) | 0.717 (1.000) | 0.142 (1.000) | 0.604 (1.000) | 0.099 (1.000) | 0.470 (1.000) | 0.863 (1.000) | 0.547 (1.000) | 0.075 (1.000) | 0.634 (1.000) |
| cbx-17m | 0.946 (0.904) | 0.951 (0.924) | 0.824 (0.729) | 0.918 (0.875) | 0.833 (0.714) | 0.860 (0.787) | 0.759 (0.621) | 0.760 (0.625) | 0.966 (1.000) | 0.976 (0.962) |
| lr | 0.746 (0.741) | 0.940 (0.895) | 0.813 (0.771) | 0.909 (0.938) | 0.778 (1.000) | 0.907 (0.936) | 0.910 (0.869) | 0.815 (0.793) | 0.966 (1.000) | 0.894 (0.864) |
| potion-lr | 0.597 (0.785) | 0.816 (0.867) | 0.400 (0.875) | 0.756 (0.969) | 0.177 (1.000) | 0.765 (0.936) | 0.787 (0.779) | 0.539 (0.712) | 0.788 (0.929) | 0.728 (0.923) |
| positive / negative lines | 135 / 282 | 105 / 233 | 48 / 1028 | 32 / 163 | 7 / 149 | 47 / 181 | 145 / 80 | 208 / 811 | 14 / 395 | 235 / 757 |

| ECE | gated | full |
|---|---|---|
| lr | 0.0678 [0.0289, 0.1369] | 0.0671 [0.0291, 0.1474] |
| potion-lr | 0.1081 [0.0342, 0.2369] | 0.1143 [0.0354, 0.2481] |

## Issue reports, validation split (7.4)

5432 candidates (586 secrets) in 5371 texts; 524 occur several times in their text (the first occurrence is the candidate), 0 are not found in it.

| | precision | recall | F1 | false alarms | ECE | end to end: flags | precision over all flags | F1 | precision over flags on a candidate | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| generator | 0.2222 | 0.7884 | 0.3467 | 0.3337 | — | 16426 | 0.0289 | 0.0558 | 0.2134 | 0.3359 |
| secretgate-default | 0.7379 | 0.6007 | 0.6623 | 0.0258 | — | 1467 | 0.2399 | 0.3429 | 0.7379 | 0.6623 |
| gitleaks | 0.8635 | 0.5614 | 0.6805 | 0.0107 | — | 1055 | 0.3166 | 0.4049 | 0.8630 | 0.6803 |
| betterleaks | 0.5684 | 0.5887 | 0.5784 | 0.0541 | — | 2024 | 0.1764 | 0.2714 | 0.5758 | 0.5822 |
| detect-secrets | 0.3594 | 0.3140 | 0.3352 | 0.0677 | — | 1893 | 0.1342 | 0.1880 | 0.4276 | 0.3621 |
| credsweeper-ml | 0.3713 | 0.6894 | 0.4827 | 0.1411 | — | 5163 | 0.1247 | 0.2112 | 0.4760 | 0.5632 |
| credsweeper-noml | 0.3115 | 0.8020 | 0.4487 | 0.2144 | — | 9692 | 0.0782 | 0.1425 | 0.3929 | 0.5275 |
| lr | 0.4888 | 0.8908 | 0.6312 | 0.1127 | 0.0335 | — | — | — | — | — |
| potion-lr | 0.2413 | 0.9096 | 0.3814 | 0.3459 | 0.0192 | — | — | — | — | — |
| lr-e2e | 0.6971 | 0.7304 | 0.7133 | 0.0384 | — | 2218 | 0.1943 | 0.3070 | 0.6952 | 0.7123 |
| potion-lr-e2e | 0.3738 | 0.7304 | 0.4945 | 0.1480 | — | 6613 | 0.0659 | 0.1209 | 0.3686 | 0.4899 |

| obfuscation sensitivity (candidate F1) | values as published | through CredData's `get_obfuscated_value` | difference |
|---|---:|---:|---:|
| generator | 0.3467 | 0.2836 | +0.0631 |
| secretgate-default | 0.6623 | 0.5990 | +0.0633 |
| gitleaks | 0.6805 | 0.6205 | +0.0599 |
| betterleaks | 0.5784 | 0.5084 | +0.0700 |
| detect-secrets | 0.3352 | 0.2852 | +0.0500 |
| credsweeper-ml | 0.4827 | 0.4065 | +0.0762 |
| credsweeper-noml | 0.4487 | 0.3627 | +0.0860 |
| lr | 0.6312 | 0.6627 | -0.0315 |
| potion-lr | 0.3814 | 0.3992 | -0.0178 |

585 positive candidate strings were passed through the function; 26 stay as published (the function failed on them).

## Clean real text (7.7)

8825 files (list sha256 `b25d912775de152941d98462fcae3e96fc15dbb6292614adbfeb22083e74cc53`).

| as the census: no source path, no pragma for the default engine (the gate's reading) | files with an accepted candidate | share | accepted |
|---|---:|---:|---:|
| generator | 2161 | 24.49% | 14756 |
| secretgate-default | 11 | 0.12% | 57 |
| lr | 77 | 0.87% | 323 |
| potion-lr | 401 | 4.54% | 969 |

| with the file's path | files with an accepted candidate | share | accepted |
|---|---:|---:|---:|
| generator | 187 | 2.12% | 1186 |
| secretgate-default | 7 | 0.08% | 47 |

## Value-only probe (section 9)

A logistic regression on the value's own features (len, entropy, chars_per_token, digits, upper, lower, other, word_run), trained on 19946 labelled CredData training candidates, scored on 1589 CredData validation candidates (384 positives): PR-AUC 0.8202 (the share of positives is 0.2417), ROC-AUC 0.9197, best F1 0.7200. A high score means the shape of a value alone separates T from F and X.

## Gate

- G1: fails (no result of model)
- G2: fails (no result of model)
- G3: fails (no result of model)
- G4: fails (no issue-report result of model)
- G5: fails (no ECE of model)
- G6: fails (no native latency)
- G7: fails (no clean-text result of model)
- gate: REHEARSAL on validation: the gate would not hold on these numbers; nothing is decided before the test
