# s6-secrets v2 (s6-secrets-verify-pico): results on the validation split (docs/s6-secrets2-prereg.md)

**Rehearsal.** These are the validation repositories, not the test: no number below is a result of the pre-registration, and the gate at the end is the test's computation run on validation.

**`lr` sits in the model's seat**: the gate lines below judge it as they will judge the model.

16 repositories, 452 files; T rows: 522 (gated universe), 659 (full). Largest repository: `7738e44d`, 217 gated T rows (the split's own largest, a stand-in for the test's).

## CredData, line level (7.1)

Gated universe: the rows whose CredData category is exactly `UUID` are removed, for every system (amendment 2). That choice was made after reading the training and validation rows and before any test file was read. Full universe: every meta row.

| gated universe | precision | recall | F1 | false alarms | F1 without the largest | macro F1 (≥ 20 T rows) |
|---|---:|---:|---:|---:|---:|---:|
| lr | 0.8843 | 0.8359 | 0.8594 | 0.0255 | 0.7937 | 0.8238 (6) |
| generator | 0.2753 | 0.9540 | 0.4272 | 0.5848 | 0.3236 | 0.6393 (6) |
| secretgate-default | 0.9270 | 0.3611 | 0.5197 | 0.0066 | 0.6136 | 0.5118 (6) |
| gitleaks | 0.9850 | 0.2867 | 0.4441 | 0.0010 | 0.5114 | 0.3793 (6) |
| betterleaks | 0.7500 | 0.3151 | 0.4438 | 0.0245 | 0.5022 | 0.4386 (6) |
| detect-secrets | 0.6069 | 0.3851 | 0.4712 | 0.0581 | 0.6048 | 0.5435 (6) |
| credsweeper-ml | 0.9202 | 0.9847 | 0.9514 | 0.0199 | 0.9253 | 0.9476 (6) |
| credsweeper-noml | 0.3218 | 1.0000 | 0.4869 | 0.4906 | 0.3873 | 0.7434 (6) |
| cbx-17m | 0.9814 | 0.8096 | 0.8873 | 0.0036 | 0.8191 | 0.8473 (6) |
| potion-lr | 0.5308 | 0.8096 | 0.6412 | 0.1666 | 0.5255 | 0.6850 (6) |

| full universe | precision | recall | F1 | false alarms | F1 without the largest | macro F1 (≥ 20 T rows) |
|---|---:|---:|---:|---:|---:|---:|
| lr | 0.8843 | 0.6575 | 0.7542 | 0.0256 | 0.7154 | 0.7410 (6) |
| generator | 0.2842 | 0.7762 | 0.4161 | 0.5820 | 0.3287 | 0.6187 (6) |
| secretgate-default | 0.9270 | 0.2840 | 0.4348 | 0.0067 | 0.5378 | 0.4493 (6) |
| gitleaks | 0.9850 | 0.2255 | 0.3669 | 0.0010 | 0.4420 | 0.3237 (6) |
| betterleaks | 0.7500 | 0.2478 | 0.3726 | 0.0246 | 0.4419 | 0.4006 (6) |
| detect-secrets | 0.6069 | 0.3029 | 0.4041 | 0.0584 | 0.5466 | 0.4989 (6) |
| credsweeper-ml | 0.9550 | 0.9862 | 0.9704 | 0.0138 | 0.9519 | 0.9680 (6) |
| credsweeper-noml | 0.3788 | 0.9983 | 0.5492 | 0.4872 | 0.4361 | 0.7810 (6) |
| cbx-17m | 0.9814 | 0.6368 | 0.7724 | 0.0036 | 0.7292 | 0.7717 (6) |
| potion-lr | 0.5380 | 0.6454 | 0.5869 | 0.1650 | 0.5000 | 0.6415 (6) |

| | version | unit | unlabelled-line alarms | per 1,000 files (7.3) | CPU s |
|---|---|---|---:|---:|---:|
| lr | scikit-learn 1.9.1, SGDClassifier alpha 0.0001, 2^18 int8 weights | candidate | 176 | 389.4 | 1.6 |
| generator | sg-cand-1 revision 1, secretgate 4e93cfa2ee6e216caca9bc8e6f74cb565a991350 | candidate | 2279 | 5042.0 | 17.2 |
| secretgate-default | secretgate 4e93cfa2ee6e216caca9bc8e6f74cb565a991350 | span | 88 | 194.7 | 8.7 |
| gitleaks | Gitleaks 8.30.1 | span | 1 | 2.2 | 20.6 |
| betterleaks | Betterleaks 1.10.1 | span | 12 | 26.5 | 2.2 |
| detect-secrets | detect-secrets 1.5.0 | line | 50 | 110.6 | 115.0 |
| credsweeper-ml | CredSweeper 1.19.6 | span | 0 | 0.0 | 43.9 |
| credsweeper-noml | CredSweeper 1.19.6 | span | 0 | 0.0 | 24.5 |
| cbx-17m | cbx-secrets-detector-17M at 89362a0938f9459e5706e703e2a91c5cf2ffe40e | span | 2074 | 4588.5 | 13816.6 |
| potion-lr | model2vec 0.10.0 potion-base-2M, scikit-learn 1.9.1 LogisticRegression | candidate | 395 | 873.9 | 0.7 |

CredSweeper's ML was trained on a random 85% of all CredData lines: its CredData numbers are in-sample and enter no gate (section 6).

## CredData, candidate level (7.2)

Gated universe: the rows whose CredData category is exactly `UUID` are removed, for every system (amendment 2). That choice was made after reading the training and validation rows and before any test file was read. Full universe: every meta row.

| gated universe | candidates | positives | threshold | precision | recall | F1 | PR-AUC | recall at 1% false alarms | ECE | ECE on P(real) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| lr | 1779 | 448 | 0.1530 | 0.8790 | 0.8594 | 0.8691 | 0.9142 | 0.6429 | 0.0677 | 0.1190 |
| generator | 1779 | 448 | — | 0.2518 | 1.0000 | 0.4023 | — | — | — | — |
| potion-lr | 1779 | 448 | 0.0868 | 0.5108 | 0.8438 | 0.6364 | 0.6330 | 0.1116 | 0.1065 | 0.1217 |

| full universe | candidates | positives | threshold | precision | recall | F1 | PR-AUC | recall at 1% false alarms | ECE | ECE on P(real) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| lr | 1779 | 459 | 0.1530 | 0.8790 | 0.8388 | 0.8584 | 0.9033 | 0.6275 | 0.0661 | 0.1177 |
| generator | 1779 | 459 | — | 0.2580 | 1.0000 | 0.4102 | — | — | — | — |
| potion-lr | 1779 | 459 | 0.0868 | 0.5176 | 0.8344 | 0.6389 | 0.6316 | 0.1089 | 0.1127 | 0.1279 |

Candidates on a labelled line that overlap no labelled value are dropped (3.1); ECE is over 15 equal-width bins of the answer's confidence, as sky-eval computes it.

## CredData, span level (section 9, overlap with a T value)

Gated universe: the rows whose CredData category is exactly `UUID` are removed, for every system (amendment 2). That choice was made after reading the training and validation rows and before any test file was read. Full universe: every meta row.

| | gated recall | gated precision | gated F1 | full recall | full precision | full F1 | flags: T / X / F / no row / unlabelled |
|---|---:|---:|---:|---:|---:|---:|---|
| lr | 0.7452 | 0.8730 | 0.8041 | 0.5903 | 0.8730 | 0.7043 | 459 / 117 / 1203 / 176 / 2279 |
| generator | 0.8506 | 0.2295 | 0.3615 | 0.6904 | 0.2348 | 0.3504 | 459 / 117 / 1203 / 176 / 2279 |
| secretgate-default | 0.3238 | 0.9270 | 0.4799 | 0.2564 | 0.9270 | 0.4018 | 165 / 2 / 10 / 1 / 88 |
| gitleaks | 0.2586 | 0.9851 | 0.4097 | 0.2049 | 0.9851 | 0.3392 | 132 / 0 / 2 / 0 / 1 |
| betterleaks | 0.2816 | 0.7164 | 0.4043 | 0.2231 | 0.7164 | 0.3402 | 144 / 13 / 40 / 4 / 12 |
| credsweeper-ml | 0.8851 | 0.9257 | 0.9049 | 0.9059 | 0.9524 | 0.9286 | 820 / 0 / 41 / 0 / 0 |
| credsweeper-noml | 0.8966 | 0.3742 | 0.5280 | 0.9150 | 0.4252 | 0.5806 | 839 / 27 / 1107 / 0 / 0 |
| cbx-17m | 0.8257 | 0.9651 | 0.8900 | 0.6540 | 0.9651 | 0.7797 | 443 / 0 / 12 / 4 / 2074 |
| potion-lr | 0.7241 | 0.4803 | 0.5775 | 0.5812 | 0.4867 | 0.5297 | 459 / 117 / 1203 / 176 / 2279 |

The generator's row is its recall (2.2): the share of T rows a candidate overlaps.

## By CredData category and by path scope (section 9), gated universe, line F1 (recall)

| | Password | Token | Key | Secret | API | Auth | other | scope test | scope src | scope other |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| lr | 0.743 (0.741) | 0.945 (0.905) | 0.800 (0.750) | 0.892 (0.906) | 0.778 (1.000) | 0.907 (0.936) | 0.910 (0.869) | 0.812 (0.788) | 0.966 (1.000) | 0.895 (0.868) |
| generator | 0.575 (0.970) | 0.550 (1.000) | 0.140 (0.938) | 0.383 (1.000) | 0.136 (1.000) | 0.455 (0.979) | 0.814 (0.903) | 0.388 (0.909) | 0.452 (1.000) | 0.464 (0.991) |
| secretgate-default | 0.444 (0.296) | 0.552 (0.381) | 0.737 (0.583) | 0.755 (0.625) | 0.750 (0.857) | 0.255 (0.149) | 0.574 (0.414) | 0.667 (0.519) | 0.600 (0.429) | 0.351 (0.217) |
| gitleaks | 0.349 (0.215) | 0.500 (0.333) | 0.609 (0.438) | 0.609 (0.438) | 0.000 (0.000) | 0.082 (0.043) | 0.520 (0.352) | 0.586 (0.418) | 0.526 (0.357) | 0.285 (0.166) |
| betterleaks | 0.595 (0.556) | 0.507 (0.343) | 0.286 (0.167) | 0.476 (0.312) | 0.222 (0.143) | 0.157 (0.085) | 0.256 (0.159) | 0.429 (0.322) | 1.000 (1.000) | 0.408 (0.268) |
| detect-secrets | 0.747 (0.756) | 0.090 (0.048) | 0.364 (0.292) | 0.408 (0.656) | 0.636 (1.000) | 0.241 (0.149) | 0.427 (0.303) | 0.649 (0.601) | 0.444 (0.857) | 0.253 (0.166) |
| credsweeper-ml | 0.978 (0.978) | 0.986 (0.971) | 0.750 (1.000) | 0.968 (0.938) | 1.000 (1.000) | 0.979 (0.979) | 0.924 (1.000) | 0.904 (0.976) | 1.000 (1.000) | 0.994 (0.991) |
| credsweeper-noml | 0.736 (1.000) | 0.717 (1.000) | 0.142 (1.000) | 0.604 (1.000) | 0.099 (1.000) | 0.470 (1.000) | 0.863 (1.000) | 0.547 (1.000) | 0.075 (1.000) | 0.634 (1.000) |
| cbx-17m | 0.946 (0.904) | 0.951 (0.924) | 0.824 (0.729) | 0.918 (0.875) | 0.833 (0.714) | 0.860 (0.787) | 0.759 (0.621) | 0.760 (0.625) | 0.966 (1.000) | 0.976 (0.962) |
| potion-lr | 0.576 (0.741) | 0.836 (0.876) | 0.418 (0.875) | 0.756 (0.969) | 0.179 (1.000) | 0.772 (0.936) | 0.775 (0.759) | 0.536 (0.692) | 0.750 (0.857) | 0.732 (0.911) |
| positive / negative lines | 135 / 282 | 105 / 233 | 48 / 1028 | 32 / 163 | 7 / 149 | 47 / 181 | 145 / 80 | 208 / 811 | 14 / 395 | 235 / 757 |

## Uncertainty (7.6): paired bootstrap over repositories, 1000 resamples, 95%

| line F1 of lr minus | gated | gated, without the largest | full | full, without the largest |
|---|---|---|---|---|
| generator | +0.4322 [0.2856, 0.5517] | +0.4701 [0.3349, 0.5769] | +0.3381 [0.2156, 0.4729] | +0.3867 [0.2696, 0.5318] |
| secretgate-default | +0.3397 [0.0748, 0.5455] | +0.1800 [0.0375, 0.3173] | +0.3194 [0.0818, 0.4964] | +0.1776 [0.0513, 0.3047] |
| gitleaks | +0.4153 [0.1134, 0.6114] | +0.2823 [0.0675, 0.5302] | +0.3872 [0.1206, 0.5496] | +0.2734 [0.0782, 0.4948] |
| betterleaks | +0.4156 [0.1590, 0.5863] | +0.2914 [0.0891, 0.4721] | +0.3816 [0.1587, 0.5143] | +0.2736 [0.0891, 0.4098] |
| detect-secrets | +0.3882 [0.0876, 0.6938] | +0.1888 [0.0626, 0.3439] | +0.3501 [0.0839, 0.6075] | +0.1688 [0.0616, 0.3173] |
| credsweeper-ml | -0.0920 [-0.2474, -0.0102] | -0.1317 [-0.3128, -0.0057] | -0.2162 [-0.3162, -0.1678] | -0.2364 [-0.3592, -0.1584] |
| credsweeper-noml | +0.3724 [0.1332, 0.5728] | +0.4064 [0.1429, 0.6044] | +0.2050 [-0.0315, 0.4632] | +0.2793 [0.0200, 0.5552] |
| cbx-17m | -0.0279 [-0.1579, 0.1193] | -0.0254 [-0.2205, 0.1649] | -0.0182 [-0.1512, 0.1222] | -0.0138 [-0.2163, 0.1658] |
| potion-lr | +0.2181 [0.1233, 0.3341] | +0.2681 [0.1700, 0.3692] | +0.1673 [0.0936, 0.2722] | +0.2154 [0.1508, 0.3030] |

| ECE | gated | full |
|---|---|---|
| lr | 0.0677 [0.0291, 0.1356] | 0.0661 [0.0293, 0.1454] |
| potion-lr | 0.1065 [0.0331, 0.2359] | 0.1127 [0.0348, 0.2472] |

## Issue reports, validation split (7.4)

5432 candidates (586 secrets) in 5371 texts; 524 occur several times in their text (the first occurrence is the candidate), 0 are not found in it.

| | precision | recall | F1 | false alarms | ECE | end to end: flags | precision over all flags | F1 | precision over flags on a candidate | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| lr | 0.4852 | 0.8652 | 0.6217 | 0.1110 | 0.0346 | — | — | — | — | — |
| generator | 0.2222 | 0.7884 | 0.3467 | 0.3337 | — | 16426 | 0.0289 | 0.0558 | 0.2134 | 0.3359 |
| secretgate-default | 0.7379 | 0.6007 | 0.6623 | 0.0258 | — | 1467 | 0.2399 | 0.3429 | 0.7379 | 0.6623 |
| gitleaks | 0.8635 | 0.5614 | 0.6805 | 0.0107 | — | 1055 | 0.3166 | 0.4049 | 0.8630 | 0.6803 |
| betterleaks | 0.5684 | 0.5887 | 0.5784 | 0.0541 | — | 2024 | 0.1764 | 0.2714 | 0.5758 | 0.5822 |
| detect-secrets | 0.3594 | 0.3140 | 0.3352 | 0.0677 | — | 1893 | 0.1342 | 0.1880 | 0.4276 | 0.3621 |
| credsweeper-ml | 0.3713 | 0.6894 | 0.4827 | 0.1411 | — | 5163 | 0.1247 | 0.2112 | 0.4760 | 0.5632 |
| credsweeper-noml | 0.3115 | 0.8020 | 0.4487 | 0.2144 | — | 9692 | 0.0782 | 0.1425 | 0.3929 | 0.5275 |
| potion-lr | 0.2573 | 0.8976 | 0.4000 | 0.3132 | 0.0271 | — | — | — | — | — |

| obfuscation sensitivity (candidate F1) | values as published | through CredData's `get_obfuscated_value` | difference |
|---|---:|---:|---:|
| generator | 0.3467 | 0.2836 | +0.0631 |
| secretgate-default | 0.6623 | 0.5990 | +0.0633 |
| gitleaks | 0.6805 | 0.6205 | +0.0599 |
| betterleaks | 0.5784 | 0.5084 | +0.0700 |
| detect-secrets | 0.3352 | 0.2852 | +0.0500 |
| credsweeper-ml | 0.4827 | 0.4065 | +0.0762 |
| credsweeper-noml | 0.4487 | 0.3627 | +0.0860 |

585 positive candidate strings were passed through the function; 26 stay as published (the function failed on them).

## Clean real text (7.7)

8825 files (list sha256 `b25d912775de152941d98462fcae3e96fc15dbb6292614adbfeb22083e74cc53`).

| as the census: no source path, no pragma for the default engine (the gate's reading) | files with an accepted candidate | share | accepted |
|---|---:|---:|---:|
| generator | 2161 | 24.49% | 14756 |
| secretgate-default | 11 | 0.12% | 57 |

| with the file's path | files with an accepted candidate | share | accepted |
|---|---:|---:|---:|
| generator | 187 | 2.12% | 1186 |
| secretgate-default | 7 | 0.08% | 47 |

## Value-only probe (section 9)

A logistic regression on the value's own features (len, entropy, chars_per_token, digits, upper, lower, other, word_run), trained on 19946 labelled CredData training candidates, scored on 1589 CredData validation candidates (384 positives): PR-AUC 0.8202 (the share of positives is 0.2417), ROC-AUC 0.9197, best F1 0.7200. A high score means the shape of a value alone separates T from F and X.

## Gate

- G1, every repository: holds (lr 0.8594, best rule scanner secretgate-default 0.5197 + 0.15 = 0.6697; full universe 0.7542 against 0.4348)
- G1, without the largest repository: holds (lr 0.7937, best rule scanner secretgate-default 0.6136 + 0.15 = 0.7636; full universe 0.7154 against 0.5378)
- G2, every repository: holds (lr 0.8594, cbx-17m 0.8873 - 0.1 = 0.7873; full universe 0.7542 against 0.7724)
- G2, without the largest repository: holds (lr 0.7937, cbx-17m 0.8191 - 0.1 = 0.7191; full universe 0.7154 against 0.7292)
- G3, every repository: fails (lr 0.8594, lr 0.8594; full universe 0.7542 against 0.7542)
- G3, without the largest repository: fails (lr 0.7937, lr 0.7937; full universe 0.7154 against 0.7154)
- G4: fails (issue-report candidate F1 0.6217, Gitleaks 0.6805 + 0.08 = 0.7605)
- G5: fails (ECE 0.0677 on the gated universe, at most 0.05; full universe 0.0661)
- G6: fails (no native latency)
- G7: fails (no clean-text result of lr)
- announced goal, G2 at -0.05: met
- gate: REHEARSAL on validation: the gate would not hold on these numbers; nothing is decided before the test
