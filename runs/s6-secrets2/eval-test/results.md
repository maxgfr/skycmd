# s6-secrets v2 (s6-secrets-verify-pico): results on the test split (docs/s6-secrets2-prereg.md)

221 repositories, 7372 files; T rows: 10463 (gated universe), 11603 (full). Largest repository: `2ba83c6a`, 5165 gated T rows.

## CredData, line level (7.1)

Gated universe: the rows whose CredData category is exactly `UUID` are removed, for every system (amendment 2). That choice was made after reading the training and validation rows and before any test file was read. Full universe: every meta row.

| gated universe | precision | recall | F1 | false alarms | F1 without the largest | macro F1 (≥ 20 T rows) |
|---|---:|---:|---:|---:|---:|---:|
| model | 0.8387 | 0.8741 | 0.8561 | 0.0599 | 0.8114 | 0.8091 (51) |
| generator | 0.2942 | 0.9418 | 0.4483 | 0.8053 | 0.3231 | 0.4967 (51) |
| secretgate-default | 0.8176 | 0.6560 | 0.7279 | 0.0521 | 0.5072 | 0.4878 (51) |
| gitleaks | 0.8546 | 0.5980 | 0.7037 | 0.0363 | 0.4080 | 0.3436 (51) |
| betterleaks | 0.7694 | 0.7002 | 0.7332 | 0.0748 | 0.5367 | 0.5606 (51) |
| detect-secrets | 0.5504 | 0.2329 | 0.3273 | 0.0678 | 0.5014 | 0.5299 (51) |
| credsweeper-ml | 0.9946 | 0.9925 | 0.9935 | 0.0019 | 0.9872 | 0.9888 (51) |
| credsweeper-noml | 0.4512 | 0.9970 | 0.6213 | 0.4320 | 0.5428 | 0.7153 (51) |
| cbx-17m | 0.8265 | 0.5820 | 0.6830 | 0.0435 | 0.8493 | 0.8133 (51) |
| lr | 0.7345 | 0.8652 | 0.7945 | 0.1114 | 0.7179 | 0.7287 (51) |
| potion-lr | 0.4966 | 0.8955 | 0.6389 | 0.3234 | 0.5536 | 0.6552 (51) |

| full universe | precision | recall | F1 | false alarms | F1 without the largest | macro F1 (≥ 20 T rows) |
|---|---:|---:|---:|---:|---:|---:|
| model | 0.8398 | 0.7922 | 0.8153 | 0.0591 | 0.7348 | 0.7113 (60) |
| generator | 0.2954 | 0.8606 | 0.4399 | 0.8029 | 0.3158 | 0.4477 (60) |
| secretgate-default | 0.8176 | 0.5935 | 0.6877 | 0.0518 | 0.4445 | 0.4166 (60) |
| gitleaks | 0.8546 | 0.5410 | 0.6626 | 0.0360 | 0.3507 | 0.2950 (60) |
| betterleaks | 0.7694 | 0.6334 | 0.6948 | 0.0743 | 0.4786 | 0.4915 (60) |
| detect-secrets | 0.5530 | 0.2129 | 0.3075 | 0.0673 | 0.4534 | 0.4715 (60) |
| credsweeper-ml | 0.9963 | 0.9914 | 0.9938 | 0.0014 | 0.9889 | 0.9898 (60) |
| credsweeper-noml | 0.4761 | 0.9954 | 0.6441 | 0.4285 | 0.5891 | 0.7360 (60) |
| cbx-17m | 0.8266 | 0.5265 | 0.6433 | 0.0432 | 0.7614 | 0.7165 (60) |
| lr | 0.7345 | 0.7829 | 0.7579 | 0.1107 | 0.6510 | 0.6446 (60) |
| potion-lr | 0.4956 | 0.8135 | 0.6160 | 0.3238 | 0.5172 | 0.5844 (60) |

| | version | unit | unlabelled-line alarms | per 1,000 files (7.3) | CPU s |
|---|---|---|---:|---:|---:|
| model | full-depth.gguf sha256 01e6db3fa71cc2c5563dea222c68e040d3994ee2abbb02b25491a51b8cb54f27 | candidate | 735 | 99.7 | — |
| generator | sg-cand-1 revision 1, secretgate 4e93cfa2ee6e216caca9bc8e6f74cb565a991350 | candidate | 32951 | 4469.8 | 77.1 |
| secretgate-default | secretgate 4e93cfa2ee6e216caca9bc8e6f74cb565a991350 | span | 225 | 30.5 | 33.4 |
| gitleaks | Gitleaks 8.30.1 | span | 199 | 27.0 | 121.8 |
| betterleaks | Betterleaks 1.10.1 | span | 288 | 39.1 | 8.8 |
| detect-secrets | detect-secrets 1.5.0 | line | 2407 | 326.5 | 704.7 |
| credsweeper-ml | CredSweeper 1.19.6 | span | 0 | 0.0 | 300.7 |
| credsweeper-noml | CredSweeper 1.19.6 | span | 0 | 0.0 | 79.2 |
| cbx-17m | cbx-secrets-detector-17M at 89362a0938f9459e5706e703e2a91c5cf2ffe40e | span | 3587 | 486.6 | 57098.4 |
| lr | scikit-learn 1.9.1, SGDClassifier alpha 0.0001, 2^18 int8 weights | candidate | 1065 | 144.5 | 12.5 |
| potion-lr | model2vec 0.10.0 potion-base-2M, scikit-learn 1.9.1 LogisticRegression | candidate | 6576 | 892.0 | 6.6 |

CredSweeper's ML was trained on a random 85% of all CredData lines: its CredData numbers are in-sample and enter no gate (section 6).

## CredData, candidate level (7.2)

Gated universe: the rows whose CredData category is exactly `UUID` are removed, for every system (amendment 2). That choice was made after reading the training and validation rows and before any test file was read. Full universe: every meta row.

| gated universe | candidates | positives | threshold | precision | recall | F1 | PR-AUC | recall at 1% false alarms | ECE | ECE on P(real) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| model | 33809 | 9623 | 0.2598 | 0.8363 | 0.9344 | 0.8827 | 0.9335 | 0.5556 | 0.0073 | 0.0101 |
| generator | 33809 | 9623 | — | 0.2846 | 1.0000 | 0.4431 | — | — | — | — |
| lr | 33809 | 9623 | 0.1550 | 0.7331 | 0.9282 | 0.8192 | 0.8648 | 0.2657 | 0.0786 | 0.1112 |
| potion-lr | 33809 | 9623 | 0.0786 | 0.4882 | 0.9504 | 0.6451 | 0.7103 | 0.0924 | 0.0972 | 0.1144 |

| full universe | candidates | positives | threshold | precision | recall | F1 | PR-AUC | recall at 1% false alarms | ECE | ECE on P(real) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| model | 33924 | 9641 | 0.2598 | 0.8374 | 0.9343 | 0.8832 | 0.9335 | 0.5596 | 0.0072 | 0.0105 |
| generator | 33924 | 9641 | — | 0.2842 | 1.0000 | 0.4426 | — | — | — | — |
| lr | 33924 | 9641 | 0.1550 | 0.7331 | 0.9266 | 0.8185 | 0.8644 | 0.2652 | 0.0786 | 0.1111 |
| potion-lr | 33924 | 9641 | 0.0786 | 0.4865 | 0.9501 | 0.6435 | 0.7091 | 0.0920 | 0.0970 | 0.1140 |

Candidates on a labelled line that overlap no labelled value are dropped (3.1); ECE is over 15 equal-width bins of the answer's confidence, as sky-eval computes it.

## CredData, span level (section 9, overlap with a T value)

Gated universe: the rows whose CredData category is exactly `UUID` are removed, for every system (amendment 2). That choice was made after reading the training and validation rows and before any test file was read. Full universe: every meta row.

| | gated recall | gated precision | gated F1 | full recall | full precision | full F1 | flags: T / X / F / no row / unlabelled |
|---|---:|---:|---:|---:|---:|---:|---|
| model | 0.8656 | 0.8309 | 0.8479 | 0.7820 | 0.8319 | 0.8062 | 9641 / 1601 / 22682 / 1235 / 32951 |
| generator | 0.9260 | 0.2752 | 0.4243 | 0.8367 | 0.2742 | 0.4131 | 9641 / 1601 / 22682 / 1235 / 32951 |
| secretgate-default | 0.6507 | 0.8168 | 0.7243 | 0.5868 | 0.8168 | 0.6830 | 6756 / 221 / 1290 / 4 / 225 |
| gitleaks | 0.5938 | 0.8544 | 0.7006 | 0.5356 | 0.8544 | 0.6584 | 6167 / 16 / 1034 / 1 / 199 |
| betterleaks | 0.6937 | 0.7686 | 0.7292 | 0.6256 | 0.7686 | 0.6898 | 7220 / 831 / 1252 / 91 / 288 |
| credsweeper-ml | 0.9923 | 0.9960 | 0.9941 | 0.9912 | 0.9973 | 0.9943 | 22371 / 1 / 59 / 0 / 0 |
| credsweeper-noml | 0.9964 | 0.6170 | 0.7621 | 0.9949 | 0.6295 | 0.7711 | 22525 / 638 / 12618 / 0 / 0 |
| cbx-17m | 0.5843 | 0.8103 | 0.6790 | 0.5270 | 0.8104 | 0.6387 | 7061 / 21 / 1558 / 73 / 3587 |
| lr | 0.8593 | 0.7321 | 0.7906 | 0.7750 | 0.7319 | 0.7528 | 9641 / 1601 / 22682 / 1235 / 32951 |
| potion-lr | 0.8795 | 0.4744 | 0.6163 | 0.7944 | 0.4722 | 0.5923 | 9641 / 1601 / 22682 / 1235 / 32951 |

The generator's row is its recall (2.2): the share of T rows a candidate overlaps.

## By CredData category and by path scope (section 9), gated universe, line F1 (recall)

| | Password | Token | Key | Secret | API | Auth | other | scope test | scope src | scope other |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| model | 0.838 (0.893) | 0.824 (0.766) | 0.864 (0.957) | 0.905 (0.860) | 0.541 (0.469) | 0.881 (0.852) | 0.873 (0.799) | 0.884 (0.915) | 0.767 (0.852) | 0.756 (0.722) |
| generator | 0.459 (0.985) | 0.298 (0.811) | 0.400 (0.991) | 0.694 (0.995) | 0.114 (0.512) | 0.568 (0.981) | 0.769 (0.858) | 0.577 (0.978) | 0.189 (0.918) | 0.250 (0.807) |
| secretgate-default | 0.406 (0.309) | 0.685 (0.534) | 0.822 (0.869) | 0.912 (0.880) | 0.439 (0.301) | 0.258 (0.150) | 0.653 (0.497) | 0.779 (0.741) | 0.675 (0.587) | 0.480 (0.344) |
| gitleaks | 0.099 (0.053) | 0.668 (0.511) | 0.824 (0.870) | 0.902 (0.825) | 0.419 (0.273) | 0.229 (0.130) | 0.628 (0.464) | 0.764 (0.694) | 0.667 (0.536) | 0.383 (0.242) |
| betterleaks | 0.545 (0.642) | 0.672 (0.517) | 0.838 (0.867) | 0.897 (0.815) | 0.459 (0.306) | 0.251 (0.146) | 0.638 (0.496) | 0.770 (0.754) | 0.667 (0.681) | 0.583 (0.499) |
| detect-secrets | 0.612 (0.692) | 0.427 (0.279) | 0.128 (0.075) | 0.321 (0.250) | 0.282 (0.287) | 0.203 (0.119) | 0.333 (0.213) | 0.260 (0.165) | 0.465 (0.416) | 0.476 (0.463) |
| credsweeper-ml | 0.991 (0.983) | 0.994 (0.990) | 0.994 (0.997) | 0.996 (0.995) | 0.978 (0.971) | 0.996 (0.993) | 0.991 (0.993) | 0.997 (0.996) | 0.959 (0.994) | 0.986 (0.980) |
| credsweeper-noml | 0.648 (0.994) | 0.555 (0.997) | 0.532 (1.000) | 0.898 (1.000) | 0.267 (0.995) | 0.670 (0.998) | 0.916 (0.995) | 0.692 (0.998) | 0.378 (1.000) | 0.481 (0.991) |
| cbx-17m | 0.904 (0.838) | 0.929 (0.874) | 0.266 (0.197) | 0.945 (0.897) | 0.898 (0.818) | 0.936 (0.885) | 0.876 (0.801) | 0.634 (0.530) | 0.880 (0.836) | 0.826 (0.741) |
| lr | 0.689 (0.800) | 0.746 (0.653) | 0.795 (0.959) | 0.909 (0.922) | 0.358 (0.378) | 0.837 (0.804) | 0.854 (0.768) | 0.842 (0.926) | 0.567 (0.710) | 0.643 (0.658) |
| potion-lr | 0.547 (0.883) | 0.633 (0.716) | 0.607 (0.963) | 0.836 (0.954) | 0.232 (0.435) | 0.795 (0.923) | 0.804 (0.822) | 0.700 (0.943) | 0.449 (0.861) | 0.471 (0.720) |
| positive / negative lines | 1714 / 6394 | 777 / 3288 | 4029 / 13092 | 1389 / 1603 | 209 / 1987 | 1016 / 2078 | 2966 / 1587 | 7903 / 13446 | 317 / 3045 | 2077 / 12408 |

## Uncertainty (7.6): paired bootstrap over repositories, 1000 resamples, 95%

| line F1 of model minus | gated | gated, without the largest | full | full, without the largest |
|---|---|---|---|---|
| generator | +0.4078 [0.3223, 0.5243] | +0.4883 [0.4409, 0.5373] | +0.3754 [0.3107, 0.4632] | +0.4190 [0.3599, 0.4754] |
| secretgate-default | +0.1281 [0.0464, 0.3939] | +0.3042 [0.1938, 0.4216] | +0.1276 [0.0473, 0.3794] | +0.2903 [0.1881, 0.4063] |
| gitleaks | +0.1524 [0.0565, 0.4907] | +0.4034 [0.2899, 0.5157] | +0.1527 [0.0575, 0.4723] | +0.3840 [0.2741, 0.4989] |
| betterleaks | +0.1229 [0.0413, 0.3613] | +0.2748 [0.1743, 0.3796] | +0.1205 [0.0414, 0.3434] | +0.2562 [0.1578, 0.3614] |
| detect-secrets | +0.5288 [0.2356, 0.7007] | +0.3100 [0.2239, 0.4040] | +0.5078 [0.2034, 0.6919] | +0.2814 [0.1929, 0.3781] |
| credsweeper-ml | -0.1375 [-0.2190, -0.1157] | -0.1758 [-0.2276, -0.1390] | -0.1785 [-0.3177, -0.1311] | -0.2541 [-0.3335, -0.1885] |
| credsweeper-noml | +0.2348 [0.1937, 0.3159] | +0.2687 [0.2025, 0.3256] | +0.1712 [0.0693, 0.2164] | +0.1457 [0.0525, 0.2291] |
| cbx-17m | +0.1730 [-0.0654, 0.3074] | -0.0379 [-0.0765, -0.0058] | +0.1720 [-0.0533, 0.3047] | -0.0266 [-0.0638, 0.0042] |
| lr | +0.0615 [0.0435, 0.1197] | +0.0935 [0.0646, 0.1302] | +0.0574 [0.0414, 0.1102] | +0.0838 [0.0560, 0.1183] |
| potion-lr | +0.2172 [0.1837, 0.3005] | +0.2579 [0.2066, 0.3171] | +0.1993 [0.1738, 0.2533] | +0.2176 [0.1767, 0.2687] |

| ECE | gated | full |
|---|---|---|
| model | 0.0073 [0.0040, 0.0174] | 0.0072 [0.0041, 0.0174] |
| lr | 0.0786 [0.0235, 0.1309] | 0.0786 [0.0237, 0.1305] |
| potion-lr | 0.0972 [0.0168, 0.1661] | 0.0970 [0.0167, 0.1655] |

## Issue reports, test split (7.4)

8123 candidates (888 secrets) in 7986 texts; 774 occur several times in their text (the first occurrence is the candidate), 0 are not found in it.

| | precision | recall | F1 | false alarms | ECE | end to end: flags | precision over all flags | F1 | precision over flags on a candidate | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| model | 0.7484 | 0.8975 | 0.8162 | 0.0370 | 0.0211 | — | — | — | — | — |
| generator | 0.2198 | 0.7883 | 0.3437 | 0.3435 | — | 25931 | 0.0271 | 0.0523 | 0.2078 | 0.3289 |
| secretgate-default | 0.7699 | 0.5991 | 0.6738 | 0.0220 | — | 2295 | 0.2301 | 0.3325 | 0.7719 | 0.6746 |
| gitleaks | 0.9028 | 0.5856 | 0.7104 | 0.0077 | — | 1817 | 0.2966 | 0.3938 | 0.9059 | 0.7113 |
| betterleaks | 0.6102 | 0.6171 | 0.6137 | 0.0484 | — | 3318 | 0.1733 | 0.2706 | 0.6216 | 0.6194 |
| detect-secrets | 0.3461 | 0.2950 | 0.3185 | 0.0684 | — | 3212 | 0.1083 | 0.1585 | 0.4104 | 0.3433 |
| credsweeper-ml | 0.3740 | 0.7185 | 0.4919 | 0.1476 | — | 8345 | 0.1190 | 0.2042 | 0.4788 | 0.5746 |
| credsweeper-noml | 0.3087 | 0.8074 | 0.4466 | 0.2220 | — | 16104 | 0.0720 | 0.1322 | 0.3994 | 0.5344 |
| cbx-17m | 0.8830 | 0.8412 | 0.8616 | 0.0137 | — | 5197 | 0.1537 | 0.2600 | 0.8829 | 0.8615 |
| lr | 0.5042 | 0.8874 | 0.6430 | 0.1071 | 0.0379 | — | — | — | — | — |
| potion-lr | 0.2417 | 0.9223 | 0.3831 | 0.3551 | 0.0198 | — | — | — | — | — |
| lr-e2e | 0.7030 | 0.7331 | 0.7178 | 0.0380 | — | 3738 | 0.1747 | 0.2822 | 0.7044 | 0.7185 |
| model-e2e | 0.7607 | 0.7196 | 0.7396 | 0.0278 | — | 3068 | 0.2073 | 0.3219 | 0.7626 | 0.7405 |
| potion-lr-e2e | 0.3683 | 0.7230 | 0.4880 | 0.1522 | — | 10245 | 0.0628 | 0.1155 | 0.3612 | 0.4818 |

| obfuscation sensitivity (candidate F1) | values as published | through CredData's `get_obfuscated_value` | difference |
|---|---:|---:|---:|
| model | 0.8162 | 0.8498 | -0.0336 |
| generator | 0.3437 | 0.2883 | +0.0555 |
| secretgate-default | 0.6738 | 0.6142 | +0.0597 |
| gitleaks | 0.7104 | 0.6485 | +0.0618 |
| betterleaks | 0.6137 | 0.5401 | +0.0736 |
| detect-secrets | 0.3185 | 0.2766 | +0.0419 |
| credsweeper-ml | 0.4919 | 0.4188 | +0.0731 |
| credsweeper-noml | 0.4466 | 0.3710 | +0.0756 |
| cbx-17m | 0.8616 | 0.8022 | +0.0594 |
| lr | 0.6430 | 0.6791 | -0.0361 |
| potion-lr | 0.3831 | 0.3906 | -0.0075 |

886 positive candidate strings were passed through the function; 41 stay as published (the function failed on them).

## Clean real text (7.7)

8825 files (list sha256 `b25d912775de152941d98462fcae3e96fc15dbb6292614adbfeb22083e74cc53`).

| as the census: no source path, no pragma for the default engine (the gate's reading) | files with an accepted candidate | share | accepted |
|---|---:|---:|---:|
| model | 31 | 0.35% | 110 |
| generator | 2161 | 24.49% | 14756 |
| secretgate-default | 11 | 0.12% | 57 |
| lr | 77 | 0.87% | 323 |
| potion-lr | 401 | 4.54% | 969 |

| with the file's path | files with an accepted candidate | share | accepted |
|---|---:|---:|---:|
| generator | 187 | 2.12% | 1186 |
| secretgate-default | 7 | 0.08% | 47 |

## Latency and size (7.5)

full-depth.gguf: 249,984 bytes, 445,345 parameters, tau* 0.2598. Native, 68110 candidates: median 1606 µs at full depth, 372 µs at the first exit; early-exit rate at the stored thresholds 0.000.

| exit | accuracy (real question) | ECE |
|---:|---:|---:|
| 1 | 0.9461 | 0.0120 |
| 2 | 0.9580 | 0.0044 |
| 3 | 0.9600 | 0.0069 |
| 4 | 0.9614 | 0.0066 |

WASM (200 candidates, thresholds at 1): median 2666 µs, max |p_wasm - p_native| 4.17e-07, argmax agree 200.

## Value-only probe (section 9)

A logistic regression on the value's own features (len, entropy, chars_per_token, digits, upper, lower, other, word_run), trained on 19946 labelled CredData training candidates, scored on 1589 CredData validation candidates (384 positives): PR-AUC 0.8202 (the share of positives is 0.2417), ROC-AUC 0.9197, best F1 0.7200. A high score means the shape of a value alone separates T from F and X.

## Gate

- G1, every repository: fails (model 0.8561, best rule scanner betterleaks 0.7332 + 0.15 = 0.8832; full universe 0.8153 against 0.6948)
- G1, without the largest repository: holds (model 0.8114, best rule scanner betterleaks 0.5367 + 0.15 = 0.6867; full universe 0.7348 against 0.4786)
- G2, every repository: holds (model 0.8561, cbx-17m 0.6830 - 0.1 = 0.5830; full universe 0.8153 against 0.6433)
- G2, without the largest repository: holds (model 0.8114, cbx-17m 0.8493 - 0.1 = 0.7493; full universe 0.7348 against 0.7614)
- G3, every repository: holds (model 0.8561, lr 0.7945; full universe 0.8153 against 0.7579)
- G3, without the largest repository: holds (model 0.8114, lr 0.7179; full universe 0.7348 against 0.6510)
- G4: holds (issue-report candidate F1 0.8162, Gitleaks 0.7104 + 0.08 = 0.7904)
- G5: holds (ECE 0.0073 on the gated universe, at most 0.05; full universe 0.0072)
- G6: holds (native median 1606 µs per candidate, at most 3000)
- G7: holds (31 of 8825 clean files with an accepted candidate, 0.35%, at most 1.0%)
- announced goal, G2 at -0.05: met
- gate: FAIL: not published
