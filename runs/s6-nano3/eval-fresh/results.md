# s6-route (s6-nano v3): gate passed on the fresh slates

**Disclosure.** v3 was designed after v2's test was read, from an exploratory reading of v2's test dumps. Its parameters come from validation. Its gate is read on requests no system had scored; on T2-tools these requests are about the same held-out tools as v2's test. The latency bar is 30 times v2's and was set after v2 was measured. The system needs distinctive tool names.

System: p(i) ∝ (p_model(i) · p_potion-cos-4bit(i))^beta, β = 0.81, τ_ask3 = 0.671047 (runs/s6-nano3/fit.json); potion-cos-4bit θ = 0.396120, T = 0.0671021; 3,422,864 bytes.

| condition | value | bar | pass |
|---|---:|---:|:---:|
| G1 system − bm25 on f_tools | 0.0698 [0.0455, 0.0952] | > 0, lower bound > 0 | yes |
| G1 system vs minilm-cos − 0.03 | 0.8868 | ≥ 0.8748 | yes |
| G1 system vs the model alone | 0.8868 | > 0.7926 | yes |
| G1 system vs potion-cos-4bit alone | 0.8868 | > 0.7386 | yes |
| G2 f_metatool: system vs max(model, potion-cos-4bit) − 0.02 | 0.5175 | ≥ 0.5140 | yes |
| G2 f_clinc: system vs max(model, potion-cos-4bit) − 0.02 | 0.8040 | ≥ 0.7655 | yes |
| G2 f_massive: system vs max(model, potion-cos-4bit) − 0.02 | 0.5340 | ≥ 0.4555 | yes |
| G3 ECE on f_tools (15 bins) | 0.0162 | ≤ 0.05 | yes |
| G4 max abs p gap, WASM vs Python (200 slates) | 0.0000016 | ≤ 1e-3 | yes |
| G4 same answer | 200 | ≥ 199 of 200 | yes |
| G4 WASM median, tool list cached (µs) | 30879.1 | ≤ 60000 | yes |
| G4 the two files (bytes) | 3692224 | ≤ 4,000,000 | yes |

Gate: **passed**

Paired bootstrap of the accuracy differences on f_tools (1,000 resamples of the slates, seed 0, 95%):

| system vs | difference | interval |
|---|---:|---|
| bm25 | 0.0698 | [0.0455, 0.0952] |
| minilm-cos | -0.0180 | [-0.0381, 0.0032] |
| model | 0.0942 | [0.0720, 0.1185] |
| potion-cos-4bit | 0.1481 | [0.1228, 0.1714] |
| bm25+potion-cos-4bit | 0.0243 | [0.0085, 0.0413] |
| s6-route-fast | 0.0032 | [-0.0063, 0.0127] |

## Announced goals (not gates)

- accuracy ≥ 0.87 on f_tools: 0.8868 (met)
- ≥ 0.95 among answered at τ_ask3 with coverage ≥ 0.80: 0.9433 at coverage 0.8593 (not met)
- the fast mode within 0.015 of full depth: 0.8836 against 0.8868 (met)

## Readings of section 6 (reported, not gated)

Clusters: 273 (f_tools.clusters); the three largest hold 300 slates.

Without the three largest clusters (645 slates):

| system | accuracy |
|---|---:|
| model | 0.8016 |
| model-exit0 | 0.8171 |
| potion-cos | 0.7039 |
| potion-cos-4bit | 0.7023 |
| static-ft | 0.7070 |
| pair-lr | 0.3302 |
| bm25 | 0.8605 |
| minilm-cos | 0.9070 |
| toolrouter | 0.7612 |
| name-prior | 0.2047 |
| s6-route | 0.8806 |
| s6-route-fast | 0.8791 |
| bm25+potion-cos-4bit | 0.8620 |

| system vs | difference, without the three largest (slate bootstrap) | interval | difference, all slates (cluster bootstrap) | interval |
|---|---:|---|---:|---|
| bm25 | 0.0202 | [-0.0062, 0.0481] | 0.0698 | [-0.0079, 0.1404] |
| minilm-cos | -0.0264 | [-0.0543, 0.0016] | -0.0180 | [-0.0661, 0.0194] |
| model | 0.0791 | [0.0527, 0.1054] | 0.0942 | [0.0543, 0.1325] |
| potion-cos-4bit | 0.1783 | [0.1442, 0.2109] | 0.1481 | [0.1168, 0.2016] |

Names masked (f_tools_masked; the system is expected to fail there):

| system | accuracy |
|---|---:|
| model | 0.0720 |
| model-exit0 | 0.0762 |
| potion-cos | 0.7249 |
| potion-cos-4bit | 0.7280 |
| static-ft | 0.7365 |
| pair-lr | 0.4106 |
| bm25 | 0.8138 |
| minilm-cos | 0.9005 |
| toolrouter | 0.7513 |
| name-prior | 0.1238 |
| s6-route | 0.2497 |
| s6-route-fast | 0.2899 |
| bm25+potion-cos-4bit | 0.8423 |

## Every system, every file

| system | file | n | accuracy | tool gold | none gold | none P | none R | none F1 | ECE | NLL | mean conf | none answers | at τ_ask3: accuracy | coverage |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| model | f_tools | 945 | 0.7926 | 0.8558 | 0.5397 | 0.7338 | 0.5397 | 0.6220 | 0.0234 | 0.5682 | 0.8003 | 0.1471 | — | — |
| model | f_tools_masked | 945 | 0.0720 | 0.0899 | 0.0000 | — | 0.0000 | — | 0.7248 | 9.6331 | 0.7968 | 0.0000 | — | — |
| model | f_metatool | 2000 | 0.3190 | 0.3556 | 0.1725 | 0.2604 | 0.1725 | 0.2075 | 0.2390 | 2.1997 | 0.5574 | 0.1325 | — | — |
| model | f_clinc | 2000 | 0.4240 | 0.5100 | 0.0800 | 0.5517 | 0.0800 | 0.1397 | 0.0584 | 1.6450 | 0.4809 | 0.0290 | — | — |
| model | f_massive | 2000 | 0.3680 | 0.4531 | 0.0275 | 0.6471 | 0.0275 | 0.0528 | 0.0652 | 1.7397 | 0.4332 | 0.0085 | — | — |
| model-exit0 | f_tools | 945 | 0.8000 | 0.8571 | 0.5714 | 0.7660 | 0.5714 | 0.6545 | 0.0319 | 0.5853 | 0.7908 | 0.1492 | — | — |
| model-exit0 | f_tools_masked | 945 | 0.0762 | 0.0952 | 0.0000 | — | 0.0000 | — | 0.7091 | 8.6487 | 0.7836 | 0.0000 | — | — |
| model-exit0 | f_metatool | 2000 | 0.3140 | 0.3431 | 0.1975 | 0.2524 | 0.1975 | 0.2216 | 0.2257 | 2.1521 | 0.5397 | 0.1565 | — | — |
| model-exit0 | f_clinc | 2000 | 0.4135 | 0.4863 | 0.1225 | 0.4757 | 0.1225 | 0.1948 | 0.0689 | 1.6373 | 0.4809 | 0.0515 | — | — |
| model-exit0 | f_massive | 2000 | 0.3690 | 0.4506 | 0.0425 | 0.4857 | 0.0425 | 0.0782 | 0.0765 | 1.6965 | 0.4450 | 0.0175 | — | — |
| potion-cos | f_tools | 945 | 0.7386 | 0.7831 | 0.5608 | 0.5856 | 0.5608 | 0.5730 | 0.0375 | 0.7642 | 0.7460 | 0.1915 | — | — |
| potion-cos | f_tools_masked | 945 | 0.7249 | 0.7963 | 0.4392 | 0.5461 | 0.4392 | 0.4868 | 0.0448 | 0.8335 | 0.7115 | 0.1608 | — | — |
| potion-cos | f_metatool | 2000 | 0.5350 | 0.5400 | 0.5150 | 0.3583 | 0.5150 | 0.4226 | 0.1225 | 1.4536 | 0.6568 | 0.2875 | — | — |
| potion-cos | f_clinc | 2000 | 0.7800 | 0.7750 | 0.8000 | 0.5442 | 0.8000 | 0.6478 | 0.0280 | 0.6244 | 0.8025 | 0.2940 | — | — |
| potion-cos | f_massive | 2000 | 0.4730 | 0.3887 | 0.8100 | 0.2632 | 0.8100 | 0.3973 | 0.2939 | 1.9417 | 0.7669 | 0.6155 | — | — |
| potion-cos-4bit | f_tools | 945 | 0.7386 | 0.7950 | 0.5132 | 0.5879 | 0.5132 | 0.5480 | 0.0243 | 0.7599 | 0.7441 | 0.1746 | — | — |
| potion-cos-4bit | f_tools_masked | 945 | 0.7280 | 0.8095 | 0.4021 | 0.5547 | 0.4021 | 0.4663 | 0.0361 | 0.8324 | 0.7103 | 0.1450 | — | — |
| potion-cos-4bit | f_metatool | 2000 | 0.5340 | 0.5531 | 0.4575 | 0.3526 | 0.4575 | 0.3983 | 0.1175 | 1.4348 | 0.6514 | 0.2595 | — | — |
| potion-cos-4bit | f_clinc | 2000 | 0.7855 | 0.7844 | 0.7900 | 0.5623 | 0.7900 | 0.6570 | 0.0272 | 0.6146 | 0.7953 | 0.2810 | — | — |
| potion-cos-4bit | f_massive | 2000 | 0.4755 | 0.3962 | 0.7925 | 0.2642 | 0.7925 | 0.3962 | 0.2799 | 1.8887 | 0.7554 | 0.6000 | — | — |
| static-ft | f_tools | 945 | 0.7450 | 0.8042 | 0.5079 | 0.6194 | 0.5079 | 0.5581 | 0.0458 | 0.7573 | 0.7568 | 0.1640 | — | — |
| static-ft | f_tools_masked | 945 | 0.7365 | 0.8148 | 0.4233 | 0.6250 | 0.4233 | 0.5047 | 0.0423 | 0.8247 | 0.7240 | 0.1354 | — | — |
| static-ft | f_metatool | 2000 | 0.5265 | 0.5469 | 0.4450 | 0.3518 | 0.4450 | 0.3929 | 0.1304 | 1.4547 | 0.6569 | 0.2530 | — | — |
| static-ft | f_clinc | 2000 | 0.7870 | 0.7769 | 0.8275 | 0.5554 | 0.8275 | 0.6647 | 0.0279 | 0.5912 | 0.8085 | 0.2980 | — | — |
| static-ft | f_massive | 2000 | 0.4795 | 0.3987 | 0.8025 | 0.2627 | 0.8025 | 0.3958 | 0.3028 | 1.9613 | 0.7814 | 0.6110 | — | — |
| pair-lr | f_tools | 945 | 0.3153 | 0.3532 | 0.1640 | 0.2793 | 0.1640 | 0.2067 | 0.1220 | 1.7825 | 0.4362 | 0.1175 | — | — |
| pair-lr | f_tools_masked | 945 | 0.4106 | 0.5132 | 0.0000 | — | 0.0000 | — | 0.1031 | 2.2761 | 0.5136 | 0.0000 | — | — |
| pair-lr | f_metatool | 2000 | 0.1605 | 0.2006 | 0.0000 | — | 0.0000 | — | 0.2770 | 2.6634 | 0.4375 | 0.0000 | — | — |
| pair-lr | f_clinc | 2000 | 0.1550 | 0.1938 | 0.0000 | 0.0000 | 0.0000 | — | 0.5102 | 4.0719 | 0.6652 | 0.0005 | — | — |
| pair-lr | f_massive | 2000 | 0.0975 | 0.1000 | 0.0875 | 0.2011 | 0.0875 | 0.1220 | 0.3605 | 2.7992 | 0.4580 | 0.0870 | — | — |
| bm25 | f_tools | 945 | 0.8169 | 0.8426 | 0.7143 | 0.5745 | 0.7143 | 0.6368 | 0.0779 | 0.6720 | 0.7555 | 0.2487 | — | — |
| bm25 | f_tools_masked | 945 | 0.8138 | 0.8307 | 0.7460 | 0.5685 | 0.7460 | 0.6453 | 0.0806 | 0.6781 | 0.7484 | 0.2624 | — | — |
| bm25 | f_metatool | 2000 | 0.4330 | 0.3738 | 0.6700 | 0.2672 | 0.6700 | 0.3820 | 0.1237 | 1.6309 | 0.5567 | 0.5015 | — | — |
| bm25 | f_clinc | 2000 | 0.4990 | 0.3881 | 0.9425 | 0.2873 | 0.9425 | 0.4404 | 0.1300 | 1.2430 | 0.5326 | 0.6560 | — | — |
| bm25 | f_massive | 2000 | 0.3205 | 0.1550 | 0.9825 | 0.2272 | 0.9825 | 0.3690 | 0.2305 | 1.7384 | 0.5148 | 0.8650 | — | — |
| minilm-cos | f_tools | 945 | 0.9048 | 0.9259 | 0.8201 | 0.7828 | 0.8201 | 0.8010 | 0.0254 | 0.3253 | 0.9156 | 0.2095 | — | — |
| minilm-cos | f_tools_masked | 945 | 0.9005 | 0.9259 | 0.7989 | 0.7906 | 0.7989 | 0.7947 | 0.0245 | 0.3486 | 0.9042 | 0.2021 | — | — |
| minilm-cos | f_metatool | 2000 | 0.6570 | 0.6913 | 0.5200 | 0.4185 | 0.5200 | 0.4638 | 0.1329 | 1.0589 | 0.7899 | 0.2485 | — | — |
| minilm-cos | f_clinc | 2000 | 0.8790 | 0.8675 | 0.9250 | 0.6740 | 0.9250 | 0.7798 | 0.0159 | 0.3390 | 0.8893 | 0.2745 | — | — |
| minilm-cos | f_massive | 2000 | 0.5730 | 0.5044 | 0.8475 | 0.3116 | 0.8475 | 0.4556 | 0.2487 | 1.4861 | 0.8199 | 0.5440 | — | — |
| toolrouter | f_tools | 945 | 0.7735 | 0.9167 | 0.2011 | 0.5507 | 0.2011 | 0.2946 | 0.0940 | 0.9660 | 0.8675 | 0.0730 | — | — |
| toolrouter | f_tools_masked | 945 | 0.7513 | 0.8915 | 0.1905 | 0.4675 | 0.1905 | 0.2707 | 0.0719 | 1.0340 | 0.8232 | 0.0815 | — | — |
| toolrouter | f_metatool | 2000 | 0.5910 | 0.6687 | 0.2800 | 0.3567 | 0.2800 | 0.3137 | 0.1803 | 1.7780 | 0.7695 | 0.1570 | — | — |
| toolrouter | f_clinc | 2000 | 0.7470 | 0.8625 | 0.2850 | 0.4691 | 0.2850 | 0.3546 | 0.1057 | 1.1428 | 0.8527 | 0.1215 | — | — |
| toolrouter | f_massive | 2000 | 0.5900 | 0.6875 | 0.2000 | 0.3791 | 0.2000 | 0.2619 | 0.1241 | 1.6449 | 0.7141 | 0.1055 | — | — |
| name-prior | f_tools | 945 | 0.2000 | 0.0000 | 1.0000 | 0.2000 | 1.0000 | 0.3333 | 0.8000 | 22.1048 | 1.0000 | 1.0000 | — | — |
| name-prior | f_tools_masked | 945 | 0.1238 | 0.1548 | 0.0000 | — | 0.0000 | — | 0.8762 | 24.2100 | 1.0000 | 0.0000 | — | — |
| name-prior | f_metatool | 2000 | 0.2000 | 0.0000 | 1.0000 | 0.2000 | 1.0000 | 0.3333 | 0.8000 | 22.1048 | 1.0000 | 1.0000 | — | — |
| name-prior | f_clinc | 2000 | 0.1565 | 0.0756 | 0.4800 | 0.1924 | 0.4800 | 0.2747 | 0.8435 | 23.3068 | 1.0000 | 0.4990 | — | — |
| name-prior | f_massive | 2000 | 0.0000 | 0.0000 | 0.0000 | — | 0.0000 | — | 1.0000 | 27.6310 | 1.0000 | 0.0000 | — | — |
| s6-route | f_tools | 945 | 0.8868 | 0.9101 | 0.7937 | 0.7538 | 0.7937 | 0.7732 | 0.0162 | 0.3767 | 0.8913 | 0.2106 | 0.9433 | 0.8593 |
| s6-route | f_tools_masked | 945 | 0.2497 | 0.3122 | 0.0000 | — | 0.0000 | — | 0.5357 | 5.5362 | 0.7855 | 0.0000 | 0.2669 | 0.7058 |
| s6-route | f_metatool | 2000 | 0.5175 | 0.5456 | 0.4050 | 0.3240 | 0.4050 | 0.3600 | 0.1820 | 1.5537 | 0.6995 | 0.2500 | 0.6565 | 0.5385 |
| s6-route | f_clinc | 2000 | 0.8040 | 0.8337 | 0.6850 | 0.6919 | 0.6850 | 0.6884 | 0.0135 | 0.5737 | 0.7923 | 0.1980 | 0.9331 | 0.7025 |
| s6-route | f_massive | 2000 | 0.5340 | 0.5156 | 0.6075 | 0.2896 | 0.6075 | 0.3923 | 0.1561 | 1.3537 | 0.6901 | 0.4195 | 0.6973 | 0.5285 |
| s6-route-fast | f_tools | 945 | 0.8836 | 0.9074 | 0.7884 | 0.7680 | 0.7884 | 0.7781 | 0.0189 | 0.3800 | 0.8929 | 0.2053 | 0.9442 | 0.8529 |
| s6-route-fast | f_tools_masked | 945 | 0.2899 | 0.3624 | 0.0000 | — | 0.0000 | — | 0.4869 | 4.8305 | 0.7769 | 0.0000 | 0.3040 | 0.6963 |
| s6-route-fast | f_metatool | 2000 | 0.5130 | 0.5300 | 0.4450 | 0.3296 | 0.4450 | 0.3787 | 0.1837 | 1.5367 | 0.6964 | 0.2700 | 0.6620 | 0.5385 |
| s6-route-fast | f_clinc | 2000 | 0.8020 | 0.8187 | 0.7350 | 0.6592 | 0.7350 | 0.6950 | 0.0208 | 0.5784 | 0.7930 | 0.2230 | 0.9269 | 0.7110 |
| s6-route-fast | f_massive | 2000 | 0.5300 | 0.4956 | 0.6675 | 0.2899 | 0.6675 | 0.4042 | 0.1808 | 1.4085 | 0.7098 | 0.4605 | 0.6545 | 0.5775 |
| bm25+potion-cos-4bit | f_tools | 945 | 0.8624 | 0.8942 | 0.7354 | 0.6915 | 0.7354 | 0.7128 | 0.0843 | 0.4956 | 0.7786 | 0.2127 | — | — |
| bm25+potion-cos-4bit | f_tools_masked | 945 | 0.8423 | 0.8783 | 0.6984 | 0.6316 | 0.6984 | 0.6633 | 0.0928 | 0.5666 | 0.7503 | 0.2212 | — | — |
| bm25+potion-cos-4bit | f_metatool | 2000 | 0.5420 | 0.5081 | 0.6775 | 0.3238 | 0.6775 | 0.4382 | 0.0492 | 1.2643 | 0.5863 | 0.4185 | — | — |
| bm25+potion-cos-4bit | f_clinc | 2000 | 0.7750 | 0.7338 | 0.9400 | 0.4928 | 0.9400 | 0.6466 | 0.0721 | 0.6812 | 0.7029 | 0.3815 | — | — |
| bm25+potion-cos-4bit | f_massive | 2000 | 0.4515 | 0.3306 | 0.9350 | 0.2636 | 0.9350 | 0.4112 | 0.2372 | 1.6169 | 0.6772 | 0.7095 | — | — |

Risk–coverage of the system on f_tools:

| coverage | accuracy | min confidence |
|---:|---:|---:|
| 0.050 | 1.0000 | 1.0000 |
| 0.099 | 1.0000 | 0.9999 |
| 0.150 | 1.0000 | 0.9999 |
| 0.200 | 0.9947 | 0.9997 |
| 0.250 | 0.9958 | 0.9996 |
| 0.301 | 0.9965 | 0.9992 |
| 0.350 | 0.9970 | 0.9986 |
| 0.400 | 0.9947 | 0.9973 |
| 0.450 | 0.9953 | 0.9955 |
| 0.499 | 0.9958 | 0.9922 |
| 0.550 | 0.9923 | 0.9840 |
| 0.600 | 0.9877 | 0.9695 |
| 0.650 | 0.9837 | 0.9390 |
| 0.701 | 0.9773 | 0.8982 |
| 0.750 | 0.9676 | 0.8541 |
| 0.800 | 0.9577 | 0.7698 |
| 0.850 | 0.9452 | 0.6828 |
| 0.899 | 0.9294 | 0.6009 |
| 0.950 | 0.9087 | 0.4950 |
| 1.000 | 0.8868 | 0.2468 |

## Latency

- the model, native (sky-eval --latency, 200 slates of f_tools): {"n": 200, "batch": 1, "default": {"p50_us": 24648.333, "p95_us": 45597.526799999985}, "no_cache": {"p50_us": 24801.6045, "p95_us": 46860.19615}}
- potion-cos: 0.79 ms of CPU per slate (process CPU time of the whole file, model loading and caches included)
- potion-cos-4bit: 0.75 ms of CPU per slate (process CPU time of the whole file, model loading and caches included)
- static-ft: 0.75 ms of CPU per slate (process CPU time of the whole file, model loading and caches included)
- pair-lr: 1.34 ms of CPU per slate (process CPU time of the whole file, model loading and caches included)
- bm25: 0.16 ms of CPU per slate (process CPU time of the whole file, model loading and caches included)
- minilm-cos: 12.25 ms of CPU per slate (process CPU time of the whole file, model loading and caches included)
- toolrouter: 0.47 ms of CPU per slate (process CPU time of the whole file, model loading and caches included)
- name-prior: 0.00 ms of CPU per slate (process CPU time of the whole file, model loading and caches included)
- WASM: median 30879.1 µs cached, first 45279.4 µs
