# s6-route (s6-nano v3): rehearsal on the validation slates (not the gate's verdict)

**Disclosure.** v3 was designed after v2's test was read, from an exploratory reading of v2's test dumps. Its parameters come from validation. Its gate is read on requests no system had scored; on T2-tools these requests are about the same held-out tools as v2's test. The latency bar is 30 times v2's and was set after v2 was measured. The system needs distinctive tool names.

System: p(i) ∝ (p_model(i) · p_potion-cos-4bit(i))^beta, β = 0.81, τ_ask3 = 0.671047 (runs/s6-nano3/fit.json); potion-cos-4bit θ = 0.396120, T = 0.0671021; 3,422,864 bytes.

**Rehearsal check: the fused system on v_tools reproduces runs/s6-nano3/fit.json** (accuracy 0.9135 against 0.9135; ECE 0.0254 with fit.json's bins against 0.0254, 0.0254 with v2's scorer; NLL 0.2608; τ_ask3 0.671047; model dump vs runs/s6-nano2/v_tools.dump.jsonl: max |Δp| 0.000000000).

| condition | value | bar | pass |
|---|---:|---:|:---:|
| G1 system − bm25 on v_tools | 0.0586 [0.0402, 0.0778] | > 0, lower bound > 0 | yes |
| G1 system vs minilm-cos − 0.03 | 0.9135 | ≥ 0.8826 | yes |
| G1 system vs the model alone | 0.9135 | > 0.8339 | yes |
| G1 system vs potion-cos-4bit alone | 0.9135 | > 0.7229 | yes |
| G2 | — | no other file on validation | no |
| G3 ECE on v_tools (15 bins) | 0.0254 | ≤ 0.05 | yes |
| G4 | not measured | the WASM bench's JSON | no |

Gate: **not passed** (a rehearsal)

Paired bootstrap of the accuracy differences on v_tools (1,000 resamples of the slates, seed 0, 95%):

| system vs | difference | interval |
|---|---:|---|
| bm25 | 0.0586 | [0.0402, 0.0778] |
| minilm-cos | 0.0009 | [-0.0175, 0.0184] |
| model | 0.0795 | [0.0621, 0.0979] |
| potion-cos-4bit | 0.1906 | [0.1670, 0.2150] |
| bm25+potion-cos-4bit | 0.0437 | [0.0280, 0.0612] |
| s6-route-fast | 0.0079 | [0.0000, 0.0157] |

## Announced goals (not gates)

- accuracy ≥ 0.87 on v_tools: 0.9135 (met)
- ≥ 0.95 among answered at τ_ask3 with coverage ≥ 0.80: 0.9508 at coverage 0.8881 (met)
- the fast mode within 0.015 of full depth: 0.9056 against 0.9135 (met)

## Readings of section 6 (reported, not gated)

Clusters: 732 (stand-in: the gold tool's name (validation slates carry no cluster file)); the three largest hold 28 slates.

Without the three largest clusters (1116 slates):

| system | accuracy |
|---|---:|
| model | 0.8297 |
| model-exit0 | 0.8280 |
| potion-cos | 0.7195 |
| potion-cos-4bit | 0.7168 |
| static-ft | 0.7294 |
| pair-lr | 0.4758 |
| bm25 | 0.8522 |
| minilm-cos | 0.9104 |
| toolrouter | 0.7778 |
| name-prior | 0.2052 |
| s6-route | 0.9113 |
| s6-route-fast | 0.9032 |
| bm25+potion-cos-4bit | 0.8665 |

| system vs | difference, without the three largest (slate bootstrap) | interval | difference, all slates (cluster bootstrap) | interval |
|---|---:|---|---:|---|
| bm25 | 0.0591 | [0.0394, 0.0797] | 0.0586 | [0.0389, 0.0804] |
| minilm-cos | 0.0009 | [-0.0179, 0.0197] | 0.0009 | [-0.0190, 0.0207] |
| model | 0.0815 | [0.0618, 0.1022] | 0.0795 | [0.0594, 0.1003] |
| potion-cos-4bit | 0.1944 | [0.1711, 0.2195] | 0.1906 | [0.1620, 0.2194] |

Names masked (v_tools_masked; the system is expected to fail there):

| system | accuracy |
|---|---:|
| model | 0.1171 |
| model-exit0 | 0.1215 |
| potion-cos | 0.7325 |
| potion-cos-4bit | 0.7351 |
| static-ft | 0.7517 |
| pair-lr | 0.4100 |
| bm25 | 0.8444 |
| minilm-cos | 0.9003 |
| toolrouter | 0.7622 |
| name-prior | 0.1101 |
| s6-route | 0.3086 |
| s6-route-fast | 0.3462 |
| bm25+potion-cos-4bit | 0.8575 |

## Every system, every file

| system | file | n | accuracy | tool gold | none gold | none P | none R | none F1 | ECE | NLL | mean conf | none answers | at τ_ask3: accuracy | coverage |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| model | v_tools | 1144 | 0.8339 | 0.8918 | 0.6026 | 0.8214 | 0.6026 | 0.6952 | 0.0272 | 0.4749 | 0.8410 | 0.1469 | — | — |
| model | v_tools_masked | 1144 | 0.1171 | 0.1464 | 0.0000 | 0.0000 | 0.0000 | — | 0.6852 | 8.5058 | 0.8000 | 0.0009 | — | — |
| model-exit0 | v_tools | 1144 | 0.8322 | 0.8863 | 0.6157 | 0.8103 | 0.6157 | 0.6998 | 0.0363 | 0.5048 | 0.8270 | 0.1521 | — | — |
| model-exit0 | v_tools_masked | 1144 | 0.1215 | 0.1519 | 0.0000 | 0.0000 | 0.0000 | — | 0.6665 | 7.7626 | 0.7880 | 0.0026 | — | — |
| potion-cos | v_tools | 1144 | 0.7255 | 0.7781 | 0.5153 | 0.5756 | 0.5153 | 0.5438 | 0.0198 | 0.7940 | 0.7223 | 0.1792 | — | — |
| potion-cos | v_tools_masked | 1144 | 0.7325 | 0.8055 | 0.4410 | 0.5580 | 0.4410 | 0.4927 | 0.0542 | 0.7634 | 0.7123 | 0.1582 | — | — |
| potion-cos-4bit | v_tools | 1144 | 0.7229 | 0.7836 | 0.4803 | 0.6077 | 0.4803 | 0.5366 | 0.0297 | 0.8030 | 0.7207 | 0.1582 | — | — |
| potion-cos-4bit | v_tools_masked | 1144 | 0.7351 | 0.8175 | 0.4061 | 0.5776 | 0.4061 | 0.4769 | 0.0451 | 0.7764 | 0.7092 | 0.1407 | — | — |
| static-ft | v_tools | 1144 | 0.7351 | 0.7934 | 0.5022 | 0.6216 | 0.5022 | 0.5556 | 0.0378 | 0.7679 | 0.7325 | 0.1617 | — | — |
| static-ft | v_tools_masked | 1144 | 0.7517 | 0.8372 | 0.4105 | 0.6267 | 0.4105 | 0.4960 | 0.0433 | 0.7525 | 0.7215 | 0.1311 | — | — |
| pair-lr | v_tools | 1144 | 0.4843 | 0.5421 | 0.2533 | 0.3766 | 0.2533 | 0.3029 | 0.0484 | 1.4573 | 0.4610 | 0.1346 | — | — |
| pair-lr | v_tools_masked | 1144 | 0.4100 | 0.5126 | 0.0000 | — | 0.0000 | — | 0.1053 | 2.1756 | 0.5152 | 0.0000 | — | — |
| bm25 | v_tools | 1144 | 0.8549 | 0.9311 | 0.5502 | 0.7500 | 0.5502 | 0.6348 | 0.0712 | 0.5498 | 0.8260 | 0.1469 | — | — |
| bm25 | v_tools_masked | 1144 | 0.8444 | 0.9137 | 0.5677 | 0.6915 | 0.5677 | 0.6235 | 0.0600 | 0.5478 | 0.8137 | 0.1643 | — | — |
| minilm-cos | v_tools | 1144 | 0.9126 | 0.9279 | 0.8515 | 0.7800 | 0.8515 | 0.8142 | 0.0209 | 0.2807 | 0.9057 | 0.2185 | — | — |
| minilm-cos | v_tools_masked | 1144 | 0.9003 | 0.9213 | 0.8166 | 0.7510 | 0.8166 | 0.7824 | 0.0134 | 0.2914 | 0.8964 | 0.2177 | — | — |
| toolrouter | v_tools | 1144 | 0.7832 | 0.9377 | 0.1659 | 0.5758 | 0.1659 | 0.2576 | 0.0860 | 1.0545 | 0.8687 | 0.0577 | — | — |
| toolrouter | v_tools_masked | 1144 | 0.7622 | 0.9104 | 0.1703 | 0.4333 | 0.1703 | 0.2445 | 0.0607 | 1.1560 | 0.8230 | 0.0787 | — | — |
| name-prior | v_tools | 1144 | 0.2002 | 0.0000 | 1.0000 | 0.2002 | 1.0000 | 0.3336 | 0.7998 | 22.1000 | 1.0000 | 1.0000 | — | — |
| name-prior | v_tools_masked | 1144 | 0.1101 | 0.1377 | 0.0000 | — | 0.0000 | — | 0.8899 | 24.5877 | 1.0000 | 0.0000 | — | — |
| s6-route | v_tools | 1144 | 0.9135 | 0.9552 | 0.7467 | 0.8636 | 0.7467 | 0.8009 | 0.0254 | 0.2608 | 0.9119 | 0.1731 | 0.9508 | 0.8881 |
| s6-route | v_tools_masked | 1144 | 0.3086 | 0.3858 | 0.0000 | — | 0.0000 | — | 0.4794 | 4.8178 | 0.7879 | 0.0000 | 0.3502 | 0.7063 |
| s6-route-fast | v_tools | 1144 | 0.9056 | 0.9519 | 0.7205 | 0.8594 | 0.7205 | 0.7838 | 0.0179 | 0.2773 | 0.9087 | 0.1678 | 0.9507 | 0.8872 |
| s6-route-fast | v_tools_masked | 1144 | 0.3462 | 0.4328 | 0.0000 | — | 0.0000 | — | 0.4367 | 4.2869 | 0.7828 | 0.0000 | 0.3832 | 0.6958 |
| bm25+potion-cos-4bit | v_tools | 1144 | 0.8698 | 0.9366 | 0.6026 | 0.7667 | 0.6026 | 0.6748 | 0.0536 | 0.4155 | 0.8161 | 0.1573 | — | — |
| bm25+potion-cos-4bit | v_tools_masked | 1144 | 0.8575 | 0.9333 | 0.5546 | 0.7175 | 0.5546 | 0.6256 | 0.0576 | 0.4638 | 0.8009 | 0.1547 | — | — |

Risk–coverage of the system on v_tools:

| coverage | accuracy | min confidence |
|---:|---:|---:|
| 0.050 | 1.0000 | 1.0000 |
| 0.100 | 1.0000 | 1.0000 |
| 0.150 | 1.0000 | 0.9999 |
| 0.200 | 1.0000 | 0.9998 |
| 0.250 | 1.0000 | 0.9997 |
| 0.300 | 1.0000 | 0.9995 |
| 0.350 | 1.0000 | 0.9990 |
| 0.400 | 0.9978 | 0.9979 |
| 0.450 | 0.9981 | 0.9965 |
| 0.500 | 0.9965 | 0.9940 |
| 0.550 | 0.9968 | 0.9892 |
| 0.600 | 0.9956 | 0.9798 |
| 0.650 | 0.9933 | 0.9682 |
| 0.700 | 0.9863 | 0.9454 |
| 0.750 | 0.9814 | 0.9083 |
| 0.800 | 0.9749 | 0.8567 |
| 0.850 | 0.9630 | 0.7677 |
| 0.900 | 0.9466 | 0.6482 |
| 0.950 | 0.9393 | 0.5081 |
| 1.000 | 0.9135 | 0.3034 |

## Latency

- the model, native (sky-eval --latency, 200 slates of v_tools): {"n": 200, "batch": 1, "default": {"p50_us": 25993.4375, "p95_us": 57267.577099999995}, "no_cache": {"p50_us": 31563.7915, "p95_us": 74655.67704999991}}
- potion-cos: 0.21 ms of CPU per slate (process CPU time of the whole file, model loading and caches included)
- potion-cos-4bit: 0.18 ms of CPU per slate (process CPU time of the whole file, model loading and caches included)
- static-ft: 0.23 ms of CPU per slate (process CPU time of the whole file, model loading and caches included)
- pair-lr: 1.42 ms of CPU per slate (process CPU time of the whole file, model loading and caches included)
- bm25: 0.17 ms of CPU per slate (process CPU time of the whole file, model loading and caches included)
- minilm-cos: 11.25 ms of CPU per slate (process CPU time of the whole file, model loading and caches included)
- toolrouter: 0.55 ms of CPU per slate (process CPU time of the whole file, model loading and caches included)
- name-prior: 0.00 ms of CPU per slate (process CPU time of the whole file, model loading and caches included)
- WASM: not measured
