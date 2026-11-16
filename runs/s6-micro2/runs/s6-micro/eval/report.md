# Evaluation report: skycmd-s6-micro

Checkpoint: `runs/s6-micro2/runs/s6-micro-g3/decide/last`  
Split: `test` (33534 questions)  
Exits after state blocks [2, 4, 6]. Metrics use the calibrated temperatures of `calibration.json` per exit, type and bucket; ECE uses 15 bins on the top probability. The early-exit answer stops at the first exit whose calibrated top probability reaches τ of the question type.

## Summary

| model | last-exit acc | early-exit acc | τ (choice/score/noul) | mean layers | early-exit rate | perm. max diff | coherence pairs | mean abs(p(X)+p(¬X)−1) |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| GGUF `skycmd-s6-micro-g3-ternary.gguf` | 0.5808 | 0.5815 | 0.50/0.50/0.50 | 3.21 | 0.703 | 4.7e-7 | 0 | n/a |

Permutation invariance: max |Δp| over all exits when the options of up to 256 questions are shuffled, computed on the CPU device (equivariant by construction, so this only measures float rounding; on the Metal GPU, batch-dependent kernel rounding alone reaches ~1e-3). Coherence: noul questions with the same source and state whose instructions are negations of each other (`not`/`no`/`never` inserted or an antonym swap); `n/a` when the split has no such pairs.

## GGUF `skycmd-s6-micro-g3-ternary.gguf`

#### Exit 0 (after 2 state blocks), per source

| source | n | accuracy | NLL | Brier | ECE |
|---|---:|---:|---:|---:|---:|
| all | 33534 | 0.5807 | 1.0220 | 0.3912 | 0.0502 |
| game:dodge | 2991 | 0.9609 | 0.3357 | 0.0418 | 0.0946 |
| game:doom | 4731 | 0.9011 | 0.6396 | 0.0627 | 0.1063 |
| game:flappy | 3100 | 0.8471 | 0.4281 | 0.0099 | 0.0952 |
| game:rescue | 3043 | 0.9034 | 0.6053 | 0.0022 | 0.1149 |
| notam:deel_subject | 4169 | 0.0866 | 2.5351 | 0.9184 | 0.0253 |
| notam:notam_multi | 4882 | 0.3531 | 1.0412 | 0.6213 | 0.1423 |
| numeric:between | 870 | 0.6172 | 0.6760 | 0.4805 | 0.0426 |
| numeric:bucket | 842 | 0.7435 | 0.7030 | 0.3600 | 0.0491 |
| numeric:closest | 854 | 0.2354 | 1.4869 | 0.7678 | 0.0239 |
| numeric:extreme | 771 | 0.2490 | 1.4846 | 0.7677 | 0.0338 |
| numeric:greater | 843 | 0.5219 | 0.6917 | 0.4985 | 0.0315 |
| numeric:sign | 808 | 0.4530 | 1.0019 | 0.6061 | 0.0171 |
| typed_decisions_bench | 2000 | 0.5560 | 1.0526 | 0.1604 | 0.0571 |
| when2call | 3630 | 0.3862 | 1.1929 | 0.7153 | 0.2190 |

#### Exit 1 (after 4 state blocks), per source

| source | n | accuracy | NLL | Brier | ECE |
|---|---:|---:|---:|---:|---:|
| all | 33534 | 0.5804 | 1.0205 | 0.3904 | 0.0498 |
| game:dodge | 2991 | 0.9585 | 0.3382 | 0.0427 | 0.0938 |
| game:doom | 4731 | 0.9023 | 0.6383 | 0.0623 | 0.1081 |
| game:flappy | 3100 | 0.8432 | 0.4273 | 0.0093 | 0.0910 |
| game:rescue | 3043 | 0.9021 | 0.6048 | 0.0020 | 0.1145 |
| notam:deel_subject | 4169 | 0.0892 | 2.5323 | 0.9179 | 0.0224 |
| notam:notam_multi | 4882 | 0.3486 | 1.0383 | 0.6193 | 0.1447 |
| numeric:between | 870 | 0.6195 | 0.6746 | 0.4803 | 0.0431 |
| numeric:bucket | 842 | 0.7399 | 0.7009 | 0.3560 | 0.0459 |
| numeric:closest | 854 | 0.2365 | 1.4869 | 0.7678 | 0.0180 |
| numeric:extreme | 771 | 0.2374 | 1.4854 | 0.7679 | 0.0292 |
| numeric:greater | 843 | 0.5053 | 0.6907 | 0.4977 | 0.0492 |
| numeric:sign | 808 | 0.4728 | 0.9865 | 0.5985 | 0.0175 |
| typed_decisions_bench | 2000 | 0.5560 | 1.0534 | 0.1605 | 0.0519 |
| when2call | 3630 | 0.3928 | 1.1899 | 0.7150 | 0.2215 |

#### Exit 2 (after 6 state blocks), per source

| source | n | accuracy | NLL | Brier | ECE |
|---|---:|---:|---:|---:|---:|
| all | 33534 | 0.5808 | 1.0197 | 0.3900 | 0.0483 |
| game:dodge | 2991 | 0.9589 | 0.3394 | 0.0433 | 0.0956 |
| game:doom | 4731 | 0.9019 | 0.6387 | 0.0626 | 0.1086 |
| game:flappy | 3100 | 0.8352 | 0.4272 | 0.0092 | 0.0825 |
| game:rescue | 3043 | 0.9027 | 0.6051 | 0.0021 | 0.1159 |
| notam:deel_subject | 4169 | 0.0895 | 2.5298 | 0.9175 | 0.0216 |
| notam:notam_multi | 4882 | 0.3593 | 1.0338 | 0.6150 | 0.1322 |
| numeric:between | 870 | 0.6126 | 0.6735 | 0.4797 | 0.0375 |
| numeric:bucket | 842 | 0.7399 | 0.7074 | 0.3582 | 0.0348 |
| numeric:closest | 854 | 0.2272 | 1.4870 | 0.7678 | 0.0267 |
| numeric:extreme | 771 | 0.2387 | 1.4851 | 0.7679 | 0.0306 |
| numeric:greater | 843 | 0.5077 | 0.6904 | 0.4975 | 0.0413 |
| numeric:sign | 808 | 0.4740 | 0.9823 | 0.5968 | 0.0193 |
| typed_decisions_bench | 2000 | 0.5535 | 1.0546 | 0.1610 | 0.0504 |
| when2call | 3630 | 0.3926 | 1.1891 | 0.7157 | 0.2263 |

#### Early exit per source

| source | n | accuracy | mean layers | early-exit rate |
|---|---:|---:|---:|---:|
| all | 33534 | 0.5815 | 3.21 | 0.703 |
| game:dodge | 2991 | 0.9612 | 2.06 | 0.988 |
| game:doom | 4731 | 0.9013 | 2.12 | 0.972 |
| game:flappy | 3100 | 0.8471 | 2.00 | 1.000 |
| game:rescue | 3043 | 0.9017 | 2.61 | 0.852 |
| notam:deel_subject | 4169 | 0.0895 | 6.00 | 0.000 |
| notam:notam_multi | 4882 | 0.3531 | 2.62 | 0.844 |
| numeric:between | 870 | 0.6172 | 2.00 | 1.000 |
| numeric:bucket | 842 | 0.7470 | 2.62 | 0.861 |
| numeric:closest | 854 | 0.2272 | 6.00 | 0.000 |
| numeric:extreme | 771 | 0.2387 | 5.99 | 0.001 |
| numeric:greater | 843 | 0.5219 | 2.00 | 1.000 |
| numeric:sign | 808 | 0.4728 | 5.24 | 0.210 |
| typed_decisions_bench | 2000 | 0.5515 | 4.27 | 0.440 |
| when2call | 3630 | 0.3926 | 3.09 | 0.747 |

