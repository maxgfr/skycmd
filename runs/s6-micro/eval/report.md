# Evaluation report: skycmd-s6-micro

Checkpoint: `runs/s6-micro-g3/decide/last`  
Split: `test` (33534 questions)  
Exits after state blocks [2, 4, 6]. Metrics use the calibrated temperatures of `calibration.json` per exit, type and bucket; ECE uses 15 bins on the top probability. The early-exit answer stops at the first exit whose calibrated top probability reaches τ of the question type.

## Summary

| model | last-exit acc | early-exit acc | τ (choice/score/noul) | mean layers | early-exit rate | perm. max diff | coherence pairs | mean abs(p(X)+p(¬X)−1) |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| GGUF `skycmd-s6-micro-g3-ternary.gguf` | 0.5838 | 0.5825 | 0.50/0.50/0.50 | 3.33 | 0.671 | 4.2e-7 | 0 | n/a |

Permutation invariance: max |Δp| over all exits when the options of up to 256 questions are shuffled, computed on the CPU device (equivariant by construction, so this only measures float rounding; on the Metal GPU, batch-dependent kernel rounding alone reaches ~1e-3). Coherence: noul questions with the same source and state whose instructions are negations of each other (`not`/`no`/`never` inserted or an antonym swap); `n/a` when the split has no such pairs.

## GGUF `skycmd-s6-micro-g3-ternary.gguf`

#### Exit 0 (after 2 state blocks), per source

| source | n | accuracy | NLL | Brier | ECE |
|---|---:|---:|---:|---:|---:|
| all | 33534 | 0.5808 | 1.0308 | 0.3874 | 0.0662 |
| game:dodge | 2991 | 0.9626 | 0.3241 | 0.0359 | 0.0957 |
| game:doom | 4731 | 0.9036 | 0.6355 | 0.0609 | 0.1120 |
| game:flappy | 3100 | 0.8506 | 0.4264 | 0.0085 | 0.0956 |
| game:rescue | 3043 | 0.9017 | 0.6059 | 0.0028 | 0.1168 |
| notam:deel_subject | 4169 | 0.0521 | 2.6444 | 0.9363 | 0.0717 |
| notam:notam_multi | 4882 | 0.3689 | 1.0556 | 0.6233 | 0.2508 |
| numeric:between | 870 | 0.6161 | 0.6650 | 0.4723 | 0.0313 |
| numeric:bucket | 842 | 0.7304 | 0.7482 | 0.3788 | 0.0493 |
| numeric:closest | 854 | 0.2155 | 1.4822 | 0.7668 | 0.0415 |
| numeric:extreme | 771 | 0.2451 | 1.4805 | 0.7654 | 0.0164 |
| numeric:greater | 843 | 0.5492 | 0.6856 | 0.4926 | 0.0575 |
| numeric:sign | 808 | 0.3911 | 1.0448 | 0.6342 | 0.0752 |
| typed_decisions_bench | 2000 | 0.5485 | 1.0594 | 0.1638 | 0.0722 |
| when2call | 3630 | 0.4193 | 1.1271 | 0.6572 | 0.1237 |

#### Exit 1 (after 4 state blocks), per source

| source | n | accuracy | NLL | Brier | ECE |
|---|---:|---:|---:|---:|---:|
| all | 33534 | 0.5820 | 1.0273 | 0.3857 | 0.0697 |
| game:dodge | 2991 | 0.9672 | 0.3219 | 0.0349 | 0.0965 |
| game:doom | 4731 | 0.9049 | 0.6346 | 0.0606 | 0.1086 |
| game:flappy | 3100 | 0.8535 | 0.4257 | 0.0083 | 0.0962 |
| game:rescue | 3043 | 0.8942 | 0.6049 | 0.0022 | 0.1118 |
| notam:deel_subject | 4169 | 0.0525 | 2.6383 | 0.9352 | 0.0663 |
| notam:notam_multi | 4882 | 0.3722 | 1.0492 | 0.6172 | 0.2530 |
| numeric:between | 870 | 0.6184 | 0.6634 | 0.4708 | 0.0203 |
| numeric:bucket | 842 | 0.7268 | 0.7463 | 0.3779 | 0.0508 |
| numeric:closest | 854 | 0.2213 | 1.4812 | 0.7664 | 0.0445 |
| numeric:extreme | 771 | 0.2335 | 1.4790 | 0.7647 | 0.0203 |
| numeric:greater | 843 | 0.5480 | 0.6860 | 0.4930 | 0.0416 |
| numeric:sign | 808 | 0.4022 | 1.0397 | 0.6304 | 0.0629 |
| typed_decisions_bench | 2000 | 0.5505 | 1.0577 | 0.1633 | 0.0703 |
| when2call | 3630 | 0.4226 | 1.1177 | 0.6547 | 0.1231 |

#### Exit 2 (after 6 state blocks), per source

| source | n | accuracy | NLL | Brier | ECE |
|---|---:|---:|---:|---:|---:|
| all | 33534 | 0.5838 | 1.0262 | 0.3852 | 0.0712 |
| game:dodge | 2991 | 0.9682 | 0.3214 | 0.0346 | 0.0968 |
| game:doom | 4731 | 0.9055 | 0.6348 | 0.0606 | 0.1066 |
| game:flappy | 3100 | 0.8555 | 0.4257 | 0.0083 | 0.0978 |
| game:rescue | 3043 | 0.8958 | 0.6048 | 0.0021 | 0.1093 |
| notam:deel_subject | 4169 | 0.0544 | 2.6349 | 0.9345 | 0.0615 |
| notam:notam_multi | 4882 | 0.3748 | 1.0454 | 0.6141 | 0.2521 |
| numeric:between | 870 | 0.6184 | 0.6623 | 0.4697 | 0.0298 |
| numeric:bucket | 842 | 0.7304 | 0.7509 | 0.3799 | 0.0671 |
| numeric:closest | 854 | 0.2213 | 1.4812 | 0.7664 | 0.0393 |
| numeric:extreme | 771 | 0.2399 | 1.4784 | 0.7645 | 0.0204 |
| numeric:greater | 843 | 0.5457 | 0.6866 | 0.4935 | 0.0201 |
| numeric:sign | 808 | 0.4072 | 1.0420 | 0.6303 | 0.0574 |
| typed_decisions_bench | 2000 | 0.5475 | 1.0580 | 0.1635 | 0.0665 |
| when2call | 3630 | 0.4270 | 1.1158 | 0.6546 | 0.1174 |

#### Early exit per source

| source | n | accuracy | mean layers | early-exit rate |
|---|---:|---:|---:|---:|
| all | 33534 | 0.5825 | 3.33 | 0.671 |
| game:dodge | 2991 | 0.9656 | 2.06 | 0.987 |
| game:doom | 4731 | 0.9055 | 2.11 | 0.975 |
| game:flappy | 3100 | 0.8506 | 2.00 | 1.000 |
| game:rescue | 3043 | 0.8985 | 2.64 | 0.841 |
| notam:deel_subject | 4169 | 0.0544 | 6.00 | 0.000 |
| notam:notam_multi | 4882 | 0.3689 | 2.62 | 0.844 |
| numeric:between | 870 | 0.6161 | 2.00 | 1.000 |
| numeric:bucket | 842 | 0.7328 | 2.70 | 0.848 |
| numeric:closest | 854 | 0.2213 | 6.00 | 0.001 |
| numeric:extreme | 771 | 0.2399 | 5.99 | 0.003 |
| numeric:greater | 843 | 0.5492 | 2.00 | 1.000 |
| numeric:sign | 808 | 0.4072 | 5.19 | 0.224 |
| typed_decisions_bench | 2000 | 0.5450 | 4.36 | 0.421 |
| when2call | 3630 | 0.4275 | 4.19 | 0.466 |

