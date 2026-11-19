# Evaluation report: skycmd-s6-nano

Checkpoint: `runs/s6-nano-g3/decide/last`  
Split: `test` (33534 questions)  
Exits after state blocks [1, 2, 3, 4]. Metrics use the calibrated temperatures of `calibration.json` per exit, type and bucket; ECE uses 15 bins on the top probability. The early-exit answer stops at the first exit whose calibrated top probability reaches τ of the question type.

## Summary

| model | last-exit acc | early-exit acc | τ (choice/score/noul) | mean layers | early-exit rate | perm. max diff | coherence pairs | mean abs(p(X)+p(¬X)−1) |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| GGUF `skycmd-s6-nano-g3-ternary.gguf` | 0.5908 | 0.5872 | 0.50/0.50/0.50 | 2.01 | 0.681 | 9.1e-7 | 0 | n/a |

Permutation invariance: max |Δp| over all exits when the options of up to 256 questions are shuffled, computed on the CPU device (equivariant by construction, so this only measures float rounding; on the Metal GPU, batch-dependent kernel rounding alone reaches ~1e-3). Coherence: noul questions with the same source and state whose instructions are negations of each other (`not`/`no`/`never` inserted or an antonym swap); `n/a` when the split has no such pairs.

## GGUF `skycmd-s6-nano-g3-ternary.gguf`

#### Exit 0 (after 1 state blocks), per source

| source | n | accuracy | NLL | Brier | ECE |
|---|---:|---:|---:|---:|---:|
| all | 33534 | 0.5748 | 1.0496 | 0.3974 | 0.0473 |
| game:dodge | 2991 | 0.9592 | 0.3415 | 0.0441 | 0.0885 |
| game:doom | 4731 | 0.8975 | 0.6584 | 0.0700 | 0.1020 |
| game:flappy | 3100 | 0.8397 | 0.4362 | 0.0153 | 0.0855 |
| game:rescue | 3043 | 0.7427 | 0.8377 | 0.1325 | 0.0887 |
| notam:deel_subject | 4169 | 0.0540 | 2.6013 | 0.9297 | 0.0659 |
| notam:notam_multi | 4882 | 0.4807 | 0.9762 | 0.5615 | 0.0989 |
| numeric:between | 870 | 0.6207 | 0.6666 | 0.4735 | 0.0220 |
| numeric:bucket | 842 | 0.6805 | 0.8913 | 0.4413 | 0.0424 |
| numeric:closest | 854 | 0.2506 | 1.4778 | 0.7640 | 0.0270 |
| numeric:extreme | 771 | 0.2348 | 1.4780 | 0.7638 | 0.0202 |
| numeric:greater | 843 | 0.5575 | 0.6850 | 0.4918 | 0.0086 |
| numeric:sign | 808 | 0.3923 | 1.0705 | 0.6477 | 0.0386 |
| typed_decisions_bench | 2000 | 0.4850 | 1.1256 | 0.1964 | 0.0386 |
| when2call | 3630 | 0.4022 | 1.1359 | 0.6720 | 0.1392 |

#### Exit 1 (after 2 state blocks), per source

| source | n | accuracy | NLL | Brier | ECE |
|---|---:|---:|---:|---:|---:|
| all | 33534 | 0.5912 | 1.0256 | 0.3833 | 0.0493 |
| game:dodge | 2991 | 0.9679 | 0.3257 | 0.0367 | 0.0931 |
| game:doom | 4731 | 0.9011 | 0.6422 | 0.0640 | 0.1015 |
| game:flappy | 3100 | 0.8510 | 0.4279 | 0.0096 | 0.0961 |
| game:rescue | 3043 | 0.9027 | 0.6143 | 0.0063 | 0.1088 |
| notam:deel_subject | 4169 | 0.0568 | 2.6148 | 0.9318 | 0.0659 |
| notam:notam_multi | 4882 | 0.4799 | 0.9757 | 0.5588 | 0.0659 |
| numeric:between | 870 | 0.6195 | 0.6674 | 0.4741 | 0.0257 |
| numeric:bucket | 842 | 0.7078 | 0.8573 | 0.4246 | 0.0591 |
| numeric:closest | 854 | 0.2447 | 1.4767 | 0.7635 | 0.0154 |
| numeric:extreme | 771 | 0.2114 | 1.4813 | 0.7654 | 0.0436 |
| numeric:greater | 843 | 0.5552 | 0.6851 | 0.4920 | 0.0063 |
| numeric:sign | 808 | 0.4084 | 1.0603 | 0.6422 | 0.0213 |
| typed_decisions_bench | 2000 | 0.4825 | 1.1265 | 0.1968 | 0.0281 |
| when2call | 3630 | 0.3942 | 1.1366 | 0.6724 | 0.1542 |

#### Exit 2 (after 3 state blocks), per source

| source | n | accuracy | NLL | Brier | ECE |
|---|---:|---:|---:|---:|---:|
| all | 33534 | 0.5918 | 1.0261 | 0.3832 | 0.0498 |
| game:dodge | 2991 | 0.9672 | 0.3247 | 0.0361 | 0.0901 |
| game:doom | 4731 | 0.9011 | 0.6421 | 0.0639 | 0.1020 |
| game:flappy | 3100 | 0.8516 | 0.4277 | 0.0094 | 0.0963 |
| game:rescue | 3043 | 0.9037 | 0.6113 | 0.0044 | 0.1080 |
| notam:deel_subject | 4169 | 0.0556 | 2.6242 | 0.9335 | 0.0737 |
| notam:notam_multi | 4882 | 0.4863 | 0.9737 | 0.5572 | 0.0617 |
| numeric:between | 870 | 0.6195 | 0.6682 | 0.4749 | 0.0219 |
| numeric:bucket | 842 | 0.7007 | 0.8565 | 0.4244 | 0.0575 |
| numeric:closest | 854 | 0.2447 | 1.4774 | 0.7638 | 0.0225 |
| numeric:extreme | 771 | 0.2088 | 1.4809 | 0.7652 | 0.0474 |
| numeric:greater | 843 | 0.5563 | 0.6851 | 0.4920 | 0.0092 |
| numeric:sign | 808 | 0.4121 | 1.0605 | 0.6424 | 0.0166 |
| typed_decisions_bench | 2000 | 0.4810 | 1.1276 | 0.1975 | 0.0279 |
| when2call | 3630 | 0.3942 | 1.1359 | 0.6732 | 0.1526 |

#### Exit 3 (after 4 state blocks), per source

| source | n | accuracy | NLL | Brier | ECE |
|---|---:|---:|---:|---:|---:|
| all | 33534 | 0.5908 | 1.0261 | 0.3832 | 0.0533 |
| game:dodge | 2991 | 0.9682 | 0.3240 | 0.0356 | 0.0916 |
| game:doom | 4731 | 0.9017 | 0.6418 | 0.0639 | 0.1030 |
| game:flappy | 3100 | 0.8535 | 0.4273 | 0.0091 | 0.0987 |
| game:rescue | 3043 | 0.9037 | 0.6111 | 0.0045 | 0.1078 |
| notam:deel_subject | 4169 | 0.0549 | 2.6279 | 0.9341 | 0.0748 |
| notam:notam_multi | 4882 | 0.4807 | 0.9751 | 0.5583 | 0.0780 |
| numeric:between | 870 | 0.6184 | 0.6683 | 0.4749 | 0.0167 |
| numeric:bucket | 842 | 0.6936 | 0.8550 | 0.4227 | 0.0566 |
| numeric:closest | 854 | 0.2342 | 1.4779 | 0.7640 | 0.0219 |
| numeric:extreme | 771 | 0.2114 | 1.4810 | 0.7653 | 0.0436 |
| numeric:greater | 843 | 0.5575 | 0.6856 | 0.4925 | 0.0106 |
| numeric:sign | 808 | 0.4109 | 1.0569 | 0.6410 | 0.0164 |
| typed_decisions_bench | 2000 | 0.4810 | 1.1274 | 0.1972 | 0.0266 |
| when2call | 3630 | 0.3939 | 1.1324 | 0.6720 | 0.1525 |

#### Early exit per source

| source | n | accuracy | mean layers | early-exit rate |
|---|---:|---:|---:|---:|
| all | 33534 | 0.5872 | 2.01 | 0.681 |
| game:dodge | 2991 | 0.9632 | 1.03 | 0.993 |
| game:doom | 4731 | 0.8985 | 1.07 | 0.983 |
| game:flappy | 3100 | 0.8397 | 1.00 | 1.000 |
| game:rescue | 3043 | 0.8929 | 1.52 | 0.938 |
| notam:deel_subject | 4169 | 0.0549 | 4.00 | 0.000 |
| notam:notam_multi | 4882 | 0.4787 | 1.47 | 0.844 |
| numeric:between | 870 | 0.6207 | 1.00 | 1.000 |
| numeric:bucket | 842 | 0.6912 | 1.59 | 0.831 |
| numeric:closest | 854 | 0.2342 | 4.00 | 0.000 |
| numeric:extreme | 771 | 0.2114 | 4.00 | 0.000 |
| numeric:greater | 843 | 0.5575 | 1.00 | 1.000 |
| numeric:sign | 808 | 0.4109 | 3.93 | 0.031 |
| typed_decisions_bench | 2000 | 0.4795 | 2.90 | 0.371 |
| when2call | 3630 | 0.3934 | 2.50 | 0.536 |

