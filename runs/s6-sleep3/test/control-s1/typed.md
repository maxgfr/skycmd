# Evaluation report: skycmd-s6-nano-control3

Checkpoint: `runs/s6-sleep3/nano/control-s1/b/last`  
Split: `test` (2000 questions)  
Exits after state blocks [1, 2, 3, 4]. Metrics use the calibrated temperatures of `calibration.json` per exit, type and bucket; ECE uses 15 bins on the top probability. The early-exit answer stops at the first exit whose calibrated top probability reaches τ of the question type.

## Summary

| model | last-exit acc | early-exit acc | τ (choice/score/noul) | mean layers | early-exit rate | perm. max diff | coherence pairs | mean abs(p(X)+p(¬X)−1) |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| latent f32 (final QAT master weights) | 0.4460 | 0.4510 | 0.50/0.50/0.59 | 3.08 | 0.364 | 4.5e-6 | 0 | n/a |
| ternary (fake-quant) | 0.4590 | 0.4515 | 0.50/0.50/0.50 | 2.81 | 0.405 | 9.5e-7 | 0 | n/a |

Permutation invariance: max |Δp| over all exits when the options of up to 256 questions are shuffled, computed on the CPU device (equivariant by construction, so this only measures float rounding; on the Metal GPU, batch-dependent kernel rounding alone reaches ~1e-3). Coherence: noul questions with the same source and state whose instructions are negations of each other (`not`/`no`/`never` inserted or an antonym swap); `n/a` when the split has no such pairs.

## latent f32 (final QAT master weights)

#### Exit 0 (after 1 state blocks), per source

| source | n | accuracy | NLL | Brier | ECE |
|---|---:|---:|---:|---:|---:|
| all | 2000 | 0.4345 | 1.1770 | 0.2179 | 0.0309 |
| typed_decisions_bench | 2000 | 0.4345 | 1.1770 | 0.2179 | 0.0309 |

#### Exit 1 (after 2 state blocks), per source

| source | n | accuracy | NLL | Brier | ECE |
|---|---:|---:|---:|---:|---:|
| all | 2000 | 0.4525 | 1.1743 | 0.2176 | 0.0312 |
| typed_decisions_bench | 2000 | 0.4525 | 1.1743 | 0.2176 | 0.0312 |

#### Exit 2 (after 3 state blocks), per source

| source | n | accuracy | NLL | Brier | ECE |
|---|---:|---:|---:|---:|---:|
| all | 2000 | 0.4490 | 1.1801 | 0.2214 | 0.0430 |
| typed_decisions_bench | 2000 | 0.4490 | 1.1801 | 0.2214 | 0.0430 |

#### Exit 3 (after 4 state blocks), per source

| source | n | accuracy | NLL | Brier | ECE |
|---|---:|---:|---:|---:|---:|
| all | 2000 | 0.4460 | 1.1808 | 0.2226 | 0.0517 |
| typed_decisions_bench | 2000 | 0.4460 | 1.1808 | 0.2226 | 0.0517 |

#### Early exit per source

| source | n | accuracy | mean layers | early-exit rate |
|---|---:|---:|---:|---:|
| all | 2000 | 0.4510 | 3.08 | 0.364 |
| typed_decisions_bench | 2000 | 0.4510 | 3.08 | 0.364 |

## ternary (fake-quant)

#### Exit 0 (after 1 state blocks), per source

| source | n | accuracy | NLL | Brier | ECE |
|---|---:|---:|---:|---:|---:|
| all | 2000 | 0.4515 | 1.1139 | 0.1920 | 0.0519 |
| typed_decisions_bench | 2000 | 0.4515 | 1.1139 | 0.1920 | 0.0519 |

#### Exit 1 (after 2 state blocks), per source

| source | n | accuracy | NLL | Brier | ECE |
|---|---:|---:|---:|---:|---:|
| all | 2000 | 0.4545 | 1.1149 | 0.1924 | 0.0615 |
| typed_decisions_bench | 2000 | 0.4545 | 1.1149 | 0.1924 | 0.0615 |

#### Exit 2 (after 3 state blocks), per source

| source | n | accuracy | NLL | Brier | ECE |
|---|---:|---:|---:|---:|---:|
| all | 2000 | 0.4540 | 1.1144 | 0.1920 | 0.0569 |
| typed_decisions_bench | 2000 | 0.4540 | 1.1144 | 0.1920 | 0.0569 |

#### Exit 3 (after 4 state blocks), per source

| source | n | accuracy | NLL | Brier | ECE |
|---|---:|---:|---:|---:|---:|
| all | 2000 | 0.4590 | 1.1148 | 0.1921 | 0.0637 |
| typed_decisions_bench | 2000 | 0.4590 | 1.1148 | 0.1921 | 0.0637 |

#### Early exit per source

| source | n | accuracy | mean layers | early-exit rate |
|---|---:|---:|---:|---:|
| all | 2000 | 0.4515 | 2.81 | 0.405 |
| typed_decisions_bench | 2000 | 0.4515 | 2.81 | 0.405 |

