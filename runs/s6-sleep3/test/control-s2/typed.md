# Evaluation report: skycmd-s6-nano-control3

Checkpoint: `runs/s6-sleep3/nano/control-s2/b/last`  
Split: `test` (2000 questions)  
Exits after state blocks [1, 2, 3, 4]. Metrics use the calibrated temperatures of `calibration.json` per exit, type and bucket; ECE uses 15 bins on the top probability. The early-exit answer stops at the first exit whose calibrated top probability reaches τ of the question type.

## Summary

| model | last-exit acc | early-exit acc | τ (choice/score/noul) | mean layers | early-exit rate | perm. max diff | coherence pairs | mean abs(p(X)+p(¬X)−1) |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| latent f32 (final QAT master weights) | 0.4130 | 0.4155 | 0.50/0.53/0.51 | 2.81 | 0.435 | 2.8e-6 | 0 | n/a |
| ternary (fake-quant) | 0.4925 | 0.4930 | 0.50/0.50/0.50 | 2.95 | 0.353 | 8.7e-7 | 0 | n/a |

Permutation invariance: max |Δp| over all exits when the options of up to 256 questions are shuffled, computed on the CPU device (equivariant by construction, so this only measures float rounding; on the Metal GPU, batch-dependent kernel rounding alone reaches ~1e-3). Coherence: noul questions with the same source and state whose instructions are negations of each other (`not`/`no`/`never` inserted or an antonym swap); `n/a` when the split has no such pairs.

## latent f32 (final QAT master weights)

#### Exit 0 (after 1 state blocks), per source

| source | n | accuracy | NLL | Brier | ECE |
|---|---:|---:|---:|---:|---:|
| all | 2000 | 0.4195 | 1.1889 | 0.2268 | 0.0361 |
| typed_decisions_bench | 2000 | 0.4195 | 1.1889 | 0.2268 | 0.0361 |

#### Exit 1 (after 2 state blocks), per source

| source | n | accuracy | NLL | Brier | ECE |
|---|---:|---:|---:|---:|---:|
| all | 2000 | 0.4225 | 1.1938 | 0.2315 | 0.0597 |
| typed_decisions_bench | 2000 | 0.4225 | 1.1938 | 0.2315 | 0.0597 |

#### Exit 2 (after 3 state blocks), per source

| source | n | accuracy | NLL | Brier | ECE |
|---|---:|---:|---:|---:|---:|
| all | 2000 | 0.4175 | 1.2029 | 0.2364 | 0.0694 |
| typed_decisions_bench | 2000 | 0.4175 | 1.2029 | 0.2364 | 0.0694 |

#### Exit 3 (after 4 state blocks), per source

| source | n | accuracy | NLL | Brier | ECE |
|---|---:|---:|---:|---:|---:|
| all | 2000 | 0.4130 | 1.2094 | 0.2395 | 0.0766 |
| typed_decisions_bench | 2000 | 0.4130 | 1.2094 | 0.2395 | 0.0766 |

#### Early exit per source

| source | n | accuracy | mean layers | early-exit rate |
|---|---:|---:|---:|---:|
| all | 2000 | 0.4155 | 2.81 | 0.435 |
| typed_decisions_bench | 2000 | 0.4155 | 2.81 | 0.435 |

## ternary (fake-quant)

#### Exit 0 (after 1 state blocks), per source

| source | n | accuracy | NLL | Brier | ECE |
|---|---:|---:|---:|---:|---:|
| all | 2000 | 0.4845 | 1.1040 | 0.1854 | 0.0424 |
| typed_decisions_bench | 2000 | 0.4845 | 1.1040 | 0.1854 | 0.0424 |

#### Exit 1 (after 2 state blocks), per source

| source | n | accuracy | NLL | Brier | ECE |
|---|---:|---:|---:|---:|---:|
| all | 2000 | 0.4920 | 1.1032 | 0.1847 | 0.0453 |
| typed_decisions_bench | 2000 | 0.4920 | 1.1032 | 0.1847 | 0.0453 |

#### Exit 2 (after 3 state blocks), per source

| source | n | accuracy | NLL | Brier | ECE |
|---|---:|---:|---:|---:|---:|
| all | 2000 | 0.4895 | 1.1041 | 0.1850 | 0.0417 |
| typed_decisions_bench | 2000 | 0.4895 | 1.1041 | 0.1850 | 0.0417 |

#### Exit 3 (after 4 state blocks), per source

| source | n | accuracy | NLL | Brier | ECE |
|---|---:|---:|---:|---:|---:|
| all | 2000 | 0.4925 | 1.1044 | 0.1848 | 0.0469 |
| typed_decisions_bench | 2000 | 0.4925 | 1.1044 | 0.1848 | 0.0469 |

#### Early exit per source

| source | n | accuracy | mean layers | early-exit rate |
|---|---:|---:|---:|---:|
| all | 2000 | 0.4930 | 2.95 | 0.353 |
| typed_decisions_bench | 2000 | 0.4930 | 2.95 | 0.353 |

