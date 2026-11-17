# Evaluation report: skycmd-s6-sleep3-nano

Checkpoint: `runs/s6-sleep3/nano/sleep-s1/b/last`  
Split: `test` (2000 questions)  
Exits after state blocks [1, 2, 3, 4]. Metrics use the calibrated temperatures of `calibration.json` per exit, type and bucket; ECE uses 15 bins on the top probability. The early-exit answer stops at the first exit whose calibrated top probability reaches τ of the question type.

## Summary

| model | last-exit acc | early-exit acc | τ (choice/score/noul) | mean layers | early-exit rate | perm. max diff | coherence pairs | mean abs(p(X)+p(¬X)−1) |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| latent f32 (final QAT master weights) | 0.4130 | 0.4170 | 0.50/0.50/0.57 | 3.08 | 0.353 | 3.8e-6 | 0 | n/a |
| ternary (fake-quant) | 0.4865 | 0.4825 | 0.50/0.50/0.50 | 2.97 | 0.355 | 9.5e-7 | 0 | n/a |

Permutation invariance: max |Δp| over all exits when the options of up to 256 questions are shuffled, computed on the CPU device (equivariant by construction, so this only measures float rounding; on the Metal GPU, batch-dependent kernel rounding alone reaches ~1e-3). Coherence: noul questions with the same source and state whose instructions are negations of each other (`not`/`no`/`never` inserted or an antonym swap); `n/a` when the split has no such pairs.

## latent f32 (final QAT master weights)

#### Exit 0 (after 1 state blocks), per source

| source | n | accuracy | NLL | Brier | ECE |
|---|---:|---:|---:|---:|---:|
| all | 2000 | 0.3890 | 1.2027 | 0.2337 | 0.0444 |
| typed_decisions_bench | 2000 | 0.3890 | 1.2027 | 0.2337 | 0.0444 |

#### Exit 1 (after 2 state blocks), per source

| source | n | accuracy | NLL | Brier | ECE |
|---|---:|---:|---:|---:|---:|
| all | 2000 | 0.4340 | 1.1956 | 0.2292 | 0.0351 |
| typed_decisions_bench | 2000 | 0.4340 | 1.1956 | 0.2292 | 0.0351 |

#### Exit 2 (after 3 state blocks), per source

| source | n | accuracy | NLL | Brier | ECE |
|---|---:|---:|---:|---:|---:|
| all | 2000 | 0.4205 | 1.1971 | 0.2300 | 0.0349 |
| typed_decisions_bench | 2000 | 0.4205 | 1.1971 | 0.2300 | 0.0349 |

#### Exit 3 (after 4 state blocks), per source

| source | n | accuracy | NLL | Brier | ECE |
|---|---:|---:|---:|---:|---:|
| all | 2000 | 0.4130 | 1.1962 | 0.2301 | 0.0471 |
| typed_decisions_bench | 2000 | 0.4130 | 1.1962 | 0.2301 | 0.0471 |

#### Early exit per source

| source | n | accuracy | mean layers | early-exit rate |
|---|---:|---:|---:|---:|
| all | 2000 | 0.4170 | 3.08 | 0.353 |
| typed_decisions_bench | 2000 | 0.4170 | 3.08 | 0.353 |

## ternary (fake-quant)

#### Exit 0 (after 1 state blocks), per source

| source | n | accuracy | NLL | Brier | ECE |
|---|---:|---:|---:|---:|---:|
| all | 2000 | 0.4825 | 1.1176 | 0.1917 | 0.0720 |
| typed_decisions_bench | 2000 | 0.4825 | 1.1176 | 0.1917 | 0.0720 |

#### Exit 1 (after 2 state blocks), per source

| source | n | accuracy | NLL | Brier | ECE |
|---|---:|---:|---:|---:|---:|
| all | 2000 | 0.4865 | 1.1120 | 0.1895 | 0.0740 |
| typed_decisions_bench | 2000 | 0.4865 | 1.1120 | 0.1895 | 0.0740 |

#### Exit 2 (after 3 state blocks), per source

| source | n | accuracy | NLL | Brier | ECE |
|---|---:|---:|---:|---:|---:|
| all | 2000 | 0.4830 | 1.1133 | 0.1899 | 0.0652 |
| typed_decisions_bench | 2000 | 0.4830 | 1.1133 | 0.1899 | 0.0652 |

#### Exit 3 (after 4 state blocks), per source

| source | n | accuracy | NLL | Brier | ECE |
|---|---:|---:|---:|---:|---:|
| all | 2000 | 0.4865 | 1.1129 | 0.1897 | 0.0706 |
| typed_decisions_bench | 2000 | 0.4865 | 1.1129 | 0.1897 | 0.0706 |

#### Early exit per source

| source | n | accuracy | mean layers | early-exit rate |
|---|---:|---:|---:|---:|
| all | 2000 | 0.4825 | 2.97 | 0.355 |
| typed_decisions_bench | 2000 | 0.4825 | 2.97 | 0.355 |

