# Evaluation report: skycmd-s6-sleep3-nano

Checkpoint: `runs/s6-sleep3/nano/sleep-s2/b/last`  
Split: `test` (2000 questions)  
Exits after state blocks [1, 2, 3, 4]. Metrics use the calibrated temperatures of `calibration.json` per exit, type and bucket; ECE uses 15 bins on the top probability. The early-exit answer stops at the first exit whose calibrated top probability reaches τ of the question type.

## Summary

| model | last-exit acc | early-exit acc | τ (choice/score/noul) | mean layers | early-exit rate | perm. max diff | coherence pairs | mean abs(p(X)+p(¬X)−1) |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| latent f32 (final QAT master weights) | 0.4430 | 0.4430 | 0.50/0.56/0.50 | 2.88 | 0.405 | 2.6e-6 | 0 | n/a |
| ternary (fake-quant) | 0.4845 | 0.4825 | 0.50/0.50/0.50 | 2.86 | 0.395 | 9.5e-7 | 0 | n/a |

Permutation invariance: max |Δp| over all exits when the options of up to 256 questions are shuffled, computed on the CPU device (equivariant by construction, so this only measures float rounding; on the Metal GPU, batch-dependent kernel rounding alone reaches ~1e-3). Coherence: noul questions with the same source and state whose instructions are negations of each other (`not`/`no`/`never` inserted or an antonym swap); `n/a` when the split has no such pairs.

## latent f32 (final QAT master weights)

#### Exit 0 (after 1 state blocks), per source

| source | n | accuracy | NLL | Brier | ECE |
|---|---:|---:|---:|---:|---:|
| all | 2000 | 0.4395 | 1.1794 | 0.2190 | 0.0292 |
| typed_decisions_bench | 2000 | 0.4395 | 1.1794 | 0.2190 | 0.0292 |

#### Exit 1 (after 2 state blocks), per source

| source | n | accuracy | NLL | Brier | ECE |
|---|---:|---:|---:|---:|---:|
| all | 2000 | 0.4505 | 1.1894 | 0.2259 | 0.0515 |
| typed_decisions_bench | 2000 | 0.4505 | 1.1894 | 0.2259 | 0.0515 |

#### Exit 2 (after 3 state blocks), per source

| source | n | accuracy | NLL | Brier | ECE |
|---|---:|---:|---:|---:|---:|
| all | 2000 | 0.4440 | 1.1876 | 0.2249 | 0.0551 |
| typed_decisions_bench | 2000 | 0.4440 | 1.1876 | 0.2249 | 0.0551 |

#### Exit 3 (after 4 state blocks), per source

| source | n | accuracy | NLL | Brier | ECE |
|---|---:|---:|---:|---:|---:|
| all | 2000 | 0.4430 | 1.1935 | 0.2266 | 0.0465 |
| typed_decisions_bench | 2000 | 0.4430 | 1.1935 | 0.2266 | 0.0465 |

#### Early exit per source

| source | n | accuracy | mean layers | early-exit rate |
|---|---:|---:|---:|---:|
| all | 2000 | 0.4430 | 2.88 | 0.405 |
| typed_decisions_bench | 2000 | 0.4430 | 2.88 | 0.405 |

## ternary (fake-quant)

#### Exit 0 (after 1 state blocks), per source

| source | n | accuracy | NLL | Brier | ECE |
|---|---:|---:|---:|---:|---:|
| all | 2000 | 0.4810 | 1.1122 | 0.1902 | 0.0636 |
| typed_decisions_bench | 2000 | 0.4810 | 1.1122 | 0.1902 | 0.0636 |

#### Exit 1 (after 2 state blocks), per source

| source | n | accuracy | NLL | Brier | ECE |
|---|---:|---:|---:|---:|---:|
| all | 2000 | 0.4805 | 1.1093 | 0.1892 | 0.0555 |
| typed_decisions_bench | 2000 | 0.4805 | 1.1093 | 0.1892 | 0.0555 |

#### Exit 2 (after 3 state blocks), per source

| source | n | accuracy | NLL | Brier | ECE |
|---|---:|---:|---:|---:|---:|
| all | 2000 | 0.4825 | 1.1102 | 0.1896 | 0.0595 |
| typed_decisions_bench | 2000 | 0.4825 | 1.1102 | 0.1896 | 0.0595 |

#### Exit 3 (after 4 state blocks), per source

| source | n | accuracy | NLL | Brier | ECE |
|---|---:|---:|---:|---:|---:|
| all | 2000 | 0.4845 | 1.1098 | 0.1890 | 0.0567 |
| typed_decisions_bench | 2000 | 0.4845 | 1.1098 | 0.1890 | 0.0567 |

#### Early exit per source

| source | n | accuracy | mean layers | early-exit rate |
|---|---:|---:|---:|---:|
| all | 2000 | 0.4825 | 2.86 | 0.395 |
| typed_decisions_bench | 2000 | 0.4825 | 2.86 | 0.395 |

