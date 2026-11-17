# Evaluation report: skycmd-s6-nano-control3

Checkpoint: `runs/s6-sleep3/nano/control-s0/b/last`  
Split: `test` (2000 questions)  
Exits after state blocks [1, 2, 3, 4]. Metrics use the calibrated temperatures of `calibration.json` per exit, type and bucket; ECE uses 15 bins on the top probability. The early-exit answer stops at the first exit whose calibrated top probability reaches τ of the question type.

## Summary

| model | last-exit acc | early-exit acc | τ (choice/score/noul) | mean layers | early-exit rate | perm. max diff | coherence pairs | mean abs(p(X)+p(¬X)−1) |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| latent f32 (final QAT master weights) | 0.4645 | 0.4565 | 0.50/0.50/0.50 | 2.84 | 0.424 | 3.1e-6 | 0 | n/a |
| ternary (fake-quant) | 0.4910 | 0.4885 | 0.50/0.50/0.50 | 3.07 | 0.315 | 9.5e-7 | 0 | n/a |

Permutation invariance: max |Δp| over all exits when the options of up to 256 questions are shuffled, computed on the CPU device (equivariant by construction, so this only measures float rounding; on the Metal GPU, batch-dependent kernel rounding alone reaches ~1e-3). Coherence: noul questions with the same source and state whose instructions are negations of each other (`not`/`no`/`never` inserted or an antonym swap); `n/a` when the split has no such pairs.

## latent f32 (final QAT master weights)

#### Exit 0 (after 1 state blocks), per source

| source | n | accuracy | NLL | Brier | ECE |
|---|---:|---:|---:|---:|---:|
| all | 2000 | 0.4415 | 1.1819 | 0.2189 | 0.0500 |
| typed_decisions_bench | 2000 | 0.4415 | 1.1819 | 0.2189 | 0.0500 |

#### Exit 1 (after 2 state blocks), per source

| source | n | accuracy | NLL | Brier | ECE |
|---|---:|---:|---:|---:|---:|
| all | 2000 | 0.4735 | 1.1712 | 0.2132 | 0.0355 |
| typed_decisions_bench | 2000 | 0.4735 | 1.1712 | 0.2132 | 0.0355 |

#### Exit 2 (after 3 state blocks), per source

| source | n | accuracy | NLL | Brier | ECE |
|---|---:|---:|---:|---:|---:|
| all | 2000 | 0.4640 | 1.1787 | 0.2169 | 0.0306 |
| typed_decisions_bench | 2000 | 0.4640 | 1.1787 | 0.2169 | 0.0306 |

#### Exit 3 (after 4 state blocks), per source

| source | n | accuracy | NLL | Brier | ECE |
|---|---:|---:|---:|---:|---:|
| all | 2000 | 0.4645 | 1.1830 | 0.2192 | 0.0398 |
| typed_decisions_bench | 2000 | 0.4645 | 1.1830 | 0.2192 | 0.0398 |

#### Early exit per source

| source | n | accuracy | mean layers | early-exit rate |
|---|---:|---:|---:|---:|
| all | 2000 | 0.4565 | 2.84 | 0.424 |
| typed_decisions_bench | 2000 | 0.4565 | 2.84 | 0.424 |

## ternary (fake-quant)

#### Exit 0 (after 1 state blocks), per source

| source | n | accuracy | NLL | Brier | ECE |
|---|---:|---:|---:|---:|---:|
| all | 2000 | 0.4840 | 1.1135 | 0.1890 | 0.0539 |
| typed_decisions_bench | 2000 | 0.4840 | 1.1135 | 0.1890 | 0.0539 |

#### Exit 1 (after 2 state blocks), per source

| source | n | accuracy | NLL | Brier | ECE |
|---|---:|---:|---:|---:|---:|
| all | 2000 | 0.4955 | 1.1100 | 0.1871 | 0.0654 |
| typed_decisions_bench | 2000 | 0.4955 | 1.1100 | 0.1871 | 0.0654 |

#### Exit 2 (after 3 state blocks), per source

| source | n | accuracy | NLL | Brier | ECE |
|---|---:|---:|---:|---:|---:|
| all | 2000 | 0.4945 | 1.1107 | 0.1876 | 0.0632 |
| typed_decisions_bench | 2000 | 0.4945 | 1.1107 | 0.1876 | 0.0632 |

#### Exit 3 (after 4 state blocks), per source

| source | n | accuracy | NLL | Brier | ECE |
|---|---:|---:|---:|---:|---:|
| all | 2000 | 0.4910 | 1.1088 | 0.1870 | 0.0632 |
| typed_decisions_bench | 2000 | 0.4910 | 1.1088 | 0.1870 | 0.0632 |

#### Early exit per source

| source | n | accuracy | mean layers | early-exit rate |
|---|---:|---:|---:|---:|
| all | 2000 | 0.4885 | 3.07 | 0.315 |
| typed_decisions_bench | 2000 | 0.4885 | 3.07 | 0.315 |

