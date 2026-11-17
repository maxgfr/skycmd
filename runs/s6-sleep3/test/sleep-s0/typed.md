# Evaluation report: skycmd-s6-sleep3-nano

Checkpoint: `runs/s6-sleep3/nano/sleep-s0/b/last`  
Split: `test` (2000 questions)  
Exits after state blocks [1, 2, 3, 4]. Metrics use the calibrated temperatures of `calibration.json` per exit, type and bucket; ECE uses 15 bins on the top probability. The early-exit answer stops at the first exit whose calibrated top probability reaches τ of the question type.

## Summary

| model | last-exit acc | early-exit acc | τ (choice/score/noul) | mean layers | early-exit rate | perm. max diff | coherence pairs | mean abs(p(X)+p(¬X)−1) |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| latent f32 (final QAT master weights) | 0.3755 | 0.3720 | 0.50/0.50/0.50 | 2.80 | 0.423 | 3.5e-6 | 0 | n/a |
| ternary (fake-quant) | 0.4980 | 0.4960 | 0.50/0.50/0.50 | 3.04 | 0.322 | 1.0e-6 | 0 | n/a |

Permutation invariance: max |Δp| over all exits when the options of up to 256 questions are shuffled, computed on the CPU device (equivariant by construction, so this only measures float rounding; on the Metal GPU, batch-dependent kernel rounding alone reaches ~1e-3). Coherence: noul questions with the same source and state whose instructions are negations of each other (`not`/`no`/`never` inserted or an antonym swap); `n/a` when the split has no such pairs.

## latent f32 (final QAT master weights)

#### Exit 0 (after 1 state blocks), per source

| source | n | accuracy | NLL | Brier | ECE |
|---|---:|---:|---:|---:|---:|
| all | 2000 | 0.3780 | 1.2332 | 0.2589 | 0.0903 |
| typed_decisions_bench | 2000 | 0.3780 | 1.2332 | 0.2589 | 0.0903 |

#### Exit 1 (after 2 state blocks), per source

| source | n | accuracy | NLL | Brier | ECE |
|---|---:|---:|---:|---:|---:|
| all | 2000 | 0.3910 | 1.2368 | 0.2613 | 0.0856 |
| typed_decisions_bench | 2000 | 0.3910 | 1.2368 | 0.2613 | 0.0856 |

#### Exit 2 (after 3 state blocks), per source

| source | n | accuracy | NLL | Brier | ECE |
|---|---:|---:|---:|---:|---:|
| all | 2000 | 0.3835 | 1.2429 | 0.2643 | 0.0879 |
| typed_decisions_bench | 2000 | 0.3835 | 1.2429 | 0.2643 | 0.0879 |

#### Exit 3 (after 4 state blocks), per source

| source | n | accuracy | NLL | Brier | ECE |
|---|---:|---:|---:|---:|---:|
| all | 2000 | 0.3755 | 1.2450 | 0.2645 | 0.0856 |
| typed_decisions_bench | 2000 | 0.3755 | 1.2450 | 0.2645 | 0.0856 |

#### Early exit per source

| source | n | accuracy | mean layers | early-exit rate |
|---|---:|---:|---:|---:|
| all | 2000 | 0.3720 | 2.80 | 0.423 |
| typed_decisions_bench | 2000 | 0.3720 | 2.80 | 0.423 |

## ternary (fake-quant)

#### Exit 0 (after 1 state blocks), per source

| source | n | accuracy | NLL | Brier | ECE |
|---|---:|---:|---:|---:|---:|
| all | 2000 | 0.4930 | 1.1063 | 0.1860 | 0.0818 |
| typed_decisions_bench | 2000 | 0.4930 | 1.1063 | 0.1860 | 0.0818 |

#### Exit 1 (after 2 state blocks), per source

| source | n | accuracy | NLL | Brier | ECE |
|---|---:|---:|---:|---:|---:|
| all | 2000 | 0.4970 | 1.1072 | 0.1863 | 0.0781 |
| typed_decisions_bench | 2000 | 0.4970 | 1.1072 | 0.1863 | 0.0781 |

#### Exit 2 (after 3 state blocks), per source

| source | n | accuracy | NLL | Brier | ECE |
|---|---:|---:|---:|---:|---:|
| all | 2000 | 0.4945 | 1.1080 | 0.1867 | 0.0768 |
| typed_decisions_bench | 2000 | 0.4945 | 1.1080 | 0.1867 | 0.0768 |

#### Exit 3 (after 4 state blocks), per source

| source | n | accuracy | NLL | Brier | ECE |
|---|---:|---:|---:|---:|---:|
| all | 2000 | 0.4980 | 1.1098 | 0.1876 | 0.0840 |
| typed_decisions_bench | 2000 | 0.4980 | 1.1098 | 0.1876 | 0.0840 |

#### Early exit per source

| source | n | accuracy | mean layers | early-exit rate |
|---|---:|---:|---:|---:|
| all | 2000 | 0.4960 | 3.04 | 0.322 |
| typed_decisions_bench | 2000 | 0.4960 | 3.04 | 0.322 |

