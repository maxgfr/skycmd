# Evaluation report: skycmd-s6-secrets-pico

Checkpoint: `runs/s6-secrets-pico/decide/last`  
Split: `test` (7984 questions)  
Exits after state blocks [1, 2, 3, 4]. Metrics use the calibrated temperatures of `calibration.json` per exit, type and bucket; ECE uses 15 bins on the top probability. The early-exit answer stops at the first exit whose calibrated top probability reaches τ of the question type.

## Summary

| model | last-exit acc | early-exit acc | τ (choice/score/noul) | mean layers | early-exit rate | perm. max diff | coherence pairs | mean abs(p(X)+p(¬X)−1) |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| latent f32 (final QAT master weights) | 0.7124 | 0.7129 | 0.50/0.90/0.50 | 1.72 | 0.812 | 8.9e-6 | 0 | n/a |
| ternary (fake-quant) | 0.9160 | 0.8995 | 0.50/0.90/0.50 | 1.00 | 1.000 | 4.3e-7 | 0 | n/a |
| GGUF `skycmd-s6-secrets-pico-ternary.gguf` | 0.9158 | 0.8995 | 0.50/0.90/0.50 | 1.00 | 1.000 | 4.4e-7 | 0 | n/a |

Permutation invariance: max |Δp| over all exits when the options of up to 256 questions are shuffled, computed on the CPU device (equivariant by construction, so this only measures float rounding; on the Metal GPU, batch-dependent kernel rounding alone reaches ~1e-3). Coherence: noul questions with the same source and state whose instructions are negations of each other (`not`/`no`/`never` inserted or an antonym swap); `n/a` when the split has no such pairs.

## latent f32 (final QAT master weights)

#### Exit 0 (after 1 state blocks), per source

| source | n | accuracy | NLL | Brier | ECE |
|---|---:|---:|---:|---:|---:|
| all | 7984 | 0.6880 | 1.0522 | 0.4636 | 0.0549 |
| secret_bench:kind | 1996 | 0.7174 | 0.9152 | 0.4155 | 0.0683 |
| secret_bench:line | 1996 | 0.6353 | 1.1268 | 0.5225 | 0.0414 |
| secret_bench:secret | 1996 | 0.8176 | 0.4053 | 0.2662 | 0.0687 |
| secret_bench:service | 1996 | 0.5817 | 1.7615 | 0.6502 | 0.1778 |

#### Exit 1 (after 2 state blocks), per source

| source | n | accuracy | NLL | Brier | ECE |
|---|---:|---:|---:|---:|---:|
| all | 7984 | 0.7095 | 1.0339 | 0.4578 | 0.0685 |
| secret_bench:kind | 1996 | 0.7355 | 0.9501 | 0.4138 | 0.0688 |
| secret_bench:line | 1996 | 0.6528 | 1.0622 | 0.5030 | 0.0801 |
| secret_bench:secret | 1996 | 0.8211 | 0.4418 | 0.2851 | 0.0946 |
| secret_bench:service | 1996 | 0.6288 | 1.6814 | 0.6294 | 0.1990 |

#### Exit 2 (after 3 state blocks), per source

| source | n | accuracy | NLL | Brier | ECE |
|---|---:|---:|---:|---:|---:|
| all | 7984 | 0.7151 | 0.9972 | 0.4395 | 0.0576 |
| secret_bench:kind | 1996 | 0.7410 | 0.8898 | 0.3929 | 0.0843 |
| secret_bench:line | 1996 | 0.6613 | 1.0376 | 0.4890 | 0.0940 |
| secret_bench:secret | 1996 | 0.8267 | 0.4093 | 0.2638 | 0.0902 |
| secret_bench:service | 1996 | 0.6313 | 1.6523 | 0.6125 | 0.1811 |

#### Exit 3 (after 4 state blocks), per source

| source | n | accuracy | NLL | Brier | ECE |
|---|---:|---:|---:|---:|---:|
| all | 7984 | 0.7124 | 0.9876 | 0.4322 | 0.0548 |
| secret_bench:kind | 1996 | 0.7370 | 0.8592 | 0.3850 | 0.0788 |
| secret_bench:line | 1996 | 0.6583 | 1.0276 | 0.4831 | 0.0813 |
| secret_bench:secret | 1996 | 0.8322 | 0.3900 | 0.2494 | 0.0808 |
| secret_bench:service | 1996 | 0.6222 | 1.6736 | 0.6111 | 0.1714 |

#### Early exit per source

| source | n | accuracy | mean layers | early-exit rate |
|---|---:|---:|---:|---:|
| all | 7984 | 0.7129 | 1.72 | 0.812 |
| secret_bench:kind | 1996 | 0.7580 | 1.62 | 0.862 |
| secret_bench:line | 1996 | 0.6478 | 1.51 | 0.883 |
| secret_bench:secret | 1996 | 0.8176 | 1.00 | 1.000 |
| secret_bench:service | 1996 | 0.6283 | 2.75 | 0.504 |

## ternary (fake-quant)

#### Exit 0 (after 1 state blocks), per source

| source | n | accuracy | NLL | Brier | ECE |
|---|---:|---:|---:|---:|---:|
| all | 7984 | 0.8995 | 1.6411 | 0.1916 | 0.0946 |
| secret_bench:kind | 1996 | 0.9990 | 0.0025 | 0.0013 | 0.0019 |
| secret_bench:line | 1996 | 0.6227 | 6.4250 | 0.7237 | 0.3657 |
| secret_bench:secret | 1996 | 0.9990 | 0.0028 | 0.0017 | 0.0014 |
| secret_bench:service | 1996 | 0.9775 | 0.1340 | 0.0398 | 0.0168 |

#### Exit 1 (after 2 state blocks), per source

| source | n | accuracy | NLL | Brier | ECE |
|---|---:|---:|---:|---:|---:|
| all | 7984 | 0.9098 | 1.5487 | 0.1713 | 0.0836 |
| secret_bench:kind | 1996 | 0.9990 | 0.0026 | 0.0014 | 0.0007 |
| secret_bench:line | 1996 | 0.6623 | 6.0814 | 0.6460 | 0.3199 |
| secret_bench:secret | 1996 | 0.9990 | 0.0032 | 0.0018 | 0.0007 |
| secret_bench:service | 1996 | 0.9790 | 0.1078 | 0.0359 | 0.0143 |

#### Exit 2 (after 3 state blocks), per source

| source | n | accuracy | NLL | Brier | ECE |
|---|---:|---:|---:|---:|---:|
| all | 7984 | 0.9162 | 1.5597 | 0.1626 | 0.0797 |
| secret_bench:kind | 1996 | 0.9990 | 0.0036 | 0.0017 | 0.0007 |
| secret_bench:line | 1996 | 0.6864 | 6.1259 | 0.6141 | 0.3028 |
| secret_bench:secret | 1996 | 0.9990 | 0.0039 | 0.0019 | 0.0009 |
| secret_bench:service | 1996 | 0.9805 | 0.1054 | 0.0327 | 0.0145 |

#### Exit 3 (after 4 state blocks), per source

| source | n | accuracy | NLL | Brier | ECE |
|---|---:|---:|---:|---:|---:|
| all | 7984 | 0.9160 | 1.6498 | 0.1635 | 0.0803 |
| secret_bench:kind | 1996 | 0.9990 | 0.0049 | 0.0019 | 0.0010 |
| secret_bench:line | 1996 | 0.6864 | 6.4698 | 0.6163 | 0.3047 |
| secret_bench:secret | 1996 | 0.9990 | 0.0042 | 0.0019 | 0.0009 |
| secret_bench:service | 1996 | 0.9795 | 0.1202 | 0.0339 | 0.0156 |

#### Early exit per source

| source | n | accuracy | mean layers | early-exit rate |
|---|---:|---:|---:|---:|
| all | 7984 | 0.8995 | 1.00 | 1.000 |
| secret_bench:kind | 1996 | 0.9990 | 1.00 | 1.000 |
| secret_bench:line | 1996 | 0.6227 | 1.00 | 1.000 |
| secret_bench:secret | 1996 | 0.9990 | 1.00 | 1.000 |
| secret_bench:service | 1996 | 0.9775 | 1.00 | 1.000 |

## GGUF `skycmd-s6-secrets-pico-ternary.gguf`

#### Exit 0 (after 1 state blocks), per source

| source | n | accuracy | NLL | Brier | ECE |
|---|---:|---:|---:|---:|---:|
| all | 7984 | 0.8995 | 1.6420 | 0.1918 | 0.0949 |
| secret_bench:kind | 1996 | 0.9990 | 0.0024 | 0.0012 | 0.0019 |
| secret_bench:line | 1996 | 0.6227 | 6.4287 | 0.7243 | 0.3656 |
| secret_bench:secret | 1996 | 0.9990 | 0.0027 | 0.0017 | 0.0012 |
| secret_bench:service | 1996 | 0.9775 | 0.1342 | 0.0399 | 0.0166 |

#### Exit 1 (after 2 state blocks), per source

| source | n | accuracy | NLL | Brier | ECE |
|---|---:|---:|---:|---:|---:|
| all | 7984 | 0.9099 | 1.5504 | 0.1718 | 0.0836 |
| secret_bench:kind | 1996 | 0.9990 | 0.0026 | 0.0014 | 0.0007 |
| secret_bench:line | 1996 | 0.6628 | 6.0877 | 0.6479 | 0.3198 |
| secret_bench:secret | 1996 | 0.9990 | 0.0031 | 0.0018 | 0.0007 |
| secret_bench:service | 1996 | 0.9790 | 0.1081 | 0.0360 | 0.0149 |

#### Exit 2 (after 3 state blocks), per source

| source | n | accuracy | NLL | Brier | ECE |
|---|---:|---:|---:|---:|---:|
| all | 7984 | 0.9156 | 1.5611 | 0.1630 | 0.0797 |
| secret_bench:kind | 1996 | 0.9990 | 0.0036 | 0.0017 | 0.0007 |
| secret_bench:line | 1996 | 0.6844 | 6.1314 | 0.6158 | 0.3031 |
| secret_bench:secret | 1996 | 0.9990 | 0.0039 | 0.0019 | 0.0009 |
| secret_bench:service | 1996 | 0.9800 | 0.1056 | 0.0328 | 0.0151 |

#### Exit 3 (after 4 state blocks), per source

| source | n | accuracy | NLL | Brier | ECE |
|---|---:|---:|---:|---:|---:|
| all | 7984 | 0.9158 | 1.6514 | 0.1639 | 0.0804 |
| secret_bench:kind | 1996 | 0.9990 | 0.0049 | 0.0019 | 0.0010 |
| secret_bench:line | 1996 | 0.6859 | 6.4763 | 0.6178 | 0.3051 |
| secret_bench:secret | 1996 | 0.9990 | 0.0042 | 0.0019 | 0.0009 |
| secret_bench:service | 1996 | 0.9795 | 0.1204 | 0.0340 | 0.0154 |

#### Early exit per source

| source | n | accuracy | mean layers | early-exit rate |
|---|---:|---:|---:|---:|
| all | 7984 | 0.8995 | 1.00 | 1.000 |
| secret_bench:kind | 1996 | 0.9990 | 1.00 | 1.000 |
| secret_bench:line | 1996 | 0.6227 | 1.00 | 1.000 |
| secret_bench:secret | 1996 | 0.9990 | 1.00 | 1.000 |
| secret_bench:service | 1996 | 0.9775 | 1.00 | 1.000 |

