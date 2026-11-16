# s6-micro: results (docs/s6-gen-prereg.md)

| model | params | typed_decisions_bench | zs_massive_en | zs_massive_fr | when2call | notam:deel_subject | notam:notam_multi | all |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| s6-micro (τ = 1) | 5,587,201 | 0.553 | 0.531 | 0.314 | 0.401 | 0.092 | 0.353 | 0.380 |
| s3-pico (τ = 1) | 445,345 | 0.382 | 0.187 | 0.117 | 0.368 | 0.163 | 0.319 | 0.268 |
| s4-micro (τ = 1) | 5,587,201 | 0.352 | 0.335 | 0.240 | 0.399 | 0.065 | 0.298 | 0.285 |
| s5-lite (τ = 1) | 17,060,097 | 0.696 | 0.726 | 0.444 | 0.376 | 0.177 | 0.554 | 0.500 |

Published typed_decisions_bench accuracies: s3-pico 0.382, s4-micro 0.343, oscar-1-17m 0.6775, s5-lite 0.696. Majority class: 0.398.

| model | ECE (all) | mean block passes | early-exit rate | median µs / decision (native) |
|---|---:|---:|---:|---:|
| s6-micro (τ = 1) | 0.0588 | 6.00 | 0.000 | 8970 |
| s3-pico (τ = 1) | 0.0629 | 4.00 | 0.000 | 590 |
| s4-micro (τ = 1) | 0.0666 | 6.00 | 0.000 | 9139 |
| s5-lite (τ = 1) | 0.0328 | 0.00 | 0.000 | 11825 |

In distribution (trained on, not zero-shot): zs_banking77 0.763 (n 2000).

| arena mean score | flappy | rescue | dodge | doom |
|---|---:|---:|---:|---:|
| s3-pico | 200.00 | 3.00 | 9.47 | 3.27 |
| s6-micro (τ = 1) | 200.00 | 3.00 | 9.47 | 3.27 |
| s6-micro (CRC) | 200.00 | 3.00 | 9.47 | 3.27 |

Truncated (first 500 records per source, n 258309): state 14.6%, head 11.6%, either 24.0%.
On typed_decisions_bench: state 4.2%, head 0.0%.
Pretraining: 31,700 steps, 0.52B tokens (93 per parameter), epoch 5, MLM loss 2.081.
WASM (200 typed_decisions_bench decisions): median 60673 µs, max |p_wasm - p_native| 8.94e-07, argmax agree 200.

- gate at τ = 1: holds
- gate at CRC: fails (CRC thresholds not kept)
- announced goal typed_decisions_bench >= 0.5: met (0.5530)
- gate: PASS: publish at full depth
