# s6-nano: results (docs/s6-gen-prereg.md)

| model | params | typed_decisions_bench | zs_massive_en | zs_massive_fr | when2call | notam:deel_subject | notam:notam_multi | all |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| s6-nano (τ = 1) | 1,486,225 | 0.480 | 0.425 | 0.239 | 0.404 | 0.055 | 0.493 | 0.360 |
| s3-pico (τ = 1) | 445,345 | 0.382 | 0.187 | 0.117 | 0.368 | 0.163 | 0.319 | 0.268 |
| s4-micro (τ = 1) | 5,587,201 | 0.352 | 0.335 | 0.240 | 0.399 | 0.065 | 0.298 | 0.285 |
| s5-lite (τ = 1) | 17,060,097 | 0.696 | 0.726 | 0.444 | 0.376 | 0.177 | 0.554 | 0.500 |

Published typed_decisions_bench accuracies: s3-pico 0.382, s4-micro 0.343, oscar-1-17m 0.6775, s5-lite 0.696. Majority class: 0.398.

| model | ECE (all) | mean block passes | early-exit rate | median µs / decision (native) |
|---|---:|---:|---:|---:|
| s6-nano (τ = 1) | 0.0479 | 4.00 | 0.000 | 1488 |
| s3-pico (τ = 1) | 0.0629 | 4.00 | 0.000 | 518 |
| s4-micro (τ = 1) | 0.0666 | 6.00 | 0.000 | 7340 |
| s5-lite (τ = 1) | 0.0328 | 0.00 | 0.000 | 10090 |

In distribution (trained on, not zero-shot): zs_banking77 0.631 (n 2000).

| arena mean score | flappy | rescue | dodge | doom |
|---|---:|---:|---:|---:|
| s3-pico | 200.00 | 3.00 | 9.47 | 3.27 |
| s6-nano (τ = 1) | 200.00 | 3.00 | 9.47 | 3.27 |
| s6-nano (CRC) | 200.00 | 2.80 | 9.33 | 3.27 |

Truncated (first 500 records per source, n 258309): state 12.5%, head 10.5%, either 21.7%.
On typed_decisions_bench: state 1.4%, head 0.0%.
Pretraining: 53,360 steps, 0.87B tokens (588 per parameter), epoch 8, MLM loss 2.356.
WASM (200 typed_decisions_bench decisions): median 7321 µs, max |p_wasm - p_native| 4.51e-02, argmax agree 199.

- gate at τ = 1: holds
- gate at CRC: fails (CRC thresholds not kept)
- gate: PASS: publish at full depth
