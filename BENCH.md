# skycmd-s1 training throughput

Measured by `sky-train bench` (mlx-rs 0.32 / MLX 0.32.2, Metal, compiled loss + gradient) on random tokens. Warmup steps are excluded; timing covers forward + backward + optimizer update (Muon on the encoder hidden matrices, AdamW elsewhere). `bf16` runs matmuls and activations in bfloat16 with f32 master weights and f32 losses. `tok/s` counts input tokens (batch x seq); `params` counts the trained model (encoder + MLM head for `mlm`, encoder + decision head for `decision`).

| model | params | phase | precision | batch | seq | step (ms) | tok/s | peak mem (MB) | steps |
|---|---:|---|---|---:|---:|---:|---:|---:|---|
| s1-pico | 152960 | mlm | bf16 | 128 | 128 | 24.3 | 673236 | 367 | 10+50 |
| s1-pico | 202305 | decision | bf16 | 64 | 192 | 20.5 | 598104 | 308 | 10+50 |
| s1-nano | 937216 | mlm | bf16 | 128 | 128 | 55.8 | 293518 | 1035 | 10+50 |
| s1-nano | 1134209 | decision | bf16 | 64 | 192 | 50.6 | 243002 | 926 | 10+50 |
| s1-micro | 5053952 | mlm | bf16 | 128 | 128 | 172.7 | 94883 | 1939 | 10+50 |
| s1-micro | 6630913 | decision | bf16 | 64 | 192 | 173.4 | 70881 | 1702 | 10+50 |
| s1-pico | 152960 | mlm | f32 | 128 | 128 | 28.4 | 576123 | 626 | 10+50 |
| s1-pico | 202305 | decision | f32 | 64 | 192 | 28.6 | 429542 | 603 | 10+50 |
| s1-nano | 937216 | mlm | f32 | 128 | 128 | 113.5 | 144362 | 1880 | 10+50 |
| s1-nano | 1134209 | decision | f32 | 64 | 192 | 97.9 | 125460 | 1805 | 10+50 |
| s1-micro | 5053952 | mlm | f32 | 128 | 128 | 287.5 | 56984 | 3507 | 10+50 |
| s1-micro | 6630913 | decision | f32 | 64 | 192 | 286.7 | 42857 | 3320 | 10+50 |

## skycmd-s2 training throughput

Measured by `sky-train s2-bench` (mlx-rs 0.32, Metal) on random tokens with ~1/6 `[NUM]` tokens. Warmup steps are excluded; timing covers forward + backward + optimizer update (Muon on the encoder hidden matrices, AdamW elsewhere). `num-mlm` is the numeric MLM over the state stack (CE + 0.5 MSE). `decision` runs the static encoder once per distinct question, the state encoder per row, and the fusion head + scorer for every exit (deep supervision); `decision+qat` adds ternary fake-quant. `tok/s` counts real (unpadded) input tokens: static tokens of the distinct questions plus state tokens. `q/s` counts training questions per second. The GPU may be shared with other jobs while measuring.

| model | params | phase | precision | graph | batch | shape | step (ms) | tok/s | q/s | peak mem (MB) | steps |
|---|---:|---|---|---|---:|---|---:|---:|---:|---:|---|
| s2-pico | 195265 | num-mlm | f32 | compiled | 128 | seq 128 | 48.0 | 341026 | - | 757 | 10+50 |
| s2-pico | 310337 | decision | f32 | compiled | 64 | 64q x static 64 + state 32, 4 opts | 26.1 | 235154 | 2450 | 235 | 10+50 |
| s2-pico | 310337 | decision+qat | f32 | compiled | 64 | 64q x static 64 + state 32, 4 opts | 27.3 | 225140 | 2345 | 234 | 10+50 |
| s2-pico | 195265 | num-mlm | bf16 | compiled | 128 | seq 128 | 43.4 | 377824 | - | 426 | 10+50 |
| s2-pico | 310337 | decision | bf16 | compiled | 64 | 64q x static 64 + state 32, 4 opts | 23.9 | 256784 | 2675 | 121 | 10+50 |
| s2-pico | 310337 | decision+qat | bf16 | compiled | 64 | 64q x static 64 + state 32, 4 opts | 24.6 | 250259 | 2607 | 119 | 10+50 |
| s2-nano | 939649 | num-mlm | f32 | compiled | 128 | seq 128 | 116.6 | 140514 | - | 1953 | 10+50 |
| s2-nano | 1532033 | decision | f32 | compiled | 64 | 64q x static 64 + state 32, 4 opts | 59.9 | 102503 | 1068 | 652 | 10+50 |
| s2-nano | 1532033 | decision+qat | f32 | compiled | 64 | 64q x static 64 + state 32, 4 opts | 40.3 | 152580 | 1589 | 643 | 10+50 |
| s2-nano | 939649 | num-mlm | bf16 | compiled | 128 | seq 128 | 99.0 | 165449 | - | 1046 | 10+50 |
| s2-nano | 1532033 | decision | bf16 | compiled | 64 | 64q x static 64 + state 32, 4 opts | 64.4 | 95438 | 994 | 336 | 10+50 |
| s2-nano | 1532033 | decision+qat | bf16 | compiled | 64 | 64q x static 64 + state 32, 4 opts | 59.7 | 102895 | 1072 | 329 | 10+50 |
| s2-micro | 5058817 | num-mlm | f32 | compiled | 128 | seq 128 | 333.4 | 49137 | - | 3729 | 10+50 |
| s2-micro | 8079105 | decision | f32 | compiled | 64 | 64q x static 64 + state 32, 4 opts | 102.5 | 59933 | 624 | 1688 | 10+50 |
| s2-micro | 8079105 | decision+qat | f32 | compiled | 64 | 64q x static 64 + state 32, 4 opts | 96.0 | 63980 | 666 | 1716 | 10+50 |
| s2-micro | 5058817 | num-mlm | bf16 | compiled | 128 | seq 128 | 196.0 | 83607 | - | 1997 | 10+50 |
| s2-micro | 8079105 | decision | bf16 | compiled | 64 | 64q x static 64 + state 32, 4 opts | 52.9 | 116201 | 1210 | 894 | 10+50 |
| s2-micro | 8079105 | decision+qat | bf16 | compiled | 64 | 64q x static 64 + state 32, 4 opts | 83.5 | 73600 | 767 | 903 | 10+50 |

### s2 training runs (`scripts/train-s2.sh pico nano`)

Wall-clock budgets on an M5 whose GPU was shared with an s1 training job the whole time, so the step rates are roughly half of what the benchmark above gives. Throughput is the median over the run logs (`tok/s` = real tokens, `q/s` = training questions). Accuracy is on the `test` split of `data/decisions` + `data/games` (25,319 questions: the three games, typed_decisions_bench, notam, when2call) with the calibrated temperatures; the f32 GGUF is the snapshot taken when QAT starts, the ternary GGUF is the final QAT model.

| model | pretrain | MLM tok/s | final token CE | decide | decide q/s | f32 GGUF acc | ternary GGUF acc | dodge / flappy / rescue (ternary) | mean layers | early-exit rate | GGUF size f32 / ternary |
|---|---|---:|---:|---|---:|---:|---:|---|---:|---:|---|
| s2-pico | 8 min, 14,161 steps | 510k | 2.96 | 15 min, 4,598 x 128 | 728 | 0.498 | 0.455 | 0.826 / 0.751 / 0.577 | 1.83 / 3 | 0.59 | 1.2 MB / 177 KB |
| s2-nano | 12 min, 9,858 steps | 227k | 2.84 | 20 min, 5,521 x 128 | 613 | 0.494 | 0.512 | 0.944 / 0.842 / 0.888 | 2.86 / 4 | 0.42 | 6.1 MB / 685 KB |

Notes:
- Ternarization without QAT costs 6-11 accuracy points on val (pico 0.470 → 0.406, nano 0.542 → 0.432); 30% of the budget in QAT recovers it (pico 0.482, nano 0.556 on val). The QAT master weights evaluated in f32 are poor (0.31-0.32 on test), hence the f32 export from the pre-QAT snapshot.
- s2-pico's single 64-wide head in a single fusion block is the bottleneck: its first exit is as good as its last, and rescue (7 options, matching option texts against sensor fields) stays at 0.58. s2-nano (2 heads, 2 fusion blocks) matches or beats s1-pico on the games (s1-pico: 0.944 / 0.825 / 0.875) with exits that differ: exit 0 is weak on rescue, exits 1 and 2 agree, and the calibrated τ sends 78-92% of game questions out early.
- Numeric MLM: with the SPEC weight 0.5 the number MSE (target variance ~10 on web text, |u| up to 74) swamps the token CE, which stays on the unigram plateau (4.81 after 1.5 min vs 3.07 without the MSE and 3.19 with weight 0.05). The runs use `--num-weight 0.05` and the target u is clipped to ±12.
- Permutation invariance (CPU device): max |Δp| ≤ 3e-6 on every model and exit. On the Metal GPU the batch-dependent kernel rounding alone reaches ~1e-3, so the check is run on the CPU.

## Inference

Latency of `sky-infer`, natively (`sky-bench`) and in WASM (`node crates/sky-infer/bench/bench-node.mjs`, the `SkyEngine` module built with `wasm-pack --target nodejs`; Node 24.10 / V8). Apple M5, rustc 1.97.1, on a machine shared with training jobs: each number is the best of 3 interleaved rounds of the median over 3000 (native) / 3×2000 (WASM) frames, median / p95 in µs per frame (one question). "before" is the engine of the last commit (`a6ab1fd`), rebuilt the same way.

Frames: Flappy Drone state `vy -2 dx 35 up 20 dn -5 nxt -18 alt 32 t N` (a new state every frame; 9 tokens with value-fused numbers, 16 without) with the action question (2 options); Doom: `Doom::state_text` (14 `key value` pairs: 26 fused tokens for s3-pico, 38 for s2-nano-doom) with the 8-action question. The static stream (question + options) is cached unless stated. Models: `site/public/models/s1-pico.gguf`, `models/ship/s2-nano.gguf` (ternary), `site/public/models/s2-nano-doom.gguf` (ternary), the trained `models/s3-pico/skycmd-s3-pico-{ternary,f32}.gguf` (1 block × 4 loops, d 96, one 96-wide head, fused numbers), and a random-init s3-pico (`sky-convert --random s3-pico --tokenizer data/tok/tok-s2-1024.json`) at τ = 1 (4 block applications + last exit, worst case) and τ = 0 (first exit). Ternary files compute in f32 (`TernaryCompute::Auto`) unless stated.

| model | native before | native | WASM before (SIMD128) | WASM SIMD128 | WASM relaxed SIMD | mean depth |
|---|---:|---:|---:|---:|---:|---:|
| s1-pico | 189 / 213 | 176 / 198 | 323 / 345 | 301 / 330 | 211 / 247 | – |
| s2-nano ternary | 417 / 452 | 272 / 284 | 634 / 642 | 459 / 477 | 378 / 388 | 4 / 4 |
| s2-nano ternary, no static cache | 696 / 725 | 536 / 557 | 1162 / 1205 | 996 / 1034 | 790 / 821 | 4 / 4 |
| **s3-pico ternary (trained)** | – | **25 / 26** | – | **44 / 45** | **37 / 39** | 1 / 4 |
| s3-pico ternary (trained), `TernaryCompute::Int8` | – | 21 / 21 | – | – | – | 1 / 4 |
| s3-pico ternary (trained), τ = 1 | – | 80 / 83 | – | 142 / 148 | 112 / 118 | 4 / 4 |
| s3-pico ternary (trained), τ = 1, `Int8` | – | 63 / 64 | – | – | – | 4 / 4 |
| s3-pico ternary (trained), no static cache | – | 126 / 130 | – | 253 / 258 | 199 / 208 | 1 / 4 |
| s3-pico f32 (trained) | – | 25 / 26 | – | 44 / 45 | 37 / 41 | 1 / 4 |
| s3-pico random f32, τ = 1 | – | 80 / 83 | – | 143 / 148 | 113 / 117 | 4 / 4 |
| s3-pico random f32, τ = 0 | – | 25 / 28 | – | 44 / 45 | 38 / 39 | 1 / 4 |
| Doom: s2-nano-doom ternary | – | 740 / 752 | 1601 / 1623 | 1282 / 1303 | 1002 / 1043 | 4 / 4 |
| Doom: s3-pico ternary (trained) | – | 77 / 80 | – | 138 / 149 | 109 / 118 | 1.11 / 4 |

The old engine cannot load s3 files; the old `sky-bench` has no Doom preset. "WASM relaxed SIMD" is the same module built with `-C target-feature=+simd128,+relaxed-simd` (`f32x4.relaxed_madd`, fused on arm64): Chrome, Edge and Node run it, Safari does not, so a site would load it behind feature detection. `SkyEngine` has no option for the int8 path, hence no WASM row for it.

What made it faster (s3-pico frame at the first exit: ~0.9 M multiply-adds, ~75% of the time in matrix products):
- **Panel kernel for few rows** (`kernels::PanelMatrix`): f32 weights are repacked k-major in panels of 8 outputs, and an outer-product micro-kernel keeps 16-24 independent accumulators whatever the row count (balanced tiles of 3-5 rows x 2 panels, 2-3 rows x 3-4 panels; inputs loaded 4 at a time and broadcast by lane, `fmla v, v, v.s[l]` on NEON, a lane shuffle that V8 turns into `fmul` by element on wasm32; `relaxed_madd` behind `cfg(target_feature = "relaxed-simd")`). On 6-11 rows the 4 x 4 dot-product tiles fell to 30-43 GMAC/s (the remainder tiles are latency-bound); the panels hold 55-62 GMAC/s natively, about 80% of the M5's FMA peak, for every row count.
- **Folded weights**: the first state-block projection of a frame reads `T = tok_embd · W_qkvᵀ` (vocab x 3d, 1.2 MB for s3-pico) and a 17-input correction for value tokens instead of a d-input product (exact up to rounding; not used on carried frames); the fusion keys and values of the state tokens are folded into the 2 option queries (`q·k_j = (W_kᵀq)·h_j + q·b_k`, `Σp_j v_j = W_v Σp_j h_j + b_v`), `2·n·d²` instead of `2·S·d²` multiply-adds per block; the first fusion block's queries and their attention over the cached question tokens are precomputed with the static stream and merged with the state keys by an online softmax.
- **No allocation per frame**: the state encoding reuses the buffers of the evicted cache entry, attention over ≤ 32 tokens packs Q·Kᵀ and P·V into panel products, LayerNorm moments use SIMD partial sums (the scalar `f32` sum chain was 10% of a frame), 8-float copies avoid `memcpy` calls on wasm32, value features use one `sin_cos` and double-angle steps instead of 16 trigonometric calls.
- Net effect on the same files: s2-nano 1.5x faster natively and 1.4x (SIMD128) / 1.7x (relaxed) in WASM, Doom s2-nano 1.25x / 1.6x in WASM; s3-pico answers a Flappy frame in 25 µs natively and 44 µs (SIMD128) / 37 µs (relaxed) in WASM, and a Doom frame in 77 / 138 / 109 µs.
- **Target (≤ 30 µs per decision in WASM): not reached.** At the first exit a frame is still ~0.9 M multiply-adds; SIMD128 has no fused multiply-add (`fmul` + `fadd`: ~36 GMAC/s at most on this core, 31 reached), so ~30 µs is the floor of the products alone. Relaxed SIMD reaches 37 µs per decision. Natively the target is met (25 µs, 21 µs with `Int8`).

Ternary products on the s3 ladder shapes (µs per product; 9 rows = a 9-token state, 2 rows = the option markers). Native: `sky-bench --kernels`; WASM: `node crates/sky-infer/bench/kernels-node.mjs` (module built with `--features wasm-bench`):

| shape (n x k), rows | native f32 panels | native int8 | native LUT 1 pass | native LUT 2 passes | native packed | WASM f32 | WASM LUT 1 / 2 passes | WASM packed | WASM relaxed f32 | WASM relaxed int8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 288 x 96, 9 | 3.99 | 2.77 | 4.32 | 8.40 | 12.23 | 7.73 | 9.83 / 16.24 | 13.91 | 5.82 | 5.14 |
| 384 x 96, 9 | 5.37 | 3.50 | 5.35 | 10.44 | 17.47 | 10.32 | 12.12 / 20.47 | 18.50 | 7.71 | 6.43 |
| 96 x 192, 9 | 2.64 | 2.04 | 4.45 | 8.44 | 8.04 | 5.07 | 10.52 / 15.18 | 8.52 | 3.49 | 4.58 |
| 192 x 96, 2 | 0.65 | 0.42 | 0.75 | 1.45 | 2.67 | 1.15 | 1.89 / 2.92 | 3.20 | 0.89 | 0.89 |
| 384 x 128, 9 | 7.05 | 4.70 | 7.00 | 13.68 | 21.36 | 14.06 | 15.34 / 26.06 | 23.74 | 9.71 | 7.95 |
| 512 x 128, 9 | 9.59 | 6.11 | 8.78 | 17.25 | 28.48 | 18.05 | 18.99 / 32.95 | 31.74 | 13.05 | 10.32 |
| 128 x 256, 9 | 5.02 | 3.30 | 6.67 | 12.83 | 14.09 | 8.83 | 14.51 / 23.19 | 14.80 | 6.49 | 6.95 |
| 256 x 128, 2 | 1.16 | 0.71 | 1.18 | 2.29 | 4.64 | 2.06 | 2.76 / 4.57 | 5.45 | 1.54 | 1.51 |

Reading:
- **Lookup tables (SPEC-S3 §8)** (`lut::LutMatrix`, `TernaryCompute::Lut`: 2-bit weights as `p`/`n` masks per group of 4 inputs, int8 activations with one scale per row, 16-entry subset-sum tables kept as two byte planes, `vqtbl1q_u8` / `i8x16.swizzle` lookups, int16 accumulation). With one activation pass the probabilities of the golden models move by up to 4e-2, so `Lut` runs two passes (int8 + residual, ≤ 3.5e-4 on the s2/s3/s3t goldens). It beats the packed add/sub kernel but not the f32 panels: with one pass it takes 0.9-1.7x the f32 time natively (a tie on the 128-input matrices, slower elsewhere) and 1.05-2.1x in WASM; with the two passes needed for accuracy, 2-3x. These matrices are tiny and stay in L1, so the products are compute-bound, not memory-bound as in the LLM setting of T-MAC: 4 lookups plus 8 widening adds per 16 outputs x 4 inputs cost as much as 16 FMAs. `TernaryCompute::Auto` therefore keeps the f32 path on every platform; the file stays 2-bit.
- **int8 dot products** (`TernaryCompute::Int8`, opt-in: `sdot` by lane on aarch64, `i32x4.relaxed_dot_i8x16_i7x16_add` in relaxed WASM, two activation passes `x ≈ s (q + e / 254)`) are the fastest path natively (1.3-1.6x per product, 21 vs 25 µs per decision) and in relaxed WASM on k ≤ 128, at ≤ 5.5e-4 deviation on the goldens. Without relaxed SIMD only a scalar fallback exists (10x slower).
- The s3 options keep the s2 advantages: value-fused numbers nearly halve the state (9 vs 16 tokens for the Flappy state, 26 vs 38 for Doom), the shared static stream and early exit do the rest. A trained s3-pico answers Flappy frames at the first exit, so it costs one block application per frame; a Doom frame (26 tokens) costs 3x more.
- Accuracy: the engine matches the mlx-rs graphs of `sky-train s2-golden` within 6.3e-6 (f32 and ternary files, f32 and packed compute) on `s2.json`, `s3.json` (fused numbers, 1 x 4 loops, head_dim 96) and `s3t.json` (temporal carry across frames).

## Doom-like game (`scripts/train-doom.sh`)

`sky-arena --exit-threshold 0.95`, 50 episodes per game, seed 7, Apple M5 shared with other training jobs (latencies are upper bounds). Doom: completion of easy levels, kill rate (kills / monsters), deaths, mean ticks at 35 Hz. The model is s2-nano-g2 fine-tuned fully ternary for 2 rounds (oracle data, then one DAgger round over the four games). Chapter 18 of the tutorial has the details.

| game | metric | oracle | s2-nano-doom | s2-nano-g2 (base) | s2-nano |
|---|---|---:|---:|---:|---:|
| Doom | completion / kill rate / deaths | 1.00 / 0.75 / 0 | 1.00 / 0.75 / 0 | 0.00 / 0.00 / 0 | not trained on doom |
| Doom (100 episodes, seed 2026) | completion / kill rate / deaths | 1.00 / 0.69 / 0 | 1.00 / 0.69 / 0 | | |
| Flappy Drone | mean pipes (crashes), cap 200 | 200 (0) | 197.6 (2) | 196.2 (1) | 30.6 (50) |
| Rescue Run | completion | 1.00 | 1.00 | 1.00 | 1.00 |
| Obstacle Dodge | completion | 0.78 | 0.72 | 0.74 | 0.58 |

The Doom model is 1,532,033 parameters and 702 KB as a ternary GGUF. A doom state is about 38 tokens and closed-loop play stays at full depth (3.99 of 4 layers on average).

| doom decision | native (`sky-arena`, action question) | WASM (`bench-node.mjs --game doom`, 3 questions + physics) |
|---|---:|---:|
| s2-nano-doom, τ = 0.95 | 1,055 µs p50 / 1,406 µs p95 | 1,830 µs median / 2,030 µs p95 |


## skycmd-s3 (s3-pico): value-fused, looped, ternary

s3 = s2 + `fuse_numbers` (a `key value` pair is one token) + `state_loops` (one weight-shared state block applied 4 times, an exit after each loop) + `head_count` 1 (head_dim 96), see `docs/SPEC-S3.md`. Training and evaluation on an Apple M5 whose GPU was shared with another training job the whole time (doom), so step rates are lower than an idle machine gives.

| model | params | ternary GGUF | state blocks x loops | exits | static / fusion | d / ff / vocab |
|---|---:|---:|---|---|---|---|
| s3-pico | 445,345 | 251 KB (f32: 1.8 MB) | 1 x 4 | [1, 2, 3, 4] | 1 / 2 | 96 / 192 / 1024 |
| s3-nano (config only, not trained) | 1,204,353 | - | 2 x 3 | [2, 4, 6] | 2 / 2 | 128 / 256 / 2048 |
| s2-nano (ship) | 1,532,033 | 702 KB | 4 x 1 | [1, 2, 4] | 2 / 2 | 128 / 256 / 2048 |
| s1-pico (site) | 202,305 | 545 KB (f16) | - | - | - | 64 / - / 1024 |

### s3 training runs (`scripts/train-s3.sh pico`, `scripts/specialize-s3.sh pico 3 8`)

| phase | budget | steps | median throughput | result |
|---|---|---:|---:|---|
| s2-pretrain (web text + 10% numeric curriculum, fused rows) | 8 min | 4,814 x 128 x 128 | 167k tok/s | token CE 2.86 (val) |
| s2-decide (games 0.6 / numeric 0.1 / decisions 0.3, QAT after 10%, loop dropout 0.25) | 15 min | 6,440 x 128 | 982 q/s | val 0.530 ternary; test GGUF 0.534 (decisions + games + numeric) |
| DAgger r1 (games 0.5 / r1 0.35 / numeric 0.05 / decisions 0.1, QAT 100%) | 8 min | 4,683 | 1,398 q/s | val 0.555 |
| DAgger r2 | 8 min | 5,136 | 1,495 q/s | val 0.561 |
| DAgger r3 | 8 min | 5,313 | 1,516 q/s | val 0.574 |

- Compiling one mlx graph per batch shape dominated the first decide attempt (~250 q/s, stopped after 4 min): loop dropout is now an exit-weight input (one graph for every dropout outcome, same gradients) and `--pad-to 32` buckets the batch shapes; 4-5x faster.
- Numeric curriculum: val accuracy on `data/numeric` stays at 0.41-0.43 (chance is about 0.33 averaged over the six families), so the value channel learned little exact comparison in 15 min.

### DAgger rounds (30 episodes per game, seed 7, full depth τ = 1, ternary GGUF in sky-arena)

| model | flappy pipes (of 200) | crashes | rescue rate | dodge completion (oracle 0.867) | dodge vetoes |
|---|---:|---:|---:|---:|---:|
| s3-pico base | 145.3 | 14 | 0.08 | 0.567 | 9,137 |
| round 1 | 5.4 | 30 | 1.00 | 0.667 | 4,205 |
| round 2 | 200.0 | 0 | 1.00 | 0.833 | 1,030 |
| round 3 | **200.0** | **0** | **1.00** | **0.867** | **14** |

Round 3 meets the gate (200/200 pipes, rescue >= 95%, dodge completion >= 95% of the oracle with < 100 vetoes, 0 crashes).

### Closed-loop exit calibration (`scripts/calibrate-exit.sh`, 20 episodes per game, same seed)

| τ | flappy | rescue | dodge | mean layers (flappy / rescue / dodge) | accepted |
|---|---|---|---|---|---|
| 1.0 | 200, 0 crashes | 1.00 | 0.90, 4 vetoes | 4.00 / 4.00 / 4.00 | reference |
| 0.5 | 200, 0 crashes | 1.00 | 0.90, 5 vetoes | 1.00 / 1.18 / 1.02 | no (one extra veto) |
| **0.6** | 200, 0 crashes | 1.00 | 0.90, 4 vetoes | **1.44 / 1.48 / 1.05** | **yes: shipped** |

`models/ship/s3-pico.gguf` = round 3 with `exit_threshold.{choice,score,noul} = 0.6`. Re-checked on 30 episodes at its own τ: 200/200 pipes, 0 crashes, rescue 1.00, dodge 0.867 with 14 vetoes (same as full depth), mean layers 1.44 / 1.52 / 1.05.

### s3-pico vs s2-nano vs s1-pico

Held-out accuracy: `test` split of `data/decisions` + `data/games` (30,050 questions; `s2-eval --gguf` for s2/s3 with the calibrated temperatures, `sky-train eval` on `runs/s1-pico` for s1). s3 was trained with the doom records that are now part of `data/games`; s2-nano and s1-pico were not, so the column without doom is the fair one.

| model | test acc | acc without doom (25,319) | dodge / flappy / rescue (test) | doom (test) |
|---|---:|---:|---|---:|
| s3-pico (ship, τ 0.6) | 0.594 | 0.543 | 0.958 / 0.836 / 0.889 | 0.870 |
| s2-nano (ship, τ 0.95) | 0.496 | 0.544 | 0.959 / 0.840 / 0.900 | 0.237 |
| s1-pico (runs/s1-pico) | 0.428 | 0.479 | 0.944 / 0.743 / 0.875 | 0.152 |

Closed loop, `sky-arena`, 30 episodes per game at each model's own τ (machine shared with a GPU training job during these runs, so p50 latencies are indicative):

| model | flappy | rescue | dodge completion / vetoes | mean layers (flappy / rescue / dodge) | p50 µs per decision (flappy / rescue / dodge) |
|---|---|---:|---|---|---|
| s3-pico | 200, 0 crashes | 1.00 | 0.867 / 14 | 1.44 / 1.52 / 1.05 of 4 | 24 / 60 / 15 |
| s2-nano | 200, 0 crashes | 1.00 | 0.867 / 25 | 3.43 / 3.16 / 3.28 of 4 | 249 / 528 / 163 |
| s1-pico | 200, 0 crashes | 1.00 | 0.833 / 1,026 | - | 211 / 344 / 192 |

Native `sky-bench <model> --questions 1|2 --iters 4000` (Flappy state, static cache on; median µs per frame):

| model | 1 question | 2 questions | layers on this state |
|---|---:|---:|---|
| s3-pico ship (τ 0.6) | 27.5 | 35.0 | 1 of 4 (exit 0) |
| s3-pico ship, τ = 1 | 95.0 | 103.1 | 4 of 4 |
| s2-nano ship (τ 0.95) | 273.0 | 315.3 | 4 of 4 |
| s2-nano ship, τ = 1 | 240.7 | 255.0 | 4 of 4 |
| s1-pico site | 137.6 | 300.6 | - |

Reading: s3-pico plays the three games as well as s2-nano (and better than s1-pico on dodge vetoes), ties it on held-out accuracy without doom, with 3.4x fewer parameters, a 2.8x smaller file, and about 10x less time per decision in closed loop: the fused state is ~half as many tokens and the calibrated τ answers at the first loop on most frames.

### Temporal recursion (SPEC-S3 §7), experimental, not shipped

`configs/s3-pico-temporal.json` (+ `carry_gate`, `carry_norm`: 445,537 params), 10 min fine-tune of round 3 on the round-3 data plus 50% sequence steps (`s2-seqgen`: 15,000 sequences x 4 frames per game, carry dropped with p = 0.3, every exit of every frame supervised). Cold val accuracy unchanged (0.573 vs 0.574). Closed loop with the mlx-rs player (`sky-train s2-play`, which reproduces sky-arena's metrics exactly on the same episodes), 5 episodes, flappy capped at 30 pipes:

| τ | variant | flappy layers | rescue layers (cold / warm frames) | dodge layers | game metrics |
|---|---|---:|---|---:|---|
| 0.6 | temporal, carry on (warm frames) | 1.61 | 1.42 / 1.76 | 1.06 | 30/30, 1.00, 0.80 (17 vetoes) |
| 0.6 | temporal, carry off | 1.45 | 1.56 | 1.09 | 30/30, 1.00, 0.80 (12 vetoes) |
| 0.6 | round 3 (no temporal training) | 1.43 | 1.60 | 1.05 | 30/30, 1.00, 0.80 (2 vetoes) |
| 0.9 | temporal, carry on | 2.21 | 2.37 / 2.46 | 1.52 | 20/20, 1.00, 0.67 |
| 0.9 | temporal, carry off | 2.10 | 2.49 | 1.60 | 20/20, 1.00, 0.67 |

The carry does not reduce depth: warm frames use as many layers as cold ones, within noise (exits stay bimodal, exit 0 or the last loop), and vetoes are slightly higher. Not shipped; `skycmd-s2.temporal_carry` stays off by default.

## skycmd-s4 (s4-micro): generic decisions distilled from Qwen3.5, experimental

`configs/s4-micro.json`: s3 at language scale, 5,587,201 params, ternary GGUF 2.4 MB (f32 22 MB), d / ff / vocab 256 / 768 / 4096, 2 state blocks x 3 loops, exits [2, 4, 6], 4 heads, state / head budget 384 / 128 tokens. One night on an Apple M5 (16 GB), `scripts/night-s4.sh`, chapter 22.

| phase | budget | steps | throughput | result |
|---|---|---:|---:|---|
| teacher Qwen3.5-4B 4-bit (mlx-lm, letter logits, one prefill) | ~3 h | 74,721 records | 3.8 ex/s on the bench, 11-13 ex/s on short records | 44,721 cases + 30,000 generic snippets |
| teacher Qwen3.5-0.8B 8-bit (bench only) | - | 2,000 | 18.5 ex/s | bench 0.412 |
| s2-pretrain (MLM, `--seq 256 --batch 64`) | 30 min | 6,236 | 56k tok/s | token CE 2.754 (val) |
| s2-decide (gold 0.45 / mix 0.5 / numeric 0.05, QAT after 10%, loop dropout 0.25, DIS 0.5, length pool 16) | 150 min | 26,180 x 64 | ~210 q/s | val 0.510 ternary (last exit); 8 non-finite batches skipped |

Test split, `sky-eval` (the shipping engine, each GGUF with its own tokenizer, early exit at the calibrated τ):

| model | bench | Banking77 zs | MASSIVE en zs | MASSIVE fr zs | When2Call | all | ECE | mean passes | median native µs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| s3-pico | 0.382 | 0.247 | 0.187 | 0.117 | 0.367 | 0.290 | 0.062 | 3.70 / 4 | 554 |
| s4-micro f32 (QAT master) | 0.348 | 0.131 | 0.192 | 0.186 | 0.365 | 0.268 | 0.072 | 5.28 / 6 | 4,838 |
| s4-micro ternary | 0.343 | 0.303 | 0.335 | 0.240 | 0.387 | 0.340 | 0.066 | 4.92 / 6 | 4,533 |
| Qwen3.5-4B 4-bit (teacher) | 0.501 | 0.898 | 0.893 | 0.873 | - | - | - | - | - |
| oscar-1-17m (published) | 0.6775 | - | - | - | - | - | - | - | - |
| MiniLM-L6 cosine, no training (22.7M) | - | 0.855 | 0.773 | 0.493 | - | - | - | - | 6-9 ms CPU |
| BERT-Tiny fine-tuned 70 min CPU (4.4M) | 0.396 | 0.794 | 0.570 | 0.302 | 0.468 | 0.519 | 0.114 | - | 1.6 ms CPU batched |

Gate (bench > 0.55): **failed**, published as experimental on `/live/`. Validation of the cases (agreement with the target): tools 0.771, triage 0.890, moderation 0.976 (= the base rate of "does not violate"). WASM vs native on 40 requests: max |Δp| 6.3e-7, 40/40 argmax, median 6.9 ms per uncached decision under Node; natively a short cached input takes 0.8 ms (int8 kernel), a long bench state about 18 ms.

## skycmd-s5 (s5-lite): iso decision model on Ettin-17M, pre-registered

Pre-registration: [`docs/s5-prereg.md`](docs/s5-prereg.md) and its three amendments. Verdicts and intervals: the [results page](https://maxgfr.github.io/skycmd/results/), read from `site/public/results/s5.json` (`sky-s5-report summary`).

**Training throughput.** `sky-train s5-train`, Apple M5, bf16, batch 32, mlx-rs 0.32, MLX allocator cache capped at 1 GB:

- one B-i run alone: about 42k tok/s;
- three runs at once: about 9–10k tok/s each, which is not faster in total;
- 2000 steps take 19 min for B-i and 27–31 min for A′, three at a time.

**Evaluation.** `sky-eval` on CPU, f32 GGUF calibrated on the val split:

| run | variant | lr | λ | steps | val acc | val NLL | bench acc | bench ECE | bench NLL |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| f-a-s0 | A | 2e-4 | 0 | 2000 | 0.436 | 1.136 | 0.5285 | 0.080 | 1.062 |
| f-ap-s0 (shipped by the rule) | A′ | 2e-4 | 0 | 2000 | 0.426 | 1.097 | 0.5410 | 0.051 | 1.047 |
| f-ap-s1 | A′ | 2e-4 | 0 | 2000 | 0.373 | 1.155 | 0.3670 | 0.058 | 1.210 |
| f-ap-s2 | A′ | 2e-4 | 0 | 2000 | 0.374 | 1.156 | 0.3795 | 0.062 | 1.209 |
| s-ap-1e-4-s0 | A′ | 1e-4 | 0 | 2000 | 0.455 | 1.079 | 0.5740 | 0.057 | 1.034 |
| s-ap-1e-4-s1 | A′ | 1e-4 | 0 | 2000 | 0.464 | 1.064 | 0.6045 | 0.071 | 1.004 |
| s-ap-1e-4-s2 | A′ | 1e-4 | 0 | 2000 | 0.478 | 1.052 | 0.6065 | 0.075 | 0.999 |
| f-b-s0 | B-i | 2e-4 | 0 | 2000 | 0.529 | 0.998 | 0.6665 | 0.078 | 0.964 |
| f-b-s1 | B-i | 2e-4 | 0 | 2000 | 0.521 | 0.995 | 0.6780 | 0.096 | 0.960 |
| f-b-s2 | B-i | 2e-4 | 0 | 2000 | 0.529 | 0.995 | 0.6775 | 0.090 | 0.957 |
| l-ap-0.1-s0 | A′ | 2e-4 | 0.1 | 2000 | 0.428 | 1.093 | 0.5285 | 0.056 | 1.068 |
| l-ap-0.1-s1 | A′ | 2e-4 | 0.1 | 2000 | 0.437 | 1.094 | 0.5345 | 0.039 | 1.054 |
| l-ap-0.1-s2 | A′ | 2e-4 | 0.1 | 2000 | 0.436 | 1.097 | 0.5500 | 0.053 | 1.030 |
| long-b-s0 (/live, outside the bets) | B-i | 2e-4 | 0 | 8000 | 0.581 | 0.902 | 0.6950 | 0.088 | 0.945 |

**Notes:**

- **The bench** is `typed_decisions_test`: 2000 questions, our rendering of the typed-decisions test split. On the same records, oscar-1-17m scores 0.682 (published: 0.6775).
- **Inits.** Every A′ and B-i run starts from `ettin-reranker-17m`, the init chosen on validation.
- **Quantization.** q8_0 changes the bench ECE of the B-i finalists by −0.0024, −0.0003 and −0.0005, and their accuracy by at most 0.3 points.
- **In the browser.** The /live model answers a 6-option question in about 20 ms in headless Chrome (wasm, SIMD128), and its state is reused across questions.
