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

The old engine cannot load s3 files; the old `sky-bench` has no Doom preset. "WASM relaxed SIMD" is the same module built with `-C target-feature=+simd128,+relaxed-simd` (`f32x4.relaxed_madd`, fused on arm64). Since 14/11/2026 `scripts/build-wasm.sh` builds the site module both ways (`sky_web_bg.wasm` and `sky_web_relaxed_bg.wasm`, one shared `sky_web.js`), and `loadWasm()` loads the relaxed one when `WebAssembly.validate` accepts a relaxed instruction, falling back to SIMD128 if it does not or if the relaxed file fails to load. The WebAssembly feature table ([webassembly.org/features](https://webassembly.org/features/), as updated on 10/11/2026) lists relaxed SIMD in Chrome from 114, Firefox from 145 and Node from 21, and in Safari only behind a JavaScriptCore flag; other browsers get the SIMD128 module. `SkyEngine` has no option for the int8 path, hence no WASM row for it.

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

## skycmd-s6 (s6-pico): Parallax attention and Muon v2, code only

`docs/SPEC-S6.md`. These costs were measured before any s6 training, on random data, Apple M5, two runs each (the range shown). The pre-registered pico ablation then adopted none of these additions (`runs/recipe/recipe.json`); the trained s6 models are in the sections below.

| | s3-pico | s6-pico |
|---|---:|---:|
| params (decision / pretraining stack) | 445,345 / 203,393 | 463,969 / 212,705 |
| `sky-train s2-bench`: numeric MLM step, batch 128 × 128, f32 | 55.7–55.9 ms | 62.3–63.7 ms |
| `sky-train s2-bench`: decision step, batch 64, f32 | 17.5–17.8 ms | 17.9–18.1 ms |
| `sky-bench --random`: ternary, τ = 1, per decision | 68.2 µs | 74.1 µs |

The Muon v2 Newton–Schulz polish (`s2-bench --config configs/s3-pico.json --ns-polish 0` vs `2`) changes nothing measurable on the MLM step (57.8–58.5 ms both ways). On the decision step it goes from 17.3 ms to 17.5–18.8 ms, within the machine's noise.

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

## skycmd-s6 (s6-nano, s6-micro): general-purpose decisions and games, pre-registered

Pre-registration: [`docs/s6-gen-prereg.md`](docs/s6-gen-prereg.md), its gates in §6 and its amendments 1 to 17. Results: `runs/s6-nano/eval/results.md` and `runs/s6-micro/eval/results.md` (`scripts/s6-gate.py`), with the JSON files next to them; for the second version of s6-micro, [`runs/s6-micro2/runs/s6-micro/eval/results.md`](runs/s6-micro2/runs/s6-micro/eval/results.md) and the files next to it, committed.

**Two versions of s6-micro.** The first (v1, published as `s6-micro-v1`) failed its ECE guard. The second (v2, published as `s6-micro`) was pre-registered after that reading (amendment 12: 360 minutes of decision training instead of 180, everything else the same) and passed the same gate on the same test records (amendment 17). Its pass is therefore a second look at these test sets, not a first one. Below, "s6-micro" without a version is v1 where the text predates v2.

**Recipe.** Chosen by the pre-registered rules of the pico ablation (`runs/recipe/recipe.json`): AdamW, learning-rate scale 2.0, and none of the additions (no Muon, Parallax, XSA, SODA, exit distillation, EMA, StableQAT or curriculum; each missed its pre-registered gain or lost a game). The rate pilots kept ×1 for s6-nano and ×0.5 for s6-micro. Without the Parallax probe rows, s6-nano has 1,486,225 parameters instead of the 1,569,457 of the pre-registration; s6-micro has the s4-micro shape, 5,587,201. The GLM-5.3-flash teacher labelled 747,077 of the 747,198 selected training questions (0.9998, `runs/all/nano-glm.log`).

**Runs.** Throttled mode (a duty cycle and cooling pauses, amendments 4 to 10), so the budgets count working minutes only. Steps actually done:

| phase | s6-nano budget | s6-nano steps | s6-micro v1 budget | s6-micro v1 steps | s6-micro v2 budget | s6-micro v2 steps |
|---|---:|---:|---:|---:|---:|---:|
| MLM pretraining | 90 min | 53,360 (0.87B tokens, 588 per parameter, epoch 8, MLM loss 2.356) | 180 min | 31,700 (0.52B tokens, 93 per parameter, epoch 5, MLM loss 2.081) | (v1's) | (v1's, not run again) |
| decision training | 90 min | 19,893 | 180 min | 38,826 | 360 min, uncompiled | 65,471 |
| DAgger rounds 1, 2, 3 | 15 min each | 4,645, 4,660, 4,644 | 20 min each | 5,951, 5,898, 5,952 | 20 min each | 3,822, 4,673, 4,969 |

Pretraining steps, tokens, epoch and loss are from the results files; the decision and DAgger steps are the `done:` lines of `runs/all/{nano,micro}-{train,dagger}.log`. Two decision trainings were stopped and resumed from their last checkpoint: s6-nano's by the memory cap at step 16,020, resumed from step 15,573 (amendment 8), and s6-micro's by hand at step 27,280 after about 34 minutes in swap, resumed from step 27,161 with the uncompiled step (amendment 11); the steps done in swap are kept, so s6-micro has fewer decision steps than its 180 minutes would otherwise give.

v2 (`runs/s6-micro2/micro2.log` and the step logs next to it): its decision training died on 15/11 at 21:40 at step 62,520 on a file-access error ("Operation not permitted" while a checkpoint was renamed: the session lost access to the folder) and was resumed on 16/11 at 00:55 from its checkpoint at step 61,051 (327.5 of 360 minutes); the 1,469 steps after the checkpoint were done again and counted once. The resumed minutes, the DAgger rounds, the exit step and the evaluation ran while other jobs used the machine (decision training: a median of 17,550 tokens/s after the restart, against 26,018 over the first launch), which is why v2's DAgger rounds made fewer steps than v1's in the same minutes and why v2's latencies below are not comparable with the other rows. The trainer binary is not v1's (the code of other runs was merged in between, its options off).

**Test sets.** `sky-eval`, the ternary GGUF of the last DAgger round, at most 2,000 questions per source (the first ones), at full depth (τ = 1); the published models on the same records:

| model | params | `typed_decisions_bench` | MASSIVE en | MASSIVE fr | When2Call | NOTAM deel_subject | NOTAM notam_multi | all | ECE (all) | mean block passes | median native µs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **s6-nano** | 1,486,225 | **0.480** | 0.425 | 0.239 | 0.404 | 0.055 | 0.493 | 0.360 | **0.0479** | 4.00 | 1,488 |
| **s6-micro (v2)** | 5,587,201 | **0.553** | 0.531 | 0.314 | 0.401 | 0.092 | 0.353 | 0.380 | **0.0588** | 6.00 | 8,970 (loaded machine) |
| s6-micro-v1 | 5,587,201 | 0.547 | 0.518 | 0.275 | 0.430 | 0.052 | 0.394 | 0.378 | 0.0821 | 6.00 | 7,346 |
| s6-micro-v1 (CRC thresholds) | 5,587,201 | 0.544 | 0.518 | 0.275 | 0.431 | 0.052 | 0.391 | 0.377 | 0.0804 | 4.59 | 4,896 |
| s3-pico | 445,345 | 0.382 | 0.187 | 0.117 | 0.368 | 0.163 | 0.319 | 0.268 | 0.0629 | 4.00 | 518 |
| s4-micro | 5,587,201 | 0.352 | 0.335 | 0.240 | 0.399 | 0.065 | 0.298 | 0.285 | 0.0666 | 6.00 | 7,340 |
| s5-lite | 17,060,097 | 0.696 | 0.726 | 0.444 | 0.376 | 0.177 | 0.554 | 0.500 | 0.0328 | n/a (iso) | 10,090 |

The reference rows are those of the s6-nano evaluation; the s6-micro evaluations (v1 and v2) give the same accuracies (v1's latencies differ by under 1%). Published bench scores: s3-pico 0.382, s4-micro 0.343 (older record layout), oscar-1-17m 0.6775, s5-lite 0.696; majority class 0.398. At its conformal thresholds s6-micro-v1 exits early on 36.0% of the questions. v2 was not evaluated at its conformal thresholds, which the arena check rejected first (below).

- **In distribution, not zero-shot:** `zs_banking77` 0.631 (s6-nano), 0.705 (s6-micro-v1) and 0.763 (s6-micro v2), n 2,000. Banking77 is in the training data.
- **MASSIVE English is not strictly zero-shot.** 206 of the 2,000 English test utterances read here also appear as states in the training split, 201 of them as questions of the same intent task in a decision source derived from MASSIVE (`calibrated:mteb/amazon_massive_intent` in `data/decisions/calibrated_typed.jsonl`), the other 5 in `calibrated:clinc/clinc_oos`; none of the 1,000 French ones does (exact match after lowercasing). The training data hold 3,113 questions of that source (3 options instead of 8). The seen utterances are not answered better: s6-nano 0.393 on the 206 against 0.429 on the 1,794 others, s6-micro 0.495 against 0.521; without them the pooled ECE is 0.0486 and 0.0837, so neither gate changes (`docs/s6-gen-prereg.md`, amendment 14). This overlap was found after the evaluation, when this page was written.
- **ECE by source and the interval of the pooled ECE** (reported numbers, not gates; amendment 12; `scripts/s6-ece-by-source.py` on the evaluation dumps, which gives back the pooled 0.0479 and 0.0821 exactly). s6-nano: bench 0.0261, MASSIVE en 0.0248, MASSIVE fr 0.1112, When2Call 0.1412, NOTAM deel_subject 0.0746, NOTAM notam_multi 0.0710; pooled 0.0479, 95 % bootstrap interval [0.0416, 0.0564] (1,000 resamples of the items). s6-micro-v1: 0.0660, 0.1334, 0.0313, 0.1176, 0.0636, 0.2558; pooled 0.0821, [0.0746, 0.0900]. Details in [`docs/big-run-report.md`](docs/big-run-report.md) §6. s6-micro v2 (`runs/s6-micro2/runs/s6-micro/eval/ece-by-source.json`, which gives back its pooled 0.0588): 0.0498, 0.0715, 0.0300, 0.2165, 0.0185, 0.1409; pooled 0.0588, [0.0526, 0.0687]. On When2Call v2 is more over-confident than v1 (mean confidence 0.612 for an accuracy of 0.401).
- **Constant answers and the training prior** (amendment 16, `scripts/s6-answer-priors.py`; `docs/s6-gen-prereg.md`, amendment 17). On `typed_decisions_bench`, answering each question with its most common training answer scores 0.4815; s6-nano 0.4805, s6-micro-v1 0.547, v2 0.553. One answer covers at least 95% of the items on 17 of the 20 bench tasks for s6-nano, 7 for v1 and 11 for v2. On When2Call the training prior (`cannot_answer`) scores 0.359; v2 answers `request_for_info` to 1,405 of 2,000 items and `tool_call` to 45 (v1: 246; gold `tool_call`: 696).
- **Truncated records** (`sky-train s2-truncation`, the first 500 records of each source, n 258,309): s6-nano state 12.5%, head 10.5%, either 21.7%, and on `typed_decisions_bench` state 1.4%, head 0.0%; s6-micro (both versions: same tokenizer and windows) 14.6%, 11.6%, 24.0%, and on the bench 4.2%, 0.0%.

**Arena.** 30 episodes per game, the same seeds for every model, mean score (`models/s6-*/…-arena.json`):

| model | Flappy | Rescue | Dodge | Doom-like |
|---|---:|---:|---:|---:|
| s3-pico | 200.00 | 3.00 | 9.47 | 3.27 |
| s6-nano, full depth | 200.00 | 3.00 | 9.47 | 3.27 |
| s6-nano, CRC thresholds | 200.00 | 2.80 | 9.33 | 3.27 |
| s6-micro-v1, full depth | 200.00 | 3.00 | 9.47 | 3.27 |
| s6-micro-v1, CRC thresholds | 200.00 | 3.00 | 9.47 | 3.27 |
| s6-micro v2, full depth | 200.00 | 3.00 | 9.47 | 3.27 |
| s6-micro v2, CRC thresholds (rejected) | 200.00 | 3.00 | 9.47 | 3.27 |

At full depth s6-nano plays at the oracle's score in every game: 0 crashes, Rescue 100%, Dodge completion 87% (= oracle) with 23 reflex vetoes, Doom-like completion 100% and kill rate 0.772 (= oracle) with 3 vetoes. At its CRC thresholds (α = 0.02) it uses 1.01 to 1.83 of its 4 passes per game, but Rescue falls to 93% and the Dodge vetoes rise to 1,016: outside the pre-registered 2%, so the thresholds are not kept. s6-micro v2 at full depth: 0 crashes, Rescue 100%, Dodge completion 87% with 20 vetoes, Doom-like completion 100% and kill rate 0.77 (= oracle) with 2 vetoes; at its CRC thresholds (2.0 to 2.7 of its 6 passes per game) the scores are the same and there are no crashes, but Dodge has 21 vetoes, one more than at full depth, which the pre-registered rule does not allow, so they are not kept.

**Gates** (`docs/s6-gen-prereg.md` §6; each also needs no game more than 5% below s3-pico and ECE ≤ 0.08):

- **s6-nano: passed, published at full depth.** Bench 0.480 > 0.398, ECE 0.0479, no game below s3-pico. The CRC thresholds are not kept, so the published file `site/public/models/s6-nano.gguf` is the last DAgger round with its three exit thresholds set to 1 (`gguf-set`; the calibration had left them at 0.5). The bar of 0.398 is the global majority class, and a constant answer per task passes it: answering each question with its most common answer in `typed_decisions_train` scores 0.4815 on the bench.
- **s6-nano: what its gate did not measure** (`docs/s6-nano2-study.md`, 15/11/2026, after the gate; from the evaluation dumps and from validation splits). On 17 of the 20 tasks of `typed_decisions_bench` it gives one answer to at least 95% of the items (13 of them at 100%); its 0.4805 equals the most-common-training-answer prior (0.4815). The bench's tasks were 0.4% of its decision-training draws (about 10,200 of 2,546,304). A hashed character n-gram logistic regression of at most 715,136 bytes reaches 0.703 on the validation split of the 20 tasks (per-task prior 0.525), and 0.947 / 0.944 on MASSIVE English / French validation once trained on MASSIVE's train split (a rehearsal on validation, not a test). French in the decision data: two yes/no tasks (`pawsx_fr`, `allocine`). When2Call: `tool_call` is 13.9% of the training split and 35% of the test, and s6-nano predicts it 39 times in 2,000. NOTAM: no training data. The verdict stands. A narrower second version (a router for tool and intent decisions) was studied and is not scheduled; the study puts its chance of clearing all its gates at about 10 to 20%.
- **s6-micro-v1: failed on calibration only.** Bench 0.547 ≥ 0.45, and the announced goal of 0.50 is met; no game below s3-pico; but ECE 0.0821 at full depth and 0.0804 at the CRC thresholds, above 0.08. It failed its gate; its file is published as a result (owner's decision of 15/11/2026), as `site/public/models/s6-micro-v1.gguf`: the last DAgger round with its three exit thresholds set to 1 (`gguf-set`), the weights of the evaluated file byte for byte (amendments 13 and 15). The verdict does not change: it is recommended for no use.
- **s6-micro v2: passed at τ = 1, published at full depth as `s6-micro`; a second look** (amendment 17). Bench 0.553 ≥ 0.45 (the announced goal of 0.50 is met), ECE 0.0588 ≤ 0.08, no game below s3-pico. It was designed after v1's ECE failure and judged on the same test records, so its pass is not a first look. Its CRC thresholds are rejected (one extra Dodge veto), so the published file `site/public/models/s6-micro.gguf` is the last DAgger round with its three exit thresholds set to 1 (2,420,032 bytes, sha256 `f66d7021…40ff`); the other metadata and every tensor byte are those of the evaluated file. Deviations: one restart from its checkpoint after a technical failure, a loaded machine for the end of the run (latencies not comparable), a trainer binary that is not v1's.

**From nano to micro:** bench accuracy 0.480 → 0.547 (v1) and 0.553 (v2), ECE 0.0479 → 0.0821 (v1) and 0.0588 (v2), median native latency 1,488 → 7,346 µs per decision (v1; v2's 8,970 µs was measured on a loaded machine), file 715,136 → 2,420,032 bytes.

**WASM (s6-nano, the published file).** `node crates/sky-web/bench/parity-node.mjs site/public/models/s6-nano.gguf runs/s6-nano/eval/dump-s6-nano.jsonl --n 200` on the first 200 bench questions, SIMD128 build: max |p_wasm − p_native| 9.5e-7, argmax agreement 200 of 200, median 14.6 to 15.2 ms per decision under Node in two runs (long JSON states; natively 7,385 µs on the same source). On short game states, `bench-node.mjs --game flappy` gives 436 µs per Flappy frame (two questions and the physics step, full depth, SIMD128 build).

Both builds, three alternating runs each (`--relaxed` for the relaxed module), on 14/11 on the Apple M5 while a GPU training was running, Node 24:

| s6-nano | SIMD128: median (range) | relaxed SIMD: median (range) |
|---|---:|---:|
| first 200 bench questions (`parity-node.mjs`) | 15.8 ms (15.6 to 16.2) | 12.5 ms (12.0 to 13.1) |
| max \|p_wasm − p_native\|, argmax agreement | 9.5e-7, 200 / 200 | 8.9e-7, 200 / 200 |
| Flappy frame (`bench-node.mjs --game flappy`) | 445 µs (424 to 450) | 358 µs (340 to 364) |

The relaxed build is 1.26x faster on the bench questions and 1.24x on a Flappy frame here. In the same session s3-pico took 55.4 µs (53.6 to 58.4) per Flappy frame with SIMD128 and 47.9 µs with relaxed SIMD (two runs at 47.6 and 47.9 µs, one outlier at 98.5 µs). The WASM line of the v1 results files was measured with early exits at 0.5 against a full-depth native dump, so it does not describe a published configuration and is not quoted here. s6-micro v2's evaluation ran it on the full-depth copy (amendment 13), the published file: max |p_wasm − p_native| 8.94e-7, argmax agreement 200 of 200, median 60,673 µs per decision on the first 200 bench questions, on a machine loaded by other jobs.

## s6-secrets-pico: API-key and secret detection, pre-registered

Pre-registration: [`docs/s6-secrets-prereg.md`](docs/s6-secrets-prereg.md), gate in §6, amendments 1 to 6. Results: `runs/s6-secrets-pico/report.md` (`s2-eval`), the rule scanner's score `runs/s6-secrets-pico/baseline.txt` (written before any training), and the run log `runs/all/secrets-train.log`.

**Not a scanner.** The gate below is passed on a synthetic bench. On real files with no secret, measured after publication, the model flags 59.8% of the files at p ≥ 0.9 and 49.4% at p ≥ 0.999 ([real text](#real-text-1411-after-publication), amendment 6). It is published as the result of its pre-registration, not as a tool: do not use it to scan real files.

**Model and recipe.** The s3-pico graph (1 state block × 4 loops, exits after each pass, d 96, vocabulary 1,024) with a 320-token state window and a 192-token head window: 445,345 parameters (203,393 in the pretraining stack), the same count as s3-pico, since the ablation did not adopt Parallax. Recipe from `runs/recipe/recipe.json`: AdamW, learning-rate scale 2.0, no EMA, so `last` is the exported checkpoint. Ternary GGUF: 250,272 bytes.

**Run.** Seed 0, one run, in a throttled mode, so the budgets count working minutes; the log records two cooling pauses (582 s and 900 s) and no change of mode, so the run stayed in its launch mode (duty 0.5, 45 minutes of work and 15 of rest). Steps actually done (`runs/all/secrets-train.log`):

| phase | budget | steps |
|---|---:|---:|
| MLM pretraining, sequences of 256 tokens (`data/text-open`, `data/text`, numeric curriculum) | 8 min | 6,336 (final val loss 2.2398) |
| decision training on `data/secrets` (183,712 training questions), QAT after a 10 % float warm-up, loop dropout 0.25 | 15 min | 7,555 |

**Test bench.** `secret_bench`: 1,996 texts in four file types the training never shows (docker-compose, toml, go, ini), 7,984 questions. Ternary GGUF, last exit, full depth (`report.md`, section GGUF, "Exit 3"); the rule scanner on the same texts:

| question | n | s6-secrets-pico | rule scanner | ECE |
|---|---:|---:|---:|---:|
| secret (noul) | 1,996 | **0.9990** | 0.977 (precision 0.966, recall 0.989) | 0.0009 |
| service (15 options) | 1,996 | **0.9795** | 0.787 | 0.0154 |
| kind (5 options) | 1,996 | 0.9990 | not asked | 0.0010 |
| line (3 to 8 options) | 1,996 | 0.6859 | not asked | 0.3051 |
| all | 7,984 | 0.9158 | | 0.0804 |

**Gate** (§6): secret > 0.977 and service > 0.787 with the ternary GGUF at full depth. **Passed.** The published file `site/public/models/s6-secrets-pico.gguf` has the evaluated weights byte for byte, with its three exit thresholds set to 1 (`gguf-set`, checked with `gguf-inspect`), as amendment 13 of [`docs/s6-gen-prereg.md`](docs/s6-gen-prereg.md) defines a publication at full depth.

- **Per exit** (GGUF, after 1, 2, 3, 4 passes): secret 0.9990 at every exit; kind 0.9990 at every exit; service 0.9775, 0.9790, 0.9800, 0.9795; line 0.6227, 0.6628, 0.6844, 0.6859; all 0.8995, 0.9099, 0.9156, 0.9158.
- **Early exit.** At the calibrated thresholds stored by the run (0.5 / 0.9 / 0.5), every question stops after the first pass (early-exit rate 1.000): secret 0.9990, kind 0.9990, service 0.9775, line 0.6227, all 0.8995. The first exit already gives the full-depth score on secret and kind; the gate was read at full depth, and that is the published setting.
- **Validation** (seen templates, the `secret_synth` val split): ternary 0.998 at every exit, on the 4,000 val questions the training log evaluates, pooled over the four questions (per question: not computed); ECE 0.0010 after calibration on the 9,546 val questions. The latent f32 master weights score 0.756 on val and 0.7124 on test: the trained model is the ternary one (QAT over the last 90 %).
- **The rule scanner** (`sky-data secret-baseline`): its 35 false positives are all random values that are not credentials, under secret-sounding names; its 11 misses are all generic secrets. The model gets the secret question right on all 46 texts the scanner gets wrong. It is wrong on 2 texts, both `x_api_key` with a 32-digit hex value in an ini file, which it calls not a credential; the scanner is right on both. These head-to-head counts come from the native dump of the published file and a port of the scanner's rules that reproduces its published counts (999 / 35 / 11 / 951 and 0.787) exactly.
- **Where service fails.** 41 misses, 34 of them in Go code, where the model names `database` for generic (15), AWS (13), Slack (5) and npm (1) keys; 4 in docker-compose (generic keys named `aws`) and 3 in ini files.
- **Line is weak, and overconfident.** On the 1,010 texts that expose a secret, the model names the right line 383 times; on the 986 texts that expose none, `none` is right every time, and those carry the 0.6859. Most misses point one line (327) or three lines (261) above the exposed one. These breakdowns are computed from the native dump; the pre-registration does not ask for them.

**Size and latency.** 445,345 parameters, 250,272 bytes. Measured on 14/11 on the Apple M5 while a GPU training was running, so an idle machine may be faster:

- native, `sky-eval` over the 7,984 bench questions at full depth: median 789 and 786 µs per decision in two runs (771 to 818 µs per question type);
- WASM, `node crates/sky-web/bench/parity-node.mjs site/public/models/s6-secrets-pico.gguf <dump> --n 200` (Node 24) on the first 200 bench questions, SIMD128 build: max |p_wasm − p_native| 1.2e-6, argmax agreement 200 of 200, median 1,420 and 1,427 µs per decision in two runs;
- both builds later the same day, three alternating runs each, a training still running: SIMD128 1,580 µs (1,577 to 1,581), max gap 1.2e-6; relaxed SIMD 1,219 µs (1,196 to 1,237), max gap 6.0e-7; argmax agreement 200 of 200 in every run.

**What this bench does not say** (§6). It is synthetic: the formats, the file templates and the class shares are our generator's, and every key is a provider's public format with a random body. Beating the scanner here shows that the model reads formats and context (the variable name, a test or publishable key, a placeholder) in file types it never saw. It says nothing about real repositories, and no such conclusion is drawn; the false positives on real files were measured afterwards (next subsection), the recall on real secrets was not.

### Real text (14/11, after publication)

Amendment 6 of [`docs/s6-secrets-prereg.md`](docs/s6-secrets-prereg.md). The published file was run on real files outside its generator, for a possible use in [secretgate](https://github.com/maxgfr/secretgate), a rule-based secrets firewall for coding agents. The measurement, its script and its limits are in that repository (`docs/model-comparison.md`, [pull request 19](https://github.com/maxgfr/secretgate/pull/19)); secretgate did not adopt the model.

**Corpus.** 719 real source, configuration and documentation files in which the rules of secretgate find nothing (its own sources and tests, a sample of its `node_modules`, and `crates/` and `docs/` of this repository), cut into 22,433 chunks of at most 7 lines and 300 characters, the length of the training texts. The secret question is asked on each chunk; a file counts as flagged when one of its chunks is. Since no file holds a secret, every flag is a false positive.

| root | files | chunks | chunks flagged, p ≥ 0.9 / 0.999 | files flagged, p ≥ 0.9 / 0.999 |
|---|---:|---:|---|---|
| secretgate `src/` | 36 | 1,231 | 5.4% / 2.3% | 61.1% / 38.9% |
| secretgate `tests/` | 39 | 1,260 | 10.6% / 6.3% | 79.5% / 69.2% |
| secretgate `README.md` | 1 | 100 | 5.0% / 2.0% | 1 of 1 / 1 of 1 |
| `node_modules/.pnpm` (sample) | 395 | 6,673 | 10.3% / 4.8% | 40.3% / 29.4% |
| skycmd `crates/` | 227 | 11,726 | 10.2% / 6.0% | 86.8% / 78.0% |
| skycmd `docs/` | 21 | 1,443 | 20.7% / 12.5% | 95.2% / 95.2% |
| **total** | **719** | **22,433** | **10.6% / 5.9%** | **59.8% / 49.4%** |

- **The flagged chunks are ordinary text** (imports, prose, JSON); none of the 25 highest scores holds a credential-shaped value.
- **The probabilities are not calibrated outside the generator.** Over the clean chunks the median P(secret) is 0.00009 and the 90th percentile 0.93, so raising the threshold barely helps. Asking the kind and service questions as well does not change it, and neither do smaller or larger chunks.
- **The same script finds the bench score again** on `secret_bench`: recall 0.9980, precision 1.0000, accuracy 0.9990, against 0.9772, 0.7183 and 0.7946 for the rules of secretgate, most of whose false positives there are the test and publishable keys it redacts on purpose.
- **What it means.** The gate is on `secret_bench` and stays passed; nothing in it is reread or changed. The model cannot be used to scan real files, and it stays published as the result of its pre-registration, not as a tool.
- **What it does not test.** Recall on real secrets was not measured (no real secret is in that corpus), and the files come from two projects and one dependency tree.
- **Next.** A second version, if one is made, would be a different task (a rule finds the candidate and the model judges it in its context), with its own pre-registration before any run. Nothing is promised.
- **Outcome (15/11).** That second version was pre-registered, trained once and tested on real repositories. It failed its gate on G1; its file is published as a result (owner's decision of 15/11/2026) (section "s6-secrets-verify-pico" below).

## s6-ear-nano: first run, gates failed

Pre-registration: [`docs/s6-ear-prereg.md`](docs/s6-ear-prereg.md), gates in §6, amendments 2 to 4. Gate file: `runs/s6-ear-nano/eval/gate.json` (`scripts/ear-gate.py gate`); measurements: `runs/s6-ear-nano/eval/eval-{full,crc,control}.json` and `runs/all/ear-train.log`.

- **Model.** 1,532,449 parameters, ternary, 899,200 bytes; without Parallax (not adopted by the ablation) instead of the 1,615,681 of the pre-registration. Smart Turn v3.2 has 8M, 5.2 times more.
- **Runs** (working minutes, throttled mode). Mimi pretraining: 48,626 steps; codebook-0 accuracy on masked val frames 0.2196 against a technical bar of 0.00488. Decision training: 48,205 steps, after a first run of 31,480 steps was discarded for a memory defect before any evaluation (amendment 4). The control without Mimi pretraining: 49,131 steps.

Test split, ternary, full depth:

| gate | n | s6-ear-nano | control (no Mimi pretraining) | bar | verdict |
|---|---:|---:|---:|---:|---|
| end of turn (`ear:turn_end`) | 31,527 | 0.5681 | 0.7647 | 0.9263 (Smart Turn v3.2 on the same clips, Pipecat's report; the v1 harness read 0.8518, see the note below) | **fail** |
| keywords, 12 classes (`ear:keyword`) | 4,890 | 0.8885 | 0.9217 | 0.90 | **fail** |
| zero-shot keywords, English | 1,000 | 0.2100 | 0.3020 | 0.20 (2 × chance) | pass |
| zero-shot keywords, French | 1,000 | 0.1330 | 0.1520 | 0.20 (2 × chance) | **fail** |

Three gates of four fail: s6-ear-nano failed its gate; its file is published as a result (owner's decision of 15/11/2026), as evaluated (`site/public/models/s6-ear-nano.gguf`, amendment 5), with the exit thresholds its run stored (choice 0.5, score 0.9, noul 0.68); its control is not published. The verdict does not change. The control, trained the same way without the Mimi pretraining, is more accurate on all four sets, so in this run the pretraining did not help. ECE over all test sources is 0.0081 at full depth. At the CRC thresholds the model uses 3.42 of its 4 passes on average (early exit on 21.1% of the questions), with the same verdicts. Smart Turn v3.2 has a precision of 0.783 and a recall of 0.971 on the end-of-turn clips; the pre-registration asks for the same two numbers for s6-ear, and the run did not compute them.

**Correction of the end-of-turn bar (16/11/2026).** The table first gave 0.8518 for Smart Turn v3.2, as read by `scripts/ear-baseline.py`. That harness pads a short clip at the end; Smart Turn pads it at the start, and Pipecat reports 0.9263 for v3.2 (CPU, int8) on the same 31,527 clips. On 3,000 of them the v1 harness reads 0.8437 where the reference padding reads 0.9243 (`docs/s6-ear2-study.md`). The bar is therefore 0.9263; s6-ear-nano (0.5681) and its control (0.7647) are under both numbers, and the verdict does not change. The harness itself is fixed and checked against that reference in `docs/s6-ear2-prereg.md` §8.1.

## s6-cast-pico: probabilistic forecasting, pre-registered, gate failed

Pre-registration: [`docs/s6-cast-prereg.md`](docs/s6-cast-prereg.md), gate in §6, amendments 1 to 8. Results: `data/gift-eval-out/s6-cast-pico/summary.json` (`scripts/gift-eval.py score`), with `all_results.csv` next to it and the same files for the flipped model, our TinyCast port and our seasonal naive; run logs `runs/all/cast-{pilot,train,export,eval}.log`. The final report of the run is [`docs/big-run-report.md`](docs/big-run-report.md).

**Model and recipe.** The s6 encoder on patches of 32 points: 2 blocks × 2 loops, an exit after each loop, d 96, context 2,048 points, horizon 64, nine deciles. 369,792 parameters (`cast-train.log`), without Parallax or XSA, which the recipe did not adopt (`runs/recipe/recipe.json`), instead of the 388,320 of the pre-registration. AdamW at 3e-3, batch 256, QAT over the last 20 %, EMA 0.999, seed 1, TinyCast as teacher on 50 % of the windows with a weight annealed from 0.5 to 0. The pilots (3,000 steps each) kept 3e-3 (validation pinball 0.05089, against 0.05121 at 1e-3 and 0.05788 at 1e-2) and the teacher (0.05076 against 0.050891 without). Ternary GGUF of the EMA weights (validation pinball 0.0431, against 0.0442 for the last weights): 142,912 bytes, exit tolerance 0.0128 from the calibration on 2,000 synthetic windows.

**Run.** One run, 480 working minutes, from 14/11 06:07 to 19:05, in the brake modes of amendments 4 to 8 (night 0.95 55:5, idle 0.85 50:10, in use or on battery 0.5 45:15): **250,092 steps**. No restart.

**Test bench.** GIFT-Eval, 97 configurations, 371,330 windows, the official protocol (gluonts `evaluate_model`, the 9 deciles); every metric is divided by the seasonal naive and averaged geometrically. 2,017 windows in 12 configurations have no observation in the model's last 2,048 points and are forecast by the seasonal naive, in both passes (`cast-eval.log`).

| model | nGMASE | nWQL | nMSIS |
|---|---:|---:|---:|
| **s6-cast-pico, without flip** (the reported model) | **0.9471** | **0.6646** | 0.7190 |
| s6-cast-pico, with flip | 0.9099 | 0.6417 | 0.6860 |
| gate: FLAIR, training-free (TinyCast paper) | 0.838 | 0.587 | |
| goal: TinyCast as published | 0.774 | 0.545 | |
| TinyCast, our port, with flip | 0.7856 | 0.5570 | 0.5781 |
| seasonal naive, our harness | 0.9999 | 0.8876 | 2.6778 |

**Gate** (§6): nGMASE ≤ 0.838 and nWQL ≤ 0.587, without flip. **Failed on both metrics.** It failed its gate; its file is published as a result (owner's decision of 15/11/2026), as evaluated (`site/public/models/s6-cast-pico.gguf`, 142,912 bytes, item 8 of the amendments); the verdict does not change. The goal of beating TinyCast is not met either: the gap to the published TinyCast is +0.1731 nGMASE and +0.1196 nWQL, to our generic TinyCast port +0.1615 and +0.1076. With flip the model still misses both bars.

- **Against the seasonal naive.** Better on nGMASE (0.9471) and on nWQL (0.6646 against 0.8876). Per configuration, its MASE is below the seasonal naive's in 52 of 97 and its WQL in 84 of 97; our TinyCast port has a lower MASE in 86 configurations and a lower WQL in 87.
- **By term** (geometric means of the per-configuration ratios, computed from the `all_results.csv` files): MASE 0.860 short, 1.046 medium, 1.105 long, against 0.756, 0.806, 0.848 for our TinyCast port. Medium and long horizons are worse than the seasonal naive.
- **By domain**, MASE / WQL: Econ/Fin 1.018 / 0.901, Energy 1.080 / 0.800, Healthcare 0.760 / 0.614, Nature 0.862 / 0.467, Sales 0.753 / 0.457, Transport 0.833 / 0.644, Web/CloudOps 0.983 / 0.662; our TinyCast port 0.894 / 0.841, 0.882 / 0.661, 0.735 / 0.639, 0.757 / 0.358, 0.734 / 0.445, 0.669 / 0.510, 0.750 / 0.562. Healthcare is the only domain where its WQL is below TinyCast's. The worst configurations are `solar/10T/long` and `solar/10T/medium` (MASE ratios 2.729 and 2.282, against 0.973 and 0.955 for our TinyCast port).
- **Not measured by the run:** the latency, the mean number of exits used on GIFT-Eval (the calibration gives 3.64 of 4 on its synthetic windows) and the gap to Timestral-5M, whose GIFT-Eval scores are not in the repository.

## s6-secrets-verify-pico (s6-secrets v2): a verifier of rule candidates, pre-registered, gate failed

Pre-registration: [`docs/s6-secrets2-prereg.md`](docs/s6-secrets2-prereg.md), gate in §8, amendments 1 to 8 (all before the run) and its result block of 15/11. Results: [`runs/s6-secrets2/eval-test/results.md`](runs/s6-secrets2/eval-test/results.md) and `gate.json` next to it (the test, opened once), `runs/s6-secrets2/eval-validation/results.md` (the validation rehearsal), the freeze of the generator [`runs/s6-secrets2/generator.json`](runs/s6-secrets2/generator.json), and the run's report and logs in `runs/s6-secrets-verify-pico/` (`report.md`, `pretrain.log`, `decide.log`).

**Failed on G1.** It failed its gate; its file is published as a result (owner's decision of 15/11/2026): `site/public/models/s6-secrets-verify-pico.gguf`, the evaluated copy with its thresholds at 1 (sha256 `01e6db3fa71cc2c5563dea222c68e040d3994ee2abbb02b25491a51b8cb54f27`, amendment 9), with a card and no demo. On the CredData test (221 real repositories), the model's line F1 is 0.8561, against 0.7332 for the best rule scanner (Betterleaks): +0.123 where the gate asked for +0.15. The six other gates hold. As §11 wrote before the run, "on real repositories the verifier adds too little to the rules it filters. s6-secrets is not continued in this form." The model is not rerun, and does not enter secretgate. This is the project's best result on real data, and it is a failed gate.

**The task.** v1 (above) reads a whole text and fails on real files. v2 judges one candidate at a time. A rule proposes a candidate span, and a 250 KB ternary model answers about that value, in its context: "Is this value a real credential that should be revoked?" (gated) and "What kind of value is this?" (a real credential, an example or test value, not a credential; reported only). It is trained and tested on real repositories (Samsung CredData) and on issue reports whose test values are real (Zenodo 10.5281/zenodo.19622962). Synthetic candidates are used in training only.

**The generator, `sg-cand-1`.** secretgate's own engine in a candidate mode: the 222 gitleaks rules it ships plus its built-in rules, with the entropy gates, the placeholder test, the stopwords and the allowlists turned off (the filters the verifier is meant to replace), plus a `generic-keyword` rule (a credential keyword in a name, a separator, a value) and, after the one revision allowed (amendment 2), an `auth-scheme` rule. Frozen at secretgate commit `4e93cfa` before any test file was read (amendment 3).

- **Recall on the training and validation repositories** (share of T rows a candidate overlaps): 0.8763 on the gated universe (2,606 of 2,974), 0.6383 on the full one (2,649 of 4,150), against 0.3894 and 0.2790 for secretgate's default engine. The target was 0.95; it was missed and the generator was kept as it was, as §2.2 says. 13.5 candidates per 1,000 lines.
- **On the test repositories** it overlaps 0.9260 of the gated T rows and 0.8367 of all of them. Its misses count as false negatives for every candidate-based system.

**The gated universe.** The rows whose CredData category is exactly `UUID` are removed from the line and candidate universes, for every system (amendment 2): in the training repositories most of their T rows are identifiers of Postman collection exports (783 of 1,176, in two repositories). **This choice was made after reading training and validation rows and before any test file was read.** G1, G2, G3 and G5 are read on the gated universe, and every number below is also given on the full universe.

**Data and splits, by repository.**

- **CredData** at `01e3d135`, 337 repositories. Training: 100 repositories of the cbx-secrets-detector-17M list (licences checked, all permissive); validation: 16 of that list, chosen by a hash fixed before the download; test: the other 221 (7,372 files; 10,463 T rows on the gated universe, 11,603 on the full one). One test repository holds 5,165 of the 10,463 gated T rows, so G1 to G3 had to hold twice: on all 221 repositories and on the 220 left without it (amendment 1).
- **Issue reports**: the paper's splits (test 8,123 candidates, 888 secrets). Test values stay as published; training values are replaced by a word-preserving shape, whose rules were corrected three times before training (amendments 4, 5 and 7).
- **Decontamination** (a training or validation line that also occurs in a test repository is removed): 260 of the 2,127 T candidates of training, 75 of the 459 of validation (amendment 4).
- **Weak negatives**: candidates on unlabelled lines of the training repositories, 4,986 distinct states from 95 repositories after the correction of amendment 4; two hand readings of 200 found no real credential (amendments 4 and 5).
- **Training items read by the decision training**: CredData 49,864, issue reports 40,057, synthetic 40,000, drawn with the weights 0.45 / 0.35 / 0.20.
- **A deviation (amendment 4).** Before any training, while the data folder was explored, the header and the first two rows of the meta file of one test repository were printed: two T rows of category Password, with their line numbers and offsets. No content of a test file was read and nothing was chosen from it; since then every read of `meta/` goes through a guard on the split.

**Model and run.** The s6-secrets-pico graph as it is (s3-pico: 1 state block × 4 loops, exits after each loop, d 96, vocabulary 1,024), state window 320 (the 5% rule of §4.3 kept it: 3.31% of 93,938 states truncated), head window 192: 445,345 parameters, ternary GGUF of 249,984 bytes. The pretraining starts from v1's pretraining stack (amendment 7). One run, seed 0, on 14/11 from 19:39 to 20:42, brake mode duty 0.85 with 50 minutes of work and 10 of rest, exit code 0, no restart (`report.md`):

| phase | budget | steps |
|---|---:|---:|
| MLM pretraining: `data/text-open`, `data/text`, the obfuscated files of the CredData training repositories (30% of the text), numeric curriculum | 12 working min | 11,296 (final val loss 1.4062) |
| decision training on the three sources, QAT over the last 90 %, loop dropout 0.25 | 30 working min | 14,117 |

The operating threshold τ\* = 0.2598 (F1-optimal on the 1,589 CredData validation candidates) was stored in the GGUF with the F0.5 and F2 presets (0.468437 and 0.0780795) before the test was opened. Every reading below is the ternary file at full depth.

**Test, CredData, line level** (the gated unit: a line is positive if it holds a T row; a system flags a line if it accepts a candidate on it or reports a span over it; generator misses are false negatives):

| system | gated precision | gated recall | **gated F1** | gated F1 without the largest | full F1 | full F1 without the largest |
|---|---:|---:|---:|---:|---:|---:|
| **model** | 0.8387 | 0.8741 | **0.8561** | 0.8114 | 0.8153 | 0.7348 |
| Betterleaks 1.10.1 | 0.7694 | 0.7002 | 0.7332 | 0.5367 | 0.6948 | 0.4786 |
| secretgate default engine | 0.8176 | 0.6560 | 0.7279 | 0.5072 | 0.6877 | 0.4445 |
| Gitleaks 8.30.1 | 0.8546 | 0.5980 | 0.7037 | 0.4080 | 0.6626 | 0.3507 |
| detect-secrets 1.5.0 | 0.5504 | 0.2329 | 0.3273 | 0.5014 | 0.3075 | 0.4534 |
| cbx-secrets-detector-17M | 0.8265 | 0.5820 | 0.6830 | 0.8493 | 0.6433 | 0.7614 |
| size-matched logistic regression (256 KiB) | 0.7345 | 0.8652 | 0.7945 | 0.7179 | 0.7579 | 0.6510 |
| potion-base-2M + logistic regression | 0.4966 | 0.8955 | 0.6389 | 0.5536 | 0.6160 | 0.5172 |
| generator alone | 0.2942 | 0.9418 | 0.4483 | 0.3231 | 0.4399 | 0.3158 |
| CredSweeper 1.19.6, ML off | 0.4512 | 0.9970 | 0.6213 | 0.5428 | 0.6441 | 0.5891 |
| CredSweeper 1.19.6, ML on (in sample, no gate) | 0.9946 | 0.9925 | 0.9935 | 0.9872 | 0.9938 | 0.9889 |

- **False alarms** (share of F and X lines flagged, gated): model 0.0599, Betterleaks 0.0748, Gitleaks 0.0363, cbx-17M 0.0435, logistic regression 0.1114.
- **Macro F1** over the 51 test repositories with at least 20 gated T rows: model 0.8091, cbx-17M 0.8133, logistic regression 0.7287, Betterleaks 0.5606.

**Test, CredData, candidate level** (gated universe: 33,809 candidates, 9,623 positive):

| system | precision | recall | F1 | PR-AUC | recall at 1% false alarms | ECE |
|---|---:|---:|---:|---:|---:|---:|
| **model** (τ\* 0.2598) | 0.8363 | 0.9344 | **0.8827** | 0.9335 | 0.5556 | **0.0073** |
| logistic regression (threshold 0.1550) | 0.7331 | 0.9282 | 0.8192 | 0.8648 | 0.2657 | 0.0786 |
| potion-base-2M + LR (threshold 0.0786) | 0.4882 | 0.9504 | 0.6451 | 0.7103 | 0.0924 | 0.0972 |
| generator alone | 0.2846 | 1.0000 | 0.4431 | | | |

On the full universe the model's candidate F1 is 0.8832 and its ECE 0.0072.

**Test, issue reports** (8,123 candidates, 888 secrets, values as published; candidate level):

| system | precision | recall | F1 |
|---|---:|---:|---:|
| **model** | 0.7484 | 0.8975 | **0.8162** (ECE 0.0211) |
| cbx-secrets-detector-17M | 0.8830 | 0.8412 | 0.8616 |
| Gitleaks 8.30.1 | 0.9028 | 0.5856 | 0.7104 |
| secretgate default engine | 0.7699 | 0.5991 | 0.6738 |
| logistic regression | 0.5042 | 0.8874 | 0.6430 |
| Betterleaks 1.10.1 | 0.6102 | 0.6171 | 0.6137 |
| CredSweeper, ML on (out of sample here) | 0.3740 | 0.7185 | 0.4919 |
| CredSweeper, ML off | 0.3087 | 0.8074 | 0.4466 |
| potion-base-2M + LR | 0.2417 | 0.9223 | 0.3831 |
| generator alone | 0.2198 | 0.7883 | 0.3437 |
| detect-secrets | 0.3461 | 0.2950 | 0.3185 |

- **End to end** (the generator and the model on the issue texts, the dataset's positives as the reference; 3,068 accepted flags): F1 0.3219 when the precision counts every accepted flag, 0.7405 when it counts the flags on a dataset candidate only. The logistic regression: 0.2822 and 0.7185.
- **Obfuscation sensitivity** (candidate F1 as published minus F1 after CredData's obfuscation function): model −0.0336 (0.8498 on the obfuscated values), logistic regression −0.0361, Gitleaks +0.0618, cbx-17M +0.0594. The function also rewrites the key names inside a candidate string, so this is a reported number only (amendment 6).

**Clean real text** (§7.7: 8,825 real source, configuration and documentation files of secretgate, its dependency tree and this repository, read without their path, as secretgate's census reads them):

| system | files with an accepted candidate | share | accepted candidates |
|---|---:|---:|---:|
| **model** | 31 | **0.35%** | 110 |
| logistic regression | 77 | 0.87% | 323 |
| potion-base-2M + LR | 401 | 4.54% | 969 |
| secretgate default engine | 11 | 0.12% | 57 |
| generator alone | 2,161 | 24.49% | 14,756 |

v1 flagged 59.8% of a 719-file subset of real files (amendment 6 of [`docs/s6-secrets-prereg.md`](docs/s6-secrets-prereg.md)). With the file's path given, the generator proposes candidates in 2.12% of the files and the default engine flags 0.08%.

**Hand reading of the 110 accepted candidates (§7.7).** The pre-registration asks for every accepted candidate on clean text to be read and for the count that looks like a real credential; its result block listed this as still to do. It was done on 15/11, after the gate was read, and completes §7.7. The 110 candidates of the 31 files were read one by one with their line. **None of the 110 looks like a real credential.** By structure:

| group | candidates | what they are |
|---|---:|---|
| test fixtures shaped like secrets | 34 | mixed-class passwords, passwords inside connection strings and fixture keys split into parts, in secretgate's tests and fixture file (33) and in a test of this repository's synthetic secret generator (1): 26 distinct values in 6 files |
| provider-format keys on a rule's allowlist | 16 | GCP-format values in secretgate's generated rule file, listed in the allowlist of the `gcp-api-key` rule of the gitleaks rule set it ships; 15 have random-looking bodies and one is a sequential-alphabet placeholder. By shape they cannot be told from a live key; whether any is live was not checked |
| example credentials of library documentation | 10 | one example GitHub token of a library's README, 7 times; base64 example values of another README whose decoded text says it is not a real password, 3 times |
| masked placeholders | 2 | a run of X's under a password name, in a library's source |
| public hashes, fingerprints and public keys | 5 | certificate public-key pins in documentation comments (2), release GPG key fingerprints (2), a base64 trust-root file of public keys (1) |
| code | 28 | type and state names, hexadecimal character codes, function calls, array lookups and an npm script (26 in dependencies and in this repository's sources, 2 in secretgate's tests) |
| prose and package metadata | 6 | author fields of package files (4), a translated compiler message, an emoji keyword list |
| other test values, not shaped like a secret | 9 | short dummies, paths, repeated letters, a placeholder word, a dictionary word, base64 of plain words |
| **total** | **110** | 16 in secretgate's sources (1 file), 44 in its tests (7 files), 48 in its dependency tree (21 files), 2 in this repository (2 files) |

The gate uses the raw share (0.35%), as §7.7 says; this reading changes nothing in it.

**Latency and size** (§7.5; measured on the Apple M5 while the GPU was idle, before the baselines ran): native median 1,606 µs per candidate at full depth over 68,110 candidates (372 µs at the first exit); WASM (`parity-node.mjs`, 200 test candidates, a copy with its thresholds at 1) median 2,666 µs, largest gap with the native probabilities 4.17e-07, same answer on 200 of 200. Accuracy on the real question by exit: 0.9461, 0.9580, 0.9600, 0.9614.

**The seven gates, as read** (`gate.json`):

| gate | condition | measured | verdict |
|---|---|---|---|
| G1, every repository | line F1 ≥ best rule scanner + 0.15 | 0.8561 against Betterleaks 0.7332 + 0.15 = 0.8832 | **fails** |
| G1, without the largest repository | same | 0.8114 against Betterleaks 0.5367 + 0.15 = 0.6867 | holds |
| G2, both readings | line F1 ≥ cbx-17M − 0.10 | 0.8561 against 0.6830 − 0.1; 0.8114 against 0.8493 − 0.1 | holds |
| G3, both readings | line F1 > size-matched logistic regression | 0.8561 against 0.7945; 0.8114 against 0.7179 | holds |
| G4 | issue-report candidate F1 ≥ Gitleaks + 0.08 | 0.8162 against 0.7104 + 0.08 = 0.7904 | holds |
| G5 | ECE ≤ 0.05 on CredData test candidates | 0.0073 | holds |
| G6 | native median ≤ 3 ms per candidate | 1,606 µs | holds |
| G7 | clean real files flagged ≤ 1.0% | 0.35% (31 of 8,825) | holds |

The announced goal (G2 at −0.05) is met; it is not a gate and changes nothing. **Verdict: failed on G1**; its file is published as a result (owner's decision of 15/11/2026). A later attempt would be a new design with its own pre-registration.

**Reported with the result, not a way around the gate.**

- **Uncertainty** (paired bootstrap over test repositories, 1,000 resamples, 95%, gated universe): line F1 of the model minus Betterleaks +0.1229 [0.0413, 0.3613] on every repository and +0.2748 [0.1743, 0.3796] without the largest; minus cbx-17M +0.1730 [−0.0654, 0.3074] and −0.0379 [−0.0765, −0.0058]; minus the logistic regression +0.0615 [0.0435, 0.1197] and +0.0935 [0.0646, 0.1302]. ECE 0.0073 [0.0040, 0.0174].
- **The largest repository** holds 5,165 of the 10,463 gated T rows, and the rule scanners do well on it: without it Betterleaks goes from 0.7332 to 0.5367 and the model from 0.8561 to 0.8114. The gate asked for both readings, and the one on every repository is missed.
- **The bar** of +0.15 was fixed before any measurement, with Gitleaks' published 0.442 in mind; on this test Gitleaks is at 0.7037 and Betterleaks at 0.7332. It was not moved.
- **By category** (gated, line F1; positive lines: Password 1,714, Token 777, Key 4,029, Secret 1,389, API 209, Auth 1,016, other 2,966):

  | system | Password | Token | Key | Secret | API | Auth | other | scope `test` | scope `src` |
  |---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
  | model | 0.838 | 0.824 | 0.864 | 0.905 | 0.541 | 0.881 | 0.873 | 0.884 | 0.767 |
  | Betterleaks | 0.545 | 0.672 | 0.838 | 0.897 | 0.459 | 0.251 | 0.638 | 0.770 | 0.667 |
  | cbx-17M | 0.904 | 0.929 | 0.266 | 0.945 | 0.898 | 0.936 | 0.876 | 0.634 | 0.880 |
  | logistic regression | 0.689 | 0.746 | 0.795 | 0.909 | 0.358 | 0.837 | 0.854 | 0.842 | 0.567 |

  The model's weakest category is API (0.541), where cbx-17M is at 0.898; cbx-17M's weakest is Key (0.266), where the model is at 0.864.
- **The UUID rows** were removed from the gated universe by a choice made after reading training and validation rows and before any test file; the full-universe numbers are in the tables above (model line F1 0.8153 against 0.6948 for Betterleaks).
- **The deviation of amendment 4** (one test meta file seen before training) is above.
- **CredSweeper's ML was trained on a random 85% of all CredData lines**, so its CredData numbers (line F1 0.9935) are in sample and enter no gate. On the issue reports, out of sample, its candidate F1 is 0.4919.
- **cbx-17M was rerun out of sample here.** It was trained on the 116 repositories of its list, our training and validation repositories, and none of the 221 test repositories; it ran once on the CPU through the same scorer (57,098 CPU seconds). Its card's numbers (0.805 and so on) are on another test, which is not public, and are not compared.
- **Validation** (the rehearsal, read before the test, no gate): line F1 0.8603 for the model and 0.8604 for the logistic regression on every validation repository.
- **The value-only probe** (§9; trained on CredData training, scored on validation): ROC-AUC 0.9197, PR-AUC 0.8202. The shape of a value alone separates T from F and X on CredData, the artefact §10 warns about; G4, on real values, holds.
- **Unlabelled-line alarms** (accepted flags on lines with no meta row, per 1,000 test files; CredData never judged those lines): model 99.7, Gitleaks 27.0, secretgate default 30.5, Betterleaks 39.1, cbx-17M 486.6, generator 4,469.8.
- **Not measured:** the whole-scan latency of §9 (not built); the accuracy of the kind question on the test and the `test_wild.csv` file of the issue-report archive are not in the results files.

### s6-secrets v3: the rules and the verifier together, a second look, gate failed

Pre-registration: [`docs/s6-secrets3-prereg.md`](docs/s6-secrets3-prereg.md), frozen on 15/11/2026, with its result block. Results: [`runs/s6-secrets3/eval-test/results.md`](runs/s6-secrets3/eval-test/results.md) and `gate.json` next to it, the thresholds [`runs/s6-secrets3/tau3.json`](runs/s6-secrets3/tau3.json), the validation rehearsal `runs/s6-secrets3/eval-validation/`, and the diagnosis that led to it, made on training and validation repositories only, [`docs/s6-secrets3-diagnosis.md`](docs/s6-secrets3-diagnosis.md).

**Failed on G1 again (15/11/2026); the work on this model ends.** v3 is not a new model. It is the system a user of secretgate would get: a line is flagged when secretgate's default engine (commit `4e93cfa`, filters on) flags it, or when the v2 verifier (the evaluated file above, unchanged) accepts a candidate of the frozen generator on it with P(real) ≥ tau3 = 0.34933. Nothing was trained and no scanner, model or baseline was rerun: every flag is the v2 evaluation's. It was judged on the seven gates of v2, on test files already read once for v2. G1 fails again on every repository: the union is at 0.8477 against Betterleaks 0.7332 + 0.15 = 0.8832, where the model alone was at 0.8561. The six other gates hold. On validation all seven held (union 0.8963 on every validation repository); the 221 test repositories did not confirm it. As §3 of the pre-registration wrote, there is no third reading. The reading of v2 does not change.

**Order of operations.**

1. The diagnosis (training and validation repositories only, no test file opened) and the pre-registration, frozen before any test file was read for v3 (`19778c7`).
2. tau3 (0.34933) and the union threshold of the logistic regression (0.155049) chosen on the 16 validation repositories and committed in `runs/s6-secrets3/tau3.json` at 09:05:11 on 15/11 (`370f60f`), before the test read.
3. The gate script at 09:07. Before any v3 computation, the unchanged scorer was rerun and gave the files of v2 again, byte for byte.
4. The test read once, at 09:07:55 (`runs/s6-secrets3/TEST-READ`: "tau3.json at 370f60f8d6c96c47bb54533f1ea96dfe67f48c2d").

**On every table below.** Gated universe: the rows whose CredData category is exactly `UUID` are removed, for every system (amendment 2 of `docs/s6-secrets2-prereg.md`); that choice was made after reading the training and validation rows and before any test file was read; the full universe keeps every meta row. Second look: v3 reads test files that were read once for v2, and its system (the union of the default engine and the model, with tau3) was defined after that reading.

**The seven gates, as read** (`gate.json`, gated universe):

| gate | condition | measured | verdict |
|---|---|---|---|
| G1, every repository | line F1 of the union ≥ best rule scanner + 0.15 | 0.8477 against Betterleaks 0.7332 + 0.15 = 0.8832 | **fails** |
| G1, without the largest repository | same | 0.8062 against Betterleaks 0.5367 + 0.15 = 0.6867 | holds |
| G2, both readings | line F1 of the union ≥ cbx-17M − 0.10 | 0.8477 against 0.6830 − 0.1; 0.8062 against 0.8493 − 0.1 | holds |
| G3, both readings | line F1 of the union > the same union with the size-matched logistic regression in the model's seat (at 0.155049) | 0.8477 against 0.7882; 0.8062 against 0.7171 | holds |
| G4 | issue-report candidate F1 of the union ≥ Gitleaks + 0.08 | 0.8073 against 0.7104 + 0.08 = 0.7904 | holds |
| G5 | ECE of the model ≤ 0.05 (v2's) | 0.0073 | holds |
| G6 | native median ≤ 3 ms per candidate (v2's) | 1,606 µs | holds |
| G7 | clean real files flagged by the union ≤ 1.0% | 0.34% (30 of 8,825) | holds |

**The union against the model alone** (test, CredData, line level):

| system | gated precision | gated recall | **gated F1** | gated F1 without the largest | full F1 | full F1 without the largest |
|---|---:|---:|---:|---:|---:|---:|
| union at tau3 0.34933 (v3) | 0.8039 | 0.8965 | **0.8477** | 0.8062 | 0.8086 | 0.7314 |
| union at v2's τ\* 0.2598 (reported) | 0.7872 | 0.9038 | 0.8415 | 0.7990 | 0.8034 | 0.7268 |
| model alone at τ\* 0.2598 (v2) | 0.8387 | 0.8741 | 0.8561 | 0.8114 | 0.8153 | 0.7348 |
| secretgate default engine | 0.8176 | 0.6560 | 0.7279 | 0.5072 | 0.6877 | 0.4445 |
| Betterleaks 1.10.1 | 0.7694 | 0.7002 | 0.7332 | 0.5367 | 0.6948 | 0.4786 |

- **Paired bootstrap** over test repositories (1,000 resamples, 95%), line F1 of the union minus the model alone: −0.0084 [−0.0181, 0.0072] on every repository and −0.0052 [−0.0188, 0.0118] without the largest (gated); −0.0067 [−0.0158, 0.0074] and −0.0034 [−0.0162, 0.0112] (full). On validation the union was above the model alone (0.8963 at tau3, 0.8834 at τ\*, against 0.8603).
- **The lines the union adds over the model alone** (gated, line units; T a positive line, X a negative line with an X row, F the other negative lines): at tau3 it adds 305 T, 135 X and 649 F, and drops 75 T, 39 X and 224 F (lines the model accepts at τ\* and not at tau3, which the default engine does not flag); at τ\* it adds the same and drops none. The default engine flags 8,262 of the 39,196 gated line units, at a precision of 0.8176 on the test, so the lines it adds cost more in false alarms than they bring.
- **Issue reports**: candidate F1 of the union 0.8073 (precision 0.7178, recall 0.9223), against 0.8162 for the model alone. **Clean real text**: 30 files flagged by the union (3 by the default engine only, 19 by the verifier only, 8 by both), against 31 for the model alone.
- **What it means.** On these test files, neither the verifier alone (v2) nor the system made of the rules and the verifier (v3) reaches a line F1 0.15 above the best rule scanner over all repositories; both do on 220 repositories of 221, and both hold every other gate.

## s6-sleep: the pico pilot, memory not shown, no nano run

Pre-registration: [`docs/s6-sleep-prereg.md`](docs/s6-sleep-prereg.md), the pilot's rules in §5, amendments 1 to 5 (all before any run). Run: `scripts/train-sleep.sh`; log `runs/s6-sleep.log`; the probe readings `runs/s6-sleep/pilot/p{1,2,3,4}/dev/{recall,rule110}.json`; the three rules applied by `scripts/s6-sleep-report.py pilot`, written to `runs/s6-sleep/pilot.json` (run again on a copy of the probe files on 15/11, it writes the same file). The report of §10 (`REPORT.md`) needs the nano runs, so it does not exist.

**What was tested.** Whether a fast-weight memory with "sleep" passes (the model reruns the last frames N − 1 more times while no decision is asked; arXiv 2605.26099) helps a small ternary decision model on a stream of frames. Before any nano run, a pilot on s6-pico (`runs/s6lr/softmax-adamw-x2/decide/last`) checks that the memory is used at all. Four runs, seed 0, warm-started from that base, each step a batch of sequences with probability 0.5 (`recall` 40%, `rule110` 12%, each game 12%), else a stateless batch of the base's decision training, QAT from the first step:

| run | memory | sleep passes N in training | backpropagation | parameters | steps | working minutes |
|---|---|---|---|---:|---:|---:|
| P1, the fast path | yes | drawn in {1, 2, 3, 4} | truncated | 482,497 | 3,684 (its budget: 20 working minutes) | 20.01 |
| P2 | yes | 4 | truncated | 482,497 | 3,684 | 22.45 |
| P3, the paper's way | yes | 4 | through every pass | 482,497 | 3,684 | 30.99 |
| P4, the control | no | | | 445,345 | 3,684 | 14.51 |

The other three runs train for P1's number of steps. The pilot ended on 14/11 at 23:58 (the time of `pilot.json`); every run logged the brake mode duty 0.85 with 50 minutes of work and 10 of rest.

**Readings** (`sky-train s2-probe`, ternary weights, last exit, the 500 dev episodes of each probe from seed 1,000,000,000; `recall` is the accuracy on the ask frames, where a model without memory is at 0.25; `rule110` is the accuracy on the question frames):

| run | `recall` N = 1 | N = 2 | N = 4 | `recall` N = 1 by delay (1–4, 5–8, 9–16, 17–32 frames) | `rule110` N = 1 | N = 2 | N = 4 |
|---|---:|---:|---:|---|---:|---:|---:|
| P1 | 0.2460 | 0.2460 | 0.2460 | 0.234, 0.252, 0.269, 0.229 | 0.4674 | 0.4700 | 0.4691 |
| P2 | 0.2565 | 0.2565 | 0.2565 | 0.251, 0.272, 0.248, 0.255 | 0.5423 | 0.5423 | 0.5423 |
| P3 | 0.2430 | 0.2430 | 0.2430 | 0.231, 0.244, 0.252, 0.245 | 0.5423 | 0.5423 | 0.5423 |
| P4 (no memory, one reading) | 0.2650 | | | 0.262, 0.264, 0.287, 0.247 | 0.5423 | | |

In each memory run the `recall` score is the same at N = 1, 2 and 4, overall and in every delay bucket. The `rule110` scores are reported and decide nothing (§5). For the record, the final validation accuracy of the four runs on the stateless mix (ternary, last exit) is 0.5555, 0.5537, 0.5560 and 0.5535 (`runs/s6-sleep.log`).

**The three rules of §5, as the script read them** (`{"memory_works": false, "truncation_kept": true, "drawn_n_kept": false}`):

| rule | condition | measured | result |
|---|---|---|---|
| the memory works | P1 at N = 1 ≥ P4 + 0.10 | 0.2460 against 0.2650 + 0.10 (P1 − P4 = −0.019) | **no** |
| truncation is kept | P2 at N = 4 ≥ P3 at N = 4 − 0.01 | 0.2565 against 0.2430 − 0.01 | yes |
| the drawn N is kept | P1 at N = 4 ≥ P2 at N = 4 − 0.01 | 0.2460 against 0.2565 − 0.01 | no |

**Decision: no nano run.** The log ends with "the pilot says the memory does not work: no nano run (docs/s6-sleep-prereg.md, section 5)". As §5 says, no nano run takes place: H0 to H3 are not tested, the test seeds of the probes are not read, and no `s6-sleep-nano` model exists or is published. The warm start, the third departure from the paper, was not tested (§5).

**The code, checked after the pilot (15/11/2026): the negative result stands.** The code and the saved runs were read to see whether a defect made the pilot uninformative ([`docs/s6-sleep-diagnosis.md`](docs/s6-sleep-diagnosis.md), with file-and-line references; nothing was built or rerun). None was found:

- **The number of passes reaches the computation.** N goes from the command line to the sleep passes in training and in the probe, and the memory they write is read by every later frame; a unit test asserts that the logits differ between N = 1 and N = 4, and P1's `rule110` answers differ between N = 1 and N = 2 on 22 of 500 episodes. On `recall`, all 4,000 per-ask results are identical across N for P1, P2 and P3.
- **The write, carry and read paths are correct**: the delta-rule step has the right shapes, nothing resets or masks the memory between the write and the read, and nothing is detached beyond the pre-registered truncation.
- **Every memory weight moved** from its initial value (the zero-initialised read-out to a mean |w| of 0.078, 0.044 and 0.077 in P1, P2 and P3).
- **The memory never lowered the training loss**: the sequence losses of the memory runs differ from the control's by −0.0007 (sd 0.0068) for P1, +0.0042 for P2 and +0.0015 for P3, where solving `recall` would be worth about 0.069.
- **Each model answers a near-constant option on the ask frames**: P2 and P3 are both right on only 15 of 4,000 asks, where about 250 would be expected if they answered independently at chance.
- **The probe is sound**: the fact is on screen for one frame, the ask frame shows only the door, and the answer key and the option order are checked by tests and on the training data.
- **A fact, not an explanation:** at P1's last step the gradients reaching the memory projections (1.6e-5 to 3.2e-5) are 10 to 20 times smaller than those of the base blocks (about 3e-4).
- **Not tested:** no unit test checks that a hand-written fact reads back as the right answer through the memory (a show frame, idle frames, then an ask giving the right colour). That test is still to add. There is no rerun.

## LFM3-1.7B: System 1 first, System 2 when needed, pre-registered, failed

Pre-registration: [`docs/lfm3-s1s2-prereg.md`](docs/lfm3-s1s2-prereg.md), its criteria in §6 and its amendments 1 to 5 (all before any run; amendment 2 cancels amendment 1, so the run is the original local protocol). No weight of LFM3 is trained: `LiquidAI/LFM3-1.7B-MLX-4bit` at revision `deb2986b`, 4-bit, mlx-vlm 0.7.11, on the Apple M5. Run: `scripts/lfm3-s1s2.sh` (`experiments/lfm3-s1s2/s1s2.py`); log `runs/lfm3-s1s2.log`; tables `runs/lfm3-s1s2/report-calib.md` (calibration) and `runs/lfm3-s1s2/report.md` (test), numbers `runs/lfm3-s1s2/summary.json`.

**What was tested.** Whether a confidence probe on the answer given without thinking (System 1, an empty `<think></think>`, greedy) can decide, question by question, when the model should think (System 2: thinking on, Liquid AI's sampling for maths, the thinking closed after B tokens, B = 256 or 1,024), and keep the accuracy of always thinking for half its tokens. The probe is a logistic regression on 6 features of the System 1 output, fitted on the calibration split; the threshold τ comes from conformal risk control at α = 0.02 on the calibration split (n = 600), where a question costs 1 when the router is wrong and System 2 alone is right. A question sent to System 2 counts the tokens of both modes.

**The run.** One run, started by `runs/chain.log` on 14/11/2026 at 23:58:15 and ended on 15/11/2026 at 12:44:33 with exit code 0 (12 h 46 min). Brake modes (`runs/brake.log`): 0.85 with 50:10 from the start to 08:31, 0.5 with 45:15 (on battery) from 08:31 to 09:08, then 0.95 with 55:5 to the end. The brake only sleeps between two questions; the tokens and `seconds` are not affected (amendments 2 to 5). Calibration (600 questions: 300 GSM8K train, 300 MATH train) was generated first, then the test (1,000 questions: 500 GSM8K test, 500 MATH-500), in the three modes s1, s2-256 and s2-1024.

**The chosen budget.** B\* is the budget whose always-System-2 accuracy is best on the calibration split, fixed before the test was read: **1,024** (0.797 against 0.742 for 256). τ = 0.332, calibration risk 0.018, the same for both budgets (the probe reads only System 1).

**s2-1024, the chosen budget** (`s1 share`: the share of questions answered by System 1; `s2 forced`: the share of questions whose System 2 thinking reached the budget and was closed):

| split | task | n | s1 acc | s2 acc | router acc | oracle | s1 tok | s2 tok | router tok | s1 share | s2 forced |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| calib | gsm8k | 300 | 0.887 | 0.923 | **0.913** | 0.947 | 329 | 640 | 428 | 0.93 | 0.17 |
| calib | math | 300 | 0.487 | 0.670 | **0.690** | 0.693 | 740 | 1227 | 1578 | 0.77 | 0.50 |
| calib | all | 600 | 0.687 | 0.797 | **0.802** | 0.820 | 535 | 934 | 1003 | 0.85 | 0.33 |
| test | gsm8k | 500 | 0.824 | 0.878 | **0.862** | 0.910 | 396 | 728 | 547 | 0.90 | 0.23 |
| test | math | 500 | 0.510 | 0.714 | **0.710** | 0.718 | 722 | 1188 | 1518 | 0.78 | 0.46 |
| test | all | 1000 | 0.667 | 0.796 | **0.786** | 0.814 | 559 | 958 | 1032 | 0.84 | 0.35 |

**s2-256, reported:**

| split | task | n | s1 acc | s2 acc | router acc | oracle | s1 tok | s2 tok | router tok | s1 share | s2 forced |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| calib | gsm8k | 300 | 0.887 | 0.917 | **0.913** | 0.950 | 329 | 455 | 394 | 0.93 | 0.62 |
| calib | math | 300 | 0.487 | 0.567 | **0.593** | 0.597 | 740 | 843 | 1303 | 0.89 | 0.87 |
| calib | all | 600 | 0.687 | 0.742 | **0.753** | 0.773 | 535 | 649 | 848 | 0.91 | 0.74 |
| test | gsm8k | 500 | 0.824 | 0.854 | **0.850** | 0.884 | 396 | 506 | 495 | 0.90 | 0.69 |
| test | math | 500 | 0.510 | 0.594 | **0.608** | 0.614 | 722 | 819 | 1245 | 0.88 | 0.86 |
| test | all | 1000 | 0.667 | 0.724 | **0.729** | 0.749 | 559 | 663 | 870 | 0.89 | 0.77 |

**The pre-registered claims** (test, B\* = 1,024; the block of `runs/lfm3-s1s2.log`):

- **H1 (diagnostic, no threshold):** thinking helps LFM3-1.7B in maths. System 1 0.824 and System 2 0.878 on GSM8K; 0.510 and 0.714 on MATH-500.
- **H2, accuracy kept:** router 0.786 ≥ System 2 0.796 − 0.02: **yes**.
- **H3, compute halved:** router 1,032 tokens ≤ 0.5 × System 2's 958: **no**. System 1 alone already uses 559 tokens on average, more than half of System 2's, and every question sent to System 2 pays for both.
- **H4, better than both modes:** router ≥ max(System 1, System 2) + 0.01: **no** (0.786 against 0.796 for System 2 alone).
- **Verdict: failure** (success needs H2 and H3). The oracle, the better of the two answers for each question, is 0.814: the ceiling of a perfect router at this budget.
