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

Latency of `sky-infer` (native) and of the browser module `sky-web` (WASM), measured on an Apple M5 (rustc 1.97.1, Node 24.10 / V8 for WASM) on a shared machine: every number is the best median of 3 interleaved rounds. s1 rows use the trained models (`site/public/models/s1-pico.gguf`, `models/s1-nano` converted to f16 GGUF). **s2 rows use random-init models of the ladder shapes** (`sky-convert --random s2-pico|s2-nano` with the real tokenizers `data/tok/tok-s2-{1024,2048}.json`) until trained ones exist in `models/s2-*`: their latency is representative, their answers are not. Random weights never reach the default τ, so s2 rows are given at full depth (τ = 1, worst case) and at the first exit (τ = 0, best case); a calibrated model lands in between.

The state is a Flappy Drone state text (`vy -2 dx 35 up 20 dn -5 nxt -18 alt 32 t N`, a new state every frame); a frame answers the action question and, with 2 questions, the `danger` question on the same state. s1 re-encodes question + options + state for every question (only the tokenization of the prefix is cached); s2 encodes the static stream once (LRU cache), the state once per frame for both questions, and runs the fusion head per question.

Native, `sky-bench <model> --questions 1|2 --iters 4000` (µs per frame, median / p95):

| model | params | file | compute | 1 question | 2 questions (same state) |
|---|---:|---:|---|---:|---:|
| s1-pico (trained) | 202,305 | 533 KB (f16) | f32 | 188 / 208 | 407 / 443 |
| s1-nano (trained) | 1,134,209 | 2.7 MB (f16) | f32 | 974 / 1007 | 1980 / 2066 |
| s2-pico, full depth | 310,337 | 1.2 MB (f32) | f32 | 67 / 70 | 73 / 79 |
| s2-pico, first exit (1 of 3 layers) | 310,337 | 1.2 MB (f32) | f32 | 28 / 31 | 34 / 37 |
| s2-pico, full depth, no static cache | 310,337 | 1.2 MB (f32) | f32 | 164 / 173 | 313 / 322 |
| s2-pico ternary, full depth | 310,337 | 176 KB (ternary) | f32 (dequantized at load) | 67 / 70 | 73 / 79 |
| s2-pico ternary, full depth | 310,337 | 176 KB (ternary) | packed add/sub | 141 / 146 | 148 / 152 |
| s2-nano, full depth | 1,532,033 | 5.9 MB (f32) | f32 | 306 / 316 | 338 / 353 |
| s2-nano, first exit (1 of 4 layers) | 1,532,033 | 5.9 MB (f32) | f32 | 118 / 129 | 147 / 162 |
| s2-nano, full depth, no static cache | 1,532,033 | 5.9 MB (f32) | f32 | 599 / 613 | 1015 / 1088 |
| s2-nano ternary, full depth | 1,532,033 | 684 KB (ternary) | f32 (dequantized at load) | 298 / 313 | 329 / 349 |
| s2-nano ternary, full depth | 1,532,033 | 684 KB (ternary) | packed add/sub | 667 / 701 | 706 / 724 |

WASM (SIMD128), `node crates/sky-web/bench/bench-node.mjs`: `bench()` of the play page, one Flappy Drone decision = both questions + the physics step, µs per decision (median / p95):

| model | file | compute | static cache | µs / decision |
|---|---:|---|---|---:|
| s1-pico (trained) | 533 KB | f32 | prefix tokens | 694 / 730 |
| s1-nano (trained) | 2.7 MB | f32 | prefix tokens | 3541 / 3619 |
| s2-pico, full depth | 1.2 MB | f32 | on | 104 / 115 |
| s2-pico, first exit | 1.2 MB | f32 | on | 51 / 53 |
| s2-pico, full depth | 1.2 MB | f32 | off | 560 / 575 |
| s2-pico ternary, full depth | 176 KB | f32 (dequantized) | on | 105 / 114 |
| s2-pico ternary, full depth | 176 KB | packed add/sub | on | 166 / 179 |
| s2-nano, full depth | 5.9 MB | f32 | on | 478 / 517 |
| s2-nano, first exit | 5.9 MB | f32 | on | 196 / 209 |
| s2-nano, full depth | 5.9 MB | f32 | off | 1746 / 1816 |
| s2-nano ternary, full depth | 684 KB | f32 (dequantized) | on | 501 / 523 |
| s2-nano ternary, full depth | 684 KB | packed add/sub | on | 788 / 822 |

Ternary kernels vs f32 products on the s2 ladder shapes (`sky-bench --kernels`, native NEON, 20 rows = a state stream, 2 rows = the fusion markers; µs per product):

| model | shape (n x k) | rows | f32 | ternary packed | packed / f32 |
|---|---|---:|---:|---:|---:|
| s2-pico | 192 x 64 | 20 | 4.42 | 12.71 | 2.88 |
| s2-pico | 256 x 64 | 20 | 5.92 | 16.94 | 2.86 |
| s2-pico | 64 x 128 | 20 | 2.91 | 8.07 | 2.77 |
| s2-pico | 128 x 64 | 2 | 1.23 | 1.30 | 1.06 |
| s2-nano | 384 x 128 | 20 | 17.42 | 48.33 | 2.77 |
| s2-nano | 128 x 256 | 20 | 11.29 | 31.40 | 2.78 |
| s2-nano | 256 x 128 | 2 | 3.60 | 4.94 | 1.37 |
| s2-micro | 768 x 256 | 20 | 68.00 | 188.44 | 2.77 |
| s2-micro | 512 x 256 | 2 | 12.89 | 20.88 | 1.62 |

Reading:
- Per frame, s2-pico is 5.6x faster than s1-pico natively (73 vs 407 µs, full depth) and 6.7x faster in WASM (104 vs 694 µs): the question and options leave the per-frame path (static cache) and the second question costs only its fusion head (~6 µs) because the state encoding is shared.
- Without the static cache, s2 loses most of its advantage (it encodes the static stream every question): the cache is what makes the architecture fast.
- Early exit at the first exit halves the per-question cost again (pico: 67 → 28 µs native).
- Ternary storage cuts the file 7x (pico 1.2 MB → 176 KB, nano 5.9 MB → 684 KB). The packed add/sub kernels cost three vector ops (XOR, AND, ADD) per 4 weights against one FMA, and these matrices fit in L1, so the f32 path is 2-3x faster per product and 1.6-2.2x faster per decision on both NEON and SIMD128. `TernaryCompute::Auto` therefore expands ternary weights to f32 at load on every platform: the download stays small, the compute is f32. `--ternary-compute packed` (native) / `Engine.withOptions(bytes, true, true)` (WASM) keep the packed kernels.
- Accuracy: the s2 engine matches the mlx-rs graph of `sky-train s2-golden` within 4e-6 on every exit probability (f32 and ternary files, packed and f32 compute; tolerances 1e-4 / 1e-3).

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

