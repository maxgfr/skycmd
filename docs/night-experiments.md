# Night experiments: smaller and faster than s3-pico (2026-10-12)

Goal of the night: find the smallest and fastest decision model that still plays all four games (flappy, rescue, dodge, doom) at oracle level, and raise general decision accuracy. Reference: `models/ship/s3-pico.gguf` (445,345 parameters, 251 KB ternary, τ = 0.6). Every number below was measured during this night (unless attributed to BENCH.md) on one Apple M5 (10 cores, 16 GB, rustc 1.97.1). GPU training ran one job at a time; the CPU phases (DAgger collection, arena play, benchmarks) overlapped with training, so absolute latencies carry some load noise. Comparisons are always interleaved on the same load.

## Results

| candidate | params | KB | held-out acc (no doom) | typed_decisions_bench | flappy | rescue | dodge completion | doom completion/kills | mean layers f/r/d/doom | arena p50 µs f/r/d/doom | sky-bench µs flappy/doom | gate |
|---|---:|---:|---:|---:|---|---|---|---|---|---|---|---|
| s3-pico (ship, reference) | 445,345 | 251.1 | 0.543 | 0.382 | 200/200, 0 crash | 1.00 (o 1.00) | 0.867 (o 0.867), 14 veto | 1.00/0.77 (o 1.00/0.77) | 1.44/1.52/1.05/1.73 | 21/55/14/71 | 27.5/71.0 | **pass** |
| **s3-pico-long** | 445,345 | 251.1 | 0.565 | 0.364 | 200/200, 0 crash | 1.00 (o 1.00) | 0.867 (o 0.867), 19 veto | 1.00/0.77 (o 1.00/0.77) | 1.47/1.54/1.05/1.81 | 22/55/14/71 | 27.6/71.0 | **pass** |
| s3-femto (round 3, best of 7) | 221,121 | 160.8 | 0.523 | 0.311 | 179/200, 6 crash | 1.00 (o 1.00) | 0.867 (o 0.867), 24 veto | 1.00/0.77 (o 1.00/0.77) | 1.49/1.56/1.05/1.52 | 12/31/7/41 | 15.7/41.1 | fail |
| s3-femto-es (expand-and-sparsify, round 3) | 237,505 | 163.8 | 0.554 | 0.290 | 200/200, 0 crash | 0.00 (o 1.00) | 0.667 (o 0.867), 1901 veto | 1.00/0.77 (o 1.00/0.77) | 1.47/3.40/1.34/1.83 | 18/158/7/61 | 24.6/60.8 | fail |
| s3-femto-f1 (base: decide only, τ 1 play) | 187,521 | 147.2 | 0.526 | 0.312 | 123/200, 20 crash | 0.48 (o 1.00) | 0.233 (o 0.867), 20983 veto | 1.00/0.77 (o 1.00/0.77) | 4.00/4.00/4.00/4.00 | 37/81/4/124 | 13.6/33.6 | fail |
| s3-femto-v512 (base: decide only, τ 1 play) | 188,353 | 111.6 | 0.498 | 0.297 | 0/200, 30 crash | 0.02 (o 1.00) | 0.200 (o 0.867), 15427 veto | 0.07/0.15 (o 1.00/0.77) | 4.00/4.00/4.00/4.00 | 81/208/8/16 | 22.7/251.9 | fail |

How to read the table:
- **held-out acc**: early-exit accuracy of the GGUF at its own τ on the `test` split without doom (data/decisions + flappy, rescue, dodge test records: 25,319 questions). `sky-train s2-eval --gguf ... --gguf-only` through `scripts/night-eval.sh`. typed_decisions_bench is the LocalLLaMA typed-decisions benchmark (2,000 test-only questions, never trained on).
- **games**: `sky-arena`, 30 seeded episodes per game (seed 7), at the file's τ, against the oracle on the same episodes ("o" = oracle). Gate: flappy 200/200 with 0 crashes; rescue ≥ 0.95; dodge completion ≥ 95 % of the oracle's with < 100 vetoes; doom completion ≥ 95 % and kill rate ≥ 90 % of the oracle's (`scripts/night-gate.py`).
- **KB**: file size in 1,000 bytes (the s3-pico file is 251.1 KB = 245.2 KiB; the sections quote the export log's KiB).
- **mean layers**: state-block applications per decision in that arena run (of 4).
- **arena p50**: median µs per decision inside `sky-arena` (one question at a time, every game state).
- **sky-bench**: native median µs per decision, one question, static cache on, file τ, Flappy and Doom presets, best of 3 rounds interleaved across models (`scripts/night-bench.sh`).
- Rows marked "base" did not get DAgger rounds (see below), so their game columns are the decide-stage model at full depth, not a gate attempt.


## 1. s3-femto: half of s3-pico

`configs/s3-femto.json`: s3-pico with d = 64, one 64-wide head, 1 state block × 4 loops, 1 static block, 2 fusion blocks, ff = 128, vocab 1024, value-fused numbers, exits after every loop. 221,121 parameters (s3-pico: 445,345), 157.0 KiB ternary GGUF (s3-pico: 245.2 KiB). Variants: `s3-femto-f1` (1 fusion block: 187,521 parameters, 143.8 KiB) and `s3-femto-v512` (vocab 512 from a new `data/tok/tok-s2-512.json`, trained in 22 s with `sky-data train-tokenizer --num --vocab 512`: 188,353 parameters, 109.0 KiB).

Pipeline (`scripts/night-s3.sh`): the train-s3.sh recipe with the s3-pico budgets (pretrain 8 min with the numeric curriculum, decide 15 min on decisions + games (doom included) + numeric with QAT after a 10 % float warm-up and loop dropout 0.25), then DAgger rounds of 8 min on **all four games** (s3-pico's rounds used three), then closed-loop τ calibration (`scripts/calibrate-exit.sh` with GAMES including doom) and the 30-episode gate.

| stage | steps | val acc (ternary, last exit) | flappy (pipes, crashes) | rescue | dodge (completion, vetoes) | doom (completion / kills) |
|---|---:|---:|---|---:|---|---|
| pretrain 8 min | 12,373 (s3-pico 4,814) | token CE 2.936 (s3-pico 2.864) | | | | |
| decide 15 min (base) | 9,424 (s3-pico 6,440) | 0.549 (s3-pico 0.530) | 21.9, 30 | 0.49 | 0.833, 213 | 1.00 / 0.77 |
| round 1 (r1 from base) | | 0.570 | 180.2, 6 | 0.97 | 0.867, 40 | 1.00 / 0.77 |
| round 2 (r2 from g1) | | 0.573 | 47.8, 30 | 1.00 | 0.700, 28 | 1.00 / 0.77 |
| **round 3 (r3 from g2)** | | 0.577 | **178.7, 6** | **1.00** | **0.867, 24** | **1.00 / 0.77** |
| round 4: aggregated r1..r4 (r4 from g3) | | (0.626*) | 122.6, 19 | 1.00 | 0.867, 14 | 1.00 / 0.77 |
| round 5: aggregated + 60k flappy-only DAgger from g4 | | (0.646*) | 5.0, 30 | 1.00 | 0.867, 9 | 1.00 / 0.77 |
| round 6: aggregated r1..r6 from g3, 0.3× learning rate | | | 13.7, 30 | 1.00 | 0.867, 14 | 1.00 / 0.77 |
| round 7: only r1 + r3 (collected from crashing policies), from g3 | | 0.579 | 12.2, 30 | 1.00 | 0.867, 23 | 1.00 / 0.77 |

Arena at full depth (τ = 1), 30 episodes, oracle: flappy 200/0, rescue 1.00, dodge 0.867, doom 1.00 / 0.77. (*) val includes the DAgger groups of every round, not comparable with the other rows.

What happened:
- From the first DAgger round on, femto plays rescue, dodge and doom at oracle level (dodge completion = the oracle's with 9-40 vetoes; doom completion 1.00 and kill rate = the oracle's 0.772 in every round). It even learns faster than s3-pico, whose round 1 was 5.4 pipes and rescue 1.00, dodge 0.667 / 4,205 vetoes.
- **Flappy is the one game it does not master.** Every round lands in one of two modes: about 6 crashes in 30 episodes (rounds 1 and 3) or a collapse (12-48 pipes, every episode crashing). The g5 trace shows the collapse: thrust/glide probabilities sit at 0.43-0.48 / 0.52-0.57 on every frame, including frames where the drone is below the gap and falling. τ does not change it (g3 at τ 0.6, 0.8, 0.95, 1.0: the same 178.7 pipes and 6 crashes; g4: the same 122.6 / 19).
- Hypotheses tested with one round each and rejected: (a) the DAgger sets of a good policy lack recovery states (round 7 trained only on sets collected from crashing policies: collapse); (b) the per-round restart at peak learning rate causes forgetting (round 6 at 0.3× the learning rate: collapse); (c) flappy DAgger data from a crashing policy has uninformative soft labels (max target < 0.7 on 39 % of f5's action records vs 47 % for r1 and 50 % for the oracle data: no difference); (d) textbook DAgger aggregation (round 4: worse than round 3). A d = 64 single-head femto seems to sit right at the capacity edge for flappy's knife-edge thrust decisions (21,300 consecutive correct decisions per 200-pipe episode).
- Best femto: **g3**, calibrated in closed loop to τ = 0.6 (the same τ as s3-pico; τ = 0.5 adds 3 crashes): it passes rescue, dodge and doom and fails flappy with 178.7 pipes and 6 crashes in 30 episodes.
- Speed of femto g3 at τ = 0.6: **15.7 µs per Flappy decision vs 27.5 µs for s3-pico (1.75× faster)** and **41.1 vs 71.0 µs per Doom decision (1.73×)** in `sky-bench` (quiet machine, interleaved, best of 3); arena p50 12 / 31 / 7 / 41 µs (flappy / rescue / dodge / doom) vs 21 / 55 / 14 / 71 µs for s3-pico, both re-run on the quiet machine. Mean layers 1.49 / 1.56 / 1.05 / 1.52, the same pattern as s3-pico.
- Held-out accuracy (test without doom) 0.523 vs 0.543 for s3-pico; per source the games stay close (dodge 0.944 vs 0.958, flappy 0.828 vs 0.836, rescue 0.889 vs 0.889) and the general sources drop (typed_decisions_bench 0.311 vs 0.382, notam deel_subject 0.059 vs 0.160).

Variants (decide stage only; no DAgger rounds, see "Budget" below): `s3-femto-f1` reached val 0.551 (13,251 steps) and held-out 0.526, the same as femto at that stage, so the second fusion block buys nothing measurable before DAgger and costs 33,600 parameters (13.2 KiB). `s3-femto-v512` is clearly worse: val 0.492 (femto 0.549), held-out 0.498, and its base model cannot play (flappy 0 pipes with 30 crashes, rescue 0.02, doom completion 0.07): halving the vocabulary splits the game keys and instructions into more pieces (measured with the engine's tokenizer: a Flappy state is 21 tokens instead of 9, a Doom state 41 instead of 26, a Doom question with two options 72 instead of 56), so value fusion helps less and every frame costs more, and 13.6k pretraining steps were not enough to recover.

## 2. Expand-and-sparsify FFN (SPEC-S3 §9)

Implementation, behind the new optional config field `ffn_es` (absent = unchanged GeGLU; no existing file, config or golden changes):
- **sky-train** (`crates/sky-train/src/s2/es.rs`, `model.rs`, `config.rs`, `gguf.rs`, `golden.rs`): every encoder block (state and static) drops `ffn_up`; its FFN is `x + W_down · kWTA(P · LN(x))`. `P` (`8·d × d`) is a fixed sparse ±1/√fan_in matrix (fan_in = 8 distinct inputs per unit) generated by splitmix64 from a u64 seed. k-WTA keeps, per token, the units ≥ the k-th largest pre-activation (k = round(0.05 · 8·d) = 26 for d = 64) with their value; the threshold carries no gradient (`partition` under `stop_gradient`). `W_down` (`d × 8·d`) is ternary like every encoder matrix.
- **GGUF**: only the keys `skycmd-s2.ffn_es_seed` (u64), `ffn_es_units`, `ffn_es_k`, `ffn_es_fan_in`. No tensor for `P` (0 bytes of weights).
- **sky-infer** (`crates/sky-infer/src/s2/es.rs`, `mod.rs`, `model.rs`): the projection is regenerated from the seed at load; the forward pass computes the pre-activations, finds the k-th largest per token and **reads only the k active rows of `W_downᵀ`** (k · d multiply-adds instead of d · 8·d).
- **Golden**: `sky-train s2-golden --config configs/s3-femto-es.json --prefix s3es` → `crates/sky-infer/tests/golden/s3es{.json,-f32.gguf,-ternary.gguf}`, checked by `tests/golden_s3es.rs`: max |Δp| 7.0e-6 (f32), 2.9e-6 / 3.4e-6 (ternary packed / f32 compute), 1.8e-4 / 1.5e-4 (int8 / LUT compute), within the 1e-4 / 1e-3 tolerances. The generator is duplicated in both crates and pinned by a fixed-output test whose constants were computed by an independent Python port.

Kernel work (9-token state, d = 64, microbenchmark `es_forward_timing`): the first version (scalar gathers over the 8 non-zeros + `select_nth_unstable`) made a full-depth Flappy decision 122 µs vs 56 µs for GeGLU. A dense panel product for `P` plus a k-element running minimum was worse (310 µs: the running minimum mispredicts; selection alone took 49 µs of the 58 µs FFN). The kept version: dense panel product (5.4 µs for 9 rows) + a branchless pre-filter at mean + 1.2·std then quickselect on the ≈ 60 survivors (exact, with a fallback when fewer than k survive): 4.9 µs; the FFN went from 58 to 11.6 µs. A row-vectorized sparse variant (transpose to [d][rows], 8 broadcast FMAs per unit) was not faster (13.0 µs) and was dropped.

Comparison with GeGLU femto (same pretrain / decide / 3 DAgger budgets, same data):

| | GeGLU femto | ES femto |
|---|---:|---:|
| parameters / ternary file | 221,121 / 157.0 KiB | 237,505 / 159.9 KiB (+2.9 KiB: `W_down` is d × 8d = 32,768 vs 3 · d · ff = 24,576 per block) |
| pretrain 8 min: steps, val token CE | 12,373, 2.936 | 9,477, **2.862** |
| decide 15 min: steps, val acc | 9,424, **0.549** | 8,536, 0.533 |
| val acc after rounds 1 / 2 / 3 | **0.570 / 0.573 / 0.577** | 0.550 / 0.558 / 0.552 |
| held-out (no doom), at τ 0.6: all / typed_decisions_bench / rescue / notam_multi | 0.523 / **0.311** / **0.889** / 0.352 | **0.554** / 0.291 / 0.747 / 0.635 |
| gate at τ 0.6 (30 episodes) | fails flappy (178.7, 6 crashes) | fails rescue (0.00) and dodge (0.667, 1,901 vetoes) |
| sky-bench at τ 0.6 (Flappy / Doom) | **15.7 / 41.1 µs** | 24.6 / 60.8 µs |
| round 3 at full depth: flappy / rescue / dodge / doom | 178.7 (6 crashes) / **1.00** / **0.867, 24 vetoes** / 1.00, 0.77 | **200 (0 crashes)** / 0.00 / 0.667, 1,901 vetoes / 1.00, 0.77 |
| µs per decision at equal depth, 1 loop (Flappy / Doom) | **16.1 / 43.1** | 25.0 / 61.3 |
| µs per decision at equal depth, 4 loops (Flappy / Doom) | **52.5 / 128.2** | 87.2 / 202.3 |

Reading: at a slightly *larger* file (+1.8 %), the ES FFN is worse on everything that needs reading option texts against the state (rescue play never rose above 0.04 in four models, rescue test accuracy 0.747 vs 0.889; dodge kept 1,045-2,744 vetoes) and on typed_decisions_bench (0.291 vs 0.311), and it is 1.4-1.7× slower at equal depth. Its higher overall held-out number (0.554 vs 0.523) comes entirely from notam_multi (0.635 vs 0.352), a source where answering "false" to every yes/no question already scores 0.575 (see section 3), not from better decisions. On speed: its fixed up-projection has 8·d = 512 outputs per token where GeGLU has 2·ff = 256, so even computed densely at full SIMD speed it costs more than the learned matrix it replaces, and the top-k selection is extra. It is better at two things: masked-LM token CE per wall-clock minute (2.862 vs 2.936) and flappy (200/200 with 0 crashes from the decide stage on), consistent with the random expansion acting as a good fixed feature map for low-dimensional numeric states. Not a win for these models; kept behind `ffn_es` for future work (an equal-size variant with `expand: 6` would be the next test).


## 3. s3-pico-long: the s3-pico shape with longer budgets

Same config as s3-pico (`configs/s3-pico.json`, 445,345 parameters, 245.2 KiB). Pretrain **20 min** (24,617 steps, val token CE 2.709; s3-pico: 8 min, 2.864) and decide **35 min** (14,504 steps; s3-pico: 15 min, 6,440) with more weight on the HF decision mix (`decisions=0.5, games=0.45, numeric=0.05`; s3-pico used 0.3 / 0.6 / 0.1), then the same three 8-minute DAgger rounds (on all four games), closed-loop τ calibration and the gate. The budgets are below the 30 / 45 min asked for: see "Budget and deviations".

| stage | val acc (ternary) | flappy | rescue | dodge (completion, vetoes) | doom | held-out (no doom) | typed_decisions_bench |
|---|---:|---|---:|---|---|---:|---:|
| s3-pico (ship) | 0.574 (round 3) | 200, 0 | 1.00 | 0.867, 14 | 1.00 / 0.77 | 0.543 | 0.382 |
| pico-long decide (base, τ 1 play) | 0.574 (s3-pico decide: 0.530) | 200, 0 | 1.00 | 0.667, 1,129 | 1.00 / 0.77 | 0.565 | 0.351 |
| round 1 | 0.587 (s3-pico 0.555) | 200, 0 | 1.00 | 0.800, 1,175 | 1.00 / 0.77 | | |
| round 2 | 0.588 (s3-pico 0.561) | 200, 0 | 1.00 | 0.767, 29 | 1.00 / 0.77 | | |
| **round 3 = s3-pico-long, τ 0.6** | 0.587 (s3-pico 0.574) | **200, 0** | **1.00** | **0.867, 19** | **1.00 / 0.77** | **0.565** | 0.364 |

Closed-loop calibration picks τ = 0.6 (τ = 0.5 adds one flappy crash), the same as s3-pico; at τ 0.6 the 30-episode gate passes with mean layers 1.47 / 1.54 / 1.05 / 1.81 (flappy / rescue / dodge / doom). Same shape, same speed: 27.6 µs per Flappy decision and 71.0 µs per Doom decision in `sky-bench` (s3-pico: 27.5 / 71.0).

Held-out accuracy per source (early exit at each file's τ, test split without doom):

| model | all | dodge | flappy | rescue | deel_subject | notam_multi | typed_decisions_bench | when2call | all without notam_multi |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| s3-pico (ship) | 0.543 | 0.958 | 0.836 | 0.889 | 0.160 | 0.320 | 0.382 | 0.367 | 0.596 |
| s3-femto g3 | 0.523 | 0.944 | 0.828 | 0.889 | 0.059 | 0.352 | 0.311 | 0.360 | 0.563 |
| s3-femto-es | 0.554 | 0.950 | 0.830 | 0.747 | 0.034 | 0.635 | 0.290 | 0.351 | 0.535 |
| s3-femto-f1 base | 0.526 | 0.909 | 0.808 | 0.867 | 0.047 | 0.450 | 0.312 | 0.338 | 0.544 |
| s3-femto-v512 base | 0.498 | 0.843 | 0.741 | 0.533 | 0.062 | 0.580 | 0.297 | 0.376 | 0.478 |
| s3-pico-long base | 0.565 | 0.952 | 0.826 | 0.886 | 0.049 | 0.559 | 0.351 | 0.368 | 0.567 |
| s3-pico-long (final) | 0.565 | 0.955 | 0.843 | 0.895 | 0.058 | 0.521 | 0.364 | 0.365 | 0.575 |

The overall held-out number is fragile. notam_multi (4,882 of the 25,319 questions, 84 % of them yes/no with 68 % "false" labels) is out of distribution for every model here; answering "false" to every yes/no question already scores 0.575 on it, and the models swing between 0.32 and 0.63 on it depending on their yes/no bias. typed_decisions_bench (2,000 questions, LocalLLaMA, never trained on) and the game sources are the more informative columns.

## Budget and deviations

- One Apple M5 GPU, one training job at a time, enforced by a global lock in `scripts/night-s3.sh` (GPU phases: pretrain, decide, calibrate, export, eval). CPU phases (DAgger collection over four games, arena play, benchmarks) overlapped with other candidates' training. About 5 h of GPU time was available between the build (01:06) and the last training job (06:09).
- **pico-long ran 20 / 35 min instead of about 30 / 45 min.** At 03:20 the queued GPU work (femto rounds, ES, f1, v512, pico-long) added up to about 200 min, which would have ended after the deadline; pico-long had not started yet, so I relaunched it with 20 / 35 (still 2.5× / 2.3× s3-pico's budgets).
- **s3-femto-f1 and s3-femto-v512 got no DAgger rounds** (decide stage, calibration and base arena only). Their rows are not gate attempts. v512 was clearly behind after decide; f1 matched femto's decide-stage accuracy but its three rounds would have pushed pico-long past the deadline.
- **s3-femto got 7 DAgger rounds instead of 3** (rounds 4-7 were the flappy experiments in section 1); the reported femto is round 3, the best one.
- DAgger rounds use all four games (doom included), as asked; s3-pico's rounds used three.
- The final `sky-bench` numbers were taken at the end with nothing else running, interleaved across models, best of 3 (s3-pico: 27.5 µs Flappy, as in BENCH.md). The arena p50 of the four gate rows were re-run sequentially on the quiet machine; the full-depth per-round arenas ran while other jobs trained, so their latencies are only indicative. Earlier runs under load gave the same ratios (femto 1.75-1.83× faster than s3-pico).

## Files

New (nothing shipped or committed was modified; every new behaviour is off by default):
- `crates/sky-train/src/s2/es.rs`, `crates/sky-infer/src/s2/es.rs`: the seeded projection, k-WTA and the sparse down-projection; fixed-output and dense-reference tests, an ignored microbenchmark (`es_forward_timing`).
- `crates/sky-infer/tests/golden_s3es.rs` and `crates/sky-infer/tests/golden/s3es{.json,-f32.gguf,-ternary.gguf}`: the ES golden.
- `configs/s3-femto.json`, `s3-femto-f1.json`, `s3-femto-v512.json`, `s3-femto-es.json`; `data/tok/tok-s2-512.json`.
- `scripts/night-s3.sh` (candidate pipeline with the GPU lock), `night-dagger-agg.sh` (aggregated DAgger round), `night-eval.sh` (held-out eval), `night-bench.sh` (interleaved latency), `night-gate.py` (gate verdict), `night-collect.py` (results table); `runs/night-s3-femto/round{5,6,7}.sh` (the femto flappy experiments).
- Outputs: `models/candidates/<name>/`, `runs/night-<name>/`, `runs/night-baseline/`, `data/dagger-night-<name>/`.

Modified (additive, behind new fields or flags):
- `crates/sky-train/src/s2/config.rs` (`ffn_es` field, `EsFfn`), `model.rs` (ES parameters and forward, test), `gguf.rs` (ES keys, round-trip test), `golden.rs` (ES note), `mod.rs`, `eval.rs` + `main.rs` (`s2-eval --gguf-only`).
- `crates/sky-infer/src/s2/mod.rs` (`ffn_es` from the GGUF keys), `model.rs` (`Block.up` optional, `Block.es`), `random.rs` (`ffn_es: None`).
- Tests: `cargo test --workspace --exclude sky-train --release` (38 test binaries) and `cargo test --release -p sky-train` (73 tests) pass on the final code (also run after the first ES version); existing goldens are unchanged.

## Conclusion

**No candidate beats s3-pico on size or speed while passing the gate.** The one that passes, s3-pico-long, has s3-pico's exact shape, size and speed.

- **s3-femto (round 3) is the closest miss.** 221,121 parameters (0.50× s3-pico), 160.8 KB (0.64×), 15.7 vs 27.5 µs per Flappy decision (1.75× faster) and 41.1 vs 71.0 µs per Doom decision (1.73×), with the same calibrated τ (0.6) and early-exit profile. It plays rescue, dodge (0.867 = the oracle's, 24 vetoes) and doom (completion 1.00, kills = the oracle's) at oracle level. It fails the gate on flappy: 178.7 / 200 pipes with 6 crashes in 30 episodes. Seven DAgger rounds and four recipe changes (aggregation, a flappy-heavy set, a 0.3× learning rate, crash-rich data only) never reached 0 crashes: flappy lands at about 6 crashes or collapses. Held-out accuracy drops from 0.543 to 0.523 (typed_decisions_bench 0.382 → 0.311).
- **The femto variants did not help.** s3-femto-f1 (one fusion block, 147 KB, 13.6 µs) matched femto after decide (held-out 0.526) but got no DAgger rounds tonight, so it is untested on the gate. s3-femto-v512 (112 KB) is clearly worse: a 512 vocabulary inflates the state (Flappy 21 vs 9 tokens, Doom 41 vs 26), held-out 0.498, and its base model cannot play.
- **Expand-and-sparsify FFN: implemented end to end, not worth it here.** It is behind `ffn_es` in sky-train and sky-infer, with a u64 seed and no weights in the GGUF, top-5 % k-WTA and a down-projection that reads only the active units. Its golden matches within 7e-6 (f32) and 2e-4 (int8 / LUT compute). At +1.8 % file size it fails the gate (rescue 0.00, dodge 0.667 with 1,901 vetoes) and is 1.4-1.7× slower than GeGLU at equal depth (24.6 vs 15.7 µs at τ 0.6). The 512 fixed outputs per token cost more than the 256 learned ones they replace, and the selection is extra. It does win on MLM token CE (2.862 vs 2.936) and plays flappy perfectly, which suggests the random expansion is a good fixed feature map for small numeric states. It is not a good general FFN at d = 64.
- **s3-pico-long passes the gate and meets one of its two accuracy goals.** Pretrain 20 min, decide 35 min, then the same three DAgger rounds; τ 0.6 again. Results: flappy 200/0, rescue 1.00, dodge 0.867 with 19 vetoes, doom completion 1.00 with kills = the oracle's. Held-out accuracy rose from 0.543 to 0.565 (above the 0.543 goal). typed_decisions_bench is 0.364, below s3-pico's 0.382, so that goal is missed. The 0.565 is mostly a yes/no bias on notam_multi (0.320 → 0.521; always answering "false" there scores 0.575). Without that source, pico-long is at 0.575 against s3-pico's 0.596. So I would not claim better general decisions. The longer budgets clearly helped the games: the base model already played flappy 200/0 and rescue 1.00, where s3-pico's base played 145 pipes and 0.08 (BENCH.md). Caveat: the budgets were 20 / 35 min, not 30 / 45.
- **Recommendation.** Keep `models/ship/s3-pico.gguf`. s3-femto round 3 (`models/candidates/s3-femto/s3-femto-g3-tau.gguf`) is a drop-in 1.75× faster, 36 % smaller model for rescue, dodge and doom, but not for flappy. A width between 64 and 96, or two heads at d = 64, is the obvious next test for flappy; it was not tried tonight.

Files changed: new `crates/sky-{train,infer}/src/s2/es.rs`, `crates/sky-infer/tests/golden_s3es.rs` with its golden files, `configs/s3-femto{,-f1,-v512,-es}.json`, `data/tok/tok-s2-512.json`, `scripts/night-{s3.sh,dagger-agg.sh,eval.sh,bench.sh,gate.py,collect.py}` and this document. Modified, additively and behind new fields or flags: sky-train `s2/{config,model,gguf,golden,eval,mod}.rs` and `main.rs` (`ffn_es`, `--gguf-only`); sky-infer `s2/{mod,model,random}.rs`. Nothing under `site/`, `models/ship/`, `runs/ship-*`, the existing configs, scripts or goldens was touched. All tests pass.

