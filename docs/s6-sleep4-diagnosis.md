# s6-sleep v3: why the memory reaches 1.00 at the pico size and about 0.49 at the nano size

Written on 17/11/2026, after the result of `docs/s6-sleep3-prereg.md` (Result block of 17/11/2026). This is a diagnosis from files that already existed. Nothing was trained, evaluated, probed or exported for it. The numbers come from:

- the training logs of the pilot and the nano runs (`runs/s6-sleep3/pilot/<run>/log.jsonl`, `runs/s6-sleep3/nano/<arm>-s<seed>/{a,b}/log.jsonl`) and their commands (`cmd.txt`);
- the console logs `runs/all/sleep3-pilot.log` and `runs/all/sleep3-nano.log`;
- the dev2 readings (`runs/s6-sleep3/pilot/{m,c}/dev2/`, `runs/s6-sleep3/nano/<arm>-s<seed>/dev2/recall.json`);
- the configs that the script wrote (`runs/s6-sleep3/pilot/sleep3.json`, `runs/s6-sleep3/nano/sleep3.json`);
- the code that writes these files (`crates/sky-train/src/s2/train.rs`, `memory.rs`, `crates/sky-train/src/optim.rs`, `scripts/train-sleep3.sh`).

The `runs/` folders are not in the repository: they are gitignored files of the worktree where the runs took place. Small standard-library scripts outside the repository tabulated the logs. No test-seed file was opened; the test numbers below are quoted from `runs/s6-sleep3/REPORT.md`.

How to read the log fields (`train.rs`, lines 2445 to 2462). Each sequence step writes one line with `seq_game`, `passes` (N), `seq_step_loss` and `grad_norm`. A `recall` step also writes:

- `ask_ce`: the deep-supervision cross-entropy over the ask frames;
- `ask_acc`: the share of ask frames that the last exit answers right (chance 0.25);
- `ask_splits`: the counts behind `ask_acc`, by delay bucket and by "one colour / several colours";
- `aux_ce` and `aux_acc`: the auxiliary head, divided by asks × memory layers, so at the nano size they are the mean over two layers.

Each reading is made on the step's own training batch, before its update, on train sequences. "Recall step" below means a sequence step whose batch is `recall`. "Excess" means accuracy minus the chance level of 0.25.

## 1. Summary

1. **The nano memory learns several times more slowly per training step than the pico memory did.** After the same number of `recall` steps (200 to 224), the pilot answered 0.995 of the asks of its batches and its auxiliary head decoded 0.998. The three nano runs were at 0.38 to 0.50 and their auxiliary heads at 0.47 to 0.61. In sequences, nano phase A saw 3,728 `recall` sequences; the pilot had finished after about 3,400.
2. **The minute budget bought fewer than half the pilot's steps.** A nano sequence step took about 2.6 working seconds against about 0.77, 3.4 times longer. With 1.5 times more minutes, phase A ran 292 steps against 663 and phase B 877 against 2,051. In all, nano saw 393 `recall` steps against 933 for the pilot.
3. **The runs ended in the middle of their learning curve.** At the end of phase A, train accuracy was still rising by about 0.04 per 25 `recall` steps. Phase B first lost ground: accuracy fell from 0.41 to 0.29 for seed 0, and the pilot fell too, from 0.999 to 0.887. Phase B then won it back. It ended about where phase A had ended, at +0.007 to +0.078 in accuracy.
4. **The fall with the delay is the shape of a memory that is only partly trained, and at the nano size it is a steeper shape.** At similar overall accuracy, the excess at 17 to 32 frames is 0.17 to 0.36 of the excess at 1 to 4 frames at the nano size. It was 0.6 to 0.93 in the pilot while it was learning. The decay α = 0.9975 per frame predicts 0.615 → 0.595 over the four buckets, not 0.615 → 0.330. A per-frame retention of 0.93 to 0.95 fits the curve, a half-life of 10 to 13 frames, which points to later writes erasing earlier ones.
5. **What the files rule out:** the decay α as the cause of the delay curve; stale colours of a door shown again (the "one / several colours" gap is fully predicted by the delay mix); a gap made by the export, the probe or the ternary reader (train readings, ternary under QAT, agree with dev2); a dying gradient, v2's failure (memory gradient norm 5 to 22 throughout nano phase A); seed variance (a range of 0.08); phase B as a lasting loss.
6. **What the files cannot decide:** which difference between the sizes slows the learning per step: the rate (0.82 times the pilot's), two memory layers whose auxiliary loss is averaged (0.5 per layer), two heads of 72 against one of 96, two loops against four, another base and mix. Nor whether more steps would reach 1.00, nor any f32 reading.
7. **Not checked before the nano runs.** C5, the CPU check, ran only at the pico size. No CPU check asked whether the nano memory trains within the steps that its minute budget buys.

## 2. The candidates

### (a) Fewer `recall` steps in phase A

The curves, in windows of 25 `recall` steps of phase A. The source is `log.jsonl` of `pilot/m-a` and `nano/sleep-s{0,1,2}/a`, fields `ask_acc` and `aux_acc`; `ask_ce` is reported where it matters.

| recall steps | pilot `ask_acc` | pilot `aux_acc` | nano s0 `ask_acc` | s0 `aux_acc` | s1 `ask_acc` | s1 `aux_acc` | s2 `ask_acc` | s2 `aux_acc` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 75–99 | 0.268 | 0.638 | 0.264 | 0.433 | 0.306 | 0.409 | 0.238 | 0.362 |
| 100–124 | 0.323 | 0.789 | 0.271 | 0.493 | 0.358 | 0.466 | 0.273 | 0.381 |
| 125–149 | 0.357 | 0.911 | 0.317 | 0.574 | 0.415 | 0.497 | 0.298 | 0.387 |
| 150–174 | 0.599 | 0.961 | 0.329 | 0.529 | 0.443 | 0.473 | 0.297 | 0.390 |
| 175–199 | 0.888 | 0.993 | 0.384 | 0.590 | 0.486 | 0.505 | 0.333 | 0.414 |
| 200–224 | 0.995 | 0.998 | 0.415 | 0.606 | 0.500 (200–214, end) | 0.523 | 0.376 (200–222, end) | 0.469 |
| 225–232 | 0.995 | 0.998 | 0.449 (end) | 0.605 | | | | |
| 525 (end of A) | 1.000 | 1.000 | | | | | | |

- **Was nano still improving when phase A ended?** Yes, all three seeds. The answer was still rising by 0.03 to 0.04 per window. The auxiliary head was close to level for seed 0 (0.590, 0.606, 0.605) and still rising for seeds 1 and 2. Over the last 50 `recall` steps of phase A, `ask_ce` was 1.2663, 1.1685 and 1.3219 (chance 1.3863).
- **Where was the pilot at the same count of `recall` steps?** It was done. At 200 to 224 steps it read `ask_acc` 0.995 and `ask_ce` 0.054.
- **When the answer left chance.** This is the first of ten consecutive `recall` steps with a mean `ask_ce` under 1.36, the rule of `s6-sleep3-report.py`. The pilot left chance at `recall` step 139 (step 167). The nano seeds left at steps 161, 99 and 171, which is about the same moment. The difference is what came after. The pilot went from chance to 0.995 in about 70 `recall` steps. Nano seed 1, which left earliest, went from chance to 0.50 in 116.
- **In sequences.** Every step has 16 sequences at both sizes. Nano phase A drew 3,728 `recall` sequences; the pilot was at 0.995 after about 3,400 (some 210 steps). There are 8,950 `recall` train sequences. Over both phases the pilot drew 14,928, 1.7 passes over them; nano seed 0 drew 6,288, 0.7 passes.

**Verdict: supported as a contributor, ruled out as the cause by itself.** The nano runs stopped while still improving. But at an equal count of steps, or of sequences, the pilot had finished and nano was at 0.38 to 0.50. The larger factor is the slower learning per step, and the files cannot say what causes it (see (d) and (e)).

### (b) What the minute budget buys at the nano size

Budgets, steps and step times. The source is `log.jsonl`: `elapsed_min` on the lines with `seq_loss` (one every 50 sequence steps) and the final `step`.

| | pilot M | pilot control | nano sleep s0 | nano control s0 |
|---|---:|---:|---:|---:|
| phase A budget | 8 working minutes | 663 steps (5.80 min) | 12 working minutes | 292 steps (7.83 min) |
| phase A steps per working minute | 82.9 | | 24.3 | |
| seconds per sequence step, phase A (windows of 50, about 10% stateless steps included) | 0.67–0.96, about 0.77 | 0.50–0.77, about 0.57 | 2.32–3.19, about 2.6 (s1 2.54–2.86, s2 2.41–2.82) | 1.58–2.30, about 1.8 |
| phase B budget | 12 working minutes | 2,051 steps (8.53 min) | 18 working minutes | 877 steps (9.98 min) |
| phase B steps per working minute | 170.9 | | 48.7 | |
| seconds per sequence step, phase B | 0.60–0.85 | | 2.26–2.65 | 1.23–1.73 |

- **Sizes of a step.** Batch sizes are the same at both sizes (`cmd.txt`): 128 stateless rows, 16 sequences of 64 `recall` frames, `--seq-frac` 0.9 in phase A and 0.5 in phase B. A sequence step costs 3.4 times more at the nano size. The cost of the memory is 1.35 times at pico and 1.44 times at nano, as the ratio of the M arm to the control. The rest is the model: d 144 against 96, two static layers against one, a vocabulary of 2,048 against 1,024.
- **`--sleep-chunk 2`** (amendment 3) splits each batch into 8 parts. The nano runs could not run without it, so its cost can be estimated only against the first nano run, which amendment 3 describes: it was unchunked and stopped by the memory guard. That run was at step 203 after 5.8 working minutes. The chunked seed-0 run was at about step 131 at 5.8 minutes, interpolated between step 110 at 4.95 minutes and step 165 at 7.21 minutes. So chunking cost at least a third of the steps per minute. "At least", because the unchunked run was at 16 GB of 16 and may itself have been slowed. Its log was removed, so this rests on one line of amendment 3.
- **The clock counts the two validations at step 0** (`train.rs`, the `qat_start` block inside the training loop). The first 20 steps took 0.38 working minutes in the pilot and 1.16 at nano. This is small next to the step times.
- **The brake does not change the steps.** `Brake::rest` (`session.rs`, lines 228 to 255) moves the run clock forward by the time it rests. The pilot ran at duty 0.5; the nano log shows duty 0.95 and 0.5 in turn.

**Verdict: supported.** One and a half times the minutes bought 0.44 times the steps in both phases: 292 against 663, and 877 against 2,051. Nano made 19.4 `recall` steps per working minute in phase A, the pilot 65.8. The minute budget set the number of steps, and the step time at the nano size was never measured before the budget was fixed.

### (c) Phase B wearing away what phase A learned

The readings below are in the same order in every cell: `ask_ce`, `ask_acc`, `aux_ce`, `aux_acc`. "Last 50" means the last 50 `recall` steps of the phase.

| run | end of A (last 50) | start of B (first 25) | end of B (last 50) |
|---|---|---|---|
| pilot | 0.0014, 0.9994, 0.0000, 1.0000 | 0.3606, 0.8872, 0.5682, 0.9306 | 0.0069, 0.9983, 0.0045, 0.9984 |
| nano s0 | 1.2663, 0.4084, 0.9088, 0.6018 | 1.4075, 0.2859, 1.3218, 0.4952 | 1.2888, 0.4152, 1.0084, 0.5696 |
| nano s1 | 1.1685, 0.4781, 1.0708, 0.4977 | 1.3210, 0.3950, 1.3222, 0.4527 | 1.1846, 0.4866, 1.1313, 0.5157 |
| nano s2 | 1.3219, 0.3525, 1.1877, 0.4385 | 1.3632, 0.3241, 1.3844, 0.3914 | 1.2581, 0.4309, 1.1733, 0.4698 |

- **Both sizes drop at the change of phase, and both recover.** The pilot was back above 0.96 within 50 to 75 `recall` steps of phase B and at 1.000 at its end. Nano ended phase B at +0.007, +0.009 and +0.078 in accuracy over its end of A, and −0.064 to +0.023 in `ask_ce`. Phase B held 160, 186 and 174 `recall` steps at the nano size, against 407 in the pilot. The first 50 to 100 of them went to winning back the drop.
- **The drop does not come from the sleep passes.** In phase B's N = 1 batches, nano seed 0 reads 0.364 over the whole phase, against 0.408 at the end of phase A. The phase change has three parts, and the files cannot separate them:
  - a fresh AdamW;
  - a new schedule: `wsd` in `optim.rs` (lines 177 to 186) ends phase A at 0.1 of the peak rate and starts phase B back at the peak after a warm-up of 2% of the phase;
  - a new mix, with games and stateless steps.
- **Sleep passes in phase B** trade long delays against short ones, at both sizes:
  - pilot, second half of phase B: 0.987 at 17 to 32 frames for N = 1 against 0.946 for N = 4;
  - nano, test readings in `REPORT.md`: 1 to 4 frames 0.6147 → 0.6290 and 17 to 32 frames 0.3302 → 0.2867 from N = 1 to N = 4.

**Verdict: ruled out as the cause of the gap, supported as a cost.** Phase B lowered no nano run for good, but at the nano size it spent most of its few `recall` steps winning back its own drop. The control arm stays at chance in both phases, as it should: `ask_ce` is about 1.387, with two windows of 25 steps at 1.46 and 1.49.

### (d) Capacity or interference of the memory itself

**The dimensions.** Sources: the configs; `memory.rs` (`specs`, `Memory::zeros`); the parameter counts printed in the console logs.

| | pico (pilot) | nano |
|---|---:|---:|
| d, heads × head dimension | 96, 1 × 96 | 144, 2 × 72 |
| memory layers (`state_layers`) | 1 | 2 |
| loops of the state block (reads per frame per layer) | 4 | 2 |
| state per stream (floats) | 9,216 | 20,736 |
| memory parameters (with the auxiliary heads) | 28,320 | 126,432 |
| write strength β, decay α | 0.5, σ(6) = 0.99753 | 0.5, 0.99753 |

The write (`memory.rs`, lines 449 to 453) is `S ← αS + Σᵢ (β/n)(vᵢ − αS·kᵢ)kᵢᵀ` over the n state tokens of every frame. Every frame writes: idle frames, the shows of other doors and the ask frames themselves.

**What the decay predicts.** Delays are drawn uniformly within each bucket (`sky-games/src/recall.rs`, `plan`). The mean of α^d over the four buckets is 0.9938, 0.9840, 0.9695 and 0.9412; α^32 = 0.924, a half-life of 280 frames. Suppose the excess accuracy is proportional to what is kept, and anchor it on the 1–4 bucket. The decay alone then predicts 0.615, 0.611, 0.606 and 0.595 for the test N = 1 curve. The curve is 0.6147, 0.5590, 0.4530 and 0.3302. The pilot, with the same α, answered 1.000 at every delay on dev2. **The decay is ruled out.**

**What fits instead.** Under the same proportional reading, a retention r per frame fitted to the excess gives:

| curve | r per frame | half-life |
|---|---:|---:|
| test, N = 1 | 0.937 | 10.7 frames |
| dev2, seed 0 | 0.947 | 12.7 frames |
| dev2, seed 1 | 0.935 | 10.3 frames |
| dev2, seed 2 | 0.934 | 10.2 frames |
| test, N = 4 | 0.928 | 9.3 frames |

That is about 6% lost per frame, where α loses 0.25%. In this write rule, a frame whose keys overlap the door's key erases part of what is stored under it. The loss is about β·mean(cos²) per frame. With β = 0.5, 6% would be a mean squared cosine of about 0.13 between the door's key and the keys of later frames. This is arithmetic on an assumed model, not a measurement. The nano runs logged no memory trace. The only measured overlap is C2's at initialisation at the pico size: 0.73 between a shown door's key and an idle frame's keys (`docs/s6-sleep3-diagnosis.md` §5). Training lowered the overlap at the pico size (§6.1 there).

**The same picture, from the logs.** At similar overall accuracy, long delays lag much more at the nano size. The reading is the excess at 17–32 divided by the excess at 1–4. Training batches come from `ask_splits`; dev2 is pooled over the three seeds.

| reading | overall accuracy | excess ratio, 17–32 over 1–4 |
|---|---:|---:|
| pilot A, recall steps 100–149 | 0.340 | 0.61 |
| pilot A, recall steps 150–199 | 0.744 | 0.93 |
| pilot B, first 50 recall steps | 0.863 | 0.82 |
| diagnosis, seed 1, steps 200–250 (`docs/s6-sleep3-diagnosis.md` §6.1) | about 0.72 | 0.81 |
| nano s0 A, last 33 recall steps | 0.423 | 0.29 |
| nano s1 A, recall steps 150–199 | 0.464 | 0.26 |
| nano s1 A, last 15 recall steps | 0.500 | 0.36 |
| nano s0 B, last 10 recall steps | 0.489 | 0.17 |
| nano dev2, three seeds, N = 1 | 0.482 | 0.21 |

At the pico size, learning lifted every delay together. At the nano size, the short delays came first and the long ones lagged. So the nano curve is not the pico curve cut short. The learned nano memory loses more per intervening frame.

**More writes, more loss.** Sleep passes replay the last frames, up to 8 (`SLEEP_EVERY`), and each replayed frame writes again. At the nano size they raise 1–4 and lower 17–32: on dev2, seed 0, 0.568 → 0.588 and 0.334 → 0.274 from N = 1 to N = 4; on test, 0.6147 → 0.6290 and 0.3302 → 0.2867. This is what an erasing write predicts: replays refresh recent shows and erase old ones more.

**Shows of the same door, with the "one colour / several colours" split.** On test, a door shown with several colours scores 0.5052 against 0.4717 for one colour (`REPORT.md`). On dev2 (three seeds pooled, N = 1, `episodes[].asks` and `episodes[].shown`), asks at long delays are more often "one colour": 1,650 of 3,000 at 17–32, against 1,338 of 3,000 at 1–4. Take each bucket's pooled accuracy and the bucket mix of each group: that predicts 0.4740 for one colour and 0.4900 for several. Observed: 0.4745 and 0.4896. Within the buckets:

| bucket | one colour | several colours |
|---|---:|---:|
| 1–4 | 0.602 | 0.605 |
| 5–8 | 0.537 | 0.547 |
| 9–16 | 0.449 | 0.467 |
| 17–32 | 0.340 | 0.306 |

The gap at 17–32 is about 1.9 standard errors. **A stale colour of the same door costs nothing at short delays, and the pooled gap is a delay effect.**

**Shows of other doors and idle frames.** Both grow with the delay. The probe files hold no frame positions, so the two cannot be separated. The load of the memory since the start of the episode does not explain the curve: within a bucket, the asks of the second half of an episode are not consistently worse than those of the first half. For example, dev2 seed 0, N = 1, early against late: 0.545 against 0.597 at 1–4 and 0.445 against 0.465 at 9–16. The exception is seed 2 at 5–8: 0.532 against 0.470.

**The gradient did not die.** The norm of the memory's gradient group (`grad_norm_groups.mem`, every 50 sequence steps) stayed at 5 to 22 throughout nano phase A. This is not v2's failure.

**Verdict.**

- The decay α as the cause of the delay curve: **ruled out**.
- Overwriting by later frames, at a rate of about 6% per frame, as what shapes the curve: **supported**. Its mechanism, keys of later frames overlapping the door's key, **cannot be decided from the files**, because no key overlap was logged at the nano size.
- Capacity in the sense of too few floats: **ruled out as stated**. The nano state is 2.25 times larger. What the nano memory has less of is room per head (72 dimensions against 96) and reads per layer (2 loops against 4); whether either matters cannot be decided.

### (e) Configuration differences other than the size

Sources: `cmd.txt` of each run, the configs, `scripts/train-sleep3.sh`, the console logs.

| | pilot (pico) | nano | where it is written |
|---|---|---|---|
| base | `runs/s6lr/softmax-adamw-x2/decide/last` | `runs/s6-nano-g3/decide/last` | v2 §5, §6 |
| stateless mix | decisions 0.3, games 0.6, numeric 0.1 | decisions 0.25, distill 0.30, games 0.35, numeric 0.10 | `train-sleep3.sh` lines 93 to 95 |
| peak rate (AdamW) | 2.0e-3; `mem.*` 2.0e-2 | 1.63e-3; `mem.*` 1.63e-2 | `rate 144 1` (line 139: 2e-3·√(96/144)); `LR_X` = 1 from `runs/s6-nano-lr/summary.json` |
| auxiliary loss | weight 1 on one memory layer | weight 1, **averaged over two layers** (0.5 per layer's head) | `memory.rs` lines 962 to 977; not in the pre-registration (see below) |
| state loops / state layers / static layers / heads | 4 / 1 / 1 / 1 | 2 / 2 / 2 / 2 | the base configs |
| sequence batches, length, `--seq-frac`, passes, QAT | 16 × 64 frames, 0.9 then 0.5, N = 1 then 1–4, ternary from step 0 | the same | `cmd.txt` |
| `--sleep-chunk` | none | 2 | amendment 3 |
| `SKY_MLX_CACHE_GB` | 2 (default) | 1 | amendment 3; the logs read `mlx_cache_mb` 2,148 and 1,075 |
| brake | duty 0.5 | duty 0.95 and 0.5 in turn | console logs; no effect on steps |
| phase A share | 8 of 20 min; 24.4% of the steps | 12 of 30 min; 25.0% of the steps | v2 M4 says "the first 40% of the steps" |
| schedule clock | minutes (M) | minutes (seed 0), steps (seeds 1, 2) | `--minutes` against `--steps`; the same `wsd` shape |
| stop rule after phase A | computed: 0.0014 against 1.336 | **not computed** by the script; computed here: 1.2663, 1.1685, 1.3219 | `train-sleep3.sh` `nano()` runs no `rep stop` |
| global gradient clip 1.0, recall steps of phase A above it | 46% (median norm 2.8 → 0.05 over the phase) | 88%, 98%, 92% (median about 9) | `optim.rs` line 162; `grad_norm` |

- **The auxiliary loss is not the one the pre-registration describes.** v3 T2 says the head "reads what the last memory layer retrieves at the door token". In the code (`memory.rs`, `specs` and `seq_frames`), every memory layer has its own head, which reads the mean over the frame's valid tokens of what that layer retrieves. The term in the loss is the mean over the asks and over the layers. At the pico size, with one layer, the only departure is "mean over tokens" against "at the door token", which is the diagnosis's form (§6.1). At the nano size there are two heads, each weighted 0.5, and the first one sits on layer 0, not on the last layer. The Result block records the two heads (1,152 parameters), and `docs/s6-sleep3-readings.md` item 5 records the averaging of `aux_acc`. Neither says that this halves the auxiliary gradient per layer.
- **The stop rule would not have stopped any nano run.** Seed 2 passed by 0.014.
- **The rate.** At 0.82 times the pilot's, it accounts for at most a 1.2-fold slowdown if learning is proportional to the rate. That is far from the gap of (a).

**Verdict: cannot be decided from the files.** Several differences go with the size. The rate, the auxiliary weight per layer, the base and the mix all differ. None was varied alone at the nano size, so the slower learning per step cannot be put on one of them.

### (f) Ternary quantisation

- Both sizes train with ternary projections from step 0 (`--qat-frac 1.0`; `prepare` in `model.rs` ternarises the `mem.*` projections under QAT; the console logs say "projections ternary under QAT"). So every training reading above was made with the ternary forward pass.
- The dev2 probe also reads ternary weights. The two agree. At the end of phase B, the nano train batches read 0.415, 0.487 and 0.431 (mixed N); dev2 reads 0.4728, 0.5275 and 0.4467 at N = 1. The pilot reads 0.998 and 1.000.
- No f32 reading of `recall` exists at either size. The only f32 and ternary pairs in the logs are the stateless validation accuracies on the `final` lines (for example nano s0 phase A: 0.4945 f32, 0.5847 ternary), and those are not `recall`.

**Verdict: cannot be decided from the files.** Whether an f32 memory would learn faster at the nano size has no reading. What is ruled out is a gap made by the export or by the reader: train and dev2 agree at both sizes.

### (g) Seed variance

- Nano dev2 at N = 1: 0.4728, 0.5275 and 0.4467, a range of 0.081 and a standard deviation of 0.041. The controls read 0.2542, 0.2567 and 0.2522.
- The order of the seeds is the order of their training curves. It matches the `recall` step at which the answer left chance (s1 at 99, s0 at 161, s2 at 171) and the accuracy over the last 50 `recall` steps of phase A (0.478, 0.408, 0.353).
- At the pico size, the step at which the answer left chance varied from 156 to 240 over the CPU runs (C5, batches of 8). Every pico run then reached at least 0.99 on its train batches. The pilot itself is one seed.

**Verdict: ruled out** as an explanation of 1.00 against 0.49. Seed variance is a spread of about 0.08 around 0.48, and it follows how early each run started to use its memory.

## 3. Pico pilot against nano, side by side

Sources: the console logs (parameters), the configs, `log.jsonl` (steps, minutes, readings), the dev2 files, `runs/s6-sleep3/pilot.json`, `n.json`. Nano columns show seed 0, then seeds 1 and 2 where they differ. "Last 50" readings are pooled over the last 50 `recall` steps of the phase, on train batches before their update. Phase B readings mix N from 1 to 4.

| | pico pilot (M) | nano sleep arm |
|---|---|---|
| parameters (trained; control) | 473,665; 445,345 | 1,612,657; 1,486,225 |
| d, heads × head dimension, memory layers, loops | 96, 1 × 96, 1, 4 | 144, 2 × 72, 2, 2 |
| state per stream; memory parameters | 9,216 floats; 28,320 | 20,736 floats; 126,432 |
| peak rate, `mem.*` rate | 2.0e-3, 2.0e-2 | 1.63e-3, 1.63e-2 |
| auxiliary weight per memory layer | 1 | 0.5 (two layers, averaged) |
| phase A: working minutes; steps | 8; 663 | 12; 292 |
| phase A: sequence steps (of which `recall`) | 613 (526) | 267 (233); s1 255 (215); s2 263 (223) |
| phase B: working minutes; steps | 12; 2,051 | 18; 877 |
| phase B: sequence steps (of which `recall`) | 1,028 (407) | 437 (160); s1 451 (186); s2 428 (174) |
| seconds per sequence step (phase A; B) | about 0.77; 0.60–0.85 | about 2.6; 2.26–2.65 |
| sequences drawn, A + B (of which `recall`) | 9,808 + 16,448 (8,416 + 6,512) | 4,272 + 6,992 (3,728 + 2,560) |
| answer left chance (`recall` step; step) | 139; 167 | 161; 200 (s1 99; 146, s2 171; 224) |
| end of A, last 50: `ask_ce`, `ask_acc` | 0.0014, 0.9994 | 1.2663, 0.4084 (s1 1.1685, 0.4781; s2 1.3219, 0.3525) |
| end of A, last 50: `aux_ce`, `aux_acc` | 0.0000, 1.0000 | 0.9088, 0.6018 (s1 1.0708, 0.4977; s2 1.1877, 0.4385) |
| end of B, last 50: `ask_ce`, `ask_acc` | 0.0069, 0.9983 | 1.2888, 0.4152 (s1 1.1846, 0.4866; s2 1.2581, 0.4309) |
| end of B, last 50: `aux_ce`, `aux_acc` | 0.0045, 0.9984 | 1.0084, 0.5696 (s1 1.1313, 0.5157; s2 1.1733, 0.4698) |
| dev2 `recall`, N = 1, 2, 4 | 1.0000, 1.0000, 0.9980 | 0.4728, 0.4835, 0.4765 (s1 0.5275, 0.5245, 0.5160; s2 0.4467, 0.4652, 0.4645) |
| dev2 by delay at N = 1 (1–4, 5–8, 9–16, 17–32) | 1.000 at every bucket | 0.568, 0.534, 0.455, 0.334 (s1 0.687, 0.589, 0.497, 0.337; s2 0.556, 0.504, 0.424, 0.303) |
| dev2 memory zeroed; control | 0.2567; 0.2542 | not read; 0.2542, 0.2567, 0.2522 |

## 4. What a next experiment should change, and what it should measure first on the CPU

The items are ordered by the strength of the evidence above. Each comes with the cheap check that would confirm it before any GPU run. No experiment was run for this file.

1. **Budget phase A in `recall` steps, not in minutes, and end it on an in-sample reading with a cap.** For example: go on until `ask_acc` is at least 0.95 over 50 `recall` steps, up to a fixed maximum. *CPU first:*
   - train C5's setting at the nano size: the nano base and config, T1 and T2, train sequences only, `--memory-off-eval 4`;
   - run it for 1,000 `recall` steps on two seeds;
   - read the curve in windows of 50: `ask_acc`, `aux_acc`, and the excess ratio between 17–32 and 1–4.

   This says whether the nano memory reaches C5's bars at all, and in how many steps. It is the check v3 never made at this size. If it does not reach them in 1,000 steps, more GPU minutes will not help, and item 2 comes first.
2. **Find what slows the learning per step at the nano size, one difference at a time.** Same CPU setting as item 1, 400 to 600 `recall` steps each, two seeds:
   - (i) the auxiliary loss summed over the layers, weight 1 per layer, instead of averaged;
   - (ii) the auxiliary loss on the last layer only, as T2's text says;
   - (iii) the pilot's rates, 2e-3 with `mem.*` at 2e-2.

   Measure the `recall` steps until `aux_acc` passes 0.8 and until `ask_acc` passes 0.9, against item 1's run. Each run needs a per-layer `aux_acc`: the trainer would have to log the two layers apart. Today it logs their mean.
3. **Measure the overlap of the keys at the nano size before blaming it.** Two readings:
   - C2's mean squared cosine between a shown door's key and an idle frame's keys at initialisation, on the nano base (`s2-memcheck --c2-only`), against the pico's 0.73;
   - the key overlap of `--memory-trace` during item 1's run.

   If the overlap stays well above the pico's while the long delays lag (excess ratio under 0.4 at an overall accuracy around 0.5), the interference reading of (d) holds. Then a write that can replace becomes the change to test: the slot form of `docs/s6-sleep3-diagnosis.md` §6.2, written and never run. Its first check is C2's reading at initialisation at the nano size.
4. **Soften the change of phase.** Options: a lower peak rate in phase B, a longer warm-up, or the optimiser state carried over. *CPU first:* the step-in-parts test of amendment 3 shows how to replay a phase change cheaply. Run phase A's last checkpoint from item 1 into 100 steps of phase B's mix. Check that the first 50 `recall` steps do not fall below the end of phase A. Today they fall by 0.11 (pilot) and 0.03 to 0.12 (nano).
5. **Measure the step time and the memory of a nano sequence step before the budget is fixed.** That means `--sleep-chunk` at 2, 4 and 8 with `SKY_MEM_TRACE=1`. The peak memory can be read on the CPU, as amendment 3 did; the step time is the GPU's. This matters only if item 1 keeps a minute budget. With a step budget, the time becomes a cost and stops being a variable of the result.
6. **Read sleep (H1) only on a memory that is trained.** Item 1 should end at the bars. Then train sleep passes on the CPU with `ask_splits` by N. At the nano size today, N = 4 helps short delays and hurts long ones, which is what extra writes to an interfering memory do. On a memory at 0.49 that says nothing about sleep.
7. **Read the memory zeroed on dev2 at the nano size too.** The pilot did. The nano runs never did, so H0 at the nano size rests on the control alone.

## 5. Not verified

- **Which configuration difference slows the learning per step.** The rate, the auxiliary weight per layer, two heads of 72, two loops, the base and the mix all change with the size, and none was varied alone.
- **The accuracy of each nano layer's auxiliary head.** `aux_acc` is the mean of the two layers (`train.rs` lines 2458 to 2461). A layer at 0.9 and another at 0.3 would read the same 0.6.
- **The key overlap at the nano size.** No nano run used `--memory-trace`. The "6% erased per frame" of (d) is a fit under an assumed proportional model, not a measured property of the keys.
- **Shows of other doors against idle frames as the erasing writes.** The probe files keep, for each ask, only its bucket, its outcome and the colours its door had been shown with. They keep no frame positions, so the two cannot be separated.
- **The number of state tokens per `recall` frame for each tokenizer.** An approximate byte-level BPE in Python gave 14 to 16 tokens at the nano size against 16 to 18 at the pico size. That count does not reproduce the trainer's pipeline (`fuse_numbers`, the state template) and disagrees with the diagnosis's "about nine". It is not used above.
- **The speed of an unchunked nano run.** It rests on one line of amendment 3 (step 203 at 5.8 working minutes, under memory pressure); that run's log was removed.
- **Whether more steps would bring the nano memory to 1.00.** The curves were still rising, and nothing in the files says where they would stop.
- **Any f32 reading of `recall`**, at either size.
- **The nano memory zeroed on dev2**, never read.
- **The effect of the gradient clip** at the nano size, where nearly every `recall` step of phase A was clipped. AdamW is close to insensitive to a constant scale; whether the clip matters here was not looked at.
- **The pilot's seed spread.** The pilot is one seed. The CPU runs at the pico size (C5 and the diagnosis) all reached at least 0.99 on train batches, but they are not dev readings.
- **The cause of the phase-B drop.** The fresh optimiser, the jump of the rate, and the new mix change together.
- **Nothing here reads a test-seed file.** The test numbers are quoted from `runs/s6-sleep3/REPORT.md`.
