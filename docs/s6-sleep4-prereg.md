# s6-sleep v4 (s6-sleep4-nano): pre-registration

**Frozen on 17/11/2026 by the lead, before any computation and before the code of §11 is written; every later change is a dated amendment that precedes the computation it concerns.** No line of v4's code is written, no v4 check has run, no v4 model exists. Every element below was checked against the code at the commit of the draft; what the code cannot do today is listed in §11, and nothing of §3 or §4 may run before it is written and tested. The values that had to be chosen are in §12, each with its reason and its alternative; the lead's decisions at the freeze are in §13.

**State on 19/11/2026 (evening).** K3 failed on both seeds on 18/11/2026 and the frozen route is closed (§15). By amendment 2 (§14), written on 19/11/2026 before K1 runs, v4 goes on to K1 and to K1 only: if K1 fails on either seed, v4 stops at K1 and that reading is its result; if K1 passes on both seeds, nothing further runs until the lead writes the next amendment. **K1 was read on 19/11/2026 and failed** (§16): `ask_acc` over the last 100 `recall` steps of its training batches 0.8442 on seed 0 and 1.0000 on seed 1, against 0.95 on both seeds. v4 stopped at K1, that reading is its result, and v4 is closed.

This file is short where it can be: **v4 is v3 (`docs/s6-sleep3-prereg.md`) trained to the end of its learning curve, with checks at the nano size first.** Everything v3, v2 (`docs/s6-sleep2-prereg.md`) and v1 (`docs/s6-sleep-prereg.md`) wrote stands unless this file changes it: the memory (v2 §4 with v3's T1 and T2), the probes (v1 §3), the readers (v2 §5.2), the hypotheses (v1 §7), the gate (v1 §8, v2 §7), the export and the parity of `sky-infer`. There is no pico pilot: v3's pilot stands as read (amendment 2 of v3). Names: `s6-sleep4-*`; outputs go to `runs/s6-sleep4/`.

The basis of this file is the diagnosis of v3, `docs/s6-sleep4-diagnosis.md`, made from the existing logs, and the note of 17/11/2026 at the end of `docs/s6-sleep3-prereg.md`. Every v3 number quoted here comes from them or from the files they cite.

## 1. The question

Can the nano memory reach the pico pilot's level when it is trained to the end of its learning curve, and can it do so without the games losing?

v3 showed that the nano memory is used: on the `recall` test, 0.4892 at N = 1 against 0.2440 for the control (H0, +0.2452 [+0.2370, +0.2532]). It also showed two things that v4 answers:

- **The runs ended halfway up their learning curve.** The minute budget bought 393 `recall` training steps at seed 0 (233 in phase A, 160 in phase B) against 933 for the pico pilot, and the answer's accuracy on the training batches was still rising by 0.03 to 0.04 per 25 `recall` steps when phase A ended (diagnosis §1, §2 (a), (b)). The pico pilot reached 1.0000 on dev2 at every delay; nano reached about half of that.
- **Two games lost against the control** (H2: Flappy 15.87 against 71.05, Dodge 5.89 against 7.94), and the control itself lost against its base at Flappy (71.05 on the test arena, where the base plays 200 pipes in the s6-nano gate's arena; the two readers differ, v3 Result block).

**Works**, at the nano size, means v1's H0 to H3 and the s6-nano gate on the exported seed-0 file, with their bars unchanged, plus the memory zeroed within 0.03 of the control (§6). **The pilot's level** is an announced goal, not a gate: `recall` at least 0.90 at N = 1 on the test.

## 2. What changes from v3, and why

| # | change | evidence (v3's files) |
|---|---|---|
| V1 | **Checks at the nano size, on training sequences, before any budgeted run** (stage 1, §3). | C5 ran at the pico size only (`scripts/train-sleep3.sh`, lines 250 to 285; note of 17/11, item 2). No check asked whether the nano memory trains within the steps its budget buys (diagnosis §1, item 7). |
| V2 | **Phase A ends by a rule on its own training batches, with a cap set by stage 1**, not after a number of working minutes (§4.2). | At the end of phase A, `ask_acc` was 0.41, 0.48 and 0.35 over the last 50 `recall` steps (seeds 0, 1, 2) and still rising (diagnosis §2 (a)); the pilot was at 0.9994. Minutes bought 0.44 times the pilot's steps (§2 (b)). |
| V3 | **Phase B is a number of steps** (2,000), not minutes (§4.3). | 877 nano steps against 2,051 in the pilot; nano spent most of its 160 to 186 `recall` steps of phase B winning back the drop at the change of phase (diagnosis §2 (c)). |
| V4 | **Four factors, each tried alone, may change the recipe by a rule** (K2: the auxiliary weight per layer, the auxiliary head as T2 was written, the pilot's rates, the clip). | The auxiliary loss is averaged over two layers at nano, 0.5 per layer (note of 17/11, item 1); the rate is 0.82 times the pilot's; 88% to 98% of nano `recall` steps of phase A were clipped at 1.0, against 46% in the pilot (note, item 5). None was varied alone (diagnosis §2 (e)). |
| V5 | **A frozen base, checked first** (K3); only if it fails, the four factors (K1, K2) and a preservation term (K5). | H2 failed, and the control lost at Flappy against its own base (v3 Result block). With the base frozen and the memory zeroed, the file computes its base's outputs exactly (§3.4). |
| V6 | **The memory zeroed is read at the nano size**, on dev and on the test, and gated. | Never read at nano; H0 rests on the control alone (note, item 6). |
| V7 | **New dev and test seeds** (§5). | v3 read dev2 for its choices and read its test seeds once. |

**Not changed, and why.**

- **The memory itself** (v2 §4; `memory_qk: shared`, `memory_value: frame`, the average write, the gates held at β = 0.5 and α = σ(6), T1): the diagnosis ruled out the decay as the cause of the delay curve and found the gradient alive throughout nano phase A (§2 (d)). It supports overwriting by later frames, about 6% per frame, without being able to measure its mechanism; the slot write (`memory_write: slot`, written for v3's diagnosis, never run on a base) is not used. v4 asks whether more steps close the gap, and only that; if they do not, the interference reading belongs to a later file.
- **The mixes, the batch sizes, N = 1 in phase A and N drawn in {1, 2, 3, 4} in phase B, QAT from step 0, `--sleep-chunk 2`, `SKY_MLX_CACHE_GB=1`, `mem.*` at ×10 with the gates at ×0** (v3, amendment 3).
- **The change of phase** (a fresh AdamW and schedule, `s2-decide --init`). §4.4 says why no check of a softer change (K4) is run.

## 3. Stage 1: checks on training sequences, at the nano size

### 3.1 The common setting

Every check of stage 1 is the command of v3's nano phase A (`runs/s6-sleep3/nano/sleep-s0/a/cmd.txt`): the base `runs/s6-nano-g3/decide/last`, the config `runs/s6-sleep3/nano/sleep3.json` (v3's memory with `memory_aux`), stateless records and weights as v3's nano runs, batch 128, `--qat-frac 1.0 --loop-dropout 0.25 --length-pool 16 --pad-to 32 --step-mem-gb 3`, AdamW (`--no-muon`) at `--adam-lr 0.00163`, `--seq-log`, `--sleep-records recall,rule110 --sleep-weights recall=0.85,rule110=0.15 --sleep-batch 16 --seq-frac 0.9 --sleep-passes 1`, `--mem-lr-mult 10 --mem-gate-lr-mult 0 --memory-aux-weight 1`, `--sleep-chunk 2`, with these differences:

- **a step budget**: `--steps 1310` (the schedule of every check; at 0.9 × 0.85 = 0.765 `recall` steps per step, 1,002 `recall` steps expected), and `--stop-after 654` for the shorter checks of K2 (about 500 `recall` steps under the same schedule);
- **`--memory-off-eval 4`**, as C5: one sequence step in four is also answered by the same weights without the memory, on the same batch, before its update (reported, deciding nothing);
- **the training seeds 0 and 1**, as C5;
- **the GPU**, one run at a time, never during another budgeted GPU run (§3.7).

Only training sequences are read: each step's own batch, before its update, from `log.jsonl`. No dev, validation or test seed of any probe or game is read in stage 1. The one exception is K5 (§3.5), which reads closed-loop Flappy on check seeds used nowhere else in the project. A run stopped by a technical failure (a crash, a full disk, the memory guard) restarts from scratch with the same seed; no check is run twice for any other reason.

By reading of `crates/sky-train/src/s2/train.rs` (lines 2377 to 2545), the random draws of a step are the choice of a sequence step, the batch, N, the stateless batch and the loop dropout, and none of them depends on the rate, the clip, the auxiliary loss or which parameters train: two checks with the same seed train on the same batches. The report verifies it by comparing the `seq_game` and `passes` of their log lines.

**Readings**, per window of 50 `recall` steps (a `recall` step is a log line with `seq_game` `recall` and asks; they are counted from 1 in each run), each pooled over the asks of the window:

- the answer's accuracy at the last exit, `ask_acc`, and the ask-frame cross-entropy `ask_ce`;
- the auxiliary accuracy `aux_acc` (today the mean over the two layers; each layer apart once §11 item 2 is written);
- **the excess ratio**: (accuracy at delays 17 to 32 − 0.25) / (accuracy at delays 1 to 4 − 0.25), from `ask_splits` (reported; not defined while the 1 to 4 bucket is within 0.02 of chance);
- the same weights with the memory off (`off`), and `grad_norm` (median per window);
- **S90**: the first `recall` step n ≥ 50 at which `ask_acc` pooled over `recall` steps n − 49 to n is at least 0.90 (none if never);
- **S95**: the first `recall` step n ≥ 100 at which `ask_acc` pooled over `recall` steps n − 99 to n is at least 0.95 (none if never).

**K1's bar** (used by every check that needs one): `ask_acc` pooled over the last 100 `recall` steps of the run is at least 0.95, **on both seeds**. A pass on one seed only is a failure. A run that meets the bar has an S95, since its last 100 `recall` steps qualify.

### 3.2 K3, a frozen base, first

- **What.** v3's recipe unchanged (K3 runs before K1 and K2, so nothing else in it changes), with the base frozen: every parameter that exists in s6-nano (every name not starting with `mem.`) keeps its value, and only the memory (`mem.{l}.norm`, `qk`, `v`, `out`, with the gates held as in v3) and the auxiliary heads (`mem.{l}.aux`) train. `--steps 1310`, seeds 0 and 1. Needs code (§11 item 1).
- **The model equals s6-nano's checkpoint at step 0: this already holds.** The read-out `mem.{l}.out.weight` is initialised at zero (`crates/sky-train/src/s2/memory.rs`, lines 116 to 117), its ternary form under QAT is zero too (`ternarize` in `crates/sky-train/src/s2/model.rs`, lines 378 to 381, floors the scale at 1e-8), and the read adds `r·Wo` to the block output (`memory.rs`, line 298). The test `a_zeroed_read_leaves_the_logits_of_the_base` (`memory.rs`, lines 2220 to 2242) checks it to 1e-6, in f32 and without QAT; a bit-exact test under QAT at the nano size is §11 item 1's.
- **Why first.** With the base frozen and the memory zeroed (S = 0, so r = S·q = 0), the file computes its base's outputs whatever `Wo` has become: the language tasks and the games cannot lose when the memory is empty, by construction, and what remains to measure is what a written memory does to the games, which is H2. If the memory trains this way, K1, K2 and K5 have nothing left to decide, and their cost is saved.
- **Bar:** K1's bar.
- **Reads** the windows of §3.1, S90 and S95.

### 3.3 K1, more steps (only if K3 fails)

- **What.** v3's recipe unchanged, everything trained, phase A alone, `--steps 1310`, seed 0 then seed 1.
- **Reads** the windows of §3.1, S90 and S95.
- **Bar:** K1's bar.
- **What follows.** K1 sets the reference S90 of K2, and its seed-0 run fixes the clip of K2 (iv). If it passes and K2 adopts nothing, R\* (§3.4) is v3's recipe.

### 3.4 K2, what slows it (only if K3 fails, after K1)

- **What.** Seed 0, `--steps 1310 --stop-after 654`, one factor changed at a time against K1's seed 0, on the same batches:
  - **(i) the auxiliary loss summed over the layers**, weight 1 per layer, the pilot's effective weight: `--memory-aux-weight 2`. The code divides the auxiliary term by asks × memory layers (`memory.rs`, lines 962 to 976), so weight 2 with two layers is weight 1 per layer. Possible today.
  - **(ii) the auxiliary loss on the last memory layer only, read at the door token of the ask frame, weight 1**, as T2 was written. Needs code (§11 item 3).
  - **(iii) the pilot's rates**: `--adam-lr 0.002`, so `mem.*` at 0.02 with `--mem-lr-mult 10`. Possible today.
  - **(iv) the clip at the measured norm**: `--clip-norm c`, with c the median of `grad_norm` over `recall` steps 51 to 250 of K1's seed 0, rounded up to one decimal. Possible today, once K1's seed 0 has ended. **Dropped alternative: the median over the first 50 steps.** In v3 the median over the first 50 nano `recall` steps was 0.92, 1.51 and 2.27 (seeds 0, 2, 1) and the norm rose to 6 to 11 from `recall` step 75 on (note of 17/11 in `docs/s6-sleep3-prereg.md`, item 5): a clip at the first-50 median would sit near the default 1.0 and test nothing.
- **Rule.** A factor is **adopted** if its S90 is at most 0.75 times the S90 of K1's seed 0. A factor that does not reach 0.90 within its run is not adopted. If K1's seed 0 never reaches 0.90, its S90 counts as 1,001 for this rule.
- **(i) and (ii) exclude each other.** If both are adopted, the one with the lower S90 is kept; on a tie, (ii), the wording that was pre-registered.
- **The confirmation.** If at least one factor is adopted, the adopted factors together are run as K1 is run (`--steps 1310`, seeds 0 and 1). It must pass K1's bar.
- **The recipe R\*.** If the confirmation ran and passed, R\* is v3's recipe with the adopted factors; otherwise R\* is v3's recipe. **"R\* passed K1's bar"** means: R\* is v3's recipe and K1 passed, or R\* carries the factors and the confirmation passed.
- **No frozen base again.** Once K3 has failed, R\* is not run with a frozen base: the frozen route is closed for v4.

### 3.5 K5, the preservation term (only if K3 fails and R\* passed K1's bar)

- **The term.** On every stateless batch (all the records: decisions, distill, games, numeric) and on every frame of the game sequences of phase B (`flappy`, `rescue`, `dodge`, `doom`; not `recall`, not `rule110`), the loss adds λ · KL(p_base ‖ p_θ) at the last exit, temperature 1, mean over the rows and the frames, with **λ = 1.0**. p_base is the base checkpoint `runs/s6-nano-g3/decide/last` answering each frame alone with its ternary weights (as under QAT), without a gradient; p_θ is the model being trained, with its memory in the sequences. **Both arms carry the same term**, the control too, so that H2 still compares the memory with the same training without it. Needs code (§11 item 6).
- **What.** From the final checkpoint of R\*'s seed-0 run in stage 1 (K1's or the confirmation's), 1,000 steps of phase B's mix (§4.3), seed 0, twice: with the term (K5a) and without it (K5b). The same batches.
- **Bar, both conditions on K5a:**
  1. **Flappy's training loss does not rise**: the mean `seq_step_loss` of the last 20 `flappy` steps is at most that of the first 20;
  2. **closed-loop Flappy on the check seeds is at least 0.98 times the base's**: `sky-train s2-probe <checkpoint> --game flappy --from-seed 8000 --episodes 50 --passes 1`, for K5a's end and for the base.
- **The check seeds.** Flappy's courses from seed 8000 are a check on seeds used nowhere else: not v3's arena (5000), v4's test arena (6000), the s6-nano gate's (`sky-arena`'s default, 7), the trajectories' (2026, 101) or the smoke's (9000). They are read for K5 only, for K5a's end, the base, and, as reported, K5b's end and the start checkpoint.
- **Why the second condition.** In v3, the training loss on the game sequences did not separate the arms: over the last 20 `flappy` steps of phase B, the memory arm read 0.384, 0.378 and 0.372 and the control 0.380, 0.377 and 0.371 (seeds 0, 1, 2; `runs/s6-sleep3/nano/*/b/log.jsonl`), while the closed-loop test read 15.87 against 71.05. Condition 1 alone would have passed v3.

### 3.6 The decision rules of stage 1, with no choice left

1. **K3 runs first**, on seeds 0 and 1. **If K3 passes K1's bar on both seeds, v4 takes the frozen route**: stage 2 trains v3's recipe with the base frozen, the control is the base file (§4.1), the cap of phase A comes from K3's two S95 (§4.2), and K1, K2, the confirmation and K5 are not run.
2. **If K3 fails on either seed** (a pass on one seed only is a failure), the frozen route is closed for v4. K1 runs (seed 0, then seed 1), then K2 (i) to (iv) on seed 0, then the confirmation if at least one factor is adopted; R\* follows from §3.4.
3. **If R\* passed K1's bar**, K5 runs from the final checkpoint of R\*'s seed-0 run. **If K5a meets both its conditions, v4 takes the preserved route**: stage 2 trains R\* with every parameter trained and the term of §3.5 in both arms, and the cap of phase A comes from the two S95 of R\*'s stage-1 runs (§4.2).
4. **If R\* did not pass K1's bar, or K5a fails either condition, v4 stops** after stage 1 and reports it; no stage-2 run takes place.

Each check's reading and verdict are committed before the next check starts.

### 3.7 Device, duration, memory

**Device.** Stage 1 runs **on the GPU**, in steps, one run at a time, never during another budgeted GPU run (the lead's decision at the freeze). The trainer runs on the GPU unless `--cpu` is given (`crates/sky-train/src/main.rs`, lines 28 to 31 and 722 to 724: MLX's default device becomes the CPU); `s2-memcheck` (`memcheck.rs`, line 616) and `s2-probe` (`probe.rs`, line 328) always run on the CPU, so K5's Flappy probes do. Every check is in steps, so the device is a cost and not a variable of the result, and stage 2 runs on the GPU too, so the S values that set its cap come from the same device. The CPU at `nice -n 19`, the practice of v2's and v3's checks, is the alternative not taken (§12, C1).

**Step times** (working seconds, `log.jsonl`, windows of 50 sequence steps, stateless steps included):

| | GPU, measured (v3) | CPU, for the record |
|---|---:|---:|
| nano memory arm, phase A | 2.47 s per step (292 steps in 12.03 min) | about 21 s per step (23 s per sequence step), not measured |
| nano memory arm, phase B | 1.23 s per step (877 in 18.02 min) | about 12 s per step, not measured |
| nano control, phase A / B | 1.73 / 0.70 s per step | |
| pico, 8 sequences, C5 (`--no-compile`, `--memory-off-eval 4`) | | 3.1 to 3.9 s per sequence step, measured |

The CPU figures scale C5's measurement by the GPU's ratio between the sizes (a pico sequence step of 16 on the GPU takes 0.77 s against about 3.5 s for 8 on the CPU, a ratio near 9).

**Duration of each check** (GPU, working minutes): K3 45 per seed if §11 item 1 skips the base's weight gradients (about 0.8 times a full step, not measured), 54 if it does not; K1 54 per seed; K2 27 per factor; the confirmation 54 per seed; K5 about 45 for its two runs of 1,000 phase-B steps, plus three Flappy probes of 50 episodes on the CPU (not measured).

**Cost of stage 1 by path** (GPU, working minutes):

| path | checks run | GPU working minutes |
|---|---|---:|
| K3 passes: the frozen route | K3 × 2 | 90 to 108 (1.5 to 1.8 hours) |
| K3 fails, K2 adopts no factor (no confirmation), then K5 | K3 × 2, K1 × 2, K2 × 4, K5 | 351 to 369 (5.9 to 6.2 hours) |
| K3 fails, a factor adopted, the confirmation, then K5: **the maximum** | K3 × 2, K1 × 2, K2 × 4, confirmation × 2, K5 | 459 to 477 (7.7 to 8.0 hours) |
| K3 fails and v4 stops before K5 | K3, K1, K2 (and the confirmation) | 306 to 432 (5.1 to 7.2 hours) |

**Expected:** 1.5 to 1.8 hours if K3 passes; 5.9 to 8.0 hours if it fails and v4 goes on to K5. Under the old order (K1 and K2 first, then K3) every path cost at least 5.1 hours.

**Memory.** On the CPU, a nano sequence step of 16 in parts of 2 peaks at 1.7 GB of MLX memory at N = 1 (commit `ce20cb7`, measured on the CPU at the nano size), next to about 5.1 GB of loaded stateless records (v3 amendment 3): about 7 GB for the process, which bounds K5's CPU probes too. On the GPU, the six v3 nano runs with `--sleep-chunk 2` ran without a stop by the memory guard (limit 13 GB for the largest `sky-train` process, `scripts/auto-brake.sh`, line 39); amendment 3 estimated 7.5 to 9 GB. K3 needs no more than K1; K5 adds the base's parameters (6.5 MB) and a forward pass without gradient.

## 4. Stage 2: the nano runs, only if stage 1 allows

### 4.1 Arms

- **The memory arm**, seeds 0, 1 and 2: on the frozen route, v3's recipe with the base frozen; on the preserved route, R\* with every parameter trained and the term of §3.5.
- **The control on the frozen route is the base file.** A model with a frozen base and no memory has nothing to train, so the control is the base checkpoint itself, `runs/s6-nano-g3/decide/last` (and, for the gate, the file exported from it): no control run takes place, it is read once on each seed set and paired with each of the three memory seeds. Two of its readings are then a **parity check, not a hypothesis**: the memory model with its memory zeroed (`recall`, `--memory-off`) and the memory model on the language reading with an empty memory (`typed_decisions_bench`, each question a frame alone) must equal the base's readings exactly, answer for answer; a difference is a bug, and it stops the run until it is explained. H2 compares the memory model, its memory written as it plays, against the base.
- **The control on the preserved route** is the same training without memory (no auxiliary loss) with **the same preservation term**, seeds 0, 1 and 2, each seed training its memory seed's steps, phase by phase.

### 4.2 Phase A, ended by a rule

- **The rule.** Phase A ends at the first `recall` step at which `ask_acc` pooled over the last 100 `recall` steps is at least **0.98**. The trainer saves `last`, writes its final line and stops (§11 item 5).
- **The cap.** S95max is the largest S95 of the two stage-1 runs of the recipe stage 2 trains: on the frozen route, K3's own two seeds; on the preserved route, the two runs of R\* (K1's, or the confirmation's). The cap is 1.5 × S95max `recall` steps, converted to the run's budget as `--steps` = ⌈1.5 × S95max / 0.765⌉; the schedule (`wsd`, `crates/sky-train/src/optim.rs`, lines 177 to 186) is laid over that budget, so a run ended by the rule ends on the plateau of its rate.
- **At the cap.** Phase A ends there, phase B runs as for any seed, and nothing is retried or extended. The report gives that seed's pooled `ask_acc` over its last 100 `recall` steps and says that it ended at the cap. v2's stop (ask-frame cross-entropy above 1.336 over the last 50 `recall` steps of phase A) still applies to every memory seed: if it holds for one, no phase B runs and v4 reports a failure in-sample.
- **The control of the preserved route** trains its memory seed's phase-A steps (`--steps`, the same number), on the same batches.

### 4.3 Phase B, in steps

- **2,000 steps** for every run (`--steps 2000`), v3's mix (sequence batches at 0.5: `recall` 0.40, `rule110` 0.12, each game 0.12; N drawn in {1, 2, 3, 4}, truncated), from phase A's `last` with a fresh optimiser and schedule, T1 and T2 as in phase A.
- **Why 2,000.** The pilot's phase B had 2,051 steps and 407 `recall` steps; 2,000 steps give 400 `recall` steps expected (2,000 × 0.5 × 0.40). In v3 the first 50 to 100 `recall` steps of phase B went to winning back the drop at the change of phase, at both sizes (diagnosis §2 (c)); at 400 that is a quarter at most.
- **Working minutes** (GPU, v3's step times): the memory arm 2,000 × 1.23 s ≈ 41 minutes per seed; the preserved route's control 2,000 × 0.70 s ≈ 23 minutes per seed (the term adds a forward pass of the base, not measured).

### 4.4 The change of phase: left as it is (no K4)

The change from phase A to phase B is left as v3's. No check of a softer change is run, for three reasons. Both sizes recovered from the drop within 50 to 100 `recall` steps and phase B lowered no run for good (diagnosis §2 (c)). One of the drop's three parts was the jump of the rate from 0.1 of the peak at the end of phase A's decay back to the peak; a phase A ended by the rule ends on its plateau, at the peak rate, so that jump is gone. And a third change would need its own check, run and read before stage 2. **Reported, deciding nothing:** `ask_acc` over the first 25 `recall` steps of phase B against the last 50 of phase A, per seed.

## 5. Readings and seeds

- **dev3, for the choices**: the first 500 episode seeds ≡ 19 (mod 20) from **3,000,000,000** upward. v1's dev seeds (from 1,000,000,000) and v3's dev2 seeds (from 2,000,000,000) were read for the choices of v1 and v3 (v3's pilot rule and its choice of N), and dev2 again by the diagnosis; v4 reads neither. On dev3: `recall` and `rule110` at N = 1, 2 and 4 for each memory seed, the control once per seed (the base once on the frozen route), the memory zeroed (`--memory-off`) at N = 1 on `recall` for each memory seed. **The choice of N** is v1 §6's rule on dev3 (the smallest N in {1, 2, 4} within 0.01 of the best mean of the three memory seeds). Reported, deciding nothing: the splits by delay and by one colour against several, with the memory and zeroed.
- **The test, read once**, after every stage-2 run has ended and N is chosen:
  - `recall` and `rule110`: the first 1,000 episode seeds ≡ 19 (mod 20) from **6,000,000,000** upward, at N = 1, 2 and 4 for each memory seed, the control once, and the memory zeroed at N = 1 on `recall` for each memory seed;
  - the arena of H2: the four games, `s2-probe --from-seed 6000 --episodes 200`, at the chosen N, through the same reader for both arms;
  - `typed_decisions_bench` with a zero memory (`s2-eval`), as v3.
- **Seeds that v4 never reads for any decision**: v3's test seeds (`recall` and `rule110` from 5,000,000,000, the arena from 5000), dev2 (2,000,000,000), v1's dev (1,000,000,000), the smoke's (1000, 3000, 9000; `train-sleep3.sh`, line 103), the browser's validation seeds of the Memory page (≡ 18 mod 20 from 7,000,000,018; `site/.vitepress/theme/lib/memory.ts`, line 29), and K5's check seeds (from 8000) outside K5.
- **The s6-nano gate's own arena** (`sky-arena`, its default seed 7, 30 episodes per game, against s3-pico) is the fixed protocol of that gate for every model and stays as it is, with its bar; it has been read by every gate of the project, v3's included.
- **How the report script must change** (§11 item 8): `scripts/s6-sleep3-report.py` fixes `TEST_SEED, TEST_EPISODES = 5_000_000_000, 1000` and `ARENA_SEED = 5000` (lines 66 and 67) and checks only the first seed of the `recall` readings (`check_test`, lines 414 to 418); its dev2 check comes from `scripts/s6-sleep2-report.py` (`DEV2_SEED`, line 33); `scripts/train-sleep3.sh` fixes `dev_seed`, `test_seed` and `arena_seed` (lines 90 and 91). v4's reader takes dev3, the test from 6,000,000,000 and the arena from 6000, checks `from_seed` and `n_episodes` of every reading (`recall`, `rule110` and each game), refuses any reading whose seeds are v3's, reads the zeroed `recall` files as required rows, and, on the frozen route, pairs each memory seed with the one reading of the base.

## 6. Hypotheses and gate

Every reading is on the test seeds of §5, ternary weights, τ = 1 (last exit), the mean of the three memory seeds; intervals are paired bootstraps over episodes (10,000 draws), the seeds pooled. The bars are v1's and v2's, unchanged.

- **H0, the memory is used:** `recall` at N = 1 minus the control ≥ 0.10.
- **H1, sleep helps:** `recall` at N = 4 minus N = 1, the lower bound of the 95% interval above 0. H1 stays a hypothesis read on a trained memory: v4 reads it only on runs whose phase A ended at 0.98 on their training batches, or at the cap (reported as such).
- **H1b** (secondary, no gate): the same on `rule110`.
- **H2, the games do not lose:** on the arena of §5 at the chosen N, no game more than 2% under the control. On the frozen route the control is the base file, and H2 compares the memory model, its memory written as it plays, against the base.
- **H3, the language tasks do not lose:** `typed_decisions_bench` at most 0.01 under the control, read by `s2-eval` as in v3. On the frozen route H3 also compares the memory model against the base; each question of the bench is a frame alone, so the memory it reads is the empty memory it starts with, and the reading must equal the base's exactly (the parity check of §4.1): a difference is a bug that stops the run.
- **Z, the memory zeroed** (new, the pilot's rule of v2 §5.3 (2)): `recall` at N = 1 with the memory zeroed minus the control ≤ 0.03, on the preserved route. On the frozen route Z is the parity check of §4.1: the zeroed reading must equal the base's exactly, answer for answer, and a difference is a bug that stops the run.
- **The goal, announced and not gated:** `recall` at N = 1 at least 0.90.

**Gate** (v1 §8, v2 §7, plus Z). `s6-sleep4-nano` (seed 0, the chosen N) is published under that name only if all of these hold: the s6-nano gate on the exported seed-0 file (`typed_decisions_bench` above 0.398, ECE ≤ 0.08, no game more than 5% under s3-pico in the gate's arena); H0, H1, H2, H3 and Z (on the frozen route, the two parity checks); `sky-infer`'s parity with the training code on the golden memory sequences, natively and in WASM. On the frozen route, the exported file is the base's weights plus the memory; its calibration is computed again by `s2-calibrate` with a zero memory, as in v3.

## 7. What each outcome means

- **Stage 1 stops: K3 fails and R\* does not pass K1's bar.** On its own training batches, within 1,000 `recall` steps, the nano memory does not reach 0.95 on both seeds with a frozen base, nor trained everything as v3 or with the adopted change. v3's gap to the pilot is then not a matter of budget alone. No stage-2 GPU minute is spent. The next step, by a separate decision and file, is the interference reading of the diagnosis (§2 (d)): the key overlap at the nano size (`s2-memcheck --c2-only`, `--memory-trace`), then a write that can replace (the slot form).
- **K3 fails, R\* passes, K5 fails.** The nano memory trains, but only with a base that moves, and the term tried does not keep Flappy. v4 stops; the choice between the memory and the games is the result.
- **The frozen route, H0 and the goal hold, H2 holds, H1 fails.** The memory reaches the pilot's level at the nano size and leaves the games alone, and sleep does not help. As v1 §8 says, nothing is published as a model that sleeps; the memory becomes available as a plain recurrent memory, without sleep, by a separate decision.
- **The frozen route, H2 fails.** The base is intact, so the loss comes from what the memory writes and reads during a game: a memory that reads on every frame of every game. A memory gated by the game, or read only where a question needs it, would be a later file.
- **H0 holds and the goal is missed.** The memory fits its training episodes (0.98 on its batches) and holds less on unseen ones: 8,950 training episodes of `recall`, a generalisation gap, measured by the difference between the in-sample 0.98 and the test.
- **Z fails** (the preserved route). Part of the gain is not the memory but a better prior of the memory arm; it is a defect to explain in an amendment before anything else is reported, as v2 §5.3 says of the pilot.
- **H1 holds and everything else holds.** A model that sleeps, published as `s6-sleep4-nano`.
- **In every case** the report is published, failures included, and a file that fails its gate is published as a result under the owner's decision of 15/11/2026.

## 8. Disclosure

v4 is designed after v3's test was read: every number of `runs/s6-sleep3/REPORT.md`, `verdict.json` and `gate.json` was seen, as were v3's training logs, its dev2 files and its configs, by the diagnosis (`docs/s6-sleep4-diagnosis.md`). Its changes, and where each comes from:

1. **Checks at the nano size before the runs** (V1): from the code of v3's script (C5 at pico only).
2. **Phase A ended by an in-sample rule with a cap, phase B in steps** (V2, V3): from the training logs (the curves still rising at the end of each phase; the steps that the minutes bought).
3. **The four factors of K2** (V4): from the code (the auxiliary loss averaged over layers, the head on the mean over tokens), the configs (the rate) and the training logs (the share of clipped steps, the norms).
4. **The frozen base and the preservation term** (V5): **from a test reading.** H2's failure on the test arena (seeds from 5000) and the s6-nano gate's arena are what motivate them; v4 reads neither of v3's arenas for any decision, and its H2 is read on new seeds.
5. **The memory zeroed, read and gated at the nano size** (V6): from the files (it was never read).
6. **New dev and test seeds** (V7). **The s6-nano gate's arena is the exception**: `sky-arena`'s default seed 7, 30 episodes per game, against s3-pico, kept with its bar unchanged as that gate's fixed protocol; it has already been read by every earlier gate of the project, v3's included.
7. **The bars.** K1's bar (0.95) and the rule of phase A (0.98) were set after the pilot's in-sample 0.9994 and v3's nano curves (0.35 to 0.48) were seen; the goal of 0.90 after v3's test (0.4892) and the pilot's dev2 (1.0000).

Every report of v4 states this section. The disclosure of v3 §5 (and v2 §10) stands.

## 9. Cost

One MacBook (Apple M5, 16 GB); `scripts/auto-brake.sh` stops a training whose process holds more than 13 GB four readings in a row. Nothing else trains on the GPU during a stage-1 check or a stage-2 run.

- **Stage 1, GPU** (§3.7): 1.5 to 1.8 working hours if K3 passes (the frozen route); 5.9 to 6.2 if K3 fails, K2 adopts nothing and K5 runs; **at most 7.7 to 8.0** if K3 fails, a factor is adopted and the confirmation and K5 run; 5.1 to 7.2 if v4 stops before K5. A CPU stage 1 would have taken about 65 working hours (not measured at this size), six to seven days of wall clock under the brake; it is not used.
- **Stage 2, GPU** (v3's step times): phase A at 2.47 s per step for the memory arm, so 1,000 `recall` steps (1,307 steps) take about 54 working minutes and the cap 1.5 times S95max of that rate; phase B 41 minutes. For S95max = 700 `recall` steps (cap 1,050, 1,373 steps, 57 minutes), the memory arm takes at most about 1 hour 40 per seed, **4 hours 50 for three seeds**.
  - **The preserved route** adds the three controls, at most 3 hours 10 (1.73 and 0.70 s per step), and the term's forward pass of the base at every step (not measured): **at most about 8 hours**.
  - **The frozen route has no control run** (the base is read, not trained): **at most 4 hours 50**, and about 3 hours 55 if §11 item 1 skips the base's weight gradients (about 0.8 times a full step, not measured).

| route | stage 1 | stage 2 (S95max = 700 `recall` steps) | GPU, in all |
|---|---:|---:|---:|
| the frozen route (K3 passes) | 1.5 to 1.8 h | 3.9 to 4.8 h | **5.4 to 6.6 h** |
| the preserved route (K3 fails, K5 passes) | 5.9 to 8.0 h | at most about 8 h, plus the term | **at most about 16 h** |
| v4 stops after stage 1 | 5.1 to 8.0 h | none | **5.1 to 8.0 h** |

S95max is known only after stage 1; stage 2 scales with it (each 100 `recall` steps of S95max add about 8 working minutes per memory seed and 5.5 per control seed).
- **Readings, CPU:** v3's export, test reading and gate took about 11 hours of wall clock for six models (`runs/all/sleep3-gate.log`, 07:39 to 18:54 on 17/11); v4 adds the zeroed readings and the dev3 readings: about 13 to 15 hours on the preserved route, less on the frozen route (three trained files and the base, read once).
- **Disk:** about 25 MB per run of one phase (`last`, `pre-qat`, the log); stage 1 at most 12 runs, stage 2 at most 12, the probes about 10 MB, six GGUF files about 4.5 MB: under 1 GB. The disk had 12 GiB free on 17/11/2026.

## 10. Forbidden

- Any v4 computation before this file is frozen and the code of §11 is written and tested.
- Reading a dev3 seed before stage 1 has ended and its outcome is committed; reading a test seed before every stage-2 run has ended and N is chosen; reading any seed of v3, v1, dev2, the smoke or the browser (§5) for a v4 decision.
- Running K1, K2, the confirmation or K5 after K3 has passed on both seeds; running a frozen base again after K3 has failed; running any stage-1 check on the CPU, two at a time, or during another budgeted GPU run.
- A check beyond K3, K1, K2 (i) to (iv), the confirmation and K5; a factor beyond the four; a second confirmation; running a check again except after a technical failure, which restarts the same run with the same seed.
- Choosing R\*, the route or the cap by anything but the rules of §3 and §4.2.
- Extending phase A beyond the cap, retrying a seed, or adding a seed.
- Training on episodes of the validation or test splits (`split_for_seed`).
- Changing a bar of §3, §4.2 or §6 after a reading it concerns.
- Publishing under `s6-sleep4-nano` a file that fails the gate.

**What a failure would mean for each bar.** K1's bar: v3's recipe does not reach 0.95 on its own batches within about 1,000 `recall` steps at the nano size. K2's rule: no single factor of the four explains a quarter of the slowness. The confirmation: the adopted factors do not add up. K3's bar (on both seeds): the memory cannot reach 0.95 on its batches within about 1,000 `recall` steps without the base moving. K5: the term does not keep Flappy where the base plays it. The rule of phase A ending at the cap: the run did not reach 0.98 on its batches in 1.5 times stage 1's longest time to 0.95. H0 to H3, Z and the s6-nano gate: as v1 §7, §8 and v2 §5.3 say.

## 11. Code to write before any run

| # | file | what | needed by |
|---|---|---|---|
| 1 | `crates/sky-train/src/optim.rs`, `crates/sky-train/src/main.rs`, `crates/sky-train/src/s2/train.rs` | **A frozen base**: an `OptConfig` field (e.g. `base_lr_mult`, 1 by default and then not written, as `mem_lr_mult`) and its flag (`--freeze-base`); a frozen parameter gets no update and no weight decay, and the global clip is computed over the trained parameters only (today `Optimizer::update` sums every gradient into the norm, lines 271 to 285, so the base's gradients would still shrink the memory's update). Best, `train.rs` differentiates the memory's parameters only, so that the base's weight gradients are not computed. Tests: after some steps every non-`mem.*` tensor is bit-identical to its start; a fresh memory model under QAT at the nano config gives its base's logits bit for bit. | K3, which runs first; the frozen route |
| 2 | `crates/sky-train/src/s2/memory.rs`, `crates/sky-train/src/s2/train.rs` | **The auxiliary readings per layer**: `seq_frames` (lines 932 to 939) sums `aux_ce` and `aux_right` over the layers; return them per layer and log `aux_acc_layers` next to the mean (lines 2458 to 2461). | the readings of §3.1; K2 (ii) |
| 3 | `crates/sky-train/src/s2/config.rs`, `crates/sky-train/src/s2/seq.rs`, `crates/sky-train/src/s2/memory.rs`, `crates/sky-train/src/main.rs` | **The auxiliary loss as T2 was written**: a choice of layers (all, or the last) and of where the head reads (the mean over the valid tokens, or the door token of the ask frame). `seq.rs` carries no token position today (`SeqData`, lines 128 to 141, and the batch's `Asks`); the door token is found as `memcheck.rs` finds it (`Encoder::door`, lines 313 to 335: the one token that changes when the door does) and must be carried per ask frame through `FrameBatch` and its parts (`--sleep-chunk`). The normalisation (`memory.rs`, lines 962 to 976) divides by the layers that carry a head, and so does the log. | K2 (ii) |
| 4 | `crates/sky-train/src/optim.rs` | Not required: a clip of the memory's parameters and of the rest by their own norms, the alternative of K2 (iv) (§12, C6). | not used: the lead kept the value of C6 |
| 5 | `crates/sky-train/src/s2/train.rs`, `crates/sky-train/src/main.rs` | **The stop rule of phase A**: `--stop-recall-acc 0.98 --stop-recall-window 100` ends the run (save `last`, the final validation, a final line saying `stopped_by_rule`) at the first `recall` step where the pooled `ask_acc` of the window reaches the bar; the loop's own end is `keep_going` (`crates/sky-train/src/session.rs`, lines 98 to 100). | stage 2 |
| 6 | `crates/sky-train/src/s2/train.rs`, `crates/sky-train/src/s2/model.rs`, `crates/sky-train/src/s2/memory.rs`, `crates/sky-train/src/main.rs` | **The preservation term**: `--keep-close-to <checkpoint> --keep-close-weight λ`; load the base's parameters once, answer each stateless batch and each game-sequence frame with them (no memory, ternary, no gradient), add λ · KL(p_base ‖ p_θ) at the last exit to the decision loss and to the sequence loss of the four games; the KL of `exit_kd_loss` (`model.rs`, lines 1222 onwards) is the pattern. Tests: the term is zero when the model is the base; its gradient reaches the base's parameters and, in sequences, the memory's. | K5, the preserved route |
| 7 | `scripts/train-sleep4.sh` (new) | `k3` first, then `k1`, `k2`, `conf`, `k5` only as §3.6 allows (refused otherwise), `nano` (phase A by the rule with the cap, phase B in steps, the control per seed or the base), `export`, `eval` (dev3, the test of §5, the zeroed readings), `gate`; `--steps` and `--stop-after` as §3.1; the GPU, one run at a time, refused while another training runs (as `no_other_training` in `train-sleep3.sh`, lines 193 to 203). | everything |
| 8 | `scripts/s6-sleep4-report.py` (new) and its tests | the windows of 50, S90, S95 and the excess ratio from `log.jsonl` (a run stopped by `--stop-after` has no final line: the reader must not require one); the rules of §3.2 to §3.6 written to `runs/s6-sleep4/stage1.json`, each step refused unless the previous verdict allows it; the cap of §4.2; the check of identical batches (§3.1); the seeds of §5 and the refusal of v3's; Z in the gate; on the frozen route, the pairing with the base and the two parity checks of §4.1 (exact equality, answer for answer; a difference stops the run). | everything |

Possible today without code: K1 (every flag exists), K2 (i) (`--memory-aux-weight 2`), K2 (iii) (`--adam-lr 0.002`), K2 (iv) as a value (`--clip-norm`), the shorter checks (`--stop-after`), the device (`--cpu`), phase B in steps (`--steps`), the zeroed readings (`s2-probe --memory-off`), the new seeds (`--from-seed`).

## 12. Choices (the first value of each line is the one taken at the freeze; the alternatives are recorded)

| | choice | taken | alternatives | why |
|---|---|---|---|---|
| C1 | device of stage 1 | **the GPU, in steps, one run at a time, never during another budgeted GPU run** (the lead) | the CPU at `nice -n 19`, as v2's and v3's checks | in steps, the device changes the cost, not the result; about 8 working hours against about 65 (six to seven days of wall clock); stage 2 runs on the GPU, so the S values that set its cap come from the same device |
| C2 | the length of K1 | **1,310 steps, about 1,002 `recall` steps** | an exact stop at 1,000 `recall` steps (code) | possible today; every check shares one schedule |
| C3 | the windows of S90 and S95 | **trailing 50 `recall` steps for S90, 100 for S95, pooled over the asks** | per step; 100 for both | 6,400 asks give a standard error near 0.004 at 0.90; S95 uses the window of K1's bar and of the rule of phase A |
| C4 | the seed of K2 | **0** | 1; the slower seed of K1 | v3's seed 0 was the median of the three (the answer left chance at `recall` step 161, between 99 and 171) |
| C5 | the length of K2 | **K1's schedule cut at 654 steps (`--stop-after`), about 500 `recall` steps** | 500 `recall` steps under their own schedule | the same batches and the same rate at every step as K1's seed 0: S90 compares one factor |
| C6 | the clip of K2 (iv) | **the median of `grad_norm` over `recall` steps 51 to 250 of K1's seed 0, rounded up to one decimal** | **dropped: the median of the first 50 steps**; no clip (`--clip-norm 0`); the memory and the base clipped apart (code) | in v3 the first-50 median was 0.92 to 2.27 and the norm then rose to 6 to 11: the first-50 rule would put the clip near the default 1.0 and test nothing; 51 to 250 is where v3's steps were clipped |
| C7 | (i) and (ii) both adopted | **the lower S90; on a tie, (ii)** | always (i), which needs no code | the faster one; (ii) is what was pre-registered |
| C8 | the control of the frozen route | **the base file itself, read once; the zeroed and empty-memory readings are parity checks (exact equality)** (the lead) | a trained control as in v3 | a frozen model without memory trains nothing; v3's control lost at Flappy against its own base |
| C9 | the preservation weight | **λ = 1.0, KL at the last exit, temperature 1, in both arms** | λ = 0.3; λ = 3; KL at every exit | the same scale as the frame cross-entropy next to it; the gate reads the last exit; both arms so that H2 still isolates the memory |
| C10 | where the term applies | **every stateless batch and every game-sequence frame of phase B** | the stateless batches only; the game frames only | H2's loss is closed-loop on games, where the memory is written; the stateless records hold the typed states of H3 |
| C11 | K5's bar | **Flappy's training loss does not rise, and closed-loop Flappy on check seeds used nowhere else at least 0.98 times the base's** (the lead) | the training loss alone | v3's training losses did not separate the arms (0.384 against 0.380 over the last 20 `flappy` steps, seed 0) while the test did (15.87 against 71.05) |
| C12 | K5's check seeds | **`s2-probe --game flappy --from-seed 8000 --episodes 50`** | 30 episodes; the gate's arena (seed 7) | used by nothing else; 50 episodes for a 2% comparison; the gate's seeds stay the gate's |
| C13 | phase B | **2,000 steps** | 877 (v3); 3,000 | the pilot's 2,051 and 407 `recall` steps; 400 expected |
| C14 | the change of phase | **as v3** | a lower peak in phase B; a longer warm-up (code); the optimiser carried over (code), each after a check K4 | §4.4: the drop was recovered, the rule removes the jump of the rate, and a third change needs a third check |
| C15 | the control's steps on the preserved route | **each control seed trains its memory seed's phase-A steps** | the steps of memory seed 0, as v3 | phase A ends by a rule, so the seeds differ |
| C16 | dev seeds | **dev3, 500 seeds ≡ 19 mod 20 from 3,000,000,000** | 1,000 episodes | never read by anything; v3's size |
| C17 | test seeds | **`recall` and `rule110` from 6,000,000,000 (1,000 episodes), the arena of H2 from 6000 (200 per game)** | | never read by anything; v1's sizes |
| C18 | `--memory-off-eval 4` in the checks | **yes** | no | C5's control on the same batch, for a few per cent of the time |
| C19 | the in-sample stop of v2 | **kept for every memory seed of stage 2** | dropped | it costs nothing and stage 1 makes it unlikely |
| C20 | the order of stage 1 | **K3 first; K1, K2, the confirmation and K5 only if K3 fails** (the lead) | K1 and K2 first, then K3 (the draft) | a frozen base that trains makes the other checks moot: 1.5 to 1.8 GPU hours instead of at least 5.1 when K3 passes |
| C21 | the s6-nano gate's arena | **`sky-arena` seed 7, 30 episodes, its bar unchanged** (the lead) | new gate seeds | the gate's fixed protocol for every model; disclosed as read by every earlier gate (§8) |

## 13. The lead's decisions at the freeze (17/11/2026)

The draft of 17/11/2026 was accepted with these changes, all written into the sections above before the freeze:

1. Stage 1 runs on the GPU, in steps, one run at a time, never during another budgeted GPU run (§3.1, §3.7; C1).
2. K3 runs first. If it passes K1's bar on both seeds, the frozen route is taken, K1 and K2 are not run, and the cap of stage 2 comes from K3's own S95; only if K3 fails do K1, K2, the confirmation and K5 take place. A pass on one seed only is a failure (§3.6; C20).
3. K2 (iv): the median `grad_norm` over K1 seed 0's `recall` steps 51 to 250; the first 50 steps are the dropped alternative (§3.4; C6).
4. K5 reads closed-loop Flappy on its check seeds from 8000 (50 episodes, at least 0.98 times the base), seeds used nowhere else, in addition to the training-loss condition (§3.5; C11, C12).
5. On the frozen route the control is the base file; the zeroed reading and the language reading with an empty memory must equal the base exactly, a parity check whose failure is a bug that stops the run; H2 and H3 compare the memory model against the base. On the preserved route the control carries the same preservation term (§4.1, §6; C8, C9).
6. The s6-nano gate's arena (seed 7, 30 episodes) is kept with its bar unchanged and disclosed as already read by every earlier gate (§5, §8; C21).
7. Stage 2's cost is accepted as estimated; on the frozen route it has no control run (§9).
8. The other proposals stand: K2 on seed 0; the tie rule for (i) and (ii); λ = 1 at the last exit; phase B in 2,000 steps; the change of phase as in v3; dev3 from 3,000,000,000, the test from 6,000,000,000, the arena of H2 from 6000.

## 14. Amendments

### Amendment 1 (18/11/2026), owed before stage 2, recorded after K3's reading

The lead's two readings of the frozen text (TODO.md [removed on 19/11/2026; last version at commit `0e45b7b`], item 5, statement of 18/11/2026), written down here before any stage-2 run:

1. **The language parity check of the frozen route** (§4.1, §6 H3) compares the `s2-eval` report of the memory model with the base's line by line: every line of the report (each exit and source: accuracy, NLL, ECE, the early exit, the permutation and coherence checks), for the float and the ternary rows, equal, but for the two lines that name the model and its checkpoint (`typed_parity` in `scripts/s6-sleep4-report.py`). `s2-eval` writes no per-question answer, so this is a necessary condition; a per-answer dump would be the exact check that §4.1's "answer for answer" asks for.
2. **v2's stop** (§4.2, C19: the ask-frame cross-entropy above 1.336 over the last 50 `recall` steps of phase A) is read as one rule for all seeds: it is read on every memory seed, and if it holds for any one of them, no phase B runs for any seed and v4 reports a failure in-sample (`cmd_phase_a`, which sets `stop` when it holds for any seed).

**Both are moot for the frozen route**, which K3's failure closed (§15). On the preserved route, (2) still applies, to the memory seeds of that route, as §4.2 writes it. (1) does not: on the preserved route every parameter trains, so the memory model is not the base and H3 is read as v3 read it, `typed_decisions_bench` at most 0.01 under that route's trained control, with no parity check.

### Amendment 2 (19/11/2026), written before K1 runs: K1, and K1 only

Written on 19/11/2026, after K3's reading (§15) and before any step of K1 has run: no K1 run, log or reading exists when this is written. It records the owner's decision of 19/11/2026, taken by the lead on the literature review of the same day.

**The decision.**

1. **v4 goes on past K3 to K1, and to K1 only for now.** K1 runs as §3.3 writes it: v3's recipe unchanged, everything trained, phase A alone, `--steps 1310`, seed 0 then seed 1, one run at a time on the GPU (§3.1, §3.7), with `--memory-off-eval 4` (§3.1). Its command is K3's (`runs/s6-sleep4/stage1/k3-s*/cmd.txt`) without `--freeze-base`, and it is checked against v3's nano phase A by `s6-sleep4-report.py recipe`, as K3's was.
2. **No stop before the end of the budget.** K1 carries neither `--stop-after` (the shorter checks of K2, §3.1) nor `--stop-recall-acc` (the rule of stage 2, §4.2). K1's bar reads the last 100 `recall` steps of the run, and an early stop would move them. K3 ran without either flag. Seed 1 runs whatever seed 0 reads: the bar is read on both seeds (§3.1), and the frozen text has no rule that stops a check after one seed.
3. **The reading is §3.1's, unchanged**: the windows of 50 `recall` steps, S90, S95, and K1's bar per seed and on both seeds; a pass on one seed only is a failure. `s6-sleep4-report.py k1` writes it to `runs/s6-sleep4/stage1/k1.json` and `k1.md`, and adds K1's part to `runs/s6-sleep4/stage1.json`, keeping K3's part as it was read. It also reports, deciding nothing, whether each K1 seed trained on the same batches as K3's run of the same seed (§3.1's check, by `seq_game` and `passes`), when K3's log is present.
4. **If K1's bar fails, v4 stops at K1.** No K2, no confirmation, no K5, no stage 2. That reading is v4's result, and it is reported and published as one (§7, last item).
5. **If K1 passes on both seeds, nothing further runs until the lead writes the next amendment.** K2 (ii), K5 and the preserved route need code that is not written (§11 items 3 and 6, and their parts of items 7 and 8). The next amendment is bound by §10, as this one is.
6. **Not part of this decision:** the key-overlap reading that the review suggests next to K1 (C2's reading at the nano initialisation, `s2-memcheck --c2-only`). It is not run under this amendment.

**Why.**

- **K3's reading** (§15). With the base frozen, the answer stayed at chance on both seeds (0.2579 and 0.2546 over the last 100 `recall` steps), and the auxiliary heads decoded the colour only in part, on seed 0 only. That closes the frozen route. It says nothing about K1's question: does v3's recipe, with everything trained, reach 0.95 on its own batches within about 1,000 `recall` steps? The diagnosis left that question open (`docs/s6-sleep4-diagnosis.md` §5: "Whether more steps would bring the nano memory to 1.00").
- **The review** (`docs/research-s6-sleep4.md`, 19/11/2026, §13.2 and §13.3) recommends running K1 before anything else of the remaining route is written. It calls K1 the cheapest step that separates "too few steps" from "interference". It adds that the literature's strongest lever against interference, a curriculum on the delay, is not one of K2's four factors, and §10 forbids adding a factor. The review states its own confidence as **low to moderate**: the closest evidence is one unreviewed 2026 preprint, at other sizes, on token sequences rather than game frames (§13.3, §14). It is a reason for the order of the work, not evidence about what K1 will read. The review proposed K1 at seed 0 alone with the key-overlap reading. This decision runs both seeds, as §3.3 writes, and leaves the key-overlap reading out.
- **The cost of the order.** From K3's failure, the frozen text's route costs 216 to about 369 GPU working minutes of checks and up to about 8 hours of stage 2, plus the code of §11 items 3 and 6 (§15). K1 costs 108 of those minutes, and its trainer flags all exist (§11, last paragraph).

**What this changes in the frozen text, and why it is the conservative direction.**

- **§3.6, rule 2, and §15's list "The rule that applies"** run K1, then K2 (i) to (iv), then the confirmation if a factor is adopted, whatever K1 reads. By rule 4, v4 stops only if R\* did not pass K1's bar or if K5a fails. So under the frozen text, a failed K1 could still lead to K2: a factor adopted, a confirmation that passes, then K5 and the preserved route. **This amendment removes that path: a K1 failure stops v4.** It also puts a pause after a K1 pass: K2 waits for a further amendment. §3.6's other rules and §3.3's "What follows" are not changed: K1's logs keep what K2 would need (the reference S90 and the clip of K2 (iv)) if a later amendment lets K2 run, and this amendment computes neither.
- **§10 ("Forbidden"), line by line.** Nothing in it forbids running K1 alone or stopping after it. These are the lines that bear on it:
  - "Any v4 computation before ... the code of §11 is written and tested." K1 needs only its parts of items 7 and 8: the stage of `scripts/train-sleep4.sh` and the reading of `scripts/s6-sleep4-report.py`, written and tested with this amendment. It also needs the trainer flags, which exist. Items 3 and 6 serve K2 (ii) and K5, which do not run. K3 ran under the same reading of that line (§15).
  - "A check beyond K3, K1, K2 (i) to (iv), the confirmation and K5." No check is added.
  - "Running a check again except after a technical failure." K1 runs once. A run cut by a technical failure restarts from scratch with the same seed (§3.1), and the script refuses a second K1 once it is read.
  - "Choosing R\*, the route or the cap by anything but the rules of §3 and §4.2." Nothing is chosen. If K1 fails, there is no R\*, no route and no cap. If K1 passes, the choice waits for an amendment written before the readings it concerns.
  - "Changing a bar ... after a reading it concerns." No bar moves, and this is written before K1's reading.
- **Why it is the conservative direction.** Stopping earlier can turn a later pass into a stop, but never a failure into a pass. The amendment adds no check, factor, seed or reading, and it spends fewer GPU minutes than any path of §3.7. It is written before the reading it concerns, so it cannot be fitted to K1's result. A pre-registration may always stop earlier than its rules allow, when the stop is written before the run.

**What each outcome of K1 will mean.**

- **K1 fails** (on either seed). In §10's words, v3's recipe does not reach 0.95 on its own batches within about 1,000 `recall` steps at the nano size. As §7 says when stage 1 stops, v3's gap to the pilot is then not a matter of budget alone. K2 does not run, so the reading cannot say whether one of the four factors would have changed it: the auxiliary weight per layer, the head as T2 was written, the pilot's rates, or the clip. v4 ends at K1, with no stage-2 GPU minute spent, and that reading is its result. As §7 says, any next step is a separate decision and file. S90, S95 and the windows are reported as read; they decide nothing.
- **One seed passes and the other does not.** That is a failure (§3.1), so v4 stops at K1, as above.
- **K1 passes on both seeds.** On its own training batches, v3's recipe with everything trained reaches 0.95 within about 1,000 `recall` steps on both seeds. v3's nano gap on those batches was then mostly the budget that its minutes bought (diagnosis §2 (a), (b)). This is an in-sample reading on training sequences. It says nothing about dev3 or the test, the games (H2, which K5 was to protect), sleep (H1), or the gate, and it does not settle R\* (K2 has not run). v4 waits for the next amendment.

**Cost.** By §3.7's figures, K1 costs 54 GPU working minutes per seed, 108 for the two (1.8 hours). K3 measured 97.2, so stage 1 comes to about 205 working minutes (3.4 hours) if v4 stops at K1. That is below §9's 5.1 to 8.0 hours for v4 stopping after stage 1, because the frozen text had no path that stops at K1. Wall clock is longer under the brake: K3's 97.2 working minutes took 4 hours 35 minutes (13:35 to 18:10 on 18/11/2026, §15). K1 computes the base's gradients, which K3 skipped, so its steps are expected to be slower than K3's (not measured).

**Nothing else moves.** These stay as frozen:

- the bars: K1's bar, 0.95 pooled over the last 100 `recall` steps, on both seeds;
- the windows of S90 and S95;
- the schedule: `--steps 1310`, `wsd`, v3's rates and clip;
- the seeds: training seeds 0 and 1, with no dev3, validation or test seed read;
- the readings of §3.1;
- amendment 1.

§15 found that the code adds an auxiliary term on `rule110` steps that v3's T2 does not have. That holds for K1, which runs v3's recipe unchanged, and this amendment does not change it.

**The code.** `scripts/train-sleep4.sh k1` and `scripts/s6-sleep4-report.py k1-ready` and `k1`, tested in `scripts/test_s6_sleep4_report.py`, written with this amendment before K1 runs. No trainer code is written for it. The stage refuses to run unless `runs/s6-sleep4/stage1.json` records K3's failure and is committed (§3.6: each verdict committed before the next check). It stops after K1's reading, whatever K1 reads, with the rule of points 4 and 5.

## 15. K3, read on 18/11/2026: failed; the frozen route is closed

**What ran.** `scripts/train-sleep4.sh k3` from `main`'s commit `c6794ac`, through the job queue (`runs/all/queue/`, job `21-sleep4-k3`, started 18/11/2026 at 13:35:32; its text is `runs/s6-sleep4/stage1/queue-job.txt`, its console `runs/s6-sleep4/stage1/console.log`). A first job, `20-sleep4-k3`, failed at 13:17:32 with status 126 ("Permission denied": the script lacked its executable bit); no step ran, so it is no run of K3. Seed 0 ran from 13:35 to 15:24 (38.2 working minutes), seed 1 from 15:25 to 18:10 (59.0 working minutes), each 1,310 steps on the GPU, one after the other, with `--freeze-base --memory-off-eval 4` and v3's recipe otherwise (each command checked against v3's by `s6-sleep4-report.py recipe`; `runs/s6-sleep4/stage1/k3-s*/cmd.txt`). The frozen base was checked bit for bit at every save and at the end (`"frozen_base_checked": true` in both final lines, `k3-s*/final.json`); the validation at the end equals the one at step 0 on both seeds (f32 0.5132 and 0.5015, ternary 0.5985 and 0.6128). The reading is `runs/s6-sleep4/stage1/k3.json` and `runs/s6-sleep4/stage1.json`, written at 18:10 by `s6-sleep4-report.py k3`; the job's status 1 is the script's exit on K3's failure, not a technical failure. Every number below comes from these files and from each run's `log.jsonl` (each step's own batch, before its update).

**The bar (K1's, §3.1): failed on both seeds.**

| seed | `recall` steps | `ask_acc`, last 100 `recall` steps | bar | S90 | S95 |
|---|---:|---:|---:|---|---|
| 0 | 1,019 | 0.2579 | 0.95 | never | never |
| 1 | 1,000 | 0.2546 | 0.95 | never | never |

Chance is 0.25 (four colours).

**The windows of 50 `recall` steps (§3.1).** `ask_acc` pooled over the window; `aux_acc` per memory layer, pooled over the window; the excess ratio (n/a while the 1 to 4 bucket is within 0.02 of chance); `off`, the same weights without the memory on the steps of the window that read it (one sequence step in four; their number in brackets).

| `recall` steps | seed 0 `ask_acc` | seed 0 `aux_acc` L0 / L1 | seed 0 excess | seed 0 `off` | seed 1 `ask_acc` | seed 1 `aux_acc` L0 / L1 | seed 1 excess | seed 1 `off` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1-50 | 0.104 | 0.256 / 0.254 | 0.99 | 0.233 (14) | 0.091 | 0.248 / 0.255 | 1.10 | 0.229 (13) |
| 51-100 | 0.047 | 0.259 / 0.260 | 1.01 | 0.230 (11) | 0.028 | 0.236 / 0.241 | 0.99 | 0.236 (14) |
| 101-150 | 0.112 | 0.234 / 0.237 | 0.99 | 0.219 (10) | 0.053 | 0.245 / 0.255 | 0.98 | 0.202 (11) |
| 151-200 | 0.191 | 0.252 / 0.249 | 1.38 | 0.246 (11) | 0.131 | 0.255 / 0.248 | 0.81 | 0.201 (12) |
| 201-250 | 0.210 | 0.247 / 0.250 | 1.16 | 0.212 (13) | 0.175 | 0.252 / 0.240 | 0.78 | 0.212 (13) |
| 251-300 | 0.236 | 0.253 / 0.252 | n/a | 0.227 (11) | 0.204 | 0.246 / 0.256 | 1.64 | 0.227 (13) |
| 301-350 | 0.231 | 0.247 / 0.251 | n/a | 0.236 (12) | 0.219 | 0.251 / 0.242 | 1.00 | 0.202 (12) |
| 351-400 | 0.239 | 0.255 / 0.260 | n/a | 0.243 (12) | 0.245 | 0.251 / 0.251 | n/a | 0.241 (11) |
| 401-450 | 0.223 | 0.245 / 0.249 | 1.03 | 0.220 (11) | 0.225 | 0.255 / 0.243 | 1.35 | 0.222 (13) |
| 451-500 | 0.241 | 0.253 / 0.245 | 1.06 | 0.224 (12) | 0.244 | 0.250 / 0.255 | n/a | 0.221 (10) |
| 501-550 | 0.242 | 0.260 / 0.259 | n/a | 0.220 (13) | 0.243 | 0.254 / 0.251 | n/a | 0.224 (11) |
| 551-600 | 0.246 | 0.255 / 0.258 | -0.03 | 0.232 (14) | 0.256 | 0.251 / 0.248 | n/a | 0.207 (12) |
| 601-650 | 0.257 | 0.257 / 0.257 | n/a | 0.224 (13) | 0.246 | 0.246 / 0.255 | n/a | 0.218 (12) |
| 651-700 | 0.247 | 0.278 / 0.281 | n/a | 0.228 (12) | 0.233 | 0.250 / 0.250 | 0.65 | 0.202 (11) |
| 701-750 | 0.256 | 0.292 / 0.338 | n/a | 0.229 (11) | 0.252 | 0.250 / 0.249 | n/a | 0.208 (11) |
| 751-800 | 0.260 | 0.353 / 0.362 | 0.35 | 0.219 (12) | 0.255 | 0.253 / 0.252 | n/a | 0.228 (13) |
| 801-850 | 0.258 | 0.362 / 0.365 | n/a | 0.227 (14) | 0.262 | 0.261 / 0.265 | n/a | 0.232 (12) |
| 851-900 | 0.264 | 0.359 / 0.374 | n/a | 0.200 (13) | 0.245 | 0.235 / 0.250 | 0.49 | 0.208 (13) |
| 901-950 | 0.259 | 0.373 / 0.385 | n/a | 0.231 (12) | 0.256 | 0.253 / 0.250 | n/a | 0.193 (10) |
| 951-1000 | 0.257 | 0.373 / 0.385 | n/a | 0.219 (13) | 0.254 | 0.258 / 0.249 | n/a | 0.215 (11) |
| 1001-1019 | 0.256 | 0.379 / 0.389 | n/a | 0.227 (5) | | | | |

**The auxiliary heads do not decode the colour on `recall`.** On seed 1, `aux_acc` stays at chance in every window (0.2524 over the last 100 `recall` steps, `aux_ce` 1.388 against ln 4 = 1.386). On seed 0 it leaves chance from `recall` step 651 on and reaches 0.38 (0.3804 over the last 100, `aux_ce` 1.248), while `ask_acc` stays between 0.247 and 0.264 over the same windows. No `recall` step reaches an `aux_acc` of 1.0 (the largest of one step: 0.477 on seed 0, 0.363 on seed 1). The `aux_acc` of 1.0 with `aux_ce` near 0 that the logs show (for example at step 180 of seed 0) is on `rule110` steps, 162 of seed 0's 172 and 166 of seed 1's 173: a `rule110` ask frame (`... ask d steps 4`) is an ask for the code (`is_ask` in `crates/sky-train/src/s2/seq.rs`), and since it has no colour, `answer_colour` gives none and its auxiliary target is the constant 0 (`seq.rs`, `load`: `answer_colour(&v).unwrap_or(0)`). A constant target says nothing about what the memory holds. v3's T2 says that "`rule110` and the other games have no auxiliary loss" (`docs/s6-sleep3-prereg.md` §2), so the code adds an auxiliary term on `rule110` that the text does not have. This holds for v3's runs and for K3, and it holds for K1, which runs v3's recipe unchanged. **So the numbers do not show that the memory stores and returns the colour while the frozen base ignores it.** What they show is this: with the base frozen, what the memory retrieves did not become linearly decodable for the colour within 1,000 `recall` steps on seed 1, and became so only in part on seed 0 (0.38), where the answer did not follow.

**The memory off.** The base alone answers `recall` at 0.19 to 0.25 in every window, as a model without memory must. The memory starts as the base: at the first `recall` step, the answers with and without the memory are the same (0.2422 on both seeds, with cross-entropies equal to 1e-4), because the read-out starts at zero. The memory then pushed the answer below the base, then brought it back to chance. Over the steps that read both, for `recall` steps 1 to 100: 0.079 with the memory against 0.232 without on seed 0 (25 steps), and 0.065 against 0.233 on seed 1 (27 steps), at a last-exit cross-entropy of 2.38 and 2.40 against 1.62 and 1.61. Over the last 100 `recall` steps: 0.253 against 0.225 on seed 0 (26 steps), and 0.239 against 0.205 on seed 1 (21 steps), at a cross-entropy of 1.50 and 1.45 against 1.62. This gain of 0.03 is the same at every delay: over the last 100 `recall` steps, the accuracy at 1-4, 5-8, 9-16 and 17-32 is 0.249, 0.257, 0.259 and 0.266 on seed 0, and 0.256, 0.266, 0.253 and 0.243 on seed 1.

**The rule that applies (§3.6, rule 2).** K3 fails on both seeds, so the frozen route is closed for v4, and no frozen base runs again (§3.4, §10). Next, in this order, each check's reading and verdict committed before the next starts:

1. K1 (§3.3), seed 0, then seed 1.
2. K2 (i) to (iv) (§3.4), on seed 0, against K1's seed 0. (iv)'s clip is computed from K1's seed-0 log.
3. The confirmation, if at least one factor is adopted. R\* follows from §3.4.
4. If R\* passed K1's bar, K5 (§3.5) from the final checkpoint of R\*'s seed-0 run. If K5a meets both of its conditions, v4 takes the preserved route (§4). Otherwise, or if R\* did not pass K1's bar, v4 stops after stage 1 (rules 3 and 4).

If v4 goes on, the code these checks need (§11 items 3 and 6, and K1, K2, the confirmation, K5 and the preserved route in `scripts/train-sleep4.sh` and `scripts/s6-sleep4-report.py`) must be written and tested before the first of them runs (§10).

**What follows costs** (the pre-registration's own figures, §3.7 and §9, in GPU working minutes; K3 itself took 97.2, against the 90 to 108 of §3.7):

| path from here | checks | GPU working minutes |
|---|---|---:|
| v4 stops before K5, no factor adopted | K1 × 2, K2 × 4 | 216 |
| v4 stops before K5, a factor adopted | K1 × 2, K2 × 4, confirmation × 2 | 324 |
| no factor adopted, then K5 | K1 × 2, K2 × 4, K5 | about 261, plus three Flappy probes on the CPU |
| a factor adopted, the confirmation, then K5 (the maximum) | K1 × 2, K2 × 4, confirmation × 2, K5 | about 369, plus the probes |
| then the preserved route, stage 2 | three memory seeds and three controls | at most about 8 hours, plus the term's forward pass (§9; S95max = 700) |

The figures per check are §3.7's: K1 54 per seed, K2 27 per factor, the confirmation 54 per seed, K5 about 45 for its two runs. With K3's 97.2, stage 1 comes to 313 to 466 working minutes (5.2 to 7.8 hours), within §9's 5.1 to 8.0.

**State on 18/11/2026 (evening).** The code of K1, K2, the confirmation, K5 and the preserved route (§11 items 3 and 6, and their parts of item 7 and item 8) is **not written** on `main`. No check has run since K3. Whether v4 goes on past K3 is the owner's decision, and it has not been taken. If v4 stops here, it ends at K3 with this reading, and that is a result. Here is what the reading supports, and no more. At the nano size, with v3's recipe and the base frozen, the memory did not reach K1's bar on either seed within about 1,000 `recall` steps: the answer stayed at chance (0.2579 and 0.2546). What the memory retrieves became linearly decodable for the colour only in part, on one seed (`aux_acc` 0.38 on seed 0, at chance on seed 1). With the memory on, the answer gained 0.03 over the same weights without it, and the gain did not depend on the delay, so the memory gave no sign of holding the colour across the delay. The reading does not show that the memory stores and returns the colour while a frozen base fails to use it: that would need the colour decoded well above chance, and the auxiliary heads did not decode it.

**State on 19/11/2026.** The owner's decision is taken, and amendment 2 (§14) records it. v4 goes on to K1 and to K1 only. Points 2 to 4 of "The rule that applies" above, and the costs after them, are replaced for what comes next by that amendment. K1's code is written: its stage in `scripts/train-sleep4.sh` and its reading in `scripts/s6-sleep4-report.py`, with their tests. The code of K2 (ii), K5 and the preserved route (§11 items 3 and 6) is not written, and amendment 2 does not call for it. K1 has not run. If K1 fails on either seed, v4 stops at K1 and that reading is its result. If it passes on both seeds, nothing further runs until the lead writes the next amendment.

## 16. K1, read on 19/11/2026: failed (the bar met on one seed of two); v4 stops at K1 and is closed

**What ran.** `scripts/train-sleep4.sh k1` from `main`'s commit `267d3fb`, through the job queue (`runs/all/queue/`, job `60-sleep4-k1`; its text is `runs/s6-sleep4/stage1/k1-queue-job.txt`, its console `runs/s6-sleep4/stage1/k1-console.log`). By the times of the run files, seed 0 ran from 11:19 to 13:08 and seed 1 from 13:08 to 15:43, each 1,310 steps on the GPU, one after the other, under the brake. The console's last timed lines read 57.8 working minutes at step 1,267 (seed 0) and 60.7 at step 1,283 (seed 1), against §3.7's estimate of 54 per seed. Each command is v3's nano phase A with everything trained, `--steps 1310 --memory-off-eval 4`, and neither `--stop-after` nor `--stop-recall-acc` (amendment 2, points 1 and 2); each was checked against v3's by `s6-sleep4-report.py recipe` (`runs/s6-sleep4/stage1/k1-s*/cmd.txt`). The reading is `runs/s6-sleep4/stage1/k1.json`, `k1.md` and K1's part of `runs/s6-sleep4/stage1.json` (`k1`, `k1_pass`, `next_after_k1`; K3's part unchanged), written at 15:43 by `s6-sleep4-report.py k1`. K1 has no `final.json` (K3's held the last line of each run's log, kept for its frozen-base check). Every number below comes from these files, and the windows from each run's `log.jsonl`: **each step's own training batch, before its update. This is a reading on training sequences, not a held-out test.** No dev3 or test seed of any probe or game was read. The trainer's own validation on a sample of the stateless records, which it prints at step 0 and at the end of every run, is in the console and decides nothing.

**The bar (K1's, §3.1): met on seed 1, missed on seed 0, so K1 fails.**

| seed | steps | `recall` steps | `ask_acc`, last 100 `recall` steps | bar (0.95) | S90 | S95 | same batches as K3 |
|---|---:|---:|---:|---|---|---|---|
| 0 | 1,310 | 1,019 | 0.8442 | missed | never | never | yes |
| 1 | 1,310 | 1,000 | 1.0000 | met | 465 | 516 | yes |

Chance is 0.25 (four colours). A pass on one seed only is a failure (§3.1). "Same batches as K3" is §3.1's check by `seq_game` and `passes`, reported and deciding nothing: each K1 seed trained on the batches of K3's run of the same seed, so K1 and K3 differ only in which parameters train.

**The windows of 50 `recall` steps (§3.1).** `ask_acc` pooled over the window; the accuracy at delays 1 to 4 and 17 to 32 frames (`by_delay`); `off`, the same weights without the memory on the steps of the window that read it (one sequence step in four; their number in brackets). The auxiliary accuracy per layer, the excess ratio, `ask_ce` and `grad_norm` are in `k1.json` and the console.

| `recall` steps | seed 0 `ask_acc` | seed 0 1-4 | seed 0 17-32 | seed 0 `off` | seed 1 `ask_acc` | seed 1 1-4 | seed 1 17-32 | seed 1 `off` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1-50 | 0.161 | 0.146 | 0.174 | 0.151 (14) | 0.170 | 0.176 | 0.176 | 0.162 (13) |
| 51-100 | 0.253 | 0.276 | 0.232 | 0.256 (11) | 0.251 | 0.256 | 0.256 | 0.259 (14) |
| 101-150 | 0.251 | 0.266 | 0.233 | 0.259 (10) | 0.271 | 0.279 | 0.272 | 0.256 (11) |
| 151-200 | 0.238 | 0.213 | 0.266 | 0.268 (11) | 0.285 | 0.309 | 0.280 | 0.221 (12) |
| 201-250 | 0.280 | 0.274 | 0.278 | 0.250 (13) | 0.441 | 0.540 | 0.323 | 0.243 (13) |
| 251-300 | 0.306 | 0.329 | 0.288 | 0.263 (11) | 0.558 | 0.729 | 0.367 | 0.254 (13) |
| 301-350 | 0.349 | 0.391 | 0.291 | 0.243 (12) | 0.585 | 0.771 | 0.371 | 0.255 (12) |
| 351-400 | 0.444 | 0.541 | 0.331 | 0.248 (12) | 0.598 | 0.754 | 0.397 | 0.248 (11) |
| 401-450 | 0.546 | 0.704 | 0.349 | 0.246 (11) | 0.813 | 0.904 | 0.691 | 0.274 (13) |
| 451-500 | 0.574 | 0.750 | 0.381 | 0.251 (12) | 0.991 | 0.998 | 0.976 | 0.229 (10) |
| 501-550 | 0.593 | 0.786 | 0.371 | 0.238 (13) | 0.993 | 0.999 | 0.983 | 0.249 (11) |
| 551-600 | 0.590 | 0.779 | 0.369 | 0.251 (14) | 0.998 | 0.998 | 0.994 | 0.249 (12) |
| 601-650 | 0.586 | 0.780 | 0.364 | 0.236 (13) | 0.998 | 1.000 | 0.994 | 0.244 (12) |
| 651-700 | 0.610 | 0.799 | 0.376 | 0.250 (12) | 0.998 | 0.999 | 0.994 | 0.246 (11) |
| 701-750 | 0.600 | 0.787 | 0.375 | 0.249 (11) | 0.999 | 1.000 | 0.998 | 0.240 (11) |
| 751-800 | 0.701 | 0.843 | 0.529 | 0.257 (12) | 0.999 | 1.000 | 0.994 | 0.235 (13) |
| 801-850 | 0.794 | 0.914 | 0.639 | 0.234 (14) | 0.999 | 1.000 | 0.998 | 0.237 (12) |
| 851-900 | 0.809 | 0.914 | 0.644 | 0.240 (13) | 0.999 | 1.000 | 0.997 | 0.229 (13) |
| 901-950 | 0.810 | 0.904 | 0.646 | 0.279 (12) | 1.000 | 1.000 | 1.000 | 0.164 (10) |
| 951-1000 | 0.849 | 0.945 | 0.681 | 0.248 (13) | 1.000 | 1.000 | 1.000 | 0.211 (11) |
| 1001-1019 | 0.877 | 0.956 | 0.729 | 0.247 (5) | | | | |

**What the windows show, as observations and not as a pass.** None of them changes the verdict.

1. **Seed 1: a plateau, then a jump.** `ask_acc` sat at 0.558, 0.585 and 0.598 over `recall` steps 251 to 400 (about 150 steps), then rose to 0.813 in 401-450 and 0.991 in 451-500, and stayed at 0.991 to 1.000 to the end. S90 is `recall` step 465 and S95 step 516.
2. **Seed 0: the same shape, later, and not finished.** A plateau at 0.574 to 0.610 over `recall` steps 451 to 750, then a rise to 0.701 (751-800), 0.794, 0.809, 0.810, 0.849 (951-1000) and 0.877 over the last, partial window (1001-1019, 19 steps). It was still rising when the budget ended. It was not run further: the budget and the bar were written before the run (§3.1, amendment 2), and §10 forbids running a check again except after a technical failure.
3. **Both plateaus are the short delays solved and the long ones not.** On both plateaus the accuracy at delays 1 to 4 frames was 0.73 to 0.80 and at 17 to 32 frames 0.36 to 0.40 (excess ratio 0.21 to 0.29). The jump is the long delays catching up: on seed 1, 17-32 reads 0.994 to 1.000 in every window from `recall` step 551 on, and 1.000 in the last two; on seed 0, 17-32 reads 0.681 in 951-1000 and 0.729 in 1001-1019, against 0.945 and 0.956 at 1-4.
4. **What was learned goes through the memory.** With the memory off, the same weights on the same batches answer between 0.151 and 0.279 in every window on both seeds (about 0.25, chance), while the memory-on answer rises to 0.849 (seed 0) and 1.000 (seed 1). These are 10 to 14 steps per window (5 in seed 0's last), 249 steps over seed 0 and 238 over seed 1.

**Compared with K3.** On the same batches and seeds, with the base frozen, the answer stayed at chance (0.2579 and 0.2546 over the last 100 `recall` steps, §15). With everything trained it reached 0.8442 and 1.0000. This is reported, and decides nothing: K3's route was closed by its own reading.

**The rule that applies (amendment 2, point 4).** K1 fails (a pass on one seed only is a failure), so **v4 stops at K1**: no K2, no confirmation, no K5, no stage 2. This reading is v4's result. In amendment 2's words for this outcome: v3's recipe does not reach 0.95 on its own batches within about 1,000 `recall` steps at the nano size on both seeds, so v3's gap to the pilot is not shown to be a matter of budget alone. K2 did not run, so the reading cannot say whether one of its four factors (the auxiliary weight per layer, the head as T2 was written, the pilot's rates, the clip) would have changed it. The reference S90 and the clip that K2 (iv) would have taken from K1's seed 0 are not computed. No stage-2 GPU minute was spent. Stage 1 cost K3's 97.2 working minutes and K1's two runs.

**What this reading does not show.**

- **Nothing about a held-out set.** Every number above is read on each step's own training batch, before its update. No dev3 or test seed of `recall` or of a game was read, and no language test, so K1 says nothing about how the memory does on episodes it has not trained on.
- **Nothing about sleep (H1) or the games (H2).** K1 is phase A alone: one pass (`--sleep-passes 1`), so no sleep to compare, and no game sequences of phase B. Nothing was exported, probed or gated.
- **No model.** K1's checkpoints are working files of the check and are not published. s6-sleep3-nano (v3) stays the published model of the line, with its failed gate; nothing about that file changes.
- **Two seeds, one run each.** One seed met the bar and one did not. Two runs cannot say how often a seed locks in within the budget, nor how many more steps seed 0 would have needed: its last window rose, and a rising curve is not a pass.
- **The key-overlap reading** that the review suggested next to K1 was not run (amendment 2, point 6).

**State on 19/11/2026 (after K1). v4 is closed.** It ended at K1, failed, with this reading as its result. Nothing further of v4 runs: K2, the confirmation, K5 and stage 2 are not run, and their code (§11 items 3 and 6) is not written. Nothing is scheduled after v4. A further attempt, for example longer training on both seeds or a curriculum on the delay, would be a new pre-registered line with its own file, and none is decided.
