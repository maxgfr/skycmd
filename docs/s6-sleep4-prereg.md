# s6-sleep v4 (s6-sleep4-nano): pre-registration

**DRAFT of 17/11/2026: not frozen; the lead freezes it by a dated commit before any computation.** No line of v4's code is written, no v4 check has run, no v4 model exists. Every element below was checked against the code at the commit of this draft; what the code cannot do today is listed in §11, and nothing of §3 or §4 may run before it is written and tested. The values this draft had to propose are in §12, each with its reason and its alternative; the questions only the lead can settle are in §13.

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
| V5 | **A frozen base, if it trains** (K3); otherwise a preservation term (K5). | H2 failed, and the control lost at Flappy against its own base (v3 Result block). With the base frozen and the memory zeroed, the file computes its base's outputs exactly (§3.4). |
| V6 | **The memory zeroed is read at the nano size**, on dev and on the test, and gated. | Never read at nano; H0 rests on the control alone (note, item 6). |
| V7 | **New dev and test seeds** (§5). | v3 read dev2 for its choices and read its test seeds once. |

**Not changed, and why.**

- **The memory itself** (v2 §4; `memory_qk: shared`, `memory_value: frame`, the average write, the gates held at β = 0.5 and α = σ(6), T1): the diagnosis ruled out the decay as the cause of the delay curve and found the gradient alive throughout nano phase A (§2 (d)). It supports overwriting by later frames, about 6% per frame, without being able to measure its mechanism; the slot write (`memory_write: slot`, written for v3's diagnosis, never run on a base) is not used. v4 asks whether more steps close the gap, and only that; if they do not, the interference reading belongs to a later file.
- **The mixes, the batch sizes, N = 1 in phase A and N drawn in {1, 2, 3, 4} in phase B, QAT from step 0, `--sleep-chunk 2`, `SKY_MLX_CACHE_GB=1`, `mem.*` at ×10 with the gates at ×0** (v3, amendment 3).
- **The change of phase** (a fresh AdamW and schedule, `s2-decide --init`). §4.4 says why no check of a softer change (K4) is run.

## 3. Stage 1: checks on training sequences, at the nano size

### 3.1 The common setting

Every check of stage 1 is the command of v3's nano phase A (`runs/s6-sleep3/nano/sleep-s0/a/cmd.txt`): the base `runs/s6-nano-g3/decide/last`, the config `runs/s6-sleep3/nano/sleep3.json` (v3's memory with `memory_aux`), stateless records and weights as v3's nano runs, batch 128, `--qat-frac 1.0 --loop-dropout 0.25 --length-pool 16 --pad-to 32 --step-mem-gb 3`, AdamW (`--no-muon`) at `--adam-lr 0.00163`, `--seq-log`, `--sleep-records recall,rule110 --sleep-weights recall=0.85,rule110=0.15 --sleep-batch 16 --seq-frac 0.9 --sleep-passes 1`, `--mem-lr-mult 10 --mem-gate-lr-mult 0 --memory-aux-weight 1`, `--sleep-chunk 2`, with these differences:

- **a step budget**: `--steps 1310` (the schedule of every check; at 0.9 × 0.85 = 0.765 `recall` steps per step, 1,002 `recall` steps expected), and `--stop-after 654` for the shorter checks (about 500 `recall` steps under the same schedule);
- **`--memory-off-eval 4`**, as C5: one sequence step in four is also answered by the same weights without the memory, on the same batch, before its update (reported, deciding nothing);
- **the training seeds 0 and 1**, as C5.

Only training sequences are read: each step's own batch, before its update, from `log.jsonl`. No dev, validation or test seed of any probe or game is read in stage 1, except by K5 (§3.5), whose check seeds are named there. A run stopped by a technical failure (a crash, a full disk, the memory guard) restarts from scratch with the same seed; no check is run twice for any other reason.

By reading of `crates/sky-train/src/s2/train.rs` (lines 2377 to 2545), the random draws of a step are the choice of a sequence step, the batch, N, the stateless batch and the loop dropout, and none of them depends on the rate, the clip or the auxiliary loss: two checks with the same seed train on the same batches. The report verifies it by comparing the `seq_game` and `passes` of their log lines.

**Readings**, per window of 50 `recall` steps (a `recall` step is a log line with `seq_game` `recall` and asks; they are counted from 1 in each run), each pooled over the asks of the window:

- the answer's accuracy at the last exit, `ask_acc`, and the ask-frame cross-entropy `ask_ce`;
- the auxiliary accuracy `aux_acc` (today the mean over the two layers; each layer apart once §11 item 2 is written);
- **the excess ratio**: (accuracy at delays 17 to 32 − 0.25) / (accuracy at delays 1 to 4 − 0.25), from `ask_splits` (reported; not defined while the 1 to 4 bucket is within 0.02 of chance);
- the same weights with the memory off (`off`), and `grad_norm` (median per window);
- **S90**: the first `recall` step n ≥ 50 at which `ask_acc` pooled over `recall` steps n − 49 to n is at least 0.90 (none if never);
- **S95**: the first `recall` step n ≥ 100 at which `ask_acc` pooled over `recall` steps n − 99 to n is at least 0.95 (none if never).

**The bar of a check** (K1's bar, used by every check that needs one): `ask_acc` pooled over the last 100 `recall` steps of the run is at least 0.95, on both seeds.

### 3.2 K1, more steps

- **What.** v3's recipe unchanged, phase A alone, `--steps 1310`, seeds 0 and 1.
- **Reads** the windows of §3.1, S90 and S95.
- **Bar:** K1's bar.
- **What follows.** K1 sets the reference S90 of K2 and, if it passes, the recipe of §3.6 when K2 adopts nothing. Its seed-0 run fixes the clip of K2 (iv).

### 3.3 K2, what slows it

- **What.** Seed 0 (§12, C4), `--steps 1310 --stop-after 654`, one factor changed at a time against K1's seed 0, on the same batches:
  - **(i) the auxiliary loss summed over the layers**, weight 1 per layer, the pilot's effective weight: `--memory-aux-weight 2`. The code divides the auxiliary term by asks × memory layers (`memory.rs`, lines 962 to 976), so weight 2 with two layers is weight 1 per layer. Possible today.
  - **(ii) the auxiliary loss on the last memory layer only, read at the door token of the ask frame, weight 1**, as T2 was written. Needs code (§11 item 3).
  - **(iii) the pilot's rates**: `--adam-lr 0.002`, so `mem.*` at 0.02 with `--mem-lr-mult 10`. Possible today.
  - **(iv) the clip at the measured norm**: `--clip-norm c`, with c the median of `grad_norm` over `recall` steps 51 to 250 of K1's seed 0, rounded up to one decimal. Possible today, once K1's seed 0 has ended. Why not the first 50 steps: in v3 the median over the first 50 nano `recall` steps was 0.92, 1.51 and 2.27 (seeds 0, 2, 1) and rose to 6 to 11 from `recall` step 75 on (note of 17/11, item 5); a clip at the first-50 median would sit near the default 1.0 and test nothing.
- **Rule.** A factor is **adopted** if its S90 is at most 0.75 times the S90 of K1's seed 0. A factor that does not reach 0.90 within its run is not adopted. If K1's seed 0 never reaches 0.90, its S90 counts as 1,001 for this rule.
- **(i) and (ii) exclude each other.** If both are adopted, the one with the lower S90 is kept; on a tie, (ii), the wording that was pre-registered.
- **The confirmation.** If at least one factor is adopted, the adopted factors together are run as K1 is run (`--steps 1310`, seeds 0 and 1). It must pass K1's bar.
- **The recipe R\*.** If the confirmation ran and passed, R\* is v3's recipe with the adopted factors; otherwise R\* is v3's recipe. "R\* passed K1's bar" means: K1 passed (R\* = v3's) or the confirmation passed (R\* with the factors).

### 3.4 K3, a frozen base

- **What.** R\* with the base frozen: every parameter that exists in s6-nano (every name not starting with `mem.`) keeps its value, and only the memory (`mem.{l}.norm`, `qk`, `v`, `out`, with the gates held as in v3) and the auxiliary heads (`mem.{l}.aux`) train. `--steps 1310`, seeds 0 and 1. Needs code (§11 item 1).
- **The model equals s6-nano's checkpoint at step 0: this already holds.** The read-out `mem.{l}.out.weight` is initialised at zero (`crates/sky-train/src/s2/memory.rs`, lines 116 to 117), its ternary form under QAT is zero too (`ternarize` in `crates/sky-train/src/s2/model.rs`, lines 378 to 381, floors the scale at 1e-8), and the read adds `r·Wo` to the block output (`memory.rs`, line 298). The test `a_zeroed_read_leaves_the_logits_of_the_base` (`memory.rs`, lines 2220 to 2242) checks it to 1e-6, in f32 and without QAT; a bit-exact test under QAT at the nano size is §11 item 1's.
- **Why.** With the base frozen and the memory zeroed (S = 0, so r = S·q = 0), the file computes its base's outputs whatever `Wo` has become: the language tasks and the games cannot lose when the memory is empty, by construction. What remains to measure is what a written memory does to the games, which is H2.
- **Bar:** K1's bar.
- **Reported, deciding nothing:** S90, S95, and the windows of §3.1.

### 3.5 K5, the preservation term (only on the route that needs it)

K5 runs only if K3 fails and R\* passed K1's bar (§3.6).

- **The term.** On every stateless batch (all the records: decisions, distill, games, numeric) and on every frame of the game sequences of phase B (`flappy`, `rescue`, `dodge`, `doom`; not `recall`, not `rule110`), the loss adds λ · KL(p_base ‖ p_θ) at the last exit, temperature 1, mean over the rows and the frames, with λ = 1.0 (§12, C9). p_base is the base checkpoint `runs/s6-nano-g3/decide/last` answering each frame alone with its ternary weights (as under QAT), without a gradient; p_θ is the model being trained, with its memory in the sequences. **Both arms carry the term**, the control too, so that H2 still compares the memory with the same training without it. Needs code (§11 item 6).
- **What.** From the final checkpoint of R\*'s seed-0 run in stage 1 (K1's or the confirmation's), 1,000 steps of phase B's mix (§4.3), seed 0, twice: with the term (K5a) and without it (K5b). The same batches.
- **Bar, both on K5a:**
  1. Flappy's training loss does not rise: the mean `seq_step_loss` of the last 20 `flappy` steps is at most that of the first 20;
  2. closed-loop Flappy on the check seeds is at least 0.98 times the base's: `sky-train s2-probe <checkpoint> --game flappy --from-seed 8000 --episodes 50 --passes 1` for K5a's end and for the base.
- **Why the second condition.** In v3, the training loss on the game sequences did not separate the arms: over the last 20 `flappy` steps of phase B, the memory arm read 0.384, 0.378 and 0.372 and the control 0.380, 0.377 and 0.371 (seeds 0, 1, 2; `runs/s6-sleep3/nano/*/b/log.jsonl`), while the closed-loop test read 15.87 against 71.05. Condition 1 alone would have passed v3. Seeds from 8000 are none of v3's (5000), v4's test (6000), the s6-nano gate's (`sky-arena`'s default, 7) or the trajectories' (2026, 101); they are read for K5 only, and K5b's end and its start checkpoint are read on them too, as reported.
- **Rule.** If K5a passes, v4 trains everything with the term. If it fails, v4 stops here and reports.

### 3.6 What stage 1 decides, with no choice left

| K3 | R\* passed K1's bar | v4 |
|---|---|---|
| passes | either | **the frozen route**: stage 2 trains R\* with the base frozen; the control is the base itself (§4.1) |
| fails | yes | K5; if it passes, **the preserved route**: stage 2 trains R\* with everything trained and the term of §3.5, in both arms |
| fails | yes, K5 fails | **v4 stops** and reports stage 1 |
| fails | no | **v4 stops** and reports stage 1 |

The order: K1 (seed 0, then seed 1), K2 (i) to (iv), the confirmation if any factor is adopted, K3, K5 if its route is taken. Each check's reading and verdict are committed before the next one starts.

### 3.7 Device, duration, memory

**Device today.** The trainer runs on the GPU unless `--cpu` is given (`crates/sky-train/src/main.rs`, lines 28 to 31 and 722 to 724: MLX's default device becomes the CPU). `scripts/train-sleep3.sh c5` forces `--cpu` at `nice -n 19`; `s2-memcheck` (`memcheck.rs`, line 616) and `s2-probe` (`probe.rs`, line 328) always run on the CPU. **This draft proposes the GPU for stage 1, with nothing else training** (§12, C1; §13, question 1). Every check is in steps, so the device is a cost and not a variable of the result, and stage 2 runs on the GPU too, so the S values that set its cap come from the same device.

**Step times** (working seconds, `log.jsonl`, windows of 50 sequence steps, stateless steps included):

| | GPU, measured (v3) | CPU |
|---|---:|---:|
| nano memory arm, phase A | 2.47 s per step (292 steps in 12.03 min) | about 21 s per step (23 s per sequence step), **not measured** |
| nano memory arm, phase B | 1.23 s per step (877 in 18.02 min) | about 12 s per step, not measured |
| nano control, phase A / B | 1.73 / 0.70 s per step | |
| pico, 8 sequences, C5 (`--no-compile`, `--memory-off-eval 4`) | | 3.1 to 3.9 s per sequence step, measured |

The CPU estimate scales C5's measurement by the GPU's ratio between the sizes: a pico sequence step of 16 on the GPU takes 0.77 s against about 3.5 s for 8 on the CPU, a ratio near 9, applied to nano's 2.57 s per sequence step. It could be off by half either way; the first 50 sequence steps of K1 measure it.

| check | runs × steps | GPU, working minutes | CPU, working hours (estimate) |
|---|---|---:|---:|
| K1 | 2 × 1,310 | 2 × 54 | 2 × 7.5 |
| K2 | 4 × 654 | 4 × 27 | 4 × 3.8 |
| confirmation | 2 × 1,310 | 2 × 54 | 2 × 7.5 |
| K3 | 2 × 1,310 | 2 × 45 to 54 (no base gradient: about 0.8 times K1, if §11 item 1 skips it) | 2 × 6 to 7.5 |
| K5 | 2 × 1,000 phase-B steps, and 3 Flappy probes of 50 episodes | about 45, plus the probes on the CPU | about 6.5 |
| **stage 1, at most** | | **about 8 working hours** | **about 65 working hours** (40 to 85) |

Under the machine's brake (duty 0.5, 45:15), C5 took about 2.4 times its working time in wall clock; a CPU stage 1 would take six to seven days.

**Memory.** On the CPU, a nano sequence step of 16 in parts of 2 peaks at 1.7 GB of MLX memory at N = 1 (commit `ce20cb7`, measured on the CPU at the nano size), next to about 5.1 GB of loaded stateless records (v3 amendment 3): about 7 GB for the process. On the GPU, the six v3 nano runs with `--sleep-chunk 2` ran without a stop by the memory guard (limit 13 GB for the largest `sky-train` process, `scripts/auto-brake.sh`, line 39); amendment 3 estimated 7.5 to 9 GB. K3 needs no more than K1; K5 adds the base's parameters (6.5 MB) and a forward pass without gradient.

## 4. Stage 2: the nano runs, only if stage 1 allows

### 4.1 Arms

- **The memory arm**: R\*, seeds 0, 1 and 2, the frozen base on the frozen route, the term of §3.5 on the preserved route.
- **The control**: on the preserved route, the same training without memory (no auxiliary loss, the term of §3.5), seeds 0, 1 and 2, each seed training its memory seed's steps, phase by phase. **On the frozen route, a model with a frozen base and no memory has nothing to train: the control is the base checkpoint itself**, `runs/s6-nano-g3/decide/last`, read once on each seed set and paired with each of the three memory seeds (§13, question 4).

### 4.2 Phase A, ended by a rule

- **The rule.** Phase A ends at the first `recall` step at which `ask_acc` pooled over the last 100 `recall` steps is at least **0.98**. The trainer saves `last`, writes its final line and stops (§11 item 5).
- **The cap.** S95max is the largest S95 of the two stage-1 runs of the recipe stage 2 trains (the frozen route: K3's two seeds; the preserved route: K1's or the confirmation's). The cap is 1.5 × S95max `recall` steps, converted to the run's budget as `--steps` = ⌈1.5 × S95max / 0.765⌉; the schedule (`wsd`, `crates/sky-train/src/optim.rs`, lines 177 to 186) is laid over that budget, so a run ended by the rule ends on the plateau of its rate.
- **At the cap.** Phase A ends there, phase B runs as for any seed, and nothing is retried or extended. The report gives that seed's pooled `ask_acc` over its last 100 `recall` steps and says that it ended at the cap. v2's stop (ask-frame cross-entropy above 1.336 over the last 50 `recall` steps of phase A) still applies to every memory seed: if it holds for one, no phase B runs and v4 reports a failure in-sample.
- **The control** trains its memory seed's phase-A steps (`--steps`, the same number), on the same batches.

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
- **H2, the games do not lose:** on the arena of §5 at the chosen N, no game more than 2% under the control (the base on the frozen route).
- **H3, the language tasks do not lose:** `typed_decisions_bench` with a zero memory at most 0.01 under the control. On the frozen route the two files compute the same outputs, so the difference must be exactly 0: any other value is a defect to explain before anything is reported.
- **Z, the memory zeroed** (new, the pilot's rule of v2 §5.3 (2)): `recall` at N = 1 with the memory zeroed minus the control ≤ 0.03. On the frozen route it must be exactly 0, as for H3.
- **The goal, announced and not gated:** `recall` at N = 1 at least 0.90.

**Gate** (v1 §8, v2 §7, plus Z). `s6-sleep4-nano` (seed 0, the chosen N) is published under that name only if all of these hold: the s6-nano gate on the exported seed-0 file (`typed_decisions_bench` above 0.398, ECE ≤ 0.08, no game more than 5% under s3-pico in the gate's arena); H0, H1, H2, H3 and Z; `sky-infer`'s parity with the training code on the golden memory sequences, natively and in WASM. On the frozen route, the exported file is the base's weights plus the memory; its calibration is computed again by `s2-calibrate` with a zero memory, as in v3.

## 7. What each outcome means

- **Stage 1 stops: no run of R\* reaches K1's bar and K3 fails.** On its own training batches, within 1,000 `recall` steps, the nano memory trained as v3 or with the one adopted change does not reach 0.95, nor with a frozen base. v3's gap to the pilot is then not a matter of budget alone. No stage-2 GPU minute is spent. The next step, by a separate decision and file, is the interference reading of the diagnosis (§2 (d)): the key overlap at the nano size (`s2-memcheck --c2-only`, `--memory-trace`), then a write that can replace (the slot form).
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
6. **New dev and test seeds** (V7).
7. **The bars.** K1's bar (0.95) and the rule of phase A (0.98) were set after the pilot's in-sample 0.9994 and v3's nano curves (0.35 to 0.48) were seen; the goal of 0.90 after v3's test (0.4892) and the pilot's dev2 (1.0000).

Every report of v4 states this section. The disclosure of v3 §5 (and v2 §10) stands.

## 9. Cost

One MacBook (Apple M5, 16 GB); `scripts/auto-brake.sh` stops a training whose process holds more than 13 GB four readings in a row. Nothing else trains on the GPU during a stage-1 check or a stage-2 run.

- **Stage 1:** about 8 working hours of GPU at most (§3.7), or about 65 working hours of CPU at `nice -n 19` (40 to 85, not measured at this size), six to seven days of wall clock under the brake.
- **Stage 2, GPU** (v3's step times): phase A at 2.47 s per step for the memory arm, so 1,000 `recall` steps (1,307 steps) take about 54 working minutes and the cap 1.5 times S95max of that rate; phase B 41 minutes. For S95max = 700 `recall` steps (cap 1,050, 1,373 steps, 57 minutes), the memory arm takes at most about 1 hour 40 per seed, 4 hours 50 for three seeds; the preserved route's controls (1.73 and 0.70 s per step) add at most 3 hours 10, and the term a forward pass of the base per step (not measured). The frozen route has no control run.
- **Readings, CPU:** v3's export, test reading and gate took about 11 hours of wall clock for six models (`runs/all/sleep3-gate.log`, 07:39 to 18:54 on 17/11); v4 adds the zeroed readings and the dev3 readings: about 13 to 15 hours.
- **Disk:** about 25 MB per run of one phase (`last`, `pre-qat`, the log); stage 1 at most 12 runs, stage 2 at most 12, the probes about 10 MB, six GGUF files about 4.5 MB: under 1 GB. The disk had 12 GiB free on 17/11/2026.

## 10. Forbidden

- Any v4 computation before this file is frozen and the code of §11 is written and tested.
- Reading a dev3 seed before stage 1 has ended and its outcome is committed; reading a test seed before every stage-2 run has ended and N is chosen; reading any seed of v3, v1, dev2, the smoke or the browser (§5) for a v4 decision.
- A check beyond K1, K2 (i) to (iv), the confirmation, K3 and K5; a factor beyond the four; a second confirmation; running a check again except after a technical failure, which restarts the same run with the same seed.
- Choosing R\*, the route or the cap by anything but the rules of §3 and §4.2.
- Extending phase A beyond the cap, retrying a seed, or adding a seed.
- Training on episodes of the validation or test splits (`split_for_seed`).
- Changing a bar of §3, §4.2 or §6 after a reading it concerns.
- Publishing under `s6-sleep4-nano` a file that fails the gate.

**What a failure would mean for each bar.** K1's bar: v3's recipe does not reach 0.95 on its own batches within about 1,000 `recall` steps at the nano size. K2's rule: no single factor of the four explains a quarter of the slowness. The confirmation: the adopted factors do not add up. K3's bar: the memory cannot reach 0.95 on its batches without the base moving. K5: the term does not keep Flappy where the base plays it. The rule of phase A ending at the cap: the run did not reach 0.98 on its batches in 1.5 times stage 1's longest time to 0.95. H0 to H3, Z and the s6-nano gate: as v1 §7, §8 and v2 §5.3 say.

## 11. Code to write before any run

| # | file | what | needed by |
|---|---|---|---|
| 1 | `crates/sky-train/src/optim.rs`, `crates/sky-train/src/main.rs`, `crates/sky-train/src/s2/train.rs` | **A frozen base**: an `OptConfig` field (e.g. `base_lr_mult`, 1 by default and then not written, as `mem_lr_mult`) and its flag (`--freeze-base`); a frozen parameter gets no update and no weight decay, and the global clip is computed over the trained parameters only (today `Optimizer::update` sums every gradient into the norm, lines 271 to 285, so the base's gradients would still shrink the memory's update). Best, `train.rs` differentiates the memory's parameters only, so that the base's weight gradients are not computed. Tests: after some steps every non-`mem.*` tensor is bit-identical to its start; a fresh memory model under QAT at the nano config gives its base's logits bit for bit. | K3, the frozen route |
| 2 | `crates/sky-train/src/s2/memory.rs`, `crates/sky-train/src/s2/train.rs` | **The auxiliary readings per layer**: `seq_frames` (lines 932 to 939) sums `aux_ce` and `aux_right` over the layers; return them per layer and log `aux_acc_layers` next to the mean (lines 2458 to 2461). | the readings of §3.1; K2 (ii) |
| 3 | `crates/sky-train/src/s2/config.rs`, `crates/sky-train/src/s2/seq.rs`, `crates/sky-train/src/s2/memory.rs`, `crates/sky-train/src/main.rs` | **The auxiliary loss as T2 was written**: a choice of layers (all, or the last) and of where the head reads (the mean over the valid tokens, or the door token of the ask frame). `seq.rs` carries no token position today (`SeqData`, lines 128 to 141, and the batch's `Asks`); the door token is found as `memcheck.rs` finds it (`Encoder::door`, lines 313 to 335: the one token that changes when the door does) and must be carried per ask frame through `FrameBatch` and its parts (`--sleep-chunk`). The normalisation (`memory.rs`, lines 962 to 976) divides by the layers that carry a head, and so does the log. | K2 (ii) |
| 4 | `crates/sky-train/src/optim.rs` | Not required: a clip of the memory's parameters and of the rest by their own norms, the alternative of K2 (iv) (§12, C6). | only if the lead takes that alternative |
| 5 | `crates/sky-train/src/s2/train.rs`, `crates/sky-train/src/main.rs` | **The stop rule of phase A**: `--stop-recall-acc 0.98 --stop-recall-window 100` ends the run (save `last`, the final validation, a final line saying `stopped_by_rule`) at the first `recall` step where the pooled `ask_acc` of the window reaches the bar; the loop's own end is `keep_going` (`crates/sky-train/src/session.rs`, lines 98 to 100). | stage 2 |
| 6 | `crates/sky-train/src/s2/train.rs`, `crates/sky-train/src/s2/model.rs`, `crates/sky-train/src/s2/memory.rs`, `crates/sky-train/src/main.rs` | **The preservation term**: `--keep-close-to <checkpoint> --keep-close-weight λ`; load the base's parameters once, answer each stateless batch and each game-sequence frame with them (no memory, ternary, no gradient), add λ · KL(p_base ‖ p_θ) at the last exit to the decision loss and to the sequence loss of the four games; the KL of `exit_kd_loss` (`model.rs`, lines 1222 onwards) is the pattern. Tests: the term is zero when the model is the base; its gradient reaches the base's parameters and, in sequences, the memory's. | K5, the preserved route |
| 7 | `scripts/train-sleep4.sh` (new) | `k1`, `k2`, `conf`, `k3`, `k5`, `nano` (phase A by the rule with the cap, phase B in steps, the control per seed or the base), `export`, `eval` (dev3, the test of §5, the zeroed readings), `gate`; `--steps` and `--stop-after` as §3.1; the GPU unless `CPU=1`. | everything |
| 8 | `scripts/s6-sleep4-report.py` (new) and its tests | the windows of 50, S90, S95 and the excess ratio from `log.jsonl` (a run stopped by `--stop-after` has no final line: the reader must not require one); the rules of §3.3 to §3.6 written to `runs/s6-sleep4/stage1.json`; the cap of §4.2; the check of identical batches (§3.1); the seeds of §5 and the refusal of v3's; Z in the gate; the frozen route's pairing with the base. | everything |

Possible today without code: K1 (every flag exists), K2 (i) (`--memory-aux-weight 2`), K2 (iii) (`--adam-lr 0.002`), K2 (iv) as a value (`--clip-norm`), the shorter checks (`--stop-after`), the device (`--cpu`), phase B in steps (`--steps`), the zeroed readings (`s2-probe --memory-off`), the new seeds (`--from-seed`).

## 12. Choices (the first value of each line is the one this draft proposes; the alternatives are recorded)

| | choice | proposed | alternatives | why the proposal |
|---|---|---|---|---|
| C1 | device of stage 1 | **the GPU, nothing else training** | the CPU at `nice -n 19`, as v2's and v3's checks | in steps, the device changes the cost, not the result; about 8 working hours against about 65 (six to seven days of wall clock); stage 2 runs on the GPU, so the S values that set its cap come from the same device |
| C2 | the length of K1 | **1,310 steps, about 1,002 `recall` steps** | an exact stop at 1,000 `recall` steps (code) | possible today; every check shares one schedule |
| C3 | the windows of S90 and S95 | **trailing 50 `recall` steps for S90, 100 for S95, pooled over the asks** | per step; 100 for both | 6,400 asks give a standard error near 0.004 at 0.90; S95 uses the window of K1's bar and of the rule of phase A |
| C4 | the seed of K2 | **0** | 1; the slower seed of K1 | v3's seed 0 was the median of the three (the answer left chance at `recall` step 161, between 99 and 171) |
| C5 | the length of K2 | **K1's schedule cut at 654 steps (`--stop-after`), about 500 `recall` steps** | 500 `recall` steps under their own schedule | the same batches and the same rate at every step as K1's seed 0: S90 compares one factor |
| C6 | the clip of K2 (iv) | **the median of `grad_norm` over `recall` steps 51 to 250 of K1's seed 0, rounded up to one decimal** | the median of the first 50 steps; no clip (`--clip-norm 0`); the memory and the base clipped apart (code) | in v3 the first-50 median was 0.92 to 2.27 and the norm then rose to 6 to 11: the first-50 rule tests nothing; 51 to 250 is where v3's steps were clipped |
| C7 | (i) and (ii) both adopted | **the lower S90; on a tie, (ii)** | always (i), which needs no code | the faster one; (ii) is what was pre-registered |
| C8 | the control of the frozen route | **the base itself, read once** | a trained control as in v3 | a frozen model without memory trains nothing; v3's control lost at Flappy against its own base |
| C9 | the preservation weight | **λ = 1.0, KL at the last exit, temperature 1, in both arms** | λ = 0.3; λ = 3; KL at every exit | the same scale as the frame cross-entropy next to it; the gate reads the last exit; both arms so that H2 still isolates the memory |
| C10 | where the term applies | **every stateless batch and every game-sequence frame of phase B** | the stateless batches only; the game frames only | H2's loss is closed-loop on games, where the memory is written; the stateless records hold the typed states of H3 |
| C11 | K5's bar | **Flappy's training loss does not rise, and closed-loop Flappy on check seeds at least 0.98 times the base's** | the training loss alone | v3's training losses did not separate the arms (0.384 against 0.380 over the last 20 `flappy` steps, seed 0) while the test did (15.87 against 71.05) |
| C12 | K5's check seeds | **`s2-probe --game flappy --from-seed 8000 --episodes 50`** | 30 episodes; the gate's arena (seed 7) | used by nothing else; 50 episodes for a 2% comparison; the gate's seeds stay the gate's |
| C13 | phase B | **2,000 steps** | 877 (v3); 3,000 | the pilot's 2,051 and 407 `recall` steps; 400 expected |
| C14 | the change of phase | **as v3** | a lower peak in phase B; a longer warm-up (code); the optimiser carried over (code), each after a check K4 | §4.4: the drop was recovered, the rule removes the jump of the rate, and a third change needs a third check |
| C15 | the control's steps on the preserved route | **each control seed trains its memory seed's phase-A steps** | the steps of memory seed 0, as v3 | phase A ends by a rule, so the seeds differ |
| C16 | dev seeds | **dev3, 500 seeds ≡ 19 mod 20 from 3,000,000,000** | 1,000 episodes | never read by anything; v3's size |
| C17 | test seeds | **`recall` and `rule110` from 6,000,000,000 (1,000 episodes), the arena of H2 from 6000 (200 per game)** | | never read by anything; v1's sizes |
| C18 | `--memory-off-eval 4` in the checks | **yes** | no | C5's control on the same batch, for a few per cent of the time |
| C19 | the in-sample stop of v2 | **kept for every memory seed of stage 2** | dropped | it costs nothing and stage 1 makes it unlikely |

## 13. Open before the freeze (for the lead)

1. **The device of stage 1** (C1): the GPU, about 8 working hours, against the CPU at `nice -n 19`, about 65 (six to seven days), the practice of v2 and v3. The CPU's figure is an estimate; no nano step was ever timed on the CPU.
2. **K2 (iv)** (C6): the median over the first 50 steps, as first drafted, would put the clip near 1.0 in v3's logs. This draft takes steps 51 to 250.
3. **K5** (C11, C12): Flappy's training loss alone would have passed v3's memory arm. The second condition reads closed-loop Flappy on check seeds during stage 1, which is not a training sequence.
4. **The frozen route's control** (C8): the base itself, so the three control seeds collapse into one file and H3 and Z become checks of the construction (exactly 0). The alternative keeps v3's comparison with a trained control, which does not isolate what the memory does to a base that does not move.
5. **The preserved route's control** (C9): this draft gives it the preservation term too.
6. **The s6-nano gate's arena** (seed 7, 30 episodes), read by every gate including v3's, stays the gate's fixed protocol.
7. **The cost of stage 2** (§9): up to about 8 working hours of GPU on the preserved route, about 5 on the frozen route, for S95max near 700 `recall` steps; S95max is not known before stage 1.

## 14. Amendments

None yet.
