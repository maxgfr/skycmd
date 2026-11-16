# s6-sleep v3 (s6-sleep3-pico, s6-sleep3-nano): pre-registration

**Status: frozen on 16/11/2026, after the diagnosis of `docs/s6-sleep3-diagnosis.md` and before any run of this version.** No GPU run of v2 or v3 exists. No dev2 seed and no test seed has been read by any version. The code of the two changes below exists and is tested (it was written for the diagnosis, every switch off by default); the scripts of this version are not written yet.

This file is short on purpose: **v3 is v2 (`docs/s6-sleep2-prereg.md`) with two changes to the training and one more check.** Everything v2 wrote stands unless this file changes it: the question (§1), the model (§4), the pilot and its decision rule (§5), the nano runs (§6), the gate (§7), the demo (§8), what is forbidden (§12). Names: `s6-sleep2-*` becomes `s6-sleep3-*`; outputs go to `runs/s6-sleep3/`.

## 1. Why a third version

v2 ended at its check C4: at initialisation the memory holds a door's colour (C1, C2), and 300 CPU steps of training did not learn to use it (ask-frame cross-entropy 1.3867 against 1.3873 with the read held at zero; chance is ln 4 = 1.3863). The diagnosis measured why, on train sequences only:

1. **The write gate closes and cannot reopen.** `β = σ(wβ·x)` reads the raw block output (norm about 96) and is trained at ten times the base rate: its mean goes from 0.96 at step 7 to 3e-7 at step 35, the memory empties, and every memory gradient falls with it (1e-11 by step 52).
2. **A live gradient is not enough.** With the gates held at their initial values the memory gradient stays near 1e-3 for 300 steps and the loss stays at chance (1.3865): the only gradient the memory receives comes through a read-out that starts at zero and an answer that does not yet depend on it.
3. **What follows the memory is not the obstacle.** With the colour asked written in the ask frame's own state, the same base answers in about thirty steps.
4. **With held gates and an auxiliary loss that asks the memory for the colour, training uses the memory.** A linear head decodes the colour asked from what the memory retrieves at the ask frames; its loss is added to the training loss. The auxiliary accuracy rises first (0.25 to above 0.8), then the answer leaves chance (around step 170), and the ask-frame cross-entropy over steps 250 to 300 is 0.7231, 0.0374 and 0.0149 in three runs on two seeds, against 1.3900 for the same training with the read held at zero. Over steps 550 to 600 of a 600-step run: 0.0007, accuracy 1.0000 on 2,880 asks; the same weights with the memory off: accuracy 0.226. Accuracy does not depend on the delay (0.997, 0.998, 0.997, 0.991 for delays 1-4, 5-8, 9-16, 17-32) nor on the door having been shown with several colours (0.996 against 0.996).

Everything in item 4 is **in-sample**: train sequences, each step's batch before its update, on the CPU, read during a diagnosis that looked at many settings. It says the mechanism can be trained. It is not a result on unseen episodes; that is what this version measures.

## 2. The two changes

- **T1, the gates are held.** `wα` and `wβ` stay at their initial value, zero, for the whole training (`--mem-gate-lr-mult 0`): `β = 0.5` for every token and `α = σ(6) = 0.9975`, constants. The memory is then a fast-weight memory with a fixed write strength and a fixed slow decay, which is how the first fast-weight memories were trained (Ba et al. 2016 use fixed coefficients). The gate weights stay in the file, at zero.
- **T2, an auxiliary recall loss in training only** (`memory_aux: true`, `--memory-aux-weight 1`): at each ask frame of `recall`, a linear head (4 outputs, one per colour) reads what the last memory layer retrieves at the door token, before the read-out; its cross-entropy against the colour asked is added to the loss with weight 1, in phase A and in phase B. The head is not exported and plays no role at inference; `rule110` and the other games have no auxiliary loss.

Nothing else changes: v2's memory (`memory_qk: shared`, value form `frame`, as C2 adopted), `mem.*` at ×10 (the gates excepted), the base, the mixes, the budgets, the seeds, the probe.

**Not used:** the slot write (`memory_write: slot`) written during the diagnosis and never run on the base; gates that learn (at the base rate they survive but reach 0.47 where held gates reach 0.99 in the same 300 steps).

## 3. One more check before any GPU run

C1 to C3 of v2 stand as read (they do not depend on T1 or T2). C4 is replaced by:

- **C5, a CPU training that uses the memory, twice.** C4's setting with T1 and T2 and **600 steps**, seeds 0 and 1, each with `--memory-off-eval 4` (the same weights answer one step in four again with the memory off). Read over steps 550 to 600, on each step's batch before its update. **Pass, on both seeds:**
  1. ask-frame cross-entropy (the deep-supervision measure) ≤ 0.30 and last-exit accuracy on ask frames ≥ 0.90;
  2. the same weights with the memory off: cross-entropy ≥ 1.30 and accuracy ≤ 0.30;
  3. auxiliary accuracy ≥ 0.95.
- **What C5 is.** A reproduction, by the script of this version, of what the diagnosis saw on the same train sequences (seed 0 was run to 600 steps there, seed 1 to 300). Its bars were set after those numbers were seen. It can catch a broken script or an unlucky seed; it proves nothing about unseen episodes. 600 steps and not 300 because the step at which the answer leaves chance moved by tens of steps from one launch to the next.
- **No retry.** If C5 fails on either seed, no GPU run takes place and the report says so.

## 4. The pilot, the nano runs, the gate

v2 §5, §6 and §7 as written, with T1 and T2 in arm M and in the sleep arm, and these precisions:

- **The control arm** has no memory and therefore no auxiliary loss, as in v2.
- **The stop after phase A** (ask-frame cross-entropy above 1.336 over the last 50 `recall` steps) stands. Phase A's number of sequence steps is reported; the diagnosis needed about 170 to 190 of them on the CPU before the answer moved.
- **The decision rule of §5.3 is unchanged**, on dev2 seeds, ternary weights, last exit, closed loop: M at N = 1 at least 0.10 above the control on `recall` with the bootstrap bound above 0, and M with its memory zeroed within 0.03 of the control. Reported with it, deciding nothing: accuracy by delay bucket and by one colour against several colours, with the memory and with it zeroed.
- **Sleep is untested.** Everything measured so far is at N = 1. Phase B draws N in {1, 2, 3, 4} as v2 wrote; whether sleep passes help or hurt a memory with held gates is what H1 reads, and §11 of v2 says what each outcome means.
- **Export.** The exported graph carries the memory (ternary `Wqk`, `Wv`, `Wo`; f32 norm and gates) and not the auxiliary head. `sky-infer` must match the training code on golden sequences before any export is evaluated, as §7 asks.

## 5. Disclosure

- This is the third version of the same idea on the same games, and its two changes were found by looking at training curves on train sequences. The readings that decide (dev2, then the test seeds) have never been made by any version.
- The auxiliary loss gives the model the colour asked as a training target at the ask frames. That is the same information the answer's own loss carries (the label is the colour); it reaches the memory by a shorter path. It is not available at inference.
- The gates are constants in this configuration. A reader should not take "learned write and decay gates" from v1's description as a property of v3.
- The literature review used for the diagnosis is `docs/research-fast-weight-memory.md`.

## 6. Cost

C5: two CPU runs of about an hour each at `nice -n 19`. The pilot: about 35 working minutes of GPU. The nano runs: six runs of 30 working minutes. Nothing else runs on the GPU during a budgeted run.

## 7. Amendments

**Amendment 1 (16/11/2026, after C5 was read and before any GPU run of v3; no dev2 seed and no test seed has been read). C5 passed on both seeds; the code of this version; where the GPU runs are launched from.**

- **C5: pass.** `scripts/train-sleep3.sh c5`, two CPU runs of 600 steps at `nice -n 19`, read by `scripts/s6-sleep3-report.py` over steps 550 to 600, pooled over the asks, each step's batch before its update (`runs/s6-sleep3/checks/c5.json`, the two logs and commands next to it):

  | | seed 0 | seed 1 | bar |
  |---|---:|---:|---:|
  | recall steps / asks read | 45 / 2,880 | 45 / 2,880 | |
  | ask-frame cross-entropy | 0.0001 | 0.0007 | ≤ 0.30 |
  | last-exit accuracy on ask frames | 1.0000 | 0.9997 | ≥ 0.90 |
  | same weights, memory off: steps / asks | 11 / 704 | 11 / 704 | |
  | same weights, memory off: cross-entropy | 2.0663 | 3.9504 | ≥ 1.30 |
  | same weights, memory off: accuracy | 0.2386 | 0.2628 | ≤ 0.30 |
  | auxiliary accuracy | 1.0000 | 0.9997 | ≥ 0.95 |
  | step at which the answer left chance (reported) | 156 | 240 | |

  As §3 says, this is a reproduction on train sequences with bars set after the diagnosis: it shows that the script of this version trains what the diagnosis trained, and nothing about unseen episodes. On seed 1 the answer left chance at step 240, later than in the diagnosis (159 to 183).
- **The code of this version, written after the freeze and before any GPU run.** `scripts/train-sleep3.sh` (c5, pilot, report, nano, export, eval, gate), `scripts/s6-sleep3-report.py` with its tests, the memory in `sky-infer` (read, write, sleep passes, memory off) with golden sequences checked against the training code natively and on wasm32, `sky-arena --memory-passes`, and the colours each `recall` ask's door had been shown with in `s2-probe` (for the split of §4). The exported graph has no auxiliary head (a golden test asserts it). Every place where this file or v2's left a choice open is listed in `docs/s6-sleep3-readings.md`, written before any GPU run. Two of them are stated here because they touch a condition: C1 to C3 of v2 are read through v2's `verdict.json` (C2 taken by the stated deviation of v2's amendment 4, C3 as the sanity check of v2's amendment 1), since the raw `c2.json` and `c3.json` say `"pass": false`; and the pilot does not use `--memory-off-eval`, which would spend working minutes of a minutes-budgeted run (the memory zeroed is read by the probe, as v2 §5 wrote).
- **Where the runs are launched from.** C5 ran in a worktree of the repository, with a trainer built at commit `3470be8` (no training code changed after it); its two runs were paused by the machine's cooling brake (45 minutes of work, 15 of rest), which changes their duration and not their steps. The GPU runs are launched from another worktree at the commit of this amendment, with the inputs of the main checkout linked or named (`runs/s6lr/softmax-adamw-x2/decide/last`, `runs/s6-sleep/seq/`, `data/`, and v2's check files through `V2_CHECKS`), the brake file of the machine, and nothing else on the GPU. A smoke run (`SMOKE=1`, throwaway prefix, seeds outside dev2 and test) comes first; its numbers are not looked at.
- **A risk, written before the pilot.** Phase A of the pilot has 8 working minutes, batches of 16 and 15% of `rule110`; if the answer has not left chance when phase A ends, v2's stop rule (ask-frame cross-entropy above 1.336 over the last 50 `recall` steps) stops the pilot, and that is reported as the result of the pilot, not retried.

**Amendment 2 (16/11/2026, after the pilot was read on the dev2 seeds and before any nano run; no test seed has been read). The pilot rule says "works": the nano runs take place.**

- **Phase A did not stop.** Arm M, phase A: 8.0 working minutes, 663 steps, of which 613 sequence steps (526 on `recall`); the ask-frame cross-entropy over its last 50 `recall` steps is **0.0014**, against v2's stop bar of 1.336; the answer left chance at step 167. The risk written in amendment 1 did not occur.
- **The runs** (`runs/s6-sleep3/pilot.json`, `runs/all/sleep3-pilot.log`): M phase B 12.0 working minutes, 2,051 steps; the control C did M's steps, phase by phase (663 and 2,051). Seed 0, one run each, no restart.
- **The rule of v2 §5.3, on the dev2 seeds** (500 episodes from seed 2,000,000,000, ternary weights, last exit, closed loop, 4,000 asks):

  | | `recall` score |
  |---|---:|
  | M, N = 1 | **1.0000** |
  | M, N = 2 | 1.0000 |
  | M, N = 4 | 0.9980 |
  | M with its memory zeroed, N = 1 | 0.2567 |
  | control C (no memory) | 0.2542 |

  M at N = 1 minus the control: **+0.7458, 95% bootstrap interval [0.7323, 0.7593]** (needed: at least 0.10 with the lower bound above 0). M with its memory zeroed minus the control: +0.0025 (needed: at most 0.03). **Both conditions hold: the verdict is "works".** By delay (1–4, 5–8, 9–16, 17–32 frames, 1,000 asks each) M is at 1.000 on every bin and the control between 0.246 and 0.270; with one colour to remember or several, M is at 1.000 on both.
- **What the pilot does not show.** `rule110` is at 0.5443 for every arm and every N, the control included: nothing was learned on it, and it is not part of the rule. Sleep passes add nothing on `recall` at this size (1.0000 at N = 1, 0.9980 at N = 4): H1 reads this on the nano runs. The pilot is the pico size on dev2 seeds; no test seed and no nano model has been read.
- **Deviations and disclosures.**
  - The smoke that precedes the pilot (not a result) was killed once at the owner's request, when the machine froze under other work, and started again from scratch; nothing of it is kept.
  - The trainer was built at the worktree's commit at the launch (`a0305c5`), not at the commit of amendment 1: it adds the router's code (`sky-data route fresh`, the embedding in `sky-infer`) and nothing in `crates/sky-train` (the diff between the two commits is empty there).
  - The pilot ran in the machine's gentlest brake mode (half duty, 45 minutes of work and 15 of rest; two cooling pauses), with CPU work of other models at the same time; budgets are in working minutes, so this changes the duration and not the steps.
- **Next, unchanged:** the six nano runs of v2 §6 (the sleep arm with T1 and T2, and the control, seeds 0, 1, 2), the choice of N on dev2, the export, the test seeds read once, the gate.
