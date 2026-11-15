# s6-sleep pilot: why recall is identical at N = 1, 2, 4

Code read on `main` (a303bf0), on 15/11/2026, after the pilot. Runs read: `runs/s6-sleep.log`, `runs/s6-sleep/pilot.json`, `runs/s6-sleep/pilot/p{1..4}/{log.jsonl,last,pre-qat,dev}`, `runs/s6-sleep/seq/recall.jsonl`. Nothing was built and nothing ran on the GPU. Checkpoints and optimizer states were read with a pure-Python safetensors reader (`nice -n 19`).

Pilot model (`runs/s6-sleep/pilot/sleep.json`): s6-pico, d = 96, 1 state block × 4 loops, **1 head** (hd = 96), so one memory layer `mem.0` with a state of 96 × 96.

## 1. How N gets from the command line to the forward pass

**Training.**
- `--sleep-passes` (default `1,2,3,4`) is defined at `crates/sky-train/src/main.rs:420-422` and `--sleep-full` at `:425`.
- Each sequence batch draws N: `train.rs:1871` `a.sleep_passes[rng.below(..)]`. It builds `Sleep { passes, every: 8, full }` (`train.rs:1872-1876`) and calls `sleep_step`, which calls `memory::seq_loss` (`train.rs:426-446`).
- In `memory.rs:289-302`, every 8 frames (or at a pause) `sleep()` runs. `sleep()` is at `memory.rs:244-261`: `passes − 1` replays of the chunk, and with truncation the memory is detached before the last pass (`:253-255`).
- Each replay goes through `state_written` (`model.rs:944-952`). That function runs the state encoder reading the memory and returns what it writes, and `after_writes` (`memory.rs:177-193`) applies the write. The memory then enters the next wake frames (`memory.rs:298`). So N changes `S`, `S` is read by every later frame, and the read changes the logits.
- The log confirms how each arm was configured:
  - P1 drew N in [1, 2, 3, 4] with truncation (`s6-sleep.log:21`);
  - P2 had N = 4 (`:208`);
  - P3 had N = 4 through every pass (`:396`);
  - P4 had no memory (`:583`).

**Probe.**
- `s2-probe --passes 1,2,4` is defined at `main.rs:545-547` and passed through at `main.rs:1014-1034`.
- `probe.rs:270-280` builds one `MemPolicy` per N, with `how.passes = n` (`probe.rs:119-123`).
- `observe` (`probe.rs:143-158`) sleeps when `chunk.len() >= 8` (or at `g.pause()`), using the same `memory::sleep`.
- `decide` (`probe.rs:160-206`) runs `wake` on the current memory and keeps the written memory for the next frame (`next`, consumed in `observe` at `:149-151`).
- A recall episode has 64 frames, so there are 7 sleeps per episode. For rule110 there is one sleep, at the pause before frame 1.

**Does N reach a tensor that reaches the answer? Yes.**
- Code: `S` after the sleep is read at every block application through `memory::read` (`memory.rs:128-135`, called at `model.rs:786-789`). Its output flows into the exits and the fusion head (`model.rs:990-992`).
- Unit test: `probe.rs:387-418` asserts that the logits differ between N = 1 and N = 4 at every frame after the pause.
- Saved runs: P1's rule110 per-episode results differ between N = 1 and N = 2 on 22 of 500 episodes, and between N = 1 and N = 4 on 19 (`p1/dev/rule110.json`). On recall, all 4,000 per-ask results are identical across N for P1, P2 and P3. So N changes the logits, but never enough to flip an argmax on an ask frame (see section 3).

## 2. Write, carry, read, and whether gradients reach the memory

- **Write.**
  - `model.rs:786-787` records the block output at each application, before the read, so the last application wins (`written[l]`).
  - `memory.rs:139-173` applies one gated delta-rule step, `S ← αS + (1/n)Σ β_i (v_i − αS k_i) k_iᵀ`. Its shapes and transposes are correct: `err = v − k·(αS)ᵀ`, `delta = (err ⊙ w)ᵀ·k`, masked to the valid tokens.
- **Carry.**
  - Training: `seq_loss` keeps `mem` across the 64 frames and resets it only once per batch (`memory.rs:290`).
  - Probe: the memory is reset only when an episode starts (`probe.rs:131-139`, `:144-147`).
  - Nothing resets it between the write and the read, and nothing hides the memory tokens (the read is added to every token; the write is masked by `valid` only).
- **Read.** `memory.rs:128-135`: `x + merge(S·q)·Wo` after every application of the block.
- **What is trained.** Every `mem.0.*` parameter is trained, with AdamW at the same rate as the base. No freezing is visible.
  - Initialisation (`memory.rs:52-69`): `out` (Wo) = 0; q, k, v ~ N(0, 0.02); `alpha` and `beta` weights = 0, so β = σ(0) = 0.5 and α = σ(0 + 6) = 0.9975 (`ALPHA_BIAS`, `:38`).
  - The four projections are fake-quantised with the straight-through estimator under QAT (`model.rs:69`, `:434-455`, `:384-386`). The STE passes gradients to the float weights, including the all-zero Wo.
- **The checkpoints show every memory parameter moved from its initial value.** Mean |w|, `pre-qat` (= init) → `last`:

| param | init | P1 | P2 | P3 |
|---|---|---|---|---|
| mem.0.out (Wo) | 0 | 0.078 | 0.044 | 0.077 |
| mem.0.q / k / v | ≈0.0145 | 0.079 / 0.085 / 0.084 | 0.040 / 0.043 / 0.051 | 0.068 / 0.067 / 0.077 |
| mem.0.alpha | 0 | 0.043 | 0.037 | 0.045 |
| mem.0.beta | 0 | 0.086 | 0.055 | 0.064 |
| mem.0.norm | 1 | 1.016 | 0.986 | 1.008 |

- **Gradient size (P1, Adam second moment at the last step, √mean v).**
  - Base block weights get about 3e-4.
  - Memory projections get 1.6e-5 to 3.2e-5, 10 to 20 times less.
  - β gets 1.3e-3.
  - α gets **about 5e-12**, effectively no gradient (σ'(6) ≈ 0.0025, times a mean over tokens).
  - So gradients reach the memory, but weakly. The decay gate is in effect frozen at its initial value, which is harmless for retention (0.9975 per frame).
- **Training loss: the memory never helped, even on the training data.**
  - All four arms consume the same batches, and their `seq_loss` traces (36 points each, every 50 sequence steps) overlap.
  - Mean difference from the control P4: P1 −0.0007 (sd 0.0068), P2 +0.0042, P3 +0.0015.
  - Solving recall would lower that mixed average by about 0.4 × (8/64) × ln 4 ≈ 0.069, and even the 1-to-4-frame bucket alone would show up.

## 3. Is the recall probe solvable, and is its answer key right?

- **Game.**
  - The fact is on screen for one frame only (`recall.rs:99-109`). The ask frame shows only `ask <door>` (`:105`, test `:339-354`).
  - The target is the colour of the last show of that door (`:111-119`; plan at `:147-180`, which keeps the door from being shown again between its show and its ask).
  - The test `:288-317` replays the plan and checks colour and delay. The test `:319-336` shows the oracle scores every ask.
  - Probe scoring: `step()` (`:122-141`) marks an ask correct when `choice == 1 + colour`. The probe passes the argmax over `canonical_options()`, which is in key order `wait, red, green, blue, yellow` (`common.rs:104-109`; the test `recall.rs:367-372` checks the key order). The key is consistent.
- **Training data.** The first episode of `runs/s6-sleep/seq/recall.jsonl` checks out:
  - its options are shuffled (for example, `hold` at index 1);
  - frames that ask nothing have the wait/hold target;
  - `ask a` at t = 5 targets red, shown for `a` at t = 2;
  - `ask c` at t = 18 targets red, shown at t = 13;
  - `ask b` at t = 19 targets blue, shown at t = 8.
- **Sequence data and colour words.**
  - The loader keeps file order and rebuilds sequences by (seq, seq_t) (`seq.rs:132-203`); `sample_frames` keeps rows aligned frame by frame (`seq.rs:275-299`).
  - The tokenizer is byte-level BPE, so the colour words stay distinct tokens or subwords.
  - The task requires memory and can be solved with it. The answer key is right.
- **What the models actually answer.** Comparing the per-ask correctness of the runs (4,000 asks each, roughly 1,000 correct each) gives:
  - P2 and P3 are both right on only **15** asks, where about 250 would be expected if they answered independently at chance. P1 and P3 overlap on 279; P1/P4 and P2/P4 on 433 and 486.
  - This is the signature of models that answer each ask with an almost constant, model-specific colour (a prior), not with information from the show frame.
  - A near-constant argmax is why a small memory-driven change in the logits never flips an ask, and so why recall is identical at N = 1, 2 and 4.
  - rule110 tells the same story: P2, P3 and P4 all score 0.5423, the same constant answer.

## 4. What the unit tests check

- `memory.rs:364-388`: parameter counts and which weights are ternary.
- `memory.rs:390-420`: a fresh model is its base, a zero memory adds nothing, and a written memory changes the logits under random weights.
- `memory.rs:422-448`: replays shrink the delta-rule error `v − S·k`. This is a storage property at the level of keys and values.
- `memory.rs:450-528`: `seq_loss` equals hand-chained wake frames, N matters only after a sleep, and every memory parameter gets a nonzero gradient, truncated or not, under QAT.
- `probe.rs:387-418`: the reader reads the memory and the sleep changes the logits.

**No test checks that a written fact can be read back into a correct answer.** None hand-sets weights (or trains a tiny model) so that a show frame followed by k idle frames and an ask frame gives the right colour through the memory. The tests prove the wiring (values change, gradients flow), not that retrieval works.

## Conclusion: (B)

The mechanism is wired and trained, and the model does not use it. No defect was found in the code that would make the pilot uninformative. The evidence:

1. **N reaches the computation.** The code path is traced above. The probe unit test asserts that N changes the logits, and P1's rule110 changes with N on about 20 of 500 episodes.
2. **Write, carry and read are correct.** Nothing resets the state, nothing masks the memory, nothing is detached beyond the pre-registered truncation, and the probe places the fact before the delay.
3. **Gradients reach every memory parameter, and the parameters moved.** Wo went from exactly 0 to a mean |w| of 0.04–0.08, and q, k, v grew 3 to 6 times. The exception is the α gate, whose gradient is about 5e-12.
4. **The memory arms' training loss equals the control's within ±0.004.** Recall alone would have been worth about 0.07, so the memory was not learned even in-sample.
5. **The probe models answer almost constant colours** (P2/P3 jointly right on 15 of 4,000 asks against about 250 under independence), so the argmax never depends on the memory. Hence the identical scores at every N, and the chance-level scores.

**Caveat for the report.** This is a negative result about this recipe (pico, about 1,840 sequence steps of which about 40% are recall, one head, a zero-initialised read-out, memory gradients 10 to 20 times smaller than the block's), not a proof that the mechanism cannot work. The missing test of section 4 is cheap to add and would separate "cannot represent" from "did not learn": with hand-set Wq = Wk on a door feature, Wv on a colour feature and Wo into the fusion's view, a show frame then 8 idle frames then an ask should give the right colour.
