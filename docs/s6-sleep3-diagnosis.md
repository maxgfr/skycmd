# s6-sleep v2 at C4: why the memory stops receiving gradient, and what would make training use it

Written on 15/11/2026, after `docs/s6-sleep2-prereg.md` ended at C4, and continued on 16/11/2026 (§0 items 5 to 8, the additions to §1, §6 from the row `a-aux-held` on, §6.1, §6.2, §7 and §8). Everything here ran on the CPU (`sky-train --cpu`, `nice -n 19`, one run at a time), on `recall`'s **train** sequences only (`runs/s6-sleep/seq/recall.jsonl`, v1's); no dev or test seed of the probes was read, and nothing ran on the GPU. This file is a diagnosis, not a pre-registration: a third version needs its own.

**The setting** is C4's, unchanged (`scripts/train-sleep2.sh`, `c4`): 300 steps from the pico base `runs/s6lr/softmax-adamw-x2/decide/last` with v2's memory (`runs/s6-sleep2/checks/sleep2.json`: shared `qk`, form `frame`), 90% of the steps batches of 8 `recall` sequences of 64 frames, rate 2e-3 with `mem.*` at ×10, N = 1, QAT from the first step, seed 0. `scripts/s6-sleep3-diag.sh <name> [flags]` runs one experiment; `scripts/s6-sleep3-trace.py` reads the logs. The number reported for a run is C4's: the ask-frame cross-entropy (the loss's own deep-supervision measure) averaged over steps 250 to 300; chance is ln 4 = 1.3863.

## 0. Summary

1. **The gradient dies because the write gate closes, and a closed sigmoid gate cannot reopen.** `β_i = σ(wβ·x_i)` reads the raw block output, whose norm is about 96, and `wβ` is trained at the ×10 rate. Between steps 7 and 35 the mean `β` over the written tokens goes from 0.96 to 3e-7 (`‖wβ‖` from 0.18 to 0.95). With `β = 0` nothing is written: `‖S‖` at the ask frames goes from 4.8 to 7e-5, the reading `r = S·q` with it, and every memory gradient is proportional to `S` or to `β(1 − β)`. Step 52 is only the first step at which C4's log reported gradient norms (every 50 sequence steps); the collapse is over by step 35 (§2).
2. **Holding the gates open keeps the gradient alive and changes nothing else.** With `wα`, `wβ` held at zero (`β = 0.5`), the memory gradient stays near 1e-3 for 300 steps, the read adds about 13% of the hidden state's norm, and the ask loss is 1.3865 (§3). A live gradient is necessary, not sufficient.
3. **What follows the memory is not the obstacle.** With the colour asked written in the ask frame's own state and the read held at zero, the same base reaches an ask loss under 0.01 at step 70 (§4). The base learns to answer from a colour it can see in about thirty steps once its first transient is over.
4. **What the memory hands over at the start is too faint to be learned from** (§5): keys of different tokens are nearly collinear (mean squared cosine 0.73 between a shown door and an idle frame, C2), so C2's 0.39 is 0.49 at delays 1 to 4 and chance beyond 8; one show writes 6% of a value in which the colour is one token in nine; and half of the asks concern a door that was shown with two or more colours, which this write rule superposes.
5. **With the gates held and an auxiliary loss, training uses the memory** (16/11/2026, §6.1). Nothing else changes: v2's memory as it is, C4's 300 CPU steps. The auxiliary loss asks one linear head to decode the colour asked from what the memory retrieves at the ask frames. C4's reading (ask-frame cross-entropy over steps 250 to 300, chance 1.3863) is **0.7231 on seed 0 and 0.0374 on seed 1**; on seed 1 the last exit answers **99.4%** of the asks of those steps, and the same weights without their memory answer 23.8% of them (cross-entropy 2.99). The same training of seed 0 with its read held at zero is at 1.3900, while its memory holds the colour all the same (auxiliary accuracy 0.99). A 600-step run of seed 0 goes to 0.0007 over its last 50 steps, every one of 2,880 asks right, and does not repeat the first run of seed 0: it reads 0.0149 over steps 250 to 300. The speed at which the answer arrives varies from run to run; that it arrives does not, in three runs.
6. **The delays and the doors shown again are served alike** once the model is trained: 0.999, 0.995, 0.992 and 0.991 at delays 1 to 4, 5 to 8, 9 to 16 and 17 to 32; 0.996 for a door shown with one colour and 0.992 for a door shown with several (seed 1, 3,008 asks of train sequences). §5's weaknesses slow the start; they are not a ceiling.
7. **What was missing was a reason, not a path.** §3's run had a live gradient and stayed at chance. With the auxiliary loss the memory first learns to return the colour (auxiliary accuracy over 0.8 at step 172 and 190) while the answer is still at chance, and the answer starts to use the read at about the same step (176 and 183). Holding the gates matters as much: with the auxiliary loss and the gates free at the base's rate, they survive and the memory is used, at 1.2078 and 47% of the asks right instead of 99%.
8. **Everything is read on train sequences**, on the batch of a training step before its update, in three runs on two seeds and their controls. No dev or test seed was read, no probe was played, nothing was exported. The recommendation is in §7, what is not known in §8.

## 1. The instrumentation

`s2-decide --seq-log --memory-trace [N]` (off by default) logs, for one sequence step in N and **before** the update: the norm of the gradient and of the weights of every `mem.*` tensor, the norm and the share of zeros of the ternary form of each projection, and, per memory layer, from a second pass over the same batch without gradient (`memory::trace`): the write strength `β` (mean and largest over the valid tokens of every frame), the decay `α` (mean and smallest), `‖S‖` of the memory each frame reads (over every frame, and over the ask frames), and at the valid tokens of the ask frames `‖x‖` (the block output), `‖r‖` (what is retrieved) and `‖r·Wo‖` (what the read adds). The trace of a fresh model is tested (`memory.rs`, `the_trace_of_a_fresh_memory_...`).

The traced baseline (`runs/s6-sleep3/d0-trace`, 90 steps of the 300-step schedule) reproduces C4's first attempt: same losses at steps 1 to 7 to four decimals, memory gradient norm 6.8e-11 at step 52 (1.4e-11 in `runs/s6-sleep2/checks/c4/mem`; the CPU runs are not bit-identical).

**Added on 16/11/2026**, for §6.1 and after:

- **The answer itself, split.** Every `--seq-log` line of a `recall` step now also holds `ask_acc`, the share of the ask frames the last exit answers right (chance 0.25, the four colours), and `ask_splits`, the counts behind it by delay bucket (1 to 4, 5 to 8, 9 to 16, 17 to 32 frames since the last show of the asked door) and by the number of distinct colours that door had been shown with before the ask (one, or several: a door shown again with another colour). The delay and the colours are read from the states of the sequence (`seq::ask_splits`, tested).
- **The control on the same weights.** `--memory-off-eval N` answers one sequence step in N a second time before its update, with the same parameters and no memory (every frame alone, which is the read held at zero; `memory::asks_read_off`, tested against a run with `--memory-read-off`), and logs it under `off`. A model that answers through its memory is then read twice on the same batch: with it, and without it.
- **The key overlap** in the trace: the squared norm of the mean over the frames of a frame's key, 1 when every frame writes under the same key and 1/64 when the keys of the frames are orthogonal.
- `scripts/s6-sleep3-trace.py splits <run> [lo hi]` and `curve <run> [width]` read these.

All of these are measured as C4 measures: on the batch of a training step, before that step's update, on train sequences. A run of 300 steps draws about 2,200 sequences (some 275 sequence steps of 8, with replacement) out of 8,950, so about nine in ten of the sequences a step is read on were not seen before by that run; they are still train sequences.

## 2. Where the gradient dies

| step | ask CE | β mean | β max | ‖wβ‖ | ‖S‖ at asks | ‖r‖ | ‖r·Wo‖ | grad Wo | grad Wqk | grad Wv | grad wβ | grad wα | α min |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1.575 | 5.0e-01 | 5.0e-01 | 0.00 | 1.1e+00 | 1.0e+00 | 0 | 1.6e-02 | 0 | 0 | 0 | 0 | 0.9975 |
| 2 | 1.698 | 5.0e-01 | 5.0e-01 | 0.00 | 1.1e+00 | 9.9e-01 | 1.4e-02 | 4.6e-02 | 1.9e-03 | 4.0e-03 | 1.1e-03 | 1.7e-04 | 0.9975 |
| 4 | 1.653 | 9.2e-01 | 9.8e-01 | 0.06 | 2.6e+00 | 2.4e+00 | 1.5e+00 | 7.9e-02 | 6.1e-02 | 1.6e-02 | 8.8e-03 | 7.1e-05 | 0.9998 |
| 7 | 1.671 | 9.6e-01 | 1.0e+00 | 0.18 | 4.8e+00 | 3.8e+00 | 6.3e+00 | 6.2e-02 | 8.0e-02 | 3.8e-02 | 9.0e-03 | 9.0e-07 | 1.0000 |
| 9 | 2.430 | 6.9e-01 | 1.0e+00 | 0.27 | 5.1e+00 | 4.0e+00 | 3.8e+00 | 4.1e-01 | 2.1e-01 | 2.6e-01 | 6.1e-01 | 7.2e-07 | 1.0000 |
| 10 | 4.265 | 2.9e-01 | 1.0e+00 | 0.30 | 4.6e+00 | 3.8e+00 | 2.4e+00 | 1.6e-01 | 2.6e-02 | 8.4e-02 | 9.1e-02 | 0 | 1.0000 |
| 12 | 3.714 | 4.4e-02 | 9.0e-01 | 0.39 | 2.4e+00 | 1.9e+00 | 3.5e+00 | 4.9e-02 | 3.0e-02 | 4.2e-02 | 3.7e-01 | 0 | 1.0000 |
| 16 | 2.393 | 1.2e-01 | 1.0e+00 | 0.55 | 3.4e+00 | 2.6e+00 | 8.0e+00 | 1.5e-01 | 1.2e-01 | 1.6e-01 | 3.9e-01 | 0 | 1.0000 |
| 20 | 2.566 | 5.1e-03 | 7.3e-01 | 0.68 | 6.3e-01 | 5.1e-01 | 1.9e+00 | 7.8e-03 | 7.1e-03 | 2.3e-02 | 3.3e-01 | 0 | 1.0000 |
| 23 | 1.922 | 6.0e-02 | 1.0e+00 | 0.76 | 2.6e+00 | 2.1e+00 | 2.4e+00 | 3.4e-02 | 1.2e-02 | 6.0e-02 | 1.4e-01 | 0 | 1.0000 |
| 26 | 1.641 | 1.0e-02 | 8.5e-01 | 0.82 | 9.9e-01 | 8.1e-01 | 7.0e-01 | 6.7e-03 | 1.4e-03 | 1.6e-02 | 2.5e-02 | 0 | 1.0000 |
| 29 | 1.655 | 1.4e-03 | 8.1e-01 | 0.87 | 1.8e-01 | 1.4e-01 | 2.1e-01 | 8.1e-04 | 1.2e-04 | 1.0e-03 | 1.3e-02 | 0 | 1.0000 |
| 31 | 1.839 | 2.1e-04 | 2.0e-01 | 0.90 | 3.9e-02 | 3.0e-02 | 5.7e-02 | 3.4e-04 | 2.0e-04 | 1.5e-03 | 1.7e-02 | 0 | 1.0000 |
| 33 | 1.583 | 2.8e-06 | 4.9e-03 | 0.92 | 5.9e-04 | 4.8e-04 | 6.9e-04 | 2.0e-06 | 4.2e-07 | 2.0e-06 | 1.1e-05 | 0 | 1.0000 |
| 35 | 1.554 | 3.1e-07 | 2.3e-04 | 0.95 | 6.8e-05 | 5.6e-05 | 7.3e-05 | 4.5e-07 | 8.6e-08 | 2.7e-07 | 3.4e-06 | 0 | 1.0000 |
| 40 | 1.572 | 1.2e-09 | 7.0e-07 | 1.00 | 3.1e-07 | 2.7e-07 | 3.5e-07 | 1.1e-09 | 7.2e-10 | 7.0e-10 | 1.6e-08 | 0 | 1.0000 |
| 52 | 1.378 | 1.9e-11 | 2.6e-09 | 1.07 | 5.7e-09 | 4.7e-09 | 1.1e-08 | 1.7e-12 | 1.4e-12 | 1.5e-12 | 6.8e-11 | 0 | 1.0000 |
| 84 | 1.390 | 1.8e-11 | 1.9e-09 | 1.11 | 3.3e-09 | 2.5e-09 | 4.2e-09 | 3.4e-13 | 2.4e-13 | 6.4e-13 | 1.8e-11 | 0 | 1.0000 |

(`runs/s6-sleep3/d0-trace/log.jsonl`; `scripts/s6-sleep3-trace.py table`. `‖x‖` at the ask tokens stays between 89 and 102.)

**The tensor and the step.** The factor that goes to zero is `β`. Every other factor of the gradient stays of order one: `‖Wo‖` grows to 9.3 at step 35 (its ternary form to 7.8, with 22% of zeros), `‖Wqk‖` to 9.6, `‖Wv‖` to 9.0, the norm's gain stays at 9.7 of 9.8, `‖x‖` is unchanged. `S`, `r` and `r·Wo` fall with `β`, by the same four orders of magnitude over the same steps (29 to 35), and the gradients of `Wo`, `Wqk` and `Wv` follow `S`. `wβ`'s own gradient carries the factor `β(1 − β)` and falls last. From step 36 on nothing in the memory can change `β` back: its gradient is 1e-8 and falling, under Adam's ε.

**Why the gate, and why so fast.** The gate's input is the raw block output, of norm about 96, where every projection of the memory reads a normalised state. A change of `wβ` of norm 0.04, which is what one Adam step at the ×10 rate does here, can move the pre-activation `wβ·x` by 3.8; over the 28 steps from 7 to 35 the mean pre-activation went from about +3 to about −15. No other memory weight can silence the read that fast: `Wo` would have to lose a norm of 9 at the same rate. Between steps 9 and 31 the gradient of `wβ` is the largest of the memory's in 9 steps out of 10 (0.61 against 0.41 for `Wo` at step 9, 0.33 against 0.008 at step 20).

**What pushes it shut.** Steps 8 to 40 are the base's own transient on `recall`, identical in the run with the read held at zero (`runs/s6-sleep2/checks/c4/zero`): it first answers `hold` everywhere (ask CE 4.1 to 4.3 at step 10 in both runs), then learns to spread its answer over the four colours on ask frames (1.39 from step 60 on). During these steps the read adds 2 to 10 in norm to a hidden state of 96, the loss of every frame moves by large amounts for reasons that have nothing to do with the memory, and the cheapest way to remove a perturbation the base has no use for is to stop writing. The gate first opens (0.96 at step 7, during the six steps of warm-up) and then closes.

**The decay gate dies too, on the other side:** `α` is 1.0000 from step 5 and its gradient is exactly 0 from step 10. It no longer forgets, and no longer learns.

**Candidates ruled out by the same trace.**

- *QAT starting at step 52, or a warm-up ending there:* no. QAT is on from step 0 (`--qat-frac 1.0`, `qat_start` logged at step 0) and the warm-up is the first 2% of the steps (6 steps, `optim.rs`, `wsd`). 52 is the 50th sequence step, the first at which `--seq-log` reports group norms.
- *A ternary read-out stuck at zero:* no. After the first update every entry of `Wo` has the same size and its ternary form is dense (0% of zeros at step 2, `‖Wo‖` = 0.019); it then holds 8% to 22% of zeros with a norm of 7 to 8.
- *Weight decay pulling `Wo` back to zero:* no, `‖Wo‖` grows throughout.
- *The LayerNorm before the projections:* no, its gain moves from 9.80 to 9.61 in norm.
- *The loss weighting:* the ask frames are 8 of 64 frames and the loss at the plateau is 0.173 = 1.386 / 8: every other frame is answered with a loss near zero, so the whole loss comes from the asks; `dL/dx` at the read is not what vanishes (the blocks keep a gradient norm of 1e-2 at step 52 where the memory has 1e-11).
- *Truncation:* C4 runs at N = 1, where nothing is detached between the write and the ask.
- *The ×30 rate and float projections* (C4's two retries) die the same way: memory gradient 0 exactly at steps 159 and 216, and 1.7e-9 at step 52 (`runs/s6-sleep2/checks/c4-x30`, `c4-float`).

## 3. Holding the gates: a live gradient, and still chance

`--mem-gate-lr-mult 0` holds `wα` and `wβ` at zero, so `β = 0.5` and `α = σ(6)` for the whole run (`runs/s6-sleep3/e1-gates-frozen`). Then:

| step | ask CE | grad Wo | grad Wqk | grad Wv | ‖S‖ at asks | ‖x‖ | ‖r‖ | ‖r·Wo‖ |
|---|---|---|---|---|---|---|---|---|
| 48 | 1.404 | 5.1e-03 | 2.0e-03 | 2.6e-03 | 17 | 111 | 11 | 23 |
| 96 | 1.388 | 1.3e-03 | 4.2e-04 | 7.2e-04 | 10 | 107 | 5.9 | 12 |
| 150 | 1.386 | 1.2e-03 | 4.2e-04 | 7.7e-04 | 10 | 126 | 7.2 | 14 |
| 210 | 1.386 | 1.3e-02 | 2.9e-03 | 1.0e-02 | 12 | 152 | 9.5 | 20 |
| 300 | 1.382 | 1.1e-03 | 2.0e-04 | 6.0e-04 | 11 | 156 | 7.9 | 20 |

The memory gradient is eight to nine orders of magnitude above the baseline's, the memory is written and read, the read adds 10% to 15% of the hidden state's norm, and the ask loss over steps 250 to 300 is **1.3865** against 1.3873 for the zeroed run. So the dead gate explains the vanished gradient and does not by itself explain the loss at chance: with the gate open, the gradient that reaches the memory carries no usable direction.

## 4. The ceiling: the base uses a colour it can see within seventy steps

`scripts/s6-sleep3-diag.sh oracle`: the train ask frames are rewritten so that the state itself names the colour asked (`show none colour red ask a noise 33` for an ask answered by red), the read is held at zero, everything else is C4's. Ask CE: 1.57 at step 1, 3.95 at step 14 (the same transient), 1.34 at step 42, 0.59 at step 51, 0.03 at step 64, **under 0.01 from step 70** (the run was stopped at step 81; `runs/s6-sleep3/oracle`). The fusion head matches a colour word in the state to the option of that colour in some thirty steps. Whatever the memory must deliver, the part of the model that follows it learns fast when the colour is there as a token of the model's own space.

## 5. Why the memory's signal is faint at the start

These are readings of files that already existed, and arithmetic on the write rule; none of them is a new experiment.

- **Keys do not separate tokens.** C2 (`runs/s6-sleep2/checks/c2.json`): the mean squared cosine between the key of a shown door and the keys of an idle frame is 0.73. Every frame writes all its tokens at `β = 0.5`, so each idle frame writes over the door's slot. C2's 0.390 is the mean of **0.494 at delays 1 to 4 and 0.286 at delays 5 to 8**; at 9 to 16 and 17 to 32 it is 0.2525 and 0.2535, chance. Half of the asks of an episode are at a delay above 8.
- **One show writes little.** `β/n` = 0.5/8: 6% of the value per write, and under form `frame` the colour is one token in the mean of about nine. Against a full token for the oracle of §4.
- **A re-shown door is superposed, not overwritten.** In the train episodes (first 3,000, 24,000 asks), the asked door had been shown once before the ask in 37.5% of the asks, twice in 34.8%, three times or more in 27.7%; with one colour in 48.2%, **two or more different colours in 51.8%**. The second show adds `0.0625·(v₂ − 0.0625·v₁)` to a slot that holds `0.0625·v₁`: the two colours end with weights 0.059 and 0.0625. C1 never played a re-shown door.

So at initialisation the read hands the base, for a quarter of the asks, a weak statistical trace of the colour, and for the rest nothing; turning that into an answer needs the keys, the gates and the values to change first, and the only gradient that could change them comes through a read-out that starts at zero and a base that is busy with its own transient.

## 6. Experiments

Each run is C4's setting with one change; "memory" and "zeroed" are C4's two numbers (ask-frame cross-entropy over steps 250 to 300; ln 4 = 1.3863). A change that touches only the memory's weights or their optimiser leaves the zeroed run what it was (with the read held at zero no memory weight receives a gradient), so those rows share C4's zeroed run, 1.3873; the CPU runs are not bit-identical from one launch to the next (§1), so that number is good to about three decimals. The auxiliary loss is not such a change: it trains the memory, and the block through the reading, whether or not the read is held at zero. Its rows have their own controls: a training with the read held at zero, and the same weights answering the same batches without their memory (§1).

| run | change from C4's first attempt | memory | zeroed | what the trace shows |
|---|---|---|---|---|
| `runs/s6-sleep2/checks/c4` | none (`mem.*` at ×10, ternary) | 1.3867 | 1.3873 | `β` → 3e-7 by step 35, memory gradient 1e-11 (§2) |
| `runs/s6-sleep2/checks/c4-x30` | `mem.*` at ×30 | 1.3876 | 1.3879 | memory gradient exactly 0 at steps 159 and 216 |
| `runs/s6-sleep2/checks/c4-float` | float memory projections (the read-out included) | 1.3871 | 1.3873 | memory gradient 1.7e-9 at step 52 |
| `runs/s6-sleep3/e1-gates-frozen` | gates held at their start (`--mem-gate-lr-mult 0`) | 1.3865 | 1.3873 (C4's) | `β` = 0.5, memory gradient about 1e-3 throughout, read 11% to 13% of `‖x‖` (§3) |
| `runs/s6-sleep3/oracle` | no memory in play: the colour asked is in the ask frame's state, read held at zero | n/a | under 0.01 from step 70 (stopped at step 81) | the ceiling of §4, not a candidate |
| `runs/s6-sleep3/c-aux-gates-frozen` | gates held, and the auxiliary loss at weight 1 (`memory_aux`, `--memory-aux-weight 1`) | cut at step 84: 1.3931 over steps 71 to 84 | not run | auxiliary accuracy 0.25 until step 50, then 0.34, 0.37 and 0.41 over steps 51 to 60, 61 to 70 and 71 to 84; see the note under the table |
| `runs/s6-sleep3/a-aux-held` | the same run in full (16/11/2026): gates held, auxiliary loss at weight 1, seed 0 | **0.7231** (last exit 0.6905) | **1.3900** (`r3-aux-held-readoff`, below) | auxiliary accuracy 0.26, 0.41, 0.59, 0.77, 0.985 and 0.996 over the six windows of 50 steps; the ask loss is 1.394, 1.393, 1.347, 1.020 and 0.721 over the last five (§6.1) |
| `runs/s6-sleep3/r3-aux-held-readoff` | the control of the row above: the same training with the read held at zero (`--memory-read-off`), seed 0 | n/a | 1.3900 (last exit 1.3908; accuracy 0.2516) | auxiliary accuracy 0.993 over steps 251 to 300: the memory holds the colour, the answer cannot read it |
| `runs/s6-sleep3/r1-aux-held-seed1` | gates held, auxiliary loss at weight 1, **seed 1**, with the split logging and the reading without the memory | **0.0374** (last exit 0.0307; accuracy 0.9940) | same weights without the memory, same steps: 2.9920 (accuracy 0.2383) | auxiliary accuracy 0.996 over steps 250 to 300; splits by delay and by colours in §6.1 |
| `runs/s6-sleep3/r2-aux-held-600` | gates held, auxiliary loss at weight 1, seed 0, **600 steps** | 0.0149 over steps 250 to 300 (accuracy 0.9971); **0.0007** over steps 550 to 600 (accuracy 1.0000 on 2,880 asks) | same weights without the memory: 2.2513 (accuracy 0.2409) over steps 250 to 300; 3.1717 (0.2259) over steps 550 to 600 | no plateau above zero; 0.0192 and accuracy 0.9957 over steps 300 to 600 (§6.1) |
| `runs/s6-sleep3/r4-aux-gates-base-rate` | auxiliary loss at weight 1, gates **free at the base's rate** (`--mem-gate-lr-mult 0.1`), seed 0 | 1.2078 (last exit 1.1984; accuracy 0.4691) | same weights without the memory: 1.4377 (accuracy 0.2643) | the gates survive: mean `β` 0.02 to 0.06 with a largest `β` of 1.00, smallest `α` down to 0.05 then 0.30 to 0.89; auxiliary accuracy 0.608; accuracy 0.54 at delays 1 to 4 and 0.36 at 17 to 32 (§6.1) |

**Not run:** a read-out started small and non-zero (`--memory-out-init`), the read-out alone in float (`--memory-out-float`; the float retry of C4 already covers it), no weight decay on the memory (`--mem-no-decay`), the auxiliary loss with gates free at ×10, a copy-path start in float (`--memory-copy-init`), and the slot form of §6.2. The switches exist and are off by default. §2 says what to expect of the first three: none of them touches the gate, and `Wo` was never the factor that vanished. After §6.1 none of them is needed for the memory to be used; the copy-path start and a non-zero read-out could shorten the hundred steps during which the colour is in the reading and the answer does not use it, and that was not tried.

**Note on the row `c-aux-gates-frozen`** (the last row on 15/11/2026). At about 21:30 on 15/11/2026 the session that ran these experiments lost its access to `~/Documents` (every open returned "Operation not permitted"); the run was started at 21:24 and ended at 21:36, at step 84 of 300, without its final checkpoint. Its log was read on 16/11/2026, once the access was back. It holds 80 sequence steps. The accuracy of the auxiliary head (the colour asked, decoded linearly from what the memory retrieves at the ask frames; chance 0.25) is 0.25 to 0.27 over each ten-step window until step 50, then 0.335 (steps 51 to 60), 0.368 (61 to 70) and 0.408 (71 to 84), with its cross-entropy going from 1.39 to 1.25. The ask loss over the same windows is 1.401, 1.393 and 1.393: at chance. So in 84 steps the memory starts to hold and return the colour once a loss asks it to, and the answer does not yet use it. This is one cut run on train sequences, with no zeroed control: it says the auxiliary loss is worth its full run, not that it works.

### 6.1 Held gates and the auxiliary loss: the memory is used

**The configuration.** C4's setting with two changes and no change to the model's form: the gates held at their start (`--mem-gate-lr-mult 0`: `β = 0.5`, `α = σ(6)` for the whole run) and the auxiliary loss at weight 1 (config `memory_aux: true`, `--memory-aux-weight 1`: the colour asked, decoded by one linear head from the mean over the tokens of what the memory retrieves at the ask frames). Everything else is C4's first attempt: v2's memory (shared `qk`, form `frame`, v1's write rule), ternary projections under QAT from step 0, a read-out started at zero, `mem.*` at ×10, 300 steps, N = 1. The exact command is in §7.

**Two seeds** (16/11/2026; windows of 50 steps; ask-frame cross-entropy in the loss's own deep-supervision measure, chance 1.3863; accuracy of the last exit on the ask frames, chance 0.25):

| steps | seed 0: ask CE | seed 0: auxiliary accuracy | seed 1: ask CE | seed 1: ask accuracy | seed 1: auxiliary accuracy | seed 1, same weights without the memory: ask CE, accuracy |
|---|---|---|---|---|---|---|
| 1-50 | 1.974 | 0.262 | 1.976 | 0.183 | 0.249 | 1.947, 0.172 |
| 51-100 | 1.394 | 0.414 | 1.401 | 0.250 | 0.273 | 1.396, 0.261 |
| 101-150 | 1.393 | 0.585 | 1.386 | 0.294 | 0.524 | 1.391, 0.244 |
| 151-200 | 1.347 | 0.769 | 1.368 | 0.359 | 0.682 | 1.402, 0.241 |
| 201-250 | 1.020 | 0.985 | 0.826 | 0.718 | 0.913 | 1.693, 0.227 |
| 251-300 | 0.721 | 0.996 | 0.034 | 0.995 | 0.997 | 2.992, 0.238 |

(`runs/s6-sleep3/a-aux-held`, seed 0, run with the binary of 15/11, which did not log the accuracy or the reading without the memory; `runs/s6-sleep3/r1-aux-held-seed1`, seed 1. `scripts/s6-sleep3-trace.py curve <run> 50`.)

**C4's reading** (steps 250 to 300): **0.7231** for seed 0 (last exit 0.6905) and **0.0374** for seed 1 (last exit 0.0307), against C4's bar of 1.30 and against **1.3900** for the same training of seed 0 with its read held at zero (`runs/s6-sleep3/r3-aux-held-readoff`: last exit 1.3908, 25.2% of 3,136 asks right). C4 asked for 1.30 at most and 0.05 under the zeroed run; seed 0 is 0.667 under it. In that control the auxiliary loss trains the memory as before (auxiliary accuracy 0.25, 0.38, 0.59, 0.70, 0.979 and 0.993 over the six windows): the colour is in what the memory retrieves in 99% of the asks, and an answer that cannot read it stays at chance. The two seeds leave chance at the same moment (the first ten sequence steps with a mean ask loss under 1.36 start at step 176 and at step 183) and then fall at different speeds: over steps 241 to 250, where the rate starts to decay, seed 1 is at 0.24 and seed 0 at 0.90; over their last ten steps, seed 1 is at 0.0008 with every ask right, seed 0 at 0.42 and still falling. A run of 300 steps ends before seed 0 has arrived; the 600-step run below says where it goes.

**The order of events is the same in both runs.** First the base's transient (§2), unchanged. Then, from about step 50 (seed 0) or 100 (seed 1), the auxiliary accuracy rises while the ask loss stays at chance for some hundred steps: the memory learns to hold and return the colour because the auxiliary loss asks for it, through a gradient that needs neither the read-out nor the answer. When the colour is decoded in about four asks out of five (the first ten sequence steps with a mean auxiliary accuracy over 0.8 start at step 172 and at step 190), the answer starts to use the read (ask loss under 1.36 from step 176 and from step 183). §3's run had the same live gradient and no such signal, and stayed at chance: what was missing was not a path for the gradient but a reason, reachable from the start, to put the colour in the memory.

**The gain is the memory's.** In seed 1's run, one sequence step in four was answered a second time by the same weights without their memory, before the update (`--memory-off-eval 4`). Over steps 250 to 300 (3,008 asks with the memory, 768 without):

| | ask CE | accuracy | delay 1-4 | 5-8 | 9-16 | 17-32 | door shown with one colour | with several |
|---|---|---|---|---|---|---|---|---|
| with the memory | 0.0374 | 0.9940 | 0.999 | 0.995 | 0.992 | 0.991 | 0.996 (1,435 asks) | 0.992 (1,573 asks) |
| same weights, no memory | 2.9920 | 0.2383 | 0.245 | 0.234 | 0.234 | 0.240 | 0.254 (366) | 0.224 (402) |

Without its memory the trained model is at chance in accuracy and far above chance in cross-entropy (2.99 against 1.39): it answers with confidence a colour it no longer has. Each delay bucket holds a quarter of the asks (752 with the memory, 192 without).

**The delays and the doors shown again.** Nothing separates them at the end: 0.999 at delays 1 to 4 and 0.991 at 17 to 32; 0.996 for a door shown with one colour and 0.992 for a door shown with several (52% of the asks). On the way there the order is the one expected, short delays and single colours first: over steps 200 to 250 of seed 1 the accuracy is 0.751, 0.738, 0.708 and 0.657 by delay, 0.731 for one colour and 0.698 for several. So §5's weaknesses are real at the start and not structural: with the gates held at `β = 0.5` and a write that averages over the tokens, the trained keys and values serve a door shown again as well as any other. How they do it was not measured (§8). The key overlap of the trace (§1) goes from 0.88 at step 1 to 0.31 at step 45 and then stays between 0.34 and 0.56 (seed 1): training separates the frames' keys within the base's transient.

**The trace** (seed 1; seed 0 is alike): `β` = 0.500 and `α` = 0.9975 throughout, as held; `‖S‖` at the asks grows from 1.2 to 10.6, what is retrieved from 1.0 to 4.4, what the read adds from 0 to 15.9 for a hidden state of norm 126 (13%); the read-out's ternary form has a norm of 14.8 with 30% of zeros; the memory gradient stays between 5e-3 and 1 until the loss is near zero.

**The stateless records** (one step in ten) are not what these runs are about, and the trainer's own validation of them is too small to say much: 256 items of the validation split of those records (decisions, games, numeric; not the probes' seeds), read at step 0 and at the end, last exit, ternary. Accuracy 0.504 before and 0.512 after the 600 steps of seed 0; 0.598 and 0.523 for seed 1 (another draw of 256); 0.504 and 0.480 for the control with the read held at zero. No collapse; nothing finer can be read from them.

**Six hundred steps** (`runs/s6-sleep3/r2-aux-held-600`: seed 0, the same flags, `--steps 600`, so the rate is constant until step 480 and the warm-up is 12 steps instead of 6):

| steps | ask CE | ask accuracy | same weights without the memory: ask CE, accuracy | auxiliary accuracy |
|---|---|---|---|---|
| 1-50 | 1.982 | 0.161 | 1.888, 0.147 | 0.257 |
| 51-100 | 1.395 | 0.243 | 1.400, 0.234 | 0.431 |
| 101-150 | 1.401 | 0.252 | 1.398, 0.219 | 0.792 |
| 151-200 | 0.941 | 0.655 | 1.595, 0.247 | 0.996 |
| 201-250 | 0.051 | 0.995 | 2.155, 0.263 | 0.996 |
| 251-300 | 0.015 | 0.997 | 2.251, 0.241 | 0.999 |
| 301-350 | 0.045 | 0.989 | 2.798, 0.250 | 0.995 |
| 351-400 | 0.006 | 0.999 | 2.454, 0.259 | 1.000 |
| 401-450 | 0.022 | 0.997 | 3.226, 0.237 | 0.999 |
| 451-500 | 0.034 | 0.991 | 3.125, 0.258 | 0.995 |
| 501-550 | 0.005 | 0.999 | 3.167, 0.236 | 1.000 |
| 551-600 | 0.0007 | 1.000 | 3.172, 0.226 | 1.000 |

The ask loss does not stop at a plateau above zero: it reaches 0.05 by step 250 and stays between 0.0007 and 0.045 per window afterwards, with two small returns (steps 301 to 350 and 451 to 500, accuracy 0.989 and 0.991). Over steps 300 to 600, 17,088 asks: cross-entropy 0.0192, accuracy **0.9957**; by delay 0.997, 0.998, 0.997 and 0.991; 0.996 for a door shown with one colour (8,169 asks) and 0.996 for a door shown with several (8,919). Over the last 50 steps every one of 2,880 asks is right. The same weights without their memory, over steps 300 to 600 (4,288 asks): cross-entropy 2.99, accuracy 0.244.

**Three runs, three speeds.** This run is seed 0 again, and it does not repeat `a-aux-held`: its ask loss is under 1.36 from step 159 and under 0.1 from step 202, where `a-aux-held` was under 1.36 from step 176 and still at 0.42 at step 300; over steps 250 to 300 it reads 0.0149 (accuracy 0.9971) against 0.7231. The two differ by the length of the schedule (a warm-up of 12 steps against 6; the rate is the same from there to step 240) and by what differs between two launches on the CPU (§1). Seed 1 sits between them at the start (step 183) and is under 0.1 from step 252. So the order of events is stable over the three runs, the moment the answer starts to use the read is stable to some 25 steps, and the time it then takes to arrive goes from 40 steps to more than 120. A window fixed at steps 250 to 300 reads that spread more than anything else.

**The gates free at the base's rate** (`runs/s6-sleep3/r4-aux-gates-base-rate`: seed 0, the same flags with `--mem-gate-lr-mult 0.1` in place of 0, so that `wα` and `wβ` are trained at 2e-3, the rate of the base; traced every five sequence steps). Was it the ×10 rate alone that killed the gates of §2, and is holding them needed? The gates survive, the memory is used, and it is used far less well:

| steps | ask CE | ask accuracy | same weights without the memory: ask CE, accuracy | auxiliary accuracy |
|---|---|---|---|---|
| 1-50 | 1.960 | 0.158 | 1.951, 0.117 | 0.258 |
| 51-100 | 1.394 | 0.243 | 1.395, 0.220 | 0.290 |
| 101-150 | 1.366 | 0.329 | 1.428, 0.240 | 0.502 |
| 151-200 | 1.263 | 0.406 | 1.436, 0.247 | 0.580 |
| 201-250 | 1.254 | 0.427 | 1.438, 0.234 | 0.587 |
| 251-300 | 1.207 | 0.469 | 1.438, 0.264 | 0.608 |

C4's reading is **1.2078** (last exit 1.1984): under C4's bar of 1.30, 0.18 under the 1.3900 of the zeroed run, and far from the 0.72, 0.04 and 0.01 of the held gates. Over steps 250 to 300 the last exit answers 46.9% of 3,136 asks, the same weights without their memory 26.4% of 768. By delay: 0.540, 0.504, 0.473 and 0.360; 0.461 for a door shown with one colour and 0.476 for a door shown with several.

What the trace shows. The write gate does not close as in §2, it becomes selective: the mean `β` over the tokens goes from 0.78 at step 12 to 0.12 at step 64 and stays between 0.02 and 0.06 from step 100, while the largest `β` is 1.00: a few tokens are written at full strength and the others hardly at all. The decay gate, which §2 found stuck at 1, now forgets: the smallest `α` of a batch falls from 0.96 at step 43 to 0.05 at step 70 (the memory is nearly emptied on some frames: `‖S‖` at the asks is 0.5 there, against 4.4 at step 43) and then wanders between 0.30 and 0.89. `‖S‖` at the asks ends at 4.5 where the held gates have 11. The gradient of `wβ` is the largest of the memory's from step 80 on (0.4 to 11, where the projections have 0.006 to 0.6), with weights that stay small (`‖wβ‖` = 0.35, `‖wα‖` = 0.16 at step 300): they still read the raw block output, of norm about 100, and a small weight is a large pre-activation.

Read with the delays: the accuracy falls from 0.54 at 1 to 4 frames to 0.36 at 17 to 32, the profile the held gates do not have at the end, and what a decay under 1 on some frames would do. The run was still improving slowly when it ended (ask loss 1.26, 1.25 and 1.21 over its last three windows); where it would go with more steps is not known. Within 300 steps, holding the gates is what separates 0.47 from 0.99.

### 6.2 A write that can replace: the slot form

§5's third weakness, a write that superposes, was the reason to prepare a write that can replace, before §6.1 was read. It is an option of the configuration (`memory_write: slot`), off by default; nothing of §6.1 uses it.

**The form** (`memory.rs`). One key, one value and one gate per frame instead of one per token. With `x̄` the mean block output over the frame's valid tokens and `y = LN(x̄) − c`:

```
read:   q = l2norm(heads(y·Wqk))        x_i ← x_i + merge(√hd · l2norm(S·q))·Wo   for every valid token i
write:  k = l2norm(heads(y·Wqk))        v = heads(y·Wv)
        β = σ(wβ·y + b)    α = σ(wα·y + 6)    S ← α·S + β·(v − α·S·k)·kᵀ
```

- **The write is the delta rule on one unit key.** With `β = 1`, `S·k = v` after the write whatever the memory held: a door shown again replaces what its slot held, and what is stored under an orthogonal key is untouched. One pair per frame leaves no sum over tokens to keep stable, so the step needs no `1/n`.
- **The centre `c`** (`mem.{l}.centre`, one row per loop of the block) is the mean of `LN(x̄)` over frames. The mean states of two frames of `recall` differ by two or three tokens in nine, so without it the frames are expected to write under nearly the same key (in v2's form the mean key of a frame has an overlap of 0.88 across the frames of a sequence at step 1, §6.1); with it, `y` holds what is particular to the frame (the door, the colour, the number). That expectation was not measured on the base for the slot form itself. It is measured on one batch at the start (`--memory-centre-init`) and can follow the batches as the base moves (`--memory-centre-ema m`: after each sequence step, `c ← (1 − m)·c + m·c_batch`); otherwise it is a trained parameter.
- **The gates read `y`**, a normalised and centred state, as the projections do, and the write gate has a bias `b` (`mem.{l}.beta_bias`, started with `--memory-gate-bias`). A normalised state has a norm of about 10 where the raw block output of §2 has 96, and the centre takes off most of it, so a step of the same size moves the gate's pre-activation at least ten times less; by how much exactly was not measured (the norm of `y` is not in the trace).
- **What is read is brought to the scale of a normalised state** (`√hd · l2norm`, per head) before the read-out, whatever the scale of the values; an empty memory still reads zero.
- It adds `loops·d + 1` parameters per memory layer (385 on the pico), all float. The state per stream is unchanged.

**What is checked without a training** (`memory.rs`, tests): a fresh slot model is its base; after a write at `β = 1`, asking with the frame that was written returns its value, and again after a second frame was written over it; the centre measured on frames makes their states sum to zero, and the sequence loss measures the same centre on the batch it plays; every parameter of the form receives a gradient. And C1 with a door shown twice (`a_door_shown_again_...`): on C1's synthetic states, door a is shown with one colour, three idle frames pass, it is shown with another, then 1, 4, 8 or 16 idle frames pass before the ask; the four readings of an episode are decoded contrastively over 20 draws, as C1 is read. The new colour is decoded in 19, 20, 20 and 16 draws of 20 by the slot form at full step, against 20, 20, 15 and 4 by v1's rule.

**What is not measured.** The slot form was never run on the base: neither C2's reading at initialisation (`s2-memcheck --c2-only --write slot`, which exists for it) nor a training. §6.1 reached its result with v2's write as it is, and the measurements ended there. The slot form is code with unit tests and a synthetic check, not a result; whether it trains faster, reaches further at long delays, or lets the gates learn is unknown.

## 7. Recommendation

*(Rewritten on 16/11/2026 after §6.1. The recommendation of 15/11 asked for three things: gates that cannot die, a gradient that does not go through the read-out and the answer, and a write rule able to overwrite. The first two were enough; the third was not needed on train sequences.)*

**The smallest change that works** is two training switches, with v2's memory as it is:

1. **The gates held** at their start: `--mem-gate-lr-mult 0` (`β = 0.5`, `α = σ(6)`). An optimiser switch: no file format, no default and no line of the model changes.
2. **The auxiliary loss**: `memory_aux: true` in the config, which adds one linear head of `4·d` weights per memory layer for the training only (it is not exported), and `--memory-aux-weight 1`.

The command of §6.1's runs, from the repository root (C4's, plus the last line):

```
sky-train --cpu s2-decide --config runs/s6-sleep3/configs/sleep3-aux.json \
  --init runs/s6lr/softmax-adamw-x2/decide/last --tokenizer runs/s6lr/softmax-adamw-x2/decide/last/tokenizer.json \
  --records data/decisions,data/games,data/numeric --weights decisions=0.3,games=0.6,numeric=0.1 \
  --batch 128 --steps 300 --seed 0 --qat-frac 1.0 --loop-dropout 0.25 --length-pool 16 --pad-to 32 \
  --out runs/s6-sleep3/<name> --step-mem-gb 3 --adam-lr 0.002 --no-muon --seq-log \
  --sleep-records runs/s6-sleep/seq/recall.jsonl --sleep-weights recall=1 --sleep-batch 8 --seq-frac 0.9 --sleep-passes 1 \
  --no-compile --val-max 256 \
  --mem-lr-mult 10 --mem-gate-lr-mult 0 --memory-aux-weight 1
```

`sleep3-aux.json` is `runs/s6-sleep2/checks/sleep2.json` (v2's pico: `memory: true`, `memory_qk: shared`, `memory_value: frame`) with `memory_aux: true`. `CFG=aux scripts/s6-sleep3-diag.sh <name> --mem-lr-mult 10 --mem-gate-lr-mult 0 --memory-aux-weight 1` runs it; `SEED=1` and `STEPS=600` are the other runs of §6.1, and `--memory-trace 20 --memory-off-eval 4` add their readings.

**Which of the two is needed** (C4's reading, seed 0 unless said):

| | no auxiliary loss | auxiliary loss |
|---|---|---|
| gates free at ×10 (C4) | 1.3867 | not run |
| gates free at the base's rate | not run | 1.2078 (accuracy 0.469) |
| gates held | 1.3865 | **0.7231** (seed 0), **0.0374** (seed 1); 0.0149 over the same steps of seed 0's 600-step run |

The auxiliary loss is needed within 300 steps: held gates without it stay at chance (§3; no longer run was made without it). Holding the gates is not needed for the memory to be used at all, and is needed for it to be used well: with the auxiliary loss, gates free at the base's rate no longer die and reach 1.2078 with 47% of the asks right, against 99% when they are held (§6.1). The ×10 rate was therefore not the whole of §2's problem. The gates still read the raw block output; whether gates that read a normalised state can be left free without that cost is what §6.2's form was written to test, and was not tested. The cell "free at ×10 with the auxiliary loss" was not run; §2's collapse is over by step 35, before the auxiliary accuracy leaves 0.25 in any run, so there is little reason to expect it to differ from C4.

**What a third version could freeze**, from what is measured here:

- **The model**: v2's memory unchanged (shared `qk`, values `frame`, v1's write rule, read-out started at zero, ternary projections), plus `memory_aux` for the training.
- **The training of the memory**: gates held (`--mem-gate-lr-mult 0`), auxiliary loss at weight 1, `mem.*` at ×10, N = 1 while the memory is being learned.
- **The length**: over three runs the answer starts to use the memory between steps 159 and 183 (sequence batches of 8) and is under 0.1 some 40 to more than 120 steps later (§6.1). A check read over steps 250 to 300 of a 300-step run passes or fails on that spread (0.72, 0.04 and 0.01 for the same configuration); a phase A must hold at least 400 such steps, and a check should be read at 600.
- **CPU checks before any GPU run**, in C4's setting with `--steps 600`, on two seeds, train sequences only: (1) the ask-frame cross-entropy over steps 550 to 600 is at most 0.30 and the accuracy of the last exit at least 0.90, on both seeds (measured here on seed 0: 0.0007 and 1.0000; seed 1 was run for 300 steps only and reads 0.0374 and 0.9940 over its last 50); (2) over the same steps the same weights without their memory (`--memory-off-eval`) are at 1.30 or above and at an accuracy of 0.30 or below (measured: 3.17 and 0.226 on seed 0, 2.99 and 0.238 on seed 1 at 300 steps); (3) the auxiliary accuracy over the same steps is at least 0.95 (measured: 1.000; 0.996 on seed 1 at 300 steps). The bars are set far from what was measured on purpose: three runs do not say how often a run is slow.
- **The pilot's rule** can stay v2's (`docs/s6-sleep2-prereg.md` §5.3): on unseen dev seeds, `recall` at least 0.10 above the memoryless control with a bootstrap lower bound above 0, and the model with its memory zeroed within 0.03 of the control. What §6.1 adds to report next to it, and what would make the result interesting: the accuracy by delay bucket and for a door shown with one colour against several, with the memory and with it zeroed. Train sequences give 0.99 everywhere; the dev reading says how much of that is the game learned and how much the sequences.

**What this does not settle** and a pre-registration must say: nothing here ran with sleep (N > 1), so "the memory is used" is established and "sleep helps" is untouched; the gates are constants in this configuration, so a hypothesis that needs a learned write strength needs another form (§6.2's is written, not measured).


## 8. What is not known

Answered on 16/11/2026 (§6.1): the auxiliary loss brings the colour into what is retrieved in nearly every ask (auxiliary accuracy 0.995 and 0.996 over steps 250 to 300 of the two seeds), the base then uses it within 300 steps, and a trained averaged write does serve a door shown again (0.992 against 0.996 for a door shown with one colour, seed 1).

Still not known:

- **Anything on unseen seeds.** Every number here is read on train sequences, on the batch of a step before its update. No dev or test seed of the probes was read, no closed-loop probe was played (`s2-probe`), and no trained model was exported.
- **How a door shown again is served.** The write averages over the tokens at `β = 0.5` and the decay is 0.9975 a frame, so the algebra of §5 still holds for one write; the trained model is right on these doors all the same. Whether the keys come to depend on the colour, whether the value written at a second show carries the difference with what the frame has just read (the read comes before the write within a frame), or something else, was not looked at.
- **Why the runs arrive at different speeds** once the answer uses the read (0.72, 0.04 and 0.01 over steps 250 to 300 for three runs of one configuration, two of them on seed 0), and how wide that spread is. Three runs on two seeds are not a distribution; no run was seen to fail, and none of this says a run cannot.
- **Whether free gates can do as well as held ones.** At the base's rate they survive and reach 0.47 in 300 steps, still improving slowly (§6.1, one run, seed 0). More steps, a smaller rate, or gates that read a normalised state (§6.2) were not tried; nor the auxiliary loss with the gates free at ×10.
- **Whether the auxiliary loss is needed to the end**, or only until the answer uses the read; at what weight; and what it costs the other games, the language tasks and `rule110`, for which the auxiliary head has no target. Every run here trained on `recall` alone for its sequence steps.
- **Whether the result holds in the mix and at scale**: phase B's mix, the GPU, the nano size, a budget in minutes, sleep passes (N > 1). Nothing here touched them.
- **The slot form** (§6.2): written and unit-tested, never run on the base.
- Why the gradient pushes the gate shut rather than open during the base's transient: §2 gives the mechanism that makes the gate the fastest lever, not a proof that the push is noise and not a consistent preference for a silent memory.
- Whether the collapse also happened in v1's pilot (one shared rate, separate `q` and `k`): v1's logs hold no gate values.

