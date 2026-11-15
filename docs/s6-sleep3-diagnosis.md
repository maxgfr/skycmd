# s6-sleep v2 at C4: why the memory stops receiving gradient, and what would make training use it

Written on 15/11/2026, after `docs/s6-sleep2-prereg.md` ended at C4. Everything here ran on the CPU (`sky-train --cpu`, `nice -n 19`, one run at a time), on `recall`'s **train** sequences only (`runs/s6-sleep/seq/recall.jsonl`, v1's); no dev or test seed of the probes was read, and nothing ran on the GPU. This file is a diagnosis, not a pre-registration: a third version needs its own.

**The setting** is C4's, unchanged (`scripts/train-sleep2.sh`, `c4`): 300 steps from the pico base `runs/s6lr/softmax-adamw-x2/decide/last` with v2's memory (`runs/s6-sleep2/checks/sleep2.json`: shared `qk`, form `frame`), 90% of the steps batches of 8 `recall` sequences of 64 frames, rate 2e-3 with `mem.*` at ×10, N = 1, QAT from the first step, seed 0. `scripts/s6-sleep3-diag.sh <name> [flags]` runs one experiment; `scripts/s6-sleep3-trace.py` reads the logs. The number reported for a run is C4's: the ask-frame cross-entropy (the loss's own deep-supervision measure) averaged over steps 250 to 300; chance is ln 4 = 1.3863.

## 0. Summary

1. **The gradient dies because the write gate closes, and a closed sigmoid gate cannot reopen.** `β_i = σ(wβ·x_i)` reads the raw block output, whose norm is about 96, and `wβ` is trained at the ×10 rate. Between steps 7 and 35 the mean `β` over the written tokens goes from 0.96 to 3e-7 (`‖wβ‖` from 0.18 to 0.95). With `β = 0` nothing is written: `‖S‖` at the ask frames goes from 4.8 to 7e-5, the reading `r = S·q` with it, and every memory gradient is proportional to `S` or to `β(1 − β)`. Step 52 is only the first step at which C4's log reported gradient norms (every 50 sequence steps); the collapse is over by step 35 (§2).
2. **Holding the gates open keeps the gradient alive and changes nothing else.** With `wα`, `wβ` held at zero (`β = 0.5`), the memory gradient stays near 1e-3 for 300 steps, the read adds about 13% of the hidden state's norm, and the ask loss is 1.3865 (§3). A live gradient is necessary, not sufficient.
3. **What follows the memory is not the obstacle.** With the colour asked written in the ask frame's own state and the read held at zero, the same base reaches an ask loss under 0.01 at step 70 (§4). The base learns to answer from a colour it can see in about thirty steps once its first transient is over.
4. **What the memory hands over at the start is too faint to be learned from** (§5): keys of different tokens are nearly collinear (mean squared cosine 0.73 between a shown door and an idle frame, C2), so C2's 0.39 is 0.49 at delays 1 to 4 and chance beyond 8; one show writes 6% of a value in which the colour is one token in nine; and half of the asks concern a door that was shown with two or more colours, which this write rule superposes.
5. **Experiments** are in §6, the recommendation in §7, what is not known in §8.

## 1. The instrumentation

`s2-decide --seq-log --memory-trace [N]` (off by default) logs, for one sequence step in N and **before** the update: the norm of the gradient and of the weights of every `mem.*` tensor, the norm and the share of zeros of the ternary form of each projection, and, per memory layer, from a second pass over the same batch without gradient (`memory::trace`): the write strength `β` (mean and largest over the valid tokens of every frame), the decay `α` (mean and smallest), `‖S‖` of the memory each frame reads (over every frame, and over the ask frames), and at the valid tokens of the ask frames `‖x‖` (the block output), `‖r‖` (what is retrieved) and `‖r·Wo‖` (what the read adds). The trace of a fresh model is tested (`memory.rs`, `the_trace_of_a_fresh_memory_...`).

The traced baseline (`runs/s6-sleep3/d0-trace`, 90 steps of the 300-step schedule) reproduces C4's first attempt: same losses at steps 1 to 7 to four decimals, memory gradient norm 6.8e-11 at step 52 (1.4e-11 in `runs/s6-sleep2/checks/c4/mem`; the CPU runs are not bit-identical).

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

Each run is C4's setting with one change; "memory" and "zeroed" are C4's two numbers (ask-frame cross-entropy over steps 250 to 300; ln 4 = 1.3863). A change that touches only the memory's weights or their optimiser leaves the zeroed run what it was (with the read held at zero no memory weight receives a gradient), so those rows share C4's zeroed run, 1.3873; the CPU runs are not bit-identical from one launch to the next (§1), so that number is good to about three decimals.

| run | change from C4's first attempt | memory | zeroed | what the trace shows |
|---|---|---|---|---|
| `runs/s6-sleep2/checks/c4` | none (`mem.*` at ×10, ternary) | 1.3867 | 1.3873 | `β` → 3e-7 by step 35, memory gradient 1e-11 (§2) |
| `runs/s6-sleep2/checks/c4-x30` | `mem.*` at ×30 | 1.3876 | 1.3879 | memory gradient exactly 0 at steps 159 and 216 |
| `runs/s6-sleep2/checks/c4-float` | float memory projections (the read-out included) | 1.3871 | 1.3873 | memory gradient 1.7e-9 at step 52 |
| `runs/s6-sleep3/e1-gates-frozen` | gates held at their start (`--mem-gate-lr-mult 0`) | 1.3865 | 1.3873 (C4's) | `β` = 0.5, memory gradient about 1e-3 throughout, read 11% to 13% of `‖x‖` (§3) |
| `runs/s6-sleep3/oracle` | no memory in play: the colour asked is in the ask frame's state, read held at zero | n/a | under 0.01 from step 70 (stopped at step 81) | the ceiling of §4, not a candidate |
| `runs/s6-sleep3/c-aux-gates-frozen` | gates held, and the auxiliary loss at weight 1 (`memory_aux`, `--memory-aux-weight 1`) | cut at step 84: 1.3931 over steps 71 to 84 | not run | auxiliary accuracy 0.25 until step 50, then 0.34, 0.37 and 0.41 over steps 51 to 60, 61 to 70 and 71 to 84; see the note below |

**Not run:** a read-out started small and non-zero (`--memory-out-init`), the read-out alone in float (`--memory-out-float`; the float retry of C4 already covers it), no weight decay on the memory (`--mem-no-decay`), the gates at the base's rate (`--mem-gate-lr-mult 0.1`), the auxiliary loss with free gates, and a copy-path start in float (`--memory-copy-init`). The switches exist and are off by default. §2 says what to expect of the first three: none of them touches the gate, and `Wo` was never the factor that vanished.

**Note on the last row.** At about 21:30 on 15/11/2026 the session that ran these experiments lost its access to `~/Documents` (every open returned "Operation not permitted"); the run was started at 21:24 and ended at 21:36, at step 84 of 300, without its final checkpoint. Its log was read on 16/11/2026, once the access was back. It holds 80 sequence steps. The accuracy of the auxiliary head (the colour asked, decoded linearly from what the memory retrieves at the ask frames; chance 0.25) is 0.25 to 0.27 over each ten-step window until step 50, then 0.335 (steps 51 to 60), 0.368 (61 to 70) and 0.408 (71 to 84), with its cross-entropy going from 1.39 to 1.25. The ask loss over the same windows is 1.401, 1.393 and 1.393: at chance. So in 84 steps the memory starts to hold and return the colour once a loss asks it to, and the answer does not yet use it. This is one cut run on train sequences, with no zeroed control: it says the auxiliary loss is worth its full run, not that it works.


## 7. Recommendation

No change tried so far makes training use the memory in C4's setting, so there is no "smallest working change" to adopt. What the measurements do support:

1. **The gates must not be able to die.** Held during phase A (`--mem-gate-lr-mult 0`, an optimiser switch that changes no file format and no default), or rebuilt for v3: read the normalised state as the projections do, train at the base's rate and not at ×10, and start `β` high enough that one show writes a usable share of its value. This is necessary (§2) and not sufficient (§3).
2. **The memory needs a gradient that does not go through the read-out and the answer.** The auxiliary head (`memory_aux`: the colour asked decoded linearly from what is retrieved at the ask frames) gives the keys, the values and the gates a direct reason to store and retrieve the colour, from step 0, while the read-out is still zero. It is implemented and tested. The one run that used it was cut at step 84, where the auxiliary accuracy had gone from 0.25 to 0.41 and was still rising (§6, last row); the full run is the first step of a third version's pilot.
3. **The write rule has to be able to overwrite and to separate keys** before any of this can go far below chance (§5): 52% of the asks concern a re-shown door.

A third version's pilot should therefore be a CPU pilot first, in this order, each step with its zeroed control: (a) the auxiliary loss with held gates in C4's setting, read by two numbers, the auxiliary accuracy at steps 250 to 300 (is the colour linearly present in `r` once the memory is trained for it?) and C4's ask loss; (b) if the colour is present in `r` and the ask loss is still at chance, the copy-path start (values and read-out at the identity in float, so that the read lands in the space the fusion head already reads, §4); (c) only then gates that learn, and a C1 that plays a re-shown door.


## 8. What is not known

- How far the auxiliary loss brings the colour into what is retrieved (0.41 at step 84 of a cut run, rising), and whether the base then uses it within 300 steps (not by step 84).
- Whether the gates survive at the base's rate (`--mem-gate-lr-mult 0.1`) or when they read a normalised state. Only "free at ×10" (dead by step 35) and "held" were measured.
- Why the gradient pushes the gate shut rather than open during the base's transient: §2 gives the mechanism that makes the gate the fastest lever, not a proof that the push is noise and not a consistent preference for a silent memory.
- Whether the collapse also happened in v1's pilot (one shared rate, separate `q` and `k`): v1's logs hold no gate values.
- How much of C2's 0.39 a trained key projection can recover at delays above 8, and whether an averaged write can ever serve a re-shown door.
- Everything about `rule110`, sleep passes (N > 1) and the nano size: nothing here touched them.

