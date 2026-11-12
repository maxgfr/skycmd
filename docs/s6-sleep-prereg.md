# s6-sleep-nano: pre-registration (TODO §10)

**Status: frozen on 12/11/2026, before any code, any training and any measure of this experiment.** No memory layer exists in `sky-train` or `sky-infer` at this date, the two probe games are not written, and s6-nano, the base model, is still training (tranche 2 of the big run).

**Changes to this file:**

- every amendment is committed, dated and motivated in section 11;
- it comes before the run it concerns;
- it never touches a threshold, an arm or a set of seeds after a test score was read.

**What is never published without its gate.** Only a model that passes the gate of section 8 goes to Hugging Face, the site or the manifest. The report is published in every case, failures included.

## 1. The question

*Do Language Models Need Sleep?* (Lee, McLeish, Goldstein, Fanti, arXiv 2605.26099) gives a model a fixed-size memory of fast weights and lets it rerun its stack N times over what it just read, while no answer is asked. At equal memory size and equal answer latency, accuracy rises with N on Rule 110 and on GSM-Infinite. Its models have between 4 layers at d = 256 and 2B parameters.

**What this work can show:** whether the same holds on a 1.6M-parameter ternary decision model that answers in microseconds, on streams of game frames, with the pauses of a game as sleep time.

**What it cannot show.** Our memory spans frames, not tokens, and our training departs from the paper in three ways (section 4). A null result here says nothing about the paper.

## 2. The model

`configs/s6-sleep-nano.json` is s6-nano (`runs/recipe/configs/s6-nano.json`: vocabulary 2,048, d = 144, 2 state blocks × 2 loops, exits 1 to 4, 2 heads) plus one memory layer after each of the 2 state blocks. A memory layer is shared across loops, like its block.

**Correction to TODO §10.1.** s6-nano has no temporal carry (`temporal_carry` is off in its config): each frame is answered alone. The memory therefore replaces nothing, and the control arm is a model with no memory at all.

**State of a stream.** One matrix `S` of 72 × 72 per head and per memory layer, in f32: 4 matrices, 20,736 floats. A new stream starts at zero.

**Read**, at every application of block `l`, on its output `x` (`[tokens, d]`):

```
q = l2norm(heads(LN(x) · Wq))        r = S · q        x ← x + merge(r) · Wo
```

`Wo` starts at zero, so the model is exactly its base at step 0. With a zero memory the read adds nothing, so a stateless question goes through the same computation as in s6-nano.

**Write**, once per frame and per memory layer, from the output `x` of the last application of block `l` that the frame ran, over its `n` valid tokens:

```
k_i = l2norm(heads(LN(x_i) · Wk))    v_i = heads(LN(x_i) · Wv)
β_i = σ(wβ · x_i)                    α = σ(mean_i(wα · x_i) + 4)
S ← α·S + (1/n) · Σ_i β_i · (v_i − α·S·k_i) · k_iᵀ
```

This is the gated delta rule of the paper, applied once per frame over the frame's tokens instead of once per token.

**Parameters.** `Wq`, `Wk`, `Wv`, `Wo` (144 × 144 each), `wα`, `wβ` and one norm per layer: 166,752 new parameters, +10.6% over s6-nano. The four projections are ternary in the export like every other matrix; the gates, the norm and the state stay f32.

**Wake.** A frame is one forward pass, with the same exits and the same conformal thresholds as s6-nano. It reads the memory, then writes it once. This is the paper's N = 1.

**Sleep.** The model reruns the frames it saw since the last sleep (at most K = 8), in order, at full depth, N − 1 more times. Each replayed frame reads the current memory and writes it. Then the frames are dropped, and only the memory remains. Sleep happens:

- every K = 8 frames of a stream;
- when the game reports a pause (a level change, the end of a cue);
- never across an episode boundary: a new episode is a new stream.

No decision is asked during sleep, so the latency of a decision does not depend on N.

## 3. The probes

A null result on the 4 games would say nothing if they never need more than one frame. Two new games in `sky-games`, outside the default arena list and outside every existing gate:

**`recall`, the primary probe.** An episode is 64 frames, and every frame asks for a decision among `wait, red, green, blue, yellow`.

- Three doors (`a`, `b`, `c`). A *show* frame gives a door its colour, and a door can be given a new colour later. An *ask* frame names a door, and the right answer is its current colour. Every other frame carries only distractor numbers, and its answer is `wait`.
- A colour is on screen during one frame only. The delay between the last *show* of a door and its *ask* is drawn uniformly in four buckets: 1 to 4 frames, 5 to 8, 9 to 16, 17 to 32.
- **Score: accuracy on the ask frames**, overall and per bucket. A model without memory is at 0.25.

**`rule110`, the secondary probe**, the paper's sanity check at our scale. Frame 0 shows 16 cells. The game then reports a pause. Each of the next 7 frames hides the cells and asks the value of one cell after t steps of Rule 110 (on a ring), t ∈ {2, 4}. **Score: accuracy on the question frames.** A model without memory is at the majority class, measured on the dev seeds.

**Seeds of the probes.** An episode seed belongs to one split (`split_for_seed`: `≡ 18 (mod 20)` is validation, `≡ 19` is test, the rest is train), and training sequences are generated from train seeds only.

- **dev**: the first 500 episode seeds `≡ 19 (mod 20)` from 1,000,000,000 upward. Used to check the code, to read the pilot and to choose N.
- **test**: the first 1,000 episode seeds `≡ 19 (mod 20)` from 5,000,000,000 upward, the same for every arm, **read once**, after every nano run has ended and N is chosen.

**Seeds of the arena** (the 4 games): `sky-arena --seed 5000`, 200 episodes per game, as a test read once. This is the protocol of the s6-nano gate, which does not separate arena seeds from training seeds; both arms share that limit.

## 4. Training

**Warm start.** The memory layers are added to a trained base, and training goes on for a short budget with the adopted recipe (`runs/recipe/recipe.json`: AdamW, ×2, no Muon), ternary QAT on from the first step.

**Batches**, the same for both arms, drawn with the same seed:

- **half are sequences** of 24 consecutive frames of one episode (3 sleep periods): `recall` 40%, `rule110` 12% (its 8 frames, one sleep after the cue), and `flappy`, `rescue`, `dodge`, `doom` 12% each, with the oracle's answer as the target of every frame, as `s2-seqgen` does;
- **half are the stateless batches of s6-nano's decision training** (same records, same weights), answered with a zero memory, so that the language tasks are not forgotten.

**Loss.** Deep supervision on every exit of every wake frame. No loss during sleep.

**The three departures from the paper** (the "fast path" of TODO §10.1):

1. **Truncated backpropagation.** The gradient flows through the wake frames of the current period and through the last sleep pass before it. The earlier passes and the memory that enters the last pass are detached. The paper backpropagates through every pass.
2. **One model for every N.** N is drawn in {1, 2, 3, 4} for each batch. The paper trains one model per N.
3. **Warm start**, instead of training from scratch.

**The control arm** is the same base, fine-tuned on the same batches for the same number of steps, with no memory layer. It sees each frame alone.

**Budgets are in steps**, so that the use of the Mac does not change what a run computes. For each size, the first run of the sleep arm (seed 0) trains for a fixed working time, and every other run of that size trains for the number of steps it did.

## 5. The pico pilot (before any nano run)

Base: `runs/s6lr/softmax-adamw-x2/decide/last`, the s6-pico trained with the adopted recipe. Four runs, seed 0, the first one for **20 working minutes**:

| run | memory | N | backpropagation |
|---|---|---|---|
| P1, the fast path | yes | drawn in {1, 2, 3, 4} | truncated |
| P2 | yes | 4 | truncated |
| P3, the paper's way | yes | 4 | through every pass |
| P4, the control | no | — | — |

All readings are on the dev seeds of `recall`, with sleep at N = 4 unless stated.

- **The memory works**: P1 read at N = 1 beats P4 by at least **0.10**. If it does not, and the code does what this file says, no nano run takes place and the report says so.
- **Truncation is kept** if P2 ≥ P3 − 0.01. Otherwise the nano runs backpropagate through every pass.
- **The drawn N is kept** if P1 ≥ P2 − 0.01. Otherwise the nano sleep arm is trained at a fixed N = 4, and N = 1 and N = 2 are read on that model.
- **The warm start is not tested.** A run from scratch would need its own pretraining and budget; this departure is accepted and stated in the report.

The pilot also gives the `rule110` scores of the four runs at N = 1, 2 and 4. They are reported and decide nothing.

## 6. The nano runs

Base: the s6-nano of tranche 2, `runs/s6-nano-g$ROUNDS/decide/$ckpt`, whatever the verdict of its own gate. The exception is a base that does not play at all (mean score at or below a random policy on 3 games out of 4): the experiment stops and the report says so.

- **Two arms, 3 seeds each (0, 1, 2)**: `s6-sleep-nano` in the form the pilot kept, and the control.
- **Budget**: the sleep arm at seed 0 trains for **30 working minutes**; the 5 other runs train for its number of steps.
- **No DAgger round**, in either arm.
- Each run is exported and calibrated like s6-nano (`s2-calibrate`, `s2-export`), with a zero memory during calibration.

**Choice of N, by rule.** On the dev seeds of `recall`, with the mean over the 3 seeds: the smallest N in {1, 2, 4} within 0.01 of the best.

## 7. Hypotheses and how they are read

Every reading is on the test seeds, ternary weights, τ = 1, the mean over the 3 training seeds. Confidence intervals are paired bootstraps over episodes (10,000 draws), the 3 seeds pooled.

- **H0, the memory is used.** On `recall`, the sleep arm at N = 1 is above the control by at least 0.10.
- **H1, sleep helps.** On `recall`, accuracy at N = 4 is above accuracy at N = 1, with the lower bound of the 95% interval above 0. The report gives N = 1, 2 and 4, overall and per delay bucket.
- **H1b, secondary.** The same reading on `rule110`. It is reported and enters no gate.
- **H2, the games do not lose.** On the arena (4 games, veto on), at the chosen N, no game has a mean score more than 2% under the control arm.
- **H3, the language tasks do not lose.** On `typed_decisions_bench`, with a zero memory, accuracy is at most 0.01 under the control arm.

A hypothesis that fails is reported as not shown. No run is started again to "see": a technical failure (a crash, a full disk) restarts the same run with the same seed, and the report notes it.

## 8. Publication gate

`s6-sleep-nano` (the seed-0 model, at the chosen N) is published only if all of these hold:

1. the gate of s6-nano (`docs/s6-gen-prereg.md` §6): `typed_decisions_bench` above 0.398, no game more than 5% under s3-pico on the arena, ECE ≤ 0.08;
2. H0, H1, H2 and H3;
3. the `sky-infer` engine gives the same answers as the training code on the golden sequences (wake frames and sleep passes), natively and in WASM.

If H1 fails and the rest holds, nothing is published: a model that remembers but does not gain from sleeping is not `s6-sleep-nano`.

## 9. Cost and modes

- The pilot is about 2 hours of working time and the nano runs about 3 hours 30, before evaluation: 12 to 20 hours of wall clock in the throttled mode, less while the Mac is left alone.
- Two modes, as in the other pre-registrations: `0.5` and `45:15` while the Mac is in use, `0.85` and `50:10` while it is not, set by `scripts/auto-brake.sh` through `SKY_BRAKE_FILE`. The budgets count working time only.
- Nothing of this experiment runs during another training.

## 10. Report

`runs/s6-sleep/REPORT.md` gives every run of the pilot and of the nano arms, the number of steps of each, the three pilot decisions, the chosen N, H0 to H3 with their intervals, the cost of a step against the control, and every technical incident. `BENCH.md` and the site take the result from it.

## 11. Amendments

### Amendment 1 (12/11/2026, the day of the freeze): whole `recall` episodes in training, a slower decay, P3 spelled out

**Before any run.** At this date the two probe games are written and tested (`crates/sky-games/src/recall.rs`, `rule110.rs`, branch `s6-sleep-nano`). No memory layer and no training code exist, and no model has read a probe.

- **`recall` training sequences are whole episodes: 64 frames, 8 sleep periods**, instead of 24 consecutive frames (section 4). *Why:* writing the game showed that a 24-frame window cuts the show frame of most asks of the 17-to-32 bucket. Their target would be a colour the model never saw, which is noise, and the longest delays of the test would never be trained. The sequences of the 4 games stay at 24 frames, and `rule110` at its 8.
- **The decay gate starts at `α = σ(· + 6)`**, not `+ 4` (section 2). *Why:* at `+ 4`, α is 0.982 per frame, and a memory left untouched keeps 56% of its content after 32 frames, the longest delay of `recall`. At `+ 6` it keeps 92%. The gate stays learned.
- **P3, "through every pass", detaches nothing**: the gradient flows through every frame and every sleep pass of the whole sequence (section 5). The text did not say whether the memory was detached between sleep periods in P3.

**What does not change.** The model, the probes, the seeds, the arms, the budgets, the pilot rules, the hypotheses and the gate.

### Amendment 2 (12/11/2026): what writing the code fixed

**Before any run.** At this date the memory layer, the sleep training, the probe reader and the run script are written and unit-tested on branch `s6-sleep-nano` (`crates/sky-train/src/s2/memory.rs`, `s2/probe.rs`, `scripts/train-sleep.sh`, `scripts/s6-sleep-report.py`). The only executions were code checks of a few steps on throwaway folders, with no score read. Each point below was left open by the text, and is now fixed as the code does it.

- **A batch drawn at N = 1 detaches nothing.** It has no sleep pass, so the gradient flows through every wake frame of the sequence. From N = 2, the memory that enters the last pass of each sleep is detached, as section 4 says. *Consequence:* the N = 1 reading is trained with the longest gradient horizon of all, so the truncation cannot favour H1.
- **What a frame writes.** The write of layer `l` takes the output of block `l` at its last application, before the read is added to it. The two gates read that output as it is, the keys and values read its normalized form (section 2).
- **Batches.** A step is a sequence batch with probability 0.5, drawn, not one step in two. A sequence batch is 16 sequences of one game; a stateless batch is the one of the base's decision training (128 questions, loop dropout 0.25). N is drawn in both arms, so the control consumes the same random numbers and sees the same batches.
- **Rate and schedule.** The peak rate is the one of the base's decision training (the recipe's scale, and for nano the scale of its learning-rate pilot), with the same warmup, plateau and decay over the run.
- **Sequences.** 10,000 episodes of `recall`, 20,000 of `rule110` and 5,000 sequences of 24 frames per game, generated once (`s2-seqgen --seed 61`); the tenth that falls in the validation and test splits is not trained on.
- **How the readings are taken.** `sky-train s2-probe` plays the checkpoint in closed loop with its ternary weights (the fake quantization of QAT), at the last exit, on the CPU. H2 uses it for the 4 games too (`sky_games::evaluate_with`, veto on, flappy capped at 200 pipes, as `sky-arena`), so both arms go through the same reader. H3 is the ternary row of `s2-eval` on `typed_decisions_bench`. The exported GGUF and `sky-infer` come in only for the publication gate (section 8, items 1 and 3).
- **The test step runs once.** `scripts/train-sleep.sh test` writes `runs/s6-sleep/test/done` and refuses to run again.

**What does not change.** The model, the probes, the seeds, the arms, the budgets, the pilot rules, the hypotheses and the gate.

### Amendment 3 (13/11/2026): a third brake mode, and the memory of a step

Before any run. **A third brake mode.** At the owner's request, `scripts/auto-brake.sh` writes `1.0 55:5` between 23:00 and 08:00 when the Mac has had no input for 30 minutes and nothing else works on it, and the memory cap needs four readings in a row above 13 GB, as in amendment 7 of `docs/s6-gen-prereg.md`. The stateless half of the batches may use `--step-mem-gb` as the base's decision training does (same amendment): the same update with less memory, in both arms. Budgets in working minutes and in steps are unchanged. Nothing else changes.
