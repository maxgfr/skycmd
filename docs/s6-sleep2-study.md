# s6-sleep v2: design study

Written on 15/11/2026, after the v1 pico pilot and its diagnosis (`docs/s6-sleep-diagnosis.md`). Read: `docs/s6-sleep-prereg.md`, `docs/s6-sleep-diagnosis.md`, `crates/sky-train/src/s2/memory.rs`, `s2/model.rs`, `s2/probe.rs`, `s2/train.rs`, `crates/sky-train/src/optim.rs`, `crates/sky-games/src/recall.rs`, `scripts/train-sleep.sh`, `runs/s6-sleep.log`, `runs/s6-sleep/pilot.json`, `runs/s6-sleep/pilot/*/log.jsonl`, `runs/s6-sleep/seq/recall.jsonl`. Line numbers are on `main` at d46aad1; the code there is the same as on branch `s6-sleep-nano`. Nothing was built, trained or run on the GPU. The only computations are on the logs and some arithmetic, given below.

`crates/sky-infer/src/s2/` has no memory code at all: the engine cannot run a memory model yet. This does not matter for a pilot, whose readings all go through `sky-train s2-probe`. It does matter for the publication gate (v1 §8, item 3).

## 0. Summary

- **The cause I believe in most.** At initialisation, the memory cannot return anything related to what was written, so the only memory weight that gets a gradient learns from noise. Three facts produce this:
  - the read-out `Wo` is zero, so at step 0 the gradient of every other memory weight is exactly zero;
  - the query and key projections are two independent random matrices, so `q·k` does not depend on whether the ask frame and the show frame name the same door;
  - nothing ties the colour to the door in what a show frame writes, unless the base's attention has already put the colour into the door token.

  With the memory returning noise, the loss on the non-ask frames pushes toward a silent memory. The ask frames are the only signal, and they are 8 of 64 frames in about 20% of the updates. The measured drift of the memory weights fits an Adam random walk on noise in size (§1.4).
- **Recommended change, smallest first.**
  1. One shared query/key projection.
  2. Values that carry the frame's context, chosen between two forms by a CPU check.
  3. The memory weights at 10× the base rate.
  4. A recall-heavy first phase without sleep.
  5. Per-game sequence-loss logging.

  About 60 changed lines in the trainer, plus about 250 lines of checks and tests.
- **Checks before any GPU time.**
  - C1, a hand-set retrieval test of the write/read algebra under interference.
  - C2, a readout of the colour from the memory at initialisation, on the real pico base, on the CPU.
  - C3, gradient consistency of `mem.out` at step 0.
  - C4, a 300-step CPU training on recall that must lower the ask-frame loss below ln 4 by a margin that a zeroed memory does not reach.
- **Odds.**
  - About 0.7 that the CPU checks pass.
  - About 0.65 that a 20-minute pilot then beats the control by 0.10 on dev `recall`.
  - About 0.45 unconditional.
  - About 0.8 that a pilot run after passing checks shows recall clearly above chance (≥ 0.32).

## 1. The failure, precisely

### 1.1 The read path at step 0

- `Wo` (`mem.{l}.out.weight`) is zero-initialised (`memory.rs:64-65`). The read is `x + merge(S·q)·Wo` (`memory.rs:128-135`), so `∂L/∂r = ∂L/∂(x_out)·Woᵀ = 0`. **At step 0 the gradients of `q`, `k`, `v`, `norm`, `alpha` and `beta` are exactly zero.** Only `Wo` receives a gradient: `∂L/∂Wo = Σ_t r_tᵀ·∂L/∂x_t`, with `r_t = S·q_t`.
- `q`, `k` and `v` are three independent `TruncNormal(0.02)` matrices (`memory.rs:57-63`). The retrieval weight of a written token `j` for a reading token `t` is `q_t·k_j = LN(x_t)ᵀ·Wqᵀ·Wk·LN(x_j)` after normalisation. With `Wq` and `Wk` independent, this is a random bilinear form:
  - its sign and size do not depend on whether `x_t` and `x_j` are the same door;
  - its normalised value has a spread of about `1/√hd ≈ 0.10` (hd = 96 for pico, `runs/s6-sleep/pilot/sleep.json`: 1 head, d = 96).

  So `r_t` at an ask frame is a random mixture of everything written, not the value of the matching show. The expected `∂L/∂Wo` therefore carries no information about the colour. `Wo` moves, but along noise.
- With a shared projection (`Wq = Wk`), `q_t·k_j ≈ cos(LN x_t, LN x_j) ± O(1/√hd)` (a random projection keeps angles: Johnson–Lindenstrauss). The door token of an ask frame then retrieves, more than anything else, the door token of the show frame that named the same door. This is the auto-associative setting of Hopfield memories.
- The read is added to the raw residual stream before the exit norms (`model.rs:786-800`). Nothing normalises it away: its scale is `Wo`'s to learn. `bf16` is off (`s2/config.rs:369`). Global clipping at 1.0 (`optim.rs:120`) applied to about 1% of P1's logged steps (median gradient norm 0.24, from `p1/log.jsonl`). Neither is a cause.

### 1.2 The write: what is stored, and how strongly

- One frame writes all its valid tokens at once, each with weight `β_i/n` (`memory.rs:153`, `:170`). A recall frame is `show a colour red ask none noise 46` (`recall.rs:99-109`; first record of `runs/s6-sleep/seq/recall.jsonl`), about 8 state tokens. With `β = σ(0) = 0.5` at init (`memory.rs:67`), **each token stores about 6% (0.5/8) of its value per write**.
- **Binding.** For the ask `ask a` to be answered, the slot keyed by the show frame's `a` must hold "red". The `red` token is two positions after `a`. In the value `v_i = LN(x_i)·Wv` (`memory.rs:150`), "red" is present at `a` only if the base's attention already moved it there. The base (s6-pico) was never trained on `recall`. Whether its door token carries the colour is unknown, and C2 measures it. Wang, Shi and Fox (§2) show that a one-layer associative memory solves recall only when key and value are formed from neighbouring tokens, for example by a short convolution.
- **Overwriting.** Every frame repeats the template (`show … colour … ask … noise …`). The delta rule contracts the stored content along direction `u` by `1 − (β/n)·Σ_i (k_i·u)²` per frame. With mean squared cosine `c̄²` between the door key and an idle frame's keys, that is about `1 − 0.5·c̄²` per idle frame. For `c̄² = 0.2`, an assumed value that C2 measures, a fact keeps:
  - 66% after 4 frames;
  - 43% after 8;
  - 19% after 16;
  - 3% after 32.

  The decay gate adds a factor of `σ(6)^32 = 0.92` over 32 frames (`memory.rs:38`, amendment 1 of v1), which is negligible.
- **Conclusion.** Even with perfect addressing, the longest bucket would be hard at init and the short buckets possible. This bounds what a short pilot can show, which is why the decision rule below does not need the long buckets.

### 1.3 What the loss asks of the memory

- **Mix.**
  - Half of the steps are sequence steps, drawn (`train.rs:1867-1877`, `--seq-frac 0.5` at `scripts/train-sleep.sh:132`).
  - `recall` is 40% of those (`:131`).
  - In P1, 3,684 updates gave about 1,840 sequence steps and **about 740 recall batches**, each 16 episodes × 8 asks (`recall.rs:22`: 2 asks per bucket × 4 buckets). That is about 94,000 ask frames in all.
- **Weight.** The sequence loss is the mean over the 64 frames of the deep-supervised cross-entropy (`memory.rs:270-312`, `model.rs:998-1008`). An ask frame weighs 1/64 of its batch's loss.
- **Stateless batches give the memory no gradient.** The other half of the steps run without memory (`train.rs:268-283`, `forward_prepared_mem(…, None)`). Adam keeps moving the memory weights on momentum alone.
- **No shortcut exists for the asks.**
  - The colour is uniform and independent of the ask frame (`recall.rs:147-180`).
  - Without memory, the best ask answer is the uniform prior, ln 4 = 1.386 per ask, and the base found it. The probe models answer a near-constant, model-specific colour (diagnosis §3).
  - The 56 other frames of an episode (`wait`) are trivial.
- **But there is a pull toward silence.** On every frame that does not need the memory (56/64 of `recall`, all 24 frames of the four games, `rule110`), a non-zero read perturbs answers the base already gets right. When retrieval is uninformative, the gradient on these frames favours `Wo → 0`. When retrieval is informative, the 8 asks per episode must outweigh that pull. In v1 they had nothing to offer.
- **Truncation is not the cause.**
  - P3 backpropagated through everything and failed equally (`pilot.json`: 0.243).
  - The detach at `memory.rs:253-255` does cut the gradient from an ask to a show more than one sleep earlier in P1 and P2.
  - Batches drawn at N = 1 detach nothing (v1 amendment 2).

### 1.4 What the checkpoints say about the drift

The learning rate is `rate 96 1 = 1e-3 × 2 × √(96/96) × 1 = 2e-3` (`scripts/train-sleep.sh:72`, `:149`). AdamW has β1 = 0.9, β2 = 0.95, eps = 1e-8 and weight decay 0.01 (`optim.rs:105-126`). Under the defaults, the memory sits in the decayed AdamW group at the base's rate.

- **Pure noise.** An Adam update has a long-run displacement of about `lr·√n` after `n` updates (an exponential average of white noise has the same long-run variance as the noise). For n ≈ 1,840 updates with a gradient, the spread is about 0.086, so the mean |w| of a weight that starts at zero is about 0.8 × 0.086 ≈ 0.069. For `q`, `k` and `v` starting at a mean |w| of 0.0145, it is about 0.070.
- **A consistent gradient** would allow up to `lr·n ≈ 3.7`.
- **Observed** (diagnosis §2), mean |w| at `last`:

  | run | `Wo` | `q` / `k` / `v` |
  |---|---|---|
  | P1 | 0.078 | 0.079 / 0.085 / 0.084 |
  | P3 | 0.077 | 0.068 / 0.067 / 0.077 |
  | P2 | 0.044 | 0.040–0.051 |

- **Reading.** P1 and P3 are within about 20% of the noise prediction and 50 times below what a signal would give. P2's smaller drift is not explained by this argument; one possibility is heavier-tailed gradients, which give a smaller `m/√v`. This supports "moved along noise" but does not prove it.
- **The decay gate.** Its gradient (about 5e-12) is below Adam's eps of 1e-8. Its update is about `lr × 5e-4` per step, so it is frozen, harmlessly (0.9975 per frame). `σ'(6) = 0.0025` is the reason.

### 1.5 Ternary quantisation from step 0

The four memory projections are fake-quantised with the straight-through estimator from the first step (`model.rs:69`, `:434-455`, `--qat-frac 1.0` at `scripts/train-sleep.sh:127`). At `Wo = 0`, `ternarize` returns 0, because the scale has a floor of 1e-8 (`model.rs:378-381`), and the STE still passes the gradient (`:384-386`). Ternary random projections keep sign structure, so they keep angles roughly. **This is not a blocker.** It may cost retrieval precision once the memory works; that is choice O5 of the draft.

### 1.6 Ranking of the candidate causes

| rank | cause | evidence | confidence |
|---|---|---|---|
| 1 | Content-free retrieval at init (independent `Wq`/`Wk`, zero `Wo`): no informative first gradient, and noise keeps separate `Wq`/`Wk` from ever aligning | algebra of §1.1; weights drifted at the noise scale (§1.4); training loss equal to the control within ±0.004 (diagnosis §2) | high |
| 2 | Weak signal against a pull toward silence: about 740 recall batches, 8/64 frames each, against every other frame penalising a noisy read | §1.3 counts | high that it slows learning; medium that it alone blocks it |
| 3 | No door→colour binding in the written values unless the base provides it | §1.2; Wang et al. §6 | medium (unmeasured: C2) |
| 4 | 6% write strength and template overwriting | §1.2 arithmetic, with an assumed key overlap | medium for the long buckets, low for the short ones |
| 5 | Shared learning rate for a zero-initialised read-out | LoRA+ (§2) | medium as a speed factor |
| — | Truncation, normalisation, bf16, clipping, ternary STE | §1.1, §1.3, §1.5 | ruled out or minor |

## 2. What makes small fast-weight and external memories train (sources)

Dates are first arXiv versions unless stated. "Read" means the page was opened for this study. "Known" means it is cited from prior knowledge and was not re-read here.

- **The paper v1 follows.** Lee, McLeish, Goldstein, Fanti, *Do Language Models Need Sleep? Offline Recurrence for Improved Online Inference*, arXiv 2605.26099, v1 25/05/2026, v3 05/06/2026, https://arxiv.org/abs/2605.26099 (read, HTML).
  - The memory is Gated DeltaNet fast weights, zero-initialised, and the gradient flows through every sleep pass.
  - The optimiser is Muon at 2e-3 for the synthetic tasks, with AdamW fixed at 5e-5.
  - For the sliding-window runs, **the new memory layers are warmed up alone for one epoch first, and "using hard eviction for the warm-up stage is crucial for the model to learn to refine the fast weights."**
  - Training at larger N "can make training slow and unstable."
  - Their non-looped automaton baseline needs about 55B training tokens to reach about 10% exact accuracy; the models are 4 layers at d = 256 and up.
  - *Lesson:* even the source warms up the memory before the joint training and trains for budgets far above ours.
- **Gated delta rule.** Yang, Kautz, Hatamizadeh, *Gated Delta Networks: Improving Mamba2 with Delta Rule*, arXiv 2412.06464, 09/12/2024, ICLR 2025, https://arxiv.org/abs/2412.06464 (read, abstract and summaries). The rule is per token, `S_t = α_t S_{t−1}(I − β_t k_t k_tᵀ) + β_t v_t k_tᵀ`, with L2-normalised q and k. The gate erases quickly, and the delta rule makes targeted updates. Chunkwise parallel form: Yang et al., arXiv 2406.06484, 06/2024, https://arxiv.org/abs/2406.06484 (seen).
- **Fast weight programmers.** Schlag, Irie, Schmidhuber, *Linear Transformers Are Secretly Fast Weight Programmers*, arXiv 2102.11174, 02/2021, ICML 2021, https://arxiv.org/abs/2102.11174 (read, summary). Purely additive outer products hit a capacity limit. The delta rule lets the programmer correct a stored association, with a learned, input-dependent write rate (our β). Earlier: Ba et al., *Using Fast Weights to Attend to the Recent Past*, arXiv 1610.06258, 10/2016 (known).
- **Associative recall needs addressing and value formation.**
  - Arora et al., *Zoology*, arXiv 2312.04927, 12/2023, https://arxiv.org/abs/2312.04927 (read, search). MQAR is the standard probe. They sweep four learning rates in [1e-4, 1e-2] and train for 64 epochs for every model, because recall is sensitive to the rate.
  - Wang, Shi, Fox, *Test-time regression*, arXiv 2501.12352, 01/2025, https://arxiv.org/abs/2501.12352 (read, §6). In "One short conv is all you need for next-token recall", a single layer can store a cue→answer binding only if each key/value pair is built from neighbouring tokens: "the appropriate key-value pairs to regress." *Lesson for us:* the door→colour binding must be available in what is written, not hoped for.
  - Larson, *Constant-Memory Recall: Learned Associations in a Fixed Matrix State*, arXiv 2610.00232, 23/09/2026, https://arxiv.org/abs/2610.00232 (read, abstract).
    - A small DeltaNet variant with **fixed token-specific key biases** (addressing given at init) reaches 99.95% recall of 32 pairs.
    - **Zeroing the memory removes the recall**, which is the ablation we adopt as a check of use.
    - Parameter-matched non-memory baselines stay near chance.
    - A 48-pair variant stays near chance after a quarter of the budget, so budgets matter.
- **Shared query/key.** Kitaev, Kaiser, Levskaya, *Reformer*, arXiv 2001.04451, 01/2020, ICLR 2020, https://arxiv.org/abs/2001.04451 (read, search): "A shared query-key space does not perform worse than regular attention." Modern Hopfield layers (Ramsauer et al., arXiv 2008.02217, 07/2020, known) retrieve by similarity of the query to the stored patterns, which is the auto-associative case.
- **Surprise-gated and test-time-training memories.**
  - Behrouz, Zhong, Mirrokni, *Titans*, arXiv 2501.00663, 31/12/2024, https://arxiv.org/abs/2501.00663 (read, search). The memory is written by the gradient of `‖M(k) − v‖²` (surprise), with momentum and a forgetting gate, and is used as context, gate or layer.
  - TTT layers: Sun et al., arXiv 2407.04620, 07/2024 (known). *Test-Time Training Done Right* (large chunks), arXiv 2505.23884, 05/2025 (seen).
  - Zhao, Jones, *Fast-weight Product Key Memory*, arXiv 2601.00671, 02/01/2026, v2 22/02/2026, https://arxiv.org/abs/2601.00671 (read, abstract). Fast weights are updated by a local memory-rewrite objective, next to the global loss.
  - *Lesson:* when the memory has its own local objective, it is not left to an outer loss alone to discover it. Our delta rule already is such a local objective. What is missing is that its addressing is random at init.
- **Recurrent memory transformers: curriculum.**
  - Bulatov et al., *Scaling Transformer to 1M tokens and beyond with RMT*, arXiv 2304.11062, 19/04/2023, v2 06/02/2024, https://arxiv.org/abs/2304.11062 (read, HTML).
    - "At the beginning of training, RMT is fine-tuned on shortest one segment version of the task."
    - Trained directly on four segments, "RMT does not learn neither to solve the task, nor to extrapolate."
    - Mixing segment counts at each curriculum step helps.
  - Rodkin et al., *Associative Recurrent Memory Transformer*, arXiv 2407.04841, 05/07/2024, v2 13/02/2025, https://arxiv.org/abs/2407.04841 (read, HTML).
    - The memory is delta-rule associative, written per segment, with a sigmoid write gate.
    - The curriculum is 2, 3, 5, 8, 16, 32 segments, with segment counts drawn at random.
    - On language modelling, "ARMT tends to keep in memory only the last segment", which they attribute to training rather than to the architecture.
- **NTM and DNC lessons.** Collier, Beel, *Implementing Neural Turing Machines*, arXiv 1807.08518, 07/2018, ICANN 2018, https://arxiv.org/abs/1807.08518 (read, search). "The choice of memory contents initialization scheme is crucial", and a small constant initialisation converges about 2× faster than learned or random ones. Graves et al. trained the NTM and DNC with curricula (known). *Lesson:* initialisation and curriculum, not architecture, usually decide whether a memory trains.
- **Auxiliary losses.** Trinh, Dai, Luong, Le, *Learning Longer-term Dependencies in RNNs with Auxiliary Losses*, arXiv 1803.00144, 03/2018, ICML 2018, https://arxiv.org/abs/1803.00144 (read, search). Reconstruction or prediction losses at anchor points make truncated BPTT work and help full BPTT.
- **A zero-initialised output at the base rate learns slowly.** Hayou, Ghosh, Yu, *LoRA+*, arXiv 2402.12354, 02/2024, ICML 2024, https://arxiv.org/abs/2402.12354 (read, search). With `B` zero-initialised and `A` random, one shared rate is inefficient. `η_B = λ·η_A` with λ ≫ 1; λ = 16 is their default, a value known rather than re-read. ControlNet's zero convolutions (arXiv 2302.05543, known) train because the input of the zero layer already carries the condition. The analogy is that a zero `Wo` is fine only if `r` is informative.
- **Memory in decision tasks.** Ni et al., *When Do Transformers Shine in RL? Decoupling Memory from Credit Assignment*, arXiv 2307.03864, 07/2023, NeurIPS 2023, https://arxiv.org/abs/2307.03864 (read, search). The Passive T-Maze needs memory with immediate credit; LSTMs fail beyond about 250 steps. Our `recall` is that kind of task: long memory, credit on the frame itself.
- **Sleep as consolidation.** Tadros, Krishnan, Ramyaa, Bazhenov, *Sleep-like unsupervised replay reduces catastrophic forgetting in artificial neural networks*, Nature Communications 13:7742, 15/12/2022, https://www.nature.com/articles/s41467-022-34938-7 (read, search). Offline phases of local Hebbian plasticity on noisy input protect old tasks. In our design the replays of a sleep are exactly local delta-rule writes repeated, so sleep strengthens weak writes (v1 test `memory.rs:422-448`). That makes the 1/n write worth keeping: it is the case where sleep can help.

**What the sources agree on**, as fixes for a memory that is ignored, from most to least directly applicable here:

1. Make addressing work at init: shared or fixed keys (Reformer, Larson, Hopfield).
2. Give the memory key/value pairs that already contain the binding (Wang et al. §6).
3. Warm up the memory path before, or with more weight than, the joint training (Lee et al., RMT, ARMT).
4. Use curricula from short to long (RMT, ARMT, NTM and DNC).
5. Give a zero-initialised output a larger rate (LoRA+).
6. Use auxiliary losses at the read (Trinh et al., FwPKM).
7. Use ablation by zeroing the memory as the test that it is used (Larson).

Not supported as first fixes for this failure:

- full backpropagation through every pass (already tested in P3);
- gating or normalisation changes (no wash-out found);
- detaching the base (there is no shortcut to block).

## 3. Ranked changes

Each line gives what, why, where and about how many lines, the cheap test that shows it works before the GPU, and the risk.

1. **One shared query/key projection, `mem.{l}.qk`** (replaces `q` and `k`).
   - **Why.** Retrieval by similarity at step 0 (§1.1). It cannot be decorrelated by noise, which separate copies would be after a few hundred Adam steps (§1.4: drift 5× the init scale). It also saves `d²` per layer.
   - **Where.** `memory.rs` `specs` (`:52-69`), `read` (`:130`), `write` (`:149`), `MATRICES` (`:42`), `param_count` (`:72-75`), the parameter-count test (`:378`). About 15 lines, plus a config switch `memory_qk: "shared" | "separate"` in `s2/config.rs` (about 8 lines) so v1 checkpoints still load.
   - **Test.** C1, and C2 with both forms.
   - **Risk.** Low for `recall`. A query that should differ from the key (asymmetric relations) loses expressivity. Rule110 might prefer separate projections.
2. **Values that carry the frame's context**: `v_i = heads(LN(x_i + x̄)·Wv)`, with `x̄` the mean of the block output over the frame's valid tokens (form `frame`), or v1's `v_i = heads(LN(x_i)·Wv)` (form `token`). **Chosen by C2 before training**: the form with the higher readout accuracy on delays 1–8.
   - **Why.** `x̄` contains the colour token's residual linearly, so the door slot can hold "red" without any learned binding (§1.2, Wang et al.).
   - **Where.** `memory.rs` `write` (`:148-151`), about 8 lines, plus a config switch (about 5 lines).
   - **Test.** C2.
   - **Risk.** Medium. Template slots then also carry the colours of recent shows of other doors, which act as confusers. That is why C2 decides between the two forms rather than assuming.
3. **The `mem.*` rate at 10× the base's, `--mem-lr-mult 10`.**
   - **Why.** LoRA+; the zero-initialised read-out; the short budget.
   - **Where.** `optim.rs` next to `head_lr_mult` (`:79-82`, `:257-261`, the same pattern, keyed on the `mem.` prefix), about 6 lines; `main.rs` flag, about 4 lines.
   - **Test.** C4 at 1× and 10×.
   - **Risk.** Faster noise drift if the signal is absent, which C3 detects first. The delta rule cannot diverge (unit keys, β ∈ (0, 1), 1/n), so the instability risk is low.
4. **Phase A, a recall-heavy start without sleep**: the first 40% of the steps with `--seq-frac 0.9`, `--sleep-weights recall=0.85,rule110=0.15` and `--sleep-passes 1`. Then phase B: v1's mix with N drawn in {1, 2, 3, 4}, truncated.
   - **Why.**
     - Lee et al.'s warm-up; RMT and ARMT curricula.
     - It multiplies the recall batches about 3.5× in the same time.
     - N = 1 gives the full gradient path at the lowest cost.
     - The asks of delays 1–4 are already a quarter of all asks, so the data carries its own short-delay curriculum.
   - **Where.** No trainer code: two `s2-decide` calls in `scripts/train-sleep.sh` (phase B with `--init` from phase A's `last` and a step budget), about 30 lines.
   - **Test.** C4 runs phase A's mix.
   - **Risk.** Phase B restarts the optimiser (Adam's moments are lost) and the mix shift may erode what phase A learned. The pilot's in-sample check after phase A tells the two apart. Base drift on the games during phase A is measured by H2 in the long run only.
5. **Per-game sequence-loss logging, with the ask-frame cross-entropy of `recall` separated, and per-group gradient norms** (`mem.*`, `blk.*`, `fusion.*`).
   - **Why.** v1 logged only a mixed mean (`train.rs:1886-1896`), in which recall's possible gain of about 0.07 was buried.
   - **Where.** `train.rs` near `:1885`, about 25 lines (`sl.sample` already returns the game index).
   - **Test.** Present in C4's output.
   - **Risk.** None.
6. **Float memory projections during phase A, ternary QAT on them from phase B** (open choice O5, not recommended first).
   - **Why.** It removes one source of retrieval noise while addressing forms.
   - **Where.** `model.rs` `prepare` (`:440`), with a `--mem-qat-from` step, about 8 lines.
   - **Test.** C4 with and without it.
   - **Risk.** The switch to ternary at phase B could break a fragile retrieval. Evidence that this matters here is weak (§1.5).
7. **Exact per-token delta rule** instead of the 1/n frame average. The chunk is solved in closed form: `T = (I + tril(diag(β)KKᵀ, −1))⁻¹`, and since the strictly lower-triangular part is nilpotent, `(I + A)⁻¹ = Π_{i<⌈log₂ n⌉}(I + (−A)^{2^i})`, five 32×32 matmuls for 32 tokens.
   - **Why.** About 8× stronger writes (§1.2).
   - **Where.** `memory.rs` `write`, about 30 lines.
   - **Not recommended for v2.** The 1/n form is what lets sleep strengthen a weak write (v1's H1), and stronger writes also mean stronger overwrites by the template. Keep it as plan B inside the fast-weight design.
8. **An auxiliary recall loss**, a small head decoding the ask's colour from `r` at the ask frame.
   - **Not recommended.** It needs game-specific labels in the trainer, and phase A gives most of its benefit with no new loss.
9. **Plan B outside fast weights:** a fixed FIFO of the last 32 frames' state tokens, attended to by the state encoder or the fusion head (a cache memory). Attention solves associative recall reliably (Zoology), so the odds of recall above chance would be about 0.8. But it changes the question: a cache of past frames is not a memory consolidated by sleep, and H1 would have no meaning. Keep it as the answer if v2 fails and the owner wants a model that remembers, whatever the mechanism.

**What is kept from v1, and why.**

- `Wo = 0`: a fresh model is its base, and v1's tests rely on it.
- `α = σ(· + 6)`: retention is fine and its frozen gradient is harmless.
- `β` init 0.5.
- The 1/n write, which is the sleep hypothesis.
- The probes, seeds and the 0.10 bar.

## 4. The checks before any GPU time

All run on the CPU (MLX `Device::cpu()`, as the unit tests already do, `memory.rs:393`), on episodes drawn from **train** seeds only, and at `nice -n 19` while a GPU training runs.

- **C1, retrieval algebra (unit test, seconds).**
  - **Setup.** Synthetic hidden states, d = 96: one random unit vector per word plus a shared component tuned so that tokens have mean cosine 0.4. `Wqk = Wv = I` in f32, `Wo` not used, real `write`/`read`.
  - **Episode.** A show frame `show a colour red …`, then `d` idle frames, then `ask a`.
  - **Pass.** At the ask's door token, `r`'s projection on the four colour directions has its argmax at red for d ∈ {1, 4, 8} under form `frame`. The decay for d = 16 and 32 is printed, which turns §1.2's arithmetic into a measured curve. Run under form `token`, the test should fail; it documents why form `frame` exists.
  - This is the "hand-written fact is read back" test that the diagnosis asked for, at the level of the memory layer.
- **C2, readout at init on the real base (new subcommand `s2-memcheck`, minutes).**
  - **Setup.** Load the pico base (`runs/s6lr/softmax-adamw-x2/decide/last`) with fresh memory weights (shared `qk`, both value forms; separate `q`/`k` as v1's reference). Play 2,000 `recall` train episodes through `wake`, taking `r` before `Wo` at the ask frame's door token and its mean over the frame.
  - **Classifier.** Nearest centroid on cosine: centroids from 1,000 episodes, accuracy on the other 1,000. No solver is needed, and it is conservative.
  - **It also reports** the measured `c̄²` between show-frame door keys and idle-frame keys, and a linear-probe accuracy of the colour from the base's own door-token hidden state at the show frame (the binding the base provides).
  - **Pass.** For the chosen form, accuracy ≥ 0.40 on delays 1–8 (chance 0.25; the 1,000 held-out episodes hold about 4,000 asks of that range, a standard error of about 0.008). The chosen form is the higher of the two. v1's form is reported, and expected near 0.25.
- **C3, the first gradient (same subcommand).**
  - **Setup.** At step 0, `∂L/∂Wo` on 10 independent recall batches of 16 episodes; the mean pairwise cosine between them; the gradient norm per group (`mem.*`, `blk.*`, `fusion.*`, `static.*`).
  - **Pass.** Mean pairwise cosine ≥ 0.1 under the v2 form, which is a consistent direction. The v1 form is expected near 0. Every other `mem.*` gradient must be exactly 0, which checks §1.1.
- **C4, a CPU training (about 300 steps, 20–40 CPU minutes).**
  - **Setup.** From the pico base, phase A's mix with recall only, batch 8 sequences, rate 2e-3 with `mem.*` at ×10, N = 1. Run twice: v2 memory, and v2 memory with the read zeroed (`Wo` forced to 0 at every step).
  - **Pass.** The ask-frame cross-entropy, averaged over steps 250–300, is ≤ 1.30 (ln 4 = 1.386), and at least 0.05 below the zeroed run. Running on the CPU needs a `--cpu` switch in `sky-train` that sets MLX's default device (about 5 lines).
  - If C4 fails while C2 passed, the problem is optimisation, not representation. Try `--mem-lr-mult 30` and the float memory (choice O5) once each on the CPU before deciding.

Only when C1–C4 pass does the GPU pilot run.

## 5. What a working s6-sleep is for

- **A game that needs memory, as a demo.**
  - `recall` as a "doors and keys" page on the site: a colour flashes on a door once, and later the door is asked.
  - The model plays it in WASM with its memory shown as a 96 × 96 heatmap that changes at each show and each sleep.
  - A "memory off" switch drops it to one in four.
  - This needs the `sky-infer` memory (§0), about the same code as `memory.rs`'s read and write, plus a parity test on golden sequences (v1 gate item 3).
- **A real game.** A Doom-like variant where a keycard's colour is seen in one room and needed at a door later. It is the same structure as `recall` inside an existing game, and would make H2's arena reading meaningful for memory.
- **A decision model with a session memory.**
  - The project's models answer typed questions about a state in microseconds. With a memory, a fact given once ("deploy to staging", "the user writes in French") is carried to a later decision without being resent in every state, at a fixed cost of 9,216 floats per session for pico.
  - Training data would be made the way `recall` is: fact frames, distractor decisions, question frames, from the existing decision templates.
  - This is the use that justifies the parameters outside games.
- **What it is not for.** Long documents or many facts. At about 6% write strength and with template overwriting, a pico memory will hold a handful of recent facts. That matches the task's 3 doors, not a knowledge base.

## 6. Odds, and how they are reached

- **P(C1 passes) ≈ 0.95.** It is algebra.
- **P(C2 passes for one of the two forms) ≈ 0.75.**
  - Form `frame` puts the colour into the door slot by construction.
  - The risks are a key overlap high enough that idle frames erase the slot within a few frames, and colour confusers from other doors.
  - The 0.25 margin comes from not knowing `c̄²`.
- **P(C4 passes | C2) ≈ 0.85.** About 19,000 asks pass in 300 steps of 8 episodes. With an informative first gradient and 10× the rate, a drop of 0.09 nats in-sample is modest.
- **P(pilot passes M − C ≥ 0.10 at N = 1 | C1–C4 pass) ≈ 0.65.**
  - Solving the two short buckets at 0.6 and the two long at chance gives 0.425, which is +0.16 over C.
  - Solving delays 1–4 alone at 0.75 gives 0.375, which is +0.12.
  - So the bar needs about one bucket solved well, out of sample, on dev seeds, after phase B's mix shift and with ternary memory projections.
  - The 0.35 against comes from: generalisation from at most a few thousand recall batches (an estimate: v1 does not log the time of a sequence step); phase B eroding phase A; the ternary projections; and the probe being closed-loop, with τ = 1 at the last exit.
- **Unconditional:** 0.95 × 0.75 × 0.85 × 0.65 ≈ **0.40–0.45**.
- **Conditional on the checks passing:** 0.65.
- **"Clearly above chance", dev `recall` ≥ 0.32 with the memory-off reading at chance, given the checks:** about **0.8**.
- **Wasted GPU time:** a failed pilot costs about 35 working minutes, and the in-sample check after phase A can stop the run after about 13 (8 for the memory arm's phase A, about 5 for the control's).

## 7. What could not be determined without running

- The key overlap `c̄²` of the base's hidden states, and so the real retention curve (C1, C2).
- Whether the base's door token already carries its colour after 4 loops, which would make form `token` enough (C2's linear probe).
- Whether the drift in §1.4 is purely noise. P2's smaller drift is unexplained.
- How fast learning goes with the fix, and whether 10× is too much or too little (C4, pilot).
- Whether phase B's mix and the return of sleep (N > 1) erase or strengthen what phase A stored.
- Whether ternary memory projections cost retrieval at this width (choice O5).
- Whether sleep helps in v2 (v1's H1). Nothing in this study predicts it. The 1/n write makes it possible, not likely.
- The cost per step of the shared projection and of form `frame`, which is expected to be within noise.
