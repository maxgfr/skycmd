# s6-sleep v2 (s6-sleep2-pico pilot, then s6-sleep2-nano): pre-registration

**Status: frozen on 15/11/2026.** No line of v2's code is written, no v2 check has run, no v2 model exists. This file fixes:

- the memory form;
- the checks that must pass before any training;
- the pilot and its decision rule;
- the nano runs that follow a passed pilot;
- the hypotheses and the gate.

One thing is fixed later, by a rule written here before any training: the value form (§3, C2). The design study, `docs/s6-sleep2-study.md`, is committed with this file and holds every number quoted here.

**Why this experiment.** v1's pico pilot found recall at chance in every arm: 0.246, 0.2565 and 0.243 with memory against 0.265 without (`runs/s6-sleep/pilot.json`). The diagnosis (`docs/s6-sleep-diagnosis.md`) found the memory wired, written, carried, read and trained, but never used: the training loss equals the control's within ±0.004. The study traces this to an optimisation failure at initialisation:

- the read-out is zero and the query and key projections are independent, so the first gradient of the memory is noise;
- the memory weights drifted at the size of an Adam random walk.

The owner wants a memory that works, even if imperfect, without GPU hours spent on poor odds. v2 changes how the memory starts and how it is first trained, keeps v1's probes and bars, and spends no GPU time before four CPU checks pass.

**Changes to this file:**

- every amendment is committed, dated and justified in §13;
- an amendment comes before the run or the measurement it concerns;
- an amendment never moves a threshold after a v2 dev or test score has been read.

**Measurements allowed before the freeze**, written down with their date:

- C1 to C4 of §3, on train seeds only;
- throughput.

None of them reads a dev or test seed of `recall` or `rule110`.

## 1. The question

Can a fast-weight memory with sleep, added to a sub-megabyte ternary decision model, learn to carry a fact across frames, when:

- its addressing works from the first step;
- what it writes already holds the fact;
- it is first trained where the fact is what the loss asks for?

**Works** means the pilot rule of §5.3: on `recall`, at least 0.10 above the memoryless control, and the gain disappears when the memory is zeroed.

v1's other questions are unchanged and asked only if the pilot passes:

- does sleep help (H1)?
- do the games and the language tasks not lose (H2, H3)?

## 2. What changes from v1, and why

| # | change | v1 evidence (study §1) | external evidence (study §2) |
|---|---|---|---|
| M1 | **One shared query/key projection** `mem.{l}.qk` per memory layer, instead of `q` and `k`. | With `Wo = 0` (`memory.rs:64-65`), the gradients of every other memory weight are exactly zero at step 0. With independent `Wq` and `Wk` (`:57-63`), `q·k` does not depend on whether two tokens name the same door, so `Wo`'s first gradient is noise. Over the pilot, `q`, `k`, `v` and `Wo` drifted to the size of an Adam random walk (about 0.07 predicted against 0.08 observed in P1 and P3, against up to 3.7 under a consistent gradient). | Reformer: a shared query-key space "does not perform worse"; Hopfield-style retrieval by similarity; fixed token keys make a small DeltaNet recall (arXiv 2610.00232). |
| M2 | **Values that carry the frame**: form `frame`, `v_i = heads(LN(x_i + x̄)·Wv)` with `x̄` the mean block output over the frame's valid tokens, or v1's form `token`. **Chosen by check C2** (§3). | The colour is two tokens after the door (`recall.rs:99-109`). In form `token`, the door slot holds the colour only if the base's attention put it there, and the base never saw `recall`. | A one-layer associative memory recalls only when key and value are built from neighbouring tokens (Wang, Shi, Fox, arXiv 2501.12352, §6). |
| M3 | **The `mem.*` parameters at 10× the base's AdamW rate.** | One shared rate (`optim.rs:245-261`; `scripts/train-sleep.sh:72`); a zero-initialised read-out. | LoRA+ (arXiv 2402.12354): a zero-initialised factor needs a rate λ ≫ 1 times its partner's. |
| M4 | **Phase A**: the first 40% of the steps with sequence steps at 0.9, sequences `recall` 0.85 and `rule110` 0.15, and no sleep (N = 1). **Phase B**: v1's mix and N drawn in {1, 2, 3, 4}, truncated. | About 740 recall batches in 3,684 steps (8 asks in 64 frames each). Every other frame pulls toward a silent memory. | Lee et al. (arXiv 2605.26099) warm up the memory layers alone first and call the warm-up crucial; RMT without a curriculum "does not learn"; ARMT uses a segment curriculum. |
| M5 | **Logging per game of the sequence loss, with `recall`'s ask-frame cross-entropy apart, and the gradient norm per group.** | v1 logged a mixed mean in which recall's possible gain (about 0.07) was invisible (`train.rs:1886-1896`). | Larson (arXiv 2610.00232): zeroing the memory as the test of use, adopted in §5.3. |

**Not changed, and why.**

- **`Wo = 0` at init.** A fresh model is exactly its base; v1's tests rely on it. Once M1 makes `r` informative, a zero read-out trains, as ControlNet's zero convolutions do.
- **The decay gate `α = σ(· + 6)`.** It keeps 92% over 32 frames. Its gradient is frozen (about 5e-12, below Adam's eps), which is harmless.
- **`β` at init (0.5) and the 1/n frame write.** The weak per-write strength (about 6%) is what replays can strengthen, so it is the condition under which H1 means something. The exact per-token delta rule is recorded as plan B (study §3, item 7).
- **Ternary QAT of the memory projections from step 0.** The STE passes gradients through a zero `Wo` (`model.rs:378-386`). The evidence that ternary projections hurt retrieval is weak. Choice O5 records the alternative.
- **Truncated backpropagation.** P3, with backpropagation through every pass, failed like the others, so truncation was not the cause. v1's truncation and drawn-N decisions were taken at chance level and carry no information. v2 adopts v1's fast path (P1's form) without a new test: choice O7.
- **The probes, the training seeds and sequences, the arms, the hypotheses H0 to H3, the gate.** These are v1's (`docs/s6-sleep-prereg.md` §3, §7, §8), except the dev seeds of §5.2.

## 3. Checks before any training

All checks run on the CPU (MLX `Device::cpu()`) at `nice -n 19`, on `recall` episodes from **train** seeds only, from the pico base `runs/s6lr/softmax-adamw-x2/decide/last`. Their outputs go to `runs/s6-sleep2/checks/`. No GPU training starts before C1 to C4 pass.

- **C1, retrieval algebra (unit test in `memory.rs`).**
  - **Setup.** Synthetic hidden states at d = 96: a random unit vector per word plus a shared component, tuned so that tokens have mean cosine 0.4. `Wqk = Wv = I` in f32, and the real `write` and `read`.
  - **Episode.** A show frame `show a colour red ask none noise 46`, then `d` idle frames, then `ask a`.
  - **Pass.** Under form `frame`, the projection of `r` at the ask's door token on the four colour directions has its argmax at red for d = 1, 4 and 8. The values for d = 16 and 32 are printed and recorded.
- **C2, readout at initialisation (`sky-train s2-memcheck`).**
  - **Setup.** The base with fresh v2 memory weights, under each value form, and v1's form (separate `q`/`k`, form `token`) as reference. 2,000 train episodes are played through `wake`, collecting `r` before `Wo` at each ask frame's door token.
  - **Classifier.** Nearest centroid on cosine: centroids from 1,000 episodes, accuracy on the other 1,000.
  - **Also reported:**
    - the mean squared cosine between show-frame door keys and idle-frame keys;
    - a nearest-centroid accuracy of the colour from the base's door-token hidden state at the show frame.
  - **Pass.** The better of the two v2 forms reaches ≥ 0.40 on the asks of delays 1–8 (chance 0.25; about 4,000 asks, standard error about 0.008). **That form is adopted**; the other is recorded as not taken. If both reach 0.40 and differ by less than 0.02, `frame` is taken.
- **C3, the first gradient (`s2-memcheck`).**
  - **Setup.** At step 0, `∂L/∂Wo` on 10 independent `recall` batches of 16 episodes, and the gradient norm of every group (`mem.*`, `blk.*`, `fusion.*`, `static.*`).
  - **Pass.** Both of these hold:
    - the mean pairwise cosine of the ten `∂L/∂Wo` is ≥ 0.10 under the adopted form;
    - every other `mem.*` gradient is exactly zero.
  - v1's form is reported next to it.
- **C4, a CPU training.**
  - **Setup.** 300 steps of phase A's mix with recall only, batch 8 sequences, rate 2e-3 with `mem.*` at ×10, N = 1, QAT as in the pilot. Run twice from the same seed: the v2 memory, and the v2 memory with its read forced to zero.
  - **Pass.** The ask-frame cross-entropy averaged over steps 250–300 is ≤ 1.30 (ln 4 = 1.386) and at least 0.05 below the zeroed run.
  - **Retries, the only ones allowed.** If C4 fails after C2 passed, it is run again once with `mem.*` at ×30 and once with float memory projections. If either passes, that setting replaces M3 or O5 by amendment, before any GPU run.

**If C2 fails for both forms**, or C4 fails after its retries, no GPU run takes place. The report says which check failed and with what numbers.

## 4. The model

`s6-sleep2-pico` (pilot) and `s6-sleep2-nano` (nano runs) are v1's configurations with `memory_qk: "shared"` and `memory_value` set by C2.

**Memory layer l**, on the output `x` of block l (`[tokens, d]`):

```
read:   q = l2norm(heads(LN(x)·Wqk))                   x ← x + merge(S·q)·Wo
write:  k_i = l2norm(heads(LN(x_i)·Wqk))               v_i = heads(LN(x_i [+ x̄])·Wv)
        β_i = σ(wβ·x_i)    α = σ(mean_i(wα·x_i) + 6)   S ← α·S + (1/n)·Σ_i β_i·(v_i − α·S·k_i)·k_iᵀ
```

`[+ x̄]` is present in form `frame` only. Wake, sleep, K = 8 and the episode boundaries are as in v1 §2.

**Parameters.**

| | per memory layer | pico | nano |
|---|---|---|---|
| v2 | `3d² + 3d` | 27,936 (model 473,281) | 125,280 |
| v1 | | 37,152 | 166,752 |

The state per stream is unchanged: 9,216 floats (pico), 20,736 (nano). `Wqk`, `Wv` and `Wo` are ternary in the export; the gates, the norm and the state stay f32.

## 5. The pico pilot

### 5.1 Arms and budget

| arm | memory | phase A (first part) | phase B |
|---|---|---|---|
| M | v2 | **8 working minutes**, N = 1 | **12 working minutes**, N drawn in {1, 2, 3, 4}, truncated |
| C, the control | none | the same number of steps as M's phase A | the same number of steps as M's phase B |

- **Shared settings.** Both arms: seed 0, the same base, the same batches (N is drawn in both, as in v1), and `mem.*` at ×10 in M. The peak rate is v1's (`rate 96 1`).
- **How the phases run.** Phase B starts from phase A's `last` with a fresh optimiser and schedule (`s2-decide --init`), in both arms.
- **Budgets.** M's budgets are in working minutes and C's are in M's steps (v1 §4). Total: about 35 working minutes of training.
- **Stop after phase A.** If M's ask-frame cross-entropy, averaged over its last 50 `recall` steps of phase A, is above **1.336** (ln 4 − 0.05), phase B and arm C do not run. The pilot is reported as failed in-sample.

### 5.2 Readings

- **Seeds: dev2**, the first 500 episode seeds `≡ 19 (mod 20)` from 2,000,000,000 upward, never read by v1. v1's dev seeds were read at chance level and are not reused for decisions.
- **Reader.** `sky-train s2-probe`, ternary weights, last exit, closed loop, as in v1.
- **What is read.** `recall` and `rule110` at N = 1, 2 and 4 for M, and once for C. Plus **M with its memory zeroed before every frame** (`--memory-off`) on `recall` at N = 1.
- **Reported, deciding nothing.** Accuracy per delay bucket; `rule110`; the training curves of M5.

### 5.3 Decision rule: the memory works

Both of these must hold:

1. **M at N = 1 − C ≥ 0.10** on dev2 `recall`, with the lower bound of the paired bootstrap 95% interval over episodes (10,000 draws) above 0. This is v1's bar, kept.
2. **M with its memory zeroed − C ≤ 0.03**: the gain comes from the memory, not from a better prior.

- If both hold, the nano runs of §6 take place.
- If (1) holds and (2) fails, no nano run takes place. The reading is treated as a defect to explain in an amendment before anything else runs.
- If (1) fails, see §11.

## 6. The nano runs, if the pilot passes

v1 §6, with the v2 memory, kept as written except for these changes:

- **Base.** The s6-nano of tranche 2, under v1's exception for a base that does not play.
- **Arms.** Two arms, 3 seeds (0, 1, 2) each: `s6-sleep2-nano` and the control.
- **Budget.** The sleep arm at seed 0 trains for **30 working minutes**: phase A 12 minutes, phase B 18. The five other runs train for its steps, phase by phase.
- **No DAgger round.** Calibration and export are as in v1. The exported graph needs the `sky-infer` memory of §15, written and checked for parity before any export.
- **Choice of N** by v1's rule, on dev2.
- **Hypotheses H0 to H3** exactly as v1 §7, read once on v1's test seeds (`≡ 19 (mod 20)` from 5,000,000,000), which have never been read. The arena is as in v1.

## 7. Gate

v1 §8 unchanged. `s6-sleep2-nano` (seed 0, the chosen N) is published under that name only if:

- the s6-nano gate holds;
- H0, H1, H2 and H3 hold;
- `sky-infer` matches the training code on golden sequences, natively and in WASM.

## 8. Demo (reported, not gated)

If H0 holds on the nano runs, the project page for this model is a "doors and keys" game:

- `recall` rendered in the browser and played in WASM, with the memory shown as a heatmap, a mark at each sleep, and a "memory off" switch;
- the per-bucket accuracy from the test reading next to it.

The session-memory use (a fact given once and used by a later decision) is described in the study, §5. It needs its own data and is not part of this file.

## 9. Cost and modes

- **CPU checks:** about 1 to 1.5 CPU hours in all, C4 being most of it. They may run during another GPU training, at `nice -n 19`.
- **Pilot:** about 35 working minutes of training, plus CPU probes.
- **Nano runs:** about 3 hours 30, as in v1.
- **Brake modes:** as in v1 §9 and its amendments 3 and 5. Nothing of the pilot or the nano runs trains during another training.

## 10. Disclosure

v2 was designed after reading v1's pilot:

- its logs;
- its dev scores, all at chance;
- its checkpoints and optimiser states;
- its training sequences.

The changes answer an optimisation failure seen in training and in the weights, not a test score. v1's test seeds of `recall` and `rule110`, and the arena seeds of this experiment, have never been read. v1's dev seeds were read; v2 decides on a fresh dev set. Every report of v2 states this paragraph.

## 11. What a failure would mean

- **C2 fails for both forms.** Under the 1/n frame write and this base, the memory cannot hold a door's colour in a retrievable form at initialisation. No GPU time is spent. The next step, by a separate decision and a new pre-registration, is one of two:
  - the exact per-token delta rule;
  - a cache of the last 32 frames (study §3, items 7 and 9).
- **C4 fails after its two retries.** The memory can represent the fact but does not learn it in 300 CPU steps. No GPU time is spent; the same two options apply.
- **The pilot stops after phase A,** or (1) of §5.3 fails. A fast-weight memory with sleep, at pico scale and with a 20-minute budget, does not learn `recall` even with working addressing. The s6-sleep fast-weight line stops, and the report says so with C1 to C4's numbers. A model that remembers is still possible with a frame cache. That is a different question, with no role for sleep, and needs its own file.
- **The pilot passes, the nano H0 passes, H1 fails.** The memory works and sleep does not help. As v1 §8 says, nothing is published as `s6-sleep2-nano`. Whether the file is published as a result, as for the project's other failed gates, is the owner's decision. The memory itself becomes available as a plain recurrent memory, without sleep, by a separate decision.
- **In every case** the report is published, failures included.

## 12. Forbidden

- Reading a dev2 seed before C1 to C4 pass, or a test seed before every nano run has ended and N is chosen.
- A second pilot seed, a second pilot, or a pilot arm not listed in §5.1, except after a technical failure (a crash, a full disk), which restarts the same run with the same seed.
- Any CPU retry of C4 beyond the two of §3.
- Training on episodes from the validation or test splits (v1's `split_for_seed`).
- Changing a bar of §3 or §5.3 after a dev2 score is read.

## 13. Amendments

None yet.

## 14. Choices (the first value of each line is the one taken at the freeze; the alternatives are recorded)

| | choice | recommended | alternatives | why the recommendation |
|---|---|---|---|---|
| O1 | query/key | **shared `Wqk`** | separate (v1); separate initialised equal | retrieval by similarity at step 0; separate copies decorrelate under noise within a few hundred steps (study §1.4) |
| O2 | value form | **by C2's rule** | `frame` fixed; `token` fixed | the binding the base provides is unmeasured; C2 measures it at no GPU cost |
| O3 | `mem.*` rate | **×10** | ×3; ×16 (LoRA+'s default); ×30 | the delta rule cannot diverge (unit keys, β < 1, 1/n), so a large factor is safe; ×30 is C4's retry |
| O4 | phase A share | **40% of the steps** | 25%; 60%; phase A only | enough recall batches for the short buckets while leaving phase B time for the mix and for sleep |
| O5 | memory QAT | **ternary from step 0** | float in phase A, ternary in phase B; float throughout (pico +110 KB, nano +498 KB in f32) | the STE trains through zero weights and the export stays ternary; float is C4's retry |
| O6 | sleep in phase A | **none (N = 1)** | N drawn | full gradient path at the lowest cost; sleep is what H1 tests, later |
| O7 | phase B form | **N drawn in {1, 2, 3, 4}, truncated (v1's P1)** | N = 4 fixed; backpropagation through every pass | v1's tests of these were at chance and carry no information; P1's form is the cheapest |
| O8 | dev seeds | **fresh dev2** | v1's dev seeds | v1's were read; a fresh set costs nothing |
| O9 | pilot budget | **8 + 12 working minutes** | 10 + 15; 6 + 9 | the owner's limit of well under an hour of training, with both arms about 35 minutes |
| O10 | in-sample stop after phase A | **yes, at 1.336** | none | saves about 20 minutes of GPU time when phase A has learned nothing |
| O11 | phase A mix | **`recall` 0.85, `rule110` 0.15, sequence steps at 0.9** | `recall` 1.0; v1's mix with more steps | keeps a second memory task in view and some stateless steps against drift of the language tasks |

## 15. Code to write, by file

| file | what | about |
|---|---|---|
| `crates/sky-train/src/s2/config.rs` | `memory_qk` (`"separate"` by default, so v1 checkpoints load) and `memory_value` (`"token"` by default) | 15 lines |
| `crates/sky-train/src/s2/memory.rs` | shared `qk` in `specs`, `read`, `write`, `MATRICES` and `param_count`; form `frame` in `write`; tests: updated counts (27,936 pico and 125,280 nano under v2), C1, "at step 0 only `mem.out` has a gradient", "a zeroed read leaves the logits of v1's base" | 30 lines of code, 120 of tests |
| `crates/sky-train/src/optim.rs` | `mem_lr_mult` for names starting `mem.`, next to `head_lr_mult` | 8 lines |
| `crates/sky-train/src/main.rs` | `--mem-lr-mult`, `--cpu` (MLX default device), the `s2-memcheck` subcommand, `s2-probe --memory-off` | 30 lines |
| `crates/sky-train/src/s2/memcheck.rs` (new) | C2 (episodes, `r` at ask door tokens, nearest centroid, key overlap, base binding probe) and C3 (pairwise gradient cosines, gradient norm per group) | 180 lines |
| `crates/sky-train/src/s2/train.rs` | per-game sequence loss and `recall` ask-frame cross-entropy in the log; gradient norm per group every 50 sequence steps; a read-zeroing switch for C4's control | 35 lines |
| `crates/sky-train/src/s2/probe.rs` | `--memory-off`: the memory reset before every frame | 10 lines |
| `scripts/train-sleep.sh` | `checks` and `pilot2` modes: C1 to C4, then M phase A, the in-sample stop, M phase B, C phase A and B, the dev2 probes; `nano2` mode for §6 | 70 lines |
| `scripts/s6-sleep-report.py` | the §5.3 rule, the bootstrap, the per-bucket and memory-off readings, `runs/s6-sleep2/pilot.json` | 60 lines |
| `crates/sky-infer/src/s2/` | the memory read, write and sleep in the engine, and golden-sequence parity, **only if the pilot passes** | 200 lines |

**Amendment 1 (15/11/2026, after C1 failed and before anything else ran; no v2 training exists, C2 and C4 have not run, no dev or test seed has been read). C1 failed; why; and how the write rule is revised before the checks go on.**

- **C1, as read.** With synthetic states at d = 96 (mean cosine between words 0.401, seed 0), under both value forms, the colour read at the ask's door is right after 1 idle frame and wrong after 4 and 8 (and after 16 and 32). §3 lets no GPU training start, and §11 had no branch for a failed C1: this amendment is that branch.
- **What the gradient report showed against the study.** At step 0 every memory gradient but the read-out's is exactly zero in v1 and in v2 alike (it has to be, with a read-out at zero), and the cosine between batches of the read-out's gradient is 0.79 for v1, 0.72 for v2: C3's bar of 0.10 does not separate the two forms. C3 is kept as a sanity check and is no longer read as evidence for v2.
- **The cause C1 points to.** The write is `S ← αS + (1/n)·Σ β_i (v_i − αS k_i) k_iᵀ` with β near 0.5 at the start: one frame moves the memory by a small fraction of what would store its pairs, and every later frame adds its own pairs along keys that are not orthogonal to the door's (cosine 0.4). After a few idle frames the door's slot holds mostly other frames. A sleep pass replays the same small additive step, which is why v1's scores were the same at 1, 2 and 4 passes.
- **The revision, chosen by algebra on synthetic states only.** A sweep runs on the C1 setting (synthetic states, no game episode from any split, no training, CPU) over these write rules, in this order of preference: (W1) the delta rule token by token at full step, `S ← S + (v_i − S k_i) k_iᵀ` for each token in turn (keys have unit norm), with the gates α and β multiplying as before but initialised so that a fresh model takes full steps; (W2) one exact step per frame, `S ← S + (V − S Kᵀ)(K Kᵀ + λI)⁻¹ K` with λ = 1e-3; (W3) the rule of v1 with keys centred on the mean key of the frame. For each rule the sweep records the colour decoded at the ask after d = 1, 4, 8, 16 and 32 idle frames, with 1, 2, 4 and 8 passes over the frames (the wake pass included). **The first rule of the list that decodes the right colour at d = 1, 4 and 8 with at most 4 passes is adopted**, with the smallest number of passes that does it as the pilot's N; if none does, v2 stops here and the report says so. The sweep's table is committed with the choice, before C2.
- **What this makes of the hypothesis.** If a rule passes only with more than one pass, then "sleep helps" has a precise meaning that v1 could not test: replaying the same frames with an error-correcting write removes the interference between memories. The table of decoded colours by number of passes is reported with the pilot.
- **Readings fixed as implemented:** in C1, `d` is the number of idle frames between the show frame and the ask; the word vectors are drawn once with seed 0; "ask-frame cross-entropy" in C4 and §5.1 is the loss's own deep-supervision measure; the pilot lives in `scripts/train-sleep2.sh` and `scripts/s6-sleep2-report.py`; a failed C1 or C3 stops the script.
- **What does not change:** C2 and C4 and their thresholds, the pilot, its budget and its decision rule, §12.

**Amendment 2 (15/11/2026, after the sweep of amendment 1 and before anything else ran; no v2 training exists, C2 and C4 have not run, no dev or test seed has been read). The sweep's result, a doubt about what C1 measures, and the one diagnostic that settles it.**

- **The sweep, as read.** No rule passes: under form `frame`, v1's rule decodes the right colour only at 1 idle frame, and W1, W2 and W3 decode the same wrong colour (blue) at every delay and every number of passes, under form `token` too. By amendment 1, v2 stops here, unless what follows shows that the criterion did not measure retrieval.
- **Why the criterion is in doubt.** W2 stores a frame's pairs exactly: after the show frame alone, reading the door's key returns the door's value, by construction. Yet it decodes blue after one idle frame with a margin as large as after thirty-two, and so do the two other rules. A colour that never changes with the delay, the rule or the number of passes looks like a bias of the decoder and not like forgetting: the value stored under form `frame` is the door's state plus the mean of the frame, in which the colour word weighs one token in about nine, next to words that are the same in every episode; and C1 reads the colour by the largest raw projection on four word vectors drawn once (seed 0). A constant word that happens to lean toward one colour vector would decide every reading.
- **The diagnostic, decided before it is run.** On the same synthetic setting: (D1) with W2, after the show frame alone (no idle frame), check numerically that reading the door's key returns the stored value, and decode the colour from the stored value itself with C1's raw projection, for each of the four colours shown in turn and for 20 draws of the word vectors (seeds 0 to 19). (D2) the same reading decoded contrastively: the four readings of one episode played with each colour, minus their mean, each matched to the nearest colour direction.
  - **If the stored value itself decodes right under C1's raw projection** in at least 90% of the 80 cases, the criterion was sound, the rules really fail, and v2 stops as amendment 1 says.
  - **Otherwise the criterion is replaced**: C1 is read with the contrastive decoding of D2 over the 20 draws, "right" meaning the four colours of an episode all matched in at least 90% of the draws; the sweep of amendment 1 is run once more with it, same rules, same order, same adoption rule, and its table replaces the first one, which stays on record. There is no further revision of C1 after that.
- **Unchanged:** C2 (which decodes by centroids over a thousand real episodes and so is not exposed to this bias), C4, the pilot and its rule, §12.

**Amendment 3 (15/11/2026, after the diagnostic of amendment 2 and before anything else ran; no v2 training exists, C2 and C4 have not run, no dev or test seed has been read). C1's criterion was biased; read as corrected, the pre-registered form passes it and the revision of amendment 1 is withdrawn.**

- **The diagnostic, as read.** D1: with exact storage and no idle frame, reading the door's key returns the stored value to within 3.1e-3, and C1's raw projection decodes that stored value right in 47 of 80 cases (59%), under the 90% of amendment 2: the criterion of C1 did not measure retrieval. D2: the contrastive decoding matches the four colours in 20 of 20 draws. C1 is therefore read contrastively over 20 draws, as amendment 2 decided.
- **The sweep with the corrected criterion** (form `frame`, draws right out of 20 at 1, 4, 8, 16 and 32 idle frames; right means at least 18): the write rule of v1, which §4 of this file kept for v2, is right in every cell at one pass: 20, 20, 20, 20, 18. The three rules of amendment 1 are worse: W1 13, 12, 11, 8, 4 (15 at best with up to 4 passes), W2 13 then 0, W3 0. None of them is adopted.
- **What follows.** Amendment 1 revised the write rule because C1 had failed for the pre-registered form. With the criterion corrected, the pre-registered form (the write rule of v1, a shared query and key projection, values that include the frame) passes C1 at every delay, and the analysis of amendment 1 was wrong: storing each frame exactly overwrites what earlier frames stored along the directions keys share, and the small gated step does not. **The revision of amendment 1 is withdrawn, its clause "if none does, v2 stops here" with it**: it was written for the case where the pre-registered form could not retrieve, and that case did not occur. The write rule stays as §4 wrote it; the checks go on with C2, then C3 as a sanity check, then C4, unchanged. Both sweeps stay on record (`runs/s6-sleep2/checks/sweep.md`, `sweep2.md`).
- **What this costs the hypothesis about sleep.** With this rule one pass already retrieves at every delay tried, and more passes change nothing in the sweep: the checks say nothing in favour of "sleep helps". The pilot is about whether the memory is used at all; its arms and its rule (§5) are unchanged, and N stays 1 for the pilot as §5 wrote it.
- **Said plainly:** this is the third amendment on C1 in one day. Each was written before the measurement it concerns; the criterion that is kept is the one whose decoder was shown to read a stored value correctly, and C2, which decodes by centroids over a thousand real episodes, is the independent test of the same thing.
