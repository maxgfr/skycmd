# 23 s5-lite: invariant by construction, tested in the open

s4 failed: 0.343 on the generic decision bench, below the 0.398 of always answering the majority class ([chapter 22](/tutorial/22-s4-generic-decisions)). s5 changes three things at once:

- **The starting point.** It starts from a pretrained encoder (Ettin-17M) instead of from scratch.
- **The label sources.** It learns from Apache-2.0 teachers and code generators instead of a hosted LLM.
- **What it claims.** It makes no claim that was not fixed in advance.

## Three bets, fixed before any test

The [pre-registration](https://github.com/maxgfr/skycmd/blob/main/docs/s5-prereg.md) fixes, before any test run:

- the frozen evaluation sets and their sha256;
- the renderings;
- the run budget;
- the validation-only selection rules;
- the metrics, the margins and the success criteria.

`sky-eval` enforces it. On the test split of a frozen set it refuses to run in two cases:

- `docs/s5-prereg.md` is not committed, or has local changes;
- the set's hash is not listed in it.

Every test pass is logged with the model's hash and the commit of the pre-registration.

1. **Invariance by construction (bet 1).** An architecture where the options cannot see each other and their keys are never read, so permutations and label swaps are impossible to fail. The bet is about what it costs: accuracy and calibration against the same model in the usual format, at the same data. The margins are 2 points of accuracy and 0.01 of ECE.
2. **Coherence across questions (bet 2).** Ask "is the answer x?" and "is the answer not x?" as separate questions: the two probabilities should sum to 1, and the "is x" probabilities over all options should sum to 1 as well. Bet 2 trains these identities with three small losses. It then measures one it never trained on, monotonicity: "x or y" must be at least as likely as "x". That measurement only counts if the model is competent on "x or y" questions at all.
3. **Quantization (Phase 5) and escalation (Phase 7).**
   - Phase 5: does 8-bit quantization keep the calibration, once recalibrated?
   - Phase 7: can a threshold guarantee that the answers the model gives alone are wrong at most α of the time (Learn-then-Test)?

## The iso architecture

Variant A is the Laya format used by oscar and s1:

```text
[CLS] "choice question: Which category?" [SEP] [MASK] " billing: Invoices…" [MASK] " bug: …" [SEP] state [SEP]
```

There, every option sees every other option and its own key. In **iso** (variant B) the attention mask has three segments:

```text
state      → sees the state
question   → sees the state and the question
option i   → sees the state, the question and option i only
```

**Positions.** In B-i, every option starts at the same offset L, right after the question. In B-ii, every option is reset to the same logical position. The sliding window of the local layers works on these logical positions.

**Readout.** Each option is the mean of its span, h. It is scored by an MLP on [h, q, h ⊙ q, |h − q|] against the question summary q, then a softmax over the options.

**What the model reads.** The keys are never rendered: `billing: Invoices, charges…` reaches the model as `Invoices, charges…`. A true/false question without definitions reads "The statement is false." and "The statement is true.".

**What follows, exactly:**

- the options are invariant to permutation and to renaming;
- s + n = 1 holds when a negation only swaps the two definitions;
- the state encoding is independent of the question. It is computed once per state and cached in the browser.

The tests check these to 1e-6, natively and on wasm32. They are guarantees, not results: a model can satisfy them and still be wrong.

## Data without a hosted LLM

No GLM output enters s5: training, validation, calibration, thresholds and published sets are all free of it. The labels come from five sources:

- **gold labels** of permissively licensed datasets;
- **a teacher**: laya-typed-decisions (Apache-2.0), run in Rust through mlx-rs, two option orders averaged. It is calibrated per group, and kept only where it agrees with the gold;
- **code generators**:
  - written rules with 50 % violations, and minimal pairs;
  - tone pairs;
  - buried facts, and a uniform target when the evidence is missing;
  - "none of the above" with the right option removed on a quarter of the choice questions;
  - paraphrase and distractor pairs;
- **coherence families** of the typed-decisions training states: the choice, "is x" and the reformulated "is not x". The "x or y" questions only exist in the test families;
- **hand-written files**: class definitions of the evaluation sets, evaluation paraphrases and the demo cases.

Decontamination removes any training record that covers half of an evaluation or demo state in 13-grams, or is a MinHash near duplicate (Jaccard ≥ 0.8).

## Training

The recipe:

- full fine-tuning with AdamW (head learning rate × 20), warmup then linear decay, EMA of the weights;
- loss: soft cross-entropy + Brier + RPS on the ordered `score` questions;
- a symmetric KL between the members of each pair, the same for every variant;
- one augmentation stream for all variants: order, random keys, true/false polarity.

The variants:

- **A** is the Laya format as is.
- **A′** adds the random keys and polarity.
- **B-i / B-ii** are iso.

Everything is chosen on validation only:

- learning rate (5e-5, 1e-4, 2e-4) and initialization (Ettin encoder or Ettin reranker), on short runs;
- then three seeds per finalist.

## Results

The full tables, with 95% intervals (paired bootstrap over items and seeds), are on [the results page](/results/), read from the analysis files. In short, every pre-registered criterion was missed, each for a reason worth knowing.

### Iso is more accurate but calibrated worse

On the fitted bench, B\* (iso) reaches 0.674 against 0.429 for A′ (Laya format), three seeds each. The ECE margin fails: 0.088 against 0.057.

The cause shows on validation. B\* is underconfident on typed decisions (confidence 0.59 for an accuracy of 0.71). Its temperatures were fitted on a validation split dominated by the corpus, and they do not transfer.

NLL and Brier, proper scores that also reward calibration, favour B\*. Calibrating per source would probably fix the ECE, but it was not pre-registered, so bet 1 stays a failure.

### Choosing a learning rate on short runs is risky

All 12 exploration runs lasted 600 steps. Both variants picked the highest rate, 2e-4. At 2000 steps the A′ finalists became unstable: one seed plateaued after a gradient spike at step 400.

The sensitivity analysis declared during the night (A′ at 1e-4) narrows the gap to +0.079 in accuracy. It does not change the verdict.

### Coherence losses work on what they train

On A′, λ = 0.1 cuts \|Σ s − 1\| from 0.79 to 0.12 and \|s − c\| from 0.19 to 0.04. The held-out identity, monotonicity, could not be measured: the model at λ = 0 is no better than the majority class on "x or y" questions.

### Quantization costs nothing measurable

Per seed, q8_0 moves the ECE of B\* by less than 0.003 and its accuracy by at most 0.3 points. The pre-registered interval resamples seeds independently, which makes it too wide for the 0.01 margin, so Phase 5 is reported as failed.

### The model on /live

The model on /live is a longer B\* run: 8000 steps instead of 2000, declared during the night and outside the bets. It reaches 0.696 on the fitted bench (oscar-1-17m: 0.682 on the same records) and 21 of the 30 English demo cases. It is a 20.7 MB GGUF that answers in about 20 ms in the browser.

## What is not done yet

- **s5 in French** on mmBERT-small, with a pruned vocabulary.
- **The baselines that score labels** through the MLM head.

The plan and its order are in the pre-registration.
