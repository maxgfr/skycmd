# 23 s5-lite: invariant by construction, tested in the open

s4 failed: 0.343 on the generic decision bench, below the 0.398 of always answering the majority class ([chapter 22](/tutorial/22-s4-generic-decisions)). s5 changes three things at once:

- **The starting point.** It starts from a pretrained encoder (Ettin-17M) instead of from scratch.
- **The label sources.** It learns from Apache-2.0 teachers and code generators instead of a hosted LLM whose terms forbid training on its outputs.
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

The z.ai terms forbid using GLM outputs to train models, so no GLM output enters s5: training, validation, calibration, thresholds and published sets are all free of it. The labels come from five sources:

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

The verdicts, with their 95% intervals (paired bootstrap over items and seeds), are on [the results page](/results/). That page reads them from the analysis files, so nothing is copied by hand.

## What is not done yet

- **s5 in French** on mmBERT-small, with a pruned vocabulary.
- **The baselines that score labels** through the MLM head.
- **Q4 quantization.**
- **The online escalation bet**, with an audited share of the accepted answers.

The plan and its order are in the pre-registration.
