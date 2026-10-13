---
title: Results
outline: [2, 3]
---

# s5 results: what works, what does not

s5 is a small decision model (Ettin-17M for s5-lite, English) tested on three bets, each judged against criteria fixed before any test run. The [pre-registration](https://github.com/maxgfr/skycmd/blob/main/docs/s5-prereg.md) was committed first. The evaluation tool refuses the test split of a frozen set unless that file is committed and lists the set's hash, and it logs every test pass.

Failures are published like successes. "Not significant" never counts as a success; only non-inferiority within a margin fixed in advance does.

Every property below carries a status:

- **guaranteed**: holds by construction and is checked by an exact test (to 1e-6, natively and in WASM). It is not an experimental result.
- **learned**: trained for, measured on held-out data.
- **held out of training**: never in the loss or the targets, so a gain there is a real generalization.
- **result**: a measured cost or benefit.

## What is guaranteed by construction (s5 iso architecture)

| property | how |
|---|---|
| permutation of the options changes nothing | each option sees the state, the question and itself only, with shared positions |
| renaming the option keys changes nothing (the yes/no swap of 2609.26758) | the keys are never rendered: the model reads the definitions only |
| s + n = 1 when a negation only swaps the two definitions | same reason |
| the state encoding does not depend on the question | the state never attends to the question or the options (cache per state) |

These are tests, not results. The training side is in `crates/sky-train/src/s5/iso.rs`. The served side is in `crates/sky-web/tests/iso.rs`, run natively and on wasm32 under Node in CI.

## Measured

<ClientOnly>
  <S5Results />
</ClientOnly>

## Not done yet

- **s5 in French** (mmBERT-small, Phase 6) has not been trained. The French demos fall back on the English presets.
- **Baselines B1 to B3** were not run: label log-probability through the MLM head, a cascade, and a linear head on the frozen backbone. The comparison with Jev on the original 2609.33209 battery was not run either. These deviations were declared in the pre-registration before any result.
- **The online version of the escalation bet** (with audit) and **Q4 quantization** are left for later.
