---
title: Live
outline: [2, 2]
---

# Live: a decision model in your browser

The game models on [Play](/play/) only know their games. **s5-lite** is a decision model for questions you write yourself:

- you give the question, the options (each with a one-line definition) and the state;
- one pass returns calibrated probabilities over the options.

It is Ettin-17M, fine-tuned in Rust, quantized to 8 bits. It runs in WASM on this page and sends nothing to a server.

**s6-nano** (1,486,225 ternary parameters, a 715 KB file) is the generalist of the s6 family: it plays the four games on [Play](/play/) and answers typed questions in free text. It can be chosen in every demo below, and [its own section](#s6-nano-the-questions-it-was-trained-for) asks it the game and numeric questions it was trained for. On the questions of the demos above it is a weaker model than s5-lite: 0.480 on `typed_decisions_bench` against 0.696 ([results](/results/)).

What makes s5 different is its **iso** architecture:

- **Options read by definition only.** The model reads each option by its definition. The key you give it (`billing`, `yes`, `A`) never reaches the model, so renaming the keys or swapping yes and no cannot change the answer.
- **Options isolated from each other.** Each option sees the state, the question and itself, never the other options, all at the same positions. Reordering the options cannot change the answer either.
- **State encoded once.** The state is encoded once, then reused by every question asked about it.

These are guarantees by construction, checked by exact tests. Whether the answers are good is a different matter, measured on held-out data: the numbers, including the failures, are on [the results page](/results/).

**Calibration:** probabilities are calibrated per question type and number of options, by temperature scaling on a held-out calibration split.

**Presets:** they are the demo cases every s5 model is evaluated on (`eval/s5/demo-cases.jsonl`). A ✓ marks the expected answer while a preset is unmodified.

::: info French
s5 is a small English model; there is no French version. Choosing *Français* in a demo shows a notice and keeps the English presets.
:::

## Playground

Any question, any options, any type: `choice`, `score` (levels from lowest to highest) or `noul` (a true/false statement).

<ClientOnly>
  <LiveDecide mode="playground" />
</ClientOnly>

## Tool router for an agent

Describe the tools in one line each. The model picks the tool for the request, or `no_tool` when none fits.

<ClientOnly>
  <LiveDecide mode="tools" />
</ClientOnly>

## Inbox and ticket triage

Your own categories, with an `unsure` way out. When no category clearly wins, the demo shows the abstention instead of a guess.

<ClientOnly>
  <LiveDecide mode="triage" />
</ClientOnly>

## Guardrail on a written rule

The statement to check is a `noul` question: *The message violates this rule: "…"*. The model returns p(true).

<ClientOnly>
  <LiveDecide mode="guard" />
</ClientOnly>

## s6-nano: the questions it was trained for

The four games and the numeric curriculum, as typed questions you can edit: the state is the text the game gives the model (`vy 11 dx 35 …`), and every option, `noul` answers included, is the record's own text. The presets follow a fixed rule: the first record of the test split of each source and question type in the game and numeric sets. They are kept whatever the model answers: natively, the published file picks the expected option on 11 of the 15, and the 4 misses are numeric questions. Some instructions are in French, as in the training data.

The file is published at **full depth**: every answer uses all 4 block passes, because the early-exit thresholds were rejected by the pre-registered arena check ([results](/results/)). The time shown is measured in your browser for each decision. On the build machine, under Node, these short game and number states take about 1.6 ms per decision; the long JSON states of `typed_decisions_bench` take about 15 ms.

<ClientOnly>
  <LiveDecide mode="s6" prefer="s6-nano" />
</ClientOnly>

## Proof 1: keys and order cannot matter (guaranteed)

Every key moves to another definition, then the order is reversed. s5-lite gives each definition the same probability every time, up to float rounding. s4-micro reads the keys and the other options, so it moves (the yes/no label swap of [2609.26758](https://arxiv.org/abs/2609.26758) is the same effect).

<ClientOnly>
  <LiveProof mode="guarantee" />
</ClientOnly>

## Proof 2: other words for the same options (learned)

Here nothing is guaranteed. Robustness to paraphrased definitions is trained, and it is measured on paraphrases written by hand and never seen in training. Try your own wordings.

<ClientOnly>
  <LiveProof mode="paraphrase" />
</ClientOnly>

## Proof 3: a Dutch book on the model's own beliefs

The model is asked three kinds of question, each separately:

- the choice itself;
- for each option, "The correct answer is: x";
- "The correct answer is not: x".

A coherent model gives s + n = 1 and Σ s = 1. When Σ s drifts from 1, a bookie can bet against the model's prices and win for sure. Bet 2 trains this coherence and measures it, together with identities it never trained on (results page).

<ClientOnly>
  <LiveProof mode="dutch" />
</ClientOnly>

## Cascade: s5 alone, or the auditor

When s5 is not sure enough, it escalates the question to a System 2. Here the System 2 is an auditor: the dataset gold, standing in for a human. The auditor also checks a random share π of the answers s5 gave alone, and the threshold of each question type moves with the error estimated on those audits. This is bet 3, run after the offline results and therefore exploratory; it fails, and the demo shows why. It replays a recorded run: no model call, no network call, no GLM.

<ClientOnly>
  <LiveCascade />
</ClientOnly>
