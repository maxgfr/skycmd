---
title: Live
outline: [2, 2]
---

# Live: a generic decision model in your browser

The game models on [Play](/play/) only know their games. **s4** is the first skycmd model trained for decisions in general: you write the question and the options in plain text, and one forward pass returns calibrated probabilities over them. The labels can be ones it has never seen. It is the s3 architecture (a bidirectional encoder with a looped, weight-shared state stack and ternary weights) scaled to language, and distilled from Qwen3.5. The whole model is a GGUF file of a few megabytes that runs in WASM on this page. Nothing is sent to a server.

What each demo shows:

- **Probabilities** are calibrated per question type and option count (temperature scaling on held-out data), so 70% should mean right about 70% of the time.
- **Adaptive depth**: the state encoder has three exits (after 2, 4 and 6 block passes). The model stops at the first exit whose confidence clears a calibrated threshold, so it "thinks" longer on hard inputs. Untick the box to force the full depth.
- **Abstention**: the triage and playground demos add an option *"unsure: I don't know, none of these fits"*. The model was trained with label sets where the right answer had been removed and this option was the target.
- **Static cache**: the question and options are encoded once and cached. When only the state changes (a new message, a new request), only the state is encoded.

These are small-model answers. Check the numbers in [the s4 chapter](/tutorial/22-s4-generic-decisions) before relying on them.

## Playground

Any question, any options, any type (`choice`, `score` from lowest to highest level, or `noul` for a true/false statement).

<ClientOnly>
  <LiveDecide mode="playground" />
</ClientOnly>

## Tool router for an agent

Describe the tools in one line each. The model picks the tool for the request, or `no_tool` when none is needed. An agent can ask this before calling a large model: the tool list stays cached, so each new request costs one state encoding.

<ClientOnly>
  <LiveDecide mode="tools" />
</ClientOnly>

## Inbox and ticket triage

Your own categories, with an "I don't know" way out. When no category clearly wins, the demo shows the abstention instead of a guess.

<ClientOnly>
  <LiveDecide mode="triage" />
</ClientOnly>

## Guardrail on a written rule

Write a rule in plain language. The model returns p(the message violates the rule), a `noul` question built as *The message violates this rule: "…"*.

<ClientOnly>
  <LiveDecide mode="guard" />
</ClientOnly>
