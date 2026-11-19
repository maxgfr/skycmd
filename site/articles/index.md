---
title: Articles
outline: false
---

# Articles

Each article tells the story of one line of experiments, for a wide technical audience: what was tried, what failed, what held and what it does not prove. The reference stays [What we bring](/novelty/): every claim of the project in one table, with its measurement and its evidence, and [Results](/results/) for the numbers. An article repeats neither. Newest first.

- **[A 752 KB model that remembers, and still fails its gate: three versions of a fast-weight memory](/articles/a-memory-that-works-and-a-model-that-does-not)**, 17 November 2026. A fast-weight memory with sleep passes for a tiny ternary decision model: a first version whose memory was never used, a second stopped by its own CPU checks before any GPU time, and a third whose memory is used and fades with the delay, while sleep adds nothing measurable and two games lose.
- **[A 632 KB model that knows when you have finished speaking: one failure, one harness bug, then a pass](/articles/a-632-kb-end-of-turn-detector)**, 17 November 2026. An end-of-turn detector for voice agents: a first version that failed three of its four gates, a padding bug that had been hiding the reference model's true score, then a second version designed before training around one job, a teacher to distil and a stopwatch-chosen size, which passes, with its weakness on real conversations stated with the pass.
- **[A 250 KB secrets verifier: rules propose, the model judges, and the gate still says no](/articles/a-250-kb-secrets-verifier)**, 16 November 2026. A tiny model that judges the candidates of a rule-based secret scanner: a first version that passed on its own generated data and failed on real files, then a second, trained and tested on real repositories, that beats every rule scanner tested and still misses its pre-registered bar by 0.027.
- **[A 3.7 MB tool router built on a MacBook: two failures, then a pass and its limits](/articles/a-3-7-mb-tool-router)**, 16 November 2026, with a closing section of 19 November 2026. A small model that picks an assistant’s tool: two failed versions, published as failures, a method written down before every test, and a third version that passes, without hiding what that pass does not prove; then four versions that tried to retrain it without a shortcut, the last of which fails on tools of a new family and closes the line.
