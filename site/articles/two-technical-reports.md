---
title: "skycmd: what passed, what failed, and how to build tiny decision models on one laptop"
description: "Twenty small models trained from scratch on one laptop, fifteen judged by a bar written before training: five passed, ten failed and were published anyway. The project, its two technical reports, and the two files you can use today."
date: 2026-11-21
outline: [2, 2]
---

# skycmd: what passed, what failed, and how to build tiny decision models on one laptop

*21 November 2026*

skycmd is a research project I ran from my own laptop: train the smallest useful *decision models* from scratch, run them in a browser, and judge each one against a bar written down before it was trained. The experiments ended on 20 November 2026. Two technical reports now tell the whole story, one about what passed and what failed, the other about how the models are built. This article is a guided tour of both, with the four lessons I would keep and the two files that are ready to use.

## What skycmd is

A decision model does not write text. It reads a *state* (a game frame, a short text, a request with a list of tools, the last seconds of audio) and one or more *typed questions*, and in a single forward pass it returns a probability for every allowed option. There are three question types: `choice` picks one of several named options, `score` picks one of several ordered levels, and `noul` is a probabilistic yes or no. The probabilities are meant to be *calibrated*, which means they can be taken at face value: of the answers a calibrated model gives at 0.9, about nine in ten are right.

Every model was trained from scratch, in Rust, on one Apple M5 laptop with 16 GB of memory, the machine I work on. The files are small: from 142,912 bytes to 3,692,288 bytes for the latest family, most of them under one megabyte. The first answer to "how small can it be?" was s3-pico, 445,345 *ternary* parameters (each weight is −1, 0 or +1) in a 251,104-byte file, which plays four control games at the level of an exact oracle. Every demo runs in WebAssembly, in the browser, from the very file its numbers were measured on: what you try is what was tested.

## How each model was judged

Before a model was trained, a file in the repository, its *pre-registration*, fixed the data, the training budget, the test it would be read on and its *gate*: a set of conditions that must all hold. Three rules kept the gate honest.

- **The test is read once.** Starting with s5, the fifth generation of models, the evaluation tools refuse to open a test set until a listed hash or a marker file says they may.
- **Every change is a dated amendment**, committed before the computation it concerns.
- **A failure is published as a result**, with its file, a model card that opens with the failure, and a demo.

Of the 20 published models (143 KB to 20.7 MB), 15 were judged by a gate fixed before training. **Five passed and ten failed.** Two of the five passes are *second looks*: systems designed after a test had been read, which the reports name as such every time they appear. Nine more versions stopped without producing a model, seven of them at a check their pre-registration had named in advance.

<figure class="article-figure">
<!--@include: ../public/paper/figure-1.svg-->
<figcaption>Figure 1. Every gate, as read: one condition of each gated model against its bar (the vertical tick). Each row has its own scale and right is better on every row. A filled circle is a pass, a diamond a failure, a ring a second look. From Figure 1 of the <a href="/skycmd/paper/#_4-2-the-gates">technical report</a>.</figcaption>
</figure>

## The two technical reports

Both are dated 21 November 2026, archived on Zenodo with a DOI, and written so that every number links to its evidence in the repository. Neither has been peer-reviewed.

### Pre-registered gates for small models trained on one laptop: what passed, what failed, and what the readings show

The first report holds the protocol and every gate as it was read. It covers the 20 models, the 15 gates with their measurements and caveats side by side, the experiments that stopped before producing a model, and four lessons that cut across lines of work. It ends with a twelve-rule checklist for pre-registering a small-model experiment, each rule paired with the reading that taught it.

[Read it on the web](/paper/) · [PDF](https://maxgfr.github.io/skycmd/paper/skycmd-technical-report.pdf) · [Zenodo](https://zenodo.org/records/24421273)

### Looped ternary encoders for typed decisions: a recipe from laptop to browser

The second report is the companion on method: how to build a 251,104-byte decision model, train it on one laptop and run the same file in a browser. The model encodes the question and its options once and caches them, gives each measured value a single token (9 tokens instead of 16 for a Flappy frame), loops shared blocks with a possible exit after each pass, and stores its ternary weights at 2 bits. The same blocks give a 631,936-byte end-of-turn detector and a tool router, and a table states what each design choice bought and which choices were never measured.

[Read it on the web](/paper/methods/) · [PDF](https://maxgfr.github.io/skycmd/paper/skycmd-methods.pdf) · [Zenodo](https://zenodo.org/records/24420817)

## Four lessons

None of these was a pre-registered test. They are patterns in readings already published, drawn from few lines of work and mostly from one training seed, so take them as advice rather than laws.

**1. Check a gate against trivial baselines before freezing it.** The bar of s6-nano, the model that answers typed questions in free text, was the majority class: 0.398. It passed with 0.4805. But a "model" that answers each question with its most common training answer scores 0.4815: the bar was too low to mean much. On the tool-routing line, by contrast, a bar set at BM25 (a classic keyword-search score that needs no training) was added before training, and it caught a router that only tied it.

**2. Check early exits by running the model in its loop.** A looped model can stop after any pass once it is confident enough, which saves time. Thresholds fitted on held-out questions made one model crash in every Flappy episode on five seeds. Thresholds from conformal prediction (a method that bounds the error rate of each decision, here at α = 0.02) raised another model's safety vetoes in a game from 23 to 1,016. When the model's answer changes its next input, choose the exits by running the loop.

**3. Hide or rename the input.** The router that passed, s6-route, picks the right option on 0.8868 of fresh requests about tools it never saw in training. With every tool renamed `tool_<k>`, it falls to 0.2497, while BM25 keeps 0.8138. A diagnosis traced the fall to the data of its trained part: in every training list whose tools had been renamed, the right tool was the only one whose description the teacher model had rewritten, and the router learned to pick that description.

**4. Read each pass on data built differently.** Three of the five passes carry a limit measured that way. s6-route drops from 0.8868 to 0.5530 on a new family of tools, where BM25 reads 0.5675. The end-of-turn detector s6-ear2-nano falls from an AUC of 0.9600 (the chance that a finished turn scores above an unfinished one: 1 is perfect, 0.5 is chance) on its teacher's test clips to 0.7299 on real conversations. The secret detector s6-secrets-pico, at 0.9990 on its synthetic bench, flags 59.8% of 719 real files that hold no secret, at a probability of 0.9 or more.

## What each choice bought

The methods report ends with a table of what each design choice bought where it was measured. A selection:

| choice | what it bought | evidence |
|:-------|:---------------|:---------|
| Train the ternary weights in, rather than round a float model | Rounding cost 6 to 11 points of validation accuracy (0.470 to 0.406, and 0.542 to 0.432); spending 30% of the budget training them in gave 0.482 and 0.556 | two model sizes, one run each |
| Treat 2 bits as a storage format | Float kernels took 3.99 µs per matrix product, packed ternary kernels 12.23 µs | one matrix shape, one CPU |
| Choose exit thresholds by playing | s3-pico kept 200 Flappy pipes of 200 at 1.44 of 4 passes; with thresholds fitted on questions, s2-nano crashed in every episode | two models, five seeds |
| Fix the number of DAgger rounds in advance | Round 1 fell to 5.4 pipes with 30 crashes; rounds 2 and 3 reached 200 of 200 with none | one model, one run |
| Distil from a teacher rather than train on hard labels | Validation accuracy 0.8715 against 0.8437 | one pilot, validation only |
| Test the browser against the native engine with the same exits | Probabilities within 9.5e-7 on 200 of 200 questions | one model, 200 questions |

DAgger, in the fourth row, is a training loop in which the model plays, an oracle labels the states the model actually reached, and the model is trained again on them.

The same table lists what was never measured: a looped block against a stack of distinct blocks of the same size, loop dropout against none, the numeric curriculum against none, and one token per value against one token per digit for accuracy. The limits apply to everything above. Every recipe was trained once, on one seed, so nothing says how much a result moves when trained again. Everything ran on one machine, often shared with other work. And both reports are technical reports, not peer-reviewed papers: their claim rests on the protocol and the public evidence, not on review.

## What you can use today

Most files are results or demos. Two can go into a real project, each with a safety net, and the [Use it for real](/use/) page has the code.

- **s6-ear2-nano** tells a voice agent when the speaker has finished. It is a 631,936-byte file, right on 0.8993 of the 31,527 test clips of Smart Turn v3.2, the open model it learned from (which scores 0.9260), at 2.39 ms per decision in WebAssembly. On real conversations, at its cautious threshold, it recognises only 0.5253 of the real ends of turn, so **a fallback timer is required** (2,500 ms of silence in the demo).
- **s6-route** picks a tool from a list written at run time, or answers "none of these tools" or "ask the user". On 945 fresh requests about tools never seen in training it scores 0.8868, against 0.8169 for BM25, with better-calibrated probabilities (calibration error 0.0162, against 0.0779). Use it **when tool names are distinctive** and the tools are of familiar kinds; when names are generic or numbered, or the tools are a new family, take BM25.
- **s6-secrets-verify-pico** runs inside [secretgate](https://github.com/maxgfr/secretgate), a tool that redacts secrets in what a coding agent reads and writes, as an **opt-in experiment**. It **failed** its gate: it had to beat the best rule scanner by 0.15 of line F1 (a score that balances secrets found against false alarms) and reached +0.123. secretgate offers it off by default, and it can only add redactions to what the rules find.

## Links

- On this site: [What we bring](/novelty/) (every claim with its evidence), [Results](/results/) (every gate and its numbers) and [Models](/models/) (every file and its status)
- The code, pre-registrations and result files: [github.com/maxgfr/skycmd](https://github.com/maxgfr/skycmd)
- Every model, its Space and the data sets: [the skycmd collection on Hugging Face](https://huggingface.co/collections/maxgfr/skycmd-6b001dd30a7cb5724995b603)
- Every demo in one Space: [huggingface.co/spaces/maxgfr/skycmd](https://huggingface.co/spaces/maxgfr/skycmd)
