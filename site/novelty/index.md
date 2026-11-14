---
title: What we bring
---

# What skycmd brings

*Updated on 14 November 2026: the summary below and section 8 give what the first trained s6 models show. s6-nano and s6-secrets-pico pass their pre-registered gates and are published; s6-micro and s6-ear-nano fail theirs. Sections 1 to 7 date from 14 October 2026, after three nights of experiments (s3, s4, s5).*

## At a glance

Each line gives a claim, the measurement behind it, where the evidence is, and where to try it. Every model listed as published runs in WASM in this browser, from the same file the numbers were measured on; nothing is sent to a server.

| claim | measured | evidence | try it |
|---|---|---|---|
| Ternary decision models under 1 MB, trained from scratch on one Apple M5 | s3-pico 445,345 parameters in 251,104 bytes; s6-nano 1,486,225 in 715,136 bytes; s6-secrets-pico 445,345 in 250,272 bytes | [`manifest.json`](/models/manifest.json), [`BENCH.md`](https://github.com/maxgfr/skycmd/blob/main/BENCH.md) | [Play](/play/), [Live](/live/) |
| One small file for four games and free-text questions | s6-nano: 0.480 on `typed_decisions_bench`, above the majority class (0.398) where s3-pico (0.382) stayed below; ECE 0.0479; the oracle's score in all four games | [Results](/results/#s6-nano-passed-published-at-full-depth), [chapter 27](/tutorial/27-s6-nano) | [Live](/live/#s6-nano-the-questions-it-was-trained-for) |
| A 250 KB secret detector against a rule scanner scored before training | on file types never trained on: secret 0.9990 against 0.977, service 0.9795 against 0.787. The line answer does not work (0.6859), and the bench is synthetic | [Results](/results/#s6-secrets-pico-passed-published-at-full-depth), [chapter 28](/tutorial/28-s6-secrets-pico) | [Live](/live/#s6-secrets-pico-is-a-secret-exposed-in-this-text) |
| Criteria fixed before training, amendments dated, failures published | s4-micro 0.343 against a gate of 0.55; the three s5 bets missed; s6-micro not published (ECE 0.0821 against 0.08); s6-ear-nano failed three of four gates | [Results](/results/), [section 5](#_5-experiments-you-can-check-failures-included), [`docs/s6-gen-prereg.md`](https://github.com/maxgfr/skycmd/blob/main/docs/s6-gen-prereg.md) | |
| A pre-registered ablation that adopted none of its eight additions | each missed its gain: Muon +0.0051, Parallax +0.0054, XSA +0.0081 against 0.01 needed; SODA, exit distillation, EMA, StableQAT and the curriculum lowered accuracy | [section 8](#_8-what-s6-adds-trained-in-november-2026), [`BENCH.md`](https://github.com/maxgfr/skycmd/blob/main/BENCH.md) | |
| Early exits have to be checked in closed loop | s2-nano at thresholds fitted on i.i.d. questions crashed in every Flappy episode; s6-nano at conformal thresholds (α = 0.02): Rescue 93 %, Dodge vetoes 23 → 1,016, so they were rejected | [section 3](#_3-a-lesson-calibrate-early-exits-where-the-model-is-used), [`loopcal.json`](/results/loopcal.json) | [Play](/play/) |
| The browser gives the native answers | largest probability gap on 200 bench questions: 9.5e-7 (s6-nano) and 1.2e-6 (s6-secrets-pico), same answer on 200 of 200. Under Node: s3-pico 49.6 µs per Flappy frame, s6-nano 436 µs per Flappy frame and 14.6 to 15.2 ms per long bench state, s6-secrets-pico 1,420 to 1,427 µs per decision | [`BENCH.md`](https://github.com/maxgfr/skycmd/blob/main/BENCH.md), [Bench](/bench/) | the live counter of the [home page](/), the button on [Bench](/bench/) |
| Option order and key names cannot change an answer (s5 iso) | exact tests to 1e-6, natively and in WASM; at equal data, 0.674 against 0.429 in the usual format, but the calibration margin failed (ECE 0.088 against 0.057) | [section 4](#_4-decisions-that-cannot-depend-on-the-order-or-names-of-the-options), [Results](/results/#s5-october-2026) | [Live, proof 1](/live/#proof-1-keys-and-order-cannot-matter-guaranteed) |
| A calibration failure traced to its source | s6-micro is calibrated in distribution (ECE 0.037 on validation); one zero-shot NOTAM statement, 412 of 11,000 items, weighs up to about 0.02 of its pooled test ECE. The gate was not changed | [amendment 12](https://github.com/maxgfr/skycmd/blob/main/docs/s6-gen-prereg.md) | |
| A test overlap found and disclosed by the project | 206 of the 2,000 English MASSIVE test utterances are also in training; they are not answered better (s6-nano 0.393 on them, 0.429 on the others), and without them its ECE is 0.0486. English MASSIVE is no longer called zero-shot | [amendment 14](https://github.com/maxgfr/skycmd/blob/main/docs/s6-gen-prereg.md), [Results](/results/#s6-nano-passed-published-at-full-depth) | |

**What it rests on.** None of the building blocks is new, and each is credited with a link on [Sources](/sources/#research-the-architecture-builds-on): a shared block looped with an exit after each pass (TRM, Ouro), one token per measured value (FT-Transformer, FoNE), ternary weights trained in (BitNet b1.58), DAgger, temperature scaling, conformal risk control, and the teachers and backbones (GLM-5.3-flash, Ettin). What is ours is the combination, the scale, the closed-loop checks and the published measurements, failures included.


**In short:** a System One decision model, the kind of model Jev, Laya and Julia-1 introduced in September 2026, fits in **251 KB**. It was trained **from scratch in under an hour** on a laptop and plays four control games **at the level of an exact oracle**, in about **50 µs per frame in a web browser**. Around it, skycmd brings a decision format that is invariant to the order and names of the options by construction, a calibration lesson for early exits, experiments whose criteria were fixed before any test, and a whole chain in Rust that runs in the browser. As far as we could find, nothing this small exists in this format. Training and inference are Rust; a few gate scripts, the s4 teacher and some baselines are Python.

## 1. A decision model 40× to 300× smaller

Every open decision model we found starts from a pretrained backbone of at least 17M parameters. [s3-pico](/tutorial/21-s3-recursive-tiny) was trained from zero, and it still answers typed questions (choice / score / noul) with calibrated probabilities in one forward pass.

| | skycmd s3-pico | smallest public model in the same format | typical decision models |
|---|---|---|---|
| parameters | **445,345** | 17M ([oscar-1-17m](https://huggingface.co/mgoeckel/oscar-1-17m)) | 144M ([Julia-1](https://huggingface.co/SupersonicLabs/Julia-1)) to 27B |
| file | **251 KB** (ternary) | ~37 MB | hundreds of MB to tens of GB |
| training | **from scratch, ~47 min on an Apple M5** | fine-tuned from a pretrained encoder | fine-tuned from large pretrained models |
| per decision | **21 µs native, ~50 µs per frame in WASM** | milliseconds | milliseconds to seconds (Jev: 70–500 ms, vendor figure) |

What s3-pico does with that budget, over 30 seeded episodes per game, with the shipped file (`runs/ship-s3-pico.arena.json`):

- **Flappy Drone:** 200/200 pipes, 0 crashes.
- **Rescue Run:** 100% of hikers rescued.
- **Obstacle Dodge:** 87% of courses completed, exactly the oracle's rate. The coded safety reflex had to step in only 14 times, against 1,026 for our first model.
- **Doom-like:** 100% of levels completed and the same kill rate as the oracle, with no deaths.
- **Depth used:** 1.05 to 1.73 loops out of 4, thanks to early exits.

You can play against it on the [Play](/play/) page.

## 2. An architecture designed for decisions in a loop

Each piece below is known on its own; the combination is ours.

- **Two streams with a cache:** the question and the options are encoded once and cached, and each frame encodes only the state.
- **Options never see each other,** so the answer cannot depend on their order. This holds by construction, not by augmentation.
- **One token per measured value:** a pair such as `vy -3` becomes a single token, combining FT-Transformer and FoNE.
- **Depth from recursion, not from parameters:** one shared block is looped four times, with an exit after every loop, as in TRM and Ouro.
- **Ternary weights trained in** with QAT, BitNet-style.

## 3. A lesson: calibrate early exits where the model is used

- Early-exit thresholds are normally fitted on i.i.d. validation questions. Fitted that way (τ = 0.62 / 0.62 / 0.69 for choice / score / noul), **s2-nano failed** in closed loop: over 5 arena seeds × 30 episodes it crashed in every Flappy episode (15.9 ± 2.3 pipes of 200, 30 crashes of 30) and needed 1,931 ± 1,106 reflex vetoes in Dodge. Confident shallow exits were wrong in the critical states.
- Calibrated **by playing the games** (τ = 0.95), the same weights flew 191.9 ± 7.2 pipes (2.4 ± 2.1 crashes of 30), matched the oracle in Rescue and came within 0.1 of it in Dodge and Doom (Doom is played by s2-nano-doom, its Doom fine-tune, as on the site), at 3.2 to 4.0 of 4 layers.
- s3-pico, a different model, ships with a τ chosen in closed loop (0.6) and uses about 1.4 of its 4 loops (1.44 Flappy, 1.57 Rescue, 1.05 Dodge, 1.73 Doom). There the threshold barely matters: τ = 0.6, τ = 0.5 and full depth give the same scores and crashes on all four games and all five seeds (Dodge vetoes 9.6 against 9.8 at τ = 0.5). For s3-pico, the closed-loop calibration confirmed that shallow exits were safe rather than rescuing anything.
- So the evidence for the lesson is one model, s2-nano, on five seeds. We found little prior work on calibrating early exits in closed-loop control, and we think it is worth checking on any early-exit model that drives something; this data does not show that it is a general rule. Numbers: [`loopcal.json`](/results/loopcal.json), produced by `cargo xtask loopcal`.

A related recipe: **imitation that holds up in closed loop.** With DAgger, the model plays and the oracle labels the states it reaches. It turned "crashes before the first pipe" into "200/200" in two to three rounds of 8 minutes each ([DAgger chapter](/tutorial/20-dagger-lessons)).

## 4. Decisions that cannot depend on the order or names of the options

The s5 iso format ([chapter 23](/tutorial/23-s5-lite)) gives these properties by construction. Each one is checked by an exact test, natively and in WASM, so none of them is an experimental result:

- permuting the options changes nothing;
- renaming the option keys changes nothing, since the model reads the definitions only (this removes the yes/no swap of 2609.26758);
- s + n = 1 when a negation only swaps the two definitions;
- the state encoding does not depend on the question, so it is cached per state.

At equal data, the same encoder was much more accurate in this format than in the usual one on the fitted bench: 0.674 against 0.429 (+0.245, 95% CI [+0.136, +0.321]). It was less well calibrated, though, and the calibration margin fixed in advance failed (see below).

## 5. Experiments you can check, failures included

- **Criteria fixed before any test.** The s5 [pre-registration](https://github.com/maxgfr/skycmd/blob/main/docs/s5-prereg.md) was committed first. Every amendment is dated and says what had been seen when it was written.
- **The test split is locked by the tooling.** The evaluation tool refuses the test split of a frozen set unless the pre-registration lists the set's hash, and it logs every test pass.
- **Failures are published like successes.** s4 missed its gate, and the three pre-registered s5 bets each missed their criterion. The online escalation bet failed in all six cells. The [Results](/results/) page shows each of them with its interval and the reason.

## 6. The whole chain in Rust, down to the browser

- data from Hugging Face;
- training with [mlx-rs](https://github.com/oxiglade/mlx-rs) on Apple Silicon;
- a GGUF writer, with f32, q8_0 and Q4_0 exports;
- an inference engine compiled to WebAssembly, with SIMD kernels for 8-bit and 4-bit weights.

The same file runs in the browser, in a CLI that speaks the `/v1/systemone` API, in Node and in Python (built from source). Our s1 models also run in llama.cpp and wllama. The s5 model takes 21 MB in q8_0 and 12 MB in Q4_0, against 70 MB in f32.

The [Live](/live/) page runs everything on your machine, with no server: the decision models, and a cascade replay in which s5 answers alone or escalates to an auditor that also audits a random share of the accepted answers.

## 7. Methods worth reusing, even from a failed model

[s4-micro](/tutorial/22-s4-generic-decisions) tried to make the architecture generic by distilling Qwen3.5-4B in one night. It missed its gate (0.343 on the typed-decisions bench, against a target of 0.55), but three of its methods are worth reusing:

- **A scoring teacher with per-group bias calibration.** The 4B never generates: one prefill, the logits of the option letters only. A temperature and a bias per option are then fitted per (source, type, number of options) against the gold labels, which removed a strong "true" bias on nouls.
- **Randomized label sets with abstention.** Options are shuffled, distractors are added, an abstention option is appended, and in 12 % of the records the gold option is removed so that "I don't know" is the target.
- **Deep improvement supervision on early exits.** Each exit trains toward the target raised to a power that grows from 0.5 to 1, so the early exits learn a softer copy of the target.

Its failure is a lesson too: without language pretraining, loops cannot make up for it. A BERT-Tiny of 4.4M parameters fine-tuned on the same data reaches 0.794 on Banking77, against 0.303 for s4.

## 8. What s6 adds (trained in November 2026)

Each s6 model has a pre-registration with its gate. Four are trained: their numbers are on the [Results](/results/#s6-november-2026) page and in [`BENCH.md`](https://github.com/maxgfr/skycmd/blob/main/BENCH.md), and the points below come from there. s6-cast is still training.

**What the trained models show.**
- **One 715 KB file for the games and for questions in free text.** s6-nano (1,486,225 ternary parameters) plays all four games at the oracle's score and reaches 0.480 on `typed_decisions_bench`, above the majority class (0.398). The earlier small models stayed below it: s3-pico 0.382 and s4-micro 0.352 on the same records. Its ECE over all the test sources is 0.0479. It runs in this browser on [Live](/live/#s6-nano-the-questions-it-was-trained-for) and [Play](/play/), and its WASM answers match the native engine to 9.5e-7 on 200 bench questions.
- **No architectural change was adopted.** The pre-registered pico ablation adopted none of the additions listed below. Each needed a validation gain of 0.01 (0.005 for the training options) with no game lost: Muon brought 0.0051 and lost Flappy score, Parallax 0.0054, XSA 0.0081, and SODA, exit distillation, the EMA, StableQAT and the curriculum all lowered accuracy (`runs/recipe/recipe.json`). This is the outcome the literature warned about: gains over AdamW shrink once AdamW is tuned as carefully ([Fantastic Pretraining Optimizers](https://arxiv.org/abs/2509.02046)), and at equal time AdamW can beat Muon on small language models ([When Does Muon Actually Win?](https://arxiv.org/abs/2610.21555)). What changed from s4-micro is the pretraining: 0.87B tokens, 588 per parameter, against 30 minutes on 150 MB for s4-micro. An MLM encoder wants far more tokens per parameter than a decoder (Training compute-optimal transformer encoder models, EMNLP 2025). The GLM-5.3-flash teacher, the tokenizer and the 512-token windows changed too, and no ablation separates these causes.
- **A guarantee checked by playing, and rejected.** Conformal risk control (α = 0.02; [Fast yet Safe](https://arxiv.org/abs/2405.20915), CALM at NeurIPS 2022) cut s6-nano's depth to 1.01 to 1.83 of its 4 passes per game. In closed loop, Rescue fell to 93 % and the Dodge vetoes rose from 23 to 1,016. The pre-registered rule allows 2 %, so the model ships at full depth. The guarantee holds on held-out questions, not on the states that playing with early exits leads to: the lesson of section 3, met again with a better tool.
- **A 250 KB secret detector that beats a rule scanner on file types it never saw.** s6-secrets-pico (445,345 ternary parameters, the s3-pico graph with a 320-token state) reads a short text and answers four typed questions with the same engine as the games. On its synthetic bench of docker-compose, toml, go and ini files, none of them in its training, it scores 0.9990 on "is a usable secret exposed?" and 0.9795 on "which service issued it?", where a gitleaks-style rule scanner scores 0.977 and 0.787 on the same texts. It is right on all 46 texts the scanner gets wrong, the random values under secret-sounding names among them. Two limits come with it: the bench is synthetic, so this shows that the model reads formats and context, not that it finds real leaks; and its line answer is weak (0.6859, right on 383 of the 1,010 texts that expose a secret). It runs on [Live](/live/#s6-secrets-pico-is-a-secret-exposed-in-this-text), about 1.4 ms per decision in WASM under Node.
- **Failures published as results.** s6-micro reaches 0.547 on the bench, which meets its gate and its announced goal of 0.50, but it misses the ECE guard (0.0821, then 0.0804 at its conformal thresholds), so it is not published. s6-ear-nano fails three of its four gates, and its control without the Mimi pretraining does better on all four sets. Every amendment is dated and says what had been seen when it was written.

What follows is how the code differs, and why. The trained models use the data, the teacher and the conformal tool; the encoder and optimizer additions stay in the code, unadopted.

**A better encoder for the same weights.**
- **Parallax attention** (arXiv 2605.29157). Each head learns a probe that turns softmax attention, a local constant estimator, into a local linear one. It is one extra row block in the same fused projection, so the ternary storage and the first-block lookup table need no new tensor. With a zero probe it is softmax attention exactly, so the options stay isolated.
- **Muon v2.** Decoupled weight decay and an update scaled to AdamW's RMS (Kimi), then a two-stage Newton–Schulz iteration whose second stage pulls the singular values to within 0.5% of 1.
- **A pre-registered ablation** decides what ships, one change at a time: Parallax, exclusive self-attention, SODA weight decay, exit self-distillation, a weight EMA, the StableQAT gradient and a pretraining curriculum.

**Early exits with a guarantee.** The exit thresholds are chosen by conformal risk control: an early answer disagrees with full depth at most α of the time. The threshold is then checked by playing at it, because playing with early exits changes which states the model visits. The guarantee does not survive that change unchecked; this is the lesson of section 3, turned into a tool.

**A teacher that never heats the laptop.** GLM-5.3-flash, open weights under MIT, labels all 747,198 training decisions through the z.ai API, with four workers.
- **Resumable:** labelling restarts where it stopped, across the 5-hour usage windows.
- **Moderation-proof:** a batch refused by the moderation is halved until the refused text stands alone, so one sensitive text costs its neighbours nothing.
- **Filtered:** only the labels that agree with the gold answer are mixed in.

**Small specialists on the same engine.**
- **s6-cast:** a probabilistic forecaster, 388K parameters, pretrained on synthetic series only.
- **s6-secrets:** API-key detection. Its synthetic keys follow each provider's public format, and its bench uses file types the training never shows. The bar is a gitleaks-style rule scanner, and s6-secrets-pico clears it (above).

**System 1 first, System 2 only when needed.** The same idea as the early exits, one level up. For LFM3-1.7B, a 6-coefficient confidence probe decides, question by question, whether the model should think before answering. Thinking can make a small model abandon a good first answer.

**s6-ear: a decision model that listens** ([chapter 26](/tutorial/26-s6-ear)).
- **Mimi without Mimi.** Kyutai's semantic speech tokens, without the 48M-parameter codec. A 1.6M model reads log-mel frames at Mimi's own frame rate (80 ms) and learns, by masked prediction, the Mimi codebook-0 codes Kyutai published for LibriSpeech. Mimi never runs.
- **The question is text, the state is sound.** The options are words, so it can be asked about words it never heard in training. Its zero-shot test holds out 50 MSWC words per language, in English and French.
- **The teacher reads, the student listens.** GLM labels the transcripts, and s6-ear learns the answers from the audio: cross-modal distillation, with no non-commercial spoken-intent set.
- **A voice agent's System 1.** Every 80 ms: keep listening, acknowledge, speak or yield. Below a conformal threshold it hands the clip to a larger model.
- **Measured in the smoke run:** 5 ms per decision for 10 s of audio natively, 0.67 ms in WASM for 1 s clips. The browser front-end matches the native one bit for bit, because its log uses no libm.
- **Honest bars.** The end-of-turn gate is Smart Turn v3.2 (8M parameters, five times larger) on Pipecat's official held-out test set. We first planned a split of Smart Turn's own training data, and dropped it before any run.

## What it does not bring (honestly)

- **None of the building blocks is new.** The novelty is in how they are combined, in the scale and in the closed-loop calibration lesson. Our literature search was two research passes, not a systematic review.
- **The games are friendly.** Code measures the useful features (relative clearances, an A* waypoint) and the model decides, following the [jev-drone](https://github.com/RomanSlack/jev-drone) philosophy. A small MLP on the same features would be a fair baseline, and we have not run it yet.
- **The tiny model is a specialist.** On general decisions it is modest: 0.54 on our held-out mix and 0.38 on the external typed-decisions benchmark, against about 0.73 for the 144M Julia-1.
- **No pre-registered claim for s5.** None of the three bets is claimed: the iso format missed its calibration margin, the coherence losses improved only the identities they were trained on, and 8-bit quantization could not be shown non-inferior for lack of power.
- **s5 is a small English model.** There is no French version.
- **The results are not peer-reviewed.** They come from our own benchmarks, with single runs for s3 and s4, three seeds for s5, and one run per size for s6.
- **s6-secrets-pico is measured on our own generator only.** Its bench is synthetic (our formats, file templates and class shares, fake keys with random bodies), so its scores say nothing about real leaks in real repositories. It cannot be trusted to locate a leak: line 0.6859, and confident when wrong (ECE 0.3051).
- **s6-nano is a modest generalist.** 0.480 on `typed_decisions_bench` clears the majority class, but s5-lite reaches 0.696 and oscar-1-17m 0.6775. It fails on NOTAM (0.055 on the subject question) and stays at 0.404 on When2Call. Its English MASSIVE score is not strictly zero-shot: 206 of the 2,000 test utterances also appear in the training split, 201 of them in a source derived from MASSIVE.

## What would turn this into a paper

- Done: closed-loop vs i.i.d. early-exit calibration over 5 arena seeds (point 3, `cargo xtask loopcal`).
- Ablations of s3 over several seeds: fused values, recursion, static cache, DAgger.
- A standard control benchmark with vector states.
- Baselines: a same-size MLP on the games, an RL policy, oscar-1-17m.
- The s5 baselines B1 (label log-probability through the MLM head) and B3 (a linear head on the frozen backbone); see [Not done yet](/results/#not-done-yet).
- A systematic literature review.

The natural first target is a short workshop paper: *closed-loop calibration of early exits for tiny decision models*, with skycmd as the open-source demonstrator.

See also the [tutorial](/tutorial/), the [night experiments](https://github.com/maxgfr/skycmd/blob/main/docs/night-experiments.md) and the [sources](/sources/).
