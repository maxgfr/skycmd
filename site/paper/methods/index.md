---
title: "Methods paper: small ternary decision models trained on one laptop"
description: "The methods of skycmd: the ternary decision graph with its two streams, cache and looped exits, the training recipe on one laptop, exit thresholds chosen by playing, the task-specific systems and the Rust engine that runs every file in the browser. Not peer-reviewed."
outline: [2, 3]
---

<div class="paper">

# Small ternary decision models trained on one laptop: architecture, training, closed-loop calibration and a Rust engine for the browser

<p class="paper-meta"><strong>Maxime Golfier</strong> · 21 November 2026 · <a href="/skycmd/paper/skycmd-methods.pdf">PDF (LaTeX)</a> · <a href="https://github.com/maxgfr/skycmd/tree/main/paper/methods">LaTeX source</a> · companion: <a href="/skycmd/paper/">the technical report</a>, <a href="https://doi.org/10.5281/zenodo.24418316">DOI 10.5281/zenodo.24418316</a> · <a href="#how-to-cite">How to cite</a></p>

::: warning Technical report, not peer-reviewed
This report has not been peer-reviewed, nor reviewed by anyone outside the project. Every number in it is copied from a page or a file of the repository, and each one links to its evidence. Where a number is computed here, the text says so. It describes the methods; the protocol and every gate's reading are in the companion report [[1]](#ref-1).
:::

## Abstract

This report describes how the models of the skycmd project are built, trained, calibrated and run, precisely enough to reimplement them; a companion report [[1]](#ref-1) holds the protocol and every gate as read. The models answer typed questions (`choice`, `score` and `noul`) about a state, with a probability for every option in one forward pass. The graph encodes the question and its options once and caches them, encodes each new state with one token per measured value, loops a small stack of shared blocks with an exit after each pass, and scores every option with a fusion head in which no option sees another. Every linear layer is ternary, trained with a straight-through estimator after a float warm-up and stored at 2 bits per weight: s3-pico holds 445,345 parameters in 251,104 bytes. Training mixes game oracles written in Rust, public data sets and soft labels from an open-weight teacher, puts a loss on every exit, and runs DAgger on the games, under budgets counted in working minutes on one Apple M5 laptop. Exit thresholds are chosen by playing the games, because thresholds fitted on held-out questions made s2-nano crash in every Flappy episode. Variants of the same blocks detect the end of a turn in speech, route requests to tools, judge secret candidates proposed by rules, and forecast time series. One Rust engine runs every file natively and in WebAssembly, where s6-nano's probabilities differ from the native ones by at most 9.5e-7 on 200 bench questions. Most pieces are adopted from prior work, and the report says which; what it adds is their combination in files of a few hundred kilobytes, the closed-loop check of the exits, and the measurements.

## 1. Introduction

A *decision model* receives a state (a game frame, a short text, the last seconds of audio) and one or more typed questions, and returns a probability for every allowed option in one forward pass, without generating text. A `choice` question picks one of several named options, a `score` question one of several ordered levels, and a `noul` question is a probabilistic yes or no ([`docs/SPEC.md`](https://github.com/maxgfr/skycmd/blob/main/docs/SPEC.md)). TypeSafe introduced this format with its closed model Jev [[2]](#ref-2), and open models now serve it from 706,048 parameters for one form-filling task [[3]](#ref-3) to 421M [[4]](#ref-4).

skycmd trained such models from scratch under three constraints: small files, from 142,912 bytes to 3,692,288 bytes for the s6 family and most of them under one megabyte ([manifest](/models/manifest.json)); one Apple M5 laptop with 16 GB of memory, the machine its owner works on, for every run ([`docs/big-run-report.md`](https://github.com/maxgfr/skycmd/blob/main/docs/big-run-report.md)); and the browser, where every demo runs the file its numbers were measured on.

The companion report [[1]](#ref-1) is the record of the protocol and of every gate's reading. This report is its counterpart on method: it describes what the code does, gives at most one sentence per system on its result, with a pointer, and follows the code where the project's specifications disagree with it.

It contributes five descriptions, each with the parts it adopts named as such.

1. **The decision graph** (section 3): two streams and a cache, options isolated by the attention mask, one token per measured value, shared blocks looped with an exit after each pass, a fusion head shared by the exits, and ternary weights trained in. Each piece is adopted: order invariance from Set-LLM [[5]](#ref-5) and Amit [[6]](#ref-6), a looped shared block with an answer after each loop from TRM [[7]](#ref-7) and Ouro [[8]](#ref-8), one token per value from FT-Transformer [[9]](#ref-9) and FoNE [[10]](#ref-10), ternary weights from BitNet b1.58 [[11]](#ref-11). Built from them, s3-pico is the smallest open model we found that covers the whole typed format, and the only ternary one ([What we bring, section 1](/novelty/#_1-a-decision-model-in-251-kb)).
2. **A training recipe for one laptop** (section 4): masked-language pretraining with a loss on numbers, a loss at every exit, soft labels from an open-weight teacher kept only where they agree with the gold answer, DAgger [[12]](#ref-12) on the games, and budgets counted in working minutes. A recipe table gives the settings of seven published files.
3. **Exit thresholds chosen by playing** (section 5): temperatures per exit, type and bucket, and a threshold accepted only if the games played at that threshold stay within 2% of full depth. Choosing exit thresholds against the task's own success is what DeeR-VLA does [[13]](#ref-13); the readings here confirm that line of work and do not introduce it. A conformal variant is described with the reason it was rejected.
4. **Task-specific systems on the same blocks** (section 6): an end-of-turn student of a larger audio model, a tool router fused with a static embedding, a verifier of secret candidates, and a forecaster. The verifier's design, rules that propose and a learned model that judges, is that of Samsung's CredSweeper [[14]](#ref-14) and GitGuardian's FP Remover [[15]](#ref-15).
5. **One inference engine in Rust** (section 7): the GGUF files, the ternary kernels, the caches and the WebAssembly builds, with the measured agreement between the browser and the native engine.

What the combination adds is a chain from data to a browser in which every step fits one laptop and every claim is measured; no single piece is new ([What we bring](/novelty/#at-a-glance)).

## 2. Related work

The references are those of the companion report, which checked each one against its arXiv or project page; the project's [Sources](/sources/) page lists them all.

**Decision models.** Jev [[2]](#ref-2) launched the format; Laya [[4]](#ref-4) and oscar-1-17m [[16]](#ref-16) are open models in it on pretrained encoders, and two smaller ones appeared in 2026: cua-s1-forms [[3]](#ref-3), a byte-level encoder of width 128 for one form-filling task, and TinyDecide [[17]](#ref-17), fine-tuned from ELECTRA-small. The models of this report are trained from scratch.

**Options, depth and exits.** Set-LLM [[5]](#ref-5) and Amit [[6]](#ref-6) isolate the options from one another, and Jeva [[18]](#ref-18), concurrent work, also reuses their representations across requests. Weight sharing across depth goes back to Universal Transformers [[19]](#ref-19); a loss at every exit follows deeply-supervised nets [[20]](#ref-20), and skipping loops at random in training follows stochastic depth [[21]](#ref-21). Early exits go back to BranchyNet [[22]](#ref-22), DeeBERT [[23]](#ref-23) and FastBERT [[24]](#ref-24). Conformal risk control [[25]](#ref-25) gives exit thresholds a guarantee, in the setting of CALM [[26]](#ref-26) and Fast yet Safe [[27]](#ref-27), and DeeR-VLA [[13]](#ref-13) sets the exit thresholds of a robot policy against task success.

**Weights and training.** Ternary weights at small scale are studied by Nielsen et al. [[28]](#ref-28), and the encoder block keeps the layout of ModernBERT [[29]](#ref-29). Temperature scaling and the expected calibration error with 15 equal-width bins are those of Guo et al. [[30]](#ref-30); learning a teacher's soft labels is knowledge distillation [[31]](#ref-31). The code includes the Muon optimizer [[32]](#ref-32); that its gains over AdamW shrink once AdamW is tuned as carefully [[33]](#ref-33), and that AdamW can win at equal time on small models [[34]](#ref-34), is what the project's ablation found too (section 4.4).

**Task-specific work.** The closest small end-of-turn work found is a 1.14M-parameter model distilled into an MFCC-based student [[35]](#ref-35). Routing by embedding similarity is the method of semantic-router [[36]](#ref-36), and the router's lexical score is BM25 [[37]](#ref-37). The verifier is trained on Samsung's CredData [[38]](#ref-38). Fracast-0 [[39]](#ref-39) is a smaller forecaster that, by its abstract, does better than the one here. Training in Rust and running in WebAssembly is also done by frameworks such as Burn [[40]](#ref-40); the teachers and baselines of each system are cited in section 6.

## 3. The decision model

Figure 1 shows the graph, `skycmd-s2` in the files, used by s3-pico, s6-nano, s6-micro and the s6 specialists. The training graph is [`crates/sky-train/src/s2/model.rs`](https://github.com/maxgfr/skycmd/blob/main/crates/sky-train/src/s2/model.rs) and the inference graph [`crates/sky-infer/src/s2/model.rs`](https://github.com/maxgfr/skycmd/blob/main/crates/sky-infer/src/s2/model.rs); the specifications are [`docs/SPEC-S2.md`](https://github.com/maxgfr/skycmd/blob/main/docs/SPEC-S2.md), [`docs/SPEC-S3.md`](https://github.com/maxgfr/skycmd/blob/main/docs/SPEC-S3.md) and [`docs/SPEC-S6.md`](https://github.com/maxgfr/skycmd/blob/main/docs/SPEC-S6.md).

<figure class="article-figure">
<!--@include: ../../public/paper/methods/figure-1.svg-->
<figcaption>Figure 1. The decision graph. The question and its options are encoded once and cached (shaded); each frame, the state goes through B shared blocks up to L times, and after each pass an exit feeds the fusion head, which is shared by all exits. The answer is returned at the first exit whose top probability reaches the threshold of the question's type, or at the last. The figure is written by <a href="https://github.com/maxgfr/skycmd/blob/main/scripts/paper-methods-figure.py"><code>scripts/paper-methods-figure.py</code></a>; its text alternative is in the SVG.</figcaption>
</figure>

### 3.1 Requests and tokenization

A request holds a state and one or more questions; a question has a type, an instruction and its options. The options are rendered as Laya's runtime and llama.cpp's server render them: a `choice` option is `key` or `key: description`, a `score` option is `level i: description` for i = 0, 1, and so on, and the two `noul` options are `false: …` and `true: …`, with the defaults `no, the statement does not hold` and `yes, the statement holds` ([`crates/sky-infer/src/prompt.rs`](https://github.com/maxgfr/skycmd/blob/main/crates/sky-infer/src/prompt.rs)).

The tokenizers are byte-level BPE in the GPT-2 style with an NFC normaliser, trained with Hugging Face `tokenizers` at 1,024, 2,048 or 4,096 entries on the text the model will read: each question head, each option with its leading space, and each state ([tutorial, chapter 07](/tutorial/07-tokenizer)). Their special tokens have fixed ids: `[PAD]` 0, `[UNK]` 1, `[CLS]` 2, `[SEP]` 3, `[MASK]` 4 and `[NUM]` 5 ([`docs/SPEC-S2.md`](https://github.com/maxgfr/skycmd/blob/main/docs/SPEC-S2.md) §2). A small vocabulary keeps the embedding table close to the size of the network; the game generators clamp their values to small ranges for the same reason.

### 3.2 One token per measured value

Before BPE, every match of `-?\d+(?:\.\d+)?` on ASCII digits is replaced by `[NUM]` and its value is kept in a parallel array, unless the match is glued to a preceding ASCII letter, as in `b2` or `glm-5` ([`crates/sky-infer/src/num.rs`](https://github.com/maxgfr/skycmd/blob/main/crates/sky-infer/src/num.rs)). With `fuse_numbers`, set in every published s3 and s6 decision file, a `[NUM]` is removed and its value moved onto the token before it, unless that token is `[CLS]`, `[SEP]` or `[MASK]` or already carries a value; then the `[NUM]` stays a token of its own. A pair such as `vy -3` is then one token. The merge happens before truncation, so every window counts merged tokens ([`crates/sky-train/src/s2/data.rs`](https://github.com/maxgfr/skycmd/blob/main/crates/sky-train/src/s2/data.rs)). The embedding of token *i*, whose id is id<sub>i</sub>, whose value is v<sub>i</sub> and whose flag h<sub>i</sub> is 1 when it carries a value, is

<div class="paper-eq">
<p><em>u</em>(<em>v</em>) = sign(<em>v</em>) · ln(1 + |<em>v</em>|), &nbsp;&nbsp; ω<sub><em>k</em></sub> = 2<sup><em>k</em></sup> π / 8, &nbsp; <em>k</em> = 0, …, 7,</p>
<p>φ(<em>v</em>) = [ <em>u</em>, sin(ω<sub>0</sub><em>u</em>), …, sin(ω<sub>7</sub><em>u</em>), cos(ω<sub>0</sub><em>u</em>), …, cos(ω<sub>7</sub><em>u</em>) ] ∈ ℝ<sup>17</sup>,</p>
<p><em>e<sub>i</sub></em> = <em>E</em>[id<sub><em>i</em></sub>] + <em>h<sub>i</sub></em> · ( <em>W</em><sub>num</sub> φ(<em>v<sub>i</sub></em>) + <em>b</em><sub>num</sub> ), &nbsp;&nbsp; <em>W</em><sub>num</sub> ∈ ℝ<sup><em>d</em> × 17</sup>. <span class="paper-eq-n">(1)</span></p>
</div>

The signed logarithm compresses magnitudes, and the eight frequencies give periods from 16 down to 1/8 in *u* (computed here from ω<sub>k</sub>), so nearby values get nearby embeddings and orders of magnitude stay apart; the engine computes the sixteen sines and cosines from one pair by double-angle steps. On the Flappy state used by the benchmarks, value fusion gives 9 tokens instead of 16, `[CLS]` and `[SEP]` included ([`BENCH.md`](https://github.com/maxgfr/skycmd/blob/main/BENCH.md), section "Inference").

### 3.3 Two streams, a cache and isolated options

The **static stream** holds the question and its options. Its tokens are Q = `[CLS]` `{t} question: {ins}` `[SEP]`, followed for each option *i* by O<sub>i</sub> = `[MASK]` ` {option}`, with no final separator. The attention mask lets Q attend to Q and each O<sub>i</sub> attend to Q and to itself only. Positions for the rotary embedding are 0 to |Q| − 1 for Q, and every option starts again at |Q|, so each option's `[MASK]` marker sits at the same position. Each option is cut to its marker and 48 tokens; if the options leave less than 16 tokens of the head window, every option is cut to max(4, (max_head − 16) / n) tokens, and the question keeps at least 8 ([`crates/sky-infer/src/s2/prompt.rs`](https://github.com/maxgfr/skycmd/blob/main/crates/sky-infer/src/s2/prompt.rs)). The static encoder's output at each marker is the option's marker vector m<sub>i</sub>, and its outputs at all |Q| question positions, `[CLS]` and `[SEP]` included, are the question tokens.

Since no option attends to another and all sit at the same positions, permuting the options permutes the marker vectors and, through the head of section 3.5, the probabilities. This holds by construction, as in the prior work of section 2; on the s2 models, permuting the options moved no probability by more than 3e-6 on the CPU ([`BENCH.md`](https://github.com/maxgfr/skycmd/blob/main/BENCH.md), section "s2 training runs").

The **state stream** is `[CLS]` state `[SEP]`, at most max_state tokens, at positions 0 to S − 1, with full bidirectional attention. It does not see the question, so one encoding of a state serves every question asked about it, such as the action and danger questions of one Flappy frame.

The engine keeps an LRU **cache** of 16 static streams, keyed by type, instruction and options. An entry holds the marker vectors, the keys and values of the question tokens for every fusion block, and the first fusion block's queries with their attention over the question tokens already computed. A second cache of 4 entries, keyed by the state text, keeps the exits already computed for a state ([`crates/sky-infer/src/s2/engine.rs`](https://github.com/maxgfr/skycmd/blob/main/crates/sky-infer/src/s2/engine.rs)). In a game, the same questions come back every frame, so only the state is new.

### 3.4 Shared blocks, loops and exits

Both encoders are built from one kind of block. It applies a LayerNorm without bias (ε = 1e-5), a fused query, key and value projection without bias, rotary embeddings in the rotate-half form with θ = 10,000 on queries and keys, scaled dot-product attention and an output projection without bias; then a second LayerNorm and a GeGLU feed-forward, [u<sub>1</sub> | u<sub>2</sub>] = W<sub>up</sub> LN(x) with W<sub>up</sub> of shape 2ff × d, x ← x + W<sub>down</sub> (gelu(u<sub>1</sub>) ⊙ u<sub>2</sub>). As in ModernBERT [[29]](#ref-29), the first block of each encoder has no attention norm on its first application. The number of heads is set per file, or d / 64 by default.

The state encoder has B distinct blocks applied in order, and the sequence is repeated L times. From the second loop on, block 0 normalises its input with a norm of its own, and before loop *l* ≥ 1 (counted from 0) a learned loop embedding loop_embd[*l*] is added to every hidden state, so the shared block knows which pass it is in. Exits sit after given numbers of block passes; exit *e* normalises the hidden states with its own LayerNorm and gives H<sub>e</sub>. s3-pico has B = 1, L = 4 and exits after passes 1 to 4; s6-nano B = 2, L = 2 and exits after 1 to 4; s6-micro B = 2, L = 3 and exits after 2, 4 and 6 ([`configs/`](https://github.com/maxgfr/skycmd/tree/main/configs)). The static encoder is a plain stack of 1 or 2 blocks with a final norm.

### 3.5 Fusion head, scorer and probabilities

The fusion head turns each option's marker into a score, once per exit, with the same weights for every exit. Each option starts from x<sub>i</sub> = m<sub>i</sub> + type_embd[t], its marker plus an embedding of the question's type. Each of the 2 fusion blocks applies a cross-attention whose queries are the options and whose keys and values are [H<sub>e</sub>; question tokens], the state after exit *e* followed by the cached question tokens; then a ReLU feed-forward of width 2d. Every LayerNorm and linear layer of the head has a bias, and there is no rotary embedding. The options attend to the state and the question, never to each other. The scorer gives one logit per option and the probabilities follow:

<div class="paper-eq">
<p><em>s<sub>i</sub></em> = <em>w</em><sub>o</sub> · gelu( <em>W</em><sub>c</sub> LN(<em>x<sub>i</sub></em>) + <em>b</em><sub>c</sub> ) + <em>b</em><sub>o</sub>, &nbsp;&nbsp; <em>p</em><sub><em>i</em></sub><sup>(<em>e</em>)</sup> = exp(<em>s<sub>i</sub></em> / <em>T</em>) / Σ<sub><em>j</em></sub> exp(<em>s<sub>j</sub></em> / <em>T</em>), &nbsp;&nbsp; <em>T</em> = <em>T</em><sub><em>e</em>, <em>t</em>, <em>b</em>(<em>n</em>)</sub>. <span class="paper-eq-n">(2)</span></p>
</div>

The temperature depends on the exit *e*, the type *t* and the bucket *b*(*n*) of the number of options *n*: 2 or fewer, 3 to 5, 6 to 10, or 11 and more (section 5.1). The engine looks up the temperature of (exit, type, bucket), then (exit, type), then (type, bucket), then the type, and uses 1 when none is stored; the training code has only the first two levels ([`crates/sky-infer/src/s2/mod.rs`](https://github.com/maxgfr/skycmd/blob/main/crates/sky-infer/src/s2/mod.rs), [`crates/sky-train/src/s2/config.rs`](https://github.com/maxgfr/skycmd/blob/main/crates/sky-train/src/s2/config.rs)). The answer depends on the type ([`crates/sky-infer/src/engine.rs`](https://github.com/maxgfr/skycmd/blob/main/crates/sky-infer/src/engine.rs)):

<div class="paper-eq">
<p>choice: argmax<sub><em>i</em></sub> <em>p<sub>i</sub></em>, with confidence max(0, (<em>p</em><sub>max</sub> − 1/<em>n</em>) / (1 − 1/<em>n</em>));</p>
<p>score: the expected level Σ<sub><em>i</em></sub> <em>i</em> · <em>p<sub>i</sub></em> over the levels <em>i</em> = 0, …, <em>n</em> − 1; &nbsp;&nbsp; noul: <em>p</em><sub>1</sub>, the probability of <code>true</code>. <span class="paper-eq-n">(3)</span></p>
</div>

At inference the exits run in order, and the engine answers at

<div class="paper-eq">
<p><em>e</em>* = min ( { <em>e</em> : max<sub><em>i</em></sub> <em>p</em><sub><em>i</em></sub><sup>(<em>e</em>)</sup> ≥ τ<sub><em>t</em></sub> } ∪ { <em>E</em> } ), <span class="paper-eq-n">(4)</span></p>
</div>

where *E* is the last exit and τ<sub>t</sub> is one threshold per question type, shared by all exits; the test uses the calibrated probabilities ([`crates/sky-infer/src/s2/engine.rs`](https://github.com/maxgfr/skycmd/blob/main/crates/sky-infer/src/s2/engine.rs)). A file without a threshold reads 0.9. A threshold of 1 makes the engine start directly at the last exit, so the earlier exits are not computed; the s6 files that the manifest marks as published at full depth store 1 for each type (section 5.4). The answer reports the exit and the number of block passes used.

### 3.6 Ternary weights, trained in

Every linear layer of the encoders and of the fusion head is ternary, and so is the scorer's hidden layer. For a weight matrix W, row *r* is quantized with its own scale:

<div class="paper-eq">
<p>γ<sub><em>r</em></sub> = max( (1/<em>k</em>) Σ<sub><em>j</em></sub> |<em>W<sub>rj</sub></em>|, 10<sup>−8</sup> ), &nbsp;&nbsp; <em>Ŵ<sub>rj</sub></em> = γ<sub><em>r</em></sub> · clip( round(<em>W<sub>rj</sub></em> / γ<sub><em>r</em></sub>), −1, 1 ), &nbsp;&nbsp; <em>W</em><sub>used</sub> = <em>W</em> + sg(<em>Ŵ</em> − <em>W</em>), <span class="paper-eq-n">(5)</span></p>
</div>

where *k* is the number of columns and sg stops the gradient: the forward pass uses Ŵ, and the gradient reaches the float master weights W unchanged, the straight-through estimator ([`crates/sky-train/src/s2/model.rs`](https://github.com/maxgfr/skycmd/blob/main/crates/sky-train/src/s2/model.rs)). The scale is the mean absolute value (absmean) as in BitNet b1.58 [[11]](#ref-11), but per output row rather than per tensor. The scorer's output layer, the value projection W<sub>num</sub>, the type and loop embeddings, the norms and the biases stay in float. The token embedding is trained in float and stored in int8 at export, with one scale per row, max |row| / 127; it is not simulated in training ([`crates/sky-train/src/s2/gguf.rs`](https://github.com/maxgfr/skycmd/blob/main/crates/sky-train/src/s2/gguf.rs)).

Quantization starts when the training progress, measured in working time over the budget or in steps, reaches 1 − q, where q is the fraction of the run in QAT. The scripts pass q = 0.9, a float warm-up of 10%, for a model's first decision training, and q = 1.0, every step ternary, for the DAgger rounds; the command-line default, q = 0.3, is the one of the first specification ([`scripts/train-s6.sh`](https://github.com/maxgfr/skycmd/blob/main/scripts/train-s6.sh)). At the switch the trainer validates both forms, saves the float weights, and leaves the learning-rate schedule as it is. Why train the ternary weights rather than round a float model: on the s2 models, ternarizing without QAT cost 6 to 11 points of validation accuracy (pico 0.470 to 0.406, nano 0.542 to 0.432), and 30% of the budget in QAT recovered it (pico 0.482, nano 0.556) ([`BENCH.md`](https://github.com/maxgfr/skycmd/blob/main/BENCH.md), section "s2 training runs").

In the file, each ternary matrix X becomes `X.tq`, bytes of type I8 holding 4 weights each, 2 bits per weight from the least significant bits up (00 for 0, 01 for +1, 10 for −1; 11 is rejected), each row padded to a multiple of 4 weights, and `X.scale`, one 16-bit float per row. Everything else is stored in 32-bit float. This is how s3-pico's 445,345 parameters take 251,104 bytes, against 1.8 MB in 32-bit float ([`BENCH.md`](https://github.com/maxgfr/skycmd/blob/main/BENCH.md), section "skycmd-s3").

### 3.7 What the published decision files do not use

Some options of the code are in no published decision file. *Temporal carry* adds to each frame's state embedding g ⊙ LN(h), with h the previous frame's final latent and the gate g starting at 0; on s3-pico it did not reduce the depth used and was not shipped ([`BENCH.md`](https://github.com/maxgfr/skycmd/blob/main/BENCH.md), section "Temporal recursion"). *Parallax* and *exclusive self-attention* change the encoder's attention; the ablation of section 4.4 adopted neither, so every trained s6 file has plain softmax attention, though some configuration files still say `"parallax": true` before the recipe script overwrites it ([`scripts/s6-adopt.py`](https://github.com/maxgfr/skycmd/blob/main/scripts/s6-adopt.py)). The fast-weight memory is in section 6.5.

## 4. Training

### 4.1 Data

**Game oracles.** Four games are written in Rust: Flappy Drone, Rescue Run, Obstacle Dodge and a Doom-like game ([`crates/sky-games/`](https://github.com/maxgfr/skycmd/tree/main/crates/sky-games)). Each writes its state as short `key value` pairs of small binned integers and computes the useful relations in code: Flappy gives the room above and below the drone rather than absolute heights, so the model maps measurements to a decision ([tutorial, chapter 20](/tutorial/20-dagger-lessons)). An exact oracle gives a target distribution over the options at every decision. The generator draws an exploration rate per episode from 0, 0.05, 0.15 and 0.3, caps Flappy episodes at 12 pipes, sends 30% of the records to each game's secondary question, keeps records with a probability that balances the classes, renders the option keys with random synonyms, and splits by episode seed ([`crates/sky-games/src/gen.rs`](https://github.com/maxgfr/skycmd/blob/main/crates/sky-games/src/gen.rs)).

**Public sets and a numeric curriculum.** Hugging Face decision sets are converted into one record format, with a type, an instruction, rendered options, a state, a target distribution and a split; test-only sets such as `typed_decisions_bench` never reach training ([tutorial, chapter 05](/tutorial/05-hugging-face-data)). A numeric curriculum of `key value` states with exact answers (which key is largest, is a greater than b, and so on) is mixed in at a weight of 0.1 ([`docs/SPEC-S3.md`](https://github.com/maxgfr/skycmd/blob/main/docs/SPEC-S3.md) §3). The s6 models are pretrained on the non-web parts of Dolma 3 and TxT360, then on the earlier web text ([tutorial, chapter 27](/tutorial/27-s6-nano)).

**Teacher labels.** The teacher of s6-nano and s6-micro is GLM-5.3-flash [[41]](#ref-41), an open-weight model under the MIT licence, called through its provider's API, so the laptop does not run it. It receives batches of 20 training questions, each with its type, instruction, options and state cut to 2,000 characters, and is asked for one probability per option, summing to 1, at a sampling temperature of 0.2. Malformed labels are dropped and the others divided by their sum. Only training questions with 2 to 12 options are sent, never a benchmark, held-out or teacher-generated source, and a batch refused by the provider's moderation is halved until each refused text stands alone ([`crates/sky-data/src/distill.rs`](https://github.com/maxgfr/skycmd/blob/main/crates/sky-data/src/distill.rs), [`docs/SPEC-S6.md`](https://github.com/maxgfr/skycmd/blob/main/docs/SPEC-S6.md) §15). A label is kept only when its argmax is the gold answer, and the target becomes 0.5 · gold + 0.5 · teacher, in a separate `distill` group next to the gold records. The teacher labelled 747,077 of the 747,198 selected questions; the moderation refused the other 121 ([`docs/big-run-report.md`](https://github.com/maxgfr/skycmd/blob/main/docs/big-run-report.md) §2). Smart Turn v3.2 and TinyCast, the other teachers, come with their systems in section 6.

### 4.2 Pretraining

The state stack is pretrained as a masked language model ([`crates/sky-train/src/s2/model.rs`](https://github.com/maxgfr/skycmd/blob/main/crates/sky-train/src/s2/model.rs)). 30% of the tokens are masked, at least one per row: 80% become `[MASK]`, 10% a random token and 10% stay. A masked `[NUM]` keeps its id, its value becomes 0 and a learned mask vector is added; a token carrying a fused value is masked as a number or as a word with probability 0.5 each. The loss is the cross-entropy on the masked words plus λ times the mean squared error of a linear head that predicts the signed logarithm *u* of each masked value, clipped to ±12. The specification set λ = 0.5; every training script passes λ = 0.05, because at 0.5 the number error, with |*u*| up to 74 on web text, kept the token loss on its unigram plateau ([`BENCH.md`](https://github.com/maxgfr/skycmd/blob/main/BENCH.md), section "s2 training runs"). The s6 runs read 256-token sequences in batches of 64, 10% of them numeric rows. At the end, the state encoder starts from the pretrained stack and the static encoder from a copy of its first blocks; the fusion head and the scorer start untrained ([`crates/sky-train/src/s2/train.rs`](https://github.com/maxgfr/skycmd/blob/main/crates/sky-train/src/s2/train.rs)).

### 4.3 Decision training

Each exit *e* of the E exits gets a soft-target cross-entropy, −Σ<sub>i</sub> y<sub>i</sub> log p<sub>i</sub><sup>(e)</sup> at temperature 1, with options that are only padding masked out, and the exits are weighted by

<div class="paper-eq">
<p><em>L</em> = Σ<sub><em>e</em></sub> <em>w<sub>e</sub></em> · CE<sub><em>e</em></sub>, &nbsp;&nbsp; <em>w<sub>e</sub></em> ∝ 0.5 + 0.5 · (<em>e</em> − 1) / (<em>E</em> − 1), &nbsp;&nbsp; Σ<sub><em>e</em></sub> <em>w<sub>e</sub></em> = 1, <span class="paper-eq-n">(6)</span></p>
</div>

so the last exit weighs twice the first ([`crates/sky-train/src/s2/config.rs`](https://github.com/maxgfr/skycmd/blob/main/crates/sky-train/src/s2/config.rs)). There is no label smoothing. With probability 0.25 in every script (the *loop dropout*), a batch is supervised only on its first *n* exits, *n* drawn uniformly from 1 to E − 1, the encoder stops at that exit and the weights are renormalised over the exits kept. The draw enters the compiled graph as an input of exit weights, so one graph serves every outcome; this made the s3-pico decision step 4 to 5 times faster than one graph per shape ([`BENCH.md`](https://github.com/maxgfr/skycmd/blob/main/BENCH.md), section "skycmd-s3"). The code also holds a self-distillation of the early exits from the last one and the softened targets of s4; neither is used by the files of this report.

Records are drawn with replacement, a source group by its weight and then an item uniformly, and questions that repeat within a batch share one static stream. The s6 runs also sort batches by state length in pools of 16, round shapes up to multiples of 32 tokens, and split a batch whose step would take more than 3 GB into parts whose gradients are averaged ([`crates/sky-train/src/s2/data.rs`](https://github.com/maxgfr/skycmd/blob/main/crates/sky-train/src/s2/data.rs)).

### 4.4 Optimizer and schedule

Every s6 model is trained with AdamW only: β<sub>1</sub> = 0.9, β<sub>2</sub> = 0.95, ε = 1e-8, decoupled weight decay 0.01 on the matrices but not on the embeddings, norms and vectors, and the gradient clipped to a global norm of 1 ([`crates/sky-train/src/optim.rs`](https://github.com/maxgfr/skycmd/blob/main/crates/sky-train/src/optim.rs)). The schedule is warm-up, stable, decay: the rate rises linearly over the first 2% of the budget, stays flat until 80%, then falls linearly to 0.1 times its peak, where progress is working time over the budget. The peak rate is the phase's default (3e-3 for pretraining, 1e-3 for decision training) times a recipe scale of 2.0, times √(96/d) to move the rate found at width 96 to width *d*, times a per-size pilot scale chosen among 0.5, 1 and 2 (1 for s6-nano, 0.5 for s6-micro) ([`scripts/train-s6.sh`](https://github.com/maxgfr/skycmd/blob/main/scripts/train-s6.sh), [`docs/big-run-report.md`](https://github.com/maxgfr/skycmd/blob/main/docs/big-run-report.md)).

The code also has Muon [[32]](#ref-32) for the encoders' hidden matrices: five Newton-Schulz steps with the coefficients (3.4445, −4.7750, 2.0315), then two with (15/8, −10/8, 3/8), and an update scaled by 0.2 · √max(A, B) for an *A* × *B* matrix ([`docs/SPEC-S6.md`](https://github.com/maxgfr/skycmd/blob/main/docs/SPEC-S6.md) §2). s3-pico, trained before s6, used the code's earlier Muon recipe on those matrices and AdamW elsewhere. Before any s6 size was trained, a pre-registered ablation at the pico size, three seeds per arm, tested Muon and seven other additions; each needed a validation gain of 0.01 (0.005 for the training options) without losing a game, and none was adopted: Muon gained 0.0051 and lost Flappy score ([`runs/recipe/recipe.json`](https://github.com/maxgfr/skycmd/blob/main/runs/recipe/recipe.json); companion report [[1]](#ref-1), section 4.9).

### 4.5 DAgger on the games

A model trained on the oracle's play only sees the oracle's states; once it plays, its small mistakes take it where the oracle never goes. DAgger [[12]](#ref-12) lets the model play and asks the oracle to label the states it reaches. Algorithm 1 is what [`scripts/train-s6.sh`](https://github.com/maxgfr/skycmd/blob/main/scripts/train-s6.sh) and [`scripts/specialize-s3.sh`](https://github.com/maxgfr/skycmd/blob/main/scripts/specialize-s3.sh) run, with the collection in [`crates/sky-arena/src/main.rs`](https://github.com/maxgfr/skycmd/blob/main/crates/sky-arena/src/main.rs) and [`crates/sky-games/src/gen.rs`](https://github.com/maxgfr/skycmd/blob/main/crates/sky-games/src/gen.rs).

<div class="paper-algo" role="figure" aria-label="Algorithm 1">
<p class="paper-algo-title"><strong>Algorithm 1.</strong> DAgger rounds as implemented.</p>
<p><strong>Input:</strong> the checkpoint θ<sub>0</sub> of the decision training and its ternary file M<sub>0</sub>; R rounds (3); N records per game and round (15,000); the oracle share β<sub>1</sub> (0.2); a budget of m working minutes per round; the mixture weights w.</p>
<ol>
<li>For each round r = 1, …, R:</li>
<li class="paper-algo-l1">For each game: play episodes with M<sub>r−1</sub> in the native engine, at the exit thresholds stored in the file. At every decision, record the state with the oracle's target distribution; then the oracle acts with probability β<sub>r</sub> (its best option), and otherwise the model acts (its most probable option). Keep records through the class-balancing filter until N are kept. This gives D<sub>r</sub>.</li>
<li class="paper-algo-l1">θ<sub>r</sub> ← θ<sub>r−1</sub> trained for m working minutes on the base mixture plus D<sub>r</sub> with the weights w, ternary on every step, with a new optimizer state and schedule.</li>
<li class="paper-algo-l1">Fit the temperatures on validation, export M<sub>r</sub> and play 30 episodes per game at full depth, for the record only.</li>
<li class="paper-algo-l1">s6 models: β<sub>r+1</sub> ← β<sub>r</sub> / 2. s3-pico: β<sub>r+1</sub> ← β<sub>r</sub>.</li>
<li><strong>Output:</strong> θ<sub>R</sub> and M<sub>R</sub>, the last round, the checkpoint rule fixed in advance.</li>
</ol>
</div>

Three choices depart from the textbook. The oracle keeps a share β of the decisions, the mixed policy of the original paper applied per decision, so episodes still reach the later parts of each course. Each round trains on its own records and the base mixture, not on the union of all rounds; an aggregated variant was tried in the night experiments and not kept ([`docs/night-experiments.md`](https://github.com/maxgfr/skycmd/blob/main/docs/night-experiments.md)). And the mixture keeps earlier data: games 0.5, the round 0.35, numeric 0.05 and decisions 0.1 for s3-pico; games 0.4, the round 0.25, decisions 0.1, teacher labels 0.2 and numeric 0.05 for s6-nano and s6-micro. On s3-pico, three rounds of 8 minutes took Flappy from 145.3 pipes with 14 crashes to 200 of 200 with none, and the Dodge reflex vetoes from 9,137 to 14 ([`BENCH.md`](https://github.com/maxgfr/skycmd/blob/main/BENCH.md), section "DAgger rounds").

### 4.6 Budgets and the brake

Every run shares the laptop with its owner. A *brake* lets it run hard when the machine is idle and gently when it is in use, hot or on battery ([`docs/idle-training.md`](https://github.com/maxgfr/skycmd/blob/main/docs/idle-training.md), [`crates/sky-train/src/brake.rs`](https://github.com/maxgfr/skycmd/blob/main/crates/sky-train/src/brake.rs)). A daemon writes a duty cycle *d* and a work and rest cycle to a file that the trainer re-reads every 5 seconds: 0.5 with 45 minutes of work and 15 of rest when the machine is in use, hot or on battery, 0.85 with 50 and 10 when it is left alone, and 0.95 with 55 and 5 at night. Before each step the loop sleeps (1/*d* − 1) times the time worked since its last rest, and it naps through the rest part of the cycle. Every rest is added to the start of the run's clock, so the clock counts working time only: a budget of 90 minutes computes the same steps in every mode, and only the wall-clock time changes. The last checkpoint is the result, except where a pre-registration named the moving average of the weights (section 6); no checkpoint is chosen by its validation score.

### 4.7 Recipes

Table 1 gives the settings of seven published files.

**Table 1.** Recipes of seven published files. Parameters and bytes from the [manifest](/models/manifest.json); shapes from the configuration files and [`BENCH.md`](https://github.com/maxgfr/skycmd/blob/main/BENCH.md); budgets and steps from [`BENCH.md`](https://github.com/maxgfr/skycmd/blob/main/BENCH.md) (section of each model) unless noted. Block passes are blocks × loops; windows are state / head tokens. Times are working minutes.

| | s3-pico | s6-nano | s6-micro (v2) | s6-ear2-nano | s6-route-nano | s6-secrets-verify-pico | s6-cast2-pico |
|---|---|---|---|---|---|---|---|
| parameters | 445,345 | 1,486,225 | 5,587,201 | 1,306,178 | 5,587,201 | 445,345 | 377,568 |
| file bytes | 251,104 | 715,136 | 2,420,032 | 631,936 | 2,418,912 | 249,984 | 148,512 |
| width d / ff | 96 / 192 | 144 / 288 | 256 / 768 | 160 / 640 | 256 / 768 | 96 / 192 | 96 / 192 |
| block passes | 1 × 4 | 2 × 2 | 2 × 3 | 4 × 1 | 2 × 3 | 1 × 4 | 2 × 2 |
| vocabulary | 1,024 | 2,048 | 4,096 | none: 43 audio channels per 10 ms | 4,096 | 1,024 | none: patches of 32 points |
| windows | 64 / 96 | 512 / 160 | 384 / 128 | 4 s, 50 positions of 80 ms | 384 / 256 | 320 / 192 | 2,048 points in, 64 out per pass |
| training data | not recorded as one total: the game records of [chapter 04](/tutorial/04-game-cores-and-oracles), decision sets, numeric curriculum, web text | 0.87B pretraining tokens; 2,546,304 decision draws ([study](https://github.com/maxgfr/skycmd/blob/main/docs/s6-nano2-study.md)) | 0.52B pretraining tokens (v1's); 65,471 decision steps of 64 | 270,946 clips of `smart-turn-data-v3.2-train` ([pre-registration](https://github.com/maxgfr/skycmd/blob/main/docs/s6-ear2-prereg.md)) | 81,596 tool, 95,411 intent and 14,621 When2Call records | CredData 49,864, issue reports 40,057 and synthetic 40,000 items read | synthetic series, real series and 1,011 USCRN series (8,282,112 points) |
| pretraining | 8 min, 4,814 steps | 90 min, 53,360 steps | 180 min, 31,700 steps (v1's) | none | from s6-micro's | 12 min, 11,296 steps, from v1's stack | none |
| decision training | 15 min, 6,440 steps | 90 min, 19,893 steps | 360 min, 65,471 steps | 240 min, 305,700 steps | 120 min, 20,504 steps | 30 min, 14,117 steps | 720 min, 377,082 steps |
| DAgger rounds | 3 × 8 min: 4,683, 5,136, 5,313 steps | 3 × 15 min: 4,645, 4,660, 4,644 steps | 3 × 20 min: 3,822, 4,673, 4,969 steps | none | none | none | none |
| QAT | after 10% | after 10% | after 10% | last 15% | last 90% | last 90% | last 20% |
| teacher | none (oracles) | GLM-5.3-flash, 747,077 labels | GLM-5.3-flash | Smart Turn v3.2 | GLM-5.3-flash wrote descriptions (4,174 calls), no labels | none | TinyCast (our port) |

Two cells need a word. s6-micro (v2) kept v1's pretraining and doubled the decision training ([`BENCH.md`](https://github.com/maxgfr/skycmd/blob/main/BENCH.md), section "skycmd-s6"); its QAT cell comes from its script, which passes q = 0.9 (section 3.6), the setting that other result sections call "last 90%". No file records the total of s3-pico's training data; [tutorial, chapter 04](/tutorial/04-game-cores-and-oracles) counts 210,000 game records for the s1 and s2 models and 100,000 Doom records added for s3-pico.

## 5. Calibration and early exits

### 5.1 Temperatures

After training, a temperature is fitted for each (exit, type) pair and, where a bucket holds at least 50 validation questions, for each (exit, type, bucket), on the validation split, for the float and the ternary weights separately ([`crates/sky-train/src/s2/eval.rs`](https://github.com/maxgfr/skycmd/blob/main/crates/sky-train/src/s2/eval.rs)). The bucket is the number of options: 2 or fewer, 3 to 5, 6 to 10, 11 and more. The fit minimises the negative log-likelihood against the soft targets by a golden-section search on ln T over [0.05, 20], 80 iterations ([`crates/sky-train/src/metrics.rs`](https://github.com/maxgfr/skycmd/blob/main/crates/sky-train/src/metrics.rs)). The values go into the file under `temperature.e{E}.{type}[.{bucket}]`. The expected calibration error (ECE) is measured on the top probability with 15 equal-width bins, as in Guo et al. [[30]](#ref-30), per source and over all sources pooled.

### 5.2 Thresholds fitted on held-out questions

The first rule, in the specification of s2, fits one threshold per type on validation questions: the smallest τ in 0.50, 0.51, …, 0.99 whose early-exit accuracy is within 0.005 of full depth, and 1 if none is ([`crates/sky-train/src/s2/eval.rs`](https://github.com/maxgfr/skycmd/blob/main/crates/sky-train/src/s2/eval.rs)). For s2-nano it gave 0.62, 0.62 and 0.69 for choice, score and noul. Played at those thresholds, over 5 arena seeds of 30 episodes, s2-nano crashed in every Flappy episode (15.86 ± 2.30 pipes of 200); the same weights at τ = 0.95 flew 191.94 ± 7.16 pipes ([`loopcal.json`](/results/loopcal.json); companion report [[1]](#ref-1), section 5.1). The held-out questions come from the oracle's play, and the states a model reaches when it plays with early exits are not those.

### 5.3 Thresholds chosen by playing

The closed-loop rule ([`scripts/calibrate-exit.sh`](https://github.com/maxgfr/skycmd/blob/main/scripts/calibrate-exit.sh), Algorithm 2) plays the games at full depth, then at each candidate threshold on the same episodes, and keeps the smallest threshold whose games stay within 2% of full depth with no extra crash or veto. A *veto* is the coded safety reflex of the Dodge game taking over from the model ([tutorial, chapter 20](/tutorial/20-dagger-lessons)).

<div class="paper-algo" role="figure" aria-label="Algorithm 2">
<p class="paper-algo-title"><strong>Algorithm 2.</strong> Closed-loop exit calibration as implemented.</p>
<p><strong>Input:</strong> a ternary file M; the games G (Flappy, Rescue and Dodge); E episodes per game (20) on seed 7; the candidates 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95 and 0.98.</p>
<ol>
<li>Play E episodes of each game at τ = 1. For each game g, keep the reference a<sub>g</sub>: mean score, rescue rate, completion rate, crashes and vetoes.</li>
<li>For each candidate τ, in increasing order:</li>
<li class="paper-algo-l1">Play the same episodes with τ as the threshold of every question type, giving b<sub>g</sub>.</li>
<li class="paper-algo-l1">Accept τ if, for every game g and each of the three rates and scores k, b<sub>g,k</sub> ≥ a<sub>g,k</sub> − 0.02 |a<sub>g,k</sub>|, and b<sub>g</sub> has no more crashes and no more vetoes than a<sub>g</sub>.</li>
<li class="paper-algo-l1">If τ is accepted, stop.</li>
<li><strong>Output:</strong> the accepted τ, or 1 if none is, written as the threshold of the three types.</li>
</ol>
</div>

For s3-pico the rule accepted τ = 0.6 and rejected 0.5 for one extra veto; at 0.6 the model used 1.44, 1.48 and 1.05 of its 4 passes in Flappy, Rescue and Dodge over those 20 episodes ([`BENCH.md`](https://github.com/maxgfr/skycmd/blob/main/BENCH.md), section "Closed-loop exit calibration"). The rule chooses and checks on the same seed. A separate evaluation, `cargo xtask loopcal`, replays fixed thresholds on 5 seeds (7, 11, 13, 17 and 19) of 30 episodes and chooses nothing; on s3-pico's four games it found the same scores and no crash at τ = 1, 0.6 and 0.5 (Dodge vetoes 9.6 ± 4.1 at 1 and 0.6, 9.8 ± 4.4 at 0.5), and it produced the s2-nano numbers of section 5.2 ([`crates/xtask/src/loopcal.rs`](https://github.com/maxgfr/skycmd/blob/main/crates/xtask/src/loopcal.rs), [tutorial, chapter 21](/tutorial/21-s3-recursive-tiny)).

### 5.4 A conformal variant, and why it was rejected

For s6-nano and s6-micro, the thresholds came from conformal risk control [[25]](#ref-25), in the early-exit setting of CALM [[26]](#ref-26) and Fast yet Safe [[27]](#ref-27) ([`crates/sky-arena/src/crc.rs`](https://github.com/maxgfr/skycmd/blob/main/crates/sky-arena/src/crc.rs)). The games are played at full depth, and every decision records, for each exit, its top calibrated probability and its answer. For a candidate τ, the loss of a decision is 1 when the answer at the exit that τ selects differs from the full-depth answer, and R̂(τ) is its mean over the *n* decisions of a type. For each type, the scan goes down from 1.0 to 0.50 in steps of 0.01 and keeps the smallest τ such that

<div class="paper-eq">
<p>( <em>n</em> · <em>R̂</em>(τ′) + 1 ) / ( <em>n</em> + 1 ) ≤ α &nbsp; for τ and every larger τ′ of the grid, <span class="paper-eq-n">(7)</span></p>
</div>

with α = 0.02 in the s6 script. The bound holds when new decisions are exchangeable with the calibration decisions, and those were made at full depth. The pre-registered rule then plays the games at the chosen thresholds, on the same seed and episodes as the reference, with the acceptance test of Algorithm 2, and drops the thresholds if they fail ([`scripts/train-s6.sh`](https://github.com/maxgfr/skycmd/blob/main/scripts/train-s6.sh)). They failed for both general models. s6-nano used 1.01 to 1.83 of its 4 passes per game, but Rescue fell to 2.80 of 3 and the Dodge vetoes rose from 23 to 1,016; s6-micro v2 kept its scores but had 21 vetoes against 20 ([`BENCH.md`](https://github.com/maxgfr/skycmd/blob/main/BENCH.md), section "skycmd-s6", Arena). Both ship at full depth, with a threshold of 1 for each type. Playing with early exits changes the states visited, so a guarantee fitted on full-depth play does not cover them: the lesson of section 5.2, met again with a better tool (companion report [[1]](#ref-1), section 5.1).

## 6. Task-specific systems

### 6.1 End of turn

The end-of-turn detector answers one question: has the speaker finished? Its graph, `skycmd-turn`, reads the last 4 seconds of 16 kHz audio ([`crates/sky-infer/src/audio.rs`](https://github.com/maxgfr/skycmd/blob/main/crates/sky-infer/src/audio.rs), [`crates/sky-infer/src/turn/`](https://github.com/maxgfr/skycmd/tree/main/crates/sky-infer/src/turn), [`docs/s6-ear2-prereg.md`](https://github.com/maxgfr/skycmd/blob/main/docs/s6-ear2-prereg.md)).

- **Front end.** Each 10 ms hop of 160 samples reads a 40 ms span; a periodic Hann window of 25 ms (400 samples) centred in it gives a 512-point FFT, and 40 triangular mel filters from 20 to 8,000 Hz on the power spectrum give log10 energies, floored at 1e-8. Three prosody channels follow. The fundamental frequency f0 and the voicing come from the autocorrelation of the 40 ms span, computed as the FFT of the power spectrum and normalised by the window's own autocorrelation, searched over lags of 40 to 200 samples, that is 80 to 400 Hz, with a small octave cost and a parabolic refinement; the voicing is the peak, clamped to [0, 1], and the f0 channel is log2(f0 / 100) when the voicing reaches 0.45, else 0. The energy channel is the log10 of the summed mel energies. Each hop thus has 43 channels.
- **Normalisation.** Each 4 s window is normalised as Whisper's feature extractor does: with M the largest log-mel value of the window, each mel value x becomes (max(x, M − 8) − M + 4) / 4, and the energy (x − M + 4) / 4; f0 and voicing stay as they are. The window keeps the 400 most recent hops.
- **Network.** Eight hops (80 ms) form one position of 344 values, so 4 s are 50 positions. A float projection with bias maps each position to width 160, then 4 blocks (pre-norm, fused projections without bias, rotary embeddings, full attention and a GELU feed-forward of width 640, 2 heads) with ternary matrices, then an attention pooling, s<sub>t</sub> = W<sub>2</sub> tanh(W<sub>1</sub> h<sub>t</sub> + b<sub>1</sub>) + b<sub>2</sub>, α = softmax(s), h̄ = Σ<sub>t</sub> α<sub>t</sub> h<sub>t</sub>, and a two-layer head with one logit z. The output is σ(z / T), with T = 1.4947854 stored in the file. The shape is the fourth of four sizes, chosen before training as the first whose WebAssembly median was at most 4.0 ms ([`BENCH.md`](https://github.com/maxgfr/skycmd/blob/main/BENCH.md), section "s6-ear2-nano").
- **Distillation.** The teacher is Smart Turn v3.2 [[42]](#ref-42), which reads 8 s; the student reads 4 s. The loss is 0.5 · BCE(y, σ(z)) + 0.5 · BCE(σ(z<sub>T</sub>), σ(z)) + 0.25 · (1 − cos(P h̄, LN e<sub>T</sub>)) + 0.5 · mean<sub>s</sub>(1 − cos(P′ h̄<sub>s</sub>, LN S<sub>T,s</sub>)): the hard label, the teacher's probability, the teacher's 384-dimensional pooled embedding e<sub>T</sub>, and its encoder states pooled to 25 steps S<sub>T</sub>, with the student's positions averaged into the same 25 steps; the projections P and P′ exist only in training ([`crates/sky-train/src/turn/model.rs`](https://github.com/maxgfr/skycmd/blob/main/crates/sky-train/src/turn/model.rs)). Training also shortens the context at random and masks a few mel bands and hops. The pilot kept distillation over hard labels (validation 0.8715 against 0.8437), and the run exported the moving average of the weights at a decay of 0.999.
- **Two thresholds.** The file stores τ<sub>0.5</sub> = 0.5, the threshold the accuracy is read at, and τ<sub>85</sub> = 0.724398136, the highest validation threshold whose recall of finished turns reaches 0.85; the detector answers "finished" when σ(z / T) ≥ τ.

s6-ear2-nano passed its gate on Smart Turn's test clips and is far weaker on real conversational speech; companion report [[1]](#ref-1), section 4.6.

### 6.2 Tool routing

The router picks one tool from a list given at run time, or *none* ([`crates/sky-infer/src/route.rs`](https://github.com/maxgfr/skycmd/blob/main/crates/sky-infer/src/route.rs), [`crates/sky-infer/src/embed.rs`](https://github.com/maxgfr/skycmd/blob/main/crates/sky-infer/src/embed.rs), [`crates/sky-web/src/router.rs`](https://github.com/maxgfr/skycmd/blob/main/crates/sky-web/src/router.rs)).

- **The model part (s6-route-nano).** One `choice` question, "Which tool should handle the user's request?", whose options are the tools as `name: description` and, always last, `none of these tools fits the request`. The state is `User: <request>`, then one line `BM25: <name> score <s> rank <r>; …` with each tool's BM25 score against the request, written with two decimals, and its rank in the slate. BM25 is the Lucene variant of BM25S 0.3.16 [[43]](#ref-43) with k1 = 1.5 and b = 0.75, over the slate's own tools as documents, on lower-cased runs of two or more word characters without 33 English stopwords. The scores go into the state, not the options, because the options are encoded once per tool list and cached (section 3.3); each entry finds its option by the name, and its two numbers fuse onto `score` and `rank` (section 3.2). The model has the s6-micro graph with windows of 384 state and 256 head tokens, and its training data hold counterfactual slates, with the right tool removed or every tool renamed, so that only reading the options lowers the loss ([`BENCH.md`](https://github.com/maxgfr/skycmd/blob/main/BENCH.md), section "s6-route-nano").
- **The embedding part.** potion-base-2M [[44]](#ref-44), a static embedding of dimension 64 and 29,528 tokens, is packed to 4 bits: two codes per byte in −7 to 7 and one 16-bit scale per token, max |v| / 7, in a 1,273,312-byte file with its tokenizer. A text's WordPiece tokens are averaged and the mean is L2-normalised. The request's cosine with each `name: description` is computed, *none* gets the constant θ = 0.39611971, and p<sub>embedding</sub> = softmax([cos<sub>1</sub>, …, cos<sub>n</sub>, θ] / 0.0671021167).
- **Fusion (s6-route).** Nothing is trained. The two distributions are multiplied option by option and renormalised, p(i) ∝ (p<sub>model</sub>(i) · p<sub>embedding</sub>(i))<sup>β</sup>, with β = 0.81, the minimum of the validation negative log-likelihood on a grid from 0.01 to 4.00 ([`scripts/s6-route3-fit.py`](https://github.com/maxgfr/skycmd/blob/main/scripts/s6-route3-fit.py)). The router asks the user instead of answering when the fused top probability is below τ<sub>ask3</sub> = 0.6710473485, the smallest threshold at which accuracy among the answered validation slates reaches 0.95. The model part runs at full depth, or stops at its first exit in a fast mode, and the engine caches the static stream and the tool embeddings of the last tool list.

s6-route-nano failed its gate against BM25, and s6-route passed a gate defined after that test was read, on fresh requests; companion report [[1]](#ref-1), section 4.5.

### 6.3 Secrets: a verifier of rule candidates

The verifier judges one candidate at a time: a rule proposes a value, and the model says whether it is a real credential ([`crates/sky-infer/src/secret_cand.rs`](https://github.com/maxgfr/skycmd/blob/main/crates/sky-infer/src/secret_cand.rs), [`crates/sky-web/src/verify.rs`](https://github.com/maxgfr/skycmd/blob/main/crates/sky-web/src/verify.rs), [`docs/s6-secrets2-prereg.md`](https://github.com/maxgfr/skycmd/blob/main/docs/s6-secrets2-prereg.md)). This is the design of CredSweeper [[14]](#ref-14), Credential Digger [[45]](#ref-45) and FP Remover [[15]](#ref-15).

- **Candidates.** The rules are those of secretgate in a candidate mode: its 222 gitleaks rules and its own, with the entropy gates, the placeholder test, the stopwords and the allowlists turned off, since those are the filters the verifier is meant to replace, plus a rule for a credential keyword in a name followed by a value of 4 to 256 characters and, after the one revision allowed, a rule for authorisation schemes ([`BENCH.md`](https://github.com/maxgfr/skycmd/blob/main/BENCH.md), section "s6-secrets-verify-pico"). Each filter that was turned off becomes a 0 or 1 feature of the candidate.
- **The state.** One part per line, in this order, since a state is truncated from its end: `rule` and its id; `path` with at most 4 scope words taken from the file path (a fixed list of 57 words, such as test or example) and the extension, not the path itself; `key` and the variable name, at most 40 characters; `value` and the value, whole up to 64 characters, else its first and last 32; the features `len`, `ent` (entropy in hundredths of a bit), `cpt` (characters per token, in hundredths), `dig`, `up`, `low` and `oth` (character counts), `word` (the longest run of dictionary words) and the four filter flags; `line`, the value's line with the value replaced by `<V>`, at most 120 characters centred on it; and `prev` and `next`, one line before and one after, at most 100 characters each. Each feature is a `name value` pair, so with value fusion it costs one token. The state window is 320 tokens.
- **The question and the thresholds.** The gated question is a `noul`, "Is this value a real credential that should be revoked?", and P(real) is its probability of `true` at full depth. The model is the s3-pico graph. The file stores three thresholds, each maximising an F-score on the full-depth P(real) of the CredData validation candidates before the test was opened: tau_f1 = 0.2598114610 for F1, tau_f05 = 0.4684373736 for F0.5 and tau_f2 = 0.0780794695 for F2. A candidate is accepted when P(real) ≥ τ.

s6-secrets-verify-pico was above every rule scanner on 221 test repositories and missed its own margin; companion report [[1]](#ref-1), section 4.4.

### 6.4 Forecasting

The forecaster, `skycmd-cast`, reuses the state blocks on patches of a time series ([`crates/sky-infer/src/cast/`](https://github.com/maxgfr/skycmd/tree/main/crates/sky-infer/src/cast), [`crates/sky-train/src/cast/`](https://github.com/maxgfr/skycmd/tree/main/crates/sky-train/src/cast), [`docs/SPEC-S6-CAST.md`](https://github.com/maxgfr/skycmd/blob/main/docs/SPEC-S6-CAST.md)).

- **Input.** The last 2,048 points are scaled to [0, 1] by their minimum and maximum, and cut into 64 patches of 32 points aligned on the last point; a missing value is 0 with a mask bit. Each patch row holds its 32 values and 32 mask bits, and is projected to width 96.
- **Encoder and decoder.** 2 blocks applied twice, with an exit after each block pass. Two learned queries, each standing for 32 future steps, attend to the encoder's output through 2 cross-attention blocks, as in the fusion head; a head gives 32 × 9 outputs per query, made monotone in the quantile level by adding cumulative softplus increments above and below the median. The nine levels are the deciles 0.1 to 0.9.
- **Loss.** The pinball loss, max(τe, (τ − 1)e) for level τ and error e, averaged over observed steps and levels, at every exit with the weights of equation (6); plus the mean absolute gap to the quantiles of TinyCast [[46]](#ref-46) on 50% of the windows, with a weight annealed from 0.5 to 0.
- **The periodicity input (s6-cast2-pico).** TinyCast's period detector, which has no parameters, finds up to 4 significant periods in the context from its periodogram. Each patch gets the sine and cosine of the phase of each period at its last point, 8 channels, and each query gets a seasonal draft, the 32-step lag copy and the 32-step phase-binned mean of the first period, projected and added to it.
- **Early exit.** A pass stops when the median of an exit differs from the previous exit's by less than a tolerance at every step; the tolerance, 0.012839192 for s6-cast2-pico, is the largest whose early median stays within 0.02 of the full-depth median under a conformal bound at α = 0.05, on 2,000 synthetic windows.

s6-cast2-pico improved on its first version and missed its GIFT-Eval bars [[47]](#ref-47); companion report [[1]](#ref-1), section 4.7.

### 6.5 Fast-weight memory

s6-sleep3-nano adds to s6-nano one fast-weight memory layer after each of its two state blocks: each frame of a game writes a small matrix once, and every later frame reads it. Its write and decay gates are held at their initial values, and a loss used in training only decodes the asked colour from the memory; "sleep" passes replay the last 8 frames ([`docs/s6-sleep3-prereg.md`](https://github.com/maxgfr/skycmd/blob/main/docs/s6-sleep3-prereg.md), [`crates/sky-infer/src/s2/memory.rs`](https://github.com/maxgfr/skycmd/blob/main/crates/sky-infer/src/s2/memory.rs)). The memory was used and sleep added nothing measurable, and the file failed its gate; companion report [[1]](#ref-1), section 4.8.

## 7. The engine

**Two graphs in one language.** Training runs in Rust through mlx-rs 0.32.0 on the laptop's GPU; inference runs in plain Rust on the CPU, natively and compiled to WebAssembly ([`crates/sky-infer/`](https://github.com/maxgfr/skycmd/tree/main/crates/sky-infer)). The graph is thus written twice, and golden files tie the two: `sky-train s2-golden` writes a small random model with the probabilities of every exit, and the engine must match them within 1e-4 for float files and 1e-3 for ternary ones; on the s2, s3 and temporal-carry goldens it matched within 6.3e-6 ([`BENCH.md`](https://github.com/maxgfr/skycmd/blob/main/BENCH.md), section "Inference"). The golden tests also check that the engine's tokenizer gives the token streams of Hugging Face `tokenizers`.

**GGUF as a container.** Every file is GGUF version 3, with metadata keys and tensors ([`crates/sky-gguf/`](https://github.com/maxgfr/skycmd/tree/main/crates/sky-gguf)). Only the s1 files, which use llama.cpp's `modern-bert` architecture with the `laya` decision head, also run in llama.cpp and wllama; there, their probabilities agreed with the engine's within 4.7e-4 and 1.1e-4 ([`docs/compat.md`](https://github.com/maxgfr/skycmd/blob/main/docs/compat.md)). Every other architecture is read only by this engine: `skycmd-s2` for s2 to s6, `skycmd-iso` for s5, `skycmd-turn`, `skycmd-cast` and `skycmd-embed`.

**Kernels.** At load, the 2-bit matrices are expanded to 32-bit float: the 2-bit form is what is downloaded and stored, not what is multiplied ([`crates/sky-infer/src/s2/engine.rs`](https://github.com/maxgfr/skycmd/blob/main/crates/sky-infer/src/s2/engine.rs)). The float matrices are repacked in panels of 8 outputs for an outer-product kernel that keeps 16 to 24 accumulators whatever the number of rows; the first block reads a table of the token embeddings already multiplied by its projection, with a 17-input correction for value tokens; and the fusion keys and values of the state are folded into the option queries ([`BENCH.md`](https://github.com/maxgfr/skycmd/blob/main/BENCH.md), section "Inference"). Two opt-in kernels compute on the ternary codes. Lookup tables of the 16 subset sums of each group of 4 inputs, in the style of T-MAC, did not beat the float panels on these small matrices, which stay in the first-level cache, and need two activation passes to stay within 3.5e-4 of the goldens. Int8 dot products are the fastest path natively, 21 against 25 µs per Flappy decision for s3-pico, but the browser module has no option for them, so the float path is the default everywhere.

**Two WebAssembly builds.** [`scripts/build-wasm.sh`](https://github.com/maxgfr/skycmd/blob/main/scripts/build-wasm.sh) compiles the module twice, with SIMD128 and with SIMD128 plus relaxed SIMD, whose fused multiply-add the float kernel uses. The page loads the relaxed build when `WebAssembly.validate` accepts a relaxed instruction, and the SIMD128 build otherwise ([`site/.vitepress/theme/lib/wasm.ts`](https://github.com/maxgfr/skycmd/blob/main/site/.vitepress/theme/lib/wasm.ts)).

**Measurements.** Table 2 gathers the agreement between the browser module and the native engine, and the latencies, as each result section measured them: on the Apple M5, under Node for WebAssembly, often while another job ran, as the sources state.

**Table 2.** Agreement between WebAssembly and native probabilities, and latencies. WebAssembly is the SIMD128 build under Node unless a second value is given for the relaxed build. Sources: [`BENCH.md`](https://github.com/maxgfr/skycmd/blob/main/BENCH.md), the section of each model.

| model | what is timed | native | WebAssembly | largest probability gap, same answers |
|---|---|---:|---:|---|
| s3-pico | one Flappy question, first exit (section "Inference") | 25 µs | 44 µs; relaxed 37 µs | |
| s6-nano | a long bench question, full depth | 7,385 µs | 14.6 to 15.2 ms | 9.5e-7, 200 of 200 bench questions |
| s6-nano | a Flappy frame, two questions and physics | | 436 µs | |
| s6-secrets-pico | one bench decision, full depth | 789 and 786 µs | 1,420 and 1,427 µs | 1.2e-6, 200 of 200 |
| s6-secrets-verify-pico | one candidate, full depth | 1,606 µs | 2,666 µs | 4.17e-07, 200 of 200 |
| s6-ear2-nano | one decision on a 4 s window, one thread | 1.40 ms | 2.39 ms | 1.2e-7, 20 of 20 clips |
| s6-route-nano | one request, tool list cached | 24,031 µs | 38,929 µs; relaxed 23,532 µs | 2.2e-6, 200 of 200 |
| s6-route | one request, tool list cached | 24,648 µs (model part) | 30,879 µs; relaxed 23,536 µs | 1.6e-6 against Python, 200 of 200 |
| s6-cast2-pico | one 64-step forecast, one core | 838 µs | 1,636 µs | |

## 8. Limitations

- **One run per recipe.** Every file of Table 1 was trained once, on one seed. The intervals of the result sections are over test items, not over training runs, so nothing here says how much a recipe moves when trained again.
- **One machine.** Budgets count working minutes on one laptop, often shared with other jobs, so the steps a budget buys depend on the load, as s6-micro v2's DAgger rounds show ([`BENCH.md`](https://github.com/maxgfr/skycmd/blob/main/BENCH.md), section "skycmd-s6").
- **Little was ablated.** The pico ablation tested eight additions at one size and three seeds. Nothing separates the causes of s6-nano's gain over s4-micro (pretraining, teacher, tokenizer and windows changed together), the looped block against a stack of distinct blocks at equal size, or loop dropout, the numeric curriculum and the value fusion from their absence ([tutorial, chapter 27](/tutorial/27-s6-nano)).
- **Shortcuts in the data.** Methods that fit the training data can fit its accidents. In the router's training data, the right tool of every renamed slate was the only option whose description the teacher had rewritten, and the model learned to pick that description ([`docs/s6-route4-diagnosis.md`](https://github.com/maxgfr/skycmd/blob/main/docs/s6-route4-diagnosis.md)); s6-nano answers each question of the bench almost as its most common training answer would (companion report [[1]](#ref-1), section 5.3).
- **Calibration chosen and checked on the same episodes.** The closed-loop rule and the conformal check choose and check on seed 7; the five-seed evaluation came after, for some models only. The conformal bound was fitted on full-depth play and its exchangeability does not hold once exits fire early.
- **Friendly games.** The games compute the useful features in code, so they test the decision, not perception.
- **Two graphs to keep in step.** The training and inference graphs are separate code, tied by golden tests on small random models; a difference the goldens do not exercise would go unseen.

## 9. Reproducibility

The code is in [github.com/maxgfr/skycmd](https://github.com/maxgfr/skycmd), Apache-2.0. The README's table "Reproduce a result" gives, for each s6 model, its pre-registration, its script and its result files ([`README.md`](https://github.com/maxgfr/skycmd/blob/main/README.md#reproduce-a-result)); for the files of Table 1: `scripts/train-s3.sh numeric pico`, then `scripts/specialize-s3.sh pico 3 8` and `scripts/calibrate-exit.sh` for s3-pico ([tutorial, chapter 21](/tutorial/21-s3-recursive-tiny)); `scripts/train-all.sh nano` (which runs `scripts/train-s6.sh`) for s6-nano; `SIZE=micro scripts/train-s6.sh` with the environment of `runs/s6-micro2/env.sh` for s6-micro; `scripts/train-turn.sh` for s6-ear2-nano; `scripts/build-route-data.sh`, `scripts/train-route.sh` and `scripts/route-eval.sh` for s6-route-nano, and `scripts/route3-read.sh` for s6-route, which trains nothing; `scripts/train-secrets2.sh` for s6-secrets-verify-pico; and `scripts/train-cast2.sh` for s6-cast2-pico. The scripts read their data from a folder next to the repository, described in [`docs/data-inventory.md`](https://github.com/maxgfr/skycmd/blob/main/docs/data-inventory.md). Every published file is in [`site/public/models/`](https://github.com/maxgfr/skycmd/tree/main/site/public/models), listed in the [manifest](/models/manifest.json) with its sha256. The figure of this report is written by `python3 -I scripts/paper-methods-figure.py`, and its PDF by `scripts/build-paper.sh --methods`.

## Acknowledgements

Generative AI tools assisted with code and editing; the author checked every number and is responsible for the content.

## References

<ol class="paper-refs">
<li id="ref-1">M. Golfier. <em>Pre-registered gates for small models trained on one laptop: what passed, what failed, and what the readings show.</em> Technical report, not peer-reviewed, Zenodo, 2026. <a href="https://doi.org/10.5281/zenodo.24418316">doi:10.5281/zenodo.24418316</a>; web version: <a href="https://maxgfr.github.io/skycmd/paper/">maxgfr.github.io/skycmd/paper</a></li>
<li id="ref-2">D. Almeida. <em>Introducing System One Models &amp; Jev.</em> TypeSafe blog, September 2026. <a href="https://typesafe.ai/blog/introducing-system-one-models-and-jev">typesafe.ai/blog/introducing-system-one-models-and-jev</a></li>
<li id="ref-3">cua-ai. <em>cua-s1-forms</em>, a form-filling option scorer (MIT). Hugging Face model card, 2026. <a href="https://huggingface.co/cua-ai/cua-s1-forms">huggingface.co/cua-ai/cua-s1-forms</a></li>
<li id="ref-4">Convai Innovations. <em>Laya</em>. Hugging Face model card, 2026. <a href="https://huggingface.co/convaiinnovations/laya">huggingface.co/convaiinnovations/laya</a></li>
<li id="ref-5">B. Egressy, J. Stühmer. <em>Set-LLM: A Permutation-Invariant LLM.</em> 2025. <a href="https://arxiv.org/abs/2505.15433">arXiv:2505.15433</a></li>
<li id="ref-6">G. Amit. <em>Permutation-Robust Decision Modeling with Candidate-Independent Block-Causal Attention.</em> 2026. <a href="https://arxiv.org/abs/2610.01601">arXiv:2610.01601</a></li>
<li id="ref-7">A. Jolicoeur-Martineau. <em>Less is More: Recursive Reasoning with Tiny Networks.</em> 2025. <a href="https://arxiv.org/abs/2510.04871">arXiv:2510.04871</a></li>
<li id="ref-8">R.-J. Zhu, Z. Wang, K. Hua, et al. <em>Scaling Latent Reasoning via Looped Language Models.</em> 2025. <a href="https://arxiv.org/abs/2510.25741">arXiv:2510.25741</a></li>
<li id="ref-9">Y. Gorishniy, I. Rubachev, V. Khrulkov, A. Babenko. <em>Revisiting Deep Learning Models for Tabular Data.</em> NeurIPS 2021. <a href="https://arxiv.org/abs/2106.11959">arXiv:2106.11959</a></li>
<li id="ref-10">T. Zhou, D. Fu, M. Soltanolkotabi, R. Jia, V. Sharan. <em>FoNE: Precise Single-Token Number Embeddings via Fourier Features.</em> 2025. <a href="https://arxiv.org/abs/2502.09741">arXiv:2502.09741</a></li>
<li id="ref-11">S. Ma, H. Wang, L. Ma, et al. <em>The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits.</em> 2024. <a href="https://arxiv.org/abs/2402.17764">arXiv:2402.17764</a></li>
<li id="ref-12">S. Ross, G. J. Gordon, J. A. Bagnell. <em>A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning.</em> AISTATS 2011. <a href="https://arxiv.org/abs/1011.0686">arXiv:1011.0686</a></li>
<li id="ref-13">Y. Yue, Y. Wang, B. Kang, Y. Han, S. Wang, S. Song, J. Feng, G. Huang. <em>DeeR-VLA: Dynamic Inference of Multimodal Large Language Models for Efficient Robot Execution.</em> NeurIPS 2024. <a href="https://arxiv.org/abs/2411.02359">arXiv:2411.02359</a></li>
<li id="ref-14">Samsung. <em>CredSweeper</em>, a credential scanner with an optional ML validator (MIT). GitHub repository, 2021. <a href="https://github.com/Samsung/CredSweeper">github.com/Samsung/CredSweeper</a></li>
<li id="ref-15">F. Boas. <em>ML-powered feature FP Remover cuts 50% of False Positives.</em> GitGuardian blog, June 2024. <a href="https://blog.gitguardian.com/fp-remover-cuts-false-positives-by-half/">blog.gitguardian.com/fp-remover-cuts-false-positives-by-half</a></li>
<li id="ref-16">mgoeckel. <em>oscar-1-17m</em>, a decision model on Ettin-17M. Hugging Face model card, 2026. <a href="https://huggingface.co/mgoeckel/oscar-1-17m">huggingface.co/mgoeckel/oscar-1-17m</a></li>
<li id="ref-17">TheREZOR. <em>TinyDecide</em> (Apache-2.0). Hugging Face model card, 2026. <a href="https://huggingface.co/TheREZOR/TinyDecide">huggingface.co/TheREZOR/TinyDecide</a></li>
<li id="ref-18">Q. Li, X. Wang. <em>Jeva: System One Models from First Principles.</em> 2026. <a href="https://arxiv.org/abs/2610.18773">arXiv:2610.18773</a></li>
<li id="ref-19">M. Dehghani, S. Gouws, O. Vinyals, J. Uszkoreit, Ł. Kaiser. <em>Universal Transformers.</em> ICLR 2019. <a href="https://arxiv.org/abs/1807.03819">arXiv:1807.03819</a></li>
<li id="ref-20">C.-Y. Lee, S. Xie, P. Gallagher, Z. Zhang, Z. Tu. <em>Deeply-Supervised Nets.</em> 2014. <a href="https://arxiv.org/abs/1409.5185">arXiv:1409.5185</a></li>
<li id="ref-21">G. Huang, Y. Sun, Z. Liu, D. Sedra, K. Weinberger. <em>Deep Networks with Stochastic Depth.</em> 2016. <a href="https://arxiv.org/abs/1603.09382">arXiv:1603.09382</a></li>
<li id="ref-22">S. Teerapittayanon, B. McDanel, H. T. Kung. <em>BranchyNet: Fast Inference via Early Exiting from Deep Neural Networks.</em> 2017. <a href="https://arxiv.org/abs/1709.01686">arXiv:1709.01686</a></li>
<li id="ref-23">J. Xin, R. Tang, J. Lee, Y. Yu, J. Lin. <em>DeeBERT: Dynamic Early Exiting for Accelerating BERT Inference.</em> ACL 2020. <a href="https://arxiv.org/abs/2004.12993">arXiv:2004.12993</a></li>
<li id="ref-24">W. Liu, P. Zhou, Z. Zhao, Z. Wang, H. Deng, Q. Ju. <em>FastBERT: a Self-distilling BERT with Adaptive Inference Time.</em> ACL 2020. <a href="https://arxiv.org/abs/2004.02178">arXiv:2004.02178</a></li>
<li id="ref-25">A. N. Angelopoulos, S. Bates, A. Fisch, L. Lei, T. Schuster. <em>Conformal Risk Control.</em> ICLR 2024. <a href="https://arxiv.org/abs/2208.02814">arXiv:2208.02814</a></li>
<li id="ref-26">T. Schuster, A. Fisch, J. Gupta, M. Dehghani, D. Bahri, V. Q. Tran, Y. Tay, D. Metzler. <em>Confident Adaptive Language Modeling.</em> NeurIPS 2022. <a href="https://arxiv.org/abs/2207.07061">arXiv:2207.07061</a></li>
<li id="ref-27">M. Jazbec, A. Timans, T. Hadži Veljković, K. Sakmann, D. Zhang, C. A. Naesseth, E. Nalisnick. <em>Fast yet Safe: Early-Exiting with Risk Control.</em> NeurIPS 2024. <a href="https://arxiv.org/abs/2405.20915">arXiv:2405.20915</a></li>
<li id="ref-28">J. Nielsen, L. Galke, P. Schneider-Kamp. <em>When are 1.58 bits enough? A Bottom-up Exploration of BitNet Quantization.</em> 2024. <a href="https://arxiv.org/abs/2411.05882">arXiv:2411.05882</a></li>
<li id="ref-29">B. Warner, A. Chaffin, B. Clavié, O. Weller, et al. <em>Smarter, Better, Faster, Longer: A Modern Bidirectional Encoder for Fast, Memory Efficient, and Long Context Finetuning and Inference.</em> 2024. <a href="https://arxiv.org/abs/2412.13663">arXiv:2412.13663</a></li>
<li id="ref-30">C. Guo, G. Pleiss, Y. Sun, K. Q. Weinberger. <em>On Calibration of Modern Neural Networks.</em> ICML 2017. <a href="https://arxiv.org/abs/1706.04599">arXiv:1706.04599</a></li>
<li id="ref-31">G. Hinton, O. Vinyals, J. Dean. <em>Distilling the Knowledge in a Neural Network.</em> NIPS 2014 Deep Learning Workshop, 2015. <a href="https://arxiv.org/abs/1503.02531">arXiv:1503.02531</a></li>
<li id="ref-32">K. Jordan, Y. Jin, V. Boza, J. You, F. Cesista, L. Newhouse, J. Bernstein. <em>Muon: An optimizer for hidden layers in neural networks.</em> Blog post, December 2024. <a href="https://kellerjordan.github.io/posts/muon/">kellerjordan.github.io/posts/muon</a></li>
<li id="ref-33">K. Wen, D. Hall, T. Ma, P. Liang. <em>Fantastic Pretraining Optimizers and Where to Find Them.</em> 2025. <a href="https://arxiv.org/abs/2509.02046">arXiv:2509.02046</a></li>
<li id="ref-34">M. Akhtar, H. N. Nounou, M. Mansouri, M. N. Nounou. <em>When Does Muon Actually Win? A Paired Data-, Time-, and Tuning-Budget Benchmark.</em> 2026. <a href="https://arxiv.org/abs/2610.21555">arXiv:2610.21555</a></li>
<li id="ref-35">K. Helwani, H. Do, J. Luan, S. Srinivasan. <em>A Hierarchical End-of-Turn Model with Primary Speaker Segmentation for Real-Time Conversational AI.</em> 2026. <a href="https://arxiv.org/abs/2603.13379">arXiv:2603.13379</a></li>
<li id="ref-36">aurelio-labs. <em>semantic-router</em> (MIT). GitHub repository, 2023. <a href="https://github.com/aurelio-labs/semantic-router">github.com/aurelio-labs/semantic-router</a></li>
<li id="ref-37">S. Robertson, H. Zaragoza. <em>The Probabilistic Relevance Framework: BM25 and Beyond.</em> Foundations and Trends in Information Retrieval, 2009. <a href="https://doi.org/10.1561/1500000019">doi:10.1561/1500000019</a></li>
<li id="ref-38">Samsung. <em>CredData</em>, a credential data set of open-source repositories. GitHub repository, 2021. <a href="https://github.com/Samsung/CredData">github.com/Samsung/CredData</a></li>
<li id="ref-39">T. Zhan, H. Zhang, Y. He. <em>Fracast-0: Fractal Weight Sharing for a Time Series Foundation Model with Only 85K Parameters.</em> 2026. <a href="https://arxiv.org/abs/2609.32209">arXiv:2609.32209</a></li>
<li id="ref-40">Tracel AI. <em>Burn</em>, a deep learning framework in Rust. GitHub repository, 2022. <a href="https://github.com/tracel-ai/burn">github.com/tracel-ai/burn</a></li>
<li id="ref-41">zai-org. <em>GLM-5.3-Flash</em>, an open-weight model (MIT). Hugging Face model card, 2026. <a href="https://huggingface.co/zai-org/GLM-5.3-Flash">huggingface.co/zai-org/GLM-5.3-Flash</a></li>
<li id="ref-42">Pipecat. <em>Smart Turn v3.2</em>, an open end-of-turn model (BSD-2-Clause). <a href="https://github.com/pipecat-ai/smart-turn">github.com/pipecat-ai/smart-turn</a></li>
<li id="ref-43">X. H. Lù. <em>BM25S: Orders of magnitude faster lexical search via eager sparse scoring.</em> 2024. <a href="https://arxiv.org/abs/2407.03618">arXiv:2407.03618</a></li>
<li id="ref-44">Minish Lab. <em>potion-base-2M</em>, a Model2Vec static embedding (MIT). Hugging Face model card, 2024. <a href="https://huggingface.co/minishlab/potion-base-2M">huggingface.co/minishlab/potion-base-2M</a></li>
<li id="ref-45">SAP. <em>Credential Digger</em> (Apache-2.0). GitHub repository, now archived, 2020. Method in S. Lounici, M. Rosa, C. M. Negri, S. Trabelsi, M. Önen, <em>Optimizing Leak Detection in Open-Source Platforms with Machine Learning Techniques</em>, ICISSP 2021. <a href="https://github.com/SAP-archive/credential-digger">github.com/SAP-archive/credential-digger</a></li>
<li id="ref-46">RAWS Labs. <em>TinyCast</em>, a 146,505-parameter forecaster (Apache-2.0). Hugging Face model card, 2026. <a href="https://huggingface.co/raws-labs/tinycast">huggingface.co/raws-labs/tinycast</a></li>
<li id="ref-47">T. Aksu, G. Woo, J. Liu, X. Liu, C. Liu, S. Savarese, C. Xiong, D. Sahoo. <em>GIFT-Eval: A Benchmark For General Time Series Forecasting Model Evaluation.</em> 2024. <a href="https://arxiv.org/abs/2410.10393">arXiv:2410.10393</a>; data: <a href="https://huggingface.co/datasets/Salesforce/GiftEval">huggingface.co/datasets/Salesforce/GiftEval</a></li>
</ol>

## How to cite

```bibtex
@misc{golfier2026skycmdmethods,
  author       = {Golfier, Maxime},
  title        = {Small ternary decision models trained on one laptop:
                  architecture, training, closed-loop calibration
                  and a Rust engine for the browser},
  year         = {2026},
  publisher    = {Zenodo},
  howpublished = {\url{https://maxgfr.github.io/skycmd/paper/methods/}},
  note         = {Technical report, not peer-reviewed}
}
```

The same citation, with the companion report's, is in [`CITATION.cff`](https://github.com/maxgfr/skycmd/blob/main/CITATION.cff). The report is updated when a method's description changes; the date at the top is that of its last change.

</div>
