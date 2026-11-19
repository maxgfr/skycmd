# Teaching a tiny ternary model to read and write an explicit memory: a research dossier for s6-sleep v4

Research date: 19 November 2026. This file reviews published work on making small transformers and recurrent models store facts in an explicit memory at inference and recall them later, and reads it against the state of s6-sleep after v4's first check. It records research, not a decision: what follows from it is the owner's to decide, and any experiment gets its own pre-registration. Each fact gives its source as a number in brackets, listed at the end with its URL and title. Text marked **[Inference]** is my own reasoning, not a claim from a source. "Not verified" means I looked and could not confirm it. The project's own numbers come from `docs/s6-sleep3-prereg.md` (Result block and the note of 17/11/2026), `docs/s6-sleep4-diagnosis.md` and `docs/s6-sleep4-prereg.md` (§15, amendment 1).

---

## 0. Summary

- **The closest published match to our failure** is a 2026 preprint on fixed-state recurrent cells (delta rule, Mamba-2, RWKV-7). Cells that recall 32 pairs fall to chance when they must retrieve 4 pairs across a stretch of distractor tokens. The authors call this a training problem, not a capacity limit: a curriculum on the distance between the stored pairs and the query takes the unchanged model from 0.021 to 1.000 [1][2]. Training there is "close to a coin flip": the curriculum raises the share of seeds that lock in from 1/10 to 7/10 [2].
- **Recurrent and fast-weight memories train in narrow learning-rate windows**, often less than an order of magnitude wide, where attention is robust [4]. The 2026 preprint finds the same for delta-rule cells [1].
- **A memory on a frozen base works in the published systems** only when the base is a large pretrained model whose hidden states already separate what is stored, and when a trained reader or trained projections sit between the memory and the output [13][15][31]. One retrofit to a frozen 4B model found the hidden states of distinct words at cosine similarity above 0.98 and had to separate them with trained projections [31].
- **Plateaus followed by abrupt transitions** are well documented for induction heads and sparse attention, with timing that depends on context length, architecture, data and optimiser [25][26][27][28][29].
- **What the evidence does not cover:** no fetched source trains an explicit read-write memory below about 0.5 million parameters, none with ternary weights, and none compares a memory on a frozen small base with the same memory co-trained.
- **[Inference]** The three answers (section 13): the pattern fits interference under sparse supervision plus a slow, seed-dependent lock-in at the nano size, and a frozen base whose representations do not separate doors and colours. The cheapest discriminating step is v4's own next check, K1 at seed 0 (about 54 GPU working minutes), together with a CPU reading of key overlap at the nano initialisation. The literature's strongest lever, a delay curriculum, is not among v4's pre-registered factors.

---

## 1. Where s6-sleep stands (from the project's files)

- **v3, nano size (1.6M parameters, ternary):** `recall` 0.4892 at N = 1 on the test against 0.2440 for the control. The pico pilot (about 0.47M parameters) had answered 1.0000 at every delay on dev2 (`docs/s6-sleep3-prereg.md`, Result block).
- **The v4 diagnosis.** The nano runs ended halfway up their learning curve, with 393 `recall` steps at seed 0 against 933 for the pilot. But at an equal count of steps the pilot had finished and nano was at 0.38 to 0.50, so the slower learning per step was the larger factor. The pilot went from chance to 0.995 in about 70 `recall` steps; the fastest nano seed took 116 to reach 0.50. The decay α was ruled out as the cause of the delay curve. A retention of about 0.93 to 0.95 per frame fits it, which points to later writes erasing earlier ones. That is a fit under an assumed model; no key overlap was logged at the nano size (`docs/s6-sleep4-diagnosis.md` §1, §2 (a), (d)).
- **What differs between the sizes and was never varied alone:** the rate (0.82 times the pilot's), the auxiliary loss averaged over two memory layers (0.5 per layer), two heads of 72 dimensions against one of 96, two loops against four, the base and the mix. In phase A, 88% to 98% of nano `recall` steps were clipped at a gradient norm of 1.0, against 46% in the pilot (diagnosis §2 (e); note of 17/11, item 5).
- **K3, the memory trained on a frozen base, failed** on both seeds: `ask_acc` 0.2579 and 0.2546 over the last 100 `recall` steps against a bar of 0.95. The auxiliary heads stayed at chance on seed 1 and reached 0.38 on seed 0, and the memory's gain over the same weights without it was about 0.03 at every delay (`docs/s6-sleep4-prereg.md` §15).
- **What remains of v4.** Under its rules the frozen route is closed. Next come K1 (v3's recipe, everything trained, 1,310 steps), K2 (four single factors), a confirmation, then K5 and the preserved route. Their code is not written on `main`. The pre-registration's own figures put stage 1 at 216 to about 369 GPU working minutes from here, and the preserved route's stage 2 at up to about 8 hours more (§15).

---

## 2. Fast weights and Hebbian or associative memories

- **Fast weights** (Ba et al., NeurIPS 2016) are variables that change faster than ordinary weights and slower than activities. They store temporary memories of the recent past without keeping copies of activity patterns [8].
- **Weight matrices as associative memories.** In a two-layer transformer, global bigrams are learned fast and the induction head for in-context bigrams develops more slowly; the weight matrices act as associative memories [9]. Scaling laws in sample size and parameter size have been derived for memories built from outer products of embeddings [10].
- **Constant-Memory Recall (2026)** compares models of about 14M parameters trained from scratch on synthetic episodes. A DeltaNet variant whose two fp32 matrix states occupy 32,768 bytes learns to recall 32 new pairings per sequence. Its vector-recurrence and Transformer baselines stay near chance despite falling training loss. Its keys blend learned projections with fixed random token-specific vectors, `k = normalize(0.42 k_raw + 0.58 b_x)`, and a 48-pair test stayed near chance for every model [7].
- **Measured at:** a few million to 14M parameters, floating point, synthetic recall [7][8].
- **[Inference] Transfer.** Our memory already writes by a delta rule into a per-sequence matrix with decay (diagnosis §2 (d)). What transfers is how the working models make keys separable: by normalisation and by fixed token-specific key components.

## 3. Modern Hopfield layers

- Attention is the update rule of a modern Hopfield network. It has three kinds of fixed points: a global average over all patterns, metastable states averaging over a subset, and single-pattern fixed points. The parameter β governs which one a head reaches and strongly affects the gradient flow to the query and key matrices [11].
- **[Inference]** A read that averages over several stored patterns returns a blend instead of the stored colour. Whether our linear read sits in that regime was never measured.

## 4. Key-value memory networks and differentiable neural dictionaries

- Key-Value Memory Networks use different encodings for the addressing stage and the output stage of the read [12].
- **[Inference]** These designs, and the differentiable neural dictionary of Neural Episodic Control, are trained end to end with their controller, on question answering or reinforcement learning. They are far from our size and task and suggest no specific remedy.

## 5. kNN-LM, retrieval at the output, Memorizing Transformers, LongMem

- **kNN-LM** extends a pretrained 247M-parameter LM by interpolating its next-word distribution with a nearest-neighbour distribution over its own embeddings, with no change to the LM. The authors argue that assigning train and test instances similar representations is an easier problem than memorising the next word in parameters [13].
- **Memorizing Transformers** add an approximate kNN lookup into a non-differentiable memory of recent keys and values. They normalise keys and queries so that old and new keys do not differ in magnitude. Training from scratch with a large memory sometimes did worse than training with a small memory and fine-tuning [14].
- **LongMem** keeps the backbone LLM frozen as the memory encoder and trains a residual side network as the retriever and reader [15].
- **[Inference] Transfer.** These systems store representations that a large pretrained model already has, and the read side is trained even when the encoder is frozen.

## 6. Memory layers at scale

- Memory layers replace the feed-forward network of one or more layers with a trainable key-value lookup. They were tested over base models from 134M to 8B parameters, with memories up to 128B parameters. Keys and values are trainable parameters and only the top-k keys are read. Training can become unstable for large memory layers, especially with small base models, and qk-normalisation is used against it [16].
- **[Inference]** A memory layer stores training-time knowledge, not facts written at inference, so it does not address our task. Its stability remedy, normalised queries and keys, does carry over.

## 7. Titans, TTT and test-time memorisation

- **Titans** adds a neural long-term memory that learns to memorise historical context at test time, next to attention used as a short-term memory [17].
- **TTT layers** make the hidden state a model updated by a step of self-supervised learning, evaluated at 125M to 1.3B parameters [18].
- **Do Language Models Need Sleep?** (the paper s6-sleep started from) writes fast weights in SSM blocks during N offline recurrent passes before clearing the key-value cache. A longer sleep improves performance, most on examples that need deeper reasoning [19].
- **[Inference]** These are general forms of what s6-sleep implements. None reports a model near our size or a ternary one.

## 8. Larimar and episodic memory editing

- Larimar couples a BERT-large encoder with a GPT2-large or GPT-J-6B decoder through a 512 × 768 Kanerva-style memory matrix. It was trained on 7.6 million chunks of 64 tokens, starting from pretrained weights. The write computes addressing weights by a pseudo-inverse against a learned prior memory [20].
- **[Inference]** It works at 1.3B and 6B parameters with pretrained parts, and says nothing about a model of a few million parameters.

## 9. Complementary learning systems, sleep and replay

- **Complementary learning systems (CLS) theory** says an agent needs a slow learner of structure and a fast learner of specific experiences, and it gives replay a broadened role [21].
- **Sleep Replay Consolidation** interleaves backpropagation with offline phases of local unsupervised Hebbian plasticity on noisy input, and recovers old tasks that were otherwise forgotten [22]. A 2026 follow-up applies it after several sequential tasks, in fully connected networks with two hidden layers of 1,200 units on MNIST and Fashion-MNIST [23].
- **HOLA** keeps a bounded exact key-value store next to the Gated DeltaNet state, chosen by delta-rule write magnitude. It reports that reading that store with the delta rule's L2-normalised keys gives a nearly uniform softmax, and that a separate normalisation restores sharp retrieval [24].
- **[Inference]** The sleep work addresses forgetting across tasks, not recall within an episode, so it does not predict whether sleep passes help `recall`. At the nano size our sleep passes raised short delays and lowered long ones (diagnosis §2 (d)).

## 10. In-context associative recall: width, depth, heads and training

- **Zoology.** 82% of the perplexity gap between gated-convolution and attention models is explained by associative recall, and a 70M attention model outperforms a 1.4B gated-convolution model on it [5].
- **Okpekpe and Orvieto (2025).** Modern recurrent models succeed only in an extremely narrow window of learning rates, while Transformers are robust. This brittleness makes recurrent models favour width over depth. A one-layer attention model does not exceed chance on recall, and a deeper but narrower Mamba fails to copy where a wider one with the same parameters succeeds [4].
- **The Anatomy preprint (2026)** studies recall at a fixed state of 1,024 elements. In the authors' summary, cells that recall 32 pairs still fail, at chance, to retrieve 4 pairs across a long stretch of distractor text. They call it a training problem, not a capacity limit: a distance curriculum (training on gradually longer gaps) takes the same model from 0.021 to 1.000. Whether a run learns to retrieve is close to a coin flip, and the curriculum's shape changes the odds from 1/10 to 7/10 seeds, and to 6/6 at sequence length 512 when the curriculum advances only while measured accuracy holds [2].
- A digest of the preprint says the authors blame distractor writes overwriting the stored pairs when only one token per sequence is supervised, not limited capacity or decay [3]. In its learning-rate sweep over a decade, both delta-rule arms peak at 3 × 10⁻³ and destabilise at 10⁻², with collapsed seeds [1].
- **ARMT.** Its associative-retrieval experiments use models of about 500k parameters (four layers, hidden size 128), trained with a curriculum over the number of key-value pairs: 1, 2, 3, 5, 10, 20, 40 and more [6].
- **[Inference] Transfer.** This is the strand closest to our setting in size and task. Our probe stores three doors, asks after 1 to 32 frames, every frame writes, and only the ask frames are supervised: few pairs, distractor writes, sparse supervision.

## 11. Induction heads, phase transitions, grokking-like delays

- Transformer language models go through a phase change early in training, during which induction heads form and in-context learning improves sharply [25].
- In small attention-only models, three interacting subcircuits drive the formation of induction heads and explain data-dependent properties such as the timing of the phase change [26]. The abrupt emergence has been traced to the sequential learning of three nested logits [27].
- In a minimal setting, three dimensions account for the emergence, and there is a tight bound on the total emergence time in terms of the context length [28].
- The emergence time of sparse attention follows power laws in task structure, architecture and optimiser. Repetition can greatly speed it up, and the result is confirmed on an in-context associative recall task [29].
- Grokking: networks can go from chance to perfect generalisation long after fitting their training set, and smaller datasets need more optimisation [30].
- **[Inference] Transfer.** The pico pilot's jump from chance to 0.995 in about 70 `recall` steps looks like such a transition. A slower one at another size and configuration is expected in kind, but nothing fetched predicts its length for our model.

## 12. Frozen base against co-trained memory; quantisation

- **Frozen base.** kNN-LM needs no further training because the pretrained representation already maps similar contexts together [13]. LongMem freezes the backbone but trains the reader [15].
- Prometheus Mind retrofits memory to a frozen Qwen3-4B with 11 adapters and reports three things. Joint optimisation of the adapters collapsed to random performance, while stage-wise training on simple proxy tasks worked. Learned value encoders memorised their training data (100% train, 0% test). Hidden states of distinct words had cosine similarity above 0.98, brought down to 0.09 by trained projections. 86% of the variance of those hidden states was shared structure [31].
- Memorizing Transformers note that models cope with limited staleness of stored keys by adjusting their queries [14].
- **Ternary and quantised networks.** 1.58-bit QAT on small language and vision models from 100K to 48M parameters gives near state-of-the-art small language models when hidden sizes are doubled [32]. 1.58-bit training is on par with, or sometimes better than, 16/32-bit for MLPs, GNNs, encoder-only and encoder-decoder models, with more capacity compensating for lower precision in encoder-only models [33]. Quantising delta-rule recurrent states (not weights) in large models gives larger accumulated error for heads with longer memory lifetimes [34]. MatMul-free language models use ternary weights with a recurrent token mixer up to 2.7B parameters [35].
- **Not verified:** a search snippet said the MatMul-free authors first tried ternary query and key matrices and failed to converge. The fetched abstract does not say so.
- **[Inference]** In every working frozen-base system found, the base was a large model pretrained on the content later stored, and something trainable sat between the memory and the output. Neither held in K3: our base was never trained to separate doors and colours where the memory reads it. And nothing fetched measures ternary key, query and value projections feeding a fast-weight memory.

---

## 13. The three answers

### 13.1 Which known causes fit our pattern (pico 1.0, nano about half, frozen base at chance)

| Known cause | Pico 1.0 | Nano about 0.49 | Frozen base at chance | Evidence |
|---|---|---|---|---|
| Interference from distractor writes under sparse supervision, a training problem broken by a distance curriculum, with seed-dependent lock-in | fits, as a run that locked in | fits: long delays lag, about 6% lost per frame, every frame writes, only asks are supervised | fits in part: nothing locked in within about 1,000 steps | [1][2][3]; diagnosis §2 (d) |
| Plateau then abrupt emergence, timed by architecture and optimiser | fits: chance to 0.995 in about 70 steps | fits: left chance at about the same step, then rose several times more slowly and stopped mid-curve | weak | [25][26][27][28][29]; diagnosis §2 (a) |
| Narrow learning-rate window of recurrent memories | fits | possible: rate 0.82 times the pilot's, 88% to 98% of steps clipped | possible | [1][4]; note of 17/11, item 5 |
| Frozen representations that do not separate what is stored, with no trained reader | does not apply | does not apply | fits: what the memory retrieved was not decodable for the colour on seed 1 | [13][15][31]; prereg §15 |
| Too little capacity | no | no: three doors against 72 dimensions per head; ruled out as stated by the diagnosis | no | [2]; diagnosis §2 (d) |
| Ternary weights | not tested | not tested | not tested | [32][33] |

**[Inference]** The best-supported reading of the nano gap combines the first two rows. The published wall has the same ingredients as `recall`, and it was broken by a training signal, not by a change of architecture. The pico run may have locked in early where the nano runs were still climbing. The frozen-base failure matches the fourth row. The learning-rate window and the clipping are plausible contributors, and v4's K2 (iii) and (iv) would test them.

### 13.2 The cheapest discriminating experiment

- **K1 at seed 0 alone, before anything else of the remaining route.** It is v4's next pre-registered check: v3's recipe, everything trained, 1,310 steps, about 1,002 `recall` steps. Its trainer flags exist, and the pre-registration estimates 54 GPU working minutes per seed (§3.7, §11). Its script and report parts must still be written; they are in the unfinished patch.
- **[Inference] What it separates.** Suppose seed 0 reaches K1's bar (0.95 over its last 100 `recall` steps), or crosses S90 on a rising curve. Then the nano gap was mainly budget and transition timing, and v4's remaining route is on the path the literature expects. Suppose instead it stays near v3's level, with the excess ratio of long to short delays still low. Then the interference reading gains weight, and the lever the literature gives for it, a delay curriculum, is not one of v4's four K2 factors. v4 forbids adding a factor (§10).
- **A zero-GPU reading to go with it:** C2's key-overlap reading at nano initialisation (`s2-memcheck --c2-only`, on the CPU), against the pico's 0.73, as the diagnosis proposed (§4, item 3). The literature's attention to separable keys makes this number informative [7][16][31]: normalised queries and keys, fixed token-specific key components, trained projections against collapsed hidden states.
- For scale: the full remaining v4 route is 216 to about 369 GPU working minutes of checks, then up to about 8 hours of stage 2, plus the code (§15).

### 13.3 Recommendation

- **[Inference]** The evidence favours running K1 at seed 0, with the C2 reading, before the rest of v4's other route is written. It is the cheapest step that separates "too few steps" from "interference". And the literature's strongest lever against interference, a distance curriculum, could only be tested in a new pre-registration.
- **Confidence: low to moderate.** The closest evidence is one unreviewed 2026 preprint, at other sizes and on token sequences rather than game frames [1][2].
- **What would change it.**
  - K1 seed 0 passing would support v4's route as written.
  - K1 failing while the long delays still lag would point to a curriculum design outside v4.
  - A key overlap at nano initialisation well above the pico's 0.73 would point to key separability first.
  - A published result on a ternary or sub-megabyte memory model would outweigh everything here.

---

## 14. Limits: what the evidence does not cover

- **Size.** The smallest explicit-memory recall setups found are ARMT's 500k-parameter models [6] and the 14M-parameter DeltaNet study [7]. The frozen-base systems sit on 247M to 6B pretrained models [13][15][20][31]. Memory layers start at 134M [16]. Nothing is at 1.6M with a 64-frame episode.
- **Precision.** Every memory result above is in floating point. The ternary evidence is about whole networks on ordinary tasks [32][33], or about quantising the state of large models [34]. Nothing fetched covers ternary memory projections.
- **Frozen against co-trained.** No fetched source compares the two on one task at small size. The frozen-base evidence concerns large pretrained bases.
- **The key preprint** describes itself as preliminary results [2]. Its numbers were read through the authors' README, a digest and extracts of the paper, not its full tables [1][2][3]. Its tasks are token sequences up to 512, not frames of a game.
- **Emergence timing.** The theory gives bounds and power laws in toy settings [28][29]. It predicts nothing about how many steps our nano model needs.
- **Sleep.** The replay-consolidation literature measures forgetting across tasks [22][23], not recall within an episode, so it does not bear on H1.
- **Not verified:** the MatMul-free claim about ternary attention (section 12); Musat et al.'s exact dependence on context length (only "a tight bound in terms of the context length" was read) [28]; the full-text plateau results of Zucchet et al., read at abstract level [29].

---

## 15. Outcome (added on 19/11/2026, after K1's reading)

This section was added after the review; sections 0 to 14 are unchanged. The numbers come from `runs/s6-sleep4/stage1/k1.json`, read in `docs/s6-sleep4-prereg.md` §16. The project's lead ran K1 on both seeds, as amendment 2 of the pre-registration decided, not on seed 0 alone as section 13.2 proposed, and did not run the key-overlap reading.

**What K1 read.** v3's recipe with everything trained, 1,310 steps, seeds 0 and 1, read on each step's own training batch before its update (not a held-out test). Over the last 100 `recall` steps the answer was right on 0.8442 of the questions at seed 0 (1,019 `recall` steps; S90 never reached) and 1.0000 at seed 1 (1,000 steps; S90 at `recall` step 465, S95 at 516). The bar was 0.95 on both seeds, so K1 failed and v4 stopped there. Seed 1 sat at about 0.56 to 0.60 for about 150 `recall` steps, then jumped to 0.99 between steps 401 and 500. Seed 0 sat at about 0.57 to 0.61 from step 451 to 750, then rose to 0.85 in its window 951-1000 and 0.88 over its last 19 steps, still rising when the budget ended. On both plateaus the short delays (1 to 4 frames) were at 0.73 to 0.80 and the long ones (17 to 32) at 0.36 to 0.40. With the memory off, the same weights stayed at about 0.25 throughout on both seeds.

**Against the review's predictions.**

- **Plateau, then an abrupt transition with a seed-dependent time (section 0, section 13.1, second row; section 11): matches.** Both seeds show a long plateau and then a fast rise, at different times. This is the pattern the review named as the closest match. It does not say what sets the time; section 11 already said that nothing fetched predicts it for this model.
- **Lock-in that depends on the seed (section 0; the first row of section 13.1, from [2]): consistent, not measured.** One seed locked in within the budget and one had not finished. Two runs cannot estimate a share of seeds.
- **Section 13.2's two cases for seed 0: neither holds as written.** Seed 0 did not reach the bar and did not cross S90, so the first case ("budget and transition timing") is not shown on it. It did not "stay near v3's level" either (v3's seed 0 was at 0.41 over the last 50 `recall` steps of its phase A; K1's seed 0 ended at 0.85 to 0.88), so the second case ("interference gains weight") does not hold as written. Its long delays still lag (0.68 to 0.73 against 0.95 to 0.96 at 1 to 4 frames in its last windows).
- **Section 13.3, "K1 failing while the long delays still lag would point to a curriculum design outside v4": the condition is met on seed 0, and seed 1 qualifies it.** On seed 1 the same recipe, with no curriculum, brought the longest delays to 1.000 on its training batches. **[Inference]** At this size and budget, the lag of the long delays on seed 0 looks at least in part like a later transition, not a wall that only a curriculum breaks. One seed does not settle it, and a curriculum was not tested.
- **The frozen base (section 13.1, fourth row; section 12):** K3 had already failed. With everything trained on the same batches, both seeds left chance, which fits the reading that the frozen base was the obstacle there. K1 does not test the review's explanation of why (representations that do not separate doors and colours).

**What this does not change.** Nothing here is about a held-out test, sleep (H1) or the games (H2), and no model comes out of K1. v4 is closed. Any further attempt, such as longer training on both seeds or a delay curriculum, would be a new pre-registration, and none is decided.

---

## Sources

1. Boesch, Wee. *Anatomy of Associative Recall in Fixed-State Recurrences: A Matched-State Decomposition, an Interference Wall, and a Curriculum That Breaks It* (2026). https://arxiv.org/abs/2609.16183
2. JIBSIL/dualgoose, code and summary of the Anatomy preprint and its companion paper (GitHub README). https://github.com/JIBSIL/dualgoose
3. AI Paper Digest, 16 September 2026 (entry on the Anatomy preprint). https://amarcu.github.io/ai-paper-digest/archive/2026-09-16.html
4. Okpekpe, Orvieto. *Revisiting associative recall in modern recurrent models* (2025). https://arxiv.org/abs/2508.19029
5. Arora et al. *Zoology: Measuring and Improving Recall in Efficient Language Models* (2023). https://arxiv.org/abs/2312.04927
6. Rodkin et al. *Associative Recurrent Memory Transformer* (2024). https://arxiv.org/abs/2407.04841
7. Larson. *Constant-Memory Recall: Learned Associations in a Fixed Matrix State* (2026). https://arxiv.org/abs/2610.00232
8. Ba, Hinton, Mnih, Leibo, Ionescu. *Using Fast Weights to Attend to the Recent Past* (2016). https://arxiv.org/abs/1610.06258
9. Bietti, Cabannes, Bouchacourt, Jegou, Bottou. *Birth of a Transformer: A Memory Viewpoint* (2023). https://arxiv.org/abs/2306.00802
10. Cabannes, Dohmatob, Bietti. *Scaling Laws for Associative Memories* (2023). https://arxiv.org/abs/2310.02984
11. Ramsauer et al. *Hopfield Networks is All You Need* (2020). https://arxiv.org/abs/2008.02217
12. Miller et al. *Key-Value Memory Networks for Directly Reading Documents* (2016). https://arxiv.org/abs/1606.03126
13. Khandelwal, Levy, Jurafsky, Zettlemoyer, Lewis. *Generalization through Memorization: Nearest Neighbor Language Models* (2020). https://arxiv.org/abs/1911.00172
14. Wu, Rabe, Hutchins, Szegedy. *Memorizing Transformers* (2022). https://arxiv.org/abs/2203.08913
15. Wang et al. *Augmenting Language Models with Long-Term Memory* (LongMem, 2023). https://arxiv.org/abs/2306.07174
16. Berges et al. *Memory Layers at Scale* (2024). https://arxiv.org/abs/2412.09764
17. Behrouz, Zhong, Mirrokni. *Titans: Learning to Memorize at Test Time* (2025). https://arxiv.org/abs/2501.00663
18. Sun et al. *Learning to (Learn at Test Time): RNNs with Expressive Hidden States* (2024). https://arxiv.org/abs/2407.04620
19. Lee, McLeish, Goldstein, Fanti. *Do Language Models Need Sleep? Offline Recurrence for Improved Online Inference* (2026). https://arxiv.org/abs/2605.26099
20. Das et al. *Larimar: Large Language Models with Episodic Memory Control* (2024). https://arxiv.org/abs/2403.11901
21. Kumaran, Hassabis, McClelland. *What Learning Systems do Intelligent Agents Need? Complementary Learning Systems Theory Updated* (2016). https://pubmed.ncbi.nlm.nih.gov/27315762/
22. Tadros, Krishnan, Ramyaa, Bazhenov. *Sleep-like unsupervised replay reduces catastrophic forgetting in artificial neural networks* (2022). https://pmc.ncbi.nlm.nih.gov/articles/PMC9755223/
23. Bazhenov, Delanois, Krishnan. *Not Just After One: Sleep-Inspired Replay Prevents Catastrophic Forgetting After Sequential Tasks* (2026). https://arxiv.org/abs/2606.08447
24. Cui. *A Hippocampus for Linear Attention: An Exact Memory for What the Recurrent State Forgets* (2026). https://arxiv.org/abs/2607.02303
25. Olsson et al. *In-context Learning and Induction Heads* (2022). https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html
26. Singh, Moskovitz, Hill, Chan, Saxe. *What needs to go right for an induction head? A mechanistic study of in-context learning circuits and their formation* (2024). https://arxiv.org/abs/2404.07129
27. Reddy. *The mechanistic basis of data dependence and abrupt learning in an in-context classification task* (2023). https://arxiv.org/abs/2312.03002
28. Musat et al. *On the Emergence of Induction Heads for In-Context Learning* (2025). https://arxiv.org/abs/2511.01033
29. Zucchet, d'Angelo, Lampinen, Chan. *The emergence of sparse attention: impact of data distribution and benefits of repetition* (2025). https://arxiv.org/abs/2505.17863
30. Power, Burda, Edwards, Babuschkin, Misra. *Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets* (2022). https://arxiv.org/abs/2201.02177
31. Wind. *Prometheus Mind: Retrofitting Memory to Frozen Language Models* (2026). https://arxiv.org/abs/2601.15324
32. Nielsen, Schneider-Kamp. *BitNet b1.58 Reloaded: State-of-the-art Performance Also on Smaller Networks* (2024). https://arxiv.org/abs/2407.09527
33. Nielsen, Galke, Schneider-Kamp. *When are 1.58 bits enough? A Bottom-up Exploration of BitNet Quantization* (2024). https://arxiv.org/abs/2411.05882
34. *STEPQuant: When and Where Errors Matter in Delta-Rule Recurrent State Quantization* (2026). https://arxiv.org/abs/2609.38169
35. Zhu et al. *Scalable MatMul-free Language Modeling* (2024). https://arxiv.org/abs/2406.02528
