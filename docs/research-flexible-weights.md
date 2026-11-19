# Flexible weights for tiny ternary policies: what the literature says about TODO §7.1 to §7.4

*Note added 19/11/2026: "TODO" and `TODO.md` (item 6 and §7, its table of the post's claims) refer to the project's former to-do file, `TODO.md`, removed on 19/11/2026; its last version is at commit `0e45b7b`. The text below is unchanged.*

Research date: 19 November 2026. A dossier of 85 fetched sources, 67 of them listed at the end; the primary post (source 1) and the open-weight re-run (source 5) were read in full. Each fact gives its source number from the list at the end. Text marked **[Inference]** is my own reasoning or estimate, not a claim from a source. "Not verified" means I looked and could not confirm it. This file records research only: what follows from it for the project is decided elsewhere, and any experiment gets its own dated pre-registration first (`docs/weights-prereg.md` §9).

The project facts used here come from `TODO.md` (item 6 and §7) and `docs/weights-prereg.md` (frozen 09/11/2026): §7 plans four experiments, none started; the pre-registration chose s6-nano as the base because the float checkpoint of s3-pico no longer exists, caps GLM-5.3-flash at 400 requests for all of §7, and puts 7.1 and 7.3 in the arena on the CPU; item 6 records that s6-nano plays at the oracle's score in every game on its gate's 30 seeds and estimates 5 to 6 h of GPU for 7.2 and 7.4.

---

## 0. Summary

- **The source.** "Sparks of Weight-Space Programming: LLMs can natively read & write neural network weights", by Rikiya Takehi, Jayden Teoh, Alex Zhang and Omar Khattab (MIT CSAIL), is a **blog post** published on 2 November 2026, not a peer-reviewed paper, with one LLM (GPT-6 Astra) [1]. Writing small policies reproduces with Astra [3] and partly with Claude Opus 5.5 [4]. A benchmark, WeightBench, followed in November [12].
- **The project's LLM has already been measured on the post's tasks.** A third-party re-run gives GLM-5.3-Flash 1/4 on naming a policy's game, 0/6 on reading a transformer's middle blocks, "no edits block" on the car repair, at or below random on writing MinAtar policies (0.0 on Breakout) and an all-zero Go network [5].
- **7.1 targeted editing:** contradicted for flash-class open-weight models in one shot [5][6]; with tools, an open model repaired the car in 4 of 10 runs, but a greedy search with no LLM made the same kind of edit [6].
- **7.2 one-shot initialisation:** contradicted. GLM-5.3-Flash cannot write a working policy [5], and on 20 unpublished games "a single standard PPO update removes most of the fixed actor's success" of a written network [17].
- **7.3 online low-rank adapter:** untested at our scale, and it needs no LLM. The nearest literature warns of error accumulation and forgetting under online updates [41][42][47]; the conformal guarantee assumes exchangeable data [56][57].
- **7.4 weight audit:** weight statistics detect planted trojans in small models without an LLM [60][61]. Frontier LLMs detect that a model is poisoned but "never found the trigger" [3]. GLM-5.3-Flash was not tested on an audit.
- **[Inference]** The cheapest experiment informative either way is 7.3 in the native arena, on a base with a measured gap, with no LLM and no GPU; with s6-nano as the base it may say nothing, because s6-nano is already at the oracle's score. The evidence supports closing 7.1, 7.2 and 7.4 and running 7.3 only if a dev-seed check finds a gap (section 6).

---

## 1. The MIT post, its replications and its critics

### 1.1 What the post measured

| Experiment | Network | Given to the LLM | Result | Source |
|---|---|---|---|---|
| 1. Which game? | MLP 500 → 32 → 32 → 5, 17,253 weights, PPO on four grid games | all weights as text, no tools | game named, Breakout at 99% | [1] |
| 2. Which shape? | three DDPM MLPs, 23,824 weights | weights only, no forward pass | 3/3 shapes, drawn as outlines | [1] |
| 3. Which language? | character transformers, 7.16M weights, middle four blocks only | Python, no forward pass | 3/3 languages and every copying head | [1] |
| 4. Fix the car | Kinetix actor, 171,664 weights | weights, scene, one episode, no tools | 1,024/1,024 episodes, 3.5% of weights changed | [1] |
| 5. Write MinAtar | 600 → 32 → 32 → 6, from zero | rules and source code, no tools, never played | Breakout 517.4, Seaquest 53.1, Asterix 53.2, against PPO 500M steps 25.3 / 88.9 / 55.2 | [1] |
| 6. Write Go | 243 → 512 × 6 → 82, 1,480,274 weights | Python sandbox, 0 to 50 forward passes | with 50 passes, beats the equal-size PPO network | [1] |
| 7. Sycophancy | two Gemma 3 270M copies | weights only, then an attention tool | weights only: "unable to tell" in all four attempts; with the tool, 3/3 pairs | [1] |
| 8. Qwen maths | Qwen3.5-0.8B | Python kernel, 10/25/50 forward passes | one direction removed from 64 columns in two layers (131,072 weights, 0.016%): MATH-500 7.9% → 48.1% at 50 passes, little change at 10 or 25 | [1] |
| Extra | Gemma 3 270M, which rarely loops | same | no edit moved MATH-500 by more than about a point | [1] |

The authors' own reading of the extra experiment: "Editing weights could remove an obstacle that hid an ability, but it could not add an ability the model lacked" [1]. An addendum of 12 November says the work "is not a method" but "the name of an emerging concept" [1]. The logs are public [2].

### 1.2 Replications

- **Alan Turing Institute (three-day working note).** Astra's Breakout networks scored 514, 550 and 307. Auditing from weights alone is "possible but shallow". Their conclusion: the capability "currently amounts to compiling hand-designed programs and reading the layout of the input layer". The cost of running a frontier LLM to build weights "might well exceed the cost to train from scratch" at the sizes explored [3].
- **Claude Opus 5.5, same prompts.** 4/4 on game reading; car 1,024/1,024 with 2.15% of weights changed; Breakout 57.4, Seaquest 8.6, Asterix 72.0; sycophancy 7/12 from weights only [4].
- **Open-weight models (3 November).** On the post's experiments 1 to 7 [5]:

| Task | DeepSeek V4.1 Flash | GLM-5.3-Flash | Source |
|---|---|---|---|
| 1. Which game (of 4) | 3/4 | 1/4 | [5] |
| 2. Which shape (of 3) | 1/3 | 1/3 | [5] |
| 3. Middle-block language and heads | 0/6 | 0/6 | [5] |
| 4. Fix the car | 386 weights changed, 0/1024, worse than the base | "no edits block" | [5] |
| 5. Breakout from zero | 45.9 at the highest effort, 8.3 at high | 0.0, "at or below random on all three" games | [5] |
| 6. Go from zero, win rate vs three opponents | 0.083 / 0.054 / 0.095 | 0.021 / 0.018 / 0.020, an all-zero network (random policy 0.018 / 0.014 / 0.022) | [5] |
| 7. Sycophancy, weights only | 0/4 | 0/4 | [5] |

  The author reads the failures as "mostly format compliance, not confident wrong answers": DeepSeek and GLM "write bare shell commands when the prompt asked for a ```python block" [5].
- **Open models with tools (car task).** Strict one-shot: 0/1024 in 7 runs across 7 model IDs. With tools, a simulator, feedback and an 8,600-weight cap, DeepSeek-V4.1-Flash reached 1024/1024 in 4 of 10 runs, at about $0.14 per attempt. Random search without a model: 0 of 4 seeds; greedy unit-rewire search: 1 of 6 seeds at 1,023/1024. "This does not show that language models are uniquely good at the repair" [6].

### 1.3 Criticism

- The car edit is "25.7 times the magnitude of the original weights", with a largest new bias of 876 against an original maximum of 0.094; the re-run calls the method "a debugger for named pathology" [7].
- Raw-weight reading with no tools "only appears in experiments 1, 2 and 4, on networks of 17K–172K weights"; every language-model experiment uses Python [8]. Trade press and a sceptical blog report the same objections [9][10].
- On image CNNs of about 125,000 parameters, the reading of Claude Opus 5.5 is "better described as local arithmetic plus dataset priors than as weight-flow tracing" [11].
- WeightBench (42 tasks, 2K-parameter control policies to 7B models): frontier models "can write working control policies without ever running them, but cannot yet patch language models"; macro scores Claude Opus 5.5 0.535, DeepSeek v4-flash 0.215; "the agents largely failed to execute targeted behavioral fixes without collateral damage" [12].
- On clinical tanh networks of up to 38,401 weights, "Black-box access performed as well as white-box access", and "A well-designed script matched the comparator models": "what mattered was computation and elicitation, not access to the weights" [13].

### 1.4 Follow-ups on tiny networks

- A 253-parameter tractor steering network: LLMs with simulated feedback repaired it and wrote a working one from zero, where an evolution strategy from zero found none; in one repair an LLM set 227 weights to zero and wrote its own controller into the network [15].
- LLM-written policies checked by probabilistic model checking; the most common error is a weight matrix with a wrong number of rows [16].
- 20 unpublished games, a fixed 57k-parameter actor: written weights solve 33–56% of held-out-level episodes with feedback, the same writers' Python players 97–100%; written architectures "fail when trained from scratch, and a single standard PPO update removes most of the fixed actor's success" [17]. The author's hypothesis: "the writer first derives a symbolic solution and then expresses it in weights" [17].
- Hand-set tiny transformers: 959 weight-space programs over 13 tasks [18].

### 1.5 Not verified

- Whether the TODO's figure "0 sur 4" for sycophancy from weights alone is the post's: the post says Astra was "unable to tell" in all four attempts [1]; the open re-run scores Astra 0/4 there [5]. Consistent, but the post gives no score of its own.
- The open-weight table rests on one person's runs, one run per cell, with their own harness [5].
- Which two LLMs the tractor study used: not in the passages I read [15].

---

## 2. Weight-space learning in general

- A 2026 survey sorts the field into understanding (geometry and symmetries), representation (embeddings of weights) and generation (hypernetworks and generative models of weights) [19].
- Models that take weights as input: permutation-equivariant universal neural functionals [20]; probing-based "tree experts" [21]; a fine-tuned Qwen 2.5 3B that reads raw weights (Weight Oracles, below) [14].
- Weight generators: parameter diffusion [22], diffusion over RL policy parameters [23][24], diffusion over humanoid policy weights [25], and Text-to-LoRA, a hypernetwork that writes a LoRA adapter from a task description in one step [26].
- **[Inference]** All of these are trained for their purpose on collections of networks. None reads a ternary network, and none is an LLM writing weights without training, which is what §7 assumes.

---

## 3. Model editing and repair (7.1)

- **Forgetting under edits.** ROME and MEMIT degrade first gradually and then catastrophically under sequential editing; past that point "model editing becomes ineffective and the model is almost unusable" [28][29][30]. Edits "act as targeted suppression mechanisms" rather than removing facts [31].
- **Quantised models.** On 4-bit models, "a surgical rank-one update may be smaller than the quantization step size, causing it to be rounded to zero" [32]. **[Inference]** A ternary grid is coarser still, so 7.1's plan of editing the float checkpoint and re-exporting with the existing calibration may lose small edits; the pre-registration already counts the ternary weights that really changed.
- **Small networks before LLMs.** Editable Neural Networks train for editability with a locality term [33]. Provable repair turns a one-layer repair into linear programming, while general single-layer repair is NP-hard [34]. NNRepLayer repairs a layer of a real prosthesis controller with guarantees on the discovered unsafe samples [35]. SpecRepair penalises counter-examples and certifies the result [36].
- **Agents that interpret and repair.** An Oxford research agenda for agents that interpret a model and fix its errors [37]; agent loops for feature discovery [38]; RL agents for circuit discovery [39]; Sparse Feature Circuits, whose SHIFT edit ablates SAE features [40]. These concern language models with interpretability tools, not an LLM reading raw weights.
- **Not verified.** Any LLM edit of a ternary or binary network. Any open-weight model editing a game policy with a positive result, apart from the tool-assisted car repair [6].

---

## 4. Initialisation (7.2)

- Warm-starting from earlier weights trains faster but generalises worse, and the gap grows as data arrive [27].
- Written networks behave like programs, not like trained networks: one PPO update removes most of a written actor's success [17].
- The open-weight models tested write MinAtar policies at or near zero, and GLM-5.3-Flash below random [5].
- Cost: "the cost to run a frontier LLM as an agent to build model weights might well exceed the cost to train from scratch", given the small sizes explored [3].
- **[Inference]** 7.2 measures training steps saved from a GLM-written decision head. Both halves of its premise are contradicted: the planned LLM does not write a working head, and a written head is not a good starting point for gradient training. The test also costs GPU time (nine runs of 6,400 steps, per the pre-registration).
- **Not verified.** Any study of an LLM-written initialisation followed by supervised imitation, the training 7.2 would use; source [17] uses PPO.

---

## 5. Online adaptation and audits (7.3 and 7.4)

### 5.1 Online low-rank adaptation (7.3)

- Continual test-time adaptation suffers "error accumulation and catastrophic forgetting" from noisy pseudo-labels [41]; later methods still report forgetting across domain shifts [42]. LAME adapts only the classifier's output and, over many conditions, improves on both the non-adaptive baseline and parameter-updating methods [43]; under milder temporal correlation it can fall below the baseline [44].
- LoRA "better maintains the base model's performance on tasks outside the target domain" than full fine-tuning [45], yet introduces "intruder dimensions" that cause forgetting in continual training [46]. Fine-tuning an RL policy forgets on "the state subspace of the downstream task not visited in the initial phase of fine-tuning" [47].
- Learning from interventions: HG-DAgger trains a novice from an expert who takes over in unsafe states [48]; FIRE relabels only the states predicted to fail [49].
- Adapters over ternary or quantised bases exist, for LLMs: a ternary multiplicative adapter merged into a ternary model [50], LoRA over frozen BitNet b1.58 on edge GPUs [51], sign-only adapters for on-device fine-tuning [52].
- Training in the browser: ONNX Runtime Web supports "only single-threaded training on the CPU" [53]; a small transformer trains in the browser for teaching [54]; training under 256KB of memory on microcontrollers [55].
- Guarantees: conformal risk control assumes exchangeable data [56]; non-exchangeable conformal risk control [57] and adaptive conformal inference [58] extend guarantees under drift. The pre-registration already says its guardrail is measured, not guaranteed, because an adapter learning on a stream breaks exchangeability.
- **[Inference]** The collapse results concern unlabelled streams with pseudo-labels; in 7.3 the label comes from the veto, which is closer to learning from interventions. No source tests that combination, on any model size.

### 5.2 Weight audit (7.4)

| Detector | Uses an LLM? | Setting | Result | Source |
|---|---|---|---|---|
| Final-layer row statistics | no | TrojAI and others | 98% of trojaned networks detected, under 4% false positives | [60] |
| Linear classifier on normalised weights | no | trojan competitions | "works remarkably well across many of the existing datasets" | [61] |
| MNTD meta-classifier | no | image classifiers | about 90% AUC against an adaptive attacker | [62] |
| Spectral statistics of LoRA adapters | no | LLM adapters | poisoned adapters flagged without running the model | [63] |
| Weight Oracles (fine-tuned Qwen 2.5 3B) | yes, trained | small transformers | AUROC 0.93 on attention-routed backdoors, 0.81 on a diversified threat set | [14] |
| GPT-6 Astra, plain-text weights | yes | small poisoned LMs and adapters | 10/10, 7/10 and 23/32 detections, "but it never found the trigger"; Qwen3.8-27B succeeded on the easiest task | [3] |
| Frontier agents vs Lazarus | yes | fewer than 10 reprogrammed neurons | anomalies spotted when asked; the strongest signal is a diff against the declared base | [64][65] |

- TrojAI generated 19,264 models, about half with trojans; its RL and cyber models are "just a few small Multi-Layer Perceptron (MLP) layers" [59].
- In an auditing game, teams with weights and data found a hidden objective and the API-only team did not [66]; a presence-based frontier auditor could not tell a topology-conditioned organism from its base [67].
- **[Inference]** For 7.4, the clean published s6-secrets-pico is a reference model, and a weight diff against it is the signal the sources rank highest [64][65]; the LLM adds nothing that the diff does not give, and the pre-registration already says four models allow no statistical conclusion.
- **Not verified.** The TrojAI final report's statement that weight analysis loses effectiveness on large architectures: it appeared in search summaries, not in the passages I read [59]. MNTD's 97% average AUC: abstract summary only [62].

---

## 6. The three answers

### 6.1 Supported, contradicted or untested at our scale (0.4M to 5.6M ternary weights)

| Item | Verdict | Strongest evidence |
|---|---|---|
| 7.1 targeted editing without forgetting | Contradicted for GLM-5.3-Flash in one shot; untested with a tool loop; no LLM-specific advantage over search shown | [5][6][12][32] |
| 7.2 one-shot initialisation | Contradicted | [5][17][3] |
| 7.3 flexible weights in real time | Untested; no LLM involved | [41][43][47][56] |
| 7.4 weight audit | Supported for non-LLM detectors; for LLMs, presence only and with frontier models; untested for GLM-5.3-Flash | [60][61][3][13] |

**What the evidence does not cover.** No source tests any LLM reading or editing a ternary network or one with weights in {−1, 0, +1}; the post's networks are float MLPs and transformers [1], and the ternary adapters are for LLMs [50][51]. No source tests an online adapter whose labels come from a safety veto. No source measures GLM-5.3-Flash with a harness that fixes its output format, which [5] names as its main failure. Most results are single runs [3][5][15], several are blog posts or working notes [1][3][5], and the newest preprints are days old [12][15][16].

### 6.2 The cheapest experiment informative either way

**[Inference]** **7.3, in the native arena, on a base with a measured gap.**
- *Why this one.* It does not depend on the capability gap that predicts the other three, and neither outcome is in the literature: a gain without loss on the other games would be the first such result on a tiny ternary policy; a loss would be a measured case of the online-update forgetting [47] at this size.
- *Cost on one laptop.* No LLM call and no GPU. Because the base stays frozen, the published ternary file is enough, so the lost float checkpoint of s3-pico does not matter. About one to two days to write and test the rank-4 logit adapter, its per-veto SGD step and the guardrail check in the arena; then minutes of single-core CPU for the three arms, the learning-rate choice on dev streams and the 300-episode test streams.
- *The precondition.* The pre-registered verdict asks for a 2% gain on the target game. s6-nano is already at the oracle's score on its gate's seeds, so with s6-nano as the base the experiment may be unable to pass and its null would say nothing. A dev-seed arena pass that counts vetoes and the gap to the oracle per game, for s6-nano and for s3-pico (whose TODO entry names Dodge vetoes and Flappy crashes), costs seconds and settles this before any code is written; changing the base needs a dated amendment.

**[Inference]** *The cheapest LLM experiment, for comparison:* a 7.1 probe on a deliberately damaged copy of a policy (a planted, known fault), GLM-5.3-Flash in a tool loop against a greedy search with the same evaluation budget, a few dozen of the 400 pre-registered requests, CPU only. Its outcome is largely predicted by [5] and [6], so it would teach little either way.

### 6.3 Recommendation

**[Inference]** On the evidence, close 7.1, 7.2 and 7.4: their outcome with GLM-5.3-Flash is predicted by direct measurements of that model on the same tasks [5], their positive versions depend on frontier models [3][4], and 7.2 and 7.4 cost GPU hours for results the sources already bound [17][59]. For 7.3, run the dev-seed gap check; if a game shows a gap, pre-register and run 7.3 alone in the arena; if none does, nothing informative is left in §7. Confidence: about 75% that dropping the three LLM experiments is right; about even odds that 7.3 shows a gain, given a gap.

**What would change this.** A measured result of GLM-5.3-Flash, or of another open-weight model the project may use, repairing or writing small policies above a matched non-LLM search (a WeightBench entry, for instance [12]); a published success of LLM weight editing on a quantised or ternary network; for 7.3, a gap check that finds no game where the veto still fires and the model trails the oracle.

---

## 7. Corrections to the text of §7 and of the pre-registration

- The pre-registration says the post was read through secondary articles and is not reproduced. It has now been read at the source [1], and its writing and reading results are partly reproduced [3][4].
- The TODO's table says the post shows a model "débarrassé de ses boucles" [rid of its loops] (Qwen3.5-0.8B) and no gain on Gemma 3 270M; the source agrees [1]. It also says Breakout 517 against 25 for PPO after 500M steps; the source agrees (517.4 and 25.3) [1].
- The TODO says no one has published this on ternary models. I found nothing that contradicts it.

---

## Sources

1. Takehi, Teoh, Zhang, Khattab. Sparks of Weight-Space Programming: LLMs can natively read & write neural network weights (blog post, MIT CSAIL, 2 Nov 2026). https://rikiyat.github.io/blog/weight-space-programming/
2. RikiyaT/lm_understand-logs: complete logs of the experiments. https://github.com/RikiyaT/lm_understand-logs
3. Goworek, Hickey, Nanni, O'Reilly. Replicating and Extending Weight-Space Programming: Frontier LLMs Compile Programs into Weights and Read Input Layers (Alan Turing Institute). https://arxiv.org/html/2611.21449
4. Replication of Sparks of Weight-Space Programming with Claude models. https://github.com/Sir-Olexander/lm_understand-logs-claude
5. Weight Space Programming with Open Weights Models (3 Nov 2026). https://blog.sayan.page/wsp-with-open-weights-models/
6. Open-model replication of Sparks of Weight-Space Programming. https://github.com/nbirkner/weight-space-programming-open
7. Sparks of Weight Space Programming: What Astra Actually Did Was Debug Weights. https://johal.in/sparks-weight-space-programming-debugger
8. Honnibal. The 'Astra can read and write weights to improve Qwen' post is misleading. https://honnibal.dev/blog/astra-read-write-misleading
9. MIT team says GPT-6 Astra can improve other AI models by editing their weights. Critics say its biggest result relied on standard ML engineering (R&D World). https://www.rdworldonline.com/mit-team-says-gpt-6-astra-can-fix-an-ai-model-by-editing-its-weights-critics-say-the-method-was-standard-ml-engineering/
10. A Skeptical Look at Weight-Space Programming. https://brainwagon.org/blog/2026-11-08-a-skeptical-look-at-weight-space-programming.html
11. Ahmed. Can an LLM Actually Reason and Program about Neural Network's Weights or is it just memorization on steroids? https://arxiv.org/html/2611.08070v1
12. Marasini, Khadka, Rahimi. WeightBench: Evaluating Language Models as Weight-Space Programmers. https://arxiv.org/html/2611.18553v1
13. Liu, Yang. Computation, not weight access: large language models as auditors of clinical prediction networks. https://arxiv.org/html/2611.07847
14. Kabra, Venhoff, Schroeder de Witt. Weight Oracles: Reading Neural Network Weights with Language Models. https://arxiv.org/html/2610.07334v1
15. Oksanen. Weight-space programming for tractor guidance controller. https://arxiv.org/html/2611.21958
16. Gross. Weight-Space Programming of Policies and Their Verification via Model Checking. https://arxiv.org/html/2611.11604
17. Lehmann. Compiling Known Rules into Compact Neural Agents: Weight-Space Programming on Unseen Games. https://arxiv.org/html/2611.16981
18. Singh. Weight-Space Programming from scratch on simple tasks. https://csinva.io/weight-space-programming-autoresearch/analyze/toy_examples/report.html
19. A Survey of Weight Space Learning: Understanding, Representation, and Generation. https://arxiv.org/html/2603.10090v1
20. Universal Neural Functionals. https://arxiv.org/html/2402.05232
21. Learning on Model Weights using Tree Experts. https://arxiv.org/html/2410.13569v1
22. Neural Network Parameter Diffusion (p-diff). https://github.com/NUS-HPC-AI-Lab/Neural-Network-Parameter-Diffusion
23. Policy Network Diffusion. https://arxiv.org/html/2611.04614
24. Make-An-Agent: A Generalizable Policy Network Generator with Behavior-Prompted Diffusion. https://arxiv.org/abs/2407.10973
25. Modeling and Understanding Neural Weights of Humanoid Policies with Diffusion. https://arxiv.org/html/2610.20606v1
26. Text-to-LoRA: Instant Transformer Adaption. https://arxiv.org/abs/2506.06105
27. Ash, Adams. On Warm-Starting Neural Network Training. https://arxiv.org/pdf/1910.08475
28. Gupta et al. Model Editing at Scale leads to Gradual and Catastrophic Forgetting. https://arxiv.org/html/2401.07453v3
29. Meng et al. Mass-Editing Memory in a Transformer (MEMIT). https://arxiv.org/pdf/2210.07229
30. Rebuilding ROME: Resolving Model Collapse during Sequential Model Editing. https://arxiv.org/pdf/2403.07175
31. Exposing the Illusion of Erasure in Knowledge Editing for LLMs. https://arxiv.org/html/2606.23276v3
32. CACHE-UK: A Stability-Aware Memory Editor for Sequentially Updated Quantized LLMs in Finance. https://arxiv.org/html/2607.28292
33. Sinitsin et al. Editable Neural Networks. https://arxiv.org/html/2004.00345v2
34. Sotoudeh, Thakur. Provable Repair of Deep Neural Networks. https://arxiv.org/html/2104.04413v1
35. Safe Robot Learning in Assistive Devices through Neural Network Repair. https://arxiv.org/pdf/2303.04431
36. SpecRepair: Counter-Example Guided Safety Repair of Deep Neural Networks. https://arxiv.org/html/2106.01917
37. Automated Interpretability-Driven Model Auditing and Control: A Research Agenda (Oxford AIGI). https://aigi.ox.ac.uk/wp-content/uploads/2026/01/Automated_interp_Research_Agenda.pdf
38. Automated Interpretability and Feature Discovery in Language Models with Agents. https://www.alphaxiv.org/abs/2605.01555
39. MechRL: Reinforcement Learning Agents Perform Circuit Discovery for Mechanistic Interpretability. https://arxiv.org/html/2605.26343v2
40. Marks et al. Sparse Feature Circuits. https://arxiv.org/pdf/2403.19647
41. ViDA: Homeostatic Visual Domain Adapter for Continual Test Time Adaptation. https://arxiv.org/pdf/2306.04344
42. SloMo-Fast: Slow-Momentum and Fast-Adaptive Teachers for Source-Free Continual Test-Time Adaptation. https://arxiv.org/pdf/2511.18468
43. Boudiaf et al. Parameter-free Online Test-time Adaptation (LAME). https://arxiv.org/html/2201.05718v2
44. Distribution Alignment for Fully Test-Time Adaptation with Dynamic Online Data Streams. https://arxiv.org/pdf/2407.12128
45. Biderman et al. LoRA Learns Less and Forgets Less. https://arxiv.org/html/2405.09673v2
46. Shuttleworth et al. LoRA vs Full Fine-tuning: An Illusion of Equivalence. https://arxiv.org/pdf/2410.21228
47. Wołczyk et al. Fine-tuning Reinforcement Learning Models is Secretly a Forgetting Mitigation Problem. https://arxiv.org/abs/2402.02868
48. HG-DAgger: Interactive Imitation Learning with Human Experts. https://arxiv.org/pdf/1810.02890
49. Fighting Failures with FIRE: Failure Identification to Reduce Expert Burden in Intervention-Based Learning. https://arxiv.org/pdf/2007.00245
50. Low-Rank Ternary Adaptation for Fine-Tuning Transformers. https://arxiv.org/html/2608.24469v1
51. LoRA Fine-Tuning BitNet b1.58 LLMs on Heterogeneous Edge GPUs via QVAC Fabric. https://huggingface.co/blog/qvac/fabric-llm-finetune-bitnet
52. Signs Beat Floats: Low-Rank Double-Binary Adaptation for On-Device Fine-Tuning. https://arxiv.org/pdf/2605.24058
53. On-Device Training: Training a model in browser (ONNX Runtime). https://opensource.microsoft.com/blog/2024/02/06/on-device-training-training-a-model-in-browser/
54. Turning Language Model Training from Black Box into a Sandbox. https://arxiv.org/pdf/2601.21631
55. On-Device Training Under 256KB Memory. https://arxiv.org/pdf/2206.15472
56. Angelopoulos et al. Conformal Risk Control. https://arxiv.org/abs/2208.02814
57. Farinhas et al. Non-Exchangeable Conformal Risk Control. https://arxiv.org/abs/2310.01262
58. Gibbs, Candès. Adaptive Conformal Inference Under Distribution Shift. https://arxiv.org/abs/2106.00170
59. Trojans in Artificial Intelligence (TrojAI) Final Report. https://arxiv.org/pdf/2602.07152
60. Fields et al. Trojan Signatures in DNN Weights. https://openaccess.thecvf.com/content/ICCV2021W/AROW/papers/Fields_Trojan_Signatures_in_DNN_Weights_ICCVW_2021_paper.pdf
61. Huster, Lin, Stefanescu. Solving Trojan Detection Competitions with Linear Weight Classification. https://arxiv.org/pdf/2411.03445
62. Xu et al. Detecting AI Trojans Using Meta Neural Analysis (MNTD). https://arxiv.org/html/1910.03137v4
63. Detecting Backdoored LoRAs from Weights Alone. https://arxiv.org/pdf/2602.15195
64. Papailiopoulos. Lazare, veni foras: Reprogramming Dormant Neurons to Backdoor AI Agents (Microsoft Research, 10 Nov 2026). https://www.microsoft.com/en-us/research/blog/lazare-veni-foras-reprogramming-dormant-neurons-to-backdoor-ai-agents/
65. Seven Neurons to Root: Lazarus Weight-Edit Backdoors in Agent Models (Cloud Security Alliance). https://labs.cloudsecurityalliance.org/research/csa-research-note-lazarus-neuron-edit-agent-backdoor-2026111/
66. Marks et al. Auditing language models for hidden objectives. https://arxiv.org/pdf/2503.10965
67. Topology-Conditioned Backdoors: Language Models That Insert Vulnerabilities When They Infer They Are in a Multi-Agent System. https://arxiv.org/html/2610.05793v1
