# Ultra-small, fast decision models: state of the art and plan for skycmd (11 Oct. 2026)

*Translated into English on 19/11/2026; the French original is this file at commit `1f180a5`.*

## Abstract / TL;DR
Four recent results combine well for a decision model under one million parameters [S60][S14][S43][S2]:
- **depth through recursion of a tiny block**: TRM, a single 2-layer network, ~7M parameters [S60]; Ouro, a shared block in a loop with a learned depth [S61];
- **early exit seen as a "readout" of the trajectory**, separate from the way the trajectory is trained [S14], with a threshold calibrated under risk control [S30];
- **numbers encoded as a single token**: FoNE [S43], FT-Transformer [S62];
- **ternary quantization through QAT, which holds even at small sizes** [S2][S1].

Imitation learning with DAgger, that is a teacher that labels the states visited by the student, remains the basis for controlling a game in closed loop [S54][S53][S47].

The plan that follows for skycmd, "s3", combines these four building blocks with the cached static stream of s2. [M]

## Background & motivation
Typed decision models are already numerous on Hugging Face [S42][S52][S56].
- **A single pass**: decider-4b reads a state and typed questions, then returns a distribution over the options in one pass, without decoding [S42].
- **The three question types**: Sev distinguishes choice, ordinal score and noul [S52].
- **Calibration**: fragment-1 calibrates one temperature per question type [S56].

These models range from 100M to several billion parameters. Our goal is the reverse: a model of a few hundred thousand parameters that decides in a few microseconds in the browser. [M]

## Key papers (chronological)
- **FT-Transformer**: each column, numerical or categorical, becomes an embedding (a token), then a Transformer and a class token make the prediction [S62].
- **Early exit under risk control**: the exit threshold is computed on calibration data so as to respect a tolerated risk level; a correction ensures control in expectation [S30].
- **BitNet b1.58 at small scales**: models from 100K to 48M parameters were studied. Adding capacity compensates for the reduced precision [S2].
- **FoNE**: a number becomes a single token through Fourier features [S43]. A related work points out that xVal must bring numbers into a bounded range [S36].
- **TRM**: a single 2-layer network, applied recursively, with ~7M parameters, reaches 45 % on ARC-AGI-1 [S60].
- **Ouro (LoopLM)**: computation is done by iterating in latent space. An entropy-regularized objective learns the depth, and the 1,4B and 2,6B models match models of up to 12B [S61].
- **LoopFormer**: a looped model conditioned on a compute budget. Its variable-depth training costs about 1,5× the FLOPs of a fixed-loop model [S10].
- **Adaptive Depth in Looped Transformers**: separates the formation of the trajectory (the training objective) from the readout that chooses the exit, and compares fixed priors with learned gates [S14].
- **Stochastic stopping**: drawing the depth at random around a value, rather than a fixed depth, stabilizes the out-of-distribution generalization of loops [S49].
- **Universal Transformers Need Memory**: weight-shared recursion, adaptive halting and a state persistent across iterations are three lines of research that converge again in 2025–2026 [S8].

## Methods & approaches compared
| Lever | What the sources say | skycmd application |
|---|---|---|
| Depth without parameters | a tiny recursive block (TRM) [S60], a shared loop (Ouro) [S61] | `state_loops`: 1 state block looped 4 times [M] |
| Early exit | the readout separates from the trajectory [S14]; threshold calibrated under risk [S30] | calibrate τ on the states visited while playing [M] |
| Numbers | one token per number (FoNE) [S43]; one token per feature (FT-Transformer) [S62] | `fuse_numbers`: "key value" becomes a single token [M] |
| Ternary | 1,58-bit QAT holds from 100K to 48M parameters [S2]; a QAT model of 2,1M parameters reaches a perplexity of 10,1 against 151,9 with post-training quantization [S1] | QAT throughout the decision phase [M] |
| Imitation | on-policy distillation: the teacher evaluates the states visited by the student [S54]; same loop in autonomous driving [S47]; BC then DAgger on Flappy Bird in Stanford's RL course [S53] | DAgger with the oracle as teacher |
| Data | decider-4b: ten programmatic families with verifiable ground truth in its mix [S42] | programmatic numerical curriculum [M] |
| WASM | the Relaxed SIMD instructions (dot product, FMA) speed up existing workloads [S7] | `relaxed_madd` behind a cfg [M] |

## Findings & consensus
- **Depth.** With few parameters, recursion of a shared block is the cheapest way to get depth [S60][S61].
- **Early exit.** The exit must be evaluated on the real trajectory: the readout rule is tuned separately from training [S14], and a threshold chosen with risk control gives guarantees that the naive threshold does not have [S30]. 
> [model-hint] Internal skycmd measurement: a τ calibrated on isolated questions (0,62) made the model fail in a real game, whereas τ = 0,95 recovered full-depth quality with ~3,3 layers out of 4.

- **Ternary.** QAT is clearly better than post-training quantization at small sizes [S1], and ternary remains viable under 1M parameters [S2].
- **Controlling a game.** DAgger corrects the distribution shift of pure cloning. It is the standard recipe taught [S53][S39], and on-policy distillation generalizes it to language models [S54][S40].

## Gaps & open problems
- No source found measures a typed decision model under 1M parameters in closed loop. That is the space skycmd occupies. [M]
- For Relaxed SIMD, the source gives no int8 figures [S7].
- Behaviour cloning in ViZDoom starts from pixels [S38]. No source deals with a compact textual state like ours. [M]

## Future directions
The ideas to try tonight, ranked by expected gain in speed and quality [M]:
1. Merge each "key value" pair into one token (`fuse_numbers`), that is about half as many state tokens [M], after [S62][S43].
2. A recursive state encoder (1 block × 4 loops) with supervision at every exit and stochastic depth [M], after [S60][S61][S49].
3. Calibrate τ in closed loop through risk control on the DAgger states [M], after [S30][S14].
4. Ternary QAT throughout the decision phase [M], after [S2][S1].
5. A programmatic numerical curriculum with exact ground truth [M], after [S42].
6. Specialized WASM kernels for M ≤ 16 and `relaxed_madd` [M], after [S7][S3].


## Implementations and biological inspiration (2nd round)
- **Ternary lookup-table kernels** [S63][S64]:
  - bitnet.cpp offers ternary lookup-table (TL) kernels and an I2_S kernel for CPU inference [S63].
  - T-MAC replaces multiplication with bit-wise table lookups, without dequantization [S64].
  - This is the answer to our measurement, where ternary addition and subtraction were slower than f32. [M]
- **Temporal re-seeding of latents** [S65][S68]:
  - RIN seeds the latents of a step from the previous step, with a zero-initialized term [S65].
  - TRM passes its latent state from one supervision pass to the next, detached from the graph [S68].
  - For high-frequency control, a VLA study reuses tokens from one step to the next [S66], and the Latent Recurrent Transformer reuses the previous hidden state as memory [S76].
- **Event-driven computation.** Delta networks threshold the change in activation and skip the updates under the threshold, at the cost of a drift that requires periodic resets [S70].
- **Expand-and-sparsify** [S71]:
  - In the fly, about 50 projection neurons feed about 2 500 Kenyon cells, and k-WTA inhibition leaves only about 5 % of units active [S71].
  - This pattern has proven effective for route following in embedded robotics [S72], and its expressivity is studied formally [S73].
- **Control circuits inspired by C. elegans.** Neural Circuit Policies are organized into sensory neurons, interneurons, recurrent command neurons and motor neurons, with few parameters [S74]. Drones piloted by these networks outperformed other networks on the hard tasks, such as chasing a target in the rain [S75].

> [model-hint] Proposed connection for skycmd, "temporal recursion": the looped state encoder of s3 starts from the final latent of the previous frame, through a zero-initialized gate as in RIN and with a detached latent as in TRM. Games change little from one frame to the next, so early exit should trigger at the first loop most of the time. This would give the savings of delta networks without their drift.

## Open questions / contradictions
> [model-hint] TRM reports its results on puzzles (ARC, Sudoku), not on real-time control. The transfer to our games remains to be measured tonight.

> [model-hint] LoopFormer reports that early exit can collapse to degenerate states. Supervision at every exit and stochastic depth are our safeguards, and this is to be checked on the closed-loop metrics.

## References (see refs.bib)
All sources are listed in `refs.bib` and `sources.json`.

## Sources

- [S1] TinyStories 1-bit LLM — https://huggingface.co/euclidstellar/tinystories-1bit-llm
- [S2] When are 1.58 bits enough? A Bottom-up Exploration of BitNet Quantization — https://arxiv.org/html/2411.05882v1
- [S3] Optimizing the Deployment of Tiny Transformers on Low-Power MCUs — https://arxiv.org/html/2404.02945v1
- [S7] WebAssembly and WebGPU for Web AI — https://developer.chrome.com/blog/io24-webassembly-webgpu-1
- [S8] Universal Transformers Need Memory: Depth-State Trade-offs in Adaptive Recursive Reasoning — https://arxiv.org/pdf/2604.21999
- [S10] LoopFormer budget-conditioned looped LM — https://arxiv.org/html/2602.11451v1
- [S14] Adaptive Depth in Looped Transformers: Diagnosing Learned Halting Gates and Trajectory Readouts — https://arxiv.org/pdf/2607.20519
- [S30] Fast yet Safe: Early-Exiting with Risk Control — https://arxiv.org/pdf/2405.20915
- [S36] Efficient numeracy in language models through single-token number embeddings — https://arxiv.org/pdf/2510.06824
- [S38] Behavioural Cloning in VizDoom — https://ar5iv.labs.arxiv.org/html/2401.03993
- [S39] An Introduction to Deep Reinforcement and Imitation Learning — https://arxiv.org/pdf/2512.08052
- [S40] On-policy distillation survey — https://arxiv.org/pdf/2604.00626
- [S42] decider-4b — https://huggingface.co/Mapika/decider-4b
- [S43] FoNE: Precise Single-Token Number Embeddings via Fourier Features — https://arxiv.org/html/2502.09741v1
- [S47] OPTED: On-Policy Fine-Tuning for End-to-End Driving — https://arxiv.org/pdf/2609.20756
- [S49] Stabilizing Extrapolation in Looped Transformers via Learned Stochastic Stopping — https://www.alphaxiv.org/abs/2606.29983v1
- [S52] Sev typed-decisions CE-only — https://huggingface.co/LakoreAI/sev
- [S53] CS224R 2026 Homework 1 — https://cs224r.stanford.edu/material/hw1/CS224R_2026_Homework_1.pdf
- [S54] On-Policy Distillation — https://www.thinkingmachines.ai/blog/on-policy-distillation/
- [S56] fragment-1 — https://huggingface.co/FrameXlabs/fragment-1
- [S60] [2510.04871] Less is More: Recursive Reasoning with Tiny Networks — https://arxiv.org/abs/2510.04871
- [S61] [2510.25741] Scaling Latent Reasoning via Looped Language Models — https://arxiv.org/abs/2510.25741
- [S62] [2106.11959] Revisiting Deep Learning Models for Tabular Data — https://arxiv.org/abs/2106.11959
- [S63] Bitnet.cpp: Efficient Edge Inference for Ternary LLMs — https://arxiv.org/abs/2502.11880
- [S64] T-MAC: CPU Renaissance via Table Lookup for Low-Bit LLM Deployment on Edge — https://arxiv.org/abs/2407.00088
- [S65] Scalable Adaptive Computation for Iterative Generation (RIN) — https://arxiv.org/abs/2212.11972
- [S66] Reducing Temporal Redundancy for Efficient Vision-Language-Action Inference — https://arxiv.org/abs/2607.12287
- [S68] Evaluating Tiny Recursive Models Across Training for Code Generation — https://arxiv.org/abs/2608.29376
- [S70] Exploiting Symmetric Temporally Sparse BPTT for Efficient RNN Training — https://arxiv.org/abs/2312.09391
- [S71] Bio-Inspired Hashing for Unsupervised Similarity Search (BioHash) — https://arxiv.org/abs/2001.04907
- [S72] Vision-based route following by an embodied insect-inspired sparse neural network — https://ar5iv.labs.arxiv.org/html/2303.08109
- [S73] Expressivity of expand-and-sparsify representations — https://arxiv.org/abs/2006.03741
- [S74] Designing Worm-inspired Neural Networks for Interpretable Robotic Control (NCP) — https://arxiv.org/abs/1809.04423v2
- [S75] MIT news: liquid neural networks drones — https://news.mit.edu/2021/cause-effect-neural-networks-1014
- [S76] Latent Recurrent Transformer — https://arxiv.org/abs/2605.26797
