---
title: Sources & credits
---

# Sources & credits

skycmd stands on open data, open libraries and open research. This page lists what the shipped models were trained on, the main libraries and technologies of the codebase, and the reference models and runtimes we compared against. Exact licences, row counts and rejected datasets are in [`DATA_LICENSES.md`](https://github.com/maxgfr/skycmd/blob/main/DATA_LICENSES.md). The research notes, with their papers, are in [`docs/research.md`](https://github.com/maxgfr/skycmd/blob/main/docs/research.md) and [`docs/research-night.md`](https://github.com/maxgfr/skycmd/blob/main/docs/research-night.md).

## Training data

### Typed decisions (choice / score / noul)
| Dataset | Licence | Used as |
|---|---|---|
| [tasksource/tasksource-jev-typed-decisions](https://huggingface.co/datasets/tasksource/tasksource-jev-typed-decisions) | per row; only rows tagged `commercial` | training |
| [tasksource/procedural-typed-decisions](https://huggingface.co/datasets/tasksource/procedural-typed-decisions) | Apache-2.0 | training |
| [tasksource/synthetic-typed-decisions](https://huggingface.co/datasets/tasksource/synthetic-typed-decisions) | Apache-2.0 | training |
| [pngwn/typed-decisions](https://huggingface.co/datasets/pngwn/typed-decisions) | CC-BY-SA-4.0 | training (soft `gold_probs`) |
| [n4ze3m/typed-decisions-synth](https://huggingface.co/datasets/n4ze3m/typed-decisions-synth) | MIT | training |
| [kgrozdanovski/assistant-decisions](https://huggingface.co/datasets/kgrozdanovski/assistant-decisions) | Apache-2.0 | training |
| [helmo/synthetic-typed-decisions](https://huggingface.co/datasets/helmo/synthetic-typed-decisions) | MIT | training |
| [m-lagnajit/calibrated-typed-decisions](https://huggingface.co/datasets/m-lagnajit/calibrated-typed-decisions) | CC-BY-4.0 | training |
| [ZefanCai/Open-Jev](https://huggingface.co/datasets/ZefanCai/Open-Jev) (drone expansion split) | CC0 | training |
| [Aditharavind/swarm-laya-decisions](https://huggingface.co/datasets/Aditharavind/swarm-laya-decisions) | Apache-2.0 | training |

### Classic datasets converted into typed questions
| Dataset | Licence | Converted to |
|---|---|---|
| [google/boolq](https://huggingface.co/datasets/google/boolq) | CC-BY-SA-3.0 | noul |
| [nyu-mll/glue](https://huggingface.co/datasets/nyu-mll/glue) (MNLI) | see the GLUE card | choice |
| [stanfordnlp/snli](https://huggingface.co/datasets/stanfordnlp/snli) | CC-BY-SA-4.0 | choice |
| [allenai/ai2_arc](https://huggingface.co/datasets/allenai/ai2_arc) | CC-BY-SA-4.0 | choice |
| [tau/commonsense_qa](https://huggingface.co/datasets/tau/commonsense_qa) | MIT | choice |
| [Rowan/hellaswag](https://huggingface.co/datasets/Rowan/hellaswag) | MIT | choice |
| [PolyAI/banking77](https://huggingface.co/datasets/PolyAI/banking77) | CC-BY-4.0 | choice (8 candidates) |
| [nvidia/HelpSteer2](https://huggingface.co/datasets/nvidia/HelpSteer2) | CC-BY-4.0 | score (5 levels) |
| [nvidia/When2Call](https://huggingface.co/datasets/nvidia/When2Call) | CC-BY-4.0 | choice |
| [google-research-datasets/paws-x](https://huggingface.co/datasets/google-research-datasets/paws-x) (fr) | see the card | noul, French |
| [tblard/allocine](https://huggingface.co/datasets/tblard/allocine) | MIT | noul, French |

### Pretraining text
| Dataset | Licence |
|---|---|
| [SimpleStories/SimpleStories](https://huggingface.co/datasets/SimpleStories/SimpleStories) | MIT |
| [roneneldan/TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) | CDLA-Sharing-1.0 |
| [btecsec/tinystories-fr](https://huggingface.co/datasets/btecsec/tinystories-fr) | CDLA-Sharing-1.0 |
| [HuggingFaceFW/fineweb-edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) (one shard) | ODC-BY-1.0 |
| [HuggingFaceFW/fineweb-2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2) (`fra_Latn`, one shard) | ODC-BY-1.0 |

### Pretraining text from fully open LLMs (s6 onwards)
OLMo 3 and LLM360 publish their training data with their weights. From s6 on, pretraining also reads the non-web parts of those corpora (`sky-data fetch-text --set open`, [`docs/s6-prereg.md`](https://github.com/maxgfr/skycmd/blob/main/docs/s6-prereg.md), amendment 1). s1 to s5 were trained without them.

| Dataset | Part used | Licence |
|---|---|---|
| [allenai/dolma3_mix-150B-1025](https://huggingface.co/datasets/allenai/dolma3_mix-150B-1025) (Dolma 3, OLMo 3) | Wikipedia EN, olmOCR science PDFs, FineMath (one shard each) | ODC-BY-1.0; Wikipedia under CC-BY-SA |
| [IFM/TxT360](https://huggingface.co/datasets/IFM/TxT360) (LLM360) | PG-19 books, StackExchange, Europarl (French only) | ODC-BY-1.0; PG-19 public domain, StackExchange CC-BY-SA |

Not used, despite open models: the Lucie Training Dataset (CC-BY-NC-SA-4.0, non-commercial), The Pile behind Pythia (no licence tag, includes Books3), and Apertus' FineWeb copies with opt-out tags (no licence declared on the Hub).

### Generated data
- **Game oracles**, written in Rust (`crates/sky-games`). They give exact labels for Flappy Drone, Rescue Run, Obstacle Dodge and the Doom-like game, plus the DAgger rounds, in which the model plays and the oracle labels the states it reaches.
- **Numeric curriculum** (`sky-train s2-numeric`): comparisons, ordering and closest values, with exact answers, in English and French.
- **GLM 5.3**: 3,000 civil drone orders and about 2,000 paraphrases of question instructions (used by s1–s3 only).

### Evaluation only (never trained on)
- [LocalLLaMA/typed-decisions](https://huggingface.co/datasets/LocalLLaMA/typed-decisions) (Apache-2.0)
- [AirsideLabs/notam-typed-decisions](https://huggingface.co/datasets/AirsideLabs/notam-typed-decisions) (CC-BY-4.0)

### s4: teacher, real cases and zero-shot held-out sets
| Dataset or model | Licence | Used as |
|---|---|---|
| [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) ([MLX 4-bit](https://huggingface.co/mlx-community/Qwen3.5-4B-4bit)) | Apache-2.0 | teacher: soft labels from the option-letter logits |
| [glaiveai/glaive-function-calling-v2](https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2) | Apache-2.0 | tool routing (choice over tools + `no_tool`) |
| [NousResearch/hermes-function-calling-v1](https://huggingface.co/datasets/NousResearch/hermes-function-calling-v1) | Apache-2.0 | tool routing |
| [bitext/Bitext-customer-support-llm-chatbot-training-dataset](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset) | CDLA-Sharing-1.0 | ticket triage with an `unsure` option |
| [google/civil_comments](https://huggingface.co/datasets/google/civil_comments) | CC0-1.0 | moderation on a written rule (noul) |
| [PolyAI/banking77](https://huggingface.co/datasets/PolyAI/banking77) | CC-BY-4.0 | **held out** from s4: zero-shot evaluation |
| [mteb/amazon_massive_intent](https://huggingface.co/datasets/mteb/amazon_massive_intent) (MASSIVE, en + fr) | CC-BY-4.0 | **held out**: zero-shot evaluation |

### s6-nano and s6-micro: test sets (not trained yet)
No teacher. They train on the decision sets, the games and the numeric curriculum above, and pretrain on the s3 text plus the open set above ([pre-registration](https://github.com/maxgfr/skycmd/blob/main/docs/s6-gen-prereg.md)).

| Dataset | Licence | Used as |
|---|---|---|
| [LocalLLaMA/typed-decisions](https://huggingface.co/datasets/LocalLLaMA/typed-decisions) (test split) | Apache-2.0 | the gated benchmark, `typed_decisions_bench` |
| [mteb/amazon_massive_intent](https://huggingface.co/datasets/mteb/amazon_massive_intent) (MASSIVE, en + fr) | CC-BY-4.0 | **held out**: zero-shot evaluation |
| [AirsideLabs/notam-typed-decisions](https://huggingface.co/datasets/AirsideLabs/notam-typed-decisions) | CC-BY-4.0 | **held out**: test only |
| [nvidia/When2Call](https://huggingface.co/datasets/nvidia/When2Call) (test split) | CC-BY-4.0 | test split of a trained source |
| [PolyAI/banking77](https://huggingface.co/datasets/PolyAI/banking77) | CC-BY-4.0 | trained on: reported as in-distribution, never as zero-shot |

### s5: backbones, teacher and held-out evaluation
| Dataset or model | Licence | Used as |
|---|---|---|
| [jhu-clsp/ettin-encoder-17m](https://huggingface.co/jhu-clsp/ettin-encoder-17m) | MIT | s5-lite backbone |
| [cross-encoder/ettin-reranker-17m-v1](https://huggingface.co/cross-encoder/ettin-reranker-17m-v1) | Apache-2.0 | alternative initialization of the backbone |
| [convaiinnovations/laya-typed-decisions](https://huggingface.co/convaiinnovations/laya-typed-decisions) | Apache-2.0 | teacher: soft labels, run locally in Rust, kept only where they agree with the gold |
| [btzsc/btzsc](https://huggingface.co/datasets/btzsc/btzsc) (wikitoxic, biasframes, amazonpolarity) | per source (CC0-1.0, CC-BY-4.0, Apache-2.0) | **held out**: zero-shot evaluation; yahootopics waits for its license |
| [Davlan/sib200](https://huggingface.co/datasets/Davlan/sib200) | CC-BY-SA-4.0 | **held out**: zero-shot evaluation |
| [DeepPavlov/hwu64](https://huggingface.co/datasets/DeepPavlov/hwu64) (HWU64, [original data](https://github.com/xliuhw/NLU-Evaluation-Data)) | CC-BY-4.0 | **held out** for s5-lite only (its training excludes MASSIVE in every language) |
| [bro789/typed-decision-coherence](https://github.com/bro789/typed-decision-coherence) (ChaosNLI CC BY-NC 4.0, PubMedQA MIT) | MIT code; items under their own terms | the original coherence battery of 2609.33209: **evaluation only**, never redistributed |

No output of a hosted LLM is used by s5.

## Libraries and technologies

### Training (Rust, Apple Silicon)
- [mlx-rs](https://github.com/oxiglade/mlx-rs) 0.32: Rust bindings for [MLX](https://github.com/ml-explore/mlx), Apple's array framework, built through mlx-sys with the Xcode Metal toolchain.
- [Hugging Face tokenizers](https://github.com/huggingface/tokenizers) 0.22: trains the byte-level BPE tokenizers.
- [arrow-rs `parquet`](https://github.com/apache/arrow-rs): reads Hugging Face parquet shards.
- [ureq](https://github.com/algesten/ureq): HTTPS downloads from the Hub, with range reads.
- Written in-house: the Muon optimizer (Newton–Schulz), the WSD schedule, ternary QAT, DAgger and closed-loop exit calibration.

### Inference and formats
- **GGUF** reader and writer, written in-house (`sky-gguf`), following the [llama.cpp GGUF spec](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md).
- **Engine** written in-house (`sky-infer`): byte-level BPE, the ModernBERT/Laya graph (s1), the s2/s3 graph, and NEON and WASM SIMD128 kernels. It depends on [unicode-normalization](https://github.com/unicode-rs/unicode-normalization) for the NFC step.
- [serde](https://serde.rs) / serde_json, [clap](https://github.com/clap-rs/clap), [rand](https://github.com/rust-random/rand), [indexmap](https://github.com/indexmap-rs/indexmap).

### Browser and distribution
- [wasm-bindgen](https://github.com/wasm-bindgen/wasm-bindgen) and [wasm-pack](https://github.com/rustwasm/wasm-pack): build the browser module that runs the games and the model.
- [VitePress](https://vitepress.dev) and [Vue](https://vuejs.org): this site.
- [PyO3](https://github.com/PyO3/pyo3) and [maturin](https://github.com/PyO3/maturin): the Python wheel.
- [tiny_http](https://github.com/tiny-http/tiny-http): the `skycmd serve` HTTP server for `/v1/systemone`.
- [Freedoom](https://github.com/freedoom/freedoom) (BSD-3-Clause): textures and sprites of the Doom-like game. It is not id Software's Doom.

## Reference models and runtimes
- [mgoeckel/oscar-1-17m](https://huggingface.co/mgoeckel/oscar-1-17m) (Apache-2.0): the golden reference for the Laya decision format.
- [convaiinnovations/laya](https://huggingface.co/convaiinnovations/laya) and [SupersonicLabs/Julia-1](https://huggingface.co/SupersonicLabs/Julia-1): the decision-head format and prompt layout.
- [llama.cpp `/v1/systemone`](https://github.com/ggml-org/llama.cpp/pull/29818) and [wllama](https://github.com/ngxson/wllama): compatibility checks for s1 GGUF files (see [`docs/compat.md`](https://github.com/maxgfr/skycmd/blob/main/docs/compat.md)).

## Research the architecture builds on
- **Recursion of a tiny shared block:** [TRM, Less is More](https://arxiv.org/abs/2510.04871) and [Ouro, looped language models](https://arxiv.org/abs/2510.25741).
- **One token per feature or number:** [FT-Transformer](https://arxiv.org/abs/2106.11959) and [FoNE](https://arxiv.org/abs/2502.09741).
- **Early exit with risk control:** [Fast yet Safe](https://arxiv.org/abs/2405.20915). **Readouts on looped trajectories:** [Adaptive Depth in Looped Transformers](https://arxiv.org/abs/2607.20519).
- **Ternary weights at small scale:** [When are 1.58 bits enough?](https://arxiv.org/abs/2411.05882). **Lookup-table kernels:** [T-MAC](https://arxiv.org/abs/2407.00088) and [bitnet.cpp](https://arxiv.org/abs/2502.11880).
- **Imitation learning:** DAgger (Ross, Gordon & Bagnell, 2011) and [on-policy distillation](https://www.thinkingmachines.ai/blog/on-policy-distillation/).
- **Bio-inspired ablations:** [Neural Circuit Policies](https://arxiv.org/abs/1809.04423) and [expand-and-sparsify / FlyHash](https://arxiv.org/abs/2001.04907).

### s4
- **Teacher:** [Qwen3.5](https://huggingface.co/collections/Qwen/qwen35) through [mlx-lm](https://github.com/ml-explore/mlx-lm); the teacher scores option letters in one pass and never generates.
- **Encoders over decoders for classification at equal size:** [Ettin, Seq vs Seq](https://arxiv.org/abs/2507.11412).
- **Ternary distillation from a teacher:** [BitNet Distillation](https://arxiv.org/abs/2510.13998).
- **Supervising every loop toward an improving target:** [Deep Improvement Supervision](https://arxiv.org/abs/2511.16886), applied to the exits of the looped state stack (`--dis`).
- **Hard and soft labels together:** [arXiv 2605.26246](https://arxiv.org/abs/2605.26246).
- **Answer bias of LLM option scoring and its correction:** [Calibrate Before Use](https://arxiv.org/abs/2102.09690) (contextual calibration) and [Large Language Models Are Not Robust Multiple Choice Selectors](https://arxiv.org/abs/2309.03882).
- **Small function-calling reference:** [Extending FunctionGemma](https://arxiv.org/abs/2609.25373).

### s5
- **Option isolation and shared positions:** [arXiv 2610.01601](https://arxiv.org/abs/2610.01601) (§ III, positions; § V-C, ablations).
- **Label swaps that flip decision models:** [arXiv 2609.26758](https://arxiv.org/abs/2609.26758), the reassignment test against a 0/1 control.
- **An aligned "none" option and recalibration:** [arXiv 2610.03387](https://arxiv.org/abs/2610.03387).
- **Coherence across separate questions (negation, partition, disjunction):** [arXiv 2609.33209](https://arxiv.org/abs/2609.33209) and its battery.
- **Evaluation checklist for decision models:** [arXiv 2609.32160](https://arxiv.org/abs/2609.32160). **Rerankers as baselines:** [arXiv 2603.11991](https://arxiv.org/abs/2603.11991).
- **The backbone:** [ModernBERT](https://arxiv.org/abs/2412.13663) and [Ettin](https://arxiv.org/abs/2507.11412).
- **Escalation with a controlled error:** [Learn then Test](https://arxiv.org/abs/2110.01052) (risk control by multiple testing) and Mondrian conformal prediction (Vovk et al.).
