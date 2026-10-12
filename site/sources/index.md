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

### Generated data
- **Game oracles**, written in Rust (`crates/sky-games`). They give exact labels for Flappy Drone, Rescue Run, Obstacle Dodge and the Doom-like game, plus the DAgger rounds, in which the model plays and the oracle labels the states it reaches.
- **Numeric curriculum** (`sky-train s2-numeric`): comparisons, ordering and closest values, with exact answers, in English and French.
- **GLM 5.3**, through the [z.ai](https://z.ai) API: 3,000 civil drone orders and about 2,000 paraphrases of question instructions.

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
