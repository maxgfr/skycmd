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
| [m-lagnajit/calibrated-typed-decisions](https://huggingface.co/datasets/m-lagnajit/calibrated-typed-decisions) | CC-BY-4.0 for the set; each row's text keeps its source's licence, given per row: Apache-2.0, CC-BY-3.0 or CC-BY-4.0 | training |
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
- **GLM-5.3-Flash** (open weights, MIT): 3,000 civil drone orders and about 2,000 paraphrases of question instructions (used by s1–s3 only; [`DATA_LICENSES.md`](https://github.com/maxgfr/skycmd/blob/main/DATA_LICENSES.md)).

### Evaluation only (never trained on)
- [LocalLLaMA/typed-decisions](https://huggingface.co/datasets/LocalLLaMA/typed-decisions) (Apache-2.0), its test split (`typed_decisions_bench`)
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

### s6-nano and s6-micro: teacher and test sets
A teacher gave soft labels on every training question of the Hugging Face decision sets above (747,077 of the 747,198 selected), kept only where it agrees with the gold answer. It is [GLM-5.3-flash](https://huggingface.co/zai-org/GLM-5.3-Flash), an open-weight model (MIT), called through the z.ai API. No test question is ever sent. They train on the decision sets, the games and the numeric curriculum above, and pretrain on the s3 text plus the open set above ([pre-registration](https://github.com/maxgfr/skycmd/blob/main/docs/s6-gen-prereg.md)).

| Dataset | Licence | Used as |
|---|---|---|
| [LocalLLaMA/typed-decisions](https://huggingface.co/datasets/LocalLLaMA/typed-decisions) (test split) | Apache-2.0 | the benchmark the gates are read on, `typed_decisions_bench`; its train split (`typed_decisions_train`) is trained on |
| [mteb/amazon_massive_intent](https://huggingface.co/datasets/mteb/amazon_massive_intent) (MASSIVE, en + fr) | CC-BY-4.0 | **held out**: zero-shot evaluation; 206 of the 2,000 English test utterances also appear in the training split, 201 of them in a source derived from MASSIVE (found after the evaluation, [results](/results/#s6-november-2026)) |
| [AirsideLabs/notam-typed-decisions](https://huggingface.co/datasets/AirsideLabs/notam-typed-decisions) | CC-BY-4.0 | **held out**: test only |
| [nvidia/When2Call](https://huggingface.co/datasets/nvidia/When2Call) (test split) | CC-BY-4.0 | test split of a trained source |
| [PolyAI/banking77](https://huggingface.co/datasets/PolyAI/banking77) | CC-BY-4.0 | trained on: reported as in-distribution, never as zero-shot |

### s6-secrets-pico: generated data
No external dataset. `sky-data secret-jev` generates every text locally: each key follows its provider's public format with a random body, so none is a real key ([`docs/s6-secrets-prereg.md`](https://github.com/maxgfr/skycmd/blob/main/docs/s6-secrets-prereg.md) §1). The bar is a rule scanner written in Rust in the style of gitleaks (see [s6](#s6) below). It pretrains on the same text as s6-nano.

### s6-ear-nano: audio data and baseline
Only derived features are kept on disk (log-mel frames, and WAV for the test clips); the audio is not redistributed ([`DATA_LICENSES.md`](https://github.com/maxgfr/skycmd/blob/main/DATA_LICENSES.md)). The model failed its gates; its file is published as a result (owner's decision of 15/11/2026).

| Dataset or model | Licence | Used as |
|---|---|---|
| [pipecat-ai/smart-turn-data-v3-train](https://huggingface.co/datasets/pipecat-ai/smart-turn-data-v3-train) | CC-BY-4.0 | end-of-turn questions, train and val |
| [pipecat-ai/smart-turn-data-v3.2-test](https://huggingface.co/datasets/pipecat-ai/smart-turn-data-v3.2-test) | CC-BY-4.0 | end-of-turn test: the official held-out set of Smart Turn v3.2 |
| [pipecat-ai/smart-turn-v3](https://huggingface.co/pipecat-ai/smart-turn-v3) (Smart Turn v3.2, ONNX) | BSD-2-Clause | the baseline of the end-of-turn gate, run on the same test clips |
| Speech Commands v0.02 ([paper](https://arxiv.org/abs/1804.03209), [`google/speech_commands`](https://huggingface.co/datasets/google/speech_commands)) | CC-BY-4.0 | keyword questions, 12 classes |
| [MLCommons/ml_spoken_words](https://huggingface.co/datasets/MLCommons/ml_spoken_words) (MSWC, en + fr) | CC-BY-4.0 | zero-shot keywords: 50 words per language held out |
| [openslr/librispeech_asr](https://huggingface.co/datasets/openslr/librispeech_asr) (`clean`) | CC-BY-4.0 | audio of the pretraining |
| [shangeth/librispeech-mimi-codes](https://huggingface.co/datasets/shangeth/librispeech-mimi-codes) | CC-BY-4.0 | the Mimi codebook-0 ids predicted during pretraining; Mimi itself never runs |

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

## Our datasets on Hugging Face

Every set a skycmd model reads is published, one parquet config per source with its licence
(`scripts/hf-publish-data.py`): [skycmd-decisions](https://huggingface.co/datasets/maxgfr/skycmd-decisions), [skycmd-games](https://huggingface.co/datasets/maxgfr/skycmd-games), [skycmd-numeric](https://huggingface.co/datasets/maxgfr/skycmd-numeric), [skycmd-secrets](https://huggingface.co/datasets/maxgfr/skycmd-secrets), [skycmd-commands](https://huggingface.co/datasets/maxgfr/skycmd-commands), [skycmd-zero-shot](https://huggingface.co/datasets/maxgfr/skycmd-zero-shot), [skycmd-series](https://huggingface.co/datasets/maxgfr/skycmd-series), [skycmd-text](https://huggingface.co/datasets/maxgfr/skycmd-text), [skycmd-ear](https://huggingface.co/datasets/maxgfr/skycmd-ear), [skycmd-math](https://huggingface.co/datasets/maxgfr/skycmd-math), [skycmd-route](https://huggingface.co/datasets/maxgfr/skycmd-route), [skycmd-route7](https://huggingface.co/datasets/maxgfr/skycmd-route7), [skycmd-cast-synth](https://huggingface.co/datasets/maxgfr/skycmd-cast-synth), [skycmd-evals](https://huggingface.co/datasets/maxgfr/skycmd-evals), [skycmd-trajectories](https://huggingface.co/datasets/maxgfr/skycmd-trajectories), [skycmd-glm-labels](https://huggingface.co/datasets/maxgfr/skycmd-glm-labels). The 16 sets are also listed with the models and their Spaces in the
[skycmd collection](https://huggingface.co/collections/maxgfr/skycmd-6b001dd30a7cb5724995b603) on Hugging Face. The GLM-5.3-flash labels are AI-generated and
marked as such. skycmd-decisions holds both splits of `LocalLLaMA/typed-decisions` (Apache-2.0)
since 20 November 2026: `typed_decisions_train`, and `typed_decisions_bench`, the benchmark's test
split, for evaluation only. Not republished: the s6-ear audio (31 GB of log-mel features, rebuilt
by `sky-data fetch-audio`) and GIFT-Eval (evaluation only).

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
- **Recursion of a tiny shared block:** [TRM, Less is More](https://arxiv.org/abs/2510.04871) and [Ouro, looped language models](https://arxiv.org/abs/2510.25741), after the weight sharing across depth of [Universal Transformers](https://arxiv.org/abs/1807.03819) (Dehghani et al., ICLR 2019). **A variable number of passes:** [Adaptive Computation Time](https://arxiv.org/abs/1603.08983) (Graves, 2016) and [PonderNet](https://arxiv.org/abs/2107.05407) (Banino et al., 2021).
- **One token per feature or number:** [FT-Transformer](https://arxiv.org/abs/2106.11959) and [FoNE](https://arxiv.org/abs/2502.09741).
- **Early exits:** [BranchyNet](https://arxiv.org/abs/1709.01686) (Teerapittayanon et al., 2017), and in language encoders [DeeBERT](https://arxiv.org/abs/2004.12993) and [FastBERT](https://arxiv.org/abs/2004.02178) (both ACL 2020). **Early exit with risk control:** [Fast yet Safe](https://arxiv.org/abs/2405.20915). **Readouts on looped trajectories:** [Adaptive Depth in Looped Transformers](https://arxiv.org/abs/2607.20519).
- **A loss at every exit, and loops skipped at random in training** (`docs/SPEC-S3.md` §3): [Deeply-Supervised Nets](https://arxiv.org/abs/1409.5185) (Lee et al., 2014) and [stochastic depth](https://arxiv.org/abs/1603.09382) (Huang et al., 2016).
- **Ternary weights at small scale:** [When are 1.58 bits enough?](https://arxiv.org/abs/2411.05882). **Lookup-table kernels:** [T-MAC](https://arxiv.org/abs/2407.00088) and [bitnet.cpp](https://arxiv.org/abs/2502.11880).
- **Imitation learning:** DAgger ([Ross, Gordon & Bagnell, AISTATS 2011](https://arxiv.org/abs/1011.0686)) and [on-policy distillation](https://www.thinkingmachines.ai/blog/on-policy-distillation/).
- **Distillation from a teacher's soft labels** (GLM-5.3-flash, Smart Turn, TinyCast): [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531) (Hinton, Vinyals & Dean, 2015).
- **Bio-inspired ablations:** compact neuronal circuit policies ([Hasani et al., 2018](https://arxiv.org/abs/1809.04423); the later [Neural Circuit Policies](https://doi.org/10.1038/s42256-020-00237-3), Lechner et al., Nature Machine Intelligence 2020) and expand-and-sparsify hashing: FlyHash ([Dasgupta, Stevens & Navlakha, Science 2017](https://doi.org/10.1126/science.aam9868)) and its learned variant [BioHash](https://arxiv.org/abs/2001.04907) (Ryali et al., ICML 2020).
- **Training in Rust and running inference in WebAssembly is not new either:** [Burn](https://github.com/tracel-ai/burn) (Tracel AI), a deep learning framework in Rust, does both. skycmd's engine is its own implementation; what it adds is the combination (ternary kernels, the looped graph with an exit after each pass, one file run natively and in the browser).

### Methods the code uses throughout
Written in-house in Rust, after these:
- **Encoder layout:** [RoPE](https://arxiv.org/abs/2104.09864) and a [GeGLU](https://arxiv.org/abs/2002.05202) MLP, as in [ModernBERT](https://arxiv.org/abs/2412.13663).
- **Optimizer and schedule:** [Muon](https://kellerjordan.github.io/posts/muon/) (Keller Jordan et al., blog post, December 2024), ported from `mlx.optimizers.Muon`, for s1 to s3; the s6 ablation kept AdamW. The warmup-stable-decay schedule of [MiniCPM](https://arxiv.org/abs/2404.06395), on a time budget.
- **Ternary weights:** the absmean ternary quantization of [BitNet b1.58](https://arxiv.org/abs/2402.17764), trained in with a [straight-through estimator](https://arxiv.org/abs/1308.3432).
- **Calibration and its measure:** [temperature scaling](https://arxiv.org/abs/1706.04599) per question type, and the ECE with 15 equal-width bins, as defined in the same paper.
- **Conformal exit thresholds:** the bound `(n·R̂ + 1)/(n + 1) ≤ α` of [Conformal Risk Control](https://arxiv.org/abs/2208.02814) (ICLR 2024; conformal prediction as surveyed in [A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification](https://arxiv.org/abs/2107.07511), Angelopoulos & Bates), with the early-exit setting of [CALM](https://arxiv.org/abs/2207.07061) and [Fast yet Safe](https://arxiv.org/abs/2405.20915).

### s4
- **Teacher:** [Qwen3.5](https://huggingface.co/collections/Qwen/qwen35) through [mlx-lm](https://github.com/ml-explore/mlx-lm); the teacher scores option letters in one pass and never generates.
- **Encoders over decoders for classification at equal size:** [Ettin, Seq vs Seq](https://arxiv.org/abs/2507.11412).
- **Ternary distillation from a teacher:** [BitNet Distillation](https://arxiv.org/abs/2510.13998).
- **Supervising every loop toward an improving target:** [Latent Reasoning in TRMs is Secretly a Policy Improvement Operator](https://arxiv.org/abs/2511.16886) (Asadulaev et al.; titled "Deep Improvement Supervision" in its first versions, hence the project's name for the method), applied to the exits of the looped state stack (`--dis`).
- **Hard and soft labels together:** [arXiv 2605.26246](https://arxiv.org/abs/2605.26246).
- **Answer bias of LLM option scoring and its correction:** [Calibrate Before Use](https://arxiv.org/abs/2102.09690) (contextual calibration) and [Large Language Models Are Not Robust Multiple Choice Selectors](https://arxiv.org/abs/2309.03882).
- **Small function-calling reference:** [Extending FunctionGemma](https://arxiv.org/abs/2609.25373).

### s6
What the pre-registrations rest on ([`docs/s6-prereg.md`](https://github.com/maxgfr/skycmd/blob/main/docs/s6-prereg.md), [`docs/s6-gen-prereg.md`](https://github.com/maxgfr/skycmd/blob/main/docs/s6-gen-prereg.md), notes in [`docs/research.md`](https://github.com/maxgfr/skycmd/blob/main/docs/research.md)). The ablation adopted none of the encoder and optimizer additions: the shipped s6-nano rests on the learning-rate transfer, the pretraining budget, the QAT schedule and DAgger, and the conformal thresholds were tested on it and rejected.
- **Tested, not adopted:** [Parallax](https://arxiv.org/abs/2605.29157) attention; Muon v2 after [Muon is Scalable for LLM Training](https://arxiv.org/abs/2502.16982); exclusive self-attention ([XSA](https://arxiv.org/abs/2603.09078), Zhai, 2026); [SODA](https://arxiv.org/abs/2605.11172) weight decay; exit distillation after [SAFE-KD](https://arxiv.org/abs/2602.03043); a weight EMA (`--ema 0.999`, an exponential moving average of the weights; proposed against the oscillation of quantization-aware training by [Gupta and Asthana](https://arxiv.org/abs/2311.05109), who apply it to quantized YOLO networks, with [EMA Without the Lag](https://arxiv.org/abs/2508.00180), BEMA, on the averaging itself); [StableQAT](https://arxiv.org/abs/2601.19320); a pretraining curriculum ([2506.11300](https://arxiv.org/abs/2506.11300)).
- **Why the ablation tunes AdamW too and counts time:** [Fantastic Pretraining Optimizers](https://arxiv.org/abs/2509.02046) and [When Does Muon Actually Win?](https://arxiv.org/abs/2610.21555).
- **Learning rate across widths:** [fast hyperparameter transfer](https://arxiv.org/abs/2512.22768) and the spectral condition for μP ([2603.00541](https://arxiv.org/abs/2603.00541)): the per-size pilots are centred on the pico rate × √(96/d).
- **Pretraining budget and QAT:** Training compute-optimal transformer encoder models (EMNLP 2025), on how many tokens per parameter an MLM encoder wants; [16-bit → 1.58-bit continual QAT](https://arxiv.org/abs/2502.11895), for the float warm-up before ternary training.
- **Conformal exit thresholds:** [Fast yet Safe](https://arxiv.org/abs/2405.20915) and [CALM](https://arxiv.org/abs/2207.07061) (NeurIPS 2022), for α = 0.02. They were tested on s6-nano in closed loop and rejected.
- **DAgger with a per-turn mixture:** [Revisiting DAgger in the Era of LLM-Agents](https://arxiv.org/abs/2605.12913).
- **s6-secrets-pico's bar:** a rule scanner in the style of [gitleaks](https://github.com/gitleaks/gitleaks) (MIT): prefix rules for 20 key formats, connection strings with a password, an entropy rule on secret-sounding names and allowlists, written in Rust (`sky-data secret-baseline`), not gitleaks itself. Its data is synthetic and generated locally ([`docs/s6-secrets-prereg.md`](https://github.com/maxgfr/skycmd/blob/main/docs/s6-secrets-prereg.md); licence in [`DATA_LICENSES.md`](https://github.com/maxgfr/skycmd/blob/main/DATA_LICENSES.md)), and it pretrains on the same text as s6-nano.
- **s6-ear-nano:** Kyutai's [Mimi](https://arxiv.org/abs/2410.00037) codec (from Moshi) as the source of the pretraining targets; masked prediction of discrete targets after [HuBERT](https://arxiv.org/abs/2106.07447) and [BEST-RQ](https://arxiv.org/abs/2202.01855); continuous features rather than codec indices for classification ([DASB](https://arxiv.org/abs/2406.14294), [DualTurn](https://arxiv.org/abs/2603.08216)); a small frontal model that calls a larger one ([MoshiRAG](https://arxiv.org/abs/2604.12928)).

### s5
- **Option isolation and shared positions:** [arXiv 2610.01601](https://arxiv.org/abs/2610.01601) (§ III, positions; § V-C, ablations).
- **Label swaps that flip decision models:** [arXiv 2609.26758](https://arxiv.org/abs/2609.26758), the reassignment test against a 0/1 control.
- **An aligned "none" option and recalibration:** [arXiv 2610.03387](https://arxiv.org/abs/2610.03387).
- **Coherence across separate questions (negation, partition, disjunction):** [arXiv 2609.33209](https://arxiv.org/abs/2609.33209) and its battery.
- **Evaluation checklist for decision models:** [arXiv 2609.32160](https://arxiv.org/abs/2609.32160). **Rerankers as baselines:** [arXiv 2603.11991](https://arxiv.org/abs/2603.11991).
- **The backbone:** [ModernBERT](https://arxiv.org/abs/2412.13663) and [Ettin](https://arxiv.org/abs/2507.11412).
- **Escalation with a controlled error:** [Learn then Test](https://arxiv.org/abs/2110.01052) (risk control by multiple testing) and Mondrian conformal prediction (Vovk, Gammerman & Shafer, [Algorithmic Learning in a Random World](https://doi.org/10.1007/978-3-031-06649-8), Springer, first edition 2005, second edition 2022).

### Later s6 lines: end of turn, tool routing, secrets verifier, forecasting v2, memory
Each line's data and licences are in [`DATA_LICENSES.md`](https://github.com/maxgfr/skycmd/blob/main/DATA_LICENSES.md); the works below are those its pre-registration and the technical report rest on.

- **End of turn (s6-ear2-nano, s6-ear3-nano):** the teacher and baseline [Smart Turn v3.2](https://github.com/pipecat-ai/smart-turn) (Pipecat, BSD-2-Clause) and its test set; LiveKit's [eot-bench](https://huggingface.co/datasets/livekit/eot-bench-data) (CC-BY-4.0); for v3, the gated [otoSpeech](https://huggingface.co/datasets/otoearth/otoSpeech-full-duplex-280h) conversations (CC-BY-4.0) and [Silero VAD](https://github.com/snakers4/silero-vad) (MIT) as the second detector of the label cross-check.
- **Tool routing (s6-route-nano, s6-route, s6-route7-nano):** BM25 ([Robertson & Zaragoza, The Probabilistic Relevance Framework: BM25 and Beyond](https://doi.org/10.1561/1500000019), Foundations and Trends in Information Retrieval, 2009), computed with the Lucene variant of [BM25S](https://arxiv.org/abs/2407.03618) (Lù, 2024); the static embedding [potion-base-2M](https://huggingface.co/minishlab/potion-base-2M) (Model2Vec, MIT) and the sentence encoder [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2); fresh MCP tools from [Toucan](https://arxiv.org/abs/2510.01179) (Xu et al., 2025; [data](https://huggingface.co/datasets/Agent-Ark/Toucan-1.5M)); [When2Call](https://aclanthology.org/2025.naacl-long.174/) (Ross, Mahabaleshwarkar & Suhara, NAACL 2025); [MetaTool](https://github.com/HowieHwong/MetaTool) and [CLINC150](https://huggingface.co/datasets/clinc/clinc_oos) for the reported readings.
- **Secrets verifier (s6-secrets-verify-pico, s6-secrets v3):** Samsung's [CredData](https://github.com/Samsung/CredData); the [Secret-Leak-Detection issue reports](https://zenodo.org/records/19622962) (CC-BY-4.0); the rule scanners [Gitleaks](https://github.com/gitleaks/gitleaks), [Betterleaks](https://github.com/betterleaks/betterleaks) and [secretgate](https://github.com/maxgfr/secretgate)'s rules; the model [cbx-secrets-detector-17M](https://huggingface.co/CobuildX-AI/cbx-secrets-detector-17M), rerun on the same test.
- **Forecasting (s6-cast-pico, s6-cast2-pico):** [GIFT-Eval](https://arxiv.org/abs/2410.10393) (Aksu et al., 2024; [data](https://huggingface.co/datasets/Salesforce/GiftEval)); the Chronos zero-shot benchmark ([Chronos](https://arxiv.org/abs/2403.07815), Ansari et al., 2024); [TinyCast](https://huggingface.co/raws-labs/tinycast) (RAWS Labs, Apache-2.0), teacher and reference, with its zero-parameter period detector; FLAIR's numbers as TinyCast reports them; [Timestral-5M](https://huggingface.co/mistralai/Timestral-5M) as a browser reference.
- **Memory (s6-sleep):** the "sleep" passes of [Do Language Models Need Sleep?](https://arxiv.org/abs/2605.26099) (Lee et al., 2026).

## Related work found later
Two prior-art searches made in November 2026, after most lines had been read, found the works below. Each one is close to a claim of the project; [What we bring](/novelty/) and the [technical report](/paper/) say what each changes. None of them changes a verdict.

### The decision format
- **The format itself:** TypeSafe's [Introducing System One Models & Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev) (blog post, 15 September 2026).
- **Small open models in it:** [cua-ai/cua-s1-forms](https://huggingface.co/cua-ai/cua-s1-forms) (MIT, 706,048 parameters, a byte-level embedding and a 2-layer Transformer encoder of width 128; a form-filling option scorer with Jev's input and output contract, one task; created 18 September 2026) and [TheREZOR/TinyDecide](https://huggingface.co/TheREZOR/TinyDecide) (Apache-2.0, from ELECTRA-small; its 10.4M build is 6.2 MB with 4-bit weights, for choice, noul and score; created 5 October 2026).
- **Order invariance by construction:** [Set-LLM](https://arxiv.org/abs/2505.15433) (Egressy & Stühmer, 2025), which proves it with set attention masks and set positions; [Amit](https://arxiv.org/abs/2610.01601), already credited above; and, concurrent with s5 (13 October 2026), [Jeva: System One Models from First Principles](https://arxiv.org/abs/2610.18773) (Li & Wang), which also asks for option computation that is cacheable across requests, and Runoff, in [Consider, Then Choose: Order-Invariant Decisions in System One Models](https://arxiv.org/abs/2610.18233) (Huang et al.).

### Early exits in closed loop
- [DeeR-VLA](https://arxiv.org/abs/2411.02359) (Yue et al., NeurIPS 2024) sets the exit thresholds of a robot vision-language-action model from demonstrations or, when the environment can be used, by Bayesian optimisation against task success under compute budgets.

### By line
- **Secrets verifier, rules then a learned judge:** the design of Samsung's [CredSweeper](https://github.com/Samsung/CredSweeper) (rules, then an ML validator; its `ml_model.onnx` is 11,545,911 bytes), SAP's [Credential Digger](https://github.com/SAP-archive/credential-digger) (Lounici et al., ICISSP 2021) and GitGuardian's [FP Remover](https://blog.gitguardian.com/fp-remover-cuts-false-positives-by-half/) (26 June 2024).
- **End of turn:** [A Hierarchical End-of-Turn Model with Primary Speaker Segmentation](https://arxiv.org/abs/2603.13379) (Helwani et al., 2026), a 1.14M-parameter model distilled from wav2vec 2.0 representations into an MFCC-based student, and [turn-1-mini](https://github.com/P99Lab/turn-1-mini) (p99lab, October 2026), a streaming turn-taking system of 6,098,317 learned parameters.
- **Tool routing:** [semantic-router](https://github.com/aurelio-labs/semantic-router) (aurelio-labs, MIT), which routes a request by its embedding's similarity to each route, and [OATS](https://arxiv.org/abs/2603.13426) (Chen et al., 2026), which moves tool embeddings toward the requests where each tool succeeded, offline, with no parameter added at serving time.
- **Forecasting:** [Fracast-0](https://arxiv.org/abs/2609.32209) (Zhan et al., 26 September 2026), 85,001 parameters, reports MASE 0.808 and WQL 0.564 on the same 97 GIFT-Eval configurations: smaller than s6-cast2-pico and, by its abstract, better.

### The protocol
- **Pre-registration in machine learning:** the NeurIPS 2020 and 2021 workshops on pre-registration ([PMLR volume 148](https://proceedings.mlr.press/v148/), [PMLR volume 181](https://proceedings.mlr.press/v181/)); [Pre-registration for Predictive Modeling](https://arxiv.org/abs/2311.18807) (Hofman et al., 2023); [a pre-registered negative result on LLM compression](https://arxiv.org/abs/2609.34292) (Lu, 2026); [RSI-Jev](https://github.com/longzhenren/RSI-Jev), typed-decision models whose every failed version is published. Pre-registration and publishing failed models are not new; skycmd's part is applying them to every line from s5 on, with a test lock enforced by its tools and dated amendments: 15 of its 20 published files were judged by a gate fixed before training, and every file is published, failures included.
- **Controls that hide part of the input:** hypothesis-only baselines ([Annotation Artifacts in Natural Language Inference Data](https://arxiv.org/abs/1803.02324), Gururangan et al., NAACL 2018) and choices-only baselines ([Artifacts or Abduction](https://arxiv.org/abs/2402.12483), Balepur et al., ACL 2024).

### Added with the review of the technical report (21 November 2026)
Each was opened on its page before it was cited in the [technical report](/paper/); none changes a verdict.
- **Small or ternary decision models:** [Sparrow](https://github.com/sparrow-decider/sparrow-decider) (about 2 KB per question fixed at build time, general tiers from 1.11 MB); [blink](https://github.com/sqliteai/blink) (choices among options given at call time, 452 KiB of int8 weights); [BitSOne](https://github.com/henrythdu/BitSOne) (a ternary decision model on BitNet b1.58 2B4T, private weights); [LEV](https://arxiv.org/abs/2610.20951) (Song et al., 2026), looped decision models with a stopper per question.
- **Ternary weights on small networks:** [BitNet b1.58 Reloaded](https://arxiv.org/abs/2407.09527) (Nielsen and Schneider-Kamp, DeLTA 2024).
- **The ablation's additions:** [Muon is Scalable for LLM Training](https://arxiv.org/abs/2502.16982) (Liu et al., 2025); [Parallax](https://arxiv.org/abs/2605.29157) (Zuo et al., 2026); [Exclusive Self Attention](https://arxiv.org/abs/2603.09078) (Zhai, 2026); SODA in [Optimistic Dual Averaging Unifies Modern Optimizers](https://arxiv.org/abs/2605.11172) (Pethick et al., 2026); [SAFE-KD](https://arxiv.org/abs/2602.03043) (Khazem, 2026); [EMA Without the Lag](https://arxiv.org/abs/2508.00180) (Block and Zhang, 2025); [StableQAT](https://arxiv.org/abs/2601.19320) (Chen et al., 2026); [a pretraining curriculum](https://arxiv.org/abs/2506.11300) (Zhang et al., 2025).
- **Sets, models and tools the gates rest on:** [`LocalLLaMA/typed-decisions`](https://huggingface.co/datasets/LocalLLaMA/typed-decisions); [MASSIVE](https://aclanthology.org/2023.acl-long.235/) (FitzGerald et al., ACL 2023); the Mimi codec of [Moshi](https://arxiv.org/abs/2410.00037) (Défossez et al., 2024); [otoSpeech-full-duplex-280h](https://huggingface.co/datasets/otoearth/otoSpeech-full-duplex-280h); [Silero VAD](https://github.com/snakers4/silero-vad); the [TinyCast paper](https://arxiv.org/abs/2608.15767) (Steinhauser, 2026) and [FLAIR](https://arxiv.org/abs/2605.07222) (Honda, 2026), whose numbers it reports; [cbx-secrets-detector-17M](https://huggingface.co/CobuildX-AI/cbx-secrets-detector-17M), whose card gives Gitleaks' 0.442 (span F1, 168 CredData test repositories); [LFM3-1.7B](https://huggingface.co/LiquidAI/LFM3-1.7B); [GSM8K](https://arxiv.org/abs/2110.14168) (Cobbe et al., 2021); MATH-500 from [Let's Verify Step by Step](https://arxiv.org/abs/2305.20050) (Lightman et al., ICLR 2024), a subset of [MATH](https://arxiv.org/abs/2103.03874) (Hendrycks et al., 2021).
- **Close work by line:** GitHub's classifier of candidate secrets ([Secret protection must scale with software](https://github.blog/ai-and-ml/github-copilot/secret-protection-must-scale-with-software/), 7 October 2026); [The Importance of Speech Genre for End-of-Turn Detector Training](https://arxiv.org/abs/2610.39742) (Sharon et al., 2026); [ToolRet](https://aclanthology.org/2025.findings-acl.1258/) (Shi et al., Findings of ACL 2025).
- **Early exits in closed loop:** [Efficient Reductions for Imitation Learning](https://proceedings.mlr.press/v9/ross10a.html) (Ross and Bagnell, AISTATS 2010), on errors that compound with the horizon; [Safe Planning in Interactive Environments via Iterative Policy Updates and Adversarially Robust Conformal Prediction](https://arxiv.org/abs/2511.10586) (Mirzaeedodangeh et al., 2025), on exchangeability broken by interaction.
- **The protocol:** [Readable Is Not Steerable](https://arxiv.org/abs/2611.05577) (Modgil, 2026), pre-registered tests read once on held-out data; [Misleading Failures of Partial-input Baselines](https://aclanthology.org/P19-1554/) (Feng et al., ACL 2019).
- **The companion methods report:** [Looped ternary encoders for typed decisions](/paper/methods/) (Golfier, 2026).

### Added with the review of the methods report (21 November 2026)
Each was opened on its page before it was cited in the [methods report](/paper/methods/); none changes a verdict. The others it cites are listed above.
- **Ternary weights and their training:** the per-filter mean-absolute scale of [XNOR-Net](https://arxiv.org/abs/1603.05279) (Rastegari et al., 2016) and [Ternary Weight Networks](https://arxiv.org/abs/1605.04711) (Li et al., 2016); AdamW, in [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101) (Loshchilov and Hutter, ICLR 2019).
- **Numbers as tokens:** [xVal](https://arxiv.org/abs/2310.02989) (Golkar et al., 2023) and the periodic embeddings of [On Embeddings for Numerical Features in Tabular Deep Learning](https://arxiv.org/abs/2203.05556) (Gorishniy et al., NeurIPS 2022).
- **Depth, exits and their calibration:** [LoopFormer](https://arxiv.org/abs/2602.11451) (Jeddi et al., ICLR 2026); [Fixing Overconfidence in Dynamic Neural Networks](https://arxiv.org/abs/2302.06359) (Meronen et al., WACV 2024); CATs, in [Consistent Accelerated Inference via Confident Adaptive Transformers](https://arxiv.org/abs/2104.08803) (Schuster et al., EMNLP 2021).
- **DAgger without aggregation:** SMILe, in [Efficient Reductions for Imitation Learning](https://proceedings.mlr.press/v9/ross10a.html) (Ross and Bagnell, AISTATS 2010), the closest named method to the rounds of the methods report.
- **End of turn:** the lag-domain pitch analysis of [Boersma (1993)](https://www.fon.hum.uva.nl/paul/papers/Proceedings_1993.pdf) (Institute of Phonetic Sciences, University of Amsterdam, Proceedings 17) and the log-mel normalisation of Whisper, in [Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356) (Radford et al., 2022).
- **Tool routing and secrets:** [Model2Vec](https://doi.org/10.5281/zenodo.17270888) (Tulkens and van Dongen, 2024), the library of potion-base-2M; CredData as its authors cite it, Project CredData: A Dataset of Credentials for Research (Yun et al., Samsung, 2021).
