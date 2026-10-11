# Research notes (as of 2026-10-11)

These notes record what skycmd builds on. I checked them against the Hugging Face API, the llama.cpp repository and crates.io/npm on 2026-10-11. Numbers that vendors report about their own models are labelled as such.

## System One / decision models
In a decision model, you send a **state** plus **typed questions** (`choice`, `score`, `noul` = probabilistic yes/no). The model returns a probability for every allowed option in one forward pass and generates no tokens. TypeSafe's Jev (15 Sept 2026, closed API) launched the category. Its latency of 70–500 ms per call is vendor-reported.

Open models in the same format:

| Model | Size | Base | License | GGUF |
|---|---|---|---|---|
| `convaiinnovations/laya` | 421M | ModernBERT-large + 2-layer head | Apache-2.0 | `ggml-org/Laya-GGUF` |
| `SupersonicLabs/Julia-1` | 144M | mmBERT-small | Apache-2.0 | `ggml-org/Julia-1-GGUF` |
| `mgoeckel/oscar-1-17m` / `-32m` | 17M / 32M | ettin-encoder | Apache-2.0 | (Laya layout, convertible) |
| `jaredpalmer/kev-4b` | 4B | Qwen3.5-4B LoRA + head | Apache-2.0 | `ggml-org/Kev-4B-GGUF` |
| `interfaze-ai/lev` | 4B | Qwen3.5-4B LoRA | Apache-2.0 | `ggml-org/lev-GGUF` |
| `Cloudflare/clef`, `clef-flash` | 27B / 9.4B | Qwen | Apache-2.0 | `ggml-org/Clef-GGUF` |
| `openjev/openjev` | 27B | Qwen3.5 | CC-BY-NC-4.0 | `ggml-org/OpenJev-GGUF` |
| `bespokelabs/Bespoke-Nimble-9B-v3` | 9B | Qwen3.5 | CC-BY-NC-4.0 | yes |
| `LiquidAI/d1-3B`, `d1-omni-600M` | 3B / 600M | LFM2.5 | LFM Open (other) | yes |

According to its model card, `oscar-1-17m` scores **0.6775** on `LocalLLaMA/typed-decisions`. About 12.9M of its 17M parameters sit in the 50k-token embedding table, so its transformer itself is tiny. skycmd pushes that further with a 1k–4k vocabulary.

## How the decision format works (llama.cpp PR #29818, merged 2026-10-02)
- **Prompt**: `[CLS]{type} question: {instructions}[SEP]`, then `[MASK] {option}` for each option, then `[SEP]{state}[SEP]`.
- **Model**:
  - The encoder is ModernBERT.
  - The head runs once per question type. It computes `final_norm(enc) + type_emb[t]`, then applies H pre-LN `nn.TransformerEncoderLayer` blocks (full attention, no RoPE, ReLU FFN of size 4d, with biases), then the scorer `LayerNorm → Linear → GELU → Linear` at each `[MASK]`.
- **Readout**: `softmax(score / T)`. T is looked up per type and per option-count bucket (`2`, `3_5`, `6_10`, `11`).
  - `choice` returns the argmax and `confidence = (pmax − 1/n)/(1 − 1/n)`.
  - `score` returns the expected level.
  - `noul` returns p(true).
- **GGUF**: `general.architecture = modern-bert` with these keys:
  - `modern-bert.decision.type = "laya"`
  - `modern-bert.decision.block_count`
  - `modern-bert.decision.max_head_tokens`
  - `modern-bert.decision.temperature.<type>[.<bucket>]`
  - `tokenizer.ggml.token_type_count = 3`
  - a `systemone` chat template.

  The head layers continue the `blk.N` numbering after the encoder layers. The extra tensors are `token_types.weight` and `cls.{norm,weight,output}`.
- **Runtimes**: llama.cpp `llama-server` (`/v1/systemone`), LM Studio, and wllama 3.8.1 in the browser (`createSystemOne()`).

## Papers that shaped the training recipe (HF papers, Aug–Oct 2026)
- **2610.02486**: the published RLCD recipe trails plain cross-entropy by 2.5–3 points, and CE followed by temperature scaling calibrates just as well. → skycmd trains with CE on soft targets and then fits temperatures.
- **2610.02586 / 2609.26758**: decision heads latch onto option *labels* more than their definitions. → We augment by shuffling and renaming options.
- **2610.06744**: scoring each option independently of the others reduces flips when options are shuffled.
- **2609.33209**: checking that p(X) + p(¬X) ≈ 1 is a coherence test that needs no labels. → We use it in eval.
- **2608.07436**: Muon can collapse at the representation/readout interface. → Muon is used only on encoder hidden matrices, and AdamW on embeddings and the head.
- **2609.30535**: code-switching curricula help align languages. → We mix FR and EN.
- **2609.01404 (DroneCATS)**: small models struggle with termination actions. → `end` and `abort` are explicit actions.
- **2609.23742**: constrained decoding removes structural JSON errors but not semantic ones. → A coded validator always gets the last word.

## Datasets used
Licenses and row counts are in [`DATA_LICENSES.md`](../DATA_LICENSES.md).
- Typed decisions (Jev / System One format, soft targets kept):
  - `n4ze3m/typed-decisions-synth` (MIT), `kgrozdanovski/assistant-decisions` (Apache-2.0), `pngwn/typed-decisions` (CC-BY-SA-4.0, soft `gold_probs`);
  - `tasksource/tasksource-jev-typed-decisions`: only rows tagged `license_use = commercial`, minus sources we load directly or whose upstream is non-commercial; 120k rows shared fairly over 389 sources;
  - `tasksource/procedural-typed-decisions` (Apache-2.0, 80k over 13 procedural tasks), `tasksource/synthetic-typed-decisions` (Apache-2.0), `helmo/synthetic-typed-decisions` (MIT), `m-lagnajit/calibrated-typed-decisions` (CC-BY-4.0, permissive rows, calibrated teacher probabilities);
  - control: `ZefanCai/Open-Jev` `browser-drone-expansion-v1-redistributable` (CC0; all 18,130 drone-control rows plus 30k from the other sources) and `Aditharavind/swarm-laya-decisions` (Apache-2.0, robot-swarm actions).
- Classic sets converted to typed questions: BoolQ, MNLI, SNLI, ARC, CommonsenseQA, HellaSwag, Banking77, HelpSteer2 (one 0–4 score question per attribute), When2Call (tool_call / request_for_info / cannot_answer / direct_answer), and French: PAWS-X fr (paraphrase noul) and Allociné (sentiment noul).
- GLM teacher data: the 3,000 civil drone-order `action` records (`glm_orders`) and 40k records whose instructions are replaced by EN/FR GLM paraphrases (`paraphrased`).
- Test only: `LocalLLaMA/typed-decisions` (Apache-2.0), `AirsideLabs/notam-typed-decisions` (CC-BY-4.0) and the When2Call `mcq` test split.
- Commands (for the NL → command decoder, `data/commands/`): `NousResearch/hermes-function-calling-v1` and `glaiveai/glaive-function-calling-v2` (Apache-2.0) as `{user, tools, target_json}`, MASSIVE en + fr (CC-BY-4.0) as intents.
- MLM text: SimpleStories (MIT), TinyStories and tinystories-fr (CDLA-Sharing), fineweb-edu and fineweb-2 `fra_Latn` (ODC-BY).
- Rejected: XNLI, frenchNLI, SciQ, RACE, LogiQA 2, FQuAD, `pngwn/system-one-decisions` and the CJJones drone telemetry (non-commercial licenses), `SargeDev/jev-distill-corpus-v3` (model-distilled labels, mixed upstream licenses), PIQA (no clear license), the gated `Salesforce/xlam-function-calling-60k` and its re-licensed mirrors. Details in `DATA_LICENSES.md`.

## Tooling facts
- mlx-rs 0.32.0 (crate published 2026-09-12; repo `oxiglade/mlx-rs`):
  - It has `fast::scaled_dot_product_attention`, `fast::rope`, `fast::layer_norm`, `nn::value_and_grad` and `compile_with_state`.
  - It has no Muon and no transformer training example.
- MLX 0.32.3 was released 2026-09-29. Building it from source needs Xcode's Metal toolchain (`xcodebuild -downloadComponent MetalToolchain`).
