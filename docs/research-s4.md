# s4 research: small zero-shot deciders, intent baselines, LLM distillation and calibration

Collected 2026-10-12 from primary sources only (arXiv papers, official Hugging Face model cards, official GitHub READMEs). Each number links to its source. "Not stated" means the source does not give the number and none was guessed. Unless marked otherwise, arithmetic beyond the sources (chance rates, parameter estimates) is ours.

## 1. Smallest open zero-shot classifiers with arbitrary label names

- **MoritzLaurer NLI zero-shot family.** Method: [Laurer et al. 2023, arXiv 2312.17543](https://arxiv.org/abs/2312.17543) (abstract). NLI used as a universal task, trained on 33 datasets with 389 classes; reports +9.4% zero-shot improvement.
  - `deberta-v3-xsmall-zeroshot-v1.1-all-33`: 70.8M total params, of which 22M are backbone and the rest is the vocabulary embedding, and the card says "142 MB small" ([card](https://huggingface.co/MoritzLaurer/deberta-v3-xsmall-zeroshot-v1.1-all-33)). The card says it ran **no zero-shot evaluation**. Its Banking77 (0.627) and MASSIVE (0.734) accuracies are on datasets it was trained on, so they are not zero-shot.
  - `deberta-v3-base-zeroshot-v2.0` (~0.2B): held-out zero-shot f1_macro is **Banking77 0.446 and MASSIVE 0.520, with a 28-dataset mean of 0.643**. `deberta-v3-large-zeroshot-v2.0` scores 0.567, 0.518 and 0.673. `bart-large-mnli` scores 0.312, 0.430 and 0.497. The card says "no data from these datasets was added in the training mix" ([card, results table](https://huggingface.co/MoritzLaurer/deberta-v3-base-zeroshot-v2.0)).
  - `ModernBERT-base-zeroshot-v2.0` (0.1B): Banking77 acc 0.727 / F1 0.746, MASSIVE acc 0.788 / F1 0.738, mean acc 0.831 ([card](https://huggingface.co/MoritzLaurer/ModernBERT-base-zeroshot-v2.0)). `ModernBERT-large-zeroshot-v2.0` (0.4B): Banking77 0.744 and MASSIVE 0.800 acc ([card](https://huggingface.co/MoritzLaurer/ModernBERT-large-zeroshot-v2.0)). **Neither card says whether these datasets were held out.** The numbers sit far above DeBERTa's held-out 0.446, and the card says ModernBERT is "slightly worse than DeBERTav3 on average". So treat them as probably in-distribution (unverified). The card also says it is "multiple times faster" than DeBERTa-v3.
- **tasksource `deberta-small-long-nli`**: about 0.1B params (card size field), DeBERTa-v3-small, 1680-token context, trained on 600 tasks for 250k steps. Reported numbers include `zero-shot-label-nli` 70.0 and ANLI r1/r2/r3 57.2/46.1/47.2. **No Banking77 or MASSIVE numbers** ([card](https://huggingface.co/tasksource/deberta-small-long-nli)).
- **GLiClass (Knowledgator).** One forward pass over the text and all labels, which is the closest architectural analogue to s4. Paper: [arXiv 2508.07662](https://arxiv.org/html/2508.07662), Table 2 (params) and Table 3 (zero-shot F1).
  - `gliclass-edge-v3.0` 32.7M: banking 0.0272, massive 0.2414, avg 0.4900. The card lists 131 MB in F32 ([card](https://huggingface.co/knowledgator/gliclass-edge-v3.0)).
  - `gliclass-modern-base-v3.0` 151M: 0.3561, 0.3442, avg 0.5577.
  - `gliclass-base-v3.0` 187M: 0.4698, 0.5040, avg 0.6764.
  - `gliclass-large-v3.0` 439M: 0.5574, 0.5649, avg 0.7193.
  - Older `gliclass-small-v1.0` (144M): banking 0.1374, massive 0.2546, avg 0.5401; its `-lw` variant scores 0.1768, 0.318 and 0.5732 ([card](https://huggingface.co/knowledgator/gliclass-small-v1.0)).
- **Ettin encoders** ([Weller et al., arXiv 2507.11412](https://arxiv.org/html/2507.11412), Table 3). These are pretrained backbones, not zero-shot classifiers. Fine-tuned results:
  - Ettin-Enc-17m: GLUE avg 79.2, MNLI 79.5. For comparison, BERT-mini scores 76.4/77.2 and TinyBERT 77.0/80.9.
  - Ettin-Enc-32m: GLUE 83.5, MNLI 83.4. MiniLM-L12 scores 84.6/85.6.
  - Ettin-Enc-68m: GLUE 87.2, MNLI 87.0.
  - The 17M model has 7 layers, hidden size 256, a 50,368-token vocabulary and up to 8K context ([card](https://huggingface.co/jhu-clsp/ettin-encoder-17m)). Our estimate puts the embedding matrix at about 12.9M of the 17M params, leaving roughly 4M of transformer layers, which is about s4's size.
  - We found **no official Ettin zero-shot classifier**. One community NLI fine-tune of Ettin-68m exists ([AdamCodd/ettin-nli-classifier](https://huggingface.co/AdamCodd/ettin-nli-classifier)) with self-reported numbers only.

## 2. Zero-shot intent results (Banking77, MASSIVE)

**Caveat for s4:** every number below is **full-label-set classification**: 77-way for Banking77 and 59 or 60-way for MASSIVE. Chance is about 1.3% and 1.7%. Our setup (gold plus 7 random distractors) has 12.5% chance. A random 7-of-76 draw contains any specific confusable sibling intent only 9.2% of the time (7/76). **We found no published 8-candidate Banking77 or MASSIVE number**, so s4 scores cannot be placed directly on these scales. They only bound the difficulty from above.

- **BTZSC benchmark** ([arXiv 2603.11991](https://arxiv.org/html/2603.11991)): 22 datasets, label names or verbalizers only, macro F1 / accuracy. Per-dataset numbers are in Appendix D, Tables 5–6.

  | Model | B77 F1 | MASSIVE F1 |
  |---|---|---|
  | Qwen3-4B | 0.37 | 0.38 |
  | Qwen3-8B | 0.39 | 0.39 |
  | Llama-3.2-3B-Instruct | 0.39 | 0.37 |
  | gemma-3-1b-it | 0.10 | 0.08 |
  | gemma-3-270m-it | 0.00 | 0.00 |
  | deberta-v3-large-nli | 0.35 | 0.42 |
  | modernbert-base-nli | 0.10 | 0.25 |
  | all-MiniLM-L6-v2 (bi-encoder) | 0.43 | 0.33 |
  | gte-base-en-v1.5 (bi-encoder) | 0.66 | 0.51 |
  | Qwen3-Reranker-0.6B | 0.63 | 0.53 |

  The 22-dataset average F1 (Table 2) is 0.65 for Qwen3-4B, 0.72 for Qwen3-Reranker-8B and 0.37 for all-MiniLM-L6-v2. Takeaway: **with 77 labels in one prompt, a 4B LLM does worse than a base-size embedding model.**
- **Ganesh, Dozier and Seals** ([arXiv 2607.27421](https://arxiv.org/pdf/2607.27421)): 41 open-weight LLMs, label list in the prompt, accuracy on the **first 500 test examples** of each dataset.
  - MASSIVE (60 intents), from the supplementary table: Qwen2-7B-Instruct 0.718 (best), Llama-3.1-8B-Instruct 0.642, Qwen2.5-3B-Instruct 0.650, Llama-3.2-3B-Instruct 0.438, Qwen2.5-0.5B-Instruct 0.342, SmolLM2-360M-Instruct 0.088.
  - Banking77 across all models: mean 0.388, top 0.708 (Table 4).
  - The authors did not check that the first-500 subsets are representative (Limitations).
- **Few-shot and full-data references** ([Loukas et al., arXiv 2308.14634](https://arxiv.org/html/2308.14634), Table 4, Banking77 micro-F1):
  - GPT-4 3-shot: 83.1 with representative examples, 74.2 with random ones. GPT-3.5 1-shot: 75.2.
  - SetFit MPNet: 57.4 at 1-shot, 88.0 at 10-shot.
  - Full fine-tune: 94.0. These are not zero-shot.
- **MASSIVE full fine-tune** ([FitzGerald et al., arXiv 2204.08582](https://arxiv.org/abs/2204.08582), Table 3a and Table 8): en-US intent accuracy is XLM-R base 88.3 and mT5 encoder 89.0. The paper's "zero-shot" means cross-lingual transfer (train on en-US only, average 70.6 for XLM-R), not label-name zero-shot.

## 3. Distilling LLM labels; calibration and option bias

- **Hard plus soft targets.** [Hinton et al. 2015, arXiv 1503.02531](https://arxiv.org/pdf/1503.02531), Section 2, is the canonical recipe. It uses "a weighted average of two different objective functions": cross-entropy on temperature-T soft targets and cross-entropy on the correct labels. The best results generally came with "a considerably lower weight on the second objective". The soft term must be multiplied by T² because its gradients scale as 1/T². This matches s4's gold/teacher mix.
- **LLM annotations into small classifiers.**
  - [Wang et al. 2021, arXiv 2108.13487](https://arxiv.org/html/2108.13487) (Findings EMNLP): GPT-3 labels cost 50–96% less than human labels at equal downstream performance. The paper trains on **hard** GPT-3 labels and uses logits only as confidence for active relabeling: TREC rises from 77% to 80% at a $2.2 budget (Section 3.3.3).
  - [Pangakis and Wolken 2024, arXiv 2406.17633](https://arxiv.org/abs/2406.17633): on 14 tasks, classifiers fine-tuned on GPT-4 labels perform "comparably" to ones trained on human labels. The abstract gives no numbers.
  - [Li et al. 2024 (LLKD), arXiv 2411.08028](https://arxiv.org/html/2411.08028v1), Table 1: teacher confidence weights the samples while training uses argmax pseudo-labels. The student beats the LLaMA teacher on PubMed-RCT, 74.16 vs 67.09 accuracy.
  - **We found no primary source that directly compares soft and hard LLM option-probability targets for a small text classifier.** This remains open.
- **Contextual calibration.** [Zhao et al. 2021, arXiv 2102.09690](https://arxiv.org/abs/2102.09690) (ICML): estimate bias from a content-free input (for example "N/A") and fit parameters that make that prediction uniform. GPT-3 and GPT-2 average accuracy rises by up to 30.0 points absolute, with lower variance across prompts. [Batch Calibration, arXiv 2309.17249](https://arxiv.org/abs/2309.17249) (ICLR 2024) estimates bias from the batch itself. Its abstract gives no headline delta.
- **Selection bias in multiple choice.**
  - [Zheng et al., arXiv 2309.03882](https://arxiv.org/html/2309.03882) (ICLR 2024): the bias is mainly *token* bias toward option IDs, not position. On a balanced 1,000-sample MMLU set, llama-30B picks A/B/C/D 34.6/27.3/22.3/15.8% of the time. Bias measured as recall std (RStd) on 0-shot MMLU: gpt-3.5-turbo 5.5, llama-30B 8.5, falcon-7B 24.2. Removing option IDs cuts RStd but costs accuracy. PriDe estimates the prior from 5% of test samples by cyclic permutation and costs about ×1.15. It changes MMLU by −7.6 RStd and +1.2 accuracy, against −8.7/+4.9 for full cyclic permutation at ×n cost.
  - [Pezeshkpour and Hruschka, arXiv 2308.11483](https://arxiv.org/abs/2308.11483): reordering options causes gaps of about 13% to 75%.
  - [Robinson et al., arXiv 2210.12353](https://arxiv.org/abs/2210.12353) (ICLR 2023): letter scoring works only when the model has "multiple choice symbol binding" ability, which varies widely by model.
  - Implication: per-letter teacher calibration, as s4-mix already does, plus option-order permutation of teacher queries is supported by the literature.

## 4. Alternatives to compare in a future run

- **Fine-tune a small pretrained encoder on the same typed-decision data.** Encode options in the input, as s4 does, or as a GLiClass-style label-joint encoder.
  - Candidates: Ettin-Enc-17m (17M, GLUE 79.2) and Ettin-Enc-32m (32M, GLUE 83.5) ([Table 3](https://arxiv.org/html/2507.11412)), and BERT-Tiny (L=2, H=128; GLUE 64.2, MNLI-m 70.2 per [google-research/bert README](https://github.com/google-research/bert/blob/master/README.md)). The README states no parameter count for BERT-Tiny. Our estimate from its configuration is about 4.4M.
  - The only zero-shot evidence at this scale is gliclass-edge-v3.0 (32.7M), which is near zero on banking (0.027 F1). That suggests a small encoder alone does not handle 77-way fine-grained labels.
  - Expected 8-way accuracy is not published. It has to be measured.
- **Bi-encoder label embeddings** (cosine between text and label name). all-MiniLM-L6-v2 is 22.7M params, 384-dim, 256-token limit ([card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)). Zero-shot F1 is 0.43 on B77 and 0.33 on MASSIVE, rising to 0.66 and 0.51 for gte-base ([BTZSC Table 5](https://arxiv.org/html/2603.11991)).
  - Label embeddings can be cached, so the cost is one encoder pass per input regardless of option count.
  - The weakness is score/true-false questions, where the "label" carries little meaning on its own.
- **NLI cross-encoders.** The smallest documented models are the 70.8M DeBERTa-xsmall (no zero-shot eval) and the ~0.2B DeBERTa-base (B77 0.446). The cost is **one forward pass per option**. On an A6000 at batch size 1, deberta-v3-base-zeroshot-v2.0 drops from 24.55 ex/s with 1 label to 0.47 ex/s with 128 labels. gliclass-edge drops only from 103.81 to 82.64 ([arXiv 2508.07662, Table 6](https://arxiv.org/html/2508.07662)). For 8 options an NLI model needs 8 passes.
- **Browser or WASM latency:** none of the sources reports CPU, WASM or browser latency. All throughput figures above are GPU. This is not verifiable from primary sources and must be benchmarked locally.

## Comparison table

Zero-shot F1 on 77-way Banking77 (B77) and MASSIVE unless noted.

| Model | Params | Approach | B77 | MASSIVE | Avg | Source |
|---|---|---|---|---|---|---|
| s4 (ours) | ~4.6M ternary | joint encoder, options in input, distilled | n/a (8-way setup) | n/a | — | this repo |
| gliclass-edge-v3.0 | 32.7M | label-joint encoder | 0.027 | 0.241 | 0.490 (14 ds) | [2508.07662 T3](https://arxiv.org/html/2508.07662) |
| gliclass-small-v1.0 | 144M | label-joint encoder | 0.137 | 0.255 | 0.540 | [card](https://huggingface.co/knowledgator/gliclass-small-v1.0) |
| gliclass-base-v3.0 | 187M | label-joint encoder | 0.470 | 0.504 | 0.676 | [2508.07662 T3](https://arxiv.org/html/2508.07662) |
| deberta-v3-xsmall-zeroshot-v1.1 | 70.8M (22M backbone), 142 MB | NLI cross-encoder | not zero-shot (0.627 acc, trained on) | not zero-shot (0.734) | — | [card](https://huggingface.co/MoritzLaurer/deberta-v3-xsmall-zeroshot-v1.1-all-33) |
| deberta-v3-base-zeroshot-v2.0 | ~0.2B | NLI cross-encoder | 0.446 | 0.520 | 0.643 (28 ds) | [card](https://huggingface.co/MoritzLaurer/deberta-v3-base-zeroshot-v2.0) |
| ModernBERT-base-zeroshot-v2.0 | ~0.1B | NLI cross-encoder | 0.746 (protocol not stated) | 0.738 (protocol not stated) | 0.813 | [card](https://huggingface.co/MoritzLaurer/ModernBERT-base-zeroshot-v2.0) |
| tasksource deberta-small-long-nli | ~0.1B | NLI cross-encoder | not reported | not reported | — | [card](https://huggingface.co/tasksource/deberta-small-long-nli) |
| all-MiniLM-L6-v2 | 22.7M | bi-encoder | 0.43 | 0.33 | 0.37 (22 ds) | [2603.11991 T2/T5](https://arxiv.org/html/2603.11991) |
| gte-base-en-v1.5 | not stated in source | bi-encoder | 0.66 | 0.51 | — | [2603.11991 T5](https://arxiv.org/html/2603.11991) |
| Qwen3-Reranker-0.6B | 0.6B | reranker | 0.63 | 0.53 | 0.61 | [2603.11991](https://arxiv.org/html/2603.11991) |
| Qwen3-4B | 4B | LLM prompt | 0.37 | 0.38 | 0.65 | [2603.11991](https://arxiv.org/html/2603.11991) |
| Qwen2.5-3B-Instruct | 3B | LLM prompt (acc, first 500) | — | 0.650 | — | [2607.27421](https://arxiv.org/pdf/2607.27421) |
| GPT-4 (3-shot, not zero-shot) | not stated | LLM ICL | 83.1 micro-F1 | — | — | [2308.14634 T4](https://arxiv.org/html/2308.14634) |
| Full fine-tune (reference) | MPNet / XLM-R base | supervised | 94.0 | 88.3 (acc) | — | [2308.14634](https://arxiv.org/html/2308.14634), [2204.08582](https://arxiv.org/abs/2204.08582) |

## What this means for s4

- **Size:** the smallest documented zero-shot classifier with arbitrary labels is gliclass-edge at 32.7M, about 7× s4's parameter count, and it scores near chance on 77-way Banking77. Nothing published in s4's size class (~5M) does open-label zero-shot decisions. The only comparable-size reference is BERT-Tiny, about 4.4M by our estimate, and it is used only as a fine-tuning base.
- **Teacher ceiling:** with all 77 labels in the prompt, 3–8B LLMs reach 0.37–0.39 macro F1 on Banking77 and MASSIVE in BTZSC, and at most 0.708 (Banking77) and 0.718 (MASSIVE) accuracy in Ganesh et al., depending on protocol. The 8-candidate format is much easier, so teacher accuracy on our eval should be well above these figures. Report it measured, not assumed.
- **Fair comparison:** report s4 next to (a) gold-only fine-tunes of Ettin-17m and BERT-Tiny on the same data, (b) all-MiniLM-L6-v2 label-cosine on the same 8-way items, and (c) the teacher itself. All four must use identical candidate sets, and the report must state the 8-way protocol and its 12.5% chance level.
- **Calibration:** per-letter teacher bias is documented (PriDe, Zhao et al.). Keep the s4-mix calibration, and consider permuting option order in teacher queries so the distilled targets are not letter-biased.
- **Latency:** no primary source gives browser or WASM numbers for any of these models. A latency claim for s4 against them needs our own benchmark.
