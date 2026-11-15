# s6-nano v2 (a tool router that reads descriptions): design study

Written on 15/11/2026, after `docs/s6-nano2-study.md` and the draft of a narrow v2 router (odds put at 10 to 20% by its author). Read only: the repository, its docs, its run folders and the web. Nothing was trained, nothing was downloaded beyond metadata (the Hugging Face and GitHub APIs, for licences and file sizes), and no test file of any source was opened. Text marked **[Inference]** is reasoning, not a sourced fact. "Not verified" means looked for and not confirmed.

## 0. Summary

1. **The one job.** "Given a user request and a list of up to 7 tools, each a name and one sentence, pick the tool that should handle it, or say *none of these*; below a confidence threshold, ask the user." Tools differ per application and are given at run time; the test uses tools that no training item contains.
2. **Why this job plays to the architecture, and v1's jobs did not.** In the s2 graph each option is one marker vector, computed once from the question and its own text and cached with the static stream; the request is the state; one or two fusion blocks let each option marker cross-attend to the request's tokens (`docs/SPEC-S2.md` §2). For a router that means: the tool list is encoded once per application, each request costs a short state encoding plus 8 small cross-attentions, options are permutation-equivariant by construction, and an option the model never saw is just text. A fixed-label linear model cannot take a new tool at run time at all. On v1's bench the options were the same few strings for every item of a task, so a per-option bias, i.e. the per-question prior, was the cheapest solution, and the model found it (§3).
3. **The bar is not a linear model, it is a static embedding with cosine.** That is what a practitioner uses for run-time tool lists, and there is a published number in exactly this setting: on 10 never-seen tools offered at once, zero-shot cosine between query and tool text scores **0.76 with potion-base-2M, 0.82 with potion-4M, 0.86 with potion-8M**; a tuned static router on potion-8M (7.66 MB, MIT) reaches 0.890, 0.933 on 5-tool slates (stanley-nv/toolrouter report, October 2026). On MetaTool with about 10 candidates, all-MiniLM-L6-v2 cosine (22M) has a recall@1 of 0.716 and BM25 0.397 (OATS, March 2026). potion-base-2M is 1.89M parameters, 7.56 MB in f32, about 1 MB packed to 4 bits; s6-nano is 715 KB.
4. **Design changes that follow from v1's collapse:** every training item has its own option set; the same request appears with different option sets and different correct answers (gold removed → *none*; gold renamed; description rewritten); hard negatives among options, filtered for false negatives by the teacher; a cap per tool so that no tool name carries a prior; hard gold targets, no 50/50 soft teacher mix; one job, no games, no general decisions.
5. **Cost.** About 185 working minutes of GPU (two 20-minute pilots, nano and micro, with a pre-registered abort rule; a 120-minute main run; calibration, export, evaluation), about 65 CPU minutes, and about 5,000 calls to the GLM teacher (about 11 hours of remote labelling, no GPU).
6. **Odds.** About 50% that it beats potion-2M cosine by 3 points on held-out tools of the training families; about 25% that it also matches it on MetaTool (a family never trained on); about 30% that it clears every gate of the draft pre-registration; about 25% that the pilot rule stops the run after 45 minutes, before the main budget is spent.

## 1. What v1 and the v2 draft leave open

- v1 (s6-nano, 1,486,225 ternary parameters, 715,136-byte GGUF) gives one answer to at least 95% of the items on 17 of the 20 bench tasks and ties the per-question prior (0.4805 against 0.4815; `docs/s6-nano2-study.md` §0). Its targets were 0.4% of the draws. On MASSIVE English, where every item has its own 8 intents drawn at random, it is not constant: 0.4255 against a chance of 0.125, no position bias (study §3.2). That is the one v1 measurement where options varied per item, and it is the only one where the model visibly read them, weakly.
- The v2 draft (`s6-nano2-prereg-draft.md`) keeps fixed-label sources (MASSIVE, When2Call's 4 classes, Banking77) in its gates, where a 715 KB hashed n-gram logistic regression reaches 0.947 on MASSIVE validation (study §5). Its G3 asks the model to come within 0.05 of that on the linear model's home ground; its author put the joint odds at 10 to 20%.
- **[Inference]** The useful question is narrower than the draft's: is there any job where this graph beats what a practitioner would actually use? For run-time option lists that is not a linear classifier (it cannot represent a label it was not trained on) but a sentence embedding with cosine. The rest of this study is about that comparison.

## 2. Question 1: where free-text options beat a fixed-label model

### 2.1 What the architecture gives (repository facts)

| property | where | what it means for a router |
|---|---|---|
| static stream cached per (type, instruction, options), LRU of 16 | `docs/SPEC-S2.md` §2, "s2 fast path" | the tool list of an application is encoded once; each request costs only its own state |
| each option attends to the question and its own text only; options never attend to each other | SPEC-S2 §2 (mask), §1 item 2 | a tool's representation does not depend on which other tools are listed; shuffling the list shuffles the probabilities exactly; no position bias is possible |
| fusion: the option marker cross-attends to every request token (2 blocks for s6-nano) | SPEC-S2 §2, `runs/recipe/configs/s6-nano.json` (`fusion_layers` 2) | late interaction: richer than one-vector cosine (the option queries the request's tokens), cheaper than a full cross-encoder |
| head window 160 tokens for question plus all options; each option capped at 48 tokens; options shrunk evenly past the budget | `crates/sky-infer/src/s2/prompt.rs` 102–177 | 8 options of "name: one sentence" may exceed 160 tokens with a 2,048-token BPE; the window can grow without new parameters (RoPE, no position table; amendment 2 of `docs/s6-gen-prereg.md` did it for the state) |
| calibrated softmax over options, a temperature per exit and type | SPEC-S2 §2; v1 ECE 0.025 on in-distribution sources | a calibrated *none* option and a threshold below which the router asks the user |
| early exit; exit 0 as good as full depth on language in v1 | study §2.2 | a 4× cheaper path is available, reported |
| WASM: s6-nano answers a short game frame (two questions) in 445 µs (SIMD128) and 358 µs (relaxed SIMD) under Node on the M5 | `BENCH.md`, skycmd-s6 WASM table | **[Inference]** a request of about 30 tokens with a cached tool list should take 0.3 to 1 ms in WASM; long JSON states took 15 ms, but a router's state is short |

### 2.2 What the 2025–2026 literature and products say

| source (date) | setting | numbers | what it says for v2 |
|---|---|---|---|
| stanley-nv/toolrouter, model card and `docs/REPORT.md` (updated 7 Oct 2026; MIT) https://huggingface.co/stanley-nv/toolrouter , https://huggingface.co/stanley-nv/toolrouter/blob/main/docs/REPORT.md | query + list of tools (name, description), returns per-tool probabilities; frozen potion-base-8M int8 embeddings, hand-set cosine weights + lexical overlap + a trained identity head for 7 fixed tools; 7.66 MB; 0.10–0.16 ms per request (CPU, 10 tools) | 10 never-seen tools all offered: **0.890**; random 5-tool slates 0.933; reworded descriptions 0.780; name only 0.750; descriptions shifted onto the wrong tools 0.060. **Zero-shot cosine on the unseen tools: potion-2M 0.76, 4M 0.82, 8M 0.86.** About 210 tools at once: 0.621 / 0.490. Confidence ≥ 0.95 → 0.982 correct. Learned projections on 23 or 179 tools scored at or below zero-shot; fine-tuning on ToolRet: 0.94 → 0.75 on chat-style unseen tools; MaxSim late interaction "worse" | the closest existing product to v2, and the bar. Its test is tiny (10 tools, 100 queries, LLM-written) and its author says some choices were made after reading it. It also says the hard part: few training tools overfit; descriptions carry 10–15 points |
| stanley-nv/tool-routing-data (CC-BY-4.0 tag; 7 Oct 2026) https://huggingface.co/datasets/stanley-nv/tool-routing-data | the data of the router above | `tools`: train 179 tools (2,409 queries), validation 14 (140), test 10 (100); all text LLM-written, labels not checked by humans; the card asks users to check the generating model's terms before training other models | usable as an external **evaluation-only** probe with a published potion-2M number on the same items |
| OATS, arXiv 2603.13426 (13 Mar 2026) https://arxiv.org/html/2603.13426v1 | tool selection without an LLM, ~10 candidate tools per MetaTool query; query-level 70/30 split (not tool-disjoint) | MetaTool R@1: all-MiniLM-L6-v2 cosine 0.716, BM25 0.397; ToolBench R@1: 0.382 and 0.392; "similar choices" subtask: BM25 42.7%, MiniLM 66.4%; 3.7–4.6 ms p50 on one CPU core | lexical matching is as good as a 22M embedder on ToolBench and far worse on MetaTool, whose queries paraphrase the tool. A model that learns mostly lexical matching will lose on MetaTool |
| ToolRet, arXiv 2503.01763 (Mar 2025) https://arxiv.org/html/2503.01763v1 ; queries https://huggingface.co/datasets/mangopy/ToolRet-Queries | retrieval over a 44k-tool corpus, 7,961 queries from 35 sources | best embedder NV-Embed-v1 nDCG@10 33.83 (42.71 with instructions); ColBERTv2 below BM25 | retrieval over thousands of tools is not the N ≤ 8 job; the queries file carries no licence on its card and aggregates sources with mixed terms: not used |
| Tool-Embed / "Tools are under-documented", arXiv 2510.22670; ToolDreamer, arXiv 2510.19791 (search snippets, Oct 2025) | tool retrieval | BM25 and OpenSearch match MiniLM and text-embedding-3-small on some sets; document expansion (richer descriptions) helps | description rewrites are a training and an inference lever |
| "Training Small Language Models for Transferable Tool Selection", arXiv 2610.31483 (22 Oct 2026) https://arxiv.org/html/2610.31483v1 | 1.7–8B models trained on ToolBench G3, tested on G2, Gorilla, CRAFT, TaskBench without retraining | 1.7B 37.6% exact-set success on G3 test; licences table: ToolBench Apache-2.0, Gorilla Apache-2.0, TaskBench MIT, CRAFT unspecified | transfer to unseen tool inventories is the open problem even at 1–8B; multi-tool planning is out of scope for v2 |
| MetaTool, arXiv 2310.03128; repository MIT (GitHub API, 15/11/2026) https://github.com/HowieHwong/MetaTool | 21,127 queries (20,630 single-tool) over about 199 tools (OpenAI plugins); `plugin_des.json` holds the descriptions | LLMs "still struggle to select tools" | an out-of-family, tool-level test v2 never trains on; its queries were generated by an LLM from the descriptions, which favours paraphrase-aware models |
| BFCL v2 (Apache-2.0; HF dataset modified 29 Apr 2026) https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard | `multiple`: 200 questions, pick one of 2–4 functions; `irrelevance`: 240, call none; `live_multiple` 1,053; `live_irrelevance` 882 | — | pure selection categories exist and are small; reported, not gated (2–4 options is easy) |
| When2Call (CC-BY-4.0) https://github.com/NVIDIA/When2Call , arXiv 2504.18851 (NAACL 2025) | tool call / ask / cannot answer / direct answer | acc-norm of untrained instruct models: Llama 3.2 1B 45.1%, Qwen 2.5 0.5B 53.5%, Llama 3.1 70B 46.1%; tool hallucination 20–67% | small LLMs are not a strong bar for "should I call a tool"; the data reformat to "which tool, or none" |
| PIE, arXiv 2305.14827 (EMNLP 2023) | zero-shot intent by label name, 5-way episodes | L-SBERT-paraphrase: Banking77 86.9, HWU64 83.6, CLINC150 94.3; PIE 92.4 / 88.8 / 97.6; with 25–50 intents at once 68.7–83.1 | the bar of a 110M bi-encoder on zero-shot intents in small slates |
| BTZSC, arXiv 2603.11991 (12 Mar 2026) https://arxiv.org/html/2603.11991 | zero-shot classification over all labels from label descriptions | intent macro-F1 (Banking77, MASSIVE, biasframes): all-MiniLM-L6-v2 0.41; ms-marco-MiniLM-L6 reranker 0.30; base NLI cross-encoders 0.30; gemma-3-270m 0.13; best 0.70 (Qwen3-Reranker-8B) | over a full label set even 22–110M models are weak; slates of ≤ 8 are what small models can do |
| potion-base-2M card (MIT) https://huggingface.co/minishlab/potion-base-2M ; HF API 15/11/2026 | static embeddings, mean of token vectors | 1.89M parameters, `model.safetensors` 7,559,256 bytes (f32, 64 dimensions); MTEB average 47.55 (classification 64.13, retrieval 22.99) | the matched-size embedding baseline |
| model2vec 0.10.0 (11 Oct 2026) https://minish.ai/blog/2026-10-12-model2vec-0100/ | — | new `StaticModelForPairSimilarity` (InfoNCE, temperature 0.05) for query–document pairs; token dropout | a static model can be **fine-tuned on the same (request, tool) pairs**: the fair trained baseline |
| model2vec-js https://github.com/ShakirasHips/model2vec-js | browser/Node port, no dependency | potion-base-8M packed to 4 bits: 4.2 MB, scores "82.0% / 92.8%" at full size against "81.4% / 92.2%" packed (the README does not say on what) | 4-bit packing costs about a point; potion-2M at 4 bits is about 1 MB |
| ToolScout (bge-small, 33M), TinyToolRouter-0.6B (HF cards, 2026) | self-reported, n = 50–70 | unseen-tool top-1 98.6% (ToolScout); TinyToolRouter unseen schema 0.850 → 0.867 | small, easy tests; context only |

### 2.3 Where a free-text-option model has a real advantage, and where it does not

- **Label sets that change at run time (yes).** Tool lists per application, intents described by a sentence. A fixed-label linear model cannot answer; a pair model (request features × option features) can, and is a fair size-matched trained baseline (`pair-lr` of the v2 draft).
- **Option-order invariance (yes, but free).** The graph is equivariant by construction; any embedding-cosine baseline is too. Not a differentiator against cosine; a differentiator against small LLMs (selection bias of LLM multiple choice).
- **Key-name invariance (measurable).** Whether the model follows the description or the name can be tested directly: mask the names, swap descriptions (the toolrouter's sanity check gives 0.060 when descriptions are shifted, i.e. it follows descriptions).
- **Calibrated abstention (yes, likely the main edge).** Cosine scores need a threshold for *none*, fitted on validation, and are not probabilities; the router can carry *none of these* as an option trained with counterfactual slates (gold removed), and a temperature. v1 was calibrated in distribution (ECE 0.025).
- **Decisions over structured state (yes, but not this job).** The games and secrets v2 are the evidence; it does not carry to language routing.
- **Semantic knowledge (no, the main risk).** potion-2M is distilled from a sentence encoder trained on very large pair corpora; s6-nano learned its words from 244 MB of MLM text with a 2,048-token vocabulary. **[Inference]** On paraphrastic queries (MetaTool) the embedding will likely win; on queries that share words or close forms with the tool text (glaive, ToolACE, Seal-Tools, API-Bank), a trained late-interaction model has a real chance.

### 2.4 The baseline number to beat, and how to measure it cheaply

- **Published:** zero-shot potion-2M cosine 0.76 on 10 unseen tools offered at once (toolrouter report; n 100, LLM-written queries, no *none* option). That is the order of magnitude for an 8-option slate.
- **On the v2 test (to be measured, CPU only, minutes):** model2vec 0.10.0 with `potion-base-2M` (7.56 MB download), encode every request and every option text of the validation slates, cosine, argmax; *none* chosen when the best cosine is below a threshold fitted on validation; a softmax temperature fitted on validation for the ECE. A slate file of 10,000 items has about 90,000 texts: seconds to encode. The same script gives the 4-bit packed variant (about 1 MB), the size-cut fine-tuned variant (≤ the model's bytes), BM25 and all-MiniLM-L6-v2 for reference. It is rehearsed on validation before the freeze, as in `docs/s6-secrets2-prereg.md` amendment 6.

## 3. Question 2: why v1 collapsed, in terms that point at the fix

| cause (repository evidence) | mechanism | fix in v2 | source for the practice |
|---|---|---|---|
| the bench's tasks were 0.4% of the draws; under two passes per question (study §2.3) | small tasks in a large mixture are starved; a prior is learned before the inputs | one job, 100% of the draws; each request seen in several option sets; no games, no general decisions | examples-proportional mixing with a cap (T5, arXiv 1910.10683, §3.5.2; not re-opened here) |
| on the bench the options were the same strings for every item of a task (choice, noul, 4- or 5-level score) | the option marker vector is constant per task, so a per-option bias is a one-parameter solution that already scores the prior (0.4815) | the options differ in every item; the gold varies with the option set; no tool name may appear as gold more than a fixed number of times | episodic training with random distractors and varied name/description styles (toolrouter card); prototypical-network episodes (arXiv 1703.05175; not re-opened) |
| 29 heterogeneous sets, soft teacher targets mixed 0.5 gold + 0.5 teacher; teacher agreement 68.8% on When2Call (study §3.3) | soft targets from a disagreeing teacher flatten the margin the model needs to read; the filter keeps only labels the gold agrees with, so the teacher adds little | hard gold targets from the datasets' own calls; the teacher is used to *clean* option sets (false negatives) and to *rewrite* descriptions, never as a soft target | denoising hard negatives with a stronger scorer (RocketQA, arXiv 2010.08191; not re-opened) |
| When2Call: tool_call 13.9% of training, 35% of test; the model predicts it 39 times in 2,000 | class prior shift | *none* gold share fixed by construction (20% of slates) in every split; the prior cannot help | majority-label bias of classifiers and its correction (arXiv 2102.09690; not re-opened) |
| accuracy fell with the length of the tool list in When2Call (0.53 → 0.32–0.39, study §3.3) although truncation was 0.2% | reading a request against several descriptions was never the target of enough draws | that is the whole training signal now; slates of 2 to 8 options with hard negatives | hard negatives in tool selection: the "similar choices" subtask of MetaTool; toolrouter's 1.3k hard-case queries (+3 points) |
| micro read the state on 13 of 20 tasks, nano on 3, at equal exposure (study §2.4) | capacity helps sample efficiency | a 20-minute pilot of each size on the same data, with a mechanical rule (prereg §5) | — |

Training practices that make a cross-encoder-like model attend to the option text, with their evidence:

1. **Counterfactual slates.** The same request with the gold removed (gold becomes *none*), with the gold's description rewritten, with the gold's name masked (`tool_3`), and with a different correct tool present. Only the options change, so only reading them lowers the loss. **[Inference]** from the collapse mechanism above; the toolrouter's reworded/name-only/shifted tests measure the same invariances at test time.
2. **Hard negatives among options, filtered.** Nearest tools by an embedding and the conversation's own co-listed tools; a stronger scorer removes distractors that would also serve the request (false negatives), the RocketQA lesson. Here the scorer is the GLM teacher.
3. **Many distinct tools, a cap per tool.** The toolrouter found that learning on 23 or 179 tools scored at or below zero-shot; "Preserving zero-shot capability in supervised fine-tuning" (NAACL 2025 Findings, https://aclanthology.org/2025.findings-naacl.315.pdf) finds that fine-tuned models lose zero-shot ability even with label descriptions. ToolACE alone has 26,507 APIs; glaive, Seal-Tools and API-Bank add thousands. A cap of 30 requests per tool cluster keeps frequent glaive tools (stock prices, BMI, currency) from becoming priors.
4. **Description diversity.** Label-description training helps zero-shot classification (arXiv 2305.02239); richer tool documents help retrieval (arXiv 2510.22670); descriptions are worth 10–15 points in the toolrouter. One teacher rewrite per training tool.
5. **Option dropout of length.** Slates of 2 to 8 options in training, 8 at test, so the model does not learn a slate size.
6. **Balanced sampling by group** (tools, intents, When2Call) with fixed weights, and the *none* share fixed per group.

## 4. Question 3: the narrow v2

### 4.1 The job, in a sentence a user understands

"Give it a user's message and the tools your app has, each with a name and one sentence; it says which tool should handle the message, or that none fits, with a probability you can threshold to ask the user instead." English first; French requests against English tool text are trained and reported, not promised.

### 4.2 Data (licences checked on 15/11/2026 through the Hugging Face and GitHub APIs)

| source | licence | size on the hub | use | expected items |
|---|---|---|---|---|
| glaiveai/glaive-function-calling-v2 | Apache-2.0 | one JSON, 271,190,065 bytes, commit `e7f4b645…` | train / val / **test (tool-disjoint)**; fetcher exists (`crates/sky-data/src/commands.rs`): 24,042 deduplicated first turns, 8,674 with a call (`DATA_LICENSES.md`) | ≈ 8,700 requests with a call |
| NousResearch/hermes-function-calling-v1 | Apache-2.0 | 66.8 MB (five JSON) | same; fetcher exists; overlaps glaive (`glaive-function-calling-5k`) → deduplicated | ≈ 2,300 |
| Team-ACE/ToolACE | Apache-2.0 | `data.json` 37,154,735 bytes, commit `6bda777c…` | same; tool calls written as Python-like strings, function names with spaces; a third party found two default values that look like live keys (to be masked) | ≈ 10,000 rows, a 26,507-API pool (subset public) |
| Seal-Tools (fairyshine/Seal-Tools) | Apache-2.0 on GitHub; the HF mirror casey-martin/Seal-Tools is tagged MIT (17.8 MB) | GitHub repository | same; multi-tool items: first call, the other called tools never used as distractors | 4,076 tools, ≈ 14,000 instances |
| liminghao1630/API-Bank | MIT on the hub (the GitHub licence not confirmed) | 112.8 MB | training data only (lv1/lv2: 2,138 APIs) | ≈ 2,000–4,000 calls |
| nvidia/When2Call | CC-BY-4.0 | 63.4 MB | train/val: tool_call and request_for_info → the called tool; cannot_answer and direct_answer → *none*; test/mcq reformatted, **reported** (read in v1) | 20,108 train (on disk) |
| MASSIVE en + fr (`data/commands/massive.jsonl`) | CC-BY-4.0 | on disk | intents as tools ("set_alarm: set an alarm for a given time", one teacher description each); 14 scenarios trained, 4 held out (reported) | ≈ 15,000 en + fr train utterances after the hold-out |
| PolyAI/banking77 (`data/decisions/banking77.jsonl`) | CC-BY-4.0 | on disk | intents as tools, train only (its official test is in our train) | ≈ 10,000 |
| clinc/clinc_oos | CC-BY-3.0 | 1.38 MB, commit `155b9c71…` | **test only**: 150 intents never trained, out-of-scope queries as *none* | test 5,500 (`plus`) |
| MetaTool (HowieHwong/MetaTool) | MIT (repository) | ≈ 30 MB repository | **test only**, never trained: 8-option slates over its ~199 tools; queries generated by an LLM (its terms apply to the text) | 20,630 single-tool queries |
| gorilla-llm/Berkeley-Function-Calling-Leaderboard | Apache-2.0 | 11.7 MB | test only, reported: `multiple` (200), `irrelevance` (240) | 440 |
| stanley-nv/tool-routing-data | CC-BY-4.0 tag, with the card's caution | 1.9 MB | test only, reported: `tools/test`, 10 tools, 100 queries (published potion-2M cosine 0.76) | 100 |
| HWU64, SLURP, xLAM, ToolRet | — | — | not used: HWU64 and SLURP are MASSIVE's ancestors (and SLURP NC); xLAM is gated (`DATA_LICENSES.md` "Rejected"); ToolRet has no licence on its card | — |

### 4.3 Splits that measure reading, not memorising

- **Tool clusters.** Tools are grouped by union-find over (a) the same normalised name (lowercase, `get_`/`fetch_`/`retrieve_` and the like stripped, separators collapsed) and (b) description cosine ≥ 0.85 under a fixed embedding (all-MiniLM-L6-v2, used here as a clustering tool only). Clusters go to train / val / test 85 / 5 / 10 by sha256 of the cluster's smallest normalised name with a seed. A request goes to its gold tool's split; its slate uses tools of the same split only; duplicate requests across splits go to the test side.
- **[Inference]** Near-duplicate tools ("get_weather" against "get_current_weather") are the leakage that would make an unseen-tool test a seen-tool test; the clustering is the guard, and its counts are reported.

### 4.4 Option sets (slates)

- Training: each request yields 4 slates: (1) its conversation's own tools plus random fill, gold present; (2) hard distractors (nearest by potion-2M cosine among other clusters of the split, after the teacher's false-negative filter), gold present; (3) slate 1 or 2 with the gold removed → *none*; (4) slate 1 with the gold's name masked and its description replaced by the teacher's rewrite. Slates have 2 to 8 options including *none*, which is always the last string and always present. *None* is gold in 20% of slates per group by construction.
- Test: 8 options (up to 7 tools + *none*): the conversation's own tools, then random fill from the test split; 20% gold-removed slates; distractors within description cosine 0.8 (MiniLM) or with the same name stem as the gold excluded, mechanically, with no teacher. Not mined with any compared system, so that no baseline is tested on its own hardest cases. A "hard" slice mined with MiniLM is reported apart.

### 4.5 Teacher calls (GLM-5.3-flash through `sky-data teacher`, 20 items per call; measured throughput about 2.5 labels per second, `docs/s6-gen-prereg.md` amendment 1)

| job | items | calls | time | required |
|---|---:|---:|---:|---|
| one description per intent from its name (MASSIVE 60, Banking77 77, CLINC150 150) | ≈ 290 | ≈ 15 | minutes | yes (CLINC's descriptions are written from names alone, before any test read) |
| false-negative filter: "which of these options could also handle the request?" on the hard slate of every training and validation request | ≈ 75,000 | ≈ 3,750 | ≈ 8.3 h | yes |
| one rewrite per training tool description | ≈ 25,000 | ≈ 1,250 | ≈ 2.8 h | recommended |
| one paraphrase per training tool request | ≈ 35,000 | ≈ 1,750 | ≈ 3.9 h | optional (open choice) |

The teacher never sees a test request.

### 4.6 Size of the training set that fits the budget

- Requests: ≈ 35,000 tool requests (after the per-cluster cap) + ≈ 15,000 MASSIVE + ≈ 10,000 Banking77 + ≈ 15,000 When2Call ≈ 75,000; × 4 slates ≈ 300,000 records.
- v1 drew 2.55 million questions in 90 working minutes with long states (study §2.2), about 28,000 per minute. Router states are short and heads longer; **[Inference]** a similar rate gives ≈ 3.4 million draws in 120 minutes, about 11 passes per record and 45 per request. The secrets v2 success ran about 10 passes. The paraphrase job, if taken, lowers the passes per request by a third.

## 5. Question 4: an evaluation that would convince a reader

- **Tests:** T-tools (fresh, tool-disjoint clusters of glaive/hermes/ToolACE/Seal-Tools, 2,000 slates, gated); T-metatool (never-trained family, 2,000 slates, reported with a goal); T-clinc (zero-shot intents by description with out-of-scope as *none*, 2,000, reported with a goal); toolrouter `tools/test`, BFCL `multiple` and `irrelevance`, When2Call reformatted, MASSIVE held-out scenarios en/fr (reported).
- **Baselines:** potion-2M cosine as published (7.56 MB f32) and 4-bit packed (~1 MB); potion-2M fine-tuned on the same pairs with model2vec 0.10.0 `StaticModelForPairSimilarity`, cut to the model's bytes (`static-ft`); `pair-lr` at the model's bytes (v2 draft §6); BM25; all-MiniLM-L6-v2 cosine (22M, 90.9 MB, reference); toolrouter (7.66 MB, reference); s6-nano v1; a name-frequency prior (the option whose name was gold most often in training).
- **Not constant:** the share of *none* answers within 0.10 of the gold share (0.20); the most frequent predicted tool name at most twice its gold share; on gold-removed pairs, the answer changes on at least 80% of the requests answered correctly with the gold present. v1 showed ≥ 95% one answer on 17 of 20 tasks.
- **Reading the option text:** names masked (accuracy drop reported); descriptions swapped between gold and a distractor (share of answers that follow the description, reported; the toolrouter falls to 0.060 against the original names).
- **Calibration:** ECE (15 bins, `sky-metrics`) on T-tools; risk–coverage curve; accuracy at the stored ask-threshold chosen on validation.
- **Latency:** WASM (Node, SIMD128 and relaxed builds, M5) median per request with the tool list cached, on 200 T-tools requests, plus the first call that encodes the tool list; native median; parity.
- **Demo for the site (a tool-router playground):** a text area with up to 7 tools, one per line as `name: description`, editable, with presets (smart home, developer tools, shop, a French agenda app); a request box; bars of probability for every tool and for *none*, updated as the visitor types; a threshold slider below which the result reads "ask the user"; two buttons that show the reading checks live ("mask names", "swap two descriptions"); the latency of the first call (tools encoded) and of the following ones; the verdict of the gate and its numbers in the first line of the panel; the model file served from `site/public/models/`, no other network call.

## 6. Question 5: honest odds, and what "useful" is

- **Useful** = on T-tools (8 options, 20% *none*), top-1 accuracy at least 3 points above potion-2M cosine and above every size-matched trained baseline, ECE ≤ 0.05, not constant by the checks of §5, ≤ 2 ms per request in WASM. **[Inference]** With potion-2M cosine expected around 0.65 to 0.75 there (0.76 published on 10 tools without *none*; a threshold-based *none* costs points), useful means roughly 0.75 to 0.85.
- **Odds (judgement):**

| event | probability | why |
|---|---:|---|
| the pilot rule stops the run after 45 minutes | ≈ 25% | v1 read varied options only weakly (0.43 on MASSIVE en); a 20-minute pilot reaches about 87% of a final validation score in v1's curve |
| G1 on T-tools (≥ potion-2M cosine + 0.03 and > static-ft and pair-lr) | ≈ 50% | in-family tools share vocabulary with training; late interaction and trained *none* help; static-ft is an unknown, possibly strong bar |
| G2 not constant | ≈ 90% | the option sets vary by construction; v1 was not constant where options varied |
| G3 calibration | ≈ 80% | in-distribution ECE of v1 0.025; *none* trained |
| G4 latency | ≈ 90% | short states, cached tool lists |
| all gates | **≈ 30%** (range 20–40%) | correlated events |
| MetaTool goal (≥ potion-2M cosine) | ≈ 25% | paraphrastic queries favour distilled sentence embeddings |
| CLINC goal (≥ potion-2M cosine) | ≈ 30% | intent names in a domain never trained; MASSIVE and Banking77 overlap some of its concepts |

- **Expected GPU cost:** 0.25 × 45 + 0.75 × 185 ≈ 150 working minutes.

## 7. What could not be verified

- A potion-2M cosine number on any tool-disjoint benchmark other than the toolrouter's 10-tool test (n 100). OATS's MetaTool numbers use MiniLM on a query-level split, not unseen tools.
- The exact sizes and quality of the public ToolACE subset; Seal-Tools' licence on the hub mirror (MIT) against its GitHub repository (Apache-2.0); API-Bank's GitHub licence.
- The scores in the model2vec-js table (what "82.0% / 92.8%" measures).
- The terms of the generated text in MetaTool and tool-routing-data for training (both are used for evaluation only).
- RocketQA, T5 mixing, prototypical networks, "calibrate before use": cited from their known abstracts, not re-opened on this date.
- The token length of a "name: description" option under `tok-s6-2048` (to be measured by `s2-truncation` on the built records; the window rule of the draft decides).
- The WASM latency of s6-nano on a router request: estimated from the game-frame number, not measured.
- Whether s6-micro's 2.4 MB file and its latency on short requests make it acceptable; the pilot measures its accuracy only.

## 8. Downloads a run would need

| name | URL | size | licence |
|---|---|---:|---|
| glaive-function-calling-v2 | https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2 | 271.2 MB | Apache-2.0 |
| hermes-function-calling-v1 | https://huggingface.co/datasets/NousResearch/hermes-function-calling-v1 | 66.8 MB (all files; the three used ≈ 59 MB) | Apache-2.0 |
| ToolACE | https://huggingface.co/datasets/Team-ACE/ToolACE | 37.2 MB | Apache-2.0 |
| Seal-Tools | https://github.com/fairyshine/Seal-Tools (mirror https://huggingface.co/datasets/casey-martin/Seal-Tools, 17.8 MB) | ≈ 18–25 MB | Apache-2.0 (GitHub) |
| API-Bank | https://huggingface.co/datasets/liminghao1630/API-Bank | 112.8 MB (training files ≈ 108 MB) | MIT (hub tag) |
| When2Call | https://huggingface.co/datasets/nvidia/When2Call | 63.4 MB (on disk already as `data/decisions/when2call.jsonl`) | CC-BY-4.0 |
| CLINC150 | https://huggingface.co/datasets/clinc/clinc_oos | 1.4 MB | CC-BY-3.0 |
| MetaTool | https://github.com/HowieHwong/MetaTool | ≈ 30 MB | MIT (repository) |
| BFCL v2 | https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard | 11.7 MB (two files used, ≈ 0.5 MB) | Apache-2.0 |
| tool-routing-data | https://huggingface.co/datasets/stanley-nv/tool-routing-data | 1.9 MB | CC-BY-4.0 tag, with the card's caution |
| potion-base-2M | https://huggingface.co/minishlab/potion-base-2M | 7.6 MB (+ 0.9 MB tokenizer) | MIT |
| potion-base-8M (reference) | https://huggingface.co/minishlab/potion-base-8M | 30.2 MB | MIT |
| all-MiniLM-L6-v2 (clustering and reference) | https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2 | 90.9 MB (safetensors) | Apache-2.0 |
| toolrouter (reference) | https://huggingface.co/stanley-nv/toolrouter | 7.7 MB | MIT |
| model2vec 0.10.0, scikit-learn 1.9.1, bm25s, sentence-transformers | PyPI | — | MIT / BSD-3 / MIT / Apache-2.0 |

MASSIVE and Banking77 are on disk. Total new data ≈ 0.6 GB, models ≈ 0.14 GB.

## 9. What this study read

`docs/s6-nano2-study.md`, the v2 draft, `docs/s6-gen-prereg.md`, `docs/SPEC-S6.md` §15, `docs/SPEC-S2.md` §2, `docs/research-s6-usefulness.md` §3, §7, §8 and §9, `BENCH.md` (skycmd-s6, inference), `docs/s6-secrets2-prereg.md` with its Result, `docs/s6-cast2-prereg.md`, `DATA_LICENSES.md`, `crates/sky-infer/src/s2/prompt.rs`, `crates/sky-data/src/s4.rs` (`tool_routing`), `runs/recipe/configs/s6-nano.json`, the train split of `data/commands/massive.jsonl` (intent counts per scenario: 18 scenarios, 60 intents). No test record of any source.
