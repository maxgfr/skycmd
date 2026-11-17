# Data licenses

Every dataset used by `sky-data` is public and ungated on the Hugging Face Hub. The produced
files live under `data/` (gitignored) and are not redistributed by this repository. Row counts
are the records written by `sky-data fetch-decisions` / `fetch-commands` / `fetch-text` /
`augment-paraphrases` with the default caps (seed 1234). Unless noted, every decision source is
split 95/5 into train/val by state, so all questions about one state land in the same split.

## Decision datasets (`data/decisions/*.jsonl`)

| dataset | license | URL | how it is used | rows used |
|---|---|---|---|---|
| n4ze3m/typed-decisions-synth | MIT | https://huggingface.co/datasets/n4ze3m/typed-decisions-synth | train + validation cases; one record per question; teacher soft labels as targets (gold label as fallback) | 25,859 |
| kgrozdanovski/assistant-decisions | Apache-2.0 | https://huggingface.co/datasets/kgrozdanovski/assistant-decisions | train + validation + test cases; one record per question; gold probability distributions as targets | 15,455 |
| pngwn/typed-decisions | CC-BY-SA-4.0 | https://huggingface.co/datasets/pngwn/typed-decisions | `train` + `cal` prompts (GPT-2 token ids) decoded to text; one record per question; `gold_probs` as soft targets (one-hot when absent). yes/no → noul, consecutive integers → score, else choice | 51,019 |
| google/boolq | CC-BY-SA-3.0 | https://huggingface.co/datasets/google/boolq | train + validation; noul, state = passage, question rephrased as a statement check | 12,697 |
| nyu-mll/glue (mnli) | MultiNLI: OANC terms, with some fiction under CC-BY-SA-3.0 / CC-BY-3.0 (see the MultiNLI paper / GLUE card) | https://huggingface.co/datasets/nyu-mll/glue | train; choice {entailment, neutral, contradiction} with short descriptions; balanced 60k sample | 60,000 |
| allenai/ai2_arc (ARC-Easy + ARC-Challenge) | CC-BY-SA-4.0 | https://huggingface.co/datasets/allenai/ai2_arc | all splits; choice with keys A..E and the answer texts as descriptions; state = question | 7,784 |
| tau/commonsense_qa | MIT | https://huggingface.co/datasets/tau/commonsense_qa | train + validation (test has no answers); choice A..E | 10,962 |
| PolyAI/banking77 | CC-BY-4.0 | https://huggingface.co/datasets/PolyAI/banking77 | train + test CSVs referenced by the dataset's loading script (github.com/PolyAI-LDN/task-specific-datasets); choice over the gold intent + 7 random distractors, intent names humanized | 13,083 |
| tasksource/tasksource-jev-typed-decisions (config `default`) | per row; only rows with `license_use == "commercial"` are used (Apache-2.0, MIT, CC-BY, CC-BY-SA, CC0, ODC-BY, BSD, … as declared per row) | https://huggingface.co/datasets/tasksource/tasksource-jev-typed-decisions | train split, 634,182 commercial rows; sources that duplicate a set we load directly (BoolQ, MNLI, ARC, CommonsenseQA + CoS-E, Banking77, PAWS-X fr, HellaSwag, SNLI, HelpSteer2, procedural typed decisions) or whose upstream is non-commercial / rejected (XNLI, SciQ, RACE, LogiQA, PIQA) are dropped → 492,119 eligible rows in 389 sources; cap 120k shared fairly between sources; soft targets kept; `src = tasksource:<source>` | 119,059 |
| tasksource/procedural-typed-decisions (config `all`) | Apache-2.0 | https://huggingface.co/datasets/tasksource/procedural-typed-decisions | train + validation; states sampled by hash, one record per question with the `answers` probabilities; cap 80k shared fairly between the 13 tasks; `src = procedural:<task>` | 80,000 |
| ZefanCai/Open-Jev (config `browser-drone-expansion-v1-redistributable`) | CC0-1.0 | https://huggingface.co/datasets/ZefanCai/Open-Jev | train + calibration + validation; **all 18,130 `drone-control-v1` rows kept** (no robot-control source exists in this config); the other 13 sources capped to 30k together, shared fairly; `src = open_jev:<source>` | 47,800 |
| Aditharavind/swarm-laya-decisions | Apache-2.0 | https://huggingface.co/datasets/Aditharavind/swarm-laya-decisions | all splits; Laya questions with the gold probability distributions (robot-swarm next action) | 6,580 |
| nvidia/HelpSteer2 | CC-BY-4.0 | https://huggingface.co/datasets/nvidia/HelpSteer2 | train + validation; one score question (levels 0–4 with descriptions) per attribute (helpfulness, correctness, coherence, complexity, verbosity); state = prompt (≤ 1,200 chars) + response (≤ 2,400 chars); grouped by prompt; 106,810 records label-balanced to 60k | 60,000 |
| google-research-datasets/paws-x (`fr`) | PAWS-X terms: free for any purpose, acknowledgement of Google appreciated | https://huggingface.co/datasets/google-research-datasets/paws-x | train + validation; noul "Do these two sentences mean the same thing?", state = the two French sentences | 48,864 |
| tblard/allocine | MIT | https://huggingface.co/datasets/tblard/allocine | train; noul "Is this review positive?" on French film reviews; cap 30k | 30,000 |
| nvidia/When2Call | CC-BY-4.0 | https://huggingface.co/datasets/nvidia/When2Call | choice {tool_call, request_for_info, cannot_answer, direct_answer} with descriptions; state = user request + tool names/descriptions. Train/val from `train_sft` + the chosen responses of `train_pref`, labelled from the response text (`<TOOLCALL>` → tool_call, else a heuristic that agrees with 93.7% of the mcq reference texts); the labelled `test/mcq` split is kept as `split = "test"` (3,630) | 24,776 |
| helmo/synthetic-typed-decisions | MIT | https://huggingface.co/datasets/helmo/synthetic-typed-decisions | train; Laya questions with gold labels (soft when given) | 6,574 |
| tasksource/synthetic-typed-decisions | Apache-2.0 | https://huggingface.co/datasets/tasksource/synthetic-typed-decisions | train + validation; Laya questions with the aggregated soft `answers` | 25,186 |
| m-lagnajit/calibrated-typed-decisions | CC-BY-4.0 | https://huggingface.co/datasets/m-lagnajit/calibrated-typed-decisions | train + validation; only rows whose per-row `source_license` is permissive (all of them: Apache-2.0 2,600, CC-BY-3.0 1,300, CC-BY-4.0 2,598); calibrated `teacher_probs` as soft targets; `src = calibrated:<source>` | 6,498 |
| stanfordnlp/snli | CC-BY-SA-4.0 | https://huggingface.co/datasets/stanfordnlp/snli | train; choice {entailment, neutral, contradiction} as for MNLI; cap 40k | 40,000 |
| Rowan/hellaswag | MIT | https://huggingface.co/datasets/Rowan/hellaswag | train + validation (test has no labels); choice A..D over the endings, state = activity + context; cap 30k | 30,000 |
| LocalLLaMA/typed-decisions (config `all`) | Apache-2.0 | https://huggingface.co/datasets/LocalLLaMA/typed-decisions | **test split only**, kept as an external benchmark (`split = "test"`), never trained on | 2,000 |
| AirsideLabs/notam-typed-decisions (configs `notam_multi` + `deel_subject`) | CC-BY-4.0 | https://huggingface.co/datasets/AirsideLabs/notam-typed-decisions | **evaluation only** (`split = "test"`); rows without a label (3,118) are skipped; `src = notam:<config>` | 9,051 |
| allenai/qasc | CC-BY-4.0 | https://huggingface.co/datasets/allenai/qasc | train + validation (the test split has no answer key); choice A..H, state = the two supporting facts then the question | 9,060 |
| agentlans/refusal-classifier-data (config `english_only`) | CC-BY-4.0 | https://huggingface.co/datasets/agentlans/refusal-classifier-data | train; noul "Does the assistant refuse or decline the user's request?", state = the last user turn and the assistant reply (600 characters each); cap 30k | 30,000 |
| Aditharavind/laya-neuro-decisions | CC-BY-4.0, derived from STEW (Lim, W.L., Sourina, O., Wang, L.P. (2018), "STEW: Simultaneous Task EEG Workload Dataset", IEEE TNSRE) via its CC-BY-4.0 Hub mirror | https://huggingface.co/datasets/Aditharavind/laya-neuro-decisions | all splits; Laya rows as they are (one noul on EEG-derived workload features, soft gold); cap 20k | 20,000 |
| Anay2010/laya-trading-decisions | Apache-2.0 | https://huggingface.co/datasets/Anay2010/laya-trading-decisions | train + validation; Laya rows as they are (synthetic trading states: action, regime, confidence) | 15,792 |
| aaung/laya-tetris-decisions | Apache-2.0 | https://huggingface.co/datasets/aaung/laya-tetris-decisions | all splits; choice over the candidate placements A..H listed in the state (moved to the options), the heuristic teacher's preference as soft target | 8,000 |

Not used: allenai/sciq (CC-BY-NC-3.0), allenai/cosmos_qa (CC-BY-4.0 but script-only, no Parquet conversion), ChaosNLI (non-commercial), devinblaze/laya-multidomain-decisions (Apache-2.0, but 801 of its 849 rows are duplicates).

Auxiliary: the GPT-2 `tokenizer.json` from https://huggingface.co/openai-community/gpt2 (MIT)
is downloaded only to decode the `pngwn/typed-decisions` prompts.

## Command data (`data/commands/*.jsonl`)

One line per example: `{src, lang, user, tools?, target_json?, intent?, split}` (`target_json`
is the compact JSON text of the calls, `[]` for a plain reply). 95/5 train/val by user text,
MASSIVE keeps its official splits.

| dataset | license | URL | how it is used | rows used |
|---|---|---|---|---|
| NousResearch/hermes-function-calling-v1 | Apache-2.0 | https://huggingface.co/datasets/NousResearch/hermes-function-calling-v1 | configs `func_calling_singleturn`, `func_calling`, `glaive_func_calling`; first human turn + the `<tool_call>` blocks of the reply; tools normalized to `{name, description, parameters}`; rows with Python-literal arguments (793 extraction tasks) or no tool list (865) skipped; deduplicated | 2,557 (2,273 with calls) |
| glaiveai/glaive-function-calling-v2 | Apache-2.0 | https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2 | system prompt parsed into the tool list, first USER turn + the assistant's `<functioncall>` (single-quoted arguments parsed) or `[]`; conversations without any function capped to 10k; deduplicated (the set repeats many first turns) | 24,042 (8,674 with calls) |
| mteb/amazon_massive_intent (MASSIVE, `en` + `fr`) | MASSIVE: CC-BY-4.0 (the mteb mirror is tagged Apache-2.0) | https://huggingface.co/datasets/mteb/amazon_massive_intent | train / validation / test utterances as intent records (`intent` = label) | 32,341 |

## s4 teacher corpus (`data/s4/`, `sky-data s4-corpus`)

The records the Qwen3.5 teacher scores (`queue/`), the gold sets the s4 student trains on
(`gold/`, symlinks to `data/decisions` minus the held-out sets) and the zero-shot held-out sets
(`heldout/`, `split = "test"`, never trained on). The teacher only adds soft labels; its
outputs are not redistributed.

| dataset / model | license | URL | how it is used | rows used |
|---|---|---|---|---|
| Qwen/Qwen3.5-4B (MLX 4-bit conversion `mlx-community/Qwen3.5-4B-4bit`) | Apache-2.0 | https://huggingface.co/Qwen/Qwen3.5-4B | teacher: letter logits of the options in one non-thinking pass (`scripts/teacher-score.py`) | — |
| glaiveai/glaive-function-calling-v2, NousResearch/hermes-function-calling-v1 | Apache-2.0 | see the command data above | tool routing: the user turn, the conversation's tools plus distractor tools from other conversations and a `no_tool` option; gold = the first call (or `no_tool`) | 15,000 |
| bitext/Bitext-customer-support-llm-chatbot-training-dataset | CDLA-Sharing-1.0 | https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset | ticket triage: random subsets of its 11 categories (or 27 intents) with an `unsure` option, which is the answer when the gold category is left out (15%) | 15,000 |
| google/civil_comments | CC0-1.0 | https://huggingface.co/datasets/google/civil_comments | rule-based moderation (`validation` split): noul "the message violates this rule"; for the six rules that match a column (insult, threat, obscene, identity attack, toxicity, sexual explicit) the annotator fraction is the soft gold label, the five other rules are teacher only; toxic comments oversampled | 14,721 |
| MLM text above (`data/text`) | see below | — | generic snippets (200–700 chars) under random label sets (topic, genre, audience, tone, …), teacher only | 30,000 |
| typed-decision sets above (`data/decisions`) | see above | — | 4,000 per source (train / val), choice records label-randomized: shuffled options, distractors from other questions, an abstention option, and 12% with the gold option removed so that the abstention option is the answer | 80,000 |
| PolyAI/banking77 | CC-BY-4.0 | see above | **held out**: excluded from s4 training, 2,000 of its 8-way choices kept as `zs_banking77` | 2,000 |
| MASSIVE (`en`, `fr`) | CC-BY-4.0 | see above | **held out**: test utterances, gold intent + 7 random intents, `zs_massive_en` / `zs_massive_fr` | 2,000 + 1,000 |

## s6-nano / s6-micro: GLM teacher mix and evaluation sets (`docs/s6-gen-prereg.md`)

**The teacher is an open-weight model.** GLM-5.3-Flash is published under the MIT licence
([`zai-org/GLM-5.3-Flash`](https://huggingface.co/zai-org/GLM-5.3-Flash)). A model of that size
does not run on the laptop, so it is called through a hosted endpoint used as a proxy (the one
`ccs` configures, hosted by z.ai); the labels can be produced again from the open weights without
that endpoint. The owner's position (3 November 2026, restated on 16 November 2026): the outputs
of an open-weight MIT model are free to use as training labels and to publish. They are published
as `maxgfr/skycmd-glm-labels`, marked as AI-generated, with the texts under their sources'
licences (TODO.md §1.2).

s6-nano and s6-micro train on `data/decisions`, `data/distill`, `data/games` and `data/numeric`
(train splits only) and pretrain on `data/text-open` then `data/text`. The GLM files of
`data/decisions` (`glm_orders`, `paraphrased`) exist only if `prepare-data.sh` had a teacher
endpoint, and the report states whether they were there.

| file | how it is used | rows |
|---|---|---|
| `data/teacher/glm_labels.jsonl` | `sky-data teacher label-decisions` (`scripts/distill-glm.sh`): GLM-5.3-flash, an open-weight model (MIT, https://huggingface.co/zai-org/GLM-5.3-Flash) called through a hosted endpoint used as a proxy, gives one probability per option for every train record of `data/decisions` (benchmark, held-out and GLM-generated sources excluded) | 747,198 selected: 747,077 labelled, 121 refused by the provider's moderation (05/11/2026) |
| `data/distill/glm.jsonl` | `sky-data teacher mix-labels`: the labels whose argmax is the gold answer, `target = 0.5 · gold + 0.5 · teacher`, `src = glm:<source>`, train split; agreement per source in `agreement.json` | 560,238 (agreement 0.750) |

The test
sets are read only by the final evaluation (`scripts/train-s6.sh eval`), at most 2,000 questions
per source. `data/s4/heldout` is written by `sky-data s4-corpus --only-heldout`, without the
s4 teacher corpus.

| dataset | license | URL | how it is used | rows used |
|---|---|---|---|---|
| LocalLLaMA/typed-decisions | Apache-2.0 | see above | test split (`typed_decisions_bench`), the gated benchmark | 2,000 |
| MASSIVE (`en`, `fr`) | CC-BY-4.0 | see above | **zero-shot**: `zs_massive_en` / `zs_massive_fr` as for s4; MASSIVE is not in `data/decisions` | 2,000 + 1,000 |
| AirsideLabs/notam-typed-decisions | CC-BY-4.0 | see above | test only, never trained on | 2,000 per config |
| nvidia/When2Call | CC-BY-4.0 | see above | the `test/mcq` split; its train and val splits are trained on (same distribution) | 2,000 |
| PolyAI/banking77 | CC-BY-4.0 | see above | **in distribution**: all of it is in `data/decisions` (train + val), so `zs_banking77` is reported apart and counted in no gate | 2,000 |

## MLM text (`data/text/*.jsonl`)

| dataset | license | URL | how it is used | docs used |
|---|---|---|---|---|
| SimpleStories/SimpleStories | MIT | https://huggingface.co/datasets/SimpleStories/SimpleStories | `story` column, first row groups of train, 30 MB | 24,248 |
| roneneldan/TinyStories | CDLA-Sharing-1.0 | https://huggingface.co/datasets/roneneldan/TinyStories | `text` column, first row groups of train, 30 MB | 33,400 |
| btecsec/tinystories-fr | CDLA-Sharing-1.0 | https://huggingface.co/datasets/btecsec/tinystories-fr | French `text` column, whole train split (18.5 MB; shortfall passed to fineweb-2 fr) | 20,000 |
| HuggingFaceFW/fineweb-edu (`sample/10BT`, shard `000_00000`) | ODC-BY-1.0 | https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu | `text` column of the first row groups (HTTP range reads), 37.5 MB | 9,002 |
| HuggingFaceFW/fineweb-2 (`fra_Latn`, shard `train/000_00000`) | ODC-BY-1.0 | https://huggingface.co/datasets/HuggingFaceFW/fineweb-2 | `text` column of the first row groups (HTTP range reads), 34 MB | 11,010 |

All text is deduplicated exactly and with MinHash/LSH (word 5-grams, Jaccard ≥ 0.8). The
FineWeb datasets are also subject to the terms of use of their CommonCrawl source.

## MLM text, open set (`data/text-open/*.jsonl`)

`sky-data fetch-text --set open`, run by `scripts/prepare-data.sh`. Non-web parts of the
published training data of two fully open LLMs: Dolma 3 (OLMo 3, Ai2) and TxT360 (LLM360). s6
pretraining (the s6-pico pilot and ablation, s6-nano, s6-micro, s6-secrets) reads it before `data/text`
(`docs/s6-prereg.md` and `docs/s6-secrets-prereg.md`, amendment 1; `docs/s6-gen-prereg.md`); s1 to s5 were trained
without it, and `data/text` itself is unchanged.
The JSONL files are streamed and read only up to each budget. PG-19 books and Europarl sessions
give up to 8 pieces of at most 20,000 characters per document instead of their first piece.
Deduplication as above, within this set. Fetched on 29/10/2026: 15,011 documents and 99.5 MB of
text, against budgets that add up to 150 MB, because the Wikipedia shard read holds only 4.3 MB.

| dataset | license | URL | how it is used | docs used |
|---|---|---|---|---|
| allenai/dolma3_mix-150B-1025, `data/dolma1_7-wiki-en` | ODC-BY-1.0 (Dolma 3); Wikipedia and Wikibooks text under CC-BY-SA-4.0 / GFDL | https://huggingface.co/datasets/allenai/dolma3_mix-150B-1025 | `text` of shard `00000000`, 37.5 MB | 2,489 (4.3 MB: the shard holds less than the budget) |
| allenai/dolma3_mix-150B-1025, `data/olmocr_science_pdfs-science_math_and_technology` | ODC-BY-1.0 (Dolma 3); the PDFs keep their authors' rights, see the Dolma 3 card | same | `text` of shard `00000000` (olmOCR transcriptions of science PDFs), 37.5 MB | 1,517 (22.1 MB; 235 exact duplicates dropped) |
| allenai/dolma3_mix-150B-1025, `data/finemath-3plus` | ODC-BY-1.0 (FineMath and Dolma 3), and CommonCrawl's terms of use | same | `text` of shard `00000000`, 22.5 MB; the shard repeats documents (Dolma 3 upsamples FineMath), so most rows are exact duplicates | 4,354 (20.6 MB; 15,081 exact duplicates dropped) |
| IFM/TxT360 (formerly LLM360/TxT360), `data/pg19` | ODC-BY-1.0 (TxT360); PG-19 books are public domain (Project Gutenberg, pre-1919) | https://huggingface.co/datasets/IFM/TxT360 | `text` of `data/pg19/1-1/0_0.jsonl`, up to 8 pieces per book, 22.5 MB | 1,140 (22.5 MB) |
| IFM/TxT360, `data/stackexchange` | ODC-BY-1.0 (TxT360); Stack Exchange posts under CC-BY-SA | same | `text` of `data/stackexchange/1-1/0_2289.jsonl`, 15 MB | 4,638 (15.0 MB) |
| IFM/TxT360, `data/europarl` | ODC-BY-1.0 (TxT360); European Parliament proceedings, reuse authorised with acknowledgement of the source | same | rows with `meta.language == "fr"` of `data/europarl/1-1/0_6.jsonl` then `1_6.jsonl` (French is about 1 document in 20, so ~20× the bytes are read), up to 8 pieces per session, 15 MB | 873 (15.0 MB) |

Not used from these corpora: the `common_crawl-*` parts of Dolma 3 (overlap with FineWeb, and an
`adult_content` category), `stack_edu-*` (source code, with a license per file) and TxT360's
`hackernews` (Hacker News terms).

## Teacher-generated data (`data/teacher/`, `data/decisions/glm_orders.jsonl`, `data/decisions/paraphrased.jsonl`)

Paraphrases and drone orders were generated for s1 to s3 by GLM-5.3-Flash, the open-weight MIT
model that is also the teacher of s6, through a hosted endpoint used as a proxy (`sky-data teacher
...`). The model's name was not logged with the files at the time: it is the owner's statement
(16/11/2026), with the decision that this data is free to use and to publish. No s6 model trains on
these two files: both were absent from `data/decisions` when the s6 data was prepared
(`docs/big-run-report.md` §7, item 9).

| file | how it is used | rows |
|---|---|---|
| `data/decisions/glm_orders.jsonl` | the `action` records of the 3,000 validated civil drone orders (`teacher/orders_decisions.jsonl`, already SPEC records), `src = glm_orders`, re-split 95/5 by order text | 3,000 |
| `data/decisions/paraphrased.jsonl` | `augment-paraphrases`: records of `data/decisions` + `data/games` (non-test) whose instructions have a GLM paraphrase (EN or FR, 932 distinct instructions) get a randomly drawn paraphrase as `ins`; options, state, target and split unchanged; drawn fairly across instructions; `src = para:<src>`; cap 40k | 40,000 |

**Not used by s5 or s4.** No GLM output enters s5 training, validation, calibration, threshold
selection or any published set, and s4 excluded both files; they were used by s1–s3, whose
model cards say that part of their training data was generated with GLM.

## s5 models and evaluation sets (`models-src/`, `data/s5/src/`)

Fetched with `cargo xtask s5 fetch-models` / `fetch-evals` (`sky-data fetch-model`): each repository
is pinned to the commit below and every file is checked against the sha256 (LFS) or blob sha1 the
host publishes; the commit and hashes are recorded in `.skycmd-fetch.json` next to the files.

| repository | license | commit | how it is used |
|---|---|---|---|
| jhu-clsp/ettin-encoder-17m | MIT | `98760745` | s5-lite backbone |
| jhu-clsp/ettin-encoder-32m | MIT | `1b8ba064` | size-ladder backbone |
| cross-encoder/ettin-reranker-17m-v1 | Apache-2.0 | `9e4aa353` | alternative initialisation of the s5-lite backbone |
| jhu-clsp/mmBERT-small | MIT | `abc32620` | s5 (en + fr) backbone, vocabulary pruned |
| mgoeckel/oscar-1-17m | Apache-2.0 | `a443c181` | golden reference of sky-infer; yes/no comparison in the demos |
| mgoeckel/oscar-1-150m, oscar-1-400m | Apache-2.0 | `1015773b`, `d0a87f7a` | English teachers (soft targets) |
| convaiinnovations/laya-typed-decisions | Apache-2.0 | `e929ae5c` | English teacher (soft targets) |
| convaiinnovations/laya-multilingual | Apache-2.0 | `1720e3e3` | French teacher (soft targets) |
| LocalLLaMA/typed-decisions | Apache-2.0 | `c806f6c6` | train split for training and coherence families; test split for the bench |
| btzsc/btzsc (wikitoxic ×4, biasframes ×3, amazonpolarity, yahootopics) | per source: wikitoxic CC0-1.0, biasframes CC-BY-4.0, amazonpolarity Apache-2.0, yahootopics unverified | `fef2a2ac` | English zero-shot evaluation only; yahootopics is reported only if its license is confirmed |
| Davlan/sib200 (`fra_Latn`, `eng_Latn`) | CC-BY-SA-4.0 | `38977a66` | French zero-shot evaluation only |
| DeepPavlov/hwu64 (HWU64 of Liu et al. 2019, reformatted) | CC-BY-4.0 (the original data, [xliuhw/NLU-Evaluation-Data](https://github.com/xliuhw/NLU-Evaluation-Data); the mirror carries no license tag) | `0dd289cc` | s5-lite zero-shot evaluation only (its training excludes MASSIVE and SLURP in every language); reported with "contamination through the teachers not excluded" |
| mteb/mtop_intent (`fr`, `en`) | tagged `unknown` on the Hub | `9a200bf9` | French zero-shot evaluation, only if the license is confirmed |
| bro789/typed-decision-coherence (GitHub) | code MIT; stimuli derived from ChaosNLI and PubMedQA under their own terms | `8898ef7b` | the 2609.33209 coherence battery: evaluation only, never redistributed |

Teachers whose sources are not all declared (laya, oscar) leave a partial decontamination: the
overlap check covers our data and their declared sources only.

## Doom-like game data (`data/games/doom.jsonl`, `data/dagger-doom/`)

The Doom records use no external data. Both sources are generated procedurally from seeds on grid
maps:

- `sky-games-gen --game doom` plays the maps with the scripted oracle, which has full simulator
  access, and labels every decision with it (`action` choice, `danger` noul, `threat` score);
- `sky-arena --dagger` (DAgger) lets the model play and has the same oracle label the states the
  model's own play reaches. `data/dagger-doom/rN` holds the doom rounds; `data/dagger-doom/others`
  and `data/dagger-doom/oN` hold flappy, rescue and dodge states reached by the base and fine-tuned
  models, so the fine-tune keeps the three earlier games (`scripts/train-doom.sh`).

The browser version draws its textures and sprites from Freedoom (BSD-3-Clause,
https://github.com/freedoom/freedoom). Its licence file ships as `site/public/doom/COPYING.txt`.
No id Software asset is used.

The Doom and ViZDoom datasets on the Hugging Face Hub were checked in October 2026 and none is used.
They are listed under Rejected below, and [`docs/research.md`](docs/research.md) summarizes why.

## s6-cast time series (`data/series/`, `sky-data fetch-series`, `sky-train cast-synth`)
Read from the parquet copies of `autogluon/chronos_datasets` (https://huggingface.co/datasets/autogluon/chronos_datasets). That repository is tagged `other`, so the license that applies to each series is its upstream one, listed here. Every source below is absent from the two zero-shot test sets s6-cast is scored on (GIFT-Eval's 97 configurations and Chronos' zero-shot benchmark). One series in ten, by id, is held out for validation.

| source (`data/series/<name>.jsonl`) | config | upstream license | used |
|---|---|---|---|
| london_smart_meters | `monash_london_smart_meters` | CC-BY-4.0 (Monash Time Series Forecasting Archive, Godahewa et al. 2021, Zenodo) | first 2,000 series, last 8,192 points |
| wind_farms_hourly, wind_farms_daily | `wind_farms_hourly`, `wind_farms_daily` | CC-BY-4.0 (Monash archive) | all series (≤ 400), last 8,192 points |
| pedestrian_counts | `monash_pedestrian_counts` | CC-BY-4.0 (Monash archive) | all series |
| rideshare | `monash_rideshare` | CC-BY-4.0 (Monash archive) | every numeric column of the first 200 rows |
| wiki_daily | `wiki_daily_100k` | CC0-1.0 (Wikimedia pageview data) | first 5,000 series |
| ushcn_daily | `ushcn_daily` | public domain (NOAA NCEI, U.S. Government work) | every numeric column of the first 1,300 rows |
| games | — | ours (Apache-2.0, `sky-games`) | state variables of 60 oracle episodes per game (`cast-synth --games`) |

Not used: GIFT-Eval-Pretrain (no declared license) and every `chronos_datasets` config that is in a test set (M1, M3, M4, M5, NN5, tourism, traffic, weather, electricity, solar, hospital, covid deaths, car parts, ETT, exchange rate, FRED-MD, Dominick, ERCOT, CIF 2016, Australian electricity, KDD Cup 2018, temperature–rain, Saugeen, US births).

## s6-cast v2: new real series (`scripts/cast2-data.py`, `docs/s6-cast2-prereg.md` §3.2, C3)

The candidates of §3.2's selection rule, checked on 15/11/2026 from metadata only (`scripts/cast2-data.py list`: listings, object listings, HTTP headers, licence files); USCRN was then fetched on 15/11/2026 (`scripts/cast2-data.py fetch`: the 159 station files of 2025, 1,011 series kept, 8,282,112 points, 41 MB of JSONL, sha256 `69d9dbc1aeb4f19c5c3ac48ec48e3417a340204946b7a07eb99deedae3d83dd8`); the leakage check's counts are in `$CAST2_DATA/leak.json`. With amendment 2's rule (informative runs: at least 8 distinct values, no value on more than half of the points), 4,275,428 informative runs of USCRN against 141,203,971 of GIFT-Eval and 67,632,869 of Chronos-ZS: no match, no series dropped, all 1,011 series kept (`$CAST2_DATA/series2-checked`, what trains). Amendment 1's rule (any non-constant run) had matched 76,364 runs in 568 series, almost all runs of zeros; that run stays in the same file. Rule 1 is the licence as its publisher states it, rule 2 the exclusion list (GIFT-Eval's datasets, the 27 Chronos-ZS tasks of `eval/chronos-zs/tasks.yaml`, the v1 list below, the datasets the benchmarks draw from), rule 3 the content check of §3.3 (`scripts/cast2-data.py leak-check`), pending until the data is fetched. The list that trains is frozen by amendment 1 of the pre-registration (15/11/2026): USCRN sub-hourly only.

| source | publisher | licence (stated by) | rule 1 | rule 2 | size (metadata) | training | publication of the series |
|---|---|---|---|---|---|---|---|
| `uscrn_subhourly` (5-minute sensors: air, surface and soil temperature, precipitation, solar radiation, humidity, soil moisture, wetness, wind) | NOAA NCEI, U.S. Climate Reference Network (`ncei.noaa.gov/pub/data/uscrn/products/subhourly01`) | public domain (a U.S. Government work, as v1's USHCN; the product README states no other terms) | yes | yes | 2025: 159 station files, 2,256,400,800 bytes; at most 1,431 series, 11,722,752 points (last 8,192 points per station and column) | allowed | allowed |

The series are fetched outside the repository (`$CAST2_DATA/series2`, by default `/Users/maxime/Documents/Code/skycmd-data/s6-cast2/series2`) and never committed or redistributed by this repository. No second-level source passed; that band's weight goes to synthetic stand-ins (§3.4, amendment 1).

Not kept:

| source | why |
|---|---|
| Azure VM traces 2017 (Microsoft, Azure Public Dataset V1; CC-BY-4.0, `LICENSE` of github.com/Azure/AzurePublicDataset; 125 files, 84,202,453,652 bytes) | passes rules 1 and 2; not used by amendment 1: per-VM records for which §3.2 gives no rule to build series |
| Borg cluster data 2011 (Google, cluster-data `ClusterData2011_2`; CC-BY-4.0, `ClusterData2011_2.md`; `task_usage`: 500 files, 42,120,256,604 bytes) | passes rules 1 and 2; not used by amendment 1: per-task records for which §3.2 gives no rule to build series |
| Alibaba cluster-trace-v2018 (github.com/alibaba/clusterdata) | rule 1: no licence file; the README allows use "as long as it is for research or study purpose" |
| Salesforce/lotsa_data and Salesforce/GiftEvalPretrain copies (Alibaba, Azure and Borg traces, PEMS, LargeST, BuildingsBench, IDEAL, ...) | open choice O3 of the pre-registration (not used; "for research purposes only"); a constituent is taken from its own publisher instead |
| LargeST (PEMS 5-minute traffic) | rule 1: CC-BY-NC-4.0 |
| Caltrans PeMS | rule 1: no licence of the rule is stated |
| `autogluon/chronos_datasets` sub-hourly configs (`electricity_15min`, `solar`, `solar_1h`, `taxi_30min`) | rule 2: electricity and solar are GIFT-Eval datasets; `taxi_30min` is 30-minute data of the Chronos training corpus |

TinyCast's own overlap list (20 benchmark names and 66 gridded-weather names, `evidence/corpus-resolution.json` of raws-labs/tinycast at `628c0be`) is not published with its names in that repository, so rule 2 cannot be extended by it.

## s6-cast evaluation and reference models
| item | license | how it is used |
|---|---|---|
| Salesforce/GiftEval (https://huggingface.co/datasets/Salesforce/GiftEval) | Apache-2.0, released for research use | test windows exported locally (`scripts/gift-eval.py`) to score s6-cast and the references; never trained on, never redistributed |
| `eval/gift-eval/` (dataset properties, seasonal-naive reference) | Apache-2.0 (Salesforce, via TinyCast's `tinycast/reference/`; see `eval/gift-eval/NOTICE`) | denominators of the normalized metrics |
| `eval/chronos-zs/` (the 27 tasks of the Chronos zero-shot benchmark, the seasonal-naive reference) | Apache-2.0 (autogluon/fev at `9cf943c`; see `eval/chronos-zs/NOTICE`) | the task list and the denominators of `scripts/chronos-zs.py`; the benchmark's series are read only to score and to hash for the leakage check, never trained on. ETTh and ETTm come through `autogluon/chronos_datasets_extra`'s loading script (Apache-2.0, revision `84737b7`), which reads the ETT CSV files of github.com/zhouhaoyi/ETDataset |
| TinyCast (https://github.com/raws-labs/tinycast, https://huggingface.co/raws-labs/tinycast) | Apache-2.0 (code and weights) | ported to Rust (`sky_infer::cast::tinycast`) as a reference forecaster; its pinned GIFT-Eval results are in `eval/gift-eval/gift_eval_tinycast.csv` |
| Timestral-5M (https://huggingface.co/mistralai/Timestral-5M) | Modified MIT (attribution required; not for companies above $20M monthly revenue without a commercial license) | exported to ONNX (`scripts/timestral-onnx.py`) and shipped with its license (`site/public/models/ref/TIMESTRAL-LICENSE.txt`) as a browser reference |

## s6-secrets API-key jev (`data/secrets/`, `sky-data secret-jev`)

| source | license | how it is used |
|---|---|---|
| `secret_synth`, `secret_bench` | ours (Apache-2.0, `crates/sky-data/src/secrets.rs`) | generated locally from the seed. Keys follow each provider's public format with random bodies, so none is a real key. 50,000 training texts (95/5 train/val by text) and 2,000 bench texts in contexts the training never shows. Never committed, except seven bench texts copied as the presets of the site demo (`site/.vitepress/theme/lib/s6-cases.ts`). |

## s6-secrets v2: real candidates (`sky-data secrets2`, `docs/s6-secrets2-prereg.md`)

The model trained on them, s6-secrets-verify-pico, failed its gate; its file is published as a
result (owner's decision of 15/11/2026). None of the data below is committed or redistributed: it
is downloaded outside the repository, and logs and reports show only hashes and lengths of values.

| source | license | how it is used |
|---|---|---|
| Samsung/CredData (https://github.com/Samsung/CredData, pinned at `01e3d135a90f46c9a7965683b24a05d12e136b33`) | each file under the license of its source repository; the meta labels from CredData | obfuscated files only (CredData's own `download_data.py`; the transient clones with the real values are deleted). Training and validation: the 116 repositories of the cbx-secrets-detector-17M list (`CobuildX-AI/cbx-secrets-detector-17M`, `training_data_sources.csv`), each kept only if its copied license is MIT, Apache-2.0, BSD-2-Clause, BSD-3-Clause or ISC: all 116 (MIT 72, Apache-2.0 34, BSD-3-Clause 7, BSD-2-Clause 3). Test: the other 221 repositories, whatever their license, scored only. The obfuscated training files also enter the MLM pretraining (30% of its text) |
| CredData `word_in_path.json` | Apache-2.0 (CredData) | ported as `WORD_IN_PATH` in `crates/sky-infer/src/secret_cand.rs` |
| Secret-Leak-Detection-Issue-Report (Zenodo 10.5281/zenodo.19622962) | CC BY 4.0 | the paper's splits; each real secret of train and val replaced by a word-preserving shape before training; the test values stay as published and are only scored |
| synthetic candidates (`sky-data secret-cand`) | ours (Apache-2.0, `crates/sky-data/src/secrets.rs`) | provider formats with random bodies, at most 25% of the training items; never in a test |

## LFM3-1.7B System 1 / System 2 and its fine-tune (`data/lfm3/`, `scripts/lfm3-s1s2.sh`, `scripts/lfm3-ft.sh`, `docs/lfm3-s1s2-prereg.md`, `docs/lfm3-ft-prereg.md`)

The router (`docs/lfm3-s1s2-prereg.md`) is inference only. Since its amendment 1 (05/11/2026) it runs remotely on Hugging Face Jobs, in bf16, on the base model and on the fine-tuned one.

**Fine-tuning (planned, not run).** `docs/lfm3-ft-prereg.md` trains a LoRA on LFM3-1.7B, on Hugging Face Jobs. The training data is MIT only:
- the GSM8K and MATH train worked solutions;
- the base model's own correct answers on those train questions (self-distillation).

None of the 600 calibration questions and no test question is trained on; questions with the same normalized text are excluded too. The GLM labels are **never** used for LFM3, as its pre-registration fixed (`docs/lfm3-ft-prereg.md`).

**Where the outputs go.**
- Generations, evaluations and logs go to the private dataset repo `maxgfr/skycmd-lfm3-ft-work`.
- The merged model and its adapter go to the private model repo `maxgfr/skycmd-lfm3-1.7b-s1`.

**Derived weights stay private until the user agrees.** If they are ever published, it is under LFM Open License v1.0, with:
- the license text and a NOTICE;
- the modified files marked;
- a name that does not present the model as "LFM3": "fine-tuned from LiquidAI/LFM3-1.7B", since the license grants no trademark rights.

Commercial use only within the license's revenue threshold.

| dataset / model | license | revision | how it is used | rows |
|---|---|---|---|---:|
| LiquidAI/LFM3-1.7B-MLX-4bit | LFM Open License v1.0 (`lfm1.0`) | `deb2986b` | the local smoke test of the router only (mlx-vlm 0.7.11); the run itself moved to bf16 (amendment 1) | — |
| LiquidAI/LFM3-1.7B | LFM Open License v1.0 (`lfm1.0`), custom code (`trust_remote_code`) | pinned by `lfm3-ft.sh smoke`, then by an amendment of `docs/lfm3-ft-prereg.md` | bf16 base model: router run, self-distillation samples, LoRA fine-tuning (planned); derived weights private | — |
| openai/gsm8k (`main`) | MIT | `740312ad` | 300 train questions (calibration), 500 test questions (test), seed 1234; the other train questions and their worked solutions (`<<...>>` removed, `\boxed{}` final line) for the fine-tune | 800 + about 7,170 |
| DigitalLearningGmbH/MATH-lighteval | MIT (MATH, Hendrycks et al.) | `0530c786` | 300 train problems with a `\boxed{}` answer (calibration); the other train problems and their solutions for the fine-tune | 300 + about 7,200 |
| HuggingFaceH4/MATH-500 | MIT (a subset of the MATH test set, from PRM800K); the card has no license tag | `6e4ed1a2` | the 500 test problems (test) | 500 |
| TIGER-Lab/MMLU-Pro (test) | MIT (card license tag; questions drawn from MMLU, STEM websites, TheoremQA and SciBench) | `b189ec76` | 500 questions sampled with seed 1234, asked in s1: the forgetting gate G3 of the fine-tune; evaluation only, never trained on | 500 |

## s6-ear audio (`data/ear/`, `sky-data fetch-audio`, `sky-data audio-jev`, `docs/s6-ear-prereg.md`)

Only derived features are written (float16 log-mel `.npy`, and 16 kHz WAV of the test clips for the baselines); `data/ear` is not committed and the audio is not redistributed (the questions, the clip index and the Mimi codes are, in `skycmd-ear`), except three test clips, CC-BY-4.0 and credited where they are played: the samples of the Listen page (`site/public/ear/`) and of the s6-ear-nano Space. s6-ear-nano failed its gates; its file is published as a result (owner's decision of 15/11/2026). Counts (fetched on 4 November 2026, 31 GB of features):

| set | train | val | test |
|---|---:|---:|---:|
| Smart Turn (v3 train; test = v3.2-test) | 215,655 | 11,370 | 31,527 |
| Speech Commands v0.02 | 40,538 | 4,861 | 4,890 |
| MSWC en + fr (first archive of each split) | 120,000 | 120,000 | 120,000 |
| LibriSpeech `clean` + Mimi codes | 28,539 | 2,703 | — |

Questions written by `sky-data audio-jev`: `turn_end` 215,655 / 11,370 / 31,527, `keyword`
40,538 / 4,861 / 4,890, `keyword_zs` 113,238 / 109,049 / 2,000 (50 held-out words per language,
20 questions each).

| dataset / model | license | how it is used |
|---|---|---|
| pipecat-ai/smart-turn-data-v3-train | CC-BY-4.0 | `turn_end` (noul, `endpoint_bool`), 23 languages including French; train, 5% val by a seeded hash of the clip id, minus the clips whose id is in the test set |
| pipecat-ai/smart-turn-data-v3.2-test | CC-BY-4.0 | `turn_end` test: the official held-out set of Smart Turn v3.2 (neither model trains on it) |
| Speech Commands v0.02 (`download.tensorflow.org/data/speech_commands_v0.02.tar.gz` and the official `speech_commands_test_set_v0.02.tar.gz`; `google/speech_commands` on the Hub is a loading script over them) | CC-BY-4.0 | `keyword` (choice, the 12-class benchmark): train / val by `validation_list.txt`, test = the official test archive |
| MLCommons/ml_spoken_words (`data/wav/{en,fr}`, the first archive of each split) | CC-BY-4.0 | `keyword_zs`: 50 test words per language held out, never trained on |
| openslr/librispeech_asr (`clean`: `train.100`, `validation`) | CC-BY-4.0 | audio of the Mimi pretraining (log-mel); transcripts for the v2 `about` labels |
| shangeth/librispeech-mimi-codes (`train_clean_100`, `dev_clean`) | CC-BY-4.0 | the Mimi codebook-0 ids (12.5 Hz) predicted by `s2-pretrain --audio`, joined to LibriSpeech by utterance id; Mimi itself never runs (the codec, `kyutai/mimi`, is CC-BY-4.0 on the Hub) |
| AMI meeting corpus (v2, not fetched yet) | CC-BY-4.0 | `duplex_action`, labelled by the pre-registered rule `audio::duplex_label` |
| pipecat-ai/smart-turn-v3 (`smart-turn-v3.2-cpu.onnx`, revision `f766f81d`) | BSD-2-Clause | the baseline of the turn-end gate, run on our test clips (`scripts/ear-baseline.py`); not redistributed |
| GLM-5.3-flash labels of the LibriSpeech transcripts (v2, `teacher label-transcripts`) | the transcripts' license (CC-BY-4.0); the model is MIT | the `about` questions, soft targets on the audio |

## s6-ear v2 audio (`sky-data turn-test`, `fetch-turn`, `turn-eotbench`, `docs/s6-ear2-prereg.md`)

Only derived features are written, outside the repository (`skycmd-data/s6-ear2/`): float16 hops
of the v2 front end (the last 4 s of each clip), the teacher's targets (u8), and the 16-bit PCM
hashes of the test clips for the leakage filter. Each parquet shard is deleted once its features
and targets are written; no audio is redistributed.

| dataset / model | license | how it is used |
|---|---|---|
| pipecat-ai/smart-turn-data-v3.2-train (revision `f8c95d28`, 83 shards) | CC-BY-4.0 | training, 5% validation by a hash of the id (seed 1234); a clip is dropped when its id is a test id or its 16-bit PCM hashes to a test clip's (§3.2); the counts are in `train/LEAK.json` |
| pipecat-ai/smart-turn-data-v3.2-test | CC-BY-4.0 | test (gated), the WAVs v1's fetch wrote; never trained on, never used for a threshold |
| livekit/eot-bench-data (config `all`, revision `main`, its commit recorded) | CC-BY-4.0 (card; the code repository is Apache-2.0) | second test, reported only (§8.4): one decision per `eot` / `hold` span, the audio cut at its start; never trained on |
| pipecat-ai/smart-turn-v3 (`smart-turn-v3.2-cpu.onnx`, revision `f766f81d`) | BSD-2-Clause | the teacher (logit, pooled embedding and encoder states on the training and validation clips only, `scripts/turn-teacher.py`) and the baseline (`scripts/ear-baseline.py`); not redistributed |

## Published on Hugging Face (`scripts/hf-publish-data.py`)

One dataset repository per set, one parquet config per source, each config under its source's
licence (share-alike stays share-alike). Not republished: `LocalLLaMA/typed-decisions` (gated), the
audio of s6-ear (31 GB of log-mel features, rebuilt by `sky-data fetch-audio`) and GIFT-Eval (an
evaluation set). [skycmd-series](https://huggingface.co/datasets/maxgfr/skycmd-series) (s6-cast),
[skycmd-text](https://huggingface.co/datasets/maxgfr/skycmd-text) (MLM text and tokenizers),
[skycmd-ear](https://huggingface.co/datasets/maxgfr/skycmd-ear) (s6-ear questions, clip index, Mimi
codes), [skycmd-lfm3-math](https://huggingface.co/datasets/maxgfr/skycmd-lfm3-math) (LFM3 router
questions), [skycmd-route](https://huggingface.co/datasets/maxgfr/skycmd-route) (the slates of
s6-route-nano and s6-route and the teacher's descriptions, each row under its sources' licences;
assembled, not uploaded yet), [skycmd-cast-synth](https://huggingface.co/datasets/maxgfr/skycmd-cast-synth) (a fixed sample of
our four synthetic series generators, Apache-2.0), [skycmd-evals](https://huggingface.co/datasets/maxgfr/skycmd-evals) (our hand-written
evaluation material: demo cases, s5 zero-shot definitions and paraphrases, Apache-2.0), [skycmd-trajectories](https://huggingface.co/datasets/maxgfr/skycmd-trajectories) (complete game episodes played by
our models and the oracle, Apache-2.0), [skycmd-decisions](https://huggingface.co/datasets/maxgfr/skycmd-decisions), [skycmd-games](https://huggingface.co/datasets/maxgfr/skycmd-games), [skycmd-numeric](https://huggingface.co/datasets/maxgfr/skycmd-numeric), [skycmd-secrets](https://huggingface.co/datasets/maxgfr/skycmd-secrets), [skycmd-commands](https://huggingface.co/datasets/maxgfr/skycmd-commands), [skycmd-zero-shot](https://huggingface.co/datasets/maxgfr/skycmd-zero-shot), [skycmd-glm-labels](https://huggingface.co/datasets/maxgfr/skycmd-glm-labels).

## Game trajectories (`data/trajectories/`, `scripts/gen-trajectories.sh`)

No external data. `sky-arena <model> --trajectories` plays complete episodes of the four games
(seed 2026: 20 flappy stopped alive at 50 pipes, 200 rescue, 100 dodge, 100 doom) with the oracle
and with the published models s1-pico, s2-nano, s2-nano-doom and s3-pico, and writes every
decision: the text state, the model's probabilities, the oracle's soft target, the action, the
reward and the events until the next decision. 1,033,412 decisions and 2,100 episodes, Apache-2.0
(our game cores and models). s4-micro and s5-lite are left out: they are not game models.

## s6-route-nano: tool routing (`sky-data route`, `docs/s6-nano2-prereg.md` §4)

Downloaded by `scripts/build-route-data.sh fetch` at the commits of `scripts/route/pins.json`,
each file checked against the hash its host publishes; every file (URL, commit, bytes, sha256,
licence) is listed in `<data>/MANIFEST.json` of the data folder (`../skycmd-data/s6-route`,
outside the repository; 584,935,409 bytes in all, 15/11/2026). Since 17/11/2026, by the owner's
decision to publish the data under each source's licence, the dataset `skycmd-route`
(`scripts/hf-publish-data.py --only route`; assembled, not uploaded yet) holds: the training and
validation records the models read (`build/records/*/train.jsonl`, `val.jsonl`); the validation
slate files (`build/val/v_tools*.jsonl`); the test files built from the sources of this table
except tool-routing-data (`build/test/`: T-tools and its three reading-check copies, T-metatool,
T-clinc, the MASSIVE held-out test, the When2Call test, BFCL); the fresh files of s6-nano v3
(`build/fresh/f_*.jsonl`); every row with its `source` (the request's) and `option_sources` (the
sources that hold its option texts); the teacher's intent descriptions and description rewrites,
marked as machine-generated; and, as files, `scripts/route/pins.json`, `MANIFEST.json`,
`BUILD.json`, `FRESH.json` and the cluster of each T2-tools slate. It leaves out the raw upstream
files (`src/`, `hf-cache/`), the models, embeddings, environment and evaluation outputs, the 100
test slates of tool-routing-data (its card asks users to review the generating model's terms;
`scripts/build-route-data.sh` rebuilds them), the model's copies of the validation files (`sky-data
route render` rebuilds them) and the items and answers of the false-negative filter. Every string
of it was scanned at assembly with the builder's key rule (`sky_data::route::looks_like_key`): none
is left, apart from five tool names and six repository paths that match the rule without being
keys (listed by hash in the script). Requests
are counted after the clusters' split, the cross-split deduplication and the caps of §4.2 (30 per
cluster in training, 10 on the validation and test sides).

| source | licence (checked on the card or the LICENSE file) | URL @ commit | role | requests used (train / val / test) |
|---|---|---|---|---:|
| glaiveai/glaive-function-calling-v2 | Apache-2.0 | https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2 @ `e7f4b645` | train / val / T-tools: the first user turn answered by a call | 1,881 / 80 / 161 |
| NousResearch/hermes-function-calling-v1 (`func-calling-singleturn`, `func-calling`, `glaive-function-calling-5k`) | Apache-2.0 | https://huggingface.co/datasets/NousResearch/hermes-function-calling-v1 @ `dae3e1d2` | same; a request whose normalised text is a glaive request's is dropped | 892 / 54 / 112 |
| Team-ACE/ToolACE (`data.json`) | Apache-2.0 | https://huggingface.co/datasets/Team-ACE/ToolACE @ `6bda777c` | same; key-like strings masked (103 tools of all sources hold one in their schema or description) | 7,312 / 396 / 831 |
| Seal-Tools (`Seal-Tools_Dataset/train.jsonl`, `tool.jsonl`) | Apache-2.0 (LICENSE of the GitHub repository) | https://github.com/fairyshine/Seal-Tools @ `ce753ecd` | same; its training file only | 10,381 / 510 / 1,007 |
| liminghao1630/API-Bank (`training-data/lv1-api-train.json`, `lv2-api-train.json`) | MIT (dataset card) | https://huggingface.co/datasets/liminghao1630/API-Bank @ `12e8158b` | same; `ToolSearcher` calls dropped | 1,910 / 104 / 209 |
| nvidia/When2Call (`train/*`, `test/when2call_test_mcq.jsonl`) | CC-BY-4.0 | https://huggingface.co/datasets/nvidia/When2Call @ `0582f774` | train / val (the partition of `data/decisions/when2call.jsonl`); the first 2,000 test records of that file reformatted, reported | 12,801 / 699 / 1,863 slates |
| MASSIVE (`data/commands/massive.jsonl`, en and fr) | CC-BY-4.0 | see above | 14 scenarios trained (train partition) and validated; the test partition of the 4 held-out scenarios (datetime, email, play, recommendation) reported | 16,400 / 2,848 / 1,657 |
| PolyAI/banking77 (`data/decisions/banking77.jsonl`) | CC-BY-4.0 | see above | train / val only | 12,359 / 663 / — |
| clinc/clinc_oos (`plus`) | CC-BY-3.0 | https://huggingface.co/datasets/clinc/clinc_oos @ `155b9c71` | test only (T-clinc, 2,000 slates); its 150 intent names described by the teacher from the names alone | — / — / 2,000 slates |
| HowieHwong/MetaTool (`all_clean_data.csv`, `plugin_des.json`, `Task2-Subtask1.json`) | MIT (LICENSE); its queries were written by ChatGPT / GPT-4 | https://github.com/HowieHwong/MetaTool @ `35e81bb7` | test only (T-metatool, 2,000 slates) | — / — / 2,000 slates |
| gorilla-llm/Berkeley-Function-Calling-Leaderboard (`BFCL_v3_multiple`, `BFCL_v3_irrelevance`) | Apache-2.0 | https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard @ `61fc0608` | test only, reported | — / — / 440 slates |
| stanley-nv/tool-routing-data (`tools/test.jsonl`) | CC-BY-4.0 tag; the card asks users to review the generating model's terms before training other models | https://huggingface.co/datasets/stanley-nv/tool-routing-data @ `249f72a6` | test only, reported (never trained) | — / — / 100 slates |
| minishlab/potion-base-2M | MIT | https://huggingface.co/minishlab/potion-base-2M @ `389b9f64` | hard negatives of the training slates; baselines potion-cos, potion-cos-4bit, static-ft; s6-nano v3 (`docs/s6-nano3-prereg.md`): its table packed to 4 bits with its tokenizer in the engine's embedding file `skycmd-s6-route-embed.gguf` (`scripts/s6-route3-embed-export.py`, `sky-route embed-gguf`), a derived copy that carries the licence (MIT, from the model card), the authors (Stephan Tulkens and Thomas van Dongen, Minish Lab) and the commit in its metadata; shipped as `site/public/models/s6-route-embed.gguf`, the embedding part of the published router s6-route (and of its Hugging Face repository once uploaded), under the MIT licence | — |
| sentence-transformers/all-MiniLM-L6-v2 | Apache-2.0 | https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2 @ `1110a243` | tool clustering (§4.2), the test distractor rule (§4.4); baseline minilm-cos | — |
| stanley-nv/toolrouter | MIT | https://huggingface.co/stanley-nv/toolrouter @ `b2607a34` | baseline toolrouter (reported) | — |

Teacher: GLM-5.3-flash (MIT, https://huggingface.co/zai-org/GLM-5.3-Flash) through a hosted
endpoint used as a proxy, `sky-data teacher describe-intent | false-negatives |
rewrite-description`: one description per intent from its name (287 intents), the
false-negative filter of the hard training and validation slates, one rewrite per gold tool of
the training and validation requests. It never receives a test request or slate.

## Rejected

| dataset | reason |
|---|---|
| Speech-MASSIVE (`FBK-MT/Speech-MASSIVE`) | CC-BY-NC-SA-4.0 (non-commercial); `about` uses GLM labels of LibriSpeech transcripts instead |
| SLURP | CC-BY-NC-4.0 (non-commercial) |
| ESC-50 | CC-BY-NC-3.0 (non-commercial) |
| UrbanSound8K | CC-BY-NC-3.0 (non-commercial) |
| XNLI (`facebook/xnli`) | CC-BY-NC-4.0 (non-commercial); also dropped from the tasksource collection |
| Lucie Training Dataset (`OpenLLM-France/Lucie-Training-Dataset`, Lucie-7B) | CC-BY-NC-SA-4.0 (non-commercial) |
| The Pile (`EleutherAI/the_pile_deduplicated`, Pythia) | no license tag on the Hub, and it includes Books3, taken down over copyright |
| `swiss-ai/fineweb-2-compliant-tag`, `swiss-ai/fineweb-edu-compliant-tag` (Apertus) | no license declared on the Hub (checked October 2026); FineWeb with opt-out tags, which overlaps the FineWeb text we already use; no `fra_Latn` in the fineweb-2 one |
| frenchNLI (`CATIE-AQ/frenchNLI`) | a concatenation that includes the French XNLI split, so the CC-BY-4.0 card does not lift XNLI's non-commercial terms |
| SciQ (`allenai/sciq`) | CC-BY-NC-3.0 |
| RACE (`ehovy/race`) | non-commercial research-only terms |
| LogiQA 2.0 | upstream release is CC-BY-NC-SA-4.0 (Hub mirrors re-tag it inconsistently) |
| FQuAD | CC-BY-NC-SA-3.0 |
| SargeDev/jev-distill-corpus-v3 | labels distilled from hosted models through OpenRouter and rows built from 18 QA datasets with mixed (partly non-commercial) licenses and no per-row license; its CC0 part is Open-Jev, which we use directly |
| pngwn/system-one-decisions | CC-BY-NC-4.0 |
| CJJones drone telemetry (`CJJones/LLM_FineTune_Synhetic_Drone_Telemetry_Control`) | CC-BY-NC-SA-4.0 |
| xLAM mirrors (e.g. `NobodyExistsOnTheInternet/xlam-function-calling-60k`, `NewEden/xlam-function-calling-60k-shareGPT`) | re-uploads of the gated `Salesforce/xlam-function-calling-60k` under altered licenses (MIT, AGPL-3.0); the original requires accepting its terms |
| PIQA (`ybisk/piqa`) | no clear license on the Hub (tagged `unknown`); also dropped from the tasksource collection |
| Salesforce/xlam-function-calling-60k | gated |
| Hugging Face drone-command datasets without a declared license | not used |
| Yunncheng/gamewam-vizdoom | Apache-2.0, but 50k rollouts of reward-trained Sample-Factory APPO policies (battle, battle2, defend the center, defend the line), not oracle labels; per frame only position x/y, angle and health next to a 9-dim action, with no ammo, enemies, projectiles or walls, so our state text cannot be built |
| RohanNaga/doom-dense-arnold | CC-BY-4.0, but per-tic recordings of the Arnold RL agent in deathmatch; frames plus health, ammo, kills, deaths, frags, position and angle, with no enemy or projectile positions; the map geometry is in Arnold's own scenario WADs, not in the data |
| RohanNaga/doom-dense-arnold-latents | CC-BY-4.0; Stable Diffusion VAE latents of the frames above plus the same per-tic sidecar; nothing more than the raw set for our purpose |
| liguang0115/overseer-doom | Apache-2.0, but GZDoom traversals of Freedoom maps by a potential-field exploration controller; video plus per-tic pose and movement controls; monsters only as an alive count every 105 tics, no health, damage or enemy positions |
| anakin87/doom-defend-the-center-100k-oracle | Apache-2.0 with labels (turn left/none/right, shoot) from a programmatic oracle, but only the 100x160 RGB frames are stored; the privileged state the oracle saw is not, so there is nothing to render as state text |
| charrywhite/VizDoom_withpos | no license; the closest feature set (health, enemy count, nearest-enemy distance and relative x/y, projectile count, took-damage, position, angle) next to PPO actions, but no walls, doors or waypoints either |
| p-doom/doom-dataset | CC0-1.0, but 10M 60x80 frames with the actions of a GameNGen-reproduction PPO agent; pixels and actions only |
| chrisxx/doom-2players-mp4 | MIT, but two-player Sample-Factory video with 14-dim actions and rewards; pixels only, no game variables |
| chrisxx/doom-2players-latents | MIT; DC-AE latents of the video above; pixels only |
| chrisxx/doom-actions-gemini | MIT, but 172 human deathmatch clips with event labels annotated by Gemini; pixels only, model-made labels |
| chrisxx/doom-actions-gemini-bot | no license; evaluation video clips |
| chrisxx/doom-actions-gemini-v4 | no license; evaluation video clips |
| chrisxx/doom-wm-clips | no license; world-model output videos |
| Masao-Taketani/vizdoom-inference-pixel-dataset | Apache-2.0, but GameNGen frames with discrete actions; pixels only |
| Masao-Taketani/vizdoom-inference-latent-dataset | Apache-2.0; latents of the frames above; pixels only |
| lucrbrtv/doom-e1-gameplay | tagged Unlicense, but frames of the commercial DOOM (1993) episode 1 (id Software assets) with recorded actions; pixels only |
| lucrbrtv/doom-e1-internet-gameplay | tagged Unlicense, but YouTube DOOM (1993) frames with actions guessed by an inverse-dynamics model; pixels only |
| P-H-B-D-a16z/ViZDoom-Deathmatch-PPO | MIT, but frames, health and the discrete actions of a PPO agent; no other game variable |
| P-H-B-D-a16z/ViZDoom-Deathmatch-PPO-Lrg | no license; frames, health and PPO actions |
| P-H-B-D-a16z/ViZDoom-Deathmatch-PPO-XLrg | no license; frames, health and PPO actions |
| P-H-B-D-a16z/ViZDoom-Deathmatch-Primitive | no license; frames under fixed repeated actions |
| P-H-B-D-a16z/ViZDoom-Deathmatch-RandomAction | no license; frames under random actions |
| P-H-B-D-a16z/ViZDoom-Deathmatch-Random-XLrg | no license and no card; frames under random actions |
| P-H-B-D-a16z/ViZDoom-Basic | no license; frames, health and actions |
| arnaudstiegler/vizdoom-5-episodes-skipframe-4-lvl5 | no license; GameNGen-reproduction frames, PPO actions and health |
| arnaudstiegler/vizdoom-50-episodes-skipframe-4 | no license; GameNGen-reproduction frames, PPO actions and health |
| arnaudstiegler/vizdoom-500-episodes-skipframe-4 | no license; GameNGen-reproduction frames, PPO actions and health |
| arnaudstiegler/vizdoom-500-episodes-skipframe-4-lvl5 | no license; GameNGen-reproduction frames, PPO actions and health |
| arnaudstiegler/gameNgen_test_dataset | no license; 100 frame windows with actions and health |
| blanchon/doom_parquet | license `other`; a re-keyed copy of the unlicensed `arnaudstiegler/vizdoom-50-episodes-skipframe-4` (frames + actions) |
| johnrobinsn/ViZDoom-5 | no license; frames, PPO actions and health |
| johnrobinsn/ViZDoom-500 | no license; frames, PPO actions and health |
| johnrobinsn/gamengen-500 | no license; frames, PPO actions and health |
| indweller/doomppo | no license; frames, PPO actions and health |
| xingzhaohu/vizdoom-ppo-dataset | no license and no card; 800 GB of frames, PPO actions and health |
| edureisMD/gamengen | no license and no card; one frame/action parquet |
| dokster/vizdoom-5lvls | no license; frames and actions |
| brahmandam/DoomFrameDataset | no license; frames with policy action, reward and value estimate |
| invocation02/doom-dataset | no license and no card; frames and 6-dim actions |
| invocation02/doom-dataset-1 | no license and no card; frames and actions (1.1 TB) |
| invocation02/doom-dataset-large | no license and no card; frames and actions |
| invocation02/doom-rnd-large | no license and no card; frames and actions |
| talha1503/vizdoom_teacher_parquet | no license; frames with a VLM prompt and the action of a teacher policy (Basic: NOOP / MOVE_LEFT / MOVE_RIGHT / ATTACK) and reward |
| talha1503/vizdoom_shifted_rollout_latency_{1,2,3,4,5,10,15} | no license; the same frame / prompt / action format under injected latency |
| talha1503/deadly_corridor_zero_latency_parquet | no license; Deadly Corridor frames with prompt, action and reward |
| talha1503/deadly_corridor_zero_latency_parquet_v2 | no license; same format |
| talha1503/deadly_corridor_mixed_latency_parquet | no license; same format |
| talha1503/vizdoom_pi0_train | no license and no card; LeRobot frames and actions |
| talha1503/vizdoom_pi0_validation | no license and no card; LeRobot frames and actions |
| pufanyi/ViZDoom | no license; 120 benchmark episode specifications, no gameplay |
| pufanyi/ViZDoom-Agentic-Rollouts | no license; Qwen3.6-27B rollouts (video + actions), model outputs rather than labels |
| CV-Mistral-Hackathon/doom-mistral-final | no license and no card; ASCII renderings of the ViZDoom screen with a chosen action, label source undocumented |
| CV-Mistral-Hackathon/doom-mixtral-alpaca | no license and no card; the same ASCII-screen data in Alpaca format |
| CV-Mistral-Hackathon/doom-mixtral-text | no license and no card; the same ASCII-screen data as text |
| PJMixers/CV-Mistral-Hackathon_doom-mistral-final-latest-10-frames-ShareGPT | no license; a ShareGPT re-format of the hackathon data |
| mtpmlab/vizdoom_hgs_full | no license and no card; Health Gathering Supreme observation / action / reward arrays |
| Davids048/vizdoom | no license and no card; per-scenario tarballs |
| rzhang139/vizdoom-episodes | no license and no card; 10k PyTorch episode files |
| guychuk/frame-vizdoom-data | no license and no card; one HDF5 of frames |
| Karajan42/vizdoom-defend-the-line | no license and no card; HDF5 frames and model checkpoints |
| Karajan42/doom-engine-arena | no license and no card; world-model ONNX checkpoints, not data |
| Karajan42/doom-dungeon-55 | no license; world-model checkpoints, not data |
| micaebe/minecraft-doom | no license and no card; video shards |
| aldigobbler/doom-gameplay-dataset | no license; captioned gameplay video |
| anuj7337/doomPPO | MIT, but the repository is empty |
| fffffchopin/GameNGenPro_Frame_Actions | Apache-2.0, but the repository is empty |
| edbeeching/sample_factory_videos | MIT; six showcase videos, no data |

## Models run as baselines or references only

Not redistributed, and none gives a training target. Licences as tagged on the Hub on 16/11/2026.

| model | licence | role |
|---|---|---|
| CobuildX-AI/cbx-secrets-detector-17M | Apache-2.0 | baseline of s6-secrets-verify-pico (rerun on the same test) |
| Qwen/Qwen3.5-0.8B | Apache-2.0 | a measurement in `BENCH.md` (s4) |
| prajjwal1/bert-tiny | MIT | the BERT-Tiny reference of s4 |
| kyutai/mimi | CC-BY-4.0 | the codec whose published codes s6-ear-nano predicts; never run |
