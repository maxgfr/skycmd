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

## Teacher-generated data (`data/teacher/`, `data/decisions/glm_orders.jsonl`, `data/decisions/paraphrased.jsonl`)

Paraphrases and drone orders are generated by a GLM model through the z.ai API
(`sky-data teacher ...`); their use is subject to the provider's terms of service.

| file | how it is used | rows |
|---|---|---|
| `data/decisions/glm_orders.jsonl` | the `action` records of the 3,000 validated civil drone orders (`teacher/orders_decisions.jsonl`, already SPEC records), `src = glm_orders`, re-split 95/5 by order text | 3,000 |
| `data/decisions/paraphrased.jsonl` | `augment-paraphrases`: records of `data/decisions` + `data/games` (non-test) whose instructions have a GLM paraphrase (EN or FR, 932 distinct instructions) get a randomly drawn paraphrase as `ins`; options, state, target and split unchanged; drawn fairly across instructions; `src = para:<src>`; cap 40k | 40,000 |

## Rejected

| dataset | reason |
|---|---|
| XNLI (`facebook/xnli`) | CC-BY-NC-4.0 (non-commercial); also dropped from the tasksource collection |
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
