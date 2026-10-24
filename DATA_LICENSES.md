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

Paraphrases and drone orders were generated by a GLM model through its provider's API
(`sky-data teacher ...`); their use is subject to that provider's terms of service.

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
