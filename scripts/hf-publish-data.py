# /// script
# requires-python = ">=3.10,<3.14"
# dependencies = [
#   "pyarrow>=15",
#   "huggingface_hub>=0.23",
# ]
# ///
"""Publish skycmd's datasets to Hugging Face: one dataset repository per set, one parquet file per
source, and a card that names each source, its licence, the command that wrote the files and the
commit (TODO.md §1.2).

    hf auth login                       # a token with write access
    uv run scripts/hf-publish-data.py [--dry-run] [--only decisions,games] [--user NAME]
                                      [--data data] [--private] [--glm MIXDIR]
                                      [--route-data ../skycmd-data/s6-route]

Everything is assembled under target/hf-data/<repo>/ (parquet + README.md) before upload; with
--dry-run nothing is uploaded. Publication rules:
  * every row keeps the licence of its source (the card has one row per source);
  * gated upstream sets are not republished (LocalLLaMA/typed-decisions, train and bench);
  * the GLM-5.3-flash labels are published only with --glm MIXDIR (`sky-data teacher mix-labels
    --alpha 0 --out MIXDIR`) as `skycmd-glm-labels`, marked as AI-generated: the outputs of an
    open-weight MIT model, published by the owner's decision (DATA_LICENSES.md, TODO.md §1.2);
  * every set a model reads is published: decisions, games, numeric, secrets, commands, zero-shot,
    the s6-cast series, the MLM text with its tokenizers, the LFM3 questions and the s6-ear
    questions with their clip index and Mimi codes;
  * the audio itself (31 GB of log-mel features and test WAVs) is not: `sky-data fetch-audio`
    rebuilds it from the upstream sets; nor GIFT-Eval, which is an evaluation set never
    redistributed;
  * `skycmd-route` (`--only route`, data folder `--route-data`, outside the repository): the slates
    of s6-route-nano and s6-route, every row with its source, and the teacher's descriptions;
    refused if a file has no licence entry or a string looks like a key (the builder's rule).
"""
import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "target" / "hf-data"
REPO_URL = "https://github.com/maxgfr/skycmd"

# source file stem -> (upstream, licence id for the card, how it is used)
DECISIONS = {
    "typed_synth": ("n4ze3m/typed-decisions-synth", "mit", "teacher soft labels as targets"),
    "assistant_decisions": ("kgrozdanovski/assistant-decisions", "apache-2.0", "gold probability distributions"),
    "pngwn_typed": ("pngwn/typed-decisions", "cc-by-sa-4.0", "GPT-2 prompts decoded to text, gold_probs as soft targets"),
    "boolq": ("google/boolq", "cc-by-sa-3.0", "noul over the passage"),
    "mnli": ("nyu-mll/glue (mnli)", "other (MultiNLI: OANC terms, some fiction CC-BY-SA-3.0 / CC-BY-3.0)", "entailment / neutral / contradiction, balanced 60k"),
    "arc": ("allenai/ai2_arc", "cc-by-sa-4.0", "choice A..E"),
    "commonsense_qa": ("tau/commonsense_qa", "mit", "choice A..E"),
    "banking77": ("PolyAI/banking77", "cc-by-4.0", "gold intent + 7 distractors"),
    "tasksource_jev": ("tasksource/tasksource-jev-typed-decisions", "other (per row, commercial-use licences only: Apache-2.0, MIT, CC-BY, CC-BY-SA, CC0, ODC-BY, BSD...; see the upstream license_use column)", "commercial rows, 120k shared between sources"),
    "procedural_typed": ("tasksource/procedural-typed-decisions", "apache-2.0", "80k shared between the 13 tasks"),
    "open_jev": ("ZefanCai/Open-Jev (browser-drone-expansion-v1-redistributable)", "cc0-1.0", "drone-control kept whole, 30k others"),
    "swarm_laya": ("Aditharavind/swarm-laya-decisions", "apache-2.0", "robot-swarm next action"),
    "helpsteer2": ("nvidia/HelpSteer2", "cc-by-4.0", "score questions (0-4) per attribute"),
    "pawsx_fr": ("google-research-datasets/paws-x (fr)", "other (PAWS-X: free for any purpose, acknowledgement of Google appreciated)", "paraphrase noul in French"),
    "allocine": ("tblard/allocine", "mit", "positive review noul in French"),
    "when2call": ("nvidia/When2Call", "cc-by-4.0", "tool_call / request_for_info / cannot_answer / direct_answer"),
    "helmo_synth": ("helmo/synthetic-typed-decisions", "mit", "gold labels"),
    "tasksource_synth": ("tasksource/synthetic-typed-decisions", "apache-2.0", "aggregated soft answers"),
    "calibrated_typed": ("m-lagnajit/calibrated-typed-decisions", "cc-by-4.0", "permissive rows, calibrated teacher_probs"),
    "snli": ("stanfordnlp/snli", "cc-by-sa-4.0", "as MNLI, 40k"),
    "hellaswag": ("Rowan/hellaswag", "mit", "choice A..D over the endings, 30k"),
    "notam": ("AirsideLabs/notam-typed-decisions", "cc-by-4.0", "evaluation only (split test)"),
    "qasc": ("allenai/qasc", "cc-by-4.0", "choice A..H"),
    "refusal": ("agentlans/refusal-classifier-data (english_only)", "cc-by-4.0", "refusal noul, 30k"),
    "neuro_laya": ("Aditharavind/laya-neuro-decisions (from STEW, Lim et al. 2018)", "cc-by-4.0", "EEG workload noul"),
    "trading_laya": ("Anay2010/laya-trading-decisions", "apache-2.0", "synthetic trading states"),
    "tetris_laya": ("aaung/laya-tetris-decisions", "apache-2.0", "placement choice"),
}
# gated upstream: never republished
EXCLUDED = {
    "typed_decisions_train": "LocalLLaMA/typed-decisions is gated on the Hub",
    "typed_decisions_bench": "LocalLLaMA/typed-decisions is gated on the Hub (the benchmark test split)",
}

SCHEMA = """Each row is one typed decision question (docs/SPEC.md):

| field | type | meaning |
|---|---|---|
| `src` | string | source |
| `t` | string | `choice`, `score` or `noul` |
| `ins` | string | the question |
| `opts` | list of strings | the options, rendered as text |
| `state` | string | the state the question is about |
| `target` | list of floats | a probability per option (soft or one-hot) |
| `split` | string | `train`, `val` or `test` |
"""

MONASH = "autogluon/chronos_datasets `{}` (Monash archive, Godahewa et al. 2021)"
SERIES = {
    "london_smart_meters": (MONASH.format("monash_london_smart_meters"), "cc-by-4.0", "first 2,000 series, last 8,192 points"),
    "wind_farms_hourly": (MONASH.format("wind_farms_hourly"), "cc-by-4.0", "all series, last 8,192 points"),
    "wind_farms_daily": (MONASH.format("wind_farms_daily"), "cc-by-4.0", "all series, last 8,192 points"),
    "pedestrian_counts": (MONASH.format("monash_pedestrian_counts"), "cc-by-4.0", "all series"),
    "rideshare": (MONASH.format("monash_rideshare"), "cc-by-4.0", "every numeric column of the first 200 rows"),
    "wiki_daily": ("autogluon/chronos_datasets `wiki_daily_100k` (Wikimedia pageviews)", "cc0-1.0", "first 5,000 series"),
    "ushcn_daily": ("autogluon/chronos_datasets `ushcn_daily` (NOAA NCEI)", "public domain (U.S. Government work)", "every numeric column of the first 1,300 rows"),
    "games": ("ours: state variables of the skycmd games (`sky-train cast-synth --games`)", "apache-2.0", "60 oracle episodes per game"),
}
TEXT = {
    "simplestories": ("SimpleStories/SimpleStories", "mit", "`story` column, first row groups of train, 30 MB"),
    "tinystories": ("roneneldan/TinyStories", "cdla-sharing-1.0", "`text` column, first row groups of train, 30 MB"),
    "tinystories_fr": ("btecsec/tinystories-fr", "cdla-sharing-1.0", "French `text`, whole train split"),
    "fineweb_edu": ("HuggingFaceFW/fineweb-edu (`sample/10BT`, shard `000_00000`)", "odc-by (and CommonCrawl's terms of use)", "first row groups, 37.5 MB"),
    "fineweb2_fr": ("HuggingFaceFW/fineweb-2 (`fra_Latn`, shard `train/000_00000`)", "odc-by (and CommonCrawl's terms of use)", "first row groups, 34 MB"),
    "dolma3_wiki_en": ("allenai/dolma3_mix-150B-1025 `data/dolma1_7-wiki-en`", "odc-by (Dolma 3); Wikipedia and Wikibooks text CC-BY-SA-4.0 / GFDL", "shard `00000000`"),
    "dolma3_science_pdfs": ("allenai/dolma3_mix-150B-1025 `data/olmocr_science_pdfs-science_math_and_technology`", "odc-by (Dolma 3); the PDFs keep their authors' rights, see the Dolma 3 card", "shard `00000000`, olmOCR transcriptions"),
    "dolma3_finemath": ("allenai/dolma3_mix-150B-1025 `data/finemath-3plus`", "odc-by (FineMath, Dolma 3) and CommonCrawl's terms of use", "shard `00000000`, exact duplicates dropped"),
    "txt360_pg19": ("IFM/TxT360 `data/pg19`", "odc-by (TxT360); the books are public domain (Project Gutenberg)", "up to 8 pieces of 20,000 characters per book"),
    "txt360_stackexchange": ("IFM/TxT360 `data/stackexchange`", "odc-by (TxT360); Stack Exchange posts CC-BY-SA", "one shard, 15 MB"),
    "txt360_europarl_fr": ("IFM/TxT360 `data/europarl` (French rows)", "odc-by (TxT360); European Parliament proceedings, reuse authorised with acknowledgement", "up to 8 pieces per session, 15 MB"),
}
LFM3 = {
    "calib": ("openai/gsm8k (`main`, train, `740312ad`) and DigitalLearningGmbH/MATH-lighteval (train, `0530c786`)", "mit", "300 + 300 calibration questions, seed 1234"),
    "test": ("openai/gsm8k (`main`, test) and HuggingFaceH4/MATH-500 (`6e4ed1a2`)", "mit", "500 + 500 test questions, seed 1234"),
}
EAR = {
    "turn_end": ("pipecat-ai/smart-turn-data-v3-train (train, val) and pipecat-ai/smart-turn-data-v3.2-test (test)", "cc-by-4.0", "`turn_end` noul questions"),
    "keyword": ("Speech Commands v0.02 (Warden 2018), official splits and test archive", "cc-by-4.0", "`keyword` choice, the 12-class benchmark"),
    "keyword_zs": ("MLCommons/ml_spoken_words (en, fr; first archive of each split)", "cc-by-4.0", "`keyword_zs` choice, 50 test words per language held out"),
    "clips_smart-turn": ("pipecat-ai/smart-turn-data-v3-train and v3.2-test", "cc-by-4.0", "clip index: id, split, language, frames, label"),
    "clips_speech-commands": ("Speech Commands v0.02", "cc-by-4.0", "clip index"),
    "clips_mswc": ("MLCommons/ml_spoken_words", "cc-by-4.0", "clip index"),
    "mimi_train": ("openslr/librispeech_asr (`clean`, `train.100`) and shangeth/librispeech-mimi-codes", "cc-by-4.0", "Mimi codebook-0 ids (12.5 Hz) per utterance, the audio pretraining target"),
    "mimi_val": ("openslr/librispeech_asr (`clean`, `validation`) and shangeth/librispeech-mimi-codes", "cc-by-4.0", "same, validation"),
}

CAST_SYNTH = {
    "gp": ("ours: `sky_infer::cast::synth`, family `gp`", "apache-2.0", "KernelSynth-style Gaussian-process draws: 1 to 5 kernels (constant, linear, RBF, rational quadratic, Matérn, periodic) combined by sum or product, sampled with random Fourier features"),
    "pulse": ("ours: family `pulse`", "apache-2.0", "trapezoid pulse trains (spikes or dips) over a baseline, with noise"),
    "tsi": ("ours: family `tsi`", "apache-2.0", "trend (linear, exponential, polynomial, piecewise) + up to 3 seasonal waves + Gaussian, Laplace or Student-t noise + outliers + level shifts"),
    "sarima": ("ours: family `sarima`", "apache-2.0", "seasonal ARIMA with stationary AR parts (sampled through their partial autocorrelations), optional differencing"),
    "mixed": ("ours: the training mix", "apache-2.0", "the four families drawn with weights 0.3 / 0.15 / 0.3 / 0.25, as in s6-cast training"),
}
EVALS = {
    "demo-cases": ("ours, hand-written (`eval/s5/demo-cases.jsonl`)", "apache-2.0", "the 56 cases of the browser demos (30 en, 26 fr): tools, guard, triage, score, playground; excluded from training"),
}

TRAJ_POLICIES = {
    "oracle": "the games' exact oracle, which reads the full simulator (expert demonstrations)",
    "s1-pico": "skycmd s1-pico (ModernBERT-style, trained on flappy, rescue and dodge; never saw doom)",
    "s2-nano": "skycmd s2-nano (trained on flappy, rescue and dodge; never saw doom)",
    "s2-nano-doom": "skycmd s2-nano fine-tuned on doom with DAgger",
    "s3-pico": "skycmd s3-pico (the four games)",
}
TRAJ = {m: (f"ours: {d}", "apache-2.0", "every decision of 20 flappy, 200 rescue, 100 dodge and 100 doom episodes")
        for m, d in TRAJ_POLICIES.items()}
TRAJ.update({f"episodes_{m}": (f"ours: {m}", "apache-2.0", "one row per episode: return, score, outcome")
             for m in TRAJ_POLICIES})
TRAJ_SCHEMA = """One config per policy, one row per decision, in play order:

| field | meaning |
|---|---|
| `model`, `game`, `episode`, `step` | who plays, which game, episode index, decision index in the episode |
| `tick` | simulator ticks before the decision (flappy decides every tick, doom every 4) |
| `state` | the compact text state the policy reads (the same as in the browser) |
| `question`, `options` | the action question and its options, rendered as text |
| `probs` | the policy's probability per option |
| `oracle` | the oracle's soft target per option, computed from the full simulator |
| `action`, `oracle_action` | the action played (argmax of `probs`) and the oracle's argmax |
| `abort_probs`, `abort`, `oracle_abort` | dodge only: the abort question, answered by both |
| `reward`, `events` | reward summed until the next decision, and what happened (`passed`, `crashed`, `rescued`, `kill`, `hurt`, ...) |
| `vetoed` | the coded safety reflex replaced the action |
| `last` | last decision of the episode |

The `episodes_*` configs give one row per episode: `decisions`, `return` (every tick's reward),
`truncated` (flappy stops alive at 50 pipes), `score`, `max_score`, `success`, `crashed`,
`aborted`, `safe_abort`, `vetoes`, `steps`. Flappy has no mission to complete: its episodes stop
alive at 50 pipes (`truncated`), so its completion is 0 and its score is the pipes passed.

**What it is for.** Every state a small model reaches by itself, with the expert's answer next to
its own: imitation learning and DAgger (`oracle` is the correction), offline reinforcement learning
(rewards and outcomes from five policies of very different quality, including complete failures:
s1-pico and s2-nano never learned doom), calibration (confidence against correctness along an
episode), and the study of compounding errors.
"""

CAST_SYNTH_SCHEMA = """Each row is one synthetic series: `id` (`<family>-<i>`) and `values` (2,048 floats). The generators
are ours (`crates/sky-infer/src/cast/synth.rs`), after Reverso (arXiv 2602.17634, appendix B) and
Simulation Alone (arXiv 2601.00970), and deterministic per seed. s6-cast does not train on these
rows: it trains on an endless stream from the same generators (a refreshed pool, other seeds).
This repository is a fixed sample of that stream (`sky-train cast-synth --seed 1 --len 2048`), to
inspect the generators or to pretrain another forecaster on the same distribution.
"""
EVALS_SCHEMA = """`demo-cases` has the typed-decision schema with `opts_kv` (pairs of option id and definition)
instead of `opts`. Two files are frozen by `docs/s5-prereg.md` (their sha256 is recorded there)
and shipped as they are:
- `files/defs.json`: the hand-written questions and class definitions of the s5 zero-shot sets;
- `files/paraphrases.json`: hand-written paraphrases of the option definitions of the
  typed-decisions test split, with no template in common with the training paraphrases.
"""

SERIES_SCHEMA = """Each row is one series: `id` (`<source>:<series>`) and `values` (a list of floats, in time order).
The s6-cast trainer holds out one series in ten, by id, for validation; windows and synthetic
series (`sky-train cast-synth`) are drawn from these at training time.
"""
TEXT_SCHEMA = """Each row is one document: `src` (the source) and `text`. All of it is deduplicated exactly and with
MinHash/LSH (word 5-grams, Jaccard >= 0.8). The tokenizers trained on it are in `tokenizers/`
(`sky-data train-tokenizer --vocab N --num`).
"""
LFM3_SCHEMA = """Each row is one question: `id`, `task` (`gsm8k` or `math`), `question` and `answer` (the final
answer, compared after normalization).
"""
EAR_SCHEMA = """- the question configs (`turn_end`, `keyword`, `keyword_zs`) have the typed-decision schema of
  skycmd-decisions (`src`, `t`, `ins`, `opts`, `state`, `target`, `split`) plus `lang` and `clip`;
  `state` is the path of the clip's float16 log-mel features (`.npy`) relative to the question file;
- the `clips_*` configs index every clip: `id`, `set`, `split`, `lang`, `mel` (features path),
  `frames`, `label`, and `wav` for the test clips;
- the `mimi_*` configs give, per LibriSpeech utterance, its features path and its Mimi codes.

**The audio is not in this repository** (31 GB of log-mel features and 16 kHz test WAVs): it is
rebuilt from the upstream sets by `sky-data fetch-audio`, deterministically, with the same paths.
"""

# skycmd-route (docs/s6-nano2-prereg.md §4, docs/s6-nano3-prereg.md §4). The data folder is
# outside the repository (scripts/build-route-data.sh); it is only read.
ROUTE_DATA = Path(os.environ.get("ROUTE_DATA", ROOT.parent / "skycmd-data" / "s6-route"))
# request source (the prefix of `request`) -> (upstream, licence, pinned commit, role)
ROUTE_SOURCES = {
    "glaive": ("glaiveai/glaive-function-calling-v2", "apache-2.0", "e7f4b645", "train / val / T-tools / T2-tools: the first user turn answered by a call"),
    "hermes": ("NousResearch/hermes-function-calling-v1 (`func-calling-singleturn`, `func-calling`, `glaive-function-calling-5k`)", "apache-2.0", "dae3e1d2", "same; a request whose normalised text is a glaive request's is dropped"),
    "toolace": ("Team-ACE/ToolACE (`data.json`)", "apache-2.0", "6bda777c", "same"),
    "seal-tools": ("fairyshine/Seal-Tools on GitHub (`Seal-Tools_Dataset/train.jsonl`, `tool.jsonl`)", "apache-2.0 (LICENSE of the repository)", "ce753ecd", "same; its training file only"),
    "api-bank": ("liminghao1630/API-Bank (`training-data/lv1-api-train.json`, `lv2-api-train.json`)", "mit (dataset card)", "12e8158b", "same; `ToolSearcher` calls dropped"),
    "when2call": ("nvidia/When2Call (`train/*`, through `data/decisions/when2call.jsonl`)", "cc-by-4.0", "0582f774", "train / val: a call to the called tool, no call to *none*"),
    "when2call-mcq": ("nvidia/When2Call (`test/when2call_test_mcq.jsonl`)", "cc-by-4.0", "0582f774", "test only: the first 2,000 records reformatted"),
    "massive": ("MASSIVE (FitzGerald et al., 2022) through mteb/amazon_massive_intent, en and fr (`data/commands/massive.jsonl`)", "cc-by-4.0", "not pinned (`sky-data fetch-commands`)", "train / val: 14 of 18 scenarios; each intent a tool described by the teacher"),
    "massive-heldout": ("MASSIVE, the 4 held-out scenarios (datetime, email, play, recommendation), en and fr", "cc-by-4.0", "not pinned (`sky-data fetch-commands`)", "test: their test partition; fresh: their train and validation partitions"),
    "banking77": ("PolyAI/banking77 (`data/decisions/banking77.jsonl`)", "cc-by-4.0", "not pinned (`sky-data fetch-decisions`)", "train / val only; each intent a tool described by the teacher"),
    "clinc": ("clinc/clinc_oos (`plus`)", "cc-by-3.0", "155b9c71", "test only (T-clinc, T2-clinc); its 150 intent names described by the teacher"),
    "metatool": ("HowieHwong/MetaTool on GitHub (`all_clean_data.csv`, `plugin_des.json`, `Task2-Subtask1.json`)", "mit (LICENSE); its queries were written by ChatGPT / GPT-4 upstream", "35e81bb7", "test only (T-metatool, T2-metatool)"),
    "bfcl-multiple": ("gorilla-llm/Berkeley-Function-Calling-Leaderboard (`BFCL_v3_multiple`)", "apache-2.0", "61fc0608", "test only"),
    "bfcl-irrelevance": ("gorilla-llm/Berkeley-Function-Calling-Leaderboard (`BFCL_v3_irrelevance`)", "apache-2.0", "61fc0608", "test only"),
}
ROUTE_TEACHER = "GLM-5.3-Flash ([zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash), open weights, MIT)"
# config -> (file in the data folder, split, what it is)
ROUTE_FILES = {
    "tools_train": ("build/records/tools/train.jsonl", "train", "training records of the tool sources: four slates per request (natural, hard, removed, renamed)"),
    "intents_train": ("build/records/intents/train.jsonl", "train", "training records of MASSIVE and Banking77, the same four kinds"),
    "when2call_train": ("build/records/when2call/train.jsonl", "train", "training records of When2Call (natural, hard, renamed)"),
    "tools_val": ("build/records/tools/val.jsonl", "val", "the four kinds on the validation side"),
    "intents_val": ("build/records/intents/val.jsonl", "val", "same"),
    "when2call_val": ("build/records/when2call/val.jsonl", "val", "same"),
    "v_tools": ("build/val/v_tools.jsonl", "val", "validation slates built like T-tools (8 options); the pilots, the baselines' fit, the ask threshold"),
    "v_tools_masked": ("build/val/v_tools_masked.jsonl", "val", "v_tools with every tool renamed `tool_<k>`"),
    "v_tools_pairs": ("build/val/v_tools_pairs.jsonl", "val", "500 requests twice: gold present, gold removed"),
    "v_tools_swapped": ("build/val/v_tools_swapped.jsonl", "val", "the descriptions of the gold and of one distractor swapped"),
    "t_tools": ("build/test/t_tools.jsonl", "test", "T-tools: tools of held-out clusters, the gated test of s6-route-nano"),
    "t_tools_masked": ("build/test/t_tools_masked.jsonl", "test", "T-tools with every tool renamed"),
    "t_tools_pairs": ("build/test/t_tools_pairs.jsonl", "test", "T-tools gold present / gold removed pairs"),
    "t_tools_swapped": ("build/test/t_tools_swapped.jsonl", "test", "T-tools with two descriptions swapped"),
    "t_metatool": ("build/test/t_metatool.jsonl", "test", "T-metatool"),
    "t_clinc": ("build/test/t_clinc.jsonl", "test", "T-clinc (in scope and out of scope)"),
    "t_massive": ("build/test/t_massive.jsonl", "test", "MASSIVE held-out scenarios, test partition"),
    "t_when2call": ("build/test/t_when2call.jsonl", "test", "When2Call `test/mcq` reformatted"),
    "t_bfcl": ("build/test/t_bfcl.jsonl", "test", "BFCL `multiple` and `irrelevance`"),
    "f_tools": ("build/fresh/f_tools.jsonl", "fresh", "T2-tools: fresh requests about T-tools' held-out tools, the gated test of s6-route"),
    "f_tools_masked": ("build/fresh/f_tools_masked.jsonl", "fresh", "T2-tools with every tool renamed"),
    "f_metatool": ("build/fresh/f_metatool.jsonl", "fresh", "T2-metatool"),
    "f_clinc": ("build/fresh/f_clinc.jsonl", "fresh", "T2-clinc"),
    "f_massive": ("build/fresh/f_massive.jsonl", "fresh", "T2-massive"),
}
ROUTE_TEACHER_FILES = {
    "teacher_describe_intent": ("build/teacher/describe-intent.jsonl", "one description per intent, written from its name alone"),
    "teacher_rewrite_description": ("build/teacher/rewrite-description.jsonl", "one rewrite of each gold tool's description of the training and validation requests"),
}
# copied as they are: the record of every download and of the build
ROUTE_EXTRA = ["scripts/route/pins.json", "MANIFEST.json", "build/BUILD.json", "build/fresh/FRESH.json", "build/fresh/f_tools.clusters.jsonl"]
# files of the slate and teacher folders that are not published, and why
ROUTE_EXCLUDED = {
    "build/test/t_toolrouting.jsonl": "the 100 test slates of stanley-nv/tool-routing-data, whose card asks users to review the generating model's terms before training other models; `scripts/build-route-data.sh` rebuilds them",
    "build/val/v_tools.model.jsonl": "the model's copy of `v_tools` (the BM25 line added to each state); `sky-data route render` rebuilds it",
    "build/val/v_tools_masked.model.jsonl": "same, for `v_tools_masked`",
    "build/val/v_tools_pairs.model.jsonl": "same, for `v_tools_pairs`",
    "build/val/v_tools_swapped.model.jsonl": "same, for `v_tools_swapped`",
    "build/teacher/describe-intent.items.jsonl": "the items sent to the teacher (the intent names, also in the `id` of each description)",
    "build/teacher/false-negatives.items.jsonl": "the items of the false-negative filter (training and validation hard slates before filtering); `sky-data route split` rebuilds them",
    "build/teacher/false-negatives.jsonl": "the teacher's answers of the false-negative filter (option indices into the items above); their effect is in the hard slates",
    "build/teacher/rewrite-description.items.jsonl": "the items sent to the teacher (the tools' own descriptions, given in `original`)",
    "build/teacher/describe-intent.refused.jsonl": "items refused by the endpoint's moderation (empty)",
    "build/teacher/false-negatives.refused.jsonl": "items refused by the endpoint's moderation",
    "build/teacher/rewrite-description.refused.jsonl": "items refused by the endpoint's moderation (empty)",
}
# Tokens that the builder's key rule flags but that are not keys, by the first 16 hex digits of
# their sha256: public tool names (the builder masks requests and descriptions, never names,
# Tool::new) and repository paths in URLs. Reviewed by reading them (17/11/2026).
ROUTE_NOT_KEYS = {
    "818c8019286d10e7": "tool name, ToolACE: /v1/youtube/summarizeVideoWithToken",
    "0be4005c5ddaee7d": "tool name, ToolACE: v1GetFundHistoricalPrices",
    "79f19705af35fd38": "tool name, ToolACE: /v1/trademarkAvailabilityCheck",
    "64ad8f3153b030d8": "tool name, When2Call test: Flights_4_SearchOnewayFlight",
    "db88572027160fb0": "tool name, When2Call test: Flights_4_SearchRoundtripFlights",
    "c819d63786f59207": "URL path: co/datasets/NousResearch/hermes-function-calling-v1",
    "ebaf5fb28362ff2e": "URL path: co/datasets/liminghao1630/API-Bank",
    "320160d3beb3b298": "file path: possible_answer/BFCL_v3_multiple",
    "4e1c552e6c6d1b66": "URL path: co/sentence-transformers/all-MiniLM-L6-v2",
    "f87e0a2e258f7f02": "repository: NousResearch/hermes-function-calling-v1",
    "e9e2c8815f70dad4": "repository: sentence-transformers/all-MiniLM-L6-v2",
}

SETS = {
    "decisions": {
        "repo": "skycmd-decisions",
        "files": "decisions/*.jsonl",
        "pretty": "skycmd typed decisions",
        "summary": "Typed decision questions (choice / score / noul with a probability per option) converted from public datasets with a permissive or share-alike licence, the decision data of the skycmd s3 to s6 models.",
        "command": "sky-data fetch-decisions data/decisions (seed 1234, default caps)",
        "languages": ["en", "fr"],
        "license": "other",
        "per_source": True,
    },
    "games": {
        "repo": "skycmd-games",
        "files": "games/*.jsonl",
        "pretty": "skycmd game decisions",
        "summary": "Decision questions from the four skycmd control games (Flappy Drone, Rescue Run, Obstacle Dodge, a Doom-like shooter), labelled by the games' exact oracles. Generated by our Rust game cores: no third-party data.",
        "command": "sky-games-gen --game all --n 60000 --seed 1, and --game doom --n 100000 --seed 1 (scripts/prepare-data.sh)",
        "languages": ["en"],
        "license": "apache-2.0",
    },
    "numeric": {
        "repo": "skycmd-numeric",
        "files": "numeric/*.jsonl",
        "pretty": "skycmd numeric curriculum",
        "summary": "Generated questions about numbers (comparisons, ranges, units) of the skycmd numeric curriculum (SPEC-S3 §3). No third-party data.",
        "command": "sky-train s2-numeric --n 100000 (scripts/prepare-data.sh)",
        "languages": ["en", "fr"],
        "license": "apache-2.0",
    },
    "secrets": {
        "repo": "skycmd-secrets",
        "files": "secrets/*.jsonl",
        "pretty": "skycmd secrets jev",
        "summary": "Short texts (.env, code, configs, curl, logs, diffs) with four questions each: is a usable secret exposed, which service issued it, what kind of value it is, on which line. Every key is generated: it follows its provider's public format with a random body, so none is a real key. `secret_bench` uses four file types the training set never shows.",
        "command": "sky-data secret-jev (seed 1234)",
        "languages": ["en"],
        "license": "apache-2.0",
    },
    "commands": {
        "repo": "skycmd-commands",
        "files": "commands/*.jsonl",
        "pretty": "skycmd commands (MASSIVE intents, en + fr)",
        "summary": "MASSIVE intents in English and French as command records (`{src, lang, user, intent, split}`), official splits.",
        "command": "sky-data fetch-commands",
        "languages": ["en", "fr"],
        "license": "cc-by-4.0",
        "upstream": "MASSIVE (FitzGerald et al., 2022), CC-BY-4.0, via mteb/amazon_massive_intent",
    },
    "s4-heldout": {
        "repo": "skycmd-zero-shot",
        "files": "s4/heldout/*.jsonl",
        "pretty": "skycmd zero-shot intent sets",
        "summary": "The zero-shot test sets of the s4 and s6 models: MASSIVE intents in English (2,000) and French (1,000) and Banking77 (2,000) as typed choice questions over intents.",
        "command": "sky-data s4-corpus --only-heldout",
        "languages": ["en", "fr"],
        "license": "cc-by-4.0",
        "upstream": "MASSIVE (CC-BY-4.0) and PolyAI/banking77 (CC-BY-4.0)",
    },
    "series": {
        "repo": "skycmd-series",
        "files": "series/*.jsonl",
        "pretty": "skycmd time series (s6-cast training)",
        "summary": "The training series of the s6-cast forecaster: public series absent from GIFT-Eval and Chronos' zero-shot benchmark, and the state variables of the skycmd games.",
        "command": "sky-data fetch-series and sky-train cast-synth --games",
        "languages": [],
        "license": "other",
        "sources": SERIES,
        "schema": SERIES_SCHEMA,
        "task": "time-series-forecasting",
    },
    "text": {
        "repo": "skycmd-text",
        "files": ["text/*.jsonl", "text-open/*.jsonl"],
        "extra": "tok/*.json",
        "pretty": "skycmd pretraining text and tokenizers",
        "summary": "The text of skycmd's masked-language-model pretraining (s1 to s6), English and French, and the tokenizers trained on it. `data/text` is read by every model; `data/text-open` (non-web parts of Dolma 3 and TxT360) is read first by the s6 models.",
        "command": "sky-data fetch-text and sky-data fetch-text --set open (scripts/prepare-data.sh)",
        "languages": ["en", "fr"],
        "license": "other",
        "sources": TEXT,
        "schema": TEXT_SCHEMA,
        "task": "fill-mask",
        "note": "TinyStories, TinyStories-fr and SimpleStories are synthetic stories written by language models upstream (see their cards); skycmd generated none of this text.",
    },
    "lfm3": {
        "repo": "skycmd-lfm3-math",
        "files": "lfm3/*.jsonl",
        "pretty": "skycmd LFM3 System 1 / System 2 questions",
        "summary": "The calibration and test questions of the LFM3-1.7B System 1 / System 2 router (`docs/lfm3-s1s2-prereg.md`): GSM8K and MATH, drawn with a fixed seed at pinned revisions.",
        "command": "scripts/lfm3-s1s2.sh data",
        "languages": ["en"],
        "license": "mit",
        "sources": LFM3,
        "schema": LFM3_SCHEMA,
        "task": "question-answering",
    },
    "cast-synth": {
        "repo": "skycmd-cast-synth",
        "files": "cast-synth/*.jsonl",
        "pretty": "skycmd synthetic time series (s6-cast generators)",
        "summary": "A fixed sample of the four synthetic series generators that pretrain the s6-cast forecaster (Gaussian processes, pulses, trend-seasonality-irregular, SARIMA), and of their training mix. Generated by skycmd's own Rust code: no third-party data.",
        "command": "sky-train cast-synth --kind <family> --n 1000 --len 2048 --seed 1, and --n 4000 for the mix",
        "languages": [],
        "license": "apache-2.0",
        "sources": CAST_SYNTH,
        "schema": CAST_SYNTH_SCHEMA,
        "task": "time-series-forecasting",
    },
    "trajectories": {
        "repo": "skycmd-trajectories",
        "files": [("trajectories/*.jsonl", ""), ("trajectories/episodes/*.jsonl", "episodes_")],
        "extra": "trajectories/summary.json",
        "extra_dir": "files",
        "results": "trajectories/summary.json",
        "pretty": "skycmd game trajectories (models vs oracle)",
        "summary": "Complete episodes of the four skycmd games (Flappy Drone, Rescue Run, Obstacle Dodge, a Doom-like shooter) played by four small skycmd decision models and by the exact oracle: at every decision, the state, the model's probabilities, the oracle's soft target, the reward and the events. Generated by our Rust game cores and models: no third-party data.",
        "command": "scripts/gen-trajectories.sh (sky-arena <model> --trajectories, seed 2026)",
        "languages": ["en"],
        "license": "apache-2.0",
        "sources": TRAJ,
        "schema": TRAJ_SCHEMA,
        "task": "reinforcement-learning",
    },
    "evals": {
        "repo": "skycmd-evals",
        "base": "repo",
        "files": "eval/s5/*.jsonl",
        "extra": "eval/s5/*.json",
        "extra_dir": "files",
        "pretty": "skycmd hand-written evaluation sets",
        "summary": "The evaluation material skycmd wrote by hand: the cases of the browser demos, the questions and class definitions of the s5 zero-shot sets, and the paraphrases of the typed-decisions test definitions.",
        "command": "written by hand, frozen by docs/s5-prereg.md",
        "languages": ["en", "fr"],
        "license": "apache-2.0",
        "sources": EVALS,
        "schema": EVALS_SCHEMA,
    },
    "ear": {
        "repo": "skycmd-ear",
        "files": [("ear/jev/*.jsonl", ""), ("ear/clips/*.jsonl", "clips_"), ("ear/mimi/*.jsonl", "mimi_")],
        "pretty": "skycmd ear questions (s6-ear)",
        "summary": "The audio questions of s6-ear (turn end, keyword spotting, zero-shot keywords), the index of every clip and the Mimi codes of its audio pretraining.",
        "command": "sky-data fetch-audio and sky-data audio-jev",
        "languages": ["en", "fr"],
        "license": "cc-by-4.0",
        "sources": EAR,
        "schema": EAR_SCHEMA,
        "task": "audio-classification",
    },
}


def set_files(data: Path, s: dict) -> list:
    """(config name, path, manifest key) of every source file of a set."""
    specs = s["files"]
    specs = [specs] if isinstance(specs, str) else specs
    out = []
    for spec in specs:
        pattern, prefix = (spec, "") if isinstance(spec, str) else spec
        out += [(prefix + f.stem, f, prefix + f.name) for f in sorted(data.glob(pattern))]
    return out


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def traj_schema():
    """Explicit types of the trajectory rows: inference per block fails on lists that are empty
    or null for a whole block (`events`, `abort_probs`)."""
    import pyarrow as pa

    f32s = pa.list_(pa.float32())
    return pa.schema([
        ("model", pa.string()), ("game", pa.string()), ("episode", pa.int64()), ("step", pa.int64()),
        ("tick", pa.int64()), ("state", pa.string()), ("question", pa.string()),
        ("options", pa.list_(pa.string())), ("probs", f32s), ("oracle", f32s),
        ("action", pa.int64()), ("oracle_action", pa.int64()), ("abort_probs", f32s),
        ("abort", pa.bool_()), ("oracle_abort", f32s), ("reward", pa.float32()),
        ("events", pa.list_(pa.string())), ("vetoed", pa.bool_()), ("last", pa.bool_()),
    ])


def to_parquet(src: Path, dst: Path, schema=None) -> int:
    import pyarrow as pa
    import pyarrow.json as pj
    import pyarrow.parquet as pq

    t = pj.read_json(
        src,
        read_options=pj.ReadOptions(block_size=64 << 20),
        parse_options=pj.ParseOptions(explicit_schema=schema) if schema is not None else None,
    )
    pq.write_table(t, dst, compression="zstd")
    return t.num_rows


def card(name: str, s: dict, rows: dict, commit: str, user: str, extra_files: list = ()) -> str:
    cfgs = "\n".join(
        f"- config_name: {stem}\n  data_files: data/{stem}.parquet" for stem in sorted(rows)
    )
    langs = "\n".join(f"- {l}" for l in s["languages"])
    lang_block = f"language:\n{langs}\n" if langs else ""
    lic = s["license"]
    extra = ""
    src = DECISIONS if s.get("per_source") else s.get("sources")
    if src:
        table = "\n".join(
            f"| `{stem}` | {src[stem][0]} | {src[stem][1]} | {src[stem][2]} | {rows[stem]:,} |"
            for stem in sorted(rows)
        )
        excl = "\n".join(f"- `{k}`: {v}." for k, v in EXCLUDED.items())
        excl = f"\nNot republished:\n{excl}\n" if s.get("per_source") else ""
        extra = f"""## Sources and licences

Each config is one source and keeps **its source's licence** (share-alike sources stay share-alike).
Attribution goes to the upstream authors; the conversion is ours.

| config | upstream | licence | how it is used | rows |
|---|---|---|---|---:|
{table}
{excl}"""
    elif s.get("upstream"):
        extra = f"## Source and licence\n\n{s['upstream']}. The conversion is ours.\n"
    else:
        extra = "## Licence\n\nApache-2.0: generated by skycmd's own code, no third-party data.\n"
    files = "\n".join(f"| `data/{stem}.parquet` | {n:,} |" for stem, n in sorted(rows.items()))
    files += "".join(f"\n| `{x}` | — |" for x in extra_files)
    schema = s.get("schema", SCHEMA if name != "commands" else "")
    if s.get("results_rows"):
        res = "\n".join(
            f"| `{r['model']}` | {r['game']} | {r['episodes']} | {r['decisions']:,} | {r['agreement']:.3f} | "
            f"{r['mean_score']:g} | {r['mean_return']:.2f} | {r['completion']:.2f} | {r['crash']:.2f} |"
            for r in s["results_rows"]
        )
        schema += f"""
## Per policy and game

| policy | game | episodes | decisions | agreement with the oracle | mean score | mean return | completion | crash |
|---|---|---:|---:|---:|---:|---:|---:|---:|
{res}
"""
    note = f"\n{s['note']}\n" if s.get("note") else ""
    tags = "- skycmd\n- typed-decisions\n- system-one" if "schema" not in s else "- skycmd"
    lic_extra = (
        f"\nlicense_name: per-source\nlicense_link: {REPO_URL}/blob/main/DATA_LICENSES.md" if lic == "other" else ""
    )
    return f"""---
license: {lic}{lic_extra}
{lang_block}pretty_name: {s['pretty']}
task_categories:
- {s.get('task', 'text-classification')}
tags:
{tags}
configs:
{cfgs}
---

# {s['pretty']}

{s['summary']}

Part of [skycmd]({REPO_URL}), tiny decision models trained from scratch in Rust that run in the
browser. Written by `{s['command']}` at commit [`{commit[:8]}`]({REPO_URL}/commit/{commit}).
Licences of every set: [DATA_LICENSES.md]({REPO_URL}/blob/main/DATA_LICENSES.md).

{schema}
{extra}
## Files

| file | rows |
|---|---:|
{files}

```python
from datasets import load_dataset
ds = load_dataset("{user}/{s['repo']}", "{sorted(rows)[0]}")
```

skycmd wrote no row with a language model: the targets come from the upstream labels (some upstream
sets carry their own teacher soft labels, see their cards), the games' oracles or our generators.
{note}"""


def publish_glm(a, api, user: str, sha: str) -> None:
    """skycmd-glm-labels: the GLM-5.3-flash teacher labels of the decision records."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    mix = Path(a.glm)
    labels_path = Path(a.data) / "teacher" / "glm_labels.jsonl"

    def keep(src):
        return src.removeprefix("glm:") not in EXCLUDED

    agree = [r for r in map(json.loads, open(mix / "glm.jsonl")) if keep(r["src"])]
    labels = [r for r in map(json.loads, open(labels_path)) if keep(r["src"])]
    agreement = {k: v for k, v in json.load(open(mix / "agreement.json")).items() if keep(k)}
    refused = Path(a.data) / "teacher" / "glm_labels.refused.jsonl"
    n_ref = sum(1 for _ in open(refused)) if refused.exists() else 0
    out = OUT / "skycmd-glm-labels"
    shutil.rmtree(out, ignore_errors=True)
    (out / "data").mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(agree), out / "data" / "agree.parquet", compression="zstd")
    pq.write_table(pa.Table.from_pylist(labels), out / "data" / "labels.parquet", compression="zstd")
    (out / "agreement.json").write_text(json.dumps(agreement, indent=2) + "\n")
    n_lab = sum(v["labelled"] for v in agreement.values())
    n_ok = sum(v["agree"] for v in agreement.values())

    status = (
        f"**Complete release** ({a.glm_date}): the labelling is finished. All 747,198 training decisions\n"
        f"of skycmd ([skycmd-decisions](https://huggingface.co/datasets/{user}/skycmd-decisions) and the\n"
        f"gated `LocalLLaMA/typed-decisions`) were sent to the teacher; the {n_lab:,} labels of the\n"
        f"published sources are here."
        if a.glm_final
        else f"**Partial release** ({a.glm_date}): {n_lab:,} of the 747,198 training decisions of\n"
        f"[skycmd-decisions](https://huggingface.co/datasets/{user}/skycmd-decisions) are labelled so far. This\n"
        f"repository is updated as the labelling goes on."
    )

    def up(k):
        return DECISIONS[k][:2] if k in DECISIONS else (k, "see the source")

    table = "\n".join(
        f"| `{k}` | {up(k)[0]} | {up(k)[1]} | {agreement[k]['labelled']:,} | "
        f"{agreement[k]['agree']:,} ({agreement[k]['rate']:.3f}) |"
        for k in sorted(agreement)
    )
    readme = f"""---
license: other
license_name: per-source
license_link: {REPO_URL}/blob/main/DATA_LICENSES.md
language:
- en
- fr
pretty_name: skycmd GLM-5.3-flash teacher labels
task_categories:
- text-classification
tags:
- skycmd
- typed-decisions
- distillation
- synthetic
- ai-generated
configs:
- config_name: agree
  data_files: data/agree.parquet
  default: true
- config_name: labels
  data_files: data/labels.parquet
---

# skycmd GLM-5.3-flash teacher labels

> **AI-generated content.** Every probability in this dataset was generated by **GLM-5.3-flash**
> (Z.ai, open weights under MIT, [`zai-org/GLM-5.3-Flash`](https://huggingface.co/zai-org/GLM-5.3-Flash)),
> called through a hosted endpoint used as a proxy. The question texts are not AI-generated: they come from the public
> datasets listed below.

{status}

The teacher of the skycmd s6-nano and s6-micro models (`docs/s6-gen-prereg.md`, amendment 1): for
each typed decision question (choice / score / noul), GLM-5.3-flash gives one probability per option,
20 to 50 questions per call, temperature 0.2. Written by `sky-data teacher label-decisions` and
`sky-data teacher mix-labels --alpha 0` at commit [`{sha[:8]}`]({REPO_URL}/commit/{sha}).

## Configs

- **`agree`** (default, {len(agree):,} rows): the decision records whose teacher argmax equals the
  gold answer (the agreement filter of skycmd), with `target` = the teacher's probabilities and
  `src = glm:<source>`. Same schema as skycmd-decisions. For training, skycmd mixes them as
  `0.5 * gold + 0.5 * teacher`.
- **`labels`** ({len(labels):,} rows): every raw label, `{{id, src, probs, model}}`. `id` is
  `<source>-<hash>` of the record (`distill::record_id` in the skycmd code), so a label joins back to
  its record in skycmd-decisions.

Not included: the {n_ref:,} texts the z.ai moderation refused (code 1301), and the records of
`LocalLLaMA/typed-decisions`, a gated set.

## Agreement with the gold answer, per source

{n_ok:,} / {n_lab:,} labels agree ({n_ok / max(n_lab, 1):.3f}).

| source | upstream | licence of the texts | labelled | agree (rate) |
|---|---|---|---:|---:|
{table}

The texts keep **their source's licence** (share-alike stays share-alike); the probabilities are
generated by GLM-5.3-flash, an open-weight model under the MIT licence.
"""
    (out / "README.md").write_text(readme)
    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    print(f"glm: {user}/skycmd-glm-labels, agree {len(agree):,}, labels {len(labels):,}, {size / 1e6:.1f} MB")
    if a.dry_run:
        return
    repo_id = f"{user}/skycmd-glm-labels"
    api.create_repo(repo_id, repo_type="dataset", private=a.private, exist_ok=True)
    api.upload_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=str(out),
        commit_message=f"skycmd {sha[:8]}: GLM-5.3-flash labels ({n_lab:,} labelled)",
    )
    print(f"  uploaded: https://huggingface.co/datasets/{repo_id}")


KEY_PREFIXES = ("sk-", "sk_live_", "sk_test_", "pk_live_", "rk_live_", "AKIA", "ASIA", "ghp_", "gho_",
                "github_pat_", "xoxb-", "xoxp-", "AIza", "glpat-")
KEY_TOKEN = re.compile(r"[A-Za-z0-9\-_+/]+")


def looks_like_key(t: str) -> bool:
    """`sky_data::route::looks_like_key` (crates/sky-data/src/route/mod.rs), the rule the builder
    masks requests and descriptions with, on one token of KEY_TOKEN: a provider prefix followed by
    at least 12 more characters, or at least 24 characters with digits, both cases of letters and
    at least 4 bits of entropy per character."""
    if len(t) < 15:  # the shortest prefix (3) + 12; below 24 for the second case too
        return False
    if any(t.startswith(p) and len(t) >= len(p) + 12 for p in KEY_PREFIXES):
        return True
    if len(t) < 24 or not (any(c.isdigit() for c in t) and any(c.islower() for c in t) and any(c.isupper() for c in t)):
        return False
    n = len(t)
    return -sum(k / n * math.log2(k / n) for k in Counter(t).values()) >= 4.0


def key_scan(cfg: str, rows, found: list, reviewed: Counter) -> None:
    """Every string of `rows` through looks_like_key. A token listed in ROUTE_NOT_KEYS is counted
    in `reviewed`; any other goes to `found` as (config, row index, field), never as a string."""

    def strings(v, field):
        if isinstance(v, str):
            yield field, v
        elif isinstance(v, list):
            for x in v:
                yield from strings(x, field)
        elif isinstance(v, dict):
            for k, x in v.items():
                yield from strings(x, f"{field}.{k}" if field else k)

    for i, r in enumerate(rows):
        for field, s in strings(r, ""):
            for t in KEY_TOKEN.findall(s):
                if looks_like_key(t):
                    if hashlib.sha256(t.encode()).hexdigest()[:16] in ROUTE_NOT_KEYS:
                        reviewed[(cfg, field)] += 1
                    else:
                        found.append((cfg, i, field))


def route_schema():
    """Explicit types of the slate rows: `follow` exists in the swapped files only."""
    import pyarrow as pa

    s = pa.string()
    strs = pa.list_(s)
    return pa.schema([
        ("src", s), ("t", s), ("ins", s), ("opts", strs), ("state", s), ("target", pa.list_(pa.float32())),
        ("split", s), ("uid", s), ("kind", s), ("lang", s), ("request", s), ("tools", strs),
        ("follow", pa.int64()), ("source", s), ("option_sources", strs),
    ])


def publish_route(a, api, user: str, sha: str) -> None:
    """skycmd-route: the slates of s6-route-nano and s6-route, and the teacher's descriptions."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    data = Path(a.route_data)
    if not (data / "build").is_dir():
        print(f"route: no build folder under {data}, skipped")
        return
    # every file of the slate and teacher folders has a licence entry or a reason to be left out
    listed = {p for p, *_ in ROUTE_FILES.values()} | {p for p, _ in ROUTE_TEACHER_FILES.values()} | set(ROUTE_EXTRA)
    present = {str(f.relative_to(data)) for d in ("records", "val", "test", "fresh", "teacher")
               for f in (data / "build" / d).rglob("*.jsonl")}
    unknown = sorted(present - listed - set(ROUTE_EXCLUDED))
    if unknown:
        raise SystemExit(f"route: no licence entry for {unknown}: add them to ROUTE_FILES or ROUTE_EXCLUDED")
    # the sources that hold each tool text, and the tools' own names and descriptions
    owners = {}
    for r in map(json.loads, open(data / "build" / "stage" / "requests.jsonl")):
        for t in r.get("tools", []) + [r["gold"]] * bool(r.get("gold")) + r.get("exclude", []):
            owners.setdefault(t, set()).add(r["source"])
    tools = {t["id"]: t for t in map(json.loads, open(data / "build" / "stage" / "tools.jsonl"))}

    out = OUT / "skycmd-route"
    shutil.rmtree(out, ignore_errors=True)
    (out / "data").mkdir(parents=True)
    schema = route_schema()
    rows, manifest, nolicence, found, reviewed = {}, {}, [], [], Counter()
    by_source = Counter()  # (source, split) -> rows
    requests = {}  # (source, split) -> request ids
    lang = Counter()
    for cfg, (rel, split, _) in ROUTE_FILES.items():
        recs = []
        for i, r in enumerate(map(json.loads, open(data / rel))):
            if not set(r) <= set(schema.names) or r["split"] != split:
                raise SystemExit(f"route: {rel} row {i}: unexpected fields or split")
            r["source"] = r["request"].split(":", 1)[0]
            r["option_sources"] = sorted(set().union(*(owners.get(t, {"?"}) for t in r["tools"])))
            if r["source"] not in ROUTE_SOURCES or not set(r["option_sources"]) <= set(ROUTE_SOURCES):
                nolicence.append((cfg, i))
            by_source[(r["source"], split)] += 1
            requests.setdefault((r["source"], split), set()).add(r["request"])
            lang[r["lang"]] += 1
            recs.append(r)
        key_scan(cfg, recs, found, reviewed)
        pq.write_table(pa.Table.from_pylist(recs, schema), out / "data" / f"{cfg}.parquet", compression="zstd")
        rows[cfg] = len(recs)
        manifest[rel] = {"sha256": sha256(data / rel), "rows": len(recs)}
    for cfg, (rel, _) in ROUTE_TEACHER_FILES.items():
        recs = []
        for i, r in enumerate(map(json.loads, open(data / rel))):
            if cfg == "teacher_describe_intent":
                src, intent = r["id"].split(":", 1)
                r = {"id": r["id"], "source": src, "intent": intent, "description": r["description"]}
                ok = src in ROUTE_SOURCES
            else:
                t = tools.get(r["id"], {})
                r = {"id": r["id"], "name": t.get("name"), "original": t.get("desc"), "rewrite": r["description"],
                     "sources": sorted(owners.get(r["id"], {"?"}))}
                ok = bool(t) and set(r["sources"]) <= set(ROUTE_SOURCES)
            if not ok:
                nolicence.append((cfg, i))
            recs.append(r)
        key_scan(cfg, recs, found, reviewed)
        pq.write_table(pa.Table.from_pylist(recs), out / "data" / f"{cfg}.parquet", compression="zstd")
        rows[cfg] = len(recs)
        manifest[rel] = {"sha256": sha256(data / rel), "rows": len(recs)}
    (out / "files").mkdir()
    for rel in ROUTE_EXTRA:
        f = ROOT / rel if rel.startswith("scripts/") else data / rel
        recs = list(map(json.loads, open(f))) if f.suffix == ".jsonl" else [json.load(open(f))]
        key_scan(f.name, recs, found, reviewed)
        shutil.copy(f, out / "files" / f.name)
        manifest[rel] = {"sha256": sha256(f)}
    print(f"route: key-like strings: {len(found)} left; {sum(reviewed.values())} matches of the {len(ROUTE_NOT_KEYS)} "
          f"reviewed tokens (ROUTE_NOT_KEYS), by config and field: {dict(sorted(reviewed.items()))}")
    if found or nolicence:
        shutil.rmtree(out)
        for cfg, i, field in found[:50]:
            print(f"  key-like: {cfg} row {i} field {field}")
        for cfg, i in nolicence[:50]:
            print(f"  no licence entry for a source: {cfg} row {i}")
        raise SystemExit(f"route: refused ({len(found)} key-like strings, {len(nolicence)} rows without a licence entry)")
    (out / "MANIFEST.json").write_text(json.dumps({"commit": sha, "data": "s6-route", "source_files": manifest}, indent=2) + "\n")
    counts = json.load(open(data / "build" / "counts" / "slates.json"))["counts"]
    assigned = json.load(open(data / "build" / "counts" / "split.json"))["requests by group, split and source"]
    read = json.load(open(data / "build" / "BUILD.json"))["counts"]["requests"]
    keys = (read["read"]["keys masked in requests"], read["tools whose schema or description holds a key-like string"])
    (out / "README.md").write_text(route_card(user, sha, rows, by_source, requests, lang, counts, assigned, keys, out))
    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    print(f"route: {user}/skycmd-route, {len(rows)} configs, {sum(rows.values()):,} rows, {size / 1e6:.1f} MB")
    if a.dry_run:
        return
    repo_id = f"{user}/skycmd-route"
    api.create_repo(repo_id, repo_type="dataset", private=a.private, exist_ok=True)
    api.upload_folder(repo_id=repo_id, repo_type="dataset", folder_path=str(out),
                      commit_message=f"skycmd {sha[:8]}: tool-routing slates (s6-route-nano, s6-route)")
    print(f"  uploaded: https://huggingface.co/datasets/{repo_id}")


def route_card(user, sha, rows, by_source, requests, lang, counts, assigned, keys, out) -> str:
    splits = ("train", "val", "test", "fresh")
    keys_req, keys_tools = keys
    cfgs = "\n".join(
        f"- config_name: {c}\n  data_files: data/{c}.parquet" + ("\n  default: true" if c == "tools_train" else "")
        for c in rows
    )
    lic = "\n".join(f"| `{s}` | {u} | {l} | {c if c.startswith('not') else f'`{c}`'} | {r} |"
                    for s, (u, l, c, r) in ROUTE_SOURCES.items())

    def n(s, sp):
        return f"{by_source[(s, sp)]:,}" if by_source[(s, sp)] else "—"

    def q(s, sp):
        return f"{len(requests.get((s, sp), ())):,}" if requests.get((s, sp)) else "—"

    def dl(s):
        g = {k.split(" ")[1]: v for k, v in assigned.items() if s in v}
        return " / ".join(f"{g[sp][s]:,}" if sp in g else "—" for sp in ("train", "val", "test"))

    per = "\n".join(
        f"| `{s}` | " + " | ".join(n(s, sp) for sp in splits) + " | " + " / ".join(q(s, sp) for sp in splits)
        + f" | {dl(s)} |"
        for s in ROUTE_SOURCES
    )
    tot = " | ".join(f"**{sum(v for (s, sp), v in by_source.items() if sp == x):,}**" for x in splits)
    fn = {g: counts.get(f"{g}: dropped, the teacher says the gold cannot", 0)
          for g in ("tools train", "tools val", "intents train", "intents val", "when2call train", "when2call val")}
    t_side = sum(assigned["tools test"].values())
    files = "\n".join(
        f"| `data/{c}.parquet` | {ROUTE_FILES[c][1] if c in ROUTE_FILES else 'teacher'} | "
        f"{ROUTE_FILES[c][2] if c in ROUTE_FILES else ROUTE_TEACHER_FILES[c][1]} | {k:,} | "
        f"{(out / 'data' / f'{c}.parquet').stat().st_size:,} |"
        for c, k in rows.items()
    )
    files += "".join(f"\n| `files/{Path(x).name}` | — | the {'repository' if x.startswith('scripts/') else 'data folder'}'s "
                     f"`{x}`, copied | — | {(out / 'files' / Path(x).name).stat().st_size:,} |" for x in ROUTE_EXTRA)
    excl = "\n".join(f"- `{k}`: {v}." for k, v in ROUTE_EXCLUDED.items())
    fr = lang["fr"]
    return f"""---
license: other
license_name: per-source
license_link: {REPO_URL}/blob/main/DATA_LICENSES.md
language:
- en
- fr
pretty_name: skycmd tool-routing slates (s6-route-nano, s6-route)
task_categories:
- text-classification
tags:
- skycmd
- tool-routing
- function-calling
- ai-generated
configs:
{cfgs}
---

# skycmd tool-routing slates (s6-route-nano, s6-route)

The training, validation and test records of the skycmd tool router: a user request, a list of up
to 7 tools (each a name and one sentence) plus the option *none of these tools fits the request*,
and the right option. Built from public tool-calling and intent sets under their own licences, and
the descriptions the teacher wrote. The two models trained and tested on it:

- [s6-route-nano](https://huggingface.co/{user}/s6-route-nano), a 2.4 MB ternary decision model:
  **failed its gate** (on T-tools 0.8395 against BM25's 0.8500, and 38.9 ms per request in WASM
  against a bar of 2 ms; `docs/s6-nano2-prereg.md`, result block);
- [s6-route](https://huggingface.co/{user}/s6-route), the same file fused with a 4-bit
  potion-base-2M embedding: **passed its gate** on fresh slates (0.8868 on T2-tools against BM25's
  0.8169; `docs/s6-nano3-prereg.md`, result block).

Part of [skycmd]({REPO_URL}). Built by `scripts/build-route-data.sh` (the slates;
`docs/s6-nano2-prereg.md` §4), `scripts/route-teacher.sh` (the teacher's jobs) and
`sky-data --seed 1235 route fresh` (the fresh slates; `docs/s6-nano3-prereg.md` §4), assembled by
`scripts/hf-publish-data.py --only route` at commit [`{sha[:8]}`]({REPO_URL}/commit/{sha}).
Licences of every set: [DATA_LICENSES.md]({REPO_URL}/blob/main/DATA_LICENSES.md).

> **Machine-generated text.** The teacher, {ROUTE_TEACHER}, called through a hosted endpoint,
> wrote: the description of every intent option (MASSIVE, Banking77 and CLINC150 intents, written
> from the intent name alone: the configs `intents_*`, `t_clinc`, `f_clinc`, `t_massive`,
> `f_massive` and `teacher_describe_intent`), and the right tool's description in every `renamed`
> slate (`teacher_rewrite_description`). It also chose which hard distractors were replaced (the
> false-negative filter). It wrote no request, no label and no target, and it never saw a test
> request or a test slate. Upstream, MetaTool's queries were written by ChatGPT / GPT-4, and several
> tool-calling sets were generated with language models (see their cards).

## Configs and splits

One config per file. `split` says what a file is for: `train` (the records the models were
trained on), `val` (validation: the pilots, the calibration, the baselines' fit, the thresholds),
`test` (s6-route-nano's tests, read once on 16/11/2026), `fresh` (s6-route's tests, built after
that reading from requests no earlier file used, read once on 16/11/2026).

| file | split | what it is | rows | bytes |
|---|---|---|---:|---:|
{files}

## Record schema

The slate configs share one schema (the typed-decision format of skycmd, `docs/SPEC.md`):

| field | type | meaning |
|---|---|---|
| `src` | string | the group and source (`tools:<source>`, `intents:<source>`, `when2call:when2call`) in the training and validation records; the file (`v-tools`, `t-clinc`, `f-massive-fr`, ...) in the slate files |
| `t` | string | always `choice` |
| `ins` | string | the instruction, word for word: "Which tool should handle the user's request?" |
| `opts` | list of strings | the options: `name: description` for each tool (the description cut to its first sentence and at most 30 words), then `none of these tools fits the request`, always last |
| `state` | string | `User: <request>` (one user turn). In the training and validation records (`*_train`, `*_val`) a second line follows, `BM25: <name> score <s> rank <r>; ...`, the lexical score of each tool option against the request that the model reads; the slate files (`v_*`, `t_*`, `f_*`) carry no such line, and `sky-data route render` adds it, as the evaluation did |
| `target` | list of floats | one-hot: 1 on the right option |
| `split` | string | `train`, `val`, `test` or `fresh` |
| `uid` | string | the slate's identifier: the request's, with `/<kind>` in the training and validation records |
| `kind` | string | training and validation records: `natural` (the request's own tools, then random tools of its split), `hard` (the nearest tools by potion-base-2M, cleaned by the teacher), `removed` (the right tool taken out: the answer is *none*), `renamed` (every tool named `tool_<k>`); slate files: `test` (drawn by the test rule), `removed`, `masked` (every tool renamed), `swapped` (two descriptions swapped) |
| `lang` | string | `en` or `fr` |
| `request` | string | the request's identifier, `<source>:<row>` |
| `tools` | list of strings | the identifier of each tool option before *none*: `t` and 16 hex digits of the sha256 of its `name: description` as the source gives it |
| `follow` | int | `swapped` slates only: the index of the option that carries the right tool's description |
| `source` | string | the source of the request text (the prefix of `request`); its licence is the row's |
| `option_sources` | list of strings | every source whose data hold the text of one of the tool options (a distractor comes from another request of the same split, often of another source; one text can be in several sources) |

`teacher_describe_intent`: `id` (`<source>:<intent>`), `source`, `intent`, `description` (the
teacher's). `teacher_rewrite_description`: `id` (the tool identifier), `name` and `original` (the
tool's name and description as the source gives them, the description cut and its key-like strings
masked), `rewrite` (the teacher's, same meaning in other words; a `renamed` slate shows it cut to
its first sentence and 30 words), `sources` (the sources that hold the tool).

## Sources and licences

Each row keeps **the licence of its sources**: the request text is `source`'s, the option texts are
`option_sources`'; the teacher's descriptions are the outputs of an open-weight model under the
MIT licence. Attribution goes to the upstream authors; the conversion is ours.

| source | upstream | licence | commit | role |
|---|---|---|---|---|
{lic}

Rows and requests per split, counted from the files above, and the requests the build kept for
each side (`build/counts/split.json`; DATA_LICENSES.md gives these for training and validation and
for the test side of the tool sources, and slates for the test-only sources):

| source | rows train | rows val | rows test | rows fresh | requests train / val / test / fresh | requests kept by the build (train / val / test) |
|---|---:|---:|---:|---:|---:|---:|
{per}
| all | {tot} | | |

Why the requests in the files are fewer than those kept by the build: a training or validation
request is dropped when the teacher does not name its right tool as able in its hard slate (training
/ validation: tools {fn['tools train']:,} / {fn['tools val']:,}, intents {fn['intents train']:,} / {fn['intents val']:,}, When2Call {fn['when2call train']:,} / {fn['when2call val']:,};
`docs/s6-nano2-prereg.md` amendment 2); When2Call's no-call slates are thinned at random to the
20% share of *none* (amendment 1; {counts.get('when2call train: no-call slates dropped', 0):,} training and {counts.get('when2call val: no-call slates dropped', 0):,} validation slates dropped),
which leaves some requests with no slate; T-tools draws 2,000 slates from the {t_side:,} test-side requests, and the
fresh T2-tools file uses the others and those over the cluster cap; T-clinc and T-metatool draw
2,000 slates each from their sources; 137 of the 2,000 When2Call test items list no tool, so no
slate (`docs/s6-nano2-readings.md`, item 34). A request is in up to four training slates and in
several reading-check files.

Not in this repository:
{excl}
- the raw upstream files (`src/`, `hf-cache/`), the models (`models/`, `embed/`), the Python
  environment (`envs/`), the embeddings, stage files and evaluation outputs of the data folder:
  `scripts/build-route-data.sh` downloads every source at the commit of `files/pins.json`, checks
  each file against the hash its host publishes (`files/MANIFEST.json`: URL, commit, bytes,
  sha256, licence) and rebuilds every file here byte for byte (`files/BUILD.json`,
  `files/FRESH.json`: the hashes and counts of the build).

## Limits

- **English mostly**: {fr:,} of the {sum(lang.values()):,} slate rows are French, all from MASSIVE
  (requests in French, intent names and descriptions in English).
- **Synthetic upstream data**: several tool-calling sets and MetaTool's queries were written by
  language models upstream; the build filters and deduplicates, it does not check them by hand.
  The share of validation slates that hold a second valid tool (the label-noise estimate of
  `docs/s6-nano2-prereg.md` §4.5) is not in the result files.
- **Test sets partly in distribution**: T-tools holds tools of clusters never trained on
  ("fresh, tool-disjoint: no test tool cluster in training"), but from the same five sources as
  the training tools; "the clustering that defines 'unseen tool' uses all-MiniLM-L6-v2, which is
  also a reported baseline". "The When2Call test and MASSIVE test utterances were read in v1's
  evaluation" (`docs/s6-nano2-prereg.md` §10); When2Call and the other MASSIVE scenarios are in
  training. On T2-tools "these requests are about the same held-out tools as v2's test", and 107
  of its 945 slates repeat the normalised request of an earlier slate (`docs/s6-nano3-prereg.md`
  §7 and amendment 1). MetaTool, CLINC150 and BFCL are never trained.
- **Keys**: the builder masks key-like strings in requests and descriptions as `<KEY>`
  (`sky_data::route::mask_keys`; {keys_req:,} in requests, and {keys_tools:,} tools held one in their
  schema or description). Every string of `data/` and `files/` was scanned with the same rule at
  assembly, and none is left. The builder does not mask tool names: five tool names, and six
  repository paths in `files/`, match the rule without being keys; they are listed by hash in
  `scripts/hf-publish-data.py` (`ROUTE_NOT_KEYS`).

```python
from datasets import load_dataset
ds = load_dataset("{user}/skycmd-route", "tools_train")
```
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--only", default="", help="comma-separated sets (default: all)")
    ap.add_argument("--user", default="", help="namespace (default: the logged-in user)")
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--glm", default="", help="dir of `sky-data teacher mix-labels --alpha 0`: publish skycmd-glm-labels")
    ap.add_argument("--glm-date", default="", help="date of the release, shown in the card")
    ap.add_argument("--glm-final", action="store_true", help="the labelling is finished: a complete release")
    ap.add_argument("--route-data", default=str(ROUTE_DATA), help="the s6-route data folder (skycmd-route, `--only route`)")
    a = ap.parse_args()

    data = Path(a.data)
    only = [x for x in a.only.split(",") if x]
    commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    user = a.user
    api = None
    if not a.dry_run:
        from huggingface_hub import HfApi

        api = HfApi()
        user = user or api.whoami()["name"]
    user = user or "USER"
    if a.glm:
        publish_glm(a, api, user, commit)
    if "route" in only or not (only or a.glm):
        publish_route(a, api, user, commit)
    for name, s in SETS.items():
        if (only or a.glm) and name not in only:
            continue
        base = ROOT if s.get("base") == "repo" else data
        files = set_files(base, s)
        known = DECISIONS if s.get("per_source") else s.get("sources")
        if known is not None:
            unknown = [c for c, _, _ in files if c not in known and c not in EXCLUDED]
            if unknown:
                raise SystemExit(f"{name}: no licence entry for {unknown}: add them to its sources or EXCLUDED")
            files = [x for x in files if x[0] in known]
        if not files:
            print(f"{name}: no file under {data} for {s['files']}, skipped")
            continue
        out = OUT / s["repo"]
        shutil.rmtree(out, ignore_errors=True)
        (out / "data").mkdir(parents=True)
        rows, manifest = {}, {}
        for cfg, f, key in files:
            schema = traj_schema() if name == "trajectories" and not cfg.startswith("episodes_") else None
            rows[cfg] = to_parquet(f, out / "data" / f"{cfg}.parquet", schema)
            manifest[key] = {"sha256": sha256(f), "rows": rows[cfg]}
        extra_files = []
        if s.get("extra"):
            sub = s.get("extra_dir", "tokenizers")
            dst = out / sub
            dst.mkdir()
            for f in sorted(base.glob(s["extra"])):
                shutil.copy(f, dst / f.name)
                manifest[f.name] = {"sha256": sha256(f)}
                extra_files.append(f"{sub}/{f.name}")
        (out / "MANIFEST.json").write_text(json.dumps({"commit": commit, "source_files": manifest}, indent=2) + "\n")
        if s.get("results"):
            s = {**s, "results_rows": json.load(open(base / s["results"]))}
        (out / "README.md").write_text(card(name, s, rows, commit, user, extra_files))
        size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
        print(f"{name}: {user}/{s['repo']}, {len(rows)} files, {sum(rows.values()):,} rows, {size / 1e6:.1f} MB")
        if a.dry_run:
            continue
        repo_id = f"{user}/{s['repo']}"
        api.create_repo(repo_id, repo_type="dataset", private=a.private, exist_ok=True)
        api.upload_folder(
            repo_id=repo_id,
            repo_type="dataset",
            folder_path=str(out),
            commit_message=f"skycmd {commit[:8]}: {s['pretty']}",
        )
        print(f"  uploaded: https://huggingface.co/datasets/{repo_id}")


if __name__ == "__main__":
    main()
