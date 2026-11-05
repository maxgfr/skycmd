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

Everything is assembled under target/hf-data/<repo>/ (parquet + README.md) before upload; with
--dry-run nothing is uploaded. Publication rules:
  * every row keeps the licence of its source (the card has one row per source);
  * gated upstream sets are not republished (LocalLLaMA/typed-decisions, train and bench);
  * the GLM-5.3-flash labels are published only with --glm MIXDIR (`sky-data teacher mix-labels
    --alpha 0 --out MIXDIR`) as `skycmd-glm-labels`, marked as AI-generated (z.ai terms §III.5(d));
    the user chose to publish them despite §1(f)(xii) of the z.ai API terms (TODO.md §1.2);
  * every set a model reads is published: decisions, games, numeric, secrets, commands, zero-shot,
    the s6-cast series, the MLM text with its tokenizers, the LFM3 questions and the s6-ear
    questions with their clip index and Mimi codes;
  * the audio itself (31 GB of log-mel features and test WAVs) is not: `sky-data fetch-audio`
    rebuilds it from the upstream sets; nor GIFT-Eval, which is an evaluation set never
    redistributed.
"""
import argparse
import hashlib
import json
import shutil
import subprocess
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


def to_parquet(src: Path, dst: Path) -> int:
    import pyarrow as pa
    import pyarrow.json as pj
    import pyarrow.parquet as pq

    t = pj.read_json(src, read_options=pj.ReadOptions(block_size=64 << 20))
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
> called through the z.ai API. The question texts are not AI-generated: they come from the public
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
generated by GLM-5.3-flash. Users are responsible for complying with the terms of the z.ai API under
which the labels were produced.
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
            rows[cfg] = to_parquet(f, out / "data" / f"{cfg}.parquet")
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
