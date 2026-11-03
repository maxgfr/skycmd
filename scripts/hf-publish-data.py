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
                                      [--data data] [--private]

Everything is assembled under target/hf-data/<repo>/ (parquet + README.md) before upload; with
--dry-run nothing is uploaded. Publication rules:
  * every row keeps the licence of its source (the card has one row per source);
  * gated upstream sets are not republished (LocalLLaMA/typed-decisions, train and bench);
  * the GLM-5.3-flash labels (data/teacher, data/distill) are NOT published: z.ai's API terms
    (§1(f)(xii)) forbid using its outputs to label or train external models (TODO.md §1.2);
  * the MLM text and the audio clips are not republished: their recipe is (the sky-data command,
    the seed, the sha256 of each file in MANIFEST.json).
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
}


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


def card(name: str, s: dict, rows: dict, commit: str, user: str) -> str:
    cfgs = "\n".join(
        f"- config_name: {stem}\n  data_files: data/{stem}.parquet" for stem in sorted(rows)
    )
    langs = "\n".join(f"- {l}" for l in s["languages"])
    lic = s["license"]
    extra = ""
    if s.get("per_source"):
        table = "\n".join(
            f"| `{stem}` | {DECISIONS[stem][0]} | {DECISIONS[stem][1]} | {DECISIONS[stem][2]} | {rows[stem]:,} |"
            for stem in sorted(rows)
        )
        excl = "\n".join(f"- `{k}`: {v}." for k, v in EXCLUDED.items())
        extra = f"""## Sources and licences

Each config is one source and keeps **its source's licence** (share-alike sources stay share-alike).
Attribution goes to the upstream authors; the conversion is ours.

| config | upstream | licence | how it is used | rows |
|---|---|---|---|---:|
{table}

Not republished:
{excl}
"""
    elif s.get("upstream"):
        extra = f"## Source and licence\n\n{s['upstream']}. The conversion is ours.\n"
    else:
        extra = "## Licence\n\nApache-2.0: generated by skycmd's own code, no third-party data.\n"
    files = "\n".join(f"| `data/{stem}.parquet` | {n:,} |" for stem, n in sorted(rows.items()))
    lic_extra = (
        f"\nlicense_name: per-source\nlicense_link: {REPO_URL}/blob/main/DATA_LICENSES.md" if lic == "other" else ""
    )
    return f"""---
license: {lic}{lic_extra}
language:
{langs}
pretty_name: {s['pretty']}
task_categories:
- text-classification
tags:
- skycmd
- typed-decisions
- system-one
configs:
{cfgs}
---

# {s['pretty']}

{s['summary']}

Part of [skycmd]({REPO_URL}), tiny decision models trained from scratch in Rust that run in the
browser. Written by `{s['command']}` at commit [`{commit[:8]}`]({REPO_URL}/commit/{commit}).
Licences of every set: [DATA_LICENSES.md]({REPO_URL}/blob/main/DATA_LICENSES.md).

{SCHEMA if name != 'commands' else ''}
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
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--only", default="", help="comma-separated sets (default: all)")
    ap.add_argument("--user", default="", help="namespace (default: the logged-in user)")
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
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
    for name, s in SETS.items():
        if only and name not in only:
            continue
        files = sorted(data.glob(s["files"]))
        if s.get("per_source"):
            unknown = [f.stem for f in files if f.stem not in DECISIONS and f.stem not in EXCLUDED]
            if unknown:
                raise SystemExit(f"{name}: no licence entry for {unknown}: add them to DECISIONS or EXCLUDED")
            files = [f for f in files if f.stem in DECISIONS]
        if not files:
            print(f"{name}: no file under {data / s['files']}, skipped")
            continue
        out = OUT / s["repo"]
        shutil.rmtree(out, ignore_errors=True)
        (out / "data").mkdir(parents=True)
        rows, manifest = {}, {}
        for f in files:
            rows[f.stem] = to_parquet(f, out / "data" / f"{f.stem}.parquet")
            manifest[f.name] = {"sha256": sha256(f), "rows": rows[f.stem]}
        (out / "MANIFEST.json").write_text(json.dumps({"commit": commit, "source_files": manifest}, indent=2) + "\n")
        (out / "README.md").write_text(card(name, s, rows, commit, user))
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
