#!/usr/bin/env python3
"""Publish every skycmd model to Hugging Face: one model repo (card + GGUF) and one static Space
(demo running in the browser) per model.

    pip install -U huggingface_hub && hf auth login        # a token with write access
    scripts/build-wasm.sh && (cd site && npm run space:build)
    python3 scripts/hf-publish.py [--dry-run] [--only s5-lite,s3-pico] [--user NAME]

Inputs: hf/cards/<model>.md (model cards, `{{USER}}` replaced by the namespace), hf/spaces.json
(one entry per Space), site/public/models/ (GGUF files and manifest), site/public/pkg/ (the wasm
module), site/public/doom/ (Freedoom textures for the game Spaces) and target/hf-space-app/ (the
demo app). Everything is assembled under target/hf/<model>/{model,space} before upload; with
--dry-run nothing is uploaded.
"""
import argparse
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "target" / "hf"
APP = ROOT / "target" / "hf-space-app"
PUBLIC = ROOT / "site" / "public"


def space_readme(user: str, s: dict) -> str:
    m = s["model"]
    return f"""---
title: {m}
emoji: {s["emoji"]}
colorFrom: {s["color_from"]}
colorTo: {s["color_to"]}
sdk: static
app_file: index.html
pinned: false
license: apache-2.0
short_description: {s["short_description"]}
models:
  - {user}/{m}
---

# {m}

{s["tagline"]}

Everything runs in your browser: a Rust inference engine compiled to WebAssembly (SIMD128) reads
the GGUF file shipped with this Space. Nothing is sent to a server.

- Model card: [{user}/{m}](https://huggingface.co/{user}/{m})
- Code: [github.com/maxgfr/skycmd](https://github.com/maxgfr/skycmd)
"""


def assemble(user: str, s: dict, manifest: list) -> tuple[Path, Path]:
    m = s["model"]
    base = OUT / m
    if base.exists():
        shutil.rmtree(base)
    model_dir, space_dir = base / "model", base / "space"
    model_dir.mkdir(parents=True)
    card = (ROOT / "hf" / "cards" / f"{m}.md").read_text().replace("{{USER}}", user)
    (model_dir / "README.md").write_text(card)
    shutil.copy2(PUBLIC / "models" / f"{m}.gguf", model_dir / f"{m}.gguf")

    shutil.copytree(APP, space_dir)
    shutil.copytree(PUBLIC / "pkg", space_dir / "pkg", ignore=shutil.ignore_patterns(".gitignore", "package.json", "README.md"))
    (space_dir / "models").mkdir()
    names = [m] + s.get("extra_models", [])
    entries = [e for e in manifest if e["name"] in names]
    for e in entries:
        shutil.copy2(PUBLIC / "models" / e["file"], space_dir / "models" / e["file"])
    (space_dir / "models" / "manifest.json").write_text(json.dumps(entries, indent=2) + "\n")
    if s["kind"] == "game":
        shutil.copytree(PUBLIC / "doom", space_dir / "doom")
    cfg = {
        "model": m,
        "title": m,
        "tagline": s["tagline"],
        "kind": s["kind"],
        "proofs": s.get("proofs", False),
        "model_repo": f"https://huggingface.co/{user}/{m}",
        "notes": s.get("notes", []),
    }
    (space_dir / "space.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n")
    (space_dir / "README.md").write_text(space_readme(user, s))
    return model_dir, space_dir


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="assemble under target/hf, upload nothing")
    ap.add_argument("--only", default="", help="comma-separated model names")
    ap.add_argument("--user", default="", help="namespace (default: the logged-in user)")
    a = ap.parse_args()

    for need in [APP / "index.html", PUBLIC / "pkg" / "sky_web_bg.wasm"]:
        if not need.exists():
            raise SystemExit(f"missing {need}: run scripts/build-wasm.sh and (cd site && npm run space:build)")
    api = None
    user = a.user
    if not a.dry_run:
        from huggingface_hub import HfApi

        api = HfApi()
        user = user or api.whoami()["name"]
    user = user or "USER"

    manifest = json.loads((PUBLIC / "models" / "manifest.json").read_text())
    spaces = json.loads((ROOT / "hf" / "spaces.json").read_text())
    only = {x for x in a.only.split(",") if x}
    for s in spaces:
        if only and s["model"] not in only:
            continue
        model_dir, space_dir = assemble(user, s, manifest)
        print(f"assembled {s['model']}: {model_dir} and {space_dir}")
        if api is None:
            continue
        repo = f"{user}/{s['model']}"
        api.create_repo(repo, repo_type="model", exist_ok=True)
        api.upload_folder(repo_id=repo, repo_type="model", folder_path=model_dir,
                          commit_message=f"Publish {s['model']}")
        api.create_repo(repo, repo_type="space", space_sdk="static", exist_ok=True)
        api.upload_folder(repo_id=repo, repo_type="space", folder_path=space_dir,
                          commit_message=f"Publish the {s['model']} demo", delete_patterns=["assets/*"])
        print(f"published https://huggingface.co/{repo} and https://huggingface.co/spaces/{repo}")


if __name__ == "__main__":
    main()
