#!/usr/bin/env python3
"""Publish every skycmd model to Hugging Face: one model repo (card + GGUF) and one static Space
(demo running in the browser) per model. An entry of hf/spaces.json with `"space": false` publishes
the model repo only, for a model with no faithful demo (s6-secrets-verify-pico) or none published
(s6-route-nano, whose playground waited on a gate it failed).

    pip install -U huggingface_hub && hf auth login        # a token with write access
    scripts/build-wasm.sh && (cd site && npm run space:build)
    python3 scripts/hf-publish.py [--dry-run] [--only s5-lite,s3-pico] [--user NAME]

Inputs: hf/cards/<model>.md (model cards, `{{USER}}` replaced by the namespace), hf/spaces.json
(one entry per model), site/public/models/ (GGUF files and manifest), site/public/pkg/ (the wasm
modules, SIMD128 and relaxed SIMD), site/public/doom/ (Freedoom textures for the game Spaces),
site/public/ear/ (the sample clips of the ear Spaces), site/public/forecast/ and
site/public/models/forecast.json with TinyCast in models/ref/ (the forecast Spaces) and
target/hf-space-app/ (the demo app), and target/verify-assets/ for an entry with `"engine": true`
(scripts/build-verify-assets.sh: the engine/ folder of the model repo). Everything is assembled under target/hf/<model>/{model,space}
before upload; with --dry-run nothing is uploaded.

A manifest entry with `files` instead of `file` (s6-route: the s6-route-nano file and the
embedding file, fused by `ToolRouter`) puts every one of its files in the model repo and in the
Space; its Space is of kind `router`. An entry of hf/spaces.json may carry `readme_notice`, a
paragraph the Space's README prints first (s6-route: its disclosure).

A model whose manifest entry has `"gate": "failed"` failed its pre-registered gate and is
published as a result (owner's decision of 15/11/2026): its card says so in its first lines, and
its Space opens with the same notice (space.json `gate`, `gate_note`, `gate_url`).
"""
import argparse
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "target" / "hf"
APP = ROOT / "target" / "hf-space-app"
PUBLIC = ROOT / "site" / "public"
ENGINE = ROOT / "target" / "verify-assets"


def space_readme(user: str, s: dict, entry: dict) -> str:
    m = s["model"]
    failed = f"{s['readme_notice']}\n\n" if s.get("readme_notice") else ""
    if entry.get("gate") == "failed":
        failed = (f"**Failed its pre-registered gate: published as a result, not for use.** "
                  f"{entry.get('gate_note', '')}. [The result]({entry.get('gate_url', '')}).\n\n")
    return f"""---
title: {m}
emoji: {s["emoji"]}
colorFrom: {s["color_from"]}
colorTo: {s["color_to"]}
sdk: static
app_file: index.html
pinned: false
license: apache-2.0
short_description: {json.dumps(s["short_description"], ensure_ascii=False)}
models:
  - {user}/{m}
---

# {m}

{failed}{s["tagline"]}

Everything runs in your browser: a Rust inference engine compiled to WebAssembly (relaxed SIMD
where the browser supports it, else SIMD128) reads the GGUF {"files" if len(model_files(entry)) > 1 else "file"} shipped with this Space. Nothing
is sent to a server.

- Model card: [{user}/{m}](https://huggingface.co/{user}/{m})
- Code: [github.com/maxgfr/skycmd](https://github.com/maxgfr/skycmd)
"""


def model_files(entry: dict) -> list[str]:
    """The GGUF files of a manifest entry: `file`, or every file of `files` (a two-file router)."""
    return [f["file"] for f in entry["files"]] if entry.get("files") else [entry["file"]]


def assemble(user: str, s: dict, manifest: list) -> tuple[Path, Path | None]:
    m = s["model"]
    base = OUT / m
    if base.exists():
        shutil.rmtree(base)
    model_dir, space_dir = base / "model", base / "space"
    model_dir.mkdir(parents=True)
    card = (ROOT / "hf" / "cards" / f"{m}.md").read_text().replace("{{USER}}", user)
    entry = next(e for e in manifest if e["name"] == m)
    # a failed model's card opens with the notice: the first lines after the title
    if entry.get("gate") == "failed" and "Failed its pre-registered gate" not in card.split(f"\n# {m}\n", 1)[-1][:300]:
        raise SystemExit(f"hf/cards/{m}.md: a failed model's card must open with its failure notice")
    (model_dir / "README.md").write_text(card)
    if entry.get("files"):
        for f in model_files(entry):
            shutil.copy2(PUBLIC / "models" / f, model_dir / f)
    else:
        shutil.copy2(PUBLIC / "models" / f"{m}.gguf", model_dir / f"{m}.gguf")
    if s.get("engine"):
        # the engine assets of scripts/build-verify-assets.sh, under engine/
        if not (ENGINE / "MANIFEST.json").exists():
            raise SystemExit(f"missing {ENGINE / 'MANIFEST.json'}: run scripts/build-verify-assets.sh")
        shutil.copytree(ENGINE, model_dir / "engine")
    if not s.get("space", True):
        return model_dir, None

    shutil.copytree(APP, space_dir)
    shutil.copytree(PUBLIC / "pkg", space_dir / "pkg", ignore=shutil.ignore_patterns(".gitignore", "package.json", "README.md"))
    (space_dir / "models").mkdir()
    names = [m] + s.get("extra_models", [])
    entries = [e for e in manifest if e["name"] in names]
    for e in entries:
        for f in model_files(e):
            shutil.copy2(PUBLIC / "models" / f, space_dir / "models" / f)
    (space_dir / "models" / "manifest.json").write_text(json.dumps(entries, indent=2) + "\n")
    if s["kind"] == "game":
        shutil.copytree(PUBLIC / "doom", space_dir / "doom")
    if s["kind"] == "ear":
        shutil.copytree(PUBLIC / "ear", space_dir / "ear")
    if s["kind"] == "forecast":
        # the model and TinyCast (Apache-2.0, with its NOTICE); Timestral-5M stays on the site
        doc = json.loads((PUBLIC / "models" / "forecast.json").read_text())
        keep = [f for f in doc["models"] if f["name"] in (m, "TinyCast")]
        (space_dir / "models" / "ref").mkdir()
        for f in keep:
            for rel in (f["file"], f.get("notice")):
                if rel:
                    shutil.copy2(PUBLIC / "models" / rel, space_dir / "models" / rel)
        doc["models"] = keep
        (space_dir / "models" / "forecast.json").write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n")
        (space_dir / "forecast").mkdir()
        shutil.copy2(PUBLIC / "forecast" / "demo-series.json", space_dir / "forecast" / "demo-series.json")
    cfg = {
        "model": m,
        "title": m,
        "tagline": s["tagline"],
        "kind": s["kind"],
        "proofs": s.get("proofs", False),
        "model_repo": f"https://huggingface.co/{user}/{m}",
        "notes": s.get("notes", []),
        "game": s.get("game"),
        "gate": entry.get("gate"),
        "gate_note": entry.get("gate_note"),
        "gate_url": entry.get("gate_url"),
    }
    (space_dir / "space.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n")
    (space_dir / "README.md").write_text(space_readme(user, s, entry))
    return model_dir, space_dir


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="assemble under target/hf, upload nothing")
    ap.add_argument("--only", default="", help="comma-separated model names")
    ap.add_argument("--user", default="", help="namespace (default: the logged-in user)")
    a = ap.parse_args()

    for need in [APP / "index.html", PUBLIC / "pkg" / "sky_web_bg.wasm", PUBLIC / "pkg" / "sky_web_relaxed_bg.wasm"]:
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
        print(f"assembled {s['model']}: {model_dir}" + (f" and {space_dir}" if space_dir else " (model only, no Space)"))
        if api is None:
            continue
        repo = f"{user}/{s['model']}"
        api.create_repo(repo, repo_type="model", exist_ok=True)
        api.upload_folder(repo_id=repo, repo_type="model", folder_path=model_dir,
                          commit_message=f"Publish {s['model']}")
        if space_dir is None:
            print(f"published https://huggingface.co/{repo}")
            continue
        api.create_repo(repo, repo_type="space", space_sdk="static", exist_ok=True)
        api.upload_folder(repo_id=repo, repo_type="space", folder_path=space_dir,
                          commit_message=f"Publish the {s['model']} demo", delete_patterns=["assets/*"])
        print(f"published https://huggingface.co/{repo} and https://huggingface.co/spaces/{repo}")


if __name__ == "__main__":
    main()
