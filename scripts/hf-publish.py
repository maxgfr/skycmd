#!/usr/bin/env python3
"""Publish every skycmd model to Hugging Face: one model repo (card + GGUF) and one static Space
(demo running in the browser) per model. An entry of hf/spaces.json with `"space": false` publishes
the model repo only.

    pip install -U huggingface_hub && hf auth login        # a token with write access
    scripts/build-wasm.sh && (cd site && npm run space:build)
    python3 scripts/hf-publish.py [--dry-run] [--only s5-lite,s3-pico] [--user NAME] [--space-only]

`--space-only` assembles and uploads the Space of each selected entry and nothing else: the model
repo (card, GGUF and, for an entry with `"engine": true`, its engine/ folder) is neither assembled
nor uploaded. s6-secrets-verify-pico's Space is published that way, because the engine/ folder of
its model repo must never be re-uploaded: another project pins its files by sha256.

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
Space; its Space is of kind `router`. s6-route-nano's Space (kind `router-v2`) lists s6-route in
`extra_models`, so it carries both files too and shows the s6-route-nano file alone. A manifest
entry with `dictionary` (s6-secrets-verify-pico: the verifier's word list) puts that file in the
Space's models/ folder as well. An entry of hf/spaces.json may carry `readme_notice`, a
paragraph the Space's README prints first (s6-route: its disclosure).

A model whose manifest entry has `"gate": "failed"` failed its pre-registered gate and is
published as a result (owner's decision of 15/11/2026): its card says so in its first lines, and
its Space opens with the same notice (space.json `gate`, `gate_kind`, `gate_note`, `gate_url`;
`gate_kind: "fixed-before-training"` turns "pre-registered gate" into "gate, fixed before
training" for a gate with no pre-registration document, s4-micro's). The Space of a
model whose entry has `"gate": "passed"` opens with a notice of the same structure that says it
passed (its README too, unless the entry of hf/spaces.json has a `readme_notice`); a model with
no `gate` was published without one and gets no verdict notice.

The entry with `"kind": "hub"` has no `model`: its `space_name` (skycmd) names the global Space
`<user>/skycmd`, which has no model repository and no card. It gathers every demo: the app, the
wasm modules once, the full manifest, the union of the files the other Spaces ship (no file
twice; forecast.json keeps the forecasters those Spaces keep), and a `space.json` that holds the
`space.json` of each model's own Space. Its notes and its README count the manifest's models and
gates when it is assembled (`{models}`, `{passed}`, `{failed}`, `{ungated}` in its notes); its
README lists the models by verdict. `--only skycmd` selects it; `--space-only` changes nothing
for it. The script prints its total size and its largest files, and stops if the total is above
1 GB or a file is larger than the largest one a model's own Space already ships.
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


# How the gate was fixed (manifest `gate_kind`, as site/.vitepress/theme/lib/gate.ts reads it):
# pre-registered by default (a pre-registration document in docs/), or written down before
# training without one (s4-micro: GitHub issue #1).
GATE_WORDS = {"pre-registered": "pre-registered gate", "fixed-before-training": "gate, fixed before training"}


def gate_lead(entry: dict) -> str | None:
    """The bold lead of the verdict notice, word for word as lib/gate.ts writes it; None for a
    model published without a gate."""
    kind = entry.get("gate_kind") or "pre-registered"
    if kind not in GATE_WORDS:
        raise SystemExit(f"{entry['name']}: unknown gate_kind {kind!r}")
    if entry.get("gate") == "failed":
        return f"Failed its {GATE_WORDS[kind]}: published as a result, not for use."
    if entry.get("gate") == "passed":
        return f"Passed its {GATE_WORDS[kind]}."
    return None


def check_card(m: str, card: str, entry: dict) -> None:
    """A failed model's card opens, in the first lines after its title, with the bold lead of its
    failure notice, the one that matches its `gate_kind`, and no other."""
    if entry.get("gate") == "failed" and f"**{gate_lead(entry)}**" not in card.split(f"\n# {m}\n", 1)[-1][:300]:
        raise SystemExit(f"hf/cards/{m}.md: a failed model's card must open with its failure notice "
                         f"(**{gate_lead(entry)}**)")


def space_readme(user: str, s: dict, entry: dict) -> str:
    m = s["model"]
    failed = f"{s['readme_notice']}\n\n" if s.get("readme_notice") else ""
    lead = gate_lead(entry)
    if lead and (entry.get("gate") == "failed" or not failed):
        failed = f"**{lead}** {entry.get('gate_note', '')}. [The result]({entry.get('gate_url', '')}).\n\n"
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
where the browser supports it, else SIMD128) reads the GGUF {"files" if len(model_files(entry)) > 1 or s["kind"] == "router-v2" else "file"} shipped with this Space. Nothing
is sent to a server.

- Model card: [{user}/{m}](https://huggingface.co/{user}/{m})
- Code: [github.com/maxgfr/skycmd](https://github.com/maxgfr/skycmd)
"""


def model_files(entry: dict) -> list[str]:
    """The GGUF files of a manifest entry: `file`, or every file of `files` (a two-file router)."""
    return [f["file"] for f in entry["files"]] if entry.get("files") else [entry["file"]]


def assemble(user: str, s: dict, manifest: list, space_only: bool = False) -> tuple[Path | None, Path | None]:
    """Assembles target/hf/<model>/model and target/hf/<model>/space. With `space_only` the model
    repo (card, GGUF, engine/) is not assembled and the first path is None."""
    m = s["model"]
    base = OUT / m
    if base.exists():
        shutil.rmtree(base)
    model_dir, space_dir = base / "model", base / "space"
    card = (ROOT / "hf" / "cards" / f"{m}.md").read_text().replace("{{USER}}", user)
    entry = next(e for e in manifest if e["name"] == m)
    check_card(m, card, entry)
    if space_only:
        if not s.get("space", True):
            return None, None
        base.mkdir(parents=True)
    else:
        model_dir.mkdir(parents=True)
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

    copy_app(space_dir)
    entries, keep = copy_demo_files(s, manifest, space_dir)
    (space_dir / "models" / "manifest.json").write_text(json.dumps(entries, indent=2) + "\n")
    if s["kind"] == "forecast":
        write_forecast(space_dir, keep)
    (space_dir / "space.json").write_text(json.dumps(space_config(user, s, entry), indent=2, ensure_ascii=False) + "\n")
    (space_dir / "README.md").write_text(space_readme(user, s, entry))
    return (None if space_only else model_dir), space_dir


def key(s: dict) -> str:
    """The name of an entry of hf/spaces.json: its model, or the `space_name` of the hub."""
    return s.get("model") or s["space_name"]


def copy_app(space_dir: Path) -> None:
    """The demo app (target/hf-space-app) and the two wasm modules with their glue."""
    shutil.copytree(APP, space_dir)
    shutil.copytree(PUBLIC / "pkg", space_dir / "pkg", ignore=shutil.ignore_patterns(".gitignore", "package.json", "README.md"))


def copy_demo_files(s: dict, manifest: list, space_dir: Path) -> tuple[list, list]:
    """Copies what the demo of a model's Space reads (its model files and those of its
    `extra_models`, a word list, the Freedoom textures of a game, the sample clips of an ear
    Space, the forecasters and the demo series of a forecast Space) into `space_dir`, which may
    already hold the files of another Space (the hub): a file is then written once, at its one
    path. Returns the manifest entries of the Space and the forecast.json models it keeps."""
    m = s["model"]
    (space_dir / "models").mkdir(exist_ok=True)
    names = [m] + s.get("extra_models", [])
    entries = [e for e in manifest if e["name"] in names]
    for e in entries:
        for f in model_files(e) + ([e["dictionary"]] if e.get("dictionary") else []):
            shutil.copy2(PUBLIC / "models" / f, space_dir / "models" / f)
    if s["kind"] == "game":
        shutil.copytree(PUBLIC / "doom", space_dir / "doom", dirs_exist_ok=True)
    if s["kind"] == "ear":
        shutil.copytree(PUBLIC / "ear", space_dir / "ear", dirs_exist_ok=True)
    keep = []
    if s["kind"] == "forecast":
        # the model and TinyCast (Apache-2.0, with its NOTICE); Timestral-5M stays on the site
        doc = json.loads((PUBLIC / "models" / "forecast.json").read_text())
        keep = [f for f in doc["models"] if f["name"] in (m, "TinyCast")]
        (space_dir / "models" / "ref").mkdir(exist_ok=True)
        for f in keep:
            for rel in (f["file"], f.get("notice")):
                if rel:
                    shutil.copy2(PUBLIC / "models" / rel, space_dir / "models" / rel)
        (space_dir / "forecast").mkdir(exist_ok=True)
        shutil.copy2(PUBLIC / "forecast" / "demo-series.json", space_dir / "forecast" / "demo-series.json")
    return entries, keep


def write_forecast(space_dir: Path, keep: list) -> None:
    """forecast.json with the forecasters of `keep` only, in its own order."""
    doc = json.loads((PUBLIC / "models" / "forecast.json").read_text())
    names = {f["name"] for f in keep}
    doc["models"] = [f for f in doc["models"] if f["name"] in names]
    (space_dir / "models" / "forecast.json").write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n")


def space_config(user: str, s: dict, entry: dict) -> dict:
    """The `space.json` of a model's Space (site/space/hub.ts, `SpaceConfig`)."""
    m = s["model"]
    return {
        "model": m,
        "title": m,
        "tagline": s["tagline"],
        "kind": s["kind"],
        "proofs": s.get("proofs", False),
        "model_repo": f"https://huggingface.co/{user}/{m}",
        "notes": s.get("notes", []),
        "game": s.get("game"),
        "gate": entry.get("gate"),
        "gate_kind": entry.get("gate_kind"),
        "gate_note": entry.get("gate_note"),
        "gate_url": entry.get("gate_url"),
    }


def model_spaces(spaces: list) -> list:
    """The entries of hf/spaces.json that have a model and a Space: the demos of the hub."""
    return [s for s in spaces if s.get("model") and s.get("space", True)]


def gate_counts(manifest: list) -> dict:
    """The manifest's models, and how many passed their gate, failed it or have none."""
    passed = sum(1 for e in manifest if e.get("gate") == "passed")
    failed = sum(1 for e in manifest if e.get("gate") == "failed")
    return {"models": len(manifest), "passed": passed, "failed": failed, "ungated": len(manifest) - passed - failed}


def hub_notes(hub: dict, manifest: list) -> list[str]:
    """The notes of the hub, with the counts of the manifest in their `{...}` fields."""
    counts = gate_counts(manifest)
    return [n.format(**counts) for n in hub.get("notes", [])]


# The hub's README: the models under their verdict, as the manifest gives it.
VERDICTS = [
    ("passed", "Passed its gate"),
    ("failed", "Failed its gate, published as a result"),
    (None, "Published without a gate"),
]


def hub_readme(user: str, hub: dict, spaces: list, manifest: list) -> str:
    shown = model_spaces(spaces)
    by_name = {e["name"]: e for e in manifest}
    models = "".join(f"  - {user}/{s['model']}\n" for s in shown)
    notes = "".join(f"- {n}\n" for n in hub_notes(hub, manifest))
    groups = ""
    for gate, title in VERDICTS:
        rows = [s for s in shown if by_name[s["model"]].get("gate") == gate]
        if not rows:
            continue
        groups += f"\n## {title} ({len(rows)})\n\n" + "".join(
            f"- [{s['model']}](https://huggingface.co/spaces/{user}/{s['model']}): {s['short_description']}\n" for s in rows
        )
    return f"""---
title: {hub["title"]}
emoji: {hub["emoji"]}
colorFrom: {hub["color_from"]}
colorTo: {hub["color_to"]}
sdk: static
app_file: index.html
pinned: false
license: apache-2.0
short_description: {json.dumps(hub["short_description"], ensure_ascii=False)}
models:
{models}---

# {hub["title"]}

{hub["tagline"]}

{notes}{groups}
Everything runs in your browser: a Rust inference engine compiled to WebAssembly (relaxed SIMD
where the browser supports it, else SIMD128) reads the GGUF files shipped with this Space. A
model's files are downloaded only when its demo is opened. Nothing is sent to a server.

- Project site: [maxgfr.github.io/skycmd](https://maxgfr.github.io/skycmd)
- Code: [github.com/maxgfr/skycmd](https://github.com/maxgfr/skycmd)
"""


def assemble_hub(user: str, hub: dict, spaces: list, manifest: list) -> Path:
    """Assembles target/hf/<space_name>/space: the global Space, which opens the demo of every
    model's Space on demand (site/space/HubView.vue)."""
    base = OUT / key(hub)
    if base.exists():
        shutil.rmtree(base)
    space_dir = base / "space"
    copy_app(space_dir)
    by_name = {e["name"]: e for e in manifest}
    kept, demos = [], []
    for s in model_spaces(spaces):
        _, keep = copy_demo_files(s, manifest, space_dir)
        kept += keep
        demos.append({
            "config": space_config(user, s, by_name[s["model"]]),
            "models": [s["model"]] + s.get("extra_models", []),
            "space_url": f"https://huggingface.co/spaces/{user}/{s['model']}",
        })
    (space_dir / "models" / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if kept:
        write_forecast(space_dir, kept)
    cfg = {
        "kind": "hub",
        "title": hub["title"],
        "tagline": hub["tagline"],
        "notes": hub_notes(hub, manifest),
        "spaces": demos,
    }
    (space_dir / "space.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n")
    (space_dir / "README.md").write_text(hub_readme(user, hub, spaces, manifest))
    return space_dir


def mb(n: int) -> str:
    return f"{n / 1e6:.1f} MB"


def check_hub_size(space_dir: Path, spaces: list, manifest: list) -> None:
    """Prints the hub's total size and its largest files; stops if the total is above 1 GB or if
    a file is larger than the largest file a model's own Space ships (the files of the hub are
    stored as those Spaces store theirs)."""
    files = sorted((p for p in space_dir.rglob("*") if p.is_file()), key=lambda p: p.stat().st_size, reverse=True)
    total = sum(p.stat().st_size for p in files)
    print(f"  {len(files)} files, {mb(total)} in all; the largest:")
    for p in files[:5]:
        print(f"    {mb(p.stat().st_size):>9}  {p.relative_to(space_dir)}")
    single = 0
    for s in model_spaces(spaces):
        for e in (e for e in manifest if e["name"] in [s["model"]] + s.get("extra_models", [])):
            for f in model_files(e) + ([e["dictionary"]] if e.get("dictionary") else []):
                single = max(single, (PUBLIC / "models" / f).stat().st_size)
    if total > 1e9:
        raise SystemExit(f"the hub is {mb(total)}, above 1 GB: stopped before changing how files are stored")
    if files and files[0].stat().st_size > single:
        raise SystemExit(f"{files[0].relative_to(space_dir)} ({mb(files[0].stat().st_size)}) is larger than any file a "
                         f"model's own Space ships ({mb(single)}): stopped before changing how files are stored")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="assemble under target/hf, upload nothing")
    ap.add_argument("--only", default="", help="comma-separated model names (skycmd: the global Space)")
    ap.add_argument("--user", default="", help="namespace (default: the logged-in user)")
    ap.add_argument("--space-only", action="store_true",
                    help="assemble and upload the Space only: the model repo (and its engine/ folder) is left untouched")
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
        if only and key(s) not in only:
            continue
        if s["kind"] == "hub":
            # the global Space: no model repository, so --space-only changes nothing here
            space_dir = assemble_hub(user, s, spaces, manifest)
            print(f"assembled {key(s)}: {space_dir} (the global Space, no model repository)")
            check_hub_size(space_dir, spaces, manifest)
            if api is None:
                continue
            repo = f"{user}/{key(s)}"
            api.create_repo(repo, repo_type="space", space_sdk="static", exist_ok=True)
            api.upload_folder(repo_id=repo, repo_type="space", folder_path=space_dir,
                              commit_message="Publish every skycmd demo", delete_patterns=["assets/*"])
            print(f"published https://huggingface.co/spaces/{repo}")
            continue
        model_dir, space_dir = assemble(user, s, manifest, space_only=a.space_only)
        if model_dir is None and space_dir is None:
            print(f"skipped {s['model']}: no Space, and --space-only leaves the model repo alone")
            continue
        if model_dir is None:
            print(f"assembled {s['model']}: {space_dir} (Space only)")
        else:
            print(f"assembled {s['model']}: {model_dir}" + (f" and {space_dir}" if space_dir else " (model only, no Space)"))
        if api is None:
            continue
        repo = f"{user}/{s['model']}"
        if model_dir is not None:
            api.create_repo(repo, repo_type="model", exist_ok=True)
            api.upload_folder(repo_id=repo, repo_type="model", folder_path=model_dir,
                              commit_message=f"Publish {s['model']}")
        if space_dir is None:
            print(f"published https://huggingface.co/{repo}")
            continue
        api.create_repo(repo, repo_type="space", space_sdk="static", exist_ok=True)
        api.upload_folder(repo_id=repo, repo_type="space", folder_path=space_dir,
                          commit_message=f"Publish the {s['model']} demo", delete_patterns=["assets/*"])
        print(f"published https://huggingface.co/spaces/{repo}" if model_dir is None
              else f"published https://huggingface.co/{repo} and https://huggingface.co/spaces/{repo}")


if __name__ == "__main__":
    main()
