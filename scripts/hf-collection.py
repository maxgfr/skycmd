#!/usr/bin/env python3
"""Create or update the public Hugging Face collection "skycmd" of the project's owner: every
model, Space and data set the project has on the Hub, in one order and with one note each.

    python3 scripts/hf-collection.py [--user maxgfr]            # dry run: prints the plan
    uv run --no-project --with huggingface_hub python3 scripts/hf-collection.py --apply

The order: the hub Space (`<user>/skycmd`) first; then the models grouped by verdict, those that
passed their gate, those that failed it, those published without one (hf/spaces.json order
within a group); then the Space of each model, in the same order; then the 16 data sets
(PUBLISHED_SETS of scripts/hf-publish-data.py).

The notes are read, never typed here (except the one data set no script describes any more):
  * a model: the verdict lead of its card (the one scripts/hf-publish.py checks, from the
    manifest's `gate` and `gate_kind`), then the first sentence of its manifest `gate_note`, or,
    for a model without a gate, its tagline (hf/spaces.json);
  * a Space: its `short_description` (hf/spaces.json); the hub, its tagline;
  * a data set: the models trained or tested on it, with their verdicts (hf/spaces.json and the
    manifest, as the data set cards say), or the summary of its card when no model reads it.

Without --apply nothing is written: the script reads the collection, when it exists, and prints
what --apply would change. --apply creates the collection if there is none, then adds the
missing items, sets the notes that differ, removes the items that are not in the plan, and puts
every item at its place; a second run changes nothing. Read-only calls need no token.
"""
import argparse
import importlib.util
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TITLE = "skycmd"
DESCRIPTION = ("Tiny decision models trained from scratch in Rust that run in the browser: every model with its "
               "verdict, its demo and the data it read.")
# the Hub's limit on the note of a collection item
NOTE_MAX = 500
# the one data set whose card no script writes any more (its entry left with the LFM3 code)
NOTES_WITHOUT_SCRIPT = {
    "skycmd-math": "The calibration and test questions of the LFM3-1.7B System 1 / System 2 router, which failed "
                   "its pre-registered gate; no published model reads them.",
}
GROUPS = [("passed", "passed its gate"), ("failed", "failed its gate"), (None, "published without a gate")]


def load(name: str, file: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


hp = load("hf_publish", "hf-publish.py")
hpd = load("hf_publish_data", "hf-publish-data.py")


def first_sentence(text: str) -> str:
    """The first sentence of a manifest note: up to the first '. ' outside parentheses."""
    depth = 0
    for i, c in enumerate(text):
        depth += c in "(["
        depth -= c in ")]"
        if c == "." and depth == 0 and (i + 1 == len(text) or text[i + 1] == " "):
            return text[: i + 1]
    return text.rstrip(".") + "."


def clip(note: str) -> str:
    """A note within the Hub's limit, cut at a word."""
    if len(note) <= NOTE_MAX:
        return note
    return note[: NOTE_MAX - 1].rsplit(" ", 1)[0].rstrip(",;:") + "…"


def model_note(s: dict, entry: dict, card: str = None) -> str:
    """The note of a model: its verdict lead, then what the manifest says of its gate (its tagline
    when it has none). With `card` (the text of hf/cards/<model>.md), the card is checked to open
    with the same lead."""
    if card is not None:
        hp.check_card(s["model"], card, entry)
    lead = hp.verdict_lead(entry)
    rest = first_sentence(entry["gate_note"]) if entry.get("gate_note") else s["tagline"]
    return clip(f"{lead} {rest}")


def dataset_note(repo: str, spaces: list, manifest: list) -> str:
    rows = hpd.model_readers(repo, spaces, manifest)
    if rows:
        words = {"passed": "passed", "failed": "failed", None: "no gate"}
        names = ", ".join(f"{m} ({words[g]})" for m, g, _ in rows)
        return clip(f"Trained or tested on it: {names}.")
    if repo in NOTES_WITHOUT_SCRIPT:
        return NOTES_WITHOUT_SCRIPT[repo]
    s = next((x for x in hpd.SETS.values() if x["repo"] == repo), None)
    if s is None:
        raise SystemExit(f"{repo}: no model reads it and it has no summary")
    return clip(re.sub(r"`", "", s["summary"]))


def desired_items(user: str, spaces: list, manifest: list, cards: dict = None) -> list:
    """(item id, item type, note) of every item of the collection, in its order. `cards`, model ->
    card text, checks that each card opens with the verdict its note gives."""
    by_name = {e["name"]: e for e in manifest}
    hub = next(s for s in spaces if s["kind"] == "hub")
    shown = hp.model_spaces(spaces)
    if sorted(s["model"] for s in shown) != sorted(by_name):
        raise SystemExit("hf/spaces.json and the manifest do not hold the same models")
    models = [s for gate, _ in GROUPS for s in shown if by_name[s["model"]].get("gate") == gate]
    items = [(f"{user}/{hub['space_name']}", "space", clip(hub["tagline"]))]
    items += [(f"{user}/{s['model']}", "model", model_note(s, by_name[s["model"]], (cards or {}).get(s["model"])))
              for s in models]
    items += [(f"{user}/{s['model']}", "space", clip(f"Demo: {s['short_description']}.")) for s in models]
    items += [(f"{user}/{d}", "dataset", dataset_note(d, spaces, manifest)) for d in hpd.PUBLISHED_SETS]
    keys = [(i, t) for i, t, _ in items]
    if len(set(keys)) != len(keys):
        raise SystemExit("an item is twice in the plan")
    return items


def plan(desired: list, current: list) -> list:
    """The operations that turn `current` (dicts: id, type, object_id, note, position) into
    `desired` ((id, type, note) in order): ("add", id, type, note), ("note", object_id, note),
    ("delete", object_id, id), ("position", object_id_or_key, index). Empty when they agree."""
    ops = []
    want = {(i, t): n for i, t, n in desired}
    have = {(c["id"], c["type"]): c for c in current}
    for c in current:
        if (c["id"], c["type"]) not in want:
            ops.append(("delete", c["object_id"], c["id"]))
    for i, t, n in desired:
        c = have.get((i, t))
        if c is None:
            ops.append(("add", i, t, n))
        elif (c.get("note") or "") != n:
            ops.append(("note", c["object_id"], n))
    kept = sorted((c for c in current if (c["id"], c["type"]) in want), key=lambda c: c["position"])
    order_now = [(c["id"], c["type"]) for c in kept] + [(i, t) for i, t, _ in desired if (i, t) not in have]
    order_want = [(i, t) for i, t, _ in desired]
    if order_now != order_want:
        ops += [("position", key, n) for n, key in enumerate(order_want)]
    return ops


def current_items(collection) -> list:
    return [{"id": x.item_id, "type": x.item_type, "object_id": x.item_object_id, "note": x.note,
             "position": x.position} for x in collection.items]


def find_collection(api, user: str):
    for c in api.list_collections(owner=user, limit=100):
        if c.title == TITLE:
            return api.get_collection(c.slug)
    return None


def apply(api, user: str, desired: list) -> None:
    collection = find_collection(api, user) or api.create_collection(TITLE, namespace=user, description=DESCRIPTION,
                                                                     private=False, exists_ok=True)
    slug = collection.slug
    if collection.description != DESCRIPTION or collection.private:
        api.update_collection_metadata(slug, description=DESCRIPTION, private=False)
    for op in plan(desired, current_items(collection)):
        if op[0] == "delete":
            api.delete_collection_item(slug, op[1], missing_ok=True)
        elif op[0] == "add":
            api.add_collection_item(slug, op[1], op[2], note=op[3], exists_ok=True)
        elif op[0] == "note":
            api.update_collection_item(slug, op[1], note=op[2])
    # positions, by object id, once every item is there, and only when the order differs
    items = sorted(current_items(api.get_collection(slug)), key=lambda c: c["position"])
    if [(c["id"], c["type"]) for c in items] != [(i, t) for i, t, _ in desired]:
        by_key = {(c["id"], c["type"]): c for c in items}
        for n, (i, t, _) in enumerate(desired):
            api.update_collection_item(slug, by_key[(i, t)]["object_id"], position=n)
    left = plan(desired, current_items(api.get_collection(slug)))
    print(f"applied: https://huggingface.co/collections/{slug}" + (f" ({len(left)} operations left: run again)" if left else ""))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--user", default="maxgfr", help="the owner of the collection")
    ap.add_argument("--apply", action="store_true", help="write the collection (default: dry run)")
    a = ap.parse_args()
    spaces = json.loads((ROOT / "hf" / "spaces.json").read_text())
    manifest = json.loads((ROOT / "site" / "public" / "models" / "manifest.json").read_text())
    cards = {s["model"]: (ROOT / "hf" / "cards" / f"{s['model']}.md").read_text().replace("{{USER}}", a.user)
             for s in hp.model_spaces(spaces)}
    desired = desired_items(a.user, spaces, manifest, cards)
    for n, (i, t, note) in enumerate(desired):
        print(f"{n:2d}  {t:7s}  {i}\n      {note}")
    from huggingface_hub import HfApi

    api = HfApi()
    if a.apply:
        apply(api, a.user, desired)
        return
    existing = find_collection(api, a.user)
    if existing is None:
        print(f"dry run: no collection titled {TITLE!r} under {a.user}; --apply creates it with these {len(desired)} items")
        return
    ops = plan(desired, current_items(existing))
    print(f"dry run: https://huggingface.co/collections/{existing.slug}: " +
          (f"{len(ops)} operations with --apply" if ops else "already as planned, nothing to change"))
    for op in ops:
        print("   ", *op)


if __name__ == "__main__":
    main()
