#!/usr/bin/env python3
"""Tests of the verdict notice of scripts/hf-publish.py (the lead of the Space README and the check
of a failed model's card), of the refusal to touch a model repo whose engine/ folder another project
pins, and of the global Space (the entry of kind `hub`), on the repository's manifest, cards and
hf/spaces.json: python3 -I scripts/test_hf_publish.py"""

import importlib.util
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

spec = importlib.util.spec_from_file_location("hf_publish", os.path.join(HERE, "hf-publish.py"))
hp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hp)

with open(os.path.join(ROOT, "site", "public", "models", "manifest.json")) as f:
    MANIFEST = json.load(f)
with open(os.path.join(ROOT, "hf", "spaces.json")) as f:
    SPACES = json.load(f)

PRE_FAILED = "Failed its pre-registered gate: published as a result, not for use."
FIXED_FAILED = "Failed its gate, fixed before training: published as a result, not for use."


def card_of(m: str) -> str:
    with open(os.path.join(ROOT, "hf", "cards", f"{m}.md")) as f:
        return f.read().replace("{{USER}}", "maxgfr")


class GateLead(unittest.TestCase):
    def test_both_leads(self):
        self.assertEqual(hp.gate_lead({"name": "a", "gate": "failed"}), PRE_FAILED)
        self.assertEqual(hp.gate_lead({"name": "a", "gate": "passed"}), "Passed its pre-registered gate.")
        fixed = {"name": "a", "gate_kind": "fixed-before-training"}
        self.assertEqual(hp.gate_lead({**fixed, "gate": "failed"}), FIXED_FAILED)
        self.assertEqual(hp.gate_lead({**fixed, "gate": "passed"}), "Passed its gate, fixed before training.")
        self.assertIsNone(hp.gate_lead({"name": "a"}))
        with self.assertRaises(SystemExit):
            hp.gate_lead({"name": "a", "gate": "failed", "gate_kind": "other"})

    def test_card_check_accepts_the_matching_lead_only(self):
        pre = {"name": "m", "gate": "failed"}
        fixed = {**pre, "gate_kind": "fixed-before-training"}
        hp.check_card("m", f"---\n---\n\n# m\n\n> **{PRE_FAILED}** The gate.\n", pre)
        hp.check_card("m", f"---\n---\n\n# m\n\n> **{FIXED_FAILED}** The gate.\n", fixed)
        for card, entry in [
            (f"# m\n\n> **{FIXED_FAILED}** x\n", pre),
            (f"# m\n\n> **{PRE_FAILED}** x\n", fixed),
            ("# m\n\nIt failed its quality gate.\n", pre),
            ("# m\n\n> **Failed its gate.** x\n", fixed),
        ]:
            with self.assertRaises(SystemExit):
                hp.check_card("m", card, entry)

    def test_every_failed_card_of_the_repository(self):
        for e in MANIFEST:
            if e.get("gate") == "failed":
                hp.check_card(e["name"], card_of(e["name"]), e)

    def test_readme_opens_with_the_lead(self):
        for s in hp.model_spaces(SPACES):
            e = next(x for x in MANIFEST if x["name"] == s["model"])
            body = hp.space_readme("maxgfr", s, e).split("\n---\n", 1)[1]
            first = body.split("\n\n")[1]
            lead = hp.gate_lead(e)
            if lead and (e.get("gate") == "failed" or not s.get("readme_notice")):
                self.assertTrue(first.startswith(f"**{lead}** {e['gate_note']}. [The result]("), s["model"])
            elif not lead:
                self.assertEqual(first, s["tagline"], s["model"])
        s4 = next(x for x in MANIFEST if x["name"] == "s4-micro")
        self.assertEqual(hp.gate_lead(s4), FIXED_FAILED)


class PinnedEngine(unittest.TestCase):
    """The model repo of an entry with "engine": true (s6-secrets-verify-pico) is never touched
    without --allow-pinned-engine: another project pins its engine/ files by sha256."""

    def test_the_repository_has_a_pinned_entry(self):
        self.assertIn("s6-secrets-verify-pico", [s["model"] for s in SPACES if s.get("engine")])

    def test_refused_without_the_flag(self):
        for only in (set(), {"s6-secrets-verify-pico"}, {"s5-lite", "s6-secrets-verify-pico"}):
            why = hp.pinned_refusal(SPACES, only, False, False)
            self.assertIsNotNone(why, only)
            self.assertIn("--space-only", why)
            self.assertIn("--allow-pinned-engine", why)
            with self.assertRaises(SystemExit):
                hp.check_pinned(SPACES, only, False, False)

    def test_allowed_when_left_out_space_only_or_flagged(self):
        self.assertIsNone(hp.pinned_refusal(SPACES, {"s5-lite"}, False, False))
        self.assertIsNone(hp.pinned_refusal(SPACES, {"skycmd"}, False, False))
        self.assertIsNone(hp.pinned_refusal(SPACES, {"s6-secrets-verify-pico"}, True, False))
        self.assertIsNone(hp.pinned_refusal(SPACES, set(), True, False))
        self.assertIsNone(hp.pinned_refusal(SPACES, set(), False, True))


HUB = next(s for s in SPACES if s["kind"] == "hub")


def front_matter(readme: str) -> list[str]:
    """The keys of a README's front matter, in their order."""
    head = readme.split("\n---\n", 1)[0].removeprefix("---\n")
    return [line.split(":", 1)[0] for line in head.split("\n") if line and not line.startswith(" ")]


class Hub(unittest.TestCase):
    """The global Space, maxgfr/skycmd: the entry of kind `hub` of hf/spaces.json."""

    def test_entry_has_no_model_and_no_card(self):
        self.assertEqual([s for s in SPACES if s["kind"] == "hub"], [HUB])
        self.assertNotIn("model", HUB)
        self.assertEqual(hp.key(HUB), "skycmd")
        self.assertFalse(os.path.exists(os.path.join(ROOT, "hf", "cards", "skycmd.md")))
        shown = hp.model_spaces(SPACES)
        self.assertNotIn(HUB, shown)
        self.assertEqual(sorted(s["model"] for s in shown), sorted(e["name"] for e in MANIFEST))

    def test_notes_count_the_manifest(self):
        passed = sum(1 for e in MANIFEST if e.get("gate") == "passed")
        failed = sum(1 for e in MANIFEST if e.get("gate") == "failed")
        self.assertEqual(hp.gate_counts(MANIFEST),
                         {"models": len(MANIFEST), "passed": passed, "failed": failed, "ungated": len(MANIFEST) - passed - failed})
        first = hp.hub_notes(HUB, MANIFEST)[0]
        self.assertTrue(first.startswith(f"{len(MANIFEST)} models, "), first)
        for part in (f": {passed} passed their gate", f", {failed} failed it", f"{len(MANIFEST) - passed - failed} were published without a gate"):
            self.assertIn(part, first)
        for n in hp.hub_notes(HUB, MANIFEST):
            self.assertNotIn("{", n)

    def test_readme(self):
        readme = hp.hub_readme("maxgfr", HUB, SPACES, MANIFEST)
        single = next(s for s in SPACES if s.get("model") == "s3-pico")
        self.assertEqual(front_matter(readme), front_matter(hp.space_readme("maxgfr", single, next(e for e in MANIFEST if e["name"] == "s3-pico"))))
        self.assertTrue(readme.startswith("---\ntitle: skycmd\n"))
        self.assertIn("\nsdk: static\napp_file: index.html\n", readme)
        self.assertIn("models:\n" + "".join(f"  - maxgfr/{s['model']}\n" for s in hp.model_spaces(SPACES)) + "---\n", readme)
        body = readme.split("\n---\n", 1)[1]
        self.assertEqual(body.split("\n\n")[0], "\n# skycmd")
        self.assertEqual(body.split("\n\n")[1], HUB["tagline"])
        # every model once, under the heading of its verdict, with the count of that heading
        sections = {h.split("\n", 1)[0]: h for h in body.split("\n## ")[1:]}
        for gate, title in hp.VERDICTS:
            names = [e["name"] for e in MANIFEST if e.get("gate") == gate]
            heading = f"{title} ({len(names)})"
            self.assertIn(heading, sections)
            for m in names:
                self.assertIn(f"- [{m}](https://huggingface.co/spaces/maxgfr/{m}): ", sections[heading])
        for e in MANIFEST:
            self.assertEqual(body.count(f"- [{e['name']}]("), 1, e["name"])

    def test_assembled_files_are_the_union_of_the_spaces(self):
        """The hub holds every file a model's Space copies, once, with no model card read; the
        app and the wasm modules are left out (copy_app is replaced) and every model file is
        written empty, so the test needs no build."""
        tmp = tempfile.mkdtemp()
        saved = (hp.OUT, hp.copy_app, hp.check_card, shutil.copy2)

        def empty_copy(src, dst):
            open(dst, "w").close()

        def no_card(*args):
            raise AssertionError("the hub read a model card")

        try:
            hp.OUT = Path(tmp) / "hf"
            hp.copy_app = lambda d: d.mkdir(parents=True)
            hp.check_card = no_card
            shutil.copy2 = empty_copy
            space = hp.assemble_hub("maxgfr", HUB, SPACES, MANIFEST)
            union, kept = set(), set()
            for s in hp.model_spaces(SPACES):
                one = Path(tmp) / "one" / s["model"]
                one.mkdir(parents=True)
                _, keep = hp.copy_demo_files(s, MANIFEST, one)
                kept |= {f["name"] for f in keep}
                union |= {str(p.relative_to(one)) for p in one.rglob("*") if p.is_file()}
        finally:
            hp.OUT, hp.copy_app, hp.check_card, shutil.copy2 = saved
        files = {str(p.relative_to(space)) for p in space.rglob("*") if p.is_file()}
        written = {"space.json", "README.md", "models/manifest.json", "models/forecast.json"}
        self.assertEqual(files - written, union)
        self.assertTrue(written <= files)
        for f in ("models/s5-lite.gguf", "models/s6-route-embed.gguf", "models/s6-secrets-verify-dictionary.txt",
                  "models/ref/tinycast.gguf", "models/ref/TINYCAST-NOTICE.txt", "forecast/demo-series.json",
                  "ear/yes.wav", "doom/imp.png"):
            self.assertIn(f, union)
        with open(space / "models" / "manifest.json") as f:
            self.assertEqual(json.load(f), MANIFEST)
        with open(space / "models" / "forecast.json") as f:
            self.assertEqual({m["name"] for m in json.load(f)["models"]}, kept)
        self.assertNotIn("Timestral-5M", kept)
        with open(space / "space.json") as f:
            cfg = json.load(f)
        self.assertEqual(cfg["kind"], "hub")
        self.assertEqual(cfg["notes"], hp.hub_notes(HUB, MANIFEST))
        by_name = {e["name"]: e for e in MANIFEST}
        self.assertEqual(
            cfg["spaces"],
            [{"config": hp.space_config("maxgfr", s, by_name[s["model"]]),
              "models": [s["model"]] + s.get("extra_models", []),
              "space_url": f"https://huggingface.co/spaces/maxgfr/{s['model']}"} for s in hp.model_spaces(SPACES)],
        )
        shutil.rmtree(tmp)


if __name__ == "__main__":
    unittest.main()
