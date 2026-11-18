#!/usr/bin/env python3
"""Tests of the verdict notice of scripts/hf-publish.py (the lead of the Space README and the check
of a failed model's card), on the repository's manifest and cards: python3 -I scripts/test_hf_publish.py"""

import importlib.util
import json
import os
import unittest

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
        for s in SPACES:
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


if __name__ == "__main__":
    unittest.main()
