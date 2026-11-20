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

    def test_every_card_of_the_repository_opens_with_its_verdict(self):
        for e in MANIFEST:
            hp.check_card(e["name"], card_of(e["name"]), e)
            first = card_of(e["name"]).split(f"\n# {e['name']}\n", 1)[1].lstrip("\n").split("\n", 1)[0]
            self.assertTrue(first.startswith(f"> **{hp.verdict_lead(e)}**"), e["name"])

    def test_a_card_without_a_gate_opens_with_the_ungated_lead(self):
        none = {"name": "m"}
        hp.check_card("m", f"---\n---\n\n# m\n\n> **{hp.UNGATED_LEAD}** x\n", none)
        for card in ("---\n---\n\n# m\n\n**A model.**\n", f"---\n---\n\n# m\n\n> **{PRE_FAILED}** x\n"):
            with self.assertRaises(SystemExit):
                hp.check_card("m", card, none)

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
        single = next(s for s in SPACES if s.get("model") == "s5-lite")
        self.assertEqual(front_matter(readme), front_matter(hp.space_readme("maxgfr", single, next(e for e in MANIFEST if e["name"] == "s5-lite"))))
        self.assertTrue(readme.startswith("---\ntitle: skycmd\n"))
        self.assertIn("\nsdk: static\napp_file: index.html\n", readme)
        self.assertIn("models:\n" + "".join(f"  - maxgfr/{s['model']}\n" for s in hp.model_spaces(SPACES)) + "datasets:\n", readme)
        self.assertIn(f"Technical report: [maxgfr.github.io/skycmd/paper]({hp.REPORT})", readme)
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


BY_NAME = {e["name"]: e for e in MANIFEST}


def raw_card(m: str) -> str:
    with open(os.path.join(ROOT, "hf", "cards", f"{m}.md")) as f:
        return f.read()


class ModelCards(unittest.TestCase):
    """The conventions of the 20 model cards, checked and completed by model_card."""

    def test_every_card_assembles(self):
        for s in hp.model_spaces(SPACES):
            m = s["model"]
            card = hp.model_card("maxgfr", s, BY_NAME[m], raw_card(m))
            fm, body = hp.parse_front_matter(card)
            self.assertNotIn("{{", card, m)
            for t in hp.COMMON_TAGS + hp.VERDICT_TAGS[BY_NAME[m].get("gate")]:
                self.assertIn(t, fm["tags"], m)
            want = hp.parse_front_matter(raw_card(m))[0].get("datasets", []) + [f"maxgfr/{d}" for d in s["datasets"]["train"]]
            self.assertEqual(fm.get("datasets", []), want, m)
            # the links every card gives, in the same words
            for line in ("- **All results, failures included:** [results page](",
                         f"- **Technical report:** [maxgfr.github.io/skycmd/paper]({hp.REPORT})",
                         "- **Pre-registration:** ",
                         "- **Every demo, with its verdict:** [the skycmd Space](https://huggingface.co/spaces/maxgfr/skycmd)"):
                self.assertEqual(body.count(line), 1, (m, line))
            self.assertIn(s["results_url"], body, m)
            self.assertIn(f"https://huggingface.co/spaces/maxgfr/{m})", body, m)

    def test_every_model_has_its_results_and_data(self):
        for s in hp.model_spaces(SPACES):
            m = s["model"]
            self.assertTrue(s["results_url"].startswith((hp.SITE + "/results/#", hp.REPO + "/blob/main/BENCH.md#")), m)
            self.assertEqual(set(s["datasets"]), {"train", "test"}, m)
            gated = BY_NAME[m].get("gate") is not None
            self.assertEqual(bool(s.get("prereg")), gated or m == "s5-lite", m)
            if not s["datasets"]["train"]:
                self.assertTrue(s.get("data_note"), f"{m}: no published training set and no data_note")

    def test_links_of_three_kinds_of_entry(self):
        by = {s["model"]: s for s in hp.model_spaces(SPACES)}
        s3 = hp.links_block("maxgfr", by["s3-pico"])
        self.assertIn("- **Pre-registration:** none: this file was published without a gate", s3)
        self.assertIn(f"[results page]({hp.SITE}/results/); this model's numbers: [BENCH.md]({hp.REPO}/blob/main/BENCH.md#", s3)
        s4 = hp.links_block("maxgfr", by["s4-micro"])
        self.assertIn("[issue #1](https://github.com/maxgfr/skycmd/issues/1) (the gate, written down before training", s4)
        r7 = hp.links_block("maxgfr", by["s6-route7-nano"])
        prereg = r7.split("- **Pre-registration:** ", 1)[1].split("\n", 1)[0]
        self.assertEqual(prereg.count(f"]({hp.REPO}/blob/main/docs/"), 4)
        self.assertIn("trained on [skycmd-route7](https://huggingface.co/datasets/maxgfr/skycmd-route7); tested on "
                      "[skycmd-route7](https://huggingface.co/datasets/maxgfr/skycmd-route7), [skycmd-route](", r7)

    def test_front_matter_refusals(self):
        s = next(x for x in SPACES if x.get("model") == "s3-pico")
        e = BY_NAME["s3-pico"]
        good = raw_card("s3-pico")
        hp.model_card("maxgfr", s, e, good)
        for bad in (good.replace("  - wasm\n", ""),                      # a common tag missing
                    good.replace("  - no-gate\n", "  - passed-gate\n"),  # the verdict tag of another gate
                    good.replace("library_name: gguf\n", "library_name: gguf\ndatasets:\n  - maxgfr/skycmd-games\n"),
                    good.replace("pipeline_tag: text-classification\n", ""),
                    good.replace("{{LINKS}}\n", ""),
                    good.replace("60bf4d612be0a48a", "60bf4d612be0a48b"),  # a sha256 that is not the manifest's
                    good.replace("251,104 bytes", "251,105 bytes")):
            self.assertNotEqual(bad, good)
            with self.assertRaises(SystemExit):
                hp.model_card("maxgfr", s, e, bad)

    def test_front_matter_round_trip(self):
        for s in hp.model_spaces(SPACES):
            fm, body = hp.parse_front_matter(raw_card(s["model"]))
            self.assertEqual(hp.render_front_matter(fm) + body, raw_card(s["model"]), s["model"])
        with self.assertRaises(SystemExit):
            hp.parse_front_matter("---\nlicense: x\n  nested: y\n---\nbody")


class CardOnly(unittest.TestCase):
    """--card-only uploads README.md alone, for every model, the pinned one included."""

    def test_uploads_the_card_alone(self):
        calls = []

        class Api:
            def upload_file(self, **kw):
                calls.append(kw)

            def __getattr__(self, name):
                raise AssertionError(f"--card-only called {name}")

        tmp = tempfile.mkdtemp()
        saved = hp.OUT
        try:
            hp.OUT = Path(tmp)
            hp.publish_cards(Api(), "maxgfr", SPACES, MANIFEST, {"s6-secrets-verify-pico", "s3-pico", "skycmd"})
        finally:
            hp.OUT = saved
        self.assertEqual(sorted(c["repo_id"] for c in calls), ["maxgfr/s3-pico", "maxgfr/s6-secrets-verify-pico"])
        for c in calls:
            self.assertEqual((c["path_in_repo"], c["repo_type"]), ("README.md", "model"))
            with open(c["path_or_fileobj"]) as f:
                self.assertTrue(f.read().startswith("---\nlicense: apache-2.0\n"))
        self.assertEqual(sorted(str(p.relative_to(tmp)) for p in Path(tmp).rglob("*") if p.is_file()),
                         ["s3-pico/card/README.md", "s6-secrets-verify-pico/card/README.md"])
        shutil.rmtree(tmp)

    def test_the_pinned_model_is_not_refused(self):
        self.assertIsNone(hp.pinned_refusal(SPACES, {"s6-secrets-verify-pico"}, True, False))


class SpaceReadmes(unittest.TestCase):
    def test_front_matter_of_every_space(self):
        for s in hp.model_spaces(SPACES):
            e = BY_NAME[s["model"]]
            fm, body = hp.parse_front_matter(hp.space_readme("maxgfr", s, e))
            self.assertLessEqual(len(s["short_description"]), hp.SHORT_DESCRIPTION_MAX, s["model"])
            self.assertEqual(fm["models"], [f"maxgfr/{x}" for x in [s["model"]] + s.get("extra_models", [])])
            self.assertEqual(fm.get("datasets", []), [f"maxgfr/{d}" for d in s.get("space_datasets", [])])
            self.assertEqual(fm["tags"], hp.SPACE_TAGS + hp.VERDICT_TAGS[e.get("gate")])
            self.assertEqual(fm["sdk"], "static")
            if e.get("gate") == "failed":
                self.assertTrue(s["short_description"].startswith("Failed its gate"), s["model"])
            self.assertIn("(https://huggingface.co/spaces/maxgfr/skycmd)", body)
            self.assertIn(f"]({hp.REPORT})", body)

    def test_a_long_short_description_is_refused(self):
        s = {**next(x for x in SPACES if x.get("model") == "s3-pico"), "short_description": "x" * 61}
        with self.assertRaises(SystemExit):
            hp.space_readme("maxgfr", s, BY_NAME["s3-pico"])


if __name__ == "__main__":
    unittest.main()
