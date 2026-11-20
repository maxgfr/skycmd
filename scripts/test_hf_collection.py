#!/usr/bin/env python3
"""Tests of scripts/hf-collection.py, the order and the notes of the collection "skycmd" and the
plan that --apply follows, on the repository's manifest, cards and hf/spaces.json, without the
network: python3 -I scripts/test_hf_collection.py"""

import importlib.util
import json
import os
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

spec = importlib.util.spec_from_file_location("hf_collection", os.path.join(HERE, "hf-collection.py"))
hc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hc)

with open(os.path.join(ROOT, "site", "public", "models", "manifest.json")) as f:
    MANIFEST = json.load(f)
with open(os.path.join(ROOT, "hf", "spaces.json")) as f:
    SPACES = json.load(f)
BY_NAME = {e["name"]: e for e in MANIFEST}


def cards():
    out = {}
    for s in hc.hp.model_spaces(SPACES):
        with open(os.path.join(ROOT, "hf", "cards", f"{s['model']}.md")) as f:
            out[s["model"]] = f.read().replace("{{USER}}", "u")
    return out


ITEMS = hc.desired_items("u", SPACES, MANIFEST, cards())


class Order(unittest.TestCase):
    def test_hub_first_then_models_by_verdict_then_their_spaces_then_the_data_sets(self):
        n = len(MANIFEST)
        self.assertEqual(len(ITEMS), 1 + 2 * n + 16)
        self.assertEqual(ITEMS[0][:2], ("u/skycmd", "space"))
        models = ITEMS[1:1 + n]
        spaces = ITEMS[1 + n:1 + 2 * n]
        sets = ITEMS[1 + 2 * n:]
        self.assertEqual({t for _, t, _ in models}, {"model"})
        self.assertEqual({t for _, t, _ in spaces}, {"space"})
        self.assertEqual({t for _, t, _ in sets}, {"dataset"})
        names = [i.split("/", 1)[1] for i, _, _ in models]
        self.assertEqual(sorted(names), sorted(BY_NAME))
        gates = [BY_NAME[m].get("gate") for m in names]
        rank = {"passed": 0, "failed": 1, None: 2}
        self.assertEqual(gates, sorted(gates, key=rank.get))
        # within a group, the order of hf/spaces.json
        order = [s["model"] for s in hc.hp.model_spaces(SPACES)]
        for g in rank:
            group = [m for m in names if BY_NAME[m].get("gate") == g]
            self.assertEqual(group, [m for m in order if m in group])
        self.assertEqual([i for i, _, _ in spaces], [i for i, _, _ in models])
        self.assertEqual([i.split("/", 1)[1] for i, _, _ in sets], hc.hpd.PUBLISHED_SETS)

    def test_every_item_once(self):
        keys = [(i, t) for i, t, _ in ITEMS]
        self.assertEqual(len(keys), len(set(keys)))


class Notes(unittest.TestCase):
    def test_a_model_note_opens_with_the_verdict_of_its_card(self):
        for i, t, note in ITEMS:
            if t != "model":
                continue
            e = BY_NAME[i.split("/", 1)[1]]
            self.assertTrue(note.startswith(hc.hp.verdict_lead(e) + " "), i)
            if e.get("gate_note"):
                self.assertIn(hc.first_sentence(e["gate_note"]).rstrip(".…")[:60], note)
            self.assertLessEqual(len(note), hc.NOTE_MAX)

    def test_a_card_with_another_verdict_is_refused(self):
        c = cards()
        c["s6-nano"] = c["s6-nano"].replace("> **Passed its pre-registered gate.**", "> **Failed.**")
        with self.assertRaises(SystemExit):
            hc.desired_items("u", SPACES, MANIFEST, c)

    def test_space_and_data_set_notes(self):
        by_key = {(i, t): n for i, t, n in ITEMS}
        self.assertEqual(by_key[("u/s6-route", "space")], "Demo: 3.7 MB tool router; passed on fresh slates.")
        self.assertTrue(by_key[("u/s4-micro", "space")].startswith("Demo: Failed its gate"))
        self.assertIn("s6-route (passed)", by_key[("u/skycmd-route", "dataset")])
        self.assertIn("s6-sleep3-nano (failed)", by_key[("u/skycmd-glm-labels", "dataset")])
        self.assertIn("LFM3-1.7B", by_key[("u/skycmd-math", "dataset")])
        for (i, t), n in by_key.items():
            self.assertTrue(n, i)
            self.assertLessEqual(len(n), hc.NOTE_MAX, i)

    def test_first_sentence_and_clip(self):
        self.assertEqual(hc.first_sentence("A (b. c) d. E."), "A (b. c) d.")
        self.assertEqual(hc.first_sentence("No full stop"), "No full stop.")
        long = "word " * 200
        self.assertLessEqual(len(hc.clip(long)), hc.NOTE_MAX)
        self.assertTrue(hc.clip(long).endswith("…"))


def current_from(desired, notes=None, order=None):
    """A collection as the Hub returns it, holding `desired` (in `order`, with `notes` changed)."""
    order = order or list(range(len(desired)))
    out = []
    for pos, k in enumerate(order):
        i, t, n = desired[k]
        out.append({"id": i, "type": t, "object_id": f"o{k}", "note": (notes or {}).get(k, n), "position": pos})
    return out


class Plan(unittest.TestCase):
    def test_nothing_to_do_when_the_collection_is_as_planned(self):
        self.assertEqual(hc.plan(ITEMS, current_from(ITEMS)), [])

    def test_a_new_collection_gets_every_item_in_order(self):
        ops = hc.plan(ITEMS, [])
        self.assertEqual([op for op in ops if op[0] == "add"], [("add", i, t, n) for i, t, n in ITEMS])
        self.assertEqual([op for op in ops if op[0] not in ("add", "position")], [])

    def test_changes_are_planned_and_a_second_run_is_empty(self):
        cur = current_from(ITEMS[:-1], notes={2: "old"}, order=[1, 0] + list(range(2, len(ITEMS) - 1)))
        cur.append({"id": "u/stray", "type": "model", "object_id": "ox", "note": None, "position": 99})
        ops = hc.plan(ITEMS, cur)
        kinds = [op[0] for op in ops]
        self.assertIn(("delete", "ox", "u/stray"), ops)
        self.assertIn(("note", "o2", ITEMS[2][2]), ops)
        self.assertIn(("add", *ITEMS[-1]), ops)
        self.assertIn("position", kinds)
        # after those operations the collection is the plan, and the plan is then empty
        self.assertEqual(hc.plan(ITEMS, current_from(ITEMS)), [])

    def test_order_only(self):
        ops = hc.plan(ITEMS, current_from(ITEMS, order=[1, 0] + list(range(2, len(ITEMS)))))
        self.assertTrue(ops)
        self.assertEqual({op[0] for op in ops}, {"position"})


if __name__ == "__main__":
    unittest.main()
