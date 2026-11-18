#!/usr/bin/env python3
"""Tests of `scripts/route-embed.py --requests` (s6-route v5, docs/s6-route5-prereg.md 4.1) with an
invented static model: the rows are potion-cos's request vectors (scripts/route-baselines.py, the
same model call on the same text), L2-normalised; the guard refuses a test or fresh request before
any model is called; v2's folder is never written. No request of the data folder is read or
embedded. Run in the pinned environment (numpy):
<data>/envs/route/bin/python -I scripts/test_route_embed.py"""
import hashlib
import importlib.util
import json
import os
import sys
import tempfile
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "route"))
import common  # noqa: E402


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, path))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


emb = load("route_embed", "route-embed.py")
rb = load("route_baselines", "route-baselines.py")


class FakeStatic:
    """A static model: each text a deterministic vector of its words, not normalised."""

    def __init__(self):
        self.calls = []

    def encode(self, texts, show_progress_bar=False):
        self.calls.append(list(texts))
        out = []
        for t in texts:
            v = np.zeros(8, np.float32)
            for w in t.split():
                h = hashlib.sha256(w.encode()).digest()
                v += np.frombuffer(h[:32], dtype=np.uint8)[:8].astype(np.float32) - 120.0
            out.append(v * 3.0)
        return np.stack(out)


def plan_rows():
    return [
        {"id": "glaive:1", "group": "tools", "source": "glaive", "split": "train", "user": "  turn on the lights ",
         "gold": "t1", "natural": [], "hard": [], "reserve": []},
        {"id": "when2call:n3", "group": "when2call", "source": "when2call", "split": "train", "user": "what time is it",
         "gold": None, "natural": [], "hard": [], "reserve": []},
        {"id": "massive:7", "group": "intents", "source": "massive", "split": "val", "user": "wake me up at seven",
         "gold": "t2", "natural": [], "hard": [], "reserve": []},
    ]


def write_plan(d, rows):
    os.makedirs(os.path.join(d, "stage"), exist_ok=True)
    with open(os.path.join(d, "stage", "plan.jsonl"), "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)


class Requests(unittest.TestCase):
    def test_rows_are_potion_cos_request_vectors(self):
        with tempfile.TemporaryDirectory() as d:
            v2, b5 = os.path.join(d, "build"), os.path.join(d, "build5")
            rows = plan_rows()
            write_plan(v2, rows)
            meta = emb.embed_requests(v2, b5, model=FakeStatic())
            e = np.load(os.path.join(b5, "emb", emb.REQUESTS_NPY))
            self.assertEqual(e.shape, (3, 8))
            self.assertEqual(meta["by split"], {"train": 2, "val": 1})
            ids = "\n".join(r["id"] for r in rows)
            self.assertEqual(meta["ids_sha256"], hashlib.sha256(ids.encode()).hexdigest())
            self.assertEqual(meta["source_sha256"], common.sha256_file(os.path.join(v2, "stage", "plan.jsonl")))
            # potion-cos embeds the request of a record: its state without `User: `
            fake = FakeStatic()
            rb.potion_static = lambda: fake
            enc = rb.System("potion-cos", {"theta": 0.0}).encoder()
            for r, row in zip(rows, e):
                rec = {"state": "User: " + r["user"].strip()}
                q = enc([rb.request_of(rec)])[0]
                np.testing.assert_allclose(row, q / np.linalg.norm(q), rtol=1e-6, atol=1e-6)
                self.assertAlmostEqual(float(np.linalg.norm(row)), 1.0, places=5)
            # the tools' meta.json is not touched
            self.assertFalse(os.path.exists(os.path.join(b5, "emb", "meta.json")))

    def test_a_test_or_fresh_request_is_refused_before_any_model_call(self):
        for bad in ({"split": "test"}, {"split": "fresh"}, {"group": "test"}, {"id": "toucan:abc"},
                    {"split": None}):
            rows = plan_rows()
            rows[1] = dict(rows[1], **bad)
            with tempfile.TemporaryDirectory() as d:
                v2, b5 = os.path.join(d, "build"), os.path.join(d, "build5")
                write_plan(v2, rows)
                fake = FakeStatic()
                with self.assertRaises(common.Refused):
                    emb.embed_requests(v2, b5, model=fake)
                self.assertEqual(fake.calls, [], bad)
                self.assertFalse(os.path.exists(os.path.join(b5, "emb", emb.REQUESTS_NPY)))

    def test_never_into_v2s_folder_and_no_repeat(self):
        with tempfile.TemporaryDirectory() as d:
            v2 = os.path.join(d, "build")
            write_plan(v2, plan_rows())
            with self.assertRaises(common.Refused):
                emb.embed_requests(v2, v2, model=FakeStatic())
            rows = plan_rows()
            rows[2]["id"] = rows[0]["id"]
            with self.assertRaises(common.Refused):
                emb.requests_to_embed(rows)


if __name__ == "__main__":
    unittest.main()
