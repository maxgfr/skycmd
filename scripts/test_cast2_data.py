#!/usr/bin/env python3
"""Tests of the leakage check of scripts/cast2-data.py (docs/s6-cast2-prereg.md §3.3, amendment 2):
python3 -I scripts/test_cast2_data.py (needs numpy)."""

import importlib.util
import os
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("cast2_data", os.path.join(HERE, "cast2-data.py"))
cd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cd)


def bench(*series):
    return [("gift_eval", f"b{i}", np.asarray(s, np.float32)) for i, s in enumerate(series)]


class Leak(unittest.TestCase):
    def test_informative_runs(self):
        rng = np.random.default_rng(1)
        ok = rng.normal(size=32).astype(np.float32)
        zeros = np.zeros(32, np.float32)
        zeros[7] = 0.2
        few = np.tile(np.arange(7, dtype=np.float32), 5)[:32]  # 7 distinct values
        half = np.concatenate([np.zeros(16), rng.normal(size=16)]).astype(np.float32)
        over = np.concatenate([np.zeros(17), rng.normal(size=15)]).astype(np.float32)
        got = cd.informative(np.stack([ok, zeros, few, half, over]))
        self.assertEqual(got.tolist(), [True, False, False, True, False])

    def test_a_copied_series_is_caught(self):
        rng = np.random.default_rng(2)
        b = rng.normal(size=400).astype(np.float32)
        copy = np.concatenate([rng.normal(size=50), b[100:200]]).astype(np.float32)
        noise = rng.normal(size=300).astype(np.float32)
        r = cd.check({"src": [("copy", copy), ("noise", noise)]}, bench(b))
        s = r["sources"]["src"]
        self.assertEqual(s["matching_runs"], 100 - 31)
        self.assertEqual(s["dropped_series"], ["copy"])
        # one series of two dropped: more than 5%, the source goes whole
        self.assertFalse(s["kept"])
        self.assertEqual(s["series_kept"], 0)

    def test_a_zeros_heavy_run_does_not_count(self):
        rng = np.random.default_rng(3)
        rain = np.zeros(300, np.float32)
        rain[[40, 41, 150]] = [0.2, 0.5, 0.3]
        b = np.zeros(500, np.float32)
        b[100:300] = rain[:200]  # the same 31-zeros runs occur in the benchmark
        others = [(f"s{i}", rng.normal(size=200).astype(np.float32)) for i in range(30)]
        r = cd.check({"src": [("rain", rain), *others]}, bench(b))
        s = r["sources"]["src"]
        self.assertEqual(s["matching_runs"], 0)
        self.assertEqual(s["series_dropped"], 0)
        self.assertTrue(s["kept"])
        # amendment 1's rule would have matched them
        h1 = cd.run_hashes(rain, only_informative=False)
        self.assertTrue(cd.lookup(np.unique(cd.run_hashes(b, only_informative=False)), h1).any())

    def test_the_source_survives_below_five_percent(self):
        rng = np.random.default_rng(4)
        b = rng.normal(size=300).astype(np.float32)
        rows = [(f"s{i}", rng.normal(size=200).astype(np.float32)) for i in range(40)]
        rows[0] = ("copy", b[:120].copy())
        r = cd.check({"src": rows}, bench(b))
        s = r["sources"]["src"]
        self.assertEqual(s["dropped_series"], ["copy"])  # 1 of 40: 2.5%
        self.assertTrue(s["kept"])
        self.assertEqual(s["series_kept"], 39)
        rows[1] = ("copy2", b[150:290].copy())
        rows[2] = ("copy3", b[10:80].copy())
        s = cd.check({"src": rows}, bench(b))["sources"]["src"]
        self.assertEqual(s["series_dropped"], 3)  # 7.5%
        self.assertFalse(s["kept"])


if __name__ == "__main__":
    unittest.main()
