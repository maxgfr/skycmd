#!/usr/bin/env python3
"""Tests of the pieces of scripts/route-baselines.py that do not need a model: the fitted
threshold and temperature, the 4-bit packing, the pair features. Run in the pinned environment:
<data>/envs/route/bin/python -I scripts/route/test_baselines.py"""

import importlib.util
import os
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("route_baselines", os.path.join(os.path.dirname(HERE), "route-baselines.py"))
rb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rb)


class FakeTokenizer:
    def __init__(self, vocab):
        self.vocab = vocab

    def get_vocab_size(self):
        return len(self.vocab)


class FakeStatic:
    """A static model of 4 tokens: each word of a text is one token."""

    def __init__(self, emb, weights=None):
        self.embedding = np.asarray(emb, np.float32)
        self.weights = weights
        self.token_mapping = None
        self.normalize = True
        self.tokenizer = FakeTokenizer(["a", "b", "c", "d"])

    def tokenize(self, texts):
        return [["abcd".index(w) for w in t.split()] for t in texts]


class Baselines(unittest.TestCase):
    def test_theta(self):
        # best tool scores 0.9, 0.8 (gold a tool) and 0.3, 0.2 (gold none)
        s = [np.array([0.9, 0.1, 0]), np.array([0.8, 0.2, 0]), np.array([0.3, 0.1, 0]), np.array([0.2, 0.1, 0])]
        g = [0, 0, 2, 2]
        t = rb.fit_theta(s, g)
        self.assertTrue(0.3 < t <= 0.8)
        # of equal accuracies the lowest theta: here every theta in (0.3, 0.8] is perfect
        self.assertAlmostEqual(t, 0.8)
        # a none that never helps: theta below every score
        self.assertLessEqual(rb.fit_theta(s[:2], [0, 0]), 0.8)

    def test_temperature(self):
        rng = np.random.default_rng(0)
        logits, golds = [], []
        for _ in range(400):
            z = rng.normal(size=4)
            p = np.exp(z / 0.5) / np.exp(z / 0.5).sum()
            golds.append(int(rng.choice(4, p=p)))
            logits.append(z)
        self.assertAlmostEqual(rb.fit_temperature(logits, golds), 0.5, delta=0.15)

    def test_packing(self):
        e = np.array([[1.0, -0.5], [0.2, 0.1], [0.0, 0.0], [3.0, 3.0]])
        st = FakeStatic(e)
        full = rb.Packed(st, bits=0)
        np.testing.assert_allclose(full.encode(["a b"])[0], (e[0] + e[1]) / np.linalg.norm(e[0] + e[1]), atol=1e-6)
        p4 = rb.Packed(st)
        self.assertEqual(p4.bytes(), 4 * (1 + 2))  # two dims at 4 bits: one byte, and a float16 scale
        self.assertLess(np.abs(p4.vec - e).max(), 1.0 / 7 + 1e-6)
        cut = rb.Packed(st, keep=[0, 3])
        self.assertEqual(cut.bytes(), 2 * 3)
        # a text of dropped tokens only embeds to zero
        self.assertEqual(float(np.abs(cut.encode(["b c"])).sum()), 0.0)

    def test_pair_features(self):
        f = rb.PairFeatures(1 << 12)
        recs = [{"state": "User: turn on the lights", "opts": ["lights_on: Turns lights on.", "weather: The weather.", rb.common.NONE_OPTION],
                 "target": [1.0, 0.0, 0.0]}]
        x, slate = f.matrix(recs)
        self.assertEqual(x.shape, (3, 1 << 12))
        self.assertEqual(list(slate), [0, 0, 0])
        np.testing.assert_allclose(np.asarray(x.multiply(x).sum(axis=1)).ravel(), 1.0, rtol=1e-5)
        self.assertEqual(list(rb.labels_of(recs)), [1, 0, 0])
        # the same pair gives the same row
        y, _ = f.matrix(recs)
        self.assertEqual((x != y).nnz, 0)

    def test_name_prior(self):
        r = {"opts": ["a: x", "b: y", rb.common.NONE_OPTION]}
        self.assertEqual(list(rb.name_prior_probs([r], {"b": 3, "a": 1})[0]), [0, 1, 0])
        self.assertEqual(list(rb.name_prior_probs([r], {})[0]), [0, 0, 1])


if __name__ == "__main__":
    unittest.main()
