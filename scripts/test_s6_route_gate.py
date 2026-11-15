#!/usr/bin/env python3
"""Tests of scripts/s6-route-gate.py and scripts/route/scorer.py on synthetic inputs, a pass and a
fail for each condition of the gate: python3 -I scripts/test_s6_route_gate.py"""

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "route"))
import common  # noqa: E402
import scorer  # noqa: E402

spec = importlib.util.spec_from_file_location("s6_route_gate", os.path.join(HERE, "s6-route-gate.py"))
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)

NONE = common.NONE_OPTION


def slate(uid, k, gold, names=None, kind="test", follow=None):
    names = names or [f"tool{i}" for i in range(k)]
    opts = [f"{n}: does {n}." for n in names] + [NONE]
    t = [0.0] * len(opts)
    t[gold if gold is not None else len(opts) - 1] = 1.0
    s = {"src": "v-tools", "uid": uid, "opts": opts, "target": t, "kind": kind, "request": uid.split("/")[0], "lang": "en"}
    if follow is not None:
        s["follow"] = follow
    return s


def answer(s, pick, conf=0.9):
    n = len(s["opts"])
    p = [(1 - conf) / (n - 1)] * n
    p[pick] = conf
    return {"src": s["src"], "t": "choice", "k": n, "probs": p, "target": s["target"], "uid": s["uid"]}


def items(pairs):
    return scorer.join([answer(s, k, c) for s, k, c in pairs], [s for s, _, _ in pairs])


def good_metrics():
    return {"accuracy": 0.80, "none_answer_share": 0.2, "largest_name_excess": {"name": "a", "excess": -0.1}, "ece": 0.03}


WASM_OK = {"n": 200, "max_abs_diff": 3e-6, "argmax_agree": 200, "median_cached_us": 800.0, "median_first_us": 4000.0}


class Rules(unittest.TestCase):
    def test_pilot(self):
        r = gate.pilot_rule(0.60, 0.65, 0.70)
        self.assertEqual((r["size"], r["abort"]), ("micro", False))
        r = gate.pilot_rule(0.60, 0.639, 0.70)
        self.assertEqual((r["size"], r["abort"]), ("nano", False))
        # the chosen pilot below potion-cos - 0.15 stops the run
        r = gate.pilot_rule(0.54, 0.55, 0.70)
        self.assertEqual((r["size"], r["abort"]), ("nano", True))
        self.assertTrue(gate.pilot_rule(0.60, None, 0.70)["abort"] is False)
        self.assertTrue(gate.pilot_rule(None, None, 0.70)["abort"])

    def test_g1(self):
        base = {"potion-cos": 0.70, "static-ft": 0.72, "pair-lr": 0.75, "bm25": 0.74}
        self.assertTrue(gate.g1(0.76, base)["pass"])
        self.assertTrue(gate.g1(0.73, {**base, "pair-lr": 0.6, "static-ft": 0.6, "bm25": 0.6})["pass"])
        self.assertFalse(gate.g1(0.729, {**base, "pair-lr": 0.6, "static-ft": 0.6, "bm25": 0.6})["pass"])  # under +0.03
        self.assertFalse(gate.g1(0.75, base)["pass"])  # ties pair-lr
        self.assertFalse(gate.g1(0.76, {**base, "static-ft": 0.77})["pass"])
        self.assertFalse(gate.g1(0.76, {**base, "pair-lr": None})["pass"])  # a missing baseline fails
        # amendment 1: above bm25 too
        self.assertFalse(gate.g1(0.76, {**base, "bm25": 0.76})["pass"])
        self.assertFalse(gate.g1(0.76, {**base, "bm25": 0.80})["pass"])
        self.assertFalse(gate.g1(0.76, {k: v for k, v in base.items() if k != "bm25"})["pass"])

    def test_g2(self):
        ch = {"share": 0.9}
        self.assertTrue(gate.g2(good_metrics(), ch)["pass"])
        for none in (0.09, 0.31):
            self.assertFalse(gate.g2({**good_metrics(), "none_answer_share": none}, ch)["pass"])
        self.assertFalse(gate.g2({**good_metrics(), "largest_name_excess": {"name": "a", "excess": 0.02}}, ch)["pass"])
        self.assertFalse(gate.g2(good_metrics(), {"share": 0.79})["pass"])
        self.assertFalse(gate.g2(good_metrics(), None)["pass"])

    def test_g3(self):
        self.assertTrue(gate.g3({"ece": 0.05})["pass"])
        self.assertFalse(gate.g3({"ece": 0.0501})["pass"])
        self.assertFalse(gate.g3(None)["pass"])

    def test_g4(self):
        self.assertTrue(gate.g4(WASM_OK)["pass"])
        self.assertFalse(gate.g4({**WASM_OK, "median_cached_us": 2001.0})["pass"])
        self.assertFalse(gate.g4({**WASM_OK, "max_abs_diff": 2e-4})["pass"])
        self.assertFalse(gate.g4({**WASM_OK, "argmax_agree": 199})["pass"])
        self.assertFalse(gate.g4(None)["pass"])

    def test_verdict(self):
        base = {"potion-cos": 0.70, "static-ft": 0.72, "pair-lr": 0.75, "bm25": 0.74}
        self.assertTrue(gate.verdict(good_metrics(), base, {"share": 0.9}, WASM_OK)["pass"])
        self.assertFalse(gate.verdict(good_metrics(), base, {"share": 0.9}, None)["pass"])


class Scorer(unittest.TestCase):
    def test_metrics(self):
        s = [slate(f"r{i}", 3, i % 3 if i % 5 else None) for i in range(20)]
        its = items([(x, scorer.argmax(x["target"]), 0.9) for x in s])
        m = scorer.metrics(its, tau=0.5)
        self.assertEqual(m["accuracy"], 1.0)
        self.assertAlmostEqual(m["none_gold_share"], 4 / 20)
        self.assertAlmostEqual(m["none_answer_share"], 4 / 20)
        self.assertAlmostEqual(m["ece"], 0.1)
        self.assertEqual(m["none_f1"], 1.0)
        self.assertEqual(m["at_tau_ask"]["coverage"], 1.0)
        # a constant answer: one name predicted everywhere has a large excess over its gold share
        same = [slate(f"c{i}", 3, i % 3) for i in range(30)]
        m = scorer.metrics(items([(x, 0, 0.6) for x in same]))
        self.assertAlmostEqual(m["top_predicted_name"]["share"], 1.0)
        self.assertGreater(m["largest_name_excess"]["excess"], 0)

    def test_ece_bins(self):
        self.assertEqual(scorer.ece_bin(0.0), 0)
        self.assertEqual(scorer.ece_bin(1 / 15), 0)
        self.assertEqual(scorer.ece_bin(1 / 15 + 1e-12), 1)
        self.assertEqual(scorer.ece_bin(1.0), 14)

    def test_tau_ask(self):
        s = [slate(f"r{i}", 3, 0) for i in range(10)]
        # confident answers right, unsure ones wrong
        its = items([(x, 0 if i < 8 else 1, 0.9 if i < 8 else 0.4) for i, x in enumerate(s)])
        self.assertAlmostEqual(scorer.tau_ask(its), 0.9)
        its = items([(x, 1, 0.9) for x in s])
        self.assertIsNone(scorer.tau_ask(its))

    def test_pairs_and_swaps(self):
        present = slate("q/present", 3, 1)
        removed = slate("q/removed", 3, None, kind="removed")
        p2, r2 = slate("w/present", 3, 1), slate("w/removed", 3, None, kind="removed")
        its = items([(present, 1, 0.9), (removed, 3, 0.9), (p2, 1, 0.9), (r2, 1, 0.9)])
        c = scorer.pair_changes(its)
        self.assertEqual((c["correct_with_gold"], c["changed"]), (2, 1))
        sw = slate("z", 3, 0, kind="swapped", follow=2)
        f = scorer.follow_share(items([(sw, 2, 0.8)]))
        self.assertEqual((f["follow_description"], f["follow_name"]), (1.0, 0.0))

    def test_join_errors(self):
        s = [slate("a", 2, 0), slate("b", 2, 1)]
        with self.assertRaises(scorer.ScoreError):
            scorer.join([answer(s[0], 0)], s)
        with self.assertRaises(scorer.ScoreError):
            scorer.join([answer(s[0], 0), answer(s[0], 0), answer(s[1], 0)], s)

    def test_bootstrap(self):
        s = [slate(f"r{i}", 3, 0) for i in range(50)]
        a = items([(x, 0, 0.9) for x in s])
        b = items([(x, 0 if i % 2 else 1, 0.9) for i, x in enumerate(s)])
        d = scorer.bootstrap_diff(a, b, resamples=200)
        self.assertAlmostEqual(d["diff"], 0.5)
        self.assertLess(d["lo"], 0.5)
        self.assertGreater(d["hi"], 0.5)


class EndToEnd(unittest.TestCase):
    """The gate on synthetic validation dumps, and the guard on the test split."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = self.tmp.name
        self.env = {**os.environ, "ROUTE_DATA": root, "BUILD": os.path.join(root, "build")}
        b = os.path.join(root, "build", "val")
        os.makedirs(b)
        self.slates = [slate(f"r{i}", 7, (i % 7) if i % 5 else None) for i in range(100)]
        common.write_jsonl(os.path.join(b, "v_tools.jsonl"), self.slates)
        pairs = []
        for i in range(20):
            pairs += [slate(f"p{i}/present", 7, 2), slate(f"p{i}/removed", 7, None, kind="removed")]
        common.write_jsonl(os.path.join(b, "v_tools_pairs.jsonl"), pairs)
        self.pairs = pairs
        self.ev = os.path.join(root, "eval")

    def tearDown(self):
        self.tmp.cleanup()

    def dump(self, system, name, rows):
        common.write_jsonl(os.path.join(self.ev, "val", system, f"{name}.jsonl"), rows)

    def run_gate(self, seat="model", wasm=None):
        args = [sys.executable, "-I", os.path.join(HERE, "s6-route-gate.py"), "gate", "--eval", self.ev, "--split", "val",
                "--seat", seat, "--out-dir", os.path.join(self.tmp.name, "out")]
        if wasm:
            p = os.path.join(self.tmp.name, "wasm.json")
            common.write_json(p, wasm)
            args += ["--wasm", p]
        r = subprocess.run(args, env=self.env, capture_output=True, text=True)
        with open(os.path.join(self.tmp.name, "out", "gate.json")) as f:
            return r.returncode, json.load(f)

    def test_pass_and_fail(self):
        # the model: right on 90 of 100, calibrated near 0.9; the baselines: right on 60 to 70
        def sys_rows(right, conf):
            return [answer(s, scorer.argmax(s["target"]) if i < right else (scorer.argmax(s["target"]) + 1) % 8, conf)
                    for i, s in enumerate(self.slates)]
        self.dump("model", "v_tools", sys_rows(90, 0.9))
        self.dump("model", "v_tools_pairs", [answer(s, 2 if s["uid"].endswith("present") else 7, 0.9) for s in self.pairs])
        for name, right in (("potion-cos", 60), ("static-ft", 65), ("pair-lr", 70), ("bm25", 80)):
            self.dump(name, "v_tools", sys_rows(right, 0.6))
        code, out = self.run_gate(wasm=WASM_OK)
        self.assertEqual(code, 0, json.dumps(out["verdict"], indent=1))
        self.assertTrue(out["verdict"]["pass"])
        self.assertIn("potion-cos", out["differences"])
        # the same without a WASM measure fails G4 only
        code, out = self.run_gate()
        self.assertEqual(code, 1)
        self.assertEqual([k for k in ("G1", "G2", "G3", "G4") if not out["verdict"][k]["pass"]], ["G4"])
        # a baseline in the model's seat ties itself: G1 fails
        code, out = self.run_gate(seat="pair-lr", wasm=WASM_OK)
        self.assertFalse(out["verdict"]["G1"]["pass"])
        # amendment 1: a model at or below bm25 fails G1
        self.dump("bm25", "v_tools", sys_rows(90, 0.6))
        code, out = self.run_gate(wasm=WASM_OK)
        self.assertEqual(code, 1)
        self.assertFalse(out["verdict"]["G1"]["pass"])
        self.assertIn("bm25", out["differences"])

    def test_the_test_split_is_closed(self):
        os.makedirs(os.path.join(self.tmp.name, "build", "test"))
        common.write_jsonl(os.path.join(self.tmp.name, "build", "test", "t_tools.jsonl"), self.slates)
        r = subprocess.run([sys.executable, "-I", "-c",
                            f"import sys; sys.path.insert(0, {os.path.join(HERE, 'route')!r}); import common; common.slate_file('test', 't_tools', None)"],
                           env=self.env, capture_output=True, text=True)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("Refused", r.stderr)
        if not os.path.isfile(common.TEST_OPEN):
            with self.assertRaises(common.Refused):
                common.test_is_open(None)
        with self.assertRaises(common.Refused):
            common.slate_file("val", "t_tools")


if __name__ == "__main__":
    unittest.main()
