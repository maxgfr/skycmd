#!/usr/bin/env python3
"""Tests of scripts/s6-route3-gate.py and of the fresh split's guard (scripts/route/common.py) on
synthetic inputs: a pass and a fail for each condition of the gate, the fusion, the readings by
cluster, and the whole gate on an invented fresh folder. python3 -I scripts/test_s6_route3_gate.py"""

import hashlib
import importlib.util
import json
import os
import sys
import tempfile
import types
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "route"))
import common  # noqa: E402
import scorer  # noqa: E402

spec = importlib.util.spec_from_file_location("s6_route3_gate", os.path.join(HERE, "s6-route3-gate.py"))
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)

NONE = common.NONE_OPTION
WASM_OK = {"n": 200, "max_abs_diff": 2e-4, "argmax_agree": 200, "median_cached_us": 41000.0, "median_first_us": 90000.0,
           "bytes_model": 2418912, "bytes_embedding": 1003952}


def slate(uid, gold, k=4, src="f-tools", split="fresh"):
    opts = [f"tool{i}_{uid}: does thing {i}." for i in range(k)] + [NONE]
    t = [0.0] * len(opts)
    t[gold if gold is not None else len(opts) - 1] = 1.0
    return {"src": src, "uid": uid, "opts": opts, "target": t, "kind": "test", "request": uid, "lang": "en", "split": split}


def dump_line(s, pick, conf, gold_p=None):
    n = len(s["opts"])
    g = scorer.argmax(s["target"])
    rest = 1 - conf - (gold_p or 0.0)
    others = [i for i in range(n) if i != pick and (gold_p is None or i != g)]
    p = [0.0] * n
    for i in others:
        p[i] = rest / len(others)
    p[pick] = conf
    if gold_p is not None and pick != g:
        p[g] = gold_p
    return {"src": s["src"], "t": "choice", "k": n, "probs": p, "target": s["target"], "gold": g, "uid": s["uid"]}


def answer(s, right, conf=0.9):
    """A dump line: the gold at `conf` when right, else a wrong option at 0.5 with the gold at 0.3."""
    g = scorer.argmax(s["target"])
    if right:
        return dump_line(s, g, conf)
    return dump_line(s, (g + 1) % len(s["opts"]), 0.5, gold_p=0.3)


class Fusion(unittest.TestCase):
    def test_product_renormalised(self):
        p = gate.fuse_probs([0.5, 0.3, 0.2], [0.2, 0.6, 0.2], 1.0)
        z = 0.5 * 0.2 + 0.3 * 0.6 + 0.2 * 0.2
        for a, b in zip(p, [0.1 / z, 0.18 / z, 0.04 / z]):
            self.assertAlmostEqual(a, b, places=12)

    def test_beta_sharpens_without_changing_the_answer(self):
        a, b = [0.4, 0.35, 0.25], [0.3, 0.3, 0.4]
        lo, hi = gate.fuse_probs(a, b, 0.5), gate.fuse_probs(a, b, 2.0)
        self.assertEqual(scorer.argmax(lo), scorer.argmax(hi))
        self.assertGreater(max(hi), max(lo))

    def test_the_frozen_values(self):
        fit = {"partner": "potion-cos-4bit", "beta": 0.81, "tau_ask3": {"tau": 0.6710473485429961}}
        self.assertEqual(gate.check_fit(fit), (0.81, 0.6710473485429961))
        for bad in ({**fit, "beta": 0.8}, {**fit, "tau_ask3": {"tau": 0.66}}, {**fit, "partner": "potion-cos"}):
            with self.assertRaises(common.Refused):
                gate.check_fit(bad)


class Rules(unittest.TestCase):
    def test_g1(self):
        boot = {"diff": 0.03, "lo": 0.005, "hi": 0.05}
        self.assertTrue(gate.g1(0.89, boot, 0.90, 0.84, 0.67)["pass"])
        # the lower bound at or below 0
        self.assertFalse(gate.g1(0.89, {**boot, "lo": 0.0}, 0.90, 0.84, 0.67)["pass"])
        self.assertFalse(gate.g1(0.89, {**boot, "diff": -0.01, "lo": -0.02}, 0.90, 0.84, 0.67)["pass"])
        # MiniLM - 0.03 is the floor, included
        self.assertTrue(gate.g1(0.87, boot, 0.90, 0.84, 0.67)["minilm_within_0.03"])
        self.assertFalse(gate.g1(0.869, boot, 0.90, 0.84, 0.67)["pass"])
        # above each part, strictly
        self.assertFalse(gate.g1(0.89, boot, 0.90, 0.89, 0.67)["pass"])
        self.assertFalse(gate.g1(0.89, boot, 0.90, 0.84, 0.89)["pass"])
        # a missing input fails
        self.assertFalse(gate.g1(0.89, None, 0.90, 0.84, 0.67)["pass"])
        self.assertFalse(gate.g1(0.89, boot, None, 0.84, 0.67)["pass"])

    def test_g2(self):
        ok = {"f_clinc": {"system": 0.76, "model": 0.40, "partner": 0.77}, "f_metatool": {"system": 0.80, "model": 0.70, "partner": 0.60}}
        self.assertTrue(gate.g2(ok)["pass"])
        self.assertTrue(gate.g2({"f_clinc": {"system": 0.75, "model": 0.40, "partner": 0.77}})["pass"])
        self.assertFalse(gate.g2({**ok, "f_massive": {"system": 0.70, "model": 0.40, "partner": 0.7201}})["pass"])
        self.assertFalse(gate.g2({**ok, "f_massive": {"system": 0.70, "model": None, "partner": 0.60}})["pass"])
        self.assertFalse(gate.g2({})["pass"])

    def test_g3(self):
        self.assertTrue(gate.g3(0.05)["pass"])
        self.assertFalse(gate.g3(0.0501)["pass"])
        self.assertFalse(gate.g3(None)["pass"])

    def test_g4(self):
        self.assertTrue(gate.g4(WASM_OK)["pass"])
        r = gate.g4(None)
        self.assertEqual((r["measured"], r["pass"]), (False, False))
        self.assertTrue(gate.g4({**WASM_OK, "argmax_agree": 199})["pass"])
        for bad in ({"max_abs_diff": 1.1e-3}, {"argmax_agree": 198}, {"median_cached_us": 60001.0}, {"n": 199, "argmax_agree": 199},
                    {"bytes_embedding": 1600000}, {"bytes_model": None}):
            self.assertFalse(gate.g4({**WASM_OK, **bad})["pass"], bad)

    def test_goals(self):
        m = {"accuracy": 0.88, "at_tau_ask": {"accuracy": 0.951, "coverage": 0.81}}
        g = gate.goals(m, 0.87)
        self.assertTrue(all(x["met"] for x in g.values()))
        g = gate.goals({"accuracy": 0.86, "at_tau_ask": {"accuracy": 0.951, "coverage": 0.79}}, 0.84)
        self.assertFalse(any(x["met"] for x in g.values()))


class Clusters(unittest.TestCase):
    def items(self, rights):
        sl = [slate(f"r{i}", i % 3) for i in range(len(rights))]
        return scorer.join([answer(s, r) for s, r in zip(sl, rights)], sl)

    def test_largest_and_ties(self):
        c = {"a": "x", "b": "x", "c": "y", "d": "y", "e": "z", "f": "w", "g": "w", "h": "w"}
        self.assertEqual(gate.largest_clusters(c), ["w", "x", "y"])

    def test_cluster_bootstrap(self):
        a = self.items([True] * 30)
        b = self.items([i % 2 == 0 for i in range(30)])
        cl = {it.uid: f"c{i // 5}" for i, it in enumerate(a)}
        r1 = gate.cluster_bootstrap(a, b, cl, resamples=200)
        r2 = gate.cluster_bootstrap(a, b, cl, resamples=200)
        self.assertEqual(r1, r2)
        self.assertAlmostEqual(r1["diff"], 0.5)
        self.assertEqual(r1["clusters"], 6)
        self.assertLessEqual(r1["lo"], r1["diff"])
        self.assertGreaterEqual(r1["hi"], r1["diff"])
        with self.assertRaises(scorer.ScoreError):
            gate.cluster_bootstrap(a, b, {k: v for k, v in list(cl.items())[1:]})

    def test_proxy_clusters(self):
        s = [slate("a", 1), slate("b", None)]
        p = gate.proxy_clusters(s)
        self.assertEqual(p["a"], "name:tool1_a")
        self.assertEqual(p["b"], "request:b")


def sha(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


class Invented(unittest.TestCase):
    """The whole gate on an invented fresh folder, and the fresh split's guard."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        t = self.tmp.name
        self.saved = {k: os.environ.get(k) for k in ("ROUTE_DATA", "BUILD")}
        os.environ["ROUTE_DATA"] = os.path.join(t, "data")
        os.environ["BUILD"] = os.path.join(t, "data", "build")
        self.saved_paths = (common.FRESH_OPEN, common.FRESH_FILES)
        common.FRESH_OPEN = os.path.join(t, "runs", "FRESH-OPEN")
        common.FRESH_FILES = os.path.join(t, "runs", "test-files.json")
        fresh = os.path.join(t, "data", "build", "fresh")
        self.root = os.path.join(t, "data", "eval", "fresh")
        files = {"f_tools": 60, "f_tools_masked": 60, "f_metatool": 30, "f_clinc": 30, "f_massive": 30}
        self.slates = {}
        for name, n in files.items():
            sl = [slate(f"{name}:{i}", None if i % 5 == 4 else i % 4, src=name.replace("_", "-")) for i in range(n)]
            self.slates[name] = sl
            write_jsonl(os.path.join(fresh, f"{name}.jsonl"), sl)
        write_jsonl(os.path.join(fresh, "f_tools.clusters.jsonl"),
                    [{"uid": s["uid"], "cluster": f"k{i % 7}", "origin": "over the cluster cap"} for i, s in enumerate(self.slates["f_tools"])])
        hashes = {f: sha(os.path.join(fresh, f)) for f in os.listdir(fresh)}
        os.makedirs(os.path.join(t, "runs"))
        with open(common.FRESH_FILES, "w") as f:
            json.dump({"files": hashes}, f)
        p = os.path.join(t, "data", "eval", "fitted", "potion-cos-4bit")
        os.makedirs(p)
        with open(os.path.join(p, "params.json"), "w") as f:
            json.dump({"theta": 0.39611971378326416, "temperature": 0.06710211671167063}, f)
        self.fit = os.path.join(t, "fit.json")
        with open(self.fit, "w") as f:
            json.dump({"partner": "potion-cos-4bit", "beta": 0.81, "tau_ask3": {"tau": 0.6710473485429961}}, f)

    def tearDown(self):
        for k, v in self.saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        common.FRESH_OPEN, common.FRESH_FILES = self.saved_paths
        self.tmp.cleanup()

    def dumps(self, bm25_right):
        """Model wrong on i % 5 == 1, partner on i % 5 == 2 (the pair is wrong on neither), MiniLM on
        i % 10 == 3, the others by `bm25_right`."""
        rules = {"model": lambda i: i % 5 != 1, "model-exit0": lambda i: i % 5 != 1, "potion-cos-4bit": lambda i: i % 5 != 2,
                 "minilm-cos": lambda i: i % 10 != 3, "bm25": bm25_right}
        for name, sl in self.slates.items():
            for system, right in rules.items():
                d = os.path.join(self.root, "model") if system.startswith("model") else os.path.join(self.root, system)
                f = f"{name}.exit0.jsonl" if system == "model-exit0" else f"{name}.jsonl"
                write_jsonl(os.path.join(d, f), [answer(s, right(i)) for i, s in enumerate(sl)])

    def run_gate(self):
        out = os.path.join(self.tmp.name, "gate")
        a = types.SimpleNamespace(split="fresh", root=None, fit=self.fit, wasm=None, native=None, out_dir=out)
        rc = gate.gate(a)
        with open(os.path.join(out, "gate.json")) as f:
            return rc, json.load(f), open(os.path.join(out, "results.md")).read()

    def test_guard(self):
        with self.assertRaises(common.Refused):
            gate.guard("fresh")
        open(common.FRESH_OPEN, "w").close()
        gate.guard("fresh")
        self.assertTrue(common.slate_file("fresh", "f_tools").endswith("f_tools.jsonl"))
        with self.assertRaises(common.Refused):
            common.fresh_file("t_tools")
        # a file that is not the recorded one
        with open(os.path.join(os.environ["BUILD"], "fresh", "f_clinc.jsonl"), "a") as f:
            f.write("\n")
        with self.assertRaises(common.Refused):
            gate.guard("fresh")

    def test_the_gate(self):
        open(common.FRESH_OPEN, "w").close()
        self.dumps(lambda i: i % 3 != 0)
        rc, g, md = self.run_gate()
        v = g["verdict"]
        self.assertEqual(g["results"]["s6-route"]["f_tools"]["accuracy"], 1.0)
        self.assertTrue(v["G1"]["above_bm25"] and v["G1"]["above_model"] and v["G1"]["above_partner"])
        self.assertTrue(v["G2"]["pass"])
        self.assertFalse(v["G4"]["measured"])
        self.assertFalse(v["pass"])  # G4 not measured: never passed
        self.assertEqual(rc, 1)
        self.assertIn("G4 | not measured", md)
        self.assertIn("Disclosure", md)
        r = g["readings"]
        self.assertEqual(r["clusters"]["count"], 7)
        self.assertEqual(len(r["clusters"]["three_largest"]), 3)
        self.assertIn("bm25", r["cluster_bootstrap"])
        self.assertIn("s6-route", r["names_masked"])
        self.assertIn("bm25+potion-cos-4bit", g["results"])
        self.assertIn("at_tau_ask", g["results"]["s6-route"]["f_tools"])
        # the same with the WASM bench's JSON passes
        with open(os.path.join(self.tmp.name, "wasm.json"), "w") as f:
            json.dump(WASM_OK, f)
        out = os.path.join(self.tmp.name, "gate2")
        a = types.SimpleNamespace(split="fresh", root=None, fit=self.fit, wasm=os.path.join(self.tmp.name, "wasm.json"),
                                  native=None, out_dir=out)
        v2 = json.load(open(os.path.join(out, "gate.json")))["verdict"] if gate.gate(a) in (0, 1) else None
        self.assertTrue(v2["G4"]["pass"])
        self.assertEqual(v2["pass"], v2["G1"]["pass"] and v2["G2"]["pass"] and v2["G3"]["pass"])

    def test_bm25_as_good_fails_g1(self):
        open(common.FRESH_OPEN, "w").close()
        self.dumps(lambda i: True)
        _, g, _ = self.run_gate()
        self.assertFalse(g["verdict"]["G1"]["above_bm25"])
        self.assertFalse(g["verdict"]["G1"]["pass"])

    def test_a_missing_dump_fails(self):
        open(common.FRESH_OPEN, "w").close()
        self.dumps(lambda i: i % 3 != 0)
        os.remove(os.path.join(self.root, "potion-cos-4bit", "f_clinc.jsonl"))
        _, g, _ = self.run_gate()
        self.assertFalse(g["verdict"]["G2"]["pass"])

    def test_closed(self):
        self.dumps(lambda i: True)
        with self.assertRaises(common.Refused):
            self.run_gate()


if __name__ == "__main__":
    unittest.main()
