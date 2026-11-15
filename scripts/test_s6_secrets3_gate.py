#!/usr/bin/env python3
"""Tests of scripts/s6-secrets3-gate.py on synthetic scores: python3 -I scripts/test_s6_secrets3_gate.py"""

import importlib.util
import os
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))


def module(name, file):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, file))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


gate = module("s6_secrets3_gate", "s6-secrets3-gate.py")
v2 = module("test_s6_secrets2_gate", "test_s6_secrets2_gate.py")


def passing():
    """v2's passing inputs, with the unions: union 0.75 / 0.72, union-lr 0.70 / 0.69."""
    scores, issues, clean, _ = v2.passing()
    scores["thresholds"] = {"union": 0.35, "union-lr": 0.15, "union-tau-v2": 0.26}
    scores["systems"]["union"] = v2.system(0.75, 0.72, unit="line")
    scores["systems"]["union-lr"] = v2.system(0.70, 0.69, unit="line")
    scores["systems"]["union-tau-v2"] = v2.system(0.74, 0.71, unit="line")
    scores["systems"]["model"] = v2.system(0.60, 0.70, unit="candidate", ece=0.03)
    scores["added_over_the_model_alone"] = {u: {n: {"threshold": 0.3, "added": {"T": 1, "X": 0, "F": 1}, "dropped": {"T": 0, "X": 0, "F": 0},
                                                    "default_flags_line": 5, "units": 50} for n in ("union", "union-tau-v2")} for u in ("gated", "full")}
    fb = {k: {"positive": 1, "negative": 1} for k in ("default only", "verifier only", "both", "neither")}
    for n, f in (("union", 0.77), ("union-lr", 0.6), ("union-tau-v2", 0.76)):
        issues["systems"][n] = {"candidate": {"tp": 1, "fp": 1, "fn": 1, "tn": 1, "precision": f, "recall": f, "f1": f, "false_alarm_rate": 0.01}, "flagged_by": fb}
    u = {"files_with_an_accepted_flag": 40, "files": 8825, "share": 40 / 8825, "files_default_only": 1, "files_verifier_only": 30, "files_both": 9,
         "default_flags": 57, "verifier_accepted_flags": 90, "threshold": 0.35}
    for n in ("union", "union-lr", "union-tau-v2"):
        clean["readings"]["flags"]["systems"][n] = dict(u)
    del clean["readings"]["flags-with-path"]
    g2 = {"split": "test", "model": "model", "gates": {"G5": {"holds": True, "why": [], "ece": 0.0073}, "G6": {"holds": True, "why": [], "median_us": 1606.0}}}
    tau = {"tau3": 0.35, "tau_union_lr": 0.15, "tau_v2": 0.26, "validation": {"union": {"gated": {"all": {"f1": 0.9}}}}}
    return scores, issues, clean, g2, tau


def run(scores, issues, clean, g2, tau):
    g = gate.gates(scores, issues, clean, g2)
    return g, gate.results(scores, issues, clean, g2, tau, {"commit": "a" * 40, "committed_at": "2026-11-15T09:00:00+01:00"}, "read", g)


class Gate(unittest.TestCase):
    def test_everything_holds(self):
        g, (md, ok, verdict) = run(*passing())
        self.assertTrue(ok, [k for k, v in g.items() if not v["holds"]])
        self.assertEqual(verdict, "PASS on a second look, system redefined after the first reading")
        # every section carries the two sentences
        self.assertEqual(md.count(gate.UUID_SENTENCE), md.count("\n## "))
        self.assertEqual(md.count(gate.SECOND_LOOK), md.count("\n## ") + 1)

    def test_g1_reads_the_union_twice(self):
        s = passing()
        v2.set_f1(s[0], "union", 0.64)  # best scanner 0.50 + 0.15
        g, (md, ok, verdict) = run(*s)
        self.assertFalse(g["G1"]["holds"])
        self.assertTrue(g["G1"]["parts"]["without_largest"]["holds"])
        self.assertIn("no third reading", verdict)
        # the model alone does not enter G1
        s = passing()
        v2.set_f1(s[0], "model", 0.1)
        self.assertTrue(run(*s)[0]["G1"]["holds"])

    def test_g3_is_against_the_union_with_the_regression(self):
        s = passing()
        v2.set_f1(s[0], "union-lr", 0.72, "without_largest")
        g = run(*s)[0]
        self.assertFalse(g["G3"]["holds"])
        self.assertEqual(g["G3"]["parts"]["all"]["reference"], "union-lr")
        s = passing()
        v2.set_f1(s[0], "lr", 0.99)  # the regression alone is not the bar
        self.assertTrue(run(*s)[0]["G3"]["holds"])

    def test_g4_g7_and_the_copied_gates(self):
        s = passing()
        s[1]["systems"]["union"]["candidate"]["f1"] = 0.75  # Gitleaks 0.68 + 0.08
        g, (md, ok, verdict) = run(*s)
        self.assertFalse(g["G4"]["holds"])
        self.assertIn("no third reading", verdict)
        s = passing()
        s[2]["readings"]["flags"]["systems"]["union"]["share"] = 0.011
        g, (md, ok, verdict) = run(*s)
        self.assertFalse(g["G7"]["holds"])
        self.assertNotIn("no third reading", verdict)
        s = passing()
        s[3]["gates"]["G5"] = {"holds": False, "why": ["x"], "ece": 0.06}
        g = run(*s)[0]
        self.assertFalse(g["G5"]["holds"])
        self.assertTrue(g["G5"]["copied_from_v2"])

    def test_rehearsal(self):
        s = passing()
        s[0]["split"] = "validation"
        md, ok, verdict = run(*s)[1]
        self.assertTrue(verdict.startswith("REHEARSAL on validation"))
        self.assertIn("**Rehearsal.**", md)


if __name__ == "__main__":
    unittest.main()
