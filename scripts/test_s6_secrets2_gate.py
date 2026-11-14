#!/usr/bin/env python3
"""Tests of scripts/s6-secrets2-gate.py on synthetic scores: python3 -I scripts/test_s6_secrets2_gate.py"""

import copy
import importlib.util
import json
import os
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("s6_secrets2_gate", os.path.join(HERE, "s6-secrets2-gate.py"))
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)

CATS = ["Password", "Token", "Key", "Secret", "API", "Auth", "other"]


def m(f1):
    return {"tp": 10, "fp": 2, "fn": 3, "tn": 50, "precision": f1, "recall": f1, "f1": f1, "false_alarm_rate": 0.02}


def system(f1, f1_without=None, full=None, unit="span", ece=None, version="1.0"):
    """A system whose gated line F1 is `f1` on every repository and `f1_without` without the
    largest one; its full-universe numbers are `full` (default: 0.1 lower)."""
    f1_without = f1 if f1_without is None else f1_without
    full = f1 - 0.1 if full is None else full
    line = lambda a, b: {"all": m(a), "without_largest": m(b), "macro_f1": {"value": a, "repositories": 6},
                         "by_category": {c: m(a) for c in CATS}, "by_scope": {s: m(a) for s in ("test", "src", "other")}}
    s = {"unit": unit, "scored": ece is not None, "threshold": 0.5 if ece is not None else None, "version": version, "cpu_seconds": 12.0,
         "line": {"gated": line(f1, f1_without), "full": line(full, full)},
         "unlabelled": {"flags": 5, "lines": 5, "files": 100, "per_1000_files": 50.0}}
    if unit != "line":
        s["labels"] = {u: {"T": 10, "X": 1, "F": 5, "D": 1, "U": 5, "E": 0} for u in ("gated", "full")}
        s["span"] = {u: {"t_rows": 20, "covered": 10, "recall": 0.5, "flags_on_labelled_lines": 17, "flags_on_t": 10, "precision": 0.6, "f1": 0.55} for u in ("gated", "full")}
    if unit == "candidate":
        c = {"n": 16, "positives": 10, "tp": 8, "fp": 1, "fn": 2, "tn": 5, "precision": 0.89, "recall": 0.8, "f1": 0.84, "false_alarm_rate": 0.17}
        if ece is not None:
            c.update({"pr_auc": 0.9, "recall_at_1pct_false_alarms": 0.5, "ece": ece, "ece_positive": ece})
        s["candidate"] = {"gated": dict(c), "full": dict(c, ece=ece + 0.02 if ece is not None else None)}
    return s


def passing():
    """Inputs on which every gate holds: the best scanner at 0.50, cbx at 0.80, the LR at 0.70."""
    scores = {"split": "test", "repositories": 221, "files": 7372, "t_rows": {"gated": 9000, "full": 11603},
              "largest_repository": {"repo_id": "2ba83c6a", "rehearsal_stand_in": False, "t_rows": {"gated": 4000, "full": 5165}},
              "systems": {"model": system(0.75, 0.72, unit="candidate", ece=0.03), "generator": system(0.40, unit="candidate"),
                          "gitleaks": system(0.44), "betterleaks": system(0.45), "secretgate-default": system(0.50, 0.48),
                          "detect-secrets": system(0.46, unit="line"), "cbx-17m": system(0.80, 0.78), "lr": system(0.70, 0.69, unit="candidate", ece=0.08)},
              "bootstrap": {"resamples": 1000, "seed": 1234, "level": 0.95, "reference": "model",
                            "f1_difference": {n: {u: {p: {"value": 0.1, "lo": 0.02, "hi": 0.2} for p in ("all", "without_largest")} for u in ("gated", "full")}
                                              for n in ("secretgate-default", "cbx-17m", "lr")},
                            "ece": {"model": {u: {"value": 0.03, "lo": 0.02, "hi": 0.05} for u in ("gated", "full")}}}}
    cand = lambda f1, ece=None: {"candidate": {"tp": 1, "fp": 1, "fn": 1, "tn": 1, "precision": f1, "recall": f1, "f1": f1, "false_alarm_rate": 0.01, "ece": ece},
                                 "end_to_end": {"flags": 10, "precision": 0.3, "f1": 0.4, "precision_over_flags_on_a_candidate": 0.8, "f1_over_flags_on_a_candidate": 0.7}}
    issues = {"split": "test", "set": {"rows": 8123, "positives": 888, "distinct_texts": 8000, "several_occurrences": 700, "not_found": 0,
                                       "obfuscation": {"positive_values": 880, "left_as_published": {"unchanged": 0, "error": 30, "timeout": 1, "length": 0}}},
              "systems": {"model": cand(0.78, 0.04), "gitleaks": cand(0.68)},
              "obfuscation_sensitivity": {"model": {"f1_published": 0.78, "f1_obfuscated": 0.74, "difference": 0.04}}}
    share = lambda n: {"files_with_an_accepted_flag": n, "files": 8825, "share": n / 8825, "accepted_flags": 2 * n}
    clean = {"corpus": {"files": 8825, "bytes": 1, "list_sha256": "b" * 64}, "gate_reading": "flags",
             "readings": {"flags": {"how": "as the census", "systems": {"model": share(60), "generator": share(2161), "secretgate-default": share(11)}},
                          "flags-with-path": {"how": "with the path", "systems": {"model": share(5), "generator": share(187)}}}}
    latency = {"model": "m.gguf", "bytes": 250272, "n_params": 445345, "tau": 0.4, "candidates": 30000, "median_us_full_depth": 1800.0,
               "median_us_first_exit": 600.0, "early_exit_rate_at_stored_thresholds": 0.3, "exits": [{"exit": 1, "n": 10, "accuracy": 0.9, "ece": 0.05}]}
    return scores, issues, clean, latency


def run(scores, issues, clean, latency, model="model"):
    g, goal = gate.gates(scores, issues, clean, latency, model)
    md, ok, verdict = gate.results(scores, issues, clean, latency, model, g, goal)
    return g, goal, md, ok, verdict


def set_f1(scores, name, f1, part="all", universe="gated"):
    scores["systems"][name]["line"][universe][part]["f1"] = f1


class Gate(unittest.TestCase):
    def test_everything_holds(self):
        g, goal, md, ok, verdict = run(*passing())
        self.assertTrue(ok, md)
        self.assertTrue(all(x["holds"] for x in g.values()))
        self.assertEqual(verdict, "PASS: publish s6-secrets-verify-pico")
        # the best rule scanner is chosen in each part: secretgate-default both times here
        self.assertEqual(g["G1"]["parts"]["all"]["reference"], "secretgate-default")
        self.assertFalse(goal)  # 0.75 against cbx 0.80 - 0.05 = 0.75 holds; 0.72 against 0.78 - 0.05 does not
        self.assertIn("G1, every repository: holds", md)
        self.assertIn("G3, without the largest repository: holds", md)
        self.assertIn("announced goal, G2 at -0.05: not met", md)
        # the sentence on the UUID choice stands with every table of CredData scores
        self.assertEqual(md.count("made after reading the training and validation rows and before any test file was read"), 3)
        self.assertIn("in-sample", run(*self.with_system(passing(), "credsweeper-ml", system(0.9)))[2])

    def with_system(self, inputs, name, s):
        inputs[0]["systems"][name] = s
        return inputs

    def test_g1(self):
        s, i, c, l = passing()
        # exactly the margin holds; a hair under does not
        set_f1(s, "model", 0.65)
        set_f1(s, "cbx-17m", 0.70)
        set_f1(s, "lr", 0.60)
        self.assertTrue(run(s, i, c, l)[0]["G1"]["holds"])
        set_f1(s, "model", 0.6499)
        g = run(s, i, c, l)[0]
        self.assertFalse(g["G1"]["holds"])
        self.assertFalse(g["G1"]["parts"]["all"]["holds"])
        self.assertTrue(g["G1"]["parts"]["without_largest"]["holds"])
        # the bar is the best of the three scanners: Betterleaks moves it
        s, i, c, l = passing()
        set_f1(s, "betterleaks", 0.61)
        g, _, md, ok, verdict = run(s, i, c, l)
        self.assertEqual((g["G1"]["parts"]["all"]["reference"], g["G1"]["holds"], ok), ("betterleaks", False, False))
        self.assertEqual(verdict, "FAIL: not published")
        # holding on every repository is not enough: without the largest it must hold too
        s, i, c, l = passing()
        set_f1(s, "model", 0.60, "without_largest")
        g = run(s, i, c, l)[0]
        self.assertTrue(g["G1"]["parts"]["all"]["holds"])
        self.assertFalse(g["G1"]["holds"])
        # the full universe is reported, never read
        s, i, c, l = passing()
        for part in ("all", "without_largest"):
            set_f1(s, "model", 0.0, part, "full")
        self.assertTrue(run(s, i, c, l)[3])
        # a scanner that did not run: no bar, no pass
        s, i, c, l = passing()
        del s["systems"]["betterleaks"]
        g, _, md, ok, _ = run(s, i, c, l)
        self.assertFalse(g["G1"]["holds"])
        self.assertIn("G1: fails", md)

    def test_g2(self):
        s, i, c, l = passing()
        set_f1(s, "cbx-17m", 0.85)
        self.assertTrue(run(s, i, c, l)[0]["G2"]["holds"])  # 0.75 = 0.85 - 0.10
        set_f1(s, "cbx-17m", 0.8501)
        self.assertFalse(run(s, i, c, l)[0]["G2"]["holds"])
        s, i, c, l = passing()
        set_f1(s, "cbx-17m", 0.83, "without_largest")
        g = run(s, i, c, l)[0]
        self.assertTrue(g["G2"]["parts"]["all"]["holds"])
        self.assertFalse(g["G2"]["holds"])
        s, i, c, l = passing()
        set_f1(s, "model", 0.77, "without_largest")
        self.assertTrue(run(s, i, c, l)[1])  # within 0.05 of cbx in both parts: the announced goal
        del s["systems"]["cbx-17m"]
        self.assertFalse(run(s, i, c, l)[0]["G2"]["holds"])

    def test_g3_is_strict(self):
        s, i, c, l = passing()
        set_f1(s, "lr", 0.75)
        g, _, md, ok, _ = run(s, i, c, l)
        self.assertFalse(g["G3"]["holds"])
        self.assertIn("G3, every repository: fails", md)
        set_f1(s, "lr", 0.7499)
        self.assertTrue(run(s, i, c, l)[0]["G3"]["holds"])
        set_f1(s, "lr", 0.73, "without_largest")
        self.assertFalse(run(s, i, c, l)[0]["G3"]["holds"])

    def test_g4(self):
        s, i, c, l = passing()
        i["systems"]["model"]["candidate"]["f1"] = 0.76
        self.assertTrue(run(s, i, c, l)[0]["G4"]["holds"])  # 0.68 + 0.08
        i["systems"]["model"]["candidate"]["f1"] = 0.7599
        g, _, md, ok, _ = run(s, i, c, l)
        self.assertFalse(g["G4"]["holds"])
        self.assertFalse(ok)
        self.assertIn("G4: fails", md)
        del i["systems"]["gitleaks"]
        self.assertFalse(run(s, i, c, l)[0]["G4"]["holds"])
        self.assertFalse(run(s, None, c, l)[0]["G4"]["holds"])

    def test_g5(self):
        s, i, c, l = passing()
        s["systems"]["model"]["candidate"]["gated"]["ece"] = 0.05
        self.assertTrue(run(s, i, c, l)[0]["G5"]["holds"])
        s["systems"]["model"]["candidate"]["gated"]["ece"] = 0.0501
        self.assertFalse(run(s, i, c, l)[0]["G5"]["holds"])
        # the gated universe is read: a worse ECE on the full one is reported only
        s["systems"]["model"]["candidate"]["gated"]["ece"] = 0.04
        s["systems"]["model"]["candidate"]["full"]["ece"] = 0.2
        g, _, md, _, _ = run(s, i, c, l)
        self.assertTrue(g["G5"]["holds"])
        self.assertIn("full universe 0.2000", md)
        # a system without scores has no ECE
        s["systems"]["model"] = system(0.75, 0.72, unit="candidate")
        self.assertFalse(run(s, i, c, l)[0]["G5"]["holds"])

    def test_g6(self):
        s, i, c, l = passing()
        l["median_us_full_depth"] = 3000.0
        self.assertTrue(run(s, i, c, l)[0]["G6"]["holds"])
        l["median_us_full_depth"] = 3000.5
        self.assertFalse(run(s, i, c, l)[0]["G6"]["holds"])
        g, _, md, ok, _ = run(s, i, c, None)
        self.assertFalse(g["G6"]["holds"])
        self.assertIn("G6: fails (no native latency)", md)

    def test_g7(self):
        s, i, c, l = passing()
        share = lambda n: {"files_with_an_accepted_flag": n, "files": 8825, "share": n / 8825, "accepted_flags": n}
        c["readings"]["flags"]["systems"]["model"] = share(88)  # 0.997%
        self.assertTrue(run(s, i, c, l)[0]["G7"]["holds"])
        c["readings"]["flags"]["systems"]["model"] = share(89)  # 1.008%
        g, _, md, ok, _ = run(s, i, c, l)
        self.assertFalse(g["G7"]["holds"])
        self.assertFalse(ok)
        # the gate reads the corpus as the census read it, not the reading with the path
        self.assertEqual(c["readings"]["flags-with-path"]["systems"]["model"]["files_with_an_accepted_flag"], 5)
        del c["readings"]["flags"]["systems"]["model"]
        self.assertFalse(run(s, i, c, l)[0]["G7"]["holds"])

    def test_no_model_and_rehearsal(self):
        s, i, c, l = passing()
        del s["systems"]["model"]
        g, goal, md, ok, verdict = run(s, i, c, l)
        self.assertFalse(ok)
        self.assertFalse(any(x["holds"] for n, x in g.items() if n in ("G1", "G2", "G3", "G5")))
        self.assertIn("No system named `model` was scored", md)
        self.assertIsNone(goal)
        # another system in the model's seat, on validation: a rehearsal, never a verdict
        s, i, c, l = passing()
        s["split"] = "validation"
        s["largest_repository"]["rehearsal_stand_in"] = True
        g, goal, md, ok, verdict = run(s, i, c, l, model="lr")
        self.assertIn("REHEARSAL on validation", verdict)
        self.assertIn("**Rehearsal.**", md)
        self.assertIn("`lr` sits in the model's seat", md)
        self.assertFalse(g["G3"]["holds"])  # the LR is not better than itself

    def test_reported_extras(self):
        s, i, c, l = passing()
        g, goal = gate.gates(s, i, c, l, "model")
        wasm = {"n": 200, "max_abs_diff": 9e-7, "argmax_agree": 200, "median_wasm_us": 3900.0}
        probe = {"features": ["len", "entropy"], "training_candidates": 100, "validation_candidates": 50, "validation_positives": 10,
                 "pr_auc": 0.61, "roc_auc": 0.8, "best_f1": 0.6, "share_of_positives": 0.2}
        md, ok, _ = gate.results(s, i, c, l, "model", g, goal, wasm, probe)
        self.assertTrue(ok)
        self.assertIn("WASM (200 candidates, thresholds at 1): median 3900 µs", md)
        self.assertIn("PR-AUC 0.6100", md)
        # reported, never read by a gate
        self.assertEqual(gate.gates(s, i, c, l, "model")[0], g)

    def test_files_and_exit_status(self):
        with tempfile.TemporaryDirectory() as d:
            paths = {}
            for name, obj in zip(("scores", "issues", "clean", "latency"), passing()):
                paths[name] = os.path.join(d, name + ".json")
                with open(paths[name], "w") as f:
                    json.dump(obj, f)
            import subprocess
            import sys

            cmd = [sys.executable, "-I", os.path.join(HERE, "s6-secrets2-gate.py"), "--scores", paths["scores"], "--issues", paths["issues"],
                   "--clean", paths["clean"], "--latency", paths["latency"], "--out-dir", os.path.join(d, "out")]
            r = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)
            out = gate.load(os.path.join(d, "out", "gate.json"))
            self.assertTrue(out["all_hold"] and out["is_the_pre_registered_gate"])
            self.assertEqual(sorted(out["gates"]), ["G1", "G2", "G3", "G4", "G5", "G6", "G7"])
            self.assertIn("before any test file was read", out["uuid_choice"])
            with open(os.path.join(d, "out", "results.md")) as f:
                self.assertTrue(f.read().startswith("# s6-secrets v2"))
            # the same numbers on validation: reported, exit status 1
            scores = copy.deepcopy(passing()[0])
            scores["split"] = "validation"
            with open(paths["scores"], "w") as f:
                json.dump(scores, f)
            r = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(r.returncode, 1)
            self.assertFalse(gate.load(os.path.join(d, "out", "gate.json"))["is_the_pre_registered_gate"])
            r = subprocess.run(cmd[:3] + ["--scores", os.path.join(d, "missing.json"), "--out-dir", d], capture_output=True, text=True)
            self.assertEqual(r.returncode, 1)


if __name__ == "__main__":
    unittest.main()
