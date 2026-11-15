#!/usr/bin/env python3
"""Tests of scripts/s6-cast2-gate.py on synthetic scores: python3 -I scripts/test_s6_cast2_gate.py"""

import csv
import importlib.util
import json
import os
import tempfile
import time
import unittest
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("s6_cast2_gate", os.path.join(HERE, "s6-cast2-gate.py"))
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


def reference_rows():
    with open(gate.SN_REF, newline="") as f:
        return list(csv.DictReader(f))


def write_results(d: Path, scale_mase: dict, scale_wql: float = 1.0):
    """all_results.csv: the seasonal-naive reference times a factor per term."""
    d.mkdir(parents=True, exist_ok=True)
    rows = reference_rows()
    cols = list(rows[0])
    with open(d / "all_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            r = dict(r)
            term = r["dataset"].split("/")[2]
            r[gate.MASE] = str(float(r[gate.MASE]) * scale_mase[term])
            r[gate.WQL] = str(float(r[gate.WQL]) * scale_wql)
            w.writerow(r)


class Gate(unittest.TestCase):
    def test_g3_compares_with_the_port_without_flip(self):
        port_noflip = {"relative_MASE": 0.90, "relative_WQL": 0.75}
        self.assertTrue(gate.g3_holds({"relative_MASE": 0.97, "relative_WQL": 0.81}, port_noflip))
        # 1.083 x 0.90 = 0.9747: above it fails, on either metric
        self.assertFalse(gate.g3_holds({"relative_MASE": 0.98, "relative_WQL": 0.70}, port_noflip))
        self.assertFalse(gate.g3_holds({"relative_MASE": 0.80, "relative_WQL": 0.82}, port_noflip))
        self.assertIsNone(gate.g3_holds(None, port_noflip))
        self.assertIsNone(gate.g3_holds({"relative_MASE": 0.8, "relative_WQL": None}, port_noflip))

    def test_the_harness_must_be_validated_before_training(self):
        with tempfile.TemporaryDirectory() as t:
            check, started = Path(t, "check.json"), Path(t, "started.json")
            self.assertFalse(gate.harness_validated_before(check, started))
            check.write_text(json.dumps({"validated": True}))
            started.write_text("{}")
            os.utime(check, (time.time() - 100, time.time() - 100))
            self.assertTrue(gate.harness_validated_before(check, started))
            # validated after training started: G3 cannot stand
            os.utime(check, (time.time() + 100, time.time() + 100))
            self.assertFalse(gate.harness_validated_before(check, started))
            check.write_text(json.dumps({"validated": False}))
            os.utime(check, (time.time() - 100, time.time() - 100))
            self.assertFalse(gate.harness_validated_before(check, started))

    def test_the_gate_reads_g1_g2_and_g3_from_the_files(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            run = t / "runs" / "s6-cast2-pico"
            run.mkdir(parents=True)
            gout, cz = t / "gift", t / "cz"
            # v2 at 0.8 x the seasonal naive at every term: G1's MASE holds, its WQL too at 0.5
            write_results(gout / "s6-cast2-pico", {"short": 0.8, "medium": 0.8, "long": 0.8}, 0.5)
            (run / "checks.json").write_text(json.dumps({"ok": True}))
            (run / "latency").mkdir()
            (run / "latency" / "latency.json").write_text(json.dumps({"measured": "x", "host": "y", "native": [], "wasm": []}))
            for d, m, w in (("s6-cast2-pico", 0.95, 0.80), ("tinycast-noflip", 0.90, 0.75), ("tinycast", 0.88, 0.72)):
                (cz / d).mkdir(parents=True)
                (cz / d / "summary.json").write_text(json.dumps({"relative_MASE": m, "relative_WQL": w, "tasks": {"MASE": 27}}))
            (cz / "tinycast" / "check.json").write_text(json.dumps({"validated": True}))
            os.utime(cz / "tinycast" / "check.json", (time.time() - 100, time.time() - 100))
            (run / "started.json").write_text("{}")
            args = type("A", (), dict(run=str(run), gift_eval_out=str(gout), reference_out=str(t / "ref"),
                                     chronos_out=str(cz), chronos_v2_out=None, series2=str(t / "s2"), models=str(t / "m"), g3="gate", data=str(t / "d"),
                                     out=str(run / "GATE.md"), diagnostics=None, smoke=False))
            gate.cmd_gate(args)
            g = json.loads((run / "gate.json").read_text())
            self.assertTrue(g["G1"] and g["G2"])
            self.assertTrue(g["G3"])  # 0.95 <= 1.083 x 0.90 and 0.80 <= 1.083 x 0.75 (not the port with flip)
            self.assertEqual(g["verdict"], "PASS")
            text = (run / "GATE.md").read_text()
            self.assertGreater(text.count("Disclosure (docs/s6-cast2-prereg.md §10)"), 5)
            # a long term at the seasonal naive fails G2
            write_results(gout / "s6-cast2-pico", {"short": 0.8, "medium": 0.8, "long": 1.0}, 0.5)
            gate.cmd_gate(args)
            g = json.loads((run / "gate.json").read_text())
            self.assertFalse(g["G2"])
            self.assertEqual(g["verdict"], "FAIL")
            # G3 reported (the default, amendment 2): the verdict carries the in-sample sentence
            args.g3 = "reported"
            write_results(gout / "s6-cast2-pico", {"short": 0.8, "medium": 0.8, "long": 0.8}, 0.5)
            gate.cmd_gate(args)
            self.assertEqual(json.loads((run / "gate.json").read_text())["verdict"], "PASS")
            self.assertIn(gate.READ_IN_SAMPLE, (run / "GATE.md").read_text())


if __name__ == "__main__":
    unittest.main()
