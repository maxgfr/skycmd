#!/usr/bin/env python3
"""Tests of scripts/turn-gate.py on synthetic inputs (a pass and a fail for each condition):
python3 -I scripts/test_turn_gate.py"""

import importlib.util
import json
import math
import os
import random
import struct
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("turn_gate", os.path.join(HERE, "turn-gate.py"))
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


def jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def run_log(path, f32_acc=None, ternary_bce=None, ema_bce=None):
    rows = [{"step": 100, "loss": 0.6, "elapsed_min": 12.0}]
    if f32_acc is not None:
        rows.append({"step": 100, "mode": "f32", "val": {"acc": f32_acc, "bce": 0.4, "agree": 0.9, "n": 10}})
    if ternary_bce is not None:
        rows.append({"step": 100, "mode": "ternary", "val": {"acc": 0.8, "bce": ternary_bce, "agree": 0.9, "n": 10}})
    if ema_bce is not None:
        rows.append({"step": 100, "mode": "ema-ternary", "val": {"acc": 0.8, "bce": ema_bce, "agree": 0.9, "n": 10}})
    rows.append({"step": 100, "final": True, "elapsed_min": 12.0, "epochs": 1.0})
    jsonl(path, rows)


class Probe(unittest.TestCase):
    def run_probe(self, medians_us):
        with tempfile.TemporaryDirectory() as d:
            nat = [{"model": f"x/{r}.gguf", "median_us": 900, "p95_us": 1000, "frontend_us_per_hop": 10, "params": 1}
                   for r in gate.RUNGS]
            was = [{"model": f"{r}.gguf", "path": f"x/{r}.gguf", "median_us": m, "p95_us": m + 100,
                    "frontend_us_per_hop": 40, "params": 1, "file_bytes": 5} for r, m in zip(gate.RUNGS, medians_us)]
            jsonl(f"{d}/n.jsonl", nat)
            jsonl(f"{d}/w.jsonl", was)
            st = gate.cmd_probe(Namespace(native=f"{d}/n.jsonl", wasm=f"{d}/w.jsonl", out=f"{d}/rung.json"))
            return st, json.load(open(f"{d}/rung.json"))["rung"]

    def test_first_rung_under_four_ms(self):
        self.assertEqual(self.run_probe([5200, 3900, 3000, 2000]), (0, "R2"))
        self.assertEqual(self.run_probe([4000, 3000, 3000, 2000]), (0, "R1"))

    def test_no_rung_qualifies(self):
        self.assertEqual(self.run_probe([9000, 7000, 4100, 4001]), (1, None))


class Pilot(unittest.TestCase):
    def pilot(self, accs, hard):
        with tempfile.TemporaryDirectory() as d:
            for m, acc in zip(("0.5", "1", "2"), accs):
                run_log(f"{d}/lr{m}/log.jsonl", f32_acc=acc)
            run_log(f"{d}/hard/log.jsonl", f32_acc=hard)
            st = gate.cmd_pilot(Namespace(dir=d, base=1e-3, rates_only=False, out=f"{d}/choice.json"))
            return st, json.load(open(f"{d}/choice.json"))

    def test_distil_unless_hard_wins_by_more_than_the_margin(self):
        st, c = self.pilot([0.84, 0.86, 0.85], 0.864)
        self.assertEqual((st, c["distil"], c["lr"], c["stop"]), (0, True, 1e-3, False))
        st, c = self.pilot([0.84, 0.86, 0.85], 0.8651)
        self.assertEqual((c["distil"], c["lr"]), (False, 1e-3))
        st, c = self.pilot([0.84, 0.85, 0.87], 0.86)
        self.assertEqual(c["lr"], 2e-3)

    def test_stop_rule(self):
        st, c = self.pilot([0.80, 0.81, 0.79], 0.815)
        self.assertEqual((st, c["stop"]), (4, True))
        st, c = self.pilot([0.80, 0.81, 0.79], 0.82)
        self.assertEqual((st, c["stop"]), (0, False))

    def test_rates_only_prints_the_best_rate(self):
        with tempfile.TemporaryDirectory() as d:
            for m, acc in zip(("0.5", "1", "2"), (0.9, 0.8, 0.85)):
                run_log(f"{d}/lr{m}/log.jsonl", f32_acc=acc)
            self.assertEqual(gate.cmd_pilot(Namespace(dir=d, base=2e-3, rates_only=True, out=None)), 0)


class Checkpoint(unittest.TestCase):
    def choose(self, ternary, ema):
        with tempfile.TemporaryDirectory() as d:
            run_log(f"{d}/log.jsonl", ternary_bce=ternary, ema_bce=ema)
            import io
            from contextlib import redirect_stdout

            buf = io.StringIO()
            with redirect_stdout(buf):
                gate.cmd_checkpoint(Namespace(log=f"{d}/log.jsonl"))
            return buf.getvalue().strip()

    def test_the_ema_only_when_its_bce_is_lower(self):
        self.assertEqual(self.choose(0.40, 0.39), "last-ema")
        self.assertEqual(self.choose(0.40, 0.40), "last")
        self.assertEqual(self.choose(0.40, None), "last")


def store(d, rows, split="val"):
    """A store manifest with labels (and a teacher file of logits)."""
    Path(d).mkdir(parents=True, exist_ok=True)
    jsonl(f"{d}/manifest-00000.jsonl", [{"id": i, "hops": 400, "offset": 0, "split": split, "label": y,
                                          "lang": lang, "dataset": ds} for i, y, lang, ds in rows])


class Calibrate(unittest.TestCase):
    def test_temperature_and_tau85(self):
        rng = random.Random(1)
        rows, probs = [], []
        for k in range(4000):
            z = rng.gauss(0, 2)
            y = rng.random() < gate.sigmoid(z)
            # the model is overconfident by 2: logits doubled
            rows.append((f"c{k}", y, "eng", "d"))
            probs.append({"id": f"c{k}", "logit": 2 * z, "prob": gate.sigmoid(2 * z)})
        with tempfile.TemporaryDirectory() as d:
            store(f"{d}/train", rows)
            jsonl(f"{d}/p.jsonl", probs)
            gate.cmd_calibrate(Namespace(probs=f"{d}/p.jsonl", store=f"{d}/train", out=f"{d}/c.json"))
            c = json.load(open(f"{d}/c.json"))
        self.assertAlmostEqual(c["temperature"], 2.0, delta=0.2)
        self.assertGreaterEqual(c["val"]["end_recall@tau85"], 0.85)
        self.assertLess(c["val"]["nll"], c["val"]["nll_t1"])

    def test_tau_for_recall_is_the_highest(self):
        p = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05]
        y = [True] * 10
        self.assertEqual(gate.tau_for_recall(p, y, 0.85), 0.1)  # 9 of 10 >= 0.1; 8 of 10 >= 0.2
        self.assertEqual(gate.tau_for_recall(p, y, 0.8), 0.2)
        self.assertEqual(gate.tau_for_recall(p * 2, y * 2, 0.85), 0.1)  # 18 of 20 >= 0.1, 16 >= 0.2


class Harness(unittest.TestCase):
    def check(self, acc, timer, n=gate.TEST_CLIPS, pad="left"):
        with tempfile.TemporaryDirectory() as d:
            json.dump({"accuracy": acc, "timer_accuracy": timer, "positive_rate": timer, "n": n, "pad": pad},
                      open(f"{d}/b.json", "w"))
            st = gate.cmd_harness(Namespace(baseline=f"{d}/b.json", out=f"{d}/c.json"))
            return st, json.load(open(f"{d}/c.json"))["passed"]

    def test_the_bar(self):
        self.assertEqual(self.check(0.9263, 0.49791), (0, True))
        self.assertEqual(self.check(0.9292, 0.4979), (0, True))
        self.assertEqual(self.check(0.9294, 0.4979), (1, False))
        self.assertEqual(self.check(0.8518, 0.4979), (1, False))

    def test_the_timer_clips_and_padding(self):
        self.assertEqual(self.check(0.9263, 0.4981), (1, False))
        self.assertEqual(self.check(0.9263, 0.4979, n=3000), (1, False))
        self.assertEqual(self.check(0.9263, 0.4979, pad="right"), (1, False))


class Terms(unittest.TestCase):
    def test_g1(self):
        self.assertTrue(gate.check_g1(0.85, 0.9263))
        self.assertFalse(gate.check_g1(0.849, 0.9263))
        # with a strong teacher the second bar binds
        self.assertFalse(gate.check_g1(0.86, 0.95))
        self.assertTrue(gate.check_g1(0.87, 0.95))

    def test_g2(self):
        self.assertTrue(gate.check_g2(0.20, 0.80))
        self.assertFalse(gate.check_g2(0.201, 0.85))
        self.assertFalse(gate.check_g2(0.10, 0.799))

    def lat(self, ms=3.0, hop=0.05, size=600_000):
        return {"wasm": {"median_ms": ms, "frontend_ms_per_hop": hop}, "gguf_bytes": size}

    def test_g3(self):
        par = {"n": 20, "max_abs_diff": 3e-6, "same_decision": 20}
        self.assertTrue(gate.check_g3(self.lat(), par)[0])
        self.assertFalse(gate.check_g3(self.lat(ms=5.01), par)[0])
        self.assertFalse(gate.check_g3(self.lat(hop=0.11), par)[0])
        self.assertFalse(gate.check_g3(self.lat(size=3_000_001), par)[0])
        self.assertFalse(gate.check_g3(self.lat(), {"n": 20, "max_abs_diff": 2e-4, "same_decision": 20})[0])
        self.assertFalse(gate.check_g3(self.lat(), {"n": 20, "max_abs_diff": 1e-5, "same_decision": 19})[0])
        self.assertFalse(gate.check_g3(self.lat(), None)[0])
        self.assertFalse(gate.check_g3(None, par)[0])


class Metrics(unittest.TestCase):
    def test_counts_auc_and_bootstrap(self):
        p = [0.9, 0.8, 0.4, 0.3, 0.6, 0.1]
        y = [True, True, True, False, False, False]
        s = gate.Scores(p, y, taus=(0.5,)).stats()
        self.assertAlmostEqual(s["accuracy@0.5"], 4 / 6)
        self.assertAlmostEqual(s["false_cutoff@0.5"], 1 / 3)
        self.assertAlmostEqual(s["fpr_all@0.5"], 1 / 6)
        self.assertAlmostEqual(s["recall@0.5"], 2 / 3)
        self.assertAlmostEqual(s["auc"], 8 / 9)
        b = gate.bootstrap(gate.Scores(p * 20, y * 20), reps=200)
        self.assertLessEqual(b["accuracy@0.5"]["lo"], b["accuracy@0.5"]["value"])
        self.assertGreaterEqual(b["accuracy@0.5"]["hi"], b["accuracy@0.5"]["value"])
        o = gate.oracle_point(p, y)
        self.assertEqual(o["threshold"], 0.4)
        self.assertAlmostEqual(o["false_cutoff"], 1 / 3)


class Gate(unittest.TestCase):
    """The whole verdict on synthetic files: a passing model and a failing one."""

    def setUp(self):
        gate.BOOT = 50

    def build(self, d, acc_target, ms=3.0, harness_ok=True, with_latency=True):
        rng = random.Random(7)
        clips, probs, dump = [], [], []
        for k in range(600):
            y = k % 2 == 0
            right = rng.random() < acc_target
            p = (0.75 if y else 0.25) if right else (0.3 if y else 0.6)
            lang = "eng" if k % 3 == 0 else "fra"
            ds = "human_5" if k % 5 == 0 else "chirp3_1"
            clips.append({"id": f"t{k}", "split": "test", "lang": lang, "wav": f"wav/t{k}.wav",
                          "label": {"endpoint": y, "dataset": ds}})
            probs.append({"id": f"t{k}", "logit": math.log(p / (1 - p)), "prob": p, "positions": 100})
            dump.append({"id": f"t{k}", "prob": 0.9 if y else 0.1})
        jsonl(f"{d}/clips.jsonl", clips)
        jsonl(f"{d}/test.jsonl", probs)
        jsonl(f"{d}/dump.jsonl", dump)
        json.dump({"passed": harness_ok, "teacher_accuracy": 0.9263}, open(f"{d}/check.json", "w"))
        if with_latency:
            json.dump({"wasm": {"median_ms": ms, "frontend_ms_per_hop": 0.05, "p95_ms": ms},
                       "native": {"median_ms": 1.0}, "gguf_bytes": 700_000,
                       "smart_turn_cpu_native": {"median_ms": 31.9}}, open(f"{d}/lat.json", "w"))
        jsonl(f"{d}/parity.json", [{"mode": "turn", "n": 20, "max_abs_diff": 1e-6, "same_decision": 20}])
        json.dump({"temperature": 1.2, "tau50": 0.5, "tau85": 0.3}, open(f"{d}/calib.json", "w"))
        # eot-bench: a store, v2's and Smart Turn's probabilities
        st = Path(d) / "data" / "eotbench" / "store"
        jsonl(st / "manifest-en-00000.jsonl", [{"id": f"e{k}", "hops": 400, "offset": 0, "label": k % 2 == 0,
                                                 "lang": "en", "cohort": "general", "split": "eotbench"} for k in range(40)])
        jsonl(st / "smartturn-en-00000.jsonl", [{"id": f"e{k}", "prob": 0.8 if k % 2 == 0 else 0.2} for k in range(40)])
        jsonl(f"{d}/eot.jsonl", [{"id": f"e{k}", "prob": 0.5 + (0.1 if k % 2 == 0 else -0.1) * (1 if k % 4 else -1)} for k in range(40)])

    def run_gate(self, d, **kw):
        a = Namespace(run=f"{d}/run", test_probs=f"{d}/test.jsonl", labels=f"{d}/clips.jsonl",
                      harness=f"{d}/check.json", smart_turn_dump=f"{d}/dump.jsonl", latency=f"{d}/lat.json",
                      parity=f"{d}/parity.json", model="m.gguf", calib=f"{d}/calib.json", val_probs=None,
                      data=f"{d}/data", eotbench_probs=f"{d}/eot.jsonl", pilot=None, probe=None,
                      out=f"{d}/GATE.md", smoke=False, labels_store=None)
        for k, v in kw.items():
            setattr(a, k, v)
        Path(f"{d}/run").mkdir(exist_ok=True)
        gate.cmd_gate(a)
        return json.load(open(f"{d}/GATE.json"))

    def test_a_pass_and_a_fail(self):
        with tempfile.TemporaryDirectory() as d:
            self.build(d, 0.9)
            r = self.run_gate(d)
            self.assertEqual(r["verdict"], "pass", r["meaning"])
            self.assertTrue(r["G1"]["pass"] and r["G2"]["pass"] and r["G3"]["pass"])
            md = open(f"{d}/GATE.md").read()
            self.assertIn("Disclosure (§11)", md)
            self.assertIn("Take v2", r["sentence"])
            # eot-bench: v2 near chance against a perfect teacher -> the reading of §8.4
            self.assertFalse(r["eotbench"]["carries_over"])
            self.assertIn("does not carry over", md)
            self.assertAlmostEqual(r["agreement"]["test"]["agreement"], r["G1"]["accuracy"], places=6)
        with tempfile.TemporaryDirectory() as d:
            self.build(d, 0.75)
            r = self.run_gate(d)
            self.assertEqual(r["verdict"], "fail")
            self.assertFalse(r["G1"]["pass"])
            self.assertIn("Take Smart Turn", r["sentence"])
        with tempfile.TemporaryDirectory() as d:
            self.build(d, 0.9, ms=6.0)
            r = self.run_gate(d)
            self.assertEqual((r["verdict"], r["G3"]["pass"], r["G3"]["terms"]["wasm_median"]), ("fail", False, False))
            self.assertIn("G3 fails", r["meaning"])

    def test_no_test_score_before_the_checks(self):
        with tempfile.TemporaryDirectory() as d:
            self.build(d, 0.9, harness_ok=False)
            with self.assertRaises(SystemExit):
                self.run_gate(d)
        with tempfile.TemporaryDirectory() as d:
            self.build(d, 0.9, with_latency=False)
            with self.assertRaises(SystemExit):
                self.run_gate(d)
        with tempfile.TemporaryDirectory() as d:
            self.build(d, 0.9)
            with self.assertRaises(SystemExit):
                self.run_gate(d, harness=f"{d}/none.json")


class Data(unittest.TestCase):
    def test_every_shard_needs_features_and_targets(self):
        with tempfile.TemporaryDirectory() as d:
            t = Path(d) / "train"
            t.mkdir()
            for i in range(2):
                (t / f"manifest-{i:05d}.jsonl").write_text("")
            (t / "teacher-00000.done").write_text("{}")
            self.assertEqual(gate.cmd_data(Namespace(data=d, shards=2)), 1)
            (t / "teacher-00001.done").write_text("{}")
            self.assertEqual(gate.cmd_data(Namespace(data=d, shards=2)), 0)

    def test_teacher_logit_reads_the_record(self):
        with tempfile.TemporaryDirectory() as d:
            t = Path(d) / "train"
            t.mkdir()
            rec = struct.pack("<f", -1.5) + bytes(gate.TEACHER_RECORD - 4)
            (t / "teacher-00000.bin").write_bytes(bytes(gate.TEACHER_RECORD) + rec)
            self.assertEqual(gate.teacher_logit(d, {"_tag": "00000", "_index": 1}), -1.5)
        self.assertEqual(gate.TEACHER_RECORD, 10472)


if __name__ == "__main__":
    unittest.main()
