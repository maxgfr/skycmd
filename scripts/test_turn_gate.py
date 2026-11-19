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


# ============================================================================= s6-ear v3


def conv_store(d, recs, energy):
    """A conversation store: `recs` manifest fields (cut_hops, dmax_ms, label, split, ...), each record's
    hops with channel 42 at `energy(record, hop index)`."""
    Path(d).mkdir(parents=True, exist_ok=True)
    feat, lines, off = bytearray(), [], 0
    for r in recs:
        n = r["cut_hops"] + r["dmax_ms"] // 10
        for h in range(n):
            row = [0.0] * 43
            row[42] = energy(r, h)
            feat += struct.pack("<43e", *row)
        lines.append({**r, "hops": n, "offset": off})
        off += n * 86
    Path(d, "feat-00000.bin").write_bytes(bytes(feat))
    jsonl(Path(d, "manifest-00000.jsonl"), lines)


def v3_log(path, acc, auc, ckpt=None, ema=None):
    rows = [{"step": 10, "loss": 0.5}]
    rows.append({"step": 10, "mode": "f32", "val": {"acc": acc, "bce": 0.4, "agree": 0.9, "n": 5},
                 "conv": {"auc": auc, "bce": 0.5, "n": 9}, "ckpt_loss": 0.45})
    if ckpt is not None:
        rows.append({"step": 10, "mode": "ternary", "val": {"acc": acc, "bce": 0.4, "agree": 0.9, "n": 5},
                     "conv": {"auc": auc, "bce": 0.5, "n": 9}, "ckpt_loss": ckpt})
    if ema is not None:
        rows.append({"step": 10, "mode": "ema-ternary", "val": {"acc": acc, "bce": 0.4, "agree": 0.9, "n": 5},
                     "conv": {"auc": auc, "bce": 0.5, "n": 9}, "ckpt_loss": ema})
    rows.append({"step": 10, "final": True, "elapsed_min": 12.0})
    jsonl(path, rows)


class Bleed(unittest.TestCase):
    def run_bleed(self, leak):
        with tempfile.TemporaryDirectory() as d:
            recs = [{"id": f"r{i}", "split": "train", "label": i % 2 == 0, "cut_hops": 380, "dmax_ms": 2400}
                    for i in range(40)]
            recs.append({"id": "v", "split": "val", "label": True, "cut_hops": 380, "dmax_ms": 200})

            def energy(r, h):
                # silence after o; with a leak, the end-of-turn records are louder there from 0.6 s on
                if h >= r["cut_hops"] + 3:
                    base = -3.0 + 0.01 * (int(r["id"][1:]) % 7 if r["id"] != "v" else 0)
                    return base + (1.0 if leak and r["label"] and h >= r["cut_hops"] + 60 else 0.0)
                return 1.0
            conv_store(f"{d}/store", recs, energy)
            st = gate.cmd_bleed(Namespace(store=f"{d}/store", out=f"{d}/bleed.json"))
            return st, json.load(open(f"{d}/bleed.json"))

    def test_no_leak_passes(self):
        st, b = self.run_bleed(False)
        self.assertEqual((st, b["pass"]), (0, True))
        self.assertEqual(b["records"], 40)
        self.assertEqual(sorted(b["by_d_ms"], key=int)[0], "200")

    def test_a_leak_stops_the_build(self):
        st, b = self.run_bleed(True)
        self.assertEqual((st, b["pass"]), (1, False))
        # at 0.2-0.6 s nothing leaks yet: the AUC stays near 0.5 there
        self.assertLess(b["by_d_ms"]["200"]["auc"], 0.6)
        self.assertGreater(b["by_d_ms"]["2400"]["auc"], 0.6)


class Pilot3(unittest.TestCase):
    def run_pilot(self, arms, v2_low):
        with tempfile.TemporaryDirectory() as d:
            for name, (acc, auc) in arms.items():
                v3_log(f"{d}/pilot/{name}/log.jsonl", acc, auc)
            recs = [{"id": f"r{i}", "split": "val", "label": i % 2 == 0, "cut_hops": 380, "dmax_ms": 200}
                    for i in range(20)]
            conv_store(f"{d}/conv", recs, lambda r, h: 0.0)
            # v2's file on the same cuts: the first v2_low records' ends of turn below every hold
            # (AUC 0.6 with 8, 1.0 with none)
            probs = [{"id": r["id"], "d_ms": 200, "prob": (0.3 if i < v2_low else 0.6) if r["label"] else 0.4}
                     for i, r in enumerate(recs)]
            jsonl(f"{d}/v2.jsonl", probs)
            st = gate.cmd_pilot3(Namespace(dir=f"{d}/pilot", v2_probs=f"{d}/v2.jsonl", conv_store=f"{d}/conv",
                                           base=gate.V2_RATE, out=f"{d}/choice.json"))
            return st, json.load(open(f"{d}/choice.json"))

    def test_the_best_eligible_from_v2_arm(self):
        st, c = self.run_pilot({"P1": (0.90, 0.70), "P2": (0.89, 0.74), "P3": (0.87, 0.80), "P4": (0.90, 0.744)}, 8)
        # P3 is not eligible (0.87 < 0.88); P4 is not better than P2 by more than 0.005
        self.assertEqual((c["arm"], c["start"], c["minutes"]), ("P2", "v2", 120))
        self.assertEqual(c["lr"], float(f"{0.5 * gate.V2_RATE:.4g}"))
        self.assertAlmostEqual(c["v2_metric"], 0.6)
        self.assertEqual((st, c["stop"]), (0, False))

    def test_the_scratch_arm_when_better_by_more_than_the_margin(self):
        _, c = self.run_pilot({"P1": (0.90, 0.70), "P2": (0.89, 0.74), "P3": (0.88, 0.73), "P4": (0.90, 0.746)}, 8)
        self.assertEqual((c["arm"], c["start"], c["minutes"]), ("P4", "scratch", 240))

    def test_the_stop_rule(self):
        # v2's file ranks the cuts perfectly (AUC 1): no arm can beat it by 0.01
        st, c = self.run_pilot({"P1": (0.90, 0.95), "P2": (0.89, 0.74), "P3": (0.88, 0.73), "P4": (0.90, 0.7)}, 0)
        self.assertEqual((st, c["stop"], c["arm"]), (4, True, "P1"))
        # no eligible arm at all
        st, c = self.run_pilot({"P1": (0.80, 0.95), "P2": (0.80, 0.74), "P3": (0.8, 0.73), "P4": (0.8, 0.9)}, 8)
        self.assertEqual((st, c["stop"], c["arm"]), (4, True, None))

    def test_checkpoint3(self):
        import contextlib
        import io
        with tempfile.TemporaryDirectory() as d:
            v3_log(f"{d}/a.jsonl", 0.9, 0.8, ckpt=0.40, ema=0.39)
            v3_log(f"{d}/b.jsonl", 0.9, 0.8, ckpt=0.40, ema=0.41)
            for f, want in (("a", "last-ema"), ("b", "last")):
                out = io.StringIO()
                with contextlib.redirect_stdout(out):
                    gate.cmd_checkpoint3(Namespace(log=f"{d}/{f}.jsonl"))
                self.assertEqual(out.getvalue().strip(), want)


def gguf_bytes(tensors):
    def s(x):
        b = x.encode()
        return struct.pack("<Q", len(b)) + b
    out = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), 2)
    out += s("general.name") + struct.pack("<I", 8) + s("x")
    out += s("skycmd-turn.list") + struct.pack("<IIQ", 9, 4, 2) + struct.pack("<II", 1, 2)
    for name, dims, t in tensors:
        out += s(name) + struct.pack("<I", len(dims)) + struct.pack(f"<{len(dims)}Q", *dims) + struct.pack("<IQ", t, 0)
    return out


class RankedStats(unittest.TestCase):
    def test_auc_recall_at_fc_and_weights(self):
        p = [0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9, 0.35, 0.95]
        y = [False, False, False, False, True, True, True, True, True, False]
        rk = gate.Ranked(p, y, list(range(10)))
        w = [1.0] * 10
        self.assertAlmostEqual(rk.auc(w), gate.Scores(p, y).stats()["auc"])
        # at most 0.10 of the 5 holds: none may be above the threshold, and 0.95 is a hold
        self.assertEqual(rk.recall_at_fc(w, 0.10), 0.0)
        self.assertEqual(rk.recall_at_fc(w, 0.20), 4 / 5)
        # weights repeat units: without the 0.95 hold
        w[9] = 0.0
        self.assertEqual(rk.recall_at_fc(w, 0.10), 4 / 5)

    def test_paired_bootstrap_is_seeded_and_by_unit(self):
        a = gate.paired_bootstrap(5, {"sum": lambda w: sum(w)}, reps=50)
        self.assertEqual(a["sum"]["value"], 5.0)
        self.assertEqual((a["sum"]["lo"], a["sum"]["hi"]), (5.0, 5.0))
        b = gate.paired_bootstrap(5, {"first": lambda w: w[0]}, reps=200)
        c = gate.paired_bootstrap(5, {"first": lambda w: w[0]}, reps=200)
        self.assertEqual(b, c)
        self.assertLess(b["first"]["lo"], 1.0)

    def test_tau_c_and_gguf_tensors(self):
        p = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
        y = [False] * 10 + [True]
        # holds at or above tau: at most 1 of 10
        self.assertEqual(gate.tau_lowest_fc(p, y), 0.95)
        with tempfile.TemporaryDirectory() as d:
            Path(d, "x.gguf").write_bytes(gguf_bytes([("a.weight", (4, 2), 0), ("b", (3,), 1)]))
            self.assertEqual(gate.gguf_tensors(f"{d}/x.gguf"), [("a.weight", (4, 2), 0), ("b", (3,), 1)])


class Harness3AndGate3(unittest.TestCase):
    """Synthetic test sets with known answers for §8.1 and §9."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = Path(self.tmp.name)
        rng = random.Random(3)
        # Smart Turn's test: 50 clips, v1's manifest
        self.test = [{"id": f"t{i}", "split": "test", "lang": "eng", "wav": f"wav/t{i}.wav",
                      "label": {"endpoint": i % 2 == 0, "dataset": "d1"}} for i in range(50)]
        jsonl(self.d / "clips.jsonl", self.test)
        # eot-bench spans: 120 turns, each a hold span and an eot span; half English
        self.spans = []
        for t in range(120):
            lang = "en" if t % 2 == 0 else "fr"
            for k, lab in enumerate((False, True)):
                self.spans.append({"id": f"u{t}#{k}", "turn": f"u{t}", "label": lab, "lang": lang, "cohort": "general",
                                   "dmax_ms": 2400 if lab else 600, "short": False, "long_hold": False, "cut_hops": 400,
                                   "row": t, "cut": 64000, "split": "eotbench"})
        self.spans.append({"id": "short#0", "turn": "short", "label": True, "lang": "en", "cohort": "general",
                           "dmax_ms": 0, "short": True, "long_hold": False, "cut_hops": 400, "split": "eotbench"})
        jsonl(self.d / "spans" / "manifest-en-00000.jsonl", self.spans)
        self.noise = {s["id"]: rng.random() for s in self.spans}

    def tearDown(self):
        self.tmp.cleanup()

    def probs(self, path, skill, d_list, other_skill=None):
        """A dump whose probability separates the classes by `skill` (0: noise only); `other_skill` for the
        languages other than English."""
        rows = []
        for s in self.spans:
            for d in d_list:
                if d > s["dmax_ms"]:
                    continue
                k = skill if other_skill is None or s["lang"] == "en" else other_skill
                rows.append({"id": s["id"], "d_ms": d, "prob": 0.5 * self.noise[s["id"]] + k * (0.5 if s["label"] else 0.0)})
        jsonl(path, rows)

    def test_harness3(self):
        d = self.d
        Path(d, "policy.json").write_text(json.dumps({"passed": True, "vectors": [{}, {}, {}]}))
        jsonl(d / "v2test.jsonl", [{"id": c["id"], "prob": 0.9 if c["label"]["endpoint"] else 0.1} for c in self.test])
        self.probs(d / "v2eot.jsonl", 0.3, [0])
        jsonl(d / "spans" / "smartturn-en-00000.jsonl", [{"id": s["id"], "prob": 0.5} for s in self.spans])
        a = Namespace(policy=str(d / "policy.json"), labels=str(d / "clips.jsonl"), v2_test=str(d / "v2test.jsonl"),
                      spans=str(d / "spans"), v2_eot=str(d / "v2eot.jsonl"), out=str(d / "check.json"))
        self.assertEqual(gate.cmd_harness3(a), 1)
        c = json.load(open(d / "check.json"))
        self.assertTrue(c["checks"]["policy_port"]["ok"])
        # v2's test accuracy is 1.0 here, not 0.8993; the counts are not v2's either
        self.assertFalse(c["checks"]["smart_turn_test_v2"]["ok"])
        self.assertFalse(c["checks"]["eotbench_cuts"]["ok"])
        self.assertEqual(c["cuts_8_3"]["cuts"], 240)
        # with this set's own numbers as the bars, it passes
        saved = (gate.V2_TEST_ACC, gate.V2_EOT_CUTS, gate.V2_EOT_AUC, gate.ST_EOT_AUC, gate.TEST_CLIPS)
        try:
            gate.V2_TEST_ACC, gate.V2_EOT_CUTS, gate.TEST_CLIPS = 1.0, 241, 50
            gate.V2_EOT_AUC = c["checks"]["eotbench_v2_auc"]["value"]
            gate.ST_EOT_AUC = 0.5
            self.assertEqual(gate.cmd_harness3(a), 0)
        finally:
            gate.V2_TEST_ACC, gate.V2_EOT_CUTS, gate.V2_EOT_AUC, gate.ST_EOT_AUC, gate.TEST_CLIPS = saved

    def run_gate(self, v3_skill, v3_other_skill, acc_ok=True):
        d = self.d
        jsonl(d / "test.jsonl", [{"id": c["id"], "prob": (0.9 if c["label"]["endpoint"] else 0.1)
                                  if (acc_ok or i % 5) else (0.1 if c["label"]["endpoint"] else 0.9)}
                                 for i, c in enumerate(self.test)])
        asks = list(range(0, 2401, 200))
        self.probs(d / "eot3.jsonl", v3_skill, asks, v3_other_skill)
        self.probs(d / "eot2.jsonl", 0.3, asks)
        jsonl(d / "spans" / "smartturn200-en-00000.jsonl", [{"id": s["id"], "prob": 0.5 + 0.1 * s["label"]} for s in self.spans])
        Path(d, "calib.json").write_text(json.dumps({"temperature": 1.0, "tau50": 0.5, "tau85": 0.7}))
        Path(d, "latency.json").write_text(json.dumps({"wasm": {"median_ms": 2.4, "frontend_ms_per_hop": 0.02},
                                                       "gguf_bytes": 631_940}))
        for f in ("parity.json", "parity-conv.json"):
            Path(d, f).write_text(json.dumps({"mode": "turn", "n": 20, "max_abs_diff": 1e-7, "same_decision": 20}))
        if not Path(d, "v2.gguf").exists():
            Path(d, "v2.gguf").write_bytes(gguf_bytes([("a.weight", (4, 2), 0)]))
        Path(d, "v3.gguf").write_bytes(gguf_bytes([("a.weight", (4, 2), 0)]))
        Path(d, "h3.json").write_text(json.dumps({"passed": True}))
        a = Namespace(run=str(d / "run"), test_probs=str(d / "test.jsonl"), labels=str(d / "clips.jsonl"), labels_store=None,
                      test_ext=None, v2_test_ext=None, spans=str(d / "spans"), eot_v3=str(d / "eot3.jsonl"),
                      eot_v2=str(d / "eot2.jsonl"), conv_store=None, conv_v3=None, conv_v2=None, conv_val=None, sessions=None,
                      calib=str(d / "calib.json"), latency=str(d / "latency.json"), parity=str(d / "parity.json"),
                      parity_conv=str(d / "parity-conv.json"), model=str(d / "v3.gguf"), v2_model=str(d / "v2.gguf"),
                      harness3=str(d / "h3.json"), pilot=None, crosscheck=None, bleed=None, conv_summary=None,
                      out=str(d / "GATE.md"), v2_tau85=0.7, boot=200, smoke=False)
        gate.cmd_gate3(a)
        return json.load(open(d / "GATE.json"))

    def test_v3_better_everywhere_passes(self):
        r = self.run_gate(0.9, 0.9)
        for k in ("A1", "A2", "E1", "E2", "E3", "G3"):
            self.assertTrue(r[k]["pass"], (k, r[k]))
        self.assertEqual(r["verdict"], "pass")
        self.assertEqual(r["test_b_eotbench"]["cuts"], 240)
        self.assertIn("v3@tau85", r["test_b_eotbench"]["policy_simulated"])
        self.assertIn("Take v3", r["sentence"])
        self.assertTrue(Path(self.d, "GATE.md").read_text().startswith("# s6-ear3-nano"))

    def test_no_gain_fails_e1(self):
        r = self.run_gate(0.3, 0.3)
        self.assertFalse(r["E1"]["pass"])
        self.assertAlmostEqual(r["E1"]["delta"], 0.0)
        self.assertEqual(r["verdict"], "fail")

    def test_harm_outside_english_fails_e3(self):
        r = self.run_gate(1.5, 0.15)
        self.assertTrue(r["E1"]["pass"], r["E1"])
        self.assertFalse(r["E3"]["pass"])
        self.assertLess(r["E3"]["upper"], 0)
        self.assertIn("E3 fails", r["meaning"])

    def test_a1_and_a_differing_tensor_list(self):
        r = self.run_gate(0.9, 0.9, acc_ok=False)
        self.assertFalse(r["A1"]["pass"])
        self.assertIn("A1 or A2 fails", r["meaning"])
        Path(self.d, "v2.gguf").write_bytes(gguf_bytes([("a.weight", (4, 3), 0)]))
        r = self.run_gate(0.9, 0.9)
        self.assertFalse(r["G3"]["terms"]["same_tensor_list_as_v2"])
        self.assertFalse(r["G3"]["pass"])

    def test_the_gate_refuses_before_the_harness_check(self):
        Path(self.d, "h3.json").write_text(json.dumps({"passed": False}))
        a = Namespace(run=str(self.d), test_probs=str(self.d / "x"), labels=str(self.d / "clips.jsonl"), labels_store=None,
                      calib=str(self.d / "c"), latency=None, harness3=str(self.d / "h3.json"), smoke=False)
        with self.assertRaises(SystemExit):
            gate.cmd_gate3(a)


if __name__ == "__main__":
    unittest.main()
