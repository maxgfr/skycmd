#!/usr/bin/env python3
"""Tests of scripts/s6-sleep3-report.py on synthetic logs and readings: C5 (a pass, and a fail for
each of its three conditions, on either seed), the refusal of the pilot, the in-sample stop, the
pilot rule, the splits and the guard on the test seeds.

  python3 -I scripts/test_s6_sleep3_report.py
"""
import contextlib
import importlib.util
import io
import json
import os
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("s6_sleep3_report", os.path.join(HERE, "s6-sleep3-report.py"))
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


def write(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f)


def write_log(run, rows, final=True):
    os.makedirs(run, exist_ok=True)
    with open(os.path.join(run, "log.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
        if final:
            f.write(json.dumps({"step": rows[-1]["step"] if rows else 0, "final": True}) + "\n")


def c5_rows(ce=0.01, acc=0.99, off_ce=3.0, off_acc=0.24, aux=0.99, steps=600, early=1.39, leave=180):
    """A 600-step log: every step a recall step of 64 asks (a stateless step every tenth), one in
    four answered again without the memory; at chance until `leave`, then (ce, acc)."""
    rows, k = [], 0
    for step in range(1, steps + 1):
        if step % 10 == 0:
            rows.append({"step": step, "loss": 0.5})
            continue
        k += 1
        trained = step >= leave
        r = {"step": step, "seq_game": "recall", "passes": 1, "seq_step_loss": 0.1, "asks": 64.0,
             "ask_ce": ce if trained else early, "ask_ce_last": ce if trained else early,
             "ask_acc": acc if trained else 0.25, "aux_ce": 0.1, "aux_acc": aux if trained else 0.3,
             "ask_splits": [0.0] * 13}
        if k % 4 == 1:
            r["off"] = {"ask_ce": off_ce, "ask_ce_last": off_ce, "ask_acc": off_acc, "ask_splits": [0.0] * 13}
        rows.append(r)
    return rows


def run_c5(tmp, seed0, seed1):
    runs = []
    for s, rows in enumerate((seed0, seed1)):
        run = os.path.join(tmp, f"s{s}")
        write_log(run, rows)
        runs.append(run)
    out = os.path.join(tmp, "c5.json")
    with contextlib.redirect_stdout(io.StringIO()):
        ok = report.cmd_c5(runs[0], runs[1], out)
    with open(out) as f:
        return ok, json.load(f)


class C5(unittest.TestCase):
    def test_a_trained_memory_passes_on_both_seeds(self):
        with tempfile.TemporaryDirectory() as tmp:
            ok, res = run_c5(tmp, c5_rows(), c5_rows(ce=0.05, acc=0.97))
            self.assertTrue(ok)
            self.assertEqual(res["steps"], [550, 600])
            for s in ("seed0", "seed1"):
                self.assertTrue(all(res["seeds"][s]["conditions"].values()))
                w = res["seeds"][s]["window"]
                self.assertEqual(w["recall_steps"], 45)  # 550..600 without its 6 stateless steps
                self.assertGreater(w["off_steps"], 0)
                # The first ten recall steps with a mean under 1.36: 171 to 179 at chance (1.39)
                # and 181, the first trained one (180 is a stateless step).
                self.assertEqual(res["seeds"][s]["answer_left_chance_at_step"], 171)

    def test_condition_1_fails_on_the_cross_entropy_or_the_accuracy(self):
        for bad in (dict(ce=0.31), dict(acc=0.89)):
            with tempfile.TemporaryDirectory() as tmp:
                ok, res = run_c5(tmp, c5_rows(), c5_rows(**bad))
                self.assertFalse(ok, bad)
                c = res["seeds"]["seed1"]["conditions"]
                self.assertEqual(c, {"1_answer": False, "2_memory_off": True, "3_aux": True})
                self.assertTrue(res["seeds"]["seed0"]["pass"])

    def test_the_bars_are_inclusive(self):
        with tempfile.TemporaryDirectory() as tmp:
            ok, _ = run_c5(tmp, c5_rows(ce=0.30, acc=0.90, off_ce=1.30, off_acc=0.30, aux=0.95), c5_rows())
            self.assertTrue(ok)

    def test_condition_2_fails_when_the_answer_survives_without_the_memory(self):
        for bad in (dict(off_ce=1.29), dict(off_acc=0.31)):
            with tempfile.TemporaryDirectory() as tmp:
                ok, res = run_c5(tmp, c5_rows(**bad), c5_rows())
                self.assertFalse(ok, bad)
                c = res["seeds"]["seed0"]["conditions"]
                self.assertEqual(c, {"1_answer": True, "2_memory_off": False, "3_aux": True})

    def test_condition_2_fails_when_nothing_was_read_without_the_memory(self):
        rows = c5_rows()
        for r in rows:
            r.pop("off", None)
        with tempfile.TemporaryDirectory() as tmp:
            ok, res = run_c5(tmp, c5_rows(), rows)
            self.assertFalse(ok)
            self.assertFalse(res["seeds"]["seed1"]["conditions"]["2_memory_off"])

    def test_condition_3_fails_on_the_auxiliary_accuracy(self):
        with tempfile.TemporaryDirectory() as tmp:
            ok, res = run_c5(tmp, c5_rows(), c5_rows(aux=0.94))
            self.assertFalse(ok)
            c = res["seeds"]["seed1"]["conditions"]
            self.assertEqual(c, {"1_answer": True, "2_memory_off": True, "3_aux": False})

    def test_an_answer_at_chance_fails_and_never_left_chance(self):
        with tempfile.TemporaryDirectory() as tmp:
            ok, res = run_c5(tmp, c5_rows(leave=10_000), c5_rows())
            self.assertFalse(ok)
            self.assertIsNone(res["seeds"]["seed0"]["answer_left_chance_at_step"])

    def test_the_window_is_read_pooled_over_the_asks(self):
        rows = [{"step": 550, "seq_game": "recall", "asks": 10.0, "ask_ce": 1.0, "ask_ce_last": 1.0, "ask_acc": 0.5,
                 "aux_acc": 1.0, "off": {"ask_ce": 2.0, "ask_ce_last": 2.0, "ask_acc": 0.2}},
                {"step": 600, "seq_game": "recall", "asks": 30.0, "ask_ce": 0.0, "ask_ce_last": 0.0, "ask_acc": 1.0,
                 "aux_acc": 0.5},
                {"step": 601, "seq_game": "recall", "asks": 30.0, "ask_ce": 9.0, "ask_ce_last": 9.0, "ask_acc": 0.0}]
        w = report.c5_window(rows, 550, 600)
        self.assertAlmostEqual(w["ask_ce"], 0.25)
        self.assertAlmostEqual(w["ask_ce_step_mean"], 0.5)
        self.assertAlmostEqual(w["ask_acc"], 0.875)
        self.assertAlmostEqual(w["aux_acc"], 0.625)
        self.assertEqual((w["off_steps"], w["off_asks"]), (1, 10.0))
        self.assertAlmostEqual(w["off_ask_ce"], 2.0)

    def test_a_run_that_did_not_end_is_not_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_log(os.path.join(tmp, "s0"), c5_rows(), final=False)
            write_log(os.path.join(tmp, "s1"), c5_rows())
            with self.assertRaises(SystemExit):
                report.cmd_c5(os.path.join(tmp, "s0"), os.path.join(tmp, "s1"), os.path.join(tmp, "c5.json"))


V2_VERDICT = {
    "pass": False, "needs_amendment": False, "c1": True, "c2": False, "form": "frame",
    "c2_deviation": "C2 missed its bar by 0.010 ...", "c3": True, "c4": False,
}


class Ready(unittest.TestCase):
    def ready(self, verdict, c5):
        with tempfile.TemporaryDirectory() as tmp:
            v2c = os.path.join(tmp, "v2")
            write(os.path.join(v2c, "verdict.json"), verdict)
            root = os.path.join(tmp, "root")
            if c5 is not None:
                write(os.path.join(root, "checks", "c5.json"), {"pass": c5})
            with contextlib.redirect_stdout(io.StringIO()):
                try:
                    report.cmd_ready(root, v2c)
                    refused = False
                except SystemExit:
                    refused = True
            with open(os.path.join(root, "checks", "ready.json")) as f:
                return refused, json.load(f)

    def test_v2s_checks_as_read_and_a_passed_c5_let_the_pilot_run(self):
        refused, res = self.ready(V2_VERDICT, True)
        self.assertFalse(refused)
        self.assertTrue(res["pass"] and res["v2_pass"])
        # v2's own verdict failed on C4, which C5 replaces.
        self.assertFalse(V2_VERDICT["pass"])

    def test_a_failed_or_missing_c5_refuses_the_pilot(self):
        for c5 in (False, None):
            refused, res = self.ready(V2_VERDICT, c5)
            self.assertTrue(refused)
            self.assertFalse(res["pass"])

    def test_a_failed_v2_check_refuses_the_pilot(self):
        for change in ({"c1": False}, {"c3": False}, {"c2_deviation": None}, {"form": "token"}):
            refused, res = self.ready({**V2_VERDICT, **change}, True)
            self.assertTrue(refused, change)
            self.assertFalse(res["v2_pass"])

    def test_c2_passed_on_its_bar_needs_no_deviation(self):
        refused, _ = self.ready({**V2_VERDICT, "c2": True, "c2_deviation": None}, True)
        self.assertFalse(refused)


class Stop(unittest.TestCase):
    def stop(self, ce):
        with tempfile.TemporaryDirectory() as tmp:
            rows = [{"step": s, "seq_game": "recall", "seq_step_loss": 0.1, "asks": 64.0, "ask_ce": 1.39 if s < 50 else ce,
                     "ask_ce_last": ce, "ask_acc": 0.3} for s in range(1, 120)]
            rows += [{"step": 120, "seq_game": "rule110", "seq_step_loss": 0.2}]
            write_log(os.path.join(tmp, "pilot", "m-a"), rows)
            with contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(SystemExit) as e:
                    report.cmd_stop(tmp)
            with open(os.path.join(tmp, "pilot", "stop.json")) as f:
                return e.exception.code, json.load(f)

    def test_the_pilot_stops_above_1_336(self):
        code, res = self.stop(1.34)
        self.assertEqual(code, 3)
        self.assertTrue(res["stop"])
        self.assertEqual((res["phase_a_sequence_steps"], res["phase_a_recall_steps"]), (120, 119))

    def test_the_pilot_goes_on_at_or_below_the_bar(self):
        code, res = self.stop(1.30)
        self.assertEqual(code, 0)
        self.assertFalse(res["stop"])


def episodes(seeds, right_per_episode, asks=8, shown=None):
    out = []
    for s in seeds:
        r = right_per_episode(s)
        a = [[i % 4, i < r] for i in range(asks)]
        out.append({"seed": s, "score": float(r), "max_score": float(asks), "asks": a,
                    "shown": shown or [1 + (i % 2) for i in range(asks)]})
    return out


def probe(score_fn, passes=1, memory=True, off=False):
    seeds = report.v2.dev2_seeds()
    eps = episodes(seeds, score_fn)
    right = sum(e["score"] for e in eps)
    return {"game": "recall", "memory": memory, "memory_off": off, "passes": passes, "from_seed": report.v2.DEV2_SEED,
            "n_episodes": report.v2.DEV2_EPISODES, "score": right / (8 * len(eps)), "buckets": [], "episodes": eps}


class Pilot(unittest.TestCase):
    def test_the_rule_works_with_a_gain_from_the_memory(self):
        m = probe(lambda s: 4 + (s // 20) % 2)  # 0.5625
        c = probe(lambda s: 2, memory=False)  # 0.25
        off = probe(lambda s: 2 + ((s // 20) % 5 == 0), off=True)  # 0.275
        verdict, _, rule = report.pilot_rule(m, c, off)
        self.assertEqual(verdict, "works")
        self.assertGreater(rule["ci95"][0], 0)

    def test_a_gain_that_survives_the_zeroed_memory_is_a_defect(self):
        m = probe(lambda s: 4)
        c = probe(lambda s: 2, memory=False)
        off = probe(lambda s: 3, off=True)  # 0.375: 0.125 above the control
        self.assertEqual(report.pilot_rule(m, c, off)[0], "defect")

    def test_a_gain_under_0_10_fails(self):
        m = probe(lambda s: 2 + ((s // 20) % 3 == 0))  # about 0.29
        c = probe(lambda s: 2, memory=False)
        off = probe(lambda s: 2, off=True)
        self.assertEqual(report.pilot_rule(m, c, off)[0], "fails")

    def test_the_splits_by_delay_and_by_colours(self):
        r = {"episodes": [{"asks": [[0, True], [1, False], [0, True], [3, True]], "shown": [1, 2, 3, 1]}]}
        sp = report.splits(r)
        self.assertEqual(sp["by_delay"]["1-4"], 1.0)
        self.assertEqual(sp["by_delay"]["5-8"], 0.0)
        self.assertTrue(sp["by_delay"]["9-16"] != sp["by_delay"]["9-16"])  # NaN: no ask
        self.assertEqual(sp["by_colours"], {"one": 1.0, "several": 0.5})
        self.assertEqual(sp["by_colours_asks"], {"one": 2, "several": 2})

    def test_the_nano_runs_need_a_rule_that_passed(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(SystemExit):
                report.cmd_nano_ready(tmp)
            for verdict in ("stopped", "defect", "fails"):
                write(os.path.join(tmp, "pilot.json"), {"verdict": verdict})
                with self.assertRaises(SystemExit):
                    report.cmd_nano_ready(tmp)
            write(os.path.join(tmp, "pilot.json"), {"verdict": "works"})
            with contextlib.redirect_stdout(io.StringIO()):
                report.cmd_nano_ready(tmp)


class TestSeeds(unittest.TestCase):
    def test_no_test_seed_before_every_nano_run_ended_and_n_is_chosen(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(len(report.test_ready(tmp)), 13)
            for run in report.nano_runs(tmp):
                write_log(run, [{"step": 1, "loss": 1.0}])
            self.assertEqual(report.test_ready(tmp), ["N is not chosen (no n.json)"])
            write(os.path.join(tmp, "n.json"), {"n": 1})
            self.assertEqual(report.test_ready(tmp), [])
            write_log(report.nano_runs(tmp)[3], [{"step": 1, "loss": 1.0}], final=False)
            self.assertEqual(len(report.test_ready(tmp)), 1)
            write_log(report.nano_runs(tmp)[3], [{"step": 1, "loss": 1.0}])
            write(os.path.join(tmp, "test", "done"), {})
            self.assertEqual(report.test_ready(tmp), ["the test seeds were already read (test/done)"])

    def test_a_smoke_never_writes_under_the_real_root(self):
        with self.assertRaises(SystemExit):
            report.refuse_smoke_root("runs/s6-sleep3", True)
        report.refuse_smoke_root("runs/smoke-sleep3/runs/s6-sleep3", True)
        report.refuse_smoke_root("runs/s6-sleep3", False)


if __name__ == "__main__":
    unittest.main()
