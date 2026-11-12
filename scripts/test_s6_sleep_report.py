#!/usr/bin/env python3
"""Tests of scripts/s6-sleep-report.py: the pilot rules, the choice of N and the bootstrap.

  python3 -I scripts/test_s6_sleep_report.py
"""
import importlib.util
import os
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("report", os.path.join(HERE, "s6-sleep-report.py"))
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


class Rules(unittest.TestCase):
    def test_the_memory_must_beat_the_control_by_ten_points(self):
        self.assertTrue(report.pilot_rules(0.36, 0.5, 0.5, 0.5, 0.25)["memory_works"])
        self.assertFalse(report.pilot_rules(0.34, 0.5, 0.5, 0.5, 0.25)["memory_works"])

    def test_a_shortcut_is_kept_when_it_loses_less_than_a_point(self):
        r = report.pilot_rules(0.6, 0.695, 0.70, 0.705, 0.25)
        self.assertEqual(r, {"memory_works": True, "truncation_kept": True, "drawn_n_kept": True})
        r = report.pilot_rules(0.6, 0.68, 0.70, 0.72, 0.25)
        self.assertEqual((r["truncation_kept"], r["drawn_n_kept"]), (False, False))

    def test_the_flags_follow_the_rules(self):
        ok = {"memory_works": True, "truncation_kept": True, "drawn_n_kept": True}
        self.assertEqual(report.form_flags(ok), [])
        self.assertEqual(report.form_flags({**ok, "truncation_kept": False}), ["--sleep-full"])
        self.assertEqual(report.form_flags({**ok, "drawn_n_kept": False}), ["--sleep-passes", "4"])
        self.assertIsNone(report.form_flags({**ok, "memory_works": False}))

    def test_the_smallest_n_within_a_point_of_the_best(self):
        self.assertEqual(report.choose_n({1: 0.50, 2: 0.595, 4: 0.60}), 2)
        self.assertEqual(report.choose_n({1: 0.60, 2: 0.60, 4: 0.60}), 1)
        self.assertEqual(report.choose_n({1: 0.50, 2: 0.55, 4: 0.60}), 4)


class Bootstrap(unittest.TestCase):
    def test_a_clear_gain_has_an_interval_above_zero(self):
        pairs = [(6, 4, 8)] * 150 + [(5, 5, 8)] * 50
        d, lo, hi = report.paired_bootstrap(pairs, draws=500)
        self.assertAlmostEqual(d, (300 / 1600))
        self.assertTrue(0 < lo <= d <= hi)

    def test_no_gain_has_an_interval_around_zero(self):
        pairs = [(5, 4, 8), (4, 5, 8)] * 100
        d, lo, hi = report.paired_bootstrap(pairs, draws=500)
        self.assertEqual(d, 0)
        self.assertTrue(lo < 0 < hi)


class Typed(unittest.TestCase):
    def test_reads_the_ternary_row(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "typed.md")
            with open(path, "w") as f:
                f.write("## Summary\n\n| model | last-exit acc | early |\n|---|---:|---:|\n")
                f.write("| latent f32 (final QAT master weights) | 0.4100 | 0.40 |\n")
                f.write("| ternary (fake-quant) | 0.4321 | 0.42 |\n")
            self.assertEqual(report.typed_accuracy(path), 0.4321)


if __name__ == "__main__":
    unittest.main()
