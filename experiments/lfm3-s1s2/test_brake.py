#!/usr/bin/env python3
"""Tests of the thermal brake of s1s2.py (SKY_BRAKE_FILE): python3 experiments/lfm3-s1s2/test_brake.py"""
import os
import sys
import tempfile
import threading
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s1s2  # noqa: E402


class Brake(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.dir.name, "brake")
        self.saved = (s1s2.BRAKE_FILE, s1s2.BRAKE_FILE_PERIOD, s1s2.DUTY, s1s2.COOL)
        s1s2.BRAKE_FILE, s1s2.DUTY, s1s2.COOL = self.path, 0.5, [2700.0, 900.0]

    def tearDown(self):
        s1s2.BRAKE_FILE, s1s2.BRAKE_FILE_PERIOD, s1s2.DUTY, s1s2.COOL = self.saved
        self.dir.cleanup()

    def write(self, text):
        with open(self.path, "w") as f:
            f.write(text)

    def test_parses_a_line_as_the_trainer_does(self):
        self.assertEqual(s1s2.parse_brake("0.5 45:15\n"), (0.5, [2700.0, 900.0]))
        self.assertEqual(s1s2.parse_brake("0.85 50:10"), (0.85, [3000.0, 600.0]))
        self.assertEqual(s1s2.parse_brake("1"), (1.0, []))
        for bad in ("", "0", "1.5", "fast", "0.5 45", "0.5 45:0", "0.5 45:15 extra", "0.5 a:b"):
            self.assertIsNone(s1s2.parse_brake(bad), bad)

    def test_the_file_wins_when_valid_and_the_environment_otherwise(self):
        self.assertEqual(s1s2.brake(), (0.5, [2700.0, 900.0]))  # no file yet
        self.write("0.85 50:10\n")
        self.assertEqual(s1s2.brake(), (0.85, [3000.0, 600.0]))
        self.write("nonsense")
        self.assertEqual(s1s2.brake(), (0.5, [2700.0, 900.0]))

    def test_a_long_rest_ends_when_the_file_says_so(self):
        # 0.06 s of work in a cycle of ten minutes: the clock is in the rest almost surely.
        self.write("1 0.001:10")
        s1s2.BRAKE_FILE_PERIOD = 0.05
        threading.Timer(0.2, lambda: self.write("1")).start()
        t0 = time.time()
        s1s2.rest(0.0)
        self.assertLess(time.time() - t0, 5.0)

    def test_the_duty_sets_the_short_rest(self):
        self.write("0.5")
        t0 = time.time()
        s1s2.rest(0.1)
        self.assertTrue(0.09 <= time.time() - t0 < 1.0)


if __name__ == "__main__":
    unittest.main()
