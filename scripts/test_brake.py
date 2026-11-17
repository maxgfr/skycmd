#!/usr/bin/env python3
"""Tests of scripts/brake.py, on a fake clock (they run at once): python3 -I scripts/test_brake.py"""
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import brake  # noqa: E402
from brake import Brake, cool_wait, parse_brake_line, parse_cool  # noqa: E402


class Clock:
    """A monotonic and a wall clock that only sleeping (or `work`) moves forward."""

    def __init__(self, wall=0.0):
        self.mono = 1000.0
        self.wall_t = float(wall)
        self.sleeps = []
        self.on_sleep = None

    def monotonic(self):
        return self.mono

    def wall(self):
        return self.wall_t

    def sleep(self, s):
        self.sleeps.append(s)
        self.mono += s
        self.wall_t += s
        if self.on_sleep:
            self.on_sleep(self)

    def work(self, s):
        self.mono += s
        self.wall_t += s

    def brake(self, *args, **kwargs):
        logs = kwargs.pop("logs", [])
        return Brake(*args, monotonic=self.monotonic, wall=self.wall, sleep=self.sleep,
                     log=logs.append, **kwargs)


class Parsing(unittest.TestCase):
    def test_a_line_as_the_trainer_reads_it(self):
        self.assertEqual(parse_brake_line("0.85 50:10\n"), (0.85, (3000, 600)))
        self.assertEqual(parse_brake_line("0.5"), (0.5, None))
        self.assertEqual(parse_brake_line(" 1 45:15 "), (1.0, (2700, 900)))
        for bad in ["", "fast", "0", "1.5", "0.5 45", "0.5 45:15 more", "0.5 45:0", "0.5 a:b",
                    "nan", "0.5 inf:5", "0.5 1_0:5", "0.5 0.001:0.001"]:
            self.assertIsNone(parse_brake_line(bad), bad)

    def test_work_and_rest_in_minutes(self):
        self.assertEqual(parse_cool("45:15"), (2700, 900))
        self.assertEqual(parse_cool("0.5:0.25"), (30, 15))
        self.assertEqual(parse_cool("0.0125:1"), (1, 60))  # 0.75 s rounds half away from zero
        for bad in ["45", "0:15", "45:0", "a:b", "-1:5", "inf:5", "45:15:5", "45::15", " 45:15"]:
            self.assertIsNone(parse_cool(bad), bad)

    def test_the_cycle_works_then_rests(self):
        cycle = (45 * 60, 15 * 60)
        self.assertEqual(cool_wait(0, cycle), 0)
        self.assertEqual(cool_wait(45 * 60 - 1, cycle), 0)
        self.assertEqual(cool_wait(45 * 60, cycle), 15 * 60)
        self.assertEqual(cool_wait(59 * 60, cycle), 60)
        self.assertEqual(cool_wait(60 * 60, cycle), 0)


class FromEnv(unittest.TestCase):
    def test_defaults_and_messages(self):
        b = Brake.from_env({})
        self.assertEqual((b.duty, b.cool, b.file), (1.0, None, None))
        b = Brake.from_env({"SKY_DUTY": "", "SKY_COOL": "", "SKY_BRAKE_FILE": ""})
        self.assertEqual((b.duty, b.cool, b.file), (1.0, None, None))
        b = Brake.from_env({"SKY_DUTY": "0.5", "SKY_COOL": "45:15"})
        self.assertEqual((b.duty, b.cool), (0.5, (2700, 900)))
        with self.assertRaisesRegex(ValueError, r"SKY_DUTY is not a number: fast"):
            Brake.from_env({"SKY_DUTY": "fast"})
        for d in ["0", "-0.5", "1.5", "nan"]:
            with self.assertRaisesRegex(ValueError, r"SKY_DUTY must be in \(0, 1\]"):
                Brake.from_env({"SKY_DUTY": d})
        with self.assertRaisesRegex(ValueError, r"SKY_COOL must be work:rest in minutes, as 45:15, not 45"):
            Brake.from_env({"SKY_COOL": "45"})


class Rests(unittest.TestCase):
    def test_the_duty_sets_the_short_rest(self):
        c = Clock(wall=10)
        b = c.brake(0.5)
        c.work(2.0)
        self.assertAlmostEqual(b.rest(), 2.0)  # 2 s of work at duty 0.5: 2 s of rest
        c.work(3.0)
        b.rest()
        self.assertAlmostEqual(c.sleeps[-1], 3.0)
        b = c.brake(0.8)
        c.work(4.0)
        b.rest()
        self.assertAlmostEqual(c.sleeps[-1], 1.0)  # 4 * (1 / 0.8 - 1)

    def test_work_seconds_given_by_the_caller(self):
        c = Clock(wall=10)
        b = c.brake(0.5)
        c.work(10.0)
        b.rest(1.5)
        self.assertAlmostEqual(c.sleeps[-1], 1.5)

    def test_full_duty_never_sleeps(self):
        c = Clock(wall=10)
        b = c.brake(1.0)
        c.work(5.0)
        self.assertEqual(b.rest(), 0.0)
        self.assertEqual(c.sleeps, [])

    def test_the_long_cycle_follows_the_wall_clock(self):
        # 1 min of work, 1 of rest; at Unix time 70 the cycle is 10 s into its rest part.
        c = Clock(wall=70)
        b = c.brake(1.0, (60, 60))
        rested = b.rest()
        # duty 1: a short rest of 0 s; no file: one sleep to the end of the long rest
        self.assertEqual(c.sleeps, [0.0, 50])
        self.assertAlmostEqual(rested, 50)
        self.assertEqual(c.wall_t % 120, 0)
        self.assertEqual(b.rest(), 0.0)  # in the work part now: no rest

    def test_working_seconds_leave_out_every_rest(self):
        c = Clock(wall=0)
        b = c.brake(0.5, (60, 60))
        for _ in range(10):
            c.work(5.0)
            b.rest()
        self.assertAlmostEqual(b.working_seconds, 50.0)
        self.assertAlmostEqual(b.rested_seconds, c.mono - 1000.0 - 50.0)
        self.assertGreater(b.rested_seconds, 50.0)  # short rests and at least one long rest


class File(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.dir.name, "brake")

    def tearDown(self):
        self.dir.cleanup()

    def write(self, text):
        with open(self.path, "w") as f:
            f.write(text)

    def test_the_file_overrides_the_environment(self):
        self.write("0.85 50:10\n")
        c = Clock(wall=0)
        logs = []
        b = c.brake(0.5, (2700, 900), self.path, logs=logs)
        self.assertEqual((b.duty, b.cool), (0.85, (3000, 600)))
        self.assertEqual(logs, [f"brake: duty 0.85, 50 min of work, 10 of rest ({self.path})"])

    def test_a_missing_or_invalid_file_keeps_the_last_setting(self):
        c = Clock(wall=0)
        b = c.brake(1.0, None, self.path)  # no file yet
        self.assertEqual((b.duty, b.cool), (1.0, None))
        self.write("0.5 45:15")
        b.read_file()
        self.assertEqual((b.duty, b.cool), (0.5, (2700, 900)))
        for bad in ["soon", "0", "1.5", "0.5 45", "0.5 45:15 more", "0.5 1_0:5", ""]:
            self.write(bad)
            b.read_file()
            self.assertEqual((b.duty, b.cool), (0.5, (2700, 900)), bad)
        with open(self.path, "wb") as f:
            f.write(b"\xff\xfe")
        b.read_file()
        self.assertEqual((b.duty, b.cool), (0.5, (2700, 900)))
        os.remove(self.path)
        b.read_file()
        self.assertEqual((b.duty, b.cool), (0.5, (2700, 900)))
        self.write("1")
        b.read_file()
        self.assertEqual((b.duty, b.cool), (1.0, None))

    def test_the_file_is_read_again_every_few_seconds(self):
        self.write("1")
        c = Clock(wall=0)
        b = c.brake(1.0, None, self.path)
        self.write("0.5")
        c.work(brake.FILE_PERIOD - 1)
        b.rest()
        self.assertEqual(b.duty, 1.0)  # too early
        c.work(1)
        b.rest()
        self.assertEqual(b.duty, 0.5)

    def test_a_new_line_ends_a_long_rest(self):
        # 1 s of work in a cycle of 601 s, at Unix time 101: deep in the rest part.
        self.write("1 0.0167:10")
        c = Clock(wall=101)
        b = c.brake(1.0, None, self.path)
        self.assertEqual(b.cool, (1, 600))

        def lift(clock):
            if clock.wall_t >= 121:
                self.write("1")
        c.on_sleep = lift
        rested = b.rest()
        self.assertTrue(all(s <= brake.FILE_PERIOD for s in c.sleeps), c.sleeps)
        self.assertLess(rested, 30)  # ended by the file, not at the end of the 600 s rest
        self.assertIsNone(b.cool)


if __name__ == "__main__":
    unittest.main()
