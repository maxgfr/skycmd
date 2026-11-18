"""The guard of T5 (s6-route v7, docs/s6-route7-prereg.md 6 to 8, amendment 2): every reader of
T5 (scripts/route-baselines.py --split t5, scripts/route7-gate.sh through scripts/route/t5-guard.py)
is refused without runs/s6-route7/FRESH-OPEN, and with a file whose sha256 is not the one of
runs/s6-route7/test-files.json. On an invented fixture; standard library only.

    python3 -I scripts/test_route_t5.py
"""
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "route"))
import common  # noqa: E402


def sha(p):
    with open(p, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


class Fixture:
    """A T5 folder of invented slates, its test-files.json and (optionally) its FRESH-OPEN."""

    def __init__(self, tmp, open_=True):
        self.t5 = os.path.join(tmp, "toucan5")
        fresh = os.path.join(self.t5, "fresh")
        os.makedirs(fresh)
        files = {}
        for name in ("t5_tools", "t5_tools_masked", "t5_tools_rw", "t5_tools_masked_rw", "t5_tools.servers"):
            p = os.path.join(fresh, name + ".jsonl")
            with open(p, "w") as f:
                f.write(json.dumps({"uid": "u1", "split": "fresh", "opts": ["a: x", "b: y", "none"],
                                    "target": [1.0, 0.0, 0.0]}) + "\n")
            files[name + ".jsonl"] = sha(p)
        self.runs = os.path.join(tmp, "runs", "s6-route7")
        os.makedirs(self.runs)
        self.files = os.path.join(self.runs, "test-files.json")
        with open(self.files, "w") as f:
            json.dump({"about": "invented", "files": files}, f)
        self.open = os.path.join(self.runs, "FRESH-OPEN")
        if open_:
            with open(self.open, "w") as f:
                f.write("opened\n")
        self.fresh = fresh

    def __enter__(self):
        self.saved = (common.T5_OPEN, common.T5_FILES, os.environ.get("T5"))
        common.T5_OPEN, common.T5_FILES = self.open, self.files
        os.environ["T5"] = self.t5
        return self

    def __exit__(self, *a):
        common.T5_OPEN, common.T5_FILES, t5 = self.saved
        if t5 is None:
            os.environ.pop("T5", None)
        else:
            os.environ["T5"] = t5


class Guard(unittest.TestCase):
    def test_open_and_recorded_files_are_read(self):
        with tempfile.TemporaryDirectory() as t, Fixture(t) as fx:
            self.assertEqual(common.t5_file("t5_tools"), os.path.join(fx.fresh, "t5_tools.jsonl"))
            self.assertEqual(len(common.t5_check()), 5)

    def test_refused_without_fresh_open(self):
        with tempfile.TemporaryDirectory() as t, Fixture(t, open_=False):
            with self.assertRaisesRegex(common.Refused, "runs/s6-route7/FRESH-OPEN"):
                common.t5_file("t5_tools")
            with self.assertRaises(common.Refused):
                common.t5_check()

    def test_refused_with_a_wrong_hash(self):
        with tempfile.TemporaryDirectory() as t, Fixture(t) as fx:
            with open(os.path.join(fx.fresh, "t5_tools_rw.jsonl"), "a") as f:
                f.write("\n")
            # any changed file closes T5, whichever file is asked for
            with self.assertRaisesRegex(common.Refused, "not the file recorded"):
                common.t5_file("t5_tools")

    def test_the_blind_copies_need_their_source_and_other_names_are_refused(self):
        with tempfile.TemporaryDirectory() as t, Fixture(t) as fx:
            with self.assertRaisesRegex(common.Refused, "no "):
                common.t5_file("t5_tools.blind")
            with open(os.path.join(fx.fresh, "t5_tools.blind.jsonl"), "w") as f:
                f.write("{}\n")
            self.assertTrue(common.t5_file("t5_tools.blind").endswith("t5_tools.blind.jsonl"))
            for bad in ("f_tools", "t5_tools_rw.blind", "../t5_tools"):
                with self.assertRaises(common.Refused):
                    common.t5_file(bad)

    def test_the_guard_script_exits_3_when_refused(self):
        with tempfile.TemporaryDirectory() as t, Fixture(t, open_=False) as fx:
            env = dict(os.environ, T5=fx.t5)
            args = [sys.executable, "-I", os.path.join(HERE, "route", "t5-guard.py"),
                    "--open", fx.open, "--files", fx.files]
            r = subprocess.run(args, env=env, capture_output=True, text=True)
            self.assertEqual(r.returncode, 3, r.stderr)
            with open(fx.open, "w") as f:
                f.write("opened\n")
            r = subprocess.run(args, env=env, capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)
            with open(os.path.join(fx.fresh, "t5_tools.jsonl"), "a") as f:
                f.write("\n")
            r = subprocess.run(args, env=env, capture_output=True, text=True)
            self.assertEqual(r.returncode, 3, r.stderr)

    def test_the_t5_split_of_the_baselines_is_guarded(self):
        """scripts/route-baselines.py --split t5 reads its files through common.t5_file only."""
        with open(os.path.join(HERE, "route-baselines.py")) as f:
            src = f.read()
        self.assertIn('elif split == "t5":\n                path = common.t5_file(f)', src)
        self.assertIn('if split in ("test", "fresh", "t5"):\n                raise', src)


if __name__ == "__main__":
    unittest.main()
