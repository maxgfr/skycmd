#!/usr/bin/env python3
"""Tests of the s6-ear v3 scripts other than the gate (docs/s6-ear3-prereg.md §3.7, §13): the test
guard, the pure parts of the Silero cross-check and the optional listening check, the conversation
mode of the teacher and the offset of the Smart Turn baseline (their argument checks; no model is run).

    python3 -I scripts/test_turn3_scripts.py
"""
import array
import hashlib
import importlib.util
import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))


def load(name, file):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, file))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


guard = load("turn3_guard", "turn3-guard.py")
vad = load("turn_vadcheck", "turn-vadcheck.py")
lc = load("turn_labelcheck", "turn-labelcheck.py")


def f32_wav(path, x):
    data = array.array("f", x).tobytes()
    hdr = b"RIFF" + struct.pack("<I", 36 + len(data)) + b"WAVEfmt " + struct.pack("<IHHIIHH", 16, 3, 1, 16000, 64000, 4, 32)
    Path(path).write_bytes(hdr + b"data" + struct.pack("<I", len(data)) + data)


class Guard(unittest.TestCase):
    def test_the_tests_open_once_after_the_checks_with_the_exported_file(self):
        with tempfile.TemporaryDirectory() as d:
            run, model = Path(d, "run"), Path(d, "m.gguf")
            model.write_bytes(b"GGUF v3")
            with self.assertRaises(guard.Refused):
                guard.open_tests(run, model)
            (run / "harness3").mkdir(parents=True)
            (run / "harness3" / "check.json").write_text(json.dumps({"passed": False}))
            with self.assertRaises(guard.Refused):
                guard.open_tests(run, model)
            (run / "harness3" / "check.json").write_text(json.dumps({"passed": True, "checked": "t"}))
            with self.assertRaises(guard.Refused):  # no latencies yet
                guard.open_tests(run, model)
            (run / "latency").mkdir()
            (run / "latency" / "latency.json").write_text(json.dumps({"measured": "t"}))
            (run / "parity").mkdir()
            for f in ("parity.json", "parity-conv.json"):
                (run / "parity" / f).write_text("{}")
            with self.assertRaises(guard.Refused):  # nothing read before the tests are open
                guard.check(run, model, "eval")
            rec = guard.open_tests(run, model)
            self.assertEqual(rec["model"]["sha256"], hashlib.sha256(b"GGUF v3").hexdigest())
            with self.assertRaises(guard.Refused):
                guard.open_tests(run, model)
            guard.check(run, model, "eval")
            (run / "stages").mkdir()
            (run / "stages" / "eval.done").write_text("x")
            with self.assertRaises(guard.Refused):  # read once
                guard.check(run, model, "eval")
            guard.check(run, model, "eotbench")
            model.write_bytes(b"another file")
            with self.assertRaises(guard.Refused):
                guard.check(run, model, "eotbench")
            # the command line: exit 3 when refused
            r = subprocess.run([sys.executable, "-I", os.path.join(HERE, "turn3-guard.py"), "check", "--run", str(run),
                                "--model", str(model), "--stage", "gate"], capture_output=True, text=True)
            self.assertEqual(r.returncode, 3)


class Vadcheck(unittest.TestCase):
    def test_git_blob_hash_and_runs(self):
        # `git hash-object` of "hello\n"
        self.assertEqual(vad.git_blob_sha1(b"hello\n"), "ce013625030ba8dba906f756967f9e9ca394464a")
        self.assertEqual(vad.runs_of([False, True, True, False, True]), [[512, 1536], [2048, 2560]])
        self.assertEqual(vad.runs_of([]), [])

    def test_a_shard_through_a_fake_detector(self):
        class Fake:
            """Speech where the samples are loud, one probability per whole window."""

            def probs(self, x):
                return [1.0 if max(abs(v) for v in x[k * 512:(k + 1) * 512]) > 0.1 else 0.0 for k in range(len(x) // 512)]

        with tempfile.TemporaryDirectory() as d:
            conv = Path(d)
            (conv / "wav-00003").mkdir()
            f32_wav(conv / "wav-00003" / "s.0.wav", [0.0] * 1024 + [0.5] * 2048 + [0.0] * 700)
            f32_wav(conv / "wav-00003" / "s.1.wav", [0.0] * 4000)
            (conv / "xcheck-00003.jsonl").write_text(json.dumps({"session": "s", "wav": ["s.0.wav", "s.1.wav"],
                                                                  "energy": [], "samples": 4000}) + "\n")
            self.assertEqual(vad.run(conv, "00003", Fake()), 0)
            lines = [json.loads(l) for l in (conv / "silero-00003.jsonl").read_text().splitlines()]
            self.assertEqual(lines[0]["runs"], [[1024, 3072]])
            self.assertEqual((lines[0]["windows"], lines[1]["runs"]), (7, []))
            self.assertFalse((conv / "wav-00003").exists(), "the transient WAVs go with the shard")
            # a shard without validation sessions has nothing to do; an unknown one is an error
            (conv / "conv-00004.json").write_text("{}")
            self.assertEqual(vad.run(conv, "00004", Fake()), 0)
            with self.assertRaises(SystemExit):
                vad.run(conv, "00005", Fake())

    def test_the_wav_reader_wants_mono_float(self):
        with tempfile.TemporaryDirectory() as d:
            f32_wav(Path(d, "a.wav"), [0.25, -0.5])
            self.assertEqual(list(vad.read_wav_f32(Path(d, "a.wav"))), [0.25, -0.5])


class Labelcheck(unittest.TestCase):
    def test_the_sample_is_the_smallest_keys_per_class_and_scored_blind(self):
        with tempfile.TemporaryDirectory() as d:
            conv, out = Path(d, "conv"), Path(d, "lc")
            saved = lc.PER_CLASS
            lc.PER_CLASS = 3
            try:
                for tag, sess in (("00001", "a"), ("00002", "b")):
                    (conv / f"wav-{tag}").mkdir(parents=True)
                    for ch in (0, 1):
                        f32_wav(conv / f"wav-{tag}" / f"{sess}.{ch}.wav", [0.1 * ch] * 16000 * 12)
                    energy = [{"channel": k % 2, "o": 16000 * (2 + k), "eot": k % 2 == 0} for k in range(8)]
                    (conv / f"xcheck-{tag}.jsonl").write_text(json.dumps(
                        {"session": sess, "wav": [f"{sess}.0.wav", f"{sess}.1.wav"], "energy": energy}) + "\n")
                    lc.cmd_collect(Namespace(conv=str(conv), tag=tag, dir=str(out)))
                sample = [json.loads(l) for l in (out / "sample.jsonl").read_text().splitlines()]
                ids = [f"{s}:{k % 2}:{16000 * (2 + k)}" for s in "ab" for k in range(8)]
                for eot in (True, False):
                    want = sorted((lc.key(i) for i in ids if (int(i.split(':')[2]) // 16000 - 2) % 2 == (0 if eot else 1)))[:3]
                    self.assertEqual(sorted(x["key"] for x in sample if x["eot"] == eot), want)
                wavs = sorted(p.name for p in out.glob("*.wav"))
                self.assertEqual(wavs, sorted(f"{x['code']}.wav" for x in sample))
                self.assertTrue(all("eot" not in w and "hold" not in w for w in wavs))
                # ten seconds of two channels: o - 6 s to o + 4 s
                b = (out / wavs[0]).read_bytes()
                self.assertEqual(len(b) - 44, 10 * 16000 * 2 * 2)
                lc.cmd_sheet(Namespace(dir=str(out)))
                rows = (out / "answers.csv").read_text().splitlines()
                self.assertEqual(len(rows), 7)
                # answers: right on the eot ones, "cannot tell" on one hold, wrong on the others
                by = {x["code"]: x for x in sample}
                lines = ["code,answer"]
                tells = 0
                for c in [r.split(",")[0] for r in rows[1:]]:
                    if by[c]["eot"]:
                        lines.append(f"{c},finished")
                    elif tells == 0:
                        lines.append(f"{c},cannot tell")
                        tells += 1
                    else:
                        lines.append(f"{c},finished")
                (out / "answers.csv").write_text("\n".join(lines) + "\n")
                lc.cmd_score(Namespace(dir=str(out), out=str(Path(d, "lc.json"))))
                r = json.loads(Path(d, "lc.json").read_text())
                self.assertEqual((r["answered"], r["agreement"]), (6, 0.5))
                self.assertAlmostEqual(r["cannot_tell_share"], 1 / 6)
            finally:
                lc.PER_CLASS = saved


class Arguments(unittest.TestCase):
    def test_the_teacher_and_the_baseline_take_the_v3_options(self):
        # parsed only: the scripts' imports of numpy and onnxruntime happen inside their functions
        for script, args in (("turn-teacher.py", ["--records", "x"]), ("ear-baseline.py", ["--eotbench", "x"])):
            src = Path(HERE, script).read_text()
            self.assertIn("--records" if script == "turn-teacher.py" else "--offset-ms", src)
        tt = Path(HERE, "turn-teacher.py").read_text()
        self.assertIn("never a target", tt)
        eb = Path(HERE, "ear-baseline.py").read_text()
        self.assertIn('"smartturn" if not a.offset_ms else f"smartturn{a.offset_ms}"', eb)


if __name__ == "__main__":
    unittest.main()
