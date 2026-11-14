#!/usr/bin/env python3
"""Tests of the runners' conversions to the common result format, of the guards and of the
issue-report scoring, on hand-made inputs: python3 -I scripts/secrets2/test_runners.py

No tool is started and no file of the experiment is read. Every value is made of repeated
letters assembled in code; none has the shape of a key.
"""
import json
import os
import struct
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import clean_text  # noqa: E402
import common  # noqa: E402
import issues  # noqa: E402
import run_scanners  # noqa: E402
import run_secretgate  # noqa: E402
import scorer  # noqa: E402

V = "v" * 12  # an invented value
SPLITS = {
    "training": [{"repo_id": "aaaaaaaa"}],
    "validation": [{"repo_id": "bbbbbbbb"}, {"repo_id": "cccccccc"}],
    "test": [{"repo_id": "dddddddd"}, {"repo_id": "eeeeeeee"}],
    "largest_test_repository": {"repo_id": "dddddddd"},
}


def gguf(kvs):
    """A GGUF v3 header with float and string keys and no tensor."""
    out = b"GGUF" + struct.pack("<IQQ", 3, 0, len(kvs))
    for k, v in kvs.items():
        key = k.encode()
        out += struct.pack("<Q", len(key)) + key
        if isinstance(v, float):
            out += struct.pack("<If", 6, v)
        else:
            s = v.encode()
            out += struct.pack("<IQ", 8, len(s)) + s
    return out


class Guards(unittest.TestCase):
    def test_split_guard(self):
        val = common.Split("validation", SPLITS)
        self.assertEqual(val.repos, ["bbbbbbbb", "cccccccc"])
        self.assertEqual(val.check("data/bbbbbbbb/src/ffffffff.py"), "bbbbbbbb")
        for path in ("data/dddddddd/src/ffffffff.py", "data/aaaaaaaa/x", "data/zzzzzzzz/x", "elsewhere/bbbbbbbb/x", "/data/bbbbbbbb/x"):
            with self.assertRaises(common.Refused, msg=path):
                val.check(path)
        with self.assertRaises(common.Refused) as e:
            val.check_repo("dddddddd")
        self.assertIn("TEST", str(e.exception))
        self.assertIsNone(val.largest)
        self.assertEqual(common.Split("test", SPLITS).largest, "dddddddd")
        twice = dict(SPLITS, validation=SPLITS["validation"] + [{"repo_id": "dddddddd"}])
        with self.assertRaises(common.Refused):
            common.Split("validation", twice)

    def test_model_threshold(self):
        with tempfile.TemporaryDirectory() as d:
            with_tau = os.path.join(d, "a.gguf")
            without = os.path.join(d, "b.gguf")
            with open(with_tau, "wb") as f:
                f.write(gguf({"general.name": "m", common.TAU_KEY: 0.25, "skycmd-s2.exit_threshold.noul": 0.5}))
            with open(without, "wb") as f:
                f.write(gguf({"general.name": "m", "skycmd-s2.exit_threshold.noul": 0.5}))
            self.assertAlmostEqual(common.model_tau(with_tau), 0.25)
            self.assertEqual(common.gguf_metadata(with_tau)["general.name"], "m")
            for bad in (without, os.path.join(d, "missing.gguf"), None):
                with self.assertRaises(common.Refused):
                    common.model_tau(bad)
            with open(without, "wb") as f:
                f.write(gguf({common.TAU_KEY: 1.0}))
            with self.assertRaises(common.Refused):
                common.model_tau(without)

    def test_the_test_split_is_closed(self):
        # no TEST-OPEN file in this checkout: every entry to the test refuses, model or not
        self.assertFalse(os.path.exists(common.TEST_OPEN), "runs/s6-secrets2/TEST-OPEN exists: these tests assume a closed test")
        for call in (lambda: common.open_split("test"), lambda: common.issue_csv("test"), lambda: common.issues_dir("test"),
                     lambda: common.corpus("creddata:test"), lambda: common.corpus("issues:test"), lambda: common.test_is_open("x.gguf")):
            with self.assertRaises(common.Refused):
                call()
        with self.assertRaises(common.Refused):
            common.open_split("everything")
        with self.assertRaises(common.Refused):
            common.corpus("creddata")

    def test_the_test_split_needs_the_file_and_a_threshold(self):
        # with a stand-in for runs/s6-secrets2/TEST-OPEN and for splits.json: the file alone is
        # not enough, the model must carry tau*
        real = (common.TEST_OPEN, common.load_splits, os.environ.pop("SECRETS2_MODEL", None))
        with tempfile.TemporaryDirectory() as d:
            try:
                common.load_splits = lambda: SPLITS
                common.TEST_OPEN = os.path.join(d, "TEST-OPEN")
                with_tau, without = os.path.join(d, "a.gguf"), os.path.join(d, "b.gguf")
                with open(with_tau, "wb") as f:
                    f.write(gguf({common.TAU_KEY: 0.3}))
                with open(without, "wb") as f:
                    f.write(gguf({"general.name": "m"}))
                with self.assertRaises(common.Refused):
                    common.open_split("test", with_tau)  # no TEST-OPEN
                open(common.TEST_OPEN, "w").close()
                for model in (None, without, os.path.join(d, "missing.gguf")):
                    with self.assertRaises(common.Refused):
                        common.open_split("test", model)
                split = common.open_split("test", with_tau)
                self.assertEqual((split.repos, split.largest), (["dddddddd", "eeeeeeee"], "dddddddd"))
                with self.assertRaises(common.Refused):
                    split.check("data/bbbbbbbb/src/f.py")  # a validation file, in a test run
                # the other splits never need either
                self.assertEqual(common.open_split("validation").repos, ["bbbbbbbb", "cccccccc"])
            finally:
                common.TEST_OPEN, common.load_splits = real[0], real[1]
                if real[2] is not None:
                    os.environ["SECRETS2_MODEL"] = real[2]

    def test_flags_round_trip(self):
        with tempfile.TemporaryDirectory() as d:
            flags = [common.flag("bbbbbbbb", "data/bbbbbbbb/src/f.py", 3, 3, 4, 9, score=0.123456789, rule="r", id=7),
                     common.flag("bbbbbbbb", "data/bbbbbbbb/src/f.py", 5, 7, 0, 2), common.flag("bbbbbbbb", "data/bbbbbbbb/src/f.py", 9)]
            self.assertEqual(flags[0], {"repo": "bbbbbbbb", "file": "data/bbbbbbbb/src/f.py", "line": 3, "start": 4, "end": 9, "score": 0.123457, "rule": "r", "id": 7})
            self.assertEqual(flags[1]["end_line"], 7)
            self.assertNotIn("start", flags[2])
            meta = common.write_flags(d, "s", flags, {"version": "1", "corpus": "creddata:validation", "unit": "span"})
            self.assertEqual((meta["flags"], meta["format"]), (3, common.FORMAT))
            run, back = common.read_flags(d, "s")
            self.assertEqual(back, flags)
            self.assertEqual(common.systems_in(d), ["s"])
            with open(os.path.join(d, "s.flags.jsonl"), "a") as f:
                f.write(json.dumps(flags[2]) + "\n")
            with self.assertRaises(common.Refused):
                common.read_flags(d, "s")
            with self.assertRaises(ValueError):
                common.write_flags(d, "t", [], {"version": "1", "corpus": "c", "unit": "word"})


class Places(unittest.TestCase):
    def test_engine_lines(self):
        # the engine ends a line at "\n" only; CredData also at a lone "\r"
        text = f"a = 1\rkey = {V}\r\nnext = {V}\n"
        e = common.EngineLines(text)
        at = text.index(V)
        # engine: line 1, column 12; CredData: line 2, column 6
        self.assertEqual(e.place(1, at, 1, at + len(V)), (2, 6, 2, 6 + len(V)))
        self.assertEqual(e.place(2, 7, 2, 7 + len(V)), (3, 7, 3, 7 + len(V)))

    def test_engine_records_to_flags(self):
        with tempfile.TemporaryDirectory() as d:
            os.makedirs(os.path.join(d, "data", "bbbbbbbb", "src"))
            with open(os.path.join(d, "data", "bbbbbbbb", "src", "f.py"), "w", newline="") as f:
                f.write(f"é\U0001F511 = 1\nkey = {V}\nBEGIN\nbody\nEND\n")
            val = common.Split("validation", SPLITS)
            recs = [{"file": "data/bbbbbbbb/src/f.py", "rule": "r", "line": 2, "endLine": 2, "cpStart": 6, "cpEnd": 6 + len(V)},
                    {"file": "data/bbbbbbbb/src/f.py", "rule": "k", "line": 3, "endLine": 5, "cpStart": 0, "cpEnd": 3}]
            flags = list(run_secretgate.records_to_flags(recs, d, val.check))
            self.assertEqual(flags[0], {"repo": "bbbbbbbb", "file": "data/bbbbbbbb/src/f.py", "line": 2, "start": 6, "end": 6 + len(V), "rule": "r", "id": 0})
            self.assertEqual((flags[1]["line"], flags[1]["end_line"], flags[1]["start"], flags[1]["end"]), (3, 5, 0, 3))
            with self.assertRaises(common.Refused):
                list(run_secretgate.records_to_flags([dict(recs[0], file="data/dddddddd/src/f.py")], d, val.check))

    def leak(self, style, text, match, secret, line, end_line=None):
        """A finding as the tool reports it, from where the match really is."""
        buf = text.encode()
        nl = [0] + [i + 1 for i, b in enumerate(buf) if b == 10]
        a = buf.index(match.encode())
        b = a + len(match.encode())
        end_line = end_line or line
        ls, le = nl[line - 1], nl[end_line - 1]
        if style == "gitleaks":
            start = a + 1 if line == 1 else a - ls + 2
            end = b if end_line == 1 else b - le + 1
        else:
            start, end = a - ls + 1, b - le
        return {"File": "f", "StartLine": line, "EndLine": end_line, "StartColumn": start, "EndColumn": end, "Match": match, "Secret": secret, "RuleID": "r"}

    def test_leak_columns(self):
        text = f'é key = "{V}"\nother\nü token = "{V}x"; again\nBEGIN\nbody\nEND\n'
        for style in ("gitleaks", "betterleaks"):
            with tempfile.TemporaryDirectory() as d:
                with open(os.path.join(d, "f"), "w", newline="") as f:
                    f.write(text)
                t = run_scanners.Texts(d).load("f")
                # line 1 and a later line: the secret inside the match, columns in bytes
                for line, match, secret in ((1, f'key = "{V}"', V), (3, f'token = "{V}x"', V + "x")):
                    a, b = run_scanners.locate_leak(t, self.leak(style, text, match, secret, line), style)
                    self.assertEqual(t.buf[a:b].decode(), secret, (style, line))
                    place = t.lines.place(t.cp(a), t.cp(b))
                    self.assertEqual(place[0], line)
                    self.assertEqual(text.split("\n")[line - 1][place[1]:place[3]], secret)
                # several lines
                a, b = run_scanners.locate_leak(t, self.leak(style, text, "BEGIN\nbody\nEND", "BEGIN\nbody\nEND", 4, 6), style)
                self.assertEqual(t.lines.place(t.cp(a), t.cp(b)), (4, 0, 6, 3))
                # columns that are off: the match is found in its line
                f = self.leak(style, text, f'token = "{V}x"', V + "x", 3)
                f["StartColumn"] += 3
                a, b = run_scanners.locate_leak(t, f, style)
                self.assertEqual(t.buf[a:b].decode(), V + "x")
                # a match that is not in the file (decoded content): the columns are kept, inside the line
                f = self.leak(style, text, f'token = "{V}x"', V + "x", 3)
                f["Match"], f["Secret"] = "decoded text", "decoded"
                a, b = run_scanners.locate_leak(t, f, style)
                self.assertEqual(t.lines.place(t.cp(a), t.cp(b))[0], 3)
                # a line outside the file
                self.assertIsNone(run_scanners.locate_leak(t, dict(f, StartLine=99, EndLine=99), style))

    def test_leaks_to_flags_guard(self):
        with tempfile.TemporaryDirectory() as d:
            os.makedirs(os.path.join(d, "data", "bbbbbbbb", "src"))
            text = f'key = "{V}"\n'
            with open(os.path.join(d, "data", "bbbbbbbb", "src", "f.py"), "w") as f:
                f.write(text)
            val = common.Split("validation", SPLITS)
            corpus = common.Corpus("creddata:validation", d, ["data/bbbbbbbb"], ["data/bbbbbbbb/src/f.py"], val.check)
            finding = dict(self.leak("betterleaks", text, f'key = "{V}"', V, 1), File="/in/data/bbbbbbbb/src/f.py")
            flags = list(run_scanners.leaks_to_flags([finding], corpus, "betterleaks", strip="/in/"))
            self.assertEqual(flags, [{"repo": "bbbbbbbb", "file": "data/bbbbbbbb/src/f.py", "line": 1, "start": 7, "end": 7 + len(V), "rule": "r"}])
            with self.assertRaises(common.Refused):
                list(run_scanners.leaks_to_flags([dict(finding, File="/in/data/dddddddd/src/f.py")], corpus, "betterleaks", strip="/in/"))
            # a corpus cannot list a file outside its roots or its split
            with self.assertRaises(common.Refused):
                common.Corpus("creddata:validation", d, ["data/bbbbbbbb"], ["data/cccccccc/src/f.py"], val.check)
            with self.assertRaises(common.Refused):
                common.Corpus("creddata:validation", d, ["data/dddddddd"], ["data/dddddddd/src/f.py"], val.check)


class Issues(unittest.TestCase):
    def test_locate_and_replace(self):
        text = f"first {V}\nagain {V} and {V}z\n"
        self.assertEqual(issues.locate(text, V), (6, 6 + len(V), 3))
        self.assertIsNone(issues.locate(text, "absent"))
        self.assertIsNone(issues.locate(text, ""))
        new = "n" * len(V)
        out = issues.replace_all(text, {V: new, V + "z": "m" * (len(V) + 1)})
        # every occurrence; the longer string first, so the shorter one's replacement stands inside it
        self.assertEqual(out, f"first {new}\nagain {new} and {new}m\n")
        self.assertEqual(len(out), len(text))
        # a replacement of another length, or none, leaves the value as published
        self.assertEqual(issues.replace_all(text, {V: "short", V + "z": None}), text)
        self.assertEqual(len(issues.text_id(text)), 16)

    def setup_texts(self):
        t1 = f"token = {V}\nother = {V}\nhost = example\n"
        t2 = "nothing here\nat all\n"
        self.texts = {"1" * 16: common.Lines(t1), "2" * 16: common.Lines(t2)}
        # rows: a positive at line 1 of text 1 (first occurrence), a negative on line 3, a negative in text 2
        self.cands = [{"row": 0, "text": "1" * 16, "label": 1, "start": 8, "end": 8 + len(V)},
                      {"row": 1, "text": "1" * 16, "label": 0, "start": t1.index("example"), "end": t1.index("example") + 7},
                      {"row": 2, "text": "2" * 16, "label": 0, "start": 0, "end": 7},
                      {"row": 3, "text": "2" * 16, "label": 1}]  # a candidate that was not found in its text

    def fl(self, text, line, start=None, end=None, **kw):
        return common.flag("issues", f"texts/{text * 16}.txt", line, start=start, end=end, **kw)

    def test_scanner(self):
        self.setup_texts()
        run = {"system": "s", "unit": "span"}
        flags = [self.fl("1", 1, 8, 8 + len(V)), self.fl("1", 2, 8, 8 + len(V)), self.fl("2", 1, 0, 3)]
        out = issues.score_system(self.cands, run, flags, self.texts.get)
        c = out["candidate"]
        # the positive is hit, the negative of text 2 is hit, the second occurrence is no candidate
        self.assertEqual((c["tp"], c["fp"], c["fn"], c["tn"]), (1, 1, 1, 1))
        e = out["end_to_end"]
        self.assertEqual((e["flags"], e["flags_on_a_positive"], e["flags_on_a_candidate"]), (3, 1, 2))
        self.assertAlmostEqual(e["precision"], 1 / 3)
        self.assertAlmostEqual(e["precision_over_flags_on_a_candidate"], 1 / 2)
        self.assertAlmostEqual(e["recall"], 1 / 2)
        # a line tool: its flag is the whole line
        out = issues.score_system(self.cands, {"system": "l", "unit": "line"}, [self.fl("1", 3)], self.texts.get)
        self.assertEqual((out["candidate"]["tp"], out["candidate"]["fp"]), (0, 1))

    def test_judge_of_the_dataset_candidates(self):
        self.setup_texts()
        run = {"system": "m", "unit": "candidate", "judges": "dataset-candidates", "threshold": 0.5}
        flags = [self.fl("1", 1, 8, 20, score=0.9, id=0), self.fl("1", 3, 7, 14, score=0.6, id=1), self.fl("2", 1, 0, 7, score=0.2, id=2)]
        out = issues.score_system(self.cands, run, flags, self.texts.get)
        c = out["candidate"]
        # row 3 has no state (not found in its text): a miss
        self.assertEqual((c["tp"], c["fp"], c["fn"], c["tn"], c["judged"]), (1, 1, 1, 1, 3))
        self.assertNotIn("end_to_end", out)
        self.assertIsNotNone(c["ece"])
        with self.assertRaises(common.Refused):
            issues.score_system(self.cands, dict(run, threshold=None), flags, self.texts.get)


class CleanText(unittest.TestCase):
    def test_share_of_files(self):
        fl = lambda path, **kw: common.flag("clean", path, 1, start=0, end=4, **kw)
        flags = [fl("secretgate/src/a.ts", rule="r"), fl("secretgate/src/a.ts", rule="r"), fl("skycmd/docs/b.md", rule="k")]
        res, accepted, scored = clean_text.share_of(flags, {"system": "generator", "unit": "candidate"}, 200)
        # two files of 200 hold a candidate, whatever their number of candidates
        self.assertEqual((res["files_with_an_accepted_flag"], res["share"], res["accepted_flags"], scored), (2, 0.01, 3, False))
        self.assertEqual(res["by_rule"], {"r": 2, "k": 1})
        # a model accepts at its threshold: one file left
        flags = [fl("secretgate/src/a.ts", score=0.2, id=0), fl("secretgate/src/a.ts", score=0.4, id=1), fl("skycmd/docs/b.md", score=0.9, id=2)]
        res, accepted, scored = clean_text.share_of(flags, {"system": "model", "unit": "candidate", "threshold": 0.5}, 200)
        self.assertEqual((res["files_with_an_accepted_flag"], res["share"], [x["id"] for x in accepted], scored), (1, 0.005, [2], True))
        with self.assertRaises(common.Refused):
            clean_text.share_of(flags, {"system": "model", "unit": "candidate"}, 200)


if __name__ == "__main__":
    unittest.main()
