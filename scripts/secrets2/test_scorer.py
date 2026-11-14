#!/usr/bin/env python3
"""Tests of scripts/secrets2/scorer.py on hand-made meta rows and flags:
python3 -I scripts/secrets2/test_scorer.py

Every value below is made of repeated letters assembled in code; none has the shape of a key.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402
import scorer  # noqa: E402

A, B = "aaaaaaaa", "bbbbbbbb"
V = "v" * 8  # an invented value


def row(i, repo, name, line, truth, start=None, end=None, category="Password", line_end=None):
    return {"id": str(i), "repo": repo, "file": f"data/{repo}/{name}", "line_start": line, "line_end": line_end or line,
            "truth": truth, "value_start": start, "value_end": end, "category": category}


def lab_of(rows, texts, repos=(A, B), n_files=4, largest=None):
    return scorer.Labelled(rows, list(repos), lambda p: texts.get(p), n_files, largest)


def fl(repo, name, line, start=None, end=None, **kw):
    return common.flag(repo, f"data/{repo}/{name}", line, start=start, end=end, **kw)


RUN_C = {"system": "c", "unit": "candidate"}
RUN_S = {"system": "s", "unit": "span"}
RUN_L = {"system": "l", "unit": "line"}


class Meta(unittest.TestCase):
    def test_parse_and_guard(self):
        head = ",".join(scorer.HEADER)
        text = f'{head}\n1,ffffffff,GitHub,{A},data/{A}/src/ffffffff.py,3,3,T,5,13,,,Password\n2,ffffffff,GitHub,{A},data/{A}/src/ffffffff.py,4,4,F,-1,-1,,,"Token:UUID"\n'
        rows = scorer.parse_meta(text, A)
        self.assertEqual([(r["line_start"], r["truth"], r["value_start"], r["category"]) for r in rows],
                         [(3, "T", 5, "Password"), (4, "F", -1, "Token:UUID")])
        with self.assertRaises(common.Refused):
            scorer.parse_meta(text, B)
        with self.assertRaises(common.Refused):
            scorer.parse_meta("Id,FileID\n", A)
        bad = text.replace(",T,5,13", ",Q,5,13")
        with self.assertRaises(common.Refused):
            scorer.parse_meta(bad, A)

    def test_row_spans(self):
        lines = common.Lines("key = " + V + "\r\nsecond\nthird")
        self.assertEqual(scorer.row_span(row(1, A, "f", 1, "T", 6, 14), lines), (6, 14, False))
        # no ValueStart, empty or -1: the whole line
        self.assertEqual(scorer.row_span(row(1, A, "f", 1, "F"), lines), (0, 14, True))
        self.assertEqual(scorer.row_span(row(1, A, "f", 1, "F", -1, -1), lines), (0, 14, True))
        # a start and no end: to the end of the line
        self.assertEqual(scorer.row_span(row(1, A, "f", 1, "T", 6, None), lines), (6, 14, False))
        # several lines: one span from the start on LineStart to the end on LineEnd
        self.assertEqual(scorer.row_span(row(1, A, "f", 1, "T", 6, 3, line_end=3), lines), (6, 26, False))
        # outside the file, or empty
        self.assertIsNone(scorer.row_span(row(1, A, "f", 9, "T", 0, 3), lines))
        self.assertIsNone(scorer.row_span(row(1, A, "f", 2, "T", 6, 6), lines))

    def test_scope_and_categories(self):
        self.assertEqual(scorer.scope_of(f"data/{A}/test/src/ffffffff.py"), "test")
        self.assertEqual(scorer.scope_of(f"data/{A}/src/conf/ffffffff.py"), "src")
        self.assertEqual(scorer.scope_of(f"data/{A}/_/ffffffff.py"), "other")
        self.assertEqual(scorer.categories_of("Token:UUID"), {"Token", "other"})
        self.assertEqual(scorer.categories_of("PEM Private Key"), {"other"})
        self.assertTrue(scorer.is_uuid_only("UUID"))
        self.assertFalse(scorer.is_uuid_only("Token:UUID"))


class Lines(unittest.TestCase):
    def test_creddata_lines(self):
        lines = common.Lines("ab\r\ncd\ref\n")
        self.assertEqual(lines.starts, [0, 4, 7, 10])
        self.assertEqual(lines.ends, [2, 6, 9, 10])
        self.assertEqual(lines.place(4, 6), (2, 0, 2, 2))
        self.assertEqual(lines.place(1, 8), (1, 1, 3, 1))
        self.assertEqual(lines.offset(2, 1), 5)
        self.assertEqual(lines.offset(2, 99), 6)
        self.assertIsNone(lines.offset(9, 0))


def small():
    """Two repositories. In A, file src/f.py:
      1  pw = <V>            T Password, value
      2  token = <V>         F whole line
      3  id = <V>            T UUID (exactly)
      4  key <V> and <V>     T Key at 4..12, X Key at 17..25
      5  -----BEGIN          T multi-line 5..7 (other category)
      6  body
      7  -----END
      8  tok = <V>           T UUID and F Token on the same line
      9  nothing labelled
     10  x = <V>             F Token:UUID (a compound category stays)
    In B, file test/g.py: line 1 T Password value, line 2 X Secret value."""
    f = "\n".join([f"pw = {V}", f"token = {V}", f"id = {V}", f"key {V} and {V}", "-----BEGIN", "body", "-----END",
                   f"tok = {V}", "nothing labelled", f"x = {V}"])
    g = f"pw = {V}\nsecret = {V}\n"
    rows = [
        row(1, A, "src/f.py", 1, "T", 5, 13),
        row(2, A, "src/f.py", 2, "F"),
        row(3, A, "src/f.py", 3, "T", 5, 13, category="UUID"),
        row(4, A, "src/f.py", 4, "T", 4, 12, category="Key"),
        row(5, A, "src/f.py", 4, "X", 17, 25, category="Key"),
        row(6, A, "src/f.py", 5, "T", 0, 8, category="PEM Private Key", line_end=7),
        row(7, A, "src/f.py", 8, "T", 6, 14, category="UUID"),
        row(8, A, "src/f.py", 8, "F", 6, 14, category="Token"),
        row(9, A, "src/f.py", 10, "F", 4, 12, category="Token:UUID"),
        row(10, B, "test/g.py", 1, "T", 5, 13),
        row(11, B, "test/g.py", 2, "X", 9, 17, category="Secret"),
    ]
    texts = {f"data/{A}/src/f.py": f, f"data/{B}/test/g.py": g}
    return rows, texts


class LineLevel(unittest.TestCase):
    def setUp(self):
        self.rows, self.texts = small()
        self.lab = lab_of(self.rows, self.texts, largest=A)

    def score(self, flags, run=RUN_S, threshold=None):
        return scorer.score_system(self.lab, run, flags, threshold)[0]

    def test_universes(self):
        # the units: full has lines 1, 2, 3, 4, 5, 8, 10 of A and 1, 2 of B; gated loses line 3
        # (its only row is a UUID row) and line 8 turns negative (its T row is the UUID one)
        res = self.score([])
        full, gated = res["line"]["full"]["all"], res["line"]["gated"]["all"]
        self.assertEqual((full["fn"], full["tn"]), (6, 3))
        self.assertEqual((gated["fn"], gated["tn"]), (4, 4))
        self.assertEqual(self.lab.t_rows["full"], {A: 5, B: 1})
        self.assertEqual(self.lab.t_rows["gated"], {A: 3, B: 1})

    def test_flags_on_uuid_lines(self):
        res = self.score([fl(A, "src/f.py", 3, 5, 13), fl(A, "src/f.py", 8, 6, 14)])
        full, gated = res["line"]["full"]["all"], res["line"]["gated"]["all"]
        # full: both are true positives. gated: line 3 is not counted, line 8 is a false alarm
        self.assertEqual((full["tp"], full["fp"]), (2, 0))
        self.assertEqual((gated["tp"], gated["fp"]), (0, 1))
        self.assertEqual(res["labels"]["gated"]["E"], 1)
        self.assertEqual(res["labels"]["gated"]["F"], 1)
        self.assertEqual(res["labels"]["full"]["T"], 2)
        # neither is an alarm on an unlabelled line
        self.assertEqual(res["unlabelled"]["flags"], 0)
        # the compound category stays in the gated universe
        res = self.score([fl(A, "src/f.py", 10, 4, 12)])
        self.assertEqual(res["line"]["gated"]["all"]["fp"], 1)
        self.assertEqual(res["line"]["gated"]["by_category"]["Token"]["fp"], 1)
        self.assertEqual(res["line"]["gated"]["by_category"]["other"]["fp"], 1)

    def test_whole_line_and_several_labels(self):
        res = self.score([fl(A, "src/f.py", 2, 8, 16), fl(A, "src/f.py", 4, 10, 20)], RUN_C)
        g = res["line"]["gated"]["all"]
        self.assertEqual((g["tp"], g["fp"], g["fn"], g["tn"]), (1, 1, 3, 3))
        # the candidate of line 2 takes the whole-line F; the one of line 4 overlaps T and X: T
        self.assertEqual(res["labels"]["gated"], {"T": 1, "X": 0, "F": 1, "D": 0, "U": 0, "E": 0})
        c = res["candidate"]["gated"]
        self.assertEqual((c["n"], c["positives"], c["tp"], c["fp"]), (2, 1, 1, 1))
        # a candidate over the X value only is X
        res = self.score([fl(A, "src/f.py", 4, 17, 25)], RUN_C)
        self.assertEqual(res["labels"]["gated"]["X"], 1)
        # the line is positive (it has a T row) and it is flagged: a true positive at line level
        self.assertEqual(res["line"]["gated"]["all"]["tp"], 1)

    def test_dropped_and_unlabelled(self):
        flags = [fl(A, "src/f.py", 1, 0, 2), fl(A, "src/f.py", 9, 0, 7), fl(A, "src/h.py", 1, 0, 3)]
        res = self.score(flags, RUN_C)
        self.assertEqual(res["labels"]["gated"], {"T": 0, "X": 0, "F": 0, "D": 1, "U": 2, "E": 0})
        self.assertEqual(res["candidate"]["gated"]["n"], 0)
        self.assertEqual(res["unlabelled"], {"flags": 2, "lines": 2, "files": 4, "per_1000_files": 500.0})
        # at line level a flag on the line flags it (7.1), matched or not; the other reading is kept apart
        self.assertEqual(res["line"]["gated"]["all"]["tp"], 1)
        self.assertEqual(res["line"]["gated"]["other_readings"]["unmatched_flags_dropped"]["tp"], 0)
        # the span precision counts the unmatched flag of a labelled line, not the unlabelled ones
        self.assertEqual(res["span"]["gated"]["flags_on_labelled_lines"], 1)
        self.assertEqual(res["span"]["gated"]["precision"], 0.0)

    def test_multi_line_rows(self):
        # the unit of the row of lines 5 to 7 is line 5
        first = self.score([fl(A, "src/f.py", 5, 0, 5)])
        inside = self.score([fl(A, "src/f.py", 6, 0, 4)])
        over = self.score([common.flag(A, f"data/{A}/src/f.py", 4, end_line=6, start=20, end=2)])
        self.assertEqual(first["line"]["gated"]["all"]["tp"], 1)
        self.assertEqual(inside["line"]["gated"]["all"]["tp"], 0)
        self.assertEqual(inside["line"]["gated"]["other_readings"]["flag_on_any_line_of_the_row"]["tp"], 1)
        # a flag over lines 4 to 6 flags the units of lines 4 and 5, and covers both T rows and the X row
        self.assertEqual(over["line"]["gated"]["all"]["tp"], 2)
        self.assertEqual(over["labels"]["gated"]["T"], 1)
        # the row is one span: a flag inside line 6 overlaps it, and is not an unlabelled alarm
        self.assertEqual(inside["labels"]["gated"]["T"], 1)
        self.assertEqual(inside["span"]["gated"]["covered"], 1)
        self.assertEqual(inside["unlabelled"]["flags"], 0)

    def test_line_tool(self):
        res = self.score([fl(A, "src/f.py", 1), fl(A, "src/f.py", 2), fl(A, "src/f.py", 9)], RUN_L)
        g = res["line"]["gated"]["all"]
        self.assertEqual((g["tp"], g["fp"]), (1, 1))
        self.assertNotIn("span", res)
        self.assertNotIn("candidate", res)
        self.assertEqual(res["unlabelled"]["flags"], 1)

    def test_rates_scope_and_largest(self):
        flags = [fl(A, "src/f.py", 1, 5, 13), fl(A, "src/f.py", 2, 8, 16), fl(B, "test/g.py", 1, 5, 13), fl(B, "test/g.py", 2, 9, 17)]
        res = self.score(flags)
        g = res["line"]["gated"]
        self.assertEqual((g["all"]["tp"], g["all"]["fp"], g["all"]["fn"], g["all"]["tn"]), (2, 2, 2, 2))
        self.assertAlmostEqual(g["all"]["precision"], 0.5)
        self.assertAlmostEqual(g["all"]["recall"], 0.5)
        self.assertAlmostEqual(g["all"]["f1"], 0.5)
        self.assertAlmostEqual(g["all"]["false_alarm_rate"], 0.5)
        # without the largest repository (A): B alone
        w = g["without_largest"]
        self.assertEqual((w["tp"], w["fp"], w["fn"], w["tn"]), (1, 1, 0, 0))
        self.assertEqual(g["by_scope"]["test"]["tp"], 1)
        self.assertEqual(g["by_scope"]["src"]["tp"], 1)
        self.assertEqual(g["by_category"]["Password"]["tp"], 2)
        self.assertEqual(g["by_category"]["Secret"]["fp"], 1)
        # span level: 2 of the 4 gated T rows are overlapped, 2 of the 4 flags overlap a T row
        s = res["span"]["gated"]
        self.assertEqual((s["t_rows"], s["covered"], s["flags_on_labelled_lines"], s["flags_on_t"]), (4, 2, 4, 2))
        self.assertEqual(res["span"]["full"]["t_rows"], 6)

    def test_threshold(self):
        flags = [fl(A, "src/f.py", 1, 5, 13, score=0.9), fl(A, "src/f.py", 2, 8, 16, score=0.6), fl(B, "test/g.py", 1, 5, 13, score=0.4)]
        res = self.score(flags, {"system": "m", "unit": "candidate", "threshold": 0.5})
        g = res["line"]["gated"]["all"]
        self.assertEqual((g["tp"], g["fp"], g["fn"]), (1, 1, 3))
        c = res["candidate"]["gated"]
        self.assertEqual((c["tp"], c["fp"], c["fn"]), (1, 1, 1))
        # the threshold that would maximise F1 on these three: 0.4 (2 of 2 found, one false alarm)
        self.assertEqual(c["best_f1"]["threshold"], 0.4)
        self.assertAlmostEqual(c["best_f1"]["f1"], 0.8)
        res = self.score(flags, {"system": "m", "unit": "candidate"}, threshold=0.3)
        self.assertEqual(res["line"]["gated"]["all"]["tp"], 2)
        with self.assertRaises(common.Refused):
            self.score(flags, {"system": "m", "unit": "candidate"})


class Curves(unittest.TestCase):
    def test_pr_auc_and_recall_at_false_alarms(self):
        items = [(0.9, True), (0.8, False), (0.7, True), (0.6, True), (0.1, False)]
        # recall steps 1/3 at precisions 1, 2/3 and 3/4
        self.assertAlmostEqual(scorer.pr_auc(items), (1 + 2 / 3 + 3 / 4) / 3)
        self.assertAlmostEqual(scorer.recall_at_fa(items, 0.0), 1 / 3)
        self.assertAlmostEqual(scorer.recall_at_fa(items, 0.5), 1.0)
        # 200 negatives under every positive: 1% lets two of them in
        many = [(0.9, True), (0.8, False), (0.7, True), (0.6, False), (0.5, False), (0.4, True)] + [(0.01, False)] * 197
        self.assertAlmostEqual(scorer.recall_at_fa(many), 2 / 3)
        self.assertIsNone(scorer.pr_auc([(0.5, False)]))
        # ties are one operating point
        self.assertAlmostEqual(scorer.pr_auc([(0.5, True), (0.5, False)]), 0.5)

    def test_best_threshold(self):
        items = [(0.9, True), (0.8, False), (0.7, True), (0.6, True), (0.1, False)]
        t, f = scorer.best_f1_threshold(items)
        self.assertEqual(t, 0.6)
        self.assertAlmostEqual(f, 2 * 3 / (2 * 3 + 1))
        # equal F1 at two thresholds: the higher one
        t, f = scorer.best_f1_threshold([(0.9, True), (0.8, False), (0.7, False), (0.6, True)])
        self.assertEqual(t, 0.9)
        self.assertAlmostEqual(f, 2 / 3)

    def test_ece(self):
        # every answer in the bin [0.8, 0.867): confidence 0.8, 3 of 4 right
        items = [(0.8, True), (0.8, True), (0.8, True), (0.8, False)]
        self.assertAlmostEqual(scorer.ece(items), 0.05)
        # the answer "no" at p = 0.2 has confidence 0.8 too
        items = [(0.2, False), (0.2, False), (0.2, False), (0.2, True)]
        self.assertAlmostEqual(scorer.ece(items), 0.05)
        self.assertAlmostEqual(scorer.ece_positive(items), 0.05)
        # two bins, weighted by their sizes
        items = [(1.0, True)] * 2 + [(0.6, True), (0.6, False)]
        self.assertAlmostEqual(scorer.ece(items), 0.5 * 0.0 + 0.5 * 0.1)
        self.assertIsNone(scorer.ece([]))
        self.assertEqual(len(scorer.ece_bins(items)), 15)


class Aggregates(unittest.TestCase):
    def many(self):
        """Repository A: 30 T rows and 10 F rows, one per line. B: 20 T rows. C: 5 T rows."""
        rows, texts, i = [], {}, 0
        for repo, n_t, n_f in ((A, 30, 10), (B, 20, 0), ("cccccccc", 5, 0)):
            text = []
            for k in range(n_t + n_f):
                i += 1
                text.append(f"pw = {V}")
                rows.append(row(i, repo, "src/f.py", k + 1, "T" if k < n_t else "F", 5, 13))
            texts[f"data/{repo}/src/f.py"] = "\n".join(text)
        return rows, texts

    def test_macro_and_bootstrap(self):
        rows, texts = self.many()
        repos = [A, B, "cccccccc"]
        lab = lab_of(rows, texts, repos, n_files=3, largest=A)
        # system x finds every T of A and nothing else; system y finds half of each repository
        x = [fl(A, "src/f.py", k + 1, 5, 13) for k in range(30)]
        y = [fl(r, "src/f.py", k + 1, 5, 13) for r, n in ((A, 15), (B, 10), ("cccccccc", 2))for k in range(n)]
        out = scorer.score_all(lab, {"x": (RUN_S, x), "y": (RUN_S, y)}, reference="x")
        gx = out["systems"]["x"]["line"]["gated"]
        # pooled: 30 of 55; macro over A and B only (C has 5 T rows): (1 + 0) / 2
        self.assertAlmostEqual(gx["all"]["recall"], 30 / 55)
        self.assertEqual(gx["macro_f1"], {"value": 0.5, "repositories": 2})
        self.assertEqual(gx["without_largest"]["tp"], 0)
        self.assertEqual(out["repositories_with_20_t_rows"], {"gated": 2, "full": 2})
        self.assertEqual(out["largest_repository"]["t_rows"]["gated"], 30)
        d = out["bootstrap"]["f1_difference"]["y"]["gated"]
        fx = 2 * 30 / (2 * 30 + 25)
        fy = 2 * 27 / (2 * 27 + 28)
        self.assertAlmostEqual(d["all"]["value"], fx - fy)
        self.assertLessEqual(d["all"]["lo"], d["all"]["value"])
        self.assertGreaterEqual(d["all"]["hi"], d["all"]["value"])
        # without A, x finds nothing and y still finds half: the difference is negative in every resample
        self.assertLess(d["without_largest"]["hi"], 0)
        # the resamples are drawn once: a second run gives the same interval
        again = scorer.score_all(lab, {"x": (RUN_S, x), "y": (RUN_S, y)}, reference="x")
        self.assertEqual(again["bootstrap"], out["bootstrap"])
        self.assertEqual(out["bootstrap"]["resamples"], 1000)

    def test_ece_interval(self):
        rows, texts = self.many()
        repos = [A, B, "cccccccc"]
        lab = lab_of(rows, texts, repos, n_files=3, largest=A)
        flags = [fl(A, "src/f.py", k + 1, 5, 13, score=0.8) for k in range(40)]
        out = scorer.score_all(lab, {"m": ({"system": "m", "unit": "candidate", "threshold": 0.5}, flags)})
        c = out["systems"]["m"]["candidate"]["gated"]
        # 40 answers "yes" at 0.8, 30 right: |0.8 - 0.75|
        self.assertAlmostEqual(c["ece"], 0.05)
        self.assertAlmostEqual(c["ece_positive"], 0.05)
        e = out["bootstrap"]["ece"]["m"]["gated"]
        self.assertAlmostEqual(e["value"], 0.05)
        # every item sits in one repository: a resample holds it k times or not at all
        self.assertIn(e["lo"], (None, 0.05, e["lo"]))
        self.assertEqual(out["bootstrap"]["f1_difference"], {})


if __name__ == "__main__":
    unittest.main()
