#!/usr/bin/env python3
"""Tests of scripts/secrets2/union.py (s6-secrets v3) on hand-made rows and flags:
python3 -I scripts/secrets2/test_union.py

The rows and texts are those of test_scorer.small(); every value is made of repeated letters.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402
import scorer  # noqa: E402
import union  # noqa: E402
from test_scorer import A, B, fl, lab_of, small  # noqa: E402

DEFAULT_RUN = {"system": "secretgate-default", "unit": "span", "corpus": "creddata:validation", "version": "d"}
MODEL_RUN = {"system": "model", "unit": "candidate", "scored": True, "threshold": 0.5, "corpus": "creddata:validation", "version": "m"}


def default_flags():
    # line 1 of A (a T row) and line 2 of A (a whole-line F row)
    return [fl(A, "src/f.py", 1, 5, 13), fl(A, "src/f.py", 2, 8, 16)]


def model_flags():
    return [fl(A, "src/f.py", 1, 5, 13, score=0.1, id=0), fl(A, "src/f.py", 4, 4, 12, score=0.9, id=1),
            fl(A, "src/f.py", 5, 0, 8, score=0.3, id=2), fl(A, "src/f.py", 8, 6, 14, score=0.2, id=3),
            fl(B, "test/g.py", 1, 5, 13, score=0.5, id=4), fl(B, "test/g.py", 2, 9, 17, score=0.6, id=5)]


class CredData(unittest.TestCase):
    """Gated units: A1 T, A2 F, A4 T, A5 T, A8 F, A10 F, B1 T, B2 X."""

    def setUp(self):
        rows, texts = small()
        self.lab = lab_of(rows, texts, largest=A)

    def union_score(self, tau):
        run = union.union_run("union", DEFAULT_RUN, "model", MODEL_RUN, tau)
        return scorer.score_system(self.lab, run, union.union_flags(default_flags(), model_flags(), tau))[0]

    def test_flags(self):
        u = union.union_flags(default_flags(), model_flags(), 0.5)
        self.assertEqual(len(u), 2 + 3)
        self.assertFalse(any("score" in x for x in u))
        self.assertEqual([x["by"] for x in u], ["default"] * 2 + ["verifier"] * 3)
        run = union.union_run("union", DEFAULT_RUN, "model", MODEL_RUN, 0.5)
        self.assertEqual((run["unit"], run["scored"], run["union"]["threshold"]), ("line", False, 0.5))

    def test_line_level(self):
        g = self.union_score(0.5)["line"]["gated"]
        # A1 and A2 by the default engine, A4, B1, B2 by the model
        self.assertEqual((g["all"]["tp"], g["all"]["fp"], g["all"]["fn"], g["all"]["tn"]), (3, 2, 1, 2))
        # without the largest (A): B1 and B2
        self.assertEqual((g["without_largest"]["tp"], g["without_largest"]["fp"]), (1, 1))
        # the model alone at the same threshold misses A1 and does not flag A2
        alone = scorer.score_system(self.lab, MODEL_RUN, model_flags(), 0.5)[0]["line"]["gated"]["all"]
        self.assertEqual((alone["tp"], alone["fp"], alone["fn"]), (2, 1, 2))
        # the union never loses a line of either part
        for t in (0.1, 0.3, 0.5, 0.95):
            a = self.union_score(t)["line"]["gated"]["all"]
            m = scorer.score_system(self.lab, MODEL_RUN, model_flags(), t)[0]["line"]["gated"]["all"]
            self.assertGreaterEqual(a["tp"], m["tp"])
            self.assertGreaterEqual(a["tp"] + a["fp"], 2)

    def test_unit_table_agrees_with_the_scorer(self):
        for u in scorer.UNIVERSES:
            t = union.unit_table(self.lab, default_flags(), model_flags(), u)
            for tau in (0.05, 0.1, 0.2, 0.3, 0.5, 0.6, 0.9, 0.99):
                got = union.counts_at(t, tau)
                want = self.union_score(tau)["line"][u]["all"]
                self.assertEqual(got, (want["tp"], want["fp"], want["fn"], want["tn"]), (u, tau))
                alone = scorer.score_system(self.lab, MODEL_RUN, model_flags(), tau)[0]["line"][u]["all"]
                self.assertEqual(union.counts_at(t, tau, with_default=False), (alone["tp"], alone["fp"], alone["fn"], alone["tn"]))
        labels = {e[1] for e in union.unit_table(self.lab, [], [], "gated")}
        self.assertEqual(labels, {"T", "X", "F"})

    def test_threshold(self):
        t = union.unit_table(self.lab, default_flags(), model_flags(), "gated")
        scores = [x["score"] for x in model_flags()]
        # 0.9: 4/7; 0.6: 4/8; 0.5: 6/9; 0.3: 8/10 (best); 0.2 and 0.1: 8/11
        tau, c = union.choose_threshold(t, scores)
        self.assertEqual((tau, c), (0.3, (4, 2, 0, 2)))
        # of equal values the highest: 0.28 gives the same lines as 0.3
        self.assertEqual(union.choose_threshold(t, scores + [0.28, 0.12])[0], 0.3)
        # a score that flags nothing new above the best keeps the best where it was
        self.assertEqual(union.choose_threshold(t, scores + [0.95])[0], 0.3)
        # the model alone: 0.3 flags A4 A5 B1 B2: 6/8; 0.2 adds A8: 6/9; 0.1 adds A1: 8/10
        self.assertEqual(union.choose_threshold(t, scores, with_default=False), (0.1, (4, 2, 0, 2)))
        with self.assertRaises(common.Refused):
            union.choose_threshold(t, [])

    def test_added_lines(self):
        a = union.added_lines(self.lab, default_flags(), model_flags(), 0.3, 0.5)["gated"]
        # at 0.3 against the model alone at 0.5: A1 (T), A2 (F) by the default engine, A5 (T) by the model
        self.assertEqual(a["union"]["added"], {"T": 2, "X": 0, "F": 1})
        self.assertEqual(a["union"]["dropped"], {"T": 0, "X": 0, "F": 0})
        self.assertEqual(a["union-tau-v2"]["added"], {"T": 1, "X": 0, "F": 1})
        b = union.added_lines(self.lab, default_flags(), model_flags(), 0.95, 0.5)["gated"]
        # a union threshold above the model's drops its lines: A4 (T, 0.9), B1 (T), B2 (X)
        self.assertEqual(b["union"]["dropped"], {"T": 2, "X": 1, "F": 0})

    def test_refusals(self):
        with self.assertRaises(common.Refused):
            union.union_run("u", {**DEFAULT_RUN, "scored": True}, "model", MODEL_RUN, 0.5)
        with self.assertRaises(common.Refused):
            union.union_run("u", DEFAULT_RUN, "model", {**MODEL_RUN, "corpus": "creddata:test"}, 0.5)


class Issues(unittest.TestCase):
    def test_candidates(self):
        texts = {"t1": "aaaaa bbbb ccccc\n", "t2": "dddd eeee\n"}
        lines_of = lambda tid: common.Lines(texts[tid])
        cands = [{"row": 0, "text": "t1", "label": 1, "start": 0, "end": 5},
                 {"row": 1, "text": "t1", "label": 0, "start": 11, "end": 16},
                 {"row": 2, "text": "t2", "label": 1, "start": 0, "end": 4},
                 {"row": 3, "text": "t2", "label": 0}]
        drun = {"system": "secretgate-default", "unit": "span"}
        dflags = [{"repo": "issues", "file": "texts/t1.txt", "line": 1, "start": 2, "end": 4}]
        vrun = {"system": "model", "scored": True, "judges": "dataset-candidates"}
        vflags = [{"id": 0, "score": 0.1}, {"id": 1, "score": 0.9}, {"id": 2, "score": 0.4}, {"id": 3, "score": 0.8}]
        r = union.issues_union(cands, drun, dflags, vrun, vflags, 0.5, lines_of)
        c = r["candidate"]
        # row 0 by the default engine (TP), rows 1 and 3 by the model (FP), row 2 missed
        self.assertEqual((c["tp"], c["fp"], c["fn"], c["tn"]), (1, 2, 1, 0))
        self.assertEqual(r["flagged_by"]["default only"], {"positive": 1, "negative": 0})
        self.assertEqual(r["flagged_by"]["verifier only"], {"positive": 0, "negative": 2})
        r = union.issues_union(cands, drun, dflags, vrun, vflags, 0.3, lines_of)
        self.assertEqual(r["candidate"]["tp"], 2)
        with self.assertRaises(common.Refused):
            union.issues_union(cands, drun, dflags, {**vrun, "judges": "generator-candidates"}, vflags, 0.5, lines_of)


class Clean(unittest.TestCase):
    def test_files(self):
        d = [{"file": "r/a"}, {"file": "r/a"}, {"file": "r/b"}]
        v = [{"file": "r/b", "score": 0.9}, {"file": "r/c", "score": 0.7}, {"file": "r/d", "score": 0.1}]
        r = union.clean_union(d, v, 0.5, 100)
        self.assertEqual((r["files_with_an_accepted_flag"], r["share"]), (3, 0.03))
        self.assertEqual((r["files_default_only"], r["files_verifier_only"], r["files_both"]), (1, 1, 1))
        self.assertEqual(union.clean_union(d, v, 0.05, 100)["files_with_an_accepted_flag"], 4)


if __name__ == "__main__":
    unittest.main()
