#!/usr/bin/env python3
"""Tests of the systems that judge candidates, on hand-made tables and dumps:
  python3 -I scripts/secrets2/test_candidates.py                        the model's conversion
  <data>/eval/envs/lr/bin/python -I scripts/secrets2/test_candidates.py  the same, and the LR
(the LR tests need scikit-learn and are skipped without it).

States are short invented texts in the layout of sky_infer::secret_cand; values are repeated
letters assembled in code.
"""
import importlib.util
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import candidates  # noqa: E402
import common  # noqa: E402
import run_lr  # noqa: E402
import run_model  # noqa: E402

HAS_SKLEARN = importlib.util.find_spec("sklearn") is not None
OPTS = ["false: no, the statement does not hold", "true: yes, the statement holds"]


def state(i, real):
    value = ("q" if real else "z") * (8 + i % 5)
    return (f"rule generic-keyword\npath src .py\nkey {'token' if real else 'name'}_{i}\nvalue {value}\n"
            f"len {len(value)} ent 0.00 cpt 1.00 dig 0 up 0 low {len(value)} oth 0 word 0 gate 0 ph 0 stop 0 allow 0\n"
            f"line x = <V>\nprev\nnext")


def table_row(i, real, source="creddata", split="val", kind="cred"):
    f = {k: 0 for k in candidates.FEATURES}
    f.update({"len": 8 + i % 5, "lower": 8 + i % 5, "chars_per_token": 1.0, "entropy": 3.0 if real else 1.0, "tokens": 8})
    return {"id": f"{kind}:{split}:{i}", "source": source, "split": split, "label": "T" if real else "F", "real": int(real), "weak": False,
            "features": f, "state": state(i, real)}


def dump_line(row, p, t="noul"):
    if t == "noul":
        return {"src": "creddata:real", "t": "noul", "ins": run_model.Q_REAL, "opts": OPTS, "state": row["state"], "wav": None, "probs": [1 - p, p], "exit": 3}
    return {"src": "creddata:kind", "t": "choice", "ins": run_model.Q_KIND, "opts": list(run_model.KINDS), "state": row["state"], "wav": None, "probs": p, "exit": 3}


class Dump(unittest.TestCase):
    def test_scores_and_kind(self):
        rows = [table_row(0, True), table_row(1, False), table_row(2, False)]
        rows[2]["label"] = "X"
        dump = [dump_line(rows[0], 0.9), dump_line(rows[0], [0.7, 0.2, 0.1], "choice"),
                dump_line(rows[1], 0.2), dump_line(rows[1], [0.5, 0.3, 0.2], "choice"),
                dump_line(rows[2], 0.4)]  # the third candidate has no kind question
        scores, kind = run_model.dump_to_scores(rows, dump)
        self.assertEqual([i for i, _ in scores], ["cred:val:0", "cred:val:1", "cred:val:2"])
        self.assertEqual([round(p, 6) for _, p in scores], [0.9, 0.2, 0.4])
        # kind: the first is right (T), the second answers "real" for an F
        self.assertEqual(kind, {"n": 2, "correct": 1})

    def test_a_dump_of_other_records_is_refused(self):
        rows = [table_row(0, True), table_row(1, False)]
        good = [dump_line(rows[0], 0.9), dump_line(rows[1], 0.2)]
        for bad in (good[:1],                                                   # too short
                    good + [dump_line(rows[1], 0.2)],                           # too long
                    [dump_line(rows[1], 0.9), dump_line(rows[0], 0.2)],         # another order
                    [dict(good[0], ins="Another question?"), good[1]],
                    [dict(good[0], opts=list(reversed(OPTS))), good[1]],
                    [dict(good[0], probs=[0.5]), good[1]]):
            with self.assertRaises(common.Refused):
                run_model.dump_to_scores(rows, bad)

    def test_flags_take_the_place_of_the_generator_candidate(self):
        gen = [common.flag("bbbbbbbb", "data/bbbbbbbb/src/f.py", 3, 3, 4, 12, rule="generic-keyword", id=0),
               common.flag("bbbbbbbb", "data/bbbbbbbb/src/f.py", 9, 9, 0, 8, rule="jwt", id=1)]
        flags = list(candidates.to_flags("creddata:validation", [("cred:val:1", 0.75)], gen))
        self.assertEqual(flags, [{**gen[1], "score": 0.75}])
        for bad in ([("cred:val:7", 0.5)], [("issue:val:1", 0.5)], [("cred:val", 0.5)], [("cred:val:x", 0.5)]):
            with self.assertRaises(common.Refused):
                list(candidates.to_flags("creddata:validation", bad, gen))
        rows = [{"row": 4, "text": "1" * 16, "label": 1, "start": 0, "end": 3}]
        flags = list(candidates.to_flags("issues:validation", [("issue:val:4", 0.25)], issue_candidates=rows))
        self.assertEqual(flags, [{"repo": "issues", "file": "texts/" + "1" * 16 + ".txt", "line": 1, "score": 0.25, "id": 4}])
        flags = list(candidates.to_flags("issues:validation:obfuscated", [("issue:val:4", 0.25)], issue_candidates=rows))
        self.assertTrue(flags[0]["file"].startswith("texts-obfuscated/"))
        with self.assertRaises(common.Refused):
            list(candidates.to_flags("issues:validation", [("issue:val:5", 0.25)], issue_candidates=rows))

    def test_latency_and_exits(self):
        ev = lambda us, early: {"n_params": 445345, "sources": {"all": {"n": 30, "median_us": 9999.0, "early_exit_rate": 0.0, "ece": 0.0},
                                                               "creddata:real": {"n": 20, "median_us": us, "early_exit_rate": early, "ece": 0.01},
                                                               "creddata:kind": {"n": 10, "median_us": 5 * us, "early_exit_rate": 0.0, "ece": 0.2}}}
        exits = run_model.per_exit([
            {"src": "creddata:real", "t": "noul", "probs": [[0.4, 0.6], [0.1, 0.9]], "target": [0.0, 1.0]},
            {"src": "creddata:real", "t": "noul", "probs": [[0.3, 0.7], [0.8, 0.2]], "target": [1.0, 0.0]},
            {"src": "creddata:kind", "t": "choice", "probs": [[0.3, 0.3, 0.4], [0.8, 0.1, 0.1]], "target": [1.0, 0.0, 0.0]}])
        self.assertEqual([(e["exit"], e["n"], e["accuracy"]) for e in exits], [(1, 2, 0.5), (2, 2, 1.0)])
        with tempfile.TemporaryDirectory() as d:
            model = os.path.join(d, "m.gguf")
            with open(model, "wb") as f:
                f.write(b"GGUF")
            lat = run_model.latency_of(ev(1500.0, 0.0), ev(400.0, 1.0), ev(900.0, 0.25), exits, model, 0.4)
        # per candidate: the real question's records, not the pooled row
        self.assertEqual((lat["median_us_full_depth"], lat["median_us_first_exit"], lat["candidates"]), (1500.0, 400.0, 20))
        self.assertEqual((lat["early_exit_rate_at_stored_thresholds"], lat["bytes"], lat["tau"]), (0.25, 4, 0.4))
        with self.assertRaises(common.Refused):
            run_model.real_rows({"sources": {"all": {"n": 1}}})


@unittest.skipUnless(HAS_SKLEARN, "scikit-learn is not installed in this interpreter")
class Lr(unittest.TestCase):
    def write_tables(self, d):
        import json

        n = {"creddata": 120, "issues": 60, "synth": 40}
        k = 0
        for source in run_lr.MIX:
            for split, count in (("train", n[source]), ("val", 40)):
                with open(os.path.join(d, f"{source}-{split}.jsonl"), "w") as f:
                    for _ in range(count):
                        kind = {"creddata": "cred", "issues": "issue", "synth": "synth"}[source]
                        f.write(json.dumps(table_row(k, k % 3 == 0, source, split, kind)) + "\n")
                        k += 1

    def test_quantisation(self):
        import numpy as np

        q, scale = run_lr.quantise(np.array([0.5, -1.0, 0.004, 0.0, 0.25]))
        self.assertEqual(q.dtype, np.int8)
        self.assertEqual(list(q), [64, -127, 1, 0, 32])
        self.assertAlmostEqual(scale, 1 / 127)
        q, scale = run_lr.quantise(np.zeros(3))
        self.assertEqual((list(q), scale), ([0, 0, 0], 1.0))

    def test_mix_weights(self):
        rows = [{"source": "creddata"}] * 6 + [{"source": "issues"}] * 3 + [{"source": "synth"}]
        w = run_lr.mix_weights(rows)
        self.assertAlmostEqual(w.mean(), 1.0)
        # each source weighs its share of the mix, whatever its count
        self.assertAlmostEqual(w[:6].sum() / w.sum(), 0.45)
        self.assertAlmostEqual(w[6:9].sum() / w.sum(), 0.35)
        self.assertAlmostEqual(w[9:].sum() / w.sum(), 0.20)
        with self.assertRaises(common.Refused):
            run_lr.mix_weights([{"source": "elsewhere"}])

    def test_train_and_score(self):
        with tempfile.TemporaryDirectory() as d:
            self.write_tables(d)
            out = os.path.join(d, "lr")
            report = run_lr.train(d, out, "lr")
            self.assertEqual([a["alpha"] for a in report["alphas"]], list(run_lr.ALPHAS))
            self.assertIn(report["alpha"], run_lr.ALPHAS)
            self.assertEqual((report["int8_weights"], report["int8_bytes"]), (262144, 262144))
            self.assertEqual(report["training_candidates"], {"creddata": 120, "issues": 60, "synth": 40})
            # the two classes differ by their letters: the baseline separates them
            self.assertGreater(report["validation_f1"], 0.95)
            self.assertEqual(report["validation_f1"], max(a["validation_f1"] for a in report["alphas"]))
            model, again = run_lr.load(out, "lr")
            rows = list(candidates.read_table(os.path.join(d, "creddata-val.jsonl")))
            p = model.scores(rows)
            self.assertTrue(all(0.0 <= x <= 1.0 for x in p))
            self.assertGreater(min(x for x, r in zip(p, rows) if r["real"]), max(x for x, r in zip(p, rows) if not r["real"]))
            # the saved file is what is loaded: a changed file is refused
            with open(os.path.join(out, "lr-model.npz"), "ab") as f:
                f.write(b"x")
            with self.assertRaises(common.Refused):
                run_lr.load(out, "lr")

    def test_training_refuses_other_splits(self):
        with tempfile.TemporaryDirectory() as d:
            self.write_tables(d)
            import json

            with open(os.path.join(d, "synth-train.jsonl"), "a") as f:
                f.write(json.dumps(table_row(999, True, "synth", "test", "synth")) + "\n")
            with self.assertRaises(common.Refused):
                run_lr.train(d, os.path.join(d, "lr"), "lr")


if __name__ == "__main__":
    unittest.main()
