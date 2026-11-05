#!/usr/bin/env python3
"""CPU unit tests of the LFM3 fine-tuning experiment, on synthetic data only (no model, no network).

  python3 experiments/lfm3-ft/test_common.py
"""
import importlib.util
import math
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C  # noqa: E402
import gate as G  # noqa: E402
import smoke as SM  # noqa: E402


def load_s1s2():
    spec = importlib.util.spec_from_file_location("s1s2", os.path.join(HERE, "..", "lfm3-s1s2", "s1s2.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


S = load_s1s2()


class SameAsS1S2(unittest.TestCase):
    """common.py copies s1s2.py: same constants, same answers on the same inputs."""

    TEXTS = [
        "so \\boxed{42}.", "\\boxed{\\frac{1}{2}} and then \\boxed{0.5}", "no box, 1,234 apples then 7",
        "\\boxed{x = 3}", "\\boxed{\\text{(B)}}", "\\boxed{90^\\circ}", "\\boxed{\\dfrac{3}{4}}", "\\boxed{", "",
        "\\boxed{\\left( 1, 2 \\right)}", "-5 is it", "\\boxed{50\\%}",
    ]
    GOLDS = ["42", "0.5", "7", "3", "(B)", "90", "\\frac34", "\\frac{3}{4}", "1,234", "(1,2)", "-5", "50"]

    def test_constants(self):
        self.assertEqual(C.PROMPT, S.PROMPT)
        self.assertEqual(C.SEED, S.SEED)
        self.assertEqual(C.ANSWER_MAX, S.ANSWER_MAX)
        self.assertEqual(C.BUDGETS, S.BUDGETS)
        self.assertEqual(C.GREEDY, S.GREEDY)
        self.assertEqual(C.MATH_SAMPLING, S.MATH_SAMPLING)
        self.assertEqual((C.GSM8K, C.MATH500, C.MATH_TRAIN), (S.GSM8K, S.MATH500, S.MATH_TRAIN))

    def test_grading(self):
        for t in self.TEXTS:
            self.assertEqual(C.last_boxed(t), S.last_boxed(t), t)
            for task in ("gsm8k", "math"):
                self.assertEqual(C.extract(task, t), S.extract(task, t), t)
            self.assertEqual(C.strip_math(t), S.strip_math(t), t)
            self.assertEqual(C.as_number(t), S.as_number(t), t)
            for g in self.GOLDS:
                for task in ("gsm8k", "math"):
                    p = C.extract(task, t)
                    self.assertEqual(C.correct(task, p, g), S.correct(task, p, g), (t, g))
                    self.assertEqual(C.grade(task, t, g), (p, S.correct(task, p, g)))

    def test_answer_logprob_and_seed(self):
        pieces = [("The ", -0.1), ("answer ", -0.2), ("is ", -0.3), ("\\box", -1.0), ("ed{4", -0.5), ("2}", float("nan"))]
        self.assertEqual(C.answer_logprob(pieces), S.answer_logprob(pieces))
        self.assertEqual(C.answer_logprob(pieces[:3]), S.answer_logprob(pieces[:3]))
        for qid in ("gsm8k-test-15", "math500-test/algebra/1.json", "x"):
            self.assertEqual(C.qseed(qid), S.SEED + int.from_bytes(qid.encode()[-6:], "big") % 1_000_000)


class Data(unittest.TestCase):
    def test_clean_gsm8k(self):
        sol = "Natalia sold 48/2 = <<48/2=24>>24 clips in May.\nShe sold 48+24 = <<48+24=72>>72 clips.  \n#### 72"
        self.assertEqual(C.clean_gsm8k_solution(sol),
                         "Natalia sold 48/2 = 24 clips in May.\nShe sold 48+24 = 72 clips.\n\\boxed{72}")
        self.assertEqual(C.gsm8k_gold(sol), "72")
        self.assertEqual(C.last_boxed(C.clean_gsm8k_solution(sol)), C.gsm8k_gold(sol))
        with self.assertRaises(ValueError):
            C.clean_gsm8k_solution("no final line")

    def test_norm_and_select(self):
        self.assertEqual(C.norm_question("  What is 2+2?\n"), C.norm_question("what is 2 + 2 ?"))
        rows = [
            {"id": "gsm8k-train-0", "question": "Calib by id"},
            {"id": "gsm8k-train-1", "question": "Same as TEST, really?"},
            {"id": "gsm8k-train-2", "question": "same as calib text"},
            {"id": "math-train-3", "question": "Kept one"},
            {"id": "math-train-4", "question": "An MMLU-Pro question"},
        ]
        calib = [{"id": "gsm8k-train-0", "question": "whatever"}, {"id": "math-train-99", "question": "Same as calib text."}]
        kept, st = C.select_train(rows, calib, ["same as test really", "an mmlu pro question"])
        self.assertEqual([r["id"] for r in kept], ["math-train-3"])
        self.assertEqual(st, {"rows": 5, "calib_id": 1, "same_text": 3, "kept": 1})

    def test_hf_subset(self):
        rows = [{"id": f"g{i}", "task": "gsm8k"} for i in range(30)] + [{"id": f"m{i}", "task": "math"} for i in range(5)]
        sub = C.hf_fallback_subset(rows, per_task=10)
        self.assertEqual(len(sub), 15)
        self.assertEqual(sub, C.hf_fallback_subset(rows, per_task=10))

    def test_choice(self):
        self.assertEqual(C.extract_choice("thus \\boxed{C}"), "C")
        self.assertEqual(C.extract_choice("\\boxed{(J)}"), "J")
        self.assertEqual(C.extract_choice("\\boxed{\\text{B}}"), "B")
        self.assertEqual(C.extract_choice("The answer is (d)."), "D")
        self.assertIsNone(C.extract_choice("\\boxed{42}"))
        self.assertEqual(C.grade("mmlu_pro", "\\boxed{A}", "A"), ("A", True))
        self.assertEqual(C.grade("mmlu_pro", "\\boxed{B}", "A"), ("B", False))
        it = C.mmlu_format({"question_id": 7, "question": "Q?", "options": ["x", "y"], "answer": "B", "category": "law"})
        self.assertEqual(it["id"], "mmlupro-7")
        self.assertIn("(A) x\n(B) y", C.user_prompt(it))


def sample(qid, k, correct=True, forced=False, tokens=100, answer_tokens=20, text="so \\boxed{1}", think="hmm"):
    return {"id": qid, "k": k, "correct": correct, "forced": forced, "tokens": tokens, "answer_tokens": answer_tokens,
            "text": text, "think": think}


class Mix(unittest.TestCase):
    def test_distill_targets(self):
        ss = [
            sample("q1", 0, tokens=300, answer_tokens=30, text="long \\boxed{1}", think="t0"),
            sample("q1", 1, tokens=200, answer_tokens=10, text=" short \\boxed{1}\n", think="t1", forced=True),
            sample("q1", 2, tokens=250, answer_tokens=40, text="mid \\boxed{1}", think="t2"),
            sample("q1", 3, correct=False, tokens=50, answer_tokens=5, text="\\boxed{9}"),
            sample("q2", 0, correct=False),
            sample("q3", 0, text="correct but no box 1"),
            sample("q4", 0, forced=True, tokens=1100, answer_tokens=12, text="\\boxed{2}"),
        ]
        t = C.distill_targets(ss)
        self.assertEqual(set(t), {"q1", "q4"})
        self.assertEqual(t["q1"]["nothink"], "short \\boxed{1}")  # shortest answer, forced allowed
        self.assertEqual(t["q1"]["think"], "t2</think>mid \\boxed{1}")  # shortest unforced trace
        self.assertIsNone(t["q4"]["think"])

    def test_build_mix(self):
        rows = [{"id": "a", "task": "gsm8k", "question": "A?", "solution": "x\n\\boxed{1}"},
                {"id": "b", "task": "math", "question": "B?", "solution": "\\boxed{2}"}]
        ss = [sample("a", 0), sample("zz", 0)]  # zz is not a kept train question
        mix = C.build_mix(rows, ss)
        self.assertEqual([(m["id"], m["kind"], m["thinking"]) for m in mix],
                         [("a", "gold_nothink", False), ("b", "gold_nothink", False),
                          ("a", "distill_nothink", False), ("a", "distill_think", True)])


class FakeTok:
    """Character-level tokenizer with LFM3's template shape."""

    eos_token, eos_token_id = "\x03", 3
    bos_token_id = 2

    def apply_chat_template(self, msgs, tokenize, add_generation_prompt, enable_thinking):
        return "\x02U:" + msgs[0]["content"] + "\nA:" + ("<think>" if enable_thinking else "<think></think>")

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) for c in text]}

    def decode(self, ids, skip_special_tokens=False):
        return "".join(chr(i) for i in ids)


class FakeEngine:
    """Scripted outputs: phase-1 texts by question, phase-2 text constant."""

    def __init__(self, phase1, budget):
        self.tok, self.phase1, self.budget, self.calls = FakeTok(), phase1, budget, []

    def generate(self, reqs):
        self.calls.append(reqs)
        outs = []
        for r in reqs:
            if r.stop:  # phase 1
                text = self.phase1.pop(0)
                if C.THINK_CLOSE in text:
                    text = text[: text.index(C.THINK_CLOSE) + len(C.THINK_CLOSE)]
                    fin = "stop"
                else:
                    text, fin = text[: r.max_tokens], "length"
            else:
                text, fin = "ans \\boxed{1}", "stop"
            ids = [ord(c) for c in text]
            outs.append(C.Out(ids, text, [-0.5] * len(ids) if r.logprobs else None, fin))
        return outs


class Modes(unittest.TestCase):
    def test_s2_budget_forcing(self):
        items = [{"id": "q1", "task": "gsm8k", "question": "Q1"}, {"id": "q2", "task": "gsm8k", "question": "Q2"}]
        eng = FakeEngine(["abc</think>", "x" * 50], budget=10)
        recs = C.run_s2(eng, items, 10)
        closed, forced = recs
        self.assertFalse(closed["forced"])
        self.assertEqual(closed["think"], "abc")
        self.assertEqual(closed["think_tokens"], len("abc</think>"))
        self.assertTrue(forced["forced"])
        self.assertEqual(forced["think"], "x" * 10)
        self.assertEqual(forced["think_tokens"], 10)
        self.assertEqual(forced["tokens"], 10 + len("ans \\boxed{1}"))
        p1, p2 = eng.calls
        self.assertEqual([r.seed for r in p1], [C.qseed("q1"), C.qseed("q2")])
        self.assertEqual([r.seed for r in p2], [C.qseed("q1") + 1, C.qseed("q2") + 1])
        self.assertTrue(eng.tok.decode(p2[1].prompt_ids).endswith("x" * 10 + C.FORCE_CLOSE))
        self.assertTrue(eng.tok.decode(p2[0].prompt_ids).endswith("<think>abc</think>"))
        self.assertTrue(all(r.sampling == C.MATH_SAMPLING for r in p1 + p2))
        self.assertEqual([r.max_tokens for r in p2], [C.ANSWER_MAX] * 2)

    def test_s1_record(self):
        tok = FakeTok()
        o = C.Out([ord(c) for c in "so \\boxed{7}"], "so \\boxed{7}", [-1.0, -2.0, -3.0] + [-0.5] * 9, "stop")
        r = C.s1_record(tok, o)
        self.assertEqual(r["tokens"], 12)
        self.assertEqual(r["min_lp"], -3.0)
        self.assertAlmostEqual(r["answer_lp"], -0.5)
        self.assertTrue(math.isnan(C.s1_record(tok, C.Out([], "", [], "length"))["mean_lp"]))
        # the features of s1s2.py's router read these records unchanged
        r.update(pred="7", correct=True)
        self.assertEqual(len(S.features(r)), 6)

    def test_tokenize_mix(self):
        tok = FakeTok()
        mix = [{"id": "a", "kind": "gold_nothink", "thinking": False, "task": "gsm8k", "question": "Q", "completion": "c"},
               {"id": "b", "kind": "distill_think", "thinking": True, "task": "gsm8k", "question": "Q", "completion": "t</think>c"},
               {"id": "c", "kind": "distill_think", "thinking": True, "task": "math", "question": "Q", "completion": "z" * 5000}]
        rows, dropped = C.tokenize_mix(tok, mix)
        self.assertEqual(dropped, {"distill_think": 1})
        r = rows[0]
        n_prompt = len(tok.apply_chat_template([{"content": C.PROMPT.format(q="Q")}], False, True, False))
        self.assertEqual(r["completion_mask"], [0] * n_prompt + [1, 1])
        self.assertEqual(r["input_ids"][-1], tok.eos_token_id)
        self.assertEqual(tok.decode(rows[1]["input_ids"]).count("<think>"), 1)


def ev(qid, task="gsm8k", correct=True, finish="stop", think=0, tokens=100):
    return {"id": qid, "task": task, "correct": correct, "finish": finish, "think_tokens": think, "tokens": tokens}


def model(acc_s1, len_s1, acc_s2, think_s2, acc_mmlu):
    """Synthetic evaluation of one model with the given rates (1,000 test, 500 MMLU-Pro)."""
    s1 = [ev(f"t{i}", correct=i < acc_s1 * 1000, finish="length" if i >= 1000 - len_s1 * 1000 else "stop") for i in range(1000)]
    s2 = [ev(f"t{i}", correct=i < acc_s2 * 1000, think=think_s2) for i in range(1000)]
    mm = [ev(f"m{i}", task="mmlu_pro", correct=i < acc_mmlu * 500) for i in range(500)]
    return {"s1": s1, "s2": s2, "mmlu": mm}


class Gate(unittest.TestCase):
    def test_improved(self):
        base = model(0.60, 0.10, 0.70, 500, 0.40)
        g = G.gates(base, model(0.63, 0.05, 0.69, 300, 0.38))
        self.assertEqual(g["problems"], [])
        self.assertTrue(g["G1"]["pass"] and g["G2"]["pass"] and g["G3"]["pass"] and g["improved"])
        self.assertEqual(g["verdict"], "LFM3 improved")

    def test_edges(self):
        base = model(0.60, 0.10, 0.70, 500, 0.40)
        g = G.gates(base, model(0.629, 0.05, 0.70, 300, 0.40))  # +0.029 only
        self.assertFalse(g["G1"]["acc"])
        self.assertFalse(g["improved"])
        g = G.gates(base, model(0.70, 0.051, 0.70, 300, 0.40))  # length stops not halved
        self.assertFalse(g["G1"]["length"])
        g = G.gates(base, model(0.70, 0.0, 0.70, 300, 0.378))  # MMLU-Pro -0.022
        self.assertTrue(g["G1"]["pass"])
        self.assertFalse(g["G3"]["pass"])
        self.assertFalse(g["improved"])
        g = G.gates(base, model(0.70, 0.0, 0.689, 301, 0.40))  # G2 fails on both, verdict unaffected
        self.assertFalse(g["G2"]["acc"] or g["G2"]["think"])
        self.assertTrue(g["improved"])

    def test_incomplete(self):
        base = model(0.60, 0.10, 0.70, 500, 0.40)
        ft = model(0.70, 0.0, 0.70, 300, 0.40)
        ft["s1"] = ft["s1"][:-1]
        g = G.gates(base, ft)
        self.assertEqual(g["verdict"], "incomplete")
        self.assertTrue(g["problems"])
        self.assertIn("LFM3", G.report_md(dict(g, router={})))

    def test_router(self):
        summ = {"chosen_budget": 256, "budgets": {"256": {"test": {"all": {
            "s1_acc": 0.60, "s2_acc": 0.70, "router_acc": 0.69, "router_tokens": 300, "s2_tokens": 700}}}}}
        r = G.router(summ)
        self.assertTrue(r["H2"] and r["H3"])
        self.assertFalse(r["H4"])
        self.assertEqual(r["verdict"], "success")
        self.assertIsNone(G.router({"chosen_budget": 256, "budgets": {"256": {"test": None}}}))


class EngineDecision(unittest.TestCase):
    def test_decide(self):
        hf = [{"id": f"q{i}", "ids": list(range(64)), "pred": "1"} for i in range(10)]
        same = {"loaded": True, "records": {"s1": [dict(r) for r in hf], "s2-256": [], "s2-1024": []}}
        self.assertEqual(SM.decide(same, hf)[0], "vllm")
        garbage = {"loaded": True, "records": {"s1": [{"id": r["id"], "ids": [9] * 64, "pred": None} for r in hf],
                                                "s2-256": [], "s2-1024": []}}
        self.assertEqual(SM.decide(garbage, hf)[0], "hf")
        self.assertEqual(SM.decide({"loaded": False}, hf)[0], "hf")
        self.assertEqual(SM.prefix_agreement([1, 2, 3, 4], [1, 2, 9, 4]), 0.5)
        self.assertEqual(SM.smoke_items([{"task": "gsm8k", "id": 1}] * 30 + [{"task": "math", "id": 2}] * 30, 20).count({"task": "math", "id": 2}), 10)


if __name__ == "__main__":
    unittest.main(verbosity=1)
