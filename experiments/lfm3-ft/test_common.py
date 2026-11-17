#!/usr/bin/env python3
"""CPU unit tests of the LFM3 fine-tuning experiment, on synthetic data only (no model, no network).

  python3 experiments/lfm3-ft/test_common.py
"""
import importlib.util
import json
import math
import os
import random
import sys
import tempfile
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


class LocalMode(unittest.TestCase):
    """experiments/lfm3-ft/local.py, the M5 (amendment 1): its samples feed distill_targets."""

    def test_sample_record(self):
        import local as L

        it = {"id": "gsm8k-train-1", "task": "gsm8k", "question": "q", "answer": "4"}
        r = L.sample_record({"text": "so \\boxed{4}", "think": "t", "tokens": 30, "think_tokens": 20, "forced": False,
                             "finish": "stop"}, it, 2, 77, 1.5)
        self.assertEqual((r["answer_tokens"], r["k"], r["seed"], r["mode"], r["correct"]), (10, 2, 77, "s2-1024", True))
        self.assertEqual(C.distill_targets([r])["gsm8k-train-1"]["nothink"], "so \\boxed{4}")

    def test_user_item_keeps_the_prompts(self):
        import local as L

        math_item = {"task": "math", "question": "2+2?"}
        self.assertEqual(L.user_item(math_item)["question"], S.PROMPT.format(q="2+2?"))
        mc = {"task": "mmlu_pro", "question": "Q?", "options": "(A) x"}
        self.assertIn("(A) x", L.user_item(mc)["question"])


class Amendment5(unittest.TestCase):
    """Amendment 5 (local mode): part (a) only under the cap, the pair folder, the gates without G2 and
    the two terms of the cascade cost."""

    def test_cap_per_task(self):
        self.assertEqual((C.SFT_CAP_PER_TASK, C.SFT_EPOCHS, C.SEED), (3000, 1, 1234))
        rows = [{"id": f"g{i}", "task": "gsm8k"} for i in range(3010)]
        rows += [{"id": f"m{i}", "task": "math"} for i in range(3002)]
        cap = C.cap_per_task(rows)
        self.assertEqual(len(cap), 6000)
        self.assertEqual(sorted(r["task"] for r in cap), ["gsm8k"] * 3000 + ["math"] * 3000)
        self.assertEqual(cap, C.cap_per_task(rows))  # deterministic
        g = [r for r in rows if r["task"] == "gsm8k"]
        random.Random(C.SEED).shuffle(g)  # the draw is the seeded shuffle, among the rows given
        self.assertEqual([r["id"] for r in cap if r["task"] == "gsm8k"], [r["id"] for r in g[:3000]])
        self.assertEqual([r["id"] for r in C.cap_per_task(rows, per_task=10) if r["task"] == "gsm8k"],
                         [r["id"] for r in g[:10]])

    def test_cap_after_the_exclusions(self):
        rows = [
            {"id": "gsm8k-train-0", "task": "gsm8k", "question": "Calib by id"},
            {"id": "gsm8k-train-1", "task": "gsm8k", "question": "same as TEST text!"},
            {"id": "gsm8k-train-2", "task": "gsm8k", "question": "kept"},
            {"id": "math-train-3", "task": "math", "question": "kept too"},
            {"id": "math-train-4", "task": "math", "question": "an mmlu pro question"},
        ]
        calib = [{"id": "gsm8k-train-0", "question": "x"}]
        kept, st = C.select_train(rows, calib, ["same as test text", "an mmlu pro question"])
        self.assertEqual([r["id"] for r in kept], ["gsm8k-train-2", "math-train-3"])
        cap = C.cap_per_task(kept, per_task=1)
        self.assertEqual([r["id"] for r in cap], ["gsm8k-train-2", "math-train-3"])  # nothing excluded returns
        self.assertEqual(st, {"rows": 5, "calib_id": 1, "same_text": 2, "kept": 2})

    def test_mix_part_a_only(self):
        rows = [{"id": "a", "task": "gsm8k", "question": "A?", "solution": "x\n\\boxed{1}"},
                {"id": "b", "task": "math", "question": "B?", "solution": "\\boxed{2}"}]
        cap = C.cap_per_task(rows, per_task=1)  # one question per task: both are kept here
        mix = C.build_mix(cap, [])  # amendment 5: no self-distillation samples, no THINK target
        self.assertEqual([(m["id"], m["kind"], m["thinking"], m["completion"]) for m in mix],
                         [("a", "gold_nothink", False, "x\n\\boxed{1}"), ("b", "gold_nothink", False, "\\boxed{2}")])
        self.assertNotIn("distill_think", {m["kind"] for m in C.build_mix(rows, [])})

    def test_build_pair(self):
        import local as L

        def row(qid, mode, tag):
            return {"id": qid, "task": "gsm8k", "mode": mode, "src": tag, "correct": True, "finish": "stop",
                    "think_tokens": 0, "tokens": 10}

        split_ids = {"calib": ["gsm8k-train-1", "math-train-2"], "test": ["gsm8k-test-3", "math500-x"]}
        with tempfile.TemporaryDirectory() as runs:
            data = os.path.join(runs, "data")
            for s, ids in split_ids.items():
                C.write_jsonl(os.path.join(data, f"{s}.jsonl"), [{"id": i} for i in ids])
            for tag, names in (("ft", ["calib.s1.jsonl", "test.s1.jsonl", "mmlu.s1.jsonl"]),
                               ("base", ["calib.s1.jsonl", "calib.s2-256.jsonl", "calib.s2-1024.jsonl",
                                         "test.s1.jsonl", "test.s2-256.jsonl", "test.s2-1024.jsonl", "mmlu.s1.jsonl"])):
                d = os.path.join(runs, "eval", tag)
                os.makedirs(d)
                for n in names:
                    split, mode = n.split(".")[:2]
                    C.write_jsonl(os.path.join(d, n), [row(i, mode, tag) for i in split_ids.get(split, ["mmlupro-1"])])
            meta = L.build_pair(runs, data)
            self.assertEqual(meta["sources"], {"calib.s1.jsonl": "eval/ft", "test.s1.jsonl": "eval/ft",
                                               "calib.s2-256.jsonl": "eval/base", "calib.s2-1024.jsonl": "eval/base",
                                               "test.s2-256.jsonl": "eval/base", "test.s2-1024.jsonl": "eval/base"})
            d = os.path.join(runs, "eval", "pair")
            self.assertEqual(json.load(open(os.path.join(d, "meta.json")))["sources"], meta["sources"])
            self.assertEqual({r["src"] for r in C.read_jsonl(os.path.join(d, "calib.s1.jsonl"))}, {"ft"})
            self.assertEqual({r["src"] for r in C.read_jsonl(os.path.join(d, "test.s1.jsonl"))}, {"ft"})
            self.assertEqual([r["id"] for r in C.read_jsonl(os.path.join(d, "test.s2-1024.jsonl"))], split_ids["test"])
            self.assertEqual({r["src"] for r in C.read_jsonl(os.path.join(d, "test.s2-1024.jsonl"))}, {"base"})
            self.assertEqual(sorted(os.listdir(d)),  # no mmlu in the pair: the router reads calib and test only
                             ["calib.s1.jsonl", "calib.s2-1024.jsonl", "calib.s2-256.jsonl", "meta.json",
                              "test.s1.jsonl", "test.s2-1024.jsonl", "test.s2-256.jsonl"])
            os.remove(os.path.join(runs, "eval", "base", "test.s2-256.jsonl"))
            with self.assertRaises(SystemExit):
                L.build_pair(runs, data)
            self.assertFalse(os.path.exists(os.path.join(d, "meta.json")))  # the stale pair is not left for route

    def test_build_pair_refuses_misaligned_files(self):
        """A question present in one file and absent from another is an error, never a silent drop
        (s1s2.py's route would drop it), as are a duplicated id and a file of the wrong mode."""
        import local as L

        ids = {"calib": [f"gsm8k-train-{i}" for i in range(4)], "test": [f"gsm8k-test-{i}" for i in range(4)]}

        def setup(runs, edit):
            data = os.path.join(runs, "data")
            for s, xs in ids.items():
                C.write_jsonl(os.path.join(data, f"{s}.jsonl"), [{"id": i} for i in xs])
            for split, mode, tag in L.PAIR_FILES:
                rows = [{"id": i, "mode": mode} for i in ids[split]]
                rows = edit(split, mode, tag, rows)
                C.write_jsonl(os.path.join(runs, "eval", tag, f"{split}.{mode}.jsonl"), rows)
            return data

        cases = {
            "ft test s1 missing one": lambda s, m, t, r: r[:-1] if (s, m) == ("test", "s1") else r,
            "base calib s2-256 one extra": lambda s, m, t, r: r + [{"id": "gsm8k-test-0", "mode": m}]
            if (s, m) == ("calib", "s2-256") else r,
            "duplicate in ft calib s1": lambda s, m, t, r: r + r[:1] if (s, m) == ("calib", "s1") else r,
            "calib and test swapped": lambda s, m, t, r: [{"id": i, "mode": m} for i in ids["calib"]]
            if (s, m) == ("test", "s2-1024") else r,
            "wrong mode": lambda s, m, t, r: [dict(x, mode="s2-256") for x in r] if (s, m) == ("test", "s2-1024") else r,
        }
        for label, edit in cases.items():
            with self.subTest(label), tempfile.TemporaryDirectory() as runs:
                data = setup(runs, edit)
                with self.assertRaises(SystemExit):
                    L.build_pair(runs, data)
                self.assertFalse(os.path.exists(os.path.join(runs, "eval", "pair", "meta.json")))
        with tempfile.TemporaryDirectory() as runs:  # the clean case passes
            L.build_pair(runs, setup(runs, lambda s, m, t, r: r))

    def test_gates_g2_not_read(self):
        base, ft = model(0.60, 0.10, 0.70, 500, 0.40), model(0.70, 0.0, 0.70, 300, 0.40)
        ft["s2"] = []  # amendment 5: the fine-tuned model is never run in s2
        g = G.gates(base, ft, read_g2=False)
        self.assertEqual(g["G2"], "not read (System 2 is the base)")
        self.assertEqual(g["problems"], [])  # no s2 comparison: the base's s2 rows are not held against ft
        self.assertTrue(g["G1"]["pass"] and g["G3"]["pass"] and g["improved"])
        self.assertEqual(g["verdict"], "LFM3 improved")
        md = G.report_md(dict(g, router={}))
        self.assertIn("- G2: not read (System 2 is the base)", md)
        g = G.gates(base, ft)  # the default (hf mode) still reads G2 and now misses ft's s2 rows
        self.assertIsNone(g["G2"])
        self.assertEqual(g["verdict"], "incomplete")
        ft["s1"] = ft["s1"][:-1]  # an incomplete evaluation still gives no verdict
        g = G.gates(base, ft, read_g2=False)
        self.assertEqual(g["verdict"], "incomplete")
        self.assertTrue(g["problems"])
        self.assertEqual(g["G2"], "not read (System 2 is the base)")

    def test_pair_mode_flag(self):
        with tempfile.TemporaryDirectory() as runs:
            self.assertFalse(G.pair_mode(runs))  # no pair: G2 read as §6 wrote it (hf mode)
            os.makedirs(os.path.join(runs, "eval", "pair"))
            self.assertFalse(G.pair_mode(runs))
            C.write_jsonl(os.path.join(runs, "eval", "pair", "meta.json"), [{"pair": True}])
            self.assertTrue(G.pair_mode(runs))  # the pair exists: amendment 5, G2 not read

    def test_cascade_cost(self):
        c = G.cascade(300, 900, 0.25)  # ft s1 300 tokens against base s2-1024 900 tokens
        self.assertAlmostEqual(c["one_over_rho"], 1 / 3)
        self.assertAlmostEqual(c["w"], 0.25)
        self.assertAlmostEqual(c["one_over_rho"] + c["w"], 7 / 12)
        self.assertIsNone(G.cascade(0, 900, 0.1))
        self.assertIsNone(G.cascade(300, None, 0.1))
        self.assertIsNone(G.w_share(None, [{}]))
        self.assertIsNone(G.w_share({"chosen_budget": 1024, "budgets": {}}, [{}]))

    def test_w_counts_every_question_sent_to_system2(self):
        """w is the share whose score is under tau, the questions on which System 2 is paid. s1s2.py's
        s1_share also counts as s1 the questions sent to System 2 that gave no answer (the s1 answer is
        kept, but s1 + s2 tokens are paid): 1 - s1_share undercounts w. On the base's own reading
        (runs/lfm3-s1s2) it gives 0.158 for 275 questions of 1,000 really sent on (0.275)."""
        rng = random.Random(7)

        def s1(i):
            lp = -rng.random() * 3
            return {"id": f"q{i}", "task": "gsm8k", "answer_lp": lp, "mean_lp": lp / 2, "min_lp": lp * 2,
                    "tokens": rng.randint(20, 900), "pred": "1", "finish": "stop", "correct": lp > -1.5}

        cal = [s1(i) for i in range(60)]
        score, params = S.fit_logistic([S.features(r) for r in cal], [int(r["correct"]) for r in cal], steps=300)
        params = json.loads(json.dumps(params))  # as route writes it in summary.json
        rebuilt = G.probe_score(params)
        for r in cal:
            self.assertEqual(rebuilt(S.features(r)), score(S.features(r)))
        test = [s1(i) for i in range(100, 140)]
        scores = sorted(score(S.features(r)) for r in test)
        tau = scores[10]  # 10 of the 40 test questions under tau
        s2 = {r["id"]: {"id": r["id"], "pred": None if k % 2 else "1", "correct": False, "tokens": 500}
              for k, r in enumerate(test)}  # half of System 2's answers are empty
        routed = [S.route_one(r, s2[r["id"]], score(S.features(r)), tau) for r in test]
        s1_share = sum(x[2] for x in routed) / len(test)
        summ = {"chosen_budget": 1024, "budgets": {"1024": {"tau": tau, "probe": params, "test": {"all": {"s1_share": s1_share}}}}}
        self.assertAlmostEqual(G.w_share(summ, test), 10 / 40)
        self.assertLess(1 - s1_share, 10 / 40)  # what the first version reported


class Stop(Exception):
    """A stop of the Mac in the middle of a step."""


class Resume(unittest.TestCase):
    """A step cut by a stop resumes without redoing or duplicating work, and a partial output is never
    taken as complete (amendment 1: the Mac sleeps and is throttled)."""

    def test_repair_jsonl(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "x.jsonl")
            self.assertFalse(C.repair_jsonl(p))
            with open(p, "w") as f:
                f.write('{"id": "a"}\n{"id": "b"}\n{"id": "c", "te')
            self.assertTrue(C.repair_jsonl(p))
            self.assertEqual([r["id"] for r in C.read_jsonl(p)], ["a", "b"])
            self.assertFalse(C.repair_jsonl(p))
            with open(p, "a") as f:
                f.write('{"id": "c"}')  # whole record, its newline not written yet
            self.assertTrue(C.repair_jsonl(p))
            self.assertEqual([r["id"] for r in C.read_jsonl(p)], ["a", "b", "c"])
            with open(p, "a") as f:
                f.write('{"id": "d"}\n')
            self.assertEqual([r["id"] for r in C.read_jsonl(p)], ["a", "b", "c", "d"])

    def test_id_problems(self):
        rows = [{"id": "a", "mode": "s1"}, {"id": "b", "mode": "s1"}]
        self.assertEqual(C.id_problems("f", rows, ["a", "b"], "s1"), [])
        self.assertEqual(len(C.id_problems("f", rows + rows[:1], ["a", "b"], "s1")), 1)
        self.assertEqual(len(C.id_problems("f", rows, ["a", "b", "c"], "s1")), 1)
        self.assertEqual(len(C.id_problems("f", rows, ["a"], "s1")), 1)
        self.assertEqual(len(C.id_problems("f", rows, ["a", "b"], "s2-256")), 1)

    def fake_world(self, root, L, n_calib=24, n_test=30, n_mmlu=8):
        """data/lfm3, the smoke's report, the base's runs/lfm3-s1s2 and a fake mlx-vlm, under root."""
        data, runs, s1s2 = (os.path.join(root, x) for x in ("data", "runs", "s1s2"))
        rng = random.Random(3)
        calib = [{"id": f"{t}-train-{i}", "task": t, "question": f"c{i}?", "answer": "1"}
                 for i in range(n_calib) for t in (("gsm8k",) if i % 2 else ("math",))]
        test = [{"id": f"gsm8k-test-{i}" if i % 2 else f"math500-{i}", "task": "gsm8k" if i % 2 else "math",
                 "question": f"t{i}?", "answer": "1"} for i in range(n_test)]
        mmlu = [{"id": f"mmlupro-{i}", "task": "mmlu_pro", "question": f"m{i}?", "options": "(A) x\n(B) y",
                 "answer": "A"} for i in range(n_mmlu)]
        C.write_jsonl(os.path.join(data, "calib.jsonl"), calib)
        C.write_jsonl(os.path.join(data, "test.jsonl"), test)
        os.makedirs(os.path.join(runs, "smoke"))
        with open(os.path.join(runs, "smoke", "report.json"), "w") as f:
            json.dump({"base_source": "published"}, f)

        def s1_row(it, base):
            lp = -rng.random() * (3 if base else 2)
            return {"id": it["id"], "task": it["task"], "mode": "s1", "text": "\\boxed{1}" if lp > -1.4 else "\\boxed{2}",
                    "tokens": rng.randint(50, 1024) if base else rng.randint(20, 400), "think_tokens": 0,
                    "finish": "length" if base and lp < -2.5 else "stop", "mean_lp": lp / 2, "min_lp": lp * 2,
                    "answer_lp": lp, "pred": "1" if lp > -1.4 else "2", "correct": lp > -1.4, "seconds": 1.0}

        for split, items in (("calib", calib), ("test", test)):
            C.write_jsonl(os.path.join(s1s2, f"{split}.s1.jsonl"), [s1_row(it, True) for it in items])
            for b in (256, 1024):
                C.write_jsonl(os.path.join(s1s2, f"{split}.s2-{b}.jsonl"), [
                    {"id": it["id"], "task": it["task"], "mode": f"s2-{b}", "text": "\\boxed{1}", "think": "t",
                     "tokens": b + 100, "think_tokens": b, "forced": True, "finish": "stop",
                     "pred": None if k % 5 == 0 else "1", "correct": k % 5 != 0, "seconds": 1.0}
                    for k, it in enumerate(items)])
        ft_rows = {it["id"]: s1_row(it, False) for it in calib + test + mmlu}
        calls = []

        class FakeModel:
            def __init__(self, path=None):
                self.path = path

        def fake_gen_s1(m, item, seed):
            calls.append((m.path, item["question"]))
            if fake_gen_s1.stop_after is not None and len(calls) > fake_gen_s1.stop_after:
                raise Stop()
            qid = next(k for k, it in fake_gen_s1.by_prompt.items() if it == item["question"])
            r = dict(ft_rows[qid]) if m.path else s1_row({"id": qid, "task": "x"}, True)
            return {k: r[k] for k in ("text", "tokens", "think_tokens", "finish", "mean_lp", "min_lp", "answer_lp")}

        fake_gen_s1.stop_after = None
        fake_gen_s1.by_prompt = {it["id"]: C.user_prompt(it) for it in calib + test + mmlu}
        patches = {(L, "RUNS"): runs, (L, "SMOKE"): os.path.join(runs, "smoke"), (L, "S1S2_RUNS"): s1s2,
                   (L, "C_DATA"): data, (L, "FT_MLX"): os.path.join(root, "ft-mlx"), (L, "LocalModel"): FakeModel,
                   (L.S, "gen_s1"): fake_gen_s1, (L.S, "rest"): lambda s: None, (L.C, "mmlu_items"): lambda: list(mmlu)}
        saved = {k: getattr(*k) for k in patches}
        for (mod, name), v in patches.items():
            setattr(mod, name, v)
        self.addCleanup(lambda: [setattr(mod, name, v) for (mod, name), v in saved.items()])
        return runs, data, calls, fake_gen_s1, {"calib": calib, "test": test, "mmlu": mmlu}

    def test_eval_resumes_after_a_stop(self):
        import argparse

        import local as L

        with tempfile.TemporaryDirectory() as root:
            runs, data, calls, gen, items = self.fake_world(root, L)
            gen.stop_after = 30  # the Mac stops in the middle of the test split
            with self.assertRaises(Stop):
                L.cmd_eval(argparse.Namespace(tag="ft", limit=0))
            d = os.path.join(runs, "eval", "ft")
            self.assertFalse(os.path.exists(os.path.join(d, "meta.json")))
            with open(os.path.join(d, "test.s1.jsonl"), "a") as f:
                f.write('{"id": "gsm8k-test-9", "te')  # and a record cut mid-write
            gen.stop_after = None
            L.cmd_eval(argparse.Namespace(tag="ft", limit=0))
            total = sum(len(v) for v in items.values())
            self.assertEqual(len(calls), total + 1)  # each question once, plus the one stopped mid-generation
            for split in ("calib", "test", "mmlu"):
                rows = C.read_jsonl(os.path.join(d, f"{split}.s1.jsonl"))
                self.assertEqual(C.id_problems(split, rows, [it["id"] for it in items[split]], "s1"), [])
            self.assertEqual(sorted(f for f in os.listdir(d) if f.endswith(".jsonl")),
                             ["calib.s1.jsonl", "mmlu.s1.jsonl", "test.s1.jsonl"])  # s1 only (amendment 5)
            self.assertTrue(all(p == os.path.join(root, "ft-mlx") for p, _ in calls))
            n = len(calls)
            L.cmd_eval(argparse.Namespace(tag="ft", limit=0))  # once more: nothing to do
            self.assertEqual(len(calls), n)

    def test_base_copy_cut_is_redone(self):
        import local as L

        with tempfile.TemporaryDirectory() as root:
            runs, data, calls, gen, items = self.fake_world(root, L)
            d = os.path.join(runs, "eval", "base")
            os.makedirs(d)
            src = os.path.join(L.S1S2_RUNS, "test.s2-1024.jsonl")
            with open(src) as f, open(os.path.join(d, "test.s2-1024.jsonl"), "w") as g:
                g.write("".join(f.readlines()[:7]))  # a copy cut at a line boundary: it parses
            L.copy_base_generations(d)
            for n in os.listdir(L.S1S2_RUNS):
                with open(os.path.join(L.S1S2_RUNS, n)) as f, open(os.path.join(d, n)) as g:
                    self.assertEqual(f.read(), g.read(), n)

    def test_the_local_chain_on_fakes(self):
        """eval base (copied + MMLU-Pro), eval ft (s1 only), the pair, route and gate, end to end."""
        import argparse
        import subprocess

        import local as L

        with tempfile.TemporaryDirectory() as root:
            runs, data, calls, gen, items = self.fake_world(root, L)
            L.cmd_eval(argparse.Namespace(tag="base", limit=0))
            self.assertEqual({p for p, _ in calls}, {None})  # the published base, MMLU-Pro only
            self.assertEqual(len(calls), len(items["mmlu"]))
            L.cmd_eval(argparse.Namespace(tag="ft", limit=0))
            L.build_pair(runs, data)
            pair = os.path.join(runs, "eval", "pair")
            for s in ("calib", "test"):
                with open(os.path.join(L.S1S2_RUNS, f"{s}.s2-1024.jsonl")) as f, \
                        open(os.path.join(pair, f"{s}.s2-1024.jsonl")) as g:
                    self.assertEqual(f.read(), g.read())  # the base's s2, untouched
                with open(os.path.join(runs, "eval", "ft", f"{s}.s1.jsonl")) as f, \
                        open(os.path.join(pair, f"{s}.s1.jsonl")) as g:
                    self.assertEqual(f.read(), g.read())  # the fine-tuned s1
            env = dict(os.environ, SKY_DUTY="1", SKY_COOL="", SKY_BRAKE_FILE="")
            subprocess.run([sys.executable, "-I", os.path.join(HERE, "route.py"), "--tag", "pair", "--runs", runs],
                           check=True, env=env, stdout=subprocess.DEVNULL)
            summ = json.load(open(os.path.join(pair, "summary.json")))
            subprocess.run([sys.executable, "-I", os.path.join(HERE, "gate.py"), "--runs", runs, "--amendment5"],
                           check=True, env=env, stdout=subprocess.DEVNULL)
            g = json.load(open(os.path.join(runs, "gate.json")))
            self.assertEqual(g["G2"], G.G2_NOT_READ)
            self.assertEqual(g["verdict"], "incomplete")  # 30 test questions, not 1,000
            self.assertIsNone(g["improved"])
            # H2-H4 of the pair are those of route's summary, with s1s2.py's bars
            e = summ["budgets"][str(summ["chosen_budget"])]["test"]["all"]
            self.assertEqual(g["router"]["pair"]["H2"], e["router_acc"] >= e["s2_acc"] - 0.02)
            self.assertEqual(g["router"]["pair"]["H3"], e["router_tokens"] <= 0.5 * e["s2_tokens"])
            # the cascade: 1/rho from s1s2.py's token counts, w from the probe and tau fitted on the pair
            c = g["cascade"]
            self.assertAlmostEqual(c["one_over_rho"], summ["budgets"]["1024"]["test"]["all"]["s1_tokens"]
                                   / summ["budgets"]["1024"]["test"]["all"]["s2_tokens"])
            cal = S.read_jsonl(os.path.join(pair, "calib.s1.jsonl"))
            score, _ = S.fit_logistic([S.features(r) for r in cal], [int(r["correct"]) for r in cal])
            tau = summ["budgets"][str(summ["chosen_budget"])]["tau"]
            test = S.read_jsonl(os.path.join(pair, "test.s1.jsonl"))
            self.assertAlmostEqual(c["w"], sum(score(S.features(r)) < tau for r in test) / len(test))
            # a pair file edited after the fact is refused by route, not silently intersected
            rows = C.read_jsonl(os.path.join(pair, "test.s2-256.jsonl"))
            C.write_jsonl(os.path.join(pair, "test.s2-256.jsonl"), rows[1:])
            p = subprocess.run([sys.executable, "-I", os.path.join(HERE, "route.py"), "--tag", "pair", "--runs", runs],
                               env=env, capture_output=True, text=True)
            self.assertNotEqual(p.returncode, 0)
            self.assertIn("missing", p.stderr)

    def test_gate_reads_no_g2_in_local_mode_before_the_pair(self):
        import subprocess

        with tempfile.TemporaryDirectory() as runs:
            base, ft = model(0.60, 0.10, 0.70, 500, 0.40), model(0.70, 0.0, 0.70, 300, 0.40)
            for tag, m, files in (("base", base, G.FILES), ("ft", ft, {"s1": G.FILES["s1"], "mmlu": G.FILES["mmlu"]})):
                for k, f in files.items():
                    C.write_jsonl(os.path.join(runs, "eval", tag, f), m[k])
            subprocess.run([sys.executable, "-I", os.path.join(HERE, "gate.py"), "--runs", runs, "--amendment5"],
                           check=True, stdout=subprocess.DEVNULL)
            g = json.load(open(os.path.join(runs, "gate.json")))
            self.assertEqual((g["G2"], g["verdict"]), (G.G2_NOT_READ, "LFM3 improved"))

    def test_gate_duplicates_are_incomplete(self):
        base, ft = model(0.60, 0.10, 0.70, 500, 0.40), model(0.70, 0.0, 0.70, 300, 0.40)
        ft["s1"] = ft["s1"] + ft["s1"][:5]  # a record written twice
        g = G.gates(base, ft, read_g2=False)
        self.assertEqual(g["verdict"], "incomplete")
        self.assertIsNone(g["improved"])
        self.assertTrue(any("duplicated" in p for p in g["problems"]))

    def test_sft_done_is_not_redone_and_convert_waits_for_it(self):
        import argparse

        import local as L

        with tempfile.TemporaryDirectory() as root:
            saved = (L.SFT_RESULT, L.MERGED, L.FT_MLX)
            self.addCleanup(lambda: (setattr(L, "SFT_RESULT", saved[0]), setattr(L, "MERGED", saved[1]),
                                     setattr(L, "FT_MLX", saved[2])))
            L.SFT_RESULT, L.MERGED, L.FT_MLX = (os.path.join(root, x) for x in ("result.json", "merged", "ft-mlx"))
            os.makedirs(L.MERGED)
            with open(os.path.join(L.MERGED, "config.json"), "w") as f:
                f.write("{}")  # a merge cut after its config: not done without result.json
            with self.assertRaises(SystemExit):
                L.cmd_convert(None)
            with open(L.SFT_RESULT, "w") as f:
                f.write("{}")
            L.cmd_sft(argparse.Namespace(revision="r"))  # returns before any import of the torch stack
            with open(L.SFT_RESULT) as f:
                self.assertEqual(f.read(), "{}")

    def test_convert_cut_is_not_taken_as_done(self):
        import local as L

        with tempfile.TemporaryDirectory() as root:
            dst = os.path.join(root, "out")

            def cut(cmd, check):
                out = cmd[cmd.index("--mlx-path") + 1]
                os.makedirs(out)
                with open(os.path.join(out, "config.json"), "w") as f:
                    f.write("{}")  # the config is written, the weights are not
                raise Stop()

            with self.assertRaises(Stop):
                L.convert("src", dst, run=cut)
            self.assertFalse(os.path.exists(dst))

            def ok(cmd, check):
                out = cmd[cmd.index("--mlx-path") + 1]
                self.assertFalse(os.path.exists(out))  # the cut attempt was cleared
                os.makedirs(out)
                for n in ("config.json", "model.safetensors"):
                    with open(os.path.join(out, n), "w") as f:
                        f.write("x")

            L.convert("src", dst, run=ok)
            self.assertEqual(sorted(os.listdir(dst)), ["config.json", "model.safetensors"])
            self.assertFalse(os.path.exists(dst + ".tmp"))


if __name__ == "__main__":
    unittest.main(verbosity=1)
