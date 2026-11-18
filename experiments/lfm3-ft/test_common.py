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

    def test_memory_update_keeps_the_peaks(self):
        """The smoke's memory figures (amendment 7): the highest MPS readings, the lowest available memory."""
        import local as L

        rec = {}
        L.memory_update(rec, 5, 3, 900)
        L.memory_update(rec, 9, 2, 1000)
        L.memory_update(rec, 7, 4, 400)
        self.assertEqual(rec, {"mps_driver_peak": 9, "mps_current_peak": 4, "available_min": 400})
        L.memory_update(rec, None, None, None)  # nothing readable: nothing changes
        self.assertEqual(rec, {"mps_driver_peak": 9, "mps_current_peak": 4, "available_min": 400})

    def test_lora_choice_follows_amendment_1(self):
        """Rule 1 (peft on MPS) first, rule 2 (mlx-vlm) only when rule 1 failed, else no method."""
        import local as L

        peft = {"lora_ok": True, "per_device": 4, "grad_accum": 8}
        mlx = {"lora_ok": True, "per_device": 8, "grad_accum": 4}
        self.assertEqual(L.lora_choice(peft, mlx), {"lora_method": "peft-mps", "per_device": 4, "grad_accum": 8, "ok": True})
        self.assertEqual(L.lora_choice({"lora_ok": False}, mlx)["lora_method"], "mlx-vlm")
        self.assertEqual(L.lora_choice({"lora_ok": False}, mlx)[("per_device")], 8)
        self.assertEqual(L.lora_choice({"lora_ok": False}, {"lora_ok": False}),
                         {"lora_method": None, "per_device": None, "grad_accum": None, "ok": False})
        self.assertFalse(L.lora_choice({"lora_ok": False}, None)["ok"])  # no mlx-lora.json: nothing validated

    def test_parse_swap_used(self):
        import local as L

        self.assertEqual(L.parse_swap_used("total = 8448,00M  used = 7198,94M  free = 1249,06M  (encrypted)"), 7198.94)
        self.assertEqual(L.parse_swap_used("total = 0.00M  used = 0.00M  free = 0.00M"), 0.0)
        with self.assertRaises(ValueError):
            L.parse_swap_used("")

    def test_guarded_stops_a_child_that_swaps(self):
        """The swap guard: a child whose swap grows past the limit is killed; one that stays under it ends."""
        import local as L

        class Child:
            def __init__(self, polls):
                self.polls, self.returncode, self.killed = polls, None, False

            def poll(self):
                if self.killed or self.polls == 0:
                    self.returncode = -9 if self.killed else 0
                    return self.returncode
                self.polls -= 1
                return None

            def kill(self):
                self.killed = True

            def wait(self):
                self.returncode = -9

        swaps = iter([700.0, 900.0, 3100.0, 9000.0])  # the bound is on the swap in use, not on its growth
        child = Child(polls=10)
        g = L.guarded(["x"], swap_max_mb=3072, load_max=8, read_swap=lambda: next(swaps), read_level=lambda: 40,
                      read_load=lambda: 2.0, popen=lambda cmd: child, sleep=lambda s: None)
        self.assertTrue(g["stopped_by_guard"].startswith("swap") and child.killed)
        self.assertEqual((g["swap_start_mb"], g["swap_peak_mb"], g["memory_free_pct_min"]), (700.0, 3100.0, 40))
        loads = iter([1.0, 3.0, 8.5])
        child = Child(polls=10)
        g = L.guarded(["x"], swap_max_mb=3072, load_max=8, read_swap=lambda: 100.0, read_level=lambda: None,
                      read_load=lambda: next(loads), popen=lambda cmd: child, sleep=lambda s: None)
        self.assertEqual((g["stopped_by_guard"], g["load_peak"]), ("load 8.50", 8.5))
        calm = iter([100.0] * 5)
        g = L.guarded(["x"], swap_max_mb=3072, load_max=8, read_swap=lambda: next(calm), read_level=lambda: None,
                      read_load=lambda: 1.0, popen=lambda cmd: Child(polls=2), sleep=lambda s: None)
        self.assertIsNone(g["stopped_by_guard"])
        self.assertEqual(g["returncode"], 0)

    def test_try_batches_falls_back_to_4(self):
        """Batch 8 stopped by the guard (or out of memory): batch 4 is tried, and its record is the result."""
        import local as L

        with tempfile.TemporaryDirectory() as d:
            out = lambda n: os.path.join(d, f"try-{n}.json")  # noqa: E731
            with open(out(4), "w") as f:
                f.write('{"stale": true}')  # left by an earlier run: removed before the attempt
            seen = []

            def run(cmd):
                n = int(cmd[-1])
                seen.append(n)
                if n == 4:
                    with open(out(4), "w") as f:
                        json.dump({"trained": True, "ok": True}, f)
                return {"stopped_by_guard": "swap 3500 MB" if n == 8 else None}

            calm = lambda: (True, 0.0, 500.0)  # noqa: E731
            attempts, child = L.try_batches("smoke-lora-try", "rev", out, run=run, wait=calm)
            self.assertEqual(seen, [8, 4])
            self.assertEqual((child["per_device"], child["grad_accum"]), (4, 8))
            self.assertIsNone(attempts["8"]["child"])

            def none_fit(cmd):
                with open(out(int(cmd[-1])), "w") as f:
                    json.dump({"trained": False, "error": "out of memory"}, f)
                return {"stopped_by_guard": None}

            attempts, child = L.try_batches("smoke-lora-try", "rev", out, run=none_fit, wait=calm)
            self.assertIsNone(child)
            self.assertEqual(sorted(attempts), ["4", "8"])

            # The swap does not come back under the bound after batch 8: batch 4 is not run, nothing is chosen.
            settles = iter([(True, 0.0, 500.0), (False, 1200.0, 2900.0)])
            seen.clear()
            attempts, child = L.try_batches("smoke-lora-try", "rev", out, run=run, wait=lambda: next(settles))
            self.assertEqual((seen, child), ([8], None))
            self.assertIn("2900 MB", attempts["4"]["not_run"])

    def test_settle_waits_for_the_swap(self):
        import local as L

        swaps = iter([2500.0, 2300.0, 1900.0])
        self.assertEqual(L.settle(limit_mb=2048, wait=60, period=10, read_swap=lambda: next(swaps), sleep=lambda s: None),
                         (True, 20.0, 1900.0))
        self.assertEqual(L.settle(limit_mb=2048, wait=30, period=10, read_swap=lambda: 2500.0, sleep=lambda s: None),
                         (False, 30.0, 2500.0))

    def test_fits_is_the_recommended_working_set(self):
        import local as L

        self.assertTrue(L.fits(10, 10))
        self.assertFalse(L.fits(11, 10))
        self.assertFalse(L.fits(None, 10))

    def test_drop_weights_keeps_the_rest(self):
        """The smoke's merged model, once reloaded: its weights are removed and recorded, the rest is kept."""
        import hashlib

        import local as L

        with tempfile.TemporaryDirectory() as d:
            for name, body in (("model.safetensors", b"weights"), ("config.json", b"{}"), ("modeling_lfm3.py", b"#")):
                with open(os.path.join(d, name), "wb") as f:
                    f.write(body)
            gone = L.drop_weights(d)
            self.assertEqual(gone, {"model.safetensors": {"bytes": 7, "sha256": hashlib.sha256(b"weights").hexdigest()}})
            self.assertEqual(sorted(os.listdir(d)), ["config.json", "modeling_lfm3.py"])


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
            data, s1s2 = os.path.join(runs, "data"), os.path.join(runs, "s1s2")
            for s, ids in split_ids.items():
                C.write_jsonl(os.path.join(data, f"{s}.jsonl"), [{"id": i} for i in ids])
            all_six = ["calib.s1.jsonl", "calib.s2-256.jsonl", "calib.s2-1024.jsonl",
                       "test.s1.jsonl", "test.s2-256.jsonl", "test.s2-1024.jsonl"]
            for tag, d, names in (("ft", os.path.join(runs, "eval", "ft"), ["calib.s1.jsonl", "test.s1.jsonl", "mmlu.s1.jsonl"]),
                                  ("base", os.path.join(runs, "eval", "base"), all_six + ["mmlu.s1.jsonl"]),
                                  ("s1s2", s1s2, all_six)):
                os.makedirs(d)
                for n in names:
                    split, mode = n.split(".")[:2]
                    C.write_jsonl(os.path.join(d, n), [row(i, mode, tag) for i in split_ids.get(split, ["mmlupro-1"])])
            meta = L.build_pair(runs, data, s1s2)
            self.assertEqual(meta["sources"], {"calib.s1.jsonl": "eval/ft", "test.s1.jsonl": "eval/ft",
                                               "calib.s2-256.jsonl": s1s2, "calib.s2-1024.jsonl": s1s2,
                                               "test.s2-256.jsonl": s1s2, "test.s2-1024.jsonl": s1s2})
            self.assertEqual(meta["conversions"], {
                "calib.s1.jsonl": "local conversion", "test.s1.jsonl": "local conversion",
                "calib.s2-256.jsonl": "published 4-bit file", "calib.s2-1024.jsonl": "published 4-bit file",
                "test.s2-256.jsonl": "published 4-bit file", "test.s2-1024.jsonl": "published 4-bit file"})
            d = os.path.join(runs, "eval", "pair")
            self.assertEqual(json.load(open(os.path.join(d, "meta.json")))["sources"], meta["sources"])
            self.assertEqual({r["src"] for r in C.read_jsonl(os.path.join(d, "calib.s1.jsonl"))}, {"ft"})
            self.assertEqual({r["src"] for r in C.read_jsonl(os.path.join(d, "test.s1.jsonl"))}, {"ft"})
            self.assertEqual([r["id"] for r in C.read_jsonl(os.path.join(d, "test.s2-1024.jsonl"))], split_ids["test"])
            # System 2 is runs/lfm3-s1s2, never eval/base (amendment 6: also when eval/base holds s2 copies)
            self.assertEqual({r["src"] for r in C.read_jsonl(os.path.join(d, "test.s2-1024.jsonl"))}, {"s1s2"})
            self.assertEqual(sorted(os.listdir(d)),  # no mmlu in the pair: the router reads calib and test only
                             ["calib.s1.jsonl", "calib.s2-1024.jsonl", "calib.s2-256.jsonl", "meta.json",
                              "test.s1.jsonl", "test.s2-1024.jsonl", "test.s2-256.jsonl"])
            os.remove(os.path.join(s1s2, "test.s2-256.jsonl"))
            with self.assertRaises(SystemExit):
                L.build_pair(runs, data, s1s2)
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
            for split, mode, origin in L.PAIR_FILES:
                rows = [{"id": i, "mode": mode} for i in ids[split]]
                rows = edit(split, mode, origin, rows)
                folder = os.path.join(runs, "s1s2") if origin == "s1s2" else os.path.join(runs, "eval", origin)
                C.write_jsonl(os.path.join(folder, f"{split}.{mode}.jsonl"), rows)
            return data

        cases = {
            "ft test s1 missing one": lambda s, m, t, r: r[:-1] if (s, m) == ("test", "s1") else r,
            "runs/lfm3-s1s2 calib s2-256 one extra": lambda s, m, t, r: r + [{"id": "gsm8k-test-0", "mode": m}]
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
                    L.build_pair(runs, data, os.path.join(runs, "s1s2"))
                self.assertFalse(os.path.exists(os.path.join(runs, "eval", "pair", "meta.json")))
        with tempfile.TemporaryDirectory() as runs:  # the clean case passes
            L.build_pair(runs, setup(runs, lambda s, m, t, r: r), os.path.join(runs, "s1s2"))

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


class Amendment6(unittest.TestCase):
    """Amendment 6: the plan of `eval` in each case of the conversion check, and the budget of 1/rho."""

    def test_eval_plan_never_runs_the_base_in_s2(self):
        import local as L

        s1_only = [("calib", "s1"), ("test", "s1"), ("mmlu", "s1")]
        self.assertEqual(L.eval_plan("ft", 0), (s1_only, L.FT_MLX, f"{C.FT_REPO} (local)", "local conversion", False))
        self.assertEqual(L.eval_plan("base", 0, "converted")[::3], (s1_only, "local conversion"))
        self.assertEqual(L.eval_plan("base", 0, "converted")[1], L.BASE_MLX)
        self.assertEqual(L.eval_plan("base", 0, "published"),  # unchanged: copy, MMLU-Pro with the published file
                         ([("mmlu", "s1")], None, L.S.MODEL, "published 4-bit file", True))
        for tag, source in (("ft", None), ("base", "published"), ("base", "converted")):
            for limit in (0, 5):
                jobs = L.eval_plan(tag, limit, source)[0]
                self.assertEqual({m for _, m in jobs}, {"s1"}, (tag, source, limit))
        with self.assertRaises(SystemExit):
            L.eval_plan("base", 0, None)  # a smoke report without its conversion check: no guess

    def test_conversion_of(self):
        self.assertEqual(C.conversion_of("published"), "published 4-bit file")
        self.assertEqual(C.conversion_of("converted"), "local conversion")
        self.assertIsNone(C.conversion_of(None))

    def pair_folder(self, runs, s2_tokens):
        d = os.path.join(runs, "eval", "pair")
        C.write_jsonl(os.path.join(d, "test.s1.jsonl"), [ev(f"t{i}", tokens=100) for i in range(4)])
        for b, t in s2_tokens.items():
            C.write_jsonl(os.path.join(d, f"test.s2-{b}.jsonl"), [ev(f"t{i}", tokens=t) for i in range(4)])

    def test_the_budget_of_one_over_rho_is_the_pairs(self):
        """1/rho divides by the base's tokens at the pair's chosen budget, read from its own files."""
        with tempfile.TemporaryDirectory() as runs:
            self.pair_folder(runs, {256: 400, 1024: 1000})
            for b, s2 in ((256, 400), (1024, 1000)):
                summ = {"chosen_budget": b, "budgets": {str(b): {"test": {"all": {}}}}}
                c = G.pair_cascade(runs, summ, 100, 500)
                self.assertEqual((c["budget"], c["base_s2_tokens"]), (b, s2))
                self.assertAlmostEqual(c["one_over_rho"], 100 / s2)
                self.assertIsNone(c["w"])  # no probe in this summary
                md = G.report_md({"base": {"s1": None, "s2": None, "mmlu": None},
                                  "ft": {"s1": None, "s2": None, "mmlu": None}, "problems": [], "G1": None,
                                  "cascade": c, "router": {}})
                self.assertIn(f"s2-{b} of the base {s2}", md)
                self.assertIn(f"at s2-{b}, the budget the pair's router chose on its calibration", md)
                self.assertNotIn(f"s2-{1280 - b} ", md)

    def test_not_routed_yet(self):
        with tempfile.TemporaryDirectory() as runs:
            self.pair_folder(runs, {256: 400, 1024: 1000})
            for summ in (None, {"budgets": {}}):
                c = G.pair_cascade(runs, summ, 100, 500)
                self.assertEqual((c["routed"], c["budget"], c["one_over_rho"], c["w"]), (False, None, None, None))
                self.assertEqual((c["ft_s1_tokens"], c["base_s1_tokens"]), (100, 500))
                md = G.report_md({"base": {"s1": None, "s2": None, "mmlu": None},
                                  "ft": {"s1": None, "s2": None, "mmlu": None}, "problems": [], "G1": None,
                                  "cascade": c, "router": {}})
                self.assertIn("1/rho and w: the pair has not been routed yet (`route`)", md)
                self.assertNotIn("1/rho =", md)

    def test_provenance_before_eval(self):
        """gate.json says the base's conversion from the smoke's check when eval/base has no meta yet."""
        with tempfile.TemporaryDirectory() as runs:
            self.assertEqual(G.provenance(runs)["base_s1"], "unknown")
            os.makedirs(os.path.join(runs, "smoke"))
            with open(os.path.join(runs, "smoke", "report.json"), "w") as f:
                json.dump({"base_source": "converted"}, f)
            pv = G.provenance(runs)
            self.assertEqual((pv["base_s1"], pv["ft_s1"], pv["system2"]),
                             ("local conversion", "local conversion", "published 4-bit file"))


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

    def fake_world(self, root, L, n_calib=24, n_test=30, n_mmlu=8, base_source="published"):
        """data/lfm3, the smoke's report, the base's runs/lfm3-s1s2 and a fake mlx-vlm, under root. The fake
        model answers by its path: ft-mlx (the fine-tuned model), base-mlx (the local conversion of the
        base: its records carry "src": "converted") or None (the published 4-bit file)."""
        data, runs, s1s2 = (os.path.join(root, x) for x in ("data", "runs", "s1s2"))
        ft_mlx, base_mlx = os.path.join(root, "ft-mlx"), os.path.join(root, "base-mlx")
        for d in (ft_mlx, base_mlx):
            os.makedirs(d)
            with open(os.path.join(d, "config.json"), "w") as f:
                f.write("{}")
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
            json.dump({"base_source": base_source}, f)

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
        conv_rows = {it["id"]: dict(s1_row(it, True), src="converted", tokens=2000 + k)
                     for k, it in enumerate(calib + test + mmlu)}  # far from the published s1 (<= 1,024)
        calls = []

        class FakeModel:
            def __init__(self, path=None):
                self.path = path

        def fake_gen_s1(m, item, seed):
            calls.append((m.path, item["question"]))
            if fake_gen_s1.stop_after is not None and len(calls) > fake_gen_s1.stop_after:
                raise Stop()
            qid = next(k for k, it in fake_gen_s1.by_prompt.items() if it == item["question"])
            r = {ft_mlx: ft_rows, base_mlx: conv_rows}.get(m.path, {}).get(qid) or s1_row({"id": qid, "task": "x"}, True)
            keys = ("text", "tokens", "think_tokens", "finish", "mean_lp", "min_lp", "answer_lp")
            return {k: r[k] for k in keys + (("src",) if "src" in r else ())}

        fake_gen_s1.stop_after = None
        fake_gen_s1.by_prompt = {it["id"]: C.user_prompt(it) for it in calib + test + mmlu}
        patches = {(L, "RUNS"): runs, (L, "SMOKE"): os.path.join(runs, "smoke"), (L, "S1S2_RUNS"): s1s2,
                   (L, "C_DATA"): data, (L, "FT_MLX"): ft_mlx, (L, "BASE_MLX"): base_mlx, (L, "LocalModel"): FakeModel,
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

    def run_chain(self, root, L, base_source):
        """eval base, eval ft (s1 only), the pair, route and gate on the fakes; checks what holds in both
        cases of the conversion check and returns what each test checks on its own."""
        import argparse
        import subprocess

        runs, data, calls, gen, items = self.fake_world(root, L, base_source=base_source)
        L.cmd_eval(argparse.Namespace(tag="base", limit=0))
        base_calls = list(calls)
        L.cmd_eval(argparse.Namespace(tag="ft", limit=0))
        L.build_pair(runs, data)
        pair = os.path.join(runs, "eval", "pair")
        for s in ("calib", "test"):
            for b in (256, 1024):  # System 2 is the published base of runs/lfm3-s1s2, untouched, in both cases
                with open(os.path.join(L.S1S2_RUNS, f"{s}.s2-{b}.jsonl")) as f, \
                        open(os.path.join(pair, f"{s}.s2-{b}.jsonl")) as g:
                    self.assertEqual(f.read(), g.read())
            with open(os.path.join(runs, "eval", "ft", f"{s}.s1.jsonl")) as f, \
                    open(os.path.join(pair, f"{s}.s1.jsonl")) as g:
                self.assertEqual(f.read(), g.read())  # the fine-tuned s1
        pmeta = json.load(open(os.path.join(pair, "meta.json")))
        self.assertEqual(pmeta["base_source"], base_source)
        self.assertEqual({k: v for k, v in pmeta["conversions"].items() if ".s2-" in k},
                         {f"{s}.s2-{b}.jsonl": "published 4-bit file" for s in ("calib", "test") for b in (256, 1024)})
        env = dict(os.environ, SKY_DUTY="1", SKY_COOL="", SKY_BRAKE_FILE="")
        route = [sys.executable, "-I", os.path.join(HERE, "route.py"), "--tag", "pair", "--runs", runs]
        gate = [sys.executable, "-I", os.path.join(HERE, "gate.py"), "--runs", runs, "--amendment5"]
        subprocess.run(gate, check=True, env=env, stdout=subprocess.DEVNULL)  # before route: no budget yet
        g = json.load(open(os.path.join(runs, "gate.json")))
        self.assertEqual((g["cascade"]["routed"], g["cascade"]["one_over_rho"], g["cascade"]["w"]), (False, None, None))
        self.assertIn("the pair has not been routed yet", open(os.path.join(runs, "report.md")).read())
        subprocess.run(route, check=True, env=env, stdout=subprocess.DEVNULL)
        summ = json.load(open(os.path.join(pair, "summary.json")))
        subprocess.run(gate, check=True, env=env, stdout=subprocess.DEVNULL)
        g = json.load(open(os.path.join(runs, "gate.json")))
        md = open(os.path.join(runs, "report.md")).read()
        self.assertEqual(g["G2"], G.G2_NOT_READ)
        self.assertEqual(g["verdict"], "incomplete")  # 30 test questions, not 1,000
        self.assertIsNone(g["improved"])
        # H2-H4 of the pair are those of route's summary, with s1s2.py's bars
        b = summ["chosen_budget"]
        e = summ["budgets"][str(b)]["test"]["all"]
        self.assertEqual(g["router"]["pair"]["H2"], e["router_acc"] >= e["s2_acc"] - 0.02)
        self.assertEqual(g["router"]["pair"]["H3"], e["router_tokens"] <= 0.5 * e["s2_tokens"])
        # the cascade: 1/rho at the budget the pair's router chose (the fakes tie on calibration, so
        # s1s2.py keeps the first budget, 256: not the 1,024 the first version hard-coded)
        c = g["cascade"]
        self.assertEqual((b, c["budget"], c["routed"]), (256, 256, True))
        s2 = C.read_jsonl(os.path.join(L.S1S2_RUNS, f"test.s2-{b}.jsonl"))
        self.assertAlmostEqual(c["base_s2_tokens"], sum(r["tokens"] for r in s2) / len(s2))
        self.assertAlmostEqual(c["one_over_rho"], e["s1_tokens"] / e["s2_tokens"])
        self.assertIn(f"at s2-{b}, the budget the pair's router chose", md)
        cal = S.read_jsonl(os.path.join(pair, "calib.s1.jsonl"))
        score, _ = S.fit_logistic([S.features(r) for r in cal], [int(r["correct"]) for r in cal])
        tau = summ["budgets"][str(b)]["tau"]
        test = S.read_jsonl(os.path.join(pair, "test.s1.jsonl"))
        self.assertAlmostEqual(c["w"], sum(score(S.features(r)) < tau for r in test) / len(test))
        self.assertEqual(g["provenance"]["system2"], "published 4-bit file")
        self.assertEqual(g["provenance"]["ft_s1"], "local conversion")
        self.assertIn("System 2 of the pair (the base in s2-256 and s2-1024, runs/lfm3-s1s2): published 4-bit file", md)
        self.assertIn("- the fine-tuned model in s1: local conversion", md)
        # a pair file edited after the fact is refused by route, not silently intersected
        rows = C.read_jsonl(os.path.join(pair, "test.s2-256.jsonl"))
        C.write_jsonl(os.path.join(pair, "test.s2-256.jsonl"), rows[1:])
        p = subprocess.run(route, env=env, capture_output=True, text=True)
        self.assertNotEqual(p.returncode, 0)
        self.assertIn("missing", p.stderr)
        return runs, items, base_calls, g, md

    def test_the_local_chain_on_fakes(self):
        """The published case (the conversion check holds), unchanged: eval base copies runs/lfm3-s1s2 and
        generates MMLU-Pro only, with the published file; then ft, the pair, route and gate."""
        import local as L

        with tempfile.TemporaryDirectory() as root:
            runs, items, base_calls, g, md = self.run_chain(root, L, "published")
            self.assertEqual({p for p, _ in base_calls}, {None})  # the published base, MMLU-Pro only
            self.assertEqual(len(base_calls), len(items["mmlu"]))
            d = os.path.join(runs, "eval", "base")
            for n in os.listdir(L.S1S2_RUNS):  # the six files of runs/lfm3-s1s2, copied as they are
                with open(os.path.join(L.S1S2_RUNS, n)) as f, open(os.path.join(d, n)) as h:
                    self.assertEqual(f.read(), h.read(), n)
            meta = json.load(open(os.path.join(d, "meta.json")))
            self.assertEqual((meta["base_source"], meta["conversion"]), ("published", "published 4-bit file"))
            self.assertEqual({v["conversion"] for v in meta["files"].values()}, {"published 4-bit file"})
            self.assertEqual(meta["files"]["test.s1.jsonl"]["from"], L.S1S2_RUNS)
            self.assertEqual(meta["files"]["mmlu.s1.jsonl"]["from"], "generated here")
            base_s1 = C.read_jsonl(os.path.join(L.S1S2_RUNS, "test.s1.jsonl"))
            self.assertAlmostEqual(g["base"]["s1"]["mean_tokens"], sum(r["tokens"] for r in base_s1) / len(base_s1))
            self.assertEqual(g["provenance"]["base_s1"], "published 4-bit file")
            self.assertIn("- the base in s1 (G1 and G3, calibration, test and MMLU-Pro): published 4-bit file", md)

    def test_the_local_chain_converted(self):
        """Amendment 6, the conversion check says "converted": eval base generates s1 only, with the local
        conversion, on calib, test and MMLU-Pro; nothing runs the base in s2; G1 and G3 read those s1
        files; System 2 of the pair is still runs/lfm3-s1s2; every number says its conversion."""
        import local as L

        with tempfile.TemporaryDirectory() as root:
            runs, items, base_calls, g, md = self.run_chain(root, L, "converted")
            self.assertEqual({p for p, _ in base_calls}, {L.BASE_MLX})  # the local conversion only
            self.assertEqual(len(base_calls), sum(len(v) for v in items.values()))  # each question once, in s1
            d = os.path.join(runs, "eval", "base")
            self.assertEqual(sorted(f for f in os.listdir(d) if f.endswith(".jsonl")),
                             ["calib.s1.jsonl", "mmlu.s1.jsonl", "test.s1.jsonl"])  # no s2 file, nothing copied
            for split in ("calib", "test", "mmlu"):
                rows = C.read_jsonl(os.path.join(d, f"{split}.s1.jsonl"))
                self.assertEqual(C.id_problems(split, rows, [it["id"] for it in items[split]], "s1"), [])
                self.assertEqual({r["src"] for r in rows}, {"converted"})
            meta = json.load(open(os.path.join(d, "meta.json")))
            self.assertEqual((meta["base_source"], meta["conversion"], meta["revision"]),
                             ("converted", "local conversion", "local"))
            self.assertEqual({n: v["conversion"] for n, v in meta["files"].items()},
                             {n: "local conversion" for n in ("calib.s1.jsonl", "mmlu.s1.jsonl", "test.s1.jsonl")})
            self.assertIn("published 4-bit file", meta["system2"])
            # G1 and G3 compare the fine-tuned model with the base on the same conversion
            conv = C.read_jsonl(os.path.join(d, "test.s1.jsonl"))
            self.assertAlmostEqual(g["base"]["s1"]["mean_tokens"], sum(r["tokens"] for r in conv) / len(conv))
            self.assertGreater(g["base"]["s1"]["mean_tokens"], 2000)  # not the published s1 of runs/lfm3-s1s2
            self.assertEqual(g["base"]["mmlu"]["n"], len(items["mmlu"]))
            self.assertAlmostEqual(g["cascade"]["base_s1_tokens"], g["base"]["s1"]["mean_tokens"])
            self.assertIsNone(g["base"]["s2"])
            self.assertEqual((g["provenance"]["base_source"], g["provenance"]["base_s1"]), ("converted", "local conversion"))
            self.assertIn("- the base in s1 (G1 and G3, calibration, test and MMLU-Pro): local conversion", md)
            self.assertIn("| test s1 | base (local conversion) |", md)
            self.assertIn("| MMLU-Pro s1 | base (local conversion) |", md)
            self.assertIn("s2-256 of the base (published 4-bit file)", md)

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
