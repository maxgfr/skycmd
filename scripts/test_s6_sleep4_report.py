#!/usr/bin/env python3
"""Tests of scripts/s6-sleep4-report.py on synthetic logs and readings: K3 (a pass on both seeds,
a pass on one seed only, which fails, S90, S95, the windows, the excess ratio, the cap), K1 under
amendment 2 (a pass on both seeds, a pass on one seed only, which fails, a run that never reaches
0.90, only after K3's failure, once, its command without a frozen base or an early stop), the
command of a check against v3's, the stop of phase A, the seeds of v4 and the refusal of v3's,
the parity checks of the frozen route (a difference raises), H0 to H3 against the base, and the
gate.

  python3 -I scripts/test_s6_sleep4_report.py
"""
import contextlib
import importlib.util
import io
import json
import os
import shlex
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("s6_sleep4_report", os.path.join(HERE, "s6-sleep4-report.py"))
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


def write(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f)


def text(path):
    with open(path) as f:
        return f.read()


def quiet(f, *a):
    with contextlib.redirect_stdout(io.StringIO()):
        return f(*a)


def recall_row(step, acc, asks=64, aux=(0.5, 0.7), off=False, short=None, long_=None):
    """One recall step of a --seq-log log: `acc` of its asks right, split evenly over the four
    delay buckets unless `short` / `long_` (the accuracy at 1-4 / 17-32) are given."""
    right = round(acc * asks)
    per = asks / 4
    b = [acc] * 4
    if short is not None:
        b[0] = short
    if long_ is not None:
        b[3] = long_
    splits = [float(right)]
    for x in b:
        splits += [x * per, per]
    splits += [right / 2, asks / 2, right / 2, asks / 2]
    r = {"step": step, "seq_game": "recall", "passes": 1, "seq_step_loss": 0.5, "grad_norm": 2.0,
         "asks": float(asks), "ask_ce": 1.0 - acc, "ask_ce_last": 1.0 - acc, "ask_acc": right / asks,
         "ask_splits": splits, "aux_ce": 0.5, "aux_acc": sum(aux) / len(aux),
         "aux_ce_layers": [0.5, 0.5], "aux_acc_layers": list(aux)}
    if off:
        r["off"] = {"ask_ce": 1.5, "ask_ce_last": 1.5, "ask_acc": 0.25, "ask_splits": [0.0] * 13}
    return r


def curve(n_recall, accs, final=True, stopped=None):
    """A log with a stateless step every tenth step and `n_recall` recall steps whose accuracy is
    accs(k) at recall step k (from 1)."""
    rows, k, step = [], 0, 0
    while k < n_recall:
        step += 1
        if step % 10 == 0:
            rows.append({"step": step, "loss": 0.5})
            continue
        k += 1
        rows.append(recall_row(step, accs(k), off=(k % 4 == 1)))
    if final:
        fin = {"step": step, "final": True, "acc_f32": 0.5, "acc_ternary": 0.5, "frozen_base_checked": True}
        if stopped is not None:
            fin.update(stopped_by_rule=stopped, recall_steps=n_recall)
        rows.append(fin)
    return rows


def write_log(run, rows):
    os.makedirs(run, exist_ok=True)
    with open(os.path.join(run, "log.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


V3_CMD = ("nice -n 0 target/s3-train/release/sky-train s2-decide --config runs/s6-sleep3/nano/sleep3.json "
          "--init runs/s6-nano-g3/decide/last --tokenizer runs/s6-nano-g3/decide/last/tokenizer.json "
          "--records data/decisions,data/distill,data/games,data/numeric "
          "--weights decisions=0.25,distill=0.30,games=0.35,numeric=0.10 --batch 128 --minutes 12 --seed 0 "
          "--qat-frac 1.0 --loop-dropout 0.25 --length-pool 16 --pad-to 32 --out runs/s6-sleep3/nano/sleep-s0/a "
          "--step-mem-gb 3 --adam-lr 0.00163 --no-muon --seq-log "
          "--sleep-records runs/s6-sleep/seq/recall.jsonl,runs/s6-sleep/seq/rule110.jsonl "
          "--sleep-weights recall=0.85,rule110=0.15 --sleep-batch 16 --seq-frac 0.9 --sleep-passes 1 "
          "--mem-lr-mult 10 --mem-gate-lr-mult 0 --memory-aux-weight 1 --sleep-chunk 2 ")


def k3_cmd(seed, extra=""):
    c = V3_CMD.replace("--minutes 12", "--steps 1310").replace("--seed 0", f"--seed {seed}")
    c = c.replace("runs/s6-sleep3/nano/sleep-s0/a", f"runs/s6-sleep4/stage1/k3-s{seed}")
    c = c.replace("runs/s6-sleep3/nano/sleep3.json", "runs/s6-sleep4/sleep4.json")
    return c + "--freeze-base --memory-off-eval 4 " + extra


def k3_root(tmp, acc0, acc1, n=1000):
    for s, accs in ((0, acc0), (1, acc1)):
        run = os.path.join(tmp, "stage1", f"k3-s{s}")
        write_log(run, curve(n, accs))
        with open(os.path.join(run, "cmd.txt"), "w") as f:
            f.write(k3_cmd(s))
    return tmp


rising = lambda k: min(0.25 + k / 600, 0.99)  # 0.90 near recall step 390, 0.95 near 420
slow = lambda k: min(0.25 + k / 2000, 0.99)  # 0.75 at recall step 1000: never 0.90


class Stage1(unittest.TestCase):
    def test_k3_passes_on_both_seeds_and_sets_the_cap(self):
        with tempfile.TemporaryDirectory() as tmp:
            ok = quiet(report.cmd_k3, k3_root(tmp, rising, rising))
            self.assertTrue(ok)
            res = report.load(os.path.join(tmp, "stage1", "k3.json"))
            s1 = report.load(os.path.join(tmp, "stage1.json"))
            self.assertEqual(s1["route"], "frozen")
            x = res["seeds"]["seed0"]
            self.assertTrue(x["bar"] and x["frozen_base_checked"])
            self.assertEqual(x["recall_steps"], 1000)
            self.assertEqual(len(x["windows"]), 20)
            # S90, S95 from the curve itself, by brute force.
            rows = report.recall_rows(report.log_lines(os.path.join(tmp, "stage1", "k3-s0")))
            pooled = lambda lo, hi: sum(report.right(r) for r in rows[lo:hi]) / sum(r["asks"] for r in rows[lo:hi])
            s90 = next(n for n in range(50, 1001) if pooled(n - 50, n) >= 0.90)
            s95 = next(n for n in range(100, 1001) if pooled(n - 100, n) >= 0.95)
            self.assertEqual((x["S90"], x["S95"]), (s90, s95))
            self.assertEqual(s1["cap"]["S95max"], s95)
            self.assertEqual(s1["cap"]["steps"], -(-100 * s95 // 51))
            w = x["windows"][0]
            self.assertEqual(w["recall_steps"], [1, 50])
            self.assertAlmostEqual(w["aux_acc_layers"][0], 0.5)
            self.assertAlmostEqual(w["aux_acc_layers"][1], 0.7)
            self.assertGreater(w["off_steps"], 0)
            self.assertAlmostEqual(w["off_ask_acc"], 0.25)
            self.assertEqual(w["grad_norm_median"], 2.0)

    def test_a_pass_on_one_seed_only_is_a_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            ok = quiet(report.cmd_k3, k3_root(tmp, rising, slow))
            self.assertFalse(ok)
            s1 = report.load(os.path.join(tmp, "stage1.json"))
            self.assertEqual(s1["route"], "closed")
            self.assertNotIn("cap", s1)
            self.assertTrue(s1["k3"]["seed0"]["bar"])
            self.assertFalse(s1["k3"]["seed1"]["bar"])
            self.assertIsNone(s1["k3"]["seed1"]["S90"])
            self.assertIn("K1", s1["next"])
            # No stage-2 run on a closed route.
            with self.assertRaises(SystemExit):
                quiet(report.cmd_nano_ready, tmp)

    def test_the_bar_is_pooled_over_the_last_100_recall_steps(self):
        # 0.96 for 900 steps, then 0.90 over the last 100: the last 100 decide.
        late = lambda k: 0.96 if k <= 900 else 0.90
        with tempfile.TemporaryDirectory() as tmp:
            self.assertFalse(quiet(report.cmd_k3, k3_root(tmp, late, late)))
        early = lambda k: 0.5 if k <= 900 else 0.97
        with tempfile.TemporaryDirectory() as tmp:
            self.assertTrue(quiet(report.cmd_k3, k3_root(tmp, early, early)))

    def test_k3_wants_its_own_command_and_an_ended_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            k3_root(tmp, rising, rising)
            with open(os.path.join(tmp, "stage1", "k3-s1", "cmd.txt"), "w") as f:
                f.write(k3_cmd(1).replace("--freeze-base ", ""))
            with self.assertRaises(SystemExit):
                quiet(report.cmd_k3, tmp)
        with tempfile.TemporaryDirectory() as tmp:
            k3_root(tmp, rising, rising)
            with open(os.path.join(tmp, "stage1", "k3-s0", "cmd.txt"), "w") as f:
                f.write(k3_cmd(0, "--stop-after 654"))
            with self.assertRaises(SystemExit):
                quiet(report.cmd_k3, tmp)
        with tempfile.TemporaryDirectory() as tmp:
            k3_root(tmp, rising, rising)
            write_log(os.path.join(tmp, "stage1", "k3-s0"), curve(1000, rising, final=False))
            with self.assertRaises(SystemExit):
                quiet(report.cmd_k3, tmp)
        # A K2-type run cut by --stop-after is read without a final line.
        with tempfile.TemporaryDirectory() as tmp:
            run = os.path.join(tmp, "k2")
            write_log(run, curve(500, rising, final=False))
            r = report.stage1_reading(run, require_final=False)
            self.assertEqual(r["recall_steps"], 500)

    def test_the_excess_ratio_is_not_defined_near_chance(self):
        rows = [recall_row(i, 0.5, short=0.26, long_=0.25) for i in range(10)]
        self.assertIsNone(report.excess_ratio(rows))
        rows = [recall_row(i, 0.5, short=0.75, long_=0.50) for i in range(10)]
        self.assertAlmostEqual(report.excess_ratio(rows), 0.5)

    def test_the_cap_of_section_4_2(self):
        # Section 9's example: S95max = 700 recall steps, a cap of 1,050, 1,373 steps.
        c = report.cap_of([650, 700])
        self.assertEqual((c["S95max"], c["cap_recall_steps"], c["steps"]), (700, 1050.0, 1373))
        self.assertEqual(report.cap_of([51])["steps"], 100)

    def test_a_check_is_v3s_recipe_but_for_the_flags_of_section_3(self):
        with tempfile.TemporaryDirectory() as tmp:
            v3, v4 = os.path.join(tmp, "v3.txt"), os.path.join(tmp, "v4.txt")
            with open(v3, "w") as f:
                f.write(V3_CMD)
            with open(v4, "w") as f:
                f.write(k3_cmd(1))
            self.assertEqual(report.recipe_diff(v3, v4), [])
            with open(v4, "w") as f:
                f.write(k3_cmd(1).replace("--adam-lr 0.00163", "--adam-lr 0.002"))
            self.assertEqual(report.recipe_diff(v3, v4), ["--adam-lr"])
            with open(v4, "w") as f:
                f.write(k3_cmd(1).replace("--sleep-chunk 2 ", ""))
            self.assertEqual(report.recipe_diff(v3, v4), ["--sleep-chunk"])
            # printf %q quoting is read back.
            with open(v4, "w") as f:
                f.write(" ".join(shlex.quote(w) for w in shlex.split(k3_cmd(1))))
            self.assertEqual(report.recipe_diff(v3, v4), [])


# ----------------------------------------------------------------------------- K1 (amendment 2)


def k1_cmd(seed, extra=""):
    """K1's command: K3's without --freeze-base (section 3.3, amendment 2)."""
    return k3_cmd(seed).replace("--freeze-base ", "").replace(f"k3-s{seed}", f"k1-s{seed}") + extra


K3_FAILED = {"k3": {"seed0": {"bar": False}, "seed1": {"bar": False}}, "k3_pass": False, "route": "closed",
             "next": "the frozen route is closed for v4: K1 (seed 0, then seed 1), then K2"}


def k1_root(tmp, acc0, acc1, n=1000, stage1=K3_FAILED, k3_logs=True):
    """K3's failure recorded in stage1.json (and K3's logs on the same batches), then K1's two runs."""
    write(os.path.join(tmp, "stage1.json"), stage1)
    for s, accs in ((0, acc0), (1, acc1)):
        if k3_logs:
            write_log(os.path.join(tmp, "stage1", f"k3-s{s}"), curve(n, lambda k: 0.25))
        run = os.path.join(tmp, "stage1", f"k1-s{s}")
        write_log(run, curve(n, accs))
        with open(os.path.join(run, "cmd.txt"), "w") as f:
            f.write(k1_cmd(s))
    return tmp


class K1(unittest.TestCase):
    def test_k1_passes_on_both_seeds(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(report.k1_ready(k1_root(tmp, rising, rising)), [])
            self.assertTrue(quiet(report.cmd_k1, tmp))
            res = report.load(os.path.join(tmp, "stage1", "k1.json"))
            self.assertTrue(res["pass"])
            for name in ("seed0", "seed1"):
                x = res["seeds"][name]
                self.assertTrue(x["bar"])
                self.assertEqual(x["recall_steps"], 1000)
                self.assertEqual(len(x["windows"]), 20)
                self.assertIsNotNone(x["S90"])
                self.assertIsNotNone(x["S95"])  # a run that meets the bar has an S95 (section 3.1)
                self.assertLess(x["S90"], x["S95"])
                self.assertTrue(x["same_batches_as_k3"])
            # S90 and S95 from the curve itself, by brute force.
            rows = report.recall_rows(report.log_lines(os.path.join(tmp, "stage1", "k1-s0")))
            pooled = lambda lo, hi: sum(report.right(r) for r in rows[lo:hi]) / sum(r["asks"] for r in rows[lo:hi])
            s90 = next(n for n in range(50, 1001) if pooled(n - 50, n) >= 0.90)
            s95 = next(n for n in range(100, 1001) if pooled(n - 100, n) >= 0.95)
            self.assertEqual((res["seeds"]["seed0"]["S90"], res["seeds"]["seed0"]["S95"]), (s90, s95))
            # stage1.json keeps K3's part and gains K1's, with the rule of amendment 2.
            s1 = report.load(os.path.join(tmp, "stage1.json"))
            self.assertEqual((s1["route"], s1["k3_pass"], s1["next"]), ("closed", False, K3_FAILED["next"]))
            self.assertTrue(s1["k1_pass"])
            self.assertTrue(s1["k1"]["seed0"]["bar"] and s1["k1"]["seed1"]["bar"])
            self.assertIn("nothing further runs until the lead writes the next amendment", s1["next_after_k1"])
            md = text(os.path.join(tmp, "stage1", "k1.md"))
            self.assertIn("**Verdict:** K1 passed on both seeds.", md)
            self.assertEqual(sum(l.startswith("| K1 |") for l in md.splitlines()), 2)
            # Read once: K1 may not run again.
            self.assertTrue(any("no rerun" in w for w in report.k1_ready(tmp)))

    def test_a_pass_on_one_seed_only_is_a_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertFalse(quiet(report.cmd_k1, k1_root(tmp, rising, slow)))
            res = report.load(os.path.join(tmp, "stage1", "k1.json"))
            self.assertFalse(res["pass"])
            self.assertTrue(res["seeds"]["seed0"]["bar"])
            self.assertFalse(res["seeds"]["seed1"]["bar"])
            s1 = report.load(os.path.join(tmp, "stage1.json"))
            self.assertFalse(s1["k1_pass"])
            self.assertIn("v4 stops at K1", s1["next_after_k1"])
            self.assertIn("no K2", s1["next_after_k1"])
            self.assertIn("**Verdict:** K1 failed", text(os.path.join(tmp, "stage1", "k1.md")))
        # The other seed passing alone fails too.
        with tempfile.TemporaryDirectory() as tmp:
            self.assertFalse(quiet(report.cmd_k1, k1_root(tmp, slow, rising)))

    def test_a_run_that_never_reaches_0_90_has_no_s90_and_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertFalse(quiet(report.cmd_k1, k1_root(tmp, slow, slow)))
            res = report.load(os.path.join(tmp, "stage1", "k1.json"))
            for name in ("seed0", "seed1"):
                x = res["seeds"][name]
                self.assertFalse(x["bar"])
                self.assertIsNone(x["S90"])
                self.assertIsNone(x["S95"])
                self.assertAlmostEqual(x["last_100_ask_acc"], sum(0.25 + k / 2000 for k in range(901, 1001)) / 100, places=2)
            md = text(os.path.join(tmp, "stage1", "k1.md"))
            self.assertIn("| never | never |", md)

    def test_the_bar_is_read_on_the_last_100_recall_steps(self):
        # Above 0.95 early, under it at the end: the last 100 decide, an S95 alone does not pass.
        late = lambda k: 0.97 if k <= 900 else 0.90
        with tempfile.TemporaryDirectory() as tmp:
            self.assertFalse(quiet(report.cmd_k1, k1_root(tmp, late, late)))
            x = report.load(os.path.join(tmp, "stage1", "k1.json"))["seeds"]["seed0"]
            self.assertIsNotNone(x["S95"])
            self.assertFalse(x["bar"])

    def test_k1_runs_only_after_k3_failed(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertTrue(report.k1_ready(tmp))  # no stage1.json: K3 not read
            with self.assertRaises(SystemExit):
                quiet(report.cmd_k1_ready, tmp)
        passed = {"k3": {}, "k3_pass": True, "route": "frozen", "cap": {"steps": 1373}}
        with tempfile.TemporaryDirectory() as tmp:
            k1_root(tmp, rising, rising, stage1=passed)
            self.assertTrue(any("K3's failure" in w for w in report.k1_ready(tmp)))
            with self.assertRaises(SystemExit):
                quiet(report.cmd_k1_ready, tmp)
            with self.assertRaises(SystemExit):
                quiet(report.cmd_k1, tmp)
            self.assertFalse(os.path.exists(os.path.join(tmp, "stage1", "k1.json")))
            quiet(report.cmd_k1_ready, tmp, True)  # a smoke records it and goes on

    def test_k1_wants_its_own_command_and_ended_runs(self):
        bad = (
            k3_cmd(0),  # the base frozen: K3's command, not K1's
            k1_cmd(0, "--stop-after 654 "),  # K2's shorter run
            k1_cmd(0, "--stop-recall-acc 0.95 --stop-recall-window 100 "),  # an early stop moves the last 100
            k1_cmd(0).replace("--steps 1310", "--steps 1000"),
            k1_cmd(0).replace("--memory-off-eval 4 ", ""),
            k1_cmd(1),  # another seed's command
        )
        for cmd in bad:
            with tempfile.TemporaryDirectory() as tmp:
                k1_root(tmp, rising, rising)
                with open(os.path.join(tmp, "stage1", "k1-s0", "cmd.txt"), "w") as f:
                    f.write(cmd)
                with self.assertRaises(SystemExit, msg=cmd):
                    quiet(report.cmd_k1, tmp)
                self.assertFalse(os.path.exists(os.path.join(tmp, "stage1", "k1.json")))
        with tempfile.TemporaryDirectory() as tmp:
            k1_root(tmp, rising, rising)
            write_log(os.path.join(tmp, "stage1", "k1-s1"), curve(1000, rising, final=False))
            with self.assertRaises(SystemExit):
                quiet(report.cmd_k1, tmp)
            self.assertNotIn("k1", report.load(os.path.join(tmp, "stage1.json")))
        # K1's command is v3's recipe but for the flags of section 3.1.
        with tempfile.TemporaryDirectory() as tmp:
            v3, v4 = os.path.join(tmp, "v3.txt"), os.path.join(tmp, "v4.txt")
            with open(v3, "w") as f:
                f.write(V3_CMD)
            with open(v4, "w") as f:
                f.write(k1_cmd(0))
            self.assertEqual(report.recipe_diff(v3, v4), [])

    def test_the_batches_against_k3_are_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            k1_root(tmp, rising, rising, k3_logs=False)
            quiet(report.cmd_k1, tmp)
            self.assertIsNone(report.load(os.path.join(tmp, "stage1", "k1.json"))["seeds"]["seed0"]["same_batches_as_k3"])
        with tempfile.TemporaryDirectory() as tmp:
            k1_root(tmp, rising, rising)
            rows = curve(1000, lambda k: 0.25)
            next(r for r in rows if r.get("seq_game") == "recall")["seq_game"] = "rule110"
            write_log(os.path.join(tmp, "stage1", "k3-s1"), rows)
            self.assertTrue(quiet(report.cmd_k1, tmp))  # reported, deciding nothing
            res = report.load(os.path.join(tmp, "stage1", "k1.json"))
            self.assertTrue(res["seeds"]["seed0"]["same_batches_as_k3"])
            self.assertFalse(res["seeds"]["seed1"]["same_batches_as_k3"])


# ----------------------------------------------------------------------------- stage 2


def probe_file(path, game, start, n, scores, memory=True, off=False, passes=(1, 2, 4), asks=None):
    """An s2-probe file: one reading per N; `scores(N, i)` the score of episode i (out of 8 for
    recall, whose asks are then marked right for the first `score` of them)."""
    out = []
    seeds = report.episode_seeds(start, n)
    for p in passes if memory and not off else (1,):
        eps = []
        for i, s in enumerate(seeds):
            sc = scores(p, i)
            e = {"seed": s, "score": float(sc), "max_score": 8.0}
            if game == "recall":
                e["asks"] = [[k % 4, k < sc] for k in range(8)] if asks is None else asks(p, i)
                e["shown"] = [1 + k % 2 for k in range(8)]
            eps.append(e)
        r = {"game": game, "memory": memory, "passes": p, "from_seed": start, "n_episodes": n,
             "score": sum(e["score"] for e in eps) / (8.0 * len(eps)), "episodes": eps}
        if off:
            r["memory_off"] = True
        out.append(r)
    write(path, out)


def arena_file(path, game, start, n, score, memory=True, passes=(1,)):
    write(path, [{"game": game, "memory": memory, "passes": p, "from_seed": start, "n_episodes": n,
                  "score": score, "arena": {"mean_score": score}} for p in passes])


TYPED = """# Evaluation report: skycmd-{name}

Checkpoint: `{ckpt}`
Split: `test` (20 questions)

## Summary

| model | last-exit acc | early-exit acc |
|---|---:|---:|
| latent f32 (final QAT master weights) | 0.5000 | 0.4500 |
| ternary (fake-quant) | {acc} | 0.4400 |
"""


def typed_file(path, name, ckpt, acc="0.4900"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(TYPED.format(name=name, ckpt=ckpt, acc=acc))


def frozen_root(tmp, n=12, zeroed_differs=False, typed_differs=False, flappy=0.99):
    """A frozen route on synthetic readings, in smoke seeds (SMOKE_TEST, SMOKE_ARENA): the memory
    model answers 6 of 8 at N = 1 and 7 of 8 at N = 4, the base 2 of 8, the zeroed memory as
    the base."""
    write(os.path.join(tmp, "stage1.json"), {"k3_pass": True, "route": "frozen", "k3": {}, "cap": {"steps": 1373}})
    write(os.path.join(tmp, "n.json"), {"n": 1, "dev3_recall": {"1": 0.7}, "base_recall": 0.25})
    for s in report.SEEDS:
        for ph, stopped in (("a", True), ("b", None)):
            write_log(report.nano_run(tmp, s, ph), curve(120, lambda k: 0.99, stopped=stopped))
    t = lambda who, name: os.path.join(tmp, "test", who, name)
    st, sa = report.SMOKE_TEST, report.SMOKE_ARENA
    base_score = lambda p, i: 2
    probe_file(t("base", "recall.json"), "recall", st, n, base_score, memory=False)
    probe_file(t("base", "rule110.json"), "rule110", st, n, lambda p, i: 4, memory=False)
    for g in report.GAMES:
        arena_file(t("base", f"{g}.json"), g, sa, 1, 10.0, memory=False)
    typed_file(t("base", "typed.md"), "s6-nano", "runs/s6-sleep4/base/last")
    for s in report.SEEDS:
        who = f"sleep-s{s}"
        probe_file(t(who, "recall.json"), "recall", st, n, lambda p, i: 6 if p < 4 else 7)
        probe_file(t(who, "rule110.json"), "rule110", st, n, lambda p, i: 4)
        off = (lambda p, i: 3 if i == 5 else 2) if zeroed_differs and s == 2 else base_score
        probe_file(t(who, "recall-off.json"), "recall", st, n, off, off=True)
        for g in report.GAMES:
            arena_file(t(who, f"{g}.json"), g, sa, 1, 10.0 * (flappy if g == "flappy" else 1.0), passes=(1,))
        typed_file(t(who, "typed.md"), "s6-sleep4-nano", f"runs/s6-sleep4/nano/{who}/b/last",
                   acc="0.4800" if typed_differs and s == 1 else "0.4900")
    os.makedirs(os.path.join(tmp, "test"), exist_ok=True)
    open(os.path.join(tmp, "test", "done"), "w").close()
    return tmp


class FrozenRoute(unittest.TestCase):
    def test_the_report_reads_h0_to_h3_against_the_base(self):
        with tempfile.TemporaryDirectory() as tmp:
            quiet(report.cmd_report, frozen_root(tmp), True)
            v = report.load(os.path.join(tmp, "verdict.json"))
            self.assertEqual(v["route"], "frozen")
            self.assertTrue(v["H0"] and v["H1"] and v["H2"] and v["H3"] and v["Z"])
            self.assertFalse(v["goal"])  # 0.75 at N = 1 against the goal of 0.90
            self.assertTrue(all(x["equal"] for x in v["parity"]["recall_zeroed"].values()))
            text = open(os.path.join(tmp, "REPORT.md")).read()
            self.assertIn("section 8", text)
            self.assertIn("ended by", text)

    def test_a_game_lost_against_the_base_fails_h2(self):
        with tempfile.TemporaryDirectory() as tmp:
            quiet(report.cmd_report, frozen_root(tmp, flappy=0.97), True)
            self.assertFalse(report.load(os.path.join(tmp, "verdict.json"))["H2"])

    def test_a_zeroed_memory_that_differs_from_the_base_stops_the_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(SystemExit) as e:
                quiet(report.cmd_report, frozen_root(tmp, zeroed_differs=True), True)
            self.assertIn("PARITY FAILS", str(e.exception))
            self.assertFalse(os.path.exists(os.path.join(tmp, "verdict.json")))

    def test_an_answer_that_differs_with_the_same_score_stops_too(self):
        with tempfile.TemporaryDirectory() as tmp:
            frozen_root(tmp)
            # The same score, the right answer on another ask: answer for answer, not by score.
            path = os.path.join(tmp, "test", "sleep-s0", "recall-off.json")
            probe_file(path, "recall", report.SMOKE_TEST, 12, lambda p, i: 2, off=True,
                       asks=lambda p, i: [[k % 4, k in (0, 2)] for k in range(8)])
            with self.assertRaises(SystemExit):
                quiet(report.cmd_report, tmp, True)

    def test_a_language_reading_that_differs_from_the_base_stops_the_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(SystemExit) as e:
                quiet(report.cmd_report, frozen_root(tmp, typed_differs=True), True)
            self.assertIn("typed_decisions_bench, seed 1", str(e.exception))

    def test_a_zeroed_reading_must_be_read_with_the_memory_off(self):
        with tempfile.TemporaryDirectory() as tmp:
            frozen_root(tmp)
            path = os.path.join(tmp, "test", "sleep-s0", "recall-off.json")
            probe_file(path, "recall", report.SMOKE_TEST, 12, lambda p, i: 2, passes=(1,))
            with self.assertRaises(SystemExit):
                quiet(report.cmd_report, tmp, True)

    def test_the_seeds_of_v4_and_the_refusal_of_v3s(self):
        with tempfile.TemporaryDirectory() as tmp:
            ok = os.path.join(tmp, "ok.json")
            probe_file(ok, "recall", report.TEST_SEED, report.TEST_EPISODES, lambda p, i: 4)
            report.check_file(ok, report.TEST_SEED, report.TEST_EPISODES)
            for start, n in ((5_000_000_000, 1000), (2_000_000_000, 500), (1_000_000_000, 500), (3000, 2)):
                path = os.path.join(tmp, f"{start}.json")
                probe_file(path, "recall", start, n, lambda p, i: 4)
                with self.assertRaises(SystemExit) as e:
                    report.check_file(path, report.TEST_SEED, report.TEST_EPISODES)
                self.assertIn("never reads", str(e.exception))
            for g, start in (("flappy", 5000), ("doom", 8000)):
                path = os.path.join(tmp, f"{g}.json")
                arena_file(path, g, start, 200, 1.0)
                with self.assertRaises(SystemExit):
                    report.check_file(path, report.ARENA_SEED, report.ARENA_EPISODES)
            arena = os.path.join(tmp, "arena.json")
            arena_file(arena, "dodge", report.ARENA_SEED, 200, 1.0)
            report.check_file(arena, report.ARENA_SEED, report.ARENA_EPISODES)
            arena_file(arena, "dodge", report.ARENA_SEED, 30, 1.0)
            with self.assertRaises(SystemExit):
                report.check_file(arena, report.ARENA_SEED, report.ARENA_EPISODES)
            # A dev3 reading with other episodes than the first 500 = 19 mod 20 from 3e9.
            dev = os.path.join(tmp, "dev.json")
            probe_file(dev, "rule110", report.DEV3_SEED, report.DEV3_EPISODES, lambda p, i: 4)
            report.check_file(dev, report.DEV3_SEED, report.DEV3_EPISODES)
            r = report.load(dev)
            r[0]["episodes"][7]["seed"] += 20
            write(dev, r)
            with self.assertRaises(SystemExit):
                report.check_file(dev, report.DEV3_SEED, report.DEV3_EPISODES)
        # No constant of v4 is a seed of v3, dev2 or v1's dev.
        for s in (report.DEV3_SEED, report.TEST_SEED, report.ARENA_SEED):
            self.assertNotIn(s, report.REFUSED_SEEDS)

    def test_phase_a_ends_by_the_rule_or_the_cap_and_v2s_stop_ends_v4(self):
        with tempfile.TemporaryDirectory() as tmp:
            for s in report.SEEDS:
                write_log(report.nano_run(tmp, s, "a"), curve(150, lambda k: 0.99, stopped=(s != 1)))
            quiet(report.cmd_phase_a, tmp)
            res = report.load(os.path.join(tmp, "nano", "phase-a.json"))
            self.assertEqual([res["seeds"][f"seed{s}"]["ended_by"] for s in report.SEEDS], ["rule", "cap", "rule"])
            self.assertFalse(res["stop"])
        with tempfile.TemporaryDirectory() as tmp:
            for s in report.SEEDS:
                rows = curve(150, lambda k: 0.99 if s else 0.25, stopped=False)
                for r in rows:
                    if r.get("seq_game") == "recall" and not s:
                        r["ask_ce"] = 1.38
                write_log(report.nano_run(tmp, s, "a"), rows)
            with self.assertRaises(SystemExit) as e:
                quiet(report.cmd_phase_a, tmp)
            self.assertEqual(e.exception.code, 3)
            self.assertTrue(report.load(os.path.join(tmp, "nano", "phase-a.json"))["stop"])
        with tempfile.TemporaryDirectory() as tmp:
            for s in report.SEEDS:
                write_log(report.nano_run(tmp, s, "a"), curve(150, lambda k: 0.99))  # no rule
            with self.assertRaises(SystemExit):
                quiet(report.cmd_phase_a, tmp)

    def test_the_choice_of_n_on_dev3_after_the_zeroed_parity(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = lambda who, name: os.path.join(tmp, "dev3", who, name)
            start = report.SMOKE_DEV
            probe_file(d("base", "recall.json"), "recall", start, 20, lambda p, i: 2, memory=False)
            probe_file(d("base", "rule110.json"), "rule110", start, 20, lambda p, i: 4, memory=False)
            for s in report.SEEDS:
                # N = 2 (0.8688) is within 0.01 of N = 4 (0.875), the best: N = 2 is chosen.
                probe_file(d(f"sleep-s{s}", "recall.json"), "recall", start, 20, lambda p, i: {1: 5, 2: 7, 4: 7}[p] if i else {1: 5, 2: 6, 4: 7}[p])
                probe_file(d(f"sleep-s{s}", "rule110.json"), "rule110", start, 20, lambda p, i: 4)
                probe_file(d(f"sleep-s{s}", "recall-off.json"), "recall", start, 20, lambda p, i: 2, off=True)
            quiet(report.cmd_choose, tmp, True)
            n = report.load(os.path.join(tmp, "n.json"))
            self.assertEqual(n["n"], 2)
            probe_file(d("sleep-s1", "recall-off.json"), "recall", start, 20, lambda p, i: 1, off=True)
            with self.assertRaises(SystemExit):
                quiet(report.cmd_choose, tmp, True)

    def test_the_test_seeds_wait_for_every_run_and_n_and_are_read_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            why = report.test_ready(tmp)
            self.assertTrue(any("has not ended" in w for w in why) and any("n.json" in w for w in why))
            frozen_root(tmp)
            self.assertEqual(report.test_ready(tmp), ["the test seeds were already read (test/done)"])
            os.remove(os.path.join(tmp, "test", "done"))
            self.assertEqual(report.test_ready(tmp), [])

    def test_the_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            quiet(report.cmd_report, frozen_root(tmp), True)
            g = os.path.join(tmp, "gate")
            write(os.path.join(g, "eval-full.json"), {"sources": {"typed_decisions_bench": {"accuracy": 0.49}, "all": {"ece": 0.05}}})
            write(os.path.join(g, "arena.json"), {x: {"model": 10.0, "baseline": 10.0} for x in report.GAMES})
            write(os.path.join(g, "parity.json"), {"native": True, "wasm": True})
            for s in report.SEEDS:
                for kind in ("checkpoint", "gguf"):
                    write(os.path.join(tmp, "parity", f"sleep-s{s}-{kind}.json"), {"same": True})
            write(os.path.join(tmp, "parity", "base-copy.json"), {"same": True})
            quiet(report.cmd_gate, tmp)
            out = report.load(os.path.join(tmp, "gate.json"))
            self.assertTrue(out["pass"])
            self.assertEqual(out["published_as"], "s6-sleep4-nano")
            # The exported file is not the base's weights plus the memory: the gate fails.
            write(os.path.join(tmp, "parity", "sleep-s0-gguf.json"), {"same": False, "error": "x"})
            with self.assertRaises(SystemExit):
                quiet(report.cmd_gate, tmp)
            write(os.path.join(tmp, "parity", "sleep-s0-gguf.json"), {"same": True})
            # A game more than 5% under s3-pico in the gate's arena.
            write(os.path.join(g, "arena.json"), {x: {"model": 9.0 if x == "dodge" else 10.0, "baseline": 10.0} for x in report.GAMES})
            with self.assertRaises(SystemExit):
                quiet(report.cmd_gate, tmp)
            self.assertIsNone(report.load(os.path.join(tmp, "gate.json"))["published_as"])

    def test_a_smoke_never_writes_under_the_real_root(self):
        with self.assertRaises(SystemExit):
            report.refuse_smoke_root("runs/s6-sleep4", True)
        report.refuse_smoke_root("runs/smoke-sleep4/runs/s6-sleep4", True)


if __name__ == "__main__":
    unittest.main()
