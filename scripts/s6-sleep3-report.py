#!/usr/bin/env python3
"""Applies the rules of docs/s6-sleep3-prereg.md (v3 = docs/s6-sleep2-prereg.md, v2, with the
gates held, the auxiliary recall loss and the check C5) to the outputs of scripts/train-sleep3.sh.

  s6-sleep3-report.py c5 <run seed 0> <run seed 1> <out.json> [lo hi]
        C5 (v3 section 3) on two s2-decide --seq-log logs, read over the steps lo to hi (default
        550 to 600); exit 1 when it fails
  s6-sleep3-report.py ready <root> <v2 checks dir> [smoke]
        whether the pilot may run: C1 to C3 of v2 on file as passed (as read, amendments 1 to 4)
        and C5 passed -> <root>/checks/ready.json; exit 1 otherwise
  s6-sleep3-report.py stop <root>      the in-sample stop after M's phase A (v2 section 5.1)
                                       -> <root>/pilot/stop.json; exit 3 when the pilot stops
  s6-sleep3-report.py pilot <root> [smoke]
        the decision rule of v2 section 5.3 -> <root>/pilot.json, with the splits reported
  s6-sleep3-report.py nano-ready <root>   exit 1 unless the pilot rule passed
  s6-sleep3-report.py choose <root> [smoke]   the choice of N on dev2 (v1 section 6) -> <root>/n.json
  s6-sleep3-report.py test-ready <root>   exit 1 unless every nano run ended, N is chosen and
                                          the test seeds were never read (v2 section 12)
  s6-sleep3-report.py report <root> [smoke]   H0 to H3 on the test seeds (v1 section 7)
                                       -> <root>/REPORT.md and <root>/verdict.json
  s6-sleep3-report.py gate <root>      the publication gate (v2 section 7, v1 section 8)
                                       -> <root>/gate.json; exit 1 when it does not hold

Every threshold below is the pre-registrations'; nothing here is tuned on a result. `smoke` is
scripts/train-sleep3.sh SMOKE=1: any seeds, any sizes, never under runs/s6-sleep3.
"""
import importlib.util
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def _module(name, file):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, file))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# v2's reader (its bootstrap, its dev2 seeds, its rule of section 5.3) and v1's (H0 to H3).
v2 = _module("s6_sleep2_report", "s6-sleep2-report.py")
v1 = _module("s6_sleep_report", "s6-sleep-report.py")

LN4 = math.log(4.0)
# C5 (v3 section 3), on both seeds, over steps 550 to 600 of a 600-step run.
C5_STEPS = (550, 600)
C5_CE = 0.30  # 1. ask-frame cross-entropy (deep supervision) at most this
C5_ACC = 0.90  # 1. ... and last-exit accuracy on the ask frames at least this
C5_OFF_CE = 1.30  # 2. the same weights with the memory off: cross-entropy at least this
C5_OFF_ACC = 0.30  # 2. ... and accuracy at most this
C5_AUX = 0.95  # 3. auxiliary accuracy at least this
C5_SEEDS = (0, 1)
# Reported with C5, deciding nothing: the first window of ten recall steps whose mean ask-frame
# cross-entropy is under 1.36 (docs/s6-sleep3-diagnosis.md section 6.1).
LEFT_CHANCE_CE = 1.36
LEFT_CHANCE_WINDOW = 10
STOP_CE, STOP_LAST = v2.STOP_CE, v2.STOP_LAST  # 1.336 over the last 50 recall steps of phase A
GAIN, ZEROED = v2.GAIN, v2.ZEROED  # 0.10 with the bootstrap bound above 0; 0.03
PASSES = v2.PASSES  # 1, 2, 4
BUCKETS = v2.BUCKETS
SEEDS = (0, 1, 2)
GAMES = ("flappy", "rescue", "dodge", "doom")
TEST_SEED, TEST_EPISODES = 5_000_000_000, 1000  # v1 section 3, read once
ARENA_SEED, ARENA_EPISODES = 5000, 200
# The s6-nano gate (docs/s6-gen-prereg.md section 6), item 1 of v1 section 8.
NANO_BENCH, NANO_ECE, NANO_ARENA = 0.398, 0.08, 0.05
DISCLOSURE = (
    "This is the third version of the same idea on the same games, and its two changes (the gates "
    "held, an auxiliary recall loss in training) were found by looking at training curves on train "
    "sequences. The auxiliary loss gives the model the colour asked as a training target at the ask "
    "frames; it is not available at inference. The gates are constants in this configuration. "
    + v2.DISCLOSURE
)


def load(path):
    with open(path) as f:
        return json.load(f)


def dump(obj, path):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def refuse_smoke_root(root, smoke):
    if smoke and os.path.normpath(root).startswith(os.path.join("runs", "s6-sleep3")):
        sys.exit(f"{root}: a smoke never writes under runs/s6-sleep3")


# ----------------------------------------------------------------------------- C5


def recall_rows(run):
    """The recall sequence steps of an s2-decide --seq-log log that carry ask readings."""
    return [r for r in v2.log_lines(run) if r.get("seq_game") == "recall" and r.get("asks")]


def ratio(a, b):
    return a / b if b else float("nan")


def c5_window(rows, lo, hi):
    """Over the recall steps lo..hi (inclusive), every reading of C5 pooled over the asks (each
    step's value weighted by its asks): the ask-frame cross-entropy (deep supervision) and the
    last-exit accuracy, the same two for the same weights with the memory off (the steps that
    were also answered without it), and the auxiliary accuracy. The mean of the per-step values
    of the cross-entropy, v2's reading of C4, is reported next to it."""
    on = [r for r in rows if lo <= r["step"] <= hi]
    off = [r for r in on if "off" in r]
    aux = [r for r in on if "aux_acc" in r]
    asks = sum(r["asks"] for r in on)
    off_asks = sum(r["asks"] for r in off)
    aux_asks = sum(r["asks"] for r in aux)
    return {
        "recall_steps": len(on),
        "asks": asks,
        "ask_ce": ratio(sum(r["ask_ce"] * r["asks"] for r in on), asks),
        "ask_ce_step_mean": v2.mean(r["ask_ce"] for r in on),
        "ask_ce_last": ratio(sum(r["ask_ce_last"] * r["asks"] for r in on), asks),
        "ask_acc": ratio(sum(r["ask_acc"] * r["asks"] for r in on), asks),
        "off_steps": len(off),
        "off_asks": off_asks,
        "off_ask_ce": ratio(sum(r["off"]["ask_ce"] * r["asks"] for r in off), off_asks),
        "off_ask_acc": ratio(sum(r["off"]["ask_acc"] * r["asks"] for r in off), off_asks),
        "aux_steps": len(aux),
        "aux_acc": ratio(sum(r["aux_acc"] * r["asks"] for r in aux), aux_asks),
    }


def left_chance(rows):
    """The first step of the first LEFT_CHANCE_WINDOW consecutive recall steps whose mean
    ask-frame cross-entropy is under LEFT_CHANCE_CE (None if the answer never left chance)."""
    ce = [(r["step"], r["ask_ce"]) for r in rows]
    for i in range(len(ce) - LEFT_CHANCE_WINDOW + 1):
        if v2.mean(c for _, c in ce[i : i + LEFT_CHANCE_WINDOW]) < LEFT_CHANCE_CE:
            return ce[i][0]
    return None


def c5_conditions(w):
    """The three conditions of C5 on one seed's window; NaN (nothing to read) fails."""
    ok = lambda x, test: not math.isnan(x) and test(x)
    return {
        "1_answer": ok(w["ask_ce"], lambda x: x <= C5_CE) and ok(w["ask_acc"], lambda x: x >= C5_ACC),
        "2_memory_off": ok(w["off_ask_ce"], lambda x: x >= C5_OFF_CE) and ok(w["off_ask_acc"], lambda x: x <= C5_OFF_ACC),
        "3_aux": ok(w["aux_acc"], lambda x: x >= C5_AUX),
    }


def c5_seed(run, lo, hi):
    if not v2.finished(run):
        sys.exit(f"{run}: the run did not end")
    rows = recall_rows(run)
    last = max((r["step"] for r in rows), default=0)
    w = c5_window(rows, lo, hi)
    cond = c5_conditions(w)
    return {
        "run": run,
        "last_recall_step": last,
        "window": w,
        "conditions": cond,
        "pass": all(cond.values()),
        "answer_left_chance_at_step": left_chance(rows),
    }


def cmd_c5(run0, run1, out, lo=C5_STEPS[0], hi=C5_STEPS[1]):
    lo, hi = int(lo), int(hi)
    seeds = {f"seed{s}": c5_seed(r, lo, hi) for s, r in zip(C5_SEEDS, (run0, run1))}
    res = {
        "check": "C5 (docs/s6-sleep3-prereg.md section 3)",
        "steps": [lo, hi],
        "bars": {
            "1": f"ask-frame CE <= {C5_CE} and last-exit accuracy on ask frames >= {C5_ACC}",
            "2": f"same weights, memory off: CE >= {C5_OFF_CE} and accuracy <= {C5_OFF_ACC}",
            "3": f"auxiliary accuracy >= {C5_AUX}",
        },
        "reading": "pooled over the asks of the recall steps of the window, each step's batch read before its update",
        "seeds": seeds,
        "pass": all(s["pass"] for s in seeds.values()),
        "retry": "none: if C5 fails on either seed, no GPU run takes place",
    }
    dump(res, out)
    for name, s in seeds.items():
        w, c = s["window"], s["conditions"]
        print(
            f"C5 {name}: steps {lo}-{hi}, {w['recall_steps']} recall steps, {w['asks']} asks | "
            f"ask CE {w['ask_ce']:.4f} (per-step mean {w['ask_ce_step_mean']:.4f}), accuracy "
            f"{w['ask_acc']:.4f} [{'ok' if c['1_answer'] else 'FAIL'}] | memory off: "
            f"{w['off_steps']} steps, {w['off_asks']} asks, CE {w['off_ask_ce']:.4f}, accuracy "
            f"{w['off_ask_acc']:.4f} [{'ok' if c['2_memory_off'] else 'FAIL'}] | aux accuracy "
            f"{w['aux_acc']:.4f} [{'ok' if c['3_aux'] else 'FAIL'}] | the answer left chance at "
            f"step {s['answer_left_chance_at_step']}"
        )
    print(f"C5: {'PASS' if res['pass'] else 'FAIL'}")
    return res["pass"]


# ----------------------------------------------------------------------------- ready


def v2_checks_passed(verdict):
    """C1 to C3 of v2 as read (v3 section 3: they stand as read): C1 passed (contrastive, amendment
    3); C2 passed or was taken by the stated deviation of amendment 4; C3, a sanity check since
    amendment 1, holds (no memory weight but the read-out has a gradient at step 0). C4 is
    replaced by C5 and is not read."""
    c1 = verdict.get("c1") is True
    c2 = verdict.get("c2") is True or bool(verdict.get("c2_deviation"))
    c3 = verdict.get("c3") is True
    return {"c1": c1, "c2": c2, "c3": c3, "c2_deviation": verdict.get("c2_deviation")}, c1 and c2 and c3


def cmd_ready(root, v2_checks, smoke=False):
    v = load(os.path.join(v2_checks, "verdict.json"))
    checks, v2_ok = v2_checks_passed(v)
    if v.get("form") != "frame":
        v2_ok = False
        checks["form"] = v.get("form")
    c5_path = os.path.join(root, "checks", "c5.json")
    c5 = load(c5_path)["pass"] if os.path.exists(c5_path) else None
    ok = v2_ok and c5 is True
    res = {"v2_checks": checks, "v2_pass": v2_ok, "c5": c5, "pass": ok}
    dump(res, os.path.join(root, "checks", "ready.json"))
    print(json.dumps(res))
    if not ok:
        if smoke:
            print("SMOKE: the pilot would be refused here; going on to check the code")
            return
        sys.exit("the pilot may not run: C1 to C3 of v2 and C5 must be on file as passed "
                 "(docs/s6-sleep3-prereg.md section 3)")


# ----------------------------------------------------------------------------- the pilot


def cmd_stop(root):
    """v2 section 5.1, unchanged, and v3 section 4: phase A's number of sequence steps."""
    run = os.path.join(root, "pilot", "m-a")
    if not v2.finished(run):
        sys.exit(f"{run}: phase A did not end")
    lines = v2.log_lines(run)
    steps = v2.recall_steps(run)
    last = [ce for _, ce in steps[-STOP_LAST:]]
    ce = v2.mean(last)
    stop = not (ce <= STOP_CE)
    res = {
        "ask_ce_last_50": ce,
        "recall_steps_read": len(last),
        "bar": STOP_CE,
        "stop": stop,
        "phase_a_sequence_steps": sum(1 for r in lines if "seq_step_loss" in r),
        "phase_a_recall_steps": len(steps),
        "answer_left_chance_at_step": left_chance(recall_rows(run)),
    }
    dump(res, os.path.join(root, "pilot", "stop.json"))
    print(
        f"phase A of M: {res['phase_a_sequence_steps']} sequence steps ({len(steps)} recall); "
        f"ask-frame CE over its last {len(last)} recall steps {ce:.4f}, the bar {STOP_CE:.4f}: "
        f"{'STOP, the pilot failed in-sample' if stop else 'go on'}"
    )
    sys.exit(3 if stop else 0)


def splits(report):
    """Accuracy by delay bucket and by the colours the asked door had been shown with (one, or
    several), pooled over the episodes of a recall reading of s2-probe."""
    by_bucket = [[0, 0] for _ in BUCKETS]
    by_colours = {"one": [0, 0], "several": [0, 0]}
    for e in report["episodes"]:
        shown = e.get("shown", [])
        for i, (b, right) in enumerate(e.get("asks", [])):
            if 0 <= b < len(BUCKETS):
                by_bucket[b][0] += int(right)
                by_bucket[b][1] += 1
            if i < len(shown) and shown[i] >= 1:
                k = "one" if shown[i] == 1 else "several"
                by_colours[k][0] += int(right)
                by_colours[k][1] += 1
    acc = lambda p: ratio(p[0], p[1])
    return {
        "by_delay": {b: acc(p) for b, p in zip(BUCKETS, by_bucket)},
        "by_delay_asks": {b: p[1] for b, p in zip(BUCKETS, by_bucket)},
        "by_colours": {k: acc(p) for k, p in by_colours.items()},
        "by_colours_asks": {k: p[1] for k, p in by_colours.items()},
    }


def pilot_rule(m1, c, off):
    """v2 section 5.3 on three dev2 recall readings: M at N = 1, the control, M zeroed."""
    gain, lo, hi = v2.paired_bootstrap(v2.pairs_of(m1, c))
    zeroed_gain = off["score"] - c["score"]
    verdict, nxt = v2.decide(gain, lo, zeroed_gain)
    return verdict, nxt, {
        "m_n1_minus_c": gain,
        "ci95": [lo, hi],
        "needs": f">= {GAIN} with the lower bound above 0",
        "m_zeroed_minus_c": zeroed_gain,
        "zeroed_needs": f"<= {ZEROED}",
    }


def cmd_pilot(root, smoke=False):
    refuse_smoke_root(root, smoke)
    p = os.path.join(root, "pilot")
    stop = load(os.path.join(p, "stop.json"))
    ready = load(os.path.join(root, "checks", "ready.json"))
    c5 = load(os.path.join(root, "checks", "c5.json"))
    out = {"disclosure": DISCLOSURE, "ready": ready, "c5": c5, "phase_a": stop}
    if ready["v2_checks"].get("c2_deviation"):
        out["c2_deviation"] = ready["v2_checks"]["c2_deviation"]
    if stop["stop"] and not smoke:
        out.update(verdict="stopped", next="the pilot failed in-sample after phase A (v2 section 5.1); v2 section 11: the fast-weight line stops")
    else:
        dev = lambda run, game: os.path.join(p, run, "dev2", f"{game}.json")
        m1 = v2.reading(dev("m", "recall"), 1)
        c = v2.reading(dev("c", "recall"), 1)
        off = v2.reading(dev("m", "recall-off"), 1)
        if not smoke:
            for r, path in ((m1, dev("m", "recall")), (c, dev("c", "recall")), (off, dev("m", "recall-off"))):
                v2.check_dev2(r, path)
        if not off.get("memory_off"):
            raise SystemExit("the zeroed reading was not taken with --memory-off")
        verdict, nxt, rule = pilot_rule(m1, c, off)
        out.update(verdict=verdict, next=nxt, rule=rule)
        out["reported"] = {
            "recall": {
                "c": c["score"],
                **{f"m_n{n}": v2.reading(dev("m", "recall"), n)["score"] for n in PASSES},
                "m_zeroed_n1": off["score"],
                "splits": {
                    "c": splits(c),
                    **{f"m_n{n}": splits(v2.reading(dev("m", "recall"), n)) for n in PASSES},
                    "m_zeroed_n1": splits(off),
                },
            },
            "rule110": {
                "c": v2.reading(dev("c", "rule110"), 1)["score"],
                **{f"m_n{n}": v2.reading(dev("m", "rule110"), n)["score"] for n in PASSES},
            },
        }
    out["runs"] = {}
    for run in ("m-a", "m-b", "c-a", "c-b"):
        state = os.path.join(p, run, "last", "state.json")
        if os.path.exists(state):
            s = load(state)
            out["runs"][run] = {"steps": s["step"], "minutes": round(s["elapsed_secs"] / 60.0, 2)}
    dump(out, os.path.join(root, "pilot.json"))
    print(json.dumps({k: out[k] for k in ("verdict", "next")}), json.dumps(out.get("rule", {})))


def cmd_nano_ready(root):
    path = os.path.join(root, "pilot.json")
    if not os.path.exists(path):
        sys.exit(f"no {path}: run the pilot and its report first")
    verdict = load(path).get("verdict")
    if verdict != "works":
        sys.exit(f"the pilot's verdict is {verdict!r}: no nano run (v2 section 5.3)")
    print("the pilot rule passed: the nano runs of v2 section 6 may run")


# ----------------------------------------------------------------------------- nano, N, test


def nano_runs(root):
    return [os.path.join(root, "nano", f"{arm}-s{s}", ph) for arm in ("sleep", "control") for s in SEEDS for ph in ("a", "b")]


def cmd_choose(root, smoke=False):
    """v1 section 6 on dev2 (v2 section 6): the smallest N in {1, 2, 4} within 0.01 of the best
    mean over the three seeds of the sleep arm."""
    refuse_smoke_root(root, smoke)
    means = {}
    for n in PASSES:
        scores = []
        for s in SEEDS:
            path = os.path.join(root, "nano", f"sleep-s{s}", "dev2", "recall.json")
            r = v2.reading(path, n)
            if not smoke:
                v2.check_dev2(r, path)
            scores.append(r["score"])
        means[n] = v1.mean(scores)
    out = {"n": v1.choose_n(means), "dev2_recall": {str(n): m for n, m in means.items()}, "tolerance": v1.N_TOLERANCE}
    dump(out, os.path.join(root, "n.json"))
    print(json.dumps(out))


def test_ready(root):
    """v2 section 12: no test seed before every nano run has ended and N is chosen; read once."""
    why = []
    for run in nano_runs(root):
        if not (os.path.exists(os.path.join(run, "log.jsonl")) and v2.finished(run)):
            why.append(f"{run} has not ended")
    if not os.path.exists(os.path.join(root, "n.json")):
        why.append("N is not chosen (no n.json)")
    if os.path.exists(os.path.join(root, "test", "done")):
        why.append("the test seeds were already read (test/done)")
    return why


def cmd_test_ready(root):
    why = test_ready(root)
    if why:
        sys.exit("the test seeds may not be read: " + "; ".join(why))
    print("every nano run ended and N is chosen: the test seeds may be read, once")


def check_test(r, path, test=True):
    seeds = TEST_SEED, TEST_EPISODES
    if r["from_seed"] != seeds[0] or r["n_episodes"] != seeds[1]:
        raise SystemExit(f"{path}: not the test seeds")


def cmd_report(root, smoke=False):
    """v1 section 7 on the test seeds (v2 section 6): H0 to H3, through v1's reader."""
    refuse_smoke_root(root, smoke)
    t = os.path.join(root, "test")
    if not os.path.exists(os.path.join(t, "done")):
        sys.exit(f"{t}/done is missing: the test reading did not end")
    chosen = load(os.path.join(root, "n.json"))["n"]
    if not smoke:
        for arm in ("sleep", "control"):
            for s in SEEDS:
                path = os.path.join(t, f"{arm}-s{s}", "recall.json")
                check_test(v2.reading(path, 1), path)
    verdict, lines = hypotheses(root, chosen)
    head = ["# s6-sleep3-nano: report (docs/s6-sleep3-prereg.md)", "", DISCLOSURE, ""]
    pilot = load(os.path.join(root, "pilot.json"))
    head += [f"Pilot: **{pilot['verdict']}** ({json.dumps(pilot.get('rule', {}))}).", ""]
    head += [f"Chosen N (dev2 recall, mean of 3 seeds): **{chosen}** ({json.dumps(load(os.path.join(root, 'n.json'))['dev2_recall'])}).", ""]
    with open(os.path.join(root, "REPORT.md"), "w") as f:
        f.write("\n".join(head + lines) + "\n")
    dump({"n": chosen, **verdict}, os.path.join(root, "verdict.json"))
    print(json.dumps(verdict))


def hypotheses(root, chosen):
    """H0 to H3 exactly as v1's cmd_report reads them (its thresholds and its bootstrap)."""
    t = os.path.join(root, "test")
    score = v1.score
    mean = v1.mean
    lines = ["## Probes (test seeds, mean of 3 training seeds)", "", "| probe | control | N=1 | N=2 | N=4 |", "|---|---:|---:|---:|---:|"]
    for game in ("recall", "rule110"):
        ctl = mean(score(os.path.join(t, f"control-s{s}", f"{game}.json"), 1) for s in SEEDS)
        by_n = [mean(score(os.path.join(t, f"sleep-s{s}", f"{game}.json"), n) for s in SEEDS) for n in PASSES]
        lines.append(f"| {game} | {ctl:.4f} | " + " | ".join(f"{x:.4f}" for x in by_n) + " |")
    lines += ["", "Recall by delay and by the colours the asked door was shown with (pooled over the 3 seeds):", ""]
    lines += ["| reading | " + " | ".join(BUCKETS) + " | one colour | several |", "|---|" + "---:|" * (len(BUCKETS) + 2)]
    for label, arm, n in [("control", "control", 1)] + [(f"N = {n}", "sleep", n) for n in PASSES] + [("N = 1, memory zeroed", "sleep-off", 1)]:
        reps = [os.path.join(t, f"{arm}-s{s}", "recall.json") for s in SEEDS]
        if not all(os.path.exists(p) for p in reps):
            continue
        merged = {"episodes": [e for p in reps for e in v2.reading(p, n)["episodes"]]}
        sp = splits(merged)
        cells = [sp["by_delay"][b] for b in BUCKETS] + [sp["by_colours"]["one"], sp["by_colours"]["several"]]
        lines.append(f"| {label} | " + " | ".join(f"{x:.4f}" for x in cells) + " |")
    verdict = {}
    lines += ["", "## Hypotheses", ""]
    d, lo, hi = v1.paired_bootstrap(v1.pooled_pairs(root, "recall", "sleep", 1, "control", 1))
    verdict["H0"] = d >= v1.MEMORY_GAIN
    lines.append(f"- **H0, the memory is used**: recall at N = 1 minus the control = {d:+.4f} (95% interval {lo:+.4f} to {hi:+.4f}), needs at least {v1.MEMORY_GAIN}: **{'holds' if verdict['H0'] else 'not shown'}**.")
    d, lo, hi = v1.paired_bootstrap(v1.pooled_pairs(root, "recall", "sleep", 4, "sleep", 1))
    verdict["H1"] = d > 0 and lo > 0
    lines.append(f"- **H1, sleep helps**: recall at N = 4 minus N = 1 = {d:+.4f} (95% interval {lo:+.4f} to {hi:+.4f}), needs a lower bound above 0: **{'holds' if verdict['H1'] else 'not shown'}**.")
    d, lo, hi = v1.paired_bootstrap(v1.pooled_pairs(root, "rule110", "sleep", 4, "sleep", 1))
    lines.append(f"- H1b (secondary, no gate): rule110 at N = 4 minus N = 1 = {d:+.4f} (95% interval {lo:+.4f} to {hi:+.4f}).")
    arena, ok = [], True
    for g in GAMES:
        a = mean(score(os.path.join(t, f"sleep-s{s}", f"{g}.json"), chosen) for s in SEEDS)
        c = mean(score(os.path.join(t, f"control-s{s}", f"{g}.json"), 1) for s in SEEDS)
        loss = (c - a) / abs(c) if c else 0.0
        ok = ok and loss <= v1.ARENA_LOSS
        arena.append(f"{g} {a:.2f} vs {c:.2f} ({-loss:+.1%})")
    verdict["H2"] = ok
    lines.append(f"- **H2, the games do not lose** (N = {chosen}, sleep vs control, at most {v1.ARENA_LOSS:.0%} under): " + "; ".join(arena) + f": **{'holds' if ok else 'not shown'}**.")
    a = mean(v1.typed_accuracy(os.path.join(t, f"sleep-s{s}", "typed.md")) for s in SEEDS)
    c = mean(v1.typed_accuracy(os.path.join(t, f"control-s{s}", "typed.md")) for s in SEEDS)
    verdict["H3"] = a >= c - v1.TYPED_LOSS
    lines.append(f"- **H3, the language tasks do not lose**: typed_decisions_bench {a:.4f} vs {c:.4f} for the control, at most {v1.TYPED_LOSS} under: **{'holds' if verdict['H3'] else 'not shown'}**.")
    return verdict, lines


# ----------------------------------------------------------------------------- gate


def nano_gate(ev):
    """Item 1 of v1 section 8, the s6-nano gate (docs/s6-gen-prereg.md section 6) on the exported
    seed-0 sleep model: typed_decisions_bench above 0.398 and ECE <= 0.08 (sky-eval of the GGUF,
    full depth, a zero memory), and no arena game more than 5% under s3-pico (`arena.json`:
    {game: {"model": score, "baseline": score}})."""
    bad = []
    e = load(os.path.join(ev, "eval-full.json"))
    acc = e["sources"].get("typed_decisions_bench", {}).get("accuracy")
    if acc is None or not acc > NANO_BENCH:
        bad.append(f"typed_decisions_bench {acc} <= {NANO_BENCH}")
    ece = e["sources"]["all"]["ece"]
    if not ece <= NANO_ECE:
        bad.append(f"ECE {ece:.4f} > {NANO_ECE}")
    arena = load(os.path.join(ev, "arena.json"))
    for g in GAMES:
        m, b = arena[g]["model"], arena[g]["baseline"]
        if b and (b - m) / abs(b) > NANO_ARENA:
            bad.append(f"{g} {m:.2f} more than 5% under s3-pico's {b:.2f}")
    return bad


def cmd_gate(root):
    """v2 section 7: the s6-nano gate, H0 to H3, and sky-infer's parity with the training code on
    the golden sequences, natively and in WASM (`<root>/gate/parity.json`)."""
    g = os.path.join(root, "gate")
    verdict = load(os.path.join(root, "verdict.json"))
    out = {"n": verdict["n"]}
    out["hypotheses"] = {h: verdict[h] for h in ("H0", "H1", "H2", "H3")}
    out["s6_nano_gate"] = nano_gate(g)
    parity = load(os.path.join(g, "parity.json"))
    out["parity"] = parity
    out["pass"] = all(out["hypotheses"].values()) and not out["s6_nano_gate"] and parity.get("native") is True and parity.get("wasm") is True
    out["published_as"] = "s6-sleep3-nano" if out["pass"] else None
    dump(out, os.path.join(root, "gate.json"))
    print(json.dumps(out))
    if not out["pass"]:
        sys.exit(1)


if __name__ == "__main__":
    a = sys.argv[1:]
    smoke = bool(a) and a[-1] == "smoke"
    if smoke:
        a = a[:-1]
    if len(a) in (4, 6) and a[0] == "c5":
        sys.exit(0 if cmd_c5(*a[1:]) else 1)
    if len(a) == 3 and a[0] == "ready":
        cmd_ready(a[1], a[2], smoke)
    elif len(a) == 2 and a[0] == "stop":
        cmd_stop(a[1])
    elif len(a) == 2 and a[0] == "pilot":
        cmd_pilot(a[1], smoke)
    elif len(a) == 2 and a[0] == "nano-ready":
        cmd_nano_ready(a[1])
    elif len(a) == 2 and a[0] == "choose":
        cmd_choose(a[1], smoke)
    elif len(a) == 2 and a[0] == "test-ready":
        cmd_test_ready(a[1])
    elif len(a) == 2 and a[0] == "report":
        cmd_report(a[1], smoke)
    elif len(a) == 2 and a[0] == "gate":
        cmd_gate(a[1])
    else:
        sys.exit(__doc__)
