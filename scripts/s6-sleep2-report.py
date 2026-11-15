#!/usr/bin/env python3
"""Applies the rules of docs/s6-sleep2-prereg.md to the outputs of scripts/train-sleep2.sh.

  s6-sleep2-report.py c4 <run> <zeroed run> <out.json>   C4 (section 3) on two s2-decide logs
  s6-sleep2-report.py checks <root>   C1 to C4 -> <root>/checks/verdict.json; fails unless they pass
  s6-sleep2-report.py stop <root>     the in-sample stop after M's phase A (section 5.1)
                                      -> <root>/pilot/stop.json; exit 3 when the pilot stops there
  s6-sleep2-report.py pilot <root> [smoke]   the decision rule of section 5.3 -> <root>/pilot.json
                                      (smoke: scripts/train-sleep2.sh SMOKE=1, any seeds, never
                                      under runs/s6-sleep2)

Every threshold below is the pre-registration's; nothing here is tuned on a result. The checks
read train seeds only; the pilot reads the dev2 seeds (the first 500 episode seeds = 19 mod 20
from 2,000,000,000), and only once C1 to C4 have passed.
"""
import json
import math
import os
import random
import sys

LN4 = math.log(4.0)
C4_CE = 1.30  # C4: the ask-frame cross-entropy over steps 250-300 is at most this
C4_GAP = 0.05  # ... and at least this much below the run with its read zeroed
C4_STEPS = (250, 300)
STOP_CE = LN4 - 0.05  # 1.336: section 5.1, the stop after phase A
STOP_LAST = 50  # ... over M's last 50 recall steps of phase A
GAIN = 0.10  # section 5.3 (1): M at N = 1 minus C
ZEROED = 0.03  # section 5.3 (2): M with its memory zeroed minus C, at most
DRAWS = 10_000
PASSES = (1, 2, 4)
BUCKETS = ("1-4", "5-8", "9-16", "17-32")
DEV2_SEED, DEV2_EPISODES = 2_000_000_000, 500
AMENDMENT_4 = (
    "C2 missed its bar by 0.010 (form frame 0.3900 against 0.40); by amendment 4 of "
    "docs/s6-sleep2-prereg.md, a stated deviation, form frame was taken and C4 decided. This is a "
    "threshold not applied after it was read, on a readiness check and not on a result."
)
DISCLOSURE = (
    "v2 was designed after reading v1's pilot: its logs, its dev scores (all at chance), its "
    "checkpoints and optimiser states, and its training sequences. The changes answer an "
    "optimisation failure seen in training and in the weights, not a test score. v1's test seeds "
    "of recall and rule110, and the arena seeds of this experiment, have never been read. v1's "
    "dev seeds were read; v2 decides on a fresh dev set (docs/s6-sleep2-prereg.md, section 10)."
)


def load(path):
    with open(path) as f:
        return json.load(f)


def log_lines(run):
    path = os.path.join(run, "log.jsonl")
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def recall_steps(run):
    """(step, ask-frame cross-entropy) of every recall sequence step of an s2-decide --seq-log run."""
    return [(r["step"], r["ask_ce"]) for r in log_lines(run) if r.get("seq_game") == "recall" and "ask_ce" in r]


def finished(run):
    return any(r.get("final") for r in log_lines(run))


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def c4_window(run):
    """The ask-frame cross-entropy averaged over the recall steps 250 to 300 (the mean of their
    per-step values) and how many steps that is."""
    lo, hi = C4_STEPS
    xs = [ce for step, ce in recall_steps(run) if lo <= step <= hi]
    return mean(xs), len(xs)


def cmd_c4(run, zeroed, out):
    for r in (run, zeroed):
        if not finished(r):
            sys.exit(f"{r}: the run did not end")
    m, n_m = c4_window(run)
    z, n_z = c4_window(zeroed)
    ok = m <= C4_CE and z - m >= C4_GAP
    res = {
        "memory": {"run": run, "ask_ce_250_300": m, "recall_steps": n_m},
        "zeroed": {"run": zeroed, "ask_ce_250_300": z, "recall_steps": n_z},
        "gap": z - m,
        "bars": {"ask_ce_at_most": C4_CE, "gap_at_least": C4_GAP, "steps": C4_STEPS},
        "pass": ok,
    }
    with open(out, "w") as f:
        json.dump(res, f, indent=2)
    print(
        f"C4 {os.path.basename(os.path.dirname(out)) or ''}: ask-frame CE over steps {C4_STEPS[0]}-{C4_STEPS[1]} "
        f"{m:.4f} ({n_m} recall steps) vs zeroed read {z:.4f} ({n_z}): gap {z - m:+.4f}; "
        f"needs <= {C4_CE} and a gap >= {C4_GAP}: {'PASS' if ok else 'FAIL'}"
    )
    return ok


def cmd_checks(root):
    """C1 to C4 -> verdict.json. The pilot runs only on a verdict that passes without amendment."""
    c = os.path.join(root, "checks")
    v = {"pass": False, "needs_amendment": False}
    c1, c2, c3 = (load(os.path.join(c, f"c{i}.json")) for i in (1, 2, 3))
    v["c1"], v["c2"] = c1["pass"], c2["pass"]
    v["form"] = c2["adopted"]
    if not c2["pass"]:
        # Amendment 4: C2 missed its bar; form frame is taken and C4 decides (a deviation).
        v["form"], v["c2_deviation"] = "frame", AMENDMENT_4
    # C3 is a sanity check since amendment 1: no memory weight but the read-out has a gradient.
    v["c3"] = bool(c3["forms"][v["form"]]["other_mem_exactly_zero"])
    c4 = os.path.join(c, "c4", "c4.json")
    v["c4"] = load(c4)["pass"] if os.path.exists(c4) else None
    v["mem_lr_mult"], v["mem_float"] = 10, False
    retries = {}
    for name, setting in (("x30", {"mem_lr_mult": 30}), ("float", {"mem_float": True})):
        path = os.path.join(c, f"c4-{name}", "c4.json")
        if os.path.exists(path):
            retries[name] = load(path)["pass"]
    v["c4_retries"] = retries
    if v["c4"] is False:
        passed = [n for n, ok in retries.items() if ok]
        if len(retries) == 2 and len(passed) == 1:
            # Section 3: "If either passes, that setting replaces M3 or O5 by amendment, before
            # any GPU run."
            v.update({"x30": {"mem_lr_mult": 30}, "float": {"mem_float": True}}[passed[0]])
            v["c4_passed_with"] = passed[0]
            v["needs_amendment"] = True
        elif len(passed) == 2:
            # Both retries passed: the text does not say which setting is taken; an amendment does.
            v["c4_passed_with"] = passed
            v["needs_amendment"] = True
    v["pass"] = bool(v["c1"] and (v["c2"] or v.get("c2_deviation")) and v["c3"] and (v["c4"] or v.get("c4_passed_with")))
    with open(os.path.join(c, "verdict.json"), "w") as f:
        json.dump(v, f, indent=2)
    print(json.dumps(v))
    if not v["pass"]:
        sys.exit("the checks do not pass: no GPU run (docs/s6-sleep2-prereg.md, sections 3 and 11)")


def cmd_stop(root):
    """Section 5.1: M's ask-frame cross-entropy over its last 50 recall steps of phase A."""
    run = os.path.join(root, "pilot", "m-a")
    if not finished(run):
        sys.exit(f"{run}: phase A did not end")
    steps = recall_steps(run)
    last = [ce for _, ce in steps[-STOP_LAST:]]
    ce = mean(last)
    stop = not (ce <= STOP_CE)
    res = {"ask_ce_last_50": ce, "recall_steps": len(last), "bar": STOP_CE, "stop": stop}
    with open(os.path.join(root, "pilot", "stop.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(
        f"phase A of M: ask-frame CE over its last {len(last)} recall steps {ce:.4f}, "
        f"the bar {STOP_CE:.4f}: {'STOP, the pilot failed in-sample' if stop else 'go on'}"
    )
    sys.exit(3 if stop else 0)


def reading(path, passes):
    for r in load(path):
        if r["passes"] == passes or not r["memory"]:
            return r
    raise SystemExit(f"{path}: no reading at N = {passes}")


def check_dev2(r, path):
    if r["from_seed"] != DEV2_SEED or r["n_episodes"] != DEV2_EPISODES:
        raise SystemExit(f"{path}: not the dev2 seeds")
    if [e["seed"] for e in r["episodes"]] != dev2_seeds():
        raise SystemExit(f"{path}: episodes other than the dev2 seeds")


def dev2_seeds():
    out, s = [], DEV2_SEED
    while len(out) < DEV2_EPISODES:
        if s % 20 == 19:
            out.append(s)
        s += 1
    return out


def paired_bootstrap(pairs, draws=DRAWS, seed=0):
    """`pairs`: (right_a, right_b, asked) per episode. The difference of accuracies a - b and its
    95% interval over resampled episodes (as scripts/s6-sleep-report.py)."""
    asked = sum(p[2] for p in pairs)
    point = (sum(p[0] for p in pairs) - sum(p[1] for p in pairs)) / max(asked, 1)
    rng = random.Random(seed)
    n = len(pairs)
    diffs = []
    for _ in range(draws):
        a = b = q = 0
        for i in rng.choices(range(n), k=n):
            a += pairs[i][0]
            b += pairs[i][1]
            q += pairs[i][2]
        diffs.append((a - b) / max(q, 1))
    diffs.sort()
    return point, diffs[int(0.025 * draws)], diffs[int(0.975 * draws) - 1]


def pairs_of(a, b):
    ea, eb = a["episodes"], b["episodes"]
    if [e["seed"] for e in ea] != [e["seed"] for e in eb]:
        raise SystemExit("the two readings do not play the same episodes")
    return [(x["score"], y["score"], x["max_score"]) for x, y in zip(ea, eb)]


def decide(gain, lo, zeroed_gain):
    """Section 5.3: the verdict from (1) the gain over C and its lower bound, (2) the zeroed gain."""
    one = gain >= GAIN and lo > 0
    two = zeroed_gain <= ZEROED
    if one and two:
        return "works", "the nano runs of section 6 take place"
    if one:
        return "defect", "no nano run: the reading is a defect to explain in an amendment before anything else runs"
    return "fails", "section 11: the s6-sleep fast-weight line stops"


def cmd_pilot(root, smoke=False):
    if smoke and os.path.normpath(root).startswith(os.path.join("runs", "s6-sleep2")):
        sys.exit(f"{root}: a smoke never writes under runs/s6-sleep2")
    p = os.path.join(root, "pilot")
    stop = load(os.path.join(p, "stop.json"))
    checks = load(os.path.join(root, "checks", "verdict.json"))
    out = {"disclosure": DISCLOSURE, "checks": checks, "phase_a": stop}
    if checks.get("c2_deviation"):
        out["c2_deviation"] = checks["c2_deviation"]
        print(checks["c2_deviation"])
    if stop["stop"] and not smoke:  # a smoke goes on, to check the code of the rule
        out.update(verdict="stopped", next="the pilot failed in-sample after phase A (section 5.1); section 11: the fast-weight line stops")
    else:
        dev = lambda run, game: os.path.join(p, run, "dev2", f"{game}.json")
        m1 = reading(dev("m", "recall"), 1)
        c = reading(dev("c", "recall"), 1)
        off = reading(dev("m", "recall-off"), 1)
        for r, path in ((m1, dev("m", "recall")), (c, dev("c", "recall")), (off, dev("m", "recall-off"))):
            if not smoke:
                check_dev2(r, path)
        if not off.get("memory_off"):
            raise SystemExit("the zeroed reading was not taken with --memory-off")
        gain, lo, hi = paired_bootstrap(pairs_of(m1, c))
        zeroed_gain = off["score"] - c["score"]
        verdict, nxt = decide(gain, lo, zeroed_gain)
        out.update(
            verdict=verdict,
            next=nxt,
            rule={
                "m_n1_minus_c": gain,
                "ci95": [lo, hi],
                "needs": f">= {GAIN} with the lower bound above 0",
                "m_zeroed_minus_c": zeroed_gain,
                "zeroed_needs": f"<= {ZEROED}",
            },
            reported={
                "recall": {
                    "c": c["score"],
                    **{f"m_n{n}": reading(dev("m", "recall"), n)["score"] for n in PASSES},
                    "m_zeroed_n1": off["score"],
                    "by_delay": {
                        "c": dict(zip(BUCKETS, c["buckets"])),
                        **{f"m_n{n}": dict(zip(BUCKETS, reading(dev("m", "recall"), n)["buckets"])) for n in PASSES},
                        "m_zeroed_n1": dict(zip(BUCKETS, off["buckets"])),
                    },
                },
                "rule110": {
                    "c": reading(dev("c", "rule110"), 1)["score"],
                    **{f"m_n{n}": reading(dev("m", "rule110"), n)["score"] for n in PASSES},
                },
            },
        )
    out["runs"] = {}
    for run in ("m-a", "m-b", "c-a", "c-b"):
        state = os.path.join(p, run, "last", "state.json")
        if os.path.exists(state):
            s = load(state)
            out["runs"][run] = {"steps": s["step"], "minutes": round(s["elapsed_secs"] / 60.0, 2)}
    with open(os.path.join(root, "pilot.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps({k: out[k] for k in ("verdict", "next")}), json.dumps(out.get("rule", {})))


if __name__ == "__main__":
    cmds = {"checks": cmd_checks, "stop": cmd_stop, "pilot": cmd_pilot}
    if len(sys.argv) == 5 and sys.argv[1] == "c4":
        sys.exit(0 if cmd_c4(*sys.argv[2:]) else 1)
    if len(sys.argv) == 4 and sys.argv[1] == "pilot" and sys.argv[3] == "smoke":
        sys.exit(cmd_pilot(sys.argv[2], smoke=True))
    if len(sys.argv) != 3 or sys.argv[1] not in cmds:
        sys.exit(__doc__)
    cmds[sys.argv[1]](sys.argv[2])
