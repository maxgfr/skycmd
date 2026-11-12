#!/usr/bin/env python3
"""Applies the rules of docs/s6-sleep-prereg.md to the runs of scripts/train-sleep.sh.

  s6-sleep-report.py pilot  <root>   the three pilot rules (section 5) -> <root>/pilot.json
  s6-sleep-report.py form   <root>   prints the s2-decide flags of the nano sleep arm, or fails
                                     when the pilot says the memory does not work
  s6-sleep-report.py choose <root>   the smallest N within 0.01 of the best (section 6) -> <root>/n.json
  s6-sleep-report.py report <root>   H0 to H3 on the test seeds (section 7) -> <root>/REPORT.md

Every threshold below is the pre-registration's; nothing here is tuned on a result.
"""
import json
import os
import random
import sys

MEMORY_GAIN = 0.10  # H0 and the pilot's "the memory works"
SHORTCUT_LOSS = 0.01  # a shortcut is kept if it loses less than this to the paper's way
N_TOLERANCE = 0.01  # choice of N
ARENA_LOSS = 0.02  # H2
TYPED_LOSS = 0.01  # H3
DRAWS = 10_000
PASSES = (1, 2, 4)
GAMES = ("flappy", "rescue", "dodge", "doom")
SEEDS = (0, 1, 2)
BUCKETS = ("1-4", "5-8", "9-16", "17-32")


def load(path):
    with open(path) as f:
        return json.load(f)


def reading(path, passes):
    """The report of one probe file at `passes` (a model without memory has one reading)."""
    reports = load(path)
    for r in reports:
        if r["passes"] == passes or not r["memory"]:
            return r
    raise SystemExit(f"{path}: no reading at N = {passes}")


def score(path, passes):
    return reading(path, passes)["score"]


def run_steps(run):
    return load(os.path.join(run, "last", "state.json"))["step"]


def run_minutes(run):
    return load(os.path.join(run, "last", "state.json"))["elapsed_secs"] / 60.0


def pilot_rules(p1_n1, p1_n4, p2_n4, p3_n4, p4):
    """Section 5, on the dev seeds of recall."""
    return {
        "memory_works": p1_n1 - p4 >= MEMORY_GAIN,
        "truncation_kept": p2_n4 >= p3_n4 - SHORTCUT_LOSS,
        "drawn_n_kept": p1_n4 >= p2_n4 - SHORTCUT_LOSS,
    }


def form_flags(rules):
    """The s2-decide flags of the nano sleep arm; None when no nano run takes place."""
    if not rules["memory_works"]:
        return None
    flags = []
    if not rules["truncation_kept"]:
        flags.append("--sleep-full")
    if not rules["drawn_n_kept"]:
        flags += ["--sleep-passes", "4"]
    return flags


def choose_n(means):
    """The smallest N within N_TOLERANCE of the best mean (`means`: {N: score})."""
    best = max(means.values())
    return min(n for n, s in means.items() if s >= best - N_TOLERANCE)


def paired_bootstrap(pairs, draws=DRAWS, seed=0):
    """`pairs`: (right_a, right_b, asked) per episode. The difference of accuracies a - b (sums of
    right answers over sums of questions) and its 95% interval over resampled episodes."""
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


def pooled_pairs(root, game, arm_a, n_a, arm_b, n_b):
    """Episodes of the 3 training seeds, paired by (training seed, episode seed)."""
    pairs = []
    for s in SEEDS:
        a = reading(os.path.join(root, "test", f"{arm_a}-s{s}", f"{game}.json"), n_a)["episodes"]
        b = reading(os.path.join(root, "test", f"{arm_b}-s{s}", f"{game}.json"), n_b)["episodes"]
        if [e["seed"] for e in a] != [e["seed"] for e in b]:
            raise SystemExit(f"{game}, seed {s}: the two readings do not play the same episodes")
        pairs += [(x["score"], y["score"], x["max_score"]) for x, y in zip(a, b)]
    return pairs


def typed_accuracy(path):
    """Last-exit accuracy of the ternary row in an s2-eval report."""
    for line in open(path):
        cells = [c.strip() for c in line.split("|")]
        if len(cells) > 3 and cells[1] == "ternary (fake-quant)":
            return float(cells[2])
    raise SystemExit(f"{path}: no ternary row")


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs)


def cmd_pilot(root):
    p = os.path.join(root, "pilot")
    rec = lambda run, n: score(os.path.join(p, run, "dev", "recall.json"), n)
    rules = pilot_rules(rec("p1", 1), rec("p1", 4), rec("p2", 4), rec("p3", 4), rec("p4", 1))
    out = {
        "rules": rules,
        "nano_flags": form_flags(rules),
        "runs": {
            run: {
                "steps": run_steps(os.path.join(p, run)),
                "minutes": round(run_minutes(os.path.join(p, run)), 2),
                **{
                    game: {str(n): score(os.path.join(p, run, "dev", f"{game}.json"), n) for n in PASSES}
                    for game in ("recall", "rule110")
                },
            }
            for run in ("p1", "p2", "p3", "p4")
        },
    }
    with open(os.path.join(root, "pilot.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out["rules"]), "nano flags:", out["nano_flags"])


def cmd_form(root):
    flags = load(os.path.join(root, "pilot.json"))["nano_flags"]
    if flags is None:
        sys.exit("the pilot says the memory does not work: no nano run (docs/s6-sleep-prereg.md, section 5)")
    print(" ".join(flags))


def cmd_choose(root):
    means = {
        n: mean(score(os.path.join(root, "nano", f"sleep-s{s}", "dev", "recall.json"), n) for s in SEEDS)
        for n in PASSES
    }
    out = {"n": choose_n(means), "dev_recall": {str(n): m for n, m in means.items()}}
    with open(os.path.join(root, "n.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out))


def cmd_report(root):
    t = os.path.join(root, "test")
    chosen = load(os.path.join(root, "n.json"))["n"]
    pilot = load(os.path.join(root, "pilot.json"))
    lines = ["# s6-sleep-nano: report (docs/s6-sleep-prereg.md)", ""]
    lines += ["## Pilot (s6-pico, dev seeds)", "", "| run | steps | working min | recall N=1 | N=2 | N=4 | rule110 N=1 | N=2 | N=4 |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for run, r in pilot["runs"].items():
        cells = [f"{r[g][str(n)]:.4f}" for g in ("recall", "rule110") for n in PASSES]
        lines.append(f"| {run} | {r['steps']} | {r['minutes']} | " + " | ".join(cells) + " |")
    lines += ["", f"Rules: {json.dumps(pilot['rules'])}. Flags of the nano sleep arm: `{' '.join(pilot['nano_flags'] or []) or 'none'}`.", ""]
    lines += ["## Nano runs", "", "| run | steps | working min |", "|---|---:|---:|"]
    for arm in ("sleep", "control"):
        for s in SEEDS:
            run = os.path.join(root, "nano", f"{arm}-s{s}")
            lines.append(f"| {arm}-s{s} | {run_steps(run)} | {run_minutes(run):.1f} |")
    lines += ["", f"Chosen N (dev seeds of recall): **{chosen}**.", ""]

    verdict = {}
    lines += ["## Probes (test seeds, mean of 3 training seeds)", "", "| probe | control | N=1 | N=2 | N=4 |", "|---|---:|---:|---:|---:|"]
    for game in ("recall", "rule110"):
        ctl = mean(score(os.path.join(t, f"control-s{s}", f"{game}.json"), 1) for s in SEEDS)
        by_n = [mean(score(os.path.join(t, f"sleep-s{s}", f"{game}.json"), n) for s in SEEDS) for n in PASSES]
        lines.append(f"| {game} | {ctl:.4f} | " + " | ".join(f"{x:.4f}" for x in by_n) + " |")
    lines += ["", "Recall by delay (frames between the show and the ask):", "", "| N | " + " | ".join(BUCKETS) + " |", "|---|" + "---:|" * len(BUCKETS)]
    for n in PASSES:
        b = [mean(reading(os.path.join(t, f"sleep-s{s}", "recall.json"), n)["buckets"][i] for s in SEEDS) for i in range(len(BUCKETS))]
        lines.append(f"| {n} | " + " | ".join(f"{x:.4f}" for x in b) + " |")

    lines += ["", "## Hypotheses", ""]
    d, lo, hi = paired_bootstrap(pooled_pairs(root, "recall", "sleep", 1, "control", 1))
    verdict["H0"] = d >= MEMORY_GAIN
    lines.append(f"- **H0, the memory is used**: recall at N = 1 minus the control = {d:+.4f} (95% interval {lo:+.4f} to {hi:+.4f}), needs at least {MEMORY_GAIN}: **{'holds' if verdict['H0'] else 'not shown'}**.")
    d, lo, hi = paired_bootstrap(pooled_pairs(root, "recall", "sleep", 4, "sleep", 1))
    verdict["H1"] = d > 0 and lo > 0
    lines.append(f"- **H1, sleep helps**: recall at N = 4 minus N = 1 = {d:+.4f} (95% interval {lo:+.4f} to {hi:+.4f}), needs a lower bound above 0: **{'holds' if verdict['H1'] else 'not shown'}**.")
    d, lo, hi = paired_bootstrap(pooled_pairs(root, "rule110", "sleep", 4, "sleep", 1))
    lines.append(f"- H1b (secondary, no gate): rule110 at N = 4 minus N = 1 = {d:+.4f} (95% interval {lo:+.4f} to {hi:+.4f}).")
    arena, ok = [], True
    for g in GAMES:
        a = mean(score(os.path.join(t, f"sleep-s{s}", f"{g}.json"), chosen) for s in SEEDS)
        c = mean(score(os.path.join(t, f"control-s{s}", f"{g}.json"), 1) for s in SEEDS)
        loss = (c - a) / abs(c) if c else 0.0
        ok = ok and loss <= ARENA_LOSS
        arena.append(f"{g} {a:.2f} vs {c:.2f} ({-loss:+.1%})")
    verdict["H2"] = ok
    lines.append(f"- **H2, the games do not lose** (N = {chosen}, sleep vs control, at most {ARENA_LOSS:.0%} under): " + "; ".join(arena) + f": **{'holds' if ok else 'not shown'}**.")
    a = mean(typed_accuracy(os.path.join(t, f"sleep-s{s}", "typed.md")) for s in SEEDS)
    c = mean(typed_accuracy(os.path.join(t, f"control-s{s}", "typed.md")) for s in SEEDS)
    verdict["H3"] = a >= c - TYPED_LOSS
    lines.append(f"- **H3, the language tasks do not lose**: typed_decisions_bench {a:.4f} vs {c:.4f} for the control, at most {TYPED_LOSS} under: **{'holds' if verdict['H3'] else 'not shown'}**.")
    lines += ["", f"Hypotheses for the publication gate (section 8, item 2): **{'all hold' if all(verdict.values()) else 'not all hold'}**. Items 1 and 3 of the gate (the s6-nano gate on the exported model, and the sky-infer parity) are read separately.", ""]
    with open(os.path.join(root, "REPORT.md"), "w") as f:
        f.write("\n".join(lines))
    with open(os.path.join(root, "verdict.json"), "w") as f:
        json.dump({"n": chosen, **verdict}, f, indent=2)
    print(json.dumps(verdict))


if __name__ == "__main__":
    cmds = {"pilot": cmd_pilot, "form": cmd_form, "choose": cmd_choose, "report": cmd_report}
    if len(sys.argv) != 3 or sys.argv[1] not in cmds:
        sys.exit(__doc__)
    cmds[sys.argv[1]](sys.argv[2])
