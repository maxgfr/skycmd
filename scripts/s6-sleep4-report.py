#!/usr/bin/env python3
"""Applies the rules of docs/s6-sleep4-prereg.md (v4 = docs/s6-sleep3-prereg.md, v3, trained to
the end of its learning curve, with checks at the nano size first) to the outputs of
scripts/train-sleep4.sh: the check K3 (a frozen base) and the frozen route that follows it.

  s6-sleep4-report.py k3 <root> [smoke]
        K3 (section 3.2) on <root>/stage1/k3-s0 and k3-s1: the windows of 50 recall steps, S90,
        S95 and K1's bar on both seeds (section 3.1) -> <root>/stage1/k3.json; then the rules of
        section 3.6 and, on the frozen route, the cap of section 4.2 -> <root>/stage1.json.
        Exit 1 when K3 fails (the frozen route is closed)
  s6-sleep4-report.py recipe <v3 cmd.txt> <v4 cmd.txt>
        the command of a check against v3's nano phase A (section 3.1): exit 1 unless they differ
        by the flags section 3.1 and 3.2 change only
  s6-sleep4-report.py nano-ready <root> [smoke]
        exit 1 unless stage1.json takes the frozen route; prints the --steps of the cap
  s6-sleep4-report.py phase-a <root> [smoke]
        after the three phase A runs: how each ended (the rule of section 4.2 or the cap), and
        v2's in-sample stop -> <root>/nano/phase-a.json; exit 3 when it stops v4
  s6-sleep4-report.py choose <root> [smoke]
        the dev3 readings: the zeroed parity against the base (exact; a difference stops), the
        choice of N (v1 section 6 on dev3) -> <root>/n.json
  s6-sleep4-report.py test-ready <root>
        exit 1 unless every stage-2 run ended, N is chosen and the test seeds were never read
  s6-sleep4-report.py report <root> [smoke]
        the test seeds read once: the parity checks of section 4.1 (a difference stops), H0 to H3
        and the goal (section 6) on the frozen route -> <root>/REPORT.md and <root>/verdict.json
  s6-sleep4-report.py gate <root>
        the gate of section 6 -> <root>/gate.json; exit 1 when it does not hold

Every threshold below is the pre-registration's; nothing here is tuned on a result. Only the
frozen route is read here: the other route (K1, K2, the confirmation, K5, the preserved route)
needs code that section 11 writes only if K3 fails. `smoke` is scripts/train-sleep4.sh SMOKE=1:
its own seeds and sizes, the rules recorded and not obeyed, never under runs/s6-sleep4.
"""
import importlib.util
import json
import math
import os
import shlex
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def _module(name, file):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, file))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# v1's reader: its bootstrap, its choice of N and its bars of H0 to H3 (v4 section 6: unchanged).
# v2's and v3's readers are not used: their seeds are not v4's.
v1 = _module("s6_sleep_report", "s6-sleep-report.py")

CHANCE = 0.25
LN4 = math.log(4.0)
# Section 3.1, the readings of stage 1.
WINDOW = 50  # the windows of recall steps
S90_BAR, S90_WINDOW = 0.90, 50
S95_BAR, S95_WINDOW = 0.95, 100
K1_BAR, K1_LAST = 0.95, 100  # K1's bar: the last 100 recall steps, on both seeds
EXCESS_FLOOR = 0.02  # the excess ratio is not defined while 1-4 is within this of chance
K3_SEEDS = (0, 1)
K3_STEPS = 1310
K3_OFF_EVAL = 4
# Section 4.2, the cap of phase A: 1.5 x S95max recall steps, at 0.765 recall steps per step.
CAP_FACTOR = 1.5
RECALL_PER_STEP = 0.765  # 0.9 x 0.85
PHASE_A_ACC, PHASE_A_WINDOW = 0.98, 100
STOP_CE, STOP_LAST = LN4 - 0.05, 50  # v2's stop: 1.336 over the last 50 recall steps of phase A
PHASE_B_STEPS = 2000
SEEDS = (0, 1, 2)
PASSES = (1, 2, 4)
GAMES = ("flappy", "rescue", "dodge", "doom")
BUCKETS = ("1-4", "5-8", "9-16", "17-32")
# Section 5: the seeds of v4.
DEV3_SEED, DEV3_EPISODES = 3_000_000_000, 500
TEST_SEED, TEST_EPISODES = 6_000_000_000, 1000
ARENA_SEED, ARENA_EPISODES = 6000, 200
# Seeds that v4 never reads for a decision (section 5): any reading from one of them is refused.
# v3's test (recall and rule110, the arena), dev2, v1's dev, the smoke's, the browser's
# validation seeds of the Memory page and K5's check seeds.
REFUSED_SEEDS = {
    5_000_000_000: "v3's test seeds",
    5000: "v3's test arena",
    2_000_000_000: "dev2 (v3's dev seeds)",
    1_000_000_000: "v1's dev seeds",
    1000: "the smoke's",
    3000: "the smoke's",
    9000: "the smoke's",
    7_000_000_018: "the browser's validation seeds",
    8000: "K5's check seeds",
}
# Smoke: the seeds of scripts/train-sleep4.sh SMOKE=1 (none of v4's).
SMOKE_DEV, SMOKE_TEST, SMOKE_ARENA = 1000, 3000, 9000
GOAL = 0.90  # announced, not gated
# The s6-nano gate (docs/s6-gen-prereg.md section 6), item 1 of the gate of section 6.
NANO_BENCH, NANO_ECE, NANO_ARENA = 0.398, 0.08, 0.05
DISCLOSURE = (
    "Disclosure (docs/s6-sleep4-prereg.md section 8). v4 was designed after v3's test was read: "
    "every number of runs/s6-sleep3/REPORT.md, verdict.json and gate.json was seen, as were v3's "
    "training logs, its dev2 files and its configs. (1) The checks at the nano size come from the "
    "code of v3's script (C5 ran at the pico size only). (2) Phase A ended by an in-sample rule "
    "with a cap, and phase B in steps, come from the training logs (the curves still rising, the "
    "steps the minutes bought). (3) The four factors of K2 come from the code, the configs and the "
    "training logs. (4) The frozen base and the preservation term come from a test reading: H2's "
    "failure on v3's test arena (seeds from 5000) and the s6-nano gate's arena motivate them; v4 "
    "reads neither of v3's arenas for any decision, and its H2 is read on new seeds. (5) The memory "
    "zeroed is read and gated at the nano size because it was never read. (6) New dev and test "
    "seeds; the s6-nano gate's arena (sky-arena's seed 7, 30 episodes per game, against s3-pico) "
    "is the exception, kept with its bar as that gate's fixed protocol, and already read by every "
    "earlier gate, v3's included. (7) K1's bar (0.95) and the rule of phase A (0.98) were set after "
    "the pilot's in-sample 0.9994 and v3's nano curves (0.35 to 0.48) were seen; the goal of 0.90 "
    "after v3's test (0.4892) and the pilot's dev2 (1.0000). The disclosures of v3 section 5 and v2 "
    "section 10 stand."
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
    if smoke and os.path.normpath(root).startswith(os.path.join("runs", "s6-sleep4")):
        sys.exit(f"{root}: a smoke never writes under runs/s6-sleep4")


def ratio(a, b):
    return a / b if b else float("nan")


# ----------------------------------------------------------------------------- the logs


def log_lines(run):
    with open(os.path.join(run, "log.jsonl")) as f:
        return [json.loads(line) for line in f if line.strip()]


def final_line(run):
    """The final line of a run that ended (None for a run cut by --stop-after or not ended)."""
    finals = [r for r in log_lines(run) if r.get("final") and not r.get("ema")]
    return finals[-1] if finals else None


def recall_rows(lines):
    """The recall steps of a --seq-log log (section 3.1: a log line with seq_game recall and
    asks), in order; the n-th is recall step n (from 1)."""
    return [r for r in lines if r.get("seq_game") == "recall" and r.get("asks")]


def right(r):
    """The asks the last exit answered right at a recall step (ask_splits[0], or ask_acc x asks)."""
    sp = r.get("ask_splits")
    return sp[0] if sp else r["ask_acc"] * r["asks"]


def pooled(rows):
    """ask_acc pooled over the asks of `rows`."""
    return ratio(sum(right(r) for r in rows), sum(r["asks"] for r in rows))


def first_window(rows, width, bar):
    """The first recall step n >= width at which ask_acc pooled over recall steps n - width + 1
    to n is at least `bar` (S90: width 50, bar 0.90; S95: width 100, bar 0.95); None if never."""
    for n in range(width, len(rows) + 1):
        if pooled(rows[n - width : n]) >= bar:
            return n
    return None


def bucket_acc(rows, j):
    """Accuracy at delay bucket j (0: 1-4, ..., 3: 17-32) pooled over `rows`, from ask_splits
    ([right, (right, asks) per bucket, (right, asks) one colour, (right, asks) several])."""
    got = sum(r["ask_splits"][1 + 2 * j] for r in rows if r.get("ask_splits"))
    asks = sum(r["ask_splits"][2 + 2 * j] for r in rows if r.get("ask_splits"))
    return ratio(got, asks), asks


def excess_ratio(rows):
    """(accuracy at 17-32 - 0.25) / (accuracy at 1-4 - 0.25); None while 1-4 is within 0.02 of
    chance (section 3.1)."""
    short, _ = bucket_acc(rows, 0)
    long_, _ = bucket_acc(rows, 3)
    if math.isnan(short) or math.isnan(long_) or abs(short - CHANCE) <= EXCESS_FLOOR:
        return None
    return (long_ - CHANCE) / (short - CHANCE)


def pooled_field(rows, key):
    """A per-step mean field (ask_ce, aux_acc, ...) pooled over the asks of the rows that hold it."""
    held = [r for r in rows if key in r]
    return ratio(sum(r[key] * r["asks"] for r in held), sum(r["asks"] for r in held))


def window_reading(rows, lo):
    """The readings of section 3.1 over one window of recall steps (`lo` the number of its first)."""
    off = [r for r in rows if "off" in r]
    layers = [r["aux_acc_layers"] for r in rows if "aux_acc_layers" in r]
    per_layer = None
    if layers:
        asks = [r["asks"] for r in rows if "aux_acc_layers" in r]
        per_layer = [ratio(sum(x[l] * a for x, a in zip(layers, asks)), sum(asks)) for l in range(len(layers[0]))]
    norms = [r["grad_norm"] for r in rows if isinstance(r.get("grad_norm"), (int, float)) and not math.isnan(r["grad_norm"])]
    return {
        "recall_steps": [lo, lo + len(rows) - 1],
        "asks": sum(r["asks"] for r in rows),
        "ask_acc": pooled(rows),
        "ask_ce": pooled_field(rows, "ask_ce"),
        "aux_acc": pooled_field(rows, "aux_acc"),
        "aux_acc_layers": per_layer,
        "by_delay": {b: bucket_acc(rows, j)[0] for j, b in enumerate(BUCKETS)},
        "excess_ratio": excess_ratio(rows),
        "off_steps": len(off),
        "off_ask_acc": ratio(sum(r["off"]["ask_acc"] * r["asks"] for r in off), sum(r["asks"] for r in off)),
        "off_ask_ce": ratio(sum(r["off"]["ask_ce"] * r["asks"] for r in off), sum(r["asks"] for r in off)),
        "grad_norm_median": statistics.median(norms) if norms else float("nan"),
    }


def stage1_reading(run, require_final=True):
    """Section 3.1 on one run of stage 1: its windows of 50 recall steps, S90, S95 and K1's bar
    over its last 100 recall steps. A run cut by --stop-after has no final line: K3's runs must
    have one (`require_final`), K2's need not."""
    lines = log_lines(run)
    fin = final_line(run)
    if require_final and fin is None:
        sys.exit(f"{run}: the run did not end (no final line)")
    rows = recall_rows(lines)
    windows = [window_reading(rows[i : i + WINDOW], i + 1) for i in range(0, len(rows), WINDOW)]
    last = rows[-K1_LAST:]
    last_acc = pooled(last) if len(rows) >= K1_LAST else float("nan")
    return {
        "run": run,
        "steps": fin["step"] if fin else max((r["step"] for r in lines if "step" in r), default=0),
        "recall_steps": len(rows),
        "frozen_base_checked": bool(fin and fin.get("frozen_base_checked")),
        "windows": windows,
        "S90": first_window(rows, S90_WINDOW, S90_BAR),
        "S95": first_window(rows, S95_WINDOW, S95_BAR),
        "last_100_ask_acc": last_acc,
        "bar": not math.isnan(last_acc) and last_acc >= K1_BAR,
    }


# ----------------------------------------------------------------------------- the commands


def cmd_args(path):
    """The flags of a cmd.txt (as `printf %q` wrote it): {flag: value or True}, and the binary's
    words before the subcommand."""
    with open(path) as f:
        words = shlex.split(f.read())
    i = words.index("s2-decide")
    flags, k = {}, i + 1
    while k < len(words):
        w = words[k]
        if w.startswith("--"):
            if k + 1 < len(words) and not words[k + 1].startswith("--"):
                flags[w] = words[k + 1]
                k += 2
                continue
            flags[w] = True
        k += 1
    return words[:i], flags


# What a check of stage 1 changes in v3's nano phase A (sections 3.1 and 3.2): its budget (steps,
# not minutes), the reading without the memory, the seed, the frozen base, where it writes, and
# the config's path (v3's file, copied next to the runs).
RECIPE_CHANGES = {"--minutes", "--steps", "--memory-off-eval", "--seed", "--freeze-base", "--out", "--config"}


def recipe_diff(v3_cmd, v4_cmd):
    """The flags whose value differs between v3's nano phase A and a v4 command, outside
    RECIPE_CHANGES (empty when v4 is v3's recipe)."""
    _, a = cmd_args(v3_cmd)
    _, b = cmd_args(v4_cmd)
    return sorted(k for k in set(a) | set(b) if k not in RECIPE_CHANGES and a.get(k) != b.get(k))


def cmd_recipe(v3_cmd, v4_cmd):
    diff = recipe_diff(v3_cmd, v4_cmd)
    if diff:
        sys.exit(f"{v4_cmd} is not v3's recipe ({v3_cmd}): {diff} differ")
    print(f"{v4_cmd}: v3's nano phase A, but for {sorted(RECIPE_CHANGES)}")


def check_k3_command(run, seed, smoke=False):
    """The run is K3's: a frozen base, the reading without the memory one step in four, its seed
    and, but in a smoke, 1,310 steps (section 3.2)."""
    path = os.path.join(run, "cmd.txt")
    if not os.path.exists(path):
        sys.exit(f"{path}: no command next to the run")
    _, f = cmd_args(path)
    want = {"--freeze-base": True, "--memory-off-eval": str(K3_OFF_EVAL), "--seed": str(seed)}
    if not smoke:
        want["--steps"] = str(K3_STEPS)
    bad = {k: (f.get(k), v) for k, v in want.items() if f.get(k) != v}
    if "--stop-after" in f:
        bad["--stop-after"] = (f["--stop-after"], None)
    if bad:
        sys.exit(f"{path}: not K3's command: {bad}")


# ----------------------------------------------------------------------------- K3


def cap_of(s95s):
    """Section 4.2: S95max, the cap 1.5 x S95max recall steps and its --steps,
    ceil(1.5 x S95max / 0.765) = ceil(100 x S95max / 51), in exact integers."""
    s95max = max(s95s)
    return {
        "S95max": s95max,
        "cap_recall_steps": CAP_FACTOR * s95max,
        "steps": -((-100 * s95max) // 51),
    }


def k3_verdict(seeds):
    """Section 3.6, rules 1 and 2, on K3's two readings: the frozen route when K1's bar holds on
    both seeds (a pass on one seed only is a failure)."""
    ok = all(s["bar"] for s in seeds.values())
    out = {"k3_pass": ok}
    if ok:
        out["route"] = "frozen"
        out["cap"] = cap_of([s["S95"] for s in seeds.values()])
        out["next"] = (
            "the frozen route: stage 2 trains v3's recipe with the base frozen, the control is the "
            "base file, the cap of phase A comes from K3's two S95; K1, K2, the confirmation and K5 "
            "are not run (section 3.6, rule 1)"
        )
    else:
        out["route"] = "closed"
        out["next"] = (
            "the frozen route is closed for v4: K1 (seed 0, then seed 1), then K2 (i) to (iv) on "
            "seed 0, then the confirmation if a factor is adopted (section 3.6, rule 2)"
        )
    return out


def cmd_k3(root, smoke=False):
    refuse_smoke_root(root, smoke)
    st = os.path.join(root, "stage1")
    seeds = {}
    for s in K3_SEEDS:
        run = os.path.join(st, f"k3-s{s}")
        check_k3_command(run, s, smoke)
        r = stage1_reading(run)
        if not r["frozen_base_checked"]:
            sys.exit(f"{run}: the final line does not say the frozen base was checked")
        seeds[f"seed{s}"] = r
    v = k3_verdict(seeds)
    res = {
        "check": "K3, a frozen base (docs/s6-sleep4-prereg.md section 3.2)",
        "bar": f"ask_acc pooled over the last {K1_LAST} recall steps >= {K1_BAR}, on both seeds",
        "reading": "each step's own training batch, before its update; windows of 50 recall steps pooled over their asks",
        "seeds": seeds,
        "pass": v["k3_pass"],
    }
    dump(res, os.path.join(st, "k3.json"))
    dump({"k3": {k: {"S90": x["S90"], "S95": x["S95"], "last_100_ask_acc": x["last_100_ask_acc"],
                     "bar": x["bar"], "recall_steps": x["recall_steps"]} for k, x in seeds.items()},
          **v}, os.path.join(root, "stage1.json"))
    for name, x in seeds.items():
        print(f"K3 {name}: {x['steps']} steps, {x['recall_steps']} recall steps | last 100: "
              f"{x['last_100_ask_acc']:.4f} [{'ok' if x['bar'] else 'FAIL'}] | S90 {x['S90']} | S95 {x['S95']}")
        print("  recall steps | ask_acc | ask_ce | aux_acc (per layer) | 1-4 | 17-32 | excess | off acc | grad_norm")
        for w in x["windows"]:
            layers = "/".join(f"{a:.3f}" for a in (w["aux_acc_layers"] or []))
            ex = "n/a" if w["excess_ratio"] is None else f"{w['excess_ratio']:.2f}"
            print(f"  {w['recall_steps'][0]:>4}-{w['recall_steps'][1]:<4} | {w['ask_acc']:.4f} | {w['ask_ce']:.4f} | "
                  f"{w['aux_acc']:.4f} ({layers}) | {w['by_delay']['1-4']:.3f} | {w['by_delay']['17-32']:.3f} | {ex} | "
                  f"{w['off_ask_acc']:.3f} | {w['grad_norm_median']:.2f}")
    print(f"K3: {'PASS' if v['k3_pass'] else 'FAIL'}; route: {v['route']}" + (f"; cap {v['cap']}" if "cap" in v else ""))
    return v["k3_pass"]


# ----------------------------------------------------------------------------- stage 2


def cmd_nano_ready(root, smoke=False):
    path = os.path.join(root, "stage1.json")
    if not os.path.exists(path):
        sys.exit(f"no {path}: K3 has not been read")
    s = load(path)
    if s.get("route") != "frozen":
        if smoke:
            print("SMOKE: stage 2 would be refused here (the frozen route is not taken)")
            return
        sys.exit(f"the route is {s.get('route')!r}: no stage-2 run of the frozen route (section 3.6)")
    print(s["cap"]["steps"])


def nano_run(root, seed, phase):
    return os.path.join(root, "nano", f"sleep-s{seed}", phase)


def phase_a_reading(run):
    """How phase A ended (section 4.2): by the rule or at the cap, its pooled ask_acc over its last
    100 recall steps, and v2's stop (the mean ask-frame CE of its last 50 recall steps)."""
    fin = final_line(run)
    if fin is None:
        sys.exit(f"{run}: phase A did not end")
    if "stopped_by_rule" not in fin:
        sys.exit(f"{run}: not run with the rule of section 4.2 (--stop-recall-acc)")
    rows = recall_rows(log_lines(run))
    ce = v1.mean(r["ask_ce"] for r in rows[-STOP_LAST:]) if rows else float("nan")
    return {
        "steps": fin["step"],
        "recall_steps": len(rows),
        "ended_by": "rule" if fin["stopped_by_rule"] else "cap",
        "last_100_ask_acc": pooled(rows[-PHASE_A_WINDOW:]) if rows else float("nan"),
        "ask_ce_last_50": ce,
        "v2_stop": not (ce <= STOP_CE),
        "frozen_base_checked": bool(fin.get("frozen_base_checked")),
    }


def cmd_phase_a(root, smoke=False):
    refuse_smoke_root(root, smoke)
    seeds = {f"seed{s}": phase_a_reading(nano_run(root, s, "a")) for s in SEEDS}
    stop = any(x["v2_stop"] for x in seeds.values())
    res = {
        "rule": f"phase A ends at the first recall step whose last {PHASE_A_WINDOW} recall steps pool to >= {PHASE_A_ACC}, or at the cap",
        "v2_stop": f"ask-frame CE over the last {STOP_LAST} recall steps above {STOP_CE:.4f}",
        "seeds": seeds,
        "stop": stop,
    }
    if stop:
        res["next"] = "v2's stop holds for a memory seed: no phase B runs and v4 reports a failure in-sample (section 4.2)"
    dump(res, os.path.join(root, "nano", "phase-a.json"))
    for k, x in seeds.items():
        print(f"phase A {k}: {x['steps']} steps, {x['recall_steps']} recall steps, ended by the {x['ended_by']}; "
              f"last 100: {x['last_100_ask_acc']:.4f}; ask CE last 50 {x['ask_ce_last_50']:.4f} "
              f"[{'STOP' if x['v2_stop'] else 'go on'}]")
    if stop and not smoke:
        sys.exit(3)


# ----------------------------------------------------------------------------- readings


def reading(path, passes):
    return v1.reading(path, passes)


def episode_seeds(start, n):
    out, s = [], start
    while len(out) < n:
        if s % 20 == 19:
            out.append(s)
        s += 1
    return out


def check_seeds(r, path, start, n):
    """A reading's seeds are v4's (section 5): `from_seed` and `n_episodes` as asked, never a
    seed v4 does not read, and its episodes, when it lists them, the first n = 19 mod 20."""
    if r["from_seed"] in REFUSED_SEEDS:
        raise SystemExit(f"{path}: from seed {r['from_seed']}, {REFUSED_SEEDS[r['from_seed']]}: v4 never reads them")
    if r["from_seed"] != start or r["n_episodes"] != n:
        raise SystemExit(f"{path}: from seed {r['from_seed']} x {r['n_episodes']}, not {start} x {n}")
    eps = r.get("episodes")
    if eps and [e["seed"] for e in eps] != episode_seeds(start, n):
        raise SystemExit(f"{path}: episodes other than the first {n} seeds = 19 mod 20 from {start}")


def check_file(path, start, n):
    """Every reading of a probe file (each N) on v4's seeds."""
    for r in load(path):
        check_seeds(r, path, start, n)


def answers(r):
    """The answers of a probe reading, episode by episode: (seed, score, the asks of recall)."""
    return [(e["seed"], e["score"], [tuple(a) for a in e.get("asks", [])]) for e in r["episodes"]]


def parity(memory_off_path, base_path, what):
    """Section 4.1: the memory model with its memory zeroed answers as the base, answer for
    answer; a difference is a bug that stops the run (SystemExit)."""
    m = reading(memory_off_path, 1)
    if not m.get("memory_off"):
        raise SystemExit(f"{memory_off_path}: not read with --memory-off")
    b = reading(base_path, 1)
    if b.get("memory"):
        raise SystemExit(f"{base_path}: the base has a memory")
    am, ab = answers(m), answers(b)
    if [x[0] for x in am] != [x[0] for x in ab]:
        raise SystemExit(f"{what}: the zeroed reading and the base's play other episodes")
    for x, y in zip(am, ab):
        if x != y:
            raise SystemExit(f"{what}: PARITY FAILS at episode {x[0]}: zeroed {x[1:]} against the base's {y[1:]} "
                             "(section 4.1: a difference is a bug that stops the run until it is explained)")
    if m["score"] != b["score"]:
        raise SystemExit(f"{what}: PARITY FAILS: score {m['score']} against the base's {b['score']}")
    return {"episodes": len(am), "score": m["score"], "equal": True}


def typed_body(path):
    """An s2-eval report without the lines that name the model and its checkpoint."""
    with open(path) as f:
        return [l for l in f.read().splitlines() if not (l.startswith("# Evaluation report") or l.startswith("Checkpoint:"))]


def typed_parity(memory_path, base_path, what):
    """Section 4.1 and H3 on the frozen route: typed_decisions_bench with an empty memory equals
    the base's exactly. s2-eval writes no per-question answer, so the check is the strongest the
    report allows: every line of the report (each exit and source: accuracy, NLL, ECE, the early
    exit, the permutation and coherence checks), for the float and the ternary rows, equal."""
    a, b = typed_body(memory_path), typed_body(base_path)
    if a != b:
        diff = next(i for i, (x, y) in enumerate(zip(a + [""], b + [""])) if x != y)
        raise SystemExit(f"{what}: PARITY FAILS: {memory_path} and {base_path} differ at line {diff}: "
                         f"{(a + [''])[diff]!r} against {(b + [''])[diff]!r} (section 4.1)")
    return {"lines": len(a), "accuracy": v1.typed_accuracy(memory_path), "equal": True}


def dev_dir(root, who):
    return os.path.join(root, "dev3", who)


def cmd_choose(root, smoke=False):
    """Section 5: the dev3 readings. The zeroed parity against the base (exact), then v1 section
    6's rule on dev3: the smallest N in {1, 2, 4} within 0.01 of the best mean of the three
    memory seeds."""
    refuse_smoke_root(root, smoke)
    start, n = (SMOKE_DEV, None) if smoke else (DEV3_SEED, DEV3_EPISODES)
    base = os.path.join(dev_dir(root, "base"), "recall.json")
    files = [base, os.path.join(dev_dir(root, "base"), "rule110.json")]
    for s in SEEDS:
        files += [os.path.join(dev_dir(root, f"sleep-s{s}"), f) for f in ("recall.json", "rule110.json", "recall-off.json")]
    for path in files:
        if not smoke:
            check_file(path, start, n)
    zeroed = {f"seed{s}": parity(os.path.join(dev_dir(root, f"sleep-s{s}"), "recall-off.json"), base, f"dev3, seed {s}")
              for s in SEEDS}
    means = {}
    for p in PASSES:
        means[p] = v1.mean(reading(os.path.join(dev_dir(root, f"sleep-s{s}"), "recall.json"), p)["score"] for s in SEEDS)
    out = {
        "n": v1.choose_n(means),
        "dev3_recall": {str(p): m for p, m in means.items()},
        "tolerance": v1.N_TOLERANCE,
        "base_recall": reading(base, 1)["score"],
        "zeroed_parity": zeroed,
    }
    dump(out, os.path.join(root, "n.json"))
    print(json.dumps({k: out[k] for k in ("n", "dev3_recall", "base_recall")}))


def stage2_runs(root):
    return [nano_run(root, s, ph) for s in SEEDS for ph in ("a", "b")]


def test_ready(root):
    """Section 10: no test seed before every stage-2 run has ended and N is chosen; read once."""
    why = []
    for run in stage2_runs(root):
        if not (os.path.exists(os.path.join(run, "log.jsonl")) and final_line(run)):
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
    print("every stage-2 run ended and N is chosen: the test seeds may be read, once")


def test_path(root, who, name):
    return os.path.join(root, "test", who, name)


def pairs_with_base(root, game, passes):
    """Each memory seed's episodes at N = `passes` paired with the base's reading of the same
    episodes (section 4.1: the base is read once and paired with each memory seed)."""
    b = reading(test_path(root, "base", f"{game}.json"), 1)["episodes"]
    pairs = []
    for s in SEEDS:
        a = reading(test_path(root, f"sleep-s{s}", f"{game}.json"), passes)["episodes"]
        if [e["seed"] for e in a] != [e["seed"] for e in b]:
            raise SystemExit(f"{game}, seed {s}: the memory model and the base do not play the same episodes")
        pairs += [(x["score"], y["score"], x["max_score"]) for x, y in zip(a, b)]
    return pairs


def pairs_between(root, game, n_a, n_b):
    """The memory seeds at N = n_a against themselves at N = n_b (H1), pooled over the seeds."""
    pairs = []
    for s in SEEDS:
        path = test_path(root, f"sleep-s{s}", f"{game}.json")
        a, b = reading(path, n_a)["episodes"], reading(path, n_b)["episodes"]
        pairs += [(x["score"], y["score"], x["max_score"]) for x, y in zip(a, b)]
    return pairs


def splits(episodes):
    """Recall by delay bucket and by one colour against several, pooled over episodes."""
    by = [[0, 0] for _ in BUCKETS]
    col = {"one": [0, 0], "several": [0, 0]}
    for e in episodes:
        shown = e.get("shown", [])
        for i, (b, ok) in enumerate(e.get("asks", [])):
            if 0 <= b < len(BUCKETS):
                by[b][0] += int(ok)
                by[b][1] += 1
            if i < len(shown) and shown[i] >= 1:
                k = "one" if shown[i] == 1 else "several"
                col[k][0] += int(ok)
                col[k][1] += 1
    return {"by_delay": {b: ratio(*p) for b, p in zip(BUCKETS, by)}, "by_colours": {k: ratio(*p) for k, p in col.items()}}


def hypotheses(root, chosen, smoke=False):
    """Section 6 on the frozen route: the parity checks first (a difference stops), then H0 to H3
    against the base, and the goal."""
    t = os.path.join(root, "test")
    start, n, astart, an = (SMOKE_TEST, None, SMOKE_ARENA, None) if smoke else (TEST_SEED, TEST_EPISODES, ARENA_SEED, ARENA_EPISODES)
    if not smoke:
        for who in ["base"] + [f"sleep-s{s}" for s in SEEDS]:
            for g in ("recall", "rule110"):
                check_file(test_path(root, who, f"{g}.json"), start, n)
            for g in GAMES:
                check_file(test_path(root, who, f"{g}.json"), astart, an)
            if who != "base":
                check_file(test_path(root, who, "recall-off.json"), start, n)
    verdict, lines = {}, []
    # Section 4.1: the two parity checks of the frozen route (Z, and H3's empty memory).
    zeroed = {f"seed{s}": parity(test_path(root, f"sleep-s{s}", "recall-off.json"), test_path(root, "base", "recall.json"),
                                 f"test, seed {s}") for s in SEEDS}
    typed = {f"seed{s}": typed_parity(test_path(root, f"sleep-s{s}", "typed.md"), test_path(root, "base", "typed.md"),
                                      f"typed_decisions_bench, seed {s}") for s in SEEDS}
    verdict["Z"] = True
    verdict["parity"] = {"recall_zeroed": zeroed, "typed_empty_memory": typed}
    base = lambda g: reading(test_path(root, "base", f"{g}.json"), 1)["score"]
    mean_n = lambda g, p: v1.mean(reading(test_path(root, f"sleep-s{s}", f"{g}.json"), p)["score"] for s in SEEDS)
    lines += ["## Probes (test seeds; the memory model: mean of 3 seeds; the control: the base file)", "",
              "| probe | base | N=1 | N=2 | N=4 | zeroed N=1 |", "|---|---:|---:|---:|---:|---:|"]
    zeroed_mean = v1.mean(reading(test_path(root, "sleep-s%d" % s, "recall-off.json"), 1)["score"] for s in SEEDS)
    for g in ("recall", "rule110"):
        cells = [f"{mean_n(g, p):.4f}" for p in PASSES]
        z = f"{zeroed_mean:.4f}" if g == "recall" else ""
        lines.append(f"| {g} | {base(g):.4f} | " + " | ".join(cells) + f" | {z} |")
    lines += ["", "Recall by delay and by the colours the asked door was shown with (pooled over the 3 seeds; reported, deciding nothing):", "",
              "| reading | " + " | ".join(BUCKETS) + " | one colour | several |", "|---|" + "---:|" * (len(BUCKETS) + 2)]
    rows = [("base", [reading(test_path(root, "base", "recall.json"), 1)])]
    rows += [(f"N = {p}", [reading(test_path(root, f"sleep-s{s}", "recall.json"), p) for s in SEEDS]) for p in PASSES]
    rows += [("N = 1, memory zeroed", [reading(test_path(root, f"sleep-s{s}", "recall-off.json"), 1) for s in SEEDS])]
    for label, rs in rows:
        sp = splits([e for r in rs for e in r["episodes"]])
        cells = [sp["by_delay"][b] for b in BUCKETS] + [sp["by_colours"]["one"], sp["by_colours"]["several"]]
        lines.append(f"| {label} | " + " | ".join(f"{x:.4f}" for x in cells) + " |")
    seen = ", ".join("%s: %d episodes" % (k, v["episodes"]) for k, v in zeroed.items())
    lines += ["", "## Parity checks (section 4.1)", "",
              f"- **Z, the memory zeroed** (recall at N = 1, `--memory-off`): equal to the base's reading, answer for answer, on every seed ({seen}).",
              f"- **typed_decisions_bench with an empty memory**: every line of the s2-eval report equal to the base's, on every seed (accuracy {typed['seed0']['accuracy']:.4f}). s2-eval writes no per-question answers: the check compares every number of its report.",
              "", "## Hypotheses (section 6)", ""]
    d, lo, hi = v1.paired_bootstrap(pairs_with_base(root, "recall", 1))
    verdict["H0"] = d >= v1.MEMORY_GAIN
    lines.append(f"- **H0, the memory is used**: recall at N = 1 minus the base = {d:+.4f} (95% interval {lo:+.4f} to {hi:+.4f}), needs at least {v1.MEMORY_GAIN}: **{'holds' if verdict['H0'] else 'not shown'}**.")
    d, lo, hi = v1.paired_bootstrap(pairs_between(root, "recall", 4, 1))
    verdict["H1"] = d > 0 and lo > 0
    lines.append(f"- **H1, sleep helps**: recall at N = 4 minus N = 1 = {d:+.4f} (95% interval {lo:+.4f} to {hi:+.4f}), needs a lower bound above 0: **{'holds' if verdict['H1'] else 'not shown'}**.")
    d, lo, hi = v1.paired_bootstrap(pairs_between(root, "rule110", 4, 1))
    lines.append(f"- H1b (secondary, no gate): rule110 at N = 4 minus N = 1 = {d:+.4f} (95% interval {lo:+.4f} to {hi:+.4f}).")
    arena, ok = [], True
    for g in GAMES:
        a, c = mean_n(g, chosen), base(g)
        loss = (c - a) / abs(c) if c else 0.0
        ok = ok and loss <= v1.ARENA_LOSS
        arena.append(f"{g} {a:.2f} vs {c:.2f} ({-loss:+.1%})")
    verdict["H2"] = ok
    lines.append(f"- **H2, the games do not lose** (N = {chosen}, the memory written as it plays, against the base file, at most {v1.ARENA_LOSS:.0%} under): " + "; ".join(arena) + f": **{'holds' if ok else 'not shown'}**.")
    a = v1.mean(v1.typed_accuracy(test_path(root, f"sleep-s{s}", "typed.md")) for s in SEEDS)
    c = v1.typed_accuracy(test_path(root, "base", "typed.md"))
    verdict["H3"] = a >= c - v1.TYPED_LOSS
    lines.append(f"- **H3, the language tasks do not lose**: typed_decisions_bench {a:.4f} vs {c:.4f} for the base, at most {v1.TYPED_LOSS} under (and equal, by the parity check): **{'holds' if verdict['H3'] else 'not shown'}**.")
    r1 = mean_n("recall", 1)
    verdict["goal"] = r1 >= GOAL
    lines.append(f"- The goal, announced and not gated: recall at N = 1 at least {GOAL}: {r1:.4f}, **{'reached' if verdict['goal'] else 'missed'}**.")
    return verdict, lines


def phase_report(root):
    """Reported, deciding nothing: how each phase A ended, and ask_acc over the first 25 recall
    steps of phase B against the last 50 of phase A (section 4.4)."""
    lines = ["## Stage 2 (the frozen route)", "", "| seed | phase A steps | recall steps | ended by | last 100 | phase B steps | A, last 50 | B, first 25 |", "|---|---:|---:|---|---:|---:|---:|---:|"]
    for s in SEEDS:
        a = phase_a_reading(nano_run(root, s, "a"))
        ra = recall_rows(log_lines(nano_run(root, s, "a")))
        rb = recall_rows(log_lines(nano_run(root, s, "b")))
        fb = final_line(nano_run(root, s, "b"))
        lines.append(f"| {s} | {a['steps']} | {a['recall_steps']} | {a['ended_by']} | {a['last_100_ask_acc']:.4f} | "
                     f"{fb['step'] if fb else 'n/a'} | {pooled(ra[-50:]):.4f} | {pooled(rb[:25]):.4f} |")
    return lines + [""]


def cmd_report(root, smoke=False):
    refuse_smoke_root(root, smoke)
    t = os.path.join(root, "test")
    if not os.path.exists(os.path.join(t, "done")):
        sys.exit(f"{t}/done is missing: the test reading did not end")
    n = load(os.path.join(root, "n.json"))
    stage1 = load(os.path.join(root, "stage1.json"))
    verdict, lines = hypotheses(root, n["n"], smoke)
    head = ["# s6-sleep4-nano: report (docs/s6-sleep4-prereg.md), the frozen route", "", DISCLOSURE, ""]
    head += [f"K3: **{'pass' if stage1['k3_pass'] else 'fail'}** ({json.dumps(stage1['k3'])}); route **{stage1['route']}**, cap {json.dumps(stage1.get('cap'))}.", ""]
    head += [f"Chosen N (dev3 recall, mean of 3 seeds): **{n['n']}** ({json.dumps(n['dev3_recall'])}; the base {n['base_recall']:.4f}).", ""]
    with open(os.path.join(root, "REPORT.md"), "w") as f:
        f.write("\n".join(head + phase_report(root) + lines) + "\n")
    dump({"n": n["n"], "route": "frozen", **verdict}, os.path.join(root, "verdict.json"))
    print(json.dumps({k: v for k, v in verdict.items() if k != "parity"}))


# ----------------------------------------------------------------------------- gate


def nano_gate(ev):
    """Item 1 of the gate, the s6-nano gate (docs/s6-gen-prereg.md section 6) on the exported
    seed-0 file: typed_decisions_bench above 0.398 and ECE <= 0.08 (sky-eval at full depth, a
    zero memory), no arena game more than 5% under s3-pico (`arena.json`)."""
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


def same_base_files(root):
    """The parity of the exported files and the checkpoints with the base (`s2-same-base`)."""
    out = {}
    for s in SEEDS:
        for kind in ("checkpoint", "gguf"):
            path = os.path.join(root, "parity", f"sleep-s{s}-{kind}.json")
            out[f"seed{s}-{kind}"] = load(path).get("same") is True if os.path.exists(path) else False
    path = os.path.join(root, "parity", "base-copy.json")
    out["base-copy"] = load(path).get("same") is True if os.path.exists(path) else False
    return out


def cmd_gate(root):
    """Section 6: the s6-nano gate on the exported seed-0 file, H0 to H3, Z (on the frozen route,
    the two parity checks) and sky-infer's parity on the golden memory sequences, natively and in
    WASM; and the exported file is the base's weights plus the memory (`s2-same-base`)."""
    g = os.path.join(root, "gate")
    verdict = load(os.path.join(root, "verdict.json"))
    out = {"n": verdict["n"], "route": verdict.get("route")}
    out["hypotheses"] = {h: verdict[h] for h in ("H0", "H1", "H2", "H3", "Z")}
    out["s6_nano_gate"] = nano_gate(g)
    out["parity"] = load(os.path.join(g, "parity.json"))
    out["same_base"] = same_base_files(root)
    out["pass"] = (all(out["hypotheses"].values()) and not out["s6_nano_gate"]
                   and out["parity"].get("native") is True and out["parity"].get("wasm") is True
                   and all(out["same_base"].values()))
    out["published_as"] = "s6-sleep4-nano" if out["pass"] else None
    dump(out, os.path.join(root, "gate.json"))
    print(json.dumps(out))
    if not out["pass"]:
        sys.exit(1)


if __name__ == "__main__":
    a = sys.argv[1:]
    smoke = bool(a) and a[-1] == "smoke"
    if smoke:
        a = a[:-1]
    if len(a) == 2 and a[0] == "k3":
        sys.exit(0 if cmd_k3(a[1], smoke) else 1)
    if len(a) == 3 and a[0] == "recipe":
        cmd_recipe(a[1], a[2])
    elif len(a) == 2 and a[0] == "nano-ready":
        cmd_nano_ready(a[1], smoke)
    elif len(a) == 2 and a[0] == "phase-a":
        cmd_phase_a(a[1], smoke)
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
