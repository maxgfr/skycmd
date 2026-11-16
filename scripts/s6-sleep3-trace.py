#!/usr/bin/env python3
"""Reads the training logs of the diagnosis of docs/s6-sleep3-diagnosis.md.

  s6-sleep3-trace.py table <run> [every]   the memory trace of an `s2-decide --seq-log
                                           --memory-trace` run, one line per sequence step
                                           (every: one step in that many after step 12)
  s6-sleep3-trace.py c4 <run>...           the reading of C4 (docs/s6-sleep2-prereg.md section 3)
                                           for each run: the ask-frame cross-entropy of recall
                                           over steps 250 to 300, and over earlier windows
  s6-sleep3-trace.py splits <run> [lo hi]  over the steps lo to hi (default: the last 51 steps
                                           of the run): the ask-frame cross-entropy and the
                                           accuracy of the last exit, with the memory and for
                                           the same weights without it (`--memory-off-eval`),
                                           and the accuracy by delay and by the colours the
                                           asked door had been shown with
  s6-sleep3-trace.py curve <run> [width]   the same readings window by window (default 50)

The runs are CPU trainings on recall's train sequences; no dev or test seed is read.
"""
import json
import math
import os
import sys

LN4 = math.log(4.0)
WINDOWS = ((250, 300), (150, 250), (50, 150))


def steps(run):
    """The sequence steps of a run's log.jsonl."""
    with open(os.path.join(run, "log.jsonl")) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return [r for r in rows if "seq_step_loss" in r]


def table(run, every):
    cols = (
        "step loss ask_ce | grad: out qk v beta | beta: mean max | alpha_min | "
        "at the asks: |S| |x| |r| |r.Wo| | key overlap"
    )
    print(cols)
    for r in steps(run):
        m = r.get("memory")
        if not m or (r["step"] > 12 and r["step"] % every):
            continue
        layer = m["layers"][0]
        grad = lambda part: m["grad"].get(f"mem.0.{part}.weight", float("nan"))
        print(
            f"{r['step']:4d} {r['seq_step_loss']:.4f} {r.get('ask_ce', float('nan')):.4f} | "
            f"{grad('out'):.1e} {grad('qk'):.1e} {grad('v'):.1e} {grad('beta'):.1e} | "
            f"{layer['beta_mean']:.1e} {layer['beta_max']:.1e} | {layer['alpha_min']:.4f} | "
            f"{layer['s_norm_ask']:.1e} {layer['x_norm_ask']:.0f} {layer['r_norm_ask']:.1e} "
            f"{layer['read_norm_ask']:.1e} | {layer.get('key_overlap', float('nan')):.3f}"
        )


def mean(rows, lo, hi, key):
    xs = [r[key] for r in rows if key in r and lo <= r["step"] <= hi]
    return sum(xs) / len(xs) if xs else float("nan")


BUCKETS = ("1-4", "5-8", "9-16", "17-32")


def pooled(rows, lo, hi, off):
    """Sums over the sequence steps lo..hi: with the memory, or (off) for the steps that were
    also answered without it."""
    out = {"asks": 0.0, "ce": 0.0, "ce_last": 0.0, "splits": [0.0] * 13, "steps": 0}
    for r in rows:
        if not (lo <= r["step"] <= hi) or "ask_splits" not in r:
            continue
        if off:
            if "off" not in r:
                continue
            src, splits = r["off"], r["off"]["ask_splits"]
        else:
            src, splits = r, r["ask_splits"]
        out["asks"] += r["asks"]
        out["ce"] += src["ask_ce"] * r["asks"]
        out["ce_last"] += src["ask_ce_last"] * r["asks"]
        out["splits"] = [a + b for a, b in zip(out["splits"], splits)]
        out["steps"] += 1
    return out


def ratio(a, b):
    return a / b if b else float("nan")


def reading(p):
    s = p["splits"]
    delays = [ratio(s[1 + 2 * j], s[2 + 2 * j]) for j in range(4)]
    return {
        "steps": p["steps"],
        "asks": int(p["asks"]),
        "ce": ratio(p["ce"], p["asks"]),
        "ce_last": ratio(p["ce_last"], p["asks"]),
        "acc": ratio(s[0], p["asks"]),
        "delays": delays,
        "delay_asks": [int(s[2 + 2 * j]) for j in range(4)],
        "one": ratio(s[9], s[10]),
        "several": ratio(s[11], s[12]),
        "one_asks": int(s[10]),
        "several_asks": int(s[12]),
    }


def splits(run, lo, hi):
    rows = steps(run)
    last = max((r["step"] for r in rows), default=0)
    if hi is None:
        lo, hi = last - 50, last
    print(f"{run}: sequence steps {lo} to {hi} (last step {last}); ln 4 = {LN4:.4f}, chance 0.25")
    for name, off in (("with the memory", False), ("same weights, no memory", True)):
        r = reading(pooled(rows, lo, hi, off))
        if not r["steps"]:
            print(f"  {name}: not logged")
            continue
        print(
            f"  {name}: {r['steps']} steps, {r['asks']} asks | ask CE {r['ce']:.4f} "
            f"(last exit {r['ce_last']:.4f}) | accuracy {r['acc']:.4f}"
        )
        by_delay = " | ".join(
            f"{b}: {a:.3f} ({n})" for b, a, n in zip(BUCKETS, r["delays"], r["delay_asks"])
        )
        print(f"    by delay: {by_delay}")
        print(
            f"    door shown with one colour: {r['one']:.3f} ({r['one_asks']}) | "
            f"with several: {r['several']:.3f} ({r['several_asks']})"
        )


def curve(run, width):
    rows = steps(run)
    last = max((r["step"] for r in rows), default=0)
    print("steps | ask CE, accuracy | no memory: ask CE, accuracy | aux accuracy | "
          "by delay | one colour, several")
    for lo in range(1, last + 1, width):
        hi = lo + width - 1
        on, off = reading(pooled(rows, lo, hi, False)), reading(pooled(rows, lo, hi, True))
        aux = mean(rows, lo, hi, "aux_acc")
        if not on["steps"]:
            print(f"{lo:4d}-{hi:<4d} | ask CE {mean(rows, lo, hi, 'ask_ce'):.4f} | "
                  f"aux {aux:.3f}")
            continue
        delays = " ".join(f"{a:.2f}" for a in on["delays"])
        print(
            f"{lo:4d}-{hi:<4d} | {on['ce']:.4f} {on['acc']:.3f} | {off['ce']:.4f} "
            f"{off['acc']:.3f} | {aux:.3f} | {delays} | {on['one']:.2f} {on['several']:.2f}"
        )


def c4(runs):
    print(f"ln 4 = {LN4:.4f}; ask-frame cross-entropy (deep supervision), mean over the steps of a window")
    for run in runs:
        rows = steps(run)
        last = max((r["step"] for r in rows), default=0)
        parts = [f"{lo}-{hi}: {mean(rows, lo, hi, 'ask_ce'):.4f}" for lo, hi in WINDOWS]
        aux = mean(rows, *WINDOWS[0], "aux_acc")
        extra = "" if math.isnan(aux) else f" | aux accuracy 250-300: {aux:.3f}"
        print(f"{run}: {' | '.join(parts)} | last exit 250-300: "
              f"{mean(rows, *WINDOWS[0], 'ask_ce_last'):.4f}{extra} | last step {last}")


def main():
    if len(sys.argv) >= 3 and sys.argv[1] == "table":
        table(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 1)
    elif len(sys.argv) >= 3 and sys.argv[1] == "c4":
        c4(sys.argv[2:])
    elif len(sys.argv) in (3, 5) and sys.argv[1] == "splits":
        lo, hi = (int(sys.argv[3]), int(sys.argv[4])) if len(sys.argv) == 5 else (None, None)
        splits(sys.argv[2], lo, hi)
    elif len(sys.argv) in (3, 4) and sys.argv[1] == "curve":
        curve(sys.argv[2], int(sys.argv[3]) if len(sys.argv) == 4 else 50)
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
