#!/usr/bin/env python3
"""Reads the training logs of the diagnosis of docs/s6-sleep3-diagnosis.md.

  s6-sleep3-trace.py table <run> [every]   the memory trace of an `s2-decide --seq-log
                                           --memory-trace` run, one line per sequence step
                                           (every: one step in that many after step 12)
  s6-sleep3-trace.py c4 <run>...           the reading of C4 (docs/s6-sleep2-prereg.md section 3)
                                           for each run: the ask-frame cross-entropy of recall
                                           over steps 250 to 300, and over earlier windows

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
        "at the asks: |S| |x| |r| |r.Wo|"
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
            f"{layer['read_norm_ask']:.1e}"
        )


def mean(rows, lo, hi, key):
    xs = [r[key] for r in rows if key in r and lo <= r["step"] <= hi]
    return sum(xs) / len(xs) if xs else float("nan")


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
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
