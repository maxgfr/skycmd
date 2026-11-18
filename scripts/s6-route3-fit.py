#!/usr/bin/env python3
"""s6-nano v3 (docs/s6-nano3-prereg.md, section 2): the two values of the fused system that are
read on the validation slates, and its validation numbers.

  beta       p(i) ∝ (p_model(i) · p_embedding(i))^beta; the value of the grid 0.01 .. 4.00 that
             minimises the NLL on the validation slates
  tau_ask3   v2's rule on the fused probabilities: the smallest top probability at which the
             accuracy among answered requests is >= 0.95

Inputs are validation dumps only (the model's dump of v_tools at full depth and at its first
exit, and the dump of potion-cos-4bit as fitted for v2). No test file is read.

Usage: scripts/s6-route3-fit.py --model runs/s6-nano2/v_tools.dump.jsonl \
         --exit0 runs/s6-nano2/explore/v_tools.exit0.jsonl --data ../skycmd-data/s6-route \
         --out runs/s6-nano3/fit.json
"""
import argparse
import json
import math
import os

PARTNER = "potion-cos-4bit"
OTHERS = ["bm25", "minilm-cos", "potion-cos", "static-ft", "toolrouter"]


def load(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def argmax(p):
    best = 0
    for i, v in enumerate(p):
        if v > p[best]:
            best = i
    return best


def fuse(a, b, beta):
    out = []
    for ra, rb in zip(a, b):
        logs = [beta * (math.log(max(x, 1e-12)) + math.log(max(y, 1e-12))) for x, y in zip(ra["probs"], rb["probs"])]
        m = max(logs)
        e = [math.exp(v - m) for v in logs]
        s = sum(e)
        out.append({"probs": [v / s for v in e], "gold": ra["gold"]})
    return out


def accuracy(rows):
    return sum(1 for r in rows if argmax(r["probs"]) == r["gold"]) / len(rows)


def nll(rows):
    return -sum(math.log(max(r["probs"][r["gold"]], 1e-12)) for r in rows) / len(rows)


def ece(rows, bins=15):
    cells = [[0, 0.0, 0] for _ in range(bins)]
    for r in rows:
        k = argmax(r["probs"])
        c = r["probs"][k]
        b = min(bins - 1, int(c * bins))
        cells[b][0] += 1
        cells[b][1] += c
        cells[b][2] += int(k == r["gold"])
    return sum(n / len(rows) * abs(conf / n - ok / n) for n, conf, ok in cells if n)


def at_coverage(rows, coverage):
    kept = sorted(rows, key=lambda r: -max(r["probs"]))[: int(len(rows) * coverage)]
    return accuracy(kept)


def tau_ask(rows, target=0.95):
    """The smallest top probability at which the accuracy among answered requests is >= target
    (1.0 when none reaches it)."""
    ranked = sorted(rows, key=lambda r: -max(r["probs"]))
    ok = 0
    best = {"tau": 1.0, "accuracy": None, "coverage": 0.0}
    for i, r in enumerate(ranked, 1):
        ok += int(argmax(r["probs"]) == r["gold"])
        if ok / i >= target:
            best = {"tau": max(r["probs"]), "accuracy": ok / i, "coverage": i / len(ranked)}
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--exit0", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    val = os.path.join(args.data, "eval", "val")
    model = load(args.model)
    exit0 = load(args.exit0)
    golds = [r["gold"] for r in model]
    dumps = {s: load(os.path.join(val, s, "v_tools.jsonl")) for s in [PARTNER] + OTHERS}
    for name, rows in list(dumps.items()) + [("exit0", exit0)]:
        if [r["gold"] for r in rows] != golds:
            raise SystemExit(f"{name}: not the same slates in the same order as the model's dump")
    partner = dumps[PARTNER]
    grid = [i / 100 for i in range(1, 401)]
    beta = min(grid, key=lambda b: nll(fuse(model, partner, b)))
    fused = fuse(model, partner, beta)
    fast = fuse(exit0, partner, beta)
    out = {
        "about": "docs/s6-nano3-prereg.md section 2: read on the validation slates (v_tools) only",
        "slates": len(model),
        "partner": PARTNER,
        "beta": beta,
        "nll_at_beta": nll(fused),
        "nll_at_0.5": nll(fuse(model, partner, 0.5)),
        "tau_ask3": tau_ask(fused),
        "system": {
            "accuracy": accuracy(fused),
            "ece": ece(fused),
            "accuracy_at_coverage": {str(c): at_coverage(fused, c) for c in (0.5, 0.7, 0.8, 0.9)},
        },
        "fast_mode": {"accuracy": accuracy(fast), "ece": ece(fast)},
        "alone": {"model": accuracy(model), "model_exit0": accuracy(exit0), **{s: accuracy(r) for s, r in dumps.items()}},
        "bm25_with_partner": accuracy(fuse(dumps["bm25"], partner, 0.5)),
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
        f.write("\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
