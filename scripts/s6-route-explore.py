#!/usr/bin/env python3
"""Exploratory readings of the s6-route-nano dumps, after its gate was read.

Nothing here is gated and nothing is fitted: every fusion is the plain geometric mean of two
systems' probabilities over the options of a slate (equal weights, renormalised), and the
agreement reading compares two argmaxes. The numbers are read on the test dumps after the gate
(docs/s6-nano2-prereg.md, result block), so they are exploratory by construction.

Usage: scripts/s6-route-explore.py --data ../skycmd-data/s6-route --split test [--json out.json]
"""
import argparse
import json
import math
import os

SOURCES = ["t_tools", "t_tools_masked", "t_metatool", "t_clinc", "t_bfcl", "t_when2call", "t_massive"]
PAIRS = [
    ("model", "bm25"),
    ("model", "potion-cos-4bit"),
    ("bm25", "potion-cos-4bit"),
    ("bm25", "potion-cos"),
    ("bm25", "minilm-cos"),
    ("model", "minilm-cos"),
]


def load(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def argmax(p):
    best = 0
    for i, v in enumerate(p):
        if v > p[best]:
            best = i
    return best


def accuracy(rows):
    return sum(1 for r in rows if argmax(r["probs"]) == r["gold"]) / len(rows)


def fuse(a, b):
    out = []
    for ra, rb in zip(a, b):
        logs = [0.5 * (math.log(max(x, 1e-12)) + math.log(max(y, 1e-12))) for x, y in zip(ra["probs"], rb["probs"])]
        m = max(logs)
        e = [math.exp(v - m) for v in logs]
        s = sum(e)
        out.append({"probs": [v / s for v in e], "gold": ra["gold"]})
    return out


def agreement(a, b):
    """Answer when both argmaxes agree, ask otherwise."""
    n = len(a)
    agree = [(ra, rb) for ra, rb in zip(a, b) if argmax(ra["probs"]) == argmax(rb["probs"])]
    right = sum(1 for ra, _ in agree if argmax(ra["probs"]) == ra["gold"])
    split = [(ra, rb) for ra, rb in zip(a, b) if argmax(ra["probs"]) != argmax(rb["probs"])]
    a_right = sum(1 for ra, _ in split if argmax(ra["probs"]) == ra["gold"])
    b_right = sum(1 for _, rb in split if argmax(rb["probs"]) == rb["gold"])
    return {
        "coverage": len(agree) / n,
        "accuracy_when_agree": right / len(agree) if agree else None,
        "disagree": len(split),
        "first_right_when_disagree": a_right,
        "second_right_when_disagree": b_right,
        "neither_right_when_disagree": len(split) - a_right - b_right,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="the route data folder; the dumps are under <data>/eval/<split>/<system>/")
    ap.add_argument("--split", default="test")
    ap.add_argument("--json")
    args = ap.parse_args()
    root = os.path.join(args.data, "eval", args.split)
    out = {"split": args.split, "note": "exploratory, read after the gate; equal-weight geometric means, nothing fitted", "sources": {}}
    for src in SOURCES:
        dumps = {}
        for system in sorted({s for pair in PAIRS for s in pair}):
            path = os.path.join(root, system, src + ".jsonl")
            if os.path.exists(path):
                dumps[system] = load(path)
        if not dumps:
            continue
        entry = {"n": len(next(iter(dumps.values()))), "single": {s: accuracy(r) for s, r in dumps.items()}, "fused": {}, "agreement": {}}
        for a, b in PAIRS:
            if a in dumps and b in dumps:
                if [r["gold"] for r in dumps[a]] != [r["gold"] for r in dumps[b]]:
                    raise SystemExit(f"{src}: {a} and {b} do not hold the same slates in the same order")
                entry["fused"][f"{a}+{b}"] = accuracy(fuse(dumps[a], dumps[b]))
                entry["agreement"][f"{a}+{b}"] = agreement(dumps[a], dumps[b])
        out["sources"][src] = entry
        print(f"== {src} (n {entry['n']})")
        for s, v in entry["single"].items():
            print(f"  {s:<28} {v:.4f}")
        for s, v in entry["fused"].items():
            g = entry["agreement"][s]
            print(
                f"  {s:<28} fused {v:.4f} | agree on {g['coverage']:.4f}, accuracy there {g['accuracy_when_agree']:.4f}"
                f" | disagree {g['disagree']}: first right {g['first_right_when_disagree']},"
                f" second right {g['second_right_when_disagree']}, neither {g['neither_right_when_disagree']}"
            )
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=1)
            f.write("\n")


if __name__ == "__main__":
    main()
