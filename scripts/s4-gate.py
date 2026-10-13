#!/usr/bin/env python3
"""s4 gate and results table (issue #1).

Reads the `sky-eval --json` files of the models (runs/<name>/eval-*.json) and the teacher
summaries (runs/<name>/teacher/*.summary.json), prints a Markdown table of accuracy / ECE /
depth / latency per test source and checks the gate: typed_decisions_bench accuracy of the
ternary s4 GGUF above GATE (0.55 by default; oscar-1-17m reaches 0.6775, s3-pico 0.382).

Usage: s4-gate.py runs/s4-micro [--gate 0.55] [--out runs/s4-micro/results.md]
Exit status 0 when the gate passes, 1 otherwise.
"""
import argparse
import glob
import json
import os
import sys

OSCAR = {"typed_decisions_bench": 0.6775}
ORDER = ["typed_decisions_bench", "zs_banking77", "zs_massive_en", "zs_massive_fr", "when2call", "all"]


def load(path):
    with open(path) as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--gate", type=float, default=0.55)
    ap.add_argument("--out")
    a = ap.parse_args()

    models = []
    for p in sorted(glob.glob(os.path.join(a.run, "eval-*.json"))):
        d = load(p)
        name = os.path.basename(p)[len("eval-") : -len(".json")]
        models.append((name, d))
    teacher = {}
    for p in glob.glob(os.path.join(a.run, "teacher", "*.summary.json")):
        try:
            d = load(p)
        except (ValueError, OSError):
            continue
        src = os.path.basename(p)[: -len(".summary.json")]
        teacher[src] = d

    srcs = [s for s in ORDER if any(s in d["sources"] for _, d in models)]
    lines = ["| model | params | " + " | ".join(srcs) + " |", "|---|---:|" + "---:|" * len(srcs)]
    for name, d in models:
        cells = []
        for s in srcs:
            r = d["sources"].get(s)
            cells.append(f"{r['accuracy']:.3f}" if r else "—")
        lines.append(f"| {name} | {d['n_params']:,} | " + " | ".join(cells) + " |")
    if teacher:
        model = next(iter(teacher.values())).get("model", "teacher")
        cells = [f"{teacher[s]['acc']:.3f}" if s in teacher and "acc" in teacher[s] else "—" for s in srcs]
        lines.append(f"| teacher {model} | — | " + " | ".join(cells) + " |")
    lines.append(
        "| oscar-1-17m (published) | 17M | " + " | ".join(f"{OSCAR[s]:.4f}" if s in OSCAR else "—" for s in srcs) + " |"
    )
    lines.append("")
    lines.append("| model | ECE (all) | mean block passes | early-exit rate | median µs / decision (native) |")
    lines.append("|---|---:|---:|---:|---:|")
    for name, d in models:
        r = d["sources"]["all"]
        lines.append(
            f"| {name} | {r['ece']:.4f} | {r['mean_layers']:.2f} | {r['early_exit_rate']:.3f} | {r['median_us']:.0f} |"
        )
    md = "\n".join(lines) + "\n"
    print(md)
    if a.out:
        with open(a.out, "w") as f:
            f.write(md)

    tern = [d for n, d in models if "ternary" in n]
    if not tern or "typed_decisions_bench" not in tern[0]["sources"]:
        print("gate: no ternary s4 evaluation found")
        return 1
    acc = tern[0]["sources"]["typed_decisions_bench"]["accuracy"]
    ok = acc > a.gate
    print(f"gate: typed_decisions_bench {acc:.4f} {'>' if ok else '<='} {a.gate} -> {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
