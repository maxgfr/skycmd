#!/usr/bin/env python3
"""The S1/S2 router of docs/lfm3-s1s2-prereg.md on the pulled generations of one model (amendment 1 of
that pre-registration): experiments/lfm3-s1s2/s1s2.py's `route`, unchanged, pointed at
runs/lfm3-ft/eval/<tag> instead of runs/lfm3-s1s2. Local, CPU, standard library only.

  route.py --tag base|ft|pair [--runs runs/lfm3-ft]

Runs `route --split calib` (probe and tau on the calibration split) then `route --split test` (the
single read of the test split for this model). Writes report-calib.md, report.md and summary.json in
the eval directory. --tag pair is amendment 5 (local mode): eval/pair holds the fine-tuned model's s1
files and the base model's s2 files; the base itself is not read again, its reading stays the one of
runs/lfm3-s1s2. Amendment 6: the pair's s2 files are always the published base's generations of
runs/lfm3-s1s2, even when the base's s1 was evaluated again with the local conversion (eval/base then
holds s1 files only, and --tag base has nothing to route in local mode). The budget the pair's router
chooses on its calibration (chosen_budget in summary.json) is the budget of 1/rho in gate.py.
"""
import argparse
import importlib.util
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
S1S2 = os.path.join(HERE, "..", "lfm3-s1s2", "s1s2.py")


def load_s1s2():
    spec = importlib.util.spec_from_file_location("s1s2", S1S2)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def alignment_problems(d):
    """s1s2.py's route keeps only the ids present in both the s1 and the s2 file of a split and drops
    the others without a word. On the pair, assembled from two models' folders, that would hide a gap:
    each split's three files must hold the same ids, once each."""
    sys.path.insert(0, HERE)
    import common as C

    out = []
    for split in ("calib", "test"):
        rows = {m: C.read_jsonl(os.path.join(d, f"{split}.{m}.jsonl")) for m in ("s1", "s2-256", "s2-1024")}
        if not rows["s1"]:
            out.append(f"{split}.s1.jsonl: empty or missing")
            continue
        ids = [r["id"] for r in rows["s1"]]
        for m, rs in rows.items():
            out += C.id_problems(f"{split}.{m}.jsonl", rs, ids, m)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", choices=("base", "ft", "pair"), required=True)
    ap.add_argument("--runs", default="runs/lfm3-ft")
    a = ap.parse_args()
    d = os.path.join(a.runs, "eval", a.tag)
    meta_path = os.path.join(d, "meta.json")
    if not os.path.exists(meta_path):
        sys.exit(f"{meta_path} missing: run `scripts/lfm3-ft.sh eval` first")
    with open(meta_path) as f:
        meta = json.load(f)
    if a.tag == "pair":
        problems = alignment_problems(d)
        if problems:
            sys.exit("the pair's files do not hold the same questions, nothing is routed:\n  " + "\n  ".join(problems))
    s = load_s1s2()
    s.RUNS = d
    s.MODEL = f"{meta['model']} ({meta.get('weights', 'bf16')}, {meta['engine']})"
    s.MODEL_REV = meta["revision"]
    s.cmd_route(argparse.Namespace(split="calib"))
    s.cmd_route(argparse.Namespace(split="test"))


if __name__ == "__main__":
    main()
