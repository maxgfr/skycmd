#!/usr/bin/env python3
"""The S1/S2 router of docs/lfm3-s1s2-prereg.md on the pulled generations of one model (amendment 1 of
that pre-registration): experiments/lfm3-s1s2/s1s2.py's `route`, unchanged, pointed at
runs/lfm3-ft/eval/<tag> instead of runs/lfm3-s1s2. Local, CPU, standard library only.

  route.py --tag base|ft [--runs runs/lfm3-ft]

Runs `route --split calib` (probe and tau on the calibration split) then `route --split test` (the
single read of the test split for this model). Writes report-calib.md, report.md and summary.json in
the eval directory.
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


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", choices=("base", "ft"), required=True)
    ap.add_argument("--runs", default="runs/lfm3-ft")
    a = ap.parse_args()
    d = os.path.join(a.runs, "eval", a.tag)
    meta_path = os.path.join(d, "meta.json")
    if not os.path.exists(meta_path):
        sys.exit(f"{meta_path} missing: run `scripts/lfm3-ft.sh pull` after the eval job")
    with open(meta_path) as f:
        meta = json.load(f)
    s = load_s1s2()
    s.RUNS = d
    s.MODEL = f"{meta['model']} (bf16, {meta['engine']})"
    s.MODEL_REV = meta["revision"]
    s.cmd_route(argparse.Namespace(split="calib"))
    s.cmd_route(argparse.Namespace(split="test"))


if __name__ == "__main__":
    main()
