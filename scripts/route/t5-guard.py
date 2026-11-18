"""The guard of T5 (s6-route v7): exit 0 when runs/s6-route7/FRESH-OPEN exists and every file of
runs/s6-route7/test-files.json under <data>/toucan5/fresh (or $T5/fresh) has its sha256; exit 3
otherwise. scripts/route7-gate.sh runs it before every command that reads T5 (sky-data route
render and blind, sky-eval, sky-route fuse), which have no guard of their own.

    python3 -I scripts/route/t5-guard.py [--open runs/s6-route7/FRESH-OPEN] [--files runs/s6-route7/test-files.json]
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--open", default=common.T5_OPEN)
    ap.add_argument("--files", default=common.T5_FILES)
    a = ap.parse_args()
    common.T5_OPEN, common.T5_FILES = a.open, a.files
    try:
        want = common.t5_check()
    except common.Refused as e:
        print(f"refused: {e}", file=sys.stderr)
        return 3
    print(json.dumps({"T5": common.t5_root(), "files": len(want), "open": a.open}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
