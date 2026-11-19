#!/usr/bin/env python3
"""The test guard of s6-ear v3 (docs/s6-ear3-prereg.md §3.5, §8.1, §8.5, §13): no v3 file reads a test
before §8.1's checks and §8.5's latencies are on file, the tests are read with the file as exported, and
each reading is made once. scripts/train-turn3.sh runs it before every command that reads a test with
v3 (sky-turn on Smart Turn's test clips, on eot-bench, on the conversation test records), which have no
guard of their own.

    python3 -I scripts/turn3-guard.py open  --run runs/s6-ear3-nano --model models/s6-ear3/s6-ear3-nano.gguf
    python3 -I scripts/turn3-guard.py check --run runs/s6-ear3-nano --model models/s6-ear3/s6-ear3-nano.gguf --stage eval

* open: once, after checking that the harness check passed (`<run>/harness3/check.json`), that the
  latencies are on file (`<run>/latency/latency.json`) and that the WASM parity was measured
  (`<run>/parity/parity.json`, `parity-conv.json`), writes `<run>/TEST-OPEN` with the sha256 of the
  model and of its calibration. Refused (exit 3) when TEST-OPEN exists.
* check: exit 0 when TEST-OPEN exists, the model is the file it records, and the stage has not written
  its marker yet (`<run>/stages/<stage>.done`: a test is read once); exit 3 otherwise.
"""
import argparse
import datetime
import hashlib
import json
import sys
from pathlib import Path

TEST_STAGES = ("eval", "eotbench", "conv-test", "gate")


class Refused(Exception):
    pass


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def need(path, what):
    if not Path(path).is_file():
        raise Refused(f"no {path}: {what}")


def open_tests(run, model):
    run = Path(run)
    if (run / "TEST-OPEN").exists():
        raise Refused(f"{run / 'TEST-OPEN'} exists: the tests were opened once")
    need(run / "harness3" / "check.json", "the harness check of §8.1 comes first")
    if not json.loads((run / "harness3" / "check.json").read_text()).get("passed"):
        raise Refused("the harness check of §8.1 failed: an amendment fixes it before any v3 test score")
    need(run / "latency" / "latency.json", "the latencies of §8.5 come first")
    need(run / "parity" / "parity.json", "the WASM parity of §7 comes first")
    need(run / "parity" / "parity-conv.json", "the WASM parity of §7 comes first")
    need(model, "the exported file")
    calib = run / "calib.json"
    rec = {"about": "s6-ear v3 (docs/s6-ear3-prereg.md): the tests opened by scripts/train-turn3.sh eval, once; the "
                    "file read on them", "opened": datetime.datetime.now().isoformat(timespec="seconds"),
           "model": {"file": str(model), "sha256": sha(model)},
           "calibration": {"file": str(calib), "sha256": sha(calib) if calib.exists() else None},
           "harness3": json.loads((run / "harness3" / "check.json").read_text()).get("checked"),
           "latency": json.loads((run / "latency" / "latency.json").read_text()).get("measured")}
    (run / "TEST-OPEN").write_text(json.dumps(rec, indent=1) + "\n")
    return rec


def check(run, model, stage):
    run = Path(run)
    if stage not in TEST_STAGES:
        raise Refused(f"unknown test stage {stage} ({', '.join(TEST_STAGES)})")
    need(run / "TEST-OPEN", "the tests are opened by the eval stage, after §8.1 and §8.5")
    rec = json.loads((run / "TEST-OPEN").read_text())
    if sha(model) != rec["model"]["sha256"]:
        raise Refused(f"{model} is not the file the tests were opened with")
    if (run / "stages" / f"{stage}.done").exists():
        raise Refused(f"{run / 'stages' / (stage + '.done')} exists: a test is read once")
    return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["open", "check"])
    ap.add_argument("--run", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--stage")
    a = ap.parse_args()
    try:
        rec = open_tests(a.run, a.model) if a.action == "open" else check(a.run, a.model, a.stage)
    except Refused as e:
        print(f"refused: {e}", file=sys.stderr)
        return 3
    print(json.dumps({"action": a.action, "stage": a.stage, "model": rec["model"]["sha256"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
