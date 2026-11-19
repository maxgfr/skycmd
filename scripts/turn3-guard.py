#!/usr/bin/env python3
"""The test guard of s6-ear v3 (docs/s6-ear3-prereg.md §3.5, §7, §8.1, §8.5, §13, amendment 3): no v3 file
reads a test before §8.1's checks, §8.5's latencies and §7's WASM parity are on file, the tests are read
with the file as exported, and each reading is made once. scripts/train-turn3.sh runs it before every
command that reads a test with v3 (sky-turn on Smart Turn's test clips, on eot-bench, on the conversation
test records), which have no guard of their own; and before the GPU stages, on the teacher's output.

    python3 -I scripts/turn3-guard.py open    --run runs/s6-ear3-nano --model models/s6-ear3/s6-ear3-nano.gguf
    python3 -I scripts/turn3-guard.py check   --run runs/s6-ear3-nano --model models/s6-ear3/s6-ear3-nano.gguf --stage eval
    python3 -I scripts/turn3-guard.py parity  --run runs/s6-ear3-nano
    python3 -I scripts/turn3-guard.py teacher --conv ../skycmd-data/s6-ear3/conv

* open: once, after checking that the harness check passed (`<run>/harness3/check.json`), that the
  latencies are on file (`<run>/latency/latency.json`) and that the WASM parity was measured in full
  (`<run>/parity/parity.json`, `parity-conv.json`: see parity), writes `<run>/TEST-OPEN` with the sha256 of
  the model and of its calibration. Refused (exit 3) when TEST-OPEN exists.
* check: exit 0 when TEST-OPEN exists, the model is the file it records, and the stage has not written
  its marker yet (`<run>/stages/<stage>.done`: a test is read once); exit 3 otherwise.
* parity (§7, amendment 3): each parity file holds the line of `parity-node.mjs --turn` for the 20
  pre-registered clips or records (`n` >= 20, a finite `max_abs_diff`, a `same_decision` count); a
  missing, empty or unparsable file, or one with fewer clips, is a technical failure (exit 3), never a
  measurement. A complete measurement that misses |dp| <= 1e-4 or a decision is a result: it passes here
  and G3 reads it (§9).
* teacher (§5, amendment 3; read-only): for every shard of the conversation folder, the number of
  validation and test records the shard wrote (`conv-<tag>.json`, `written`: each has a tail) against the
  records of `teacher-conv-<tag>.jsonl`, and the store's validation and test records of the shard without a
  teacher record; prints a table; exit 3 when a shard is unscored, has fewer records than it wrote, or
  leaves a stored record without its opinion.
"""
import argparse
import datetime
import hashlib
import json
import math
import re
import sys
from pathlib import Path

TEST_STAGES = ("eval", "eotbench", "conv-test", "gate")
# §7: WASM parity on 20 Smart Turn test clips and 20 conversation test records, |dp| <= 1e-4, same decision
PARITY_CLIPS = 20
PARITY_TOL = 1e-4
PARITY_FILES = ("parity.json", "parity-conv.json")


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


def is_int(v):
    return isinstance(v, int) and not isinstance(v, bool)


def read_parity(path):
    """The line of `parity-node.mjs --turn` in `path`, refused unless it is a complete measurement."""
    p = Path(path)
    if not p.is_file():
        raise Refused(f"no {p}: the WASM parity of §7 comes first")
    text = p.read_text(errors="replace")
    lines = [l for l in text.splitlines() if l.strip().startswith("{")]
    if not lines:
        raise Refused(f"{p} is empty or holds no result line: the parity run failed (a technical failure, "
                      "fixed and rerun; amendment 3)")
    try:
        rec = json.loads(lines[-1])
    except ValueError as e:
        raise Refused(f"{p}: its result line does not parse ({e}): a technical failure (amendment 3)")
    if not isinstance(rec, dict) or rec.get("mode") != "turn":
        raise Refused(f"{p}: not the line of parity-node.mjs --turn")
    n, diff, same = rec.get("n"), rec.get("max_abs_diff"), rec.get("same_decision")
    if not is_int(n) or n < PARITY_CLIPS:
        raise Refused(f"{p}: {n} clips, §7 measures {PARITY_CLIPS}: a technical failure (amendment 3)")
    if not isinstance(diff, (int, float)) or isinstance(diff, bool) or not math.isfinite(diff):
        raise Refused(f"{p}: max_abs_diff {diff!r} is not a number")
    if not is_int(same) or not 0 <= same <= n:
        raise Refused(f"{p}: same_decision {same!r} is not a count of the {n} clips")
    rec["passed"] = diff <= PARITY_TOL and same == n
    return rec


def parity(run):
    """Both parity files of §7 read as complete measurements (pass or fail: G3 decides)."""
    run = Path(run)
    return {f: read_parity(run / "parity" / f) for f in PARITY_FILES}


def open_tests(run, model):
    run = Path(run)
    if (run / "TEST-OPEN").exists():
        raise Refused(f"{run / 'TEST-OPEN'} exists: the tests were opened once")
    need(run / "harness3" / "check.json", "the harness check of §8.1 comes first")
    if not json.loads((run / "harness3" / "check.json").read_text()).get("passed"):
        raise Refused("the harness check of §8.1 failed: an amendment fixes it before any v3 test score")
    need(run / "latency" / "latency.json", "the latencies of §8.5 come first")
    par = parity(run)
    need(model, "the exported file")
    calib = run / "calib.json"
    rec = {"about": "s6-ear v3 (docs/s6-ear3-prereg.md): the tests opened by scripts/train-turn3.sh eval, once; the "
                    "file read on them", "opened": datetime.datetime.now().isoformat(timespec="seconds"),
           "model": {"file": str(model), "sha256": sha(model)},
           "calibration": {"file": str(calib), "sha256": sha(calib) if calib.exists() else None},
           "harness3": json.loads((run / "harness3" / "check.json").read_text()).get("checked"),
           "latency": json.loads((run / "latency" / "latency.json").read_text()).get("measured"),
           "parity": {f: {k: r.get(k) for k in ("n", "max_abs_diff", "same_decision")} for f, r in par.items()}}
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


# --- the teacher's recorded opinion, shard by shard (read-only) ----------------------------------------

TAG = re.compile(r"^\d{5}$")


def expected_teacher(conv, tag):
    """The validation and test records shard `tag` wrote (each with a tail), from `conv-<tag>.json`."""
    p = Path(conv) / f"conv-{tag}.json"
    if not p.is_file():
        return None
    w = json.loads(p.read_text()).get("written") or {}
    return int(w.get("val", 0)) + int(w.get("test", 0))


def teacher_ids(path):
    """(ids with a finite probability, unreadable lines) of a `teacher-conv-<tag>.jsonl`."""
    ids, bad = set(), 0
    if not path.is_file():
        return ids, bad
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            try:
                r = json.loads(line)
                p = r["prob"]
                if isinstance(p, bool) or not isinstance(p, (int, float)) or not math.isfinite(p):
                    raise ValueError
                ids.add(r["id"])
            except (ValueError, KeyError, TypeError):
                bad += 1
    return ids, bad


def stored_val_test(store, tag):
    """The ids of the validation and test records of shard `tag` in the store (after the caps)."""
    p = Path(store) / f"manifest-{tag}.jsonl"
    out = []
    if p.is_file():
        with open(p) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    if r.get("split") in ("val", "test"):
                        out.append(r["id"])
    return out


def teacher(conv):
    """One row per shard; `ok` False on any shard whose recorded opinion is incomplete."""
    conv = Path(conv)
    if not conv.is_dir():
        raise Refused(f"no {conv}: the conversation folder of sky-data fetch-conv")
    store = conv / "store"
    tags = {p.name[len("conv-"):-len(".json")] for p in conv.glob("conv-*.json")}
    tags |= {p.name[len("manifest-"):-len(".jsonl")] for p in store.glob("manifest-*.jsonl")}
    tags = sorted(t for t in tags if TAG.match(t))
    if not tags:
        raise Refused(f"{conv} holds no shard (conv-<tag>.json)")
    rows = []
    for t in tags:
        expected = expected_teacher(conv, t)
        done = conv / f"teacher-conv-{t}.done"
        ids, bad = teacher_ids(conv / f"teacher-conv-{t}.jsonl")
        stored = stored_val_test(store, t)
        unscored = sum(1 for i in stored if i not in ids)
        why = []
        if expected is None:
            why.append(f"no conv-{t}.json")
        if not done.is_file():
            why.append("no .done (not scored)")
        if expected is not None and len(ids) < expected:
            why.append(f"{expected - len(ids)} fewer records than the shard wrote")
        if unscored:
            why.append(f"{unscored} stored records without an opinion")
        if bad:
            why.append(f"{bad} unreadable lines")
        if (conv / f"tails-{t}.jsonl").exists():
            why.append("its tails are still on disk (not scored)")
        rows.append({"shard": t, "expected": expected, "records": len(ids), "stored": len(stored),
                     "unscored": unscored, "ok": not why, "why": "; ".join(why) or "ok"})
    return rows


def print_teacher(rows, out=sys.stdout):
    print(f"{'shard':>5}  {'expected':>8}  {'records':>8}  {'stored':>7}  {'unscored':>8}  status", file=out)
    for r in rows:
        e = "-" if r["expected"] is None else r["expected"]
        print(f"{r['shard']:>5}  {e:>8}  {r['records']:>8}  {r['stored']:>7}  {r['unscored']:>8}  {r['why']}", file=out)
    bad = [r for r in rows if not r["ok"]]
    tot = sum(r["expected"] or 0 for r in rows)
    print(f"{len(rows)} shards, {len(bad)} incomplete; {sum(r['records'] for r in rows)} teacher records for {tot} "
          f"validation and test records written, {sum(r['stored'] for r in rows)} stored, "
          f"{sum(r['unscored'] for r in rows)} stored without an opinion", file=out)
    return bad


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["open", "check", "parity", "teacher"])
    ap.add_argument("--run")
    ap.add_argument("--model")
    ap.add_argument("--stage")
    ap.add_argument("--conv", help="teacher: the conversation folder (<data>/conv)")
    a = ap.parse_args()
    want = {"open": ("run", "model"), "check": ("run", "model", "stage"), "parity": ("run",), "teacher": ("conv",)}
    missing = [k for k in want[a.action] if not getattr(a, k)]
    if missing:
        ap.error(f"{a.action} needs " + ", ".join(f"--{k}" for k in missing))
    try:
        if a.action == "teacher":
            bad = print_teacher(teacher(a.conv))
            if bad:
                raise Refused(f"{len(bad)} shards with an incomplete teacher opinion: a technical failure, rescored or "
                              "rebuilt before any GPU stage (amendment 3)")
            return 0
        if a.action == "parity":
            par = parity(a.run)
            print(json.dumps({"action": "parity", **{f: {k: r.get(k) for k in ("n", "max_abs_diff", "same_decision",
                                                                            "passed")} for f, r in par.items()}}))
            return 0
        rec = open_tests(a.run, a.model) if a.action == "open" else check(a.run, a.model, a.stage)
    except Refused as e:
        print(f"refused: {e}", file=sys.stderr)
        return 3
    print(json.dumps({"action": a.action, "stage": a.stage, "model": rec["model"]["sha256"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
