#!/usr/bin/env python3
"""The optional listening check of s6-ear v3 (docs/s6-ear3-prereg.md §3.7): reported, gating nothing,
done only if the owner volunteers. Everything it writes stays on this machine (the snippets are corpus
audio: never published, never committed).

    python3 -I scripts/turn-labelcheck.py collect --conv DIR/conv --tag 00012 [--dir DIR/labelcheck]
    python3 -I scripts/turn-labelcheck.py sheet --dir DIR/labelcheck
    python3 -I scripts/turn-labelcheck.py score --dir DIR/labelcheck --out runs/s6-ear3-nano/labelcheck.json

* collect: run per shard while its validation WAVs are on disk (after `sky-data fetch-conv`, before
  `scripts/turn-vadcheck.py` removes them). The sample is 100 end-of-turn and 100 hold validation records
  drawn with seed 7, uniformly without replacement, among the eligible validation records (the population
  of the cross-check): made streaming, it keeps per class the 100 records of smallest key, a seeded hash
  of the id. Each kept record is a stereo 16-bit WAV of both channels from o - 6 s to o + 4 s, named by a
  code that does not show its label; the labels stay in `sample.jsonl`, which the listener does not open.
* sheet: `answers.csv` (code, answer), in an order drawn with seed 7, the answers blank. The listener
  plays each snippet and writes "finished" (A had finished), "not finished" or "cannot tell".
* score: the agreement with the labels, "cannot tell" counted as a disagreement, with its Wilson 95%
  interval, and the share of "cannot tell".
"""
import argparse
import array
import csv
import hashlib
import json
import math
import random
import struct
import sys
from pathlib import Path

PER_CLASS = 100
SEED = 7
RATE = 16000
BEFORE_S, AFTER_S = 6.0, 4.0


def key(cid):
    return int(hashlib.sha256(f"labelcheck/{SEED}/{cid}".encode()).hexdigest()[:16], 16)


def code(cid):
    return hashlib.sha256(f"labelcheck-code/{cid}".encode()).hexdigest()[:12]


def read_f32_wav(path):
    """A mono 32-bit float WAV -> array('f')."""
    b = Path(path).read_bytes()
    i = 12
    while i + 8 <= len(b):
        cid, size = b[i:i + 4], struct.unpack("<I", b[i + 4:i + 8])[0]
        if cid == b"data":
            a = array.array("f")
            a.frombytes(b[i + 8:i + 8 + size])
            return a
        i += 8 + size + (size & 1)
    raise ValueError(f"{path}: no data chunk")


def write_stereo16(path, left, right):
    n = min(len(left), len(right))
    data = bytearray()
    for k in range(n):
        for x in (left[k], right[k]):
            data += struct.pack("<h", max(-32768, min(32767, int(round(max(-1.0, min(1.0, x)) * 32767)))))
    hdr = b"RIFF" + struct.pack("<I", 36 + len(data)) + b"WAVEfmt " + struct.pack("<IHHIIHH", 16, 1, 2, RATE, RATE * 4, 4, 16)
    Path(path).write_bytes(hdr + b"data" + struct.pack("<I", len(data)) + bytes(data))


def snippet(x, o):
    a, b = int(o - BEFORE_S * RATE), int(o + AFTER_S * RATE)
    pad_l = max(0, -a)
    seg = list(x[max(a, 0):min(b, len(x))])
    return [0.0] * pad_l + seg + [0.0] * max(0, (b - a) - pad_l - len(seg))


def cmd_collect(a):
    conv, out = Path(a.conv), Path(a.dir)
    out.mkdir(parents=True, exist_ok=True)
    xc = conv / f"xcheck-{a.tag}.jsonl"
    if not xc.exists():
        print(f"labelcheck {a.tag}: no validation session")
        return 0
    sample_path = out / "sample.jsonl"
    sample = [json.loads(l) for l in sample_path.read_text().splitlines()] if sample_path.exists() else []
    for line in xc.read_text().splitlines():
        if not line.strip():
            continue
        s = json.loads(line)
        chans = None
        for r in s["energy"]:
            cid = f"{s['session']}:{r['channel']}:{r['o']}"
            k = key(cid)
            same = sorted((x["key"] for x in sample if x["eot"] == r["eot"]))
            if any(x["id"] == cid for x in sample) or (len(same) >= PER_CLASS and k >= same[PER_CLASS - 1]):
                continue
            if chans is None:
                chans = [read_f32_wav(conv / f"wav-{a.tag}" / w) for w in s["wav"]]
            a_ch = r["channel"]
            c = code(cid)
            write_stereo16(out / f"{c}.wav", snippet(chans[a_ch], r["o"]), snippet(chans[1 - a_ch], r["o"]))
            sample.append({"id": cid, "key": k, "eot": r["eot"], "code": c, "speaker_a_channel": "left"})
            mine = sorted((x for x in sample if x["eot"] == r["eot"]), key=lambda x: x["key"])
            for x in mine[PER_CLASS:]:
                (out / f"{x['code']}.wav").unlink(missing_ok=True)
                sample.remove(x)
    sample_path.write_text("".join(json.dumps(x) + "\n" for x in sorted(sample, key=lambda x: x["key"])))
    print(f"labelcheck {a.tag}: {sum(x['eot'] for x in sample)} end of turn, {sum(not x['eot'] for x in sample)} hold kept")
    return 0


def cmd_sheet(a):
    d = Path(a.dir)
    sample = [json.loads(l) for l in (d / "sample.jsonl").read_text().splitlines()]
    codes = [x["code"] for x in sample]
    random.Random(SEED).shuffle(codes)
    with open(d / "answers.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["code", "answer"])
        for c in codes:
            w.writerow([c, ""])
    print(f"{d / 'answers.csv'}: {len(codes)} snippets; A is the left channel; answer finished, not finished or cannot tell")
    return 0


def wilson(k, n, z=1.959963984540054):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (c - h, c + h)


def cmd_score(a):
    d = Path(a.dir)
    sample = {x["code"]: x for x in (json.loads(l) for l in (d / "sample.jsonl").read_text().splitlines())}
    answers = {}
    with open(d / "answers.csv", newline="") as f:
        for row in csv.DictReader(f):
            ans = (row.get("answer") or "").strip().lower()
            if ans:
                answers[row["code"]] = ans
    norm = {"finished": True, "a had finished": True, "not finished": False, "a had not finished": False}
    n = agree = cannot = 0
    for c, ans in answers.items():
        if c not in sample:
            continue
        n += 1
        if ans in ("cannot tell", "?"):
            cannot += 1
            continue
        if ans not in norm:
            raise SystemExit(f"{c}: answer {ans!r} (finished, not finished or cannot tell)")
        agree += norm[ans] == sample[c]["eot"]
    lo, hi = wilson(agree, n)
    out = {"about": "docs/s6-ear3-prereg.md §3.7, the listening check: reported, gates nothing", "answered": n,
           "sample": len(sample), "agreement": agree / n if n else float("nan"), "wilson95": [lo, hi],
           "cannot_tell_share": cannot / n if n else float("nan"), "rule": "cannot tell counts as a disagreement"}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out))
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("collect")
    p.add_argument("--conv", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--dir", default=None)
    for name in ("sheet", "score"):
        p = sub.add_parser(name)
        p.add_argument("--dir", default=None)
        if name == "score":
            p.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.dir is None:
        if a.cmd != "collect":
            raise SystemExit("--dir is needed (DIR/labelcheck)")
        a.dir = str(Path(a.conv).parent / "labelcheck")
    return {"collect": cmd_collect, "sheet": cmd_sheet, "score": cmd_score}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
