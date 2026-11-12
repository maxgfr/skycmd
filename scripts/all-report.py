#!/usr/bin/env python3
"""The gates of the five s6 models, in one file (TODO.md §9.4, scripts/train-all.sh report).

Reads what each training script wrote, applies no judgement, and writes one table: a model whose files
are missing is shown as "not run", and a value that cannot be read as "unreadable", never as a pass.

  all-report.py [--prefix runs/all-smoke/] --out runs/all/REPORT.md
"""
import argparse
import json
import os
import re

# docs/s6-secrets-prereg.md §6 and docs/s6-cast-prereg.md §6.
SECRETS = {"secret": 0.977, "service": 0.787}
CAST = {"nGMASE": 0.838, "nWQL": 0.587}
CAST_CONFIGS = 97


def ear(p):
    path = f"{p}runs/s6-ear-nano/eval/gate.json"
    if not os.path.exists(path):
        return "not run", path, ""
    g = json.load(open(path))
    def got(r):
        return "n/a" if r["accuracy"] is None else f"{r['accuracy']:.4f}"

    rows = "; ".join(f"{r['gate']} {got(r)} (bar >= {r['bar']:.4f})" for r in g["gates"])
    return "PASS" if g["pass"] else "FAIL", path, rows


def size(p, name):
    path = f"{p}runs/s6-{name}/eval/results.md"
    if not os.path.exists(path):
        return "not run", path, ""
    text = open(path).read()
    m = re.search(r"^- gate: (.*)$", text, flags=re.M)
    if not m:
        return "unreadable", path, ""
    # the two lines that say why: the gate at full depth and at the CRC thresholds
    why = re.findall(r"^- (gate at [^\n]*)$", text, flags=re.M)
    return ("PASS" if m.group(1).startswith("PASS") else "FAIL"), path, "; ".join([m.group(1), *why])


def secrets_acc(text, question):
    """The accuracy of secret_bench:<question> at the last exit of the GGUF, the variant and depth the
    gate names (docs/s6-secrets-prereg.md §6). An s2-eval report also has the f32 and fake-quant
    variants, one table per exit each, so the first row of that source is the wrong one."""
    sections = text.split("\n## GGUF `")
    if len(sections) < 2:
        return None
    exits = re.findall(r"^#### Exit \d+ .*?(?=^#### |\Z)", sections[-1], flags=re.M | re.S)
    if not exits:
        return None
    m = re.search(rf"^\| secret_bench:{question} \| \d+ \| (\d\.\d+) \|", exits[-1], flags=re.M)
    return float(m.group(1)) if m else None


def secrets(p):
    path = f"{p}runs/s6-secrets-pico/report.md"
    if not os.path.exists(path):
        return "not run", path, ""
    text = open(path).read()
    got = {q: secrets_acc(text, q) for q in SECRETS}
    if any(v is None for v in got.values()):
        return "unreadable", path, str(got)
    ok = all(got[q] > bar for q, bar in SECRETS.items())
    return ("PASS" if ok else "FAIL"), path, "; ".join(f"{q} {got[q]:.4f} (bar > {SECRETS[q]})" for q in SECRETS)


def cast(p):
    path = f"{p}data/gift-eval-out/s6-cast-pico/summary.json"
    if not os.path.exists(path):
        return "not run", path, ""
    s = json.load(open(path))
    # A mean over fewer configurations than the bench's is not the published metric.
    if any(s.get(k) is None or s.get("configs", {}).get(k) != CAST_CONFIGS for k in CAST):
        return "unreadable", path, str(s)
    ok = all(s[k] <= bar for k, bar in CAST.items())
    return ("PASS" if ok else "FAIL"), path, "; ".join(f"{k} {s[k]:.4f} (bar <= {CAST[k]})" for k in CAST)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prefix", default="", help="the SKY_OUT of the run")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    p = a.prefix
    rows = [("s6-ear-nano", *ear(p)), ("s6-secrets-pico", *secrets(p)), ("s6-nano", *size(p, "nano")),
            ("s6-micro", *size(p, "micro")), ("s6-cast-pico", *cast(p))]
    lines = ["# The big run: the gates of the s6 models", "", "| model | gate | detail | file read |", "|---|---|---|---|"]
    lines += [f"| {m} | **{v}** | {d} | `{f}` |" for m, v, f, d in rows]
    lines += ["", "A failed gate means the model is not published; its result stays in this report.", ""]
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
