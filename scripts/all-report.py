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


def ear(p):
    path = f"{p}runs/s6-ear-nano/eval/gate.json"
    if not os.path.exists(path):
        return "not run", path, ""
    g = json.load(open(path))
    rows = "; ".join(f"{r['gate']} {r['accuracy']} vs {r['bar']:.4f}" for r in g["gates"])
    return "PASS" if g["pass"] else "FAIL", path, rows


def size(p, name):
    path = f"{p}runs/s6-{name}/eval/results.md"
    if not os.path.exists(path):
        return "not run", path, ""
    m = re.search(r"^- gate: (.*)$", open(path).read(), flags=re.M)
    if not m:
        return "unreadable", path, ""
    return ("PASS" if m.group(1).startswith("PASS") else "FAIL"), path, m.group(1)


def secrets_acc(text, question):
    """The accuracy of secret_bench:<question> in an s2-eval report (the first number of its row)."""
    m = re.search(rf"secret_bench:{question}\b[^\n]*?(\d\.\d+)", text)
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
    return ("PASS" if ok else "FAIL"), path, "; ".join(f"{q} {got[q]:.4f} > {SECRETS[q]}" for q in SECRETS)


def cast(p):
    path = f"{p}data/gift-eval-out/s6-cast-pico/summary.json"
    if not os.path.exists(path):
        return "not run", path, ""
    s = json.load(open(path))
    if any(k not in s for k in CAST):
        return "unreadable", path, str(sorted(s))
    ok = all(s[k] <= bar for k, bar in CAST.items())
    return ("PASS" if ok else "FAIL"), path, "; ".join(f"{k} {s[k]:.4f} <= {CAST[k]}" for k in CAST)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prefix", default="", help="the SKY_OUT of the run")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    p = a.prefix
    rows = [("s6-ear-nano", *ear(p)), ("s6-secrets-pico", *secrets(p)), ("s6-nano", *size(p, "nano")),
            ("s6-micro", *size(p, "micro")), ("s6-cast-pico", *cast(p))]
    lines = ["# Le grand run : les portes des modèles s6", "", "| modèle | porte | détail | fichier lu |", "|---|---|---|---|"]
    lines += [f"| {m} | **{v}** | {d} | `{f}` |" for m, v, f, d in rows]
    lines += ["", "Une porte échouée veut dire que le modèle n'est pas publié ; son résultat reste dans le rapport.", ""]
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
