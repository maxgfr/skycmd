#!/usr/bin/env python3
"""The gates of docs/lfm3-ft-prereg.md §6, fine-tuned against base, on the pulled evaluations. Local, CPU,
standard library only.

  gate.py [--runs runs/lfm3-ft]     -> runs/lfm3-ft/gate.json and runs/lfm3-ft/report.md

G1  s1 on test (1,000):          acc_ft >= acc_base + 0.03  and  length-stop rate_ft <= 0.5 x rate_base
G2  s2-1024 on test (1,000):     acc_ft >= acc_base - 0.01  and  mean think tokens_ft <= 0.6 x base
G3  s1 on MMLU-Pro (500):        acc_ft >= acc_base - 0.02
"LFM3 improved" = G1 and G3; G2 is reported on its own. The router (H1-H4 of docs/lfm3-s1s2-prereg.md)
is reported for both models when route.py has run.
"""
import argparse
import json
import os
import sys

N = {"test": 1000, "mmlu": 500}
FILES = {"s1": "test.s1.jsonl", "s2": "test.s2-1024.jsonl", "mmlu": "mmlu.s1.jsonl"}
ALPHA = 0.02  # the router's, docs/lfm3-s1s2-prereg.md


def read_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def stats(rows):
    n = len(rows)
    if not n:
        return None
    return {"n": n, "acc": sum(bool(r["correct"]) for r in rows) / n,
            "length_rate": sum(r["finish"] == "length" for r in rows) / n,
            "mean_think_tokens": sum(r["think_tokens"] for r in rows) / n,
            "mean_tokens": sum(r["tokens"] for r in rows) / n,
            "by_task": {t: sum(bool(r["correct"]) for r in rows if r["task"] == t) / max(1, sum(r["task"] == t for r in rows))
                        for t in sorted({r["task"] for r in rows})}}


def check_pairs(base, ft, expected):
    """Both models answered exactly the same questions, all of them."""
    b, f = {r["id"] for r in base}, {r["id"] for r in ft}
    problems = []
    if b != f:
        problems.append(f"different ids ({len(b ^ f)} differ)")
    if len(b) != expected or len(f) != expected:
        problems.append(f"incomplete: base {len(b)}, ft {len(f)}, expected {expected}")
    return problems


def gates(base, ft):
    """base and ft: {"s1": rows, "s2": rows, "mmlu": rows} -> the gate dict."""
    sb, sf = {k: stats(v) for k, v in base.items()}, {k: stats(v) for k, v in ft.items()}
    g = {"base": sb, "ft": sf, "problems": []}
    for k, exp in (("s1", N["test"]), ("s2", N["test"]), ("mmlu", N["mmlu"])):
        g["problems"] += [f"{k}: {p}" for p in check_pairs(base[k], ft[k], exp)]
    if any(sb[k] is None or sf[k] is None for k in FILES):
        g.update(G1=None, G2=None, G3=None, improved=None, verdict="incomplete")
        return g
    g1a = sf["s1"]["acc"] >= sb["s1"]["acc"] + 0.03
    g1b = sf["s1"]["length_rate"] <= 0.5 * sb["s1"]["length_rate"]
    g2a = sf["s2"]["acc"] >= sb["s2"]["acc"] - 0.01
    g2b = sf["s2"]["mean_think_tokens"] <= 0.6 * sb["s2"]["mean_think_tokens"]
    g3 = sf["mmlu"]["acc"] >= sb["mmlu"]["acc"] - 0.02
    g.update(G1={"pass": g1a and g1b, "acc": g1a, "length": g1b}, G2={"pass": g2a and g2b, "acc": g2a, "think": g2b},
             G3={"pass": g3}, improved=(g1a and g1b) and g3)
    g["verdict"] = "incomplete" if g["problems"] else ("LFM3 improved" if g["improved"] else "not improved")
    return g


def router(summary):
    """H1-H4 of docs/lfm3-s1s2-prereg.md §6 from route's summary.json (test, chosen budget)."""
    if not summary:
        return None
    b = summary.get("chosen_budget")
    e = ((summary.get("budgets") or {}).get(str(b)) or {}).get("test")
    if not e:
        return None
    e = e["all"]
    h2 = e["router_acc"] >= e["s2_acc"] - ALPHA
    h3 = e["router_tokens"] <= 0.5 * e["s2_tokens"]
    h4 = e["router_acc"] >= max(e["s1_acc"], e["s2_acc"]) + 0.01
    return {"budget": b, "s1_acc": e["s1_acc"], "s2_acc": e["s2_acc"], "router_acc": e["router_acc"],
            "router_tokens": e["router_tokens"], "s2_tokens": e["s2_tokens"], "H2": h2, "H3": h3, "H4": h4,
            "verdict": ("success" if h2 and h3 else "failure") + (", H4 holds" if h4 else "")}


def report_md(g):
    yn = lambda x: "—" if x is None else ("**yes**" if x else "**no**")
    sb, sf = g["base"], g["ft"]
    L = ["# LFM3-1.7B fine-tuned against base (docs/lfm3-ft-prereg.md)", ""]
    if g["problems"]:
        L += ["**Problems:** " + "; ".join(g["problems"]), ""]
    L += ["| set | model | n | acc | length stops | mean think tok | mean tok |", "|---|---|---:|---:|---:|---:|---:|"]
    for k, label in (("s1", "test s1"), ("s2", "test s2-1024"), ("mmlu", "MMLU-Pro s1")):
        for name, s in (("base", sb[k]), ("ft", sf[k])):
            if s:
                L.append(f"| {label} | {name} | {s['n']} | {s['acc']:.3f} | {s['length_rate']:.3f} | "
                         f"{s['mean_think_tokens']:.0f} | {s['mean_tokens']:.0f} |")
    L.append("")
    if g["G1"] is not None:
        L += ["## Gates", "",
              f"- G1, s1 accuracy +0.03: {yn(g['G1']['acc'])}; length stops halved: {yn(g['G1']['length'])} -> {yn(g['G1']['pass'])}",
              f"- G2, s2-1024 accuracy -0.01 at most: {yn(g['G2']['acc'])}; thinking <= 0.6 x: {yn(g['G2']['think'])} -> {yn(g['G2']['pass'])} (reported apart)",
              f"- G3, MMLU-Pro s1 -0.02 at most: {yn(g['G3']['pass'])}",
              f"- verdict: **{g['verdict']}**", ""]
    for name in ("base", "ft"):
        r = g.get("router", {}).get(name)
        if r:
            L += [f"## Router S1/S2, {name} (s2-{r['budget']})", "",
                  f"- s1 {r['s1_acc']:.3f}, s2 {r['s2_acc']:.3f}, router {r['router_acc']:.3f} "
                  f"({r['router_tokens']:.0f} tokens against {r['s2_tokens']:.0f})",
                  f"- H2 {yn(r['H2'])}, H3 {yn(r['H3'])}, H4 {yn(r['H4'])}: **{r['verdict']}**", ""]
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default="runs/lfm3-ft")
    a = ap.parse_args()
    load = lambda tag: {k: read_jsonl(os.path.join(a.runs, "eval", tag, f)) for k, f in FILES.items()}
    base, ft = load("base"), load("ft")
    if not any(base.values()) or not any(ft.values()):
        sys.exit(f"no evaluation under {a.runs}/eval/{{base,ft}}: run `scripts/lfm3-ft.sh pull` first")
    g = gates(base, ft)
    g["router"] = {}
    for tag in ("base", "ft"):
        p = os.path.join(a.runs, "eval", tag, "summary.json")
        g["router"][tag] = router(json.load(open(p))) if os.path.exists(p) else None
    with open(os.path.join(a.runs, "gate.json"), "w") as f:
        json.dump(g, f, indent=2)
    md = report_md(g)
    with open(os.path.join(a.runs, "report.md"), "w") as f:
        f.write(md)
    print(md)


if __name__ == "__main__":
    main()
