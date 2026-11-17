#!/usr/bin/env python3
"""The gates of docs/lfm3-ft-prereg.md §6, fine-tuned against base, on the pulled evaluations. Local, CPU,
standard library only.

  gate.py [--runs runs/lfm3-ft]     -> runs/lfm3-ft/gate.json and runs/lfm3-ft/report.md

G1  s1 on test (1,000):          acc_ft >= acc_base + 0.03  and  length-stop rate_ft <= 0.5 x rate_base
G2  s2-1024 on test (1,000):     acc_ft >= acc_base - 0.01  and  mean think tokens_ft <= 0.6 x base
G3  s1 on MMLU-Pro (500):        acc_ft >= acc_base - 0.02
"LFM3 improved" = G1 and G3; G2 is reported on its own. The router (H1-H4 of docs/lfm3-s1s2-prereg.md)
is reported for both models when route.py has run.

Amendment 5 (local mode: --amendment5, which scripts/lfm3-ft.sh passes, or the pair folder eval/pair
exists): System 2 is the base model itself,
so G2 is not read; the H2-H4 of the pair and of the base's own reading (runs/lfm3-s1s2, reported as it
is) come next to them, with the cascade cost: the mean s1 tokens before and after, 1/rho and w.

Amendment 6: G1 and G3 compare the fine-tuned model with the s1 files of eval/base, which are those of
the published 4-bit file or, when the smoke's conversion check said "converted", of the local
conversion; System 2 is always the published base (the pair's s2 files, from runs/lfm3-s1s2). 1/rho
divides by the base's mean tokens at the budget the pair's router chose on its calibration (summary.json
of eval/pair), not a fixed 1,024; before `route` it is not computed. gate.json and the report say which
conversion each number comes from.
"""
import argparse
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C  # noqa: E402  (its pure parts only: standard library)

N = {"test": 1000, "mmlu": 500}
FILES = {"s1": "test.s1.jsonl", "s2": "test.s2-1024.jsonl", "mmlu": "mmlu.s1.jsonl"}
ALPHA = 0.02  # the router's, docs/lfm3-s1s2-prereg.md
G2_NOT_READ = "not read (System 2 is the base)"  # amendment 5: the value G2 takes in gate.json and the report


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
    for name, rows, ids in (("base", base, b), ("ft", ft, f)):
        if len(rows) != len(ids):  # a record written twice would count twice in the rates
            problems.append(f"{name}: {len(rows) - len(ids)} duplicated ids")
    if b != f:
        problems.append(f"different ids ({len(b ^ f)} differ)")
    if len(b) != expected or len(f) != expected:
        problems.append(f"incomplete: base {len(b)}, ft {len(f)}, expected {expected}")
    return problems


def gates(base, ft, read_g2=True):
    """base and ft: {"s1": rows, "s2": rows, "mmlu": rows} -> the gate dict. read_g2=False is amendment 5:
    System 2 is the base model itself, G2 is not read and the s2 rows are not compared."""
    sb, sf = {k: stats(v) for k, v in base.items()}, {k: stats(v) for k, v in ft.items()}
    g = {"base": sb, "ft": sf, "problems": []}
    for k, exp in (("s1", N["test"]), ("s2", N["test"]), ("mmlu", N["mmlu"])):
        if k == "s2" and not read_g2:
            continue
        g["problems"] += [f"{k}: {p}" for p in check_pairs(base[k], ft[k], exp)]
    needed = FILES if read_g2 else ("s1", "mmlu")
    if any(sb[k] is None or sf[k] is None for k in needed):
        g.update(G1=None, G2=None if read_g2 else G2_NOT_READ, G3=None, improved=None, verdict="incomplete")
        return g
    g1a = sf["s1"]["acc"] >= sb["s1"]["acc"] + 0.03
    g1b = sf["s1"]["length_rate"] <= 0.5 * sb["s1"]["length_rate"]
    g3 = sf["mmlu"]["acc"] >= sb["mmlu"]["acc"] - 0.02
    g2 = G2_NOT_READ
    if read_g2:
        g2a = sf["s2"]["acc"] >= sb["s2"]["acc"] - 0.01
        g2b = sf["s2"]["mean_think_tokens"] <= 0.6 * sb["s2"]["mean_think_tokens"]
        g2 = {"pass": g2a and g2b, "acc": g2a, "think": g2b}
    g.update(G1={"pass": g1a and g1b, "acc": g1a, "length": g1b}, G2=g2, G3={"pass": g3},
             improved=None if g["problems"] else (g1a and g1b) and g3)  # incomplete: no verdict (§6)
    g["verdict"] = "incomplete" if g["problems"] else ("LFM3 improved" if g["improved"] else "not improved")
    return g


def probe_score(p):
    """The score of s1s2.py's fit_logistic, rebuilt from the parameters route writes in summary.json."""
    mu, sd, w, b = p["mu"], p["sd"], p["w"], p["b"]

    def score(x):
        z = [(x[j] - mu[j]) / sd[j] for j in range(len(mu))]
        return 1 / (1 + math.exp(-max(-30, min(30, b + sum(wj * zj for wj, zj in zip(w, z))))))

    return score


def w_share(summary, test_s1):
    """w of the cascade cost (amendment 6's wording): the share of the test questions whose probe score
    is under tau (at the chosen budget), whether System 2 then answers or not, System 2 being paid for
    each of them.

    Not 1 - s1_share: s1s2.py's route_one keeps the s1 answer when System 2 gave none, and counts it in
    s1_share, but it has paid System 2's tokens for that question (s1 + s2 in router_tokens)."""
    if not summary or not test_s1:
        return None
    b = summary.get("chosen_budget")
    e = (summary.get("budgets") or {}).get(str(b)) or {}
    if not e.get("test") or "probe" not in e:
        return None
    features = load_s1s2().features
    score, tau = probe_score(e["probe"]), e["tau"]
    return sum(score(features(r)) < tau for r in test_s1) / len(test_s1)


def load_s1s2():
    import importlib.util

    spec = importlib.util.spec_from_file_location("s1s2", os.path.join(HERE, "..", "lfm3-s1s2", "s1s2.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def cascade(ft_s1_tokens, base_s2_tokens, w, base_s1_tokens=None, budget=None):
    """The two terms of the cascade cost (amendments 5 and 6): 1/rho, the fine-tuned model's mean s1
    tokens over the base's mean tokens at `budget`, the budget the pair's router chose on its calibration,
    on the test; and w, the share sent to System 2. A cascade pays 1/rho + w against always thinking (the
    small model is paid on every question)."""
    if not ft_s1_tokens or not base_s2_tokens:
        return None
    return {"routed": True, "budget": budget, "one_over_rho": ft_s1_tokens / base_s2_tokens, "w": w,
            "ft_s1_tokens": ft_s1_tokens, "base_s1_tokens": base_s1_tokens, "base_s2_tokens": base_s2_tokens}


def pair_cascade(runs, summary, ft_s1_tokens, base_s1_tokens):
    """The cascade of the pair (amendment 6). Its budget is the pair's chosen_budget (summary.json of
    eval/pair) and its System 2 tokens are the mean of the pair's test.s2-<budget>.jsonl, the published
    base's generations. Before `route --tag pair` no budget is chosen: 1/rho and w are not computed."""
    pair = os.path.join(runs, "eval", "pair")
    b = (summary or {}).get("chosen_budget")
    if b is None:
        return {"routed": False, "budget": None, "one_over_rho": None, "w": None, "ft_s1_tokens": ft_s1_tokens,
                "base_s1_tokens": base_s1_tokens, "base_s2_tokens": None}
    s2 = stats(read_jsonl(os.path.join(pair, f"test.s2-{b}.jsonl")))
    return cascade(ft_s1_tokens, (s2 or {}).get("mean_tokens"),
                   w_share(summary, read_jsonl(os.path.join(pair, "test.s1.jsonl"))), base_s1_tokens, budget=b)


def read_json(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def provenance(runs):
    """Which weights each number comes from (amendment 6): the base's s1 files (eval/base/meta.json, or
    the smoke's conversion check when the meta is not written yet), the fine-tuned model (always its local
    conversion) and System 2 of the pair (eval/pair/meta.json: the published base of runs/lfm3-s1s2)."""
    meta = read_json(os.path.join(runs, "eval", "base", "meta.json")) or {}
    smoke = read_json(os.path.join(runs, "smoke", "report.json")) or {}
    source = meta.get("base_source") or smoke.get("base_source")
    base = meta.get("conversion") or C.conversion_of(source) or "unknown"
    pair = read_json(os.path.join(runs, "eval", "pair", "meta.json")) or {}
    s2 = sorted({v for k, v in (pair.get("conversions") or {}).items() if ".s2-" in k}) or [C.PUBLISHED_4BIT]
    return {"base_source": source, "base_s1": base, "ft_s1": C.LOCAL_CONVERSION, "system2": ", ".join(s2),
            "base_s1s2_router": C.PUBLISHED_4BIT}


def pair_mode(runs):
    """Amendment 5's local plan is on when the pair folder exists (eval/pair/meta.json, written by the
    `pair` step): System 2 is the base model itself, G2 is not read, and the pair's router reading is
    reported. The hf mode has no pair and reads G2 as §6 wrote it."""
    return os.path.exists(os.path.join(runs, "eval", "pair", "meta.json"))


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
    pv = g.get("provenance")
    L = ["# LFM3-1.7B fine-tuned against base (docs/lfm3-ft-prereg.md)", ""]
    if g["problems"]:
        L += ["**Problems:** " + "; ".join(g["problems"]), ""]
    if pv:
        L += ["## Which weights each number comes from (amendment 6)", "",
              f"- the base in s1 (G1 and G3, calibration, test and MMLU-Pro): {pv['base_s1']}",
              f"- the fine-tuned model in s1: {pv['ft_s1']}",
              f"- System 2 of the pair (the base in s2-256 and s2-1024, runs/lfm3-s1s2): {pv['system2']}",
              f"- the base's own router reading (runs/lfm3-s1s2): {pv['base_s1s2_router']}", ""]
    names = {"base": f"base ({pv['base_s1']})", "ft": f"ft ({pv['ft_s1']})"} if pv else {}
    L += ["| set | model | n | acc | length stops | mean think tok | mean tok |", "|---|---|---:|---:|---:|---:|---:|"]
    for k, label in (("s1", "test s1"), ("s2", "test s2-1024"), ("mmlu", "MMLU-Pro s1")):
        for name, s in (("base", sb[k]), ("ft", sf[k])):
            if s:
                shown = f"base ({pv['system2']})" if pv and k == "s2" and name == "base" else names.get(name, name)
                L.append(f"| {label} | {shown} | {s['n']} | {s['acc']:.3f} | {s['length_rate']:.3f} | "
                         f"{s['mean_think_tokens']:.0f} | {s['mean_tokens']:.0f} |")
    L.append("")
    if g["G1"] is not None:
        g2 = (f"- G2, s2-1024 accuracy -0.01 at most: {yn(g['G2']['acc'])}; thinking <= 0.6 x: {yn(g['G2']['think'])} "
              f"-> {yn(g['G2']['pass'])} (reported apart)" if isinstance(g["G2"], dict) else f"- G2: {g['G2']}")
        L += ["## Gates", "",
              f"- G1, s1 accuracy +0.03: {yn(g['G1']['acc'])}; length stops halved: {yn(g['G1']['length'])} -> {yn(g['G1']['pass'])}",
              g2,
              f"- G3, MMLU-Pro s1 -0.02 at most: {yn(g['G3']['pass'])}",
              f"- verdict: **{g['verdict']}**", ""]
    c = g.get("cascade")
    if c:
        conv = (lambda k: f" ({pv[k]})") if pv else (lambda k: "")
        after = f"- mean tokens on the test: s1 of the fine-tuned model{conv('ft_s1')} {c['ft_s1_tokens']:.0f}"
        before = f", s1 of the base{conv('base_s1')} {c['base_s1_tokens']:.0f}" if c.get("base_s1_tokens") else ""
        L += ["## Cascade cost (amendments 5 and 6)", ""]
        if c.get("one_over_rho") is None:
            L += [after + before,
                  "- 1/rho and w: the pair has not been routed yet (`route`); the budget of System 2 in 1/rho is the "
                  "one the pair's router chooses on its calibration", ""]
        else:
            b = c["budget"]
            L += [after + before + f", s2-{b} of the base{conv('system2')} {c['base_s2_tokens']:.0f}",
                  f"- 1/rho = {c['one_over_rho']:.3f} ({c['ft_s1_tokens']:.0f} / {c['base_s2_tokens']:.0f}, at s2-{b}, "
                  "the budget the pair's router chose on its calibration)"]
            if c["w"] is None:
                L.append("- w: not computed (the pair's summary holds no probe or no test reading at its budget)")
            else:
                L.append(f"- w = {c['w']:.3f} of the test questions sent to System 2 (the cascade pays "
                         f"1/rho + w = {c['one_over_rho'] + c['w']:.3f} against always thinking)")
            L.append("")
    for name in ("pair", "base_s1s2", "base", "ft"):
        r = g.get("router", {}).get(name)
        if r:
            label = {"pair": "the pair (fine-tuned in s1, base in s2)"
                     + (f": s1 {pv['ft_s1']}, s2 {pv['system2']}" if pv else ""),
                     "base_s1s2": "base, as read in runs/lfm3-s1s2" + (f" ({pv['base_s1s2_router']})" if pv else ""),
                     }.get(name, name)
            L += [f"## Router S1/S2, {label} (s2-{r['budget']})", "",
                  f"- s1 {r['s1_acc']:.3f}, s2 {r['s2_acc']:.3f}, router {r['router_acc']:.3f} "
                  f"({r['router_tokens']:.0f} tokens against {r['s2_tokens']:.0f})",
                  f"- H2 {yn(r['H2'])}, H3 {yn(r['H3'])}, H4 {yn(r['H4'])}: **{r['verdict']}**", ""]
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default="runs/lfm3-ft")
    ap.add_argument("--amendment5", action="store_true",
                    help="the local plan of amendment 5 (scripts/lfm3-ft.sh passes it in MODE=local): G2 not read, "
                         "even before the pair exists")
    a = ap.parse_args()
    read_g2 = not (a.amendment5 or pair_mode(a.runs))
    load = lambda tag: {k: read_jsonl(os.path.join(a.runs, "eval", tag, f)) for k, f in FILES.items()}
    base, ft = load("base"), load("ft")
    if not any(base.values()) or not any(ft.values()):
        sys.exit(f"no evaluation under {a.runs}/eval/{{base,ft}}: run `scripts/lfm3-ft.sh pull` first")
    g = gates(base, ft, read_g2=read_g2)
    g["router"] = {}
    for tag in ("base", "ft"):
        p = os.path.join(a.runs, "eval", tag, "summary.json")
        g["router"][tag] = router(json.load(open(p))) if os.path.exists(p) else None
    if not read_g2:  # amendment 5: the pair's reading, with the base's own (runs/lfm3-s1s2) next to it
        summ = read_json(os.path.join(a.runs, "eval", "pair", "summary.json"))
        g["router"]["pair"] = router(summ)
        q = os.path.join(os.path.dirname(a.runs.rstrip("/") or "."), "lfm3-s1s2", "summary.json")
        g["router"]["base_s1s2"] = router(read_json(q))
        g["provenance"] = provenance(a.runs)  # amendment 6
        g["cascade"] = pair_cascade(a.runs, summ, (g["ft"]["s1"] or {}).get("mean_tokens"),
                                    (g["base"]["s1"] or {}).get("mean_tokens"))
    with open(os.path.join(a.runs, "gate.json"), "w") as f:
        json.dump(g, f, indent=2)
    md = report_md(g)
    with open(os.path.join(a.runs, "report.md"), "w") as f:
        f.write(md)
    print(md)


if __name__ == "__main__":
    main()
