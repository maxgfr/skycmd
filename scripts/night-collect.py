"""Collects the night candidates' numbers into one table (docs/night-experiments.md).

For each candidate name: parameters (s2-export line of its pipeline log), file size of
models/candidates/<name>/<name>.gguf, held-out accuracy (runs/night-<name>/heldout.md, early-exit
accuracy of the GGUF on the test split without doom, and typed_decisions_bench), the closed-loop
gate (runs/night-<name>/gate.json: 30 episodes per game at the calibrated tau), mean layers and
sky-arena p50, plus the native sky-bench medians (runs/night-<name>/bench-*.json).

Usage: python3 -I scripts/night-collect.py name [name ...] [--json out.json]
A name may be `ship-s3-pico` for the shipped baseline (runs/night-baseline/...).
"""

import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from importlib import util as _u  # noqa: E402

_spec = _u.spec_from_file_location("night_gate", os.path.join(os.path.dirname(os.path.abspath(__file__)), "night-gate.py"))
gate = _u.module_from_spec(_spec)
_spec.loader.exec_module(gate)


def paths(name, spec=None):
    """Files of a candidate; `spec` (from --spec candidates.json) overrides any of them."""
    if name == "ship-s3-pico":
        b = "runs/night-baseline"
        p = dict(gguf="models/ship/s3-pico.gguf", gate="runs/ship-s3-pico.arena.json", heldout=f"{b}/s3-pico-heldout.md",
                 bench=f"{b}/s3-pico", log=None, params=445345)
    else:
        r = f"runs/night-{name}"
        p = dict(gguf=f"models/candidates/{name}/{name}.gguf", gate=f"{r}/gate.json", heldout=f"{r}/heldout.md",
                 bench=f"{r}/bench", log=f"{r}/pipeline.log", params=None)
    p.update((spec or {}).get(name, {}))
    return p


def params_of(p):
    if p["params"]:
        return p["params"]
    try:
        m = re.findall(r"exported \S+ \((\d+) parameters", open(p["log"]).read())
        return int(m[-1]) if m else None
    except OSError:
        return None


def heldout(path):
    """(early-exit acc all, typed_decisions_bench, last-exit acc all, tau) of the GGUF section."""
    try:
        text = open(path).read()
    except OSError:
        return None
    m = re.search(r"\| GGUF `[^`]+` \| ([\d.]+) \| ([\d.]+) \| ([\d./]+) \| ([\d.]+) \|", text)
    sec = text[text.find("## GGUF"):]
    early = sec[sec.find("#### Early exit per source"):]
    tdb = re.search(r"\| typed_decisions_bench \| \d+ \| ([\d.]+) \|", early)
    return dict(early=float(m.group(2)), last=float(m.group(1)), tau=m.group(3), layers=float(m.group(4)),
                typed_bench=float(tdb.group(1)) if tdb else None) if m else None


def bench(prefix, game):
    try:
        j = json.load(open(f"{prefix}-{game}.json"))
    except OSError:
        return None
    return j


def row(name, spec=None):
    p = paths(name, spec)
    out = dict(name=p.get("label", name), params=params_of(p))
    out["kb"] = round(os.path.getsize(p["gguf"]) / 1000, 1) if os.path.exists(p["gguf"]) else None
    out["heldout"] = heldout(p["heldout"])
    try:
        g = json.load(open(p["gate"]))
        ok, rows = gate.verdict(g)
        out["gate"] = ok
        out["games"] = {k: dict(model=g[k]["model"], oracle=g[k]["oracle"], layers=g[k]["mean_layers_used"],
                                p50=g[k]["p50_us"]) for k in ("flappy", "rescue", "dodge", "doom") if k in g}
        out["tau"] = g.get("exit_threshold")
    except OSError:
        out["gate"] = None
    out["bench"] = {gm: bench(p["bench"], gm) for gm in ("flappy", "doom")}
    return out


def f3(x):
    return '-' if x is None else f"{x:.3f}"


def fmt(r):
    h = r["heldout"] or {}
    gm = r.get("games") or {}

    def game(k):
        if k not in gm:
            return "-"
        m, o = gm[k]["model"], gm[k]["oracle"]
        if k == "flappy":
            return f"{m['mean_score']:.0f}/200, {m['crashes']} crash"
        if k == "rescue":
            return f"{m['rescue_rate']:.2f} (o {o['rescue_rate']:.2f})"
        if k == "dodge":
            return f"{m['completion_rate']:.3f} (o {o['completion_rate']:.3f}), {m['vetoes']} veto"
        return f"{m['completion_rate']:.2f}/{m['kill_rate']:.2f} (o {o['completion_rate']:.2f}/{o['kill_rate']:.2f})"

    layers = "/".join(f"{gm[k]['layers']:.2f}" for k in ("flappy", "rescue", "dodge", "doom") if k in gm) or "-"
    p50 = "/".join(f"{gm[k]['p50']:.0f}" for k in ("flappy", "rescue", "dodge", "doom") if k in gm) or "-"
    b = r["bench"]
    nat = "/".join(f"{b[k]['median_us']:.1f}" if b.get(k) and "median_us" in b[k] else "-" for k in ("flappy", "doom"))
    g = {True: "**pass**", False: "fail", None: "-"}[r["gate"]]
    return (f"| {r['name']} | {format(r['params'], ',') if r['params'] else '-'} | {r['kb'] or '-'} | {f3(h.get('early'))} | {f3(h.get('typed_bench'))} | "
            f"{game('flappy')} | {game('rescue')} | {game('dodge')} | {game('doom')} | {layers} | {p50} | {nat} | {g} |")


if __name__ == "__main__":
    argv = sys.argv[1:]
    spec = json.load(open(argv[argv.index("--spec") + 1])) if "--spec" in argv else None
    skip = {argv[i + 1] for i, a in enumerate(argv) if a in ("--spec", "--json")}
    args = [a for a in argv if not a.startswith("--") and a not in skip]
    rows = [row(n, spec) for n in args]
    print("| candidate | params | KB | held-out acc (no doom) | typed_decisions_bench | flappy | rescue | dodge completion | doom completion/kills | mean layers f/r/d/doom | arena p50 µs f/r/d/doom | sky-bench µs flappy/doom | gate |")
    print("|---|---:|---:|---:|---:|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(fmt(r))
    if "--json" in sys.argv:
        json.dump(rows, open(sys.argv[sys.argv.index("--json") + 1], "w"), indent=2)
