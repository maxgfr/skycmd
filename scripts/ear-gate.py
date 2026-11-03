#!/usr/bin/env python3
"""s6-ear: conformal calibration of the exits and of the System 2 hand-off, and the
pre-registered gates (docs/s6-ear-prereg.md §3, §4).

    python3 -I scripts/ear-gate.py crc <exits.jsonl> [--alpha 0.02] [--route-alpha 0.05]
                                   [--json out.json] [--print-sets]
    python3 -I scripts/ear-gate.py gate --eval eval.json --baseline baseline-smart-turn.json
                                   [--control eval-control.json] [--json gate.json]

`crc` reads `sky-eval --dump-exits` on the **val** split (full depth, every exit) and picks, per
question type:
  * the early-exit threshold tau (SPEC-S6 §9, the rule of crates/sky-arena/src/crc.rs): the loss
    of a decision at tau is 1 when the exit answering at tau disagrees with full depth; tau is the
    smallest value of the grid {1.0, 0.99, ..., 0.50} whose conformal bound (n R + 1) / (n + 1),
    and that of every larger tau, stays <= alpha;
  * the routing threshold rho (System 1 / System 2, as LFM3-1.7B in docs/lfm3-s1s2-prereg.md): s6-ear
    answers when its full-depth top probability is >= rho, else it says "I don't know" and hands
    the clip to a larger model (Whisper, Smart Turn). The loss at rho is 1 for an answered
    decision whose answer is wrong; rho is chosen by the same scan, with route-alpha.
  --print-sets prints the `gguf-set` arguments of the exit thresholds.

`gate` reads `sky-eval --json` of the final ternary model on the test split and the Smart Turn
baseline (`scripts/ear-baseline.py`) and checks the three v1 gates:
  1. turn end: accuracy on `ear:turn_end` >= Smart Turn v3.2's on the same clips;
  2. keywords: accuracy on `ear:keyword` (12 options, the official test set) >= 0.90;
  3. zero-shot keywords: accuracy on `ear:keyword_zs:en` and on `ear:keyword_zs:fr` (10 options,
     held-out words) each >= 2 x chance = 0.20.
The control (no Mimi pretraining) is reported next to the model, never gated. Exit status 1
when a gate fails.
"""
import argparse
import json
import sys

KW_BAR = 0.90
ZS_OPTIONS = 10
ZS_BAR = 2.0 / ZS_OPTIONS
SMART_TURN_PARAMS = 8_000_000
TYPES = ("choice", "score", "noul")


def grid():
    return [1.0] + [i / 100 for i in range(99, 49, -1)]


def argmax(v):
    return max(range(len(v)), key=lambda i: v[i])


def scan(losses_at, n, alpha):
    """Smallest tau of the grid whose bound, and every larger tau's, is <= alpha."""

    def at(tau):
        r = sum(losses_at(tau)) / n if n else 0.0
        return {"tau": tau, "risk": r, "bound": (n * r + 1) / (n + 1), "n": n}

    best = at(1.0)
    if n == 0 or best["bound"] > alpha:
        return best
    for tau in grid()[1:]:
        c = at(tau)
        if c["bound"] > alpha:
            break
        best = c
    return best


def crc(a):
    rows = [json.loads(l) for l in open(a.exits) if l.strip()]
    out = {}
    for t in TYPES:
        rs = [r for r in rows if r["t"] == t]
        n = len(rs)
        tops = [[(max(p), argmax(p)) for p in r["probs"]] for r in rs]

        def exit_losses(tau):
            for ex in tops:
                last = len(ex) - 1
                e = last if tau >= 1.0 else next((i for i, (top, _) in enumerate(ex) if top >= tau), last)
                yield int(ex[e][1] != ex[last][1])

        golds = [argmax(r["target"]) for r in rs]

        def route_losses(rho):
            for ex, g in zip(tops, golds):
                top, ans = ex[-1]
                yield int(top >= rho and ans != g)

        ex = scan(exit_losses, n, a.alpha)
        ro = scan(route_losses, n, a.route_alpha)
        answered = sum(1 for e in tops if e[-1][0] >= ro["tau"]) / n if n else 0.0
        ro["coverage"] = answered
        out[t] = {"n": n, "exit": ex, "route": ro}
    if a.json:
        with open(a.json, "w") as f:
            json.dump({"alpha": a.alpha, "route_alpha": a.route_alpha, "types": out}, f, indent=2)
            f.write("\n")
    if a.print_sets:
        print(" ".join(f"skycmd-s2.exit_threshold.{t}={out[t]['exit']['tau']}" for t in TYPES if out[t]["n"] > 0))
    else:
        for t in TYPES:
            o = out[t]
            if o["n"]:
                print(f"{t}: n {o['n']}, exit tau {o['exit']['tau']} (risk {o['exit']['risk']:.4f}), "
                      f"route rho {o['route']['tau']} (coverage {o['route']['coverage']:.3f}, "
                      f"error among answered {o['route']['risk']:.4f})")


def acc(ev, src):
    s = ev["sources"].get(src)
    return (s["accuracy"], s["n"]) if s else (None, 0)


def gate(a):
    ev = json.load(open(a.eval))
    base = json.load(open(a.baseline))
    ctl = json.load(open(a.control)) if a.control else None
    rows, ok = [], True

    def check(name, src, got_bar, bar_label):
        nonlocal ok
        got, n = acc(ev, src)
        passed = got is not None and n > 0 and got >= got_bar
        ok &= passed
        c = acc(ctl, src)[0] if ctl else None
        rows.append({"gate": name, "source": src, "n": n, "accuracy": got, "bar": got_bar,
                     "bar_from": bar_label, "control": c, "pass": passed})

    check("turn end", "ear:turn_end", base["accuracy"], f"Smart Turn v3.2, n {base['n']}")
    check("keywords (12)", "ear:keyword", KW_BAR, "pre-registered")
    for lang in ("en", "fr"):
        check(f"zero-shot keywords {lang}", f"ear:keyword_zs:{lang}", ZS_BAR, "2 x chance")
    size = {"n_params": ev.get("n_params"), "smart_turn_params": SMART_TURN_PARAMS,
            "ratio": SMART_TURN_PARAMS / ev["n_params"] if ev.get("n_params") else None}
    for r in rows:
        c = "" if r["control"] is None else f"  control {r['control']:.4f}"
        g = "n/a" if r["accuracy"] is None else f"{r['accuracy']:.4f}"
        print(f"{'PASS' if r['pass'] else 'FAIL'}  {r['gate']:<26} {g} vs {r['bar']:.4f} ({r['bar_from']}, n {r['n']}){c}")
    if size["ratio"]:
        print(f"size: {size['n_params']:,} parameters, Smart Turn / s6-ear = {size['ratio']:.1f}")
    if a.json:
        with open(a.json, "w") as f:
            json.dump({"pass": ok, "gates": rows, "size": size}, f, indent=2)
            f.write("\n")
    sys.exit(0 if ok else 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("crc")
    c.add_argument("exits")
    c.add_argument("--alpha", type=float, default=0.02)
    c.add_argument("--route-alpha", type=float, default=0.05)
    c.add_argument("--json")
    c.add_argument("--print-sets", action="store_true")
    g = sub.add_parser("gate")
    g.add_argument("--eval", required=True)
    g.add_argument("--baseline", required=True)
    g.add_argument("--control")
    g.add_argument("--json")
    a = ap.parse_args()
    crc(a) if a.cmd == "crc" else gate(a)


if __name__ == "__main__":
    main()
