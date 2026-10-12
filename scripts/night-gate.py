"""Gate verdict of a sky-arena --json report (docs/night-experiments.md).

flappy: 200/200 pipes and 0 crashes; rescue: rate >= 0.95; dodge: completion >= 95% of the
oracle's with < 100 vetoes; doom: completion >= 95% and kill rate >= 90% of the oracle's.
Exit status 0 when every game present in the report passes and all four are present.

Usage: python3 -I scripts/night-gate.py <arena.json> [--quiet]
"""

import json
import sys


def verdict(report):
    rows, ok = [], True
    for g in ("flappy", "rescue", "dodge", "doom"):
        if g not in report:
            rows.append(f"{g:6}: missing")
            ok = False
            continue
        m, o = report[g]["model"], report[g]["oracle"]
        lay = report[g].get("mean_layers_used") or 0
        p50 = report[g].get("p50_us", 0)
        if g == "flappy":
            good = m["mean_score"] >= 200 - 1e-6 and m["crashes"] == 0
            txt = f"{m['mean_score']:.1f}/200, {m['crashes']} crashes"
        elif g == "rescue":
            good = m["rescue_rate"] >= 0.95
            txt = f"rescue {m['rescue_rate']:.3f} (oracle {o['rescue_rate']:.3f})"
        elif g == "dodge":
            good = m["completion_rate"] >= 0.95 * o["completion_rate"] - 1e-6 and m["vetoes"] < 100
            txt = f"completion {m['completion_rate']:.3f} (oracle {o['completion_rate']:.3f}), {m['vetoes']} vetoes"
        else:
            good = (m["completion_rate"] >= 0.95 * o["completion_rate"] - 1e-6
                    and m["kill_rate"] >= 0.90 * o["kill_rate"] - 1e-6)
            txt = (f"completion {m['completion_rate']:.3f} (oracle {o['completion_rate']:.3f}), "
                   f"kills {m['kill_rate']:.3f} (oracle {o['kill_rate']:.3f}), {m['crashes']} deaths, {m['vetoes']} vetoes")
        ok &= good
        rows.append(f"{g:6}: {'pass' if good else 'FAIL'}  {txt}  | layers {lay:.2f}, p50 {p50:.0f} us")
    return ok, rows


if __name__ == "__main__":
    report = json.load(open(sys.argv[1]))
    ok, rows = verdict(report)
    if "--quiet" not in sys.argv:
        print(f"gate {sys.argv[1]}: {'PASS' if ok else 'FAIL'}")
        print("\n".join("  " + r for r in rows))
    sys.exit(0 if ok else 1)
