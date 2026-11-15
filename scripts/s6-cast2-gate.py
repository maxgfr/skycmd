"""The verdict of s6-cast v2 (docs/s6-cast2-prereg.md §7-§10), from the files the run leaves.

    python3 -I scripts/s6-cast2-gate.py checks --gift-eval-out data/gift-eval-out --out <run>/checks.json
    python3 -I scripts/s6-cast2-gate.py gate --run runs/s6-cast2-pico --gift-eval-out data/gift-eval-out \
        --reference-out data/gift-eval-out --chronos-out data/chronos-zs-out --models models/s6-cast2 \
        --g3 gate --data runs/s6-cast2-data --out runs/s6-cast2-pico/GATE.md [--diagnostics <json>] [--smoke]

`checks` is §7.1's validation of the harness, read before any v2 score: the seasonal naive's
nGMASE is 1.00 +- 0.02 and the TinyCast port's files give back 0.7856 / 0.5570. It exits 1 when
either fails.

`gate` writes the verdict: G1 (nGMASE <= 0.838 and nWQL <= 0.587), G2 (the geometric-mean MASE
ratio < 1.00 at each term), G3 (relative MASE and WQL on Chronos-ZS each <= 1.083 x the TinyCast
port's on the same harness, when gated), on the ternary GGUF without flip; then the reported
items of §9 and the recommendation sentence of §8. The disclosure of §10 follows every table.
Only the standard library is used; the per-window items of §9 (coverage, window-level splits,
horizon tenths) come from `scripts/s6-cast2-diagnostics.py` when its JSON is given.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SN_REF = REPO / "eval" / "gift-eval" / "seasonal_naive.csv"
PROPS = REPO / "eval" / "gift-eval" / "dataset_properties.json"
MASE = "eval_metrics/MASE[0.5]"
WQL = "eval_metrics/mean_weighted_sum_quantile_loss"
MSIS = "eval_metrics/MSIS"

# §8
G1_MASE, G1_WQL = 0.838, 0.587
G3_FACTOR = 1.083
# §7.1
SN_TOL = 0.02
PORT_FILES = (0.7856, 0.5570)
# §8, not gated: the published TinyCast
PUBLISHED_TINYCAST = (0.774, 0.545)

DISCLOSURE = (
    "*Disclosure (docs/s6-cast2-prereg.md §10).* v2 was designed after reading v1's GIFT-Eval "
    "results on the same 97 configurations, window by window: where it lost, by band, term, "
    "domain, seasonality, context length and horizon. GIFT-Eval is therefore partly in-sample "
    "for v2's design choices, though not for its weights: no GIFT-Eval series is trained on. "
    "TinyCast's authors make the same disclosure about their own design and treat Chronos-ZS as "
    "the cleaner test. G3 exists for this reason. Every report of v2 states this paragraph."
)


# §8 and amendment 2: the sentence the verdict carries once G3 is dropped
READ_IN_SAMPLE = (
    "G3 was dropped before training (amendment 2: the Chronos-ZS harness was not validated): G1 and G2 "
    "were read on the benchmark that shaped the design, and Chronos-ZS is reported, not gated."
)


def band(cfg: str) -> str:
    freq = cfg.split("/")[1]
    unit = re.sub(r"^\d+", "", freq).split("-")[0].upper()
    if unit == "S":
        return "second"
    if unit in ("T", "MIN"):
        return "sub-hourly"
    if unit == "H":
        return "hourly"
    if unit == "D":
        return "daily"
    if unit == "W":
        return "weekly"
    return "monthly and coarser"


BANDS = ["second", "sub-hourly", "hourly", "daily", "weekly", "monthly and coarser"]
TERMS = ["short", "medium", "long"]


def read_results(d: Path) -> dict[str, dict]:
    f = d / "all_results.csv"
    if not f.exists():
        return {}
    with open(f, newline="") as fh:
        return {r["dataset"]: r for r in csv.DictReader(fh)}


def reference() -> dict[str, dict]:
    with open(SN_REF, newline="") as fh:
        return {r["dataset"]: r for r in csv.DictReader(fh)}


def ratios(res: dict[str, dict], col: str) -> dict[str, float]:
    ref = reference()
    out = {}
    for cfg, r in res.items():
        try:
            num, den = float(r[col]), float(ref[cfg][col])
        except (KeyError, TypeError, ValueError):
            continue
        if num > 0 and den > 0 and math.isfinite(num) and math.isfinite(den):
            out[cfg] = num / den
    return out


def gmean(vals) -> float | None:
    vals = list(vals)
    return math.exp(sum(math.log(v) for v in vals) / len(vals)) if vals else None


def summary(res: dict[str, dict]) -> dict:
    return {name: gmean(ratios(res, col).values()) for name, col in (("nGMASE", MASE), ("nWQL", WQL), ("nMSIS", MSIS))} | {
        "configs": len(ratios(res, MASE))
    }


def fmt(v, nd=4) -> str:
    return "n/a" if v is None else f"{v:.{nd}f}"


def cmd_checks(a) -> None:
    out = Path(a.gift_eval_out)
    sn = summary(read_results(out / "seasonal-naive"))
    tc = summary(read_results(out / "tinycast"))
    ok_sn = sn["nGMASE"] is not None and abs(sn["nGMASE"] - 1.0) <= SN_TOL and sn["configs"] == 97
    ok_tc = (
        tc["nGMASE"] is not None
        and tc["configs"] == 97
        and round(tc["nGMASE"], 4) == PORT_FILES[0]
        and round(tc["nWQL"], 4) == PORT_FILES[1]
    )
    res = {
        "seasonal_naive": {**sn, "rule": "nGMASE 1.00 +- 0.02 over 97 configurations", "ok": ok_sn},
        "tinycast_port": {**tc, "rule": "the files give back 0.7856 / 0.5570", "ok": ok_tc},
        "ok": ok_sn and ok_tc,
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=2) + "\n")
    print(f"seasonal naive nGMASE {fmt(sn['nGMASE'])} ({sn['configs']} configs): {'ok' if ok_sn else 'FAIL'}")
    print(f"TinyCast port {fmt(tc['nGMASE'])} / {fmt(tc['nWQL'])}: {'ok' if ok_tc else 'FAIL'}")
    if not res["ok"]:
        print("§7.1: the harness is corrected before v2 is read", file=sys.stderr)
        sys.exit(1)


def grouped(res: dict[str, dict], key, col=MASE) -> dict[str, float | None]:
    r = ratios(res, col)
    groups: dict[str, list[float]] = {}
    for cfg, v in r.items():
        groups.setdefault(key(cfg), []).append(v)
    return {g: gmean(v) for g, v in groups.items()}


def table(rows: list[list[str]], head: list[str]) -> list[str]:
    out = ["| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in head) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return out + ["", DISCLOSURE, ""]


def last_record(log: Path, pred) -> dict | None:
    rec = None
    if log.exists():
        for line in log.open():
            try:
                v = json.loads(line)
            except json.JSONDecodeError:
                continue
            if pred(v):
                rec = v
    return rec


def load_json(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


def g3_holds(v2: dict | None, port_noflip: dict | None) -> bool | None:
    """G3 (§8, amendment 1): v2's relative MASE and WQL on Chronos-ZS, without flip, each at most
    1.083 times the TinyCast port's without flip. None when a score is missing."""
    if not v2 or not port_noflip:
        return None
    keys = ("relative_MASE", "relative_WQL")
    if any(v2.get(k) is None or port_noflip.get(k) is None for k in keys):
        return None
    return all(v2[k] <= G3_FACTOR * port_noflip[k] for k in keys)


def harness_validated_before(check: Path, started: Path) -> bool:
    """§8, amendment 1: G3 stands only if the Chronos-ZS harness check passed before training
    started (check.json written before the run's started.json)."""
    if not check.exists() or not started.exists():
        return False
    return bool(json.loads(check.read_text()).get("validated")) and os.path.getmtime(check) < os.path.getmtime(started)


def cmd_gate(a) -> None:
    run, gout, rout = Path(a.run), Path(a.gift_eval_out), Path(a.reference_out)
    name = run.name
    v2 = read_results(gout / name)
    v2f = read_results(gout / f"{name}-flip")
    v1 = read_results(rout / "s6-cast-pico")
    v1f = read_results(rout / "s6-cast-pico-flip")
    port = read_results(rout / "tinycast")
    portnf = read_results(gout / "tinycast-noflip")
    sn = read_results(rout / "seasonal-naive")
    checks = json.loads((run / "checks.json").read_text()) if (run / "checks.json").exists() else None
    lat_file = run / "latency" / "latency.json"
    latency = json.loads(lat_file.read_text()) if lat_file.exists() else None
    out = [f"# {name}: the gate of docs/s6-cast2-prereg.md", ""]
    if a.smoke:
        out += ["**SMOKE RUN.** A few minutes under a throwaway prefix, to test that the commands start and the "
                "files chain: no number below is a result, none is to be looked at.", ""]
    problems = []
    if not checks or not checks.get("ok"):
        problems.append("§7.1's checks are not on file or failed (checks.json)")
    if not latency:
        problems.append("§7.2's latencies are not on file (latency/latency.json)")
    s2 = summary(v2)
    if not a.smoke and s2["configs"] != 97:
        problems.append(f"v2 is scored on {s2['configs']} configurations, not 97")

    # G1, G2
    g1 = None if s2["nGMASE"] is None else (s2["nGMASE"] <= G1_MASE and s2["nWQL"] <= G1_WQL)
    term_v2 = grouped(v2, lambda c: c.split("/")[2])
    g2 = None if any(term_v2.get(t) is None for t in TERMS) else all(term_v2[t] < 1.0 for t in TERMS)

    # G3: Chronos-ZS, the TinyCast port with flip on the same harness (§7.3)
    # G3: Chronos-ZS, v2 without flip against the TinyCast port without flip on the same harness
    # (§7.3, amendment 1); the harness is the one the port with flip validated (check.json)
    cz = Path(a.chronos_out)
    g3_state = a.g3
    validated_before = harness_validated_before(cz / "tinycast" / "check.json", run / "started.json")
    czv2 = Path(a.chronos_v2_out) if a.chronos_v2_out else cz
    a2, ap = load_json(czv2 / name / "summary.json"), load_json(cz / "tinycast-noflip" / "summary.json")
    g3 = g3_holds(a2, ap)
    if a.g3 == "gate" and not validated_before:
        problems.append("G3 is gated but the Chronos-ZS harness was not validated before training started: "
                        "§8 has an amendment drop G3 (G3=reported)")
    gated = [g1, g2] + ([g3] if a.g3 == "gate" else [])
    if problems:
        verdict = "NOT READ: " + "; ".join(problems)
    elif any(g is None for g in gated):
        verdict = "INCOMPLETE: a gated item has no score"
    else:
        verdict = "PASS" if all(gated) else "FAIL"
    out += [f"**Verdict: {verdict}.**", ""]
    if a.g3 == "reported":
        out += [READ_IN_SAMPLE, ""]
    rows = [
        ["G1", f"nGMASE {fmt(s2['nGMASE'])} <= {G1_MASE} and nWQL {fmt(s2['nWQL'])} <= {G1_WQL}", str(g1)],
        ["G2", "MASE ratio < 1.00 at each term: " + ", ".join(f"{t} {fmt(term_v2.get(t), 3)}" for t in TERMS), str(g2)],
    ]
    if g3 is not None:
        g3_text = (f"v2 relative MASE {fmt(a2['relative_MASE'])} <= {G3_FACTOR} x {fmt(ap['relative_MASE'])} (the port without flip) and "
                   f"relative WQL {fmt(a2['relative_WQL'])} <= {G3_FACTOR} x {fmt(ap['relative_WQL'])}")
    else:
        g3_text = "no Chronos-ZS score on file"
    rows.append([f"G3 ({g3_state})", g3_text, str(g3) if a.g3 == "gate" else f"reported: {g3}"])
    out += ["## Gate (ternary GGUF, without flip, full harness)", ""]
    out += table(rows, ["item", "measured", "holds"])
    if a.g3 == "reported":
        out += [READ_IN_SAMPLE, ""]

    rows = []
    for label, base, d in (("v2, without flip", czv2, name), ("TinyCast port, without flip", cz, "tinycast-noflip"),
                           ("TinyCast port, with flip (harness check)", cz, "tinycast"),
                           ("seasonal naive", cz, "seasonal-naive")):
        sm = load_json(base / d / "summary.json")
        rows.append([label, fmt(sm and sm.get("relative_MASE")), fmt(sm and sm.get("relative_WQL")),
                     str(sm["tasks"]["MASE"]) if sm else "0"])
    chk = load_json(cz / "tinycast" / "check.json")
    out += ["## Chronos-ZS (27 tasks, relative to fev's seasonal naive)", "",
            f"Harness check (the port with flip within 0.02 of 0.880 / 0.722): "
            f"{'validated' if chk and chk.get('validated') else 'not validated'}"
            f"{', before training started' if validated_before else ''}.", ""]
    out += table(rows, ["model", "relative MASE", "relative WQL", "tasks"])

    # §9, GIFT-Eval detail
    models = [("v1", v1), ("v1 flip", v1f), ("v2", v2), ("v2 flip", v2f), ("TinyCast port (flip)", port),
              ("TinyCast port, no flip", portnf), ("seasonal naive", sn)]
    rows = []
    for label, res in models:
        s = summary(res)
        rows.append([label, fmt(s["nGMASE"]), fmt(s["nWQL"]), fmt(s["nMSIS"]), str(s["configs"])])
    out += ["## GIFT-Eval: nGMASE, nWQL, nMSIS", ""]
    out += table(rows, ["model", "nGMASE", "nWQL", "nMSIS", "configs"])
    sp = summary(port)
    snf = summary(portnf)
    rows = [["TinyCast port, with flip", fmt(s2["nGMASE"] and sp["nGMASE"] and s2["nGMASE"] - sp["nGMASE"]),
             fmt(s2["nWQL"] and sp["nWQL"] and s2["nWQL"] - sp["nWQL"])],
            ["TinyCast port, without flip", fmt(s2["nGMASE"] and snf["nGMASE"] and s2["nGMASE"] - snf["nGMASE"]),
             fmt(s2["nWQL"] and snf["nWQL"] and s2["nWQL"] - snf["nWQL"])],
            ["published TinyCast (0.774 / 0.545)", fmt(s2["nGMASE"] and s2["nGMASE"] - PUBLISHED_TINYCAST[0]),
             fmt(s2["nWQL"] and s2["nWQL"] - PUBLISHED_TINYCAST[1])]]
    out += ["## Gap of v2 (without flip) to TinyCast (not gated)", ""]
    out += table(rows, ["reference", "nGMASE gap", "nWQL gap"])

    props = json.loads(PROPS.read_text())
    key_of = lambda c: c.split("/")[0]
    for title, key in (("term", lambda c: c.split("/")[2]), ("band", band),
                       ("domain", lambda c: props.get(key_of(c), {}).get("domain", "?"))):
        keys = TERMS if title == "term" else (BANDS if title == "band" else sorted({key(c) for c in sn}))
        g = {lab: (grouped(res, key), grouped(res, key, WQL)) for lab, res in (("v1", v1), ("v2", v2), ("port", port))}
        n = {k: sum(1 for c in ratios(v2, MASE) if key(c) == k) for k in keys}
        rows = [[k, str(n[k])] + [f"{fmt(g[m][0].get(k), 3)} / {fmt(g[m][1].get(k), 3)}" for m in ("v1", "v2", "port")]
                for k in keys]
        out += [f"## GIFT-Eval by {title} (MASE / WQL ratio to the seasonal naive)", ""]
        out += table(rows, [title, "configs", "v1", "v2", "TinyCast port"])

    above = sorted((c, v) for c, v in ratios(v2, MASE).items() if v > 1.0)
    out += ["## Configurations where v2 is above the seasonal naive (MASE)", ""]
    out += table([[c, fmt(v, 3)] for c, v in above] or [["none", ""]], ["configuration", "MASE ratio"])

    rows = []
    for col, lab in ((MASE, "MASE"), (WQL, "WQL")):
        r1, r2, rp = ratios(v1, col), ratios(v2, col), ratios(port, col)
        common = sorted(set(r1) & set(r2) & set(rp))
        gap = sum(math.log(r1[c] / rp[c]) for c in common)
        closed = sum(math.log(r1[c] / r2[c]) for c in common)
        rows.append([lab, str(len(common)), fmt(gap), fmt(closed), fmt(closed / gap if gap else None, 3)])
    out += ["## Share of v1's gap to the TinyCast port closed by v2 (sum of log ratios)", ""]
    out += table(rows, ["metric", "configs", "log gap v1 / port", "closed by v2", "share closed"])

    # coverage, window-level splits, horizon tenths
    out += ["## Coverage and front-end diagnostics (per window)", ""]
    if a.diagnostics and Path(a.diagnostics).exists():
        d = json.loads(Path(a.diagnostics).read_text())
        for title, rows_, head in d.get("tables", []):
            out += [f"### {title}", ""]
            out += table(rows_, head)
    else:
        out += ["Not computed: `scripts/s6-cast2-diagnostics.py` has no output on file (`--diagnostics`).", "",
                DISCLOSURE, ""]

    # exits used, latency, size
    out += ["## Latency and size", ""]
    rows = []
    for label, res_dir in (("v2", gout / name), ("v1", rout / "s6-cast-pico")):
        ex = res_dir / "exits.json"
        if ex.exists():
            e = json.loads(ex.read_text())
            w = sum(v["windows"] - v["fallbacks"] for v in e.values())
            mean = sum(v["mean_exits_used"] * (v["windows"] - v["fallbacks"]) for v in e.values()) / max(w, 1)
            rows.append([label, f"{mean:.3f} of {next(iter(e.values()))['n_exits']}", str(w)])
        else:
            rows.append([label, "not recorded", ""])
    out += table(rows, ["model", "mean exits used on GIFT-Eval", "windows"])
    mdir = Path(a.models)
    sizes = []
    for f in (mdir / f"{name}.gguf", REPO / "models/s6-cast/s6-cast-pico.gguf", REPO / "crates/sky-infer/tests/golden/tinycast.gguf"):
        sizes.append([f.name, str(f.stat().st_size) if f.exists() else "missing"])
    out += table(sizes, ["file", "bytes"])
    if latency:
        rows = [["native", r["model"], r["depth"], str(r["horizon"]), str(r["median_us"]), str(r["p95_us"]), str(r["runs"])]
                for r in latency["native"]]
        rows += [["wasm", w["model"], "calibrated" if w["model"] == f"{name}.gguf" else "full", str(w["horizon"]),
                  f"{w['median_us']:.0f}", f"{w['p95_us']:.0f}", str(w["iters"])] for w in latency["wasm"]]
        out += [f"Measured {latency['measured']} on {latency['host']}, one core, 2,048-point history.", ""]
        out += table(rows, ["target", "model", "depth", "horizon", "median µs", "p95 µs", "forecasts"])
        out += [recommendation(latency, s2, sp, mdir / f"{name}.gguf"), ""]
    else:
        out += ["Latency not measured (latency/latency.json).", "", DISCLOSURE, ""]

    # training
    out += ["## Training", ""]
    rows = []
    st = run / "last" / "state.json"
    if st.exists():
        s = json.loads(st.read_text())
        rows.append(["steps done", str(s["step"])])
        rows.append(["working minutes", f"{s['elapsed_secs'] / 60:.1f}"])
    fin = last_record(run / "log.jsonl", lambda v: v.get("final"))
    if fin and "windows" in fin:
        rows += [[f"windows seen, {k}", str(v)] for k, v in fin["windows"].items()]
    exp = json.loads((run / "exported.json").read_text())["checkpoint"] if (run / "exported.json").exists() else "last"
    mode = "ema-ternary" if exp == "last-ema" else "ternary"
    vb = last_record(run / "log.jsonl", lambda v: v.get("mode") == mode and "val_by_band" in v)
    if vb:
        rows += [[f"validation pinball ({mode}), {g}", f"{x['pinball']:.5f} ({x['windows']} windows)"]
                 for g, x in vb["val_by_band"].items()]
    out += table(rows or [["no training files", ""]], ["item", "value"])
    brake = []
    for log in (run / "train.log",):
        if log.exists():
            seen: dict[str, int] = {}
            for line in log.open(errors="replace"):
                if line.startswith("brake: ") or line.startswith("cooling down"):
                    k = re.sub(r"for \d+ s", "for N s", line.strip())
                    seen[k] = seen.get(k, 0) + 1
            brake += [[k, str(n)] for k, n in seen.items()]
    out += ["### Brake modes", ""]
    out += table(brake or [["no change of mode logged", ""]], ["line", "times"])
    ddir = Path(a.data)
    rows = []
    for f in sorted([*Path("data/series").glob("*.jsonl"), *Path(a.series2).glob("*.jsonl")]):
        n = pts = 0
        for line in f.open():
            if line.strip():
                n += 1
                pts += len(json.loads(line)["values"])
        rows.append([str(f), str(n), str(pts)])
    leak = ddir / "leak.json"
    out += ["### Data sources", ""]
    out += table(rows or [["none found", "", ""]], ["file", "series", "points"])
    if leak.exists():
        lk = json.loads(leak.read_text())
        out += table([[s, str(v["series"]), str(v["runs"]), str(v["matching_runs"]), str(v["kept"])]
                      for s, v in lk["sources"].items()], ["source", "series", "runs", "matching runs", "kept"])
    Path(a.out).write_text("\n".join(out) + "\n")
    (Path(a.out).with_name("gate.json")).write_text(json.dumps({
        "verdict": verdict, "G1": g1, "G2": g2, "G3": g3, "G3_state": g3_state, "nGMASE": s2["nGMASE"],
        "nWQL": s2["nWQL"], "terms": term_v2, "smoke": a.smoke}, indent=2) + "\n")
    print(f"wrote {a.out}: {verdict}")


def recommendation(lat: dict, s2: dict, sp: dict, v2_file: Path) -> str:
    """§8: which of v2 and the port a user should take, in one sentence."""
    def med(model, target):
        rows = [r for r in lat[target] if (r.get("model") in (model,) or str(r.get("model", "")).startswith(model))
                and r.get("horizon") == 64 and (target == "wasm" or r.get("depth") == "full")]
        return rows[0]["median_us"] if rows else None
    v2n, pn = med("v2", "native"), med("tinycast", "native")
    port_size = (REPO / "crates/sky-infer/tests/golden/tinycast.gguf").stat().st_size
    v2_size = v2_file.stat().st_size if v2_file.exists() else None
    if None in (v2n, pn, v2_size, s2["nGMASE"], sp["nGMASE"]):
        return "Recommendation: not stated, a measurement is missing."
    slower_larger = v2n > pn and v2_size > port_size
    better = s2["nGMASE"] < sp["nGMASE"] and s2["nWQL"] < sp["nWQL"]
    if slower_larger and not better:
        return ("Recommendation: v2 is slower and larger than the TinyCast port and does not forecast better, "
                "so the port stays the recommendation, whatever the gate says (§8).")
    return (f"Recommendation: v2 at {v2n} µs and {v2_size} bytes against the port's {pn} µs and {port_size} "
            f"bytes, nGMASE {s2['nGMASE']:.4f} against {sp['nGMASE']:.4f}: "
            + ("v2 is the one to take." if better else "the port forecasts better; take it unless size or speed decide."))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("checks")
    c.add_argument("--gift-eval-out", required=True)
    c.add_argument("--out", required=True)
    g = sub.add_parser("gate")
    g.add_argument("--run", required=True)
    g.add_argument("--gift-eval-out", required=True)
    g.add_argument("--reference-out", required=True)
    g.add_argument("--chronos-out", required=True, help="the references' Chronos-ZS scores and the check")
    g.add_argument("--chronos-v2-out", help="v2's Chronos-ZS scores (default: --chronos-out)")
    g.add_argument("--series2", default="data/series2", help="the new real series")
    g.add_argument("--models", required=True)
    g.add_argument("--g3", choices=["gate", "reported"], default="reported",
                   help="reported: G3 dropped by amendment 2 of the pre-registration")
    g.add_argument("--data", required=True)
    g.add_argument("--out", required=True)
    g.add_argument("--diagnostics")
    g.add_argument("--smoke", action="store_true")
    a = p.parse_args()
    {"checks": cmd_checks, "gate": cmd_gate}[a.cmd](a)


if __name__ == "__main__":
    main()
