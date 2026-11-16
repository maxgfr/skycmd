#!/usr/bin/env python3
"""The rules and the verdict of s6-ear v2 (docs/s6-ear2-prereg.md §4.4, §6, §7, §8, §9), from the
files the run leaves. Only the standard library is used.

    python3 -I scripts/turn-gate.py probe --native N.jsonl --wasm W.jsonl --out rung.json
    python3 -I scripts/turn-gate.py data --data DIR [--shards 83]
    python3 -I scripts/turn-gate.py pilot --dir runs/s6-ear2-pilot --base RATE [--rates-only] [--out choice.json]
    python3 -I scripts/turn-gate.py checkpoint --log runs/s6-ear2-nano/log.jsonl
    python3 -I scripts/turn-gate.py calibrate --probs val-uncal.jsonl --store DIR/train --out calib.json
    python3 -I scripts/turn-gate.py harness --baseline smart-turn.json --out check.json
    python3 -I scripts/turn-gate.py latency --native N.jsonl --wasm W.jsonl --smart-turn S.json --model F.gguf --out latency.json
    python3 -I scripts/turn-gate.py gate --run runs/s6-ear2-nano --test-probs P.jsonl --labels data/ear/clips/smart-turn.jsonl \\
        --harness check.json --smart-turn-dump D.jsonl --latency latency.json --parity parity.json --model F.gguf \\
        --calib calib.json --val-probs val.jsonl --data DIR --eotbench-probs E.jsonl --pilot choice.json \\
        --probe rung.json --out GATE.md [--smoke --labels-store DIR/train]

* probe (§4.4): the first rung of R1-R4 whose WASM median decision time is <= 4.0 ms; exit 1 when
  none qualifies.
* pilot (§6.4-6.5): the best of the three distilled rates by validation accuracy at tau 0.5 on f32
  weights (`--rates-only` prints it); distil unless the hard-label arm is higher by more than
  0.005; the stop rule: the best arm below 0.82 stops v2 (exit 4).
* checkpoint (§7): `last-ema` if the EMA's final ternary validation BCE is below last's, else `last`.
* calibrate (§7): the temperature by validation NLL; tau50 = 0.5; tau85 = the highest threshold on
  the calibrated probability whose validation END recall is >= 0.85.
* harness (§8.1): Smart Turn v3.2 CPU through the fixed harness gives 0.9263 +- 0.003 on the 31,527
  test clips, and the always-finished rule the test's share of finished clips (0.4979); exit 1
  otherwise (no v2 test score is read then).
* gate (§9): G1 accuracy at tau50 >= 0.85 and >= T - 0.08; G2 at tau85 false cut-off <= 0.20 and
  END recall >= 0.80; G3 WASM median <= 5.0 ms, front end <= 0.10 ms per hop, GGUF <= 3 MB, WASM
  parity. Refuses to run before the harness check and the latencies are on file (§13). Writes
  GATE.md and gate.json with §8.2-8.4, §10 and the disclosure of §11.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import random
import struct
import sys
import time
from collections import defaultdict
from pathlib import Path

# §4.4
PROBE_MS = 4.0
RUNGS = ["R1", "R2", "R3", "R4"]
# §6
PILOT_MARGIN = 0.005
STOP_BAR = 0.82
# §7
RECALL_TARGET = 0.85
# §8.1
ST_BAR, ST_TOL = 0.9263, 0.003
TIMER_SHARE = 0.4979
TEST_CLIPS = 31527
# §9
G1_BAR, G1_GAP = 0.85, 0.08
G2_FC, G2_RECALL = 0.20, 0.80
G3_MS, G3_HOP_MS, G3_BYTES, PARITY_TOL, PARITY_CLIPS = 5.0, 0.10, 3_000_000, 1e-4, 20
# §8.4
EOT_GAP = 0.10
BOOT = 1000
# the teacher record of sky_infer::turn::store
TEACHER_RECORD = 4 + 384 * 2 + 25 * 384 + 25 * 4

DISCLOSURE = (
    "**Disclosure (§11).** v2 was designed after v1's test results. Its design study read measurements "
    "made on 3,000 clips of the same test set: v1's two models per source, Smart Turn per source, Smart "
    "Turn with 8, 4, 3, 2 and 1 s of context, and the trailing silence of 1,500 clips. The 4 s window was "
    "chosen from Smart Turn's accuracy on those test clips, and the data change was motivated by "
    "per-source test results. The official test set is therefore partly in-sample for v2's design "
    "choices, though never for its weights: no test clip is trained on, and no threshold is chosen on "
    "test. eot-bench, which the design never read, is the cleaner check, and it is reported (§8.4)."
)
DEPLOYMENT = (
    "Operating notes (§10): run the model once when the VAD reports a pause (Silero, or the engine's "
    "energy VAD in the demo) with Pipecat's `stop_secs` 0.2; answer \"finished\" when the probability is "
    "at least the threshold (tau50 for accuracy, tau85 for 85% END recall on validation), else keep "
    "listening; a fallback timer ends the turn after a longer silence (Pipecat's "
    "`SmartTurnParams.stop_secs`); if speech resumes, the turn's audio is kept and the model runs again "
    "at the next pause."
)
EOT_NOTE = (
    "These numbers are not comparable with LiveKit's FC@300 / FC@600, which come from a policy sweep; "
    "this harness cuts each labelled pause at its start and decides eot or hold once."
)
BASELINES = [
    ("silence timer (always \"finished\")", "timer"),
    ("s6-ear-nano (v1)", 0.5681),
    ("its control", 0.7647),
    ("Smart Turn v3.2 CPU, as checked (§8.1)", "teacher"),
    ("Smart Turn v3.2 GPU (Pipecat's published)", 0.9371),
    ("v1's wrong harness number (right padding), marked as such", 0.8518),
]


def read_jsonl(path):
    out = []
    with open(path) as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def write_json(path, v):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(v, f, indent=2)
        f.write("\n")


def sigmoid(z):
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def stem(path):
    return Path(str(path)).name.split(".")[0]


# ----------------------------------------------------------------------------- metrics


class Scores:
    """Probabilities and labels of a set of clips, with the weighted statistics a bootstrap needs."""

    def __init__(self, p, y, taus=(0.5,), bins=10):
        self.p = [float(v) for v in p]
        self.y = [bool(v) for v in y]
        self.n = len(self.p)
        self.taus = list(taus)
        self.pred = [[pi >= t for pi in self.p] for t in self.taus]
        self.order = sorted(range(self.n), key=lambda i: self.p[i])
        self.bin = [min(int(pi * bins), bins - 1) for pi in self.p]
        self.bins = bins

    def stats(self, w=None):
        n, y, p = self.n, self.y, self.p
        if w is None:
            w = [1] * n
        tot = sum(w)
        pos = sum(wi for wi, yi in zip(w, y) if yi)
        neg = tot - pos
        out = {"n": tot, "positive_rate": pos / tot if tot else float("nan")}
        for t, pred in zip(self.taus, self.pred):
            tp = sum(wi for wi, yi, ri in zip(w, y, pred) if yi and ri)
            fp = sum(wi for wi, yi, ri in zip(w, y, pred) if ri and not yi)
            fn = pos - tp
            tn = neg - fp
            prec = tp / (tp + fp) if tp + fp else 0.0
            rec = tp / pos if pos else float("nan")
            k = f"@{t:g}"
            out[f"accuracy{k}"] = (tp + tn) / tot if tot else float("nan")
            out[f"precision{k}"] = prec
            out[f"recall{k}"] = rec
            out[f"f1{k}"] = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
            out[f"fpr_all{k}"] = fp / tot if tot else float("nan")
            out[f"fnr_all{k}"] = fn / tot if tot else float("nan")
            out[f"false_cutoff{k}"] = fp / neg if neg else float("nan")
        # AUC (ties shared)
        if pos and neg:
            acc_neg, s, i = 0.0, 0.0, 0
            o = self.order
            while i < n:
                j = i
                gp = gn = 0.0
                while j < n and p[o[j]] == p[o[i]]:
                    if y[o[j]]:
                        gp += w[o[j]]
                    else:
                        gn += w[o[j]]
                    j += 1
                s += gp * (acc_neg + 0.5 * gn)
                acc_neg += gn
                i = j
            out["auc"] = s / (pos * neg)
        else:
            out["auc"] = float("nan")
        # ECE over equal-width bins of the probability
        cnt = [0.0] * self.bins
        sp = [0.0] * self.bins
        sy = [0.0] * self.bins
        for wi, b, pi, yi in zip(w, self.bin, p, y):
            if wi:
                cnt[b] += wi
                sp[b] += wi * pi
                sy[b] += wi * yi
        out["ece"] = sum(abs(sp[b] - sy[b]) for b in range(self.bins) if cnt[b]) / tot if tot else float("nan")
        return out


def bootstrap(sc: Scores, reps=BOOT, seed=0):
    """Point estimates and 95% percentile intervals by resampling clips."""
    point = sc.stats()
    rng = random.Random(seed)
    draws = defaultdict(list)
    n = sc.n
    for _ in range(reps if n else 0):
        w = [0] * n
        for i in rng.choices(range(n), k=n):
            w[i] += 1
        for k, v in sc.stats(w).items():
            if k != "n" and not (isinstance(v, float) and math.isnan(v)):
                draws[k].append(v)
    out = {}
    for k, v in point.items():
        d = sorted(draws.get(k, []))
        if d:
            out[k] = {"value": v, "lo": d[int(0.025 * (len(d) - 1))], "hi": d[int(math.ceil(0.975 * (len(d) - 1)))]}
        else:
            out[k] = {"value": v}
    return out


def oracle_point(p, y, target=RECALL_TARGET):
    """Speko's protocol: the least false cut-off with END recall >= target, the threshold chosen on
    the evaluated set itself (reported only)."""
    tau = tau_for_recall(p, y, target)
    if tau is None:
        return None
    sc = Scores(p, y, taus=(tau,)).stats()
    k = f"@{tau:g}"
    return {"threshold": tau, "false_cutoff": sc[f"false_cutoff{k}"], "end_recall": sc[f"recall{k}"],
            "accuracy": sc[f"accuracy{k}"], "n": len(p)}


def tau_for_recall(p, y, target=RECALL_TARGET):
    """The highest threshold whose END recall (share of finished clips with p >= tau) is >= target."""
    pos = sorted((pi for pi, yi in zip(p, y) if yi), reverse=True)
    if not pos:
        return None
    k = max(1, math.ceil(target * len(pos) - 1e-9))
    return pos[k - 1]


# ----------------------------------------------------------------------------- probe


def cmd_probe(a):
    native = {stem(r["model"]): r for r in read_jsonl(a.native)}
    wasm = {stem(r.get("path", r["model"])): r for r in read_jsonl(a.wasm)}
    table, rung = [], None
    for r in RUNGS:
        w, nat = wasm.get(r), native.get(r)
        row = {"rung": r,
               "wasm_median_ms": w["median_us"] / 1000 if w else None,
               "wasm_p95_ms": w["p95_us"] / 1000 if w else None,
               "wasm_frontend_ms_per_hop": w["frontend_us_per_hop"] / 1000 if w else None,
               "native_median_ms": nat["median_us"] / 1000 if nat else None,
               "native_p95_ms": nat["p95_us"] / 1000 if nat else None,
               "native_frontend_ms_per_hop": nat["frontend_us_per_hop"] / 1000 if nat else None,
               "params": (w or nat or {}).get("params"), "file_bytes": (w or nat or {}).get("file_bytes")}
        row["qualifies"] = row["wasm_median_ms"] is not None and row["wasm_median_ms"] <= PROBE_MS
        if rung is None and row["qualifies"]:
            rung = r
        table.append(row)
    out = {"rung": rung, "rule": f"the first rung of R1-R4 whose WASM median is <= {PROBE_MS} ms (§4.4)",
           "table": table, "measured": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "host": host()}
    if a.out:
        write_json(a.out, out)
    print(json.dumps({"rung": rung, "medians_ms": {t["rung"]: t["wasm_median_ms"] for t in table}}))
    return 0 if rung else 1


def host():
    try:
        import subprocess

        cpu = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
    except OSError:
        cpu = ""
    return cpu or platform.processor() or platform.machine()


# ----------------------------------------------------------------------------- data


def cmd_data(a):
    d = Path(a.data) / "train"
    missing = [i for i in range(a.shards)
               if not (d / f"manifest-{i:05d}.jsonl").exists() or not (d / f"teacher-{i:05d}.done").exists()]
    leak = json.load(open(d / "LEAK.json")) if (d / "LEAK.json").exists() else {}
    print(json.dumps({"shards": a.shards, "missing": missing, "leak": leak.get("total")}))
    return 1 if missing else 0


# ----------------------------------------------------------------------------- pilot, checkpoint


def final_val(log, mode):
    """The last validation entry of `mode` in a run log."""
    best = None
    for v in read_jsonl(log):
        if v.get("mode") == mode and "val" in v:
            best = v["val"]
    return best


def cmd_pilot(a):
    d = Path(a.dir)
    rates = {}
    for mult in ("0.5", "1", "2"):
        log = d / f"lr{mult}" / "log.jsonl"
        v = final_val(log, "f32") if log.exists() else None
        if v is None:
            raise SystemExit(f"{log}: no final f32 validation")
        rates[mult] = v["acc"]
    best_mult = max(rates, key=lambda m: (rates[m], -float(m)))
    lr = float(f"{a.base * float(best_mult):.4g}")
    if a.rates_only:
        print(f"{lr:.4g}")
        return 0
    hlog = d / "hard" / "log.jsonl"
    hv = final_val(hlog, "f32") if hlog.exists() else None
    if hv is None:
        raise SystemExit(f"{hlog}: no final f32 validation")
    distil_acc, hard_acc = rates[best_mult], hv["acc"]
    distil = not (hard_acc > distil_acc + PILOT_MARGIN)
    best = max(list(rates.values()) + [hard_acc])
    stop = best < STOP_BAR
    out = {"lr": lr, "rate_multiplier": float(best_mult), "base_rate": a.base, "distil": distil,
           "stop": stop, "best_val_acc": best,
           "arms": {**{f"lr{m}": {"val_acc_f32": acc, "lr": float(f"{a.base * float(m):.4g}"), "distil": True}
                       for m, acc in rates.items()},
                    "hard": {"val_acc_f32": hard_acc, "lr": lr, "distil": False}},
           "rules": {"choice": "validation accuracy at tau 0.5 on f32 weights (§6.4)",
                     "distil": f"distil unless the hard-label arm is higher by more than {PILOT_MARGIN}",
                     "stop": f"the best arm below {STOP_BAR} stops v2 (§6.5)"}}
    if a.out:
        write_json(a.out, out)
    print(f"pilot: best rate x{best_mult} ({lr:.4g}), distil {distil} ({distil_acc:.4f} vs hard {hard_acc:.4f}), "
          f"best {best:.4f}, stop {stop}")
    return 4 if stop else 0


def cmd_checkpoint(a):
    e, l = final_val(a.log, "ema-ternary"), final_val(a.log, "ternary")
    if l is None:
        raise SystemExit(f"{a.log}: no final ternary validation")
    print("last-ema" if e is not None and e["bce"] < l["bce"] else "last")
    return 0


# ----------------------------------------------------------------------------- calibrate


def store_meta(store):
    """id -> manifest line of every part of a feature store."""
    out = {}
    for p in sorted(Path(store).glob("manifest-*.jsonl")):
        tag = p.name[len("manifest-"):-len(".jsonl")]
        for i, v in enumerate(read_jsonl(p)):
            v["_tag"], v["_index"] = tag, i
            out[v["id"]] = v
    return out


def nll(z, y, t):
    s = 0.0
    for zi, yi in zip(z, y):
        x = zi / t
        # softplus(x) - y x
        s += max(x, 0.0) + math.log1p(math.exp(-abs(x))) - (x if yi else 0.0)
    return s / len(z)


def fit_temperature(z, y):
    """Golden-section search of log T over [log 0.05, log 20] minimizing the NLL."""
    lo, hi = math.log(0.05), math.log(20.0)
    g = (math.sqrt(5) - 1) / 2
    a, b = hi - g * (hi - lo), lo + g * (hi - lo)
    fa, fb = nll(z, y, math.exp(a)), nll(z, y, math.exp(b))
    for _ in range(80):
        if fa < fb:
            hi, b, fb = b, a, fa
            a = hi - g * (hi - lo)
            fa = nll(z, y, math.exp(a))
        else:
            lo, a, fa = a, b, fb
            b = lo + g * (hi - lo)
            fb = nll(z, y, math.exp(b))
    return math.exp((lo + hi) / 2)


def cmd_calibrate(a):
    meta = store_meta(a.store)
    z, y = [], []
    for r in read_jsonl(a.probs):
        m = meta.get(r["id"])
        if m is None or m.get("split") != "val" or "logit" not in r or not isinstance(m.get("label"), bool):
            continue
        z.append(float(r["logit"]))
        y.append(m["label"])
    if not z:
        raise SystemExit("no validation clip with a label and a logit")
    t = fit_temperature(z, y)
    p = [sigmoid(zi / t) for zi in z]
    tau85 = tau_for_recall(p, y)
    sc = Scores(p, y, taus=(0.5, tau85)).stats()
    out = {"temperature": t, "tau50": 0.5, "tau85": tau85,
           "val": {"n": len(z), "nll_t1": nll(z, y, 1.0), "nll": nll(z, y, t),
                   "accuracy@0.5": sc["accuracy@0.5"], "end_recall@tau85": sc[f"recall@{tau85:g}"],
                   "false_cutoff@tau85": sc[f"false_cutoff@{tau85:g}"]},
           "rules": "temperature by validation NLL; tau50 = 0.5 on the calibrated probability; tau85 = the "
                    "highest threshold whose validation END recall is >= 0.85 (§7); neither ever from test"}
    write_json(a.out, out)
    print(json.dumps({k: out[k] for k in ("temperature", "tau50", "tau85")}))
    return 0


# ----------------------------------------------------------------------------- harness, latency


def cmd_harness(a):
    b = json.load(open(a.baseline))
    checks = {
        "bar": {"value": b["accuracy"], "want": f"{ST_BAR} +- {ST_TOL}", "ok": abs(b["accuracy"] - ST_BAR) <= ST_TOL},
        "timer": {"value": b.get("timer_accuracy", b.get("positive_rate")), "want": TIMER_SHARE,
                  "ok": round(b.get("timer_accuracy", b.get("positive_rate", -1)), 4) == TIMER_SHARE},
        "clips": {"value": b["n"], "want": TEST_CLIPS, "ok": b["n"] == TEST_CLIPS},
        "padding": {"value": b.get("pad", "left"), "want": "left", "ok": b.get("pad", "left") == "left"},
    }
    passed = all(c["ok"] for c in checks.values())
    out = {"passed": passed, "teacher_accuracy": b["accuracy"], "checks": checks, "baseline": str(a.baseline),
           "rule": "§8.1: Smart Turn v3.2 CPU through the fixed harness gives 0.9263 +- 0.003 on the 31,527 test "
                   "clips and the always-finished rule the test's share of finished clips (0.4979)",
           "checked": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    write_json(a.out, out)
    print(json.dumps({"passed": passed, "accuracy": b["accuracy"], "timer": checks["timer"]["value"]}))
    return 0 if passed else 1


def cmd_latency(a):
    nat = read_jsonl(a.native)[-1]
    w = read_jsonl(a.wasm)[-1]
    st = json.load(open(a.smart_turn)) if a.smart_turn and os.path.exists(a.smart_turn) else None
    out = {"measured": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "host": host(),
           "native": {"median_ms": nat["median_us"] / 1000, "p95_ms": nat["p95_us"] / 1000,
                      "frontend_ms_per_hop": nat["frontend_us_per_hop"] / 1000, "decisions": nat.get("decisions")},
           "wasm": {"median_ms": w["median_us"] / 1000, "p95_ms": w["p95_us"] / 1000,
                    "frontend_ms_per_hop": w["frontend_us_per_hop"] / 1000, "decisions": w.get("iters")},
           "smart_turn_cpu_native": st,
           "gguf_bytes": os.path.getsize(a.model)}
    write_json(a.out, out)
    print(json.dumps({"wasm_median_ms": out["wasm"]["median_ms"], "gguf_bytes": out["gguf_bytes"]}))
    return 0


# ----------------------------------------------------------------------------- gate


def teacher_logit(data, meta):
    """The teacher's logit of a training-store clip (its record's first 4 bytes)."""
    with open(Path(data) / "train" / f"teacher-{meta['_tag']}.bin", "rb") as f:
        f.seek(meta["_index"] * TEACHER_RECORD)
        return struct.unpack("<f", f.read(4))[0]


def fmt(m, k, digits=4):
    v = m.get(k)
    if v is None:
        return "-"
    if "lo" in v:
        return f"{v['value']:.{digits}f} [{v['lo']:.{digits}f}, {v['hi']:.{digits}f}]"
    return f"{v['value']:.{digits}f}"


def check_g1(acc, teacher):
    return acc >= G1_BAR and acc >= teacher - G1_GAP


def check_g2(fc, recall):
    return fc <= G2_FC and recall >= G2_RECALL


def check_g3(lat, parity):
    terms = {
        "wasm_median": lat is not None and lat["wasm"]["median_ms"] <= G3_MS,
        "frontend": lat is not None and lat["wasm"]["frontend_ms_per_hop"] <= G3_HOP_MS,
        "size": lat is not None and lat["gguf_bytes"] <= G3_BYTES,
        "parity": parity is not None and parity.get("n", 0) >= PARITY_CLIPS
        and parity.get("max_abs_diff", 1.0) <= PARITY_TOL and parity.get("same_decision") == parity.get("n"),
    }
    return all(terms.values()), terms


def meaning(acc, g1, g2, g3, stop):
    """§12: what the outcome means."""
    if stop:
        return ("The stop rule fired: a from-scratch ternary student of 1.3-2.7M parameters cannot learn this task, "
                "even with the teacher's states. The line stops at this size; the recommendation for end of turn "
                "becomes Smart Turn itself (BSD-2, 8.7 MB, onnxruntime), credited.")
    if acc < 0.80:
        return ("G1 is below 0.80: a from-scratch ternary student of 1.3-2.7M parameters cannot learn this task, even "
                "with the teacher's states. The line stops at this size; the recommendation for end of turn becomes "
                "Smart Turn itself (BSD-2, 8.7 MB, onnxruntime), credited.")
    if not g1:
        return "G1 is between 0.80 and 0.85 (or too far from the teacher): it works partly and does not pass."
    if not g2:
        return "G1 passes, G2 fails: it cuts users off too often at a useful recall. It does not pass."
    if not g3:
        return "G1 and G2 pass, G3 fails: the model works but not at this cost. It does not pass; no smaller rung is retrained."
    return "v2 passes."


def recommendation(passed, acc, teacher, lat):
    """§9's sentence: which model a user should take."""
    st_ms = 32.0
    if lat and lat.get("smart_turn_cpu_native"):
        st_ms = lat["smart_turn_cpu_native"].get("median_ms", st_ms)
    size = lat["gguf_bytes"] / 1e6 if lat else float("nan")
    ms = lat["wasm"]["median_ms"] if lat else float("nan")
    gap = 100 * (teacher - acc)
    if passed:
        return (f"Take v2 when size and speed matter: it gives up {gap:.1f} points of accuracy against Smart Turn "
                f"for a file of {size:.2f} MB instead of 8.7 MB and about {ms:.1f} ms per decision in WASM instead of "
                f"about {st_ms:.0f} ms natively; take Smart Turn's file when those points matter more.")
    return (f"Take Smart Turn's 8.7 MB file (about {st_ms:.0f} ms per decision on one core): v2 does not pass its gate, "
            f"and it gives up {gap:.1f} points of accuracy for a file of {size:.2f} MB and about {ms:.1f} ms in WASM.")


def table(rows, header):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return out


def eot_metrics(cuts, tau):
    """eot vs hold on the cuts both models decided: AUC, false cut-off and eot recall at tau."""
    if not cuts:
        return None
    p = [c[0] for c in cuts]
    y = [c[1] for c in cuts]
    s = Scores(p, y, taus=(tau,)).stats()
    k = f"@{tau:g}"
    return {"n": len(p), "auc": s["auc"], "false_cutoff": s[f"false_cutoff{k}"], "eot_recall": s[f"recall{k}"]}


def cmd_gate(a):
    smoke = a.smoke
    harness = json.load(open(a.harness)) if a.harness and os.path.exists(a.harness) else None
    lat = json.load(open(a.latency)) if a.latency and os.path.exists(a.latency) else None
    if not smoke:
        if harness is None:
            raise SystemExit("no harness check on file (§8.1): no v2 test score is read before it (§13)")
        if not harness["passed"]:
            raise SystemExit("the harness check failed (§8.1): an amendment fixes it before any v2 test score is read")
        if lat is None:
            raise SystemExit("no latencies on file (§8.3): no v2 test score is read before them (§13)")
    teacher = harness["teacher_accuracy"] if harness else ST_BAR
    calib = json.load(open(a.calib))
    tau50, tau85 = calib["tau50"], calib["tau85"]
    parity = None
    if a.parity and os.path.exists(a.parity):
        lines = [l for l in open(a.parity).read().splitlines() if l.strip().startswith("{")]
        parity = json.loads(lines[-1]) if lines else None
    # labels: v1's clip manifest (test), or the training store's validation clips (smoke)
    labels = {}
    if smoke:
        for cid, m in store_meta(a.labels_store).items():
            if m.get("split") == "val" and isinstance(m.get("label"), bool):
                labels[cid] = (m["label"], m.get("lang") or "", m.get("dataset") or "")
    else:
        for c in read_jsonl(a.labels):
            if c.get("split") == "test":
                lab = c.get("label") or {}
                labels[c["id"]] = (bool(lab.get("endpoint")), c.get("lang", ""), lab.get("dataset") or "")
    probs = {r["id"]: r for r in read_jsonl(a.test_probs)}
    ids = [i for i in labels if i in probs and "prob" in probs[i]]
    missing = len(labels) - len(ids)
    p = [probs[i]["prob"] for i in ids]
    y = [labels[i][0] for i in ids]
    sc = Scores(p, y, taus=(tau50, tau85))
    overall = bootstrap(sc)
    k50, k85 = f"@{tau50:g}", f"@{tau85:g}"
    acc = overall[f"accuracy{k50}"]["value"]
    fc = overall[f"false_cutoff{k85}"]["value"]
    rec85 = overall[f"recall{k85}"]["value"]
    g1, g2 = check_g1(acc, teacher), check_g2(fc, rec85)
    g3, g3_terms = check_g3(lat, parity)
    pilot = json.load(open(a.pilot)) if a.pilot and os.path.exists(a.pilot) else None
    stop = bool(pilot and pilot.get("stop"))
    passed = g1 and g2 and g3 and not stop

    def groups(idx):
        by = defaultdict(list)
        for i in ids:
            by[labels[i][idx]].append(i)
        rows = {}
        for g, members in sorted(by.items()):
            s = Scores([probs[i]["prob"] for i in members], [labels[i][0] for i in members], taus=(tau50,))
            rows[g] = bootstrap(s, reps=BOOT, seed=1)
        return rows

    by_lang, by_ds = groups(1), groups(2)
    eng = [i for i in ids if labels[i][1] in ("eng", "en")]
    human = [i for i in eng if labels[i][2] in ("human_5", "human_convcollector_1")]
    subsets = {}
    for nm, members in (("english", eng), ("english_human", human)):
        if members:
            subsets[nm] = bootstrap(Scores([probs[i]["prob"] for i in members], [labels[i][0] for i in members],
                                           taus=(tau50, tau85)), seed=2)
    speko = {nm: oracle_point([probs[i]["prob"] for i in m], [labels[i][0] for i in m])
             for nm, m in (("all", ids), ("english_human", human)) if m}
    # teacher-student agreement: validation (the teacher's logits of the store) and test (the harness dump)
    agree = {}
    if a.val_probs and os.path.exists(a.val_probs) and a.data:
        meta = store_meta(Path(a.data) / "train")
        vs = [(r["prob"], teacher_logit(a.data, meta[r["id"]])) for r in read_jsonl(a.val_probs)
              if "prob" in r and r["id"] in meta and meta[r["id"]].get("split") == "val"
              and (Path(a.data) / "train" / f"teacher-{meta[r['id']]['_tag']}.done").exists()]
        if vs:
            agree["val"] = {"n": len(vs), "agreement": sum((pv >= tau50) == (zt > 0) for pv, zt in vs) / len(vs)}
    if not smoke and a.smart_turn_dump and os.path.exists(a.smart_turn_dump):
        st = {r["id"]: r["prob"] for r in read_jsonl(a.smart_turn_dump)}
        both = [i for i in ids if i in st]
        if both:
            agree["test"] = {"n": len(both), "agreement": sum((probs[i]["prob"] >= tau50) == (st[i] > 0.5) for i in both) / len(both)}
    # eot-bench (§8.4)
    eot = None
    if not smoke and a.eotbench_probs and os.path.exists(a.eotbench_probs) and a.data:
        store = Path(a.data) / "eotbench" / "store"
        meta = store_meta(store)
        v2 = {r["id"]: r.get("prob") for r in read_jsonl(a.eotbench_probs)}
        stp = {}
        for f in sorted(store.glob("smartturn-*.jsonl")):
            for r in read_jsonl(f):
                stp[r["id"]] = r.get("prob")
        rows = [(cid, m) for cid, m in meta.items()
                if v2.get(cid) is not None and stp.get(cid) is not None and isinstance(m.get("label"), bool)
                and not (isinstance(stp[cid], float) and math.isnan(stp[cid]))]
        def by_key(key):
            out = defaultdict(lambda: ([], []))
            for cid, m in rows:
                out[m.get(key) or ""][0].append((v2[cid], m["label"]))
                out[m.get(key) or ""][1].append((stp[cid], m["label"]))
            return {g: {"v2": eot_metrics(v, tau85), "smart_turn": eot_metrics(s, 0.5)} for g, (v, s) in sorted(out.items())}
        v2_all = eot_metrics([(v2[c], m["label"]) for c, m in rows], tau85)
        st_all = eot_metrics([(stp[c], m["label"]) for c, m in rows], 0.5)
        n_hold = sum(1 for _, m in rows if not m["label"])
        eot = {"cuts": len(meta), "decided_by_both": len(rows), "v2": v2_all, "smart_turn": st_all,
               "timer": {"auc": 0.5, "false_cutoff": 1.0 if n_hold else float("nan"), "eot_recall": 1.0},
               "by_language": by_key("lang"), "by_cohort": by_key("cohort"),
               "carries_over": bool(v2_all and st_all and v2_all["auc"] >= st_all["auc"] - EOT_GAP)}
    probe = json.load(open(a.probe)) if a.probe and os.path.exists(a.probe) else None
    leak = None
    if a.data and (Path(a.data) / "train" / "LEAK.json").exists():
        leak = json.load(open(Path(a.data) / "train" / "LEAK.json"))
    training = training_report(Path(a.run) / "log.jsonl") if (Path(a.run) / "log.jsonl").exists() else None
    timer = sc.stats()["positive_rate"]
    result = {
        "verdict": "pass" if passed else "fail", "smoke": smoke,
        "G1": {"pass": g1, "accuracy": acc, "teacher": teacher, "bar": max(G1_BAR, teacher - G1_GAP)},
        "G2": {"pass": g2, "false_cutoff": fc, "end_recall": rec85, "tau85": tau85},
        "G3": {"pass": g3, "terms": g3_terms, "latency": lat, "parity": parity},
        "stop_rule": stop, "meaning": meaning(acc, g1, g2, g3, stop),
        "sentence": recommendation(passed, acc, teacher, lat),
        "test": {"clips": len(ids), "missing": missing, "overall": overall, "by_language": by_lang,
                 "by_dataset": by_ds, "subsets": subsets, "speko_oracle": speko, "timer": timer},
        "agreement": agree, "eotbench": eot, "calibration": calib, "pilot": pilot, "probe": probe,
        "leakage": leak, "training": training, "disclosure": DISCLOSURE,
    }
    write_json(Path(a.out).with_suffix(".json") if a.out.endswith(".md") else a.out + ".json", result)
    md = render(result, tau50, tau85)
    Path(a.out).write_text("\n".join(md) + "\n")
    print(f"s6-ear2: {result['verdict']} (G1 {g1}, G2 {g2}, G3 {g3}){' [SMOKE]' if smoke else ''} -> {a.out}")
    return 0


def training_report(log):
    steps = modes = 0
    elapsed = None
    vals = []
    for v in read_jsonl(log):
        if "loss" in v:
            steps = max(steps, v["step"])
            elapsed = v.get("elapsed_min", elapsed)
        if "val" in v:
            vals.append({"step": v["step"], "mode": v["mode"], "acc": v["val"]["acc"], "bce": v["val"]["bce"]})
        if v.get("final"):
            elapsed = v.get("elapsed_min", elapsed)
            modes = v.get("epochs")
    return {"steps": steps, "working_minutes": elapsed, "epochs": modes, "validation": vals}


def render(r, tau50, tau85):
    t = r["test"]
    o = t["overall"]
    k50, k85 = f"@{tau50:g}", f"@{tau85:g}"
    md = ["# s6-ear2-nano: gate (docs/s6-ear2-prereg.md §9)", ""]
    if r["smoke"]:
        md += ["**SMOKE RUN.** A few minutes under a throwaway prefix, on validation clips in place of the test "
               "clips, to test that the commands start and the files chain. None of these numbers is a result.", ""]
    md += [f"**Verdict: {r['verdict'].upper()}.** {r['meaning']}", "", r["sentence"], ""]
    md += table([
        ["G1 accuracy at tau50", f"{r['G1']['accuracy']:.4f}", f">= 0.85 and >= T - 0.08 (T = {r['G1']['teacher']:.4f})", r["G1"]["pass"]],
        ["G2 false cut-off at tau85", f"{r['G2']['false_cutoff']:.4f}", "<= 0.20", r["G2"]["pass"]],
        ["G2 END recall at tau85", f"{r['G2']['end_recall']:.4f}", ">= 0.80", r["G2"]["pass"]],
        ["G3 " + ", ".join(k for k, v in r["G3"]["terms"].items()), str(r["G3"]["terms"]), "WASM <= 5.0 ms, front end <= 0.10 ms/hop, <= 3 MB, parity", r["G3"]["pass"]],
    ], ["term", "value", "bar", "pass"])
    md += ["", f"## Test ({t['clips']} clips, {t['missing']} without a decision; 95% bootstrap intervals, {BOOT} resamples by clip)", ""]
    md += table([[k, fmt(o, k)] for k in (f"accuracy{k50}", f"precision{k50}", f"recall{k50}", f"f1{k50}",
                                           f"fpr_all{k50}", f"fnr_all{k50}", f"false_cutoff{k85}", f"recall{k85}", "auc", "ece")],
                ["metric", "value"])
    md += ["", "Baselines on the same clips:", ""]
    rows = []
    for label, v in BASELINES:
        if v == "timer":
            v = t["timer"]
        elif v == "teacher":
            v = r["G1"]["teacher"]
        rows.append([label, f"{v:.4f}"])
    rows.append(["s6-ear2-nano", fmt(o, f"accuracy{k50}")])
    md += table(rows, ["model", "accuracy"])
    for name, key in (("language", "by_language"), ("dataset", "by_dataset")):
        md += ["", f"Per {name}:", ""]
        md += table([[g, m["n"]["value"], fmt(m, f"accuracy{k50}")] for g, m in t[key].items()], [name, "clips", "accuracy"])
    for nm, m in t["subsets"].items():
        md += ["", f"Subset {nm}: accuracy {fmt(m, f'accuracy{k50}')}, false cut-off at tau85 {fmt(m, f'false_cutoff{k85}')}, END recall {fmt(m, f'recall{k85}')}"]
    for nm, s in t["speko_oracle"].items():
        if s:
            md += [f"Speko's oracle point ({nm}, threshold chosen on the evaluated set, reported only): false cut-off "
                   f"{s['false_cutoff']:.4f} at END recall {s['end_recall']:.4f}, accuracy {s['accuracy']:.4f}"]
    md += ["", "## Teacher-student agreement (§10)", "", json.dumps(r["agreement"])]
    md += ["", "## eot-bench (§8.4, reported, not gated)", ""]
    e = r["eotbench"]
    if e:
        md += table([[nm, (m or {}).get("n", "-"), f"{(m or {}).get('auc', float('nan')):.4f}",
                      f"{(m or {}).get('false_cutoff', float('nan')):.4f}", f"{(m or {}).get('eot_recall', float('nan')):.4f}"]
                     for nm, m in (("v2 (tau85)", e["v2"]), ("Smart Turn v3.2 CPU (0.5)", e["smart_turn"]), ("timer", e["timer"]))],
                    ["model", "cuts", "AUC", "false cut-off", "eot recall"])
        if not e["carries_over"]:
            md += ["", "**v2 does not carry over to real conversational speech:** its AUC is more than 0.10 below Smart Turn's on these cuts, whatever the gate says."]
        md += ["", EOT_NOTE]
    else:
        md += ["not run"]
    md += ["", "## Reported (§10)", "", "Pilot: " + json.dumps(r["pilot"]), "", "Probe: " + json.dumps(r["probe"]),
           "", "Leakage filter (§3.2): " + json.dumps((r["leakage"] or {}).get("total")),
           "Sentence-level overlap cannot be checked: spoken_text is empty, as for Smart Turn's own numbers.",
           "", "Training: " + json.dumps({k: v for k, v in (r["training"] or {}).items() if k != "validation"}),
           "", "Calibration: " + json.dumps({k: r["calibration"][k] for k in ("temperature", "tau50", "tau85")}),
           "", DEPLOYMENT, "", DISCLOSURE]
    return md


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("probe")
    p.add_argument("--native", required=True)
    p.add_argument("--wasm", required=True)
    p.add_argument("--out")
    p = sub.add_parser("data")
    p.add_argument("--data", required=True)
    p.add_argument("--shards", type=int, default=83)
    p = sub.add_parser("pilot")
    p.add_argument("--dir", required=True)
    p.add_argument("--base", type=float, required=True)
    p.add_argument("--rates-only", action="store_true")
    p.add_argument("--out")
    p = sub.add_parser("checkpoint")
    p.add_argument("--log", required=True)
    p = sub.add_parser("calibrate")
    p.add_argument("--probs", required=True)
    p.add_argument("--store", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("harness")
    p.add_argument("--baseline", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("latency")
    p.add_argument("--native", required=True)
    p.add_argument("--wasm", required=True)
    p.add_argument("--smart-turn")
    p.add_argument("--model", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("gate")
    for k in ("run", "test-probs", "labels", "harness", "smart-turn-dump", "latency", "parity", "model", "calib",
              "val-probs", "data", "eotbench-probs", "pilot", "probe", "out", "labels-store"):
        p.add_argument(f"--{k}", required=k in ("run", "test-probs", "calib", "out"))
    p.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    return {"probe": cmd_probe, "data": cmd_data, "pilot": cmd_pilot, "checkpoint": cmd_checkpoint,
            "calibrate": cmd_calibrate, "harness": cmd_harness, "latency": cmd_latency, "gate": cmd_gate}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
