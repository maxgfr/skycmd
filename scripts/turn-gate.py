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

s6-ear v3 (docs/s6-ear3-prereg.md):

    python3 -I scripts/turn-gate.py bleed --store DIR/conv/store --out bleed.json
    python3 -I scripts/turn-gate.py pilot3 --dir runs/s6-ear3-pilot --v2-probs V.jsonl --conv-store DIR/conv/store --out choice.json
    python3 -I scripts/turn-gate.py checkpoint3 --log runs/s6-ear3-nano/log.jsonl
    python3 -I scripts/turn-gate.py harness3 --policy P.json --labels data/ear/clips/smart-turn.jsonl --v2-test T.jsonl \\
        --spans DIR/eotbench/store --v2-eot E.jsonl --out check.json
    python3 -I scripts/turn-gate.py gate3 --run runs/s6-ear3-nano --test-probs P.jsonl --calib calib.json --out GATE.md ...

* bleed (§3.7): on the training records, at each d, the AUC of the mean energy of A's hops over
  [o, o + d]; exit 1 when a d gives more than 0.60.
* pilot3 (§6.4-6.5): the four arms' final f32 validation; eligible when Smart Turn validation accuracy
  is >= 0.88; the eligible from-v2 arm with the best conversation AUC at d = 0.2 s, P4 when eligible
  and better by more than 0.005; the stop rule against v2's file on the same cuts (+ 0.01; exit 4).
* checkpoint3 (§7): `last-ema` if its final ternary checkpoint loss is below last's, else `last`.
* harness3 (§8.1): the policy port, v2's file on Smart Turn's test (0.8993), the span builder at v2's cut
  (17,934 cuts, 0.7299 and 0.7838 within 0.0005), then the count of §8.3's cuts; exit 1 on a failure.
* gate3 (§9): A1, A2, E1-E3 (the paired bootstrap by turn, 2,000 resamples, seed 20261117), G3 (with v2's
  tensor list); §8.2-8.4 and §10 reported; GATE.md and GATE.json. Refuses before §8.1 and §8.5 are on file,
  and (amendment 3) on a partial input: fewer than the 31,527 test clips, a span of §8.3 or an ask without
  its reading, a conversation test record without its readings or teacher opinion, a parity file that is not a
  complete measurement, a missing file: "technical failure: partial input, no verdict written", exit 1, no
  GATE file (a smoke run, on validation stand-ins, is not checked).
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


# ============================================================================= s6-ear v3
# docs/s6-ear3-prereg.md: the bleed probe (§3.7), the pilot and its stop rule (§6.4-6.5), the
# checkpoint (§7), the harness checks (§8.1) and the gate (§9), with the paired bootstrap by turn
# and the breakdowns of §8.3.

ASKS = list(range(200, 2401, 200))
# §3.7
BLEED_BAR = 0.60
# §6.4-6.5
V2_RATE = 0.0007745
PILOT3_ARMS = {"P1": ("v2", 0.25), "P2": ("v2", 0.5), "P3": ("v2", 1.0), "P4": ("scratch", 1.0)}
PILOT3_ELIGIBLE = 0.88
PILOT3_SCRATCH_MARGIN = 0.005
PILOT3_STOP_MARGIN = 0.01
MAIN_MINUTES = {"v2": 120, "scratch": 240}
# §8.1
V2_TEST_ACC = 0.8993
V2_EOT_CUTS = 17934
V2_EOT_AUC, ST_EOT_AUC, EOT_TOL = 0.7299, 0.7838, 0.0005
# §8.3, §9
FIRST_ASK = 200
LONG_HOLD_S = 5.0
FC_GATE = 0.10
FC_REPORTED = (0.05, 0.20)
BOOT3, BOOT3_SEED = 2000, 20261117
A1_BAR = V2_TEST_ACC - 0.01
E1_MIN = 0.02
E3_FLOOR = -0.02
TAU_C_FC = 0.10

DISCLOSURE3 = (
    "**Disclosure (§11).** v3 was designed after v2's eot-bench reading: the overall, per-language and "
    "per-cohort numbers of runs/s6-ear2-nano/GATE.json were seen, and that reading motivated making eot-bench "
    "the headline, the no-harm bar outside English and the expected per-language effects. eot-bench has never "
    "been used for training or for any choice of v2's or v3's training: no eot-bench file was opened for v3's "
    "design, and the cut point (0.2 s) comes from the demo's policy and LiveKit's card, not from a reading. The "
    "variable trailing silence was motivated by the microphone demo (turn.ts), through the code. v3's study "
    "measured the trailing silence of 3,160 of v2's training clips. Smart Turn's test is still partly in-sample "
    "for the design, as v2's disclosure says; v3 adds no reading of it before the gate. The label cross-check and the "
    "caps were amended after a measurement on five training sessions, before any validation or test session was read "
    "(amendment 2: Silero's label read at each energy record's instant; every end-of-turn record kept). **What it means:** the "
    "training corpus is human-to-human and eot-bench is human-to-agent (the labels of the training "
    "conversations are the listener's behaviour, not an agent's need); the training conversations are English "
    "and eot-bench has 14 languages; the conversation test is session-disjoint, not speaker-disjoint."
)


def manifests(store):
    """Every manifest line of a store, with its part tag, in tag order."""
    out = []
    for p in sorted(Path(store).glob("manifest-*.jsonl")):
        tag = p.name[len("manifest-"):-len(".jsonl")]
        for v in read_jsonl(p):
            v["_tag"] = tag
            out.append(v)
    return out


def energy_means(store, recs, ds):
    """The mean raw log10-energy channel (42) of each record's hops lying wholly in [o, o + d], for
    every d of `ds` valid for it: {(id, d): mean}. A record's hop `cut_hops - 1` ends at o, so the
    hop of index `cut_hops + 3` is the first to start at o."""
    out = {}
    by_tag = defaultdict(list)
    for r in recs:
        by_tag[r["_tag"]].append(r)
    for tag, rs in by_tag.items():
        with open(Path(store) / f"feat-{tag}.bin", "rb") as f:
            for r in rs:
                cut, dmax = int(r["cut_hops"]), int(r["dmax_ms"])
                f.seek(int(r["offset"]))
                b = f.read(int(r["hops"]) * 86)
                for d in ds:
                    if d > dmax:
                        continue
                    lo, hi = cut + 3, cut + d // 10
                    vals = [struct.unpack_from("<e", b, h * 86 + 84)[0] for h in range(lo, hi)]
                    if vals:
                        out[(r["id"], d)] = sum(vals) / len(vals)
    return out


def cmd_bleed(a):
    """§3.7's bleed probe: on the training records, at each d, the AUC of the mean energy of A's
    hops over [o, o + d] (end of turn against hold); no d may give more than 0.60."""
    recs = [r for r in manifests(a.store) if r.get("split") == "train" and isinstance(r.get("label"), bool)]
    means = energy_means(a.store, recs, ASKS)
    by_d = {}
    for d in ASKS:
        rows = [(means[(r["id"], d)], r["label"]) for r in recs if (r["id"], d) in means]
        if rows:
            auc = Ranked([x for x, _ in rows], [y for _, y in rows], [0] * len(rows)).auc([1.0])
            by_d[str(d)] = {"n": len(rows), "eot": sum(1 for _, y in rows if y), "auc": auc}
    aucs = [v["auc"] for v in by_d.values() if not math.isnan(v["auc"])]
    worst = max(aucs) if aucs else float("nan")
    passed = bool(aucs) and worst <= BLEED_BAR
    out = {"about": "docs/s6-ear3-prereg.md §3.7, the bleed probe: on the training records, the AUC (end of turn "
                    "against hold) of the mean raw log10-energy channel of A's hops lying wholly in [o, o + d]",
           "by_d_ms": by_d, "max_auc": worst, "bar": BLEED_BAR, "pass": passed, "records": len(recs),
           "measured": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    write_json(a.out, out)
    print(json.dumps({"max_auc": worst, "pass": passed}))
    return 0 if passed else 1


def final_entry(log, mode):
    """The last log entry of `mode` with a validation."""
    best = None
    for v in read_jsonl(log):
        if v.get("mode") == mode and "val" in v:
            best = v
    return best


def labels_of(store, split):
    return {r["id"]: r["label"] for r in manifests(store) if r.get("split") == split and isinstance(r.get("label"), bool)}


def auc_of(probs, labels, d=None):
    """AUC of a probability dump (sky-turn lines; at `d` ms when given) against labels by id."""
    rows = [(r["prob"], labels[r["id"]]) for r in probs
            if "prob" in r and r["id"] in labels and (d is None or r.get("d_ms") == d)]
    if not rows:
        return float("nan"), 0
    return Scores([p for p, _ in rows], [y for _, y in rows]).stats()["auc"], len(rows)


def cmd_pilot3(a):
    """§6.4-6.5: eligible arms (Smart Turn validation accuracy at tau 0.5, d = 0, f32 >= 0.88); the
    eligible from-v2 arm of highest conversation validation AUC at d = 0.2 s, or P4 when it is eligible and
    higher by more than 0.005 (or the only eligible arm); the stop rule against v2's file on the same cuts."""
    d = Path(a.dir)
    arms = {}
    for name, (start, mult) in PILOT3_ARMS.items():
        e = final_entry(d / name / "log.jsonl", "f32") if (d / name / "log.jsonl").exists() else None
        if e is None or "conv" not in e:
            raise SystemExit(f"{d / name}/log.jsonl: no final f32 validation with the conversation records")
        arms[name] = {"start": start, "lr": float(f"{a.base * mult:.4g}"), "rate_multiplier": mult,
                      "st_val_acc_f32": e["val"]["acc"], "conv_val_auc_d200": e["conv"]["auc"],
                      "eligible": e["val"]["acc"] >= PILOT3_ELIGIBLE}
    labels = labels_of(a.conv_store, "val")
    v2_metric, n = auc_of(read_jsonl(a.v2_probs), labels, FIRST_ASK)
    elig_v2 = [k for k in ("P1", "P2", "P3") if arms[k]["eligible"]]
    chosen = None
    if elig_v2:
        chosen = max(elig_v2, key=lambda k: (arms[k]["conv_val_auc_d200"], -arms[k]["rate_multiplier"]))
    p4 = arms["P4"]
    if p4["eligible"] and (chosen is None or p4["conv_val_auc_d200"] > arms[chosen]["conv_val_auc_d200"] + PILOT3_SCRATCH_MARGIN):
        chosen = "P4"
    stop = chosen is None or not (arms[chosen]["conv_val_auc_d200"] >= v2_metric + PILOT3_STOP_MARGIN)
    out = {"arm": chosen, "start": arms[chosen]["start"] if chosen else None, "lr": arms[chosen]["lr"] if chosen else None,
           "minutes": MAIN_MINUTES[arms[chosen]["start"]] if chosen else None, "stop": stop,
           "v2_metric": v2_metric, "v2_metric_records": n, "arms": arms,
           "rules": {"metric": "AUC on the conversation validation records cut at d = 0.2 s (§6.4)",
                     "eligible": f"Smart Turn validation accuracy at tau 0.5, d = 0, f32 >= {PILOT3_ELIGIBLE}",
                     "choice": f"the eligible from-v2 arm with the highest metric; P4 instead if eligible and higher by more than {PILOT3_SCRATCH_MARGIN} (amendment 1: or the only eligible arm)",
                     "stop": f"no eligible arm, or the chosen arm below v2's file on the same cuts + {PILOT3_STOP_MARGIN} (§6.5)"}}
    write_json(a.out, out)
    print(f"pilot3: {chosen} ({out['start']}, lr {out['lr']}), v2's metric {v2_metric:.4f}, stop {stop}")
    return 4 if stop else 0


def cmd_checkpoint3(a):
    """§7: the EMA if its ternary checkpoint loss (the mean of the two validation BCEs) is below last's."""
    e, l = final_entry(a.log, "ema-ternary"), final_entry(a.log, "ternary")
    if l is None or "ckpt_loss" not in l:
        raise SystemExit(f"{a.log}: no final ternary validation with the checkpoint loss")
    print("last-ema" if e is not None and e.get("ckpt_loss", math.inf) < l["ckpt_loss"] else "last")
    return 0


def test_labels(path):
    """id -> (label, lang, dataset) of the Smart Turn test clips of v1's manifest."""
    out = {}
    for c in read_jsonl(path):
        if c.get("split") == "test":
            lab = c.get("label") or {}
            out[c["id"]] = (bool(lab.get("endpoint")), c.get("lang", ""), lab.get("dataset") or "")
    return out


def span_rows(store):
    """The eot-bench span records with the Smart Turn readings at d = 0 and d = 0.2 s."""
    recs = manifests(store)
    st0, st200 = {}, {}
    for f in sorted(Path(store).glob("smartturn-*.jsonl")):
        st0.update({r["id"]: r.get("prob") for r in read_jsonl(f)})
    for f in sorted(Path(store).glob("smartturn200-*.jsonl")):
        st200.update({r["id"]: r.get("prob") for r in read_jsonl(f)})
    return recs, st0, st200


def gated_span(r):
    """§8.3: an `eot` or `hold` span at least 0.2 s long (an ask at 0.2 s fits), holds over 5 s ignored."""
    return isinstance(r.get("label"), bool) and not r.get("short") and not r.get("long_hold") \
        and int(r.get("dmax_ms", 0)) >= FIRST_ASK


def finite(p):
    return p is not None and not (isinstance(p, float) and math.isnan(p))


def cmd_harness3(a):
    """§8.1: the policy port; v2's file through v3's evaluation path on Smart Turn's test (0.8993 to four
    decimals); the span builder at v2's cut (17,934 cuts, v2's AUC 0.7299 and Smart Turn's 0.7838 within
    0.0005); then the number of cuts of §8.3, before any model reads them."""
    checks = {}
    pol = json.load(open(a.policy))
    checks["policy_port"] = {"ok": pol.get("passed") is True, "vectors": len(pol.get("vectors", []))}
    lab = test_labels(a.labels)
    probs = {r["id"]: r for r in read_jsonl(a.v2_test)}
    ids = [i for i in lab if i in probs and "prob" in probs[i]]
    acc = sum((probs[i]["prob"] >= 0.5) == lab[i][0] for i in ids) / max(len(ids), 1)
    checks["smart_turn_test_v2"] = {"accuracy": acc, "clips": len(ids), "missing": len(lab) - len(ids),
                                    "want": V2_TEST_ACC, "ok": round(acc, 4) == V2_TEST_ACC and len(ids) == TEST_CLIPS}
    recs, st0, _ = span_rows(a.spans)
    cuts = [r for r in recs if isinstance(r.get("label"), bool)]
    v2 = {r["id"]: r.get("prob") for r in read_jsonl(a.v2_eot) if r.get("d_ms", 0) == 0}
    both = [r for r in cuts if finite(v2.get(r["id"])) and finite(st0.get(r["id"]))]
    av2 = Scores([v2[r["id"]] for r in both], [r["label"] for r in both]).stats()["auc"] if both else float("nan")
    ast = Scores([st0[r["id"]] for r in both], [r["label"] for r in both]).stats()["auc"] if both else float("nan")
    checks["eotbench_cuts"] = {"value": len(cuts), "want": V2_EOT_CUTS, "ok": len(cuts) == V2_EOT_CUTS}
    checks["eotbench_v2_auc"] = {"value": av2, "want": V2_EOT_AUC, "tol": EOT_TOL, "decided": len(both),
                                 "ok": abs(av2 - V2_EOT_AUC) <= EOT_TOL}
    checks["eotbench_smart_turn_auc"] = {"value": ast, "want": ST_EOT_AUC, "tol": EOT_TOL,
                                         "ok": abs(ast - ST_EOT_AUC) <= EOT_TOL}
    passed = all(c["ok"] for c in checks.values())
    gated = [r for r in cuts if gated_span(r)]
    out = {"passed": passed, "checks": checks,
           "cuts_8_3": {"cuts": len(gated), "eot": sum(1 for r in gated if r["label"]),
                        "hold": sum(1 for r in gated if not r["label"]),
                        "turns": len({r.get("turn") for r in gated}),
                        "rule": "spans of 0.2 s or more cut at start + 0.2 s, holds over 5 s ignored (§8.3); counted "
                                "before any model reads them"},
           "rule": "docs/s6-ear3-prereg.md §8.1", "checked": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    write_json(a.out, out)
    print(json.dumps({"passed": passed, **{k: c.get("value", c.get("accuracy", c["ok"])) for k, c in checks.items()}}))
    return 0 if passed else 1


# --- the paired bootstrap by turn (§8.3) -------------------------------------------------------


class Ranked:
    """Cuts ranked by one model's probability, for weighted AUC and recall at a false cut-off: `groups`
    holds, in ascending probability, the (unit, label) pairs of each set of tied cuts."""

    def __init__(self, p, y, unit):
        order = sorted(range(len(p)), key=lambda i: p[i])
        self.groups = []
        for i in order:
            if self.groups and p[i] == self.groups[-1][0]:
                self.groups[-1][1].append((unit[i], y[i]))
            else:
                self.groups.append((p[i], [(unit[i], y[i])]))

    def totals(self, w):
        pos = neg = 0.0
        for _, g in self.groups:
            for u, y in g:
                if y:
                    pos += w[u]
                else:
                    neg += w[u]
        return pos, neg

    def auc(self, w):
        pos, neg = self.totals(w)
        if not pos or not neg:
            return float("nan")
        s = below = 0.0
        for _, g in self.groups:
            gp = gn = 0.0
            for u, y in g:
                if y:
                    gp += w[u]
                else:
                    gn += w[u]
            s += gp * (below + 0.5 * gn)
            below += gn
        return s / (pos * neg)

    def recall_at_fc(self, w, fc):
        """The ROC point: the largest recall among thresholds (a cut is "finished" when p >= tau) whose
        false cut-off is at most `fc`."""
        pos, neg = self.totals(w)
        if not pos or not neg:
            return float("nan")
        tp = fp = best = 0.0
        for _, g in reversed(self.groups):
            for u, y in g:
                if y:
                    tp += w[u]
                else:
                    fp += w[u]
            if fp / neg <= fc + 1e-12:
                best = tp / pos
            else:
                break
        return best


def paired_bootstrap(units, stats, reps=BOOT3, seed=BOOT3_SEED):
    """Point values and percentile 95% intervals of each statistic `stats[name](w)`, `w` the count of
    each unit (a turn) in a resample by unit."""
    point = {k: f([1.0] * units) for k, f in stats.items()}
    rng = random.Random(seed)
    draws = defaultdict(list)
    for _ in range(reps):
        w = [0.0] * units
        for i in rng.choices(range(units), k=units):
            w[i] += 1.0
        for k, f in stats.items():
            v = f(w)
            if not math.isnan(v):
                draws[k].append(v)
    out = {}
    for k, v in point.items():
        d = sorted(draws[k])
        out[k] = {"value": v, "lo": d[int(0.025 * (len(d) - 1))] if d else float("nan"),
                  "hi": d[int(math.ceil(0.975 * (len(d) - 1)))] if d else float("nan"), "resamples": len(d)}
    return out


def gguf_tensors(path):
    """The tensor list of a GGUF file: [(name, shape, type)] (GGUF v2/v3)."""
    b = Path(path).read_bytes()
    if b[:4] != b"GGUF":
        raise ValueError(f"{path}: not a GGUF file")
    at = 8
    n_t, n_kv = struct.unpack_from("<QQ", b, at)
    at += 16

    def string(at):
        n = struct.unpack_from("<Q", b, at)[0]
        return b[at + 8:at + 8 + n].decode(), at + 8 + n

    sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}

    def value(t, at):
        if t == 8:
            return string(at)[1]
        if t == 9:
            et, n = struct.unpack_from("<IQ", b, at)
            at += 12
            for _ in range(n):
                at = value(et, at)
            return at
        return at + sizes[t]

    for _ in range(n_kv):
        _, at = string(at)
        t = struct.unpack_from("<I", b, at)[0]
        at = value(t, at + 4)
    out = []
    for _ in range(n_t):
        name, at = string(at)
        nd = struct.unpack_from("<I", b, at)[0]
        dims = struct.unpack_from(f"<{nd}Q", b, at + 4)
        at += 4 + 8 * nd
        t = struct.unpack_from("<I", b, at)[0]
        at += 12
        out.append((name, tuple(dims), t))
    return out


def tau_lowest_fc(p, y, fc=TAU_C_FC):
    """tau_c (§7): the lowest observed threshold whose false cut-off (holds with p >= tau) is <= fc."""
    holds = [pi for pi, yi in zip(p, y) if not yi]
    if not holds:
        return None
    for t in sorted(set(p)):
        if sum(1 for h in holds if h >= t) / len(holds) <= fc:
            return t
    return None


def at_tau(p, y, tau):
    n_pos = sum(1 for v in y if v) or 1
    n_neg = sum(1 for v in y if not v) or 1
    return {"false_cutoff": sum(1 for pi, yi in zip(p, y) if pi >= tau and not yi) / n_neg,
            "eot_recall": sum(1 for pi, yi in zip(p, y) if pi >= tau and yi) / n_pos, "tau": tau}


def by_id_d(path):
    """{(id, d_ms): prob} of a sky-turn dump."""
    return {(r["id"], r.get("d_ms", 0)): r["prob"] for r in read_jsonl(path) if "prob" in r}


def eot_block(rows, p3, p2, pst, tau3, tau2):
    """The metrics of §8.3 on one set of cuts (rows: span records), for v3, v2 and Smart Turn at 0.2 s."""
    out = {"cuts": len(rows), "turns": len({r["turn"] for r in rows})}
    for name, pr, taus in (("v3", p3, tau3), ("v2", p2, tau2), ("smart_turn", pst, {"tau50": 0.5})):
        p = [pr[r["id"]] for r in rows]
        y = [r["label"] for r in rows]
        if not p:
            continue
        rk = Ranked(p, y, [0] * len(p))
        m = {"auc": rk.auc([1.0]), f"recall@fc{FC_GATE}": rk.recall_at_fc([1.0], FC_GATE)}
        for fc in FC_REPORTED:
            m[f"recall@fc{fc}"] = rk.recall_at_fc([1.0], fc)
        for k, t in taus.items():
            m[k] = at_tau(p, y, t)
        out[name] = m
    return out


def policy_reading(spans, pr, tau):
    """§8.3, reported: the policy simulated per span at threshold tau, every ask 0.2-2.4 s inside it."""
    eot_ans, eot_time, holds_cut, n_e, n_h = 0, [], 0, 0, 0
    for r in spans:
        asks = [d for d in ASKS if d <= int(r.get("dmax_ms", 0)) and (r["id"], d) in pr]
        if not asks:
            continue
        first = next((d for d in asks if pr[(r["id"], d)] >= tau), None)
        if r["label"]:
            n_e += 1
            if first is not None:
                eot_ans += 1
                eot_time.append(first)
        else:
            n_h += 1
            holds_cut += first is not None
    eot_time.sort()
    return {"tau": tau, "eot_spans": n_e, "eot_answered_before_timer": eot_ans / max(n_e, 1),
            "eot_answer_ms_median": eot_time[len(eot_time) // 2] if eot_time else None,
            "hold_spans": n_h, "hold_with_a_false_cutoff": holds_cut / max(n_h, 1)}


def meaning3(r):
    """§12: what the outcome means (written before the run)."""
    if r.get("stop_rule"):
        return ("The stop rule fired (§6.5): twelve minutes of conversation records did not move v2's ranking on "
                "conversation validation by 0.01. The line stops at this size; v2 stays the published file, with its "
                "fallback timer; for real conversations the recommendation stays Smart Turn's file.")
    a1, a2, e1, e2, e3, g3 = (r[k]["pass"] for k in ("A1", "A2", "E1", "E2", "E3", "G3"))
    if not (a1 and a2):
        s = "A1 or A2 fails: v3 bought its gain on conversations with a loss on Smart Turn's clips. It does not pass; v2 stays the published file."
        if e1 and e2 and e3:
            s += " E1-E3 hold: v3 is better on real conversations and worse on clips, a trade, not a pass."
        return s
    if not e1:
        if r["E1"]["lower"] > 0:
            return ("E1 fails on its minimum effect (lower bound above 0, point estimate under +0.02): v3 is measurably "
                    "but not usefully better at ranking real ends of turn. It does not pass; the gain is given as measured.")
        return ("E1 fails on its lower bound: no improvement on real human-to-agent speech is shown. Test (c) separates "
                "the causes: if (c) improves over v2, the human-to-human signal does not transfer to agents; if not, the "
                "model did not learn the signal.")
    if not e2:
        return "E1 holds and E2 fails: the ranking improved, but not where a voice agent works (few cut-offs). It does not pass."
    if not e3:
        return ("E3 fails: the English gain was paid for by the other languages. It does not pass; a follow-up would need "
                "conversations in other languages, not a reweighting found on eot-bench.")
    if not g3:
        return "G3 fails: v3 is not a drop-in replacement (with the same graph, a defect of the export, investigated). It does not pass."
    return "v3 passes. It replaces v2 in the demo only by the owner's decision."


PARTIAL = "technical failure: partial input, no verdict written"


def parity_problem(path, what):
    """None when `path` holds a complete measurement of §7 (pass or fail), else why not."""
    try:
        rec = read_parity(path)
    except ValueError as e:
        return f"{what}: {path} does not parse ({e})"
    if rec is None:
        return f"{what}: no parity line in {path}"
    n, diff, same = rec.get("n"), rec.get("max_abs_diff"), rec.get("same_decision")
    if not isinstance(n, int) or isinstance(n, bool) or n < PARITY_CLIPS:
        return f"{what}: {n} clips in {path}, §7 measures {PARITY_CLIPS}"
    if not isinstance(diff, (int, float)) or isinstance(diff, bool) or not math.isfinite(diff) \
            or not isinstance(same, int) or isinstance(same, bool):
        return f"{what}: {path} lacks max_abs_diff or same_decision"
    return None


def given(path):
    return bool(path) and os.path.exists(path)


def gate3_problems(a, h3, lab, tp):
    """Amendment 3: the gate reads complete inputs or writes no verdict. Every reading the run makes must be
    whole: the 31,527 Smart Turn test clips (§8.2), every span of §8.3 (the count the harness check
    recorded before any model read them) at every ask, every conversation test record at every ask (§8.4)
    with its teacher opinion, both parity measurements (§7) and both files (§9 G3). Returns what is missing."""
    out = []
    # test (a), §8.2
    if len(lab) != TEST_CLIPS:
        out.append(f"test (a): {len(lab)} Smart Turn test clips in {a.labels}, §8.2 reads {TEST_CLIPS}")
    miss = sum(1 for i in lab if not (i in tp and finite(tp[i].get("prob"))))
    if miss:
        out.append(f"test (a): {miss} of the {len(lab)} test clips have no v3 probability in {a.test_probs}")
    # test (b), §8.3
    if not (a.spans and given(a.eot_v3) and given(a.eot_v2)):
        out.append(f"test (b): no eot-bench readings (--spans {a.spans}, --eot-v3 {a.eot_v3}, --eot-v2 {a.eot_v2})")
    else:
        recs, _, st200 = span_rows(a.spans)
        p3, p2 = by_id_d(a.eot_v3), by_id_d(a.eot_v2)
        gated = [r for r in recs if gated_span(r)]
        want = ((h3 or {}).get("cuts_8_3") or {}).get("cuts")
        if want is None:
            out.append("test (b): the harness check records no count of §8.3's cuts")
        elif len(gated) != want:
            out.append(f"test (b): {len(gated)} spans of §8.3 in {a.spans}, the harness check counted {want}")
        labelled = [r for r in recs if isinstance(r.get("label"), bool)]
        for name, pr, zero in (("v3", p3, True), ("v2", p2, False)):
            m = sum(1 for r in labelled if not r.get("long_hold")
                    for d in ASKS if d <= int(r.get("dmax_ms", 0)) and not finite(pr.get((r["id"], d))))
            m += sum(1 for r in labelled if zero and not finite(pr.get((r["id"], 0))))
            if m:
                out.append(f"test (b): {m} span readings of {name} missing (every ask inside each span"
                           f"{', and d = 0' if zero else ''})")
        m = sum(1 for r in gated if not finite(st200.get(r["id"])))
        if m:
            out.append(f"test (b): {m} of the {len(gated)} spans of §8.3 have no Smart Turn reading at + 0.2 s")
    # test (c), §8.4 (reported): when the run gives it, every test record at every ask
    if a.conv_v3 or a.conv_v2:
        if not (a.conv_store and given(a.conv_v3) and given(a.conv_v2)):
            out.append(f"test (c): readings missing (--conv-store {a.conv_store}, --conv-v3 {a.conv_v3}, "
                       f"--conv-v2 {a.conv_v2})")
        else:
            tl = [r for r in manifests(a.conv_store) if r.get("split") == "test" and isinstance(r.get("label"), bool)]
            if not tl:
                out.append(f"test (c): no conversation test record in {a.conv_store}")
            for name, path in (("v3", a.conv_v3), ("v2", a.conv_v2)):
                pr = by_id_d(path)
                m = sum(1 for r in tl for d in ASKS if d <= int(r.get("dmax_ms", 0)) and not finite(pr.get((r["id"], d))))
                if m:
                    out.append(f"test (c): {m} readings of {name} missing in {path} (every ask inside each test record)")
            if a.sessions:
                tst = {}
                for f in sorted(Path(a.sessions).glob("teacher-conv-*.jsonl")):
                    tst.update({r["id"]: r.get("prob") for r in read_jsonl(f)})
                m = sum(1 for r in tl if not finite(tst.get(r["id"])))
                if m:
                    out.append(f"test (c): {m} of the {len(tl)} test records have no teacher opinion in {a.sessions}")
    # G3, §7 and §9
    for path, what in ((a.parity, "parity (Smart Turn clips)"), (a.parity_conv, "parity (conversation records)")):
        p = parity_problem(path, what) if given(path) else f"{what}: no file {path}"
        if p:
            out.append(p)
    for path, what in ((a.model, "v3's file"), (a.v2_model, "v2's file")):
        if not given(path):
            out.append(f"G3: no {what} ({path})")
    return out


def cmd_gate3(a):
    smoke = a.smoke
    h3 = json.load(open(a.harness3)) if a.harness3 and os.path.exists(a.harness3) else None
    lat = json.load(open(a.latency)) if a.latency and os.path.exists(a.latency) else None
    if not smoke:
        if not h3 or not h3.get("passed"):
            raise SystemExit("no passed harness check on file (§8.1): no v3 test score is read before it (§13)")
        if lat is None:
            raise SystemExit("no latencies on file (§8.5): no v3 test score is read before them (§13)")
    calib = json.load(open(a.calib))
    tau50, tau85 = calib["tau50"], calib["tau85"]
    v2_tau = {"tau50": 0.5, "tau85": a.v2_tau85}
    # test (a): Smart Turn's test clips, as v2 (§8.2)
    lab = test_labels(a.labels) if not smoke else {
        m["id"]: (m["label"], m.get("lang") or "", m.get("dataset") or "")
        for m in manifests(a.labels_store) if m.get("split") == "val" and isinstance(m.get("label"), bool)}
    tp = {r["id"]: r for r in read_jsonl(a.test_probs)}
    if not smoke:
        # amendment 3: a partial input is a technical failure, never a verdict (exit 1, no GATE file)
        problems = gate3_problems(a, h3, lab, tp)
        if problems:
            raise SystemExit(f"{PARTIAL}:\n  " + "\n  ".join(problems))
    ids = [i for i in lab if i in tp and "prob" in tp[i]]
    sc = Scores([tp[i]["prob"] for i in ids], [lab[i][0] for i in ids], taus=(tau50, tau85))
    overall = bootstrap(sc)
    k50, k85 = f"@{tau50:g}", f"@{tau85:g}"
    acc, fc85, rec85 = (overall[f"accuracy{k50}"]["value"], overall[f"false_cutoff{k85}"]["value"],
                        overall[f"recall{k85}"]["value"])
    by = {}
    for nm, idx in (("by_language", 1), ("by_dataset", 2)):
        g = defaultdict(list)
        for i in ids:
            g[lab[i][idx]].append(i)
        by[nm] = {k: bootstrap(Scores([tp[i]["prob"] for i in m], [lab[i][0] for i in m], taus=(tau50,)), seed=1)
                  for k, m in sorted(g.items())}
    ext = {}
    for name, path, tau in (("v3", a.test_ext, tau50), ("v2", a.v2_test_ext, 0.5)):
        if path and os.path.exists(path):
            pr = by_id_d(path)
            for d in (200, 600):
                rows = [(pr[(i, d)], lab[i][0]) for i in ids if (i, d) in pr]
                if rows:
                    s = Scores([p for p, _ in rows], [y for _, y in rows], taus=(tau,)).stats()
                    ext[f"{name}@{d}"] = {"n": len(rows), "accuracy": s[f"accuracy@{tau:g}"], "auc": s["auc"]}
    # test (b): eot-bench at the policy's first question (§8.3)
    e3_ = e2_ = e1_ = None
    eot = None
    if a.spans and a.eot_v3 and os.path.exists(a.eot_v3):
        recs, _, st200 = span_rows(a.spans)
        p3, p2 = by_id_d(a.eot_v3), by_id_d(a.eot_v2)
        gated = [r for r in recs if gated_span(r)]
        rows = [r for r in gated if (r["id"], FIRST_ASK) in p3 and (r["id"], FIRST_ASK) in p2]
        q3 = {r["id"]: p3[(r["id"], FIRST_ASK)] for r in rows}
        q2 = {r["id"]: p2[(r["id"], FIRST_ASK)] for r in rows}
        qs = {r["id"]: st200[r["id"]] for r in rows if finite(st200.get(r["id"]))}
        turns = sorted({r["turn"] for r in rows})
        ti = {t: k for k, t in enumerate(turns)}
        unit = [ti[r["turn"]] for r in rows]
        y = [r["label"] for r in rows]
        non_en = [k for k, r in enumerate(rows) if r.get("lang") != "en"]
        rk3, rk2 = Ranked([q3[r["id"]] for r in rows], y, unit), Ranked([q2[r["id"]] for r in rows], y, unit)
        rk3n = Ranked([q3[rows[k]["id"]] for k in non_en], [y[k] for k in non_en], [unit[k] for k in non_en])
        rk2n = Ranked([q2[rows[k]["id"]] for k in non_en], [y[k] for k in non_en], [unit[k] for k in non_en])
        st_rows = [k for k, r in enumerate(rows) if r["id"] in qs]
        rkst = Ranked([qs[rows[k]["id"]] for k in st_rows], [y[k] for k in st_rows], [unit[k] for k in st_rows])
        rk3s = Ranked([q3[rows[k]["id"]] for k in st_rows], [y[k] for k in st_rows], [unit[k] for k in st_rows])
        stats = {
            "auc_delta": lambda w: rk3.auc(w) - rk2.auc(w),
            "recall@fc0.1_delta": lambda w: rk3.recall_at_fc(w, FC_GATE) - rk2.recall_at_fc(w, FC_GATE),
            "auc_delta_non_english": lambda w: rk3n.auc(w) - rk2n.auc(w),
            "auc_delta_vs_smart_turn": lambda w: rk3s.auc(w) - rkst.auc(w),
        }
        boot = paired_bootstrap(len(turns), stats, reps=a.boot)
        e1_ = boot["auc_delta"]
        e2_ = boot["recall@fc0.1_delta"]
        e3_ = boot["auc_delta_non_english"]
        tc = None
        if a.conv_val and os.path.exists(a.conv_val) and a.conv_store:
            vl = labels_of(a.conv_store, "val")
            cv = [r for r in read_jsonl(a.conv_val) if "prob" in r and r["id"] in vl and r.get("d_ms", FIRST_ASK) == FIRST_ASK]
            tc = tau_lowest_fc([r["prob"] for r in cv], [vl[r["id"]] for r in cv])
        taus3 = {"tau50": tau50, "tau85": tau85}
        groups = {"all": rows, "english": [r for r in rows if r.get("lang") == "en"],
                  "non_english": [r for r in rows if r.get("lang") != "en"]}
        st_q = {i: qs[i] for i in qs}
        st_rows_all = {k: [r for r in v if r["id"] in st_q] for k, v in groups.items()}
        eot = {"cuts": len(rows), "gated_spans": len(gated), "missing": len(gated) - len(rows), "turns": len(turns),
               "bootstrap": {"resamples": a.boot, "seed": BOOT3_SEED, "by": "turn", **boot},
               "pooled": {k: eot_block(v, q3, q2, st_q, taus3, v2_tau) for k, v in groups.items()},
               "smart_turn_cuts": {k: len(v) for k, v in st_rows_all.items()},
               "by_language": {g: eot_block([r for r in rows if r.get("lang") == g], q3, q2, st_q, taus3, v2_tau)
                               for g in sorted({r.get("lang") for r in rows})},
               "by_cohort": {g: eot_block([r for r in rows if r.get("cohort") == g], q3, q2, st_q, taus3, v2_tau)
                             for g in sorted({r.get("cohort") for r in rows})},
               "policy_by_d": {}, "policy_simulated": {}, "tau_c": tc}
        for d in ASKS:
            rd = [r for r in gated if int(r["dmax_ms"]) >= d and (r["id"], d) in p3 and (r["id"], d) in p2]
            if rd:
                yy = [r["label"] for r in rd]
                eot["policy_by_d"][str(d)] = {
                    "cuts": len(rd),
                    "v3_auc": Ranked([p3[(r["id"], d)] for r in rd], yy, [0] * len(rd)).auc([1.0]),
                    "v2_auc": Ranked([p2[(r["id"], d)] for r in rd], yy, [0] * len(rd)).auc([1.0])}
        allspans = [r for r in recs if isinstance(r.get("label"), bool) and not r.get("long_hold")]
        for name, pr, taus in (("v3", p3, {"tau85": tau85, "tau_c": tc}), ("v2", p2, {"tau85": a.v2_tau85})):
            for k, t in taus.items():
                if t is not None:
                    eot["policy_simulated"][f"{name}@{k}"] = policy_reading(allspans, pr, t)
        cont = [r for r in recs if isinstance(r.get("label"), bool) and (r["id"], 0) in p3]
        if cont:
            eot["continuity_v3_at_span_start"] = {
                "cuts": len(cont), "auc": Ranked([p3[(r["id"], 0)] for r in cont], [r["label"] for r in cont],
                                                 [0] * len(cont)).auc([1.0])}
    # test (c): the held-out conversation sessions (§8.4), reported
    conv_c = None
    if a.conv_v3 and os.path.exists(a.conv_v3) and a.conv_store:
        tl = {r["id"]: r for r in manifests(a.conv_store) if r.get("split") == "test" and isinstance(r.get("label"), bool)}
        train_spk = set()
        for f in sorted(Path(a.sessions).glob("sessions-*.jsonl")) if a.sessions else []:
            for s in read_jsonl(f):
                if s.get("split") == "train":
                    train_spk.update(x for x in (s.get("speakers") or []) if x)
        c3, c2 = by_id_d(a.conv_v3), by_id_d(a.conv_v2) if a.conv_v2 and os.path.exists(a.conv_v2) else {}
        tst = {}
        for f in sorted(Path(a.sessions).glob("teacher-conv-*.jsonl")) if a.sessions else []:
            tst.update({r["id"]: r.get("prob") for r in read_jsonl(f)})
        conv_c = {"records": len(tl), "speaker_unseen": sum(1 for r in tl.values() if r.get("speaker") not in train_spk),
                  "by_d": {}}
        for d in ASKS:
            for subset, keep in (("all", lambda r: True), ("speaker_unseen", lambda r: r.get("speaker") not in train_spk)):
                rd = [r for r in tl.values() if keep(r) and int(r["dmax_ms"]) >= d and (r["id"], d) in c3]
                if not rd:
                    continue
                block = {"records": len(rd)}
                for name, pr in (("v3", c3), ("v2", c2)):
                    ok = [r for r in rd if (r["id"], d) in pr]
                    if ok:
                        rk = Ranked([pr[(r["id"], d)] for r in ok], [r["label"] for r in ok], [0] * len(ok))
                        block[name] = {"auc": rk.auc([1.0]), f"recall@fc{FC_GATE}": rk.recall_at_fc([1.0], FC_GATE)}
                if d == FIRST_ASK:
                    ok = [r for r in rd if finite(tst.get(r["id"]))]
                    if ok:
                        rk = Ranked([tst[r["id"]] for r in ok], [r["label"] for r in ok], [0] * len(ok))
                        block["smart_turn_recorded"] = {"auc": rk.auc([1.0]), f"recall@fc{FC_GATE}": rk.recall_at_fc([1.0], FC_GATE),
                                                        "agreement_with_labels@0.5": sum((tst[r["id"]] > 0.5) == r["label"] for r in ok) / len(ok)}
                        both = [r for r in ok if (r["id"], d) in c3]
                        block["smart_turn_v3_agreement"] = sum((tst[r["id"]] > 0.5) == (c3[(r["id"], d)] >= tau50) for r in both) / max(len(both), 1)
                conv_c["by_d"].setdefault(str(d), {})[subset] = block
    # G3 (§9 item 6)
    _, g3_terms = check_g3(lat, read_parity(a.parity))
    pc = read_parity(a.parity_conv)
    g3_terms["parity_conversations"] = pc is not None and pc.get("n", 0) >= PARITY_CLIPS \
        and pc.get("max_abs_diff", 1.0) <= PARITY_TOL and pc.get("same_decision") == pc.get("n")
    same = None
    if a.model and a.v2_model and os.path.exists(a.model) and os.path.exists(a.v2_model):
        same = gguf_tensors(a.model) == gguf_tensors(a.v2_model)
    g3_terms["same_tensor_list_as_v2"] = bool(same)
    g3 = all(g3_terms.values())
    pilot = json.load(open(a.pilot)) if a.pilot and os.path.exists(a.pilot) else None
    stop = bool(pilot and pilot.get("stop"))
    nan = float("nan")
    r = {
        "verdict": None, "smoke": smoke, "stop_rule": stop,
        "A1": {"pass": acc >= A1_BAR, "accuracy": acc, "bar": A1_BAR, "v2": V2_TEST_ACC},
        "A2": {"pass": fc85 <= G2_FC and rec85 >= G2_RECALL, "false_cutoff": fc85, "end_recall": rec85, "tau85": tau85},
        "E1": {"pass": bool(e1_ and e1_["value"] >= E1_MIN and e1_["lo"] > 0), "delta": (e1_ or {}).get("value", nan),
               "lower": (e1_ or {}).get("lo", nan), "upper": (e1_ or {}).get("hi", nan), "min_effect": E1_MIN},
        "E2": {"pass": bool(e2_ and e2_["lo"] > 0), "delta": (e2_ or {}).get("value", nan), "lower": (e2_ or {}).get("lo", nan),
               "upper": (e2_ or {}).get("hi", nan), "false_cutoff": FC_GATE},
        "E3": {"pass": bool(e3_ and e3_["lo"] > E3_FLOOR), "delta": (e3_ or {}).get("value", nan),
               "lower": (e3_ or {}).get("lo", nan), "upper": (e3_ or {}).get("hi", nan), "floor": E3_FLOOR},
        "G3": {"pass": g3, "terms": g3_terms, "latency": lat},
        "test_a": {"clips": len(ids), "missing": len(lab) - len(ids), "overall": overall, **by, "extended": ext},
        "test_b_eotbench": eot, "test_c_conversations": conv_c, "calibration": calib, "pilot": pilot,
        "harness3": h3, "crosscheck": json.load(open(a.crosscheck)) if a.crosscheck and os.path.exists(a.crosscheck) else None,
        "bleed": json.load(open(a.bleed)) if a.bleed and os.path.exists(a.bleed) else None,
        "data": json.load(open(a.conv_summary)) if a.conv_summary and os.path.exists(a.conv_summary) else None,
        "training": training_report(Path(a.run) / "log.jsonl") if (Path(a.run) / "log.jsonl").exists() else None,
        "disclosure": DISCLOSURE3,
    }
    passed = not stop and all(r[k]["pass"] for k in ("A1", "A2", "E1", "E2", "E3", "G3"))
    r["verdict"] = "pass" if passed else "fail"
    r["meaning"] = meaning3(r)
    r["sentence"] = sentence3(r, eot)
    out = Path(a.out)
    write_json(out.with_suffix(".json"), r)
    out.write_text("\n".join(render3(r)) + "\n")
    print(f"s6-ear3: {r['verdict']} (A1 {r['A1']['pass']}, A2 {r['A2']['pass']}, E1 {r['E1']['pass']}, E2 {r['E2']['pass']}, "
          f"E3 {r['E3']['pass']}, G3 {g3}){' [SMOKE]' if smoke else ''} -> {out}")
    return 0


def read_parity(path):
    if not path or not os.path.exists(path):
        return None
    lines = [l for l in open(path).read().splitlines() if l.strip().startswith("{")]
    return json.loads(lines[-1]) if lines else None


def sentence3(r, eot):
    """§9's sentence: which file a user should take, on what evidence."""
    pooled = ((eot or {}).get("pooled") or {}).get("all") or {}
    a3 = (pooled.get("v3") or {}).get("auc", float("nan"))
    a2 = (pooled.get("v2") or {}).get("auc", float("nan"))
    ast = (pooled.get("smart_turn") or {}).get("auc", float("nan"))
    d = r["E1"]
    clips = f"on Smart Turn's test clips its accuracy is {r['A1']['accuracy']:.4f} against v2's {V2_TEST_ACC}"
    conv = (f"on eot-bench at the policy's first question (0.2 s into each pause) its AUC is {a3:.4f} against v2's "
            f"{a2:.4f} (change {d['delta']:+.4f}, 95% interval [{d['lower']:+.4f}, {d['upper']:+.4f}]) and Smart Turn's "
            f"{ast:.4f}")
    if r["verdict"] == "pass":
        who = "Take v3" if not (ast > a3) else "Take v3 when size and speed matter (Smart Turn's 8.7 MB file ranks real ends of turn better)"
        return f"{who}: {conv}; {clips}."
    return f"Keep v2 (or take Smart Turn's 8.7 MB file when real conversations matter most): v3 does not pass; {conv}; {clips}."


def render3(r):
    md = ["# s6-ear3-nano: gate (docs/s6-ear3-prereg.md §9)", ""]
    if r["smoke"]:
        md += ["**SMOKE RUN.** Under a throwaway prefix, on validation data in place of the tests. None of these "
               "numbers is a result.", ""]
    md += [f"**Verdict: {r['verdict'].upper()}.** {r['meaning']}", "", r["sentence"], ""]

    def f4(v):
        return "-" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.4f}"

    md += table([
        ["A1 accuracy at tau50 (Smart Turn's test)", f4(r["A1"]["accuracy"]), f">= {A1_BAR:.4f}", r["A1"]["pass"]],
        ["A2 false cut-off at tau85", f4(r["A2"]["false_cutoff"]), "<= 0.20", r["A2"]["pass"]],
        ["A2 END recall at tau85", f4(r["A2"]["end_recall"]), ">= 0.80", r["A2"]["pass"]],
        ["E1 AUC(v3) - AUC(v2), eot-bench pooled", f"{f4(r['E1']['delta'])} [{f4(r['E1']['lower'])}, {f4(r['E1']['upper'])}]",
         ">= +0.02 and lower bound > 0", r["E1"]["pass"]],
        ["E2 recall at a false cut-off of 0.10, v3 - v2", f"{f4(r['E2']['delta'])} [{f4(r['E2']['lower'])}, {f4(r['E2']['upper'])}]",
         "lower bound > 0", r["E2"]["pass"]],
        ["E3 AUC(v3) - AUC(v2), the 13 other languages", f"{f4(r['E3']['delta'])} [{f4(r['E3']['lower'])}, {f4(r['E3']['upper'])}]",
         "lower bound > -0.02", r["E3"]["pass"]],
        ["G3 " + ", ".join(k for k in r["G3"]["terms"]), json.dumps(r["G3"]["terms"]),
         "WASM <= 5.0 ms, front end <= 0.10 ms/hop, <= 3 MB, parity, v2's tensor list", r["G3"]["pass"]],
    ], ["term", "value", "bar", "pass"])
    e = r["test_b_eotbench"]
    if e:
        md += ["", f"## eot-bench at start + 0.2 s ({e['cuts']} cuts, {e['turns']} turns; paired bootstrap by turn, "
                   f"{e['bootstrap']['resamples']} resamples, seed {BOOT3_SEED})", ""]
        rows = []
        for g, b in e["pooled"].items():
            for m in ("v3", "v2", "smart_turn"):
                if m in b:
                    rows.append([g, m, f4(b[m]["auc"]), f4(b[m][f"recall@fc{FC_GATE}"]),
                                 f4(b[m]["recall@fc0.05"]), f4(b[m]["recall@fc0.2"])])
        md += table(rows, ["cuts", "model", "AUC", "recall@FC0.10", "recall@FC0.05", "recall@FC0.20"])
        md += ["", "Recall at a false cut-off is a ranking measure (the ROC point), not a deployable threshold.", ""]
        md += table([[g, b["cuts"], f4((b.get("v3") or {}).get("auc")), f4((b.get("v2") or {}).get("auc")),
                      f4((b.get("smart_turn") or {}).get("auc"))] for g, b in e["by_language"].items()],
                    ["language", "cuts", "v3", "v2", "Smart Turn"])
        md += ["", "Policy reading per ask (AUC): " + json.dumps(e["policy_by_d"]),
               "", "Policy simulated per span: " + json.dumps(e["policy_simulated"]),
               "", f"tau_c (conversation validation, false cut-off <= 0.10 at d = 0.2 s, reported, not stored): {e['tau_c']}"]
    c = r["test_c_conversations"]
    if c:
        md += ["", f"## Test (c): {c['records']} conversation test records ({c['speaker_unseen']} speaker-unseen), reported", "",
               json.dumps(c["by_d"].get(str(FIRST_ASK)))]
    t = r["test_a"]
    md += ["", f"## Test (a): Smart Turn's test ({t['clips']} clips)", "",
           json.dumps({k: v for k, v in t["overall"].items() if k in ("auc", "ece")}),
           "", "Extended by 0.2 and 0.6 s of their own background (reported): " + json.dumps(t["extended"])]
    md += ["", "## Reported (§10)", "", "Pilot: " + json.dumps(r["pilot"]),
           "", "Label cross-check (the instant label of amendment 2): " + json.dumps({k: ((r["crosscheck"] or {}).get("instant") or {}).get(k) for k in ("agreement", "drop_share", "agreement_by_class")} | {"pass": (r["crosscheck"] or {}).get("pass")}),
           "", "Bleed probe: " + json.dumps({k: (r["bleed"] or {}).get(k) for k in ("max_auc", "pass")}),
           "", "Data: " + json.dumps((r["data"] or {}).get("counts")),
           "", "Training: " + json.dumps({k: v for k, v in (r["training"] or {}).items() if k != "validation"}),
           "", "Calibration: " + json.dumps({k: r["calibration"].get(k) for k in ("temperature", "tau50", "tau85")}),
           "", DISCLOSURE3]
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
    # s6-ear v3
    p = sub.add_parser("bleed")
    p.add_argument("--store", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("pilot3")
    p.add_argument("--dir", required=True)
    p.add_argument("--v2-probs", required=True)
    p.add_argument("--conv-store", required=True)
    p.add_argument("--base", type=float, default=V2_RATE)
    p.add_argument("--out", required=True)
    p = sub.add_parser("checkpoint3")
    p.add_argument("--log", required=True)
    p = sub.add_parser("harness3")
    for k in ("policy", "labels", "v2-test", "spans", "v2-eot", "out"):
        p.add_argument(f"--{k}", required=True)
    p = sub.add_parser("gate3")
    for k in ("run", "test-probs", "labels", "labels-store", "test-ext", "v2-test-ext", "spans", "eot-v3", "eot-v2",
              "conv-store", "conv-v3", "conv-v2", "conv-val", "sessions", "calib", "latency", "parity", "parity-conv",
              "model", "v2-model", "harness3", "pilot", "crosscheck", "bleed", "conv-summary", "out"):
        p.add_argument(f"--{k}", required=k in ("run", "test-probs", "calib", "out"))
    p.add_argument("--v2-tau85", type=float, default=0.7244)
    p.add_argument("--boot", type=int, default=BOOT3)
    p.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    return {"probe": cmd_probe, "data": cmd_data, "pilot": cmd_pilot, "checkpoint": cmd_checkpoint,
            "calibrate": cmd_calibrate, "harness": cmd_harness, "latency": cmd_latency, "gate": cmd_gate,
            "bleed": cmd_bleed, "pilot3": cmd_pilot3, "checkpoint3": cmd_checkpoint3, "harness3": cmd_harness3,
            "gate3": cmd_gate3}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
