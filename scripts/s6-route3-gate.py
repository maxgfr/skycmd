#!/usr/bin/env python3
"""The gate of s6-nano v3 (docs/s6-nano3-prereg.md, sections 2, 5 and 6), from the dumps.

    s6-route3-gate.py guard --split fresh|val
    s6-route3-gate.py gate  --split fresh|val [--root DIR] [--fit runs/s6-nano3/fit.json]
                            [--wasm wasm3.json] [--native native.json] [--out-dir DIR]

The system `s6-route`: for each option i of a slate, p(i) ∝ (p_model(i) · p_potion-cos-4bit(i))^β,
β and τ_ask3 read from runs/s6-nano3/fit.json (and checked against the values the
pre-registration froze). The model's dumps are sky-eval's at full depth (`model/<file>.jsonl`)
and at the first exit (`model/<file>.exit0.jsonl`, the fast mode); the baselines' are
scripts/route-baselines.py's with the values fitted for v2 (`<system>/<file>.jsonl`), all under
--root (<data>/eval/fresh; <data>/eval/fresh/rehearsal for the validation rehearsal).

- `--split fresh`: the fresh files f_tools, f_tools_masked, f_metatool, f_clinc, f_massive,
  read through scripts/route/common.py, which refuses them until runs/s6-nano3/FRESH-OPEN exists
  and every file has its recorded sha256. Exit 0 when the gate passes, 1 when it does not.
- `--split val`: the rehearsal on v_tools and v_tools_masked (no fresh file is read). The
  readings by cluster use a stand-in (the gold tool's name; validation slates carry no cluster
  file). The fused system must reproduce runs/s6-nano3/fit.json; exit 0 when it does, 1
  otherwise. Its gate is printed as a rehearsal, never as a verdict.

Metrics are v2's scorer's (scripts/route/scorer.py: ECE in 15 bins as sky-metrics, NLL floored
at 1e-12, risk-coverage); bootstrap intervals are 95% percentile intervals over 1,000 resamples
with seed 0, as v2's gate.
"""

import argparse
import importlib.util
import json
import math
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "route"))
import common  # noqa: E402
import scorer  # noqa: E402

_spec = importlib.util.spec_from_file_location("s6_route3_fit", os.path.join(HERE, "s6-route3-fit.py"))
fit3 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fit3)

# section 2, frozen with the pre-registration
BETA = 0.81
TAU_ASK3 = 0.671047
PARTNER = fit3.PARTNER  # potion-cos-4bit
PARTNER_THETA = 0.396120
PARTNER_TEMPERATURE = 0.0671021
BM25_BETA = 0.5  # "BM25 fused with the same embedding", the exploratory reading's geometric mean
BYTES_MODEL = 2418912
BYTES_EMBEDDING = 1003952
# section 5
G1_MINILM = 0.03
G2_MARGIN = 0.02
G3_ECE = 0.05
G4_PARITY = 1e-3
G4_SLATES = 200
G4_AGREE = 199
G4_US = 60000.0
G4_BYTES = 4_000_000
GOAL_ACCURACY = 0.87
GOAL_ANSWERED = 0.95
GOAL_COVERAGE = 0.80
GOAL_FAST = 0.015
FIT_TOLERANCE = 1e-9

SYSTEM, FAST, BM25F, MODEL, MODEL0 = "s6-route", "s6-route-fast", "bm25+potion-cos-4bit", "model", "model-exit0"
BASELINES = ("potion-cos", "potion-cos-4bit", "static-ft", "pair-lr", "bm25", "minilm-cos", "toolrouter", "name-prior")
FILES = {
    "fresh": {"main": "f_tools", "masked": "f_tools_masked", "others": ("f_metatool", "f_clinc", "f_massive"),
              "clusters": "f_tools.clusters"},
    "val": {"main": "v_tools", "masked": "v_tools_masked", "others": (), "clusters": None},
}
DISCLOSURE = ("v3 was designed after v2's test was read, from an exploratory reading of v2's test dumps. Its parameters "
              "come from validation. Its gate is read on requests no system had scored; on T2-tools these requests are "
              "about the same held-out tools as v2's test. The latency bar is 30 times v2's and was set after v2 was "
              "measured. The system needs distinctive tool names.")


def finite(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


# ---------------------------------------------------------------- the fusion

def fuse_probs(pa, pb, beta):
    """p(i) ∝ (pa(i) · pb(i))^beta over the options (scripts/s6-route3-fit.py's fuse)."""
    return fit3.fuse([{"probs": pa, "gold": 0}], [{"probs": pb, "gold": 0}], beta)[0]["probs"]


def fused_items(slates, pa, pb, beta):
    """Items of the fused system on `slates`, from two {uid: probs} maps (every slate in both)."""
    out = []
    for s in slates:
        if s["uid"] not in pa or s["uid"] not in pb:
            return None
        out.append(scorer.Item(s, fuse_probs(pa[s["uid"]], pb[s["uid"]], beta)))
    return out


# ---------------------------------------------------------------- the pure rules

def g1(sys_acc, bm25_boot, minilm, model, partner):
    """G1 on T2-tools: above BM25 with the lower bound of the paired bootstrap above 0; at least
    minilm-cos - 0.03; above the model alone and above potion-cos-4bit alone."""
    d = (bm25_boot or {}).get("diff")
    lo = (bm25_boot or {}).get("lo")
    c = {
        "above_bm25": finite(d) and d > 0 and finite(lo) and lo > 0,
        "minilm_within_0.03": finite(sys_acc) and finite(minilm) and sys_acc >= minilm - G1_MINILM - 1e-12,
        "above_model": finite(sys_acc) and finite(model) and sys_acc > model,
        "above_partner": finite(sys_acc) and finite(partner) and sys_acc > partner,
    }
    return {"system": sys_acc, "bm25_difference": d, "bm25_lower_bound": lo, "minilm-cos": minilm, "model": model,
            PARTNER: partner, **c, "pass": all(c.values())}


def g2(per_file):
    """G2 on each other file: the system >= max(model alone, potion-cos-4bit alone) - 0.02. A
    file whose inputs are missing fails."""
    out, ok = {}, bool(per_file)
    for name, v in per_file.items():
        s, m, p = v.get("system"), v.get("model"), v.get("partner")
        better = max(m, p) if finite(m) and finite(p) else None
        holds = finite(s) and finite(better) and s >= better - G2_MARGIN - 1e-12
        out[name] = {"system": s, "model": m, PARTNER: p, "bar": better - G2_MARGIN if finite(better) else None,
                     "pass": bool(holds)}
        ok = ok and holds
    out["pass"] = bool(ok)
    return out


def g3(ece):
    return {"ece": ece, "pass": bool(finite(ece) and ece <= G3_ECE)}


def g4(wasm):
    """G4 from the WASM bench's JSON (`n`, `max_abs_diff`, `argmax_agree`, `median_cached_us`,
    `median_first_us`, `bytes_model`, `bytes_embedding`); absent: not measured, not passed."""
    if not wasm:
        return {"measured": False, "pass": False}
    n, gap, agree = wasm.get("n"), wasm.get("max_abs_diff"), wasm.get("argmax_agree")
    med, bm, be = wasm.get("median_cached_us"), wasm.get("bytes_model"), wasm.get("bytes_embedding")
    c = {
        "slates": isinstance(n, int) and n >= G4_SLATES,
        "parity": finite(gap) and gap <= G4_PARITY,
        "same_answer": isinstance(agree, int) and isinstance(n, int) and agree >= max(G4_AGREE, n - (G4_SLATES - G4_AGREE)),
        "latency": finite(med) and med <= G4_US,
        "size": isinstance(bm, int) and isinstance(be, int) and bm + be <= G4_BYTES,
    }
    return {"measured": True, "n": n, "max_abs_diff": gap, "argmax_agree": agree, "median_cached_us": med,
            "median_first_us": wasm.get("median_first_us"), "bytes_model": bm, "bytes_embedding": be,
            "bytes": bm + be if isinstance(bm, int) and isinstance(be, int) else None, **c, "pass": all(c.values())}


def goals(main_metrics, fast_acc):
    """The announced goals of section 5 (not gates)."""
    m = main_metrics or {}
    acc, at = m.get("accuracy"), m.get("at_tau_ask") or {}
    return {
        "accuracy_0.87": {"value": acc, "met": bool(finite(acc) and acc >= GOAL_ACCURACY)},
        "answered_0.95_coverage_0.80": {"accuracy": at.get("accuracy"), "coverage": at.get("coverage"),
                                        "met": bool(finite(at.get("accuracy")) and at["accuracy"] >= GOAL_ANSWERED
                                                    and finite(at.get("coverage")) and at["coverage"] >= GOAL_COVERAGE)},
        "fast_within_0.015": {"full": acc, "fast": fast_acc,
                              "met": bool(finite(acc) and finite(fast_acc) and acc - fast_acc <= GOAL_FAST + 1e-12)},
    }


def largest_clusters(cluster_of, k=3):
    """The k clusters with the most slates (ties by label)."""
    count = {}
    for c in cluster_of.values():
        count[c] = count.get(c, 0) + 1
    return [c for c, _ in sorted(count.items(), key=lambda kv: (-kv[1], kv[0]))[:k]]


def cluster_bootstrap(a, b, cluster_of, resamples=scorer.BOOTSTRAP, seed=0):
    """accuracy(a) - accuracy(b) on the same slates and its 95% percentile interval over
    `resamples` resamples of the clusters (with replacement; every slate of a drawn cluster)."""
    ia = {it.uid: it for it in a}
    ib = {it.uid: it for it in b}
    if set(ia) != set(ib) or set(ia) != set(cluster_of):
        raise scorer.ScoreError("the two systems and the clusters do not cover the same slates")
    by = {}
    for u in sorted(ia):
        by.setdefault(cluster_of[u], []).append(u)
    names = sorted(by)
    rng = random.Random(seed)
    point = scorer.accuracy(list(ia.values())) - scorer.accuracy(list(ib.values()))
    diffs = []
    for _ in range(resamples):
        uids = [u for _ in names for u in by[names[rng.randrange(len(names))]]]
        diffs.append(scorer.accuracy([ia[u] for u in uids]) - scorer.accuracy([ib[u] for u in uids]))
    diffs.sort()
    return {"diff": point, "lo": diffs[int(0.025 * resamples)], "hi": diffs[min(resamples - 1, int(0.975 * resamples))],
            "resamples": resamples, "clusters": len(names)}


def proxy_clusters(slates):
    """The validation stand-in for the cluster file: the gold tool's name, or the request for a
    none-gold slate."""
    out = {}
    for s in slates:
        g = scorer.argmax(s["target"])
        out[s["uid"]] = ("name:" + scorer.option_name(s["opts"][g]).lower()) if g < len(s["opts"]) - 1 else "request:" + s.get("request", s["uid"])
    return out


# ---------------------------------------------------------------- reading the inputs

def check_fit(fit):
    """fit.json must hold the values the pre-registration froze."""
    beta, tau = fit.get("beta"), (fit.get("tau_ask3") or {}).get("tau")
    if fit.get("partner") != PARTNER or not finite(beta) or abs(beta - BETA) > 1e-12 or not finite(tau) or abs(tau - TAU_ASK3) > 5e-7:
        raise common.Refused(f"runs/s6-nano3/fit.json does not hold the frozen values (beta {BETA}, tau_ask3 {TAU_ASK3}, {PARTNER})")
    return beta, tau


def check_partner_params():
    p = os.path.join(common.eval_root(), "fitted", PARTNER, "params.json")
    if not os.path.isfile(p):
        raise common.Refused(f"no {p}: v3 reads the values fitted for v2")
    v = common.load_json(p)
    if abs(v.get("theta", 0) - PARTNER_THETA) > 5e-7 or abs(v.get("temperature", 0) - PARTNER_TEMPERATURE) > 5e-8:
        raise common.Refused(f"{p} is not the fit of section 2 (theta {PARTNER_THETA}, T {PARTNER_TEMPERATURE})")
    return v


def slate_path(split, name):
    return common.fresh_file(name) if split == "fresh" else common.slate_file("val", name)


def guard(split):
    names = FILES[split]
    for n in (names["main"], names["masked"], *names["others"]):
        slate_path(split, n)
    if names["clusters"]:
        slate_path(split, names["clusters"])


def load_probs(path, slates):
    """{uid: probs} of a dump that answers every slate once, or None when there is no dump."""
    if not os.path.isfile(path):
        return None
    return {it.uid: it.probs for it in scorer.join(common.read_jsonl(path), slates)}


def systems_on(root, name, slates, beta):
    """Every system's items on one file: the model (full depth, first exit), the baselines, the
    fused system, its fast mode, BM25 fused with the partner. A missing input leaves its systems out."""
    probs = {MODEL: load_probs(os.path.join(root, "model", f"{name}.jsonl"), slates),
             MODEL0: load_probs(os.path.join(root, "model", f"{name}.exit0.jsonl"), slates)}
    for s in BASELINES:
        probs[s] = load_probs(os.path.join(root, s, f"{name}.jsonl"), slates)
    out = {s: [scorer.Item(x, p[x["uid"]]) for x in slates] for s, p in probs.items() if p is not None}
    for sys_name, a, b, bt in ((SYSTEM, MODEL, PARTNER, beta), (FAST, MODEL0, PARTNER, beta), (BM25F, "bm25", PARTNER, BM25_BETA)):
        if probs.get(a) is not None and probs.get(b) is not None:
            out[sys_name] = fused_items(slates, probs[a], probs[b], bt)
    return out


def acc(items):
    return scorer.accuracy(items) if items else None


def rehearsal_check(fit, main_items, slates, beta):
    """The validation rehearsal: the fused system on v_tools against runs/s6-nano3/fit.json."""
    s, fast = main_items.get(SYSTEM), main_items.get(FAST)
    if not s:
        return {"reproduces": False, "why": "no fused system (a missing dump)"}
    rows = [{"probs": it.probs, "gold": it.gold} for it in s]
    got = {"accuracy": scorer.accuracy(s), "ece_fit_convention": fit3.ece(rows), "ece_scorer": scorer.ece(s),
           "nll": scorer.mean(it.nll for it in s), "tau_ask3": fit3.tau_ask(rows),
           "fast_accuracy": acc(fast), "alone": {k: acc(main_items.get(k)) for k in (MODEL, MODEL0, PARTNER, "bm25", "minilm-cos")},
           "bm25_with_partner": acc(main_items.get(BM25F))}
    want = {"accuracy": fit["system"]["accuracy"], "ece_fit_convention": fit["system"]["ece"], "nll": fit["nll_at_beta"],
            "tau": fit["tau_ask3"]["tau"], "fast_accuracy": fit["fast_mode"]["accuracy"],
            "alone": {MODEL: fit["alone"]["model"], MODEL0: fit["alone"]["model_exit0"], PARTNER: fit["alone"][PARTNER],
                      "bm25": fit["alone"]["bm25"], "minilm-cos": fit["alone"]["minilm-cos"]},
            "bm25_with_partner": fit["bm25_with_partner"]}

    def close(a, b, tol=FIT_TOLERANCE):
        return finite(a) and finite(b) and abs(a - b) <= tol

    checks = {k: close(got[k], want[k]) for k in ("accuracy", "ece_fit_convention", "nll", "fast_accuracy", "bm25_with_partner")}
    checks["tau"] = close(got["tau_ask3"]["tau"], want["tau"])
    checks["alone"] = all(close(got["alone"][k], want["alone"][k]) for k in want["alone"])
    ref = os.path.join(common.REPO, "runs", "s6-nano2", "v_tools.dump.jsonl")
    model_gap = None
    if os.path.isfile(ref) and main_items.get(MODEL):
        r = {d["uid"]: d["probs"] for d in common.read_jsonl(ref)}
        model_gap = max(abs(a - b) for it in main_items[MODEL] for a, b in zip(it.probs, r.get(it.uid, [math.inf] * len(it.probs))))
    return {"reproduces": all(checks.values()), "checks": checks, "got": got, "want": want,
            "model_dump_max_abs_diff_vs_runs_s6-nano2": model_gap}


def gate(a):
    split = a.split
    names = FILES[split]
    fit = common.load_json(a.fit)
    beta, tau = check_fit(fit)
    partner_params = check_partner_params()
    root = a.root or (os.path.join(common.eval_root(), "fresh") if split == "fresh" else os.path.join(common.eval_root(), "fresh", "rehearsal"))
    slates = {}
    for n in (names["main"], names["masked"], *names["others"]):
        slates[n] = common.read_jsonl(slate_path(split, n))
    if names["clusters"]:
        cluster_of = {c["uid"]: c["cluster"] for c in common.read_jsonl(slate_path(split, names["clusters"]))}
        cluster_source = names["clusters"]
    else:
        cluster_of = proxy_clusters(slates[names["main"]])
        cluster_source = "stand-in: the gold tool's name (validation slates carry no cluster file)"

    per = {n: systems_on(root, n, slates[n], beta) for n in slates}
    results = {}
    for n, systems in per.items():
        for s, its in systems.items():
            results.setdefault(s, {})[n] = scorer.metrics(its, tau if s in (SYSTEM, FAST) else None)

    main = per[names["main"]]
    sys_m = results.get(SYSTEM, {}).get(names["main"])
    diffs = {}
    if SYSTEM in main:
        for other in ("bm25", "minilm-cos", MODEL, PARTNER, BM25F, FAST):
            if other in main:
                diffs[other] = scorer.bootstrap_diff(main[SYSTEM], main[other])
    v = {"G1": g1(acc(main.get(SYSTEM)), diffs.get("bm25"), acc(main.get("minilm-cos")), acc(main.get(MODEL)), acc(main.get(PARTNER))),
         "G2": g2({n: {"system": acc(per[n].get(SYSTEM)), "model": acc(per[n].get(MODEL)), "partner": acc(per[n].get(PARTNER))}
                   for n in names["others"]}),
         "G3": g3((sys_m or {}).get("ece")),
         "G4": g4(common.load_json(a.wasm) if a.wasm and os.path.isfile(a.wasm) else None)}
    if split == "val":
        v["G2"] = {"pass": False, "note": "no other file on validation"}
    v["pass"] = all(v[k]["pass"] for k in ("G1", "G2", "G3", "G4"))

    # section 6: without the three largest clusters, the cluster bootstrap, the names masked
    top = largest_clusters({u: cluster_of[u] for u in (it.uid for it in main.get(SYSTEM, []))} if SYSTEM in main else cluster_of)
    rest = {s: [it for it in its if cluster_of.get(it.uid) not in top] for s, its in main.items()}
    readings = {
        "clusters": {"source": cluster_source, "count": len(set(cluster_of.values())), "three_largest": top,
                     "slates_in_three_largest": sum(1 for c in cluster_of.values() if c in top)},
        "without_three_largest": {"slates": len(rest.get(SYSTEM, [])), "accuracy": {s: acc(its) for s, its in rest.items()},
                                  "differences": {o: scorer.bootstrap_diff(rest[SYSTEM], rest[o]) for o in ("bm25", "minilm-cos", MODEL, PARTNER)
                                                  if SYSTEM in rest and o in rest and rest[SYSTEM]}},
        "cluster_bootstrap": {o: cluster_bootstrap(main[SYSTEM], main[o], cluster_of) for o in ("bm25", "minilm-cos", MODEL, PARTNER)
                              if SYSTEM in main and o in main},
        "names_masked": {s: acc(its) for s, its in per[names["masked"]].items()},
    }
    latency = {"native_model": common.load_json(a.native) if a.native and os.path.isfile(a.native) else None,
               "baselines_cpu": {s: common.load_json(p) for s in BASELINES
                                 if os.path.isfile(p := os.path.join(root, s, f"{names['main']}.time.json"))},
               "wasm": v["G4"]}
    model_info = os.path.join(root, "model", "model.json")
    out = {"split": split, "root": root, "rehearsal": split == "val",
           "system": {"definition": "p(i) ∝ (p_model(i) · p_potion-cos-4bit(i))^beta", "beta": beta, "tau_ask3": tau,
                      "partner_theta": partner_params.get("theta"), "partner_temperature": partner_params.get("temperature"),
                      "bytes": BYTES_MODEL + BYTES_EMBEDDING,
                      "model_file": common.load_json(model_info) if os.path.isfile(model_info) else None},
           "verdict": v, "differences": diffs, "goals": goals(sys_m, acc(main.get(FAST))), "readings": readings,
           "latency": latency, "results": results, "disclosure": DISCLOSURE}
    if split == "val":
        out["note"] = "rehearsal on the validation slates: not the gate's verdict"
        out["rehearsal_check"] = rehearsal_check(fit, main, slates[names["main"]], beta)
    os.makedirs(a.out_dir, exist_ok=True)
    common.write_json(os.path.join(a.out_dir, "gate.json"), out)
    with open(os.path.join(a.out_dir, "results.md"), "w") as f:
        f.write(report(out))
    if split == "val":
        ok = out["rehearsal_check"]["reproduces"]
        print(f"wrote {a.out_dir}/results.md and gate.json: rehearsal, {'reproduces' if ok else 'DOES NOT reproduce'} runs/s6-nano3/fit.json")
        return 0 if ok else 1
    print(f"wrote {a.out_dir}/results.md and gate.json: {'PASS' if v['pass'] else 'FAIL'}")
    return 0 if v["pass"] else 1


# ---------------------------------------------------------------- the report

def fmt(x, d=4):
    return f"{x:.{d}f}" if finite(x) else "—"


def yes(b):
    return "yes" if b else "no"


def report(o):
    v, split = o["verdict"], o["split"]
    main = FILES[split]["main"]
    if o["rehearsal"]:
        title = "# s6-route (s6-nano v3): rehearsal on the validation slates (not the gate's verdict)"
    else:
        title = f"# s6-route (s6-nano v3): {'gate passed' if v['pass'] else 'gate not passed'} on the fresh slates"
    sysd = o["system"]
    lines = [title, "", f"**Disclosure.** {o['disclosure']}", "",
             f"System: {sysd['definition']}, β = {sysd['beta']}, τ_ask3 = {fmt(sysd['tau_ask3'], 6)} (runs/s6-nano3/fit.json); "
             f"potion-cos-4bit θ = {fmt(sysd['partner_theta'], 6)}, T = {fmt(sysd['partner_temperature'], 7)}; "
             f"{sysd['bytes']:,} bytes.", ""]
    if o.get("rehearsal_check"):
        rc = o["rehearsal_check"]
        g = rc["got"]
        lines += [f"**Rehearsal check: the fused system on v_tools {'reproduces' if rc['reproduces'] else 'does NOT reproduce'} "
                  f"runs/s6-nano3/fit.json** (accuracy {fmt(g['accuracy'])} against {fmt(rc['want']['accuracy'])}; ECE "
                  f"{fmt(g['ece_fit_convention'])} with fit.json's bins against {fmt(rc['want']['ece_fit_convention'])}, "
                  f"{fmt(g['ece_scorer'])} with v2's scorer; NLL {fmt(g['nll'])}; τ_ask3 {fmt(g['tau_ask3']['tau'], 6)}; "
                  f"model dump vs runs/s6-nano2/v_tools.dump.jsonl: max |Δp| {fmt(rc['model_dump_max_abs_diff_vs_runs_s6-nano2'], 9)}).", ""]
    g1v, g3v, g4v = v["G1"], v["G3"], v["G4"]
    d = o["differences"].get("bm25") or {}
    lines += ["| condition | value | bar | pass |", "|---|---:|---:|:---:|",
              f"| G1 system − bm25 on {main} | {fmt(g1v['bm25_difference'])} [{fmt(d.get('lo'))}, {fmt(d.get('hi'))}] | > 0, lower bound > 0 | {yes(g1v['above_bm25'])} |",
              f"| G1 system vs minilm-cos − 0.03 | {fmt(g1v['system'])} | ≥ {fmt((g1v['minilm-cos'] if finite(g1v['minilm-cos']) else float('nan')) - G1_MINILM)} | {yes(g1v['minilm_within_0.03'])} |",
              f"| G1 system vs the model alone | {fmt(g1v['system'])} | > {fmt(g1v['model'])} | {yes(g1v['above_model'])} |",
              f"| G1 system vs potion-cos-4bit alone | {fmt(g1v['system'])} | > {fmt(g1v[PARTNER])} | {yes(g1v['above_partner'])} |"]
    for n, x in v["G2"].items():
        if isinstance(x, dict):
            lines.append(f"| G2 {n}: system vs max(model, potion-cos-4bit) − 0.02 | {fmt(x['system'])} | ≥ {fmt(x['bar'])} | {yes(x['pass'])} |")
    if v["G2"].get("note"):
        lines.append(f"| G2 | — | {v['G2']['note']} | no |")
    lines.append(f"| G3 ECE on {main} (15 bins) | {fmt(g3v['ece'])} | ≤ 0.05 | {yes(g3v['pass'])} |")
    if g4v.get("measured"):
        lines += [f"| G4 max abs p gap, WASM vs Python ({g4v['n']} slates) | {fmt(g4v['max_abs_diff'], 7)} | ≤ 1e-3 | {yes(g4v['parity'])} |",
                  f"| G4 same answer | {g4v['argmax_agree']} | ≥ 199 of 200 | {yes(g4v['same_answer'])} |",
                  f"| G4 WASM median, tool list cached (µs) | {fmt(g4v['median_cached_us'], 1)} | ≤ 60000 | {yes(g4v['latency'])} |",
                  f"| G4 the two files (bytes) | {g4v['bytes']} | ≤ 4,000,000 | {yes(g4v['size'])} |"]
    else:
        lines.append("| G4 | not measured | the WASM bench's JSON | no |")
    lines += ["", f"Gate: **{'passed' if v['pass'] else 'not passed'}**" + (" (a rehearsal)" if o["rehearsal"] else ""), ""]
    if o["differences"]:
        lines += [f"Paired bootstrap of the accuracy differences on {main} (1,000 resamples of the slates, seed 0, 95%):", "",
                  "| system vs | difference | interval |", "|---|---:|---|"]
        for s, x in o["differences"].items():
            lines.append(f"| {s} | {fmt(x['diff'])} | [{fmt(x['lo'])}, {fmt(x['hi'])}] |")
        lines.append("")
    gl = o["goals"]
    lines += ["## Announced goals (not gates)", "",
              f"- accuracy ≥ 0.87 on {main}: {fmt(gl['accuracy_0.87']['value'])} ({'met' if gl['accuracy_0.87']['met'] else 'not met'})",
              f"- ≥ 0.95 among answered at τ_ask3 with coverage ≥ 0.80: {fmt(gl['answered_0.95_coverage_0.80']['accuracy'])} at coverage "
              f"{fmt(gl['answered_0.95_coverage_0.80']['coverage'])} ({'met' if gl['answered_0.95_coverage_0.80']['met'] else 'not met'})",
              f"- the fast mode within 0.015 of full depth: {fmt(gl['fast_within_0.015']['fast'])} against {fmt(gl['fast_within_0.015']['full'])} "
              f"({'met' if gl['fast_within_0.015']['met'] else 'not met'})", ""]
    r = o["readings"]
    lines += ["## Readings of section 6 (reported, not gated)", "",
              f"Clusters: {r['clusters']['count']} ({r['clusters']['source']}); the three largest hold {r['clusters']['slates_in_three_largest']} slates.", "",
              f"Without the three largest clusters ({r['without_three_largest']['slates']} slates):", "",
              "| system | accuracy |", "|---|---:|"]
    for s, x in r["without_three_largest"]["accuracy"].items():
        lines.append(f"| {s} | {fmt(x)} |")
    lines += ["", "| system vs | difference, without the three largest (slate bootstrap) | interval | difference, all slates (cluster bootstrap) | interval |",
              "|---|---:|---|---:|---|"]
    for s, x in r["cluster_bootstrap"].items():
        w = r["without_three_largest"]["differences"].get(s) or {}
        lines.append(f"| {s} | {fmt(w.get('diff'))} | [{fmt(w.get('lo'))}, {fmt(w.get('hi'))}] | {fmt(x['diff'])} | [{fmt(x['lo'])}, {fmt(x['hi'])}] |")
    lines += ["", f"Names masked ({FILES[split]['masked']}; the system is expected to fail there):", "", "| system | accuracy |", "|---|---:|"]
    for s, x in r["names_masked"].items():
        lines.append(f"| {s} | {fmt(x)} |")
    lines += ["", "## Every system, every file", "",
              "| system | file | n | accuracy | tool gold | none gold | none P | none R | none F1 | ECE | NLL | mean conf | none answers | at τ_ask3: accuracy | coverage |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for s, per in o["results"].items():
        for n, m in per.items():
            at = m.get("at_tau_ask") or {}
            lines.append(f"| {s} | {n} | {m['n']} | {fmt(m['accuracy'])} | {fmt(m['accuracy_tool_gold'])} | {fmt(m['accuracy_none_gold'])} | "
                         f"{fmt(m['none_precision'])} | {fmt(m['none_recall'])} | {fmt(m['none_f1'])} | {fmt(m['ece'])} | {fmt(m['nll'])} | "
                         f"{fmt(m['mean_confidence'])} | {fmt(m['none_answer_share'])} | {fmt(at.get('accuracy'))} | {fmt(at.get('coverage'))} |")
    sm = o["results"].get(SYSTEM, {}).get(main)
    if sm:
        lines += ["", f"Risk–coverage of the system on {main}:", "", "| coverage | accuracy | min confidence |", "|---:|---:|---:|"]
        for p in sm["risk_coverage"]:
            lines.append(f"| {fmt(p['coverage'], 3)} | {fmt(p['accuracy'])} | {fmt(p['min_confidence'])} |")
    lat = o["latency"]
    lines += ["", "## Latency", ""]
    nat = (lat["native_model"] or {}).get("latency")
    lines.append(f"- the model, native (sky-eval --latency, 200 slates of {main}): " + (json.dumps(nat) if nat else "not measured"))
    for s, t in lat["baselines_cpu"].items():
        lines.append(f"- {s}: {fmt(t.get('cpu_ms_per_slate'), 2)} ms of CPU per slate ({t.get('note')})")
    lines.append("- WASM: " + (f"median {fmt(lat['wasm'].get('median_cached_us'), 1)} µs cached, first {fmt(lat['wasm'].get('median_first_us'), 1)} µs"
                               if lat["wasm"].get("measured") else "not measured"))
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    gd = sub.add_parser("guard")
    gd.add_argument("--split", choices=("fresh", "val"), required=True)
    g = sub.add_parser("gate")
    g.add_argument("--split", choices=("fresh", "val"), required=True)
    g.add_argument("--root", help="where the dumps are (default <data>/eval/fresh, or <data>/eval/fresh/rehearsal for val)")
    g.add_argument("--fit", default=os.path.join(common.REPO, "runs", "s6-nano3", "fit.json"))
    g.add_argument("--wasm")
    g.add_argument("--native")
    g.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    try:
        if a.cmd == "guard":
            guard(a.split)
            check_fit(common.load_json(os.path.join(common.REPO, "runs", "s6-nano3", "fit.json")))
            return 0
        return gate(a)
    except common.Refused as e:
        print(f"refused: {e}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    sys.exit(main())
