"""The gate of s6-route-nano (docs/s6-nano2-prereg.md, sections 5, 7 and 8), from the dumps.

    s6-route-gate.py pilot --nano D --micro D --potion D --slates v_tools.jsonl --out summary.json
    s6-route-gate.py tau   --dump D --slates v_tools.jsonl [--out tau.json]
    s6-route-gate.py gate  --eval DIR --split val|test [--seat model] [--model GGUF]
                           [--wasm wasm.json] [--native native.json] --out-dir DIR

A dump is `sky-eval --dump-probs` or a baseline's file (scripts/route-baselines.py), in
<eval>/<split>/<system>/<file>.jsonl; the model's are under `model`. The slates are read through
scripts/route/common.py, which refuses the test until runs/s6-nano2/TEST-OPEN exists and the
model holds tau_ask. `--seat` puts another system in the model's place (the validation
rehearsal of section 6). G1 (amendment 1) also asks for an accuracy above `bm25`'s. A failed gate is a result: results.md and gate.json are written and the
exit status is 1; an input that is missing fails its condition.
"""

import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "route"))
import common  # noqa: E402
import scorer  # noqa: E402

G1_MARGIN = 0.03
G2_NONE = (0.10, 0.30)
G2_NAME_EXCESS = 0.0  # predicted share <= 2 x gold share + 0.01
G2_CHANGED = 0.80
G3_ECE = 0.05
G4_MS = 2.0
G4_PARITY = 1e-4
PILOT_MICRO = 0.04
PILOT_ABORT = 0.15
GATED = ("potion-cos", "static-ft", "pair-lr", "bm25")
REPORTED = ("potion-cos-4bit", "minilm-cos", "toolrouter", "name-prior", "s6-nano-v1")
SOURCES = {
    "val": ("v_tools",),
    "test": ("t_tools", "t_metatool", "t_clinc", "t_toolrouting", "t_bfcl", "t_when2call", "t_massive"),
}


def finite(x):
    return x is not None and isinstance(x, (int, float)) and math.isfinite(x)


# ---------------------------------------------------------------- the pure rules

def pilot_rule(nano, micro, potion):
    """Section 5: the size, and whether the main run happens. Validation accuracies."""
    size = "micro" if finite(micro) and finite(nano) and micro >= nano + PILOT_MICRO else "nano"
    chosen = micro if size == "micro" else nano
    bar = potion - PILOT_ABORT if finite(potion) else None
    abort = not (finite(chosen) and finite(bar) and chosen >= bar)
    return {"nano": nano, "micro": micro, "potion_cos": potion, "size": size, "chosen_accuracy": chosen,
            "abort_bar": bar, "abort": abort,
            "rule": "micro if its accuracy >= nano's + 0.04; abort if the chosen pilot's accuracy < potion-cos - 0.15"}


def g1(model_acc, base):
    """G1: >= potion-cos + 0.03, > static-ft, > pair-lr (section 8), and > bm25 (amendment 1):
    point estimates on T-tools."""
    out = {"model": model_acc}
    ok = finite(model_acc)
    for name in GATED:
        b = base.get(name)
        out[name] = b
        if not finite(b):
            ok = False
        elif name == "potion-cos":
            ok = ok and model_acc >= b + G1_MARGIN - 1e-12
        else:
            ok = ok and model_acc > b
    out["pass"] = bool(ok)
    return out


def g2(m, changes):
    none = m.get("none_answer_share") if m else None
    excess = (m or {}).get("largest_name_excess", {}).get("excess")
    share = (changes or {}).get("share")
    c1 = finite(none) and G2_NONE[0] <= none <= G2_NONE[1]
    c2 = excess is None or excess == float("-inf") or (finite(excess) and excess <= G2_NAME_EXCESS + 1e-12)
    c3 = finite(share) and share >= G2_CHANGED
    return {"none_answer_share": none, "largest_name_excess": (m or {}).get("largest_name_excess"),
            "gold_removed_changed": changes, "none_in_range": c1, "no_dominant_name": bool(c2 and m),
            "answers_change": c3, "pass": bool(c1 and c2 and m and c3)}


def g3(m):
    e = (m or {}).get("ece")
    return {"ece": e, "pass": bool(finite(e) and e <= G3_ECE)}


def g4(wasm):
    med = (wasm or {}).get("median_cached_us")
    gap = (wasm or {}).get("max_abs_diff")
    agree = (wasm or {}).get("argmax_agree") == (wasm or {}).get("n") and (wasm or {}).get("n")
    ok = finite(med) and med <= G4_MS * 1000 and finite(gap) and gap <= G4_PARITY and bool(agree)
    return {"median_cached_us": med, "median_first_us": (wasm or {}).get("median_first_us"), "max_abs_diff": gap,
            "argmax_agree": (wasm or {}).get("argmax_agree"), "n": (wasm or {}).get("n"), "pass": bool(ok)}


def verdict(seat_metrics, base_acc, changes, wasm):
    v = {"G1": g1((seat_metrics or {}).get("accuracy"), base_acc), "G2": g2(seat_metrics, changes),
         "G3": g3(seat_metrics), "G4": g4(wasm)}
    v["pass"] = all(v[k]["pass"] for k in ("G1", "G2", "G3", "G4"))
    return v


# ---------------------------------------------------------------- reading the dumps

def items(eval_dir, split, system, name, model):
    path = os.path.join(eval_dir, split, system, f"{name}.jsonl")
    if not os.path.isfile(path):
        return None
    return scorer.load(path, common.slate_file(split, name, model))


def gate(a):
    split, seat = a.split, a.seat
    pre = "v" if split == "val" else "t"
    systems = [seat] + [s for s in GATED + REPORTED if s != seat]
    tau = None
    if seat == "model" and a.model:
        md = common.gguf_metadata(a.model)
        tau = md.get(common.TAU_KEY)
    results, main = {}, {}
    for system in systems:
        per = {}
        for name in SOURCES[split]:
            its = items(a.eval, split, system, name, a.model)
            if its is None:
                continue
            per[name] = scorer.metrics(its, tau if system == seat else None)
            if name == f"{pre}_tools":
                main[system] = its
        checks = {}
        pairs = items(a.eval, split, system, f"{pre}_tools_pairs", a.model)
        if pairs:
            checks["gold_removed"] = scorer.pair_changes(pairs)
        masked = items(a.eval, split, system, f"{pre}_tools_masked", a.model)
        if masked:
            checks["names_masked_accuracy"] = scorer.accuracy(masked)
        swapped = items(a.eval, split, system, f"{pre}_tools_swapped", a.model)
        if swapped:
            checks["descriptions_swapped"] = scorer.follow_share(swapped)
        if per or checks:
            results[system] = {"sources": per, "reading_checks": checks}
    key = f"{pre}_tools"
    seat_m = results.get(seat, {}).get("sources", {}).get(key)
    base = {s: results.get(s, {}).get("sources", {}).get(key, {}).get("accuracy") for s in GATED}
    wasm = common.load_json(a.wasm) if a.wasm and os.path.isfile(a.wasm) else None
    v = verdict(seat_m, base, results.get(seat, {}).get("reading_checks", {}).get("gold_removed"), wasm)
    diffs = {}
    if seat in main:
        for s in GATED:
            if s in main:
                diffs[s] = scorer.bootstrap_diff(main[seat], main[s])
    out = {"split": split, "seat": seat, "tau_ask": tau, "verdict": v, "differences": diffs,
           "native": common.load_json(a.native) if a.native and os.path.isfile(a.native) else None,
           "results": results}
    if split != "test":
        out["note"] = "validation: a rehearsal of the gate, not its verdict"
    os.makedirs(a.out_dir, exist_ok=True)
    common.write_json(os.path.join(a.out_dir, "gate.json"), out)
    with open(os.path.join(a.out_dir, "results.md"), "w") as f:
        f.write(report(out))
    print(f"wrote {a.out_dir}/results.md and gate.json: {'PASS' if v['pass'] else 'FAIL'}")
    return 0 if v["pass"] else 1


def fmt(x, d=4):
    return f"{x:.{d}f}" if finite(x) else "—"


def report(o):
    v = o["verdict"]
    lines = [f"# s6-route-nano: {'gate passed' if v['pass'] else 'gate not passed'} ({o['split']}, seat `{o['seat']}`)", ""]
    if o.get("note"):
        lines += [f"**{o['note']}.**", ""]
    g = v["G1"]
    lines += ["| condition | value | bar | pass |", "|---|---:|---:|:---:|",
              f"| G1 accuracy vs potion-cos + 0.03 | {fmt(g['model'])} | {fmt((g['potion-cos'] or float('nan')) + G1_MARGIN)} | {'yes' if v['G1']['pass'] else 'no'} |",
              f"| G1 vs static-ft | {fmt(g['model'])} | > {fmt(g['static-ft'])} | |",
              f"| G1 vs pair-lr | {fmt(g['model'])} | > {fmt(g['pair-lr'])} | |",
              f"| G1 vs bm25 (amendment 1) | {fmt(g['model'])} | > {fmt(g['bm25'])} | |",
              f"| G2 none answers | {fmt(v['G2']['none_answer_share'])} | 0.10–0.30 | {'yes' if v['G2']['none_in_range'] else 'no'} |",
              f"| G2 largest name excess | {fmt((v['G2']['largest_name_excess'] or {}).get('excess'))} | ≤ 0 | {'yes' if v['G2']['no_dominant_name'] else 'no'} |",
              f"| G2 answer changes, gold removed | {fmt((v['G2']['gold_removed_changed'] or {}).get('share'))} | ≥ 0.80 | {'yes' if v['G2']['answers_change'] else 'no'} |",
              f"| G3 ECE | {fmt(v['G3']['ece'])} | ≤ 0.05 | {'yes' if v['G3']['pass'] else 'no'} |",
              f"| G4 WASM median, tool list cached (µs) | {fmt(v['G4']['median_cached_us'], 1)} | ≤ 2000 | {'yes' if v['G4']['pass'] else 'no'} |",
              f"| G4 max abs p gap | {fmt(v['G4']['max_abs_diff'], 7)} | ≤ 1e-4 | |", ""]
    if o["differences"]:
        lines += ["Paired bootstrap of the accuracy differences (1,000 resamples of the slates, 95%):", "",
                  "| vs | difference | interval |", "|---|---:|---|"]
        for s, d in o["differences"].items():
            lines.append(f"| {s} | {fmt(d['diff'])} | [{fmt(d['lo'])}, {fmt(d['hi'])}] |")
        lines.append("")
    lines += ["## Every system, every source", "",
              "| system | source | n | accuracy | tool gold | none gold | none P | none R | none F1 | ECE | NLL | mean conf | none answers |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for s, r in o["results"].items():
        for src, m in r["sources"].items():
            lines.append(f"| {s} | {src} | {m['n']} | {fmt(m['accuracy'])} | {fmt(m['accuracy_tool_gold'])} | {fmt(m['accuracy_none_gold'])} | "
                         f"{fmt(m['none_precision'])} | {fmt(m['none_recall'])} | {fmt(m['none_f1'])} | {fmt(m['ece'])} | {fmt(m['nll'])} | "
                         f"{fmt(m['mean_confidence'])} | {fmt(m['none_answer_share'])} |")
    lines += ["", "## Reading checks", "", "| system | answer changes, gold removed | names masked: accuracy | swapped: follows description | follows name |",
              "|---|---:|---:|---:|---:|"]
    for s, r in o["results"].items():
        c = r["reading_checks"]
        lines.append(f"| {s} | {fmt((c.get('gold_removed') or {}).get('share'))} | {fmt(c.get('names_masked_accuracy'))} | "
                     f"{fmt((c.get('descriptions_swapped') or {}).get('follow_description'))} | {fmt((c.get('descriptions_swapped') or {}).get('follow_name'))} |")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pilot")
    for k in ("nano", "micro", "potion"):
        p.add_argument(f"--{k}", required=True)
    p.add_argument("--slates", required=True)
    p.add_argument("--out", required=True)
    t = sub.add_parser("tau")
    t.add_argument("--dump", required=True)
    t.add_argument("--slates", required=True)
    t.add_argument("--out")
    g = sub.add_parser("gate")
    g.add_argument("--eval", default=common.eval_root())
    g.add_argument("--split", choices=common.SPLITS, required=True)
    g.add_argument("--seat", default="model")
    g.add_argument("--model")
    g.add_argument("--wasm")
    g.add_argument("--native")
    g.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    try:
        if a.cmd == "pilot":
            acc = {}
            for k in ("nano", "micro", "potion"):
                path = getattr(a, k)
                acc[k] = scorer.accuracy(scorer.load(path, a.slates)) if os.path.isfile(path) else None
            r = pilot_rule(acc["nano"], acc["micro"], acc["potion"])
            r["inputs"] = {k: getattr(a, k) for k in ("nano", "micro", "potion")}
            r["slates_sha256"] = common.sha256_file(a.slates)
            common.write_json(a.out, r)
            print(json.dumps({k: r[k] for k in ("size", "abort", "nano", "micro", "potion_cos", "abort_bar")}))
            return 0
        if a.cmd == "tau":
            its = scorer.load(a.dump, a.slates)
            t_ = scorer.tau_ask(its)
            r = {"tau_ask": t_, "reached": t_ is not None,
                 "at_tau": scorer.at_threshold(its, t_ if t_ is not None else 1.0), "n": len(its),
                 "rule": "the smallest top probability at which validation accuracy among answered requests is >= 0.95 (section 3); 1.0 when none reaches it"}
            if a.out:
                common.write_json(a.out, r)
            print(repr(float(t_ if t_ is not None else 1.0)))
            return 0
        return gate(a)
    except common.Refused as e:
        print(f"refused: {e}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    sys.exit(main())
