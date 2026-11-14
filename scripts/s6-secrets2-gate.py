#!/usr/bin/env python3
"""Publication gate and results of s6-secrets v2, s6-secrets-verify-pico
(docs/s6-secrets2-prereg.md, sections 7 to 9, amendments 1 and 2).

Reads what scripts/secrets2-eval.sh writes in the data folder (hashes and numbers only):
  scores.json        scripts/secrets2/scorer.py: every system on the CredData split
  issues.json        scripts/secrets2/issues.py score: the issue-report split
  clean.json         scripts/secrets2/clean_text.py share: the clean-text corpus
  model-latency.json scripts/secrets2/run_model.py eval: native latency, size, exits

The model is published only if all of these hold, on the TEST split, ternary GGUF, full depth,
tau* from validation:
  G1  CredData line F1 >= F1 of the best rule scanner (Gitleaks 8.30.1, Betterleaks 1.10.1,
      secretgate default) + 0.15
  G2  CredData line F1 >= F1 of cbx-secrets-detector-17M - 0.10   (announced goal: - 0.05)
  G3  CredData line F1 > F1 of the size-matched LR
  G4  issue-report candidate F1 >= Gitleaks F1 on the same candidates + 0.08
  G5  ECE <= 0.05 on the CredData candidates
  G6  native median latency <= 3 ms per candidate, full depth
  G7  clean real text: files with at least one accepted candidate <= 1.0%
G1, G2 and G3 must each hold twice: on every repository of the split, and without the largest
test repository (amendment 1). G1, G2, G3 and G5 are read on the gated universe, where the rows
of category exactly "UUID" are removed (amendment 2); the full-universe numbers are written
next to them. G7 is read on the corpus as the census read it (no source path).

Usage: s6-secrets2-gate.py --scores scores.json [--issues issues.json] [--clean clean.json]
           [--latency model-latency.json] [--model model] --out-dir DIR
Writes DIR/results.md and DIR/gate.json. Exit status 0 when the gate passes, 1 otherwise. On a
split other than test the same computation is a rehearsal and says so; `--model lr` puts
another system in the model's seat to rehearse the gate before a model exists.
"""
import argparse
import json
import os
import sys

EPS = 1e-9
G1_MARGIN, G2_MARGIN, G2_GOAL, G4_MARGIN = 0.15, 0.10, 0.05, 0.08
MAX_ECE, MAX_LATENCY_US, MAX_CLEAN_SHARE = 0.05, 3000.0, 0.010
RULE_SCANNERS = ("gitleaks", "betterleaks", "secretgate-default")
CBX, LR, GITLEAKS = "cbx-17m", "lr", "gitleaks"
ORDER = ("model", "generator", "secretgate-default", "gitleaks", "betterleaks", "detect-secrets", "credsweeper-ml", "credsweeper-noml", "cbx-17m", "lr", "potion-lr")
PARTS = (("all", "every repository"), ("without_largest", "without the largest repository"))
UUID_SENTENCE = ("Gated universe: the rows whose CredData category is exactly `UUID` are removed, for every system (amendment 2). "
                 "That choice was made after reading the training and validation rows and before any test file was read. "
                 "Full universe: every meta row.")


def load(path):
    with open(path) as f:
        return json.load(f)


def maybe(path):
    return load(path) if path and os.path.exists(path) else None


def fmt(x, digits=4):
    return "—" if x is None else f"{x:.{digits}f}"


def ordered(names, model):
    first = [model] if model in names else []
    known = [n for n in ORDER if n in names and n != model]
    return first + known + sorted(n for n in names if n not in known and n != model)


def line_f1(scores, system, part, universe="gated"):
    s = scores["systems"].get(system)
    return None if s is None else s["line"][universe][part]["f1"]


def gates(scores, issues, clean, latency, model):
    """{gate: {"holds": bool, "why": [...], ...numbers}} for G1 to G7."""
    out = {}
    sysm = scores["systems"].get(model)

    def twice(name, reference, test, describe):
        g = {"holds": True, "why": [], "parts": {}}
        for part, label in PARTS:
            m = line_f1(scores, model, part)
            refs = reference(part)
            if m is None or refs is None:
                why = f"no result of {model}" if m is None else "no result of the reference"
                if why not in g["why"]:
                    g["why"].append(why)
                g["holds"] = False
                continue
            ref_name, ref = refs
            ok = test(m, ref)
            g["parts"][part] = {"model": m, "reference": ref_name, "reference_f1": ref, "holds": ok,
                                "model_full_universe": line_f1(scores, model, part, "full"), "reference_full_universe": line_f1(scores, ref_name, part, "full")}
            if not ok:
                g["why"].append(f"{label}: {m:.4f} {describe(ref_name, ref)}")
                g["holds"] = False
        out[name] = g

    def best_scanner(part):
        have = [(line_f1(scores, s, part), s) for s in RULE_SCANNERS if s in scores["systems"]]
        if len(have) != len(RULE_SCANNERS):
            return None
        f, s = max(have)
        return s, f

    def one(system):
        return lambda part: None if system not in scores["systems"] else (system, line_f1(scores, system, part))

    twice("G1", best_scanner, lambda m, r: m >= r + G1_MARGIN - EPS, lambda n, r: f"< {n} {r:.4f} + {G1_MARGIN}")
    twice("G2", one(CBX), lambda m, r: m >= r - G2_MARGIN - EPS, lambda n, r: f"< {n} {r:.4f} - {G2_MARGIN}")
    twice("G3", one(LR), lambda m, r: m > r, lambda n, r: f"<= {n} {r:.4f}")

    g4 = {"holds": False, "why": []}
    im = issues["systems"].get(model) if issues else None
    ig = issues["systems"].get(GITLEAKS) if issues else None
    if im is None or ig is None:
        g4["why"].append("no issue-report result of " + (model if im is None else "Gitleaks"))
    else:
        m, r = im["candidate"]["f1"], ig["candidate"]["f1"]
        g4.update({"model": m, "gitleaks": r, "holds": m >= r + G4_MARGIN - EPS})
        if not g4["holds"]:
            g4["why"].append(f"{m:.4f} < Gitleaks {r:.4f} + {G4_MARGIN}")
    out["G4"] = g4

    g5 = {"holds": False, "why": []}
    cand = (sysm or {}).get("candidate", {}).get("gated") if sysm else None
    if cand is None or cand.get("ece") is None:
        g5["why"].append(f"no ECE of {model}")
    else:
        g5.update({"ece": cand["ece"], "ece_full_universe": sysm["candidate"]["full"].get("ece"), "holds": cand["ece"] <= MAX_ECE + EPS})
        if not g5["holds"]:
            g5["why"].append(f"ECE {cand['ece']:.4f} > {MAX_ECE}")
    out["G5"] = g5

    g6 = {"holds": False, "why": []}
    if latency is None or latency.get("median_us_full_depth") is None:
        g6["why"].append("no native latency")
    else:
        us = latency["median_us_full_depth"]
        g6.update({"median_us": us, "holds": us <= MAX_LATENCY_US + EPS})
        if not g6["holds"]:
            g6["why"].append(f"{us:.0f} µs > {MAX_LATENCY_US:.0f} µs")
    out["G6"] = g6

    g7 = {"holds": False, "why": []}
    cm = None
    if clean:
        cm = clean["readings"].get(clean.get("gate_reading", "flags"), {}).get("systems", {}).get(model)
    if cm is None:
        g7["why"].append(f"no clean-text result of {model}")
    else:
        g7.update({"files": cm["files_with_an_accepted_flag"], "of": cm["files"], "share": cm["share"], "holds": cm["share"] <= MAX_CLEAN_SHARE + EPS})
        if not g7["holds"]:
            g7["why"].append(f"{100 * cm['share']:.2f}% of the files > {100 * MAX_CLEAN_SHARE:.1f}%")
    out["G7"] = g7

    goal = None
    if "all" in out["G2"]["parts"] and "without_largest" in out["G2"]["parts"]:
        goal = all(p["model"] >= p["reference_f1"] - G2_GOAL - EPS for p in out["G2"]["parts"].values())
    return out, goal


def results(scores, issues, clean, latency, model, g, goal):
    split = scores["split"]
    systems = scores["systems"]
    names = ordered(systems, model)
    L = [f"# s6-secrets v2 (s6-secrets-verify-pico): results on the {split} split (docs/s6-secrets2-prereg.md)", ""]
    rehearsal = split != "test"
    if rehearsal:
        L += [f"**Rehearsal.** These are the {split} repositories, not the test: no number below is a result of the pre-registration, and the gate at the end is the test's computation run on {split}.", ""]
    if model not in systems:
        L += [f"No system named `{model}` was scored: the baselines are reported, the gate has no model to judge.", ""]
    elif model != "model":
        L += [f"**`{model}` sits in the model's seat**: the gate lines below judge it as they will judge the model.", ""]
    big = scores["largest_repository"]
    L += [f"{scores['repositories']} repositories, {scores['files']} files; T rows: {scores['t_rows']['gated']} (gated universe), {scores['t_rows']['full']} (full). "
          f"Largest repository: `{big['repo_id']}`, {big['t_rows']['gated']} gated T rows" + (" (the split's own largest, a stand-in for the test's)" if big.get("rehearsal_stand_in") else "") + ".", ""]

    L += ["## CredData, line level (7.1)", "", UUID_SENTENCE, ""]
    for u in ("gated", "full"):
        L += [f"| {u} universe | precision | recall | F1 | false alarms | F1 without the largest | macro F1 (≥ 20 T rows) |", "|---|---:|---:|---:|---:|---:|---:|"]
        for n in names:
            x = systems[n]["line"][u]
            a = x["all"]
            L.append(f"| {n} | {fmt(a['precision'])} | {fmt(a['recall'])} | {fmt(a['f1'])} | {fmt(a['false_alarm_rate'])} | {fmt(x['without_largest']['f1'])} | "
                     f"{fmt(x['macro_f1']['value'])} ({x['macro_f1']['repositories']}) |")
        L.append("")
    L += ["| | version | unit | unlabelled-line alarms | per 1,000 files (7.3) | CPU s |", "|---|---|---|---:|---:|---:|"]
    for n in names:
        s = systems[n]
        L.append(f"| {n} | {s.get('version') or '—'} | {s.get('unit') or '—'} | {s['unlabelled']['flags']} | {fmt(s['unlabelled']['per_1000_files'], 1)} | {fmt(s.get('cpu_seconds'), 1)} |")
    L.append("")
    if "credsweeper-ml" in systems:
        L += ["CredSweeper's ML was trained on a random 85% of all CredData lines: its CredData numbers are in-sample and enter no gate (section 6).", ""]
    partial = [n for n in names if systems[n].get("unit") == "candidate" and n != "generator" and "generator" in systems
               and sum(systems[n]["labels"]["full"].values()) < sum(systems["generator"]["labels"]["full"].values())]
    for n in partial:
        L += [f"`{n}` scored {sum(systems[n]['labels']['full'].values())} of the generator's {sum(systems['generator']['labels']['full'].values())} candidates: "
              "a line it could only reach through an unscored candidate counts as not flagged, and its unlabelled-line alarms are not counted.", ""]

    cands = [n for n in names if "candidate" in systems[n]]
    if cands:
        L += ["## CredData, candidate level (7.2)", "", UUID_SENTENCE, ""]
        for u in ("gated", "full"):
            L += [f"| {u} universe | candidates | positives | threshold | precision | recall | F1 | PR-AUC | recall at 1% false alarms | ECE | ECE on P(real) |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
            for n in cands:
                c = systems[n]["candidate"][u]
                L.append(f"| {n} | {c['n']} | {c['positives']} | {fmt(systems[n].get('threshold'))} | {fmt(c['precision'])} | {fmt(c['recall'])} | {fmt(c['f1'])} | "
                         f"{fmt(c.get('pr_auc'))} | {fmt(c.get('recall_at_1pct_false_alarms'))} | {fmt(c.get('ece'))} | {fmt(c.get('ece_positive'))} |")
            L.append("")
        L += ["Candidates on a labelled line that overlap no labelled value are dropped (3.1); ECE is over 15 equal-width bins of the answer's confidence, as sky-eval computes it.", ""]

    spans = [n for n in names if "span" in systems[n]]
    if spans:
        L += ["## CredData, span level (section 9, overlap with a T value)", "", UUID_SENTENCE, "",
              "| | gated recall | gated precision | gated F1 | full recall | full precision | full F1 | flags: T / X / F / no row / unlabelled |", "|---|---:|---:|---:|---:|---:|---:|---|"]
        for n in spans:
            a, b, lab = systems[n]["span"]["gated"], systems[n]["span"]["full"], systems[n]["labels"]["full"]
            L.append(f"| {n} | {fmt(a['recall'])} | {fmt(a['precision'])} | {fmt(a['f1'])} | {fmt(b['recall'])} | {fmt(b['precision'])} | {fmt(b['f1'])} | "
                     f"{lab['T']} / {lab['X']} / {lab['F']} / {lab['D']} / {lab['U']} |")
        L += ["", "The generator's row is its recall (2.2): the share of T rows a candidate overlaps.", ""]

    L += ["## By CredData category and by path scope (section 9), gated universe, line F1 (recall)", ""]
    cats = list(next(iter(systems.values()))["line"]["gated"]["by_category"]) if systems else []
    scopes = ["test", "src", "other"]
    L += ["| | " + " | ".join(cats + [f"scope {s}" for s in scopes]) + " |", "|---|" + "---:|" * (len(cats) + len(scopes))]
    for n in names:
        x = systems[n]["line"]["gated"]
        cells = [f"{fmt(x['by_category'][c]['f1'], 3)} ({fmt(x['by_category'][c]['recall'], 3)})" for c in cats]
        cells += [f"{fmt(x['by_scope'][s]['f1'], 3)} ({fmt(x['by_scope'][s]['recall'], 3)})" for s in scopes]
        L.append(f"| {n} | " + " | ".join(cells) + " |")
    if names:
        x = systems[names[0]]["line"]["gated"]
        count = lambda v: f"{v['tp'] + v['fn']} / {v['fp'] + v['tn']}"
        L.append("| positive / negative lines | " + " | ".join([count(x["by_category"][c]) for c in cats] + [count(x["by_scope"][s]) for s in scopes]) + " |")
    L.append("")

    b = scores.get("bootstrap") or {}
    if b.get("f1_difference"):
        L += [f"## Uncertainty (7.6): paired bootstrap over repositories, {b['resamples']} resamples, 95%", "",
              f"| line F1 of {b['reference']} minus | gated | gated, without the largest | full | full, without the largest |", "|---|---|---|---|---|"]
        cell = lambda d: f"{d['value']:+.4f} [{fmt(d['lo'])}, {fmt(d['hi'])}]"
        for n in ordered(b["f1_difference"], model):
            d = b["f1_difference"][n]
            L.append(f"| {n} | {cell(d['gated']['all'])} | {cell(d['gated']['without_largest'])} | {cell(d['full']['all'])} | {cell(d['full']['without_largest'])} |")
        L.append("")
    if b.get("ece"):
        L += ["| ECE | gated | full |", "|---|---|---|"]
        for n in ordered(b["ece"], model):
            d = b["ece"][n]
            L.append(f"| {n} | {fmt(d['gated']['value'])} [{fmt(d['gated']['lo'])}, {fmt(d['gated']['hi'])}] | {fmt(d['full']['value'])} [{fmt(d['full']['lo'])}, {fmt(d['full']['hi'])}] |")
        L.append("")

    if issues:
        st = issues["set"]
        L += [f"## Issue reports, {issues['split']} split (7.4)", "",
              f"{st['rows']} candidates ({st['positives']} secrets) in {st['distinct_texts']} texts; {st['several_occurrences']} occur several times in their text (the first occurrence is the candidate), "
              f"{st['not_found']} are not found in it.", "",
              "| | precision | recall | F1 | false alarms | ECE | end to end: flags | precision over all flags | F1 | precision over flags on a candidate | F1 |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for n in ordered(issues["systems"], model):
            s = issues["systems"][n]
            c, e = s["candidate"], s.get("end_to_end") or {}
            L.append(f"| {n} | {fmt(c['precision'])} | {fmt(c['recall'])} | {fmt(c['f1'])} | {fmt(c['false_alarm_rate'])} | {fmt(c.get('ece'))} | {e.get('flags', '—')} | "
                     f"{fmt(e.get('precision'))} | {fmt(e.get('f1'))} | {fmt(e.get('precision_over_flags_on_a_candidate'))} | {fmt(e.get('f1_over_flags_on_a_candidate'))} |")
        L.append("")
        if issues.get("obfuscation_sensitivity"):
            o = st.get("obfuscation", {})
            left = sum(o.get("left_as_published", {}).values()) if o else None
            L += ["| obfuscation sensitivity (candidate F1) | values as published | through CredData's `get_obfuscated_value` | difference |", "|---|---:|---:|---:|"]
            for n in ordered(issues["obfuscation_sensitivity"], model):
                d = issues["obfuscation_sensitivity"][n]
                L.append(f"| {n} | {fmt(d['f1_published'])} | {fmt(d['f1_obfuscated'])} | {d['difference']:+.4f} |")
            L += ["", f"{o.get('positive_values', '—')} positive candidate strings were passed through the function; {left} stay as published (the function failed on them)." if o else "", ""]

    if clean:
        L += ["## Clean real text (7.7)", "", f"{clean['corpus']['files']} files (list sha256 `{clean['corpus']['list_sha256']}`).", ""]
        for key, r in clean["readings"].items():
            L += [f"| {r['how']}" + (" (the gate's reading)" if key == clean.get("gate_reading") else "") + " | files with an accepted candidate | share | accepted |", "|---|---:|---:|---:|"]
            for n in ordered(r["systems"], model):
                s = r["systems"][n]
                L.append(f"| {n} | {s['files_with_an_accepted_flag']} | {100 * s['share']:.2f}% | {s['accepted_flags']} |")
            L.append("")

    if latency:
        L += ["## Latency and size (7.5)", "",
              f"{latency['model']}: {latency['bytes']:,} bytes, {latency.get('n_params') or 0:,} parameters, tau* {fmt(latency.get('tau'))}. Native, {latency['candidates']} candidates: "
              f"median {latency['median_us_full_depth']:.0f} µs at full depth, {latency['median_us_first_exit']:.0f} µs at the first exit; "
              f"early-exit rate at the stored thresholds {fmt(latency.get('early_exit_rate_at_stored_thresholds'), 3)}.", ""]
        if latency.get("exits"):
            L += ["| exit | accuracy (real question) | ECE |", "|---:|---:|---:|"] + [f"| {e['exit']} | {fmt(e['accuracy'])} | {fmt(e['ece'])} |" for e in latency["exits"]] + [""]

    L += ["## Gate", ""]
    def two(name, text):
        for part, label in PARTS:
            p = g[name]["parts"].get(part)
            if p:
                full = f"; full universe {fmt(p['model_full_universe'])} against {fmt(p['reference_full_universe'])}"
                L.append(f"- {name}, {label}: {'holds' if p['holds'] else 'fails'} ({model} {p['model']:.4f}, {text(p)}{full})")
        for why in g[name]["why"]:
            if "no result" in why:
                L.append(f"- {name}: fails ({why})")

    two("G1", lambda p: f"best rule scanner {p['reference']} {p['reference_f1']:.4f} + {G1_MARGIN} = {p['reference_f1'] + G1_MARGIN:.4f}")
    two("G2", lambda p: f"{p['reference']} {p['reference_f1']:.4f} - {G2_MARGIN} = {p['reference_f1'] - G2_MARGIN:.4f}")
    two("G3", lambda p: f"{p['reference']} {p['reference_f1']:.4f}")
    simple = {"G4": lambda x: f"issue-report candidate F1 {x['model']:.4f}, Gitleaks {x['gitleaks']:.4f} + {G4_MARGIN} = {x['gitleaks'] + G4_MARGIN:.4f}",
              "G5": lambda x: f"ECE {x['ece']:.4f} on the gated universe, at most {MAX_ECE}; full universe {fmt(x.get('ece_full_universe'))}",
              "G6": lambda x: f"native median {x['median_us']:.0f} µs per candidate, at most {MAX_LATENCY_US:.0f}",
              "G7": lambda x: f"{x['files']} of {x['of']} clean files with an accepted candidate, {100 * x['share']:.2f}%, at most {100 * MAX_CLEAN_SHARE:.1f}%"}
    keys = {"G4": "model", "G5": "ece", "G6": "median_us", "G7": "share"}
    for name in ("G4", "G5", "G6", "G7"):
        x = g[name]
        L.append(f"- {name}: " + (("holds" if x["holds"] else "fails") + f" ({simple[name](x)})" if keys[name] in x else "fails (" + "; ".join(x["why"]) + ")"))
    if goal is not None:
        L.append(f"- announced goal, G2 at -{G2_GOAL}: {'met' if goal else 'not met'}")
    ok = all(x["holds"] for x in g.values())
    if rehearsal:
        verdict = f"REHEARSAL on {split}: the gate {'would hold' if ok else 'would not hold'} on these numbers; nothing is decided before the test"
    else:
        verdict = "PASS: publish s6-secrets-verify-pico" if ok else "FAIL: not published"
    L.append(f"- gate: {verdict}")
    return "\n".join(L) + "\n", ok, verdict


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--scores", required=True)
    ap.add_argument("--issues")
    ap.add_argument("--clean")
    ap.add_argument("--latency")
    ap.add_argument("--model", default="model", help="the system in the model's seat (default: model)")
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    scores = maybe(a.scores)
    if scores is None:
        print(f"gate: no {a.scores}")
        return 1
    issues, clean, latency = maybe(a.issues), maybe(a.clean), maybe(a.latency)
    g, goal = gates(scores, issues, clean, latency, a.model)
    md, ok, verdict = results(scores, issues, clean, latency, a.model, g, goal)
    os.makedirs(a.out_dir, exist_ok=True)
    with open(os.path.join(a.out_dir, "results.md"), "w") as f:
        f.write(md)
    with open(os.path.join(a.out_dir, "gate.json"), "w") as f:
        json.dump({"experiment": "s6-secrets v2 (docs/s6-secrets2-prereg.md, section 8)", "split": scores["split"], "model": a.model,
                   "is_the_pre_registered_gate": scores["split"] == "test" and a.model == "model", "universe_of_G1_G2_G3_G5": "gated",
                   "uuid_choice": UUID_SENTENCE, "thresholds": {"G1": G1_MARGIN, "G2": G2_MARGIN, "G2_goal": G2_GOAL, "G4": G4_MARGIN, "G5": MAX_ECE,
                                                                 "G6_us": MAX_LATENCY_US, "G7": MAX_CLEAN_SHARE},
                   "gates": g, "announced_goal_met": goal, "all_hold": ok, "verdict": verdict}, f, indent=1)
        f.write("\n")
    print(md)
    return 0 if ok and scores["split"] == "test" else 1


if __name__ == "__main__":
    sys.exit(main())
