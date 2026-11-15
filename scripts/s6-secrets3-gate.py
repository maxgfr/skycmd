#!/usr/bin/env python3
"""Gate and results of s6-secrets v3: the union of secretgate's default engine and
s6-secrets-verify-pico (docs/s6-secrets3-prereg.md; every definition it inherits is v2's,
docs/s6-secrets2-prereg.md sections 7 and 8 and its amendments).

Reads what scripts/secrets2/union.py score writes in the data folder (numbers only):
  scores.json   every v2 system and the three unions on the CredData split (reference: union)
  issues.json   the issue-report split, with the unions
  clean.json    the clean-text corpus (census reading), with the unions
and the v2 gate of the same split (G5 and G6 are the model's, copied from it), and tau3.json.

The seven gates of v2, read on the union (`union`: default engine or the model at tau3):
  G1  line F1 of the union >= the best rule scanner's (Gitleaks, Betterleaks, secretgate
      default; chosen separately on each reading) + 0.15
  G2  line F1 of the union >= cbx-17M's - 0.10
  G3  line F1 of the union > that of `union-lr` (default engine or the size-matched logistic
      regression at its own union threshold)
  G4  issue-report test, candidate F1 of the union >= Gitleaks' + 0.08
  G5  ECE of the model <= 0.05 (copied from v2)
  G6  native median latency of the model <= 3 ms per candidate (copied from v2)
  G7  clean real text: files flagged by the union <= 1.0%
G1 to G3 hold twice (every repository, and without the largest), on the gated universe.

Usage: s6-secrets3-gate.py --scores S --issues I --clean C --v2-gate G --tau T --out-dir DIR
Writes DIR/results.md and DIR/gate.json; exit 0 when the gate passes on the test, 1 otherwise.
"""
import argparse
import json
import os
import subprocess
import sys

EPS = 1e-9
G1_MARGIN, G2_MARGIN, G4_MARGIN = 0.15, 0.10, 0.08
MAX_ECE, MAX_LATENCY_US, MAX_CLEAN_SHARE = 0.05, 3000.0, 0.010
RULE_SCANNERS = ("gitleaks", "betterleaks", "secretgate-default")
UNION, UNION_LR, UNION_V2, MODEL, CBX, GITLEAKS = "union", "union-lr", "union-tau-v2", "model", "cbx-17m", "gitleaks"
ORDER = (UNION, UNION_LR, UNION_V2, MODEL, "lr", "secretgate-default", "generator", "gitleaks", "betterleaks", "detect-secrets",
         "credsweeper-ml", "credsweeper-noml", "cbx-17m", "potion-lr")
PARTS = (("all", "every repository"), ("without_largest", "without the largest repository"))
UUID_SENTENCE = ("Gated universe: the rows whose CredData category is exactly `UUID` are removed, for every system (amendment 2 of docs/s6-secrets2-prereg.md). "
                 "That choice was made after reading the training and validation rows and before any test file was read. Full universe: every meta row.")
SECOND_LOOK = ("Second look: v3 reads test files that were read once for v2 (docs/s6-secrets2-prereg.md, Result of 15/11/2026), "
               "and its system (the union of the default engine and the model, with tau3) was defined after that reading.")
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)


def load(path):
    with open(path) as f:
        return json.load(f)


def fmt(x, digits=4):
    return "—" if x is None else f"{x:.{digits}f}"


def ordered(names):
    known = [n for n in ORDER if n in names]
    return known + sorted(n for n in names if n not in known)


def notes():
    return [UUID_SENTENCE, "", SECOND_LOOK, ""]


def line_f1(scores, system, part, universe="gated"):
    s = scores["systems"].get(system)
    return None if s is None else s["line"][universe][part]["f1"]


def gates(scores, issues, clean, v2):
    out = {}

    def twice(name, reference, test, describe):
        g = {"holds": True, "why": [], "parts": {}}
        for part, label in PARTS:
            m = line_f1(scores, UNION, part)
            refs = reference(part)
            if m is None or refs is None:
                g["holds"] = False
                g["why"].append(f"{label}: no result of " + (UNION if m is None else "the reference"))
                continue
            ref_name, ref = refs
            ok = test(m, ref)
            g["parts"][part] = {"union": m, "reference": ref_name, "reference_f1": ref, "holds": ok,
                                "union_full_universe": line_f1(scores, UNION, part, "full"), "reference_full_universe": line_f1(scores, ref_name, part, "full")}
            if not ok:
                g["holds"] = False
                g["why"].append(f"{label}: {m:.4f} {describe(ref_name, ref)}")
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
    twice("G3", one(UNION_LR), lambda m, r: m > r, lambda n, r: f"<= {n} {r:.4f}")

    g4 = {"holds": False, "why": []}
    iu = issues["systems"].get(UNION) if issues else None
    ig = issues["systems"].get(GITLEAKS) if issues else None
    if iu is None or ig is None:
        g4["why"].append("no issue-report result of " + (UNION if iu is None else "Gitleaks"))
    else:
        m, r = iu["candidate"]["f1"], ig["candidate"]["f1"]
        g4.update({"union": m, "gitleaks": r, "holds": m >= r + G4_MARGIN - EPS})
        if not g4["holds"]:
            g4["why"].append(f"{m:.4f} < Gitleaks {r:.4f} + {G4_MARGIN}")
    out["G4"] = g4

    for name in ("G5", "G6"):
        g = dict(v2["gates"][name])
        g["copied_from_v2"] = True
        out[name] = g

    g7 = {"holds": False, "why": []}
    cu = clean["readings"][clean.get("gate_reading", "flags")]["systems"].get(UNION) if clean else None
    if cu is None:
        g7["why"].append("no clean-text result of the union")
    else:
        g7.update({"files": cu["files_with_an_accepted_flag"], "of": cu["files"], "share": cu["share"], "holds": cu["share"] <= MAX_CLEAN_SHARE + EPS})
        if not g7["holds"]:
            g7["why"].append(f"{100 * cu['share']:.2f}% of the files > {100 * MAX_CLEAN_SHARE:.1f}%")
    out["G7"] = g7
    return out


def tau_commit(tau_path):
    rel = os.path.relpath(os.path.abspath(tau_path), REPO)
    r = subprocess.run(["git", "-C", REPO, "log", "-1", "--format=%H %cI", "--", rel], capture_output=True, text=True)
    parts = r.stdout.split()
    return {"commit": parts[0], "committed_at": parts[1]} if len(parts) == 2 else None


def test_read():
    p = os.path.join(REPO, "runs", "s6-secrets3", "TEST-READ")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return f.read().strip()


def results(scores, issues, clean, v2, tau, commit, read, g):
    split = scores["split"]
    systems = scores["systems"]
    names = ordered(systems)
    th = scores["thresholds"]
    rehearsal = split != "test"
    L = [f"# s6-secrets v3 (secretgate's default engine and s6-secrets-verify-pico together): results on the {split} split (docs/s6-secrets3-prereg.md)", ""]
    if rehearsal:
        L += [f"**Rehearsal.** These are the {split} repositories, on which tau3 was chosen: no number below is a result of the pre-registration, and the gate at the end is the test's computation run on {split}.", ""]
    L += [f"**{SECOND_LOOK}** The reading of v2 does not change: it failed its gate on G1 and stays failed.", ""]
    L += ["**The systems.** `union`: a line is flagged when secretgate's default engine (secretgate 4e93cfa, filters on) flags it, or when the model of v2 "
          f"(the evaluated file, unchanged) accepts a candidate of the frozen generator on it with P(real) >= tau3 = {th[UNION]}. `union-lr`: the same with the size-matched "
          f"logistic regression in the model's seat, at its own union threshold {th[UNION_LR]} (G3). `union-tau-v2`: the union at v2's tau* {fmt(th[UNION_V2])} (reported). "
          "`model` is the model alone at tau*, v2's system. On the issue reports a dataset candidate is flagged when a default finding overlaps it or the verifier accepts it; "
          "on the clean text a file counts when either flags it. No scanner, model or baseline was rerun: every flag is the v2 evaluation's.", ""]
    when = f", committed at {commit['committed_at']} in `{commit['commit'][:7]}`" if commit else ""
    L += [f"tau3 and the regression's union threshold were chosen on the 16 validation repositories by the rule of section 1 (`runs/s6-secrets3/tau3.json`{when}); "
          f"validation line F1 of the union at tau3, gated: {fmt(tau['validation']['union']['gated']['all']['f1'])}."
          + (f" Test read for v3: {read}." if read and not rehearsal else ""), ""]
    big = scores["largest_repository"]
    L += [f"{scores['repositories']} repositories, {scores['files']} files; T rows: {scores['t_rows']['gated']} (gated universe), {scores['t_rows']['full']} (full). "
          f"Largest repository: `{big['repo_id']}`, {big['t_rows']['gated']} gated T rows" + (" (the split's own largest, a stand-in for the test's)" if big.get("rehearsal_stand_in") else "") + ".", ""]

    L += ["## CredData, line level (7.1)", ""] + notes()
    for u in ("gated", "full"):
        L += [f"| {u} universe | precision | recall | F1 | false alarms | F1 without the largest | macro F1 (≥ 20 T rows) |", "|---|---:|---:|---:|---:|---:|---:|"]
        for n in names:
            x = systems[n]["line"][u]
            a = x["all"]
            L.append(f"| {n} | {fmt(a['precision'])} | {fmt(a['recall'])} | {fmt(a['f1'])} | {fmt(a['false_alarm_rate'])} | {fmt(x['without_largest']['f1'])} | "
                     f"{fmt(x['macro_f1']['value'])} ({x['macro_f1']['repositories']}) |")
        L.append("")
    L += ["CredSweeper's ML was trained on a random 85% of all CredData lines: its CredData numbers are in-sample and enter no gate (section 6 of v2).", ""]

    L += ["## Lines the union adds over the model alone (reported, section 4)", ""] + notes()
    L += ["Line units flagged by the union and not by the model alone at tau* (added), and the reverse (dropped), by label: T a positive line, X a negative line with an X row, F the other negative lines.", "",
          "| universe | union | threshold | added T | added X | added F | dropped T | dropped X | dropped F | lines the default engine flags | line units |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for u in ("gated", "full"):
        for n, a in scores["added_over_the_model_alone"][u].items():
            L.append(f"| {u} | {n} | {fmt(a['threshold'])} | {a['added']['T']} | {a['added']['X']} | {a['added']['F']} | {a['dropped']['T']} | {a['dropped']['X']} | {a['dropped']['F']} | "
                     f"{a['default_flags_line']} | {a['units']} |")
    L.append("")

    L += ["## Unlabelled-line alarms (7.3), versions and CPU time", ""] + notes()
    L += ["| | version | unit | unlabelled-line alarms: flags | lines | flags per 1,000 files | CPU s |", "|---|---|---|---:|---:|---:|---:|"]
    for n in names:
        s = systems[n]
        L.append(f"| {n} | {s.get('version') or '—'} | {s.get('unit') or '—'} | {s['unlabelled']['flags']} | {s['unlabelled']['lines']} | {fmt(s['unlabelled']['per_1000_files'], 1)} | {fmt(s.get('cpu_seconds'), 1)} |")
    L += ["", "A union's flags are the default engine's plus the verifier's accepted candidates: a line both flag counts twice in the flags column, once in the lines column.", ""]

    cands = [n for n in names if "candidate" in systems[n]]
    if cands:
        L += ["## CredData, candidate level (7.2): the scored candidates of v2", ""] + notes()
        L += ["The union is a line system (the default engine gives no score), so it has no candidate-level row; the model's and the baselines' rows are v2's.", ""]
        for u in ("gated", "full"):
            L += [f"| {u} universe | candidates | positives | threshold | precision | recall | F1 | PR-AUC | recall at 1% false alarms | ECE | ECE on P(real) |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
            for n in cands:
                c = systems[n]["candidate"][u]
                L.append(f"| {n} | {c['n']} | {c['positives']} | {fmt(systems[n].get('threshold'))} | {fmt(c['precision'])} | {fmt(c['recall'])} | {fmt(c['f1'])} | "
                         f"{fmt(c.get('pr_auc'))} | {fmt(c.get('recall_at_1pct_false_alarms'))} | {fmt(c.get('ece'))} | {fmt(c.get('ece_positive'))} |")
            L.append("")

    spans = [n for n in names if "span" in systems[n]]
    if spans:
        L += ["## CredData, span level (section 9 of v2, overlap with a T value)", ""] + notes()
        L += ["| | gated recall | gated precision | gated F1 | full recall | full precision | full F1 | flags: T / X / F / no row / unlabelled |", "|---|---:|---:|---:|---:|---:|---:|---|"]
        for n in spans:
            a, b, lab = systems[n]["span"]["gated"], systems[n]["span"]["full"], systems[n]["labels"]["full"]
            L.append(f"| {n} | {fmt(a['recall'])} | {fmt(a['precision'])} | {fmt(a['f1'])} | {fmt(b['recall'])} | {fmt(b['precision'])} | {fmt(b['f1'])} | "
                     f"{lab['T']} / {lab['X']} / {lab['F']} / {lab['D']} / {lab['U']} |")
        L += ["", "The flag counts of a scored system are of all its candidates; those of a union, of its flags: the default engine's and the verifier's accepted candidates.", ""]

    L += ["## By CredData category and by path scope (section 9 of v2), gated universe, line F1 (recall)", ""] + notes()
    cats = list(systems[names[0]]["line"]["gated"]["by_category"])
    scopes = ["test", "src", "other"]
    L += ["| | " + " | ".join(cats + [f"scope {s}" for s in scopes]) + " |", "|---|" + "---:|" * (len(cats) + len(scopes))]
    for n in names:
        x = systems[n]["line"]["gated"]
        cells = [f"{fmt(x['by_category'][c]['f1'], 3)} ({fmt(x['by_category'][c]['recall'], 3)})" for c in cats]
        cells += [f"{fmt(x['by_scope'][s]['f1'], 3)} ({fmt(x['by_scope'][s]['recall'], 3)})" for s in scopes]
        L.append(f"| {n} | " + " | ".join(cells) + " |")
    x = systems[names[0]]["line"]["gated"]
    count = lambda v: f"{v['tp'] + v['fn']} / {v['fp'] + v['tn']}"
    L += ["| positive / negative lines | " + " | ".join([count(x["by_category"][c]) for c in cats] + [count(x["by_scope"][s]) for s in scopes]) + " |", ""]

    b = scores.get("bootstrap") or {}
    if b.get("f1_difference"):
        L += [f"## Uncertainty (7.6): paired bootstrap over repositories, {b['resamples']} resamples, 95%", ""] + notes()
        L += [f"| line F1 of {b['reference']} minus | gated | gated, without the largest | full | full, without the largest |", "|---|---|---|---|---|"]
        cell = lambda d: f"{d['value']:+.4f} [{fmt(d['lo'])}, {fmt(d['hi'])}]"
        for n in ordered(b["f1_difference"]):
            d = b["f1_difference"][n]
            L.append(f"| {n} | {cell(d['gated']['all'])} | {cell(d['gated']['without_largest'])} | {cell(d['full']['all'])} | {cell(d['full']['without_largest'])} |")
        L.append("")
    if b.get("ece"):
        L += ["| ECE (scored candidates of v2) | gated | full |", "|---|---|---|"]
        for n in ordered(b["ece"]):
            d = b["ece"][n]
            L.append(f"| {n} | {fmt(d['gated']['value'])} [{fmt(d['gated']['lo'])}, {fmt(d['gated']['hi'])}] | {fmt(d['full']['value'])} [{fmt(d['full']['lo'])}, {fmt(d['full']['hi'])}] |")
        L.append("")

    if issues:
        st = issues["set"]
        L += [f"## Issue reports, {issues['split']} split (7.4)", ""] + notes()
        L += [f"{st['rows']} candidates ({st['positives']} secrets) in {st['distinct_texts']} texts; {st['several_occurrences']} occur several times in their text (the first occurrence is the candidate), "
              f"{st['not_found']} are not found in it.", "",
              "| | precision | recall | F1 | false alarms | ECE | end to end: flags | precision over all flags | F1 | precision over flags on a candidate | F1 |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for n in ordered(issues["systems"]):
            s = issues["systems"][n]
            c, e = s["candidate"], s.get("end_to_end") or {}
            L.append(f"| {n} | {fmt(c['precision'])} | {fmt(c['recall'])} | {fmt(c['f1'])} | {fmt(c['false_alarm_rate'])} | {fmt(c.get('ece'))} | {e.get('flags', '—')} | "
                     f"{fmt(e.get('precision'))} | {fmt(e.get('f1'))} | {fmt(e.get('precision_over_flags_on_a_candidate'))} | {fmt(e.get('f1_over_flags_on_a_candidate'))} |")
        L += ["", "`<union>-e2e`: the default engine's findings on the texts and the verifier's accepted candidates of the generator (end to end); its candidate columns read the candidates by overlap.", ""]
        L += ["| who flags the candidates, values as published | default only: positive / negative | verifier only | both | neither |", "|---|---|---|---|---|"]
        for n in (UNION, UNION_LR, UNION_V2):
            p = issues["systems"][n]["flagged_by"]
            L.append(f"| {n} | " + " | ".join(f"{p[k]['positive']} / {p[k]['negative']}" for k in ("default only", "verifier only", "both", "neither")) + " |")
        L.append("")
        if issues.get("obfuscation_sensitivity"):
            L += ["| obfuscation sensitivity (candidate F1) | values as published | through CredData's `get_obfuscated_value` | difference |", "|---|---:|---:|---:|"]
            for n in ordered(issues["obfuscation_sensitivity"]):
                d = issues["obfuscation_sensitivity"][n]
                L.append(f"| {n} | {fmt(d['f1_published'])} | {fmt(d['f1_obfuscated'])} | {d['difference']:+.4f} |")
            L.append("")

    if clean:
        L += ["## Clean real text (7.7)", ""] + notes()
        L += [f"{clean['corpus']['files']} files (list sha256 `{clean['corpus']['list_sha256']}`), read as the census read them (no source path, no pragma for the default engine): G7's reading.", "",
              "| | files with an accepted flag | share | default engine only | verifier only | both | accepted flags |", "|---|---:|---:|---:|---:|---:|---:|"]
        for n in ordered(clean["readings"]["flags"]["systems"]):
            s = clean["readings"]["flags"]["systems"][n]
            if "files_both" in s:
                L.append(f"| {n} | {s['files_with_an_accepted_flag']} | {100 * s['share']:.2f}% | {s['files_default_only']} | {s['files_verifier_only']} | {s['files_both']} | "
                         f"{s['default_flags']} + {s['verifier_accepted_flags']} |")
            else:
                L.append(f"| {n} | {s['files_with_an_accepted_flag']} | {100 * s['share']:.2f}% | — | — | — | {s['accepted_flags']} |")
        L.append("")

    L += ["## Gate (docs/s6-secrets3-prereg.md section 3)", ""] + notes()

    def two(name, text):
        for part, label in PARTS:
            p = g[name]["parts"].get(part)
            if p:
                L.append(f"- {name}, {label}: {'holds' if p['holds'] else 'fails'} (union {p['union']:.4f}, {text(p)}; "
                         f"full universe {fmt(p['union_full_universe'])} against {fmt(p['reference_full_universe'])})")
        for why in g[name]["why"]:
            if "no result" in why:
                L.append(f"- {name}: fails ({why})")

    two("G1", lambda p: f"best rule scanner {p['reference']} {p['reference_f1']:.4f} + {G1_MARGIN} = {p['reference_f1'] + G1_MARGIN:.4f}")
    two("G2", lambda p: f"{p['reference']} {p['reference_f1']:.4f} - {G2_MARGIN} = {p['reference_f1'] - G2_MARGIN:.4f}")
    two("G3", lambda p: f"{p['reference']} {p['reference_f1']:.4f}")
    x = g["G4"]
    L.append("- G4: " + (("holds" if x["holds"] else "fails") + f" (issue-report candidate F1 of the union {x['union']:.4f}, Gitleaks {x['gitleaks']:.4f} + {G4_MARGIN} = {x['gitleaks'] + G4_MARGIN:.4f})"
                         if "union" in x else "fails (" + "; ".join(x["why"]) + ")"))
    x = g["G5"]
    L.append(f"- G5: {'holds' if x['holds'] else 'fails'} (ECE of the model {fmt(x.get('ece'))} on the gated universe, at most {MAX_ECE}; copied from v2's {split} gate)")
    x = g["G6"]
    L.append(f"- G6: {'holds' if x['holds'] else 'fails'} (native median {fmt(x.get('median_us'), 0)} µs per candidate, at most {MAX_LATENCY_US:.0f}; copied from v2's {split} gate)")
    x = g["G7"]
    L.append("- G7: " + (("holds" if x["holds"] else "fails") + f" ({x['files']} of {x['of']} clean files flagged by the union, {100 * x['share']:.2f}%, at most {100 * MAX_CLEAN_SHARE:.1f}%)"
                         if "files" in x else "fails (" + "; ".join(x["why"]) + ")"))
    ok = all(v["holds"] for v in g.values())
    if rehearsal:
        verdict = f"REHEARSAL on {split}: the gate {'would hold' if ok else 'would not hold'} on these numbers; nothing is decided on {split}"
    elif ok:
        verdict = "PASS on a second look, system redefined after the first reading"
    else:
        failed = [k for k, v in g.items() if not v["holds"]]
        verdict = f"FAIL on a second look ({', '.join(failed)})"
        if "G1" in failed or "G4" in failed:
            verdict += ": the work on this model ends, no third reading (section 3)"
    L.append(f"- gate: {verdict}")
    return "\n".join(L) + "\n", ok, verdict


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    for k in ("scores", "issues", "clean", "v2-gate", "tau", "out-dir"):
        ap.add_argument(f"--{k}", required=True)
    a = ap.parse_args()
    scores, issues, clean, v2, tau = load(a.scores), load(a.issues), load(a.clean), load(a.v2_gate), load(a.tau)
    if v2.get("split") != scores["split"] or v2.get("model") != MODEL:
        print("gate: the v2 gate is not the model's on the same split")
        return 1
    for n, t in (("union", tau["tau3"]), ("union-lr", tau["tau_union_lr"]), ("union-tau-v2", tau["tau_v2"])):
        if scores["thresholds"][n] != t or issues["thresholds"][n] != t:
            print(f"gate: the threshold of {n} is not the one of {a.tau}")
            return 1
    commit, read = tau_commit(a.tau), test_read()
    g = gates(scores, issues, clean, v2)
    md, ok, verdict = results(scores, issues, clean, v2, tau, commit, read, g)
    os.makedirs(a.out_dir, exist_ok=True)
    with open(os.path.join(a.out_dir, "results.md"), "w") as f:
        f.write(md)
    with open(os.path.join(a.out_dir, "gate.json"), "w") as f:
        json.dump({"experiment": "s6-secrets v3 (docs/s6-secrets3-prereg.md, section 3)", "split": scores["split"], "system": UNION,
                   "is_the_pre_registered_gate": scores["split"] == "test", "second_look": SECOND_LOOK, "uuid_choice": UUID_SENTENCE,
                   "universe_of_G1_G2_G3": "gated", "thresholds_of_the_systems": scores["thresholds"], "tau3_json": commit, "test_read": read if scores["split"] == "test" else None,
                   "thresholds": {"G1": G1_MARGIN, "G2": G2_MARGIN, "G4": G4_MARGIN, "G5": MAX_ECE, "G6_us": MAX_LATENCY_US, "G7": MAX_CLEAN_SHARE},
                   "gates": g, "all_hold": ok, "verdict": verdict}, f, indent=1)
        f.write("\n")
    print(md)
    return 0 if ok and scores["split"] == "test" else 1


if __name__ == "__main__":
    sys.exit(main())
