#!/usr/bin/env python3
"""s6-secrets v3 (docs/s6-secrets3-prereg.md): the union of secretgate's default engine and a
verifier, as a system, computed from the flags the v2 evaluation wrote. Nothing is rerun: no
scanner, no model, no baseline; every definition not written here is v2's (scorer.py, issues.py,
clean_text.py, docs/s6-secrets2-prereg.md sections 7 and 8 and its amendments).

The union at a threshold tau:
  CredData       a line is flagged when the default engine flags it (one of its spans covers
                 the line) or when the verifier accepts a candidate on it, P(real) >= tau. As
                 flags: every flag of the default engine, plus every candidate of the verifier
                 whose score is >= tau, without its score: one unscored system of unit `line`,
                 which scorer.py reads unchanged.
  issue reports  a dataset candidate is flagged when a finding of the default engine overlaps
                 it in its text (as issues.py reads a scanner), or when the verifier accepts it
                 (the score of its row is >= tau).
  clean text     a file counts when the default engine flags it (the census reading, the one G7
                 reads) or when the verifier accepts a candidate in it.
The verifier is the model of v2 (`model`), or the size-matched logistic regression (`lr`) in its
seat (G3). Systems: `union` (default + model at tau3), `union-lr` (default + lr at its own union
threshold), `union-tau-v2` (default + model at v2's tau*, reported).

  union.py tau [--out runs/s6-secrets3/tau3.json]
      the validation split only. tau3: the threshold that maximises the union's line-level F1
      on the gated universe (every validation repository), among the P(real) observed on the
      validation candidates (the model's flags on creddata:validation); of equal F1 values the
      highest. The regression's union threshold by the same rule, among its own scores. Writes
      the thresholds, the validation precision, recall and F1 of each union at its threshold
      and of the model alone at tau*, and the sha256 of every input, the test's included.
  union.py score --split validation|test [--tau runs/s6-secrets3/tau3.json] [--out DIR]
      every system of v2 and the three unions through the scorer (CredData, reference `union`
      for the bootstrap), the issue-report candidates, the clean text; the lines the union adds
      over the model alone and their labels. Writes DIR/scores.json, DIR/issues.json,
      DIR/clean.json, DIR/added.json (default DIR: <data>/eval/<split>/v3).
      The test is refused unless tau3.json is committed, unchanged, and its hashes of the
      inputs hold; the first test run writes runs/s6-secrets3/TEST-READ and a second one is
      refused (docs/s6-secrets3-prereg.md section 5: no third reading, nothing rerun).
"""
import argparse
import datetime
import os
import subprocess
import sys
from fractions import Fraction

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402
import issues  # noqa: E402
import scorer  # noqa: E402

DEFAULT = "secretgate-default"
UNIONS = {"union": "model", "union-lr": "lr", "union-tau-v2": "model"}
RUNS = os.path.join(common.REPO, "runs", "s6-secrets3")
TAU_JSON = os.path.join(RUNS, "tau3.json")
TEST_READ = os.path.join(RUNS, "TEST-READ")
EXPERIMENT = "s6-secrets v3 (docs/s6-secrets3-prereg.md)"


# ---------------------------------------------------------------- the union as flags

def lines_of_flag(x):
    return range(x["line"], x.get("end_line", x["line"]) + 1)


def accepted(flags, tau):
    return [x for x in flags if x["score"] >= tau]


def union_flags(default_flags, verifier_flags, tau):
    """The union as unscored flags: every flag of the default engine, then every candidate of
    the verifier with P(real) >= tau (its score dropped, marked `by: verifier`)."""
    out = [{**x, "by": "default"} for x in default_flags]
    for x in accepted(verifier_flags, tau):
        y = {k: v for k, v in x.items() if k != "score"}
        y["by"] = "verifier"
        out.append(y)
    return out


def union_run(name, default_run, verifier_name, verifier_run, tau):
    if default_run.get("corpus") != verifier_run.get("corpus"):
        raise common.Refused(f"{name}: the default engine's flags are of {default_run.get('corpus')}, the verifier's of {verifier_run.get('corpus')}")
    if default_run.get("scored") or not verifier_run.get("scored"):
        raise common.Refused(f"{name}: the default engine has no score and the verifier has one")
    return {"format": common.FORMAT, "system": name, "corpus": default_run["corpus"], "unit": "line", "scored": False,
            "version": f"{default_run.get('version')} or {verifier_name} ({verifier_run.get('version')}) at P(real) >= {tau}",
            "union": {"default": DEFAULT, "verifier": verifier_name, "threshold": tau}}


# ---------------------------------------------------------------- line units and the threshold

def unit_table(lab, default_flags, verifier_flags, universe):
    """One entry per line unit of `universe`, as scorer.score_system counts it: (repo, label,
    flagged by the default engine, the highest verifier score among the candidates that cover
    the unit's line or None). label: T for a positive unit, else X when one of its rows is X,
    else F."""
    def by_file(flags):
        out = {}
        for x in flags:
            out.setdefault(x["file"], []).append(x)
        return out

    d, v = by_file(default_flags), by_file(verifier_flags)
    out = []
    for path in sorted(lab.files):
        units, _, _, rows = lab.view(path, universe)
        if not units:
            continue
        dl = set()
        for x in d.get(path, []):
            dl.update(lines_of_flag(x))
        best = {}
        for x in v.get(path, []):
            for line in lines_of_flag(x):
                if line not in best or x["score"] > best[line]:
                    best[line] = x["score"]
        truths = {}
        for r in rows:
            truths.setdefault(r["line_start"], set()).add(r["truth"])
        repo = rows[0]["repo"]
        for start, unit in units.items():
            label = "T" if unit["pos"] else "X" if "X" in truths[start] else "F"
            out.append((repo, label, start in dl, best.get(start)))
    return out


def flagged(entry, tau, with_default=True):
    _, _, d, s = entry
    return (with_default and d) or (s is not None and s >= tau)


def counts_at(units, tau, with_default=True, repos=None):
    tp = fp = fn = tn = 0
    for e in units:
        if repos is not None and e[0] not in repos:
            continue
        f = flagged(e, tau, with_default)
        if e[1] == "T":
            tp, fn = tp + f, fn + (not f)
        else:
            fp, tn = fp + f, tn + (not f)
    return tp, fp, fn, tn


def f1_exact(tp, fp, fn):
    return Fraction(2 * tp, 2 * tp + fp + fn) if tp else Fraction(0)


def choose_threshold(units, thresholds, with_default=True):
    """(tau, (tp, fp, fn, tn)) maximising the line F1 of "the default engine, or a verifier
    score >= tau" over `thresholds`; of equal F1 values (compared exactly) the highest tau."""
    ts = sorted(set(thresholds), reverse=True)
    if not ts:
        raise common.Refused("no threshold to choose from")
    # entries that only the verifier can flag, by score; the rest is fixed
    base_tp = sum(1 for e in units if e[1] == "T" and with_default and e[2])
    base_fp = sum(1 for e in units if e[1] != "T" and with_default and e[2])
    n_pos = sum(1 for e in units if e[1] == "T")
    n_neg = len(units) - n_pos
    free = sorted(((e[3], e[1] == "T") for e in units if not (with_default and e[2]) and e[3] is not None), key=lambda x: -x[0])
    best, best_f, k, tp, fp = None, Fraction(-1), 0, base_tp, base_fp
    for t in ts:  # decreasing: a strictly greater F1 replaces, an equal one keeps the higher tau
        while k < len(free) and free[k][0] >= t:
            tp, fp = tp + free[k][1], fp + (not free[k][1])
            k += 1
        f = f1_exact(tp, fp, n_pos - tp)
        if f > best_f:
            best, best_f = t, f
            counts = (tp, fp, n_pos - tp, n_neg - fp)
    return best, counts


def prf(c):
    return scorer.prf(*c)


# ---------------------------------------------------------------- issue reports and clean text

def issues_union(cands, default_run, default_flags, verifier_run, verifier_flags, tau, lines_of):
    """Candidate level on the issue-report set: a dataset candidate is flagged when a default
    finding overlaps it in its text or the verifier's score of its row is >= tau."""
    if verifier_run.get("judges") != "dataset-candidates":
        raise common.Refused(f"{verifier_run.get('system')}: does not judge the dataset's candidates")
    if default_run.get("scored"):
        raise common.Refused("the default engine has no score")
    spans = {}
    for x in default_flags:
        tid = os.path.basename(x["file"])[: -len(".txt")]
        s = issues.flag_span(lines_of(tid), x)
        if s is not None:
            spans.setdefault(tid, []).append(s)
    by_row = {x["id"]: x for x in verifier_flags}
    c = [0, 0, 0, 0]
    parts = {k: {"positive": 0, "negative": 0} for k in ("default only", "verifier only", "both", "neither")}
    for cand in cands:
        d = "start" in cand and any(a < cand["end"] and cand["start"] < b for a, b in spans.get(cand["text"], []))
        x = by_row.get(cand["row"])
        m = x is not None and x["score"] >= tau
        scorer._add(c, bool(cand["label"]), d or m)
        k = "both" if d and m else "default only" if d else "verifier only" if m else "neither"
        parts[k]["positive" if cand["label"] else "negative"] += 1
    return {"unit": "candidate", "threshold": tau, "candidate": {**scorer._m(c), "candidates": len(cands), "judged": len(by_row)}, "flagged_by": parts}


def clean_union(default_flags, verifier_flags, tau, n_files):
    d = {x["file"] for x in default_flags}
    acc = accepted(verifier_flags, tau)
    v = {x["file"] for x in acc}
    files = d | v
    return {"files_with_an_accepted_flag": len(files), "files": n_files, "share": len(files) / n_files if n_files else None, "threshold": tau,
            "files_default_only": len(d - v), "files_verifier_only": len(v - d), "files_both": len(d & v),
            "default_flags": len(default_flags), "verifier_accepted_flags": len(acc)}


# ---------------------------------------------------------------- inputs and guards

def inputs(split):
    """The flag files a v3 computation reads on a split (the v2 evaluation's), by role."""
    e = common.eval_root()
    cd, iss, obf, cl = (os.path.join(e, split, "flags"), os.path.join(e, "issues", split, "flags"),
                        os.path.join(e, "issues", split, "flags-obfuscated"), os.path.join(e, "clean", "flags"))
    out = {}
    for where, d, names in (("creddata", cd, (DEFAULT, "model", "lr")), ("issues", iss, (DEFAULT, "model", "lr", "model-e2e", "lr-e2e")),
                            ("issues-obfuscated", obf, (DEFAULT, "model", "lr")), ("clean", cl, (DEFAULT, "model", "lr"))):
        for n in names:
            f, r = common.flags_paths(d, n)
            out[f"{where}/{n}"] = {"flags": f, "run": r}
    return out


def hashes(split):
    root = common.data_root()
    return {k: {"flags": os.path.relpath(v["flags"], root), "flags_sha256": common.sha256_file(v["flags"]), "run_sha256": common.sha256_file(v["run"])}
            for k, v in inputs(split).items()}


def git(*a):
    return subprocess.run(["git", "-C", common.REPO, *a], capture_output=True, text=True)


def committed(path):
    """The commit that last changed `path`, if the file is tracked and unchanged; else Refused."""
    rel = os.path.relpath(path, common.REPO)
    if git("ls-files", "--error-unmatch", rel).returncode != 0:
        raise common.Refused(f"{rel} is not committed")
    if git("diff", "--quiet", "HEAD", "--", rel).returncode != 0:
        raise common.Refused(f"{rel} differs from its committed version")
    r = git("log", "-1", "--format=%H %cI", "--", rel)
    commit, when = r.stdout.split()
    return {"commit": commit, "committed_at": when}


def load_tau(path, split):
    tau = common.load_json(path)
    if tau.get("experiment") != EXPERIMENT:
        raise common.Refused(f"{path}: not the tau3 file of {EXPERIMENT}")
    if split == "test":
        tau["_commit"] = committed(path)
        for k, h in tau["inputs"]["test"].items():
            got = common.sha256_file(os.path.join(common.data_root(), h["flags"]))
            if got != h["flags_sha256"]:
                raise common.Refused(f"{h['flags']}: sha256 {got}, recorded in tau3.json {h['flags_sha256']}")
    for k, h in tau["inputs"]["validation"].items():
        got = common.sha256_file(os.path.join(common.data_root(), h["flags"]))
        if got != h["flags_sha256"]:
            raise common.Refused(f"{h['flags']}: sha256 {got}, recorded in tau3.json {h['flags_sha256']}")
    return tau


# ---------------------------------------------------------------- tau3

def tau3(out):
    split = common.open_split("validation")
    files = common.creddata_files(split)
    lab = scorer.load_labelled(split, files)
    d = os.path.join(common.eval_root(), "validation", "flags")
    default_run, default_flags = common.read_flags(d, DEFAULT)
    res = {"experiment": EXPERIMENT, "rule": "docs/s6-secrets3-prereg.md section 1: the threshold that maximises the union's line-level F1 on the gated universe "
           "of the 16 validation repositories, among the probabilities observed on the validation candidates (the verifier's scores on creddata:validation); "
           "of equal values the highest", "written_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
           "thresholds": {}, "validation": {}}
    units = {u: None for u in scorer.UNIVERSES}
    v2_tau = None
    versions = {DEFAULT: default_run.get("version")}
    for name, verifier in (("union", "model"), ("union-lr", "lr")):
        vrun, vflags = common.read_flags(d, verifier)
        if vrun.get("corpus") != "creddata:validation":
            raise common.Refused(f"{verifier}: flags of {vrun.get('corpus')}")
        versions[verifier] = vrun.get("version")
        tables = {u: unit_table(lab, default_flags, vflags, u) for u in scorer.UNIVERSES}
        observed = [x["score"] for x in vflags]
        t, c = choose_threshold(tables["gated"], observed)
        # the same choice through the scorer, which must agree
        res_sc, _ = scorer.score_system(lab, union_run(name, default_run, verifier, vrun, t), union_flags(default_flags, vflags, t))
        if (res_sc["line"]["gated"]["all"]["tp"], res_sc["line"]["gated"]["all"]["fp"], res_sc["line"]["gated"]["all"]["fn"]) != c[:3]:
            raise common.Refused(f"{name}: the threshold sweep and the scorer disagree at {t}")
        others = set(lab.repos) - {lab.largest}
        # the other reading of "the validation candidates": only those that reach a line unit
        t_units, _ = choose_threshold(tables["gated"], [e[3] for e in tables["gated"] if e[3] is not None])
        res["thresholds"][name] = {"verifier": verifier, "threshold": t, "observed_scores": len(observed), "distinct_scores": len(set(observed)),
                                   "same_over_the_scores_on_labelled_lines_only": t_units == t, "verifier_threshold_of_v2": vrun["threshold"]}
        res["validation"][name] = {
            u: {"all": prf(counts_at(tables[u], t)), "without_largest": prf(counts_at(tables[u], t, repos=others))} for u in scorer.UNIVERSES}
        res["validation"][f"{name} at the verifier's v2 threshold"] = {
            u: {"all": prf(counts_at(tables[u], vrun["threshold"])), "without_largest": prf(counts_at(tables[u], vrun["threshold"], repos=others))}
            for u in scorer.UNIVERSES}
        res["validation"][f"{verifier} alone at its v2 threshold"] = {
            u: {"all": prf(counts_at(tables[u], vrun["threshold"], False)), "without_largest": prf(counts_at(tables[u], vrun["threshold"], False, repos=others))}
            for u in scorer.UNIVERSES}
        if verifier == "model":
            v2_tau = vrun["threshold"]
            units = tables
    res["tau3"] = res["thresholds"]["union"]["threshold"]
    res["tau_union_lr"] = res["thresholds"]["union-lr"]["threshold"]
    res["tau_v2"] = v2_tau
    res["largest_validation_repository"] = lab.largest
    res["inputs"] = {"versions": versions, "splits.json_sha256": common.frozen()["splits.json"]["sha256"],
                     "validation": hashes("validation"), "test": hashes("test")}
    res["note"] = ("Chosen on validation only; the test files are hashed here (sha256 of the v2 evaluation's flag files), nothing is computed on them. "
                   "v3 is a second look at test files read once for v2, with the system redefined after that reading.")
    common.write_json(out, res)
    return res


# ---------------------------------------------------------------- score

def added_lines(lab, default_flags, model_flags, tau, tau_v2):
    """Units the union flags and the model alone (at tau*, v2's system) does not, and the reverse,
    by label, in both universes; for the union at tau3 and at tau*."""
    out = {}
    for u in scorer.UNIVERSES:
        t = unit_table(lab, default_flags, model_flags, u)
        out[u] = {}
        for name, ut in (("union", tau), ("union-tau-v2", tau_v2)):
            add = {k: 0 for k in "TXF"}
            drop = {k: 0 for k in "TXF"}
            for e in t:
                a, b = flagged(e, ut), flagged(e, tau_v2, False)
                if a and not b:
                    add[e[1]] += 1
                elif b and not a:
                    drop[e[1]] += 1
            out[u][name] = {"threshold": ut, "added": add, "dropped": drop, "default_flags_line": sum(1 for e in t if e[2]),
                            "units": len(t)}
    return out


def score(split, tau_path, out_dir):
    tau = load_tau(tau_path, split)
    if split == "test":
        if os.path.exists(TEST_READ):
            raise common.Refused(f"{TEST_READ} exists: the v3 test has been read once and is not read again")
        common.test_is_open()
    s = common.open_split(split)
    if split == "test":
        os.makedirs(RUNS, exist_ok=True)
        with open(TEST_READ, "w") as f:
            f.write(datetime.datetime.now().astimezone().isoformat(timespec="seconds") + f" tau3.json at {tau['_commit']['commit']}\n")
    thresholds = {"union": tau["tau3"], "union-lr": tau["tau_union_lr"], "union-tau-v2": tau["tau_v2"]}
    files = common.creddata_files(s)
    lab = scorer.load_labelled(s, files)
    d = os.path.join(common.eval_root(), split, "flags")
    systems = {}
    for n in common.systems_in(d):
        run, flags = common.read_flags(d, n)
        if run.get("corpus") != f"creddata:{split}":
            raise common.Refused(f"{n}: its flags are of {run.get('corpus')}")
        for x in flags:
            s.check(x["file"])
        systems[n] = (run, flags)
    if abs(systems["model"][0]["threshold"] - tau["tau_v2"]) > 1e-12:
        raise common.Refused("the model's threshold is not the one tau3.json records")
    default_run, default_flags = systems[DEFAULT]
    for name, verifier in UNIONS.items():
        vrun, vflags = systems[verifier]
        systems[name] = (union_run(name, default_run, verifier, vrun, thresholds[name]), union_flags(default_flags, vflags, thresholds[name]))
    ordered = {k: systems[k] for k in (*UNIONS, *(n for n in systems if n not in UNIONS))}
    sc = {"experiment": EXPERIMENT, "split": split, "thresholds": thresholds, **scorer.score_all(lab, ordered, "union")}
    sc["added_over_the_model_alone"] = added_lines(lab, default_flags, systems["model"][1], thresholds["union"], thresholds["union-tau-v2"])
    common.write_json(os.path.join(out_dir, "scores.json"), sc)

    # issue reports: v2's scoring of every system, then the unions
    ib = common.issues_dir(split)
    idir, odir = os.path.join(ib, "flags"), os.path.join(ib, "flags-obfuscated")
    iss = issues.score(split, idir, odir)
    cands = issues.read_candidates(ib)
    for key, folder, obfuscated in (("systems", idir, False), ("systems_on_obfuscated_values", odir, True)):
        corpus = common.issues_corpus(split, obfuscated)
        cache = {}

        def lines_of(tid, corpus=corpus, cache=cache):
            if tid not in cache:
                with open(os.path.join(ib, corpus.roots[0], f"{tid}.txt"), "rb") as f:
                    cache[tid] = common.Lines(f.read().decode("utf-8", "replace"))
            return cache[tid]

        drun, dflags = common.read_flags(folder, DEFAULT)
        for name, verifier in UNIONS.items():
            vrun, vflags = common.read_flags(folder, verifier)
            for r in (drun, vrun):
                if r.get("corpus") != corpus.name:
                    raise common.Refused(f"{r.get('system')}: its flags are of {r.get('corpus')}, not {corpus.name}")
            iss[key][name] = issues_union(cands, drun, dflags, vrun, vflags, thresholds[name], lines_of)
            if not obfuscated:
                # end to end: the default engine's findings and the verifier's accepted candidates on the texts
                erun, eflags = common.read_flags(folder, f"{verifier}-e2e")
                merged = union_flags(dflags, eflags, thresholds[name])
                e = issues.score_system(cands, {"system": f"{name}-e2e", "unit": "line", "scored": False}, merged, lines_of)
                e["threshold"] = thresholds[name]
                iss[key][f"{name}-e2e"] = e
    for name in UNIONS:
        a, b = iss["systems"][name]["candidate"]["f1"], iss["systems_on_obfuscated_values"][name]["candidate"]["f1"]
        iss["obfuscation_sensitivity"][name] = {"f1_published": a, "f1_obfuscated": b, "difference": a - b}
    iss["experiment"] = EXPERIMENT
    iss["thresholds"] = thresholds
    common.write_json(os.path.join(out_dir, "issues.json"), iss)

    # clean text, the census reading (G7)
    import clean_text

    cdir = os.path.join(clean_text.clean_dir(), "flags")
    listing = clean_text.corpus_list()
    corpus = clean_text.corpus(listing)
    cl = {"experiment": EXPERIMENT, "corpus": {"files": listing["files"], "bytes": listing["bytes"], "list_sha256": listing["listSha256"]},
          "gate_reading": "flags", "readings": {"flags": {"how": clean_text.READINGS["flags"], "systems": {}}}}
    runs = {}
    for n in common.systems_in(cdir):
        run, flags = common.read_flags(cdir, n)
        if run.get("corpus") != "clean":
            raise common.Refused(f"clean/{n}: flags of {run.get('corpus')}")
        for x in flags:
            corpus.check(x["file"])
        runs[n] = (run, flags)
        cl["readings"]["flags"]["systems"][n] = clean_text.share_of(flags, run, len(corpus.files))[0]
    for name, verifier in UNIONS.items():
        cl["readings"]["flags"]["systems"][name] = clean_union(runs[DEFAULT][1], runs[verifier][1], thresholds[name], len(corpus.files))
    common.write_json(os.path.join(out_dir, "clean.json"), cl)
    return sc, iss, cl


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("what", choices=("tau", "score"))
    ap.add_argument("--split", choices=("validation", "test"))
    ap.add_argument("--tau", default=TAU_JSON)
    ap.add_argument("--out")
    a = ap.parse_args()
    try:
        if a.what == "tau":
            if a.split not in (None, "validation"):
                raise common.Refused("tau3 is chosen on validation only")
            r = tau3(a.out or TAU_JSON)
            for n, t in r["thresholds"].items():
                g = r["validation"][n]["gated"]["all"]
                common.log(f"{n:<10} threshold {t['threshold']}  validation gated F1 {g['f1']:.4f} (P {g['precision']:.4f} R {g['recall']:.4f})")
        else:
            if a.split is None:
                raise common.Refused("--split validation or test")
            out = a.out or os.path.join(common.eval_root(), a.split, "v3")
            sc, iss, cl = score(a.split, a.tau, out)
            for n in UNIONS:
                g = sc["systems"][n]["line"]["gated"]
                common.log(f"{n:<14} gated F1 {g['all']['f1']:.4f}, without the largest {g['without_largest']['f1']:.4f}; "
                           f"issues F1 {iss['systems'][n]['candidate']['f1']:.4f}; clean {cl['readings']['flags']['systems'][n]['files_with_an_accepted_flag']} files")
    except common.Refused as e:
        common.log(f"union: refused: {e}")
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
