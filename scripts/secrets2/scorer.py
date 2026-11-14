#!/usr/bin/env python3
"""The scorer of s6-secrets v2 (docs/s6-secrets2-prereg.md, sections 7.1, 7.2, 7.3, 7.6 and 9,
amendments 1 and 2). Every system goes through it, from its flags in the common format
(scripts/secrets2/common.py).

  python3 -I scripts/secrets2/scorer.py --split validation --flags DIR --out scores.json
          [--systems a,b] [--reference NAME] [--threshold NAME=0.5]...

Two universes (amendment 2). `full`: every meta row. `gated`: the rows whose Category is exactly
"UUID" are removed, for every system; a line whose only rows are such rows is neither positive
nor negative and a flag on it is not counted; compound categories (Token:UUID) stay.

Line level (7.1), the gated unit. One unit per (file, LineStart) that has a row. It is positive
when one of its rows is T. A system flags it when one of its accepted flags lies on line
LineStart: a flag covers the lines `line` to `end_line`, whatever its columns. Precision,
recall, F1, and the false-alarm rate (the share of units without a T row that are flagged).
A positive unit no flag reaches is a false negative, so the generator's misses count.
Reported: pooled over the split's repositories; pooled without the largest test repository
(amendment 1); the macro average of F1 over the repositories with at least 20 T rows; by
CredData category and by path scope (section 9).

Candidate level (7.2), for the systems whose unit is the candidate. A flag with columns is
matched to the rows it overlaps, as the generator's labels were (amendment 2):
  - a row is one span from ValueStart on LineStart to ValueEnd on LineEnd; a row without
    ValueStart is its whole line(s); a row with a start and no end runs to the end of LineEnd;
  - a flag that overlaps rows of several labels takes T if one is T, else X, else F;
  - a flag on a labelled line that overlaps no row is dropped and counted (D); a flag on a
    line without any row is unlabelled (U); in the gated universe a flag on a line whose only
    rows are exactly-UUID rows is not counted (E).
Positive: T. Precision, recall and F1 at the system's threshold, PR-AUC (average precision),
recall at a 1% false-alarm rate, ECE over 15 equal-width bins (as sky-eval computes it: the
confidence of the answer, max(p, 1 - p), against whether the answer is right), and the same
calibration error read on P(real) itself (`ece_positive`).

Span level (section 9, "in the cbx sense"), for every system that gives columns: recall is the
share of T rows overlapped by an accepted flag (the generator's recall of 2.2 when the system
is the generator alone); precision is the share of accepted flags on labelled lines that
overlap a T row.

Unlabelled-line alarms (7.3): accepted flags on lines without any meta row, per 1,000 files of
the split. Counted, never scored.

Uncertainty (7.6): a paired bootstrap over repositories (1,000 resamples, 95%, seed 1234) of
the line-F1 difference between `--reference` and every other system, and of the ECE of every
scored system.

Only the files of the split's own repositories are read (the guard of common.py).
"""
import argparse
import csv
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

UNIVERSES = ("gated", "full")
CATEGORIES = ("Password", "Token", "Key", "Secret", "API", "Auth")
MACRO_MIN_T_ROWS = 20
BINS = 15
RESAMPLES = 1000
SEED = 1234
HEADER = "Id,FileID,Domain,RepoName,FilePath,LineStart,LineEnd,GroundTruth,ValueStart,ValueEnd,CryptographyKey,PredefinedPattern,Category".split(",")


def is_uuid_only(category):
    return category == "UUID"


# ---------------------------------------------------------------- meta

def _int(v):
    return None if v is None or v == "" else int(v)


def parse_meta(text, repo=None):
    """The rows of one meta CSV. With `repo`, a row that names another repository stops the run."""
    rows = list(csv.reader(text.splitlines(keepends=True)))
    if not rows or rows[0] != HEADER:
        raise common.Refused(f"meta of {repo}: unexpected header")
    out = []
    for r in rows[1:]:
        if not r:
            continue
        row = {"id": r[0], "repo": r[3], "file": r[4], "line_start": _int(r[5]), "line_end": _int(r[6]), "truth": r[7],
               "value_start": _int(r[8]), "value_end": _int(r[9]), "category": r[12]}
        if repo is not None and (row["repo"] != repo or not row["file"].startswith(f"data/{repo}/")):
            raise common.Refused(f"meta of {repo}: row {row['id']} names another repository")
        if row["truth"] not in ("T", "F", "X"):
            raise common.Refused(f"meta of {repo}: row {row['id']} has GroundTruth {row['truth']!r}")
        out.append(row)
    return out


def read_meta(split):
    rows = []
    for repo in split.repos:
        split.check_repo(repo)
        with open(os.path.join(common.creddata_base(), "meta", f"{repo}.csv"), newline="") as f:
            rows += parse_meta(f.read(), repo)
    return rows


def row_span(row, lines):
    """[start, end) of a row in its file's text, as code-point offsets; None for a row outside
    its file or with an empty span."""
    a, b = row["line_start"] - 1, row["line_end"] - 1
    if not (0 <= a <= b < len(lines)):
        return None
    whole = row["value_start"] is None or row["value_start"] < 0
    start = lines.starts[a] if whole else min(lines.starts[a] + row["value_start"], lines.ends[a])
    open_end = whole or row["value_end"] is None or row["value_end"] < 0
    end = lines.ends[b] if open_end else min(lines.starts[b] + row["value_end"], lines.ends[b])
    if end <= start:
        return None
    return start, end, whole


def categories_of(category):
    return {c if c in CATEGORIES else "other" for c in category.split(":")}


def scope_of(path):
    """`test`, `src` or `other` from the scope directories CredData files a path under
    (data/<RepoID>/<scope words>/<FileID><ext>): `test` when one of them is "test", else `src`
    when one is "src"."""
    words = path.split("/")[2:-1]
    return "test" if "test" in words else "src" if "src" in words else "other"


class Labelled:
    """The meta rows of a split, placed in their files. `texts` maps a path to its text (None:
    the file is empty or binary) and is asked only for the files that have rows."""

    def __init__(self, rows, repos, text_of, n_files, largest=None):
        self.repos = list(repos)
        self.n_files = n_files
        self.largest = largest
        self.files = {}
        for row in rows:
            self.files.setdefault(row["file"], []).append(row)
        self.lines = {}
        self.t_rows = {u: {r: 0 for r in self.repos} for u in UNIVERSES}
        for path, frows in self.files.items():
            text = text_of(path)
            lines = common.Lines(text) if text is not None else None
            self.lines[path] = lines
            for row in frows:
                row["uuid_only"] = is_uuid_only(row["category"])
                row["span"] = row_span(row, lines) if lines is not None else None
                if row["truth"] == "T":
                    self.t_rows["full"][row["repo"]] += 1
                    if not row["uuid_only"]:
                        self.t_rows["gated"][row["repo"]] += 1
        self._cache = {}

    def view(self, path, universe):
        """Units, labelled lines and row spans of one file in one universe."""
        key = (path, universe)
        if key not in self._cache:
            rows = [r for r in self.files.get(path, []) if universe == "full" or not r["uuid_only"]]
            units = {}
            labelled = set()
            for r in rows:
                u = units.setdefault(r["line_start"], {"pos": False, "lines": set(), "cats": set()})
                u["pos"] = u["pos"] or r["truth"] == "T"
                u["lines"].update(range(r["line_start"], r["line_end"] + 1))
                u["cats"] |= categories_of(r["category"])
                if r["span"] is not None:
                    labelled.update(range(r["line_start"], r["line_end"] + 1))
            self._cache[key] = (units, labelled, [r for r in rows if r["span"] is not None], rows)
        return self._cache[key]


def load_labelled(split, files):
    """The labelled rows of a CredData split, read from disk after the guard."""
    base = common.creddata_base()

    def text_of(path):
        split.check(path)
        full = os.path.join(base, path)
        return common.read_text(full) if os.path.isfile(full) else None

    rows = read_meta(split)
    largest, stand_in = split.largest, False
    if largest is None:
        # No repository of this split is named in splits.json: the rehearsal removes the one
        # with the most T rows, so that the "without the largest" path runs.
        counts = {}
        for r in rows:
            if r["truth"] == "T":
                counts[r["repo"]] = counts.get(r["repo"], 0) + 1
        largest = max(sorted(counts), key=lambda k: counts[k]) if counts else None
        stand_in = True
    lab = Labelled(rows, split.repos, text_of, len(files), largest)
    lab.largest_is_stand_in = stand_in
    return lab


# ---------------------------------------------------------------- metrics

def prf(tp, fp, fn, tn=None):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    out = {"tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0.0}
    if tn is not None:
        out["tn"] = tn
        out["false_alarm_rate"] = fp / (fp + tn) if fp + tn else 0.0
    return out


def f1_of(tp, fp, fn):
    return 2 * tp / (2 * tp + fp + fn) if tp else 0.0


def ece_bins(items):
    """Sums per bin of (confidence, hit, count), the confidence being that of the answer."""
    b = [[0.0, 0.0, 0] for _ in range(BINS)]
    for p, pos in items:
        conf = max(p, 1.0 - p)
        hit = 1.0 if (p >= 0.5) == bool(pos) else 0.0
        i = min(int(conf * BINS), BINS - 1)
        b[i][0] += conf
        b[i][1] += hit
        b[i][2] += 1
    return b


def ece_from_bins(b):
    n = sum(x[2] for x in b)
    return sum(abs(x[0] - x[1]) for x in b if x[2]) / n if n else None


def ece(items):
    return ece_from_bins(ece_bins(items))


def ece_positive(items):
    """Calibration of P(real) itself: bins of p, mean p against the share of positives."""
    b = [[0.0, 0.0, 0] for _ in range(BINS)]
    for p, pos in items:
        i = min(int(p * BINS), BINS - 1)
        b[i][0] += p
        b[i][1] += 1.0 if pos else 0.0
        b[i][2] += 1
    return ece_from_bins(b)


def curve(items):
    """Operating points by decreasing threshold: (threshold, tp, fp), one per distinct score."""
    pts = []
    tp = fp = 0
    items = sorted(items, key=lambda x: -x[0])
    for i, (p, pos) in enumerate(items):
        if pos:
            tp += 1
        else:
            fp += 1
        if i + 1 == len(items) or items[i + 1][0] != p:
            pts.append((p, tp, fp))
    return pts


def pr_auc(items):
    """Average precision: the sum over thresholds of (recall step) x precision."""
    n_pos = sum(1 for _, pos in items if pos)
    if not n_pos:
        return None
    ap, prev = 0.0, 0
    for _, tp, fp in curve(items):
        ap += (tp - prev) / n_pos * (tp / (tp + fp))
        prev = tp
    return ap


def recall_at_fa(items, rate=0.01):
    """The highest recall among thresholds whose false-alarm rate is at most `rate`."""
    n_pos = sum(1 for _, pos in items if pos)
    n_neg = len(items) - n_pos
    if not n_pos:
        return None
    best = 0.0
    for _, tp, fp in curve(items):
        if n_neg == 0 or fp / n_neg <= rate:
            best = max(best, tp / n_pos)
    return best


def best_f1_threshold(items):
    """(threshold, F1) maximising F1 of "accept when score >= threshold" over the distinct
    scores; among equal F1 the highest threshold."""
    n_pos = sum(1 for _, pos in items if pos)
    best = (None, -1.0)
    for t, tp, fp in curve(items):
        f = f1_of(tp, fp, n_pos - tp)
        if f > best[1]:
            best = (t, f)
    return best


def at_threshold(items, threshold):
    tp = sum(1 for p, pos in items if pos and p >= threshold)
    fp = sum(1 for p, pos in items if not pos and p >= threshold)
    fn = sum(1 for p, pos in items if pos and p < threshold)
    return prf(tp, fp, fn, len(items) - tp - fp - fn)


def percentile(sorted_values, q):
    if not sorted_values:
        return None
    k = (len(sorted_values) - 1) * q
    lo = int(k)
    hi = min(lo + 1, len(sorted_values) - 1)
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (k - lo)


def resamples(repos, n=RESAMPLES, seed=SEED):
    rng = random.Random(seed)
    k = len(repos)
    return [[repos[rng.randrange(k)] for _ in range(k)] for _ in range(n)]


def interval(values, value):
    v = sorted(x for x in values if x is not None)
    return {"value": value, "lo": percentile(v, 0.025), "hi": percentile(v, 0.975)}


# ---------------------------------------------------------------- one system

def _zero():
    return [0, 0, 0, 0]  # tp, fp, fn, tn


def _add(counts, pos, flagged):
    counts[(0 if flagged else 2) if pos else (1 if flagged else 3)] += 1


def _m(c):
    return prf(c[0], c[1], c[2], c[3])


def _sum(per_repo, repos):
    out = _zero()
    for r in repos:
        c = per_repo.get(r)
        if c:
            for i in range(4):
                out[i] += c[i]
    return out


def score_system(lab, run, flags, threshold=None):
    """Everything of one system, in both universes. Returns (result, internals); the internals
    (per-repository counts and candidate items) feed the bootstrap."""
    scored = any("score" in x for x in flags) or run.get("scored", False)
    if threshold is None:
        threshold = run.get("threshold")
    if scored and threshold is None:
        raise common.Refused(f"{run.get('system')}: a scored system needs a threshold")
    accepted = lambda x: (not scored) or x.get("score", 0.0) >= threshold
    by_file = {}
    for x in flags:
        by_file.setdefault(x["file"], []).append(x)
    repos = lab.repos
    others = [r for r in repos if r != lab.largest]
    has_spans = any("start" in x for x in flags)
    out = {"unit": run.get("unit"), "scored": scored, "threshold": threshold if scored else None, "line": {}, "unlabelled": None}
    internals = {"line": {}, "items": {}}

    # Unlabelled-line alarms: the same in both universes (a line with a UUID row has a row).
    alarms = alarm_lines = 0
    for path, fl in by_file.items():
        _, labelled, _, _ = lab.view(path, "full")
        seen = set()
        for x in fl:
            if not accepted(x):
                continue
            ls = range(x["line"], x.get("end_line", x["line"]) + 1)
            if not any(l in labelled for l in ls):
                alarms += 1
                seen.add(x["line"])
        alarm_lines += len(seen)
    out["unlabelled"] = {"flags": alarms, "lines": alarm_lines, "files": lab.n_files,
                         "per_1000_files": 1000.0 * alarms / lab.n_files if lab.n_files else None}

    for u in UNIVERSES:
        line = {r: _zero() for r in repos}
        line_range = _zero()
        line_matched = _zero()
        by_cat = {c: _zero() for c in (*CATEGORIES, "other")}
        by_scope = {s: _zero() for s in ("test", "src", "other")}
        labels = {k: 0 for k in "TXFDUE"}
        items = {r: [] for r in repos}
        t_rows = covered = on_labelled = on_t = 0
        for path in sorted(set(lab.files) | set(by_file)):
            units, labelled, spans, rows = lab.view(path, u)
            _, labelled_full, _, _ = lab.view(path, "full")
            lines = lab.lines.get(path)
            flagged = set()
            flagged_matched = set()
            hit_rows = set()
            for x in by_file.get(path, []):
                ok = accepted(x)
                ls = range(x["line"], x.get("end_line", x["line"]) + 1)
                label = None
                if "start" in x:
                    if not any(l in labelled_full for l in ls):
                        label = "U"
                    elif not any(l in labelled for l in ls):
                        label = "E"
                    else:
                        a = lines.offset(x["line"], x["start"])
                        b = lines.offset(x.get("end_line", x["line"]), x["end"])
                        over = [r for r in spans if a is not None and b is not None and a < r["span"][1] and r["span"][0] < b]
                        truths = {r["truth"] for r in over}
                        label = "T" if "T" in truths else "X" if "X" in truths else "F" if truths else "D"
                        if ok:
                            on_labelled += 1
                            t_over = [r["id"] for r in over if r["truth"] == "T"]
                            hit_rows.update(t_over)
                            on_t += bool(t_over)
                    labels[label] += 1
                    if label in "TXF" and run.get("unit") == "candidate":
                        repo = path.split("/")[1] if path.startswith("data/") else x["repo"]
                        items.setdefault(repo, []).append((x.get("score", 1.0), label == "T"))
                if ok:
                    flagged.update(ls)
                    if label is None or label in "TXF":
                        flagged_matched.update(ls)
            if not units:
                continue
            repo = rows[0]["repo"]
            scope = scope_of(path)
            for start, unit in units.items():
                f = start in flagged
                _add(line[repo], unit["pos"], f)
                _add(line_range, unit["pos"], bool(unit["lines"] & flagged))
                _add(line_matched, unit["pos"], start in flagged_matched)
                for c in unit["cats"]:
                    _add(by_cat[c], unit["pos"], f)
                _add(by_scope[scope], unit["pos"], f)
            for r in rows:
                if r["truth"] == "T":
                    t_rows += 1
                    covered += r["id"] in hit_rows

        macro_repos = [r for r in repos if lab.t_rows[u][r] >= MACRO_MIN_T_ROWS]
        macro = [f1_of(*line[r][:3]) for r in macro_repos]
        out["line"][u] = {
            "all": _m(_sum(line, repos)),
            "without_largest": _m(_sum(line, others)),
            "macro_f1": {"value": sum(macro) / len(macro) if macro else None, "repositories": len(macro_repos)},
            "by_category": {c: _m(v) for c, v in by_cat.items()},
            "by_scope": {s: _m(v) for s, v in by_scope.items()},
            "other_readings": {"flag_on_any_line_of_the_row": _m(line_range), "unmatched_flags_dropped": _m(line_matched)},
        }
        internals["line"][u] = line
        if has_spans:
            p = on_t / on_labelled if on_labelled else 0.0
            r = covered / t_rows if t_rows else 0.0
            out.setdefault("span", {})[u] = {"t_rows": t_rows, "covered": covered, "recall": r, "flags_on_labelled_lines": on_labelled,
                                             "flags_on_t": on_t, "precision": p, "f1": 2 * p * r / (p + r) if p + r else 0.0}
            out.setdefault("labels", {})[u] = labels
        if run.get("unit") == "candidate":
            everything = [i for r in repos for i in items[r]]
            rest = [i for r in others for i in items[r]]
            cand = {"n": len(everything), "positives": sum(1 for _, pos in everything if pos)}
            t = threshold if scored else 0.5
            cand.update({k: v for k, v in at_threshold(everything, t).items()})
            cand["without_largest"] = at_threshold(rest, t)
            if scored:
                cand.update({"pr_auc": pr_auc(everything), "recall_at_1pct_false_alarms": recall_at_fa(everything),
                             "ece": ece(everything), "ece_positive": ece_positive(everything)})
                best_t, best_f = best_f1_threshold(everything)
                cand["best_f1"] = {"threshold": best_t, "f1": best_f}
            out.setdefault("candidate", {})[u] = cand
            internals["items"][u] = items
    return out, internals


# ---------------------------------------------------------------- bootstrap

def bootstrap(lab, internals, reference=None):
    """F1 differences `reference` - other (line level), and the ECE of every scored system."""
    out = {"resamples": RESAMPLES, "seed": SEED, "level": 0.95, "reference": reference, "f1_difference": {}, "ece": {}}
    sets = {"all": lab.repos, "without_largest": [r for r in lab.repos if r != lab.largest]}
    draws = {k: resamples(v) for k, v in sets.items()}

    def f1s(per_repo, draw):
        return [f1_of(*_sum_list(per_repo, d)) for d in draw]

    def _sum_list(per_repo, d):
        tp = fp = fn = 0
        for r in d:
            c = per_repo[r]
            tp += c[0]
            fp += c[1]
            fn += c[2]
        return tp, fp, fn

    if reference is not None and reference in internals:
        for name, it in internals.items():
            if name == reference:
                continue
            out["f1_difference"][name] = {}
            for u in UNIVERSES:
                out["f1_difference"][name][u] = {}
                for k, repos in sets.items():
                    a = f1s(internals[reference]["line"][u], draws[k])
                    b = f1s(it["line"][u], draws[k])
                    value = f1_of(*_sum_list(internals[reference]["line"][u], repos)) - f1_of(*_sum_list(it["line"][u], repos))
                    out["f1_difference"][name][u][k] = interval([x - y for x, y in zip(a, b)], value)
    for name, it in internals.items():
        if not it.get("scored") or not it["items"]:
            continue
        out["ece"][name] = {}
        for u in UNIVERSES:
            bins = {r: ece_bins(it["items"][u].get(r, [])) for r in lab.repos}
            total = lambda d: ece_from_bins([[sum(bins[r][i][j] for r in d) for j in range(3)] for i in range(BINS)])
            out["ece"][name][u] = interval([total(d) for d in draws["all"]], total(lab.repos))
    return out


def score_all(lab, systems, reference=None, thresholds=None):
    """`systems`: {name: (run, flags)}. The scores of every system and the bootstrap."""
    thresholds = thresholds or {}
    out = {"universes": {"gated": 'rows whose Category is exactly "UUID" removed (amendment 2)', "full": "every meta row"},
           "repositories": len(lab.repos), "files": lab.n_files,
           "largest_repository": {"repo_id": lab.largest, "rehearsal_stand_in": getattr(lab, "largest_is_stand_in", False),
                                  "t_rows": {u: lab.t_rows[u].get(lab.largest) for u in UNIVERSES}},
           "t_rows": {u: sum(lab.t_rows[u].values()) for u in UNIVERSES},
           "repositories_with_20_t_rows": {u: sum(1 for r in lab.repos if lab.t_rows[u][r] >= MACRO_MIN_T_ROWS) for u in UNIVERSES},
           "systems": {}}
    internals = {}
    for name, (run, flags) in systems.items():
        res, it = score_system(lab, run, flags, thresholds.get(name))
        res["version"] = run.get("version")
        res["cpu_seconds"] = run.get("cpu_seconds")
        res["wall_seconds"] = run.get("wall_seconds")
        it["scored"] = res["scored"]
        out["systems"][name] = res
        internals[name] = it
    out["bootstrap"] = bootstrap(lab, internals, reference)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--split", required=True, choices=common.CREDDATA_SPLITS)
    ap.add_argument("--flags", required=True, help="directory of <system>.flags.jsonl and <system>.run.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--systems", help="comma-separated names (default: every system of the directory)")
    ap.add_argument("--reference", help="the system whose F1 differences are bootstrapped (the model)")
    ap.add_argument("--threshold", action="append", default=[], metavar="NAME=VALUE")
    a = ap.parse_args()
    try:
        split = common.open_split(a.split)
        files = common.creddata_files(split)
        lab = load_labelled(split, files)
        names = a.systems.split(",") if a.systems else common.systems_in(a.flags)
        systems = {}
        for n in names:
            run, flags = common.read_flags(a.flags, n)
            if run.get("corpus") != f"creddata:{a.split}":
                raise common.Refused(f"{n}: its flags are of {run.get('corpus')}, not creddata:{a.split}")
            for x in flags:
                split.check(x["file"])
            systems[n] = (run, flags)
        thresholds = {k: float(v) for k, v in (t.split("=", 1) for t in a.threshold)}
        out = {"experiment": "s6-secrets v2 (docs/s6-secrets2-prereg.md)", "split": a.split, **score_all(lab, systems, a.reference, thresholds)}
    except common.Refused as e:
        common.log(f"scorer: refused: {e}")
        return 3
    common.write_json(a.out, out)
    for n, s in out["systems"].items():
        g, f = s["line"]["gated"]["all"], s["line"]["full"]["all"]
        common.log(f"{n:<24} gated F1 {g['f1']:.4f} (P {g['precision']:.4f} R {g['recall']:.4f})  full F1 {f['f1']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
