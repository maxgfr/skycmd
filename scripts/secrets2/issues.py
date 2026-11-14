#!/usr/bin/env python3
"""The issue-report scoring of s6-secrets v2 (docs/s6-secrets2-prereg.md, section 7.4; Zenodo
10.5281/zenodo.19622962, CC BY 4.0). The set holds real values: everything this script writes
stays in the data folder, and nothing it prints or scores carries a value.

  issues.py prepare   --split validation
      reads the split file (after the guard and its frozen hash) and writes, under
      <data>/eval/issues/<split>/: texts/<id>.txt, one file per distinct issue text (id: the
      first 16 hex digits of the sha256 of the text), so that the file scanners can read them;
      candidates.jsonl, one line per row: its index in the split file, its text, its label and
      the place of the candidate string in the text (its FIRST occurrence, as the data side
      locates it: code-point offsets); index.json, the counts.
  issues.py obfuscate --split validation
      writes texts-obfuscated/: the same texts where every positive candidate string is
      replaced, wherever it stands in its text, by what CredData's `get_obfuscated_value`
      makes of it (run from the CredData checkout, inside its container, without a network).
      Lengths are kept, so the places of candidates.jsonl hold for both.
  issues.py score     --split validation --flags DIR [--flags-obfuscated DIR] --out FILE
      every system of DIR (flags of the corpus issues:<split>), three ways:
      candidate level   each row is one candidate; a scanner flags it when one of its accepted
                        flags overlaps the candidate in the text; a system that judges the
                        dataset's candidates itself (run.json: judges = dataset-candidates, the
                        model and the size-matched LR) gives one score per row, by its `id`.
                        Precision, recall, F1, false-alarm rate; ECE and PR-AUC with scores.
      end to end        the dataset's positives as the reference, for the systems that read
                        the texts: recall as above; precision over ALL the accepted flags of
                        the system (a flag that overlaps no positive is a false alarm), and
                        over the flags that overlap a candidate of the dataset.
      obfuscation       with --flags-obfuscated: candidate-level F1 on the texts as published
      sensitivity       minus F1 on the obfuscated texts.

The test split is refused unless the test is open (common.py).
"""
import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402
import scorer  # noqa: E402

SEED = 1234
CREDDATA_IMAGE = "s6secrets2-creddata:py310"
CREDDATA_FILES = ("obfuscate_creds.py", "meta_row.py", "constants.py")


def text_id(text):
    return hashlib.sha256(text.encode("utf-8", "surrogatepass")).hexdigest()[:16]


def read_rows(path):
    """(candidate, text, label) of every row of a split file, in order."""
    csv.field_size_limit(1 << 31)
    with open(path, newline="", encoding="utf-8") as f:
        rows = csv.reader(f)
        header = next(rows)
        c, t, l = (header.index(k) for k in ("candidate_string", "text", "label"))
        out = []
        for i, r in enumerate(rows):
            if len(r) != len(header) or r[l].strip() not in ("0", "1"):
                raise common.Refused(f"{path}: row {i} is not a row of the set")
            out.append((r[c], r[t], int(r[l].strip())))
    return out


def locate(text, candidate):
    """(start, end, occurrences) of the first occurrence of the candidate in its text, or None."""
    if not candidate:
        return None
    at = text.find(candidate)
    if at == -1:
        return None
    return at, at + len(candidate), text.count(candidate)


def replace_all(text, mapping):
    """`text` with every occurrence of every key of `mapping` replaced by its value of the same
    length; longer keys first, so where two overlap the shorter one's replacement stands inside
    the longer one's. Occurrences are looked up in the text as given."""
    out = list(text)
    for old in sorted(mapping, key=lambda s: (-len(s), s)):
        new = mapping[old]
        if not old or new is None or len(new) != len(old):
            continue
        at = text.find(old)
        while at != -1:
            out[at:at + len(old)] = new
            at = text.find(old, at + len(old))
    return "".join(out)


def prepare(split):
    path = common.issue_csv(split)
    base = common.issues_dir(split)
    texts_dir = os.path.join(base, "texts")
    os.makedirs(texts_dir, mode=0o700, exist_ok=True)
    rows = read_rows(path)
    texts, cands = {}, []
    counts = {"rows": len(rows), "positives": 0, "not_found": 0, "not_found_positives": 0, "several_occurrences": 0, "over_several_lines": 0}
    for i, (cand, text, label) in enumerate(rows):
        tid = text_id(text)
        texts.setdefault(tid, text)
        if texts[tid] != text:
            raise common.Refused(f"two texts of {path} share the id {tid}")
        counts["positives"] += label
        rec = {"row": i, "text": tid, "label": label}
        at = locate(text, cand)
        if at is None:
            counts["not_found"] += 1
            counts["not_found_positives"] += label
        else:
            rec.update({"start": at[0], "end": at[1]})
            counts["several_occurrences"] += at[2] > 1
            counts["over_several_lines"] += "\n" in cand
        cands.append(rec)
    binary = 0
    for tid, text in texts.items():
        data = text.encode("utf-8", "surrogatepass")
        binary += (not data) or b"\0" in data[:8192]
        with open(os.open(os.path.join(texts_dir, f"{tid}.txt"), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), "wb") as f:
            f.write(data)
    with open(os.open(os.path.join(base, "candidates.jsonl"), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), "w") as f:
        for rec in cands:
            f.write(json.dumps(rec) + "\n")
    index = {"split": split, "source": os.path.basename(path), "source_sha256": common.sha256_file(path), **counts,
             "distinct_texts": len(texts), "texts_empty_or_with_a_nul_byte": binary,
             "bytes": sum(len(t.encode("utf-8", "surrogatepass")) for t in texts.values()), "texts": sorted(texts)}
    common.write_json(os.path.join(base, "index.json"), index)
    return {k: v for k, v in index.items() if k != "texts"}


def obfuscate(split):
    path = common.issue_csv(split)
    base = common.issues_dir(split)
    index = common.load_json(os.path.join(base, "index.json"))
    rows = read_rows(path)
    by_text = {}
    for cand, text, label in rows:
        if label and cand and cand in text:
            by_text.setdefault(text_id(text), set()).add(cand)
    values = sorted({c for s in by_text.values() for c in s})
    work = os.path.join(base, "obfuscation-work")
    shutil.rmtree(work, ignore_errors=True)
    os.makedirs(work, mode=0o700)
    try:
        with open(os.path.join(work, "values.json"), "w") as f:
            json.dump(values, f)
        if not subprocess.run(["docker", "image", "inspect", CREDDATA_IMAGE], capture_output=True).returncode == 0:
            raise common.Refused(f"no Docker image {CREDDATA_IMAGE} (the image CredData was downloaded with, tools/docker in the data folder)")
        common.check_creddata()
        mounts = []
        for name in CREDDATA_FILES:
            mounts += ["-v", f"{os.path.join(common.creddata_base(), name)}:/creddata/{name}:ro"]
        r = subprocess.run(common.nice(["docker", "run", "--rm", "--network", "none", "--read-only", "--tmpfs", "/tmp", "--user", f"{os.getuid()}:{os.getgid()}",
                                        *mounts, "-v", f"{os.path.join(common.HERE, 'obfuscate_values.py')}:/run/obfuscate_values.py:ro", "-v", f"{work}:/work",
                                        "-e", "PYTHONDONTWRITEBYTECODE=1", "--entrypoint", "python", CREDDATA_IMAGE, "/run/obfuscate_values.py", "/creddata", "/work", str(SEED)]),
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise common.Refused(f"the obfuscation failed in its container (exit {r.returncode})")
        report = json.loads(r.stdout.strip().splitlines()[-1])
        mapping = dict(zip(values, common.load_json(os.path.join(work, "obfuscated.json"))))
    finally:
        shutil.rmtree(work, ignore_errors=True)
    out_dir = os.path.join(base, "texts-obfuscated")
    os.makedirs(out_dir, mode=0o700, exist_ok=True)
    changed = 0
    for tid in index["texts"]:
        with open(os.path.join(base, "texts", f"{tid}.txt"), "rb") as f:
            text = f.read().decode("utf-8", "surrogatepass")
        new = replace_all(text, {c: mapping[c] for c in by_text.get(tid, ())})
        if len(new) != len(text):
            raise common.Refused("an obfuscated text changed its length")
        changed += new != text
        with open(os.open(os.path.join(out_dir, f"{tid}.txt"), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), "wb") as f:
            f.write(new.encode("utf-8", "surrogatepass"))
    index["obfuscation"] = {"function": "CredData get_obfuscated_value at " + common.frozen()["creddata"]["commit"], "seed": SEED,
                            "meta_row": "Category and PredefinedPattern empty", "positive_values": report["values"],
                            "left_as_published": {k: report[k] for k in ("unchanged", "error", "timeout", "length")}, "texts_changed": changed}
    common.write_json(os.path.join(base, "index.json"), index)
    return index["obfuscation"]


# ---------------------------------------------------------------- scoring

def read_candidates(base):
    with open(os.path.join(base, "candidates.jsonl")) as f:
        return [json.loads(line) for line in f if line.strip()]


def flag_span(lines, x):
    """[start, end) of a flag in its text; a flag without columns is its whole line(s)."""
    end_line = x.get("end_line", x["line"])
    if "start" in x:
        a, b = lines.offset(x["line"], x["start"]), lines.offset(end_line, x["end"])
    else:
        a = lines.offset(x["line"], 0)
        b = lines.ends[end_line - 1] if 0 < end_line <= len(lines) else None
    if a is None or b is None:
        return None
    return a, max(b, a + 1)


def score_system(cands, run, flags, lines_of):
    """One system on the candidates of a split. `lines_of(text id)` gives the Lines of a text."""
    scored = any("score" in x for x in flags) or run.get("scored", False)
    threshold = run.get("threshold")
    if scored and threshold is None:
        raise common.Refused(f"{run.get('system')}: a scored system needs a threshold")
    accepted = lambda x: (not scored) or x.get("score", 0.0) >= threshold
    out = {"unit": run.get("unit"), "scored": scored, "threshold": threshold if scored else None, "version": run.get("version"),
           "cpu_seconds": run.get("cpu_seconds")}
    if run.get("judges") == "dataset-candidates":
        by_row = {x["id"]: x for x in flags}
        items, c = [], [0, 0, 0, 0]
        for cand in cands:
            x = by_row.get(cand["row"])
            hit = x is not None and accepted(x)
            scorer._add(c, bool(cand["label"]), hit)
            if x is not None and "score" in x:
                items.append((x["score"], bool(cand["label"])))
        out["candidate"] = {**scorer._m(c), "candidates": len(cands), "judged": len(by_row)}
        if scored:
            out["candidate"].update({"ece": scorer.ece(items), "ece_positive": scorer.ece_positive(items), "pr_auc": scorer.pr_auc(items)})
        return out
    spans = {}
    n_accepted = 0
    for x in flags:
        if not accepted(x):
            continue
        tid = os.path.basename(x["file"])[: -len(".txt")]
        s = flag_span(lines_of(tid), x)
        if s is not None:
            spans.setdefault(tid, []).append(s)
            n_accepted += 1
    c = [0, 0, 0, 0]
    on_positive, on_candidate = set(), set()
    for cand in cands:
        hits = []
        if "start" in cand:
            hits = [k for k, (a, b) in enumerate(spans.get(cand["text"], [])) if a < cand["end"] and cand["start"] < b]
        scorer._add(c, bool(cand["label"]), bool(hits))
        for k in hits:
            on_candidate.add((cand["text"], k))
            if cand["label"]:
                on_positive.add((cand["text"], k))
    out["candidate"] = {**scorer._m(c), "candidates": len(cands)}
    recall = out["candidate"]["recall"]
    p_all = len(on_positive) / n_accepted if n_accepted else 0.0
    p_judged = len(on_positive) / len(on_candidate) if on_candidate else 0.0
    f = lambda p: 2 * p * recall / (p + recall) if p + recall else 0.0
    out["end_to_end"] = {"flags": n_accepted, "flags_on_a_positive": len(on_positive), "flags_on_a_candidate": len(on_candidate), "recall": recall,
                         "precision": p_all, "f1": f(p_all), "precision_over_flags_on_a_candidate": p_judged, "f1_over_flags_on_a_candidate": f(p_judged)}
    return out


def score_dir(split, flags_dir, obfuscated):
    base = common.issues_dir(split)
    corpus = common.issues_corpus(split, obfuscated)
    cands = read_candidates(base)
    cache = {}

    def lines_of(tid):
        if tid not in cache:
            with open(os.path.join(base, corpus.roots[0], f"{tid}.txt"), "rb") as f:
                cache[tid] = common.Lines(f.read().decode("utf-8", "replace"))
        return cache[tid]

    out = {}
    for name in common.systems_in(flags_dir):
        run, flags = common.read_flags(flags_dir, name)
        if run.get("corpus") != corpus.name:
            raise common.Refused(f"{name}: its flags are of {run.get('corpus')}, not {corpus.name}")
        for x in flags:
            corpus.check(x["file"])
        out[name] = score_system(cands, run, flags, lines_of)
    return out


def score(split, flags_dir, flags_obfuscated=None):
    index = common.load_json(os.path.join(common.issues_dir(split), "index.json"))
    out = {"experiment": "s6-secrets v2 (docs/s6-secrets2-prereg.md, section 7.4)", "split": split,
           "set": {k: v for k, v in index.items() if k != "texts"}, "systems": score_dir(split, flags_dir, False)}
    if flags_obfuscated:
        obf = score_dir(split, flags_obfuscated, True)
        out["systems_on_obfuscated_values"] = obf
        out["obfuscation_sensitivity"] = {
            n: {"f1_published": out["systems"][n]["candidate"]["f1"], "f1_obfuscated": obf[n]["candidate"]["f1"],
                "difference": out["systems"][n]["candidate"]["f1"] - obf[n]["candidate"]["f1"]}
            for n in out["systems"] if n in obf}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("what", choices=("prepare", "obfuscate", "score"))
    ap.add_argument("--split", required=True, choices=sorted(common.ISSUE_SPLITS))
    ap.add_argument("--flags")
    ap.add_argument("--flags-obfuscated")
    ap.add_argument("--out")
    a = ap.parse_args()
    try:
        if a.what == "prepare":
            print(json.dumps(prepare(a.split), indent=1))
        elif a.what == "obfuscate":
            print(json.dumps(obfuscate(a.split), indent=1))
        else:
            out = score(a.split, a.flags, a.flags_obfuscated)
            common.write_json(a.out, out)
            for n, s in out["systems"].items():
                c = s["candidate"]
                common.log(f"{n:<24} candidate F1 {c['f1']:.4f} (P {c['precision']:.4f} R {c['recall']:.4f})")
    except common.Refused as e:
        common.log(f"issues: refused: {e}")
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
