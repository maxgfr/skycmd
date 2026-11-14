#!/usr/bin/env python3
"""What the systems that judge candidates share (the model, the size-matched LR, potion + LR):
the table of states they read, and the way from one score per candidate to the common result
format (docs/s6-secrets2-prereg.md, sections 4.1, 6 and 7).

The table of a set of candidates is the data side's `tables/<source>-<split>.jsonl`
(crates/sky-data/src/secrets2/mix.rs): one JSON line per candidate, in the order of the
decision records, with
  id        `cred:<train|val|test>:<index>`: the index of the candidate in the generator's
            export of its split; `issue:<split>:<row>`: the row of the issue-report split file;
            `clean:all:<index>`: the index in the generator's export of the clean-text corpus;
            `issue-e2e:<split>:<index>`: the index in the generator's export of the issue texts
  state     the state text the model reads (sky_infer::secret_cand)
  features  the hand features of section 4.2 by name
  label     T, X or F when known (training and validation), `real` 0/1, `weak`, `source`

A score is P(real). On a CredData corpus a scored candidate becomes a flag at the place of the
generator's candidate of the same index (DIR/generator.flags.jsonl, written by
run_secretgate.py generator); on an issue corpus it becomes the judgement of the dataset's own
candidate of that row (run.json: judges = dataset-candidates). End to end (section 7.4), the
candidates are the generator's on the issue texts: a scored candidate is a flag at the place of
the generator's candidate, as on CredData, and the system is scored like one that reads the
texts.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

# Section 4.2, in its order, by the names of the data side's tables. `tokens` is in the tables
# too: it is the denominator of chars_per_token, not a feature of its own.
FEATURES = ("len", "entropy", "chars_per_token", "digits", "upper", "lower", "other", "word_run", "gate", "placeholder", "stopword", "allowlist")


def read_table(path):
    with open(path) as f:
        for n, line in enumerate(f):
            if not line.strip():
                continue
            row = json.loads(line)
            if "id" not in row or "state" not in row:
                raise common.Refused(f"{path}: line {n + 1} is not a row of a table of states")
            yield row


def index_of(row_id, kind):
    """The index at the end of an id of the given kind (`cred` or `issue`)."""
    parts = row_id.split(":")
    if len(parts) != 3 or parts[0] != kind or not parts[2].isdigit():
        raise common.Refused(f"{row_id!r} is not the id of a {kind} candidate")
    return int(parts[2])


def to_flags(corpus_name, scored, generator_flags=None, issue_candidates=None, e2e=False):
    """Flags of `scored`, an iterable of (id, P(real)).

    CredData: `generator_flags` are the generator's flags of the same corpus, whose `id` is the
    index of the candidate; a scored candidate that is not among them stops the run. Issue
    corpora: `issue_candidates` are the rows of candidates.jsonl (issues.py prepare); the flag
    is the judgement of one row, read by its `id` (its `line` is not a place: the row's own
    span, in candidates.jsonl, is)."""
    if corpus_name.startswith("creddata:") or corpus_name == "clean" or e2e:
        kind = "clean" if corpus_name == "clean" else "issue-e2e" if e2e else "cred"
        places = {x["id"]: x for x in generator_flags}
        for row_id, p in scored:
            x = places.get(index_of(row_id, kind))
            if x is None:
                raise common.Refused(f"{row_id}: no candidate of that index in the generator's export of {corpus_name}")
            yield {**x, "score": round(float(p), 6)}
    elif corpus_name.startswith("issues:"):
        texts = {c["row"]: c["text"] for c in issue_candidates}
        root = "texts-obfuscated" if corpus_name.endswith(":obfuscated") else "texts"
        for row_id, p in scored:
            row = index_of(row_id, "issue")
            if row not in texts:
                raise common.Refused(f"{row_id}: no row of that index in {corpus_name}")
            yield {"repo": "issues", "file": f"{root}/{texts[row]}.txt", "line": 1, "score": round(float(p), 6), "id": row}
    else:
        raise common.Refused(f"unknown corpus {corpus_name!r}")


def write_scored(out_dir, system, corpus_name, scored, threshold, run, generator_dir=None, e2e=False):
    """Writes the flags and run.json of a system that scored candidates. `e2e`: the candidates
    are the generator's on the texts of an issue corpus, not the dataset's own."""
    generator_flags = issue_candidates = None
    if e2e and not (corpus_name.startswith("issues:") and not corpus_name.endswith(":obfuscated")):
        raise common.Refused(f"end to end is read on the issue texts as published, not on {corpus_name}")
    if corpus_name.startswith("issues:") and not e2e:
        with open(os.path.join(common.issues_dir(corpus_name.split(":")[1]), "candidates.jsonl")) as f:
            issue_candidates = [json.loads(line) for line in f if line.strip()]
    if corpus_name.startswith("creddata:") or corpus_name == "clean" or e2e:
        gen_run, generator_flags = common.read_flags(generator_dir or out_dir, "generator")
        if gen_run.get("corpus") != corpus_name:
            raise common.Refused(f"the generator's flags are of {gen_run.get('corpus')}, not {corpus_name}")
    meta = {**run, "corpus": corpus_name, "unit": "candidate", "scored": True, "threshold": threshold}
    if corpus_name.startswith("issues:") and not e2e:
        meta["judges"] = "dataset-candidates"
    if e2e:
        meta["judges"] = "generator-candidates"
    return common.write_flags(out_dir, system, to_flags(corpus_name, scored, generator_flags, issue_candidates, e2e), meta)
