#!/usr/bin/env python3
"""Two readings of an s6 evaluation dump against per-question answer priors.

docs/s6-gen-prereg.md, amendment 16 (after the study of s6-nano, docs/s6-nano2-study.md §3.1):
for each test source,

1. the share of items that get the model's most frequent answer. A question is an instruction
   (`ins`) of the source; an answer is the text of the option with the largest probability. For
   each question the count of its most frequent answer is taken, and the share is the sum of these
   counts over the items of the source. Also given: the number of questions, the number on which
   one answer covers at least 95% of the items, and the share of the single most frequent answer
   of the whole source (the same as the first share when the source has one question);
2. the score of answering each question with its most common training answer: the most frequent
   gold answer (text of the option with the largest target) among the training records with the
   same instruction (split `train` of the training source below; ties go to the answer met first
   in the file). An item whose question has no training record, or whose options do not hold that
   answer, counts as wrong. Sources with no training question of the same form get no score.

Also given as a reference, not a baseline: the score of answering each question with its most
frequent gold answer on the test items themselves (an oracle prior).

The gold answers and the dump order are those of scripts/s6-ece-by-source.py (split `test`, the
first `--max` records of each source, the same filters), and each dump line is checked against its
record. Model-free columns depend only on the test and training files.

    python3 scripts/s6-answer-priors.py --dump runs/s6-micro2/runs/s6-micro/eval/dump-s6-micro.jsonl --root .
"""

import argparse
import collections
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("ece_by_source", HERE / "s6-ece-by-source.py")
ebs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ebs)

# test source -> (training file, training source); None: no training question of the same form.
TRAIN = {
    "typed_decisions_bench": ("data/decisions/typed_decisions_train.jsonl", "typed_decisions_train"),
    "when2call": ("data/decisions/when2call.jsonl", "when2call"),
    "zs_massive_en": None,  # MASSIVE intents are trained only as 3-option questions of another source
    "zs_massive_fr": None,
    "notam:deel_subject": None,  # never trained
    "notam:notam_multi": None,
}


def gold_text(opts, target):
    return opts[ebs.argmax(target)]


def train_priors(root, rel, src):
    counts = collections.defaultdict(collections.Counter)
    order = collections.defaultdict(dict)
    for line in open(root / rel, encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("split") != "train" or r.get("src") != src:
            continue
        opts = [o for o in r.get("opts") or [] if isinstance(o, str)]
        target = [x for x in r.get("target") or [] if isinstance(x, (int, float))]
        if len(opts) < 2 or len(target) != len(opts):
            continue
        g = gold_text(opts, target)
        q = r.get("ins")
        counts[q][g] += 1
        order[q].setdefault(g, len(order[q]))
    prior = {}
    n = {}
    for q, c in counts.items():
        prior[q] = max(c, key=lambda g: (c[g], -order[q][g]))
        n[q] = sum(c.values())
    return prior, n


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dump", required=True)
    ap.add_argument("--root", default=".", help="checkout that holds data/")
    ap.add_argument("--max", type=int, default=2000)
    ap.add_argument("--json")
    a = ap.parse_args()
    root = Path(a.root)

    dump = [json.loads(l) for l in open(a.dump, encoding="utf-8") if l.strip()]
    items = collections.defaultdict(list)  # src -> [(ins, opts, gold, predicted)]
    i = -1
    for i, (r, opts, target) in enumerate(ebs.records(root, a.max)):
        if i >= len(dump):
            raise SystemExit(f"more records than dump lines ({len(dump)})")
        d = dump[i]
        same = (d["src"], d["t"], d["ins"], d["opts"], d["state"]) == (
            r.get("src", "?"), r.get("t"), r.get("ins"), opts, r.get("state"))
        if not same:
            raise SystemExit(f"dump line {i + 1} does not match its record")
        pred = opts[ebs.argmax(d["probs"])]
        items[d["src"]].append((d["ins"], opts, gold_text(opts, target), pred))
    if i + 1 != len(dump):
        raise SystemExit(f"{i + 1} records for {len(dump)} dump lines")

    out = {"dump": a.dump, "max": a.max, "sources": {}}
    for src in sorted(items):
        rows = items[src]
        n = len(rows)
        by_q = collections.defaultdict(list)
        for row in rows:
            by_q[row[0]].append(row)
        top_q = {q: collections.Counter(p for _, _, _, p in rs).most_common(1)[0][1] for q, rs in by_q.items()}
        whole_answer, whole = collections.Counter(p for _, _, _, p in rows).most_common(1)[0]
        gold_top = sum(collections.Counter(g for _, _, g, _ in rs).most_common(1)[0][1] for rs in by_q.values())
        entry = {
            "n": n,
            "questions": len(by_q),
            "model_accuracy": sum(g == p for _, _, g, p in rows) / n,
            "share_most_frequent_answer_per_question": sum(top_q.values()) / n,
            "questions_one_answer_95pct": sum(1 for q in by_q if top_q[q] >= 0.95 * len(by_q[q])),
            "share_most_frequent_answer_whole_source": whole / n,
            "most_frequent_answer_whole_source": whole_answer,
            "test_majority_per_question_oracle": gold_top / n,
        }
        spec = TRAIN.get(src)
        if spec is None:
            entry["train_prior"] = None
        else:
            prior, ntrain = train_priors(root, *spec)
            covered = [row for row in rows if row[0] in prior and prior[row[0]] in row[1]]
            entry["train_prior"] = {
                "train_source": spec[1],
                "answer_if_one_question": next(iter(prior.values())) if len(prior) == 1 else None,
                "train_questions": len(prior),
                "train_records": sum(ntrain.values()),
                "items_covered": len(covered),
                "accuracy": sum(prior[q] == g for q, _, g, _ in covered) / n,
            }
        out["sources"][src] = entry
    text = json.dumps(out, indent=2)
    print(text)
    if a.json:
        Path(a.json).write_text(text + "\n")


if __name__ == "__main__":
    main()
