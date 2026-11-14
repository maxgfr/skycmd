#!/usr/bin/env python3
"""ECE of each source and a bootstrap interval of the pooled ECE, from an s6 evaluation dump.

docs/s6-gen-prereg.md, amendment 12: the reports give, as reported numbers and not gates, the
ECE of each source and a 95% bootstrap interval of the pooled ECE (1,000 resamples of the items).

The dump of `sky-eval --dump` (scripts/train-s6.sh eval) holds the probabilities of each
question but not its gold option, so the gold is read again from the test records, in the order
and with the filters of `sky-eval` (split `test`, the first `--max` records of each source, at
least two options and one target value per option). Each dump line is checked against its record
(source, type, instruction, options, state).

The ECE is top-label, 15 equal-width bins, as `sky-metrics` (`scores::ece`): the confidence is
the largest probability (first one on ties), bin b covers (b/15, (b+1)/15]. The pooled value is
printed with the binning of `sky-eval` too ([b/15, (b+1)/15)), which wrote the gate value, so
that the two can be compared. The interval is a percentile bootstrap (linear interpolation, as
`sky-metrics::bootstrap::quantile`); the resamples are drawn by Python's `random`, not by the
generator of `sky-metrics`, so its bounds are this script's.

    python3 scripts/s6-ece-by-source.py --dump runs/s6-nano/eval/dump-s6-nano.jsonl --root .
"""

import argparse
import json
import math
import random
from pathlib import Path

TESTS = [
    "data/decisions/typed_decisions_bench.jsonl",
    "data/s4/heldout/zs_massive_en.jsonl",
    "data/s4/heldout/zs_massive_fr.jsonl",
    "data/decisions/when2call.jsonl",
    "data/decisions/notam.jsonl",
]
BINS = 15


def argmax(v):
    best = 0
    for i, x in enumerate(v):
        if x > v[best]:
            best = i
    return best


def bin_metrics(p):
    """sky-metrics: the b with b/bins < p <= (b+1)/bins."""
    edge = lambda b: b / BINS
    b = min(max(math.ceil(p * BINS) - 1, 0), BINS - 1)
    while b > 0 and p <= edge(b):
        b -= 1
    while b + 1 < BINS and p > edge(b + 1):
        b += 1
    return b


def bin_eval(p):
    """sky-eval: floor(p * bins), the last bin closed."""
    return min(int(p * BINS), BINS - 1)


def ece(pairs, binner=bin_metrics):
    acc = [[0.0, 0.0, 0] for _ in range(BINS)]
    for c, h in pairs:
        a = acc[binner(c)]
        a[0] += c
        a[1] += h
        a[2] += 1
    n = sum(a[2] for a in acc)
    if n == 0:
        return float("nan")
    return sum(k / n * abs(sp / k - sy / k) for sp, sy, k in acc if k > 0)


def quantile(sorted_v, q):
    if not sorted_v:
        return float("nan")
    pos = q * (len(sorted_v) - 1)
    lo = math.floor(pos)
    hi = min(lo + 1, len(sorted_v) - 1)
    return sorted_v[lo] + (sorted_v[hi] - sorted_v[lo]) * (pos - lo)


def records(root, max_per_source):
    counts = {}
    for rel in TESTS:
        for line in open(root / rel, encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("split") != "test":
                continue
            src = r.get("src", "?")
            if counts.get(src, 0) >= max_per_source:
                continue
            counts[src] = counts.get(src, 0) + 1
            if r.get("opts_kv") is not None:
                raise SystemExit(f"{rel}: opts_kv records are not handled by this script")
            opts = [o for o in r.get("opts") or [] if isinstance(o, str)]
            target = [x for x in r.get("target") or [] if isinstance(x, (int, float))]
            if len(opts) < 2 or len(target) != len(opts):
                continue
            yield r, opts, target


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dump", required=True)
    ap.add_argument("--root", default=".", help="checkout that holds data/")
    ap.add_argument("--max", type=int, default=2000)
    ap.add_argument("--resamples", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json")
    a = ap.parse_args()

    dump = [json.loads(l) for l in open(a.dump, encoding="utf-8") if l.strip()]
    items = []  # (src, confidence, correct)
    for i, (r, opts, target) in enumerate(records(Path(a.root), a.max)):
        if i >= len(dump):
            raise SystemExit(f"more records than dump lines ({len(dump)})")
        d = dump[i]
        same = (d["src"], d["t"], d["ins"], d["opts"], d["state"]) == (
            r.get("src", "?"), r.get("t"), r.get("ins"), opts, r.get("state"))
        if not same:
            raise SystemExit(f"dump line {i + 1} does not match its record")
        p = d["probs"]
        k = argmax(p)
        items.append((d["src"], p[k], float(k == argmax(target))))
    if len(items) != len(dump):
        raise SystemExit(f"{len(items)} records for {len(dump)} dump lines")

    out = {"dump": a.dump, "n": len(items), "bins": BINS, "sources": {}}
    for src in sorted({s for s, _, _ in items}):
        sub = [(c, h) for s, c, h in items if s == src]
        out["sources"][src] = {
            "n": len(sub),
            "accuracy": sum(h for _, h in sub) / len(sub),
            "mean_confidence": sum(c for c, _ in sub) / len(sub),
            "ece": ece(sub),
        }
    pairs = [(c, h) for _, c, h in items]
    pooled = ece(pairs)
    rng = random.Random(a.seed)
    n = len(pairs)
    draws = sorted(ece([pairs[rng.randrange(n)] for _ in range(n)]) for _ in range(a.resamples))
    out["pooled"] = {
        "ece": pooled,
        "ece_sky_eval_bins": ece(pairs, bin_eval),
        "accuracy": sum(h for _, h in pairs) / n,
        "bootstrap": {"resamples": a.resamples, "seed": a.seed, "level": 0.95,
                      "lo": quantile(draws, 0.025), "hi": quantile(draws, 0.975)},
    }
    text = json.dumps(out, indent=2)
    print(text)
    if a.json:
        Path(a.json).write_text(text + "\n")


if __name__ == "__main__":
    main()
