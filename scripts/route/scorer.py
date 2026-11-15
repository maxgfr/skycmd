"""The scorer of s6-route-nano (docs/s6-nano2-prereg.md 7.2 and 7.3): one function for the model
and for every baseline, from a dump and the slate file it answers.

A dump is `sky-eval --dump-probs` (one line per record: `{src, t, k, probs, target, gold, uid}`)
or a baseline's file in the same format (scripts/route-baselines.py). It is joined with the
slate file by `uid`; every slate must be answered once.

The answer of a slate is the option of highest probability (the first of equal ones, as
`sky-metrics`); *none* is the last option. ECE: 15 equal-width bins of the top probability, a
bin holding (k/15, (k+1)/15], the first [0, 1/15] (`sky-metrics` `calibration_error`). NLL: the
probability of the gold floored at 1e-12.
"""

import json
import math
import random

NLL_FLOOR = 1e-12
ECE_BINS = 15
BOOTSTRAP = 1000


class ScoreError(Exception):
    pass


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def argmax(v):
    best = 0
    for i, x in enumerate(v):
        if x > v[best]:
            best = i
    return best


def option_name(opt):
    """The name of an option `name: description`."""
    return opt.split(": ", 1)[0]


class Item:
    __slots__ = ("uid", "probs", "gold", "pred", "conf", "kind", "request", "opts", "follow", "lang", "src")

    def __init__(self, slate, probs):
        self.uid = slate["uid"]
        self.opts = slate["opts"]
        if len(probs) != len(self.opts):
            raise ScoreError(f"{self.uid}: {len(probs)} probabilities for {len(self.opts)} options")
        self.probs = [float(p) for p in probs]
        self.gold = argmax(slate["target"])
        self.pred = argmax(self.probs)
        self.conf = self.probs[self.pred]
        self.kind = slate.get("kind", "")
        self.request = slate.get("request", "")
        self.follow = slate.get("follow")
        self.lang = slate.get("lang", "en")
        self.src = slate.get("src", "")

    @property
    def none(self):
        return len(self.opts) - 1

    @property
    def correct(self):
        return self.pred == self.gold

    @property
    def nll(self):
        return -math.log(max(self.probs[self.gold], NLL_FLOOR))


def join(dump, slates):
    """Items of a dump and of its slate file (lists of dicts), every slate answered once."""
    by_uid = {}
    for d in dump:
        uid = d.get("uid")
        if uid is None:
            raise ScoreError("a dump line without uid")
        if uid in by_uid:
            raise ScoreError(f"{uid} answered twice")
        by_uid[uid] = d
    items = []
    for s in slates:
        d = by_uid.pop(s["uid"], None)
        if d is None:
            raise ScoreError(f"{s['uid']} not answered")
        if d.get("target") is not None and argmax(d["target"]) != argmax(s["target"]):
            raise ScoreError(f"{s['uid']}: the dump's target is not the slate's")
        items.append(Item(s, d["probs"]))
    if by_uid:
        raise ScoreError(f"{len(by_uid)} answers of no slate (e.g. {next(iter(by_uid))})")
    return items


def load(dump_path, slate_path):
    return join(read_jsonl(dump_path), read_jsonl(slate_path))


def ece_bin(p, bins=ECE_BINS):
    edge = lambda b: b / bins
    b = min(max(math.ceil(p * bins) - 1, 0), bins - 1)
    while b > 0 and p <= edge(b):
        b -= 1
    while b + 1 < bins and p > edge(b + 1):
        b += 1
    return b


def ece(items, bins=ECE_BINS):
    if not items:
        return float("nan")
    acc = [[0.0, 0.0, 0] for _ in range(bins)]
    for it in items:
        b = acc[ece_bin(it.conf, bins)]
        b[0] += it.conf
        b[1] += it.correct
        b[2] += 1
    n = len(items)
    return sum(k / n * abs(sp / k - sy / k) for sp, sy, k in acc if k)


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def accuracy(items):
    return mean(it.correct for it in items)


def at_threshold(items, tau):
    """Accuracy among the answered slates (top probability >= tau) and their share."""
    answered = [it for it in items if it.conf >= tau]
    return {"tau": tau, "accuracy": accuracy(answered), "coverage": len(answered) / len(items) if items else float("nan")}


def risk_coverage(items, points=20):
    """Accuracy of the most confident `c` share of the slates, c = 1/points .. 1."""
    order = sorted(items, key=lambda it: -it.conf)
    out = []
    for k in range(1, points + 1):
        n = max(1, round(len(order) * k / points))
        out.append({"coverage": n / len(order), "accuracy": accuracy(order[:n]), "min_confidence": order[n - 1].conf})
    return out


def tau_ask(items, target=0.95):
    """The smallest top probability at which the accuracy among answered slates is >= target
    (section 3); `None` when no threshold reaches it."""
    for t in sorted({it.conf for it in items}):
        answered = [it for it in items if it.conf >= t]
        if answered and accuracy(answered) >= target:
            return t
    return None


def predicted_names(items):
    """Shares of the predicted tool names and of the gold tool names (none apart)."""
    pred, gold = {}, {}
    for it in items:
        if it.pred != it.none:
            n = option_name(it.opts[it.pred])
            pred[n] = pred.get(n, 0) + 1
        if it.gold != it.none:
            n = option_name(it.opts[it.gold])
            gold[n] = gold.get(n, 0) + 1
    total = len(items)
    return {n: c / total for n, c in pred.items()}, {n: c / total for n, c in gold.items()}


def metrics(items, tau=None):
    """The metrics of 7.2 on one source."""
    if not items:
        raise ScoreError("no item")
    tool_gold = [it for it in items if it.gold != it.none]
    none_gold = [it for it in items if it.gold == it.none]
    none_pred = [it for it in items if it.pred == it.none]
    tp = sum(1 for it in none_pred if it.gold == it.none)
    prec = tp / len(none_pred) if none_pred else float("nan")
    rec = tp / len(none_gold) if none_gold else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if none_pred and none_gold and prec + rec > 0 else float("nan")
    pred, gold = predicted_names(items)
    top = max(pred.items(), key=lambda kv: (kv[1], kv[0]), default=(None, 0.0))
    excess = max(((p - (2 * gold.get(n, 0.0) + 0.01), n) for n, p in pred.items()), default=(float("-inf"), None))
    out = {
        "n": len(items),
        "accuracy": accuracy(items),
        "accuracy_tool_gold": accuracy(tool_gold),
        "accuracy_none_gold": accuracy(none_gold),
        "none_gold_share": len(none_gold) / len(items),
        "none_answer_share": len(none_pred) / len(items),
        "none_precision": prec,
        "none_recall": rec,
        "none_f1": f1,
        "ece": ece(items),
        "nll": mean(it.nll for it in items),
        "mean_confidence": mean(it.conf for it in items),
        "top_predicted_name": {"name": top[0], "share": top[1], "gold_share": gold.get(top[0], 0.0)},
        "largest_name_excess": {"name": excess[1], "excess": excess[0]},
        "risk_coverage": risk_coverage(items),
    }
    if tau is not None:
        out["at_tau_ask"] = at_threshold(items, tau)
    return out


def bootstrap_diff(a, b, stat=accuracy, resamples=BOOTSTRAP, seed=0):
    """The difference stat(a) - stat(b) on the same slates (paired by uid) and its 95%
    percentile interval over `resamples` resamples of the slates."""
    ia = {it.uid: it for it in a}
    ib = {it.uid: it for it in b}
    if set(ia) != set(ib):
        raise ScoreError("the two systems did not answer the same slates")
    uids = sorted(ia)
    rng = random.Random(seed)
    point = stat([ia[u] for u in uids]) - stat([ib[u] for u in uids])
    diffs = []
    for _ in range(resamples):
        pick = [uids[rng.randrange(len(uids))] for _ in uids]
        diffs.append(stat([ia[u] for u in pick]) - stat([ib[u] for u in pick]))
    diffs.sort()
    lo = diffs[int(0.025 * resamples)]
    hi = diffs[min(resamples - 1, int(0.975 * resamples))]
    return {"diff": point, "lo": lo, "hi": hi, "resamples": resamples}


def pair_changes(pairs):
    """7.3: among the requests answered correctly with the gold present, the share whose answer
    changes on the twin slate, where the gold's place holds another distractor and every other
    option keeps its place: an answer changes when its position does."""
    by = {}
    for it in pairs:
        req, side = it.uid.rsplit("/", 1)
        by.setdefault(req, {})[side] = it
    right = [v for v in by.values() if "present" in v and "removed" in v and v["present"].correct]
    changed = sum(1 for v in right if v["removed"].pred != v["present"].pred)
    return {"pairs": len(by), "correct_with_gold": len(right), "changed": changed,
            "share": changed / len(right) if right else float("nan")}


def follow_share(swapped):
    """7.3: on slates whose gold and one distractor swapped descriptions, the share of answers
    that follow the description (the distractor carrying the gold's) and that follow the name."""
    n = len(swapped)
    return {"n": n,
            "follow_description": sum(1 for it in swapped if it.pred == it.follow) / n if n else float("nan"),
            "follow_name": sum(1 for it in swapped if it.pred == it.gold) / n if n else float("nan")}
