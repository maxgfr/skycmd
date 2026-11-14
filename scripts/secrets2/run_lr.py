#!/usr/bin/env python3
"""The two trained baselines of s6-secrets v2 that read the model's own state text
(docs/s6-secrets2-prereg.md, section 6). Run with the interpreter of their pinned environment
(scripts/secrets2/setup_tools.py envs):

  <data>/eval/envs/lr/bin/python -I scripts/secrets2/run_lr.py train --tables BUILD/tables --out DIR
  <data>/eval/envs/lr/bin/python -I scripts/secrets2/run_lr.py score --model DIR --table T.jsonl
        --corpus creddata:validation --out FLAGS
  (the same two commands with `--kind potion` and <data>/eval/envs/potion/bin/python)

lr (gated by G3): the size-matched logistic regression.
  input      hashed character 1- to 5-grams of the state text (HashingVectorizer: analyzer
             "char", 2^18 columns, signed hashing, L2-normalised rows, case kept), plus the 12
             hand features of section 4.2, each standardised with its mean and deviation on the
             training candidates
  model      scikit-learn 1.9.1 SGDClassifier(loss="log_loss", penalty="l2", alpha,
             random_state=1234), every other setting at its default
  size       the 2^18 n-gram weights are quantised to int8 after training (one scale: the
             largest absolute weight is 127): 262,144 bytes. The 12 feature weights, the
             intercept and the 24 standardisation numbers stay in float32 (148 bytes)
  alpha      1e-7, 1e-6, 1e-5 or 1e-4: the one whose quantised model has the highest F1 on the
             CredData validation candidates, each at its own F1-maximising threshold (the first
             of the list among equals)
  threshold  the one that maximises that F1 (the highest among equals)

potion (reported only): model2vec 0.10.0, potion-base-2M at its pinned commit embeds the state
text (64 dimensions); scikit-learn LogisticRegression(C=1.0, max_iter=1000) on top; the
threshold chosen the same way. Nothing else is tuned.

Training candidates: the three training tables of the data side (creddata with its weak
negatives, issues, synth), every candidate once, weighted so that the three sources weigh
0.45 / 0.35 / 0.20 as in the model's training mix (section 3.4).

`score` writes one flag per candidate of the table, with its score and the chosen threshold.

  .../envs/lr/bin/python -I scripts/secrets2/run_lr.py probe --tables BUILD/tables --out FILE
the value-only probe of section 9: LogisticRegression(max_iter=1000) on the eight features of
section 4.2 that are read from the value alone (length, entropy, characters per token, the four
character counts, the longest word run), standardised; trained on the labelled CredData
training candidates (no weak negative), scored on the CredData validation candidates. A high
score means the shape of a value alone separates T from F and X, i.e. CredData's obfuscation
is readable. Writes FILE (PR-AUC, ROC-AUC, the best F1 over thresholds).
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import candidates  # noqa: E402
import common  # noqa: E402
import scorer  # noqa: E402

ALPHAS = (1e-7, 1e-6, 1e-5, 1e-4)
N_HASH = 1 << 18
SEED = 1234
MIX = {"creddata": 0.45, "issues": 0.35, "synth": 0.20}


def features_of(rows):
    import numpy as np

    return np.array([[float(r["features"][k]) for k in candidates.FEATURES] for r in rows], dtype=np.float64)


def mix_weights(rows):
    """One weight per training candidate: its source's share of the mix over its source's count,
    scaled to a mean of 1."""
    import numpy as np

    counts = {}
    for r in rows:
        counts[r["source"]] = counts.get(r["source"], 0) + 1
    unknown = set(counts) - set(MIX)
    if unknown:
        raise common.Refused(f"training candidates of unknown sources: {sorted(unknown)}")
    total = sum(MIX[s] for s in counts)
    w = np.array([MIX[r["source"]] / total / counts[r["source"]] for r in rows], dtype=np.float64)
    return w * len(rows)


def hashed(states):
    from sklearn.feature_extraction.text import HashingVectorizer

    return HashingVectorizer(analyzer="char", ngram_range=(1, 5), n_features=N_HASH, alternate_sign=True, norm="l2", lowercase=False).transform(states)


def quantise(w):
    """int8 weights and their scale: the largest absolute weight becomes 127."""
    import numpy as np

    top = float(np.max(np.abs(w))) if w.size else 0.0
    scale = top / 127.0 if top > 0 else 1.0
    return np.clip(np.rint(w / scale), -127, 127).astype(np.int8), scale


class Lr:
    """The quantised model: scores = sigmoid(scale * X_hash . q + X_feat . w + b)."""

    def __init__(self, q, scale, w_feat, intercept, mean, std):
        self.q, self.scale, self.w_feat, self.intercept, self.mean, self.std = q, scale, w_feat, intercept, mean, std

    def scores(self, rows):
        import numpy as np

        z = hashed([r["state"] for r in rows]) @ (self.q.astype(np.float32) * np.float32(self.scale))
        z = z + ((features_of(rows) - self.mean) / self.std) @ self.w_feat + self.intercept
        return 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))

    def save(self, path):
        import numpy as np

        np.savez(path, q=self.q, scale=np.float32(self.scale), w_feat=self.w_feat.astype(np.float32), intercept=np.float32(self.intercept),
                 mean=self.mean.astype(np.float32), std=self.std.astype(np.float32))

    @classmethod
    def load(cls, path):
        import numpy as np

        z = np.load(path)
        return cls(z["q"], float(z["scale"]), z["w_feat"].astype(np.float64), float(z["intercept"]), z["mean"].astype(np.float64), z["std"].astype(np.float64))


def fit_lr(train, alpha):
    import numpy as np
    from scipy.sparse import csr_matrix, hstack
    from sklearn.linear_model import SGDClassifier

    f = features_of(train)
    mean, std = f.mean(axis=0), f.std(axis=0)
    std[std == 0] = 1.0
    x = hstack([hashed([r["state"] for r in train]), csr_matrix((f - mean) / std)]).tocsr()
    y = np.array([int(r["real"]) for r in train])
    clf = SGDClassifier(loss="log_loss", penalty="l2", alpha=alpha, random_state=SEED)
    clf.fit(x, y, sample_weight=mix_weights(train))
    w = clf.coef_[0]
    q, scale = quantise(w[:N_HASH])
    return Lr(q, scale, w[N_HASH:].copy(), float(clf.intercept_[0]), mean, std), int(clf.n_iter_)


class Potion:
    def __init__(self, clf, model_dir):
        self.clf, self.model_dir = clf, model_dir
        self._embedder = None

    def embed(self, rows):
        if self._embedder is None:
            from model2vec import StaticModel

            self._embedder = StaticModel.from_pretrained(self.model_dir)
        return self._embedder.encode([r["state"] for r in rows])

    def scores(self, rows):
        return self.clf.predict_proba(self.embed(rows))[:, 1]


def potion_dir():
    import setup_tools

    for a in setup_tools.pins()["artifacts"]:
        if a["group"] == "potion" and not setup_tools.check_artifact(a):
            raise common.Refused(f"{a['file']} of potion-base-2M is not fetched")
    return setup_tools.downloads_dir("potion")


def fit_potion(train):
    import numpy as np
    from sklearn.linear_model import LogisticRegression

    p = Potion(LogisticRegression(C=1.0, max_iter=1000), potion_dir())
    p.clf.fit(p.embed(train), np.array([int(r["real"]) for r in train]), sample_weight=mix_weights(train))
    return p


def items_of(model, rows):
    return [(float(p), bool(int(r["real"]))) for p, r in zip(model.scores(rows), rows)]


def table_files(tables):
    return {f"{s}-{k}.jsonl": os.path.join(tables, f"{s}-{k}.jsonl") for s in MIX for k in ("train", "val")}


def check_build(tables, files):
    """sha256 of the BUILD.json next to the tables; the tables must be the ones it names."""
    path = os.path.join(os.path.dirname(os.path.abspath(tables)), "BUILD.json")
    if not os.path.isfile(path):
        return None
    outputs = common.load_json(path)["outputs"]
    for name, p in files.items():
        want = outputs.get(f"tables/{name}", {}).get("sha256")
        if want != common.sha256_file(p):
            raise common.Refused(f"{p} is not the table BUILD.json names (sha256 {want})")
    return common.sha256_file(path)


def train(tables, out, kind):
    files = table_files(tables)
    rows = [r for s in MIX for r in candidates.read_table(files[f"{s}-train.jsonl"])]
    for r in rows:
        if r.get("split") != "train" or r.get("source") not in MIX:
            raise common.Refused(f"{r['id']} is not a training candidate")
    val = list(candidates.read_table(files["creddata-val.jsonl"]))
    for r in val:
        if r.get("split") != "val" or r.get("source") != "creddata":
            raise common.Refused(f"{r['id']} is not a CredData validation candidate")
    os.makedirs(out, exist_ok=True)
    report = {"baseline": kind, "seed": SEED, "mix": MIX, "build_json_sha256": check_build(tables, files),
              "training_candidates": {s: sum(1 for r in rows if r["source"] == s) for s in MIX},
              "weak_negatives": sum(1 for r in rows if r.get("weak")), "positives": sum(int(r["real"]) for r in rows),
              "validation_candidates": {"n": len(val), "positives": sum(int(r["real"]) for r in val), "what": "tables/creddata-val.jsonl"},
              "tables": {name: {"bytes": os.path.getsize(p), "sha256": common.sha256_file(p)} for name, p in sorted(files.items())}}
    with common.Clock() as clock:
        if kind == "lr":
            import sklearn

            report["scikit_learn"] = sklearn.__version__
            report["features"] = list(candidates.FEATURES)
            best = None
            report["alphas"] = []
            for alpha in ALPHAS:
                model, n_iter = fit_lr(rows, alpha)
                t, f1 = scorer.best_f1_threshold(items_of(model, val))
                report["alphas"].append({"alpha": alpha, "epochs": n_iter, "validation_f1": f1, "threshold": t,
                                         "nonzero_int8_weights": int((model.q != 0).sum())})
                if best is None or f1 > best[1]:
                    best = (alpha, f1, t, model)
            alpha, f1, threshold, model = best
            path = os.path.join(out, "lr-model.npz")
            model.save(path)
            report.update({"alpha": alpha, "int8_weights": N_HASH, "int8_bytes": int(model.q.nbytes), "float32_numbers": 12 + 1 + 24})
        else:
            import pickle

            import model2vec
            import sklearn

            report["scikit_learn"], report["model2vec"] = sklearn.__version__, model2vec.__version__
            model = fit_potion(rows)
            threshold, f1 = scorer.best_f1_threshold(items_of(model, val))
            path = os.path.join(out, "potion-lr.pkl")
            with open(path, "wb") as f:
                pickle.dump(model.clf, f)
            report["embedder"] = {"name": "potion-base-2M", "bytes": os.path.getsize(os.path.join(model.model_dir, "model.safetensors"))}
    report.update({"threshold": threshold, "validation_f1": f1, "model_file": {"name": os.path.basename(path), "bytes": os.path.getsize(path), "sha256": common.sha256_file(path)},
                   "cpu_seconds": round(clock.own + clock.children, 1)})
    common.write_json(os.path.join(out, f"{kind}.json"), report)
    return report


VALUE_FEATURES = ("len", "entropy", "chars_per_token", "digits", "upper", "lower", "other", "word_run")


def probe(tables, out):
    import numpy as np
    from sklearn.linear_model import LogisticRegression

    files = table_files(tables)
    train_rows = [r for r in candidates.read_table(files["creddata-train.jsonl"]) if not r.get("weak")]
    val = list(candidates.read_table(files["creddata-val.jsonl"]))
    if any(r.get("split") != "train" for r in train_rows) or any(r.get("split") != "val" for r in val):
        raise common.Refused("the probe reads the CredData training and validation tables only")
    x = lambda rows: np.array([[float(r["features"][k]) for k in VALUE_FEATURES] for r in rows])
    y = lambda rows: np.array([int(r["real"]) for r in rows])
    mean, std = x(train_rows).mean(axis=0), x(train_rows).std(axis=0)
    std[std == 0] = 1.0
    clf = LogisticRegression(max_iter=1000).fit((x(train_rows) - mean) / std, y(train_rows))
    p = clf.predict_proba((x(val) - mean) / std)[:, 1]
    items = [(float(a), bool(b)) for a, b in zip(p, y(val))]
    pos = [a for a, b in items if b]
    neg = [a for a, b in items if not b]
    wins = sum((a > b) + 0.5 * (a == b) for a in pos for b in neg)
    t, f1 = scorer.best_f1_threshold(items)
    report = {"what": "value-only probe (section 9): logistic regression on the value's own features", "features": list(VALUE_FEATURES),
              "training_candidates": len(train_rows), "training_positives": int(y(train_rows).sum()), "validation_candidates": len(val),
              "validation_positives": len(pos), "pr_auc": scorer.pr_auc(items), "roc_auc": wins / (len(pos) * len(neg)) if pos and neg else None,
              "best_f1": f1, "best_f1_threshold": t, "share_of_positives": len(pos) / len(val) if val else None}
    common.write_json(out, report)
    return report


def load(model_dir, kind):
    report = common.load_json(os.path.join(model_dir, f"{kind}.json"))
    path = os.path.join(model_dir, report["model_file"]["name"])
    if common.sha256_file(path) != report["model_file"]["sha256"]:
        raise common.Refused(f"{path} is not the model {kind}.json describes")
    if kind == "lr":
        return Lr.load(path), report
    import pickle

    with open(path, "rb") as f:  # written by this script, in the data folder
        return Potion(pickle.load(f), potion_dir()), report


def score(model_dir, kind, table, corpus_name, out, system=None):
    model, report = load(model_dir, kind)
    rows = list(candidates.read_table(table))
    with common.Clock() as clock:
        scores = model.scores(rows) if rows else []
    name = system or ("lr" if kind == "lr" else "potion-lr")
    return candidates.write_scored(out, name, corpus_name, [(r["id"], float(p)) for r, p in zip(rows, scores)], report["threshold"], {
        "version": (f"scikit-learn {report['scikit_learn']}, SGDClassifier alpha {report['alpha']}, 2^18 int8 weights" if kind == "lr"
                    else f"model2vec {report['model2vec']} potion-base-2M, scikit-learn {report['scikit_learn']} LogisticRegression"),
        "how": "the state text of each candidate; threshold maximising F1 on the CredData validation candidates",
        "table": {"name": os.path.basename(table), "sha256": common.sha256_file(table), "candidates": len(rows)},
        "cpu_seconds": round(clock.own + clock.children, 1), "wall_seconds": round(clock.wall, 1)})


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("what", choices=("train", "score", "probe"))
    ap.add_argument("--kind", choices=("lr", "potion"), default="lr")
    ap.add_argument("--tables")
    ap.add_argument("--model")
    ap.add_argument("--table")
    ap.add_argument("--corpus")
    ap.add_argument("--system")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    try:
        if a.what == "train":
            r = train(a.tables, a.out, a.kind)
            print(json.dumps({k: v for k, v in r.items() if k != "tables"}, indent=1))
        elif a.what == "probe":
            print(json.dumps(probe(a.tables, a.out), indent=1))
        else:
            common.corpus(a.corpus)  # the guard of the split
            meta = score(a.model, a.kind, a.table, a.corpus, a.out, a.system)
            common.log(f"{meta['system']}: {meta['flags']} candidates scored on {meta['corpus']}, threshold {meta['threshold']}")
    except common.Refused as e:
        common.log(f"run_lr: refused: {e}")
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
