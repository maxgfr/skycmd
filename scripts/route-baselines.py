"""The baselines of s6-route-nano (docs/s6-nano2-prereg.md, section 6), fitted on the same
training items as the model and scored on the same slates, with dumps in `sky-eval --dump-probs`
format that scripts/route/scorer.py reads.

    route-baselines.py fit   --system S [--budget B]          (validation and training only)
    route-baselines.py score --system S --split val|test [--model GGUF]

Systems: potion-cos, potion-cos-4bit, static-ft, pair-lr, bm25, minilm-cos, toolrouter,
name-prior. s6-nano v1 is scored by sky-eval (scripts/route-eval.sh).

- cosine systems (potion-cos, potion-cos-4bit, static-ft, minilm-cos) and bm25: one score per
  tool option, the threshold theta for *none*; theta maximises the accuracy on the validation
  slates (v_tools; of equal accuracies, the lowest theta); the probabilities are
  softmax([scores, theta] / T), T minimising the validation NLL;
- potion-cos-4bit: potion-base-2M's token vectors packed to 4 bits (per token, a float16 scale
  and 4-bit codes in -7..7);
- static-ft: model2vec 0.10.0 `StaticModelForPairSimilarity` (InfoNCE, its defaults, on the CPU)
  on the training (request, gold option) pairs; then 4-bit packing and the vocabulary cut to the
  most frequent tokens of the training texts so that the packed vectors hold at most B bytes;
  epochs among 1, 3, 10 by validation accuracy (of equal ones, the fewest);
- pair-lr: a binary SGDClassifier(log_loss, l2) of scikit-learn 1.9.1 on every (request,
  option) pair of every training slate, *none* as its option text; features hashed into B
  columns (character 1- to 5-grams of the request, of the option, and request-word x
  option-word pairs, three namespaces, binary, the row L2-normalised); weights stored as int8;
  alpha among 1e-6, 1e-5, 1e-4 by validation accuracy; softmax of the decision values over a
  slate with a fitted T. The training set does not fit in memory: partial_fit over chunks, with
  the stopping rule of `fit` (tol 1e-3 on the epoch's training loss, 5 epochs without
  improvement, at most 1,000 epochs) and a shuffle per epoch;
- bm25: bm25s (Lucene variant, its defaults) over the tool options of the slate, the request as
  the query;
- toolrouter: its own `scores` and temperature, *none* as its tool `other` with its own
  description;
- name-prior: the option whose name was gold most often in the training slates; *none* when no
  option's name ever was; all its probability on that option.

Every output goes to the data folder: <data>/eval/fitted/<system>/ and
<data>/eval/<split>/<system>/<file>.jsonl. The test split goes through common.slate_file, which
refuses it until runs/s6-nano2/TEST-OPEN exists and the model holds tau_ask.
"""

import argparse
import json
import math
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "route"))
import common  # noqa: E402
import scorer  # noqa: E402

import numpy as np  # noqa: E402

SYSTEMS = ("potion-cos", "potion-cos-4bit", "static-ft", "pair-lr", "bm25", "minilm-cos", "toolrouter", "name-prior")
FILES = {
    "val": ("v_tools", "v_tools_masked", "v_tools_pairs", "v_tools_swapped"),
    "test": ("t_tools", "t_tools_masked", "t_tools_pairs", "t_tools_swapped", "t_metatool", "t_clinc",
             "t_toolrouting", "t_bfcl", "t_when2call", "t_massive"),
}
DEFAULT_BUDGET = 715136  # s6-nano v1's file; the evaluation passes the evaluated file's bytes
SEED = 0


def request_of(rec):
    s = rec["state"]
    return s[len("User: "):] if s.startswith("User: ") else s


def softmax(z):
    z = np.asarray(z, dtype=np.float64)
    e = np.exp(z - z.max())
    return e / e.sum()


# ---------------------------------------------------------------- fitted threshold and temperature

def fit_theta(score_lists, golds):
    """theta for *none* (none when the best tool score is below it): the accuracy on the
    validation slates is highest; of equal ones, the lowest theta."""
    best = [(s[:-1].max() if len(s) > 1 else -np.inf) for s in score_lists]
    arg = [int(np.argmax(s[:-1])) if len(s) > 1 else 0 for s in score_lists]
    none = [len(s) - 1 for s in score_lists]
    cands = sorted(set(best)) + [np.inf]
    top = None
    for t in cands:
        acc = sum((none[i] if best[i] < t else arg[i]) == golds[i] for i in range(len(golds)))
        if top is None or acc > top[1]:
            top = (t, acc)
    return float(top[0]) if np.isfinite(top[0]) else 1e9


def fit_temperature(logit_lists, golds):
    """T minimising the mean NLL of softmax(logits / T) on the validation slates (a grid on
    log T, then a golden-section refinement)."""
    def nll(logt):
        t = math.exp(logt)
        return sum(-math.log(max(softmax(np.asarray(z) / t)[g], scorer.NLL_FLOOR)) for z, g in zip(logit_lists, golds)) / len(golds)
    grid = np.linspace(math.log(1e-4), math.log(100.0), 61)
    vals = [nll(x) for x in grid]
    k = int(np.argmin(vals))
    a, b = grid[max(0, k - 1)], grid[min(len(grid) - 1, k + 1)]
    phi = (math.sqrt(5) - 1) / 2
    c, d = b - phi * (b - a), a + phi * (b - a)
    for _ in range(40):
        if nll(c) < nll(d):
            b = d
        else:
            a = c
        c, d = b - phi * (b - a), a + phi * (b - a)
    return math.exp((a + b) / 2)


# ---------------------------------------------------------------- encoders

class Packed:
    """A model2vec static model whose token vectors are packed to 4 bits (per token: a float16
    scale, codes in -7..7), optionally on a cut vocabulary."""

    def __init__(self, static, keep=None, bits=4):
        self.static = static
        e = np.asarray(static.embedding, dtype=np.float32)
        v = static.tokenizer.get_vocab_size()
        mapping = np.asarray(static.token_mapping) if static.token_mapping is not None else np.arange(v)
        tok = e[mapping]
        if static.weights is not None:
            tok = tok * np.asarray(static.weights, dtype=np.float32)[:, None]
        if bits:
            lim = 2 ** (bits - 1) - 1
            scale = (np.abs(tok).max(axis=1) / lim).astype(np.float16).astype(np.float32)
            safe = np.where(scale == 0, 1, scale)
            codes = np.clip(np.rint(tok / safe[:, None]), -lim, lim)
            tok = codes * scale[:, None]
        self.keep = np.ones(len(tok), bool) if keep is None else np.zeros(len(tok), bool)
        if keep is not None:
            self.keep[np.asarray(sorted(keep), dtype=np.int64)] = True
        self.vec = tok.astype(np.float32)
        self.dim = tok.shape[1]
        self.bits = bits

    def bytes(self):
        per = (self.dim * self.bits + 7) // 8 + 2 if self.bits else self.dim * 4
        return int(self.keep.sum()) * per

    def encode(self, texts):
        out = np.zeros((len(texts), self.dim), np.float32)
        for i, ids in enumerate(self.static.tokenize(list(texts))):
            ids = [t for t in ids if self.keep[t]]
            if ids:
                out[i] = self.vec[ids].mean(axis=0)
        n = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.where(n == 0, 1, n) if self.static.normalize else out


def potion_static():
    from model2vec import StaticModel
    return StaticModel.from_pretrained(common.model_dir("potion-base-2M"))


class Cached:
    def __init__(self, encode):
        self.encode_batch = encode
        self.cache = {}

    def __call__(self, texts):
        todo = [t for t in dict.fromkeys(texts) if t not in self.cache]
        for i in range(0, len(todo), 4096):
            chunk = todo[i:i + 4096]
            for t, e in zip(chunk, self.encode_batch(chunk)):
                self.cache[t] = np.asarray(e, np.float32)
        return np.stack([self.cache[t] for t in texts]) if texts else np.zeros((0, 1), np.float32)


def cosine_scores(recs, enc, theta):
    out = []
    for r in recs:
        q = enc([request_of(r)])[0]
        tools = enc(r["opts"][:-1]) if len(r["opts"]) > 1 else np.zeros((0, len(q)), np.float32)
        out.append(np.concatenate([tools @ q, [theta]]))
    return out


# ---------------------------------------------------------------- the training records

def training_pairs():
    """Unique (request, gold option) pairs of the training slates whose gold is a tool."""
    seen, a, b = set(), [], []
    for r in common.training_records("train"):
        g = scorer.argmax(r["target"])
        if g == len(r["opts"]) - 1:
            continue
        k = (request_of(r), r["opts"][g])
        if k not in seen:
            seen.add(k)
            a.append(k[0])
            b.append(k[1])
    return a, b


def build_info():
    p = os.path.join(common.build_root(), "BUILD.json")
    m = common.load_json(p) if os.path.isfile(p) else {}
    return {"build": p, "build_sha256": common.sha256_file(p) if os.path.isfile(p) else None,
            "provisional": m.get("provisional", True)}


# ---------------------------------------------------------------- pair-lr

TOKEN = re.compile(r"(?u)\b\w\w+\b")


class PairFeatures:
    """Hashed features of (request, option) pairs in `n` columns, three namespaces."""

    def __init__(self, n):
        from sklearn.feature_extraction.text import HashingVectorizer
        self.n = n
        self.hv = HashingVectorizer(analyzer="char", ngram_range=(1, 5), n_features=n, alternate_sign=False,
                                    norm=None, binary=True, lowercase=True)
        self.rows = {}
        self.words = {}

    def _perm(self, idx, ns):
        # a bijection of the columns per namespace (n is not a multiple of these primes)
        k, c = ((1000003, 7919), (999983, 104729))[ns]
        return (idx.astype(np.int64) * k + c) % self.n

    def chars(self, texts, ns):
        todo = [t for t in dict.fromkeys(texts) if (ns, t) not in self.rows]
        if todo:
            x = self.hv.transform(todo).tocsr()
            for i, t in enumerate(todo):
                idx = x.indices[x.indptr[i]:x.indptr[i + 1]]
                self.rows[(ns, t)] = idx.astype(np.int64) if ns == 1 else self._perm(idx, 0)
        return [self.rows[(ns, t)] for t in texts]

    def word_hashes(self, text):
        h = self.words.get(text)
        if h is None:
            from sklearn.utils import murmurhash3_32
            h = np.array(sorted({murmurhash3_32(w, seed=17, positive=True) for w in TOKEN.findall(text.lower())}), dtype=np.int64)
            self.words[text] = h
        return h

    def matrix(self, recs):
        """The rows of every option of every slate, in order, and the slate of each row."""
        import scipy.sparse as sp
        indptr, indices, slate = [0], [], []
        for si, r in enumerate(recs):
            q = request_of(r)
            (rq,) = self.chars([q], 0)
            opts = self.chars(r["opts"], 1)
            wq = self.word_hashes(q)
            for o, ro in zip(r["opts"], opts):
                wo = self.word_hashes(o)
                pairs = self._perm(((wq[:, None] * 1000003 + wo[None, :]) % self.n).ravel(), 1) if len(wq) and len(wo) else np.zeros(0, np.int64)
                row = np.unique(np.concatenate([rq, ro, pairs]))
                indices.append(row)
                indptr.append(indptr[-1] + len(row))
                slate.append(si)
        idx = np.concatenate(indices) if indices else np.zeros(0, np.int64)
        data = np.ones(len(idx), np.float32)
        x = sp.csr_matrix((data, idx.astype(np.int32), np.asarray(indptr, np.int64)), shape=(len(slate), self.n))
        norms = np.sqrt(np.diff(x.indptr)).astype(np.float32)
        x.data /= np.repeat(np.where(norms == 0, 1, norms), np.diff(x.indptr))
        return x, np.asarray(slate)


def labels_of(recs):
    y = []
    for r in recs:
        g = scorer.argmax(r["target"])
        y.extend(1 if i == g else 0 for i in range(len(r["opts"])))
    return np.asarray(y)


def fit_pair_lr(train, alpha, feats, log, chunk=8000):
    from sklearn.linear_model import SGDClassifier
    clf = SGDClassifier(loss="log_loss", penalty="l2", alpha=alpha, random_state=SEED)
    order = np.arange(len(train))
    best, stale, epoch = np.inf, 0, 0
    while epoch < 1000 and stale < 5:
        rng = np.random.default_rng(SEED + epoch)
        rng.shuffle(order)
        loss, n = 0.0, 0
        for c in range(0, len(order), chunk):
            recs = [train[i] for i in order[c:c + chunk]]
            x, _ = feats.matrix(recs)
            y = labels_of(recs)
            perm = rng.permutation(len(y))
            x, y = x[perm], y[perm]
            if hasattr(clf, "coef_"):
                p = 1 / (1 + np.exp(-np.clip(clf.decision_function(x), -30, 30)))
                loss += float(-(y * np.log(np.maximum(p, 1e-15)) + (1 - y) * np.log(np.maximum(1 - p, 1e-15))).sum())
            else:
                loss += math.log(2) * len(y)
            n += len(y)
            clf.partial_fit(x, y, classes=np.array([0, 1]))
        loss /= max(n, 1)
        epoch += 1
        if loss > best - 1e-3:
            stale += 1
        else:
            stale = 0
        best = min(best, loss)
        log(f"pair-lr alpha {alpha:g}: epoch {epoch}, training loss {loss:.5f}")
    w = clf.coef_.ravel().astype(np.float64)
    s = np.abs(w).max() / 127 if np.abs(w).max() > 0 else 1.0
    q = np.clip(np.rint(w / s), -127, 127).astype(np.int8)
    return {"q": q, "scale": float(s), "bias": float(clf.intercept_[0]), "epochs": epoch}


def pair_lr_scores(recs, model, feats):
    x, slate = feats.matrix(recs)
    d = x @ (model["q"].astype(np.float32) * model["scale"]) + model["bias"]
    out, k = [], 0
    for r in recs:
        n = len(r["opts"])
        out.append(np.asarray(d[k:k + n], np.float64))
        k += n
    return out


# ---------------------------------------------------------------- the systems

class System:
    """A fitted system: `scores(recs)` (one score per option, *none* last) and `probs`."""

    def __init__(self, name, fitted):
        self.name, self.p = name, fitted
        self.enc = None

    def scores(self, recs):
        n = self.name
        if n in ("potion-cos", "potion-cos-4bit", "static-ft", "minilm-cos"):
            return cosine_scores(recs, self.encoder(), self.p["theta"])
        if n == "bm25":
            return bm25_scores(recs, self.p["theta"])
        if n == "pair-lr":
            return pair_lr_scores(recs, self.encoder(), self.feats())
        raise ValueError(n)

    def feats(self):
        if not hasattr(self, "_feats"):
            self._feats = PairFeatures(self.p["budget"])
        return self._feats

    def encoder(self):
        if self.enc is not None:
            return self.enc
        n = self.name
        if n == "potion-cos":
            m = potion_static()
            self.enc = Cached(lambda t: m.encode(t, show_progress_bar=False))
        elif n == "potion-cos-4bit":
            self.enc = Cached(Packed(potion_static()).encode)
        elif n == "static-ft":
            from model2vec import StaticModel
            d = os.path.join(fitted_dir(n), "model")
            keep = np.load(os.path.join(fitted_dir(n), "keep.npy"))
            self.enc = Cached(Packed(StaticModel.from_pretrained(d), keep=keep).encode)
        elif n == "minilm-cos":
            from sentence_transformers import SentenceTransformer
            m = SentenceTransformer(common.model_dir("all-MiniLM-L6-v2"), device="cpu")
            self.enc = Cached(lambda t: m.encode(t, batch_size=256, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False))
        elif n == "pair-lr":
            z = np.load(os.path.join(fitted_dir(n), "weights.npz"))
            self.enc = {"q": z["q"], "scale": float(z["scale"]), "bias": float(z["bias"])}
        return self.enc

    def probs(self, recs):
        n = self.name
        if n == "toolrouter":
            return toolrouter_probs(recs)
        if n == "name-prior":
            return name_prior_probs(recs, self.p["counts"])
        return [softmax(s / self.p["temperature"]) for s in self.scores(recs)]


def bm25_scores(recs, theta):
    import bm25s
    out = []
    for r in recs:
        docs = r["opts"][:-1]
        if not docs:
            out.append(np.array([theta]))
            continue
        toks = bm25s.tokenize(docs, show_progress=False)
        q = bm25s.tokenize([request_of(r)], return_ids=False, show_progress=False)[0]
        if not toks.vocab or not q:
            # no token in the documents or in the request: every score is 0
            s = np.zeros(len(docs))
        else:
            ret = bm25s.BM25(method="lucene")
            ret.index(toks, show_progress=False)
            s = ret.get_scores(q)
        out.append(np.concatenate([np.asarray(s, np.float64), [theta]]))
    return out


_ROUTER = None


def toolrouter_probs(recs):
    global _ROUTER
    if _ROUTER is None:
        sys.path.insert(0, common.model_dir("toolrouter"))
        from toolrouter import DEFAULT_CRITERIA, ToolRouter
        _ROUTER = (ToolRouter(os.path.join(common.model_dir("toolrouter"), "toolrouter.npz")), DEFAULT_CRITERIA["other"])
    router, other = _ROUTER
    out = []
    for r in recs:
        crit, seen = {}, set()
        for i, o in enumerate(r["opts"][:-1]):
            name, _, desc = o.partition(": ")
            key = name
            while key in seen or key == "other":
                key = f"{key}#{i}"
            seen.add(key)
            crit[key] = desc
        crit["other"] = other
        z = router.scores(request_of(r), crit) / router.T
        out.append(softmax(z))
    return out


def name_prior_probs(recs, counts):
    out = []
    for r in recs:
        c = [counts.get(scorer.option_name(o), 0) for o in r["opts"][:-1]]
        p = np.zeros(len(r["opts"]))
        p[int(np.argmax(c)) if c and max(c) > 0 else len(r["opts"]) - 1] = 1.0
        out.append(p)
    return out


# ---------------------------------------------------------------- fit and score

def fitted_dir(name):
    return os.path.join(common.eval_root(), "fitted", name)


def log(msg):
    print(f"[route-baselines] {msg}", file=sys.stderr, flush=True)


def val_slates():
    return common.read_jsonl(common.slate_file("val", "v_tools"))


def fit(name, budget):
    out = fitted_dir(name)
    os.makedirs(out, exist_ok=True)
    val = val_slates()
    golds = [scorer.argmax(r["target"]) for r in val]
    params = {"system": name, "budget": budget, **build_info(), "fitted_on": "v_tools (validation slates); training records"}
    if name in ("potion-cos", "potion-cos-4bit", "minilm-cos", "bm25"):
        s = System(name, {"theta": 0.0})
        raw = s.scores(val) if name != "bm25" else bm25_scores(val, 0.0)
        params["theta"] = fit_theta(raw, golds)
        logits = [np.concatenate([x[:-1], [params["theta"]]]) for x in raw]
        params["temperature"] = fit_temperature(logits, golds)
        if name == "potion-cos-4bit":
            params["bytes"] = Packed(potion_static()).bytes()
        if name == "potion-cos":
            # the packed encoder without packing is model2vec's own: a check of Packed
            st = potion_static()
            t = [request_of(r) for r in val[:200]]
            assert np.abs(Packed(st, bits=0).encode(t) - st.encode(t, show_progress_bar=False)).max() < 1e-5
    elif name == "static-ft":
        params.update(fit_static_ft(val, golds, budget, out))
    elif name == "pair-lr":
        params.update(fit_pair_lr_alpha(val, golds, budget, out))
    elif name == "name-prior":
        counts = {}
        for r in common.training_records("train"):
            g = scorer.argmax(r["target"])
            if g != len(r["opts"]) - 1:
                n = scorer.option_name(r["opts"][g])
                counts[n] = counts.get(n, 0) + 1
        params["counts"] = counts
    elif name == "toolrouter":
        params["model"] = common.fetched(common.model_dir("toolrouter"))["commit"]
    s = System(name, params)
    acc = scorer.accuracy([scorer.Item(r, p) for r, p in zip(val, s.probs(val))])
    params["validation_accuracy_v_tools"] = acc
    common.write_json(os.path.join(out, "params.json"), params)
    log(f"{name}: fitted, v_tools accuracy {acc:.4f}")


def fit_static_ft(val, golds, budget, out):
    import torch
    from model2vec.train import StaticModelForPairSimilarity
    torch.set_num_threads(4)
    a, b = training_pairs()
    texts = list(dict.fromkeys(a + b))
    best = None
    tried = {}
    for epochs in (1, 3, 10):
        m = StaticModelForPairSimilarity.from_pretrained(path=common.model_dir("potion-base-2M"))
        m.fit(text_a=a, text_b=b, max_epochs=epochs, device="cpu")
        st = m.to_static_model()
        counts = np.zeros(st.tokenizer.get_vocab_size(), np.int64)
        for ids in st.tokenize(texts):
            np.add.at(counts, ids, 1)
        full = Packed(st)
        per = full.bytes() // len(full.keep)
        n_keep = min(int((counts > 0).sum()), budget // per)
        keep = np.argsort(-counts, kind="stable")[:n_keep]
        enc = Cached(Packed(st, keep=keep).encode)
        raw = cosine_scores(val, enc, 0.0)
        theta = fit_theta(raw, golds)
        acc = sum((len(s) - 1 if (len(s) == 1 or s[:-1].max() < theta) else int(np.argmax(s[:-1]))) == g for s, g in zip(raw, golds)) / len(golds)
        tried[epochs] = acc
        log(f"static-ft: {epochs} epochs, {n_keep} tokens kept, v_tools accuracy {acc:.4f}")
        if best is None or acc > best[0]:
            st.save_pretrained(os.path.join(out, "model"))
            np.save(os.path.join(out, "keep.npy"), keep)
            logits = [np.concatenate([s[:-1], [theta]]) for s in raw]
            best = (acc, {"epochs": epochs, "theta": theta, "temperature": fit_temperature(logits, golds),
                          "tokens_kept": n_keep, "bytes": Packed(st, keep=keep).bytes(), "pairs": len(a)})
    best[1]["accuracy_by_epochs"] = tried
    return best[1]


def fit_pair_lr_alpha(val, golds, budget, out):
    train = common.training_records("train")
    feats = PairFeatures(budget)
    best, tried = None, {}
    for alpha in (1e-6, 1e-5, 1e-4):
        m = fit_pair_lr(train, alpha, feats, log)
        raw = pair_lr_scores(val, m, feats)
        acc = sum(int(np.argmax(s)) == g for s, g in zip(raw, golds)) / len(golds)
        tried[f"{alpha:g}"] = {"accuracy": acc, "epochs": m["epochs"]}
        log(f"pair-lr: alpha {alpha:g}, v_tools accuracy {acc:.4f}")
        if best is None or acc > best[0]:
            np.savez(os.path.join(out, "weights.npz"), q=m["q"], scale=m["scale"], bias=m["bias"])
            best = (acc, {"alpha": alpha, "temperature": fit_temperature(raw, golds), "epochs": m["epochs"],
                          "columns": budget, "bytes": budget})
    best[1]["by_alpha"] = tried
    best[1]["training_slates"] = len(train)
    return best[1]


def score(name, split, model):
    p = common.load_json(os.path.join(fitted_dir(name), "params.json"))
    s = System(name, p)
    out = os.path.join(common.eval_root(), split, name)
    os.makedirs(out, exist_ok=True)
    for f in FILES[split]:
        try:
            path = common.slate_file(split, f, model)
        except common.Refused as e:
            if split == "test":
                raise
            log(f"{name}: {f}: {e}")
            continue
        recs = common.read_jsonl(path)
        rows = []
        for r, pr in zip(recs, s.probs(recs)):
            rows.append({"src": r["src"], "t": "choice", "k": len(r["opts"]), "probs": [float(x) for x in pr],
                         "target": r["target"], "gold": scorer.argmax(r["target"]), "uid": r["uid"]})
        common.write_jsonl(os.path.join(out, f"{f}.jsonl"), rows)
        log(f"{name}: {f} ({len(rows)} slates)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=("fit", "score"))
    ap.add_argument("--system", required=True, choices=SYSTEMS)
    ap.add_argument("--split", choices=common.SPLITS, default="val")
    ap.add_argument("--model", help="the trained GGUF (the test needs its tau_ask)")
    ap.add_argument("--budget", type=int, default=DEFAULT_BUDGET, help="the byte budget B of static-ft and pair-lr")
    a = ap.parse_args()
    try:
        if a.action == "fit":
            fit(a.system, a.budget)
        else:
            score(a.system, a.split, a.model)
    except common.Refused as e:
        print(f"refused: {e}", file=sys.stderr)
        sys.exit(3)


if __name__ == "__main__":
    main()
