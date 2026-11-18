"""The embeddings of the s6-route-nano build (docs/s6-nano2-prereg.md 4.2 and 4.4).

Reads <build>/stage/texts.jsonl (one `name: description` per tool of the training groups and of
the tool test, written by `sky-data route requests`) and writes, one float32 row per line, in
its order:

    <build>/emb/minilm.npy   all-MiniLM-L6-v2 (pinned commit): the clustering of 4.2 and the
                             test distractor rule of 4.4 (cosine >= 0.8 to the gold excluded)
    <build>/emb/potion.npy   potion-base-2M (pinned commit): the nearest tools of the hard
                             training slates (4.4)
    <build>/emb/meta.json    the sha256 of the texts file, the models' commits, the versions

Neither model sees a request. Run in the pinned environment of scripts/route/env:
    UV_PROJECT_ENVIRONMENT=<data>/envs/route uv run --frozen --project scripts/route/env \
        python -I scripts/route-embed.py [--build <build>] [--threads 4]

s6-route v5 (docs/s6-route5-prereg.md 4.1), `--requests --v2 <v2 build> --build <v5 build>`: the
training and validation requests of v2's plan (<v2>/stage/plan.jsonl, one row per request with a
hard slate, in its order) embedded by potion-base-2M, the same static embedding and the same text
as `potion-cos` (scripts/route-baselines.py: the record's state without `User: `, that is the
request trimmed), L2-normalised as the tools' rows:

    <v5 build>/emb/potion-requests.npy        one float32 row per request of the plan
    <v5 build>/emb/potion-requests.meta.json  the sha256 of the plan and of its ids, the counts by
                                              split, the model's commit, the versions

The guard refuses the whole file, before any model is loaded, when a row is not a training or
validation request (its split, its group, a fresh test's id); it never writes into v2's folder
and never touches <v5 build>/emb/meta.json. MiniLM is not run in this mode.
"""

import argparse
import hashlib
import importlib.metadata as md
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "route"))
import common  # noqa: E402

REQUESTS_NPY = "potion-requests.npy"
REQUESTS_META = "potion-requests.meta.json"
SPLITS = ("train", "val")


def minilm(texts, threads):
    import torch
    from sentence_transformers import SentenceTransformer

    torch.set_num_threads(threads)
    m = SentenceTransformer(common.model_dir("all-MiniLM-L6-v2"), device="cpu")
    return m.encode(texts, batch_size=256, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)


def potion_model():
    from model2vec import StaticModel

    return StaticModel.from_pretrained(common.model_dir("potion-base-2M"))


def potion(texts, model=None):
    import numpy as np

    m = model if model is not None else potion_model()
    e = np.asarray(m.encode(texts, show_progress_bar=False)).astype(np.float32)
    n = np.linalg.norm(e, axis=1, keepdims=True)
    return e / np.where(n == 0, 1, n)


def requests_to_embed(rows):
    """The ids, splits and texts of the rows of v2's plan, after the guard of 4.1 (and 12): every
    row must be a training or validation request (split train or val, a group of the training
    groups, not a fresh test's `toucan:` id); one row that is not refuses the whole file. The text
    is the one `potion-cos` embeds: the state `User: <request>` without its prefix, the request
    trimmed."""
    ids, splits, texts = [], [], []
    for i, r in enumerate(rows):
        rid, split, group, user = r.get("id"), r.get("split"), r.get("group"), r.get("user")
        if not isinstance(rid, str) or not isinstance(user, str):
            raise common.Refused(f"row {i}: not a request of v2's plan")
        if split not in SPLITS:
            raise common.Refused(f"{rid}: of the {split} split; only training and validation requests are embedded (docs/s6-route5-prereg.md 4.1)")
        if group not in ("tools", "when2call", "intents"):
            raise common.Refused(f"{rid}: of the group {group}; only the training groups' requests are embedded")
        if rid.startswith("toucan:"):
            raise common.Refused(f"{rid}: a fresh test's request is never embedded")
        ids.append(rid)
        splits.append(split)
        texts.append(user.strip())
    if len(set(ids)) != len(ids):
        raise common.Refused("a request repeats in the plan")
    return ids, splits, texts


def embed_requests(v2, build, model=None):
    """4.1: <build>/emb/potion-requests.npy and its meta from <v2>/stage/plan.jsonl."""
    import numpy as np

    if os.path.realpath(build) == os.path.realpath(v2):
        raise common.Refused(f"the request embeddings go to v5's build, never to v2's ({v2})")
    src = os.path.join(v2, "stage", "plan.jsonl")
    ids, splits, texts = requests_to_embed(common.read_jsonl(src))
    out = os.path.join(build, "emb")
    os.makedirs(out, exist_ok=True)
    e = np.ascontiguousarray(potion(texts, model), dtype="<f4")
    assert e.shape[0] == len(texts), e.shape
    path = os.path.join(out, REQUESTS_NPY)
    np.save(path, e)
    meta = {
        "about": "s6-route v5 (docs/s6-route5-prereg.md 4.1): the training and validation requests of v2's plan, "
                 "embedded by potion-base-2M as potion-cos embeds a request",
        "source": src, "source_sha256": common.sha256_file(src),
        "ids_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest(),
        "rows": int(e.shape[0]), "dim": int(e.shape[1]), "sha256": common.sha256_file(path),
        "by split": {s: splits.count(s) for s in SPLITS},
    }
    if model is None:
        f = common.fetched(common.model_dir("potion-base-2M"))
        meta["model"] = {"repo": f["repo"], "commit": f["commit"]}
        meta["versions"] = {p: md.version(p) for p in ("model2vec", "numpy")}
    common.write_json(os.path.join(out, REQUESTS_META), meta)
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", default=common.build_root())
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--requests", action="store_true",
                    help="s6-route v5: the training and validation requests of --v2's plan, into --build")
    ap.add_argument("--v2", help="--requests: v2's build folder (read only)")
    a = ap.parse_args()
    if a.requests:
        if not a.v2:
            sys.exit("--requests needs --v2 <v2 build>")
        try:
            meta = embed_requests(a.v2, a.build)
        except common.Refused as e:
            print(f"refused: {e}", file=sys.stderr)
            sys.exit(3)
        print(json.dumps({k: meta[k] for k in ("rows", "dim", "by split")}))
        return
    import numpy as np

    src = os.path.join(a.build, "stage", "texts.jsonl")
    rows = common.read_jsonl(src)
    texts = [r["text"] for r in rows]
    out = os.path.join(a.build, "emb")
    os.makedirs(out, exist_ok=True)
    files = {}
    for name, f in (("minilm", lambda: minilm(texts, a.threads)), ("potion", lambda: potion(texts))):
        e = np.ascontiguousarray(f(), dtype="<f4")
        assert e.shape[0] == len(texts), (name, e.shape)
        path = os.path.join(out, f"{name}.npy")
        np.save(path, e)
        files[name] = {"rows": int(e.shape[0]), "dim": int(e.shape[1]), "sha256": common.sha256_file(path)}
    models = {}
    for name in ("all-MiniLM-L6-v2", "potion-base-2M"):
        f = common.fetched(common.model_dir(name))
        models[name] = {"repo": f["repo"], "commit": f["commit"]}
    common.write_json(os.path.join(out, "meta.json"), {
        "texts": src, "texts_sha256": common.sha256_file(src), "rows": len(texts),
        "files": files, "models": models, "threads": a.threads,
        "versions": {p: md.version(p) for p in ("torch", "sentence-transformers", "model2vec", "numpy")},
    })
    print(f"embedded {len(texts)} texts: {', '.join(f'{k} {v['dim']}' for k, v in files.items())}")


if __name__ == "__main__":
    main()
