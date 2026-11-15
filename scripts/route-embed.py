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
"""

import argparse
import importlib.metadata as md
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "route"))
import common  # noqa: E402


def minilm(texts, threads):
    import torch
    from sentence_transformers import SentenceTransformer

    torch.set_num_threads(threads)
    m = SentenceTransformer(common.model_dir("all-MiniLM-L6-v2"), device="cpu")
    return m.encode(texts, batch_size=256, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)


def potion(texts):
    import numpy as np
    from model2vec import StaticModel

    m = StaticModel.from_pretrained(common.model_dir("potion-base-2M"))
    e = m.encode(texts, show_progress_bar=False).astype(np.float32)
    n = np.linalg.norm(e, axis=1, keepdims=True)
    return e / np.where(n == 0, 1, n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", default=common.build_root())
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
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
