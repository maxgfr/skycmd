# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = [
#   "torch==2.14.0",
#   "datasets==5.1.0",
#   "huggingface_hub>=1.31,<2",
# ]
# ///
# The engine comes from `--with` (scripts/lfm3-ft.sh): vllm==0.33.0, or transformers==5.12.1 and
# accelerate==1.15.0 for the fallback. They cannot share one environment (vLLM 0.33 needs
# transformers 5.16-5.20 and tokenizers >= 0.23.2, transformers 5.12.1 needs tokenizers <= 0.23.0).
"""Evaluation of one model, base or fine-tuned, on Hugging Face Jobs (docs/lfm3-ft-prereg.md §5).

Modes s1, s2-256 and s2-1024 on the calibration (600) and test (1,000) splits of data/lfm3, and s1 on
the 500 MMLU-Pro questions. The records are those of experiments/lfm3-s1s2/s1s2.py's `gen` (s1 with the
log-probability features of the router), so `route` reads them unchanged. Each file goes to the private
work repo as eval/<tag>/<split>.<mode>.jsonl after every chunk; a restarted job skips the ids already
there (an interrupted job resumes, it is never rerun to "see").

  eval.py --common common.py --calib calib.jsonl --test test.jsonl --model LiquidAI/LFM3-1.7B \
          --revision <sha> --tag base --engine vllm
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

LOCAL = "/tmp/lfm3-ft"
CHUNK = 400


def run_mode(engine, items, mode):
    t0 = time.time()
    if mode == "s1":
        recs = C.run_s1(engine, items)
    else:
        recs = C.run_s2(engine, items, int(mode.split("-")[1]))
    return C.finish_records(items, recs, mode, time.time() - t0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--common", help="common.py, shipped with the job (unused otherwise)")
    ap.add_argument("--calib", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", required=True)
    ap.add_argument("--tag", choices=("base", "ft"), required=True)
    ap.add_argument("--engine", choices=("vllm", "hf"), required=True)
    ap.add_argument("--modes", default="s1,s2-256,s2-1024")
    ap.add_argument("--splits", default="calib,test")
    ap.add_argument("--no-mmlu", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="first N questions per split (never for the run)")
    ap.add_argument("--batch-size", type=int, default=128, help="transformers fallback only")
    ap.add_argument("--work", default=C.WORK)
    a = ap.parse_args()

    splits = {"calib": C.load_shipped(a.calib), "test": C.load_shipped(a.test)}
    jobs = [(s, m) for s in a.splits.split(",") for m in a.modes.split(",")]
    if not a.no_mmlu:
        splits["mmlu"] = C.mmlu_items()
        jobs.append(("mmlu", "s1"))
    prefix = f"eval/{a.tag}" + ("-limit" if a.limit else "")
    kw = {"batch_size": a.batch_size} if a.engine == "hf" else {}
    engine = C.make_engine(a.engine, a.model, a.revision, **kw)
    meta = {"model": a.model, "revision": a.revision, "tag": a.tag, "engine": a.engine,
            "versions": C.versions(), "data_sha256": {k: C.sha256(getattr(a, k)) for k in ("calib", "test")},
            "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "files": {}}
    for split, mode in jobs:
        items = splits[split][: a.limit] if a.limit else splits[split]
        name = f"{split}.{mode}.jsonl"
        local = f"{LOCAL}/{prefix}/{name}"
        if not os.path.exists(local):
            C.hub_download(a.work, f"{prefix}/{name}", LOCAL)
        done = {r["id"] for r in C.read_jsonl(local)}
        todo = [it for it in items if it["id"] not in done]
        print(f"{a.tag} {mode} on {split}: {len(todo)} to generate ({len(done)} done)", flush=True)
        for i in range(0, len(todo), CHUNK):
            chunk = todo[i : i + CHUNK]
            t0 = time.time()
            recs = run_mode(engine, chunk, mode)
            C.append_jsonl(local, recs)
            C.hub_upload(a.work, local, f"{prefix}/{name}", f"eval {a.tag} {split} {mode}: {len(done) + i + len(chunk)}")
            n_tok = sum(r["tokens"] for r in recs)
            print(f"  {i + len(chunk)}/{len(todo)}  {n_tok / (time.time() - t0):.0f} tok/s", flush=True)
        rows = C.read_jsonl(local)
        meta["files"][name] = {"n": len(rows), "acc": sum(r["correct"] for r in rows) / max(1, len(rows))}
    meta["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    path = f"{LOCAL}/{prefix}/meta.json"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)
    C.hub_upload(a.work, path, f"{prefix}/meta.json", f"eval {a.tag}: meta")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
