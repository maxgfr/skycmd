# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = [
#   "torch==2.14.0",
#   "datasets==5.1.0",
#   "huggingface_hub>=1.31,<2",
# ]
# ///
# The engine comes from `--with` (scripts/lfm3-ft.sh), as for eval.py.
"""Self-distillation samples on the train questions, on Hugging Face Jobs (docs/lfm3-ft-prereg.md §3 b).

The base model answers each kept train question (GSM8K train and MATH train, minus the calibration ids
and any question whose normalized text is a calibration, test or MMLU-Pro question) 4 times in s2-1024
(budget forcing as in s1s2.py), seed qseed(id) + 1,000,003 k for sample k. With the transformers
fallback engine, only a seeded subset of 1,000 questions per task is sampled (pre-registered).

Records go to gen/samples/part-<first sample>.jsonl in the private work repo, one part per chunk, with gen/stats.json at the
end; a restarted job skips the (id, k) already there.

  gen.py --common common.py --calib calib.jsonl --test test.jsonl --revision <sha> --engine vllm
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

LOCAL = "/tmp/lfm3-ft"
CHUNK = 2048  # sequences per chunk


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--common", help="common.py, shipped with the job (unused otherwise)")
    ap.add_argument("--calib", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--revision", required=True, help="base model revision pinned by the smoke")
    ap.add_argument("--engine", choices=("vllm", "hf"), required=True)
    ap.add_argument("--budget", type=int, default=1024)
    ap.add_argument("--limit", type=int, default=0, help="first N questions (never for the run)")
    ap.add_argument("--batch-size", type=int, default=128, help="transformers fallback only")
    ap.add_argument("--work", default=C.WORK)
    a = ap.parse_args()

    calib, test = C.load_shipped(a.calib), C.load_shipped(a.test)
    mmlu = C.mmlu_items()
    rows, stats = C.select_train(C.load_train_rows(), calib, [r["question"] for r in test + mmlu])
    if a.engine == "hf":
        rows = C.hf_fallback_subset(rows)
    if a.limit:
        rows = rows[: a.limit]
    stats.update(sampled_questions=len(rows), engine=a.engine, revision=a.revision, budget=a.budget,
                 samples_per_question=C.N_SAMPLES)
    print(json.dumps(stats), flush=True)

    prefix = "gen/samples" + ("-limit" if a.limit else "")
    prev = C.hub_parts(a.work, prefix, LOCAL)
    done = {(r["id"], r["k"]) for r in prev}
    todo = [(r, k) for r in rows for k in range(C.N_SAMPLES) if (r["id"], k) not in done]
    print(f"{len(todo)} samples to generate ({len(done)} done)", flush=True)
    engine = C.make_engine(a.engine, C.BASE, a.revision, **({"batch_size": a.batch_size} if a.engine == "hf" else {}))
    mode = f"s2-{a.budget}"
    for i in range(0, len(todo), CHUNK):
        chunk = todo[i : i + CHUNK]
        items = [r for r, _ in chunk]
        seeds = [C.qseed(r["id"]) + C.SAMPLE_STRIDE * k for r, k in chunk]
        t0 = time.time()
        recs = C.finish_records(items, C.run_s2(engine, items, a.budget, seeds), mode, time.time() - t0)
        for rec, (_, k), s in zip(recs, chunk, seeds):
            rec.update(k=k, seed=s)
        part = f"{prefix}/part-{len(done) + i:06d}.jsonl"
        C.write_jsonl(f"{LOCAL}/{part}", recs)
        C.hub_upload(a.work, f"{LOCAL}/{part}", part, f"gen: {len(done) + i + len(chunk)} samples")
        n_tok = sum(r["tokens"] for r in recs)
        print(f"  {i + len(chunk)}/{len(todo)}  {n_tok / (time.time() - t0):.0f} tok/s  "
              f"acc {sum(r['correct'] for r in recs) / len(recs):.3f}", flush=True)

    samples = C.hub_parts(a.work, prefix, LOCAL)
    targets = C.distill_targets(samples)
    stats.update(samples=len(samples), correct=sum(s["correct"] for s in samples),
                 forced=sum(s["forced"] for s in samples),
                 questions_with_nothink=sum(1 for t in targets.values() if t["nothink"]),
                 questions_with_think=sum(1 for t in targets.values() if t["think"]),
                 versions=C.versions(), data_sha256={k: C.sha256(getattr(a, k)) for k in ("calib", "test")})
    path = f"{LOCAL}/gen/stats" + ("-limit" if a.limit else "") + ".json"
    with open(path, "w") as f:
        json.dump(stats, f, indent=2)
    C.hub_upload(a.work, path, os.path.relpath(path, LOCAL), "gen: stats")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
