"""Runs inside the container of scripts/secrets2/docker/cbx only (run_scanners.py cbx starts it):
cbx-secrets-detector-17M over a list of files, on CPU, with the model's own code and its default
settings (`SecretDetector().extract(text)`, one text per file).

    python cbx_infer.py /model /in /out THREADS [allowed RepoID ...]

/model  the pinned model folder (read-only), with its cbx_secrets.py
/in     the base of the corpus (read-only): only the roots of the split are mounted below it
/out    files.txt (the paths to read, below /in) comes in; spans.jsonl and stats.json go out

One line per predicted span: file, start, end (code-point offsets in the file's text) and the
label. Never the text of a span. A file is read as the generator reads it: skipped when empty,
larger than 64 MiB or with a NUL byte in its first 8192 bytes; UTF-8 with invalid bytes
replaced. With RepoIDs given, a path that is not `data/<one of them>/...` stops the run.
"""
import json
import os
import re
import resource
import sys
import time

MAX_BYTES = 64 * 1024 * 1024


def main():
    model, base, out, threads = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
    allowed = set(sys.argv[5:])
    with open(os.path.join(out, "files.txt")) as f:
        files = [l for l in f.read().split("\n") if l]
    for path in files:
        m = re.match(r"^data/([0-9a-f]{8})/", path)
        if allowed and (not m or m.group(1) not in allowed):
            sys.exit(f"refused: {path} is not a file of the repositories this run was given")
        if path.startswith("/") or ".." in path.split("/"):
            sys.exit(f"refused: {path}")

    import torch

    torch.set_num_threads(threads)
    sys.path.insert(0, model)
    from cbx_secrets import SecretDetector

    det = SecretDetector(model, device="cpu")
    began = time.monotonic()
    stats = {"files": len(files), "read": 0, "skipped": 0, "spans": 0, "characters": 0, "threads": threads,
             "torch": torch.__version__, "max_length": det.max_length}
    with open(os.path.join(out, "spans.jsonl"), "w") as sink:
        for i, path in enumerate(files):
            with open(os.path.join(base, path), "rb") as f:
                buf = f.read(MAX_BYTES + 1)
            if not buf or len(buf) > MAX_BYTES or b"\0" in buf[:8192]:
                stats["skipped"] += 1
                continue
            text = buf.decode("utf-8", "replace")
            stats["read"] += 1
            stats["characters"] += len(text)
            for s in det.extract(text):
                sink.write(json.dumps({"file": path, "start": s["start"], "end": s["end"], "label": s["label"]}) + "\n")
                stats["spans"] += 1
            if (i + 1) % 50 == 0:
                print(f"cbx: {i + 1}/{len(files)} files, {stats['spans']} spans, {time.monotonic() - began:.0f} s", file=sys.stderr, flush=True)
    r = resource.getrusage(resource.RUSAGE_SELF)
    stats["cpu_seconds"] = round(r.ru_utime + r.ru_stime, 1)
    stats["wall_seconds"] = round(time.monotonic() - began, 1)
    stats["max_rss_mb"] = round(r.ru_maxrss / 1024, 1)
    with open(os.path.join(out, "stats.json"), "w") as f:
        json.dump(stats, f, indent=1)


if __name__ == "__main__":
    main()
