#!/usr/bin/env python3
"""Two baselines of s6-secrets v2 that are secretgate itself (docs/s6-secrets2-prereg.md,
section 6), at the frozen commit of runs/s6-secrets2/generator.json:

  run_secretgate.py generator --corpus creddata:validation --out DIR
      the generator sg-cand-1 alone, every candidate accepted: the checkout's own
      scripts/rule-candidates.mjs --candidates --raw, as the generator's author ran it. Writes
      DIR/generator.flags.jsonl (one flag per candidate, `id` its index in the export) and,
      next to it, the raw export itself: DIR/raw/candidates.raw.jsonl holds VALUES and stays
      in the data folder. On training and validation the export must be, byte for byte, the
      frozen one of generator.json.
  run_secretgate.py default --corpus creddata:validation --out DIR
      the default engine, filters on: scan(text, {sourcePath}) per file. Writes
      DIR/secretgate-default.flags.jsonl.

On an issue corpus (issues:<split>, issues:<split>:obfuscated) a text has no path (section
7.4): the default engine is given none, and the generator runs through `candidateRecords` of
the same script without a source path (engine.mjs), its records carrying neither path nor
extension; DIR/raw/candidates.where.jsonl names the text of each record.

Both refuse a checkout that is not the frozen one, and a file outside the corpus.
"""
import argparse
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

MAX_BYTES = 64 * 1024 * 1024


def node():
    return os.environ.get("NODE", "node")


def records_to_flags(records, base, check, with_id=True):
    """Flags from engine records (file, rule, line, endLine, cpStart, cpEnd), each placed in
    CredData's lines of its file."""
    cache = (None, None)
    for i, r in enumerate(records):
        repo = check(r["file"])
        if cache[0] != r["file"]:
            cache = (r["file"], common.EngineLines(common.read_text(os.path.join(base, r["file"]))))
        line, start, end_line, end = cache[1].place(r["line"], r["cpStart"], r["endLine"], r["cpEnd"])
        yield common.flag(repo, r["file"], line, end_line, start, end, rule=r["rule"], id=i if with_id else None)


def read_jsonl(path):
    with open(path) as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def listing_of(corpus, raw):
    path = os.path.join(raw, "files.txt")
    with open(path, "w") as f:
        f.write("\n".join(corpus.files) + "\n")
    return path


def engine(secretgate, mode, listing, corpus, out, stats, extra=()):
    repos = [r[len("data/"):] for r in corpus.roots if r.startswith("data/")]
    cmd = [node(), "--max-old-space-size=8192", os.path.join(common.HERE, "engine.mjs"), "--secretgate", secretgate, "--mode", mode,
           "--files-from", listing, "--base", corpus.base, "--out", out, "--max-bytes", str(MAX_BYTES), "--stats", stats, *extra]
    if repos:
        cmd += ["--repos", ",".join(repos)]
    subprocess.run(common.nice(cmd), check=True)


def generator(corpus, out, secretgate, gen):
    raw = os.path.join(out, "raw")
    os.makedirs(raw, mode=0o700, exist_ok=True)
    listing = listing_of(corpus, raw)
    cand = os.path.join(raw, "candidates.raw.jsonl")
    stats_path = os.path.join(raw, "stats.json")
    creddata = corpus.name.startswith("creddata:")
    with common.Clock() as clock:
        if creddata:
            subprocess.run(common.nice([node(), "--max-old-space-size=8192", os.path.join(secretgate, "scripts", "rule-candidates.mjs"),
                                        "--candidates", "--raw", "--files-from", listing, "--base", corpus.base, "--out", cand,
                                        "--dropped", os.path.join(raw, "dropped.jsonl"), "--stats", stats_path,
                                        "--max-bytes", str(MAX_BYTES)]), check=True)
        else:
            engine(secretgate, "candidates", listing, corpus, cand, stats_path, ["--where", os.path.join(raw, "candidates.where.jsonl")])
    digest = common.sha256_file(cand)
    reproduced = None
    frozen_name = f"candidates-{corpus.name.split(':')[1]}.raw.jsonl" if creddata else None
    if frozen_name in gen["files"]:
        reproduced = digest == gen["files"][frozen_name]["sha256"]
        if not reproduced:
            raise common.Refused(f"{cand}: sha256 {digest} is not the frozen export of generator.json ({gen['files'][frozen_name]['sha256']})")
    records = read_jsonl(cand)
    if not creddata:
        records = ({**r, "file": w["file"]} for r, w in zip(read_jsonl(cand), read_jsonl(os.path.join(raw, "candidates.where.jsonl"))))
    flags = records_to_flags(records, corpus.base, corpus.check)
    stats = common.load_json(stats_path)
    run = {"version": f"sg-cand-1 revision {gen['revision']}, secretgate {gen['secretgate']['commit']}",
           "how": "scripts/rule-candidates.mjs --candidates --raw, every candidate accepted" if creddata
           else "candidateRecords() of scripts/rule-candidates.mjs without a source path, every candidate accepted",
           "corpus": corpus.name, "unit": "candidate", "files": len(corpus.files), "export_sha256": digest, "is_the_frozen_export": reproduced,
           "cpu_seconds": round(clock.children, 1), "wall_seconds": round(clock.wall, 1)}
    if creddata:
        run.update({"lines": stats["lines"], "candidates_per_1000_lines": round(1000.0 * stats["candidates"] / stats["lines"], 3)})
    return common.write_flags(out, "generator", flags, run)


def default(corpus, out, secretgate, gen):
    raw = os.path.join(out, "raw")
    os.makedirs(raw, mode=0o700, exist_ok=True)
    listing = listing_of(corpus, raw)
    found = os.path.join(raw, "secretgate-default.jsonl")
    creddata = corpus.name.startswith("creddata:")
    with common.Clock() as clock:
        engine(secretgate, "default", listing, corpus, found, os.path.join(raw, "secretgate-default.stats.json"), [] if creddata else ["--no-path"])
    flags = records_to_flags(read_jsonl(found), corpus.base, corpus.check, with_id=False)
    return common.write_flags(out, "secretgate-default", flags, {
        "version": f"secretgate {gen['secretgate']['commit']}",
        "how": "scan(text, {sourcePath}) per file, filters on" if creddata else "scan(text) per text, no source path, filters on",
        "corpus": corpus.name, "unit": "span", "files": len(corpus.files),
        "cpu_seconds": round(clock.children, 1), "wall_seconds": round(clock.wall, 1)})


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("what", choices=("generator", "default"))
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    try:
        gen = common.check_generator()
        meta = (generator if a.what == "generator" else default)(common.corpus(a.corpus), a.out, common.secretgate_root(), gen)
    except common.Refused as e:
        common.log(f"run_secretgate: refused: {e}")
        return 3
    common.log(f"{meta['system']}: {meta['flags']} flags on {meta['corpus']}, {meta['cpu_seconds']} s of CPU")
    return 0


if __name__ == "__main__":
    sys.exit(main())
