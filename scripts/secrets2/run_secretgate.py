#!/usr/bin/env python3
"""Two baselines of s6-secrets v2 that are secretgate itself (docs/s6-secrets2-prereg.md,
section 6), at the frozen commit of runs/s6-secrets2/generator.json:

  run_secretgate.py generator --split validation --out DIR
      the generator sg-cand-1 alone, every candidate accepted: the checkout's own
      scripts/rule-candidates.mjs --candidates --raw, as the generator's author ran it. Writes
      DIR/generator.flags.jsonl (one flag per candidate, `id` its index in the export) and,
      next to it, the raw export itself: DIR/raw/candidates.raw.jsonl holds VALUES and stays
      in the data folder. On training and validation the export must be, byte for byte, the
      frozen one of generator.json.
  run_secretgate.py default --split validation --out DIR
      the default engine, filters on: scan(text, {sourcePath}) per file. Writes
      DIR/secretgate-default.flags.jsonl.

Both refuse a checkout that is not the frozen one, and a file outside the split.
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


def generator(split, out, secretgate, gen):
    raw = os.path.join(out, "raw")
    os.makedirs(raw, mode=0o700, exist_ok=True)
    files = common.creddata_files(split)
    listing = os.path.join(raw, "files.txt")
    with open(listing, "w") as f:
        f.write("\n".join(files) + "\n")
    cand = os.path.join(raw, "candidates.raw.jsonl")
    with common.Clock() as clock:
        subprocess.run(common.nice([node(), "--max-old-space-size=8192", os.path.join(secretgate, "scripts", "rule-candidates.mjs"),
                                    "--candidates", "--raw", "--files-from", listing, "--base", common.creddata_base(), "--out", cand,
                                    "--dropped", os.path.join(raw, "dropped.jsonl"), "--stats", os.path.join(raw, "stats.json"),
                                    "--max-bytes", str(MAX_BYTES)]), check=True)
    digest = common.sha256_file(cand)
    frozen_name = f"candidates-{split.name}.raw.jsonl"
    reproduced = None
    if frozen_name in gen["files"]:
        reproduced = digest == gen["files"][frozen_name]["sha256"]
        if not reproduced:
            raise common.Refused(f"{cand}: sha256 {digest} is not the frozen export of generator.json ({gen['files'][frozen_name]['sha256']})")
    flags = records_to_flags(read_jsonl(cand), common.creddata_base(), split.check)
    stats = common.load_json(os.path.join(raw, "stats.json"))
    return common.write_flags(out, "generator", flags, {
        "version": f"sg-cand-1 revision {gen['revision']}, secretgate {gen['secretgate']['commit']}",
        "how": "scripts/rule-candidates.mjs --candidates --raw, every candidate accepted",
        "corpus": f"creddata:{split.name}", "unit": "candidate", "files": len(files), "lines": stats["lines"],
        "candidates_per_1000_lines": round(1000.0 * stats["candidates"] / stats["lines"], 3),
        "export_sha256": digest, "is_the_frozen_export": reproduced,
        "cpu_seconds": round(clock.children, 1), "wall_seconds": round(clock.wall, 1)})


def default(split, out, secretgate, gen):
    raw = os.path.join(out, "raw")
    os.makedirs(raw, mode=0o700, exist_ok=True)
    files = common.creddata_files(split)
    listing = os.path.join(raw, "files.txt")
    with open(listing, "w") as f:
        f.write("\n".join(files) + "\n")
    found = os.path.join(raw, "secretgate-default.jsonl")
    with common.Clock() as clock:
        subprocess.run(common.nice([node(), "--max-old-space-size=8192", os.path.join(common.HERE, "engine.mjs"), "--secretgate", secretgate,
                                    "--mode", "default", "--files-from", listing, "--base", common.creddata_base(), "--out", found,
                                    "--repos", ",".join(split.repos), "--max-bytes", str(MAX_BYTES),
                                    "--stats", os.path.join(raw, "secretgate-default.stats.json")]), check=True)
    flags = records_to_flags(read_jsonl(found), common.creddata_base(), split.check, with_id=False)
    return common.write_flags(out, "secretgate-default", flags, {
        "version": f"secretgate {gen['secretgate']['commit']}", "how": "scan(text, {sourcePath}) per file, filters on",
        "corpus": f"creddata:{split.name}", "unit": "span", "files": len(files),
        "cpu_seconds": round(clock.children, 1), "wall_seconds": round(clock.wall, 1)})


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("what", choices=("generator", "default"))
    ap.add_argument("--split", required=True, choices=common.CREDDATA_SPLITS)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    try:
        gen = common.check_generator()
        split = common.open_split(a.split)
        meta = (generator if a.what == "generator" else default)(split, a.out, common.secretgate_root(), gen)
    except common.Refused as e:
        common.log(f"run_secretgate: refused: {e}")
        return 3
    common.log(f"{meta['system']}: {meta['flags']} flags, {meta['cpu_seconds']} s of CPU")
    return 0


if __name__ == "__main__":
    sys.exit(main())
