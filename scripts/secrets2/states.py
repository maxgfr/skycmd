#!/usr/bin/env python3
"""States for every candidate of a corpus, from the data side's own state builder
(docs/s6-secrets2-prereg.md, section 4.1): the model and the trained baselines judge all the
candidates of a split, not only the labelled ones the training build keeps.

  states.py --corpus creddata:validation --candidates FLAGS/raw/candidates.raw.jsonl --out DIR/creddata
  states.py --corpus clean --candidates <data>/eval/clean/flags/raw/candidates.raw.jsonl --out <data>/eval/clean/states/clean

writes OUT.jsonl (the table: id, state, features, one line per candidate, in the export's
order), OUT.records.jsonl (the decision records of the real question, same order, split "val")
and OUT.states.json (what built them: hashes only).

Nothing of the state is computed here. Each generator record is put, as it is, in the item
envelope the data side reads (crates/sky-data/src/secrets2: id, source, split, label, rec) and
`sky-data secrets2 mix` of the data side's build turns the items into states and records, with
the tokenizer and the dictionary whose hashes BUILD.json names (anything else is refused). The
envelope's label is a placeholder (F): the truth of a candidate is not known here, the scorer
reads it from the meta rows. Ids: `cred:<val|test>:<index>` on CredData, `clean:all:<index>`
on the clean-text corpus, the index being the candidate's line in the export.

A CredData candidate in a file directly under its repository's folder (a licence copy) has no
scope directory: the data side's builder refuses such a path, so these candidates get no state
and are counted (they sit on unlabelled lines).

SKY_DATA names the binary (default: target/release/sky-data of the data side's worktree).
"""
import argparse
import json
import os
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

SEED = "1234"
SLOTS = [(s, k) for s in ("creddata", "issues", "synth") for k in ("train", "val")]


def sky_data():
    return os.environ.get("SKY_DATA") or os.path.join(common.main_checkout(), ".claude", "worktrees", "secrets2", "target", "release", "sky-data")


def build_inputs(build):
    """The tokenizer and the dictionary of the training build, checked against BUILD.json."""
    path = os.path.join(build, "BUILD.json")
    if not os.path.isfile(path):
        raise common.Refused(f"{path}: the data side's build is not finished")
    b = common.load_json(path)
    tok = b["inputs"]["tokenizer"]
    tok_path = os.environ.get("TOKENIZER") or os.path.join(common.main_checkout(), "data", "tok", tok["name"])
    dic = b["inputs"]["dictionary_source"]
    for p, want in ((tok_path, tok["file"]["sha256"]), (dic["path"], dic["file"]["sha256"])):
        if not os.path.isfile(p) or common.sha256_file(p) != want:
            raise common.Refused(f"{p} is not the file BUILD.json names (sha256 {want})")
    return {"build_json_sha256": common.sha256_file(path), "tokenizer": tok_path, "tokenizer_sha256": tok["file"]["sha256"],
            "dictionary": dic["path"], "dictionary_sha256": dic["file"]["sha256"]}


def envelope(corpus_name, index, rec):
    """(slot source, item) of one generator record, or None for a record the builder cannot place."""
    if corpus_name.startswith("creddata:"):
        if rec["file"].count("/") < 3:
            return None
        short = "test" if corpus_name.endswith(":test") else "val"
        return "creddata", {"id": f"cred:{short}:{index}", "source": "creddata", "split": "val", "label": "F", "kind": False, "rec": rec}
    if corpus_name == "clean":
        # the path inside its repository: the first component of the corpus path names the repository
        rec = {**rec, "file": rec["file"].split("/", 1)[1]}
        return "synth", {"id": f"clean:all:{index}", "source": "synth", "split": "val", "label": "F", "kind": False, "rec": rec}
    raise common.Refused(f"no states for the corpus {corpus_name!r}")


def build_states(corpus_name, candidates, out, build):
    inputs = build_inputs(build)
    exe = sky_data()
    if not os.path.isfile(exe):
        raise common.Refused(f"{exe}: no sky-data of the data side (SKY_DATA)")
    work = out + ".work"
    shutil.rmtree(work, ignore_errors=True)
    os.makedirs(os.path.join(work, "candidates"), mode=0o700)
    ids, skipped, source = [], 0, None
    try:
        files = {slot: open(os.open(os.path.join(work, "candidates", f"{slot[0]}-{slot[1]}.jsonl"), os.O_WRONLY | os.O_CREAT, 0o600), "w") for slot in SLOTS}
        try:
            with open(candidates) as f:
                for index, line in enumerate(l for l in f if l.strip()):
                    item = envelope(corpus_name, index, json.loads(line))
                    if item is None:
                        skipped += 1
                        continue
                    source = item[0]
                    files[(source, "val")].write(json.dumps(item[1]) + "\n")
                    ids.append(item[1]["id"])
        finally:
            for f in files.values():
                f.close()
        r = subprocess.run(common.nice([exe, "--seed", SEED, "secrets2", "mix", "--out", work, "--tokenizer", inputs["tokenizer"], "--dict", inputs["dictionary"]]),
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise common.Refused(f"sky-data secrets2 mix failed (exit {r.returncode}): {r.stderr.strip().splitlines()[-1] if r.stderr.strip() else ''}")
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        table, records = out + ".jsonl", out + ".records.jsonl"
        if ids:
            shutil.move(os.path.join(work, "tables", f"{source}-val.jsonl"), table)
            shutil.move(os.path.join(work, "records", source, "val.jsonl"), records)
        else:
            open(table, "w").close()
            open(records, "w").close()
    finally:
        shutil.rmtree(work, ignore_errors=True)
    with open(table) as f:
        got = [json.loads(line)["id"] for line in f if line.strip()]
    with open(records) as f:
        n_records = sum(1 for line in f if line.strip())
    if got != ids or n_records != len(ids):
        raise common.Refused(f"{table}: not one state and one record per candidate, in the export's order")
    meta = {"corpus": corpus_name, "candidates": len(ids) + skipped, "states": len(ids), "without_a_scope_directory": skipped,
            "export_sha256": common.sha256_file(candidates), "table_sha256": common.sha256_file(table), "records_sha256": common.sha256_file(records),
            "state_builder": {"binary": os.path.basename(exe), "binary_sha256": common.sha256_file(exe), "command": "sky-data secrets2 mix"},
            **{k: v for k, v in inputs.items() if k.endswith("sha256")}}
    common.write_json(out + ".states.json", meta)
    return meta


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--build", default=os.path.join(common.data_root(), "build"))
    a = ap.parse_args()
    try:
        if a.corpus != "clean":
            common.corpus(a.corpus)  # the guard of the split
        meta = build_states(a.corpus, a.candidates, a.out, a.build)
    except common.Refused as e:
        common.log(f"states: refused: {e}")
        return 3
    common.log(f"states of {meta['corpus']}: {meta['states']} of {meta['candidates']} candidates ({meta['without_a_scope_directory']} in a file with no scope directory)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
