#!/usr/bin/env python3
"""States for every candidate of a corpus, from the data side's own state builder
(docs/s6-secrets2-prereg.md, section 4.1): the model and the trained baselines judge all the
candidates of a split, not only the labelled ones the training build keeps.

  states.py --corpus creddata:validation --candidates FLAGS/raw/candidates.raw.jsonl --out DIR/creddata
  states.py --corpus clean --candidates <data>/eval/clean/flags/raw/candidates.raw.jsonl --out <data>/eval/clean/states/clean
  states.py --corpus issues:validation --out DIR/issues
  states.py --corpus issues:validation:obfuscated --out DIR/issues-obfuscated
  states.py --corpus issues:validation:e2e --candidates ISSUES/flags/raw/candidates.raw.jsonl --out DIR/issues-e2e

writes OUT.jsonl (the table: id, state, features, one line per candidate, in the order of the
input), OUT.records.jsonl (the decision records, same order, split "val": the real question,
and the kind question of a CredData candidate whose label is known) and OUT.states.json (what
built them: hashes and counts only).

Nothing of the state is computed here: `sky-data secrets2 states` of the data side
(crates/sky-data/src/secrets2/states.rs) builds the table and the records, with the tokenizer
and the dictionary whose hashes BUILD.json names (anything else is refused). Ids:
`cred:<val|test>:<index>` on CredData and `clean:all:<index>` on the clean-text corpus, the
index being the candidate's line in the generator's export; `issue:<val|test>:<row>` for the
dataset's own candidates of the issue reports, as published or through CredData's obfuscation
(issues.py prepare and obfuscate wrote the texts and the places); `issue-e2e:<val|test>:<index>`
for the generator's candidates on the issue texts.

A CredData candidate takes its label from the meta rows (the data side's matching rules); the
candidates no row labels, and those of the clean-text corpus and of the end-to-end run, have
none: the scorer reads the truth itself. A CredData candidate in a file directly under its
repository's folder (a licence copy) has no scope directory: it gets no state and is counted.

The test split goes through two guards: this script's (common.py) and the command's own, which
is given the model and runs/s6-secrets2/TEST-OPEN.

SKY_DATA names the binary (default: target/s3-train/release/sky-data of the main checkout,
as scripts/build-secrets2-data.sh builds it).
"""
import argparse
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

SEED = "1234"


def sky_data():
    return os.environ.get("SKY_DATA") or os.path.join(common.main_checkout(), "target", "s3-train", "release", "sky-data")


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


def needs_export(corpus_name):
    return corpus_name == "clean" or corpus_name.startswith("creddata:") or corpus_name.endswith(":e2e")


def build_states(corpus_name, candidates, out, build):
    inputs = build_inputs(build)
    exe = sky_data()
    if not os.path.isfile(exe):
        raise common.Refused(f"{exe}: no sky-data of the data side (SKY_DATA)")
    if needs_export(corpus_name) and not (candidates and os.path.isfile(candidates)):
        raise common.Refused(f"{corpus_name}: no export of the generator ({candidates})")
    cmd = [exe, "--seed", SEED, "secrets2", "states", "--data", common.data_root(), "--corpus", corpus_name, "--out", out,
           "--tokenizer", inputs["tokenizer"], "--dict", inputs["dictionary"], "--test-open", common.TEST_OPEN]
    if needs_export(corpus_name):
        cmd += ["--candidates", candidates]
    if os.environ.get("SECRETS2_MODEL"):
        cmd += ["--model", os.environ["SECRETS2_MODEL"]]
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    r = subprocess.run(common.nice(cmd), capture_output=True, text=True)
    if r.returncode != 0:
        raise common.Refused(f"sky-data secrets2 states failed (exit {r.returncode}): {r.stderr.strip().splitlines()[-1] if r.stderr.strip() else ''}")
    counts = json.loads(r.stdout)
    table, records = out + ".jsonl", out + ".records.jsonl"
    for path, key in ((table, "table"), (records, "records_file")):
        if common.sha256_file(path) != counts[key]["sha256"]:
            raise common.Refused(f"{path} is not the file the command wrote")
    with open(table) as f:
        n_rows = sum(1 for line in f if line.strip())
    if n_rows != counts["states"]:
        raise common.Refused(f"{table}: not one state per candidate")
    meta = {"corpus": corpus_name, "candidates": counts["candidates"], "states": counts["states"],
            "without_a_scope_directory": counts["without_a_scope_directory"], "not_found_in_their_text": counts["not_found_in_their_text"],
            "records": counts["records"], "labels": counts["labels"],
            "input_sha256": counts["input"]["sha256"], "table_sha256": counts["table"]["sha256"], "records_sha256": counts["records_file"]["sha256"],
            "state_builder": {"binary": os.path.basename(exe), "binary_sha256": common.sha256_file(exe), "command": "sky-data secrets2 states"},
            **{k: v for k, v in inputs.items() if k.endswith("sha256")}}
    common.write_json(out + ".states.json", meta)
    return meta


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--candidates")
    ap.add_argument("--out", required=True)
    ap.add_argument("--build", default=os.path.join(common.data_root(), "build"))
    a = ap.parse_args()
    try:
        if a.corpus != "clean":
            # the guard of the split (the end-to-end candidates are those of the issue texts)
            common.corpus(a.corpus[: -len(":e2e")] if a.corpus.endswith(":e2e") else a.corpus)
        meta = build_states(a.corpus, a.candidates, a.out, a.build)
    except common.Refused as e:
        common.log(f"states: refused: {e}")
        return 3
    common.log(f"states of {meta['corpus']}: {meta['states']} of {meta['candidates']} candidates "
               f"({meta['without_a_scope_directory']} in a file with no scope directory, {meta['not_found_in_their_text']} not found in their text)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
