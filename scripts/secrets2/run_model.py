#!/usr/bin/env python3
"""The model's runner of s6-secrets v2 (docs/s6-secrets2-prereg.md, sections 5, 7.1, 7.2, 7.5):
how the trained ternary GGUF scores candidates, and how its answers reach the common result
format. The engine is the shipping one (`sky-eval`, native), at full depth.

  run_model.py eval    --model M.gguf --records R.jsonl --split val --out DIR
      sky-eval M.gguf --records R.jsonl --split val --exit-threshold 1.0 --json
      DIR/model-eval-full.json --dump DIR/model-dump.jsonl (full depth: the gated numbers and
      the latency of G6), then once more with --exit-threshold 0.0 (the first exit: reported).
      and once with the thresholds stored in the file (the early-exit rate: reported).
      Writes DIR/model-latency.json: median microseconds per candidate at both depths (the
      records of the real question), accuracy and ECE of the real question at each exit,
      parameters and file size.
  run_model.py wasm    --model M.gguf --dump DIR/model-dump.jsonl --out DIR
      section 7.5, reported: crates/sky-web/bench/parity-node.mjs on the first 200 candidates
      of the dump (their real question), on a copy of the file whose exit thresholds are set
      to 1 (amendment 13 of docs/s6-gen-prereg.md). Writes DIR/model-wasm.json: max
      |p_wasm - p_native|, argmax agreement, median microseconds. A parity failure is kept.
  run_model.py convert --model M.gguf --dump DIR/model-dump.jsonl --table T.jsonl
                       --corpus creddata:validation --out FLAGS [--system model]
      one flag per candidate with score = P(real), the threshold being tau* read from the
      model's metadata (`skycmd-s2.secrets2.tau_f1`, stored before any test file is read). Writes
      FLAGS/model.flags.jsonl; FLAGS must hold the generator's flags of the same corpus.

The records and the table come from the data side (records/<source>/<split>.jsonl and
tables/<source>-<split>.jsonl): the table has one line per candidate, in the order of the
records; a candidate gives the real question (noul) and, when its kind is known, the kind
question right after it. The dump has one line per record sky-eval answered, in the same
order, with the state it read: every line is checked against the table (same state, the
question word for word) before a score is taken from it.

SKY_EVAL names the binary (default: target/s3-train/release/sky-eval of the main checkout,
as scripts/train-secrets.sh builds it).
"""
import argparse
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import candidates  # noqa: E402
import common  # noqa: E402

# crates/sky-infer/src/secret_cand.rs (data side): the questions, word for word.
Q_REAL = "Is this value a real credential that should be revoked?"
Q_KIND = "What kind of value is this?"
KINDS = ("a real credential", "an example, test value or placeholder", "not a credential")
KIND_OF_LABEL = {"T": 0, "X": 1, "F": 2}


def sky_eval():
    return os.environ.get("SKY_EVAL") or os.path.join(common.main_checkout(), "target", "s3-train", "release", "sky-eval")


def read_jsonl(path):
    with open(path) as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def dump_to_scores(table, dump):
    """(scores, kind) from the rows of a table and the lines of a dump of the same records.

    scores: [(id, P(real))], one per row of the table. kind: {"n", "correct"} over the rows
    whose kind question was asked and whose label is known. Raises Refused when the dump is not
    the dump of this table."""
    dump = list(dump)
    scores = []
    kind = {"n": 0, "correct": 0}
    k = 0
    for row in table:
        if k >= len(dump):
            raise common.Refused(f"the dump ends before the table ({row['id']})")
        d = dump[k]
        if d.get("t") != "noul" or d.get("ins") != Q_REAL:
            raise common.Refused(f"dump line {k + 1} is not the real question of {row['id']}")
        if d.get("state") != row["state"]:
            raise common.Refused(f"dump line {k + 1} is not the state of {row['id']}")
        opts, probs = d.get("opts") or [], d.get("probs") or []
        if len(opts) != 2 or len(probs) != 2 or not opts[0].startswith("false") or not opts[1].startswith("true"):
            raise common.Refused(f"dump line {k + 1}: not the two answers of a noul question")
        scores.append((row["id"], float(probs[1])))
        k += 1
        if k < len(dump) and dump[k].get("t") == "choice" and dump[k].get("ins") == Q_KIND and dump[k].get("state") == row["state"]:
            if list(dump[k].get("opts") or []) != list(KINDS):
                raise common.Refused(f"dump line {k + 1}: not the options of the kind question")
            if row.get("label") in KIND_OF_LABEL:
                p = dump[k]["probs"]
                kind["n"] += 1
                kind["correct"] += max(range(len(p)), key=p.__getitem__) == KIND_OF_LABEL[row["label"]]
            k += 1
    if k != len(dump):
        raise common.Refused(f"the dump holds {len(dump) - k} lines after the last row of the table")
    return scores, kind


def per_exit(lines):
    """Accuracy and ECE of the real question at each exit, from sky-eval --dump-exits lines
    ({src, t, probs: [per exit], target})."""
    import scorer

    exits = {}
    for d in lines:
        if d.get("t") != "noul" or not str(d.get("src", "")).endswith(":real"):
            continue
        pos = d["target"][1] >= 0.5
        for e, p in enumerate(d["probs"]):
            exits.setdefault(e, []).append((float(p[1]), pos))
    return [{"exit": e + 1, "n": len(items), "accuracy": sum((p >= 0.5) == pos for p, pos in items) / len(items), "ece": scorer.ece(items)}
            for e, items in sorted(exits.items())]


def real_rows(ev):
    rows = [v for k, v in ev["sources"].items() if k.endswith(":real")]
    if not rows:
        raise common.Refused("the evaluation holds no record of the real question")
    return rows


def latency_of(full, first, stored, exits, model, tau):
    """Median microseconds per candidate (the records of the real question) at full depth and
    at the first exit, the early-exit rate at the stored thresholds, and the per-exit numbers.
    With several sources of real questions the median reported is the largest one."""
    n = sum(r["n"] for r in real_rows(full))
    return {"model": os.path.basename(model), "model_sha256": common.sha256_file(model), "bytes": os.path.getsize(model), "n_params": full.get("n_params"),
            "tau": tau, "candidates": n, "median_us_full_depth": max(r["median_us"] for r in real_rows(full)),
            "median_us_first_exit": max(r["median_us"] for r in real_rows(first)),
            "early_exit_rate_at_stored_thresholds": sum(r["early_exit_rate"] * r["n"] for r in real_rows(stored)) / sum(r["n"] for r in real_rows(stored)),
            "exits": exits, "engine": "sky-eval (native), one question per call"}


def run_eval(model, records, split, out):
    tau = common.model_tau(model)
    os.makedirs(out, exist_ok=True)
    exe = sky_eval()
    if not os.path.isfile(exe):
        raise common.Refused(f"{exe}: no sky-eval binary (cargo build --release -p sky-infer, or SKY_EVAL)")
    path = lambda name: os.path.join(out, name)
    run = lambda *args: subprocess.run(common.nice([exe, model, "--records", records, "--split", split, *args]), check=True, stdout=subprocess.DEVNULL)
    run("--exit-threshold", "1.0", "--json", path("model-eval-full.json"), "--dump", path("model-dump.jsonl"), "--dump-exits", path("model-exits.jsonl"))
    run("--exit-threshold", "0.0", "--json", path("model-eval-first-exit.json"))
    run("--json", path("model-eval-stored-thresholds.json"))
    latency = latency_of(common.load_json(path("model-eval-full.json")), common.load_json(path("model-eval-first-exit.json")),
                         common.load_json(path("model-eval-stored-thresholds.json")), per_exit(read_jsonl(path("model-exits.jsonl"))), model, tau)
    common.write_json(path("model-latency.json"), latency)
    return latency


def wasm(model, dump, out, n=200):
    main = common.main_checkout()
    script = os.path.join(main, "crates", "sky-web", "bench", "parity-node.mjs")
    gguf_set = os.path.join(os.path.dirname(sky_eval()), "gguf-set")
    if not os.path.isfile(os.path.join(main, "site", "public", "pkg", "sky_web.js")):
        raise common.Refused(f"{main}/site/public/pkg: no WASM build (scripts/build-wasm.sh)")
    if not os.path.isfile(gguf_set):
        raise common.Refused(f"{gguf_set}: no gguf-set binary")
    os.makedirs(out, exist_ok=True)
    real = os.path.join(out, "model-dump-real.jsonl")
    with open(real, "w") as f:
        k = 0
        for d in read_jsonl(dump):
            if d.get("t") == "noul" and d.get("ins") == Q_REAL:
                f.write(json.dumps(d) + "\n")
                k += 1
                if k == n:
                    break
    full = os.path.join(out, "full-depth.gguf")
    subprocess.run([gguf_set, model, full, "skycmd-s2.exit_threshold.choice=1", "skycmd-s2.exit_threshold.score=1", "skycmd-s2.exit_threshold.noul=1"],
                   check=True, capture_output=True)
    r = subprocess.run(common.nice([os.environ.get("NODE", "node"), script, full, real, "--n", str(n)]), capture_output=True, text=True)
    try:
        result = json.loads(r.stdout.strip().splitlines()[-1])
    except (IndexError, ValueError):
        raise common.Refused(f"parity-node.mjs gave no result (exit {r.returncode})")
    result["parity"] = r.returncode == 0
    common.write_json(os.path.join(out, "model-wasm.json"), result)
    return result


def convert(model, dump, table, corpus_name, out, system):
    tau = common.model_tau(model)
    scores, kind = dump_to_scores(candidates.read_table(table), read_jsonl(dump))
    meta = candidates.write_scored(out, system, corpus_name, scores, tau, {
        "version": f"{os.path.basename(model)} sha256 {common.sha256_file(model)}", "how": "sky-eval --dump at full depth, P(real) >= tau* of the file's metadata",
        "kind_question": {**kind, "accuracy": kind["correct"] / kind["n"] if kind["n"] else None}})
    return meta


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("what", choices=("eval", "wasm", "convert"))
    ap.add_argument("--model", required=True)
    ap.add_argument("--records")
    ap.add_argument("--split", default="val")
    ap.add_argument("--dump")
    ap.add_argument("--table")
    ap.add_argument("--corpus")
    ap.add_argument("--system", default="model")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    try:
        if a.what == "eval":
            print(json.dumps(run_eval(a.model, a.records, a.split, a.out), indent=1))
        elif a.what == "wasm":
            print(json.dumps(wasm(a.model, a.dump, a.out)))
        else:
            common.corpus(a.corpus)  # the guard of the split
            meta = convert(a.model, a.dump, a.table, a.corpus, a.out, a.system)
            common.log(f"{meta['system']}: {meta['flags']} candidates scored on {meta['corpus']}, tau* {meta['threshold']}")
    except common.Refused as e:
        common.log(f"run_model: refused: {e}")
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
