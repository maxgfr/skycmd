#!/usr/bin/env python3
"""The clean-text measure of s6-secrets v2 (docs/s6-secrets2-prereg.md, section 7.7, gate G7):
on real files that hold no secret, the share of files with at least one accepted candidate.

  clean_text.py prepare
      copies the corpus of <data>/gen/clean-corpus.json (frozen with the generator: its hash is
      in runs/s6-secrets2/generator.json) to <data>/eval/clean/corpus/<repository>/<path>,
      every file checked against its sha256 in the list. secretgate's files come from the
      frozen checkout, this repository's from `git show <commit>:<path>` of the commit the
      list names. A file that differs stops the run.
  clean_text.py run
      the default engine and the generator alone on the corpus, each read two ways:
        as the census   no source path given to the engine, and for the default engine no
                        inline pragma: how secretgate's hooks scan tool output, and how the
                        census of section 7.7 was counted (17 files of 8,821 at the time)
        with the path   the file's path given to the engine, as on CredData (section 6): the
                        global path allowlist then exempts node_modules, most of the corpus
      Writes <data>/eval/clean/flags/ (census reading: generator, secretgate-default) and
      <data>/eval/clean/flags-with-path/, and the generator's raw exports next to them (they
      hold VALUES and stay in the data folder).
  clean_text.py share [--out FILE]
      <data>/eval/clean/clean.json: for every system of the two flag folders, the files with
      at least one accepted flag and their share of the corpus. For a system that scores
      candidates (the model) it also writes <data>/eval/clean/<system>-accepted.jsonl: every
      accepted candidate with its file, line, rule, name and value, for the reading by hand
      (section 7.7). That file holds values and stays in the data folder.

Nothing is tuned on this corpus.
"""
import argparse
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402
import run_secretgate  # noqa: E402

MAX_BYTES = 2 * 1024 * 1024  # the corpus holds no larger file (clean-corpus.json, selection)
READINGS = {"flags": "as the census: no source path, no pragma for the default engine", "flags-with-path": "with the file's path"}


def clean_dir():
    return os.path.join(common.eval_root(), "clean")


def corpus_list():
    gen = common.load_json(common.GENERATOR_JSON)
    path = os.path.join(common.data_root(), "gen", "clean-corpus.json")
    want = gen["files"]["clean-corpus.json"]
    got = common.sha256_file(path)
    if got != want["sha256"]:
        raise common.Refused(f"{path}: sha256 {got}, frozen {want['sha256']}")
    listing = common.load_json(path)
    if listing["files"] != want["files"] or listing["listSha256"] != want["listSha256"] or len(listing["list"]) != want["files"]:
        raise common.Refused(f"{path} is not the list generator.json names")
    return listing


def corpus(listing=None):
    """The corpus as files below <data>/eval/clean/corpus, after `prepare`."""
    listing = listing or corpus_list()
    base = os.path.join(clean_dir(), "corpus")
    files = [f"{f['repository']}/{f['path']}" for f in listing["list"]]
    known = set(files)

    def check(path):
        if path not in known:
            raise common.Refused(f"{path} is not a file of the clean-text corpus")
        return "clean"

    return common.Corpus("clean", base, sorted({f["repository"] for f in listing["list"]}), files, check)


def prepare():
    listing = corpus_list()
    gen = common.check_generator()
    if listing["repositories"]["secretgate"]["commit"] != gen["secretgate"]["commit"]:
        raise common.Refused("the corpus list is pinned at another secretgate commit")
    sources = {"secretgate": common.secretgate_root()}
    skycmd_commit = listing["repositories"]["skycmd"]["commit"]
    base = os.path.join(clean_dir(), "corpus")
    n = 0
    for f in listing["list"]:
        dest = os.path.join(base, f["repository"], f["path"])
        if os.path.isfile(dest) and os.path.getsize(dest) == f["bytes"] and common.sha256_file(dest) == f["sha256"]:
            n += 1
            continue
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        if f["repository"] == "skycmd":
            data = subprocess.run(["git", "-C", common.main_checkout(), "show", f"{skycmd_commit}:{f['path']}"], capture_output=True, check=True).stdout
        elif f["repository"] in sources:
            with open(os.path.join(sources[f["repository"]], f["path"]), "rb") as src:
                data = src.read()
        else:
            raise common.Refused(f"unknown repository {f['repository']} in the corpus list")
        with open(dest, "wb") as out:
            out.write(data)
        if len(data) != f["bytes"] or common.sha256_file(dest) != f["sha256"]:
            os.remove(dest)
            raise common.Refused(f"{f['repository']}:{f['path']} is not the file the corpus list names (sha256 {f['sha256']})")
        n += 1
    return {"files": n, "bytes": listing["bytes"], "list_sha256": listing["listSha256"]}


def run():
    gen = common.check_generator()
    secretgate = common.secretgate_root()
    listing = corpus_list()
    c = corpus(listing)
    for f in listing["list"]:
        if not os.path.isfile(os.path.join(c.base, f["repository"], f["path"])):
            raise common.Refused("the corpus is not prepared (clean_text.py prepare)")
    out = {}
    for folder, census in (("flags", True), ("flags-with-path", False)):
        d = os.path.join(clean_dir(), folder)
        raw = os.path.join(d, "raw")
        os.makedirs(raw, mode=0o700, exist_ok=True)
        records = {"generator": [], "secretgate-default": []}
        cpu = {"generator": 0.0, "secretgate-default": 0.0}
        export = os.path.join(raw, "candidates.raw.jsonl")
        with open(os.open(export, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), "w") as export_f:
            for repo in c.roots:
                base = os.path.join(c.base, repo)
                rel = [f["path"] for f in listing["list"] if f["repository"] == repo]
                listing_path = os.path.join(raw, f"files-{repo}.txt")
                with open(listing_path, "w") as f:
                    f.write("\n".join(rel) + "\n")
                part = os.path.join(raw, f"candidates-{repo}.raw.jsonl")
                found = os.path.join(raw, f"default-{repo}.jsonl")
                with common.Clock() as clock:
                    if census:
                        cmd = [run_secretgate.node(), os.path.join(common.HERE, "engine.mjs"), "--secretgate", secretgate, "--mode", "candidates",
                               "--files-from", listing_path, "--base", base, "--out", part, "--max-bytes", str(MAX_BYTES)]
                    else:
                        cmd = [run_secretgate.node(), os.path.join(secretgate, "scripts", "rule-candidates.mjs"), "--candidates", "--raw",
                               "--files-from", listing_path, "--base", base, "--out", part, "--max-bytes", str(MAX_BYTES)]
                    subprocess.run(common.nice(cmd), check=True, stderr=subprocess.DEVNULL)
                cpu["generator"] += clock.children
                with common.Clock() as clock:
                    subprocess.run(common.nice([run_secretgate.node(), os.path.join(common.HERE, "engine.mjs"), "--secretgate", secretgate, "--mode", "default",
                                                "--files-from", listing_path, "--base", base, "--out", found, "--max-bytes", str(MAX_BYTES),
                                                *(["--no-path", "--no-pragmas"] if census else [])]), check=True, stderr=subprocess.DEVNULL)
                cpu["secretgate-default"] += clock.children
                for r in run_secretgate.read_jsonl(part):
                    export_f.write(json.dumps({**r, "file": f"{repo}/{r['file']}"}) + "\n")
                    records["generator"].append({**{k: r[k] for k in ("rule", "line", "endLine", "cpStart", "cpEnd")}, "file": f"{repo}/{r['file']}"})
                for r in run_secretgate.read_jsonl(found):
                    records["secretgate-default"].append({**r, "file": f"{repo}/{r['file']}"})
                os.remove(part)
        how = READINGS[folder]
        for name, unit in (("generator", "candidate"), ("secretgate-default", "span")):
            flags = run_secretgate.records_to_flags(records[name], c.base, c.check, with_id=name == "generator")
            meta = common.write_flags(d, name, flags, {
                "version": (f"sg-cand-1 revision {gen['revision']}, " if name == "generator" else "") + f"secretgate {gen['secretgate']['commit']}",
                "how": how, "corpus": "clean", "unit": unit, "files": len(c.files), "cpu_seconds": round(cpu[name], 1)})
            out[f"{folder}/{name}"] = meta["flags"]
    return out


def share_of(flags, run, n_files):
    scored = any("score" in x for x in flags) or run.get("scored", False)
    threshold = run.get("threshold")
    if scored and threshold is None:
        raise common.Refused(f"{run.get('system')}: a scored system needs a threshold")
    accepted = [x for x in flags if not scored or x.get("score", 0.0) >= threshold]
    files = {x["file"] for x in accepted}
    by_rule = {}
    for x in accepted:
        by_rule[x.get("rule")] = by_rule.get(x.get("rule"), 0) + 1
    return {"files_with_an_accepted_flag": len(files), "files": n_files, "share": len(files) / n_files if n_files else None,
            "accepted_flags": len(accepted), "flags": len(flags), "by_rule": dict(sorted(by_rule.items(), key=lambda kv: -kv[1])),
            "threshold": threshold if scored else None, "version": run.get("version")}, accepted, scored


def share(out_path=None):
    listing = corpus_list()
    c = corpus(listing)
    out = {"experiment": "s6-secrets v2 (docs/s6-secrets2-prereg.md, section 7.7)", "corpus": {"files": listing["files"], "bytes": listing["bytes"], "list_sha256": listing["listSha256"]},
           "gate_reading": "flags", "readings": {}}
    for folder, how in READINGS.items():
        d = os.path.join(clean_dir(), folder)
        if not os.path.isdir(d):
            continue
        res = {}
        for name in common.systems_in(d):
            run, flags = common.read_flags(d, name)
            if run.get("corpus") != "clean":
                raise common.Refused(f"{d}/{name}: its flags are of {run.get('corpus')}")
            for x in flags:
                c.check(x["file"])
            res[name], accepted, scored = share_of(flags, run, len(c.files))
            if scored and run.get("unit") == "candidate":
                # the accepted candidates with their values, for the reading by hand
                wanted = {x["id"]: x for x in accepted}
                path = os.path.join(clean_dir(), f"{name}-accepted{'' if folder == 'flags' else '-with-path'}.jsonl")
                with open(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), "w") as f:
                    for i, r in enumerate(run_secretgate.read_jsonl(os.path.join(d, "raw", "candidates.raw.jsonl"))):
                        if i in wanted:
                            f.write(json.dumps({"n": i, "score": wanted[i]["score"], "file": r["file"], "line": r["line"], "rule": r["rule"], "name": r["name"],
                                                "value": r["value"], "lineText": r["lineText"][:300], "looks_like_a_real_credential": None}) + "\n")
                res[name]["accepted_candidates_for_the_reading_by_hand"] = os.path.basename(path)
        out["readings"][folder] = {"how": how, "systems": res}
    common.write_json(out_path or os.path.join(clean_dir(), "clean.json"), out)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("what", choices=("prepare", "run", "share"))
    ap.add_argument("--out")
    a = ap.parse_args()
    try:
        if a.what == "prepare":
            print(json.dumps(prepare()))
        elif a.what == "run":
            print(json.dumps(run()))
        else:
            out = share(a.out)
            for folder, r in out["readings"].items():
                for n, s in r["systems"].items():
                    common.log(f"{folder:<16} {n:<20} {s['files_with_an_accepted_flag']} of {s['files']} files ({100 * s['share']:.2f}%), {s['accepted_flags']} flags")
    except common.Refused as e:
        common.log(f"clean_text: refused: {e}")
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
