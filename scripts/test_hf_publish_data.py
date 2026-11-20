#!/usr/bin/env python3
"""Tests of skycmd-route7's checks in scripts/hf-publish-data.py (`--only route7`): the hash
refusal (a file whose sha256 differs from the one runs/s6-route7 records), the licence refusals (a
row whose source has no licence entry, a source without its row in DATA_LICENSES.md, a file of the
data folder without an entry or a reason), the row-by-row checks of the files no run file hashes
(the blind copies, the teacher's rewrites) and the absolute-path guard of the copied files, on an
invented fixture of the data folder's layout; the merge of a published manifest
(`--merge-published`) and the refusal of an upload that would drop published configs; the
agreement of skycmd-glm-labels recomputed from its two files and the refused count it requires.
No pyarrow and no network: the parquet writer is replaced by a recorder, and the published
manifests are invented. python3 -I scripts/test_hf_publish_data.py"""

import contextlib
import hashlib
import importlib.util
import io
import json
import os
import tempfile
import unittest
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = Path(HERE).parent

spec = importlib.util.spec_from_file_location("hf_publish_data", os.path.join(HERE, "hf-publish-data.py"))
hpd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hpd)

LICENCES = (ROOT / "DATA_LICENSES.md").read_text()
NONE = "none of these tools fits the request"
# invented tools: two of the training pool (v2's stage), two of T5 (Toucan's stage)
POOL = {"ta": ("lights_on", "Turns the lights on in a room."), "tb": ("get_weather", "Gives the weather forecast for a city.")}
TOUCAN = {"tc": ("search_posts", "Search the posts of a forum."), "td": ("get_post", "Get one post by its id.")}
REWRITE_V2 = {"ta": "Switches on the lights of a given room."}
REWRITE_V4 = {"tb": "Tells the weather forecast for a given city."}
REWRITE_T5 = {"td": "Fetches a single forum post from its identifier."}


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def write_jsonl(p: Path, rows) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))


def write_json(p: Path, obj) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=1) + "\n")


def slate(request, tools, descs, split, names=None, state="User: turn on the lights", **extra):
    names = names or [(POOL.get(t) or TOUCAN.get(t))[0] for t in tools]
    r = {"src": "x", "t": "choice", "ins": "Which tool should handle the user's request?",
         "opts": [f"{n}: {d}" for n, d in zip(names, descs)] + [NONE], "state": state,
         "target": [1.0] + [0.0] * len(tools), "split": split, "uid": request, "kind": "test", "lang": "en",
         "request": request, "tools": tools, "cell_level": 0, "roles": ["right"] + ["drawn"] * (len(tools) - 1)}
    r.update(extra)
    return r


def scored(r, scores=None):
    n = len(r["tools"])
    scores = scores or [0.0] * n
    return {**r, "opts": [f"score {s:.2f} rank {i + 1}; {o}" for i, (s, o) in enumerate(zip(scores, r["opts"][:n]))]
            + r["opts"][n:]}


def blind(r):
    return {**r, "state": "User: ?"}


def masked(r):
    n = len(r["tools"])
    return {**r, "kind": "masked", "opts": [f"tool_{i + 1}: {o.split(': ', 1)[1]}" for i, o in enumerate(r["opts"][:n])]
            + r["opts"][n:]}


def build(root: Path, mutate=None) -> tuple:
    """The fixture: a data folder and a run folder that pass every check. `mutate(data)` runs after
    the data files are written and before the run files record their hashes."""
    data, runs = root / "data", root / "runs"
    write_jsonl(data / "build/stage/tools.jsonl", [{"id": t, "name": n, "desc": d} for t, (n, d) in POOL.items()])
    write_jsonl(data / "build/stage/requests.jsonl", [{"source": "glaive", "tools": ["ta", "tb"], "gold": "ta"}])
    write_jsonl(data / "build/teacher/rewrite-description.jsonl", [{"id": k, "description": v} for k, v in REWRITE_V2.items()])
    write_jsonl(data / "build/teacher/describe-intent.jsonl", [{"id": "massive:lights", "description": "Turn lights on."}])
    write_jsonl(data / "build/fresh/f_tools.jsonl", [slate("glaive:9", ["ta", "tb"], [POOL["ta"][1], POOL["tb"][1]], "fresh")])
    write_jsonl(data / "build4/teacher/rewrite-description.jsonl", [{"id": k, "description": v} for k, v in REWRITE_V4.items()])
    for x in ("items", "refused"):
        write_jsonl(data / f"build4/teacher/rewrite-description.{x}.jsonl", [])
    # the records: the first option shown as v2's rewrite
    rec = scored(slate("glaive:1", ["ta", "tb"], [REWRITE_V2["ta"], POOL["tb"][1]], "train",
                       kind="natural", rewritten=[True, False]), [1.2, 0.0])
    for g in ("tools", "intents", "when2call"):
        for split in ("train", "val"):
            write_jsonl(data / f"build7/records/{g}/{split}.jsonl", [{**rec, "split": split}])
    v5 = slate("glaive:2", ["ta", "tb"], [POOL["ta"][1], POOL["tb"][1]], "val")
    rw = {**v5, "opts": [v5["opts"][0], f"get_weather: {REWRITE_V4['tb']}", NONE], "follow": 1}
    files = {"v5_tools": v5, "v5_tools_masked": masked(v5), "v5_tools_rw": rw, "v5_tools_masked_rw": masked(rw)}
    for name, r in files.items():
        write_jsonl(data / f"build7/val/{name}.jsonl", [r])
        write_jsonl(data / f"build7/val/{name}.model.jsonl", [scored(r)])
    for name in ("v4_tools", "v4_tools_masked", "v_tools", "v_tools_masked", "r_tools_natural", "r_tools_renamed"):
        write_jsonl(data / f"build7/val/{name}.model.jsonl", [scored(v5)])
    for name in ("v5_tools", "v5_tools_masked"):
        b = blind(files[name])
        write_jsonl(data / f"build7/val/{name}.blind.jsonl", [b])
        write_jsonl(data / f"build7/val/{name}.blind-bare.model.jsonl", [b])
        write_jsonl(data / f"build7/val/{name}.blind.model.jsonl", [scored(b)])
        for fix in ("fix", "fix7"):
            write_jsonl(data / f"build7/{fix}/val/{name}.jsonl", [files[name]])
    for d in ("teacher", "fix/teacher", "fix7/teacher"):
        for x in ("false-negatives.items", "false-negatives", "false-negatives.refused"):
            write_jsonl(data / f"build7/{d}/{x}.jsonl", [])
    for d in ("stage", "fix/stage", "fix7/stage"):
        write_jsonl(data / f"build7/{d}/plan5.jsonl", [])
    for d in ("fix/stage", "fix7/stage"):
        write_jsonl(data / f"build7/{d}/false-negatives5.items.jsonl", [])
    write_json(data / "build7/counts/v5-plan.json", {"about": "plan"})
    write_json(data / "build7/counts/v5-slates.json", {"about": "slates"})
    # T5
    write_jsonl(data / "toucan5/stage/tools.jsonl",
                [{"id": t, "name": n, "desc": d, "servers": ["forum-server"]} for t, (n, d) in TOUCAN.items()])
    write_jsonl(data / "toucan5/stage/rows.jsonl", [])
    write_jsonl(data / "toucan5/stage/texts.jsonl", [])
    write_jsonl(data / "toucan5/teacher/rewrite-description.jsonl", [{"id": k, "description": v} for k, v in REWRITE_T5.items()])
    for x in ("items", "refused"):
        write_jsonl(data / f"toucan5/teacher/rewrite-description.{x}.jsonl", [])
    t5 = slate("toucan:1", ["tc", "td"], [TOUCAN["tc"][1], TOUCAN["td"][1]], "fresh", state="User: find the post")
    t5 = {**t5, "lang": "und"}
    t5rw = {**t5, "opts": [t5["opts"][0], f"get_post: {REWRITE_T5['td']}", NONE], "follow": 1}
    t5files = {"t5_tools": t5, "t5_tools_masked": masked(t5), "t5_tools_rw": t5rw, "t5_tools_masked_rw": masked(t5rw)}
    for name, r in t5files.items():
        write_jsonl(data / f"toucan5/fresh/{name}.jsonl", [r])
    for name in ("t5_tools", "t5_tools_masked"):
        b = blind(t5files[name])
        write_jsonl(data / f"toucan5/fresh/{name}.blind.jsonl", [b])
        write_jsonl(data / f"toucan5/fresh/{name}.blind-bare.model.jsonl", [b])
        write_jsonl(data / f"toucan5/fresh/{name}.blind.model.jsonl", [scored(b)])
    write_jsonl(data / "toucan5/fresh/t5_tools.servers.jsonl", [{"uid": "toucan:1", "server": "forum-server", "cluster": "k1"}])
    write_json(data / "MANIFEST4.json", {"about": "fetch"})
    if mutate:
        mutate(data)
    # the records of the hashes, written last
    write_json(data / "build/BUILD.json", {"outputs": {
        k: {"sha256": sha(data / "build" / k)}
        for k in ("teacher/describe-intent.jsonl", "teacher/rewrite-description.jsonl", "stage/requests.jsonl", "stage/tools.jsonl")}})
    write_json(data / "build/fresh/FRESH.json", {"files": {"f_tools.jsonl": sha(data / "build/fresh/f_tools.jsonl")}})
    for name in ("v5_tools", "v5_tools_masked"):
        write_json(data / f"build7/logs7/blind-{name}.json", {"files": {
            f"{name}{x}": sha(data / f"build7/val/{name}{x}") for x in (".blind.jsonl", ".blind.model.jsonl", ".blind-bare.model.jsonl")}})
    b7 = [str(p.relative_to(data / "build7")) for d in ("records", "val", "counts") for p in (data / "build7" / d).rglob("*")
          if p.is_file() and ".blind" not in p.name]
    write_json(runs / "inputs6.json", {"files": {k: {"sha256": sha(data / "build7" / k)} for k in sorted(b7)}})
    t5h = {n: sha(data / "toucan5/fresh" / n) for n in
           ("t5_tools.jsonl", "t5_tools.servers.jsonl", "t5_tools_masked.jsonl", "t5_tools_rw.jsonl", "t5_tools_masked_rw.jsonl")}
    write_json(runs / "test-files.json", {"files": t5h})
    write_json(runs / "FRESH-OPEN", {"T5": t5h})
    write_json(runs / "t5-counts.json", {"about": "counts"})
    write_json(runs / "placeholder.json", {"placeholder": "?"})
    return data, runs


class Route7(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.written = {}

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, recs, path, slates):
        self.written[path.name] = len(recs)

    def assemble(self, data, runs, licences=LICENCES):
        return hpd.route7_assemble(data, self.root / "out", self.write, runs=runs, licences=licences)

    def refused(self, data, runs, *words, licences=LICENCES):
        with self.assertRaises(SystemExit) as e:
            self.assemble(data, runs, licences)
        for w in words:
            self.assertIn(w, str(e.exception))
        return str(e.exception)

    def test_fixture_passes(self):
        data, runs = build(self.root)
        r = self.assemble(data, runs)
        self.assertEqual(set(r["rows"]), set(hpd.ROUTE7_FILES) | set(hpd.ROUTE7_TEACHER_FILES) | {hpd.ROUTE7_SERVERS[0]})
        self.assertEqual(r["rows"]["tools_train"], 1)
        self.assertEqual(r["manifest"]["build7/records/tools/train.jsonl"]["check"], "sha256 recorded in runs/s6-route7/inputs6.json")
        self.assertEqual(r["v4_used"], 1)
        self.assertEqual(r["t5_used"], 1)

    # the hash refusal
    def test_refuses_a_training_file_whose_hash_differs(self):
        data, runs = build(self.root)
        with open(data / "build7/records/tools/train.jsonl", "a") as f:
            f.write("\n")
        self.refused(data, runs, "build7/records/tools/train.jsonl", "runs/s6-route7/inputs6.json", "refused")

    def test_refuses_a_test_file_whose_hash_differs(self):
        data, runs = build(self.root)
        (data / "toucan5/fresh/t5_tools_masked.jsonl").write_text("{}\n")
        self.refused(data, runs, "toucan5/fresh/t5_tools_masked.jsonl", "FRESH-OPEN", "refused")

    def test_refuses_when_test_files_and_fresh_open_disagree(self):
        data, runs = build(self.root)
        fo = json.loads((runs / "FRESH-OPEN").read_text())
        fo["T5"]["t5_tools.jsonl"] = "0" * 64
        write_json(runs / "FRESH-OPEN", fo)
        self.refused(data, runs, "test-files.json and FRESH-OPEN record different hashes")

    def test_refuses_a_file_read_twice_that_differs(self):
        data, runs = build(self.root)
        with open(data / "build7/fix7/val/v5_tools.jsonl", "a") as f:
            f.write("\n")
        self.refused(data, runs, "build7/fix7/val/v5_tools.jsonl", "the same bytes")

    def test_refuses_a_skycmd_route_file_whose_hash_differs(self):
        data, runs = build(self.root)
        with open(data / "build/teacher/describe-intent.jsonl", "a") as f:
            f.write("\n")
        self.refused(data, runs, "build/teacher/describe-intent.jsonl", "BUILD.json")

    # the licence refusals
    def test_refuses_a_row_whose_source_has_no_licence_entry(self):
        def unlicensed(data):
            p = data / "build7/records/tools/train.jsonl"
            r = json.loads(p.read_text())
            write_jsonl(p, [{**r, "request": "tool-routing-data:1", "uid": "tool-routing-data:1"}])

        data, runs = build(self.root, unlicensed)
        self.refused(data, runs, "1 rows without a licence entry")
        self.assertFalse((self.root / "out").exists())

    def test_refuses_an_option_whose_source_has_no_licence_entry(self):
        def unlicensed(data):
            write_jsonl(data / "build/stage/requests.jsonl", [{"source": "glaive", "tools": ["ta"], "gold": "ta"},
                                                              {"source": "metatool", "tools": ["tb"], "gold": None}])

        data, runs = build(self.root, unlicensed)
        self.refused(data, runs, "rows without a licence entry")

    def test_refuses_a_source_without_its_row_in_data_licenses(self):
        data, runs = build(self.root)
        text = "\n".join(ln for ln in LICENCES.splitlines() if not ln.startswith("| Agent-Ark/Toucan-1.5M"))
        self.refused(data, runs, "no licence row in DATA_LICENSES.md for ['toucan']", licences=text)

    def test_the_repository_licence_table_has_every_source(self):
        hpd.route7_licences(LICENCES)

    def test_refuses_a_file_without_an_entry_or_a_reason(self):
        data, runs = build(self.root)
        write_jsonl(data / "build7/val/v6_tools.jsonl", [])
        self.refused(data, runs, "no licence entry for ['build7/val/v6_tools.jsonl']")

    # the files no run file hashes
    def test_refuses_a_blind_copy_that_does_not_follow_from_its_parent(self):
        data, runs = build(self.root)
        p = data / "toucan5/fresh/t5_tools.blind.jsonl"
        r = json.loads(p.read_text())
        write_jsonl(p, [{**r, "state": "User: find the post"}])
        write_jsonl(data / "toucan5/fresh/t5_tools.blind-bare.model.jsonl", [{**r, "state": "User: find the post"}])
        self.refused(data, runs, "do not match the file they derive from")

    def test_refuses_a_teacher_file_that_does_not_match_the_records(self):
        data, runs = build(self.root)
        write_jsonl(data / "build4/teacher/rewrite-description.jsonl", [{"id": "tb", "description": "Something else."}])
        self.refused(data, runs, "do not match the file they derive from")

    def test_refuses_a_copied_file_with_a_path_of_this_machine(self):
        data, runs = build(self.root)
        write_json(data / "MANIFEST4.json", {"file": str(Path.home() / "x")})
        self.refused(data, runs, "do not match")

    def test_refuses_a_key_like_string(self):
        def leak(data):
            p = data / "toucan5/fresh/t5_tools.jsonl"
            r = json.loads(p.read_text())
            write_jsonl(p, [{**r, "state": "User: my key is sk_live_" + "Ab3dE5gH7jK9mN1pQ3"}])

        data, runs = build(self.root, leak)
        self.refused(data, runs, "1 key-like strings")


class Helpers(unittest.TestCase):
    def test_description_of_an_option(self):
        self.assertEqual(hpd.route7_desc("score 1.07 rank 1; Chart: Songs: Fetches a chart.", "Chart: Songs"), "Fetches a chart.")
        self.assertEqual(hpd.route7_desc("tool_3: Fetches a chart.", "Chart: Songs"), "Fetches a chart.")

    def test_cut(self):
        self.assertTrue(hpd.route7_cut_of("Fetches a chart.", "Fetches a chart. Then more."))
        self.assertFalse(hpd.route7_cut_of("Fetches a list.", "Fetches a chart."))
        self.assertFalse(hpd.route7_cut_of(None, "Fetches a chart."))

    def test_v7_schema_keeps_v2_fields_in_order(self):
        names = hpd.ROUTE_FIELDS[:-2] + hpd.ROUTE7_FIELDS + hpd.ROUTE_FIELDS[-2:]
        self.assertEqual(names[:len(hpd.ROUTE_FIELDS) - 2], hpd.ROUTE_FIELDS[:-2])
        self.assertEqual(names[-2:], ["source", "option_sources"])

    def test_head_commit_names_a_full_sha(self):
        # a dry run only warns about uncommitted changes, so this holds in any working tree
        self.assertRegex(hpd.head_commit(True), r"^[0-9a-f]{40}$")

    def test_leftovers_are_listed_not_deleted(self):
        class Api:
            def list_repo_files(self, repo_id, repo_type):
                return ["README.md", "data/kept.parquet", "data/old.parquet"]

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            (out / "data").mkdir()
            (out / "data" / "kept.parquet").write_bytes(b"")
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                hpd.warn_remote_leftovers(Api(), "u/set", out)
        self.assertIn("data/old.parquet", buf.getvalue())
        self.assertNotIn("data/kept.parquet", buf.getvalue())


OLD, NEW, REV = "a" * 40, "b" * 40, "c" * 40


def published(**entries):
    """A published MANIFEST.json written at commit OLD."""
    return {"commit": OLD, "source_files": entries}


class MergePublished(unittest.TestCase):
    """--merge-published and the guard against an upload that drops published configs."""

    def test_carries_what_the_local_build_lacks(self):
        rows = {"typed_decisions_train": 6, "typed_decisions_bench": 2}
        manifest = {"typed_decisions_train.jsonl": {"sha256": "t", "rows": 6},
                    "typed_decisions_bench.jsonl": {"sha256": "b", "rows": 2}}
        pub = published(**{"boolq.jsonl": {"sha256": "q", "rows": 12}, "arc.jsonl": {"sha256": "r", "rows": 7}})
        carried, extras, rebuilt = hpd.merge_published("decisions", hpd.DECISIONS, rows, manifest, pub, REV)
        self.assertEqual(sorted(carried), ["arc", "boolq"])
        self.assertEqual(rebuilt, ["typed_decisions_bench", "typed_decisions_train"])
        self.assertEqual(extras, [])
        self.assertEqual(rows["boolq"], 12)
        self.assertEqual(manifest["boolq.jsonl"], {"sha256": "q", "rows": 12, "carried_from": {"commit": OLD, "revision": REV}})
        self.assertNotIn("carried_from", manifest["typed_decisions_train.jsonl"])

    def test_local_files_win(self):
        rows, manifest = {"boolq": 13}, {"boolq.jsonl": {"sha256": "new", "rows": 13}}
        pub = published(**{"boolq.jsonl": {"sha256": "old", "rows": 12}})
        carried, _, rebuilt = hpd.merge_published("decisions", hpd.DECISIONS, rows, manifest, pub, REV)
        self.assertEqual(carried, {})
        self.assertEqual(rebuilt, ["boolq"])
        self.assertEqual(manifest["boolq.jsonl"], {"sha256": "new", "rows": 13})
        self.assertEqual(rows, {"boolq": 13})

    def test_an_entry_carried_twice_keeps_its_first_origin(self):
        first = {"commit": "d" * 40, "revision": "e" * 40}
        pub = published(**{"arc.jsonl": {"sha256": "r", "rows": 7, "carried_from": first}})
        rows, manifest = {"boolq": 1}, {"boolq.jsonl": {"sha256": "q", "rows": 1}}
        carried, _, _ = hpd.merge_published("decisions", hpd.DECISIONS, rows, manifest, pub, REV)
        self.assertEqual(carried["arc"], first)
        self.assertEqual(manifest["arc.jsonl"]["carried_from"], first)

    def test_extra_files_are_carried_without_rows(self):
        known = {"en": None}
        pub = published(**{"en.jsonl": {"sha256": "x", "rows": 3}, "tok.json": {"sha256": "y"}})
        rows, manifest = {}, {}
        carried, extras, _ = hpd.merge_published("text", known, rows, manifest, pub, REV)
        self.assertEqual(list(carried), ["en"])
        self.assertEqual(extras, ["tok.json"])
        self.assertNotIn("tok", rows)

    def test_refuses_a_carried_config_without_a_licence_entry(self):
        pub = published(**{"gone.jsonl": {"sha256": "x", "rows": 3}})
        with self.assertRaises(SystemExit) as e:
            hpd.merge_published("decisions", hpd.DECISIONS, {"boolq": 1}, {"boolq.jsonl": {}}, pub, REV)
        self.assertIn("['gone']", str(e.exception))

    def test_upload_refused_when_the_build_lacks_published_configs(self):
        pub = published(**{"boolq.jsonl": {"sha256": "q", "rows": 12}, "arc.jsonl": {"sha256": "r", "rows": 7},
                           "tok.json": {"sha256": "y"}})
        with self.assertRaises(SystemExit) as e:
            hpd.check_not_dropping("decisions", ["typed_decisions_train", "typed_decisions_bench"], pub)
        msg = str(e.exception)
        self.assertIn("missing ['arc', 'boolq']", msg)
        self.assertIn("--merge-published", msg)
        self.assertNotIn("tok", msg)

    def test_upload_allowed_when_every_published_config_is_rebuilt(self):
        pub = published(**{"boolq.jsonl": {"sha256": "q", "rows": 12}})
        hpd.check_not_dropping("decisions", ["boolq", "arc"], pub)

    def test_card_names_rebuilt_and_carried_configs_and_their_commits(self):
        rows = {"typed_decisions_train": 6, "typed_decisions_bench": 2, "boolq": 12, "arc": 7}
        carried = {c: {"commit": OLD, "revision": REV} for c in ("boolq", "arc")}
        text = hpd.card("decisions", hpd.SETS["decisions"], rows, NEW, "u", [], carried)
        self.assertIn("`typed_decisions_bench` and `typed_decisions_train` are written at commit [`bbbbbbbb`]", text)
        self.assertIn("2 configs are carried unchanged from the release written at commit [`aaaaaaaa`]", text)
        carried["arc"] = {"commit": "d" * 40, "revision": REV}
        text = hpd.card("decisions", hpd.SETS["decisions"], rows, NEW, "u", [], carried)
        self.assertIn("from earlier releases (1 at commit [`aaaaaaaa`]", text)
        self.assertIn("1 at commit [`dddddddd`]", text)
        for c in rows:
            self.assertIn(f"- config_name: {c}\n", text)
            self.assertIn(f"| `data/{c}.parquet` |", text)
        self.assertIn("test split of the public benchmark", text)
        self.assertIn("never train on it", text)
        self.assertNotIn("gated", text)
        self.assertNotIn("Not republished", text)

    def test_card_without_carried_configs_names_one_commit(self):
        text = hpd.card("decisions", hpd.SETS["decisions"], {"boolq": 12}, NEW, "u")
        self.assertIn("at commit [`bbbbbbbb`]", text)
        self.assertNotIn("carried", text)
        self.assertNotIn("test split of the public benchmark", text)


class GlmLabels(unittest.TestCase):
    """skycmd-glm-labels without agreement.json and glm_labels.refused.jsonl."""

    LABELS = [{"src": "boolq"}, {"src": "boolq"}, {"src": "boolq"}, {"src": "typed_decisions_train"},
              {"src": "typed_decisions_train"}, {"src": "tasksource:x/y"}]
    AGREE = [{"src": "glm:boolq"}, {"src": "glm:boolq"}, {"src": "glm:typed_decisions_train"}]

    def test_agreement_is_recomputed_per_source(self):
        a = hpd.compute_agreement(self.LABELS, self.AGREE)
        self.assertEqual(list(a), ["boolq", "tasksource:x/y", "typed_decisions_train"])
        self.assertEqual(a["boolq"], {"labelled": 3, "agree": 2, "rate": 2 / 3})
        self.assertEqual(a["tasksource:x/y"], {"labelled": 1, "agree": 0, "rate": 0.0})
        self.assertEqual(a["typed_decisions_train"]["rate"], 0.5)
        # the published file's layout: keys in order, two-space indent, floats as Python writes them
        self.assertEqual(json.dumps({"boolq": a["boolq"]}, indent=2),
                         '{\n  "boolq": {\n    "labelled": 3,\n    "agree": 2,\n    "rate": 0.6666666666666666\n  }\n}')

    def test_agreement_refuses_a_kept_record_without_a_label(self):
        with self.assertRaises(SystemExit) as e:
            hpd.compute_agreement(self.LABELS, self.AGREE + [{"src": "glm:arc"}])
        self.assertIn("['arc']", str(e.exception))

    def test_refused_count_is_required_when_the_file_is_gone(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "glm_labels.refused.jsonl"
            with self.assertRaises(SystemExit) as e:
                hpd.refused_count(p, None)
            self.assertIn("--glm-refused", str(e.exception))
            self.assertEqual(hpd.refused_count(p, 121), 121)
            write_jsonl(p, [{}, {}])
            self.assertEqual(hpd.refused_count(p, None), 2)
            self.assertEqual(hpd.refused_count(p, 2), 2)
            with self.assertRaises(SystemExit):
                hpd.refused_count(p, 121)

    def test_every_labelled_source_has_a_licence(self):
        self.assertEqual(hpd.glm_source("typed_decisions_train")[1], "apache-2.0")
        up, lic = hpd.glm_source("calibrated:mteb/banking77")
        self.assertIn("m-lagnajit/calibrated-typed-decisions", up)
        self.assertEqual(lic, "cc-by-4.0 for the set; the texts per row, see `calibrated_typed`")
        self.assertIn("source_license", hpd.glm_source("calibrated_typed")[1])
        self.assertEqual(hpd.glm_source("tasksource:x/y")[1], "per row, see `tasksource_jev`")
        for k in ("unknown", "nowhere:x", "tasksource:"):
            with self.assertRaises(SystemExit):
                hpd.glm_source(k)

    def test_no_decision_source_is_left_out(self):
        self.assertFalse(hasattr(hpd, "EXCLUDED"))
        self.assertIn("evaluation only", hpd.DECISIONS["typed_decisions_bench"][2])


if __name__ == "__main__":
    unittest.main()
