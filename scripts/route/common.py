"""Shared paths and guards of the s6-route-nano scripts (docs/s6-nano2-prereg.md).

The data folder is outside every repository: ROUTE_DATA, default `<parent of the main
checkout>/skycmd-data/s6-route`. Its layout:

    src/<source>/        the pinned downloads (scripts/route/pins.json), each with .skycmd-fetch.json
    models/<model>/      potion-base-2M, all-MiniLM-L6-v2, toolrouter
    envs/route/          the uv environment of scripts/route/env (its lock is committed)
    build/               the data build (scripts/build-route-data.sh): records/, val/, test/, ...
    eval/<split>/        the evaluation's dumps and results (scripts/route-eval.sh)

The test split is refused unless runs/s6-nano2/TEST-OPEN exists (its owner creates it by hand
after training) and the model file holds tau_ask in its metadata; every test file must also
have the sha256 that runs/s6-nano2/test-files.json records (written by the build, committed
with the amendment of section 4.6).

The fresh split of s6-nano v3 (docs/s6-nano3-prereg.md, section 4; <build>/fresh/f_*.jsonl) is
refused unless runs/s6-nano3/FRESH-OPEN exists and every fresh file has the sha256 that
runs/s6-nano3/test-files.json records.
"""

import hashlib
import json
import os
import struct
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
PINS = os.path.join(HERE, "pins.json")
ENV_PROJECT = os.path.join(HERE, "env")
TEST_OPEN = os.path.join(REPO, "runs", "s6-nano2", "TEST-OPEN")
TEST_FILES = os.path.join(REPO, "runs", "s6-nano2", "test-files.json")
TAU_KEY = "skycmd-s2.route.tau_ask"
NONE_OPTION = "none of these tools fits the request"
INSTRUCTION = "Which tool should handle the user's request?"
# s6-nano v3 (docs/s6-nano3-prereg.md, section 4): the fresh files, under <build>/fresh, refused
# until runs/s6-nano3/FRESH-OPEN exists (its owner creates it after the count amendment) and
# every file has the sha256 that runs/s6-nano3/test-files.json records.
FRESH_OPEN = os.path.join(REPO, "runs", "s6-nano3", "FRESH-OPEN")
FRESH_FILES = os.path.join(REPO, "runs", "s6-nano3", "test-files.json")
SPLITS = ("val", "test", "fresh")


class Refused(Exception):
    """A guard said no: the test is closed, a file is not the frozen one."""


def main_checkout():
    """The main checkout of this repository (a worktree shares its data folder)."""
    try:
        common = subprocess.run(["git", "-C", REPO, "rev-parse", "--path-format=absolute", "--git-common-dir"],
                                capture_output=True, text=True, check=True).stdout.strip()
        return os.path.dirname(common)
    except (OSError, subprocess.CalledProcessError):
        return REPO


def data_root():
    return os.environ.get("ROUTE_DATA") or os.path.join(os.path.dirname(main_checkout()), "skycmd-data", "s6-route")


def build_root():
    return os.environ.get("BUILD") or os.path.join(data_root(), "build")


def model_dir(name):
    return os.path.join(data_root(), "models", name)


def eval_root():
    return os.path.join(data_root(), "eval")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_json(path):
    with open(path) as f:
        return json.load(f)


def write_json(path, obj):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
        f.write("\n")
    os.replace(tmp, path)


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    os.replace(tmp, path)


def fetched(path):
    """The `.skycmd-fetch.json` of a download folder: repo, commit, files."""
    return load_json(os.path.join(path, ".skycmd-fetch.json"))


# ---------------------------------------------------------------- GGUF metadata

_GGUF_SCALARS = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}


def gguf_metadata(path):
    """The metadata keys of a GGUF file (v2 or v3), without its tensors."""
    with open(path, "rb") as f:
        def take(fmt):
            size = struct.calcsize(fmt)
            buf = f.read(size)
            if len(buf) != size:
                raise ValueError(f"{path}: truncated GGUF header")
            return struct.unpack(fmt, buf)[0]

        def string():
            return f.read(take("<Q")).decode("utf-8", "replace")

        def value(kind):
            if kind in _GGUF_SCALARS:
                return take(_GGUF_SCALARS[kind])
            if kind == 8:
                return string()
            if kind == 9:
                inner = take("<I")
                return [value(inner) for _ in range(take("<Q"))]
            raise ValueError(f"{path}: unknown GGUF value type {kind}")

        if f.read(4) != b"GGUF":
            raise ValueError(f"{path}: not a GGUF file")
        if take("<I") not in (2, 3):
            raise ValueError(f"{path}: GGUF version")
        take("<Q")
        return {string(): value(take("<I")) for _ in range(take("<Q"))}


def model_tau(path):
    """tau_ask of a model file, or Refused when its metadata does not hold one."""
    if not path or not os.path.isfile(path):
        raise Refused(f"no model file ({path!r})")
    tau = gguf_metadata(path).get(TAU_KEY)
    if not isinstance(tau, float) or not 0.0 <= tau <= 1.0:
        raise Refused(f"{path}: no {TAU_KEY} in its metadata; tau_ask is stored before any test file is read")
    return tau


# ---------------------------------------------------------------- the files of a split

def test_hashes():
    """The sha256 of every test file, as runs/s6-nano2/test-files.json records them."""
    if not os.path.isfile(TEST_FILES):
        raise Refused("runs/s6-nano2/test-files.json does not exist: the test files' hashes are recorded before any training")
    return load_json(TEST_FILES)["files"]


def test_is_open(model):
    if not os.path.isfile(TEST_OPEN):
        raise Refused("the test split is closed: runs/s6-nano2/TEST-OPEN does not exist (its owner creates it by hand after training)")
    return model_tau(model)


def slate_file(split, name, model=None):
    """The slate file `name` (e.g. `t_tools`, `t_tools_masked`; `v_tools` on validation) of a
    split, after its guards. The test needs TEST-OPEN, a model with tau_ask and the recorded hash."""
    if split not in SPLITS:
        raise Refused(f"unknown split {split!r} ({', '.join(SPLITS)})")
    if split == "val":
        if not name.startswith("v_"):
            raise Refused(f"{name} is not a validation file")
        path = os.path.join(build_root(), "val", f"{name}.jsonl")
    elif split == "fresh":
        return fresh_file(name)
    else:
        if not name.startswith("t_"):
            raise Refused(f"{name} is not a test file")
        test_is_open(model)
        path = os.path.join(build_root(), "test", f"{name}.jsonl")
        want = test_hashes().get(f"{name}.jsonl")
        if want is None or sha256_file(path) != want:
            raise Refused(f"{path} is not the test file recorded in runs/s6-nano2/test-files.json")
    if not os.path.isfile(path):
        raise Refused(f"no {path}")
    return path


def fresh_hashes():
    """The sha256 of every fresh file, as runs/s6-nano3/test-files.json records them."""
    if not os.path.isfile(FRESH_FILES):
        raise Refused("runs/s6-nano3/test-files.json does not exist: the fresh files are hashed before any scoring")
    return load_json(FRESH_FILES)["files"]


def fresh_is_open():
    if not os.path.isfile(FRESH_OPEN):
        raise Refused("the fresh split is closed: runs/s6-nano3/FRESH-OPEN does not exist "
                      "(its owner creates it after the hashes and the count amendment are committed)")


def fresh_file(name):
    """A fresh file of s6-nano v3 (`f_tools`, `f_tools_masked`, `f_metatool`, `f_clinc`,
    `f_massive`, `f_tools.clusters`), after its guards: FRESH-OPEN and the recorded sha256."""
    if not name.startswith("f_"):
        raise Refused(f"{name} is not a fresh file")
    fresh_is_open()
    want = fresh_hashes().get(f"{name}.jsonl")
    path = os.path.join(build_root(), "fresh", f"{name}.jsonl")
    if want is None:
        raise Refused(f"{name}.jsonl is not recorded in runs/s6-nano3/test-files.json")
    if not os.path.isfile(path):
        raise Refused(f"no {path}")
    if sha256_file(path) != want:
        raise Refused(f"{path} is not the fresh file recorded in runs/s6-nano3/test-files.json")
    return path


def records_dirs():
    b = build_root()
    return [os.path.join(b, "records", g) for g in ("tools", "intents", "when2call")]


def training_records(split="train"):
    """Every training (or validation) record of the three groups."""
    out = []
    for d in records_dirs():
        p = os.path.join(d, f"{split}.jsonl")
        if os.path.isfile(p):
            out.extend(read_jsonl(p))
    return out


def nice(cmd):
    return ["nice", "-n", "19", *cmd]
