#!/usr/bin/env python3
"""Shared by the evaluation scripts of s6-secrets v2 (docs/s6-secrets2-prereg.md).

What lives here: where the data folder and the frozen generator are, the split guard every
runner goes through, the frozen hashes, the common result format, and the small helpers the
runners share (CredData's line convention, hashing, CPU time of a child process).

The data folder is outside every repository and nothing in it is ever committed:
  SECRETS2_DATA        default: <parent of the main checkout>/skycmd-data/s6-secrets2
  SECRETS2_SECRETGATE  default: <parent of the main checkout>/secretgate/.claude/worktrees/gen

The split guard. A runner names the split it works on and gets its repositories from
`splits.json`; every path it reads or writes a flag for goes through `Split.check`, which
refuses a RepoID of another split. The test split (CredData test repositories, the
issue-report test) is refused by `open_split` unless `runs/s6-secrets2/TEST-OPEN` exists in
this checkout and SECRETS2_MODEL names a GGUF whose metadata holds the threshold (section 5:
tau* is stored before any test file is read).

The common result format ("s6-secrets2-flags-1"). For one system on one corpus:
  <system>.flags.jsonl   one line per flag (a finding, or a candidate with its score)
      repo      RepoID (CredData), "issues" or "clean"
      file      path below the corpus base, e.g. data/<RepoID>/src/<FileID>.py
      line      1-based line of the flag's first character, lines cut as CredData cuts them
                ("\r\n", "\r" and "\n" all end a line)
      end_line  optional, 1-based line of its last character (default: line)
      start,end optional columns in Unicode code points (Python indices): start on `line`,
                end (exclusive) on `end_line`. Without them the flag is its whole line(s)
      score     optional, P(real) in [0, 1]; a system without a score accepts every flag
      rule, id  optional, informative (rule id, index of the candidate in its export)
  <system>.run.json      what ran: format, system, version, corpus, unit (candidate, span or
                         line), threshold (scored systems), files, cpu_seconds, wall_seconds
No value is ever written to either file.
"""
import hashlib
import json
import os
import re
import resource
import subprocess
import sys
import time

FORMAT = "s6-secrets2-flags-1"
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
FROZEN = os.path.join(HERE, "frozen.json")
GENERATOR_JSON = os.path.join(REPO, "runs", "s6-secrets2", "generator.json")
TEST_OPEN = os.path.join(REPO, "runs", "s6-secrets2", "TEST-OPEN")
# The metadata key of the operating threshold tau* in the model's GGUF (section 5); the F0.5
# and F2 presets sit next to it.
TAU_KEY = "skycmd-secrets2.tau"
TAU_PRESET_KEYS = {"f0.5": "skycmd-secrets2.tau_f0.5", "f2": "skycmd-secrets2.tau_f2"}
CREDDATA_SPLITS = ("training", "validation", "test")
ISSUE_SPLITS = {"training": "train", "validation": "val", "test": "test"}


class Refused(Exception):
    """A guard said no: wrong split, test not open, a frozen hash that differs."""


def _main_parent():
    try:
        common = subprocess.run(["git", "-C", REPO, "rev-parse", "--path-format=absolute", "--git-common-dir"],
                                capture_output=True, text=True, check=True).stdout.strip()
        return os.path.dirname(os.path.dirname(common))
    except (OSError, subprocess.CalledProcessError):
        return os.path.dirname(REPO)


def data_root():
    return os.environ.get("SECRETS2_DATA") or os.path.join(_main_parent(), "skycmd-data", "s6-secrets2")


def secretgate_root():
    return os.environ.get("SECRETS2_SECRETGATE") or os.path.join(_main_parent(), "secretgate", ".claude", "worktrees", "gen")


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
        json.dump(obj, f, indent=1, sort_keys=False)
        f.write("\n")
    os.replace(tmp, path)


def frozen():
    return load_json(FROZEN)


# ---------------------------------------------------------------- GGUF metadata

_GGUF_SCALARS = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}


def gguf_metadata(path):
    """The metadata keys of a GGUF file (v2 or v3), without its tensors."""
    import struct

    with open(path, "rb") as f:
        def take(fmt):
            size = struct.calcsize(fmt)
            buf = f.read(size)
            if len(buf) != size:
                raise ValueError(f"{path}: truncated GGUF header")
            return struct.unpack(fmt, buf)[0]

        def string():
            n = take("<Q")
            return f.read(n).decode("utf-8", "replace")

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
        version = take("<I")
        if version not in (2, 3):
            raise ValueError(f"{path}: GGUF version {version}")
        take("<Q")  # tensors
        out = {}
        for _ in range(take("<Q")):
            key = string()
            out[key] = value(take("<I"))
        return out


def model_tau(path):
    """tau* of a model file, or Refused when its metadata does not hold one."""
    if not path or not os.path.isfile(path):
        raise Refused(f"no model file ({path!r})")
    tau = gguf_metadata(path).get(TAU_KEY)
    if not isinstance(tau, float) or not 0.0 < tau < 1.0:
        raise Refused(f"{path}: no threshold in its metadata ({TAU_KEY}); tau* is stored before any test file is read")
    return tau


# ---------------------------------------------------------------- frozen things

def check_generator(secretgate=None):
    """The generator checkout is the frozen one: commit, clean tree, the four hashed files; and
    the copy of generator.json in the data folder is the committed one. Returns generator.json."""
    gen = load_json(GENERATOR_JSON)
    root = secretgate or secretgate_root()
    git = lambda *a: subprocess.run(["git", "-C", root, *a], capture_output=True, text=True, check=True).stdout.strip()
    try:
        head = git("rev-parse", "HEAD")
        dirty = git("status", "--porcelain")
    except (OSError, subprocess.CalledProcessError) as e:
        raise Refused(f"{root}: not a git checkout of secretgate ({e})")
    want = gen["secretgate"]["commit"]
    if head != want:
        raise Refused(f"{root} is at {head}; the generator is frozen at {want}")
    if dirty:
        raise Refused(f"{root} has uncommitted changes")
    for rel, digest in gen["sha256"].items():
        got = sha256_file(os.path.join(root, rel))
        if got != digest:
            raise Refused(f"{root}/{rel}: sha256 {got}, frozen {digest}")
    local = os.path.join(data_root(), "gen", "generator.json")
    if sha256_file(local) != sha256_file(GENERATOR_JSON):
        raise Refused(f"{local} is not the committed runs/s6-secrets2/generator.json")
    return gen


def check_generator_files(split):
    """The frozen generator outputs of a training or validation split, by their hashes."""
    gen = load_json(GENERATOR_JSON)
    for name in (f"candidates-{split}.raw.jsonl", f"labels-{split}.jsonl", f"files-{split}.txt"):
        path = os.path.join(data_root(), "gen", name)
        got = sha256_file(path)
        if got != gen["files"][name]["sha256"]:
            raise Refused(f"{path}: sha256 {got}, frozen {gen['files'][name]['sha256']}")


# ---------------------------------------------------------------- splits

class Split:
    """The repositories of one CredData split, and the guard on every path."""

    def __init__(self, name, splits):
        self.name = name
        self.repos = sorted(r["repo_id"] for r in splits[name])
        self._set = set(self.repos)
        self._other = {r["repo_id"]: s for s in CREDDATA_SPLITS if s != name for r in splits[s]}
        if self._set & set(self._other):
            raise Refused(f"splits.json lists a repository of {name} in another split")
        self.largest = None
        if name == "test":
            self.largest = splits["largest_test_repository"]["repo_id"]
            if self.largest not in self._set:
                raise Refused("largest_test_repository is not a test repository")

    def check_repo(self, repo):
        if repo not in self._set:
            where = self._other.get(repo)
            raise Refused(f"RepoID {repo} is not in the {self.name} split" + (f" (it is a {where.upper()} repository)" if where else ""))
        return repo

    def check(self, path):
        """The RepoID of `data/<RepoID>/...`, after the guard."""
        m = re.match(r"^data/([0-9a-f]{8})/", path)
        if not m:
            raise Refused(f"{path} is not a file of a CredData repository")
        return self.check_repo(m.group(1))


def load_splits():
    path = os.path.join(data_root(), "splits.json")
    got = sha256_file(path)
    want = frozen()["splits.json"]["sha256"]
    if got != want:
        raise Refused(f"{path}: sha256 {got}, frozen {want}")
    return load_json(path)


def test_is_open(model=None):
    """Refused unless the owner opened the test and a model with tau* is given."""
    if not os.path.isfile(TEST_OPEN):
        raise Refused("the test split is closed: runs/s6-secrets2/TEST-OPEN does not exist (its owner creates it by hand after training)")
    return model_tau(model or os.environ.get("SECRETS2_MODEL"))


def open_split(name, model=None):
    """A CredData split after its guards. `test` needs TEST-OPEN and a model with tau*."""
    if name not in CREDDATA_SPLITS:
        raise Refused(f"unknown split {name!r} (training, validation, test)")
    if name == "test":
        test_is_open(model)
    return Split(name, load_splits())


def issue_csv(split, model=None):
    """The issue-report file of a split, after the guard and its hash."""
    if split not in ISSUE_SPLITS:
        raise Refused(f"unknown split {split!r}")
    if split == "test":
        test_is_open(model)
    name = ISSUE_SPLITS[split] + ".csv"
    path = os.path.join(data_root(), "issues", "unzipped", "Secret-Leak-Detection-Issue-Report", "Data", name)
    want = frozen()["issues"][name]
    got = sha256_file(path)
    if got != want:
        raise Refused(f"{path}: sha256 {got}, frozen {want}")
    return path


_LISTING = None


def creddata_listing():
    """sha256 of every data file, from the listing written at download (its own hash is frozen)."""
    global _LISTING
    if _LISTING is None:
        path = os.path.join(data_root(), "logs", "creddata-data.sha256")
        want = frozen()["creddata"]["data_listing_sha256"]
        got = sha256_file(path)
        if got != want:
            raise Refused(f"{path}: sha256 {got}, frozen {want}")
        _LISTING = {}
        with open(path) as f:
            for line in f:
                digest, rel = line.rstrip("\n").split("  ", 1)
                _LISTING[rel] = digest
    return _LISTING


def creddata_base():
    return os.path.join(data_root(), "creddata")


def creddata_files(split, verify=True):
    """Every file of the split's repositories as `data/<RepoID>/...`, in the generator's order
    (repositories sorted, names sorted at each directory). With `verify`, each file's sha256
    is checked against the download listing: a changed or added file stops the run."""
    base = creddata_base()
    out = []

    def walk(rel):
        with os.scandir(os.path.join(base, rel)) as it:
            entries = sorted(it, key=lambda e: e.name)
        for e in entries:
            if e.is_symlink():
                continue
            if e.is_dir():
                walk(f"{rel}/{e.name}")
            elif e.is_file():
                out.append(f"{rel}/{e.name}")

    for repo in split.repos:
        walk(f"data/{repo}")
    for rel in out:
        split.check(rel)
    if verify:
        listing = creddata_listing()
        for rel in out:
            want = listing.get(rel)
            if want is None:
                raise Refused(f"{rel} is not in the download listing")
            got = sha256_file(os.path.join(base, rel))
            if got != want:
                raise Refused(f"{rel}: sha256 {got}, listed {want}")
    return out


class Corpus:
    """What a file scanner is given: a base directory, the roots below it that it may scan,
    the files below them, and the guard that names the repository of a path or refuses it."""

    def __init__(self, name, base, roots, files, check):
        self.name = name
        self.base = base
        self.roots = list(roots)
        self.files = list(files)
        self.check = check
        for f in self.files:
            check(f)
            if not any(f == r or f.startswith(r + "/") for r in self.roots):
                raise Refused(f"{f} is outside the roots of {name}")


def creddata_corpus(split_name, model=None):
    split = open_split(split_name, model)
    files = creddata_files(split)
    return Corpus(f"creddata:{split_name}", creddata_base(), [f"data/{r}" for r in split.repos], files, split.check)


def issues_dir(split, model=None):
    """<data>/eval/issues/<split>, after the guard of the split."""
    if split not in ISSUE_SPLITS:
        raise Refused(f"unknown split {split!r}")
    if split == "test":
        test_is_open(model)
    return os.path.join(eval_root(), "issues", split)


def issues_corpus(split, obfuscated=False, model=None):
    """The issue texts of a split as files (scripts/secrets2/issues.py prepare writes them)."""
    base = issues_dir(split, model)
    root = "texts-obfuscated" if obfuscated else "texts"
    index = load_json(os.path.join(base, "index.json"))
    files = [f"{root}/{t}.txt" for t in index["texts"]]

    def check(path):
        if not re.match(rf"^{root}/[0-9a-f]{{16}}\.txt$", path):
            raise Refused(f"{path} is not an issue text of {split}")
        return "issues"

    return Corpus(f"issues:{split}" + (":obfuscated" if obfuscated else ""), base, [root], files, check)


def corpus(name, model=None):
    """creddata:<split>, issues:<split> or issues:<split>:obfuscated."""
    parts = name.split(":")
    if parts[0] == "creddata" and len(parts) == 2:
        return creddata_corpus(parts[1], model)
    if parts[0] == "issues" and len(parts) in (2, 3) and parts[2:] in ([], ["obfuscated"]):
        return issues_corpus(parts[1], len(parts) == 3, model)
    raise Refused(f"unknown corpus {name!r}")


# ---------------------------------------------------------------- text and lines

def read_text(path):
    """A file's text as the generator reads it (UTF-8, invalid bytes replaced), or None for a
    file it skips: empty, or a NUL byte in its first 8192 bytes."""
    with open(path, "rb") as f:
        buf = f.read()
    if not buf or b"\0" in buf[:8192]:
        return None
    return buf.decode("utf-8", "replace")


_EOL = re.compile(r"\r\n|\r|\n")


class Lines:
    """CredData's lines of a text (obfuscate_creds.py: "\\r\\n" and "\\r" become "\\n", then the
    text is split on "\\n"): where each line starts and where its content ends, as code-point
    offsets in the text as it is on disk."""

    def __init__(self, text):
        self.starts = [0]
        self.ends = []
        for m in _EOL.finditer(text):
            self.ends.append(m.start())
            self.starts.append(m.end())
        self.ends.append(len(text))

    def __len__(self):
        return len(self.starts)

    def line_of(self, offset):
        """0-based index of the line that holds `offset`."""
        lo, hi = 0, len(self.starts) - 1
        while lo < hi:
            mid = (lo + hi + 1) >> 1
            if self.starts[mid] <= offset:
                lo = mid
            else:
                hi = mid - 1
        return lo

    def offset(self, line, col):
        """Offset of column `col` of 1-based `line`, not past the line's end; None outside the text."""
        i = line - 1
        if not 0 <= i < len(self.starts):
            return None
        return min(self.starts[i] + max(col, 0), self.ends[i])

    def place(self, start, end):
        """(line, col, end_line, end_col) of the span [start, end): 1-based lines, columns in
        code points, the end column on the line of the span's last character."""
        a = self.line_of(start)
        b = self.line_of(max(start, end - 1))
        return a + 1, start - self.starts[a], b + 1, end - self.starts[b]


class EngineLines:
    """Places given by secretgate's engine (1-based lines that end at "\\n" only, columns in
    code points) turned into CredData's lines of the same text."""

    def __init__(self, text):
        self.lines = Lines(text)
        self.starts = [0]
        i = text.find("\n")
        while i != -1:
            self.starts.append(i + 1)
            i = text.find("\n", i + 1)

    def place(self, line, cp_start, end_line, cp_end):
        """(line, col, end_line, end_col) in CredData's lines."""
        return self.lines.place(self.starts[line - 1] + cp_start, self.starts[end_line - 1] + cp_end)


# ---------------------------------------------------------------- flags

def flag(repo, file, line, end_line=None, start=None, end=None, score=None, rule=None, id=None):
    out = {"repo": repo, "file": file, "line": int(line)}
    if end_line is not None and end_line != line:
        out["end_line"] = int(end_line)
    if start is not None:
        out["start"] = int(start)
        out["end"] = int(end)
    if score is not None:
        out["score"] = round(float(score), 6)
    if rule is not None:
        out["rule"] = rule
    if id is not None:
        out["id"] = id
    return out


def flags_paths(out_dir, system):
    return os.path.join(out_dir, f"{system}.flags.jsonl"), os.path.join(out_dir, f"{system}.run.json")


def write_flags(out_dir, system, flags, run):
    """Writes the two files of one system. `run` needs version, corpus and unit."""
    os.makedirs(out_dir, exist_ok=True)
    fpath, rpath = flags_paths(out_dir, system)
    tmp = fpath + ".tmp"
    n = 0
    with open(tmp, "w") as f:
        for x in flags:
            f.write(json.dumps(x, ensure_ascii=True, separators=(",", ":")) + "\n")
            n += 1
    os.replace(tmp, fpath)
    for k in ("version", "corpus", "unit"):
        if k not in run:
            raise ValueError(f"run.json of {system}: no {k}")
    if run["unit"] not in ("candidate", "span", "line"):
        raise ValueError(f"run.json of {system}: unit {run['unit']!r}")
    meta = {"format": FORMAT, "system": system, **run, "flags": n, "flags_sha256": sha256_file(fpath)}
    write_json(rpath, meta)
    return meta


def read_flags(out_dir, system):
    """(run, flags) of one system; refuses a flags file that is not the one its run.json names."""
    fpath, rpath = flags_paths(out_dir, system)
    run = load_json(rpath)
    if run.get("format") != FORMAT:
        raise Refused(f"{rpath}: format {run.get('format')!r}")
    if sha256_file(fpath) != run["flags_sha256"]:
        raise Refused(f"{fpath} is not the file {rpath} describes")
    flags = []
    with open(fpath) as f:
        for line in f:
            if line.strip():
                flags.append(json.loads(line))
    return run, flags


def systems_in(out_dir):
    return sorted(n[: -len(".run.json")] for n in os.listdir(out_dir) if n.endswith(".run.json"))


# ---------------------------------------------------------------- child processes

class Clock:
    """CPU seconds (user + system) of the child processes started inside the block, and wall
    seconds. A tool that runs in a container reports its own CPU time instead."""

    def __enter__(self):
        self._c = resource.getrusage(resource.RUSAGE_CHILDREN)
        self._s = resource.getrusage(resource.RUSAGE_SELF)
        self._t = time.monotonic()
        return self

    def __exit__(self, *exc):
        c = resource.getrusage(resource.RUSAGE_CHILDREN)
        s = resource.getrusage(resource.RUSAGE_SELF)
        self.children = (c.ru_utime - self._c.ru_utime) + (c.ru_stime - self._c.ru_stime)
        self.own = (s.ru_utime - self._s.ru_utime) + (s.ru_stime - self._s.ru_stime)
        self.wall = time.monotonic() - self._t
        return False


def nice(cmd):
    return ["nice", "-n", "19", *cmd]


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def check_creddata():
    """The CredData checkout is at the pinned commit and its meta/ is that commit's."""
    root = creddata_base()
    git = lambda *a: subprocess.run(["git", "-C", root, *a], capture_output=True, text=True, check=True).stdout.strip()
    want = frozen()["creddata"]["commit"]
    head = git("rev-parse", "HEAD")
    if head != want:
        raise Refused(f"{root} is at {head}, pinned {want}")
    if git("status", "--porcelain", "--", "meta"):
        raise Refused(f"{root}/meta differs from the pinned commit")


def check_all(split, model=None):
    """Every guard of a run on one split: the frozen generator, the frozen files, the split."""
    check_generator()
    check_creddata()
    s = open_split(split, model)
    if split != "test":
        check_generator_files(split)
    issue_csv(split, model)
    return s


if __name__ == "__main__":
    # python3 -I scripts/secrets2/common.py check <split>: exit 0 when a run on the split may start.
    if len(sys.argv) == 3 and sys.argv[1] == "check":
        try:
            s = check_all(sys.argv[2])
        except Refused as e:
            log(f"refused: {e}")
            sys.exit(3)
        print(f"{s.name}: {len(s.repos)} repositories; generator, splits and frozen files are the frozen ones")
    else:
        log("usage: common.py check <training|validation|test>")
        sys.exit(2)
