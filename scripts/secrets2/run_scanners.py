#!/usr/bin/env python3
"""The third-party baselines of s6-secrets v2 (docs/s6-secrets2-prereg.md, section 6), each at
its pinned version and its default settings, on one corpus, written in the common result format.

  python3 -I scripts/secrets2/run_scanners.py <tool> --corpus creddata:validation --out DIR

  gitleaks          Gitleaks 8.30.1 (the host's binary): `gitleaks dir`, default config, JSON
                    report read from its standard output                        unit: span
  betterleaks       Betterleaks 1.10.1 in its container (no network, the corpus read-only):
                    `betterleaks dir`, default config                           unit: span
  detect-secrets    detect-secrets 1.5.0: `scan --all-files`, default plugins   unit: line
  credsweeper-ml    CredSweeper 1.19.6, ML at its default threshold (medium)    unit: span
  credsweeper-noml  CredSweeper 1.19.6, `--ml_threshold 0`                      unit: span
  cbx-17m           cbx-secrets-detector-17M at its pinned commit, in its container, on CPU,
                    the model's own cbx_secrets.py with its defaults            unit: span

A corpus is creddata:<split>, issues:<split> or issues:<split>:obfuscated; the guard of
common.py decides what may be read, and every flag's path is checked again. The reports of the
tools are read in memory, or written hashed, so that no value lands in a file; only places
(file, line, columns) are kept.

CredSweeper's ML was trained on a random 85% of all CredData lines: its CredData numbers are
in-sample and enter no gate (section 6).
"""
import argparse
import json
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402
import setup_tools  # noqa: E402

CONTAINER_CPUS = os.environ.get("SECRETS2_CPUS", "4")


def raw_dir(out, name):
    d = os.path.join(out, "raw", name)
    os.makedirs(d, mode=0o700, exist_ok=True)
    return d


class Texts:
    """The bytes and the text of the files of a corpus, one at a time."""

    def __init__(self, base):
        self.base = base
        self.path = None

    def load(self, path):
        if path != self.path:
            with open(os.path.join(self.base, path), "rb") as f:
                self.buf = f.read()
            self.text = self.buf.decode("utf-8", "replace")
            self.lines = common.Lines(self.text)
            self.nl = [0] + [m.end() for m in re.finditer(rb"\n", self.buf)]
            self.path = path
        return self

    def cp(self, byte_offset):
        """Code-point offset of a byte offset."""
        return len(self.buf[:byte_offset].decode("utf-8", "replace"))

    def nl_end(self, line):
        """Byte offset of the end of 1-based line `line` (lines that end at "\\n")."""
        return self.nl[line] - 1 if line < len(self.nl) else len(self.buf)


def locate_leak(t, f, style):
    """Byte span of the secret of one Gitleaks or Betterleaks finding in its file.

    Gitleaks counts a column from the newline before the line (so a line after the first is
    shifted by one), its end column is the last byte; Betterleaks counts 1-based bytes of the
    line, end inclusive. Both give the span of the whole match; the secret is looked for
    inside it. The match is checked against the file; when it is not where the columns say,
    it is searched in the lines the finding names, and failing that the columns are kept."""
    a_line, b_line = f["StartLine"], f["EndLine"]
    if not (1 <= a_line <= len(t.nl) and 1 <= b_line <= len(t.nl)):
        return None
    ls, le = t.nl[a_line - 1], t.nl[b_line - 1]
    if style == "gitleaks":
        a = f["StartColumn"] - 1 if a_line == 1 else ls + f["StartColumn"] - 2
        b = f["EndColumn"] if b_line == 1 else le - 1 + f["EndColumn"]
    else:
        a = ls + f["StartColumn"] - 1
        b = le + f["EndColumn"]
    match = f.get("Match", "").encode("utf-8")
    secret = f.get("Secret", "").encode("utf-8")
    found = bool(match) and t.buf[a:a + len(match)] == match
    if not found and match:
        i = t.buf.find(match, ls, t.nl_end(b_line) + 1)
        if i != -1:
            a, found = i, True
    if found:
        b = a + len(match)
        j = match.find(secret) if secret else -1
        if j != -1:
            a, b = a + j, a + j + len(secret)
    a = max(ls, min(a, len(t.buf)))
    b = max(a + 1, min(b, t.nl_end(b_line)))
    return a, b


def leaks_to_flags(findings, corpus, style, strip=""):
    texts = Texts(corpus.base)
    for f in sorted(findings, key=lambda f: (f["File"], f["StartLine"], f["StartColumn"])):
        path = f["File"][len(strip):] if strip and f["File"].startswith(strip) else f["File"]
        repo = corpus.check(path)
        t = texts.load(path)
        span = locate_leak(t, f, style)
        if span is None:
            continue
        line, start, end_line, end = t.lines.place(t.cp(span[0]), t.cp(span[1]))
        yield common.flag(repo, path, line, end_line, start, end, rule=f.get("RuleID"))


def no_local_config(corpus):
    """Default config means no config or ignore file of the corpus is picked up."""
    for d in [corpus.base] + [os.path.join(corpus.base, r) for r in corpus.roots]:
        for name in (".gitleaks.toml", ".gitleaksignore", ".betterleaks.toml", ".betterleaksignore"):
            if os.path.exists(os.path.join(d, name)):
                raise common.Refused(f"{d}/{name} would change the scanner's default configuration")


def gitleaks(corpus, out):
    version = subprocess.run(["gitleaks", "version"], capture_output=True, text=True, check=True).stdout.strip()
    if version != setup_tools.pins()["host"]["gitleaks"]:
        raise common.Refused(f"gitleaks is {version}, pinned {setup_tools.pins()['host']['gitleaks']}")
    no_local_config(corpus)
    findings = []
    with common.Clock() as clock:
        for root in corpus.roots:
            r = subprocess.run(common.nice(["gitleaks", "dir", root, "--report-format", "json", "--report-path", "-", "--no-banner",
                                            "--exit-code", "0", "--log-level", "error"]), cwd=corpus.base, capture_output=True, check=True)
            findings += json.loads(r.stdout or b"[]")
    flags = leaks_to_flags(findings, corpus, "gitleaks")
    return common.write_flags(out, "gitleaks", flags, {
        "version": f"Gitleaks {version}", "how": "gitleaks dir <repository>, default config", "corpus": corpus.name, "unit": "span",
        "files": len(corpus.files), "cpu_seconds": round(clock.children, 1), "wall_seconds": round(clock.wall, 1)})


def docker_mounts(corpus, target="/in"):
    args = []
    for root in corpus.roots:
        args += ["-v", f"{os.path.join(corpus.base, root)}:{target}/{root}:ro"]
    return args


def cpu_usec(stderr):
    m = re.search(r"usage_usec (\d+)", stderr)
    return int(m.group(1)) / 1e6 if m else None


def betterleaks(corpus, out):
    no_local_config(corpus)
    image = setup_tools.image_tag("betterleaks")
    script = ("betterleaks dir /in --report-format json --report-path - --no-banner --exit-code 0 --log-level error;"
              " status=$?; cat /sys/fs/cgroup/cpu.stat >&2; exit $status")
    with common.Clock() as clock:
        r = subprocess.run(common.nice(["docker", "run", "--rm", "--network", "none", "--read-only", "--cpus", CONTAINER_CPUS, *docker_mounts(corpus),
                                        "--entrypoint", "/bin/sh", image, "-c", script]), capture_output=True, check=True)
    version = subprocess.run(["docker", "run", "--rm", "--network", "none", image, "version"], capture_output=True, text=True, check=True).stdout.strip()
    flags = leaks_to_flags(json.loads(r.stdout or b"[]"), corpus, "betterleaks", strip="/in/")
    return common.write_flags(out, "betterleaks", flags, {
        "version": f"Betterleaks {version}", "how": f"betterleaks dir, default config, container {image}, no network", "corpus": corpus.name,
        "unit": "span", "files": len(corpus.files), "cpu_seconds": cpu_usec(r.stderr.decode("utf-8", "replace")), "wall_seconds": round(clock.wall, 1)})


def detect_secrets(corpus, out):
    exe = os.path.join(setup_tools.env_dir("detect-secrets"), "bin", "detect-secrets")
    setup_tools.python_of("detect-secrets")
    version = subprocess.run([exe, "--version"], capture_output=True, text=True, check=True).stdout.strip()
    flags = []
    with common.Clock() as clock:
        for root in corpus.roots:
            r = subprocess.run(common.nice([exe, "scan", "--all-files", root]), cwd=corpus.base, capture_output=True, check=True)
            for path, found in sorted(json.loads(r.stdout)["results"].items()):
                for x in found:
                    flags.append(common.flag(corpus.check(path), path, x["line_number"], rule=x["type"]))
    return common.write_flags(out, "detect-secrets", flags, {
        "version": f"detect-secrets {version}", "how": "detect-secrets scan --all-files <repository>, default plugins", "corpus": corpus.name,
        "unit": "line", "files": len(corpus.files), "cpu_seconds": round(clock.children, 1), "wall_seconds": round(clock.wall, 1)})


def credsweeper(corpus, out, ml):
    name = "credsweeper-ml" if ml else "credsweeper-noml"
    exe = os.path.join(setup_tools.env_dir("credsweeper"), "bin", "credsweeper")
    setup_tools.python_of("credsweeper")
    version = subprocess.run([exe, "--version"], capture_output=True, text=True, check=True).stdout.strip()
    # --hashed: line, variable and value are hashed in the report; the places stay.
    report = os.path.join(raw_dir(out, name), "report.json")
    cmd = [exe, "--path", *corpus.roots, "--save-json", report, "--hashed", "--no-stdout", "--log", "error"]
    if not ml:
        cmd += ["--ml_threshold", "0"]
    with common.Clock() as clock:
        subprocess.run(common.nice(cmd), cwd=corpus.base, check=True, stdout=subprocess.DEVNULL)
    flags = []
    for cred in common.load_json(report):
        for d in cred["line_data_list"]:
            repo = corpus.check(d["path"])
            start, end = d.get("value_start", -1), d.get("value_end", -1)
            if start is None or end is None or start < 0 or end <= start:
                flags.append(common.flag(repo, d["path"], d["line_num"], rule=cred["rule"]))
            else:
                flags.append(common.flag(repo, d["path"], d["line_num"], start=start, end=end, rule=cred["rule"]))
    flags.sort(key=lambda x: (x["file"], x["line"], x.get("start", -1)))
    return common.write_flags(out, name, flags, {
        "version": version, "how": "credsweeper --path <repositories>" + ("" if ml else " --ml_threshold 0") + (", ML at its default threshold (medium)" if ml else ", ML off"),
        "corpus": corpus.name, "unit": "span", "files": len(corpus.files),
        "in_sample_on_creddata": "its ML was trained on a random 85% of all CredData lines (section 6)" if ml else None,
        "cpu_seconds": round(clock.children, 1), "wall_seconds": round(clock.wall, 1)})


def cbx(corpus, out):
    image = setup_tools.image_tag("cbx")
    model = setup_tools.downloads_dir("cbx")
    for a in setup_tools.pins()["artifacts"]:
        if a["group"] == "cbx" and not setup_tools.check_artifact(a):
            raise common.Refused(f"{a['file']} of the cbx model is not fetched")
    commit = next(a["version"] for a in setup_tools.pins()["artifacts"] if a["group"] == "cbx")
    work = raw_dir(out, "cbx-17m")
    with open(os.path.join(work, "files.txt"), "w") as f:
        f.write("\n".join(corpus.files) + "\n")
    repos = sorted({m.group(1) for m in (re.match(r"^data/([0-9a-f]{8})$", r) for r in corpus.roots) if m})
    with common.Clock() as clock:
        subprocess.run(common.nice(["docker", "run", "--rm", "--network", "none", "--read-only", "--tmpfs", "/tmp", "--cpus", CONTAINER_CPUS,
                                    "-v", f"{model}:/model:ro", "-v", f"{os.path.join(common.HERE, 'cbx_infer.py')}:/work/cbx_infer.py:ro",
                                    *docker_mounts(corpus), "-v", f"{work}:/out", "--user", f"{os.getuid()}:{os.getgid()}", "-e", "HOME=/tmp",
                                    image, "python", "-I", "/work/cbx_infer.py", "/model", "/in", "/out", CONTAINER_CPUS, *repos]), check=True)
    stats = common.load_json(os.path.join(work, "stats.json"))
    texts = Texts(corpus.base)

    def flags():
        with open(os.path.join(work, "spans.jsonl")) as f:
            for line in f:
                s = json.loads(line)
                repo = corpus.check(s["file"])
                t = texts.load(s["file"])
                ln, start, end_line, end = t.lines.place(s["start"], s["end"])
                yield common.flag(repo, s["file"], ln, end_line, start, end, rule=s["label"])

    return common.write_flags(out, "cbx-17m", flags(), {
        "version": f"cbx-secrets-detector-17M at {commit}", "how": f"SecretDetector().extract(text) per file, CPU, {stats['threads']} threads, torch {stats['torch']}, container {image}, no network",
        "corpus": corpus.name, "unit": "span", "files": len(corpus.files), "files_read": stats["read"], "characters": stats["characters"],
        "cpu_seconds": stats["cpu_seconds"], "wall_seconds": stats["wall_seconds"], "max_rss_mb": stats["max_rss_mb"]})


TOOLS = {"gitleaks": gitleaks, "betterleaks": betterleaks, "detect-secrets": detect_secrets,
         "credsweeper-ml": lambda c, o: credsweeper(c, o, True), "credsweeper-noml": lambda c, o: credsweeper(c, o, False), "cbx-17m": cbx}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("tool", choices=sorted(TOOLS))
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    try:
        meta = TOOLS[a.tool](common.corpus(a.corpus), a.out)
    except common.Refused as e:
        common.log(f"run_scanners: refused: {e}")
        return 3
    common.log(f"{meta['system']}: {meta['flags']} flags on {meta['corpus']}, {meta['cpu_seconds']} s of CPU")
    return 0


if __name__ == "__main__":
    sys.exit(main())
