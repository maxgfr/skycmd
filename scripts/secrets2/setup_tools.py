#!/usr/bin/env python3
"""The tools of the baselines of s6-secrets v2 (docs/s6-secrets2-prereg.md, section 6), each at
its pinned version, and the record of every download.

  python3 -I scripts/secrets2/setup_tools.py [fetch] [envs] [images] [record]   (default: all)

  fetch    the files of scripts/secrets2/pins.json (the Betterleaks release, the cbx model and
           its code, potion-base-2M) into <data>/eval/downloads/<group>/; a file whose size or
           sha256 differs from its pin stops the run
  envs     one uv environment per Python tool, from the lock next to its pyproject.toml
           (scripts/secrets2/envs/<name>/uv.lock), created in <data>/eval/envs/<name>
  images   two Docker images, from base images pinned by digest: Betterleaks (the release
           binary on alpine) and the environment of cbx-secrets-detector-17M (the lock of
           scripts/secrets2/envs/cbx). Both run without a network.
  record   <data>/eval/DOWNLOADS.json: every file fetched, with its URL, version or commit,
           size and sha256: the pinned files, the base images, and the wheel of every package
           installed in each environment (read from the locks)

Nothing here reads a data file of the experiment.
"""
import itertools
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tomllib
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

HOST_ENVS = ("detect-secrets", "credsweeper", "lr", "potion")
ENVS = os.path.join(common.HERE, "envs")
DOCKER = os.path.join(common.HERE, "docker")


def pins():
    return common.load_json(os.path.join(common.HERE, "pins.json"))


def downloads_dir(group):
    return os.path.join(common.eval_root(), "downloads", group)


def env_dir(name):
    return os.path.join(common.eval_root(), "envs", name)


def image_tag(name):
    if name == "betterleaks":
        return "s6secrets2-betterleaks:1.10.1"
    lock = common.sha256_file(os.path.join(ENVS, "cbx", "uv.lock"))[:12]
    return f"s6secrets2-cbx:{lock}"


def check_artifact(a):
    path = os.path.join(downloads_dir(a["group"]), a["file"])
    if not os.path.isfile(path):
        return None
    if os.path.getsize(path) != a["bytes"] or common.sha256_file(path) != a["sha256"]:
        raise common.Refused(f"{path} is not the pinned file ({a['bytes']} bytes, sha256 {a['sha256']})")
    return path


def fetch():
    for a in pins()["artifacts"]:
        if check_artifact(a):
            continue
        path = os.path.join(downloads_dir(a["group"]), a["file"])
        os.makedirs(os.path.dirname(path), exist_ok=True)
        common.log(f"fetch {a['url']}")
        with urllib.request.urlopen(a["url"], timeout=600) as r, open(path + ".part", "wb") as f:
            shutil.copyfileobj(r, f)
        os.replace(path + ".part", path)
        check_artifact(a)


def uv_env(name):
    """The environment variables that make uv use the environment of `name` outside the repository."""
    return {**os.environ, "UV_PROJECT_ENVIRONMENT": env_dir(name), "UV_PYTHON_PREFERENCE": "only-managed"}


def python_of(name):
    """The interpreter of an environment; refuses when the lock was not installed."""
    path = os.path.join(env_dir(name), "bin", "python")
    if not os.path.isfile(path):
        raise common.Refused(f"{env_dir(name)}: no environment (scripts/secrets2/setup_tools.py envs)")
    return path


def envs():
    for name in HOST_ENVS:
        subprocess.run(common.nice(["uv", "sync", "--frozen", "--no-dev", "--project", os.path.join(ENVS, name), "-q"]),
                       env=uv_env(name), check=True)


def images():
    p = pins()
    for a in p["artifacts"]:
        if not check_artifact(a):
            raise common.Refused(f"{a['file']} is not fetched")
    ctx = os.path.join(common.eval_root(), "docker", "betterleaks")
    shutil.rmtree(ctx, ignore_errors=True)
    os.makedirs(ctx)
    with tarfile.open(os.path.join(downloads_dir("betterleaks"), "betterleaks_1.10.1_linux_arm64.tar.gz")) as tar:
        member = tar.getmember("betterleaks")
        if not member.isfile():
            raise common.Refused("the Betterleaks tarball does not hold a plain file named betterleaks")
        with tar.extractfile(member) as src, open(os.path.join(ctx, "betterleaks"), "wb") as dst:
            shutil.copyfileobj(src, dst)
    os.chmod(os.path.join(ctx, "betterleaks"), 0o755)
    subprocess.run(common.nice(["docker", "build", "-q", "--build-arg", f"ALPINE={p['images']['alpine']}", "-t", image_tag("betterleaks"),
                                "-f", os.path.join(DOCKER, "betterleaks", "Dockerfile"), ctx]), check=True)
    subprocess.run(common.nice(["docker", "build", "-q", "--build-arg", f"PYTHON={p['images']['python']}", "--build-arg", f"UV={p['images']['uv']}",
                                "-t", image_tag("cbx"), "-f", os.path.join(DOCKER, "cbx", "Dockerfile"), os.path.join(ENVS, "cbx")]), check=True)


# The wheels a lock installed: the wheel of each installed distribution is the one of the lock
# whose file name carries the tags its WHEEL file lists.
_INSTALLED = r'''
import glob, json, os, sys
out = {}
for d in glob.glob(os.path.join(sys.argv[1], "lib", "python*", "site-packages", "*.dist-info")):
    name = None
    for line in open(os.path.join(d, "METADATA"), encoding="utf-8", errors="replace"):
        if line.startswith("Name: "):
            name = line[6:].strip()
            break
    wheel = os.path.join(d, "WHEEL")
    out[name] = sorted(x[5:].strip() for x in open(wheel) if x.startswith("Tag: ")) if os.path.exists(wheel) else []
print(json.dumps(out))
'''


def norm(name):
    return name.lower().replace("_", "-").replace(".", "-")


def wheel_tags(filename):
    parts = filename[: -len(".whl")].split("-")
    py, abi, plat = parts[-3:]
    return sorted(f"{a}-{b}-{c}" for a, b, c in itertools.product(py.split("."), abi.split("."), plat.split(".")))


def packages(name, installed):
    """The wheels of one environment: lock entries matched to the installed distributions."""
    with open(os.path.join(ENVS, name, "uv.lock"), "rb") as f:
        lock = tomllib.load(f)
    tags = {norm(k): sorted(x.strip() for x in v) for k, v in installed.items()}
    out = []
    for pkg in lock["package"]:
        if pkg.get("source", {}).get("virtual"):
            continue
        want = tags.get(norm(pkg["name"]))
        if want is None:
            continue
        chosen = None
        for w in pkg.get("wheels", []):
            fn = w["url"].rsplit("/", 1)[-1]
            if wheel_tags(urllib.request.unquote(fn)) == want:
                chosen = w
                break
        if chosen is None and pkg.get("sdist"):
            chosen = pkg["sdist"]
        entry = {"name": pkg["name"], "version": pkg["version"]}
        if chosen:
            entry.update({"url": chosen["url"], "bytes": chosen.get("size"), "sha256": chosen["hash"].split(":", 1)[-1]})
        else:
            entry["note"] = "no wheel of the lock carries the installed tags"
        out.append(entry)
    return sorted(out, key=lambda e: e["name"])


def record():
    p = pins()
    out = {"what": "s6-secrets v2: every file fetched for the evaluation side (scripts/secrets2/setup_tools.py). The owner of the experiment approved these downloads.",
           "artifacts": [], "docker": {}, "python_environments": {}}
    for a in p["artifacts"]:
        check_artifact(a)
        out["artifacts"].append({k: a[k] for k in ("group", "file", "url", "version", "licence", "bytes", "sha256")})
    inspect = lambda ref, fmt: subprocess.run(["docker", "image", "inspect", ref, "--format", fmt], capture_output=True, text=True).stdout.strip()
    for key in ("python", "uv", "alpine"):
        out["docker"][key] = {"tag": p["images"][key + "_tag"], "digest": p["images"][key], "bytes": int(inspect(p["images"][key], "{{.Size}}") or 0) or None}
    for name in ("betterleaks", "cbx"):
        out["docker"][f"built:{name}"] = {"tag": image_tag(name), "image_id": inspect(image_tag(name), "{{.Id}}") or None}
    for name in HOST_ENVS:
        r = subprocess.run([python_of(name), "-I", "-c", _INSTALLED, env_dir(name)], capture_output=True, text=True, check=True)
        pk = packages(name, json.loads(r.stdout))
        out["python_environments"][name] = {"lock": f"scripts/secrets2/envs/{name}/uv.lock", "lock_sha256": common.sha256_file(os.path.join(ENVS, name, "uv.lock")),
                                            "where": "host, uv", "packages": pk, "bytes": sum(e.get("bytes") or 0 for e in pk)}
    r = subprocess.run(["docker", "run", "--rm", "--network", "none", image_tag("cbx"), "python", "-I", "-c", _INSTALLED, "/opt/venv"],
                       capture_output=True, text=True)
    if r.returncode == 0:
        pk = packages("cbx", json.loads(r.stdout))
        out["python_environments"]["cbx"] = {"lock": "scripts/secrets2/envs/cbx/uv.lock", "lock_sha256": common.sha256_file(os.path.join(ENVS, "cbx", "uv.lock")),
                                             "where": f"container {image_tag('cbx')}", "packages": pk, "bytes": sum(e.get("bytes") or 0 for e in pk)}
    uv_python = subprocess.run(["uv", "python", "find", "3.12"], capture_output=True, text=True).stdout.strip()
    out["host_tools"] = {"gitleaks": subprocess.run(["gitleaks", "version"], capture_output=True, text=True).stdout.strip(),
                         "uv": subprocess.run(["uv", "--version"], capture_output=True, text=True).stdout.strip(),
                         "python_of_the_environments": uv_python, "note": "already installed: nothing was fetched for them"}
    common.write_json(os.path.join(common.eval_root(), "DOWNLOADS.json"), out)
    total = sum(a["bytes"] for a in out["artifacts"]) + sum(e["bytes"] for e in out["python_environments"].values())
    common.log(f"DOWNLOADS.json: {len(out['artifacts'])} pinned files, {sum(len(e['packages']) for e in out['python_environments'].values())} packages, {total} bytes")


def main():
    steps = sys.argv[1:] or ["fetch", "envs", "images", "record"]
    try:
        for s in steps:
            {"fetch": fetch, "envs": envs, "images": images, "record": record}[s]()
    except KeyError:
        common.log(__doc__)
        return 2
    except common.Refused as e:
        common.log(f"setup_tools: refused: {e}")
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
