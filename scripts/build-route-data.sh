#!/usr/bin/env bash
# Build the data of s6-route-nano (docs/s6-nano2-prereg.md, section 4). No training, no GPU, no
# evaluation: CPU only, everything at nice -n 19. The data folder is outside every repository.
#
# Usage: scripts/build-route-data.sh [step ...]   (default: every step below, in order)
#   fetch      every download of scripts/route/pins.json at its pinned commit, each file checked
#              against the hash its host publishes (sky-data fetch-model); <data>/MANIFEST.json
#              records URL, commit, bytes, sha256 and licence of each file
#   intents    the intent names the teacher describes (<build>/teacher/describe-intent.items.jsonl)
#   requests   every source read (needs the teacher's descriptions: scripts/route-teacher.sh
#              describe-intent)
#   embed      scripts/route-embed.py in the pinned environment (all-MiniLM-L6-v2, potion-base-2M)
#   split      clusters, splits, caps; the test files (written once, never read again here);
#              the validation slates; the plan; the items of the two other teacher jobs
#   slates     the training and validation records (needs scripts/route-teacher.sh
#              false-negatives and rewrite-description; PROVISIONAL=1 builds without them, for a
#              smoke run and the rehearsal only: such a build is never trained on)
#   truncation sky-train s2-truncation on the training and validation records, for each size and
#              head window 160, 256, 320 (section 3; SKY_TRAIN names a built sky-train: it reads
#              records only, no GPU)
#   manifest   <build>/BUILD.json (counts, hashes, the head window of each size) and
#              runs/s6-nano2/test-files.json (the test files' sha256, for the amendment of 4.6)
#
# The same inputs and the same seed give the same bytes (checked: two builds, identical hashes).
#
#   ROUTE_DATA=../skycmd-data/s6-route   BUILD=$ROUTE_DATA/build   SEED=1234
#   DECISIONS=data/decisions   MASSIVE=data/commands/massive.jsonl   (on disk, from v1)
#   SKY_TRAIN=$CARGO_TARGET_DIR/release/sky-train   PROVISIONAL=0
set -euo pipefail
cd "$(dirname "$0")/.."
export DEVELOPER_DIR=${DEVELOPER_DIR:-/Applications/Xcode.app/Contents/Developer}
export CARGO_TARGET_DIR=${CARGO_TARGET_DIR:-$PWD/target/s3-train}
data=${ROUTE_DATA:-../skycmd-data/s6-route}
build=${BUILD:-$data/build}
seed=${SEED:-1234}
decisions=${DECISIONS:-data/decisions}
massive=${MASSIVE:-data/commands/massive.jsonl}
train=${SKY_TRAIN:-$CARGO_TARGET_DIR/release/sky-train}
export ROUTE_DATA=$data BUILD=$build
steps=("$@")
[ ${#steps[@]} -eq 0 ] && steps=(fetch intents requests embed split slates truncation manifest)

nice -n 19 cargo build -q --release -p sky-data
D=$CARGO_TARGET_DIR/release/sky-data
src=(--data "$data" --when2call "$decisions/when2call.jsonl" --banking77 "$decisions/banking77.jsonl" --massive "$massive")
mkdir -p "$build/logs"
# step <name> <command...>: at low priority; what it prints (counts only) goes to <build>/logs.
step() {
  local name=$1
  shift
  echo "== $name"
  nice -n 19 "$@" >"$build/logs/$name.json"
}

fetch() {
  python3 -I -c '
import json, sys
p = json.load(open(sys.argv[1]))
for s in p["sources"] + p["models"]:
    print(s["name"], s["kind"], s["repo"], s["rev"], ",".join(s["include"]))' scripts/route/pins.json |
    while read -r name kind repo rev inc; do
      local dir=$data/src/$name flag=()
      [ "$kind" = model ] && dir=$data/models/$name
      [ "$kind" = dataset ] && flag=(--dataset)
      [ "$kind" = github ] && flag=(--github)
      echo "== fetch $name"
      nice -n 19 "$D" --cache "$data/hf-cache" fetch-model "$repo" --rev "$rev" ${flag[@]+"${flag[@]}"} \
        --include "$inc" --out "$dir" --max-gb 1
    done
  # the record of every download: URL, commit, bytes, sha256, licence
  python3 -I - "$data" scripts/route/pins.json <<'EOF'
import json, os, sys
data, pins = sys.argv[1:]
p = json.load(open(pins))
out, total = [], 0
for s in p["sources"] + p["models"]:
    d = os.path.join(data, "models" if s["kind"] == "model" else "src", s["name"])
    f = json.load(open(os.path.join(d, ".skycmd-fetch.json")))
    host = "https://github.com/" if s["kind"] == "github" else ("https://huggingface.co/datasets/" if s["kind"] == "dataset" else "https://huggingface.co/")
    for x in f["files"]:
        total += x["size"]
        out.append({"source": s["name"], "url": f"{host}{s['repo']}", "commit": f["commit"], "file": x["path"],
                    "bytes": x["size"], "sha256": x["sha256"], "git_sha1": x["git_sha1"], "licence": s["license"], "role": s["role"]})
json.dump({"about": "s6-route-nano: every download of scripts/route/pins.json", "bytes": total, "files": out},
          open(os.path.join(data, "MANIFEST.json"), "w"), indent=1)
print(f"{len(out)} files, {total / 1e6:.1f} MB")
if total > 3e9:
    sys.exit("the downloads exceed 3 GB")
EOF
}

intents() { step intents "$D" --seed "$seed" route intents "${src[@]}" --out "$build"; }

requests() {
  [ -s "$build/teacher/describe-intent.jsonl" ] || {
    echo "no $build/teacher/describe-intent.jsonl: scripts/route-teacher.sh describe-intent first" >&2
    exit 1
  }
  step requests "$D" --seed "$seed" route requests "${src[@]}" --out "$build"
}

embed() {
  UV_PROJECT_ENVIRONMENT=$data/envs/route UV_PYTHON_PREFERENCE=only-managed \
    nice -n 19 uv sync --frozen --no-dev --project scripts/route/env -q
  echo "== embed"
  nice -n 19 "$data/envs/route/bin/python" -I scripts/route-embed.py --build "$build"
}

split() { step split "$D" --seed "$seed" route split --out "$build"; }

slates() {
  local p=()
  [ "${PROVISIONAL:-0}" = 1 ] && p=(--provisional)
  step slates "$D" --seed "$seed" route slates --out "$build" ${p[@]+"${p[@]}"}
}

# The head window of section 3, for each size with its own config and tokenizer: the share of
# training and validation records whose head (question and options) a window cuts.
truncation() {
  [ -x "$train" ] || { echo "no sky-train at $train: set SKY_TRAIN (s2-truncation reads records only)" >&2; exit 1; }
  local size w cfg tok records=$build/records/tools,$build/records/intents,$build/records/when2call
  mkdir -p "$build/truncation"
  for size in nano micro; do
    tok=data/tok/tok-s6-2048.json
    [ "$size" = micro ] && tok=data/tok/tok-s6-4096.json
    for w in 160 256 320; do
      cfg=$build/truncation/s6-$size-$w.json
      python3 -I -c 'import json, sys; c = json.load(open(sys.argv[1])); c["max_head_tokens"] = int(sys.argv[3]); json.dump(c, open(sys.argv[2], "w"), indent=2)' \
        "runs/recipe/configs/s6-$size.json" "$cfg" "$w"
      step "truncation-$size-$w" "$train" s2-truncation --config "$cfg" --tokenizer "$tok" --records "$records" \
        --per-source 1000000000 --json "$build/truncation/$size-$w.json"
    done
  done
}

manifest() {
  local t=() size w
  for size in nano micro; do
    for w in 160 256 320; do
      [ -f "$build/truncation/$size-$w.json" ] && t+=("$size:$w=$build/truncation/$size-$w.json")
    done
  done
  local list
  list=$(IFS=,; echo "${t[*]:-}")
  step manifest "$D" --seed "$seed" route manifest --data "$data" --out "$build" ${list:+--truncation "$list"}
  mkdir -p runs/s6-nano2
  python3 -I - "$build/BUILD.json" runs/s6-nano2/test-files.json <<'EOF'
import json, sys
m = json.load(open(sys.argv[1]))
key = next(k for k in m if k.startswith("test files"))
files = {k.split("/", 1)[1]: v["sha256"] for k, v in m[key].items()}
json.dump({"about": "s6-route-nano: the test files as the build wrote them (docs/s6-nano2-prereg.md 4.6, item 1); "
           "scripts/route/common.py refuses a test file whose sha256 differs", "build": sys.argv[1], "files": files},
          open(sys.argv[2], "w"), indent=1)
print(f"{len(files)} test files recorded in {sys.argv[2]}; provisional build: {m['provisional']}")
EOF
}

for s in "${steps[@]}"; do
  case "$s" in
    fetch | intents | requests | embed | split | slates | truncation | manifest) "$s" ;;
    *) echo "unknown step $s (fetch intents requests embed split slates truncation manifest)" >&2; exit 2 ;;
  esac
done
echo "== done: $build"
