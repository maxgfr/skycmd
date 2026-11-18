#!/usr/bin/env bash
# Build the data of s6-route v4 (docs/s6-route4-prereg.md, sections 3.4, 4 and 6). No training,
# no GPU, no evaluation of a fresh file: CPU only, at nice -n 19. v2's build folder is read, never
# written; v4's build and the Toucan folder are outside every repository. This script never calls
# the teacher: it prints the command the lead runs, and nothing it hands the teacher holds a
# request (the job items are tools; `sky-data teacher` refuses a test or fresh item).
#
# Usage: scripts/build-route4-data.sh [stage ...]   (default: every stage below, in order)
#   fetch         Toucan-1.5M's SFT parquet files at the commit of scripts/route/pins.json
#                 (v4_sources), each checked against the hash its host publishes (sky-data
#                 fetch-model); <data>/MANIFEST4.json records URL, commit, bytes, sha256 and
#                 licence of each file; the parquet bytes must be the pinned 425,496,735
#   teacher-jobs  sky-data route v4-plan: v2's split recomputed and checked, the plan of v4
#                 (D3, D5, D7), v4_tools and v4_tools_masked (seed 1237), and the items of the one
#                 teacher job (rewrite-description: the tools without a rewrite). Then the lead
#                 runs: BUILD=$B4 caffeinate -i scripts/route-teacher.sh rewrite-description loop
#   latency       the check of 3.4 (validation only, no training running): v2's published file
#                 with its head window set to 320 (gguf-set copy), the first 200 slates of
#                 v4_tools rendered by D4, every request meeting its tool list for the first time,
#                 WASM SIMD128 (crates/sky-web/bench/route3-node.mjs --uncached; the module of
#                 scripts/build-wasm.sh, built from this tree); median <= 50,000 us: option (i)
#   build         sky-data route v4-slates: the records (D1, D2, D4, D5), the probe copies of
#                 v4_tools and the model's copies (refused until the teacher has answered and
#                 the latency check took option (i))
#   checks        scripts/s6-route4-checks.py (C1 to C5) on the records and on v4_tools and its
#                 masked copy; writes checks.passed only when every judged check holds; a failing
#                 rule stops here (REMEDY=surface-cell runs teacher-jobs, build and checks once
#                 more with the one remedy of 4.4, only when the checks say it is open)
#   fresh         Toucan's rows (sky-data route toucan-rows: stopped, see its message), the tool
#                 table, the MiniLM embedding (scripts/route-embed.py --build $T4), and the fresh
#                 test (sky-data --seed 1236 route toucan: t4_tools, its masked copy and servers,
#                 written once, FRESH4.json, runs/s6-route4/test-files.json), and the items of the
#                 probe copies' job (fresh-tools: a tool's name and description). Then the lead
#                 runs: BUILD=$T4 caffeinate -i scripts/route-teacher.sh rewrite-description loop
#   probes        sky-data --seed 1236 route toucan-probes: t4_tools_rw and t4_tools_masked_rw
#   counts        the request-blind rules of 4.4 C2 counted on the fresh files (option texts and
#                 the target's position only; reported, never judged) and the counts the lead
#                 writes into the amendment of 6.3 before any scoring; stops when fewer than 600
#                 slates remain (6.3: v4 is not trained)
#
# Each stage refuses to run when the one before it has not written its marker
# ($B4/markers/<stage>.done; checks: checks.passed). DRY_RUN=1 prints every command and runs
# nothing (the markers are not required).
#
#   ROUTE_DATA=../skycmd-data/s6-route   V2=$ROUTE_DATA/build (read only)
#   B4=$ROUTE_DATA/build4   T4=$ROUTE_DATA/toucan   RUNS=runs/s6-route4
#   MODEL=site/public/models/s6-route-nano.gguf  EMBED=site/public/models/s6-route-embed.gguf
#   REMEDY=surface-cell (only as the checks allow it, once)
set -euo pipefail
cd "$(dirname "$0")/.."
export DEVELOPER_DIR=${DEVELOPER_DIR:-/Applications/Xcode.app/Contents/Developer}
export CARGO_TARGET_DIR=${CARGO_TARGET_DIR:-$PWD/target/s3-train}
dry=${DRY_RUN:-0}
data=${ROUTE_DATA:-../skycmd-data/s6-route}
v2=${V2:-$data/build}
b4=${B4:-$data/build4}
t4=${T4:-$data/toucan}
runs=${RUNS:-runs/s6-route4}
model=${MODEL:-site/public/models/s6-route-nano.gguf}
embed=${EMBED:-site/public/models/s6-route-embed.gguf}
remedy=${REMEDY:-}
D=$CARGO_TARGET_DIR/release/sky-data
G=$CARGO_TARGET_DIR/release/gguf-set
stages=(fetch teacher-jobs latency build checks fresh probes counts)

x() {
  if [ "$dry" = 1 ]; then
    printf '+'
    printf ' %q' "$@"
    printf '\n'
  else
    nice -n 19 "$@"
  fi
}
# logged <name> <command...>: what a step prints (counts only) goes to $b4/logs/<name>.json
logged() {
  local name=$1
  shift
  if [ "$dry" = 1 ]; then
    x "$@"
  else
    mkdir -p "$b4/logs"
    echo "== $name"
    nice -n 19 "$@" >"$b4/logs/$name.json"
  fi
}
note() { echo "# $*"; }
marker() { echo "$b4/markers/$1"; }
# need <stage>: the stage before has written its marker
need() {
  local m
  m=$(marker "$1")
  if [ "$dry" = 1 ]; then
    note "requires $m"
    return
  fi
  [ -f "$m" ] || { echo "no $m: run scripts/build-route4-data.sh $2 first" >&2; exit 1; }
}
done_() {
  if [ "$dry" = 1 ]; then
    note "writes $(marker "$1")"
  else
    mkdir -p "$b4/markers"
    date '+%F %T' >"$(marker "$1")"
  fi
}
no_training() {
  if [ "$dry" != 1 ] && pgrep -f 'sky-train' >/dev/null; then
    echo "a training runs (pgrep sky-train): the latency check of 3.4 runs with no training running" >&2
    exit 1
  fi
}

fetch() {
  local line name kind repo rev inc bytes
  line=$(python3 -I -c '
import json, sys
p = json.load(open(sys.argv[1]))
s = p["v4_sources"][0]
print(s["name"], s["kind"], s["repo"], s["rev"], ",".join(s["include"]), s["bytes"])' scripts/route/pins.json)
  read -r name kind repo rev inc bytes <<<"$line"
  [ "$kind" = dataset ] || { echo "pins.json: $name is not a dataset" >&2; exit 2; }
  x "$D" --cache "$data/hf-cache" fetch-model "$repo" --rev "$rev" --dataset --include "$inc" \
    --out "$data/src/$name" --max-gb 1
  note "record $data/MANIFEST4.json (URL, commit, bytes, sha256, licence) and check the parquet bytes against $bytes"
  if [ "$dry" != 1 ]; then
    python3 -I - "$data" scripts/route/pins.json <<'EOF'
import json, os, sys
data, pins = sys.argv[1:]
s = json.load(open(pins))["v4_sources"][0]
d = os.path.join(data, "src", s["name"])
f = json.load(open(os.path.join(d, ".skycmd-fetch.json")))
if f["commit"] != s["rev"]:
    sys.exit(f"fetched commit {f['commit']}, pinned {s['rev']}")
out = [{"source": s["name"], "url": f"https://huggingface.co/datasets/{s['repo']}", "commit": f["commit"], "file": x["path"],
        "bytes": x["size"], "sha256": x["sha256"], "git_sha1": x["git_sha1"], "licence": s["license"], "role": s["role"]}
       for x in f["files"]]
parquet = sum(x["bytes"] for x in out if x["file"].endswith(".parquet"))
if parquet != s["bytes"]:
    sys.exit(f"the parquet files hold {parquet} bytes, the pin says {s['bytes']}")
json.dump({"about": "s6-route v4: the downloads of scripts/route/pins.json v4_sources", "bytes": sum(x["bytes"] for x in out),
           "files": out}, open(os.path.join(data, "MANIFEST4.json"), "w"), indent=1)
print(f"{len(out)} files, {parquet} parquet bytes, as pinned")
EOF
  fi
  done_ fetch.done
}

teacher_jobs() {
  need fetch.done fetch
  local r=()
  if [ -n "$remedy" ]; then
    [ "$remedy" = surface-cell ] || { echo "REMEDY=$remedy: the one remedy of 4.4 is surface-cell" >&2; exit 2; }
    if [ "$dry" != 1 ]; then
      [ -f "$(marker remedy.open)" ] || { echo "the checks have not opened the remedy ($(marker remedy.open))" >&2; exit 1; }
      [ ! -f "$(marker remedy.used)" ] || { echo "the remedy was used once: 4.4 allows no second (v4 stops before training)" >&2; exit 1; }
    fi
    r=(--remedy surface-cell)
  fi
  logged v4-plan "$D" --seed 1234 route v4-plan --v2 "$v2" --out "$b4" ${r[@]+"${r[@]}"}
  if [ -n "$remedy" ]; then
    done_ remedy.used
  fi
  note "the guard of the job's items: tools only, no test or fresh split"
  if [ "$dry" != 1 ]; then
    python3 -I - "$b4/teacher/rewrite-description.items.jsonl" <<'EOF'
import json, sys
n = 0
for l in open(sys.argv[1]):
    x = json.loads(l)
    n += 1
    if set(x) != {"id", "split", "name", "description"} or x["split"] not in ("train", "val"):
        sys.exit(f"an item that is not a training or validation tool: {x.get('id')}")
print(f"{n} rewrite items ({(n + 19) // 20} calls of 20)")
EOF
  fi
  done_ teacher-jobs.done
  note "next, by the lead: BUILD=$b4 caffeinate -i scripts/route-teacher.sh rewrite-description loop"
}

latency() {
  need teacher-jobs.done teacher-jobs
  no_training
  local dir=$runs/latency
  x mkdir -p "$dir"
  x "$G" "$model" "$dir/v2-head320.gguf" skycmd-s2.max_head_tokens=u32:320
  note "the module: site/public/pkg from scripts/build-wasm.sh on this tree (ToolRouter.setRender, clearCaches)"
  if [ "$dry" = 1 ]; then
    x node crates/sky-web/bench/route3-node.mjs "$dir/v2-head320.gguf" "$embed" "$b4/val/v4_tools.jsonl" - \
      --n 200 --render scores-in-options --uncached
  else
    nice -n 19 node crates/sky-web/bench/route3-node.mjs "$dir/v2-head320.gguf" "$embed" "$b4/val/v4_tools.jsonl" - \
      --n 200 --render scores-in-options --uncached >"$dir/wasm-uncached.json"
  fi
  note "the rule of 3.4: median per request <= 50,000 us gives option (i); otherwise (ii'), which this tree does not implement"
  if [ "$dry" != 1 ]; then
    python3 -I - "$dir/wasm-uncached.json" "$runs/latency-check.json" <<'EOF'
import json, sys
m = json.load(open(sys.argv[1]))
if m["n"] != 200 or m["render"] != "scores-in-options" or m["mode"] != "uncached" or m["build"] != "simd128":
    sys.exit(f"not the measurement of 3.4: {m}")
option = "i" if m["median_us"] <= 50000 else "ii'"
out = {"about": "s6-route v4: the latency check of docs/s6-route4-prereg.md 3.4 (validation only)", "rule": "median per request <= 50000 us: option (i)",
       "median_us": m["median_us"], "model_median_us": m["model"]["median_us"], "option": option, "measurement": m}
json.dump(out, open(sys.argv[2], "w"), indent=1)
print(json.dumps({"median_us": m["median_us"], "option": option}))
if option != "i":
    sys.exit("option (ii') is taken by 3.4: its rendering (engine labels t1 ... t7) and its tokenizer check are not written in this tree; v4 waits for them")
EOF
  fi
  done_ latency.done
}

build() {
  need latency.done latency
  logged v4-slates "$D" --seed 1234 route v4-slates --v2 "$v2" --out "$b4"
  done_ build.done
}

checks() {
  need build.done build
  x mkdir -p "$runs"
  local st=0
  if [ "$dry" = 1 ]; then
    x python3 -I scripts/s6-route4-checks.py --build "$b4" --requests "$v2/stage/requests.jsonl" \
      --val "$b4/val/v4_tools.jsonl" "$b4/val/v4_tools_masked.jsonl" --out "$runs/checks.json" --md "$runs/checks.md"
  else
    nice -n 19 python3 -I scripts/s6-route4-checks.py --build "$b4" --requests "$v2/stage/requests.jsonl" \
      --val "$b4/val/v4_tools.jsonl" "$b4/val/v4_tools_masked.jsonl" --out "$runs/checks.json" --md "$runs/checks.md" || st=$?
  fi
  case "$st" in
    0) done_ checks.passed ;;
    1) echo "C1 or C3 fails: the build has a defect, fixed before anything else ($runs/checks.md)" >&2; exit 1 ;;
    2)
      if python3 -I -c 'import json,sys; sys.exit(0 if json.load(open(sys.argv[1]))["verdict"]["the remedy of 4.4 is open"] else 1)' "$runs/checks.json" &&
        [ ! -f "$(marker remedy.used)" ]; then
        done_ remedy.open
        echo "a surface rule of C2 fails: the one remedy of 4.4 is open, once:" >&2
        echo "  REMEDY=surface-cell scripts/build-route4-data.sh teacher-jobs latency build checks" >&2
      else
        echo "a rule of C2 fails and no remedy is left: v4 stops before training, the table is published ($runs/checks.md)" >&2
      fi
      exit 2
      ;;
    *) exit "$st" ;;
  esac
}

fresh() {
  need checks.passed checks
  local parquet
  parquet=$(python3 -I -c '
import json, sys
s = json.load(open(sys.argv[1]))["v4_sources"][0]
print(",".join(sys.argv[2] + "/src/" + s["name"] + "/" + f for f in s["include"] if f.endswith(".parquet")))' scripts/route/pins.json "$data")
  logged toucan-rows "$D" route toucan-rows --input "$parquet" --out "$t4"
  logged toucan-stage "$D" route toucan-stage --t4 "$t4"
  x env UV_PROJECT_ENVIRONMENT="$data/envs/route" UV_PYTHON_PREFERENCE=only-managed \
    uv sync --frozen --no-dev --project scripts/route/env -q
  x "$data/envs/route/bin/python" -I scripts/route-embed.py --build "$t4"
  logged toucan "$D" --seed 1236 route toucan --v2 "$v2" --t4 "$t4" --record "$runs"
  done_ fresh.done
  note "next, by the lead: BUILD=$t4 caffeinate -i scripts/route-teacher.sh rewrite-description loop (fresh-tools: tools only)"
}

probes() {
  need fresh.done fresh
  logged toucan-probes "$D" --seed 1236 route toucan-probes --t4 "$t4" --record "$runs"
  done_ probes.done
}

counts() {
  need probes.done probes
  local f=$t4/fresh
  x python3 -I scripts/s6-route4-checks.py --fresh "$f/t4_tools.jsonl" "$f/t4_tools_masked.jsonl" "$f/t4_tools_rw.jsonl" \
    "$f/t4_tools_masked_rw.jsonl" --fresh-manifest "$f/FRESH4.json" --out "$runs/t4-checks.json" --md "$runs/t4-checks.md"
  note "the counts of 6.3 for the amendment: $runs/counts.json, $runs/test-files.json, $runs/t4-checks.md"
  if [ "$dry" != 1 ]; then
    python3 -I - "$runs/counts.json" <<'EOF'
import json, sys
c = json.load(open(sys.argv[1]))
keys = ["rows read by subset", "rows dropped, by filter", "slates", "servers", "gold clusters",
        "share of the slates on the three largest servers", "none share", "slates by subset"]
print(json.dumps({k: c.get(k) for k in keys}, indent=1))
if c["slates"] < 600:
    sys.exit("fewer than 600 slates (6.3): v4 is not trained; the amendment says so")
EOF
  fi
  done_ counts.done
}

run=("$@")
[ ${#run[@]} -eq 0 ] && run=("${stages[@]}")
for s in "${run[@]}"; do
  case " ${stages[*]} " in
    *" $s "*) ;;
    *) echo "unknown stage $s (${stages[*]})" >&2; exit 2 ;;
  esac
done
if [ "$dry" = 1 ]; then
  note "DRY_RUN: nothing is built or run"
else
  nice -n 19 cargo build -q --release -p sky-data -p sky-gguf
fi
for s in "${run[@]}"; do
  echo "== stage $s"
  case "$s" in
    teacher-jobs) teacher_jobs ;;
    *) "$s" ;;
  esac
done
echo "== done: $b4"
