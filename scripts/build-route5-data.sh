#!/usr/bin/env bash
# Build the data of s6-route v5 (docs/s6-route5-prereg.md, sections 4 and 6, with amendment 1). No
# training, no GPU, no evaluation of a fresh file: CPU only, at nice -n 19. v2's build folder is
# read, never written; v4's build folder is read for its rewrites (build4/teacher/) and its
# validation slates; v5's build and the Toucan folder are outside every repository. This script
# never calls the teacher: it prints the command the lead runs; the false-negative items hold
# training and validation requests only, and `sky-data teacher` refuses a test or fresh item.
#
# Usage: scripts/build-route5-data.sh [stage ...]   (default: every stage below, in order)
#   embed-requests  scripts/route-embed.py --requests: the training and validation requests of v2's
#                   plan embedded by potion-base-2M (<b5>/emb/potion-requests.npy, 4.1); a test or
#                   fresh request is refused before any model call
#   plan            sky-data route v5-plan: v2's split recomputed and checked; the plan of v5 (every
#                   drawn distractor within the right tool's surface cell, the hard slates the 16
#                   candidates nearest to the request), v5_tools and v5_tools_masked (seed 1238),
#                   the counts by group, kind, split and level, the false-negative items (staged)
#   c6              scripts/s6-route5-checks.py --plan: C6 at plan time (amendment 1); above 0.10 at
#                   L5 or L6 in a judged cell, v5 stops here, before any teacher call
#   teacher-jobs    the false-negative items moved to <b5>/teacher/ after their guard (training and
#                   validation requests only, v2's item form); refused without c6's marker. Then the
#                   lead runs: BUILD=$B5 caffeinate -i scripts/route-teacher.sh false-negatives loop
#   build           the rebuildable outputs of v4's build removed first (4.7: build4/records,
#                   build4/stage, the model and probe copies of build4/val; never build4/teacher,
#                   nor build4/val/v4_tools.jsonl and its masked copy, which 4.5 and 5 read), listed
#                   with their sizes in the log; then sky-data route v5-slates: the false-negative
#                   filter, the records, the probe copies and the model's copies (refused until the
#                   teacher has answered every item)
#   checks          scripts/s6-route5-checks.py: C6 confirmed, C1 to C5, C2 and N1 to N5 judged on the
#                   records and on v5_tools and its masked copy; writes checks.passed only when every
#                   judged check holds; a failing rule stops v5 before training (no remedy)
#   baselines-val   4.5: BM25 and potion-cos by the frozen baseline code (v2's fitted values) on
#                   v_tools, v4_tools and v5_tools and their masked copies (reported, for the
#                   amendment that writes the pilot's bar); runs/s6-route5/baselines-val.json
#   fresh           Toucan's three parquet files hash-checked against <data>/MANIFEST4.json, then
#                   read (sky-data route toucan-rows), the tool table, the MiniLM embedding and the
#                   fresh test by v5's cell (sky-data --seed 1236 route toucan --rule v5: t5_tools,
#                   its masked copy and servers, written once, FRESH5.json,
#                   runs/s6-route5/test-files.json), and the items of the probe copies' job. Then the
#                   lead runs: BUILD=$T5 caffeinate -i scripts/route-teacher.sh rewrite-description loop
#   probes          sky-data --seed 1236 route toucan-probes --rule v5: t5_tools_rw, t5_tools_masked_rw
#   counts          the rules of 4.4 (C2 and N1 to N5, whole slate and restrictions), the slates by
#                   level and C6's share counted on the fresh files (reported, never judged), and the
#                   counts the lead writes into the amendment of 6.3 before any scoring; stops when
#                   fewer than 600 slates remain (v5 is not trained)
#
# Each stage refuses to run when the one before it has not written its marker
# ($B5/markers/<stage>.done; c6: c6.passed; checks: checks.passed). Before every stage the volume
# of the data folder must hold MIN_FREE_GB (3, 4.7) free, else the stage stops (a technical stop,
# resumed once the space is there). DRY_RUN=1 prints every command and runs nothing (the markers
# and the disk are not required).
#
#   ROUTE_DATA=../skycmd-data/s6-route   V2=$ROUTE_DATA/build (read only)   B4=$ROUTE_DATA/build4
#   B5=$ROUTE_DATA/build5   T5=$ROUTE_DATA/toucan5   RUNS=runs/s6-route5   MIN_FREE_GB=3
set -euo pipefail
cd "$(dirname "$0")/.."
export DEVELOPER_DIR=${DEVELOPER_DIR:-/Applications/Xcode.app/Contents/Developer}
export CARGO_TARGET_DIR=${CARGO_TARGET_DIR:-$PWD/target/s3-train}
dry=${DRY_RUN:-0}
data=${ROUTE_DATA:-../skycmd-data/s6-route}
v2=${V2:-$data/build}
b4=${B4:-$data/build4}
b5=${B5:-$data/build5}
t5=${T5:-$data/toucan5}
runs=${RUNS:-runs/s6-route5}
min_free=${MIN_FREE_GB:-3}
export ROUTE_DATA=$data
D=$CARGO_TARGET_DIR/release/sky-data
PY=$data/envs/route/bin/python
stages=(embed-requests plan c6 teacher-jobs build checks baselines-val fresh probes counts)

x() {
  if [ "$dry" = 1 ]; then
    printf '+'
    printf ' %q' "$@"
    printf '\n'
  else
    nice -n 19 "$@"
  fi
}
# logged <name> <command...>: what a step prints (counts only) goes to $b5/logs/<name>.json
logged() {
  local name=$1
  shift
  if [ "$dry" = 1 ]; then
    x "$@"
  else
    mkdir -p "$b5/logs"
    echo "== $name"
    nice -n 19 "$@" >"$b5/logs/$name.json"
  fi
}
note() { echo "# $*"; }
marker() { echo "$b5/markers/$1"; }
# need <marker> <stage>: the stage before has written its marker
need() {
  local m
  m=$(marker "$1")
  if [ "$dry" = 1 ]; then
    note "requires $m"
    return
  fi
  [ -f "$m" ] || { echo "no $m: run scripts/build-route5-data.sh $2 first" >&2; exit 1; }
}
done_() {
  if [ "$dry" = 1 ]; then
    note "writes $(marker "$1")"
  else
    mkdir -p "$b5/markers"
    date '+%F %T' >"$(marker "$1")"
  fi
}
# 4.7: at least MIN_FREE_GB free on the volume of the data folder before a stage writes
disk_floor() {
  if [ "$dry" = 1 ]; then
    note "requires $min_free GB free on the volume of $data (4.7)"
    return
  fi
  local kb
  kb=$(df -Pk "$data" | awk 'NR == 2 { print $4 }')
  if [ "$kb" -lt $((min_free * 1024 * 1024)) ]; then
    echo "$((kb / 1024)) MB free on the volume of $data, below the floor of $min_free GB (docs/s6-route5-prereg.md 4.7): a technical stop, resumed once the space is there" >&2
    exit 4
  fi
  note "$((kb / 1024 / 1024)) GB free on the volume of $data (floor $min_free GB)"
}

embed_requests() {
  [ -f "$v2/stage/plan.jsonl" ] || [ "$dry" = 1 ] || { echo "no $v2/stage/plan.jsonl" >&2; exit 1; }
  x env UV_PROJECT_ENVIRONMENT="$data/envs/route" UV_PYTHON_PREFERENCE=only-managed \
    uv sync --frozen --no-dev --project scripts/route/env -q
  x "$PY" -I scripts/route-embed.py --requests --v2 "$v2" --build "$b5"
  done_ embed-requests.done
}

plan() {
  need embed-requests.done embed-requests
  logged v5-plan "$D" --seed 1234 route v5-plan --v2 "$v2" --v4 "$b4" --out "$b5"
  done_ plan.done
}

c6() {
  need plan.done plan
  x mkdir -p "$runs"
  local st=0
  if [ "$dry" = 1 ]; then
    x python3 -I scripts/s6-route5-checks.py --plan "$b5/counts/v5-plan.json" --out "$runs/c6-plan.json" --md "$runs/c6-plan.md"
  else
    nice -n 19 python3 -I scripts/s6-route5-checks.py --plan "$b5/counts/v5-plan.json" --out "$runs/c6-plan.json" \
      --md "$runs/c6-plan.md" || st=$?
  fi
  case "$st" in
    0) done_ c6.passed ;;
    2) echo "C6 fails at plan time: v5 stops before any teacher call; the plan's counts by level are published ($runs/c6-plan.md)" >&2; exit 2 ;;
    *) exit "$st" ;;
  esac
}

teacher_jobs() {
  need c6.passed c6
  note "the guard of the job's items: training and validation requests, v2's item form; then moved to $b5/teacher/"
  if [ "$dry" != 1 ]; then
    python3 -I - "$b5/stage/false-negatives5.items.jsonl" "$b5/counts/v5-plan.json" <<'EOF'
import json, sys
items, plan = sys.argv[1:]
n, ids = 0, set()
for l in open(items):
    x = json.loads(l)
    n += 1
    if set(x) != {"id", "split", "request", "options"} or x["split"] not in ("train", "val"):
        sys.exit(f"an item that is not a training or validation request with its hard slate: {x.get('id')}")
    if x["id"] in ids:
        sys.exit(f"the item {x['id']} repeats")
    ids.add(x["id"])
want = json.load(open(plan))["teacher items"]["false-negatives"]
if n != want:
    sys.exit(f"{n} items, the plan counted {want}")
print(f"{n} false-negative items ({(n + 19) // 20} calls of 20)")
EOF
  fi
  x mkdir -p "$b5/teacher"
  x mv "$b5/stage/false-negatives5.items.jsonl" "$b5/teacher/false-negatives.items.jsonl"
  done_ teacher-jobs.done
  note "next, by the lead: BUILD=$b5 caffeinate -i scripts/route-teacher.sh false-negatives loop"
}

# 4.7: v4's rebuildable outputs, listed with their sizes, then removed (never build4/teacher/)
remove_v4_outputs() {
  local p
  local list=("$b4/records" "$b4/stage")
  if [ -d "$b4/val" ] || [ "$dry" = 1 ]; then
    for p in "$b4"/val/*.model.jsonl "$b4"/val/v4_tools_rw.jsonl "$b4"/val/v4_tools_masked_rw.jsonl; do
      [ -e "$p" ] || [ "$dry" = 1 ] && list+=("$p")
    done
  fi
  [ "$dry" = 1 ] || mkdir -p "$b5/logs"
  for p in "${list[@]}"; do
    case "$p" in
      "$b4"/teacher | "$b4"/teacher/* | "$b4"/val/v4_tools.jsonl | "$b4"/val/v4_tools_masked.jsonl)
        echo "refusing to remove $p" >&2
        exit 2
        ;;
    esac
    if [ "$dry" = 1 ]; then
      x rm -rf "$p"
    elif [ -e "$p" ]; then
      du -sk "$p" | awk -v p="$p" '{ printf "removed %s (%d KB)\n", p, $1 }' | tee -a "$b5/logs/v4-removed.txt"
      rm -rf "$p"
    fi
  done
}

build() {
  need teacher-jobs.done teacher-jobs
  note "4.7: the rebuildable outputs of v4's build are removed first, listed in $b5/logs/v4-removed.txt"
  remove_v4_outputs
  logged v5-slates "$D" --seed 1234 route v5-slates --v2 "$v2" --v4 "$b4" --out "$b5"
  done_ build.done
}

checks() {
  need build.done build
  x mkdir -p "$runs"
  local st=0
  local args=(--build "$b5" --requests "$v2/stage/requests.jsonl" --tools "$v2/stage/tools.jsonl"
    --pools "$b5/stage/pools5.json" --plan-counts "$b5/counts/v5-plan.json"
    --val "$b5/val/v5_tools.jsonl" "$b5/val/v5_tools_masked.jsonl" --out "$runs/checks.json" --md "$runs/checks.md")
  if [ "$dry" = 1 ]; then
    x python3 -I scripts/s6-route5-checks.py "${args[@]}"
  else
    nice -n 19 python3 -I scripts/s6-route5-checks.py "${args[@]}" || st=$?
  fi
  case "$st" in
    0) done_ checks.passed ;;
    1) echo "C1 or C3 fails: the build has a defect, fixed before anything else ($runs/checks.md)" >&2; exit 1 ;;
    2) echo "C6, a rule of C2 or one of N1 to N5 fails: v5 stops before training, the table is published; no remedy ($runs/checks.md)" >&2; exit 2 ;;
    *) exit "$st" ;;
  esac
}

baselines_val() {
  need checks.passed checks
  local out=$data/eval/route5-val sys split files
  for sys in potion-cos bm25; do
    for split in val v4val v5val; do
      case "$split" in
        val) files=v_tools,v_tools_masked ;;
        v4val) files=v4_tools,v4_tools_masked ;;
        v5val) files=v5_tools,v5_tools_masked ;;
      esac
      x env BUILD="$v2" BUILD4="$b4" BUILD5="$b5" "$PY" -I scripts/route-baselines.py score --system "$sys" \
        --split "$split" --files "$files" --out-root "$out/$split"
    done
  done
  note "the accuracies (dump lines whose argmax is the gold), reported for the amendment of 4.5: $runs/baselines-val.json"
  if [ "$dry" != 1 ]; then
    mkdir -p "$runs"
    python3 -I - "$out" "$runs/baselines-val.json" <<'EOF'
import json, os, sys
root, out = sys.argv[1:]
res = {}
for split in ("val", "v4val", "v5val"):
    for sysname in ("potion-cos", "bm25"):
        d = os.path.join(root, split, sysname)
        for f in sorted(os.listdir(d)) if os.path.isdir(d) else []:
            if not f.endswith(".jsonl"):
                continue
            rows = [json.loads(l) for l in open(os.path.join(d, f)) if l.strip()]
            right = sum(max(range(len(r["probs"])), key=lambda i: r["probs"][i]) == r["gold"] for r in rows)
            res.setdefault(sysname, {})[f[:-len(".jsonl")]] = {"slates": len(rows), "accuracy": right / max(len(rows), 1)}
json.dump({"about": "s6-route v5 (docs/s6-route5-prereg.md 4.5): BM25 and potion-cos by the frozen baseline code, v2's fitted values; reported, not judged",
           "accuracy": res}, open(out, "w"), indent=1)
print(json.dumps(res))
EOF
  fi
  done_ baselines-val.done
}

fresh() {
  need baselines-val.done baselines-val
  local parquet
  parquet=$(python3 -I -c '
import json, sys
s = json.load(open(sys.argv[1]))["v4_sources"][0]
print(",".join(sys.argv[2] + "/src/" + s["name"] + "/" + f for f in s["include"] if f.endswith(".parquet")))' scripts/route/pins.json "$data")
  note "6: the parquet files hash-checked against $data/MANIFEST4.json before they are read"
  logged toucan-rows "$D" route toucan-rows --input "$parquet" --out "$t5" --manifest "$data/MANIFEST4.json"
  logged toucan-stage "$D" route toucan-stage --t4 "$t5"
  x env UV_PROJECT_ENVIRONMENT="$data/envs/route" UV_PYTHON_PREFERENCE=only-managed \
    uv sync --frozen --no-dev --project scripts/route/env -q
  x "$PY" -I scripts/route-embed.py --build "$t5"
  logged toucan "$D" --seed 1236 route toucan --rule v5 --v2 "$v2" --t4 "$t5" --record "$runs"
  done_ fresh.done
  note "next, by the lead: BUILD=$t5 caffeinate -i scripts/route-teacher.sh rewrite-description loop (fresh-tools: tools only)"
}

probes() {
  need fresh.done fresh
  logged toucan-probes "$D" --seed 1236 route toucan-probes --rule v5 --t4 "$t5" --record "$runs"
  done_ probes.done
}

counts() {
  need probes.done probes
  local f=$t5/fresh
  x python3 -I scripts/s6-route5-checks.py --requests "$v2/stage/requests.jsonl" --tools "$v2/stage/tools.jsonl" \
    --pools "$b5/stage/pools5.json" --fresh "$f/t5_tools.jsonl" "$f/t5_tools_masked.jsonl" "$f/t5_tools_rw.jsonl" \
    "$f/t5_tools_masked_rw.jsonl" --fresh-manifest "$f/FRESH5.json" --out "$runs/t5-checks.json" --md "$runs/t5-checks.md"
  note "the counts of 6.3 for the amendment: $runs/counts.json, $runs/test-files.json, $runs/t5-checks.md"
  if [ "$dry" != 1 ]; then
    python3 -I - "$runs/counts.json" <<'EOF'
import json, sys
c = json.load(open(sys.argv[1]))
keys = ["rows read by subset", "rows dropped, by filter", "slates", "servers", "gold clusters",
        "share of the slates on the three largest servers", "none share", "slates by subset",
        "slates by level (6.2 item 5, reported)", "C6 (reported, not judged: 6.2)"]
print(json.dumps({k: c.get(k) for k in keys}, indent=1))
if c["slates"] < 600:
    sys.exit("fewer than 600 slates (6.3): v5 is not trained; the amendment says so")
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
  disk_floor
  nice -n 19 cargo build -q --release -p sky-data
fi
for s in "${run[@]}"; do
  echo "== stage $s"
  disk_floor
  case "$s" in
    embed-requests) embed_requests ;;
    teacher-jobs) teacher_jobs ;;
    baselines-val) baselines_val ;;
    *) "$s" ;;
  esac
done
echo "== done: $b5"
