#!/usr/bin/env bash
# The data stages of s6-route v6 (docs/s6-route6-prereg.md 4, 6 and 14, with amendment 1). No
# training, no GPU, no system scored on a fresh file: CPU only, at nice -n 19. v2's, v4's and v5's
# build folders are read, never written: v6 writes a full copy of v5's build under build6/ and the
# fix's plan and teacher files under build6/fix/. This script never calls the teacher: it prints
# the command the lead runs; the fix's items hold training and validation requests only.
#
# Usage: scripts/build-route6-data.sh [stage ...]   (default: every stage below, in order)
#   inputs            v5's build (build5/) against runs/s6-route6/inputs.json, the hashes of 4.1
#   fix-plan          the full copy build5/ -> build6/ (APFS clones where the volume has them),
#                     each file checked by sha256 (runs/s6-route6/copy.json); then sky-data route
#                     v5-plan --when2call-exempt none into build6/fix/ (amendment 1), and the
#                     check that every plan outside the When2Call group and the fix's v5_tools
#                     files are v5's
#   fix-teacher-jobs  sky-data route v5-fix-items: the items of the When2Call hard slates without a
#                     tool gold and without an answer by id (at most 1,043), with their guard.
#                     Then the lead runs: BUILD=$B6/fix caffeinate -i scripts/route-teacher.sh
#                     false-negatives loop
#   fix-build         sky-data route v5-slates --only-group when2call --fix build6/fix: the
#                     group's 14,732 records rebuilt by uid, every other record file rendered and
#                     checked against the copy's sha256; then every file of build6/ against
#                     build5/ (all equal but the two When2Call record files and
#                     counts/v5-slates.json), written to runs/s6-route6/rebuild.json, and the
#                     amendment's table printed (file | bytes | sha256)
#   fix-checks        scripts/s6-route6-checks.py checks: v5's checks reported; the stop rule of
#                     4.8 (N6 or its inverse more than 0.10 above the none share in a judged
#                     family) exits 2 and stops v6
#   spread            the spread table and the levels of the When2Call group, after the fix and in
#                     v5's build (runs/s6-route6/spread.md)
#   inputs-ok         after the lead's amendment of 4.8: build6/ against runs/s6-route6/inputs6.json
#                     (the hashes that amendment lists); writes runs/s6-route6/inputs.ok
#   placeholder       sky-data route blind --check-placeholder on every training and validation
#                     record and slate file: runs/s6-route6/placeholder.json (? then … then -; it
#                     must be the one the published-files diagnostic took, if it ran first)
#   support           scripts/s6-route6-blind.py support (4.3): runs/s6-route6/support.json
#   checks-read       scripts/s6-route5-checks.py on the other files v6 reads (4.4):
#                     runs/s6-route6/checks-read-files.md
#   blind-val         sky-data route blind on v5_tools and v5_tools_masked (B0, B1)
#   baselines-val     4.5: BM25 and potion-cos (v2's fitted values) on v_tools, v4_tools, v5_tools
#                     and their masked copies, and on the blind copies of v5_tools and its masked
#                     copy (BM25 at c exactly); runs/s6-route6/baselines-val.json
#   fresh             as v5's: Toucan's parquet files hash-checked against MANIFEST4.json, read, the
#                     tool table, the MiniLM embedding, and T5 (sky-data --seed 1236 route toucan
#                     --rule v5, --record runs/s6-route6)
#   fresh-tools       the probe copies' teacher job: refused until the lead has run BUILD=$T5
#                     scripts/route-teacher.sh rewrite-description loop
#   probes            sky-data --seed 1236 route toucan-probes --rule v5 --record runs/s6-route6
#   counts            the counts of the amendment of 6.3: the rules of 4.4 on T5 (reported), T5's
#                     chance c, the placeholder checked against T5's requests, the server-mates of
#                     each tool-right slate; stops when fewer than 600 slates remain
#
# Each stage refuses to run when the one before it has not written its marker
# ($B6/markers/<stage>.done). Before every stage the volume of the data folder must hold
# MIN_FREE_GB (3, 4.7) free, else the stage stops (exit 4). DRY_RUN=1 prints every command and
# writes nothing.
#
#   ROUTE_DATA=../skycmd-data/s6-route   V2=$ROUTE_DATA/build   B4=$ROUTE_DATA/build4
#   B5=$ROUTE_DATA/build5   B6=$ROUTE_DATA/build6   T5=$ROUTE_DATA/toucan5   RUNS=runs/s6-route6
#   MIN_FREE_GB=3
# For a later version over this one (scripts/build-route7-data.sh): FIX (the fix's folder,
# default $B6/fix), MARKERS and LOGS (default $B6/markers and $B6/logs), PLAN_FLAGS (the fix's
# plan switches, default --when2call-exempt none), ALSO (more folders whose teacher answers apply
# by id), CHECK_AGAINST (a previous fix's folder: the plans and records that may change are
# checked against it), BEFORE_REPORT and BEFORE_LABEL (the report and the build set beside the
# checks and the spread table; default v5's), EVAL_OUT (the baselines' folder, default
# $ROUTE_DATA/eval/route6-val).
set -euo pipefail
cd "$(dirname "$0")/.."
export DEVELOPER_DIR=${DEVELOPER_DIR:-/Applications/Xcode.app/Contents/Developer}
export CARGO_TARGET_DIR=${CARGO_TARGET_DIR:-$PWD/target/s3-train}
dry=${DRY_RUN:-0}
data=${ROUTE_DATA:-../skycmd-data/s6-route}
v2=${V2:-$data/build}
b4=${B4:-$data/build4}
b5=${B5:-$data/build5}
b6=${B6:-$data/build6}
t5=${T5:-$data/toucan5}
runs=${RUNS:-runs/s6-route6}
min_free=${MIN_FREE_GB:-3}
export ROUTE_DATA=$data
D=$CARGO_TARGET_DIR/release/sky-data
PY=$data/envs/route/bin/python
fix=${FIX:-$b6/fix}
markers=${MARKERS:-$b6/markers}
logs=${LOGS:-$b6/logs}
read -r -a plan_flags <<<"${PLAN_FLAGS:---when2call-exempt none}"
also=()
[ -n "${ALSO:-}" ] && also=(--also "$ALSO")
check_against=${CHECK_AGAINST:-}
before_report=${BEFORE_REPORT:-runs/s6-route5/checks.json}
before_label=${BEFORE_LABEL:-v5\'s build}
stages=(inputs fix-plan fix-teacher-jobs fix-build fix-checks spread inputs-ok placeholder support checks-read
  blind-val baselines-val fresh fresh-tools probes counts)

x() {
  if [ "$dry" = 1 ]; then
    printf '+'
    printf ' %q' "$@"
    printf '\n'
  else
    nice -n 19 "$@"
  fi
}
logged() {
  local name=$1
  shift
  if [ "$dry" = 1 ]; then
    x "$@"
  else
    mkdir -p "$logs"
    echo "== $name"
    nice -n 19 "$@" >"$logs/$name.json"
  fi
}
note() { echo "# $*"; }
marker() { echo "$markers/$1"; }
need() {
  local m
  m=$(marker "$1")
  if [ "$dry" = 1 ]; then
    note "requires $m"
    return
  fi
  [ -f "$m" ] || { echo "no $m: run scripts/build-route6-data.sh $2 first" >&2; exit 1; }
}
done_() {
  if [ "$dry" = 1 ]; then
    note "writes $(marker "$1")"
  else
    mkdir -p "$markers"
    date '+%F %T' >"$(marker "$1")"
  fi
}
disk_floor() {
  if [ "$dry" = 1 ]; then
    note "requires $min_free GB free on the volume of $data (4.7)"
    return
  fi
  local kb
  kb=$(df -Pk "$data" | awk 'NR == 2 { print $4 }')
  if [ "$kb" -lt $((min_free * 1024 * 1024)) ]; then
    echo "$((kb / 1024)) MB free on the volume of $data, below the floor of $min_free GB (docs/s6-route6-prereg.md 4.7): a technical stop, resumed once the space is there" >&2
    exit 4
  fi
  note "$((kb / 1024 / 1024)) GB free on the volume of $data (floor $min_free GB)"
}
# hashes <want.json> <dir> <out.json> [<files that may differ>...]: every file of want against dir
hashes() {
  python3 -I - "$@" <<'EOF'
import hashlib, json, os, sys
want_path, root, out = sys.argv[1:4]
may = set(sys.argv[4:])
want = json.load(open(want_path))["files"]
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()
bad, rows = [], {}
for f, w in sorted(want.items()):
    p = os.path.join(root, f)
    got = {"bytes": os.path.getsize(p), "sha256": sha(p)} if os.path.exists(p) else None
    rows[f] = got
    if got != w and f not in may:
        bad.append(f)
json.dump({"against": want_path, "root": root, "files": rows, "differ": bad}, open(out, "w"), indent=1)
if bad:
    sys.exit("these files differ from %s: %s (a technical failure, fixed before anything else)" % (want_path, ", ".join(bad)))
print("%d files as %s%s" % (len(want) - len(may & set(want)), want_path, ", %d allowed to differ" % len(may & set(want)) if may else ""))
EOF
}

inputs() {
  x mkdir -p "$runs"
  if [ "$dry" = 1 ]; then
    note "check $b5 against $runs/inputs.json (4.1)"
  else
    mkdir -p "$b6"
    hashes "$runs/inputs.json" "$b5" "$runs/inputs-check.json"
  fi
  done_ inputs.done
}

fix_plan() {
  need inputs.done inputs
  note "the full copy $b5 -> $b6 (clones where the volume has them), checked file by file"
  if [ "$dry" = 1 ]; then
    x cp -cR "$b5/." "$b6/"
  else
    (cp -cR "$b5/." "$b6/" 2>/dev/null || cp -R "$b5/." "$b6/")
    python3 -I - "$b5" "$b6" "$runs/copy.json" <<'EOF'
import hashlib, json, os, sys
a, b, out = sys.argv[1:]
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for x in iter(lambda: f.read(1 << 20), b""):
            h.update(x)
    return h.hexdigest()
files = {}
for d, _, fs in os.walk(a):
    for f in fs:
        rel = os.path.relpath(os.path.join(d, f), a)
        ha, hb = sha(os.path.join(a, rel)), sha(os.path.join(b, rel))
        if ha != hb:
            sys.exit(f"{rel}: the copy differs")
        files[rel] = {"bytes": os.path.getsize(os.path.join(a, rel)), "sha256": ha}
json.dump({"about": "s6-route v6 (4.1): the full copy of build5/ into build6/, file by file", "files": files}, open(out, "w"), indent=1)
print(f"{len(files)} files copied, every one with its sha256")
EOF
  fi
  x mkdir -p "$fix/emb"
  x cp "$b5/emb/potion-requests.npy" "$b5/emb/potion-requests.meta.json" "$fix/emb/"
  logged v6-fix-plan "$D" --seed 1234 route v5-plan --v2 "$v2" --v4 "$b4" --out "$fix" "${plan_flags[@]}"
  note "every plan outside the When2Call group, and the fix's v5_tools files, are v5's"
  if [ "$dry" != 1 ]; then
    python3 -I - "$b5" "$fix" <<'EOF'
import hashlib, json, os, sys
b5, fix = sys.argv[1:]
def lines(p):
    with open(p) as f:
        return [l for l in f if l.strip()]
a, b = lines(os.path.join(b5, "stage/plan5.jsonl")), lines(os.path.join(fix, "stage/plan5.jsonl"))
if len(a) != len(b):
    sys.exit("the fix's plan holds %d requests, v5's %d" % (len(b), len(a)))
n = w = 0
for x, y in zip(a, b):
    if json.loads(x)["group"] == "when2call":
        w += 1
        if json.loads(x)["id"] != json.loads(y)["id"]:
            sys.exit("the plans are not in the same order")
        continue
    if x != y:
        sys.exit("a plan outside the When2Call group differs: %s" % json.loads(x)["id"])
    n += 1
for f in ("val/v5_tools.jsonl", "val/v5_tools_masked.jsonl"):
    if hashlib.sha256(open(os.path.join(b5, f), "rb").read()).hexdigest() != hashlib.sha256(open(os.path.join(fix, f), "rb").read()).hexdigest():
        sys.exit(f"{f}: the fix's plan wrote another file")
print(f"{n} plans outside the When2Call group as v5's, {w} of the group replanned")
EOF
  fi
  if [ -n "$check_against" ]; then
    note "against $check_against: only the no-call plans that had no anchor (both slates at L6) may change"
    if [ "$dry" != 1 ]; then
      python3 -I - "$check_against/stage/plan5.jsonl" "$fix/stage/plan5.jsonl" "$runs/plan-diff.json" <<'EOF'
import json, sys
a, b, out = sys.argv[1:]
changed, same, bad = [], 0, []
for x, y in zip(open(a), open(b)):
    if x == y:
        same += 1
        continue
    p = json.loads(x)
    if p["group"] == "when2call" and p["gold"] is None and p["natural"]["level"] == 6 and p["hard"]["level"] == 6:
        changed.append(p["id"])
    else:
        bad.append(p["id"])
json.dump({"against": a, "plans unchanged": same, "plans changed (no-call, no anchor before)": len(changed),
           "requests changed": changed, "other plans changed": bad}, open(out, "w"), indent=1)
if bad:
    sys.exit("%d plans changed that had an anchor or a tool gold: %s" % (len(bad), bad[:5]))
print(f"{same} plans as the previous fix's, {len(changed)} no-call plans without an anchor replanned")
EOF
    fi
  fi
  done_ fix-plan.done
}

fix_teacher_jobs() {
  need fix-plan.done fix-plan
  logged v6-fix-items "$D" route v5-fix-items --v2 "$v2" --base "$b6" --fix "$fix" --group when2call ${also[@]+"${also[@]}"}
  note "the guard of the items: training and validation requests with their hard slate, at most 1,043"
  if [ "$dry" != 1 ]; then
    python3 -I - "$fix/teacher/false-negatives.items.jsonl" <<'EOF'
import json, sys
n, ids = 0, set()
for l in open(sys.argv[1]):
    x = json.loads(l)
    n += 1
    if set(x) != {"id", "split", "request", "options"} or x["split"] not in ("train", "val"):
        sys.exit(f"an item that is not a training or validation request with its hard slate: {x.get('id')}")
    if not x["id"].startswith("when2call:") or x["id"] in ids:
        sys.exit(f"an item of another group, or repeated: {x['id']}")
    ids.add(x["id"])
if n > 1043:
    sys.exit(f"{n} items, more than the 1,043 hard no-call slates of 4.8")
print(f"{n} false-negative items ({(n + 19) // 20} calls of 20)")
EOF
  fi
  done_ fix-teacher-jobs.done
  note "next, by the lead: BUILD=$fix caffeinate -i scripts/route-teacher.sh false-negatives loop"
}

fix_build() {
  need fix-teacher-jobs.done fix-teacher-jobs
  logged v6-fix-build "$D" --seed 1234 route v5-slates --v2 "$v2" --v4 "$b4" --out "$b6" --only-group when2call --fix "$fix" \
    ${also[@]+"${also[@]}"}
  note "every file of $b6 against $b5 (the copy): all equal but the When2Call records and counts/v5-slates.json"
  if [ "$dry" != 1 ]; then
    hashes "$runs/copy.json" "$b6" "$runs/rebuild.json" records/when2call/train.jsonl records/when2call/val.jsonl \
      counts/v5-slates.json
    python3 -I - "$b6" "$fix" <<'EOF'
import hashlib, os, sys
b6, fix = sys.argv[1:]
print("for the amendment of 4.8 (file | bytes | sha256):")
print("| file | bytes | sha256 |")
print("|---|---:|---|")
for root, rel in ((b6, "records/when2call/train.jsonl"), (b6, "records/when2call/val.jsonl"), (b6, "counts/v5-slates.json"),
                  (fix, "teacher/false-negatives.items.jsonl"), (fix, "teacher/false-negatives.jsonl"),
                  (fix, "teacher/false-negatives.refused.jsonl"), (fix, "stage/plan5.jsonl")):
    p = os.path.join(root, rel)
    if os.path.exists(p):
        h = hashlib.sha256(open(p, "rb").read()).hexdigest()
        name = os.path.relpath(p, b6)
        print(f"| `{name}` | {os.path.getsize(p):,} | `{h}` |")
EOF
  fi
  if [ -n "$check_against" ] && [ "$dry" != 1 ]; then
    note "the records of the group against the copy's: only the none-right natural and hard slates of the replanned requests may change"
    python3 -I - "$b5" "$b6" "$runs/plan-diff.json" "$runs/records-diff.json" <<'EOF'
import json, os, sys
b5, b6, diff, out = sys.argv[1:]
changed = set(json.load(open(diff))["requests changed"])
res, bad = {}, []
for split in ("train", "val"):
    rel = f"records/when2call/{split}.jsonl"
    a = [l for l in open(os.path.join(b5, rel)) if l.strip()]
    b = [l for l in open(os.path.join(b6, rel)) if l.strip()]
    if len(a) != len(b):
        sys.exit(f"{rel}: {len(b)} records against {len(a)}")
    same = diff_n = 0
    for x, y in zip(a, b):
        rx, ry = json.loads(x), json.loads(y)
        if rx["uid"] != ry["uid"]:
            sys.exit(f"{rel}: not the same uids")
        if x == y:
            same += 1
            continue
        diff_n += 1
        if not (rx["request"] in changed and rx["kind"] in ("natural", "hard") and rx["target"][-1] == 1.0):
            bad.append(rx["uid"])
    res[rel] = {"records identical": same, "records rebuilt": diff_n}
json.dump({"files": res, "other records changed": bad}, open(out, "w"), indent=1)
if bad:
    sys.exit("records changed that are not the replanned no-call slates: %s" % bad[:5])
print(json.dumps(res))
EOF
  fi
  done_ fix-build.done
}

fix_checks() {
  need fix-build.done fix-build
  x mkdir -p "$runs"
  local st=0
  local args=(checks --build "$b6" --requests "$v2/stage/requests.jsonl" --tools "$v2/stage/tools.jsonl"
    --pools "$b6/stage/pools5.json" --plan-counts "$b6/counts/v5-plan.json"
    --val "$b6/val/v5_tools.jsonl" "$b6/val/v5_tools_masked.jsonl" --v5-report "$before_report"
    --out "$runs/checks.json" --md "$runs/checks.md")
  if [ "$dry" = 1 ]; then
    x python3 -I scripts/s6-route6-checks.py "${args[@]}"
  else
    nice -n 19 python3 -I scripts/s6-route6-checks.py "${args[@]}" || st=$?
  fi
  case "$st" in
    0) done_ fix-checks.done ;;
    2) echo "the stop rule of 4.8 fires: v6 stops before training, the table is published ($runs/checks.md)" >&2; exit 2 ;;
    *) exit "$st" ;;
  esac
}

spread() {
  need fix-checks.done fix-checks
  x python3 -I scripts/s6-route6-checks.py spread --build "$b6" --v5-build "$b5" --before-label "$before_label" --plan-counts "$b6/counts/v5-plan.json" \
    --out "$runs/spread.json" --md "$runs/spread.md"
  done_ spread.done
  note "next, by the lead: the amendment of 4.8 (the hashes above, the counts by level, the teacher's counts, the checks and $runs/spread.md), with $runs/inputs6.json"
}

inputs_ok() {
  need spread.done spread
  if [ "$dry" = 1 ]; then
    note "check $b6 against $runs/inputs6.json (the amendment of 4.8); write $runs/inputs.ok"
  else
    [ -f "$runs/inputs6.json" ] || { echo "no $runs/inputs6.json: the amendment of 4.8 writes it" >&2; exit 1; }
    hashes "$runs/inputs6.json" "$b6" "$runs/inputs6-check.json"
    cp "$runs/inputs6.json" "$runs/inputs.ok"
  fi
  done_ inputs-ok.done
}

record_files() {
  local f
  for f in "$b6"/records/*/*.jsonl "$b6/val/v5_tools.jsonl" "$b6/val/v5_tools_masked.jsonl"; do
    printf '%s,' "$f"
  done
}

placeholder() {
  need inputs-ok.done inputs-ok
  local files st=0
  files=$(record_files)
  files=${files%,}
  if [ -f "$runs/placeholder.json" ] && [ "$dry" != 1 ]; then
    note "the placeholder was taken before (the published-files diagnostic): checked against v6's requests"
    nice -n 19 "$D" route blind --check-placeholder "$files" --keep --placeholder "$runs/placeholder.json" \
      --out "$runs/placeholder-check.json" || st=$?
  else
    x "$D" route blind --check-placeholder "$files" --out "$runs/placeholder.json" || st=$?
  fi
  [ "$st" = 0 ] || { echo "the placeholder occurs as a request ($runs/placeholder*.json): 4.3 takes the next one before anything reads it" >&2; exit 2; }
  done_ placeholder.done
}

support() {
  need placeholder.done placeholder
  x python3 -I scripts/s6-route6-blind.py support --records "$b6"/records/*/*.jsonl --out "$runs/support.json"
  done_ support.done
}

checks_read() {
  need support.done support
  x python3 -I scripts/s6-route5-checks.py --requests "$v2/stage/requests.jsonl" --tools "$v2/stage/tools.jsonl" \
    --pools "$b6/stage/pools5.json" --plan-counts "$b6/counts/v5-plan.json" --val "$v2/val/v_tools.jsonl" "$v2/val/v_tools_masked.jsonl" \
    "$b4/val/v4_tools.jsonl" "$b4/val/v4_tools_masked.jsonl" "$b6/val/r_tools_renamed.model.jsonl" \
    "$b6/val/r_tools_natural.model.jsonl" --fresh "$v2/fresh/f_tools.jsonl" "$v2/fresh/f_tools_masked.jsonl" \
    --out "$runs/checks-read-files.json" --md "$runs/checks-read-files.md"
  done_ checks-read.done
}

blind_val() {
  need checks-read.done checks-read
  local f
  for f in v5_tools v5_tools_masked; do
    logged "blind-$f" "$D" route blind --in "$b6/val/$f.jsonl" --out "$b6/val/$f" --placeholder "$runs/placeholder.json" \
      --render scores-in-options
  done
  done_ blind-val.done
}

baselines_val() {
  need blind-val.done blind-val
  local out=${EVAL_OUT:-$data/eval/route6-val} sys split files
  for sys in potion-cos bm25; do
    for split in val v4val v5val; do
      case "$split" in
        val) files=v_tools,v_tools_masked ;;
        v4val) files=v4_tools,v4_tools_masked ;;
        v5val) files=v5_tools,v5_tools_masked,v5_tools.blind,v5_tools_masked.blind ;;
      esac
      x env BUILD="$v2" BUILD4="$b4" BUILD5="$b6" "$PY" -I scripts/route-baselines.py score --system "$sys" \
        --split "$split" --files "$files" --out-root "$out/$split"
    done
  done
  if [ "$dry" != 1 ]; then
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
json.dump({"about": "s6-route v6 (docs/s6-route6-prereg.md 4.5): BM25 and potion-cos by the frozen baseline code, v2's fitted values, ordinary and blind (B0); reported",
           "accuracy": res}, open(out, "w"), indent=1)
print(json.dumps(res))
EOF
  fi
  note "the blind statistics of the baselines: scripts/s6-route6-blind.py read --slates $b6/val/v5_tools.blind.jsonl --dump $out/v5val/<system>/v5_tools.blind.jsonl"
  done_ baselines-val.done
}

fresh() {
  need baselines-val.done baselines-val
  local parquet
  parquet=$(python3 -I -c '
import json, sys
s = json.load(open(sys.argv[1]))["v4_sources"][0]
print(",".join(sys.argv[2] + "/src/" + s["name"] + "/" + f for f in s["include"] if f.endswith(".parquet")))' scripts/route/pins.json "$data")
  logged toucan-rows "$D" route toucan-rows --input "$parquet" --out "$t5" --manifest "$data/MANIFEST4.json"
  logged toucan-stage "$D" route toucan-stage --t4 "$t5"
  x env UV_PROJECT_ENVIRONMENT="$data/envs/route" UV_PYTHON_PREFERENCE=only-managed \
    uv sync --frozen --no-dev --project scripts/route/env -q
  x "$PY" -I scripts/route-embed.py --build "$t5"
  logged toucan "$D" --seed 1236 route toucan --rule v5 --v2 "$v2" --t4 "$t5" --record "$runs"
  done_ fresh.done
  note "next, by the lead: BUILD=$t5 caffeinate -i scripts/route-teacher.sh rewrite-description loop (fresh-tools: tools only)"
}

fresh_tools() {
  need fresh.done fresh
  if [ "$dry" != 1 ]; then
    python3 -I - "$t5/teacher/rewrite-description.items.jsonl" "$t5/teacher/rewrite-description.jsonl" <<'EOF' ||
import json, os, sys
items, ans = sys.argv[1:]
ids = {json.loads(l)["id"] for l in open(items)}
done = {json.loads(l)["id"] for l in open(ans)} if os.path.exists(ans) else set()
ref = ans.replace(".jsonl", ".refused.jsonl")
done |= {json.loads(l)["id"] for l in open(ref)} if os.path.exists(ref) else set()
left = len(ids - done)
print(f"{len(ids)} fresh-tools items, {left} left")
sys.exit(1 if left else 0)
EOF
    { echo "the fresh-tools job is not done: BUILD=$t5 caffeinate -i scripts/route-teacher.sh rewrite-description loop" >&2; exit 1; }
  fi
  done_ fresh-tools.done
}

probes() {
  need fresh-tools.done fresh-tools
  logged toucan-probes "$D" --seed 1236 route toucan-probes --rule v5 --t4 "$t5" --record "$runs"
  done_ probes.done
}

counts() {
  need probes.done probes
  local f=$t5/fresh st=0
  x python3 -I scripts/s6-route5-checks.py --requests "$v2/stage/requests.jsonl" --tools "$v2/stage/tools.jsonl" \
    --pools "$b6/stage/pools5.json" --fresh "$f/t5_tools.jsonl" "$f/t5_tools_masked.jsonl" "$f/t5_tools_rw.jsonl" \
    "$f/t5_tools_masked_rw.jsonl" --fresh-manifest "$f/FRESH5.json" --out "$runs/t5-checks.json" --md "$runs/t5-checks.md"
  if [ "$dry" = 1 ]; then
    x "$D" route blind --check-placeholder "$f/t5_tools.jsonl" --keep --placeholder "$runs/placeholder.json" --out "$runs/t5-placeholder.json"
  else
    nice -n 19 "$D" route blind --check-placeholder "$f/t5_tools.jsonl" --keep --placeholder "$runs/placeholder.json" \
      --out "$runs/t5-placeholder.json" || st=$?
    [ "$st" = 0 ] || { echo "the placeholder is a request of T5 ($runs/t5-placeholder.json): the counts amendment says so" >&2; exit 2; }
    python3 -I - "$runs/counts.json" "$f/t5_tools.jsonl" <<'EOF'
import collections, json, sys
c = json.load(open(sys.argv[1]))
recs = [json.loads(l) for l in open(sys.argv[2]) if l.strip()]
tr = [r for r in recs if r["target"].index(1.0) < len(r["opts"]) - 1]
mates = collections.Counter(sum(x == "exempt" for x in r.get("roles", [])) for r in tr)
keys = ["rows read by subset", "rows dropped, by filter", "slates", "servers", "gold clusters",
        "share of the slates on the three largest servers", "none share", "slates by subset",
        "slates by level (6.2 item 5, reported)", "C6 (reported, not judged: 6.2)"]
out = {k: c.get(k) for k in keys}
out["T5's chance c (the mean of 1/k over its tool-right slates)"] = sum(1 / (len(r["opts"]) - 1) for r in tr) / max(len(tr), 1)
out["tool-right slates by the number of the gold's server's tools shown"] = dict(sorted(mates.items()))
print(json.dumps(out, indent=1))
if c["slates"] < 600:
    sys.exit("fewer than 600 slates (6.3): v6 is not trained; the amendment says so")
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
    fix-plan) fix_plan ;;
    fix-teacher-jobs) fix_teacher_jobs ;;
    fix-build) fix_build ;;
    fix-checks) fix_checks ;;
    inputs-ok) inputs_ok ;;
    checks-read) checks_read ;;
    blind-val) blind_val ;;
    baselines-val) baselines_val ;;
    fresh-tools) fresh_tools ;;
    *) "$s" ;;
  esac
done
echo "== done: $b6"
