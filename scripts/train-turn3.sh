#!/usr/bin/env bash
# s6-ear v3 (s6-ear3-nano): the end-of-turn detector of docs/s6-ear3-prereg.md, v2's graph trained on
# the silences the deployed policy asks about and on real two-speaker conversations. Modelled on
# scripts/train-turn.sh (v2) and scripts/route7-gate.sh (stages and markers).
#
# Usage: scripts/train-turn3.sh <stage> [<stage> ...]   (stages are launched one by one, in this order;
#   each is refused until the one before it has written its marker, <run>/stages/<stage>.done)
#   conv-data      §3.2-3.3 (CPU, about 49 GB streamed): shard by shard of otoSpeech, sky-data fetch-conv
#                  (the download, every session through §3.2, the records, the shard deleted), Smart Turn's
#                  recorded opinion on the shard's validation and test tails (scripts/turn-teacher.py
#                  --records), [LABELCHECK=1: the listening sample's snippets], Silero on its validation
#                  channels (scripts/turn-vadcheck.py), each removing its transient inputs; then
#                  fetch-conv --finalize (the caps' exact draw, conv/CONV.json). Resumable shard by shard;
#                  exit status 3 when the disk rule stops it
#   st-extend      §3.4 (CPU, 41 GB streamed): sky-data turn-extend over v2's 83 training shards, then --test
#   eotbench-data  §8.3 (CPU, 2 GB streamed): per file, sky-data turn-eotbench --spans at v2's commit, Smart
#                  Turn at the span's start (the harness check) and at + 0.2 s (scripts/ear-baseline.py), the
#                  file removed; no label is read
#   teacher-conv   §5: every shard's recorded opinion on file (a shard left over is scored)
#   crosscheck     §3.7 as amended (amendment 2): sky-data conv-crosscheck, the instant label of every
#                  eligible energy record; both bars (agreement >= 0.85, drop share <= 0.30), else the build
#                  stops before any training
#   bleed          §3.7: scripts/turn-gate.py bleed (no d above 0.60, else the build stops); the counts of
#                  §3.2 and both numbers are copied to runs/s6-ear3-nano/data/, to be committed
#   labelcheck     §3.7, optional (LABELCHECK=1 during conv-data, then the owner's answers): the sheet, or
#                  with the answers the score; no stage needs it
#   harness3       §8.1: the policy port (sky-data turn-policy-check); v2's file through the evaluation path
#                  on Smart Turn's test; v2 at the span's start (d = 0 only); scripts/turn-gate.py harness3
#   smoke          §6.3 (GPU, 5 working minutes): 2,000 records of each source from v2's weights, into
#                  runs/s6-ear3-smoke; its numbers are not looked at. Refused until the counts, the
#                  cross-check and the bleed probe are committed (§3.7)
#   pilot          §6.4-6.5 (GPU, 4 x 12 working minutes): P1-P3 from v2 at 0.25, 0.5 and 1 x 0.0007745, P4
#                  from scratch; v2's file on the conversation validation at 0.2 s; scripts/turn-gate.py
#                  pilot3 -> runs/s6-ear3-pilot/choice.json (exit 4: the stop rule fired, no main run)
#   train          §6.6 (GPU, 120 working minutes from v2, 240 from scratch): the chosen start and rate, batch
#                  128, seed 1, QAT over the last 15%, EMA 0.999, the brake
#   export         §7: the checkpoint of scripts/turn-gate.py checkpoint3, ternary GGUF
#   calibrate      §7: temperature and tau85 on Smart Turn validation at d = 0 (v2's rules), the calibrated
#                  file; v3 on the conversation validation at 0.2 s (tau_c, reported)
#   latency        §8.5: v3 and v2's file natively and in WASM in the same session, no training running
#   wasm           §7: parity on 20 Smart Turn test clips and 20 conversation test records (no label read)
#   eval           §8.2: opens the tests (scripts/turn3-guard.py open, once), v3 on Smart Turn's test (the
#                  WAV path), v3 and v2 on the test clips extended by 0.2 and 0.6 s
#   eotbench       §8.3: v3 and v2 at every ask of the span records, v3 at the span's start
#   conv-test      §8.4: v3 and v2 at every ask of the conversation test records
#   gate           §9: scripts/turn-gate.py gate3 -> runs/s6-ear3-nano/GATE.md, GATE.json
#   Every command that reads a test with v3 runs after scripts/turn3-guard.py (the tests opened once, with the
#   file as exported); eval, eotbench, conv-test and gate are each run once.
#
# Environment:
#   TURN3_DATA=<main checkout>/../skycmd-data/s6-ear3   v3's data folder (outside the repository; the main
#                       checkout is found through git, so a worktree resolves to the same folder; printed at start)
#   TURN_DATA=<main checkout>/../skycmd-data/s6-ear2    v2's (its train/ and test/ stores, reused)
#   EAR=data/ear        v1's data (clips/smart-turn.jsonl, jev/turn_end.jsonl and the test WAVs they name)
#   V2_CKPT=$TURN_DATA/checkpoints/last-ema   v2's EMA checkpoint (sha256 in runs/s6-ear3-nano/init.json)
#   V2_MODEL=site/public/models/s6-ear2-nano.gguf   v2's file (its sha256 is checked)
#   HF_TOKEN            the gated corpus's token; read from the Hugging Face cache when unset (never printed)
#   MIN_FREE_GB=8       free disk every step leaves (§3.8); checked before every download and build
#   THREADS=4  PROCS=4  feature threads, teacher processes (CPU)
#   SHARDS              conv-data on these shards only (`0 1 2`; default all 151)
#   LABELCHECK=1        conv-data also keeps the listening check's snippets (optional, §3.7)
#   TIME_SCALE=1        multiplies every working-minute budget
#   SKY_DUTY=0.5  SKY_COOL=45:15  SKY_BRAKE_FILE=runs/brake   the brake (scripts/auto-brake.sh)
#   SKY_MLX_CACHE_GB=2  SKY_BIN (prebuilt release binaries; else built here)
#   RESUME=1            continue a training from its last checkpoint (only after a technical failure, §6.7)
#   SKY_OUT             a prefix for everything written under runs/ and models/
#   SMOKE=1             the chain under a throwaway prefix (default runs/turn3-smoke/) on a synthetic
#                       conversation fixture (sky-data turn3-fixture) and v2's own clips, validation data in
#                       place of every test, budgets of 1-2 minutes (SMOKE_CPU=1: training on the CPU);
#                       nothing of it is a result
#   DRY_RUN=1           print the commands (nothing is built, downloaded, trained, read or written)
set -euo pipefail
cd "$(dirname "$0")/.."
export DEVELOPER_DIR=${DEVELOPER_DIR:-/Applications/Xcode.app/Contents/Developer}
export CARGO_TARGET_DIR=${CARGO_TARGET_DIR:-$PWD/target/s3-train}
export SKY_DUTY=${SKY_DUTY:-0.5}
export SKY_COOL=${SKY_COOL-45:15}
export SKY_BRAKE_FILE=${SKY_BRAKE_FILE:-$PWD/runs/brake}
export SKY_MLX_CACHE_GB=${SKY_MLX_CACHE_GB:-2}
dry=${DRY_RUN:-0}
smoke=${SMOKE:-}
if [ -n "$smoke" ]; then
  SKY_OUT=${SKY_OUT:-runs/turn3-smoke/}
fi
sky_out=${SKY_OUT:-}
B=${SKY_BIN:-$CARGO_TARGET_DIR/release}
T=$B/sky-train
D=$B/sky-data
S=$B/sky-turn
name=s6-ear3-nano
cfg=configs/$name.json
# the data folders sit beside the main checkout (../skycmd-data), found through git's common folder so that
# any other working tree of the repository resolves to the same place; TURN3_DATA and TURN_DATA come first
git_common=$(git rev-parse --path-format=absolute --git-common-dir 2>/dev/null || echo "$PWD/.git")
data_root=$(cd "$git_common/../.." && pwd)/skycmd-data
v2data=${TURN_DATA:-$data_root/s6-ear2}
data=${TURN3_DATA:-$data_root/s6-ear3}
[ -n "$smoke" ] && data=${sky_out}data
echo "== data: $data (v2's: $v2data)"
ear=${EAR:-data/ear}
v2_ckpt=${V2_CKPT:-$v2data/checkpoints/last-ema}
v2_model=${V2_MODEL:-site/public/models/s6-ear2-nano.gguf}
v2_sha=204354653fd864708c3206f486f255d3fedff81956fd7b82aae560185a9be783
v2_calib=runs/s6-ear2-nano/calib.json
init_list=runs/$name/init.json
run=${sky_out}runs/$name
markers=$run/stages
pilot_dir=${sky_out}runs/s6-ear3-pilot
smoke_dir=${sky_out}runs/s6-ear3-smoke
m=${sky_out}models/s6-ear3
model=$m/$name.gguf
conv=$data/conv
store=$conv/store
spans=$data/eotbench/store
eot_commit=ae834af9e9bddf73d1f550c190aaeb99bf31f523
min_free=${MIN_FREE_GB:-8}
threads=${THREADS:-4}
procs=${PROCS:-4}
tmin() { LC_ALL=C awk -v a="$1" -v b="${TIME_SCALE:-1}" 'BEGIN { printf "%g", a * b }'; }
asks=200,400,600,800,1000,1200,1400,1600,1800,2000,2200,2400
iters=1000
eval_split="test"
smoke_min=5 pilot_min=12
max_st=0
cpu=()
if [ -n "$smoke" ]; then
  iters=20 eval_split="val" smoke_min=1 pilot_min=1 max_st=2000
  [ -n "${SMOKE_CPU:-}" ] && cpu=(--cpu)
  case "$run" in
    "runs/$name" | "./runs/$name" | "$PWD/runs/$name")
      echo "SMOKE=1 needs a prefix that is not the real run's (SKY_OUT=$sky_out)" >&2
      exit 2
      ;;
  esac
  echo "== SMOKE: everything under $sky_out; nothing of it is a result"
fi
stages=(conv-data st-extend eotbench-data teacher-conv crosscheck bleed harness3 smoke pilot train export calibrate
  latency wasm eval eotbench conv-test gate)
# the gated corpus's token, from the cache when unset; it is never printed (DRY_RUN prints arguments only)
if [ -z "${HF_TOKEN:-}" ] && [ -s "${HF_HOME:-$HOME/.cache/huggingface}/token" ]; then
  HF_TOKEN=$(cat "${HF_HOME:-$HOME/.cache/huggingface}/token")
  export HF_TOKEN
fi

# x <command...>: runs it (nice), or prints it under DRY_RUN=1.
x() {
  if [ "$dry" = 1 ]; then
    printf '+'
    printf ' %q' "$@"
    printf '\n'
  else
    nice -n 19 "$@"
  fi
}
# logged <file> <command...>: x, with what the command prints kept in <file>.
logged() {
  local log=$1
  shift
  if [ "$dry" = 1 ]; then
    x "$@"
  else
    mkdir -p "$(dirname "$log")"
    nice -n 19 "$@" 2>&1 | tee -a "$log"
  fi
}
note() { echo "# $*"; }
die() {
  echo "$1" >&2
  exit "${2:-1}"
}
gate_py() { x python3 -I scripts/turn-gate.py "$@"; }
uvpy() { x uv run --quiet --python 3.12 "$@"; }
# need <stage>: the marker of the stage before
need() {
  if [ "$dry" = 1 ]; then
    note "requires $markers/$1.done"
    return
  fi
  [ -f "$markers/$1.done" ] || die "no $markers/$1.done: scripts/train-turn3.sh $1 first"
}
done_() {
  if [ "$dry" = 1 ]; then
    note "writes $markers/$1.done"
  else
    mkdir -p "$markers"
    date '+%F %T' >"$markers/$1.done"
  fi
}
# the disk floor (§3.8), before every download and build: MIN_FREE_GB + <gb> free on the data disk
floor() {
  local need_gb=$1 kb
  if [ "$dry" = 1 ]; then
    note "requires $((min_free + need_gb)) GB free on the disk of $data"
    return
  fi
  mkdir -p "$data"
  kb=$(df -Pk "$data" | awk 'NR == 2 { print $4 }')
  [ "$kb" -ge $(((min_free + need_gb) * 1024 * 1024)) ] ||
    die "below the disk floor: $((kb / 1024 / 1024)) GB free on $data, $((min_free + need_gb)) GB needed (MIN_FREE_GB=$min_free)" 3
}
# stop_on_disk <status>: the disk rule stopped a step (exit status 3); it resumes when run again
stop_on_disk() {
  if [ "$1" = 3 ]; then
    die "== stopped by the disk rule (MIN_FREE_GB=$min_free): free some space and run the stage again (it resumes)" 3
  fi
  exit "$1"
}
test_guard() {
  if [ "$dry" = 1 ]; then
    x python3 -I scripts/turn3-guard.py check --run "$run" --model "$model" --stage "$1"
  else
    python3 -I scripts/turn3-guard.py check --run "$run" --model "$model" --stage "$1" >/dev/null ||
      die "the tests are refused (scripts/turn3-guard.py): $1" 3
  fi
}
# the teacher's recorded opinion complete on every shard (amendment 3; read-only): a table, exit 3 otherwise
teacher_guard() {
  if [ "$dry" = 1 ]; then
    x python3 -I scripts/turn3-guard.py teacher --conv "$conv"
  else
    python3 -I scripts/turn3-guard.py teacher --conv "$conv" ||
      die "== technical failure: the teacher's recorded opinion is incomplete (the shards above); rescore or rebuild them, then run the stage again (amendment 3)" 3
  fi
}
sha() { shasum -a 256 "$1" | cut -d' ' -f1; }
v2_file() {
  if [ "$dry" != 1 ]; then
    [ "$(sha "$v2_model")" = "$v2_sha" ] || die "$v2_model is not v2's file (sha256 $v2_sha)"
  fi
}
build() {
  if [ "$dry" != 1 ] && [ -z "${SKY_BIN:-}" ]; then
    floor 2
    nice -n 19 cargo build -q --release -p sky-infer -p sky-data -p sky-train
  fi
}
tag5() { printf '%05d' "$1"; }

# the recipe's optimizer flags (OPT), as scripts/train-turn.sh reads them
opt=()
recipe() {
  local OPT='' LR_SCALE='' CONFIG='' env_sh
  if [ -f runs/recipe/recipe.json ]; then
    # on its own line, so that a failure of s6-adopt.py stops the script (set -e) instead of an empty eval
    env_sh=$(python3 -I scripts/s6-adopt.py env s6-ear-nano)
    eval "$env_sh"
  elif [ "$dry" = 1 ]; then
    note "no runs/recipe/recipe.json: the commands below show the default options"
    OPT=--no-muon
  else
    die "no runs/recipe/recipe.json: the recipe is adopted first (scripts/train-all.sh adopt)"
  fi
  : "$LR_SCALE" "$CONFIG"
  read -r -a opt <<<"$OPT"
}

# ----------------------------------------------------------------------------- data

conv_data() {
  if [ -n "$smoke" ]; then
    note "SMOKE: a synthetic conversation fixture and a stand-in extension of v2's first clips"
    x "$D" turn3-fixture --out "$data" --sessions 25 --st "$v2data/train" --st-max "$max_st"
    done_ conv-data
    return
  fi
  local shards i t
  read -r -a shards <<<"${SHARDS:-$(seq -s ' ' 0 150)}"
  for i in "${shards[@]}"; do
    t=$(tag5 "$i")
    floor 1
    x "$D" fetch-conv --out "$data" --shards "$i" --threads "$threads" --min-free-gb "$min_free" || stop_on_disk $?
    uvpy scripts/turn-teacher.py --records "$conv" --tag "$t" --procs "$procs"
    if [ -n "${LABELCHECK:-}" ]; then
      x python3 -I scripts/turn-labelcheck.py collect --conv "$conv" --tag "$t"
    fi
    uvpy scripts/turn-vadcheck.py --conv "$conv" --tag "$t" --pin "$run/silero.json"
  done
  if [ -n "${SHARDS:-}" ]; then
    note "SHARDS=$SHARDS: the other shards remain; the stage is finished by a run over all of them"
    return
  fi
  x "$D" fetch-conv --out "$data" --finalize
  x mkdir -p "$run/data"
  x cp "$conv/CONV.json" "$run/data/CONV.json"
  done_ conv-data
}

st_extend() {
  need conv-data
  if [ -n "$smoke" ]; then
    note "SMOKE: the fixture's stand-in extension ($data/st-ext); no test extension"
    done_ st-extend
    return
  fi
  floor 7
  x "$D" turn-extend --out "$data" --v2 "$v2data" --threads "$threads" --min-free-gb "$min_free" || stop_on_disk $?
  x "$D" turn-extend --out "$data" --v2 "$v2data" --test --clips "$ear/clips/smart-turn.jsonl" --threads "$threads" \
    --min-free-gb "$min_free" || stop_on_disk $?
  done_ st-extend
}

eotbench_data() {
  need st-extend
  if [ -n "$smoke" ]; then
    note "SMOKE: no eot-bench"
    done_ eotbench-data
    return
  fi
  local k files=15
  for ((k = 0; k < files; k++)); do
    floor 1
    x "$D" turn-eotbench --out "$data" --files "$k" --spans --revision "$eot_commit" --threads "$threads" \
      --min-free-gb "$min_free" || stop_on_disk $?
    # Smart Turn on the file's spans, at their start (§8.1) and at + 0.2 s (§8.3); no label is read
    uvpy scripts/ear-baseline.py --eotbench "$data/eotbench" --tag all
    uvpy scripts/ear-baseline.py --eotbench "$data/eotbench" --tag all --offset-ms 200
    x "$D" turn-eotbench --out "$data" --files "$k" --spans --revision "$eot_commit" --drop
  done
  done_ eotbench-data
}

teacher_conv() {
  need eotbench-data
  if [ -n "$smoke" ]; then
    note "SMOKE: no teacher pass on the fixture"
    done_ teacher-conv
    return
  fi
  local f t missing=0
  if [ "$dry" = 1 ]; then
    note "for every $conv/conv-<tag>.json without teacher-conv-<tag>.done: scripts/turn-teacher.py --records"
    uvpy scripts/turn-teacher.py --records "$conv" --tag "<tag>" --procs "$procs"
  else
    for f in "$conv"/conv-*.json; do
      t=$(basename "$f" .json)
      t=${t#conv-}
      [ -f "$conv/teacher-conv-$t.done" ] && continue
      uvpy scripts/turn-teacher.py --records "$conv" --tag "$t" --procs "$procs" || missing=1
    done
    [ "$missing" = 0 ] || die "a shard's recorded opinion is missing: its tails are gone; rebuild the shard"
  fi
  teacher_guard
  done_ teacher-conv
}

crosscheck() {
  need teacher-conv
  local st=0
  x "$D" conv-crosscheck --out "$data" || st=$?
  x mkdir -p "$run/data"
  x cp "$conv/CROSSCHECK.json" "$run/data/CROSSCHECK.json"
  [ "$st" = 0 ] || die "== the label cross-check fails a bar (§3.7): the build stops before any training; only an amendment can revise the rule, the filter or the corpus ($run/data/CROSSCHECK.json)" 5
  done_ crosscheck
}

bleed() {
  need crosscheck
  local st=0
  x mkdir -p "$run/data"
  gate_py bleed --store "$store" --out "$run/data/bleed.json" || st=$?
  [ "$st" = 0 ] || die "== the bleed probe is above 0.60 (§3.7): v3 stops before training; an amendment tightens the bleed filter ($run/data/bleed.json)" 5
  note "commit $run/data/{CONV,CROSSCHECK,bleed}.json (and $run/silero.json) before the smoke stage (§3.7)"
  done_ bleed
}

labelcheck() {
  local lc=$data/labelcheck
  if [ -f "$lc/answers.csv" ] && grep -q ',.' "$lc/answers.csv"; then
    x python3 -I scripts/turn-labelcheck.py score --dir "$lc" --out "$run/data/labelcheck.json"
  else
    x python3 -I scripts/turn-labelcheck.py sheet --dir "$lc"
  fi
}

harness3() {
  need bleed
  local h=$run/harness3
  if [ -n "$smoke" ]; then
    note "SMOKE: no harness check (it reads the test labels); the policy port only"
    x "$D" turn-policy-check --out "$h/policy.json"
    done_ harness3
    return
  fi
  v2_file
  x mkdir -p "$h"
  x "$D" turn-policy-check --out "$h/policy.json"
  # v2's file through v3's evaluation path on Smart Turn's test, and at the span's start only (§3.5)
  x "$S" "$v2_model" --records "$ear/jev/turn_end.jsonl" --split test --out "$h/v2-test.jsonl"
  x "$S" "$v2_model" --store "$spans" --trail-ms 0 --out "$h/v2-eot-d0.jsonl"
  local st=0
  gate_py harness3 --policy "$h/policy.json" --labels "$ear/clips/smart-turn.jsonl" --v2-test "$h/v2-test.jsonl" \
    --spans "$spans" --v2-eot "$h/v2-eot-d0.jsonl" --out "$h/check.json" || st=$?
  [ "$st" = 0 ] || die "== a harness check failed (§8.1): its cause is found and fixed by an amendment before any v3 test score is read ($h/check.json)" 1
  done_ harness3
}

# ----------------------------------------------------------------------------- training

# a run of the pre-registration is made once: an existing one is continued only with RESUME=1
fresh_or_resume() {
  local dir=$1
  if [ "$dry" != 1 ] && [ -f "$dir/last/state.json" ] && [ -z "${RESUME:-}" ] && [ -z "$smoke" ]; then
    die "$dir already holds a run: RESUME=1 continues it after a technical failure (§6.7)"
  fi
  # SMOKE=1 only: the run folder also holds the stages' markers and the gate's inputs, so only the trainer's
  # outputs go there; the other folders are the trainer's alone
  if [ -n "$smoke" ] && [ -z "${RESUME:-}" ] && [ "$dir" = "$run" ]; then x rm -rf "$dir"/{last,last-ema,pre-qat,args.json,log.jsonl,started.json,train.log}
  elif [ -n "$smoke" ] && [ -z "${RESUME:-}" ]; then x rm -rf "$dir"; fi
  return 0
}

# common <records of each source>: the arguments of every v3 training (0: every record)
common() {
  echo --config "$cfg" --data "$v2data/train" --conv "$store" --st-ext "$data/st-ext" --conv-share 0.5 --st-d0 0.25 \
    --batch 128 --seed 1 --max-train "$1" --max-val "$(($1 / 10))" --conv-max-train "$1" --conv-max-val "$(($1 / 10))"
}
from_v2() { echo --init "$v2_ckpt" --init-sha256 "$init_list"; }
scaled() { LC_ALL=C awk -v a="$1" -v b="$2" 'BEGIN { printf "%.4g\n", a * b }'; }

committed() {
  local f
  for f in "$run/data/CONV.json" "$run/data/CROSSCHECK.json" "$run/data/bleed.json"; do
    if [ "$dry" = 1 ]; then
      note "requires $f committed (git)"
      continue
    fi
    if ! git ls-files --error-unmatch "$f" >/dev/null 2>&1 || ! git diff --quiet HEAD -- "$f"; then
      die "$f is not committed as it is: the counts and both numbers are committed before the smoke run (§3.7)"
    fi
  done
}

smoke_() {
  need harness3
  [ -n "$smoke" ] || committed
  # the GPU stages start only on a complete teacher opinion (amendment 3; SMOKE=1 has no teacher pass)
  [ -n "$smoke" ] || teacher_guard
  recipe
  local c f
  # §6.3: 2,000 records of each source
  read -r -a c <<<"$(common 2000)"
  read -r -a f <<<"$(from_v2)"
  fresh_or_resume "$smoke_dir"
  logged "$smoke_dir.log" "$T" turn-train "${c[@]}" "${f[@]}" --out "$smoke_dir" \
    --minutes "$(tmin "$smoke_min")" --qat-frac 0.15 --ema 0.999 --eval-every 1000 --curve-val 200 \
    --adam-lr "$(scaled 0.0007745 0.5)" ${opt[@]+"${opt[@]}"} ${cpu[@]+"${cpu[@]}"}
  note "the smoke's numbers are not looked at (§6.3)"
  done_ smoke
}

pilot() {
  need smoke
  recipe
  v2_file
  local c f arm mult
  read -r -a c <<<"$(common "$max_st")"
  read -r -a f <<<"$(from_v2)"
  for arm in P1 P2 P3 P4; do
    case $arm in P1) mult=0.25 ;; P2) mult=0.5 ;; P3) mult=1 ;; P4) mult=1 ;; esac
    fresh_or_resume "$pilot_dir/$arm"
    if [ "$arm" = P4 ]; then
      logged "$pilot_dir/$arm.log" "$T" turn-train "${c[@]}" --out "$pilot_dir/$arm" --minutes "$(tmin "$pilot_min")" --resume \
        --qat-frac 0 --eval-every 1000000000 --adam-lr "$(scaled 0.0007745 "$mult")" ${opt[@]+"${opt[@]}"} ${cpu[@]+"${cpu[@]}"}
    else
      logged "$pilot_dir/$arm.log" "$T" turn-train "${c[@]}" "${f[@]}" --out "$pilot_dir/$arm" --minutes "$(tmin "$pilot_min")" \
        --resume --qat-frac 0 --eval-every 1000000000 --adam-lr "$(scaled 0.0007745 "$mult")" ${opt[@]+"${opt[@]}"} \
        ${cpu[@]+"${cpu[@]}"}
    fi
  done
  # v2's file on the same validation cuts (§6.5; v2 reads validation only here)
  x "$S" "$v2_model" --store "$store" --split val --trail-ms 200 --out "$pilot_dir/v2-conv-val.jsonl"
  local st=0
  gate_py pilot3 --dir "$pilot_dir" --v2-probs "$pilot_dir/v2-conv-val.jsonl" --conv-store "$store" \
    --out "$pilot_dir/choice.json" || st=$?
  if [ "$st" = 4 ] && [ -z "$smoke" ]; then
    die "== the stop rule fired (§6.5): no main run; v3 is recorded as failed by rule ($pilot_dir/choice.json)" 4
  fi
  [ "$st" = 0 ] || [ "$st" = 4 ] || exit "$st"
  done_ pilot
}

choice() { python3 -I -c 'import json, sys; print(json.load(open(sys.argv[1]))[sys.argv[2]])' "$pilot_dir/choice.json" "$1"; }

train() {
  need pilot
  recipe
  local c start lr minutes init=()
  read -r -a c <<<"$(common "$max_st")"
  if [ "$dry" = 1 ]; then
    start="<the pilot's start>" lr="<the pilot's rate>" minutes="<120 from v2, 240 from scratch>"
    read -r -a init <<<"$(from_v2)"
    note "--init only if the pilot chose a start from v2"
  else
    [ "$(choice stop)" != True ] || [ -n "$smoke" ] || die "the pilot's stop rule fired (§6.5): no main run" 4
    start=$(choice start) lr=$(choice lr) minutes=$(tmin "$(choice minutes)")
    [ -n "$smoke" ] && minutes=$(tmin 2)
    [ "$start" = v2 ] && read -r -a init <<<"$(from_v2)"
  fi
  fresh_or_resume "$run"
  if [ "$dry" != 1 ] && [ ! -f "$run/started.json" ]; then
    mkdir -p "$run"
    python3 -I -c 'import json, sys, time; json.dump({"started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "start": sys.argv[2]}, open(sys.argv[1], "w"))' "$run/started.json" "$start"
  fi
  echo "== $name: $minutes working minutes from $start at $lr (brake $SKY_DUTY ${SKY_COOL:-no long rests}, file $SKY_BRAKE_FILE)"
  logged "$run/train.log" "$T" turn-train "${c[@]}" ${init[@]+"${init[@]}"} --out "$run" --minutes "$minutes" --resume \
    --qat-frac 0.15 --ema 0.999 --eval-every 2000 --curve-val 2000 --adam-lr "$lr" ${opt[@]+"${opt[@]}"} ${cpu[@]+"${cpu[@]}"}
  done_ train
}

export_model() {
  need train
  local ck
  if [ "$dry" = 1 ]; then ck="<last or last-ema>"; else ck=$(python3 -I scripts/turn-gate.py checkpoint3 --log "$run/log.jsonl"); fi
  echo "== exporting $run/$ck"
  x mkdir -p "$m"
  x "$T" turn-export --run "$run/$ck" --format ternary --out "$m/$name-uncal.gguf"
  [ "$dry" = 1 ] || echo "{\"checkpoint\": \"$ck\"}" >"$run/exported.json"
  done_ export
}

calibrate() {
  need export
  local ck
  if [ "$dry" = 1 ]; then ck="<last or last-ema>"; else ck=$(python3 -I -c 'import json, sys; print(json.load(open(sys.argv[1]))["checkpoint"])' "$run/exported.json"); fi
  x "$S" "$m/$name-uncal.gguf" --store "$v2data/train" --split val --out "$run/val-uncal.jsonl"
  gate_py calibrate --probs "$run/val-uncal.jsonl" --store "$v2data/train" --out "$run/calib.json"
  x "$T" turn-export --run "$run/$ck" --format ternary --calib "$run/calib.json" --out "$model"
  x "$S" "$model" --store "$v2data/train" --split val --out "$run/val.jsonl"
  # the conversation operating point tau_c (§7, reported, not stored)
  x "$S" "$model" --store "$store" --split val --trail-ms 200 --out "$run/conv-val.jsonl"
  done_ calibrate
}

wasm_pkg() {
  x env RUSTFLAGS="-C target-feature=+simd128" CARGO_TARGET_DIR="$CARGO_TARGET_DIR/sky-web-node" \
    wasm-pack build crates/sky-web --target nodejs --release --out-dir "$PWD/$1" --no-typescript
}

latency() {
  need calibrate
  v2_file
  local lat=$run/latency pkg=$run/latency/pkg
  x mkdir -p "$lat"
  floor 1
  wasm_pkg "$pkg"
  if [ "$dry" = 1 ]; then
    x "$S" "$model" --bench "$iters"
    x node crates/sky-web/bench/turn-bench.mjs --pkg "$pkg" --iters "$iters" "$model" "$v2_model"
  else
    nice -n 19 "$S" "$model" --bench "$iters" | tee "$lat/native.jsonl"
    nice -n 19 "$S" "$v2_model" --bench "$iters" | tee "$lat/native-v2.jsonl"
    nice -n 19 node crates/sky-web/bench/turn-bench.mjs --pkg "$pkg" --iters "$iters" "$model" "$v2_model" | tee "$lat/wasm-both.jsonl"
    head -1 "$lat/wasm-both.jsonl" >"$lat/wasm.jsonl"
  fi
  gate_py latency --native "$lat/native.jsonl" --wasm "$lat/wasm.jsonl" --model "$model" --out "$lat/latency.json"
  done_ latency
}

wasm() {
  need latency
  local p=$run/parity pkg=$run/latency/pkg
  x mkdir -p "$p"
  # 20 Smart Turn test clips and 20 conversation test records through the WAV path natively, then in WASM
  # (no label read)
  if [ -n "$smoke" ]; then
    note "SMOKE: no WAVs of the fixture; parity on nothing"
    done_ wasm
    return
  fi
  x "$S" "$model" --records "$ear/jev/turn_end.jsonl" --split "$eval_split" --max 20 --out "$p/native.jsonl"
  x "$S" "$model" --records "$conv/parity/records.jsonl" --split test --max 20 --out "$p/native-conv.jsonl"
  # parity-node.mjs exits 1 both when it crashes and when a complete measurement misses |dp| <= 1e-4 or a
  # decision, so its status (kept through tee by pipefail) is reported and the files decide (amendment 3):
  # a complete measurement, passed or not, goes to G3; anything else stops here, without the marker
  local st=0 st_conv=0
  if [ "$dry" = 1 ]; then
    x node crates/sky-web/bench/parity-node.mjs --turn --pkg "$pkg" "$model" "$p/native.jsonl"
    x node crates/sky-web/bench/parity-node.mjs --turn --pkg "$pkg" "$model" "$p/native-conv.jsonl"
    x python3 -I scripts/turn3-guard.py parity --run "$run"
  else
    rm -f "$p/parity.json" "$p/parity-conv.json"
    nice -n 19 node crates/sky-web/bench/parity-node.mjs --turn --pkg "$pkg" "$model" "$p/native.jsonl" | tee "$p/parity.json" || st=$?
    nice -n 19 node crates/sky-web/bench/parity-node.mjs --turn --pkg "$pkg" "$model" "$p/native-conv.jsonl" | tee "$p/parity-conv.json" || st_conv=$?
    note "parity-node.mjs exit status: $st (Smart Turn clips), $st_conv (conversation records)"
    python3 -I scripts/turn3-guard.py parity --run "$run" ||
      die "== technical failure: the WASM parity was not measured in full ($p); fix it and run the wasm stage again, no test is opened (amendment 3)" 3
  fi
  done_ wasm
}

eval_test() {
  need wasm
  if [ -n "$smoke" ]; then
    note "SMOKE: the validation clips in place of the test clips"
    x "$S" "$model" --store "$v2data/train" --split val --out "$run/test-probs.jsonl"
    done_ eval
    return
  fi
  if [ "$dry" = 1 ]; then
    x python3 -I scripts/turn3-guard.py open --run "$run" --model "$model"
  else
    [ -f "$run/TEST-OPEN" ] || python3 -I scripts/turn3-guard.py open --run "$run" --model "$model" ||
      die "the tests cannot be opened (scripts/turn3-guard.py)" 3
  fi
  test_guard eval
  v2_file
  x mkdir -p "$run/v2"
  x "$S" "$model" --records "$ear/jev/turn_end.jsonl" --split test --out "$run/test-probs.jsonl"
  x "$S" "$model" --store "$v2data/test" --ext "$data/st-ext-test" --trail-ms 200,600 --out "$run/test-ext.jsonl"
  x "$S" "$v2_model" --store "$v2data/test" --ext "$data/st-ext-test" --trail-ms 200,600 --out "$run/v2/test-ext.jsonl"
  done_ eval
}

eotbench() {
  need eval
  if [ -n "$smoke" ]; then
    note "SMOKE: no eot-bench"
    done_ eotbench
    return
  fi
  test_guard eotbench
  v2_file
  x "$S" "$model" --store "$spans" --trail-ms "0,$asks" --out "$run/eotbench-probs.jsonl"
  x "$S" "$v2_model" --store "$spans" --trail-ms "$asks" --out "$run/v2/eotbench-probs.jsonl"
  done_ eotbench
}

conv_test() {
  need eotbench
  local split=test
  [ -n "$smoke" ] && split="val"
  [ -n "$smoke" ] || test_guard conv-test
  v2_file
  x mkdir -p "$run/v2"
  x "$S" "$model" --store "$store" --split "$split" --trail-ms "$asks" --out "$run/conv-test.jsonl"
  x "$S" "$v2_model" --store "$store" --split "$split" --trail-ms "$asks" --out "$run/v2/conv-test.jsonl"
  done_ conv-test
}

gate() {
  need conv-test
  [ -n "$smoke" ] || test_guard gate
  local v2tau extra=()
  if [ "$dry" = 1 ]; then v2tau="<v2's tau85>"; else v2tau=$(python3 -I -c 'import json, sys; print(json.load(open(sys.argv[1]))["tau85"])' "$v2_calib"); fi
  [ -n "$smoke" ] && extra=(--smoke --labels-store "$v2data/train")
  gate_py gate3 --run "$run" --test-probs "$run/test-probs.jsonl" --labels "$ear/clips/smart-turn.jsonl" \
    --test-ext "$run/test-ext.jsonl" --v2-test-ext "$run/v2/test-ext.jsonl" --spans "$spans" \
    --eot-v3 "$run/eotbench-probs.jsonl" --eot-v2 "$run/v2/eotbench-probs.jsonl" --conv-store "$store" \
    --conv-v3 "$run/conv-test.jsonl" --conv-v2 "$run/v2/conv-test.jsonl" --conv-val "$run/conv-val.jsonl" \
    --sessions "$conv" --calib "$run/calib.json" --latency "$run/latency/latency.json" \
    --parity "$run/parity/parity.json" --parity-conv "$run/parity/parity-conv.json" --model "$model" \
    --v2-model "$v2_model" --v2-tau85 "$v2tau" --harness3 "$run/harness3/check.json" --pilot "$pilot_dir/choice.json" \
    --crosscheck "$run/data/CROSSCHECK.json" --bleed "$run/data/bleed.json" --conv-summary "$run/data/CONV.json" \
    --out "$run/GATE.md" ${extra[@]+"${extra[@]}"}
  done_ gate
}

run_stages=("$@")
[ ${#run_stages[@]} -gt 0 ] || die "give a stage: ${stages[*]} (or labelcheck); they are launched one by one" 2
for s in "${run_stages[@]}"; do
  case " ${stages[*]} labelcheck " in
    *" $s "*) ;;
    *) die "unknown stage $s (${stages[*]} labelcheck)" 2 ;;
  esac
done
[ "$dry" = 1 ] && note "DRY_RUN: nothing is built, downloaded, trained, read or written"
build
for s in "${run_stages[@]}"; do
  echo "== stage $s"
  if [ "$s" != labelcheck ] && [ "$s" != conv-data ] && [ "$dry" != 1 ] && [ -f "$markers/$s.done" ]; then
    case $s in
      eval | eotbench | conv-test | gate) die "$markers/$s.done exists: a test is read once (§13)" 3 ;;
      *) echo "== $s: done ($markers/$s.done kept)"; continue ;;
    esac
  fi
  case "$s" in
    conv-data) conv_data ;;
    st-extend) st_extend ;;
    eotbench-data) eotbench_data ;;
    teacher-conv) teacher_conv ;;
    crosscheck) crosscheck ;;
    bleed) bleed ;;
    labelcheck) labelcheck ;;
    harness3) harness3 ;;
    smoke) smoke_ ;;
    pilot) pilot ;;
    train) train ;;
    export) export_model ;;
    calibrate) calibrate ;;
    latency) latency ;;
    wasm) wasm ;;
    eval) eval_test ;;
    eotbench) eotbench ;;
    conv-test) conv_test ;;
    gate) gate ;;
  esac
done
