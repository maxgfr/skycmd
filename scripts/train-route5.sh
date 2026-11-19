#!/usr/bin/env bash
# Line closed on 18/11/2026: v5 stopped at its data checks, before any training, so this script
# never ran its training steps; kept because docs/s6-route5-prereg.md (section 5) cites it.
# Train s6-route v5's model part (docs/s6-route5-prereg.md, section 5): as v4 (scripts/train-route4.sh,
# docs/s6-route4-prereg.md 5) with v5's names: s6-micro's graph from its pretraining init, on the
# data of scripts/build-route5-data.sh, rendered by D4 (option (i), kept from v4's latency check,
# 3.4). Validation only: no test or fresh file is read here. Folders runs/s6-route5*,
# models/s6-route5; nothing of v2's, v3's or v4's folders is written.
#
# What differs from scripts/train-route4.sh: the names (v5_tools in place of v4_tools, B5, the
# folders); the disk floor of 4.7 (MIN_FREE_GB=3 free on the volumes of the data folder and of
# this checkout before every step); the pilot's readings add v4_tools as a reported reading
# (scripts/s6-route5-pilot.py); the rule of the pilot is read on v5_tools and v5_tools_masked_rw.
# The recipe, the schedule, the budgets, the brake and the latency check (runs/s6-route4, not run
# again) are v4's.
#
# Steps (default: window pilot main export, in order):
#   window   the head window of 3.3 (v2's rule, amendment 2): sky-train s2-truncation on the
#            training and validation records at 160, 256, 320 tokens; the smallest at which at
#            most 5% of the records have an option shrunk, else the largest; runs/s6-route5/window.json
#   pilot    one 20-minute pilot, micro from runs/s6-micro/pretrain/init, QAT after minute 2, a
#            checkpoint kept every 5 minutes (sky-train s2-decide --keep-checkpoints); at every
#            kept checkpoint (exported ternary, full depth, after its own calibration on
#            v5_tools): the model part on v5_tools, v5_tools_masked, the two probe copies, the
#            renamed and natural validation records, v4_tools and v_tools (reported), and the pair
#            (the embedding file's beta, 0.81) on v5_tools and v5_tools_masked
#            (scripts/s6-route5-pilot.py read); then the pilot rule at the last checkpoint
#            (scripts/s6-route5-pilot.py summary, POTION_BAR = potion-cos on v5_tools - 0.15 as
#            the amendment of 4.5 wrote it): runs/s6-route5-pilot/summary.json
#   main     120 working minutes, QAT after minute 12, seed 0, a checkpoint kept every 10
#            minutes, each read as the pilot's (reported; none is chosen: `last` is the file);
#            refused when the pilot rule aborted
#   export   s2-calibrate on v5_tools, ternary export, tau_ask by v2's rule on v5_tools,
#            gguf-set: exit thresholds 1, skycmd-s2.route.tau_ask and
#            skycmd-s2.route.render = scores-in-options. The fit of beta and tau_ask5
#            (scripts/s6-route5-fit.py) was never written: v5 stopped before training.
#
# Refuses to start when the data checks have not passed ($B5/markers/checks.passed), when v4's
# latency check did not take option (i) (runs/s6-route4/latency-check.json), when SKY_BRAKE_FILE
# is unset, when another training runs (pgrep sky-train), or below the disk floor (exit 4).
#
#   ROUTE_DATA=../skycmd-data/s6-route   B5=$ROUTE_DATA/build5   MIN_FREE_GB=3
#   PILOT_MIN=20  MAIN_MIN=120  TIME_SCALE=1   working minutes (the brake's rests do not count)
#   SKY_DUTY=0.5  SKY_COOL=45:15  SKY_BRAKE_FILE (required)   the brake (docs/s6-gen-prereg.md)
#   SKY_MLX_CACHE_GB=2   POTION_BAR   the bar of the pilot rule (1), from the amendment of 4.5:
#                 potion-cos on v5_tools by v2's fitted values (scripts/build-route5-data.sh
#                 baselines-val, runs/s6-route5/baselines-val.json), minus 0.15
#   EMBED=site/public/models/s6-route-embed.gguf   the embedding part of the pilot's pair
#   RESUME=1   continue after a technical failure only (section 5)
#   SMOKE=1    the smoke run of section 5 (a pilot of 0.75 minute and a main run of 1, every step,
#              under SKY_OUT, default runs/route5-smoke/; its numbers are not looked at and its
#              files are deleted at the end unless KEEP_SMOKE=1)
#   DRY_RUN=1  print the commands instead of running them
set -euo pipefail
export DEVELOPER_DIR=${DEVELOPER_DIR:-/Applications/Xcode.app/Contents/Developer}
export CARGO_TARGET_DIR=${CARGO_TARGET_DIR:-$PWD/target/s3-train}
export SKY_DUTY=${SKY_DUTY:-0.5}
export SKY_COOL=${SKY_COOL-45:15}
export SKY_MLX_CACHE_GB=${SKY_MLX_CACHE_GB:-2}
if [ -z "${SKY_BRAKE_FILE:-}" ]; then
  echo "SKY_BRAKE_FILE is unset: the brake file of docs/s6-gen-prereg.md (scripts/auto-brake.sh writes it) is required" >&2
  exit 2
fi
export SKY_BRAKE_FILE
dry=${DRY_RUN:-0}
B=$CARGO_TARGET_DIR/release
T=$B/sky-train
E=$B/sky-eval
R=$B/sky-route
name=s6-route5-nano
data=${ROUTE_DATA:-../skycmd-data/s6-route}
b5=${B5:-$data/build5}
min_free=${MIN_FREE_GB:-3}
embed=${EMBED:-site/public/models/s6-route-embed.gguf}
recipe_json=runs/recipe/recipe.json
smoke=${SMOKE:-}
if [ -n "$smoke" ]; then
  SKY_OUT=${SKY_OUT:-runs/route5-smoke/}
  PILOT_MIN=${PILOT_MIN:-0.75}
  MAIN_MIN=${MAIN_MIN:-1}
fi
out=${SKY_OUT:-}
run=${out}runs/s6-route5
pilot_dir=${out}runs/s6-route5-pilot
m=${out}models/s6-route5
model=$m/skycmd-$name-ternary.gguf
records=$b5/records/tools,$b5/records/intents,$b5/records/when2call
weights=tools=0.60,intents=0.25,when2call=0.15
val=$b5/val
# micro: the graph of the published model part (3.1)
width=256 batch=64 tok=data/tok/tok-s6-4096.json init_src=runs/s6-micro/pretrain/init want_x=0.5
if [ -n "$smoke" ]; then
  case "$run" in
    runs/s6-route5 | ./runs/s6-route5 | "$PWD/runs/s6-route5")
      echo "SMOKE=1 needs a prefix that is not the real run's (SKY_OUT=$SKY_OUT)" >&2
      exit 2
      ;;
  esac
  echo "== SMOKE: everything under $SKY_OUT; nothing of it is a result"
fi

note() { [ "$dry" != 1 ] || echo "# $*"; }
x() {
  if [ "$dry" = 1 ]; then
    printf '+'
    printf ' %q' "$@"
    printf '\n'
  else
    "$@"
  fi
}
logged() {
  local log=$1
  shift
  if [ "$dry" = 1 ]; then
    x "$@"
  else
    mkdir -p "$(dirname "$log")"
    "$@" 2>&1 | tee -a "$log"
  fi
}

# 4.7: at least MIN_FREE_GB free on the volumes of the data folder and of this checkout (the
# checkpoints and the readings go to runs/ and models/) before every step
disk_floor() {
  note "require $min_free GB free on the volumes of $data and of $PWD (4.7)"
  [ "$dry" = 1 ] && return
  local p kb
  for p in "$data" "$PWD"; do
    kb=$(df -Pk "$p" | awk 'NR == 2 { print $4 }')
    if [ "$kb" -lt $((min_free * 1024 * 1024)) ]; then
      echo "$((kb / 1024)) MB free on the volume of $p, below the floor of $min_free GB (docs/s6-route5-prereg.md 4.7): a technical stop, resumed once the space is there" >&2
      exit 4
    fi
  done
}

# the refusals of the header
guard() {
  note "require $b5/markers/checks.passed, runs/s6-route4/latency-check.json option (i) (3.4: kept from v4), no other training"
  [ "$dry" = 1 ] && return
  [ -f "$b5/markers/checks.passed" ] || { echo "the data checks have not passed ($b5/markers/checks.passed): scripts/build-route5-data.sh checks" >&2; exit 1; }
  python3 -I -c '
import json, sys
c = json.load(open(sys.argv[1]))
if c.get("option") != "i":
    sys.exit("the latency check of v4 3.4 did not take option (i): " + str(c.get("option")))' runs/s6-route4/latency-check.json
  if pgrep -f 'sky-train' >/dev/null; then
    echo "another training runs (pgrep sky-train): one training at a time" >&2
    exit 1
  fi
}

# The recipe, with no hand choice (scripts/s6-adopt.py env), as v2.
OPT='' LR_SCALE=1 DECIDE=''
if [ -f "$recipe_json" ]; then
  # on its own line, so that a failure of s6-adopt.py stops the script (set -e, pipefail)
  env_sh=$(python3 -I scripts/s6-adopt.py env s6-micro | grep -v '^CONFIG=')
  eval "$env_sh"
elif [ "$dry" = 1 ]; then
  echo "# no $recipe_json: the commands below show the default options"
  OPT=--no-muon LR_SCALE=2.0
else
  echo "no $recipe_json: the recipe is adopted first (scripts/train-all.sh adopt)" >&2
  exit 1
fi
read -r -a opt <<<"$OPT"
read -r -a dec <<<"$DECIDE"
case " $OPT " in
  *" --muon-lr "* | *" --adam-lr "*)
    echo "the recipe's OPT carries a rate: LR_SCALE sets it" >&2
    exit 2
    ;;
esac
[[ " $DECIDE " == *" --ema "* ]] && { echo "the recipe adopts EMA; section 5 keeps checkpoint last" >&2; exit 2; }
lr_flag=--muon-lr
[[ " $OPT " == *" --no-muon "* ]] && lr_flag=--adam-lr
tmin() { LC_ALL=C awk -v a="$1" -v b="${TIME_SCALE:-1}" 'BEGIN { printf "%g", a * b }'; }
pilot_min=$(tmin "${PILOT_MIN:-20}")
main_min=$(tmin "${MAIN_MIN:-120}")
resume=()
[ -n "${RESUME:-}" ] && resume=(--resume)
rate() {
  local x=$want_x
  if [ -f "runs/s6-micro-lr/summary.json" ]; then
    x=$(python3 -I -c 'import json,sys; print(json.load(open(sys.argv[1]))["chosen"])' "runs/s6-micro-lr/summary.json")
  fi
  [ "$x" = "$want_x" ] || { echo "runs/s6-micro-lr/summary.json chose x$x; v2's recipe says x$want_x" >&2; exit 2; }
  LC_ALL=C awk -v s="$LR_SCALE" -v d="$width" -v x="$x" 'BEGIN { printf "%.3g", 1e-3 * s * sqrt(96 / d) * x }'
}

window_of() {
  if [ ! -f "$run/window.json" ]; then
    [ "$dry" = 1 ] && { echo "<the head window of runs/s6-route5/window.json>"; return; }
    echo "no $run/window.json: the window step first" >&2
    exit 1
  fi
  python3 -I -c 'import json,sys; print(json.load(open(sys.argv[1]))["window"])' "$run/window.json"
}

window() {
  local w cfg
  x mkdir -p "$run/truncation"
  for w in 160 256 320; do
    cfg=$run/truncation/s6-micro-$w.json
    x python3 -I -c 'import json, sys; c = json.load(open(sys.argv[1])); c["max_head_tokens"] = int(sys.argv[3]); json.dump(c, open(sys.argv[2], "w"), indent=2)' \
      runs/recipe/configs/s6-micro.json "$cfg" "$w"
    x "$T" s2-truncation --config "$cfg" --tokenizer "$tok" --records "$records" --per-source 1000000000 \
      --json "$run/truncation/micro-$w.json"
  done
  note "the rule of 3.3 on the three reports: $run/window.json"
  [ "$dry" = 1 ] && return
  python3 -I - "$run" <<'EOF'
import json, os, sys
run = sys.argv[1]
r = {w: json.load(open(os.path.join(run, "truncation", f"micro-{w}.json")))["all"]["head"] for w in (160, 256, 320)}
ok = [w for w in sorted(r) if r[w] <= 0.05]
w = ok[0] if ok else max(r)
json.dump({"about": "s6-route v5: the head window of docs/s6-route5-prereg.md 3 (v4 3.3: v2's rule, amendment 2)",
           "head share shrunk": r, "window": w}, open(os.path.join(run, "window.json"), "w"), indent=1)
print(f"head window {w}: {r}")
EOF
}

# prepare <dir>: the config (the recipe's micro graph, the head window, this name) and the init,
# copied and compared by sha256 before and after
prepare() {
  local dir=$1 w
  w=$(window_of)
  note "write $dir/config.json (runs/recipe/configs/s6-micro.json, named $name, max_head_tokens $w); copy $init_src into $dir/init with its sha256"
  [ "$dry" = 1 ] && return
  mkdir -p "$dir"
  python3 -I - runs/recipe/configs/s6-micro.json "$dir/config.json" "$name" "$w" <<'EOF'
import json, sys
src, out, name, w = sys.argv[1:]
c = json.load(open(src))
if (c["vocab_size"], c["hidden_size"], c["state_layers"], c["state_loops"]) != (4096, 256, 2, 3):
    sys.exit(f"{src} is not s6-micro's graph")
c["name"] = name
c["max_head_tokens"] = int(w)
json.dump(c, open(out, "w"), indent=2)
EOF
  if [ ! -f "$dir/init.sha256" ]; then
    local before
    before=$(shasum -a 256 "$init_src/model.safetensors" | cut -d' ' -f1)
    rm -rf "$dir/init"
    cp -R "$init_src" "$dir/init"
    echo "$before  $init_src/model.safetensors (before)" >"$dir/init.sha256"
  fi
  local after
  after=$(shasum -a 256 "$dir/init/model.safetensors" | cut -d' ' -f1)
  grep -q "^$after " "$dir/init.sha256" || { echo "$dir/init is not $init_src" >&2; exit 1; }
}

# decide <dir> <minutes> <checkpoint minutes>
decide() {
  local dir=$1 minutes=$2 every=$3 lr
  lr=$(rate)
  note "sky-train s2-decide must offer --keep-checkpoints (docs/s6-route4-prereg.md 14, item 5; reused by v5)"
  if [ "$dry" != 1 ] && ! "$T" s2-decide --help | grep -q -- '--keep-checkpoints'; then
    echo "this sky-train has no s2-decide --keep-checkpoints (prereg 14, item 5): the pilot cannot keep its checkpoints" >&2
    exit 1
  fi
  [ -n "${RESUME:-}" ] || x rm -rf "$dir/decide"
  logged "$dir/decide.log" "$T" s2-decide --config "$dir/config.json" --init "$dir/init" --tokenizer "$tok" \
    --records "$records" --weights "$weights" --batch "$batch" --minutes "$minutes" --qat-frac 0.9 \
    --loop-dropout 0.25 --length-pool 16 --pad-to 32 --step-mem-gb 3 --no-compile --seed 0 \
    --checkpoint-minutes "$every" --keep-checkpoints \
    --out "$dir/decide" "$lr_flag" "$lr" ${opt[@]+"${opt[@]}"} ${dec[@]+"${dec[@]}"} ${resume[@]+"${resume[@]}"}
}

# read_ckpt <checkpoint dir> <reading dir>: calibrated on v5_tools, ternary, full depth, D4's key;
# the model part on every validation reading and the pair on v5_tools and its masked copy
read_ckpt() {
  local ck=$1 dir=$2 f
  x mkdir -p "$dir"
  x "$T" s2-calibrate --run "$ck" --records "$val/v5_tools.model.jsonl"
  x "$T" s2-export --run "$ck" --format ternary --out "$dir/model.gguf"
  x "$B/gguf-set" "$dir/model.gguf" "$dir/full-depth.gguf" skycmd-s2.exit_threshold.choice=1 skycmd-s2.exit_threshold.score=1 \
    skycmd-s2.exit_threshold.noul=1 skycmd-s2.route.render=str:scores-in-options
  for f in v5_tools v5_tools_masked v5_tools_rw v5_tools_masked_rw r_tools_renamed r_tools_natural v4_tools v_tools; do
    x "$E" "$dir/full-depth.gguf" --records "$val/$f.model.jsonl" --split val --max 100000 --exit-threshold 1.0 \
      --dump-probs "$dir/$f.dump.jsonl" --json "$dir/$f.eval.json"
  done
  for f in v5_tools v5_tools_masked; do
    x "$R" fuse --model "$dir/full-depth.gguf" --embed "$embed" --slates "$val/$f.jsonl" --out "$dir/pair_$f.jsonl"
  done
  x python3 -I scripts/s6-route5-pilot.py read --ckpt "$dir" --val "$val"
}

# read_all <run dir>: every kept checkpoint, then last
read_all() {
  local dir=$1 ck
  if [ "$dry" = 1 ]; then
    read_ckpt "$dir/decide/ckpt-<minutes>" "$dir/read/ckpt-<minutes>"
    read_ckpt "$dir/decide/last" "$dir/read/last"
    return
  fi
  for ck in "$dir"/decide/ckpt-*; do
    [ -d "$ck" ] && read_ckpt "$ck" "$dir/read/$(basename "$ck")"
  done
  read_ckpt "$dir/decide/last" "$dir/read/last"
}

pilot() {
  guard
  if [ -z "${POTION_BAR:-}" ] && [ -z "$smoke" ] && [ "$dry" != 1 ]; then
    echo "POTION_BAR is unset: the bar of the pilot rule (potion-cos on v5_tools - 0.15) is written in the amendment of 4.5 before the pilot" >&2
    exit 2
  fi
  echo "== pilot micro: ${pilot_min} working minutes, a checkpoint kept every 5"
  prepare "$pilot_dir"
  decide "$pilot_dir" "$pilot_min" 5
  read_all "$pilot_dir"
  x python3 -I scripts/s6-route5-pilot.py summary --pilot "$pilot_dir/read" --bar "${POTION_BAR:-0}" \
    --out "$pilot_dir/summary.json"
}

main_() {
  guard
  note "refuse when $pilot_dir/summary.json says abort (section 5; SMOKE excepted)"
  if [ "$dry" != 1 ]; then
    python3 -I -c '
import json, sys
s = json.load(open(sys.argv[1]))
if s["abort"] and not sys.argv[2]:
    sys.exit("the pilot rule aborted the main run: " + "; ".join(s["why"]))' "$pilot_dir/summary.json" "$smoke"
  fi
  echo "== main run: micro, ${main_min} working minutes, seed 0"
  prepare "$run"
  decide "$run" "$main_min" 10
  note "check that $init_src/model.safetensors still has the sha256 of $run/init.sha256"
  if [ "$dry" != 1 ]; then
    local now
    now=$(shasum -a 256 "$init_src/model.safetensors" | cut -d' ' -f1)
    grep -q "^$now " "$run/init.sha256" || { echo "$init_src changed during the run" >&2; exit 1; }
  fi
  read_all "$run"
}

export_() {
  x mkdir -p "$m"
  x "$T" s2-calibrate --run "$run/decide/last" --records "$val/v5_tools.model.jsonl"
  x "$T" s2-export --run "$run/decide/last" --format ternary --out "$model"
  x "$B/gguf-set" "$model" "$run/full-depth.gguf" skycmd-s2.exit_threshold.choice=1 skycmd-s2.exit_threshold.score=1 \
    skycmd-s2.exit_threshold.noul=1 skycmd-s2.route.render=str:scores-in-options
  x "$E" "$run/full-depth.gguf" --records "$val/v5_tools.model.jsonl" --split val --max 100000 --exit-threshold 1.0 \
    --dump-probs "$run/v5_tools.dump.jsonl" --json "$run/v5_tools.eval.json"
  local tau='<tau>'
  if [ "$dry" != 1 ]; then
    tau=$(python3 -I scripts/s6-route-gate.py tau --dump "$run/v5_tools.dump.jsonl" --slates "$val/v5_tools.jsonl" --out "$m/tau.json")
  fi
  x "$B/gguf-set" "$model" "$model" skycmd-s2.exit_threshold.choice=1 skycmd-s2.exit_threshold.score=1 \
    skycmd-s2.exit_threshold.noul=1 "skycmd-s2.route.tau_ask=$tau" skycmd-s2.route.render=str:scores-in-options
  echo "== $model: tau_ask $tau; the fit of beta and tau_ask5 (scripts/s6-route5-fit.py) was never written: v5 stopped before training"
}

steps=("$@")
[ ${#steps[@]} -eq 0 ] && steps=(window pilot main export)
for s in "${steps[@]}"; do
  case "$s" in
    window | pilot | main | export) ;;
    *) echo "unknown step $s (window, pilot, main, export)" >&2; exit 2 ;;
  esac
done
[ "$dry" = 1 ] || nice -n 19 cargo build -q --release -p sky-train -p sky-gguf -p sky-infer
for s in "${steps[@]}"; do
  disk_floor
  case "$s" in
    window) window ;;
    pilot) pilot ;;
    main) main_ ;;
    export) export_ ;;
  esac
done
if [ -n "$smoke" ] && [ "$dry" != 1 ] && [ -z "${KEEP_SMOKE:-}" ]; then
  echo "== SMOKE: removing $SKY_OUT"
  rm -rf "${SKY_OUT:?}"
fi
