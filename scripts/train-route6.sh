#!/usr/bin/env bash
# Train s6-route v6's model part (docs/s6-route6-prereg.md, section 5): v5's run
# (scripts/train-route5.sh, itself v4's) with v6's names, on the records of build6/ (v5's build,
# its When2Call group rebuilt by amendment 1), and the blind readings of 4.3. Validation only: no
# test or fresh file is read here. Folders runs/s6-route6*, models/s6-route6.
#
# What differs from scripts/train-route5.sh:
#   - the data: B6 (build6/) in place of B5; the guard reads runs/s6-route6/inputs.ok (the hashes
#     of the amendment of 4.8) and runs/s6-route6/placeholder.json in place of checks.passed;
#   - at every kept checkpoint, besides v5's readings: the model part in B0 and B1 on v5_tools
#     and v5_tools_masked (sky-eval on <val>/<file>.blind.model.jsonl and
#     <file>.blind-bare.model.jsonl) and the pair in B0 (sky-route fuse on <file>.blind.jsonl: the
#     embedding part reads the placeholder), read by scripts/s6-route6-pilot.py;
#   - the pilot rule adds rules 3 and 4 (the model part's blind accuracy on v5_tools and on
#     v5_tools_masked above 1/7 + 0.05 = 0.1929: abort);
#   - the fit is scripts/s6-route6-fit.py (beta and tau_ask6 on v5_tools).
# The recipe, the schedule, the budgets, the brake, the disk floor (MIN_FREE_GB=3) and v4's
# latency check (not run again) are v5's.
#
# Steps (default: window pilot main export, in order): as scripts/train-route5.sh, on v6's names.
#
#   ROUTE_DATA=../skycmd-data/s6-route   B6=$ROUTE_DATA/build6   MIN_FREE_GB=3
#   PILOT_MIN=20  MAIN_MIN=120  TIME_SCALE=1   SKY_BRAKE_FILE (required)   POTION_BAR (the
#   amendment of 4.5: potion-cos on v5_tools - 0.15)   EMBED   RESUME=1   SMOKE=1   DRY_RUN=1
# For a later version over this one (scripts/train-route7.sh): ROUTE_TAG (default s6-route6: the
# folders runs/<tag>, runs/<tag>-pilot, models/<tag>, the name <tag>-nano and the guard's files
# runs/<tag>/inputs.ok and runs/<tag>/placeholder.json).
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
tag=${ROUTE_TAG:-s6-route6}
name=$tag-nano
data=${ROUTE_DATA:-../skycmd-data/s6-route}
b6=${B6:-$data/build6}
min_free=${MIN_FREE_GB:-3}
embed=${EMBED:-site/public/models/s6-route-embed.gguf}
recipe_json=runs/recipe/recipe.json
smoke=${SMOKE:-}
if [ -n "$smoke" ]; then
  SKY_OUT=${SKY_OUT:-runs/${tag#s6-}-smoke/}
  PILOT_MIN=${PILOT_MIN:-0.75}
  MAIN_MIN=${MAIN_MIN:-1}
fi
out=${SKY_OUT:-}
run=${out}runs/$tag
pilot_dir=${out}runs/$tag-pilot
m=${out}models/$tag
model=$m/skycmd-$name-ternary.gguf
records=$b6/records/tools,$b6/records/intents,$b6/records/when2call
weights=tools=0.60,intents=0.25,when2call=0.15
val=$b6/val
# micro: the graph of the published model part (3.1)
width=256 batch=64 tok=data/tok/tok-s6-4096.json init_src=runs/s6-micro/pretrain/init want_x=0.5
if [ -n "$smoke" ]; then
  case "$run" in
    "runs/$tag" | "./runs/$tag" | "$PWD/runs/$tag")
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
      echo "$((kb / 1024)) MB free on the volume of $p, below the floor of $min_free GB (docs/s6-route6-prereg.md 4.7): a technical stop, resumed once the space is there" >&2
      exit 4
    fi
  done
}

# the refusals of the header
guard() {
  note "require runs/$tag/inputs.ok, runs/$tag/placeholder.json, runs/s6-route4/latency-check.json option (i) (3.4: kept from v4), no other training"
  [ "$dry" = 1 ] && return
  [ -f "runs/$tag/inputs.ok" ] || { echo "no runs/$tag/inputs.ok: the data script's stage inputs-ok, after the amendment of 4.8" >&2; exit 1; }
  [ -f "runs/$tag/placeholder.json" ] || { echo "no runs/$tag/placeholder.json: the data script's stage placeholder" >&2; exit 1; }
  [ -f "$val/v5_tools.blind.model.jsonl" ] || { echo "no blind copies of v5_tools: scripts/build-route6-data.sh blind-val" >&2; exit 1; }
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
  eval "$(python3 -I scripts/s6-adopt.py env s6-micro | grep -v '^CONFIG=')"
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
    [ "$dry" = 1 ] && { echo "<the head window of $run/window.json>"; return; }
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
json.dump({"about": "s6-route v6 (and later versions over it): the head window of docs/s6-route6-prereg.md 3 (v4 3.3: v2's rule, amendment 2)",
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
  # the blind readings (4.3): the model part in B0 and B1, the pair in B0 (no blind slate is trained on)
  for f in v5_tools v5_tools_masked; do
    x "$E" "$dir/full-depth.gguf" --records "$val/$f.blind.model.jsonl" --split val --max 100000 --exit-threshold 1.0 \
      --dump-probs "$dir/$f.blind.dump.jsonl" --json "$dir/$f.blind.eval.json"
    x "$E" "$dir/full-depth.gguf" --records "$val/$f.blind-bare.model.jsonl" --split val --max 100000 --exit-threshold 1.0 \
      --dump-probs "$dir/$f.blind-bare.dump.jsonl" --json "$dir/$f.blind-bare.eval.json"
    x "$R" fuse --model "$dir/full-depth.gguf" --embed "$embed" --slates "$val/$f.blind.jsonl" --out "$dir/pair_$f.blind.jsonl"
  done
  x python3 -I scripts/s6-route6-pilot.py read --ckpt "$dir" --val "$val"
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
  x python3 -I scripts/s6-route6-pilot.py summary --pilot "$pilot_dir/read" --bar "${POTION_BAR:-0}" \
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
  echo "== $model: tau_ask $tau; next: scripts/s6-route6-fit.py (beta and tau_ask6 on v5_tools) and the embedding file that stores them"
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
