#!/usr/bin/env bash
# Train s6-route-nano (docs/s6-nano2-prereg.md, section 5): the s6 decision graph trained on tool
# and intent routing only, from the pretraining of its size (no pretraining is run), on the data
# of scripts/build-route-data.sh. One main run, seed 0. No test file is read here: the
# evaluation is scripts/route-eval.sh test, after training, once the owner creates
# runs/s6-nano2/TEST-OPEN.
#
# Steps (default: window pilot main export report, in order):
#   window   the head window of each size (section 3): sky-train s2-truncation on the training
#            and validation records at 160, 256, 320 tokens, then BUILD.json
#            (scripts/build-route-data.sh truncation manifest); CPU only, the trainer reads
#            records. It could not run before the trainer was there; it runs first here
#   pilot    the two 20-minute pilots on the same data, nano from runs/s6-nano/pretrain/init and
#            micro from runs/s6-micro/pretrain/init, the full schedule compressed (QAT after a
#            10% float warm-up: minute 2); each calibrated on the validation slates and exported
#            ternary, then read at full depth on the validation slates (v_tools); the pilot rule
#            (scripts/s6-route-gate.py pilot, against potion-cos on the same slates):
#            micro if its accuracy is at least nano's + 0.04; abort when the chosen pilot is
#            below potion-cos - 0.15. Written to runs/s6-nano2-pilot/summary.json
#   main     the chosen size, 120 working minutes from its init (copied into runs/s6-nano2/ and
#            compared by sha256 before and after), QAT after minute 12; refused when the pilot
#            rule aborted
#   export   s2-calibrate on the validation slates, s2-export --format ternary, tau_ask (the
#            smallest top probability at which validation accuracy among answered requests is
#            >= 0.95, read at full depth on v_tools), gguf-set: exit thresholds 1 and
#            skycmd-s2.route.tau_ask
#   report   runs/s6-nano2/report.md: steps, working minutes, brake modes, pilots, tau_ask
#
# Unchanged from v1 (section 2): the graph (the recipe's config; the head window of section 3,
# from BUILD.json), the tokenizer, AdamW (the recipe's --no-muon), the rate LR_SCALE x
# sqrt(96 / d) x the size's pilot scale x 1e-3 (nano: 2.0 x sqrt(96/144) x 1 = 1.63e-3; micro:
# x 0.5), batch 128 (micro 64), loop dropout 0.25, --length-pool 16 --pad-to 32 --step-mem-gb 3
# --no-compile, checkpoint `last`. Weights tools=0.60, intents=0.25, when2call=0.15.
#
#   ROUTE_DATA=../skycmd-data/s6-route   BUILD=$ROUTE_DATA/build (records/, val/, BUILD.json)
#   PILOT_MIN=20  MAIN_MIN=120   working minutes (the brake's rests do not count)
#   TIME_SCALE=1  multiplies the budgets (scripts/speed-probe.sh; docs/s6-prereg.md, amendment 4)
#   SKY_DUTY=0.5  SKY_COOL=45:15  SKY_BRAKE_FILE=runs/brake   the brake (docs/s6-gen-prereg.md,
#                 amendments 5, 6 and 10; scripts/auto-brake.sh writes the file)
#   SKY_MLX_CACHE_GB=2   cap of MLX's buffer cache
#   RESUME=1      continue from the checkpoints of an interrupted run (only a technical failure
#                 justifies it, section 5)
#   SKY_OUT       a prefix for everything this script writes
#   SMOKE=1       the first contact with sky-train: pilots of 0.75 minute and a main run of 1,
#                 every step, everything under SKY_OUT (default runs/route-smoke/), never in
#                 runs/s6-nano2* or models/s6-nano2; a provisional build is accepted; its numbers
#                 are not looked at, and its files are deleted at the end (KEEP_SMOKE=1 keeps them)
#   DRY_RUN=1     print the commands instead of running them (nothing is built or trained)
set -euo pipefail
export DEVELOPER_DIR=${DEVELOPER_DIR:-/Applications/Xcode.app/Contents/Developer}
export CARGO_TARGET_DIR=${CARGO_TARGET_DIR:-$PWD/target/s3-train}
export SKY_DUTY=${SKY_DUTY:-0.5}
export SKY_COOL=${SKY_COOL-45:15}
export SKY_BRAKE_FILE=${SKY_BRAKE_FILE:-$PWD/runs/brake}
export SKY_MLX_CACHE_GB=${SKY_MLX_CACHE_GB:-2}
dry=${DRY_RUN:-0}
B=$CARGO_TARGET_DIR/release
T=$B/sky-train
E=$B/sky-eval
name=s6-route-nano
data=${ROUTE_DATA:-../skycmd-data/s6-route}
build=${BUILD:-$data/build}
export ROUTE_DATA=$data BUILD=$build
recipe_json=runs/recipe/recipe.json
smoke=${SMOKE:-}
if [ -n "$smoke" ]; then
  SKY_OUT=${SKY_OUT:-runs/route-smoke/}
  PILOT_MIN=${PILOT_MIN:-0.75}
  MAIN_MIN=${MAIN_MIN:-1}
fi
out=${SKY_OUT:-}
run=${out}runs/s6-nano2
pilot_dir=${out}runs/s6-nano2-pilot
m=${out}models/s6-nano2
model=$m/skycmd-$name-ternary.gguf
records=$build/records/tools,$build/records/intents,$build/records/when2call
weights=tools=0.60,intents=0.25,when2call=0.15
val=$build/val/v_tools.jsonl
# the model's copy of the validation slates: the same slates, each state with the BM25 line of
# its tool options (amendment 1; sky-data route render, in the build's slates step)
val_model=$build/val/v_tools.model.jsonl
potion_dump=$data/eval/val/potion-cos/v_tools.jsonl
if [ -n "$smoke" ]; then
  case "$run" in
    runs/s6-nano2 | ./runs/s6-nano2 | "$PWD/runs/s6-nano2")
      echo "SMOKE=1 needs a prefix that is not the real run's (SKY_OUT=$SKY_OUT)" >&2
      exit 2
      ;;
  esac
  echo "== SMOKE: everything under $SKY_OUT; nothing of it is a result"
fi

# The recipe, with no hand choice: CONFIG, OPT, LR_SCALE, DECIDE (scripts/s6-adopt.py env).
OPT='' LR_SCALE=1 DECIDE=''
if [ -f "$recipe_json" ]; then
  # on its own line, so that a failure of s6-adopt.py stops the script (set -e, pipefail)
  env_sh=$(python3 -I scripts/s6-adopt.py env s6-nano | grep -v '^CONFIG=')
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
[[ " $DECIDE " == *" --ema "* ]] && { echo "the recipe adopts EMA; section 2 keeps checkpoint last" >&2; exit 2; }
lr_flag=--muon-lr
[[ " $OPT " == *" --no-muon "* ]] && lr_flag=--adam-lr
tmin() { LC_ALL=C awk -v a="$1" -v b="${TIME_SCALE:-1}" 'BEGIN { printf "%g", a * b }'; }
pilot_min=$(tmin "${PILOT_MIN:-20}")
main_min=$(tmin "${MAIN_MIN:-120}")
resume=()
[ -n "${RESUME:-}" ] && resume=(--resume)

# size <nano|micro>: width, batch, tokenizer, init, pilot scale of the rate
size_of() {
  case "$1" in
    nano) width=144 batch=128 tok=data/tok/tok-s6-2048.json init_src=runs/s6-nano/pretrain/init want_x=1 ;;
    micro) width=256 batch=64 tok=data/tok/tok-s6-4096.json init_src=runs/s6-micro/pretrain/init want_x=0.5 ;;
    *) echo "unknown size $1" >&2; exit 2 ;;
  esac
}
# rate <size>: 1e-3 x LR_SCALE x sqrt(96 / d) x the size's pilot scale (runs/s6-<size>-lr)
rate() {
  size_of "$1"
  local x=$want_x
  if [ -f "runs/s6-$1-lr/summary.json" ]; then
    x=$(python3 -I -c 'import json,sys; print(json.load(open(sys.argv[1]))["chosen"])' "runs/s6-$1-lr/summary.json")
  fi
  if [ "$x" != "$want_x" ]; then
    echo "runs/s6-$1-lr/summary.json chose x$x; section 2 says x$want_x" >&2
    exit 2
  fi
  LC_ALL=C awk -v s="$LR_SCALE" -v d="$width" -v x="$x" 'BEGIN { printf "%.3g", 1e-3 * s * sqrt(96 / d) * x }'
}

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
note() { [ "$dry" != 1 ] || echo "# $*"; }

# The build: the files BUILD.json names, not provisional (unless SMOKE), and the head window of
# each size (section 3).
check_build() {
  note "check $build/BUILD.json: not provisional (SMOKE accepts one), every record file and v_tools as hashed there, a head window for the size"
  [ "$dry" = 1 ] && return
  python3 -I - "$build" "$smoke" <<'EOF'
import hashlib, json, os, sys
build, smoke = sys.argv[1:]
m = json.load(open(os.path.join(build, "BUILD.json")))
if m["provisional"] and not smoke:
    sys.exit("BUILD.json is provisional (the teacher's answers are missing): not trained on outside a smoke run")
for rel in [f"records/{g}/{s}.jsonl" for g in ("tools", "intents", "when2call") for s in ("train", "val")] + ["val/v_tools.jsonl", "val/v_tools.model.jsonl"]:
    h = hashlib.sha256(open(os.path.join(build, rel), "rb").read()).hexdigest()
    if h != m["outputs"][rel]["sha256"]:
        sys.exit(f"{build}/{rel} is not the file of BUILD.json")
EOF
}
window_of() {
  python3 -I -c '
import json, os, sys
path, size, dry = sys.argv[1:]
w = (json.load(open(path))["truncation"]["window"] or {}).get(size) if os.path.isfile(path) else None
if w is None:
    if dry == "1":
        print(f"<the head window of {size} in BUILD.json>")
        sys.exit(0)
    sys.exit(f"BUILD.json has no head window for {size}: the rule of section 3 gave none, or the truncation step did not run")
print(w)' "$build/BUILD.json" "$1" "$dry"
}

# prepare <size> <dir>: the config (the recipe's graph, the head window, this name) and the init
# copied into the run, with its sha256 before and after the copy
prepare() {
  local size=$1 dir=$2 w
  size_of "$size"
  w=$(window_of "$size")
  note "write $dir/config.json: runs/recipe/configs/s6-$size.json, named $name, max_head_tokens $w"
  note "copy $init_src into $dir/init; sha256 of its model.safetensors before and after the copy into $dir/init.sha256"
  [ "$dry" = 1 ] && return
  mkdir -p "$dir"
  python3 -I - "runs/recipe/configs/s6-$size.json" "$dir/config.json" "$name" "$w" "$size" <<'EOF'
import json, sys
src, out, name, w, size = sys.argv[1:]
c = json.load(open(src))
graph = (c["vocab_size"], c["hidden_size"], c["state_layers"], c["state_loops"])
want = {"nano": (2048, 144, 2, 2), "micro": (4096, 256, 2, 3)}[size]
if graph != want:
    sys.exit(f"{src} is not the graph of s6-{size}: {graph}")
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

# decide <size> <dir> <minutes>
decide() {
  local size=$1 dir=$2 minutes=$3 lr
  lr=$(rate "$size")
  size_of "$size"
  [ -n "${RESUME:-}" ] || x rm -rf "$dir/decide"
  logged "$dir/decide.log" "$T" s2-decide --config "$dir/config.json" --init "$dir/init" --tokenizer "$tok" \
    --records "$records" --weights "$weights" --batch "$batch" --minutes "$minutes" --qat-frac 0.9 \
    --loop-dropout 0.25 --length-pool 16 --pad-to 32 --step-mem-gb 3 --no-compile --seed 0 \
    --out "$dir/decide" "$lr_flag" "$lr" ${opt[@]+"${opt[@]}"} ${dec[@]+"${dec[@]}"} ${resume[@]+"${resume[@]}"}
}

# calibrated <dir> <gguf>: s2-calibrate on the validation slates, ternary export, and the copy
# at full depth read on v_tools (<dir>/v_tools.dump.jsonl), all on the model's copy of them
calibrated() {
  local dir=$1 g=$2
  x mkdir -p "$(dirname "$g")"
  x "$T" s2-calibrate --run "$dir/decide/last" --records "$val_model"
  x "$T" s2-export --run "$dir/decide/last" --format ternary --out "$g"
  x "$B/gguf-set" "$g" "$dir/full-depth.gguf" skycmd-s2.exit_threshold.choice=1 skycmd-s2.exit_threshold.score=1 \
    skycmd-s2.exit_threshold.noul=1
  x "$E" "$dir/full-depth.gguf" --records "$val_model" --split val --max 100000 --exit-threshold 1.0 \
    --dump-probs "$dir/v_tools.dump.jsonl" --json "$dir/v_tools.eval.json"
}

window() {
  note "the head window of each size: s2-truncation at 160, 256, 320, then BUILD.json"
  x env SKY_TRAIN="$T" scripts/build-route-data.sh truncation manifest
}

pilot() {
  check_build
  local size
  for size in nano micro; do
    echo "== pilot $size: ${pilot_min} working minutes"
    prepare "$size" "$pilot_dir/$size"
    decide "$size" "$pilot_dir/$size" "$pilot_min"
    calibrated "$pilot_dir/$size" "$pilot_dir/$size/model.gguf"
  done
  note "the potion-cos dump on the same validation slates: $potion_dump (scripts/route-eval.sh val fit baselines)"
  if [ "$dry" != 1 ] && [ ! -f "$potion_dump" ]; then
    echo "no $potion_dump: run scripts/route-eval.sh val env fit baselines first (CPU)" >&2
    exit 1
  fi
  x python3 -I scripts/s6-route-gate.py pilot --nano "$pilot_dir/nano/v_tools.dump.jsonl" \
    --micro "$pilot_dir/micro/v_tools.dump.jsonl" --potion "$potion_dump" --slates "$val" --out "$pilot_dir/summary.json"
}

chosen() {
  if [ "$dry" = 1 ] && [ ! -f "$pilot_dir/summary.json" ]; then
    echo nano
    return
  fi
  python3 -I -c '
import json, sys
s = json.load(open(sys.argv[1]))
if s["abort"] and not sys.argv[2]:
    sys.exit("the pilot rule aborted the main run (runs/s6-nano2-pilot/summary.json): section 11 applies")
print(s["size"])' "$pilot_dir/summary.json" "$smoke"
}

main_() {
  check_build
  local size
  size=$(chosen)
  echo "== main run: $size, ${main_min} working minutes, seed 0"
  prepare "$size" "$run"
  decide "$size" "$run" "$main_min"
  note "check that $init_src/model.safetensors still has the sha256 of $run/init.sha256"
  if [ "$dry" != 1 ]; then
    size_of "$size"
    local now
    now=$(shasum -a 256 "$init_src/model.safetensors" | cut -d' ' -f1)
    grep -q "^$now " "$run/init.sha256" || { echo "$init_src changed during the run" >&2; exit 1; }
  fi
}

export_() {
  calibrated "$run" "$model"
  note "tau_ask from $run/v_tools.dump.jsonl (scripts/s6-route-gate.py tau), then gguf-set: exit thresholds 1 and skycmd-s2.route.tau_ask"
  if [ "$dry" = 1 ]; then
    x "$B/gguf-set" "$model" "$model" skycmd-s2.exit_threshold.choice=1 skycmd-s2.exit_threshold.score=1 \
      skycmd-s2.exit_threshold.noul=1 "skycmd-s2.route.tau_ask=<tau>"
    return
  fi
  local tau
  tau=$(python3 -I scripts/s6-route-gate.py tau --dump "$run/v_tools.dump.jsonl" --slates "$val" --out "$m/tau.json")
  "$B/gguf-set" "$model" "$model" skycmd-s2.exit_threshold.choice=1 skycmd-s2.exit_threshold.score=1 \
    skycmd-s2.exit_threshold.noul=1 "skycmd-s2.route.tau_ask=$tau"
  echo "== $model: tau_ask $tau stored; the file is final (the test is opened by hand: runs/s6-nano2/TEST-OPEN)"
}

report() {
  note "write $run/report.md: steps, working minutes, brake modes, the pilots and their rule, tau_ask"
  [ "$dry" = 1 ] && return
  python3 -I - "$run" "$pilot_dir" "$m/tau.json" "$SKY_BRAKE_FILE" "$SKY_DUTY ${SKY_COOL:-no long rests}" "$smoke" <<'EOF'
import json, os, re, sys
run, pilot, tau, brake_file, brake_env, smoke = sys.argv[1:]
out = ["# s6-route-nano: the run", ""]
if smoke:
    out += ["**SMOKE RUN.** No number below is a result.", ""]
out += ["Pre-registration: `docs/s6-nano2-prereg.md`, section 5. One run, seed 0; checkpoint `last`.", "",
        "| phase | steps | working minutes | budget |", "|---|---:|---:|---:|"]
for phase, d in (("pilot nano", f"{pilot}/nano"), ("pilot micro", f"{pilot}/micro"), ("main", run)):
    p = f"{d}/decide/last/state.json"
    if os.path.exists(p):
        s = json.load(open(p))
        out.append(f"| {phase} | {s['step']} | {s['elapsed_secs'] / 60:.2f} | {s['budget_secs'] / 60:g} |")
out += ["", f"Brake at start: `SKY_DUTY SKY_COOL` = `{brake_env}`, file `{brake_file}`.", ""]
seen = {}
for d in (f"{pilot}/nano", f"{pilot}/micro", run):
    p = f"{d}/decide.log"
    if os.path.exists(p):
        for line in open(p, errors="replace"):
            if line.startswith("brake: ") or line.startswith("cooling down"):
                k = (d, re.sub(r"for \d+ s", "for N s", line.strip()))
                seen[k] = seen.get(k, 0) + 1
out += [f"- `{d}`: {l} ({n} times)" for (d, l), n in seen.items()] or ["- no change of mode and no long rest was logged"]
if os.path.exists(f"{pilot}/summary.json"):
    s = json.load(open(f"{pilot}/summary.json"))
    out += ["", "## Pilots (validation slates, ternary, full depth)", "", "| nano | micro | potion-cos | abort bar | size | abort |",
            "|---:|---:|---:|---:|---|---|", f"| {s['nano']} | {s['micro']} | {s['potion_cos']} | {s['abort_bar']} | {s['size']} | {s['abort']} |"]
if os.path.exists(tau):
    t = json.load(open(tau))
    out += ["", f"tau_ask = {t['tau_ask']} ({t['rule']}); at it on validation: {t['at_tau']}."]
open(os.path.join(run, "report.md"), "w").write("\n".join(out) + "\n")
print(f"wrote {run}/report.md")
EOF
}

steps=("$@")
[ ${#steps[@]} -eq 0 ] && steps=(window pilot main export report)
for s in "${steps[@]}"; do
  case "$s" in
    window | pilot | main | export | report) ;;
    *) echo "unknown step $s (window, pilot, main, export, report)" >&2; exit 2 ;;
  esac
done
[ "$dry" = 1 ] || nice -n 19 cargo build -q --release -p sky-train -p sky-gguf -p sky-infer
for s in "${steps[@]}"; do
  case "$s" in
    window) window ;;
    pilot) pilot ;;
    main) main_ ;;
    export) export_ ;;
    report) report ;;
  esac
done
if [ -n "$smoke" ] && [ "$dry" != 1 ] && [ -z "${KEEP_SMOKE:-}" ]; then
  echo "== SMOKE: removing $SKY_OUT"
  rm -rf "${SKY_OUT:?}"
fi
