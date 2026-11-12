#!/usr/bin/env bash
# s6-sleep: the pico pilot and the nano runs of docs/s6-sleep-prereg.md (TODO §10).
#
#   scripts/train-sleep.sh [data] [pilot] [nano] [test] [report]     # default: all, in this order
#
#   data     sequences of the two probes (recall: whole episodes of 64 frames; rule110: 8 frames)
#            and of the 4 games (24 frames), in <root>/seq
#   pilot    P1 to P4 on s6-pico (§5), read on the dev seeds; the three rules are applied by
#            scripts/s6-sleep-report.py and written to <root>/pilot.json
#   nano     the two arms, 3 seeds each (§6), in the form the pilot kept; N is then chosen on the
#            dev seeds (<root>/n.json)
#   test     the test seeds of the probes, the arena and typed_decisions_bench, READ ONCE (§7):
#            the step refuses to run again once <root>/test/done exists
#   report   <root>/REPORT.md
#
# Budgets are in steps: the first run of a size trains for a working time (PILOT_MIN, NANO_MIN) and
# every other run of that size for the number of steps it did. Runs that ended are skipped, so the
# script can be started again after a stop (RESUME=1 also resumes the run that was cut).
#
#   SMOKE=1        tiny budgets, few episodes and data/numeric as the only stateless records, under
#                  runs/s6-sleep-smoke (code check, no result)
#   SKY_TRAIN      a sky-train binary to use as is, instead of building the release one
#   DRY_RUN=1      print the commands
#   PICO_BASE      runs/s6lr/softmax-adamw-x2/decide/last (the s6-pico of the adopted recipe)
#   NANO_BASE      runs/s6-nano-g$ROUNDS/decide/last (the s6-nano of tranche 2)
#   OPT LR_SCALE   the recipe (eval "$(python3 -I scripts/s6-adopt.py env s6-nano)")
set -euo pipefail
export DEVELOPER_DIR=${DEVELOPER_DIR:-/Applications/Xcode.app/Contents/Developer}
export CARGO_TARGET_DIR=${CARGO_TARGET_DIR:-$PWD/target/s3-train}
# The thermal brake, as scripts/train-all.sh: throttled unless set, the brake file of
# scripts/auto-brake.sh when it runs, and a cap on the MLX buffer cache.
export SKY_DUTY=${SKY_DUTY:-0.5}
export SKY_COOL=${SKY_COOL-45:15}
export SKY_BRAKE_FILE=${SKY_BRAKE_FILE-$PWD/runs/brake}
export SKY_MLX_CACHE_GB=${SKY_MLX_CACHE_GB:-2}
dry=${DRY_RUN:-0}
smoke=${SMOKE:-0}
T=${SKY_TRAIN:-$CARGO_TARGET_DIR/release/sky-train}
root=runs/s6-sleep
pilot_min=${PILOT_MIN:-20} nano_min=${NANO_MIN:-30}
dev_eps=500 test_eps=1000 arena_eps=200
n_recall=10000 n_rule=20000 n_game=5000
if [ "$smoke" = 1 ]; then
  root=runs/s6-sleep-smoke
  pilot_min=${PILOT_MIN:-0.5} nano_min=${NANO_MIN:-0.5}
  dev_eps=4 test_eps=4 arena_eps=1
  n_recall=200 n_rule=200 n_game=100
fi
dev_seed=1000000000 test_seed=5000000000 arena_seed=5000
seq=$root/seq
games=(flappy rescue dodge doom)
pico_base=${PICO_BASE:-runs/s6lr/softmax-adamw-x2/decide/last}
nano_base=${NANO_BASE:-runs/s6-nano-g${ROUNDS:-3}/decide/last}
read -r -a opt <<<"${OPT:---no-muon}"
lr_flag=--muon-lr
[[ " ${OPT:---no-muon} " == *" --no-muon "* ]] && lr_flag=--adam-lr
resume=()
[ -n "${RESUME:-}" ] && resume=(--resume)

x() {
  if [ "$dry" = 1 ]; then
    printf '+'
    printf ' %q' "$@"
    printf '\n'
  else
    "$@"
  fi
}

# rate <width> <size pilot scale>: the peak rate of the base's decision training (scripts/train-s6.sh).
rate() { LC_ALL=C awk -v b="${LR_SCALE:-2.0}" -v c="$2" -v d="$1" 'BEGIN { printf "%.3g", 1e-3 * b * sqrt(96 / d) * c }'; }

# config <base checkpoint> <out.json> <memory: true|false> <name>
config() {
  [ "$dry" = 1 ] && { echo "+ config $*"; return; }
  python3 -I - "$1/config.json" "$2" "$3" "$4" <<'EOF'
import json, sys
src, dst, memory, name = sys.argv[1:]
c = json.load(open(src))
c["temperature"] = {}
c["name"] = name
if memory == "true":
    c["memory"] = True
else:
    c.pop("memory", None)
json.dump(c, open(dst, "w"), indent=2)
EOF
}

# steps_of <run dir>: the steps of a run that ended (empty otherwise).
steps_of() {
  [ -f "$1/last/state.json" ] || return 0
  python3 -I - "$1" <<'EOF'
import json, os, sys
log = os.path.join(sys.argv[1], "log.jsonl")
done = os.path.exists(log) and any(json.loads(l).get("final") for l in open(log) if l.strip())
if done:
    print(json.load(open(os.path.join(sys.argv[1], "last", "state.json")))["step"])
EOF
}

data() {
  x mkdir -p "$seq"
  [ -f "$seq/recall.jsonl" ] || x "$T" s2-seqgen --games recall --n-seq "$n_recall" --k 64 --seed 61 --out "$seq"
  [ -f "$seq/rule110.jsonl" ] || x "$T" s2-seqgen --games rule110 --n-seq "$n_rule" --k 8 --seed 61 --out "$seq"
  local g
  for g in "${games[@]}"; do
    [ -f "$seq/$g.jsonl" ] || x "$T" s2-seqgen --games "$g" --n-seq "$n_game" --k 24 --seed 61 --out "$seq"
  done
}

# run <out> <config> <base> <records> <weights> <batch> <rate> <seed> <budget flag> <budget> [flags]
# Half of the steps are batches of 16 sequences (recall 40%, rule110 12%, each game 12%), the
# others the stateless batches of the base's decision training. QAT from the first step.
run() {
  local out=$1 cfg=$2 base=$3 records=$4 weights=$5 batch=$6 lr=$7 seed=$8 flag=$9 budget=${10}
  shift 10
  if [ "$dry" != 1 ] && [ -n "$(steps_of "$out")" ]; then
    echo "== $out: done ($(steps_of "$out") steps), skipped"
    return
  fi
  [ -n "${RESUME:-}" ] || x rm -rf "$out"
  echo "== $out: $flag $budget, seed $seed $*"
  x "$T" s2-decide --config "$cfg" --init "$base" --tokenizer "$base/tokenizer.json" \
    --records "$records" --weights "$weights" --batch "$batch" "$flag" "$budget" --seed "$seed" \
    --qat-frac 1.0 --loop-dropout 0.25 --length-pool 16 --pad-to 32 --out "$out" \
    "$lr_flag" "$lr" ${opt[@]+"${opt[@]}"} ${resume[@]+"${resume[@]}"} \
    --sleep-records "$seq/recall.jsonl,$seq/rule110.jsonl,$seq/flappy.jsonl,$seq/rescue.jsonl,$seq/dodge.jsonl,$seq/doom.jsonl" \
    --sleep-weights recall=0.40,rule110=0.12,flappy=0.12,rescue=0.12,dodge=0.12,doom=0.12 \
    --sleep-batch 16 --seq-frac 0.5 "$@"
}

# probes <run dir> <out dir> <first seed> <episodes>: both probes at N = 1, 2 and 4.
probes() {
  local g
  x mkdir -p "$2"
  for g in recall rule110; do
    [ "$dry" != 1 ] && [ -f "$2/$g.json" ] && continue
    x "$T" s2-probe "$1/last" --game "$g" --from-seed "$3" --episodes "$4" --passes 1,2,4 --json "$2/$g.json"
  done
}

pilot() {
  local p=$root/pilot records=data/decisions,data/games,data/numeric weights=decisions=0.3,games=0.6,numeric=0.1
  local lr steps
  [ "$smoke" = 1 ] && records=data/numeric weights=numeric=1
  lr=$(rate 96 1)
  x mkdir -p "$p"
  config "$pico_base" "$p/sleep.json" true s6-sleep-pico
  config "$pico_base" "$p/control.json" false s6-pico-control
  run "$p/p1" "$p/sleep.json" "$pico_base" "$records" "$weights" 128 "$lr" 0 --minutes "$pilot_min"
  steps=$(steps_of "$p/p1")
  [ "$dry" = 1 ] && steps=STEPS
  [ -n "$steps" ] || { echo "pilot P1 did not end" >&2; exit 1; }
  run "$p/p2" "$p/sleep.json" "$pico_base" "$records" "$weights" 128 "$lr" 0 --steps "$steps" --sleep-passes 4
  run "$p/p3" "$p/sleep.json" "$pico_base" "$records" "$weights" 128 "$lr" 0 --steps "$steps" --sleep-passes 4 --sleep-full
  run "$p/p4" "$p/control.json" "$pico_base" "$records" "$weights" 128 "$lr" 0 --steps "$steps"
  local r
  for r in p1 p2 p3 p4; do
    probes "$p/$r" "$p/$r/dev" "$dev_seed" "$dev_eps"
  done
  x python3 -I scripts/s6-sleep-report.py pilot "$root"
}

nano() {
  local n=$root/nano records=data/decisions,data/distill,data/games,data/numeric
  local weights=decisions=0.25,distill=0.30,games=0.35,numeric=0.10 lr lr_x=${LR_X:-} steps seed
  local -a form=()
  [ "$smoke" = 1 ] && records=data/numeric weights=numeric=1
  if [ "$dry" = 1 ]; then
    lr_x=${lr_x:-1}
  else
    [ -f "$root/pilot.json" ] || { echo "no $root/pilot.json: run the pilot first" >&2; exit 1; }
    # The pilot's verdict: stop, or the flags of the sleep arm (docs/s6-sleep-prereg.md §5).
    local flags
    flags=$(python3 -I scripts/s6-sleep-report.py form "$root") || exit 1
    read -r -a form <<<"$flags"
    [ -n "$lr_x" ] || lr_x=$(python3 -I -c 'import json,sys; print(json.load(open(sys.argv[1]))["chosen"])' runs/s6-nano-lr/summary.json)
  fi
  lr=$(rate 144 "$lr_x")
  x mkdir -p "$n"
  config "$nano_base" "$n/sleep.json" true s6-sleep-nano
  config "$nano_base" "$n/control.json" false s6-nano-control
  run "$n/sleep-s0" "$n/sleep.json" "$nano_base" "$records" "$weights" 128 "$lr" 0 --minutes "$nano_min" ${form[@]+"${form[@]}"}
  steps=$(steps_of "$n/sleep-s0")
  [ "$dry" = 1 ] && steps=STEPS
  [ -n "$steps" ] || { echo "the first nano run did not end" >&2; exit 1; }
  for seed in 1 2; do
    run "$n/sleep-s$seed" "$n/sleep.json" "$nano_base" "$records" "$weights" 128 "$lr" "$seed" --steps "$steps" ${form[@]+"${form[@]}"}
  done
  for seed in 0 1 2; do
    run "$n/control-s$seed" "$n/control.json" "$nano_base" "$records" "$weights" 128 "$lr" "$seed" --steps "$steps"
  done
  for seed in 0 1 2; do
    probes "$n/sleep-s$seed" "$n/sleep-s$seed/dev" "$dev_seed" "$dev_eps"
    probes "$n/control-s$seed" "$n/control-s$seed/dev" "$dev_seed" "$dev_eps"
  done
  x python3 -I scripts/s6-sleep-report.py choose "$root"
}

test_() {
  local n=$root/nano t=$root/test arm seed g chosen=1 r
  if [ "$dry" != 1 ]; then
    [ -f "$t/done" ] && { echo "$t/done exists: the test seeds were read, they are not read again" >&2; exit 1; }
    chosen=$(python3 -I -c 'import json,sys; print(json.load(open(sys.argv[1]))["n"])' "$root/n.json")
  fi
  x mkdir -p "$t"
  for arm in sleep control; do
    for seed in 0 1 2; do
      r=$n/$arm-s$seed
      probes "$r" "$t/$arm-s$seed" "$test_seed" "$test_eps"
      for g in "${games[@]}"; do
        [ "$dry" != 1 ] && [ -f "$t/$arm-s$seed/$g.json" ] && continue
        x "$T" s2-probe "$r/last" --game "$g" --from-seed "$arena_seed" --episodes "$arena_eps" \
          --passes "$chosen" --json "$t/$arm-s$seed/$g.json"
      done
      [ "$dry" != 1 ] && [ -f "$t/$arm-s$seed/typed.md" ] && continue
      x "$T" s2-eval --run "$r/last" --records data/decisions/typed_decisions_bench.jsonl \
        --report "$t/$arm-s$seed/typed.md"
    done
  done
  x touch "$t/done"
}

[ "$dry" = 1 ] || [ -n "${SKY_TRAIN:-}" ] || cargo build -q --release -p sky-train
steps=("$@")
[ ${#steps[@]} -eq 0 ] && steps=(data pilot nano test report)
for s in "${steps[@]}"; do
  case "$s" in
    data) data ;;
    pilot) pilot ;;
    nano) nano ;;
    test) test_ ;;
    report) x python3 -I scripts/s6-sleep-report.py report "$root" ;;
    *) echo "unknown step $s (data, pilot, nano, test, report)" >&2; exit 2 ;;
  esac
done
