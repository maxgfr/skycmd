#!/usr/bin/env bash
# The big run on one machine (TODO.md §9): the s6 recipe, then the five s6 models, in the brake mode
# of docs/s6-prereg.md amendment 5, with no decision taken by hand. Run it from the main checkout.
#
# Usage: scripts/train-all.sh [step...]
#   steps, in order: pilot ablation adopt ear secrets nano micro cast report
#   (default: all of them; SMOKE=1: ear secrets nano micro cast report, the recipe must exist)
#
#   DRY_RUN=1       print the commands; nothing is built, trained or written
#   SMOKE=1         budgets of a few minutes, ROUNDS=1, DAGGER_N=200, EPISODES=2. Everything goes under
#                   runs/all-smoke/, never in runs/s6-*: with RESUME=1 a real run would otherwise
#                   continue from a smoke checkpoint
#   SKY_DUTY=0.5  SKY_COOL=45:15  RESUME=1   the defaults here; RESUME= (empty) starts each training over
#   SKY_BRAKE_FILE=runs/brake                a file that sets the brake while the run goes on, written by
#                                            scripts/auto-brake.sh (0.85 and 50:10 when the Mac is left alone)
#   SKY_MLX_CACHE_GB=2                       cap of MLX's buffer cache
#
# Each step is idempotent: it leaves a marker in runs/all/done/ and a log in runs/all/. A step cut by
# a shutdown starts again and continues from its checkpoints. A gate that fails is a result: it goes
# in the report and the run continues. Only a crash stops the run, to be noted in the report.
set -euo pipefail
cd "$(dirname "$0")/.."
export DEVELOPER_DIR=${DEVELOPER_DIR:-/Applications/Xcode.app/Contents/Developer}
export SKYCMD_STORE=${SKYCMD_STORE:-$PWD}
export SKY_DUTY=${SKY_DUTY:-0.5}
export SKY_COOL=${SKY_COOL-45:15}
export RESUME=${RESUME-1}
# A 16 GB machine: without a cap, MLX keeps every freed buffer for reuse.
export SKY_MLX_CACHE_GB=${SKY_MLX_CACHE_GB:-2}
# scripts/auto-brake.sh writes this file while the run goes on; without it SKY_DUTY and SKY_COOL apply.
export SKY_BRAKE_FILE=${SKY_BRAKE_FILE:-$PWD/runs/brake}
dry=${DRY_RUN:-0}
all=runs/all
if [ -n "${SMOKE:-}" ]; then
  all=runs/all-smoke
  export SKY_OUT=runs/all-smoke/
  export PRE_MIN=2 DEC_MIN=2 CTL_MIN=1 ROUND_MINUTES=1 LR_MIN=1 PRETRAIN_MIN=1 DECIDE_MIN=2 MINUTES=3
  export ROUNDS=1 DAGGER_N=200 EPISODES=2 PILOT_STEPS=100
fi
recipe_json=runs/recipe/recipe.json
# The records GLM had to label (TODO.md §1.1); data/distill/agreement.json gives those it labelled.
GLM_SELECTED=747198
GLM_MIN_COVER=0.95
# docs/s6-ear-prereg.md, amendment 2: 10 x chance over the 2,048 entries of Mimi's codebook 0.
EAR_MIN_CB0=$(LC_ALL=C awk 'BEGIN { printf "%.6f", 10 / 2048 }')

# with_recipe <model> <command...>: the command with the recipe's variables of that model. Always
# called inside the subshell of run_step, so that nothing leaks to the next model.
with_recipe() {
  local model=$1
  shift
  if [ -f "$recipe_json" ]; then
    eval "$(python3 -I scripts/s6-adopt.py env "$model")"
    export CONFIG OPT LR_SCALE PRETRAIN DECIDE CAST_OPT
  elif [ "$dry" = 1 ]; then
    echo "# no $recipe_json yet: the commands below show the default options"
  else
    echo "no $recipe_json: run pilot, ablation and adopt first" >&2
    return 1
  fi
  "$@"
}

# run_step <name> <command...>: skipped when its marker exists; the log is appended to.
run_step() {
  local name=$1
  shift
  if [ -f "$all/done/$name" ]; then
    echo "== $name: already done, skipped"
    return 0
  fi
  echo "== $(date '+%F %T') $name"
  if [ "$dry" = 1 ]; then
    ("$@")
    return 0
  fi
  mkdir -p "$all/done"
  ("$@") 2>&1 | tee -a "$all/$name.log"
  touch "$all/done/$name"
}

pilot() {
  if [ "$dry" = 1 ]; then
    echo "+ cargo xtask s6lr"
    echo "+ cargo xtask s6lr --extend"
  else
    cargo xtask s6lr
    cargo xtask s6lr --extend
  fi
}

ablation() {
  if [ "$dry" = 1 ] && [ ! -f runs/s6lr/summary.json ]; then
    echo "+ cargo xtask s6ab --adamw-lr-scale <from runs/s6lr/summary.json> --muon-lr-scale <from runs/s6lr/summary.json>"
    return 0
  fi
  local a m
  read -r a m <<<"$(python3 -I -c 'import json, sys
s = json.load(open(sys.argv[1]))["lr_scale"]
print(s["adamw"], s["muon"])' runs/s6lr/summary.json)"
  if [ "$dry" = 1 ]; then
    echo "+ cargo xtask s6ab --adamw-lr-scale $a --muon-lr-scale $m"
  else
    cargo xtask s6ab --adamw-lr-scale "$a" --muon-lr-scale "$m"
  fi
}

# The recipe is written once: its timestamp and sha256 are part of the record.
adopt() {
  if [ -f "$recipe_json" ]; then
    echo "$recipe_json exists: kept as is"
  elif [ "$dry" = 1 ]; then
    echo "+ python3 -I scripts/s6-adopt.py adopt --pilot runs/s6lr/summary.json --ablation runs/s6ab/summary.json --out runs/recipe"
  else
    python3 -I scripts/s6-adopt.py adopt --pilot runs/s6lr/summary.json --ablation runs/s6ab/summary.json --out runs/recipe
  fi
}

check_ear() {
  if [ "$dry" = 1 ]; then
    echo "+ check: final val_acc_cb0 of ${SKY_OUT:-}runs/s6-ear-nano/pretrain/log.jsonl >= $EAR_MIN_CB0"
    return 0
  fi
  if [ -n "${SMOKE:-}" ]; then
    echo "smoke budget: val_acc_cb0 is not checked"
    return 0
  fi
  python3 -I -c 'import json, sys
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
acc = [r["val_acc_cb0"] for r in rows if r.get("final")][-1]
bar = float(sys.argv[2])
print(f"val_acc_cb0 {acc:.5f} (bar {bar:.5f}, docs/s6-ear-prereg.md amendment 2)")
sys.exit(0 if acc >= bar else 1)' "runs/s6-ear-nano/pretrain/log.jsonl" "$EAR_MIN_CB0"
}

check_glm() {
  if [ "$dry" = 1 ]; then
    echo "+ check: data/distill/glm.jsonl exists, labels cover at least $GLM_MIN_COVER of $GLM_SELECTED records"
    return 0
  fi
  [ -s data/distill/glm.jsonl ] || { echo "no data/distill/glm.jsonl" >&2; return 1; }
  python3 -I -c 'import json, sys
n = sum(v["labelled"] for v in json.load(open(sys.argv[1])).values())
cover = n / int(sys.argv[2])
print(f"GLM labels: {n} of {sys.argv[2]} ({cover:.4f})")
sys.exit(0 if cover >= float(sys.argv[3]) else 1)' data/distill/agreement.json "$GLM_SELECTED" "$GLM_MIN_COVER"
}

ear() {
  run_step ear-pretrain with_recipe s6-ear-nano scripts/train-ear.sh pretrain
  run_step ear-check check_ear
  run_step ear-train with_recipe s6-ear-nano scripts/train-ear.sh decide eval
}

secrets() {
  run_step secrets-train with_recipe s6-secrets-pico scripts/train-secrets.sh train eval
}

s6_size() { # nano or micro
  local size=$1 s
  run_step "$size-glm" check_glm
  for s in lr train dagger exit eval; do
    run_step "$size-$s" with_recipe "s6-$size" env "SIZE=$size" scripts/train-s6.sh "$s"
  done
}

cast() {
  local s
  for s in pilot train export eval; do
    run_step "cast-$s" with_recipe s6-cast-pico scripts/train-cast.sh "$s"
  done
}

report() {
  # Written again at each call, from the files of the steps that have finished.
  local cmd=(python3 -I scripts/all-report.py --prefix "${SKY_OUT:-}" --out "$all/REPORT.md")
  if [ "$dry" = 1 ]; then
    printf '+'
    printf ' %q' "${cmd[@]}"
    printf '\n'
  else
    "${cmd[@]}"
  fi
}

steps=("$@")
if [ ${#steps[@]} -eq 0 ]; then
  steps=(pilot ablation adopt ear secrets nano micro cast report)
  [ -z "${SMOKE:-}" ] || steps=(ear secrets nano micro cast report)
fi
if [ "$dry" != 1 ] && [ "${steps[*]}" != report ] && pgrep -x sky-train >/dev/null; then
  echo "a sky-train is already running: one training at a time on this GPU" >&2
  exit 1
fi
echo "== train-all: ${steps[*]} | SKY_DUTY=$SKY_DUTY SKY_COOL=$SKY_COOL RESUME=$RESUME${SMOKE:+ SMOKE (runs/all-smoke/)}"
for s in "${steps[@]}"; do
  case "$s" in
    pilot) run_step pilot pilot ;;
    ablation) run_step ablation ablation ;;
    adopt) adopt ;;
    ear) ear ;;
    secrets) secrets ;;
    nano | micro) s6_size "$s" ;;
    cast) cast ;;
    report) report ;;
    *)
      echo "unknown step $s (pilot, ablation, adopt, ear, secrets, nano, micro, cast, report)" >&2
      exit 2
      ;;
  esac
done
