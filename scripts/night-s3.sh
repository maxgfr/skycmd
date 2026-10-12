#!/usr/bin/env bash
# Overnight s3 candidate pipeline (docs/night-experiments.md): the train-s3.sh recipe
# (pretrain, decide with QAT after a 10% float warm-up, loop dropout 0.25) followed by DAgger
# rounds on all four games (flappy, rescue, dodge, doom), closed-loop tau calibration and the
# 30-episode gate. Every GPU phase (mlx-rs training, calibration, export, eval) takes a global
# lock, so two candidates can run side by side and only one GPU job ever runs at a time: the
# CPU phases (DAgger collection and arena play) of one overlap the training of the other.
#
# Usage: scripts/night-s3.sh <name> [stage ...]   (default stages: all)
#   stages: pretrain decide base dagger calib gate all
# Environment:
#   CFG=configs/<name>.json   VOCAB=<from config>   TOK=data/tok/tok-s2-$VOCAB.json
#   PRE=8 DEC=15 RMIN=8 ROUNDS=3 BATCH=128        (minutes / rounds, as for s3-pico)
#   PRETRAIN_FROM=runs/night-<other>/pretrain      (reuse a pretraining run of the same encoder)
#   DEC_WEIGHTS=games=0.6,numeric=0.1,decisions=0.3
#   START=r                                        (resume the DAgger rounds at round r)
# Outputs: runs/night-<name>/, models/candidates/<name>/, data/dagger-night-<name>/rN
set -euo pipefail
cd "$(dirname "$0")/.."
export DEVELOPER_DIR=${DEVELOPER_DIR:-/Applications/Xcode.app/Contents/Developer}
export CARGO_TARGET_DIR=${CARGO_TARGET_DIR:-$PWD/target/night}
T=${T:-$CARGO_TARGET_DIR/release/sky-train}
A=${A:-$CARGO_TARGET_DIR/release/sky-arena}
RECORDS=data/decisions,data/games,data/numeric
GAMES=${GAMES:-flappy,rescue,dodge,doom}

name=$1
shift
stages=("$@")
[ ${#stages[@]} -eq 0 ] && stages=(all)
cfg=${CFG:-configs/$name.json}
vocab=${VOCAB:-$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['vocab_size'])" "$cfg")}
tok=${TOK:-data/tok/tok-s2-$vocab.json}
pre=${PRE:-8} dec=${DEC:-15} rmin=${RMIN:-8} rounds=${ROUNDS:-3} batch=${BATCH:-128}
dec_weights=${DEC_WEIGHTS:-games=0.6,numeric=0.1,decisions=0.3}
run=runs/night-$name
m=models/candidates/$name
mkdir -p "$run" "$m"

LOCK=${GPU_LOCK:-/tmp/skycmd-night-gpu.lock}
# Runs "$@" holding the GPU lock (mkdir is atomic); the lock is released even on failure.
gpu() {
  until mkdir "$LOCK" 2>/dev/null; do sleep 3; done
  echo "$$ $name $(date +%T) $*" >"$LOCK/owner"
  local rc=0
  "$@" || rc=$?
  rm -rf "$LOCK"
  return $rc
}

pretrain() {
  echo "== $name: pretrain ${pre}m"
  gpu "$T" s2-pretrain --config "$cfg" --tokenizer "$tok" --text data/text --minutes "$pre" --num-weight 0.05 \
    --numeric data/numeric --numeric-weight 0.1 --out "$run/pretrain"
}

decide() {
  local init=${PRETRAIN_FROM:-$run/pretrain}
  echo "== $name: decide ${dec}m from $init"
  rm -rf "$run/decide"
  gpu "$T" s2-decide --config "$cfg" --init "$init/init" --tokenizer "$tok" --records "$RECORDS" \
    --weights "$dec_weights" --batch "$batch" --minutes "$dec" \
    --qat-frac 0.9 --loop-dropout 0.25 --pad-to 32 --out "$run/decide"
}

# Calibrates, exports the ternary GGUF of the decision run and plays the base model.
base() {
  gpu "$T" s2-calibrate --run "$run/decide/last" --records "$RECORDS"
  gpu "$T" s2-export --run "$run/decide/last" --format ternary --out "$m/$name-base.gguf"
  "$A" "$m/$name-base.gguf" --episodes 30 --exit-threshold 1.0 --game "$GAMES" --json "$run/arena-base.json"
}

dagger() {
  local start=${START:-1} init model
  if [ "$start" -gt 1 ]; then
    init=$run/g$((start - 1))/decide/last
    model=$m/$name-g$((start - 1)).gguf
  else
    init=$run/decide/last
    model=$m/$name-base.gguf
  fi
  for r in $(seq "$start" "$rounds"); do
    local dag=data/dagger-night-$name/r$r out=$m/$name-g$r.gguf
    echo "== $name round $r: DAgger from $model"
    "$A" "$model" --dagger 15000 --seed "$r" --game "$GAMES" --out "$dag"
    rm -rf "$run/g$r/decide"
    gpu "$T" s2-decide --config "$cfg" --init "$init" --tokenizer "$tok" \
      --records "$RECORDS,$dag" --weights "games=0.5,r$r=0.35,numeric=0.05,decisions=0.1" \
      --batch "$batch" --minutes "$rmin" --qat-frac 1.0 --loop-dropout 0.25 --pad-to 32 --out "$run/g$r/decide"
    gpu "$T" s2-calibrate --run "$run/g$r/decide/last" --records "$RECORDS"
    gpu "$T" s2-export --run "$run/g$r/decide/last" --format ternary --out "$out"
    "$A" "$out" --episodes 30 --exit-threshold 1.0 --game "$GAMES" --json "$run/g$r/arena.json"
    python3 -I scripts/night-gate.py "$run/g$r/arena.json" || true
    init=$run/g$r/decide/last
    model=$out
  done
}

# Closed-loop tau (scripts/calibrate-exit.sh on all four games) of the last round -> <name>.gguf.
calib() {
  local last=${LAST:-$m/$name-g$rounds.gguf}
  A=$A GAMES=$GAMES EPISODES=${EPISODES:-20} CALIB_DIR=$run/calibrate-exit GGUF_TARGET_DIR=$CARGO_TARGET_DIR \
    scripts/calibrate-exit.sh "$last" "$m/$name.gguf"
}

gate() {
  "$A" "$m/$name.gguf" --episodes 30 --game "$GAMES" --json "$run/gate.json"
  python3 -I scripts/night-gate.py "$run/gate.json"
}

for s in "${stages[@]}"; do
  case "$s" in
    pretrain) pretrain ;;
    decide) decide ;;
    base) base ;;
    dagger) dagger ;;
    calib) calib ;;
    gate) gate ;;
    all)
      [ -n "${PRETRAIN_FROM:-}" ] || pretrain
      decide
      base
      dagger
      calib
      gate || true
      ;;
    *) echo "unknown stage $s" >&2; exit 1 ;;
  esac
done
