#!/usr/bin/env bash
# Overnight s4 pipeline (issue #1): a generic System One decision model distilled from Qwen3.5.
# The teacher (scripts/teacher-score.py, mlx-lm) scores the option letters of the s4 corpus in
# one pass per record; the student is the s3 architecture at language scale
# (configs/s4-micro.json) trained on the gold sets plus the gold / teacher mix, with ternary
# QAT after a 10% float warm-up, loop dropout and deep improvement supervision (--dis).
# Every GPU phase takes the global lock of scripts/night-s3.sh, so only one GPU job runs.
#
# Usage: scripts/night-s4.sh [stage ...]   (default: all)
#   stages: corpus label mix tokenizer pretrain decide post evaluate teacher-eval train all
#   train = mix and pretraining side by side (mix is CPU only), then decide, post, evaluate
# Environment:
#   TEACHER=mlx-community/Qwen3.5-4B-4bit   LABEL_UNTIL=HH:MM (stop labelling at this time)
#   PRE=30 DEC=110 BATCH=64 ALPHA=0.5 DIS=0.5
#   NAME=s4-micro CFG=configs/$NAME.json
# Outputs: data/s4/{queue,scored,mix,gold,heldout}, runs/$NAME/, models/$NAME/
set -euo pipefail
cd "$(dirname "$0")/.."
export DEVELOPER_DIR=${DEVELOPER_DIR:-/Applications/Xcode.app/Contents/Developer}
export CARGO_TARGET_DIR=${CARGO_TARGET_DIR:-$PWD/target/s4}
# MLX keeps freed buffers for reuse: capped so that the student and the teacher fit in 16 GB.
export SKY_MLX_CACHE_GB=${SKY_MLX_CACHE_GB:-2}
T=${T:-$CARGO_TARGET_DIR/release/sky-train}
D=${D:-$CARGO_TARGET_DIR/release/sky-data}
E=${E:-$CARGO_TARGET_DIR/release/sky-eval}
PY=${PY:-.venv-teacher/bin/python}
TEACHER=${TEACHER:-mlx-community/Qwen3.5-4B-4bit}
NAME=${NAME:-s4-micro}
cfg=${CFG:-configs/$NAME.json}
vocab=$(python3 -I -c "import json,sys;print(json.load(open(sys.argv[1]))['vocab_size'])" "$cfg")
tok=${TOK:-data/tok/tok-s4-$vocab.json}
pre=${PRE:-30} dec=${DEC:-110} batch=${BATCH:-64} alpha=${ALPHA:-0.5} dis=${DIS:-0.5}
run=runs/$NAME
m=models/$NAME
S4=data/s4
QUEUE=(p3-cases p2-generic p1-decisions)
RECORDS=$S4/gold,$S4/mix,$S4/heldout,data/numeric
WEIGHTS=${WEIGHTS:-gold=0.45,mix=0.5,heldout=0,numeric=0.05}
mkdir -p "$run" "$m" "$S4/scored"

LOCK=${GPU_LOCK:-/tmp/skycmd-night-gpu.lock}
# Runs "$@" holding the GPU lock (mkdir is atomic); the lock is released even on failure.
gpu() {
  until mkdir "$LOCK" 2>/dev/null; do sleep 3; done
  echo "$$ $NAME $(date +%T) $*" >"$LOCK/owner"
  local rc=0
  "$@" || rc=$?
  rm -rf "$LOCK"
  return $rc
}

corpus() {
  # --gold-cap bounds the gold set held in RAM by s2-decide (40k train/val records per source).
  "$D" s4-corpus --generic "${GENERIC:-30000}" --per-case "${PER_CASE:-15000}" --gold-cap "${GOLD_CAP:-40000}"
}

# Minutes left until LABEL_UNTIL (HH:MM, today or tomorrow), 0 when unset.
minutes_left() {
  [ -n "${LABEL_UNTIL:-}" ] || { echo 0; return; }
  python3 -I -c '
import sys, datetime as dt
h, m = map(int, sys.argv[1].split(":"))
now = dt.datetime.now()
end = now.replace(hour=h, minute=m, second=0, microsecond=0)
if end <= now - dt.timedelta(hours=12):
    end += dt.timedelta(days=1)
print(max(0.0, (end - now).total_seconds() / 60))' "$LABEL_UNTIL"
}

label() {
  for q in "${QUEUE[@]}"; do
    local left
    left=$(minutes_left)
    if [ -n "${LABEL_UNTIL:-}" ] && python3 -I -c 'import sys; sys.exit(float(sys.argv[1]) > 1)' "$left"; then
      echo "== label: time is up before $q"
      break
    fi
    echo "== label $q with $TEACHER (deadline ${left} min)"
    gpu "$PY" -I scripts/teacher-score.py --model "$TEACHER" --in "$S4/queue/$q.jsonl" \
      --out "$S4/scored/$q.jsonl" --deadline "$left"
  done
}

mix() {
  local scored=()
  for q in "${QUEUE[@]}"; do scored+=("$S4/scored/$q.jsonl"); done
  "$D" s4-mix --scored "$(IFS=,; echo "${scored[*]}")" --queue "$S4/queue" --out "$S4/mix" --alpha "$alpha"
}

tokenizer() {
  [ -f data/numeric/numeric.jsonl ] || "$T" s2-numeric --n 100000 --out data/numeric/numeric.jsonl
  mkdir -p data/tok
  "$D" train-tokenizer --vocab "$vocab" --num --inputs data/text,data/decisions,data/s4/queue \
    --out "$tok"
}

pretrain() {
  echo "== $NAME: pretrain ${pre}m"
  gpu "$T" s2-pretrain --config "$cfg" --tokenizer "$tok" --text data/text --minutes "$pre" --num-weight 0.05 \
    --seq "${SEQ:-256}" --batch "${PBATCH:-64}" \
    --numeric data/numeric --numeric-weight 0.1 --out "$run/pretrain"
}

decide() {
  echo "== $NAME: decide ${dec}m (QAT after a 10% float warm-up, loop dropout 0.25, DIS beta0 $dis)"
  rm -rf "$run/decide"
  gpu "$T" s2-decide --config "$cfg" --init "$run/pretrain/init" --tokenizer "$tok" --records "$RECORDS" \
    --weights "$WEIGHTS" --batch "$batch" --minutes "$dec" \
    --qat-frac 0.9 --loop-dropout 0.25 --dis "$dis" --length-pool "${POOL:-16}" --pad-to 32 --out "$run/decide"
}

post() {
  gpu "$T" s2-calibrate --run "$run/decide/last" --records "$RECORDS"
  gpu "$T" s2-calibrate --run "$run/decide/pre-qat" --records "$RECORDS"
  gpu "$T" s2-export --run "$run/decide/pre-qat" --format f32 --out "$m/skycmd-$NAME-f32.gguf"
  gpu "$T" s2-export --run "$run/decide/last" --format ternary --out "$m/skycmd-$NAME-ternary.gguf"
}

# s2-eval report of the s4 checkpoint and GGUFs (per exit, early exit, permutation check), then
# sky-eval of every GGUF with its own tokenizer on the test sets (typed_decisions_bench, the
# zero-shot held-out sets, When2Call test), the s3-pico baseline included, and the gate.
eval_() {
  gpu "$T" s2-eval --run "$run/decide/last" --records "$RECORDS" \
    --gguf "$m/skycmd-$NAME-ternary.gguf,$m/skycmd-$NAME-f32.gguf" --report "$run/report.md"
  local tests=data/decisions/typed_decisions_bench.jsonl,$S4/heldout,data/decisions/when2call.jsonl
  local g
  for g in "$m/skycmd-$NAME-ternary.gguf" "$m/skycmd-$NAME-f32.gguf" "${BASELINE:-site/public/models/s3-pico.gguf}"; do
    [ -f "$g" ] || continue
    "$E" "$g" --records "$tests" --json "$run/eval-$(basename "$g" .gguf).json"
  done
  python3 -I scripts/s4-gate.py "$run" --out "$run/results.md" || true
}

# The teacher on the same test sets (accuracy of the letter scoring, raw temperature 1).
teacher_eval() {
  mkdir -p "$run/teacher"
  for f in data/decisions/typed_decisions_bench.jsonl "$S4"/heldout/*.jsonl; do
    local b
    b=$(basename "$f" .jsonl)
    [ -s "$run/teacher/$b.summary.json" ] && continue
    gpu "$PY" -I scripts/teacher-score.py --model "$TEACHER" --in "$f" --out "$run/teacher/$b.jsonl" --eval \
      | tail -1 >"$run/teacher/$b.summary.json"
  done
}

train() {
  mix &
  local mix_pid=$!
  [ -f "$tok" ] || tokenizer
  pretrain
  wait "$mix_pid"
  decide
  post
  eval_
}

stages=("$@")
[ ${#stages[@]} -eq 0 ] && stages=(all)
for s in "${stages[@]}"; do
  case "$s" in
    corpus) corpus ;;
    label) label ;;
    mix) mix ;;
    tokenizer) tokenizer ;;
    pretrain) pretrain ;;
    decide) decide ;;
    post) post ;;
    evaluate) eval_ ;;
    teacher-eval) teacher_eval ;;
    train) train ;;
    all)
      [ -f "$S4/queue/p1-decisions.jsonl" ] || corpus
      label
      train
      ;;
    *) echo "unknown stage $s" >&2; exit 1 ;;
  esac
done
