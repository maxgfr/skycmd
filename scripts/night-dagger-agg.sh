#!/usr/bin/env bash
# One aggregated DAgger round for a night candidate (scripts/night-s3.sh): the latest model plays
# and the oracle labels its states (round R), then the model is fine-tuned on the union of every
# round's DAgger set (r1..rR), the textbook DAgger aggregate, instead of the last round alone.
# Same budget, QAT and loop dropout as the other rounds; then calibration, export and the
# 30-episode full-depth arena.
# Usage: scripts/night-dagger-agg.sh <name> <round R>   (env: CFG, TOK, RMIN=8, BATCH=128, T, A)
set -euo pipefail
cd "$(dirname "$0")/.."
export DEVELOPER_DIR=${DEVELOPER_DIR:-/Applications/Xcode.app/Contents/Developer}
T=${T:-$PWD/target/night-frozen/v1/sky-train}
A=${A:-$PWD/target/night-frozen/v1/sky-arena}
name=$1 r=$2 prev=$(($2 - 1))
cfg=${CFG:-configs/$name.json}
vocab=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['vocab_size'])" "$cfg")
tok=${TOK:-data/tok/tok-s2-$vocab.json}
rmin=${RMIN:-8} batch=${BATCH:-128}
GAMES=${GAMES:-flappy,rescue,dodge,doom}
RECORDS=data/decisions,data/games,data/numeric
run=runs/night-$name m=models/candidates/$name dag=data/dagger-night-$name
LOCK=${GPU_LOCK:-/tmp/skycmd-night-gpu.lock}
gpu() {
  until mkdir "$LOCK" 2>/dev/null; do sleep 3; done
  echo "$$ $name $(date +%T) $*" >"$LOCK/owner"
  local rc=0
  "$@" || rc=$?
  rm -rf "$LOCK"
  return $rc
}
echo "== $name aggregated round $r: DAgger from $m/$name-g$prev.gguf"
"$A" "$m/$name-g$prev.gguf" --dagger 15000 --seed "$r" --game "$GAMES" --out "$dag/r$r"
# The newest round weighs as much as the older ones together.
recs=$RECORDS w="games=0.4,numeric=0.05,decisions=0.1"
for i in $(seq 1 "$r"); do
  recs="$recs,$dag/r$i"
  if [ "$i" -eq "$r" ]; then w="$w,r$i=0.225"; else w="$w,r$i=$(python3 -c "print(round(0.225/$prev, 4))")"; fi
done
rm -rf "$run/g$r/decide"
gpu "$T" s2-decide --config "$cfg" --init "$run/g$prev/decide/last" --tokenizer "$tok" \
  --records "$recs" --weights "$w" --batch "$batch" --minutes "$rmin" --qat-frac 1.0 --loop-dropout 0.25 \
  --pad-to 32 --out "$run/g$r/decide"
gpu "$T" s2-calibrate --run "$run/g$r/decide/last" --records "$RECORDS"
gpu "$T" s2-export --run "$run/g$r/decide/last" --format ternary --out "$m/$name-g$r.gguf"
"$A" "$m/$name-g$r.gguf" --episodes 30 --exit-threshold 1.0 --game "$GAMES" --json "$run/g$r/arena.json"
python3 -I scripts/night-gate.py "$run/g$r/arena.json" || true
