#!/usr/bin/env bash
# DAgger rounds for s3: the ternary model plays, the oracle labels the states it reaches, and the
# model is fine-tuned fully ternary (QAT on every step) on games + DAgger + numeric + a little HF
# data. After each round the model plays 30 episodes per game at full depth (tau = 1).
# Usage: scripts/specialize-s3.sh <pico|nano> <rounds> [minutes-per-round]
#   START=r     resume at round r (INIT / MODEL default to round r-1's checkpoint / GGUF)
#   GAMES=...   games played (default flappy,rescue,dodge: the gate games)
set -euo pipefail
export DEVELOPER_DIR=${DEVELOPER_DIR:-/Applications/Xcode.app/Contents/Developer}
name=$1 rounds=${2:-3} minutes=${3:-8}
case "$name" in pico) vocab=1024 batch=128 ;; nano) vocab=2048 batch=128 ;; *) echo "unknown $name" >&2; exit 1 ;; esac
T=${T:-target/s3-train/release/sky-train}
A=${A:-target/s3-arena/release/sky-arena}
games=${GAMES:-flappy,rescue,dodge}
cfg=configs/s3-$name.json
start=${START:-1}
if [ "$start" -gt 1 ]; then
  prev=$((start - 1))
  init=${INIT:-runs/s3-$name-g$prev/decide/last}
  model=${MODEL:-models/s3-$name-g$prev/skycmd-s3-$name-g$prev-ternary.gguf}
else
  init=${INIT:-runs/s3-$name/decide/last}
  model=${MODEL:-models/s3-$name/skycmd-s3-$name-ternary.gguf}
fi
for r in $(seq "$start" "$rounds"); do
  run=runs/s3-$name-g$r
  dag=data/dagger-s3-$name/r$r
  echo "== s3-$name round $r: DAgger from $model"
  $A "$model" --dagger 15000 --seed "$r" --game "$games" --out "$dag"
  $T s2-decide --config "$cfg" --init "$init" --tokenizer data/tok/tok-s2-$vocab.json \
    --records data/decisions,data/games,data/numeric,"$dag" \
    --weights "games=0.5,r$r=0.35,numeric=0.05,decisions=0.1" \
    --batch "$batch" --minutes "$minutes" --qat-frac 1.0 --loop-dropout 0.25 --pad-to 32 --out "$run/decide"
  $T s2-calibrate --run "$run/decide/last" --records data/decisions,data/games,data/numeric
  mkdir -p "models/s3-$name-g$r"
  out=models/s3-$name-g$r/skycmd-s3-$name-g$r-ternary.gguf
  $T s2-export --run "$run/decide/last" --format ternary --out "$out"
  $A "$out" --episodes 30 --exit-threshold 1.0 --game "$games" --json "$run/arena.json"
  init=$run/decide/last
  model=$out
done
