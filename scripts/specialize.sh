#!/usr/bin/env bash
# Game specialization with DAgger rounds: the model plays, the oracle labels the states it
# reaches, and the model is fine-tuned on games + DAgger data (+ a little HF data to keep
# general decisions alive).
# Usage: scripts/specialize.sh <pico|nano|micro> <rounds> <minutes-per-round>
set -euo pipefail
export DEVELOPER_DIR=${DEVELOPER_DIR:-/Applications/Xcode.app/Contents/Developer}
name=$1 rounds=${2:-2} minutes=${3:-6}
case "$name" in pico) vocab=1024 ;; nano) vocab=2048 ;; micro) vocab=4096 ;; esac
T=target/sky-train/release/sky-train
A=target/sky-arena/release/sky-arena
CARGO_TARGET_DIR=$PWD/target/sky-arena cargo build -q --release -p sky-arena
init=runs/s1-$name/decide/last
model=models/s1-$name
for r in $(seq 1 "$rounds"); do
  run=runs/s1-$name-g$r
  dag=data/dagger-$name/r$r
  echo "== s1-$name round $r: DAgger from $model"
  $A "$model" --dagger 15000 --seed "$r" --out "$dag"
  $T decide --config configs/s1-$name.json --init "$init" --tokenizer data/tok/tok-$vocab.json \
    --records data/decisions,data/games,"$dag" --weights "games=0.5,r$r=0.4,decisions=0.1" \
    --minutes "$minutes" --out "$run/decide"
  $T calibrate --run "$run/decide/last" --records data/decisions,data/games
  $T export --run "$run" --out "models/s1-$name-g$r"
  $A "models/s1-$name-g$r" --episodes 30 --json "$run/arena.json"
  init=$run/decide/last
  model=models/s1-$name-g$r
done
