#!/usr/bin/env bash
# DAgger rounds for s2: the ternary model plays, the oracle labels the states it reaches, and
# the model is fine-tuned fully ternary (QAT on every step) on games + DAgger + a little HF data.
# Usage: scripts/specialize-s2.sh <pico|nano> <rounds> <minutes-per-round>
set -euo pipefail
export DEVELOPER_DIR=${DEVELOPER_DIR:-/Applications/Xcode.app/Contents/Developer}
name=$1 rounds=${2:-2} minutes=${3:-8}
case "$name" in pico) vocab=1024 batch=128 ;; nano) vocab=2048 batch=128 ;; micro) vocab=4096 batch=64 ;; esac
T=target/sky-train-s2/release/sky-train
A=target/sky-infer-s2/release/sky-arena
[ -x "$A" ] || CARGO_TARGET_DIR="${A%/release/sky-arena}" cargo build -q --release -p sky-arena
cfg=configs/s2-$name.json
init=${INIT:-runs/s2-$name/decide/last}
model=${MODEL:-models/s2-$name/skycmd-s2-$name-ternary.gguf}
for r in $(seq "${START:-1}" "$rounds"); do
  run=runs/s2-$name-g$r
  dag=data/dagger-s2-$name/r$r
  echo "== s2-$name round $r: DAgger from $model"
  $A "$model" --dagger 15000 --seed "$r" --out "$dag"
  $T s2-decide --config "$cfg" --init "$init" --tokenizer data/tok/tok-s2-$vocab.json \
    --records data/decisions,data/games,"$dag" --weights "games=0.5,r$r=0.4,decisions=0.1" \
    --batch "$batch" --minutes "$minutes" --qat-frac 1.0 --out "$run/decide"
  $T s2-calibrate --run "$run/decide/last" --records data/decisions,data/games
  mkdir -p "models/s2-$name-g$r"
  out=models/s2-$name-g$r/skycmd-s2-$name-g$r-ternary.gguf
  $T s2-export --run "$run/decide/last" --format ternary --out "$out"
  $A "$out" --episodes 30 --json "$run/arena.json"
  init=$run/decide/last
  model=$out
done
