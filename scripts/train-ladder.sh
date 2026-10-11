#!/usr/bin/env bash
# Train the skycmd-s1 size ladder end to end on Apple Silicon (mlx-rs).
# Usage: scripts/train-ladder.sh [pico|nano|micro ...]
set -euo pipefail
export DEVELOPER_DIR=${DEVELOPER_DIR:-/Applications/Xcode.app/Contents/Developer}
export CARGO_TARGET_DIR=${CARGO_TARGET_DIR:-$PWD/target/sky-train}
cargo build -q --release -p sky-train
T=$CARGO_TARGET_DIR/release/sky-train

# name  vocab  pretrain-min  decide-min  game-weight
ladder() {
  case "$1" in
    pico)  echo "1024 8 12 0.65" ;;
    nano)  echo "2048 12 20 0.65" ;;
    micro) echo "4096 15 25 0.5" ;;
  esac
}

for name in "${@:-pico nano micro}"; do
  read -r vocab pre dec gw <<<"$(ladder "$name")"
  run=runs/s1-$name
  tok=data/tok/tok-$vocab.json
  cfg=configs/s1-$name.json
  echo "== s1-$name: pretrain ${pre}m, decide ${dec}m, games weight $gw"
  $T pretrain --config "$cfg" --tokenizer "$tok" --text data/text --minutes "$pre" --out "$run/pretrain"
  $T decide --config "$cfg" --init "$run/pretrain/last" --tokenizer "$tok" \
    --records data/decisions,data/games --weights "games=$gw,decisions=$(echo "1 - $gw" | bc -l)" \
    --minutes "$dec" --out "$run/decide"
  $T calibrate --run "$run/decide/last" --records data/decisions,data/games
  $T export --run "$run" --out "models/s1-$name"
  $T eval --run "$run" --records data/decisions,data/games
done
