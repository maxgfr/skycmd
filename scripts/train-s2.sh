#!/usr/bin/env bash
# Train the skycmd-s2 ladder end to end on Apple Silicon (mlx-rs): numeric MLM
# pretraining, decision training with deep supervision and ternary QAT over the
# last 30%, calibration (f32 + ternary), GGUF export and the evaluation report.
# Usage: scripts/train-s2.sh [golden] [bench] [pico|nano|micro ...] [post-<name> ...]
#   golden       regenerate crates/sky-infer/tests/golden/s2{.json,-f32.gguf,-ternary.gguf}
#   bench        append s2 training throughput rows to BENCH.md
#   <name>       full run of s2-<name>
#   post-<name>  only calibration, export and evaluation of an existing run
set -euo pipefail
export DEVELOPER_DIR=${DEVELOPER_DIR:-/Applications/Xcode.app/Contents/Developer}
export CARGO_TARGET_DIR=${CARGO_TARGET_DIR:-$PWD/target/sky-train-s2}
cargo build -q --release -j 4 -p sky-train
T=$CARGO_TARGET_DIR/release/sky-train
RECORDS=data/decisions,data/games

# name  vocab  pretrain-min  decide-min  batch
ladder() {
  case "$1" in
    pico)  echo "1024 8 15 128" ;;
    nano)  echo "2048 12 20 128" ;;
    micro) echo "4096 15 25 64" ;;
    *) echo "unknown model $1" >&2; exit 1 ;;
  esac
}

train() {
  local name=$1 vocab pre dec batch
  read -r vocab pre dec batch <<<"$(ladder "$name")"
  local run=runs/s2-$name cfg=configs/s2-$name.json tok=data/tok/tok-s2-$vocab.json
  echo "== s2-$name: pretrain ${pre}m, decide ${dec}m (QAT last 30%), games weight 0.6"
  # --num-weight 0.05: with the SPEC weight 0.5 the number MSE (variance ~10
  # on web text) keeps the token CE at the unigram plateau (~4.8).
  $T s2-pretrain --config "$cfg" --tokenizer "$tok" --text data/text --minutes "$pre" --num-weight 0.05 \
    --out "$run/pretrain"
  $T s2-decide --config "$cfg" --init "$run/pretrain/init" --records "$RECORDS" \
    --weights games=0.6,decisions=0.4 --batch "$batch" --minutes "$dec" --qat-frac 0.3 --out "$run/decide"
}

post() {
  local name=$1 run=runs/s2-$1 m=models/s2-$1
  # f32 model = the snapshot taken when QAT starts (the QAT master weights are
  # tuned for ternary and lose accuracy in f32); ternary model = final weights.
  $T s2-calibrate --run "$run/decide/last" --records "$RECORDS"
  $T s2-calibrate --run "$run/decide/pre-qat" --records "$RECORDS"
  mkdir -p "$m"
  $T s2-export --run "$run/decide/pre-qat" --format f32 --out "$m/skycmd-s2-$name-f32.gguf"
  $T s2-export --run "$run/decide/last" --format ternary --out "$m/skycmd-s2-$name-ternary.gguf"
  $T s2-eval --run "$run/decide/last" --records "$RECORDS" \
    --gguf "$m/skycmd-s2-$name-f32.gguf,$m/skycmd-s2-$name-ternary.gguf" --report "$run/report.md"
}

names=("$@")
[ ${#names[@]} -eq 0 ] && names=(golden bench pico nano)
for name in "${names[@]}"; do
  case "$name" in
    golden) $T s2-golden --out crates/sky-infer/tests/golden ;;
    bench)
      for c in pico nano micro; do
        for p in f32 bf16; do
          $T s2-bench --config "configs/s2-$c.json" --precision "$p" --qat
        done
      done ;;
    post-*) post "${name#post-}" ;;
    *) train "$name" && post "$name" ;;
  esac
done
