#!/usr/bin/env bash
# Train the skycmd-s3 ladder (docs/SPEC-S3.md) on Apple Silicon (mlx-rs): s2 with value-fused
# number tokens and a weight-shared looped state encoder. Numeric MLM pretraining (web text +
# 10% numeric curriculum), decision training with deep supervision over every exit, loop
# dropout and ternary QAT after a 10% float warm-up, calibration, GGUF export and evaluation.
# Usage: scripts/train-s3.sh [golden] [numeric] [pico|nano ...] [post-<name> ...]
#   golden       regenerate crates/sky-infer/tests/golden/s3{.json,-f32.gguf,-ternary.gguf}
#   numeric      regenerate data/numeric/numeric.jsonl (100k records)
#   <name>       full run of s3-<name>
#   decide-<name> decision training from the existing pretraining run, then post
#   temporal-<name> SPEC-S3 §7 fine-tune of the last DAgger round (ROUND=3) with the latent
#                carried across frames (configs/s3-<name>-temporal.json, data/seq sequences)
#   post-<name>  only calibration, export and evaluation of an existing run
set -euo pipefail
export DEVELOPER_DIR=${DEVELOPER_DIR:-/Applications/Xcode.app/Contents/Developer}
export CARGO_TARGET_DIR=${CARGO_TARGET_DIR:-$PWD/target/s3-train}
cargo build -q --release -j 4 -p sky-train
T=$CARGO_TARGET_DIR/release/sky-train
RECORDS=data/decisions,data/games,data/numeric

# name  vocab  pretrain-min  decide-min  batch
ladder() {
  case "$1" in
    pico) echo "1024 8 15 128" ;;
    nano) echo "2048 10 18 128" ;;
    *) echo "unknown model $1" >&2; exit 1 ;;
  esac
}

train() {
  local name=$1 vocab pre dec batch
  read -r vocab pre dec batch <<<"$(ladder "$name")"
  local run=runs/s3-$name cfg=configs/s3-$name.json tok=data/tok/tok-s2-$vocab.json
  [ -f data/numeric/numeric.jsonl ] || $T s2-numeric --n 100000 --out data/numeric/numeric.jsonl
  echo "== s3-$name: pretrain ${pre}m (numeric MLM 10%)"
  # --num-weight 0.05 as for s2: the SPEC weight 0.5 keeps the token CE at the unigram plateau.
  $T s2-pretrain --config "$cfg" --tokenizer "$tok" --text data/text --minutes "$pre" --num-weight 0.05 \
    --numeric data/numeric --numeric-weight 0.1 --out "$run/pretrain"
  decide "$name"
}

decide() {
  local name=$1 vocab pre dec batch
  read -r vocab pre dec batch <<<"$(ladder "$name")"
  local run=runs/s3-$name cfg=configs/s3-$name.json
  echo "== s3-$name: decide ${dec}m (QAT after a 10% float warm-up, loop dropout 0.25)"
  # --pad-to 32: coarse batch shapes, so few graphs get compiled (compiling dominated otherwise).
  rm -rf "$run/decide"
  $T s2-decide --config "$cfg" --init "$run/pretrain/init" --records "$RECORDS" \
    --weights games=0.6,numeric=0.1,decisions=0.3 --batch "$batch" --minutes "$dec" \
    --qat-frac 0.9 --loop-dropout 0.25 --pad-to 32 --out "$run/decide"
}

post() {
  local name=$1 run=runs/s3-$1 m=models/s3-$1
  # f32 model = the snapshot taken when QAT starts; ternary model = final weights.
  $T s2-calibrate --run "$run/decide/last" --records "$RECORDS"
  $T s2-calibrate --run "$run/decide/pre-qat" --records "$RECORDS"
  mkdir -p "$m"
  $T s2-export --run "$run/decide/pre-qat" --format f32 --out "$m/skycmd-s3-$name-f32.gguf"
  $T s2-export --run "$run/decide/last" --format ternary --out "$m/skycmd-s3-$name-ternary.gguf"
  $T s2-eval --run "$run/decide/last" --records "$RECORDS" \
    --gguf "$m/skycmd-s3-$name-ternary.gguf" --report "$run/report.md"
}

temporal() {
  local name=$1 vocab pre dec batch r=${ROUND:-3}
  read -r vocab pre dec batch <<<"$(ladder "$name")"
  local run=runs/s3-$name-temporal m=models/s3-$name-temporal
  [ -d data/seq ] || $T s2-seqgen --games flappy,rescue,dodge --n-seq 15000 --k 4 --seed 11 --out data/seq
  echo "== s3-$name-temporal: fine-tune of round $r with temporal carry (50% sequence steps, carry drop 0.3)"
  rm -rf "$run/decide"
  $T s2-decide --config "configs/s3-$name-temporal.json" --init "runs/s3-$name-g$r/decide/last" \
    --tokenizer "data/tok/tok-s2-$vocab.json" \
    --records data/decisions,data/games,data/numeric,"data/dagger-s3-$name/r$r" \
    --weights "games=0.5,r$r=0.35,numeric=0.05,decisions=0.1" --batch "$batch" --minutes "${MINUTES:-10}" \
    --qat-frac 1.0 --loop-dropout 0.25 --pad-to 32 --seq-records data/seq --seq-frac 0.5 --carry-drop 0.3 \
    --out "$run/decide"
  $T s2-calibrate --run "$run/decide/last" --records "$RECORDS"
  mkdir -p "$m"
  $T s2-export --run "$run/decide/last" --format ternary --out "$m/skycmd-s3-$name-temporal-ternary.gguf"
}

names=("$@")
[ ${#names[@]} -eq 0 ] && names=(golden pico)
for name in "${names[@]}"; do
  case "$name" in
    golden) $T s2-golden --config configs/s3-pico.json --out crates/sky-infer/tests/golden ;;
    numeric) $T s2-numeric --n 100000 --out data/numeric/numeric.jsonl ;;
    post-*) post "${name#post-}" ;;
    decide-*) decide "${name#decide-}" && post "${name#decide-}" ;;
    temporal-*) temporal "${name#temporal-}" ;;
    *) train "$name" && post "$name" ;;
  esac
done
