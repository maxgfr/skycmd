#!/usr/bin/env bash
# Train s6-secrets-pico (docs/s6-secrets-prereg.md): the s6 architecture of s6-pico with a
# 320-token state window, specialized in API-key and secret detection. The API-key jev
# (`sky-data secret-jev`) and the rule scanner's score on its bench come first, then numeric MLM
# pretraining, decision training with QAT on data/secrets only, calibration, ternary export and
# evaluation on secret_bench (contexts the training never shows).
# Usage: scripts/train-secrets.sh [data] [train] [eval]   (default: all three, in order)
#   OPT="..."       optimizer flags of every phase (as scripts/train-s6.sh)
#   PRETRAIN="..."  pretraining flags
#   DECIDE="..."    decision flags (--exit-kd 0.5 --ema 0.999 ...); with --ema the EMA weights
#                   are calibrated and exported
#   PRETRAIN_MIN=8  DECIDE_MIN=15
set -euo pipefail
export DEVELOPER_DIR=${DEVELOPER_DIR:-/Applications/Xcode.app/Contents/Developer}
export CARGO_TARGET_DIR=${CARGO_TARGET_DIR:-$PWD/target/s3-train}
cargo build -q --release -p sky-data -p sky-train
B=$CARGO_TARGET_DIR/release
T=$B/sky-train
cfg=configs/s6-secrets-pico.json
read -r -a opt <<<"${OPT:-}"
read -r -a pre <<<"${PRETRAIN:-}"
read -r -a dec <<<"${DECIDE:-}"
records=data/secrets
run=runs/s6-secrets-pico
m=models/s6-secrets-pico
ckpt=last
[[ " ${DECIDE:-} " == *" --ema "* ]] && ckpt=last-ema

data() {
  echo "== API-key jev: data/secrets and the rule scanner's score on secret_bench"
  "$B/sky-data" secret-jev --out "$records"
  mkdir -p "$run"
  "$B/sky-data" secret-baseline --data "$records/secret_bench.jsonl" | tee "$run/baseline.txt"
}

train() {
  [ -f data/tok/tok-s2-1024.json ] && [ -d data/text ] && [ -d data/numeric ] || {
    echo "data/tok, data/text and data/numeric come from scripts/prepare-data.sh" >&2
    exit 1
  }
  echo "== s6-secrets-pico: pretrain ${PRETRAIN_MIN:-8}m (sequences of 256 tokens)"
  $T s2-pretrain --config "$cfg" --tokenizer data/tok/tok-s2-1024.json --text data/text \
    --minutes "${PRETRAIN_MIN:-8}" --seq 256 --batch 64 --num-weight 0.05 \
    --numeric data/numeric --numeric-weight 0.1 --out "$run/pretrain" \
    ${opt[@]+"${opt[@]}"} ${pre[@]+"${pre[@]}"}
  echo "== s6-secrets-pico: decide ${DECIDE_MIN:-15}m on $records (QAT after a 10% float warm-up)"
  rm -rf "$run/decide"
  $T s2-decide --config "$cfg" --init "$run/pretrain/init" --records "$records" --batch 64 \
    --minutes "${DECIDE_MIN:-15}" --qat-frac 0.9 --loop-dropout 0.25 --pad-to 32 \
    --out "$run/decide" ${opt[@]+"${opt[@]}"} ${dec[@]+"${dec[@]}"}
}

evaluate() {
  mkdir -p "$m"
  $T s2-calibrate --run "$run/decide/$ckpt" --records "$records"
  $T s2-export --run "$run/decide/$ckpt" --format ternary --out "$m/skycmd-s6-secrets-pico-ternary.gguf"
  $T s2-eval --run "$run/decide/$ckpt" --records "$records" \
    --gguf "$m/skycmd-s6-secrets-pico-ternary.gguf" --report "$run/report.md"
  echo "== rule scanner (the bar of docs/s6-secrets-prereg.md):"
  cat "$run/baseline.txt"
}

steps=("$@")
[ ${#steps[@]} -eq 0 ] && steps=(data train eval)
for s in "${steps[@]}"; do
  case "$s" in
    data) data ;;
    train) train ;;
    eval) evaluate ;;
    *) echo "unknown step $s (data, train, eval)" >&2; exit 2 ;;
  esac
done
