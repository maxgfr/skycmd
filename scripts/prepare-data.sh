#!/usr/bin/env bash
# Every dataset an s3/s6 run reads, from scratch: Hugging Face decisions and MLM text (data/text,
# and data/text-open from Dolma 3 and TxT360, which only s6 reads), the game oracle data (doom
# included), the GLM teacher data when a teacher endpoint is set, the s2/s3 tokenizer with [NUM]
# and the numeric curriculum. Steps whose output exists are skipped.
# Usage: scripts/prepare-data.sh
#   FORCE=1     rebuild every output (the Hugging Face cache data/hf is kept)
#   REFRESH=1   also empty data/hf first, so every source is downloaded again
#   SKY_TEACHER_URL / SKY_TEACHER_KEY   regenerate the GLM teacher data (drone orders and
#               question paraphrases); without them those two sources are skipped
set -euo pipefail
export DEVELOPER_DIR=${DEVELOPER_DIR:-/Applications/Xcode.app/Contents/Developer}
export CARGO_TARGET_DIR=${CARGO_TARGET_DIR:-$PWD/target/s3-train}
cargo build -q --release -p sky-data -p sky-games -p sky-train
B=$CARGO_TARGET_DIR/release
D=$B/sky-data
force=${FORCE:-0}

if [ "${REFRESH:-0}" = 1 ]; then
  echo "== REFRESH=1: emptying the download cache data/hf"
  rm -rf data/hf
  force=1
fi

# step <output> <label> <command...>: runs the command unless <output> exists (FORCE=1 reruns).
step() {
  local out=$1 label=$2
  shift 2
  if [ "$force" != 1 ] && [ -e "$out" ]; then
    echo "== $label: $out exists, skipped"
    return
  fi
  echo "== $label"
  "$@"
}

teacher=0
[ -n "${SKY_TEACHER_URL:-}" ] && [ -n "${SKY_TEACHER_KEY:-}" ] && teacher=1

if [ "$teacher" = 1 ]; then
  step data/teacher/orders_decisions.jsonl "GLM teacher: 3,000 civil drone orders" \
    "$D" teacher drone-orders --n 3000
fi
step data/decisions/.done "decisions (Hugging Face)" sh -c "'$D' fetch-decisions && touch data/decisions/.done"
step data/text/.done "MLM text (Hugging Face)" sh -c "'$D' fetch-text && touch data/text/.done"
# Streamed: about 450 MB are read for the 150 MB kept (Europarl French is 1 document in 20).
step data/text-open/.done "open MLM text (Dolma 3, TxT360)" sh -c \
  "'$D' fetch-text --set open && touch data/text-open/.done"
# `--game all` writes doom too (60k): the 100k doom run comes after it, in the same step.
step data/games/.done "game oracle data (doom 100k)" sh -c \
  "'$B/sky-games-gen' --game all --n 60000 --seed 1 \
   && '$B/sky-games-gen' --game doom --n 100000 --seed 1 --out data/games/doom.jsonl \
   && touch data/games/.done"
if [ "$teacher" = 1 ]; then
  step data/teacher/questions.json "questions to paraphrase" "$D" teacher export-questions
  step data/teacher/paraphrases.jsonl "GLM teacher: EN/FR question paraphrases" \
    "$D" teacher paraphrase-questions
  step data/decisions/paraphrased.jsonl "paraphrased records (40k)" "$D" augment-paraphrases
else
  echo "== GLM teacher data skipped: set SKY_TEACHER_URL and SKY_TEACHER_KEY to regenerate it"
fi
step data/tok/tok-s2-1024.json "s3/s6-pico tokenizer (1024, [NUM])" \
  "$D" train-tokenizer --vocab 1024 --num
step data/numeric/numeric.jsonl "numeric curriculum (100k)" \
  "$B/sky-train" s2-numeric --n 100000 --out data/numeric/numeric.jsonl

echo "== data ready"
du -sh data/decisions data/text data/games data/numeric data/tok 2>/dev/null || true
