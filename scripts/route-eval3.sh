#!/usr/bin/env bash
# The evaluation of s6-nano v3 (docs/s6-nano3-prereg.md, sections 4 to 6): v2's model file and
# every baseline of v2 on the fresh files, then scripts/s6-route3-gate.py (the fused system, the
# gate, the readings). Everything at `nice -n 19`; the baselines on the CPU, in the pinned uv
# environment, with the values fitted for v2 (never refitted).
#
# Usage: scripts/route-eval.sh <fresh|rehearse3> [step ...]   (or this script directly)
#   fresh      the fresh files <build>/fresh/f_{tools,tools_masked,metatool,clinc,massive}.jsonl
#   rehearse3  the same chain on the validation slates v_tools and v_tools_masked, everything
#              written under <data>/eval/fresh/rehearsal (no fresh file is read)
#   steps (default: all, in this order)
#     baselines  every baseline of v2 scored on the split's files (scripts/route-baselines.py score)
#     model      the model (MODEL) by sky-eval on the model's copy of each file (each state with
#                its BM25 line, amendment 1 of v2), on a copy with exit thresholds 1: one dump at
#                full depth (--exit-threshold 1) and one at the first exit (--exit-threshold 0)
#                per file; the native latency on the main file (200 slates) and the native dump
#                the WASM bench reads
#     gate       scripts/s6-route3-gate.py: results.md and gate.json
#
#   MODEL=models/s6-nano2/skycmd-s6-route-nano-ternary.gguf   v2's evaluated file (2,418,912 bytes)
#   ROUTE_DATA=../skycmd-data/s6-route  BUILD=$ROUTE_DATA/build
#   WASM3=<root>/model/wasm3.json   the WASM bench's JSON for G4 (absent: G4 not measured)
#
# THE FRESH SPLIT IS REFUSED unless runs/s6-nano3/FRESH-OPEN exists (the owner creates it after the
# hashes and the count amendment are committed) and every fresh file has the sha256 that
# runs/s6-nano3/test-files.json records.
#
# Outputs: <data>/eval/fresh/<system>/<file>.jsonl (dumps), <data>/eval/fresh/model/,
# <data>/eval/fresh/gate/ (results.md, gate.json), copied to runs/s6-nano3/eval-fresh/ (numbers
# only); for rehearse3 the same under <data>/eval/fresh/rehearsal, copied to
# runs/s6-nano3/rehearsal-val/.
set -euo pipefail
cd "$(dirname "$0")/.."
export DEVELOPER_DIR=${DEVELOPER_DIR:-/Applications/Xcode.app/Contents/Developer}
export CARGO_TARGET_DIR=${CARGO_TARGET_DIR:-$PWD/target/s3-train}

mode=${1:-}
case "$mode" in
  fresh | rehearse3) ;;
  *) echo "usage: scripts/route-eval.sh <fresh|rehearse3> [baselines model gate]" >&2; exit 2 ;;
esac
shift
steps=("$@")
[ ${#steps[@]} -eq 0 ] && steps=(baselines model gate)

data=${ROUTE_DATA:-../skycmd-data/s6-route}
export ROUTE_DATA=$data
export BUILD=${BUILD:-$data/build}
model=${MODEL:-models/s6-nano2/skycmd-s6-route-nano-ternary.gguf}
model_bytes=2418912
py=$data/envs/route/bin/python
B=$CARGO_TARGET_DIR/release
systems=(potion-cos potion-cos-4bit static-ft pair-lr bm25 minilm-cos toolrouter name-prior)

if [ "$mode" = fresh ]; then
  split="fresh"
  root=$data/eval/fresh
  src=$BUILD/fresh
  files=(f_tools f_tools_masked f_metatool f_clinc f_massive)
  kept=runs/s6-nano3/eval-fresh
else
  split="val"
  root=$data/eval/fresh/rehearsal
  src=$BUILD/val
  files=(v_tools v_tools_masked)
  kept=runs/s6-nano3/rehearsal-val
fi
main=${files[0]}
wasm3=${WASM3:-$root/model/wasm3.json}

# The guard, before anything is read: FRESH-OPEN and the recorded hashes (scripts/route/common.py)
nice -n 19 python3 -I scripts/s6-route3-gate.py guard --split "$split" || exit 3
if [ ! -f "$model" ] || [ "$(wc -c <"$model" | tr -d ' ')" != "$model_bytes" ]; then
  echo "MODEL=$model is not v2's evaluated file ($model_bytes bytes)" >&2
  exit 3
fi

baselines() {
  local s extra=()
  [ "$split" = val ] && extra=(--out-root "$root" --files "$(IFS=,; echo "${files[*]}")")
  for s in "${systems[@]}"; do
    if [ ! -f "$data/eval/fitted/$s/params.json" ]; then
      echo "== $s: no fitted values in $data/eval/fitted/$s: refused (v3 reads v2's fitted values)" >&2
      exit 3
    fi
    echo "== score $s on $split (v2's fitted values)"
    nice -n 19 "$py" -I scripts/route-baselines.py score --system "$s" --split "$split" ${extra[@]+"${extra[@]}"}
  done
}

model_() {
  nice -n 19 cargo build -q --release -p sky-data -p sky-infer -p sky-gguf
  local f d=$root/model r=$root/render
  mkdir -p "$d" "$r"
  shasum -a 256 "$model" | awk -v b="$model_bytes" '{printf "{\"model\": \"%s\", \"sha256\": \"%s\", \"bytes\": %d}\n", $2, $1, b}' >"$d/model.json"
  local full=$d/full-depth.gguf
  nice -n 19 "$B/gguf-set" "$model" "$full" skycmd-s2.exit_threshold.choice=1 skycmd-s2.exit_threshold.score=1 \
    skycmd-s2.exit_threshold.noul=1 >/dev/null
  for f in "${files[@]}"; do
    echo "== model on $f (full depth and first exit)"
    nice -n 19 "$B/sky-data" route render --in "$src/$f.jsonl" --out "$r/$f.jsonl" >/dev/null
    nice -n 19 "$B/sky-eval" "$full" --records "$r/$f.jsonl" --split "$split" --max 100000 --exit-threshold 1.0 \
      --dump-probs "$d/$f.jsonl" --json "$d/$f.eval.json" >/dev/null
    nice -n 19 "$B/sky-eval" "$full" --records "$r/$f.jsonl" --split "$split" --max 100000 --exit-threshold 0 \
      --dump-probs "$d/$f.exit0.jsonl" --json "$d/$f.exit0.eval.json" >/dev/null
  done
  # the native latency on the main file, and the dump the WASM bench reads
  nice -n 19 "$B/sky-eval" "$full" --records "$r/$main.jsonl" --split "$split" --max 200 \
    --exit-threshold 1.0 --latency --json "$d/native.json" --dump "$d/$main.native-dump.jsonl" >/dev/null
}

gate() {
  local out=$root/gate
  nice -n 19 python3 -I scripts/s6-route3-gate.py gate --split "$split" --root "$root" --wasm "$wasm3" \
    --native "$root/model/native.json" --out-dir "$out" || {
    [ -s "$out/results.md" ] || exit 1
    echo "== gate not passed (or a rehearsal that does not reproduce runs/s6-nano3/fit.json): see $out/results.md"
  }
  mkdir -p "$kept"
  cp "$out/results.md" "$out/gate.json" "$kept/"
  echo "wrote $out/results.md (copied to $kept)"
}

for s in "${steps[@]}"; do
  case "$s" in
    baselines) baselines ;;
    model) model_ ;;
    gate) gate ;;
    *) echo "unknown step $s (baselines model gate)" >&2; exit 2 ;;
  esac
done
