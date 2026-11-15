#!/usr/bin/env bash
# The evaluation of s6-route-nano (docs/s6-nano2-prereg.md, sections 6 to 8): the model, every
# baseline through the same scorer, the reading checks, latency natively and in WASM, the gate.
# Everything at `nice -n 19`; the baselines on the CPU, in the pinned uv environment.
#
# Usage: scripts/route-eval.sh <val|test> [step ...]
#   steps (default: all, in this order)
#     env        the uv environment of scripts/route/env (its lock), in <data>/envs/route
#     fit        (validation only) every baseline fitted on the training records and the
#                validation slates (scripts/route-baselines.py fit), B = the model file's bytes
#     baselines  every baseline scored on the split's slate files
#     model      the model (MODEL=...) by sky-eval at full depth on a copy with exit thresholds 1:
#                one dump per slate file; exit 0 and the native latency on T-tools (or v_tools)
#     v1         s6-nano v1 (V1=site/public/models/s6-nano.gguf), full depth, same slates
#     wasm       the WASM latency and parity of the model on 200 requests of T-tools (v_tools on
#                validation): node crates/sky-web/bench/parity-node.mjs --router, SIMD128 and
#                relaxed builds
#     gate       scripts/s6-route-gate.py: results.md and gate.json
#
#   MODEL=models/s6-nano2/skycmd-s6-route-nano-ternary.gguf   the trained ternary GGUF; its
#                metadata holds skycmd-s2.route.tau_ask, stored before any test file is read
#   SEAT=model   the system the gate judges (SEAT=potion-cos etc. rehearses it without a model)
#   ROUTE_DATA=../skycmd-data/s6-route  BUILD=$ROUTE_DATA/build
#   BUDGET       the byte budget of static-ft and pair-lr (default: MODEL's size, else 715136)
#
# THE TEST SPLIT IS REFUSED unless runs/s6-nano2/TEST-OPEN exists (the owner creates it by hand
# after training and calibration), MODEL holds tau_ask, and every test file has the sha256 that
# runs/s6-nano2/test-files.json records. The baselines are fitted on validation only.
#
# Outputs: <data>/eval/<split>/<system>/<file>.jsonl (dumps), <data>/eval/<split>/gate/
# (results.md, gate.json), copied to runs/s6-nano2/eval-<split>/ (numbers only).
set -euo pipefail
cd "$(dirname "$0")/.."
export DEVELOPER_DIR=${DEVELOPER_DIR:-/Applications/Xcode.app/Contents/Developer}
export CARGO_TARGET_DIR=${CARGO_TARGET_DIR:-$PWD/target/s3-train}

split=${1:-}
case "$split" in
  val | test) ;;
  *) echo "usage: scripts/route-eval.sh <val|test> [env fit baselines model v1 wasm gate]" >&2; exit 2 ;;
esac
shift
steps=("$@")
[ ${#steps[@]} -eq 0 ] && steps=(env fit baselines model v1 wasm gate)

data=${ROUTE_DATA:-../skycmd-data/s6-route}
export ROUTE_DATA=$data
export BUILD=${BUILD:-$data/build}
ev=$data/eval
model=${MODEL:-}
seat=${SEAT:-model}
v1=${V1:-site/public/models/s6-nano.gguf}
py=$data/envs/route/bin/python
B=$CARGO_TARGET_DIR/release
pre=v
[ "$split" = test ] && pre=t
files_val=(v_tools v_tools_masked v_tools_pairs v_tools_swapped)
files_test=(t_tools t_tools_masked t_tools_pairs t_tools_swapped t_metatool t_clinc t_toolrouting t_bfcl t_when2call t_massive)
if [ "$split" = test ]; then files=("${files_test[@]}"); else files=("${files_val[@]}"); fi

# The guard, before anything is read: the test needs TEST-OPEN, tau_ask in MODEL and the
# recorded hashes of every test file (scripts/route/common.py).
guard() {
  nice -n 19 python3 -I - "$split" "$model" "${files[@]}" <<'EOF'
import os, sys
sys.path.insert(0, os.path.join(os.getcwd(), "scripts", "route"))
import common
split, model, *files = sys.argv[1:]
try:
    for f in files:
        common.slate_file(split, f, model or None)
except common.Refused as e:
    sys.exit(f"refused: {e}")
EOF
}
guard || exit 3

budget=${BUDGET:-}
if [ -z "$budget" ]; then
  if [ -n "$model" ] && [ -f "$model" ]; then budget=$(wc -c <"$model" | tr -d ' '); else budget=715136; fi
fi

env_() {
  UV_PROJECT_ENVIRONMENT=$data/envs/route UV_PYTHON_PREFERENCE=only-managed \
    nice -n 19 uv sync --frozen --no-dev --project scripts/route/env -q
}

fit() {
  if [ "$split" != val ]; then
    echo "== fit: on validation only (scripts/route-eval.sh val fit)"
    return
  fi
  local s
  for s in potion-cos potion-cos-4bit static-ft pair-lr bm25 minilm-cos toolrouter name-prior; do
    echo "== fit $s (B = $budget bytes)"
    nice -n 19 "$py" -I scripts/route-baselines.py fit --system "$s" --budget "$budget"
  done
}

baselines() {
  local s
  for s in potion-cos potion-cos-4bit static-ft pair-lr bm25 minilm-cos toolrouter name-prior; do
    if [ ! -f "$ev/fitted/$s/params.json" ]; then
      echo "== $s: not fitted (scripts/route-eval.sh val fit): skipped"
      continue
    fi
    echo "== score $s on $split"
    nice -n 19 "$py" -I scripts/route-baselines.py score --system "$s" --split "$split" ${model:+--model "$model"}
  done
}

# sky_eval <gguf> <system>: one --dump-probs file per slate file, at full depth
sky_eval() {
  local g=$1 name=$2 f dir=val
  [ "$split" = test ] && dir="test"
  mkdir -p "$ev/$split/$name"
  local full=$ev/$split/$name/full-depth.gguf
  nice -n 19 "$B/gguf-set" "$g" "$full" skycmd-s2.exit_threshold.choice=1 skycmd-s2.exit_threshold.score=1 \
    skycmd-s2.exit_threshold.noul=1 >/dev/null
  for f in "${files[@]}"; do
    nice -n 19 "$B/sky-eval" "$full" --records "$BUILD/$dir/$f.jsonl" \
      --split "$split" --max 100000 --exit-threshold 1.0 --dump-probs "$ev/$split/$name/$f.jsonl" \
      --json "$ev/$split/$name/$f.eval.json" >/dev/null
  done
}

model_() {
  if [ -z "$model" ]; then
    echo "== model: no MODEL given: skipped"
    return
  fi
  sky_eval "$model" model
  # exit 0 (one block pass) and the native latency on the main file
  local main=$BUILD/val/v_tools.jsonl
  [ "$split" = test ] && main=$BUILD/test/t_tools.jsonl
  nice -n 19 "$B/sky-eval" "$ev/$split/model/full-depth.gguf" --records "$main" --split "$split" --max 100000 \
    --exit-threshold 0 --dump-probs "$ev/$split/model/${pre}_tools.exit0.jsonl" --json "$ev/$split/model/exit0.json" >/dev/null
  nice -n 19 "$B/sky-eval" "$ev/$split/model/full-depth.gguf" --records "$main" --split "$split" --max 200 \
    --exit-threshold 1.0 --latency --json "$ev/$split/model/native.json" \
    --dump "$ev/$split/model/${pre}_tools.native-dump.jsonl" >/dev/null
}

v1_() {
  [ -f "$v1" ] || { echo "== v1: no $v1: skipped"; return; }
  sky_eval "$v1" s6-nano-v1
}

wasm() {
  local d=$ev/$split/model
  if [ ! -f "$d/${pre}_tools.native-dump.jsonl" ]; then
    echo "== wasm: no native dump (the model step): skipped"
    return
  fi
  [ -d site/public/pkg ] || scripts/build-wasm.sh
  # a parity failure (exit 1) is kept in the JSON and read by the gate
  nice -n 19 node crates/sky-web/bench/parity-node.mjs "$d/full-depth.gguf" "$d/${pre}_tools.native-dump.jsonl" \
    --n 200 --router >"$d/wasm.json" || true
  nice -n 19 node crates/sky-web/bench/parity-node.mjs "$d/full-depth.gguf" "$d/${pre}_tools.native-dump.jsonl" \
    --n 200 --router --relaxed >"$d/wasm-relaxed.json" || true
}

gate() {
  local out=$ev/$split/gate
  [ "$seat" = model ] || out=$ev/$split/gate-seat-$seat
  nice -n 19 python3 -I scripts/s6-route-gate.py gate --eval "$ev" --split "$split" --seat "$seat" \
    ${model:+--model "$model"} --wasm "$ev/$split/model/wasm.json" --native "$ev/$split/model/native.json" \
    --out-dir "$out" || {
    [ -s "$out/results.md" ] || exit 1
    echo "== gate not passed (or a rehearsal): see $out/results.md"
  }
  local kept=runs/s6-nano2/eval-$split
  [ "$seat" = model ] || kept=$kept/seat-$seat
  mkdir -p "$kept"
  cp "$out/results.md" "$out/gate.json" "$kept/"
  echo "wrote $out/results.md (copied to $kept)"
}

[ "$split" = val ] || [ -n "$model" ] || { echo "the test needs MODEL" >&2; exit 3; }
for s in "${steps[@]}"; do
  case "$s" in
    env) env_ ;;
    fit) fit ;;
    baselines) baselines ;;
    model) model_ ;;
    v1) v1_ ;;
    wasm) wasm ;;
    gate) gate ;;
    *) echo "unknown step $s (env fit baselines model v1 wasm gate)" >&2; exit 2 ;;
  esac
done
