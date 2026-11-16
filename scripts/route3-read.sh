#!/usr/bin/env bash
# s6-nano v3 (docs/s6-nano3-prereg.md): the one reading of the fresh files, in the order of
# amendment 2: the baselines and the model on every fresh file, the fused reference of T2-tools,
# the WASM bench of the engine against it (G4), then the gate.
#
#   ROUTE_DATA=../skycmd-data/s6-route MODEL=<v2's evaluated file> EMBED=<the embedding GGUF> \
#     scripts/route3-read.sh
#
# Refused by scripts/route-eval.sh until runs/s6-nano3/FRESH-OPEN and the recorded hashes exist.
set -euo pipefail
cd "$(dirname "$0")/.."
data=${ROUTE_DATA:?ROUTE_DATA}
model=${MODEL:?MODEL}
embed=${EMBED:-models/s6-nano3/skycmd-s6-route-embed.gguf}
d=$data/eval/fresh/model
slates=$data/build/fresh/f_tools.jsonl
bench=crates/sky-web/bench/route3-node.mjs

scripts/route-eval.sh fresh baselines model

nice -n 19 python3 -I scripts/s6-route3-reference.py --model "$d/f_tools.jsonl" \
  --embedding "$data/eval/fresh/potion-cos-4bit/f_tools.jsonl" --out "$d/f_tools.fused.jsonl" >/dev/null
nice -n 19 python3 -I scripts/s6-route3-reference.py --model "$d/f_tools.exit0.jsonl" \
  --embedding "$data/eval/fresh/potion-cos-4bit/f_tools.jsonl" --out "$d/f_tools.fused-exit0.jsonl" >/dev/null

# a parity failure is kept in the JSON and read by the gate
nice -n 19 node "$bench" "$model" "$embed" "$slates" "$d/f_tools.fused.jsonl" --n 200 >"$d/wasm3.json" || true
nice -n 19 node "$bench" "$model" "$embed" "$slates" "$d/f_tools.fused.jsonl" --n 200 --relaxed >"$d/wasm3-relaxed.json" || true
nice -n 19 node "$bench" "$model" "$embed" "$slates" "$d/f_tools.fused-exit0.jsonl" --n 200 --fast >"$d/wasm3-fast.json" || true

scripts/route-eval.sh fresh gate
mkdir -p runs/s6-nano3/eval-fresh/latency
cp "$d/wasm3.json" "$d/wasm3-relaxed.json" "$d/wasm3-fast.json" "$d/native.json" runs/s6-nano3/eval-fresh/latency/
