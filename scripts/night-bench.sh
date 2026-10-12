#!/usr/bin/env bash
# Native latency of the night candidates (docs/night-experiments.md): sky-bench median per
# decision (one question, static cache on, the file's own tau) on the Flappy and Doom presets,
# best of 3 rounds interleaved across the models so they share the same machine load.
# Usage: scripts/night-bench.sh <outdir-prefix>=<model.gguf> ...
#   e.g. scripts/night-bench.sh runs/night-s3-femto/bench=models/candidates/s3-femto/s3-femto.gguf
# Writes <prefix>-flappy.json and <prefix>-doom.json (the best round's sky-bench --json line).
set -euo pipefail
cd "$(dirname "$0")/.."
B=${B:-$PWD/target/night-frozen/v2/sky-bench}
ITERS=${ITERS:-4000}
ROUNDS=${ROUNDS:-3}
tmp=$(mktemp -d)
for r in $(seq 1 "$ROUNDS"); do
  for pair in "$@"; do
    prefix=${pair%%=*} model=${pair#*=}
    for game in flappy doom; do
      key=$(echo "$prefix-$game" | tr '/' '_')
      "$B" "$model" --questions 1 --iters "$ITERS" --game "$game" --json >"$tmp/$key.$r"
    done
  done
done
for pair in "$@"; do
  prefix=${pair%%=*}
  mkdir -p "$(dirname "$prefix")"
  for game in flappy doom; do
    key=$(echo "$prefix-$game" | tr '/' '_')
    python3 -I - "$prefix-$game.json" "$tmp/$key".* <<'EOF'
import json, sys
runs = [json.loads(open(p).read().strip().splitlines()[-1]) for p in sys.argv[2:]]
best = min(runs, key=lambda j: j["median_us"])
best["rounds_median_us"] = [j["median_us"] for j in runs]
json.dump(best, open(sys.argv[1], "w"))
print(f"{sys.argv[1]}: {best['median_us']:.1f} us (rounds {best['rounds_median_us']}), layers {best['mean_layers_used']}")
EOF
  done
done
rm -rf "$tmp"
