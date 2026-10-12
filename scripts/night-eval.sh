#!/usr/bin/env bash
# Held-out accuracy of a GGUF on the test split without doom (data/decisions + flappy, rescue,
# dodge: 25,319 questions), with s2-eval under the night GPU lock (scripts/night-s3.sh).
# Usage: scripts/night-eval.sh <run checkpoint dir> <model.gguf> <report.md>
set -euo pipefail
cd "$(dirname "$0")/.."
export DEVELOPER_DIR=${DEVELOPER_DIR:-/Applications/Xcode.app/Contents/Developer}
T=${T:-$PWD/target/night-frozen/v3/sky-train}
LOCK=${GPU_LOCK:-/tmp/skycmd-night-gpu.lock}
until mkdir "$LOCK" 2>/dev/null; do sleep 3; done
echo "$$ eval $(date +%T) $2" >"$LOCK/owner"
trap 'rm -rf "$LOCK"' EXIT
"$T" s2-eval --run "$1" --records data/decisions,data/games/flappy.jsonl,data/games/rescue.jsonl,data/games/dodge.jsonl \
  --gguf "$2" --report "$3" --gguf-only
