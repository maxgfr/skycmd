#!/usr/bin/env bash
# Teacher soft labels of s6-nano / s6-micro through a GLM endpoint (docs/s6-gen-prereg.md,
# amendment 1). The teacher runs remotely, so nothing heavy runs on this machine. The endpoint and
# the key come from the zai provider of ccs (Claude Code Switch) unless SKY_TEACHER_URL and
# SKY_TEACHER_KEY are set; the key is never printed.
# Usage: scripts/distill-glm.sh [count] [label] [mix]   (default: label mix)
#   count   eligible records and the calls a label run would make (no API call)
#   label   sky-data teacher label-decisions -> data/teacher/glm_labels.jsonl (resumable)
#   mix     sky-data teacher mix-labels -> data/distill/glm.jsonl and agreement.json
#   MODEL=glm-5.3-flash  PER_SOURCE=1000  BATCH=20  JOBS=4  MAX_CALLS=5000  ALPHA=0.5
#   DATA=data            where data/decisions lives
set -euo pipefail
export CARGO_TARGET_DIR=${CARGO_TARGET_DIR:-$PWD/target/s3-train}
cargo build -q --release -p sky-data
D=$CARGO_TARGET_DIR/release/sky-data
data=${DATA:-data}
labels=$data/teacher/glm_labels.jsonl
model=${MODEL:-glm-5.3-flash}
batch=${BATCH:-20}

endpoint() {
  if [ -z "${SKY_TEACHER_URL:-}" ]; then
    SKY_TEACHER_URL=$(ccs config get zai base_url)
    export SKY_TEACHER_URL
  fi
  if [ -z "${SKY_TEACHER_KEY:-}" ]; then
    SKY_TEACHER_KEY=$(ccs config get zai api_key)
    export SKY_TEACHER_KEY
  fi
  [ -n "$SKY_TEACHER_URL" ] && [ -n "$SKY_TEACHER_KEY" ] || {
    echo "no GLM endpoint: set SKY_TEACHER_URL / SKY_TEACHER_KEY, or configure ccs (ccs use zai)" >&2
    exit 1
  }
}

# label <max calls>
label() {
  endpoint
  "$D" teacher --model "$model" --max-calls "$1" --batch "$batch" label-decisions \
    --from "$data/decisions" --out "$labels" --per-source "${PER_SOURCE:-1000}" --jobs "${JOBS:-4}"
}

steps=("$@")
[ ${#steps[@]} -eq 0 ] && steps=(label mix)
for s in "${steps[@]}"; do
  case "$s" in
    count) label 0 ;;
    label) label "${MAX_CALLS:-5000}" ;;
    mix) "$D" teacher mix-labels --labels "$labels" --from "$data/decisions" --out "$data/distill" \
      --alpha "${ALPHA:-0.5}" ;;
    *) echo "unknown step $s (count, label, mix)" >&2; exit 2 ;;
  esac
done
