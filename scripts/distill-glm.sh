#!/usr/bin/env bash
# Teacher soft labels of s6-nano / s6-micro through a GLM endpoint (docs/s6-gen-prereg.md,
# amendment 1): GLM-5.3-flash, an open-weight model (MIT, zai-org/GLM-5.3-Flash), called through
# the z.ai API, so nothing heavy runs on this machine. The endpoint and the key come from the zai
# provider of ccs (Claude Code Switch) unless SKY_TEACHER_URL and SKY_TEACHER_KEY are set; the key
# is never printed.
# Usage: scripts/distill-glm.sh [count] [label] [loop] [mix]   (default: loop mix)
#   count   records left to label and the calls that takes (no API call)
#   label   one pass of sky-data teacher label-decisions -> data/teacher/glm_labels.jsonl; it
#           resumes where the last pass stopped, and a worker stops after 3 failed batches in a
#           row (a spent z.ai usage window, a network outage)
#   loop    label passes until at most COVER_MISS% of the selection is left (default 1), pausing
#           PAUSE seconds (default 1800) after a pass that labels nothing (the z.ai usage window
#           resets every 5 h), and giving up after TRIES such pauses (default 24)
#   mix     sky-data teacher mix-labels -> data/distill/glm.jsonl and agreement.json
#   MODEL=glm-5.3-flash  PER_SOURCE=0 (all; N: a sample per source family)  BATCH=20
#   JOBS=4  parallel calls: z.ai throttles beyond (measured on 29/10/2026: about 2.5 labels/s
#           whatever the number of workers, and retries from 8 workers up)
#   MAX_CALLS=60000 per pass (retries count)  ALPHA=0.5  DATA=data (where data/decisions lives)
# Unattended, across usage windows: caffeinate -i scripts/distill-glm.sh loop mix
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
    --from "$data/decisions" --out "$labels" --per-source "${PER_SOURCE:-0}" --jobs "${JOBS:-4}"
}

# left: "<records left> <records selected>", from a pass without any call.
left() {
  label 0 2>&1 | sed -n 's/^\[distill\] \([0-9]*\) records to label (\([0-9]*\) already labelled).*/\1 \2/p' |
    awk '{ print $1, $1 + $2 }'
}

loop() {
  local n total last=-1 pauses=0
  while :; do
    read -r n total <<<"$(left)"
    [ -n "${total:-}" ] || { echo "could not count the records left (scripts/distill-glm.sh count)" >&2; return 1; }
    echo "== $(date '+%F %T') $n of $total records left"
    if [ $((n * 100)) -le $((total * ${COVER_MISS:-1})) ]; then
      echo "== coverage reached"
      return
    fi
    if [ "$n" = "$last" ]; then
      pauses=$((pauses + 1))
      if [ "$pauses" -gt "${TRIES:-24}" ]; then
        echo "== no progress after $((pauses - 1)) pauses: stopping (scripts/distill-glm.sh count)" >&2
        return 1
      fi
      echo "== no progress: pausing ${PAUSE:-1800}s (the z.ai usage window resets every 5 h)"
      sleep "${PAUSE:-1800}"
    else
      pauses=0
    fi
    last=$n
    label "${MAX_CALLS:-60000}" || true
  done
}

steps=("$@")
[ ${#steps[@]} -eq 0 ] && steps=(loop mix)
for s in "${steps[@]}"; do
  case "$s" in
    count) label 0 ;;
    label) label "${MAX_CALLS:-60000}" ;;
    loop) loop ;;
    mix) "$D" teacher mix-labels --labels "$labels" --from "$data/decisions" --out "$data/distill" \
      --alpha "${ALPHA:-0.5}" ;;
    *) echo "unknown step $s (count, label, loop, mix)" >&2; exit 2 ;;
  esac
done
