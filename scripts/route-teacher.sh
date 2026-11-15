#!/usr/bin/env bash
# The teacher jobs of s6-route-nano (docs/s6-nano2-prereg.md, 4.5) through the GLM endpoint of
# scripts/distill-glm.sh: GLM-5.3-flash (MIT, zai-org/GLM-5.3-Flash) called through the z.ai API,
# 20 items per call, nothing heavy on this machine and no GPU. The endpoint and the key come from
# the zai provider of ccs unless SKY_TEACHER_URL and SKY_TEACHER_KEY are set; the key is never
# printed.
#
# Usage: scripts/route-teacher.sh <job> [count|label|loop]   (default: loop)
#   job     describe-intent | false-negatives | rewrite-description (the items are written by
#           sky-data route: intents for the first, split for the two others)
#   count   items left and the calls they take (no call to the model; the endpoint is resolved)
#   label   one pass: sky-data teacher <job>, resumable; a worker stops after 3 failed batches in
#           a row; a batch the z.ai moderation refuses is halved until the refused items stand
#           alone (<job>.refused.jsonl, never sent again)
#   loop    passes until at most COVER_MISS% of the items are left (default 0), pausing PAUSE
#           seconds (default 1800) after a pass that answers nothing, giving up after TRIES such
#           pauses (default 24)
#   ROUTE_DATA=../skycmd-data/s6-route   the data folder; BUILD=$ROUTE_DATA/build
#   MODEL=glm-5.3-flash  BATCH=20  JOBS=4  MAX_CALLS=60000 (per pass; retries count)
#   A job refuses an item file that holds a test item (the teacher never sees a test request).
# Unattended, across usage windows: caffeinate -i scripts/route-teacher.sh false-negatives loop
set -euo pipefail
export CARGO_TARGET_DIR=${CARGO_TARGET_DIR:-$PWD/target/s3-train}
data=${ROUTE_DATA:-../skycmd-data/s6-route}
build=${BUILD:-$data/build}
job=${1:-}
case "$job" in
  describe-intent | false-negatives | rewrite-description) ;;
  *) echo "usage: scripts/route-teacher.sh <describe-intent|false-negatives|rewrite-description> [count|label|loop]" >&2; exit 2 ;;
esac
shift
model=${MODEL:-glm-5.3-flash}
batch=${BATCH:-20}
items=$build/teacher/$job.items.jsonl
out=$build/teacher/$job.jsonl
[ -f "$items" ] || { echo "no $items: run the build up to its step first (scripts/build-route-data.sh)" >&2; exit 1; }
nice -n 19 cargo build -q --release -p sky-data
D=$CARGO_TARGET_DIR/release/sky-data

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
  "$D" teacher --model "$model" --max-calls "$1" --batch "$batch" --min-interval-ms "${MIN_INTERVAL_MS:-1500}" \
    "$job" --items "$items" --out "$out" --jobs "${JOBS:-4}"
}

# left: "<items left> <items in all>", from a pass without any call
left() {
  label 0 2>&1 |
    sed -En 's/^\[route\] [a-z-]+: ([0-9]+) items to do \(([0-9]+) answered, ([0-9]+) refused.*/\1 \2 \3/p' |
    awk '{ print $1, $1 + $2 + $3 }'
}

loop() {
  local n total last=-1 pauses=0
  while :; do
    read -r n total <<<"$(left)"
    [ -n "${total:-}" ] || { echo "could not count the items left (scripts/route-teacher.sh $job count)" >&2; return 1; }
    echo "== $(date '+%F %T') $job: $n of $total items left"
    if [ $((n * 100)) -le $((total * ${COVER_MISS:-0})) ]; then
      echo "== $job: done"
      return
    fi
    if [ "$n" = "$last" ]; then
      pauses=$((pauses + 1))
      if [ "$pauses" -gt "${TRIES:-24}" ]; then
        echo "== no progress after $((pauses - 1)) pauses: stopping" >&2
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

step=${1:-loop}
case "$step" in
  count) label 0 ;;
  label) label "${MAX_CALLS:-60000}" ;;
  loop) loop ;;
  *) echo "unknown step $step (count, label, loop)" >&2; exit 2 ;;
esac
