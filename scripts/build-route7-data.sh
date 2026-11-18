#!/usr/bin/env bash
# The data stages of s6-route v7 (docs/s6-route7-prereg.md): v6's stages
# (scripts/build-route6-data.sh, run with the names below), over v6's build. No training, no GPU,
# no system scored on a fresh file; this script never calls the teacher.
#
#   inputs            build6/ against runs/s6-route7/inputs.json (the 37 files of
#                     runs/s6-route6/rebuild.json and the fix's files of v6's result block)
#   fix-plan          the full copy build6/ -> build7/, file by file (runs/s6-route7/copy.json);
#                     sky-data route v5-plan --when2call-exempt none --when2call-anchor nearest
#                     into build7/fix7/; every plan outside the When2Call group is v5's, and against
#                     v6's fix (build7/fix/, the copy's) only the no-call plans that had no anchor
#                     (both slates at L6) change (runs/s6-route7/plan-diff.json)
#   fix-teacher-jobs  the items of the rebuilt hard no-call slates (v6's answers apply by id
#                     through --also build7/fix). Then the lead runs: BUILD=$B7/fix7 caffeinate -i
#                     scripts/route-teacher.sh false-negatives loop
#   fix-build         the When2Call group rebuilt by uid; every file of build7/ against the copy
#                     (all equal but the two When2Call record files and counts/v5-slates.json), and
#                     the records against v6's: only the none-right natural and hard slates of the
#                     replanned requests differ (runs/s6-route7/rebuild.json, records-diff.json)
#   fix-checks        v6's checks and its stop rule (4.8, exit 2), v6's report beside
#   spread            the spread table of the group, after v7's fix and in v6's build
#   then v6's later stages (inputs-ok, placeholder, support, checks-read, blind-val,
#   baselines-val, fresh, fresh-tools, probes, counts), on build7/ and runs/s6-route7/; inputs-ok
#   reads runs/s6-route7/inputs6.json, written with amendment 1 of v7.
#
# Markers and logs go to build7/markers7/ and build7/logs7/ (build6/'s, copied, stay as they were).
#
#   ROUTE_DATA=../skycmd-data/s6-route   B6=$ROUTE_DATA/build6   B7=$ROUTE_DATA/build7
#   T5=$ROUTE_DATA/toucan5   MIN_FREE_GB=3   DRY_RUN=1
set -euo pipefail
cd "$(dirname "$0")/.."
data=${ROUTE_DATA:-../skycmd-data/s6-route}
b6=${B6:-$data/build6}
b7=${B7:-$data/build7}
exec env ROUTE_DATA="$data" B5="$b6" B6="$b7" FIX="$b7/fix7" MARKERS="$b7/markers7" LOGS="$b7/logs7" \
  PLAN_FLAGS="--when2call-exempt none --when2call-anchor nearest" ALSO="$b7/fix" CHECK_AGAINST="$b7/fix" \
  RUNS=runs/s6-route7 BEFORE_REPORT=runs/s6-route6/checks.json BEFORE_LABEL="v6's build" \
  EVAL_OUT="$data/eval/route7-val" scripts/build-route6-data.sh "$@"
