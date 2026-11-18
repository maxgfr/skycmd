#!/usr/bin/env bash
# Train s6-route v7's model part (docs/s6-route7-prereg.md): v6's run (scripts/train-route6.sh,
# its recipe, schedule, budgets, brake, pilot rule and blind readings unchanged) on the records of
# build7/, under v7's names: runs/s6-route7, runs/s6-route7-pilot, models/s6-route7, the name
# s6-route7-nano, the guard on runs/s6-route7/inputs.ok and runs/s6-route7/placeholder.json.
# Validation only: no test or fresh file is read here.
#
#   ROUTE_DATA=../skycmd-data/s6-route   B7=$ROUTE_DATA/build7   and every variable of
#   scripts/train-route6.sh (SKY_BRAKE_FILE required, POTION_BAR, PILOT_MIN, MAIN_MIN, SMOKE,
#   DRY_RUN...)
set -euo pipefail
data=${ROUTE_DATA:-../skycmd-data/s6-route}
exec env ROUTE_DATA="$data" B6="${B7:-$data/build7}" ROUTE_TAG=s6-route7 "$(dirname "$0")/train-route6.sh" "$@"
