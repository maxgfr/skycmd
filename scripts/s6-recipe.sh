#!/usr/bin/env bash
# Night 1 of the big run (TODO.md §9.4): the s6 recipe, with no decision taken by hand
# (docs/s6-prereg.md §2 to §4 and amendment 4).
#
#   caffeinate -i scripts/s6-recipe.sh        # from the main checkout
#   DRY_RUN=1 scripts/s6-recipe.sh            # print the commands, run nothing
#   SKY_DUTY=0.5                              # share of the time the GPU works (default here;
#                                             # 1 = full speed). The steps and the results do not move.
#   SKY_COOL=45:15                            # minutes of work, then of complete rest (default here;
#                                             # empty = no long rests)
#   MAX_FAILS=5 RETRY_WAIT=60                 # give up after this many failures in a row with no
#                                             # finished step; seconds between two attempts
#
# 1. pilot     cargo xtask s6lr              6 runs, about 2 h 30
# 2. extension cargo xtask s6lr --extend     only when a choice falls on the edge (x0.5 or x2)
# 3. ablation  cargo xtask s6ab --adamw-lr-scale A --muon-lr-scale M   (A, M from the pilot)
#              11 arms x 3 seeds, about 15 h
# 4. recipe    scripts/s6-adopt.py adopt     -> runs/recipe/recipe.json and runs/recipe/configs/
#
# Closing the lid suspends the run and opening it goes on. After a shutdown, run it again: the
# finished steps are skipped (xtask stamps) and only the step in progress starts over. A failure
# (a GPU error after a wake, say) is retried on its own.
set -euo pipefail
cd "$(dirname "$0")/.."
export SKYCMD_STORE=${SKYCMD_STORE:-$PWD}
export SKY_DUTY=${SKY_DUTY:-0.5}
export SKY_COOL=${SKY_COOL-45:15}

if [ -z "${DRY_RUN:-}" ] && [ -z "${S6_RECIPE_WORKER:-}" ]; then
  stamps() { find "$SKYCMD_STORE/runs/s5-state/s6lr" "$SKYCMD_STORE/runs/s5-state/s6ab" -name '*.done' 2>/dev/null | wc -l || true; }
  fails=0
  while :; do
    before=$(stamps)
    rc=0
    S6_RECIPE_WORKER=1 "$PWD/scripts/s6-recipe.sh" || rc=$?
    [ "$rc" = 0 ] && exit 0
    [ "$rc" -gt 128 ] && exit "$rc" # interrupted by hand (Ctrl-C)
    if [ "$(stamps)" -gt "$before" ]; then fails=1; else fails=$((fails + 1)); fi
    if [ "$fails" -ge "${MAX_FAILS:-5}" ]; then
      echo "== $(date '+%F %T') $fails failures in a row with no finished step: giving up" >&2
      exit "$rc"
    fi
    echo "== $(date '+%F %T') failed (exit $rc): next attempt in ${RETRY_WAIT:-60} s" >&2
    sleep "${RETRY_WAIT:-60}"
  done
fi

dry=()
[ -n "${DRY_RUN:-}" ] && dry=(--dry-run)

echo "== $(date '+%F %T') 1. pilot"
cargo xtask s6lr ${dry[@]+"${dry[@]}"}
echo "== $(date '+%F %T') 2. extension (nothing to do unless a choice is on the edge)"
if [ -n "${DRY_RUN:-}" ]; then
  echo "would run: cargo xtask s6lr --extend (it reads the pilot's logs)"
else
  cargo xtask s6lr --extend
fi

scales() {
  python3 -I -c 'import json, sys
s = json.load(open(sys.argv[1]))["lr_scale"]
print(s["adamw"], s["muon"])' "$SKYCMD_STORE/runs/s6lr/summary.json"
}
if [ -n "${DRY_RUN:-}" ] && [ ! -f "$SKYCMD_STORE/runs/s6lr/summary.json" ]; then
  read -r a m <<<"1 1"
else
  read -r a m <<<"$(scales)"
fi
echo "== $(date '+%F %T') 3. ablation (adamw x$a, muon x$m)"
cargo xtask s6ab --adamw-lr-scale "$a" --muon-lr-scale "$m" ${dry[@]+"${dry[@]}"}

echo "== $(date '+%F %T') 4. recipe"
if [ -n "${DRY_RUN:-}" ]; then
  echo "would run: python3 -I scripts/s6-adopt.py adopt (in $SKYCMD_STORE)"
else
  r=$SKYCMD_STORE/runs
  python3 -I scripts/s6-adopt.py adopt --pilot "$r/s6lr/summary.json" --ablation "$r/s6ab/summary.json" --out "$r/recipe"
  echo "== recipe in runs/recipe/recipe.json: commit it with the report (TODO.md §9.4, step 8)"
fi
