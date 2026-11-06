#!/usr/bin/env bash
# Night 1 of the big run (TODO.md §9.4): the s6 recipe, with no decision taken by hand
# (docs/s6-prereg.md §2 to §4 and amendment 4).
#
#   caffeinate -i scripts/s6-recipe.sh        # from the main checkout, Mac plugged in and open
#   DRY_RUN=1 scripts/s6-recipe.sh            # print the commands, run nothing
#
# 1. pilot     cargo xtask s6lr              6 runs, about 2 h 30
# 2. extension cargo xtask s6lr --extend     only when a choice falls on the edge (x0.5 or x2)
# 3. ablation  cargo xtask s6ab --adamw-lr-scale A --muon-lr-scale M   (A, M from the pilot)
#              11 arms x 3 seeds, about 15 h
# 4. recipe    scripts/s6-adopt.py adopt     -> runs/recipe/recipe.json and runs/recipe/configs/
#
# Every step resumes where it stopped (xtask stamps): after a crash or a sleep, run it again.
set -euo pipefail
cd "$(dirname "$0")/.."
export SKYCMD_STORE=${SKYCMD_STORE:-$PWD}
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
