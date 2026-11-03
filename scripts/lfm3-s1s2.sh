#!/usr/bin/env bash
# LFM3-1.7B, System 1 first and System 2 only when needed (docs/lfm3-s1s2-prereg.md): answers
# without reasoning, with reasoning under budgets of 256 and 1024 tokens, then a router whose
# threshold comes from conformal risk control on the calibration split. No training of LFM3;
# inference only, local (mlx-vlm on Apple Silicon). Every generation resumes after an interruption.
# Usage: scripts/lfm3-s1s2.sh [data] [smoke] [calib] [test] [route]   (default: data calib test route)
#   data    GSM8K, MATH-500 and MATH train, pinned revisions -> data/lfm3/{calib,test}.jsonl
#   smoke   3 calibration questions per mode into runs/lfm3-s1s2-smoke (never mixed with the run)
#   calib   s1, s2-256, s2-1024 on the 600 calibration questions, then the calibration report
#   test    s1, s2-256, s2-1024 on the 1,000 test questions (read only by `route`)
#   route   the router on the test split: runs/lfm3-s1s2/report.md and summary.json
# Overnight: caffeinate -i scripts/lfm3-s1s2.sh   (prints tok/s and s/question as it goes)
set -euo pipefail
py() {
  uv run -q --python 3.12 --with "mlx-vlm==0.7.11" --with "datasets>=3" \
    python experiments/lfm3-s1s2/s1s2.py "$@"
}
budgets=(256 1024)

gen_all() {
  local split=$1 b
  py gen --split "$split" --mode s1
  for b in "${budgets[@]}"; do py gen --split "$split" --mode s2 --budget "$b"; done
}

steps=("$@")
[ ${#steps[@]} -eq 0 ] && steps=(data calib test route)
for s in "${steps[@]}"; do
  case "$s" in
    data) [ -s data/lfm3/test.jsonl ] && echo "== data/lfm3 exists, skipped" || py data ;;
    smoke)
      [ -s data/lfm3/calib.jsonl ] || py data
      py gen --split calib --mode s1 --limit 3 --out runs/lfm3-s1s2-smoke
      py gen --split calib --mode s2 --budget 256 --limit 3 --out runs/lfm3-s1s2-smoke
      ;;
    calib) gen_all calib && py route --split calib ;;
    test) gen_all test ;;
    route) py route --split test ;;
    *) echo "unknown step $s (data, smoke, calib, test, route)" >&2; exit 2 ;;
  esac
done
