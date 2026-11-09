#!/usr/bin/env bash
# s6-ear: an s6 that listens (docs/SPEC-S6.md §16, docs/s6-ear-prereg.md). The audio data and its
# jev, the "Mimi as the teacher" pretraining (masked prediction of the Mimi codebook-0 ids of the
# 80 ms log-mel frames), the decision training on the turn-end and keyword questions with QAT, the
# control without pretraining, the exit and System 2 thresholds by conformal risk control, then
# the evaluation and the gates against Smart Turn v3.2.
# Usage: scripts/train-ear.sh [data] [pretrain] [decide] [eval]   (default: pretrain decide eval)
#   data          sky-data fetch-audio (smart-turn, speech-commands, mswc, mimi) + audio-jev; once
#   PRE_MIN=60 DEC_MIN=60 CTL_MIN=$DEC_MIN   budgets in minutes (the control gets DEC_MIN as well)
#   BATCH=64      decision batch; pretraining uses 64 crops of SEQ=128 frames (10.24 s)
#   ALPHA=0.02 (exit CRC)  ROUTE_ALPHA=0.05 (System 2 hand-off CRC)
#   MAX_CLIPS=0   clips per set for `data` (0: all; 200 for a smoke run)
#   ABOUT=data/teacher/ear_about_labels.jsonl   v2: GLM labels of the transcripts, added to the jev
#                 when the file exists (sky-data teacher label-transcripts)
#   DRY_RUN=1     print the commands instead of running them (nothing is built or read)
#   The recipe of the pico ablation (scripts/s6-adopt.py env s6-ear-nano; docs/s6-ear-prereg.md,
#   amendment 2), for the model and its control alike:
#   CONFIG=configs/s6-ear-nano.json  OPT="..." (no rate)  PRETRAIN="..."  DECIDE="..." (with --ema
#   the EMA weights are calibrated and exported)  LR_SCALE (unset: the default rates; set: the
#   pico pilot's scale moved to d = 144 by muP, LR_SCALE x sqrt(96/144) x the phase's default, as
#   scripts/train-s6.sh)  TIME_SCALE=1 (multiplies every budget in minutes, scripts/speed-probe.sh)
#   RESUME=1 (continue from the checkpoints of an interrupted run; unset: start over)
# Not to be run while another training job uses the GPU.
set -euo pipefail
export DEVELOPER_DIR=${DEVELOPER_DIR:-/Applications/Xcode.app/Contents/Developer}
export CARGO_TARGET_DIR=${CARGO_TARGET_DIR:-$PWD/target/s3-train}
dry=${DRY_RUN:-0}
B=$CARGO_TARGET_DIR/release
T=$B/sky-train
E=$B/sky-eval
D=$B/sky-data
name=s6-ear-nano
cfg=${CONFIG:-configs/$name.json}
tok=data/tok/tok-s6-2048.json
ear=data/ear
jev=$ear/jev
run=runs/$name
m=models/$name
# tmin <minutes>: a budget in minutes times TIME_SCALE (the M5's speed over this host's,
# scripts/speed-probe.sh), so that a slower machine runs the steps the M5 would (amendment 4).
tmin() { LC_ALL=C awk -v a="$1" -v b="${TIME_SCALE:-1}" 'BEGIN { printf "%g", a * b }'; }
pre_min=$(tmin "${PRE_MIN:-60}")
dec_min=$(tmin "${DEC_MIN:-60}")
ctl_min=$(tmin "${CTL_MIN:-${DEC_MIN:-60}}")
read -r -a opt <<<"${OPT:-}"
read -r -a pre <<<"${PRETRAIN:-}"
read -r -a dec <<<"${DECIDE:-}"
resume=()
[ -n "${RESUME:-}" ] && resume=(--resume)
ckpt=last
[[ " ${DECIDE:-} " == *" --ema "* ]] && ckpt=last-ema
case " ${OPT:-} " in
  *" --muon-lr "* | *" --adam-lr "*)
    echo "OPT carries a rate: set LR_SCALE instead (each phase scales its own default)" >&2
    exit 2
    ;;
esac
lr_flag=--muon-lr
[[ " ${OPT:-} " == *" --no-muon "* ]] && lr_flag=--adam-lr
# rate_args <phase default>: no flag without LR_SCALE (the default rates), else the pico
# pilot's rate moved to d = 144 as muP predicts (as scripts/train-s6.sh).
rate_args() {
  [ -n "${LR_SCALE:-}" ] || return 0
  LC_ALL=C awk -v f="$lr_flag" -v a="$1" -v b="$LR_SCALE" 'BEGIN { printf "%s %.3g\n", f, a * b * sqrt(96 / 144) }'
}
read -r -a pre_lr <<<"$(rate_args 3e-3)"
read -r -a dec_lr <<<"$(rate_args 1e-3)"
batch=${BATCH:-64}
seq=${SEQ:-128}
alpha=${ALPHA:-0.02}
route_alpha=${ROUTE_ALPHA:-0.05}
max_clips=${MAX_CLIPS:-0}
about=${ABOUT:-data/teacher/ear_about_labels.jsonl}
records=$jev/turn_end.jsonl,$jev/keyword.jsonl,$jev/keyword_zs.jsonl
weights=turn_end=0.45,keyword=0.30,keyword_zs=0.25
if [ -f "$jev/about.jsonl" ]; then
  records=$records,$jev/about.jsonl
  weights=turn_end=0.40,keyword=0.25,keyword_zs=0.20,about=0.15
fi

# x <command...>: runs it, or prints it under DRY_RUN=1.
x() {
  if [ "$dry" = 1 ]; then
    printf '+'
    printf ' %q' "$@"
    printf '\n'
  else
    "$@"
  fi
}

[ "$dry" = 1 ] || cargo build -q --release -p sky-train -p sky-infer -p sky-data -p sky-gguf
mkdir -p "$m" "$run/eval"

data() {
  x "$D" fetch-audio --set all --out "$ear" --max-clips "$max_clips"
  if [ -s "$about" ]; then
    x "$D" audio-jev --dir "$ear" --about "$about"
  else
    x "$D" audio-jev --dir "$ear"
  fi
}

pretrain() {
  x "$T" s2-pretrain --audio --config "$cfg" --tokenizer "$tok" --text "$ear/mimi" --minutes "$pre_min" \
    --seq "$seq" --batch 64 --out "$run/pretrain" ${pre_lr[@]+"${pre_lr[@]}"} ${opt[@]+"${opt[@]}"} \
    ${pre[@]+"${pre[@]}"} ${resume[@]+"${resume[@]}"}
}

# decide_ <out> <minutes> [--init dir]: decision training with QAT over the last 10%.
decide_() {
  local out=$1 min=$2
  shift 2
  [ -n "${RESUME:-}" ] || x rm -rf "$out"
  x "$T" s2-decide --config "$cfg" --tokenizer "$tok" --records "$records" --weights "$weights" \
    --batch "$batch" --minutes "$min" --qat-frac 0.1 --loop-dropout 0.25 --length-pool 16 --pad-to 32 \
    --out "$out" ${dec_lr[@]+"${dec_lr[@]}"} ${opt[@]+"${opt[@]}"} ${dec[@]+"${dec[@]}"} ${resume[@]+"${resume[@]}"} "$@"
  x "$T" s2-calibrate --run "$out/$ckpt" --records "$records"
}

decide() {
  [ "$dry" = 1 ] || [ -d "$run/pretrain/init" ] || {
    echo "no $run/pretrain/init: run the pretrain step first" >&2
    exit 1
  }
  decide_ "$run/decide" "$dec_min" --init "$run/pretrain/init"
  x "$T" s2-export --run "$run/decide/$ckpt" --format ternary --out "$m/$name.gguf"
  # the pre-registered control: the same model and decision budget, no Mimi pretraining
  decide_ "$run/control" "$ctl_min"
  x "$T" s2-export --run "$run/control/$ckpt" --format ternary --out "$m/$name-control.gguf"
}

eval_() {
  local model=$m/$name.gguf crc=$m/$name-crc.gguf ev=$run/eval
  # exit and System 2 thresholds by conformal risk control on val (full depth, every exit)
  x "$E" "$model" --records "$records" --split val --exit-threshold 1.0 --dump-exits "$ev/val-exits.jsonl"
  x python3 -I scripts/ear-gate.py crc "$ev/val-exits.jsonl" --alpha "$alpha" --route-alpha "$route_alpha" --json "$ev/crc.json"
  if [ "$dry" = 1 ]; then
    x "$B/gguf-set" "$model" "$crc" 'skycmd-s2.exit_threshold.<type>=<tau>'
  else
    read -r -a sets <<<"$(python3 -I scripts/ear-gate.py crc "$ev/val-exits.jsonl" --alpha "$alpha" --print-sets)"
    "$B/gguf-set" "$model" "$crc" ${sets[@]+"${sets[@]}"}
  fi
  # test split: full depth, CRC thresholds, the control, then Smart Turn on the same clips
  x "$E" "$model" --records "$records" --split test --exit-threshold 1.0 --json "$ev/eval-full.json" \
    --dump-probs "$ev/probs-full.jsonl"
  x "$E" "$crc" --records "$records" --split test --json "$ev/eval-crc.json" --dump "$ev/dump-crc.jsonl"
  x "$E" "$m/$name-control.gguf" --records "$records" --split test --exit-threshold 1.0 --json "$ev/eval-control.json"
  x uv run scripts/ear-baseline.py --records "$jev/turn_end.jsonl" --json "$ev/baseline-smart-turn.json"
  # WASM parity on 20 test clips through the PCM path (rows with a wav), at the thresholds of the
  # file (the browser engine uses them)
  [ "$dry" = 1 ] || [ -f site/public/pkg/sky_web_bg.wasm ] || scripts/build-wasm.sh
  if [ "$dry" = 1 ]; then
    x node crates/sky-web/bench/parity-node.mjs "$crc" "$ev/dump-wav.jsonl" --n 20
  else
    grep '"wav":"' "$ev/dump-crc.jsonl" | head -20 >"$ev/dump-wav.jsonl" || true
    node crates/sky-web/bench/parity-node.mjs "$crc" "$ev/dump-wav.jsonl" --n 20 | tee "$ev/wasm.json"
  fi
  x python3 -I scripts/ear-gate.py gate --eval "$ev/eval-full.json" --baseline "$ev/baseline-smart-turn.json" \
    --control "$ev/eval-control.json" --json "$ev/gate.json"
}

steps=("$@")
[ ${#steps[@]} -gt 0 ] || steps=(pretrain decide eval)
for s in "${steps[@]}"; do
  case "$s" in
    data) data ;;
    pretrain) pretrain ;;
    decide) decide ;;
    eval) eval_ ;;
    *)
      echo "unknown step $s (data, pretrain, decide, eval)" >&2
      exit 2
      ;;
  esac
done
