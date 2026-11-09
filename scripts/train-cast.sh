#!/usr/bin/env bash
# Train s6-cast-pico (docs/SPEC-S6-CAST.md) and score it on GIFT-Eval against the references, as
# pre-registered in docs/s6-cast-prereg.md.
# Usage: scripts/train-cast.sh [data] [pilot] [train] [export] [eval]   (default: all but pilot)
#   data    real series (sky-data fetch-series), game series, GIFT-Eval windows (local only)
#   pilot   3000-step pilots: learning rate 1e-3 / 3e-3 / 1e-2, then TinyCast distillation at the
#           chosen rate; applies the pre-registered rules and writes runs/s6-cast-pilot/choice.json
#           ({"lr", "teacher"}), which train reads when LR / TEACHER are unset
#   train   MINUTES of training (synthetic pool + real series, TinyCast as teacher unless
#           TEACHER=none), QAT over the last 20%, EMA
#   export  ternary GGUF of the checkpoint the pre-registration picks, exit calibration
#   eval    GIFT-Eval scores of s6-cast (with and without flip), seasonal naive, TinyCast
#   CONFIG=configs/s6-cast-pico.json RUN=runs/s6-cast-pico MINUTES=480 BATCH=256 LR=3e-3
#   TEACHER=crates/sky-infer/tests/golden/tinycast.gguf (or none) TEACHER_WEIGHT=0.5 TEACHER_FRAC=0.5
#   CAST_OPT="..."  optimizer flags adopted by the pico ablation, without a rate (--no-muon,
#                   --wd-mode soda, --ns-polish 0, --qat-grad-a 0.21; scripts/s6-adopt.py env
#                   s6-cast-pico), for the pilots and the run (docs/s6-cast-prereg.md, amendment 3)
#   TIME_SCALE=1    multiplies MINUTES (scripts/speed-probe.sh)
#   PILOT_STEPS=3000  steps of each pilot (only SMOKE=1 of scripts/train-all.sh lowers it)
#   SKY_OUT         a prefix for everything this script writes (scripts/train-all.sh SMOKE=1)
#   DRY_RUN=1       print the commands instead of running them (nothing is built or read)
set -euo pipefail
export DEVELOPER_DIR=${DEVELOPER_DIR:-/Applications/Xcode.app/Contents/Developer}
export CARGO_TARGET_DIR=${CARGO_TARGET_DIR:-$PWD/target/s3-train}
dry=${DRY_RUN:-0}
[ "$dry" = 1 ] || cargo build -q --release -p sky-train -p sky-infer -p sky-data
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
B=$CARGO_TARGET_DIR/release
cfg=${CONFIG:-configs/s6-cast-pico.json}
sky_out=${SKY_OUT:-}
run=${RUN:-${sky_out}runs/s6-cast-pico}
name=$(basename "$run")
pilot_steps=${PILOT_STEPS:-3000}
pilot_dir=${sky_out}runs/s6-cast-pilot
minutes=$(LC_ALL=C awk -v a="${MINUTES:-480}" -v b="${TIME_SCALE:-1}" 'BEGIN { printf "%g", a * b }')
batch=${BATCH:-256}
read -r -a copt <<<"${CAST_OPT:-}"
case " ${CAST_OPT:-} " in
  *" --muon-lr "* | *" --adam-lr "*)
    echo "CAST_OPT carries a rate: the rate is LR, chosen by the pilot" >&2
    exit 2
    ;;
esac
golden=crates/sky-infer/tests/golden/tinycast.gguf
choice_json=$pilot_dir/choice.json
# choice <key>: the pilot's pre-registered choice, when it has run
choice() { [ -f "$choice_json" ] && python3 -I -c 'import json,sys; print(json.load(open(sys.argv[1]))[sys.argv[2]])' "$choice_json" "$1"; }
lr=${LR:-$(choice lr || echo 3e-3)}
teacher=${TEACHER:-$(choice teacher || echo "$golden")}
teacher_args=()
[ "$teacher" = none ] || teacher_args=(--teacher "$teacher" --teacher-weight "${TEACHER_WEIGHT:-0.5}" --teacher-frac "${TEACHER_FRAC:-0.5}")
m=${sky_out}models/s6-cast
[ "$dry" = 1 ] || mkdir -p "$m"

data() {
  [ -f data/series/wiki_daily.jsonl ] || "$B/sky-data" fetch-series --out data/series
  [ -f data/series/games.jsonl ] || "$B/sky-train" cast-synth --games flappy,rescue,dodge,doom \
    --episodes 60 --out data/series/games.jsonl
  if [ ! -f data/gift-eval/m4_hourly__H__short.json ]; then
    [ -d data/gift-eval-src ] || uv run --python 3.12 scripts/gift-eval.py download --data data/gift-eval-src
    uv run --python 3.12 scripts/gift-eval.py export --data data/gift-eval-src --out data/gift-eval
  fi
}

# Final validation pinball (normalized) of a run's last exit in `mode`.
final_pinball() {
  python3 -I - "$1/log.jsonl" "$2" <<'EOF'
import json, sys
best = None
for line in open(sys.argv[1]):
    v = json.loads(line)
    if v.get("mode") == sys.argv[2] and "val" in v:
        exits = v["val"]["pinball"]
        best = exits[max(exits, key=lambda k: int(k[4:]))]
print(best if best is not None else "nan")
EOF
}

# The pilot (docs/s6-cast-prereg.md §2, amendment 3): the rates, the rule, then the distillation
# at the chosen rate and its rule; writes runs/s6-cast-pilot/choice.json.
pilot() {
  local lr_x p=() r
  if [ "$dry" = 1 ]; then
    for lr_x in 1e-3 3e-3 1e-2; do
      x "$B/sky-train" cast-train --config "$cfg" --out "$pilot_dir/lr-$lr_x" --resume --steps "$pilot_steps" \
        --batch "$batch" --qat-frac 0 --series data/series --adam-lr "$lr_x" --eval-every "$pilot_steps" ${copt[@]+"${copt[@]}"}
    done
    echo "would apply the pilot's two rules and write $choice_json (it reads the pilots' logs)"
    return
  fi
  for lr_x in 1e-3 3e-3 1e-2; do
    r=$pilot_dir/lr-$lr_x
    "$B/sky-train" cast-train --config "$cfg" --out "$r" --resume \
      --steps "$pilot_steps" --batch "$batch" --qat-frac 0 --series data/series --adam-lr "$lr_x" --eval-every "$pilot_steps" \
      ${copt[@]+"${copt[@]}"}
    p+=("$lr_x=$(final_pinball "$r" f32)")
    echo "lr $lr_x: final f32 val pinball ${p[${#p[@]}-1]#*=}"
  done
  # keep 3e-3 unless another rate lowers the pinball by at least 2%
  local best
  best=$(python3 -I -c '
import sys
p = {k: float(v) for k, v in (a.split("=") for a in sys.argv[1:])}
b = min(p, key=p.get)
print(b if p[b] <= 0.98 * p["3e-3"] else "3e-3")' "${p[@]}")
  echo "pre-registered rule: keep 3e-3 unless another rate lowers the pinball by >= 2% -> $best"
  local d=$pilot_dir/distill-$best
  "$B/sky-train" cast-train --config "$cfg" --out "$d" --resume \
    --steps "$pilot_steps" --batch "$batch" --qat-frac 0 --series data/series --adam-lr "$best" --eval-every "$pilot_steps" \
    --teacher "$golden" ${copt[@]+"${copt[@]}"}
  local pd pn
  pd=$(final_pinball "$d" f32)
  pn=$(final_pinball "$pilot_dir/lr-$best" f32)
  # distil unless the pinball is worse by more than 1%
  python3 -I -c '
import json, sys
lr, pd, pn, golden, out = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), sys.argv[4], sys.argv[5]
teacher = "none" if pd > 1.01 * pn else golden
json.dump({"lr": lr, "teacher": teacher, "pinball_distill": pd, "pinball_plain": pn}, open(out, "w"), indent=2)
print(f"distillation: {pd:.5g} vs {pn:.5g} without -> teacher {teacher}")' "$best" "$pd" "$pn" "$golden" "$choice_json"
  echo "pre-registered rule: distil unless the pinball is worse by more than 1% -> $choice_json"
}

train() {
  x "$B/sky-train" cast-train --config "$cfg" --out "$run" --minutes "$minutes" --batch "$batch" \
    --series data/series --qat-frac 0.2 --ema 0.999 --adam-lr "$lr" --resume ${teacher_args[@]+"${teacher_args[@]}"} \
    ${copt[@]+"${copt[@]}"}
}

# The exported checkpoint: the EMA if its final ternary pinball is lower, else the last weights.
pick() {
  local e l
  e=$(final_pinball "$run" ema-ternary)
  l=$(final_pinball "$run" ternary)
  python3 -I -c "import sys; e, l = float(sys.argv[1]), float(sys.argv[2]); print('last-ema' if e == e and e < l else 'last')" "$e" "$l"
}

export_model() {
  local ck
  if [ "$dry" = 1 ]; then ck="<last or last-ema>"; else ck=$(pick); fi
  echo "== exporting $run/$ck"
  x "$B/sky-train" cast-export --run "$run/$ck" --format ternary --out "$m/$name-full.gguf"
  x "$B/sky-cast-eval" calibrate --model "$m/$name-full.gguf" --out "$m/$name.gguf"
}

score() {
  x uv run --python 3.12 scripts/gift-eval.py score --data data/gift-eval-src --forecasts "$1" --name "$2"
}

eval_all() {
  local out=${sky_out}data/gift-eval-out
  [ -f "$out/seasonal-naive/summary.json" ] || {
    x "$B/sky-cast-eval" naive --windows data/gift-eval --out "$out/seasonal-naive"
    score "$out/seasonal-naive" Seasonal_Naive
  }
  x "$B/sky-cast-eval" run --model "$m/$name.gguf" --windows data/gift-eval --out "$out/$name"
  score "$out/$name" "$name"
  x "$B/sky-cast-eval" run --model "$m/$name.gguf" --windows data/gift-eval --out "$out/$name-flip" --flip
  score "$out/$name-flip" "$name-flip"
  if [ -f crates/sky-infer/tests/golden/tinycast.gguf ] && [ ! -f "$out/tinycast/summary.json" ]; then
    x "$B/sky-cast-eval" run --model crates/sky-infer/tests/golden/tinycast.gguf --windows data/gift-eval \
      --out "$out/tinycast" --flip
    score "$out/tinycast" TinyCast-ours
  fi
}

steps=("$@")
[ ${#steps[@]} -eq 0 ] && steps=(data train export eval)
for s in "${steps[@]}"; do
  case "$s" in
    data) data ;;
    pilot) pilot ;;
    train) train ;;
    export) export_model ;;
    eval) eval_all ;;
    *) echo "unknown step $s (data, pilot, train, export, eval)" >&2; exit 2 ;;
  esac
done
