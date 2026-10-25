#!/usr/bin/env bash
# Train s6-pico (docs/SPEC-S6.md) with what the ablation adopted (docs/s6-prereg.md): numeric MLM
# pretraining and decision training with QAT, DAgger rounds whose oracle share halves each round,
# exit thresholds by conformal risk control, then a check of the play at those thresholds.
# Usage: scripts/train-s6.sh [train] [dagger] [exit]   (default: all three, in order)
#   CONFIG=configs/s6-pico.json   model config
#   OPT="..."       optimizer flags of every training phase (the pilot's rate: --muon-lr 0.006;
#                   --wd-mode soda; --ns-polish 0)
#   PRETRAIN="..."  pretraining flags (--curriculum 0.1)
#   DECIDE="..."    decision flags, first run and DAgger rounds (--exit-kd 0.5 --ema 0.999
#                   --qat-grad-a 0.21); with --ema the EMA weights are calibrated and exported
#   ROUNDS=3  ROUND_MINUTES=8  BETA=0.2 (oracle share of round 1)  ALPHA=0.02 (CRC bound)
#   GAMES=flappy,rescue,dodge,doom  EPISODES=30
set -euo pipefail
export DEVELOPER_DIR=${DEVELOPER_DIR:-/Applications/Xcode.app/Contents/Developer}
export CARGO_TARGET_DIR=${CARGO_TARGET_DIR:-$PWD/target/s3-train}
cargo build -q --release -p sky-train -p sky-arena -p sky-gguf
B=$CARGO_TARGET_DIR/release
T=$B/sky-train
A=$B/sky-arena
cfg=${CONFIG:-configs/s6-pico.json}
read -r -a opt <<<"${OPT:-}"
read -r -a pre <<<"${PRETRAIN:-}"
read -r -a dec <<<"${DECIDE:-}"
rounds=${ROUNDS:-3}
games=${GAMES:-flappy,rescue,dodge,doom}
episodes=${EPISODES:-30}
alpha=${ALPHA:-0.02}
records=data/decisions,data/games,data/numeric
run=runs/s6-pico
m=models/s6-pico
ckpt=last
[[ " ${DECIDE:-} " == *" --ema "* ]] && ckpt=last-ema
mkdir -p "$m"

# post <decide dir> <gguf>: calibration, ternary export, play at full depth.
post() {
  $T s2-calibrate --run "$1/$ckpt" --records "$records"
  $T s2-export --run "$1/$ckpt" --format ternary --out "$2"
  $A "$2" --episodes "$episodes" --game "$games" --exit-threshold 1.0 --json "${2%.gguf}-arena.json"
}

train() {
  echo "== s6-pico: pretrain 8m, decide 15m ($cfg)"
  $T s2-pretrain --config "$cfg" --tokenizer data/tok/tok-s2-1024.json --text data/text --minutes 8 \
    --num-weight 0.05 --numeric data/numeric --numeric-weight 0.1 --out "$run/pretrain" \
    ${opt[@]+"${opt[@]}"} ${pre[@]+"${pre[@]}"}
  rm -rf "$run/decide"
  $T s2-decide --config "$cfg" --init "$run/pretrain/init" --records "$records" \
    --weights games=0.6,numeric=0.1,decisions=0.3 --batch 128 --minutes 15 \
    --qat-frac 0.9 --loop-dropout 0.25 --pad-to 32 --out "$run/decide" \
    ${opt[@]+"${opt[@]}"} ${dec[@]+"${dec[@]}"}
  post "$run/decide" "$m/skycmd-s6-pico-ternary.gguf"
}

dagger() {
  local init=$run/decide/$ckpt model=$m/skycmd-s6-pico-ternary.gguf beta=${BETA:-0.2} r
  for r in $(seq 1 "$rounds"); do
    local dir=$run-g$r data=data/dagger-s6-pico/r$r out=$m/skycmd-s6-pico-g$r-ternary.gguf
    echo "== s6-pico round $r: DAgger from $model (oracle share $beta)"
    $A "$model" --dagger 15000 --seed "$r" --game "$games" --dagger-beta "$beta" --out "$data"
    rm -rf "$dir/decide"
    $T s2-decide --config "$cfg" --init "$init" --tokenizer data/tok/tok-s2-1024.json \
      --records "$records,$data" --weights "games=0.5,r$r=0.35,numeric=0.05,decisions=0.1" \
      --batch 128 --minutes "${ROUND_MINUTES:-8}" --qat-frac 1.0 --loop-dropout 0.25 --pad-to 32 \
      --out "$dir/decide" ${opt[@]+"${opt[@]}"} ${dec[@]+"${dec[@]}"}
    post "$dir/decide" "$out"
    init=$dir/decide/$ckpt
    model=$out
    beta=$(python3 -c "print($beta / 2)")
  done
}

# The last model's exit thresholds by conformal risk control (disagreement with full depth
# <= ALPHA), written into a -crc copy; the guarantee holds for full-depth play, so the play at
# those thresholds is checked against full depth (within 2%, no extra crash or veto).
exits() {
  local model=$m/skycmd-s6-pico-g$rounds-ternary.gguf
  [ "$rounds" -gt 0 ] || model=$m/skycmd-s6-pico-ternary.gguf
  local crc=${model%.gguf}-crc.gguf json=${model%.gguf}-crc.json
  echo "== exit thresholds (CRC, alpha $alpha) for $model"
  $A "$model" --calibrate-crc "$alpha" --episodes "$episodes" --game "$games" --json "$json"
  read -r -a sets <<<"$(python3 -I - "$json" <<'EOF'
import json, sys
r = json.load(open(sys.argv[1]))
print(" ".join(f"skycmd-s2.exit_threshold.{k}={r[k]['tau']}" for k in ("choice", "score", "noul") if r[k]["n"] > 0))
EOF
)"
  "$B/gguf-set" "$model" "$crc" ${sets[@]+"${sets[@]}"}
  $A "$crc" --episodes "$episodes" --game "$games" --json "${crc%.gguf}-arena.json"
  python3 -I - "${model%.gguf}-arena.json" "${crc%.gguf}-arena.json" <<'EOF'
import json, sys
ref, run = (json.load(open(p)) for p in sys.argv[1:])
bad = []
for g, v in ref.items():
    if not (isinstance(v, dict) and "model" in v):
        continue
    a, b = v["model"], run[g]["model"]
    for k in ("mean_score", "rescue_rate", "completion_rate"):
        if b[k] < a[k] - 0.02 * abs(a[k]) - 1e-9:
            bad.append(f"{g} {k} {b[k]:.3f} < {a[k]:.3f}")
    if b["crashes"] > a["crashes"] or b["vetoes"] > a["vetoes"]:
        bad.append(f"{g} crashes {b['crashes']} vetoes {b['vetoes']} (full depth {a['crashes']} / {a['vetoes']})")
if bad:
    print("CRC thresholds rejected: " + "; ".join(bad) + "\nkeep full depth, or run scripts/calibrate-exit.sh (closed loop)")
    sys.exit(1)
print("CRC thresholds kept: play within 2% of full depth")
EOF
}

steps=("$@")
[ ${#steps[@]} -eq 0 ] && steps=(train dagger exit)
for s in "${steps[@]}"; do
  case "$s" in
    train) train ;;
    dagger) dagger ;;
    exit) exits ;;
    *) echo "unknown step $s (train, dagger, exit)" >&2; exit 2 ;;
  esac
done
