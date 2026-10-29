#!/usr/bin/env bash
# Train the general-purpose s6 models, s6-nano and s6-micro (docs/SPEC-S6.md §15,
# docs/s6-gen-prereg.md), with the recipe the pico ablation adopted (docs/s6-prereg.md): numeric
# MLM pretraining on data/text-open then data/text, decision training with QAT on the decision
# sets, the games and the numeric curriculum, DAgger rounds whose oracle share halves each round
# and that keep 30% decisions, exit thresholds by conformal risk control, then the evaluation and
# the publication gate.
# Usage: SIZE=nano|micro scripts/train-s6.sh [lr] [train] [dagger] [exit] [eval]
#        (default: train dagger exit eval; lr is the size's learning-rate pilot, run first)
#   SIZE=nano       nano (configs/s6-nano.json, 2048 tokens) or micro (configs/s6-micro.json, 4096)
#   OPT="..."       optimizer flags of every phase adopted by the ablation, without a rate
#                   (--wd-mode soda; --ns-polish 0; --no-muon if AdamW was kept)
#   LR_SCALE=1      the pico pilot's scale of the adopted optimizer (runs/s6lr/summary.json):
#                   the peak rate is LR_SCALE x sqrt(96 / d) x LR_X x the phase's default (3e-3
#                   pretraining, 1e-3 decision), on --muon-lr (--adam-lr with --no-muon): the pico
#                   rate (d = 96, as cargo xtask s6ab applies it) moved to width d as muP predicts
#   LR_X            this size's pilot scale; read from runs/s6-$SIZE-lr/summary.json when unset
#   PRETRAIN="..."  pretraining flags (--curriculum 0.1)
#   DECIDE="..."    decision flags, first run and DAgger rounds (--exit-kd 0.5 --ema 0.999
#                   --qat-grad-a 0.21); with --ema the EMA weights are calibrated and exported
#   PRE_MIN DEC_MIN ROUND_MINUTES LR_MIN   budgets in minutes (nano 90 90 15 20, micro 180 180 20 20)
#   ROUNDS=3  BETA=0.2 (oracle share of round 1)  ALPHA=0.02 (CRC bound)
#   GAMES=flappy,rescue,dodge,doom  EPISODES=30  BASELINE=site/public/models/s3-pico.gguf
#   DRY_RUN=1       print the commands instead of running them (nothing is built or read)
set -euo pipefail
export DEVELOPER_DIR=${DEVELOPER_DIR:-/Applications/Xcode.app/Contents/Developer}
export CARGO_TARGET_DIR=${CARGO_TARGET_DIR:-$PWD/target/s3-train}
dry=${DRY_RUN:-0}
B=$CARGO_TARGET_DIR/release
T=$B/sky-train
A=$B/sky-arena
E=$B/sky-eval
size=${SIZE:-nano}
case "$size" in
  nano) vocab=2048 width=144 bpre=90 bdec=90 bround=15 batch=128 ;;
  micro) vocab=4096 width=256 bpre=180 bdec=180 bround=20 batch=64 ;;
  *) echo "SIZE must be nano or micro, not $size" >&2; exit 2 ;;
esac
name=s6-$size
cfg=configs/$name.json
tok=data/tok/tok-s6-$vocab.json
pre_min=${PRE_MIN:-$bpre}
dec_min=${DEC_MIN:-$bdec}
round_min=${ROUND_MINUTES:-$bround}
lr_min=${LR_MIN:-20}
read -r -a opt <<<"${OPT:-}"
read -r -a pre <<<"${PRETRAIN:-}"
read -r -a dec <<<"${DECIDE:-}"
rounds=${ROUNDS:-3}
games=${GAMES:-flappy,rescue,dodge,doom}
episodes=${EPISODES:-30}
alpha=${ALPHA:-0.02}
baseline=${BASELINE:-site/public/models/s3-pico.gguf}
records=data/decisions,data/games,data/numeric
weights=decisions=0.55,games=0.35,numeric=0.10
run=runs/$name
m=models/$name
ckpt=last mode=ternary
[[ " ${DECIDE:-} " == *" --ema "* ]] && ckpt=last-ema mode=ema-ternary
case " ${OPT:-} " in
  *" --muon-lr "* | *" --adam-lr "*)
    echo "OPT carries a rate: set LR_SCALE instead (each phase scales its own default)" >&2
    exit 2
    ;;
esac
lr_flag=--muon-lr
[[ " ${OPT:-} " == *" --no-muon "* ]] && lr_flag=--adam-lr

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

# rate <phase default> <scale>: the peak rate of one phase, centred on the muP move from d = 96.
rate() { LC_ALL=C awk -v a="$1" -v b="${LR_SCALE:-1}" -v c="$2" -v d="$width" 'BEGIN { printf "%.3g", a * b * sqrt(96 / d) * c }'; }

# The size's pilot scale: LR_X, else the pilot's summary. The pre-registration needs the pilot.
lr_x() {
  if [ -n "${LR_X:-}" ]; then
    echo "$LR_X"
  elif [ -f "$run-lr/summary.json" ]; then
    python3 -I -c 'import json,sys; print(json.load(open(sys.argv[1]))["chosen"])' "$run-lr/summary.json"
  elif [ "$dry" = 1 ]; then
    echo 1
  else
    echo "no $run-lr/summary.json: run the lr step first, or set LR_X" >&2
    return 1
  fi
}

[ "$dry" = 1 ] || cargo build -q --release -p sky-train -p sky-arena -p sky-gguf -p sky-infer
mkdir -p "$m"

# pretrain <out> <minutes> <scale>, decide <init> <out> <minutes> <scale> <records> <weights> <qat frac>
pretrain() {
  x "$T" s2-pretrain --config "$cfg" --tokenizer "$tok" --text data/text-open,data/text --minutes "$2" \
    --seq 256 --batch 64 --num-weight 0.05 --numeric data/numeric --numeric-weight 0.1 --out "$1" \
    "$lr_flag" "$(rate 3e-3 "$3")" ${opt[@]+"${opt[@]}"} ${pre[@]+"${pre[@]}"}
}
decide() {
  x rm -rf "$2"
  x "$T" s2-decide --config "$cfg" --init "$1" --tokenizer "$tok" --records "$5" --weights "$6" \
    --batch "$batch" --minutes "$3" --qat-frac "$7" --loop-dropout 0.25 --length-pool 16 --pad-to 32 \
    --out "$2" "$lr_flag" "$(rate 1e-3 "$4")" ${opt[@]+"${opt[@]}"} ${dec[@]+"${dec[@]}"}
}

# post <decide dir> <gguf>: calibration, ternary export, play at full depth.
post() {
  x "$T" s2-calibrate --run "$1/$ckpt" --records "$records"
  x "$T" s2-export --run "$1/$ckpt" --format ternary --out "$2"
  x "$A" "$2" --episodes "$episodes" --game "$games" --exit-threshold 1.0 --json "${2%.gguf}-arena.json"
}

# The size's learning-rate pilot (docs/s6-gen-prereg.md §3): x0.5, x1 and x2 of the adopted rate,
# LR_MIN minutes of pretraining and of decision each, seed 0; x1 is kept unless another scale beats
# its final ternary validation accuracy (last exit) by at least 0.01.
lr() {
  local s
  for s in 0.5 1 2; do
    echo "== $name lr pilot x$s: pretrain ${lr_min}m, decide ${lr_min}m"
    pretrain "$run-lr/x$s/pretrain" "$lr_min" "$s"
    decide "$run-lr/x$s/pretrain/init" "$run-lr/x$s/decide" "$lr_min" "$s" "$records" "$weights" 0.9
  done
  [ "$dry" = 1 ] && return
  python3 -I - "$run-lr" "$mode" <<'EOF'
import json, os, sys
root, mode = sys.argv[1:]
acc = {}
for s in ("0.5", "1", "2"):
    log = [json.loads(l) for l in open(os.path.join(root, f"x{s}", "decide", "log.jsonl")) if l.strip()]
    last = next(e for e in reversed(log) if e.get("mode") == mode and "val" in e)
    k = max(last["val"], key=lambda k: int(k[len("exit"):]))
    acc[s] = last["val"][k]["all"]["accuracy"]
best = max(acc, key=acc.get)
chosen = best if acc[best] - acc["1"] >= 0.01 else "1"
json.dump({"mode": mode, "val_accuracy": acc, "chosen": chosen}, open(os.path.join(root, "summary.json"), "w"), indent=2)
print(f"lr pilot: {acc} -> x{chosen}")
EOF
}

train() {
  local s
  s=$(lr_x)
  echo "== $name: pretrain ${pre_min}m, decide ${dec_min}m ($cfg, rate x${LR_SCALE:-1} x$s)"
  pretrain "$run/pretrain" "$pre_min" "$s"
  decide "$run/pretrain/init" "$run/decide" "$dec_min" "$s" "$records" "$weights" 0.9
  post "$run/decide" "$m/skycmd-$name-ternary.gguf"
}

dagger() {
  local init=$run/decide/$ckpt model=$m/skycmd-$name-ternary.gguf beta=${BETA:-0.2} r s
  s=$(lr_x)
  for r in $(seq 1 "$rounds"); do
    local dir=$run-g$r data=data/dagger-$name/r$r out=$m/skycmd-$name-g$r-ternary.gguf
    echo "== $name round $r: DAgger from $model (oracle share $beta)"
    x "$A" "$model" --dagger 15000 --seed "$r" --game "$games" --dagger-beta "$beta" --out "$data"
    decide "$init" "$dir/decide" "$round_min" "$s" "$records,$data" \
      "games=0.4,r$r=0.25,decisions=0.3,numeric=0.05" 1.0
    post "$dir/decide" "$out"
    init=$dir/decide/$ckpt
    model=$out
    beta=$(LC_ALL=C awk -v b="$beta" 'BEGIN { printf "%g", b / 2 }')
  done
}

final() {
  if [ "$rounds" -gt 0 ]; then echo "$m/skycmd-$name-g$rounds-ternary.gguf"; else echo "$m/skycmd-$name-ternary.gguf"; fi
}

# The last model's exit thresholds by conformal risk control (disagreement with full depth
# <= ALPHA), written into a -crc copy; the guarantee holds for full-depth play, so the play at
# those thresholds is checked against full depth (within 2%, no extra crash or veto); otherwise
# the copy is removed and the model ships at full depth.
exits() {
  local model
  model=$(final)
  local crc=${model%.gguf}-crc.gguf json=${model%.gguf}-crc.json
  echo "== exit thresholds (CRC, alpha $alpha) for $model"
  x "$A" "$model" --calibrate-crc "$alpha" --episodes "$episodes" --game "$games" --json "$json"
  [ "$dry" = 1 ] && return
  read -r -a sets <<<"$(python3 -I - "$json" <<'EOF'
import json, sys
r = json.load(open(sys.argv[1]))
print(" ".join(f"skycmd-s2.exit_threshold.{k}={r[k]['tau']}" for k in ("choice", "score", "noul") if r[k]["n"] > 0))
EOF
)"
  "$B/gguf-set" "$model" "$crc" ${sets[@]+"${sets[@]}"}
  "$A" "$crc" --episodes "$episodes" --game "$games" --json "${crc%.gguf}-arena.json"
  # A rejected -crc copy is removed, so eval reads full depth only.
  python3 -I - "${model%.gguf}-arena.json" "${crc%.gguf}-arena.json" <<'EOF' || rm -f "$crc"
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

# The pre-registered evaluation (docs/s6-gen-prereg.md §5) of the final ternary GGUF, at full
# depth (tau = 1) and at its CRC thresholds, with the references under the same conditions:
# s2-eval report, sky-eval on the language test sets (Banking77 apart: it is trained on), arena
# of the same seeds for s3-pico, the share of truncated records, the WASM latency per decision,
# then the gate (scripts/s6-gate.py).
eval_() {
  local model crc ev=$run/eval g tests
  model=$(final)
  crc=${model%.gguf}-crc.gguf
  tests=data/decisions/typed_decisions_bench.jsonl,data/s4/heldout/zs_massive_en.jsonl,data/s4/heldout/zs_massive_fr.jsonl
  tests=$tests,data/decisions/when2call.jsonl,data/decisions/notam.jsonl
  x mkdir -p "$ev"
  local dir=$run/decide
  [ "$rounds" -gt 0 ] && dir=$run-g$rounds/decide
  x "$T" s2-eval --run "$dir/$ckpt" --records "$records" --gguf "$model" --gguf-only --report "$ev/report.md"
  x "$E" "$model" --records "$tests" --max 2000 --exit-threshold 1.0 --json "$ev/eval-$name-full.json" \
    --dump "$ev/dump-$name.jsonl"
  [ "$dry" = 1 ] || [ -f "$crc" ] && x "$E" "$crc" --records "$tests" --max 2000 --json "$ev/eval-$name-crc.json"
  x "$E" "$model" --records data/s4/heldout/zs_banking77.jsonl --exit-threshold 1.0 --json "$ev/indist-$name.json"
  for g in "$baseline" site/public/models/s4-micro.gguf site/public/models/s5-lite.gguf; do
    [ "$dry" = 1 ] || [ -f "$g" ] || continue
    x "$E" "$g" --records "$tests" --max 2000 --exit-threshold 1.0 --json "$ev/ref-$(basename "$g" .gguf).json"
  done
  x "$A" "$baseline" --episodes "$episodes" --game "$games" --exit-threshold 1.0 --json "$ev/arena-baseline.json"
  x "$T" s2-truncation --config "$cfg" --tokenizer "$tok" --records "$records,data/s4/heldout" \
    --json "$ev/truncation.json"
  [ "$dry" = 1 ] || [ -d site/public/pkg ] || x scripts/build-wasm.sh
  # A parity failure (exit 1) is kept in wasm.json and read by the gate.
  x sh -c "node crates/sky-web/bench/parity-node.mjs '$model' '$ev/dump-$name.jsonl' --n 200 >'$ev/wasm.json' || true"
  x python3 -I scripts/s6-gate.py "$ev" --size "$size" --model "${model%.gguf}" \
    --pretrain-log "$run/pretrain/log.jsonl" --out "$ev/results.md"
}

steps=("$@")
[ ${#steps[@]} -eq 0 ] && steps=(train dagger exit eval)
for s in "${steps[@]}"; do
  case "$s" in
    lr) lr ;;
    train) train ;;
    dagger) dagger ;;
    exit) exits ;;
    eval | evaluate) eval_ ;;
    *) echo "unknown step $s (lr, train, dagger, exit, eval)" >&2; exit 2 ;;
  esac
done
