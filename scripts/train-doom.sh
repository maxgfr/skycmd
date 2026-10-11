#!/usr/bin/env bash
# Teach s2-nano the Doom-like game while it keeps playing flappy, rescue and dodge.
#
#   1. oracle data: data/games/doom.jsonl (sky-games-gen --game doom);
#   2. a DAgger set for the three other games, collected once from s2-nano's own play
#      (data/dagger-doom/others), so the fine-tune keeps visiting their hard states;
#   3. round 0 fine-tunes the DAgger-improved runs/s2-nano-g2/decide/last on the HF decisions + every game, fully
#      ternary (--qat-frac 1.0: the model is already ternary), then calibrates, exports the
#      ternary GGUF and plays 50 easy levels;
#   4. each DAgger round lets the latest model play doom, labels the states it reaches with the
#      oracle (data/dagger-doom/rN) and fine-tunes again on everything, the newest round
#      weighted most. It stops when the gate holds: model completion >= 90% of the oracle's and
#      kill rate >= 70% of the oracle's over 50 episodes;
#   5. the final model goes to models/s2-nano-doom/ and site/public/models/s2-nano-doom.gguf.
#
# Early exit: the calibrated per-type thresholds let a 1-layer exit answer critical game states,
# which hurts closed-loop play. Every arena run uses τ = $TAU (default 0.95) and the published
# GGUF carries τ = $TAU for every question type (gguf-set).
#
# Usage: scripts/train-doom.sh [max-rounds] [minutes-per-round] [model: nano|micro]
# Environment: START=<round> resumes at a round (its init is the previous round's checkpoint).
set -euo pipefail
cd "$(dirname "$0")/.."
export DEVELOPER_DIR=${DEVELOPER_DIR:-/Applications/Xcode.app/Contents/Developer}
rounds=${1:-4} minutes=${2:-12} name=${3:-nano}
case "$name" in
  nano) vocab=2048 batch=128 ;;
  micro) vocab=4096 batch=64 ;;
  *) echo "unknown model $name" >&2; exit 1 ;;
esac
TRAIN_DIR=${TRAIN_TARGET_DIR:-$PWD/target/doom-train}
TOOLS_DIR=${TOOLS_TARGET_DIR:-$PWD/target/doom}
CARGO_TARGET_DIR=$TRAIN_DIR cargo build -q --release -j 4 -p sky-train
CARGO_TARGET_DIR=$TOOLS_DIR cargo build -q --release -p sky-arena -p sky-games -p sky-gguf \
  --bin sky-arena --bin sky-games-gen --bin gguf-set
T=$TRAIN_DIR/release/sky-train
S=$TOOLS_DIR/release/gguf-set
TAU=${TAU:-0.95}
A=$TOOLS_DIR/release/sky-arena
G=$TOOLS_DIR/release/sky-games-gen
cfg=configs/s2-$name.json
tok=data/tok/tok-s2-$vocab.json
base_init=${BASE_INIT:-runs/s2-$name-g2/decide/last}
base_model=${BASE_MODEL:-models/s2-$name-g2/skycmd-s2-$name-g2-ternary.gguf}
doom_data=${DOOM_DATA:-data/games/doom.jsonl}
out=runs/doom/s2-$name

[[ -f $doom_data ]] || $G --game doom --n 100000 --seed 1 --out "$doom_data"
[[ -f data/dagger-doom/others/dodge.jsonl ]] ||
  $A "$base_model" --exit-threshold "$TAU" --game flappy,rescue,dodge --dagger 15000 --seed 101 \
    --out data/dagger-doom/others

# One group per game file (the group name is the file stem) so the weights are explicit.
games="data/games/flappy.jsonl,data/games/rescue.jsonl,data/games/dodge.jsonl,$doom_data"
cal_records="data/decisions,$games"
base_arena=$out-base-arena.json
mkdir -p runs/doom
[[ -f $base_arena ]] ||
  $A "$base_model" --exit-threshold "$TAU" --game flappy,rescue,dodge --episodes 50 --seed 7 --json "$base_arena"

# Prints the gate verdict of an arena report and returns 0 when it holds: doom completion >= 90%
# and kill rate >= 70% of the oracle's, and the other games no worse than the base model (one
# episode in 50 of tolerance on completion, 5% on flappy's mean score).
gate() {
  python3 -I - "$1" "$base_arena" <<'EOF'
import json, sys
r, b = json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))
m, o = r["doom"]["model"], r["doom"]["oracle"]
c = m["completion_rate"] / max(o["completion_rate"], 1e-9)
k = m["kill_rate"] / max(o["kill_rate"], 1e-9)
ok = c >= 0.9 and k >= 0.7
print(f"gate doom: completion {m['completion_rate']:.2f} / oracle {o['completion_rate']:.2f} = {c:.2f} (>= 0.90), "
      f"kills {m['kill_rate']:.2f} / oracle {o['kill_rate']:.2f} = {k:.2f} (>= 0.70), "
      f"deaths {m['crashes']}, mean ticks {m['mean_steps']:.0f}")
for g in ("flappy", "rescue", "dodge"):
    mm, bb = r[g]["model"], b[g]["model"]
    if g == "flappy":
        good = mm["mean_score"] >= 0.95 * bb["mean_score"]
        print(f"gate {g}: mean pipes {mm['mean_score']:.1f} vs base {bb['mean_score']:.1f}, crashes {mm['crashes']} vs {bb['crashes']}")
    else:
        good = mm["completion_rate"] >= bb["completion_rate"] - 0.021
        print(f"gate {g}: completion {mm['completion_rate']:.2f} vs base {bb['completion_rate']:.2f}")
    ok = ok and good
print("gate:", "PASS" if ok else "not yet")
sys.exit(0 if ok else 1)
EOF
}

# DAgger share of the weights: the newest round gets half, the older ones split the rest.
share() { # total n k
  python3 -I -c "t, n, k = $1, $2, $3; print(round(t if n == 1 else (t / 2 if k == n - 1 else t / 2 / (n - 1)), 4))"
}

init=$base_init
start=${START:-0}
if (( start > 0 )); then init=$out-r$((start - 1))/decide/last; fi
final=""
for ((r = start; r <= rounds; r++)); do
  run=$out-r$r
  # DAgger from the previous round's model: doom (dN) and the three other games (oN).
  if (( r > 0 )); then
    prev=$out-r$((r - 1))/skycmd-s2-$name-doom-ternary.gguf
    [[ -f data/dagger-doom/r$r/doom.jsonl ]] ||
      $A "$prev" --exit-threshold "$TAU" --game doom --dagger 40000 --seed "$r" --out "data/dagger-doom/r$r"
    [[ -f data/dagger-doom/o$r/dodge.jsonl ]] ||
      $A "$prev" --exit-threshold "$TAU" --game flappy,rescue,dodge --dagger 12000 --seed $((100 + r)) \
        --out "data/dagger-doom/o$r"
  fi
  records="data/decisions,$games,data/dagger-doom/others"
  if (( r == 0 )); then
    # decisions 0.12, the three 2D games 0.30 + their DAgger 0.10, doom 0.48
    weights="decisions=0.12,flappy=0.11,rescue=0.09,dodge=0.10,others=0.10,doom=0.48"
  else
    # decisions 0.10, 2D games 0.27 + base DAgger 0.06, doom 0.15, doom DAgger 0.24, 2D DAgger 0.18
    weights="decisions=0.10,flappy=0.12,rescue=0.06,dodge=0.09,others=0.06,doom=0.15"
    for ((k = 1; k <= r; k++)); do
      records="$records,data/dagger-doom/r$k,data/dagger-doom/o$k"
      weights="$weights,r$k=$(share 0.24 "$r" $((k - 1))),o$k=$(share 0.18 "$r" $((k - 1)))"
    done
  fi
  echo "== doom round $r: init $init"
  echo "   records $records"
  echo "   weights $weights"
  $T s2-decide --config "$cfg" --init "$init" --tokenizer "$tok" --records "$records" --weights "$weights" \
    --batch "$batch" --minutes "$minutes" --qat-frac 1.0 --out "$run/decide"
  $T s2-calibrate --run "$run/decide/last" --records "$cal_records"
  gguf=$run/skycmd-s2-$name-doom-ternary.gguf
  $T s2-export --run "$run/decide/last" --format ternary --out "$gguf"
  $S "$gguf" "$gguf" skycmd-s2.exit_threshold.choice="$TAU" skycmd-s2.exit_threshold.score="$TAU" \
    skycmd-s2.exit_threshold.noul="$TAU"
  $A "$gguf" --exit-threshold "$TAU" --episodes 50 --seed 7 --json "$run/arena.json"
  final=$gguf
  if gate "$run/arena.json"; then
    break
  fi
  init=$run/decide/last
done

# Publish the last model.
mkdir -p "models/s2-$name-doom" site/public/models
cp "$final" "models/s2-$name-doom/skycmd-s2-$name-doom-ternary.gguf"
cp "$final" "site/public/models/s2-$name-doom.gguf"
cp "$(dirname "$final")/arena.json" "models/s2-$name-doom/skycmd-s2-$name-doom-ternary.arena.json"
echo "done: models/s2-$name-doom/skycmd-s2-$name-doom-ternary.gguf (arena: next to it; base: $base_arena)"
