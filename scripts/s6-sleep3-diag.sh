#!/usr/bin/env bash
# The CPU experiments of docs/s6-sleep3-diagnosis.md: the C4 setting of scripts/train-sleep2.sh
# (300 steps from the pico base, recall as the only sequence game, 8 sequences a batch, rate 2e-3,
# N = 1, QAT from the first step, seed 0), one run per call, with the flags under test.
#
#   scripts/s6-sleep3-diag.sh <name> [s2-decide flags]    -> runs/s6-sleep3/<name>
#
#   scripts/s6-sleep3-diag.sh baseline --mem-lr-mult 10 --memory-trace
#   scripts/s6-sleep3-diag.sh gates-frozen --mem-lr-mult 10 --mem-gate-lr-mult 0 --memory-trace
#   scripts/s6-sleep3-diag.sh gates-frozen-zero --mem-lr-mult 10 --mem-gate-lr-mult 0 --memory-read-off
#   scripts/s6-sleep3-diag.sh oracle           # builds the oracle sequences, read held at zero
#
# Always on the CPU (--cpu) at nice 19, one run at a time: it never shares the GPU with a training.
# Train sequences only; no dev or test seed of the probes is read.
#
#   CFG        the model config (default runs/s6-sleep2/checks/sleep2.json: v2's memory, form frame);
#              CFG=aux adds the auxiliary head (memory_aux) to it, CFG=slot takes the slot write
#              (memory_write: slot, one key and one value per frame), CFG=slot-aux both
#   SEQ        the recall sequences (default runs/s6-sleep/seq/recall.jsonl, v1's)
#   STOP       stop after that many steps (the schedule stays that of STEPS steps)
#   STEPS      the steps of the schedule (default 300, C4's)
#   SEED       the seed of the run (default 0, C4's)
#   DATA       the folder of the stateless records (default data)
#   PICO_BASE  runs/s6lr/softmax-adamw-x2/decide/last
#   SKY_TRAIN  a sky-train binary to use as is
set -euo pipefail
export DEVELOPER_DIR=${DEVELOPER_DIR:-/Applications/Xcode.app/Contents/Developer}
export CARGO_TARGET_DIR=${CARGO_TARGET_DIR:-$PWD/target/s3-train}
export SKY_DUTY=${SKY_DUTY:-0.5}
export SKY_COOL=${SKY_COOL-45:15}
export SKY_BRAKE_FILE=${SKY_BRAKE_FILE-$PWD/runs/brake}
export SKY_MLX_CACHE_GB=${SKY_MLX_CACHE_GB:-2}
T=${SKY_TRAIN:-$CARGO_TARGET_DIR/release/sky-train}
[ $# -ge 1 ] || { sed -n '2,26p' "$0" >&2; exit 2; }
name=$1
shift
root=runs/s6-sleep3
base=${PICO_BASE:-runs/s6lr/softmax-adamw-x2/decide/last}
seq=${SEQ:-runs/s6-sleep/seq/recall.jsonl}
cfg=${CFG:-runs/s6-sleep2/checks/sleep2.json}
data=${DATA:-data}
out=$root/$name
mkdir -p "$root/configs"

case "$cfg" in
aux | slot | slot-aux)
  form=$cfg
  cfg=$root/configs/sleep3-$form.json
  python3 -I - runs/s6-sleep2/checks/sleep2.json "$cfg" "$form" <<'EOF'
import json, sys
c = json.load(open(sys.argv[1]))
form = sys.argv[3]
c["name"] = "s6-sleep3-pico-" + form
if "aux" in form:
    c["memory_aux"] = True
if "slot" in form:
    c["memory_write"] = "slot"
    del c["memory_value"]  # one value per frame: the value forms do not apply
json.dump(c, open(sys.argv[2], "w"), indent=2)
EOF
  ;;
esac

flags=("$@")
if [ "$name" = oracle ]; then
  # The ceiling of what follows the memory: on every train ask frame the state itself names the
  # colour asked (`colour none` -> `colour <answer>`), and the read is held at zero. How fast the
  # base learns to answer from a colour it is handed says how fast the read-out could be used.
  seq=$root/seq-oracle/recall.jsonl
  if [ ! -f "$seq" ]; then
    mkdir -p "$(dirname "$seq")"
    python3 -I - "${SEQ:-runs/s6-sleep/seq/recall.jsonl}" "$seq" <<'EOF'
import json, sys
with open(sys.argv[1]) as src, open(sys.argv[2], "w") as dst:
    for line in src:
        r = json.loads(line)
        w = r["state"].split()
        if r.get("split", "train") == "train" and w[w.index("ask") + 1] != "none":
            answer = r["opts"][r["target"].index(max(r["target"]))].split(":")[0]
            w[w.index("colour") + 1] = answer
            r["state"] = " ".join(w)
        dst.write(json.dumps(r, separators=(",", ":")) + "\n")
EOF
  fi
  flags=(--mem-lr-mult 10 --memory-read-off ${flags[@]+"${flags[@]}"})
fi

stop=()
[ -n "${STOP:-}" ] && stop=(--stop-after "$STOP")
rm -rf "$out"
mkdir -p "$out"
echo "== $out: ${flags[*]-}"
nice -n 19 "$T" --cpu s2-decide --config "$cfg" --init "$base" --tokenizer "$base/tokenizer.json" \
  --records "$data/decisions,$data/games,$data/numeric" --weights decisions=0.3,games=0.6,numeric=0.1 \
  --batch 128 --steps "${STEPS:-300}" --seed "${SEED:-0}" --qat-frac 1.0 --loop-dropout 0.25 --length-pool 16 --pad-to 32 \
  --out "$out" --step-mem-gb 3 --adam-lr 0.002 --no-muon --seq-log \
  --sleep-records "$seq" --sleep-weights recall=1 --sleep-batch 8 --seq-frac 0.9 --sleep-passes 1 \
  --no-compile --val-max 256 ${stop[@]+"${stop[@]}"} ${flags[@]+"${flags[@]}"} >"$out/stdout.txt" 2>&1
python3 -I scripts/s6-sleep3-trace.py c4 "$out"
