#!/usr/bin/env bash
# Training speed of this machine, for budgets in minutes that mean the same steps on any Mac
# (docs/s6-prereg.md, amendment 4): 300 steps of s2-decide on the s6-nano config (decision
# training with QAT, as in the final runs), from scratch, on data/decisions.
# Usage: scripts/speed-probe.sh [probe]    writes runs/speed/<host>.json {steps_per_sec, ...}
#        scripts/speed-probe.sh scale      prints TIME_SCALE = the M5's steps/s (configs/speed-m5.json,
#                                          measured once on the M5) / this host's
#   HOST=$(hostname -s)  STEPS=300
set -euo pipefail
export DEVELOPER_DIR=${DEVELOPER_DIR:-/Applications/Xcode.app/Contents/Developer}
export CARGO_TARGET_DIR=${CARGO_TARGET_DIR:-$PWD/target/s3-train}
host=${HOST:-$(hostname -s)}
steps=${STEPS:-300}
out=runs/speed/$host.json

probe() {
  cargo build -q --release -p sky-train
  local dir=runs/speed/probe-$host
  rm -rf "$dir"
  "$CARGO_TARGET_DIR/release/sky-train" s2-decide --config configs/s6-nano.json \
    --tokenizer data/tok/tok-s6-2048.json --records data/decisions --batch 128 --steps "$steps" \
    --qat-frac 0.9 --loop-dropout 0.25 --length-pool 16 --pad-to 32 --out "$dir" >/dev/null
  python3 -I - "$dir/last/state.json" "$out" "$host" "$steps" <<'EOF'
import json, platform, subprocess, sys
state, out, host, steps = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
secs = json.load(open(state))["elapsed_secs"]
chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
r = {"host": host, "chip": chip, "macos": platform.mac_ver()[0], "config": "configs/s6-nano.json",
     "steps": steps, "secs": secs, "steps_per_sec": steps / secs}
json.dump(r, open(out, "w"), indent=2)
print(f"{host} ({chip}): {r['steps_per_sec']:.3f} steps/s")
EOF
  rm -rf "$dir"
}

scale() {
  [ -f configs/speed-m5.json ] || { echo "no configs/speed-m5.json (the reference, measured on the M5)" >&2; exit 1; }
  [ -f "$out" ] || probe >&2
  python3 -I -c '
import json, sys
ref, me = (json.load(open(p))["steps_per_sec"] for p in sys.argv[1:])
print(f"{ref / me:.4g}")' configs/speed-m5.json "$out"
}

case "${1:-probe}" in
  probe) mkdir -p runs/speed && probe ;;
  scale) scale ;;
  *) echo "usage: $0 [probe|scale]" >&2; exit 2 ;;
esac
