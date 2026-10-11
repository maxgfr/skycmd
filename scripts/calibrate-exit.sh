#!/usr/bin/env bash
# Closed-loop early-exit calibration (docs/SPEC-S3.md §5): tau is chosen on the states the model
# visits while playing, not on i.i.d. validation questions.
#   1. play N episodes per game at full depth (tau = 1.0): the reference metrics;
#   2. sweep tau in {0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98} on the same episodes (same seed);
#   3. keep the smallest tau whose game metrics (flappy score, rescue rate, dodge completion and
#      score) stay within 2% of full depth with no more crashes or vetoes than full depth;
#   4. write it as skycmd-s2.exit_threshold.{choice,score,noul} with gguf-set.
# Usage: scripts/calibrate-exit.sh <model.gguf> [out.gguf]   (default: rewrite <model.gguf> in place)
#   EPISODES=20  GAMES=flappy,rescue,dodge  SEED=7
set -euo pipefail
export DEVELOPER_DIR=${DEVELOPER_DIR:-/Applications/Xcode.app/Contents/Developer}
model=$1
out=${2:-$1}
A=${A:-target/s3-arena/release/sky-arena}
episodes=${EPISODES:-20}
games=${GAMES:-flappy,rescue,dodge}
seed=${SEED:-7}
dir=${CALIB_DIR:-runs/calibrate-exit/$(basename "${model%.gguf}")}
mkdir -p "$dir"
for tau in 1.0 0.5 0.6 0.7 0.8 0.85 0.9 0.95 0.98; do
  echo "== tau $tau"
  $A "$model" --episodes "$episodes" --seed "$seed" --game "$games" --exit-threshold "$tau" \
    --json "$dir/tau-$tau.json" | grep -v '^$' || true
done
tau=$(python3 - "$dir" <<'EOF'
import json, sys
d = sys.argv[1]
taus = ["0.5", "0.6", "0.7", "0.8", "0.85", "0.9", "0.95", "0.98"]
load = lambda t: json.load(open(f"{d}/tau-{t}.json"))
ref = load("1.0")
games = [g for g, v in ref.items() if isinstance(v, dict) and "model" in v]
def ok(run):
    for g in games:
        a, b = ref[g]["model"], run[g]["model"]
        for k in ("mean_score", "rescue_rate", "completion_rate"):
            if b[k] < a[k] - 0.02 * abs(a[k]) - 1e-9:
                return False, f"{g} {k} {b[k]:.3f} < {a[k]:.3f}"
        if b["crashes"] > a["crashes"] or b["vetoes"] > a["vetoes"]:
            return False, f"{g} crashes {b['crashes']} vetoes {b['vetoes']} (full depth {a['crashes']} / {a['vetoes']})"
    return True, ""
rows = []
best = None
for t in ["1.0"] + taus:
    run = load(t)
    good, why = ok(run) if t != "1.0" else (True, "reference")
    layers = [run[g]["mean_layers_used"] or 0 for g in games]
    us = [run[g]["p50_us"] for g in games]
    m = {g: run[g]["model"] for g in games}
    summary = "; ".join(
        f"{g} score {m[g]['mean_score']:.2f} resc {m[g]['rescue_rate']:.2f} compl {m[g]['completion_rate']:.2f} "
        f"crash {m[g]['crashes']} veto {m[g]['vetoes']} layers {l:.2f} p50 {u:.0f}us"
        for g, l, u in zip(games, layers, us))
    print(f"tau {t:>4}: {'ok ' if good else 'NO '} {summary} {why}", file=sys.stderr)
    rows.append({"tau": float(t), "ok": good, "why": why,
                 "mean_layers": {g: l for g, l in zip(games, layers)},
                 "p50_us": {g: u for g, u in zip(games, us)},
                 "metrics": m})
    if t != "1.0" and good and best is None:
        best = t
json.dump({"chosen": float(best) if best else 1.0, "rows": rows}, open(f"{d}/summary.json", "w"), indent=2)
print(best or "1.0")
EOF
)
echo "== chosen tau = $tau (smallest within 2% of full depth, no extra crashes / vetoes)"
CARGO_TARGET_DIR=${GGUF_TARGET_DIR:-target/s3-train} cargo run -q --release -p sky-gguf --bin gguf-set -- \
  "$model" "$out" "skycmd-s2.exit_threshold.choice=$tau" "skycmd-s2.exit_threshold.score=$tau" \
  "skycmd-s2.exit_threshold.noul=$tau"
echo "summary: $dir/summary.json"
