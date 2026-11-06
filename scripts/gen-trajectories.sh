#!/usr/bin/env bash
# skycmd-trajectories: complete episodes of the four games played by the published game models
# and by the oracle, every decision with the model's probabilities, the oracle's target, the
# reward and the events (`sky-arena --trajectories`). CPU only, a few minutes.
#
#   scripts/gen-trajectories.sh            # -> data/trajectories/<model>.jsonl (decisions)
#                                          #    data/trajectories/episodes/<model>.jsonl
#
# Episodes per game: flappy 20 (stopped alive at 50 pipes; a decision every frame), rescue 200,
# dodge 100, doom 100. Seed 2026: other courses than the arena reports (seed 7) and the training
# records (seed 1). s4-micro and s5-lite are left out: they are not game models.
set -euo pipefail
cd "$(dirname "$0")/.."
cargo build -q --release -p sky-arena
A=target/release/sky-arena
OUT=${OUT:-data/trajectories}
SEED=${SEED:-2026}
MODELS=${MODELS:-oracle s1-pico s2-nano s2-nano-doom s3-pico}
tmp=$OUT/.parts
rm -rf "$tmp" && mkdir -p "$tmp" "$OUT/episodes"
for m in $MODELS; do
  src=$m
  [ "$m" = oracle ] || src=site/public/models/$m.gguf
  : >"$OUT/$m.jsonl"
  : >"$OUT/episodes/$m.jsonl"
  for spec in flappy:20 rescue:200 dodge:100 doom:100; do
    g=${spec%%:*} n=${spec##*:}
    "$A" "$src" --trajectories "$tmp/$m-$g" --game "$g" --episodes "$n" --seed "$SEED" --max-pipes 50
    cat "$tmp/$m-$g/steps.jsonl" >>"$OUT/$m.jsonl"
    cat "$tmp/$m-$g/episodes.jsonl" >>"$OUT/episodes/$m.jsonl"
  done
done
rm -rf "$tmp"
# per model and game: episodes, decisions, agreement with the oracle, mean score and return,
# completion and crash rates (the table of the dataset card)
python3 -I - "$OUT" $MODELS <<'PY'
import json, sys
from collections import defaultdict
out, models = sys.argv[1], sys.argv[2:]
rows = []
for m in models:
    agree, dec = defaultdict(int), defaultdict(int)
    for line in open(f"{out}/{m}.jsonl"):
        r = json.loads(line)
        dec[r["game"]] += 1
        agree[r["game"]] += r["action"] == r["oracle_action"]
    eps = defaultdict(list)
    for line in open(f"{out}/episodes/{m}.jsonl"):
        e = json.loads(line)
        eps[e["game"]].append(e)
    for g, es in eps.items():
        n = len(es)
        rows.append({"model": m, "game": g, "episodes": n, "decisions": dec[g],
                     "agreement": round(agree[g] / max(dec[g], 1), 4),
                     "mean_score": round(sum(e["score"] for e in es) / n, 3),
                     "mean_return": round(sum(e["return"] for e in es) / n, 3),
                     "completion": round(sum(e["success"] for e in es) / n, 3),
                     "crash": round(sum(e["crashed"] for e in es) / n, 3)})
json.dump(rows, open(f"{out}/summary.json", "w"), indent=1)
for r in rows:
    print(r)
PY
wc -l "$OUT"/*.jsonl "$OUT"/episodes/*.jsonl
