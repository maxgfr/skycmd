#!/usr/bin/env bash
# The gate of s6-route v7 (docs/s6-route7-prereg.md §7 to §9, as v6 §7, §8 and §14 items 3 and 8),
# after the training (scripts/train-route7.sh window pilot main export). One entry point, its
# stages in this order, each refused until the one before it has written its marker
# (runs/s6-route7/gate-stages/<stage>.done):
#
#   fit    β and τ_ask7 of the pair on v5_tools only (scripts/s6-route6-fit.py, v3's rules): the
#          model part's first-exit dump on v5_tools, potion-cos-4bit on v5_tools (v2's fitted
#          values), then runs/s6-route7/fit.json. Refused once T5 is open.
#   embed  the embedding file of v7, models/s6-route7/skycmd-s6-route7-embed.gguf: the published
#          s6-route-embed.gguf with the fit's β and τ_ask7 (sky-route embed-params; the ask
#          threshold keeps v3's key tau_ask3, which the engine reads). The published file is read
#          only and its sha256 checked before and after. Refused once T5 is open.
#   open   the moment T5 becomes readable, once: after checking that the pilot's summary exists and
#          did not abort, that the exported model, the fit and the embedding file exist, and that
#          every file of runs/s6-route7/test-files.json has its sha256, it writes
#          runs/s6-route7/FRESH-OPEN (the hashes of the model, the embedding file and the fit).
#          test-files.json's own sentence names runs/s6-route5/FRESH-OPEN, the builder's text for
#          v5's rule (amendment 2): for v7 the file is runs/s6-route7/FRESH-OPEN.
#   read   every reading of §7: on T5-tools, its masked copy and the two probe copies, the pair
#          (and its fast mode), its model part (full depth and first exit), every baseline of v3
#          (v2's fitted values), the published pair (v3) for comparison; the blind copies of
#          T5-tools and of its masked copy (B0 and B1, the kept placeholder) and their readings:
#          the model part in B0 and B1, the pair, its embedding part and every baseline in B0;
#          the pair on T2-tools (G5, s6-nano v3's fresh file, open since v3); the pair on v5_tools
#          and its copies, ordinary and blind (reported). The statistics: runs/s6-route7/readings.json.
#   g4     the Python reference of the pair under D4 on T5-tools (scripts/s6-route7-reference.py),
#          the engine's WASM bench on 200 of its slates (crates/sky-web/bench/route3-node.mjs
#          --render scores-in-options), G4's JSON runs/s6-route7/g4.json
#   gate   scripts/s6-route6-gate.py with v7's paths: runs/s6-route7/gate.json and GATE.md
#
# T5 is read once: `open`, `read`, `g4` and `gate` each refuse to run again once their marker is
# written, and all four refuse once runs/s6-route7/gate.json exists. Every command that reads T5
# runs after scripts/route/t5-guard.py (FRESH-OPEN and the recorded hashes).
#
#   ROUTE_DATA=../skycmd-data/s6-route  B7=$ROUTE_DATA/build7  T5=$ROUTE_DATA/toucan5
#   RUNS=runs/s6-route7  MODELS=models/s6-route7  PILOT=runs/s6-route7-pilot/summary.json (the
#   tests' fixtures only; scripts/route-baselines.py --split t5 reads runs/s6-route7 itself)
#   SKY_BIN (release binaries already built: sky-data, sky-eval, sky-route, gguf-set; else built
#   here, never sky-train)  MIN_FREE_GB=3  DRY_RUN=1 (prints every command, writes nothing)
#
# Usage: scripts/route7-gate.sh [stage ...]   (default: fit embed open read g4 gate)
set -euo pipefail
cd "$(dirname "$0")/.."
export DEVELOPER_DIR=${DEVELOPER_DIR:-/Applications/Xcode.app/Contents/Developer}
export CARGO_TARGET_DIR=${CARGO_TARGET_DIR:-$PWD/target/s3-train}
dry=${DRY_RUN:-0}
data=${ROUTE_DATA:-../skycmd-data/s6-route}
export ROUTE_DATA=$data
b7=${B7:-$data/build7}
export T5=${T5:-$data/toucan5}
fresh=$T5/fresh
min_free=${MIN_FREE_GB:-3}
B=${SKY_BIN:-$CARGO_TARGET_DIR/release}
PY=$data/envs/route/bin/python
run=${RUNS:-runs/s6-route7}
markers=$run/gate-stages
m=${MODELS:-models/s6-route7}
model=$m/skycmd-s6-route7-nano-ternary.gguf
full=$run/full-depth.gguf
embed7=$m/skycmd-s6-route7-embed.gguf
published=site/public/models/s6-route-embed.gguf
published_sha=c53fb76dc8a56a696d859e80b7383d7e9db168e54c024525f220851fbacda67a
v3_model=site/public/models/s6-route-nano.gguf
pilot=${PILOT:-runs/s6-route7-pilot/summary.json}
ev=$data/eval/t5
val_ev=$data/eval/route7-val
t2=$data/build/fresh/f_tools.jsonl
systems=(potion-cos potion-cos-4bit static-ft pair-lr bm25 minilm-cos toolrouter name-prior)
plain=(t5_tools t5_tools_masked t5_tools_rw t5_tools_masked_rw)
blinds=(t5_tools t5_tools_masked)
stages=(fit embed open read g4 gate)

x() {
  if [ "$dry" = 1 ]; then
    printf '+'
    printf ' %q' "$@"
    printf '\n'
  else
    nice -n 19 "$@"
  fi
}
note() { echo "# $*"; }
die() { echo "$*" >&2; exit "${2:-1}"; }
need() {
  if [ "$dry" = 1 ]; then
    note "requires $markers/$1.done"
    return
  fi
  [ -f "$markers/$1.done" ] || die "no $markers/$1.done: scripts/route7-gate.sh $1 first"
}
done_() {
  if [ "$dry" = 1 ]; then
    note "writes $markers/$1.done"
  else
    mkdir -p "$markers"
    date '+%F %T' >"$markers/$1.done"
  fi
}
# the fresh test is read once
once() {
  if [ "$dry" = 1 ]; then
    note "refused if $run/gate.json or $markers/$1.done exists (T5 is read once)"
    return
  fi
  [ ! -f "$run/gate.json" ] || die "$run/gate.json exists: T5 has been read and gated, nothing reads it again" 3
  [ ! -f "$markers/$1.done" ] || die "$markers/$1.done exists: the stage $1 ran, T5 is read once" 3
}
closed() {
  if [ "$dry" = 1 ]; then
    note "refused if $run/FRESH-OPEN exists (nothing of the system changes once T5 is open)"
    return
  fi
  [ ! -f "$run/FRESH-OPEN" ] || die "$run/FRESH-OPEN exists: the fit and the embedding file are fixed before T5 is opened" 3
}
t5_guard() {
  if [ "$dry" = 1 ]; then
    x python3 -I scripts/route/t5-guard.py --open "$run/FRESH-OPEN" --files "$run/test-files.json"
  else
    python3 -I scripts/route/t5-guard.py --open "$run/FRESH-OPEN" --files "$run/test-files.json" >/dev/null ||
      die "T5 is refused (scripts/route/t5-guard.py)" 3
  fi
}
sha() { shasum -a 256 "$1" | cut -d' ' -f1; }
exists() {
  local f
  for f in "$@"; do
    if [ "$dry" = 1 ]; then
      note "requires $f"
    else
      [ -e "$f" ] || die "no $f"
    fi
  done
}

fit() {
  closed
  exists "$model" "$full" "$run/v5_tools.dump.jsonl" "$b7/val/v5_tools.model.jsonl"
  note "the model part's first exit on v5_tools (the fast mode of the fit)"
  x "$B/sky-eval" "$full" --records "$b7/val/v5_tools.model.jsonl" --split val --max 100000 --exit-threshold 0 \
    --dump-probs "$run/v5_tools.exit0.jsonl" --json "$run/v5_tools.exit0.eval.json"
  note "potion-cos-4bit on v5_tools (v2's fitted values)"
  x env BUILD="$data/build" BUILD4="$data/build4" BUILD5="$b7" "$PY" -I scripts/route-baselines.py score \
    --system potion-cos-4bit --split v5val --files v5_tools --out-root "$val_ev/v5val"
  x python3 -I scripts/s6-route6-fit.py --model "$run/v5_tools.dump.jsonl" --exit0 "$run/v5_tools.exit0.jsonl" \
    --partner "$val_ev/v5val/potion-cos-4bit/v5_tools.jsonl" --out "$run/fit.json"
  done_ fit
}

embed() {
  need fit
  closed
  exists "$published" "$run/fit.json"
  if [ "$dry" != 1 ]; then
    [ "$(sha "$published")" = "$published_sha" ] || die "$published is not the published embedding file"
  fi
  x "$B/sky-route" embed-params --embed "$published" --fit "$run/fit.json" --out "$embed7" --name s6-route7-embed \
    --params-from "runs/s6-route7/fit.json (scripts/s6-route6-fit.py on v5_tools; docs/s6-route7-prereg.md)"
  if [ "$dry" != 1 ]; then
    [ "$(sha "$published")" = "$published_sha" ] || die "$published changed: it must stay byte-identical"
    python3 -I -c 'import json, sys; json.dump({"about": "s6-route v7: the embedding file of the pair (scripts/route7-gate.sh embed)",
      "embedding": sys.argv[1], "sha256": sys.argv[2], "bytes": int(sys.argv[3]), "from": sys.argv[4], "from_sha256": sys.argv[5],
      "fit": json.load(open(sys.argv[6]))}, open(sys.argv[7], "w"), indent=1)' \
      "$embed7" "$(sha "$embed7")" "$(wc -c <"$embed7" | tr -d ' ')" "$published" "$published_sha" "$run/fit.json" "$run/embed.json"
  fi
  done_ embed
}

open_() {
  need embed
  once open
  closed
  exists "$pilot" "$model" "$embed7" "$run/fit.json" "$run/test-files.json" "$run/placeholder.json"
  note "the pilot did not abort; every file of $run/test-files.json has its sha256; then $run/FRESH-OPEN"
  if [ "$dry" != 1 ]; then
    python3 -I - "$pilot" "$run/test-files.json" "$fresh" "$model" "$embed7" "$run/fit.json" "$run/FRESH-OPEN" <<'EOF'
import datetime, hashlib, json, os, sys
pilot, files, fresh, model, embed, fit, out = sys.argv[1:]
s = json.load(open(pilot))
if s.get("abort"):
    sys.exit("the pilot rule aborted: %s (no main run, T5 stays closed)" % s.get("why"))
bar = s.get("bar (rule 1, from the amendment of 4.5)")
if bar is None or abs(bar - 0.578147) > 1e-9:
    sys.exit(f"the pilot's bar is {bar}, not amendment 2's 0.578147 (POTION_BAR)")
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()
want = json.load(open(files))["files"]
for name, h in sorted(want.items()):
    p = os.path.join(fresh, name)
    if not os.path.isfile(p) or sha(p) != h:
        sys.exit(f"{p}: not the file of {files}")
json.dump({"about": "s6-route v7 (docs/s6-route7-prereg.md): T5 opened by scripts/route7-gate.sh open, once; the system read on it",
           "opened": datetime.datetime.now().isoformat(timespec="seconds"),
           "model": {"file": model, "sha256": sha(model)}, "embedding": {"file": embed, "sha256": sha(embed)},
           "fit": {"file": fit, "sha256": sha(fit)}, "T5": want}, open(out, "w"), indent=1)
print(f"{out}: T5 is open ({len(want)} files)")
EOF
  fi
  t5_guard
  done_ open
}

# the pair, its parts and the baselines on one file: <slates> <name> <out dir>
pair() {
  x "$B/sky-route" fuse --model "$model" --embed "$embed7" --slates "$1" --out "$3/pair/$2.jsonl"
}

read_() {
  need open
  once read
  t5_guard
  x mkdir -p "$ev/render" "$ev/model" "$ev/pair" "$ev/pair-fast" "$ev/pair-v3" "$ev/t2/pair" "$val_ev/pair"
  local f
  note "the blind copies of T5-tools and of its masked copy (B0, B1), the kept placeholder, after FRESH-OPEN"
  for f in "${blinds[@]}"; do
    x "$B/sky-data" route blind --in "$fresh/$f.jsonl" --out "$fresh/$f" --placeholder "$run/placeholder.json" \
      --render scores-in-options --open "$run/FRESH-OPEN"
  done
  note "the model part (full depth, and the first exit on T5-tools) on D4's model copies"
  for f in "${plain[@]}"; do
    x "$B/sky-data" route render --in "$fresh/$f.jsonl" --out "$ev/render/$f.model.jsonl" --render scores-in-options
    x "$B/sky-eval" "$full" --records "$ev/render/$f.model.jsonl" --split fresh --max 100000 --exit-threshold 1.0 \
      --dump-probs "$ev/model/$f.jsonl" --json "$ev/model/$f.eval.json"
  done
  x "$B/sky-eval" "$full" --records "$ev/render/t5_tools.model.jsonl" --split fresh --max 100000 --exit-threshold 0 \
    --dump-probs "$ev/model/t5_tools.exit0.jsonl" --json "$ev/model/t5_tools.exit0.eval.json"
  for f in "${blinds[@]}"; do
    x "$B/sky-eval" "$full" --records "$fresh/$f.blind.model.jsonl" --split fresh --max 100000 --exit-threshold 1.0 \
      --dump-probs "$ev/model/$f.blind.jsonl" --json "$ev/model/$f.blind.eval.json"
    x "$B/sky-eval" "$full" --records "$fresh/$f.blind-bare.model.jsonl" --split fresh --max 100000 --exit-threshold 1.0 \
      --dump-probs "$ev/model/$f.blind-bare.jsonl" --json "$ev/model/$f.blind-bare.eval.json"
  done
  note "the pair (and its fast mode), and the published pair (v3), ordinary and in B0"
  for f in "${plain[@]}" t5_tools.blind t5_tools_masked.blind; do
    pair "$fresh/$f.jsonl" "$f" "$ev"
    x "$B/sky-route" fuse --model "$model" --embed "$embed7" --slates "$fresh/$f.jsonl" --fast --out "$ev/pair-fast/$f.jsonl"
    x "$B/sky-route" fuse --model "$v3_model" --embed "$published" --slates "$fresh/$f.jsonl" --out "$ev/pair-v3/$f.jsonl"
  done
  note "every baseline of v3 on T5 and its blind copies (v2's fitted values; scripts/route-baselines.py --split t5)"
  local s
  for s in "${systems[@]}"; do
    x "$PY" -I scripts/route-baselines.py score --system "$s" --split t5
  done
  note "G5: the pair on T2-tools (s6-nano v3's fresh file, open since v3)"
  if [ "$dry" != 1 ]; then
    python3 -I -c 'import os, sys; sys.path.insert(0, "scripts/route"); import common
try: common.fresh_file("f_tools")
except common.Refused as e: sys.exit("refused: %s" % e)' || die "T2-tools is refused" 3
  fi
  pair "$t2" f_tools "$ev/t2"
  note "the pair on v5_tools and its copies, ordinary and blind (validation, reported)"
  for f in v5_tools v5_tools_masked v5_tools_rw v5_tools_masked_rw v5_tools.blind v5_tools_masked.blind; do
    pair "$b7/val/$f.jsonl" "$f" "$val_ev"
  done
  readings
  done_ read
}

# the statistics of every reading: runs/s6-route7/readings.json (accuracies, and the blind ones)
readings() {
  if [ "$dry" = 1 ]; then
    x python3 -I - "$ev" "$fresh" "$val_ev" "$b7/val" "$run/readings.json"
    return
  fi
  python3 -I - "$ev" "$fresh" "$val_ev" "$b7/val" "$run/readings.json" "${systems[@]}" <<'EOF'
import importlib.util, json, os, sys
ev, fresh, val_ev, val, out, *systems = sys.argv[1:]
spec = importlib.util.spec_from_file_location("b6", os.path.join("scripts", "s6-route6-blind.py"))
b6 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b6)
load = b6.load
servers = {x["uid"]: x["server"] for x in load(os.path.join(fresh, "t5_tools.servers.jsonl"))}
def acc(slates, dump, field="probs"):
    ok = [max(range(len(d[field])), key=lambda i: d[field][i]) == r["target"].index(1.0) for r, d in zip(slates, dump)]
    return {"slates": len(ok), "accuracy": sum(ok) / max(len(ok), 1)}
res = {"T5": {}, "T5 blind": {}, "validation": {}}
for f in ("t5_tools", "t5_tools_masked", "t5_tools_rw", "t5_tools_masked_rw"):
    sl = load(os.path.join(fresh, f + ".jsonl"))
    r = {"pair": acc(sl, load(os.path.join(ev, "pair", f + ".jsonl"))),
         "pair, fast mode": acc(sl, load(os.path.join(ev, "pair-fast", f + ".jsonl"))),
         "model part": acc(sl, load(os.path.join(ev, "model", f + ".jsonl"))),
         "embedding part (potion-cos-4bit, in the pair)": acc(sl, load(os.path.join(ev, "pair", f + ".jsonl")), "embedding"),
         "published pair (v3)": acc(sl, load(os.path.join(ev, "pair-v3", f + ".jsonl")))}
    for s in systems:
        p = os.path.join(ev, s, f + ".jsonl")
        if os.path.exists(p):
            r[s] = acc(sl, load(p))
    res["T5"][f] = r
for f in ("t5_tools", "t5_tools_masked"):
    sl = load(os.path.join(fresh, f + ".blind.jsonl"))
    r = {}
    for name, slates, dump, field in (
            ("model part, B0", f + ".blind.model.jsonl", ("model", f + ".blind.jsonl"), "probs"),
            ("model part, B1", f + ".blind-bare.model.jsonl", ("model", f + ".blind-bare.jsonl"), "probs"),
            ("pair, B0", f + ".blind.jsonl", ("pair", f + ".blind.jsonl"), "probs"),
            ("the pair's model part, B0", f + ".blind.jsonl", ("pair", f + ".blind.jsonl"), "model"),
            ("embedding part, B0", f + ".blind.jsonl", ("pair", f + ".blind.jsonl"), "embedding"),
            ("published pair (v3), B0", f + ".blind.jsonl", ("pair-v3", f + ".blind.jsonl"), "probs")):
        r[name], _ = b6.read(load(os.path.join(fresh, slates)), load(os.path.join(ev, *dump)), field, servers, "server")
    for s in systems:
        p = os.path.join(ev, s, f + ".blind.jsonl")
        if os.path.exists(p):
            r[s + ", B0"], _ = b6.read(sl, load(p), "probs", servers, "server")
    res["T5 blind"][f] = r
for f in ("v5_tools", "v5_tools_masked", "v5_tools_rw", "v5_tools_masked_rw", "v5_tools.blind", "v5_tools_masked.blind"):
    sl = load(os.path.join(val, f + ".jsonl"))
    d = load(os.path.join(val_ev, "pair", f + ".jsonl"))
    res["validation"][f] = b6.read(sl, d, "probs")[0] if f.endswith(".blind") else acc(sl, d)
json.dump({"about": "s6-route v7 (docs/s6-route7-prereg.md 7 and 9): every reading of scripts/route7-gate.sh read; reported, the gate is gate.json",
           "readings": res}, open(out, "w"), indent=1)
print(json.dumps({f: {k: round(v["accuracy"], 4) for k, v in r.items()} for f, r in res["T5"].items()}))
EOF
}

g4() {
  need read
  once g4
  t5_guard
  x mkdir -p "$ev/g4"
  x python3 -I scripts/s6-route7-reference.py reference --slates "$fresh/t5_tools.jsonl" \
    --rendered "$ev/render/t5_tools.model.jsonl" --model "$ev/model/t5_tools.jsonl" \
    --embedding "$ev/potion-cos-4bit/t5_tools.jsonl" --fit "$run/fit.json" --out "$ev/g4/t5_tools.fused.jsonl"
  if [ "$dry" = 1 ]; then
    note "scripts/build-wasm.sh if site/public/pkg does not exist"
  else
    [ -d site/public/pkg ] || nice -n 19 scripts/build-wasm.sh
  fi
  note "the WASM bench, 200 slates of T5-tools, D4 (a parity failure exits 1 and is kept in the JSON)"
  if [ "$dry" = 1 ]; then
    x node crates/sky-web/bench/route3-node.mjs "$model" "$embed7" "$fresh/t5_tools.jsonl" "$ev/g4/t5_tools.fused.jsonl" \
      --n 200 --render scores-in-options
  else
    nice -n 19 node crates/sky-web/bench/route3-node.mjs "$model" "$embed7" "$fresh/t5_tools.jsonl" \
      "$ev/g4/t5_tools.fused.jsonl" --n 200 --render scores-in-options >"$ev/g4/wasm7.json" || true
  fi
  x python3 -I scripts/s6-route7-reference.py g4 --bench "$ev/g4/wasm7.json" --out "$run/g4.json"
  done_ g4
}

gate() {
  need g4
  once gate
  t5_guard
  local st=0
  local args=(--root "$ev" --slates "$fresh" --test-files "$run/test-files.json" --open "$run/FRESH-OPEN"
    --t2 "$ev/t2/pair/f_tools.jsonl" --t2-slates "$t2" --g4 "$run/g4.json" --chance 0.14285714285714285
    --out "$run/gate.json" --md "$run/GATE.md")
  if [ "$dry" = 1 ]; then
    x python3 -I scripts/s6-route6-gate.py "${args[@]}"
  else
    nice -n 19 python3 -I scripts/s6-route6-gate.py "${args[@]}" || st=$?
    [ -s "$run/gate.json" ] || die "the gate wrote no $run/gate.json (exit $st)"
    note "the gate's verdict (exit $st: 0 passes, 1 fails): $run/GATE.md"
  fi
  done_ gate
}

run_stages=("$@")
[ ${#run_stages[@]} -eq 0 ] && run_stages=("${stages[@]}")
for s in "${run_stages[@]}"; do
  case " ${stages[*]} " in
    *" $s "*) ;;
    *) die "unknown stage $s (${stages[*]})" 2 ;;
  esac
done
if [ "$dry" = 1 ]; then
  note "DRY_RUN: nothing is built, read or written"
else
  kb=$(df -Pk "$data" | awk 'NR == 2 { print $4 }')
  [ "$kb" -ge $((min_free * 1024 * 1024)) ] || die "below the disk floor of $min_free GB on $data" 4
  [ -n "${SKY_BIN:-}" ] || nice -n 19 cargo build -q --release -p sky-data -p sky-infer -p sky-gguf
fi
for s in "${run_stages[@]}"; do
  echo "== stage $s"
  case "$s" in
    open) open_ ;;
    read) read_ ;;
    *) "$s" ;;
  esac
done
