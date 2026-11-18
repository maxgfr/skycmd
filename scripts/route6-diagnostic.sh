#!/usr/bin/env bash
# The diagnostic of the published files (docs/s6-route6-prereg.md 9.1; reported, files already
# read): the published model part (s6-route-nano.gguf, v2's line rendering) and the published pair
# (with s6-route-embed.gguf, beta 0.81, tau_ask 0.671047) read blind on v_tools, v_tools_masked,
# T2-tools and its masked copy. It runs after v6's freeze and before its training; nothing of v6
# depends on it. CPU only, at nice -n 19; no file is trained on, no fresh file of v4, v5 or v6 is
# read.
#
# Usage: scripts/route6-diagnostic.sh [file ...]   (default: the four files below; any other name
#        is refused)
#   v_tools v_tools_masked      <data>/s6-route/build/val/
#   f_tools f_tools_masked      <data>/s6-route/build/fresh/ (T2-tools, opened by
#                               runs/s6-nano3/FRESH-OPEN; its cluster bootstrap by
#                               f_tools.clusters.jsonl, as v3)
#
# Steps for each file:
#   1. the placeholder (4.3): runs/s6-route6/placeholder.json if it exists, else taken now by
#      sky-data route blind --check-placeholder on every training and validation record of v5's
#      build (the same requests as v6's) and on v5_tools and its masked copy;
#   2. the blind copies (sky-data route blind --render bm25-line: B0 is the state `User: <P>` with
#      a BM25 line of zeros, as v2's file reads; B1 the bare options) into
#      <data>/s6-route/eval/route6-diagnostic/;
#   3. the model part by sky-eval on B0 and B1, at full depth; the pair and its embedding part by
#      sky-route fuse on the blind plain file (the scores computed from P: all 0);
#   4. scripts/s6-route6-blind.py read for each system and reading;
#   then runs/s6-route6/diagnostic.json and .md.
#
#   ROUTE_DATA=../skycmd-data/s6-route   MODEL=site/public/models/s6-route-nano.gguf
#   EMBED=site/public/models/s6-route-embed.gguf   MIN_FREE_GB=3   DRY_RUN=1
set -euo pipefail
cd "$(dirname "$0")/.."
export DEVELOPER_DIR=${DEVELOPER_DIR:-/Applications/Xcode.app/Contents/Developer}
export CARGO_TARGET_DIR=${CARGO_TARGET_DIR:-$PWD/target/s3-train}
dry=${DRY_RUN:-0}
data=${ROUTE_DATA:-../skycmd-data/s6-route}
model=${MODEL:-site/public/models/s6-route-nano.gguf}
embed=${EMBED:-site/public/models/s6-route-embed.gguf}
min_free=${MIN_FREE_GB:-3}
runs=runs/s6-route6
out=$data/eval/route6-diagnostic
B=$CARGO_TARGET_DIR/release
D=$B/sky-data
FIXED=(v_tools v_tools_masked f_tools f_tools_masked)

files=("$@")
[ ${#files[@]} -eq 0 ] && files=("${FIXED[@]}")
for f in "${files[@]}"; do
  case " ${FIXED[*]} " in
    *" $f "*) ;;
    *) echo "refusing $f: the diagnostic of 9.1 reads ${FIXED[*]} and nothing else" >&2; exit 2 ;;
  esac
done

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
path_of() {
  case "$1" in
    v_tools | v_tools_masked) echo "$data/build/val/$1.jsonl" ;;
    f_tools | f_tools_masked) echo "$data/build/fresh/$1.jsonl" ;;
  esac
}
# split_of <name>: the split its records carry, which sky-eval reads (--split; its default is
# test): `val` for v_tools, `fresh` for T2-tools, as v3's reading passed it (scripts/route-eval3.sh)
split_of() {
  case "$1" in
    v_tools | v_tools_masked) echo val ;;
    f_tools | f_tools_masked) echo fresh ;;
  esac
}

if [ "$dry" != 1 ]; then
  kb=$(df -Pk "$data" | awk 'NR == 2 { print $4 }')
  [ "$kb" -ge $((min_free * 1024 * 1024)) ] || { echo "below the disk floor of $min_free GB" >&2; exit 4; }
  nice -n 19 cargo build -q --release -p sky-data -p sky-infer
  mkdir -p "$out" "$runs"
fi

if [ ! -f "$runs/placeholder.json" ]; then
  recs=""
  for r in "$data"/build5/records/*/*.jsonl "$data/build5/val/v5_tools.jsonl" "$data/build5/val/v5_tools_masked.jsonl"; do
    recs="$recs$r,"
  done
  note "the placeholder of 4.3, checked against every training and validation request of v5's build"
  x "$D" route blind --check-placeholder "${recs%,}" --out "$runs/placeholder.json"
fi

for f in "${files[@]}"; do
  src=$(path_of "$f")
  split=$(split_of "$f")
  open=()
  case "$f" in f_*) open=(--open runs/s6-nano3/FRESH-OPEN) ;; esac
  x "$D" route blind --in "$src" --out "$out/$f" --placeholder "$runs/placeholder.json" --render bm25-line \
    ${open[@]+"${open[@]}"}
  x "$B/sky-eval" "$model" --records "$out/$f.blind.model.jsonl" --split "$split" --max 100000 --exit-threshold 1.0 \
    --dump-probs "$out/$f.blind.dump.jsonl" --json "$out/$f.blind.eval.json"
  x "$B/sky-eval" "$model" --records "$out/$f.blind-bare.model.jsonl" --split "$split" --max 100000 --exit-threshold 1.0 \
    --dump-probs "$out/$f.blind-bare.dump.jsonl" --json "$out/$f.blind-bare.eval.json"
  x "$B/sky-route" fuse --model "$model" --embed "$embed" --slates "$out/$f.blind.jsonl" --out "$out/pair_$f.blind.jsonl"
done

note "the statistics of 9.1: $runs/diagnostic.json and .md"
if [ "$dry" = 1 ]; then
  x python3 -I scripts/s6-route6-blind.py read --slates "$out/<file>.blind.model.jsonl" --dump "$out/<file>.blind.dump.jsonl"
  exit 0
fi
python3 -I - "$out" "$runs" "$data/build/fresh/f_tools.clusters.jsonl" "${files[@]}" <<'EOF'
import importlib.util, json, os, sys
out, runs, clusters_path = sys.argv[1:4]
files = sys.argv[4:]
spec = importlib.util.spec_from_file_location("b6", os.path.join("scripts", "s6-route6-blind.py"))
b6 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b6)
clusters = {x["uid"]: x["cluster"] for x in b6.load(clusters_path)}
res = {}
for f in files:
    servers = clusters if f.startswith("f_") else None
    unit = "cluster"
    r = {}
    for name, slates, dump, field in (
            ("model part, B0", f + ".blind.model.jsonl", f + ".blind.dump.jsonl", "probs"),
            ("model part, B1", f + ".blind-bare.model.jsonl", f + ".blind-bare.dump.jsonl", "probs"),
            ("pair, B0", f + ".blind.jsonl", "pair_" + f + ".blind.jsonl", "probs"),
            ("embedding part (potion-cos-4bit), B0", f + ".blind.jsonl", "pair_" + f + ".blind.jsonl", "embedding")):
        s, _ = b6.read(b6.load(os.path.join(out, slates)), b6.load(os.path.join(out, dump)), field, servers, unit)
        r[name] = s
    res[f] = r
json.dump({"about": "s6-route v6 (docs/s6-route6-prereg.md 9.1): the published model part and pair read blind on files already read; reported, nothing of v6 depends on it",
           "placeholder": json.load(open(os.path.join(runs, "placeholder.json")))["placeholder"], "files": res},
          open(os.path.join(runs, "diagnostic.json"), "w"), indent=1)
lines = ["# s6-route v6: the published files read blind (docs/s6-route6-prereg.md 9.1)", "",
         "| file | system | tool-right | blind accuracy | c | 95% (slates) | none share | blind none gap | fewest-words share |",
         "|---|---|---:|---:|---:|---|---:|---:|---:|"]
for f, r in res.items():
    for name, s in r.items():
        iv = s["intervals, slates resampled"]["blind accuracy"]
        lines.append("| %s | %s | %d | %.4f | %.4f | [%.4f, %.4f] | %.4f | %s | %.4f |" % (
            f, name, s["tool-right slates"], s["blind accuracy"], s["chance c"], iv[0], iv[1], s["none share"],
            "%+.4f" % s["blind none gap"] if "blind none gap" in s else "", s["fewest-words share of the blind picks"]))
open(os.path.join(runs, "diagnostic.md"), "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
EOF
