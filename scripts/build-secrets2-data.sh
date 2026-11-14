#!/usr/bin/env bash
# Build the data of s6-secrets v2 (docs/s6-secrets2-prereg.md, sections 3 to 5): everything
# between the frozen candidate generator's output and the training files. No training, no
# evaluation, no GPU.
#
# The data folder is outside every repository and nothing in it is ever committed:
#   <data>/gen/generator.json          the frozen generator (the build refuses to run without it)
#   <data>/gen/candidates-*.raw.jsonl  its candidates, <data>/gen/labels-*.jsonl their labels
#   <data>/creddata, <data>/issues, <data>/splits.json
#
# Steps, in order, each a mode of sky-data (counts only on the terminal, never a value):
#   creddata       labelled candidates of the training and validation repositories, the
#                  decontamination against the test repositories (their lines, never their
#                  meta), the weak negatives (one per distinct state, the repositories in turn)
#   issues         the issue-report train and validation splits, decontaminated against the test
#                  split (its texts' hashes only), every positive value replaced (the context
#                  words of the training negatives kept)
#   secret-cand    synthetic candidates
#   mix            states, decision records, tables for the baselines
#   hand-read      the 200 weak negatives that are read by hand (they stay in the data folder)
#   truncation     sky-train s2-truncation on the states (section 4.3: the window rule)
#   pretrain-text  the CredData share of the pretraining text
#   manifest       BUILD.json: every count, the hashes of the inputs and of the outputs
#
# Outputs under <out> (default <data>/build):
#   records/{creddata,issues,synth}/{train,val}.jsonl   for s2-decide:
#       --records <out>/records/creddata,<out>/records/issues,<out>/records/synth
#       --weights creddata=0.45,issues=0.35,synth=0.20
#   tables/<source>-<split>.jsonl    one line per candidate: label, features, state
#   candidates/<source>-<split>.jsonl the candidate records
#   pretrain-text/creddata.jsonl     for s2-pretrain --text data/text-open,data/text,<out>/pretrain-text
#   dictionary.txt, truncation.json, BUILD.json
#
# The same inputs and the same seed give the same bytes.
#
# Usage: scripts/build-secrets2-data.sh <data> [out]
#   SEED=1234
#   TOKENIZER=data/tok/tok-s2-1024.json   the model's tokenizer
#   TEXT=data/text-open,data/text         the text the pretraining already reads
#   DICT=/usr/share/dict/words            the word list of the dictionary
#   CONFIG=configs/s6-secrets-pico.json   the 320-token window the truncation is measured at
#   SKY_TRAIN=<built sky-train>           for s2-truncation only; it is not built here
#   HAND_READ=<out>/handread/result.json  the result of the reading by hand, once it exists
set -euo pipefail
export DEVELOPER_DIR=${DEVELOPER_DIR:-/Applications/Xcode.app/Contents/Developer}
export CARGO_TARGET_DIR=${CARGO_TARGET_DIR:-$PWD/target/s3-train}

data=${1:?usage: scripts/build-secrets2-data.sh <data> [out]}
out=${2:-$data/build}
seed=${SEED:-1234}
tok=${TOKENIZER:-data/tok/tok-s2-1024.json}
text=${TEXT:-data/text-open,data/text}
dict=${DICT:-/usr/share/dict/words}
cfg=${CONFIG:-configs/s6-secrets-pico.json}
train=${SKY_TRAIN:-$CARGO_TARGET_DIR/release/sky-train}
hand=${HAND_READ:-$out/handread/result.json}

if [ ! -f "$data/gen/generator.json" ]; then
  echo "$data/gen/generator.json does not exist: the generator is not frozen, nothing is built" >&2
  exit 1
fi
if [ ! -x "$train" ]; then
  echo "no sky-train at $train: set SKY_TRAIN to a built one (s2-truncation reads records only)" >&2
  exit 1
fi
for f in "$tok" "$dict" "$cfg"; do
  [ -f "$f" ] || { echo "$f does not exist" >&2; exit 1; }
done

nice -n 19 cargo build -q --release -p sky-data
D=$CARGO_TARGET_DIR/release/sky-data
mkdir -p "$out/logs" "$out/truncation"
# step <name> <command...>: runs it at low priority; its counts go to <out>/logs/<name>.json.
step() {
  local name=$1
  shift
  echo "== $name"
  nice -n 19 "$@" >"$out/logs/$name.json"
}

step creddata "$D" --seed "$seed" secrets2 creddata --data "$data" --out "$out" \
  --tokenizer "$tok" --dict "$dict"
step issues "$D" --seed "$seed" secrets2 issues --data "$data" --out "$out" --dict "$dict"
step secret-cand "$D" --seed "$seed" secret-cand --out "$out" --dict "$dict"
step mix "$D" --seed "$seed" secrets2 mix --out "$out" --tokenizer "$tok" --dict "$dict"
step hand-read "$D" --seed "$seed" secrets2 hand-read --out "$out"

# Section 4.3: the share of states the 320-token window cuts, over the training and validation
# candidates of the three sources; then file by file, to say where they are.
records=$out/records/creddata,$out/records/issues,$out/records/synth
step truncation "$train" s2-truncation --config "$cfg" --tokenizer "$tok" --records "$records" \
  --per-source 1000000000 --json "$out/truncation.json"
for source in creddata issues synth; do
  for split in train val; do
    step "truncation-$source-$split" "$train" s2-truncation --config "$cfg" --tokenizer "$tok" \
      --records "$out/records/$source/$split.jsonl" --per-source 1000000000 \
      --json "$out/truncation/$source-$split.json"
  done
done

step pretrain-text "$D" --seed "$seed" secrets2 pretrain-text --data "$data" --out "$out" \
  --tokenizer "$tok" --text "$text"

manifest=("$D" --seed "$seed" secrets2 manifest --data "$data" --out "$out" --tokenizer "$tok"
  --dict "$dict" --truncation "$out/truncation.json" --config "$cfg")
[ -f "$hand" ] && manifest+=(--hand-read "$hand")
step manifest "${manifest[@]}"
echo "== done: $out/BUILD.json"
