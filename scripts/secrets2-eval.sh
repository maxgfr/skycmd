#!/usr/bin/env bash
# The evaluation of s6-secrets v2, s6-secrets-verify-pico (docs/s6-secrets2-prereg.md, sections 6
# to 9): the generator alone, every baseline, the model when there is one, the scorer and the
# gate, for one split. Everything runs at `nice -n 19`, one tool at a time.
#
# Usage: scripts/secrets2-eval.sh <validation|test> [step ...]
#   steps (default: all, in this order; `clean` before `states`, which reads its export)
#     tools      the pinned tools: downloads checked against scripts/secrets2/pins.json, uv
#                environments from their locks, the two containers; <data>/eval/DOWNLOADS.json
#     creddata   on the CredData repositories of the split: the generator alone, the default
#                engine, Gitleaks, Betterleaks, detect-secrets, CredSweeper (ML on, ML off),
#                cbx-secrets-detector-17M
#     issues     the issue-report split as texts, as published and through CredData's
#                obfuscation, and the scanners on both
#     states     the state of every candidate of the split (CredData, the issue reports as
#                published, obfuscated and end to end) and of the clean-text corpus, from the
#                data side's own builder (scripts/secrets2/states.py: sky-data secrets2 states;
#                SKY_DATA names the binary)
#     trained    the size-matched LR and potion-base-2M + LR: trained once (validation run),
#                then scored on the split's candidates
#     model      the model (MODEL=...): sky-eval at full depth, its answers as flags
#     clean      the clean-text corpus: the default engine, the generator alone, the model
#     score      scripts/secrets2/scorer.py and issues.py score
#     gate       scripts/s6-secrets2-gate.py: results.md and gate.json
#
#   MODEL=models/.../x-ternary.gguf   the trained ternary GGUF; its metadata holds tau*
#                                     (skycmd-s2.secrets2.tau_f1), stored before any test file is read
#   SEAT=model                        the system the gate judges (SEAT=lr rehearses it without a model)
#   SECRETS2_DATA, SECRETS2_SECRETGATE  the data folder and the frozen secretgate checkout
#                                       (defaults in scripts/secrets2/common.py)
#   SECRETS2_CPUS=4                   processors given to the two containers
#   BUILD=<data>/build                the data side's build (tables and records)
#
# THE TEST SPLIT IS REFUSED unless MODEL names a file whose metadata holds tau* and
# runs/s6-secrets2/TEST-OPEN exists (the owner of the experiment creates it by hand after
# training). Every run is refused when the secretgate checkout is not the frozen commit, or
# when a frozen hash differs: splits.json, the generator's files, the CredData checkout and
# its file listing, the issue-report split file. Every runner checks the split again.
#
# The candidates the model and the trained baselines judge come as tables of states, all from
# the data side's state builder (`sky-data secrets2 states`), written by the `states` step:
#   <data>/eval/<split>/states/creddata.{jsonl,records.jsonl}   the generator's candidates on the
#                                                               split's repositories
#   <data>/eval/<split>/states/issues.*              the dataset's candidates, values as published
#   <data>/eval/<split>/states/issues-obfuscated.*   the same through CredData's obfuscation
#   <data>/eval/<split>/states/issues-e2e.*          the generator's candidates on the issue texts
#   <data>/eval/clean/states/clean.*                 the generator's candidates on the clean text
# The issue-report states need the `issues` step first (the texts and the generator's export on
# them). A system whose states are missing is skipped and said so; the gate then fails on the
# missing result.
#
# Outputs, all in the data folder (outside every repository): <data>/eval/<split>/. Nothing
# written there or printed holds a value except the raw exports under raw/, which stay there.
# results.md and gate.json (numbers and hashes) are copied to runs/s6-secrets2/eval-<split>/.
set -euo pipefail
cd "$(dirname "$0")/.."

split=${1:-}
case "$split" in
  validation | test) ;;
  *)
    echo "usage: scripts/secrets2-eval.sh <validation|test> [tools creddata issues clean states trained model score gate]" >&2
    exit 2
    ;;
esac
shift
steps=("$@")
[ ${#steps[@]} -eq 0 ] && steps=(tools creddata issues clean states trained model score gate)

S=scripts/secrets2
py() { nice -n 19 python3 -I "$@"; }
export SECRETS2_MODEL=${MODEL:-}
seat=${SEAT:-model}

# The guards, before anything is read: the frozen generator, the frozen files, and for the
# test the model with its tau* and the TEST-OPEN file.
py $S/common.py check "$split" || exit 3

data=$(python3 -I -c "import sys; sys.path.insert(0, '$S'); import common; print(common.data_root())")
E=$data/eval
out=$E/$split
flags=$out/flags
iss=$E/issues/$split
build=${BUILD:-$data/build}
mkdir -p "$flags"

# table <name>: the table of states of a set of candidates of this split, or nothing.
table() {
  if [ -f "$out/states/$1.jsonl" ]; then
    echo "$out/states/$1.jsonl"
  elif [ "$split" = validation ] && [ "$1" != issues-obfuscated ] && [ -f "$build/tables/$1-val.jsonl" ]; then
    echo "$build/tables/$1-val.jsonl"
  fi
}
records() {
  if [ -f "$out/states/$1.records.jsonl" ]; then
    echo "$out/states/$1.records.jsonl"
  elif [ "$split" = validation ] && [ "$1" != issues-obfuscated ] && [ -f "$build/records/$1/val.jsonl" ]; then
    echo "$build/records/$1/val.jsonl"
  fi
}
# corpora: <table name> <corpus> <flags folder> [e2e]. The last one is the end-to-end run of
# section 7.4: the generator's candidates on the issue texts, judged and written as flags at
# their places (system <name>-e2e).
corpora() {
  echo "creddata creddata:$split $flags"
  echo "issues issues:$split $iss/flags"
  echo "issues-obfuscated issues:$split:obfuscated $iss/flags-obfuscated"
  echo "issues-e2e issues:$split $iss/flags e2e"
}

tools() {
  py $S/setup_tools.py fetch envs images record
}

creddata() {
  py $S/run_secretgate.py generator --corpus "creddata:$split" --out "$flags"
  py $S/run_secretgate.py default --corpus "creddata:$split" --out "$flags"
  local t
  for t in gitleaks betterleaks detect-secrets credsweeper-ml credsweeper-noml cbx-17m; do
    py $S/run_scanners.py "$t" --corpus "creddata:$split" --out "$flags"
  done
}

issues() {
  py $S/issues.py prepare --split "$split" >/dev/null
  py $S/issues.py obfuscate --split "$split" >/dev/null
  local c d t
  for c in "issues:$split" "issues:$split:obfuscated"; do
    d=$iss/flags
    [ "$c" = "issues:$split" ] || d=$iss/flags-obfuscated
    mkdir -p "$d"
    py $S/run_secretgate.py generator --corpus "$c" --out "$d"
    py $S/run_secretgate.py default --corpus "$c" --out "$d"
    for t in gitleaks betterleaks detect-secrets credsweeper-ml credsweeper-noml cbx-17m; do
      py $S/run_scanners.py "$t" --corpus "$c" --out "$d"
    done
  done
}

states() {
  py $S/states.py --corpus "creddata:$split" --candidates "$flags/raw/candidates.raw.jsonl" --out "$out/states/creddata" --build "$build"
  py $S/states.py --corpus clean --candidates "$E/clean/flags/raw/candidates.raw.jsonl" --out "$E/clean/states/clean" --build "$build"
  # the issue reports: the dataset's candidates as published and through CredData's
  # obfuscation, and the generator's candidates on the texts (end to end)
  if [ ! -f "$iss/candidates.jsonl" ]; then
    echo "== states of the issue reports: no texts in $iss (the issues step writes them): skipped"
    return
  fi
  py $S/states.py --corpus "issues:$split" --out "$out/states/issues" --build "$build"
  if [ -d "$iss/texts-obfuscated" ]; then
    py $S/states.py --corpus "issues:$split:obfuscated" --out "$out/states/issues-obfuscated" --build "$build"
  else
    echo "== states of the obfuscated issue texts: no $iss/texts-obfuscated: skipped"
  fi
  if [ -f "$iss/flags/raw/candidates.raw.jsonl" ]; then
    py $S/states.py --corpus "issues:$split:e2e" --candidates "$iss/flags/raw/candidates.raw.jsonl" --out "$out/states/issues-e2e" --build "$build"
  else
    echo "== states of the generator's candidates on the issue texts: no export in $iss/flags/raw: skipped"
  fi
}

trained() {
  local kind name t corpus dir mode
  for kind in lr potion; do
    name=$kind
    [ "$kind" = potion ] && name=potion-lr
    if [ "$split" = validation ]; then
      # the regularisation strength and the threshold are chosen here, on validation (section 6)
      if [ ! -f "$build/tables/creddata-train.jsonl" ]; then
        echo "== $name: no tables in $build/tables (the data side's build): skipped"
        continue
      fi
      nice -n 19 "$E/envs/$kind/bin/python" -I $S/run_lr.py train --kind "$kind" --tables "$build/tables" --out "$E/$kind" >/dev/null
      # the value-only probe (section 9): trained on training, scored on validation
      [ "$kind" != lr ] || nice -n 19 "$E/envs/lr/bin/python" -I $S/run_lr.py probe --tables "$build/tables" --out "$E/value-probe.json" >/dev/null
    elif [ ! -f "$E/$kind/$kind.json" ]; then
      echo "== $name: not trained (run the validation split first): skipped"
      continue
    fi
    while read -r t corpus dir mode; do
      if [ -z "$(table "$t")" ]; then
        echo "== $name on $corpus${mode:+ ($mode)}: no table of states ($out/states/$t.jsonl): skipped"
        continue
      fi
      nice -n 19 "$E/envs/$kind/bin/python" -I $S/run_lr.py score --kind "$kind" --model "$E/$kind" --table "$(table "$t")" --corpus "$corpus" --out "$dir" ${mode:+--e2e}
    done < <(corpora)
  done
}

model() {
  if [ -z "${MODEL:-}" ]; then
    echo "== model: no MODEL given: skipped"
    return
  fi
  local t corpus dir mode
  while read -r t corpus dir mode; do
    if [ -z "$(table "$t")" ] || [ -z "$(records "$t")" ]; then
      echo "== model on $corpus${mode:+ ($mode)}: no table or records of states ($out/states/$t.jsonl, $t.records.jsonl): skipped"
      continue
    fi
    py $S/run_model.py eval --model "$MODEL" --records "$(records "$t")" --split val --out "$out/model/$t" >/dev/null
    py $S/run_model.py convert --model "$MODEL" --dump "$out/model/$t/model-dump.jsonl" --table "$(table "$t")" --corpus "$corpus" --out "$dir" ${mode:+--e2e}
    # the WASM figure is reported, not gated: a parity failure stays in model-wasm.json
    [ "$t" != creddata ] || py $S/run_model.py wasm --model "$MODEL" --dump "$out/model/$t/model-dump.jsonl" --out "$out/model/$t" >/dev/null || true
  done < <(corpora)
  if [ -f "$E/clean/states/clean.jsonl" ] && [ -f "$E/clean/states/clean.records.jsonl" ]; then
    py $S/run_model.py eval --model "$MODEL" --records "$E/clean/states/clean.records.jsonl" --split val --out "$E/clean/model" >/dev/null
    py $S/run_model.py convert --model "$MODEL" --dump "$E/clean/model/model-dump.jsonl" --table "$E/clean/states/clean.jsonl" --corpus clean --out "$E/clean/flags"
  else
    echo "== model on the clean-text corpus: no states ($E/clean/states/clean.jsonl): skipped"
  fi
}

clean() {
  py $S/clean_text.py prepare >/dev/null
  # the default engine and the generator alone do not depend on the split: run once
  [ -f "$E/clean/flags/generator.run.json" ] && [ -f "$E/clean/flags-with-path/generator.run.json" ] || py $S/clean_text.py run >/dev/null
  py $S/clean_text.py share
}

score() {
  local ref=()
  [ -f "$flags/$seat.run.json" ] && ref=(--reference "$seat")
  py $S/scorer.py --split "$split" --flags "$flags" --out "$out/scores.json" ${ref[@]+"${ref[@]}"}
  py $S/issues.py score --split "$split" --flags "$iss/flags" --flags-obfuscated "$iss/flags-obfuscated" --out "$out/issues.json"
}

gate() {
  local dir=$out
  [ "$seat" = model ] || dir=$out/seat-$seat
  # A failed gate is a result, not a crash: it stays in results.md.
  py scripts/s6-secrets2-gate.py --scores "$out/scores.json" --issues "$out/issues.json" --clean "$E/clean/clean.json" \
    --latency "$out/model/creddata/model-latency.json" --wasm "$out/model/creddata/model-wasm.json" --probe "$E/value-probe.json" \
    --model "$seat" --out-dir "$dir" >/dev/null || {
    [ -s "$dir/results.md" ] || exit 1
    echo "== gate not passed or not the test: see $dir/results.md"
  }
  # numbers and hashes only: a copy in the checkout, next to runs/s6-secrets2/generator.json
  local kept=runs/s6-secrets2/eval-$split
  [ "$seat" = model ] || kept=$kept/seat-$seat
  mkdir -p "$kept"
  cp "$dir/results.md" "$dir/gate.json" "$kept/"
  echo "wrote $dir/results.md and $dir/gate.json (copied to $kept)"
}

for s in "${steps[@]}"; do
  case "$s" in
    tools | creddata | issues | clean | states | trained | model | score | gate)
      echo "== secrets2-eval $split: $s"
      "$s"
      ;;
    *)
      echo "unknown step $s (tools, creddata, issues, clean, states, trained, model, score, gate)" >&2
      exit 2
      ;;
  esac
done
