#!/usr/bin/env bash
# s6-ear v2 (s6-ear2-nano): the end-of-turn detector of docs/s6-ear2-prereg.md, a ternary student
# of 1.3-2.7M parameters on 4 s of log-mel and prosody, distilled from Smart Turn v3.2. Modelled on
# scripts/train-ear.sh (v1) and scripts/train-cast2.sh.
#
# Usage: scripts/train-turn.sh [probe] [data] [eotbench-data] [pilot] [train] [export] [calibrate]
#                              [harness] [latency] [wasm] [eval] [eotbench] [gate]
#   (default: probe pilot train export calibrate harness latency wasm eval eotbench gate, in this
#   order; data and eotbench-data are long CPU jobs run first and on their own; no v2 test score
#   is read before the harness check of §8.1 and the latencies of §8.3 are on file, §13)
#   probe     §4.4: one random-weight ternary GGUF per rung (sky-train turn-probe), timed natively
#             (sky-turn --bench 1000) and in WASM (crates/sky-web/bench/turn-bench.mjs, Node,
#             SIMD128, one thread, 1,000 decisions on a full 4 s window, features computed); the
#             first rung of R1-R4 whose WASM median is <= 4.0 ms -> runs/s6-ear2-probe/rung.json.
#             If none qualifies, the script stops: an amendment says so and v2 is not run
#   data      §3, §5: sky-data turn-test (the hops and PCM hashes of v1's test WAVs), then shard by
#             shard: sky-data fetch-turn (download, the leakage filter of §3.2, the hops),
#             scripts/turn-teacher.py (Smart Turn's targets, CPU), the shard removed. Resumable;
#             exit status 3 (and a stop) when the free disk would go below MIN_FREE_GB
#   eotbench-data  §8.4: file by file, sky-data turn-eotbench (the cuts), Smart Turn on the cuts
#             (scripts/ear-baseline.py --eotbench, no label read), the file removed
#   pilot     §6.4-6.5: the rate at 0.5x, 1x and 2x the recipe's (moved to the rung's d by muP),
#             12 working minutes each with full distillation, then the hard labels at the best
#             rate; choice by validation accuracy at tau 0.5 on f32 weights (distil unless the
#             hard arm is higher by more than 0.005); the stop rule (best arm below 0.82: v2 stops,
#             failed by rule) -> runs/s6-ear2-pilot/choice.json
#   train     §6.6: 240 working minutes, batch 128, seed 1, the recipe's optimizer at the chosen
#             rate, WSD, ternary QAT over the last 15%, EMA 0.999, the brake
#   export    §7: the EMA if its ternary validation BCE is below last's, else last; ternary GGUF
#   calibrate §7: the temperature (validation NLL) and tau85 (the highest validation threshold
#             with END recall >= 0.85) written into models/s6-ear2/s6-ear2-nano.gguf
#   harness   §8.1: scripts/ear-baseline.py (left padding) on the 31,527 test clips: 0.9263 +-
#             0.003, and the timer at the test's share of finished clips (0.4979)
#   latency   §8.3: v2 natively and in WASM (decision median and p95, front end per hop), Smart
#             Turn CPU through onnxruntime natively, the GGUF size
#   wasm      §7: WASM parity on 20 test clips (|dp| <= 1e-4 and the same decision)
#   eval      §8.2: v2's probabilities on the test clips (the stored features of turn-test)
#   eotbench  §8.4: v2's probabilities on the eot-bench cuts
#   gate      §9: the verdict (G1-G3) with §8.2-8.4 and §10, and §11's disclosure
#             (scripts/turn-gate.py), into runs/s6-ear2-nano/GATE.md and gate.json
#
# Environment:
#   TURN_DATA=/Users/maxime/Documents/Code/skycmd-data/s6-ear2   the data folder (outside the
#                       repository): test/, train/, shards/ (transient), eotbench/, hf/
#   EAR=data/ear        v1's data (clips/smart-turn.jsonl and the test WAVs it names)
#   MIN_FREE_GB=8       free disk the data steps always leave (§3.4)
#   THREADS=4  PROCS=4  feature threads, teacher processes (CPU)
#   MINUTES=240  PILOT_MINUTES=12  TIME_SCALE=1   working minutes (TIME_SCALE multiplies both)
#   RUNG                the rung, instead of runs/s6-ear2-probe/rung.json (an amendment's)
#   The recipe is read from runs/recipe/recipe.json (scripts/s6-adopt.py env s6-ear-nano): the
#   optimizer flags (OPT) and the rate scale (LR_SCALE). The default rate is the decision
#   phase's 1e-3 x LR_SCALE x sqrt(96 / d), as scripts/train-ear.sh moved it to d = 144.
#   SKY_DUTY=0.5  SKY_COOL=45:15  SKY_BRAKE_FILE=runs/brake   the brake (v1's amendments 2-4;
#                       scripts/auto-brake.sh writes the file and stops a training over its cap)
#   SKY_MLX_CACHE_GB=2  cap of MLX's buffer cache
#   SKY_BIN             a folder with the release binaries (default $CARGO_TARGET_DIR/release,
#                       built here); with SKY_BIN set nothing is built
#   RESUME=1            continue the training from its last checkpoint (only after a technical
#                       failure, §6.7); without it an existing run is refused
#   SKY_OUT             a prefix for everything this script writes under runs/ and models/
#   SMOKE=1             §6.3: the chain under a throwaway prefix (default runs/turn-smoke/), never
#                       runs/s6-ear2* or models/s6-ear2: 5 working minutes of training on 2,000
#                       clips (pilot arms of 1 minute), 20 decisions per latency, the validation
#                       clips in place of the test clips (no test score at all); its numbers are not
#                       looked at
#   DRY_RUN=1           print the commands instead of running them (nothing is built, trained or
#                       read)
set -euo pipefail
export DEVELOPER_DIR=${DEVELOPER_DIR:-/Applications/Xcode.app/Contents/Developer}
export CARGO_TARGET_DIR=${CARGO_TARGET_DIR:-$PWD/target/s3-train}
export SKY_DUTY=${SKY_DUTY:-0.5}
export SKY_COOL=${SKY_COOL-45:15}
export SKY_BRAKE_FILE=${SKY_BRAKE_FILE:-$PWD/runs/brake}
export SKY_MLX_CACHE_GB=${SKY_MLX_CACHE_GB:-2}
dry=${DRY_RUN:-0}
smoke=${SMOKE:-}
if [ -n "$smoke" ]; then
  SKY_OUT=${SKY_OUT:-runs/turn-smoke/}
  MINUTES=${MINUTES:-5}
  PILOT_MINUTES=${PILOT_MINUTES:-1}
fi
sky_out=${SKY_OUT:-}
B=${SKY_BIN:-$CARGO_TARGET_DIR/release}
T=$B/sky-train
D=$B/sky-data
S=$B/sky-turn
name=s6-ear2-nano
cfg=configs/$name.json
data=${TURN_DATA:-/Users/maxime/Documents/Code/skycmd-data/s6-ear2}
ear=${EAR:-data/ear}
run=${sky_out}runs/$name
pilot_dir=${sky_out}runs/s6-ear2-pilot
probe_dir=${sky_out}runs/s6-ear2-probe
m=${sky_out}models/s6-ear2
min_free=${MIN_FREE_GB:-8}
threads=${THREADS:-4}
procs=${PROCS:-4}
tmin() { LC_ALL=C awk -v a="$1" -v b="${TIME_SCALE:-1}" 'BEGIN { printf "%g", a * b }'; }
minutes=$(tmin "${MINUTES:-240}")
pilot_minutes=$(tmin "${PILOT_MINUTES:-12}")
iters=1000
max_train=0 max_val=0
eval_split="test"
if [ -n "$smoke" ]; then
  iters=20 max_train=2000 max_val=200 eval_split=val
  case "$run" in
    "runs/$name" | "./runs/$name" | "$PWD/runs/$name")
      echo "SMOKE=1 needs a prefix that is not the real run's (SKY_OUT=$sky_out)" >&2
      exit 2
      ;;
  esac
  echo "== SMOKE: everything under $sky_out; nothing of it is a result"
fi

# the recipe (OPT, LR_SCALE), as scripts/train-ear.sh reads it; the training steps need it
opt=()
LR_SCALE=
recipe() {
  local OPT=
  if [ -f runs/recipe/recipe.json ]; then
    eval "$(python3 -I scripts/s6-adopt.py env s6-ear-nano)"
  elif [ "$dry" = 1 ]; then
    echo "# no runs/recipe/recipe.json: the commands below show the default options"
    OPT=--no-muon LR_SCALE=2.0
  else
    echo "no runs/recipe/recipe.json: the recipe is adopted first (scripts/train-all.sh adopt)" >&2
    exit 1
  fi
  read -r -a opt <<<"$OPT"
}

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
# logged <file> <command...>: x, with what the command prints kept in <file>.
logged() {
  local log=$1
  shift
  if [ "$dry" = 1 ]; then
    x "$@"
  else
    mkdir -p "$(dirname "$log")"
    "$@" 2>&1 | tee -a "$log"
  fi
}
# note <text>: what a step done in this script would do, under DRY_RUN=1.
note() { [ "$dry" != 1 ] || echo "# $*"; }
gate_py() { x python3 -I scripts/turn-gate.py "$@"; }

# The rung (§4.4): RUNG, else the probe's choice.
rung() {
  if [ -n "${RUNG:-}" ]; then
    echo "$RUNG"
  elif [ -f "$probe_dir/rung.json" ]; then
    python3 -I -c 'import json, sys; print(json.load(open(sys.argv[1]))["rung"])' "$probe_dir/rung.json"
  elif [ "$dry" = 1 ] || [ -n "$smoke" ]; then
    echo R3
  else
    echo "no $probe_dir/rung.json: run the probe step first (§4.4)" >&2
    exit 1
  fi
}

# The recipe's default rate at the rung's d (muP, as v1).
base_rate() {
  local d
  d=$(python3 -I -c 'import json, sys; print(json.load(open(sys.argv[1]))["rungs"][sys.argv[2]]["hidden_size"])' "$cfg" "$1")
  LC_ALL=C awk -v s="${LR_SCALE:-1}" -v d="$d" 'BEGIN { printf "%.4g\n", 1e-3 * s * sqrt(96 / d) }'
}
scaled() { LC_ALL=C awk -v a="$1" -v b="$2" 'BEGIN { printf "%.4g\n", a * b }'; }

# The nodejs build of sky-web (TurnDetector, turnSynthSpeech) for the WASM measurements.
wasm_pkg() {
  local pkg=$1
  x env RUSTFLAGS="-C target-feature=+simd128" CARGO_TARGET_DIR="$CARGO_TARGET_DIR/sky-web-node" \
    nice -n 19 wasm-pack build crates/sky-web --target nodejs --release --out-dir "$PWD/$pkg" --no-typescript
}

probe() {
  if [ -f "$probe_dir/rung.json" ]; then
    echo "== probe: done, $probe_dir/rung.json kept"
    return
  fi
  local pkg=$probe_dir/pkg r
  x "$T" turn-probe --config "$cfg" --out "$probe_dir/files"
  x mkdir -p "$probe_dir"
  [ "$dry" = 1 ] || : >"$probe_dir/native.jsonl"
  for r in R1 R2 R3 R4; do
    if [ "$dry" = 1 ]; then
      x "$S" "$probe_dir/files/$r.gguf" --bench "$iters"
    else
      "$S" "$probe_dir/files/$r.gguf" --bench "$iters" | tee -a "$probe_dir/native.jsonl"
    fi
  done
  wasm_pkg "$pkg"
  if [ "$dry" = 1 ]; then
    x node crates/sky-web/bench/turn-bench.mjs --pkg "$pkg" --iters "$iters" "$probe_dir"/files/R{1,2,3,4}.gguf
  else
    node crates/sky-web/bench/turn-bench.mjs --pkg "$pkg" --iters "$iters" "$probe_dir"/files/R{1,2,3,4}.gguf | tee "$probe_dir/wasm.jsonl"
  fi
  if ! gate_py probe --native "$probe_dir/native.jsonl" --wasm "$probe_dir/wasm.jsonl" --out "$probe_dir/rung.json"; then
    echo "== probe: no rung has a WASM median <= 4.0 ms: an amendment says so and v2 is not run (§4.4)" >&2
    exit 1
  fi
}

# stop_on_disk <status>: the disk rule stopped a step (exit status 3)
stop_on_disk() {
  if [ "$1" = 3 ]; then
    echo "== stopped by the disk rule (MIN_FREE_GB=$min_free): free some space and run the step again (it resumes)" >&2
    exit 3
  fi
  exit "$1"
}

data() {
  x mkdir -p "$data"
  x nice -n 19 "$D" turn-test --clips "$ear/clips/smart-turn.jsonl" --out "$data" --threads "$threads" \
    --min-free-gb "$min_free" || stop_on_disk $?
  local i shards=83
  for ((i = 0; i < shards; i++)); do
    x nice -n 19 "$D" fetch-turn --out "$data" --shards "$i" --threads "$threads" --min-free-gb "$min_free" \
      || stop_on_disk $?
    x nice -n 19 uv run --python 3.12 scripts/turn-teacher.py --data "$data" --shard "$i" --procs "$procs" \
      --min-free-gb "$min_free" || stop_on_disk $?
    x "$D" fetch-turn --out "$data" --shards "$i" --drop
  done
  note "check that every shard has its features and its targets ($data/train/LEAK.json)"
  [ "$dry" = 1 ] || python3 -I scripts/turn-gate.py data --data "$data" --shards "$shards"
}

eotbench_data() {
  local k files=15
  for ((k = 0; k < files; k++)); do
    x nice -n 19 "$D" turn-eotbench --out "$data" --files "$k" --threads "$threads" --min-free-gb "$min_free" \
      || stop_on_disk $?
  done
  # Smart Turn on every cut (8 s, left-padded), then the files removed; no label is read
  x nice -n 19 uv run --python 3.12 scripts/ear-baseline.py --eotbench "$data/eotbench" --tag all
  x "$D" turn-eotbench --out "$data" --drop
}

# A run of the pre-registration is made once: an existing one is continued only with RESUME=1.
fresh_or_resume() {
  local dir=$1
  if [ -f "$dir/last/state.json" ] && [ -z "${RESUME:-}" ] && [ -z "$smoke" ]; then
    echo "$dir already holds a run: RESUME=1 continues it after a technical failure (§6.7)" >&2
    exit 1
  fi
  [ -n "$smoke" ] && [ -z "${RESUME:-}" ] && x rm -rf "$dir"
  return 0
}

train_args() {
  echo --config "$cfg" --rung "$(rung)" --data "$data/train" --batch 128 --seed 1 \
    --max-train "$max_train" --max-val "$max_val"
}

pilot() {
  if [ -f "$pilot_dir/choice.json" ]; then
    echo "== pilot: done, $pilot_dir/choice.json kept"
    return
  fi
  recipe
  local r base mult lr arm
  r=$(rung)
  base=$(base_rate "$r")
  read -r -a common <<<"$(train_args)"
  for mult in 0.5 1 2; do
    lr=$(scaled "$base" "$mult")
    arm=$pilot_dir/lr$mult
    fresh_or_resume "$arm"
    logged "$arm.log" "$T" turn-train "${common[@]}" --out "$arm" --minutes "$pilot_minutes" --resume \
      --qat-frac 0 --eval-every 1000000000 --adam-lr "$lr" ${opt[@]+"${opt[@]}"}
  done
  # the hard-label arm, at the best of the three rates
  if [ "$dry" = 1 ]; then
    lr="<the best rate of the three>"
  else
    lr=$(python3 -I scripts/turn-gate.py pilot --dir "$pilot_dir" --rates-only --base "$base")
  fi
  arm=$pilot_dir/hard
  fresh_or_resume "$arm"
  logged "$arm.log" "$T" turn-train "${common[@]}" --out "$arm" --minutes "$pilot_minutes" --resume \
    --qat-frac 0 --eval-every 1000000000 --adam-lr "$lr" --hard-only ${opt[@]+"${opt[@]}"}
  note "apply §6.4-6.5 (distil unless hard is higher by more than 0.005; stop below 0.82) -> $pilot_dir/choice.json"
  [ "$dry" = 1 ] && return
  local st=0
  python3 -I scripts/turn-gate.py pilot --dir "$pilot_dir" --base "$base" --out "$pilot_dir/choice.json" || st=$?
  if [ "$st" = 4 ] && [ -z "$smoke" ]; then
    echo "== the stop rule fired (§6.5): no main run; v2 is recorded as failed by rule ($pilot_dir/choice.json)" >&2
    exit 4
  fi
  [ "$st" = 0 ] || [ "$st" = 4 ] || exit "$st"
}

choice() { python3 -I -c 'import json, sys; c = json.load(open(sys.argv[1])); print(c[sys.argv[2]])' "$pilot_dir/choice.json" "$1"; }

train() {
  recipe
  local lr hard=()
  if [ "$dry" = 1 ] && [ ! -f "$pilot_dir/choice.json" ]; then
    lr="<the pilot's rate>"
    hard=("<--hard-only if the pilot chose the hard labels>")
  else
    if [ "$(choice stop)" = True ] && [ -z "$smoke" ]; then
      echo "the pilot's stop rule fired (§6.5): no main run" >&2
      exit 4
    fi
    lr=$(choice lr)
    [ "$(choice distil)" = True ] || hard=(--hard-only)
  fi
  read -r -a common <<<"$(train_args)"
  fresh_or_resume "$run"
  if [ "$dry" != 1 ] && [ ! -f "$run/started.json" ]; then
    mkdir -p "$run"
    python3 -I -c 'import json, sys, time; json.dump({"started": time.strftime("%Y-%m-%dT%H:%M:%S%z")}, open(sys.argv[1], "w"))' "$run/started.json"
  fi
  echo "== $name: ${minutes} working minutes (brake $SKY_DUTY ${SKY_COOL:-no long rests}, file $SKY_BRAKE_FILE)"
  logged "$run/train.log" "$T" turn-train "${common[@]}" --out "$run" --minutes "$minutes" --resume \
    --qat-frac 0.15 --ema 0.999 --eval-every 2000 --adam-lr "$lr" ${hard[@]+"${hard[@]}"} ${opt[@]+"${opt[@]}"}
}

export_model() {
  local ck
  if [ "$dry" = 1 ]; then ck="<last or last-ema>"; else ck=$(python3 -I scripts/turn-gate.py checkpoint --log "$run/log.jsonl"); fi
  echo "== exporting $run/$ck"
  x mkdir -p "$m"
  x "$T" turn-export --run "$run/$ck" --format ternary --out "$m/$name-uncal.gguf"
  [ "$dry" = 1 ] || echo "{\"checkpoint\": \"$ck\"}" >"$run/exported.json"
}

calibrate() {
  local ck
  if [ "$dry" = 1 ]; then ck="<last or last-ema>"; else ck=$(python3 -I -c 'import json, sys; print(json.load(open(sys.argv[1]))["checkpoint"])' "$run/exported.json"); fi
  x "$S" "$m/$name-uncal.gguf" --store "$data/train" --split val --out "$run/val-uncal.jsonl"
  gate_py calibrate --probs "$run/val-uncal.jsonl" --store "$data/train" --out "$run/calib.json"
  x "$T" turn-export --run "$run/$ck" --format ternary --calib "$run/calib.json" --out "$m/$name.gguf"
  x "$S" "$m/$name.gguf" --store "$data/train" --split val --out "$run/val.jsonl"
}

harness() {
  local h=${sky_out}runs/s6-ear2-harness
  if [ -n "$smoke" ]; then
    echo "== SMOKE: no harness check (it reads the test labels)"
    return
  fi
  if [ ! -f "$h/smart-turn.json" ]; then
    x nice -n 19 uv run --python 3.12 scripts/ear-baseline.py --clips "$ear/clips/smart-turn.jsonl" \
      --json "$h/smart-turn.json" --dump "$h/smart-turn-test.jsonl"
  fi
  if ! gate_py harness --baseline "$h/smart-turn.json" --out "$h/check.json"; then
    echo "== the harness check failed (§8.1): the decoding is investigated and fixed by an amendment before any v2 test score is read" >&2
    exit 1
  fi
}

latency() {
  local lat=$run/latency pkg
  pkg=$lat/pkg
  x mkdir -p "$lat"
  if [ "$dry" = 1 ]; then
    x "$S" "$m/$name.gguf" --bench "$iters"
  else
    "$S" "$m/$name.gguf" --bench "$iters" | tee "$lat/native.jsonl"
  fi
  wasm_pkg "$pkg"
  if [ "$dry" = 1 ]; then
    x node crates/sky-web/bench/turn-bench.mjs --pkg "$pkg" --iters "$iters" "$m/$name.gguf"
  else
    node crates/sky-web/bench/turn-bench.mjs --pkg "$pkg" --iters "$iters" "$m/$name.gguf" | tee "$lat/wasm.jsonl"
  fi
  # the reference in the same session: Smart Turn v3.2 CPU through onnxruntime, natively
  x nice -n 19 uv run --python 3.12 scripts/ear-baseline.py --bench "$iters" --json "$lat/smart-turn.json"
  gate_py latency --native "$lat/native.jsonl" --wasm "$lat/wasm.jsonl" --smart-turn "$lat/smart-turn.json" \
    --model "$m/$name.gguf" --out "$lat/latency.json"
}

wasm() {
  local p=$run/parity pkg=$run/latency/pkg
  x mkdir -p "$p"
  [ -f "$pkg/sky_web.js" ] || wasm_pkg "$pkg"
  # 20 test clips through the WAV path natively, then in WASM (no label read)
  x "$S" "$m/$name.gguf" --records "$ear/jev/turn_end.jsonl" --split "$eval_split" --max 20 --out "$p/native.jsonl"
  if [ "$dry" = 1 ]; then
    x node crates/sky-web/bench/parity-node.mjs --turn --pkg "$pkg" "$m/$name.gguf" "$p/native.jsonl"
  else
    node crates/sky-web/bench/parity-node.mjs --turn --pkg "$pkg" "$m/$name.gguf" "$p/native.jsonl" | tee "$p/parity.json" || true
  fi
}

eval_test() {
  note "require ${sky_out}runs/s6-ear2-harness/check.json (§8.1) and $run/latency/latency.json (§8.3) before any v2 test score"
  if [ "$dry" != 1 ] && [ -z "$smoke" ]; then
    [ -f "${sky_out}runs/s6-ear2-harness/check.json" ] || { echo "no harness check on file: run the harness step first (§8.1, §13)" >&2; exit 1; }
    [ -f "$run/latency/latency.json" ] || { echo "no latencies on file: run the latency step first (§8.3, §13)" >&2; exit 1; }
  fi
  if [ -n "$smoke" ]; then
    x "$S" "$m/$name.gguf" --store "$data/train" --split val --out "$run/test-probs.jsonl"
  else
    x "$S" "$m/$name.gguf" --store "$data/test" --out "$run/test-probs.jsonl"
  fi
}

eotbench() {
  [ -n "$smoke" ] && { echo "== SMOKE: no eot-bench"; return; }
  x "$S" "$m/$name.gguf" --store "$data/eotbench/store" --out "$run/eotbench-probs.jsonl"
}

gate() {
  local extra=()
  [ -n "$smoke" ] && extra=(--smoke --labels-store "$data/train")
  gate_py gate --run "$run" --test-probs "$run/test-probs.jsonl" --labels "$ear/clips/smart-turn.jsonl" \
    --harness "${sky_out}runs/s6-ear2-harness/check.json" --smart-turn-dump "${sky_out}runs/s6-ear2-harness/smart-turn-test.jsonl" \
    --latency "$run/latency/latency.json" --parity "$run/parity/parity.json" --model "$m/$name.gguf" \
    --calib "$run/calib.json" --val-probs "$run/val.jsonl" --data "$data" \
    --eotbench-probs "$run/eotbench-probs.jsonl" --pilot "$pilot_dir/choice.json" --probe "$probe_dir/rung.json" \
    --out "$run/GATE.md" ${extra[@]+"${extra[@]}"}
}

steps=("$@")
if [ ${#steps[@]} -eq 0 ]; then
  steps=(probe pilot train export calibrate harness latency wasm eval eotbench gate)
fi
for s in "${steps[@]}"; do
  case "$s" in
    probe | data | eotbench-data | pilot | train | export | calibrate | harness | latency | wasm | eval | eotbench | gate) ;;
    *) echo "unknown step $s (probe, data, eotbench-data, pilot, train, export, calibrate, harness, latency, wasm, eval, eotbench, gate)" >&2; exit 2 ;;
  esac
done
# build what the steps run, unless SKY_BIN names prebuilt binaries
if [ "$dry" != 1 ] && [ -z "${SKY_BIN:-}" ]; then
  nice -n 19 cargo build -q --release -p sky-infer -p sky-data -p sky-train
fi
for s in "${steps[@]}"; do
  case "$s" in
    probe) probe ;;
    data) data ;;
    eotbench-data) eotbench_data ;;
    pilot) pilot ;;
    train) train ;;
    export) export_model ;;
    calibrate) calibrate ;;
    harness) harness ;;
    latency) latency ;;
    wasm) wasm ;;
    eval) eval_test ;;
    eotbench) eotbench ;;
    gate) gate ;;
  esac
done
