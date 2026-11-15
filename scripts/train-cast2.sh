#!/usr/bin/env bash
# Train s6-cast2-pico and evaluate it, as pre-registered in docs/s6-cast2-prereg.md (s6-cast v2:
# the computed-periodicity front end, band-matched real series, the v2 generator periods, the
# sign flip at 0.5, 720 working minutes). Modelled on scripts/train-cast.sh (v1).
#
# Usage: scripts/train-cast2.sh [data] [chronos-check] [pilot] [train] [export] [latency] [eval]
#                              [chronos] [gate]
#   (default: all of them but data, in this order; the order is the pre-registration's: no v2
#   score is read before the checks of §7.1, the latencies of §7.2 and the harness check of §7.3)
#   data     the v1 real series (sky-data fetch-series, cast-synth --games) and GIFT-Eval's windows
#            as v1; the candidate sources of §3.2 (scripts/cast2-data.py list, metadata only), the
#            fetch of the sources in SOURCES, and the leakage check of §3.3 (leak.json)
#   pilot    §5: at 3e-3 only, 3,000 steps without and with the TinyCast teacher; v1's rule
#            (distil unless the validation pinball is worse by more than 1%) writes
#            runs/s6-cast2-pilot/choice.json, which train reads
#   train    §5: 720 working minutes, batch 256, AdamW 3e-3 with v1's recipe flags, WSD, ternary QAT
#            over the last 20%, EMA 0.999, seed 1, the v2 data flags (real share 0.6 by band, the
#            v2 periods, flip and reversal at 0.5); the teacher by the pilot's choice
#   export   §6: the EMA if its final ternary validation pinball is below last's, else last;
#            ternary GGUF (<name>-full.gguf), exit calibration (alpha 0.05, delta 0.02, 2,000
#            unseen synthetic windows) into <name>.gguf
#   latency  §7.2: v1, v2 and the TinyCast port in one session: median and p95 over 1,000
#            forecasts, one core, 2,048-point history, 64-step horizon, full depth, natively
#            (sky-cast --bench, the default TernaryCompute mode) and in WASM (Node, sky-web's
#            Forecaster, SIMD128); also v2 with its calibrated exits and a 720-step horizon
#   eval     §7.1: the two checks (seasonal naive nGMASE 1.00 +- 0.02, the TinyCast port's files
#            give back 0.7856 / 0.5570), the TinyCast port without flip once, then v2 without flip
#            (gated) and with flip (reported)
#   chronos-check  §7.3: the Chronos-ZS export and the harness check (the TinyCast port with flip
#            within 0.02 of 0.880 / 0.722); G3 is a gate only if it passed before training starts
#   chronos  the harness check if not on file, then v2 without flip on Chronos-ZS
#   gate     §8: the verdict (G1, G2, G3) with the reported items of §9 and the disclosure of §10
#            (scripts/s6-cast2-gate.py), into <run>/GATE.md and gate.json
#
# Environment:
#   MINUTES=720  TIME_SCALE=1       working minutes of training (TIME_SCALE multiplies them)
#   PILOT_STEPS=3000                steps of each pilot
#   SOURCES=uscrn_subhourly         the new sources the data step fetches (the list of §3.2, as
#                                   the amendment that freezes it gives it)
#   BAND_PERIODS=second=360,weekly=52,monthly=12:4
#                                   periods of the synthetic stand-ins of the bands without real
#                                   series (§3.4; a reading to settle by amendment)
#   G3=gate                         gate | reported: reported once an amendment has dropped G3 (§8)
#   SKY_DUTY=0.5  SKY_COOL=45:15  SKY_BRAKE_FILE=runs/brake   the brake (amendments 4-8 of
#                                   docs/s6-cast-prereg.md; scripts/auto-brake.sh writes the file
#                                   and stops a training over its memory cap)
#   SKY_MLX_CACHE_GB=2              cap of MLX's buffer cache
#   RESUME=1                        continue the training from its last checkpoint (only after a
#                                   technical failure, §5); without it an existing run is refused
#   SKY_OUT                         a prefix for everything this script writes
#   SMOKE=1                         a few minutes under a throwaway prefix (default
#                                   runs/cast2-smoke/), never runs/s6-cast2* or models/s6-cast2:
#                                   pilots of PILOT_STEPS=40, MINUTES=2 of training on a small
#                                   pool, 20 forecasts per latency, GIFT-Eval on two
#                                   configurations, no Chronos-ZS. It tests that the commands start
#                                   and the files chain; none of its numbers is a result
#   DRY_RUN=1                       print the commands instead of running them (nothing is built,
#                                   trained or read)
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
  SKY_OUT=${SKY_OUT:-runs/cast2-smoke/}
  PILOT_STEPS=${PILOT_STEPS:-40}
  MINUTES=${MINUTES:-2}
fi
sky_out=${SKY_OUT:-}
B=$CARGO_TARGET_DIR/release
T=$B/sky-train
name=s6-cast2-pico
cfg=configs/$name.json
run=${sky_out}runs/$name
pilot_dir=${sky_out}runs/s6-cast2-pilot
data_dir=${sky_out}runs/s6-cast2-data
m=${sky_out}models/s6-cast2
gout=${sky_out}data/gift-eval-out
cout=${sky_out}data/chronos-zs-out
golden=crates/sky-infer/tests/golden/tinycast.gguf
v1=models/s6-cast/s6-cast-pico
pilot_steps=${PILOT_STEPS:-3000}
minutes=$(LC_ALL=C awk -v a="${MINUTES:-720}" -v b="${TIME_SCALE:-1}" 'BEGIN { printf "%g", a * b }')
read -r -a sources <<<"${SOURCES:-uscrn_subhourly}"
g3=${G3:-gate}
case "$g3" in gate | reported) ;; *) echo "G3=$g3: gate or reported" >&2; exit 2 ;; esac
# a smoke run never writes where the real run does
if [ -n "$smoke" ]; then
  case "$run" in
    "runs/$name" | "./runs/$name" | "$PWD/runs/$name")
      echo "SMOKE=1 needs a prefix that is not the real run's (SKY_OUT=$sky_out)" >&2
      exit 2
      ;;
  esac
  echo "== SMOKE: everything under $sky_out; nothing of it is a result"
fi

# The training flags: v1's recipe (runs/s6-cast-pico/args.json: AdamW with --no-muon, decoupled
# weight decay 0.01, clip 1, no StableQAT, augment 0.3, mixup 0.1, missing 0.1, a pool of 4,096
# refreshed by 256 every 50 steps), then what v2 changes (§3, §5). Each flag is checked in
# crates/sky-train/src/cast/train.rs (TrainArgs).
pool=4096 refresh=256 val_synth=512 log_every=50 eval_every=500
# SMOKE: a small pool and validation set, so that a run of two minutes reaches its checkpoints
[ -n "$smoke" ] && pool=256 refresh=32 val_synth=64 log_every=10 eval_every=20
recipe=(--batch 256 --seed 1 --adam-lr 3e-3 --no-muon --weight-decay 0.01 --wd-mode decoupled
  --clip-norm 1.0 --qat-grad-a 0 --augment 0.3 --mixup 0.1 --missing 0.1
  --pool "$pool" --refresh "$refresh" --refresh-every 50 --val-synth "$val_synth" --log-every "$log_every")
bands=hourly=31,subhourly=30,daily=15,weekly=8,monthly=7,second=6
v2_data=(--real-frac 0.6 --synth-periods v2 --flip-prob 0.5 --band-weights "$bands"
  --band-periods "${BAND_PERIODS:-second=360,weekly=52,monthly=12:4}")
# the real series: v1's, and the new ones of §3.2 once fetched and checked (without them C3 is
# not made, which only the amendment of §3.2 allows)
series=data/series
[ -d data/series2 ] && series=data/series,data/series2

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
# logged <file> <command...>: x, with what the command prints kept in <file> (the brake says
# there when its mode changes).
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

# Final validation pinball (normalized) of a run's last exit in `mode`.
final_pinball() {
  python3 -I - "$1/log.jsonl" "$2" <<'EOF'
import json, sys
best = None
for line in open(sys.argv[1]):
    v = json.loads(line)
    if v.get("mode") == sys.argv[2] and "val" in v:
        exits = v["val"]["pinball"]
        best = exits[max(exits, key=lambda k: int(k[4:]))]
print(best if best is not None else "nan")
EOF
}

data() {
  [ -f data/series/wiki_daily.jsonl ] || x "$B/sky-data" fetch-series --out data/series
  [ -f data/series/games.jsonl ] || x "$T" cast-synth --games flappy,rescue,dodge,doom \
    --episodes 60 --out data/series/games.jsonl
  if [ ! -f data/gift-eval/m4_hourly__H__short.json ]; then
    [ -d data/gift-eval-src ] || x uv run --python 3.12 scripts/gift-eval.py download --data data/gift-eval-src
    x uv run --python 3.12 scripts/gift-eval.py export --data data/gift-eval-src --out data/gift-eval
  fi
  x mkdir -p "$data_dir"
  x python3 -I scripts/cast2-data.py list --out "$data_dir/list.json"
  local s
  for s in ${sources[@]+"${sources[@]}"}; do
    [ -f "data/series2/$s.jsonl" ] || x python3 -I scripts/cast2-data.py fetch --source "$s" --out data/series2
  done
  # §3.3: run once, its counts recorded before training
  [ -f "$data_dir/leak.json" ] || x uv run --python 3.12 scripts/cast2-data.py leak-check --series data/series2 \
    --gift-eval data/gift-eval-src --chronos-cache data/chronos-zs-src --out "$data_dir/leak.json"
}

# No training before the leakage check: every file of data/series2 was checked and kept.
check_leak() {
  # a smoke run reads the real run's check (the data is shared; the smoke writes none)
  local leak=$data_dir/leak.json
  [ -n "$smoke" ] && [ ! -f "$leak" ] && leak=runs/s6-cast2-data/leak.json
  note "check that $leak covers every file of data/series2 and keeps each"
  [ "$dry" = 1 ] && return
  if ! compgen -G "data/series2/*.jsonl" >/dev/null; then
    echo "== no new real series in data/series2: C3 is not made (§3.2: only by amendment)"
    return
  fi
  python3 -I - "$leak" data/series2 <<'EOF'
import json, pathlib, sys
leak, d = sys.argv[1], pathlib.Path(sys.argv[2])
r = json.load(open(leak))["sources"]
for f in sorted(d.glob("*.jsonl")):
    s = r.get(f.stem)
    if s is None:
        sys.exit(f"{f} is not in {leak}: run the data step")
    if not s["kept"]:
        sys.exit(f"{f}: {s['matching_runs']} runs match a benchmark series; §3.3 drops the source: remove it")
print(f"leakage check: {len(r)} source(s), all kept")
EOF
}

# A run of the pre-registration is made once: an existing one is continued only with RESUME=1.
fresh_or_resume() {
  local dir=$1
  if [ -f "$dir/last/state.json" ] && [ -z "${RESUME:-}" ] && [ -z "$smoke" ]; then
    echo "$dir already holds a run: RESUME=1 continues it after a technical failure (§5)" >&2
    exit 1
  fi
}

pilot() {
  if [ -f "$pilot_dir/choice.json" ]; then
    echo "== pilot: done, $pilot_dir/choice.json kept"
    return
  fi
  check_leak
  local p
  for p in plain distill; do
    local extra=()
    [ "$p" = distill ] && extra=(--teacher "$golden" --teacher-weight 0.5 --teacher-frac 0.5)
    fresh_or_resume "$pilot_dir/$p"
    logged "$pilot_dir/$p.log" "$T" cast-train --config "$cfg" --out "$pilot_dir/$p" --resume \
      --steps "$pilot_steps" --qat-frac 0 --eval-every "$pilot_steps" --series "$series" \
      "${recipe[@]}" "${v2_data[@]}" ${extra[@]+"${extra[@]}"}
  done
  note "apply v1's rule (distil unless the pinball is worse by more than 1%) -> $pilot_dir/choice.json"
  [ "$dry" = 1 ] && return
  local pd pn
  pd=$(final_pinball "$pilot_dir/distill" f32)
  pn=$(final_pinball "$pilot_dir/plain" f32)
  python3 -I -c '
import json, sys
pd, pn, golden, out = float(sys.argv[1]), float(sys.argv[2]), sys.argv[3], sys.argv[4]
teacher = "none" if pd > 1.01 * pn else golden
json.dump({"lr": "3e-3", "teacher": teacher, "pinball_distill": pd, "pinball_plain": pn}, open(out, "w"), indent=2)
print(f"distillation: {pd:.5g} vs {pn:.5g} without -> teacher {teacher}")' "$pd" "$pn" "$golden" "$pilot_dir/choice.json"
}

train() {
  check_leak
  local teacher=()
  if [ "$dry" = 1 ] && [ ! -f "$pilot_dir/choice.json" ]; then
    teacher=(--teacher "<$golden, or none, by $pilot_dir/choice.json>" --teacher-weight 0.5 --teacher-frac 0.5)
  else
    local t
    t=$(python3 -I -c 'import json, sys; print(json.load(open(sys.argv[1]))["teacher"])' "$pilot_dir/choice.json")
    [ "$t" = none ] || teacher=(--teacher "$t" --teacher-weight 0.5 --teacher-frac 0.5)
  fi
  fresh_or_resume "$run"
  if [ "$dry" != 1 ] && [ ! -f "$run/started.json" ]; then
    mkdir -p "$run"
    python3 -I -c 'import json, sys, time; json.dump({"started": time.strftime("%Y-%m-%dT%H:%M:%S%z")}, open(sys.argv[1], "w"))' "$run/started.json"
  fi
  echo "== $name: ${minutes} working minutes (brake $SKY_DUTY ${SKY_COOL:-no long rests}, file $SKY_BRAKE_FILE)"
  logged "$run/train.log" "$T" cast-train --config "$cfg" --out "$run" --minutes "$minutes" --resume \
    --series "$series" --qat-frac 0.2 --ema 0.999 --eval-every "$eval_every" "${recipe[@]}" "${v2_data[@]}" \
    ${teacher[@]+"${teacher[@]}"}
}

# The exported checkpoint: the EMA if its final ternary pinball is lower, else the last weights.
pick() {
  local e l
  e=$(final_pinball "$run" ema-ternary)
  l=$(final_pinball "$run" ternary)
  python3 -I -c "import sys; e, l = float(sys.argv[1]), float(sys.argv[2]); print('last-ema' if e == e and e < l else 'last')" "$e" "$l"
}

export_model() {
  local ck
  if [ "$dry" = 1 ]; then ck="<last or last-ema>"; else ck=$(pick); fi
  echo "== exporting $run/$ck"
  x mkdir -p "$m"
  x "$T" cast-export --run "$run/$ck" --format ternary --out "$m/$name-full.gguf"
  x "$B/sky-cast-eval" calibrate --model "$m/$name-full.gguf" --out "$m/$name.gguf" --alpha 0.05 --delta 0.02 --n 2000
  [ "$dry" = 1 ] || echo "{\"checkpoint\": \"$ck\"}" >"$run/exported.json"
}

latency() {
  local iters=1000 lat=$run/latency pkg
  [ -n "$smoke" ] && iters=20
  pkg=$lat/pkg
  x mkdir -p "$lat"
  # native: one core (the engine runs a forecast on one thread), the default TernaryCompute mode
  local native=(
    "v1 full $v1-full.gguf 64"
    "v2 full $m/$name-full.gguf 64"
    "tinycast full $golden 64"
    "v2 calibrated $m/$name.gguf 64"
    "v2 full $m/$name-full.gguf 720"
    "v1 full $v1-full.gguf 720"
    "tinycast full $golden 720"
  )
  local line model depth file h
  for line in "${native[@]}"; do
    read -r model depth file h <<<"$line"
    local fd=()
    [ "$depth" = full ] && fd=(--full-depth)
    if [ "$dry" = 1 ]; then
      x "$B/sky-cast" "$file" --synth tsi --seed 7 --len 2048 --horizon "$h" ${fd[@]+"${fd[@]}"} --bench "$iters"
    else
      printf '%s %s %s ' "$model" "$depth" "$h" >>"$lat/native.txt"
      "$B/sky-cast" "$file" --synth tsi --seed 7 --len 2048 --horizon "$h" ${fd[@]+"${fd[@]}"} --bench "$iters" | tee -a "$lat/native.txt"
    fi
  done
  # WASM: sky-web's Forecaster in Node, SIMD128 (the module of the site's baseline build)
  x env RUSTFLAGS="-C target-feature=+simd128" CARGO_TARGET_DIR="$CARGO_TARGET_DIR/sky-web-node" \
    wasm-pack build crates/sky-web --target nodejs --release --out-dir "$PWD/$pkg" --no-typescript
  for h in 64 720; do
    if [ "$dry" = 1 ]; then
      x node crates/sky-infer/bench/forecast-node.mjs --pkg "$pkg" --iters "$iters" --len 2048 --horizon "$h" \
        "$v1-full.gguf" "$m/$name-full.gguf" "$golden" "$m/$name.gguf"
    else
      node crates/sky-infer/bench/forecast-node.mjs --pkg "$pkg" --iters "$iters" --len 2048 --horizon "$h" \
        "$v1-full.gguf" "$m/$name-full.gguf" "$golden" "$m/$name.gguf" | tee -a "$lat/wasm.jsonl"
    fi
  done
  note "collect $lat/native.txt and $lat/wasm.jsonl into $lat/latency.json (with the host)"
  [ "$dry" = 1 ] && return
  python3 -I - "$lat" <<'EOF'
import json, platform, re, subprocess, sys, time
lat = sys.argv[1]
native = []
for line in open(f"{lat}/native.txt"):
    m = re.match(r"(\S+) (\S+) (\d+) .*?: (\d+) points of history, horizon (\d+): median (\d+) µs, p90 (\d+) µs, p95 (\d+) µs \((\d+) runs\)", line)
    if m:
        native.append({"model": m[1], "depth": m[2], "horizon": int(m[5]), "history": int(m[4]),
                       "median_us": int(m[6]), "p95_us": int(m[8]), "runs": int(m[9])})
wasm = [json.loads(l) for l in open(f"{lat}/wasm.jsonl") if l.strip()]
cpu = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
json.dump({"measured": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "host": cpu or platform.processor(),
           "native": native, "wasm": wasm}, open(f"{lat}/latency.json", "w"), indent=2)
print(f"wrote {lat}/latency.json")
EOF
}

score() {
  x uv run --python 3.12 scripts/gift-eval.py score --data data/gift-eval-src --forecasts "$1" --name "$2" \
    ${gcfg[@]+"${gcfg[@]}"}
}

eval_all() {
  local gcfg=() rcfg=()
  if [ -n "$smoke" ]; then
    gcfg=(--configs m4_hourly/H/short ett1/H/long)
    rcfg=(--configs "m4_hourly/H/short,ett1/H/long")
  fi
  # §7.2 and §7.1's checks come first; nothing of v2 is scored before them
  note "require $run/latency/latency.json (§7.2) before any v2 score"
  [ "$dry" = 1 ] || [ -f "$run/latency/latency.json" ] || { echo "no $run/latency/latency.json: run the latency step first (§7.2, §12)" >&2; exit 1; }
  note "check: data/gift-eval-out/seasonal-naive nGMASE within 1.00 +- 0.02; data/gift-eval-out/tinycast gives back 0.7856 / 0.5570"
  [ "$dry" = 1 ] || python3 -I scripts/s6-cast2-gate.py checks --gift-eval-out data/gift-eval-out --out "$run/checks.json"
  if [ ! -f "$gout/tinycast-noflip/summary.json" ]; then
    x "$B/sky-cast-eval" run --model "$golden" --windows data/gift-eval --out "$gout/tinycast-noflip" ${rcfg[@]+"${rcfg[@]}"}
    score "$gout/tinycast-noflip" TinyCast-ours-noflip
  fi
  x "$B/sky-cast-eval" run --model "$m/$name.gguf" --windows data/gift-eval --out "$gout/$name" ${rcfg[@]+"${rcfg[@]}"}
  score "$gout/$name" "$name"
  x "$B/sky-cast-eval" run --model "$m/$name.gguf" --windows data/gift-eval --out "$gout/$name-flip" --flip ${rcfg[@]+"${rcfg[@]}"}
  score "$gout/$name-flip" "$name-flip"
}

# §7.3: the harness check, the TinyCast port with flip as published. G3 stays a gate only when it
# passes before training starts (§8): run this step first.
chronos_check() {
  if [ -n "$smoke" ]; then
    echo "== SMOKE: no Chronos-ZS"
    return
  fi
  [ -f data/chronos-zs/monash_traffic__w0.json ] || x uv run --python 3.12 scripts/chronos-zs.py export \
    --cache data/chronos-zs-src --out data/chronos-zs
  if [ ! -f "$cout/tinycast/check.json" ]; then
    x "$B/sky-cast-eval" run --model "$golden" --windows data/chronos-zs --out "$cout/tinycast" --flip
    x uv run --python 3.12 scripts/chronos-zs.py score --cache data/chronos-zs-src --forecasts "$cout/tinycast" --name TinyCast-ours
    x uv run --python 3.12 scripts/chronos-zs.py check --forecasts "$cout/tinycast" || true
  fi
}

chronos() {
  [ -n "$smoke" ] && { echo "== SMOKE: no Chronos-ZS"; return; }
  chronos_check
  x "$B/sky-cast-eval" run --model "$m/$name.gguf" --windows data/chronos-zs --out "$cout/$name"
  x uv run --python 3.12 scripts/chronos-zs.py score --cache data/chronos-zs-src --forecasts "$cout/$name" --name "$name"
}

gate() {
  local diag=()
  # §9's per-window items (coverage, horizon tenths, window-level splits), by the study's method
  if [ -z "$smoke" ]; then
    x uv run --python 3.12 scripts/s6-cast2-diagnostics.py --data data/gift-eval-src --windows data/gift-eval \
      --model v1=data/gift-eval-out/s6-cast-pico --model "v2=$gout/$name" --model port=data/gift-eval-out/tinycast \
      --model sn=data/gift-eval-out/seasonal-naive --out "$run/diagnostics.json"
    diag=(--diagnostics "$run/diagnostics.json")
  fi
  x python3 -I scripts/s6-cast2-gate.py gate --run "$run" --gift-eval-out "$gout" \
    --reference-out data/gift-eval-out --chronos-out "$cout" --models "$m" --g3 "$g3" \
    --data "$data_dir" --out "$run/GATE.md" ${diag[@]+"${diag[@]}"} ${smoke:+--smoke}
}

steps=("$@")
if [ ${#steps[@]} -eq 0 ]; then
  steps=(pilot train export latency eval chronos gate)
fi
for s in "${steps[@]}"; do
  case "$s" in
    data | chronos-check | pilot | train | export | latency | eval | chronos | gate) ;;
    *) echo "unknown step $s (data, chronos-check, pilot, train, export, latency, eval, chronos, gate)" >&2; exit 2 ;;
  esac
done
[ "$dry" = 1 ] || cargo build -q --release -p sky-train -p sky-infer -p sky-data
for s in "${steps[@]}"; do
  case "$s" in
    data) data ;;
    chronos-check) chronos_check ;;
    pilot) pilot ;;
    train) train ;;
    export) export_model ;;
    latency) latency ;;
    eval) eval_all ;;
    chronos) chronos ;;
    gate) gate ;;
  esac
done
