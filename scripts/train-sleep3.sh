#!/usr/bin/env bash
# s6-sleep v3: the check C5, the pico pilot, the nano runs, their export, the test reading and the
# gate of docs/s6-sleep3-prereg.md (v3 = docs/s6-sleep2-prereg.md, v2, with T1 and T2).
#
#   scripts/train-sleep3.sh [c5] [pilot] [report] [nano] [export] [eval] [gate]
#                                                  # no step: c5 only (nothing else is automatic)
#
#   c5      v3 §3, on the CPU at nice 19: C4's setting of v2 (scripts/train-sleep2.sh) with T1 (the
#           gates held, --mem-gate-lr-mult 0) and T2 (the auxiliary recall loss, memory_aux and
#           --memory-aux-weight 1), 600 steps, seed 0 then seed 1 (never in parallel), each with
#           --memory-off-eval 4; read over steps 550 to 600 by scripts/s6-sleep3-report.py c5 ->
#           <root>/checks/c5.json. No retry (v3 §3)
#   pilot   refused unless C1 to C3 of v2 are on file as passed (V2_CHECKS/verdict.json, as read
#           with amendments 1 to 4) and C5 passed (<root>/checks/ready.json). v2 §5 with T1 and T2
#           in arm M: M phase A (8 working minutes), the in-sample stop, M phase B (12 working
#           minutes), C phase A and B (M's steps, phase by phase), the dev2 readings (s2-probe,
#           ternary, last exit, closed loop; M at N = 1, 2, 4, C once, M with its memory zeroed)
#   report  the rule of v2 §5.3 with its bootstrap, the splits by delay and by one colour against
#           several (with the memory and zeroed) -> <root>/pilot.json
#   nano    refused unless the pilot rule passed. v2 §6: the sleep arm (T1, T2) and the control,
#           seeds 0, 1, 2; the sleep arm at seed 0 trains 12 + 18 working minutes, the five others
#           its steps, phase by phase; dev2 readings and the choice of N -> <root>/n.json
#   export  calibration (zero memory) and ternary export of the six nano runs (v1 §6); the
#           exported graph carries the memory and not the auxiliary head
#   eval    refused before every nano run has ended and N is chosen; the test seeds, READ ONCE
#           (v1 §7): probes, arena (s2-probe), typed_decisions_bench; then H0 to H3 -> REPORT.md
#   gate    v2 §7: the s6-nano gate on the exported seed-0 sleep model, H0 to H3, and sky-infer's
#           parity with the training code on the golden sequences, natively and in WASM
#
# Runs that ended are skipped, so the script can be started again after a stop; RESUME=1 also
# resumes the run that was cut (a technical failure restarts the same run with the same seed).
#
#   SMOKE=1        tiny budgets, sizes and episodes, data/numeric as the only stateless records, the
#                  verdicts recorded but not obeyed, everything under SKY_OUT (default
#                  runs/smoke-sleep3/; never runs/s6-sleep3*): a code check, no result
#   SKY_OUT        a prefix for everything this script writes (empty for the real run)
#   DRY_RUN=1      print the commands
#   CPU=1          every training on the CPU (--cpu); C5 and the probes always are
#   SKY_TRAIN      a sky-train binary to use as is, instead of building the release one
#   PICO_BASE      runs/s6lr/softmax-adamw-x2/decide/last (the s6-pico of the adopted recipe)
#   NANO_BASE      runs/s6-nano-g${ROUNDS:-3}/decide/last (the s6-nano of tranche 2)
#   SEQ            the training sequences (default runs/s6-sleep/seq, v1's)
#   DATA           the folder of the stateless records (default data)
#   V2_CHECKS      v2's check outputs (default runs/s6-sleep2/checks): verdict.json, sleep2.json
#   LR_X           the nano rate scale (default: runs/s6-nano-lr/summary.json)
#   OPT LR_SCALE   the recipe, as scripts/train-sleep2.sh (default --no-muon, scale 2)
#   STEP_MEM_GB=3  GPU memory of one stateless step
#   SLEEP_CHUNK    the nano runs only, both arms, phases A and B: --sleep-chunk, the sequences of one
#                  part of a sequence batch (empty: the batch of 16 at once, as before). The same
#                  batches, loss and update (the gradients of the parts are summed before the one
#                  update; equal up to the order of floating-point sums), with the memory of one
#                  part: a part holds only the static streams its sequences use, where the whole
#                  batch pads every frame to 32 of them. SLEEP_CHUNK=2 for the nano runs (an
#                  unchunked nano sequence step peaked at 13.3 GB of MLX memory and the run was
#                  stopped by the memory brake; in parts of 2 the CPU probe peaks at 1.7 to 2.1 GB)
#   SKY_DUTY SKY_COOL SKY_BRAKE_FILE   the thermal brake (scripts/auto-brake.sh; started with the run
#                  by scripts/train-when-idle.sh -- scripts/train-sleep3.sh <steps>: docs/idle-training.md)
set -euo pipefail
export DEVELOPER_DIR=${DEVELOPER_DIR:-/Applications/Xcode.app/Contents/Developer}
export CARGO_TARGET_DIR=${CARGO_TARGET_DIR:-$PWD/target/s3-train}
export SKY_DUTY=${SKY_DUTY:-0.5}
export SKY_COOL=${SKY_COOL-45:15}
export SKY_BRAKE_FILE=${SKY_BRAKE_FILE-$PWD/runs/brake}
export SKY_MLX_CACHE_GB=${SKY_MLX_CACHE_GB:-2}
dry=${DRY_RUN:-0}
smoke=${SMOKE:-0}
B=$CARGO_TARGET_DIR/release
T=${SKY_TRAIN:-$B/sky-train}
if [ "$smoke" = 1 ]; then
  SKY_OUT=${SKY_OUT:-runs/smoke-sleep3/}
fi
out=${SKY_OUT:-}
root=${out}runs/s6-sleep3
models=${out}models/s6-sleep3-nano
if [ "$smoke" = 1 ]; then
  case "$root" in
    runs/s6-sleep3* | ./runs/s6-sleep3* | "$PWD"/runs/s6-sleep3*)
      echo "SMOKE=1 needs a prefix that is not the real run's (SKY_OUT=$out)" >&2
      exit 2
      ;;
  esac
  echo "== SMOKE: everything under $out; nothing of it is a result"
fi
smoke_arg=()
[ "$smoke" = 1 ] && smoke_arg=(smoke)

# Budgets and sizes (v2 §3, §5, §6; v3 §3).
phase_a_min=${PHASE_A_MIN:-8} phase_b_min=${PHASE_B_MIN:-12}
nano_a_min=${NANO_A_MIN:-12} nano_b_min=${NANO_B_MIN:-18}
c5_steps=600 c5_lo=550 c5_hi=600 c5_val=256
dev_eps=500 dev_seed=2000000000
test_eps=1000 test_seed=5000000000 arena_eps=200 arena_seed=5000
data=${DATA:-data}
records=$data/decisions,$data/games,$data/numeric weights=decisions=0.3,games=0.6,numeric=0.1
nano_records=$data/decisions,$data/distill,$data/games,$data/numeric
nano_weights=decisions=0.25,distill=0.30,games=0.35,numeric=0.10
if [ "$smoke" = 1 ]; then
  phase_a_min=${PHASE_A_MIN:-0.3} phase_b_min=${PHASE_B_MIN:-0.3}
  nano_a_min=${NANO_A_MIN:-0.3} nano_b_min=${NANO_B_MIN:-0.3}
  c5_steps=12 c5_lo=1 c5_hi=12 c5_val=16
  dev_eps=3 test_eps=3 arena_eps=1
  # A smoke reads no dev2 seed, no test seed and no arena seed of the experiment: episodes far
  # from all of them (v2 §5.2, v1 §3).
  dev_seed=1000 test_seed=3000 arena_seed=9000
  records=$data/numeric weights=numeric=1
  nano_records=$data/numeric nano_weights=numeric=1
fi
seq=${SEQ:-runs/s6-sleep/seq}
games=(flappy rescue dodge doom)
pico_base=${PICO_BASE:-runs/s6lr/softmax-adamw-x2/decide/last}
nano_base=${NANO_BASE:-runs/s6-nano-g${ROUNDS:-3}/decide/last}
v2c=${V2_CHECKS:-runs/s6-sleep2/checks}
read -r -a opt <<<"${OPT:---no-muon}"
lr_flag=--muon-lr
[[ " ${OPT:---no-muon} " == *" --no-muon "* ]] && lr_flag=--adam-lr
resume=()
[ -n "${RESUME:-}" ] && resume=(--resume)
cpu=()
[ "${CPU:-0}" = 1 ] && cpu=(--cpu)
# SLEEP_CHUNK: the sequence batches of the nano runs in parts (both arms; empty: unchanged).
nano_chunk=()
[ -n "${SLEEP_CHUNK:-}" ] && nano_chunk=(--sleep-chunk "$SLEEP_CHUNK")
# T1 and T2 (v3 §2), with v2's M3 (mem.* at x10): the flags of every training with a memory.
t12=(--mem-lr-mult 10 --mem-gate-lr-mult 0 --memory-aux-weight 1)

x() {
  if [ "$dry" = 1 ]; then
    printf '+'
    printf ' %q' "$@"
    printf '\n'
  else
    "$@"
  fi
}

# rep <command> [args]: scripts/s6-sleep3-report.py, the rules of the pre-registrations.
rep() { x python3 -I scripts/s6-sleep3-report.py "$@"; }

# rate <width> <size pilot scale>: v1's peak rate (scripts/train-sleep.sh): `rate 96 1` = 2e-3.
rate() { LC_ALL=C awk -v b="${LR_SCALE:-2.0}" -v c="$2" -v d="$1" 'BEGIN { printf "%.3g", 1e-3 * b * sqrt(96 / d) * c }'; }

# config <base checkpoint> <out.json> <memory: frame|none> <name>: the base's config with v2's
# memory (shared qk, values of form frame, as C2 and amendment 4 took it) and the auxiliary head
# of T2 (memory_aux: training only, never exported), or without memory (the control).
config() {
  [ "$dry" = 1 ] && { echo "+ config $*"; return; }
  mkdir -p "$(dirname "$2")"
  python3 -I - "$1/config.json" "$2" "$3" "$4" <<'EOF'
import json, sys
src, dst, form, name = sys.argv[1:]
c = json.load(open(src))
c["temperature"] = {}
c["name"] = name
for k in ("memory", "memory_qk", "memory_value", "memory_write", "memory_aux"):
    c.pop(k, None)
if form != "none":
    c.update(memory=True, memory_qk="shared", memory_value=form, memory_aux=True)
json.dump(c, open(dst, "w"), indent=2)
EOF
}

# same_memory <v3 config> <v2 config>: v3 §2, "nothing else changes": v3's pico config is v2's
# sleep2.json plus memory_aux, whatever the name.
same_memory() {
  [ "$dry" = 1 ] && return 0
  python3 -I - "$1" "$2" <<'EOF'
import json, sys
a, b = (json.load(open(p)) for p in sys.argv[1:])
for c in (a, b):
    c.pop("name", None)
    c.pop("memory_aux", None)
if a != b:
    keys = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
    sys.exit(f"the v3 config differs from v2's sleep2.json beyond memory_aux: {keys}")
EOF
}

# json_get <file> <key>: a top-level value of a JSON file.
json_get() { python3 -I -c 'import json,sys; v=json.load(open(sys.argv[1])).get(sys.argv[2]); print("" if v is None else (str(v).lower() if isinstance(v, bool) else v))' "$1" "$2"; }

# steps_of <run dir>: the steps of a run that ended (empty otherwise).
steps_of() {
  [ -f "$1/last/state.json" ] || return 0
  python3 -I - "$1" <<'EOF'
import json, os, sys
log = os.path.join(sys.argv[1], "log.jsonl")
done = os.path.exists(log) and any(json.loads(l).get("final") for l in open(log) if l.strip())
if done:
    print(json.load(open(os.path.join(sys.argv[1], "last", "state.json")))["step"])
EOF
}

# No pilot or nano run trains during another training (v2 §9); C5, on the CPU, may.
no_other_training() {
  [ "$dry" = 1 ] && return 0
  [ "${CPU:-0}" = 1 ] && return 0
  local others
  others=$(pgrep -lf 'sky-train (s2-decide|s2-pretrain|decide|pretrain|cast-train|s5-train)' | grep -v -- ' --cpu' || true)
  if [ -n "$others" ]; then
    echo "another training runs on this machine; this step waits for it to end:" >&2
    echo "$others" >&2
    exit 1
  fi
}

# decide <out> <config> <init> <tokenizer> <records> <weights> <rate> <seed> <budget flag> <budget>
# [flags]: one s2-decide run with v1's shared settings (stateless batches of 128, QAT from the
# first step, loop dropout 0.25) and the training log of v2's M5 (--seq-log, which the auxiliary
# loss needs). NICE=19 runs it at nice 19.
decide() {
  local dir=$1 cfg=$2 init=$3 tok=$4 recs=$5 wts=$6 lr=$7 seed=$8 flag=$9 budget=${10}
  shift 10
  if [ "$dry" != 1 ] && [ -n "$(steps_of "$dir")" ]; then
    echo "== $dir: done ($(steps_of "$dir") steps), skipped"
    return
  fi
  [ -n "${RESUME:-}" ] || x rm -rf "$dir"
  x mkdir -p "$dir"
  echo "== $dir: $flag $budget, seed $seed $*"
  local -a cmd=(nice -n "${NICE:-0}" "$T" ${cpu[@]+"${cpu[@]}"} s2-decide --config "$cfg" --init "$init" --tokenizer "$tok"
    --records "$recs" --weights "$wts" --batch 128 "$flag" "$budget" --seed "$seed"
    --qat-frac 1.0 --loop-dropout 0.25 --length-pool 16 --pad-to 32 --out "$dir"
    --step-mem-gb "${STEP_MEM_GB:-3}" "$lr_flag" "$lr" ${opt[@]+"${opt[@]}"} ${resume[@]+"${resume[@]}"} --seq-log "$@")
  # The command of a run, kept next to its log (a file of the run, not a result).
  [ "$dry" = 1 ] || printf '%q ' "${cmd[@]}" >"$dir/cmd.txt"
  x "${cmd[@]}"
}

# probe <checkpoint> <game> <first seed> <episodes> <passes> <out.json> [flags]: s2-probe on the CPU
# at nice 19 (ternary weights, last exit, closed loop), skipped when the reading exists.
probe() {
  local ck=$1 game=$2 from=$3 n=$4 passes=$5 json=$6
  shift 6
  if [ "$dry" != 1 ] && [ -f "$json" ]; then
    return
  fi
  x mkdir -p "$(dirname "$json")"
  x nice -n 19 "$T" s2-probe "$ck" --game "$game" --from-seed "$from" --episodes "$n" --passes "$passes" --json "$json" "$@"
}

# v2 §2 M4. Phase A: sequence steps at 0.9, recall 0.85 and rule110 0.15, no sleep (N = 1).
phase_a=(--sleep-records "$seq/recall.jsonl,$seq/rule110.jsonl" --sleep-weights 'recall=0.85,rule110=0.15'
  --sleep-batch 16 --seq-frac 0.9 --sleep-passes 1)
# Phase B: v1's mix, N drawn in {1, 2, 3, 4}, truncated.
phase_b=(--sleep-records "$seq/recall.jsonl,$seq/rule110.jsonl,$seq/flappy.jsonl,$seq/rescue.jsonl,$seq/dodge.jsonl,$seq/doom.jsonl"
  --sleep-weights 'recall=0.40,rule110=0.12,flappy=0.12,rescue=0.12,dodge=0.12,doom=0.12'
  --sleep-batch 16 --seq-frac 0.5 --sleep-passes '1,2,3,4')

# ----------------------------------------------------------------------------- C5

c5() {
  local c=$root/checks seed
  x mkdir -p "$c/c5"
  config "$pico_base" "$c/sleep3.json" frame s6-sleep3-pico
  if [ "$smoke" = 1 ] && [ ! -f "$v2c/sleep2.json" ]; then
    echo "== SMOKE: no $v2c/sleep2.json to compare the config with"
  else
    same_memory "$c/sleep3.json" "$v2c/sleep2.json"
  fi
  # C4's setting (v2 §3; scripts/train-sleep2.sh c4): recall as the only sequence game, batches of
  # 8 sequences, rate 2e-3 with mem.* at x10, N = 1, QAT from the first step; plus T1 and T2.
  local -a c5_flags=(--sleep-records "$seq/recall.jsonl" --sleep-weights recall=1 --sleep-batch 8
    --seq-frac 0.9 --sleep-passes 1 --no-compile --val-max "$c5_val" "${t12[@]}" --memory-off-eval 4)
  if [ "$dry" != 1 ] && [ -f "$c/c5.json" ]; then
    echo "== C5 read: $(json_get "$c/c5.json" pass) (no retry, v3 §3)"
    return
  fi
  # One seed after the other, never in parallel; always on the CPU.
  local saved=("${cpu[@]+"${cpu[@]}"}")
  cpu=(--cpu)
  for seed in 0 1; do
    NICE=19 decide "$c/c5/s$seed" "$c/sleep3.json" "$pico_base" "$pico_base/tokenizer.json" \
      "$records" "$weights" "$(rate 96 1)" "$seed" --steps "$c5_steps" "${c5_flags[@]}"
  done
  cpu=("${saved[@]+"${saved[@]}"}")
  local st=0
  rep c5 "$c/c5/s0" "$c/c5/s1" "$c/c5.json" "$c5_lo" "$c5_hi" || st=$?
  if [ "$st" != 0 ]; then
    if [ "$smoke" = 1 ]; then
      echo "== SMOKE: C5 does not pass at these sizes, as expected"
    else
      echo "== C5 FAILED: no GPU run takes place, and no retry (v3 §3)" >&2
      exit 1
    fi
  fi
}

# ----------------------------------------------------------------------------- the pilot

pilot() {
  local p=$root/pilot a_steps b_steps
  no_other_training
  # Refused unless C1 to C3 of v2 (as read) and C5 are on file as passed (v3 §3).
  rep ready "$root" "$v2c" ${smoke_arg[@]+"${smoke_arg[@]}"}
  x mkdir -p "$p"
  config "$pico_base" "$p/sleep3.json" frame s6-sleep3-pico
  config "$pico_base" "$p/control.json" none s6-pico-control3
  local tok=$pico_base/tokenizer.json lr
  lr=$(rate 96 1)
  # M, phase A (T1, T2), then the in-sample stop (v2 §5.1).
  decide "$p/m-a" "$p/sleep3.json" "$pico_base" "$tok" "$records" "$weights" "$lr" 0 --minutes "$phase_a_min" "${phase_a[@]}" "${t12[@]}"
  local st=0
  rep stop "$root" || st=$?
  if [ "$st" = 3 ]; then
    if [ "$smoke" != 1 ]; then
      echo "== the pilot stops after phase A: phase B and arm C do not run (v2 §5.1)"
      rep pilot "$root"
      exit 0
    fi
    echo "== SMOKE: the stop would end the pilot here; going on to check the code"
  elif [ "$st" != 0 ]; then
    exit "$st"
  fi
  # M, phase B: from phase A's last, a fresh optimiser and schedule, T1 and T2 again.
  decide "$p/m-b" "$p/sleep3.json" "$p/m-a/last" "$tok" "$records" "$weights" "$lr" 0 --minutes "$phase_b_min" "${phase_b[@]}" "${t12[@]}"
  a_steps=$(steps_of "$p/m-a") b_steps=$(steps_of "$p/m-b")
  [ "$dry" = 1 ] && a_steps=STEPS_A b_steps=STEPS_B
  [ -n "$a_steps" ] && [ -n "$b_steps" ] || { echo "M did not end" >&2; exit 1; }
  # C: no memory, so no auxiliary loss (v3 §4); M's steps, phase by phase, on the same batches.
  decide "$p/c-a" "$p/control.json" "$pico_base" "$tok" "$records" "$weights" "$lr" 0 --steps "$a_steps" "${phase_a[@]}"
  decide "$p/c-b" "$p/control.json" "$p/c-a/last" "$tok" "$records" "$weights" "$lr" 0 --steps "$b_steps" "${phase_b[@]}"
  # The dev2 readings (v2 §5.2).
  local g
  for g in recall rule110; do
    probe "$p/m-b/last" "$g" "$dev_seed" "$dev_eps" 1,2,4 "$p/m/dev2/$g.json"
    probe "$p/c-b/last" "$g" "$dev_seed" "$dev_eps" 1 "$p/c/dev2/$g.json"
  done
  probe "$p/m-b/last" recall "$dev_seed" "$dev_eps" 1 "$p/m/dev2/recall-off.json" --memory-off
}

report() { rep pilot "$root" ${smoke_arg[@]+"${smoke_arg[@]}"}; }

# ----------------------------------------------------------------------------- nano

# The nano rate scale: LR_X, else the size's learning-rate pilot (v1 §6, scripts/train-sleep.sh).
lr_x() {
  if [ -n "${LR_X:-}" ]; then
    echo "$LR_X"
  elif [ "$dry" = 1 ] || [ "$smoke" = 1 ]; then
    echo 1
  else
    python3 -I -c 'import json,sys; print(json.load(open(sys.argv[1]))["chosen"])' runs/s6-nano-lr/summary.json
  fi
}

nano() {
  local n=$root/nano lr seed arm a_steps b_steps g tok=$nano_base/tokenizer.json
  no_other_training
  if [ "$dry" != 1 ]; then
    if [ "$smoke" = 1 ]; then
      rep nano-ready "$root" || echo "== SMOKE: the nano runs would be refused here; going on to check the code"
    else
      rep nano-ready "$root"
    fi
  fi
  lr=$(rate 144 "$(lr_x)")
  x mkdir -p "$n"
  config "$nano_base" "$n/sleep3.json" frame s6-sleep3-nano
  config "$nano_base" "$n/control.json" none s6-nano-control3
  # The sleep arm at seed 0: 12 + 18 working minutes (v2 §6).
  decide "$n/sleep-s0/a" "$n/sleep3.json" "$nano_base" "$tok" "$nano_records" "$nano_weights" "$lr" 0 --minutes "$nano_a_min" "${phase_a[@]}" "${t12[@]}" ${nano_chunk[@]+"${nano_chunk[@]}"}
  decide "$n/sleep-s0/b" "$n/sleep3.json" "$n/sleep-s0/a/last" "$tok" "$nano_records" "$nano_weights" "$lr" 0 --minutes "$nano_b_min" "${phase_b[@]}" "${t12[@]}" ${nano_chunk[@]+"${nano_chunk[@]}"}
  a_steps=$(steps_of "$n/sleep-s0/a") b_steps=$(steps_of "$n/sleep-s0/b")
  [ "$dry" = 1 ] && a_steps=STEPS_A b_steps=STEPS_B
  [ -n "$a_steps" ] && [ -n "$b_steps" ] || { echo "the first nano run did not end" >&2; exit 1; }
  # The five others: its steps, phase by phase.
  for seed in 1 2; do
    decide "$n/sleep-s$seed/a" "$n/sleep3.json" "$nano_base" "$tok" "$nano_records" "$nano_weights" "$lr" "$seed" --steps "$a_steps" "${phase_a[@]}" "${t12[@]}" ${nano_chunk[@]+"${nano_chunk[@]}"}
    decide "$n/sleep-s$seed/b" "$n/sleep3.json" "$n/sleep-s$seed/a/last" "$tok" "$nano_records" "$nano_weights" "$lr" "$seed" --steps "$b_steps" "${phase_b[@]}" "${t12[@]}" ${nano_chunk[@]+"${nano_chunk[@]}"}
  done
  # The control answers each frame alone but takes the same sequence steps (64 frames in one
  # graph): it gets the same parts.
  for seed in 0 1 2; do
    decide "$n/control-s$seed/a" "$n/control.json" "$nano_base" "$tok" "$nano_records" "$nano_weights" "$lr" "$seed" --steps "$a_steps" "${phase_a[@]}" ${nano_chunk[@]+"${nano_chunk[@]}"}
    decide "$n/control-s$seed/b" "$n/control.json" "$n/control-s$seed/a/last" "$tok" "$nano_records" "$nano_weights" "$lr" "$seed" --steps "$b_steps" "${phase_b[@]}" ${nano_chunk[@]+"${nano_chunk[@]}"}
  done
  # dev2 readings (v2 §5.2's reader), then the choice of N (v1 §6, on dev2).
  for seed in 0 1 2; do
    for g in recall rule110; do
      probe "$n/sleep-s$seed/b/last" "$g" "$dev_seed" "$dev_eps" 1,2,4 "$n/sleep-s$seed/dev2/$g.json"
      probe "$n/control-s$seed/b/last" "$g" "$dev_seed" "$dev_eps" 1 "$n/control-s$seed/dev2/$g.json"
    done
  done
  rep choose "$root" ${smoke_arg[@]+"${smoke_arg[@]}"}
}

# Calibration with a zero memory (stateless records) and the ternary export (v1 §6). s2-export
# writes the memory (mem.{l}.qk, v, out ternary; the norm and the gates f32) and never the
# auxiliary head (s2::gguf::to_gguf).
export_() {
  local arm seed r
  x mkdir -p "$models"
  for arm in sleep control; do
    for seed in 0 1 2; do
      r=$root/nano/$arm-s$seed/b/last
      if [ "$dry" != 1 ] && [ -f "$models/$arm-s$seed.gguf" ]; then
        continue
      fi
      x nice -n 19 "$T" s2-calibrate --run "$r" --records "$nano_records"
      x nice -n 19 "$T" s2-export --run "$r" --format ternary --out "$models/$arm-s$seed.gguf"
    done
  done
}

# The test seeds, read once (v1 §7, amendment 2): the probes at N = 1, 2, 4 (the control once), the
# arena games at the chosen N through the same reader, typed_decisions_bench with a zero memory.
eval_() {
  local t=$root/test arm seed g chosen=1 r
  if [ "$dry" != 1 ]; then
    if [ "$smoke" = 1 ]; then
      rep test-ready "$root" || echo "== SMOKE: the test reading would be refused here; going on to check the code"
      rm -f "$t/done"
    else
      rep test-ready "$root"
    fi
    chosen=$(json_get "$root/n.json" n)
  fi
  x mkdir -p "$t"
  for arm in sleep control; do
    for seed in 0 1 2; do
      r=$root/nano/$arm-s$seed/b/last
      local passes=1,2,4
      [ "$arm" = control ] && passes=1
      for g in recall rule110; do
        probe "$r" "$g" "$test_seed" "$test_eps" "$passes" "$t/$arm-s$seed/$g.json"
      done
      for g in "${games[@]}"; do
        probe "$r" "$g" "$arena_seed" "$arena_eps" "$chosen" "$t/$arm-s$seed/$g.json"
      done
      [ "$dry" != 1 ] && [ -f "$t/$arm-s$seed/typed.md" ] && continue
      x nice -n 19 "$T" s2-eval --run "$r" --records "$data/decisions/typed_decisions_bench.jsonl" \
        --report "$t/$arm-s$seed/typed.md"
    done
  done
  x touch "$t/done"
  rep report "$root" ${smoke_arg[@]+"${smoke_arg[@]}"}
}

# v2 §7 (v1 §8): the s6-nano gate on the exported seed-0 sleep model (sky-eval at full depth with a
# zero memory; the arena through sky-arena with the memory at the chosen N, s3-pico on the same
# seeds), H0 to H3, and the golden parity of sky-infer, natively and in WASM.
gate() {
  local g=$root/gate model=$models/sleep-s0.gguf chosen=1
  local baseline=${BASELINE:-site/public/models/s3-pico.gguf}
  [ "$dry" = 1 ] || chosen=$(json_get "$root/n.json" n)
  x mkdir -p "$g"
  x nice -n 19 "$B/sky-eval" "$model" --records "$data/decisions/typed_decisions_bench.jsonl,$data/s4/heldout/zs_massive_en.jsonl,$data/s4/heldout/zs_massive_fr.jsonl,$data/decisions/when2call.jsonl,$data/decisions/notam.jsonl" \
    --max 2000 --exit-threshold 1.0 --json "$g/eval-full.json"
  # The s6-nano gate's arena (docs/s6-gen-prereg.md §6): 4 games x 30 episodes, the same seeds
  # (sky-arena's) for the model and s3-pico, tau = 1.
  local gate_eps=30
  [ "$smoke" = 1 ] && gate_eps=1
  x nice -n 19 "$B/sky-arena" "$model" --episodes "$gate_eps" --game all --exit-threshold 1.0 \
    --memory-passes "$chosen" --json "$g/arena-model.json"
  x nice -n 19 "$B/sky-arena" "$baseline" --episodes "$gate_eps" --game all --exit-threshold 1.0 \
    --json "$g/arena-baseline.json"
  if [ "$dry" != 1 ]; then
    python3 -I - "$g/arena-model.json" "$g/arena-baseline.json" "$g/arena.json" <<'EOF'
import json, sys
m, b = (json.load(open(p)) for p in sys.argv[1:3])
out = {g: {"model": m[g]["model"]["mean_score"], "baseline": b[g]["model"]["mean_score"]}
       for g in ("flappy", "rescue", "dodge", "doom")}
json.dump(out, open(sys.argv[3], "w"), indent=2)
EOF
  fi
  local native=false wasm=false
  if [ "$dry" = 1 ]; then
    x nice -n 19 cargo test -q --release -p sky-infer --test golden_s6m
    x nice -n 19 wasm-pack test --release --node crates/sky-web --test memory
  else
    nice -n 19 cargo test -q --release -p sky-infer --test golden_s6m && native=true
    nice -n 19 wasm-pack test --release --node crates/sky-web --test memory && wasm=true
    printf '{"native": %s, "wasm": %s}\n' "$native" "$wasm" >"$g/parity.json"
  fi
  if [ "$smoke" = 1 ]; then
    rep gate "$root" || echo "== SMOKE: the gate does not hold at these sizes, as expected"
  else
    rep gate "$root"
  fi
}

[ "$dry" = 1 ] || [ -n "${SKY_TRAIN:-}" ] || nice -n 19 cargo build -q --release -p sky-train
steps=("$@")
[ ${#steps[@]} -eq 0 ] && steps=(c5)
for s in "${steps[@]}"; do
  case "$s" in
    c5) c5 ;;
    pilot) pilot ;;
    report) report ;;
    nano) nano ;;
    export) export_ ;;
    eval) eval_ ;;
    gate)
      [ "$dry" = 1 ] || nice -n 19 cargo build -q --release -p sky-infer -p sky-arena
      gate
      ;;
    *) echo "unknown step $s (c5, pilot, report, nano, export, eval, gate)" >&2; exit 2 ;;
  esac
done
