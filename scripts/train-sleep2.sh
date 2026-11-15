#!/usr/bin/env bash
# s6-sleep v2: the CPU checks and the pico pilot of docs/s6-sleep2-prereg.md (§3, §5).
#
#   scripts/train-sleep2.sh [data] [checks] [pilot] [report]   # default: all, in this order
#
#   data     v1's training sequences (runs/s6-sleep/seq, the same seqgen seed 61), generated into
#            <root>/seq only when they are absent
#   checks   C1 to C4 (§3) on the CPU, on train seeds only, at nice 19, into <root>/checks: C1 to
#            C3 by `sky-train s2-memcheck`, then C4 (two 300-step CPU trainings, the second with
#            its read held at zero), then its two retries if C4 fails; the first failing check stops
#            the script. <root>/checks/verdict.json says whether the pilot may run.
#   pilot    refuses to start unless <root>/checks/verdict.json passes. M phase A (8 working
#            minutes), the in-sample stop (§5.1), M phase B (12 working minutes), C phase A and B
#            (M's steps, phase by phase), then the dev2 probes (§5.2)
#   report   the rule of §5.3 (scripts/s6-sleep2-report.py) -> <root>/pilot.json
#
# Budgets: M's phases are in working minutes (PHASE_A_MIN, PHASE_B_MIN), C's in M's steps. Runs
# that ended are skipped, so the script can be started again after a stop; RESUME=1 also resumes
# the run that was cut.
#
#   SMOKE=1        tiny budgets, sizes and episodes, data/numeric as the only stateless records, the
#                  verdicts recorded but not obeyed, everything under SKY_OUT (default
#                  runs/smoke-sleep2/; never runs/s6-sleep2*): a code check, no result
#   SKY_OUT        a prefix for everything this script writes (empty for the real run)
#   CPU=1          every sky-train call on the CPU (--cpu); the checks are always on the CPU
#   SKY_TRAIN      a sky-train binary to use as is, instead of building the release one
#   DRY_RUN=1      print the commands
#   AMENDED=1      the pilot may start on a verdict that needs an amendment (a C4 retry passed),
#                  once that amendment is committed in docs/s6-sleep2-prereg.md §13
#   PICO_BASE      runs/s6lr/softmax-adamw-x2/decide/last (the s6-pico of the adopted recipe)
#   SEQ            the training sequences (default runs/s6-sleep/seq, v1's)
#   OPT LR_SCALE   the recipe, as scripts/train-sleep.sh (default --no-muon, scale 2)
#   STEP_MEM_GB=3  GPU memory of one stateless step, as scripts/train-sleep.sh
set -euo pipefail
export DEVELOPER_DIR=${DEVELOPER_DIR:-/Applications/Xcode.app/Contents/Developer}
export CARGO_TARGET_DIR=${CARGO_TARGET_DIR:-$PWD/target/s3-train}
# The thermal brake, as scripts/train-sleep.sh.
export SKY_DUTY=${SKY_DUTY:-0.5}
export SKY_COOL=${SKY_COOL-45:15}
export SKY_BRAKE_FILE=${SKY_BRAKE_FILE-$PWD/runs/brake}
export SKY_MLX_CACHE_GB=${SKY_MLX_CACHE_GB:-2}
dry=${DRY_RUN:-0}
smoke=${SMOKE:-0}
T=${SKY_TRAIN:-$CARGO_TARGET_DIR/release/sky-train}
if [ "$smoke" = 1 ]; then
  SKY_OUT=${SKY_OUT:-runs/smoke-sleep2/}
fi
out=${SKY_OUT:-}
root=${out}runs/s6-sleep2
if [ "$smoke" = 1 ]; then
  case "$root" in
    runs/s6-sleep2*|./runs/s6-sleep2*|"$PWD"/runs/s6-sleep2*)
      echo "SMOKE=1 needs a prefix that is not the real run's (SKY_OUT=$out)" >&2
      exit 2
      ;;
  esac
  echo "== SMOKE: everything under $out; nothing of it is a result"
fi
phase_a_min=${PHASE_A_MIN:-8} phase_b_min=${PHASE_B_MIN:-12}
dev_eps=500 dev_seed=2000000000
c2_eps=2000 c2_rows=50 c3_batches=10 c3_batch=16 c4_steps=300 c4_val=256
n_recall=10000 n_rule=20000 n_game=5000
records=data/decisions,data/games,data/numeric weights=decisions=0.3,games=0.6,numeric=0.1
if [ "$smoke" = 1 ]; then
  phase_a_min=${PHASE_A_MIN:-0.3} phase_b_min=${PHASE_B_MIN:-0.3}
  dev_eps=3 c2_eps=8 c2_rows=4 c3_batches=2 c3_batch=2 c4_steps=6 c4_val=16
  n_recall=40 n_rule=40 n_game=20
  records=data/numeric weights=numeric=1
fi
seq=${SEQ:-runs/s6-sleep/seq}
[ "$smoke" = 1 ] && seq=${SEQ:-$root/seq}
games=(flappy rescue dodge doom)
pico_base=${PICO_BASE:-runs/s6lr/softmax-adamw-x2/decide/last}
read -r -a opt <<<"${OPT:---no-muon}"
lr_flag=--muon-lr
[[ " ${OPT:---no-muon} " == *" --no-muon "* ]] && lr_flag=--adam-lr
resume=()
[ -n "${RESUME:-}" ] && resume=(--resume)
cpu=()
[ "${CPU:-0}" = 1 ] && cpu=(--cpu)

x() {
  if [ "$dry" = 1 ]; then
    printf '+'
    printf ' %q' "$@"
    printf '\n'
  else
    "$@"
  fi
}

# rep <command> [args]: scripts/s6-sleep2-report.py, the rules of the pre-registration.
rep() { x python3 -I scripts/s6-sleep2-report.py "$@"; }

# rate <width> <size pilot scale>: v1's peak rate (scripts/train-sleep.sh): `rate 96 1` = 2e-3.
rate() { LC_ALL=C awk -v b="${LR_SCALE:-2.0}" -v c="$2" -v d="$1" 'BEGIN { printf "%.3g", 1e-3 * b * sqrt(96 / d) * c }'; }

# config <base checkpoint> <out.json> <form: frame|token|none> <name> [write rule]: the base's
# config with the v2 memory (shared qk, the value form of C2, the write rule of amendment 1's
# sweep) or without memory (the control).
config() {
  [ "$dry" = 1 ] && { echo "+ config $*"; return; }
  python3 -I - "$1/config.json" "$2" "$3" "$4" "${5:-average}" <<'EOF'
import json, sys
src, dst, form, name, write = sys.argv[1:]
c = json.load(open(src))
c["temperature"] = {}
c["name"] = name
for k in ("memory", "memory_qk", "memory_value", "memory_write"):
    c.pop(k, None)
if form != "none":
    c.update(memory=True, memory_qk="shared", memory_value=form)
    if write != "average":
        c["memory_write"] = write
json.dump(c, open(dst, "w"), indent=2)
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

# No pilot run trains during another training (docs/s6-sleep2-prereg.md §9); the CPU checks may.
no_other_training() {
  [ "$dry" = 1 ] && return 0
  # CPU=1 shares no GPU with the other training.
  [ "${CPU:-0}" = 1 ] && return 0
  local others
  others=$(pgrep -lf 'sky-train (s2-decide|s2-pretrain|decide|pretrain|cast-train|s5-train)' | grep -v -- ' --cpu' || true)
  if [ -n "$others" ]; then
    echo "another training runs on this machine; the pilot waits for it to end:" >&2
    echo "$others" >&2
    exit 1
  fi
}

data() {
  echo "== sequences: $seq"
  x mkdir -p "$seq"
  [ -f "$seq/recall.jsonl" ] || x "$T" s2-seqgen --games recall --n-seq "$n_recall" --k 64 --seed 61 --out "$seq"
  [ -f "$seq/rule110.jsonl" ] || x "$T" s2-seqgen --games rule110 --n-seq "$n_rule" --k 8 --seed 61 --out "$seq"
  local g
  for g in "${games[@]}"; do
    [ -f "$seq/$g.jsonl" ] || x "$T" s2-seqgen --games "$g" --n-seq "$n_game" --k 24 --seed 61 --out "$seq"
  done
}

# decide <out> <config> <init> <budget flag> <budget> [flags]: one s2-decide run with the shared
# settings of the pilot (v1's: stateless batches of 128, QAT from the first step, loop dropout 0.25,
# the rate of `rate 96 1`, seed 0) and the training log of M5.
decide() {
  local dir=$1 cfg=$2 init=$3 flag=$4 budget=$5
  shift 5
  if [ "$dry" != 1 ] && [ -n "$(steps_of "$dir")" ]; then
    echo "== $dir: done ($(steps_of "$dir") steps), skipped"
    return
  fi
  [ -n "${RESUME:-}" ] || x rm -rf "$dir"
  echo "== $dir: $flag $budget $*"
  x nice -n "${NICE:-0}" "$T" ${cpu[@]+"${cpu[@]}"} s2-decide --config "$cfg" --init "$init" --tokenizer "$pico_base/tokenizer.json" \
    --records "$records" --weights "$weights" --batch 128 "$flag" "$budget" --seed 0 \
    --qat-frac 1.0 --loop-dropout 0.25 --length-pool 16 --pad-to 32 --out "$dir" \
    --step-mem-gb "${STEP_MEM_GB:-3}" \
    "$lr_flag" "$(rate 96 1)" ${opt[@]+"${opt[@]}"} ${resume[@]+"${resume[@]}"} --seq-log "$@"
}

# §2 M4. Phase A: sequence steps at 0.9, recall 0.85 and rule110 0.15, no sleep (N = 1).
phase_a=(--sleep-records "$seq/recall.jsonl,$seq/rule110.jsonl" --sleep-weights 'recall=0.85,rule110=0.15'
  --sleep-batch 16 --seq-frac 0.9 --sleep-passes 1)
# Phase B: v1's mix (scripts/train-sleep.sh), N drawn in {1, 2, 3, 4}, truncated.
phase_b=(--sleep-records "$seq/recall.jsonl,$seq/rule110.jsonl,$seq/flappy.jsonl,$seq/rescue.jsonl,$seq/dodge.jsonl,$seq/doom.jsonl"
  --sleep-weights 'recall=0.40,rule110=0.12,flappy=0.12,rescue=0.12,dodge=0.12,doom=0.12'
  --sleep-batch 16 --seq-frac 0.5 --sleep-passes '1,2,3,4')

# passed <check json>: true when the check passed (always "true" in a dry run).
passed() {
  [ "$dry" = 1 ] && { echo true; return; }
  json_get "$1" pass
}

# failed <check> <what follows>: the pre-registration's next step; the smoke goes on.
failed() {
  echo "== $1 FAILED. $2" >&2
  [ "$smoke" = 1 ] && { echo "== SMOKE: going on to check the code"; return 0; }
  exit 1
}

# c4 <dir> [flags]: C4 (§3): 300 steps of phase A's mix with recall as the only sequence game,
# batch 8 sequences, rate 2e-3 with mem.* at x10, N = 1, QAT as in the pilot, on the CPU; twice
# from seed 0, the second with the read held at zero.
c4() {
  local dir=$1
  shift
  local -a c4_flags=(--sleep-records "$seq/recall.jsonl" --sleep-weights recall=1 --sleep-batch 8
    --seq-frac 0.9 --sleep-passes 1 --no-compile --val-max "$c4_val")
  local cfg=$root/checks/sleep2.json
  if [ "$dry" != 1 ] && [ -f "$dir/c4.json" ]; then
    echo "== $dir: done"
    return
  fi
  local saved=${CPU:-0}
  CPU=1 cpu=(--cpu)
  NICE=19 decide "$dir/mem" "$cfg" "$pico_base" --steps "$c4_steps" "${c4_flags[@]}" "$@"
  NICE=19 decide "$dir/zero" "$cfg" "$pico_base" --steps "$c4_steps" "${c4_flags[@]}" "$@" --memory-read-off
  CPU=$saved cpu=()
  [ "$saved" = 1 ] && cpu=(--cpu)
  rep c4 "$dir/mem" "$dir/zero" "$dir/c4.json" || true
}

checks() {
  local c=$root/checks form
  x mkdir -p "$c"
  local -a sizes=(--episodes "$c2_eps" --rows "$c2_rows" --batches "$c3_batches" --batch "$c3_batch")
  # C1 first, alone: the first failing check stops the script.
  [ "$dry" != 1 ] && [ -f "$c/c1.json" ] || x nice -n 19 "$T" --cpu s2-memcheck --base "$pico_base" --seq "$seq/recall.jsonl" --out "$c" --c1-only
  # Amendment 1: C1 failed with v1's write rule; the write rule is chosen by the sweep on C1's
  # synthetic states (W1 delta, W2 exact, W3 centred, in that order), before C2.
  local write=average
  if [ "$dry" = 1 ] || [ "$(passed "$c/c1.json")" != true ]; then
    [ "$dry" != 1 ] && [ -f "$c/sweep.json" ] || x nice -n 19 "$T" --cpu s2-memcheck --base "$pico_base" --seq "$seq/recall.jsonl" --out "$c" --sweep
    if [ "$dry" = 1 ]; then
      write=RULE
    else
      write=$(python3 -I -c 'import json,sys; a=json.load(open(sys.argv[1]))["adopted"]; print(a["rule"] if a else "")' "$c/sweep.json")
    fi
    if [ -z "$write" ]; then
      # Amendment 2: D1 and D2; when C1's raw decoding is unsound, the sweep once more with C1
      # read contrastively over 20 draws (its adoption replaces the first sweep's).
      [ "$dry" != 1 ] && [ -f "$c/diagnostic.json" ] || x nice -n 19 "$T" --cpu s2-memcheck --base "$pico_base" --seq "$seq/recall.jsonl" --out "$c" --diagnose
      if [ "$dry" != 1 ]; then
        write=$(python3 -I -c 'import json,sys; a=json.load(open(sys.argv[1])).get("adopted"); print(a["rule"] if a else "")' "$c/diagnostic.json")
      fi
    fi
    if [ -z "$write" ]; then
      failed sweep "No write rule is right at d = 1, 4 and 8 with at most 4 passes: v2 stops here (amendments 1 and 2)."
      write="delta" # SMOKE only
    fi
  fi
  [ "$dry" != 1 ] && [ -f "$c/c3.json" ] || x nice -n 19 "$T" --cpu s2-memcheck --base "$pico_base" --seq "$seq/recall.jsonl" --out "$c" --write "$write" "${sizes[@]}"
  [ "$(passed "$c/c2.json")" = true ] || failed C2 "Both v2 forms are under 0.40: no GPU run; the next step is the per-token delta rule or a frame cache, by a new pre-registration (§11)."
  [ "$(passed "$c/c3.json")" = true ] || failed C3 "No GPU training starts before C1 to C4 pass (§3); §11 names no next step for C3, which an amendment or a new pre-registration settles."
  form=FORM
  [ "$dry" = 1 ] || form=$(json_get "$c/c2.json" adopted)
  [ -n "$form" ] || form=frame # SMOKE only: C2 adopted nothing
  config "$pico_base" "$c/sleep2.json" "$form" s6-sleep2-pico "$write"
  c4 "$c/c4" --mem-lr-mult 10
  if [ "$(passed "$c/c4/c4.json")" != true ]; then
    # §3: the only retries allowed, after C2 passed: mem.* at x30, and float memory projections.
    echo "== C4 failed: its two retries (docs/s6-sleep2-prereg.md §3)"
    c4 "$c/c4-x30" --mem-lr-mult 30
    c4 "$c/c4-float" --mem-lr-mult 10 --memory-float
  fi
  if [ "$smoke" = 1 ]; then
    rep checks "$root" || echo "== SMOKE: the checks do not pass, as expected at these sizes"
  else
    rep checks "$root"
  fi
}

pilot() {
  local p=$root/pilot v=$root/checks/verdict.json form=FORM write=RULE mult=10 a_steps b_steps
  local -a mem_flags=()
  no_other_training
  if [ "$dry" != 1 ]; then
    [ -f "$v" ] || { echo "no $v: run the checks first (docs/s6-sleep2-prereg.md §3)" >&2; exit 1; }
    if [ "$(json_get "$v" pass)" != true ]; then
      if [ "$smoke" = 1 ]; then
        echo "== SMOKE: the checks did not pass; going on to check the code"
      else
        echo "the checks did not pass ($v): no GPU run (docs/s6-sleep2-prereg.md §3, §11)" >&2
        exit 1
      fi
    fi
    if [ "$(json_get "$v" needs_amendment)" = true ] && [ "${AMENDED:-0}" != 1 ] && [ "$smoke" != 1 ]; then
      echo "a C4 retry passed: its setting replaces M3 or O5 by an amendment of §13 first; then AMENDED=1" >&2
      exit 1
    fi
    form=$(json_get "$v" form)
    write=$(json_get "$root/checks/c2.json" write)
    [ -n "$form" ] || form=frame # SMOKE only
    mult=$(json_get "$v" mem_lr_mult)
    [ "$(json_get "$v" mem_float)" = true ] && mem_flags+=(--memory-float)
  fi
  mem_flags+=(--mem-lr-mult "$mult")
  x mkdir -p "$p"
  config "$pico_base" "$p/sleep2.json" "$form" s6-sleep2-pico "$write"
  config "$pico_base" "$p/control.json" none s6-pico-control2
  # M, phase A, then the in-sample stop (§5.1).
  decide "$p/m-a" "$p/sleep2.json" "$pico_base" --minutes "$phase_a_min" "${phase_a[@]}" "${mem_flags[@]}"
  local st=0
  rep stop "$root" || st=$?
  if [ "$st" = 3 ]; then
    if [ "$smoke" != 1 ]; then
      echo "== the pilot stops after phase A: phase B and arm C do not run (§5.1)"
      rep pilot "$root"
      exit 0
    fi
    echo "== SMOKE: the stop would end the pilot here; going on to check the code"
  elif [ "$st" != 0 ]; then
    exit "$st"
  fi
  # M, phase B: from phase A's last, a fresh optimiser and schedule.
  decide "$p/m-b" "$p/sleep2.json" "$p/m-a/last" --minutes "$phase_b_min" "${phase_b[@]}" "${mem_flags[@]}"
  a_steps=$(steps_of "$p/m-a") b_steps=$(steps_of "$p/m-b")
  [ "$dry" = 1 ] && a_steps=STEPS_A b_steps=STEPS_B
  [ -n "$a_steps" ] && [ -n "$b_steps" ] || { echo "M did not end" >&2; exit 1; }
  # C: the same steps as M, phase by phase, on the same batches.
  decide "$p/c-a" "$p/control.json" "$pico_base" --steps "$a_steps" "${phase_a[@]}"
  decide "$p/c-b" "$p/control.json" "$p/c-a/last" --steps "$b_steps" "${phase_b[@]}"
  # The dev2 readings (§5.2): ternary weights, last exit, closed loop, on the CPU.
  local g
  x mkdir -p "$p/m/dev2" "$p/c/dev2"
  for g in recall rule110; do
    [ "$dry" != 1 ] && [ -f "$p/m/dev2/$g.json" ] || x "$T" s2-probe "$p/m-b/last" --game "$g" --from-seed "$dev_seed" --episodes "$dev_eps" --passes 1,2,4 --json "$p/m/dev2/$g.json"
    [ "$dry" != 1 ] && [ -f "$p/c/dev2/$g.json" ] || x "$T" s2-probe "$p/c-b/last" --game "$g" --from-seed "$dev_seed" --episodes "$dev_eps" --passes 1 --json "$p/c/dev2/$g.json"
  done
  [ "$dry" != 1 ] && [ -f "$p/m/dev2/recall-off.json" ] || x "$T" s2-probe "$p/m-b/last" --game recall --from-seed "$dev_seed" --episodes "$dev_eps" --passes 1 --memory-off --json "$p/m/dev2/recall-off.json"
}

report() {
  if [ "$smoke" = 1 ]; then
    rep pilot "$root" smoke
  else
    rep pilot "$root"
  fi
}

[ "$dry" = 1 ] || [ -n "${SKY_TRAIN:-}" ] || cargo build -q --release -p sky-train
steps=("$@")
[ ${#steps[@]} -eq 0 ] && steps=(data checks pilot report)
for s in "${steps[@]}"; do
  case "$s" in
    data) data ;;
    checks) checks ;;
    pilot) pilot ;;
    report) report ;;
    *) echo "unknown step $s (data, checks, pilot, report)" >&2; exit 2 ;;
  esac
done
