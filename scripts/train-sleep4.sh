#!/usr/bin/env bash
# s6-sleep v4: the check K3 (a frozen base) and the frozen route that follows it, from
# docs/s6-sleep4-prereg.md (v4 = docs/s6-sleep3-prereg.md, v3, trained to the end of its learning
# curve, with checks at the nano size first).
#
#   scripts/train-sleep4.sh [k3] [nano] [dev] [export] [eval] [gate]
#                                                  # no step: k3 only (nothing else is automatic)
#
#   k3      §3.2, on the GPU: v3's nano phase A (§3.1: the base runs/s6-nano-g3/decide/last, v3's
#           config, batch 128, AdamW at 0.00163, mem.* at x10 with the gates held, the auxiliary
#           loss, --sleep-chunk 2) with the base frozen (--freeze-base), --steps 1310 and
#           --memory-off-eval 4, seed 0 then seed 1, one run at a time; read by
#           scripts/s6-sleep4-report.py k3 -> <root>/stage1/k3.json and <root>/stage1.json (the
#           rules of §3.6 and, on the frozen route, the cap of §4.2). K3_SEED=0 or K3_SEED=1 runs
#           that seed alone (seed 1 waits for seed 0); the reading waits for both. No rerun once
#           read (§10)
#   nano    refused unless stage1.json takes the frozen route and is committed. §4: the memory
#           arm, seeds 0, 1, 2, with the base frozen: phase A ended by the rule of §4.2
#           (--stop-recall-acc 0.98 --stop-recall-window 100) or at the cap (--steps from
#           stage1.json); the three phase A runs, then v2's stop (-> <root>/nano/phase-a.json; it
#           ends v4 in-sample if it holds for a seed), then phase B, 2,000 steps (§4.3). No control
#           run: on the frozen route the control is the base file (§4.1)
#   dev     refused before every stage-2 run has ended and stage1.json is committed. §5 on dev3:
#           recall and rule110 at N = 1, 2, 4 for each memory seed, the base once, the memory
#           zeroed at N = 1 on recall (its parity with the base, exact); the choice of N
#           -> <root>/n.json
#   export  the base copied to <root>/base/last (and checked against it, bit for bit), the
#           calibration with a zero memory and the ternary export of the three memory seeds and
#           of the base; the parity of every exported file and checkpoint with the base
#           (sky-train s2-same-base -> <root>/parity/)
#   eval    refused before every stage-2 run has ended and N is chosen; the test seeds, READ ONCE
#           (§5): recall and rule110 at N = 1, 2, 4, the memory zeroed, the four games at the
#           chosen N, typed_decisions_bench; the base once; then the parity checks of §4.1 and H0
#           to H3 -> <root>/REPORT.md, <root>/verdict.json
#   gate    §6: the s6-nano gate on the exported seed-0 file, H0 to H3, Z (the parity checks),
#           sky-infer's parity on the golden memory sequences natively and in WASM -> gate.json
#
# K1, K2, the confirmation, K5 and the preserved route (§3.3 to §3.5, §4 when K3 fails) are not in
# this script: they need the code of §11 items 2 (as K2 (ii) reads it), 3 and 6, which is written
# only if K3 fails. If K3 fails, this script stops after its reading.
#
# Every reading and verdict is a file under <root>; the script commits nothing (the lead does,
# before the next stage: §3.6). A run that ended is skipped, so the script can be started again
# after a stop; a run cut by a technical failure restarts from scratch with the same seed (§3.1).
#
#   SMOKE=1        on the CPU (--cpu), under SKY_OUT (default runs/smoke-sleep4/; never
#                  runs/s6-sleep4*): its own inputs (a byte tokenizer, numeric records, a few
#                  sequences of 16 frames, a pico base of three steps), a few steps of each run,
#                  its own seeds (1000, 3000, 9000: none of v4's), the verdicts recorded but not
#                  obeyed; the gate is not part of it. A code check, no result
#   SKY_OUT        a prefix for everything this script writes (empty for the real run)
#   DRY_RUN=1      print the commands, run nothing
#   SKY_TRAIN      a sky-train binary to use as is, instead of building the release one
#   NANO_BASE      runs/s6-nano-g3/decide/last (the base of §3.1)
#   V3_CONFIG      v3's nano config (default runs/s6-sleep3/nano/sleep3.json); when it is missing,
#                  the config is built from the base as scripts/train-sleep3.sh built it
#   V3_CMD         v3's nano phase A command (default runs/s6-sleep3/nano/sleep-s0/a/cmd.txt): when
#                  it exists, each K3 command is checked against it (the same recipe but for the
#                  flags §3.1 and §3.2 change)
#   SEQ            the training sequences (default runs/s6-sleep/seq, v1's)
#   DATA           the folder of the stateless records (default data)
#   SKY_DUTY SKY_COOL SKY_BRAKE_FILE   the thermal brake (scripts/auto-brake.sh); SKY_BRAKE_FILE
#                  must be set for a GPU run
set -euo pipefail
export DEVELOPER_DIR=${DEVELOPER_DIR:-/Applications/Xcode.app/Contents/Developer}
export CARGO_TARGET_DIR=${CARGO_TARGET_DIR:-$PWD/target/s3-train}
export SKY_DUTY=${SKY_DUTY:-0.5}
export SKY_COOL=${SKY_COOL-45:15}
# v3's amendment 3, kept by v4 (§2): the MLX cache at 1 GB for every nano run.
export SKY_MLX_CACHE_GB=1
dry=${DRY_RUN:-0}
smoke=${SMOKE:-0}
B=$CARGO_TARGET_DIR/release
T=${SKY_TRAIN:-$B/sky-train}
if [ "$smoke" = 1 ]; then
  SKY_OUT=${SKY_OUT:-runs/smoke-sleep4/}
fi
out=${SKY_OUT:-}
root=${out}runs/s6-sleep4
models=${out}models/s6-sleep4-nano
if [ "$smoke" = 1 ]; then
  case "$root" in
    runs/s6-sleep4* | ./runs/s6-sleep4* | "$PWD"/runs/s6-sleep4*)
      echo "SMOKE=1 needs a prefix that is not the real run's (SKY_OUT=$out)" >&2
      exit 2
      ;;
  esac
  echo "== SMOKE: everything under $out, on the CPU; nothing of it is a result"
fi
smoke_arg=()
[ "$smoke" = 1 ] && smoke_arg=(smoke)

# Budgets, sizes and seeds (§3.1, §4.2, §4.3, §5).
k3_steps=1310
phase_b_steps=2000
stop_acc=0.98 stop_window=100
batch=128 sleep_batch=16
dev_eps=500 dev_seed=3000000000
test_eps=1000 test_seed=6000000000 arena_eps=200 arena_seed=6000
data=${DATA:-data}
seq=${SEQ:-runs/s6-sleep/seq}
nano_base=${NANO_BASE:-runs/s6-nano-g3/decide/last}
# Stateless records and weights as v3's nano runs (§3.1).
nano_records=$data/decisions,$data/distill,$data/games,$data/numeric
nano_weights=decisions=0.25,distill=0.30,games=0.35,numeric=0.10
typed=$data/decisions/typed_decisions_bench.jsonl
cpu=()
small=()
if [ "$smoke" = 1 ]; then
  data=${out}data seq=${out}seq nano_base=${out}base/decide/last
  nano_records=$data/numeric nano_weights=numeric=1 typed=$data/numeric/numeric.jsonl
  k3_steps=4 phase_b_steps=3 batch=16 sleep_batch=4
  # The rule of §4.2 at a bar the smoke reaches (its plumbing, not its bar), and a cap of its own:
  # a smoke's K3 has no S95.
  stop_acc=0.01 stop_window=2 smoke_cap=6
  # A smoke reads no dev3, test or arena seed of v4, nor any seed v4 refuses to read for a
  # decision but the smoke's own (§5).
  dev_seed=1000 dev_eps=2 test_seed=3000 test_eps=2 arena_seed=9000 arena_eps=1
  cpu=(--cpu)
  small=(--no-compile --val-max 16)
fi
games=(flappy rescue dodge doom)
v3_config=${V3_CONFIG:-runs/s6-sleep3/nano/sleep3.json}
v3_cmd=${V3_CMD:-runs/s6-sleep3/nano/sleep-s0/a/cmd.txt}
cfg=$root/sleep4.json

x() {
  if [ "$dry" = 1 ]; then
    printf '+'
    printf ' %q' "$@"
    printf '\n'
  else
    "$@"
  fi
}

# rep <command> [args]: scripts/s6-sleep4-report.py, the rules of the pre-registration.
rep() { x python3 -I scripts/s6-sleep4-report.py "$@"; }

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

# No stage-1 check or stage-2 run trains during another training, nor on the CPU (§3.7, §9); the
# brake file must be set, so that scripts/auto-brake.sh can rest the run. A smoke, on the CPU,
# needs neither.
gpu_run_allowed() {
  [ "$smoke" = 1 ] && return 0
  if [ -z "${SKY_BRAKE_FILE:-}" ]; then
    if [ "$dry" = 1 ]; then
      echo "== DRY_RUN: SKY_BRAKE_FILE is unset; a real run would be refused here"
    else
      echo "SKY_BRAKE_FILE is unset: a GPU run of v4 runs under the machine's brake (scripts/auto-brake.sh)" >&2
      exit 1
    fi
  fi
  [ "$dry" = 1 ] && return 0
  local others
  others=$(pgrep -lf 'sky-train (s2-decide|s2-pretrain|decide|pretrain|cast-train|s5-train)' | grep -v -- ' --cpu' || true)
  if [ -n "$others" ]; then
    echo "another training runs on this machine; this step waits for it to end (§3.7: one run at a time):" >&2
    echo "$others" >&2
    exit 1
  fi
}

# committed <file>: the verdict a stage waits for is written and committed (§3.6: each reading and
# verdict is committed before the next check starts). The script commits nothing itself.
committed() {
  [ "$dry" = 1 ] && return 0
  [ -f "$1" ] || { echo "no $1: the stage before has not written its verdict" >&2; exit 1; }
  [ "$smoke" = 1 ] && return 0
  if ! git ls-files --error-unmatch "$1" >/dev/null 2>&1 || ! git diff --quiet HEAD -- "$1"; then
    echo "$1 is not committed as it is: commit it (git add -f) before the next stage" >&2
    exit 1
  fi
}

# config: v3's nano config (§3.1: v3's memory with memory_aux), named for v4, kept next to the runs.
# Built from the base as scripts/train-sleep3.sh built it when v3's file is not here; checked equal
# to it, but for the name, when it is.
config() {
  [ "$dry" = 1 ] && { echo "+ config $cfg (from $v3_config, or built from $nano_base/config.json)"; return; }
  mkdir -p "$root"
  python3 -I - "$nano_base/config.json" "$v3_config" "$cfg" <<'EOF'
import json, os, sys
src, v3, dst = sys.argv[1:]
c = json.load(open(src))
c["temperature"] = {}
for k in ("memory", "memory_qk", "memory_value", "memory_write", "memory_aux"):
    c.pop(k, None)
c.update(memory=True, memory_qk="shared", memory_value="frame", memory_aux=True)
if os.path.exists(v3):
    a = json.load(open(v3))
    a.pop("name", None)
    b = dict(c)
    b.pop("name", None)
    if a != b:
        keys = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
        sys.exit(f"{v3} is not v3's memory on this base: {keys}")
    print(f"config: v3's {v3}, named s6-sleep4-nano")
else:
    print(f"config: built from {src} as scripts/train-sleep3.sh built v3's ({v3} is not here)")
c["name"] = "s6-sleep4-nano"
json.dump(c, open(dst, "w"), indent=2)
EOF
}

# decide <out> <init> <seed> <steps> [flags]: one s2-decide run with v3's nano phase A settings
# (§3.1: stateless batches of 128, QAT from the first step, loop dropout 0.25, AdamW at 0.00163,
# the training log of v2's M5). With check_v3=1 its command is checked against v3's first.
check_v3=0
decide() {
  local dir=$1 init=$2 seed=$3 steps=$4
  shift 4
  if [ "$dry" != 1 ] && [ -n "$(steps_of "$dir")" ]; then
    echo "== $dir: done ($(steps_of "$dir") steps), skipped"
    return
  fi
  # A technical failure restarts the run from scratch, with the same seed (§3.1).
  x rm -rf "$dir"
  x mkdir -p "$dir"
  echo "== $dir: --steps $steps, seed $seed $*"
  local -a cmd=(nice -n 0 "$T" ${cpu[@]+"${cpu[@]}"} s2-decide --config "$cfg" --init "$init" --tokenizer "$nano_base/tokenizer.json"
    --records "$nano_records" --weights "$nano_weights" --batch "$batch" --steps "$steps" --seed "$seed"
    --qat-frac 1.0 --loop-dropout 0.25 --length-pool 16 --pad-to 32 --out "$dir"
    --step-mem-gb 3 --adam-lr 0.00163 --no-muon --seq-log ${small[@]+"${small[@]}"} "$@")
  # The command of a run, kept next to its log (a file of the run, not a result).
  [ "$dry" = 1 ] || printf '%q ' "${cmd[@]}" >"$dir/cmd.txt"
  if [ "$check_v3" = 1 ] && [ "$smoke" != 1 ] && [ "$dry" != 1 ]; then
    if [ -f "$v3_cmd" ]; then
      rep recipe "$v3_cmd" "$dir/cmd.txt"
    else
      echo "== no $v3_cmd: the command is not compared with v3's (V3_CMD)"
    fi
  fi
  x "${cmd[@]}"
}

# probe <checkpoint> <game> <first seed> <episodes> <passes> <out.json> [flags]: s2-probe (on the
# CPU, at nice 19; ternary weights, last exit, closed loop), skipped when the reading exists.
probe() {
  local ck=$1 game=$2 from=$3 n=$4 passes=$5 json=$6
  shift 6
  if [ "$dry" != 1 ] && [ -f "$json" ]; then
    return
  fi
  x mkdir -p "$(dirname "$json")"
  x nice -n 19 "$T" s2-probe "$ck" --game "$game" --from-seed "$from" --episodes "$n" --passes "$passes" --json "$json" "$@"
}

# v3's phase A (v2 §2 M4): sequence steps at 0.9, recall 0.85 and rule110 0.15, N = 1; and v3's
# T1 and T2 with v2's M3 (mem.* at x10, the gates held, the auxiliary loss at weight 1), in parts
# of 2 sequences (amendment 3).
phase_a=(--sleep-records "$seq/recall.jsonl,$seq/rule110.jsonl" --sleep-weights 'recall=0.85,rule110=0.15'
  --sleep-batch "$sleep_batch" --seq-frac 0.9 --sleep-passes 1)
phase_b=(--sleep-records "$seq/recall.jsonl,$seq/rule110.jsonl,$seq/flappy.jsonl,$seq/rescue.jsonl,$seq/dodge.jsonl,$seq/doom.jsonl"
  --sleep-weights 'recall=0.40,rule110=0.12,flappy=0.12,rescue=0.12,dodge=0.12,doom=0.12'
  --sleep-batch "$sleep_batch" --seq-frac 0.5 --sleep-passes '1,2,3,4')
t12=(--mem-lr-mult 10 --mem-gate-lr-mult 0 --memory-aux-weight 1 --sleep-chunk 2)
# §3.2: the base frozen, in K3 and in every run of the frozen route.
frozen=(--freeze-base)

# ----------------------------------------------------------------------------- smoke inputs

# A smoke's own inputs, under SKY_OUT: the byte-level tokenizer of the golden models (its port of
# s2::golden::byte_tokenizer_json), programmatic numeric records, sequences of the six games and a
# pico base trained three steps from scratch. No model writes any of it.
smoke_inputs() {
  [ "$smoke" = 1 ] || return 0
  [ -f "$nano_base/state.json" ] && return 0
  x mkdir -p "$data/numeric" "$seq" "${out}base"
  if [ "$dry" != 1 ]; then
    python3 -I - "${out}base/tokenizer.json" <<'EOF'
import json, sys
bs = list(range(ord("!"), ord("~") + 1)) + list(range(0xA1, 0xAD)) + list(range(0xAE, 0x100))
cs, n = bs[:], 0
for b in range(256):
    if b not in bs:
        bs.append(b)
        cs.append(256 + n)
        n += 1
chars = {b: chr(c) for b, c in zip(bs, cs)}
specials = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "[NUM]"]
vocab = {s: i for i, s in enumerate(specials)}
for b in range(256):
    vocab[chars[b]] = 6 + b
added = [{"id": i, "content": s, "single_word": False, "lstrip": s == "[NUM]", "rstrip": False,
          "normalized": False, "special": True} for i, s in enumerate(specials)]
level = {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": True, "use_regex": True}
tok = {"version": "1.0", "truncation": None, "padding": None, "added_tokens": added, "normalizer": None,
       "pre_tokenizer": level, "post_processor": None, "decoder": level,
       "model": {"type": "BPE", "dropout": None, "unk_token": None, "continuing_subword_prefix": None,
                 "end_of_word_suffix": None, "fuse_unk": False, "byte_fallback": False,
                 "ignore_merges": False, "vocab": vocab, "merges": []}}
json.dump(tok, open(sys.argv[1], "w"))
EOF
  fi
  x nice -n 19 "$T" s2-numeric --n 400 --seed 9 --out "$data/numeric/numeric.jsonl"
  # Frames per sequence as v1's (scripts/train-sleep.sh), shorter: a rule110 episode holds fewer
  # than 16 decisions, so its sequences are of 8, as v1's.
  x nice -n 19 "$T" s2-seqgen --games recall --n-seq 12 --k 16 --seed 9 --out "$seq"
  x nice -n 19 "$T" s2-seqgen --games rule110,flappy,rescue,dodge,doom --n-seq 12 --k 8 --seed 9 --out "$seq"
  x nice -n 19 "$T" --cpu s2-decide --config configs/s6-pico.json --tokenizer "${out}base/tokenizer.json" \
    --records "$data/numeric" --weights numeric=1 --batch 16 --steps 3 --qat-frac 1.0 --no-compile \
    --val-max 16 --out "${out}base/decide"
}

# ----------------------------------------------------------------------------- K3

k3() {
  local st=$root/stage1 seed seeds
  gpu_run_allowed
  if [ "$dry" != 1 ] && [ -f "$root/stage1.json" ]; then
    echo "== K3 read: route $(json_get "$root/stage1.json" route) (no rerun, §10)"
    return
  fi
  smoke_inputs
  config
  x mkdir -p "$st"
  seeds=${K3_SEED:-0 1}
  case "$seeds" in
    "0 1" | 0) ;;
    1)
      if [ "$dry" != 1 ] && [ -z "$(steps_of "$st/k3-s0")" ]; then
        echo "K3 runs seed 0 first: $st/k3-s0 has not ended" >&2
        exit 1
      fi
      ;;
    *) echo "K3_SEED is 0 or 1, not $seeds" >&2; exit 2 ;;
  esac
  # One seed after the other, never in parallel (§3.7).
  check_v3=1
  for seed in $seeds; do
    decide "$st/k3-s$seed" "$nano_base" "$seed" "$k3_steps" "${phase_a[@]}" "${t12[@]}" "${frozen[@]}" --memory-off-eval 4
  done
  check_v3=0
  if [ "$dry" != 1 ] && { [ -z "$(steps_of "$st/k3-s0")" ] || [ -z "$(steps_of "$st/k3-s1")" ]; }; then
    echo "== K3: the reading waits for both seeds"
    return
  fi
  local rc=0
  rep k3 "$root" ${smoke_arg[@]+"${smoke_arg[@]}"} || rc=$?
  if [ "$rc" != 0 ]; then
    if [ "$smoke" = 1 ]; then
      echo "== SMOKE: K3 does not pass at these sizes, as expected; going on to check the code"
    else
      echo "== K3 FAILED: the frozen route is closed for v4 (§3.6, rule 2). K1, K2, the confirmation and K5 need code that is not written yet (§11 items 2, 3, 6); this script stops here" >&2
      exit 1
    fi
  fi
}

# ----------------------------------------------------------------------------- stage 2, the frozen route

nano() {
  local n=$root/nano seed cap
  gpu_run_allowed
  committed "$root/stage1.json"
  smoke_inputs
  config
  if [ "$dry" = 1 ]; then
    cap=CAP_STEPS
    rep nano-ready "$root"
  elif [ "$smoke" = 1 ]; then
    rep nano-ready "$root" smoke
    cap=$smoke_cap
  else
    cap=$(python3 -I scripts/s6-sleep4-report.py nano-ready "$root")
  fi
  # Phase A of the three memory seeds: ended by the rule of §4.2, or at the cap.
  for seed in 0 1 2; do
    decide "$n/sleep-s$seed/a" "$nano_base" "$seed" "$cap" "${phase_a[@]}" "${t12[@]}" "${frozen[@]}" \
      --stop-recall-acc "$stop_acc" --stop-recall-window "$stop_window"
  done
  # v2's stop, for every memory seed (§4.2, C19): if it holds for one, no phase B runs.
  local rc=0
  rep phase-a "$root" ${smoke_arg[@]+"${smoke_arg[@]}"} || rc=$?
  if [ "$rc" = 3 ]; then
    echo "== v2's stop holds after phase A: no phase B runs, v4 reports a failure in-sample (§4.2)"
    exit 0
  elif [ "$rc" != 0 ]; then
    exit "$rc"
  fi
  # Phase B: from phase A's last, a fresh optimiser and schedule, 2,000 steps (§4.3).
  for seed in 0 1 2; do
    decide "$n/sleep-s$seed/b" "$n/sleep-s$seed/a/last" "$seed" "$phase_b_steps" "${phase_b[@]}" "${t12[@]}" "${frozen[@]}"
  done
}

# Every stage-2 run ended (and phase A did not stop v4).
stage2_ended() {
  [ "$dry" = 1 ] && return 0
  local seed ph
  for seed in 0 1 2; do
    for ph in a b; do
      [ -n "$(steps_of "$root/nano/sleep-s$seed/$ph")" ] || { echo "$root/nano/sleep-s$seed/$ph has not ended" >&2; exit 1; }
    done
  done
  if [ "$(json_get "$root/nano/phase-a.json" stop)" = true ] && [ "$smoke" != 1 ]; then
    echo "v2's stop held after phase A: no reading" >&2
    exit 1
  fi
}

# §5: dev3, for the choices. The base once; the memory zeroed against it (parity, exact).
dev() {
  local d=$root/dev3 seed g
  committed "$root/stage1.json"
  stage2_ended
  for seed in 0 1 2; do
    for g in recall rule110; do
      probe "$root/nano/sleep-s$seed/b/last" "$g" "$dev_seed" "$dev_eps" 1,2,4 "$d/sleep-s$seed/$g.json"
    done
    probe "$root/nano/sleep-s$seed/b/last" recall "$dev_seed" "$dev_eps" 1 "$d/sleep-s$seed/recall-off.json" --memory-off
  done
  for g in recall rule110; do
    probe "$nano_base" "$g" "$dev_seed" "$dev_eps" 1 "$d/base/$g.json"
  done
  rep choose "$root" ${smoke_arg[@]+"${smoke_arg[@]}"}
}

# The calibration with a zero memory and the ternary export (v1 §6), of the memory seeds and of the
# base, whose copy is calibrated in place of the base's own folder; every file and checkpoint
# against the base, tensor for tensor (§4.1; a difference stops the script).
export_() {
  local p=$root/parity seed r
  stage2_ended
  x mkdir -p "$models" "$p"
  if [ "$dry" = 1 ] || [ ! -d "$root/base/last" ]; then
    x mkdir -p "$root/base"
    x cp -R "$nano_base" "$root/base/last"
  fi
  x nice -n 19 "$T" s2-same-base --model "$root/base/last" --base "$nano_base" --json "$p/base-copy.json"
  for seed in 0 1 2; do
    r=$root/nano/sleep-s$seed/b/last
    x nice -n 19 "$T" s2-same-base --model "$r" --base "$nano_base" --json "$p/sleep-s$seed-checkpoint.json"
    if [ "$dry" = 1 ] || [ ! -f "$models/sleep-s$seed.gguf" ]; then
      x nice -n 19 "$T" s2-calibrate --run "$r" --records "$nano_records"
      x nice -n 19 "$T" s2-export --run "$r" --format ternary --out "$models/sleep-s$seed.gguf"
    fi
  done
  if [ "$dry" = 1 ] || [ ! -f "$models/base.gguf" ]; then
    x nice -n 19 "$T" s2-calibrate --run "$root/base/last" --records "$nano_records"
    x nice -n 19 "$T" s2-export --run "$root/base/last" --format ternary --out "$models/base.gguf"
  fi
  for seed in 0 1 2; do
    x nice -n 19 "$T" s2-same-base --model "$models/sleep-s$seed.gguf" --base "$models/base.gguf" --json "$p/sleep-s$seed-gguf.json"
  done
}

# §5: the test seeds, read once, after every stage-2 run has ended and N is chosen.
eval_() {
  local t=$root/test seed g chosen=1 r base=$root/base/last
  if [ "$dry" != 1 ]; then
    if [ "$smoke" = 1 ]; then
      rep test-ready "$root" || echo "== SMOKE: the test reading would be refused here; going on to check the code"
      rm -f "$t/done"
    else
      rep test-ready "$root"
    fi
    chosen=$(json_get "$root/n.json" n)
    [ -d "$base" ] || { echo "no $base: run the export first" >&2; exit 1; }
  fi
  x mkdir -p "$t"
  for seed in 0 1 2; do
    r=$root/nano/sleep-s$seed/b/last
    for g in recall rule110; do
      probe "$r" "$g" "$test_seed" "$test_eps" 1,2,4 "$t/sleep-s$seed/$g.json"
    done
    probe "$r" recall "$test_seed" "$test_eps" 1 "$t/sleep-s$seed/recall-off.json" --memory-off
    for g in "${games[@]}"; do
      probe "$r" "$g" "$arena_seed" "$arena_eps" "$chosen" "$t/sleep-s$seed/$g.json"
    done
    if [ "$dry" = 1 ] || [ ! -f "$t/sleep-s$seed/typed.md" ]; then
      x nice -n 19 "$T" s2-eval --run "$r" --records "$typed" --report "$t/sleep-s$seed/typed.md"
    fi
  done
  # The control of the frozen route: the base, read once (§4.1).
  for g in recall rule110; do
    probe "$base" "$g" "$test_seed" "$test_eps" 1 "$t/base/$g.json"
  done
  for g in "${games[@]}"; do
    probe "$base" "$g" "$arena_seed" "$arena_eps" 1 "$t/base/$g.json"
  done
  if [ "$dry" = 1 ] || [ ! -f "$t/base/typed.md" ]; then
    x nice -n 19 "$T" s2-eval --run "$base" --records "$typed" --report "$t/base/typed.md"
  fi
  x touch "$t/done"
  rep report "$root" ${smoke_arg[@]+"${smoke_arg[@]}"}
}

# §6 (v1 §8, v2 §7): the s6-nano gate on the exported seed-0 file (sky-eval at full depth with a
# zero memory; the gate's own arena through sky-arena, its default seed 7 and 30 episodes per
# game, the memory at the chosen N, s3-pico on the same seeds), H0 to H3 and Z, and the golden
# parity of sky-infer, natively and in WASM.
gate() {
  local g=$root/gate model=$models/sleep-s0.gguf chosen=1
  local baseline=${BASELINE:-site/public/models/s3-pico.gguf}
  if [ "$smoke" = 1 ]; then
    echo "== SMOKE: the gate is not part of the smoke (release builds of sky-eval, sky-arena and sky-infer, wasm-pack)"
    return
  fi
  [ "$dry" = 1 ] || chosen=$(json_get "$root/n.json" n)
  x mkdir -p "$g"
  x nice -n 19 "$B/sky-eval" "$model" --records "$data/decisions/typed_decisions_bench.jsonl,$data/s4/heldout/zs_massive_en.jsonl,$data/s4/heldout/zs_massive_fr.jsonl,$data/decisions/when2call.jsonl,$data/decisions/notam.jsonl" \
    --max 2000 --exit-threshold 1.0 --json "$g/eval-full.json"
  x nice -n 19 "$B/sky-arena" "$model" --episodes 30 --game all --exit-threshold 1.0 \
    --memory-passes "$chosen" --json "$g/arena-model.json"
  x nice -n 19 "$B/sky-arena" "$baseline" --episodes 30 --game all --exit-threshold 1.0 \
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
  rep gate "$root"
}

[ "$dry" = 1 ] || [ -n "${SKY_TRAIN:-}" ] || nice -n 19 cargo build -q --release -p sky-train
steps=("$@")
[ ${#steps[@]} -eq 0 ] && steps=(k3)
for s in "${steps[@]}"; do
  case "$s" in
    k3) k3 ;;
    nano) nano ;;
    dev) dev ;;
    export) export_ ;;
    eval) eval_ ;;
    gate)
      [ "$dry" = 1 ] || nice -n 19 cargo build -q --release -p sky-infer -p sky-arena
      gate
      ;;
    *) echo "unknown step $s (k3, nano, dev, export, eval, gate)" >&2; exit 2 ;;
  esac
done
