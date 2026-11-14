#!/usr/bin/env bash
# Train s6-secrets-verify-pico (docs/s6-secrets2-prereg.md, section 5): the graph of
# s6-secrets-pico as it is, a verifier that judges one rule candidate in its context. One run,
# seed 0, on the data of scripts/build-secrets2-data.sh. No test file is read here.
#
# Steps (default: all, in order):
#   train   MLM pretraining, 12 working minutes, from the parameters of the s6-secrets-pico
#           pretraining, on data/text-open, data/text and the build's CredData text (30% of the
#           text sequences), plus 10% numeric curriculum, sequences of 256, batch 64; then the
#           decision training, 30 working minutes, on the mix creddata 0.45 / issues 0.35 /
#           synth 0.20, batch 64, ternary QAT after a 10% float warm-up, loop dropout 0.25,
#           uncompiled (docs/s6-gen-prereg.md, amendment 11)
#   export  s2-calibrate on the CredData validation candidates, ternary GGUF
#   tau     the operating thresholds on P(real), from the GGUF at full depth on the CredData
#           validation candidates: the one that maximises F1, and the F0.5 and F2 presets,
#           written into the GGUF (skycmd-s2.secrets2.tau_f1, .tau_f05, .tau_f2) and, with
#           their validation precision, recall and F, into tau.json next to it; and a copy of
#           the GGUF with its exit thresholds at 1 for the evaluation (amendment 13)
#   report  s2-eval on the validation records, then report.md: steps done, working minutes,
#           brake modes seen, thresholds, validation numbers per source and per exit
#
# The checkpoint is `last` (`last-ema` if the recipe adopts EMA). No other checkpoint is read.
#
# Usage: scripts/train-secrets2.sh [train] [export] [tau] [report]
#   BUILD=../skycmd-data/s6-secrets2/build   the data build (records/, pretrain-text/, BUILD.json)
#   The recipe is read from runs/recipe/recipe.json (scripts/s6-adopt.py env s6-secrets-pico):
#   the optimizer flags, the rate scale, the pretraining and decision flags, the config. None is
#   taken from the environment.
#   INIT_PARAMS=runs/s6-secrets-pico/pretrain/last/model.safetensors
#                   the parameters the pretraining starts from. s2-pretrain --init-params wants
#                   the tensors of the pretraining stack, MLM heads included; pretrain/init holds
#                   the decision model and has no MLM head. The script checks that every tensor
#                   the two files share is the same, so the stack starts from the parameters of
#                   pretrain/init
#   PRETRAIN_MIN=12  DECIDE_MIN=30   working minutes (the brake's rests do not count)
#   TIME_SCALE=1    multiplies both budgets (scripts/speed-probe.sh; docs/s6-prereg.md, amendment 4)
#   STEP_MEM_GB=0   GPU memory one decision step may take (s2-decide --step-mem-gb, with
#                   --length-pool 16); 0: no limit and every batch drawn alone, as s6-secrets-pico
#   SKY_DUTY=0.5  SKY_COOL=45:15  SKY_BRAKE_FILE=runs/brake   the brake (docs/s6-gen-prereg.md,
#                   amendments 5, 6 and 10; scripts/auto-brake.sh writes the file)
#   SKY_MLX_CACHE_GB=2   cap of MLX's buffer cache
#   RESUME=1        continue from the checkpoints of an interrupted run (unset: start over)
#   SKY_OUT         a prefix for everything this script writes (scripts/train-all.sh SMOKE=1)
#   SMOKE=1         the first contact with sky-train, in the manner of scripts/train-all.sh: 1
#                   minute of pretraining and 2 of decision training (PRETRAIN_MIN, DECIDE_MIN to
#                   change them), every step, everything under SKY_OUT (default
#                   runs/secrets2-smoke/), never in runs/s6-secrets-verify-pico or
#                   models/s6-secrets-verify-pico. It tests that the commands start and the
#                   files chain; its numbers are not looked at
#   DRY_RUN=1       print the commands instead of running them (nothing is built or trained)
set -euo pipefail
export DEVELOPER_DIR=${DEVELOPER_DIR:-/Applications/Xcode.app/Contents/Developer}
export CARGO_TARGET_DIR=${CARGO_TARGET_DIR:-$PWD/target/s3-train}
export SKY_DUTY=${SKY_DUTY:-0.5}
export SKY_COOL=${SKY_COOL-45:15}
export SKY_BRAKE_FILE=${SKY_BRAKE_FILE:-$PWD/runs/brake}
export SKY_MLX_CACHE_GB=${SKY_MLX_CACHE_GB:-2}
dry=${DRY_RUN:-0}
B=$CARGO_TARGET_DIR/release
T=$B/sky-train
E=$B/sky-eval
name=s6-secrets-verify-pico
recipe_model=s6-secrets-pico
recipe_json=runs/recipe/recipe.json
build=${BUILD:-../skycmd-data/s6-secrets2/build}
tok=data/tok/tok-s2-1024.json
init_params=${INIT_PARAMS:-runs/s6-secrets-pico/pretrain/last/model.safetensors}
init_ref=runs/s6-secrets-pico/pretrain/init/model.safetensors
step_mem=${STEP_MEM_GB:-0}
smoke=${SMOKE:-}
if [ -n "$smoke" ]; then
  SKY_OUT=${SKY_OUT:-runs/secrets2-smoke/}
  PRETRAIN_MIN=${PRETRAIN_MIN:-1}
  DECIDE_MIN=${DECIDE_MIN:-2}
fi
weights=creddata=0.45,issues=0.35,synth=0.20
records=$build/records/creddata,$build/records/issues,$build/records/synth
val_cred=$build/records/creddata/val.jsonl
run=${SKY_OUT:-}runs/$name
m=${SKY_OUT:-}models/$name
model=$m/skycmd-$name-ternary.gguf
ev=$run/eval
cfg=$run/config.json
# a smoke run never writes where the real run does
if [ -n "$smoke" ]; then
  case "$run" in
    "runs/$name" | "./runs/$name" | "$PWD/runs/$name")
      echo "SMOKE=1 needs a prefix that is not the real run's (SKY_OUT=$SKY_OUT)" >&2
      exit 2
      ;;
  esac
  echo "== SMOKE: everything under $SKY_OUT; nothing of it is a result"
fi

# The recipe, with no hand choice: CONFIG, OPT, LR_SCALE, PRETRAIN, DECIDE.
CONFIG=configs/$recipe_model.json OPT='' LR_SCALE=1 PRETRAIN='' DECIDE=''
if [ -f "$recipe_json" ]; then
  eval "$(python3 -I scripts/s6-adopt.py env "$recipe_model")"
elif [ "$dry" = 1 ]; then
  echo "# no $recipe_json: the commands below show the default options"
else
  echo "no $recipe_json: the recipe is adopted first (scripts/train-all.sh adopt)" >&2
  exit 1
fi
read -r -a opt <<<"$OPT"
read -r -a pre <<<"$PRETRAIN"
read -r -a dec <<<"$DECIDE"
case " $OPT " in
  *" --muon-lr "* | *" --adam-lr "*)
    echo "the recipe's OPT carries a rate: LR_SCALE sets it (each phase scales its own default)" >&2
    exit 2
    ;;
esac
lr_flag=--muon-lr
[[ " $OPT " == *" --no-muon "* ]] && lr_flag=--adam-lr
ckpt=last
[[ " $DECIDE " == *" --ema "* ]] && ckpt=last-ema
# rate <phase default>: the peak rate of one phase (d = 96: the pico rate as it is).
rate() { LC_ALL=C awk -v a="$1" -v b="$LR_SCALE" 'BEGIN { printf "%g", a * b }'; }
# tmin <minutes>: a budget in minutes times TIME_SCALE (the M5's speed over this host's).
tmin() { LC_ALL=C awk -v a="$1" -v b="${TIME_SCALE:-1}" 'BEGIN { printf "%g", a * b }'; }
pre_min=$(tmin "${PRETRAIN_MIN:-12}")
dec_min=$(tmin "${DECIDE_MIN:-30}")
resume=()
[ -n "${RESUME:-}" ] && resume=(--resume)
mem=()
[ "$step_mem" != 0 ] && mem=(--length-pool 16 --step-mem-gb "$step_mem")

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
    "$@" 2>&1 | tee -a "$log"
  fi
}
# note <text>: what a step done in this script would do, under DRY_RUN=1.
note() { [ "$dry" != 1 ] || echo "# $*"; }

# The window of section 4.3 is the one the rule gave on the built data (BUILD.json).
window=320
if [ -f "$build/BUILD.json" ]; then
  window=$(python3 -I -c 'import json, sys; print(json.load(open(sys.argv[1]))["truncation"]["window"])' "$build/BUILD.json")
elif [ "$dry" != 1 ]; then
  echo "no $build/BUILD.json: run scripts/build-secrets2-data.sh first, or set BUILD" >&2
  exit 1
fi

# The files of the build are those BUILD.json names, and the model is the recipe's graph with
# the state window of the rule: $cfg, the recipe's config under this model's name.
prepare() {
  note "check the sha256 of the record files and of the pretraining text against $build/BUILD.json"
  note "write $cfg: $CONFIG as it is, named $name, max_state_tokens $window, max_head_tokens 192"
  note "check that $init_params and $init_ref hold the same state stack"
  [ "$dry" = 1 ] && return
  for f in "$tok" "$init_params" "$init_ref"; do
    [ -f "$f" ] || { echo "$f does not exist" >&2; exit 1; }
  done
  for d in data/text-open data/text data/numeric; do
    [ -d "$d" ] || { echo "$d comes from scripts/prepare-data.sh" >&2; exit 1; }
  done
  mkdir -p "$run" "$m" "$ev"
  python3 -I - "$build" "$CONFIG" "$cfg" "$name" "$window" "$init_params" "$init_ref" <<'EOF'
import hashlib, json, struct, sys
build, config, out, name, window, start, ref = sys.argv[1:]

# 1. the files trained on are the build's
manifest = json.load(open(f"{build}/BUILD.json"))
wanted = [f"records/{s}/{p}.jsonl" for s in ("creddata", "issues", "synth") for p in ("train", "val")]
wanted.append("pretrain-text/creddata.jsonl")
for rel in wanted:
    h = hashlib.sha256()
    with open(f"{build}/{rel}", "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    if h.hexdigest() != manifest["outputs"][rel]["sha256"]:
        sys.exit(f"{build}/{rel} is not the file of BUILD.json")
if manifest["truncation"]["window"] != int(window):
    sys.exit("the window is not the one of BUILD.json")

# 2. the recipe's graph, this model's name, the window of the rule
cfg = json.load(open(config))
cfg["name"] = name
cfg["max_state_tokens"] = int(window)
if cfg["max_head_tokens"] != 192:
    sys.exit(f"{config}: max_head_tokens {cfg['max_head_tokens']}, section 4.3 says 192")
graph = (cfg["vocab_size"], cfg["hidden_size"], cfg["state_layers"], cfg["state_loops"], cfg["exits"], cfg["fuse_numbers"])
if graph != (1024, 96, 1, 4, [1, 2, 3, 4], True):
    sys.exit(f"{config} is not the graph of s6-secrets-pico: {graph}")
json.dump(cfg, open(out, "w"), indent=2)

# 3. the pretraining starts from the state stack of pretrain/init
def tensors(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        head = json.loads(f.read(n))
        out = {}
        for k, v in head.items():
            if k == "__metadata__":
                continue
            a, b = v["data_offsets"]
            f.seek(8 + n + a)
            out[k] = (v["dtype"], tuple(v["shape"]), hashlib.sha256(f.read(b - a)).hexdigest())
        return out
a, b = tensors(start), tensors(ref)
shared = sorted(set(a) & set(b))
differ = [k for k in shared if a[k] != b[k]]
heads = sorted(k for k in a if k not in b)
if not shared or differ or any(not k.startswith("mlm") for k in heads):
    sys.exit(f"{start} is not the pretraining stack of {ref}: differ {differ}, apart {heads}")
print(f"{start}: {len(shared)} tensors of the state stack as in {ref}, {len(heads)} MLM-head tensors")
EOF
}

train() {
  prepare
  local from=(--init-params "$init_params")
  # a resumed pretraining keeps its own weights: --init-params would put the start back
  [ -n "${RESUME:-}" ] && [ -f "$run/pretrain/last/state.json" ] && from=()
  echo "== $name: pretrain ${pre_min}m (sequences of 256 tokens, from $init_params)"
  logged "$run/pretrain.log" "$T" s2-pretrain --config "$cfg" --tokenizer "$tok" \
    --text "data/text-open,data/text,$build/pretrain-text" --minutes "$pre_min" --seq 256 --batch 64 \
    --num-weight 0.05 --numeric data/numeric --numeric-weight 0.1 --seed 0 \
    ${from[@]+"${from[@]}"} --out "$run/pretrain" \
    "$lr_flag" "$(rate 3e-3)" ${opt[@]+"${opt[@]}"} ${pre[@]+"${pre[@]}"} ${resume[@]+"${resume[@]}"}
  echo "== $name: decide ${dec_min}m on $records (QAT after a 10% float warm-up)"
  [ -n "${RESUME:-}" ] || x rm -rf "$run/decide"
  logged "$run/decide.log" "$T" s2-decide --config "$cfg" --init "$run/pretrain/init" --tokenizer "$tok" \
    --records "$records" --weights "$weights" --batch 64 --minutes "$dec_min" --qat-frac 0.9 \
    --loop-dropout 0.25 --pad-to 32 --seed 0 --no-compile ${mem[@]+"${mem[@]}"} \
    --out "$run/decide" "$lr_flag" "$(rate 1e-3)" ${opt[@]+"${opt[@]}"} ${dec[@]+"${dec[@]}"} ${resume[@]+"${resume[@]}"}
}

export_() {
  x mkdir -p "$m"
  x "$T" s2-calibrate --run "$run/decide/$ckpt" --records "$val_cred"
  x "$T" s2-export --run "$run/decide/$ckpt" --format ternary --out "$model"
}

# The thresholds are read on what a host computes: the ternary GGUF through the engine, at full
# depth, one P(real) per CredData validation candidate (the records of source creddata:real;
# the weak negatives are training items only and have none here).
tau() {
  x mkdir -p "$ev"
  x "$E" "$model" --records "$val_cred" --split val --exit-threshold 1.0 \
    --json "$ev/val-creddata.json" --dump-probs "$ev/val-creddata-probs.jsonl"
  note "from $ev/val-creddata-probs.jsonl: the thresholds that maximise F1, F0.5 and F2 -> $m/tau.json"
  x_tau
  # A model published at full depth carries its exit thresholds at 1: the copy the evaluation
  # and the WASM parity read (docs/s6-gen-prereg.md, amendment 13).
  x "$B/gguf-set" "$model" "$ev/full-depth.gguf" skycmd-s2.exit_threshold.choice=1 \
    skycmd-s2.exit_threshold.score=1 skycmd-s2.exit_threshold.noul=1
}
x_tau() {
  if [ "$dry" = 1 ]; then
    x "$B/gguf-set" "$model" "$model" "skycmd-s2.secrets2.tau_f1=<tau>" \
      "skycmd-s2.secrets2.tau_f05=<tau>" "skycmd-s2.secrets2.tau_f2=<tau>"
    return
  fi
  local sets
  read -r -a sets <<<"$(python3 -I - "$ev/val-creddata-probs.jsonl" "$build/BUILD.json" "$model" "$m/tau.json" <<'EOF'
import hashlib, json, struct, sys
probs, manifest, model, out = sys.argv[1:]
rows = [json.loads(l) for l in open(probs) if l.strip()]
rows = [r for r in rows if r["src"] == "creddata:real"]
want = json.load(open(manifest))["mix"]["sources"]["creddata"]["val"]["candidates"]
if len(rows) != want:
    sys.exit(f"{len(rows)} creddata:real validation records, BUILD.json has {want} candidates")
# P(real) is the probability of `true`; a candidate is positive when its target says so
scored = sorted((r["probs"][1], r["gold"] == 1) for r in rows)
positives = sum(y for _, y in scored)
f32 = lambda v: struct.unpack("<f", struct.pack("<f", v))[0]

def best(beta):
    """The threshold (accept when P(real) >= tau) of the highest F-beta; of equal ones, the highest."""
    top = None
    tp = fp = 0
    i = len(scored)
    while i > 0:
        tau = scored[i - 1][0]
        while i > 0 and scored[i - 1][0] == tau:
            i -= 1
            tp += scored[i][1]
            fp += not scored[i][1]
        p, r = tp / (tp + fp), tp / max(positives, 1)
        f = (1 + beta * beta) * p * r / (beta * beta * p + r) if p + r > 0 else 0.0
        if top is None or f > top["f"]:
            top = {"tau": tau, "precision": p, "recall": r, "f": f, "tp": tp, "fp": fp, "fn": positives - tp,
                   "accepted": tp + fp}
    if top["tau"] != f32(top["tau"]):
        sys.exit("a probability of the engine is not a float32")
    return top

taus = {"tau_f1": best(1.0), "tau_f05": best(0.5), "tau_f2": best(2.0)}
h = hashlib.sha256(open(model, "rb").read()).hexdigest()
json.dump({
    "model": model, "model_sha256_before_the_keys": h,
    "on": "CredData validation candidates (records creddata:real), ternary GGUF at full depth",
    "candidates": len(scored), "positives": positives,
    "rule": "accept when P(real) >= tau; tau maximises F-beta on these candidates, the highest of equal ones",
    **{k: {**v, "beta": b} for (k, v), b in zip(taus.items(), (1.0, 0.5, 2.0))},
}, open(out, "w"), indent=2)
print(" ".join(f"skycmd-s2.secrets2.{k}={v['tau']!r}" for k, v in taus.items()))
EOF
)"
  [ ${#sets[@]} -eq 3 ] || { echo "the thresholds were not computed" >&2; exit 1; }
  "$B/gguf-set" "$model" "$model" "${sets[@]}"
}

report() {
  x mkdir -p "$ev"
  x "$T" s2-eval --run "$run/decide/$ckpt" --records "$records" --split val --gguf "$model" \
    --report "$ev/validation.md"
  note "write $run/report.md: steps, working minutes, brake modes, thresholds, then $ev/validation.md"
  [ "$dry" = 1 ] && return
  python3 -I - "$run" "$ckpt" "$m/tau.json" "$ev/validation.md" "$build/BUILD.json" "$SKY_BRAKE_FILE" \
    "$SKY_DUTY ${SKY_COOL:-no long rests}" "$init_params" "$weights" "$smoke" <<'EOF'
import hashlib, json, os, re, sys
run, ckpt, tau_path, validation, manifest, brake_file, brake_env, start, weights, smoke = sys.argv[1:]
out = ["# s6-secrets-verify-pico: the run", ""]
if smoke:
    out += ["**SMOKE RUN.** A few minutes under a throwaway prefix, to test that the commands start and the files "
            "chain: no number below is a result, none is to be looked at.", ""]
out += [
       "Pre-registration: `docs/s6-secrets2-prereg.md`, section 5. One run, seed 0; the checkpoint read is "
       f"`{ckpt}`, no other.", ""]
build = json.load(open(manifest))
out += [f"Data: `BUILD.json` sha256 `{hashlib.sha256(open(manifest, 'rb').read()).hexdigest()}`, "
        f"state window {build['truncation']['window']}, mix `{weights}`.",
        f"Pretraining started from `{start}`.", "", "## Steps and working minutes", "",
        "| phase | steps | working minutes | budget |", "|---|---:|---:|---:|"]
for phase, state in (("pretraining", f"{run}/pretrain/last/state.json"), ("decision", f"{run}/decide/{ckpt}/state.json")):
    s = json.load(open(state))
    out.append(f"| {phase} | {s['step']} | {s['elapsed_secs'] / 60:.2f} | {s['budget_secs'] / 60:g} |")
out += ["", "## Brake modes seen", "",
        f"At start: `SKY_DUTY SKY_COOL` = `{brake_env}`, brake file `{brake_file}`. Budgets count working time only.", ""]
modes = []
for log in ("pretrain.log", "decide.log"):
    path = os.path.join(run, log)
    if os.path.exists(path):
        for line in open(path, errors="replace"):
            if line.startswith("brake: ") or line.startswith("cooling down"):
                modes.append((log, re.sub(r"\s+", " ", line.strip())))
seen = {}
for log, line in modes:
    key = (log, re.sub(r"for \d+ s", "for N s", line))
    seen[key] = seen.get(key, 0) + 1
out += [f"- `{log}`: {line} ({n} times)" for (log, line), n in seen.items()] or ["- no change of mode and no long rest was logged"]
t = json.load(open(tau_path))
out += ["", "## Operating thresholds", "",
        f"On {t['candidates']} CredData validation candidates ({t['positives']} positive), {t['on'].split(', ', 1)[1]}; {t['rule']}.", "",
        "| preset | GGUF key | tau | precision | recall | F | accepted |", "|---|---|---:|---:|---:|---:|---:|"]
for k, label in (("tau_f1", "F1"), ("tau_f05", "F0.5"), ("tau_f2", "F2")):
    v = t[k]
    out.append(f"| {label} | `skycmd-s2.secrets2.{k}` | {v['tau']:.6g} | {v['precision']:.4f} | {v['recall']:.4f} | {v['f']:.4f} | {v['accepted']} |")
out += ["", "## Validation, per source and per exit", "",
        "From `sky-train s2-eval` on the validation records of the three sources (below, its report as written).", ""]
body = open(validation).read()
out.append(re.sub(r"^#", "##", body, flags=re.M))
open(os.path.join(run, "report.md"), "w").write("\n".join(out) + "\n")
print(f"wrote {run}/report.md")
EOF
}

steps=("$@")
[ ${#steps[@]} -eq 0 ] && steps=(train export tau report)
for s in "${steps[@]}"; do
  case "$s" in
    train | export | tau | report) ;;
    *) echo "unknown step $s (train, export, tau, report)" >&2; exit 2 ;;
  esac
done
[ "$dry" = 1 ] || cargo build -q --release -p sky-train -p sky-gguf -p sky-infer
for s in "${steps[@]}"; do
  case "$s" in
    train) train ;;
    export) export_ ;;
    tau) tau ;;
    report) report ;;
  esac
done
