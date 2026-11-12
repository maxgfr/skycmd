#!/usr/bin/env bash
# LFM3-1.7B fine-tuned so that answering without thinking (System 1) works better
# (docs/lfm3-ft-prereg.md). Every GPU step is a Hugging Face Job; artifacts go through the PRIVATE
# dataset repo $WORK, the model to the PRIVATE model repo $FT_REPO. Nothing is public.
#
# Usage: scripts/lfm3-ft.sh <step>...      (run from the main checkout: data/lfm3 must exist)
#   smoke   ~20 calibration questions on an L4: base revision, vLLM or transformers, LoRA module
#           names, TRL/peft on the custom code. Its revision and engine are then pinned below by a
#           dated amendment of the pre-registration, before any other job.
#   gen     self-distillation samples (base model, s2-1024, 4 per train question) -> $WORK gen/
#   sft     LoRA training, merge, push to $FT_REPO (private)                     -> $WORK sft/
#   eval    s1, s2-256, s2-1024 on calib + test, s1 on MMLU-Pro-500, for each of $EVAL_MODELS
#           (base, ft; one job each)                                           -> $WORK eval/<tag>/
#   pull    downloads $WORK (but the gen samples) into runs/lfm3-ft/              (local, free)
#   route   the S1/S2 router of docs/lfm3-s1s2-prereg.md on base and ft         (local CPU, free)
#   gate    G1-G3 -> runs/lfm3-ft/gate.json and report.md                       (local CPU, free)
#
# MODE=local (the default since docs/lfm3-ft-prereg.md, amendment 1) runs everything on the M5, for
# nothing, with SKY_DUTY=0.5 and SKY_COOL=45:15 unless set: smoke, gen, sft (then the MLX conversion),
# eval, route, gate; `pull` has nothing to do. MODE=hf keeps the Hugging Face Jobs below.
#
# DRY_RUN=1 prints the commands and launches nothing. With MODE=hf and without DRY_RUN, each paid job
# prints its flavor, estimated and maximum cost, and waits for "y" (YES=1 answers for you: only once the
# user has agreed to that cost). Jobs are detached: follow them with `hf jobs ps` and `hf jobs logs <id>`.
set -euo pipefail
cd "$(dirname "$0")/.."
export LC_ALL=C # awk prints 0.48, not 0,48
MODE=${MODE:-local}
case "$MODE" in local | hf) ;; *) echo "MODE must be local or hf, not '$MODE'" >&2; exit 2 ;; esac

BASE=LiquidAI/LFM3-1.7B
WORK=${WORK:-maxgfr/skycmd-lfm3-ft-work}
FT_REPO=${FT_REPO:-maxgfr/skycmd-lfm3-1.7b-s1}
# Recopied from the smoke by a dated amendment of docs/lfm3-ft-prereg.md (§10); empty until then.
BASE_REV_PINNED=""
ENGINE_PINNED=""
BASE_REV=${BASE_REV:-$BASE_REV_PINNED}
ENGINE=${ENGINE:-$ENGINE_PINNED}
EVAL_MODELS=${EVAL_MODELS:-base ft}

# Flavor, timeout and expected hours per job (`hf jobs hardware` for the list and prices).
FLAVOR_SMOKE=${FLAVOR_SMOKE:-l4x1}
TIMEOUT_SMOKE=${TIMEOUT_SMOKE:-1h}
HOURS_SMOKE=${HOURS_SMOKE:-0.6}
FLAVOR_GEN=${FLAVOR_GEN:-a100-large}
TIMEOUT_GEN=${TIMEOUT_GEN:-4h}
HOURS_GEN=${HOURS_GEN:-2}
FLAVOR_SFT=${FLAVOR_SFT:-a100-large}
TIMEOUT_SFT=${TIMEOUT_SFT:-4h}
HOURS_SFT=${HOURS_SFT:-2}
FLAVOR_EVAL=${FLAVOR_EVAL:-a100-large}
TIMEOUT_EVAL=${TIMEOUT_EVAL:-3h}
HOURS_EVAL=${HOURS_EVAL:-1.5}

EXP=experiments/lfm3-ft
DATA=data/lfm3
RUNS=runs/lfm3-ft
DRY_RUN=${DRY_RUN:-0}

price() { # $/h of a flavor, from `hf jobs hardware` on 05/11/2026
  case "$1" in
    l4x1) echo 0.80 ;; l40sx1) echo 1.80 ;; a10g-large) echo 1.50 ;; a100-large) echo 2.50 ;;
    h200) echo 5.00 ;; rtx-pro-6000) echo 2.75 ;; *) echo "" ;;
  esac
}

hours() { # 4h, 90m, 3600s -> hours
  local t=$1
  case "$t" in
    *h) echo "${t%h}" ;; *m) awk -v m="${t%m}" 'BEGIN{print m/60}' ;; *s) awk -v s="${t%s}" 'BEGIN{print s/3600}' ;;
    *) awk -v s="$t" 'BEGIN{print s/3600}' ;;
  esac
}

engine_with() { # the engine's packages, added with --with (they cannot share one environment)
  case "$1" in
    vllm) echo "--with vllm==0.33.0" ;;
    hf) echo "--with transformers==5.12.1 --with accelerate==1.15.0" ;;
    *) echo "unknown engine '$1' (vllm or hf)" >&2; exit 2 ;;
  esac
}

need_pins() {
  if [ -z "$BASE_REV" ] || [ -z "$ENGINE" ]; then
    if [ "$DRY_RUN" = 1 ]; then
      echo "# BASE_REV / ENGINE not pinned yet (the smoke decides, then an amendment): shown as placeholders"
      BASE_REV=${BASE_REV:-"<BASE_REV>"}
      ENGINE=${ENGINE:-vllm}
    else
      echo "BASE_REV and ENGINE are not pinned: run the smoke, then pin them by an amendment (docs/lfm3-ft-prereg.md §10)" >&2
      exit 2
    fi
  fi
  case "$ENGINE" in vllm | hf) ;; *) echo "ENGINE must be vllm or hf, not '$ENGINE'" >&2; exit 2 ;; esac
}

need_data() {
  local f
  for f in "$DATA/calib.jsonl" "$DATA/test.jsonl"; do
    if [ ! -s "$f" ] && [ "$DRY_RUN" != 1 ]; then
      echo "$f missing: run from the main checkout (scripts/lfm3-s1s2.sh data writes it)" >&2
      exit 2
    fi
  done
}

launch() { # name flavor timeout expected-hours script-and-args...
  local name=$1 flavor=$2 timeout=$3 h=$4
  shift 4
  local rate est max
  rate=$(price "$flavor")
  if [ -n "$rate" ]; then
    est=$(awk -v r="$rate" -v h="$h" 'BEGIN{printf "%.2f", r*h}')
    max=$(awk -v r="$rate" -v h="$(hours "$timeout")" 'BEGIN{printf "%.2f", r*h}')
  else
    est="?" max="?"
  fi
  local cmd=(hf jobs uv run --flavor "$flavor" --secrets HF_TOKEN --timeout "$timeout" --name "lfm3-ft-$name" --detach "$@")
  echo "== $name: $flavor at ${rate:-?} \$/h, about $h h = about \$$est, at most \$$max (timeout $timeout)"
  printf '%q ' "${cmd[@]}"
  echo
  [ "$DRY_RUN" = 1 ] && return 0
  if [ -z "${HF_TOKEN:-}" ]; then
    HF_TOKEN=$(hf auth token)
    export HF_TOKEN
  fi
  if [ "${YES:-0}" != 1 ]; then
    local ans
    read -r -p "Launch this paid job (about \$$est, at most \$$max)? [y/N] " ans
    if [ "$ans" != y ]; then
      echo "   not launched"
      return 0
    fi
  fi
  "${cmd[@]}"
}

ft_rev() { # the model repo commit written by sft.py (pulled), or $FT_REV
  if [ -n "${FT_REV:-}" ]; then
    echo "$FT_REV"
  elif [ -s "$RUNS/sft/result.json" ]; then
    python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["commit"])' "$RUNS/sft/result.json"
  elif [ "$DRY_RUN" = 1 ]; then
    echo "<FT_REV from sft/result.json>"
  else
    echo "no $RUNS/sft/result.json: run sft, then pull (or set FT_REV)" >&2
    exit 2
  fi
}

[ $# -eq 0 ] && { sed -n '2,27p' "$0"; exit 2; }

# ----------------------------------------------------------------------------- the M5 (amendment 1)
# Two environments: mlx-vlm generates (as scripts/lfm3-s1s2.sh), the torch stack trains the LoRA.
run_local() {
  printf '+'
  printf ' %q' "$@"
  printf '\n'
  [ "$DRY_RUN" = 1 ] || "$@"
}
py_mlx() {
  run_local uv run -q --python 3.12 --with "mlx-vlm==0.7.11" --with "datasets>=3" python "$EXP/local.py" "$@"
}
py_lora() {
  run_local uv run -q --python 3.12 --with torch==2.14.0 --with transformers==5.12.1 --with trl==1.17.1 \
    --with peft==0.21.2 --with accelerate==1.15.0 --with datasets==5.1.0 --with "huggingface_hub>=1.31,<2" \
    python "$EXP/local.py" "$@"
}
# The bf16 revision the smoke returned, pinned by a dated amendment before `gen` (BASE_REV_PINNED above).
need_rev() {
  [ -n "$BASE_REV" ] && return 0
  if [ "$DRY_RUN" = 1 ]; then
    BASE_REV="<BASE_REV>"
  else
    echo "BASE_REV is not pinned: run the smoke, then pin it by an amendment (docs/lfm3-ft-prereg.md §10)" >&2
    exit 2
  fi
}
local_step() {
  case "$1" in
    smoke)
      need_data
      py_lora smoke-lora ${BASE_REV:+--revision "$BASE_REV"}
      py_mlx smoke-mlx
      ;;
    gen)
      need_data
      need_rev
      py_mlx gen
      ;;
    sft)
      need_data
      need_rev
      py_lora sft --revision "$BASE_REV"
      py_mlx convert
      ;;
    eval)
      need_data
      need_rev
      for tag in $EVAL_MODELS; do py_mlx eval --tag "$tag"; done
      ;;
    pull) echo "== pull: nothing to do on the M5, everything is already in $RUNS" ;;
  esac
}
if [ "$MODE" = local ]; then
  export SKY_DUTY=${SKY_DUTY:-0.5}
  export SKY_COOL=${SKY_COOL-45:15}
  # The brake file of scripts/auto-brake.sh, when it runs, wins over the two values (amendment 2).
  export SKY_BRAKE_FILE=${SKY_BRAKE_FILE-$PWD/runs/brake}
fi

for s in "$@"; do
  # Not inside an `if`: `set -e` has to stop a failed step before the next command of the step.
  case "$MODE:$s" in
    local:smoke | local:gen | local:sft | local:eval | local:pull)
      local_step "$s"
      continue
      ;;
  esac
  case "$s" in
    smoke)
      need_data
      launch smoke "$FLAVOR_SMOKE" "$TIMEOUT_SMOKE" "$HOURS_SMOKE" "$EXP/smoke.py" -- \
        --common "$EXP/common.py" --calib "$DATA/calib.jsonl" --n 20 --work "$WORK"
      ;;
    gen)
      need_data
      need_pins
      # shellcheck disable=SC2046
      launch gen "$FLAVOR_GEN" "$TIMEOUT_GEN" "$HOURS_GEN" $(engine_with "$ENGINE") "$EXP/gen.py" -- \
        --common "$EXP/common.py" --calib "$DATA/calib.jsonl" --test "$DATA/test.jsonl" \
        --revision "$BASE_REV" --engine "$ENGINE" --work "$WORK"
      ;;
    sft)
      need_data
      need_pins
      launch sft "$FLAVOR_SFT" "$TIMEOUT_SFT" "$HOURS_SFT" "$EXP/sft.py" -- \
        --common "$EXP/common.py" --calib "$DATA/calib.jsonl" --test "$DATA/test.jsonl" \
        --revision "$BASE_REV" --repo "$FT_REPO" --work "$WORK"
      ;;
    eval)
      need_data
      need_pins
      for tag in $EVAL_MODELS; do
        case "$tag" in
          base) model=$BASE rev=$BASE_REV ;;
          ft) model=$FT_REPO rev=$(ft_rev) ;;
          *) echo "unknown model $tag (base, ft)" >&2; exit 2 ;;
        esac
        # shellcheck disable=SC2046
        launch "eval-$tag" "$FLAVOR_EVAL" "$TIMEOUT_EVAL" "$HOURS_EVAL" $(engine_with "$ENGINE") "$EXP/eval.py" -- \
          --common "$EXP/common.py" --calib "$DATA/calib.jsonl" --test "$DATA/test.jsonl" \
          --model "$model" --revision "$rev" --tag "$tag" --engine "$ENGINE" --work "$WORK"
      done
      ;;
    pull)
      cmd=(hf download "$WORK" --repo-type dataset --local-dir "$RUNS" --exclude "gen/samples*/*")
      printf '%q ' "${cmd[@]}"
      echo
      [ "$DRY_RUN" = 1 ] || "${cmd[@]}"
      ;;
    route)
      for tag in base ft; do
        if [ -s "$RUNS/eval/$tag/meta.json" ]; then
          echo "== router on $tag"
          [ "$DRY_RUN" = 1 ] || python3 "$EXP/route.py" --tag "$tag" --runs "$RUNS"
        else
          echo "== router on $tag: $RUNS/eval/$tag/meta.json missing (eval, then pull), skipped"
        fi
      done
      ;;
    gate)
      echo "== gates G1-G3 -> $RUNS/gate.json, $RUNS/report.md"
      [ "$DRY_RUN" = 1 ] || python3 "$EXP/gate.py" --runs "$RUNS"
      ;;
    *) echo "unknown step $s (smoke, gen, sft, eval, pull, route, gate)" >&2; exit 2 ;;
  esac
done
