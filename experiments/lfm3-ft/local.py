#!/usr/bin/env python3
"""The LFM3-1.7B fine-tuning experiment on the M5 (docs/lfm3-ft-prereg.md, amendments 1, 5 and 6).

scripts/lfm3-ft.sh runs each sub-command in its own uv environment: mlx-vlm for generation, the torch
stack for the LoRA (they cannot share one). Generation reuses experiments/lfm3-s1s2/s1s2.py (Model,
gen_s1, gen_s2, rest: the same prompts, sampling and seeds as the router's run), and the selection,
the mix and the trainer come from common.py. Everything stays under runs/lfm3-ft and models/lfm3-ft.

  smoke-lora   revision, template, LoRA names, 2 steps at maximal length, merge and reload  (torch)
  smoke-mlx-lora  amendment 1, rule 2: mlx-vlm's LoRA trainer on the same points, when peft failed  (mlx)
  smoke-mlx    throughput, the local MLX conversion against the published base; report.json (mlx)
  gen          not run: amendment 5 drops part (b) of §3, the marker gen/not-run.json is written
  sft          the LoRA on part (a) only (3,000 questions per task, seed 1234, one epoch), merged (torch)
  convert      the merged model to MLX 4 bits                                               (mlx)
  eval         base: the runs/lfm3-s1s2 generations copied, s1 on MMLU-Pro-500 (the smoke says
               "published"), or s1 only on calib, test and MMLU-Pro-500 with the local conversion
               (it says "converted", amendment 6); ft: s1 only on the three (amendment 5)   (mlx)
  pair         eval/pair: the ft s1 files and the published base's s2 files of runs/lfm3-s1s2,
               copied, for `route`, in both cases (amendment 6)                          (nothing)

SKY_DUTY and SKY_COOL are honoured after every sample and every LoRA step (s1s2.py's rest). Generation
appends and skips what is there; the LoRA resumes from its last technical checkpoint, never read to
choose anything.
"""
import argparse
import importlib.util
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C  # noqa: E402
import smoke as SM  # noqa: E402  (its helpers only: what is heavy is imported inside its functions)

C_DATA = "data/lfm3"
RUNS = "runs/lfm3-ft"
MODELS = "models/lfm3-ft"
FT_MLX = f"{MODELS}/ft-mlx-4bit"
BASE_MLX = f"{MODELS}/base-mlx-4bit"
MERGED = f"{MODELS}/merged"
SMOKE = f"{RUNS}/smoke"
S1S2_RUNS = "runs/lfm3-s1s2"
SMOKE_N = 20
SFT_RESULT = f"{RUNS}/sft/result.json"  # written last by `sft`: its presence means the merged model is complete
SAVE_EVERY = 20  # optimizer steps between two technical checkpoints (about 188 steps in all: a stop loses < 20)


def load_s1s2():
    spec = importlib.util.spec_from_file_location("s1s2", os.path.join(HERE, "..", "lfm3-s1s2", "s1s2.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


S = load_s1s2()


class LocalModel(S.Model):
    """s1s2.py's Model, on the published MLX 4-bit weights or on a local MLX directory. `question` is
    already the full user message (common.user_prompt: the math prompt or the MMLU-Pro one)."""

    def __init__(self, path=None):
        if path is None:
            super().__init__()
            return
        from mlx_vlm import load

        self.model, self.processor = load(path)
        self.tok = getattr(self.processor, "tokenizer", self.processor)

    def prompt(self, question, thinking):
        msgs = [{"role": "user", "content": question}]
        return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=thinking)


def user_item(it):
    return {"question": C.user_prompt(it)}


def graded(r, it, mode, seconds):
    """The record of one generation: s1s2.py's cmd_gen fields, with common.py's grading (MMLU-Pro too)."""
    r["pred"], r["correct"] = C.grade(it["task"], r["text"], it["answer"])
    r.update(id=it["id"], task=it["task"], mode=mode, seconds=seconds)
    return r


def sample_record(r, it, k, seed, seconds):
    """A self-distillation sample: distill_targets reads answer_tokens, which s1s2.py does not write."""
    r["answer_tokens"] = r["tokens"] - r["think_tokens"]
    r.update(k=k, seed=seed)
    return graded(r, it, "s2-1024", seconds)


def splits(limit=0):
    out = {"calib": C.load_shipped(f"{C_DATA}/calib.jsonl"), "test": C.load_shipped(f"{C_DATA}/test.jsonl")}
    return {k: v[:limit] if limit else v for k, v in out.items()}


def train_rows():
    """The kept train rows and the counts of each exclusion (docs/lfm3-ft-prereg.md §3)."""
    calib, test = C.load_shipped(f"{C_DATA}/calib.jsonl"), C.load_shipped(f"{C_DATA}/test.jsonl")
    return C.select_train(C.load_train_rows(), calib, [r["question"] for r in test + C.mmlu_items()])


def cmd_gen(_):
    """Amendment 5: part (b) of §3, the self-distillation, is NOT RUN — the adapter is trained on the
    written solutions only (it is System 1) and the base stays System 2, so no THINK target exists.
    The step prints one line, writes its marker and exits 0."""
    reason = ("part (b) of §3 (self-distillation) is not run by docs/lfm3-ft-prereg.md, amendment 5: "
              "part (a) only, the adapter is System 1 and the base stays System 2, no THINK target")
    marker = {"not_run": True, "amendment": 5, "reason": reason,
              "date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    os.makedirs(f"{RUNS}/gen", exist_ok=True)
    with open(f"{RUNS}/gen/not-run.json", "w") as f:
        json.dump(marker, f, indent=2)
    print(f"gen: not run by amendment 5 -> {RUNS}/gen/not-run.json")
    print(json.dumps(marker, indent=2))


def memory_update(rec, driver, current, available):
    """Keep the peaks of one memory reading in `rec` (bytes): the MPS driver's allocation and the tensors
    alive at the reading (highest), the system's available memory (lowest)."""
    for key, value, pick in (("mps_driver_peak", driver, max), ("mps_current_peak", current, max),
                             ("available_min", available, min)):
        if value is not None:
            rec[key] = value if rec.get(key) is None else pick(rec[key], value)
    return rec


def memory_reading():
    """(MPS driver bytes, MPS current bytes, available system bytes); None for what cannot be read."""
    import torch

    try:
        import psutil

        available = psutil.virtual_memory().available
    except ImportError:
        available = None
    if not torch.backends.mps.is_available():
        return None, None, available
    return torch.mps.driver_allocated_memory(), torch.mps.current_allocated_memory(), available


def brake_callback(record=None):
    """The thermal brake between two optimizer steps (s1s2.py's rest). With `record` (the smoke), each
    step's seconds without its rest go to record["step_seconds"] and the memory peaks are kept in it,
    read before the rest."""
    from transformers import TrainerCallback

    class Brake(TrainerCallback):
        def on_step_begin(self, args, state, control, **kw):
            self.t0 = time.time()

        def on_substep_end(self, args, state, control, **kw):
            if record is not None:
                memory_update(record, *memory_reading())

        def on_step_end(self, args, state, control, **kw):
            dt = time.time() - self.t0
            if record is not None:
                record.setdefault("step_seconds", []).append(dt)
                memory_update(record, *memory_reading())
            S.rest(dt)

    return Brake()


def lora_settings():
    """The batch the smoke validated (amendment 1: 8 x 4, or the memory fallback 4 x 8)."""
    path = f"{SMOKE}/report.json"
    if not os.path.exists(path):
        raise SystemExit(f"{path} missing: run `scripts/lfm3-ft.sh smoke` first")
    with open(path) as f:
        rep = json.load(f)
    if not rep.get("ok") or rep.get("lora_method") not in ("peft-mps", "mlx-vlm"):
        raise SystemExit(f"{path}: the smoke validated no LoRA method (amendment 1: stop and report)")
    if rep["lora_method"] != "peft-mps":
        raise SystemExit(f"{path}: the smoke selected {rep['lora_method']} (amendment 1, rule 2); this `sft` trains "
                         "with peft only, and the mlx-vlm training step is not written yet")
    return rep["per_device"], rep["grad_accum"]


def replace_dir(tmp, dst):
    """Put a finished directory in place of dst (dst only ever holds a complete output)."""
    if os.path.exists(dst):
        shutil.rmtree(dst)
    os.replace(tmp, dst)


def cmd_sft(a):
    if os.path.exists(SFT_RESULT) and os.path.exists(os.path.join(MERGED, "config.json")):
        # Done: a second run would resume from the last checkpoint, train nothing, and overwrite the
        # record of the real run (its metrics, its time) with those of an empty one.
        print(f"sft: done ({SFT_RESULT}), kept; remove it to train again", flush=True)
        return
    from transformers.trainer_utils import get_last_checkpoint

    per_device, grad_accum = lora_settings()
    rows, excl = train_rows()
    kept_per_task = {t: sum(r["task"] == t for r in rows) for t in ("gsm8k", "math")}
    if os.path.exists(f"{RUNS}/gen/samples.jsonl"):
        print(f"{RUNS}/gen/samples.jsonl exists and is ignored (amendment 5: part (b) is not run)", flush=True)
    mix = C.build_mix(C.cap_per_task(rows), [])  # amendment 5: part (a) only, no THINK target
    tok = C.load_tokenizer(C.BASE, a.revision)
    tok_rows, dropped = C.tokenize_mix(tok, mix)
    kinds = {}
    for r in tok_rows:
        kinds[r["kind"]] = kinds.get(r["kind"], 0) + 1
    stats = {"exclusions": excl, "kept_per_task": kept_per_task, "cap_per_task": C.SFT_CAP_PER_TASK,
             "seed": C.SEED, "epochs": C.SFT_EPOCHS, "part_b": "absent (gen not run, amendment 5)",
             "examples_per_task": {t: sum(m["task"] == t for m in mix) for t in ("gsm8k", "math")},
             "examples": kinds, "dropped_over_max_len": dropped,
             "tokens": sum(len(r["input_ids"]) for r in tok_rows), "per_device": per_device, "grad_accum": grad_accum}
    print(json.dumps(stats, indent=2), flush=True)
    out_dir = f"{RUNS}/sft/checkpoints"
    trainer, tok, total = C.make_trainer(C.BASE, a.revision, tok_rows, out_dir, epochs=C.SFT_EPOCHS,
                                         per_device=per_device, grad_accum=grad_accum,
                                         save_steps=SAVE_EVERY, callbacks=[brake_callback()])
    if not C.check_completion_mask(trainer, tok_rows[0]):
        raise SystemExit("the collator does not mask the prompt as expected: stop (completion-only loss)")
    last = get_last_checkpoint(out_dir) if os.path.isdir(out_dir) else None
    print(f"{len(tok_rows)} examples, {total} optimizer steps, resuming from {last}", flush=True)
    t0 = time.time()
    out = trainer.train(resume_from_checkpoint=last)
    adapter = f"{RUNS}/sft/adapter"
    trainer.save_model(adapter)
    tmp = f"{MERGED}.tmp"
    if os.path.exists(tmp):
        shutil.rmtree(tmp)
    C.merge_and_save(C.BASE, a.revision, adapter, tmp, epochs="1 epoch (amendment 5)", distillation=False)
    replace_dir(tmp, MERGED)
    result = {"base": C.BASE, "base_revision": a.revision, "steps": total, "train_seconds_with_rests": time.time() - t0,
              "resumed_from": last, "train_metrics": out.metrics, "log_history": trainer.state.log_history, "mix": stats,
              "merged": MERGED, "versions": C.versions()}
    with open(f"{SFT_RESULT}.tmp", "w") as f:
        json.dump(result, f, indent=2)
    os.replace(f"{SFT_RESULT}.tmp", SFT_RESULT)
    print(json.dumps({k: v for k, v in result.items() if k != "log_history"}, indent=2))


def convert(src, dst, revision=None, run=subprocess.run):
    """The MLX 4-bit conversion of amendment 1 (the same command for the base and the fine-tuned model).
    It writes to dst.tmp and renames at the end: a conversion cut by a stop is never taken as done."""
    if os.path.exists(os.path.join(dst, "config.json")):
        print(f"{dst} exists, kept", flush=True)
        return
    tmp = f"{dst}.tmp"
    if os.path.exists(tmp):
        shutil.rmtree(tmp)
    cmd = [sys.executable, "-m", "mlx_vlm.convert", "--hf-path", src, "--mlx-path", tmp, "-q", "--q-bits", "4",
           "--q-mode", "affine", "--trust-remote-code"] + (["--revision", revision] if revision else [])
    print(" ".join(cmd), flush=True)
    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    run(cmd, check=True)
    replace_dir(tmp, dst)


def cmd_convert(_):
    if not os.path.exists(SFT_RESULT) or not os.path.exists(os.path.join(MERGED, "config.json")):
        raise SystemExit(f"{SFT_RESULT} missing: `scripts/lfm3-ft.sh sft` has not finished")
    convert(MERGED, FT_MLX)


def base_source():
    with open(f"{SMOKE}/report.json") as f:
        return json.load(f)["base_source"]


def copy_base_generations(d, src=None):
    """The router's run is the base model's evaluation (amendment 1): same prompts, seeds and engine.
    Each file is copied whole (temporary file, then rename) unless an identical-size copy is there: a
    copy cut by a stop is redone, never kept as if complete."""
    src = src or S1S2_RUNS
    names = [f"{s}.{m}.jsonl" for s in ("calib", "test") for m in ("s1", "s2-256", "s2-1024")]
    missing = [n for n in names if not os.path.exists(f"{src}/{n}")]
    if missing:
        raise SystemExit(f"{src} lacks {missing}: run scripts/lfm3-s1s2.sh first")
    os.makedirs(d, exist_ok=True)
    for n in names:
        if not os.path.exists(f"{d}/{n}") or os.path.getsize(f"{d}/{n}") != os.path.getsize(f"{src}/{n}"):
            C.copy_atomic(f"{src}/{n}", f"{d}/{n}")
    print(f"base: the generations of {src} are copied into {d}", flush=True)
    return names


def eval_plan(tag, limit, source=None):
    """What `eval --tag <tag>` does (amendments 5 and 6): (jobs to generate, weights path or None for the
    published file, model label, conversion of the generated files, copy the runs/lfm3-s1s2 files?).

    No step evaluates the base again in the three modes (amendment 6 replaces amendment 1's full
    re-evaluation): System 2 of the pair is always the published base of runs/lfm3-s1s2.
      ft                   s1 on calib, test and MMLU-Pro, the local conversion of the merged model
      base, "published"    the six files of runs/lfm3-s1s2 copied, s1 on MMLU-Pro with the published file
      base, "converted"    s1 on calib, test and MMLU-Pro with the local conversion of the base
    With --limit (a check of the engine, never the run) nothing is copied and only s1 is generated."""
    s1_only = [(s, "s1") for s in ("calib", "test", "mmlu")]
    if tag == "ft":
        return s1_only, FT_MLX, f"{C.FT_REPO} (local)", C.LOCAL_CONVERSION, False
    if source == "converted":
        return s1_only, BASE_MLX, f"{C.BASE} (converted locally)", C.LOCAL_CONVERSION, False
    if source != "published":
        raise SystemExit(f"{SMOKE}/report.json: base_source is {source!r}, not 'published' or 'converted'")
    if limit:
        return s1_only, None, S.MODEL, C.PUBLISHED_4BIT, False
    return [("mmlu", "s1")], None, S.MODEL, C.PUBLISHED_4BIT, True


def cmd_eval(a):
    d = f"{RUNS}/eval/{a.tag}" + ("-limit" if a.limit else "")
    data = splits(a.limit)
    data["mmlu"] = C.mmlu_items()[: a.limit] if a.limit else C.mmlu_items()
    source = base_source() if a.tag == "base" else None
    jobs, path, label, conversion, copy = eval_plan(a.tag, a.limit, source)
    if path is not None and not os.path.exists(os.path.join(path, "config.json")):
        raise SystemExit(f"{path} missing: run `scripts/lfm3-ft.sh "
                         + ("smoke` (it converts the base)" if a.tag == "base" else "sft` (it converts the merged model)"))
    copied = copy_base_generations(d) if copy else []
    m = None
    meta = {"model": label, "revision": S.MODEL_REV if path is None else "local", "tag": a.tag, "engine": "mlx-vlm",
            "weights": "MLX 4-bit", "base_source": source, "conversion": conversion, "versions": C.versions(),
            "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "files": {}}
    if a.tag == "base":
        meta["system2"] = (f"not generated here: the pair's System 2 is the published base, {S1S2_RUNS} "
                           f"({C.PUBLISHED_4BIT}), in both cases (amendment 6)")
    for split, mode in jobs:
        out = f"{d}/{split}.{mode}.jsonl"
        if C.repair_jsonl(out):  # a record cut by a stop: dropped, so it is generated again
            print(f"{out}: its last line was cut mid-write, repaired", flush=True)
        done = {r["id"] for r in C.read_jsonl(out)}
        todo = [it for it in data[split] if it["id"] not in done]
        print(f"{a.tag} {mode} on {split}: {len(todo)} to generate ({len(done)} done)", flush=True)
        if todo and m is None:
            m = LocalModel(path)
        os.makedirs(d, exist_ok=True)
        with open(out, "a") as f:
            for it in todo:
                seed = C.qseed(it["id"])
                t0 = time.time()
                r = S.gen_s1(m, user_item(it), seed) if mode == "s1" else S.gen_s2(m, user_item(it), seed, int(mode[3:]))
                r = graded(r, it, mode, time.time() - t0)
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                f.flush()
                S.rest(r["seconds"])
    generated = {f"{split}.{mode}.jsonl" for split, mode in jobs}
    for name in sorted(os.listdir(d)):
        if name.endswith(".jsonl"):
            rows = C.read_jsonl(f"{d}/{name}")
            meta["files"][name] = {"n": len(rows), "acc": sum(bool(r["correct"]) for r in rows) / max(1, len(rows)),
                                   "conversion": conversion if name in generated or name in copied else "unknown",
                                   "from": S1S2_RUNS if name in copied else "generated here"}
    meta["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with open(f"{d}/meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2))


PAIR_FILES = (  # (split, mode, origin): the pair's s1 comes from the fine-tuned model, its s2 from the
    ("calib", "s1", "ft"), ("test", "s1", "ft"),  # published base's generations of runs/lfm3-s1s2
    ("calib", "s2-256", "s1s2"), ("calib", "s2-1024", "s1s2"),
    ("test", "s2-256", "s1s2"), ("test", "s2-1024", "s1s2"),
)


def build_pair(runs=None, data=None, s1s2=None):
    """The pair of amendment 5: the fine-tuned model answers in s1 (System 1) and the base model in
    s2-256 and s2-1024 (System 2, untouched). eval/pair holds the fine-tuned s1 files (eval/ft) and the
    published base's s2 files (runs/lfm3-s1s2), copied, never generated; meta.json says where each file
    came from and which conversion it was generated with, and `route --tag pair` reads the folder like
    any model directory (s1s2.py's cmd_route, unchanged).

    System 2 is read from runs/lfm3-s1s2 whatever the smoke's conversion check said (amendment 6): it is
    the untouched published model, and eval/base holds no s2 file when the base was converted locally.

    s1s2.py's route pairs s1 and s2 records by id and silently drops an id missing from either file, so
    every file is checked first against its split (data/lfm3/<split>.jsonl): exactly its ids, once each,
    records of the file's mode. Any gap is an error and no pair is written. meta.json, the marker that
    `route` and `gate` look for, is removed first and written last."""
    runs, data, s1s2 = runs or RUNS, data or C_DATA, s1s2 or S1S2_RUNS
    d = os.path.join(runs, "eval", "pair")
    smoke = os.path.join(runs, "smoke", "report.json")
    source = json.load(open(smoke)).get("base_source") if os.path.exists(smoke) else None
    meta = {"pair": True, "amendment": 6, "model": "pair: ft in s1, base in s2", "weights": "MLX 4-bit",
            "engine": "mlx-vlm", "revision": "local", "s1": f"ft (the adapter is System 1, {C.LOCAL_CONVERSION})",
            "s2": f"the published base, {s1s2} (System 2, untouched, {C.PUBLISHED_4BIT})",
            "base_source": source, "sources": {}, "conversions": {}}
    if os.path.exists(os.path.join(d, "meta.json")):
        os.remove(os.path.join(d, "meta.json"))
    expected = {s: [r["id"] for r in C.load_shipped(os.path.join(data, f"{s}.jsonl"))] for s in ("calib", "test")}
    origins = {"ft": (os.path.join(runs, "eval", "ft"), "eval/ft", C.LOCAL_CONVERSION, "`scripts/lfm3-ft.sh eval`"),
               "s1s2": (s1s2, s1s2, C.PUBLISHED_4BIT, "scripts/lfm3-s1s2.sh")}
    files, problems = [], []
    for split, mode, origin in PAIR_FILES:
        name = f"{split}.{mode}.jsonl"
        folder, label, conversion, step = origins[origin]
        src = os.path.join(folder, name)
        if not os.path.exists(src):
            raise SystemExit(f"{src} missing: run {step} first")
        problems += C.id_problems(f"{label}/{name}", C.read_jsonl(src), expected[split], mode)
        files.append((name, src, label, conversion))
    if problems:
        raise SystemExit("the pair is not written, its files do not hold the same questions:\n  " + "\n  ".join(problems))
    for name, src, label, conversion in files:
        C.copy_atomic(src, os.path.join(d, name))
        meta["sources"][name] = label
        meta["conversions"][name] = conversion
    with open(os.path.join(d, "meta.json.tmp"), "w") as f:
        json.dump(meta, f, indent=2)
    os.replace(os.path.join(d, "meta.json.tmp"), os.path.join(d, "meta.json"))
    return meta


def cmd_pair(_):
    print(json.dumps(build_pair(), indent=2))


def smoke_items():
    return SM.smoke_items(C.load_shipped(f"{C_DATA}/calib.jsonl"), SMOKE_N)


def max_length_rows(tok, n):
    """n rows of exactly MAX_LEN tokens (a real example repeated): the memory the longest batch needs."""
    train = C.load_train_rows()[:n]
    rows, _ = C.tokenize_mix(tok, C.build_mix(train, []))
    out = []
    for r in rows[:n]:
        p = r["completion_mask"].index(1)
        body = r["input_ids"][p:]
        ids = r["input_ids"][:p] + (body * (C.MAX_LEN // max(1, len(body)) + 1))[: C.MAX_LEN - p]
        out.append({"input_ids": ids, "completion_mask": [0] * p + [1] * (C.MAX_LEN - p), "kind": r["kind"]})
    return out


def try_lora(rev, rows, per_device, record):
    """Two optimizer steps with this batch; (seconds per step without the brake's rests, losses, mask
    check, trainer). `record` receives each step's seconds and the memory peaks (brake_callback)."""
    trainer, _, _ = C.make_trainer(C.BASE, rev, rows, f"{SMOKE}/sft-out", max_steps=2, per_device=per_device, grad_accum=1,
                                   callbacks=[brake_callback(record)])
    mask = C.check_completion_mask(trainer, rows[0])
    out = trainer.train()
    losses = [h["loss"] for h in trainer.state.log_history if "loss" in h] or [out.training_loss]
    steps = record.get("step_seconds") or [float("nan")]
    return sum(steps) / len(steps), losses, mask, trainer


def drop_weights(d):
    """Remove the *.safetensors of a directory once it has been reloaded, keeping every other file, and
    return {name: {bytes, sha256}} of what was removed. The smoke's merged model is checked, not used
    again (sft merges its own), and 16 GB of disk cannot hold it next to the base and its conversion."""
    import hashlib

    out = {}
    for name in sorted(os.listdir(d)):
        if name.endswith(".safetensors"):
            p = os.path.join(d, name)
            h = hashlib.sha256()
            with open(p, "rb") as f:
                for block in iter(lambda: f.read(1 << 24), b""):  # 16 MB at a time: the file is 3.5 GB
                    h.update(block)
            out[name] = {"bytes": os.path.getsize(p), "sha256": h.hexdigest()}
            os.remove(p)
    return out


# The memory bound of the LoRA smoke (amendment 7). MPS can be capped (PYTORCH_MPS_HIGH_WATERMARK_RATIO), MLX
# cannot (its memory limit is a guideline), and on this Mac a batch that does not fit swaps until the machine
# lags for its owner. Each batch size is tried in a child process, stopped at once when the system's swap in
# use passes SWAP_MAX_MB or the 1-minute load average passes LOAD_MAX: that batch does not fit, and amendment
# 1's fallback is taken by try_batches.
SWAP_MAX_MB = float(os.environ.get("SKY_SWAP_MAX_GB") or 3) * 1024
LOAD_MAX = float(os.environ.get("SKY_LOAD_MAX") or 8)
BATCHES = ((8, 4), (4, 8))  # amendment 1: 8 x 4, or the memory fallback 4 x 8 (32 examples per step)


def parse_swap_used(text):
    """MB of swap in use, from `sysctl -n vm.swapusage` ("total = 0,00M  used = 1234,56M  free = ...")."""
    m = re.search(r"used = ([0-9][0-9.,]*)M", text)
    if not m:
        raise ValueError(f"cannot read the swap from {text!r}")
    return float(m.group(1).replace(",", "."))


def read_swap_mb():
    return parse_swap_used(subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout)


def read_memory_level():
    """The system's free memory percentage (kern.memorystatus_level, what `memory_pressure` prints)."""
    out = subprocess.run(["sysctl", "-n", "kern.memorystatus_level"], capture_output=True, text=True).stdout.strip()
    return int(out) if out.isdigit() else None


def read_load():
    return os.getloadavg()[0]


def guarded(cmd, swap_max_mb=SWAP_MAX_MB, load_max=LOAD_MAX, period=0.5, read_swap=read_swap_mb,
            read_level=read_memory_level, read_load=read_load, popen=subprocess.Popen, sleep=time.sleep):
    """Run one attempt in a child process, stopped (killed) at once when the swap in use passes swap_max_mb
    or the 1-minute load passes load_max. Returns what was seen: returncode, whether and why it was stopped,
    the swap at the start and at its peak (MB), the highest load, the lowest free memory percentage, seconds."""
    start = peak = read_swap()
    level, load = read_level(), read_load()
    t0 = time.time()
    p = popen(cmd)
    stopped = None
    while p.poll() is None:
        sleep(period)
        now, ld, lv = read_swap(), read_load(), read_level()
        peak, load = max(peak, now), max(load, ld)
        if lv is not None:
            level = lv if level is None else min(level, lv)
        if now > swap_max_mb or ld > load_max:
            p.kill()
            p.wait()
            stopped = f"swap {now:.0f} MB" if now > swap_max_mb else f"load {ld:.2f}"
            break
    return {"returncode": p.returncode, "stopped_by_guard": stopped, "swap_start_mb": start, "swap_peak_mb": peak,
            "load_peak": load, "memory_free_pct_min": level, "seconds": time.time() - t0,
            "swap_max_mb": swap_max_mb, "load_max": load_max}


SETTLE_MB = SWAP_MAX_MB - 1024  # an attempt starts only once the swap in use is back under this
SETTLE_WAIT = 20 * 60  # seconds an attempt waits for it; then it is not run, and the smoke stops


def settle(limit_mb=SETTLE_MB, wait=SETTLE_WAIT, period=10.0, read_swap=read_swap_mb, sleep=time.sleep):
    """Wait until the swap in use is under limit_mb (macOS gives back the swap of the other applications
    slowly after a stopped attempt). Returns (settled, seconds waited, swap in MB at the end)."""
    waited, now = 0.0, read_swap()
    while now > limit_mb and waited < wait:
        sleep(period)
        waited += period
        now = read_swap()
    return now <= limit_mb, waited, now


def try_batches(sub, rev, out, run=guarded, wait=settle):
    """Amendment 1's memory rule over child attempts: batch 8, then 4 if 8 did not train (out of memory, or
    stopped by the memory bound). `sub` is the child sub-command, `out(per_device)` the file it writes.
    An attempt starts only once the swap has settled under the bound; if it does not, that attempt and the
    next are not run. Returns (attempts, the child record of the batch that trained or None)."""
    attempts = {}
    for per_device, grad_accum in BATCHES:
        path = out(per_device)
        if os.path.exists(path):
            os.remove(path)  # never read a record left by an earlier run
        settled, waited, swap = wait()
        if not settled:
            attempts[str(per_device)] = {"not_run": f"the swap stayed at {swap:.0f} MB, over {SETTLE_MB:.0f} MB, "
                                                    f"for {waited:.0f} s", "guard": None, "child": None}
            return attempts, None
        g = run([sys.executable, os.path.abspath(__file__), sub, "--revision", rev, "--per-device", str(per_device)])
        g["settle_seconds"] = waited
        child = None
        if os.path.exists(path):
            with open(path) as f:
                child = json.load(f)
        attempts[str(per_device)] = {"guard": g, "child": child}
        if child and child.get("trained") and not g["stopped_by_guard"]:
            child.update(per_device=per_device, grad_accum=grad_accum)
            return attempts, child
    return attempts, None


def cmd_smoke_lora_try(a):
    """One batch size of the peft smoke, in its own process (try_batches): two steps at the maximal length,
    then, if they trained, the adapter, the merge and the reload. Writes smoke/lora-try-<n>.json."""
    import torch
    from transformers import AutoModelForCausalLM

    n, rev = a.per_device, a.revision
    res, record = {"per_device": n, "trained": False}, {}
    res["memory"] = record
    tok = C.load_tokenizer(C.BASE, rev)
    try:
        sec, losses, mask, trainer = try_lora(rev, max_length_rows(tok, n), n, record)
        res.update(trained=True, seconds_per_batch_at_max_len=sec, losses=losses, mask_ok=mask,
                   ok=mask and len(losses) >= 1 and all(math.isfinite(x) for x in losses))
    except Exception:  # noqa: BLE001  (out of memory on MPS is a RuntimeError; anything else is reported too)
        res["error"] = traceback.format_exc()[-3000:]
    if res.get("ok"):
        try:
            trainer.save_model(f"{SMOKE}/adapter")
            del trainer
            torch.mps.empty_cache()
            merged = C.merge_and_save(C.BASE, rev, f"{SMOKE}/adapter", f"{SMOKE}/merged")
            m = AutoModelForCausalLM.from_pretrained(merged, trust_remote_code=True, dtype=torch.bfloat16).to("mps")
            on = C.chat_text(tok, {"task": "gsm8k", "question": "1+1?"}, True)
            x = torch.tensor([C.encode(tok, on)], device="mps")
            y = m.generate(input_ids=x, max_new_tokens=16, do_sample=False, temperature=None, top_p=None, top_k=None)
            res["reload_text"] = C.decode(tok, y[0, x.shape[1]:].tolist())
            res["merged_files"] = sorted(os.listdir(merged))
            res["reload_ok"] = True
            del m
            res["merged_weights_removed"] = drop_weights(merged)  # checked, never used again: the disk needs the room
        except Exception:  # noqa: BLE001
            res["reload_error"] = traceback.format_exc()[-3000:]
    with open(f"{SMOKE}/lora-try-{n}.json", "w") as f:
        json.dump(res, f, indent=2)


def cmd_smoke_lora(a):
    import torch
    from huggingface_hub import HfApi
    from transformers import AutoConfig, AutoModelForCausalLM

    rev = a.revision or HfApi().model_info(C.BASE).sha
    rep = {"base": C.BASE, "revision": rev, "versions_torch": C.versions(), "mps": torch.backends.mps.is_available(),
           "mps_high_watermark_ratio": os.environ.get("PYTORCH_MPS_HIGH_WATERMARK_RATIO"),
           "mps_low_watermark_ratio": os.environ.get("PYTORCH_MPS_LOW_WATERMARK_RATIO")}
    tok = C.load_tokenizer(C.BASE, rev)
    it = {"task": "gsm8k", "question": "1+1?"}
    on, off = C.chat_text(tok, it, True), C.chat_text(tok, it, False)
    rep["template_ok"] = on.endswith("<think>") and off.endswith("<think></think>")
    # The module names from the custom code built on the meta device: no weight is loaded, so this parent
    # holds no 3.5 GB copy of the model while its children train (run 6 of 18/11: both copies swapped).
    cfg = AutoConfig.from_pretrained(C.BASE, revision=rev, trust_remote_code=True)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True, dtype=torch.bfloat16)
    names = C.linear_names(model)
    rep["linear_names"], rep["lora_targets_match"] = names, sorted(C.LORA_TARGETS) == names
    del model
    os.makedirs(SMOKE, exist_ok=True)
    rep["memory_total"] = __import__("psutil").virtual_memory().total
    rep["attempts"], child = try_batches("smoke-lora-try", rev, lambda n: f"{SMOKE}/lora-try-{n}.json")
    if child:
        rep.update({k: child.get(k) for k in ("per_device", "grad_accum", "seconds_per_batch_at_max_len", "losses",
                                              "mask_ok", "reload_text", "merged_files", "merged_weights_removed")})
    rep["lora_ok"] = bool(child and child.get("ok") and child.get("reload_ok") and rep["lora_targets_match"]
                          and rep["template_ok"])
    with open(f"{SMOKE}/lora.json", "w") as f:
        json.dump(rep, f, indent=2)
    print(json.dumps({k: v for k, v in rep.items() if k != "attempts"}, indent=2)[:4000])
    for n, att in rep["attempts"].items():
        print(f"batch {n}:", json.dumps(att["guard"]), "trained" if (att["child"] or {}).get("trained") else "did not train")


MLX_LORA = f"{SMOKE}/mlx-lora.json"


def lora_choice(lora, mlx):
    """Amendment 1's rule, in its order: peft on MPS if its smoke validated every point, else the LoRA
    trainer of mlx-vlm if its smoke did, else nothing (stop and report). Returns the fields report.json
    carries: lora_method, per_device, grad_accum, ok."""
    if lora.get("lora_ok"):
        return {"lora_method": "peft-mps", "per_device": lora["per_device"], "grad_accum": lora["grad_accum"], "ok": True}
    if (mlx or {}).get("lora_ok"):
        return {"lora_method": "mlx-vlm", "per_device": mlx["per_device"], "grad_accum": mlx["grad_accum"], "ok": True}
    return {"lora_method": None, "per_device": None, "grad_accum": None, "ok": False}


def fits(peak, limit):
    """MLX's memory limit is only a guideline (beyond it, MLX swaps): a batch fits when the peak of its two
    steps stays within the device's recommended working set, the cap PyTorch's MPS run had (amendment 7)."""
    return peak is not None and limit is not None and peak <= limit


def mlx_optimizer(total):
    """AdamW as TRL's SFTTrainer has it (§4: lr 1e-4, cosine, warmup 3 % of the steps; betas 0.9 and
    0.999, eps 1e-8, no weight decay, gradients clipped at norm 1.0), for mlx-vlm's trainer."""
    import mlx.optimizers as optim

    warm = math.ceil(0.03 * total)
    sched = optim.cosine_decay(1e-4, max(1, total - warm))
    if warm:
        sched = optim.join_schedules([optim.linear_schedule(0.0, 1e-4, warm), sched], [warm])

    class ClippedAdamW(optim.AdamW):
        def apply_gradients(self, gradients, parameters):
            gradients, _ = optim.clip_grad_norm(gradients, 1.0)
            return super().apply_gradients(gradients, parameters)

    return ClippedAdamW(learning_rate=sched, betas=[0.9, 0.999], eps=1e-8, weight_decay=0.0)


def mlx_rows(rows):
    import numpy as np

    return [{"input_ids": np.array(r["input_ids"], dtype=np.int32),
             "attention_mask": np.ones(len(r["input_ids"]), dtype=np.int32),
             "completion_mask": np.array(r["completion_mask"], dtype=np.int32)} for r in rows]


def mlx_mask_ok(row, pad_id):
    """mlx-vlm's collator keeps the completion mask of a row as it is (the loss reads completion_mask[1:]
    and attention_mask[1:]): the prompt is masked and the completion is not."""
    import numpy as np
    from mlx_vlm.trainer.vlm.sft.runtime import collate_vlm_batch

    b = collate_vlm_batch([row], C.MAX_LEN, pad_to_multiple=32, pad_token_id=pad_id)
    n = len(row["input_ids"])
    got = np.array(b["completion_mask"])[0]
    return (bool((got[:n] == row["completion_mask"]).all()) and not got[n:].any()
            and bool((np.array(b["input_ids"])[0][:n] == row["input_ids"]).all()))


def mlx_linear_names(model):
    import mlx.nn as nn

    return sorted({n.split(".")[-1] for n, m in model.named_modules()
                   if isinstance(m, (nn.Linear, nn.QuantizedLinear)) and not n.endswith("lm_head")})


def mlx_brake_callback(record):
    """The thermal brake between two optimizer steps, and the smoke's record of each step."""
    import mlx.core as mx
    from mlx_vlm.trainer.common.callbacks import TrainingCallback

    class Brake(TrainingCallback):
        def on_train_loss_report(self, info):
            record.setdefault("step_seconds", []).append(info["step_time"])
            record.setdefault("losses", []).append(info["train_loss"])
            record["peak"] = max(record.get("peak", 0), mx.get_peak_memory())
            S.rest(info["step_time"])

    return Brake()


def mlx_fuse_and_save(model, snap, out):
    """Fuse the LoRA layers into the bf16 weights and write a directory mlx_vlm.load reads: the weights,
    the base's config, custom code and tokenizer files (unmodified copies)."""
    import glob

    from mlx.utils import tree_unflatten
    from mlx_vlm.utils import save_weights

    fused = [(n, m.fuse()) for n, m in model.named_modules() if hasattr(m, "fuse")]
    model.update_modules(tree_unflatten(fused))
    os.makedirs(out, exist_ok=True)
    save_weights(out, model, donate_weights=True)
    for pattern in ("*.py", "*.json", "*.jinja", "LICENSE"):
        for f in glob.glob(os.path.join(snap, pattern)):
            if os.path.basename(f) != "model.safetensors.index.json":
                shutil.copy(f, os.path.join(out, os.path.basename(f)))
    return len(fused)


def cmd_smoke_mlx_lora_try(a):
    """One batch size of the mlx-vlm smoke, in its own process (try_batches): two steps at the maximal
    length, then, if they trained within the recommended working set, the merge and the reload. Writes
    smoke/mlx-lora-try-<n>.json."""
    import mlx.core as mx
    from huggingface_hub import snapshot_download
    from mlx_vlm import load
    from mlx_vlm.trainer import Trainer
    from mlx_vlm.trainer.common.model import prepare_model_for_training
    from mlx_vlm.trainer.vlm.config import VLMTrainingArgs

    n, rev = a.per_device, a.revision
    snap = snapshot_download(C.BASE, revision=rev)
    limit = mx.device_info()["max_recommended_working_set_size"]
    mx.set_memory_limit(limit)
    record = {}
    res = {"per_device": n, "trained": False, "memory_limit": limit, "memory": record}
    tok = C.load_tokenizer(C.BASE, rev)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    try:
        model, processor = load(snap)
        names = mlx_linear_names(model)
        res["linear_names"], res["lora_targets_match"] = names, sorted(C.LORA_TARGETS) == names
        prepare_model_for_training(model, train_type="lora", lora_rank=32, lora_alpha=64.0, lora_dropout=0.05,
                                   target_modules=list(C.LORA_TARGETS), verbose=False)
        res["lora_layers"] = sum(type(m).__name__ == "LoRALinear" for _, m in model.named_modules())
        rows = mlx_rows(max_length_rows(tok, n))
        res["mask_ok"] = mlx_mask_ok(rows[0], pad)
        args = VLMTrainingArgs(batch_size=n, iters=2, gradient_accumulation_steps=1, max_seq_length=C.MAX_LEN,
                               train_on_completions=True, grad_checkpoint=True, steps_per_report=1, steps_per_eval=None,
                               steps_per_save=10**9, adapter_file=f"{SMOKE}/mlx-adapter/adapters.safetensors")
        trainer = Trainer(model, args, train_dataset=rows, processing_class=processor, optimizer=mlx_optimizer(2),
                          prepared=True, seed=C.SEED, training_callback=mlx_brake_callback(record))
        trainer.train()
        record["peak"] = max(record.get("peak", 0), mx.get_peak_memory())
        record["fits"] = fits(record["peak"], limit)
        losses = record.get("losses", [])
        res.update(trained=record["fits"], losses=losses,
                   seconds_per_batch_at_max_len=sum(record["step_seconds"]) / max(1, len(record["step_seconds"])),
                   ok=record["fits"] and res["mask_ok"] and len(losses) == 2 and all(math.isfinite(x) for x in losses))
        if not record["fits"]:
            res["error"] = f"peak {record['peak']} bytes over the recommended working set {limit}"
    except Exception:  # noqa: BLE001
        res["error"] = traceback.format_exc()[-3000:]
    if res.get("ok"):
        try:
            merged = f"{SMOKE}/mlx-merged"
            res["fused_layers"] = mlx_fuse_and_save(model, snap, merged)
            del trainer, model
            mx.clear_cache()
            m = LocalModel(merged)
            on = C.chat_text(tok, {"task": "gsm8k", "question": "1+1?"}, False)
            res["reload_text"] = "".join(t for t, _, _ in m.stream(on, 16, C.SEED, temperature=0.0))
            res["merged_files"] = sorted(os.listdir(merged))
            res["reload_ok"] = True
            del m
            res["merged_weights_removed"] = drop_weights(merged)  # checked, never used again: the disk needs the room
        except Exception:  # noqa: BLE001
            res["reload_error"] = traceback.format_exc()[-3000:]
    with open(f"{SMOKE}/mlx-lora-try-{n}.json", "w") as f:
        json.dump(res, f, indent=2)


def cmd_smoke_mlx_lora(_):
    """Amendment 1, rule 2: when peft on MPS did not validate, the LoRA trainer of mlx-vlm on the bf16
    weights, at the same hyperparameters, validates the same points: the LoRA names, the prompt mask, two
    steps of finite loss on a batch of the maximal length (8, or the memory fallback 4), the merge and the
    reload. Writes smoke/mlx-lora.json."""
    with open(f"{SMOKE}/lora.json") as f:
        lora = json.load(f)
    if lora.get("lora_ok"):
        with open(MLX_LORA, "w") as f:
            json.dump({"needed": False, "reason": "peft on MPS validated (amendment 1, rule 1)"}, f, indent=2)
        print("smoke-mlx-lora: not needed, peft on MPS validated")
        return
    import mlx.core as mx

    rev = lora["revision"]
    rep = {"needed": True, "base": C.BASE, "revision": rev, "versions": C.versions(), "mlx": mx.__version__,
           "lora": {"r": 32, "alpha": 64, "dropout": 0.05, "targets": list(C.LORA_TARGETS)}}
    rep["attempts"], child = try_batches("smoke-mlx-lora-try", rev, lambda n: f"{SMOKE}/mlx-lora-try-{n}.json")
    first = next((att["child"] for att in rep["attempts"].values() if att["child"]), None) or {}
    rep["linear_names"], rep["lora_targets_match"] = first.get("linear_names"), first.get("lora_targets_match")
    if child:
        rep.update({k: child.get(k) for k in ("per_device", "grad_accum", "seconds_per_batch_at_max_len", "losses",
                                              "mask_ok", "lora_layers", "fused_layers", "reload_text", "merged_files",
                                              "merged_weights_removed", "memory_limit")})
    rep["lora_ok"] = bool(child and child.get("ok") and child.get("reload_ok") and child.get("lora_targets_match")
                          and lora.get("template_ok"))
    with open(MLX_LORA, "w") as f:
        json.dump(rep, f, indent=2)
    print(json.dumps({k: v for k, v in rep.items() if k != "attempts"}, indent=2)[:4000])
    for n, att in rep["attempts"].items():
        print(f"batch {n}:", json.dumps(att["guard"]), "trained" if (att["child"] or {}).get("trained") else "did not train")


def s1_prefix(m, tok, items):
    out = []
    for it in items:
        r = S.gen_s1(m, user_item(it), C.qseed(it["id"]))
        r["ids"] = C.encode(tok, r["text"])
        r["pred"], _ = C.grade(it["task"], r["text"], it["answer"])
        r["id"] = it["id"]
        out.append(r)
    return out


def cmd_smoke_mlx(_):
    with open(f"{SMOKE}/lora.json") as f:
        rep = json.load(f)
    items = smoke_items()
    published = LocalModel()
    t0 = time.time()
    a = s1_prefix(published, published.tok, items)
    rep["s1_tok_per_s"] = sum(r["tokens"] for r in a) / max(1e-9, time.time() - t0)
    t0 = time.time()
    s2 = [S.gen_s2(published, user_item(it), C.qseed(it["id"]), 256) for it in items[:4]]
    rep["s2_256_tok_per_s"] = sum(r["tokens"] for r in s2) / max(1e-9, time.time() - t0)
    convert(C.BASE, BASE_MLX, rep["revision"])
    converted = LocalModel(BASE_MLX)
    b = s1_prefix(converted, published.tok, items)
    tok_agree = sum(SM.prefix_agreement(x["ids"], y["ids"]) for x, y in zip(a, b)) / len(a)
    ans_agree = sum(x["pred"] == y["pred"] for x, y in zip(a, b)) / len(a)
    rep.update(token_agreement=tok_agree, answer_agreement=ans_agree, engine="mlx-vlm", versions_mlx=C.versions())
    rep["base_source"] = "published" if tok_agree >= SM.AGREE_MIN and ans_agree >= SM.AGREE_MIN else "converted"
    mlx = None
    if os.path.exists(MLX_LORA):
        with open(MLX_LORA) as f:
            mlx = json.load(f)
        rep["mlx_lora"] = {k: v for k, v in mlx.items() if k not in ("linear_names",)}
    rep.update(lora_choice(rep, mlx))
    with open(f"{SMOKE}/report.json", "w") as f:
        json.dump(rep, f, indent=2)
    print(json.dumps({k: v for k, v in rep.items() if k not in ("losses", "linear_names")}, indent=2)[:4000])
    print("REVISION", rep["revision"])
    print("LORA", rep["lora_method"], rep.get("per_device"), rep.get("grad_accum"))
    print("BASE_SOURCE", rep["base_source"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("smoke-lora")
    p.add_argument("--revision")
    for name in ("smoke-lora-try", "smoke-mlx-lora-try"):  # one batch size, in a child of try_batches
        p = sub.add_parser(name)
        p.add_argument("--revision", required=True)
        p.add_argument("--per-device", type=int, required=True)
    sub.add_parser("smoke-mlx-lora")  # amendment 1, rule 2: only does something when peft on MPS failed
    sub.add_parser("smoke-mlx")
    sub.add_parser("gen")  # amendment 5: writes the not-run marker, nothing is generated
    p = sub.add_parser("sft")
    p.add_argument("--revision", required=True)
    sub.add_parser("convert")
    p = sub.add_parser("eval")
    p.add_argument("--tag", choices=("base", "ft"), required=True)
    p.add_argument("--limit", type=int, default=0, help="first N questions per split (never for the run)")
    sub.add_parser("pair")
    a = ap.parse_args()
    {"smoke-lora": cmd_smoke_lora, "smoke-lora-try": cmd_smoke_lora_try, "smoke-mlx-lora": cmd_smoke_mlx_lora,
     "smoke-mlx-lora-try": cmd_smoke_mlx_lora_try, "smoke-mlx": cmd_smoke_mlx, "gen": cmd_gen, "sft": cmd_sft, "convert": cmd_convert,
     "eval": cmd_eval, "pair": cmd_pair}[a.cmd](a)


if __name__ == "__main__":
    main()
