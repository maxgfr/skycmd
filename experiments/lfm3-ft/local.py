#!/usr/bin/env python3
"""The LFM3-1.7B fine-tuning experiment on the M5 (docs/lfm3-ft-prereg.md, amendments 1, 5 and 6).

scripts/lfm3-ft.sh runs each sub-command in its own uv environment: mlx-vlm for generation, the torch
stack for the LoRA (they cannot share one). Generation reuses experiments/lfm3-s1s2/s1s2.py (Model,
gen_s1, gen_s2, rest: the same prompts, sampling and seeds as the router's run), and the selection,
the mix and the trainer come from common.py. Everything stays under runs/lfm3-ft and models/lfm3-ft.

  smoke-lora   revision, template, LoRA names, 2 steps at maximal length, merge and reload  (torch)
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
import os
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


def brake_callback():
    """The thermal brake between two optimizer steps (s1s2.py's rest)."""
    from transformers import TrainerCallback

    class Brake(TrainerCallback):
        def on_step_begin(self, args, state, control, **kw):
            self.t0 = time.time()

        def on_step_end(self, args, state, control, **kw):
            S.rest(time.time() - self.t0)

    return Brake()


def lora_settings():
    """The batch the smoke validated (amendment 1: 8 x 4, or the memory fallback 4 x 8)."""
    path = f"{SMOKE}/report.json"
    if not os.path.exists(path):
        raise SystemExit(f"{path} missing: run `scripts/lfm3-ft.sh smoke` first")
    with open(path) as f:
        rep = json.load(f)
    if not rep.get("ok") or rep.get("lora_method") != "peft-mps":
        raise SystemExit(f"{path}: the smoke did not validate peft on MPS (amendment 1: stop and report)")
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


def try_lora(rev, rows, per_device):
    """Two optimizer steps with this batch; (seconds per step, losses, trainer)."""
    trainer, _, _ = C.make_trainer(C.BASE, rev, rows, f"{SMOKE}/sft-out", max_steps=2, per_device=per_device, grad_accum=1,
                                   callbacks=[brake_callback()])
    mask = C.check_completion_mask(trainer, rows[0])
    t0 = time.time()
    out = trainer.train()
    losses = [h["loss"] for h in trainer.state.log_history if "loss" in h] or [out.training_loss]
    return (time.time() - t0) / 2, losses, mask, trainer


def cmd_smoke_lora(a):
    import math

    import torch
    from huggingface_hub import HfApi
    from transformers import AutoModelForCausalLM

    rev = a.revision or HfApi().model_info(C.BASE).sha
    rep = {"base": C.BASE, "revision": rev, "versions_torch": C.versions(), "mps": torch.backends.mps.is_available()}
    tok = C.load_tokenizer(C.BASE, rev)
    it = {"task": "gsm8k", "question": "1+1?"}
    on, off = C.chat_text(tok, it, True), C.chat_text(tok, it, False)
    rep["template_ok"] = on.endswith("<think>") and off.endswith("<think></think>")
    model = AutoModelForCausalLM.from_pretrained(C.BASE, revision=rev, trust_remote_code=True, dtype=torch.bfloat16)
    names = C.linear_names(model)
    rep["linear_names"], rep["lora_targets_match"] = names, sorted(C.LORA_TARGETS) == names
    del model
    os.makedirs(SMOKE, exist_ok=True)
    ok = False
    for per_device, grad_accum in ((8, 4), (4, 8)):  # amendment 1: the memory fallback keeps 32 examples per step
        try:
            sec, losses, mask, trainer = try_lora(rev, max_length_rows(tok, per_device), per_device)
            rep.update(per_device=per_device, grad_accum=grad_accum, seconds_per_batch_at_max_len=sec, losses=losses, mask_ok=mask)
            ok = mask and all(math.isfinite(x) for x in losses)
            break
        except Exception:  # noqa: BLE001  (out of memory on MPS is a RuntimeError; anything else is reported too)
            rep[f"error_{per_device}"] = traceback.format_exc()[-3000:]
            trainer = None
            torch.mps.empty_cache()
    reload_ok = False
    if ok:
        try:
            trainer.save_model(f"{SMOKE}/adapter")
            del trainer
            torch.mps.empty_cache()
            merged = C.merge_and_save(C.BASE, rev, f"{SMOKE}/adapter", f"{SMOKE}/merged")
            m = AutoModelForCausalLM.from_pretrained(merged, trust_remote_code=True, dtype=torch.bfloat16).to("mps")
            x = torch.tensor([C.encode(tok, on)], device="mps")
            y = m.generate(input_ids=x, max_new_tokens=16, do_sample=False, temperature=None, top_p=None, top_k=None)
            rep["reload_text"] = C.decode(tok, y[0, x.shape[1]:].tolist())
            reload_ok = True
        except Exception:  # noqa: BLE001
            rep["reload_error"] = traceback.format_exc()[-3000:]
    rep["lora_ok"] = bool(ok and reload_ok and rep["lora_targets_match"] and rep["template_ok"])
    with open(f"{SMOKE}/lora.json", "w") as f:
        json.dump(rep, f, indent=2)
    print(json.dumps(rep, indent=2)[:4000])


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
    rep["lora_method"] = "peft-mps" if rep["lora_ok"] else None
    rep["ok"] = bool(rep["lora_ok"])
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
    {"smoke-lora": cmd_smoke_lora, "smoke-mlx": cmd_smoke_mlx, "gen": cmd_gen, "sft": cmd_sft, "convert": cmd_convert,
     "eval": cmd_eval, "pair": cmd_pair}[a.cmd](a)


if __name__ == "__main__":
    main()
