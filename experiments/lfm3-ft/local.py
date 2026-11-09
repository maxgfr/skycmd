#!/usr/bin/env python3
"""The LFM3-1.7B fine-tuning experiment on the M5 (docs/lfm3-ft-prereg.md, amendment 1).

scripts/lfm3-ft.sh runs each sub-command in its own uv environment: mlx-vlm for generation, the torch
stack for the LoRA (they cannot share one). Generation reuses experiments/lfm3-s1s2/s1s2.py (Model,
gen_s1, gen_s2, rest: the same prompts, sampling and seeds as the router's run), and the selection,
the mix and the trainer come from common.py. Everything stays under runs/lfm3-ft and models/lfm3-ft.

  smoke-lora   revision, template, LoRA names, 2 steps at maximal length, merge and reload  (torch)
  smoke-mlx    throughput, the local MLX conversion against the published base; report.json (mlx)
  gen          4 samples of s2-1024 on 1,000 train questions per task                       (mlx)
  sft          the LoRA, merged into bf16 weights                                           (torch)
  convert      the merged model to MLX 4 bits                                               (mlx)
  eval         s1, s2-256, s2-1024 on calib and test, s1 on MMLU-Pro-500, for --tag base|ft (mlx)

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
SAVE_EVERY = 100  # optimizer steps between two technical checkpoints


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


def cmd_gen(a):
    rows, stats = train_rows()
    rows = C.hf_fallback_subset(rows)
    if a.limit:
        rows = rows[: a.limit]
    path = f"{RUNS}/gen/samples" + ("-limit" if a.limit else "") + ".jsonl"
    done = {(r["id"], r["k"]) for r in C.read_jsonl(path)}
    todo = [(r, k) for r in rows for k in range(C.N_SAMPLES) if (r["id"], k) not in done]
    print(f"{len(todo)} samples to generate ({len(done)} done) -> {path}", flush=True)
    if todo:
        m = LocalModel()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        t_all, n_tok = time.time(), 0
        with open(path, "a") as f:
            for i, (it, k) in enumerate(todo):
                seed = C.qseed(it["id"]) + C.SAMPLE_STRIDE * k
                t0 = time.time()
                r = S.gen_s2(m, user_item(it), seed, a.budget)
                r = sample_record(r, it, k, seed, time.time() - t0)
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                f.flush()
                S.rest(r["seconds"])
                n_tok += r["tokens"]
                if (i + 1) % 10 == 0 or i + 1 == len(todo):
                    el = time.time() - t_all
                    print(f"  {i + 1}/{len(todo)}  {n_tok / el:.0f} tok/s  {el / (i + 1):.1f} s/sample", flush=True)
    samples = C.read_jsonl(path)
    targets = C.distill_targets(samples)
    stats.update(sampled_questions=len(rows), samples=len(samples), correct=sum(s["correct"] for s in samples),
                 forced=sum(s["forced"] for s in samples), budget=a.budget, samples_per_question=C.N_SAMPLES,
                 questions_with_nothink=sum(1 for t in targets.values() if t["nothink"]),
                 questions_with_think=sum(1 for t in targets.values() if t["think"]), engine="mlx-vlm",
                 weights=f"{S.MODEL}@{S.MODEL_REV}", versions=C.versions())
    with open(f"{RUNS}/gen/stats" + ("-limit" if a.limit else "") + ".json", "w") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2))


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


def cmd_sft(a):
    from transformers.trainer_utils import get_last_checkpoint

    per_device, grad_accum = lora_settings()
    rows, excl = train_rows()
    samples = C.read_jsonl(f"{RUNS}/gen/samples.jsonl")
    if not samples:
        raise SystemExit(f"no {RUNS}/gen/samples.jsonl: run `scripts/lfm3-ft.sh gen` first")
    mix = C.build_mix(rows, samples)
    tok = C.load_tokenizer(C.BASE, a.revision)
    tok_rows, dropped = C.tokenize_mix(tok, mix)
    kinds = {}
    for r in tok_rows:
        kinds[r["kind"]] = kinds.get(r["kind"], 0) + 1
    stats = {"exclusions": excl, "samples": len(samples), "examples": kinds, "dropped_over_max_len": dropped,
             "tokens": sum(len(r["input_ids"]) for r in tok_rows), "per_device": per_device, "grad_accum": grad_accum}
    print(json.dumps(stats, indent=2), flush=True)
    out_dir = f"{RUNS}/sft/checkpoints"
    trainer, tok, total = C.make_trainer(C.BASE, a.revision, tok_rows, out_dir, per_device=per_device, grad_accum=grad_accum,
                                         save_steps=SAVE_EVERY, callbacks=[brake_callback()])
    if not C.check_completion_mask(trainer, tok_rows[0]):
        raise SystemExit("the collator does not mask the prompt as expected: stop (completion-only loss)")
    last = get_last_checkpoint(out_dir) if os.path.isdir(out_dir) else None
    print(f"{len(tok_rows)} examples, {total} optimizer steps, resuming from {last}", flush=True)
    t0 = time.time()
    out = trainer.train(resume_from_checkpoint=last)
    adapter = f"{RUNS}/sft/adapter"
    trainer.save_model(adapter)
    C.merge_and_save(C.BASE, a.revision, adapter, MERGED)
    result = {"base": C.BASE, "base_revision": a.revision, "steps": total, "train_seconds_with_rests": time.time() - t0,
              "resumed_from": last, "train_metrics": out.metrics, "log_history": trainer.state.log_history, "mix": stats,
              "merged": MERGED, "versions": C.versions()}
    with open(f"{RUNS}/sft/result.json", "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps({k: v for k, v in result.items() if k != "log_history"}, indent=2))


def convert(src, dst, revision=None):
    """The MLX 4-bit conversion of amendment 1 (the same command for the base and the fine-tuned model)."""
    if os.path.exists(os.path.join(dst, "config.json")):
        print(f"{dst} exists, kept", flush=True)
        return
    cmd = [sys.executable, "-m", "mlx_vlm.convert", "--hf-path", src, "--mlx-path", dst, "-q", "--q-bits", "4",
           "--q-mode", "affine", "--trust-remote-code"] + (["--revision", revision] if revision else [])
    print(" ".join(cmd), flush=True)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    subprocess.run(cmd, check=True)


def cmd_convert(_):
    if not os.path.exists(os.path.join(MERGED, "config.json")):
        raise SystemExit(f"{MERGED} missing: run `scripts/lfm3-ft.sh sft` first")
    convert(MERGED, FT_MLX)


def base_source():
    with open(f"{SMOKE}/report.json") as f:
        return json.load(f)["base_source"]


def copy_base_generations(d):
    """The router's run is the base model's evaluation (amendment 1): same prompts, seeds and engine."""
    names = [f"{s}.{m}.jsonl" for s in ("calib", "test") for m in ("s1", "s2-256", "s2-1024")]
    missing = [n for n in names if not os.path.exists(f"{S1S2_RUNS}/{n}")]
    if missing:
        raise SystemExit(f"{S1S2_RUNS} lacks {missing}: run scripts/lfm3-s1s2.sh first")
    os.makedirs(d, exist_ok=True)
    for n in names:
        if not os.path.exists(f"{d}/{n}"):
            shutil.copy(f"{S1S2_RUNS}/{n}", f"{d}/{n}")
    print(f"base: the generations of {S1S2_RUNS} are copied into {d}", flush=True)
    return names


def cmd_eval(a):
    d = f"{RUNS}/eval/{a.tag}" + ("-limit" if a.limit else "")
    data = splits(a.limit)
    data["mmlu"] = C.mmlu_items()[: a.limit] if a.limit else C.mmlu_items()
    jobs = [(s, m) for s in ("calib", "test") for m in ("s1", "s2-256", "s2-1024")] + [("mmlu", "s1")]
    source, path, label = None, None, S.MODEL
    if a.tag == "ft":
        path, label = FT_MLX, f"{C.FT_REPO} (local)"
    else:
        source = base_source()
        if source == "published" and not a.limit:
            copy_base_generations(d)
            jobs = [("mmlu", "s1")]
        elif source == "converted":
            path, label = BASE_MLX, f"{C.BASE} (converted locally)"
    m = None
    meta = {"model": label, "revision": S.MODEL_REV if path is None else "local", "tag": a.tag, "engine": "mlx-vlm",
            "weights": "MLX 4-bit", "base_source": source, "versions": C.versions(),
            "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "files": {}}
    for split, mode in jobs:
        out = f"{d}/{split}.{mode}.jsonl"
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
    for name in sorted(os.listdir(d)):
        if name.endswith(".jsonl"):
            rows = C.read_jsonl(f"{d}/{name}")
            meta["files"][name] = {"n": len(rows), "acc": sum(bool(r["correct"]) for r in rows) / max(1, len(rows))}
    meta["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with open(f"{d}/meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2))


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
    p = sub.add_parser("gen")
    p.add_argument("--budget", type=int, default=1024)
    p.add_argument("--limit", type=int, default=0, help="first N questions (never for the run)")
    p = sub.add_parser("sft")
    p.add_argument("--revision", required=True)
    sub.add_parser("convert")
    p = sub.add_parser("eval")
    p.add_argument("--tag", choices=("base", "ft"), required=True)
    p.add_argument("--limit", type=int, default=0, help="first N questions per split (never for the run)")
    a = ap.parse_args()
    {"smoke-lora": cmd_smoke_lora, "smoke-mlx": cmd_smoke_mlx, "gen": cmd_gen, "sft": cmd_sft, "convert": cmd_convert,
     "eval": cmd_eval}[a.cmd](a)


if __name__ == "__main__":
    main()
