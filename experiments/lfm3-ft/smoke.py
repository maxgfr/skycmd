# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = [
#   "torch==2.14.0",
#   "transformers==5.12.1",
#   "trl==1.17.1",
#   "peft==0.21.2",
#   "accelerate==1.15.0",
#   "datasets==5.1.0",
#   "huggingface_hub>=1.31,<2",
# ]
# ///
"""Smoke test of the LFM3 fine-tuning experiment on Hugging Face Jobs, before any other job
(docs/lfm3-ft-prereg.md §2 and §8). About 20 calibration questions, never a test question.

1. Resolves and records the base model revision (the sha every later job pins).
2. vLLM 0.33.0, in its own uv environment (it cannot share one with transformers 5.12.1): does it load
   LFM3, and run s1, s2-256 and s2-1024 on the 20 questions?
3. transformers 5.12.1 (the fallback engine): the same three modes; the chat template with thinking on
   and off; the nn.Linear names against LORA_TARGETS.
4. TRL / peft on the custom code: the collator masks the prompt, two LoRA steps give a finite loss, the
   merged model is saved with its custom code and reloads.
5. The engine decision (pre-registered): vLLM if it loaded, ran the three modes, and agrees with
   transformers on s1 (greedy) for at least 0.8 of the first 64 tokens on average and on at least 0.8
   of the s1 answers; the transformers fallback otherwise.

Writes smoke/report.json to the private work repo and prints it.

  smoke.py --common common.py --calib calib.jsonl [--revision <sha>]
"""
import argparse
import json
import math
import os
import subprocess
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

LOCAL = "/tmp/lfm3-ft/smoke"
VLLM_ENV = ["vllm==0.33.0", "datasets==5.1.0", "huggingface_hub>=1.31,<2"]
AGREE_TOKENS = 64
AGREE_MIN = 0.8


def smoke_items(calib, n):
    half = n // 2
    return [r for r in calib if r["task"] == "gsm8k"][:half] + [r for r in calib if r["task"] == "math"][: n - half]


def mode_stats(recs, seconds):
    n = len(recs)
    return {"n": n, "acc": sum(r["correct"] for r in recs) / n,
            "length_stops": sum(r["finish"] == "length" for r in recs) / n,
            "mean_tokens": sum(r["tokens"] for r in recs) / n,
            "mean_think_tokens": sum(r["think_tokens"] for r in recs) / n,
            "forced": sum(bool(r.get("forced")) for r in recs) / n,
            "tok_per_s": sum(r["tokens"] for r in recs) / max(1e-9, seconds)}


def run_modes(engine, items):
    out, recs = {}, {}
    for mode in ("s1", "s2-256", "s2-1024"):
        t0 = time.time()
        r = C.run_s1(engine, items, keep_ids=True) if mode == "s1" else C.run_s2(engine, items, int(mode[3:]))
        dt = time.time() - t0
        recs[mode] = C.finish_records(items, r, mode, dt)
        out[mode] = mode_stats(recs[mode], dt)
        print(mode, json.dumps(out[mode]), flush=True)
    return out, recs


def vllm_check(a):
    """Runs inside the vLLM environment; writes vllm.json."""
    res = {"versions": C.versions()}
    try:
        items = smoke_items(C.load_shipped(a.calib), a.n)
        t0 = time.time()
        engine = C.VLLMEngine(C.BASE, a.revision)
        res["load_seconds"] = time.time() - t0
        res["stats"], recs = run_modes(engine, items)
        res["records"] = recs
        res["loaded"] = True
    except Exception:
        res["loaded"] = res.get("loaded", False)
        res["error"] = traceback.format_exc()[-4000:]
    with open(a.out, "w") as f:
        json.dump(res, f)


def prefix_agreement(a_ids, b_ids, k=AGREE_TOKENS):
    a_ids, b_ids = a_ids[:k], b_ids[:k]
    m = min(len(a_ids), len(b_ids))
    if m == 0:
        return 1.0 if len(a_ids) == len(b_ids) else 0.0
    n = 0
    while n < m and a_ids[n] == b_ids[n]:
        n += 1
    return n / m


def decide(vllm, hf_s1):
    """The pre-registered engine decision from the two engines' s1 records."""
    if not vllm.get("loaded") or "records" not in vllm or set(vllm["records"]) != {"s1", "s2-256", "s2-1024"}:
        return "hf", {"reason": "vLLM did not load or did not run the three modes"}
    v = {r["id"]: r for r in vllm["records"]["s1"]}
    pairs = [(v[r["id"]], r) for r in hf_s1 if r["id"] in v]
    tok_agree = sum(prefix_agreement(x["ids"], y["ids"]) for x, y in pairs) / max(1, len(pairs))
    ans_agree = sum(x["pred"] == y["pred"] for x, y in pairs) / max(1, len(pairs))
    detail = {"token_agreement": tok_agree, "answer_agreement": ans_agree, "n": len(pairs)}
    return ("vllm" if tok_agree >= AGREE_MIN and ans_agree >= AGREE_MIN else "hf"), detail


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--common", help="common.py, shipped with the job (unused otherwise)")
    ap.add_argument("--calib", required=True)
    ap.add_argument("--revision", help="default: the current main of the base repo, resolved here")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--work", default=C.WORK)
    ap.add_argument("--vllm-check", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--out", help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.vllm_check:
        return vllm_check(a)

    from huggingface_hub import HfApi

    os.makedirs(LOCAL, exist_ok=True)
    rev = a.revision or HfApi().model_info(C.BASE).sha
    report = {"base": C.BASE, "revision": rev, "versions_hf": C.versions(), "data_sha256": C.sha256(a.calib)}
    print(f"{C.BASE} revision {rev}", flush=True)

    # 2. vLLM first, alone on the GPU, in its own environment.
    vout = f"{LOCAL}/vllm.json"
    cmd = ["uv", "run", "--isolated", "--no-project", "--python", f"{sys.version_info[0]}.{sys.version_info[1]}"]
    for d in VLLM_ENV:
        cmd += ["--with", d]
    cmd += ["python", os.path.abspath(__file__), "--calib", a.calib, "--revision", rev, "--n", str(a.n),
            "--vllm-check", "--out", vout]
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True)
    vllm = json.load(open(vout)) if os.path.exists(vout) else {"loaded": False}
    vllm["returncode"], vllm["seconds"] = p.returncode, time.time() - t0
    if p.returncode != 0 or not vllm.get("loaded"):
        vllm["stderr_tail"] = p.stderr[-4000:]
    report["vllm"] = {k: v for k, v in vllm.items() if k != "records"}
    print("vllm:", json.dumps(report["vllm"])[:3000], flush=True)

    # 3. transformers: template, modules, the three modes.
    tok = C.load_tokenizer(C.BASE, rev)
    it = {"task": "gsm8k", "question": "1+1?"}
    on, off = C.chat_text(tok, it, True), C.chat_text(tok, it, False)
    ids_on = C.encode(tok, on)
    report["template"] = {
        "thinking_on_ends_with_think": on.endswith("<think>"),
        "thinking_off_ends_with_empty_think": off.endswith("<think></think>"),
        "bos_once": ids_on.count(tok.bos_token_id) == 1 if tok.bos_token_id is not None else None,
        "eos_token": tok.eos_token, "think_close_ids": C.encode(tok, C.THINK_CLOSE),
        "prompt_on": on, "prompt_off": off,
    }
    items = smoke_items(C.load_shipped(a.calib), a.n)
    engine = C.HFEngine(C.BASE, rev, batch_size=32)
    names = C.linear_names(engine.model)
    report["linear_names"] = names
    report["lora_targets_match"] = sorted(C.LORA_TARGETS) == names
    report["hf"], hf_recs = run_modes(engine, items)
    report["examples"] = {m: {k: hf_recs[m][0].get(k) for k in ("id", "think", "text", "pred", "correct", "finish")}
                          for m in hf_recs}
    del engine
    import torch

    torch.cuda.empty_cache()

    # 4. TRL / peft: mask, two steps, merge, reload.
    try:
        rows = [r for r in C.load_train_rows() if r["id"] not in {c["id"] for c in items}][:8]
        mix = C.build_mix(rows, [])
        mix.append({"id": "think-example", "kind": "distill_think", "thinking": True, "question": rows[0]["question"],
                    "task": rows[0]["task"], "completion": "Let me add.</think>" + rows[0]["solution"]})
        tok_rows, _ = C.tokenize_mix(tok, mix)
        trainer, _, _ = C.make_trainer(C.BASE, rev, tok_rows, f"{LOCAL}/sft-out", max_steps=2, per_device=2, grad_accum=1)
        mask_ok = C.check_completion_mask(trainer, tok_rows[0]) and C.check_completion_mask(trainer, tok_rows[-1])
        out = trainer.train()
        trainer.save_model(f"{LOCAL}/adapter")
        losses = [h["loss"] for h in trainer.state.log_history if "loss" in h] or [out.training_loss]
        del trainer
        torch.cuda.empty_cache()
        merged = C.merge_and_save(C.BASE, rev, f"{LOCAL}/adapter", f"{LOCAL}/merged")
        from transformers import AutoModelForCausalLM

        m = AutoModelForCausalLM.from_pretrained(merged, trust_remote_code=True, dtype=torch.bfloat16, device_map="cuda")
        x = torch.tensor([ids_on], device="cuda")
        y = m.generate(input_ids=x, max_new_tokens=16, do_sample=False, temperature=None, top_p=None, top_k=None)
        report["trl"] = {"ok": mask_ok and all(math.isfinite(l) for l in losses), "mask_ok": mask_ok,
                         "losses": losses, "eos_last": tok_rows[0]["input_ids"][-1] == tok.eos_token_id,
                         "merged_files": sorted(os.listdir(merged)),
                         "merged_reload_text": C.decode(tok, y[0, len(ids_on):].tolist())}
        del m
        torch.cuda.empty_cache()
    except Exception:
        report["trl"] = {"ok": False, "error": traceback.format_exc()[-4000:]}
    print("trl:", json.dumps(report["trl"])[:3000], flush=True)

    # 5. Engine decision.
    engine_name, detail = decide(vllm, hf_recs["s1"])
    report["engine"], report["engine_detail"] = engine_name, detail
    report["ok"] = bool(report["lora_targets_match"] and report["trl"]["ok"] and report["template"]["thinking_on_ends_with_think"]
                        and report["template"]["thinking_off_ends_with_empty_think"])
    path = f"{LOCAL}/report.json"
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    C.hub_upload(a.work, path, "smoke/report.json", f"smoke: {rev[:8]}, engine {engine_name}")
    for name, recs in hf_recs.items():
        C.write_jsonl(f"{LOCAL}/hf.{name}.jsonl", recs)
        C.hub_upload(a.work, f"{LOCAL}/hf.{name}.jsonl", f"smoke/hf.{name}.jsonl", f"smoke: hf {name}")
    print(json.dumps({k: report[k] for k in ("revision", "engine", "engine_detail", "linear_names",
                                              "lora_targets_match", "ok")}, indent=2))
    print("REVISION", rev)
    print("ENGINE", engine_name)


if __name__ == "__main__":
    main()
