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
"""LoRA fine-tuning of LFM3-1.7B on Hugging Face Jobs (docs/lfm3-ft-prereg.md §3 and §4).

Builds the SFT mix (worked solutions of the kept train questions as NO-THINK targets, plus the
self-distillation targets of gen/samples in the work repo), trains the pre-registered LoRA with TRL
(completion-only loss), merges it, and pushes the merged model with its adapter (adapter/) to the
PRIVATE model repo. Only the last checkpoint exists. sft/result.json in the work repo records the mix,
the training log and the model repo commit that eval.py loads.

  sft.py --common common.py --calib calib.jsonl --test test.jsonl --revision <sha>
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

LOCAL = "/tmp/lfm3-ft"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--common", help="common.py, shipped with the job (unused otherwise)")
    ap.add_argument("--calib", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--revision", required=True, help="base model revision pinned by the smoke")
    ap.add_argument("--repo", default=C.FT_REPO, help="private model repo")
    ap.add_argument("--work", default=C.WORK)
    a = ap.parse_args()

    calib, test = C.load_shipped(a.calib), C.load_shipped(a.test)
    mmlu = C.mmlu_items()
    rows, excl = C.select_train(C.load_train_rows(), calib, [r["question"] for r in test + mmlu])
    samples = C.hub_parts(a.work, "gen/samples", LOCAL)
    if not samples:
        raise SystemExit(f"no gen/samples in {a.work}: run `scripts/lfm3-ft.sh gen` first")
    mix = C.build_mix(rows, samples)
    tok = C.load_tokenizer(C.BASE, a.revision)
    tok_rows, dropped = C.tokenize_mix(tok, mix)
    kinds = {}
    for r in tok_rows:
        kinds[r["kind"]] = kinds.get(r["kind"], 0) + 1
    stats = {"exclusions": excl, "samples": len(samples), "examples": kinds, "dropped_over_max_len": dropped,
             "tokens": sum(len(r["input_ids"]) for r in tok_rows),
             "completion_tokens": sum(sum(r["completion_mask"]) for r in tok_rows)}
    print(json.dumps(stats, indent=2), flush=True)

    trainer, tok, total = C.make_trainer(C.BASE, a.revision, tok_rows, f"{LOCAL}/sft-out")
    if not C.check_completion_mask(trainer, tok_rows[0]):
        raise SystemExit("the collator does not mask the prompt as expected: stop (completion-only loss)")
    print(f"{len(tok_rows)} examples, {total} optimizer steps", flush=True)
    t0 = time.time()
    out = trainer.train()
    adapter = f"{LOCAL}/adapter"
    trainer.save_model(adapter)
    merged = C.merge_and_save(C.BASE, a.revision, adapter, f"{LOCAL}/merged")

    api = C.ensure_private(a.repo, "model")
    commit = api.upload_folder(folder_path=merged, repo_id=a.repo, repo_type="model",
                               commit_message=f"LoRA fine-tune of {C.BASE}@{a.revision[:8]} (docs/lfm3-ft-prereg.md)")
    result = {"repo": a.repo, "commit": commit.oid, "base": C.BASE, "base_revision": a.revision,
              "steps": total, "train_seconds": time.time() - t0, "train_metrics": out.metrics,
              "log_history": trainer.state.log_history, "mix": stats, "versions": C.versions()}
    path = f"{LOCAL}/sft/result.json"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    C.hub_upload(a.work, path, "sft/result.json", f"sft: {a.repo}@{commit.oid[:8]}")
    print(json.dumps({k: v for k, v in result.items() if k != "log_history"}, indent=2))


if __name__ == "__main__":
    main()
