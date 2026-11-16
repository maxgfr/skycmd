#!/usr/bin/env python3
"""s6-nano v3 (docs/s6-nano3-prereg.md, section 5 G4): the Python reference of the fused
probabilities, one JSON line per slate ({uid, gold, probs}, in slate order), from the model's dump
and the potion-cos-4bit dump with `fuse` of scripts/s6-route3-fit.py at the beta of
runs/s6-nano3/fit.json. The engine (sky-route fuse, crates/sky-web/bench/route3-node.mjs) is
compared with it.

    python3 -I scripts/s6-route3-reference.py --model runs/s6-nano2/v_tools.dump.jsonl \
        --embedding ../skycmd-data/s6-route/eval/val/potion-cos-4bit/v_tools.jsonl \
        --out ../skycmd-data/s6-route/embed/val/v_tools.fused.jsonl
"""
import argparse
import importlib.util
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--embedding", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fit", default=os.path.join(os.path.dirname(HERE), "runs", "s6-nano3", "fit.json"))
    a = ap.parse_args()
    spec = importlib.util.spec_from_file_location("fit3", os.path.join(HERE, "s6-route3-fit.py"))
    fit3 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fit3)
    beta = json.load(open(a.fit))["beta"]
    model, emb = fit3.load(a.model), fit3.load(a.embedding)
    if [r["gold"] for r in model] != [r["gold"] for r in emb] or [r.get("uid") for r in model] != [r.get("uid") for r in emb]:
        raise SystemExit("the two dumps are not the same slates in the same order")
    fused = fit3.fuse(model, emb, beta)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        for r, x in zip(model, fused):
            f.write(json.dumps({"uid": r.get("uid"), "gold": x["gold"], "probs": x["probs"]}) + "\n")
    print(json.dumps({"slates": len(fused), "beta": beta, "accuracy": fit3.accuracy(fused),
                      "accuracy_embedding": fit3.accuracy(emb), "accuracy_model": fit3.accuracy(model)}))


if __name__ == "__main__":
    main()
