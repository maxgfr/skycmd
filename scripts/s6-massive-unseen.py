#!/usr/bin/env python3
"""English MASSIVE on the test utterances not seen in training (docs/s6-gen-prereg.md, amendment 14).

From an s6 evaluation dump: the accuracy on the `zs_massive_en` items whose state is also the
state of a training question of `data/decisions/` (206 of 2,000), on the others (1,794), and the
pooled ECE without the seen ones. Reads the gold as scripts/s6-ece-by-source.py does.

    python3 scripts/s6-massive-unseen.py runs/s6-micro2/runs/s6-micro/eval/dump-s6-micro.jsonl
"""
import importlib.util, json, sys, glob
from pathlib import Path
spec = importlib.util.spec_from_file_location("e", "scripts/s6-ece-by-source.py")
e = importlib.util.module_from_spec(spec); spec.loader.exec_module(e)
dump = [json.loads(l) for l in open(sys.argv[1], encoding="utf-8") if l.strip()]
items = []
for i, (r, opts, target) in enumerate(e.records(Path("."), 2000)):
    d = dump[i]
    assert (d["src"], d["ins"], d["opts"], d["state"]) == (r.get("src", "?"), r.get("ins"), opts, r.get("state"))
    p = d["probs"]; k = e.argmax(p)
    items.append((d["src"], d["state"], p[k], float(k == e.argmax(target))))
assert len(items) == len(dump)
train = {}
for f in sorted(glob.glob("data/decisions/*.jsonl")):
    for line in open(f, encoding="utf-8"):
        if '"train"' not in line: continue
        r = json.loads(line)
        if r.get("split") == "train" and isinstance(r.get("state"), str):
            train.setdefault(r["state"], set()).add(r.get("src", "?"))
en = [x for x in items if x[0] == "zs_massive_en"]
seen = [x for x in en if x[1] in train]
unseen = [x for x in en if x[1] not in train]
acc = lambda v: sum(h for *_, h in v) / len(v)
print("en", len(en), acc(en), "seen", len(seen), acc(seen), "unseen", len(unseen), acc(unseen))
srcs = {}
for x in seen:
    for s in train[x[1]]: srcs[s] = srcs.get(s, 0) + 1
print(srcs)
fr = [x for x in items if x[0] == "zs_massive_fr"]
print("fr seen", sum(1 for x in fr if x[1] in train))
allp = [(c, h) for _, _, c, h in items]
wo = [(c, h) for s, st, c, h in items if not (s == "zs_massive_en" and st in train)]
print("pooled ece", e.ece(allp), e.ece(allp, e.bin_eval), "without seen", len(wo), e.ece(wo), e.ece(wo, e.bin_eval))
