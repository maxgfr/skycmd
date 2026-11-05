#!/usr/bin/env python3
"""The s6 recipe from the pico pilot and ablation, by the pre-registered rules (docs/s6-prereg.md §4,
amendment 4): nothing here is a judgement call, every number comes from the two summaries.

  s6-adopt.py adopt [--pilot runs/s6lr/summary.json] [--ablation runs/s6ab/summary.json] [--out runs/recipe]
      writes <out>/recipe.json (the rules, their numbers, the adopted flags) and the configs of the
      five models with Parallax and XSA as adopted (<out>/configs/<model>.json)
  s6-adopt.py env MODEL [--recipe runs/recipe/recipe.json]
      prints the shell assignments of MODEL's training script (CONFIG, OPT, LR_SCALE, PRETRAIN,
      DECIDE), one per line, shell-quoted; MODEL: s6-nano s6-micro s6-secrets-pico s6-ear-nano s6-cast-pico

Standard library only.
"""

import argparse
import datetime
import hashlib
import json
import os
import shlex
import sys

# docs/s6-prereg.md §3 and §4
MIN_GAIN = 0.01  # significant gain: mean accuracy, and larger than the std of both arms
MIN_GAIN_SLOW_MUON = 0.02  # rule 1 when Muon costs more than 15% of train_minutes
MUON_OVERHEAD = 0.15
ARENA_LOSS = 0.05  # a game whose mean score drops by more than 5% of the reference
ISOLATED_GAIN = 0.005  # rule 4
ISOLATED_ECE = 0.01  # rule 4: ECE no more than 0.01 worse
NOPOLISH_FLOOR = -0.005  # rule 5: non-inferior
NOPOLISH_SPEEDUP = 0.02  # rule 5: at least 2% faster

# rule 4: the isolated changes, read against parallax-muon
ISOLATED = [
    ("soda", "parallax-muon-soda"),
    ("exit_kd", "parallax-muon-kd"),
    ("ema", "parallax-muon-ema"),
    ("stableqat", "parallax-muon-stableqat"),
    ("curriculum", "parallax-muon-curriculum"),
]
# Each arm's flags, as crates/xtask/src/s6ab.rs `arms()` defines them; the summary carries them too
# (`opt`, `pretrain`, `decide`), and wins when present.
ARM_FLAGS = {
    "parallax-muon-soda": {"opt": ["--wd-mode", "soda"]},
    "parallax-muon-nopolish": {"opt": ["--ns-polish", "0"]},
    "parallax-muon-kd": {"decide": ["--exit-kd", "0.5"]},
    "parallax-muon-ema": {"decide": ["--ema", "0.999"]},
    "parallax-muon-stableqat": {"decide": ["--qat-grad-a", "0.21"]},
    "parallax-muon-curriculum": {"pretrain": ["--curriculum", "0.1"]},
}
MODELS = {
    "s6-nano": "configs/s6-nano.json",
    "s6-micro": "configs/s6-micro.json",
    "s6-secrets-pico": "configs/s6-secrets-pico.json",
    "s6-ear-nano": "configs/s6-ear-nano.json",
    "s6-cast-pico": "configs/s6-cast-pico.json",
}


def arm_map(summary):
    return {a["name"]: a for a in summary["arms"]}


def acc(a):
    return a["val"]["accuracy"]["mean"], a["val"]["accuracy"]["std"]


def significant(a, ref, need):
    """Gain of `a` over `ref`: at least `need`, and larger than the std of each arm."""
    (ma, sa), (mr, sr) = acc(a), acc(ref)
    gain = ma - mr
    return gain >= need and gain > sa and gain > sr, gain


def arena_losses(a, ref):
    """The games where `a` loses more than 5% of `ref`'s mean score."""
    out = []
    for g, r in ref["arena_score"].items():
        x = a["arena_score"][g]["mean"]
        if x < r["mean"] - ARENA_LOSS * abs(r["mean"]):
            out.append(f"{g} {x:.4g} < {r['mean']:.4g}")
    return out


def minutes(a):
    return a["train_minutes"]["mean"]


def flags(a, key):
    if key in a:
        return list(a[key])
    return list(ARM_FLAGS.get(a["name"], {}).get(key, []))


def rule_pair(arms, name, ref, need):
    """Rules 1–3: a significant gain and no arena loss."""
    a, r = arms[name], arms[ref]
    ok, gain = significant(a, r, need)
    loss = arena_losses(a, r)
    return {
        "arm": name, "against": ref, "gain": gain, "need": need,
        "std": [acc(a)[1], acc(r)[1]], "arena_loss": loss, "adopted": ok and not loss,
    }


def adopt(pilot, ablation):
    arms = arm_map(ablation)
    rules = {}

    # Rule 1: Muon v2 against AdamW, the gain needed rises to 0.02 if Muon is >15% slower.
    over = minutes(arms["softmax-muon"]) / minutes(arms["softmax-adamw"]) - 1
    r1 = rule_pair(arms, "softmax-muon", "softmax-adamw", MIN_GAIN_SLOW_MUON if over > MUON_OVERHEAD else MIN_GAIN)
    r1["overhead"] = over
    rules["1_muon"] = r1
    muon = r1["adopted"]

    # Rule 2: Parallax, against its softmax pair of the optimizer rule 1 kept.
    rules["2_parallax"] = (rule_pair(arms, "parallax-muon", "softmax-muon", MIN_GAIN) if muon
                           else rule_pair(arms, "parallax-adamw", "softmax-adamw", MIN_GAIN))
    # Rule 3: XSA against softmax-muon (it has no AdamW pair).
    rules["3_xsa"] = rule_pair(arms, "xsa-muon", "softmax-muon", MIN_GAIN)

    # Rule 4: the isolated changes against parallax-muon.
    ref = arms["parallax-muon"]
    for key, name in ISOLATED:
        a = arms[name]
        gain = acc(a)[0] - acc(ref)[0]
        d_ece = a["val"]["ece"]["mean"] - ref["val"]["ece"]["mean"]
        loss = arena_losses(a, ref)
        rules[f"4_{key}"] = {
            "arm": name, "against": "parallax-muon", "gain": gain, "need": ISOLATED_GAIN,
            "ece_delta": d_ece, "arena_loss": loss,
            "adopted": gain >= ISOLATED_GAIN and not loss and d_ece <= ISOLATED_ECE,
        }

    # Rule 5: no polish, non-inferior and at least 2% faster (a Muon option: moot under AdamW).
    a = arms["parallax-muon-nopolish"]
    gain = acc(a)[0] - acc(ref)[0]
    speed = 1 - minutes(a) / minutes(ref)
    r5 = {"arm": a["name"], "against": "parallax-muon", "gain": gain, "speedup": speed,
          "adopted": muon and gain >= NOPOLISH_FLOOR and speed >= NOPOLISH_SPEEDUP}
    if not muon:
        r5["note"] = "AdamW kept by rule 1: the Newton-Schulz polish is not used"
    rules["5_nopolish"] = r5

    # Rule 6: combine the adopted changes, without a new ablation.
    opt, pre, dec = ([] if muon else ["--no-muon"]), [], []
    for key, name in ISOLATED:
        if rules[f"4_{key}"]["adopted"]:
            opt += flags(arms[name], "opt")
            pre += flags(arms[name], "pretrain")
            dec += flags(arms[name], "decide")
    if r5["adopted"]:
        opt += flags(a, "opt")
    lr = pilot["lr_scale"]["muon" if muon else "adamw"]
    return {
        "about": "s6 recipe by the pre-registered rules (docs/s6-prereg.md §4, amendment 4), from the pico pilot and ablation",
        "optimizer": "muon" if muon else "adamw",
        "lr_scale": lr,
        "parallax": rules["2_parallax"]["adopted"],
        "xsa": rules["3_xsa"]["adopted"],
        "opt": opt,
        "pretrain": pre,
        "decide": dec,
        "ema": "--ema" in dec,
        "rules": rules,
    }


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def write(recipe, pilot_path, ablation_path, out, root="."):
    os.makedirs(os.path.join(out, "configs"), exist_ok=True)
    recipe = dict(recipe)
    recipe["inputs"] = {p: sha256(p) for p in (pilot_path, ablation_path)}
    recipe["written"] = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    recipe["configs"] = {}
    for model, base in MODELS.items():
        cfg = json.load(open(os.path.join(root, base)))
        cfg["parallax"] = recipe["parallax"]
        if recipe["xsa"]:
            cfg["xsa"] = True
        else:
            cfg.pop("xsa", None)
        p = os.path.join(out, "configs", f"{model}.json")
        with open(p, "w") as f:
            json.dump(cfg, f, indent=2)
            f.write("\n")
        recipe["configs"][model] = p
    p = os.path.join(out, "recipe.json")
    with open(p, "w") as f:
        json.dump(recipe, f, indent=2)
        f.write("\n")
    return p


def env(recipe, model):
    """Shell assignments for MODEL's training script."""
    if model not in MODELS:
        raise SystemExit(f"unknown model {model} ({', '.join(MODELS)})")
    v = {"CONFIG": recipe["configs"][model]}
    opt, pre, dec = list(recipe["opt"]), list(recipe["pretrain"]), list(recipe["decide"])
    if model == "s6-cast-pico":
        # s6-cast takes the optimizer-level adoptions (s6-cast-prereg amendment 3): the optimizer,
        # SODA, the polish, StableQAT; its EMA and QAT are its own, exit KD and the curriculum are
        # text-decision options it does not have.
        if "--qat-grad-a" in dec:
            i = dec.index("--qat-grad-a")
            opt += dec[i:i + 2]
        v["CAST_OPT"] = opt
        return v
    if model == "s6-ear-nano":
        # the curriculum orders text documents; the audio pretraining has none (s6-ear amendment 2)
        pre = []
    v.update({"OPT": opt, "LR_SCALE": recipe["lr_scale"], "PRETRAIN": pre, "DECIDE": dec})
    return v


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("adopt")
    a.add_argument("--pilot", default="runs/s6lr/summary.json")
    a.add_argument("--ablation", default="runs/s6ab/summary.json")
    a.add_argument("--out", default="runs/recipe")
    e = sub.add_parser("env")
    e.add_argument("model")
    e.add_argument("--recipe", default="runs/recipe/recipe.json")
    args = ap.parse_args()
    if args.cmd == "adopt":
        recipe = adopt(json.load(open(args.pilot)), json.load(open(args.ablation)))
        p = write(recipe, args.pilot, args.ablation, args.out)
        for k, r in recipe["rules"].items():
            print(f"rule {k}: {'adopted' if r['adopted'] else 'not adopted'} (gain {r['gain']:+.4f})")
        print(f"recipe: {recipe['optimizer']} x{recipe['lr_scale']}, parallax {recipe['parallax']}, "
              f"xsa {recipe['xsa']}, OPT {recipe['opt']}, PRETRAIN {recipe['pretrain']}, DECIDE {recipe['decide']}")
        print(f"wrote {p}")
    else:
        for k, x in env(json.load(open(args.recipe)), args.model).items():
            val = " ".join(x) if isinstance(x, list) else str(x)
            print(f"{k}={shlex.quote(val)}")


if __name__ == "__main__":
    sys.exit(main())
