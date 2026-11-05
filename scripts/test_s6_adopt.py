#!/usr/bin/env python3
"""Tests of scripts/s6-adopt.py on synthetic summaries: python3 -I scripts/test_s6_adopt.py"""

import copy
import importlib.util
import json
import os
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("s6_adopt", os.path.join(HERE, "s6-adopt.py"))
s6 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s6)

NAMES = ["softmax-adamw", "softmax-muon", "parallax-adamw", "parallax-muon", "xsa-muon",
         "parallax-muon-soda", "parallax-muon-nopolish", "parallax-muon-kd", "parallax-muon-ema",
         "parallax-muon-stableqat", "parallax-muon-curriculum"]
GAMES = ["flappy", "rescue", "dodge", "doom"]


def stat(mean, std=0.002):
    return {"mean": mean, "std": std, "n": 3}


def ablation(**over):
    """Every arm at 0.60 accuracy, ECE 0.05, score 10 on each game, 20 minutes; `over` changes
    an arm: name=dict(acc=, std=, ece=, minutes=, score={game: x})."""
    arms = []
    for n in NAMES:
        o = over.get(n.replace("-", "_"), {})
        score = {g: stat(o.get("score", {}).get(g, 10.0)) for g in GAMES}
        arms.append({
            "name": n, "opt": [], "pretrain": [], "decide": [],
            "val": {"accuracy": stat(o.get("acc", 0.60), o.get("std", 0.002)),
                    "ece": stat(o.get("ece", 0.05)), "margin": stat(1.0), "erank": stat(10.0)},
            "arena_score": score, "train_minutes": stat(o.get("minutes", 20.0)),
        })
    for a in arms:
        for k, v in s6.ARM_FLAGS.get(a["name"], {}).items():
            a[k] = v
    return {"arms": arms}


PILOT = {"lr_scale": {"adamw": 0.5, "muon": 2.0}}


class Rules(unittest.TestCase):
    def test_nothing_wins_keeps_adamw_and_defaults(self):
        r = s6.adopt(PILOT, ablation())
        self.assertEqual(r["optimizer"], "adamw")
        self.assertEqual(r["lr_scale"], 0.5)
        self.assertEqual(r["opt"], ["--no-muon"])
        self.assertFalse(r["parallax"])
        self.assertFalse(r["xsa"])
        self.assertEqual((r["pretrain"], r["decide"]), ([], []))
        # no polish is a Muon option: moot under AdamW, even if it is faster
        r = s6.adopt(PILOT, ablation(parallax_muon_nopolish={"minutes": 10}))
        self.assertFalse(r["rules"]["5_nopolish"]["adopted"])

    def test_muon_needs_a_significant_gain(self):
        self.assertTrue(s6.adopt(PILOT, ablation(softmax_muon={"acc": 0.611}))["rules"]["1_muon"]["adopted"])
        # below 0.01
        self.assertFalse(s6.adopt(PILOT, ablation(softmax_muon={"acc": 0.609}))["rules"]["1_muon"]["adopted"])
        # a gain inside the std of one arm
        self.assertFalse(s6.adopt(PILOT, ablation(softmax_muon={"acc": 0.615, "std": 0.02}))["rules"]["1_muon"]["adopted"])
        # an arena loss of more than 5% on one game
        r = s6.adopt(PILOT, ablation(softmax_muon={"acc": 0.62, "score": {"doom": 9.4}}))
        self.assertFalse(r["rules"]["1_muon"]["adopted"])
        self.assertEqual(len(r["rules"]["1_muon"]["arena_loss"]), 1)
        # 5% exactly is not a loss
        self.assertTrue(s6.adopt(PILOT, ablation(softmax_muon={"acc": 0.62, "score": {"doom": 9.5}}))["rules"]["1_muon"]["adopted"])

    def test_slow_muon_needs_002(self):
        slow = {"acc": 0.615, "minutes": 23.1}  # 15.5% slower
        self.assertFalse(s6.adopt(PILOT, ablation(softmax_muon=slow))["rules"]["1_muon"]["adopted"])
        slow["acc"] = 0.621
        r = s6.adopt(PILOT, ablation(softmax_muon=slow))
        self.assertTrue(r["rules"]["1_muon"]["adopted"])
        self.assertEqual(r["rules"]["1_muon"]["need"], 0.02)
        # 15% is not more than 15%
        self.assertEqual(s6.adopt(PILOT, ablation(softmax_muon={"acc": 0.615, "minutes": 23.0}))["rules"]["1_muon"]["need"], 0.01)

    def test_parallax_is_read_against_the_kept_optimizer(self):
        # AdamW kept: parallax-adamw against softmax-adamw
        r = s6.adopt(PILOT, ablation(parallax_adamw={"acc": 0.62}))
        self.assertEqual(r["rules"]["2_parallax"]["against"], "softmax-adamw")
        self.assertTrue(r["parallax"])
        # Muon kept: parallax-muon against softmax-muon
        r = s6.adopt(PILOT, ablation(softmax_muon={"acc": 0.62}, parallax_muon={"acc": 0.635}))
        self.assertEqual(r["rules"]["2_parallax"]["against"], "softmax-muon")
        self.assertTrue(r["parallax"])
        self.assertEqual((r["optimizer"], r["lr_scale"], r["opt"]), ("muon", 2.0, []))

    def test_isolated_changes(self):
        base = {"softmax_muon": {"acc": 0.62}, "parallax_muon": {"acc": 0.63}}
        r = s6.adopt(PILOT, ablation(**base, parallax_muon_soda={"acc": 0.635},
                                     parallax_muon_ema={"acc": 0.636},
                                     parallax_muon_curriculum={"acc": 0.64},
                                     parallax_muon_stableqat={"acc": 0.64, "ece": 0.061},
                                     parallax_muon_kd={"acc": 0.634}))
        self.assertTrue(r["rules"]["4_soda"]["adopted"])
        self.assertTrue(r["rules"]["4_ema"]["adopted"])
        self.assertTrue(r["rules"]["4_curriculum"]["adopted"])
        self.assertFalse(r["rules"]["4_stableqat"]["adopted"])  # ECE 0.011 worse
        self.assertFalse(r["rules"]["4_exit_kd"]["adopted"])  # 0.004 < 0.005
        self.assertEqual(r["opt"], ["--wd-mode", "soda"])
        self.assertEqual(r["pretrain"], ["--curriculum", "0.1"])
        self.assertEqual(r["decide"], ["--ema", "0.999"])
        self.assertTrue(r["ema"])

    def test_nopolish(self):
        base = {"softmax_muon": {"acc": 0.62}, "parallax_muon": {"acc": 0.63}}
        ok = s6.adopt(PILOT, ablation(**base, parallax_muon_nopolish={"acc": 0.626, "minutes": 19.5}))
        self.assertTrue(ok["rules"]["5_nopolish"]["adopted"])
        self.assertEqual(ok["opt"], ["--ns-polish", "0"])
        # inferior by more than 0.005
        self.assertFalse(s6.adopt(PILOT, ablation(**base, parallax_muon_nopolish={"acc": 0.624, "minutes": 19}))["rules"]["5_nopolish"]["adopted"])
        # less than 2% faster
        self.assertFalse(s6.adopt(PILOT, ablation(**base, parallax_muon_nopolish={"acc": 0.63, "minutes": 19.7}))["rules"]["5_nopolish"]["adopted"])

    def test_flags_fall_back_on_the_arm_table(self):
        abl = ablation(softmax_muon={"acc": 0.62}, parallax_muon_curriculum={"acc": 0.64})
        for a in abl["arms"]:
            a.pop("pretrain")  # a summary written before the field existed
        self.assertEqual(s6.adopt(PILOT, abl)["pretrain"], ["--curriculum", "0.1"])


class Output(unittest.TestCase):
    def test_configs_and_env(self):
        root = os.path.dirname(HERE)
        abl = ablation(softmax_muon={"acc": 0.62}, parallax_muon={"acc": 0.635}, xsa_muon={"acc": 0.64},
                       parallax_muon_stableqat={"acc": 0.64}, parallax_muon_curriculum={"acc": 0.64},
                       parallax_muon_ema={"acc": 0.64})
        with tempfile.TemporaryDirectory() as d:
            pp, ap = os.path.join(d, "pilot.json"), os.path.join(d, "abl.json")
            json.dump(PILOT, open(pp, "w"))
            json.dump(abl, open(ap, "w"))
            rec = s6.adopt(PILOT, abl)
            out = os.path.join(d, "recipe")
            p = s6.write(rec, pp, ap, out, root=root)
            r = json.load(open(p))
            self.assertEqual(set(r["inputs"]), {pp, ap})
            for model, path in r["configs"].items():
                cfg = json.load(open(path))
                base = json.load(open(os.path.join(root, s6.MODELS[model])))
                self.assertTrue(cfg["parallax"] and cfg["xsa"])
                self.assertEqual(cfg["name"], base["name"])
                self.assertEqual({k: v for k, v in cfg.items() if k not in ("parallax", "xsa")},
                                 {k: v for k, v in base.items() if k not in ("parallax", "xsa")})
            nano = s6.env(r, "s6-nano")
            self.assertEqual(nano["DECIDE"], ["--ema", "0.999", "--qat-grad-a", "0.21"])
            self.assertEqual(nano["PRETRAIN"], ["--curriculum", "0.1"])
            self.assertEqual(nano["LR_SCALE"], 2.0)
            self.assertEqual(s6.env(r, "s6-ear-nano")["PRETRAIN"], [])
            cast = s6.env(r, "s6-cast-pico")
            self.assertEqual(cast["CAST_OPT"], ["--qat-grad-a", "0.21"])
            self.assertNotIn("DECIDE", cast)

    def test_xsa_off_removes_the_key(self):
        root = os.path.dirname(HERE)
        with tempfile.TemporaryDirectory() as d:
            pp, ap = os.path.join(d, "p.json"), os.path.join(d, "a.json")
            abl = ablation()
            json.dump(PILOT, open(pp, "w"))
            json.dump(abl, open(ap, "w"))
            r = json.load(open(s6.write(s6.adopt(PILOT, abl), pp, ap, os.path.join(d, "r"), root=root)))
            cfg = json.load(open(r["configs"]["s6-nano"]))
            self.assertFalse(cfg["parallax"])
            self.assertNotIn("xsa", cfg)


if __name__ == "__main__":
    unittest.main()
