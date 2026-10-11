"""Produce reference outputs of the Laya runtime for the oscar model (run once, offline).

From the repository root:
    uv run --python 3.12 --with torch --with transformers --with safetensors --with numpy \
        python crates/sky-infer/tests/golden/make_golden.py

It loads models-src/rl_common.py + rl_agent_api.py, builds the ModernBERT encoder from
models-src/oscar/encoder/config.json (AutoModel.from_config, no download) and the weights from
models-src/oscar/model.safetensors, then writes tests/golden/oscar.json with, per case: the request,
the rendered option texts, token ids, marker positions, raw logits and the probabilities after the
temperature (float32 on CPU).
"""
import json
import os
import sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
SRC = os.path.join(ROOT, "models-src")
MODEL_DIR = os.path.join(SRC, "oscar")
sys.path.insert(0, SRC)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from rl_agent_api import RLAgent  # noqa: E402
from rl_common import QTYPES, build_sequence, collate_items, render_options, serialize_state, temp_bucket  # noqa: E402

LONG_STATE = " ".join("step %d: the bird is at altitude %d with vertical speed %d and the next gap is %d units ahead."
                      % (i, 10 + (i * 7) % 50, (i % 9) - 4, 40 - i % 30) for i in range(60))

CASES = [
    ("choice2", "Flap now or wait?", "choice", {"flap": "push the bird up", "wait": "let the bird fall"},
     "alt 12 vy -3 dx 40 gt 20 gb 35"),
    ("choice5", "Which direction should the rescue drone move?", "choice", ["up", "down", "left", "right", "stay"],
     "drone x 4 y 7 target x 9 y 2 wind east 3 battery 61"),
    ("choice12", "Pick the month with the best weather for a beach holiday in France.", "choice",
     ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october",
      "november", "december"], "The user lives in Lyon and likes warm sea water."),
    ("choice12_long", "Which option best describes the situation?", "choice",
     {"opt%d" % i: "a fairly long description number %d that keeps going with many extra words to use up the option budget" % i
      for i in range(12)}, "nothing special happens"),
    ("choice3_json", "What should the player do next?", "choice",
     {"jump": "jump over the obstacle", "duck": None, "run": "keep running"},
     {"player": {"x": 3, "y": 0}, "obstacle": {"x": 5, "h": 2}, "speed": 4, "name": "dodge"}),
    ("score5", "How dangerous is the current state for the bird?", "score",
     ["safe", "slightly risky", "risky", "very risky", "about to crash"], "alt 3 vy -9 dx 4 gt 20 gb 35"),
    ("score3", "How confident is the agent in its plan?", "score", ["low", "medium", "high"],
     "The agent has explored 80 percent of the map and found two exits."),
    ("noul_default", "The bird is above the next gap.", "noul", None, "alt 40 vy 1 dx 12 gt 20 gb 35"),
    ("noul_custom", "Is the message polite?", "noul",
     {"false": "the message is rude or aggressive", "true": "the message is polite"},
     "Could you please send me the report when you have a moment? Thanks a lot!"),
    ("long_state", "Will the bird survive the next ten frames?", "noul", None, LONG_STATE),
    ("french", "Le client est-il satisfait ?", "choice",
     {"oui": "le client est content", "non": "le client est mécontent", "neutre": None},
     "Bonjour, j'ai reçu ma commande hier : l'emballage était abîmé mais le produit fonctionne très bien. Ça me va !"),
    ("instr_json", {"task": "classify", "labels": ["ok", "ko"], "note": "été"}, "choice", ["ok", "ko"],
     "status: all systems nominal, élan 3"),
    ("game_choice3", "Which action keeps the bird alive?", "choice", {"flap": None, "glide": None, "dive": None},
     "alt 25 vy 2 dx 8 gt 22 gb 31"),
]


def main():
    torch.manual_seed(0)
    agent = RLAgent(MODEL_DIR, device="cpu")
    model, tok, cfg = agent.model, agent.tok, agent.cfg
    out_cases = []
    for name, ins, t, crit, state in CASES:
        qdef = {"type": t, "instructions": ins}
        if crit is not None:
            qdef["criteria"] = crit
        q = RLAgent._to_internal(qdef)
        ids, markers = build_sequence(tok, state, q, cfg["max_len"], cfg["head_max_len"])
        opts = render_options(q)
        assert len(markers) == len(opts), name
        item = {"ids": ids, "markers": markers, "qtype": QTYPES[t], "target": [0.0] * len(markers), "label": -1,
                "episode": 0, "ep_step": 0, "ep_len": 1, "src": "golden"}
        b = collate_items([[item]], tok.pad_token_id)
        with torch.no_grad():
            logits, _ = model(b["input_ids"], b["attention_mask"], b["marker_pos"], b["marker_mask"], b["qtype"])
        logits = logits.float().numpy()[0, :len(markers)]
        k = len(markers)
        temp = agent.temperature_by_options.get(temp_bucket(QTYPES[t], k), agent.temperature[QTYPES[t]])
        z = logits / temp
        p = np.exp(z - z.max())
        p = p / p.sum()
        answer = agent.system_one(state, {"q": qdef})["answers"]["q"]
        out_cases.append({
            "name": name,
            "question": qdef,
            "state": state,
            "state_text": serialize_state(state),
            "instructions_text": q["ins"],
            "qtype": t,
            "option_texts": opts,
            "ids": ids,
            "markers": markers,
            "logits": [float(v) for v in logits],
            "temperature": float(temp),
            "probs": [float(v) for v in p],
            "answer": answer,
        })
        print("%-14s T=%3d k=%2d probs=%s" % (name, len(ids), k, np.round(p, 4)))
    meta = {"model": "mgoeckel/oscar-1-17m", "torch": torch.__version__,
            "max_len": cfg["max_len"], "head_max_len": cfg["head_max_len"], "temperature": agent.temperature}
    import transformers
    meta["transformers"] = transformers.__version__
    with open(os.path.join(HERE, "oscar.json"), "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "cases": out_cases}, f, ensure_ascii=False, indent=1)
    print("wrote %d cases" % len(out_cases))


if __name__ == "__main__":
    main()
