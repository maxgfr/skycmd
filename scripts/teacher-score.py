#!/usr/bin/env python3
"""Qwen3.5 teacher for s4 (mlx-lm): scores the option letters of decision records in one pass.

Each record (docs/SPEC.md decision schema) becomes a non-thinking chat prompt listing the
state, the question and the lettered options. The teacher is never sampled: one batched
prefill, then the next-token logits restricted to the option letters. Prompts are right-padded,
which is exact for a causal model (attention and the Gated DeltaNet recurrence only look back),
and only the last real position of each row goes through the output projection.

Output: the input records plus `uid` (line index in the input) and `tlogits` (letter logits,
temperature 1). The file is appended and flushed every batch, so a killed run resumes where it
stopped (`uid`s already present are skipped).

Usage:
  teacher-score.py --model mlx-community/Qwen3.5-4B-4bit --in recs.jsonl --out scored.jsonl
  teacher-score.py --model ... --in bench.jsonl --out b.jsonl --eval   # accuracy / NLL / speed
"""
import argparse
import json
import math
import os
import sys
import time

import mlx.core as mx
from mlx_lm import load

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
HINT = {
    "choice": "Choose the best option.",
    "score": "Choose the level that fits best.",
    "noul": "Decide whether the statement holds.",
}
MAX_STATE_CHARS = 2400


def prompt(tok, r):
    opts = "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(r["opts"]))
    state = (r.get("state") or "").strip()[:MAX_STATE_CHARS]
    parts = []
    if state:
        parts.append(f"Context:\n{state}")
    parts.append(f"Question: {r['ins'].strip()}\n{HINT.get(r['t'], '')}")
    parts.append(f"Options:\n{opts}")
    parts.append("Answer with the letter of the correct option only.")
    msgs = [{"role": "user", "content": "\n\n".join(parts)}]
    text = tok.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    return tok.encode(text, add_special_tokens=False)


def letter_ids(tok):
    ids = []
    for c in LETTERS:
        t = tok.encode(c, add_special_tokens=False)
        assert len(t) == 1, (c, t)
        ids.append(t[0])
    return ids


def text_model(model):
    lm = getattr(model, "language_model", model)
    return lm


def score_batch(model, rows, lids, pad_id):
    """rows: list of token lists -> list of letter-logit arrays (len = n letters)."""
    lm = text_model(model)
    T = max(len(r) for r in rows)
    ids = mx.array([r + [pad_id] * (T - len(r)) for r in rows])
    h = lm.model(ids)  # [B, T, D]
    last = mx.array([len(r) - 1 for r in rows])
    h = mx.take_along_axis(h, last.reshape(-1, 1, 1), axis=1)  # [B, 1, D]
    if lm.args.tie_word_embeddings:
        logits = lm.model.embed_tokens.as_linear(h)
    else:
        logits = lm.lm_head(h)
    logits = logits[:, 0, :].astype(mx.float32)
    sel = mx.take(logits, mx.array(lids), axis=1)  # [B, 26]
    mx.eval(sel)
    return sel.tolist()


def softmax(xs, t=1.0):
    m = max(xs)
    e = [math.exp((x - m) / t) for x in xs]
    s = sum(e)
    return [v / s for v in e]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch-tokens", type=int, default=12000)
    ap.add_argument("--max-batch", type=int, default=48)
    ap.add_argument("--window", type=int, default=2048, help="records sorted by length per window")
    ap.add_argument("--eval", action="store_true", help="print accuracy / NLL against target argmax")
    ap.add_argument("--deadline", type=float, default=0, help="stop after this many minutes")
    ap.add_argument("--cache-gb", type=float, default=2.0, help="MLX buffer cache limit")
    a = ap.parse_args()
    # MLX keeps freed buffers in a cache that otherwise grows to most of the RAM (12 GB on a
    # 16 GB Mac after an hour of variable batch shapes), which pushes the student into swap.
    mx.set_cache_limit(int(a.cache_gb * (1 << 30)))

    model, tok = load(a.model)
    lids = letter_ids(tok)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    done = set()
    if os.path.exists(a.out):
        with open(a.out) as f:
            for line in f:
                try:
                    done.add(json.loads(line)["uid"])
                except (ValueError, KeyError):
                    pass
    recs = []
    with open(a.inp) as f:
        for i, line in enumerate(f):
            if a.limit and i >= a.limit:
                break
            if i in done:
                continue
            r = json.loads(line)
            if not 2 <= len(r["opts"]) <= len(LETTERS):
                continue
            r["uid"] = i
            recs.append(r)
    print(f"[teacher] {a.model}: {len(recs)} records to score ({len(done)} already done)", flush=True)

    out = open(a.out, "a")
    t0 = time.time()
    n = ntok = 0
    hit = nll = 0.0
    last_log = t0
    for w in range(0, len(recs), a.window):
        win = recs[w : w + a.window]
        enc = [(r, prompt(tok, r)) for r in win]
        enc.sort(key=lambda x: len(x[1]))
        i = 0
        while i < len(enc):
            j = i
            T = 0
            while j < len(enc) and j - i < a.max_batch:
                T2 = max(T, len(enc[j][1]))
                if (j - i + 1) * T2 > a.batch_tokens and j > i:
                    break
                T = T2
                j += 1
            chunk = enc[i:j]
            sel = score_batch(model, [x[1] for x in chunk], lids, pad_id)
            for (r, ids), lg in zip(chunk, sel):
                k = len(r["opts"])
                r["tlogits"] = [round(v, 3) for v in lg[:k]]
                out.write(json.dumps(r, ensure_ascii=False) + "\n")
                ntok += len(ids)
                if a.eval:
                    p = softmax(r["tlogits"])
                    g = max(range(k), key=lambda q: r["target"][q])
                    hit += p.index(max(p)) == g
                    nll -= math.log(max(p[g], 1e-9))
            out.flush()
            n += len(chunk)
            i = j
            now = time.time()
            if now - last_log > 30:
                last_log = now
                dt = now - t0
                print(f"[teacher] {n} done, {n / dt:.1f} ex/s, {ntok / dt:.0f} tok/s", flush=True)
            if a.deadline and (now - t0) / 60 > a.deadline:
                print("[teacher] deadline reached", flush=True)
                break
        if a.deadline and (time.time() - t0) / 60 > a.deadline:
            break
    dt = time.time() - t0
    summary = {"model": a.model, "n": n, "ex_per_s": n / max(dt, 1e-9), "tok_per_s": ntok / max(dt, 1e-9)}
    if a.eval and n:
        summary.update(acc=hit / n, nll=nll / n)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    sys.exit(main())
