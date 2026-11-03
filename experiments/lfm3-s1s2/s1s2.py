#!/usr/bin/env python3
"""LFM3-1.7B, System 1 first and System 2 only when needed (docs/lfm3-s1s2-prereg.md).

LFM3-1.7B reasons by default, and on a small model longer reasoning can talk it out of a correct
first answer. This experiment answers every question twice, offline:

  s1      reasoning off (`<think></think>`), greedy: the first answer, with the log-probability of
          every generated token;
  s2-B    reasoning on with a budget of B tokens (Liquid AI's math sampling: temperature 1.0,
          top_p 0.95, top_k 20). When the budget is spent without `</think>`, the thinking is
          closed and the answer is asked for (budget forcing).

then routes each question (`route`): keep the s1 answer when a confidence score computed from s1
alone is at least tau, else take the s2 answer (or the s1 answer when s2 gave none). The score is
a logistic regression over s1 features, fitted on the calibration split; tau is the smallest
threshold whose conformal risk-control bound on the loss of accuracy against always-s2 is at most
ALPHA, also on the calibration split. The test split is read once, by `route --split test`.

Usage (scripts/lfm3-s1s2.sh wraps these with the pinned environment):
  s1s2.py data                                   data/lfm3/{calib,test}.jsonl
  s1s2.py gen --split calib --mode s1 [--limit N]
  s1s2.py gen --split calib --mode s2 --budget 1024
  s1s2.py route [--split test]                    runs/lfm3-s1s2/report.md, summary.json
Generation appends to runs/lfm3-s1s2/<split>.<mode>.jsonl and skips the ids already there, so an
interrupted run resumes.
"""
import argparse
import json
import math
import os
import random
import re
import sys
import time

MODEL = "LiquidAI/LFM3-1.7B-MLX-4bit"
MODEL_REV = "deb2986b9660c15ff7dd4c2bb1211ce3bbb814d4"
GSM8K = ("openai/gsm8k", "740312add88f781978c0658806c59bc2815b9866")
MATH500 = ("HuggingFaceH4/MATH-500", "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be")
MATH_TRAIN = ("DigitalLearningGmbH/MATH-lighteval", "0530c78699ea5e8eb5530600900e1f328b48acad")
SEED = 1234
N_CALIB = 300  # per task
N_TEST_GSM8K = 500
ANSWER_MAX = 1024  # tokens after the thinking (s2) or in total (s1)
ALPHA = 0.02
BUDGETS = (256, 1024)
DATA = "data/lfm3"
RUNS = "runs/lfm3-s1s2"
PROMPT = "{q}\n\nSolve the problem. Put the final answer in \\boxed{{}}."


# ----------------------------------------------------------------------------- data


def last_boxed(text):
    """Content of the last \\boxed{...} (balanced braces), or None."""
    i = text.rfind("\\boxed")
    if i < 0:
        return None
    j = text.find("{", i)
    if j < 0:
        return None
    depth, k = 0, j
    while k < len(text):
        if text[k] == "{":
            depth += 1
        elif text[k] == "}":
            depth -= 1
            if depth == 0:
                return text[j + 1 : k]
        k += 1
    return None


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def cmd_data(_):
    from datasets import load_dataset

    rng = random.Random(SEED)
    g_train = load_dataset(GSM8K[0], "main", split="train", revision=GSM8K[1])
    g_test = load_dataset(GSM8K[0], "main", split="test", revision=GSM8K[1])
    m_train = load_dataset(MATH_TRAIN[0], split="train", revision=MATH_TRAIN[1])
    m_test = load_dataset(MATH500[0], split="test", revision=MATH500[1])

    def gsm(rows, split):
        return [
            {"id": f"gsm8k-{split}-{i}", "task": "gsm8k", "question": r["question"],
             "answer": r["answer"].split("####")[-1].strip()}
            for i, r in rows
        ]

    gi = list(enumerate(g_train))
    rng.shuffle(gi)
    calib = gsm(gi[:N_CALIB], "train")
    mi = [(i, r) for i, r in enumerate(m_train) if last_boxed(r["solution"]) is not None]
    rng.shuffle(mi)
    calib += [
        {"id": f"math-train-{i}", "task": "math", "question": r["problem"], "answer": last_boxed(r["solution"])}
        for i, r in mi[:N_CALIB]
    ]
    ti = list(enumerate(g_test))
    rng.shuffle(ti)
    test = gsm(ti[:N_TEST_GSM8K], "test")
    test += [
        {"id": f"math500-{r['unique_id']}", "task": "math", "question": r["problem"], "answer": r["answer"]}
        for r in m_test
    ]
    write_jsonl(f"{DATA}/calib.jsonl", calib)
    write_jsonl(f"{DATA}/test.jsonl", test)
    print(f"calib {len(calib)} (gsm8k {N_CALIB}, math {N_CALIB}), test {len(test)} "
          f"(gsm8k {N_TEST_GSM8K}, math-500 {len(m_test)}) -> {DATA}/")


# ----------------------------------------------------------------------------- grading


def strip_math(s):
    s = s.strip().strip("$").strip()
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
    for a, b in (("\\left", ""), ("\\right", ""), ("\\!", ""), ("\\,", ""), ("\\;", ""), ("\\ ", ""),
                 ("dfrac", "frac"), ("tfrac", "frac"), ("^{\\circ}", ""), ("^\\circ", ""), ("\\%", ""),
                 ("%", ""), (" ", "")):
        s = s.replace(a, b)
    s = s.rstrip(".")
    if s.startswith("x=") or s.startswith("y="):
        s = s[2:]
    return s


def as_number(s):
    s = s.replace(",", "").replace("\\$", "").replace("$", "")
    try:
        return float(s)
    except ValueError:
        m = re.fullmatch(r"\\frac\{(-?\d+)\}\{(-?\d+)\}", s)
        if m and float(m.group(2)) != 0:
            return float(m.group(1)) / float(m.group(2))
    return None


def extract(task, text):
    """The answer of a generation: the last \\boxed{}, else (gsm8k) the last number."""
    b = last_boxed(text)
    if b is not None:
        return b
    if task == "gsm8k":
        nums = re.findall(r"-?\d[\d,]*(?:\.\d+)?", text)
        return nums[-1] if nums else None
    return None


def correct(task, pred, gold):
    if pred is None:
        return False
    p, g = strip_math(pred), strip_math(gold)
    if p == g:
        return True
    pn, gn = as_number(p), as_number(g)
    return pn is not None and gn is not None and abs(pn - gn) <= 1e-6 * max(1.0, abs(gn))


# ----------------------------------------------------------------------------- generation


class Model:
    def __init__(self):
        from huggingface_hub import snapshot_download
        from mlx_vlm import load

        path = snapshot_download(MODEL, revision=MODEL_REV)
        self.model, self.processor = load(path)
        self.tok = getattr(self.processor, "tokenizer", self.processor)

    def prompt(self, question, thinking):
        msgs = [{"role": "user", "content": PROMPT.format(q=question)}]
        return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                            enable_thinking=thinking)

    def stream(self, prompt, max_tokens, seed, **sampling):
        """Yields (text piece, log-probability of the sampled token, finish reason)."""
        import mlx.core as mx
        from mlx_vlm import stream_generate

        mx.random.seed(seed)
        for r in stream_generate(self.model, self.processor, prompt, max_tokens=max_tokens, **sampling):
            lp = float(r.logprobs[r.token].item()) if r.logprobs is not None else float("nan")
            yield r.text, lp, r.finish_reason


GREEDY = {"temperature": 0.0}
MATH_SAMPLING = {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "repetition_penalty": 1.0}


def answer_logprob(pieces):
    """Mean log-probability of the tokens from the last \\boxed onwards (all tokens if none)."""
    text = "".join(p for p, _ in pieces)
    start = text.rfind("\\boxed")
    if start < 0:
        lps = [lp for _, lp in pieces]
    else:
        lps, pos = [], 0
        for p, lp in pieces:
            if pos + len(p) > start:
                lps.append(lp)
            pos += len(p)
    lps = [x for x in lps if math.isfinite(x)]
    return sum(lps) / len(lps) if lps else float("nan")


def gen_s1(m, item, seed):
    pieces, finish = [], None
    for text, lp, fin in m.stream(m.prompt(item["question"], False), ANSWER_MAX, seed, **GREEDY):
        pieces.append((text, lp))
        finish = fin or finish
    text = "".join(p for p, _ in pieces)
    lps = [lp for _, lp in pieces if math.isfinite(lp)]
    return {
        "text": text,
        "tokens": len(pieces),
        "think_tokens": 0,
        "finish": finish,
        "mean_lp": sum(lps) / len(lps) if lps else float("nan"),
        "min_lp": min(lps) if lps else float("nan"),
        "answer_lp": answer_logprob(pieces),
    }


def gen_s2(m, item, seed, budget):
    prompt = m.prompt(item["question"], True)
    think, closed, n_think = [], False, 0
    after, finish = [], None
    for text, _, fin in m.stream(prompt, budget + ANSWER_MAX, seed, **MATH_SAMPLING):
        if not closed:
            think.append(text)
            n_think += 1
            joined = "".join(think)
            if "</think>" in joined:
                closed = True
                head, tail = joined.split("</think>", 1)
                think, after = [head], [tail]
                n_after = 0
            elif n_think >= budget:
                break
        else:
            after.append(text)
            n_after += 1
            if n_after >= ANSWER_MAX:
                break
        finish = fin or finish
    forced = not closed
    if forced:
        # Budget spent: close the thinking and ask for the answer (budget forcing).
        prompt2 = prompt + "".join(think) + "\n</think>\n\n"
        after = []
        for text, _, fin in m.stream(prompt2, ANSWER_MAX, seed + 1, **MATH_SAMPLING):
            after.append(text)
            finish = fin or finish
    answer_text = "".join(after)
    return {
        "text": answer_text,
        "think": "".join(think),
        "tokens": n_think + len(after),
        "think_tokens": n_think,
        "forced": forced,
        "finish": finish,
    }


def cmd_gen(a):
    items = read_jsonl(f"{DATA}/{a.split}.jsonl")
    if not items:
        sys.exit(f"no {DATA}/{a.split}.jsonl: run `data` first")
    if a.limit:
        items = items[: a.limit]
    mode = "s1" if a.mode == "s1" else f"s2-{a.budget}"
    out = f"{a.out or RUNS}/{a.split}.{mode}.jsonl"
    done = {r["id"] for r in read_jsonl(out)}
    todo = [it for it in items if it["id"] not in done]
    print(f"{mode} on {a.split}: {len(todo)} to generate ({len(done)} done) -> {out}", flush=True)
    if not todo:
        return
    m = Model()
    os.makedirs(os.path.dirname(out), exist_ok=True)
    t_all, n_tok = time.time(), 0
    with open(out, "a") as f:
        for k, it in enumerate(todo):
            seed = SEED + int.from_bytes(it["id"].encode()[-6:], "big") % 1_000_000
            t0 = time.time()
            r = gen_s1(m, it, seed) if a.mode == "s1" else gen_s2(m, it, seed, a.budget)
            pred = extract(it["task"], r["text"])
            r.update(id=it["id"], task=it["task"], mode=mode, pred=pred,
                     correct=correct(it["task"], pred, it["answer"]), seconds=time.time() - t0)
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            f.flush()
            n_tok += r["tokens"]
            if (k + 1) % 10 == 0 or k + 1 == len(todo):
                el = time.time() - t_all
                print(f"  {k + 1}/{len(todo)}  {n_tok / el:.0f} tok/s  {el / (k + 1):.1f} s/question", flush=True)


# ----------------------------------------------------------------------------- routing


def features(r):
    """s1 features of the confidence score (nan replaced by a low log-probability)."""
    f = lambda x: x if (x is not None and math.isfinite(x)) else -10.0
    return [f(r["answer_lp"]), f(r["mean_lp"]), f(r["min_lp"]), math.log1p(r["tokens"]),
            1.0 if r["pred"] is not None else 0.0, 1.0 if r["finish"] == "length" else 0.0]


def fit_logistic(X, y, steps=4000, lr=0.1, l2=1e-3):
    """Standardized logistic regression by gradient descent (no dependency)."""
    n, d = len(X), len(X[0])
    mu = [sum(x[j] for x in X) / n for j in range(d)]
    sd = [max(1e-6, (sum((x[j] - mu[j]) ** 2 for x in X) / n) ** 0.5) for j in range(d)]
    Z = [[(x[j] - mu[j]) / sd[j] for j in range(d)] for x in X]
    w, b = [0.0] * d, 0.0
    for _ in range(steps):
        gw, gb = [0.0] * d, 0.0
        for z, t in zip(Z, y):
            p = 1 / (1 + math.exp(-max(-30, min(30, b + sum(wj * zj for wj, zj in zip(w, z))))))
            e = p - t
            gb += e
            for j in range(d):
                gw[j] += e * z[j]
        b -= lr * gb / n
        w = [wj - lr * (gj / n + l2 * wj) for wj, gj in zip(w, gw)]

    def score(x):
        z = [(x[j] - mu[j]) / sd[j] for j in range(d)]
        return 1 / (1 + math.exp(-max(-30, min(30, b + sum(wj * zj for wj, zj in zip(w, z))))))

    return score, {"mu": mu, "sd": sd, "w": w, "b": b}


def route_one(s1, s2, score, tau):
    """(correct, tokens, used s1) of the routed answer of one question."""
    if score >= tau:
        return s1["correct"], s1["tokens"], True
    if s2["pred"] is None:  # System 2 gave no answer: keep the first one
        return s1["correct"], s1["tokens"] + s2["tokens"], True
    return s2["correct"], s1["tokens"] + s2["tokens"], False


def crc_tau(rows, alpha):
    """Smallest tau (most s1 answers) with (n R(tau) + 1) / (n + 1) <= alpha, where the loss of a
    question is 1 when the router is wrong and always-s2 right. The loss can only fall as tau grows
    (more questions go to s2), so the bound is monotone and the smallest feasible tau is valid."""
    n = len(rows)
    cands = sorted({sc for _, _, sc in rows}) + [1.0 + 1e-9]
    for tau in cands:
        risk = sum(max(0, int(s2["correct"]) - int(route_one(s1, s2, sc, tau)[0])) for s1, s2, sc in rows) / n
        if (n * risk + 1) / (n + 1) <= alpha:
            return tau, risk
    return cands[-1], 0.0


def pair(split, budget):
    s1 = {r["id"]: r for r in read_jsonl(f"{RUNS}/{split}.s1.jsonl")}
    s2 = {r["id"]: r for r in read_jsonl(f"{RUNS}/{split}.s2-{budget}.jsonl")}
    ids = [i for i in s1 if i in s2]
    return [(s1[i], s2[i]) for i in ids]


def summarize(rows, score, tau):
    out = {}
    for task in ("gsm8k", "math", "all"):
        sel = [(a, b) for a, b in rows if task == "all" or a["task"] == task]
        if not sel:
            continue
        n = len(sel)
        routed = [route_one(a, b, score(features(a)), tau) for a, b in sel]
        out[task] = {
            "n": n,
            "s1_acc": sum(a["correct"] for a, _ in sel) / n,
            "s2_acc": sum(b["correct"] for _, b in sel) / n,
            "router_acc": sum(r[0] for r in routed) / n,
            "oracle_acc": sum(a["correct"] or b["correct"] for a, b in sel) / n,
            "s1_tokens": sum(a["tokens"] for a, _ in sel) / n,
            "s2_tokens": sum(b["tokens"] for _, b in sel) / n,
            "router_tokens": sum(r[1] for r in routed) / n,
            "s1_share": sum(r[2] for r in routed) / n,
            "s2_forced": sum(bool(b.get("forced")) for _, b in sel) / n,
        }
    return out


def cmd_route(a):
    os.makedirs(RUNS, exist_ok=True)
    summary = {"alpha": ALPHA, "model": f"{MODEL}@{MODEL_REV}", "budgets": {}}
    best = None
    for budget in BUDGETS:
        cal = pair("calib", budget)
        if not cal:
            print(f"s2-{budget}: no calibration pairs, skipped")
            continue
        X = [features(s1) for s1, _ in cal]
        y = [1 if s1["correct"] else 0 for s1, _ in cal]
        score, params = fit_logistic(X, y)
        rows = [(s1, s2, score(features(s1))) for s1, s2 in cal]
        tau, risk = crc_tau(rows, ALPHA)
        entry = {"n_calib": len(cal), "tau": tau, "calib_risk": risk, "probe": params,
                 "calib": summarize(cal, score, tau)}
        if a.split == "test":
            test = pair("test", budget)
            entry["test"] = summarize(test, score, tau) if test else None
        summary["budgets"][str(budget)] = entry
        s2_cal = entry["calib"]["all"]["s2_acc"]
        if best is None or s2_cal > best[1]:
            best = (budget, s2_cal)
    if best is None:
        sys.exit("nothing to route: generate s1 and s2 first")
    summary["chosen_budget"] = best[0]
    with open(f"{RUNS}/summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    lines = [f"# LFM3-1.7B: System 1 first, System 2 when needed ({a.split})", "",
             f"Model `{MODEL}@{MODEL_REV[:8]}`, alpha {ALPHA}. The chosen budget (best always-s2 calibration "
             f"accuracy) is **{best[0]}** tokens.", ""]
    for budget, e in summary["budgets"].items():
        lines += [f"## s2-{budget} (tau {e['tau']:.3f}, calibration risk {e['calib_risk']:.3f})", "",
                  "| split | task | n | s1 acc | s2 acc | router acc | oracle | s1 tok | s2 tok | router tok | s1 share | s2 forced |",
                  "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for split in ("calib", "test"):
            for task, s in (e.get(split) or {}).items():
                lines.append(
                    f"| {split} | {task} | {s['n']} | {s['s1_acc']:.3f} | {s['s2_acc']:.3f} | **{s['router_acc']:.3f}** | "
                    f"{s['oracle_acc']:.3f} | {s['s1_tokens']:.0f} | {s['s2_tokens']:.0f} | {s['router_tokens']:.0f} | "
                    f"{s['s1_share']:.2f} | {s['s2_forced']:.2f} |")
        lines.append("")
    if a.split == "test":
        e = summary["budgets"][str(best[0])]["test"]["all"]
        h2 = e["router_acc"] >= e["s2_acc"] - ALPHA
        h3 = e["router_tokens"] <= 0.5 * e["s2_tokens"]
        h4 = e["router_acc"] >= max(e["s1_acc"], e["s2_acc"]) + 0.01
        lines += ["## Pre-registered claims (test, chosen budget)", "",
                  f"- H2, accuracy kept: router {e['router_acc']:.3f} >= s2 {e['s2_acc']:.3f} - {ALPHA}: **{'yes' if h2 else 'no'}**",
                  f"- H3, compute halved: router {e['router_tokens']:.0f} <= 0.5 x s2 {e['s2_tokens']:.0f} tokens: **{'yes' if h3 else 'no'}**",
                  f"- H4, better than both modes: router >= max(s1, s2) + 0.01: **{'yes' if h4 else 'no'}**",
                  f"- verdict: **{'success' if h2 and h3 else 'failure'}**{' and LFM3 improved' if h4 else ''}", ""]
    md = "\n".join(lines) + "\n"
    with open(f"{RUNS}/report.md" if a.split == "test" else f"{RUNS}/report-calib.md", "w") as f:
        f.write(md)
    print(md)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("data")
    g = sub.add_parser("gen")
    g.add_argument("--split", choices=("calib", "test"), required=True)
    g.add_argument("--mode", choices=("s1", "s2"), required=True)
    g.add_argument("--budget", type=int, default=1024)
    g.add_argument("--limit", type=int, default=0, help="first N questions only (smoke test)")
    g.add_argument("--out", help="output directory (default runs/lfm3-s1s2)")
    r = sub.add_parser("route")
    r.add_argument("--split", choices=("calib", "test"), default="calib")
    a = ap.parse_args()
    {"data": cmd_data, "gen": cmd_gen, "route": cmd_route}[a.cmd](a)


if __name__ == "__main__":
    main()
