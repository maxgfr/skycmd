"""Shared code of the LFM3-1.7B fine-tuning experiment (docs/lfm3-ft-prereg.md).

Imported by smoke.py, gen.py, sft.py and eval.py, which run on Hugging Face Jobs: scripts/lfm3-ft.sh
passes this file as a script argument, so `hf jobs uv run` ships it next to the job script (both land
in /data) and `sys.path.insert(0, dirname(__file__))` finds it.

Copied, not imported, from experiments/lfm3-s1s2/s1s2.py, with identical semantics (test_common.py
checks it against s1s2.py on the same inputs): PROMPT, last_boxed, strip_math, as_number, extract,
correct, answer_logprob, the per-question seed, GREEDY and MATH_SAMPLING, ANSWER_MAX, and the s1 / s2-B
decoding (s1: empty think, greedy, at most 1,024 tokens; s2-B: think budget B, then `</think>` forced
and the answer asked for, at most 1,024 tokens after it). What changes is the engine: vLLM (or the
transformers fallback) on bf16 weights instead of mlx-vlm on the MLX 4-bit export, prompts tokenized
from the chat template's text without adding special tokens (the template already starts with BOS).

Everything that needs torch, transformers, vllm, datasets or huggingface_hub imports it inside the
function, so the pure parts (grading, cleaning, dedup, mix) run on a laptop CPU with no dependency.
"""
import json
import math
import os
import random
import re
import unicodedata

BASE = "LiquidAI/LFM3-1.7B"
WORK = "maxgfr/skycmd-lfm3-ft-work"  # private dataset repo: artifacts between jobs
FT_REPO = "maxgfr/skycmd-lfm3-1.7b-s1"  # private model repo: merged model + adapter/
GSM8K = ("openai/gsm8k", "740312add88f781978c0658806c59bc2815b9866")
MATH500 = ("HuggingFaceH4/MATH-500", "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be")
MATH_TRAIN = ("DigitalLearningGmbH/MATH-lighteval", "0530c78699ea5e8eb5530600900e1f328b48acad")
MMLU_PRO = ("TIGER-Lab/MMLU-Pro", "b189ec765aa7ed75c8acfea42df31fdae71f97be")
SEED = 1234
ANSWER_MAX = 1024  # tokens after the thinking (s2) or in total (s1)
BUDGETS = (256, 1024)
N_MMLU = 500
N_SAMPLES = 4  # self-distillation samples per train question
SAMPLE_STRIDE = 1_000_003  # seed offset between the samples of one question
HF_FALLBACK_PER_TASK = 1000  # gen: train questions per task when the engine is the transformers fallback
MAX_LEN = 2048  # SFT: longer examples are dropped (counted), never truncated
PROMPT = "{q}\n\nSolve the problem. Put the final answer in \\boxed{{}}."
MC_PROMPT = "{q}\n\nOptions:\n{options}\n\nChoose the correct option. Put the letter of the final answer in \\boxed{{}}."
LETTERS = "ABCDEFGHIJ"
THINK_CLOSE = "</think>"
FORCE_CLOSE = "\n</think>\n\n"  # what s1s2.py appends when the budget is spent
GREEDY = {"temperature": 0.0}
MATH_SAMPLING = {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "repetition_penalty": 1.0}
# Every nn.Linear of LFM3 but lm_head (tied to the embeddings), from modeling_lfm3.py at 0dc4268b:
# attention q/k/v/o and its per-head output gate, the MLP w1/w2/w3, the short-conv in/out projections.
# smoke.py checks this set against the loaded model before any training.
LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "w1", "w2", "w3", "in_proj", "out_proj")
CUSTOM_FILES = ("configuration_lfm3.py", "modeling_lfm3.py", "lfm3_conv.py", "chat_template.jinja", "LICENSE")

# ----------------------------------------------------------------------------- io


def read_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def append_jsonl(path, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def qseed(qid):
    """The per-question seed of s1s2.py."""
    return SEED + int.from_bytes(qid.encode()[-6:], "big") % 1_000_000


# ----------------------------------------------------------------------------- grading (copied from s1s2.py)


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


# ----------------------------------------------------------------------------- grading (new: MMLU-Pro)


def extract_choice(text):
    """The letter of a multiple-choice answer: the last \\boxed{} if it names a letter A-J, else the
    last 'answer is (X)'. None if neither."""
    b = last_boxed(text)
    if b is not None:
        s = re.sub(r"\\text\{([^}]*)\}", r"\1", b).strip().strip("()$ ").upper()
        m = re.match(r"\(?([A-J])\)?(?:$|[\s.:)])", s + " ")
        if m:
            return m.group(1)
    ms = re.findall(r"answer is:?\s*\(?([A-J])\)?", text, flags=re.IGNORECASE)
    return ms[-1].upper() if ms else None


def grade(task, text, gold):
    """(pred, correct) of one generation, the same for every mode and both models."""
    if task == "mmlu_pro":
        pred = extract_choice(text)
        return pred, pred is not None and pred == gold
    pred = extract(task, text)
    return pred, correct(task, pred, gold)


# ----------------------------------------------------------------------------- data


def norm_question(q):
    """Normalized text for train/test dedup: NFKC, lower case, letters and digits only."""
    return re.sub(r"[^0-9a-z]+", "", unicodedata.normalize("NFKC", q).lower())


def clean_gsm8k_solution(sol):
    """A GSM8K worked solution as a NO-THINK target: calculator annotations <<...>> removed, the final
    '#### answer' line replaced by \\boxed{answer}."""
    body, _, ans = sol.rpartition("####")
    if not _:
        raise ValueError("GSM8K solution without ####")
    body = re.sub(r"<<[^>]*>>", "", body)
    body = "\n".join(l.rstrip() for l in body.strip().splitlines())
    return f"{body}\n\\boxed{{{ans.strip()}}}"


def gsm8k_gold(sol):
    """The gold answer of s1s2.py's data step."""
    return sol.split("####")[-1].strip()


def select_train(rows, calib, exclude_questions):
    """Train rows kept for the SFT mix and the self-distillation: not a calibration id, and whose
    normalized question is neither a calibration, nor a test, nor an MMLU-Pro evaluation question.
    Returns (kept rows, counts of each exclusion)."""
    calib_ids = {r["id"] for r in calib}
    banned = {norm_question(q) for q in exclude_questions} | {norm_question(r["question"]) for r in calib}
    kept, stats = [], {"rows": len(rows), "calib_id": 0, "same_text": 0, "kept": 0}
    for r in rows:
        if r["id"] in calib_ids:
            stats["calib_id"] += 1
        elif norm_question(r["question"]) in banned:
            stats["same_text"] += 1
        else:
            kept.append(r)
    stats["kept"] = len(kept)
    return kept, stats


def load_train_rows():
    """GSM8K train and MATH train (pinned revisions), with s1s2.py's ids: gsm8k-train-<i> and
    math-train-<i>, i the row index in the split. MATH rows without a \\boxed{} answer are skipped,
    as in s1s2.py. Each row carries its worked solution as a NO-THINK target."""
    from datasets import load_dataset

    g = load_dataset(GSM8K[0], "main", split="train", revision=GSM8K[1])
    m = load_dataset(MATH_TRAIN[0], split="train", revision=MATH_TRAIN[1])
    rows = [{"id": f"gsm8k-train-{i}", "task": "gsm8k", "question": r["question"],
             "answer": gsm8k_gold(r["answer"]), "solution": clean_gsm8k_solution(r["answer"])}
            for i, r in enumerate(g)]
    rows += [{"id": f"math-train-{i}", "task": "math", "question": r["problem"],
              "answer": last_boxed(r["solution"]), "solution": r["solution"].strip()}
             for i, r in enumerate(m) if last_boxed(r["solution"]) is not None]
    return rows


def mmlu_format(row):
    """One MMLU-Pro row as an evaluation item."""
    options = "\n".join(f"({LETTERS[i]}) {o}" for i, o in enumerate(row["options"]))
    return {"id": f"mmlupro-{row['question_id']}", "task": "mmlu_pro", "question": row["question"],
            "options": options, "answer": row["answer"].strip().upper(), "category": row.get("category")}


def mmlu_items():
    """The 500 MMLU-Pro test questions of the gate G3: a seeded sample (graine 1234) of the test split."""
    from datasets import load_dataset

    ds = load_dataset(MMLU_PRO[0], split="test", revision=MMLU_PRO[1])
    idx = sorted(random.Random(SEED).sample(range(len(ds)), N_MMLU))
    return [mmlu_format(ds[i]) for i in idx]


def hf_fallback_subset(rows, per_task=HF_FALLBACK_PER_TASK):
    """The seeded subset of train questions sampled when the transformers fallback is the engine."""
    out = []
    for task in ("gsm8k", "math"):
        sel = [r for r in rows if r["task"] == task]
        random.Random(SEED).shuffle(sel)
        out += sel[:per_task]
    return out


def user_prompt(item):
    if item["task"] == "mmlu_pro":
        return MC_PROMPT.format(q=item["question"], options=item["options"])
    return PROMPT.format(q=item["question"])


# ----------------------------------------------------------------------------- SFT mix


def distill_targets(samples):
    """Self-distillation targets from the s2-1024 samples of one or more questions.

    For each question, among its correct samples whose answer has a \\boxed{}:
      - NO-THINK: the shortest post-`</think>` answer (fewest answer tokens), stripped;
      - THINK: the shortest full trace (fewest tokens) among the samples that closed their thinking
        themselves (forced traces are cut mid-thought and are never THINK targets).
    Ties go to the lowest sample index. Returns {id: {"nothink": str|None, "think": str|None}}.
    """
    by_q = {}
    for s in samples:
        by_q.setdefault(s["id"], []).append(s)
    out = {}
    for qid, ss in by_q.items():
        ok = sorted((s for s in ss if s["correct"] and last_boxed(s["text"]) is not None), key=lambda s: s["k"])
        if not ok:
            continue
        a = min(ok, key=lambda s: (s["answer_tokens"], s["k"]))
        full = [s for s in ok if not s["forced"]]
        t = min(full, key=lambda s: (s["tokens"], s["k"])) if full else None
        out[qid] = {
            "nothink": a["text"].strip(),
            "think": (t["think"] + THINK_CLOSE + t["text"]) if t else None,
        }
    return out


def build_mix(train_rows, samples):
    """The SFT examples: (a) every kept train row's worked solution as a NO-THINK target, (b) the
    self-distillation targets of distill_targets. Each example is {id, kind, thinking, completion}
    (the prompt is the chat template of the question with thinking on or off)."""
    kept = {r["id"] for r in train_rows}
    mix = [{"id": r["id"], "kind": "gold_nothink", "thinking": False, "question": r["question"],
            "task": r["task"], "completion": r["solution"]} for r in train_rows]
    qs = {r["id"]: r for r in train_rows}
    for qid, t in sorted(distill_targets([s for s in samples if s["id"] in kept]).items()):
        r = qs[qid]
        if t["nothink"]:
            mix.append({"id": qid, "kind": "distill_nothink", "thinking": False, "question": r["question"],
                        "task": r["task"], "completion": t["nothink"]})
        if t["think"]:
            mix.append({"id": qid, "kind": "distill_think", "thinking": True, "question": r["question"],
                        "task": r["task"], "completion": t["think"]})
    return mix


# ----------------------------------------------------------------------------- tokenizer and prompts


def load_tokenizer(model, revision):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model, revision=revision, trust_remote_code=True)


def chat_text(tok, item, thinking):
    msgs = [{"role": "user", "content": user_prompt(item)}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=thinking)


def encode(tok, text):
    return list(tok(text, add_special_tokens=False)["input_ids"])


def decode(tok, ids):
    return tok.decode(ids, skip_special_tokens=False)


def pieces_from_ids(tok, ids, lps):
    """(text piece, log-probability) per generated token, like mlx-vlm's stream: each piece is what
    the token adds to the decoded text (empty while a multi-byte character is incomplete)."""
    pieces, prev = [], ""
    for k, lp in enumerate(lps):
        cur = decode(tok, ids[: k + 1])
        if cur.endswith("\ufffd") or not cur.startswith(prev):
            pieces.append(("", lp))
            continue
        pieces.append((cur[len(prev):], lp))
        prev = cur
    return pieces


# ----------------------------------------------------------------------------- engines


class Req:
    def __init__(self, prompt_ids, max_tokens, sampling, seed, stop=(), logprobs=False):
        self.prompt_ids, self.max_tokens, self.sampling = list(prompt_ids), max_tokens, dict(sampling)
        self.seed, self.stop, self.logprobs = seed, tuple(stop), logprobs


class Out:
    """ids and text without the end-of-sequence token; finish 'stop' or 'length'."""

    def __init__(self, ids, text, lps, finish):
        self.ids, self.text, self.lps, self.finish = ids, text, lps, finish


class VLLMEngine:
    """vLLM offline engine. Log-probabilities are vLLM's of the sampled token (logprobs=1, raw model
    distribution). Every request carries its own seed."""

    name = "vllm"

    def __init__(self, model, revision, max_model_len=4096):
        from vllm import LLM

        self.tok = load_tokenizer(model, revision)
        self.eos = self.tok.eos_token_id
        self.llm = LLM(model=model, revision=revision, tokenizer_revision=revision, trust_remote_code=True,
                       dtype="bfloat16", max_model_len=max_model_len, seed=SEED, generation_config="vllm",
                       enable_prefix_caching=True, gpu_memory_utilization=0.85)

    def generate(self, reqs):
        from vllm import SamplingParams

        params = [SamplingParams(n=1, max_tokens=r.max_tokens, seed=r.seed, stop=list(r.stop) or None,
                                 include_stop_str_in_output=True, skip_special_tokens=False,
                                 logprobs=1 if r.logprobs else None, **r.sampling) for r in reqs]
        outs = self.llm.generate([{"prompt_token_ids": r.prompt_ids} for r in reqs], params, use_tqdm=True)
        res = []
        for r, o in zip(reqs, outs):
            c = o.outputs[0]
            ids = list(c.token_ids)
            lps = [c.logprobs[i][t].logprob for i, t in enumerate(ids)] if r.logprobs else None
            if ids and ids[-1] == self.eos:
                ids = ids[:-1]
                lps = lps[:-1] if lps is not None else None
            res.append(Out(ids, decode(self.tok, ids), lps, "length" if c.finish_reason == "length" else "stop"))
        return res


class HFEngine:
    """transformers generate, the fallback when vLLM cannot load LFM3. Requests are grouped by
    sampling parameters, sorted by prompt length and batched (left padding); a batch is seeded with
    the seed of its first request. Log-probabilities come from one teacher-forced forward pass over
    the finished sequence (raw distribution, like vLLM's)."""

    name = "hf"

    def __init__(self, model, revision, batch_size=128):
        import torch
        from transformers import AutoModelForCausalLM

        self.torch = torch
        self.tok = load_tokenizer(model, revision)
        self.tok.padding_side = "left"
        self.eos, self.pad = self.tok.eos_token_id, self.tok.pad_token_id
        self.model = AutoModelForCausalLM.from_pretrained(model, revision=revision, trust_remote_code=True,
                                                          dtype=torch.bfloat16, device_map="cuda")
        self.model.eval()
        self.batch_size = batch_size

    def _score(self, prompt_ids, ids):
        torch = self.torch
        if not ids:
            return []
        x = torch.tensor([prompt_ids + ids], device=self.model.device)
        with torch.no_grad():
            logits = self.model(input_ids=x).logits[0, len(prompt_ids) - 1 : -1].float()
        lp = torch.log_softmax(logits, dim=-1).gather(1, torch.tensor(ids, device=x.device)[:, None])
        return lp[:, 0].tolist()

    def generate(self, reqs):
        torch = self.torch
        key = lambda r: (tuple(sorted(r.sampling.items())), r.max_tokens, r.stop, r.logprobs)
        order = sorted(range(len(reqs)), key=lambda i: (str(key(reqs[i])), len(reqs[i].prompt_ids)))
        res = [None] * len(reqs)
        i = 0
        while i < len(order):
            j = i
            while j < len(order) and j - i < self.batch_size and key(reqs[order[j]]) == key(reqs[order[i]]):
                j += 1
            batch = [reqs[k] for k in order[i:j]]
            r0 = batch[0]
            L = max(len(r.prompt_ids) for r in batch)
            x = torch.tensor([[self.pad] * (L - len(r.prompt_ids)) + r.prompt_ids for r in batch], device="cuda")
            mask = torch.tensor([[0] * (L - len(r.prompt_ids)) + [1] * len(r.prompt_ids) for r in batch], device="cuda")
            sample = r0.sampling.get("temperature", 0.0) > 0
            kw = dict(max_new_tokens=r0.max_tokens, do_sample=sample, repetition_penalty=r0.sampling.get("repetition_penalty", 1.0),
                      eos_token_id=self.eos, pad_token_id=self.pad)
            if sample:
                kw.update(temperature=r0.sampling["temperature"], top_p=r0.sampling.get("top_p", 1.0),
                          top_k=r0.sampling.get("top_k", 0) or None)
            else:
                kw.update(temperature=None, top_p=None, top_k=None)
            if r0.stop:
                kw.update(stop_strings=list(r0.stop), tokenizer=self.tok)
            torch.manual_seed(r0.seed)
            with torch.no_grad():
                seqs = self.model.generate(input_ids=x, attention_mask=mask, **kw)[:, L:].tolist()
            for k, r, row in zip(order[i:j], batch, seqs):
                cut, finish = len(row), "length"
                for t, tid in enumerate(row):
                    if tid in (self.eos, self.pad):
                        cut, finish = t, "stop"
                        break
                ids = row[:cut]
                text = decode(self.tok, ids)
                if finish == "length" and any(text.endswith(s) for s in r.stop):
                    finish = "stop"
                lps = self._score(r.prompt_ids, ids) if r.logprobs else None
                res[k] = Out(ids, text, lps, finish)
            i = j
        return res


def make_engine(name, model, revision, **kw):
    return VLLMEngine(model, revision, **kw) if name == "vllm" else HFEngine(model, revision, **kw)


# ----------------------------------------------------------------------------- the modes


def finite(xs):
    return [x for x in xs if x is not None and math.isfinite(x)]


def s1_record(tok, o):
    """The s1 record of s1s2.py's gen_s1, from one engine output."""
    lps = finite(o.lps or [])
    return {
        "text": o.text,
        "tokens": len(o.ids),
        "think_tokens": 0,
        "finish": o.finish,
        "mean_lp": sum(lps) / len(lps) if lps else float("nan"),
        "min_lp": min(lps) if lps else float("nan"),
        "answer_lp": answer_logprob(pieces_from_ids(tok, o.ids, o.lps or [])),
    }


def run_s1(engine, items, keep_ids=False):
    """s1 on every item: empty think, greedy, at most ANSWER_MAX tokens, with log-probabilities."""
    tok = engine.tok
    reqs = [Req(encode(tok, chat_text(tok, it, False)), ANSWER_MAX, GREEDY, qseed(it["id"]), logprobs=True)
            for it in items]
    recs = []
    for o in engine.generate(reqs):
        r = s1_record(tok, o)
        if keep_ids:
            r["ids"] = o.ids
        recs.append(r)
    return recs


def run_s2(engine, items, budget, seeds=None):
    """s2-B on every item: phase 1 thinks up to `budget` tokens and stops at `</think>`, phase 2 writes
    the answer (at most ANSWER_MAX tokens). Returns s1s2.py's s2 records."""
    tok = engine.tok
    seeds = seeds or [qseed(it["id"]) for it in items]
    prompts = [encode(tok, chat_text(tok, it, True)) for it in items]
    first = engine.generate([Req(p, budget, MATH_SAMPLING, s, stop=(THINK_CLOSE,)) for p, s in zip(prompts, seeds)])
    force_ids = encode(tok, FORCE_CLOSE)
    second, meta = [], []
    for p, s, o in zip(prompts, seeds, first):
        closed = THINK_CLOSE in o.text
        if closed:
            head, tail = o.text.split(THINK_CLOSE, 1)
            cont = p + o.ids
        else:
            head, tail = o.text, ""
            cont = p + o.ids + force_ids
        second.append(Req(cont, ANSWER_MAX, MATH_SAMPLING, s + 1))
        meta.append((closed, head, tail, len(o.ids)))
    outs = engine.generate(second)
    recs = []
    for (closed, head, tail, n_think), o in zip(meta, outs):
        recs.append({
            "text": tail + o.text,
            "think": head,
            "tokens": n_think + len(o.ids),
            "think_tokens": n_think,
            "answer_tokens": len(o.ids),
            "forced": not closed,
            "finish": o.finish,
        })
    return recs


def finish_records(items, recs, mode, seconds):
    """id, task, mode, pred, correct of each record, like s1s2.py's cmd_gen."""
    out = []
    for it, r in zip(items, recs):
        pred, ok = grade(it["task"], r["text"], it["answer"])
        r.update(id=it["id"], task=it["task"], mode=mode, pred=pred, correct=ok, seconds=seconds / max(1, len(items)))
        out.append(r)
    return out


# ----------------------------------------------------------------------------- hub


def hub_download(repo, path, local_dir, repo_type="dataset"):
    """One file of a repo, or None if it is not there yet."""
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError, RepositoryNotFoundError

    try:
        return hf_hub_download(repo, path, repo_type=repo_type, local_dir=local_dir)
    except (EntryNotFoundError, RepositoryNotFoundError):
        return None


def hub_parts(repo, prefix, local_dir):
    """All rows of the jsonl parts under prefix/ in the work repo (empty if none yet)."""
    import glob

    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import RepositoryNotFoundError

    try:
        snapshot_download(repo, repo_type="dataset", allow_patterns=[f"{prefix}/*.jsonl"], local_dir=local_dir)
    except RepositoryNotFoundError:
        return []
    rows = []
    for p in sorted(glob.glob(f"{local_dir}/{prefix}/*.jsonl")):
        rows += read_jsonl(p)
    return rows


def ensure_private(repo, repo_type):
    """Create the repo private if missing; refuse to write to it if it is public."""
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo, repo_type=repo_type, private=True, exist_ok=True)
    if not api.repo_info(repo, repo_type=repo_type).private:
        raise SystemExit(f"{repo} is public: nothing is written to it (docs/lfm3-ft-prereg.md §9)")
    return api


def hub_upload(repo, local_path, path_in_repo, message, repo_type="dataset"):
    from huggingface_hub import HfApi

    api = ensure_private(repo, repo_type)
    return api.upload_file(path_or_fileobj=local_path, path_in_repo=path_in_repo, repo_id=repo,
                           repo_type=repo_type, commit_message=message)


def versions():
    out = {}
    for m in ("torch", "transformers", "vllm", "trl", "peft", "datasets", "huggingface_hub", "accelerate"):
        try:
            out[m] = __import__(m).__version__
        except Exception:
            pass
    return out


def load_shipped(path):
    """A data file shipped as a job argument (`hf jobs uv run` mounts it under /data)."""
    if not path or not os.path.exists(path):
        raise SystemExit(f"{path}: missing (scripts/lfm3-ft.sh ships data/lfm3/*.jsonl with the job)")
    return read_jsonl(path)


def sha256(path):
    import hashlib

    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


# ----------------------------------------------------------------------------- training (sft.py, smoke.py)


def tokenize_mix(tok, mix, max_len=MAX_LEN):
    """Pre-tokenized SFT rows: the prompt is the chat template of the question (thinking on or off, as at
    inference), the completion its target plus the end-of-turn token; the loss is on the completion
    only (completion_mask). Examples longer than max_len are dropped, never truncated."""
    eos = tok.eos_token
    rows, dropped = [], {}
    for ex in mix:
        p = encode(tok, chat_text(tok, ex, ex["thinking"]))
        c = encode(tok, ex["completion"] + eos)
        if len(p) + len(c) > max_len:
            dropped[ex["kind"]] = dropped.get(ex["kind"], 0) + 1
            continue
        rows.append({"input_ids": p + c, "completion_mask": [0] * len(p) + [1] * len(c), "kind": ex["kind"]})
    return rows, dropped


def make_trainer(model_id, revision, rows, out_dir, max_steps=-1, epochs=2, per_device=8, grad_accum=4):
    """TRL SFTTrainer with the pre-registered LoRA (r 32, alpha 64, dropout 0.05, LORA_TARGETS), 2 epochs,
    lr 1e-4 cosine, warmup 3 % of the steps, bf16, seed 1234, completion-only loss, no intermediate
    checkpoint (the last one is the model)."""
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM
    from trl import SFTConfig, SFTTrainer

    tok = load_tokenizer(model_id, revision)
    model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision, trust_remote_code=True,
                                                 dtype=torch.bfloat16, attn_implementation="sdpa")
    model.config.use_cache = False
    ds = Dataset.from_list([{"input_ids": r["input_ids"], "completion_mask": r["completion_mask"]} for r in rows])
    total = max_steps if max_steps > 0 else math.ceil(len(ds) / (per_device * grad_accum)) * epochs
    cfg = SFTConfig(
        output_dir=out_dir, num_train_epochs=epochs, max_steps=max_steps, learning_rate=1e-4,
        lr_scheduler_type="cosine", warmup_steps=math.ceil(0.03 * total), per_device_train_batch_size=per_device,
        gradient_accumulation_steps=grad_accum, bf16=True, seed=SEED, data_seed=SEED, max_length=MAX_LEN,
        completion_only_loss=True, packing=False, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False}, logging_steps=10, save_strategy="no",
        report_to="none",
    )
    lora = LoraConfig(r=32, lora_alpha=64, lora_dropout=0.05, target_modules=list(LORA_TARGETS), bias="none",
                      task_type="CAUSAL_LM")
    trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds, processing_class=tok, peft_config=lora)
    return trainer, tok, total


def check_completion_mask(trainer, row):
    """The collator masks the prompt (-100) and keeps the completion: True if the labels of `row` are
    exactly its input ids on the completion and -100 elsewhere."""
    b = trainer.data_collator([{"input_ids": row["input_ids"], "completion_mask": row["completion_mask"]}])
    labels, ids = b["labels"][0].tolist(), b["input_ids"][0].tolist()
    n = len(row["input_ids"])
    ok_prompt = all(l == -100 for l, m in zip(labels[:n], row["completion_mask"]) if m == 0)
    ok_compl = all(l == i for l, i, m in zip(labels[:n], ids[:n], row["completion_mask"]) if m == 1)
    return ok_prompt and ok_compl and all(l == -100 for l in labels[n:])


def linear_names(model):
    """Leaf names of every nn.Linear of a model, lm_head excluded."""
    import torch

    return sorted({n.split(".")[-1] for n, m in model.named_modules()
                   if isinstance(m, torch.nn.Linear) and not n.endswith("lm_head")})


MODEL_CARD = """---
license: other
license_name: lfm1.0
license_link: LICENSE
base_model: {base}
base_model_relation: finetune
library_name: transformers
tags:
- custom_code
- research
---

# skycmd-lfm3-1.7b-s1 (private research checkpoint)

Fine-tuned from [{base}](https://huggingface.co/{base}) (revision `{base_rev}`) by maxgfr for the
skycmd experiment `docs/lfm3-ft-prereg.md` (https://github.com/maxgfr/skycmd). **Not an official Liquid AI
model**, not affiliated with or endorsed by Liquid AI. The LFM Open License v1.0 grants no trademark
rights: this checkpoint is not "LFM3", it is a model fine-tuned from it.

- **Licence**: LFM Open License v1.0 (`LICENSE`, unmodified copy). Commercial use only within its terms
  (organisations under the licence's revenue threshold). See `NOTICE`.
- **Training**: LoRA r 32 on every linear projection, 2 epochs, merged into the weights (`adapter/` holds
  the adapter). Data: GSM8K train and MATH train worked solutions (MIT) and the base model's own correct
  answers on those train questions (self-distillation). No test question, no calibration question.
- **Modified files**: `model.safetensors` (weights changed by the fine-tuning), `config.json` and
  `generation_config.json` (re-saved by transformers), `README.md` and `NOTICE` (new), `adapter/` (new).
  Unmodified copies of the base repository: `modeling_lfm3.py`, `configuration_lfm3.py`, `lfm3_conv.py`,
  `chat_template.jinja`, the tokenizer files, `LICENSE`.
"""

NOTICE = """This repository contains a modified version of {base} (revision {base_rev}),
Copyright Liquid AI, licensed under the LFM Open License v1.0 (see LICENSE).

Modified by maxgfr on {date}: LoRA fine-tuning (skycmd, docs/lfm3-ft-prereg.md), adapter merged into the
weights. The files changed or added are listed in README.md. "LFM" and "Liquid" are trademarks of
Liquid AI; no trademark right is granted or claimed.
"""


def merge_and_save(model_id, revision, adapter_dir, out_dir):
    """Merge the adapter into bf16 base weights and write a loadable repository: weights, tokenizer,
    the base repo's custom code, chat template and LICENSE (unmodified copies), README and NOTICE."""
    import shutil
    import time as _time

    import torch
    from huggingface_hub import snapshot_download
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    base = AutoModelForCausalLM.from_pretrained(model_id, revision=revision, trust_remote_code=True,
                                                dtype=torch.bfloat16)
    merged = PeftModel.from_pretrained(base, adapter_dir).merge_and_unload()
    merged.save_pretrained(out_dir, safe_serialization=True)
    load_tokenizer(model_id, revision).save_pretrained(out_dir)
    snap = snapshot_download(model_id, revision=revision, allow_patterns=list(CUSTOM_FILES) + ["config.json"])
    for f in CUSTOM_FILES:
        if os.path.exists(os.path.join(snap, f)):
            shutil.copy(os.path.join(snap, f), os.path.join(out_dir, f))
    with open(os.path.join(snap, "config.json")) as f:
        auto_map = json.load(f)["auto_map"]
    with open(os.path.join(out_dir, "config.json")) as f:
        cfg = json.load(f)
    cfg["auto_map"] = auto_map  # the local copies, never a pointer back to the base repo
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    fill = {"base": model_id, "base_rev": revision, "date": _time.strftime("%Y-%m-%d", _time.gmtime())}
    with open(os.path.join(out_dir, "README.md"), "w") as f:
        f.write(MODEL_CARD.format(**fill))
    with open(os.path.join(out_dir, "NOTICE"), "w") as f:
        f.write(NOTICE.format(**fill))
    shutil.copytree(adapter_dir, os.path.join(out_dir, "adapter"), dirs_exist_ok=True)
    return out_dir
