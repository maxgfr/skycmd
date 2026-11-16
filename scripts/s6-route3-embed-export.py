#!/usr/bin/env python3
"""s6-nano v3 (docs/s6-nano3-prereg.md, section 2 and section 10 item 2): the raw export of the
embedding part `potion-cos-4bit`, exactly as scripts/route-baselines.py computes it, for the Rust
command that writes the engine's embedding file, and the fixture the engine must reproduce.

    nice -n 19 ../skycmd-data/s6-route/envs/route/bin/python -I scripts/s6-route3-embed-export.py \
        --data ../skycmd-data/s6-route --out ../skycmd-data/s6-route/embed/potion-cos-4bit \
        --fixture crates/sky-infer/tests/fixtures/route_embed.json
    nice -n 19 $CARGO_TARGET_DIR/release/sky-route embed-gguf \
        --dump ../skycmd-data/s6-route/embed/potion-cos-4bit \
        --out models/s6-nano3/skycmd-s6-route-embed.gguf

What the embedding part is (model2vec 0.10.0, `StaticModel.from_pretrained` of the snapshot of
`minishlab/potion-base-2M`, then `Packed` of scripts/route-baselines.py):

- tokens: the snapshot's `tokenizer.json` (BertNormalizer: clean text, CJK characters padded,
  accents stripped, lower case; BertPreTokenizer; WordPiece, `##`, at most 100 characters per
  word) through `StaticModel.tokenize`: no [CLS]/[SEP] added, truncated to the first 512 tokens
  (model2vec's default `max_length`, set on the tokenizer), then every [UNK] removed. The literal
  texts `[PAD]` `[UNK]` `[CLS]` `[SEP]` `[MASK]` are matched in the raw text (added tokens).
- vectors: each token's row of the table (no token mapping, no weights in this model), packed per
  token: scale = max |v| / 7 rounded to float16, codes = rint(v / scale) clipped to -7..7, value
  = code * scale (float32);
- a text: the mean of its tokens' vectors (float32), L2-normalised (the model's `normalize`); a
  text without any token is the zero vector;
- a slate: the cosine (dot product of the normalised vectors) of the request with each tool
  option `name: description`, *none* at theta, probabilities softmax([cosines, theta] / T), with
  theta and T from <data>/eval/fitted/potion-cos-4bit/params.json.

The normaliser is exported as a per-character map read from the tokenizers library itself (every
code point through `normalizer.normalize_str`), as runs of code points: identity, deleted,
padded with spaces (CJK), shifted by a constant, replaced by a string, or a Hangul syllable
decomposed by the Unicode algorithm. The pre-tokenizer's classes (whitespace, punctuation) are
read the same way on every character the normaliser can output. The map is applied character by
character: this script checks that this equals the library's normaliser on every text it
exports (NFD's canonical reordering of combining marks is the one place where a per-character
map could differ; the check counts it).

Outputs in --out: codes.bin (n x 32 bytes, two 4-bit codes per byte, the even dimension in the
low nibble, two's complement), scales.bin (n float16, little-endian), vocab.txt (the tokens in id
order, one per line), tables.json (the normaliser and pre-tokenizer tables), meta.json (the
parameters, provenance and checks), tokens.jsonl (token ids of every validation text, for the
engine's full parity check; validation records only, no test or fresh file is read).
"""
import argparse
import hashlib
import importlib.util
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
FIT = os.path.join(REPO, "runs", "s6-nano3", "fit.json")
SYSTEM = "potion-cos-4bit"

# ops of the normaliser runs (the Rust side reads the same numbers: sky_infer::embed)
OP_ID, OP_DEL, OP_PAD, OP_DELTA, OP_STR, OP_HANGUL = range(6)

# hand-written edge cases of the fixture (no validation text needed for these)
EDGE = [
    "",
    "   ",
    "Réveille-moi à sept heures demain, s'il te plaît.",
    "Quelle est la météo à Paris ? Ça caille !",
    "Mets de la musique de Céline Dion, s'il te plaît",
    "Envoie un e-mail à Zoé : « je serai en retard »",
    "Où est le rendez-vous de jeudi ? Rappelle-moi l'heure.",
    "naïve façade coöperate Æsir œuvre Straße ÉCOLE",
    "İstanbul'da hava nasıl? ΣΊΣΥΦΟΣ",
    "Ελληνικά: Καλημέρα κόσμε, τελικό σίγμα ς",
    "日本語のテキストを翻訳して 東京の天気",
    "오늘 서울 날씨 어때?",
    "Привет, как дела? Включи свет.",
    "مرحبا، كيف حالك؟",
    "हिन्दी में अनुवाद करें",
    "Emoji test \U0001F600\U0001F389 ❤️ thumbs \U0001F44D\U0001F3FD",
    "Fullwidth ＡＢＣ１２３！？",
    "tabs\tand\nnewlines\r\nand nbsp em space",
    "zero​width‍joiner﻿bom­soft",
    "control \x00\x01\x7f chars � replacement",
    "[CLS] literal [SEP] tokens [UNK] [MASK][PAD] and [cls] lower [CLS]x[SEP]",
    "é combining acute, ä ộ stacked x̧́ ̧́",
    "punctuation!!! ... ??? --- ((brackets)) {braces} <angle> \"quotes\" 'single' `back` ~tilde^ caret|pipe\\backslash",
    "numbers 3.14159 1,000,000 -42 +7 50% $9.99 €12 £3 ¥500 ½ ² ℵ",
    "ǅungla ǈ ﬁ ligature ﬀ ẞ capital sharp s ℌ",
    "a" * 100,
    "a" * 101,
    "supercalifragilisticexpialidocious antidisestablishmentarianism",
    "unassigned ͸ private  tag \U000E0001 noncharacter ﷐",
    "turn on the lights in the hall " * 90,
    ("Réservez une table pour deux ce soir à vingt heures, près de la gare, s'il vous plaît; " * 30),
]

KEYLIKE = re.compile(r"(?i)(sk-|ghp_|xox[bap]-|AKIA|<KEY>|api[_-]?key|secret|token=)|[A-Za-z0-9_\-]{24,}")


def load_baselines():
    spec = importlib.util.spec_from_file_location("route_baselines", os.path.join(HERE, "route-baselines.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def request_of(rec):
    s = rec["state"]
    s = s.rsplit("\nBM25: ", 1)[0] if "\nBM25: " in s else s
    return s[len("User: "):] if s.startswith("User: ") else s


# ---------------------------------------------------------------- the packed table

def pack(static, rb):
    if static.token_mapping is not None or static.weights is not None:
        raise SystemExit("this model has a token mapping or weights: not handled by the export")
    e = np.asarray(static.embedding, dtype=np.float32)
    lim = 7
    scale16 = (np.abs(e).max(axis=1) / lim).astype(np.float16)
    scale = scale16.astype(np.float32)
    safe = np.where(scale == 0, 1, scale)
    codes = np.clip(np.rint(e / safe[:, None]), -lim, lim)
    value = (codes * scale[:, None]).astype(np.float32)
    ref = rb.Packed(static)
    if not np.array_equal(ref.vec, value):
        raise SystemExit("the packing differs from route-baselines.Packed")
    c = codes.astype(np.int8)
    if c.shape[1] % 2:
        raise SystemExit("odd dimension")
    lo = (c[:, 0::2].astype(np.uint8) & 0xF)
    hi = (c[:, 1::2].astype(np.uint8) & 0xF) << 4
    packed = (lo | hi).astype(np.uint8)
    return packed, scale16, ref


# ---------------------------------------------------------------- the normaliser as a table

def hangul(c):
    s = c - 0xAC00
    l, v, t = 0x1100 + s // 588, 0x1161 + (s % 588) // 28, 0x11A7 + s % 28
    return chr(l) + chr(v) + (chr(t) if t != 0x11A7 else "")


def tables(tok):
    norm, pre = tok.normalizer, tok.pre_tokenizer
    ops = []
    strings = []
    outs = set()
    for c in range(0x110000):
        if 0xD800 <= c <= 0xDFFF:
            ops.append(None)
            continue
        s = chr(c)
        o = norm.normalize_str(s)
        outs.update(o)
        if o == s:
            ops.append((OP_ID, 0))
        elif o == "":
            ops.append((OP_DEL, 0))
        elif o == " " + s + " ":
            ops.append((OP_PAD, 0))
        elif 0xAC00 <= c <= 0xD7A3 and o == hangul(c):
            ops.append((OP_HANGUL, 0))
        elif len(o) == 1:
            ops.append((OP_DELTA, ord(o) - c))
        else:
            ops.append((OP_STR, o))
    starts, kinds, args = [], [], []
    prev = None
    for c, op in enumerate(ops):
        if op is None:  # surrogates: never in a text; continue the run before
            continue
        if op[0] == OP_STR:
            starts.append(c)
            kinds.append(OP_STR)
            args.append(len(strings))
            strings.append(op[1])
            prev = None
            continue
        if op != prev:
            starts.append(c)
            kinds.append(op[0])
            args.append(op[1])
            prev = op
    space, punct = [], []
    for ch in sorted(outs):
        r = pre.pre_tokenize_str("a" + ch + "a")
        if len(r) == 2:
            space.append(ord(ch))
        elif len(r) == 3:
            punct.append(ord(ch))
        elif len(r) != 1:
            raise SystemExit(f"pre-tokenizer: unexpected split of U+{ord(ch):04X}: {r}")
    t = {"starts": starts, "ops": kinds, "args": args, "strings": strings,
         "space": ranges(space), "punct": ranges(punct)}
    return t


def ranges(xs):
    out = []
    for x in xs:
        if out and out[-1][1] == x - 1:
            out[-1][1] = x
        else:
            out.append([x, x])
    return out


class Mirror:
    """The engine's algorithm in Python, on the exported tables (sky_infer::embed mirrors it)."""

    def __init__(self, t, vocab, added, unk, max_chars, max_len):
        self.t = t
        self.vocab = vocab
        self.added = added
        self.unk = unk
        self.max_chars = max_chars
        self.max_len = max_len
        self.space = {c for a, b in t["space"] for c in range(a, b + 1)}
        self.punct = {c for a, b in t["punct"] for c in range(a, b + 1)}
        import bisect
        self.bisect = bisect

    def norm_char(self, ch):
        c = ord(ch)
        i = self.bisect.bisect_right(self.t["starts"], c) - 1
        op, arg = self.t["ops"][i], self.t["args"][i]
        if op == OP_ID:
            return ch
        if op == OP_DEL:
            return ""
        if op == OP_PAD:
            return " " + ch + " "
        if op == OP_DELTA:
            return chr(c + arg)
        if op == OP_STR:
            return self.t["strings"][arg]
        return hangul(c)

    def normalize(self, s):
        return "".join(self.norm_char(ch) for ch in s)

    def words(self, s):
        out, cur = [], []
        for ch in s:
            c = ord(ch)
            if c in self.space:
                if cur:
                    out.append("".join(cur))
                    cur = []
            elif c in self.punct:
                if cur:
                    out.append("".join(cur))
                    cur = []
                out.append(ch)
            else:
                cur.append(ch)
        if cur:
            out.append("".join(cur))
        return out

    def wordpiece(self, w):
        if len(w) > self.max_chars:
            return [self.unk]
        out, start = [], 0
        while start < len(w):
            end = len(w)
            hit = None
            while start < end:
                sub = w[start:end] if start == 0 else "##" + w[start:end]
                if sub in self.vocab:
                    hit = self.vocab[sub]
                    break
                end -= 1
            if hit is None:
                return [self.unk]
            out.append(hit)
            start = end
        return out

    def segments(self, text):
        i, seg, out = 0, 0, []
        while i < len(text):
            for a, aid in self.added:
                if text.startswith(a, i):
                    out.append((text[seg:i], None))
                    out.append((None, aid))
                    i += len(a)
                    seg = i
                    break
            else:
                i += 1
        out.append((text[seg:], None))
        return out

    def ids(self, text):
        ids = []
        for s, aid in self.segments(text):
            if aid is not None:
                ids.append(aid)
                continue
            for w in self.words(self.normalize(s)):
                ids.extend(self.wordpiece(w))
        return [i for i in ids[: self.max_len] if i != self.unk]


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="the s6-route data folder")
    ap.add_argument("--out", required=True, help="the raw export folder (under <data>/embed/)")
    ap.add_argument("--fixture", help="write the engine's fixture here")
    ap.add_argument("--fixture-slates", type=int, default=40)
    ap.add_argument("--fixture-ids", type=int, default=150)
    args = ap.parse_args()
    data = os.path.abspath(args.data)
    os.environ["ROUTE_DATA"] = data
    rb = load_baselines()
    from model2vec import StaticModel
    import model2vec
    import tokenizers

    model_dir = os.path.join(data, "models", "potion-base-2M")
    static = StaticModel.from_pretrained(model_dir)
    tok = static.tokenizer
    params = json.load(open(os.path.join(data, "eval", "fitted", SYSTEM, "params.json")))
    fit = json.load(open(FIT))
    fetch = json.load(open(os.path.join(model_dir, ".skycmd-fetch.json")))
    readme = open(os.path.join(model_dir, "README.md"), encoding="utf-8").read()
    lic = re.search(r"(?m)^license:\s*(\S+)", readme)
    if not lic or lic.group(1).lower() != "mit":
        raise SystemExit("the model card does not say MIT")

    packed, scale16, ref = pack(static, rb)
    n, dim = static.embedding.shape
    vocab = tok.get_vocab()
    tokens = [t for t, _ in sorted(vocab.items(), key=lambda x: x[1])]
    if [vocab[t] for t in tokens] != list(range(n)) or any(("\n" in t) or (" " in t) for t in tokens):
        raise SystemExit("the vocabulary is not 0..n-1 or holds a line break")
    tj = json.load(open(os.path.join(model_dir, "tokenizer.json"), encoding="utf-8"))
    added = sorted([(a["content"], a["id"]) for a in tj["added_tokens"]], key=lambda x: -len(x[0]))
    if any(a.get("normalized") or a.get("lstrip") or a.get("rstrip") or a.get("single_word") for a in tj["added_tokens"]):
        raise SystemExit("an added token is normalised or stripped: not handled")
    wp = tj["model"]
    trunc = tok.truncation
    if static.max_length != 512 or not trunc or trunc["max_length"] != 512 or trunc["direction"] != "right":
        raise SystemExit(f"unexpected truncation {trunc} / max_length {static.max_length}")
    t = tables(tok)
    mirror = Mirror(t, vocab, added, static.unk_token_id, wp["max_input_chars_per_word"], 512)

    # every validation text: the library's ids, the per-character normaliser, the mirror
    texts = {}
    for f in ("v_tools", "v_tools_masked", "v_tools_pairs", "v_tools_swapped"):
        for r in read_jsonl(os.path.join(data, "build", "val", f + ".jsonl")):
            texts[request_of(r)] = "slates"
            for o in r["opts"][:-1]:
                texts.setdefault(o, "slates")
    for kind in ("tools", "intents", "when2call"):
        for r in read_jsonl(os.path.join(data, "build", "records", kind, "val.jsonl")):
            texts.setdefault(request_of(r), "records-" + kind)
    for e in EDGE:
        texts[e] = "edge"
    norm = tok.normalizer
    all_texts = list(texts)
    lib_ids = static.tokenize(all_texts)
    bad_norm, bad_ids = [], []
    for s, ids in zip(all_texts, lib_ids):
        if mirror.normalize(s) != norm.normalize_str(s):
            bad_norm.append(s)
        if mirror.ids(s) != list(ids):
            bad_ids.append(s)
    print(f"[export] {len(all_texts)} texts: per-character normaliser differs on {len(bad_norm)}, "
          f"mirror ids differ on {len(bad_ids)}", file=sys.stderr)
    for s in bad_ids[:10]:
        print(f"[export]   ids differ: {s[:120]!r}", file=sys.stderr)

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "codes.bin"), "wb") as f:
        f.write(packed.tobytes())
    with open(os.path.join(args.out, "scales.bin"), "wb") as f:
        f.write(scale16.astype("<f2").tobytes())
    with open(os.path.join(args.out, "vocab.txt"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(tokens))
    with open(os.path.join(args.out, "tables.json"), "w", encoding="utf-8") as f:
        json.dump(t, f, ensure_ascii=False)
    with open(os.path.join(args.out, "tokens.jsonl"), "w", encoding="utf-8") as f:
        for s, ids in zip(all_texts, lib_ids):
            f.write(json.dumps({"from": texts[s], "text": s, "ids": list(ids)}, ensure_ascii=False) + "\n")
    meta = {
        "about": "docs/s6-nano3-prereg.md section 2: the embedding part potion-cos-4bit, exported by scripts/s6-route3-embed-export.py",
        "system": SYSTEM,
        "source": {"repo": fetch["repo"], "commit": fetch["commit"], "license": "MIT",
                   "license_from": "model card front matter (README.md: license: mit)",
                   "authors": "Stephan Tulkens and Thomas van Dongen (Minish Lab)",
                   "model_safetensors_sha256": sha256(os.path.join(model_dir, "model.safetensors")),
                   "tokenizer_json_sha256": sha256(os.path.join(model_dir, "tokenizer.json")),
                   "url": "https://huggingface.co/" + fetch["repo"]},
        "library": {"model2vec": model2vec.__version__, "tokenizers": tokenizers.__version__,
                    "numpy": np.__version__},
        "n_vocab": int(n), "dim": int(dim), "bits": 4,
        "packed_bytes": int(packed.nbytes + scale16.nbytes), "packed_bytes_baselines": int(ref.bytes()),
        "unk_id": int(static.unk_token_id), "continuing_prefix": wp["continuing_subword_prefix"],
        "max_input_chars_per_word": int(wp["max_input_chars_per_word"]), "max_length": 512,
        "normalize": bool(static.normalize),
        "added_tokens": [a for a, _ in added], "added_ids": [int(i) for _, i in added],
        "theta": params["theta"], "temperature": params["temperature"],
        "beta": fit["beta"], "tau_ask3": fit["tau_ask3"]["tau"],
        "params_from": "<data>/eval/fitted/potion-cos-4bit/params.json; runs/s6-nano3/fit.json",
        "checks": {"texts": len(all_texts), "normaliser_per_character_differs": len(bad_norm),
                   "mirror_ids_differ": len(bad_ids)},
    }
    with open(os.path.join(args.out, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1, ensure_ascii=False)
        f.write("\n")
    print(f"[export] wrote {args.out}: {n} tokens x {dim}, {meta['packed_bytes']} packed bytes, "
          f"{len(t['starts'])} normaliser runs, {len(t['strings'])} strings", file=sys.stderr)

    if args.fixture:
        write_fixture(args, data, rb, params, static, mirror)


def write_fixture(args, data, rb, params, static, mirror):
    slates = read_jsonl(os.path.join(data, "build", "val", "v_tools.jsonl"))
    dump = read_jsonl(os.path.join(data, "eval", "val", SYSTEM, "v_tools.jsonl"))
    step = max(1, len(slates) // args.fixture_slates)
    order = list(range(0, len(slates), step))
    order += [i for i in range(len(slates)) if i % step]
    picked = []
    for i in order:
        r = slates[i]
        blob = request_of(r) + "\n" + "\n".join(r["opts"])
        if KEYLIKE.search(blob):
            continue
        picked.append(i)
        if len(picked) == args.fixture_slates:
            break
    sysm = rb.System(SYSTEM, params)
    probs = sysm.probs([slates[i] for i in picked])
    texts, index = [], {}

    def idx(s):
        if s not in index:
            index[s] = len(texts)
            texts.append(s)
        return index[s]

    out_slates = []
    worst = 0.0
    for i, p in zip(picked, probs):
        r = slates[i]
        worst = max(worst, float(np.abs(np.asarray(p) - np.asarray(dump[i]["probs"])).max()))
        out_slates.append({"row": i, "request": idx(request_of(r)), "tools": [idx(o) for o in r["opts"][:-1]],
                           "probs": [round(float(x), 9) for x in p]})
    # token ids: the first --fixture-ids texts of the slates and every edge case (the slates' other
    # texts are checked through their probabilities; every validation text is in tokens.jsonl)
    with_ids = list(range(min(args.fixture_ids, len(texts))))
    for e in EDGE:
        with_ids.append(idx(e))
    with_ids = sorted(set(with_ids))
    ids = static.tokenize([texts[k] for k in with_ids])
    fixture = {
        "about": "scripts/s6-route3-embed-export.py: potion-cos-4bit as scripts/route-baselines.py computes it, on "
                 f"{len(out_slates)} validation slates of build/val/v_tools.jsonl (rows listed) and hand-written edge cases; "
                 "sky_infer::embed must give the same token ids and the probabilities within 1e-5",
        "theta": params["theta"], "temperature": params["temperature"],
        "texts": texts, "ids": {str(k): list(map(int, x)) for k, x in zip(with_ids, ids)}, "slates": out_slates,
    }
    s = json.dumps(fixture, ensure_ascii=False, separators=(",", ":"))
    if KEYLIKE.search("\n".join(texts[: len(texts) - len(EDGE)])):
        raise SystemExit("a key-like string reached the fixture")
    with open(args.fixture, "w", encoding="utf-8") as f:
        f.write(s + "\n")
    print(f"[export] fixture {args.fixture}: {len(texts)} texts, {len(out_slates)} slates, {len(s.encode())} bytes; "
          f"largest gap to the potion-cos-4bit dump {worst:.2e}", file=sys.stderr)


if __name__ == "__main__":
    main()
