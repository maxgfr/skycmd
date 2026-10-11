"""Generate tokenizer fixtures with the reference HF `tokenizers` library.

Run once from the repository root:
    uv run --python 3.12 --with tokenizers python crates/sky-infer/tests/fixtures/make_fixtures.py

Outputs (next to this script):
  small_tokenizer.json        a tiny GPT-2 style byte-level BPE (skycmd layout: [PAD]=0 .. [MASK]=4)
  small_tokenizer_strmerges.json   the same with merges in the legacy "a b" string form
  tokenizer_cases.json        {"small": [...], "oscar": [...]} with {"text", "ids"} per case
"""
import json
import os
import sys

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
OSCAR = os.path.join(ROOT, "models-src", "oscar", "tokenizer", "tokenizer.json")

CASES = [
    "",
    "hello world",
    "Hello World!",
    "alt 12 vy -3 dx 40 gt 20 gb 35",
    "12345678 3.14159 -0.5 1e-9 -42 +7 100,000",
    "x=-3,y=+4;z=0",
    "I'm don't we'll they've she'd it's YOU'RE 'S 'tis",
    "naïve café 日本語 🙂 👩‍👩‍👧 ½ ٣٤ Ⅻ",
    "Le garçon a mangé à l'école, où est-ce ? Ça va très bien, merci !",
    "Élève ÉTÉ œuvre « guillemets » — tiret… déjà-vu",
    "a   b    c     d",
    "   leading spaces",
    "trailing spaces   ",
    " single leading",
    "line1\n\nline2\t\tx\r\nend\n",
    "tabs\tand nbsp em space　ideographic",
    '{"alt": 12, "vy": -3, "gaps": [20, 35], "ok": true, "name": "flappy"}',
    "choice question: Which action keeps the bird alive?",
    "[CLS] hello [SEP] world [MASK] x",
    "a[MASK]b  [MASK]   c",
    "decomposed é vs composed é, Å",
    "x" + " " * 30 + "y",
    "run of " + " " * 24 + "spaces and " + " " * 2 + "two",
    "|||IP_ADDRESS||| <|endoftext|> <|padding|> [UNK] [unused0]",
    "ﬁ ligature, ＦＵＬＬ width, ２０２４",
    "mixed123abc 4ever v2.0.1 #hashtag @user http://example.com/a?b=c&d=e",
    "''' \"quoted\" ''s '''re",
    "\n",
    "   ",
    "emoji🙂🙂 and symbols ©®™ ∑∫√ ≤≥",
]


def train_small():
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    tok.decoder = decoders.ByteLevel()
    corpus = [
        "alt 12 vy -3 dx 40 gt 20 gb 35",
        "the bird flies over the pipes and the gap is wide",
        "choice question: which action should the agent take now?",
        "score question: how dangerous is the current state of the game?",
        "noul question: the statement holds when the bird is above the gap",
        "Le garçon a mangé à l'école. Il fait très beau aujourd'hui.",
        '{"alt": 12, "vy": -3, "gaps": [20, 35]}',
        "flap wait up down left right rescue dodge",
    ] * 20 + [str(i) for i in range(-100, 200)]
    trainer = trainers.BpeTrainer(vocab_size=400, min_frequency=2,
                                  special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False)
    tok.train_from_iterator(corpus, trainer)
    return tok


def main():
    small = train_small()
    small_path = os.path.join(HERE, "small_tokenizer.json")
    small.save(small_path)
    with open(small_path, encoding="utf-8") as f:
        j = json.load(f)
    j["model"]["merges"] = [" ".join(m) if isinstance(m, list) else m for m in j["model"]["merges"]]
    with open(os.path.join(HERE, "small_tokenizer_strmerges.json"), "w", encoding="utf-8") as f:
        json.dump(j, f, ensure_ascii=False)

    out = {"small": [{"text": t, "ids": small.encode(t, add_special_tokens=False).ids} for t in CASES]}
    if os.path.exists(OSCAR):
        oscar = Tokenizer.from_file(OSCAR)
        out["oscar"] = [{"text": t, "ids": oscar.encode(t, add_special_tokens=False).ids} for t in CASES]
    else:
        print("models-src/oscar is absent: oscar cases skipped", file=sys.stderr)
    with open(os.path.join(HERE, "tokenizer_cases.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("wrote %d cases" % len(CASES))


if __name__ == "__main__":
    main()
