# 22 · s4: a generic decision model distilled from Qwen3.5

The s1 to s3 models know their games and the typed-decision sets they were trained on. On generic decisions they stay low: s3-pico scores 0.382 on `typed_decisions_bench`, against 0.6775 for oscar-1-17m ([chapter 11](./11-calibration-and-evaluation)). Two causes were documented: the sequences were truncated (96.8 % of them with the 1k vocabulary, [chapter 12](./12-size-ladder)), and the models learned language from scratch with no teacher. s4 attacks both. It is the s3 architecture at language scale, trained in one night on gold labels plus soft labels from Qwen3.5, and it runs in the browser on the [Live](/live/) page.

<!-- RESULTS -->

## The model: s3, wider, with a longer context

`configs/s4-micro.json` keeps the s3 recipe (numbers fused into the preceding token, a weight-shared looped state stack, ternary QAT, early exits) and changes the sizes:

| | s3-pico | s4-micro |
|---|---:|---:|
| vocabulary | 1,024 | 4,096 (`[NUM]` tokenizer) |
| hidden / FFN | 96 / 192 | 256 / 768 |
| heads | 1 | 4 |
| state stack | 1 block × 4 loops | 2 blocks × 3 loops |
| static + fusion blocks | 1 + 2 | 2 + 2 |
| exits (block passes) | 1, 2, 3, 4 | 2, 4, 6 |
| state / head tokens | 64 / 96 | 384 / 128 |
| parameters | 445,345 | 5,587,201 |

The static encoder starts as a copy of the first state blocks after pretraining, so `static_layers` cannot exceed `state_layers`: that is why s4 has two distinct state blocks looped three times rather than one block looped six times. The runtime does not change. s4 exports to the same `skycmd-s2` GGUF, and the browser module loads it like any s3 model. A random s4 model already ran in `sky-bench` and in the wasm build before any training started.

### Truncation

With 4,096 tokens and separate budgets for the state (384) and for the question with its options (128), truncation drops from 96.8 % of the sequences to 12.2 % of the states and 13.2 % of the heads, measured on 500 records per source (all gold sources and the s4 corpus). On `typed_decisions_bench` it is 7.4 % of the states and 5.2 % of the heads. Long states are still cut in HelpSteer2 (70 %), procedural typed decisions (59 %) and Open-Jev (44 %), where the state is a long document or a long JSON object.

## The teacher: one pass, the option letters only

`scripts/teacher-score.py` (mlx-lm) renders every record as a non-thinking chat prompt with the context, the question and the lettered options, runs one batched prefill and reads the next-token logits of the letters `A`, `B`, … only. The teacher never generates. Prompts are right-padded, which is exact for a causal model: attention and the Gated DeltaNet recurrence of Qwen3.5 only look back, and the output projection runs on the last real position of each row. The output is appended and flushed after every batch with the line index (`uid`), so a killed run resumes where it stopped.

Measured on the 2,000 records of `typed_decisions_bench` on the M5 (16 GB):

| teacher | accuracy | NLL | examples / s | tokens / s |
|---|---:|---:|---:|---:|
| Qwen3.5-0.8B (8-bit) | 0.412 | 1.744 | 18.5 | 6,223 |
| Qwen3.5-4B (4-bit) | 0.501 | 1.266 | 3.8 | 1,289 |

Neither reaches oscar-1-17m (0.6775), which was trained on that distribution. The 4B labelled the s4 corpus.

### The teacher has an answer bias, and calibration removes it

Per type, the 4B is at chance on the bench's nouls (0.505). It answers "true" to 591 of the 600, almost whatever the input: these nouls ask whether a human must take over a support conversation or review an agent trace, and the model escalates by default. A temperature cannot fix that. `sky-data s4-mix` therefore fits, per group of questions (source, type, number of options) with at least 150 gold training records, a temperature **and a bias per option** on the teacher's letter logits, `p = softmax(logits / T + b)`, by minimizing the cross-entropy against the gold labels. This is the contextual calibration of *Calibrate Before Use* (Zhao et al., 2021), fitted on labelled data. On the first scored moderation records it lifted the teacher's validation accuracy from 0.42 to 0.92. Groups without gold labels (the generic snippets) keep the global temperature.

## The corpus: randomized label sets

`sky-data s4-corpus` writes the records the teacher scores, in labelling order:

1. **Real cases** (`p3-cases`, 44,721): tool routing from Glaive and Hermes (the conversation's tools, distractor tools from other conversations and a `no_tool` option; gold = the first call), ticket triage from Bitext (random subsets of its categories or intents, with an `unsure` option), and moderation on a written rule from Civil Comments (a noul "the message violates this rule"; the annotator fraction is the soft gold label for the six rules that match a column, the teacher alone labels the five others).
2. **Generic snippets** (`p2-generic`, 30,000): 200 to 700 characters of web text and stories (English and French) under random label sets: topic, genre, audience, tone, time, section of a website, plus score and noul questions. There is no gold label: the teacher's distribution is the target.
3. **Typed decisions** (`p1-decisions`, 80,000): 4,000 records per source, with their choice questions label-randomized: options shuffled, distractors drawn from other questions, an abstention option added, and 12 % of the confident records with the gold option removed so that the abstention option becomes the answer.

The randomization is what forces the zero-shot behaviour: the model cannot learn that an option *name* is the answer, only that it fits the state. The abstention options (`none`, `unsure`, `other`) are the model's way to say "I don't know".

The typed decisions keep their gold labels: randomizing a label set does not invalidate a gold target, and on that distribution the gold is better than the teacher. They are not sent to the teacher, so the 4B spends its time on the records that need it.

Two sets are **held out entirely** for the zero-shot evaluation: Banking77 (excluded from training, 2,000 of its 8-way choices) and MASSIVE (2,000 English and 1,000 French test utterances, gold intent plus 7 random intents). The tokenizer was trained on text that includes Banking77 utterances (vocabulary only, no labels), and the Bitext triage data is close to the banking domain, so Banking77 is a new label set rather than a new domain.

## Targets: gold, teacher, or both

For a record with a gold label the target is `α · gold + (1 − α) · teacher` with α = 0.5; for a record without one it is the calibrated teacher distribution. Mixing hard and soft labels works better than either alone. The `target` field of the JSONL schema already accepts distributions, so the student's loss did not change.

## Deep improvement supervision

s3 supervised every exit against the same target. s4 adds `--dis β0` to `s2-decide`: exit `e` trains toward `normalize(target^β_e)`, with β rising linearly from β0 at the first exit to 1 at the last (`[0.5, 0.75, 1.0]` here). The early exits learn a softer copy of the target and the last exit the target itself. Each loop is thus supervised toward a target that improves on the previous one, the idea of *Deep Improvement Supervision* (arXiv 2511.16886) applied to the exits of a looped encoder. An early exit becomes confident only where even the softened target is confident, which is what the confidence-based early exit needs. A one-hot target is unchanged by the power, so the gold-only records behave as before. `model::exit_targets` is tested: entropies decrease from exit to exit, the last exit equals the target, and with every β at 1 the loss equals the plain deep supervision.

<!-- TRAINING -->

## Evaluating any model on the same records

`sky-train s2-eval` tokenizes the records with the tokenizer of the run it evaluates, so it cannot evaluate s3-pico (1k vocabulary) on s4's records. `sky-eval` (a binary of `sky-infer`) evaluates any GGUF with the tokenizer embedded in that GGUF, through the same engine the browser runs: per-source accuracy, NLL, ECE, mean block passes, early-exit rate and median latency. It reproduces the published 0.382 of s3-pico on `typed_decisions_bench`.

```bash
target/s4/release/sky-eval site/public/models/s3-pico.gguf \
  --records data/decisions/typed_decisions_bench.jsonl,data/s4/heldout --json eval.json
```

## Commands

```bash
export CARGO_TARGET_DIR=$PWD/target/s4
cargo build --release -p sky-data -p sky-train -p sky-infer --bins
uv venv --python 3.12 .venv-teacher && VIRTUAL_ENV=.venv-teacher uv pip install mlx mlx-lm
target/s4/release/sky-data fetch-decisions && target/s4/release/sky-data fetch-text --max-mb 150
target/s4/release/sky-data fetch-commands
scripts/night-s4.sh corpus tokenizer           # s4 corpus, 4,096-token [NUM] tokenizer
LABEL_UNTIL=02:00 scripts/night-s4.sh label    # teacher, resumable, stops at the given time
scripts/night-s4.sh train                      # mix + pretrain, decide, calibrate, export, evaluate
scripts/night-s4.sh teacher-eval               # the teacher on the same test sets
python3 scripts/s4-gate.py runs/s4-micro       # results table and gate
```

## Next

The teacher scored about 9 records per second. A larger labelled share (the typed decisions too, or a second teacher pass with the options reversed to cancel position bias) and a longer decision run are the obvious next steps. Porting the teacher's forward pass to mlx-rs would make the whole pipeline Rust.
