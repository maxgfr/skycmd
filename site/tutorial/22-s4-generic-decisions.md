# 22 · s4: a generic decision model distilled from Qwen3.5

The s1 to s3 models know their games and the typed-decision sets they were trained on. On generic decisions they stay low: s3-pico scores 0.382 on `typed_decisions_bench`, against 0.6775 for oscar-1-17m ([chapter 11](./11-calibration-and-evaluation)). Two causes were documented: the sequences were truncated (96.8 % of them with the 1k vocabulary, [chapter 12](./12-size-ladder)), and the models learned language from scratch with no teacher. s4 attacks both. It is the s3 architecture at language scale, trained in one night on gold labels plus soft labels from Qwen3.5, and it runs in the browser on the [Live](/live/) page.

::: warning Experimental
s4-micro fails its gate. The target was more than 0.55 on `typed_decisions_bench`; the ternary model scores **0.343**, below s3-pico (0.382). It is published on the [Live](/live/) page as an experiment, with the numbers below. s3-pico stays the model of the games.
:::

## Results

All rows are measured by `sky-eval` on the test split, through the same engine the browser runs, with early exit on (τ = 0.5). The zero-shot sets were never seen in training: Banking77 is 8-way (the gold intent plus 7 random ones), MASSIVE is the gold intent plus 7 random ones.

| model | params | file | bench | Banking77 (zero-shot) | MASSIVE en | MASSIVE fr | When2Call | all |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| s3-pico | 445k | 251 KB | **0.382** | 0.247 | 0.187 | 0.117 | 0.367 | 0.290 |
| s4-micro, f32 (QAT master weights) | 5.6M | 22 MB | 0.348 | 0.131 | 0.192 | 0.186 | 0.365 | 0.268 |
| **s4-micro, ternary** | 5.6M | 2.4 MB | 0.343 | **0.303** | **0.335** | **0.240** | **0.387** | **0.340** |
| teacher Qwen3.5-4B (4-bit) | 4B | ≈ 3 GB | 0.501 | 0.898 | 0.893 | 0.873 | — | — |
| oscar-1-17m (published) | 17M | | 0.6775 | — | — | — | — | — |
| chance | | | | 0.125 | 0.125 | 0.125 | | |

| model | ECE (all) | mean block passes | early-exit rate | median native µs / decision |
|---|---:|---:|---:|---:|
| s3-pico | 0.062 | 3.70 of 4 | 0.10 | 554 |
| s4-micro, ternary | 0.066 | 4.92 of 6 | 0.28 | 4,533 |

What the numbers say:

- **Zero-shot improves, the bench does not.** On labels it never saw, s4 is 1.2 to 2 times better than s3-pico (Banking77 0.303 against 0.247, MASSIVE English 0.335 against 0.187). That is still far from the teacher (0.89) and from a pretrained sentence encoder (below). On the bench, s4 is below s3-pico, which trained on that distribution for longer.
- **The loops barely help.** The three exits of the ternary model score within 0.01 of each other on every source (bench 0.344 / 0.345 / 0.352). The student is limited by what it knows of language, not by depth.
- **Calibration holds.** The ECE stays at 0.066 with the per-type temperatures, and the calibrated model abstains when its top probability is low (see the inbox demo).
- **The real cases follow the targets.** On the validation split of the cases, agreement with the target is 0.77 for tool routing (the calibrated teacher: 0.78) and 0.89 for ticket triage (the teacher: 0.75; the student learned the gold half of the target). Moderation reads 0.976, but 97.6 % of those validation records are "does not violate": the student has learned the base rate and nothing more. The guardrail demo is therefore a demonstration of the interface, not a working moderator.
- **Hand-written requests expose the gap.** On the Live presets, written by hand and unlike the Glaive and Hermes tool lists, the router answers `no_tool` for the calendar, weather and arithmetic requests (0.55, 0.77, 0.27); 58 % of the tool records have `no_tool` as their target (turns without a call), and the student learned that prior better than the tool descriptions. Triage gets the password request right (0.88, first exit) and the double charge only narrowly (billing 0.36 against unsure 0.34).
- **The browser matches the native engine.** On 40 requests (held-out and cases), the wasm module gives the native probabilities to 6.3e-7, with the same argmax everywhere, at a median of 6.9 ms per uncached decision under Node. A short cached input runs in 0.8 ms natively with the int8 kernel; long states (the bench) take about 18 ms. The ≈ 1 ms target holds only for short inputs.

### Two baselines that did better

To know what the student was missing, two models pretrained by others ran on the same records (Python, outside the Rust pipeline, only as comparisons):

| baseline | params | bench | Banking77 | MASSIVE en | MASSIVE fr | When2Call | latency |
|---|---:|---:|---:|---:|---:|---:|---:|
| MiniLM-L6 (`all-MiniLM-L6-v2`), cosine between state and option, no training | 22.7M | — | 0.855 | 0.773 | 0.493 | — | 6–9 ms (CPU) |
| BERT-Tiny (`google/bert_uncased_L-2_H-128_A-2`), fine-tuned 70 min on CPU on the s4 gold and an early mix | 4.4M | 0.396 | 0.794 | 0.570 | 0.302 | 0.468 | 1.6 ms (CPU, batched) |

BERT-Tiny has fewer parameters than s4 and reaches 0.794 zero-shot on Banking77 after the same fine-tuning: the difference is the language pretraining (BERT-Tiny read BooksCorpus and Wikipedia; s4 had 30 minutes of masked-language modelling on 150 MB). Ettin (arXiv 2507.11412) and BitNet Distillation (arXiv 2510.13998) both start from a pretrained model; s4 started from scratch, and that is the lesson of the night.

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

Per type, the 4B is at chance on the bench's nouls (0.505). It answers "true" to 591 of the 600, almost whatever the input: these nouls ask whether a human must take over a support conversation or review an agent trace, and the model escalates by default. A temperature cannot fix that. `sky-data s4-mix` therefore fits, per group of questions (source, type, number of options) with at least 150 gold training records, a temperature **and a bias per option** on the teacher's letter logits, `p = softmax(logits / T + b)`, by minimizing the cross-entropy against the gold labels. This is the contextual calibration of *Calibrate Before Use* (Zhao et al., 2021), fitted on labelled data. On moderation it lifted the teacher's validation accuracy from 0.55 to 0.91, mostly by learning that most comments do not violate the rule; on tool routing (0.78) and triage (0.75) the fitted biases are small and the accuracy does not change. Groups without gold labels (the generic snippets) keep the global temperature.

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

## The night run

All on the M5 (16 GB), in `scripts/night-s4.sh`:

| stage | time | what happened |
|---|---:|---|
| teacher labelling | ≈ 3 h | 44,721 case records and 30,000 generic snippets scored by the 4B; the typed decisions kept their gold labels |
| mix | CPU | 154,721 targets, per-group calibration (table of `teacher-report.json`) |
| pretrain (MLM) | 30 min | 6,236 steps, 56k tokens/s, validation loss 2.754 |
| decide | 150 min | 26,180 steps of 64 questions, ≈ 210 questions/s; QAT from 10 %, loop dropout 0.25, DIS β0 = 0.5, length-pooled batches (pool 16) |
| calibrate, export, evaluate | 18 min | per-type temperatures, f32 and ternary GGUF, `s2-eval` and `sky-eval` |

The ternary validation accuracy at the last exit rose from 0.444 at step 4,000 to 0.510 at the end (0.586 on the mixed targets), still slowly rising when the time ran out.

Two things went wrong and were fixed during the night:

- **Memory.** The student (14 GB) and the teacher (12 GB) together pushed 15 GB into swap. MLX keeps freed buffers for reuse; capping that cache (`SKY_MLX_CACHE_GB=2` for `sky-train`, `mx.set_cache_limit` for the teacher) and capping each gold source to 40k records brought both back into memory.
- **A non-finite batch.** At step 4,021 the loss became NaN and the update had already been applied, so the run stopped with poisoned weights. `apply_update` now evaluates the loss first and leaves the parameters untouched when it is not finite; the decision loop logs the batch and skips it, and stops after 20 skips in a row. The run resumed from its last checkpoint and skipped 8 batches out of 26,180. Seven of them had a static stream (question and options) of exactly 160 tokens: the cause is not found yet, the guard keeps it harmless.

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

In order of expected gain:

1. **Start from a pretrained encoder.** The baselines above show that language pretraining matters more than anything else here. The next s4 either pretrains for hours rather than 30 minutes, or initializes its embeddings and blocks by distilling a small pretrained encoder (Ettin-17M, MiniLM) before the decision run.
2. **More teacher labels.** The 4B scored about 4 bench records and 13 short records per second. Labelling the typed decisions too, plus a second pass with the options reversed to cancel position bias, gives the student more of the teacher's knowledge.
3. **Balanced case sets.** Civil Comments is 97 % non-violating and 58 % of the tool records expect `no_tool`; both need resampling, and the tool lists need hand-written, one-line descriptions like the ones people type.
4. **Find the 160-token NaN.**

Porting the teacher's forward pass to mlx-rs would make the whole pipeline Rust.
