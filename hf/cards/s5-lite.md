---
license: apache-2.0
language:
  - en
base_model: cross-encoder/ettin-reranker-17m-v1
library_name: gguf
pipeline_tag: text-classification
tags:
  - gguf
  - decision-model
  - system-one
  - zero-shot-classification
  - calibration
  - rust
  - wasm
  - skycmd
datasets:
  - LocalLLaMA/typed-decisions
---

# s5-lite

**A 17M-parameter decision model whose options cannot see each other.** You give it a state, a
question and options (each with a one-line definition); one forward pass returns calibrated
probabilities over the options. It ships as a single 20.7 MB GGUF file (q8_0) and runs in the
browser through WebAssembly, in about 15–20 ms per decision.

- **Demo:** [huggingface.co/spaces/{{USER}}/s5-lite](https://huggingface.co/spaces/{{USER}}/s5-lite)
  (everything runs in your browser)
- **Code:** [github.com/maxgfr/skycmd](https://github.com/maxgfr/skycmd), 100 % Rust (training with
  mlx-rs, inference engine compiled to wasm32)
- **Pre-registered results, failures included:** [maxgfr.github.io/skycmd/results](https://maxgfr.github.io/skycmd/results/)

## What is different: the "iso" architecture

The encoder is Ettin-17M (ModernBERT), fine-tuned in full. Unlike the usual format, where every
option attends to every other option and to its own key, s5-lite isolates them:

```text
state     → attends to the state
question  → attends to the state and the question
option i  → attends to the state, the question and option i only (shared positions)
```

Each option is read through its **definition only**: the key you give it (`billing`, `yes`, `A`)
never reaches the model. Two properties follow by construction and are checked by exact tests
(to 1e-6, natively and in WASM), not measured:

- reordering the options cannot change any probability;
- renaming the keys, including swapping yes and no, cannot change any probability.

The state is encoded once and reused for every question asked about it.

## Results

All numbers come from the pre-registered evaluation of the project. This checkpoint (`long-b-s0`)
is a longer run (8,000 steps) of the iso configuration, added for the demo outside the
pre-registered bets and chosen by validation NLL; its numbers are reported, not claimed.

| set | accuracy [95% CI] | note |
|---|---|---|
| typed-decisions test (2,000 questions) | 0.696 [0.675, 0.716] | oscar-1-17m scores 0.682 on the same records |
| amazonpolarity (zero-shot) | 0.771 | majority class 0.523 |
| wikitoxic toxicaggregated (zero-shot) | 0.674 | majority class 0.509 |
| hwu64 (zero-shot, 8 options) | 0.763 | contamination through the teacher not excluded |
| sib200 en (zero-shot, 7 options) | 0.549 | majority class 0.250 |
| demo cases (English) | 21 / 30 | |

Robustness and coherence (test sets held out of training):

- label reassignment (the yes/no swap of arXiv 2609.26758): **0 % flips**, as guaranteed;
- paraphrased definitions: 26 % of answers flip (accuracy 0.696 → 0.602), so robustness to wording
  is learned, not guaranteed;
- distractors (JevOut-lite): 6.8 % flips under the worst distractor;
- 8-bit quantization: ECE moves by less than 0.003 against f32.

## Limitations

- **Imbalanced yes/no tasks.** On four test sets it does worse than always answering the majority
  class (biasframes intent and sex, wikitoxic insult and threat): it over-predicts the rare
  positive class. Do not use it as a guardrail without per-task thresholds.
- **Calibration** is underconfident on typed decisions; temperatures were fitted on a validation
  set dominated by other sources.
- **No controlled escalation yet:** with the available calibration data, Learn-then-Test certified
  no acceptance threshold at α ≤ 10 %.
- English only (there is no French version).
- It is a demo and research artifact, not a production classifier.

## Usage

The file runs in the skycmd engine (browser, Node, Python, CLI, server), not in llama.cpp.

```sh
cargo install --git https://github.com/maxgfr/skycmd sky-serve   # the `skycmd` binary
skycmd decide --model s5-lite.gguf --request req.json --pretty
```

```json
{
  "state": "Hi, I was charged twice for my March subscription, can you send the money back?",
  "questions": {
    "category": {
      "type": "choice",
      "instructions": "Which category does this customer message belong to?",
      "criteria": {
        "billing": "Invoices, charges, payments or refunds.",
        "bug": "Something in the app is broken or crashes.",
        "account": "Login, password or profile problems.",
        "unsure": "None of the listed categories fits the message."
      }
    }
  }
}
```

Python: `pip install "skycmd @ git+https://github.com/maxgfr/skycmd#subdirectory=crates/sky-py"`,
then `skycmd.Model("s5-lite.gguf").decide({...})`. In the browser, see the
[install guide](https://maxgfr.github.io/skycmd/install/).

## Training

- **Recipe:** AdamW (head learning rate × 20), warmup then linear decay, EMA of the weights,
  soft cross-entropy + Brier + RPS on ordered questions, symmetric KL between paraphrase and
  distractor pairs; 8,000 steps of batch 32, bf16, on an Apple M5; temperatures per question type
  and option count fitted on a validation split.
- **Data (English):** gold labels of permissively licensed datasets (typed-decisions train split,
  BoolQ, MNLI, ARC, CommonsenseQA, Banking77, Open-Jev CC0 traces, tasksource sets and others),
  soft labels from [laya-typed-decisions](https://huggingface.co/convaiinnovations/laya-typed-decisions)
  (Apache-2.0) run locally and kept only where they agree with the gold, and data generators
  written in code (rules, tone, buried facts, paraphrase and distractor pairs, coherence families).
  No data generated by GLM is used. Evaluation and demo states were removed from training
  (13-gram coverage and MinHash near-duplicates).
- Full list and licenses: [DATA_LICENSES.md](https://github.com/maxgfr/skycmd/blob/main/DATA_LICENSES.md).

## Files

- `s5-lite.gguf`: q8_0 encoder, f32 readout, calibrated temperatures, tokenizer
  (`general.architecture = skycmd-iso`).
