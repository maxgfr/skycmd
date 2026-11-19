# s6-secrets: pre-registration (first training and publication gate)

*Translated into English on 19/11/2026. The text frozen on 26/10/2026 was written in French; that original, which remains the reference in case of doubt, is this file at commit `1f180a5`. The translation changes no rule, number or threshold.*

**Status: frozen on 26/10/2026, before any training of s6-secrets.** This file fixes the data, the run, the choice of checkpoint, the evaluation and the publication gate. The results will be read through this text.

**Changes to this file:**

- every amendment is committed, dated and justified in section 7;
- it precedes the run it concerns;
- it never touches a threshold after a bench score has been read.

## 1. The task and the data

Detect an API key or an exposed secret in a short text: a `.env` file, code, a config, a `curl` command, a log, a diff. The jev has four questions per text (`docs/SPEC-S6.md` §s6-secrets):

- **secret** (noul): does the text expose a usable secret, to be revoked?
- **service** (choice, 15 options): which service issued the key (`none` if there is none)?
- **kind** (choice, 5 options): secret, test key, public key, placeholder (or example, masked value, reference), or not a credential;
- **line** (choice): which line exposes the secret?

**Data.** `sky-data secret-jev` (seed 1234, default values) writes the files into `data/secrets`. Everything is synthetic and generated locally: each key follows the public format of its provider with a random body, so none is a real key, and nothing is committed.

- `secret_synth`: 50 000 texts, about 193 000 questions, split 95/5 into train and val by text. They are drawn from 9 templates: env, export, python, js, yaml, json, curl, log, diff.
- `secret_bench`: 2 000 texts, 7 984 questions, in test only. They come from 4 templates that training never sees: docker-compose, toml, go, ini.

The two sources have the same class shares:

- secrets: 50 %;
- test keys: 10 %;
- public keys: 8 %;
- placeholders: 17 %;
- values that are not credentials: 15 %.

## 2. The model

`configs/s6-secrets-pico.json` takes the architecture of s6-pico identically (same number of parameters), with two widened windows:

- **state: 320 tokens instead of 64.** With the s6 tokenizer (BPE 1 024 with [NUM]), a random key costs about one token per character. The states of the jev are 93 tokens at the median, 179 at the 95th percentile and 312 at the maximum.
- **head: 192 tokens instead of 96.** The "service" question and its 15 options run to 179 tokens.

These data do not go into `data/decisions`: the general s6 would truncate them before the key.

## 3. The run (`scripts/train-secrets.sh`)

- **MLM pre-processing**: 8 minutes on `data/text` plus 10 % of numeric curriculum, in sequences of 256 tokens.
- **Decision training**: 15 minutes on `data/secrets` alone, batch 64.
  - Ternary QAT after 10 % of floating-point warm-up.
  - Loop dropout 0,25.
  - Same optimiser options as s6-pico (`OPT`).
- Seed 0, once only. Only a technical failure justifies a rerun, and it is noted in the report.

## 4. Exported checkpoint

- `last` (or `last-ema` if `DECIDE` contains `--ema`, which is decided before the run), calibrated by `s2-calibrate` on the validation, then exported as ternary.
- No intermediate checkpoint is considered.

## 5. Evaluation

`s2-eval` of the ternary GGUF on `secret_bench`, last output, full depth. Each question has its own source (`secret_bench:secret`, `:service`, `:kind`, `:line`), so one accuracy is read per question.

**Reference: the rule-based scanner** (`sky-data secret-baseline`), measured on this same bench before any training. It combines:

- gitleaks-style prefixes for 20 formats;
- connection strings that carry their password;
- an entropy rule (at least 3,5 bits per character, 20 characters at least, one digit) on names that sound secret;
- a list of exceptions (EXAMPLE, test keys and public keys).

Its scores:

| question | accuracy | detail |
|---|---:|---|
| secret | **0,977** | precision 0,966, recall 0,989 (999 true positives, 35 false positives, 11 missed, 951 true negatives) |
| service | **0,787** | it names the service only for what it flags, and calls `generic` any key without a prefix |

The scanner's 35 false positives are all random values that are not credentials, under a name that sounds secret (`CACHE_KEY`, `KEY_FINGERPRINT`). The 11 missed are generic secrets with no distinctive sign.

## 6. Objective and publication gate

**Objective: beat the rule-based scanner on the two questions it handles.** With the ternary GGUF at full depth, it needs:

- a **secret accuracy > 0,977**,
- **and** a **service accuracy > 0,787**.

s6-secrets-pico is published (site, manifest, Hugging Face) if the objective is reached. Otherwise, it is not published, and the report says so.

Whatever the result, the report gives:

- the accuracy of the kind and line questions;
- the accuracy on validation (templates seen);
- the early exit;
- the size;
- the latency.

**What this bench does not say.** It is synthetic: the formats, the templates and the class shares come from our generator. Beating the scanner here shows that the model reads the formats and the context (variable name, public or test key, placeholder) in files of a type it has never seen. It says nothing of its recall on real repositories: no conclusion of that kind will be drawn.

## 7. Amendments

**Amendment 1 (28/10/2026, before any training of s6-secrets, no score read). Pre-training corpus.**

- **Change.** The MLM pre-processing of §3 reads `data/text-open` then `data/text` (`--text data/text-open,data/text` in `scripts/train-secrets.sh`), instead of `data/text` alone.
- **Why.** It is the same change as amendment 1 of `docs/s6-prereg.md`, whose options s6-secrets takes. The added text (StackExchange Q&A, often technical, scientific PDFs, maths) is also closer to the configuration and code files of the jev than the stories are.
- **What does not change.** The duration (8 minutes), the sequences of 256 tokens, the tokenizer, the decision data (`data/secrets` alone), the bench and the publication gate.

**Amendment 2 (29/10/2026, before any training of s6-secrets, no score read). Optimiser options.**

- **Change.** The final model s6-pico no longer exists (`docs/s6-prereg.md`, amendment 2). "Same optimiser options as s6-pico" (§3) means: the options adopted by the pico ablation, `OPT` without a rate, plus `LR_SCALE`. `scripts/train-secrets.sh` applies `LR_SCALE` to the default rate of each phase (3e-3 in pre-training, 1e-3 in decision) and refuses an `OPT` that would carry a rate.
- **Why.** A `--muon-lr` passed in `OPT` would have given the same rate to both phases, which the ablation does not do.
- **What does not change.** The architecture, the data, the budgets, the bench and the gate.

**Amendment 3 (09/11/2026, before any training of s6-secrets: only the pico recipe is running, none of its results has been read). The recipe goes through the file, in braked mode.** The adopted options come from `runs/recipe/recipe.json` (`scripts/s6-adopt.py env s6-secrets-pico`), with no choice by hand. The run is done in braked mode (`SKY_DUTY=0.5`, `SKY_COOL=45:15`), like amendment 5 of `docs/s6-prereg.md`: the budgets in minutes count only work time, and the report gives the number of steps actually done. Nothing else changes.

**Amendment 4 (12/11/2026, before any training of s6-secrets: the recipe is written (`runs/recipe/recipe.json`), no score of s6-secrets exists). Two modes.** At the owner's request (11/11/2026), the run uses `SKY_DUTY=0.85`, `SKY_COOL=50:10` while the Mac is not in use, and the braked mode of the previous amendment (`0.5`, `45:15`) the rest of the time, as in amendment 6 of `docs/s6-prereg.md`. Budgets in minutes still count work time only. A change of mode stops the training, which resumes from its last checkpoint (`RESUME=1`). The report gives the steps actually done and the modes used. Nothing else changes.

**Amendment 5 (12/11/2026, before any training of s6-secrets: no score of s6-secrets exists). The brake without a restart, and a memory cap.** The mode now changes while a run goes on: `sky-train` reads `SKY_BRAKE_FILE` again every few seconds, and `scripts/auto-brake.sh` writes it (the Mac counts as not in use after 10 minutes with no key or pointer input, unless a video or a call keeps the display awake, it runs on battery, or macOS reports thermal pressure). `scripts/train-all.sh` caps MLX's buffer cache at 2 GB, and the script stops a training that holds more than 13 GB of memory (see amendment 4 of `docs/s6-ear-prereg.md`). None of this changes what is computed.

**Amendment 6 (14/11/2026, after the gate was read and the model published). Measured on real text for the first time: the model is not usable as a scanner.**

- **What was measured.** After the publication, the published file was run on real files outside its generator, for a possible use in `secretgate` (a rule-based secrets firewall for coding agents). The measurement, its script and its limits are in that repository (`docs/model-comparison.md`, [pull request 19](https://github.com/maxgfr/secretgate/pull/19)). 719 real source, configuration and documentation files in which the rules of secretgate find nothing (its own sources and tests, a sample of its `node_modules`, and `crates/` and `docs/` of this repository) were cut into 22,433 chunks of at most 7 lines and 300 characters, the length of the training texts, and the secret question was asked on each chunk.
- **The result.** At p >= 0.9 the model flags 10.6% of these chunks and 59.8% of the files; at p >= 0.999, 5.9% and 49.4%. The flagged chunks are ordinary text (imports, prose, JSON); none of the 25 highest scores holds a credential-shaped value. The probabilities are not calibrated outside the generator: over the clean chunks the median is 0.00009 and the 90th percentile 0.93. Asking the kind and service questions as well does not change it, and neither do smaller or larger chunks. On `secret_bench` the same script finds the published score again (recall 0.9980, precision 1.0000, accuracy 0.9990), against 0.9772, 0.7183 and 0.7946 for the rules of secretgate, most of whose false positives there are the test and publishable keys it redacts on purpose.
- **What it means.** The gate (section 6) is on `secret_bench` and stays passed: nothing in it is reread or changed. But section 1 already said that the bench is synthetic and that beating a scanner on it says nothing of real repositories; this is now measured, and the answer is that the model cannot be used to scan real files. Every place that reports the model says so next to its bench scores. The model stays published as the result of this pre-registration, not as a tool.
- **What it does not test.** Recall on real secrets was not measured (no real secret is in that corpus), and the files come from two projects and one dependency tree.
- **Next.** A second version, if one is made, gets its own pre-registration before any run: a different task (a rule finds the candidate and the model judges it in its context), real-text negatives in training, and a public benchmark of real repositories rather than a generated one (`docs/research-s6-usefulness.md`, section 8).

