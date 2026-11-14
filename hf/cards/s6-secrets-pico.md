---
license: apache-2.0
language:
  - en
library_name: gguf
pipeline_tag: text-classification
tags:
  - gguf
  - decision-model
  - system-one
  - ternary
  - secret-detection
  - rust
  - wasm
  - skycmd
---

# s6-secrets-pico

> **Not a scanner. Do not use it on real files.** Measured after publication on 719 real source,
> configuration and documentation files that hold no secret, the model flags 59.8% of the files at
> p ≥ 0.9 and 49.4% at p ≥ 0.999. Its bench scores below stand, but they come from our own
> generator and do not carry over to real text. Measurement:
> [maxgfr/secretgate pull request 19](https://github.com/maxgfr/secretgate/pull/19); details in
> [On real text](#on-real-text-measured-after-publication).

**A 250 KB ternary model trained to say whether a short generated text exposes a secret.** 445,345
parameters. It answers four typed questions in one forward pass: is a usable secret exposed, which
service issued it, what kind of value it is, and on which line. On a synthetic bench of file types it
never saw in training, it beats a gitleaks-style rule scanner on the two questions the scanner
answers, and it passed a publication gate fixed before training. It is published as the result of
that pre-registration, not as a tool.

- **Demo:** [huggingface.co/spaces/{{USER}}/s6-secrets-pico](https://huggingface.co/spaces/{{USER}}/s6-secrets-pico)
  (presets from the generator's bench; everything runs in your browser, nothing is sent anywhere)
- **Code:** [github.com/maxgfr/skycmd](https://github.com/maxgfr/skycmd) (100 % Rust)
- **Pre-registration:** [docs/s6-secrets-prereg.md](https://github.com/maxgfr/skycmd/blob/main/docs/s6-secrets-prereg.md)
- **All results, failures included:** [results page](https://maxgfr.github.io/skycmd/results/)

## Results

`secret_bench`: 1,996 synthetic texts in four file types the training never shows (docker-compose,
toml, go, ini), 7,984 questions. Ternary GGUF, last exit, full depth; the rule scanner on the same
texts, measured before any training:

| question | s6-secrets-pico | rule scanner |
|---|---:|---:|
| secret: does the text expose a usable secret? | **0.9990** | 0.977 (precision 0.966, recall 0.989) |
| service: which service issued it? (15 options) | **0.9795** | 0.787 |
| kind: secret, test key, publishable key, placeholder, not a credential | 0.9990 | not asked |
| line: which line exposes it? | 0.6859 (weak, see the limitations) | not asked |

- **The gate** (pre-registered): secret > 0.977 and service > 0.787, ternary, full depth. Passed.
- **Against the scanner.** Its 35 false positives are random values that are not credentials,
  under secret-sounding names (`CACHE_KEY`); its 11 misses are generic secrets. The model is right
  on the secret question for all 46 of those texts. It is wrong on 2 texts the scanner gets right:
  an `x_api_key` with a hex value in an ini file, which it calls not a credential.
- **Validation** (seen file types): 0.998, pooled over the four questions.
- **Calibration** (ECE, test): secret 0.0009, service 0.0154, kind 0.0010, line 0.3051. This
  holds on the generator's texts only (see below).

## On real text (measured after publication)

On 14/11, after the gate was read and the model published, the published file was run on real
files outside its generator (amendment 6 of the
[pre-registration](https://github.com/maxgfr/skycmd/blob/main/docs/s6-secrets-prereg.md); the
measurement, its script and its limits are in
[maxgfr/secretgate pull request 19](https://github.com/maxgfr/secretgate/pull/19)). 719 real
source, configuration and documentation files in which a rule scanner finds nothing were cut into
22,433 chunks of at most 7 lines and 300 characters, the length of the training texts, and the
secret question was asked on each chunk. Every flag is a false positive:

| threshold | chunks flagged | files flagged |
|---|---:|---:|
| p ≥ 0.9 | 10.6% | 59.8% |
| p ≥ 0.999 | 5.9% | 49.4% |

- The flagged chunks are ordinary text (imports, prose, JSON).
- The probabilities are not calibrated outside the generator: over the clean chunks the median
  P(secret) is 0.00009 and the 90th percentile 0.93.
- The gate, on `secret_bench`, stays passed: nothing in it is changed. But the model cannot be used
  to scan real files.
- Recall on real secrets was not measured.
- A second version, if one is made, would be a different task (a rule finds the candidate, the
  model judges it in its context), with its own pre-registration. Nothing is promised.

Latency, measured by us on an Apple M5 while a GPU training was running (an idle machine may be
faster): median 786 to 789 µs per decision natively over the 7,984 bench questions, 1,420 to
1,427 µs in WASM under Node on the first 200. WASM and native agree to 1.2e-6 on those 200
questions, with the same answer on all 200.

## Full depth

The file answers at full depth: all 4 block passes, every time. Its three exit thresholds are set
to 1 in the GGUF (`skycmd-s2.exit_threshold.{choice,score,noul}`), because the gate was read at
full depth and consumers read the thresholds from the file. The weights are those of the evaluated
file, byte for byte. The model has an exit after each pass. The first exit already scores 0.9990 on
secret and kind, but at the thresholds the calibration stored (0.5 / 0.9 / 0.5) every question stops
there, and service drops to 0.9775 and line to 0.6227. You can still pass another threshold at run
time (`setExitThreshold` in the browser).

## How to ask it

The model only knows the questions of its training data. Ask them with exactly these texts
(`sky-data secret-jev`, `crates/sky-data/src/secrets.rs`), the state being the text itself:

- `noul`: "Does the text expose a usable secret (an API key, token, password or private key) that
  should be revoked?", options `false: no, the statement does not hold` and
  `true: yes, the statement holds`;
- `choice`: "Which service issued the key or token in the text?", options `aws`, `github`,
  `gitlab`, `stripe`, `slack`, `openai`, `anthropic`, `google`, `huggingface`, `sendgrid`, `npm`,
  `private_key: a PEM private key`, `database: a connection string with its password`,
  `generic: a key or token of another service`, `none: no actual key or token`;
- `choice`: "What kind of value is assigned or sent in the text?", options
  `secret: a live, usable credential`, `test: a test or sandbox credential`,
  `public: a publishable key meant to be public`,
  `placeholder: a placeholder, an example, a redacted value or a reference`,
  `other: not a credential, even if it looks random`;
- `choice`: "Which line exposes the secret?", options `line 1` to `line N` and
  `none: no line exposes a secret`, for texts of two lines or more.

## Limitations

- **The bench is synthetic.** Its formats, file templates and class shares are those of our own
  generator, and every key is a provider's public format with a random body. Beating the scanner
  there shows that the model reads formats and context (the variable name, a test or publishable
  key, a placeholder) in file types it never saw. On real files it fails (see
  [On real text](#on-real-text-measured-after-publication)): do not use it as a secret scanner.
- **The line answer does not work yet.** 0.6859 overall, and confident when wrong (ECE 0.3051). On
  the 1,010 bench texts that expose a secret it names the right line 383 times; the texts without
  one, where `none` is right, carry the score.
- **Service misses concentrate on Go code:** 34 of its 41, where it names `database` for keys of
  other services.
- **Short texts.** The training texts have 2 to 7 lines and the state window holds 320 tokens; a
  longer text is cut.
- One run, one seed.

## Usage

Runs in the skycmd engine only (browser, Node, Python, CLI, server):

```sh
cargo install --git https://github.com/maxgfr/skycmd sky-serve
skycmd decide --model s6-secrets-pico.gguf --request req.json --pretty
```

Request format and bindings: [install guide](https://maxgfr.github.io/skycmd/install/).

## Training

From scratch in Rust (mlx-rs) on an Apple M5, in a throttled mode, so the budgets are working
minutes: masked-language pretraining in sequences of 256 tokens (8 minutes, 6,336 steps), then
decision training on the synthetic texts alone with ternary QAT after a 10 % float warm-up
(15 minutes, 7,555 steps, 183,712 training questions from 9 file types: env, export, python, js,
yaml, json, curl, log, diff). The architecture is the s3-pico graph (1 state block looped 4 times,
d 96, vocabulary 1,024) with a 320-token state window and a 192-token question window, trained with
AdamW, the recipe chosen by the pre-registered s6 ablation. Pretraining text: SimpleStories,
TinyStories, FineWeb and the non-web parts of Dolma 3 and TxT360. The decision data is generated
locally by `sky-data secret-jev`; no real key is used. Licenses:
[DATA_LICENSES.md](https://github.com/maxgfr/skycmd/blob/main/DATA_LICENSES.md).

## Files

- `s6-secrets-pico.gguf`: ternary weights, tokenizer, temperatures, and exit thresholds set to 1
  (full depth).
