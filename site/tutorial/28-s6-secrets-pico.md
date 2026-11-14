# 28 s6-secrets-pico: is a secret exposed in this text?

*Trained in November 2026 and published on 14 November 2026, at full depth. It passed a gate fixed before training, against a rule scanner whose score was written down before the first training step. Its bench is synthetic, and on real files it fails: measured after publication, it flags 59.8% of clean real files at p ≥ 0.9. It is published as the result of its pre-registration, not as a scanner ([Limits](#limits)).*

The game models and s6-nano answer many kinds of question. s6-secrets-pico answers four, about one kind of text: a short file (an `.env`, a config, a few lines of code, a `curl` command, a log, a diff) that may expose a credential. It has 445,345 parameters in a 250,272-byte ternary file, the same engine as the games, and it runs in the browser, so a pasted text never leaves the page.

## The four questions

Each text gets four typed questions, each its own source in the evaluation ([`docs/SPEC-S6.md`](https://github.com/maxgfr/skycmd/blob/main/docs/SPEC-S6.md) §14):

| question | type | options |
|---|---|---|
| **secret**: does the text expose a usable secret that should be revoked? | noul | false / true |
| **service**: which service issued the key or token? | choice | 14 services, or `none` |
| **kind**: what kind of value is it? | choice | a live secret, a test key, a publishable key, a placeholder (or an example, a redacted value, a reference), or not a credential |
| **line**: which line exposes the secret? | choice | one option per line, plus `none` |

The kind question is what a scanner usually lacks: a Stripe-style live key (`sk_live_<random>`) must be revoked, the same prefix with `test` is a sandbox key, a publishable key is meant to be public, and a value like `<your-key-here>` is a placeholder.

## Data: generated, never real

`sky-data secret-jev` (seed 1234) writes `data/secrets` on your machine; nothing of it is committed ([pre-registration](https://github.com/maxgfr/skycmd/blob/main/docs/s6-secrets-prereg.md) §1):

- **Keys** follow each provider's public format with a random body, so none of them is a real key.
- **`secret_synth`**, the training data: 50,000 texts in 9 file types (env, export, python, js, yaml, json, curl, log, diff), split 95 / 5 into train and validation by text. 183,712 training questions.
- **`secret_bench`**, the test: texts in 4 file types the training never shows (docker-compose, toml, go, ini). The evaluation reads 1,996 texts, 7,984 questions.
- **Class shares**, the same in both: secrets 50 %, test keys 10 %, publishable keys 8 %, placeholders 17 %, values that are not credentials 15 %. Many negatives sit under secret-sounding names, which is where scanners fail.

The jev stays out of `data/decisions`: when it was designed, the general s6 had a 64-token state and would have read these texts cut off before the key ([`docs/SPEC-S6.md`](https://github.com/maxgfr/skycmd/blob/main/docs/SPEC-S6.md) §14).

## The bar, written before training

The rule scanner (`sky-data secret-baseline`, in Rust, in the style of [gitleaks](https://github.com/gitleaks/gitleaks), not gitleaks itself) was run on the bench before any training, and its score went into the pre-registration (§5):

- gitleaks-style prefixes for 20 key formats;
- connection strings that carry their password;
- an entropy rule (at least 3.5 bits per character, 20 characters, a digit) on names that sound secret;
- allowlists for documentation examples, test keys and publishable keys.

On `secret_bench` it scores **0.977** on the secret question (precision 0.966, recall 0.989: 999 true positives, 35 false positives, 11 misses, 951 true negatives) and **0.787** on the service question. Its 35 false positives are all random values that are not credentials, under names like `CACHE_KEY`; its 11 misses are all generic secrets with no distinctive prefix.

## The model and the run

- **Graph.** The s3-pico graph: one state block looped 4 times with an exit after each pass, d = 96, a 1,024-token vocabulary. A random key costs about one token per character, so the state window is 320 tokens instead of 64 (the states run to 93 tokens at the median and 312 at most) and the head 192 instead of 96 (the service question and its 15 options reach 179). RoPE has no position table, so the count is s3-pico's: 445,345.
- **Recipe.** From the pre-registered s6 ablation, applied by file (`runs/recipe/recipe.json`): AdamW at a learning-rate scale of 2.0, none of the eight s6 additions ([chapter 27](./27-s6-nano#the-recipe-chosen-by-an-ablation-not-by-hand)).
- **Run.** Seed 0, once, in a throttled mode, so the budgets count working minutes (`BENCH.md`):

| phase | budget | steps |
|---|---:|---:|
| MLM pretraining on `data/text-open`, `data/text` and the numeric curriculum, 256-token sequences | 8 min | 6,336 (final validation loss 2.2398) |
| decision training on `data/secrets` alone, QAT after a 10 % float warm-up, loop dropout 0.25 | 15 min | 7,555 |

```bash
scripts/train-secrets.sh data train eval
```

## The gate, and how to read it

The gate (§6): beat the scanner on both questions it answers, with the ternary file at full depth: secret **> 0.977** and service **> 0.787**.

| question | n | s6-secrets-pico | rule scanner | ECE |
|---|---:|---:|---:|---:|
| secret | 1,996 | **0.9990** | 0.977 | 0.0009 |
| service (15 options) | 1,996 | **0.9795** | 0.787 | 0.0154 |
| kind (5 options) | 1,996 | 0.9990 | not asked | 0.0010 |
| line (3 to 8 options) | 1,996 | 0.6859 | not asked | 0.3051 |

**The gate passes**, on the synthetic bench; on real files the model fails ([Limits](#limits)). Read against the scanner, text by text:

- **Where the scanner fails, the model does not.** It answers the secret question right on all 46 texts the scanner gets wrong: the 35 random values under secret-sounding names and the 11 generic secrets.
- **Where the model fails, the scanner does not.** The model is wrong on 2 texts, both an `x_api_key` with a 32-digit hex value in an ini file, which it calls not a credential; the scanner is right on both.
- **Service misses.** 41, of which 34 are in Go code, where the model names `database` for generic, AWS, Slack and npm keys.
- **Early exit is not needed for secret and kind.** At the thresholds the calibration stored, every question stops after the first pass: secret and kind keep 0.9990, service drops to 0.9775 and line to 0.6227. The gate was read at full depth, so the published file stores a threshold of 1.
- **The trained model is the ternary one.** The float master weights behind the quantization score 0.7124 on the test, against 0.9158 for the ternary file over the four questions: QAT ran over the last 90 % of training.

## Limits

- **The bench is synthetic.** The formats, the file templates and the class shares are our generator's, and every key is a public format with a random body. Beating the scanner here shows that the model reads formats and context (the variable name, a test or publishable key, a placeholder) in file types it never saw ([pre-registration](https://github.com/maxgfr/skycmd/blob/main/docs/s6-secrets-prereg.md) §6).
- **On real text it fails, so it is not a scanner.** After publication the file was run on 719 real source, configuration and documentation files in which a rule scanner finds nothing, cut into 22,433 chunks of at most 7 lines and 300 characters, the length of the training texts ([amendment 6](https://github.com/maxgfr/skycmd/blob/main/docs/s6-secrets-prereg.md); the measurement, its script and its limits: [secretgate pull request 19](https://github.com/maxgfr/secretgate/pull/19)). Every flag is a false positive:

  | threshold | chunks flagged | files flagged |
  |---|---:|---:|
  | p ≥ 0.9 | 10.6% | 59.8% |
  | p ≥ 0.999 | 5.9% | 49.4% |

  The flagged chunks are ordinary text (imports, prose, JSON), and the probabilities are not calibrated outside the generator: over the clean chunks the median is 0.00009 and the 90th percentile 0.93. The gate, on `secret_bench`, stays passed. The model stays published as the result of its pre-registration, not as a tool: do not use it to scan real files.
- **Recall on real secrets was not measured.** No real secret is in that corpus.
- **What a second version would be.** A different task: a rule finds the candidate, and the model judges it in its context. It would get its own pre-registration before any run; nothing is promised.
- **Do not rely on the line answer.** 0.6859, and confident when wrong (ECE 0.3051). On the 1,010 texts that expose a secret it names the right line 383 times; the 986 texts without one, where `none` is right, carry the score. Most misses point one line (327) or three lines (261) above the right one.
- **One run, one seed,** and no interval.

## Size and speed

Measured on the build machine (Apple M5) while a GPU training was running, so an idle machine may be faster:

- native (`sky-eval`, the 7,984 bench questions, full depth): a median of 789 and 786 µs per decision in two runs;
- WASM under Node (`parity-node.mjs`, the first 200 bench questions, SIMD128 build): 1,420 and 1,427 µs per decision, with a largest probability gap of 1.2e-6 to the native engine and the same answer on all 200;
- both builds later the same day, three alternating runs each: 1,580 µs with SIMD128 and 1,219 µs with relaxed SIMD (largest gap 6.0e-7, same answers), the build the site loads where the browser supports it.

## Try it

- **On [Live](/live/#s6-secrets-pico-is-a-secret-exposed-in-this-text):** take a preset from the test bench, or paste a text (an answer on a real file means little, see [Limits](#limits)) (taken by a fixed rule: the first text of each kind, the first text where the scanner is wrong on the secret question and the model right, and the first text where the model is wrong). The model asks itself the four questions, and each decision is timed in your browser. The presets' key bodies are drawn at random when the page loads, so no key-like value is stored in the site's source.
- **In your own page**, with the module of the [Install](/install/) page and the exact texts of the trained question (`site/.vitepress/theme/lib/s6-cases.ts`):

```js
import SkyCmd from "https://maxgfr.github.io/skycmd/lib/skycmd.js";

const model = await SkyCmd.load("https://maxgfr.github.io/skycmd/models/s6-secrets-pico.gguf");
const [pNo, pYes] = model.decideRaw(
  "noul",
  "Does the text expose a usable secret (an API key, token, password or private key) that should be revoked?",
  ["false: no, the statement does not hold", "true: yes, the statement holds"],
  fileText,  // the text to check, as a string
);
```

Ask it only its four questions, with these texts: it was trained on nothing else. Do not use it as a scanner on real files ([Limits](#limits)). The numbers of this chapter are in [`BENCH.md`](https://github.com/maxgfr/skycmd/blob/main/BENCH.md) (section "s6-secrets-pico") and on the [results page](/results/#s6-secrets-pico-passed-published-at-full-depth).
