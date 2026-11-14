# s6-secrets v2 (s6-secrets-verify-pico): pre-registration

**Status: frozen on 14/11/2026, before any download of a test file.** No CredData file or issue-report file has been downloaded. No candidate exists, no baseline has run, no line of the experiment's code is written. The results are read through this text. Three things are fixed later by rules written here, each before any test file is read: the generator (§2.4), the state window (§4.3) and the list of files of the clean-text corpus (§7.7).

**Why this experiment.** On 14/11/2026 the published s6-secrets-pico was measured on real text for the first time and flagged 59.8% of 719 clean files at p ≥ 0.9 (`docs/s6-secrets-prereg.md`, amendment 6). It passed its gate on a generated bench and cannot be used on real files. v2 is a different task, judged on real repositories.

**Changes to this file:**

- every amendment is committed, dated and justified in §13;
- an amendment comes before the run or the measurement it concerns;
- an amendment never moves a threshold after a test score has been read;
- measurements on training and validation repositories (generator recall, truncation, the value-only probe) are allowed before the freeze of §2.4 and §4. They are written down with their date and never touch a test file.

Background and sources: `docs/research-s6-usefulness.md` §1, §7, §8.1, §10, and the feasibility study, `docs/s6-secrets2-feasibility.md`, committed with this file.

## 1. The question

s6-secrets-pico (`docs/s6-secrets-prereg.md`) reads a whole short text and asks whether it exposes a secret. It was trained and tested on synthetic data only, and its line answer is weak (0.6859, `BENCH.md`).

v2 changes the job. A high-recall rule finds a candidate span, and the model judges that one candidate in its context. The rule gives the span, so the model never outputs one. The intended host is secretgate's rule engine (§2).

For each candidate, two typed questions:

- **real** (noul, gated): "Is this value a real credential that should be revoked?" T rows are true; F and X rows are false (§3.1).
- **kind** (choice, 3 options, reported only):
  - a real credential;
  - an example, test value or placeholder (CredData `X`);
  - not a credential (CredData `F`).

**What the gate asks.** On public data with real repository structure, does a 250 KB ternary verifier placed behind secretgate's rules:

- beat the rule scanners clearly;
- come within reach of the open 17M span model (cbx-secrets-detector-17M, 67 MB);
- beat a simple model of the same size;
- keep that advantage on real (not obfuscated) values;
- stay calibrated and under a few milliseconds per candidate;
- and stay quiet on real files that hold no secret, where v1 failed?

## 2. The candidate generator (`sg-cand-1`)

### 2.1 Definition
The generator is secretgate's own engine (https://github.com/maxgfr/secretgate, `src/engine/scanner.ts`), pinned at a commit, in a **candidate mode**:

- **Rules.** All rules of `rules/gitleaks.toml`: gitleaks commit `09242ce9c8a60d9b051fc2d166f9e849b88c7ac0`, 222 rules (`rules/UPSTREAM`). Plus secretgate's built-in rules (`scanner.ts`, `BUILTIN_RULES`).
- **What stays on.** The keyword prefilter, the global path allowlist (images, fonts, lock files, `node_modules`, vendored Go modules), the `post` and `postMatch` gates of the built-in rules (Luhn and IIN for card numbers, the URL username test), and the `dedupe` precedence.
- **What is turned off.** The filters the verifier is meant to replace:
  - the entropy gates;
  - `looksLikePlaceholder`;
  - the per-rule allowlists and the 1,456 stopwords of `generic-api-key`;
  - the global allowlist regexes and stopwords.

  The values they would have dropped become candidates. Each candidate records which of these filters it would have failed, as hand features (§4.2).
- **One new rule, `generic-keyword`**, modelled on CredSweeper's keyword template (`credsweeper/common/keyword_pattern.py`, v1.19.6):
  1. a variable name that contains one of `password, passwd, pwd, pass, secret, token, key, api, auth, credential, cred, nonce, salt`;
  2. an optional quote;
  3. a separator: `:`, `=`, `:=`, `=>`, `==`, `!=`;
  4. an optional quote;
  5. a value of 4 to 256 characters, up to the closing quote or the first whitespace, `;`, `,` or end of line.

  Case-insensitive. It is the high-recall rule that gitleaks' rules lack (feasibility §3.2).
- **Candidate record**: rule id, value, the variable or key name, the value's line and columns, the line, the previous line and the next line, path scope and extension (§4.1), and the engine's entropy.

The candidate mode and `generic-keyword` are code in the secretgate repository, pinned by commit: either a mode of the engine or a development script that calls the engine's own matching functions. The experiment calls that code through Node and never re-implements the matching, so the generator that is evaluated is the one a release would ship. By default the export masks each value (same length and character classes); the experiment uses the unmasked mode locally, and its files are never committed.

### 2.2 Recall check, on training and validation repositories only
Before the freeze, the generator runs on the CredData training and validation repositories (§3.1).

- **Recall** = share of T rows whose value span is overlapped by at least one candidate.
- **Volume** = candidates per 1,000 lines.

The target is a recall **≥ 0.95**. If the first version is under it, one revision of `generic-keyword` is allowed: keywords, separators, value length. It is dated in §13 with the misses it addressed (read from training and validation rows only). After that the generator is frozen whatever its recall, and the number is reported.

### 2.3 Why the recall is not a gate
Every system's line-level recall includes the generator's misses (§7.1), so a weak generator costs the gate directly. CredData's labels come from CredSweeper's candidates (`Category` column), so CredData favours a CredSweeper-like generator; the issue-report set (TruffleHog-derived regexes) is a second view.

### 2.4 Freeze
The secretgate commit, a hash of the rules file, and the generator's recall and volume on training and validation are written to `runs/s6-secrets2/generator.json` and committed before any test file is read.

## 3. Data and splits (by repository, never by line)

### 3.1 CredData
- **Source**: https://github.com/Samsung/CredData, pinned at the full SHA of HEAD on the day of download (`01e3d135a9…` on 12/11/2026), with the MD5 of `meta/`.
- **Download**: `download_data.py` in a Linux container, `--noise` left at its default. The transient clones hold the real values before obfuscation: they are made in a folder outside the repository and deleted when the script ends. Only the obfuscated `data/` is kept, outside the repository, and it is never committed.
- **Labels**: one per value from the meta rows (`GroundTruth`).
  - `T` → real = true, kind = real credential;
  - `X` → false, example/test/placeholder;
  - `F` → false, not a credential.
- **Matching a candidate to a row**:
  - on a labelled line, a candidate takes the label of the row whose `[ValueStart, ValueEnd)` its value overlaps;
  - a whole-line row (`ValueStart` empty or -1, F only) labels every candidate of its line false;
  - a candidate on a labelled line that overlaps no labelled value is dropped from training and evaluation, and counted.
- **Candidates on unlabelled lines of the training repositories are weak negatives**: real = false, kind = not a credential. CredData's labels cover the lines that one of seven tools flagged, so these are lines none of them flagged, and they are the ordinary assignments a deployed generator proposes all day; v1 never saw such text and failed on it. They are drawn at random (seed 1234) up to **25% of the CredData training items**, and tagged so that every training count separates them. Before training, 200 of them drawn at random are read by hand and the share that looks like a real credential is reported (label noise); they are used whatever that share. On validation and test repositories, candidates on unlabelled lines are never scored as negatives: they are counted (§7.3).
- **Training repositories**: the 116 repositories of cbx's `training_data_sources.csv` (HF repo `CobuildX-AI/cbx-secrets-detector-17M`, pinned commit), at their listed commits.
  - Each is kept only if the LICENSE copied by `collect_licenses()` at its pinned commit is MIT, Apache-2.0, BSD-2/3-Clause or ISC; the others are dropped and listed.
  - **Validation** is 16 of them, chosen by the lowest sha256 of the repository name with the seed 1234 appended (fixed now, before any download).
  - The rest are training.
- **Test repositories**: every other CredData repository at the pinned meta (about 221), whatever its licence. Testing does not redistribute them. cbx-17M never trained on any of them, so cbx can be rerun there fairly. This is **not** cbx's 168-repository test set, which is not public (feasibility §1.4). The card's numbers (0.805 and so on) are context only; every compared number is rerun through our scorer.
- **Decontamination**: every training or validation candidate whose normalised line also occurs in a test repository is removed. Normalised means trimmed, inner whitespace collapsed, and, for T rows, the value replaced by its length. This reads test inputs, never test labels. The number removed is reported.

### 3.2 The issue-report set (Zenodo 10.5281/zenodo.19622962, CC BY 4.0)
- **Download**: `Secret-Leak-Detection-Issue-Report.zip`, 134,502,037 bytes, version 19622962, with its sha256 recorded. It goes to a folder outside the repository; it is never committed and never sent to any remote service.
- **Splits**: the paper's (Table 7: train 40,593 / val 5,432 / test 8,123 candidates; 4,377 / 586 / 888 secrets). They are random per candidate, not by issue. The counts are checked on download, and any difference is written in §13 before training.
- **Decontamination**: every train or val row whose issue text also occurs in a test row is removed from train or val. The test is never touched.
- **Training values**: each real secret in train or val is replaced by a **word-preserving shape replacement**:
  - letter runs of 4 or more characters that are dictionary words become another word of the same length and case pattern;
  - every other character becomes a random one of its class;
  - known provider prefixes and separators are kept.
  The dictionary is fixed before the freeze (`/usr/share/dict/words`, lowercase, 4–12 letters). The replacement is not CredData's obfuscation on purpose: it keeps the "a word inside a password" shape that CredData destroys (§10).
- **Test values stay as published**. They are already public, and they are the point of this set: real values. Logs and reports show only hashes and lengths, never a value.

### 3.3 Synthetic candidates
- **Source**: a new `sky-data secret-cand` mode of the existing generator (`crates/sky-data/src/secrets.rs`, seed 1234), writing candidate records, not whole texts. Every key follows a public provider format with a random body; nothing is real.
- **Share**: at most **25%** of training items.
- **Positives**:
  - provider-format keys under their usual names;
  - **word-based passwords**: one or two dictionary words with digits and a symbol, under password-like names in assignments. CredData cannot provide these (§10).
- **Hard negatives**, following GitHub's list (13 Oct 2026 post) and the jev's own classes:
  - test and publishable keys;
  - placeholders and masked values (`xxxx`, `<your-key>`, `changeme`-like words, env and template references);
  - public keys and certificates;
  - checksums and integrity hashes (sha1, sha256, sha512 in lock-file and `integrity=` lines);
  - UUIDs;
  - git commit ids;
  - base64 image or font data;
  - documentation lines;
  - test fixtures under a `test` scope;
  - **near-miss pairs**: the same line once with a random secret and once with a placeholder or reference.
- **Never used**: the synthetic set never enters a test.

### 3.4 Mix and sizes
- **Decision training** draws from CredData training, issue-report train and synthetic with weights **0.45 / 0.35 / 0.20**.
- **Validation** is CredData validation, issue-report val and 2,000 synthetic candidates, kept apart by source.
- **Expected sizes (estimates)**:
  - CredData training: about 7,000 labelled values (cbx counts 7,244 training rows over the 116 repositories);
  - issue-report train: about 40,000;
  - synthetic: about 20,000.
  The actual counts are reported.

## 4. The model

### 4.1 Input (the state)
One state per candidate, in this order, because the state is truncated from the end (`crates/sky-infer/src/s2/prompt.rs:182-206`):

1. **Header**: `rule <rule id>`, then `path <scope> <extension>`. The scope is CredData's `word_in_path.json` mapping (`test`, `docs`, `src`, `security`, `build`, `other`), computed the same way at deployment from the real path, since CredData's paths keep only the scope and the extension.
2. **Key**: the variable or key name, at most 40 characters.
3. **Value**: the whole value if it has 64 characters or fewer, else its first 32 and last 32 characters with a marker between them.
4. **Features** (§4.2), as `name value` pairs (one fused value per pair, SPEC-S3 §1).
5. **Line**: the candidate's line with the value replaced by a short marker, at most 120 characters centred on the value.
6. **Context**: the previous and next lines, each cut to 100 characters.

The questions (§1) go in the head stream, word for word in English.

### 4.2 Hand features
These are computed by one shared definition with a fixture file that the Rust trainer and the TypeScript host must both pass (as `num_extract.json` does for numbers):

- value length;
- Shannon entropy of the value (bits per character, as in `src/engine/entropy.ts`);
- characters per token of the value under the model's own BPE (token efficiency, after Betterleaks);
- counts of digits, upper-case letters, lower-case letters and other characters;
- length of the longest dictionary-word run;
- 0/1 flags for each turned-off filter the value would have failed (§2.1): entropy gate, placeholder, stopword, allowlist.

### 4.3 Size and windows
- **s6-secrets-pico's graph as it is**: s3-pico, 1 state block × 4 loops, exits after each loop, d 96, vocabulary 1,024 (`data/tok/tok-s2-1024.json`), `fuse_numbers`, no Parallax (as adopted). That is 445,345 parameters and a ternary GGUF of about 250 KB.
- **State window rule**, decided before training:
  - `sky-train s2-truncation` runs on the training and validation candidate records;
  - if more than 5% of states are truncated at 320 tokens, the window becomes 384; otherwise it stays at 320;
  - the head window stays at 192.
  - RoPE has no position table, so the parameter count does not change.
  - Estimated need (feasibility §5): about 215 tokens before the neighbouring lines, so 320 is expected to hold.
- **Why not nano**:
  - s6-nano (1,486,225 parameters, 715 KB) takes a median 1,488 µs natively and 14.6–15.2 ms in WASM per decision on long states (`BENCH.md`, s6-nano). That is too slow for a hook judging tens of candidates per payload.
  - The job is narrower than general decisions.
  - Nano is the next step only if gate G2 fails while G3 and G4 pass (§11).

## 5. The run (`scripts/train-secrets2.sh`, to be written)

- **MLM pretraining, 12 minutes**:
  - starts from the parameters of `runs/s6-secrets-pico/pretrain/init` (start-from-parameters option of `s2-pretrain`, `init_params` in `crates/sky-train/src/s2/train.rs`);
  - text: `data/text-open`, `data/text`, and the obfuscated files of the CredData *training* repositories, 30% of sequences;
  - plus 10% numeric curriculum;
  - sequences of 256 tokens, batch 64.
- **Decision training, 30 minutes**:
  - on the §3.4 mix, batch 64;
  - QAT ternary after a 10% float warm-up, loop dropout 0.25;
  - the recipe of `runs/recipe/recipe.json` (`scripts/s6-adopt.py env s6-secrets-pico`), with no hand choice;
  - `--no-compile` (amendment 11 of `docs/s6-gen-prereg.md`).
  - About 900,000 questions at s6-secrets-pico's rate, so about 10 passes over the data.
- **Seed and brake**: seed 0, one run. Only a technical failure justifies a restart, and the report says so. The brake modes of `docs/s6-gen-prereg.md` amendments 5, 6 and 10 apply. Budgets count working minutes, and the report gives the steps done.
- **Calibration**: `s2-calibrate` on the CredData validation candidates, then ternary export.
- **Operating threshold τ\***: the threshold on P(real) that maximises F1 on CredData validation candidates. Presets for F0.5 and F2 are chosen the same way. All three are stored in the GGUF metadata before any test file is read.
- **Checkpoint**: `last` (or `last-ema` if the recipe adopts EMA). No intermediate checkpoint is considered.

## 6. Baselines (all rerun on the same test files, through the same scorer)

| baseline | version | how it runs | unit |
|---|---|---|---|
| generator alone | `sg-cand-1` | every candidate accepted | line, candidate |
| secretgate default engine | the pinned commit, filters on | `scan()` per file, with the path | line, span |
| Gitleaks | 8.30.1 (installed) | `gitleaks dir --report-format json`, default config | line, span |
| Betterleaks | 1.10.1 | `betterleaks dir`, default config | line, span |
| detect-secrets | 1.5.0 | `scan --all-files`, default plugins | line |
| CredSweeper | 1.19.6 | ML at its default threshold (medium, 0.622), and ML off (`--ml_threshold 0`) | line, span |
| cbx-secrets-detector-17M | pinned HF commit | `cbx_secrets.py`, CPU, default config; a line is flagged if a predicted span overlaps it | line, span |
| size-matched LR | fixed now | hashed character 1–5-grams of the same state text (2^18 weights, int8 after training, 256 KiB) plus the §4.2 features; logistic regression (scikit-learn 1.9.1, `SGDClassifier`, log loss, L2, the regularisation strength chosen among 1e-7, 1e-6, 1e-5, 1e-4 on validation F1) trained on the same training candidates, weak negatives included; threshold maximising F1 on the same validation | line, candidate |
| potion-base-2M + LR | model2vec 0.10.0 | the state text embedded, LR on top (7.6 MB) | line, candidate (reported only) |

- **CredSweeper's ML was trained on a random 85% of all CredData lines** (`experiment/train.py`). Its CredData numbers are in-sample, reported with that mention, and enter no gate. On the issue-report set it is out of sample.
- **The rule scanners give no score**, so they enter at their default operating point.

## 7. Metrics

### 7.1 CredData test, line level (the gated unit)
- **Universe**: labelled lines of the test repositories (one per `FileID`, `LineStart`).
- **A line is positive** if it has a T row.
- **A system flags a line** if it accepts at least one candidate on it (for the model: P(real) ≥ τ\*), or, for a span tool, if a reported span overlaps the line.
- **Metrics**: precision, recall and F1.
- **False-alarm rate**: the share of F/X lines flagged, as the cbx card defines it.
- Generator misses count as false negatives.

### 7.2 CredData test, candidate level
- **A candidate is positive** if its value overlaps a T value span.
- **Metrics**:
  - precision, recall and F1 at τ\*;
  - PR-AUC;
  - recall at a 1% false-alarm rate;
  - **ECE**, 15 bins, over all test candidates on labelled lines, for the real question.

### 7.3 Unlabelled-line alarms
Accepted candidates on lines with no meta row, per 1,000 test files, for every system. Reported, not scored: CredData never judged those lines.

### 7.4 Issue-report test (real values)
**Candidate level**: the dataset's own `candidate_string` in its text. The model judges it with the state built from its line and neighbouring lines (no path, scope `other`). A rule scanner flags a candidate if one of its findings overlaps it in the text. Precision, recall, F1, ECE.

**Also reported**:
- **end to end**: `sg-cand-1` and the model on the issue texts, with the dataset's positives as the reference;
- **obfuscation sensitivity**: the F1 difference between the test values as published and the same values passed through CredData's `get_obfuscated_value`.

### 7.5 Latency and size
- **Native**: `sky-eval` over the CredData test candidates at full depth, median µs per candidate. Also at the first exit.
- **WASM**: `crates/sky-web/bench/parity-node.mjs` on 200 test candidates, on a copy whose thresholds are set to 1 (amendment 13 of `docs/s6-gen-prereg.md`). Reported: max |p_wasm − p_native|, argmax agreement, median µs.
- **Size**: GGUF bytes.
- **CPU time** of each baseline over the test files.

### 7.6 Uncertainty
A paired bootstrap over test repositories (1,000 resamples, 95%) for the F1 differences of G1–G3 and for ECE. Reported, not gates.

### 7.7 Clean real text (where v1 failed)
- **Corpus**: the census corpus of secretgate's comparison (`docs/model-comparison.md` in that repository, section 5: real source, configuration and documentation files of secretgate, of its dependency tree and of this repository; 8,821 files on 14/11/2026, in which its default rules raise 65 findings in 17 files, none a real credential on inspection). None of these files is in CredData or in the issue-report set, and none is used for training. The list of files and their hashes are written to `runs/s6-secrets2/clean-corpus.json` when the generator is frozen (§2.4), before training.
- **Measure**: the share of files with at least one accepted candidate (generator `sg-cand-1`, then P(real) ≥ τ\*), and the same share for secretgate's default engine and for the generator alone. Every accepted candidate of the model is then read by hand, and the report says how many look like a real credential; the gate uses the raw share.

## 8. Gate (ternary GGUF, full depth, τ\* from validation)

The model is published as `s6-secrets-verify-pico` only if **all** of these hold:

| | condition | at today's published numbers this means about |
|---|---|---|
| **G1** | CredData line F1 ≥ F1 of the best rule scanner (max of Gitleaks 8.30.1, Betterleaks 1.10.1, secretgate default) **+ 0.15** | ≥ 0.59 if the best rule scanner stays near Gitleaks' 0.442 |
| **G2** | CredData line F1 ≥ F1 of cbx-secrets-detector-17M **− 0.10** | ≥ 0.70 if cbx stays near 0.805 |
| **G3** | CredData line F1 **>** F1 of the size-matched LR | — |
| **G4** | issue-report test candidate F1 ≥ Gitleaks 8.30.1 F1 on the same candidates **+ 0.08** | ≥ 0.76 if Gitleaks stays near 0.678 |
| **G5** | ECE ≤ **0.05** on CredData test candidates | — |
| **G6** | native median latency ≤ **3 ms** per candidate, full depth (Apple M5) | — |
| **G7** | clean real text (§7.7): files with at least one accepted candidate ≤ **1.0%** | the default rules flag 0.19% of these files (17 of 8,821); v1 flagged 59.8% of a 719-file subset |

**Announced goal, not a gate**: G2 at −0.05 (within 0.05 of cbx-17M at about 1/270 of its file size).

**Why these numbers.**

- **G1, +0.15.** On the cbx card, Gitleaks is at 0.442 and cbx-17M at 0.805. +0.15 is about 40% of that gap. A smaller gain would not justify adding a model to a hook whose rules already run. Gitleaks' precision is 0.90 at recall 0.29, so the gain has to come from recall without giving back precision. The best of three scanners is taken so that the bar is the strongest rule set available, including secretgate's own.
- **G2, −0.10.** The model is about 270× smaller than cbx-17M (about 250 KB against 67,220,156 bytes) and needs no torch. At −0.10 it still sits in the upper part of the rules-to-cbx range, and a hook author choosing between the two could reasonably take it. A larger gap would make the cbx model the right recommendation wherever 67 MB fits. The number is a judgement, not derived from a source.
- **G3, strict.** If a hashed n-gram LR of the same size and the same features matches the transformer, the transformer has no reason to exist and the LR should ship instead.
- **G4, +0.08.** This is the guard against learning CredData's obfuscation: the issue-report values are real.
  - cbx-17M is +0.158 above Gitleaks there.
  - The split is random per candidate, which favours trained models (they see sibling candidates of the same issues) and does nothing for rules, so the margin is set above a token +0.05.
  - The gate asks for about half of cbx's margin.
- **G5, 0.05.** This is a binary question in distribution, with a temperature fitted on validation. s6-secrets-pico reached 0.0009 on its synthetic bench. The general s6 guard (0.08) was set for a pool of zero-shot multi-class sources. 0.05 is a guard, not a target, and the report gives the exact value.
- **G7, 1.0%.** This is the failure that ended v1, so it is a gate and not a reported number. The generator's candidate mode turns the rules' filters off and adds a keyword rule, so it proposes many values in clean files; the model has to reject nearly all of them. 1.0% is about five times the rate of the default rules on the same files, the price accepted for the recall asked by G1, and sixty times under v1. The number is a judgement.
- **G6, 3 ms.** secretgate's whole engine runs in about 50 ms per scan (secretgate `README.md:70`). Ten candidates at 3 ms add 30 ms natively.
  - s6-secrets-pico takes a median 789 µs natively with states of 93 tokens at the median (`BENCH.md`).
  - States here are 2–3× longer, so 1.5–2.5 ms is expected.
  - The WASM figure is reported, not gated.

## 9. Reported, not gated
- Generator recall and volume (train, val, test).
- Kind accuracy.
- Accuracy and ECE per exit, and the early-exit rate at the stored thresholds.
- Every metric of §7 for every baseline, including span-level F1 in the cbx sense (overlap with a T value, label-agnostic) and CredSweeper's in-sample CredData number.
- Results by CredData category (Password, Token, Key, Secret, API, Auth, other), and by scope (`test` vs `src`).
- The value-only probe: an LR on the value features alone, trained on CredData training and scored on CredData validation. A high score means the value shape alone separates T from F/X, i.e. the obfuscation is readable.
- Obfuscation sensitivity (§7.4).
- The end-to-end issue-report run.
- Decontamination counts and the licence-dropped repositories.
- Truncation share, steps done, minutes and brake modes.
- The latency of the whole secretgate scan with the verifier, on secretgate's FP corpus (`tests/fixtures/fp-corpus`): candidates per file and added ms.

## 10. What this benchmark does not say
- CredData's true values are random characters of the right classes, lengths and prefixes, while its false values are raw (`obfuscate_creds.py`). A good CredData score can partly come from reading that difference. G4 and the sensitivity measure test for it; no claim about word-based passwords is made from CredData.
- The issue-report test shares issues with its training split; its numbers are optimistic for every trained system, ours included.
- Recall is that of the system (generator plus model), on lines CredData chose to label. Nothing is claimed for secrets in lines that no rule flags.
- The test is not cbx's test; our numbers and the card's are not interchangeable.

## 11. What a failure would mean
- **G1 fails**: on real repositories the verifier adds too little to the rules it filters. s6-secrets is not continued in this form.
- **G2 fails, the others pass**: the design works but 445K parameters with this input are short of the 17M model. A next attempt (s6-nano graph or a 384/512 window, pre-registered anew) is justified. This model is not published.
- **G3 fails**: a linear model on n-grams and features does as well at the same size. The report says so, and whether to ship that LR in secretgate becomes a separate decision, outside this file.
- **G4 fails while G1–G2 pass**: the model learned CredData's obfuscation, or does not transfer to real values. More real-value training data is needed before any further attempt.
- **G7 fails**: the verifier is too noisy on ordinary files to sit in a firewall, whatever its benchmark scores. Not published; real clean text has to weigh more in training before any further attempt.
- **G5 or G6 fails alone**: not published. The report gives the numbers. Calibration or speed work comes before a new pre-registration; this run is not repeated.
- **Whatever the result**, the report gives every number of §7 and §9, and a failure is written into `BENCH.md` as plainly as a pass.

## 12. Forbidden
- Reading any test file (CredData test repositories, the issue-report test) before §2.4 and §5 are frozen, except to decontaminate (§3.1) without labels.
- Revising the generator after its single allowed revision, or after any test read.
- Choosing τ\*, a preset, a checkpoint or a window on test data.
- Running again to "see".
- Training on a copyleft repository or on any test repository.
- Writing any real value into a log, a report, a commit or a model card.

## 13. Amendments

**Amendment 1 (14/11/2026, after the downloads and before anything else: no candidate exists, no scanner or model has run on any file, no content of a test file has been opened; the files were hashed and counted, and the split was built from `meta/` and from the cbx list). What the downloads showed, and the sentences of §3 and §4.1 that were ambiguous once applied.**

- **What was fetched.** CredData at `01e3d135a90f46c9a7965683b24a05d12e136b33` (337 repositories, 11,565 obfuscated files, 67,896 labelled rows; its meta and data MD5, computed as CredData defines them, equal those of its `.ci/benchmark.txt`), the cbx list at Hugging Face commit `89362a0938f9459e5706e703e2a91c5cf2ffe40e` (116 repositories), and the issue-report archive (134,502,037 bytes, MD5 as on the Zenodo record). Hashes, counts and the split are in `MANIFEST.json` and `splits.json` of the data folder, which stays outside every repository. The clones lived in a container volume that was removed when the script ended.
- **"The repository name" (§3.1, validation).** It is the `repository` field of the cbx list, a full URL, byte for byte, with `1234` appended and no separator; the 16 lowest sha256 digests are validation. This is the reading that was applied when the split was built, before any content was read; the other readings (owner and name, name only, CredData's RepoID) give other sets and are not used.
- **Licences (§3.1).** A repository is kept if `collect_licenses()` copied at least one file and every copied file is MIT, Apache-2.0, BSD-2-Clause, BSD-3-Clause or ISC. All 116 are kept (MIT 72, Apache-2.0 34, BSD-3-Clause 7, BSD-2-Clause 3). A cbx repository that a later check would drop goes to no split: it is never moved to the test.
- **The split.** Training 100 repositories (24,231 labelled rows: 3,491 T, 19,619 F, 1,121 X), validation 16 (2,653: 659, 1,841, 153), test 221 (41,012: 11,603, 26,601, 2,808). The "about 7,000 labelled values" of §3.4 was the cbx card's count, not the meta's; the estimate is replaced by these counts and nothing else changes: the mix weights stay 0.45 / 0.35 / 0.20, the weak negatives stay capped at 25% of the labelled CredData training candidates, the budgets stay.
- **One repository holds 5,165 of the 11,603 T rows of the test** (from the row counts of `meta/`, read before any run). A pooled F1 would mostly be the score on that repository, and this project has just seen a pooled number decided by one source (`docs/s6-gen-prereg.md`, amendment 12). So **G1, G2 and G3 must each hold twice**: on all 221 test repositories, and on the 220 left when that repository is removed. The repository is named in `splits.json` now. This can only make the gate harder. The report also gives the macro average of F1 over the test repositories that have at least 20 T rows, as a reported number.
- **Eight T values of four test repositories are identical before and after CredData's obfuscation**, so they are original values. They stay in the test, since the files are CredData's reference files, and the rule of §12 covers them: no value in a log, a report, a commit or a card. The download log that printed them stays in the data folder.
- **The issue-report set.** The three split files have exactly the counts of the paper's Table 7 (40,593 / 5,432 / 8,123 candidates; 4,377 / 586 / 888 secrets) and are the ones used; `main_data.csv` has 54,203 rows where the paper says 54,148, and is not used. The archive also holds `test_wild.csv` (1,488 rows, 29 secrets), which §3.2 did not know: it is scored and reported like the test of §7.4 if the archive's documentation says what it is, and it enters no gate.
- **The path scope of §4.1.** CredData does not map a path to one of six words: `get_file_scope()` keeps every word of the 57 of `word_in_path.json` that the path contains, nested, and writes `_` when there is none. The header of the state carries those words in the function's order, at most four, or `_`; at deployment the same function, ported with a fixture that both sides must pass, computes them from the real path.
- **What does not change.** The question, the generator and its single allowed revision, the model, the budgets, the baselines, G4 to G7, the thresholds of every gate, and everything forbidden in §12.

## 14. Choices made at the freeze
The feasibility study listed alternatives; these are the ones taken, with the alternative that was left.

1. **State window**: 320, with the 5% rule of §4.3 (left: 384 fixed).
2. **Size**: pico, 445K (left: the s6-nano graph, too slow in WASM for a hook).
3. **Training repositories**: the 116 of cbx, licence-checked, so that cbx-17M can be rerun fairly on the test repositories (left: every permissive non-test repository).
4. **Unlabelled candidates of training repositories**: weak negatives, capped at 25% of the CredData training items, with a hand-read sample (§3.1). The draft excluded them; they are in because v1 failed on text it had never seen. The size-matched baseline gets the same items.
5. **Teacher**: none (left: soft labels from cbx-17M, which would be another experiment).
6. **Symmetric obfuscation**: no; the value-only probe (§9) and G4 measure the artefact instead.
7. **Issue-report test values**: kept as published, on the local disk only, never in a log, a commit or a remote service.
8. **Size-matched baseline**: scikit-learn in a pinned `uv` environment (left: Rust in `sky-data`).
9. **τ\***: F1-optimal on CredData validation, with F0.5 and F2 presets stored next to it.
10. **X labels**: negative for the real question.
11. **Decision budget**: 30 working minutes.
12. **Kind question**: asked, reported, not gated.
13. **Mix weights**: 0.45 / 0.35 / 0.20 (CredData with its weak negatives / issue reports / synthetic).

**Amendment 2 (14/11/2026, after the single measurement of the generator on the training and validation repositories (§2.2) and before its revision; no test file has been opened, no model or baseline has run). The first measurement, the one revision, a class of rows that leaves the gated universe, and the readings of §2 and §3.1 that the implementation had to choose.**

- **The measurement.** `sg-cand-1` at secretgate commit `c5ba568`, on the 100 training and 16 validation repositories: recall 0.5773 (2,396 of 4,150 T rows; training 0.5766, validation 0.5812), against 0.2790 for the default engine by the same definition; 13.3 candidates per 1,000 lines. The target of §2.2 was 0.95. By category: Password 0.9256, Key 0.9682, Secret 0.9220, Token 0.7958, Auth 0.5194, Bearer Authorization 0.1228, Basic Authorization 0.1429, UUID 0.0955. All numbers are in `gen/recall.json` of the data folder.
- **What the misses are (1,754 rows, described by structure, no value read into any report).** 1,033 are UUID-shaped values under names that hold no credential keyword (`id`, `_postman_id`, `collectionId`, `folder`, or no name at all). 219 are authorization headers: a scheme word (Bearer, Basic) stands between the separator and the value, or the header name is on the previous line. 55 follow an escaped quote (a string inside a string), 53 a bracket (a list), 16 a subscript. The rest are forms a keyword rule does not express (bucket names, domains, call arguments, URL parameters, command-line passwords).
- **Rows of category exactly `UUID` leave the gated universe.** In the training and validation repositories 1,176 such rows are T and 3,433 are F; 783 of the T rows are in two repositories, and the names in front of them are those of Postman collection exports (`id`, `_postman_id`, `collectionId`, `folder`). The label says which file the identifier is in, not whether it is a credential, and a firewall for coding agents has no reason to hide identifiers: proposing every UUID as a candidate would be the opposite of what G7 asks. So:
  - the generator gets no UUID rule;
  - in §7.1 and §7.2, a row whose `Category` is exactly `UUID` is removed from the universe, for every system: a line whose only rows are such rows is neither positive nor negative, and a flag on it is not counted. Rows with a compound category (`Token:UUID`, `Key:UUID` and the like) stay. G1, G2, G3, G5 and the bootstrap are read on that universe;
  - every number of §7 is also reported on the full universe, UUID rows included, for every system, next to the gated one. This choice was made after reading the training and validation rows and before any test file: it is said wherever the scores are reported;
  - in training, a candidate that overlaps a UUID row keeps that row's label, like any other.
- **The one revision of the generator (§2.2), which is wider than "keywords, separators, value length".** §2.2 allowed one revision and limited it to three things; the measurement shows forms those three cannot express. The revision stays single and is defined here before it is coded or measured; after it the generator is frozen whatever its recall, as §2.2 says.
  - `generic-keyword`, revised: (1) the name, as implemented: a maximal run of letters, digits, `_`, `.`, `$`, `-` that contains a keyword, the keyword list unchanged; (2) an optional closing quote of the name, plain or escaped with a backslash, then an optional `]` or `)`; (3) a separator of the same list, with spaces or tabs around it; (4) an optional `[`; (5) an optional opening quote, plain or escaped; (6) an optional scheme word, one of `Bearer`, `Basic`, `Token`, `Digest`, `Negotiate`, `ApiKey`, case-insensitive, followed by spaces, which is not part of the value; (7) the value, 4 to 256 characters, as before: a quoted value runs to the same quote, escaped if it was opened escaped; a bare value runs to the first whitespace, `;`, `,` or the end of the line.
  - One new rule, `auth-scheme`: the word `Bearer` or `Basic` at a word boundary, case-insensitive, spaces, then a value of 16 to 512 characters among letters, digits and `. _ ~ + / = -`. It covers a header value whose header name is on another line. It ranks with `generic-keyword`, below every engine rule.
  - Not added, on purpose: a UUID rule, call arguments (`name(value`), names on the previous line, URL parameters, bucket and domain rules. Their rows stay in the universe as misses, except the UUID rows above.
- **Readings fixed as implemented** (the implementation chose them before the measurement; none is changed after it): a keyword may sit anywhere inside the name; quotes are `"`, `'` and the backtick; the separators are exactly the six listed; a text longer than 256 characters is not a candidate; lengths count UTF-16 units; inside the `post` gates of the built-in rules the placeholder test is off and the rest stays on; the engine's own placeholders and its own rule sources are never candidates; `generic-keyword` and `auth-scheme` rank below every engine rule in `dedupe`; the filter flags of a candidate are those of the rule that names it; files up to 64 MiB are read and files with a NUL byte in their first 8,192 bytes are skipped; the volume counts the lines of every file of the repositories.
- **Matching (§2.2, §3.1).** A multi-line row is one span from `ValueStart` on `LineStart` to `ValueEnd` on `LineEnd`. A T row with no `ValueStart` (95 in training and validation, although CredData's README reserves whole-line rows to F) is covered by any candidate on its line. A row with a start and no end runs to the end of its line. A candidate that overlaps rows of several labels takes T if one of them is T, else X if one is X, else F.
- **The clean-text corpus (§7.7).** Its list is the one written at the freeze, `gen/clean-corpus.json`: 8,825 files at this date (8,821 when §7.7 was written; files were added since). Nothing has been run on it.
- **What does not change.** The 0.95 stays the target and the recall after the revision is reported, on both universes. The model, the data, the budgets, the baselines, the thresholds of G1 to G7, and §12.

**Amendment 3 (14/11/2026, after the one revision of the generator and its single measurement; no test file has been opened, nothing has run on the clean-text corpus, no model or baseline exists). The generator is frozen (§2.4).**

- **The frozen generator.** `sg-cand-1`, revision 1, at secretgate commit `4e93cfa2ee6e216caca9bc8e6f74cb565a991350` (branch `chore/sg-cand-1`, pushed): the revision of amendment 2, coded word for word, with the narrow readings its implementation had to take written at the top of `scripts/rule-candidates.mjs` and pinned by tests (nothing between the steps; an optional element that is present is never the start of the value; a scheme word is followed by one or more spaces; an escaped quote is one backslash and a quote; `auth-scheme` reads ASCII letters and takes the whole run of allowed characters; of two overlapping candidates the wider wins). No bug fix was needed after the run, and there was one run. `runs/s6-secrets2/generator.json`, committed with this amendment, holds the commit, the hashes of the rules, of the script, of the candidate and label files and of the clean-text list, and every number below.
- **Recall on the training and validation repositories, reported as §2.2 asks.** On the gated universe (rows of category exactly `UUID` removed): 0.8763 (2,606 of 2,974 T rows; training 0.8817, validation 0.8506), against 0.3894 for the default engine. On the full universe: 0.6383 (2,649 of 4,150), against 0.2790. The target was 0.95 and is missed on both; the generator stays as it is. 13.5 candidates per 1,000 lines (216,036 candidates), of which 23,599 on labelled lines.
- **What it still misses on the gated universe (368 rows):** values with no credential keyword on their line (bucket names, domains, seeds: 190), a later element of a list (53), call arguments (23), command-line and SQL passwords (22), and smaller forms. These misses count as false negatives in every line-level score of §7.1, as §2.3 says: with a recall of 0.88 before the model, the system's recall cannot exceed it.
- **Labels of the candidates (training, validation):** T 2,127 and 459; X 726 and 117; F 17,875 and 1,203; on unlabelled lines 190,158 and 2,279 (the pool of the weak negatives of §3.1); on a labelled line but overlapping no labelled value, dropped and counted as §3.1 says, 916 and 176. 16,136 of the F labels and 99 of the T labels come from a whole-line row.
- **The pinned commit and the later history of secretgate.** The experiment calls the generator at this commit. The branch it sits on is being changed by fixes that are unrelated to the experiment (one of them adds a rule to the engine); a generator rebuilt on a later commit is another generator, and a release that ships a verifier has to ship the candidate mode of this commit or be measured again.
- **The clean-text corpus.** Its list was regenerated at the frozen commit: 8,825 files, sha256 of the list `99877d1ea8f41d7c52980a114a537afa63ab64bd100604b0471ad20112f1e073` (in `generator.json`). The list itself (2.3 MB) stays in the data folder.

