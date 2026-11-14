# s6-secrets v2: feasibility study (a calibrated verifier for rule candidates)

Research date: 14 November 2026. Read-only: no data set was downloaded, nothing was installed, nothing was run on CredData or on the issue-report set. Each fact gives its source (URL or file path). **[Inference]** marks my own reasoning. "Not verified" means I looked and could not confirm. No credential value appears in this file; formats are described in words.

## 0. Verdict

- **Feasible on one laptop with public data, with three caveats.**
  1. CredData's `download_data.py` expects Linux. On this Mac it would run in a Linux container (Docker is installed at `/usr/local/bin/docker`).
  2. The cbx test split cannot be rebuilt exactly, because its 168 test repositories are not listed anywhere. A cbx-compatible split *can* be built: cbx publishes its 116 training repositories with their commits, so every other CredData repository is unseen by cbx.
  3. The temporary clones of CredData and the issue-report archive both hold real credentials on disk.
- **Compute.** Every baseline and every evaluation fits on a laptop CPU. The slowest is cbx-17M on the test files: an estimated 70 min with 8 threads (§4.5). The training budget proposed in the draft is about 40 working minutes.
- **The biggest risks.**
  1. **The obfuscation artefact (§6.1).** CredData regenerates every true value with random characters and keeps every false value raw. The labels can therefore be read from the shape of the value.
  2. **Generator recall (§3).** Recall is capped by the candidate generator, and secretgate's default rules alone reach only about 0.3 on CredData.
  3. **The issue-report split is random per candidate (§2).** Train and test share issues, so its scores are optimistic.

---

## 1. Samsung CredData

### 1.1 Structure
- **Repository**: https://github.com/Samsung/CredData. HEAD was `01e3d135a9` on 12 Nov 2026, and the `meta/` folder was last edited in Aug–Sep 2026 (GitHub API). The licence of the repository itself is Apache-2.0.
- **Meta files**: 337 CSV files under `meta/`, 6.27 MB in total, one per repository. Source: the tree API, https://api.github.com/repos/Samsung/CredData/git/trees/main?recursive=1.
- **Header, exact** (written by `download_data.py`; the same header appears in `meta/528ba141.csv` and `meta/2cca776b.csv`):
  `Id,FileID,Domain,RepoName,FilePath,LineStart,LineEnd,GroundTruth,ValueStart,ValueEnd,CryptographyKey,PredefinedPattern,Category`
- **Column semantics** (README, https://github.com/Samsung/CredData/blob/main/README.md):
  - `LineStart` and `LineEnd` are 1-based.
  - `ValueStart` and `ValueEnd` are character offsets on the line. An empty value or -1 means the whole line is marked, and is used for False rows only.
  - `FileID` is the first 8 hex characters of the sha256 of the short path.
  - `Category` holds one or more CredSweeper rule names joined by `:`.
- **Labels** (`constants.py`, https://raw.githubusercontent.com/Samsung/CredData/main/constants.py):
  - `T`: a real-looking credential;
  - `F`: a false positive;
  - `X`: a test, example or placeholder value.
  The README counts True = T, and False = F or X.
- **Category values**: about 60 rule names, listed in https://raw.githubusercontent.com/Samsung/CredData/main/.ci/benchmark.txt. They include Password, Token, Secret, Key, API, Auth, UUID, PEM Private Key, JSON Web Token, AWS Multi, URL Credentials and Basic Authorization.
- **Counts**:
  - `.ci/benchmark.txt` (current) gives 11,565 files, 17,020,109 lines and "MARKUP: 64797 items". It also gives 18,222 positives and 54,116 negatives counted per rule. Those two add up to more than 64,797, probably because rows with several categories are counted once per category **[Inference]**.
  - The README still gives the older figures: 297 repos, 19.46M lines, 73,842 labelled lines, 4,583 True.
  - **The current number of T rows is not verified**: counting it means fetching all 337 meta files.
- **How a line is identified**: (`RepoName` as the 8-hex RepoID, `FileID`, `LineStart`). A value is (`LineStart`, `ValueStart`, `ValueEnd`).
- **Labelling depends on what tools flagged [Inference]**: the categories are CredSweeper rule names. A line that no rule flagged at labelling time has no row, so "unlabelled" means "not judged", not "negative".

### 1.2 Obfuscation (what is preserved)
Source: `obfuscate_creds.py`, https://raw.githubusercontent.com/Samsung/CredData/main/obfuscate_creds.py.

- **Only `T` rows are obfuscated.** The multi-line categories are skipped: AWS Multi, Google Multi, JWK, Alibaba Multi and Twilio Multi.
- **Single values** go through `process_single_value` → `get_obfuscated_value`. What survives:
  - **length**;
  - **the class of each character**: lower or upper hex, base32, digit, case;
  - **separators and escapes**: UUID dashes, `%xx`, `\x` and `\u` escapes, `0x` byte arrays;
  - **known provider prefixes** (`get_prefix`): AWS identifiers, Stripe live and test prefixes, GitHub and npm token prefixes, Slack `xox`, JWTs re-encoded from their `eyJ` start, Anthropic, Docker and others.
- **`generate_value` rejects some draws.** It draws again if the new value contains "api", "key", "pass" or "pwd", has 4 or more repeated characters, or has a 4-character ascending or descending run. **Obfuscated true values therefore never contain such substrings**, while raw false values often do.
- **PEM keys** (`process_pem_key` → `create_new_key`): the BEGIN and END markers and the first 64 characters are kept. After that, letters become random letters and digits become random digits.
- **Determinism**: the seed comes from `ValueStart`, `LineStart`, `FileID` and an optional `--noise` value.
- **Left as they were**:
  - **variable and key names**;
  - **the values of F and X rows**;
  - **every unlabelled line**.
  Every file that has a meta row is still decoded as UTF-8 and rewritten with LF line endings.
- **Paths are obfuscated.** A file becomes `data/<crc32 of snapshot key>/<scope>/<FileID><ext>`. The scope comes from `word_in_path.json` (seen values: `test`, `docs`, `src`, `security`, `build`, `_` for other). Only this scope word and the file extension survive.

### 1.3 Fetching
Source: `download_data.py`, https://raw.githubusercontent.com/Samsung/CredData/main/download_data.py.

- **Method**: for each repository, `git init`, `remote add`, `fetch --depth 1 origin <commit>`, `checkout`. The commit is the first 40 hex characters of a 64-hex key in `snapshot.json`. There is no release tarball. `snapshot.json` lists 337 repositories, all on github.com, with plain URLs. The README says the repository identities are not published, which contradicts that file.
- **Requirements**: Linux (the README cites filename problems on other file systems), Python 3.10, `requirements.txt` (GitPython, tabulate, pybase62, colorama) and git.
- **Size on disk and run time**: **not stated, not verified**.
  - **[Inference]** The kept files hold about 17M lines; at about 50 bytes per line that is about 0.9 GB.
  - **[Inference]** The depth-1 clones of 337 whole repositories are transient and could take several GB. This Mac has 52 GiB free (`df`, 14 Nov).
  - **[Inference]** Expect about an hour to a few hours of network-bound fetching.
- **Real credentials**: the clones are the original repositories at their pinned commits, so **they hold the real values before obfuscation**. Only the copy written to `data/` is obfuscated. The clones must stay in an isolated folder and be deleted after the script runs. **[Inference]** from the fetch-then-obfuscate design.
- **Licences**: there is no licence column. `collect_licenses()` copies each repository's LICENSE or COPYING files into `data/<RepoID>/` at download time. The README mentions a "license directory" that does not exist in the repository.
- **Current licences on GitHub** (GraphQL, current metadata, *not* the licence at each pinned commit), 337 repositories:

  | licence | repos |
  |---|---:|
  | MIT | 185 |
  | Apache-2.0 | 94 |
  | BSD-3-Clause | 16 |
  | ISC | 4 |
  | BSD-2-Clause | 4 |
  | CC0 | 3 |
  | UPL | 1 |
  | **permissive, total** | **307 (91%)** |
  | copyleft (GPL 12, LGPL 3, MPL 2, EPL 1, OSL 1) | 19 |
  | none or NOASSERTION | 11 |

- **Train/test split**: CredData has none. Its README benchmark was run "combining training and test data". The only built-in separation is the path scope (`src` vs `test`).
- **Published benchmark numbers**:
  - The README table is dated April 2022. CredSweeper reaches F1 0.859 (P 0.917 / R 0.808), Gitleaks 0.334, detect-secrets 0.206 and truffleHog3 0.235.
  - CredSweeper's current CI file (https://raw.githubusercontent.com/Samsung/CredSweeper/main/.ci/benchmark.txt, changed 30 Sep 2026, same meta MD5 as CredData) reports TP 17,802, FP 161, FN 420: P 0.991, R 0.977, F1 0.984.
  - That CI number is not held out (§4.4).

### 1.4 The cbx-secrets-detector-17M split
Sources: card https://huggingface.co/CobuildX-AI/cbx-secrets-detector-17M/resolve/main/README.md and API https://huggingface.co/api/models/CobuildX-AI/cbx-secrets-detector-17M.

- **The model**: v0.1, Apache-2.0, created 30 Oct 2026. Fine-tuned from Ettin-17M (ModernBERT), 16,803,865 F32 parameters, `model.safetensors` 67,220,156 bytes. The 32M variant's file is 127,577,684 bytes.
- **Model interface**:
  - 12 BIO labels;
  - `max_length` of 7,999 tokens, so a whole text is read in one pass;
  - longer texts are cut with a 64-token overlap (`cbx_secrets.py`);
  - the shortest span kept is 10 characters, or 4 for PASSWORD and DATABASE_CREDENTIAL (`secrets_config.json`).
- **Split as the card states it**:
  - test is "the test half of the repositories": 168 repositories with 3,493 secrets and 17,961 labelled non-secret lines;
  - training is 116 repositories under MIT, Apache-2.0 or BSD.
- **Scoring**: span F1, independent of the label. A prediction counts if it overlaps a true secret. "False alarms" is the share of non-secret lines on which something is flagged.
- **`training_data_sources.csv`**: 10,724 bytes, columns `repository,commit,licence,training_rows`.
  - 116 rows: MIT 72, Apache-2.0 34, BSD 10. `training_rows` sums to 7,244.
  - All 116 commits appear in CredData's `snapshot.json`.
  - **No list of test repositories exists** in the 17M repo, the 32M repo, or any CobuildX-AI dataset (https://huggingface.co/api/datasets?author=CobuildX-AI lists only CredDetX, a synthetic CC-BY-4.0 set of 3,027 texts).
- **Reported results** (from the card):

  | model | CredData test F1 (P / R) | false alarms | issue reports F1 (P / R) | false alarms |
  |---|---|---:|---|---:|
  | cbx-17M | 0.805 (0.80 / 0.81) | 0.9% | 0.836 (0.90 / 0.78) | 0.9% |
  | cbx-32M | 0.820 (0.77 / 0.87) | 1.3% | 0.861 (0.86 / 0.86) | 1.4% |
  | Gitleaks 8.30.1 | 0.442 (0.90 / 0.29) | 0.6% | 0.678 (0.93 / 0.54) | 0.5% |
  | detect-secrets 1.5.0 | 0.460 (0.47 / 0.45) | 9.8% | 0.290 | 4.0% |
  | TruffleHog 3.99.2 (offline) | 0.042 (0.38 / 0.02) | 0.8% | 0.228 | 0.9% |

- **Latency**: 37 ms per 1,000-token text on an i7-10700 CPU with 8 threads (p95 38 ms); 6 ms on a TITAN RTX in fp16.
- **Can the cbx protocol be reproduced exactly? No.**
  1. The 168 test repositories are not listed. 116 + 168 = 284 of 337. A plausible reading is a 169/168 split from which 53 repositories of the training half failed the licence filter **[Inference, not verified]**.
  2. No CredData commit and no meta MD5 is given, and the meta folder changed during 2026.
  3. The card does not say how T, F and X map to positive and negative. "17,961 non-secret lines" reads as F and X rows **[Inference]**.
  4. The `--noise` seed of the obfuscation is not stated (0, the default, is likely).
  5. It is not said how flags on unlabelled lines, or multi-line PEM spans, are counted.
- **What can be built**: a **cbx-compatible split**.
  - **Training** uses only repositories from cbx's 116, at their listed commits.
  - **Test** is every other CredData repository (about 221 at the pinned meta).
  - **[Inference]** cbx-17M never trained on any test repository, so rerunning it there is a fair comparison. The numbers will not equal the card's 0.805, so every baseline is rerun through the same scorer.

## 2. Secret leaks in issue reports (Zenodo 10.5281/zenodo.19622962)

**Record** (https://zenodo.org/api/records/19622962):
- published 16 Apr 2026, resource type "Software", licence **CC BY 4.0**;
- Zenodo lists one creator (Md Nafiu Rahman); the paper has five authors (Ahmed, Rahman, Wahab, Uddin, Shahriyar);
- concept record 17430335 has 4 versions: 17430336 and 17463769 (both 24 Oct 2025), 17903162 (12 Dec 2025), and 19622962, the latest.

**Files** (not downloaded):
- `Secret-Leak-Detection-Issue-Report.zip`, **134,502,037 bytes**;
- `README.md`, 7,859 bytes.

**Archive contents** (Zenodo preview, https://zenodo.org/records/19622962/preview/Secret-Leak-Detection-Issue-Report.zip), under `Data/`:

| file | size | note |
|---|---:|---|
| `main_data.csv` | 262.0 MB | |
| `train.csv` | 176.7 MB | |
| `val.csv` | 23.0 MB | |
| `test.csv` | 35.2 MB | |
| `test_wild.csv` | 5.5 MB | 178 repositories |
| `CODING_GUIDELINES.md` | 9.9 kB | |

**Columns**:
- No document lists them.
- The authors' code (https://github.com/nafiurahman00/Secret-Breach-Prevention-In-Software-Issue-Reports, `Data-Handling/Dataset_Generation_labelled.ipynb`, `Model/bert_training.ipynb`) reads `Issue_id`, `text`, `candidate_string` and `label` (0/1), plus a pandas index column.
- These columns are **not verified against the released CSVs**.

**Paper** (arXiv 2410.23657 v4, https://arxiv.org/html/2410.23657, §3.1):
- **Collection**: issues from active non-fork repositories, fetched through the GitHub REST API by keyword: 50,680 issues in 27,121 repositories.
- **Candidates**: 761 regexes taken from SecretBench, 751 of them TruffleHog-derived and 10 from Meli et al. Before matching, 22 further regexes remove benign strings (commit hashes, UUIDs, build IDs). The regex list is `Secret-Regular-Expression.xlsx` in https://github.com/disa-lab/IssueGuard.
- **Labels**: manual, from the context, with no live validation "due to ethical, legal, and security concerns". A second rater labelled 1,000 random candidates: Cohen's kappa 0.9545.
- **Size**:
  - 54,148 candidates, 5,881 of them secrets;
  - the paper says 48,297 non-secrets, which does not add up; the README's 48,267 does.
- **Splits (Table 7)**: 75/10/15 "preserving class proportions".

  | split | candidates | secrets | non-secrets |
  |---|---:|---:|---:|
  | train | 40,593 | 4,377 | 36,216 |
  | val | 5,432 | 586 | 4,846 |
  | test | 8,123 | 888 | 7,235 |

  The secrets in the three splits sum to 5,851, not 5,881.
- **How the split was made: random per candidate, not by issue or repository.** The notebook calls `train_test_split(random_state=42)` separately on secrets and non-secrets. `Issue_id` is a random UUID per row, so candidates from the same issue text can fall in train and in test.
- **Regex-only baseline**: recall 100% by construction, precision 6.8%. Regex plus entropy: recall 97.5%.

**How cbx used it** (cbx card):
- **Training**: every real secret in the train split was "replaced by an invented value of the same shape", and the texts were cut into windows.
- **Test**: "949 secrets and 8,132 non-secret candidates", after removing texts that also appear in training. That makes 9,081 rows, more than the paper's 8,123 for its test split. **Why they differ is not verified.** The card does not say whether the test values were replaced.

**Real credentials**:
- The archive holds them in clear: these are secrets that were already public in issues.
- The only statement on handling them is the README's note that the data "contains only publicly available information from GitHub issues that were already exposed at the time of collection".
- CC BY 4.0 needs attribution (the DOI and the paper).
- **[Inference]** The issue texts are GitHub user content. The CC BY 4.0 grant comes from the dataset authors, not from each issue author, and the model card should say so.

## 3. Candidate generation with secretgate's engine

### 3.1 What a candidate is
- **Engine**: secretgate `main` at `ab9c9a3`. Its rules are the gitleaks default config at commit `09242ce9c8a60d9b051fc2d166f9e849b88c7ac0`, fetched 13 Jul 2026 (`rules/UPSTREAM` in that repository): 222 `[[rules]]` in `rules/gitleaks.toml`, compiled into `src/engine/rules.gen.ts`. On top of them, `src/engine/scanner.ts` adds six built-in rules (lines 195–295): `credit-card-number`, `url-credentials`, `password-assignment`, `groq-api-key`, `replicate-api-token`, `xai-api-key`, `discord-webhook-url`.
- **What a finding holds** (`Finding`, `scanner.ts:7-17`): `ruleId`, `match` (the whole regex match), `secret` (the capture group, picked by `pickSecret`, lines 386–400), `start`/`end` (the offsets of the secret in the scanned text), `entropy`, and `line` (0-based). The scanner also knows the source path (`ScanConfig.sourcePath`) and the text, so the variable name and the surrounding lines can be cut from `match` and `text`.
- **[Inference] Candidate for v2** = {rule id, value (the secret group), variable/key name (the text of `match` left of the separator), value span, line text, a bounded context (the previous and next lines), path scope and extension, the engine's entropy}.
- **The filters run inside `scan()`, before a finding is returned** (lines 425–463):
  - `post` and `postMatch` gates;
  - the placeholder test `looksLikePlaceholder` (line 441);
  - the entropy gate (line 444): 3.5 for `generic-api-key`, 3 for `password-assignment`;
  - pragma lines;
  - per-rule allowlists, including the 1,456 stopwords of `generic-api-key` (`rules/gitleaks.toml:638-`);
  - global regexes and stopwords;
  - user allowlists;
  - then `dedupe` (lines 476–491).

  **[Inference]** These filters are what the verifier is meant to replace, so the generator needs a *candidate mode* that returns the matches before the entropy, stopword and placeholder filters. The global path allowlist (lock files, images, `node_modules`) and the keyword prefilter stay on. No such mode exists today; it is a small change in secretgate (a parallel secretgate branch, `feat/s6-secrets-model`, is preparing a comparison harness whose adapter contract already foresees "units built from the scanner's findings": `scripts/s6-secrets-pico.mjs:9-11`, uncommitted).

### 3.2 Would the default rules reach high recall on CredData-style generic secrets? No.
- **The generic rules are narrow.**
  - `generic-api-key` (`rules/gitleaks.toml:640`) needs a keyword, a separator, and a value of 10–150 characters in `[\w.=-]` (or a base64-like run), with entropy above 3.5. Short passwords, values with punctuation (`!@#$%`), and values that contain a stopword are lost.
  - secretgate's `password-assignment` (`scanner.ts:247-261`) widens the character set. It still needs a *quoted* value of 6–200 characters with no whitespace and entropy above 3.
- **Published numbers**:
  - Gitleaks 8.30.1 recall on the cbx CredData test: 0.29 at precision 0.90 (cbx card).
  - CredData README (2022): Gitleaks recall 0.244.
  - **[Inference]** secretgate's default engine derives from the same rules and adds about one rule for generic passwords, so its recall is likely of the same order, perhaps 0.3–0.45. **Not measured.**
- **Recall with the filters off**: Betterleaks' author claims 98.6% recall on CredData for token-efficiency filtering, against 70.4% for entropy. This is not in the Betterleaks repository and has not been reproduced, so **not verified** (https://rafter.so/blog/secrets/betterleaks-replaces-gitleaks ; https://appsecsanta.com/betterleaks). **[Inference]** Raw gitleaks-style matches with the entropy and stopword filters off reach somewhere between 0.7 and 0.95 recall. Unquoted short values remain out of reach.
- **An extra high-recall generic rule is needed**, as the high-recall systems all have:
  - **CredSweeper** (https://github.com/Samsung/CredSweeper, `credsweeper/rules/config.yaml`, v1.19.6, MIT):
    - 153 rules: 138 pattern, 9 keyword, 5 multi, 1 pem_key; 24 of them have `use_ml: true`.
    - The 9 keyword rules are API, Auth, Credential, Key, Nonce, Password, Salt, Secret and Token. All of them go to ML.
    - Each keyword rule fills the template in `credsweeper/common/keyword_pattern.py`, in this order:
      1. an optional directive;
      2. a variable made of a prefix, the keyword and up to 80 name characters;
      3. a separator: `:`, `=`, `:=`, `=>`, `==`, `!=` or `<<<`;
      4. an optional wrap such as a `get(...)` or `getenv(...)` call or a bracket;
      5. an optional string prefix and quote;
      6. an optional auth scheme (bearer, basic, token…);
      7. a value of 4 to 8,000 characters up to the closing quote or delimiter;
      8. the closing quote.
    - The template is compiled IGNORECASE and DOTALL.
    - CredData's labels were made on CredSweeper's candidates (the `Category` column holds CredSweeper rule names). **[Inference]** A CredSweeper-style keyword rule therefore reaches close to full recall on CredData by construction, and other generators are measured against a candidate space they did not define.
  - **IssueGuard** (https://github.com/disa-lab/IssueGuard, no licence file): the 761 regexes, compiled case-insensitive (`api/regex_manager.py`). They include a "Generic Pattern": a keyword (pass, token, cred, secret, key), then up to 40 characters, then a run of 16–64 printable ASCII characters. Its regex stage recall is 100% by construction.
  - **GitHub** (13 Oct 2026, https://github.blog/ai-and-ml/githubs-model-for-secret-detection-finding-more-secrets-with-less-noise/): "Regex filters scan content for potential secrets with high recall". The classifier reads "the value together with its file path, variable names, and surrounding code", plus the secret type.
- **[Inference] Recommendation**:
  - generator = secretgate rules in candidate mode;
  - plus one CredSweeper-style keyword rule: keyword in the variable name, a separator, a value of at least 4 characters, quoted or unquoted up to a delimiter;
  - its recall is measured on the *training and validation* repositories only, before it is frozen.

## 4. Baselines that can be rerun locally

| baseline | version (date) | licence | install | output / threshold | notes |
|---|---|---|---|---|---|
| Gitleaks | **8.30.1 installed** (`gitleaks version` → 8.30.1, `/opt/homebrew/bin/gitleaks`). It is the latest release (21 Mar 2026), and the project is "feature complete, security patches only" | MIT | none | `gitleaks dir` (alias `files`) scans plain folders. JSON output with RuleID, StartLine, EndLine, StartColumn, EndColumn, Match, Secret, File, Entropy, Fingerprint (https://github.com/gitleaks/gitleaks/blob/master/report/finding.go). No score; a curve only by sweeping the logged Entropy afterwards | minutes on CPU **[Inference]** |
| Betterleaks | 1.10.1 stable (12 Nov 2026); 2.0.0-rc.8 is a pre-release (13 Nov 2026) | MIT | `brew install betterleaks`, darwin_arm64 tarball 9.8 MB | v1 reads Gitleaks configs; v2 does not (`docs/v2_migration.md`). Token efficiency = bytes per token under cl100k_base, embedded in the binary. A candidate fails if it holds a dictionary word of 5 or more letters, or if its ratio is ≥ 2.5 (≥ 2.1 under 12 characters with a 4-letter word). `tokenRatio()` is exposed in filters, so it can be swept | https://github.com/betterleaks/betterleaks ; `internal/exprruntime/bindings_filter.go` (default branch) |
| detect-secrets | 1.5.0 (6 May 2024), still the last release; last commit 2 Apr 2026 | Apache-2.0 | wheel 120 KB, plus pyyaml and requests | `scan --all-files DIR`. Reports file, line, type and a hashed secret (fields from memory, **not verified**). No score; partial curves through `--base64-limit` (default 4.5) and `--hex-limit` (default 3.0) | https://pypi.org/pypi/detect-secrets/json |
| CredSweeper, ML on and off | 1.19.6 (5 Nov 2026) | MIT | wheel 10.8 MB, about 30 dependencies (onnxruntime, pandas, lxml, cryptography); installed size **not verified** | `ml_probability` in the output gives a full curve. `--ml_threshold` presets: lowest 0.229, low 0.357, medium 0.622 (default), high 0.798, highest 0.930. `--ml_threshold 0` turns ML off; `--no-filters` exists | model `ml_model.onnx` 11,545,911 bytes, ONNX CPU. Features (`ml_model/ml_config.json`, 34): rule severity, Shannon entropy, line/variable/value lengths, 18 regex tests on value or variable, word lists for variable, value, context and path, morpheme density, file extension (170), rule name (16). Char inputs: one-hot over 99 characters, the line up to 128 characters centred on the value, the first 64 characters of the variable and of the value (`ml_validator.py`, `common/constants.py`). No CPU figure published. **Contaminated on CredData** (see below) |
| cbx-secrets-detector-17M | v0.1 (30 Oct 2026) | Apache-2.0 | `model.safetensors` 67,220,156 bytes, `tokenizer.json` 3.58 MB, `cbx_secrets.py`; needs torch ≥ 2.1 and transformers ≥ 5.0 (sizes not verified) | spans with a label and offsets, **no score**: no curve without extra code reading the logits | trained on 116 CredData repositories and the issue-report train split |
| size-matched simple baseline | scikit-learn 1.9.1 (BSD-3), or a Rust implementation in `sky-data` | BSD-3 | scikit-learn and numpy are **not installed** (system Python 3.15.2 has neither; `uv` is available) | hashed character n-grams (1–5) of the same rendered candidate, plus the same hand features, logistic regression; a full curve | 2^18 weights: 1.0 MiB at f32, 256 KiB at int8, the size of the pico GGUF (250,272 bytes) |
| Model2Vec potion-base-2M + logistic regression | model2vec 0.10.0 (11 Oct 2026) | MIT | `model.safetensors` 7,559,256 bytes; tokenizer bge-base-en-v1.5 (WordPiece) | `StaticModelForClassification` is the documented head and needs torch to train | 30x the pico file, so reported, not size-matched (https://huggingface.co/minishlab/potion-base-2M) |
| TruffleHog (offline) | 3.99.9 (12 Nov 2026) | AGPL-3.0 | darwin_arm64 tarball 75.9 MB | `filesystem DIR --no-verification`; `--filter-entropy` | F1 0.042 on the cbx CredData test; not worth running **[Inference]** |

- **CPU cost of cbx-17M on the test split (estimate).**
  - The current CredData data holds 17,020,109 lines in 337 repositories, so the test repositories hold about 221/337 × 17.0M ≈ 11.2M lines.
  - At about 10 tokens per line that is about 112M tokens. At 37 ms per 1,000 tokens (i7-10700, 8 threads) this comes to **about 69 min**, assuming linear scaling.
  - The same arithmetic for the card's 168 repositories gives about 68 min.
  - Not measured on this Mac, which shares its cores with a GPU training run.
- **What cannot be reproduced.**
  - **CredSweeper's ML is not a held-out baseline on CredData.** `experiment/train.py` uses `train_test_split(df_all, test_size=0.15, ...)`, a random split by line over all of CredData. Its CredData numbers can only be reported as "seen in training".
  - **Betterleaks' 98.6% recall** has no published protocol.
  - **cbx's exact test set** (§1.4).
  - **GitHub's and GitGuardian's models** are not available.
  - **IssueGuard's CodeBERT** (code public, no licence; about 125M parameters) is not size-relevant; its published 92.70 F1 is on the random split.

## 5. Input format for the model

- **Tokenizer**: s6-secrets-pico uses `data/tok/tok-s2-1024.json` (`scripts/train-secrets.sh:72-77`). It is a byte-level BPE with an NFC normalizer, the GPT-2 split regex and **only 762 merges**. Special ids are `[PAD] [UNK] [CLS] [SEP] [MASK] [NUM]`.
- **Numeric extraction** (`crates/sky-infer/src/num.rs:25-82`) runs first:
  - every `-?\d+(?:\.\d+)?` becomes `[NUM]` with its f32 value, unless it is glued to a preceding ASCII letter;
  - with `fuse_numbers` (SPEC-S3 §1), the value moves onto the previous token, so `key value` costs one token for the value.
- **Random strings cost about one token per character.** The GPT-2 split separates letters, digits and punctuation, and 762 merges rarely join random characters. Measured with a throwaway Python re-implementation of this pipeline, since deleted: numeric extraction, the GPT-2 split, the ranked merges of `tok-s2-1024.json`, and `[NUM]` fusion. It approximates `crates/sky-infer/src/tokenizer`; it is not the engine itself. On 200 synthetic random strings each:

  | string | tokens per character |
  |---|---:|
  | hex, 40 characters | 0.94 |
  | base64, 40 | 0.97 |
  | alphanumeric, 32 | 0.97 |
  | 12-character password with symbols | 0.96 |
  | lowercase, 10 | 0.86 |

  This matches the s6-secrets prereg, which says "about one token per character" (`docs/s6-secrets-prereg.md` §2).
- **Common words are not single tokens.** "key", "path", "value", "password", "secret" and "token" each take several tokens (vocabulary check). Ordinary text such as a short config line costs about 0.6 tokens per character: 37 tokens for a 60-character `export` line, 28 for a 47-character path.
- **Digits inside values.** A digit run after a separator (`_`, `/`, `.`, `=`, a quote) is pulled out as a number, keeping its magnitude but not its exact digits. Literals beyond the f32 range become 0 (`num.rs:71-74`). For obfuscated values this is harmless; for value shape it blurs the length of digit runs **[Inference]**.
- **Truncation keeps the start.** `build_state` (`crates/sky-infer/src/s2/prompt.rs:182-206`) truncates the state to `max_state_tokens - 1` and keeps the beginning, so the value, key and features must come first and the context last.
- **Budget [Inference]**, in a 320-token state:

  | part | tokens |
  |---|---:|
  | header (rule id, path scope, extension) | about 20 |
  | key name (at most 40 characters) | about 25 |
  | value (first 32 and last 32 characters if longer than 64) | at most 64 |
  | about 8 hand features as `name value` pairs | about 25 |
  | the line, with the value replaced by a short marker, at most 120 characters | about 80 |
  | the previous and next lines, cut to fill the rest | the remainder |

  The parts before the neighbour lines total about 215 tokens, which leaves about 100 tokens of neighbour context in 320. The s6-secrets-pico states ran to 93 tokens (median), 179 (p95) and 312 (max) (`BENCH.md`, s6-secrets-pico; SPEC-S6 §14), so 320 already covers this. A rule in the draft measures truncation on training candidates and moves to 384 only if needed. RoPE has no position table, so the parameter count does not change (SPEC-S6 §14).
- **Hand features known to help.**
  - **CredSweeper**: Shannon entropy; line, variable and value lengths; regex tests on value and variable (env reference, camelCase, UUID, hash names, PEM prefixes); word lists for variable, value, context and path; file extension; rule name (§4).
  - **Betterleaks**: token efficiency, bytes per BPE token.
  - **GitHub**: path, variable name, secret type.
- **Can the number tokens carry them? Yes.**
  - Each `name value` pair costs the tokens of the name plus nothing for the value (fused).
  - The value goes through φ(v) = [u, sin(u·ω_k), cos(u·ω_k)] with u = sign(v)·ln(1+|v|) and ω_k = 2^k·π/8, k = 0..7 (SPEC-S2 §2). That resolves entropy (0–6 bits) and lengths (1–200) well.
  - **[Inference]** Token efficiency should be computed with the model's own 1,024-token BPE (characters per token of the value): it costs nothing extra, and both the trainer and the TypeScript host can compute it. cl100k_base would add a large table to the host.
  - Features should be computed by one shared definition in Rust (training) and TypeScript (secretgate), checked by a fixture both must pass, as `num_extract.json` is for numbers (SPEC-S2 §2).
- **Path in CredData**: only the scope word and the extension survive (§1.2). **[Inference]** The input should render the path as `scope + extension` everywhere, computed at deployment with CredData's own `word_in_path.json` mapping, so that training and use see the same thing.

## 6. Risks

### 6.1 Obfuscation artefacts (the largest risk)
- **What is kept** in true values (§1.2): length, character class per position, separators and escapes, provider prefixes. Variable names and context are kept.
- **What is lost**: the actual characters, including dictionary words inside real passwords. Draws containing "api", "key", "pass" or "pwd", 4 repeated characters or 4-character runs are rejected.
- **Why it matters**: F and X values are raw. A model can learn that a value containing a word, a repetition or a run is not true, which is an artefact of the data set and not a property of secrets. Real passwords made of a word, digits and a symbol would then be under-called in real use. **[Inference]**
- **Mitigations (proposed in the draft)**:
  - a real-value gate on the issue-report test set;
  - an *obfuscation sensitivity* measure: the issue-report test scored with its true values raw, and again passed through CredData's own obfuscation function;
  - word-preserving replacement for the issue-report training positives;
  - synthetic word-based password positives;
  - a value-only probe that measures how far value shape alone separates T from F/X on CredData validation.

### 6.2 Leakage
- **CredData**: no split exists. Splitting by repository is required (cbx: by repository; GitHub: "grouped by repo/value before splitting"). The same value or line can still repeat across repositories (copied fixtures, vendored code, forks); `snapshot.json` gives the URLs needed to check forks. The draft removes from training any candidate whose normalised line also occurs in a test repository. This reads test inputs, never test labels.
- **The issue-report set** is split at random per candidate (§2), so test candidates share their issue text with training candidates, and its test numbers (IssueGuard, cbx, ours) are optimistic. The draft drops from training every row whose text also appears in test.
- **CredSweeper's ML** was trained on a random 85% of CredData lines (§4), so its CredData number is in-sample.

### 6.3 Label noise
- The CredData labels are manual. No agreement figure was found (**not verified**).
- The issue-report labels have kappa 0.9545 on 1,000 candidates.
- CredData only labels lines that a rule flagged (§1.1). Candidates on unlabelled lines are unjudged, so counting them as negatives would mislabel real secrets that no CredSweeper rule caught. **[Inference]**
- The X label (example or test) is a judgement call; the draft counts it as negative, like CredData's README.

### 6.4 Licences
- **CredData**:
  - each file keeps its project's licence;
  - the current GitHub metadata says 307 of 337 repositories are permissive, but the licence at each pinned commit is what counts (`collect_licenses` copies it);
  - cbx trained only on 116 MIT, Apache-2.0 and BSD repositories and published their list with commits and licences.
  - **[Inference]** Training on permissive code and publishing the weights with an attribution file of the same shape is the practice to follow. Copyleft repositories are used for testing only and never for training.
- **The issue-report set**: CC BY 4.0, so attribution is required (DOI, paper). The underlying issue texts are GitHub user content (see §2).
- **cbx-17M** (Apache-2.0) as a baseline or teacher is unproblematic. **CredSweeper** (MIT) is only a baseline.

### 6.5 What the model card must disclose
- The training repositories with commit and licence, as a CSV, plus the attribution for the issue-report set.
- That CredData values are obfuscated, and what that means for value shape.
- How the real issue-report values were handled (replaced in training, never published).
- The exact test protocol and how it differs from the cbx card: a different repository set, our scorer, the CredData commit, the meta MD5.
- That CredSweeper's CredData number is in-sample.
- The candidate generator (secretgate commit, rule-file hash). The model judges candidates; it does not find secrets on its own.
- The known blind spots: word-based passwords, multi-line keys, non-English contexts, recall bounded by the generator.
- That no live validation is done.
- The threshold presets and their validation-set precision and recall.
- The synthetic share of the training data.

## 7. Downloads the experiment needs

| name | URL | size | licence | real credentials? |
|---|---|---|---|---|
| CredData repository (meta, scripts, `snapshot.json`) | https://github.com/Samsung/CredData (pin the full SHA of `01e3d135a9`) | meta 6.27 MB, plus scripts | Apache-2.0 (repository); each data file keeps its project's licence | meta: no |
| CredData source repositories via `download_data.py` | 337 github.com repositories at pinned commits (`snapshot.json`) | **not verified**; estimate about 1 GB kept, several GB transient | per repository; 307 of 337 permissive (current metadata) | **yes, in the transient clones** (original values before obfuscation) |
| Python 3.10 environment for `download_data.py` (GitPython, tabulate, pybase62, colorama), run in a Linux container | PyPI; Docker is installed | small (not verified) | various permissive | no |
| cbx `training_data_sources.csv` | https://huggingface.co/CobuildX-AI/cbx-secrets-detector-17M/resolve/main/training_data_sources.csv | 10,724 bytes | Apache-2.0 | no |
| cbx-secrets-detector-17M (`model.safetensors`, `tokenizer.json`, `cbx_secrets.py`, `secrets_config.json`) | https://huggingface.co/CobuildX-AI/cbx-secrets-detector-17M (pin a commit) | 67,220,156 + about 3.58 MB | Apache-2.0 | no |
| torch ≥ 2.1, transformers ≥ 5.0 (for cbx) | PyPI | **not verified** (on the order of a few hundred MB) | BSD-3 / Apache-2.0 | no |
| Issue-report set | https://zenodo.org/records/19622962 (`Secret-Leak-Detection-Issue-Report.zip`) | 134,502,037 bytes zipped, about 503 MB unzipped (sum of the listed CSVs) | CC BY 4.0 | **yes** (real values already exposed in public issues) |
| Gitleaks 8.30.1 | already installed | 0 | MIT | no |
| Betterleaks 1.10.1 | https://github.com/betterleaks/betterleaks/releases | 9.8 MB (darwin_arm64) | MIT | no |
| detect-secrets 1.5.0 | https://pypi.org/project/detect-secrets/ | 120 KB, plus pyyaml and requests | Apache-2.0 | no |
| CredSweeper 1.19.6 | https://pypi.org/project/credsweeper/ | wheel 10.8 MB, plus about 30 dependencies (onnxruntime, pandas…) | MIT | no |
| scikit-learn 1.9.1 and numpy, unless the simple baseline is written in Rust | PyPI | tens of MB (not verified) | BSD-3 | no |
| potion-base-2M and model2vec 0.10.0 (optional, reported only) | https://huggingface.co/minishlab/potion-base-2M | 7,559,256 bytes | MIT | no |
| TruffleHog 3.99.9 (not recommended) | https://github.com/trufflesecurity/trufflehog/releases | 75.9 MB | AGPL-3.0 | no |

Nothing else is needed: the tokenizer, the pretraining text and the synthetic generator (`crates/sky-data/src/secrets.rs`) are already in the repository.
