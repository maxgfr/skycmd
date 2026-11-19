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
  - skycmd
  - failed-gate
  - negative-result
---

# s6-secrets-verify-pico

> **Failed its pre-registered gate: published as a result, not for use.** It is not recommended as
> a standalone detector; secretgate ships it only as an experimental option, off by default (below).
> Gate G1
> ([`docs/s6-secrets2-prereg.md`](https://github.com/maxgfr/skycmd/blob/main/docs/s6-secrets2-prereg.md)
> §8) asked for a line F1 at least 0.15 above the best rule scanner on the CredData test, on every
> repository and without the largest one. On every repository the model reached **0.8561**,
> against **0.7332 + 0.15** for Betterleaks (+0.123): **it failed G1 on its every-repository
> reading.** The six other gates held. The file is published so that the result can be checked
> and run (owner's decision of 15/11/2026, amendment 9 of the pre-registration); the verdict does
> not change, and it is not continued in this form. Since version 1.9.0 (16/11/2026),
> [secretgate](https://github.com/maxgfr/secretgate) offers this file as an experimental option,
> off by default: its rules decide first and their result never changes, the verifier only adds
> redactions in tool output, and its page says that the model failed its gate. Numbers:
> [BENCH.md, section s6-secrets-verify-pico](https://github.com/maxgfr/skycmd/blob/main/BENCH.md#s6-secrets-verify-pico-s6-secrets-v2-a-verifier-of-rule-candidates-pre-registered-gate-failed)
> and [`runs/s6-secrets2/eval-test/results.md`](https://github.com/maxgfr/skycmd/blob/main/runs/s6-secrets2/eval-test/results.md).
>
> **The system with the rules was read too, and failed the same gate.** s6-secrets v3
> ([`docs/s6-secrets3-prereg.md`](https://github.com/maxgfr/skycmd/blob/main/docs/s6-secrets3-prereg.md))
> flags a line when secretgate's default rules flag it or when this model, unchanged, accepts a
> candidate on it (P(real) ≥ 0.34933). Read on the same seven gates, on a second look at the same
> test files with the system defined after the first reading, it reached **0.8477** against
> **0.7332 + 0.15**: G1 failed again; the six other gates held. There is no third reading, and the
> work on this model ends
> ([`runs/s6-secrets3/eval-test/results.md`](https://github.com/maxgfr/skycmd/blob/main/runs/s6-secrets3/eval-test/results.md)).

**A 250 KB ternary verifier of secret candidates.** 445,345 parameters. A rule engine proposes a
candidate (a value, its line, its neighbours, its file), and the model judges that one value in
its context: "Is this value a real credential that should be revoked?". It is the second version
of [s6-secrets-pico](https://huggingface.co/{{USER}}/s6-secrets-pico), which read whole texts and
failed on real files; this one was trained and tested on real repositories. It is the project's
best result on real data, and it is a failed gate.

- **Demo:** on the site at [/secrets-verify/](https://maxgfr.github.io/skycmd/secrets-verify/) and as
  the Space [{{USER}}/s6-secrets-verify-pico](https://huggingface.co/spaces/{{USER}}/s6-secrets-verify-pico)
  (every published model gets a Space, owner's decision of 17/11/2026). It proposes candidates with a
  simplified one-rule proposer, not secretgate's engine, and sends the filter flags as false, so it
  shows the model's judgement on those records, not the measured system.
- **Code:** [github.com/maxgfr/skycmd](https://github.com/maxgfr/skycmd) (100 % Rust)
- **Pre-registration:** [docs/s6-secrets2-prereg.md](https://github.com/maxgfr/skycmd/blob/main/docs/s6-secrets2-prereg.md)
  (the gates in §8, amendments 1 to 8 before the run, the result block, amendment 9)
- **All results, failures included:** [results page](https://maxgfr.github.io/skycmd/results/)

## The seven gates, as read

| gate | condition | measured | verdict |
|---|---|---|---|
| G1, every repository | line F1 ≥ best rule scanner + 0.15 | 0.8561 against Betterleaks 0.7332 + 0.15 = 0.8832 | **fails** |
| G1, without the largest repository | same | 0.8114 against Betterleaks 0.5367 + 0.15 = 0.6867 | holds |
| G2, both readings | line F1 ≥ cbx-17M − 0.10 | 0.8561 against 0.6830 − 0.1; 0.8114 against 0.8493 − 0.1 | holds |
| G3, both readings | line F1 > size-matched logistic regression | 0.8561 against 0.7945; 0.8114 against 0.7179 | holds |
| G4 | issue-report candidate F1 ≥ Gitleaks + 0.08 | 0.8162 against 0.7104 + 0.08 = 0.7904 | holds |
| G5 | ECE ≤ 0.05 on CredData test candidates | 0.0073 | holds |
| G6 | native median ≤ 3 ms per candidate | 1,606 µs | holds |
| G7 | clean real files flagged ≤ 1.0% | 0.35% (31 of 8,825) | holds |

The largest test repository holds 5,165 of the 10,463 gated positive rows, and the rule scanners
do well on it. The paired bootstrap over repositories (1,000 resamples, 95%) gives the model minus
Betterleaks +0.1229 [0.0413, 0.3613] on every repository and +0.2748 [0.1743, 0.3796] without the
largest. The bar of +0.15 was fixed before any measurement and was not moved.

## Results

CredData test, 221 real repositories, line level, gated universe (the rows of category `UUID`
removed, a choice made after reading training and validation rows and before any test file was
read; full-universe numbers in BENCH.md):

| system | precision | recall | **line F1** | line F1 without the largest |
|---|---:|---:|---:|---:|
| **s6-secrets-verify-pico** | 0.8387 | 0.8741 | **0.8561** | 0.8114 |
| Betterleaks 1.10.1 | 0.7694 | 0.7002 | 0.7332 | 0.5367 |
| secretgate default engine | 0.8176 | 0.6560 | 0.7279 | 0.5072 |
| Gitleaks 8.30.1 | 0.8546 | 0.5980 | 0.7037 | 0.4080 |
| cbx-secrets-detector-17M | 0.8265 | 0.5820 | 0.6830 | 0.8493 |
| size-matched logistic regression (256 KiB) | 0.7345 | 0.8652 | 0.7945 | 0.7179 |

- **Candidate level** (33,809 candidates, 9,623 positive): F1 0.8827, PR-AUC 0.9335, ECE 0.0073.
- **Issue reports** (8,123 candidates, 888 secrets, values as published): F1 0.8162, against
  0.8616 for cbx-secrets-detector-17M and 0.7104 for Gitleaks.
- **Clean real text** (8,825 files of secretgate, its dependency tree and this repository): an
  accepted candidate in 31 files (0.35%), 110 candidates. Read one by one after the gate: none of
  the 110 looks like a real credential (test fixtures, documentation examples, provider-format
  values on a rule's allowlist, code, public hashes). v1 flagged 59.8% of
  a 719-file subset.

## What it does well, and badly

- **Well:** better than every rule scanner tested on CredData, with or without the largest
  repository; calibrated on real candidates (ECE 0.0073); few alarms on clean real files (0.35%).
- **Badly:** not 0.15 above the rules on every repository, which is what its gate asked. Its
  weakest category is API keys (line F1 0.541, where cbx-17M is at 0.898); on the issue reports
  cbx-17M is better (0.8616 against 0.8162).
- **Most of its API misses are not its own.** A later diagnosis
  ([`docs/s6-secrets4-diagnosis.md`](https://github.com/maxgfr/skycmd/blob/main/docs/s6-secrets4-diagnosis.md),
  18/11/2026, no model run) found that most missed API lines get no candidate from the rules, so
  no verifier can recover them; no number on this card changes.
- **It only judges what the rules propose.** On the test repositories the generator overlaps
  0.9260 of the gated positive rows: its misses are misses of the model.
- **On unlabelled lines** (lines CredData never judged) it accepts 99.7 candidates per 1,000 test
  files, against 27.0 for Gitleaks and 39.1 for Betterleaks.

## How to ask it

The state is built by `sky_infer::secret_cand::state` from a candidate of secretgate's engine in
its candidate mode (`sg-cand-1`, frozen at secretgate commit `4e93cfa`): one line per part, in
this order, `rule`, `path` (scope words and extension), `key`, `value`, the hand features as
`name value` pairs, `line`, `prev`, `next`. `crates/sky-infer/tests/fixtures/secret_cand.json`
holds inputs and expected states that any other implementation must reproduce. The questions:

- `noul`: "Is this value a real credential that should be revoked?", options
  `false: no, the statement does not hold` and `true: yes, the statement holds` (the gated one);
- `choice`: "What kind of value is this?", options `a real credential`,
  `an example, test value or placeholder`, `not a credential` (reported only; its test accuracy is
  not in the results files).

The file stores three operating thresholds on P(true) of the first question, fixed on the 1,589
CredData validation candidates before the test was opened: `skycmd-s2.secrets2.tau_f1` 0.2598
(F1-optimal, the one every test reading uses), `tau_f05` 0.468437 and `tau_f2` 0.0780795.

## Full depth

The file answers at full depth: its three exit thresholds are set to 1, the copy every reading was
made on (sha256 `01e6db3fa71cc2c5563dea222c68e040d3994ee2abbb02b25491a51b8cb54f27`). Native median
1,606 µs per candidate at full depth (372 µs at the first exit); in WASM under Node, median
2,666 µs on 200 test candidates, within 4.17e-07 of the native probabilities, the same answer on
200 of 200. Accuracy on the real question by exit: 0.9461, 0.9580, 0.9600, 0.9614.

## Training

From scratch in Rust (mlx-rs) on an Apple M5, one run, seed 0, in a throttled mode, so the
budgets are working minutes. The s6-secrets-pico graph (1 state block × 4 loops, an exit after
each loop, d 96, vocabulary 1,024, state window 320, head window 192). Masked-language pretraining
starting from s6-secrets-pico's pretraining stack (12 minutes, 11,296 steps) on the project's
pretraining text, a numeric curriculum and the obfuscated files of the CredData training
repositories (30% of the text), then decision training with ternary QAT
over the last 90 % (30 minutes, 14,117 steps) on three sources drawn 0.45 / 0.35 / 0.20:

- **CredData** (Samsung, pinned at `01e3d135`): 100 training and 16 validation repositories of
  the cbx-secrets-detector-17M list, every one license-checked (MIT 72, Apache-2.0 34,
  BSD-3-Clause 7, BSD-2-Clause 3), in CredData's obfuscated form, plus weak negatives drawn from
  their unlabelled lines. The 221 test repositories were never trained on, and training lines
  that also occur in a test repository were removed.
- **Issue reports** (Zenodo 10.5281/zenodo.19622962, CC BY 4.0): the paper's training split, each
  real secret replaced by a word-preserving shape.
- **Synthetic candidates** from the project's generator (Apache-2.0): provider formats with random
  bodies; nothing is real.

CredData is read in its obfuscated form only, and the real secrets of the issue reports'
training split are replaced; the issue reports' test values are real and already public, and are
only scored. Licenses:
[DATA_LICENSES.md](https://github.com/maxgfr/skycmd/blob/main/DATA_LICENSES.md).

## Running it outside the browser (`engine/`)

`engine/` holds the engine that runs this file from JavaScript (Node 21+, Deno, browsers): `sky_web_bg.wasm` (SIMD128) and `sky_web_relaxed_bg.wasm` (relaxed SIMD), the glue `sky_web.js` or the glue-free host `verify-host.mjs`, the word list `dictionary.txt` and this model, with bytes, sha256 and source commit in `MANIFEST.json`; `loadVerifier(wasm, gguf, dictionary)` (or `new SecretVerifier(gguf, dictionary)` with the glue) gives `judge(candidate)` → `{p_real, state_tokens, truncated, micros}` at full depth, `judgeMany([...])`, `thresholds()` → `{tau_f1, tau_f05, tau_f2}` (real when `p_real >= tau`) and `state(candidate)`, the text the model reads.
A candidate is a secretgate `sg-cand-1` record as JSON: `rule` and `value` required; `file` (a path relative to the repository root) or `scope` (CredData's path words) with `ext`; `name`, `lineText` with `start`, `end` and `lineFrom` in UTF-16 units, `line`, `endLine`, `prevLine`, `nextLine`, and the booleans `filters.entropy`, `placeholder`, `stopword`, `allowlist`; `null` is absent, other fields are ignored, a wrong type is refused (`crates/sky-web/src/verify.rs`).

## Files

- `s6-secrets-verify-pico.gguf`: ternary weights, tokenizer, temperatures, exit thresholds set to
  1 (full depth) and the three operating thresholds. 249,984 bytes.
