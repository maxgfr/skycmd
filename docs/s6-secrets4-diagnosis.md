# s6-secrets-verify-pico: diagnosis of the API category, for a possible v4

*Status on 19/11/2026: closed on 18/11/2026, by the lead's decision on this diagnosis: no v4 of s6-secrets-verify-pico is pre-registered; see section 5.*

Date: 17-18/11/2026. Read-only analysis. No model was run, nothing was trained, no scanner was rerun. The test was read once for v2 and once more for v3. This diagnosis tabulates the published test outputs into aggregates. To do that it reads the test meta rows again, and the text of the test files that hold a T row on an API line, through the scorer's own loader. That makes it a further look at the test. **Nothing below may be used to choose anything that is later read on the same test files.** This report holds counts, rates, lengths, entropies, class counts, rule ids, extensions, category names and keyword ids from a fixed list. It holds no value, no source line and no variable name.

## 1. Summary

- On the CredData test (gated universe), the API category has 209 positive lines. The model finds 98 of them, misses 111 and flags 55 negative API lines: recall 0.469, precision 0.640, F1 0.541. **The loss is mostly recall, and mostly before the verifier.**
- **(1) No candidate: 102 of the 111 missed lines.** The generator proposes nothing on them, so no verifier can recover them. 99 of the 102 sit in one test repository (`4a099ada`). There the labelled value is a quoted string that itself holds escaped quotes (a serialized JSON fragment). `generic-keyword` stops reading at the first inner quote, after 2 characters, which is under its 4-character minimum, and no other rule fires. Generator recall on API is 41/41 in training, 7/7 in validation, 107/209 on the test, and 106/109 on the test outside `4a099ada`.
- **(2) A candidate scored too low: 8 missed lines** (median P(real) 0.086), plus 1 line where a candidate exists but overlaps no T value. The other side is the 55 false-positive lines. The model accepts API T candidates as often as T candidates elsewhere (0.925 against 0.935).
- **(3) A scoring artefact: none.** The line unit ignores columns, every missed row is a single line, and no flag lies within 2 lines of any miss.
- **Share of the G1 gap (0.0271).** Making every API line that has a candidate right lifts line F1 from 0.8561 to 0.8588 (+0.0027, 10% of the gap). Adding every generator miss, with perfect acceptance, gives 0.8643 (+0.0083, 31%). The bar is 0.8832. **The API category cannot close G1, and the verifier can reach only a tenth of the gap through it.** On the same computation, removing the false positives of the Key category alone gives 0.9005.

## 2. The numbers

**Method.** The line-level classification reimplements `scorer.score_system` unit by unit for the model (published flags, τ\* = 0.2598). It reproduces `scores.json` exactly: gated TP 9,001, FP 1,731, FN 1,296, F1 0.8561; API TP 98, FP 55, FN 111, TN 1,932.

The classes are those of `docs/s6-secrets3-diagnosis.md` §1:

- **miss**: no generator candidate covers the line;
- **no T**: candidates on the line, none overlapping a T value;
- **rejected**: a T candidate scored below τ\*.

**Sources:**

- **Published test outputs** (data folder, read-only): `eval/test/scores.json`; `eval/test/flags/{generator,model,cbx-17m,betterleaks}.flags.jsonl` and their `.run.json`; `eval/test/flags/raw/stats.json`; `eval/test/flags/raw/dropped.jsonl` (places, rules and gates only); the `features` field of `eval/test/states/creddata.jsonl`.
- **Test meta and text**: the test meta rows, and the text of the test files that hold a row of a line unit, read through `common.open_split("test")` and `scorer.load_labelled`.
- **Validation**: the same files under `eval/validation/`.
- **Training**: `gen/recall.json`; `build/counts/creddata.json`; the training meta through the same loader; the line positions of `gen/candidates-{training,validation}.raw.jsonl`.
- **Committed**: `runs/s6-secrets2/eval-test/results.md` and `BENCH.md`.

### 2.1 (a) Recall or precision, and where (test, published outputs)

| API lines, test, gated | count |
|---|---:|
| positive lines | 209 |
| TP | 98 |
| FN, generator miss | **102** |
| FN, candidates but none on the T value | 1 |
| FN, T candidate rejected | 8 |
| FP (negative lines flagged) | 55 (50 with an F row, 5 with an X row) |
| TN | 1,932 |

**Generator recall on API lines** (share of positive API lines with a candidate on them):

| split | positive API lines | with a candidate | source |
|---|---:|---:|---|
| training | 41 | 41 | `gen/recall.json` (rows) and training meta (line units) |
| validation | 7 | 7 | same |
| test, all 221 repositories | 209 | 107 (0.512) | `generator.flags.jsonl`, published |
| test, without `4a099ada` | 109 | 106 (0.972) | same |

Validation cannot speak for this category: its 7 positive API lines are all in one repository (`a09d9e50`), all in `.js` or `.ndjson` files, 6 of the 7 with values of 9 to 16 characters.

**The 102 misses (test).** Measured on the T rows, in aggregate:

| property | count of 102 |
|---|---:|
| in repository `4a099ada` | 99 (that repository has 100 positive API lines) |
| category set `API:Token` | 99 |
| file extension `.json` / `.html` / other | 61 / 37 / 4 |
| keywords before the value on its line (fixed list): `api` and `token` | 99 |
| a listed separator between the name and the value | 101 |
| value opened by a double quote, closed by the same quote right after it | 101 |
| value holds a backslash and its own opening quote | 99 |
| ... of which: exactly 2 characters before the first inner quote, 4 escaped-quoted runs inside, the longest of 32 or more characters, no credential keyword inside | 99 |
| value length 33-64 / 65-256 / shorter | 77 / 23 / 2 (76 are exactly 64 characters) |
| distinct values | 102 |
| value holds whitespace | 23 |
| a Python copy of the `generic-keyword` regex, anchored at the name, matches | 0 (it matches 100 of the 101 TP T rows) |
| a match dropped by a kept gate on the line (`dropped.jsonl`) | 1 (`credit-card-number`, gate `post`) |
| a generator flag within 2 lines | 0 |
| multi-line row | 0 |

An invented example of the form, made only of the words EXAMPLE and FAKE and digits (not taken from any file):

```
"EXAMPLE_API_TOKEN": "{\"FAKE1\": \"EXAMPLE0123456789EXAMPLE0123456789\", \"FAKE2\": \"FAKE\"}"
```

CredData labels the whole outer string as the value. `generic-keyword` reads a quoted value "to the same quote", so its value here would be the 2 characters before the first inner quote, under its minimum of 4. A bare value cannot begin with a quote. The inner names hold no listed keyword. No match was dropped by a kept gate, so the regexes did not match at all. That last step is inferred from the rule definitions and the dropped-match log: the 222 gitleaks rules were not traced one by one.

The form is absent from training. T rows whose value holds two or more escaped quotes number 4 in training (none of them API, all 4 on a line with a candidate), 0 in validation, and 100 on the test (99 API, 0 on a line with a candidate).

**Verdicts on (a):**

- (1) generator ceiling: **confirmed, 102 of 111 FN**, concentrated in one repository and one value form;
- (2) verifier: **minor on recall** (8 rejected, 1 no-T line), and it carries the 55 FP lines;
- (3) scoring artefact: **rejected**. The scorer counts a line as flagged when any accepted flag covers its LineStart, whatever the columns, so span differences cost nothing at line level. All 102 missed rows are single-line, and none has a flag within 2 lines. The published alternative reading "flag on any line of the row" moves the overall F1 from 0.8561 to 0.8565 (`scores.json`, `other_readings`).

### 2.2 (b) What sets this category apart

**How the category is defined.** CredData's `Category` column names the CredSweeper rules that matched the value (CredData README, "Category"). `API` is CredSweeper's keyword rule `api(?!tal)` (`credsweeper/rules/config.yaml` of the pinned 1.19.6 environment: type keyword, `GeneralKeyword` filter, minimum line length 11). So API means "a name containing `api`", not a key format. It is often compound: `API:Key`, `API:Token`, `API:Password`.

The scorer maps each colon-separated name to one of the six categories or to `other`, and gives a line unit the union of the categories of all its rows (T, F and X). A provider rule such as `MailGun API Key` counts as `other`, not API.

**Share of training against share of test** (gated line units, the scorer's definition):

| split | line units | API units | positive units | API positive units | API share of positives | positive rate inside API |
|---|---:|---:|---:|---:|---:|---:|
| training | 19,325 | 1,918 | 2,283 | 41 (15 repositories, largest 8) | 1.8% | 2.1% |
| validation | 2,420 | 156 | 457 | 7 (1 repository) | 1.5% | 4.5% |
| test | 39,196 | 2,196 | 10,297 | 209 (29 repositories, largest 100) | 2.0% | 9.5% (5.3% without `4a099ada`) |

The share of positives is about the same in training and test. What differs is one test repository, and a value form that training never shows.

**The T candidates the verifier sees** (best T candidate of each positive line; test, published flags and states):

| | API lines | other lines |
|---|---:|---:|
| T candidates | 106 | 9,463 |
| accepted at τ\* | 98 (0.925) | 8,850 (0.935) |
| median P(real) | 0.950 | 0.940 |
| median value length | 26 | 48 |
| median entropy (bits per character) | 3.84 | 3.80 |
| median upper / lower / digits / other | 0 / 15 / 7 / 0 | 0 / 17 / 20 / 0 |
| quoted | 56 | 3,337 |
| top rules | `generic-api-key` 51, `password-assignment` 40, `generic-keyword` 11 | `generic-api-key` 5,313, `generic-keyword` 1,647, `password-assignment` 1,131, `private-key` 1,076 |
| value longer than 64 (elided in the state), decisive candidate of TP lines | 2.0% | 18.6% |

On the candidates it gets, the verifier is not worse on API than elsewhere.

**False positives.** The 55 FP lines are concentrated:

- 16 of them in one repository (`5cecf769`, all `.mlir`) and 9 in another (`28728ab4`);
- 42 are on lines that hold a whole-line F row (CredData marks the whole line "not a credential");
- their decisive candidates are short and low-entropy (median length 9, entropy 2.58), and 16 carry the stopword flag;
- the decisive rule is `generic-keyword` on 35 and `generic-api-key` on 18.

The false-alarm rate on negative API lines is 2.8% (55 of 1,987), against 6.0% over all negative lines (1,731 of 28,899). The low precision comes from how few positives the category has once the 99 misses are out, not from a higher false-alarm rate.

**Is the information in the verifier's input?** The state is built by `sky_infer::secret_cand::state` (`crates/sky-infer/src/secret_cand.rs`). `SecretVerifier::state` in `crates/sky-web/src/verify.rs` and the training build (`record_state` in `crates/sky-data/src/secrets2/mod.rs`) both call it, so serving and training build the same state. It holds, in this order:

- the rule id;
- the scope words and the extension;
- the name, up to 40 characters, with its keywords readable;
- the value, whole up to 64 characters, else its first and last 32 around an elision mark;
- the hand features: length, entropy, characters per token, digit, upper, lower and other counts, the longest dictionary word run, and four filter flags;
- a 120-character window of the line;
- 100 characters of the previous and of the next line.

The state is truncated from its end at 320 tokens. For 102 of the 111 missed lines there is no candidate, so no state exists: the question of the input does not arise. If the generator proposed the whole outer string, 79 of these values would fit in the 64-character value field and 23 would be elided. Per-candidate truncation on the test is not in the published files (§4). The proxy: no API decisive candidate has a state text of 600 characters or more. The dictionary feature reads 0 for the median API candidate, of every class.

### 2.3 (c) What cbx-secrets-detector-17M does differently

From `scripts/secrets2/cbx_infer.py`, `docs/s6-secrets2-feasibility.md` §1.4 and its tool table, and the published flags:

- **It has no candidate generator.** It reads each file as one text (`SecretDetector().extract(text)`) with a `max_length` of 7,999 tokens; longer texts are cut with a 64-token overlap.
- **It tags tokens** with 12 BIO labels and keeps spans of at least 10 characters (4 for passwords).
- **No name, separator or quoting pattern** is needed for it to flag a span inside a JSON string.
- **Its training data**: the 116 training and validation repositories of this project, and the issue-report training split.

API lines, test, line level (published flags; `4a099ada` against the other 28 repositories with API lines):

| system | in `4a099ada`: TP / FP / FN | F1 | elsewhere: TP / FP / FN | F1 |
|---|---|---:|---|---:|
| model | 1 / 1 / 99 | 0.020 | 97 / 54 / 12 | 0.746 |
| cbx-17M | 79 / 0 / 21 | 0.883 | 92 / 1 / 17 | 0.911 |
| Betterleaks | 0 / 0 / 100 | 0 | 64 / 6 / 45 | 0.715 |

cbx flags 78 of the 102 lines the generator misses. Outside that repository, the model's recall on API is higher than cbx's (0.890 against 0.844). Its precision is much lower (0.642 against 0.989).

The difference is granularity. cbx judges spans in the whole text. The model judges only what the rules propose, and its API false positives are rule candidates on lines that CredData marks as not credentials. The same granularity costs cbx the Key category (line F1 0.266, `BENCH.md`).

### 2.4 (d) How much of the G1 gap this category accounts for

Line F1 on the test, gated, with one category's errors removed and everything else unchanged (published per-line flags; a line in several categories is fixed by each). The bar is Betterleaks 0.7332 + 0.15 = 0.8832, the gap 0.0271.

| what is fixed | every repository | gain | share of the gap | without `2ba83c6a` (the largest) |
|---|---:|---:|---:|---:|
| nothing (the published model) | 0.8561 | | | 0.8114 |
| API: FP lines only | 0.8583 | +0.0022 | 8% | 0.8157 |
| API: FN lines that have a candidate only | 0.8565 | +0.0005 | 2% | 0.8125 |
| **API: every line with a candidate** | **0.8588** | **+0.0027** | **10%** | 0.8168 |
| API: the 99 misses of `4a099ada` only, all accepted | 0.8614 | +0.0053 | 20% | 0.8231 |
| **API: everything, misses included** | **0.8643** | **+0.0083** | **31%** | 0.8288 |
| for comparison, Key: FP lines only | 0.9005 | +0.0444 | 164% | 0.8277 |
| Key: every line with a candidate | 0.9081 | +0.0520 | | 0.8362 |
| Password: FP lines only | 0.8730 | +0.0169 | 62% | 0.8436 |
| `other`: everything, misses included | 0.8915 | +0.0355 | | 0.8839 |

The reading without the largest repository already passed G1 (0.8114 against 0.6867). The failing reading is the one over every repository. There a perfect API category leaves the model 0.019 under the bar. Most of what separates the model from the bar sits in the false-positive lines of Key and Password: 1,037 and 407 lines (`scores.json`). The Key part is largely in the largest repository: +0.0444 over every repository, +0.0163 without it.

## 3. What a next version could change

Ordered by evidence, then by who must change. Each estimate is an upper bound on the 0.0271 gap, computed on the test as in §2.4. It is not a prediction, and a change designed from these test aggregates cannot be read on the same test again (§3 of `docs/s6-secrets3-prereg.md` already excluded a third reading).

1. **The generator, in secretgate: read a quoted value with backslash escapes as one string**, or propose the escaped-quoted runs inside it.
   - **Evidence:** 99 of the 102 API misses; 100 test T rows of this form, none covered.
   - **Cheapest check:** run the changed generator (no model) on the training and validation repositories and on the clean-text corpus. Count the new candidates on labelled lines by label (T, F, X from meta), the new unlabelled candidates per 1,000 files, and the new files flagged in the clean corpus (G7). Validation has no row of this form, so **the recall gain cannot be confirmed on validation at all**. Only its cost can be measured there.
   - **Recovery:** at most +0.0053, if the verifier accepted all 99 and added no false positive. Probably less. The verifier has never been trained on such a value (4 training rows), and 23 of the values would reach it elided.
2. **The verifier's training data: API precision on lines CredData marks whole as not credentials.**
   - **Evidence:** 42 of the 55 FP lines hold a whole-line F row; the values are short and low-entropy; 25 of the 55 are in two repositories.
   - **Cheapest check:** none that validation can carry. It has 5 FP API lines, all in one repository. The training side can be counted: how many whole-line F rows on API lines the training items keep, and with which rules.
   - **Recovery:** at most +0.0022.
3. **The verifier itself, on the 8 rejected T lines** (7 of them `generic-api-key`, median P(real) 0.086).
   - **Cheapest check:** the same rows on validation, where API has no rejected line.
   - **Recovery:** at most +0.0005. Not worth a version.
4. **The scoring: nothing to change.** No artefact was found. The line unit already ignores columns and the multi-line reading changes nothing here. Recovery 0.

**Plainly:** most of the API loss cannot be fixed by the verifier. 102 of the 111 missed lines never reach it. With items 1 to 4 all at their ceilings, the model would stand at 0.8643, 0.019 under the bar. A version aimed at the API category is not worth pre-registering for G1. A next version aimed at G1 would have to start from the false positives of the Key and Password categories, diagnosed on training and validation first, and it would need test repositories that none of v2, v3 or this diagnosis has read.

## 4. Not verified

- **Rule by rule, why no rule of the 222 gitleaks rules or of the built-in ones matched the 102 missed lines.** It is inferred from the rule shapes, a Python copy of `generic-keyword` and the dropped-match log. The engine itself was not run.
- **What the true credential inside the serialized strings is,** and whether CredData's value bounds are capped: 76 of the 102 are exactly 64 characters.
- **Whether the verifier would accept such a value** if the generator proposed it. That would need a model run on candidates that do not exist yet.
- **Per-candidate truncation at 320 tokens on the test:** the published table has no token counts, and the state's character length is only a proxy.
- **cbx's behaviour per span:** cbx was not run. §2.3 rests on its documentation, its runner and its published flags.
- **The causes of the Key and Password false positives,** named in §2.4 and §3 as the larger part of the gap and not diagnosed here.
- **Bootstrap intervals** for the bounds of §2.4: not computed. The API numbers rest on 209 positive lines, 100 of them in one repository.
- **Any validation reading for API:** its 7 positive lines in one repository support no conclusion on this category.

The scripts that produced these numbers were throwaway and printed aggregates only. They were deleted, and none is committed. The method is the one of `docs/s6-secrets3-diagnosis.md` §1, applied to the published test flags, and it reproduces `scores.json` exactly (see §2).

## 5. Decision and open items (note of 19/11/2026)

Moved here on 19/11/2026 from the project's former to-do file (`TODO.md`, removed that day; last
version at commit `0e45b7b`), where they were recorded on 18/11/2026.

- **Closed: not worth pre-registering** (the lead's decision of 18/11/2026, on this diagnosis). No
  version of s6-secrets-verify-pico aimed at the API category is pre-registered. This closes the
  model's part of the quality round of 17/11/2026 (one targeted defect per model).
- **Open, not scheduled** (no pre-registration, no date):
  1. **In secretgate's candidate generator, read a quoted value with backslash escapes as one
     string** (§3, item 1). The gain is at most +0.0053 on line F1, and it cannot be checked on
     validation, which holds no value of that form. The generator is frozen as `sg-cand-1`
     (`docs/s6-secrets2-prereg.md` §2), so this is a change of that contract: **changing
     `sg-cand-1` is the owner's call.**
  2. **A verifier version aimed at the false positives of the Key and Password categories**
     (§2.4 and §3). It would need test repositories that none of v2, v3 or this diagnosis has
     read.
