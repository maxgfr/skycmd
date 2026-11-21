---
title: "A 250 KB secrets verifier: rules propose, the model judges, and the gate still says no"
description: "A tiny model that judges the candidates of a rule-based secret scanner: a first version that passed on its own generated data and failed on real files, then a second, trained and tested on real repositories, that beats every rule scanner tested and still misses its pre-registered bar by 0.027."
date: 2026-11-16
outline: [2, 2]
---

# A 250 KB secrets verifier: rules propose, the model judges, and the gate still says no

*16 November 2026*

The tools that catch leaked credentials are mostly built on rules. I wanted to see whether a model of a quarter of a megabyte could do better. Three attempts followed. The first passed its test and failed on real files. The second beats every rule scanner I tested on real repositories and still misses the bar I had set. The third is the second wired the way it would ship, and it misses the same bar. What it taught is the point.

## The context: skycmd

skycmd gathers very small "decision models" with ternary weights (each is −1, 0 or +1), written in Rust, run natively or in the browser, and trained on one MacBook (Apple M5) in a braked mode that keeps it usable.

::: info Pre-registration and gates
Before an experiment runs, I commit the test, the bar to clear and what a failure would mean. The bar is the "gate": conditions that must all hold. The test is read once, and a failure is published like a success, so the rule cannot be adjusted after the results are seen.
:::

## The job: finding secrets before they leak

A secret, here, is a credential: an API key, a token, a password. A secret scanner reads text and flags values that look like one, so they can be hidden before the text is committed, logged or sent. Most scanners, such as Gitleaks, use rules: regular expressions for known key formats, plus checks on how random a value looks.

Rules miss generic secrets: a password under `db_password` has no format to match. And they raise false alarms. A false positive is a flag on something that is not a secret: a checksum, a test key, a placeholder like `EXAMPLE_KEY_0000`. A detector that flags half the files is useless: people stop reading its warnings.

Three measures come back below. Precision is the share of flags that are real secrets. Recall is the share of real secrets that get flagged. F1 combines the two into one number between 0 and 1, and it is high only when both are.

## v1: passed on its own data, failed on real files

The first model, [s6-secrets-pico](https://huggingface.co/maxgfr/s6-secrets-pico), has 445,345 parameters in about 250 KB. It read a whole short text, such as a `.env` file, and said whether a usable secret was exposed, which service issued it, what kind of value it was, and on which line.

All its data came from a generator I wrote: keys in each provider's public format with random bodies, placed in file templates. On 1,996 generated texts in four file types the training never showed, it beat a rule scanner on both questions the scanner answers: 0.9990 against 0.977 on "is a secret exposed", 0.9795 against 0.787 on the service. It passed its gate.

After publication, it was run on 719 real files in which rules find nothing, from [secretgate](https://github.com/maxgfr/secretgate) (a rule-based secrets firewall for coding agents that I also maintain), a sample of its dependencies, and this repository. At a probability of 0.9 or more, it flagged 59.8% of these files; at 0.999 or more, 49.4%. Every flag was a false positive, on ordinary imports, prose and JSON.

The gate stays passed, but the model cannot scan real files, and every page that reports it says so first. A bench you generate yourself measures your generator.

## v2: rules propose, the model judges

The second version, [s6-secrets-verify-pico](https://huggingface.co/maxgfr/s6-secrets-verify-pico), keeps the same size (445,345 parameters, 249,984 bytes) and changes the job.

A rule engine runs first: secretgate's engine with the filters that normally discard values turned off, plus a broad rule for any value assigned to a name containing a word like `password` or `token`. It proposes candidates. For each one, the model receives the value, its variable name, its line and the lines around it, the kind of file, and a few hand-computed features such as length and randomness. It answers one question: "Is this value a real credential that should be revoked?"

The model never searches. The cheap, deterministic part finds; the small learned part judges one value.

The split is not mine. Samsung's [CredSweeper](https://github.com/Samsung/CredSweeper) runs rules and then a machine-learning validator, whose model file (`ml_model.onnx`) is 11,545,911 bytes; SAP's [Credential Digger](https://github.com/SAP-archive/credential-digger) filters the hits of its regular expressions with machine-learning models; GitGuardian's [FP Remover](https://blog.gitguardian.com/fp-remover-cuts-false-positives-by-half/) removes false positives from what its generic detectors flag. What this version adds is the size, 249,984 bytes, and a comparison on 221 CredData repositories that it never trained on (*added on 21 November 2026*).

The data was real this time. The main source is CredData, Samsung's public benchmark of real repositories, where each value is labelled a real credential, an example or test value, or not a credential. It is used in its obfuscated form: every real secret replaced by random characters of the same shape. Training used 100 of its repositories and validation 16, each licence-checked (MIT, Apache-2.0 or BSD). Public issue reports containing leaked secrets, their training values replaced by look-alikes, and synthetic candidates make up the rest. The test: 221 other CredData repositories, never trained on, and the issue reports' test split, whose values are real.

Training was one run on the MacBook, from 19:39 to 20:42 on 14 November: 12 working minutes of pretraining on text, then 30 on the question itself (the braked mode's pauses not counted).

## The results on 221 real repositories

The unit is the line: a line is positive when CredData labels a real credential on it. A positive line with no candidate counts as a miss. Values of CredData's category `UUID` are left out for every system, a choice made after reading training and validation rows, before any test file; the full numbers are in the result files. The last column leaves out the largest test repository, for a reason given below.

| System | Size | Precision | Recall | Line F1, all 221 | Line F1, without the largest |
|---|---:|---:|---:|---:|---:|
| **s6-secrets-verify-pico (v2)** | 250 KB | 0.8387 | 0.8741 | **0.8561** | 0.8114 |
| v3: rules or the model (second look) | rules + 250 KB | 0.8039 | 0.8965 | 0.8477 | 0.8062 |
| logistic regression, same inputs | 256 KiB | 0.7345 | 0.8652 | 0.7945 | 0.7179 |
| Betterleaks 1.10.1, best rule scanner | rules | 0.7694 | 0.7002 | 0.7332 | 0.5367 |
| secretgate's default rules | rules | 0.8176 | 0.6560 | 0.7279 | 0.5072 |
| Gitleaks 8.30.1 | rules | 0.8546 | 0.5980 | 0.7037 | 0.4080 |
| cbx-secrets-detector-17M | 67 MB | 0.8265 | 0.5820 | 0.6830 | 0.8493 |

The model is above every rule scanner, on both readings, and above a logistic regression (a simple linear model) of the same size trained on the same candidates; had it matched the model, it should have shipped instead. cbx-secrets-detector-17M, an open 67 MB model rerun on the same test, is below on all 221 repositories and above without the largest. CredSweeper's machine-learning filter scores 0.9935, but it was trained on 85% of all CredData lines, so it enters no comparison.

Three other numbers matter:

- **Calibration.** A model is well calibrated when its probabilities mean what they say: of the values it gives 80%, about 80% are real. The calibration error (ECE), the average gap between stated confidence and actual accuracy, is 0.0073 on 33,809 test candidates; the logistic regression's is 0.0786.
- **False alarms on clean files.** On 8,825 real files with no known secret, the model accepts a candidate in 31 files, 0.35%, where the rule engine proposes candidates in 24.49%. v1 flagged 59.8% of a 719-file subset. Read one by one, none of the 110 accepted values looks like a real credential.
- **Speed.** A median of 1,606 µs per candidate natively on the M5, 2,666 µs in WebAssembly.

<figure class="article-figure">
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 400 425" width="400" height="425" role="img" aria-labelledby="fig1-title fig1-desc" fill="currentColor" font-size="14" style="display:block;width:100%;max-width:400px;height:auto;margin:0 auto">
<title id="fig1-title">Line F1 on the CredData test, all 221 repositories</title>
<desc id="fig1-desc">Horizontal bar chart, from 0 to 1, with a dashed vertical line at the gate's bar, 0.8832. s6-secrets-verify-pico, 250 KB: 0.8561. v3, rules or the model: 0.8477. Logistic regression, 256 KiB: 0.7945. Betterleaks: 0.7332. secretgate default rules: 0.7279. Gitleaks: 0.7037. cbx-secrets-detector-17M, 67 MB: 0.6830. No bar reaches the gate's line.</desc>
<g><title>s6-secrets-verify-pico (v2) · 250 KB: 0.8561</title><text x="0" y="39" font-weight="700">s6-secrets-verify-pico · 250 KB</text><path d="M0,46h270a4,4 0 0 1 4,4v8a4,4 0 0 1 -4,4h-270z" style="fill:var(--vp-c-brand-1,#0284c7)"/><text x="400" y="59" text-anchor="end" font-weight="700">0.8561</text></g>
<g><title>v3, rules or the model (second look): 0.8477</title><text x="0" y="93">v3: rules or the model</text><path d="M0,100h267.3a4,4 0 0 1 4,4v8a4,4 0 0 1 -4,4h-267.3z" fill-opacity=".6"/><text x="400" y="113" text-anchor="end">0.8477</text></g>
<g><title>logistic regression, same inputs · 256 KiB: 0.7945</title><text x="0" y="147">logistic regression · 256 KiB</text><path d="M0,154h250.2a4,4 0 0 1 4,4v8a4,4 0 0 1 -4,4h-250.2z" fill-opacity=".6"/><text x="400" y="167" text-anchor="end">0.7945</text></g>
<g><title>Betterleaks, best rule scanner: 0.7332</title><text x="0" y="201">Betterleaks, best rule scanner</text><path d="M0,208h230.6a4,4 0 0 1 4,4v8a4,4 0 0 1 -4,4h-230.6z" fill-opacity=".6"/><text x="400" y="221" text-anchor="end">0.7332</text></g>
<g><title>secretgate default rules: 0.7279</title><text x="0" y="255">secretgate default rules</text><path d="M0,262h228.9a4,4 0 0 1 4,4v8a4,4 0 0 1 -4,4h-228.9z" fill-opacity=".6"/><text x="400" y="275" text-anchor="end">0.7279</text></g>
<g><title>Gitleaks: 0.7037</title><text x="0" y="309">Gitleaks</text><path d="M0,316h221.2a4,4 0 0 1 4,4v8a4,4 0 0 1 -4,4h-221.2z" fill-opacity=".6"/><text x="400" y="329" text-anchor="end">0.7037</text></g>
<g><title>cbx-secrets-detector-17M · 67 MB: 0.6830</title><text x="0" y="363">cbx-secrets-detector-17M · 67 MB</text><path d="M0,370h214.6a4,4 0 0 1 4,4v8a4,4 0 0 1 -4,4h-214.6z" fill-opacity=".6"/><text x="400" y="383" text-anchor="end">0.6830</text></g>
<g><title>The gate's bar: Betterleaks 0.7332 + 0.15 = 0.8832</title><line x1="282.5" y1="22" x2="282.5" y2="394" stroke="currentColor" stroke-width="1.5" stroke-dasharray="5 4"/><text x="282.5" y="14" text-anchor="middle" font-size="13">gate: 0.7332 + 0.15 = 0.8832</text></g>
<g stroke="currentColor" stroke-opacity=".5"><line x1="0" y1="394.5" x2="321" y2="394.5"/><line x1="0.5" y1="394" x2="0.5" y2="399"/><line x1="80.5" y1="394" x2="80.5" y2="399"/><line x1="160.5" y1="394" x2="160.5" y2="399"/><line x1="240.5" y1="394" x2="240.5" y2="399"/><line x1="320.5" y1="394" x2="320.5" y2="399"/></g>
<g fill-opacity=".8" font-size="13" text-anchor="middle"><text x="0" y="417" text-anchor="start">0</text><text x="80" y="417">0.25</text><text x="160" y="417">0.5</text><text x="240" y="417">0.75</text><text x="320" y="417">1</text></g>
</svg>
<figcaption>Figure 1. Line F1 on the CredData test, all 221 repositories (the axis starts at zero). The model, in colour, is above every other system shown; the dashed line is the bar its gate asked for, 0.15 above the best rule scanner. No bar reaches it.</figcaption>
</figure>

## The gate: six conditions of seven

The gate had seven conditions, frozen before any test file was downloaded:

| Gate | Condition | Measured | Verdict |
|---|---|---|---|
| G1, all 221 | line F1 ≥ best rule scanner + 0.15 | 0.8561 against 0.7332 + 0.15 = 0.8832 | **fails** |
| G1, without the largest | same | 0.8114 against 0.5367 + 0.15 = 0.6867 | holds |
| G2 | within 0.10 of cbx-secrets-detector-17M | 0.8561 and 0.8114, against 0.6830 and 0.8493 | holds |
| G3 | above the logistic regression of the same size | 0.8561 and 0.8114, against 0.7945 and 0.7179 | holds |
| G4 | issue reports, real values: Gitleaks + 0.08 | 0.8162 against 0.7104 + 0.08 = 0.7904 | holds |
| G5 | calibration error ≤ 0.05 | 0.0073 | holds |
| G6 | ≤ 3 ms per candidate, natively | 1,606 µs | holds |
| G7 | clean real files flagged ≤ 1.0% | 0.35% (31 of 8,825) | holds |

The model needed 0.8832 and reached 0.8561: it missed by 0.027.

Why two readings? Before running anything, I counted the rows: one test repository holds 5,165 of the 10,463 positive rows, so a pooled score would mostly be that repository's. I amended the pre-registration to require G1 to G3 twice: on all 221 repositories, and on the 220 left without it. That could only make the gate harder.

It did. The rule scanners do well on that repository. Without it, Betterleaks falls from 0.7332 to 0.5367 and the model only to 0.8114, 0.27 above. With it, the margin is 0.123.

I did not move the bar. I had set +0.15 with one number in mind: the 0.442 that the card of cbx-secrets-detector-17M reports for Gitleaks, which would have put the bar near 0.59. On this test Gitleaks scores 0.7037, and the bar was 0.8832. A bar that moves when you miss it is not a bar. Both readings are reported side by side.

## v3: the system as it would ship, on a second look

In secretgate, the default rules always run, and a model may add findings, never remove one. So the system a user would get is the union: a line is flagged when the rules flag it or the model accepts a candidate on it.

On validation, the union beat the model alone. I wrote a new pre-registration, v3: no training, the same file, a threshold chosen on validation (0.34933), the same seven gates. It is a second look at test files already read once, with a system defined after that reading, and it is labelled as such. On validation, all seven gates held.

On the test, G1 failed again: 0.8477 against 0.8832. The union was slightly below the model alone: −0.0084, with a 95% interval of [−0.0181, +0.0072] from resampling the repositories. The rules added 305 positive lines and 784 negative ones. The six other gates held. The pre-registration said a new G1 failure ends the work, with no third reading: the work stops there, in this form.

## What exists anyway

The file is published as a failed gate, its numbers on the first lines of its model page.

::: warning What secretgate does with it
Since version 1.9.0, secretgate offers this file as an experimental option, off by default. Its rules decide first, and their result never changes. The verifier can only add redactions in tool output. secretgate's page says the model failed its gate. This repository holds no measurement of it in that use.
:::

## The limits

- On API values the model is weak: line F1 0.541, where cbx-secrets-detector-17M is at 0.898.
- On lines CredData never labelled, it accepts 99.7 candidates per 1,000 test files, against 27.0 for Gitleaks and 39.1 for Betterleaks; nobody knows how many are right.
- It only judges what the rules propose: on the test, the candidates reach 92.6% of the positive rows, and no judge recovers the rest.
- In CredData, a value's shape alone goes a long way to tell real from not real, an artefact of the obfuscation. The issue-report gate, on real values, guards against it, and it held.
- Before training, two rows of one test label file were printed by accident; nothing was chosen from them, and the pre-registration says so.
- One run, one seed. For v3, the 16 validation repositories did not predict the 221 of the test.

## What I take from it

**Test on real data from the start.** v1's generated bench could not see that it failed on real files. In v2, training and test came from real repositories, and clean real files became a gate (G7), cleared with room to spare: 0.35% against 1.0%.

**Give the small model one narrow question.** Searching text is a job for cheap, deterministic rules; judging one candidate in its context is a job a 250 KB model does better than those rules. The same split, a cheap algorithm computes and the model judges, is what the [tool router](/articles/a-3-7-mb-tool-router) was later built on.

**Fix the bar before looking, and keep it.** Missing by 0.027 is still missing. The second reading was added to make the gate harder, and it is the one that passed; reporting both is what gives either its value.

**A failed gate can still be a usable part.** Good calibration and few false alarms on clean files make the model reasonable to wire in where it cannot remove what the rules find, only add. That is a weaker claim than a pass, and the only one I make.

**All of it fits on a laptop.** The v2 run took about an hour on the MacBook, braked; the 67 MB baseline alone took 57,098 CPU seconds to evaluate.

## Check it

Everything is public: the pre-registrations of [v1](https://github.com/maxgfr/skycmd/blob/main/docs/s6-secrets-prereg.md), [v2](https://github.com/maxgfr/skycmd/blob/main/docs/s6-secrets2-prereg.md) (gates in §8, then the amendments and the result) and [v3](https://github.com/maxgfr/skycmd/blob/main/docs/s6-secrets3-prereg.md); the result files of [v2](https://github.com/maxgfr/skycmd/blob/main/runs/s6-secrets2/eval-test/results.md) and [v3](https://github.com/maxgfr/skycmd/blob/main/runs/s6-secrets3/eval-test/results.md); the [diagnosis](https://github.com/maxgfr/skycmd/blob/main/docs/s6-secrets3-diagnosis.md) that led to v3; the model pages of [v1](https://huggingface.co/maxgfr/s6-secrets-pico) and [v2](https://huggingface.co/maxgfr/s6-secrets-verify-pico); [secretgate](https://github.com/maxgfr/secretgate); [all the results](/results/) and [the code](https://github.com/maxgfr/skycmd).
