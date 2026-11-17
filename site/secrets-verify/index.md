---
title: Secrets verifier
outline: [2, 2]
---

# Secrets verifier: is this value a real credential?

::: danger Failed its pre-registered gate (G1): published as a result, not for use
**s6-secrets-verify-pico failed gate G1.** On the CredData test, 221 real repositories, its line F1 is **0.8561**, where the gate asked for the best rule scanner plus 0.15: **0.7332 + 0.15 = 0.8832** (Betterleaks). The six other gates held. The file is published so that the result can be checked and run (owner's decision of 15/11/2026); the verdict does not change. The system with secretgate's default rules and this model together was read too and failed the same gate (0.8477).
:::

**What it does well.** It is above every rule scanner tested on CredData (Betterleaks 0.7332, secretgate's default engine 0.7279, Gitleaks 0.7037), calibrated on real candidates (ECE 0.0073), and it accepts a candidate in 0.35% of 8,825 clean real files (31 files). Since version 1.9.0, [secretgate](https://github.com/maxgfr/secretgate) offers it as an experimental option, off by default: its rules decide first and their result never changes, and the verifier can only add redactions.

**s6-secrets-verify-pico** is a 250 KB ternary model (445,345 parameters). A rule engine proposes a candidate, a value with its name, its line, the lines around it and its file, and the model answers one question about that value: "Is this value a real credential that should be revoked?". It does not scan a text itself: it only judges what a rule proposes.

The demo below does not run secretgate's rule engine: the real generator is secretgate's, frozen as sg-cand-1, and this demo proposes with one simple rule so that the model's judgement can be seen in a browser.

<ClientOnly>
  <SecretsVerifyDemo />
</ClientOnly>

## What you are looking at

- **The proposer.** One rule, `generic-keyword`: a value of at least 8 characters assigned to a name that contains password, passwd, secret, token, api key, apikey, auth, credential or private key, in the forms `NAME=value`, `NAME: value`, `"name": "value"` and `name = "value"`. secretgate's engine has many more rules, and filters (entropy, placeholders, stopwords, allowlists) whose flags are part of what the model reads; this demo does not compute them and sends them as false.
- **P(real).** The model's probability that the value is a real credential, at full depth. The bar's dark mark is the threshold.
- **The thresholds.** Three are stored in the file, fixed on the 1,589 CredData validation candidates before the test was opened: τ_f1 = 0.2598 (F1-optimal, the one every test reading used, and the default here), τ_f05 = 0.468437 (fewer alarms) and τ_f2 = 0.0780795 (fewer misses). A candidate is accepted when P(real) is at or above the threshold.
- **The file name.** The model reads the scope words and the extension of the file's path (`test`, `src`, `conf`, `.py`…); change it to see what the path does.
- **The time.** Per candidate, from the record to its answer. Measured under Node on 200 test candidates, the median in WebAssembly is 2,666 µs; the first candidate of a page pays the start-up.
- **The text the model reads.** Each candidate's disclosure shows the state built from the record by the same code the training data was built with (`sky_infer::secret_cand`): rule, path, key, value, the hand features, the line with the value replaced by `<V>`, and the lines before and after.
- **The presets** were written for this page, before the model was run on them, and every value in them is fake, built from placeholder words. They are kept whatever the model answers, and it accepts some of them at τ_f1.

## What the result does not say

- **It is not a scanner.** It only judges what the rules propose. On the test repositories secretgate's generator overlaps 0.9260 of the gated positive rows, so its misses are mostly misses of the model.
- **Its weakest category is API keys** (line F1 0.541, where cbx-secrets-detector-17M is at 0.898), and on the issue reports cbx-17M is better (0.8616 against 0.8162).
- **On lines CredData never labelled** it accepts 99.7 candidates per 1,000 test files, against 27.0 for Gitleaks and 39.1 for Betterleaks.

Every number: [its section of Results](/results/#s6-secrets-verify-pico-failed). The model: [its card in the catalogue](/models/#s6-secrets-verify-pico) and [its model page on Hugging Face](https://huggingface.co/maxgfr/s6-secrets-verify-pico). The story of the run: [A 250 KB secrets verifier](/articles/a-250-kb-secrets-verifier). The pre-registration: [`docs/s6-secrets2-prereg.md`](https://github.com/maxgfr/skycmd/blob/main/docs/s6-secrets2-prereg.md). The engine: `SecretVerifier` in `crates/sky-web/src/verify.rs`.
