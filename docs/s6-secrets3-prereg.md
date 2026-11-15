# s6-secrets v3: the system that would ship (rules and verifier together): pre-registration

**Status: frozen on 15/11/2026, before anything of it is computed on a test file.** This is a second look at test files that were read once for v2 (`docs/s6-secrets2-prereg.md`, "Result (15/11/2026)"), and it is designed after that reading: every place that reports v3 says so. No model is trained. Nothing here changes the reading of v2, which failed its gate on G1 and stays failed.

## 1. Why, and what changes

v2 judged the verifier alone: a line was flagged when the model accepted a candidate on it, even on lines secretgate's default rules already flag. That is not how it would run. In secretgate the default rule engine always runs, and the integration rule is that a model may add findings and never remove one. The system a user would get is the union.

The diagnosis of 15/11 (`docs/s6-secrets3-diagnosis.md`, made on the training and validation repositories only, without opening a test file) found that on validation the model rejects 24 of the 165 positive lines the default engine already flags; at the stored threshold the union is at 0.8834 where the model alone is at 0.8603 (line F1, gated universe), with a bootstrap interval that includes zero, [-0.006, +0.089]; and that on the issue-report validation split the union is lower than the model alone, 0.7793 against 0.8154. It also found no sign of under-training (validation flat over the last 4,000 steps, no gap between the ternary and the float model), so nothing is retrained.

**The system `s6-secrets v3`:** a line is flagged when secretgate's default engine (at the frozen commit of v2) flags it, or when the model of v2 (`s6-secrets-verify-pico`, the evaluated file, unchanged) accepts a candidate of the frozen generator on it, P(real) >= tau3. On the issue-report sets, a dataset candidate is flagged when a finding of the default engine overlaps it or the model accepts it.

**tau3** is chosen once, on the 16 validation repositories: the threshold that maximises the line-level F1 of the union on the gated universe, among the probabilities observed on the validation candidates; of equal values the highest is kept. It is written to `runs/s6-secrets3/tau3.json` before any test file is read for v3.

## 2. What does not change

The generator, the model file, the data, the splits, the universes (rows of category exactly `UUID` leave the gated one; the full one is reported next to it), the scorer and every reading fixed by the amendments of `docs/s6-secrets2-prereg.md`, the baselines and their outputs on the test files (they are not rerun: their flags are those of the v2 evaluation, whose hashes are recorded).

## 3. Gate: the seven gates of v2, read on the union

- **G1:** line F1 of the union >= the best rule scanner's + 0.15, on every test repository and on the test without its largest repository, both.
- **G2:** line F1 of the union >= cbx-17M's - 0.10, both readings.
- **G3:** line F1 of the union > the line F1 of the same union with the size-matched logistic regression in the model's seat (default engine or the regression at its own threshold, which is chosen on validation by the same rule as tau3), both readings. The verifier has to add more to the rules than a linear model of its size does.
- **G4:** issue-report test, candidate F1 of the union >= Gitleaks' + 0.08.
- **G5:** ECE of the model <= 0.05 (unchanged from v2: 0.0073).
- **G6:** native median latency of the model <= 3 ms per candidate (unchanged from v2: 1,606 µs).
- **G7:** clean real text: files flagged by the union <= 1.0% of the 8,825.

**What a pass would mean, and what it would not.** A pass means that the system made of secretgate's rules and this verifier clears, on a second look, the bar the verifier alone missed; it is reported as "passed on a second look, system redefined after the first reading", never as a first-look result. A fail on G1 again, or on G4 (the diagnosis says the union may lose there), ends the work on this model: no third reading.

## 4. Reported, not gated

Everything v2 reported, for the union and for the model alone side by side; the number of lines the union adds over the model alone and their labels; the same union with tau of v2 (0.2598) instead of tau3; the paired bootstrap of the difference between the union and the model alone.

## 5. Forbidden

Choosing tau3 or anything else on a test file; retraining; changing the generator; a third reading.
