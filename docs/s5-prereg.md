# s5: pre-registration (s5-lite, bets 1 and 2, phases 5 and 7)

*Translated into English on 19/11/2026. The text frozen on 13/10/2026 was written in French; that original, which remains the reference in case of doubt, is this file at commit `1f180a5`. The translation changes no rule, number or threshold.*

**Status: frozen on 13/10/2026, before any pass on a `test` split.** This file fixes the evaluation sets, the renderings, the run budget, the selection rules, the metrics, the margins and the success criteria. The results will be compared with this text, and with no later version.

**Safeguard in the code.** `sky-eval` refuses the `test` split of a frozen set (`data/s5/eval/`) in three cases:

- this file is not committed;
- this file carries local modifications;
- the set's sha256 does not appear below.

Each pass on a frozen set is appended to `runs/s5-test-log.jsonl` in the store: date, sha256 of the model and of the sets, commit of this file, arguments.

**Changes to this file:**

- Every amendment is committed, dated and justified in section 9.
- It **precedes** the pass on the test that it concerns.
- It never touches a margin or a criterion after a test result has been read.

## 1. Frozen sets

The sets are built by `sky-data s5-build` (seed 1234). The manifest is in `data/s5/MANIFEST.json`, in the store.

### Evaluation (`data/s5/eval/`, split `test`)

| file | questions | sha256 |
|---|---|---|
| `typed_decisions_test.jsonl` (bench fitted) | 2000 | `72331bda4c04ea46168447535cb1cd461adbba465167d52bea66c28ed8b9f6b3` |
| `wikitoxic_insult.jsonl` | 1000 | `6ca30c5fc05ac86b11947821be9de5fbe36cdc4a6fbb7347f5c0d54fd642a628` |
| `wikitoxic_obscene.jsonl` | 1000 | `bf0f78c9bfe1f945975f0d4850264c8a86135932e5c93e6026ef64facf6d79d4` |
| `wikitoxic_threat.jsonl` | 1000 | `fc9e285210d7a55f70c3eca4349c2b01b3ebecdbfde0468c2e6d8dee0ba3e126` |
| `wikitoxic_toxicaggregated.jsonl` | 1000 | `d6f518050a8a180be9d7e238d2d622fd8a150c4e85a779a523031b3e6c0c0851` |
| `biasframes_intent.jsonl` | 1000 | `6a7fca56b2425d1964ef09963d1d2f5d3aaef408e646e03f0a088ea07eb59518` |
| `biasframes_offensive.jsonl` | 1000 | `781b1da8c8ddc506021795075422665330136489d52097ba82982bb778713b3a` |
| `biasframes_sex.jsonl` | 1000 | `abc790c182574db9b4870fdd8daf17fd2cd16f02e786ad693e8bb046da48792f` |
| `amazonpolarity.jsonl` | 1000 | `59a6012b67cdea0a503acad85a342815ab5efff312f848d6802afb55be748a70` |
| `yahootopics.jsonl` (licence not confirmed: neither evaluated nor published for s5-lite, amendment 1) | 1000 | `e208f70b80e28a6e00371ff8a44cbd0880568be2880bc37e5d47bd14af36c682` |
| `hwu64.jsonl` (s5-lite only, column « contamination via les professeurs non exclue » [contamination through the teachers not excluded]) | 1000 | `2246f781daead75dc44b97d9752792fd428090f94b7a24c5af4f8c863988d575` |
| `sib200_en.jsonl` (secondary, outside the plan's list) | 204 | `9d92f595e64c7e3aba361c79ba61f558192050b8b0f30fa926a8227ab7df1953` |
| `sib200_fr.jsonl` (s5 en + fr only) | 204 | `d09c0c766155d37d5496580b62656ef6af2277c31b0b3e2dee77c38d3c0abc99` |
| `coherence_test.jsonl` (families CH, S_k, N_k, D_ij of the test states) | 10200 | `bf9dff43877d2a680e3e9bbac7be550b67e1772d5ce7eab454c838014f398042` |
| `paraphrase_test.jsonl` (definitions paraphrased by hand, two calls) | 4000 | `211076641b132e7148fcacf166e6e3383ecb212f73860757d87c2d9fb36044ba` |
| `jevout_lite.jsonl` (500 bases × 8 distractors held out of training) | 4500 | `ae6c74c6135eb301854fa70e739c026cbc18a8cc8dce312dfbc749c2be135d8d` |
| `reassign_test.jsonl` (2609.26758: 0/1 control, seen and unseen polarities, random) | 5400 | `79395a73af6734ce966824c64a96b0b14b1f56691554924af6f986738f55f1e6` |
| `policy_test.jsonl` (mini-PolicyBench, 4 policies held out of training) | 800 | `dc28b0dc5f23805c4a062bb60ebb3416ba3220688417c83c6c2a37a0d3e52ea5` |
| `coverage_test.jsonl` (omission and BM25 menus k = 2, 5, 10; 30 % calibration) | 5000 | `80c0aa3fc8adaadf317648664b32f2e0721242572b88b9edf26b9d80c4c636c1` |

### Files written by hand, committed (`eval/s5/`)

| file | sha256 |
|---|---|
| `defs.json` (questions and class definitions of each set) | `364469ae28488bd44008f59f867c06b178e4fd4abedfc77625cdc3cccc09358d` |
| `paraphrases.json` (evaluation paraphrases, disjoint from the training templates) | `6411792c9aafdc37abcbdceaeb1d700c6071e0c24321820759f0adc4deb94d43` |
| `demo-cases.jsonl` (56 demo cases, 30 en, 26 fr; excluded from training) | `c2dea1a5499a5391323ba246622600b4a052dfc3f316fe7d8484c715602fb612` |

### Training (`data/s5/train/`, train / val / cal splits fixed per state)

| file | records (en + fr) | sha256 |
|---|---|---|
| `corpus.jsonl` | 157005 | `61adeaea89d227e1ceda2f0efb5021226eddf70c1aeb3be46b679c1e18972e14` |
| `gen.jsonl` | 25328 | `505ffef41c2fcc6269c968bcf000200ac1e789e768132dc6f8f1f2ea4df6ad72` |
| `pairs.jsonl` | 7828 | `7265bb9aa9da97b59ced59c85ebe30d04272a2fe7eaadc06d5e83a47fcbe5fbf` |
| `coherence.jsonl` | 12628 | `2bf3f00bb9c316d9d2d05350b2e076a2bce4a3c6800169074c26609ff50736e2` |

**Volumes in English:**

| file | train | val | cal |
|---|---|---|---|
| corpus | 139465 | 7604 | 7624 |
| gen | 11393 | 613 | 585 |
| pairs | 7041 | 367 | 420 |
| coherence | 11303 | 605 | 720 |

**Mixed targets:**

- `data/s5/mixed/` (`sky-data s5-mix`) takes the same records, with the targets described in section 3.
- Its hashes appear in amendment 1, committed **before the first exploration run**.

## 2. Renderings and prompts

They are identical for A, A′, B and the baselines. The reference code is `crates/sky-train/src/s5/record.rs`, and `crates/sky-infer/src/bin/sky-eval.rs` for the evaluation.

### Record

A record carries the fields `{src, t, ins, opts_kv: [[id, definition]...], state, target, split, uid, lang}`.

- `t` is `choice`, `noul` or `score`.
- `target` is a distribution, soft or one-hot. The gold is its argmax.

### Native rendering (A, A′)

| type | rendering |
|---|---|
| `choice` | `id: definition` |
| `score` | `level i: definition` |
| `noul` | `false` / `true`, or `false: definition` |

### Definition-only rendering (B, iso)

- `choice`: the definition alone.
- `score`: `level i of K-1: definition`.
- `noul` without a definition: the generic definitions.

| language | false | true |
|---|---|---|
| en | « The statement is false. » | « The statement is true. » |
| fr | « L'affirmation est fausse. » [The statement is false.] | « L'affirmation est vraie. » [The statement is true.] |

The identifiers are never rendered.

### Sequence A / A′ (Laya format)

- Shape: `[CLS] "{t} question: {ins}" [SEP] ([MASK] " option")* [SEP] state [SEP]`.
- Readout: one per `[MASK]`, with one temperature per type.

### Sequence B (iso)

**Segments:**

- the state sees the state;
- the question sees the state and the question;
- option i sees the state, the question and itself.

**Positions:**

- (i) all options start at the same offset L (B-i);
- (ii) each option is reset to the same logical position (B-ii).

**Local window:** computed on the logical positions.

**Readout:** mean over the option's span, then MLP([h, q, h⊙q, |h−q|]) and softmax.

**State budget:** fixed, equal to `max_len − head_max_len − 16` tokens. The state cache therefore never depends on the question.

### Lengths

- `max_len` = 512, `head_max_len` = 192.
- The evaluation sets keep N = min(8, K) options: the gold, plus N − 1 options drawn at random at build time (fixed seed).

## 3. Training data and teacher

### Sources excluded from s5

- `typed_decisions_bench`, rottentomatoes and all the sources of the evaluation sets: amazon_polarity, bias_frames, jigsaw / wiki_toxic, yahoo, SIB-200, HWU64.
- **MASSIVE, SLURP and MTOP, in all languages.** Their exclusion is the condition for admitting HWU64 for s5-lite.
- **Any GLM output** (`glm_orders`, `para:` paraphrases), because of its provider's terms of use.

### Decontamination

The check covers the **states** of the evaluation sets and of the demo cases. A record is removed in two cases:

- 13-gram coverage ≥ 0.5 of a reference;
- MinHash Jaccard ≥ 0.8.

**Records removed:**

| set | removed |
|---|---|
| typed-decisions train | 165 |
| generated rules (format of the demos' safeguard) | 160 |
| pairs | 172 |
| coherence families | 432 |
| others | 17 |

### Class guard

No class exceeds 60 % per source, with a deterministic rebalancing. The coherence families are exempt from it, so that they stay whole.

### Teacher

**laya-typed-decisions** (Apache-2.0), run in Rust (mlx).

- **Rendering:** native rendering, two option orders averaged.
- **Calibration per group** (source × type × K): temperature and bias, on the gold train records. Below 150 records, the group takes the global temperature.
- **Agreement filter:** the teacher's target is taken only if its argmax equals the gold argmax. Otherwise, the gold target is kept.
- **Final target:** target = 0.5 · gold + 0.5 · teacher, for the corpus and the first member of the pairs, **of the train split only**. The second member of a pair takes the target of the first. The val and cal splits keep the gold: selection and calibration are judged against the gold (amendment 1).
- **Keep their exact gold target:** the generated records and the coherence families.

**Deviation from the plan:** the teacher is single. oscar-1-150m (downloaded, 0.7755 on the bench against 0.7665) would have cost about 45 more minutes of GPU, at the measured throughput of 50 to 140 lines/s. With a single teacher:

- there is no choice of teacher per group;
- the measurement of the teacher in native and definition-only renderings is not done.

## 4. Recipe, variants, budget and selection (validation only)

### Recipe common to all runs (`sky-train s5-train`)

**Optimisation:**

- AdamW (without Muon), full fine-tuning;
- peak LR according to the grid below; head × 20;
- warmup over 6 % of the steps, then linear decay;
- weight decay 0.01; EMA 0.999 (weights evaluated: the EMA);
- batch 32, bf16.

**Loss:**

- soft CE + Brier (1) + RPS (0.25, `score` questions);
- symmetric KL 0.5 on the pairs (definition paraphrases and distractors), **identical for all variants**.

**Group sampling:** typed 0.25; corpus 0.4; gen 0.15; pairs 0.1; coherence 0.1. The coherence families are drawn whole.

**Augmentation:** a single stream, common to all variants:

- option order;
- random names (p = 0.5);
- `noul` polarity (p = 0.25).

It has no effect on B, since B does not render the labels.

**Data:** English only (`--langs en`).

### Variants

| variant | rendering | random names | polarity |
|---|---|---|---|
| A | native | no | no |
| A′ | native | yes | yes |
| B-i | iso, positions (i) | common stream | common stream |
| B-ii | iso, positions (ii) | common stream | common stream |

### Run budget (`cargo xtask s5 <stage>`)

1. **`explore`**: A′ and B-i × LR {5e-5, 1e-4, 2e-4} × initialisation {ettin-encoder-17m, ettin-reranker-17m}, at seed 0 and 600 steps, i.e. 12 runs.
2. **`explore-b-ii`**: B-ii at the best configuration (init, LR) of B-i, at 600 steps.
3. **`final`**: F steps, F fixed mechanically by the throughput of the exploration (amendment 1).
   - A′ at seeds 0, 1 and 2;
   - B\* at seeds 0, 1 and 2;
   - A at seed 0, at the configuration chosen for A′.
4. **`lambda`** (bet 2, on its support): λ ∈ {0.1, 0.3} at seed 0.
5. **`lambda-final`**: λ\* at seeds 1 and 2.

### Selection rules

They are frozen in `runs/s5-select/*.json` at their first computation, and use only the validation.

- **Configuration of a variant:** the lowest validation NLL, over all the English val records (9189 questions, at the end of the run).
- **B\*:** B-i or B-ii, according to the lowest validation NLL at the chosen configuration.
- **λ\*:** the lowest validation MAD_neg among the λ whose validation accuracy is ≥ that of λ = 0 (seed 0) − 0.01. If no λ passes, bet 2 stops, and that is published.
- **Support of bet 2:** B\* if bet 1 succeeds, otherwise A′. The arm λ = 0 is formed of the support's three finalist runs.
- **Delivered model (s5-lite):** in the chosen arm, the run with the lowest validation NLL, in calibrated q8_0 format.
  - The chosen arm is B\* if bet 1 succeeds, otherwise A′.
  - The λ\* runs replace it if bet 2 succeeds.

## 5. Calibration and evaluated formats

For each evaluated run, the chain is as follows:

1. **f32 and q8_0 export**, with the readout and the head in f32.
   - B: `sky-train s5-export`;
   - A and A′: `sky-train export` then `sky-convert`.
2. **Calibration.** The temperatures are fitted per type and per type × number of options (n ≥ 100), by minimising the NLL on the English `val` split of `data/s5/train` (9189 questions, already consumed by the selection; amendment 1). They are written into the GGUF. The calibrated model then passes over the `cal` split (9349 questions), reserved for the escalation thresholds of phase 7.
3. **Pass on the test** of all the English sets above and of the demo cases, with the production engine (`sky-eval`, CPU).

## 6. Main comparisons and success criteria

### Bet 1: B\* against A′ (3 seeds each, calibrated f32, bench fitted)

**Main criteria (both are required):**

- **Accuracy:** upper bound of the 95 % CI of (A′ − B\*) ≤ 0.02.
- **ECE** (top-label, 15 bins): upper bound of the 95 % CI of (B\* − A′) ≤ 0.01.

**Secondary, with p-values adjusted by Holm** over these 6 tests:

- bench NLL;
- bench Brier;
- flip rate under paraphrase (two calls);
- JevOut-lite flip rate (worst distractor per base, chosen against the evaluated model);
- macro accuracy of the 8 English zero-shot sets whose licence is confirmed (wikitoxic × 4, biasframes × 3, amazonpolarity; yahootopics removed by amendment 1);
- macro ECE of the same sets.

**Descriptive, without a test:**

- reassignment (flips per arm, minus the 0/1 control, and AUC);
- mini-PolicyBench;
- candidate coverage;
- A, on one seed.

**In case of failure:** the definition-only rendering is not adopted, and the failure is published.

### Bet 2: λ\* against λ = 0 (3 seeds each, calibrated f32)

**Losses trained:** L_neg, L_part and L_xf only. No D question, no monotonicity, description or disjunction loss.

**D competence, measured first at λ = 0:** the lower bound of the CI of the accuracy of the D_ij questions against the gold must be > rate of the majority class of their golds + 0.10.

- The "chance" used here is the majority class, stricter than 0.5.
- Without competence, the identities held out of training are declared **not measurable** and do not count as a success.

**Main criteria (all required, competence included):**

| criterion | measurement | threshold |
|---|---|---|
| MAD_neg | mean over the families of \|s_k + n_k − 1\| for λ\*, negations rephrased as separate queries, mean of the 3 seeds | < 0.03 |
| partition | mean of Σ_k s_k for λ\* | in [0.97, 1.03] |
| bench accuracy | non-inferiority, margin 0.02 (as in bet 1) | — |
| monotonicity | share of the checks s_i ≤ d_ij violated by more than 0.05 | relative drop ≥ 30 %, and upper bound of the CI of (λ\* − λ = 0) < 0 |

**If only the trained identities improve** (MAD_neg, partition and accuracy pass, but not monotonicity or competence), the result is published as such.

**Secondary, with Holm over these 6 tests:**

- MAD_neg;
- \|Σ s − 1\|;
- \|s − c\|;
- additivity \|d_ij − (s_i + s_j)\|;
- description invariance \|d_ij − n_m\| (K = 3);
- bench ECE.

### Phase 5: quantisation

- **Measurement:** ΔECE = ECE(recalibrated q8_0) − ECE(recalibrated f32) on the bench, paired, for A′ and B\* (3 seeds).
- **Criterion:** upper bound of the CI ≤ 0.01.
- Q4 is not done here.

### Gates of s5-lite (delivered model, calibrated q8_0)

| gate | threshold |
|---|---|
| bench fitted | ≥ 0.65 |
| English demo cases | ≥ 90 % (27/30) |
| each set | ≥ its majority baseline (share of the most frequent gold identifier) |

**Any set above 0.735** (the teacher's agreement ceiling) is investigated before publication.

### Phase 7: error-controlled escalation (offline)

**Acceptance thresholds on max p, per type, chosen by Learn-then-Test:**

- binomial p-value of the error of the accepted;
- fixed-sequence test over 200 thresholds, from highest to lowest;
- δ = 0.05.

**Calibration data:**

- **Main:** the `cal` split of the typed-decisions records, exchangeable with the bench (325 questions).
- **Secondary:** the whole English `cal` split (more data, but the formal guarantee is not valid under shift).

**Comparison:** the rule « ensemble conforme de taille 1 » [conformal set of size 1] (Mondrian, s = 1 − p_y).

**Reported:**

- error of the accepted and coverage for α ∈ {1, 2, 5, 10} %;
- AURC;
- coverage at ≤ 5 % error.

**Promise:** P(error | accepted) ≤ α, with respect to the gold. Bet 3 (online, with audit) is not launched here; it was launched later, as an exploratory analysis (amendment 4).

## 7. Statistics

- **Intervals:** 95 % percentile bootstrap, B = 2000, seed 0 (`sky-s5-report`, `sky-metrics`).
- **Several seeds of a variant:** two-level bootstrap.
  - Each draw resamples the items, then the seeds with replacement.
  - It takes the mean of the statistic over the drawn seeds.
  - A paired comparison draws the **same** items for both arms, and their seeds independently.
  - The p-value is the two-sided bootstrap p-value.
- **Units:**
  - question for the sets;
  - family for coherence;
  - base for the paraphrases and JevOut-lite.
- **"Not significant"** is never a success criterion. Only non-inferiority at a fixed margin is.

## 8. Deviations from the plan, declared before any result

- **Single teacher** (section 3).
- **Reduced exploration grid:**
  - A and B-ii do not go through the grid: A takes the configuration of A′, and B-ii that of B-i;
  - steps fixed by the compute budget: 600 in exploration, F for the finalists and bet 2, with F drawn from the exploration's throughput alone (amendment 1).
- **Not done:**
  - A′ + perm-KL;
  - baselines B1 (log-prob of the labels by the MLM head), B2 (cascade) and B3 (linear head). Only the majority class and the uniform are reported.
- **Comparison with Jev (MAD_neg 0.064, Σ s 1.14, |s − c| 0.091):** done on the original battery of 2609.33209, converted into an s5 set (amendment 1). It remains **descriptive**: a single measurement per model, without a hypothesis test. Our rephrased negations form another set, reported separately.
- **Postponed:**
  - Q4 (reported later as a secondary result, without a criterion);
  - online bet 3 (launched later, exploratory: amendment 4);
  - phase 6 (s5 en + fr, mmBERT-small), closed later without being launched (note of 14/10/2026 below).
- **French fallback:** the French demos of /live display « français indisponible : s5-fr en échec, voir résultats » [French unavailable: s5-fr failed, see results] and switch to the English presets.
- **Licences:**
  - MTOP is excluded (unknown licence);
  - the licence of yahootopics is not verified (flagged);
  - HWU64 is reported for s5-lite only, in the column « contamination via les professeurs non exclue » [contamination through the teachers not excluded], because Laya's sources are not all declared.

## 9. Amendments

### Amendment 1: 13/10/2026, before the first exploration run and before any pass on the test

**Corrections to the mix, made before its first pass** (commit `d5a7975`):

- **Agreement filter.** With a single teacher, the code counted the teacher as unanimous with itself, so that every record took its target. From now on, only agreement with the gold counts, as section 3 already describes.
- **val and cal splits.** Only the train split receives the mixed target. Val and cal keep the gold, so that the selection (validation NLL) and the calibration are judged against the gold.

**Temperatures.** They were planned on the `cal` split, which must also fix the escalation thresholds of phase 7: fitting both on the same data weakens the Learn-then-Test guarantee, and the planned code moreover computed the thresholds on the uncalibrated probabilities. The temperatures are therefore fitted on the `val` split, already consumed by the selection. The `cal` split, read by the calibrated model, now serves only for the thresholds.

**Number of steps of the finalists and of bet 2.** The planned throughput bench could not run before the exploration (GPU taken by the teacher until about midnight). F is therefore fixed mechanically after the exploration, from its throughput alone, without looking at any validation metric:

- t = median, over the 12 exploration runs (launched three at a time), of the training minutes divided by 600;
- F = the largest value of (2000, 3000, 4000, 6000) such that 5 waves × F × t ≤ 180 minutes (three waves of finalists, two of bet 2);
- the computation and F are frozen in `runs/s5-select/steps.json` before the finalists are launched.

**Diverged runs.** A run that diverges (non-finite loss, hence without `val.json`) is excluded from the selection and recorded as a failure in `runs/s5-select/`. A diverged B-ii leaves B-i as B\*.

**yahootopics.** Its licence is not confirmed, and `DATA_LICENSES.md` reserves its publication for a confirmed licence. It is therefore neither evaluated nor published for s5-lite. This removes:

- the set `yahootopics.jsonl`;
- the yahootopics lines of `coverage_test`;
- its place in the zero-shot macro mean of bet 1, which goes from 9 to 8 sets.

**Battery 2609.33209.** It is downloaded in phase 0 (`bro789/typed-decision-coherence`, commit `8898ef7b`), contrary to what section 8 wrote. `sky-data s5-battery` converts it into s5 records:

- 160 items: 100 ChaosNLI (CC BY-NC 4.0) and 60 PubMedQA (MIT), in evaluation only, never redistributed;
- one family per item: CH with the options in the order of the classes, S_k and N_k with the battery's questions, D_k = « l'une des deux autres classes » [one of the two other classes];
- targets: the human votes for ChaosNLI, the expert label for PubMedQA.

Each run is evaluated on it in f32, and `sky-s5-report coherence --set battery_2609_33209` gives MAD_neg, Σ s and |s − c| on the same items as Jev. The comparison is descriptive.

| file | records | sha256 |
|---|---|---|
| `data/s5/battery/coherence_2609_33209.jsonl` | 1600 | `f10f8dc32508c4d5e7e1c49533ff3b5875d629cb3c860fcaa2f92f4692f8eae8` |

**oscar-1-17m reference (descriptive, outside the bets).** The step `cargo xtask s5 oscar-ref` evaluates oscar-1-17m (Laya format, published weights, original temperatures) on two sets:

- **`reassign_test`**: the yes/no swap of 2609.26758, next to the iso guarantee;
- **the bench**: our rendering of the s5 records must give back its published score, 0.6775; a gap would be investigated.

No criterion depends on it.

**Mixed targets** (`data/s5/mixed/`). Teacher: laya-td; α = 0.5. Tally: corpus: 72596 mixed targets, 66869 keep the gold, 2057 not scored (the French records of the train split, outside s5-lite); pairs: 3025 mixed targets, 467 keep the gold, 0 not scored.

| file | lines | sha256 |
|---|---|---|
| `REPORT.json` | – | `6e9e0c26586517652b60299ccd442efa01d5d53109435328745f58d782336009` |
| `coherence.jsonl` | 12628 | `2bf3f00bb9c316d9d2d05350b2e076a2bce4a3c6800169074c26609ff50736e2` |
| `corpus.jsonl` | 157005 | `86342e24bce543b70c93159e2e6f5054a6cc6a4e76855a75e326775707250825` |
| `gen.jsonl` | 25328 | `505ffef41c2fcc6269c968bcf000200ac1e789e768132dc6f8f1f2ea4df6ad72` |
| `pairs.jsonl` | 7828 | `6151b962dca5b1a921ce7183c2600a8bdb5ee1aa59e545fb1d742ed651836720` |

### Amendment 2: 14/10/2026 around 01:05, before any result of bet 1

**Finding, on training and validation only.** The A′ finalists are unstable over 2000 steps, at the LR of 2e-4 chosen over 600 steps:

- **seed 1:** gradient-norm spike (22) at step 400, then the loss stays around 1.3; validation accuracy 0.373;
- **seed 0:** spikes up to 36 between steps 1000 and 1400; validation accuracy 0.426.

B-i, seed 0, reaches 0.529 on the same validation. Selecting on short runs favours an LR too high for long runs: this known bias can favour B against a badly tuned A′.

**Sensitivity analysis added (secondary):**

- **runs:** A′ at the chosen initialisation and at the next lower LR of the grid (1e-4), seeds 0, 1 and 2, same number of steps as the finalists (`cargo xtask s5 sens`), evaluated like the finalists;
- **comparison:** B\* against this A′, with the criteria and margins of bet 1 (`runs/s5-report/pari1-sensitivity.json`).

**What does not change.** The main verdict of bet 1 remains that of the pre-registered comparison (B\* against the selected A′). The sensitivity is reported alongside. If it contradicts it, both are published, and no superiority of B is claimed.

**Test results seen before this decision.** Only one: the accuracy of f-b-s0 on amazonpolarity (0.679), a secondary zero-shot set. No test result of A′, no bench result, no comparison.

### Amendment 3: 14/10/2026 around 01:20, model of the /live demo, outside the bets

**Finding, after the test results of f-b-s0** (a single seed of B\*, no comparison):

- the bench gate passes (0.667);
- the gate of the English demo cases fails (70 %, against 90 % required);
- five sets remain below their majority baseline.

**Long run, outside the bets.** One additional B\* run, `long-b-s0` (`cargo xtask s5 long`):

- same configuration as the B\* finalists (init, LR), seed 0, λ = 0;
- 8000 steps instead of 2000;
- evaluated on the same sets.

**Rule, fixed before its result.** /live serves, between the model delivered by the pre-registered rule and `long-b-s0`, the one whose validation NLL is the lowest. The choice does not use the test.

**What does not change:**

- no bet uses this run;
- the verdicts, the s5-lite gates and the claims bear on the model delivered by the pre-registered rule;
- the figures of the long run are published separately, marked « hors pré-enregistrement, décidé après avoir vu les résultats de test d'une graine » [outside the pre-registration, decided after seeing the test results of one seed].

### Editorial note: 14/10/2026, after the results

A line of section 3 (excluded sources) was rephrased so as no longer to name GLM's provider. No set, no criterion, no margin and no rule changed. The original version remains in the repository's history; its hashes are those logged with each pass on the test.

### Amendment 4: 14/10/2026 around 11 h, bet 3 launched after the fact, parameters fixed before its first pass

**Status.** Bet 3 (online escalation with audit) had been postponed (section 8). It is launched after the results of the offline Phase 7, which are known: on the pre-registered calibration, Learn-then-Test certifies no threshold for α ≤ 10 %. Bet 3 is therefore **exploratory**. It is published as is, success or failure, and supports no confirmatory claim. Everything that follows is fixed in the code (`sky-s5-report pari3`, commit of this amendment) before its first pass.

**Model and data:**

- the model delivered by the pre-registered rule, in calibrated q8_0 format;
- the stream replays this model's bench dump (`typed_decisions_test`), already analysed offline;
- the calibration is its `cal` dump, restricted to the source `typed_decisions_train`, like the main calibration of Phase 7.

**Starting threshold, per type.** It is the lowest of the 200 max p thresholds whose empirical error of the calibration accepted is ≤ α. This "plug-in" choice has no guarantee, since LTT certifies nothing at these α. The same threshold, kept fixed, serves as the **frozen threshold** for comparison.

**Stream.** With n the size of the bench, each stream counts 2n questions drawn with replacement (seed s):

- **first half:** uniform draw;
- **second half, prevalence drift:** weight × 3 for a hard question (max p below the median of its type) and × 3 for a minority answer (gold option whose count in its type is below the median of the counts of the gold options of that type). The two factors multiply.

**Audit.** The auditor receives all the escalated and each accepted one with a probability π, drawn at random (seed 1000 + s). In the experiment, the auditor is the set's gold.

**Update, per type:**

- ê_t = err_t · O_t / P(O_t), with P(O_t) = 1 for an escalated one and π for an accepted one;
- λ_{t+1} = λ_t + η · 1{accepted_t} · (ê_t − α), bounded in [0, 1].

**Grid:** α ∈ {5, 10} %, π ∈ {0.1, 0.2, 0.5}, η = 0.005, 20 stream seeds (0 to 19).

**Criterion, per cell (α, π):**

- the mean over the seeds of the estimated error of the accepted of the second half is ≤ α + 0.01;
- the mean of the escalations of the second half is at most 1.2 times that of the frozen threshold on the same streams.

The bet succeeds if all the cells succeed.

**Reported (`runs/s5-report/pari3.json`):** estimated error and true error of the accepted before and after the drift, coverage, escalations online and at the frozen threshold, audit rate, and number of seeds that meet the error criterion. A trajectory of seed 0 (200 steps, without any GLM output) feeds the cascade demo of /live.

### Note: 14/10/2026, French is out of scope

Phase 6 (s5 en + fr, on mmBERT-small) is closed without being launched: s5 remains a small English model. No bet, no gate and no criterion of s5-lite changes. The gates that bore only on the French version (SIB-200 fr, MTOP fr and the French demo cases) are moot. The banner of /live now says « s5 est un petit modèle anglais ; il n'a pas de version française » [s5 is a small English model; it has no French version], and the French demos keep the English presets. The French data (`sib200_fr`, the fr demo cases) remain in the store.

### Amendment 5: 14/10/2026 around 15:30, calibration per source, exploratory, fixed before its computation

**Status.** **Exploratory** analysis, decided after the results of bet 1, which are known. It never changes the verdict of bet 1 nor any criterion: bet 1 remains judged on the pre-registered calibration (section 5). It is published as is, gain or loss. No new pass on the test: it reuses the bench dumps already written. Everything that follows is fixed in the code (`sky-s5-report retemper`, commit that follows this amendment) before its first computation.

**Question.** The pre-registered calibration fits the temperatures on the whole `val` split, all sources combined. The bench comes from typed-decisions. Do temperatures fitted on the typed-decisions questions alone calibrate the bench better, and does the gap between B\* and A′ change?

**Temperatures per source.** For each run of A′ and of B\* (3 seeds each, f32):

- we keep the lines of the `val` dump at T = 1 (`val.f32.dump.jsonl`) whose source starts with `typed_decisions` (`typed_decisions_train` and its variants `+distract`, `+none`, `+pair`, `+para`);
- we fit T_src per type and per type × number of options (n ≥ 100) with the functions of the pre-registered calibration (`fit_t`, `temperatures`), same fallback rule (type × options, otherwise type, otherwise 1).

**Application.** The bench dump was produced at the applied temperature T, read from `temperatures.f32.json`. Each question receives p′ = softmax(ln p / T′), with T′ = T_src / T, which amounts to dividing the logits by T_src. The argmax does not change, so neither does the accuracy. The re-tempered dumps go into the run's `eval-f32-src`.

**Reported:**

- `pari1-src`: the comparison of bet 1 (B\* against A′), redone on the re-tempered dumps. Only the bench is re-tempered: the secondary comparisons on the other sets are absent (NaN);
- `src-vs-global-{bras}`: for each arm, the bench ECE before and after (paired difference, 95 % CI).
