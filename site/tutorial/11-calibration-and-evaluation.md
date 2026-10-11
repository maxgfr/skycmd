# 11 Calibration and evaluation

A decision model returns probabilities, and the game controllers and safety validators use those numbers as they are. After training, `sky-train calibrate` fits softmax temperatures per question type and option count on the validation split. `sky-train eval` then writes a per-source report of accuracy, NLL, Brier score and ECE on the held-out test split.

## Why temperatures

Cross-entropy training gives logits whose ranking is right far more often than their scale. Temperature scaling divides every logit by one constant `T` before the softmax. The argmax does not change, so accuracy is unaffected, but the probabilities get sharper (`T < 1`) or flatter (`T > 1`) until they match observed frequencies. The research notes cite this as the reason skycmd trains with plain CE and fits temperatures afterwards (paper 2610.02486).

The two-option yes/no questions and the ten-option choices do not need the same scaling, so Laya keys temperatures by type **and** option-count bucket.

## Buckets: two spellings of the same key

`sky-train` stores temperatures in the Laya `rl_agent_config.json` format, where a bucket key looks like `choice:3-5`. The GGUF metadata and the llama.cpp server use `choice.3_5`. Both spellings live side by side in `sequence.rs`:

`crates/sky-train/src/sequence.rs`
```rust
/// Laya temperature bucket key (`rl_common.temp_bucket`), e.g. `"choice:3-5"`.
pub fn temp_bucket(qtype: usize, k: usize) -> String {
    let size = match k {
        0..=2 => "2",
        3..=5 => "3-5",
        6..=10 => "6-10",
        _ => "11+",
    };
    format!("{}:{}", QTYPES[qtype], size)
}

/// The same bucket in the GGUF / llama.cpp server form, e.g. `"choice.3_5"`.
pub fn gguf_bucket(laya_key: &str) -> String {
    laya_key.replace(':', ".").replace('-', "_").trim_end_matches('+').to_string()
}
```

The lookup falls back from bucket to type to 1.0, in training and at inference alike:

`crates/sky-train/src/export.rs`
```rust
pub fn temperature_of(rl: &Value, qtype: usize, k: usize) -> f32 {
    if let Some(t) = rl["temperature_by_options"][temp_bucket(qtype, k)].as_f64() {
        return t as f32;
    }
    rl["temperature"][qtype].as_f64().unwrap_or(1.0) as f32
}
```

`crates/sky-infer/src/model.rs`
```rust
    pub fn temperature(&self, type_name: &str, bucket: &str) -> f32 {
        let full = format!("{type_name}.{bucket}");
        for key in [full.as_str(), type_name] {
            if let Some((_, t)) = self.temperatures.iter().find(|(k, _)| k == key) {
                return *t;
            }
        }
        1.0
    }
```

## Fitting a temperature

`fit_temperature` minimises the summed NLL of the soft targets over `ln T`. It uses a golden-section search on `[ln 0.05, ln 20]` with 80 iterations, which is far more precision than needed and costs nothing next to the forward passes:

`crates/sky-train/src/metrics.rs`
```rust
pub fn fit_temperature(preds: &[&Pred]) -> f32 {
    if preds.is_empty() {
        return 1.0;
    }
    let f = |lt: f64| -> f64 {
        let t = lt.exp() as f32;
        preds.iter().map(|p| nll(&p.logits, &p.target, t)).sum::<f64>()
    };
    let (mut a, mut b) = (0.05f64.ln(), 20f64.ln());
    let g = (5f64.sqrt() - 1.0) / 2.0;
    let mut c = b - g * (b - a);
    let mut d = a + g * (b - a);
    let (mut fc, mut fd) = (f(c), f(d));
```

The search runs in log space because temperatures are scale factors: 0.5 and 2.0 are equally far from 1. The `temperature_recovers_overconfidence` test builds predictions three times too sharp and checks that the fit returns 3.0 within 0.01.

`calibrate` first runs the model once over the chosen split (identity option order, raw logits). It fits one temperature per question type, then one per `type:bucket` for every bucket with at least `--min-bucket` questions (default 50). Buckets below the threshold fall back to the type temperature at lookup time.

### Running it

```bash
export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer
export CARGO_TARGET_DIR=$PWD/target/sky-train
T=$CARGO_TARGET_DIR/release/sky-train
$T calibrate --run runs/s1-pico/decide/last --records data/decisions,data/games
```

`--split` defaults to `val`. The command prints each temperature and the NLL and ECE before and after, then rewrites `rl_agent_config.json` in the checkpoint. This is the calibrated config of the current pico run (it may be retrained):

`runs/s1-pico/decide/last/rl_agent_config.json`
```json
  "temperature": [
    0.9923004508018494,
    0.9412853717803955,
    0.9828470349311829
  ],
  "temperature_by_options": {
    "choice:2": 1.0647698640823364,
    "choice:3-5": 1.0317163467407227,
    "choice:6-10": 0.9566465020179749,
    "noul:2": 0.9828470349311829,
    "score:3-5": 0.941575288772583
  },
```

Every fitted value lies between 0.94 and 1.07. Soft-target cross-entropy already left pico close to calibrated on its own validation data, and the fit only makes small corrections. `sky-train export` copies these values into the exported `rl_agent_config.json` and into a `temperature` tensor, which is how oscar stores them. From there they reach the GGUF file ([chapter 13](./13-gguf-export-int8)).

## Metrics

`metrics::compute` reports four numbers for any set of predictions, each prediction scaled by its own temperature:

`crates/sky-train/src/metrics.rs`
```rust
    for p in preds {
        let t = temp(p);
        let q = softmax_t(&p.logits, t);
        let hit = (argmax(&q) == argmax(&p.target)) as u8 as f64;
        n += 1;
        acc += hit;
        nll_sum += nll(&p.logits, &p.target, t);
        brier += q.iter().zip(&p.target).map(|(a, &b)| (a - b as f64).powi(2)).sum::<f64>();
        conf.push(q[argmax(&q)]);
        correct.push(hit);
    }
```

- **Accuracy**: the predicted argmax equals the target's argmax. For soft targets this is the most probable gold option.
- **NLL**: cross-entropy against the full soft target. It is the quantity that calibration minimises.
- **Brier**: squared error between the probability vector and the target, summed over options.
- **ECE**: expected calibration error of the top probability, with 15 equal-width bins, ported from `rl_common.ece_score`. It measures how far "70% confident" is from "right 70% of the time".

## The evaluation report

```bash
$T eval --run runs/s1-pico --records data/decisions,data/games
```

`--run` accepts `runs/<name>`, `runs/<name>/decide` or `runs/<name>/decide/last`. `--split` defaults to `test`, and the command falls back to `val` when no test items exist. `--report` overrides the output path, which defaults to `runs/<name>/report.md`. The report has four tables: per source (`src`), per data group, per question type (all calibrated), and per data group at T = 1 for comparison.

### Reading the s1-pico report

These rows come from `runs/s1-pico/report.md` (test split, 11,134 questions, calibrated). They describe the current pico checkpoint, and the numbers will change if the ladder is retrained.

| source | n | accuracy | NLL | Brier | ECE |
|---|---:|---:|---:|---:|---:|
| all | 11134 | 0.7813 | 0.6449 | 0.0879 | 0.0738 |
| game:dodge | 2991 | 0.9442 | 0.3383 | 0.0451 | 0.0562 |
| game:flappy | 3100 | 0.8245 | 0.5182 | 0.0762 | 0.0839 |
| game:rescue | 3043 | 0.8751 | 0.6979 | 0.0415 | 0.1189 |
| typed_decisions_bench | 2000 | 0.3280 | 1.2192 | 0.2406 | 0.0611 |

Four things stand out.

1. **The games are learned; open-domain decisions are not.** The three game sources reach 0.82–0.94 accuracy. On `typed_decisions_bench` (`LocalLLaMA/typed-decisions`) pico scores 0.3280. The research notes quote 0.6775 for `oscar-1-17m` on the same dataset, a model card figure for a model about 85 times larger, so the evaluation protocols may differ. `data/STATS.md` also shows that 96.8% of these benchmark sequences are truncated with the 1024 vocabulary, against 79.0% with 4096. Nano and micro results on this benchmark are TBD.
2. **`score` questions are the hardest type.** By question type, `score` reaches 0.6375 accuracy against 0.8078 for `choice` and 0.8057 for `noul`. Adjacent levels are often close calls, and accuracy counts an answer that misses by one level as wrong. NLL and Brier give partial credit for those near misses.
3. **Calibration barely moves the test numbers.** Overall NLL is 0.6449 calibrated and 0.6448 at T = 1, and ECE is 0.0738 against 0.0728. With temperatures this close to 1, this is expected. The val fit also cannot see the benchmark source, which is test-only and makes up 2000 of the 11,134 test questions.
4. **Low ECE is not the same as useful.** `typed_decisions_bench` has the second-lowest ECE in the table (0.0611) at 0.33 accuracy. The model is honestly unsure. ECE has to be read next to accuracy and NLL.

## From logits to an answer at inference

The browser engine applies the same readout as the llama.cpp server. The logits are divided by the bucket temperature and passed through a softmax. A `choice` answer reports the argmax with `confidence = (pmax − 1/n)/(1 − 1/n)`, a `score` answer reports the expected level, and a `noul` answer reports p(true):

`crates/sky-infer/src/prompt.rs`
```rust
/// Choice confidence `(p_max - 1/n) / (1 - 1/n)` (llama.cpp server, TypeSafe formula).
pub fn confidence_choice(p: &[f32]) -> f32 {
    if p.len() < 2 {
        return 1.0;
    }
    let u = 1.0 / p.len() as f32;
    let pmax = p.iter().fold(0.0f32, |a, &b| a.max(b));
    ((pmax - u) / (1.0 - u)).max(0.0)
}
```

This confidence rescales the top probability so that a uniform answer scores 0 whatever the number of options. It is only meaningful when the probabilities are calibrated, which is the point of this chapter.

## What the report does not measure yet

- **Coherence without labels.** `docs/research.md` lists the check that p(X) + p(not X) ≈ 1 for a statement and its negation (paper 2609.33209) as an evaluation idea. `sky-train eval` does not compute it yet, so the result is TBD.
- **Shuffle stability.** `eval` scores every question once in record order. Flip rates under option shuffling (paper 2610.06744) are TBD.
- **Closed-loop play.** Accuracy on single questions does not tell you whether a model can fly a whole episode. Episode metrics come from the game harness, covered in later chapters.
- **Nano and micro.** Their reports (`runs/s1-nano/report.md`, `runs/s1-micro/report.md`) are TBD until the ladder finishes training.

## Next

[12 The size ladder](./12-size-ladder)
