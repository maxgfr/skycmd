# 01 Why tiny decision models

skycmd trains "System One" decision models with a few hundred thousand to a few million parameters. They score a fixed set of options in one forward pass, small enough to fly a game drone at 60 Hz inside a browser tab. This chapter covers what a decision model is, why it can be this small, and where the size limit shows up.

## Generation versus scoring

A chat model writes its answer one token at a time and has to be parsed afterwards. A decision model does not generate anything. You send it a **state** and one or more **typed questions**, and it returns one probability per allowed option. The skycmd encoder reads everything in a single bidirectional pass, and the decision head produces one logit per option.

The prompt layout is fixed by the spec (`docs/SPEC.md`) and matches llama.cpp's `/v1/systemone` server:

```text
[CLS] "{t} question: {ins}" [SEP] ([MASK] " {opt_i}")* [SEP] {state} [SEP]
```

Each `[MASK]` marks an option. The head reads the encoder output at those positions, scores them, and turns the scores into probabilities with a temperature-scaled softmax. The model can only answer with an option it was given, so a structurally invalid answer cannot happen.

The answer types are plain Rust enums. Here is how probabilities become an answer, exactly as the reference server does it:

`crates/sky-schema/src/systemone.rs`

```rust
    pub fn from_probs(question: &Question, probs: &[f64]) -> Answer {
        let keys = question.option_keys();
        assert_eq!(keys.len(), probs.len(), "one probability per option");
        let probabilities: IndexMap<String, f64> =
            keys.iter().cloned().zip(probs.iter().copied()).collect();
        match question {
            Question::Choice { .. } => {
                let mut best = 0;
                for (i, p) in probs.iter().enumerate() {
                    if *p > probs[best] {
                        best = i;
                    }
                }
                Answer::Choice {
                    choice: keys[best].clone(),
                    probabilities,
                    confidence: confidence_choice(probs),
                }
            }
```

A `choice` returns the argmax and a confidence. A `score` returns the expected level. A `noul` (probabilistic yes/no) returns p(true). [Chapter 03](./03-typed-decisions) goes through all three.

## What "tiny" means here

The open decision models listed in `docs/research.md` range from 17M parameters (`mgoeckel/oscar-1-17m`) up to 27B. Even the 17M model is mostly vocabulary: about 12.9M of its parameters sit in a 50k-token embedding table. skycmd keeps the same architecture (ModernBERT encoder plus the Laya decision head) but shrinks both the vocabulary and the transformer.

The size ladder comes from `docs/SPEC.md` and `configs/s1-*.json`:

| name | d | layers | heads | intermediate | head layers | vocab |
|---|---:|---:|---:|---:|---:|---:|
| s1-pico | 64 | 2 | 1 | 128 | 1 | 1024 |
| s1-nano | 128 | 4 | 2 | 256 | 1 | 2048 |
| s1-micro | 256 | 6 | 4 | 512 | 2 | 4096 |

All three share `max_len = 256` and `head_max_len = 96`. The pico config, for example:

`configs/s1-pico.json`

```json
  "vocab_size": 1024,
  "hidden_size": 64,
  "num_hidden_layers": 2,
  "num_attention_heads": 1,
  "intermediate_size": 128,
  "global_attn_every_n_layers": 3,
  "local_attention": 128,
  "global_rope_theta": 160000.0,
  "local_rope_theta": 10000.0,
```

Parameter counts for the decision models (encoder plus head), taken from `BENCH.md`:

| model | decision params |
|---|---:|
| s1-pico | 202,305 |
| s1-nano | 1,134,209 |
| s1-micro | 6,630,913 |

The exported `s1-pico.gguf` is 545,280 bytes (`site/public/models/manifest.json`), small enough to ship with the web page.

## Why small is enough for games

Three design choices make a model this small usable.

1. **Compact state text.** The games don't describe the world in prose. They emit a short list of `key value` pairs with small, binned integers, so every number is a single token of a 1k vocabulary. An integration test enforces at most 24 whitespace-separated pieces.
2. **The code measures, the model decides.** Distances, clearances and battery costs are computed in Rust, and the model only has to map those features to a decision. The Flappy state text puts this in a comment:

`crates/sky-games/src/flappy.rs`

```rust
        // Clearances are relative to the drone, so the model reads "room above / below" directly
        // instead of subtracting absolute heights (the code measures, the model decides).
        format!(
            "vy {} dx {} up {} dn {} nxt {} alt {}",
            bin(self.vy, 25.0, -16, 16),
            bin(dx, UNIT, -12, 60),
            bin(p.gap_top - (self.y + DRONE_HH), UNIT, -20, 20),
            bin((self.y - DRONE_HH) - p.gap_bottom, UNIT, -20, 20),
            bin(next.center() - p.center(), UNIT, -20, 20),
            bin(self.y, UNIT, 0, 64),
        )
```

3. **A coded safety layer.** A deterministic veto replaces any action that would crash, so the model never has the last word on safety ([chapter 03](./03-typed-decisions)).

### Latency

Native latency of `s1-pico` per decision, measured by `sky-arena` on the build machine (`runs/s1-pico/arena.json`):

| game | decisions | p50 (µs) | p95 (µs) |
|---|---:|---:|---:|
| flappy | 1560 | 224.6 | 347.8 |
| rescue | 3913 | 429.8 | 542.2 |
| dodge | 30660 | 243.2 | 316.9 |

That table is the first s1-pico checkpoint, before the DAgger rounds of [chapter 20](./20-dagger-lessons). The shipped s1-pico decides a Flappy frame in 201 µs at the median (p95 221 µs) in `sky-arena` (`runs/ship-s1-pico.arena.json`). `BENCH.md` ("Inference") gives the cleaner `sky-bench` numbers on the same M5:

| model | native, 1 question | native, 2 questions on one state | WASM, one Flappy decision (2 questions + physics) |
|---|---:|---:|---:|
| s1-pico | 188 µs | 407 µs | 694 µs |
| s1-nano | 974 µs | 1,980 µs | 3,541 µs |
| s2-nano (ternary file, full depth) | 298 µs | 329 µs | 501 µs |

s1-micro (6.6M parameters) is not in `BENCH.md`. Measured for [chapter 12](./12-size-ladder) on a busy machine, it takes about 4,600 µs per question natively and about 32,600 µs per WASM decision, two full 60 Hz frames. The s2-nano row comes from a randomly initialised model of the same shape, so only its speed is meaningful ([chapter 19](./19-s2-architecture)).

## Where the size limit shows

Small models have clear limits, and skycmd publishes them rather than hiding them. The held-out test report for s1-pico (`runs/s1-pico/report.md`, calibrated temperatures):

| source | n | accuracy |
|---|---:|---:|
| game:dodge | 2991 | 0.9442 |
| game:rescue | 3043 | 0.8751 |
| game:flappy | 3100 | 0.8245 |
| typed_decisions_bench | 2000 | 0.3280 |

Two lessons come out of these numbers.

- **General decisions need capacity.** On the external `LocalLLaMA/typed-decisions` benchmark, s1-pico scores 0.3280. The oscar-1-17m model card reports 0.6775 on the same benchmark (a vendor-reported number). s1-nano scores 0.3975 and s1-micro 0.3100 (`runs/s1-nano/report.md`, `runs/s1-micro/report.md`), so a model 33 times larger than pico did not do better here.
- **Per-decision accuracy is not policy quality.** In `runs/s1-pico/arena.json`, the same model that labels 82% of Flappy test states correctly crashes in all 30 Flappy episodes (mean score 0.0, against 200.0 for the oracle). Flappy needs a decision every frame, so one wrong frame near a pipe ends the episode. Rescue holds up better: a 0.63 rescue rate with 130 vetoes from the safety reflex, against 1.0 for the oracle. This is why skycmd evaluates in closed loop and runs DAgger rounds (`scripts/specialize.sh`), and why the safety veto exists. After two rounds the same 0.2M-parameter model flies 200 of 200 pipes in all 30 episodes (`runs/ship-s1-pico.arena.json`, [chapter 20](./20-dagger-lessons)).

## Research that shaped the recipe

`docs/research.md` lists the papers behind each choice. In short:

- Plain cross-entropy on soft targets, followed by temperature scaling, instead of RLCD.
- Options are shuffled and renamed during training, because decision heads tend to latch onto option labels.
- `end` and `abort` are explicit actions, because small models struggle with termination.
- A coded validator always gets the last word, because constrained outputs remove structural errors but not semantic ones.
- French and English are mixed in the training data.

## Map of the repository

| crate | role in the tutorial |
|---|---|
| `sky-schema` | request/response types, decision records, plan validator (chapter 03) |
| `sky-games` | game cores, state text, oracles, dataset generator (chapter 04) |
| `sky-data` | Hugging Face pipeline, tokenizer training, GLM teacher (chapters 05 to 07) |
| `sky-train` | mlx-rs training, calibration, export |
| `sky-gguf` | GGUF v3 reader and writer |
| `sky-infer` | the inference engine, native and wasm32 |
| `sky-web` | one wasm module that steps a game and the model together |
| `sky-arena` | closed-loop evaluation and DAgger data collection |

## Next

[02 Setup on Apple Silicon](./02-setup-apple-silicon)
