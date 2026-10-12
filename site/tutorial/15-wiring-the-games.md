# 15 Wiring the games

The browser runs one WASM module, `crates/sky-web`. It holds the game cores and the decision model, so a whole AI frame (state text, tokenization, forward passes, readout, action, safety reflex, physics) happens without leaving WASM. JavaScript only draws. This chapter walks through that module, the JSON it hands to the page, and how the page keeps 60 frames per second.

## Why one module

One alternative would load only the engine (`sky-infer`'s `SkyEngine`, chapter 14) and step the games in TypeScript. Every frame would then cross the JS/WASM boundary several times (state text out, probabilities back, action in), and the game logic would exist twice, in Rust for training data and in TypeScript for the page. `sky-web` avoids both. The game the model plays in the browser is the same `sky-games` code that generated its training records and that `sky-arena` scores natively.

```text
JS (render loop)                      sky-web (WASM)
----------------                      -------------------------------------------------
game.step_ai(engine)  ───────────▶    state_text() → Engine::decide_raw (×2 questions)
                                      → argmax / abort → game.apply() (reflex + physics)
JSON {probs, action, …} ◀───────────  serde_json::to_string(&AiStep)
game.snapshot() → JSON view → draw
```

From `crates/sky-web/Cargo.toml`:

```toml
[dependencies]
sky-games = { path = "../sky-games" }
sky-infer = { path = "../sky-infer", features = ["wasm"] }
serde = { version = "1.0.219", features = ["derive"] }
serde_json = "1.0.145"
wasm-bindgen = "0.2.117"
```

## The module's layout

| file | role |
|---|---|
| `src/lib.rs` | the `#[wasm_bindgen]` surface: `Engine`, `GameSession`, `bench`, `benchGame`, `version` |
| `src/model.rs` | `Model`: a `sky_infer::Engine` loaded with f32 compute, plus the file size |
| `src/session.rs` | `Session`: one game, its questions, the decision schedule, timing statistics |
| `src/clock.rs` | a microsecond clock: `performance.now()` in WASM, `Instant` natively |

The Rust types (`Model`, `Session`, `session::bench`) work natively too, so `cargo test -p sky-web` runs every game with a model on the host. The `#[wasm_bindgen]` types are thin wrappers that turn `String` errors into JS exceptions.

## Loading the model

`Model::load` always asks the engine for f32 compute. As chapter 14 explains, baseline SIMD128 has no fast int8 dot product, so Q8_0 weights are dequantized at load, and the 2-bit weights of s2 and s3 files are expanded to f32 the same way.

From `crates/sky-web/src/model.rs`:

```rust
impl Model {
    pub fn load(gguf: &[u8]) -> Result<Model, String> {
        Self::load_with(gguf, EngineOptions::default())
    }

    /// Load with explicit engine options (`dequantize` is always set: f32 compute).
    pub fn load_with(gguf: &[u8], opts: EngineOptions) -> Result<Model, String> {
        let opts = EngineOptions {
            dequantize: true,
            ..opts
        };
        let inner = sky_infer::Engine::from_gguf_with(gguf, opts).map_err(|e| e.to_string())?;
        Ok(Model {
            inner,
            file_bytes: gguf.len(),
        })
    }
```

`Engine.info()` returns the model description as JSON (`name`, `n_params`, `hidden_size`, the fitted temperatures, `file_bytes`), and the page shows it next to the game. For s3-pico, the first model in `site/public/models/manifest.json` and the one the page loads by default, it starts with `{"name":"skycmd-s3-pico","arch":"skycmd-s2","n_params":445345,"vocab_size":1024,"hidden_size":96,...`; for s1-pico with `{"name":"skycmd-s1-pico","arch":"modern-bert","n_params":202305,...`.

## Questions come from the game

Each game core declares its System One questions as `QuestionSpec`s, with several English and French phrasings used for training augmentation. The session always asks the canonical phrasing (variant 0), rendered the same way as in the decision records.

From `crates/sky-web/src/session.rs`:

```rust
impl Question {
    fn from_spec(q: &QuestionSpec) -> Question {
        Question {
            id: q.id,
            qtype: QType::from_name(q.qtype.as_str()).expect("sky-games question types"),
            instructions: q.instructions(),
            keys: q.keys(),
            options: q.canonical_options(),
        }
    }
}
```

For Flappy Drone, `game.questions()` returns two questions:

```json
[{"id":"action","type":"choice","instructions":"Which input keeps the drone flying through the next gap?",
  "keys":["thrust","glide"],
  "options":["thrust: fire the rotors for an upward impulse","glide: let gravity pull the drone down"]},
 {"id":"danger","type":"noul","instructions":"Will the drone crash within the next half second if it keeps gliding?",
  "keys":["false","true"],
  "options":["false: no, the statement does not hold","true: yes, the statement holds"]}]
```

The model sees the state as compact text, for example `vy 0 dx 36 up 9 dn 7 nxt 10 alt 32`. `Flappy::state_text` measures clearances relative to the drone (`up`, `dn`), so the model reads "room above / below" directly: the code measures, the model decides.

Rescue asks `move` (a 7-way choice) and `risk` (a 5-level score). Dodge asks `maneuver` (a 4-way choice) and `abort` (a noul). The Doom-like game of [chapter 18](./18-doom) asks three questions. The first question is always the action. The second, called "secondary" in the JSON, is shown as a gauge; any further ones go to `others`.

## One AI frame

`Session::step_ai` is the heart of the module. When the game wants a decision (`decision_due()`) and the throttle allows it, the policy answers every question from the same state text. The primary argmax goes through the game's safety reflex and the physics advances one tick. Otherwise the last decision is held and the game ticks.

From `crates/sky-web/src/session.rs`:

```rust
        let mut decide = false;
        if self.game.decision_due() {
            decide = self.due_count.is_multiple_of(self.decide_every as u64);
            self.due_count += 1;
        }

        let outcome: StepOutcome;
        if decide {
            let state = self.game.state_text();
            let oracle = matches!(policy, Policy::Oracle).then(|| self.game.oracle());
            let mut stats = Vec::with_capacity(2);
            let p0 = self.ask(&mut policy, &oracle, 0, &state, &mut stats)?;
            let p1 = if self.questions.len() > 1 {
                Some(self.ask(&mut policy, &oracle, 1, &state, &mut stats)?)
            } else {
                None
            };
            let mut others = Vec::new();
            for qi in 2..self.questions.len() {
                let p = self.ask(&mut policy, &oracle, qi, &state, &mut stats)?;
                others.extend(self.secondary(qi, &p));
            }
            let info = match &policy {
                Policy::Model(m) => Some(DecisionInfo::from_stats(m, &stats)),
                Policy::Oracle => None,
            };
            let choice = argmax(&p0);
            let abort = self.kind == sky_games::dodge::NAME
                && p1.as_ref().is_some_and(|p| p.len() == 2 && p[1] > p[0]);
            outcome = self.game.apply(choice, abort);
```

Points worth noticing:

- **The model proposes, the code disposes.** `game.apply` runs the coded safety reflex for Rescue and Dodge (Flappy has none). When it replaces the requested move, `vetoed` is true and `executed` differs from `action`. The page can show both. `set_veto(false)` turns the reflex off, to watch the bare model.
- **Termination is an explicit action.** In Dodge, `abort` wins when the secondary noul question says `true` with p > 0.5 (`DroneCATS`, docs/research.md).
- **The oracle is a policy too.** `Policy::Oracle` returns the soft targets of the full simulator, which the training records also use. The page falls back to it when no model is available, for example before `models/` holds a trained model (`scripts/export-models.sh` prints "the site falls back to the oracle policy").
- **Decisions are not always due every frame.** Flappy decides every frame; Dodge decides every sixth frame (`dodge_decides_every_sixth_frame` in `crates/sky-web/tests/session.rs`).

The frame returns a serializable `AiStep`. A real Flappy frame with s3-pico (`GameSession("flappy", 42)`, 90th step, Node 24):

```json
{"policy":"model","decided":true,"action":"glide","executed":"glide",
 "probs":[{"key":"thrust","p":0.26800653},{"key":"glide","p":0.7319935}],
 "secondary":{"id":"danger","type":"noul","value":0.061188255,"max":1.0,
              "probs":[{"key":"false","p":0.9388117},{"key":"true","p":0.061188255}]},
 "others":[],
 "info":{"family":"s2","exit":0,"n_exits":4,"layers_used":1,"mean_layers":1.0,"max_layers":4,
         "static_hit":true,"state_reused":true},
 "vetoed":false,"micros":52.75,"state_text":"vy 9 dx 12 up 5 dn 11 nxt 10 alt 36",
 "event":"none","reward":0.01,"done":false,"score":0}
```

`micros` is the wall time of the whole frame inside WASM: decision plus physics. `info` says how the model answered: here s3-pico stopped at its first exit, after one pass of its shared block (an s1 model reports `"family":"s1"` and no exit). The same step with s1-pico took 588 µs.

## The frame budget and `set_decide_every`

At 60 Hz a frame has about 16.7 ms. Measured on an Apple M5 with Node 24 / V8, the `bench()` median per AI decision on Flappy (two forward passes plus physics, `wasm_us` in `site/public/bench.json`) is 49.6 µs for s3-pico, the default model, and 588.6 µs for s1-pico. Even the slower s1-pico uses under 4% of the budget, so the page runs WASM **on the main thread**, with no worker and no message passing.

A bigger model may not fit. The session can then decide on every n-th due frame and hold the last decision in between:

From `crates/sky-web/src/session.rs`:

```rust
    /// Ask the policy on every `n`-th frame where the game wants a decision; the frames in
    /// between hold the last decision (`tick`). Use it when a model is slower than the frame
    /// budget.
    pub fn set_decide_every(&mut self, n: u32) {
        self.decide_every = n.max(1);
    }
```

Held frames call `game.tick()`, the game's "do nothing" step: Flappy glides, Rescue hovers, Dodge keeps its maneuver. They repeat the last probabilities, so the gauges do not flicker. `GameSession.timing()` keeps a rolling window of the last 120 decision times (`{n, median_us, p95_us, mean_us}`). The page can read it and raise `decide_every` when the median approaches the budget.

| model | wasm decision (Flappy, 2 questions + physics) | share of a 16.7 ms frame | `decide_every` needed at 60 Hz |
|---|---:|---:|---:|
| s3-pico (default) | 49.6 µs | 0.3% | 1 |
| s1-pico | 588.6 µs | 3.5% | 1 |
| s1-nano (earlier engine) | 3,541 µs | 21% | 1 |
| s1-micro (earlier engine) | about 32,600 µs | about 195% | 2 or more |
| s2-nano (τ = 0.95) | 409.1 µs | 2.5% | 1 |
| s2-nano-doom (3 questions + physics) | 1,830 µs | 11% | 1 (a doom decision is due every 114 ms) |

The s3-pico, s1-pico and s2-nano numbers are the `wasm_us` medians of `site/public/bench.json`; s2-nano-doom is from the Doom section of `BENCH.md`. The s1-nano and s1-micro numbers were measured with the engine as it was before the s3 kernel work ([chapter 21](./21-s3-recursive-tiny)) and not re-measured; s1-micro was measured on a busy machine for [chapter 12](./12-size-ladder). s1-micro is the only model that would need throttling, and it is not shipped.

## Benchmarking inside the page

`bench(engine, n)` builds its own Flappy session (seed `0x5eed`) and warms the prefix and BPE caches with 8 frames. It then times decisions in batches of 10. Without cross-origin isolation, browsers coarsen `performance.now()` to 100 µs or more, so timing one sub-millisecond decision at a time would mostly measure that rounding.

From `crates/sky-web/src/session.rs`:

```rust
    while done < n {
        let k = batch.min(n - done);
        let t0 = now_us();
        let mut decided = 0;
        while decided < k {
            if s.done() {
                s.reset();
            }
            decided += s.step_ai(Policy::Model(model))?.decided as usize;
        }
        samples.push((now_us() - t0) / k as f64);
        done += k;
    }
```

It returns `{game, name, family, params, file_bytes, n, batch, median_us, p95_us, mean_us, decisions_per_s, forwards_per_decision, mean_layers_used, max_layers, static_hit_rate, state_reuse_rate}`. `benchGame(engine, game, n)` does the same for another game (`crates/sky-web/bench/bench-node.mjs --game doom` uses it). The `wasm_us` field of `site/public/bench.json` holds this number as measured under Node on the build machine (`scripts/export-models.sh` itself only measures native latency and writes `null` there), and the Bench page can also run `bench()` in your browser.

## The JavaScript side

The page code lives in the VitePress theme:

- `site/.vitepress/theme/lib/wasm.ts` loads `pkg/sky_web.js` once, fetches the GGUF files listed in `site/public/models/manifest.json`, and creates `Engine` and `GameSession` objects. The page starts with the first entry of the manifest, s3-pico.
- `site/.vitepress/theme/lib/render.ts` draws the `view` field of `game.snapshot()` on a canvas.
- `site/.vitepress/theme/components/PlayGame.vue` is the full game on `/play/` (human, model or oracle, veto toggle, probability bars, timing).
- `site/.vitepress/theme/components/HeroGame.vue` is the self-playing game on the home page.

The contract between Rust and the page is small:

```js
import init, { Engine, GameSession, bench } from "./pkg/sky_web.js";
await init();
const engine = new Engine(new Uint8Array(await (await fetch("s1-pico.gguf")).arrayBuffer()));
const game = new GameSession("flappy", 42);
const step = JSON.parse(game.step_ai(engine));   // or game.step_oracle() without a model
const view = JSON.parse(game.snapshot());
```

The snippet is the doc comment of `crates/sky-web/src/lib.rs`; any GGUF on the site (s1, s2 or s3) loads the same way. On the deployed site, every URL must carry the GitHub Pages base path `/skycmd/`; chapter 17 covers this. A human player calls `game.step_human(key)` with one of `game.action_keys()` (`""` holds), and the reflex applies to humans too.

## Testing without a browser

`crates/sky-web/tests/session.rs` drives every game natively. It covers the oracle finishing each game, throttling holding the last decision, human keys and the reflex, and a random-weight s1-pico answering every game with probabilities that sum to 1.

```sh
export CARGO_TARGET_DIR=$PWD/target/sky-web
cargo test -p sky-web
```

To score a trained model over many episodes without a page, `sky-arena` plays every game natively with the same engine. Its usage string is `sky-arena <model.gguf | laya_dir> [--episodes N] [--seed S] [--game G|all|a,b] [--json out.json] [--max-pipes N] [--exit-threshold T]`. `--exit-threshold` only matters for s2 and s3 files ([chapter 19](./19-s2-architecture)). `--game` also accepts `doom`, and `--seed` defaults to 7. The DAgger scripts also call it with `--dagger N --out DIR`, which is not in the usage line:

```sh
export CARGO_TARGET_DIR=$PWD/target/sky-arena
cargo run --release -p sky-arena -- site/public/models/s1-pico.gguf --episodes 30 --game flappy
```

Arena scores of the published models, 30 episodes per game with the default seed (`runs/ship-s3-pico.arena.json`, `runs/ship-s1-pico.arena.json`, `runs/ship-s2-nano.arena.json`; `runs/` is local and gitignored, `sky-arena --json` writes these files):

| model | Flappy pipes (crashes) | Rescue rate | Dodge completion | Dodge reflex vetoes | Flappy p50 |
|---|---:|---:|---:|---:|---:|
| oracle | 200.0 (0) | 1.00 | 0.87 | 0 | |
| s3-pico (0.45M, default, 3 DAgger rounds, τ = 0.6) | 200.0 (0) | 1.00 | 0.87 | 14 | 21 µs |
| s1-pico (0.2M, after 2 DAgger rounds) | 200.0 (0) | 1.00 | 0.83 | 1,026 | 201 µs |
| s2-nano (1.5M, after 2 DAgger rounds, τ = 0.95) | 200.0 (0) | 1.00 | 0.87 | 25 | 378 µs |

s3-pico also plays the Doom-like game in the same run: completion 1.00 and kill rate 0.77, both equal to the oracle's.

Flappy is capped at 200 pipes (`--max-pipes`), so 200.0 means every episode reached the cap. The other rungs of the ladder and every DAgger round are in [chapter 20](./20-dagger-lessons), and the s3-pico rounds in [chapter 21](./21-s3-recursive-tiny).

## Next

[16 Bonus: llama.cpp /v1/systemone and wllama](./16-llama-cpp-and-wllama)
