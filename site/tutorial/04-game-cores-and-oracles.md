# 04 Game cores and oracles

The skycmd games are pure-Rust simulators. They expose their state as compact text, ask typed questions, and come with an oracle that reads the full simulator to produce soft targets. This chapter covers the `sky-games` crate: the shared game interface, the three drone games and their oracles, the dataset generator, and the rollout harness that tests whether a policy can actually play.

## Design constraints

The games run in three places: the browser at 60 Hz (through `sky-web`), the native evaluator (`sky-arena`), and the dataset generator. That rules out clocks, threads and I/O inside a game:

`crates/sky-games/src/lib.rs`

```rust
//! Deterministic game cores for skycmd's System One agents.
//!
//! Four seedable games with fixed-timestep `f32` physics and no I/O, threads or clocks, so
//! the browser can step them in real time through WASM:
//! - [`flappy`]: "Flappy Drone", a decision every frame (`thrust` / `glide`).
//! - [`rescue`]: "Rescue Run", a grid mission with battery management and sensor dropouts.
//! - [`dodge`]: "Obstacle Dodge", a side-scroller with maneuvers and a coded safety reflex.
//! - [`doom`]: a Doom-like first-person shooter on a grid at 35 Hz, a decision every 4 ticks.
//!
//! Each game exposes its System One questions, a compact state text, and an oracle computed
//! from the full simulator. [`rollout`] evaluates policies; [`gen`] builds training records.
```

The fourth game, the Doom-like shooter, has its own chapter ([chapter 18](./18-doom)); this chapter covers the three drone games. Their timestep is `DT = 1.0 / 60.0` (`crates/sky-games/src/common.rs`). Each game owns a `SmallRng`, and `reset()` derives the next course from `(seed, episode)` with splitmix64. The same seed therefore replays the same courses on every platform. `rand` is built without its `std` feature so that the crate compiles for `wasm32-unknown-unknown` (`crates/sky-games/Cargo.toml`).

## One interface for every game

The rollout harness, the generator and the web session all talk to a game through one object-safe trait:

`crates/sky-games/src/common.rs`

```rust
pub trait DecisionGame {
    fn name(&self) -> &'static str;
    /// Starts a new episode (a new course drawn from the game's seeded RNG).
    fn reset(&mut self);
    fn done(&self) -> bool;
    /// True when the agent should be asked for a new decision before the next tick.
    fn decision_due(&self) -> bool;
    /// Questions asked at each decision; index 0 is the primary (action) question.
    fn questions(&self) -> &'static [QuestionSpec];
    fn state_text(&self) -> String;
    /// Per-question soft targets over canonical options, in [`questions`](Self::questions) order.
    fn oracle(&self) -> Vec<Vec<f32>>;
    /// Applies a decision: `choice` indexes the primary question's canonical options, `abort`
    /// is the answer to an abort question (ignored by games without one). Advances one tick.
    fn apply(&mut self, choice: usize, abort: bool) -> StepOutcome;
    /// Advances one tick, holding the last decision.
    fn tick(&mut self) -> StepOutcome;
    fn stats(&self) -> EpisodeStats;
    /// Enables or disables the safety reflex (no-op for games without one).
    fn set_veto(&mut self, _on: bool) {}
    /// Render data as JSON.
    fn snapshot_json(&self) -> serde_json::Value;
}
```

Each drone game asks exactly two questions: a primary action question and a secondary one (the Doom-like game asks three). Each `QuestionSpec` carries English and French instruction variants, and each `OptionSpec` carries synonyms and description variants. Index 0 is always the canonical wording, which the runtime and the UI use.

| game | primary question | secondary question | decision rate |
|---|---|---|---|
| flappy | `action` choice: `thrust`, `glide` | `danger` noul | every frame |
| rescue | `move` choice: `north`, `south`, `east`, `west`, `hover`, `return_home`, `unknown` | `risk` score, 5 levels | every step |
| dodge | `maneuver` choice: `over`, `around`, `brake`, `continue` | `abort` noul | every `DECISION_EVERY = 6` frames |

## State text: binned integers

Every value in a state text goes through one of two helpers, so it stays a small integer:

`crates/sky-games/src/common.rs`

```rust
/// Rounds and clamps a value for the state text.
pub(crate) fn bin(v: f32, scale: f32, lo: i32, hi: i32) -> i32 {
    ((v / scale).round() as i32).clamp(lo, hi)
}

/// Floors and clamps a value for the state text.
pub(crate) fn bin_floor(v: f32, scale: f32, lo: i32, hi: i32) -> i32 {
    ((v / scale).floor() as i32).clamp(lo, hi)
}
```

The test `state_text_fits_token_budget` (`crates/sky-games/tests/games.rs`) plays 3000 mixed oracle/random decisions per game. It checks that every state text has at most 24 pieces (30 for Doom), that the pieces form key/value pairs, and that every value is either an integer in `-20..=99` or a token of at most two characters, such as `x`, `?` or `ok`.

## Flappy Drone

The drone sits at a fixed x position. Gravity pulls it down (`GRAVITY = 900.0` px/s²) and `thrust` sets its vertical speed to `THRUST_V = 320.0`. Pipe pairs scroll by with a gap of `GAP_H = 150.0` px. The state text is `vy dx up dn nxt alt` in 8 px units, with clearances measured relative to the drone ([chapter 01](./01-why-tiny-decision-models) shows the code).

### The oracle: a pruned lookahead

Pipes are stored in world coordinates, so the simulator can predict any future frame exactly. The oracle searches a binary tree of thrust/glide plans over `HORIZON = 40` frames. Inside a plan, two thrusts must be at least `MIN_THRUST_GAP = 8` frames apart. A plan's value is its worst safety margin (capped), minus small penalties for being off-center and for thrusting. The two action values then become a soft target:

`crates/sky-games/src/flappy.rs`

```rust
    /// Soft targets: `action` over `[thrust, glide]`, `danger` over `[false, true]`.
    pub fn oracle(&self) -> Vec<Vec<f32>> {
        let [vt, vg] = self.action_values();
        let p_thrust = sigmoid(((vt - vg) / TARGET_T).clamp(-3.0, 3.0));
        let crash = self.glide_crash_frame(60).map_or(60.0, |f| f as f32);
        let p_danger = sigmoid(((DANGER_FRAMES + 0.5 - crash) / 2.0).clamp(-3.0, 3.0));
        vec![
            vec![p_thrust, 1.0 - p_thrust],
            vec![1.0 - p_danger, p_danger],
        ]
    }
```

Clamping the logit to ±3 keeps every target strictly between 0.047 and 0.953. When thrusting now and thrusting a frame later are almost equally good, the target says so, and the model is not pushed to be overconfident about a coin flip. The `danger` horizon is half a second (`DANGER_FRAMES = 30.0`). The code comment explains that a full second of gliding almost always ends on the ground, which would make the label nearly constant.

## Rescue Run

The map is a 12×12 grid (`N = 12`) with rocks, wind cells, a base and 3 hikers. The battery holds `BATTERY_MAX = 40` units. Moving against the wind costs 3 units and every other move costs 1. The base recharges the battery. Map generation rejects any layout where a hiker round trip plus `MARGIN = 3` exceeds the battery.

The oracle precomputes Dijkstra cost maps to the base and to each hiker. For each of the four moves, the state text then reports two numbers: the cost of the cheapest "rescue a hiker, then fly home" trip through that move, and the cost of flying home through it:

`crates/sky-games/src/rescue.rs`

```rust
    /// For each move `n s e w`: `<m>h` = cost of rescuing the cheapest hiker through that move
    /// and flying home (`?` on sensor dropout, `-` when no hiker is left), `<m>b` = cost of
    /// flying home through that move (`x` = blocked); then battery, wind at the drone, hikers
    /// left and sensor status.
    pub fn state_text(&self) -> String {
        let f = self.features();
        let hk = self.hikers_left();
        let mut s = String::with_capacity(96);
        for d in DIRS {
            let m = f[d.index()];
            let l = d.letter();
            if m.blocked {
                s.push_str(&format!("{l}h x {l}b x "));
                continue;
            }
```

The `move` target is a softmax over trip costs, with temperature 0.6, scaled by the probability that the battery covers the best trip plus the margin. Whatever probability is left goes to `return_home`, or to `hover` when the drone is already at base. The `risk` score maps the battery margin onto a continuous level in `[0, 4]` through piecewise-linear knots, then spreads it over the 5 levels with a Gaussian of width 0.45. [Chapter 03](./03-typed-decisions) covers the `unknown` branch and the veto.

## Obstacle Dodge

The drone flies forward through a corridor (`|y| <= Y_MAX = 5.0` m) at a cruise speed of `CRUISE_V = 11.0` m/s. Each obstacle has a height, may sit under a deck (a low ceiling such as a bridge), and may leave a lateral gap. An obstacle is impassable when it leaves neither a vertical slot nor a wide enough gap:

`crates/sky-games/src/dodge.rs`

```rust
    /// Free vertical slot between obstacle + margin and deck - margin (negative = too tight).
    fn slot(&self) -> f32 {
        (self.ceil - MARGIN - HH) - (self.top + MARGIN + HH)
    }
    fn over_ok(&self) -> bool {
        self.slot() >= 0.0
    }
    fn around_ok(&self) -> bool {
        self.gap >= 2.0 * (HW + MARGIN)
    }
    pub fn impassable(&self) -> bool {
        !self.over_ok() && !self.around_ok()
    }
```

The state text `d v dz hr gw dy cl` gives the distance, speed, climb needed, free slot, usable gap, lateral offset to the gap, and clearance under the deck. When the obstacle ahead is impassable, the oracle puts 0.95 on `abort` and recommends `brake`. Otherwise it compares the time slack of going over against going around, and keeps `continue` only when the drone is already clear or has more than enough time. The secondary `abort` question gives the model an explicit way to end the mission, and an abort counts as safe only in front of an impassable obstacle (`EpisodeStats::safe_abort`).

## From oracle to dataset

`gen.rs` plays episodes and labels every visited decision with the oracle. Four techniques keep the data from being too easy:

1. **Noisy play (DAgger style).** Each episode draws an exploration rate from `EPSILONS = [0.0, 0.05, 0.15, 0.3]`. Off-course states are still labeled by the oracle, so the model learns how to recover.
2. **Paraphrase, rename, shuffle.** 30% of records are French (`FRENCH_SHARE`). Choice keys are swapped for synonyms half the time, and option order is shuffled. The target follows the shuffled order through `RenderedQuestion::map_target`. This counters the label-latching effect described in `docs/research.md`.
3. **Class balancing.** Rare decisions such as `thrust`, `brake` and `abort` are kept more often than common ones (`BALANCE_POWER = 0.3`).
4. **Splits by episode seed.** All records from one episode land in the same split:

`crates/sky-games/src/gen.rs`

```rust
/// Split of an episode seed: seeds `≡ 18 (mod 20)` are validation, `≡ 19` test, others train.
pub fn split_for_seed(episode_seed: u64) -> Split {
    match episode_seed % 20 {
        18 => Split::Val,
        19 => Split::Test,
        _ => Split::Train,
    }
}
```

The secondary question gets 30% of records (`SECONDARY_SHARE`), and Flappy episodes are capped at 12 pipes so that the data covers many different courses.

### Generating records

The binary takes four flags:

`crates/sky-games/src/bin/sky-games-gen.rs`

```rust
const USAGE: &str =
    "usage: sky-games-gen --game {flappy,rescue,dodge,doom,all} --n N --seed S [--out PATH]";
```

(`doom` is the fourth game, added in [chapter 18](./18-doom).)

```bash
export CARGO_TARGET_DIR=$PWD/target/sky-games
cargo run --release -p sky-games --bin sky-games-gen -- --game all --n 20000 --seed 1
cargo run --release -p sky-games --bin sky-games-gen -- --game rescue --n 5000 --seed 7 --out /tmp/rescue.jsonl
```

With `all`, `--out` is a directory (default `data/games`) and `N` records are written per game. Label-balance statistics are printed to stderr. The s1 and s2 models were trained on 90,000 Flappy, 60,000 Rescue and 60,000 Dodge records, 210,000 in total, of which 189,880 are in the train split (`data/STATS.md`, written by `sky-data stats`, "Decision records (games)"). `scripts/train-doom.sh` later added 100,000 Doom records (`data/games/doom.jsonl`), which s3-pico also trained on ([chapter 21](./21-s3-recursive-tiny)).

## The rollout harness

A policy sees exactly what the model sees, which is the text:

`crates/sky-games/src/rollout.rs`

```rust
pub trait Policy {
    fn decide(
        &mut self,
        game: &str,
        question_id: &str,
        state_text: &str,
        rendered_opts: &[String],
    ) -> Vec<f32>;

    /// Called with the full game before each decision. Text-only policies ignore it; the oracle
    /// baseline uses it to read the simulator.
    fn observe(&mut self, _game: &dyn DecisionGame) {}
}
```

`evaluate_with` plays N episodes and reports `Metrics`: mean score, rescue rate, completion rate, crashes, vetoes, aborts and safe aborts. `OraclePolicy` and `RandomPolicy` serve as the upper and lower baselines:

```bash
cargo run --release -p sky-games --example eval
```

`sky-arena` wraps the inference engine in the same `Policy` trait and plays real models:

```bash
export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer
export CARGO_TARGET_DIR=$PWD/target/sky-arena
cargo run --release -p sky-arena -- models/s1-pico --episodes 30 --json runs/s1-pico/arena.json
```

For s1-pico over 30 episodes, the model against the oracle (`runs/s1-pico/arena.json`):

| game | model | oracle |
|---|---|---|
| flappy | mean score 0.0, 30 crashes | mean score 200.0 (capped), 0 crashes |
| rescue | rescue rate 0.63, completion 0.57, 130 vetoes | rescue rate 1.0, completion 1.0 |
| dodge | completion 0.67, 4 safe aborts, 0 crashes | completion 0.87, 4 safe aborts |

The gap between test accuracy and closed-loop play is what `scripts/specialize.sh` targets. It runs `sky-arena <model> --dagger N` to collect oracle labels on the states the model actually reaches, then fine-tunes on them. After two rounds of 6 minutes each, the same s1-pico closes the gap almost entirely (`runs/s1-pico-g2/arena.json`, 30 episodes):

| game | before DAgger | after 2 rounds (shipped) | oracle |
|---|---|---|---|
| flappy | 0.0 pipes, 30 crashes | 200.0 pipes, 0 crashes | 200.0 pipes |
| rescue | rescue rate 0.63, 130 vetoes | rescue rate 1.0, 0 vetoes | 1.0 |
| dodge | completion 0.67, 5,828 vetoes | completion 0.83, 1,026 vetoes | 0.87 |

[Chapter 20](./20-dagger-lessons) tells the whole story, including the state-text fix that had to come first and the round that made things worse.

## Next

[05 Pulling data from Hugging Face in Rust](./05-hugging-face-data)
