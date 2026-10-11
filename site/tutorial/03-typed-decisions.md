# 03 Typed decisions: schema, unknown and the veto rule

Every decision in skycmd is a typed question with a closed set of options. Abstaining is one of those options, and deterministic code reviews the answer before anything moves. This chapter walks through the `sky-schema` crate, the explicit `unknown` option in the Rescue game, and the veto rule that gives coded safety checks the last word.

## The three question types

`sky-schema` mirrors the llama.cpp `/v1/systemone` API. A request is a free-form `state` plus an ordered map of questions. Each question is tagged by its `type`:

`crates/sky-schema/src/systemone.rs`

```rust
#[serde(tag = "type", rename_all = "lowercase")]
pub enum Question {
    /// Pick one key. Descriptions may be `null`.
    Choice {
        instructions: String,
        criteria: IndexMap<String, Option<String>>,
    },
    /// Rate on 2..=10 ordered levels.
    Score {
        instructions: String,
        criteria: Vec<String>,
    },
    /// Probability that the statement in `instructions` holds.
    Noul {
        instructions: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        criteria: Option<NoulCriteria>,
    },
}
```

- **choice**: pick one key from a set. Keys keep their insertion order, so the crate uses `IndexMap` and enables `serde_json`'s `preserve_order` feature (see `crates/sky-schema/Cargo.toml`).
- **score**: rate on 2 to 10 ordered levels.
- **noul**: a probabilistic yes/no. The answer is p(true).

The type also picks a row of the decision head's type embedding (`QType::index`: choice 0, score 1, noul 2). The head therefore knows what kind of answer to produce before it scores any option.

A request trimmed from the crate's tests (`REQUEST` in `systemone.rs`):

```json
{"state":"alt 31 vy -2 dx 12 gt 38 gb 22 nc 30","questions":{"action":{"type":"choice","instructions":"Which input keeps the drone flying through the next gap?","criteria":{"thrust":"fire the rotors upward","glide":null}},"risk":{"type":"score","instructions":"How risky is the battery margin?","criteria":["negligible","low","critical"]}}}
```

## Rendering options

The model never sees the JSON. It sees option **texts**, built with the same rules as the reference runtime (`render_options` in `rl_common.py`):

`crates/sky-schema/src/record.rs`

```rust
/// Score options: `level i: desc`.
pub fn render_score<S: AsRef<str>>(levels: &[S]) -> Vec<String> {
    levels
        .iter()
        .enumerate()
        .map(|(i, c)| format!("level {i}: {}", c.as_ref()))
        .collect()
}

/// Noul options, always `[false, true]` so that `p[1]` is the noul answer. Missing or empty
/// descriptions fall back to the reference defaults.
pub fn render_noul(true_desc: Option<&str>, false_desc: Option<&str>) -> Vec<String> {
    let pick = |d: Option<&str>, default: &'static str| match d {
        Some(s) if !s.is_empty() => s.to_string(),
        _ => default.to_string(),
    };
    vec![
        format!("false: {}", pick(false_desc, NOUL_FALSE_DEFAULT)),
        format!("true: {}", pick(true_desc, NOUL_TRUE_DEFAULT)),
    ]
}
```

A choice renders as `key` or `key: desc`. A score renders as `level i: desc`. A noul is always `["false: ...", "true: ..."]`, and the defaults are `no, the statement does not hold` and `yes, the statement holds`. These strings must match the reference runtime exactly, because the model learns on these texts.

## Answers and confidence

The engine returns calibrated probabilities in option order, and `Answer::from_probs` shapes the reply ([chapter 01](./01-why-tiny-decision-models) shows the code). For a choice, confidence is the probability mass above uniform:

`crates/sky-schema/src/systemone.rs`

```rust
/// `(p_max - 1/n) / (1 - 1/n)`, clamped at 0.
pub fn confidence_choice(probs: &[f64]) -> f64 {
    if probs.len() < 2 {
        return 1.0;
    }
    let uniform = 1.0 / probs.len() as f64;
    let p_max = probs.iter().copied().fold(f64::MIN, f64::max);
    ((p_max - uniform) / (1.0 - uniform)).max(0.0)
}
```

For example, `[0.75, 0.25]` gives confidence `0.5`, and a uniform distribution gives `0`. A score returns the expected level `sum(i * p_i)`, and its confidence measures how concentrated the probabilities are around the mode. A noul returns `probs[1]`.

## The decision record

Training data uses one JSONL line per question. The record is the contract between the data producers (`sky-data`, `sky-games`) and `sky-train`:

`crates/sky-schema/src/record.rs`

```rust
pub struct DecisionRecord {
    /// Source tag, e.g. `boolq` or `game:flappy`.
    pub src: String,
    /// Question type.
    pub t: QType,
    /// Instructions (the question text).
    pub ins: String,
    /// Rendered option texts, in the order the model sees them.
    pub opts: Vec<String>,
    /// Free text or compact game state.
    pub state: String,
    /// Probability vector over `opts` (one-hot or soft).
    pub target: Vec<f32>,
    pub split: Split,
}
```

`opts` hold option texts that are already rendered. `target` can be soft: the game oracles emit soft targets, and some Hugging Face sets ship soft labels. `DecisionRecord::validate` enforces the invariants the trainer relies on:

- one target value per option;
- targets are finite, non-negative, and sum to 1 within `1e-3`;
- a choice has at least 2 options;
- score option `i` starts with `level i: `, and a score has 2 to 10 levels;
- a noul is exactly `false: ...` then `true: ...`.

A Flappy record written by `sky-games-gen` looks like this:

```json
{"src":"game:flappy","t":"noul","ins":"If no thrust is applied for half a second, does the drone hit something?","opts":["false: no, the statement does not hold","true: yes, the statement holds"],"state":"vy -1 dx 35 up 7 dn 9 nxt 1 alt 32","target":[0.95257413,0.047425874],"split":"train"}
```

## `unknown` is a real option

A model that must always pick a direction will invent one when its input is missing. In Rescue Run, the hiker sensor drops out on 12% of steps (`DROPOUT = 0.12`). When that happens, the state text shows `?` instead of the hiker costs. The `move` question therefore has a seventh option:

`crates/sky-games/src/rescue.rs`

```rust
            OptionSpec {
                key: "unknown",
                synonyms: &["unsure", "cannot_tell", "inconnu"],
                desc_en: &[
                    "the sensor data is missing, the right move cannot be decided",
                    "not enough information",
                    "cannot tell from the readings",
                ],
                desc_fr: &["les données du capteur manquent", "impossible de décider"],
            },
```

The oracle teaches the model when to abstain and when abstaining is no longer safe:

`crates/sky-games/src/rescue.rs`

```rust
        let home_here = self.home_cost();
        if !self.sensor_ok {
            if home_here + MARGIN > self.battery {
                p[ret] = 1.0;
            } else {
                p[RescueAction::Unknown.index()] = 1.0;
            }
            return with_floor(p, 0.01);
        }
```

With enough battery, the target is `unknown`. When the battery barely covers the trip home, the target becomes `return_home`, because waiting is no longer free. In the simulator, `unknown` costs the same as a hover (one battery unit away from base). The game holds position until the sensor recovers.

The same idea appears in Dodge's `abort` question and in the drone command plans, where `end`, `abort`, `clarify` and `refuse` are explicit actions (`crates/sky-schema/src/plan.rs`). `docs/research.md` explains why: small models struggle with termination unless it is a first-class choice.

## The veto rule

The model proposes and the code decides. Constrained outputs remove structural errors, since the head can only pick a listed option, but they cannot remove semantic ones. So every action passes through a deterministic check that may replace it. skycmd applies this rule in two places.

### Game reflexes

Rescue and Dodge each have a `veto` function. Rescue's version blocks flights into rocks and any move that would leave too little battery to get home:

`crates/sky-games/src/rescue.rs`

```rust
    pub fn veto(&self, action: RescueAction) -> Option<RescueAction> {
        let (after_pos, cost) = match action {
            RescueAction::ReturnHome => return None,
            RescueAction::Hover | RescueAction::Unknown => (self.pos, 1),
            a => {
                let d = a.dir().expect("move action");
                match self.blocked(self.pos, d) {
                    Some(j) => (j, self.enter_cost(j, d)),
                    None => {
                        let hover_ok = self.pos == self.base || self.battery > self.home_cost();
                        return Some(if hover_ok {
                            RescueAction::Hover
                        } else {
                            RescueAction::ReturnHome
                        });
                    }
                }
            }
        };
```

`step` runs the veto before any physics happens, records it, and reports it in `StepInfo { event, vetoed }`. The UI can show the requested action and the executed action side by side. Dodge's reflex simulates the chosen maneuver over the stopping horizon and falls back to the first safe option among `brake`, `over`, `around` and `continue`. It never vetoes `abort`.

The integration tests in `crates/sky-games/tests/games.rs` show what the reflex buys. Random play on Rescue with the reflex turned off must crash in at least 20 of 30 episodes. With the reflex on, random play must trigger at least 50 more vetoes than the oracle, and on Dodge the oracle may trigger at most 3. In closed loop, s1-pico played 30 Rescue episodes with 130 vetoes and no crashes (`runs/s1-pico/arena.json`).

### The command-plan validator

For free-form drone plans, `sky-schema::validate` sits between the plan decoder and the flight controller. It never rejects a plan outright. It returns a corrected copy that is always safe to execute, plus the list of violations:

`crates/sky-schema/src/plan.rs`

```rust
/// Checks a plan against the limits and returns a corrected copy.
///
/// Rules: at most [`MAX_ACTIONS`] actions; non-finite numbers are dropped; altitudes are clamped
/// to `[0, max_alt_m]` and speeds to `[0, max_speed_mps]`; any step that would leave the
/// geofence is dropped; nothing runs after `end`/`abort`; below `min_battery_pct` the plan is
/// replaced by `return_home` (an explicit `land`/`abort`/`return_home` plan is kept).
pub fn validate(plan: &Plan, telemetry: &Telemetry, limits: &SafetyLimits) -> Verdict {
```

The default limits are 120 m altitude, 15 m/s speed, a 500 m geofence and 25% minimum battery (`SafetyLimits::default`). The validator tracks position and altitude step by step, so a relative `move` that would cross the geofence is dropped even when each step looks harmless on its own.

## Try it

```bash
export CARGO_TARGET_DIR=$PWD/target/sky-schema
cargo test -p sky-schema

export CARGO_TARGET_DIR=$PWD/target/sky-games
cargo test -p sky-games --test games rescue_oracle_beats_random
cargo test -p sky-games veto
```

The `sky-schema` tests cover JSON round-trips that preserve question order, the confidence formulas, altitude and speed clamping, geofence drops, the low-battery override and truncation after `end`.

## Next

[04 Game cores and oracles](./04-game-cores-and-oracles)
