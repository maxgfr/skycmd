//! Deterministic game cores for skycmd's System One agents.
//!
//! Three seedable games with fixed-timestep `f32` physics and no I/O, threads or clocks, so
//! the browser can step them at 60 Hz through WASM:
//! - [`flappy`]: "Flappy Drone", a decision every frame (`thrust` / `glide`).
//! - [`rescue`]: "Rescue Run", a grid mission with battery management and sensor dropouts.
//! - [`dodge`]: "Obstacle Dodge", a side-scroller with maneuvers and a coded safety reflex.
//!
//! Each game exposes its System One questions, a compact state text, and an oracle computed
//! from the full simulator. [`rollout`] evaluates policies; [`gen`] builds training records.

pub mod common;
pub mod dodge;
pub mod flappy;
pub mod gen;
pub mod rescue;
pub mod rollout;

pub use common::{
    DecisionGame, EpisodeStats, Event, OptionSpec, QuestionSpec, StepInfo, StepOutcome,
};
pub use dodge::{Dodge, DodgeAction};
pub use flappy::{Flappy, FlappyAction};
pub use rescue::{Rescue, RescueAction};
pub use rollout::{
    evaluate, evaluate_with, make_game, EvalOptions, Metrics, OraclePolicy, Policy, RandomPolicy,
    GAMES,
};
