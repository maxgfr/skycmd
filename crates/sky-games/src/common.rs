//! Types shared by the three games: step outcomes, question specs with paraphrases, and the
//! object-safe [`DecisionGame`] interface used by the rollout harness and the generator.

use serde::Serialize;
use sky_schema::{render_choice, render_noul, render_score, QType, Question};

/// Fixed physics timestep (60 Hz).
pub const DT: f32 = 1.0 / 60.0;

/// What happened during a step.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Event {
    None,
    /// Flappy: a pipe was passed. Dodge: an obstacle was passed.
    Passed,
    Crashed,
    /// Rescue: a hiker was picked up.
    Rescued,
    /// Rescue: battery recharged at base.
    Recharged,
    /// The mission was completed.
    Completed,
    /// Dodge: the drone stopped and landed after an abort.
    Aborted,
    /// Step budget exhausted.
    Timeout,
}

/// Extra information about a step.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct StepInfo {
    pub event: Event,
    /// True when the safety reflex replaced the requested action.
    pub vetoed: bool,
}

/// Result of one step.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct StepOutcome {
    pub reward: f32,
    pub done: bool,
    pub info: StepInfo,
}

impl StepOutcome {
    pub(crate) fn new(reward: f32, done: bool, event: Event, vetoed: bool) -> Self {
        StepOutcome {
            reward,
            done,
            info: StepInfo { event, vetoed },
        }
    }
}

/// One option of a question, with its paraphrases.
///
/// For score questions the key is the level index; for noul questions the two options are
/// `false` then `true`, and an empty description means "use the reference default".
#[derive(Debug, Clone, Copy)]
pub struct OptionSpec {
    /// Canonical key (what the game acts on).
    pub key: &'static str,
    /// Alternative keys for renaming (choice only). Disjoint across options of a question.
    pub synonyms: &'static [&'static str],
    /// English descriptions; index 0 is the canonical one.
    pub desc_en: &'static [&'static str],
    /// French descriptions.
    pub desc_fr: &'static [&'static str],
}

/// A question asked at decision time, with its EN/FR paraphrases.
#[derive(Debug, Clone, Copy)]
pub struct QuestionSpec {
    pub id: &'static str,
    pub qtype: QType,
    /// English instructions; index 0 is canonical (used by the runtime and the UI).
    pub ins_en: &'static [&'static str],
    /// French instructions.
    pub ins_fr: &'static [&'static str],
    pub options: &'static [OptionSpec],
}

impl QuestionSpec {
    /// Canonical instructions (variant 0).
    pub fn instructions(&self) -> &'static str {
        self.ins_en[0]
    }

    /// Canonical option keys, in canonical order.
    pub fn keys(&self) -> Vec<&'static str> {
        self.options.iter().map(|o| o.key).collect()
    }

    /// Canonical rendered option texts (variant 0), in canonical order.
    pub fn canonical_options(&self) -> Vec<String> {
        match self.qtype {
            QType::Choice => {
                let items: Vec<(&str, &str)> =
                    self.options.iter().map(|o| (o.key, o.desc_en[0])).collect();
                render_choice(&items)
            }
            QType::Score => {
                let levels: Vec<&str> = self.options.iter().map(|o| o.desc_en[0]).collect();
                render_score(&levels)
            }
            QType::Noul => render_noul(
                Some(self.options[1].desc_en[0]),
                Some(self.options[0].desc_en[0]),
            ),
        }
    }

    /// The canonical question as a `/v1/systemone` question.
    pub fn to_systemone(&self) -> Question {
        let instructions = self.instructions().to_string();
        match self.qtype {
            QType::Choice => Question::Choice {
                instructions,
                criteria: self
                    .options
                    .iter()
                    .map(|o| {
                        let d = o.desc_en[0];
                        (o.key.to_string(), (!d.is_empty()).then(|| d.to_string()))
                    })
                    .collect(),
            },
            QType::Score => Question::Score {
                instructions,
                criteria: self
                    .options
                    .iter()
                    .map(|o| o.desc_en[0].to_string())
                    .collect(),
            },
            QType::Noul => {
                let desc = |i: usize| {
                    let d = self.options[i].desc_en[0];
                    (!d.is_empty()).then(|| d.to_string())
                };
                let (f, t) = (desc(0), desc(1));
                Question::Noul {
                    instructions,
                    criteria: (f.is_some() || t.is_some()).then_some(sky_schema::NoulCriteria {
                        true_desc: t,
                        false_desc: f,
                    }),
                }
            }
        }
    }
}

/// Summary of the current episode.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize)]
pub struct EpisodeStats {
    /// Flappy: pipes passed. Rescue: hikers rescued. Dodge: obstacles passed.
    pub score: f32,
    /// Mission completed (rescue: all hikers home; dodge: finish line reached).
    pub success: bool,
    pub crashed: bool,
    pub aborted: bool,
    /// Dodge: the abort happened in front of an impassable obstacle.
    pub safe_abort: bool,
    pub vetoes: u32,
    pub steps: u32,
}

/// Object-safe view of a game, used by the rollout harness and the dataset generator.
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

/// Index of the largest value (first one on ties).
pub fn argmax(v: &[f32]) -> usize {
    let mut best = 0;
    for (i, x) in v.iter().enumerate() {
        if *x > v[best] {
            best = i;
        }
    }
    best
}

pub(crate) fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Adds a probability floor to every entry and renormalizes.
pub(crate) fn with_floor(mut p: Vec<f32>, floor: f32) -> Vec<f32> {
    let sum: f32 = p.iter().sum();
    let n = p.len() as f32;
    for x in p.iter_mut() {
        *x = (*x / sum) * (1.0 - floor * n) + floor;
    }
    p
}

/// Mixes a game seed and an episode counter into a fresh RNG seed (splitmix64).
pub(crate) fn mix_seed(seed: u64, episode: u64) -> u64 {
    let mut z = seed ^ episode.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Rounds and clamps a value for the state text.
pub(crate) fn bin(v: f32, scale: f32, lo: i32, hi: i32) -> i32 {
    ((v / scale).round() as i32).clamp(lo, hi)
}

/// Floors and clamps a value for the state text.
pub(crate) fn bin_floor(v: f32, scale: f32, lo: i32, hi: i32) -> i32 {
    ((v / scale).floor() as i32).clamp(lo, hi)
}
