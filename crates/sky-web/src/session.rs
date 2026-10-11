//! A game session: one game core plus its System One questions, stepped by a human, by the
//! model or by the oracle. A full AI frame (state text, tokenization, forward passes, readout,
//! action, safety reflex, physics) runs here, without leaving WASM.

use serde::Serialize;
use serde_json::{json, Value};
use sky_games::common::argmax;
use sky_games::{make_game, DecisionGame, QuestionSpec, StepOutcome, GAMES};
use sky_infer::QType;

use crate::clock::{now_us, summarize};
use crate::model::Model;

/// Number of recent decision timings kept for the rolling statistics.
const TIMING_WINDOW: usize = 120;

/// One System One question of a game, in its canonical (variant 0) phrasing.
#[derive(Debug, Clone)]
pub struct Question {
    pub id: &'static str,
    pub qtype: QType,
    pub instructions: &'static str,
    /// Canonical option keys (what the game acts on).
    pub keys: Vec<&'static str>,
    /// Rendered option texts, as the model sees them.
    pub options: Vec<String>,
}

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

/// Who answers the questions in [`Session::step_ai`].
pub enum Policy<'a> {
    Model(&'a mut Model),
    /// The game's oracle (soft targets from the full simulator): the fallback without a model.
    Oracle,
}

impl Policy<'_> {
    fn name(&self) -> &'static str {
        match self {
            Policy::Model(_) => "model",
            Policy::Oracle => "oracle",
        }
    }
}

/// Probability of one option.
#[derive(Debug, Clone, Serialize)]
pub struct OptionProb {
    pub key: &'static str,
    pub p: f32,
}

/// Answer to the secondary question (flappy `danger`, rescue `risk`, dodge `abort`).
#[derive(Debug, Clone, Serialize)]
pub struct Secondary {
    pub id: &'static str,
    #[serde(rename = "type")]
    pub qtype: &'static str,
    /// noul: p(true); score: expected level `sum(i * p_i)`; choice: p of the argmax.
    pub value: f32,
    /// Maximum of `value` (1 for noul, `n - 1` for score), for gauges.
    pub max: f32,
    pub probs: Vec<OptionProb>,
}

/// Result of one AI frame.
#[derive(Debug, Clone, Serialize)]
pub struct AiStep {
    pub policy: &'static str,
    /// False when this frame held the previous decision (no decision due, or throttled).
    pub decided: bool,
    /// The action the policy picked (primary question argmax, or `abort`).
    pub action: &'static str,
    /// The action the game executed (differs from `action` when the reflex fired).
    pub executed: String,
    pub probs: Vec<OptionProb>,
    pub secondary: Option<Secondary>,
    /// True when the coded safety reflex replaced the requested action on this frame.
    pub vetoed: bool,
    /// Wall time of the whole frame inside WASM (decision + physics), in microseconds.
    pub micros: f64,
    pub state_text: String,
    pub event: sky_games::Event,
    pub reward: f32,
    pub done: bool,
    pub score: u32,
}

/// Result of one human frame.
#[derive(Debug, Clone, Serialize)]
pub struct HumanStep {
    pub executed: String,
    pub vetoed: bool,
    pub event: sky_games::Event,
    pub reward: f32,
    pub done: bool,
    pub score: u32,
}

/// The last decision, repeated on frames that hold it.
#[derive(Debug, Clone)]
struct Held {
    action: &'static str,
    probs: Vec<OptionProb>,
    secondary: Option<Secondary>,
    state_text: String,
}

pub struct Session {
    kind: &'static str,
    game: Box<dyn DecisionGame>,
    questions: Vec<Question>,
    decide_every: u32,
    due_count: u64,
    held: Option<Held>,
    timings: Vec<f64>,
    timing_pos: usize,
}

impl Session {
    /// A new session of `kind` (`flappy`, `rescue` or `dodge`).
    pub fn new(kind: &str, seed: u64) -> Result<Session, String> {
        let kind = GAMES
            .iter()
            .copied()
            .find(|g| *g == kind)
            .ok_or_else(|| format!("unknown game {kind:?} (expected one of {GAMES:?})"))?;
        let game = make_game(kind, seed).expect("known game");
        let questions = game.questions().iter().map(Question::from_spec).collect();
        Ok(Session {
            kind,
            game,
            questions,
            decide_every: 1,
            due_count: 0,
            held: None,
            timings: Vec::with_capacity(TIMING_WINDOW),
            timing_pos: 0,
        })
    }

    pub fn kind(&self) -> &'static str {
        self.kind
    }

    pub fn questions(&self) -> &[Question] {
        &self.questions
    }

    /// Ask the policy on every `n`-th frame where the game wants a decision; the frames in
    /// between hold the last decision (`tick`). Use it when a model is slower than the frame
    /// budget.
    pub fn set_decide_every(&mut self, n: u32) {
        self.decide_every = n.max(1);
    }

    pub fn decide_every(&self) -> u32 {
        self.decide_every
    }

    /// Enable or disable the coded safety reflex (rescue, dodge; flappy has none).
    pub fn set_veto(&mut self, on: bool) {
        self.game.set_veto(on);
    }

    pub fn done(&self) -> bool {
        self.game.done()
    }

    pub fn score(&self) -> u32 {
        self.game.stats().score as u32
    }

    /// Start a new episode (a new course from the session's seeded generator).
    pub fn reset(&mut self) {
        self.game.reset();
        self.held = None;
        self.due_count = 0;
    }

    pub fn state_text(&self) -> String {
        self.game.state_text()
    }

    /// Render data: the game's snapshot plus score and episode statistics.
    pub fn snapshot(&self) -> Value {
        json!({
            "kind": self.kind,
            "score": self.score(),
            "done": self.game.done(),
            "stats": self.game.stats(),
            "view": self.game.snapshot_json(),
        })
    }

    /// Rolling statistics over the last decisions.
    pub fn timing(&self) -> Value {
        let (median, p95, mean) = summarize(&self.timings);
        json!({ "n": self.timings.len(), "median_us": median, "p95_us": p95, "mean_us": mean })
    }

    fn record_timing(&mut self, us: f64) {
        if self.timings.len() < TIMING_WINDOW {
            self.timings.push(us);
        } else {
            self.timings[self.timing_pos] = us;
            self.timing_pos = (self.timing_pos + 1) % TIMING_WINDOW;
        }
    }

    /// The action keys a human can send: the primary options, plus `abort` for dodge.
    pub fn action_keys(&self) -> Vec<&'static str> {
        let mut keys = self.questions[0].keys.clone();
        if self.kind == sky_games::dodge::NAME {
            keys.push("abort");
        }
        keys
    }

    /// The action actually executed after `requested` (reads the snapshot when the reflex fired).
    fn executed(&self, requested: &str, vetoed: bool) -> String {
        if !vetoed {
            return requested.to_string();
        }
        let field = match self.kind {
            "rescue" => "last_action",
            _ => "action",
        };
        self.game
            .snapshot_json()
            .get(field)
            .and_then(Value::as_str)
            .unwrap_or(requested)
            .to_string()
    }

    /// One frame driven by a human. An empty key (or `none`) holds: flappy glides, rescue
    /// hovers, dodge keeps its maneuver. The reflex still applies.
    pub fn step_human(&mut self, key: &str) -> Result<HumanStep, String> {
        let key = key.trim();
        let outcome = if key.is_empty() || key == "none" {
            self.game.tick()
        } else if self.kind == sky_games::dodge::NAME && key == "abort" {
            self.game.apply(0, true)
        } else {
            let i = self.questions[0]
                .keys
                .iter()
                .position(|k| *k == key)
                .ok_or_else(|| {
                    format!(
                        "unknown action {key:?} for {} (expected one of {:?})",
                        self.kind,
                        self.action_keys()
                    )
                })?;
            self.game.apply(i, false)
        };
        let shown = if key.is_empty() { "none" } else { key };
        Ok(HumanStep {
            executed: self.executed(shown, outcome.info.vetoed),
            vetoed: outcome.info.vetoed,
            event: outcome.info.event,
            reward: outcome.reward,
            done: outcome.done,
            score: self.score(),
        })
    }

    fn ask(
        &self,
        policy: &mut Policy<'_>,
        oracle: &Option<Vec<Vec<f32>>>,
        qi: usize,
        state: &str,
    ) -> Result<Vec<f32>, String> {
        let q = &self.questions[qi];
        match policy {
            Policy::Model(m) => m.probs(q.qtype, q.instructions, &q.options, state),
            Policy::Oracle => Ok(oracle.as_ref().expect("oracle targets")[qi].clone()),
        }
    }

    fn option_probs(&self, qi: usize, p: &[f32]) -> Vec<OptionProb> {
        self.questions[qi]
            .keys
            .iter()
            .zip(p)
            .map(|(&key, &p)| OptionProb { key, p })
            .collect()
    }

    fn secondary(&self, p: &[f32]) -> Option<Secondary> {
        let q = self.questions.get(1)?;
        let (value, max) = match q.qtype {
            QType::Noul => (p[1], 1.0),
            QType::Score => (
                p.iter().enumerate().map(|(i, &v)| i as f32 * v).sum(),
                (p.len() - 1) as f32,
            ),
            QType::Choice => (p[argmax(p)], 1.0),
        };
        Some(Secondary {
            id: q.id,
            qtype: q.qtype.name(),
            value,
            max,
            probs: self.option_probs(1, p),
        })
    }

    /// One AI frame: when a decision is due, the policy answers every question of the game from
    /// the state text, the primary argmax (or `abort` for dodge) goes through the game's safety
    /// reflex and the physics advances one tick. Otherwise the last decision is held.
    pub fn step_ai(&mut self, mut policy: Policy<'_>) -> Result<AiStep, String> {
        let t0 = now_us();
        let policy_name = policy.name();
        if self.game.done() {
            let held = self.held.clone();
            return Ok(AiStep {
                policy: policy_name,
                decided: false,
                action: held.as_ref().map_or("none", |h| h.action),
                executed: "none".into(),
                probs: held.as_ref().map(|h| h.probs.clone()).unwrap_or_default(),
                secondary: held.as_ref().and_then(|h| h.secondary.clone()),
                vetoed: false,
                micros: now_us() - t0,
                state_text: held.map(|h| h.state_text).unwrap_or_default(),
                event: sky_games::Event::None,
                reward: 0.0,
                done: true,
                score: self.score(),
            });
        }

        let mut decide = false;
        if self.game.decision_due() {
            decide = self.due_count.is_multiple_of(self.decide_every as u64);
            self.due_count += 1;
        }

        let outcome: StepOutcome;
        if decide {
            let state = self.game.state_text();
            let oracle = matches!(policy, Policy::Oracle).then(|| self.game.oracle());
            let p0 = self.ask(&mut policy, &oracle, 0, &state)?;
            let p1 = if self.questions.len() > 1 {
                Some(self.ask(&mut policy, &oracle, 1, &state)?)
            } else {
                None
            };
            let choice = argmax(&p0);
            let abort = self.kind == sky_games::dodge::NAME
                && p1.as_ref().is_some_and(|p| p.len() == 2 && p[1] > p[0]);
            outcome = self.game.apply(choice, abort);
            self.held = Some(Held {
                action: if abort {
                    "abort"
                } else {
                    self.questions[0].keys[choice]
                },
                probs: self.option_probs(0, &p0),
                secondary: p1.as_deref().and_then(|p| self.secondary(p)),
                state_text: state,
            });
        } else {
            outcome = self.game.tick();
        }

        let held = self.held.clone().unwrap_or_else(|| Held {
            action: "none",
            probs: Vec::new(),
            secondary: None,
            state_text: String::new(),
        });
        let vetoed = outcome.info.vetoed;
        let executed = self.executed(held.action, vetoed);
        let micros = now_us() - t0;
        if decide {
            self.record_timing(micros);
        }
        Ok(AiStep {
            policy: policy_name,
            decided: decide,
            action: held.action,
            executed,
            probs: held.probs,
            secondary: held.secondary,
            vetoed,
            micros,
            state_text: held.state_text,
            event: outcome.info.event,
            reward: outcome.reward,
            done: outcome.done,
            score: self.score(),
        })
    }
}

/// Decisions per timing sample in [`bench`]: browsers coarsen `performance.now()` (to 100 µs or
/// more without cross-origin isolation), so single decisions are not timed individually.
const BENCH_BATCH: usize = 10;

/// Times `n` AI decisions of a model on Flappy Drone (each decision answers both questions and
/// steps the physics), restarting episodes as needed. Decisions are timed in batches of
/// [`BENCH_BATCH`]; median and p95 are over the per-decision average of each batch. Returns
/// `{n, batch, median_us, p95_us, mean_us, decisions_per_s, forwards_per_decision, game}`.
pub fn bench(model: &mut Model, n: usize) -> Result<Value, String> {
    let n = n.max(1);
    let mut s = Session::new(sky_games::flappy::NAME, 0x5eed)?;
    // Warm-up: fills the prefix and BPE caches.
    for _ in 0..8 {
        s.step_ai(Policy::Model(model))?;
    }
    let batch = BENCH_BATCH.min(n);
    let mut samples = Vec::with_capacity(n / batch + 1);
    let (mut done, t_start) = (0, now_us());
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
    let mean = (now_us() - t_start) / n as f64;
    let (median, p95, _) = summarize(&samples);
    Ok(json!({
        "game": sky_games::flappy::NAME,
        "n": n,
        "batch": batch,
        "median_us": median,
        "p95_us": p95,
        "mean_us": mean,
        "decisions_per_s": if mean > 0.0 { 1e6 / mean } else { 0.0 },
        "forwards_per_decision": s.questions().len(),
    }))
}
