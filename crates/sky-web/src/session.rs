//! A game session: one game core plus its System One questions, stepped by a human, by the
//! model or by the oracle. A full AI frame (state text, tokenization, forward passes, readout,
//! action, safety reflex, physics) runs here, without leaving WASM.

use serde::Serialize;
use serde_json::{json, Value};
use sky_games::common::argmax;
use sky_games::{make_game, DecisionGame, QuestionSpec, StepOutcome, GAMES};
use sky_infer::{Arch, QType};

use crate::clock::{now_us, summarize};
use crate::model::Model;

/// Number of recent decision timings kept for the rolling statistics.
const TIMING_WINDOW: usize = 120;

/// Model family of an architecture: `s1` (ModernBERT + decision head), `s2` (static cache,
/// early exits) or `iso` (s5: options isolated by construction).
pub fn family(arch: Arch) -> &'static str {
    match arch {
        Arch::S1 => "s1",
        Arch::S2 => "s2",
        Arch::Iso => "iso",
    }
}

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

/// Answer to a secondary question (flappy `danger`, rescue `risk`, dodge `abort`, doom `danger`
/// and `threat`).
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

/// How the model answered the questions of a decision frame.
#[derive(Debug, Clone, Serialize)]
pub struct DecisionInfo {
    /// Model family: `s1`, `s2` or `iso`.
    pub family: &'static str,
    /// s2: exit that answered the primary question.
    pub exit: Option<usize>,
    /// s2: number of exits of the model (0 for s1).
    pub n_exits: usize,
    /// s2: state-encoder blocks used by the primary question.
    pub layers_used: Option<usize>,
    /// s2: mean blocks used over the frame's questions.
    pub mean_layers: Option<f32>,
    /// s2: state-encoder depth of the model.
    pub max_layers: Option<usize>,
    /// Every question found its static part cached (s1: tokenized prefix; s2: static encoding).
    pub static_hit: bool,
    /// s2: the questions after the first reused the state encoding of the frame.
    pub state_reused: bool,
}

impl DecisionInfo {
    fn from_stats(model: &Model, stats: &[sky_infer::DecisionStats]) -> DecisionInfo {
        let s2 = model.arch() == Arch::S2;
        let layers: Vec<f32> = stats
            .iter()
            .filter_map(|s| s.layers_used)
            .map(|l| l as f32)
            .collect();
        DecisionInfo {
            family: family(model.arch()),
            exit: stats.first().and_then(|s| s.exit),
            n_exits: stats.first().map_or(0, |s| s.n_exits),
            layers_used: stats.first().and_then(|s| s.layers_used),
            mean_layers: (!layers.is_empty())
                .then(|| layers.iter().sum::<f32>() / layers.len() as f32),
            max_layers: s2.then(|| model.max_layers()),
            static_hit: !stats.is_empty() && stats.iter().all(|s| s.static_hit),
            state_reused: s2 && stats.len() > 1 && stats[1..].iter().all(|s| s.state_hit),
        }
    }
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
    /// Answers to the questions after the secondary one (doom: `threat`).
    pub others: Vec<Secondary>,
    /// How the model answered (exit, layers used, cache hits); `None` for the oracle.
    pub info: Option<DecisionInfo>,
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
    others: Vec<Secondary>,
    info: Option<DecisionInfo>,
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
    /// A new session of `kind` (`flappy`, `rescue`, `dodge` or `doom`).
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
    /// hovers, dodge keeps its maneuver, doom stands still. A human may act on every frame,
    /// even in games where the agent decides every few frames. The reflex still applies.
    pub fn step_human(&mut self, key: &str) -> Result<HumanStep, String> {
        let key = key.trim();
        let outcome = if (key.is_empty() || key == "none") && self.kind == sky_games::doom::NAME {
            // Releasing the keys stops the player (doom holds actions only for the agent).
            self.game.apply_human(usize::MAX)
        } else if key.is_empty() || key == "none" {
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
            self.game.apply_human(i)
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

    /// Probabilities of question `qi`; the model's statistics of the question go to `stats`.
    fn ask(
        &self,
        policy: &mut Policy<'_>,
        oracle: &Option<Vec<Vec<f32>>>,
        qi: usize,
        state: &str,
        stats: &mut Vec<sky_infer::DecisionStats>,
    ) -> Result<Vec<f32>, String> {
        let q = &self.questions[qi];
        match policy {
            Policy::Model(m) => {
                let p = m.probs(q.qtype, q.instructions, &q.options, state)?;
                stats.extend(m.last_stats().cloned());
                Ok(p)
            }
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

    fn secondary(&self, qi: usize, p: &[f32]) -> Option<Secondary> {
        let q = self.questions.get(qi)?;
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
            probs: self.option_probs(qi, p),
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
                others: held.as_ref().map(|h| h.others.clone()).unwrap_or_default(),
                info: held.as_ref().and_then(|h| h.info.clone()),
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
            self.held = Some(Held {
                action: if abort {
                    "abort"
                } else {
                    self.questions[0].keys[choice]
                },
                probs: self.option_probs(0, &p0),
                secondary: p1.as_deref().and_then(|p| self.secondary(1, p)),
                others,
                info,
                state_text: state,
            });
        } else {
            outcome = self.game.tick();
        }

        let held = self.held.clone().unwrap_or_else(|| Held {
            action: "none",
            probs: Vec::new(),
            secondary: None,
            others: Vec::new(),
            info: None,
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
            others: held.others,
            info: held.info,
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
/// `{name, family, params, file_bytes, n, batch, median_us, p95_us, mean_us, decisions_per_s,
/// forwards_per_decision, mean_layers_used, max_layers, static_hit_rate, state_reuse_rate, game}`
/// (`mean_layers_used` and `max_layers` are null for s1).
pub fn bench(model: &mut Model, n: usize) -> Result<Value, String> {
    bench_game(model, sky_games::flappy::NAME, n)
}

/// [`bench`] on any game: each decision answers every question of the game and steps the
/// physics (frames that only hold a decision are not timed separately but add their physics
/// time to the next batch).
pub fn bench_game(model: &mut Model, game: &str, n: usize) -> Result<Value, String> {
    let n = n.max(1);
    let mut s = Session::new(game, 0x5eed)?;
    // Warm-up: fills the prefix / static and BPE caches.
    for _ in 0..8 {
        s.step_ai(Policy::Model(model))?;
    }
    let batch = BENCH_BATCH.min(n);
    let mut samples = Vec::with_capacity(n / batch + 1);
    let (mut layers, mut n_layers, mut static_hits, mut reuses) = (0.0f64, 0usize, 0usize, 0usize);
    let (mut done, t_start) = (0, now_us());
    while done < n {
        let k = batch.min(n - done);
        let t0 = now_us();
        let mut decided = 0;
        while decided < k {
            if s.done() {
                s.reset();
            }
            let step = s.step_ai(Policy::Model(model))?;
            if step.decided {
                decided += 1;
                if let Some(info) = &step.info {
                    if let Some(l) = info.mean_layers {
                        layers += l as f64;
                        n_layers += 1;
                    }
                    static_hits += info.static_hit as usize;
                    reuses += info.state_reused as usize;
                }
            }
        }
        samples.push((now_us() - t0) / k as f64);
        done += k;
    }
    let mean = (now_us() - t_start) / n as f64;
    let (median, p95, _) = summarize(&samples);
    let s2 = model.arch() == Arch::S2;
    Ok(json!({
        "game": s.kind(),
        "name": model.name(),
        "family": family(model.arch()),
        "params": model.n_params(),
        "file_bytes": model.file_bytes(),
        "n": n,
        "batch": batch,
        "median_us": median,
        "p95_us": p95,
        "mean_us": mean,
        "decisions_per_s": if mean > 0.0 { 1e6 / mean } else { 0.0 },
        "forwards_per_decision": s.questions().len(),
        "mean_layers_used": (n_layers > 0).then(|| layers / n_layers as f64),
        "max_layers": s2.then(|| model.max_layers()),
        "static_hit_rate": static_hits as f64 / n as f64,
        "state_reuse_rate": if s2 { reuses as f64 / n as f64 } else { 0.0 },
    }))
}
