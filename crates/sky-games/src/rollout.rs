//! Rollout harness: plays a game with a [`Policy`] and reports metrics.

use crate::common::{argmax, DecisionGame, EpisodeStats, Event};
use crate::{dodge, doom, flappy, rescue};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use serde::Serialize;

/// Names of the games, in a stable order.
pub const GAMES: [&str; 4] = [flappy::NAME, rescue::NAME, dodge::NAME, doom::NAME];

/// Creates a game by name.
pub fn make_game(name: &str, seed: u64) -> Option<Box<dyn DecisionGame>> {
    match name {
        flappy::NAME => Some(Box::new(flappy::Flappy::new(seed))),
        rescue::NAME => Some(Box::new(rescue::Rescue::new(seed))),
        dodge::NAME => Some(Box::new(dodge::Dodge::new(seed))),
        doom::NAME => Some(Box::new(doom::Doom::new(seed))),
        _ => None,
    }
}

/// A decision maker. `decide` returns one probability per rendered option, exactly like the
/// System One model does from the same text inputs.
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

/// Plays the oracle's soft targets.
#[derive(Debug, Default)]
pub struct OraclePolicy {
    ids: Vec<&'static str>,
    targets: Vec<Vec<f32>>,
}

impl Policy for OraclePolicy {
    fn decide(
        &mut self,
        _game: &str,
        question_id: &str,
        _state: &str,
        rendered_opts: &[String],
    ) -> Vec<f32> {
        self.ids
            .iter()
            .position(|id| *id == question_id)
            .map(|i| self.targets[i].clone())
            .unwrap_or_else(|| vec![1.0 / rendered_opts.len() as f32; rendered_opts.len()])
    }

    fn observe(&mut self, game: &dyn DecisionGame) {
        self.ids = game.questions().iter().map(|q| q.id).collect();
        self.targets = game.oracle();
    }
}

/// Uniformly random probabilities (a random argmax).
#[derive(Debug)]
pub struct RandomPolicy {
    rng: SmallRng,
}

impl RandomPolicy {
    pub fn new(seed: u64) -> Self {
        RandomPolicy {
            rng: SmallRng::seed_from_u64(seed),
        }
    }
}

impl Policy for RandomPolicy {
    fn decide(
        &mut self,
        _game: &str,
        _q: &str,
        _state: &str,
        rendered_opts: &[String],
    ) -> Vec<f32> {
        let mut p: Vec<f32> = rendered_opts
            .iter()
            .map(|_| self.rng.gen::<f32>() + 1e-6)
            .collect();
        let s: f32 = p.iter().sum();
        p.iter_mut().for_each(|x| *x /= s);
        p
    }
}

/// Aggregate results of [`evaluate`].
#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct Metrics {
    pub game: String,
    pub episodes: usize,
    /// Flappy: mean pipes passed. Rescue: mean hikers rescued. Dodge: mean obstacles passed.
    /// Doom: mean kills.
    pub mean_score: f32,
    /// Doom: kills over monsters, summed over the episodes.
    pub kill_rate: f32,
    /// Rescue: fraction of hikers rescued.
    pub rescue_rate: f32,
    /// Fraction of episodes that completed the mission (rescue, dodge, doom: exit reached).
    pub completion_rate: f32,
    /// Crashes (doom: deaths).
    pub crashes: u32,
    pub vetoes: u32,
    pub aborts: u32,
    /// Dodge: aborts in front of an impassable obstacle.
    pub safe_aborts: u32,
    pub mean_steps: f32,
}

/// Episode limits and options of [`evaluate_with`].
#[derive(Debug, Clone, Copy)]
pub struct EvalOptions {
    /// Flappy episodes stop (alive) after this many pipes.
    pub flappy_max_pipes: u32,
    /// Enable the safety reflex (rescue, dodge).
    pub veto: bool,
}

impl Default for EvalOptions {
    fn default() -> Self {
        EvalOptions {
            flappy_max_pipes: 1000,
            veto: true,
        }
    }
}

/// [`evaluate_with`] with default options.
pub fn evaluate(game: &str, policy: &mut dyn Policy, episodes: usize, seed: u64) -> Metrics {
    evaluate_with(game, policy, episodes, seed, EvalOptions::default())
}

/// Plays `episodes` episodes of `game` (episode `i` is the `i`-th course of `Game::new(seed)`).
/// The primary question picks the action; dodge also asks its `abort` question.
pub fn evaluate_with(
    game: &str,
    policy: &mut dyn Policy,
    episodes: usize,
    seed: u64,
    opts: EvalOptions,
) -> Metrics {
    let mut g = make_game(game, seed).unwrap_or_else(|| panic!("unknown game {game:?}"));
    let questions = g.questions();
    let rendered: Vec<Vec<String>> = questions.iter().map(|q| q.canonical_options()).collect();
    let mut m = Metrics {
        game: game.to_string(),
        episodes,
        ..Metrics::default()
    };
    let (mut score, mut steps, mut completed) = (0.0f64, 0.0f64, 0usize);
    let mut max_score = 0.0f64;
    for ep in 0..episodes {
        if ep > 0 {
            g.reset();
        }
        g.set_veto(opts.veto);
        while !g.done() {
            if game == flappy::NAME && g.stats().score as u32 >= opts.flappy_max_pipes {
                break;
            }
            if g.decision_due() {
                policy.observe(&*g);
                let text = g.state_text();
                let p = policy.decide(game, questions[0].id, &text, &rendered[0]);
                let abort = game == dodge::NAME && {
                    let pa = policy.decide(game, questions[1].id, &text, &rendered[1]);
                    pa[1] > pa[0]
                };
                g.apply(argmax(&p), abort);
            } else {
                g.tick();
            }
        }
        let s = g.stats();
        score += s.score as f64;
        max_score += s.max_score as f64;
        steps += s.steps as f64;
        completed += s.success as usize;
        m.crashes += s.crashed as u32;
        m.vetoes += s.vetoes;
        m.aborts += s.aborted as u32;
        m.safe_aborts += s.safe_abort as u32;
    }
    let n = episodes.max(1) as f64;
    m.mean_score = (score / n) as f32;
    m.mean_steps = (steps / n) as f32;
    m.completion_rate = (completed as f64 / n) as f32;
    if game == rescue::NAME {
        m.rescue_rate = m.mean_score / rescue::N_HIKERS as f32;
    }
    if max_score > 0.0 {
        m.kill_rate = (score / max_score) as f32;
    }
    m
}

/// One decision of a recorded episode ([`record_with`]): what the policy saw and answered, what
/// the oracle would have answered, and what followed until the next decision.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct TrajStep {
    pub game: String,
    pub episode: usize,
    /// Decision index within the episode.
    pub step: usize,
    /// Simulator ticks played before this decision.
    pub tick: u32,
    /// The text state the policy read.
    pub state: String,
    /// The primary (action) question.
    pub question: &'static str,
    pub options: Vec<String>,
    /// The policy's probabilities over `options`.
    pub probs: Vec<f32>,
    /// The oracle's soft target over `options`, from the full simulator.
    pub oracle: Vec<f32>,
    /// The action played (argmax of `probs`).
    pub action: usize,
    /// The oracle's argmax.
    pub oracle_action: usize,
    /// Dodge: the policy's probabilities on its `abort` question, the abort played, and the
    /// oracle's target.
    pub abort_probs: Option<Vec<f32>>,
    pub abort: Option<bool>,
    pub oracle_abort: Option<Vec<f32>>,
    /// Reward summed from this decision up to the next one.
    pub reward: f32,
    /// Events (other than `none`) from this decision up to the next one.
    pub events: Vec<Event>,
    /// The safety reflex replaced the requested action.
    pub vetoed: bool,
    /// Last decision of the episode.
    pub last: bool,
}

/// The outcome of a recorded episode ([`record_with`]).
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct TrajEpisode {
    pub game: String,
    pub episode: usize,
    pub decisions: usize,
    /// Sum of the rewards of every tick.
    #[serde(rename = "return")]
    pub ret: f32,
    /// Flappy: stopped alive at `flappy_max_pipes`.
    pub truncated: bool,
    #[serde(flatten)]
    pub stats: EpisodeStats,
}

/// Plays `episodes` episodes of `game` like [`evaluate_with`] (same courses, same decisions) and
/// hands every decision to `on_step` and every episode to `on_episode`.
pub fn record_with(
    game: &str,
    policy: &mut dyn Policy,
    episodes: usize,
    seed: u64,
    opts: EvalOptions,
    mut on_step: impl FnMut(TrajStep),
    mut on_episode: impl FnMut(TrajEpisode),
) {
    let mut g = make_game(game, seed).unwrap_or_else(|| panic!("unknown game {game:?}"));
    let questions = g.questions();
    let rendered: Vec<Vec<String>> = questions.iter().map(|q| q.canonical_options()).collect();
    let is_dodge = game == dodge::NAME;
    for ep in 0..episodes {
        if ep > 0 {
            g.reset();
        }
        g.set_veto(opts.veto);
        let mut pending: Option<TrajStep> = None;
        let (mut ret, mut decisions, mut truncated) = (0.0f32, 0usize, false);
        while !g.done() {
            if game == flappy::NAME && g.stats().score as u32 >= opts.flappy_max_pipes {
                truncated = true;
                break;
            }
            let out = if g.decision_due() {
                if let Some(s) = pending.take() {
                    on_step(s);
                }
                policy.observe(&*g);
                let oracle = g.oracle();
                let text = g.state_text();
                let probs = policy.decide(game, questions[0].id, &text, &rendered[0]);
                let abort_probs =
                    is_dodge.then(|| policy.decide(game, questions[1].id, &text, &rendered[1]));
                let abort = abort_probs.as_ref().map(|pa| pa[1] > pa[0]);
                let action = argmax(&probs);
                let tick = g.stats().steps;
                let out = g.apply(action, abort.unwrap_or(false));
                pending = Some(TrajStep {
                    game: game.to_string(),
                    episode: ep,
                    step: decisions,
                    tick,
                    state: text,
                    question: questions[0].id,
                    options: rendered[0].clone(),
                    probs,
                    oracle_action: argmax(&oracle[0]),
                    oracle: oracle[0].clone(),
                    abort_probs,
                    abort,
                    oracle_abort: is_dodge.then(|| oracle[1].clone()),
                    action,
                    reward: 0.0,
                    events: Vec::new(),
                    vetoed: false,
                    last: false,
                });
                decisions += 1;
                out
            } else {
                g.tick()
            };
            ret += out.reward;
            if let Some(s) = pending.as_mut() {
                s.reward += out.reward;
                if out.info.event != Event::None {
                    s.events.push(out.info.event);
                }
                s.vetoed |= out.info.vetoed;
            }
        }
        if let Some(mut s) = pending.take() {
            s.last = true;
            on_step(s);
        }
        on_episode(TrajEpisode {
            game: game.to_string(),
            episode: ep,
            decisions,
            ret,
            truncated,
            stats: g.stats(),
        });
    }
}
