//! Rollout harness: plays a game with a [`Policy`] and reports metrics.

use crate::common::{argmax, DecisionGame};
use crate::{dodge, flappy, rescue};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use serde::Serialize;

/// Names of the games, in a stable order.
pub const GAMES: [&str; 3] = [flappy::NAME, rescue::NAME, dodge::NAME];

/// Creates a game by name.
pub fn make_game(name: &str, seed: u64) -> Option<Box<dyn DecisionGame>> {
    match name {
        flappy::NAME => Some(Box::new(flappy::Flappy::new(seed))),
        rescue::NAME => Some(Box::new(rescue::Rescue::new(seed))),
        dodge::NAME => Some(Box::new(dodge::Dodge::new(seed))),
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
    pub mean_score: f32,
    /// Rescue: fraction of hikers rescued.
    pub rescue_rate: f32,
    /// Fraction of episodes that completed the mission (rescue, dodge).
    pub completion_rate: f32,
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
    m
}
