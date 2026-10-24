//! Dataset generator: plays games with a mix of oracle and epsilon-noisy policies (DAgger
//! style: off-distribution states are labeled by the oracle too) and emits decision records
//! with paraphrased instructions, renamed and shuffled options.

use crate::common::{argmax, mix_seed, DecisionGame, QuestionSpec};
use crate::rollout::make_game;
use crate::{dodge, doom, flappy, rescue};
pub use rand::rngs::SmallRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use sky_schema::{render_choice, render_noul, render_score, DecisionRecord, QType, Split};

/// Share of samples asking a secondary question (split evenly between them when a game has
/// several, like doom's `danger` and `threat`).
pub const SECONDARY_SHARE: f64 = 0.3;
/// Share of samples rendered in French.
pub const FRENCH_SHARE: f64 = 0.3;
/// Per-episode exploration rates.
const EPSILONS: [f32; 4] = [0.0, 0.05, 0.15, 0.3];
/// Tempering exponent of the class-balancing filter (0 = uniform labels, 1 = no balancing).
const BALANCE_POWER: f32 = 0.3;
/// Dodge: probability of executing an abort the oracle asks for. Below 1 the drone keeps
/// approaching (braking toward) impassable obstacles, so those states are covered too.
const DODGE_ABORT_EXEC: f64 = 0.3;

/// Split of an episode seed: seeds `≡ 18 (mod 20)` are validation, `≡ 19` test, others train.
pub fn split_for_seed(episode_seed: u64) -> Split {
    match episode_seed % 20 {
        18 => Split::Val,
        19 => Split::Test,
        _ => Split::Train,
    }
}

/// A question rendered with a random paraphrase.
#[derive(Debug, Clone, PartialEq)]
pub struct RenderedQuestion {
    pub ins: String,
    pub opts: Vec<String>,
    /// Canonical option index of each rendered option (key -> canonical action mapping).
    pub canon: Vec<usize>,
}

impl RenderedQuestion {
    /// Reorders a canonical target to follow the rendered options.
    pub fn map_target(&self, canonical: &[f32]) -> Vec<f32> {
        self.canon.iter().map(|&i| canonical[i]).collect()
    }
}

fn pick<'a, R: Rng>(rng: &mut R, items: &[&'a str]) -> &'a str {
    items[rng.gen_range(0..items.len())]
}

/// Renders `spec` with a random language, instruction variant, option keys, descriptions and
/// (for choices) option order.
pub fn render_variant<R: Rng>(spec: &QuestionSpec, rng: &mut R) -> RenderedQuestion {
    let fr = rng.gen_bool(FRENCH_SHARE);
    let ins = pick(rng, if fr { spec.ins_fr } else { spec.ins_en }).to_string();
    let descs = |o: &crate::common::OptionSpec| if fr { o.desc_fr } else { o.desc_en };
    match spec.qtype {
        QType::Choice => {
            let mut canon: Vec<usize> = (0..spec.options.len()).collect();
            canon.shuffle(rng);
            let items: Vec<(&str, &str)> = canon
                .iter()
                .map(|&i| {
                    let o = &spec.options[i];
                    let key = if o.synonyms.is_empty() || rng.gen_bool(0.5) {
                        o.key
                    } else {
                        pick(rng, o.synonyms)
                    };
                    let desc = if rng.gen_bool(0.15) {
                        ""
                    } else {
                        pick(rng, descs(o))
                    };
                    (key, desc)
                })
                .collect();
            RenderedQuestion {
                ins,
                opts: render_choice(&items),
                canon,
            }
        }
        QType::Score => {
            let n_variants = spec
                .options
                .iter()
                .map(|o| descs(o).len())
                .min()
                .unwrap_or(1);
            let j = rng.gen_range(0..n_variants);
            let levels: Vec<&str> = spec.options.iter().map(|o| descs(o)[j]).collect();
            RenderedQuestion {
                ins,
                opts: render_score(&levels),
                canon: (0..levels.len()).collect(),
            }
        }
        QType::Noul => {
            let (f, t) = (descs(&spec.options[0]), descs(&spec.options[1]));
            let opts = if rng.gen_bool(0.5) {
                render_noul(None, None)
            } else {
                let j = rng.gen_range(0..f.len().min(t.len()));
                render_noul(Some(t[j]), Some(f[j]))
            };
            RenderedQuestion {
                ins,
                opts,
                canon: vec![0, 1],
            }
        }
    }
}

/// Label balance statistics of a generation run.
#[derive(Debug, Clone, Default)]
pub struct GenStats {
    pub game: String,
    pub records: usize,
    pub episodes: u64,
    /// `[train, val, test]` record counts.
    pub splits: [usize; 3],
    /// Per question: id, canonical keys and the argmax-label histogram of emitted records.
    pub labels: Vec<(String, Vec<String>, Vec<usize>)>,
}

impl GenStats {
    /// Human-readable summary.
    pub fn report(&self) -> String {
        let mut s = format!(
            "{}: {} records from {} episodes (train {}, val {}, test {})\n",
            self.game, self.records, self.episodes, self.splits[0], self.splits[1], self.splits[2]
        );
        for (id, keys, counts) in &self.labels {
            let total: usize = counts.iter().sum::<usize>().max(1);
            s.push_str(&format!("  {id} ({total}):"));
            for (k, c) in keys.iter().zip(counts) {
                s.push_str(&format!(" {k}={:.1}%", 100.0 * *c as f64 / total as f64));
            }
            s.push('\n');
        }
        s
    }
}

/// Random action for epsilon-noisy play: `(primary choice, abort)`.
fn noisy_action(game: &str, rng: &mut SmallRng) -> (usize, bool) {
    match game {
        flappy::NAME => (usize::from(!rng.gen_bool(0.25)), false),
        rescue::NAME => (rng.gen_range(0..7), false),
        doom::NAME => (rng.gen_range(0..8), false),
        _ => (rng.gen_range(0..4), rng.gen_bool(0.01)),
    }
}

/// Base keep rate of a decision (consecutive flappy frames, and doom decisions 4 ticks apart,
/// are strongly correlated).
fn base_keep(game: &str) -> f32 {
    match game {
        flappy::NAME => 0.5,
        doom::NAME => 0.6,
        _ => 1.0,
    }
}

/// Class-balancing keep probability of `label` given the labels seen so far: the kept
/// distribution is the seen distribution tempered as `f^BALANCE_POWER`, so rare decisions
/// (thrust, brake, abort) are over-represented without discarding the common ones entirely.
/// Classes rarer than 0.2% are always kept and ignored for the normalization.
fn balance_keep(seen: &[f32], label: usize) -> f32 {
    let total: f32 = seen.iter().sum();
    if total < 50.0 {
        return 1.0;
    }
    let ratio = |c: f32| (c / total).powf(BALANCE_POWER) / (c / total);
    let f = seen[label] / total;
    if f < 0.002 {
        return 1.0;
    }
    let max = seen
        .iter()
        .filter(|c| **c / total >= 0.002)
        .map(|c| ratio(*c))
        .fold(0.0f32, f32::max);
    (ratio(seen[label]) / max).clamp(0.02, 1.0)
}

/// Picks the `(choice, abort)` played at a decision point while collecting data.
pub type Actor<'a> = dyn FnMut(&dyn DecisionGame, &mut SmallRng) -> (usize, bool) + 'a;

/// Default share β of decisions an [`Actor`] hands back to the oracle during DAgger collection,
/// so the episodes still reach the later parts of each course (the mixing policy
/// `β·oracle + (1 − β)·student` of DAgger; arXiv 2605.12913 interpolates the same way per turn).
pub const DAGGER_ORACLE_SHARE: f32 = 0.2;

/// Generates `n` records for `game`, calling `emit` for each one. Returns the statistics.
pub fn generate_with(
    game: &str,
    n: usize,
    seed: u64,
    emit: impl FnMut(DecisionRecord),
) -> Result<GenStats, String> {
    generate_with_actor(game, n, seed, None, DAGGER_ORACLE_SHARE, emit)
}

/// Like [`generate_with`], but when `actor` is set it drives the episodes (DAgger): the states
/// come from the actor's own play, except for a share `oracle_share` (β) of decisions the oracle
/// takes, and the targets still come from the oracle.
pub fn generate_with_actor(
    game: &str,
    n: usize,
    seed: u64,
    mut actor: Option<&mut Actor<'_>>,
    oracle_share: f32,
    mut emit: impl FnMut(DecisionRecord),
) -> Result<GenStats, String> {
    let probe = make_game(game, 0).ok_or_else(|| format!("unknown game {game:?}"))?;
    let questions = probe.questions();
    let src = format!("game:{game}");
    let mut rng = SmallRng::seed_from_u64(mix_seed(seed, 0x0067_656e));
    let mut stats = GenStats {
        game: game.to_string(),
        labels: questions
            .iter()
            .map(|q| {
                (
                    q.id.to_string(),
                    q.keys().iter().map(|k| k.to_string()).collect(),
                    vec![0; q.options.len()],
                )
            })
            .collect(),
        ..GenStats::default()
    };
    // Labels seen before filtering, per question, for the class-balancing filter.
    let mut seen: Vec<Vec<f32>> = questions
        .iter()
        .map(|q| vec![0.0; q.options.len()])
        .collect();

    let mut secondary = 0usize;
    let mut ep = 0u64;
    while stats.records < n {
        let episode_seed = seed.wrapping_mul(1_000_000).wrapping_add(ep);
        let split = split_for_seed(episode_seed);
        // Drawn at random rather than from the episode index, so that every split gets the
        // same mix of clean and noisy episodes.
        let eps = EPSILONS[rng.gen_range(0..EPSILONS.len())];
        let mut g = make_game(game, episode_seed).expect("known game");
        ep += 1;
        play_episode(
            &mut *g,
            eps,
            actor.as_deref_mut(),
            oracle_share,
            &mut rng,
            |g, targets, rng| {
                if stats.records >= n {
                    return;
                }
                // Quota: the secondary question gets SECONDARY_SHARE of the emitted records.
                let secondary_due =
                    (secondary as f64) < SECONDARY_SHARE * (stats.records + 1) as f64;
                let qi = if questions.len() > 1 && secondary_due {
                    1 + secondary % (questions.len() - 1)
                } else {
                    0
                };
                let label = argmax(&targets[qi]);
                seen[qi][label] += 1.0;
                if rng.gen::<f32>() >= base_keep(game) * balance_keep(&seen[qi], label) {
                    return;
                }
                let r = render_variant(&questions[qi], rng);
                let rec = DecisionRecord {
                    src: src.clone(),
                    t: questions[qi].qtype,
                    target: r.map_target(&targets[qi]),
                    ins: r.ins,
                    opts: r.opts,
                    state: g.state_text(),
                    split,
                };
                stats.records += 1;
                stats.splits[split as usize] += 1;
                stats.labels[qi].2[label] += 1;
                secondary += usize::from(qi > 0);
                emit(rec);
            },
        );
        stats.episodes = ep;
    }
    Ok(stats)
}

/// Collects [`generate_with`] into a vector.
pub fn generate(
    game: &str,
    n: usize,
    seed: u64,
) -> Result<(Vec<DecisionRecord>, GenStats), String> {
    let mut out = Vec::with_capacity(n);
    let stats = generate_with(game, n, seed, |r| out.push(r))?;
    Ok((out, stats))
}

/// Plays one episode, calling `visit` at every decision before acting.
fn play_episode(
    g: &mut dyn DecisionGame,
    eps: f32,
    mut actor: Option<&mut Actor<'_>>,
    oracle_share: f32,
    rng: &mut SmallRng,
    mut visit: impl FnMut(&dyn DecisionGame, &[Vec<f32>], &mut SmallRng),
) {
    let name = g.name();
    // Flappy episodes are capped so that the data covers many courses.
    let max_score = if name == flappy::NAME { 12.0 } else { f32::MAX };
    while !g.done() && g.stats().score < max_score {
        if !g.decision_due() {
            g.tick();
            continue;
        }
        let t = g.oracle();
        visit(&*g, &t, rng);
        let use_actor = actor.is_some() && rng.gen::<f32>() >= oracle_share;
        let (choice, abort) = if use_actor {
            let act = actor.as_deref_mut().expect("actor checked above");
            act(&*g, rng)
        } else if actor.is_none() && rng.gen::<f32>() < eps {
            noisy_action(name, rng)
        } else {
            let abort = name == dodge::NAME && t[1][1] > 0.5 && rng.gen_bool(DODGE_ABORT_EXEC);
            (argmax(&t[0]), abort)
        };
        g.apply(choice, abort);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn render_variant_keeps_target_aligned() {
        let mut rng = SmallRng::seed_from_u64(9);
        let spec = &rescue::QUESTIONS[0];
        let canonical: Vec<f32> = (0..7).map(|i| i as f32).collect();
        for _ in 0..50 {
            let r = render_variant(spec, &mut rng);
            let t = r.map_target(&canonical);
            for (k, opt) in r.opts.iter().enumerate() {
                let o = &spec.options[t[k] as usize];
                let key = opt.split(':').next().unwrap();
                assert!(
                    key == o.key || o.synonyms.contains(&key),
                    "{opt} vs {}",
                    o.key
                );
            }
        }
    }

    #[test]
    fn splits_follow_seed_ranges() {
        assert_eq!(split_for_seed(0), Split::Train);
        assert_eq!(split_for_seed(18), Split::Val);
        assert_eq!(split_for_seed(39), Split::Test);
    }
}
