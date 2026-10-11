//! Integration tests: determinism, oracle vs random, state-text budget, questions.

use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use sky_games::common::argmax;
use sky_games::*;

/// Plays `steps` decisions mixing the oracle and random actions, returning the visited state
/// texts and final snapshot.
fn trace(game: &str, seed: u64, steps: usize) -> (Vec<String>, serde_json::Value) {
    let mut g = make_game(game, seed).unwrap();
    let mut rng = SmallRng::seed_from_u64(seed);
    let n_opts = g.questions()[0].options.len();
    let mut texts = Vec::new();
    for _ in 0..steps {
        if g.done() {
            g.reset();
        }
        if g.decision_due() {
            texts.push(g.state_text());
            let choice = if rng.gen_bool(0.2) {
                rng.gen_range(0..n_opts)
            } else {
                argmax(&g.oracle()[0])
            };
            g.apply(choice, false);
        } else {
            g.tick();
        }
    }
    (texts, g.snapshot_json())
}

#[test]
fn same_seed_same_trajectory() {
    for game in GAMES {
        let a = trace(game, 42, 400);
        let b = trace(game, 42, 400);
        assert_eq!(a, b, "{game} is not deterministic");
        let c = trace(game, 43, 400);
        assert_ne!(a.0, c.0, "{game}: different seeds gave the same trajectory");
    }
}

#[test]
fn reset_draws_a_new_course_deterministically() {
    for game in GAMES {
        let mut a = make_game(game, 5).unwrap();
        let mut b = make_game(game, 5).unwrap();
        let first = a.snapshot_json();
        a.reset();
        b.reset();
        assert_eq!(a.snapshot_json(), b.snapshot_json(), "{game}");
        assert_ne!(
            a.snapshot_json(),
            first,
            "{game}: reset replayed the same course"
        );
    }
}

/// Pieces allowed in a state text: 24 for the 2D games, 30 for doom (14 measured quantities:
/// enemy, aim, walls, waypoint and projectile, still well inside the 64-token state budget).
fn piece_budget(game: &str) -> usize {
    if game == "doom" {
        30
    } else {
        24
    }
}

#[test]
fn state_text_fits_token_budget() {
    for game in GAMES {
        let (texts, _) = trace(game, 7, 3000);
        assert!(!texts.is_empty());
        for t in &texts {
            let pieces: Vec<&str> = t.split_whitespace().collect();
            assert!(
                pieces.len() <= piece_budget(game),
                "{game}: {} pieces in {t:?}",
                pieces.len()
            );
            assert_eq!(pieces.len() % 2, 0, "{game}: not key/value pairs: {t:?}");
            for v in pieces.iter().skip(1).step_by(2) {
                let small_int = v.parse::<i32>().is_ok_and(|x| (-20..=99).contains(&x));
                assert!(small_int || v.len() <= 2, "{game}: value {v:?} in {t:?}");
            }
        }
    }
}

#[test]
fn questions_and_oracle_shapes_agree() {
    for game in GAMES {
        let g = make_game(game, 1).unwrap();
        let qs = g.questions();
        assert_eq!(qs.len(), if game == "doom" { 3 } else { 2 }, "{game}");
        let targets = g.oracle();
        assert_eq!(targets.len(), qs.len());
        for (q, t) in qs.iter().zip(&targets) {
            assert_eq!(t.len(), q.options.len(), "{game}/{}", q.id);
            assert!(
                (t.iter().sum::<f32>() - 1.0).abs() < 1e-4,
                "{game}/{}: {t:?}",
                q.id
            );
            assert_eq!(q.to_systemone().rendered_options(), q.canonical_options());
            assert!(!q.ins_en.is_empty() && !q.ins_fr.is_empty());
            for o in q.options {
                assert!(
                    !o.desc_en.is_empty() && !o.desc_fr.is_empty(),
                    "{game}/{}/{}",
                    q.id,
                    o.key
                );
            }
            // Keys and synonyms are unique within a question.
            let mut keys: Vec<&str> = q
                .options
                .iter()
                .flat_map(|o| std::iter::once(o.key).chain(o.synonyms.iter().copied()))
                .collect();
            let n = keys.len();
            keys.sort_unstable();
            keys.dedup();
            assert_eq!(keys.len(), n, "{game}/{}: duplicate keys", q.id);
        }
    }
}

const QUICK: EvalOptions = EvalOptions {
    flappy_max_pipes: 15,
    veto: true,
};

#[test]
fn flappy_oracle_beats_random() {
    let o = evaluate_with("flappy", &mut OraclePolicy::default(), 3, 11, QUICK);
    let r = evaluate_with("flappy", &mut RandomPolicy::new(1), 3, 11, QUICK);
    assert_eq!(o.crashes, 0, "{o:?}");
    assert!(o.mean_score >= 15.0, "{o:?}");
    assert!(r.mean_score <= 1.0, "{r:?}");
}

/// The oracle must survive 500 pipes on average (release builds only: ~150k lookahead frames).
#[test]
#[cfg_attr(debug_assertions, ignore = "slow in debug builds, runs with --release")]
fn flappy_oracle_survives_500_pipes() {
    let opts = EvalOptions {
        flappy_max_pipes: 600,
        veto: true,
    };
    let o = evaluate_with("flappy", &mut OraclePolicy::default(), 3, 2026, opts);
    assert!(o.mean_score >= 500.0, "{o:?}");
}

#[test]
fn rescue_oracle_beats_random() {
    let o = evaluate("rescue", &mut OraclePolicy::default(), 30, 3);
    let r = evaluate("rescue", &mut RandomPolicy::new(2), 30, 3);
    assert!(o.rescue_rate >= 0.95 && o.completion_rate >= 0.9, "{o:?}");
    assert_eq!(o.crashes, 0);
    assert!(r.rescue_rate <= 0.4 && r.completion_rate <= 0.1, "{r:?}");
    assert!(
        r.vetoes > o.vetoes + 50,
        "the reflex should fire for random play: {r:?}"
    );
    let off = EvalOptions {
        veto: false,
        ..EvalOptions::default()
    };
    let r_off = evaluate_with("rescue", &mut RandomPolicy::new(2), 30, 3, off);
    assert!(
        r_off.crashes >= 20,
        "random play without the reflex should crash: {r_off:?}"
    );
    let o_off = evaluate_with("rescue", &mut OraclePolicy::default(), 30, 3, off);
    assert_eq!(o_off.crashes, 0, "{o_off:?}");
}

/// Random maneuvers, never aborting: exercises the dodge safety reflex.
struct RandomManeuver(SmallRng);

impl Policy for RandomManeuver {
    fn decide(
        &mut self,
        _game: &str,
        question_id: &str,
        _state: &str,
        opts: &[String],
    ) -> Vec<f32> {
        if question_id == "abort" {
            return vec![1.0, 0.0];
        }
        let mut p = vec![0.0; opts.len()];
        p[self.0.gen_range(0..opts.len())] = 1.0;
        p
    }
}

#[test]
fn dodge_oracle_beats_random() {
    let o = evaluate("dodge", &mut OraclePolicy::default(), 30, 5);
    let safe = (o.completion_rate * 30.0).round() as u32 + o.safe_aborts;
    assert!(safe >= 29 && o.crashes == 0, "{o:?}");
    assert_eq!(
        o.aborts, o.safe_aborts,
        "oracle aborted a passable course: {o:?}"
    );
    assert!(
        o.vetoes <= 3,
        "the reflex should rarely fire for the oracle: {o:?}"
    );
    let r = evaluate("dodge", &mut RandomPolicy::new(3), 30, 5);
    assert!(r.completion_rate <= 0.1, "{r:?}");
    let mut rm = RandomManeuver(SmallRng::seed_from_u64(4));
    let with = evaluate("dodge", &mut rm, 30, 5);
    let mut rm = RandomManeuver(SmallRng::seed_from_u64(4));
    let without = evaluate_with(
        "dodge",
        &mut rm,
        30,
        5,
        EvalOptions {
            veto: false,
            ..EvalOptions::default()
        },
    );
    assert!(with.vetoes > 0, "{with:?}");
    assert!(
        without.crashes > with.crashes,
        "reflex on {with:?} vs off {without:?}"
    );
    assert!(
        o.completion_rate > with.completion_rate + 0.3,
        "oracle {o:?} vs random {with:?}"
    );
}

#[test]
fn snapshots_serialize() {
    let f = Flappy::new(1).snapshot();
    assert_eq!(f.pipes.len(), 4);
    let r = Rescue::new(1).snapshot();
    assert_eq!(r.cells.len(), 144);
    assert_eq!(r.hikers.len(), 3);
    let d = Dodge::new(1).snapshot();
    assert_eq!(d.obstacles.len(), dodge::N_OBSTACLES);
    let m = Doom::new(1).snapshot();
    assert_eq!(m.cells.len(), m.width * m.height);
    assert!(m.monsters > 0 && m.sprites.len() >= m.monsters as usize);
    for game in GAMES {
        let v = make_game(game, 1).unwrap().snapshot_json();
        assert!(v.is_object());
    }
}

#[test]
fn dodge_veto_is_reported_in_step_info() {
    // Hold `continue` on every decision: the reflex must fire before the first obstacle.
    let mut g = Dodge::new(8);
    let mut fired = false;
    for _ in 0..2000 {
        let out = g.step(DodgeAction::Continue);
        fired |= out.info.vetoed;
        if out.done {
            break;
        }
    }
    assert!(fired);
    assert!(!g.crashed || g.vetoes > 0);
}

/// Doom gates on easy levels: the oracle completes at least 90% of 100 seeded levels with no
/// death; random play completes none (release builds only: the oracle simulates ahead).
#[test]
#[cfg_attr(debug_assertions, ignore = "slow in debug builds, runs with --release")]
fn doom_oracle_clears_easy_levels() {
    let o = evaluate("doom", &mut OraclePolicy::default(), 100, 2026);
    assert!(o.completion_rate >= 0.9, "{o:?}");
    assert_eq!(o.crashes, 0, "the oracle died: {o:?}");
    let r = evaluate("doom", &mut RandomPolicy::new(5), 100, 2026);
    assert_eq!(r.completion_rate, 0.0, "{r:?}");
    assert!(
        o.kill_rate > r.kill_rate + 0.3,
        "oracle {o:?} vs random {r:?}"
    );
}

#[test]
fn doom_oracle_beats_random_quickly() {
    let o = evaluate("doom", &mut OraclePolicy::default(), 4, 11);
    let r = evaluate("doom", &mut RandomPolicy::new(1), 4, 11);
    assert!(o.completion_rate >= 0.75, "{o:?}");
    assert_eq!(r.completion_rate, 0.0, "{r:?}");
}
