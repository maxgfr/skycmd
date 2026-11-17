//! The `recall` probe of the browser (`sky_web::recall`, the Memory page of s6-sleep3-nano): its
//! episodes are the evaluation's (`sky_games::make_game("recall", seed)`, what `sky-train
//! s2-probe` plays), its question is the one `s2-probe` asks, an ask is scored by the game, a
//! fixed seed gives a fixed episode and fixed answers, and the memory switch is the engine's.
//! On the golden memory model of `crates/sky-infer/tests/golden/` (a tiny random model) and on the
//! published s6-sleep3-nano file and its control.
//!
//!   cargo test -p sky-web --test recall

use sky_games::make_game;
use sky_games::recall::{Recall, BUCKETS, FRAMES, QUESTIONS};
use sky_web::recall::{describe, Probe, StepView};

const GOLDEN: &[u8] = include_bytes!("../../sky-infer/tests/golden/s6m-ternary.gguf");
const SLEEP3: &[u8] = include_bytes!("../../../site/public/models/s6-sleep3-nano.gguf");
const CONTROL: &[u8] = include_bytes!("../../../site/public/models/s6-sleep3-nano-control.gguf");

/// Seeds of the validation split (≡ 18 mod 20, never trained on), as the browser plays them, a
/// train seed and a small one.
const SEEDS: [u64; 4] = [7_000_000_018, 7_000_000_038, 1_000_000_000, 5];

fn argmax(v: &[f32]) -> usize {
    (0..v.len()).fold(0, |b, i| if v[i] > v[b] { i } else { b })
}

fn steps(p: &mut Probe, seed: u64, n: usize) -> Vec<StepView> {
    p.start(seed);
    (0..n).map(|_| p.step().expect("a frame")).collect()
}

#[test]
fn an_episode_is_the_evaluations_episode() {
    let keys = QUESTIONS[0].keys();
    for seed in SEEDS {
        let frames = describe(seed);
        // What s2-probe plays: make_game("recall", seed), which is Recall::new(seed).
        let mut g = make_game("recall", seed).unwrap();
        let mut r = Recall::new(seed);
        assert_eq!(frames.len(), FRAMES as usize);
        for f in &frames {
            assert_eq!(f.frame, g.stats().steps);
            assert_eq!(f.state, g.state_text(), "seed {seed} frame {}", f.frame);
            assert_eq!(f.state, r.state_text());
            let answer = argmax(&g.oracle()[0]);
            assert_eq!(f.answer, keys[answer]);
            assert_eq!(
                f.kind == "ask",
                answer != 0,
                "an ask is the only frame not answered wait"
            );
            assert_eq!(
                f.state.starts_with("show none"),
                f.kind != "show",
                "{}",
                f.state
            );
            g.apply(answer, false);
            r.step(answer);
        }
        assert!(g.done());
        // The delays are the game's own record of its asks.
        let asks = r.asks();
        assert!(asks.len() >= 6 && asks.len() <= BUCKETS.len() * 2);
        for a in asks {
            let f = &frames[a.frame as usize];
            assert_eq!(f.kind, "ask");
            assert_eq!((f.delay, f.bucket), (Some(a.delay), Some(a.bucket)));
            assert!((BUCKETS[a.bucket].0..=BUCKETS[a.bucket].1).contains(&a.delay));
        }
        assert_eq!(
            frames.iter().filter(|f| f.kind == "ask").count(),
            asks.len()
        );
    }
}

#[test]
fn the_question_is_the_one_s2_probe_asks() {
    let mut p = Probe::load(GOLDEN).unwrap();
    let info = p.info();
    let q = &QUESTIONS[0];
    assert_eq!(info["instructions"], q.instructions());
    assert_eq!(info["options"], serde_json::json!(q.canonical_options()));
    assert_eq!(info["keys"], serde_json::json!(q.keys()));
    assert_eq!(info["exit_threshold"], 1.0);
    assert_eq!(info["frames"], 64);
}

#[test]
fn a_fixed_seed_gives_a_fixed_episode_and_fixed_answers() {
    let mut p = Probe::load(GOLDEN).unwrap();
    assert!(p.has_memory());
    let seed = SEEDS[0];
    assert_eq!(p.start(seed), p.start(seed));
    let a = p.run_episode(seed).unwrap();
    let b = p.run_episode(seed).unwrap();
    assert_eq!(a.asks, b.asks);
    assert_eq!(a.decisions, FRAMES as usize);
    // Frame by frame, the same answers as the whole episode.
    let s = steps(&mut p, seed, FRAMES as usize);
    assert!(s.last().unwrap().done && p.done());
    assert!(p.step().is_err(), "no frame after the last one");
    let asked: Vec<_> = s
        .iter()
        .filter_map(|v| v.correct.map(|c| (v.view.frame, v.view.delay.unwrap(), c)))
        .collect();
    let whole: Vec<_> = a
        .asks
        .iter()
        .map(|x| (x.frame, x.delay, x.correct))
        .collect();
    assert_eq!(asked, whole);
    // Another seed, another episode.
    assert_ne!(describe(SEEDS[1]), describe(seed));
}

#[test]
fn an_ask_is_scored_by_the_game() {
    let mut p = Probe::load(GOLDEN).unwrap();
    for v in steps(&mut p, SEEDS[1], FRAMES as usize) {
        assert_eq!(v.probs.len(), 5);
        assert_eq!(QUESTIONS[0].keys()[argmax(&v.probs)], v.choice);
        match v.view.kind {
            "ask" => assert_eq!(v.correct, Some(v.choice == v.view.answer)),
            _ => {
                assert_eq!(v.correct, None);
                assert_eq!(v.view.answer, "wait");
            }
        }
    }
}

#[test]
fn the_memory_switch_and_the_sleep_passes_are_the_engines() {
    let seed = SEEDS[0];
    let n = 24;
    let mut p = Probe::load(GOLDEN).unwrap();
    let on = steps(&mut p, seed, n);
    p.set_memory("zeroed").unwrap();
    let zeroed = steps(&mut p, seed, n);
    // The first frame reads a zero memory either way.
    assert_eq!(on[0].probs, zeroed[0].probs);
    assert!(
        on.iter().zip(&zeroed).any(|(a, b)| a.probs != b.probs),
        "a zeroed memory answers otherwise"
    );
    // With the memory reset before every frame, a sleep has nothing to replay.
    p.set_sleep_passes(4).unwrap();
    let zeroed4 = steps(&mut p, seed, n);
    assert!(zeroed.iter().zip(&zeroed4).all(|(a, b)| a.probs == b.probs));
    // With the memory on, N = 4 replays the first 8 frames before frame 8.
    p.set_memory("on").unwrap();
    let on4 = steps(&mut p, seed, n);
    assert!(on.iter().zip(&on4).take(8).all(|(a, b)| a.probs == b.probs));
    assert!(on.iter().zip(&on4).skip(8).any(|(a, b)| a.probs != b.probs));
    assert!(p.set_sleep_passes(0).is_err() && p.set_memory("off").is_err());
}

#[test]
fn the_published_files_load_and_only_the_model_has_a_memory() {
    let mut m = Probe::load(SLEEP3).unwrap();
    assert!(m.has_memory());
    let mut c = Probe::load(CONTROL).unwrap();
    assert!(!c.has_memory());
    assert!(c.set_memory("zeroed").is_err());
    let seed = SEEDS[2];
    let n = 16;
    let on = steps(&mut m, seed, n);
    m.set_memory("zeroed").unwrap();
    let zeroed = steps(&mut m, seed, n);
    assert_eq!(on[0].probs, zeroed[0].probs);
    assert!(on.iter().zip(&zeroed).any(|(a, b)| a.probs != b.probs));
    for v in on.iter().chain(&zeroed).chain(&steps(&mut c, seed, n)) {
        let total: f32 = v.probs.iter().sum();
        assert!((total - 1.0).abs() < 1e-4, "{total}");
    }
}
