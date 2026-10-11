//! Native tests of the game sessions with the oracle and with a random-weight model.

use std::path::PathBuf;

use serde_json::Value;
use sky_gguf::GgmlType;
use sky_infer::convert::{ladder_config, random_gguf};
use sky_infer::Tokenizer;
use sky_web::{session, Engine, Model, Policy, Session};

fn random_pico() -> Vec<u8> {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../sky-infer/tests/fixtures/small_tokenizer.json");
    let tok = Tokenizer::from_tokenizer_json(&std::fs::read_to_string(p).unwrap()).unwrap();
    random_gguf(&ladder_config("s1-pico").unwrap(), &tok, 3, GgmlType::F16).unwrap()
}

fn sum(p: &[sky_web::session::OptionProb]) -> f32 {
    p.iter().map(|o| o.p).sum()
}

#[test]
fn unknown_game_is_an_error() {
    assert!(Session::new("pong", 1).is_err());
}

#[test]
fn questions_use_the_canonical_phrasing() {
    let s = Session::new("rescue", 1).unwrap();
    let q = &s.questions()[0];
    assert_eq!(q.id, "move");
    assert_eq!(
        q.instructions,
        "Which move should the rescue drone make next?"
    );
    assert_eq!(q.keys.len(), 7);
    assert_eq!(q.options[0], "north: move one cell north");
    assert_eq!(s.questions()[1].id, "risk");
}

#[test]
fn oracle_flies_flappy_through_pipes() {
    let mut s = Session::new("flappy", 7).unwrap();
    for _ in 0..1500 {
        let r = s.step_ai(Policy::Oracle).unwrap();
        assert!(r.decided);
        assert_eq!(r.probs.len(), 2);
        assert!((sum(&r.probs) - 1.0).abs() < 1e-3);
        let sec = r.secondary.as_ref().unwrap();
        assert_eq!(sec.id, "danger");
        assert!((0.0..=1.0).contains(&sec.value));
        assert!(!r.vetoed, "flappy has no reflex");
        assert!(r.state_text.contains("alt "), "{}", r.state_text);
        if r.done {
            break;
        }
    }
    assert!(s.score() >= 3, "oracle passed only {} pipes", s.score());
}

#[test]
fn oracle_plays_every_game_to_the_end() {
    for kind in ["rescue", "dodge"] {
        let mut s = Session::new(kind, 11).unwrap();
        let mut decisions = 0;
        for _ in 0..20_000 {
            let r = s.step_ai(Policy::Oracle).unwrap();
            decisions += r.decided as u32;
            if r.done {
                break;
            }
        }
        assert!(s.done(), "{kind} did not finish");
        assert!(decisions > 5, "{kind}: {decisions} decisions");
        let snap = s.snapshot();
        assert_eq!(snap["kind"], kind);
        assert_eq!(snap["done"], Value::Bool(true));
        assert!(snap["view"].is_object());
        // A finished game holds.
        assert!(s.step_ai(Policy::Oracle).unwrap().done);
    }
}

#[test]
fn dodge_decides_every_sixth_frame() {
    let mut s = Session::new("dodge", 2).unwrap();
    let decided: Vec<bool> = (0..12)
        .map(|_| s.step_ai(Policy::Oracle).unwrap().decided)
        .collect();
    assert_eq!(decided.iter().filter(|d| **d).count(), 2, "{decided:?}");
    assert!(decided[0] && decided[6]);
}

#[test]
fn throttling_holds_the_last_decision() {
    let mut s = Session::new("flappy", 3).unwrap();
    s.set_decide_every(3);
    let r: Vec<_> = (0..6).map(|_| s.step_ai(Policy::Oracle).unwrap()).collect();
    let decided: Vec<bool> = r.iter().map(|x| x.decided).collect();
    assert_eq!(decided, [true, false, false, true, false, false]);
    assert_eq!(
        r[1].probs.len(),
        2,
        "held frames repeat the last probabilities"
    );
    assert!(s.timing()["n"].as_u64().unwrap() == 2);
}

#[test]
fn human_keys_and_reflex() {
    let mut s = Session::new("flappy", 1).unwrap();
    assert!(s.step_human("thrust").is_ok());
    assert!(s.step_human("").is_ok());
    assert!(s.step_human("jump").is_err());

    let mut d = Session::new("dodge", 1).unwrap();
    assert!(d.action_keys().contains(&"abort"));
    assert!(d.step_human("continue").is_ok());

    // Flying north repeatedly on the rescue grid ends against the border or a rock, where the
    // reflex replaces the move.
    let mut r = Session::new("rescue", 5).unwrap();
    let mut vetoed = false;
    for _ in 0..12 {
        let h = r.step_human("north").unwrap();
        if h.vetoed {
            vetoed = true;
            assert_ne!(h.executed, "north");
            break;
        }
    }
    assert!(vetoed, "the reflex never fired");
}

#[test]
fn model_answers_every_game() {
    let mut engine = Engine::load(&random_pico()).unwrap();
    let m: &mut Model = engine.model_mut();
    assert!(m.n_params() > 100_000);
    assert!(m.file_bytes() > 0);
    let info = m.info();
    assert_eq!(info["hidden_size"], 64);
    assert!(info["file_bytes"].as_u64().unwrap() > 0);
    for (kind, n0, n1) in [("flappy", 2, 2), ("rescue", 7, 5), ("dodge", 4, 2)] {
        let mut s = Session::new(kind, 9).unwrap();
        let r = s.step_ai(Policy::Model(m)).unwrap();
        assert_eq!(r.policy, "model");
        assert!(r.decided);
        assert_eq!(r.probs.len(), n0, "{kind}");
        assert!((sum(&r.probs) - 1.0).abs() < 1e-3, "{kind}");
        let sec = r.secondary.unwrap();
        assert_eq!(sec.probs.len(), n1, "{kind}");
        assert!(sec.value >= 0.0 && sec.value <= sec.max, "{kind}");
        assert!(r.micros >= 0.0);
        let json = serde_json::to_value(&r.probs).unwrap();
        assert!(json[0]["key"].is_string());
    }
}

#[test]
fn bench_reports_latency() {
    let mut engine = Engine::load(&random_pico()).unwrap();
    let b = session::bench(engine.model_mut(), 20).unwrap();
    assert_eq!(b["n"], 20);
    assert_eq!(b["forwards_per_decision"], 2);
    let med = b["median_us"].as_f64().unwrap();
    let p95 = b["p95_us"].as_f64().unwrap();
    assert!(med > 0.0 && p95 >= med, "{b}");
    assert!(b["decisions_per_s"].as_f64().unwrap() > 0.0);
}

#[test]
fn bad_gguf_is_rejected() {
    assert!(Engine::load(b"not a gguf").is_err());
}
