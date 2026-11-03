//! s6-ear (SPEC-S6 §16): the s2 engine with an audio state stream, on random-weight models.

use sky_infer::audio::{LogMel, FRAME_DIM, FRAME_SAMPLES, SAMPLE_RATE};
use sky_infer::s2::random::{ladder_config, random_gguf, test_tokenizer, WeightFormat};
use sky_infer::{Engine, EngineOptions, Error, QType, S2Config};

fn ear_config() -> S2Config {
    let mut cfg = ladder_config("s6-pico").unwrap();
    cfg.name = "s6-ear-test".into();
    cfg.audio_frame = FRAME_DIM;
    cfg.max_state_tokens = 128;
    cfg.temperatures = vec![("choice".into(), 0.2), ("noul".into(), 0.2)];
    cfg
}

fn gguf(cfg: &S2Config, format: WeightFormat) -> Vec<u8> {
    random_gguf(cfg, &test_tokenizer(), 7, format).unwrap()
}

/// A chirp with some noise, 16 kHz.
fn clip(secs: f32, f0: f32) -> Vec<f32> {
    let n = (SAMPLE_RATE as f32 * secs) as usize;
    let mut s = 12345u32;
    (0..n)
        .map(|i| {
            let t = i as f32 / SAMPLE_RATE as f32;
            s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let noise = (s >> 8) as f32 / (1u32 << 24) as f32 - 0.5;
            0.4 * (std::f32::consts::TAU * (f0 + 300.0 * t) * t).sin() + 0.05 * noise
        })
        .collect()
}

const TURN: &str = "Has the speaker finished their turn?";
const KW: &str = "Which word was said?";

#[test]
fn audio_decisions_run_cache_and_keep_the_last_frames() {
    let cfg = ear_config();
    for format in [WeightFormat::F32, WeightFormat::Ternary] {
        let bytes = gguf(&cfg, format);
        let mut e = Engine::from_gguf(&bytes).unwrap();
        assert!(e.is_audio());
        let a = clip(2.0, 220.0);
        let p = e
            .decide_audio(QType::Noul, TURN, &["false", "true"], &a)
            .unwrap();
        assert_eq!(p.len(), 2);
        assert!((p.iter().sum::<f32>() - 1.0).abs() < 1e-5 && p.iter().all(|x| x.is_finite()));
        assert!(!e.last_stats().unwrap().state_hit);
        // a second question on the same clip reuses its state encoding
        let opts = ["yes", "no", "up", "down", "none"];
        let q = e.decide_audio(QType::Choice, KW, &opts, &a).unwrap();
        assert_eq!(q.len(), 5);
        let st = e.last_stats().unwrap();
        assert!(st.state_hit && !st.static_hit);
        assert!(st.tokens > 25); // static tokens + the 25 frames of 2 s
                                 // another clip, another answer
        let b = clip(2.0, 900.0);
        let r = e.decide_audio(QType::Choice, KW, &opts, &b).unwrap();
        assert!(!e.last_stats().unwrap().state_hit);
        assert!(
            q.iter().zip(&r).any(|(x, y)| (x - y).abs() > 1e-4),
            "{q:?} {r:?}"
        );
        // longer than 10.24 s: only the last 128 frames count, aligned on the clip start
        let long = clip(15.0, 300.0);
        let n = long.len() / FRAME_SAMPLES;
        let tail = &long[(n - 128) * FRAME_SAMPLES..];
        let pl = e
            .decide_audio(QType::Noul, TURN, &["false", "true"], &long)
            .unwrap();
        let pt = e
            .decide_audio(QType::Noul, TURN, &["false", "true"], tail)
            .unwrap();
        for (x, y) in pl.iter().zip(&pt) {
            assert!((x - y).abs() < 1e-5, "{pl:?} {pt:?}");
        }
        // under 80 ms there is no frame
        let short = vec![0.1f32; FRAME_SAMPLES - 1];
        assert!(matches!(
            e.decide_audio(QType::Noul, TURN, &["false", "true"], &short),
            Err(Error::InvalidRequest(_))
        ));
    }
}

#[test]
fn early_exit_and_full_depth_agree_at_tau_one() {
    let cfg = ear_config();
    let bytes = gguf(&cfg, WeightFormat::F32);
    let opts = EngineOptions {
        exit_threshold: Some(1.0),
        ..Default::default()
    };
    let mut e = Engine::from_gguf_with(&bytes, opts).unwrap();
    let a = clip(1.0, 440.0);
    let p = e
        .decide_audio(QType::Noul, TURN, &["false", "true"], &a)
        .unwrap();
    let s2 = e.s2_engine_mut().unwrap();
    let all = s2
        .question_audio_all_exits(QType::Noul, TURN, &["false", "true"], &a)
        .unwrap();
    assert_eq!(all.len(), cfg.exits.len());
    let last = all.last().unwrap();
    assert!(p.iter().zip(last).all(|(x, y)| (x - y).abs() < 1e-6));
    // the frames the engine embeds are the stacked log-mel hops
    let (fr, n) = s2.audio_frames(&a);
    assert_eq!((n, &fr), {
        let (f, k) = LogMel::new().frames(&a);
        (k, &f.clone())
    });
    // precomputed frames answer like the PCM they come from
    let pf = e
        .decide_frames(QType::Noul, TURN, &["false", "true"], &fr, n)
        .unwrap();
    assert!(p.iter().zip(&pf).all(|(x, y)| (x - y).abs() < 1e-6));
}

#[test]
fn text_models_refuse_audio_and_audio_keys_round_trip() {
    let text = ladder_config("s6-pico").unwrap();
    let bytes = gguf(&text, WeightFormat::F32);
    let mut e = Engine::from_gguf(&bytes).unwrap();
    assert!(!e.is_audio());
    let a = clip(1.0, 440.0);
    assert!(matches!(
        e.decide_audio(QType::Noul, TURN, &["false", "true"], &a),
        Err(Error::InvalidRequest(_))
    ));
    let g = sky_gguf::Gguf::parse(&bytes).unwrap();
    assert!(g.get("skycmd-s2.state_kind").is_none());
    assert_eq!(S2Config::from_gguf(&g).unwrap().audio_frame, 0);
    let ear = ear_config();
    let bytes = gguf(&ear, WeightFormat::F32);
    let g = sky_gguf::Gguf::parse(&bytes).unwrap();
    assert_eq!(g.get_str("skycmd-s2.state_kind"), Some("audio"));
    let back = S2Config::from_gguf(&g).unwrap();
    assert_eq!(back.audio_frame, FRAME_DIM);
    assert!(g.tensor("audio_proj.weight").is_some() && g.tensor("audio_proj.bias").is_some());
    // n_params counts audio_proj
    let m = Engine::from_gguf(&bytes).unwrap().n_params();
    let t = Engine::from_gguf(&gguf(&text, WeightFormat::F32))
        .unwrap()
        .n_params();
    assert_eq!(m - t, ear.hidden_size * FRAME_DIM + ear.hidden_size);
}
