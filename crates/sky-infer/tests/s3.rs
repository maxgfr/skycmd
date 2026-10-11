//! SPEC-S3 options of the s2 engine on random-weight models: value-fused tokens
//! (`fuse_numbers`), a looped weight-shared state encoder (`state_loops`, exits counted in block
//! applications) and a head count other than `d / 64`, checked against the plain f64 reference
//! of `tests/common` and against an independent implementation of the merge rule.

mod common;

use common::Reference;
use sky_infer::s2::prompt::{fuse_numbers, S2PromptTokens};
use sky_infer::s2::random::{ladder_config, random_gguf, test_tokenizer, WeightFormat};
use sky_infer::{Engine, EngineOptions, QType, TernaryCompute};

fn gguf(name: &str, format: WeightFormat) -> Vec<u8> {
    let mut cfg = ladder_config(name).unwrap();
    // sharp temperatures: random weights give logits close to each other
    cfg.temperatures = vec![
        ("e0.choice".into(), 0.3),
        ("e1.choice.3_5".into(), 0.4),
        ("e3.choice.2".into(), 0.2),
        ("choice".into(), 0.25),
        ("score".into(), 0.25),
        ("e2.noul".into(), 0.3),
        ("noul".into(), 0.2),
    ];
    random_gguf(&cfg, &test_tokenizer(), 11, format).unwrap()
}

fn engine(bytes: &[u8], opts: EngineOptions) -> Engine {
    Engine::from_gguf_with(bytes, opts).unwrap()
}

struct Case {
    qtype: QType,
    ins: &'static str,
    opts: Vec<&'static str>,
    state: &'static str,
}

fn cases() -> Vec<Case> {
    vec![
        Case {
            qtype: QType::Choice,
            ins: "Which action keeps the bird alive?",
            opts: vec!["flap: push the bird up", "wait: let the bird fall"],
            state: "vy -1 dx 35 up 7 dn 9 nxt 1 alt 32",
        },
        Case {
            qtype: QType::Choice,
            ins: "Pick a lane with 2 free slots at speed 3.5",
            opts: vec![
                "lane 1, gap 3.5 m",
                "3 cells left",
                "brake",
                "10-20 m ahead",
            ],
            state: "12 dist 7 w 2 h -1.25 range 10-20 speed 3",
        },
        Case {
            qtype: QType::Score,
            ins: "How risky is the battery level?",
            opts: vec![
                "level 0: negligible",
                "level 1: low",
                "level 2: medium",
                "level 3: high",
                "level 4: critical",
            ],
            state: "bat 31 home 9 hikers 2 wind -4 b2 7 8 9",
        },
        Case {
            qtype: QType::Noul,
            ins: "Will it crash within 0.5 s if it keeps gliding?",
            opts: vec![
                "false: no, the statement does not hold",
                "true: yes, the statement holds",
            ],
            state: "alt 2 vy -6 dx 8 gt 3 gb 30",
        },
    ]
}

/// Engine vs reference on every exit, for all cases.
fn check_against_reference(bytes: &[u8], opts: EngineOptions, tol: f64) {
    let r = Reference::new(bytes);
    assert!(r.cfg.fuse_numbers);
    let mut e = engine(bytes, opts);
    let mut max_dev = 0.0f64;
    for c in cases() {
        let s2 = e.s2_engine_mut().unwrap();
        let st = s2.static_tokens(c.qtype, c.ins, &c.opts);
        let (sids, svals, shv) = s2.state_stream(c.state);
        assert_eq!(st.has_value.len(), st.ids.len());
        let want = r.forward(
            &st.ids,
            &st.values,
            &st.has_value,
            st.q_len,
            &st.options,
            &sids,
            &svals,
            &shv,
            c.qtype,
        );
        let got = e.probs_per_exit(c.qtype, c.ins, &c.opts, c.state).unwrap();
        assert_eq!(got.len(), r.cfg.exits.len());
        let u = 1.0 / c.opts.len() as f64;
        let dev = want
            .iter()
            .flatten()
            .fold(0.0f64, |m, p| m.max((p - u).abs()));
        max_dev = max_dev.max(dev);
        for (g, w) in got.iter().zip(&want) {
            for (a, b) in g.iter().zip(w) {
                assert!(
                    (*a as f64 - b).abs() < tol,
                    "{:?} {}: {g:?} vs {w:?}",
                    c.qtype,
                    c.ins
                );
            }
        }
    }
    // random weights still give far-from-uniform answers (the comparison is meaningful)
    assert!(max_dev > 0.05, "{max_dev}");
}

#[test]
fn matches_reference_f32() {
    for name in ["s3-pico", "s3-nano"] {
        check_against_reference(
            &gguf(name, WeightFormat::F32),
            EngineOptions::default(),
            1e-4,
        );
    }
}

#[test]
fn matches_reference_f16_and_ternary() {
    check_against_reference(
        &gguf("s3-pico", WeightFormat::F16),
        EngineOptions::default(),
        1e-4,
    );
    for name in ["s3-pico", "s3-nano"] {
        let t = gguf(name, WeightFormat::Ternary);
        for compute in [TernaryCompute::Packed, TernaryCompute::F32] {
            let opts = EngineOptions {
                ternary: compute,
                ..Default::default()
            };
            check_against_reference(&t, opts, 1e-4);
        }
        // int8 activations (two passes, ~15-bit): a looser tolerance
        let opts = EngineOptions {
            ternary: TernaryCompute::Int8,
            ..Default::default()
        };
        check_against_reference(&t, opts, 1e-3);
    }
}

/// SPEC-S3 §1, written independently of the engine: `(ids, values, has_value)` of an s2 stream
/// (numbers are standalone `[NUM]` tokens) after the merge rule.
fn merge_reference(ids: &[u32], values: &[f32]) -> (Vec<u32>, Vec<f32>, Vec<bool>) {
    const CLS: u32 = 2;
    const SEP: u32 = 3;
    const MASK: u32 = 4;
    const NUM: u32 = 5;
    let mut out: (Vec<u32>, Vec<f32>, Vec<bool>) = Default::default();
    for (&id, &v) in ids.iter().zip(values) {
        if id == NUM {
            if let Some(&prev) = out.0.last() {
                let carries = *out.2.last().unwrap();
                if ![CLS, SEP, MASK].contains(&prev) && !carries {
                    *out.1.last_mut().unwrap() = v;
                    *out.2.last_mut().unwrap() = true;
                    continue;
                }
            }
        }
        out.0.push(id);
        out.1.push(v);
        out.2.push(id == NUM);
    }
    out
}

#[test]
fn merge_rule() {
    let pt = S2PromptTokens {
        cls: 2,
        sep: 3,
        mask: 4,
        num: 5,
        mask_text: "[MASK]",
        fuse: true,
    };
    let run = |ids: &[u32], values: &[f32], from: usize| {
        let (mut i, mut v) = (ids.to_vec(), values.to_vec());
        let mut h: Vec<bool> = ids.iter().map(|&x| x == 5).collect();
        fuse_numbers(&pt, &mut i, &mut v, &mut h, from);
        (i, v, h)
    };
    // [CLS] [NUM]=1 a [NUM]=2 [NUM]=3 b [NUM]=4 [SEP]
    let (i, v, h) = run(
        &[2, 5, 10, 5, 5, 11, 5, 3],
        &[0.0, 1.0, 0.0, 2.0, 3.0, 0.0, 4.0, 0.0],
        0,
    );
    assert_eq!(i, vec![2, 5, 10, 5, 11, 3]);
    assert_eq!(v, vec![0.0, 1.0, 2.0, 3.0, 4.0, 0.0]);
    assert_eq!(h, vec![false, true, true, true, true, false]);
    // a [NUM] at position 0, after [MASK] or [SEP], or after a standalone [NUM], stays standalone
    let (i, _, h) = run(&[5, 5, 7, 5, 4, 5, 3, 5], &[1.0; 8], 0);
    assert_eq!(i, vec![5, 5, 7, 4, 5, 3, 5]);
    assert_eq!(h, vec![true, true, true, false, true, false, true]);
    // merging from an offset: the token before it counts
    let (i, v, _) = run(&[2, 5, 9, 5], &[0.0, 1.0, 0.0, 2.0], 1);
    assert_eq!((i, v), (vec![2, 5, 9], vec![0.0, 1.0, 2.0]));
    let (i, v, h) = run(&[9, 5, 5], &[0.0, 1.0, 2.0], 1);
    assert_eq!((i, v, h), (vec![9, 5], vec![1.0, 2.0], vec![true, true]));

    // engine streams: an s3 file merges exactly what the s2 stream of the same tokenizer holds
    let s3 = gguf("s3-pico", WeightFormat::F32);
    let mut cfg2 = ladder_config("s3-pico").unwrap();
    cfg2.fuse_numbers = false;
    let s2 = random_gguf(&cfg2, &test_tokenizer(), 11, WeightFormat::F32).unwrap();
    let (mut a, mut b) = (
        engine(&s3, EngineOptions::default()),
        engine(&s2, EngineOptions::default()),
    );
    let (fa, fb) = (a.s2_engine_mut().unwrap(), b.s2_engine_mut().unwrap());
    let mut merged_any = false;
    for c in cases() {
        let (ids, values) = fb.state_tokens(c.state);
        let want = merge_reference(&ids, &values);
        assert_eq!(fa.state_stream(c.state), want, "{}", c.state);
        merged_any |= want.0.len() < ids.len();
        // static stream: the head and every option are merged, then truncated
        let sa = fa.static_tokens(c.qtype, c.ins, &c.opts);
        let sb = fb.static_tokens(c.qtype, c.ins, &c.opts);
        let head = merge_reference(&sb.ids[..sb.q_len], &sb.values[..sb.q_len]);
        assert_eq!(sa.ids[..sa.q_len], head.0[..]);
        assert_eq!(sa.has_value[..sa.q_len], head.2[..]);
        assert_eq!(sa.options.len(), sb.options.len());
        for (&(s, l), &(s2, l2)) in sa.options.iter().zip(&sb.options) {
            let o = merge_reference(&sb.ids[s2..s2 + l2], &sb.values[s2..s2 + l2]);
            assert_eq!(sa.ids[s..s + l], o.0[..]);
            assert_eq!(sa.values[s..s + l], o.1[..]);
            assert_eq!(sa.has_value[s..s + l], o.2[..]);
            assert_eq!(sa.ids[s], 4, "the marker stays first");
        }
        // s2 streams flag exactly the [NUM] tokens
        let (ids, _, hv) = fb.state_stream(c.state);
        assert!(ids.iter().zip(&hv).all(|(&i, &h)| h == (i == 5)));
    }
    assert!(merged_any);
    // `10-20` gives two numbers in a row: the second one stays a [NUM] token
    let (ids, values, hv) = fa.state_stream("range 10-20");
    let nums: Vec<(u32, f32, bool)> = ids
        .iter()
        .zip(&values)
        .zip(&hv)
        .filter(|(_, &h)| h)
        .map(|((&i, &v), &h)| (i, v, h))
        .collect();
    assert_eq!(nums.len(), 2);
    assert_ne!(nums[0].0, 5);
    assert_eq!((nums[0].1, nums[1].0, nums[1].1), (10.0, 5, -20.0));

    // the real s2 tokenizer: a game state costs one token per `key value` pair
    let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../data/tok/tok-s2-1024.json");
    if let Ok(json) = std::fs::read_to_string(path) {
        let tok = sky_infer::Tokenizer::from_tokenizer_json(&json).unwrap();
        let cfg = ladder_config("s3-pico").unwrap();
        let bytes = random_gguf(&cfg, &tok, 3, WeightFormat::F32).unwrap();
        let mut e = engine(&bytes, EngineOptions::default());
        let (ids, values, hv) = e
            .s2_engine_mut()
            .unwrap()
            .state_stream("vy -1 dx 35 up 7 dn 9 nxt 1 alt 32");
        assert_eq!(ids.len(), 8, "{ids:?}");
        assert_eq!(
            values[1..7],
            [-1.0, 35.0, 7.0, 9.0, 1.0, 32.0],
            "{values:?}"
        );
        assert!(hv[1..7].iter().all(|&h| h) && !hv[0] && !hv[7]);
    }
}

#[test]
fn loops_exits_and_heads() {
    let bytes = gguf("s3-pico", WeightFormat::F32);
    let mut e = engine(&bytes, EngineOptions::default());
    let info = e.model_info();
    // 1 block x 4 loops: the effective depth is 4, exits count block applications
    assert_eq!((info.num_layers, info.num_heads), (4, 1));
    assert_eq!(info.exits, vec![1, 2, 3, 4]);
    let cfg = &e.s2_model().unwrap().cfg;
    assert_eq!(
        (cfg.state_blocks, cfg.state_loops, cfg.head_dim()),
        (1, 4, 96)
    );
    assert_eq!(e.s2_model().unwrap().state_blocks.len(), 1);
    let c = &cases()[0];
    let all = e.probs_per_exit(c.qtype, c.ins, &c.opts, c.state).unwrap();
    assert_eq!(all.len(), 4);
    // every loop changes the answer
    for w in all.windows(2) {
        assert!(w[0].iter().zip(&w[1]).any(|(a, b)| (a - b).abs() > 1e-4));
    }
    for (tau, exit) in [(0.0, 0), (1.0, 3)] {
        e.set_exit_threshold(Some(tau));
        let p = e.decide_raw(c.qtype, c.ins, &c.opts, c.state).unwrap();
        let s = e.last_stats().unwrap();
        assert_eq!(
            (s.exit, s.layers_used, s.n_exits),
            (Some(exit), Some(exit + 1), 4)
        );
        assert_eq!(p, all[exit]);
    }
    // the default τ: the first exit whose top probability reaches 0.9
    e.set_exit_threshold(None);
    let p = e.decide_raw(c.qtype, c.ins, &c.opts, c.state).unwrap();
    let want = all
        .iter()
        .position(|p| p.iter().cloned().fold(0.0, f32::max) >= 0.9)
        .unwrap_or(3);
    assert_eq!(e.last_stats().unwrap().exit, Some(want));
    assert_eq!(p, all[want]);
    // 2 blocks x 3 loops, exits [2, 4, 6], 2 heads of 64
    let bytes = gguf("s3-nano", WeightFormat::F32);
    let mut e = engine(&bytes, EngineOptions::default());
    let info = e.model_info();
    assert_eq!(
        (info.num_layers, info.num_heads, info.exits.clone()),
        (6, 2, vec![2, 4, 6])
    );
    e.set_exit_threshold(Some(0.0));
    e.decide_raw(c.qtype, c.ins, &c.opts, c.state).unwrap();
    assert_eq!(e.last_stats().unwrap().layers_used, Some(2));
    // parameters: one shared block, plus loop_embd and attn_norm_loop
    let m = e.s2_model().unwrap();
    let (d, l) = (m.cfg.hidden_size, m.cfg.state_loops);
    assert_eq!(m.loop_emb.as_ref().unwrap().len(), l * d);
    assert!(m.attn_norm_loop.is_some());
    assert_eq!(e.n_params(), e.model_info().n_params);
}

#[test]
fn s2_files_keep_their_meaning() {
    // no s3 key: one pass, no merging, h = d / 64 heads (head_count is written anyway)
    let bytes = random_gguf(
        &ladder_config("s2-nano").unwrap(),
        &test_tokenizer(),
        1,
        WeightFormat::F32,
    )
    .unwrap();
    let g = sky_gguf::Gguf::parse(&bytes).unwrap();
    assert!(g.get("skycmd-s2.state_loops").is_none() && g.get("skycmd-s2.fuse_numbers").is_none());
    assert!(g.tensor("loop_embd.weight").is_none());
    let mut e = engine(&bytes, EngineOptions::default());
    let cfg = e.s2_model().unwrap().cfg.clone();
    assert_eq!(
        (
            cfg.state_blocks,
            cfg.state_loops,
            cfg.state_layers,
            cfg.fuse_numbers
        ),
        (4, 1, 4, false)
    );
    let (ids, _, hv) = e.s2_engine_mut().unwrap().state_stream("alt 12 vy -3");
    assert_eq!(ids.iter().filter(|&&i| i == 5).count(), 2);
    assert!(ids.iter().zip(&hv).all(|(&i, &h)| h == (i == 5)));
    // an s3 config the engine cannot run is rejected
    let mut bad = ladder_config("s3-pico").unwrap();
    bad.state_layers = 3;
    assert!(random_gguf(&bad, &test_tokenizer(), 1, WeightFormat::F32).is_err());
}

#[test]
fn caches_and_buffer_reuse_change_nothing() {
    let bytes = gguf("s3-pico", WeightFormat::F32);
    let mut cached = engine(&bytes, EngineOptions::default());
    let mut cold = engine(
        &bytes,
        EngineOptions {
            prefix_cache: 0,
            state_cache: 0,
            ..Default::default()
        },
    );
    for e in [&mut cached, &mut cold] {
        e.set_exit_threshold(Some(1.0));
    }
    // a game loop: two questions per frame on the same state, the state changes every frame
    let (a, b) = (&cases()[0], &cases()[3]);
    for i in 0..12 {
        let state = format!(
            "vy {} dx {} up 7 dn 9 nxt 1 alt {}",
            i % 3 - 1,
            35 - i,
            32 + i
        );
        for c in [a, b] {
            let p = cached.decide_raw(c.qtype, c.ins, &c.opts, &state).unwrap();
            let q = cold.decide_raw(c.qtype, c.ins, &c.opts, &state).unwrap();
            assert_eq!(p, q, "frame {i}");
        }
        assert!(cached.last_stats().unwrap().state_hit);
        assert!(!cold.last_stats().unwrap().state_hit);
    }
    // a longer state after shorter ones, then a shorter one again (reused buffers)
    let long = "a 1 b 2 c 3 d 4 e 5 f 6 g 7 h 8 i 9 j 10 k 11 l 12 m 13 n 14";
    for s in [long, "x 1", long] {
        let p = cached.decide_raw(a.qtype, a.ins, &a.opts, s).unwrap();
        let q = cold.decide_raw(a.qtype, a.ins, &a.opts, s).unwrap();
        assert_eq!(p, q);
    }
    // permutation equivariance holds with the precomputed first fusion block
    let opts = ["over", "around: go 2 cells left", "brake", "continue"];
    let perm = [2, 0, 3, 1];
    let shuffled: Vec<&str> = perm.iter().map(|&i| opts[i]).collect();
    let pa = cached
        .probs_per_exit(QType::Choice, "Pick a maneuver", &opts, "dist 7 w 2")
        .unwrap();
    let pb = cached
        .probs_per_exit(QType::Choice, "Pick a maneuver", &shuffled, "dist 7 w 2")
        .unwrap();
    for (x, y) in pa.iter().zip(&pb) {
        for (j, &i) in perm.iter().enumerate() {
            assert!((x[i] - y[j]).abs() <= 1e-5, "{x:?} vs {y:?}");
        }
    }
}

/// Regression: states longer than the small-attention limit (32 tokens) used to panic (the
/// tiled attention read an emptied position buffer). Doom states have 14-30 `key value` pieces
/// (~38 tokens unfused); the longest are truncated to `max_state_tokens`.
#[test]
fn long_states_fused_and_unfused() {
    let doom = "ea -3 ed 5 al 1 en 2 hp 8 am 20 wf 6 wl 2 wr 3 dr 0 gb 4 gd 12 pb 0 pd 0";
    let thirty: String = (0..30)
        .map(|i| format!("k{} {}", (b'a' + (i % 26) as u8) as char, i * 7 - 40))
        .collect::<Vec<_>>()
        .join(" ");
    let thirty = thirty.replace('k', "key ");
    let huge = "word 12 other -3 ".repeat(40);
    let mut s2pico = ladder_config("s2-pico").unwrap();
    s2pico.temperatures = vec![("choice".into(), 0.3)];
    let files = [
        gguf("s3-pico", WeightFormat::F32),
        gguf("s3-nano", WeightFormat::Ternary),
        random_gguf(&s2pico, &test_tokenizer(), 5, WeightFormat::F32).unwrap(),
        random_gguf(
            &ladder_config("s2-nano").unwrap(),
            &test_tokenizer(),
            5,
            WeightFormat::Ternary,
        )
        .unwrap(),
    ];
    let (ins, opts) = (
        "Which action clears the room?",
        ["fire", "turn left", "turn right", "forward"],
    );
    for bytes in &files {
        let r = Reference::new(bytes);
        for compute in [TernaryCompute::F32, TernaryCompute::Packed] {
            let mut e = engine(
                bytes,
                EngineOptions {
                    ternary: compute,
                    ..Default::default()
                },
            );
            let mut lens = Vec::new();
            for state in [doom, thirty.as_str(), huge.as_str(), "x 1"] {
                let s2 = e.s2_engine_mut().unwrap();
                let st = s2.static_tokens(QType::Choice, ins, &opts);
                let (sids, svals, shv) = s2.state_stream(state);
                lens.push(sids.len());
                let want = r.forward(
                    &st.ids,
                    &st.values,
                    &st.has_value,
                    st.q_len,
                    &st.options,
                    &sids,
                    &svals,
                    &shv,
                    QType::Choice,
                );
                let got = e.probs_per_exit(QType::Choice, ins, &opts, state).unwrap();
                for (g, w) in got.iter().zip(&want) {
                    for (a, b) in g.iter().zip(w) {
                        assert!((*a as f64 - b).abs() < 1e-4, "{state}: {g:?} vs {w:?}");
                    }
                }
                e.decide_raw(QType::Choice, ins, &opts, state).unwrap();
            }
            // the long states go past the small-attention path, the longest is truncated
            assert!(lens[1] > 32, "{lens:?}");
            assert_eq!(lens[2], r.cfg.max_state_tokens);
        }
    }
}

fn carry_gguf(format: WeightFormat) -> Vec<u8> {
    let mut cfg = ladder_config("s3-pico").unwrap();
    cfg.temporal_carry = true;
    cfg.temperatures = vec![("choice".into(), 0.25), ("noul".into(), 0.25)];
    random_gguf(&cfg, &test_tokenizer(), 13, format).unwrap()
}

/// The reference probabilities at every exit and the latents of one frame (cold or carried).
fn reference_frame(
    r: &Reference<'_>,
    e: &mut Engine,
    c: &Case,
    state: &str,
    carry: Option<&common::M>,
) -> (Vec<Vec<f64>>, Vec<common::M>) {
    let s2 = e.s2_engine_mut().unwrap();
    let st = s2.static_tokens(c.qtype, c.ins, &c.opts);
    let (sids, svals, shv) = s2.state_stream(state);
    r.forward_carry(
        &st.ids,
        &st.values,
        &st.has_value,
        st.q_len,
        &st.options,
        &sids,
        &svals,
        &shv,
        c.qtype,
        carry,
    )
}

fn assert_close(got: &[f32], want: &[f64], what: &str) {
    for (a, b) in got.iter().zip(want) {
        assert!((*a as f64 - b).abs() < 1e-4, "{what}: {got:?} vs {want:?}");
    }
}

/// SPEC-S3 §7: one carry buffer per stream, used only when the state token ids match the
/// previous frame's; the previous latent is the output of the last block applied.
#[test]
fn temporal_carry_streams() {
    for format in [WeightFormat::F32, WeightFormat::Ternary] {
        let bytes = carry_gguf(format);
        let g = sky_gguf::Gguf::parse(&bytes).unwrap();
        assert!(g.tensor("carry_gate.weight").is_some());
        let r = Reference::new(&bytes);
        assert!(r.cfg.temporal_carry);
        let mut e = engine(&bytes, EngineOptions::default());
        let c = &cases()[0];
        let danger = &cases()[3];
        let frames = [
            "vy -1 dx 35 up 7 dn 9 nxt 1 alt 32",
            "vy 2 dx 33 up 6 dn 10 nxt 1 alt 30",
            "vy 3 dx 31 up 5 dn 11 nxt 0 alt 29",
        ];
        // frame 1: cold, full depth
        e.set_exit_threshold(Some(1.0));
        let p1 = e
            .decide_raw_stream(7, c.qtype, c.ins, &c.opts, frames[0])
            .unwrap();
        let (w1, l1) = reference_frame(&r, &mut e, c, frames[0], None);
        assert_close(&p1, &w1[3], "frame 1");
        assert!(!e.s2_engine_mut().unwrap().stream_carried(7));
        // a second question on the same frame shares its encoding
        e.decide_raw_stream(7, danger.qtype, danger.ins, &danger.opts, frames[0])
            .unwrap();
        assert!(e.last_stats().unwrap().state_hit);
        // frame 2: same keys, carried from the full-depth latent of frame 1
        let p2 = e
            .decide_raw_stream(7, c.qtype, c.ins, &c.opts, frames[1])
            .unwrap();
        assert!(e.s2_engine_mut().unwrap().stream_carried(7));
        let (w2, l2) = reference_frame(&r, &mut e, c, frames[1], Some(&l1[3]));
        assert_close(&p2, &w2[3], "frame 2 (carried)");
        let (cold2, _) = reference_frame(&r, &mut e, c, frames[1], None);
        assert!(
            w2[3]
                .iter()
                .zip(&cold2[3])
                .any(|(a, b)| (a - b).abs() > 1e-3),
            "the carry changes the answer"
        );
        // the stateless paths stay cold
        let q2 = e.decide_raw(c.qtype, c.ins, &c.opts, frames[1]).unwrap();
        assert_close(&q2, &cold2[3], "decide_raw is stateless");
        // frame 3 answers at the first exit: the latent passed on is the depth-1 one
        e.set_exit_threshold(Some(0.0));
        let p3 = e
            .decide_raw_stream(7, c.qtype, c.ins, &c.opts, frames[2])
            .unwrap();
        let (w3, l3) = reference_frame(&r, &mut e, c, frames[2], Some(&l2[3]));
        assert_close(&p3, &w3[0], "frame 3");
        e.set_exit_threshold(Some(1.0));
        let p4 = e
            .decide_raw_stream(7, c.qtype, c.ins, &c.opts, frames[0])
            .unwrap();
        let (w4, _) = reference_frame(&r, &mut e, c, frames[0], Some(&l3[0]));
        assert_close(&p4, &w4[3], "frame 4 (carried from depth 1)");
        // other keys: cold start
        let other = "alt 2 vy -6 dx 8 gt 3 gb 30";
        let p5 = e
            .decide_raw_stream(7, c.qtype, c.ins, &c.opts, other)
            .unwrap();
        assert!(!e.s2_engine_mut().unwrap().stream_carried(7));
        let (w5, _) = reference_frame(&r, &mut e, c, other, None);
        assert_close(&p5, &w5[3], "frame 5 (cold)");
        // streams are independent, and a reset starts cold
        let q = e
            .decide_raw_stream(8, c.qtype, c.ins, &c.opts, frames[1])
            .unwrap();
        assert_close(&q, &cold2[3], "new stream");
        e.decide_raw_stream(8, c.qtype, c.ins, &c.opts, frames[0])
            .unwrap();
        e.reset_stream(8);
        let q = e
            .decide_raw_stream(8, c.qtype, c.ins, &c.opts, frames[1])
            .unwrap();
        assert_close(&q, &cold2[3], "after reset");
        // JSON requests stay stateless
        let req = format!(
            r#"{{"state": "{}", "questions": {{"a": {{"type": "noul", "instructions": "x"}}}}}}"#,
            frames[1]
        );
        let a = e.decide_json(&req).unwrap();
        assert_eq!(a, e.decide_json(&req).unwrap());
    }
    // a model without temporal_carry: streams answer like decide_raw
    let bytes = gguf("s3-pico", WeightFormat::F32);
    let mut e = engine(&bytes, EngineOptions::default());
    let c = &cases()[0];
    let s = "vy -1 dx 35 up 7 dn 9 nxt 1 alt 32";
    let a = e.decide_raw_stream(1, c.qtype, c.ins, &c.opts, s).unwrap();
    e.decide_raw_stream(1, c.qtype, c.ins, &c.opts, "vy 2 dx 3")
        .unwrap();
    let b = e.decide_raw_stream(1, c.qtype, c.ins, &c.opts, s).unwrap();
    assert_eq!(a, b);
    assert_eq!(a, e.decide_raw(c.qtype, c.ins, &c.opts, s).unwrap());
}
