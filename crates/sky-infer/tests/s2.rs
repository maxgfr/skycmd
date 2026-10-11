//! s2 engine tests on random-weight models: a plain f64 reference forward written straight from
//! SPEC-S2 (explicit attention masks, no caches, no SIMD), early exit, caches, permutation
//! equivariance, ternary storage and the request / response API.

use sky_gguf::{GgmlType, Gguf};
use sky_infer::s2::random::{ladder_config, random_gguf, test_tokenizer, WeightFormat};
use sky_infer::s2::S2Config;
use sky_infer::{Arch, DecisionRequest, Engine, EngineOptions, QType, TernaryCompute};

fn gguf(name: &str, format: WeightFormat, temps: bool) -> Vec<u8> {
    let mut cfg = ladder_config(name).unwrap();
    if temps {
        cfg.temperatures = vec![
            ("e0.choice".into(), 1.7),
            ("e0.choice.3_5".into(), 2.3),
            ("e1.choice.2".into(), 0.6),
            ("score".into(), 1.4),
            ("e1.noul".into(), 0.8),
        ];
    }
    random_gguf(&cfg, &test_tokenizer(), 7, format).unwrap()
}

fn engine(bytes: &[u8], opts: EngineOptions) -> Engine {
    Engine::from_gguf_with(bytes, opts).unwrap()
}

// ----------------------------------------------------------------------------- reference

fn erf(x: f64) -> f64 {
    if x.abs() > 6.0 {
        return x.signum();
    }
    // Maclaurin series: erf(x) = 2/sqrt(pi) * sum (-1)^n x^(2n+1) / (n! (2n+1))
    let mut sum = 0.0f64;
    let mut term = x;
    let mut n = 0.0f64;
    while term.abs() > 1e-17 * sum.abs().max(1e-300) || n < 3.0 {
        sum += term / (2.0 * n + 1.0);
        n += 1.0;
        term *= -x * x / n;
        if n > 400.0 {
            break;
        }
    }
    sum * 2.0 / std::f64::consts::PI.sqrt()
}

fn gelu(x: f64) -> f64 {
    0.5 * x * (1.0 + erf(x / std::f64::consts::SQRT_2))
}

type M = Vec<Vec<f64>>;

struct Reference<'a> {
    g: Gguf<'a>,
    cfg: S2Config,
}

impl<'a> Reference<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        let g = Gguf::parse(bytes).unwrap();
        let cfg = S2Config::from_gguf(&g).unwrap();
        Reference { g, cfg }
    }

    fn raw(&self, name: &str) -> Option<Vec<f64>> {
        let info = self.g.tensor(name)?;
        let v = sky_gguf::dequantize(info.dtype, self.g.tensor_bytes(info)).unwrap();
        Some(v.into_iter().map(f64::from).collect())
    }

    fn t(&self, name: &str) -> Vec<f64> {
        self.raw(name).unwrap_or_else(|| panic!("missing {name}"))
    }

    /// `[n_out][n_in]` weights of a (possibly ternary) matrix.
    fn mat(&self, base: &str, n_out: usize, n_in: usize) -> M {
        if let Some(info) = self.g.tensor(&format!("{base}.tq")) {
            let bytes = self.g.tensor_bytes(info);
            let scale = self.t(&format!("{base}.scale"));
            let rb = n_in.div_ceil(4);
            (0..n_out)
                .map(|r| {
                    (0..n_in)
                        .map(|c| {
                            let code = (bytes[r * rb + c / 4] >> (2 * (c % 4))) & 3;
                            let t = match code {
                                0 => 0.0,
                                1 => 1.0,
                                2 => -1.0,
                                _ => panic!("bad code"),
                            };
                            t * scale[r]
                        })
                        .collect()
                })
                .collect()
        } else {
            let w = self.t(&format!("{base}.weight"));
            w.chunks(n_in).map(|r| r.to_vec()).collect()
        }
    }

    fn lin(&self, x: &M, base: &str, n_out: usize, n_in: usize, bias: bool) -> M {
        let w = self.mat(base, n_out, n_in);
        let b = if bias {
            Some(self.t(&format!("{base}.bias")))
        } else {
            None
        };
        x.iter()
            .map(|row| {
                (0..n_out)
                    .map(|o| {
                        let s: f64 = row.iter().zip(&w[o]).map(|(a, c)| a * c).sum();
                        s + b.as_ref().map_or(0.0, |b| b[o])
                    })
                    .collect()
            })
            .collect()
    }

    fn ln(&self, x: &M, base: &str, bias: bool) -> M {
        let w = self.t(&format!("{base}.weight"));
        let b = if bias {
            Some(self.t(&format!("{base}.bias")))
        } else {
            None
        };
        let eps = self.cfg.norm_eps as f64;
        x.iter()
            .map(|r| {
                let n = r.len() as f64;
                let mean = r.iter().sum::<f64>() / n;
                let var = r.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / n;
                let rstd = 1.0 / (var + eps).sqrt();
                (0..r.len())
                    .map(|i| (r[i] - mean) * rstd * w[i] + b.as_ref().map_or(0.0, |b| b[i]))
                    .collect()
            })
            .collect()
    }

    fn embed(&self, ids: &[u32], values: &[f32]) -> M {
        let d = self.cfg.hidden_size;
        let emb: Vec<f64> = match self.raw("token_embd.weight") {
            Some(e) => e,
            None => {
                let info = self.g.tensor("token_embd.q8").unwrap();
                let q = self.g.tensor_bytes(info);
                let s = self.t("token_embd.scale");
                q.iter()
                    .enumerate()
                    .map(|(i, &b)| b as i8 as f64 * s[i / d])
                    .collect()
            }
        };
        ids.iter()
            .zip(values)
            .map(|(&id, &v)| {
                let mut row = emb[id as usize * d..(id as usize + 1) * d].to_vec();
                if id == self.cfg.num_token_id {
                    // phi(v) = [u, sin(u w_k), cos(u w_k)], w_k = 2^k pi / 8
                    let v = v as f64;
                    let u = v.signum() * v.abs().ln_1p();
                    let u = if v == 0.0 { 0.0 } else { u };
                    let mut phi = vec![u];
                    let w = |k: i32| 2f64.powi(k) * std::f64::consts::PI / 8.0;
                    phi.extend((0..8).map(|k| (u * w(k)).sin()));
                    phi.extend((0..8).map(|k| (u * w(k)).cos()));
                    let p = self.lin(&vec![phi], "num_proj", d, 17, true);
                    for (a, b) in row.iter_mut().zip(&p[0]) {
                        *a += b;
                    }
                }
                row
            })
            .collect()
    }

    fn rope(&self, v: &mut [f64], pos: usize) {
        let hd = v.len();
        let half = hd / 2;
        for i in 0..half {
            let f = 1.0 / (self.cfg.rope_theta as f64).powf(2.0 * i as f64 / hd as f64);
            let a = pos as f64 * f;
            let (x1, x2) = (v[i], v[i + half]);
            v[i] = x1 * a.cos() - x2 * a.sin();
            v[i + half] = x2 * a.cos() + x1 * a.sin();
        }
    }

    /// Multi-head attention of `q` rows over `k`/`v` rows where `allowed(i, j)`.
    fn mha(&self, q: &M, k: &M, v: &M, allowed: &dyn Fn(usize, usize) -> bool) -> M {
        let d = self.cfg.hidden_size;
        let hd = self.cfg.head_dim();
        let mut out = vec![vec![0.0; d]; q.len()];
        for h in 0..d / hd {
            let r = h * hd..(h + 1) * hd;
            for i in 0..q.len() {
                let js: Vec<usize> = (0..k.len()).filter(|&j| allowed(i, j)).collect();
                let s: Vec<f64> = js
                    .iter()
                    .map(|&j| {
                        q[i][r.clone()]
                            .iter()
                            .zip(&k[j][r.clone()])
                            .map(|(a, b)| a * b)
                            .sum::<f64>()
                            / (hd as f64).sqrt()
                    })
                    .collect();
                let m = s.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let e: Vec<f64> = s.iter().map(|x| (x - m).exp()).collect();
                let z: f64 = e.iter().sum();
                for (&j, p) in js.iter().zip(&e) {
                    for c in r.clone() {
                        out[i][c] += p / z * v[j][c];
                    }
                }
            }
        }
        out
    }

    fn block(
        &self,
        x: &mut M,
        prefix: &str,
        n: usize,
        pos: &[usize],
        allowed: &dyn Fn(usize, usize) -> bool,
    ) {
        let (d, ff) = (self.cfg.hidden_size, self.cfg.intermediate_size);
        let hd = self.cfg.head_dim();
        let p = |s: &str| format!("{prefix}{n}.{s}");
        let h = if n == 0 {
            x.clone()
        } else {
            self.ln(x, &p("attn_norm"), false)
        };
        let qkv = self.lin(&h, &p("attn_qkv"), 3 * d, d, false);
        let mut q: M = qkv.iter().map(|r| r[..d].to_vec()).collect();
        let mut k: M = qkv.iter().map(|r| r[d..2 * d].to_vec()).collect();
        let v: M = qkv.iter().map(|r| r[2 * d..].to_vec()).collect();
        for (i, &ps) in pos.iter().enumerate() {
            for hh in 0..d / hd {
                self.rope(&mut q[i][hh * hd..(hh + 1) * hd], ps);
                self.rope(&mut k[i][hh * hd..(hh + 1) * hd], ps);
            }
        }
        let a = self.mha(&q, &k, &v, allowed);
        let o = self.lin(&a, &p("attn_output"), d, d, false);
        add(x, &o);
        let h = self.ln(x, &p("ffn_norm"), false);
        let up = self.lin(&h, &p("ffn_up"), 2 * ff, d, false);
        let act: M = up
            .iter()
            .map(|u| (0..ff).map(|j| gelu(u[j]) * u[ff + j]).collect())
            .collect();
        let o = self.lin(&act, &p("ffn_down"), d, ff, false);
        add(x, &o);
    }

    /// Probabilities at every exit.
    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        st_ids: &[u32],
        st_vals: &[f32],
        q_len: usize,
        options: &[(usize, usize)],
        state_ids: &[u32],
        state_vals: &[f32],
        qtype: QType,
    ) -> Vec<Vec<f64>> {
        let c = &self.cfg;
        let d = c.hidden_size;
        // static stream
        let t = st_ids.len();
        let group: Vec<Option<usize>> = (0..t)
            .map(|i| options.iter().position(|&(s, l)| i >= s && i < s + l))
            .collect();
        let pos: Vec<usize> = (0..t)
            .map(|i| match group[i] {
                None => i,
                Some(o) => q_len + i - options[o].0,
            })
            .collect();
        let allowed =
            |i: usize, j: usize| j < q_len || (group[i].is_some() && group[i] == group[j]);
        let mut x = self.embed(st_ids, st_vals);
        for n in 0..c.static_layers {
            self.block(&mut x, "static.blk.", n, &pos, &allowed);
        }
        let x = self.ln(&x, "static_norm", false);
        let markers: M = options.iter().map(|&(s, _)| x[s].clone()).collect();
        let q_tokens: M = x[..q_len].to_vec();
        // state stream
        let s = state_ids.len();
        let spos: Vec<usize> = (0..s).collect();
        let mut h = self.embed(state_ids, state_vals);
        let te = self.t("type_embd.weight");
        let mut out = Vec::new();
        let mut done = 0;
        for (e, &depth) in c.exits.iter().enumerate() {
            while done < depth {
                self.block(&mut h, "blk.", done, &spos, &|_, _| true);
                done += 1;
            }
            let he = self.ln(&h, &format!("exit.{e}.norm"), false);
            let kv_in: M = he.iter().chain(q_tokens.iter()).cloned().collect();
            let mut xm: M = markers
                .iter()
                .map(|m| (0..d).map(|i| m[i] + te[qtype.index() * d + i]).collect())
                .collect();
            for b in 0..c.fusion_layers {
                let p = |s: &str| format!("fusion.{b}.{s}");
                let hq = self.ln(&xm, &p("q_norm"), true);
                let q = self.lin(&hq, &p("attn_q"), d, d, true);
                let hk = self.ln(&kv_in, &p("kv_norm"), true);
                let kv = self.lin(&hk, &p("attn_kv"), 2 * d, d, true);
                let k: M = kv.iter().map(|r| r[..d].to_vec()).collect();
                let v: M = kv.iter().map(|r| r[d..].to_vec()).collect();
                let a = self.mha(&q, &k, &v, &|_, _| true);
                let o = self.lin(&a, &p("attn_output"), d, d, true);
                add(&mut xm, &o);
                let hf = self.ln(&xm, &p("ffn_norm"), true);
                let up: M = self
                    .lin(&hf, &p("ffn_up"), 2 * d, d, true)
                    .into_iter()
                    .map(|r| r.into_iter().map(|v| v.max(0.0)).collect())
                    .collect();
                let o = self.lin(&up, &p("ffn_down"), d, 2 * d, true);
                add(&mut xm, &o);
            }
            let hc = self.ln(&xm, "cls.norm", true);
            let g: M = self
                .lin(&hc, "cls", d, d, true)
                .into_iter()
                .map(|r| r.into_iter().map(gelu).collect())
                .collect();
            let logits: Vec<f64> = self
                .lin(&g, "cls.output", 1, d, true)
                .iter()
                .map(|r| r[0])
                .collect();
            let bucket = sky_infer::prompt::temp_bucket(logits.len());
            let temp = c.temperature(e, qtype.name(), bucket) as f64;
            let m = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let ex: Vec<f64> = logits.iter().map(|l| ((l - m) / temp).exp()).collect();
            let z: f64 = ex.iter().sum();
            out.push(ex.iter().map(|v| v / z).collect());
        }
        out
    }
}

fn add(x: &mut M, y: &M) {
    for (a, b) in x.iter_mut().zip(y) {
        for (p, q) in a.iter_mut().zip(b) {
            *p += q;
        }
    }
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
            ins: "Which action keeps the drone alive?",
            opts: vec!["thrust: push the drone up", "glide: let it fall"],
            state: "alt 12 vy -3 dx 40 gt 20 gb 35",
        },
        Case {
            qtype: QType::Choice,
            ins: "Pick a maneuver at speed 3.5",
            opts: vec!["over", "around: go 2 cells left", "brake", "continue"],
            state: "dist 7 w 2 h -1.25 speed 3",
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
            state: "bat 31 home 9 hikers 2 wind -4",
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
    let mut e = engine(bytes, opts);
    let mut max_dev = 0.0f64;
    for c in cases() {
        let s2 = e.s2_engine_mut().unwrap();
        let st = s2.static_tokens(c.qtype, c.ins, &c.opts);
        let (sids, svals) = s2.state_tokens(c.state);
        let want = r.forward(
            &st.ids,
            &st.values,
            st.q_len,
            &st.options,
            &sids,
            &svals,
            c.qtype,
        );
        let got = e.probs_per_exit(c.qtype, c.ins, &c.opts, c.state).unwrap();
        assert_eq!(got.len(), want.len());
        // random weights still give far-from-uniform answers (the comparison is meaningful)
        let u = 1.0 / c.opts.len() as f64;
        let dev = want
            .iter()
            .flatten()
            .fold(0.0f64, |m, p| m.max((p - u).abs()));
        assert!(dev > 0.002, "{want:?}");
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
    assert!(max_dev > 0.05, "{max_dev}");
}

#[test]
fn matches_reference_f32() {
    for name in ["s2-pico", "s2-nano"] {
        check_against_reference(
            &gguf(name, WeightFormat::F32, true),
            EngineOptions::default(),
            1e-4,
        );
    }
}

#[test]
fn matches_reference_f16_and_ternary() {
    check_against_reference(
        &gguf("s2-pico", WeightFormat::F16, false),
        EngineOptions::default(),
        1e-4,
    );
    let t = gguf("s2-nano", WeightFormat::Ternary, true);
    for compute in [TernaryCompute::Packed, TernaryCompute::F32] {
        let opts = EngineOptions {
            ternary: compute,
            ..Default::default()
        };
        check_against_reference(&t, opts, 1e-4);
    }
}

#[test]
fn ternary_file_is_small_and_packed_compute_agrees() {
    let f = gguf("s2-nano", WeightFormat::Ternary, false);
    let full = gguf("s2-nano", WeightFormat::F32, false);
    assert!(f.len() * 5 < full.len(), "{} vs {}", f.len(), full.len());
    let mut a = engine(
        &f,
        EngineOptions {
            ternary: TernaryCompute::Packed,
            ..Default::default()
        },
    );
    let mut b = engine(
        &f,
        EngineOptions {
            ternary: TernaryCompute::F32,
            ..Default::default()
        },
    );
    assert!(a.model_info().ternary_packed && !b.model_info().ternary_packed);
    assert_eq!(a.model_info().weight_format, "ternary");
    for c in cases() {
        let pa = a.probs_per_exit(c.qtype, c.ins, &c.opts, c.state).unwrap();
        let pb = b.probs_per_exit(c.qtype, c.ins, &c.opts, c.state).unwrap();
        for (x, y) in pa.iter().flatten().zip(pb.iter().flatten()) {
            assert!((x - y).abs() < 1e-5);
        }
    }
}

#[test]
fn early_exit_and_threshold_override() {
    let bytes = gguf("s2-nano", WeightFormat::F32, true);
    let c = &cases()[1];
    let mut e = engine(&bytes, EngineOptions::default());
    let all = e.probs_per_exit(c.qtype, c.ins, &c.opts, c.state).unwrap();
    assert_eq!(all.len(), 3);
    // tau = 0: the first exit answers
    e.set_exit_threshold(Some(0.0));
    let p = e.decide_raw(c.qtype, c.ins, &c.opts, c.state).unwrap();
    let s = e.last_stats().unwrap().clone();
    assert_eq!((s.exit, s.layers_used, s.n_exits), (Some(0), Some(1), 3));
    assert_eq!(p, all[0]);
    // tau = 1: full depth
    e.set_exit_threshold(Some(1.0));
    let p = e.decide_raw(c.qtype, c.ins, &c.opts, c.state).unwrap();
    assert_eq!(e.last_stats().unwrap().layers_used, Some(4));
    assert_eq!(p, all[2]);
    // the default tau (0.9): the first exit whose top probability reaches it
    e.set_exit_threshold(None);
    let p = e.decide_raw(c.qtype, c.ins, &c.opts, c.state).unwrap();
    let want = all
        .iter()
        .position(|p| p.iter().cloned().fold(0.0, f32::max) >= 0.9)
        .unwrap_or(2);
    assert_eq!(e.last_stats().unwrap().exit, Some(want));
    assert_eq!(p, all[want]);
    // the option applies at load
    let mut f = engine(
        &bytes,
        EngineOptions {
            exit_threshold: Some(0.0),
            ..Default::default()
        },
    );
    f.decide_raw(c.qtype, c.ins, &c.opts, c.state).unwrap();
    assert_eq!(f.last_stats().unwrap().exit, Some(0));
}

#[test]
fn caches_hit_and_change_nothing() {
    let bytes = gguf("s2-pico", WeightFormat::F32, false);
    let c = &cases()[0];
    let mut e = engine(&bytes, EngineOptions::default());
    let p1 = e.decide_raw(c.qtype, c.ins, &c.opts, c.state).unwrap();
    let s = e.last_stats().unwrap().clone();
    assert!(!s.static_hit && !s.state_hit);
    let p2 = e
        .decide_raw(c.qtype, c.ins, &c.opts, "alt 11 vy -2 dx 39 gt 20 gb 35")
        .unwrap();
    let s = e.last_stats().unwrap().clone();
    assert!(s.static_hit && !s.state_hit);
    assert_ne!(p1, p2);
    // another question on the same state reuses the state encoding
    let d = &cases()[3];
    e.decide_raw(d.qtype, d.ins, &d.opts, "alt 11 vy -2 dx 39 gt 20 gb 35")
        .unwrap();
    let s = e.last_stats().unwrap().clone();
    assert!(!s.static_hit && s.state_hit);
    // same answers with every cache disabled
    let mut f = engine(
        &bytes,
        EngineOptions {
            prefix_cache: 0,
            state_cache: 0,
            ..Default::default()
        },
    );
    let q1 = f.decide_raw(c.qtype, c.ins, &c.opts, c.state).unwrap();
    let q2 = f.decide_raw(c.qtype, c.ins, &c.opts, c.state).unwrap();
    assert!(!f.last_stats().unwrap().static_hit && !f.last_stats().unwrap().state_hit);
    assert_eq!(q1, p1);
    assert_eq!(q2, p1);
    // LRU of 16 entries
    for i in 0..20 {
        e.decide_raw(
            QType::Choice,
            &format!("question {i}"),
            &["a", "b"],
            c.state,
        )
        .unwrap();
    }
    let s2 = e.s2_engine_mut().unwrap();
    assert_eq!(s2.static_cache_len(), 16);
    assert!(s2.is_static_cached(QType::Choice, "question 19", &["a", "b"]));
    assert!(!s2.is_static_cached(QType::Choice, "question 3", &["a", "b"]));
}

#[test]
fn permutation_equivariance() {
    let bytes = gguf("s2-nano", WeightFormat::F32, false);
    let mut e = engine(&bytes, EngineOptions::default());
    let opts = ["over", "around: go 2 cells left", "brake", "continue"];
    let perm = [2, 0, 3, 1];
    let shuffled: Vec<&str> = perm.iter().map(|&i| opts[i]).collect();
    let a = e
        .probs_per_exit(QType::Choice, "Pick a maneuver", &opts, "dist 7 w 2")
        .unwrap();
    let b = e
        .probs_per_exit(QType::Choice, "Pick a maneuver", &shuffled, "dist 7 w 2")
        .unwrap();
    for (pa, pb) in a.iter().zip(&b) {
        for (j, &i) in perm.iter().enumerate() {
            assert!((pa[i] - pb[j]).abs() <= 1e-5, "{pa:?} vs {pb:?}");
        }
    }
}

#[test]
fn numbers_are_values() {
    let bytes = gguf("s2-pico", WeightFormat::F32, false);
    let mut e = engine(&bytes, EngineOptions::default());
    let s2 = e.s2_engine_mut().unwrap();
    let (ids, vals) = s2.state_tokens("alt 12 vy -3.5 b2");
    let nums: Vec<f32> = ids
        .iter()
        .zip(&vals)
        .filter(|(&i, _)| i == 5)
        .map(|(_, &v)| v)
        .collect();
    assert_eq!(nums, vec![12.0, -3.5]);
    assert_eq!(ids.first(), Some(&2));
    assert_eq!(ids.last(), Some(&3));
    // a literal [NUM] in user text is not a number token
    let (ids, _) = s2.state_tokens("x [NUM] y");
    assert!(!ids.contains(&5));
    let st = s2.static_tokens(QType::Choice, "go 3 cells", &["left 2", "right"]);
    assert_eq!(st.ids.iter().filter(|&&i| i == 5).count(), 2);
    assert_eq!(st.values.len(), st.ids.len());
    // the value changes the decision, not just the token
    let p = |e: &mut Engine, s: &str| e.decide_raw(QType::Choice, "go?", &["a", "b"], s).unwrap();
    assert_ne!(p(&mut e, "alt 3"), p(&mut e, "alt 30"));
}

#[test]
fn truncation_rules() {
    let bytes = gguf("s2-pico", WeightFormat::F32, false);
    let mut e = engine(&bytes, EngineOptions::default());
    let s2 = e.s2_engine_mut().unwrap();
    let long = "word ".repeat(200);
    let (ids, vals) = s2.state_tokens(&long);
    assert_eq!(ids.len(), 64);
    assert_eq!(vals.len(), 64);
    assert_eq!(ids[63], 3);
    let st = s2.static_tokens(QType::Choice, &long, &[&long, "b"]);
    // the first option is capped at 48 tokens + marker; the question keeps max(8, budget)
    assert_eq!(st.options[0].1, 49);
    assert_eq!(st.options[1].0, st.options[0].0 + 49);
    let opt_total: usize = st.options.iter().map(|o| o.1).sum();
    assert_eq!(st.q_len, 2 + (96 - opt_total).max(8));
    // many options: shrunk evenly, every option keeps its marker
    let many: Vec<String> = (0..30)
        .map(|i| format!("option number {i} with a long text"))
        .collect();
    let refs: Vec<&str> = many.iter().map(|s| s.as_str()).collect();
    let st = s2.static_tokens(QType::Choice, "pick", &refs);
    assert_eq!(st.options.len(), 30);
    assert!(st.options.iter().all(|o| o.1 == 4));
    let p = e.decide_raw(QType::Choice, "pick", &refs, "x").unwrap();
    assert_eq!(p.len(), 30);
    assert!((p.iter().sum::<f32>() - 1.0).abs() < 1e-5);
}

#[test]
fn json_requests_and_dispatch() {
    let bytes = gguf("s2-pico", WeightFormat::F32, false);
    let mut e = engine(&bytes, EngineOptions::default());
    assert_eq!(e.arch(), Arch::S2);
    assert!(e.s1_model().is_none() && e.s2_model().is_some());
    let info = e.model_info();
    assert_eq!(
        (info.arch.as_str(), info.num_layers, info.head_layers),
        ("skycmd-s2", 3, 1)
    );
    assert_eq!(info.exits, vec![1, 3]);
    assert_eq!(info.n_params, e.n_params());
    let j: serde_json::Value = serde_json::from_str(&info.to_json()).unwrap();
    assert_eq!(j["exits"], serde_json::json!([1, 3]));
    assert_eq!(j["weight_format"], "f32");
    let req = r#"{"state": "alt 12 vy -3", "questions": {
        "action": {"type": "choice", "instructions": "Which action?", "criteria": {"thrust": "go up", "glide": null}},
        "danger": {"type": "noul", "instructions": "Will it crash soon?"},
        "risk": {"type": "score", "instructions": "Risk?", "criteria": ["low", "mid", "high"]}}}"#;
    let out: serde_json::Value = serde_json::from_str(&e.decide_json(req).unwrap()).unwrap();
    for q in ["action", "danger", "risk"] {
        let a = &out["answers"][q];
        let exit = a["exit"].as_u64().unwrap();
        assert!(exit < 2);
        assert_eq!(a["layers_used"].as_u64().unwrap(), [1, 3][exit as usize]);
    }
    assert!(out["answers"]["danger"]["noul"].as_f64().is_some());
    // the second and third questions reused the state encoding
    assert!(e.last_stats().unwrap().state_hit);
    let r = DecisionRequest::from_json(req).unwrap();
    let resp = e.decide(&r).unwrap();
    assert_eq!(resp.exits.len(), 3);
    assert!(resp.input_tokens > 3 * 6);
    // the debugging sequence is static stream + state stream
    let (ids, markers) = e.build_sequence(QType::Choice, "go?", &["a", "b"], "alt 1");
    assert_eq!(markers.len(), 2);
    assert_eq!(ids[markers[0]], 4);
    // s1 files still load as s1
    let s1 = sky_infer::convert::ladder_config("s1-pico").unwrap();
    let tok =
        sky_infer::Tokenizer::from_tokenizer_json(include_str!("fixtures/small_tokenizer.json"))
            .unwrap();
    let b1 = sky_infer::convert::random_gguf(&s1, &tok, 1, GgmlType::F32).unwrap();
    let mut e1 = Engine::from_gguf(&b1).unwrap();
    assert_eq!(e1.arch(), Arch::S1);
    let out: serde_json::Value = serde_json::from_str(&e1.decide_json(req).unwrap()).unwrap();
    assert!(out["answers"]["action"].get("exit").is_none());
    assert_eq!(e1.model_info().arch, "modern-bert");
}

#[test]
fn rejects_bad_files() {
    let bytes = gguf("s2-pico", WeightFormat::F32, false);
    // a tokenizer without [NUM] at id 5
    let cfg = ladder_config("s2-pico").unwrap();
    let tok =
        sky_infer::Tokenizer::from_tokenizer_json(include_str!("fixtures/small_tokenizer.json"))
            .unwrap();
    assert!(random_gguf(&cfg, &tok, 1, WeightFormat::F32).is_err());
    // truncated file
    assert!(Engine::from_gguf(&bytes[..bytes.len() / 2]).is_err());
    // exits not ending at the state depth
    let mut bad = cfg.clone();
    bad.exits = vec![1, 2];
    assert!(random_gguf(&bad, &test_tokenizer(), 1, WeightFormat::F32).is_err());
}

#[test]
fn num_token_absorbs_the_space_on_its_left() {
    // the s2 tokenizers register [NUM] with lstrip = true (like [MASK] in oscar's tokenizer)
    let tok = test_tokenizer();
    let ids = tok.encode("alt [NUM] vy [NUM]");
    let plain = tok.encode("alt vy");
    assert_eq!(ids.iter().filter(|&&i| i == 5).count(), 2);
    assert_eq!(ids.len(), plain.len() + 2, "{ids:?} vs {plain:?}");
    // the flag survives the GGUF export and is read back
    let bytes = gguf("s2-pico", WeightFormat::F32, false);
    let mut e = engine(&bytes, EngineOptions::default());
    let num = e
        .tokenizer()
        .added_tokens()
        .iter()
        .find(|a| a.id == 5)
        .unwrap()
        .clone();
    assert!(num.lstrip && !num.rstrip && num.special);
    let (state_ids, vals) = e.s2_engine_mut().unwrap().state_tokens("alt 12 vy -3");
    let mut want = vec![2];
    want.extend(tok.encode("alt [NUM] vy [NUM]"));
    want.push(3);
    assert_eq!(state_ids, want);
    assert_eq!(vals.iter().filter(|&&v| v != 0.0).count(), 2);
    // the real s2 tokenizers, when present
    let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../data/tok/tok-s2-1024.json");
    if let Ok(json) = std::fs::read_to_string(path) {
        let t = sky_infer::Tokenizer::from_tokenizer_json(&json).unwrap();
        assert_eq!(t.token_to_id("[NUM]"), Some(5));
        let a = t.added_tokens().iter().find(|a| a.id == 5).unwrap();
        assert!(a.lstrip);
        let ids = t.encode("alt [NUM] vy [NUM]");
        assert_eq!(ids.len(), t.encode("alt vy").len() + 2, "{ids:?}");
    }
}
