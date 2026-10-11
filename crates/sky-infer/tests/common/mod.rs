//! A plain f64 reference forward of `skycmd-s2` files written straight from SPEC-S2 and SPEC-S3
//! (explicit attention masks, no caches, no SIMD, no folded weights): value-fused tokens through
//! `has_value`, looped state blocks with `loop_embd` and `blk.0.attn_norm_loop`, any head count.

#![allow(dead_code)]

use sky_gguf::Gguf;
use sky_infer::s2::S2Config;
use sky_infer::QType;

pub fn erf(x: f64) -> f64 {
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

pub fn gelu(x: f64) -> f64 {
    0.5 * x * (1.0 + erf(x / std::f64::consts::SQRT_2))
}

pub type M = Vec<Vec<f64>>;

pub struct Reference<'a> {
    pub g: Gguf<'a>,
    pub cfg: S2Config,
}

impl<'a> Reference<'a> {
    pub fn new(bytes: &'a [u8]) -> Self {
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

    /// `has_value` empty: `id == [NUM]` (s2).
    fn embed(&self, ids: &[u32], values: &[f32], has_value: &[bool]) -> M {
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
            .enumerate()
            .map(|(r, (&id, &v))| {
                let mut row = emb[id as usize * d..(id as usize + 1) * d].to_vec();
                let hv = has_value
                    .get(r)
                    .copied()
                    .unwrap_or(id == self.cfg.num_token_id);
                if hv {
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

    /// One block; `norm` is the input norm (`None`: identity).
    #[allow(clippy::too_many_arguments)]
    fn block(
        &self,
        x: &mut M,
        prefix: &str,
        n: usize,
        norm: Option<&str>,
        pos: &[usize],
        allowed: &dyn Fn(usize, usize) -> bool,
    ) {
        let (d, ff) = (self.cfg.hidden_size, self.cfg.intermediate_size);
        let hd = self.cfg.head_dim();
        let p = |s: &str| format!("{prefix}{n}.{s}");
        let h = match norm {
            None => x.clone(),
            Some(name) => self.ln(x, name, false),
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

    /// Probabilities at every exit (`has_value` arrays may be empty: `id == [NUM]`).
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        st_ids: &[u32],
        st_vals: &[f32],
        st_hv: &[bool],
        q_len: usize,
        options: &[(usize, usize)],
        state_ids: &[u32],
        state_vals: &[f32],
        state_hv: &[bool],
        qtype: QType,
    ) -> Vec<Vec<f64>> {
        self.forward_carry(
            st_ids, st_vals, st_hv, q_len, options, state_ids, state_vals, state_hv, qtype, None,
        )
        .0
    }

    /// [`Reference::forward`] with the SPEC-S3 §7 carry: the state stream starts from
    /// `embed + carry_gate ⊙ LN(carry_norm)(carry)`. Also returns the hidden states after every
    /// block application (`[depth][S][d]`, before any exit norm).
    #[allow(clippy::too_many_arguments)]
    pub fn forward_carry(
        &self,
        st_ids: &[u32],
        st_vals: &[f32],
        st_hv: &[bool],
        q_len: usize,
        options: &[(usize, usize)],
        state_ids: &[u32],
        state_vals: &[f32],
        state_hv: &[bool],
        qtype: QType,
        carry: Option<&M>,
    ) -> (Vec<Vec<f64>>, Vec<M>) {
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
        let mut x = self.embed(st_ids, st_vals, st_hv);
        for n in 0..c.static_layers {
            let norm = format!("static.blk.{n}.attn_norm");
            let norm = (n > 0).then_some(norm.as_str());
            self.block(&mut x, "static.blk.", n, norm, &pos, &allowed);
        }
        let x = self.ln(&x, "static_norm", false);
        let markers: M = options.iter().map(|&(s, _)| x[s].clone()).collect();
        let q_tokens: M = x[..q_len].to_vec();
        // state stream
        let s = state_ids.len();
        let spos: Vec<usize> = (0..s).collect();
        let mut h = self.embed(state_ids, state_vals, state_hv);
        if let Some(prev) = carry {
            let gate = self.t("carry_gate.weight");
            let n = self.ln(prev, "carry_norm", false);
            for (row, nr) in h.iter_mut().zip(&n) {
                for ((x, g), v) in row.iter_mut().zip(&gate).zip(nr) {
                    *x += g * v;
                }
            }
        }
        let mut latents = Vec::new();
        let te = self.t("type_embd.weight");
        let mut out = Vec::new();
        let mut done = 0;
        // SPEC-S3 §2: block applications `a` run block `a % B` on pass `a / B`; every pass after
        // the first adds loop_embd[pass] and normalizes block 0's input with attn_norm_loop
        let nb = c.state_blocks;
        let loop_emb = self.raw("loop_embd.weight");
        for (e, &depth) in c.exits.iter().enumerate() {
            while done < depth {
                let (pass, b) = (done / nb, done % nb);
                let norm = if b == 0 && pass > 0 {
                    let le = loop_emb.as_ref().expect("loop_embd.weight");
                    for row in h.iter_mut() {
                        for (v, &l) in row.iter_mut().zip(&le[pass * d..(pass + 1) * d]) {
                            *v += l;
                        }
                    }
                    Some("blk.0.attn_norm_loop".to_string())
                } else if b == 0 {
                    None
                } else {
                    Some(format!("blk.{b}.attn_norm"))
                };
                self.block(&mut h, "blk.", b, norm.as_deref(), &spos, &|_, _| true);
                done += 1;
                latents.push(h.clone());
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
        (out, latents)
    }
}

pub fn add(x: &mut M, y: &M) {
    for (a, b) in x.iter_mut().zip(y) {
        for (p, q) in a.iter_mut().zip(b) {
            *p += q;
        }
    }
}
