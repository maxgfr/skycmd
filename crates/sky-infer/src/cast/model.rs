//! The s6-cast forward pass on the CPU (native and wasm32), mirroring sky-train `cast::model`.
//!
//! The context's leading empty patches (left padding of a short history) are dropped instead of
//! masked: RoPE scores depend only on relative positions, so the kept patches keep their absolute
//! positions and give the same outputs as the masked training graph.

use sky_gguf::Gguf;

use super::prep;
use super::CastConfig;
use crate::kernels::{self, Attention, AttnScratch, Rope, SelfAttnScratch};
use crate::s2::model::{Block, Lin, Ln, Src};
use crate::s2::TernaryCompute;
use crate::{Error, Result};

/// Longest stream attended with the plain kernel instead of the tiled one (as in s2).
const SMALL_ATTN: usize = 32;
/// Numerical floor of `‖v‖²` in exclusive self-attention (sky-train `s2::model::XSA_EPS`).
const XSA_EPS: f32 = 1e-6;

/// A cross-attention block of the decoder (the s2 fusion block layout, every norm and linear
/// layer with a bias, ReLU feed-forward of width `2d`).
#[derive(Clone, Debug)]
pub struct DecBlock {
    pub q_norm: Ln,
    pub kv_norm: Ln,
    pub q: Lin,
    /// `[2d, d]`: keys then values.
    pub kv: Lin,
    pub out: Lin,
    pub ffn_norm: Ln,
    pub up: Lin,
    pub down: Lin,
}

/// A loaded s6-cast model.
#[derive(Clone, Debug)]
pub struct CastModel {
    pub cfg: CastConfig,
    /// `[d, 2·patch]` with bias: a patch's values then its observation mask.
    pub patch_proj: Lin,
    pub blocks: Vec<Block>,
    pub loop_emb: Option<Vec<f32>>,
    pub attn_norm_loop: Option<Ln>,
    pub exit_norms: Vec<Ln>,
    /// `[horizon_patches][d]`.
    pub query: Vec<f32>,
    pub dec: Vec<DecBlock>,
    pub out_norm: Ln,
    /// `[patch · levels, d]` with bias.
    pub head: Lin,
    rope: Rope,
}

/// Options of [`CastModel::forecast`].
#[derive(Clone, Copy, Debug, Default)]
pub struct ForecastOptions {
    /// Average with the sign-flipped context's forecast (`(f(x) − f(−x)) / 2` level by level):
    /// two passes, usually a little more accurate (Reverso, TinyCast).
    pub flip: bool,
    /// Run every exit, ignoring `exit_tolerance`.
    pub full_depth: bool,
}

/// A probabilistic forecast.
#[derive(Clone, Debug, PartialEq)]
pub struct Forecast {
    /// Quantile levels.
    pub levels: Vec<f32>,
    /// `[horizon][levels]`, row-major, in the units of the history.
    pub quantiles: Vec<f32>,
    pub horizon: usize,
    /// Exits run by the last pass, and the model's exit count.
    pub exits_used: usize,
    pub n_exits: usize,
    /// Model passes (more than one when the horizon is longer than one pass, ×2 with flip).
    pub passes: usize,
}

impl Forecast {
    /// The forecast at level index `k` over the horizon.
    pub fn level(&self, k: usize) -> Vec<f32> {
        let q = self.levels.len();
        (0..self.horizon)
            .map(|h| self.quantiles[h * q + k])
            .collect()
    }

    /// The median over the horizon.
    pub fn median(&self) -> Vec<f32> {
        self.level(self.levels.len() / 2)
    }
}

/// `log(1 + e^x)`, stable (sky-train uses the same formula).
#[inline]
pub fn softplus(x: f32) -> f32 {
    x.max(0.0) + (-x.abs()).exp().ln_1p()
}

/// Non-crossing quantiles from one step's raw head outputs: the middle output is the median,
/// each level above adds `softplus` of its output to the one below, each level below subtracts.
pub fn monotone(raw: &[f32], out: &mut [f32]) {
    let q = raw.len();
    let m = q / 2;
    out[m] = raw[m];
    for k in m + 1..q {
        out[k] = out[k - 1] + softplus(raw[k]);
    }
    for k in (0..m).rev() {
        out[k] = out[k + 1] - softplus(raw[k]);
    }
}

/// Scratch buffers of a pass.
#[derive(Default)]
struct Scratch {
    hn: Vec<f32>,
    qkv: Vec<f32>,
    att: Vec<f32>,
    up: Vec<f32>,
    act: Vec<f32>,
    iota: Vec<usize>,
    attn: AttnScratch,
    self_attn: SelfAttnScratch,
}

fn grow(v: &mut Vec<f32>, n: usize) -> &mut [f32] {
    if v.len() < n {
        v.resize(n, 0.0);
    }
    &mut v[..n]
}

impl CastModel {
    /// Load an `skycmd-cast` GGUF file (ternary matrices stay packed).
    pub fn from_gguf(g: &Gguf<'_>) -> Result<Self> {
        let cfg = CastConfig::from_gguf(g)?;
        let enc = cfg.encoder();
        let src = Src { g };
        let d = cfg.hidden_size;
        let nq = cfg.quantiles.len();
        let blocks = (0..cfg.state_blocks)
            .map(|n| src.block("blk.", n, &enc))
            .collect::<Result<Vec<_>>>()?;
        let (loop_emb, attn_norm_loop) = if cfg.state_loops > 1 {
            (
                Some(src.f32("loop_embd.weight", &[cfg.state_loops, d])?),
                Some(src.ln("blk.0.attn_norm_loop", d, false)?),
            )
        } else {
            (None, None)
        };
        let exit_norms = (0..cfg.exits.len())
            .map(|e| src.ln(&format!("exit.{e}.norm"), d, false))
            .collect::<Result<Vec<_>>>()?;
        let dec = (0..cfg.decoder_layers)
            .map(|n| {
                let p = |s: &str| format!("dec.{n}.{s}");
                Ok(DecBlock {
                    q_norm: src.ln(&p("q_norm"), d, true)?,
                    kv_norm: src.ln(&p("kv_norm"), d, true)?,
                    q: src.lin(&p("attn_q"), d, d, true)?,
                    kv: src.lin(&p("attn_kv"), 2 * d, d, true)?,
                    out: src.lin(&p("attn_output"), d, d, true)?,
                    ffn_norm: src.ln(&p("ffn_norm"), d, true)?,
                    up: src.lin(&p("ffn_up"), 2 * d, d, true)?,
                    down: src.lin(&p("ffn_down"), d, 2 * d, true)?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(CastModel {
            patch_proj: src.lin("patch_proj", d, 2 * cfg.patch, true)?,
            blocks,
            loop_emb,
            attn_norm_loop,
            exit_norms,
            query: src.f32("cast_query.weight", &[cfg.horizon_patches, d])?,
            dec,
            out_norm: src.ln("out_norm", d, true)?,
            head: src.lin("quant_head", cfg.patch * nq, d, true)?,
            rope: Rope::new(cfg.head_dim(), cfg.rope_theta, cfg.context_patches + 1),
            cfg,
        })
    }

    /// Load from the bytes of a GGUF file.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        Self::from_gguf(&Gguf::parse(bytes)?)
    }

    fn lins_mut(&mut self) -> Vec<&mut Lin> {
        let mut v: Vec<&mut Lin> = vec![&mut self.patch_proj, &mut self.head];
        for b in &mut self.blocks {
            v.extend([&mut b.qkv, &mut b.out, &mut b.down]);
            if let Some(up) = &mut b.up {
                v.push(up);
            }
        }
        for f in &mut self.dec {
            v.extend([&mut f.q, &mut f.kv, &mut f.out, &mut f.up, &mut f.down]);
        }
        v
    }

    /// Number of scalars of the model.
    pub fn n_params(&mut self) -> usize {
        let lins: usize = self.lins_mut().iter().map(|l| l.n_params()).sum();
        let norms: usize = self
            .blocks
            .iter()
            .flat_map(|b| {
                b.attn_norm
                    .iter()
                    .map(Ln::n_params)
                    .chain([b.ffn_norm.n_params()])
                    .chain(b.px_norm.iter().map(Vec::len))
            })
            .sum::<usize>()
            + self.exit_norms.iter().map(Ln::n_params).sum::<usize>()
            + self
                .dec
                .iter()
                .map(|f| f.q_norm.n_params() + f.kv_norm.n_params() + f.ffn_norm.n_params())
                .sum::<usize>()
            + self.out_norm.n_params()
            + self.attn_norm_loop.as_ref().map_or(0, Ln::n_params);
        lins + norms + self.query.len() + self.loop_emb.as_ref().map_or(0, Vec::len)
    }

    /// Set up the compute of every matrix (see [`TernaryCompute`]); f32 matrices are repacked
    /// into panels.
    pub fn prepare_compute(&mut self, mode: TernaryCompute) {
        let mode = mode.resolve();
        for l in self.lins_mut() {
            match mode {
                TernaryCompute::F32 => l.dequantize(),
                TernaryCompute::Int8 => l.ternary_to_i8(),
                TernaryCompute::Lut => l.ternary_to_lut(),
                _ => {}
            }
            l.pack_panels();
        }
    }

    /// One encoder block over `t` rows of `x` at positions `pos` (the s2 block, full attention).
    #[allow(clippy::too_many_arguments)]
    fn block(
        &self,
        blk: &Block,
        attn_norm: Option<&Ln>,
        x: &mut [f32],
        t: usize,
        pos: &[usize],
        sc: &mut Scratch,
    ) {
        let c = &self.cfg;
        let (d, ff, nh, hd, eps) = (
            c.hidden_size,
            c.intermediate_size,
            c.num_heads,
            c.head_dim(),
            c.norm_eps,
        );
        let w = blk.qkv.n_out;
        let Scratch {
            hn,
            qkv,
            att,
            up,
            act,
            iota,
            attn,
            self_attn,
        } = sc;
        let x = &mut x[..t * d];
        let qkv = grow(qkv, t * w);
        match attn_norm {
            Some(n) => {
                let hn = grow(hn, t * d);
                n.apply(x, d, eps, hn);
                blk.qkv.forward(hn, t, qkv, false);
            }
            None => blk.qkv.forward(x, t, qkv, false),
        }
        for (r, &p) in pos.iter().enumerate().take(t) {
            let row = &mut qkv[r * w..(r + 1) * w];
            for h in 0..nh {
                self.rope.apply(&mut row[h * hd..(h + 1) * hd], p);
                self.rope.apply(&mut row[d + h * hd..d + (h + 1) * hd], p);
            }
            if let Some(g) = &blk.px_norm {
                for h in 0..nh {
                    let rho = &mut row[3 * d + h * hd..3 * d + (h + 1) * hd];
                    kernels::rms_norm_inplace(rho, g, eps);
                    self.rope.apply(rho, p);
                }
            }
        }
        let probe = blk.px_norm.is_some();
        let att = grow(att, t * d);
        if t <= SMALL_ATTN {
            kernels::self_attention_probe(qkv, w, probe, t, nh, hd, att, d, self_attn);
        } else {
            if iota.len() < t {
                *iota = (0..t).collect();
            }
            attn.ensure(t, hd);
            let a = Attention {
                k: &qkv[d..],
                v: &qkv[2 * d..],
                kv_stride: w,
                n_kv: t,
                n_heads: nh,
                head_dim: hd,
                window: None,
            };
            let rho = probe.then(|| &qkv[3 * d..]);
            a.run_probe(qkv, rho, w, &iota[..t], att, d, attn);
        }
        if c.xsa {
            for r in 0..t {
                let v = &qkv[r * w + 2 * d..r * w + 3 * d];
                let o = &mut att[r * d..(r + 1) * d];
                for h in 0..nh {
                    kernels::exclusive_inplace(
                        &mut o[h * hd..(h + 1) * hd],
                        &v[h * hd..(h + 1) * hd],
                        XSA_EPS,
                    );
                }
            }
        }
        blk.out.forward(att, t, x, true);
        let hn = grow(hn, t * d);
        blk.ffn_norm.apply(x, d, eps, hn);
        let up = grow(up, t * 2 * ff);
        blk.up
            .as_ref()
            .expect("s6-cast blocks are GeGLU")
            .forward(hn, t, up, false);
        let act = grow(act, t * ff);
        for r in 0..t {
            kernels::geglu(
                &up[r * 2 * ff..(r + 1) * 2 * ff],
                &mut act[r * ff..(r + 1) * ff],
            );
        }
        blk.down.forward(act, t, x, true);
    }

    /// The decoder over one exit's latents `h` (`[t][d]`): normalized quantiles
    /// `[horizon][levels]`.
    fn decode(&self, h: &[f32], t: usize) -> Vec<f32> {
        let c = &self.cfg;
        let (d, nh, hd, eps) = (c.hidden_size, c.num_heads, c.head_dim(), c.norm_eps);
        let nq = c.horizon_patches;
        let mut x = self.query.clone();
        let mut hn = vec![0.0; nq * d];
        let mut q = vec![0.0; nq * d];
        let mut kvn = vec![0.0; t * d];
        let mut kv = vec![0.0; t * 2 * d];
        let mut o = vec![0.0; nq * d];
        let mut u = vec![0.0; nq * 2 * d];
        let mut s = vec![0.0; t];
        let scale = 1.0 / (hd as f32).sqrt();
        for f in &self.dec {
            f.q_norm.apply(&x, d, eps, &mut hn);
            f.q.forward(&hn, nq, &mut q, false);
            f.kv_norm.apply(h, d, eps, &mut kvn);
            f.kv.forward(&kvn, t, &mut kv, false);
            for i in 0..nq {
                for head in 0..nh {
                    let qi = &q[i * d + head * hd..i * d + (head + 1) * hd];
                    let mut mx = f32::NEG_INFINITY;
                    for (j, sj) in s.iter_mut().enumerate() {
                        let kj = &kv[j * 2 * d + head * hd..j * 2 * d + (head + 1) * hd];
                        *sj = qi.iter().zip(kj).map(|(a, b)| a * b).sum::<f32>() * scale;
                        mx = mx.max(*sj);
                    }
                    let mut z = 0.0;
                    for sj in s.iter_mut() {
                        *sj = (*sj - mx).exp();
                        z += *sj;
                    }
                    let oi = &mut o[i * d + head * hd..i * d + (head + 1) * hd];
                    oi.fill(0.0);
                    for (j, &pj) in s.iter().enumerate() {
                        let vj = &kv[j * 2 * d + d + head * hd..j * 2 * d + d + (head + 1) * hd];
                        oi.iter_mut().zip(vj).for_each(|(a, &b)| *a += pj / z * b);
                    }
                }
            }
            f.out.forward(&o, nq, &mut x, true);
            f.ffn_norm.apply(&x, d, eps, &mut hn);
            f.up.forward(&hn, nq, &mut u, false);
            u.iter_mut().for_each(|v| *v = v.max(0.0));
            f.down.forward(&u, nq, &mut x, true);
        }
        self.out_norm.apply(&x, d, eps, &mut hn);
        let levels = c.quantiles.len();
        let mut raw = vec![0.0; nq * c.patch * levels];
        self.head.forward(&hn, nq, &mut raw, false);
        let mut out = vec![0.0; raw.len()];
        for (r, o) in raw.chunks_exact(levels).zip(out.chunks_exact_mut(levels)) {
            monotone(r, o);
        }
        out
    }

    /// One pass over a prepared window (`patches` values and mask, `context_patches · patch`
    /// each): the normalized quantiles `[horizon][levels]` of every exit run. With `full`, every
    /// exit runs; otherwise the pass stops at the first exit whose median moved less than
    /// `exit_tolerance` from the previous exit's.
    pub fn forward_window(&self, vals: &[f32], mask: &[f32], full: bool) -> Result<Vec<Vec<f32>>> {
        let c = &self.cfg;
        let (d, p) = (c.hidden_size, c.patch);
        let n = c.context();
        if vals.len() != n || mask.len() != n {
            return Err(Error::InvalidRequest(format!(
                "window of {} values, expected {n}",
                vals.len()
            )));
        }
        let valid = prep::patch_valid(mask, p);
        let Some(j0) = valid.iter().position(|&v| v) else {
            return Err(Error::InvalidRequest(
                "the context has no observation".into(),
            ));
        };
        let t = c.context_patches - j0;
        let pos: Vec<usize> = (j0..c.context_patches).collect();
        let mut u = vec![0.0; t * 2 * p];
        for (r, j) in (j0..c.context_patches).enumerate() {
            u[r * 2 * p..r * 2 * p + p].copy_from_slice(&vals[j * p..(j + 1) * p]);
            u[r * 2 * p + p..(r + 1) * 2 * p].copy_from_slice(&mask[j * p..(j + 1) * p]);
        }
        let mut x = vec![0.0; t * d];
        self.patch_proj.forward(&u, t, &mut x, false);
        let mut sc = Scratch::default();
        let nb = c.state_blocks;
        let mut outs: Vec<Vec<f32>> = Vec::new();
        let mut hbuf = vec![0.0; t * d];
        let med = c.median_index();
        let levels = c.quantiles.len();
        for a in 0..c.depth() {
            let (lp, l) = (a / nb, a % nb);
            if l == 0 && lp > 0 {
                let e = self.loop_emb.as_ref().expect("looped model has loop_embd");
                for row in x.chunks_exact_mut(d) {
                    row.iter_mut()
                        .zip(&e[lp * d..(lp + 1) * d])
                        .for_each(|(v, &b)| *v += b);
                }
            }
            let blk = &self.blocks[l];
            let norm = match (a, l) {
                (0, _) => None,
                (_, 0) => self.attn_norm_loop.as_ref(),
                _ => blk.attn_norm.as_ref(),
            };
            self.block(blk, norm, &mut x, t, &pos, &mut sc);
            if let Some(e) = c.exits.iter().position(|&ex| ex == a + 1) {
                self.exit_norms[e].apply(&x, d, c.norm_eps, &mut hbuf);
                let q = self.decode(&hbuf, t);
                let stop = !full
                    && c.exit_tolerance > 0.0
                    && outs.last().is_some_and(|prev: &Vec<f32>| {
                        prev.iter()
                            .skip(med)
                            .step_by(levels)
                            .zip(q.iter().skip(med).step_by(levels))
                            .all(|(a, b)| (a - b).abs() < c.exit_tolerance)
                    });
                outs.push(q);
                if stop {
                    break;
                }
            }
        }
        Ok(outs)
    }

    /// One pass on the raw `context` (its last `context()` points): the quantiles of the last
    /// exit run, in the context's units, and the number of exits run.
    fn pass(&self, context: &[f32], full: bool) -> Result<(Vec<f32>, usize)> {
        let c = &self.cfg;
        let (norm, vals, mask) = prep::prepare(context, c.patch, c.context_patches);
        let outs = self.forward_window(&vals, &mask, full)?;
        let used = outs.len();
        let q = outs.into_iter().last().expect("at least one exit");
        Ok((q.into_iter().map(|v| norm.invert(v)).collect(), used))
    }

    /// Quantile forecast of `horizon` steps after `history` (NaN = missing). Horizons longer
    /// than one pass are rolled out: each pass's median is appended to the context of the next.
    pub fn forecast(
        &self,
        history: &[f32],
        horizon: usize,
        opts: ForecastOptions,
    ) -> Result<Forecast> {
        let c = &self.cfg;
        let levels = c.quantiles.len();
        let step = c.horizon();
        let mut ctx: Vec<f32> = history.to_vec();
        let mut quantiles = Vec::with_capacity(horizon * levels);
        let (mut passes, mut used) = (0, 0);
        while quantiles.len() < horizon * levels {
            let (mut q, u) = self.pass(&ctx, opts.full_depth)?;
            passes += 1;
            used = u;
            if opts.flip {
                let neg: Vec<f32> = ctx.iter().map(|v| -v).collect();
                let (qf, _) = self.pass(&neg, opts.full_depth)?;
                passes += 1;
                for h in 0..step {
                    for k in 0..levels {
                        let mirrored = -qf[h * levels + (levels - 1 - k)];
                        q[h * levels + k] = 0.5 * (q[h * levels + k] + mirrored);
                    }
                }
            }
            let take = (horizon - quantiles.len() / levels).min(step);
            quantiles.extend_from_slice(&q[..take * levels]);
            let med = c.median_index();
            ctx.extend((0..take).map(|h| q[h * levels + med]));
        }
        Ok(Forecast {
            levels: c.quantiles.clone(),
            quantiles,
            horizon,
            exits_used: used,
            n_exits: c.exits.len(),
            passes,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn monotone_quantiles_never_cross_and_keep_the_median() {
        let raw = [3.0, -2.0, 0.5, -9.0, 1.25, 4.0, -30.0, 0.0, 2.0];
        let mut out = [0.0; 9];
        monotone(&raw, &mut out);
        assert_eq!(out[4], 1.25);
        assert!(out.windows(2).all(|w| w[0] <= w[1]), "{out:?}");
        assert!((out[5] - (1.25 + softplus(4.0))).abs() < 1e-6);
        assert!((out[3] - (1.25 - softplus(-9.0))).abs() < 1e-6);
        assert!((softplus(0.0) - std::f32::consts::LN_2).abs() < 1e-7);
        assert!((softplus(50.0) - 50.0).abs() < 1e-6 && softplus(-50.0) > 0.0);
    }
}
