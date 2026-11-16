//! The s6-ear v2 forward pass on the CPU (native and wasm32), mirroring sky-train `turn::model`.
//!
//! A window of `P` positions runs as a sequence of `P` rows (no padding): RoPE scores depend only
//! on relative positions, so a short window gives the outputs of the masked training graph.

use sky_gguf::Gguf;

use super::TurnConfig;
use crate::audio::{normalize_window, TurnFrontEnd};
use crate::kernels::{self, Attention, AttnScratch, Rope, SelfAttnScratch};
use crate::s2::model::{Lin, Ln, Src};
use crate::s2::TernaryCompute;
use crate::{Error, Result};

/// Longest sequence attended with the plain kernel instead of the tiled one (as in s2).
const SMALL_ATTN: usize = 32;

/// One encoder block: pre-LN on both sublayers (weights only), fused QKV, RoPE, full attention,
/// a GELU feed-forward; every matrix without bias.
#[derive(Clone, Debug)]
pub struct TurnBlock {
    pub attn_norm: Ln,
    /// `[3d, d]`.
    pub qkv: Lin,
    pub out: Lin,
    pub ffn_norm: Ln,
    /// `[ff, d]`.
    pub up: Lin,
    /// `[d, ff]`.
    pub down: Lin,
}

/// A loaded s6-ear v2 model.
#[derive(Clone, Debug)]
pub struct TurnModel {
    pub cfg: TurnConfig,
    /// `[d, hops_per_position · 43]` with bias.
    pub patch_proj: Lin,
    pub blocks: Vec<TurnBlock>,
    pub out_norm: Ln,
    /// Attention pooling: `[64, d]` + bias, then `[1, 64]` + bias.
    pub pool_hidden: Lin,
    pub pool_score: Lin,
    /// Head: `[64, d]` + bias, then `[1, 64]` + bias.
    pub cls_hidden: Lin,
    pub cls_output: Lin,
    rope: Rope,
}

/// One decision.
#[derive(Clone, Debug, PartialEq)]
pub struct TurnOutput {
    /// The raw logit (before the temperature).
    pub logit: f32,
    /// `σ(logit / temperature)`: the calibrated probability that the turn is finished.
    pub prob: f32,
    /// Positions of the window the decision read.
    pub positions: usize,
}

#[derive(Default)]
struct Scratch {
    hn: Vec<f32>,
    qkv: Vec<f32>,
    att: Vec<f32>,
    up: Vec<f32>,
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

/// `σ(x)`.
#[inline]
pub fn sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

impl TurnModel {
    /// Load an `skycmd-turn` GGUF file (ternary matrices stay packed).
    pub fn from_gguf(g: &Gguf<'_>) -> Result<Self> {
        let cfg = TurnConfig::from_gguf(g)?;
        let src = Src { g };
        let (d, ff) = (cfg.hidden_size, cfg.intermediate_size);
        let blocks = (0..cfg.blocks)
            .map(|n| {
                let p = |s: &str| format!("blk.{n}.{s}");
                Ok(TurnBlock {
                    attn_norm: src.ln(&p("attn_norm"), d, false)?,
                    qkv: src.lin(&p("attn_qkv"), 3 * d, d, false)?,
                    out: src.lin(&p("attn_output"), d, d, false)?,
                    ffn_norm: src.ln(&p("ffn_norm"), d, false)?,
                    up: src.lin(&p("ffn_up"), ff, d, false)?,
                    down: src.lin(&p("ffn_down"), d, ff, false)?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(TurnModel {
            patch_proj: src.lin("patch_proj", d, cfg.input_width(), true)?,
            blocks,
            out_norm: src.ln("out_norm", d, false)?,
            pool_hidden: src.lin("attn_pool.hidden", cfg.pool_hidden, d, true)?,
            pool_score: src.lin("attn_pool.score", 1, cfg.pool_hidden, true)?,
            cls_hidden: src.lin("cls.hidden", cfg.head_hidden, d, true)?,
            cls_output: src.lin("cls.output", 1, cfg.head_hidden, true)?,
            rope: Rope::new(cfg.head_dim(), cfg.rope_theta, cfg.max_positions() + 1),
            cfg,
        })
    }

    /// Load from the bytes of a GGUF file.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        Self::from_gguf(&Gguf::parse(bytes)?)
    }

    fn lins_mut(&mut self) -> Vec<&mut Lin> {
        let mut v: Vec<&mut Lin> = vec![
            &mut self.patch_proj,
            &mut self.pool_hidden,
            &mut self.pool_score,
            &mut self.cls_hidden,
            &mut self.cls_output,
        ];
        for b in &mut self.blocks {
            v.extend([&mut b.qkv, &mut b.out, &mut b.up, &mut b.down]);
        }
        v
    }

    /// Number of scalars of the model (as exported: no training-only projection).
    pub fn n_params(&mut self) -> usize {
        let lins: usize = self.lins_mut().iter().map(|l| l.n_params()).sum();
        let norms: usize = self
            .blocks
            .iter()
            .map(|b| b.attn_norm.n_params() + b.ffn_norm.n_params())
            .sum::<usize>()
            + self.out_norm.n_params();
        lins + norms
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

    fn block(&self, blk: &TurnBlock, x: &mut [f32], t: usize, sc: &mut Scratch) {
        let c = &self.cfg;
        let (d, ff, nh, hd, eps) = (
            c.hidden_size,
            c.intermediate_size,
            c.num_heads,
            c.head_dim(),
            c.norm_eps,
        );
        let w = 3 * d;
        let Scratch {
            hn,
            qkv,
            att,
            up,
            iota,
            attn,
            self_attn,
        } = sc;
        let x = &mut x[..t * d];
        let hn = grow(hn, t * d);
        blk.attn_norm.apply(x, d, eps, hn);
        let qkv = grow(qkv, t * w);
        blk.qkv.forward(hn, t, qkv, false);
        for r in 0..t {
            let row = &mut qkv[r * w..(r + 1) * w];
            for h in 0..nh {
                self.rope.apply(&mut row[h * hd..(h + 1) * hd], r);
                self.rope.apply(&mut row[d + h * hd..d + (h + 1) * hd], r);
            }
        }
        let att = grow(att, t * d);
        if t <= SMALL_ATTN {
            kernels::self_attention_probe(qkv, w, false, t, nh, hd, att, d, self_attn);
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
            a.run(qkv, w, &iota[..t], att, d, attn);
        }
        blk.out.forward(att, t, x, true);
        blk.ffn_norm.apply(x, d, eps, hn);
        let up = grow(up, t * ff);
        blk.up.forward(hn, t, up, false);
        up.iter_mut().for_each(|v| *v = kernels::gelu(*v));
        blk.down.forward(up, t, x, true);
    }

    /// The normalized hidden states `[t][d]` and the pooled vector `[d]` of a prepared window
    /// (`x`: `[t][input_width]`, [`normalize_window`]'s output).
    pub fn encode(&self, x: &[f32], t: usize) -> Result<(Vec<f32>, Vec<f32>)> {
        let c = &self.cfg;
        let d = c.hidden_size;
        if t == 0 {
            return Err(Error::InvalidRequest(
                "the window is shorter than one position".into(),
            ));
        }
        if t > c.max_positions() || x.len() != t * c.input_width() {
            return Err(Error::InvalidRequest(format!(
                "a window of {} values for {t} positions (at most {} positions of {})",
                x.len(),
                c.max_positions(),
                c.input_width()
            )));
        }
        let mut h = vec![0.0; t * d];
        self.patch_proj.forward(x, t, &mut h, false);
        let mut sc = Scratch::default();
        for blk in &self.blocks {
            self.block(blk, &mut h, t, &mut sc);
        }
        let mut hn = vec![0.0; t * d];
        self.out_norm.apply(&h, d, c.norm_eps, &mut hn);
        // attention pooling: Linear(d, 64) → tanh → Linear(64, 1), softmax over the positions
        let mut a = vec![0.0; t * c.pool_hidden];
        self.pool_hidden.forward(&hn, t, &mut a, false);
        a.iter_mut().for_each(|v| *v = v.tanh());
        let mut s = vec![0.0; t];
        self.pool_score.forward(&a, t, &mut s, false);
        let m = s.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut z = 0.0;
        for v in s.iter_mut() {
            *v = (*v - m).exp();
            z += *v;
        }
        let mut pooled = vec![0.0f32; d];
        for (r, &wt) in s.iter().enumerate() {
            let wt = wt / z;
            for (p, &v) in pooled.iter_mut().zip(&hn[r * d..(r + 1) * d]) {
                *p += wt * v;
            }
        }
        Ok((hn, pooled))
    }

    /// One decision on a prepared window.
    pub fn forward(&self, x: &[f32], t: usize) -> Result<TurnOutput> {
        let (_, pooled) = self.encode(x, t)?;
        let c = &self.cfg;
        let mut u = vec![0.0; c.head_hidden];
        self.cls_hidden.forward(&pooled, 1, &mut u, false);
        u.iter_mut().for_each(|v| *v = kernels::gelu(*v));
        let mut z = [0.0f32];
        self.cls_output.forward(&u, 1, &mut z, false);
        Ok(TurnOutput {
            logit: z[0],
            prob: sigmoid(z[0] / c.temperature),
            positions: t,
        })
    }

    /// One decision on raw hops `[n][43]` (the last `window_hops` are read).
    pub fn decide_hops(&self, hops: &[f32], n: usize) -> Result<TurnOutput> {
        let c = &self.cfg;
        let keep = n.min(c.window_hops);
        let tail = &hops[(n - keep) * c.channels..n * c.channels];
        let (x, t) = normalize_window(tail, keep, c.hops_per_position);
        self.forward(&x, t)
    }

    /// One decision on 16 kHz PCM (its last `window_hops` hops), through `fe`.
    pub fn decide_pcm(&self, fe: &mut TurnFrontEnd, pcm: &[f32]) -> Result<TurnOutput> {
        let (h, n) = fe.hops(pcm, self.cfg.window_hops);
        self.decide_hops(&h, n)
    }

    /// True when the calibrated probability says "finished" at threshold `tau`.
    pub fn is_end(&self, out: &TurnOutput, tau: f32) -> bool {
        out.prob >= tau
    }
}
