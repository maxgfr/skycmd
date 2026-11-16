//! The fast-weight memory of a model that sleeps (docs/s6-sleep-prereg.md §2, the v2 forms of
//! docs/s6-sleep2-prereg.md §4), as sky-train computes it (`s2::memory`).
//!
//! One memory layer follows each state block and is shared across loops, like its block. The
//! state of a stream is one matrix `S` `[hd][hd]` per head and per layer (f32), zero for a new
//! stream.
//!
//! - **Read**, after every application of block `l`, on its output `x` (before that
//!   application's exit is taken): `q = l2norm(heads(LN(x)·Wq))`, `x ← x + merge(S·q)·Wo`. A zero
//!   memory adds nothing, so a stateless question is answered as by the base.
//! - **Write**, once per frame and per layer, from the output `x` of the block's last application
//!   at full depth (before its read), over the frame's `n` tokens:
//!   `k_i = l2norm(heads(LN(x_i)·Wk))`, `v_i = heads(LN(x_i)·Wv)` (form `token`) or
//!   `heads(LN(x_i + x̄)·Wv)` with `x̄` the mean of the `x_i` (form `frame`), `β_i = σ(wβ·x_i)`,
//!   `α = σ(mean_i(wα·x_i) + 6)`, `S ← α·S + (1/n)·Σ_i β_i·(v_i − α·S·k_i)·k_iᵀ`.
//! - **Sleep**: the frames since the last sleep are replayed `N − 1` more times, in order and at
//!   full depth; each replayed frame reads the memory and writes it.
//!
//! `memory_qk = shared` uses one projection `Wqk` for the queries and the keys. Only the write
//! rule of v1 (`memory_write` absent or `average`) is run: the other rules of sky-train's
//! diagnoses were never trained on a model. The auxiliary head of a `memory_aux` training is not
//! in the exported file and plays no role here.

use super::model::{Lin, Ln, S2Workspace, Src, StateEnc};
use super::S2Config;
use crate::{Error, Result};

/// Bias of the decay gate: `α = σ(6) ≈ 0.9975` per frame before training (sky-train
/// `s2::memory::ALPHA_BIAS`).
pub const ALPHA_BIAS: f32 = 6.0;
/// Floor of `‖x‖²` in the key and query normalization (sky-train `s2::memory::NORM_EPS`).
pub const NORM_EPS: f32 = 1e-6;
/// Frames between two sleeps of a stream (`K`, sky-train `s2::memory::SLEEP_EVERY`).
pub const SLEEP_EVERY: usize = 8;

/// The form of a memory (GGUF `memory_qk`, `memory_value`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MemorySpec {
    /// One projection `mem.{l}.qk` gives the queries and the keys (v2's M1); else `q` and `k`.
    pub shared_qk: bool,
    /// Values carry the frame: `v_i = heads(LN(x_i + x̄)·Wv)` (v2's M2); else `heads(LN(x_i)·Wv)`.
    pub frame_value: bool,
}

impl MemorySpec {
    /// The form from the values of the GGUF keys `memory_qk`, `memory_value` and `memory_write`
    /// (absent: v1's form).
    pub fn from_keys(qk: Option<&str>, value: Option<&str>, write: Option<&str>) -> Result<Self> {
        let shared_qk = match qk {
            None | Some("separate") => false,
            Some("shared") => true,
            Some(k) => return Err(Error::Unsupported(format!("memory_qk {k:?}"))),
        };
        let frame_value = match value {
            None | Some("token") => false,
            Some("frame") => true,
            Some(k) => return Err(Error::Unsupported(format!("memory_value {k:?}"))),
        };
        match write {
            None | Some("average") => {}
            Some(k) => {
                return Err(Error::Unsupported(format!(
                    "memory_write {k:?}: the engine runs the write rule of v1 only"
                )))
            }
        }
        Ok(MemorySpec {
            shared_qk,
            frame_value,
        })
    }
}

/// The weights of one memory layer.
#[derive(Clone, Debug)]
pub struct MemLayer {
    /// `mem.{l}.norm` (LayerNorm without bias).
    pub norm: Ln,
    /// `mem.{l}.q`, or `mem.{l}.qk` when shared.
    pub q: Lin,
    /// `mem.{l}.k` (`None` when the projection is shared).
    pub k: Option<Lin>,
    pub v: Lin,
    pub out: Lin,
    /// `mem.{l}.alpha.weight`, `mem.{l}.beta.weight` (`d`).
    pub alpha: Vec<f32>,
    pub beta: Vec<f32>,
}

impl MemLayer {
    pub(crate) fn load(src: &Src<'_, '_>, cfg: &S2Config, l: usize) -> Result<Self> {
        let spec = cfg.memory.expect("a memory config");
        let d = cfg.hidden_size;
        let p = |s: &str| format!("mem.{l}.{s}");
        let (q, k) = if spec.shared_qk {
            (src.lin(&p("qk"), d, d, false)?, None)
        } else {
            (
                src.lin(&p("q"), d, d, false)?,
                Some(src.lin(&p("k"), d, d, false)?),
            )
        };
        Ok(MemLayer {
            norm: src.ln(&p("norm"), d, false)?,
            q,
            k,
            v: src.lin(&p("v"), d, d, false)?,
            out: src.lin(&p("out"), d, d, false)?,
            alpha: src.f32(&p("alpha.weight"), &[d])?,
            beta: src.f32(&p("beta.weight"), &[d])?,
        })
    }

    /// The linear layers (for the compute preparation of the model).
    pub(crate) fn lins_mut(&mut self) -> Vec<&mut Lin> {
        let mut v = vec![&mut self.q, &mut self.v, &mut self.out];
        if let Some(k) = &mut self.k {
            v.push(k);
        }
        v
    }

    pub(crate) fn n_params(&self) -> usize {
        self.norm.n_params()
            + self.q.n_params()
            + self.k.as_ref().map_or(0, Lin::n_params)
            + self.v.n_params()
            + self.out.n_params()
            + self.alpha.len()
            + self.beta.len()
    }
}

/// Floats of the memory of one stream (every layer, `[layer][head][hd][hd]`).
pub fn floats(cfg: &S2Config) -> usize {
    cfg.state_blocks * cfg.num_heads * cfg.head_dim() * cfg.head_dim()
}

/// Scratch buffers of the read and the write (grown on demand).
#[derive(Clone, Debug, Default)]
pub struct MemScratch {
    y: Vec<f32>,
    p: Vec<f32>,
    r: Vec<f32>,
    v: Vec<f32>,
    mean: Vec<f32>,
}

fn grow(v: &mut Vec<f32>, n: usize) -> &mut [f32] {
    if v.len() < n {
        v.resize(n, 0.0);
    }
    &mut v[..n]
}

/// Each `hd`-wide head of each row divided by `sqrt(‖head‖² + NORM_EPS)`.
fn l2norm_heads(x: &mut [f32], hd: usize) {
    for h in x.chunks_exact_mut(hd) {
        let n = (h.iter().map(|v| v * v).sum::<f32>() + NORM_EPS).sqrt();
        h.iter_mut().for_each(|v| *v /= n);
    }
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// The read of a memory layer on the `t` rows of `x` (`[t][d]`), from its memory `s`
/// (`[heads][hd][hd]`): `x ← x + merge(S·q)·Wo`.
pub fn read(m: &MemLayer, cfg: &S2Config, x: &mut [f32], t: usize, s: &[f32], sc: &mut MemScratch) {
    let (d, hd) = (cfg.hidden_size, cfg.head_dim());
    let y = grow(&mut sc.y, t * d);
    m.norm.apply(&x[..t * d], d, cfg.norm_eps, y);
    let q = grow(&mut sc.p, t * d);
    m.q.forward(y, t, q, false);
    l2norm_heads(q, hd);
    let r = grow(&mut sc.r, t * d);
    for (qr, rr) in q.chunks_exact(d).zip(r.chunks_exact_mut(d)) {
        for (h, (qh, rh)) in qr.chunks_exact(hd).zip(rr.chunks_exact_mut(hd)).enumerate() {
            let sh = &s[h * hd * hd..(h + 1) * hd * hd];
            for (i, o) in rh.iter_mut().enumerate() {
                *o = dot(&sh[i * hd..(i + 1) * hd], qh);
            }
        }
    }
    m.out.forward(r, t, &mut x[..t * d], true);
}

/// The write of a memory layer from the block output `x` (`[t][d]`, the frame's `t` tokens) into
/// its memory `s` (`[heads][hd][hd]`): one gated delta-rule step over the frame's tokens.
pub fn write(
    m: &MemLayer,
    cfg: &S2Config,
    x: &[f32],
    t: usize,
    s: &mut [f32],
    sc: &mut MemScratch,
) {
    let spec = cfg.memory.expect("a memory config");
    let (d, nh, hd) = (cfg.hidden_size, cfg.num_heads, cfg.head_dim());
    let x = &x[..t * d];
    let n = t.max(1) as f32;
    // keys: l2norm(heads(LN(x)·Wk))
    let y = grow(&mut sc.y, t * d);
    m.norm.apply(x, d, cfg.norm_eps, y);
    let k = grow(&mut sc.p, t * d);
    m.k.as_ref().unwrap_or(&m.q).forward(y, t, k, false);
    l2norm_heads(k, hd);
    // values: heads(LN(x)·Wv), or heads(LN(x + x̄)·Wv) for form frame
    if spec.frame_value {
        let mean = grow(&mut sc.mean, d);
        mean.fill(0.0);
        for row in x.chunks_exact(d) {
            mean.iter_mut().zip(row).for_each(|(o, &v)| *o += v);
        }
        mean.iter_mut().for_each(|o| *o /= n);
        let xm = grow(&mut sc.r, t * d);
        for (o, row) in xm.chunks_exact_mut(d).zip(x.chunks_exact(d)) {
            for ((o, &a), &b) in o.iter_mut().zip(row).zip(mean.iter()) {
                *o = a + b;
            }
        }
        m.norm.apply(xm, d, cfg.norm_eps, y);
    }
    let v = grow(&mut sc.v, t * d);
    m.v.forward(y, t, v, false);
    // gates on the raw block output
    let alpha = sigmoid(x.chunks_exact(d).map(|r| dot(r, &m.alpha)).sum::<f32>() / n + ALPHA_BIAS);
    // S ← αS, then S += Σ_i (β_i / n) (v_i − αS k_i) k_iᵀ with the kept αS in the error
    s.iter_mut().for_each(|v| *v *= alpha);
    let err = grow(&mut sc.r, t * d);
    for h in 0..nh {
        let sh = &s[h * hd * hd..(h + 1) * hd * hd];
        for i in 0..t {
            let ki = &k[i * d + h * hd..i * d + (h + 1) * hd];
            for a in 0..hd {
                err[i * d + h * hd + a] =
                    v[i * d + h * hd + a] - dot(&sh[a * hd..(a + 1) * hd], ki);
            }
        }
    }
    for i in 0..t {
        let w = sigmoid(dot(&x[i * d..(i + 1) * d], &m.beta)) / n;
        for h in 0..nh {
            let sh = &mut s[h * hd * hd..(h + 1) * hd * hd];
            let ki = &k[i * d + h * hd..i * d + (h + 1) * hd];
            let ei = &err[i * d + h * hd..i * d + (h + 1) * hd];
            for (a, &e) in ei.iter().enumerate() {
                let we = w * e;
                for (o, &kb) in sh[a * hd..(a + 1) * hd].iter_mut().zip(ki) {
                    *o += we * kb;
                }
            }
        }
    }
}

impl super::S2Model {
    /// The write of a frame into its stream's memory `mem` (`[layer][head][hd][hd]`): the state
    /// encoder of `st` (embedded, reading the memory the frame started from) is run to full depth
    /// if it stopped at an early exit, then every layer writes from its block's last output.
    pub fn memory_write(&self, st: &mut StateEnc, mem: &mut [f32], ws: &mut S2Workspace) {
        let c = &self.cfg;
        if self.memory.is_empty() {
            return;
        }
        debug_assert!(st.mem_on, "a frame that reads no memory writes none");
        self.state_advance(st, c.state_layers, ws);
        let (d, s, smax) = (c.hidden_size, st.len, c.max_state_tokens);
        let hh = c.num_heads * c.head_dim() * c.head_dim();
        for (l, m) in self.memory.iter().enumerate() {
            write(
                m,
                c,
                &st.written[l * smax * d..l * smax * d + s * d],
                s,
                &mut mem[l * hh..(l + 1) * hh],
                &mut ws.mem,
            );
        }
    }

    /// One replayed frame of a sleep pass (no question is asked): the state stream `ids`,
    /// `values`, `has_value` is embedded into `st`, read against `mem` at full depth, and written
    /// into `mem`.
    pub fn memory_replay(
        &self,
        st: &mut StateEnc,
        frame: &MemFrame,
        mem: &mut [f32],
        ws: &mut S2Workspace,
    ) -> Result<()> {
        self.state_reset(st, &frame.ids, &frame.values, &frame.has_value)?;
        self.memory_start(st, mem);
        self.memory_write(st, mem, ws);
        Ok(())
    }

    /// The frame embedded in `st` reads `mem` (a copy: the memory of the stream when the frame
    /// starts).
    pub fn memory_start(&self, st: &mut StateEnc, mem: &[f32]) {
        if self.memory.is_empty() {
            return;
        }
        if st.mem.len() != mem.len() {
            st.mem = vec![0.0; mem.len()];
        }
        st.mem.copy_from_slice(mem);
        let need = self.cfg.state_blocks * self.cfg.max_state_tokens * self.cfg.hidden_size;
        if st.written.len() < need {
            st.written = vec![0.0; need];
        }
        st.mem_on = true;
    }
}

/// The state stream of a frame kept for the sleep passes.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct MemFrame {
    pub ids: Vec<u32>,
    pub values: Vec<f32>,
    pub has_value: Vec<bool>,
}

/// How a stream with a memory sleeps and reads (docs/s6-sleep-prereg.md §2).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SleepOptions {
    /// Passes over the frames since the last sleep, the wake pass included (`N`; 1 = no sleep).
    pub passes: usize,
    /// Frames between two sleeps (`K`).
    pub every: usize,
    /// The memory is reset to zero before every frame (the "memory off" switch, sky-train
    /// `s2-probe --memory-off`).
    pub memory_off: bool,
}

impl Default for SleepOptions {
    fn default() -> Self {
        SleepOptions {
            passes: 1,
            every: SLEEP_EVERY,
            memory_off: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::s2::model::LinWeight;

    fn layer(d: usize, seed: u32) -> MemLayer {
        let mut r = seed;
        let mut next = move || {
            r = r.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            ((r >> 8) as f32 / (1u32 << 24) as f32) - 0.5
        };
        let mut lin = |scale: f32| Lin {
            w: LinWeight::F32((0..d * d).map(|_| next() * scale).collect()),
            b: None,
            n_in: d,
            n_out: d,
        };
        let (q, v, out) = (lin(1.0), lin(1.0), lin(0.5));
        MemLayer {
            norm: Ln {
                w: vec![1.0; d],
                b: None,
            },
            q,
            k: None,
            v,
            out,
            alpha: vec![0.0; d],
            beta: vec![0.0; d],
        }
    }

    fn cfg(d: usize, heads: usize, frame: bool) -> S2Config {
        S2Config {
            hidden_size: d,
            num_heads: heads,
            memory: Some(MemorySpec {
                shared_qk: true,
                frame_value: frame,
            }),
            ..crate::s2::random::ladder_config("s3-pico").unwrap()
        }
    }

    #[test]
    fn a_zero_memory_reads_nothing_and_a_written_one_does() {
        let (d, h) = (8, 2);
        let c = cfg(d, h, true);
        let m = layer(d, 3);
        let x0: Vec<f32> = (0..3 * d).map(|i| (i as f32 * 0.37).sin()).collect();
        let mut sc = MemScratch::default();
        let mut s = vec![0.0; h * (d / h) * (d / h)];
        let mut x = x0.clone();
        read(&m, &c, &mut x, 3, &s, &mut sc);
        assert_eq!(x, x0);
        write(&m, &c, &x0, 3, &mut s, &mut sc);
        assert!(s.iter().any(|v| v.abs() > 1e-4));
        read(&m, &c, &mut x, 3, &s, &mut sc);
        assert!(x.iter().zip(&x0).any(|(a, b)| (a - b).abs() > 1e-4));
    }

    #[test]
    fn one_write_is_the_gated_delta_rule() {
        // One token, one head, gates at zero: β = 1/2, α = σ(6); from S = 0 the write is
        // S = (β / n) v kᵀ with k of unit norm, so S·k = v / 2.
        let d = 4;
        let c = cfg(d, 1, false);
        let m = layer(d, 7);
        let x = [0.3, -1.2, 0.8, 0.1];
        let mut sc = MemScratch::default();
        let mut s = vec![0.0; d * d];
        write(&m, &c, &x, 1, &mut s, &mut sc);
        let mut y = vec![0.0; d];
        m.norm.apply(&x, d, c.norm_eps, &mut y);
        let (mut k, mut v) = (vec![0.0; d], vec![0.0; d]);
        m.q.forward(&y, 1, &mut k, false);
        m.v.forward(&y, 1, &mut v, false);
        l2norm_heads(&mut k, d);
        for a in 0..d {
            let sk = dot(&s[a * d..(a + 1) * d], &k);
            assert!((sk - v[a] / 2.0).abs() < 1e-5, "{sk} vs {}", v[a] / 2.0);
        }
    }

    #[test]
    fn only_the_write_rule_of_v1_is_accepted() {
        assert!(MemorySpec::from_keys(Some("shared"), Some("frame"), None).is_ok());
        assert!(MemorySpec::from_keys(None, None, Some("average")).is_ok());
        assert!(MemorySpec::from_keys(None, None, Some("slot")).is_err());
        assert!(MemorySpec::from_keys(Some("tied"), None, None).is_err());
    }
}
