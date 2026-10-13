//! Variant B, "iso": the input is cut into segments — the state, the question, and one segment
//! per option rendered by its definition only — and the attention mask makes the model invariant
//! by construction:
//!
//! - the state attends to the state only (its hidden states, hence its keys and values at every
//!   layer, do not depend on the question or the options: they can be cached per state);
//! - the question attends to the state and the question;
//! - option `i` attends to the state, the question and itself, never to another option.
//!
//! Positions: the state takes `0..Ls`, the question `Ls..L`. Every option starts at the same
//! position, either the shared offset `L` (sub-variant i) or 0, the reset of 2610.01601 §III
//! (sub-variant ii). The local window of the ModernBERT local layers is computed on these logical
//! positions. An option's representation is therefore a function of (state, question, its own
//! text) only, the same whatever its rank: permuting the options permutes the logits exactly,
//! renaming them changes nothing (names are never rendered), and swapping the two definitions of
//! a `noul` question swaps its two probabilities.
//!
//! Readout: the mean of the option's hidden states `h`, the mean of the question's `q`, then
//! `MLP([h, q, h ⊙ q, |h - q|])` (4d → d → 1) and a softmax over the options.

use mlx_rs::{error::Exception, ops, Array, Dtype};

use crate::config::{ModelConfig, SpecialTokens, HEAD_DIM};
use crate::model::{self, gelu_erf, layer_norm};
use crate::params::ParamMap;
use crate::s5::data::{Draw, PairArrays, S5Item};

type R<T> = std::result::Result<T, Exception>;

/// Where the options start.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Positions {
    /// Every option starts at the end of the prefix (sub-variant i).
    Shared,
    /// Every option starts at 0 (sub-variant ii, 2610.01601 §III).
    Reset,
}

/// Segment ids.
pub const SEG_PAD: i32 = -1;
pub const SEG_STATE: i32 = 0;
pub const SEG_QUESTION: i32 = 1;
/// Option at display position `j` has segment `SEG_OPTION + j`.
pub const SEG_OPTION: i32 = 2;

/// One iso sequence.
#[derive(Debug, Clone, PartialEq)]
pub struct IsoRow {
    pub ids: Vec<u32>,
    pub seg: Vec<i32>,
    pub pos: Vec<i32>,
    pub qtype: usize,
    /// Target in display order.
    pub target: Vec<f32>,
    pub order: Vec<usize>,
}

impl IsoRow {
    pub fn k(&self) -> usize {
        self.order.len()
    }
}

/// Option token lists under the Laya budget: options are cut evenly when they do not leave 16
/// tokens of `head_max_len` to the question; the question keeps the rest (at least 8).
pub fn budget(head: usize, opts: &mut [Vec<u32>], head_max_len: usize) -> usize {
    let total: usize = opts.iter().map(Vec::len).sum();
    let mut left = head_max_len as i64 - total as i64;
    if left < 16 {
        let per = 4.max((head_max_len as i64 - 16).div_euclid(opts.len().max(1) as i64)) as usize;
        for o in opts.iter_mut() {
            o.truncate(per);
        }
        left = head_max_len as i64 - opts.iter().map(Vec::len).sum::<usize>() as i64;
    }
    head.min(left.max(8) as usize)
}

/// Tokens of the state segment (`[CLS] state [SEP]`): a fixed budget, so that the state's ids,
/// hence its cached keys and values, never depend on the question or the options. The rest
/// (question, its `[SEP]`, options) takes at most `head_max_len + 9` tokens.
pub fn state_budget(max_len: usize, head_max_len: usize) -> usize {
    max_len.saturating_sub(head_max_len + 16).max(8)
}

/// The iso sequence of an item under a draw: `[CLS] state [SEP] | question [SEP] | options`.
/// `None` when the options alone do not fit.
pub fn iso_row(
    item: &S5Item,
    d: &Draw,
    max_len: usize,
    head_max_len: usize,
    positions: Positions,
    sp: &SpecialTokens,
) -> Option<IsoRow> {
    let mut opts: Vec<Vec<u32>> = d
        .order
        .iter()
        .map(|&i| {
            let o = &item.defs[i];
            if o.is_empty() {
                vec![sp.unk]
            } else {
                o.clone()
            }
        })
        .collect();
    let head_keep = budget(item.head.len(), &mut opts, head_max_len);
    let n_opt: usize = opts.iter().map(Vec::len).sum();
    let state_keep = item
        .state
        .len()
        .min(state_budget(max_len, head_max_len) - 2);
    if state_keep + 3 + head_keep + n_opt > max_len {
        return None;
    }
    let mut ids = Vec::with_capacity(state_keep + 3 + head_keep + n_opt);
    let mut seg = Vec::with_capacity(ids.capacity());
    ids.push(sp.cls);
    ids.extend_from_slice(&item.state[..state_keep]);
    ids.push(sp.sep);
    seg.resize(ids.len(), SEG_STATE);
    ids.extend_from_slice(&item.head[..head_keep]);
    ids.push(sp.sep);
    seg.resize(ids.len(), SEG_QUESTION);
    let prefix = ids.len() as i32;
    let mut pos: Vec<i32> = (0..prefix).collect();
    for (j, o) in opts.iter().enumerate() {
        ids.extend_from_slice(o);
        seg.resize(ids.len(), SEG_OPTION + j as i32);
        let start = match positions {
            Positions::Shared => prefix,
            Positions::Reset => 0,
        };
        pos.extend((0..o.len() as i32).map(|t| start + t));
    }
    Some(IsoRow {
        ids,
        seg,
        pos,
        qtype: item.qtype,
        target: d.order.iter().map(|&i| item.target[i]).collect(),
        order: d.order.clone(),
    })
}

/// Batch arrays of variant B.
pub struct IsoBatch {
    pub ids: Array,
    pub seg: Array,
    pub pos: Array,
    /// `[B, K, T]`: 1 / len over the tokens of option `k`.
    pub pool: Array,
    /// `[B, 1, T]`: 1 / len over the question tokens.
    pub qpool: Array,
    pub opt_mask: Array,
    pub target: Array,
    pub is_score: Array,
    pub k: Array,
    pub pairs: PairArrays,
    pub n_tokens: usize,
}

/// Pads iso rows into a batch (length rounded up to a multiple of 16).
pub fn iso_batch(rows: &[IsoRow], pairs: &[(usize, usize)]) -> IsoBatch {
    let b = rows.len();
    let t = rows
        .iter()
        .map(|r| r.ids.len())
        .max()
        .unwrap_or(1)
        .div_ceil(16)
        * 16;
    let k = rows.iter().map(IsoRow::k).max().unwrap_or(1).max(1);
    let mut ids = vec![0i32; b * t];
    let mut seg = vec![SEG_PAD; b * t];
    let mut pos = vec![0i32; b * t];
    let mut pool = vec![0f32; b * k * t];
    let mut qpool = vec![0f32; b * t];
    let mut omask = vec![false; b * k];
    let mut target = vec![0f32; b * k];
    let mut n_tokens = 0;
    for (i, r) in rows.iter().enumerate() {
        n_tokens += r.ids.len();
        let mut counts = vec![0usize; r.k()];
        let mut nq = 0usize;
        for (j, (&id, &s)) in r.ids.iter().zip(&r.seg).enumerate() {
            ids[i * t + j] = id as i32;
            seg[i * t + j] = s;
            pos[i * t + j] = r.pos[j];
            if s >= SEG_OPTION {
                counts[(s - SEG_OPTION) as usize] += 1;
            } else if s == SEG_QUESTION {
                nq += 1;
            }
        }
        for (j, &s) in r.seg.iter().enumerate() {
            if s >= SEG_OPTION {
                let o = (s - SEG_OPTION) as usize;
                pool[(i * k + o) * t + j] = 1.0 / counts[o] as f32;
            } else if s == SEG_QUESTION {
                qpool[i * t + j] = 1.0 / nq.max(1) as f32;
            }
        }
        for o in 0..r.k() {
            omask[i * k + o] = true;
            target[i * k + o] = r.target[o];
        }
    }
    let orders: Vec<Vec<usize>> = rows.iter().map(|r| r.order.clone()).collect();
    let (bi, ti, ki) = (b as i32, t as i32, k as i32);
    IsoBatch {
        ids: Array::from_slice(&ids, &[bi, ti]),
        seg: Array::from_slice(&seg, &[bi, ti]),
        pos: Array::from_slice(&pos, &[bi, ti]),
        pool: Array::from_slice(&pool, &[bi, ki, ti]),
        qpool: Array::from_slice(&qpool, &[bi, 1, ti]),
        opt_mask: Array::from_slice(&omask, &[bi, ki]),
        target: Array::from_slice(&target, &[bi, ki]),
        is_score: Array::from_iter(
            rows.iter()
                .map(|r| f32::from(u8::from(r.qtype == crate::decision::SCORE))),
            &[bi],
        ),
        k: Array::from_iter(rows.iter().map(|r| r.k() as f32), &[bi]),
        pairs: crate::s5::data::pair_arrays(pairs, &orders, k),
        n_tokens,
    }
}

/// The iso attention mask `[B, 1, T, T]` (true = attend), with the local window on the
/// logical positions when `window` is given. Padding sees only itself.
pub fn iso_mask(seg: &Array, pos: &Array, window: Option<i32>) -> R<Array> {
    let (b, t) = (seg.dim(0), seg.dim(1));
    let sq = seg.reshape(&[b, 1, t, 1])?;
    let sk = seg.reshape(&[b, 1, 1, t])?;
    let state = sk
        .eq(Array::from_int(SEG_STATE))?
        .logical_and(&sq.ge(Array::from_int(SEG_STATE))?)?;
    let question = sk
        .eq(Array::from_int(SEG_QUESTION))?
        .logical_and(&sq.ge(Array::from_int(SEG_QUESTION))?)?;
    let own = sq
        .eq(&sk)?
        .logical_and(&sq.ge(Array::from_int(SEG_OPTION))?)?;
    let r = Array::from_iter(0..t, &[t]);
    let eye = r.reshape(&[t, 1])?.eq(r.reshape(&[1, t])?)?;
    let mut m = state.logical_or(&question)?.logical_or(&own)?;
    if let Some(w) = window {
        let pq = pos.reshape(&[b, 1, t, 1])?;
        let pk = pos.reshape(&[b, 1, 1, t])?;
        m = m.logical_and(&ops::abs(pq.subtract(&pk)?)?.le(Array::from_int(w))?)?;
    }
    m.logical_or(&eye)
}

/// cos / sin of rotate_half RoPE at explicit positions `[B, T]` → `[B, 1, T, hd / 2]`.
fn rope_tables(pos: &Array, theta: f32, like_bf16: bool) -> R<(Array, Array)> {
    let half = HEAD_DIM / 2;
    let inv: Vec<f32> = (0..half)
        .map(|i| theta.powf(-(2.0 * i as f32) / HEAD_DIM as f32))
        .collect();
    let (b, t) = (pos.dim(0), pos.dim(1));
    let ang = pos
        .as_dtype(Dtype::Float32)?
        .reshape(&[b, 1, t, 1])?
        .multiply(Array::from_slice(&inv, &[half]))?;
    let (c, s) = (ops::cos(&ang)?, ops::sin(&ang)?);
    if like_bf16 {
        Ok((c.as_dtype(Dtype::Bfloat16)?, s.as_dtype(Dtype::Bfloat16)?))
    } else {
        Ok((c, s))
    }
}

fn apply_rope(x: &Array, c: &Array, s: &Array) -> R<Array> {
    let parts = x.split_at_indices(&[HEAD_DIM / 2], -1)?;
    let (x1, x2) = (&parts[0], &parts[1]);
    ops::concatenate(
        &[
            x1.multiply(c)?.subtract(x2.multiply(s)?)?,
            x2.multiply(c)?.add(x1.multiply(s)?)?,
        ],
        -1,
    )
}

/// The ModernBERT encoder (s1 weights `encoder.*`) under the iso mask and explicit positions:
/// `[B, T]` → `[B, T, d]` after the final norm.
pub fn iso_encoder(
    p: &ParamMap,
    cfg: &ModelConfig,
    ids: &Array,
    seg: &Array,
    pos: &Array,
) -> R<Array> {
    let p = &*model::compute_params(p, cfg)?;
    let eps = cfg.norm_eps;
    let h = cfg.n_heads() as i32;
    let global = iso_mask(seg, pos, None)?;
    let local = iso_mask(seg, pos, Some((cfg.local_attention / 2) as i32))?;
    let bf16 = cfg.bf16;
    let (gc, gs) = rope_tables(pos, cfg.global_rope_theta, bf16)?;
    let (lc, ls) = rope_tables(pos, cfg.local_rope_theta, bf16)?;
    let emb = model::w(p, "encoder.embeddings.tok_embeddings.weight")?.take_axis(ids, 0)?;
    let mut x = layer_norm(
        &emb,
        model::w(p, "encoder.embeddings.norm.weight")?,
        None,
        eps,
    )?;
    for l in 0..cfg.num_hidden_layers {
        let pre = format!("encoder.layers.{l}");
        let y = if l == 0 {
            x.clone()
        } else {
            layer_norm(
                &x,
                model::w(p, &format!("{pre}.attn_norm.weight"))?,
                None,
                eps,
            )?
        };
        let qkv = model::linear(&y, model::w(p, &format!("{pre}.attn.Wqkv.weight"))?, None)?;
        let (q, k, v) = model::split_heads(&qkv, h)?;
        let (c, s, mask) = if cfg.is_global(l) {
            (&gc, &gs, &global)
        } else {
            (&lc, &ls, &local)
        };
        let q = apply_rope(&q, c, s)?;
        let k = apply_rope(&k, c, s)?;
        let a = model::merge_heads(&model::attend(&q, &k, &v, Some(mask))?)?;
        x = x.add(model::linear(
            &a,
            model::w(p, &format!("{pre}.attn.Wo.weight"))?,
            None,
        )?)?;
        let y = layer_norm(
            &x,
            model::w(p, &format!("{pre}.mlp_norm.weight"))?,
            None,
            eps,
        )?;
        let u = model::linear(&y, model::w(p, &format!("{pre}.mlp.Wi.weight"))?, None)?;
        let parts = u.split_at_indices(&[cfg.intermediate_size as i32], -1)?;
        let m = gelu_erf(&parts[0])?.multiply(&parts[1])?;
        x = x.add(model::linear(
            &m,
            model::w(p, &format!("{pre}.mlp.Wo.weight"))?,
            None,
        )?)?;
    }
    layer_norm(&x, model::w(p, "encoder.final_norm.weight")?, None, eps)
}

/// Names and shapes of the iso readout.
pub fn head_specs(d: i32) -> Vec<(&'static str, Vec<i32>, i32)> {
    vec![
        ("iso.score.0.weight", vec![d, 4 * d], 4 * d),
        ("iso.score.0.bias", vec![d], 4 * d),
        ("iso.score.1.weight", vec![1, d], d),
        ("iso.score.1.bias", vec![1], d),
    ]
}

/// A fresh readout (PyTorch `Linear` init, `U(±1/√fan_in)`).
pub fn init_head(cfg: &ModelConfig, seed: u64) -> anyhow::Result<ParamMap> {
    let mut key = mlx_rs::random::key(seed)?;
    let mut out = ParamMap::new();
    for (name, shape, fan_in) in head_specs(cfg.hidden_size as i32) {
        let (k, next) = mlx_rs::random::split(&key, 2)?;
        key = next;
        let a = 1.0 / (fan_in as f32).sqrt();
        out.insert(
            name.into(),
            mlx_rs::random::uniform::<f32, f32>(-a, a, &shape[..], &k)?,
        );
    }
    crate::params::eval(&out)?;
    Ok(out)
}

/// Masked option logits `[B, K]` of an iso batch.
#[allow(clippy::too_many_arguments)]
pub fn iso_logits(
    p: &ParamMap,
    cfg: &ModelConfig,
    ids: &Array,
    seg: &Array,
    pos: &Array,
    pool: &Array,
    qpool: &Array,
    opt_mask: &Array,
) -> R<Array> {
    let x = iso_encoder(p, cfg, ids, seg, pos)?;
    let x = x.as_dtype(Dtype::Float32)?;
    let h = ops::matmul(pool, &x)?; // [B, K, d]
    let q = ops::broadcast_to(ops::matmul(qpool, &x)?, h.shape())?;
    let z = ops::concatenate(
        &[
            h.clone(),
            q.clone(),
            h.multiply(&q)?,
            ops::abs(h.subtract(&q)?)?,
        ],
        -1,
    )?;
    let s = gelu_erf(&model::linear(
        &z,
        model::w(p, "iso.score.0.weight")?,
        Some(model::w(p, "iso.score.0.bias")?),
    )?)?;
    let s = model::linear(
        &s,
        model::w(p, "iso.score.1.weight")?,
        Some(model::w(p, "iso.score.1.bias")?),
    )?;
    let (b, k) = (pool.dim(0), pool.dim(1));
    model::masked_logits(&s.reshape(&[b, k])?, opt_mask)
}

/// [`iso_logits`] of a batch.
pub fn forward(p: &ParamMap, cfg: &ModelConfig, b: &IsoBatch) -> R<Array> {
    iso_logits(
        p,
        cfg,
        &b.ids,
        &b.seg,
        &b.pos,
        &b.pool,
        &b.qpool,
        &b.opt_mask,
    )
}

/// The guaranteed properties of the construction, checked exactly (≤ 1e-6) on random weights.
/// They are properties of the architecture, not results: they hold for any weights.
#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::host_f32;
    use crate::params::{self, Parts};
    use crate::s5::data::tokenize;
    use crate::s5::record::S5Record;
    use crate::sequence::tests::byte_tok;
    use crate::test_util;

    const TOL: f32 = 1e-6;

    fn cfg() -> ModelConfig {
        // 3 layers: global, local (window 2 on each side), global.
        let mut c = ModelConfig::tiny(300);
        c.num_hidden_layers = 3;
        c.rl_agent.max_len = 256;
        c.rl_agent.head_max_len = 96;
        c
    }

    fn weights(c: &ModelConfig, seed: u64) -> ParamMap {
        let mut p = params::init(
            c,
            Parts {
                decision: false,
                mlm: false,
            },
            seed,
        )
        .unwrap();
        p.extend(init_head(c, seed + 1).unwrap());
        p
    }

    fn rec(t: &str, ins: &str, kv: &[(&str, &str)], state: &str) -> S5Record {
        S5Record {
            src: "x".into(),
            t: t.into(),
            ins: ins.into(),
            opts_kv: kv
                .iter()
                .map(|(a, b)| (a.to_string(), b.to_string()))
                .collect(),
            state: state.into(),
            target: vec![1.0; kv.len()],
            split: "train".into(),
            uid: String::new(),
            lang: String::new(),
        }
    }

    fn items(recs: Vec<S5Record>) -> Vec<S5Item> {
        let (_d, tok) = byte_tok();
        tokenize(&tok, &SpecialTokens::default(), recs).unwrap()
    }

    /// Logits of one item under an option order.
    fn logits(
        p: &ParamMap,
        c: &ModelConfig,
        it: &S5Item,
        order: &[usize],
        pos: Positions,
    ) -> Vec<f32> {
        let d = Draw {
            order: order.to_vec(),
            names: None,
        };
        let r = iso_row(it, &d, 256, 96, pos, &SpecialTokens::default()).unwrap();
        host_f32(&forward(p, c, &iso_batch(&[r], &[])).unwrap()).unwrap()
    }

    fn close(a: &[f32], b: &[f32]) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (x - y).abs() <= TOL)
    }

    #[test]
    fn permuting_the_options_permutes_the_logits_exactly() {
        let _g = test_util::lock();
        mlx_rs::with_device(mlx_rs::Device::cpu(), || {
            let c = cfg();
            let p = weights(&c, 7);
            let it = &items(vec![rec(
                "choice",
                "which one?",
                &[
                    ("a", "the first and longest option text"),
                    ("b", "second"),
                    ("c", "a third one, of medium length"),
                    ("d", "4"),
                ],
                "a state long enough to reach past the local window of the small model",
            )])[0];
            for pos in [Positions::Shared, Positions::Reset] {
                let base = logits(&p, &c, it, &[0, 1, 2, 3], pos);
                for perm in [[3, 2, 1, 0], [1, 3, 0, 2], [2, 0, 3, 1]] {
                    let got = logits(&p, &c, it, &perm, pos);
                    let want: Vec<f32> = perm.iter().map(|&i| base[i]).collect();
                    assert!(close(&got, &want), "{pos:?} {perm:?}: {got:?} vs {want:?}");
                }
            }
        });
    }

    #[test]
    fn names_are_never_rendered() {
        let c = cfg();
        let a = items(vec![
            rec(
                "choice",
                "q",
                &[("yes", "approve it"), ("no", "reject it")],
                "s",
            ),
            rec(
                "choice",
                "q",
                &[("xg6a6", "approve it"), ("e97ce", "reject it")],
                "s",
            ),
        ]);
        let d = Draw {
            order: vec![0, 1],
            names: Some(vec!["zz".into(), "yy".into()]),
        };
        let sp = SpecialTokens::default();
        let r0 = iso_row(&a[0], &d, 256, 96, Positions::Shared, &sp).unwrap();
        let r1 = iso_row(&a[1], &d, 256, 96, Positions::Shared, &sp).unwrap();
        assert_eq!(r0, r1);
        let _ = c;
    }

    #[test]
    fn swapping_the_noul_definitions_gives_s_plus_n_one() {
        let _g = test_util::lock();
        mlx_rs::with_device(mlx_rs::Device::cpu(), || {
            let c = cfg();
            let p = weights(&c, 11);
            let it = items(vec![
                rec(
                    "noul",
                    "the alert is real",
                    &[("false", "benign"), ("true", "malicious")],
                    "logs",
                ),
                rec(
                    "noul",
                    "the alert is real",
                    &[("false", "malicious"), ("true", "benign")],
                    "logs",
                ),
            ]);
            for pos in [Positions::Shared, Positions::Reset] {
                let s = crate::metrics::softmax_t(&logits(&p, &c, &it[0], &[0, 1], pos), 1.0);
                let n = crate::metrics::softmax_t(&logits(&p, &c, &it[1], &[0, 1], pos), 1.0);
                assert!((s[1] + n[1] - 1.0).abs() < 1e-6, "{pos:?}: {s:?} {n:?}");
            }
        });
    }

    #[test]
    fn the_state_does_not_depend_on_the_question_or_the_options() {
        let _g = test_util::lock();
        mlx_rs::with_device(mlx_rs::Device::cpu(), || {
            let c = cfg();
            let p = weights(&c, 3);
            let state = "temperature 39.5, heart rate 120, the patient is confused";
            let it = items(vec![
                rec(
                    "choice",
                    "triage level?",
                    &[("a", "now"), ("b", "later")],
                    state,
                ),
                rec(
                    "score",
                    "how urgent is it, really?",
                    &[("0", "not"), ("1", "a bit"), ("2", "very")],
                    state,
                ),
            ]);
            let sp = SpecialTokens::default();
            let enc = |i: usize| {
                let d = Draw {
                    order: (0..it[i].k()).collect(),
                    names: None,
                };
                let r = iso_row(&it[i], &d, 256, 96, Positions::Shared, &sp).unwrap();
                let n_state = r.seg.iter().filter(|&&s| s == SEG_STATE).count();
                let b = iso_batch(&[r], &[]);
                let x = iso_encoder(&p, &c, &b.ids, &b.seg, &b.pos).unwrap();
                let d = c.hidden_size;
                host_f32(&x).unwrap()[..n_state * d].to_vec()
            };
            let (a, b) = (enc(0), enc(1));
            assert!(!a.is_empty());
            assert!(close(&a, &b), "state rows differ");
        });
    }

    #[test]
    fn local_window_uses_logical_positions() {
        let _g = test_util::lock();
        // Prefix of 6 tokens (state 0..3, question 3..6), then two options of 3 tokens.
        let seg = [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3];
        let pos = [0, 1, 2, 3, 4, 5, 6, 7, 8, 6, 7, 8];
        let m = iso_mask(
            &Array::from_slice(&seg, &[1, 12]),
            &Array::from_slice(&pos, &[1, 12]),
            Some(2),
        )
        .unwrap();
        m.eval().unwrap();
        let m = m.as_slice::<bool>();
        let row = |q: usize| (0..12).map(|k| m[q * 12 + k]).collect::<Vec<_>>();
        // The first token of each option sees the last two question tokens and itself, and
        // nothing of the other option: the same pattern at index 6 and at index 9.
        let r6 = row(6);
        let r9 = row(9);
        assert_eq!(
            r6,
            [false, false, false, false, true, true, true, true, true, false, false, false]
        );
        assert_eq!(
            r9,
            [false, false, false, false, true, true, false, false, false, true, true, true]
        );
        // The state never sees the question or the options.
        assert!(row(1)[3..].iter().all(|&v| !v));
    }
}
