//! `sky-train s5-train`: fine-tuning of an imported encoder (Ettin, mmBERT) into a decision
//! model, by steps (not time), so that every variant and seed sees the same number of updates.
//!
//! Recipe: AdamW everywhere (no Muon), the fresh head at `head_lr_mult` times the encoder rate,
//! linear warmup then linear decay, an exponential moving average of the weights (evaluated and
//! exported), loss = soft CE + Brier + RPS on `score` questions (+ symmetric KL on pairs, +
//! `coh` × the coherence losses of [`crate::s5::coh`]), no label smoothing.
//!
//! Variants: **A** is the Laya layout with the records' option names; **A′** the same layout
//! with the shared augmentation's random names and polarity swaps rendered (see
//! [`crate::s5::data`]); **B-i** / **B-ii** the iso layout of [`crate::s5::iso`] (definitions
//! only, options isolated), with the options at a shared offset (i) or reset to 0 (ii).
//!
//! **Sampling.** Training records fall in groups (`typed`: the typed-decisions train split;
//! `coherence`: the families of [`crate::s5::coh`]; `pairs`; `gen`; `corpus`: the rest), drawn
//! with `--group-weights`. A coherence family is always drawn whole (CH, every S_k and N_k),
//! whatever `coh`, so that every value of λ trains on the same data. Every variant draws the
//! same groups, items and augmentations from the same seed.
//!
//! A checkpoint `<out>/last` holds `model.safetensors` (the EMA weights, in the s1 names, so
//! that `sky-train export` turns variant A into the Laya / oscar layout), `raw.safetensors`
//! and `optimizer.safetensors` (to resume), `config.json`, `tokenizer.json`,
//! `rl_agent_config.json` and `s5.json` (the training settings). At the end, `<out>/val.json`
//! reports the EMA model on the whole val split: overall, per group, and the coherence metrics
//! of the val families (the selection criteria of the pre-registration).

use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{bail, Context, Result};
use mlx_rs::{transforms::eval, Array};
use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::config::ModelConfig;
use crate::metrics::{self, Pred};
use crate::model::{self, DecisionBatch};
use crate::optim::{warmup_linear, OptConfig, Optimizer};
use crate::params::{self, ParamMap, Parts};
use crate::rng::Rng;
use crate::s5::coh::{self, CohSpec};
use crate::s5::data::{
    self, draw, laya_batch, laya_row, link_pairs, native_tokens, Augment, Draw, LayaRow, S5Item,
};
use crate::s5::iso::{self, iso_batch, iso_row, IsoRow, Positions};
use crate::s5::loss::{self, LossWeights, Pairs};
use crate::s5::score::read_s5;
use crate::sequence::Tok;
use crate::session::{apply_update, sorted_names, CompiledGrad};
use crate::train::{rl_agent_json, JsonLog};

/// Model variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Variant {
    /// Laya layout, the records' own option names.
    A,
    /// Laya layout, random names and polarity swaps rendered.
    APrime,
    /// Iso layout, options at the shared offset `L` (sub-variant i).
    BShared,
    /// Iso layout, options reset to position 0 (sub-variant ii).
    BReset,
}

impl Variant {
    pub fn parse(s: &str) -> Result<Self> {
        Ok(match s {
            "a" => Variant::A,
            "a-prime" | "a'" => Variant::APrime,
            "b-i" | "b" => Variant::BShared,
            "b-ii" => Variant::BReset,
            other => bail!("unknown variant {other:?} (a, a-prime, b-i, b-ii)"),
        })
    }

    /// The option positions of an iso variant (`None` for the Laya layout).
    pub fn iso(self) -> Option<Positions> {
        match self {
            Variant::BShared => Some(Positions::Shared),
            Variant::BReset => Some(Positions::Reset),
            _ => None,
        }
    }

    /// Whether the augmentation's names are rendered.
    fn renders_names(self) -> bool {
        matches!(self, Variant::APrime)
    }
}

#[derive(Debug, Clone, clap::Args, Serialize, Deserialize)]
pub struct TrainArgs {
    /// Imported checkpoint (`sky-train import-hf`): model.safetensors, config.json, tokenizer.json.
    #[arg(long)]
    pub init: PathBuf,
    /// s5 records (JSONL files or directories, comma separated): split `train` and `val`.
    #[arg(long, value_delimiter = ',')]
    pub records: Vec<PathBuf>,
    #[arg(long)]
    pub out: PathBuf,
    /// a | a-prime | b-i | b-ii
    #[arg(long, default_value = "a")]
    pub variant: String,
    #[arg(long, default_value_t = 3000)]
    pub steps: u64,
    #[arg(long, default_value_t = 32)]
    pub batch: usize,
    /// Peak AdamW learning rate of the encoder.
    #[arg(long, default_value_t = 5e-5)]
    pub lr: f32,
    /// Learning-rate multiplier of the freshly initialized head.
    #[arg(long, default_value_t = 20.0)]
    pub head_lr_mult: f32,
    /// Warmup fraction of the steps.
    #[arg(long, default_value_t = 0.06)]
    pub warmup: f64,
    #[arg(long, default_value_t = 0.01)]
    pub weight_decay: f32,
    /// EMA decay of the weights (0 disables).
    #[arg(long, default_value_t = 0.999)]
    pub ema: f32,
    #[arg(long, default_value_t = 1.0)]
    pub brier: f32,
    #[arg(long, default_value_t = 0.25)]
    pub rps: f32,
    /// Weight of the symmetric KL between pair members.
    #[arg(long, default_value_t = 0.0)]
    pub kl: f32,
    /// Weight λ of the coherence losses (bet 2).
    #[arg(long, default_value_t = 0.0)]
    pub coh: f32,
    #[arg(long, default_value_t = 0.5)]
    pub p_names: f64,
    #[arg(long, default_value_t = 0.25)]
    pub p_polarity: f64,
    #[arg(long, default_value_t = 512)]
    pub max_len: usize,
    #[arg(long, default_value_t = 192)]
    pub head_max_len: usize,
    /// Languages of the training records (`en`: records without `lang`).
    #[arg(long, value_delimiter = ',', default_value = "en")]
    pub langs: Vec<String>,
    /// Sampling weights of the record groups.
    #[arg(
        long,
        default_value = "typed=0.25,corpus=0.4,gen=0.15,pairs=0.1,coherence=0.1"
    )]
    pub group_weights: String,
    #[arg(long, default_value_t = 0)]
    pub seed: u64,
    #[arg(long, default_value_t = 50)]
    pub log_every: u64,
    #[arg(long, default_value_t = 500)]
    pub eval_every: u64,
    /// Validation questions per intermediate evaluation (the final one uses every val record).
    #[arg(long, default_value_t = 2000)]
    pub val_max: usize,
    #[arg(long, default_value_t = 1000)]
    pub checkpoint_every: u64,
    /// Continue from <out>/last.
    #[arg(long)]
    pub resume: bool,
    #[arg(long)]
    pub bf16: bool,
}

/// Progress saved with a checkpoint.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct State {
    step: u64,
    rng: Rng,
    args: TrainArgs,
    elapsed_secs: f64,
}

/// Exponential moving average of the parameters.
pub fn ema_update(ema: &mut ParamMap, p: &ParamMap, decay: f32) -> Result<()> {
    let d = Array::from_f32(decay);
    let e = Array::from_f32(1.0 - decay);
    for (k, v) in ema.iter_mut() {
        *v = v.multiply(&d)?.add(p[k].multiply(&e)?)?;
    }
    eval(ema.values())?;
    Ok(())
}

/// Group of a training record.
pub fn group_of(it: &S5Item) -> &'static str {
    let src = it.rec.src.as_str();
    if src.starts_with("coh_") {
        "coherence"
    } else if it.rec.uid.contains('#') {
        "pairs"
    } else if src.starts_with("typed_decisions_train") {
        "typed"
    } else if src.starts_with("gen_") {
        "gen"
    } else {
        "corpus"
    }
}

fn parse_weights(spec: &str) -> Result<Vec<(String, f64)>> {
    spec.split(',')
        .filter(|p| !p.trim().is_empty())
        .map(|p| {
            let (k, v) = p
                .split_once('=')
                .with_context(|| format!("bad weight {p:?}"))?;
            Ok((k.trim().to_string(), v.trim().parse()?))
        })
        .collect()
}

/// Item indices per group, and the coherence families (CH first, then S_k / N_k by k).
struct Sampler {
    groups: Vec<(String, f64, Vec<usize>)>,
    families: Vec<Vec<usize>>,
}

impl Sampler {
    fn new(train: &[S5Item], weights: &[(String, f64)]) -> Self {
        let mut by: BTreeMap<&str, Vec<usize>> = BTreeMap::new();
        let mut fam: BTreeMap<String, Vec<(char, usize, usize)>> = BTreeMap::new();
        for (i, it) in train.iter().enumerate() {
            let g = group_of(it);
            if g == "coherence" {
                if let Some((gid, kind, k)) = coh::parse(&it.rec.uid) {
                    fam.entry(gid.to_string()).or_default().push((kind, k, i));
                }
            }
            by.entry(g).or_default().push(i);
        }
        let families: Vec<Vec<usize>> = fam
            .into_values()
            .filter(|f| f.iter().any(|x| x.0 == 'C') && f.iter().any(|x| x.0 == 'S'))
            .map(|mut f| {
                f.sort_by_key(|&(kind, k, _)| (kind != 'C', k, kind));
                f.into_iter().map(|x| x.2).collect()
            })
            .collect();
        let groups = weights
            .iter()
            .map(|(name, w)| {
                let idx = by.get(name.as_str()).cloned().unwrap_or_default();
                let w = if name == "coherence" {
                    if families.is_empty() {
                        0.0
                    } else {
                        *w
                    }
                } else if idx.is_empty() {
                    0.0
                } else {
                    *w
                };
                (name.clone(), w, idx)
            })
            .collect();
        Self { groups, families }
    }
}

/// One row of either layout.
enum AnyRow {
    Laya(LayaRow),
    Iso(IsoRow),
}

impl AnyRow {
    fn order(&self) -> &[usize] {
        match self {
            AnyRow::Laya(r) => &r.order,
            AnyRow::Iso(r) => &r.order,
        }
    }
}

/// The row of an item under a draw (`None` when its options do not fit).
fn build_row(
    item: &S5Item,
    d: &Draw,
    variant: Variant,
    tok: &Tok,
    cfg: &ModelConfig,
) -> Result<Option<AnyRow>> {
    if let Some(pos) = variant.iso() {
        return Ok(iso_row(
            item,
            d,
            cfg.rl_agent.max_len,
            cfg.rl_agent.head_max_len,
            pos,
            &cfg.special,
        )
        .map(AnyRow::Iso));
    }
    let opts = if variant.renders_names() {
        native_tokens(item, d, tok, &cfg.special)?
    } else {
        item.native.clone()
    };
    let r = laya_row(
        item,
        d,
        &opts,
        cfg.rl_agent.max_len,
        cfg.rl_agent.head_max_len,
        &cfg.special,
    );
    Ok((r.markers.len() == item.k()).then_some(AnyRow::Laya(r)))
}

/// Display position of the "true" option of a noul row (1 otherwise).
fn true_pos(item: &S5Item, row: &AnyRow) -> i32 {
    if item.rec.t == "noul" {
        row.order().iter().position(|&o| o == 1).unwrap_or(1) as i32
    } else {
        1
    }
}

/// Batch arrays in the order of the compiled function.
fn batch_arrays(
    rows: Vec<AnyRow>,
    pairs: &[(usize, usize)],
    spec: &CohSpec,
) -> (Vec<Array>, usize) {
    let n = rows.len();
    let mut out = if matches!(rows.first(), Some(AnyRow::Iso(_))) {
        let r: Vec<IsoRow> = rows
            .into_iter()
            .filter_map(|r| match r {
                AnyRow::Iso(x) => Some(x),
                AnyRow::Laya(_) => None,
            })
            .collect();
        let b = iso_batch(&r, pairs);
        let t = b.n_tokens;
        (
            vec![
                b.ids,
                b.seg,
                b.pos,
                b.pool,
                b.qpool,
                b.opt_mask,
                b.target,
                b.is_score,
                b.k,
                b.pairs.left,
                b.pairs.right,
                b.pairs.map,
                b.pairs.mask,
                b.pairs.weight,
            ],
            t,
        )
    } else {
        let r: Vec<LayaRow> = rows
            .into_iter()
            .filter_map(|r| match r {
                AnyRow::Laya(x) => Some(x),
                AnyRow::Iso(_) => None,
            })
            .collect();
        let b = laya_batch(&r, pairs);
        let t = b.n_tokens;
        (
            vec![
                b.ids,
                b.key_valid,
                b.qtype,
                b.marker_pos,
                b.marker_mask,
                b.target,
                b.is_score,
                b.k,
                b.pairs.left,
                b.pairs.right,
                b.pairs.map,
                b.pairs.mask,
                b.pairs.weight,
            ],
            t,
        )
    };
    out.0.extend(coh::arrays(spec, n));
    out
}

fn compiled(
    cfg: &ModelConfig,
    variant: Variant,
    names: Vec<std::rc::Rc<str>>,
    w: LossWeights,
    lambda: f32,
) -> CompiledGrad {
    let cfg = cfg.clone();
    let iso_layout = variant.iso().is_some();
    CompiledGrad::new(names, move |p, b| {
        // Layout-specific inputs, then (target, is_score, k), the 5 pair arrays, coherence.
        let (logits, t0) = if iso_layout {
            (
                iso::iso_logits(p, &cfg, &b[0], &b[1], &b[2], &b[3], &b[4], &b[5])?,
                6,
            )
        } else {
            let batch = DecisionBatch {
                ids: b[0].clone(),
                key_valid: Some(b[1].clone()),
                qtype: b[2].clone(),
                marker_pos: b[3].clone(),
                marker_mask: b[4].clone(),
                target: b[5].clone(),
                n_tokens: 0,
            };
            (model::decision_forward(p, &cfg, &batch)?, 5)
        };
        let pairs = Pairs {
            left: b[t0 + 3].clone(),
            right: b[t0 + 4].clone(),
            map: b[t0 + 5].clone(),
            mask: b[t0 + 6].clone(),
            weight: b[t0 + 7].clone(),
        };
        let base = loss::total(&logits, &b[t0], &b[t0 + 1], &b[t0 + 2], Some(&pairs), &w)?;
        if lambda == 0.0 {
            return Ok(base);
        }
        let c = coh::loss(&logits, &b[t0 + 8..t0 + 8 + coh::N_ARRAYS])?;
        base.add(c.multiply(Array::from_f32(lambda))?)
    })
}

/// The rows of a training batch: groups drawn by weight; a coherence family whole; an item
/// with a pair partner (KL on) with its partner. Every row has its own augmentation draw.
#[allow(clippy::too_many_arguments)]
fn sample(
    train: &[S5Item],
    sampler: &Sampler,
    a: &TrainArgs,
    variant: Variant,
    aug: &Augment,
    rng: &mut Rng,
    tok: &Tok,
    cfg: &ModelConfig,
    w: &LossWeights,
) -> Result<(Vec<Array>, usize)> {
    let mut rows: Vec<AnyRow> = Vec::with_capacity(2 * a.batch);
    let mut pairs = Vec::new();
    let mut spec = CohSpec::default();
    let weights: Vec<f64> = sampler.groups.iter().map(|g| g.1).collect();
    while rows.len() < a.batch {
        let gi = rng.weighted(&weights);
        let (name, _, idx) = &sampler.groups[gi];
        if name == "coherence" {
            let fam = &sampler.families[rng.below(sampler.families.len())];
            let start = rows.len();
            let mut built = Vec::with_capacity(fam.len());
            let mut ok = true;
            for &i in fam {
                let d = draw(&train[i], aug, rng);
                match build_row(&train[i], &d, variant, tok, cfg)? {
                    Some(r) => built.push((i, r)),
                    None => ok = false,
                }
            }
            if !ok {
                continue;
            }
            let mut ch: Option<(usize, Vec<usize>)> = None;
            let mut s_rows: BTreeMap<usize, usize> = BTreeMap::new();
            let mut n_rows: BTreeMap<usize, usize> = BTreeMap::new();
            for (j, (i, r)) in built.into_iter().enumerate() {
                let row_idx = start + j;
                spec.tpos.resize(row_idx, 1);
                spec.tpos.push(true_pos(&train[i], &r));
                match coh::parse(&train[i].rec.uid) {
                    Some((_, 'C', _)) => ch = Some((row_idx, r.order().to_vec())),
                    Some((_, 'S', k)) => {
                        s_rows.insert(k, row_idx);
                    }
                    Some((_, 'N', k)) => {
                        n_rows.insert(k, row_idx);
                    }
                    _ => {}
                }
                rows.push(r);
            }
            for (k, &s) in &s_rows {
                if let Some(&n) = n_rows.get(k) {
                    spec.neg.push((s, n));
                }
                if let Some((c, order)) = &ch {
                    if let Some(pos) = order.iter().position(|o| o == k) {
                        spec.xf.push((s, *c, pos));
                    }
                }
            }
            if ch.as_ref().is_some_and(|(_, o)| o.len() == s_rows.len()) {
                spec.part.push(s_rows.values().copied().collect());
            }
            continue;
        }
        let i = idx[rng.below(idx.len())];
        let d = draw(&train[i], aug, rng);
        let partner = train[i].pair.filter(|_| w.kl != 0.0);
        let dp = partner.map(|j| draw(&train[j], aug, rng));
        let Some(r) = build_row(&train[i], &d, variant, tok, cfg)? else {
            continue;
        };
        let at = rows.len();
        spec.tpos.resize(at, 1);
        spec.tpos.push(true_pos(&train[i], &r));
        rows.push(r);
        if let (Some(j), Some(dp)) = (partner, dp) {
            if let Some(r) = build_row(&train[j], &dp, variant, tok, cfg)? {
                spec.tpos.push(true_pos(&train[j], &r));
                rows.push(r);
                pairs.push((at, at + 1));
            }
        }
    }
    Ok(batch_arrays(rows, &pairs, &spec))
}

/// Logits of `items` (identity order) under an iso variant.
pub fn predict_iso(
    params: &ParamMap,
    cfg: &ModelConfig,
    positions: Positions,
    items: &[&S5Item],
    max_tokens: usize,
) -> Result<Vec<Pred>> {
    let rows: Vec<Option<IsoRow>> = items
        .iter()
        .map(|it| {
            let d = Draw {
                order: (0..it.k()).collect(),
                names: None,
            };
            iso_row(
                it,
                &d,
                cfg.rl_agent.max_len,
                cfg.rl_agent.head_max_len,
                positions,
                &cfg.special,
            )
        })
        .collect();
    let mut order: Vec<usize> = (0..items.len()).filter(|&i| rows[i].is_some()).collect();
    order.sort_by_key(|&i| rows[i].as_ref().map_or(0, |r| r.ids.len()));
    let mut out: Vec<Option<Pred>> = vec![None; items.len()];
    let mut start = 0;
    while start < order.len() {
        let len = |i: usize| rows[order[i]].as_ref().map_or(0, |r| r.ids.len());
        let mut end = start + 1;
        while end < order.len() && len(end).div_ceil(16) * 16 * (end - start + 1) <= max_tokens {
            end += 1;
        }
        let sel: Vec<IsoRow> = order[start..end]
            .iter()
            .filter_map(|&i| rows[i].clone())
            .collect();
        let logits = iso::forward(params, cfg, &iso_batch(&sel, &[]))?;
        let k = logits.dim(1) as usize;
        let flat = model::host_f32(&logits)?;
        for (r, &i) in order[start..end].iter().enumerate() {
            let it = items[i];
            out[i] = Some(Pred {
                logits: flat[r * k..r * k + it.k()].to_vec(),
                target: it.target.clone(),
                qtype: it.qtype,
                src: it.rec.src.clone(),
                group: it.rec.uid.clone(),
            });
        }
        start = end;
    }
    Ok(out.into_iter().flatten().collect())
}

fn decision_batch(b: &data::LayaBatch) -> DecisionBatch {
    DecisionBatch {
        ids: b.ids.clone(),
        key_valid: Some(b.key_valid.clone()),
        qtype: b.qtype.clone(),
        marker_pos: b.marker_pos.clone(),
        marker_mask: b.marker_mask.clone(),
        target: Array::from_f32(0.0),
        n_tokens: b.n_tokens,
    }
}

/// Logits of `items` (identity order, the records' names) under `params`.
pub fn predict(
    params: &ParamMap,
    cfg: &ModelConfig,
    items: &[&S5Item],
    max_tokens: usize,
) -> Result<Vec<Pred>> {
    let mut order: Vec<usize> = (0..items.len()).collect();
    let rows: Vec<LayaRow> = items
        .iter()
        .map(|it| {
            let d = Draw {
                order: (0..it.k()).collect(),
                names: None,
            };
            laya_row(
                it,
                &d,
                &it.native,
                cfg.rl_agent.max_len,
                cfg.rl_agent.head_max_len,
                &cfg.special,
            )
        })
        .collect();
    order.retain(|&i| rows[i].markers.len() == items[i].k());
    order.sort_by_key(|&i| rows[i].ids.len());
    let mut out: Vec<Option<Pred>> = vec![None; items.len()];
    let mut start = 0;
    while start < order.len() {
        let mut end = start + 1;
        while end < order.len()
            && rows[order[end]].ids.len().div_ceil(16) * 16 * (end - start + 1) <= max_tokens
        {
            end += 1;
        }
        let sel: Vec<LayaRow> = order[start..end].iter().map(|&i| rows[i].clone()).collect();
        let b = laya_batch(&sel, &[]);
        let logits = model::decision_forward(params, cfg, &decision_batch(&b))?;
        let k = logits.dim(1) as usize;
        let flat = model::host_f32(&logits)?;
        for (r, &i) in order[start..end].iter().enumerate() {
            let it = items[i];
            out[i] = Some(Pred {
                logits: flat[r * k..r * k + it.k()].to_vec(),
                target: it.target.clone(),
                qtype: it.qtype,
                src: it.rec.src.clone(),
                group: it.rec.uid.clone(),
            });
        }
        start = end;
    }
    Ok(out.into_iter().flatten().collect())
}

/// Accuracy, NLL, Brier and ECE (T = 1) per source and overall.
pub fn report(preds: &[Pred]) -> BTreeMap<String, metrics::Metrics> {
    crate::train::grouped_metrics(preds, |p| p.src.clone(), |_| 1.0)
}

fn save(
    dir: &Path,
    raw: &ParamMap,
    ema: &ParamMap,
    opt: &Optimizer,
    cfg: &ModelConfig,
    state: &State,
    tok_json: &[u8],
) -> Result<()> {
    let tmp = PathBuf::from(format!("{}.tmp", dir.display()));
    if tmp.exists() {
        std::fs::remove_dir_all(&tmp)?;
    }
    std::fs::create_dir_all(&tmp)?;
    params::save(&tmp.join("model.safetensors"), ema, None)?;
    params::save(&tmp.join("raw.safetensors"), raw, None)?;
    params::save(
        &tmp.join("optimizer.safetensors"),
        &opt.state_arrays(),
        None,
    )?;
    let mut st = serde_json::to_value(state)?;
    st["optimizer_step"] = opt.step.into();
    st["optimizer"] = serde_json::to_value(&opt.cfg)?;
    std::fs::write(tmp.join("s5.json"), serde_json::to_string_pretty(&st)?)?;
    std::fs::write(tmp.join("config.json"), serde_json::to_string_pretty(cfg)?)?;
    std::fs::write(tmp.join("tokenizer.json"), tok_json)?;
    let rl = rl_agent_json(
        cfg,
        json!({"training": {"steps": state.step, "variant": state.args.variant,
               "recipe": "s5: AdamW, warmup + linear decay, EMA, CE + Brier + RPS"}}),
    );
    std::fs::write(
        tmp.join("rl_agent_config.json"),
        serde_json::to_string_pretty(&rl)?,
    )?;
    if dir.exists() {
        std::fs::remove_dir_all(dir)?;
    }
    std::fs::rename(&tmp, dir).with_context(|| format!("renaming to {}", dir.display()))?;
    Ok(())
}

/// The final validation report: overall, per group, coherence of the val families.
fn val_report(preds: &[Pred], val: &[S5Item]) -> serde_json::Value {
    let by_uid: HashMap<&str, &S5Item> = val.iter().map(|it| (it.rec.uid.as_str(), it)).collect();
    let all = metrics::compute(preds.iter(), |_| 1.0);
    let mut groups: BTreeMap<&str, Vec<&Pred>> = BTreeMap::new();
    let mut coh_items: Vec<(&str, Vec<f64>)> = Vec::new();
    for p in preds {
        let Some(it) = by_uid.get(p.group.as_str()) else {
            continue;
        };
        let g = group_of(it);
        groups.entry(g).or_default().push(p);
        if g == "coherence" {
            coh_items.push((it.rec.uid.as_str(), metrics::softmax_t(&p.logits, 1.0)));
        }
    }
    let per: BTreeMap<&str, metrics::Metrics> = groups
        .iter()
        .map(|(g, v)| (*g, metrics::compute(v.iter().copied(), |_| 1.0)))
        .collect();
    let (mad, part, part_dev, xf, n_pairs) = coh::metrics(&coh_items);
    json!({
        "all": all,
        "groups": per,
        "coherence": {"mad_neg": mad, "partition_mean": part, "partition_abs_dev": part_dev,
                      "xf_abs": xf, "pairs": n_pairs},
    })
}

pub fn train(a: &TrainArgs) -> Result<()> {
    let variant = Variant::parse(&a.variant)?;
    let last = a.out.join("last");
    let resume = a.resume && last.join("s5.json").exists();
    let mut cfg: ModelConfig = serde_json::from_str(
        &std::fs::read_to_string(a.init.join("config.json")).context("init config.json")?,
    )?;
    cfg.rl_agent.max_len = a.max_len;
    cfg.rl_agent.head_max_len = a.head_max_len;
    cfg.bf16 = a.bf16;
    cfg.validate()?;
    let tok_path = a.init.join("tokenizer.json");
    let tok = Tok::load_with(&tok_path, &cfg.special)?;
    let tok_json = std::fs::read(&tok_path)?;

    let t_load = Instant::now();
    let mut recs = read_s5(&a.records)?;
    let n_all = recs.len();
    recs.retain(|r| {
        let l = if r.lang.is_empty() { "en" } else { &r.lang };
        a.langs.iter().any(|x| x == l)
    });
    let (mut train, mut val) = (Vec::new(), Vec::new());
    for r in recs {
        match r.split.as_str() {
            "train" => train.push(r),
            "val" => val.push(r),
            _ => {}
        }
    }
    let mut train = data::tokenize(&tok, &cfg.special, train)?;
    let val = data::tokenize(&tok, &cfg.special, val)?;
    link_pairs(&mut train);
    let weights = parse_weights(&a.group_weights)?;
    let sampler = Sampler::new(&train, &weights);
    println!(
        "{n_all} records; languages {:?}: {} train, {} val ({:.1}s)",
        a.langs,
        train.len(),
        val.len(),
        t_load.elapsed().as_secs_f64()
    );
    for (g, w, idx) in &sampler.groups {
        println!("  group {g}: {} items, weight {w}", idx.len());
    }
    println!("  coherence families: {}", sampler.families.len());
    if train.is_empty() {
        bail!("no training records");
    }

    let opt_cfg = OptConfig {
        adam_lr: a.lr,
        weight_decay: a.weight_decay,
        muon: false,
        head_lr_mult: a.head_lr_mult,
        beta2: 0.999,
        ..OptConfig::default()
    };
    let (mut raw, mut ema, mut opt, mut st) = if resume {
        let text = std::fs::read_to_string(last.join("s5.json"))?;
        let v: serde_json::Value = serde_json::from_str(&text)?;
        let st: State = serde_json::from_value(v.clone())?;
        let mut opt = Optimizer::new(serde_json::from_value(v["optimizer"].clone())?);
        opt.load_state(
            v["optimizer_step"].as_u64().unwrap_or(0),
            &params::load(&last.join("optimizer.safetensors"))?,
        )?;
        println!("resumed {} at step {}", last.display(), st.step);
        (
            params::load(&last.join("raw.safetensors"))?,
            params::load(&last.join("model.safetensors"))?,
            opt,
            st,
        )
    } else {
        let mut p = match variant.iso() {
            None => params::init(&cfg, Parts::DECISION, a.seed)?,
            Some(_) => {
                let mut p = params::init(
                    &cfg,
                    Parts {
                        decision: false,
                        mlm: false,
                    },
                    a.seed,
                )?;
                p.extend(iso::init_head(&cfg, a.seed ^ 0x150)?);
                p
            }
        };
        let src = params::load(&a.init.join("model.safetensors"))?;
        let n = params::load_matching(&mut p, &src)?;
        let n_enc = p.keys().filter(|k| k.starts_with("encoder.")).count();
        if n < n_enc {
            bail!(
                "{} provides {n} of the {n_enc} encoder tensors",
                a.init.display()
            );
        }
        let ema = params::deep_clone(&p);
        (
            p,
            ema,
            Optimizer::new(opt_cfg),
            State {
                step: 0,
                rng: Rng::new(a.seed),
                args: a.clone(),
                elapsed_secs: 0.0,
            },
        )
    };
    println!(
        "{} variant {:?}: {} parameters ({} encoder), {} steps of {}, lr {} (head x{}), λ {}, kl {}, {}",
        cfg.name,
        variant,
        params::count(&raw),
        params::count_prefix(&raw, "encoder."),
        a.steps,
        a.batch,
        a.lr,
        a.head_lr_mult,
        a.coh,
        a.kl,
        if a.bf16 { "bf16" } else { "f32" }
    );
    let aug = Augment {
        permute: true,
        p_names: a.p_names,
        p_polarity: a.p_polarity,
    };
    let w = LossWeights {
        brier: a.brier,
        rps: a.rps,
        kl: a.kl,
    };
    let val_refs: Vec<&S5Item> = val.iter().collect();
    let val_items: Vec<&S5Item> = {
        let mut idx: Vec<usize> = (0..val.len()).collect();
        if idx.len() > a.val_max {
            Rng::new(a.seed ^ 0xA11).shuffle(&mut idx);
            idx.truncate(a.val_max);
            idx.sort_unstable();
        }
        idx.into_iter().map(|i| &val[i]).collect()
    };
    std::fs::create_dir_all(&a.out)?;
    let mut log = JsonLog::open(&a.out.join("log.jsonl"))?;
    let mut f = compiled(&cfg, variant, sorted_names(&raw), w, a.coh);
    let run_predict = |p: &ParamMap, items: &[&S5Item]| -> Result<Vec<Pred>> {
        match variant.iso() {
            Some(pos) => predict_iso(p, &cfg, pos, items, 16_384),
            None => predict(p, &cfg, items, 16_384),
        }
    };
    let validate = |ema: &ParamMap, step: u64, log: &mut JsonLog| -> Result<()> {
        if val_items.is_empty() {
            return Ok(());
        }
        let preds = run_predict(ema, &val_items)?;
        let m = report(&preds);
        let all = &m["all"];
        println!(
            "  val step {step}: n {} acc {:.4} nll {:.4} brier {:.4} ece {:.4}",
            all.n, all.accuracy, all.nll, all.brier, all.ece
        );
        log.write(&json!({"step": step, "val": m}))?;
        Ok(())
    };

    let t0 = Instant::now();
    let base = st.elapsed_secs;
    let (mut wl, mut wt, mut wn, mut win) = (0.0f64, 0usize, 0u64, Instant::now());
    while st.step < a.steps {
        let lr_scale = warmup_linear(st.step as f64 / a.steps as f64, a.warmup) as f32;
        let (arrays, n_tokens) = sample(
            &train,
            &sampler,
            a,
            variant,
            &aug,
            &mut st.rng,
            &tok,
            &cfg,
            &w,
        )?;
        let (loss_arr, grads) = f.call(&raw, arrays)?;
        let (l, gn) = apply_update(&mut raw, &mut opt, loss_arr, &grads, lr_scale)?;
        if !l.is_finite() {
            bail!("non-finite loss at step {}", st.step);
        }
        if a.ema > 0.0 {
            ema_update(&mut ema, &raw, a.ema)?;
        } else {
            ema = raw.clone();
        }
        st.step += 1;
        st.elapsed_secs = base + t0.elapsed().as_secs_f64();
        wl += l as f64;
        wt += n_tokens;
        wn += 1;
        if st.step % a.log_every == 0 {
            let dt = win.elapsed().as_secs_f64();
            println!(
                "step {:>6} loss {:.4} gnorm {gn:.3} lr x{lr_scale:.3} {:.0} tok/s",
                st.step,
                wl / wn as f64,
                wt as f64 / dt
            );
            log.write(
                &json!({"step": st.step, "loss": wl / wn as f64, "grad_norm": gn,
                "lr_scale": lr_scale, "tok_s": wt as f64 / dt, "elapsed": st.elapsed_secs}),
            )?;
            (wl, wt, wn, win) = (0.0, 0, 0, Instant::now());
        }
        if st.step % a.eval_every == 0 && st.step < a.steps {
            validate(&ema, st.step, &mut log)?;
        }
        if st.step % a.checkpoint_every == 0 && st.step < a.steps {
            save(&last, &raw, &ema, &opt, &cfg, &st, &tok_json)?;
        }
    }
    save(&last, &raw, &ema, &opt, &cfg, &st, &tok_json)?;
    // The final report on every val record.
    let preds = run_predict(&ema, &val_refs)?;
    let mut rep = val_report(&preds, &val);
    rep["step"] = json!(st.step);
    rep["variant"] = json!(a.variant);
    rep["lr"] = json!(a.lr);
    rep["init"] = json!(a.init);
    rep["coh"] = json!(a.coh);
    rep["seed"] = json!(a.seed);
    rep["minutes"] = json!(st.elapsed_secs / 60.0);
    std::fs::write(
        a.out.join("val.json"),
        serde_json::to_string_pretty(&rep)? + "\n",
    )?;
    println!(
        "done: {} steps in {:.1} min; val (all {} records): acc {:.4} nll {:.4} ece {:.4}; \
         coherence mad_neg {:.4}; checkpoint {}",
        st.step,
        st.elapsed_secs / 60.0,
        rep["all"]["n"],
        rep["all"]["accuracy"].as_f64().unwrap_or(f64::NAN),
        rep["all"]["nll"].as_f64().unwrap_or(f64::NAN),
        rep["all"]["ece"].as_f64().unwrap_or(f64::NAN),
        rep["coherence"]["mad_neg"].as_f64().unwrap_or(f64::NAN),
        last.display()
    );
    Ok(())
}
