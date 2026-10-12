//! s2 post-training: temperature + early-exit calibration, the evaluation
//! report and the GGUF export.

use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
};

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use sky_gguf::Gguf;

use crate::{
    export::resolve_run,
    metrics::{self, softmax_t, Metrics, Pred},
    params::{self, ParamMap},
    rng::Rng,
    s2::{
        config::{bucket, S2Config},
        data::{self, S2Data, S2Item, StaticSeq, Stream},
        gguf::{config_from_gguf, params_from_gguf, to_gguf, WeightFormat},
        train::{exit_metrics, load_tok, predict, S2Pred},
    },
    sequence::{Tok, QTYPES},
};

/// Calibration of one weight mode.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct Calibration {
    pub temperature: BTreeMap<String, f32>,
    pub exit_threshold: [f32; 3],
}

/// Calibrations of both modes (`calibration.json` in the checkpoint).
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct Calibrations {
    pub f32: Option<Calibration>,
    pub ternary: Option<Calibration>,
}

/// A loaded s2 decision checkpoint.
pub struct S2Model {
    pub cfg: S2Config,
    pub params: ParamMap,
    pub tok: Tok,
    pub ckpt: PathBuf,
    pub root: PathBuf,
    pub calib: Calibrations,
}

impl S2Model {
    pub fn load(run: &Path) -> Result<Self> {
        let paths = resolve_run(run)?;
        let cfg: S2Config =
            serde_json::from_str(&std::fs::read_to_string(paths.ckpt.join("config.json"))?)
                .with_context(|| format!("{} is not an s2 checkpoint", paths.ckpt.display()))?;
        let params = params::load(&paths.ckpt.join("model.safetensors"))?;
        if !params.contains_key("cls.output.weight") {
            bail!("{} is not an s2 decision checkpoint", paths.ckpt.display());
        }
        let tok = load_tok(&paths.ckpt.join("tokenizer.json"), &cfg)?;
        let calib = match std::fs::read_to_string(paths.ckpt.join("calibration.json")) {
            Ok(t) => serde_json::from_str(&t)?,
            Err(_) => Calibrations::default(),
        };
        Ok(Self {
            cfg,
            params,
            tok,
            ckpt: paths.ckpt,
            root: paths.root,
            calib,
        })
    }

    /// The config with the calibration of a mode applied (if any).
    pub fn cfg_for(&self, ternary: bool) -> S2Config {
        let mut c = self.cfg.clone();
        if let Some(cal) = if ternary {
            &self.calib.ternary
        } else {
            &self.calib.f32
        } {
            c.temperature = cal.temperature.clone();
            c.exit_threshold = cal.exit_threshold;
        }
        c
    }
}

/// Early-exit decision of one prediction: `(exit, calibrated probs)`.
/// τ ≥ 1 disables early exit.
pub fn early_exit(p: &S2Pred, cfg: &S2Config, tau: f32) -> (usize, Vec<f64>) {
    let n = p.logits.len();
    for e in 0..n {
        let q = softmax_t(&p.logits[e], cfg.temperature_of(e, p.qtype, p.target.len()));
        let top = q.iter().cloned().fold(0.0, f64::max);
        if e + 1 == n || (tau < 1.0 && top >= tau as f64) {
            return (e, q);
        }
    }
    unreachable!("at least one exit")
}

fn argmax(v: &[f64]) -> usize {
    (0..v.len()).fold(0, |b, i| if v[i] > v[b] { i } else { b })
}

fn argmax_f32(v: &[f32]) -> usize {
    (0..v.len()).fold(0, |b, i| if v[i] > v[b] { i } else { b })
}

/// Fits temperatures per (exit, type) and (exit, type, bucket), then τ per
/// type: the smallest τ whose early-exit accuracy is within 0.5 points of
/// full depth.
pub fn fit_calibration(preds: &[S2Pred], cfg: &S2Config, min_bucket: usize) -> Calibration {
    let mut temperature = BTreeMap::new();
    let n_exits = cfg.n_exits();
    for e in 0..n_exits {
        for (q, t) in QTYPES.iter().enumerate() {
            let sel: Vec<Pred> = preds
                .iter()
                .filter(|p| p.qtype == q)
                .map(|p| p.at(e))
                .collect();
            if sel.is_empty() {
                continue;
            }
            let refs: Vec<&Pred> = sel.iter().collect();
            temperature.insert(format!("e{e}.{t}"), metrics::fit_temperature(&refs));
            let mut buckets: BTreeMap<&str, Vec<&Pred>> = BTreeMap::new();
            for p in &sel {
                buckets.entry(bucket(p.logits.len())).or_default().push(p);
            }
            for (b, v) in buckets {
                if v.len() >= min_bucket {
                    temperature.insert(format!("e{e}.{t}.{b}"), metrics::fit_temperature(&v));
                }
            }
        }
    }
    let mut c = cfg.clone();
    c.temperature = temperature.clone();
    let mut exit_threshold = [0.9f32; 3];
    for (q, tau_out) in exit_threshold.iter_mut().enumerate() {
        let sel: Vec<&S2Pred> = preds.iter().filter(|p| p.qtype == q).collect();
        if sel.is_empty() {
            continue;
        }
        let acc = |tau: f32| -> f64 {
            sel.iter()
                .map(|p| {
                    let (_, probs) = early_exit(p, &c, tau);
                    (argmax(&probs) == argmax_f32(&p.target)) as u8 as f64
                })
                .sum::<f64>()
                / sel.len() as f64
        };
        let full = acc(1.0);
        *tau_out = 1.0;
        for i in 50..100 {
            let tau = i as f32 / 100.0;
            if acc(tau) >= full - 0.005 {
                *tau_out = tau;
                break;
            }
        }
    }
    Calibration {
        temperature,
        exit_threshold,
    }
}

fn records_data(m: &S2Model, records: &[PathBuf]) -> Result<S2Data> {
    data::load_cfg(records, &m.tok, &m.cfg)
}

/// `s2-calibrate`: fits the f32 and ternary calibrations on `split` and writes
/// `calibration.json` (and the f32 values into `config.json`).
pub fn calibrate(
    run: &Path,
    records: &[PathBuf],
    split: &str,
    min_bucket: usize,
) -> Result<Calibrations> {
    let m = S2Model::load(run)?;
    let data = records_data(&m, records)?;
    let items = data.split(split);
    if items.is_empty() {
        bail!("no {split} items to calibrate on");
    }
    let mut out = Calibrations::default();
    for ternary in [false, true] {
        let preds = predict(
            &m.params,
            &m.cfg,
            &data.statics,
            &data.groups,
            &items,
            512,
            ternary,
        )?;
        let cal = fit_calibration(&preds, &m.cfg, min_bucket);
        let mut c = m.cfg.clone();
        c.temperature = cal.temperature.clone();
        c.exit_threshold = cal.exit_threshold;
        let last = m.cfg.n_exits() - 1;
        let before = metrics::compute(
            &preds.iter().map(|p| p.at(last)).collect::<Vec<_>>(),
            |_| 1.0,
        );
        let after = metrics::compute(&preds.iter().map(|p| p.at(last)).collect::<Vec<_>>(), |p| {
            c.temperature_of(last, p.qtype, p.logits.len())
        });
        let (layers, early) = early_stats(&preds, &c);
        println!(
            "{} calibration on {} {split} items: last-exit NLL {:.4} -> {:.4}, ECE {:.4} -> {:.4}; tau {:?}; mean layers {layers:.2}, early-exit rate {early:.3}",
            if ternary { "ternary" } else { "f32" },
            preds.len(),
            before.nll,
            after.nll,
            before.ece,
            after.ece,
            cal.exit_threshold
        );
        if ternary {
            out.ternary = Some(cal);
        } else {
            out.f32 = Some(cal);
        }
    }
    std::fs::write(
        m.ckpt.join("calibration.json"),
        serde_json::to_string_pretty(&out)?,
    )?;
    let mut cfg = m.cfg.clone();
    if let Some(c) = &out.f32 {
        cfg.temperature = c.temperature.clone();
        cfg.exit_threshold = c.exit_threshold;
    }
    std::fs::write(
        m.ckpt.join("config.json"),
        serde_json::to_string_pretty(&cfg)?,
    )?;
    Ok(out)
}

/// Mean layers used and early-exit rate under the config's τ.
pub fn early_stats(preds: &[S2Pred], cfg: &S2Config) -> (f64, f64) {
    let (mut layers, mut early) = (0.0, 0.0);
    for p in preds {
        let (e, _) = early_exit(p, cfg, cfg.exit_threshold[p.qtype]);
        layers += cfg.exits[e] as f64;
        early += (e + 1 < cfg.n_exits()) as u8 as f64;
    }
    let n = preds.len().max(1) as f64;
    (layers / n, early / n)
}

/// `s2-export`: writes the calibrated model as a `skycmd-s2` GGUF.
pub fn export(run: &Path, out: &Path, format: WeightFormat) -> Result<()> {
    let m = S2Model::load(run)?;
    let cfg = m.cfg_for(format == WeightFormat::Ternary);
    let tok_json = std::fs::read_to_string(m.ckpt.join("tokenizer.json"))?;
    let name = format!("skycmd-{}", cfg.name);
    let bytes = to_gguf(&m.params, &cfg, &tok_json, format, &name)?;
    if let Some(dir) = out.parent() {
        std::fs::create_dir_all(dir)?;
    }
    std::fs::write(out, &bytes)?;
    println!(
        "exported {} ({} parameters, {}) to {} ({:.1} KiB)",
        name,
        params::count(&m.params),
        format.name(),
        out.display(),
        bytes.len() as f64 / 1024.0
    );
    Ok(())
}

/// Reorders the options of a static stream (positions restart per option, so
/// the result is what the encoder would build from the permuted record).
pub fn permute_static(st: &StaticSeq, order: &[usize]) -> StaticSeq {
    let ends: Vec<usize> = (0..st.k())
        .map(|i| st.markers.get(i + 1).copied().unwrap_or(st.tokens.len()))
        .collect();
    let mut tokens = Stream::default();
    for t in 0..st.q_len {
        tokens.push_with(st.tokens.ids[t], st.tokens.values[t], st.tokens.has(t));
    }
    let mut seg = st.seg[..st.q_len].to_vec();
    let mut pos = st.pos[..st.q_len].to_vec();
    let mut markers = Vec::new();
    for (j, &i) in order.iter().enumerate() {
        markers.push(tokens.len());
        let (a, b) = (st.markers[i], ends[i]);
        for t in a..b {
            tokens.push_with(st.tokens.ids[t], st.tokens.values[t], st.tokens.has(t));
        }
        seg.extend(std::iter::repeat_n(j as i32 + 1, b - a));
        pos.extend_from_slice(&st.pos[a..b]);
    }
    StaticSeq {
        tokens,
        seg,
        pos,
        markers,
        q_len: st.q_len,
    }
}

/// Max probability difference (all exits) under a random option shuffle.
pub fn permutation_check(
    p: &ParamMap,
    cfg: &S2Config,
    data: &S2Data,
    items: &[&S2Item],
    ternary: bool,
    seed: u64,
) -> Result<(usize, f64)> {
    let mut rng = Rng::new(seed);
    let sel: Vec<&S2Item> = items
        .iter()
        .copied()
        .filter(|i| i.k() >= 2)
        .take(256)
        .collect();
    let mut statics: Vec<StaticSeq> = Vec::new();
    let mut permuted: Vec<S2Item> = Vec::new();
    let mut orig: Vec<S2Item> = Vec::new();
    let mut orders = Vec::new();
    for it in &sel {
        let mut order: Vec<usize> = (0..it.k()).collect();
        while order.iter().enumerate().all(|(i, &o)| i == o) {
            rng.shuffle(&mut order);
        }
        let st = &data.statics[it.stat];
        statics.push(st.clone());
        orig.push(S2Item {
            stat: statics.len() - 1,
            ..(*it).clone()
        });
        statics.push(permute_static(st, &order));
        permuted.push(S2Item {
            stat: statics.len() - 1,
            target: order.iter().map(|&i| it.target[i]).collect(),
            ..(*it).clone()
        });
        orders.push(order);
    }
    let g = data.groups.clone();
    let a = predict(
        p,
        cfg,
        &statics,
        &g,
        &orig.iter().collect::<Vec<_>>(),
        256,
        ternary,
    )?;
    let b = predict(
        p,
        cfg,
        &statics,
        &g,
        &permuted.iter().collect::<Vec<_>>(),
        256,
        ternary,
    )?;
    let mut worst = 0f64;
    for ((pa, pb), order) in a.iter().zip(&b).zip(&orders) {
        for e in 0..pa.logits.len() {
            let qa = softmax_t(&pa.logits[e], 1.0);
            let qb = softmax_t(&pb.logits[e], 1.0);
            for (j, &i) in order.iter().enumerate() {
                worst = worst.max((qb[j] - qa[i]).abs());
            }
        }
    }
    Ok((sel.len(), worst))
}

/// Words of an instruction, lower-cased.
fn words(s: &str) -> Vec<String> {
    s.to_lowercase()
        .split(|c: char| !c.is_alphanumeric() && c != '\'')
        .filter(|w| !w.is_empty())
        .map(String::from)
        .collect()
}

/// True when two instructions are a statement and its negation: they differ
/// only by inserting `not` / `no` / `never` (or a `n't` contraction), or by
/// swapping one antonym pair (`yes`/`no`, `true`/`false`, `can`/`cannot`, ...).
pub fn is_negation_pair(a: &str, b: &str) -> bool {
    let (wa, wb) = (words(a), words(b));
    let neg = ["not", "no", "never"];
    let (long, short) = if wa.len() > wb.len() {
        (&wa, &wb)
    } else {
        (&wb, &wa)
    };
    if long.len() == short.len() + 1 {
        let i = (0..short.len())
            .find(|&i| long[i] != short[i])
            .unwrap_or(short.len());
        return neg.contains(&long[i].as_str()) && long[i + 1..] == short[i..];
    }
    if wa.len() != wb.len() {
        return false;
    }
    let diff: Vec<usize> = (0..wa.len()).filter(|&i| wa[i] != wb[i]).collect();
    if diff.len() != 1 {
        return false;
    }
    let (x, y) = (wa[diff[0]].as_str(), wb[diff[0]].as_str());
    let pairs = [
        ("yes", "no"),
        ("true", "false"),
        ("can", "cannot"),
        ("can", "can't"),
        ("is", "isn't"),
        ("does", "doesn't"),
        ("will", "won't"),
        ("safe", "unsafe"),
        ("possible", "impossible"),
    ];
    pairs
        .iter()
        .any(|&(p, q)| (x == p && y == q) || (x == q && y == p))
}

/// Coherence on paired noul questions (same source and state, negated
/// statements): number of pairs and mean |p(X) + p(¬X) − 1|.
pub fn coherence(
    data: &S2Data,
    items: &[&S2Item],
    preds: &[S2Pred],
    cfg: &S2Config,
) -> (usize, f64) {
    let mut by_state: BTreeMap<(String, Vec<u32>), Vec<usize>> = BTreeMap::new();
    for (i, it) in items.iter().enumerate() {
        if it.qtype == 2 {
            by_state
                .entry((it.src.clone(), it.state.ids.clone()))
                .or_default()
                .push(i);
        }
    }
    let (mut n, mut sum) = (0usize, 0.0);
    for idx in by_state.values().filter(|v| v.len() >= 2 && v.len() <= 64) {
        for (x, &i) in idx.iter().enumerate() {
            for &j in &idx[x + 1..] {
                let (a, b) = (
                    &data.static_keys[items[i].stat].1,
                    &data.static_keys[items[j].stat].1,
                );
                if !is_negation_pair(a, b) {
                    continue;
                }
                let p = |k: usize| early_exit(&preds[k], cfg, 1.0).1[1];
                sum += (p(i) + p(j) - 1.0).abs();
                n += 1;
            }
        }
    }
    (n, if n > 0 { sum / n as f64 } else { f64::NAN })
}

fn metrics_table(title: &str, m: &BTreeMap<String, Metrics>) -> String {
    let mut s = format!("#### {title}\n\n| source | n | accuracy | NLL | Brier | ECE |\n|---|---:|---:|---:|---:|---:|\n");
    for (k, x) in m {
        s.push_str(&format!(
            "| {k} | {} | {:.4} | {:.4} | {:.4} | {:.4} |\n",
            x.n, x.accuracy, x.nll, x.brier, x.ece
        ));
    }
    s.push('\n');
    s
}

/// `s2-eval`: per-source / per-exit metrics of the latent f32 and ternary
/// models (and optionally exported GGUF files), early-exit statistics,
/// permutation invariance and coherence → `report.md`.
pub fn eval(
    run: &Path,
    records: &[PathBuf],
    split: &str,
    ggufs: &[PathBuf],
    report: Option<&Path>,
) -> Result<PathBuf> {
    eval_with(false, run, records, split, ggufs, report)
}

/// [`eval`]; with `gguf_only` the checkpoint itself (latent f32, fake-quant ternary) is not
/// evaluated, only the GGUF files (the run still provides the tokenizer and records layout).
pub fn eval_with(
    gguf_only: bool,
    run: &Path,
    records: &[PathBuf],
    split: &str,
    ggufs: &[PathBuf],
    report: Option<&Path>,
) -> Result<PathBuf> {
    let m = S2Model::load(run)?;
    let data = records_data(&m, records)?;
    let mut used = split.to_string();
    let mut items = data.split(split);
    if items.is_empty() && split == "test" {
        used = "val".into();
        items = data.split("val");
    }
    if items.is_empty() {
        bail!("no {split} items to evaluate");
    }
    let mut md = format!(
        "# Evaluation report: skycmd-{}\n\nCheckpoint: `{}`  \nSplit: `{used}` ({} questions)  \n\
         Exits after state blocks {:?}. Metrics use the calibrated temperatures of `calibration.json` per exit, \
         type and bucket; ECE uses 15 bins on the top probability. The early-exit answer stops at the first exit \
         whose calibrated top probability reaches τ of the question type.\n\n",
        m.cfg.name,
        m.ckpt.display(),
        items.len(),
        m.cfg.exits
    );
    let mut summary = String::from("## Summary\n\n| model | last-exit acc | early-exit acc | τ (choice/score/noul) | mean layers | early-exit rate | perm. max diff | coherence pairs | mean abs(p(X)+p(¬X)−1) |\n|---|---:|---:|---|---:|---:|---:|---:|---:|\n");
    let mut details = String::new();
    let mut variants: Vec<(String, S2Config, ParamMap, bool)> = vec![
        (
            "latent f32 (final QAT master weights)".into(),
            m.cfg_for(false),
            m.params.clone(),
            false,
        ),
        (
            "ternary (fake-quant)".into(),
            m.cfg_for(true),
            m.params.clone(),
            true,
        ),
    ];
    if gguf_only {
        variants.clear();
    }
    for path in ggufs {
        let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
        let g = Gguf::parse(&bytes)?;
        let cfg = config_from_gguf(&g)?;
        let p = params_from_gguf(&g, &cfg)?;
        variants.push((
            format!(
                "GGUF `{}`",
                path.file_name()
                    .map(|f| f.to_string_lossy().into_owned())
                    .unwrap_or_default()
            ),
            cfg,
            p,
            false,
        ));
    }
    for (label, cfg, p, qat) in &variants {
        let preds = predict(p, cfg, &data.statics, &data.groups, &items, 512, *qat)?;
        let last = cfg.n_exits() - 1;
        details.push_str(&format!("## {label}\n\n"));
        for e in 0..cfg.n_exits() {
            let by_src = exit_metrics(
                &preds,
                e,
                |p| p.src.clone(),
                |q| cfg.temperature_of(e, q.qtype, q.logits.len()),
            );
            details.push_str(&metrics_table(
                &format!("Exit {e} (after {} state blocks), per source", cfg.exits[e]),
                &by_src,
            ));
        }
        // Early exit per source.
        let mut ee: BTreeMap<String, (usize, f64, f64, f64)> = BTreeMap::new();
        let mut acc_ee = 0.0;
        for pr in &preds {
            let (e, probs) = early_exit(pr, cfg, cfg.exit_threshold[pr.qtype]);
            let hit = (argmax(&probs) == argmax_f32(&pr.target)) as u8 as f64;
            acc_ee += hit;
            for key in [pr.src.clone(), "all".to_string()] {
                let r = ee.entry(key).or_default();
                r.0 += 1;
                r.1 += hit;
                r.2 += cfg.exits[e] as f64;
                r.3 += (e < last) as u8 as f64;
            }
        }
        acc_ee /= preds.len() as f64;
        details.push_str("#### Early exit per source\n\n| source | n | accuracy | mean layers | early-exit rate |\n|---|---:|---:|---:|---:|\n");
        for (k, (n, h, l, x)) in &ee {
            let n = *n as f64;
            details.push_str(&format!(
                "| {k} | {n} | {:.4} | {:.2} | {:.3} |\n",
                h / n,
                l / n,
                x / n
            ));
        }
        details.push('\n');
        let last_acc = exit_metrics(
            &preds,
            last,
            |_| String::new(),
            |q| cfg.temperature_of(last, q.qtype, q.logits.len()),
        )["all"]
            .accuracy;
        let (layers, early) = early_stats(&preds, cfg);
        // On the CPU device: the check measures the function, not the Metal
        // kernels' batch-dependent rounding (up to ~1e-3 on the GPU).
        let (n_perm, perm) = mlx_rs::with_device(mlx_rs::Device::cpu(), || {
            permutation_check(p, cfg, &data, &items, *qat, 17)
        })?;
        let (n_pairs, coh) = coherence(&data, &items, &preds, cfg);
        println!(
            "{label}: last-exit acc {last_acc:.4}, early-exit acc {acc_ee:.4}, mean layers {layers:.2}, early-exit rate {early:.3}, perm max diff {perm:.2e} ({n_perm} items), coherence {coh:.4} ({n_pairs} pairs)"
        );
        summary.push_str(&format!(
            "| {label} | {last_acc:.4} | {acc_ee:.4} | {:.2}/{:.2}/{:.2} | {layers:.2} | {early:.3} | {perm:.1e} | {n_pairs} | {} |\n",
            cfg.exit_threshold[0],
            cfg.exit_threshold[1],
            cfg.exit_threshold[2],
            if n_pairs > 0 { format!("{coh:.4}") } else { "n/a".into() }
        ));
    }
    md.push_str(&summary);
    md.push_str("\nPermutation invariance: max |Δp| over all exits when the options of up to 256 questions are shuffled, computed on the CPU device (equivariant by construction, so this only measures float rounding; on the Metal GPU, batch-dependent kernel rounding alone reaches ~1e-3). Coherence: noul questions with the same source and state whose instructions are negations of each other (`not`/`no`/`never` inserted or an antonym swap); `n/a` when the split has no such pairs.\n\n");
    md.push_str(&details);
    let out = report
        .map(Path::to_path_buf)
        .unwrap_or_else(|| m.root.join("report.md"));
    std::fs::write(&out, md)?;
    println!("report written to {}", out.display());
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn negation_pairs() {
        assert!(is_negation_pair("Is the gap safe?", "Is the gap not safe?"));
        assert!(is_negation_pair("The answer is yes.", "The answer is no."));
        assert!(is_negation_pair(
            "The drone can land",
            "The drone cannot land"
        ));
        assert!(!is_negation_pair("Is the gap safe?", "Is the gap wide?"));
        assert!(!is_negation_pair("a b c", "a b c d"));
    }

    #[test]
    fn permuted_static_matches_reencoding() {
        let (_d, tok) = crate::s2::data::tests::s2_tok();
        let rec = |opts: &[&str]| crate::sequence::DecisionRecord {
            src: "x".into(),
            t: "choice".into(),
            ins: "pick 1".into(),
            opts: opts.iter().map(|s| s.to_string()).collect(),
            state: String::new(),
            target: vec![1.0; opts.len()],
            split: "val".into(),
        };
        let (a, _) = data::encode_record(&tok, &rec(&["aa 1", "b", "cccc"]), 96, 64).unwrap();
        let (b, _) = data::encode_record(&tok, &rec(&["cccc", "aa 1", "b"]), 96, 64).unwrap();
        assert_eq!(permute_static(&a, &[2, 0, 1]), b);
    }

    #[test]
    fn calibration_picks_tau() {
        // Exit 0 is confident and right on half of the items, exit 1 always right.
        let mut cfg = S2Config::tiny(300);
        cfg.temperature.clear();
        let mut preds = Vec::new();
        for i in 0..100 {
            let good0 = i % 2 == 0;
            preds.push(S2Pred {
                logits: vec![
                    if good0 {
                        vec![4.0, 0.0]
                    } else {
                        vec![0.0, 0.3]
                    },
                    vec![3.0, 0.0],
                ],
                target: vec![1.0, 0.0],
                qtype: 2,
                src: "s".into(),
                group: "g".into(),
            });
        }
        let cal = fit_calibration(&preds, &cfg, 10);
        assert!(
            cal.temperature.contains_key("e0.noul") && cal.temperature.contains_key("e1.noul.2")
        );
        // Exit 0 is right whenever it is confident: τ is below 1 and keeps full accuracy.
        assert!(cal.exit_threshold[2] < 1.0);
        cfg.temperature = cal.temperature.clone();
        cfg.exit_threshold = cal.exit_threshold;
        let (layers, early) = early_stats(&preds, &cfg);
        assert!(early > 0.3 && layers < 2.0, "{layers} {early}");
        assert_eq!(cal.exit_threshold[0], 0.9, "no choice items: default");
    }
}
