//! `sky-train score`: soft labels of a Laya-format teacher (laya-typed-decisions,
//! laya-multilingual, oscar-1-*) on s5 records, computed with the s1 forward pass (mlx).
//!
//! Each record is rendered natively (`id: definition`) or by definition only, scored under
//! several option orders (identity, reversed, then seeded shuffles; `score` questions keep
//! their order) and the log-softmax is averaged back in record order: the teachers' heads are
//! not permutation-invariant, so one order would bias the labels. The output record is the input
//! record plus `teacher`, `render`, `tlogits` (the mean log-probabilities, T = 1) and `tflip`
//! (whether the argmax changed across orders).

use std::collections::BTreeMap;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{bail, Context, Result};
use serde_json::{json, Value};

use crate::config::{AgentConfig, ModelConfig};
use crate::decision::{self, Item, Row};
use crate::model;
use crate::params;
use crate::rng::Rng;
use crate::s5::hf::{read_json_or_null, rope_thetas, special_tokens};
use crate::s5::record::{Render, S5Record};
use crate::sequence::{piece_texts_with, qtype_id, Pieces, Tok, OPTION_MAX_TOKENS};

#[derive(Debug, Clone, clap::Args)]
pub struct ScoreArgs {
    /// Laya-format teacher (encoder/config.json, rl_agent_config.json, tokenizer/, model.safetensors).
    #[arg(long)]
    pub teacher: PathBuf,
    /// s5 records: JSONL files or directories (comma separated).
    #[arg(long, value_delimiter = ',')]
    pub records: Vec<PathBuf>,
    #[arg(long)]
    pub out: PathBuf,
    /// Option rendering: native (`id: definition`) or definition.
    #[arg(long, default_value = "native")]
    pub render: String,
    /// Option orders averaged per question (score questions keep their order).
    #[arg(long, default_value_t = 2)]
    pub perms: usize,
    /// Only records of this split.
    #[arg(long)]
    pub split: Option<String>,
    /// At most this many records (the first ones).
    #[arg(long)]
    pub max: Option<usize>,
    /// Padded tokens per forward batch.
    #[arg(long, default_value_t = 16_384)]
    pub batch_tokens: usize,
    /// Run the matmuls in bfloat16.
    #[arg(long)]
    pub bf16: bool,
    /// Local RoPE theta when encoder/config.json has none (HF ModernBERT default: 10000).
    #[arg(long, default_value_t = 10_000.0)]
    pub local_theta: f32,
    /// Name of the teacher in the output (default: the directory name).
    #[arg(long)]
    pub name: Option<String>,
}

/// The s1 configuration of a Laya-format directory, with the special tokens of its tokenizer.
pub fn laya_config(dir: &Path, local_theta: f32) -> Result<ModelConfig> {
    let enc = read_json_or_null(&dir.join("encoder/config.json"))?;
    let rl = read_json_or_null(&dir.join("rl_agent_config.json"))?;
    if enc.is_null() || rl.is_null() {
        bail!("{} is not a Laya-format directory", dir.display());
    }
    let tok_json = read_json_or_null(&dir.join("tokenizer/tokenizer.json"))?;
    let tok_cfg = read_json_or_null(&dir.join("tokenizer/tokenizer_config.json"))?;
    let special = special_tokens(&tok_json, &tok_cfg)?;
    let num = |v: &Value, k: &str| v[k].as_f64();
    let req = |k: &str| num(&enc, k).with_context(|| format!("encoder/config.json: no {k}"));
    let cfg = ModelConfig {
        name: dir
            .file_name()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_default(),
        vocab_size: req("vocab_size")? as usize,
        hidden_size: req("hidden_size")? as usize,
        num_hidden_layers: req("num_hidden_layers")? as usize,
        num_attention_heads: req("num_attention_heads")? as usize,
        intermediate_size: req("intermediate_size")? as usize,
        global_attn_every_n_layers: num(&enc, "global_attn_every_n_layers").unwrap_or(3.0) as usize,
        local_attention: num(&enc, "local_attention").unwrap_or(128.0) as usize,
        global_rope_theta: rope_thetas(&enc, local_theta).0,
        local_rope_theta: rope_thetas(&enc, local_theta).1,
        norm_eps: num(&enc, "norm_eps").unwrap_or(1e-5) as f32,
        rl_agent: AgentConfig {
            head_layers: num(&rl, "head_layers").unwrap_or(2.0) as usize,
            max_len: num(&rl, "max_len").unwrap_or(512.0) as usize,
            head_max_len: num(&rl, "head_max_len").unwrap_or(256.0) as usize,
            temperature: [1.0; 3],
            temperature_by_options: BTreeMap::new(),
        },
        extra: Default::default(),
        bf16: false,
        special,
    };
    cfg.validate()?;
    Ok(cfg)
}

/// Every s5 record under `paths` (JSONL files or directories), in file order.
pub fn read_s5(paths: &[PathBuf]) -> Result<Vec<S5Record>> {
    let mut out = Vec::new();
    for path in paths {
        for file in decision::list_jsonl(path)? {
            let f = std::fs::File::open(&file)?;
            for (i, line) in BufReader::new(f).lines().enumerate() {
                let line = line?;
                if line.trim().is_empty() {
                    continue;
                }
                let v: Value = serde_json::from_str(&line)
                    .with_context(|| format!("{}:{}", file.display(), i + 1))?;
                // s1 records (rendered `opts`) are split back into (id, definition).
                let r = if v.get("opts_kv").is_some() {
                    serde_json::from_value(v)
                } else {
                    serde_json::from_value::<crate::sequence::DecisionRecord>(v).map(|d| {
                        let uid = format!("{}:{}", file.display(), i + 1);
                        S5Record::from_decision(&d, &uid, "")
                    })
                }
                .with_context(|| format!("{}:{}", file.display(), i + 1))?;
                out.push(r);
            }
        }
    }
    Ok(out)
}

/// Option orders of a question: identity, reversed, then shuffles seeded by the record.
pub fn orders(k: usize, qtype: usize, n: usize, seed: u64) -> Vec<Vec<usize>> {
    let id: Vec<usize> = (0..k).collect();
    if qtype == decision::SCORE || k < 2 || n <= 1 {
        return vec![id];
    }
    let mut out = vec![id.clone(), id.iter().rev().copied().collect()];
    let mut rng = Rng::new(seed);
    while out.len() < n {
        let mut o = id.clone();
        rng.shuffle(&mut o);
        out.push(o);
    }
    out.truncate(n);
    out
}

/// Mean over rows of log-softmax, each row mapped back to record order by its `order`.
pub fn mean_logprobs(logits: &[(Vec<usize>, Vec<f32>)], k: usize) -> Vec<f32> {
    let mut acc = vec![0f64; k];
    for (order, l) in logits {
        let m = l.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64;
        let lse = m + l.iter().map(|&x| (x as f64 - m).exp()).sum::<f64>().ln();
        for (pos, &opt) in order.iter().enumerate() {
            acc[opt] += l[pos] as f64 - lse;
        }
    }
    acc.iter()
        .map(|&x| (x / logits.len().max(1) as f64) as f32)
        .collect()
}

fn argmax(v: &[f32]) -> usize {
    (0..v.len())
        .max_by(|&a, &b| v[a].total_cmp(&v[b]))
        .unwrap_or(0)
}

pub fn score(a: &ScoreArgs) -> Result<()> {
    let render = Render::parse(&a.render)?;
    let mut cfg = laya_config(&a.teacher, a.local_theta)?;
    cfg.bf16 = a.bf16;
    let name = a.name.clone().unwrap_or_else(|| cfg.name.clone());
    let tok = Tok::load_with(&a.teacher.join("tokenizer/tokenizer.json"), &cfg.special)?;
    let params = params::load(&a.teacher.join("model.safetensors"))?;
    println!(
        "teacher {name}: {} parameters, d {} x {} layers, specials {:?}",
        params::count(&params),
        cfg.hidden_size,
        cfg.num_hidden_layers,
        cfg.special.pairs()
    );
    let mut recs = read_s5(&a.records)?;
    if let Some(s) = &a.split {
        recs.retain(|r| &r.split == s);
    }
    if let Some(m) = a.max {
        recs.truncate(m);
    }
    for r in &recs {
        r.validate()
            .with_context(|| format!("record {} ({})", r.uid, r.src))?;
    }
    // Tokenize every piece once.
    let mut texts = Vec::new();
    for r in &recs {
        let (h, o, s) = piece_texts_with(&r.to_decision(render), &cfg.special.mask_text);
        texts.push(h);
        texts.extend(o);
        texts.push(s);
    }
    let mut ids = Vec::with_capacity(texts.len());
    for chunk in texts.chunks(50_000) {
        ids.extend(tok.encode_batch(chunk.to_vec())?);
    }
    let mut it = ids.into_iter();
    let agent = cfg.rl_agent.clone();
    // One row per (record, order).
    let mut rows: Vec<(usize, Vec<usize>, Row)> = Vec::new();
    let mut dropped = 0;
    for (ri, r) in recs.iter().enumerate() {
        let head = it.next().unwrap_or_default();
        let opts: Vec<Vec<u32>> = (0..r.opts_kv.len())
            .map(|_| {
                let mut o = it.next().unwrap_or_default();
                o.truncate(OPTION_MAX_TOKENS);
                o
            })
            .collect();
        let state = it.next().unwrap_or_default();
        let qtype = qtype_id(&r.t).unwrap_or(0);
        let k = r.opts_kv.len();
        let item = Item {
            pieces: Pieces { head, opts, state },
            qtype,
            target: r.target.clone(),
            src: r.src.clone(),
            group: 0,
            len: 0,
        };
        let seed = crate::rng::hash_str(&format!("{}/{ri}", r.uid));
        let mut ok = true;
        let mut mine = Vec::new();
        for o in orders(k, qtype, a.perms, seed) {
            let row = decision::row(&item, &o, &agent, &cfg.special);
            if row.markers.len() != k {
                ok = false;
                break;
            }
            mine.push((ri, o, row));
        }
        if ok {
            rows.extend(mine);
        } else {
            dropped += 1;
        }
    }
    println!(
        "{} records ({dropped} dropped: options do not fit), {} rows, render {:?}",
        recs.len(),
        rows.len(),
        render
    );
    // Length-sorted batches under a padded-token budget.
    let mut order: Vec<usize> = (0..rows.len()).collect();
    order.sort_by_key(|&i| rows[i].2.ids.len());
    let mut results: Vec<Vec<(Vec<usize>, Vec<f32>)>> = vec![Vec::new(); recs.len()];
    let t0 = Instant::now();
    let mut start = 0;
    let mut done = 0;
    while start < order.len() {
        let mut end = start + 1;
        while end < order.len()
            && rows[order[end]].2.ids.len().div_ceil(16) * 16 * (end - start + 1) <= a.batch_tokens
        {
            end += 1;
        }
        let batch_rows: Vec<Row> = order[start..end]
            .iter()
            .map(|&i| rows[i].2.clone())
            .collect();
        let batch = decision::make_batch(&batch_rows, 16);
        let logits = model::decision_forward(&params, &cfg, &batch)?;
        let kmax = logits.dim(1) as usize;
        let flat = model::host_f32(&logits)?;
        for (b, &i) in order[start..end].iter().enumerate() {
            let (ri, o, _) = &rows[i];
            let k = o.len();
            results[*ri].push((o.clone(), flat[b * kmax..b * kmax + k].to_vec()));
        }
        done += end - start;
        if done % 2048 < end - start {
            let rate = done as f64 / t0.elapsed().as_secs_f64();
            eprintln!("  {done}/{} rows, {rate:.1} rows/s", rows.len());
        }
        start = end;
    }
    if let Some(dir) = a.out.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let mut f = std::io::BufWriter::new(std::fs::File::create(&a.out)?);
    let (mut n, mut hits, mut flips) = (0usize, 0usize, 0usize);
    for (r, res) in recs.iter().zip(&results) {
        if res.is_empty() {
            continue;
        }
        let k = r.opts_kv.len();
        let lp = mean_logprobs(res, k);
        let flip = res
            .iter()
            .map(|(o, l)| o[argmax(l)])
            .collect::<std::collections::BTreeSet<_>>()
            .len()
            > 1;
        n += 1;
        hits += usize::from(argmax(&lp) == argmax(&r.target));
        flips += usize::from(flip);
        let mut v = serde_json::to_value(r)?;
        v["teacher"] = json!(name);
        v["render"] = json!(a.render);
        v["tlogits"] = json!(lp);
        v["tflip"] = json!(flip);
        writeln!(f, "{v}")?;
    }
    f.flush()?;
    println!(
        "wrote {n} records to {} in {:.1}s: argmax agrees with the target on {:.4}, \
         order flips {:.4}",
        a.out.display(),
        t0.elapsed().as_secs_f64(),
        hits as f64 / n.max(1) as f64,
        flips as f64 / n.max(1) as f64
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn orders_keep_score_and_cover_reversal() {
        assert_eq!(orders(3, 1, 4, 7), vec![vec![0, 1, 2]]);
        let o = orders(4, 0, 3, 7);
        assert_eq!(o[0], vec![0, 1, 2, 3]);
        assert_eq!(o[1], vec![3, 2, 1, 0]);
        assert_eq!(o.len(), 3);
        let mut s = o[2].clone();
        s.sort_unstable();
        assert_eq!(s, vec![0, 1, 2, 3]);
        assert_eq!(orders(2, 2, 1, 0), vec![vec![0, 1]]);
    }

    #[test]
    fn logprobs_are_mapped_back_to_record_order() {
        // Same distribution under two orders: the mean equals it, in record order.
        let lp = mean_logprobs(
            &[
                (vec![0, 1, 2], vec![2.0, 0.0, -1.0]),
                (vec![2, 1, 0], vec![-1.0, 0.0, 2.0]),
            ],
            3,
        );
        let p: Vec<f32> = lp.iter().map(|x| x.exp()).collect();
        assert!((p.iter().sum::<f32>() - 1.0).abs() < 1e-5);
        assert!(p[0] > p[1] && p[1] > p[2]);
        let single = mean_logprobs(&[(vec![0, 1, 2], vec![2.0, 0.0, -1.0])], 3);
        for (a, b) in lp.iter().zip(&single) {
            assert!((a - b).abs() < 1e-6);
        }
    }
}
