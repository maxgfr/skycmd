//! Decision data: record loading and pre-tokenization, augmented batch
//! sampling with per-source weights, padded batches and batched prediction.

use std::{
    io::{BufRead, BufReader},
    path::{Path, PathBuf},
};

use anyhow::{bail, Context, Result};
use mlx_rs::Array;

use crate::{
    config::{AgentConfig, ModelConfig, PAD_ID},
    metrics::Pred,
    model::{self, DecisionBatch},
    params::ParamMap,
    rng::Rng,
    sequence::{build_from_pieces, piece_texts, qtype_id, DecisionRecord, Pieces, Tok, OPTION_MAX_TOKENS},
};

/// Score questions are ordinal (`level i`), their option order is never permuted
/// (as in `rl_common.encode_record`).
pub const SCORE: usize = 1;

/// One pre-tokenized decision question.
#[derive(Debug, Clone)]
pub struct Item {
    pub pieces: Pieces,
    pub qtype: usize,
    /// Normalized target over the options, in record order.
    pub target: Vec<f32>,
    pub src: String,
    /// Index into [`DecisionData::groups`].
    pub group: usize,
    /// Sequence length in identity order.
    pub len: usize,
}

impl Item {
    pub fn k(&self) -> usize {
        self.target.len()
    }
}

/// All decision items, split by `split` and grouped by input directory.
#[derive(Debug, Default)]
pub struct DecisionData {
    /// Group names (input directory / file stem), e.g. `decisions`, `games`.
    pub groups: Vec<String>,
    /// Training items per group.
    pub train: Vec<Vec<Item>>,
    pub val: Vec<Item>,
    pub test: Vec<Item>,
    /// Records dropped (bad shape or options that do not fit).
    pub skipped: usize,
}

/// Every `*.jsonl` file under `path` (recursively, sorted), or `path` itself.
pub fn list_jsonl(path: &Path) -> Result<Vec<PathBuf>> {
    if path.is_file() {
        return Ok(vec![path.to_path_buf()]);
    }
    if !path.is_dir() {
        bail!("{} does not exist", path.display());
    }
    let mut out = Vec::new();
    let mut stack = vec![path.to_path_buf()];
    while let Some(dir) = stack.pop() {
        for entry in std::fs::read_dir(&dir)? {
            let p = entry?.path();
            if p.is_dir() {
                stack.push(p);
            } else if p.extension().is_some_and(|e| e == "jsonl") {
                out.push(p);
            }
        }
    }
    out.sort();
    Ok(out)
}

/// Reads every JSONL record of `path`.
pub fn read_records(path: &Path) -> Result<Vec<DecisionRecord>> {
    let mut out = Vec::new();
    for file in list_jsonl(path)? {
        let f = std::fs::File::open(&file).with_context(|| format!("opening {}", file.display()))?;
        for (i, line) in BufReader::new(f).lines().enumerate() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }
            let rec: DecisionRecord = serde_json::from_str(&line)
                .with_context(|| format!("{}:{}", file.display(), i + 1))?;
            out.push(rec);
        }
    }
    Ok(out)
}

/// Group name of an input path.
pub fn group_name(path: &Path) -> String {
    let stem = if path.is_file() { path.file_stem() } else { path.file_name() };
    stem.map(|s| s.to_string_lossy().into_owned()).unwrap_or_else(|| "data".into())
}

/// Validates and pre-tokenizes the records of every input path.
pub fn load(paths: &[PathBuf], tok: &Tok, agent: &AgentConfig) -> Result<DecisionData> {
    let mut data = DecisionData::default();
    for (g, path) in paths.iter().enumerate() {
        data.groups.push(group_name(path));
        data.train.push(Vec::new());
        let recs = read_records(path)?;
        // Tokenize every text of the group in large parallel batches.
        let mut texts = Vec::new();
        let mut kept = Vec::new();
        for rec in recs {
            let ok = qtype_id(&rec.t).is_some()
                && !rec.opts.is_empty()
                && rec.target.len() == rec.opts.len()
                && rec.target.iter().all(|x| x.is_finite() && *x >= 0.0)
                && rec.target.iter().sum::<f32>() > 0.0;
            if !ok {
                data.skipped += 1;
                continue;
            }
            let (h, o, s) = piece_texts(&rec);
            texts.push(h);
            texts.extend(o);
            texts.push(s);
            kept.push(rec);
        }
        let mut ids = Vec::with_capacity(texts.len());
        for chunk in texts.chunks(50_000) {
            ids.extend(tok.encode_batch(chunk.to_vec())?);
        }
        let mut it = ids.into_iter();
        for rec in kept {
            let head = it.next().unwrap_or_default();
            let opts: Vec<Vec<u32>> = (0..rec.opts.len())
                .map(|_| {
                    let mut o = it.next().unwrap_or_default();
                    o.truncate(OPTION_MAX_TOKENS);
                    o
                })
                .collect();
            let state = it.next().unwrap_or_default();
            let pieces = Pieces { head, opts, state };
            let k = rec.opts.len();
            let identity: Vec<usize> = (0..k).collect();
            let (seq, markers) = build_from_pieces(&pieces, &identity, agent.max_len, agent.head_max_len);
            if markers.len() != k {
                // Options did not fit: skip rather than train on a truncated answer space.
                data.skipped += 1;
                continue;
            }
            let sum: f32 = rec.target.iter().sum();
            let item = Item {
                pieces,
                qtype: qtype_id(&rec.t).unwrap_or(0),
                target: rec.target.iter().map(|x| x / sum).collect(),
                src: rec.src,
                group: g,
                len: seq.len(),
            };
            match rec.split.as_str() {
                "val" => data.val.push(item),
                "test" => data.test.push(item),
                _ => data.train[g].push(item),
            }
        }
    }
    Ok(data)
}

/// One built sequence ready for batching.
#[derive(Debug, Clone)]
pub struct Row {
    pub ids: Vec<u32>,
    pub markers: Vec<usize>,
    pub qtype: usize,
    pub target: Vec<f32>,
}

/// Builds the row of `item` with option `order` (target permuted accordingly).
pub fn row(item: &Item, order: &[usize], agent: &AgentConfig) -> Row {
    let (ids, markers) = build_from_pieces(&item.pieces, order, agent.max_len, agent.head_max_len);
    Row { ids, markers, qtype: item.qtype, target: order.iter().map(|&i| item.target[i]).collect() }
}

/// Random option permutation (identity for score questions).
pub fn augment_order(item: &Item, rng: &mut Rng) -> Vec<usize> {
    let mut order: Vec<usize> = (0..item.k()).collect();
    if item.qtype != SCORE {
        rng.shuffle(&mut order);
    }
    order
}

/// Pads rows into a batch; the length is rounded up to a multiple of `pad_to`.
pub fn make_batch(rows: &[Row], pad_to: usize) -> DecisionBatch {
    let b = rows.len();
    let max_t = rows.iter().map(|r| r.ids.len()).max().unwrap_or(1);
    let t = max_t.div_ceil(pad_to.max(1)) * pad_to.max(1);
    let k = rows.iter().map(|r| r.markers.len()).max().unwrap_or(1).max(1);
    let mut ids = vec![PAD_ID as i32; b * t];
    let mut valid = vec![false; b * t];
    let mut mpos = vec![0i32; b * k];
    let mut mmask = vec![false; b * k];
    let mut target = vec![0f32; b * k];
    let mut n_tokens = 0;
    for (i, r) in rows.iter().enumerate() {
        for (j, &id) in r.ids.iter().enumerate() {
            ids[i * t + j] = id as i32;
            valid[i * t + j] = true;
        }
        n_tokens += r.ids.len();
        for (j, &m) in r.markers.iter().enumerate() {
            mpos[i * k + j] = m as i32;
            mmask[i * k + j] = true;
            target[i * k + j] = r.target[j];
        }
    }
    let shape = [b as i32, t as i32];
    let key_valid = if valid.iter().all(|&v| v) { None } else { Some(Array::from_slice(&valid, &shape)) };
    DecisionBatch {
        ids: Array::from_slice(&ids, &shape),
        key_valid,
        qtype: Array::from_iter(rows.iter().map(|r| r.qtype as i32), &[b as i32]),
        marker_pos: Array::from_slice(&mpos, &[b as i32, k as i32]),
        marker_mask: Array::from_slice(&mmask, &[b as i32, k as i32]),
        target: Array::from_slice(&target, &[b as i32, k as i32]),
        n_tokens,
    }
}

/// Samples an augmented training batch: the group is drawn by `weights`, the
/// item uniformly within the group.
pub fn sample_batch(data: &DecisionData, weights: &[f64], batch: usize, rng: &mut Rng, agent: &AgentConfig) -> DecisionBatch {
    let w: Vec<f64> = weights
        .iter()
        .zip(&data.train)
        .map(|(w, items)| if items.is_empty() { 0.0 } else { *w })
        .collect();
    let rows: Vec<Row> = (0..batch)
        .map(|_| {
            let g = rng.weighted(&w);
            let item = &data.train[g][rng.below(data.train[g].len())];
            let order = augment_order(item, rng);
            let r = row(item, &order, agent);
            if r.markers.len() == item.k() {
                r
            } else {
                row(item, &(0..item.k()).collect::<Vec<_>>(), agent)
            }
        })
        .collect();
    make_batch(&rows, 16)
}

/// Runs the model over `items` (identity option order) and returns raw logits.
/// Batches are built from length-sorted items under a padded-token budget.
pub fn predict(
    params: &ParamMap,
    cfg: &ModelConfig,
    data_groups: &[String],
    items: &[&Item],
    max_tokens: usize,
) -> Result<Vec<Pred>> {
    let mut order: Vec<usize> = (0..items.len()).collect();
    order.sort_by_key(|&i| items[i].len);
    let mut out: Vec<Option<Pred>> = vec![None; items.len()];
    let mut start = 0;
    while start < order.len() {
        let mut end = start + 1;
        while end < order.len() && items[order[end]].len.div_ceil(16) * 16 * (end - start + 1) <= max_tokens {
            end += 1;
        }
        let rows: Vec<Row> = order[start..end]
            .iter()
            .map(|&i| row(items[i], &(0..items[i].k()).collect::<Vec<_>>(), &cfg.rl_agent))
            .collect();
        let batch = make_batch(&rows, 16);
        let logits = model::decision_forward(params, cfg, &batch)?;
        let k = logits.dim(1) as usize;
        let flat = model::host_f32(&logits)?;
        for (r, &i) in order[start..end].iter().enumerate() {
            let it = items[i];
            out[i] = Some(Pred {
                logits: flat[r * k..r * k + it.k()].to_vec(),
                target: it.target.clone(),
                qtype: it.qtype,
                src: it.src.clone(),
                group: data_groups.get(it.group).cloned().unwrap_or_default(),
            });
        }
        start = end;
    }
    Ok(out.into_iter().map(|p| p.expect("every item predicted")).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sequence::tests::byte_tok;

    fn write_records(dir: &Path, name: &str, recs: &[DecisionRecord]) {
        std::fs::create_dir_all(dir).unwrap();
        let body: Vec<String> = recs.iter().map(|r| serde_json::to_string(r).unwrap()).collect();
        std::fs::write(dir.join(name), body.join("\n")).unwrap();
    }

    fn rec(t: &str, opts: &[&str], target: &[f32], split: &str) -> DecisionRecord {
        DecisionRecord {
            src: "unit".into(),
            t: t.into(),
            ins: "q".into(),
            opts: opts.iter().map(|s| s.to_string()).collect(),
            state: "st".into(),
            target: target.to_vec(),
            split: split.into(),
        }
    }

    #[test]
    fn load_split_and_augment() {
        let (_d, tok) = byte_tok();
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("games");
        write_records(
            &dir,
            "a.jsonl",
            &[
                rec("choice", &["a", "b", "c"], &[0.0, 2.0, 0.0], "train"),
                rec("score", &["level 0: x", "level 1: y"], &[1.0, 0.0], "val"),
                rec("bogus", &["a"], &[1.0], "train"),
                rec("noul", &["false: n", "true: y"], &[0.0, 1.0, 0.0], "train"),
            ],
        );
        let agent = ModelConfig::tiny(300).rl_agent;
        let data = load(&[dir], &tok, &agent).unwrap();
        assert_eq!(data.groups, vec!["games"]);
        assert_eq!(data.train[0].len(), 1);
        assert_eq!(data.val.len(), 1);
        assert_eq!(data.skipped, 2);
        let item = &data.train[0][0];
        assert_eq!(item.target, vec![0.0, 1.0, 0.0]);
        // Permuted order moves the target with the option.
        let r = row(item, &[1, 2, 0], &agent);
        assert_eq!(r.target, vec![1.0, 0.0, 0.0]);
        assert_eq!(r.ids[r.markers[0] + 2], 5 + b'b' as u32);
        // Score options are never shuffled.
        let mut rng = Rng::new(0);
        for _ in 0..10 {
            assert_eq!(augment_order(&data.val[0], &mut rng), vec![0, 1]);
        }
    }

    #[test]
    fn batch_padding_layout() {
        let _g = crate::test_util::lock();
        let rows = vec![
            Row { ids: vec![2, 9, 3, 4, 7, 4, 8, 3, 3], markers: vec![3, 5], qtype: 2, target: vec![0.0, 1.0] },
            Row { ids: vec![2, 9, 3, 4, 7, 4, 8, 4, 6, 3, 3], markers: vec![3, 5, 7], qtype: 0, target: vec![0.2, 0.3, 0.5] },
        ];
        let b = make_batch(&rows, 16);
        assert_eq!(b.ids.shape(), &[2, 16]);
        assert_eq!(b.marker_pos.shape(), &[2, 3]);
        assert_eq!(b.n_tokens, 20);
        let kv = b.key_valid.unwrap();
        kv.eval().unwrap();
        let kv = kv.as_slice::<bool>();
        assert!(kv[8] && !kv[9] && kv[16 + 10] && !kv[16 + 11]);
        b.marker_mask.eval().unwrap();
        assert_eq!(b.marker_mask.as_slice::<bool>(), &[true, true, false, true, true, true]);
    }
}
