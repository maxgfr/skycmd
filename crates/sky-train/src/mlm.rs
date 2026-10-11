//! MLM pretraining data: tokenized corpus (cached), packing of records into
//! fixed-length rows, and 30% masking with the 80/10/10 rule.

use std::{
    io::{BufRead, BufReader, Read, Write},
    path::{Path, PathBuf},
};

use anyhow::{bail, Context, Result};
use mlx_rs::Array;
use serde::{Deserialize, Serialize};

use crate::{
    config::{CLS_ID, MASK_ID, N_SPECIAL, PAD_ID, SEP_ID},
    decision::list_jsonl,
    model::MlmBatch,
    rng::Rng,
    sequence::Tok,
};

#[derive(Deserialize)]
struct MlmRecord {
    text: String,
}

/// The tokenized corpus: every record followed by `[SEP]`, concatenated.
#[derive(Debug, Clone)]
pub struct Corpus {
    pub tokens: Vec<u32>,
    pub n_records: usize,
}

#[derive(Debug, Serialize, Deserialize, PartialEq)]
struct CacheKey {
    files: Vec<(String, u64)>,
    tokenizer_bytes: u64,
    max_tokens: usize,
}

fn cache_key(files: &[PathBuf], tokenizer: &Path, max_tokens: usize) -> Result<CacheKey> {
    let files = files
        .iter()
        .map(|f| Ok((f.display().to_string(), std::fs::metadata(f)?.len())))
        .collect::<Result<_>>()?;
    Ok(CacheKey {
        files,
        tokenizer_bytes: std::fs::metadata(tokenizer)?.len(),
        max_tokens,
    })
}

impl Corpus {
    /// Tokenizes every MLM record under `text` (JSONL `{"src","text"}`), up to
    /// `max_tokens` tokens. The result is cached in `cache_dir`.
    pub fn load(
        text: &Path,
        tok: &Tok,
        tokenizer_path: &Path,
        cache_dir: &Path,
        max_tokens: usize,
    ) -> Result<Self> {
        let files = list_jsonl(text)?;
        if files.is_empty() {
            bail!("no *.jsonl under {}", text.display());
        }
        let key = cache_key(&files, tokenizer_path, max_tokens)?;
        let bin = cache_dir.join("corpus.u32");
        let meta = cache_dir.join("corpus.json");
        if let (Ok(m), true) = (std::fs::read_to_string(&meta), bin.exists()) {
            let saved: serde_json::Value = serde_json::from_str(&m)?;
            if serde_json::from_value::<CacheKey>(saved["key"].clone())
                .ok()
                .as_ref()
                == Some(&key)
            {
                let mut raw = Vec::new();
                std::fs::File::open(&bin)?.read_to_end(&mut raw)?;
                let tokens = raw
                    .chunks_exact(4)
                    .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect();
                let n_records = saved["n_records"].as_u64().unwrap_or(0) as usize;
                return Ok(Self { tokens, n_records });
            }
        }
        let mut tokens = Vec::new();
        let mut n_records = 0;
        let mut batch = Vec::new();
        'files: for file in &files {
            let f =
                std::fs::File::open(file).with_context(|| format!("opening {}", file.display()))?;
            for line in BufReader::new(f).lines() {
                let line = line?;
                if line.trim().is_empty() {
                    continue;
                }
                let rec: MlmRecord = serde_json::from_str(&line)
                    .with_context(|| format!("parsing {}", file.display()))?;
                batch.push(rec.text);
                if batch.len() == 20_000 {
                    n_records += Self::append(&mut tokens, tok, std::mem::take(&mut batch))?;
                    if tokens.len() >= max_tokens {
                        break 'files;
                    }
                }
            }
        }
        if !batch.is_empty() && tokens.len() < max_tokens {
            n_records += Self::append(&mut tokens, tok, batch)?;
        }
        tokens.truncate(max_tokens);
        std::fs::create_dir_all(cache_dir)?;
        let mut w = std::io::BufWriter::new(std::fs::File::create(&bin)?);
        for t in &tokens {
            w.write_all(&t.to_le_bytes())?;
        }
        w.flush()?;
        std::fs::write(
            &meta,
            serde_json::to_string(&serde_json::json!({"key": key, "n_records": n_records}))?,
        )?;
        Ok(Self { tokens, n_records })
    }

    fn append(tokens: &mut Vec<u32>, tok: &Tok, texts: Vec<String>) -> Result<usize> {
        let n = texts.len();
        for ids in tok.encode_batch(texts)? {
            if ids.is_empty() {
                continue;
            }
            tokens.extend(ids);
            tokens.push(SEP_ID);
        }
        Ok(n)
    }
}

/// Fixed-length rows cut from the corpus stream: `[CLS] chunk [SEP]`, where
/// consecutive records are packed into the same chunk.
#[derive(Debug, Clone)]
pub struct Rows {
    pub seq_len: usize,
    /// Start offsets (into the corpus) of training chunks.
    pub train: Vec<usize>,
    /// Start offsets of held-out chunks.
    pub val: Vec<usize>,
}

impl Rows {
    /// Splits the stream into chunks of `seq_len - 2` tokens; the last
    /// `val_fraction` (at most `max_val`) of the chunks are held out.
    pub fn new(corpus: &Corpus, seq_len: usize, val_fraction: f64, max_val: usize) -> Self {
        let body = seq_len - 2;
        let n = corpus.tokens.len().div_ceil(body);
        let n_val = ((n as f64 * val_fraction).ceil() as usize)
            .min(max_val)
            .min(n.saturating_sub(1));
        let starts: Vec<usize> = (0..n).map(|c| c * body).collect();
        let (train, val) = starts.split_at(n - n_val);
        Self {
            seq_len,
            train: train.to_vec(),
            val: val.to_vec(),
        }
    }

    /// Row tokens starting at `start` (may be shorter at the end of the stream).
    pub fn row(&self, corpus: &Corpus, start: usize) -> Vec<u32> {
        let end = (start + self.seq_len - 2).min(corpus.tokens.len());
        let mut r = Vec::with_capacity(self.seq_len);
        r.push(CLS_ID);
        r.extend_from_slice(&corpus.tokens[start..end]);
        r.push(SEP_ID);
        r
    }
}

/// Position of the training data stream (epoch-shuffled chunk order).
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct Cursor {
    pub epoch: u64,
    pub pos: usize,
}

impl Cursor {
    /// The next `n` training chunk starts; the order of each epoch is a
    /// permutation derived from `(seed, epoch)` so it is reproducible.
    pub fn next(&mut self, rows: &Rows, seed: u64, n: usize) -> Vec<usize> {
        let mut out = Vec::with_capacity(n);
        let mut perm = Self::perm(rows, seed, self.epoch);
        while out.len() < n {
            if self.pos >= perm.len() {
                self.epoch += 1;
                self.pos = 0;
                perm = Self::perm(rows, seed, self.epoch);
            }
            out.push(rows.train[perm[self.pos]]);
            self.pos += 1;
        }
        out
    }

    fn perm(rows: &Rows, seed: u64, epoch: u64) -> Vec<usize> {
        let mut p: Vec<usize> = (0..rows.train.len()).collect();
        Rng::new(seed ^ epoch.wrapping_mul(0x9E37_79B9_7F4A_7C15)).shuffle(&mut p);
        p
    }
}

/// Masks rows for MLM: `ratio` of the maskable tokens (non-special) are
/// predicted; of those 80% become `[MASK]`, 10% a random token and 10% stay.
/// The prediction list is padded to a multiple of 256 with zero weights.
pub fn mask_batch(rows: &[Vec<u32>], vocab: usize, ratio: f64, rng: &mut Rng) -> MlmBatch {
    let b = rows.len();
    let t = rows.iter().map(Vec::len).max().unwrap_or(1).max(1);
    let mut ids = vec![PAD_ID as i32; b * t];
    let mut valid = vec![false; b * t];
    let (mut positions, mut labels) = (Vec::new(), Vec::new());
    let mut n_tokens = 0;
    for (i, r) in rows.iter().enumerate() {
        n_tokens += r.len();
        let mut cand = Vec::new();
        for (j, &id) in r.iter().enumerate() {
            ids[i * t + j] = id as i32;
            valid[i * t + j] = true;
            if id >= N_SPECIAL {
                cand.push(j);
            }
        }
        let n_mask =
            ((cand.len() as f64 * ratio).round() as usize).max(usize::from(!cand.is_empty()));
        // Partial Fisher–Yates: the first n_mask candidates are the chosen ones.
        for s in 0..n_mask {
            let k = s + rng.below(cand.len() - s);
            cand.swap(s, k);
        }
        let mut chosen = cand[..n_mask].to_vec();
        chosen.sort_unstable();
        for j in chosen {
            let flat = i * t + j;
            labels.push(r[j] as i32);
            positions.push(flat as i32);
            let u = rng.f64();
            if u < 0.8 {
                ids[flat] = MASK_ID as i32;
            } else if u < 0.9 {
                ids[flat] = (N_SPECIAL as usize + rng.below(vocab - N_SPECIAL as usize)) as i32;
            }
        }
    }
    let m = positions.len();
    let padded = m.div_ceil(256).max(1) * 256;
    let mut weights = vec![1f32; m];
    weights.resize(padded, 0.0);
    positions.resize(padded, 0);
    labels.resize(padded, 0);
    let shape = [b as i32, t as i32];
    let key_valid = if valid.iter().all(|&v| v) {
        None
    } else {
        Some(Array::from_slice(&valid, &shape))
    };
    MlmBatch {
        ids: Array::from_slice(&ids, &shape),
        key_valid,
        positions: Array::from_slice(&positions, &[padded as i32]),
        labels: Array::from_slice(&labels, &[padded as i32]),
        weights: Array::from_slice(&weights, &[padded as i32]),
        n_tokens,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn masking_statistics() {
        let _g = crate::test_util::lock();
        let mut rng = Rng::new(3);
        let rows: Vec<Vec<u32>> = (0..64)
            .map(|_| {
                let mut r = vec![CLS_ID];
                r.extend((0..126).map(|_| 10 + rng.below(90) as u32));
                r.push(SEP_ID);
                r
            })
            .collect();
        let b = mask_batch(&rows, 100, 0.3, &mut rng);
        b.weights.eval().unwrap();
        let m = b
            .weights
            .as_slice::<f32>()
            .iter()
            .filter(|&&w| w > 0.0)
            .count();
        assert_eq!(m, 64 * 38); // round(126 * 0.3) per row
        b.ids.eval().unwrap();
        b.positions.eval().unwrap();
        b.labels.eval().unwrap();
        let ids = b.ids.as_slice::<i32>();
        let pos = b.positions.as_slice::<i32>();
        let lab = b.labels.as_slice::<i32>();
        let (mut masked, mut kept) = (0, 0);
        for i in 0..m {
            let p = pos[i] as usize;
            assert!(
                !p.is_multiple_of(128) && p % 128 != 127,
                "special tokens are never predicted"
            );
            if ids[p] == MASK_ID as i32 {
                masked += 1;
            } else if ids[p] == lab[i] {
                kept += 1;
            }
        }
        let fm = masked as f64 / m as f64;
        assert!((0.76..0.84).contains(&fm), "{fm}");
        assert!(kept as f64 / m as f64 > 0.08);
        assert!(b.key_valid.is_none());
        assert_eq!(b.positions.dim(0) % 256, 0);
    }

    #[test]
    fn rows_and_cursor() {
        let corpus = Corpus {
            tokens: (10..1010).collect(),
            n_records: 1,
        };
        let rows = Rows::new(&corpus, 128, 0.1, 100);
        assert_eq!(rows.train.len() + rows.val.len(), 1000usize.div_ceil(126));
        assert_eq!(rows.val.len(), 1);
        let r = rows.row(&corpus, rows.train[1]);
        assert_eq!(r.len(), 128);
        assert_eq!((r[0], r[1], r[127]), (CLS_ID, 136, SEP_ID));
        let mut c = Cursor::default();
        let a = c.next(&rows, 5, 10);
        let mut c2 = Cursor::default();
        let mut b = c2.next(&rows, 5, 4);
        b.extend(c2.next(&rows, 5, 6));
        assert_eq!(a, b);
        assert_eq!(c, c2);
        assert_eq!(c.epoch, 1);
    }
}
