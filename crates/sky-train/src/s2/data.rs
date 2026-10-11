//! s2 inputs: numeric extraction + BPE of the static (question + options) and
//! state streams with the SPEC truncation rules, decision data with a shared
//! table of static streams, padded batches, and the numeric MLM corpus.

use std::{
    collections::HashMap,
    io::{BufRead, BufReader, Read, Write},
    path::Path,
};

use anyhow::{bail, Context, Result};
use mlx_rs::Array;

use crate::{
    config::{CLS_ID, MASK_ID, PAD_ID, SEP_ID},
    decision::{group_name, list_jsonl, read_records},
    mlm::Corpus,
    num::{extract, signlog, NUM_ID, NUM_TOKEN},
    rng::Rng,
    sequence::{qtype_id, DecisionRecord, Tok, MASK_TEXT, OPTION_MAX_TOKENS},
};

/// Token ids with the parallel value array (0 for non-number tokens).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Stream {
    pub ids: Vec<u32>,
    pub values: Vec<f32>,
}

impl Stream {
    pub fn len(&self) -> usize {
        self.ids.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    fn push(&mut self, id: u32, v: f32) {
        self.ids.push(id);
        self.values.push(v);
    }

    fn extend_from(&mut self, other: &Stream, n: usize) {
        let n = n.min(other.len());
        self.ids.extend_from_slice(&other.ids[..n]);
        self.values.extend_from_slice(&other.values[..n]);
    }

    fn truncate(&mut self, n: usize) {
        self.ids.truncate(n);
        self.values.truncate(n);
    }
}

/// Removes the texts of the special tokens that the extraction relies on.
fn clean(text: &str) -> String {
    text.replace(MASK_TEXT, " ").replace(NUM_TOKEN, " ")
}

/// The texts of a record that get tokenized: head `"{t} question: {ins}"`,
/// options `" " + opt`, state.
pub fn s2_texts(rec: &DecisionRecord) -> (String, Vec<String>, String) {
    let head = format!("{} question: {}", rec.t, clean(&rec.ins));
    let opts = rec.opts.iter().map(|o| format!(" {}", clean(o))).collect();
    (head, opts, clean(&rec.state))
}

/// Extracts numbers, tokenizes (batched) and attaches the values to the
/// `[NUM]` tokens in order.
pub fn tokenize(tok: &Tok, texts: &[String]) -> Result<Vec<Stream>> {
    let ex: Vec<_> = texts.iter().map(|t| extract(t)).collect();
    let mut out = Vec::with_capacity(texts.len());
    for chunk in ex.chunks(50_000) {
        let ids = tok.encode_batch(chunk.iter().map(|e| e.text.clone()).collect())?;
        for (e, ids) in chunk.iter().zip(ids) {
            let mut vals = e.values.iter();
            let values = ids
                .iter()
                .map(|&id| {
                    if id == NUM_ID {
                        *vals.next().unwrap_or(&0.0)
                    } else {
                        0.0
                    }
                })
                .collect();
            out.push(Stream { ids, values });
        }
    }
    Ok(out)
}

/// The static stream of one question: `Q = [CLS] head [SEP]`, then
/// `O_i = [MASK] " opt_i"` for every option.
#[derive(Debug, Clone, PartialEq)]
pub struct StaticSeq {
    pub tokens: Stream,
    /// 0 for Q tokens, `i + 1` for the tokens of option `i`.
    pub seg: Vec<i32>,
    /// RoPE positions: Q gets `0..|Q|`, every option restarts at `|Q|`.
    pub pos: Vec<i32>,
    /// Position of each option's `[MASK]` marker.
    pub markers: Vec<usize>,
    /// `|Q|`.
    pub q_len: usize,
}

impl StaticSeq {
    pub fn k(&self) -> usize {
        self.markers.len()
    }
}

/// Builds the static stream with the SPEC.md truncation rules: options are
/// already capped at 48 tokens; when the options leave less than 16 tokens of
/// `head_max_len`, every option (marker included) is cut to
/// `max(4, (head_max_len - 16) / n)`; the head keeps `max(8, budget)` tokens.
pub fn build_static(head: &Stream, opts: &[Stream], head_max_len: usize) -> StaticSeq {
    let mut opt_streams: Vec<Stream> = opts
        .iter()
        .map(|o| {
            let mut s = Stream::default();
            s.push(MASK_ID, 0.0);
            s.extend_from(o, OPTION_MAX_TOKENS);
            s
        })
        .collect();
    let total: usize = opt_streams.iter().map(Stream::len).sum();
    let mut budget = head_max_len as i64 - total as i64;
    if budget < 16 {
        let per =
            4.max((head_max_len as i64 - 16).div_euclid(opt_streams.len().max(1) as i64)) as usize;
        for o in &mut opt_streams {
            o.truncate(per);
        }
        budget = head_max_len as i64 - opt_streams.iter().map(Stream::len).sum::<usize>() as i64;
    }
    let keep = budget.max(8) as usize;
    let mut tokens = Stream::default();
    tokens.push(CLS_ID, 0.0);
    tokens.extend_from(head, keep);
    tokens.push(SEP_ID, 0.0);
    let q_len = tokens.len();
    let mut seg = vec![0i32; q_len];
    let mut pos: Vec<i32> = (0..q_len as i32).collect();
    let mut markers = Vec::with_capacity(opt_streams.len());
    for (i, o) in opt_streams.iter().enumerate() {
        markers.push(tokens.len());
        tokens.extend_from(o, o.len());
        seg.extend(std::iter::repeat_n(i as i32 + 1, o.len()));
        pos.extend((0..o.len() as i32).map(|j| q_len as i32 + j));
    }
    StaticSeq {
        tokens,
        seg,
        pos,
        markers,
        q_len,
    }
}

/// The state stream `[CLS] {state} [SEP]`, truncated to `max_state` tokens.
pub fn build_state(state: &Stream, max_state: usize) -> Stream {
    let mut s = Stream::default();
    s.push(CLS_ID, 0.0);
    s.extend_from(state, max_state.saturating_sub(2));
    s.push(SEP_ID, 0.0);
    s
}

/// Tokenizes one record into its static and state streams.
pub fn encode_record(
    tok: &Tok,
    rec: &DecisionRecord,
    head_max_len: usize,
    max_state: usize,
) -> Result<(StaticSeq, Stream)> {
    let (h, o, s) = s2_texts(rec);
    let mut texts = vec![h];
    texts.extend(o);
    texts.push(s);
    let mut st = tokenize(tok, &texts)?;
    let state = st.pop().unwrap_or_default();
    let head = st.remove(0);
    Ok((
        build_static(&head, &st, head_max_len),
        build_state(&state, max_state),
    ))
}

/// One decision question: a static stream (shared) and its state.
#[derive(Debug, Clone)]
pub struct S2Item {
    /// Index into [`S2Data::statics`].
    pub stat: usize,
    pub state: Stream,
    pub qtype: usize,
    /// Normalized target over the options.
    pub target: Vec<f32>,
    pub src: String,
    pub group: usize,
}

impl S2Item {
    pub fn k(&self) -> usize {
        self.target.len()
    }
}

/// Decision data with deduplicated static streams.
#[derive(Debug, Default)]
pub struct S2Data {
    pub groups: Vec<String>,
    pub statics: Vec<StaticSeq>,
    /// Question key of each static stream (`t`, `ins`, `opts`).
    pub static_keys: Vec<(String, String, Vec<String>)>,
    pub train: Vec<Vec<S2Item>>,
    pub val: Vec<S2Item>,
    pub test: Vec<S2Item>,
    pub skipped: usize,
}

impl S2Data {
    /// Items of a split (`train` = every group).
    pub fn split(&self, split: &str) -> Vec<&S2Item> {
        match split {
            "val" => self.val.iter().collect(),
            "test" => self.test.iter().collect(),
            _ => self.train.iter().flatten().collect(),
        }
    }
}

/// Loads and tokenizes the decision records of every input path.
pub fn load(
    paths: &[std::path::PathBuf],
    tok: &Tok,
    head_max_len: usize,
    max_state: usize,
) -> Result<S2Data> {
    let mut data = S2Data::default();
    let mut index: HashMap<(String, String, Vec<String>), usize> = HashMap::new();
    for (g, path) in paths.iter().enumerate() {
        data.groups.push(group_name(path));
        data.train.push(Vec::new());
        let recs = read_records(path)?;
        let mut kept = Vec::new();
        let mut texts = Vec::new();
        let mut new_statics = Vec::new();
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
            let (h, o, s) = s2_texts(&rec);
            let key = (rec.t.clone(), rec.ins.clone(), rec.opts.clone());
            if !index.contains_key(&key) {
                index.insert(key.clone(), data.statics.len() + new_statics.len());
                new_statics.push((key.clone(), h, o));
            }
            texts.push(s);
            kept.push((rec, key));
        }
        // Tokenize the new static streams.
        let mut st_texts = Vec::new();
        for (_, h, o) in &new_statics {
            st_texts.push(h.clone());
            st_texts.extend(o.iter().cloned());
        }
        let mut it = tokenize(tok, &st_texts)?.into_iter();
        for (key, _, o) in new_statics {
            let head = it.next().unwrap_or_default();
            let opts: Vec<Stream> = (0..o.len())
                .map(|_| it.next().unwrap_or_default())
                .collect();
            data.statics.push(build_static(&head, &opts, head_max_len));
            data.static_keys.push(key);
        }
        let states = tokenize(tok, &texts)?;
        for ((rec, key), state) in kept.into_iter().zip(states) {
            let sum: f32 = rec.target.iter().sum();
            let item = S2Item {
                stat: index[&key],
                state: build_state(&state, max_state),
                qtype: qtype_id(&rec.t).unwrap_or(0),
                target: rec.target.iter().map(|x| x / sum).collect(),
                src: rec.src,
                group: g,
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

/// A padded s2 decision batch (see `s2::model::forward`).
#[derive(Debug, Clone)]
pub struct S2Batch {
    /// Unique static streams `[U, Ts]`.
    pub st_ids: Array,
    pub st_values: Array,
    /// Segment ids `[U, Ts]` (-1 = padding).
    pub st_seg: Array,
    pub st_pos: Array,
    /// Flat index into `U * Ts` of each row's markers `[B, K]`.
    pub marker_idx: Array,
    /// Flat index into `U * Ts` of each row's question tokens `[B, Tq]`.
    pub q_idx: Array,
    pub q_valid: Array,
    /// State streams `[B, S]`.
    pub ids: Array,
    pub values: Array,
    pub valid: Array,
    /// `[B]` question types.
    pub qtype: Array,
    pub marker_mask: Array,
    pub target: Array,
    /// Real (non-padding) tokens of both streams.
    pub n_tokens: usize,
    pub n_rows: usize,
}

fn round_up(x: usize, m: usize) -> usize {
    x.max(1).div_ceil(m.max(1)) * m.max(1)
}

/// Pads `items` into a batch; every padded length is rounded up to a multiple
/// of `pad_to` (the static count to a multiple of 8 and the option count to an
/// even number when `pad_to > 1`) so compiled graphs get few distinct shapes.
pub fn make_batch(statics: &[StaticSeq], items: &[&S2Item], pad_to: usize) -> S2Batch {
    let b = items.len();
    let mut uniq: Vec<usize> = Vec::new();
    let mut slot: HashMap<usize, usize> = HashMap::new();
    for it in items {
        slot.entry(it.stat).or_insert_with(|| {
            uniq.push(it.stat);
            uniq.len() - 1
        });
    }
    let coarse = pad_to > 1;
    let u = if coarse {
        round_up(uniq.len(), 8)
    } else {
        uniq.len()
    };
    let ts = round_up(
        uniq.iter()
            .map(|&s| statics[s].tokens.len())
            .max()
            .unwrap_or(1),
        pad_to,
    );
    let tq = round_up(
        uniq.iter().map(|&s| statics[s].q_len).max().unwrap_or(1),
        pad_to,
    );
    let kmax = items.iter().map(|i| i.k()).max().unwrap_or(1).max(1);
    let k = if coarse { round_up(kmax, 2) } else { kmax };
    let s = round_up(
        items.iter().map(|i| i.state.len()).max().unwrap_or(1),
        pad_to,
    );
    let mut st_ids = vec![PAD_ID as i32; u * ts];
    let mut st_values = vec![0f32; u * ts];
    let mut st_seg = vec![-1i32; u * ts];
    let mut st_pos = vec![0i32; u * ts];
    let mut n_tokens = 0;
    for (j, &si) in uniq.iter().enumerate() {
        let st = &statics[si];
        n_tokens += st.tokens.len();
        for t in 0..st.tokens.len() {
            st_ids[j * ts + t] = st.tokens.ids[t] as i32;
            st_values[j * ts + t] = st.tokens.values[t];
            st_seg[j * ts + t] = st.seg[t];
            st_pos[j * ts + t] = st.pos[t];
        }
    }
    let mut marker_idx = vec![0i32; b * k];
    let mut marker_mask = vec![false; b * k];
    let mut target = vec![0f32; b * k];
    let mut q_idx = vec![0i32; b * tq];
    let mut q_valid = vec![false; b * tq];
    let mut ids = vec![PAD_ID as i32; b * s];
    let mut values = vec![0f32; b * s];
    let mut valid = vec![false; b * s];
    for (r, it) in items.iter().enumerate() {
        let j = slot[&it.stat];
        let st = &statics[it.stat];
        for (o, &m) in st.markers.iter().enumerate() {
            marker_idx[r * k + o] = (j * ts + m) as i32;
            marker_mask[r * k + o] = true;
            target[r * k + o] = it.target[o];
        }
        for t in 0..st.q_len {
            q_idx[r * tq + t] = (j * ts + t) as i32;
            q_valid[r * tq + t] = true;
        }
        n_tokens += it.state.len();
        for t in 0..it.state.len() {
            ids[r * s + t] = it.state.ids[t] as i32;
            values[r * s + t] = it.state.values[t];
            valid[r * s + t] = true;
        }
    }
    let (bi, ui, ki) = (b as i32, u as i32, k as i32);
    S2Batch {
        st_ids: Array::from_slice(&st_ids, &[ui, ts as i32]),
        st_values: Array::from_slice(&st_values, &[ui, ts as i32]),
        st_seg: Array::from_slice(&st_seg, &[ui, ts as i32]),
        st_pos: Array::from_slice(&st_pos, &[ui, ts as i32]),
        marker_idx: Array::from_slice(&marker_idx, &[bi, ki]),
        q_idx: Array::from_slice(&q_idx, &[bi, tq as i32]),
        q_valid: Array::from_slice(&q_valid, &[bi, tq as i32]),
        ids: Array::from_slice(&ids, &[bi, s as i32]),
        values: Array::from_slice(&values, &[bi, s as i32]),
        valid: Array::from_slice(&valid, &[bi, s as i32]),
        qtype: Array::from_iter(items.iter().map(|i| i.qtype as i32), &[bi]),
        marker_mask: Array::from_slice(&marker_mask, &[bi, ki]),
        target: Array::from_slice(&target, &[bi, ki]),
        n_tokens,
        n_rows: b,
    }
}

/// Samples a training batch: the group is drawn by `weights`, the item
/// uniformly within the group. Questions repeated in the batch share their
/// static stream.
pub fn sample_batch(
    data: &S2Data,
    weights: &[f64],
    batch: usize,
    rng: &mut Rng,
    pad_to: usize,
) -> S2Batch {
    let w: Vec<f64> = weights
        .iter()
        .zip(&data.train)
        .map(|(w, items)| if items.is_empty() { 0.0 } else { *w })
        .collect();
    let items: Vec<&S2Item> = (0..batch)
        .map(|_| {
            let g = rng.weighted(&w);
            &data.train[g][rng.below(data.train[g].len())]
        })
        .collect();
    make_batch(&data.statics, &items, pad_to)
}

// ----------------------------------------------------------------------------- numeric MLM

/// The tokenized MLM corpus with the value of every `[NUM]` token.
#[derive(Debug, Clone)]
pub struct NumCorpus {
    /// Tokens (each record followed by `[SEP]`), usable with [`crate::mlm::Rows`].
    pub corpus: Corpus,
    pub values: Vec<f32>,
}

#[derive(serde::Deserialize)]
struct MlmRecord {
    text: String,
}

impl NumCorpus {
    /// Tokenizes every MLM record under `text` up to `max_tokens` (cached in `cache_dir`).
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
        let key = serde_json::json!({
            "files": files.iter().map(|f| Ok((f.display().to_string(), std::fs::metadata(f)?.len()))).collect::<Result<Vec<_>>>()?,
            "tokenizer_bytes": std::fs::metadata(tokenizer_path)?.len(),
            "max_tokens": max_tokens,
            "format": "s2-num-1",
        });
        let (bin, vbin, meta) = (
            cache_dir.join("corpus.u32"),
            cache_dir.join("values.f32"),
            cache_dir.join("corpus.json"),
        );
        if let Ok(m) = std::fs::read_to_string(&meta) {
            let saved: serde_json::Value = serde_json::from_str(&m)?;
            if saved["key"] == key && bin.exists() && vbin.exists() {
                let read = |p: &Path| -> Result<Vec<u8>> {
                    let mut raw = Vec::new();
                    std::fs::File::open(p)?.read_to_end(&mut raw)?;
                    Ok(raw)
                };
                let tokens = read(&bin)?
                    .chunks_exact(4)
                    .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect();
                let values = read(&vbin)?
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect();
                let n_records = saved["n_records"].as_u64().unwrap_or(0) as usize;
                return Ok(Self {
                    corpus: Corpus { tokens, n_records },
                    values,
                });
            }
        }
        let (mut tokens, mut values, mut n_records) = (Vec::new(), Vec::new(), 0usize);
        let mut batch = Vec::new();
        let flush = |batch: &mut Vec<String>,
                     tokens: &mut Vec<u32>,
                     values: &mut Vec<f32>|
         -> Result<usize> {
            let n = batch.len();
            for s in tokenize(tok, batch)? {
                if s.is_empty() {
                    continue;
                }
                tokens.extend(s.ids);
                values.extend(s.values);
                tokens.push(SEP_ID);
                values.push(0.0);
            }
            batch.clear();
            Ok(n)
        };
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
                batch.push(clean(&rec.text));
                if batch.len() == 20_000 {
                    n_records += flush(&mut batch, &mut tokens, &mut values)?;
                    if tokens.len() >= max_tokens {
                        break 'files;
                    }
                }
            }
        }
        if !batch.is_empty() && tokens.len() < max_tokens {
            n_records += flush(&mut batch, &mut tokens, &mut values)?;
        }
        tokens.truncate(max_tokens);
        values.truncate(max_tokens);
        std::fs::create_dir_all(cache_dir)?;
        let mut w = std::io::BufWriter::new(std::fs::File::create(&bin)?);
        for t in &tokens {
            w.write_all(&t.to_le_bytes())?;
        }
        w.flush()?;
        let mut w = std::io::BufWriter::new(std::fs::File::create(&vbin)?);
        for v in &values {
            w.write_all(&v.to_le_bytes())?;
        }
        w.flush()?;
        std::fs::write(
            &meta,
            serde_json::to_string(&serde_json::json!({"key": key, "n_records": n_records}))?,
        )?;
        Ok(Self {
            corpus: Corpus { tokens, n_records },
            values,
        })
    }

    /// Number of `[NUM]` tokens.
    pub fn n_numbers(&self) -> usize {
        self.corpus.tokens.iter().filter(|&&t| t == NUM_ID).count()
    }

    /// Row `[CLS] chunk [SEP]` starting at `start` (see [`crate::mlm::Rows::row`]).
    pub fn row(&self, seq_len: usize, start: usize) -> Stream {
        let end = (start + seq_len - 2).min(self.corpus.tokens.len());
        let mut s = Stream::default();
        s.push(CLS_ID, 0.0);
        s.ids.extend_from_slice(&self.corpus.tokens[start..end]);
        s.values.extend_from_slice(&self.values[start..end]);
        s.push(SEP_ID, 0.0);
        s
    }
}

/// Bound of the signlog regression target of masked numbers (|v| ≈ 1.6e5):
/// web text holds ids and serial numbers up to |u| ≈ 74 whose squared errors
/// would otherwise swamp the token cross-entropy.
pub const NUM_TARGET_CLIP: f32 = 12.0;

/// Inputs of a numeric MLM step.
#[derive(Debug, Clone)]
pub struct NumMlmBatch {
    /// Corrupted ids `[B, T]`; masked numbers keep `[NUM]`.
    pub ids: Array,
    /// Values `[B, T]` (0 for masked numbers).
    pub values: Array,
    /// 1 at masked numbers `[B, T]` (f32).
    pub num_flag: Array,
    pub valid: Array,
    /// Flat positions / labels / weights of the masked non-number tokens `[M]`.
    pub positions: Array,
    pub labels: Array,
    pub weights: Array,
    /// Flat positions / signlog targets / weights of the masked numbers `[N]`.
    pub num_positions: Array,
    pub num_targets: Array,
    pub num_weights: Array,
    pub n_tokens: usize,
}

/// Masks `ratio` of the maskable tokens (every non-special token and `[NUM]`).
/// Masked words follow 80/10/10 (`[MASK]` / random / kept); a masked number
/// keeps the `[NUM]` id, its value becomes 0 and its flag 1.
pub fn mask_batch(rows: &[Stream], vocab: usize, ratio: f64, rng: &mut Rng) -> NumMlmBatch {
    let b = rows.len();
    let t = rows.iter().map(Stream::len).max().unwrap_or(1).max(1);
    let mut ids = vec![PAD_ID as i32; b * t];
    let mut values = vec![0f32; b * t];
    let mut flag = vec![0f32; b * t];
    let mut valid = vec![false; b * t];
    let (mut positions, mut labels) = (Vec::new(), Vec::new());
    let (mut npos, mut ntgt) = (Vec::new(), Vec::new());
    let mut n_tokens = 0;
    let first_word = NUM_ID + 1;
    for (i, r) in rows.iter().enumerate() {
        n_tokens += r.len();
        let mut cand = Vec::new();
        for (j, &id) in r.ids.iter().enumerate() {
            ids[i * t + j] = id as i32;
            values[i * t + j] = r.values[j];
            valid[i * t + j] = true;
            if id >= NUM_ID {
                cand.push(j);
            }
        }
        let n_mask =
            ((cand.len() as f64 * ratio).round() as usize).max(usize::from(!cand.is_empty()));
        for s in 0..n_mask {
            let k = s + rng.below(cand.len() - s);
            cand.swap(s, k);
        }
        let mut chosen = cand[..n_mask].to_vec();
        chosen.sort_unstable();
        for j in chosen {
            let f = i * t + j;
            if r.ids[j] == NUM_ID {
                npos.push(f as i32);
                ntgt.push(signlog(r.values[j]).clamp(-NUM_TARGET_CLIP, NUM_TARGET_CLIP));
                values[f] = 0.0;
                flag[f] = 1.0;
                continue;
            }
            labels.push(r.ids[j] as i32);
            positions.push(f as i32);
            let u = rng.f64();
            if u < 0.8 {
                ids[f] = MASK_ID as i32;
            } else if u < 0.9 {
                ids[f] = (first_word as usize + rng.below(vocab - first_word as usize)) as i32;
            }
        }
    }
    let pad = |v: &mut Vec<i32>, w: &mut Vec<f32>, n: usize, m: usize| {
        let p = n.div_ceil(m).max(1) * m;
        w.resize(n, 1.0);
        w.resize(p, 0.0);
        v.resize(p, 0);
        p
    };
    let (mut weights, mut nweights) = (Vec::new(), Vec::new());
    let m = positions.len();
    let pm = pad(&mut positions, &mut weights, m, 256);
    labels.resize(pm, 0);
    let n = npos.len();
    let pn = pad(&mut npos, &mut nweights, n, 64);
    ntgt.resize(pn, 0.0);
    let shape = [b as i32, t as i32];
    NumMlmBatch {
        ids: Array::from_slice(&ids, &shape),
        values: Array::from_slice(&values, &shape),
        num_flag: Array::from_slice(&flag, &shape),
        valid: Array::from_slice(&valid, &shape),
        positions: Array::from_slice(&positions, &[pm as i32]),
        labels: Array::from_slice(&labels, &[pm as i32]),
        weights: Array::from_slice(&weights, &[pm as i32]),
        num_positions: Array::from_slice(&npos, &[pn as i32]),
        num_targets: Array::from_slice(&ntgt, &[pn as i32]),
        num_weights: Array::from_slice(&nweights, &[pn as i32]),
        n_tokens,
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// A byte-level tokenizer.json with `[NUM]` = 5 and byte `b` at id `6 + b`.
    pub fn s2_tokenizer_json() -> String {
        crate::s2::golden::byte_tokenizer_json()
    }

    pub fn s2_tok() -> (tempfile::TempDir, Tok) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("tokenizer.json");
        std::fs::write(&path, s2_tokenizer_json()).unwrap();
        let tok = Tok::load(&path).unwrap();
        tok.check_num().unwrap();
        (dir, tok)
    }

    pub fn by(s: &str) -> Vec<u32> {
        s.bytes().map(|x| 6 + x as u32).collect()
    }

    fn rec(t: &str, ins: &str, opts: &[&str], state: &str) -> DecisionRecord {
        DecisionRecord {
            src: "unit".into(),
            t: t.into(),
            ins: ins.into(),
            opts: opts.iter().map(|s| s.to_string()).collect(),
            state: state.into(),
            target: vec![1.0; opts.len()],
            split: "train".into(),
        }
    }

    #[test]
    fn streams_with_numbers() {
        let (_d, tok) = s2_tok();
        let r = rec("choice", "go 2 m?", &["up 3", "b2"], "alt 12 vy -3");
        let (st, state) = encode_record(&tok, &r, 96, 64).unwrap();
        let mut q = vec![CLS_ID];
        // `[NUM]` absorbs the space on its left (lstrip).
        q.extend(by("choice question: go"));
        q.push(NUM_ID);
        q.extend(by(" m?"));
        q.push(SEP_ID);
        assert_eq!(st.q_len, q.len());
        assert_eq!(&st.tokens.ids[..q.len()], &q[..]);
        let i_num = q.iter().position(|&x| x == NUM_ID).unwrap();
        assert_eq!(st.tokens.values[i_num], 2.0);
        // Options: [MASK] " up" [NUM] and [MASK] " b2".
        let m0 = st.markers[0];
        assert_eq!(m0, q.len());
        assert_eq!(st.tokens.ids[m0], MASK_ID);
        assert_eq!(&st.tokens.ids[m0 + 1..m0 + 4], &by(" up")[..]);
        assert_eq!(
            (st.tokens.ids[m0 + 4], st.tokens.values[m0 + 4]),
            (NUM_ID, 3.0)
        );
        let m1 = st.markers[1];
        assert_eq!(m1, m0 + 5);
        assert_eq!(&st.tokens.ids[m1 + 1..], &by(" b2")[..]);
        // Segments and restarted positions.
        assert!(st.seg[..q.len()].iter().all(|&s| s == 0));
        assert!(st.seg[m0..m1].iter().all(|&s| s == 1));
        assert!(st.seg[m1..].iter().all(|&s| s == 2));
        assert_eq!(st.pos[m0], q.len() as i32);
        assert_eq!(st.pos[m1], q.len() as i32);
        assert_eq!(st.pos[m1 + 2], q.len() as i32 + 2);
        // State: [CLS] "alt" [NUM] " vy" [NUM] [SEP].
        let mut want = vec![CLS_ID];
        want.extend(by("alt"));
        want.push(NUM_ID);
        want.extend(by(" vy"));
        want.push(NUM_ID);
        want.push(SEP_ID);
        assert_eq!(state.ids, want);
        assert_eq!(
            state
                .values
                .iter()
                .filter(|&&v| v != 0.0)
                .copied()
                .collect::<Vec<_>>(),
            vec![12.0, -3.0]
        );
    }

    #[test]
    fn truncation_rules() {
        let (_d, tok) = s2_tok();
        // Head budget: 32 - (1+2)*2 = 26 head tokens.
        let r = rec("noul", &"x".repeat(100), &["a", "b"], &"s".repeat(100));
        let (st, state) = encode_record(&tok, &r, 32, 20).unwrap();
        assert_eq!(st.q_len, 1 + 26 + 1);
        assert_eq!(st.tokens.len(), 28 + 6);
        assert_eq!(state.len(), 20);
        assert_eq!(*state.ids.last().unwrap(), SEP_ID);
        // Shrinking: 3 options of 20 bytes -> per = max(4, 16 / 3) = 5 tokens; head keeps 17.
        let o = "o".repeat(20);
        let r = rec("choice", &"x".repeat(100), &[&o, &o, &o], "");
        let (st, state) = encode_record(&tok, &r, 32, 64).unwrap();
        assert_eq!(st.q_len, 1 + 17 + 1);
        assert_eq!(st.markers, vec![19, 24, 29]);
        assert_eq!(&st.tokens.ids[20..24], &by(" ooo")[..]);
        assert_eq!(state.ids, vec![CLS_ID, SEP_ID]);
        // Options are capped at 48 tokens before the budget.
        let r = rec("choice", "q", &[&"z".repeat(80)], "");
        let (st, _) = encode_record(&tok, &r, 200, 64).unwrap();
        assert_eq!(st.tokens.len() - st.markers[0], 49);
        // A literal [NUM] / [MASK] in the input is neutralized.
        let r = rec("noul", "a [NUM] b [MASK]", &["x"], "");
        let (st, _) = encode_record(&tok, &r, 96, 64).unwrap();
        assert!(!st.tokens.ids[..st.q_len].contains(&NUM_ID));
        assert!(!st.tokens.ids[1..st.q_len].contains(&MASK_ID));
    }

    #[test]
    fn batch_layout_shares_statics() {
        let _g = crate::test_util::lock();
        let (_d, tok) = s2_tok();
        let r1 = rec("noul", "q one", &["false: n", "true: y"], "a 1");
        let r2 = rec("choice", "q two", &["a", "b", "c"], "b 2 c");
        let (s1, st1) = encode_record(&tok, &r1, 96, 64).unwrap();
        let (s2, st2) = encode_record(&tok, &r2, 96, 64).unwrap();
        let statics = vec![s1.clone(), s2.clone()];
        let it = |stat: usize, state: &Stream, k: usize| S2Item {
            stat,
            state: state.clone(),
            qtype: 0,
            target: vec![1.0 / k as f32; k],
            src: "x".into(),
            group: 0,
        };
        let items = [it(1, &st2, 3), it(0, &st1, 2), it(1, &st1, 3)];
        let refs: Vec<&S2Item> = items.iter().collect();
        let b = make_batch(&statics, &refs, 1);
        // Two unique statics, the first one seen is static 1.
        assert_eq!(b.st_ids.dim(0), 2);
        assert_eq!(b.st_ids.dim(1), s2.tokens.len().max(s1.tokens.len()) as i32);
        assert_eq!(b.marker_idx.shape(), &[3, 3]);
        b.marker_idx.eval().unwrap();
        let mi = b.marker_idx.as_slice::<i32>();
        let ts = b.st_ids.dim(1) as usize;
        assert_eq!(mi[0] as usize, s2.markers[0]);
        assert_eq!(mi[3] as usize, ts + s1.markers[0]);
        assert_eq!(mi[6..9], mi[0..3]);
        b.marker_mask.eval().unwrap();
        assert_eq!(
            b.marker_mask.as_slice::<bool>(),
            &[true, true, true, true, true, false, true, true, true]
        );
        let c = make_batch(&statics, &refs, 16);
        assert_eq!(
            (c.st_ids.dim(0), c.st_ids.dim(1) % 16, c.ids.dim(1) % 16),
            (8, 0, 0)
        );
        assert_eq!(c.marker_idx.dim(1), 4);
    }

    #[test]
    fn numeric_masking() {
        let _g = crate::test_util::lock();
        let mut rng = Rng::new(4);
        let rows: Vec<Stream> = (0..32)
            .map(|_| {
                let mut s = Stream::default();
                s.push(CLS_ID, 0.0);
                for j in 0..60 {
                    if j % 5 == 0 {
                        s.push(NUM_ID, j as f32 - 20.0);
                    } else {
                        s.push(10 + rng.below(80) as u32, 0.0);
                    }
                }
                s.push(SEP_ID, 0.0);
                s
            })
            .collect();
        let b = mask_batch(&rows, 100, 0.3, &mut rng);
        for a in [
            &b.num_weights,
            &b.weights,
            &b.num_targets,
            &b.num_positions,
            &b.values,
            &b.num_flag,
            &b.ids,
            &b.labels,
        ] {
            a.eval().unwrap();
        }
        let nn = b
            .num_weights
            .as_slice::<f32>()
            .iter()
            .filter(|&&w| w > 0.0)
            .count();
        let nw = b
            .weights
            .as_slice::<f32>()
            .iter()
            .filter(|&&w| w > 0.0)
            .count();
        assert_eq!(nn + nw, 32 * 18);
        assert!(nn > 32 && nw > 32 * 10, "{nn} {nw}");
        let (ids, vals, flag) = (
            b.ids.as_slice::<i32>(),
            b.values.as_slice::<f32>(),
            b.num_flag.as_slice::<f32>(),
        );
        let (pos, tgt) = (
            b.num_positions.as_slice::<i32>(),
            b.num_targets.as_slice::<f32>(),
        );
        for i in 0..nn {
            let p = pos[i] as usize;
            assert_eq!(ids[p], NUM_ID as i32);
            assert_eq!(vals[p], 0.0);
            assert_eq!(flag[p], 1.0);
            let orig = rows[p / 62].values[p % 62];
            assert!((tgt[i] - signlog(orig)).abs() < 1e-6);
        }
        assert_eq!(flag.iter().sum::<f32>() as usize, nn);
        // No word is replaced by [NUM], labels are never [NUM].
        let lab = b.labels.as_slice::<i32>();
        assert!(lab[..nw].iter().all(|&l| l != NUM_ID as i32));
        let n_num_ids = ids.iter().filter(|&&x| x == NUM_ID as i32).count();
        assert_eq!(n_num_ids, 32 * 12);
    }
}
