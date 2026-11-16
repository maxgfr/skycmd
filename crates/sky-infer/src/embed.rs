//! s6-nano v3 (`docs/s6-nano3-prereg.md` §2): the embedding part `potion-cos-4bit` of the fused
//! router and the fusion with the model part.
//!
//! **The embedding part** is `minishlab/potion-base-2M` (model2vec, MIT) as
//! `scripts/route-baselines.py` computes it, read from one GGUF file (`general.architecture` =
//! `skycmd-embed`, written by `sky-route embed-gguf` from the export of
//! `scripts/s6-route3-embed-export.py`):
//!
//! - **tokens** (the snapshot's BERT WordPiece tokenizer through model2vec's `tokenize`): the
//!   literal added tokens (`[CLS]`, `[SEP]`, ...) are cut out of the raw text first; every other
//!   piece goes through the normaliser (clean text, CJK characters padded with spaces, accents
//!   stripped, lower case), stored in the file as a per-character map read from the tokenizers
//!   library itself; then split at whitespace with each punctuation character a word of its own;
//!   then greedy longest-match WordPiece (`##` continuation; a word longer than 100 characters,
//!   or one that cannot be covered, is `[UNK]`); the first 512 tokens are kept, then every
//!   `[UNK]` is removed;
//! - **vectors**: per token, 64 four-bit codes in -7..7 and a float16 scale (`code · scale`);
//! - **a text**: the float32 mean of its tokens' vectors, L2-normalised (the zero vector when
//!   it has no token);
//! - **a slate**: the cosine of the request with each tool option `name: description`, *none*
//!   at θ, probabilities softmax([cosines, θ] / T).
//!
//! **The fusion**: `p(i) ∝ (p_model(i) · p_embedding(i))^β` over the options (each probability
//! floored at 1e-12 before the logarithm, as `scripts/s6-route3-fit.py`), renormalised; the
//! router asks the user when the top fused probability is below τ_ask3. θ, T, β and τ_ask3 are
//! stored in the embedding file.
//!
//! The arithmetic follows numpy's where it matters for the last bits: the mean is a sequential
//! float32 sum divided by the count, the norm is numpy's pairwise float32 sum of squares.

use std::collections::HashMap;

#[cfg(test)]
use sky_gguf::{f16_to_f32, f32_to_f16};
use sky_gguf::{GgmlType, Gguf, GgufWriter, OwnedValue};

use crate::{Error, Result};

/// `general.architecture` of an embedding file.
pub const ARCH: &str = "skycmd-embed";
/// Probabilities are floored at this value before the logarithm of the fusion.
pub const FLOOR: f64 = 1e-12;

/// Metadata keys and tensor names of an embedding file.
pub mod keys {
    pub const DIM: &str = "skycmd-embed.dim";
    pub const N_VOCAB: &str = "skycmd-embed.n_vocab";
    pub const BITS: &str = "skycmd-embed.bits";
    pub const PACK: &str = "skycmd-embed.pack";
    pub const SYSTEM: &str = "skycmd-embed.system";
    pub const VOCAB: &str = "skycmd-embed.vocab";
    pub const UNK_ID: &str = "skycmd-embed.unk_id";
    pub const PREFIX: &str = "skycmd-embed.continuing_prefix";
    pub const MAX_CHARS: &str = "skycmd-embed.max_chars_per_word";
    pub const MAX_LENGTH: &str = "skycmd-embed.max_length";
    pub const NORMALIZE: &str = "skycmd-embed.normalize";
    pub const ADDED_TOKENS: &str = "skycmd-embed.added_tokens";
    pub const ADDED_IDS: &str = "skycmd-embed.added_ids";
    pub const NORM_STARTS: &str = "skycmd-embed.norm.starts";
    pub const NORM_OPS: &str = "skycmd-embed.norm.ops";
    pub const NORM_ARGS: &str = "skycmd-embed.norm.args";
    pub const NORM_STRINGS: &str = "skycmd-embed.norm.strings";
    pub const SPACE: &str = "skycmd-embed.pretok.space";
    pub const PUNCT: &str = "skycmd-embed.pretok.punct";
    pub const THETA: &str = "skycmd-embed.route.theta";
    pub const TEMPERATURE: &str = "skycmd-embed.route.temperature";
    pub const BETA: &str = "skycmd-embed.route.beta";
    pub const TAU_ASK: &str = "skycmd-embed.route.tau_ask3";
    pub const CODES: &str = "embed.codes";
    pub const SCALES: &str = "embed.scales";
}

/// Normaliser run operations (the numbers of `scripts/s6-route3-embed-export.py`).
pub mod op {
    /// The character is kept.
    pub const ID: u8 = 0;
    /// The character is removed.
    pub const DEL: u8 = 1;
    /// The character is surrounded by spaces (CJK ideographs).
    pub const PAD: u8 = 2;
    /// The character becomes `c + arg`.
    pub const DELTA: u8 = 3;
    /// The character becomes string `arg` of the string table (a run of one character).
    pub const STR: u8 = 4;
    /// A Hangul syllable, decomposed into its jamo by the Unicode algorithm.
    pub const HANGUL: u8 = 5;
}

/// The normaliser as runs of code points: run `k` covers `starts[k]..starts[k + 1]`.
#[derive(Clone, Debug, Default)]
pub struct NormTable {
    pub starts: Vec<u32>,
    pub ops: Vec<u8>,
    pub args: Vec<i32>,
    pub strings: Vec<String>,
}

/// Everything an embedding file holds, in memory (written by [`EmbedSpec::to_gguf`]).
#[derive(Clone, Debug)]
pub struct EmbedSpec {
    pub dim: usize,
    /// Two codes per byte per token (`dim / 2` bytes), the even dimension in the low nibble.
    pub codes: Vec<u8>,
    /// One float16 scale per token (bits).
    pub scales: Vec<u16>,
    /// The tokens, in id order.
    pub vocab: Vec<String>,
    pub unk_id: u32,
    pub continuing_prefix: String,
    pub max_chars_per_word: usize,
    pub max_length: usize,
    pub normalize: bool,
    pub added: Vec<(String, u32)>,
    pub norm: NormTable,
    /// Inclusive code point ranges of the pre-tokenizer's whitespace and punctuation.
    pub space: Vec<(u32, u32)>,
    pub punct: Vec<(u32, u32)>,
    pub theta: f64,
    pub temperature: f64,
    pub beta: f64,
    pub tau_ask: f64,
    /// Further metadata (provenance, licence), written as is.
    pub meta: Vec<(String, OwnedValue)>,
}

fn u32s(v: &[u32]) -> OwnedValue {
    OwnedValue::Array(v.iter().map(|&x| OwnedValue::U32(x)).collect())
}

fn flat(r: &[(u32, u32)]) -> Vec<u32> {
    r.iter().flat_map(|&(a, b)| [a, b]).collect()
}

impl EmbedSpec {
    /// The GGUF file.
    pub fn to_gguf(&self) -> Result<Vec<u8>> {
        let n = self.vocab.len();
        if self.dim % 2 != 0
            || self.codes.len() != n * self.dim / 2
            || self.scales.len() != n
            || self.vocab.iter().any(|t| t.contains('\n'))
        {
            return Err(Error::Format("embedding spec: inconsistent sizes".into()));
        }
        let mut w = GgufWriter::new();
        w.add_kv("general.architecture", ARCH);
        for (k, v) in &self.meta {
            w.add_kv(k.clone(), v.clone());
        }
        w.add_kv(keys::DIM, self.dim as u32)
            .add_kv(keys::N_VOCAB, n as u32)
            .add_kv(keys::BITS, 4u32)
            .add_kv(
                keys::PACK,
                "per token: scale = max |v| / 7 rounded to float16 (embed.scales); codes = rint(v / scale) \
                 clipped to -7..7, two per byte in embed.codes, the even dimension in the low nibble, \
                 two's complement; value = code * scale",
            )
            .add_kv(keys::VOCAB, self.vocab.join("\n"))
            .add_kv(keys::UNK_ID, self.unk_id)
            .add_kv(keys::PREFIX, self.continuing_prefix.as_str())
            .add_kv(keys::MAX_CHARS, self.max_chars_per_word as u32)
            .add_kv(keys::MAX_LENGTH, self.max_length as u32)
            .add_kv(keys::NORMALIZE, self.normalize)
            .add_kv(
                keys::ADDED_TOKENS,
                OwnedValue::strings(&self.added.iter().map(|a| a.0.as_str()).collect::<Vec<_>>()),
            )
            .add_kv(keys::ADDED_IDS, u32s(&self.added.iter().map(|a| a.1).collect::<Vec<_>>()))
            .add_kv(keys::NORM_STARTS, u32s(&self.norm.starts))
            .add_kv(
                keys::NORM_OPS,
                OwnedValue::Array(self.norm.ops.iter().map(|&x| OwnedValue::U8(x)).collect()),
            )
            .add_kv(
                keys::NORM_ARGS,
                OwnedValue::Array(self.norm.args.iter().map(|&x| OwnedValue::I32(x)).collect()),
            )
            .add_kv(keys::NORM_STRINGS, OwnedValue::strings(&self.norm.strings))
            .add_kv(keys::SPACE, u32s(&flat(&self.space)))
            .add_kv(keys::PUNCT, u32s(&flat(&self.punct)))
            .add_kv(keys::THETA, self.theta)
            .add_kv(keys::TEMPERATURE, self.temperature)
            .add_kv(keys::BETA, self.beta)
            .add_kv(keys::TAU_ASK, self.tau_ask);
        w.add_tensor(
            keys::CODES,
            &[(self.dim / 2) as u64, n as u64],
            GgmlType::I8,
            self.codes.clone(),
        )?;
        let scales: Vec<u8> = self.scales.iter().flat_map(|s| s.to_le_bytes()).collect();
        w.add_tensor(keys::SCALES, &[n as u64], GgmlType::F16, scales)?;
        Ok(w.to_bytes()?)
    }
}

/// A loaded static embedding (tokenizer, packed table, the router's fitted values).
pub struct StaticEmbedding {
    dim: usize,
    codes: Vec<u8>,
    scales: Vec<f32>,
    vocab: HashMap<String, u32>,
    n_vocab: usize,
    unk_id: u32,
    prefix: String,
    max_chars: usize,
    max_length: usize,
    normalize: bool,
    added: Vec<(String, u32)>,
    norm: NormTable,
    space: Vec<(u32, u32)>,
    punct: Vec<(u32, u32)>,
    /// *None*'s score.
    pub theta: f64,
    /// The softmax temperature of the embedding part.
    pub temperature: f64,
    /// The fusion exponent β.
    pub beta: f64,
    /// The ask threshold τ_ask3 on the top fused probability.
    pub tau_ask: f64,
    name: String,
    file_bytes: usize,
}

fn need<'a>(g: &'a Gguf<'_>, key: &str) -> Result<&'a sky_gguf::Value<'a>> {
    g.get(key).ok_or_else(|| Error::Missing(key.into()))
}

fn ints(g: &Gguf<'_>, key: &str) -> Result<Vec<i64>> {
    match g.get_array(key) {
        Some(a) if a.is_empty() => Ok(Vec::new()),
        Some(a) => a
            .to_i64_vec()
            .ok_or_else(|| Error::Format(format!("{key}: not integers"))),
        None => Err(Error::Missing(key.into())),
    }
}

fn pairs(v: Vec<i64>, key: &str) -> Result<Vec<(u32, u32)>> {
    if v.len() % 2 != 0 {
        return Err(Error::Format(format!("{key}: odd length")));
    }
    Ok(v.chunks(2).map(|p| (p[0] as u32, p[1] as u32)).collect())
}

fn in_ranges(r: &[(u32, u32)], c: u32) -> bool {
    let i = r.partition_point(|&(a, _)| a <= c);
    i > 0 && c <= r[i - 1].1
}

fn hangul(c: u32, out: &mut String) {
    let s = c - 0xAC00;
    let l = 0x1100 + s / 588;
    let v = 0x1161 + (s % 588) / 28;
    let t = 0x11A7 + s % 28;
    for x in [l, v] {
        out.extend(char::from_u32(x));
    }
    if t != 0x11A7 {
        out.extend(char::from_u32(t));
    }
}

/// numpy's float32 sum of `x` along a contiguous axis (pairwise: eight partial sums up to 128
/// elements; longer rows are split in halves).
fn np_sum_f32(x: &[f32]) -> f32 {
    let n = x.len();
    if n < 8 {
        let mut s = -0.0f32;
        for &v in x {
            s += v;
        }
        return s;
    }
    if n <= 128 {
        let mut r = [0f32; 8];
        r.copy_from_slice(&x[..8]);
        let mut i = 8;
        while i + 8 <= n {
            for j in 0..8 {
                r[j] += x[i + j];
            }
            i += 8;
        }
        let mut s = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        while i < n {
            s += x[i];
            i += 1;
        }
        return s;
    }
    let half = (n / 2) & !7;
    np_sum_f32(&x[..half]) + np_sum_f32(&x[half..])
}

impl StaticEmbedding {
    /// From the bytes of an embedding GGUF file.
    pub fn from_gguf(bytes: &[u8]) -> Result<StaticEmbedding> {
        let g = Gguf::parse(bytes)?;
        if g.architecture() != Some(ARCH) {
            return Err(Error::Format(format!(
                "not an embedding file (architecture {:?}, expected {ARCH})",
                g.architecture()
            )));
        }
        let u = |k: &str| -> Result<u64> {
            need(&g, k)?
                .as_u64()
                .ok_or_else(|| Error::Format(format!("{k}: not an integer")))
        };
        let f = |k: &str| -> Result<f64> {
            need(&g, k)?
                .as_f64()
                .ok_or_else(|| Error::Format(format!("{k}: not a number")))
        };
        let dim = u(keys::DIM)? as usize;
        let n_vocab = u(keys::N_VOCAB)? as usize;
        if u(keys::BITS)? != 4 || dim % 2 != 0 {
            return Err(Error::Unsupported(
                "only 4-bit tables of even dimension".into(),
            ));
        }
        let vocab_text = g
            .get_str(keys::VOCAB)
            .ok_or_else(|| Error::Missing(keys::VOCAB.into()))?;
        let mut vocab = HashMap::with_capacity(n_vocab);
        let mut n = 0usize;
        for (i, t) in vocab_text.split('\n').enumerate() {
            vocab.insert(t.to_string(), i as u32);
            n += 1;
        }
        if n != n_vocab {
            return Err(Error::Format(format!("{n} tokens, {n_vocab} declared")));
        }
        let codes_t = g
            .tensor(keys::CODES)
            .ok_or_else(|| Error::Missing(keys::CODES.into()))?;
        let codes = g.tensor_bytes(codes_t).to_vec();
        let scales = g.tensor_f32(keys::SCALES)?;
        if codes.len() != n_vocab * dim / 2 || scales.len() != n_vocab {
            return Err(Error::Format("embedding tensors: wrong sizes".into()));
        }
        let added_t = g
            .get_array(keys::ADDED_TOKENS)
            .and_then(|a| a.strings())
            .ok_or_else(|| Error::Missing(keys::ADDED_TOKENS.into()))?;
        let added_i = ints(&g, keys::ADDED_IDS)?;
        if added_t.len() != added_i.len() {
            return Err(Error::Format(
                "added tokens and ids differ in length".into(),
            ));
        }
        let mut added: Vec<(String, u32)> = added_t
            .iter()
            .zip(&added_i)
            .filter(|(t, _)| !t.is_empty())
            .map(|(t, &i)| (t.to_string(), i as u32))
            .collect();
        // leftmost-longest: at one position, the longer literal first
        added.sort_by(|a, b| b.0.len().cmp(&a.0.len()));
        let norm = NormTable {
            starts: ints(&g, keys::NORM_STARTS)?
                .into_iter()
                .map(|x| x as u32)
                .collect(),
            ops: ints(&g, keys::NORM_OPS)?
                .into_iter()
                .map(|x| x as u8)
                .collect(),
            args: ints(&g, keys::NORM_ARGS)?
                .into_iter()
                .map(|x| x as i32)
                .collect(),
            strings: g
                .get_array(keys::NORM_STRINGS)
                .and_then(|a| a.strings())
                .ok_or_else(|| Error::Missing(keys::NORM_STRINGS.into()))?
                .into_iter()
                .map(str::to_string)
                .collect(),
        };
        if norm.starts.is_empty()
            || norm.starts[0] != 0
            || norm.ops.len() != norm.starts.len()
            || norm.args.len() != norm.starts.len()
            || norm.starts.windows(2).any(|w| w[0] >= w[1])
        {
            return Err(Error::Format("normaliser table: malformed".into()));
        }
        Ok(StaticEmbedding {
            dim,
            codes,
            scales,
            vocab,
            n_vocab,
            unk_id: u(keys::UNK_ID)? as u32,
            prefix: g.get_str(keys::PREFIX).unwrap_or("##").to_string(),
            max_chars: u(keys::MAX_CHARS)? as usize,
            max_length: u(keys::MAX_LENGTH)? as usize,
            normalize: g.get_bool(keys::NORMALIZE).unwrap_or(true),
            added,
            norm,
            space: pairs(ints(&g, keys::SPACE)?, keys::SPACE)?,
            punct: pairs(ints(&g, keys::PUNCT)?, keys::PUNCT)?,
            theta: f(keys::THETA)?,
            temperature: f(keys::TEMPERATURE)?,
            beta: f(keys::BETA)?,
            tau_ask: f(keys::TAU_ASK)?,
            name: g.get_str("general.name").unwrap_or("embedding").to_string(),
            file_bytes: bytes.len(),
        })
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn n_vocab(&self) -> usize {
        self.n_vocab
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn file_bytes(&self) -> usize {
        self.file_bytes
    }

    /// Append the normalised form of one character.
    fn norm_char(&self, ch: char, out: &mut String) {
        let c = ch as u32;
        let t = &self.norm;
        let k = t.starts.partition_point(|&s| s <= c) - 1;
        match t.ops[k] {
            op::ID => out.push(ch),
            op::DEL => {}
            op::PAD => {
                out.push(' ');
                out.push(ch);
                out.push(' ');
            }
            op::DELTA => out.extend(char::from_u32((c as i64 + t.args[k] as i64) as u32)),
            op::STR => out.push_str(&t.strings[t.args[k] as usize]),
            op::HANGUL => hangul(c, out),
            _ => out.push(ch),
        }
    }

    /// The normaliser (clean text, CJK padding, accents stripped, lower case).
    pub fn normalize_text(&self, s: &str) -> String {
        let mut out = String::with_capacity(s.len() + 8);
        for ch in s.chars() {
            self.norm_char(ch, &mut out);
        }
        out
    }

    /// The pre-tokenizer: words split at whitespace, each punctuation character a word.
    fn words<'a>(&self, s: &'a str, out: &mut Vec<&'a str>) {
        let mut start = None;
        for (i, ch) in s.char_indices() {
            let c = ch as u32;
            let space = in_ranges(&self.space, c);
            let punct = !space && in_ranges(&self.punct, c);
            if space || punct {
                if let Some(a) = start.take() {
                    out.push(&s[a..i]);
                }
                if punct {
                    out.push(&s[i..i + ch.len_utf8()]);
                }
            } else if start.is_none() {
                start = Some(i);
            }
        }
        if let Some(a) = start {
            out.push(&s[a..]);
        }
    }

    /// WordPiece on one word (greedy longest match; `[UNK]` for the whole word on failure).
    fn wordpiece(&self, w: &str, out: &mut Vec<u32>, buf: &mut String) {
        if w.chars().count() > self.max_chars {
            out.push(self.unk_id);
            return;
        }
        let mark = out.len();
        let mut start = 0;
        while start < w.len() {
            let mut end = w.len();
            let mut hit = None;
            while start < end {
                let piece = &w[start..end];
                let id = if start == 0 {
                    self.vocab.get(piece)
                } else {
                    buf.clear();
                    buf.push_str(&self.prefix);
                    buf.push_str(piece);
                    self.vocab.get(buf.as_str())
                };
                if let Some(&id) = id {
                    hit = Some(id);
                    break;
                }
                // one character back
                end = w[..end]
                    .char_indices()
                    .next_back()
                    .map_or(start, |(i, _)| i);
            }
            match hit {
                Some(id) => {
                    out.push(id);
                    start = end;
                }
                None => {
                    out.truncate(mark);
                    out.push(self.unk_id);
                    return;
                }
            }
        }
    }

    /// The token ids model2vec's `tokenize` gives for `text` (truncated, `[UNK]` removed).
    pub fn token_ids(&self, text: &str) -> Vec<u32> {
        let mut ids = Vec::new();
        let mut buf = String::new();
        let mut piece = |seg: &str, ids: &mut Vec<u32>| {
            if seg.is_empty() {
                return;
            }
            let normed = self.normalize_text(seg);
            let mut words = Vec::new();
            self.words(&normed, &mut words);
            for w in &words {
                self.wordpiece(w, ids, &mut buf);
            }
        };
        // the added tokens, leftmost-longest in the raw text
        let (mut i, mut seg) = (0, 0);
        while i < text.len() {
            let rest = &text[i..];
            if let Some((lit, id)) = self
                .added
                .iter()
                .find(|(a, _)| rest.starts_with(a.as_str()))
            {
                piece(&text[seg..i], &mut ids);
                ids.push(*id);
                i += lit.len();
                seg = i;
                continue;
            }
            i += rest.chars().next().map_or(1, char::len_utf8);
        }
        piece(&text[seg..], &mut ids);
        ids.truncate(self.max_length);
        ids.retain(|&t| t != self.unk_id);
        ids
    }

    /// The vector of one token (`code · scale`).
    pub fn token_vector(&self, id: u32, out: &mut [f32]) {
        let half = self.dim / 2;
        let row = &self.codes[id as usize * half..(id as usize + 1) * half];
        let s = self.scales[id as usize];
        for (j, &b) in row.iter().enumerate() {
            let lo = ((b << 4) as i8) >> 4;
            let hi = (b as i8) >> 4;
            out[2 * j] = lo as f32 * s;
            out[2 * j + 1] = hi as f32 * s;
        }
    }

    /// The embedding of token ids: the float32 mean, L2-normalised; zero without a token.
    pub fn embed_ids(&self, ids: &[u32]) -> Vec<f32> {
        let mut sum = vec![0f32; self.dim];
        if ids.is_empty() {
            return sum;
        }
        let mut v = vec![0f32; self.dim];
        for (k, &id) in ids.iter().enumerate() {
            self.token_vector(id, &mut v);
            if k == 0 {
                sum.copy_from_slice(&v);
            } else {
                for (s, x) in sum.iter_mut().zip(&v) {
                    *s += *x;
                }
            }
        }
        let n = ids.len() as f32;
        for s in sum.iter_mut() {
            *s /= n;
        }
        if self.normalize {
            let sq: Vec<f32> = sum.iter().map(|x| x * x).collect();
            let norm = np_sum_f32(&sq).sqrt();
            let d = if norm == 0.0 { 1.0 } else { norm };
            for s in sum.iter_mut() {
                *s /= d;
            }
        }
        sum
    }

    /// The embedding of a text.
    pub fn embed(&self, text: &str) -> Vec<f32> {
        self.embed_ids(&self.token_ids(text))
    }

    /// The scores of a slate: the cosine of the request with each tool, then θ for *none*.
    pub fn scores(&self, request: &[f32], tools: &[Vec<f32>]) -> Vec<f64> {
        let mut s: Vec<f64> = tools
            .iter()
            .map(|t| {
                let d: f64 = t
                    .iter()
                    .zip(request)
                    .map(|(a, b)| *a as f64 * *b as f64)
                    .sum();
                d as f32 as f64
            })
            .collect();
        s.push(self.theta);
        s
    }

    /// softmax(scores / T).
    pub fn probs_from_scores(&self, scores: &[f64]) -> Vec<f64> {
        softmax(
            &scores
                .iter()
                .map(|s| s / self.temperature)
                .collect::<Vec<_>>(),
        )
    }

    /// The embedding part's probabilities over the tools and *none* (last).
    pub fn probs(&self, request: &str, tools: &[&str]) -> Vec<f64> {
        let q = self.embed(request);
        let t: Vec<Vec<f32>> = tools.iter().map(|o| self.embed(o)).collect();
        self.probs_from_scores(&self.scores(&q, &t))
    }
}

/// The embeddings of one tool list, kept while the list does not change.
#[derive(Clone, Debug, Default)]
pub struct ToolCache {
    tools: Vec<String>,
    vecs: Vec<Vec<f32>>,
}

impl ToolCache {
    /// The embeddings of `tools`, and whether they came from the cache.
    pub fn get(&mut self, e: &StaticEmbedding, tools: &[String]) -> (&[Vec<f32>], bool) {
        let hit =
            self.tools.len() == tools.len() && self.tools.iter().zip(tools).all(|(a, b)| a == b);
        if !hit {
            self.vecs = tools.iter().map(|t| e.embed(t)).collect();
            self.tools = tools.to_vec();
        }
        (&self.vecs, hit)
    }

    /// The embedding part's probabilities with the tool list cached.
    pub fn probs(
        &mut self,
        e: &StaticEmbedding,
        request: &str,
        tools: &[String],
    ) -> (Vec<f64>, bool) {
        let q = e.embed(request);
        let (t, hit) = self.get(e, tools);
        (e.probs_from_scores(&e.scores(&q, t)), hit)
    }
}

/// The model part's options for a tool list: the tools, then *none*.
pub fn model_options(tools: &[String]) -> Vec<String> {
    let mut o = tools.to_vec();
    o.push(crate::route::NONE_OPTION.to_string());
    o
}

/// A numerically stable softmax in f64.
pub fn softmax(z: &[f64]) -> Vec<f64> {
    let m = z.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let e: Vec<f64> = z.iter().map(|x| (x - m).exp()).collect();
    let s: f64 = e.iter().sum();
    e.iter().map(|x| x / s).collect()
}

/// The fusion: `p(i) ∝ (p_model(i) · p_embedding(i))^β`, each floored at [`FLOOR`].
pub fn fuse(model: &[f64], embedding: &[f64], beta: f64) -> Vec<f64> {
    let logs: Vec<f64> = model
        .iter()
        .zip(embedding)
        .map(|(a, b)| beta * (a.max(FLOOR).ln() + b.max(FLOOR).ln()))
        .collect();
    softmax(&logs)
}

/// The first index of the largest value.
pub fn argmax(p: &[f64]) -> usize {
    let mut best = 0;
    for (i, v) in p.iter().enumerate() {
        if *v > p[best] {
            best = i;
        }
    }
    best
}

/// The router's answer on fused probabilities over the tools and *none* (last).
#[derive(Clone, Debug, PartialEq)]
pub struct RouteAnswer {
    /// The index of the top option (`n_tools` for *none*).
    pub index: usize,
    /// The top probability.
    pub top: f64,
    /// The top option is *none*.
    pub none: bool,
    /// The top probability is below τ_ask3: ask the user.
    pub ask: bool,
}

/// The answer and the ask decision at `tau_ask`.
pub fn answer(fused: &[f64], tau_ask: f64) -> RouteAnswer {
    let index = argmax(fused);
    let top = fused.get(index).copied().unwrap_or(0.0);
    RouteAnswer {
        index,
        top,
        none: index + 1 == fused.len(),
        ask: top < tau_ask,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A tiny embedding: a vocabulary of a few words, an ASCII-only normaliser (upper case to
    /// lower case, `é` to `e`, U+200B deleted, U+4E00 padded), dimension 4.
    pub(crate) fn tiny() -> StaticEmbedding {
        let vocab: Vec<String> = [
            "[PAD]", "[UNK]", "[CLS]", "[SEP]", "turn", "on", "the", "light", "##s", "cafe", "!",
            "un", "##aff", "##able", "一", "weather",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        let n = vocab.len();
        let dim = 4;
        let mut codes = Vec::new();
        let mut scales = Vec::new();
        for i in 0..n {
            // codes (i % 7, -(i % 5), 1, -7) at scale 0.5 + i / 8
            let c = [(i % 7) as i8, -((i % 5) as i8), 1, -7];
            codes.push(((c[0] as u8) & 0xF) | (((c[1] as u8) & 0xF) << 4));
            codes.push(((c[2] as u8) & 0xF) | (((c[3] as u8) & 0xF) << 4));
            scales.push(f32_to_f16(0.5 + i as f32 / 8.0));
        }
        let norm = NormTable {
            starts: vec![
                0, 0x41, 0x5B, 0xE9, 0xEA, 0x200B, 0x200C, 0x4E00, 0x4E01, 0xAC00, 0xD7A4,
            ],
            ops: vec![
                op::ID,
                op::DELTA,
                op::ID,
                op::STR,
                op::ID,
                op::DEL,
                op::ID,
                op::PAD,
                op::ID,
                op::HANGUL,
                op::ID,
            ],
            args: vec![0, 32, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            strings: vec!["e".into()],
        };
        let spec = EmbedSpec {
            dim,
            codes,
            scales,
            vocab,
            unk_id: 1,
            continuing_prefix: "##".into(),
            max_chars_per_word: 10,
            max_length: 6,
            normalize: true,
            added: vec![
                ("[CLS]".into(), 2),
                ("[SEP]".into(), 3),
                ("[UNK]".into(), 1),
            ],
            norm,
            space: vec![(0x20, 0x20)],
            punct: vec![(0x21, 0x2F), (0x3A, 0x40)],
            theta: 0.4,
            temperature: 0.07,
            beta: 0.81,
            tau_ask: 0.67,
            meta: vec![("general.name".into(), OwnedValue::from("tiny"))],
        };
        StaticEmbedding::from_gguf(&spec.to_gguf().unwrap()).unwrap()
    }

    #[test]
    fn normaliser_runs() {
        let e = tiny();
        assert_eq!(e.normalize_text("Turn ON"), "turn on");
        assert_eq!(e.normalize_text("Café\u{200B}!"), "cafe!");
        assert_eq!(e.normalize_text("a一b"), "a 一 b");
        // 각 = U+AC01: ᄀ ᅡ ᆨ
        assert_eq!(e.normalize_text("각"), "\u{1100}\u{1161}\u{11A8}");
        assert_eq!(e.normalize_text("가"), "\u{1100}\u{1161}");
    }

    #[test]
    fn tokens_like_model2vec() {
        let e = tiny();
        assert_eq!(e.token_ids("Turn on the lights!"), vec![4, 5, 6, 7, 8, 10]);
        // the whole word is [UNK] (removed) when a piece is missing
        assert_eq!(e.token_ids("unaffable unaffx"), vec![11, 12, 13]);
        // words longer than max_chars_per_word are [UNK]
        assert_eq!(e.token_ids("lightslightsx on"), vec![5]);
        // added tokens are matched in the raw text, before the normaliser
        assert_eq!(e.token_ids("[CLS]turn[SEP] [cls]"), vec![2, 4, 3]);
        assert_eq!(e.token_ids("[UNK] on"), vec![5]);
        // CJK padding makes a word of its own
        assert_eq!(e.token_ids("turn一on"), vec![4, 14, 5]);
        // truncated to max_length (6) before [UNK] is removed
        assert_eq!(e.token_ids("on xx on on on on on on"), vec![5, 5, 5, 5, 5]);
        assert!(e.token_ids("").is_empty());
        assert!(e.token_ids("   ").is_empty());
    }

    #[test]
    fn packed_vectors_and_mean() {
        let e = tiny();
        let mut v = [0f32; 4];
        e.token_vector(4, &mut v);
        let s = f16_to_f32(f32_to_f16(0.5 + 4.0 / 8.0));
        assert_eq!(v, [4.0 * s, -4.0 * s, s, -7.0 * s]);
        let x = e.embed("turn on");
        let n: f32 = x.iter().map(|a| a * a).sum::<f32>().sqrt();
        assert!((n - 1.0).abs() < 1e-6);
        assert_eq!(e.embed("xyz"), vec![0.0; 4]);
        // no token: every cosine is 0, *none* at theta
        let p = e.probs("xyz", &["turn on", "weather"]);
        assert!((p[0] - p[1]).abs() < 1e-15 && p[2] > p[0]);
        assert!((p.iter().sum::<f64>() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn numpy_pairwise_sum() {
        let x: Vec<f32> = (0..64).map(|i| 1.0 + i as f32 * 1e-3).collect();
        let mut r = [0f32; 8];
        for (i, v) in x.iter().enumerate() {
            r[i % 8] += v;
        }
        let want = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        assert_eq!(np_sum_f32(&x), want);
        assert_eq!(np_sum_f32(&[1.0, 2.0]), 3.0);
    }

    #[test]
    fn fusion_and_answer() {
        let m = [0.7, 0.2, 0.1];
        let b = [0.5, 0.4, 0.1];
        let f = fuse(&m, &b, 1.0);
        let z: f64 = m.iter().zip(&b).map(|(x, y)| x * y).sum();
        for i in 0..3 {
            assert!((f[i] - m[i] * b[i] / z).abs() < 1e-12);
        }
        // beta changes no answer
        assert_eq!(argmax(&fuse(&m, &b, 0.3)), argmax(&fuse(&m, &b, 3.0)));
        // a zero probability is floored, not -inf
        let g = fuse(&[1.0, 0.0], &[0.5, 0.5], 0.81);
        assert!(g.iter().all(|x| x.is_finite()) && g[0] > 0.999);
        let a = answer(&[0.1, 0.3, 0.6], 0.67);
        assert_eq!(
            a,
            RouteAnswer {
                index: 2,
                top: 0.6,
                none: true,
                ask: true
            }
        );
        let a = answer(&[0.8, 0.1, 0.1], 0.67);
        assert!(!a.ask && !a.none && a.index == 0);
    }
}
