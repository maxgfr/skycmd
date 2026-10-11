//! GPT-2 style byte-level BPE tokenizer that reproduces Hugging Face `tokenizers` encodings
//! for ByteLevel BPE models.
//!
//! Pipeline (same order as `tokenizers`):
//! 1. added tokens with `normalized: false` (special tokens) are matched on the raw text
//!    (leftmost-longest, with `lstrip` / `rstrip` / `single_word`);
//! 2. the remaining pieces are normalized (optional NFC) and the added tokens with
//!    `normalized: true` are matched on them;
//! 3. every remaining piece goes through the ByteLevel pre-tokenizer (optional prefix space,
//!    GPT-2 split regex, bytes -> unicode) and the ranked BPE merges.

mod pretokenize;
mod unicode_tables;

use std::borrow::Cow;
use std::collections::{BinaryHeap, HashMap};

use sky_gguf::Gguf;
use unicode_normalization::UnicodeNormalization;

use crate::json::Json;
use crate::{Error, Result};

pub use pretokenize::{gpt2_split, is_letter, is_number};

/// GGUF token types (`llama_token_type`).
pub mod token_type {
    pub const NORMAL: i32 = 1;
    pub const UNKNOWN: i32 = 2;
    pub const CONTROL: i32 = 3;
    pub const USER_DEFINED: i32 = 4;
    pub const UNUSED: i32 = 5;
    pub const BYTE: i32 = 6;
}

/// Extra GGUF keys written by `sky-convert` so that a GGUF tokenizer behaves exactly like the
/// original `tokenizer.json` (llama.cpp ignores them).
pub mod gguf_keys {
    /// "nfc" if the HF normalizer is NFC.
    pub const NORMALIZER: &str = "skycmd.tokenizer.normalizer";
    /// Ids of the added tokens that strip the whitespace on their left (`lstrip`).
    pub const LSTRIP: &str = "skycmd.tokenizer.lstrip";
    /// Ids of the added tokens that strip the whitespace on their right (`rstrip`).
    pub const RSTRIP: &str = "skycmd.tokenizer.rstrip";
    /// BPE `ignore_merges`.
    pub const IGNORE_MERGES: &str = "skycmd.tokenizer.ignore_merges";
    /// ByteLevel `use_regex` (default true).
    pub const USE_REGEX: &str = "skycmd.tokenizer.use_regex";
}

/// An added token (`added_tokens` of `tokenizer.json`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AddedToken {
    pub id: u32,
    pub content: String,
    pub special: bool,
    pub normalized: bool,
    pub lstrip: bool,
    pub rstrip: bool,
    pub single_word: bool,
}

/// Ids of the special tokens used by the decision prompt.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SpecialTokens {
    pub cls: Option<u32>,
    pub sep: Option<u32>,
    pub mask: Option<u32>,
    pub pad: Option<u32>,
    pub unk: Option<u32>,
}

/// Leftmost-longest matcher over a small set of literal patterns.
#[derive(Clone, Debug, Default)]
struct Matcher {
    /// For each first byte: (pattern, added token index), longest first.
    by_first: Vec<Vec<(Box<[u8]>, usize)>>,
    empty: bool,
}

impl Matcher {
    fn new(patterns: Vec<(String, usize)>) -> Self {
        let mut by_first: Vec<Vec<(Box<[u8]>, usize)>> = vec![Vec::new(); 256];
        let empty = patterns.is_empty();
        for (p, idx) in patterns {
            if p.is_empty() {
                continue;
            }
            let b = p.into_bytes().into_boxed_slice();
            by_first[b[0] as usize].push((b, idx));
        }
        for v in &mut by_first {
            // longest first; equal lengths keep insertion order (first added wins, as aho-corasick)
            v.sort_by_key(|p| std::cmp::Reverse(p.0.len()));
        }
        Matcher { by_first, empty }
    }

    /// Leftmost-longest match at or after `from`: (start, end, token index).
    fn find(&self, s: &[u8], from: usize) -> Option<(usize, usize, usize)> {
        if self.empty {
            return None;
        }
        for i in from..s.len() {
            for (p, idx) in &self.by_first[s[i] as usize] {
                if s[i..].starts_with(p) {
                    return Some((i, i + p.len(), *idx));
                }
            }
        }
        None
    }
}

/// Cache of BPE results per pre-tokenized word (useful when the same words come back every frame).
#[derive(Clone, Debug, Default)]
pub struct BpeCache {
    map: HashMap<Box<[u8]>, Box<[u32]>>,
    capacity: usize,
}

impl BpeCache {
    pub fn new(capacity: usize) -> Self {
        BpeCache {
            map: HashMap::new(),
            capacity,
        }
    }

    pub fn len(&self) -> usize {
        self.map.len()
    }

    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }
}

/// Byte-level BPE tokenizer.
#[derive(Clone, Debug)]
pub struct Tokenizer {
    vocab: HashMap<String, u32>,
    /// Token strings by id (byte-level form for model tokens, raw content for added tokens).
    id_to_token: Vec<String>,
    /// (left id << 32 | right id) -> (rank, merged id)
    merges: HashMap<u64, (u32, u32)>,
    /// Token id of the single-byte symbol of each byte (u32::MAX if absent).
    byte_ids: [u32; 256],
    added: Vec<AddedToken>,
    raw_matcher: Matcher,
    norm_matcher: Matcher,
    /// Content -> index into `added`.
    added_index: HashMap<String, usize>,
    nfc: bool,
    add_prefix_space: bool,
    use_regex: bool,
    ignore_merges: bool,
    special: SpecialTokens,
}

/// GPT-2 `bytes_to_unicode`: the printable character that stands for each byte.
pub fn byte_to_char(b: u8) -> char {
    BYTE_TO_CHAR[b as usize]
}

static BYTE_TO_CHAR: [char; 256] = build_byte_to_char();

const fn build_byte_to_char() -> [char; 256] {
    let mut out = ['\0'; 256];
    let mut n = 0u32;
    let mut b = 0usize;
    while b < 256 {
        let printable =
            (b >= 0x21 && b <= 0x7e) || (b >= 0xa1 && b <= 0xac) || (b >= 0xae && b <= 0xff);
        let cp = if printable {
            b as u32
        } else {
            n += 1;
            255 + n
        };
        out[b] = match char::from_u32(cp) {
            Some(c) => c,
            None => '\0',
        };
        b += 1;
    }
    out
}

fn char_to_byte(c: char) -> Option<u8> {
    let cp = c as u32;
    if (0x21..=0x7e).contains(&cp) || (0xa1..=0xac).contains(&cp) || (0xae..=0xff).contains(&cp) {
        return Some(cp as u8);
    }
    BYTE_TO_CHAR.iter().position(|&x| x == c).map(|p| p as u8)
}

/// Encode raw bytes in the byte-level alphabet.
pub fn bytes_to_unicode(bytes: &[u8]) -> String {
    bytes.iter().map(|&b| byte_to_char(b)).collect()
}

const GPT2_PATTERN: &str =
    r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+";

fn merge_key(a: u32, b: u32) -> u64 {
    ((a as u64) << 32) | b as u64
}

#[derive(PartialEq, Eq)]
struct MergeCand {
    rank: u32,
    pos: usize,
    new_id: u32,
}

impl Ord for MergeCand {
    fn cmp(&self, o: &Self) -> std::cmp::Ordering {
        // BinaryHeap is a max-heap: lowest rank first, then leftmost position
        o.rank.cmp(&self.rank).then_with(|| o.pos.cmp(&self.pos))
    }
}

impl PartialOrd for MergeCand {
    fn partial_cmp(&self, o: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(o))
    }
}

#[derive(Clone, Copy)]
struct Symbol {
    id: u32,
    prev: isize,
    next: isize,
    alive: bool,
}

struct Options {
    nfc: bool,
    add_prefix_space: bool,
    use_regex: bool,
    ignore_merges: bool,
}

impl Tokenizer {
    fn build(
        vocab: HashMap<String, u32>,
        merges: &[(String, String)],
        added: Vec<AddedToken>,
        opts: Options,
        special_names: [&str; 5],
    ) -> Result<Self> {
        const MAX_ID: u32 = 1 << 24;
        if let Some(bad) = vocab
            .values()
            .copied()
            .chain(added.iter().map(|a| a.id))
            .find(|&id| id >= MAX_ID)
        {
            return Err(Error::Format(format!("token id {bad} is out of range")));
        }
        let n = vocab
            .values()
            .copied()
            .chain(added.iter().map(|a| a.id))
            .max()
            .map_or(0, |m| m as usize + 1);
        let mut id_to_token = vec![String::new(); n];
        for (s, &id) in &vocab {
            id_to_token[id as usize] = s.clone();
        }
        for a in &added {
            id_to_token[a.id as usize] = a.content.clone();
        }
        let mut merge_map = HashMap::with_capacity(merges.len());
        for (rank, (a, b)) in merges.iter().enumerate() {
            let (Some(&ia), Some(&ib)) = (vocab.get(a), vocab.get(b)) else {
                return Err(Error::Format(format!(
                    "merge ({a}, {b}) uses a token missing from the vocab"
                )));
            };
            let joined = format!("{a}{b}");
            let Some(&nid) = vocab.get(&joined) else {
                return Err(Error::Format(format!(
                    "merge result {joined} is missing from the vocab"
                )));
            };
            // with duplicate merges the first (lowest rank) wins, as in `tokenizers`
            merge_map
                .entry(merge_key(ia, ib))
                .or_insert((rank as u32, nid));
        }
        let mut byte_ids = [u32::MAX; 256];
        for (b, slot) in byte_ids.iter_mut().enumerate() {
            if let Some(&id) = vocab.get(byte_to_char(b as u8).to_string().as_str()) {
                *slot = id;
            }
        }
        let mut raw = Vec::new();
        let mut norm = Vec::new();
        let mut added_index = HashMap::new();
        for (i, a) in added.iter().enumerate() {
            added_index.entry(a.content.clone()).or_insert(i);
            if a.normalized {
                let content = if opts.nfc {
                    a.content.nfc().collect()
                } else {
                    a.content.clone()
                };
                norm.push((content, i));
            } else {
                raw.push((a.content.clone(), i));
            }
        }
        let mut tok = Tokenizer {
            vocab,
            id_to_token,
            merges: merge_map,
            byte_ids,
            added,
            raw_matcher: Matcher::new(raw),
            norm_matcher: Matcher::new(norm),
            added_index,
            nfc: opts.nfc,
            add_prefix_space: opts.add_prefix_space,
            use_regex: opts.use_regex,
            ignore_merges: opts.ignore_merges,
            special: SpecialTokens::default(),
        };
        let [cls, sep, mask, pad, unk] = special_names;
        tok.special = SpecialTokens {
            cls: tok.token_to_id(cls),
            sep: tok.token_to_id(sep),
            mask: tok.token_to_id(mask),
            pad: tok.token_to_id(pad),
            unk: tok.token_to_id(unk),
        };
        Ok(tok)
    }

    /// Load a Hugging Face `tokenizer.json` (BPE model with a ByteLevel pre-tokenizer).
    /// Special tokens are looked up by their usual names (`[CLS]`, `[SEP]`, `[MASK]`, `[PAD]`, `[UNK]`).
    pub fn from_tokenizer_json(json: &str) -> Result<Self> {
        Self::from_tokenizer_json_with_names(json, ["[CLS]", "[SEP]", "[MASK]", "[PAD]", "[UNK]"])
    }

    /// Same as [`Tokenizer::from_tokenizer_json`] with custom special token names
    /// (cls, sep, mask, pad, unk), e.g. read from `tokenizer_config.json`.
    pub fn from_tokenizer_json_with_names(json: &str, names: [&str; 5]) -> Result<Self> {
        let j = Json::parse(json)?;
        let model = j
            .get("model")
            .ok_or_else(|| Error::Format("tokenizer.json has no model".into()))?;
        let mtype = model.get("type").and_then(|t| t.as_str()).unwrap_or("BPE");
        if mtype != "BPE" {
            return Err(Error::Unsupported(format!("tokenizer model type {mtype}")));
        }
        for key in ["continuing_subword_prefix", "end_of_word_suffix"] {
            if let Some(v) = model.get(key).and_then(|v| v.as_str()) {
                if !v.is_empty() {
                    return Err(Error::Unsupported(format!("BPE {key}")));
                }
            }
        }
        if model.get("byte_fallback").and_then(|v| v.as_bool()) == Some(true) {
            return Err(Error::Unsupported("BPE byte_fallback".into()));
        }
        if model.get("dropout").is_some_and(|d| !d.is_null()) {
            return Err(Error::Unsupported("BPE dropout".into()));
        }
        let ignore_merges = model
            .get("ignore_merges")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        let vocab_obj = model
            .get("vocab")
            .and_then(|v| v.as_object())
            .ok_or_else(|| Error::Format("model.vocab is not an object".into()))?;
        let mut vocab = HashMap::with_capacity(vocab_obj.len());
        for (k, v) in vocab_obj {
            let id = v
                .as_u64()
                .ok_or_else(|| Error::Format(format!("bad id for token {k}")))?;
            vocab.insert(k.clone(), id as u32);
        }
        let mut merges = Vec::new();
        for m in model
            .get("merges")
            .and_then(|m| m.as_array())
            .unwrap_or(&[])
        {
            let pair = match m {
                Json::String(s) => {
                    let (a, b) = s
                        .split_once(' ')
                        .ok_or_else(|| Error::Format(format!("merge {s:?} has no space")))?;
                    (a.to_string(), b.to_string())
                }
                Json::Array(a) if a.len() == 2 => match (a[0].as_str(), a[1].as_str()) {
                    (Some(x), Some(y)) => (x.to_string(), y.to_string()),
                    _ => return Err(Error::Format("merge pair is not two strings".into())),
                },
                _ => return Err(Error::Format("bad merge entry".into())),
            };
            merges.push(pair);
        }

        let nfc = match j.get("normalizer") {
            None | Some(Json::Null) => false,
            Some(n) => parse_normalizer(n)?,
        };
        let (add_prefix_space, use_regex) = match j.get("pre_tokenizer") {
            None | Some(Json::Null) => {
                return Err(Error::Unsupported(
                    "tokenizer without a pre-tokenizer".into(),
                ))
            }
            Some(p) => parse_pre_tokenizer(p)?,
        };

        let mut added = Vec::new();
        for a in j
            .get("added_tokens")
            .and_then(|a| a.as_array())
            .unwrap_or(&[])
        {
            let get_b = |k: &str, d: bool| a.get(k).and_then(|v| v.as_bool()).unwrap_or(d);
            let content = a
                .get("content")
                .and_then(|c| c.as_str())
                .unwrap_or_default()
                .to_string();
            let id = a
                .get("id")
                .and_then(|v| v.as_u64())
                .ok_or_else(|| Error::Format("added token without id".into()))?;
            let special = get_b("special", false);
            added.push(AddedToken {
                id: id as u32,
                content,
                special,
                normalized: get_b("normalized", !special),
                lstrip: get_b("lstrip", false),
                rstrip: get_b("rstrip", false),
                single_word: get_b("single_word", false),
            });
        }
        Self::build(
            vocab,
            &merges,
            added,
            Options {
                nfc,
                add_prefix_space,
                use_regex,
                ignore_merges,
            },
            names,
        )
    }

    /// Load the tokenizer stored in a GGUF file (`tokenizer.ggml.model = "gpt2"`).
    pub fn from_gguf(g: &Gguf<'_>) -> Result<Self> {
        let model = g.get_str("tokenizer.ggml.model").unwrap_or("gpt2");
        if model != "gpt2" {
            return Err(Error::Unsupported(format!("GGUF tokenizer model {model}")));
        }
        let tokens = g
            .get_array("tokenizer.ggml.tokens")
            .and_then(|a| a.strings())
            .ok_or_else(|| Error::Missing("tokenizer.ggml.tokens".into()))?;
        let types: Vec<i64> = match g.get_array("tokenizer.ggml.token_type") {
            Some(a) => a
                .to_i64_vec()
                .ok_or_else(|| Error::Format("bad tokenizer.ggml.token_type".into()))?,
            None => vec![token_type::NORMAL as i64; tokens.len()],
        };
        if types.len() != tokens.len() {
            return Err(Error::Format(
                "token_type and tokens differ in length".into(),
            ));
        }
        let merges: Vec<(String, String)> = match g.get_array("tokenizer.ggml.merges") {
            Some(a) => a
                .strings()
                .ok_or_else(|| Error::Format("bad tokenizer.ggml.merges".into()))?
                .into_iter()
                .map(|s| {
                    s.split_once(' ')
                        .map(|(a, b)| (a.to_string(), b.to_string()))
                        .ok_or_else(|| Error::Format(format!("merge {s:?} has no space")))
                })
                .collect::<Result<_>>()?,
            None => Vec::new(),
        };
        let id_list = |key: &str| -> Vec<u32> {
            g.get_array(key)
                .and_then(|a| a.to_i64_vec())
                .map(|v| v.into_iter().map(|x| x as u32).collect())
                .unwrap_or_default()
        };
        let lstrip = id_list(gguf_keys::LSTRIP);
        let rstrip = id_list(gguf_keys::RSTRIP);

        let mut vocab = HashMap::with_capacity(tokens.len());
        let mut added = Vec::new();
        for (i, (&t, &ty)) in tokens.iter().zip(&types).enumerate() {
            let id = i as u32;
            match ty as i32 {
                token_type::CONTROL | token_type::USER_DEFINED => {
                    let special = ty as i32 == token_type::CONTROL;
                    added.push(AddedToken {
                        id,
                        content: t.to_string(),
                        special,
                        normalized: !special,
                        lstrip: lstrip.contains(&id),
                        rstrip: rstrip.contains(&id),
                        single_word: false,
                    });
                }
                token_type::UNUSED => {}
                _ => {
                    vocab.insert(t.to_string(), id);
                }
            }
        }
        // added tokens keep their raw content in GGUF; BPE merges may still produce them in
        // byte-level form (and ASCII contents without spaces are identical in both forms)
        for a in &added {
            vocab
                .entry(bytes_to_unicode(a.content.as_bytes()))
                .or_insert(a.id);
        }
        let nfc = g
            .get_str(gguf_keys::NORMALIZER)
            .is_some_and(|n| n.eq_ignore_ascii_case("nfc"));
        let opts = Options {
            nfc,
            add_prefix_space: g
                .get_bool("tokenizer.ggml.add_space_prefix")
                .unwrap_or(false),
            use_regex: g.get_bool(gguf_keys::USE_REGEX).unwrap_or(true),
            ignore_merges: g.get_bool(gguf_keys::IGNORE_MERGES).unwrap_or(false),
        };
        let name_of = |key: &str, default: &'static str| -> String {
            g.get_u64(key)
                .and_then(|id| tokens.get(id as usize))
                .map(|s| s.to_string())
                .unwrap_or_else(|| default.to_string())
        };
        let cls = g
            .get_u64("tokenizer.ggml.cls_token_id")
            .or_else(|| g.get_u64("tokenizer.ggml.bos_token_id"))
            .and_then(|id| tokens.get(id as usize))
            .map(|s| s.to_string())
            .unwrap_or_else(|| "[CLS]".to_string());
        let sep = name_of("tokenizer.ggml.seperator_token_id", "[SEP]");
        let mask = name_of("tokenizer.ggml.mask_token_id", "[MASK]");
        let pad = name_of("tokenizer.ggml.padding_token_id", "[PAD]");
        let unk = name_of("tokenizer.ggml.unknown_token_id", "[UNK]");
        Self::build(vocab, &merges, added, opts, [&cls, &sep, &mask, &pad, &unk])
    }

    /// Id of a token given its content (added tokens first, then the BPE vocab).
    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        if let Some(&i) = self.added_index.get(token) {
            return Some(self.added[i].id);
        }
        self.vocab.get(token).copied()
    }

    /// Token string of an id (byte-level form for model tokens).
    pub fn id_to_token(&self, id: u32) -> Option<&str> {
        self.id_to_token
            .get(id as usize)
            .map(|s| s.as_str())
            .filter(|s| !s.is_empty())
    }

    /// Number of ids (highest id + 1).
    pub fn vocab_size(&self) -> usize {
        self.id_to_token.len()
    }

    pub fn special(&self) -> SpecialTokens {
        self.special
    }

    pub fn added_tokens(&self) -> &[AddedToken] {
        &self.added
    }

    /// BPE merges in rank order, as (left, right) token strings.
    pub fn merges(&self) -> Vec<(String, String)> {
        let mut v: Vec<(u32, u32, u32)> = self
            .merges
            .iter()
            .map(|(&k, &(rank, _))| (rank, (k >> 32) as u32, k as u32))
            .collect();
        v.sort_unstable();
        v.into_iter()
            .map(|(_, a, b)| {
                (
                    self.id_to_token[a as usize].clone(),
                    self.id_to_token[b as usize].clone(),
                )
            })
            .collect()
    }

    /// Model vocabulary entries (byte-level token string -> id), without the added tokens.
    pub fn vocab(&self) -> &HashMap<String, u32> {
        &self.vocab
    }

    pub fn is_nfc(&self) -> bool {
        self.nfc
    }

    pub fn add_prefix_space(&self) -> bool {
        self.add_prefix_space
    }

    pub fn use_regex(&self) -> bool {
        self.use_regex
    }

    pub fn ignore_merges(&self) -> bool {
        self.ignore_merges
    }

    /// Encode text without adding special tokens (`add_special_tokens=False`).
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut out = Vec::new();
        self.encode_into(text, &mut out, None);
        out
    }

    /// Encode text, appending the ids to `out`, optionally caching the BPE of each word.
    pub fn encode_into(&self, text: &str, out: &mut Vec<u32>, mut cache: Option<&mut BpeCache>) {
        let mut push_piece = |piece: &str, out: &mut Vec<u32>| {
            let normalized: Cow<'_, str> = if self.nfc && !piece.is_ascii() {
                Cow::Owned(piece.nfc().collect())
            } else {
                Cow::Borrowed(piece)
            };
            self.split_added(&normalized, &self.norm_matcher, out, |p, out| {
                self.encode_plain(p, out, cache.as_deref_mut())
            });
        };
        self.split_added(text, &self.raw_matcher, out, |p, out| push_piece(p, out));
    }

    /// `AddedVocabulary::find_matches` + `split_with_indices`: emit added token ids and pass the
    /// text between them to `f`.
    fn split_added<F: FnMut(&str, &mut Vec<u32>)>(
        &self,
        s: &str,
        m: &Matcher,
        out: &mut Vec<u32>,
        mut f: F,
    ) {
        if s.is_empty() {
            return;
        }
        let b = s.as_bytes();
        let mut start_offset = 0usize; // end of the previous split
        let mut search = 0usize;
        while let Some((ms, me, idx)) = m.find(b, search) {
            search = me;
            let tok = &self.added[idx];
            let (mut start, mut stop) = (ms, me);
            if tok.single_word {
                let start_space = start == 0 || !ends_with_word(&s[..start]);
                let stop_space = stop == s.len() || !starts_with_word(&s[stop..]);
                if !start_space || !stop_space {
                    continue;
                }
            }
            if tok.lstrip {
                start = space_leftmost_at_end(&s[..start]).max(start_offset);
            }
            if tok.rstrip {
                stop += space_rightmost_at_start(&s[stop..]);
                search = stop;
            }
            if start_offset < start {
                f(&s[start_offset..start], out);
            }
            out.push(tok.id);
            start_offset = stop;
        }
        if start_offset < s.len() {
            f(&s[start_offset..], out);
        }
    }

    /// ByteLevel pre-tokenizer + BPE on a piece without added tokens.
    fn encode_plain(&self, piece: &str, out: &mut Vec<u32>, mut cache: Option<&mut BpeCache>) {
        if piece.is_empty() {
            return;
        }
        let owned;
        let piece = if self.add_prefix_space && !piece.starts_with(' ') {
            owned = format!(" {piece}");
            owned.as_str()
        } else {
            piece
        };
        if self.use_regex {
            for word in gpt2_split(piece) {
                self.bpe_word(word.as_bytes(), out, cache.as_deref_mut());
            }
        } else {
            self.bpe_word(piece.as_bytes(), out, cache);
        }
    }

    fn bpe_word(&self, bytes: &[u8], out: &mut Vec<u32>, cache: Option<&mut BpeCache>) {
        if let Some(c) = cache.as_ref() {
            if let Some(ids) = c.map.get(bytes) {
                out.extend_from_slice(ids);
                return;
            }
        }
        let start = out.len();
        self.bpe_word_uncached(bytes, out);
        if let Some(c) = cache {
            if c.capacity > 0 {
                if c.map.len() >= c.capacity {
                    c.map.clear();
                }
                c.map.insert(bytes.into(), out[start..].into());
            }
        }
    }

    fn bpe_word_uncached(&self, bytes: &[u8], out: &mut Vec<u32>) {
        if self.ignore_merges {
            if let Some(&id) = self.vocab.get(&bytes_to_unicode(bytes)) {
                out.push(id);
                return;
            }
        }
        let mut syms: Vec<Symbol> = Vec::with_capacity(bytes.len());
        for &b in bytes {
            let id = self.byte_ids[b as usize];
            if id == u32::MAX {
                continue; // no unk token: unknown symbols are dropped, as in `tokenizers`
            }
            let i = syms.len() as isize;
            syms.push(Symbol {
                id,
                prev: i - 1,
                next: i + 1,
                alive: true,
            });
        }
        if let Some(last) = syms.last_mut() {
            last.next = -1;
        }
        if syms.len() > 1 {
            let mut heap = BinaryHeap::with_capacity(syms.len());
            for i in 0..syms.len() - 1 {
                if let Some(&(rank, new_id)) =
                    self.merges.get(&merge_key(syms[i].id, syms[i + 1].id))
                {
                    heap.push(MergeCand {
                        rank,
                        pos: i,
                        new_id,
                    });
                }
            }
            while let Some(top) = heap.pop() {
                let cur = syms[top.pos];
                if !cur.alive || cur.next < 0 {
                    continue;
                }
                let next_pos = cur.next as usize;
                let right = syms[next_pos];
                match self.merges.get(&merge_key(cur.id, right.id)) {
                    Some(&(_, nid)) if nid == top.new_id => {}
                    _ => continue,
                }
                syms[top.pos].id = top.new_id;
                syms[top.pos].next = right.next;
                syms[next_pos].alive = false;
                if right.next >= 0 {
                    syms[right.next as usize].prev = top.pos as isize;
                }
                let cur = syms[top.pos];
                if cur.prev >= 0 {
                    let p = cur.prev as usize;
                    if let Some(&(rank, new_id)) = self.merges.get(&merge_key(syms[p].id, cur.id)) {
                        heap.push(MergeCand {
                            rank,
                            pos: p,
                            new_id,
                        });
                    }
                }
                if cur.next >= 0 {
                    let nx = cur.next as usize;
                    if let Some(&(rank, new_id)) = self.merges.get(&merge_key(cur.id, syms[nx].id))
                    {
                        heap.push(MergeCand {
                            rank,
                            pos: top.pos,
                            new_id,
                        });
                    }
                }
            }
        }
        out.extend(syms.iter().filter(|s| s.alive).map(|s| s.id));
    }

    /// Decode ids back to text (byte-level decoding; added tokens are emitted verbatim).
    pub fn decode(&self, ids: &[u32], skip_special: bool) -> String {
        let mut bytes = Vec::new();
        for &id in ids {
            let Some(tok) = self.id_to_token(id) else {
                continue;
            };
            if let Some(&i) = self.added_index.get(tok) {
                if self.added[i].id == id {
                    if !(skip_special && self.added[i].special) {
                        bytes.extend_from_slice(tok.as_bytes());
                    }
                    continue;
                }
            }
            for c in tok.chars() {
                match char_to_byte(c) {
                    Some(b) => bytes.push(b),
                    None => bytes.extend_from_slice(c.to_string().as_bytes()),
                }
            }
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }
}

fn parse_normalizer(n: &Json) -> Result<bool> {
    match n.get("type").and_then(|t| t.as_str()) {
        Some("NFC") => Ok(true),
        Some("Sequence") => {
            let mut nfc = false;
            for item in n
                .get("normalizers")
                .and_then(|v| v.as_array())
                .unwrap_or(&[])
            {
                nfc |= parse_normalizer(item)?;
            }
            Ok(nfc)
        }
        other => Err(Error::Unsupported(format!("normalizer {other:?}"))),
    }
}

/// Returns (add_prefix_space, use_regex).
fn parse_pre_tokenizer(p: &Json) -> Result<(bool, bool)> {
    match p.get("type").and_then(|t| t.as_str()) {
        Some("ByteLevel") => Ok((
            p.get("add_prefix_space")
                .and_then(|v| v.as_bool())
                .unwrap_or(false),
            p.get("use_regex").and_then(|v| v.as_bool()).unwrap_or(true),
        )),
        Some("Sequence") => {
            // accepted: [ByteLevel] or [Split(GPT-2 pattern, Isolated), ByteLevel(use_regex=false)]
            let items = p
                .get("pretokenizers")
                .and_then(|v| v.as_array())
                .unwrap_or(&[]);
            match items {
                [one] => parse_pre_tokenizer(one),
                [split, bl] if split.get("type").and_then(|t| t.as_str()) == Some("Split") => {
                    let pattern = split
                        .get("pattern")
                        .and_then(|pt| pt.get("Regex").or_else(|| pt.get("String")));
                    let behavior = split.get("behavior").and_then(|b| b.as_str());
                    let (prefix, regex) = parse_pre_tokenizer(bl)?;
                    if pattern.and_then(|v| v.as_str()) == Some(GPT2_PATTERN)
                        && behavior == Some("Isolated")
                        && !regex
                    {
                        Ok((prefix, true))
                    } else {
                        Err(Error::Unsupported(
                            "Split pre-tokenizer with a non GPT-2 pattern".into(),
                        ))
                    }
                }
                _ => Err(Error::Unsupported("pre-tokenizer sequence".into())),
            }
        }
        other => Err(Error::Unsupported(format!("pre-tokenizer {other:?}"))),
    }
}

fn is_word_char(c: char) -> bool {
    c.is_alphanumeric() || c == '_'
}

fn ends_with_word(s: &str) -> bool {
    s.chars().next_back().is_some_and(is_word_char)
}

fn starts_with_word(s: &str) -> bool {
    s.chars().next().is_some_and(is_word_char)
}

/// Byte index where the trailing whitespace of `s` starts.
fn space_leftmost_at_end(s: &str) -> usize {
    let mut idx = s.len();
    for (i, c) in s.char_indices().rev() {
        if !c.is_whitespace() {
            break;
        }
        idx = i;
    }
    idx
}

/// Byte length of the leading whitespace of `s`.
fn space_rightmost_at_start(s: &str) -> usize {
    s.char_indices()
        .find(|(_, c)| !c.is_whitespace())
        .map_or(s.len(), |(i, _)| i)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn byte_map_is_a_bijection() {
        let mut seen = std::collections::HashSet::new();
        for b in 0..=255u8 {
            let c = byte_to_char(b);
            assert!(seen.insert(c));
            assert_eq!(char_to_byte(c), Some(b));
        }
        assert_eq!(byte_to_char(b' '), 'Ġ');
        assert_eq!(byte_to_char(b'\n'), 'Ċ');
    }

    #[test]
    fn whitespace_helpers() {
        assert_eq!(space_leftmost_at_end("ab  "), 2);
        assert_eq!(space_leftmost_at_end("ab"), 2);
        assert_eq!(space_leftmost_at_end("   "), 0);
        assert_eq!(space_rightmost_at_start("  x"), 2);
        assert_eq!(space_rightmost_at_start("  "), 2);
    }
}
