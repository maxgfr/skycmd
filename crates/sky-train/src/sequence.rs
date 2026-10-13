//! Decision records and the Laya prompt layout, identical to
//! `rl_common.build_sequence`:
//! `[CLS] "{t} question: {ins}" [SEP] ([MASK] " {opt_i}")* [SEP] {state} [SEP]`.

use std::path::Path;

use anyhow::{anyhow, bail, Result};
use serde::{Deserialize, Serialize};

use crate::config::SpecialTokens;

/// Question type names, indexed by type id.
pub const QTYPES: [&str; 3] = ["choice", "score", "noul"];
/// Text of the mask token, replaced by a space in every text piece.
pub const MASK_TEXT: &str = "[MASK]";
/// Maximum number of tokens of one option text.
pub const OPTION_MAX_TOKENS: usize = 48;

/// Type id of a question type name.
pub fn qtype_id(t: &str) -> Option<usize> {
    QTYPES.iter().position(|q| *q == t)
}

/// One decision record (JSONL, produced by sky-data / sky-games).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DecisionRecord {
    pub src: String,
    pub t: String,
    pub ins: String,
    pub opts: Vec<String>,
    #[serde(default)]
    pub state: String,
    pub target: Vec<f32>,
    pub split: String,
}

/// Byte-level BPE tokenizer (HF `tokenizers`).
pub struct Tok {
    inner: tokenizers::Tokenizer,
}

impl Tok {
    /// A skycmd tokenizer (`[PAD]=0 [UNK]=1 [CLS]=2 [SEP]=3 [MASK]=4`).
    pub fn load(path: &Path) -> Result<Self> {
        Self::load_with(path, &SpecialTokens::default())
    }

    /// A tokenizer whose special tokens must have the ids of `sp`.
    pub fn load_with(path: &Path, sp: &SpecialTokens) -> Result<Self> {
        let inner = tokenizers::Tokenizer::from_file(path)
            .map_err(|e| anyhow!("loading tokenizer {}: {e}", path.display()))?;
        let tok = Self { inner };
        for (name, id) in sp.pairs() {
            if tok.inner.token_to_id(name) != Some(id) {
                bail!("tokenizer {}: {name} must have id {id}", path.display());
            }
        }
        Ok(tok)
    }

    /// Ids of `text` without special tokens (`add_special_tokens=False`).
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let enc = self
            .inner
            .encode(text, false)
            .map_err(|e| anyhow!("tokenizing: {e}"))?;
        Ok(enc.get_ids().to_vec())
    }

    /// Batch version of [`Tok::encode`] (parallel).
    pub fn encode_batch(&self, texts: Vec<String>) -> Result<Vec<Vec<u32>>> {
        let encs = self
            .inner
            .encode_batch(texts, false)
            .map_err(|e| anyhow!("tokenizing: {e}"))?;
        Ok(encs.into_iter().map(|e| e.get_ids().to_vec()).collect())
    }

    /// Vocabulary size including added tokens.
    pub fn vocab_size(&self) -> usize {
        self.inner.get_vocab_size(true)
    }

    /// Id of a token string.
    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        self.inner.token_to_id(token)
    }

    /// Token string of an id.
    pub fn id_to_token(&self, id: u32) -> Option<String> {
        self.inner.id_to_token(id)
    }

    /// Checks that `[NUM]` has the fixed s2 id and is matched as a special token.
    pub fn check_num(&self) -> Result<()> {
        let id = crate::num::NUM_ID;
        if self.token_to_id(crate::num::NUM_TOKEN) != Some(id) {
            bail!(
                "not an s2 tokenizer: {} must have id {id}",
                crate::num::NUM_TOKEN
            );
        }
        if self
            .encode(&format!("a{}b", crate::num::NUM_TOKEN))?
            .iter()
            .filter(|&&t| t == id)
            .count()
            != 1
        {
            bail!(
                "not an s2 tokenizer: {} is not matched as a special token",
                crate::num::NUM_TOKEN
            );
        }
        Ok(())
    }
}

/// The tokenized pieces of a record, independent of the option order.
#[derive(Debug, Clone, PartialEq)]
pub struct Pieces {
    /// `"{t} question: {ins}"` (untruncated).
    pub head: Vec<u32>,
    /// `" " + opt` per option, already truncated to 48 tokens (no marker).
    pub opts: Vec<Vec<u32>>,
    /// State text (untruncated).
    pub state: Vec<u32>,
}

/// The texts that get tokenized: head, options, state.
pub fn piece_texts(rec: &DecisionRecord) -> (String, Vec<String>, String) {
    piece_texts_with(rec, MASK_TEXT)
}

/// [`piece_texts`] for a tokenizer whose mask token is `mask_text`.
pub fn piece_texts_with(rec: &DecisionRecord, mask_text: &str) -> (String, Vec<String>, String) {
    let ins = rec.ins.replace(mask_text, " ");
    let head = format!("{} question: {}", rec.t, ins);
    let opts = rec
        .opts
        .iter()
        .map(|o| format!(" {}", o.replace(mask_text, " ")))
        .collect();
    let state = rec.state.replace(mask_text, " ");
    (head, opts, state)
}

/// Tokenizes the pieces of a record.
pub fn pieces(tok: &Tok, rec: &DecisionRecord) -> Result<Pieces> {
    let (head, opts, state) = piece_texts(rec);
    let mut texts = vec![head];
    texts.extend(opts);
    texts.push(state);
    let mut ids = tok.encode_batch(texts)?;
    let state = ids.pop().unwrap_or_default();
    let head = ids.remove(0);
    let opts = ids
        .into_iter()
        .map(|mut o| {
            o.truncate(OPTION_MAX_TOKENS);
            o
        })
        .collect();
    Ok(Pieces { head, opts, state })
}

/// Builds the input ids and marker positions for the given option order
/// (`rl_common.build_sequence` with `truncate_left=False`).
pub fn build_from_pieces(
    p: &Pieces,
    order: &[usize],
    max_len: usize,
    head_max_len: usize,
) -> (Vec<u32>, Vec<usize>) {
    build_from_pieces_with(p, order, max_len, head_max_len, &SpecialTokens::default())
}

/// [`build_from_pieces`] with the special tokens of `sp`.
pub fn build_from_pieces_with(
    p: &Pieces,
    order: &[usize],
    max_len: usize,
    head_max_len: usize,
    sp: &SpecialTokens,
) -> (Vec<u32>, Vec<usize>) {
    let mut opt_ids: Vec<Vec<u32>> = order
        .iter()
        .map(|&i| {
            let mut o = Vec::with_capacity(p.opts[i].len() + 1);
            o.push(sp.mask);
            o.extend_from_slice(&p.opts[i]);
            o
        })
        .collect();
    let total: usize = opt_ids.iter().map(Vec::len).sum();
    let mut opt_budget = head_max_len as i64 - total as i64;
    if opt_budget < 16 {
        // Too many / too long options: shrink every option text evenly.
        let per =
            4.max((head_max_len as i64 - 16).div_euclid(opt_ids.len().max(1) as i64)) as usize;
        for o in &mut opt_ids {
            o.truncate(per);
        }
        opt_budget = head_max_len as i64 - opt_ids.iter().map(Vec::len).sum::<usize>() as i64;
    }
    let head_keep = opt_budget.max(8) as usize;
    let mut ids = Vec::with_capacity(max_len);
    ids.push(sp.cls);
    ids.extend_from_slice(&p.head[..p.head.len().min(head_keep)]);
    ids.push(sp.sep);
    let mut markers = Vec::with_capacity(opt_ids.len());
    for o in &opt_ids {
        markers.push(ids.len());
        ids.extend_from_slice(o);
    }
    ids.push(sp.sep);
    let room = (max_len as i64 - ids.len() as i64 - 1).max(0) as usize;
    ids.extend_from_slice(&p.state[..p.state.len().min(room)]);
    ids.push(sp.sep);
    ids.truncate(max_len);
    markers.retain(|&m| m < max_len);
    (ids, markers)
}

/// Tokenizes and builds one sequence (identity order when `order` is `None`).
pub fn build_sequence(
    tok: &Tok,
    rec: &DecisionRecord,
    max_len: usize,
    head_max_len: usize,
    order: Option<&[usize]>,
) -> Result<(Vec<u32>, Vec<usize>)> {
    let p = pieces(tok, rec)?;
    let identity: Vec<usize> = (0..rec.opts.len()).collect();
    Ok(build_from_pieces(
        &p,
        order.unwrap_or(&identity),
        max_len,
        head_max_len,
    ))
}

/// Laya temperature bucket key (`rl_common.temp_bucket`), e.g. `"choice:3-5"`.
pub fn temp_bucket(qtype: usize, k: usize) -> String {
    let size = match k {
        0..=2 => "2",
        3..=5 => "3-5",
        6..=10 => "6-10",
        _ => "11+",
    };
    format!("{}:{}", QTYPES[qtype], size)
}

/// The same bucket in the GGUF / llama.cpp server form, e.g. `"choice.3_5"`.
pub fn gguf_bucket(laya_key: &str) -> String {
    laya_key
        .replace(':', ".")
        .replace('-', "_")
        .trim_end_matches('+')
        .to_string()
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::config::{CLS_ID, MASK_ID, SEP_ID};

    /// GPT-2 `bytes_to_unicode`.
    fn byte_chars() -> Vec<char> {
        let mut bs: Vec<u32> = (b'!' as u32..=b'~' as u32)
            .chain(0xA1..=0xAC)
            .chain(0xAE..=0xFF)
            .collect();
        let mut cs = bs.clone();
        let mut n = 0;
        for b in 0..256u32 {
            if !bs.contains(&b) {
                bs.push(b);
                cs.push(256 + n);
                n += 1;
            }
        }
        let mut out = vec!['\0'; 256];
        for (b, c) in bs.iter().zip(cs) {
            out[*b as usize] = char::from_u32(c).unwrap();
        }
        out
    }

    /// A byte-level BPE tokenizer.json with no merges: one token per byte,
    /// byte `b` has id `5 + b`. Specials take ids 0..5.
    pub fn byte_tokenizer_json() -> String {
        let chars = byte_chars();
        let mut vocab = serde_json::Map::new();
        for (i, s) in ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
            .iter()
            .enumerate()
        {
            vocab.insert(s.to_string(), i.into());
        }
        for (b, c) in chars.iter().enumerate() {
            vocab.insert(c.to_string(), (5 + b).into());
        }
        let added: Vec<_> = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
            .iter()
            .enumerate()
            .map(|(i, s)| {
                serde_json::json!({"id": i, "content": s, "single_word": false, "lstrip": false,
                    "rstrip": false, "normalized": false, "special": true})
            })
            .collect();
        serde_json::json!({
            "version": "1.0", "truncation": null, "padding": null, "added_tokens": added,
            "normalizer": null,
            "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true, "use_regex": true},
            "post_processor": null,
            "decoder": {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true, "use_regex": true},
            "model": {"type": "BPE", "dropout": null, "unk_token": null, "continuing_subword_prefix": null,
                "end_of_word_suffix": null, "fuse_unk": false, "byte_fallback": false, "ignore_merges": false,
                "vocab": vocab, "merges": []}
        })
        .to_string()
    }

    pub fn byte_tok() -> (tempfile::TempDir, Tok) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("tokenizer.json");
        std::fs::write(&path, byte_tokenizer_json()).unwrap();
        let tok = Tok::load(&path).unwrap();
        (dir, tok)
    }

    fn b(s: &str) -> Vec<u32> {
        s.bytes().map(|x| 5 + x as u32).collect()
    }

    fn rec(t: &str, ins: &str, opts: &[&str], state: &str) -> DecisionRecord {
        DecisionRecord {
            src: "test".into(),
            t: t.into(),
            ins: ins.into(),
            opts: opts.iter().map(|s| s.to_string()).collect(),
            state: state.into(),
            target: vec![1.0 / opts.len() as f32; opts.len()],
            split: "train".into(),
        }
    }

    #[test]
    fn byte_tokenizer_is_one_token_per_byte() {
        let (_d, tok) = byte_tok();
        assert_eq!(tok.encode("ab c").unwrap(), b("ab c"));
        assert_eq!(
            tok.encode("x[MASK]y").unwrap(),
            vec![5 + b'x' as u32, MASK_ID, 5 + b'y' as u32]
        );
    }

    #[test]
    fn simple_noul_sequence() {
        let (_d, tok) = byte_tok();
        let r = rec("noul", "ok?", &["false: no", "true: yes"], "s 1");
        let (ids, markers) = build_sequence(&tok, &r, 256, 96, None).unwrap();
        let mut want = vec![CLS_ID];
        want.extend(b("noul question: ok?"));
        want.push(SEP_ID);
        let m0 = want.len();
        want.push(MASK_ID);
        want.extend(b(" false: no"));
        let m1 = want.len();
        want.push(MASK_ID);
        want.extend(b(" true: yes"));
        want.push(SEP_ID);
        want.extend(b("s 1"));
        want.push(SEP_ID);
        assert_eq!(ids, want);
        assert_eq!(markers, vec![m0, m1]);
        assert_eq!(markers, vec![20, 31]);
    }

    #[test]
    fn option_order_and_mask_text() {
        let (_d, tok) = byte_tok();
        let r = rec("choice", "pick [MASK]", &["a", "bb", "c"], "");
        let (ids, markers) = build_sequence(&tok, &r, 256, 96, Some(&[2, 0, 1])).unwrap();
        let mut want = vec![CLS_ID];
        want.extend(b("choice question: pick  "));
        want.push(SEP_ID);
        for o in [" c", " a", " bb"] {
            want.push(MASK_ID);
            want.extend(b(o));
        }
        want.extend([SEP_ID, SEP_ID]);
        assert_eq!(ids, want);
        // [CLS] + 23 head bytes + [SEP] = 25, then 3, 3, 4 tokens per option.
        assert_eq!(markers, vec![25, 28, 31]);
    }

    #[test]
    fn truncation_rules() {
        let (_d, tok) = byte_tok();
        // Head budget: head_max_len 32 - options (1+2)+(1+2) = 26 -> head cut to 26 tokens.
        let long_ins = "x".repeat(100);
        let r = rec("noul", &long_ins, &["a", "b"], &"s".repeat(100));
        let (ids, markers) = build_sequence(&tok, &r, 48, 32, None).unwrap();
        assert_eq!(ids.len(), 48);
        assert_eq!(markers, vec![1 + 26 + 1, 1 + 26 + 1 + 3]);
        // State gets the room left: 48 - (1 + 26 + 1 + 6 + 1) - 1 = 12 tokens.
        assert_eq!(&ids[35..47], &b(&"s".repeat(12))[..]);
        assert_eq!(ids[47], SEP_ID);

        // Shrinking: 3 options of 20 bytes with head_max_len 32 -> budget < 16,
        // per = max(4, (32 - 16) / 3) = 5 tokens per option (marker included),
        // then the head keeps max(8, 32 - 15) = 17 tokens.
        let o = "o".repeat(20);
        let r = rec("choice", &long_ins, &[&o, &o, &o], "");
        let (ids, markers) = build_sequence(&tok, &r, 256, 32, None).unwrap();
        assert_eq!(markers, vec![19, 24, 29]);
        assert_eq!(ids[19], MASK_ID);
        assert_eq!(&ids[20..24], &b(" ooo")[..]);
        assert_eq!(ids.len(), 1 + 17 + 1 + 15 + 1 + 1);

        // Options are cut to 48 tokens before the budget is computed.
        let r = rec("choice", "q", &[&"z".repeat(80)], "");
        let p = pieces(&tok, &r).unwrap();
        assert_eq!(p.opts[0].len(), 48);
    }

    #[test]
    fn markers_beyond_max_len_are_dropped() {
        let (_d, tok) = byte_tok();
        let r = rec("choice", "q", &["aaaa", "bbbb", "cccc"], "");
        let (ids, markers) = build_sequence(&tok, &r, 20, 96, None).unwrap();
        assert_eq!(ids.len(), 20);
        // head: [CLS] "choice question: q" (18) [SEP] = 20 -> every marker is cut.
        assert!(markers.is_empty());
    }

    #[test]
    fn other_special_ids() {
        let (_d, tok) = byte_tok();
        let r = rec("noul", "ok?", &["false: no", "true: yes"], "s");
        let p = pieces(&tok, &r).unwrap();
        let sp = SpecialTokens {
            cls: 900,
            sep: 901,
            mask: 902,
            ..SpecialTokens::default()
        };
        let (ids, markers) = build_from_pieces_with(&p, &[0, 1], 256, 96, &sp);
        let (ids0, markers0) = build_from_pieces(&p, &[0, 1], 256, 96);
        assert_eq!(markers, markers0);
        let swap = |x: u32| match x {
            CLS_ID => 900,
            SEP_ID => 901,
            MASK_ID => 902,
            x => x,
        };
        assert_eq!(ids, ids0.into_iter().map(swap).collect::<Vec<_>>());
        // A tokenizer is checked against the ids it is loaded with.
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("t.json");
        std::fs::write(&path, byte_tokenizer_json()).unwrap();
        assert!(Tok::load_with(&path, &sp).is_err());
        let r = rec("choice", "a<mask>b", &["x"], "");
        assert_eq!(piece_texts_with(&r, "<mask>").0, "choice question: a b");
    }

    #[test]
    fn buckets() {
        assert_eq!(temp_bucket(2, 2), "noul:2");
        assert_eq!(temp_bucket(0, 4), "choice:3-5");
        assert_eq!(temp_bucket(1, 7), "score:6-10");
        assert_eq!(temp_bucket(0, 12), "choice:11+");
        assert_eq!(gguf_bucket("choice:3-5"), "choice.3_5");
        assert_eq!(gguf_bucket("choice:11+"), "choice.11");
        assert_eq!(gguf_bucket("noul:2"), "noul.2");
    }
}
