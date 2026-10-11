//! `train-tokenizer`: byte-level BPE (HF `tokenizers`) with fixed special-token ids,
//! trained on text exactly as the model will see it, plus a Rust port of
//! `rl_common.build_sequence` for sequence-length statistics.

use std::collections::BTreeMap;
use std::fs;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

use anyhow::{anyhow, bail, Result};
use serde_json::Value;
use tokenizers::models::bpe::{BpeTrainerBuilder, BPE};
use tokenizers::normalizers::NFC;
use tokenizers::pre_tokenizers::byte_level::ByteLevel;
use tokenizers::processors::template::TemplateProcessing;
use tokenizers::{
    AddedToken, DecoderWrapper, NormalizerWrapper, PostProcessorWrapper, PreTokenizerWrapper,
    Tokenizer, TokenizerBuilder,
};

use crate::num::{replace as replace_numbers, NUM_ID, NUM_TOKEN};
use crate::record::DecisionRecord;
use crate::rng::{hash64, Rng};

/// Special tokens in id order: `[PAD]=0 [UNK]=1 [CLS]=2 [SEP]=3 [MASK]=4`.
pub const SPECIALS: [&str; 5] = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"];
/// Special tokens of the s2 tokenizers (SPEC-S2 §2): the s1 ones plus `[NUM]=5`.
pub const SPECIALS_S2: [&str; 6] = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", NUM_TOKEN];
pub const PAD_ID: u32 = 0;
pub const CLS_ID: u32 = 2;
pub const SEP_ID: u32 = 3;
pub const MASK_ID: u32 = 4;

/// Default sequence budgets of our models (SPEC).
pub const MAX_LEN: usize = 256;
pub const HEAD_MAX_LEN: usize = 96;

/// Training-time segments of a decision record, i.e. the strings that
/// `build_sequence` tokenizes separately: head, each `" " + option`, state.
pub fn decision_segments(r: &DecisionRecord) -> Vec<String> {
    let mut out = Vec::with_capacity(r.opts.len() + 2);
    out.push(format!(
        "{} question: {}",
        r.t,
        r.ins.replace("[MASK]", " ")
    ));
    for o in &r.opts {
        out.push(format!(" {}", o.replace("[MASK]", " ")));
    }
    if !r.state.is_empty() {
        out.push(r.state.replace("[MASK]", " "));
    }
    out
}

#[derive(Clone, Debug)]
pub struct CorpusOptions {
    pub inputs: Vec<PathBuf>,
    /// Repetitions for game records (`games` directories).
    pub game_weight: usize,
    /// Cap on MLM text bytes fed to the trainer.
    pub max_text_mb: f64,
    /// Cap on decision records sampled per file (games included, before weighting).
    pub max_decisions_per_file: usize,
    pub seed: u64,
}

#[derive(Debug, Default)]
pub struct Corpus {
    pub segments: Vec<String>,
    pub text_bytes: u64,
    pub decision_records: usize,
    pub game_records: usize,
}

fn jsonl_files(dir: &Path) -> Vec<PathBuf> {
    let mut v: Vec<PathBuf> = fs::read_dir(dir)
        .map(|rd| {
            rd.filter_map(|e| e.ok().map(|e| e.path()))
                .filter(|p| p.extension().is_some_and(|e| e == "jsonl"))
                .collect()
        })
        .unwrap_or_default();
    v.sort();
    v
}

fn is_game_path(p: &Path) -> bool {
    p.components().any(|c| c.as_os_str() == "games")
}

/// Builds the training corpus from JSONL directories (missing directories are skipped).
pub fn build_corpus(opts: &CorpusOptions) -> Result<Corpus> {
    let mut corpus = Corpus::default();
    let mut text_files = Vec::new();
    for dir in &opts.inputs {
        if !dir.exists() {
            eprintln!("[tok] skipping missing input {dir:?}");
            continue;
        }
        let files = if dir.is_dir() {
            jsonl_files(dir)
        } else {
            vec![dir.clone()]
        };
        for f in files {
            let game = is_game_path(&f);
            let reps = if game { opts.game_weight.max(1) } else { 1 };
            let mut decisions = Vec::new();
            let mut is_text = false;
            for line in BufReader::new(fs::File::open(&f)?).lines() {
                let line = line?;
                if line.trim().is_empty() {
                    continue;
                }
                let v: Value = serde_json::from_str(&line)?;
                if v.get("opts").is_some() && v.get("ins").is_some() {
                    if let Ok(r) = serde_json::from_value::<DecisionRecord>(v) {
                        decisions.push(r);
                    }
                } else if v.get("text").is_some() {
                    is_text = true;
                    break;
                }
            }
            if is_text {
                text_files.push(f);
                continue;
            }
            if decisions.len() > opts.max_decisions_per_file {
                Rng::derived(opts.seed, &f.to_string_lossy()).shuffle(&mut decisions);
                decisions.truncate(opts.max_decisions_per_file);
            }
            for r in &decisions {
                for _ in 0..reps {
                    corpus.segments.extend(decision_segments(r));
                }
            }
            if game {
                corpus.game_records += decisions.len();
            } else {
                corpus.decision_records += decisions.len();
            }
        }
    }
    // Text: an even byte share per file, sampled deterministically by line hash.
    let budget = (opts.max_text_mb * 1e6) as u64;
    let sizes: Vec<u64> = text_files
        .iter()
        .map(|f| fs::metadata(f).map(|m| m.len()).unwrap_or(0))
        .collect();
    let total: u64 = sizes.iter().sum();
    for (f, size) in text_files.iter().zip(sizes) {
        let keep = if total <= budget {
            1.0
        } else {
            budget as f64 / total as f64
        };
        let limit = (size as f64 * keep) as u64;
        let mut used = 0u64;
        for line in BufReader::new(fs::File::open(f)?).lines() {
            let line = line?;
            if keep < 1.0 && (hash64(line.as_bytes()) % 10_000) as f64 / 1e4 >= keep {
                continue;
            }
            let Ok(v) = serde_json::from_str::<Value>(&line) else {
                continue;
            };
            if let Some(t) = v.get("text").and_then(Value::as_str) {
                used += t.len() as u64;
                corpus.text_bytes += t.len() as u64;
                corpus.segments.push(t.replace("[MASK]", " "));
            }
            if used >= limit {
                break;
            }
        }
    }
    Ok(corpus)
}

/// Segments that make the integers `0..max` frequent enough to become single tokens, both
/// with a leading space (`" 12"`, as in `alt 12`) and bare (`"12"`, as after the `" -"` of
/// `vy -12`). `reps` occurrences of each form are added; [`int_prior_reps`] scales it to the
/// corpus so the pairs compete with ordinary text merges.
pub fn int_prior_segments(max: usize, reps: usize) -> Vec<String> {
    const CHUNK: usize = 1000;
    let mut out = Vec::new();
    for n in 0..max {
        let mut left = reps;
        while left > 0 {
            let k = left.min(CHUNK);
            out.push(format!(" {n}").repeat(k));
            out.push(format!(" -{n}").repeat(k));
            left -= k;
        }
    }
    out
}

/// Repetitions per integer form: about one per 4000 corpus bytes (≈ one per 800 words).
pub fn int_prior_reps(corpus_bytes: u64) -> usize {
    ((corpus_bytes / 4000) as usize).max(1000)
}

/// Training segments for an s2 tokenizer: numbers are replaced by `[NUM]` (SPEC-S2 §2) and
/// each segment is split around `[NUM]`, exactly as the added special token splits the text at
/// encoding time, so no merge is ever learned across or inside `[NUM]`.
pub fn num_segments(segments: &[String]) -> Vec<String> {
    let mut out = Vec::with_capacity(segments.len());
    for s in segments {
        let replaced = replace_numbers(s);
        out.extend(
            replaced
                .split(NUM_TOKEN)
                .filter(|p| !p.is_empty())
                .map(str::to_string),
        );
    }
    out
}

/// Whether `tok` is an s2 tokenizer (`[NUM]` registered with its fixed id).
pub fn is_s2(tok: &Tokenizer) -> bool {
    tok.token_to_id(NUM_TOKEN) == Some(NUM_ID)
}

/// Trains a byte-level BPE tokenizer with fixed special ids and the oscar-style
/// pipeline (NFC normalizer, ByteLevel pre-tokenizer/decoder, `[CLS] $A [SEP]` template).
pub fn train_tokenizer(segments: &[String], vocab: usize, min_frequency: u64) -> Result<Tokenizer> {
    train_tokenizer_with(segments, vocab, min_frequency, &SPECIALS)
}

/// s2 tokenizer: same pipeline with `[NUM]=5` as an extra special token. `segments` must
/// already be prepared with [`num_segments`].
pub fn train_tokenizer_s2(
    segments: &[String],
    vocab: usize,
    min_frequency: u64,
) -> Result<Tokenizer> {
    let tok = train_tokenizer_with(segments, vocab, min_frequency, &SPECIALS_S2)?;
    let ids = tok
        .encode(format!("a{NUM_TOKEN}b {NUM_TOKEN}"), false)
        .map_err(|e| anyhow!(e))?;
    if ids.get_ids().iter().filter(|&&i| i == NUM_ID).count() != 2 {
        bail!("{NUM_TOKEN} is not matched as a special token");
    }
    Ok(tok)
}

fn train_tokenizer_with(
    segments: &[String],
    vocab: usize,
    min_frequency: u64,
    specials_list: &[&str],
) -> Result<Tokenizer> {
    let specials: Vec<AddedToken> = specials_list
        .iter()
        .map(|s| AddedToken::from(*s, true))
        .collect();
    let mut trainer = BpeTrainerBuilder::new()
        .vocab_size(vocab)
        .min_frequency(min_frequency)
        .show_progress(false)
        .special_tokens(specials)
        .initial_alphabet(ByteLevel::alphabet().into_iter().collect())
        .build();
    let template = TemplateProcessing::builder()
        .try_single("[CLS] $A [SEP]")
        .map_err(|e| anyhow!(e))?
        .try_pair("[CLS] $A [SEP] $B:0 [SEP]:0")
        .map_err(|e| anyhow!(e))?
        .special_tokens(vec![
            ("[CLS]".to_string(), CLS_ID),
            ("[SEP]".to_string(), SEP_ID),
        ])
        .build()?;
    let mut tok = TokenizerBuilder::<
        BPE,
        NormalizerWrapper,
        PreTokenizerWrapper,
        PostProcessorWrapper,
        DecoderWrapper,
    >::new()
    .with_model(BPE::default())
    .with_normalizer(Some(NFC.into()))
    .with_pre_tokenizer(Some(ByteLevel::new(false, true, true).into()))
    .with_post_processor(Some(template.into()))
    .with_decoder(Some(ByteLevel::new(false, true, true).into()))
    .build()
    .map_err(|e| anyhow!(e))?;
    tok.train(&mut trainer, segments.iter())
        .map_err(|e| anyhow!(e))?;
    let tok: Tokenizer = tok.into();
    for (i, s) in specials_list.iter().enumerate() {
        if tok.token_to_id(s) != Some(i as u32) {
            bail!(
                "special token {s} got id {:?}, expected {i}",
                tok.token_to_id(s)
            );
        }
    }
    Ok(tok)
}

/// Encodes one segment without specials; s2 tokenizers get numeric extraction first.
pub fn encode(tok: &Tokenizer, s: &str) -> Result<Vec<u32>> {
    let enc = if is_s2(tok) {
        tok.encode(replace_numbers(s), false)
    } else {
        tok.encode(s, false)
    };
    Ok(enc.map_err(|e| anyhow!(e))?.get_ids().to_vec())
}

/// Mean number of tokens of `[CLS] {state} [SEP]` over the game records of `dir`
/// (per file stem), the per-frame input of the s2 state encoder.
pub fn state_token_stats(
    tok: &Tokenizer,
    dir: &Path,
    sample: usize,
    seed: u64,
) -> Result<BTreeMap<String, f64>> {
    let mut out = BTreeMap::new();
    for f in jsonl_files(dir) {
        let mut recs: Vec<DecisionRecord> = crate::record::read_jsonl(&f)?;
        if recs.is_empty() {
            continue;
        }
        Rng::derived(seed, "statestats").shuffle(&mut recs);
        recs.truncate(sample);
        let mut total = 0usize;
        for r in &recs {
            total += encode(tok, &r.state.replace("[MASK]", " "))?.len() + 2;
        }
        let name = f
            .file_stem()
            .map(|x| x.to_string_lossy().into_owned())
            .unwrap_or_default();
        out.insert(name, total as f64 / recs.len() as f64);
    }
    Ok(out)
}

/// Port of `rl_common.build_sequence` (options in record order, state truncated on the right).
/// Returns the ids and the `[MASK]` marker positions.
pub fn build_sequence(
    tok: &Tokenizer,
    r: &DecisionRecord,
    max_len: usize,
    head_max_len: usize,
) -> Result<(Vec<u32>, Vec<usize>)> {
    let ins = r.ins.replace("[MASK]", " ");
    let mut head = encode(tok, &format!("{} question: {}", r.t, ins))?;
    let mut opt_ids: Vec<Vec<u32>> = Vec::with_capacity(r.opts.len());
    for o in &r.opts {
        let mut ids = vec![MASK_ID];
        let enc = encode(tok, &format!(" {}", o.replace("[MASK]", " ")))?;
        ids.extend(enc.into_iter().take(48));
        opt_ids.push(ids);
    }
    let total = |o: &Vec<Vec<u32>>| o.iter().map(Vec::len).sum::<usize>() as i64;
    let mut opt_budget = head_max_len as i64 - total(&opt_ids);
    if opt_budget < 16 {
        let per = 4.max((head_max_len as i64 - 16) / (opt_ids.len().max(1) as i64)) as usize;
        for o in opt_ids.iter_mut() {
            o.truncate(per);
        }
        opt_budget = head_max_len as i64 - total(&opt_ids);
    }
    head.truncate(8.max(opt_budget) as usize);
    let mut ids = vec![CLS_ID];
    ids.extend(head);
    ids.push(SEP_ID);
    let mut markers = Vec::new();
    for o in opt_ids {
        markers.push(ids.len());
        ids.extend(o);
    }
    ids.push(SEP_ID);
    let room = max_len.saturating_sub(ids.len() + 1);
    let st = encode(tok, &r.state.replace("[MASK]", " "))?;
    ids.extend(st.into_iter().take(room));
    ids.push(SEP_ID);
    ids.truncate(max_len);
    markers.retain(|&m| m < max_len);
    Ok((ids, markers))
}

/// Per-source sequence statistics for a trained tokenizer.
#[derive(Clone, Debug, Default)]
pub struct SeqStats {
    pub n: usize,
    pub mean_tokens: f64,
    pub mean_untruncated: f64,
    pub truncated_frac: f64,
    pub markers_lost_frac: f64,
}

/// Measures tokens per decision sequence on up to `sample` records of each `*.jsonl` in `dir`.
pub fn sequence_stats(
    tok: &Tokenizer,
    dir: &Path,
    sample: usize,
    seed: u64,
) -> Result<BTreeMap<String, SeqStats>> {
    let mut out = BTreeMap::new();
    for f in jsonl_files(dir) {
        let mut recs: Vec<DecisionRecord> = crate::record::read_jsonl(&f)?;
        if recs.is_empty() {
            continue;
        }
        Rng::derived(seed, "seqstats").shuffle(&mut recs);
        recs.truncate(sample);
        let mut s = SeqStats {
            n: recs.len(),
            ..Default::default()
        };
        for r in &recs {
            let (ids, markers) = build_sequence(tok, r, MAX_LEN, HEAD_MAX_LEN)?;
            let full: usize = decision_segments(r)
                .iter()
                .map(|x| encode(tok, x).map(|v| v.len()))
                .sum::<Result<usize>>()?
                + 4
                + r.opts.len();
            s.mean_tokens += ids.len() as f64;
            s.mean_untruncated += full as f64;
            if full > ids.len() {
                s.truncated_frac += 1.0;
            }
            if markers.len() != r.opts.len() {
                s.markers_lost_frac += 1.0;
            }
        }
        let n = s.n as f64;
        s.mean_tokens /= n;
        s.mean_untruncated /= n;
        s.truncated_frac /= n;
        s.markers_lost_frac /= n;
        let name = f
            .file_stem()
            .map(|x| x.to_string_lossy().into_owned())
            .unwrap_or_default();
        out.insert(name, s);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::record::Question;

    fn toy_corpus() -> Vec<String> {
        let mut v = Vec::new();
        for i in 0..200 {
            let q = Question::noul(format!("Is the bird above pipe number {}?", i % 7));
            let r = q.record(
                "t",
                format!("alt {} vy -3 dx {} gt 20 gb 35", i % 30, i % 50),
                vec![0.0, 1.0],
            );
            v.extend(decision_segments(&r));
            v.push("Once upon a time a little bird flew over green pipes. Il était une fois un petit oiseau.".into());
        }
        v
    }

    #[test]
    fn trains_with_fixed_special_ids_and_roundtrips() {
        let tok = train_tokenizer(&toy_corpus(), 400, 2).unwrap();
        assert_eq!(tok.token_to_id("[PAD]"), Some(0));
        assert_eq!(tok.token_to_id("[MASK]"), Some(4));
        assert!(tok.get_vocab_size(true) <= 400);
        let enc = tok
            .encode("noul question: Is the bird above pipe?", true)
            .unwrap();
        assert_eq!(enc.get_ids()[0], CLS_ID);
        assert_eq!(*enc.get_ids().last().unwrap(), SEP_ID);
        let text = "Un oiseau très rapide — ça vole! 123";
        let ids = encode(&tok, text).unwrap();
        assert_eq!(tok.decode(&ids, false).unwrap(), text);
        // Frequent schema words become single tokens (BPE never merges across the
        // GPT-2 pre-tokenizer splits, so one token per word is the floor).
        assert_eq!(
            encode(&tok, " true: yes, the statement holds")
                .unwrap()
                .len(),
            7
        );
    }

    #[test]
    fn build_sequence_matches_reference_layout() {
        let tok = train_tokenizer(&toy_corpus(), 400, 2).unwrap();
        let r = Question::noul("Is the bird above pipe number 3?").record(
            "t",
            "alt 12 vy -3",
            vec![0.0, 1.0],
        );
        let (ids, markers) = build_sequence(&tok, &r, MAX_LEN, HEAD_MAX_LEN).unwrap();
        assert_eq!(ids[0], CLS_ID);
        assert_eq!(markers.len(), 2);
        assert!(markers.iter().all(|&m| ids[m] == MASK_ID));
        assert_eq!(ids.iter().filter(|&&x| x == SEP_ID).count(), 3);
        let head = encode(&tok, "noul question: Is the bird above pipe number 3?").unwrap();
        assert_eq!(&ids[1..1 + head.len()], head.as_slice());
        assert_eq!(markers[0], head.len() + 2);
        // Truncation keeps the total at max_len.
        let long = Question::noul("x").record("t", "word ".repeat(2000), vec![1.0, 0.0]);
        let (ids, markers) = build_sequence(&tok, &long, 64, 32).unwrap();
        assert_eq!(ids.len(), 64);
        assert_eq!(*ids.last().unwrap(), SEP_ID);
        assert_eq!(markers.len(), 2);
    }

    #[test]
    fn int_prior_makes_small_integers_single_tokens() {
        let mut corpus = toy_corpus();
        corpus.extend(int_prior_segments(32, 2000));
        let tok = train_tokenizer(&corpus, 600, 2).unwrap();
        for n in [0, 7, 12, 31] {
            assert_eq!(encode(&tok, &format!(" {n}")).unwrap().len(), 1, "' {n}'");
            assert_eq!(encode(&tok, &format!("{n}")).unwrap().len(), 1, "'{n}'");
        }
        assert_eq!(int_prior_segments(2, 1500).len(), 2 * 2 * 2);
    }

    #[test]
    fn s2_tokenizer_has_num_special() {
        let segs = num_segments(&toy_corpus());
        assert!(segs.iter().all(|s| !s.contains(NUM_TOKEN)));
        assert!(segs.iter().all(|s| !s.chars().any(|c| c.is_ascii_digit())));
        let tok = train_tokenizer_s2(&segs, 400, 2).unwrap();
        assert!(is_s2(&tok));
        for (i, s) in SPECIALS_S2.iter().enumerate() {
            assert_eq!(tok.token_to_id(s), Some(i as u32));
        }
        // Numbers are extracted before BPE: one [NUM] per number, nothing learned from digits.
        let ids = encode(&tok, "alt 12 vy -3 dx 4.5").unwrap();
        assert_eq!(ids.iter().filter(|&&i| i == NUM_ID).count(), 3);
        assert!(tok
            .get_vocab(false)
            .keys()
            .all(|k| k == NUM_TOKEN || !k.contains("NUM")));
        let s1 = train_tokenizer(&toy_corpus(), 400, 2).unwrap();
        assert!(!is_s2(&s1));
        let r = Question::noul("Is pipe 3 near?").record("t", "alt 12 vy -3", vec![0.0, 1.0]);
        let (ids, markers) = build_sequence(&tok, &r, MAX_LEN, HEAD_MAX_LEN).unwrap();
        assert_eq!(ids.iter().filter(|&&i| i == NUM_ID).count(), 3);
        assert!(markers.iter().all(|&m| ids[m] == MASK_ID));
    }

    #[test]
    fn segments_follow_sequence_layout() {
        let r = Question::choice("Pick [MASK] one", vec![("a".into(), "first".into())]).record(
            "s",
            "st",
            vec![1.0],
        );
        assert_eq!(
            decision_segments(&r),
            vec!["choice question: Pick   one", " a: first", "st"]
        );
    }
}
