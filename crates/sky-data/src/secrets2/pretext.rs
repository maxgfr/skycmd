//! The CredData share of the pretraining text (`docs/s6-secrets2-prereg.md`, section 5): "the
//! obfuscated files of the CredData training repositories, 30% of sequences".
//!
//! `s2-pretrain` reads its `--text` directories into one token stream and draws its rows from
//! it uniformly, so a corpus is 30% of the text sequences when it holds 30% of the tokens. This
//! module cuts the files of the training repositories into chunks, shuffles them (seed) and
//! writes chunks, as MLM records, until their tokens are `share / (1 - share)` times those of
//! the text the pretraining already reads (`data/text-open`, `data/text`), counted the way the
//! trainer tokenizes: numeric extraction, the model's BPE, one `[SEP]` per record. With
//! `--text data/text-open,data/text,<build>/pretrain-text` the CredData text is then that share
//! of the stream.
//!
//! Only training repositories are read (never validation, never test), and only the files under
//! a scope directory: the licence copies at a repository's root are left out.

use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use serde_json::json;
use sky_infer::num::extract;
use sky_infer::tokenizer::Tokenizer;

use super::cred::Splits;
use super::{file_entry, write_json};
use crate::record::MlmRecord;
use crate::rng::Rng;

/// Characters of a chunk, at most: the size of the records of `data/text`.
pub const CHUNK_CHARS: usize = 4000;

/// The `src` of the records.
pub const SRC: &str = "creddata_text";

/// Tokens of one MLM record as `s2-pretrain` counts them: the special-token texts removed,
/// numbers extracted, the BPE of the rest, and the `[SEP]` that ends the record. An empty
/// record gives none.
pub fn record_tokens(tok: &Tokenizer, text: &str) -> usize {
    let clean = text.replace("[MASK]", " ").replace("[NUM]", " ");
    let n = tok.encode(&extract(&clean).text).len();
    if n == 0 {
        0
    } else {
        n + 1
    }
}

/// Tokens of the MLM records of a `*.jsonl` file.
fn file_tokens(tok: &Tokenizer, path: &Path) -> Result<(usize, usize)> {
    use std::io::BufRead;
    let f = std::fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
    let (mut tokens, mut records) = (0, 0);
    for line in std::io::BufReader::new(f).lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let rec: serde_json::Value =
            serde_json::from_str(&line).with_context(|| format!("parse {}", path.display()))?;
        tokens += record_tokens(tok, rec["text"].as_str().unwrap_or(""));
        records += 1;
    }
    Ok((tokens, records))
}

/// The `*.jsonl` files under a path, sorted (the trainer's `list_jsonl`).
fn list_jsonl(path: &Path) -> Result<Vec<PathBuf>> {
    if path.is_file() {
        return Ok(vec![path.to_path_buf()]);
    }
    let mut out = Vec::new();
    let mut stack = vec![path.to_path_buf()];
    while let Some(dir) = stack.pop() {
        for e in std::fs::read_dir(&dir).with_context(|| format!("list {}", dir.display()))? {
            let p = e?.path();
            if p.is_dir() {
                stack.push(p);
            } else if p.extension().is_some_and(|x| x == "jsonl") {
                out.push(p);
            }
        }
    }
    out.sort();
    Ok(out)
}

/// The byte ranges of the chunks of a text: whole lines up to [`CHUNK_CHARS`] characters, a
/// longer line cut into pieces of that many.
pub fn chunks(text: &str) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    let (mut from, mut chars) = (0usize, 0usize);
    let mut at = 0;
    for line in text.split_inclusive('\n') {
        let n = line.chars().count();
        if chars + n > CHUNK_CHARS && chars > 0 {
            out.push((from, at));
            from = at;
            chars = 0;
        }
        if n > CHUNK_CHARS {
            // a line longer than a chunk: pieces of a chunk each
            let mut piece = 0;
            for (i, _) in line.char_indices() {
                if piece == CHUNK_CHARS {
                    out.push((from, at + i));
                    from = at + i;
                    piece = 0;
                }
                piece += 1;
            }
            chars = piece;
        } else {
            chars += n;
        }
        at += line.len();
    }
    if at > from {
        out.push((from, at));
    }
    out.retain(|&(a, b)| !text[a..b].trim().is_empty());
    out
}

pub struct PretextOptions {
    /// The data folder.
    pub data: PathBuf,
    /// The build folder.
    pub out: PathBuf,
    pub seed: u64,
    pub tokenizer: Tokenizer,
    /// The text the pretraining already reads (`data/text-open`, `data/text`).
    pub text: Vec<PathBuf>,
    /// The share of the text sequences the CredData text is to be.
    pub share: f64,
}

fn walk(dir: &Path, rel: &str, out: &mut Vec<String>) -> Result<()> {
    let mut entries: Vec<_> = std::fs::read_dir(dir)
        .with_context(|| format!("list {}", dir.display()))?
        .collect::<std::io::Result<_>>()?;
    entries.sort_by_key(|e| e.file_name());
    for e in entries {
        let t = e.file_type()?;
        let name = e.file_name().to_string_lossy().into_owned();
        if t.is_symlink() {
            continue;
        }
        if t.is_dir() {
            walk(&e.path(), &format!("{rel}/{name}"), out)?;
        } else if t.is_file() {
            out.push(format!("{rel}/{name}"));
        }
    }
    Ok(())
}

/// Builds `<out>/pretrain-text/creddata.jsonl` and `<out>/counts/pretrain-text.json`.
pub fn build(o: &PretextOptions) -> Result<serde_json::Value> {
    if !(0.0..1.0).contains(&o.share) {
        bail!("--share must be in [0, 1)");
    }
    // the text the pretraining already reads, counted file by file on as many threads
    let mut files = Vec::new();
    for t in &o.text {
        let found = list_jsonl(t)?;
        if found.is_empty() {
            bail!("no *.jsonl under {}", t.display());
        }
        files.extend(found);
    }
    let counted: Vec<Result<(usize, usize)>> = std::thread::scope(|s| {
        let handles: Vec<_> = files
            .iter()
            .map(|f| s.spawn(|| file_tokens(&o.tokenizer, f)))
            .collect();
        handles
            .into_iter()
            .map(|h| {
                h.join()
                    .unwrap_or_else(|_| bail!("a counting thread failed"))
            })
            .collect()
    });
    let mut other = serde_json::Map::new();
    let mut other_tokens = 0usize;
    for (f, c) in files.iter().zip(counted) {
        let (tokens, records) = c?;
        other_tokens += tokens;
        let name = f
            .components()
            .rev()
            .take(2)
            .map(|c| c.as_os_str().to_string_lossy().into_owned())
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect::<Vec<_>>()
            .join("/");
        other.insert(
            name,
            json!({"bytes": std::fs::metadata(f)?.len(), "sha256": super::sha256_file(f)?, "records": records, "tokens": tokens}),
        );
    }
    let target = (other_tokens as f64 * o.share / (1.0 - o.share)).round() as usize;

    // every chunk of every file of the training repositories
    let splits = Splits::read(&o.data)?;
    let cred = o.data.join("creddata");
    let mut list = Vec::new();
    for repo in &splits.training {
        walk(
            &cred.join("data").join(repo),
            &format!("data/{repo}"),
            &mut list,
        )?;
    }
    let (mut read, mut skipped, mut root) = (0usize, 0usize, 0usize);
    let mut all: Vec<(u32, usize, usize)> = Vec::new();
    let mut kept_files: Vec<&str> = Vec::new();
    for file in &list {
        if file.split('/').count() == 3 {
            root += 1;
            continue;
        }
        let bytes = std::fs::read(cred.join(file))?;
        if bytes.is_empty() || bytes[..bytes.len().min(8192)].contains(&0) {
            skipped += 1;
            continue;
        }
        read += 1;
        let text = String::from_utf8_lossy(&bytes);
        let id = kept_files.len() as u32;
        kept_files.push(file);
        all.extend(chunks(&text).into_iter().map(|(a, b)| (id, a, b)));
    }
    let total_chunks = all.len();
    Rng::derived(o.seed, "s6-secrets2:pretrain-text").shuffle(&mut all);

    // chunks in the shuffled order until the tokens are there
    let path = o.out.join("pretrain-text/creddata.jsonl");
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let mut w = std::io::BufWriter::new(std::fs::File::create(&path)?);
    let (mut tokens, mut chars, mut written) = (0usize, 0usize, 0usize);
    let mut files_used = std::collections::HashSet::new();
    // The shuffled order is kept; a block of it is read file by file, so that a large file,
    // which holds many chunks, is read once per block and not once per chunk.
    'blocks: for block in all.chunks(4096) {
        if tokens >= target {
            break;
        }
        let mut by_file: Vec<usize> = (0..block.len()).collect();
        by_file.sort_unstable_by_key(|&i| block[i]);
        let mut texts: Vec<String> = vec![String::new(); block.len()];
        let mut cache: Option<(u32, String)> = None;
        for i in by_file {
            let (id, a, b) = block[i];
            if cache.as_ref().map(|c| c.0) != Some(id) {
                let bytes = std::fs::read(cred.join(kept_files[id as usize]))?;
                cache = Some((id, String::from_utf8_lossy(&bytes).into_owned()));
            }
            texts[i] = cache.as_ref().expect("the file just read").1[a..b].to_string();
        }
        for (i, text) in texts.iter().enumerate() {
            if tokens >= target {
                break 'blocks;
            }
            let n = record_tokens(&o.tokenizer, text);
            if n == 0 {
                continue;
            }
            tokens += n;
            chars += text.chars().count();
            written += 1;
            files_used.insert(block[i].0);
            use std::io::Write;
            serde_json::to_writer(
                &mut w,
                &MlmRecord {
                    src: SRC.to_string(),
                    text: text.clone(),
                },
            )?;
            w.write_all(b"\n")?;
        }
    }
    use std::io::Write;
    w.flush()?;
    drop(w);
    if tokens < target {
        bail!("the training repositories hold {tokens} tokens, {target} are needed");
    }
    let out = json!({
        "share": o.share,
        "text_already_read": {"files": other, "tokens": other_tokens},
        "target_tokens": target,
        "training_repositories": splits.training.len(),
        "files_listed": list.len(),
        "files_at_a_repository_root_left_out": root,
        "files_skipped_empty_or_binary": skipped,
        "files_read": read,
        "chunks_available": total_chunks,
        "chunk_characters_at_most": CHUNK_CHARS,
        "records": written,
        "files_used": files_used.len(),
        "characters": chars,
        "tokens": tokens,
        "share_of_the_token_stream": ((tokens as f64 / (tokens + other_tokens) as f64) * 1e6).round() / 1e6,
        "output": {"pretrain-text/creddata.jsonl": file_entry(&path)?},
    });
    write_json(&o.out.join("counts/pretrain-text.json"), &out)?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunks_keep_whole_lines_and_cover_the_text() {
        let line = format!("{}\n", "x".repeat(99));
        let text = line.repeat(100);
        let c = chunks(&text);
        assert_eq!(c.len(), 3);
        assert_eq!(c[0], (0, 4000));
        assert!(c
            .iter()
            .all(|&(a, b)| text[a..b].ends_with('\n') && b - a <= CHUNK_CHARS));
        assert_eq!(c.iter().map(|&(a, b)| b - a).sum::<usize>(), text.len());
        // a line longer than a chunk is cut into pieces, on character boundaries
        let text = format!("short\n{}\nend\n", "\u{e9}".repeat(9000));
        let c = chunks(&text);
        assert!(c
            .iter()
            .all(|&(a, b)| text[a..b].chars().count() <= CHUNK_CHARS));
        assert_eq!(c.iter().map(|&(a, b)| b - a).sum::<usize>(), text.len());
        assert_eq!(c.first().map(|c| c.0), Some(0));
        assert!(c.windows(2).all(|w| w[0].1 == w[1].0));
        // nothing for an empty or blank text
        assert!(chunks("").is_empty() && chunks(" \n\n").is_empty());
        assert_eq!(chunks("no newline"), [(0, 10)]);
    }
}
