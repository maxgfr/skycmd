//! `fetch-text`: MLM corpora from Hugging Face, size-capped and deduplicated.

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};
use serde_json::Value;

use crate::hf::{resolve_url, Hf};
use crate::minhash::Dedup;
use crate::record::{write_jsonl, MlmRecord};

/// Where a text source lives.
#[derive(Clone, Debug)]
pub enum Location {
    /// Dataset-viewer parquet conversion (`/api/datasets/<repo>/parquet/<config>/<split>`).
    Converted { config: &'static str, split: &'static str },
    /// A specific parquet file in the repo at `main`.
    File(&'static str),
}

#[derive(Clone, Debug)]
pub struct TextSource {
    pub name: &'static str,
    pub repo: &'static str,
    pub loc: Location,
    pub column: &'static str,
    /// Share of the total byte budget.
    pub share: f64,
    /// Whether a shortfall of this source is handed to the next source.
    pub pass_shortfall: bool,
}

pub fn text_sources() -> Vec<TextSource> {
    vec![
        TextSource {
            name: "simplestories",
            repo: "SimpleStories/SimpleStories",
            loc: Location::Converted { config: "default", split: "train" },
            column: "story",
            share: 0.20,
            pass_shortfall: false,
        },
        TextSource {
            name: "tinystories",
            repo: "roneneldan/TinyStories",
            loc: Location::Converted { config: "default", split: "train" },
            column: "text",
            share: 0.20,
            pass_shortfall: false,
        },
        TextSource {
            name: "fineweb_edu",
            repo: "HuggingFaceFW/fineweb-edu",
            loc: Location::File("sample/10BT/000_00000.parquet"),
            column: "text",
            share: 0.25,
            pass_shortfall: false,
        },
        // French stories come right before fineweb-2 French so that their
        // shortfall (the corpus is small) stays French.
        TextSource {
            name: "tinystories_fr",
            repo: "btecsec/tinystories-fr",
            loc: Location::Converted { config: "default", split: "train" },
            column: "text",
            share: 0.20,
            pass_shortfall: true,
        },
        TextSource {
            name: "fineweb2_fr",
            repo: "HuggingFaceFW/fineweb-2",
            loc: Location::File("data/fra_Latn/train/000_00000.parquet"),
            column: "text",
            share: 0.15,
            pass_shortfall: false,
        },
    ]
}

#[derive(Clone, Debug)]
pub struct TextOptions {
    pub out: PathBuf,
    pub max_mb: f64,
    pub seed: u64,
    pub min_chars: usize,
    pub max_chars: usize,
    pub only: Vec<String>,
}

#[derive(Clone, Debug)]
pub struct TextReport {
    pub src: String,
    pub docs: usize,
    pub bytes: u64,
    pub budget: u64,
    pub exact_dups: usize,
    pub near_dups: usize,
    pub downloaded: u64,
    pub secs: f64,
}

/// Cleans a document: normalized newlines, trimmed, length-capped on a char boundary.
pub fn clean_text(s: &str, max_chars: usize) -> String {
    let s = s.replace("\r\n", "\n").replace('\r', "\n");
    let s = s.trim();
    match s.char_indices().nth(max_chars) {
        Some((i, _)) => {
            // Cut at the last sentence or line end before the cap when possible.
            let cut = s[..i].rfind(['\n', '.', '!', '?']).filter(|&j| j > i / 2).map_or(i, |j| j + 1);
            s[..cut].trim_end().to_string()
        }
        None => s.to_string(),
    }
}

pub fn fetch_text(hf: &Hf, opts: &TextOptions) -> Result<Vec<TextReport>> {
    std::fs::create_dir_all(&opts.out)?;
    let total = (opts.max_mb * 1e6) as u64;
    let mut dedup = Dedup::new(0.8, opts.seed);
    let mut reports = Vec::new();
    let mut carry: u64 = 0;
    for src in text_sources() {
        if !opts.only.is_empty() && !opts.only.iter().any(|o| o == src.name) {
            continue;
        }
        let t0 = Instant::now();
        let budget = (total as f64 * src.share) as u64 + carry;
        carry = 0;
        let (exact0, near0) = (dedup.exact_dups, dedup.near_dups);
        eprintln!("[text] {} budget {:.1} MB ...", src.name, budget as f64 / 1e6);
        let (docs, downloaded) = collect_source(hf, &src, budget, opts, &mut dedup).with_context(|| format!("text source {}", src.name))?;
        let bytes: u64 = docs.iter().map(|d| d.text.len() as u64).sum();
        if src.pass_shortfall {
            carry = budget.saturating_sub(bytes);
        }
        write_jsonl(&opts.out.join(format!("{}.jsonl", src.name)), &docs)?;
        let r = TextReport {
            src: src.name.to_string(),
            docs: docs.len(),
            bytes,
            budget,
            exact_dups: dedup.exact_dups - exact0,
            near_dups: dedup.near_dups - near0,
            downloaded,
            secs: t0.elapsed().as_secs_f64(),
        };
        eprintln!(
            "[text] {}: {} docs, {:.1} MB (exact dups {}, near dups {}, downloaded {:.1} MB, {:.1}s)",
            r.src,
            r.docs,
            r.bytes as f64 / 1e6,
            r.exact_dups,
            r.near_dups,
            r.downloaded as f64 / 1e6,
            r.secs
        );
        reports.push(r);
    }
    Ok(reports)
}

fn collect_source(hf: &Hf, src: &TextSource, budget: u64, opts: &TextOptions, dedup: &mut Dedup) -> Result<(Vec<MlmRecord>, u64)> {
    let urls = match &src.loc {
        Location::Converted { config, split } => hf.parquet_urls(src.repo, config, split)?,
        Location::File(f) => vec![resolve_url(src.repo, "main", f)],
    };
    let mut docs = Vec::new();
    let mut bytes = 0u64;
    let mut downloaded = 0u64;
    'outer: for (si, url) in urls.iter().enumerate() {
        let rel = format!("{}/{}-{si:05}.parquet", src.repo.replace('/', "__"), src.name);
        let mut pq = hf.remote_parquet(url, &rel)?;
        for rg in 0..pq.num_row_groups() {
            downloaded += pq.ensure(rg, &[src.column])?;
            for row in pq.rows(rg, &[src.column])? {
                let Some(text) = row.get(src.column).and_then(Value::as_str) else { continue };
                let text = clean_text(text, opts.max_chars);
                if text.chars().count() < opts.min_chars || !dedup.insert(&text) {
                    continue;
                }
                bytes += text.len() as u64;
                docs.push(MlmRecord {
                    src: src.name.to_string(),
                    text,
                });
                if bytes >= budget {
                    break 'outer;
                }
            }
        }
    }
    Ok((docs, downloaded))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shares_sum_to_one() {
        let s: f64 = text_sources().iter().map(|s| s.share).sum();
        assert!((s - 1.0).abs() < 1e-9);
    }

    #[test]
    fn cleans_and_caps() {
        assert_eq!(clean_text("  a\r\nb  ", 100), "a\nb");
        let long = "First sentence. Second sentence goes on and on";
        assert_eq!(clean_text(long, 25), "First sentence.");
        assert_eq!(clean_text("ééééé", 3), "ééé");
    }
}
