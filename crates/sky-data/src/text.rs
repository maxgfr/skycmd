//! `fetch-text`: MLM corpora from Hugging Face, size-capped and deduplicated.

use std::io::{BufRead, BufReader, Read};
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
    Converted {
        config: &'static str,
        split: &'static str,
    },
    /// A specific parquet file in the repo at `main`.
    File(&'static str),
    /// JSON Lines files in the repo at `main`, streamed in order until the budget is met
    /// (`.zst` and `.gz` are decompressed on the fly).
    Jsonl(&'static [&'static str]),
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
    /// Keep only rows whose value at this JSON pointer is this string.
    pub filter: Option<(&'static str, &'static str)>,
    /// Pieces of at most `max_chars` kept per document: 1 truncates, more splits long
    /// documents (books, parliament sessions) so that the bytes downloaded are not thrown away.
    pub chunks: usize,
}

/// The source sets of `fetch-text`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, clap::ValueEnum)]
pub enum TextSet {
    /// `data/text`: stories and web text, the corpus every model so far was pretrained on.
    #[default]
    Default,
    /// `data/text-open`: non-web parts of the training data of fully open LLMs (Dolma 3 for
    /// OLMo 3, TxT360 for LLM360).
    Open,
}

impl TextSet {
    pub fn sources(self) -> Vec<TextSource> {
        match self {
            TextSet::Default => text_sources(),
            TextSet::Open => open_text_sources(),
        }
    }
}

pub fn text_sources() -> Vec<TextSource> {
    vec![
        TextSource {
            name: "simplestories",
            repo: "SimpleStories/SimpleStories",
            loc: Location::Converted {
                config: "default",
                split: "train",
            },
            column: "story",
            share: 0.20,
            pass_shortfall: false,
            filter: None,
            chunks: 1,
        },
        TextSource {
            name: "tinystories",
            repo: "roneneldan/TinyStories",
            loc: Location::Converted {
                config: "default",
                split: "train",
            },
            column: "text",
            share: 0.20,
            pass_shortfall: false,
            filter: None,
            chunks: 1,
        },
        TextSource {
            name: "fineweb_edu",
            repo: "HuggingFaceFW/fineweb-edu",
            loc: Location::File("sample/10BT/000_00000.parquet"),
            column: "text",
            share: 0.25,
            pass_shortfall: false,
            filter: None,
            chunks: 1,
        },
        // French stories come right before fineweb-2 French so that their
        // shortfall (the corpus is small) stays French.
        TextSource {
            name: "tinystories_fr",
            repo: "btecsec/tinystories-fr",
            loc: Location::Converted {
                config: "default",
                split: "train",
            },
            column: "text",
            share: 0.20,
            pass_shortfall: true,
            filter: None,
            chunks: 1,
        },
        TextSource {
            name: "fineweb2_fr",
            repo: "HuggingFaceFW/fineweb-2",
            loc: Location::File("data/fra_Latn/train/000_00000.parquet"),
            column: "text",
            share: 0.15,
            pass_shortfall: false,
            filter: None,
            chunks: 1,
        },
    ]
}

const DOLMA3: &str = "allenai/dolma3_mix-150B-1025";
const TXT360: &str = "IFM/TxT360";

/// One JSONL source of the open set, truncated to `max_chars`, unfiltered.
fn jsonl(
    name: &'static str,
    repo: &'static str,
    files: &'static [&'static str],
    share: f64,
) -> TextSource {
    TextSource {
        name,
        repo,
        loc: Location::Jsonl(files),
        column: "text",
        share,
        pass_shortfall: false,
        filter: None,
        chunks: 1,
    }
}

pub fn open_text_sources() -> Vec<TextSource> {
    vec![
        jsonl(
            "dolma3_wiki_en",
            DOLMA3,
            &["data/dolma1_7-wiki-en/shard_00000000.jsonl.zst"],
            0.25,
        ),
        jsonl(
            "dolma3_science_pdfs",
            DOLMA3,
            &["data/olmocr_science_pdfs-science_math_and_technology/shard_00000000.jsonl.zst"],
            0.25,
        ),
        jsonl(
            "dolma3_finemath",
            DOLMA3,
            &["data/finemath-3plus/shard_00000000.jsonl.zst"],
            0.15,
        ),
        // A book is ~400 KB: up to 8 pieces of each instead of its first 20,000 characters.
        TextSource {
            chunks: 8,
            ..jsonl("txt360_pg19", TXT360, &["data/pg19/1-1/0_0.jsonl"], 0.15)
        },
        jsonl(
            "txt360_stackexchange",
            TXT360,
            &["data/stackexchange/1-1/0_2289.jsonl"],
            0.10,
        ),
        // Every Europarl file mixes the 24 languages: French is about one document in twenty,
        // so this share costs about 20 times its size in download.
        TextSource {
            filter: Some(("/meta/language", "fr")),
            chunks: 8,
            ..jsonl(
                "txt360_europarl_fr",
                TXT360,
                &["data/europarl/1-1/0_6.jsonl", "data/europarl/1-1/1_6.jsonl"],
                0.10,
            )
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
    pub set: TextSet,
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
            let cut = s[..i]
                .rfind(['\n', '.', '!', '?'])
                .filter(|&j| j > i / 2)
                .map_or(i, |j| j + 1);
            s[..cut].trim_end().to_string()
        }
        None => s.to_string(),
    }
}

/// Up to `n` consecutive pieces of `clean_text`; the first is `clean_text(s, max_chars)`.
pub fn split_text(s: &str, max_chars: usize, n: usize) -> Vec<String> {
    let s = s.replace("\r\n", "\n").replace('\r', "\n");
    let mut rest = s.as_str();
    let mut pieces = Vec::new();
    while pieces.len() < n {
        let piece = clean_text(rest, max_chars);
        if piece.is_empty() {
            break;
        }
        // `piece` is a prefix of the trimmed rest.
        rest = &rest.trim_start()[piece.len()..];
        pieces.push(piece);
    }
    pieces
}

pub fn fetch_text(hf: &Hf, opts: &TextOptions) -> Result<Vec<TextReport>> {
    std::fs::create_dir_all(&opts.out)?;
    let total = (opts.max_mb * 1e6) as u64;
    let mut dedup = Dedup::new(0.8, opts.seed);
    let mut reports = Vec::new();
    let mut carry: u64 = 0;
    for src in opts.set.sources() {
        if !opts.only.is_empty() && !opts.only.iter().any(|o| o == src.name) {
            continue;
        }
        let t0 = Instant::now();
        let budget = (total as f64 * src.share) as u64 + carry;
        carry = 0;
        let (exact0, near0) = (dedup.exact_dups, dedup.near_dups);
        eprintln!(
            "[text] {} budget {:.1} MB ...",
            src.name,
            budget as f64 / 1e6
        );
        let (docs, downloaded) = collect_source(hf, &src, budget, opts, &mut dedup)
            .with_context(|| format!("text source {}", src.name))?;
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

/// The docs of one source as rows come in: filtered, cleaned, deduplicated, up to the budget.
struct Collector<'a> {
    src: &'a TextSource,
    opts: &'a TextOptions,
    dedup: &'a mut Dedup,
    budget: u64,
    bytes: u64,
    docs: Vec<MlmRecord>,
}

impl Collector<'_> {
    /// Keeps the row if it passes; returns `false` once the budget is met.
    fn push(&mut self, row: &Value) -> bool {
        if let Some((pointer, want)) = self.src.filter {
            if row.pointer(pointer).and_then(Value::as_str) != Some(want) {
                return true;
            }
        }
        let Some(text) = row.get(self.src.column).and_then(Value::as_str) else {
            return true;
        };
        for text in split_text(text, self.opts.max_chars, self.src.chunks) {
            if text.chars().count() < self.opts.min_chars || !self.dedup.insert(&text) {
                continue;
            }
            self.bytes += text.len() as u64;
            self.docs.push(MlmRecord {
                src: self.src.name.to_string(),
                text,
            });
            if self.bytes >= self.budget {
                return false;
            }
        }
        true
    }

    /// Reads JSON Lines until the budget is met or the stream ends. Lines that do not parse
    /// (a truncated last line) are skipped.
    fn read_jsonl(&mut self, r: impl Read) -> Result<()> {
        for line in BufReader::new(r).lines() {
            let Ok(row) = serde_json::from_str::<Value>(&line?) else {
                continue;
            };
            if !self.push(&row) {
                break;
            }
        }
        Ok(())
    }
}

/// Counts the bytes read through it (the compressed bytes of a JSONL stream).
struct Counted<R> {
    inner: R,
    n: u64,
}

impl<R: Read> Read for Counted<R> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let k = self.inner.read(buf)?;
        self.n += k as u64;
        Ok(k)
    }
}

/// Decompresses by file extension: `.zst`, `.gz`, anything else as is.
fn decoder<'a>(path: &str, r: impl Read + 'a) -> Result<Box<dyn Read + 'a>> {
    Ok(if path.ends_with(".zst") {
        Box::new(zstd::stream::read::Decoder::new(r)?)
    } else if path.ends_with(".gz") {
        Box::new(flate2::read::MultiGzDecoder::new(r))
    } else {
        Box::new(r)
    })
}

fn collect_source(
    hf: &Hf,
    src: &TextSource,
    budget: u64,
    opts: &TextOptions,
    dedup: &mut Dedup,
) -> Result<(Vec<MlmRecord>, u64)> {
    let mut c = Collector {
        src,
        opts,
        dedup,
        budget,
        bytes: 0,
        docs: Vec::new(),
    };
    let mut downloaded = 0u64;
    let urls = match &src.loc {
        Location::Converted { config, split } => hf.parquet_urls(src.repo, config, split)?,
        Location::File(f) => vec![resolve_url(src.repo, "main", f)],
        Location::Jsonl(files) => {
            for f in files.iter() {
                let url = resolve_url(src.repo, "main", f);
                let mut raw = Counted {
                    inner: hf.stream(&url)?,
                    n: 0,
                };
                c.read_jsonl(decoder(f, &mut raw)?)
                    .with_context(|| format!("read {url}"))?;
                downloaded += raw.n;
                if c.bytes >= budget {
                    break;
                }
            }
            return Ok((c.docs, downloaded));
        }
    };
    'outer: for (si, url) in urls.iter().enumerate() {
        let rel = format!(
            "{}/{}-{si:05}.parquet",
            src.repo.replace('/', "__"),
            src.name
        );
        let mut pq = hf.remote_parquet(url, &rel)?;
        for rg in 0..pq.num_row_groups() {
            downloaded += pq.ensure(rg, &[src.column])?;
            for row in pq.rows(rg, &[src.column])? {
                if !c.push(&row) {
                    break 'outer;
                }
            }
        }
    }
    Ok((c.docs, downloaded))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shares_sum_to_one() {
        for set in [TextSet::Default, TextSet::Open] {
            let s: f64 = set.sources().iter().map(|s| s.share).sum();
            assert!((s - 1.0).abs() < 1e-9, "{set:?}");
        }
    }

    #[test]
    fn reads_jsonl_with_filter_and_budget() {
        let mut lines = String::new();
        for i in 0..20 {
            let lang = if i % 2 == 0 { "fr" } else { "en" };
            lines += &format!(
                "{{\"text\":\"document numero {i:02} assez long\",\"meta\":{{\"language\":\"{lang}\"}}}}\n"
            );
        }
        lines += "{\"text\":\"tronqu";
        let src = TextSource {
            filter: Some(("/meta/language", "fr")),
            ..open_text_sources().remove(0)
        };
        let opts = TextOptions {
            out: PathBuf::new(),
            max_mb: 0.0,
            seed: 1,
            min_chars: 10,
            max_chars: 1000,
            only: vec![],
            set: TextSet::Open,
        };
        let run = |path: &str, bytes: &[u8], budget: u64| {
            let mut dedup = Dedup::new(0.8, 1);
            let mut c = Collector {
                src: &src,
                opts: &opts,
                dedup: &mut dedup,
                budget,
                bytes: 0,
                docs: Vec::new(),
            };
            c.read_jsonl(decoder(path, bytes).unwrap()).unwrap();
            c.docs.into_iter().map(|d| d.text).collect::<Vec<_>>()
        };
        let zst = zstd::encode_all(lines.as_bytes(), 3).unwrap();
        let mut gz = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::fast());
        std::io::Write::write_all(&mut gz, lines.as_bytes()).unwrap();
        let gz = gz.finish().unwrap();
        // Four French docs of 29 bytes meet a budget of 100.
        let want: Vec<String> = (0..4)
            .map(|i| format!("document numero {:02} assez long", 2 * i))
            .collect();
        for (path, bytes) in [
            ("a.jsonl", lines.as_bytes()),
            ("a.jsonl.zst", &zst[..]),
            ("a.jsonl.gz", &gz[..]),
        ] {
            assert_eq!(run(path, bytes, 100), want, "{path}");
        }
        // Without a budget: the ten French docs, and the truncated last line is skipped.
        assert_eq!(run("a.jsonl", lines.as_bytes(), u64::MAX).len(), 10);
    }

    #[test]
    fn cleans_and_caps() {
        assert_eq!(clean_text("  a\r\nb  ", 100), "a\nb");
        let long = "First sentence. Second sentence goes on and on";
        assert_eq!(clean_text(long, 25), "First sentence.");
        assert_eq!(clean_text("ééééé", 3), "ééé");
    }

    #[test]
    fn splits_long_documents() {
        let long = "First sentence. Second sentence goes on.\r\nThird one here. Fourth.";
        assert_eq!(split_text(long, 25, 1), [clean_text(long, 25)]);
        assert_eq!(
            split_text(long, 25, 8),
            [
                "First sentence.",
                "Second sentence goes on.",
                "Third one here. Fourth."
            ]
        );
        assert_eq!(split_text(long, 25, 2).len(), 2);
        assert!(split_text("   ", 25, 8).is_empty());
    }
}
