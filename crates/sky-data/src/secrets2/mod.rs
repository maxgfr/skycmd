//! The data side of s6-secrets v2 (`docs/s6-secrets2-prereg.md`, sections 3 to 5): everything
//! between the frozen candidate generator's output and the training files.
//!
//! - [`cred`]: CredData candidates of the training and validation repositories with their labels,
//!   the weak negatives, the decontamination against the test repositories (3.1);
//! - [`issues`]: the issue-report train and validation splits as candidate records, their
//!   decontamination against the test split and the replacement of every real value (3.2);
//! - [`replace`]: the word-preserving shape replacement;
//! - [`synth`]: synthetic candidates, `sky-data secret-cand` (3.3);
//! - [`mix`]: states, decision records and the tables the baselines read (3.4, 4.1);
//! - [`pretext`]: the pretraining text of section 5;
//! - [`build`], which writes `BUILD.json`.
//!
//! One record format serves the three sources: the candidate record of the generator
//! `sg-cand-1` ([`CandRecord`]), and one state builder, `sky_infer::secret_cand`, which the
//! engine also calls.
//!
//! The values these files hold are obfuscated (CredData), replaced (issue reports) or invented
//! (synthetic), and the lines around them are real text: every output goes to the data folder,
//! outside any repository, and nothing here prints a value or a line.

pub mod build;
pub mod cred;
pub mod issues;
pub mod mix;
pub mod pretext;
pub mod replace;
pub mod synth;

use std::collections::BTreeMap;
use std::io::Read;
use std::path::Path;

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use sky_infer::secret_cand::{self as sc, Candidate, Features, Filters, WordSet};
use sky_infer::tokenizer::Tokenizer;

/// The six filter flags of a candidate record. The state carries the four of section 4.2
/// (`entropy`, `placeholder`, `stopword`, `allowlist`).
#[derive(Serialize, Deserialize, Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RecFilters {
    pub entropy: bool,
    pub placeholder: bool,
    pub allowlist: bool,
    pub stopword: bool,
    pub context: bool,
    pub pragma: bool,
}

/// A candidate record, field for field as `sg-cand-1` writes it (secretgate,
/// `scripts/rule-candidates.mjs`): columns and offsets in UTF-16 code units, `cpStart` and
/// `cpEnd` in code points, lines 1-based.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct CandRecord {
    pub generator: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub revision: Option<u32>,
    pub file: Option<String>,
    pub ext: String,
    pub rule: String,
    pub line: u64,
    pub start: usize,
    pub end: usize,
    #[serde(rename = "cpStart")]
    pub cp_start: usize,
    #[serde(rename = "cpEnd")]
    pub cp_end: usize,
    #[serde(rename = "endLine")]
    pub end_line: u64,
    pub offset: usize,
    pub length: usize,
    pub value: String,
    pub name: Option<String>,
    #[serde(rename = "lineText")]
    pub line_text: String,
    #[serde(rename = "lineFrom")]
    pub line_from: usize,
    #[serde(rename = "prevLine")]
    pub prev_line: Option<String>,
    #[serde(rename = "nextLine")]
    pub next_line: Option<String>,
    pub entropy: f64,
    pub filters: RecFilters,
    pub reported: bool,
    pub masked: bool,
}

/// Characters the generator keeps of a line (`--max-line`).
pub const MAX_LINE: usize = 1000;

/// A candidate record for a value at `[start, end)` (byte offsets) of `text`, with the line
/// conventions of the generator: lines end at `\n`, a `\r` before it is not part of the line, a
/// line longer than [`MAX_LINE`] is cut to a window around the value, and the neighbours are the
/// line before the value's first line and the line after its last.
#[allow(clippy::too_many_arguments)]
pub fn record_at(
    generator: &str,
    file: Option<&str>,
    rule: &str,
    name: Option<&str>,
    text: &str,
    start: usize,
    end: usize,
    filters: RecFilters,
) -> CandRecord {
    let units = |s: &str| s.encode_utf16().count();
    let line_bounds = |at: usize| {
        let from = text[..at].rfind('\n').map_or(0, |i| i + 1);
        let to = text[at..].find('\n').map_or(text.len(), |i| at + i);
        let to = if to > from && text.as_bytes()[to - 1] == b'\r' {
            to - 1
        } else {
            to
        };
        (from, to.max(from))
    };
    let (ls, le) = line_bounds(start);
    // the line of the value's last character
    let last = text[start..end]
        .char_indices()
        .last()
        .map_or(start, |(i, _)| start + i);
    let (es, _) = line_bounds(last);
    let line_no = text[..ls].matches('\n').count() as u64 + 1;
    let end_line = line_no + text[ls..es].matches('\n').count() as u64;
    let value = &text[start..end];
    // the window of a long line, in UTF-16 units as the generator counts it
    let line: Vec<u16> = text[ls..le].encode_utf16().collect();
    let col = units(&text[ls..start]);
    let in_line = units(value).min(line.len().saturating_sub(col));
    let line_from = if line.len() <= MAX_LINE {
        0
    } else {
        (col as i64 - ((MAX_LINE - in_line.min(MAX_LINE)) / 2) as i64)
            .min((line.len() - MAX_LINE) as i64)
            .max(0) as usize
    };
    let cut = |from: usize, to: usize| {
        let l: Vec<u16> = text[from..to].encode_utf16().take(MAX_LINE).collect();
        String::from_utf16_lossy(&l)
    };
    let prev_line = (ls > 0).then(|| {
        let (a, b) = line_bounds(ls - 1);
        cut(a, b)
    });
    let after = text[es..].find('\n').map(|i| es + i + 1);
    let next_line = after.map(|a| {
        let (a, b) = line_bounds(a);
        cut(a, b)
    });
    CandRecord {
        generator: generator.to_string(),
        revision: None,
        file: file.map(str::to_string),
        ext: file.map_or(String::new(), |f| sc::split_extension(f).1),
        rule: rule.to_string(),
        line: line_no,
        start: col,
        end: units(&text[es..end]),
        cp_start: text[ls..start].chars().count(),
        cp_end: text[es..end].chars().count(),
        end_line,
        offset: units(&text[..start]),
        length: units(value),
        value: value.to_string(),
        name: name.map(str::to_string),
        line_text: String::from_utf16_lossy(
            &line[line_from..(line_from + MAX_LINE).min(line.len())],
        ),
        line_from,
        prev_line,
        next_line,
        entropy: (sc::entropy(value) * 1e4).round() / 1e4,
        filters,
        reported: false,
        masked: false,
    }
}

/// CredData's ground truth: `T` a real credential, `X` an example, test value or placeholder,
/// `F` not a credential.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Label {
    T,
    X,
    F,
}

impl Label {
    pub fn as_str(self) -> &'static str {
        match self {
            Label::T => "T",
            Label::X => "X",
            Label::F => "F",
        }
    }

    /// Index of the label in the options of the kind question
    /// ([`sky_infer::secret_cand::KINDS`]).
    pub fn kind_index(self) -> usize {
        match self {
            Label::T => 0,
            Label::X => 1,
            Label::F => 2,
        }
    }
}

/// The sources of the mix (section 3.4), which are also the names of the record directories
/// and of the trainer's `--weights`.
pub const CREDDATA: &str = "creddata";
pub const ISSUES: &str = "issues";
pub const SYNTH: &str = "synth";

/// One training or validation candidate with what is known of it.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct Item {
    /// `<source>:<split>:<index>`: the index of the candidate in its source file.
    pub id: String,
    pub source: String,
    /// `train` or `val`.
    pub split: String,
    pub label: Label,
    /// A weak negative of section 3.1: a candidate on a line CredData did not label.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub weak: bool,
    /// Whether the kind question can be asked: the issue-report set says real or not, never
    /// which kind of not.
    pub kind: bool,
    /// CredData: the category of the row the label comes from. Synthetic: the class of the
    /// candidate.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub class: Option<String>,
    /// CredData: `value` when the label comes from a row with a value span, `line` from a
    /// whole-line row.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub via: Option<String>,
    pub rec: CandRecord,
}

/// Scope words and extension of an item's file. A CredData path is
/// `data/<repo>/<scope…>/<file id><ext>`: its directories are what `get_file_scope()` wrote
/// from the real path, read back as they are. Any other path is a real one
/// ([`sky_infer::secret_cand::path_scope`]); no path gives no scope and no extension.
pub fn scope_of(item: &Item) -> Result<(Vec<&'static str>, String)> {
    let Some(file) = item.rec.file.as_deref() else {
        return Ok((Vec::new(), String::new()));
    };
    if item.source != CREDDATA {
        return Ok(sc::path_scope(file));
    }
    Ok((stored_scope(file)?, item.rec.ext.clone()))
}

/// The scope words of a CredData path, from its directories: what `get_file_scope()` wrote
/// from the real path. Each directory must be a word of the list, and they must come in the
/// list's order. The chain is not computed again from the stored path, which can differ: a real
/// path that held `another` is filed under `other`, and the stored `…/other/…` also holds
/// `/other`, a second word of the list.
pub fn stored_scope(file: &str) -> Result<Vec<&'static str>> {
    let parts: Vec<&str> = file.split('/').collect();
    if parts.len() < 4 || parts[0] != "data" {
        bail!("{file}: not a CredData file under a scope directory");
    }
    let dirs = &parts[2..parts.len() - 1];
    if dirs == [sc::NO_SCOPE] {
        return Ok(Vec::new());
    }
    let mut words = Vec::with_capacity(dirs.len());
    let mut list = sc::WORD_IN_PATH.iter().map(|w| w.trim_matches('/'));
    for dir in dirs {
        match list.by_ref().find(|w| w == dir) {
            Some(w) => words.push(w),
            None => bail!("{file}: its directories are not a scope get_file_scope() writes"),
        }
    }
    Ok(words)
}

/// The state of an item and its features.
pub fn state_of(item: &Item, tok: &Tokenizer, words: &WordSet) -> Result<(String, Features)> {
    let (scope, ext) = scope_of(item)?;
    Ok(record_state(&item.rec, &scope, &ext, tok, words))
}

/// The state of a candidate record under a scope and an extension, and its features.
pub fn record_state(
    r: &CandRecord,
    scope: &[&str],
    ext: &str,
    tok: &Tokenizer,
    words: &WordSet,
) -> (String, Features) {
    let filters = Filters {
        entropy: r.filters.entropy,
        placeholder: r.filters.placeholder,
        stopword: r.filters.stopword,
        allowlist: r.filters.allowlist,
    };
    let (value_start, value_end) =
        sc::value_columns(r.line, r.end_line, r.start, r.end, r.line_from);
    let c = Candidate {
        rule: &r.rule,
        scope,
        ext,
        name: r.name.as_deref(),
        value: &r.value,
        line_text: &r.line_text,
        value_start,
        value_end,
        prev_line: r.prev_line.as_deref(),
        next_line: r.next_line.as_deref(),
        filters,
    };
    let f = Features::of(&r.value, tok, words, filters);
    (sc::state_with(&c, &f), f)
}

/// sha256 of a file, in hexadecimal.
pub fn sha256_file(path: &Path) -> Result<String> {
    let mut f = std::fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut ctx = ring::digest::Context::new(&ring::digest::SHA256);
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        ctx.update(&buf[..n]);
    }
    Ok(hex(ctx.finish().as_ref()))
}

/// sha256 of a byte string, in hexadecimal.
pub fn sha256_hex(bytes: &[u8]) -> String {
    hex(ring::digest::digest(&ring::digest::SHA256, bytes).as_ref())
}

pub fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// `{bytes, sha256}` of a file, for the counts.
pub fn file_entry(path: &Path) -> Result<serde_json::Value> {
    Ok(serde_json::json!({
        "bytes": std::fs::metadata(path).with_context(|| format!("stat {}", path.display()))?.len(),
        "sha256": sha256_file(path)?,
    }))
}

/// The dictionary of section 3.2 from a word list (`/usr/share/dict/words`).
pub fn load_words(path: &Path) -> Result<WordSet> {
    let text = std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    let words = WordSet::from_text(&text);
    if words.is_empty() {
        bail!("{}: no lower-case word of 4 to 12 letters", path.display());
    }
    Ok(words)
}

/// The model's tokenizer (`data/tok/tok-s2-1024.json`).
pub fn load_tokenizer(path: &Path) -> Result<Tokenizer> {
    let json = std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    Tokenizer::from_tokenizer_json(&json).map_err(|e| anyhow::anyhow!("{}: {e}", path.display()))
}

/// Writes pretty JSON with a final newline.
pub fn write_json(path: &Path, v: &serde_json::Value) -> Result<()> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    std::fs::write(path, serde_json::to_string_pretty(v)? + "\n")
        .with_context(|| format!("write {}", path.display()))
}

pub fn read_json(path: &Path) -> Result<serde_json::Value> {
    let text = std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    serde_json::from_str(&text).with_context(|| format!("parse {}", path.display()))
}

/// A counter with sorted keys, for the counts written to JSON.
pub type Counts = BTreeMap<String, usize>;

pub fn bump(c: &mut Counts, key: &str) {
    *c.entry(key.to_string()).or_default() += 1;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn records_follow_the_generator_conventions() {
        let text = "first\r\nkey: VALUE # note\r\nlast";
        let at = text.find("VALUE").unwrap();
        let r = record_at(
            "t",
            Some("conf/app.YML"),
            "r",
            Some("key"),
            text,
            at,
            at + 5,
            RecFilters::default(),
        );
        assert_eq!((r.line, r.end_line, r.start, r.end), (2, 2, 5, 10));
        assert_eq!((r.cp_start, r.cp_end, r.offset, r.length), (5, 10, 12, 5));
        assert_eq!(r.line_text, "key: VALUE # note");
        assert_eq!(r.prev_line.as_deref(), Some("first"));
        assert_eq!(r.next_line.as_deref(), Some("last"));
        assert_eq!(r.ext, ".yml");
        assert_eq!(r.line_from, 0);
        // first and last lines have one neighbour
        let r = record_at("t", None, "r", None, text, 0, 5, RecFilters::default());
        assert_eq!(
            (r.prev_line.as_deref(), r.next_line.as_deref()),
            (None, Some("key: VALUE # note"))
        );
        let at = text.find("last").unwrap();
        let r = record_at(
            "t",
            None,
            "r",
            None,
            text,
            at,
            at + 4,
            RecFilters::default(),
        );
        assert_eq!(
            (r.prev_line.as_deref(), r.next_line.as_deref()),
            (Some("key: VALUE # note"), None)
        );
        assert_eq!(r.ext, "");
    }

    #[test]
    fn a_value_over_several_lines() {
        let text = "a\nkey: |\n  AAAA\n  BBBB\nz\n";
        let (s, e) = (text.find("AAAA").unwrap(), text.find("BBBB").unwrap() + 4);
        let r = record_at("t", None, "r", None, text, s, e, RecFilters::default());
        assert_eq!((r.line, r.end_line), (3, 4));
        assert_eq!((r.start, r.end), (2, 6));
        assert_eq!(r.line_text, "  AAAA");
        assert_eq!(r.prev_line.as_deref(), Some("key: |"));
        assert_eq!(r.next_line.as_deref(), Some("z"));
        assert_eq!(r.value, "AAAA\n  BBBB");
    }

    #[test]
    fn a_long_line_is_cut_around_the_value() {
        let text = format!("{}VALUE{}", "a".repeat(3000), "b".repeat(3000));
        let r = record_at(
            "t",
            None,
            "r",
            None,
            &text,
            3000,
            3005,
            RecFilters::default(),
        );
        assert_eq!(r.line_text.len(), MAX_LINE);
        // the generator's window: the value in its middle
        assert_eq!(r.line_from, 3000 - (MAX_LINE - 5) / 2);
        assert_eq!(
            &r.line_text[r.start - r.line_from..r.end - r.line_from],
            "VALUE"
        );
        // UTF-16 columns: a character outside the BMP takes two
        let text = "\u{1F511}\u{e9}=VALUE";
        let at = text.find("VALUE").unwrap();
        let r = record_at(
            "t",
            None,
            "r",
            None,
            text,
            at,
            at + 5,
            RecFilters::default(),
        );
        assert_eq!((r.start, r.cp_start, r.offset), (4, 3, 4));
    }

    #[test]
    fn creddata_scopes_are_read_from_the_path() {
        assert_eq!(
            stored_scope("data/0a1b2c3d/test/src/conf/0a1b2c3d.yml").unwrap(),
            ["test", "src", "conf"]
        );
        assert!(stored_scope("data/0a1b2c3d/_/0a1b2c3d.md")
            .unwrap()
            .is_empty());
        assert_eq!(
            stored_scope("data/0a1b2c3d/doc/local/0a1b2c3d").unwrap(),
            ["doc", "local"]
        );
        // a licence copy at the repository's root has no scope directory
        assert!(stored_scope("data/0a1b2c3d/LICENSE").is_err());
        // `other` is in the list twice (`other`, `/other`): once or twice in a chain
        assert_eq!(
            stored_scope("data/0a1b2c3d/test/conf/other/0a1b2c3d.config").unwrap(),
            ["test", "conf", "other"]
        );
        assert_eq!(
            stored_scope("data/0a1b2c3d/other/usr/other/0a1b2c3d")
                .unwrap()
                .len(),
            3
        );
        assert!(stored_scope("data/0a1b2c3d/other/other/other/0a1b2c3d").is_err());
        // directories in another order are not a scope the function writes
        assert!(stored_scope("data/0a1b2c3d/src/test/0a1b2c3d.py").is_err());
        assert!(stored_scope("data/0a1b2c3d/nothing/0a1b2c3d.py").is_err());
    }
}
