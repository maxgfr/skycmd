//! CredData candidates as training and validation items (`docs/s6-secrets2-prereg.md`, section
//! 3.1 and amendments 1 to 3).
//!
//! Inputs, all under the data folder: the frozen generator outputs of `gen/` (the candidates of
//! the training and validation repositories and their labels, checked against the hashes of
//! `gen/generator.json`), `splits.json`, and CredData itself (`creddata/meta`, `creddata/data`).
//!
//! - **Labels.** The frozen labels file names the label of every candidate: `T`, `X`, `F`
//!   from the overlapped row, `U` on a line no row covers, `D` on a labelled line with no
//!   labelled value under it. This module labels every candidate again from the meta rows, with
//!   its own reading of the matching rules, and counts where the two differ. The frozen file is
//!   the one the items are built from; a difference is reported, never settled here.
//! - **Weak negatives.** The `U` candidates of the training repositories are the pool. After
//!   the decontamination, at most 25% of the labelled training candidates that were kept are
//!   drawn from it (seed), labelled `F` and tagged `weak`.
//! - **Decontamination.** A candidate is removed when its normalised line also occurs in a test
//!   repository. A line is normalised by trimming it and collapsing its inner whitespace; the
//!   value of every T row on it (and, for a candidate labelled T, the candidate's own value) is
//!   replaced by its length, which on the test side, where no label is read, means any text of
//!   that length at that place. Of the test repositories this reads the lines of the files and
//!   nothing else: never `meta/`.
//!
//! No test repository's meta file is opened by this module ([`Splits::guard`]).

use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use serde::Deserialize;
use serde_json::json;

use super::{
    bump, file_entry, read_json, sha256_file, write_json, CandRecord, Counts, Item, Label, CREDDATA,
};
use crate::csv::parse_csv;
use crate::record::{read_jsonl, write_jsonl};
use crate::rng::Rng;

/// The generator the build accepts.
pub const GENERATOR: &str = "sg-cand-1";
pub const REVISION: u32 = 1;

/// Weak negatives per labelled training candidate, at most (section 3.1, amendment 1).
pub const WEAK_SHARE: f64 = 0.25;

/// The repositories of each split (`splits.json`).
#[derive(Clone, Debug)]
pub struct Splits {
    pub training: Vec<String>,
    pub validation: Vec<String>,
    pub test: Vec<String>,
}

impl Splits {
    pub fn read(data: &Path) -> Result<Self> {
        let v = read_json(&data.join("splits.json"))?;
        let ids = |k: &str| -> Result<Vec<String>> {
            let mut out: Vec<String> = v[k]
                .as_array()
                .with_context(|| format!("splits.json: no {k}"))?
                .iter()
                .map(|r| {
                    r["repo_id"]
                        .as_str()
                        .map(str::to_string)
                        .context("splits.json: an entry without repo_id")
                })
                .collect::<Result<_>>()?;
            out.sort();
            Ok(out)
        };
        let s = Self {
            training: ids("training")?,
            validation: ids("validation")?,
            test: ids("test")?,
        };
        let mut all: HashSet<&str> = HashSet::new();
        for id in s.training.iter().chain(&s.validation).chain(&s.test) {
            if id.len() != 8 || !id.bytes().all(|b| b.is_ascii_hexdigit()) {
                bail!("splits.json: {id} is not a repository id");
            }
            if !all.insert(id) {
                bail!("splits.json: {id} is in two splits");
            }
        }
        Ok(s)
    }

    /// The repositories of `training` or `validation`.
    pub fn of(&self, split: &str) -> Result<&[String]> {
        match split {
            "training" => Ok(&self.training),
            "validation" => Ok(&self.validation),
            _ => bail!("refused: split {split:?} (only training and validation have labels)"),
        }
    }

    /// Fails unless `repo` is a training or validation repository: called before any meta row
    /// of a repository is read.
    pub fn guard(&self, repo: &str) -> Result<()> {
        if self
            .training
            .iter()
            .chain(&self.validation)
            .any(|r| r == repo)
        {
            return Ok(());
        }
        if self.test.iter().any(|r| r == repo) {
            bail!("refused: {repo} is a test repository");
        }
        bail!("refused: {repo} is in no split")
    }
}

/// The repository of a CredData path (`data/<repo>/…`).
pub fn repo_of(file: &str) -> Result<&str> {
    let rest = file
        .strip_prefix("data/")
        .with_context(|| format!("{file}: not a CredData path"))?;
    let repo = rest.split('/').next().unwrap_or("");
    if repo.len() != 8 || !repo.bytes().all(|b| b.is_ascii_hexdigit()) {
        bail!("{file}: not a CredData path");
    }
    Ok(repo)
}

/// A file's text as the generator reads it, in UTF-16 code units (its offsets are JavaScript
/// string indices); `None` for a file it skips: empty, or with a NUL byte in its first 8,192.
pub fn read_utf16(path: &Path) -> Result<Option<Vec<u16>>> {
    let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
    if bytes.is_empty() || bytes[..bytes.len().min(8192)].contains(&0) {
        return Ok(None);
    }
    Ok(Some(
        String::from_utf8_lossy(&bytes).encode_utf16().collect(),
    ))
}

/// CredData's lines (`obfuscate_creds.py`: `\r\n` and `\r` end a line as `\n` does): where each
/// starts and where its content ends.
pub fn creddata_lines(u: &[u16]) -> (Vec<usize>, Vec<usize>) {
    let (mut starts, mut ends) = (vec![0], Vec::new());
    let mut i = 0;
    while i < u.len() {
        if u[i] == 10 {
            ends.push(i);
            starts.push(i + 1);
        } else if u[i] == 13 {
            ends.push(i);
            if u.get(i + 1) == Some(&10) {
                i += 1;
            }
            starts.push(i + 1);
        }
        i += 1;
    }
    ends.push(u.len());
    (starts, ends)
}

/// The offset of the `n`-th code point after `from`, not past `to` (a meta row counts Python
/// characters).
fn offset_of_code_point(u: &[u16], from: usize, to: usize, n: usize) -> usize {
    let mut i = from;
    let mut k = 0;
    while k < n && i < to {
        let high = (0xd800..=0xdbff).contains(&u[i]);
        i += if high && i + 1 < to { 2 } else { 1 };
        k += 1;
    }
    i
}

/// The line of the generator (`\n` ends a line, a `\r` before it is left out) that holds
/// `offset`.
fn generator_line(u: &[u16], offset: usize) -> (usize, usize) {
    let offset = offset.min(u.len());
    let from = u[..offset]
        .iter()
        .rposition(|&c| c == 10)
        .map_or(0, |i| i + 1);
    let to = u[offset..]
        .iter()
        .position(|&c| c == 10)
        .map_or(u.len(), |i| offset + i);
    let to = if to > from && u[to - 1] == 13 {
        to - 1
    } else {
        to
    };
    (from, to.max(from))
}

/// One meta row.
#[derive(Clone, Debug, PartialEq)]
pub struct MetaRow {
    pub id: String,
    pub file: String,
    pub line_start: i64,
    pub line_end: i64,
    pub truth: Label,
    pub value_start: Option<i64>,
    pub value_end: Option<i64>,
    pub category: String,
}

const META_HEADER: &str = "Id,FileID,Domain,RepoName,FilePath,LineStart,LineEnd,GroundTruth,\
                           ValueStart,ValueEnd,CryptographyKey,PredefinedPattern,Category";

/// The leading integer of a field, as JavaScript's `parseInt` reads it; `None` when there is
/// none (an empty field).
fn parse_int(s: &str) -> Option<i64> {
    let s = s.trim_start();
    let (neg, digits) = match s.strip_prefix('-') {
        Some(r) => (true, r),
        None => (false, s.strip_prefix('+').unwrap_or(s)),
    };
    let n = digits.bytes().take_while(u8::is_ascii_digit).count();
    let v: i64 = digits[..n].parse().ok()?;
    Some(if neg { -v } else { v })
}

/// The meta rows of a CSV text of repository `repo`.
pub fn parse_meta(text: &str, repo: &str) -> Result<Vec<MetaRow>> {
    let mut rows = parse_csv(text).into_iter();
    let header = rows.next().unwrap_or_default();
    if header.join(",") != META_HEADER {
        bail!("meta/{repo}.csv: unexpected header");
    }
    rows.filter(|r| r.len() > 1 || r.first().is_some_and(|f| !f.is_empty()))
        .map(|r| {
            if r.len() < 13 {
                bail!("meta/{repo}.csv: a row with {} fields", r.len());
            }
            let truth = match r[7].as_str() {
                "T" => Label::T,
                "F" => Label::F,
                "X" => Label::X,
                other => bail!("meta/{repo}.csv: row {} has GroundTruth {other}", r[0]),
            };
            if r[3] != repo || repo_of(&r[4])? != repo {
                bail!("meta/{repo}.csv: row {} names another repository", r[0]);
            }
            Ok(MetaRow {
                id: r[0].clone(),
                file: r[4].clone(),
                line_start: parse_int(&r[5]).unwrap_or(0),
                line_end: parse_int(&r[6]).unwrap_or(0),
                truth,
                value_start: parse_int(&r[8]),
                value_end: parse_int(&r[9]),
                category: r[12].clone(),
            })
        })
        .collect()
}

/// A row placed in its file's text.
#[derive(Clone, Debug, PartialEq)]
pub struct Span {
    pub start: usize,
    pub end: usize,
    pub truth: Label,
    /// A row with no `ValueStart`: its whole lines.
    pub whole: bool,
    pub id: String,
    pub category: String,
}

/// What this module's reading of the rules gives a candidate.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Own {
    /// Labelled by a row: its truth, `value` or `line`, the row's id and category, and how many
    /// rows the candidate overlaps.
    Row {
        label: Label,
        via: &'static str,
        row_id: String,
        category: String,
        rows: usize,
    },
    /// On a line no row covers.
    Unlabelled,
    /// On a labelled line, over no labelled value.
    Dropped,
}

impl Own {
    pub fn code(&self) -> &'static str {
        match self {
            Own::Row { label, .. } => label.as_str(),
            Own::Unlabelled => "U",
            Own::Dropped => "D",
        }
    }
}

/// The rows of one file placed in its text, and the labelled lines.
pub struct FileRows {
    pub spans: Vec<Span>,
    /// CredData line indices (0-based) covered by a row.
    pub labelled: HashSet<usize>,
    starts: Vec<usize>,
    /// Rows outside their file or with an empty span: they label nothing.
    pub bad_rows: usize,
}

impl FileRows {
    /// Section 3.1 and amendment 2: a row is one span from `ValueStart` on `LineStart` to
    /// `ValueEnd` on `LineEnd`; a row with no `ValueStart` is its whole lines; a row with a start
    /// and no end runs to the end of its last line.
    pub fn new(u: &[u16], rows: &[&MetaRow]) -> Self {
        let (starts, ends) = creddata_lines(u);
        let mut out = Self {
            spans: Vec::new(),
            labelled: HashSet::new(),
            starts,
            bad_rows: 0,
        };
        for row in rows {
            let (a, b) = (row.line_start - 1, row.line_end - 1);
            if a < 0 || b < a || b as usize >= out.starts.len() {
                out.bad_rows += 1;
                continue;
            }
            let (a, b) = (a as usize, b as usize);
            let whole = row.value_start.is_none_or(|v| v < 0);
            let start = match row.value_start {
                Some(v) if v >= 0 => offset_of_code_point(u, out.starts[a], ends[a], v as usize),
                _ => out.starts[a],
            };
            let end = match row.value_end {
                Some(v) if v >= 0 && !whole => {
                    offset_of_code_point(u, out.starts[b], ends[b], v as usize)
                }
                _ => ends[b],
            };
            if end <= start {
                out.bad_rows += 1;
                continue;
            }
            out.labelled.extend(a..=b);
            out.spans.push(Span {
                start,
                end,
                truth: row.truth,
                whole,
                id: row.id.clone(),
                category: row.category.clone(),
            });
        }
        out
    }

    fn line_of(&self, offset: usize) -> usize {
        self.starts
            .partition_point(|&s| s <= offset)
            .saturating_sub(1)
    }

    /// The label of a candidate at `[start, end)`: T if a row it overlaps is T, else X, else F
    /// (amendment 2). The row named is, among those of that label, one with a value span
    /// before a whole-line one, then the widest overlap, then the lowest id, as the frozen
    /// labels file names it.
    pub fn label(&self, start: usize, end: usize) -> Own {
        let last = self.line_of(start.max(end.saturating_sub(1)));
        if !(self.line_of(start)..=last).any(|l| self.labelled.contains(&l)) {
            return Own::Unlabelled;
        }
        let over: Vec<&Span> = self
            .spans
            .iter()
            .filter(|s| s.start < end && start < s.end)
            .collect();
        let Some(label) = [Label::T, Label::X, Label::F]
            .into_iter()
            .find(|l| over.iter().any(|s| s.truth == *l))
        else {
            return Own::Dropped;
        };
        let width = |s: &Span| s.end.min(end) as i64 - s.start.max(start) as i64;
        let id = |s: &Span| s.id.parse::<i64>().unwrap_or(i64::MAX);
        let row = over
            .iter()
            .filter(|s| s.truth == label)
            .min_by_key(|s| (s.whole, -width(s), id(s)))
            .expect("a row of the label");
        Own::Row {
            label,
            via: if row.whole { "line" } else { "value" },
            row_id: row.id.clone(),
            category: row.category.clone(),
            rows: over.len(),
        }
    }
}

/// One line of the frozen labels file.
#[derive(Deserialize, Clone, Debug)]
pub struct FrozenLabel {
    pub index: usize,
    pub file: String,
    pub offset: usize,
    pub length: usize,
    pub rule: String,
    pub label: String,
    #[serde(default)]
    pub via: Option<String>,
    #[serde(default, rename = "rowId")]
    pub row_id: Option<String>,
    #[serde(default)]
    pub category: Option<String>,
}

/// Trimmed, inner whitespace collapsed to one space.
pub fn normalise(line: &str) -> String {
    let mut out = String::with_capacity(line.len());
    for w in line.split_whitespace() {
        if !out.is_empty() {
            out.push(' ');
        }
        out.push_str(w);
    }
    out
}

/// Stands for a replaced value while a line is normalised.
const GAP: char = '\u{e000}';

/// A normalised line with values replaced by their lengths: the text between the values, and
/// the length of each value in characters.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Pattern {
    /// `gaps.len() + 1` pieces of text.
    pub fixed: Vec<String>,
    pub gaps: Vec<usize>,
    /// Characters of a line that matches.
    pub chars: usize,
}

impl Pattern {
    /// The pattern of a line whose `gaps` (ranges of UTF-16 units, sorted, disjoint) are
    /// values. `None` when the line itself holds the stand-in character.
    pub fn new(line: &[u16], gaps: &[(usize, usize)]) -> Option<Self> {
        if line.contains(&(GAP as u16)) {
            return None;
        }
        let mut text = String::new();
        let mut lens = Vec::new();
        let mut at = 0;
        for &(a, b) in gaps {
            text.push_str(&String::from_utf16_lossy(&line[at..a]));
            text.push(GAP);
            // a value's own whitespace is collapsed as the test line's would be
            let v = String::from_utf16_lossy(&line[a..b]);
            let inner: usize = v
                .split_whitespace()
                .map(|w| w.chars().count())
                .sum::<usize>()
                + v.split_whitespace().count().saturating_sub(1);
            lens.push(inner.max(1));
            at = b;
        }
        text.push_str(&String::from_utf16_lossy(&line[at..]));
        let fixed: Vec<String> = normalise(&text).split(GAP).map(str::to_string).collect();
        if fixed.len() != lens.len() + 1 {
            return None;
        }
        let chars =
            fixed.iter().map(|f| f.chars().count()).sum::<usize>() + lens.iter().sum::<usize>();
        Some(Self {
            fixed,
            gaps: lens,
            chars,
        })
    }

    /// Whether a normalised line of `chars` characters is this line with other values.
    pub fn matches(&self, line: &str) -> bool {
        let mut pos = 0;
        for (i, f) in self.fixed.iter().enumerate() {
            if !line[pos..].starts_with(f.as_str()) {
                return false;
            }
            pos += f.len();
            if let Some(&n) = self.gaps.get(i) {
                let mut it = line[pos..].char_indices();
                match it.nth(n - 1) {
                    Some((j, c)) => pos += j + c.len_utf8(),
                    None => return false,
                }
            }
        }
        pos == line.len()
    }
}

/// How a candidate's line is looked up in the test repositories.
#[derive(Clone, Debug, PartialEq, Eq)]
enum Key {
    /// The normalised line itself.
    Exact(String),
    /// Index into the patterns.
    Pattern(usize),
    /// Not looked up (a label that is never used).
    None,
}

/// The lines looked up in the test repositories and what was found.
#[derive(Default)]
pub struct Lookup {
    exact: HashMap<String, bool>,
    patterns: Vec<(Pattern, bool)>,
    by_chars: HashMap<usize, Vec<usize>>,
    pub files_read: usize,
    pub files_skipped: usize,
    pub lines: usize,
    /// sha256 over the path and the bytes of every file seen, in the order of the walk.
    digest: Option<ring::digest::Context>,
}

impl Lookup {
    fn add_exact(&mut self, line: String) -> Key {
        self.exact.entry(line.clone()).or_insert(false);
        Key::Exact(line)
    }

    fn add_pattern(&mut self, p: Pattern) -> Key {
        let i = self.patterns.len();
        self.by_chars.entry(p.chars).or_default().push(i);
        self.patterns.push((p, false));
        Key::Pattern(i)
    }

    /// One line of a test file.
    pub fn see(&mut self, line: &str) {
        self.lines += 1;
        let norm = normalise(line);
        if norm.is_empty() {
            return;
        }
        if let Some(hit) = self.exact.get_mut(&norm) {
            *hit = true;
        }
        if self.by_chars.is_empty() {
            return;
        }
        if let Some(list) = self.by_chars.get(&norm.chars().count()) {
            for &i in list {
                let (p, hit) = &mut self.patterns[i];
                if !*hit && p.matches(&norm) {
                    *hit = true;
                }
            }
        }
    }

    /// Every line of every file of a test repository's data directory (`rel` is its path
    /// below the CredData folder).
    pub fn see_dir(&mut self, dir: &Path, rel: &str) -> Result<()> {
        let mut entries: Vec<_> = std::fs::read_dir(dir)
            .with_context(|| format!("list {}", dir.display()))?
            .collect::<std::io::Result<_>>()?;
        entries.sort_by_key(|e| e.file_name());
        for e in entries {
            let t = e.file_type()?;
            if t.is_symlink() {
                continue;
            }
            let name = format!("{rel}/{}", e.file_name().to_string_lossy());
            if t.is_dir() {
                self.see_dir(&e.path(), &name)?;
            } else if t.is_file() {
                let bytes = std::fs::read(e.path())?;
                let d = self
                    .digest
                    .get_or_insert_with(|| ring::digest::Context::new(&ring::digest::SHA256));
                d.update(name.as_bytes());
                d.update(&(bytes.len() as u64).to_le_bytes());
                d.update(&bytes);
                if bytes.is_empty() || bytes[..bytes.len().min(8192)].contains(&0) {
                    self.files_skipped += 1;
                    continue;
                }
                self.files_read += 1;
                for line in String::from_utf8_lossy(&bytes).split('\n') {
                    self.see(line);
                }
            }
        }
        Ok(())
    }

    /// The digest of what was read, in hexadecimal (of nothing when no file was seen).
    pub fn digest(&mut self) -> String {
        let d = self
            .digest
            .take()
            .unwrap_or_else(|| ring::digest::Context::new(&ring::digest::SHA256));
        super::hex(d.finish().as_ref())
    }

    fn hit(&self, key: &Key) -> bool {
        match key {
            Key::Exact(s) => self.exact.get(s).copied().unwrap_or(false),
            Key::Pattern(i) => self.patterns[*i].1,
            Key::None => false,
        }
    }
}

/// The key of a candidate at `[start, end)` of a file's text: its generator line, with the
/// value spans of the T rows on it as gaps, and the candidate's own span when `own_gap`.
fn key_of(
    u: &[u16],
    start: usize,
    end: usize,
    rows: &FileRows,
    own_gap: bool,
    lookup: &mut Lookup,
) -> Key {
    let (from, to) = generator_line(u, start);
    let mut gaps: Vec<(usize, usize)> = rows
        .spans
        .iter()
        .filter(|s| s.truth == Label::T && !s.whole && s.start < to && from < s.end)
        .map(|s| (s.start.max(from), s.end.min(to)))
        .collect();
    if own_gap {
        gaps.push((start.max(from), end.min(to).max(start.max(from))));
    }
    gaps.retain(|(a, b)| b > a);
    let line = &u[from..to];
    if gaps.is_empty() {
        return lookup.add_exact(normalise(&String::from_utf16_lossy(line)));
    }
    gaps.sort_unstable();
    let mut merged: Vec<(usize, usize)> = Vec::new();
    for (a, b) in gaps {
        match merged.last_mut() {
            Some(last) if a <= last.1 => last.1 = last.1.max(b),
            _ => merged.push((a, b)),
        }
    }
    let rel: Vec<(usize, usize)> = merged.iter().map(|(a, b)| (a - from, b - from)).collect();
    match Pattern::new(line, &rel) {
        Some(p) => lookup.add_pattern(p),
        None => lookup.add_exact(normalise(&String::from_utf16_lossy(line))),
    }
}

pub struct CredOptions {
    /// The data folder (`creddata/`, `gen/`, `splits.json`).
    pub data: PathBuf,
    /// The build folder.
    pub out: PathBuf,
    pub seed: u64,
}

/// One candidate of a split while the build runs: how its line is looked up, and whether its
/// file sits directly under the repository's directory (a licence copy, with no scope).
struct Cand {
    key: Key,
    root: bool,
}

/// The candidates of a split: the records, their frozen labels, and what the build adds.
struct SplitCands {
    split: &'static str,
    short: &'static str,
    recs: Vec<CandRecord>,
    labels: Vec<FrozenLabel>,
    cands: Vec<Cand>,
}

/// The files `gen/generator.json` names, checked against their hashes.
pub fn check_frozen(data: &Path) -> Result<serde_json::Value> {
    let gen = data.join("gen");
    let g = read_json(&gen.join("generator.json")).context(
        "gen/generator.json does not exist: the generator is not frozen, no dataset is built",
    )?;
    if g["generator"] != GENERATOR || g["revision"] != REVISION {
        bail!("gen/generator.json: not {GENERATOR} revision {REVISION}");
    }
    let mut files = serde_json::Map::new();
    for split in ["training", "validation"] {
        for name in [
            format!("candidates-{split}.raw.jsonl"),
            format!("labels-{split}.jsonl"),
        ] {
            let want = g["files"][&name]["sha256"]
                .as_str()
                .with_context(|| format!("gen/generator.json: no hash of {name}"))?;
            let got = sha256_file(&gen.join(&name))?;
            if got != want {
                bail!("gen/{name}: sha256 {got}, generator.json says {want}");
            }
            let entry = g["files"][&name].clone();
            files.insert(name, entry);
        }
    }
    Ok(json!({
        "generator": g["generator"],
        "revision": g["revision"],
        "frozen": g["frozen"],
        "secretgate_commit": g["secretgate"]["commit"],
        "generator_json": file_entry(&gen.join("generator.json"))?,
        "files": files,
    }))
}

/// Builds `<out>/candidates/creddata-{train,val}.jsonl` and `<out>/counts/creddata.json`.
pub fn build(o: &CredOptions) -> Result<serde_json::Value> {
    let frozen = check_frozen(&o.data)?;
    let splits = Splits::read(&o.data)?;
    let cred = o.data.join("creddata");
    let gen = o.data.join("gen");
    let mut lookup = Lookup::default();
    let mut counts = serde_json::Map::new();
    let mut all: Vec<SplitCands> = Vec::new();
    let mut disagreements = BTreeMap::new();

    for (split, short) in [("training", "train"), ("validation", "val")] {
        let repos: HashSet<&str> = splits.of(split)?.iter().map(String::as_str).collect();
        let recs: Vec<CandRecord> = read_jsonl(&gen.join(format!("candidates-{split}.raw.jsonl")))?;
        let labels: Vec<FrozenLabel> = read_jsonl(&gen.join(format!("labels-{split}.jsonl")))?;
        if recs.len() != labels.len() {
            bail!(
                "{split}: {} candidates and {} labels",
                recs.len(),
                labels.len()
            );
        }
        let mut c = Counts::new();
        let mut own_counts = Counts::new();
        let mut frozen_counts = Counts::new();
        let mut differ = Counts::new();
        let mut uuid_only = Counts::new();
        let mut via_line = Counts::new();
        let (mut via_differ, mut row_differ, mut bad_rows, mut files_seen) = (0, 0, 0, 0);
        let mut meta: HashMap<String, Vec<MetaRow>> = HashMap::new();
        let mut cands = Vec::with_capacity(recs.len());

        let mut i = 0;
        while i < recs.len() {
            let file = recs[i].file.clone().context("a candidate without a file")?;
            let repo = repo_of(&file)?.to_string();
            if !repos.contains(repo.as_str()) {
                bail!("{file}: not a {split} repository");
            }
            splits.guard(&repo)?;
            if !meta.contains_key(&repo) {
                let text = std::fs::read_to_string(cred.join(format!("meta/{repo}.csv")))
                    .with_context(|| format!("read meta/{repo}.csv"))?;
                meta.insert(repo.clone(), parse_meta(&text, &repo)?);
            }
            let rows: Vec<&MetaRow> = meta[&repo].iter().filter(|r| r.file == file).collect();
            let u = read_utf16(&cred.join(&file))?
                .with_context(|| format!("{file}: the generator read it, it is empty or binary"))?;
            let placed = FileRows::new(&u, &rows);
            bad_rows += placed.bad_rows;
            files_seen += 1;
            let root = file.split('/').count() == 3;
            while i < recs.len() && recs[i].file.as_deref() == Some(file.as_str()) {
                let (rec, fl) = (&recs[i], &labels[i]);
                if rec.generator != GENERATOR || rec.revision != Some(REVISION) || rec.masked {
                    bail!("{split}: candidate {i} is not a raw {GENERATOR} revision {REVISION} record");
                }
                if fl.index != i
                    || fl.file != file
                    || fl.offset != rec.offset
                    || fl.length != rec.length
                    || fl.rule != rec.rule
                {
                    bail!("{split}: label {i} is not the label of candidate {i}");
                }
                let (start, end) = (rec.offset, rec.offset + rec.length);
                let own = placed.label(start, end);
                bump(&mut own_counts, own.code());
                bump(&mut frozen_counts, &fl.label);
                if own.code() != fl.label {
                    bump(
                        &mut differ,
                        &format!("frozen {} own {}", fl.label, own.code()),
                    );
                } else if let Own::Row { via, row_id, .. } = &own {
                    via_differ += (fl.via.as_deref() != Some(via)) as usize;
                    row_differ += (fl.row_id.as_deref() != Some(row_id.as_str())) as usize;
                }
                let label = match fl.label.as_str() {
                    "T" => Some(Label::T),
                    "X" => Some(Label::X),
                    "F" => Some(Label::F),
                    "U" | "D" => None,
                    other => bail!("{split}: label {other:?} of candidate {i}"),
                };
                if label.is_some() {
                    if fl.category.as_deref() == Some("UUID") {
                        bump(&mut uuid_only, &fl.label);
                    }
                    if fl.via.as_deref() == Some("line") {
                        bump(&mut via_line, &fl.label);
                    }
                }
                // looked up in the test repositories: every labelled candidate, and the pool
                // of the weak negatives
                let pool = fl.label == "U" && split == "training" && !root;
                let key = if label.is_some() || pool {
                    key_of(
                        &u,
                        start,
                        end,
                        &placed,
                        label == Some(Label::T),
                        &mut lookup,
                    )
                } else {
                    Key::None
                };
                cands.push(Cand { key, root });
                i += 1;
            }
        }
        c.insert("candidates".into(), recs.len());
        c.insert("files_with_candidates".into(), files_seen);
        c.insert("meta_rows_placing_nothing".into(), bad_rows);
        c.insert("label_differences".into(), differ.values().sum());
        c.insert("via_differences".into(), via_differ);
        c.insert("row_differences".into(), row_differ);
        counts.insert(
            split.to_string(),
            json!({
                "counts": c,
                "frozen_labels": frozen_counts,
                "own_labels": own_counts,
                "labels_from_a_whole_line_row": via_line,
                "labels_from_a_row_of_category_exactly_UUID": uuid_only,
            }),
        );
        disagreements.insert(split.to_string(), differ);
        all.push(SplitCands {
            split,
            short,
            recs,
            labels,
            cands,
        });
    }

    // the test repositories: lines only
    for repo in &splits.test {
        let dir = cred.join("data").join(repo);
        if dir.is_dir() {
            lookup.see_dir(&dir, &format!("data/{repo}"))?;
        }
    }

    let mut kept_counts = serde_json::Map::new();
    let mut weak_counts = json!({});
    for SplitCands {
        split,
        short,
        recs,
        labels,
        cands,
    } in &all
    {
        let mut removed: BTreeMap<String, Counts> = BTreeMap::new();
        let mut kept = Counts::new();
        let mut by_rule: BTreeMap<String, Counts> = BTreeMap::new();
        let mut items: Vec<Item> = Vec::new();
        let mut pool: Vec<usize> = Vec::new();
        let (mut pool_all, mut pool_root, mut pool_removed) = (0usize, 0usize, 0usize);
        for (index, cand) in cands.iter().enumerate() {
            let (rec, frozen) = (&recs[index], &labels[index]);
            let label = match frozen.label.as_str() {
                "T" => Label::T,
                "X" => Label::X,
                "F" => Label::F,
                "U" => {
                    if *split == "training" {
                        pool_all += 1;
                        if cand.root {
                            pool_root += 1;
                        } else if lookup.hit(&cand.key) {
                            pool_removed += 1;
                        } else {
                            pool.push(index);
                        }
                    }
                    continue;
                }
                _ => continue,
            };
            if lookup.hit(&cand.key) {
                let how = match &cand.key {
                    Key::Pattern(p) => {
                        // how much text stands around the replaced values: a line that is
                        // little more than its value matches any line of that length
                        let fixed: usize = lookup.patterns[*p]
                            .0
                            .fixed
                            .iter()
                            .map(|f| f.chars().count())
                            .sum();
                        let bucket = match fixed {
                            0 => "0",
                            1..=4 => "1 to 4",
                            5..=9 => "5 to 9",
                            10..=19 => "10 to 19",
                            _ => "20 and more",
                        };
                        bump(
                            removed
                                .entry(format!("{}_by_characters_around_the_value", label.as_str()))
                                .or_default(),
                            bucket,
                        );
                        "value_replaced_by_its_length"
                    }
                    _ => "exact",
                };
                bump(removed.entry(label.as_str().into()).or_default(), how);
                continue;
            }
            bump(&mut kept, label.as_str());
            bump(by_rule.entry(rec.rule.clone()).or_default(), label.as_str());
            items.push(Item {
                id: format!("cred:{short}:{index}"),
                source: CREDDATA.to_string(),
                split: short.to_string(),
                label,
                weak: false,
                kind: true,
                class: frozen.category.clone(),
                via: frozen.via.clone(),
                rec: rec.clone(),
            });
        }
        if *split == "training" {
            let labelled = items.len();
            let cap = (labelled as f64 * WEAK_SHARE).floor() as usize;
            let mut order: Vec<usize> = (0..pool.len()).collect();
            Rng::derived(o.seed, "s6-secrets2:weak-negatives").shuffle(&mut order);
            order.truncate(cap);
            order.sort_unstable();
            let mut weak_by_rule = Counts::new();
            for &k in &order {
                let index = pool[k];
                bump(&mut weak_by_rule, &recs[index].rule);
                items.push(Item {
                    id: format!("cred:{short}:{index}"),
                    source: CREDDATA.to_string(),
                    split: short.to_string(),
                    label: Label::F,
                    weak: true,
                    kind: true,
                    class: None,
                    via: None,
                    rec: recs[index].clone(),
                });
            }
            kept.insert("weak".into(), order.len());
            weak_counts = json!({
                "share": WEAK_SHARE,
                "pool": pool_all,
                "pool_in_a_file_with_no_scope_directory": pool_root,
                "pool_removed_by_decontamination": pool_removed,
                "pool_left": pool.len(),
                "labelled_training_candidates_kept": labelled,
                "cap": cap,
                "drawn": order.len(),
                "by_rule": weak_by_rule,
            });
        }
        kept.insert("items".into(), items.len());
        kept_counts.insert(
            split.to_string(),
            json!({"kept": kept, "kept_by_rule": by_rule, "removed_by_decontamination": removed}),
        );
        write_jsonl(
            &o.out.join(format!("candidates/creddata-{short}.jsonl")),
            items.iter(),
        )?;
    }

    let patterns_hit = lookup.patterns.iter().filter(|p| p.1).count();
    let test_digest = lookup.digest();
    let out = json!({
        "generator": frozen,
        "splits_json": file_entry(&o.data.join("splits.json"))?,
        "repositories": {
            "training": splits.training.len(),
            "validation": splits.validation.len(),
            "test": splits.test.len(),
        },
        "labels": counts,
        "label_differences": disagreements,
        "decontamination": {
            "test_repositories": splits.test.len(),
            "test_files_read": lookup.files_read,
            "test_files_skipped_empty_or_binary": lookup.files_skipped,
            "test_lines": lookup.lines,
            "test_files_sha256": test_digest,
            "lines_looked_up": lookup.exact.len(),
            "lines_found": lookup.exact.values().filter(|h| **h).count(),
            "lines_with_a_value_replaced_by_its_length": lookup.patterns.len(),
            "of_which_found": patterns_hit,
        },
        "weak_negatives": weak_counts,
        "items": kept_counts,
    });
    write_json(&o.out.join("counts/creddata.json"), &out)?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn u(s: &str) -> Vec<u16> {
        s.encode_utf16().collect()
    }

    fn row(id: &str, ls: i64, le: i64, truth: Label, vs: Option<i64>, ve: Option<i64>) -> MetaRow {
        MetaRow {
            id: id.into(),
            file: "data/0a1b2c3d/src/0a1b2c3d.py".into(),
            line_start: ls,
            line_end: le,
            truth,
            value_start: vs,
            value_end: ve,
            category: "Password".into(),
        }
    }

    fn span_of(text: &str, needle: &str) -> (usize, usize) {
        let at = text.find(needle).unwrap();
        let a = text[..at].encode_utf16().count();
        (a, a + needle.encode_utf16().count())
    }

    #[test]
    fn meta_rows_are_parsed() {
        let text = format!(
            "{META_HEADER}\n7,ab,GitHub,0a1b2c3d,data/0a1b2c3d/src/ab.py,3,3,T,10,22,,,Password\n\
             8,ab,GitHub,0a1b2c3d,data/0a1b2c3d/src/ab.py,5,6,F,,,,,\"Key:UUID\"\n\
             9,ab,GitHub,0a1b2c3d,data/0a1b2c3d/src/ab.py,9,9,X,-1,-1,,,Token\n"
        );
        let rows = parse_meta(&text, "0a1b2c3d").unwrap();
        assert_eq!(rows.len(), 3);
        assert_eq!(
            (rows[0].value_start, rows[0].value_end),
            (Some(10), Some(22))
        );
        assert_eq!((rows[1].value_start, rows[1].line_end), (None, 6));
        assert_eq!(rows[1].category, "Key:UUID");
        assert_eq!((rows[2].truth, rows[2].value_start), (Label::X, Some(-1)));
        assert!(parse_meta(&text, "ffffffff").is_err());
        assert!(parse_meta("Id,File\n1,2\n", "0a1b2c3d").is_err());
        assert_eq!(parse_int("12"), Some(12));
        assert_eq!(parse_int("-1"), Some(-1));
        assert_eq!(parse_int("7.0"), Some(7));
        assert_eq!(parse_int(""), None);
    }

    #[test]
    fn candidates_are_labelled_by_the_rows_they_overlap() {
        let text =
            "a = 1\npassword = \"S3cr3tValue\" # note\ntoken: abcdef\r\nplain line\nlast = zz\n";
        let t = u(text);
        let (vs, ve) = span_of(text, "S3cr3tValue");
        let line2 = text.find("password").unwrap();
        let rows = [
            row(
                "11",
                2,
                2,
                Label::T,
                Some((vs - line2) as i64),
                Some((ve - line2) as i64),
            ),
            row("12", 3, 3, Label::F, None, None),
            row("13", 5, 5, Label::X, Some(7), None),
            // outside the file, and an empty span: they label nothing
            row("14", 99, 99, Label::T, Some(0), Some(4)),
            row("15", 4, 4, Label::T, Some(3), Some(3)),
        ];
        let refs: Vec<&MetaRow> = rows.iter().collect();
        let f = FileRows::new(&t, &refs);
        assert_eq!(f.bad_rows, 2);
        assert_eq!(f.spans.len(), 3);
        // over the T value, wholly or in part
        let own = f.label(vs, ve);
        assert_eq!(own.code(), "T");
        assert!(matches!(
            &own,
            Own::Row {
                via: "value",
                rows: 1,
                ..
            }
        ));
        assert_eq!(f.label(vs + 3, ve + 9).code(), "T");
        // on the labelled line, over no value
        let (ns, ne) = span_of(text, "note");
        assert_eq!(f.label(ns, ne), Own::Dropped);
        // a whole-line row labels every candidate of its line
        let (ts, te) = span_of(text, "abcdef");
        let own = f.label(ts, te);
        assert!(matches!(
            &own,
            Own::Row {
                label: Label::F,
                via: "line",
                ..
            }
        ));
        // a line no row covers
        let (ps, pe) = span_of(text, "plain");
        assert_eq!(f.label(ps, pe), Own::Unlabelled);
        // a row with a start and no end runs to the end of its line
        let (zs, ze) = span_of(text, "zz");
        assert_eq!(f.label(zs, ze).code(), "X");
        let (ls, le) = span_of(text, "last");
        assert_eq!(f.label(ls, le), Own::Dropped);
    }

    #[test]
    fn several_rows_give_t_then_x_then_f() {
        let text = "k = \"AAAA-BBBB-CCCC\"\n";
        let t = u(text);
        let rows = [
            row("3", 1, 1, Label::F, Some(5), Some(9)),
            row("2", 1, 1, Label::X, Some(10), Some(14)),
            row("1", 1, 1, Label::T, Some(15), Some(19)),
            row("4", 1, 1, Label::F, None, None),
        ];
        let refs: Vec<&MetaRow> = rows.iter().collect();
        let f = FileRows::new(&t, &refs);
        assert_eq!(f.label(5, 19).code(), "T");
        assert_eq!(f.label(5, 14).code(), "X");
        // a value row before the whole-line row of the same label
        let own = f.label(5, 9);
        assert!(matches!(
            &own,
            Own::Row {
                label: Label::F,
                via: "value",
                rows: 2,
                ..
            }
        ));
        assert!(matches!(
            &f.label(0, 1),
            Own::Row {
                label: Label::F,
                via: "line",
                ..
            }
        ));
        // a whole-line T row covers any candidate of its line
        let rows = [
            row("1", 1, 1, Label::T, None, None),
            row("2", 1, 1, Label::F, Some(5), Some(9)),
        ];
        let refs: Vec<&MetaRow> = rows.iter().collect();
        let f = FileRows::new(&t, &refs);
        assert!(matches!(
            &f.label(5, 9),
            Own::Row {
                label: Label::T,
                via: "line",
                ..
            }
        ));
    }

    #[test]
    fn a_row_over_several_lines_is_one_span_and_columns_count_code_points() {
        let text = "x\nkey = '''AAA\nBBB\nCCC''' done\ny = 1\n";
        let t = u(text);
        let rows = [row("1", 2, 4, Label::T, Some(9), Some(3))];
        let refs: Vec<&MetaRow> = rows.iter().collect();
        let f = FileRows::new(&t, &refs);
        let (a, _) = span_of(text, "AAA");
        let (_, b) = span_of(text, "CCC");
        assert_eq!((f.spans[0].start, f.spans[0].end), (a, b));
        let (bs, be) = span_of(text, "BBB");
        assert_eq!(f.label(bs, be).code(), "T");
        let (ds, de) = span_of(text, "done");
        assert_eq!(f.label(ds, de), Own::Dropped);
        // a column after a character outside the BMP: one Python character, two units
        let text = "\u{1F511} = 'VALUE'\n";
        let t = u(text);
        let rows = [row("1", 1, 1, Label::T, Some(5), Some(10))];
        let refs: Vec<&MetaRow> = rows.iter().collect();
        let f = FileRows::new(&t, &refs);
        assert_eq!((f.spans[0].start, f.spans[0].end), span_of(text, "VALUE"));
        // CredData's lines: a lone \r ends a line, the generator's lines do not
        let text = "a\rb = 'VALUE'\nc\n";
        let t = u(text);
        let (starts, ends) = creddata_lines(&t);
        assert_eq!((starts.len(), ends[0]), (4, 1));
        let rows = [row("1", 2, 2, Label::T, Some(5), Some(10))];
        let refs: Vec<&MetaRow> = rows.iter().collect();
        let f = FileRows::new(&t, &refs);
        assert_eq!((f.spans[0].start, f.spans[0].end), span_of(text, "VALUE"));
        assert_eq!(generator_line(&t, 5), (0, text.find('\n').unwrap()));
    }

    #[test]
    fn lines_are_normalised_and_values_replaced_by_their_length() {
        assert_eq!(normalise("  a \t b\r"), "a b");
        assert_eq!(normalise(" \t "), "");
        let line = u("  password:   \"Xy12345678\"  # prod ");
        let (a, b) = span_of("  password:   \"Xy12345678\"  # prod ", "Xy12345678");
        let p = Pattern::new(&line, &[(a, b)]).unwrap();
        assert_eq!(p.fixed, ["password: \"", "\" # prod"]);
        assert_eq!((p.gaps.clone(), p.chars), (vec![10], 29));
        assert!(p.matches("password: \"Qw98765432\" # prod"));
        assert!(p.matches(
            "password: \"\u{e9}\u{e9}\u{e9}\u{e9}\u{e9}\u{e9}\u{e9}\u{e9}\u{e9}\u{e9}\" # prod"
        ));
        assert!(!p.matches("password: \"Qw9876543\" # prod"));
        assert!(!p.matches("passwort: \"Qw98765432\" # prod"));
        assert!(!p.matches("password: \"Qw98765432\" # test"));
        // two values on a line
        let text = "a=AAAA;b=BBBBBB;";
        let p = Pattern::new(&u(text), &[(2, 6), (9, 15)]).unwrap();
        assert_eq!((p.fixed.len(), p.gaps.clone()), (3, vec![4, 6]));
        assert!(p.matches("a=1234;b=123456;") && !p.matches("a=1234;c=123456;"));
        assert!(!p.matches("a=12345;b=12345;"));
        // a value alone on its line matches any line of its length
        let p = Pattern::new(&u("ABCDEFGH"), &[(0, 8)]).unwrap();
        assert!(p.matches("12345678") && !p.matches("1234567"));
    }

    #[test]
    fn lookup_marks_what_the_test_lines_hold() {
        let text = "user = admin\npassword = \"Zz11223344\"\nurl = http://x\n";
        let t = u(text);
        let (vs, ve) = span_of(text, "Zz11223344");
        let rows = [row("1", 2, 2, Label::T, Some(12), Some(22))];
        let refs: Vec<&MetaRow> = rows.iter().collect();
        let placed = FileRows::new(&t, &refs);
        let mut lookup = Lookup::default();
        let (us, ue) = span_of(text, "admin");
        let k_user = key_of(&t, us, ue, &placed, false, &mut lookup);
        let k_pass = key_of(&t, vs, ve, &placed, true, &mut lookup);
        let (hs, he) = span_of(text, "http://x");
        let k_url = key_of(&t, hs, he, &placed, false, &mut lookup);
        assert!(matches!(k_user, Key::Exact(_)) && matches!(k_pass, Key::Pattern(_)));
        // an F candidate on the line of a T row: the row's value is a gap for it too
        let (ps, pe) = span_of(text, "password");
        let k_name = key_of(&t, ps, pe, &placed, false, &mut lookup);
        assert!(matches!(k_name, Key::Pattern(_)));
        for line in ["  user   = admin  ", "password = \"Aa99887766\"\r", "other"] {
            lookup.see(line);
        }
        assert!(lookup.hit(&k_user) && lookup.hit(&k_pass) && lookup.hit(&k_name));
        assert!(!lookup.hit(&k_url));
        assert_eq!(lookup.lines, 3);
    }

    #[test]
    fn repositories_are_guarded() {
        let s = Splits {
            training: vec!["aaaaaaaa".into()],
            validation: vec!["bbbbbbbb".into()],
            test: vec!["cccccccc".into()],
        };
        assert!(s.guard("aaaaaaaa").is_ok() && s.guard("bbbbbbbb").is_ok());
        assert!(s
            .guard("cccccccc")
            .unwrap_err()
            .to_string()
            .contains("test repository"));
        assert!(s.guard("dddddddd").is_err());
        assert!(s.of("test").is_err());
        assert_eq!(repo_of("data/0a1b2c3d/src/x.py").unwrap(), "0a1b2c3d");
        assert!(repo_of("meta/0a1b2c3d.csv").is_err());
    }
}
